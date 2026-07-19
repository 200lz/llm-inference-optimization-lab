#include "serving/simulation_engine.h"

#include "serving/checked_math.h"
#include "serving/fcfs_scheduler.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace llm_lab::serving {

SimulationEngine::SimulationEngine(const Backend& backend,
                                   SimulationEngineConfig config)
    : backend_(backend),
      scheduler_(std::make_unique<FcfsScheduler>(config.max_active_requests)) {
  if (config.max_active_requests != 1) {
    throw std::invalid_argument(
        "Phase S2 supports exactly one active request");
  }
  backend_.validate_configuration();
}

void SimulationEngine::submit_request(Request request) {
  if (failed_) {
    throw std::logic_error("cannot submit to a failed simulation engine");
  }
  if (request.state != RequestState::Waiting ||
      request.generated_token_count != 0 ||
      request.admitted_time_us.has_value() ||
      request.first_scheduled_time_us.has_value() ||
      request.first_token_time_us.has_value() ||
      request.finish_time_us.has_value()) {
    throw std::invalid_argument("submitted request must be pristine");
  }
  if (request.arrival_time_us < clock_.now_us()) {
    throw std::invalid_argument("request arrival precedes current simulation time");
  }
  const RequestId id = request.request_id;
  const std::int64_t arrival_time_us = request.arrival_time_us;
  const auto inserted = requests_.emplace(id, std::move(request));
  if (!inserted.second) {
    throw std::invalid_argument("duplicate request ID");
  }
  try {
    schedule_request_event(arrival_time_us, id, EventType::RequestArrival);
  } catch (...) {
    requests_.erase(id);
    throw;
  }
}

void SimulationEngine::cancel_request(RequestId request_id) {
  if (failed_) {
    throw std::logic_error("cannot cancel in a failed simulation engine");
  }
  const auto found = requests_.find(request_id);
  if (found == requests_.end()) {
    throw std::out_of_range("unknown request ID");
  }
  Request& request = found->second;
  if (request.state == RequestState::Finished ||
      request.state == RequestState::Cancelled) {
    throw std::logic_error("terminal request cannot be cancelled");
  }
  if (request.state == RequestState::Preempted) {
    throw std::logic_error("preempted requests are not implemented");
  }

  try {
    const auto pending = pending_events_.find(request_id);
    if (request.state == RequestState::Waiting) {
      if (pending != pending_events_.end() &&
          pending->second.type != EventType::RequestArrival) {
        throw std::logic_error("waiting request has unexpected pending event");
      }
    } else if (request.state == RequestState::Prefilling) {
      if (pending == pending_events_.end() ||
          (pending->second.type != EventType::PrefillComplete &&
           pending->second.type != EventType::RequestComplete)) {
        throw std::logic_error("prefilling request has no valid pending event");
      }
    } else if (request.state == RequestState::Decoding) {
      if (pending == pending_events_.end() ||
          (pending->second.type != EventType::DecodeStepComplete &&
           pending->second.type != EventType::RequestComplete)) {
        throw std::logic_error("decoding request has no valid pending event");
      }
    }

    scheduler_->notify_cancelled(request_id, clock_.now_us());
    request.transition_to(RequestState::Cancelled);
    request.finish_time_us = clock_.now_us();
    if (pending != pending_events_.end()) {
      pending->second.authorized_cancelled_ignore = true;
    }
    if (active_request_id_.has_value() &&
        *active_request_id_ == request_id) {
      active_request_id_.reset();
    }
    if (events_.empty() ||
        events_.top().timestamp_us != clock_.now_us()) {
      start_next_if_idle();
    }
  } catch (...) {
    failed_ = true;
    throw;
  }
}

bool SimulationEngine::run_next_timestamp() {
  if (failed_) {
    throw std::logic_error("cannot run a failed simulation engine");
  }
  if (events_.empty()) {
    return false;
  }
  try {
    const std::int64_t timestamp_us = events_.top().timestamp_us;
    clock_.advance_to(timestamp_us);
    std::vector<Event> processed_at_timestamp;
    do {
      const Event event = events_.pop();
      if (process_event(event)) {
        processed_at_timestamp.push_back(event);
      }
    } while (!events_.empty() &&
             events_.top().timestamp_us == timestamp_us);
    start_next_if_idle();
    event_log_.insert(event_log_.end(), processed_at_timestamp.begin(),
                      processed_at_timestamp.end());
    return true;
  } catch (...) {
    failed_ = true;
    throw;
  }
}

void SimulationEngine::run() {
  if (failed_) {
    throw std::logic_error("cannot run a failed simulation engine");
  }
  while (run_next_timestamp()) {
  }
  if (active_request_id_.has_value() || !scheduler_->empty() ||
      !pending_events_.empty()) {
    failed_ = true;
    throw std::logic_error("simulation drained with unfinished work");
  }
}

const Request& SimulationEngine::request(RequestId request_id) const {
  require_results_available();
  const auto found = requests_.find(request_id);
  if (found == requests_.end()) {
    throw std::out_of_range("unknown request ID");
  }
  return found->second;
}

const std::map<RequestId, Request>& SimulationEngine::requests() const {
  require_results_available();
  return requests_;
}

const std::vector<Event>& SimulationEngine::event_log() const {
  require_results_available();
  return event_log_;
}

const SimulationClock& SimulationEngine::clock() const {
  require_results_available();
  return clock_;
}

const SchedulerStatistics& SimulationEngine::scheduler_statistics() const {
  require_results_available();
  return scheduler_->statistics();
}

std::size_t SimulationEngine::scheduler_waiting_count() const {
  require_results_available();
  return scheduler_->waiting_count();
}

std::size_t SimulationEngine::scheduler_running_count() const {
  require_results_available();
  return scheduler_->running_count();
}

void SimulationEngine::require_results_available() const {
  if (failed_) {
    throw std::logic_error("simulation results are unavailable after failure");
  }
}

Request& SimulationEngine::mutable_request(RequestId request_id) {
  const auto found = requests_.find(request_id);
  if (found == requests_.end()) {
    throw std::logic_error("event references unknown request ID");
  }
  return found->second;
}

void SimulationEngine::schedule_after(std::int64_t cost_us,
                                      RequestId request_id, EventType type) {
  if (cost_us < 0) {
    throw std::logic_error("backend returned a negative cost");
  }
  schedule_request_event(checked_add(clock_.now_us(), cost_us), request_id,
                         type);
}

void SimulationEngine::schedule_request_event(std::int64_t timestamp_us,
                                              RequestId request_id,
                                              EventType type) {
  const auto inserted = pending_events_.emplace(
      request_id, PendingEvent{0, type, false});
  if (!inserted.second) {
    throw std::logic_error("request already has a pending event");
  }
  try {
    inserted.first->second.event_sequence =
        events_.schedule(timestamp_us, request_id, type);
  } catch (...) {
    pending_events_.erase(inserted.first);
    throw;
  }
}

void SimulationEngine::start_next_if_idle() {
  if (active_request_id_.has_value()) {
    return;
  }

  const SchedulerDecision decision = scheduler_->choose_next_request();
  if (decision.kind() == SchedulerDecisionKind::NoWork) {
    if (decision.request_id().has_value()) {
      throw std::logic_error("NoWork decision contains a request ID");
    }
    return;
  }
  if (decision.kind() == SchedulerDecisionKind::CapacityFull) {
    if (decision.request_id().has_value()) {
      throw std::logic_error("CapacityFull decision contains a request ID");
    }
    throw std::logic_error("scheduler reports capacity with no active request");
  }
  if (decision.kind() != SchedulerDecisionKind::Admit) {
    throw std::logic_error("scheduler returned an unknown decision kind");
  }
  const std::optional<RequestId> decision_request_id = decision.request_id();
  if (!decision_request_id.has_value()) {
    throw std::logic_error("scheduler admission decision has no request ID");
  }

  const RequestId id = *decision_request_id;
  Request& next = mutable_request(id);
  if (next.state != RequestState::Waiting ||
      next.arrival_time_us > clock_.now_us()) {
    throw std::logic_error("scheduler selected an invalid waiting request");
  }

  const std::int64_t prefill_cost =
      backend_.estimate_prefill_us(next.prompt_token_count, 1);
  schedule_after(prefill_cost, id, EventType::PrefillComplete);

  scheduler_->notify_admitted(decision, clock_.now_us());
  next.transition_to(RequestState::Prefilling);
  next.admitted_time_us = clock_.now_us();
  next.first_scheduled_time_us = clock_.now_us();
  active_request_id_ = id;
}

bool SimulationEngine::process_event(const Event& event) {
  Request& current = mutable_request(event.request_id);
  const auto pending = pending_events_.find(event.request_id);
  if (pending == pending_events_.end() ||
      pending->second.event_sequence != event.event_sequence ||
      pending->second.type != event.type) {
    throw std::logic_error("event does not match the request's pending event");
  }
  if (pending->second.authorized_cancelled_ignore) {
    if (current.state != RequestState::Cancelled) {
      throw std::logic_error("cancelled-event authorization on active request");
    }
    if (ignored_cancelled_event_count_ ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("ignored cancelled event count overflow");
    }
    pending_events_.erase(pending);
    ++ignored_cancelled_event_count_;
    return false;
  }
  pending_events_.erase(pending);

  switch (event.type) {
    case EventType::RequestArrival:
      if (current.state != RequestState::Waiting ||
          current.admitted_time_us.has_value()) {
        throw std::logic_error("duplicate or stale arrival event");
      }
      scheduler_->notify_arrival(event.request_id, current.arrival_time_us);
      return true;

    case EventType::PrefillComplete:
      if (!active_request_id_.has_value() ||
          *active_request_id_ != event.request_id ||
          current.state != RequestState::Prefilling) {
        throw std::logic_error("duplicate or stale prefill completion event");
      }
      if (current.max_new_tokens == 0) {
        schedule_after(0, event.request_id, EventType::RequestComplete);
      } else {
        const std::int64_t decode_cost = backend_.estimate_decode_step_us(1);
        schedule_after(decode_cost, event.request_id,
                       EventType::DecodeStepComplete);
        current.transition_to(RequestState::Decoding);
      }
      return true;

    case EventType::DecodeStepComplete: {
      if (!active_request_id_.has_value() ||
          *active_request_id_ != event.request_id ||
          current.state != RequestState::Decoding ||
          current.generated_token_count >= current.max_new_tokens) {
        throw std::logic_error("duplicate or stale decode completion event");
      }
      const bool completes_request =
          current.generated_token_count + 1 == current.max_new_tokens;
      if (completes_request) {
        schedule_after(0, event.request_id, EventType::RequestComplete);
      } else {
        const std::int64_t decode_cost = backend_.estimate_decode_step_us(1);
        schedule_after(decode_cost, event.request_id,
                       EventType::DecodeStepComplete);
      }
      ++current.generated_token_count;
      if (!current.first_token_time_us.has_value()) {
        current.first_token_time_us = clock_.now_us();
      }
      return true;
    }

    case EventType::RequestComplete:
      if (!active_request_id_.has_value() ||
          *active_request_id_ != event.request_id ||
          (current.state != RequestState::Prefilling &&
           current.state != RequestState::Decoding) ||
          current.generated_token_count != current.max_new_tokens) {
        throw std::logic_error("duplicate or stale request completion event");
      }
      current.transition_to(RequestState::Finished);
      current.finish_time_us = clock_.now_us();
      scheduler_->notify_completed(event.request_id, clock_.now_us());
      active_request_id_.reset();
      return true;
  }
  throw std::logic_error("unknown event type");
}

}  // namespace llm_lab::serving
