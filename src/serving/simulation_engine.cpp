#include "serving/simulation_engine.h"

#include "serving/checked_math.h"

#include <stdexcept>
#include <utility>

namespace llm_lab::serving {

SimulationEngine::SimulationEngine(const Backend& backend) : backend_(backend) {
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
    events_.schedule(arrival_time_us, id, EventType::RequestArrival);
  } catch (...) {
    requests_.erase(id);
    throw;
  }
}

void SimulationEngine::run() {
  if (failed_) {
    throw std::logic_error("cannot run a failed simulation engine");
  }
  try {
    while (!events_.empty()) {
      const Event event = events_.pop();
      clock_.advance_to(event.timestamp_us);
      process_event(event);
      start_next_if_idle();
      event_log_.push_back(event);
    }
    if (active_request_id_.has_value() || !waiting_.empty()) {
      throw std::logic_error("simulation drained with unfinished work");
    }
  } catch (...) {
    failed_ = true;
    throw;
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
  events_.schedule(checked_add(clock_.now_us(), cost_us), request_id, type);
}

void SimulationEngine::start_next_if_idle() {
  if (active_request_id_.has_value() || waiting_.empty()) {
    return;
  }

  const RequestId id = waiting_.front();
  Request& next = mutable_request(id);
  if (next.state != RequestState::Waiting ||
      next.arrival_time_us > clock_.now_us()) {
    throw std::logic_error("invalid request in FIFO waiting queue");
  }

  const std::int64_t prefill_cost =
      backend_.estimate_prefill_us(next.prompt_token_count, 1);
  schedule_after(prefill_cost, id, EventType::PrefillComplete);

  waiting_.pop_front();
  next.transition_to(RequestState::Prefilling);
  next.admitted_time_us = clock_.now_us();
  next.first_scheduled_time_us = clock_.now_us();
  active_request_id_ = id;
}

void SimulationEngine::process_event(const Event& event) {
  Request& current = mutable_request(event.request_id);
  switch (event.type) {
    case EventType::RequestArrival:
      if (current.state != RequestState::Waiting ||
          current.admitted_time_us.has_value()) {
        throw std::logic_error("duplicate or stale arrival event");
      }
      waiting_.push_back(event.request_id);
      return;

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
      return;

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
      return;
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
      active_request_id_.reset();
      return;
  }
}

}  // namespace llm_lab::serving
