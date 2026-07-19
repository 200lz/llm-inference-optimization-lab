#include "serving/fcfs_scheduler.h"

#include "serving/checked_math.h"

#include <limits>
#include <stdexcept>

namespace llm_lab::serving {

SchedulerDecision::SchedulerDecision(SchedulerDecisionKind kind,
                                     std::optional<RequestId> request_id,
                                     std::uint64_t decision_epoch)
    : kind_(kind),
      request_id_(request_id),
      decision_epoch_(decision_epoch) {}

SchedulerDecision SchedulerDecision::admit(RequestId request_id,
                                           std::uint64_t decision_epoch) {
  return SchedulerDecision(SchedulerDecisionKind::Admit, request_id,
                           decision_epoch);
}

SchedulerDecision SchedulerDecision::no_work(std::uint64_t decision_epoch) {
  return SchedulerDecision(SchedulerDecisionKind::NoWork, std::nullopt,
                           decision_epoch);
}

SchedulerDecision SchedulerDecision::capacity_full(
    std::uint64_t decision_epoch) {
  return SchedulerDecision(SchedulerDecisionKind::CapacityFull, std::nullopt,
                           decision_epoch);
}

FcfsScheduler::FcfsScheduler(std::size_t max_active_requests)
    : max_active_requests_(max_active_requests) {
  if (max_active_requests_ == 0) {
    throw std::invalid_argument("scheduler capacity must be positive");
  }
}

bool FcfsScheduler::WaitingKeyLess::operator()(
    const WaitingKey& lhs, const WaitingKey& rhs) const noexcept {
  if (lhs.arrival_time_us != rhs.arrival_time_us) {
    return lhs.arrival_time_us < rhs.arrival_time_us;
  }
  return lhs.request_id < rhs.request_id;
}

void FcfsScheduler::increment(std::uint64_t& value,
                              const char* description) {
  if (value == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error(description);
  }
  ++value;
}

std::uint64_t FcfsScheduler::next_epoch() const {
  if (epoch_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("scheduler decision epoch overflow");
  }
  return epoch_ + 1;
}

void FcfsScheduler::notify_arrival(RequestId request_id,
                                   std::int64_t arrival_time_us) {
  if (arrival_time_us < 0) {
    throw std::invalid_argument("scheduler arrival time must be non-negative");
  }
  if (entries_.find(request_id) != entries_.end()) {
    throw std::logic_error("request was already known to scheduler");
  }
  if (statistics_.arrived_request_count ==
      std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("arrived request count overflow");
  }
  const std::uint64_t new_epoch = next_epoch();

  const WaitingKey key{arrival_time_us, request_id};
  const auto inserted_waiting = waiting_.insert(key);
  if (!inserted_waiting.second) {
    throw std::logic_error("duplicate FCFS waiting key");
  }
  try {
    entries_.emplace(request_id,
                     Entry{arrival_time_us, EntryState::Waiting, true,
                           std::nullopt});
    ++statistics_.arrived_request_count;
    const auto depth = static_cast<std::uint64_t>(waiting_.size());
    if (depth > statistics_.maximum_waiting_queue_depth) {
      statistics_.maximum_waiting_queue_depth = depth;
    }
    epoch_ = new_epoch;
  } catch (...) {
    entries_.erase(request_id);
    waiting_.erase(key);
    throw;
  }
}

SchedulerDecision FcfsScheduler::choose_next_request() const {
  if (waiting_.empty()) {
    return SchedulerDecision::no_work(epoch_);
  }
  if (running_count_ >= max_active_requests_) {
    return SchedulerDecision::capacity_full(epoch_);
  }
  return SchedulerDecision::admit(waiting_.begin()->request_id, epoch_);
}

FcfsScheduler::EntryMap::iterator FcfsScheduler::require_entry(
    RequestId request_id) {
  const auto found = entries_.find(request_id);
  if (found == entries_.end()) {
    throw std::logic_error("request is unknown to scheduler");
  }
  return found;
}

void FcfsScheduler::notify_admitted(const SchedulerDecision& decision,
                                    std::int64_t timestamp_us) {
  if (decision.kind() != SchedulerDecisionKind::Admit) {
    throw std::logic_error("scheduler decision is not an admission");
  }
  const std::optional<RequestId> decision_request_id = decision.request_id();
  if (!decision_request_id.has_value()) {
    throw std::logic_error("admission decision has no request ID");
  }
  if (decision.decision_epoch() != epoch_) {
    throw std::logic_error("stale scheduler admission decision");
  }
  if (waiting_.empty() ||
      waiting_.begin()->request_id != *decision_request_id) {
    throw std::logic_error("admission decision does not select FCFS head");
  }
  const RequestId request_id = *decision_request_id;
  const auto found = require_entry(request_id);
  Entry& entry = found->second;
  if (!entry.has_arrived || entry.state != EntryState::Waiting) {
    throw std::logic_error("request is not waiting for admission");
  }
  if (running_count_ >= max_active_requests_) {
    throw std::logic_error("scheduler admission exceeds capacity");
  }
  if (timestamp_us < entry.arrival_time_us) {
    throw std::logic_error("request admitted before arrival");
  }
  const std::int64_t wait_us = timestamp_us - entry.arrival_time_us;
  const std::int64_t new_total =
      checked_add(statistics_.total_queue_wait_time_us, wait_us);
  if (statistics_.admitted_request_count ==
      std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("admitted request count overflow");
  }
  const std::uint64_t new_epoch = next_epoch();

  const std::size_t erased =
      waiting_.erase(WaitingKey{entry.arrival_time_us, request_id});
  if (erased != 1) {
    throw std::logic_error("waiting request missing from FCFS order");
  }
  entry.state = EntryState::Running;
  entry.admitted_time_us = timestamp_us;
  ++running_count_;
  ++statistics_.admitted_request_count;
  statistics_.total_queue_wait_time_us = new_total;
  epoch_ = new_epoch;
}

void FcfsScheduler::notify_completed(RequestId request_id,
                                     std::int64_t timestamp_us) {
  const auto found = require_entry(request_id);
  Entry& entry = found->second;
  if (entry.state != EntryState::Running || running_count_ == 0) {
    throw std::logic_error("request is not running at completion");
  }
  if (!entry.admitted_time_us.has_value() ||
      timestamp_us < *entry.admitted_time_us) {
    throw std::logic_error("request completed before admission");
  }
  const std::uint64_t new_epoch = next_epoch();
  increment(statistics_.completed_request_count,
            "completed request count overflow");
  entry.state = EntryState::Completed;
  --running_count_;
  epoch_ = new_epoch;
}

void FcfsScheduler::notify_cancelled(RequestId request_id,
                                     std::int64_t timestamp_us) {
  if (timestamp_us < 0) {
    throw std::invalid_argument(
        "scheduler cancellation time must be non-negative");
  }
  if (statistics_.cancelled_request_count ==
      std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("cancelled request count overflow");
  }
  const std::uint64_t new_epoch = next_epoch();
  auto found = entries_.find(request_id);
  if (found == entries_.end()) {
    entries_.emplace(request_id,
                     Entry{timestamp_us, EntryState::Cancelled, false,
                           std::nullopt});
    ++statistics_.cancelled_request_count;
    epoch_ = new_epoch;
    return;
  }

  Entry& entry = found->second;
  if (entry.state == EntryState::Completed ||
      entry.state == EntryState::Cancelled) {
    throw std::logic_error("terminal request cannot be cancelled");
  }
  if (entry.state == EntryState::Running &&
      !entry.admitted_time_us.has_value()) {
    throw std::logic_error("running request has no admission time");
  }
  const std::int64_t earliest_time = entry.state == EntryState::Running
                                         ? *entry.admitted_time_us
                                         : entry.arrival_time_us;
  if (timestamp_us < earliest_time) {
    throw std::logic_error("request cancelled before its current state");
  }
  if (entry.state == EntryState::Waiting) {
    const std::size_t erased =
        waiting_.erase(WaitingKey{entry.arrival_time_us, request_id});
    if (erased != 1) {
      throw std::logic_error("cancelled request missing from FCFS order");
    }
  } else {
    if (running_count_ == 0) {
      throw std::logic_error("running scheduler count underflow");
    }
    --running_count_;
  }
  ++statistics_.cancelled_request_count;
  entry.state = EntryState::Cancelled;
  epoch_ = new_epoch;
}

std::size_t FcfsScheduler::waiting_count() const noexcept {
  return waiting_.size();
}

std::size_t FcfsScheduler::running_count() const noexcept {
  return running_count_;
}

bool FcfsScheduler::empty() const noexcept {
  return waiting_.empty() && running_count_ == 0;
}

const SchedulerStatistics& FcfsScheduler::statistics() const noexcept {
  return statistics_;
}

}  // namespace llm_lab::serving
