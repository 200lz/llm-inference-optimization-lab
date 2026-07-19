#pragma once

#include "serving/request.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace llm_lab::serving {

enum class SchedulerDecisionKind {
  Admit,
  NoWork,
  CapacityFull,
};

struct SchedulerDecision {
 public:
  static SchedulerDecision admit(RequestId request_id,
                                 std::uint64_t decision_epoch);
  static SchedulerDecision no_work(std::uint64_t decision_epoch);
  static SchedulerDecision capacity_full(std::uint64_t decision_epoch);

  SchedulerDecisionKind kind() const noexcept { return kind_; }
  std::optional<RequestId> request_id() const noexcept { return request_id_; }
  std::uint64_t decision_epoch() const noexcept { return decision_epoch_; }

 private:
  SchedulerDecision(SchedulerDecisionKind kind,
                    std::optional<RequestId> request_id,
                    std::uint64_t decision_epoch);

  SchedulerDecisionKind kind_;
  std::optional<RequestId> request_id_;
  std::uint64_t decision_epoch_;
};

struct SchedulerStatistics {
  std::uint64_t arrived_request_count{0};
  std::uint64_t admitted_request_count{0};
  std::uint64_t completed_request_count{0};
  std::uint64_t cancelled_request_count{0};
  std::uint64_t maximum_waiting_queue_depth{0};
  std::int64_t total_queue_wait_time_us{0};
};

class Scheduler {
 public:
  virtual ~Scheduler() = default;

  virtual void notify_arrival(RequestId request_id,
                              std::int64_t arrival_time_us) = 0;
  virtual SchedulerDecision choose_next_request() const = 0;
  virtual void notify_admitted(const SchedulerDecision& decision,
                               std::int64_t timestamp_us) = 0;
  virtual void notify_completed(RequestId request_id,
                                std::int64_t timestamp_us) = 0;
  // A cancellation may be the first scheduler notification when the engine
  // cancels a submitted request before its future arrival event.
  virtual void notify_cancelled(RequestId request_id,
                                std::int64_t timestamp_us) = 0;

  virtual std::size_t waiting_count() const noexcept = 0;
  virtual std::size_t running_count() const noexcept = 0;
  virtual bool empty() const noexcept = 0;
  virtual const SchedulerStatistics& statistics() const noexcept = 0;
};

}  // namespace llm_lab::serving
