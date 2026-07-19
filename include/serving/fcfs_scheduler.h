#pragma once

#include "serving/scheduler.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>

namespace llm_lab::serving {

class FcfsScheduler final : public Scheduler {
 public:
  explicit FcfsScheduler(std::size_t max_active_requests = 1);

  void notify_arrival(RequestId request_id,
                      std::int64_t arrival_time_us) override;
  SchedulerDecision choose_next_request() const override;
  void notify_admitted(const SchedulerDecision& decision,
                       std::int64_t timestamp_us) override;
  void notify_completed(RequestId request_id,
                        std::int64_t timestamp_us) override;
  void notify_cancelled(RequestId request_id,
                        std::int64_t timestamp_us) override;

  std::size_t waiting_count() const noexcept override;
  std::size_t running_count() const noexcept override;
  bool empty() const noexcept override;
  const SchedulerStatistics& statistics() const noexcept override;
  std::size_t max_active_requests() const noexcept {
    return max_active_requests_;
  }

 private:
  enum class EntryState { Waiting, Running, Completed, Cancelled };

  struct Entry {
    std::int64_t arrival_time_us;
    EntryState state;
    bool has_arrived;
    std::optional<std::int64_t> admitted_time_us;
  };

  struct WaitingKey {
    std::int64_t arrival_time_us;
    RequestId request_id;
  };

  struct WaitingKeyLess {
    bool operator()(const WaitingKey& lhs,
                    const WaitingKey& rhs) const noexcept;
  };

  using EntryMap = std::map<RequestId, Entry>;

  EntryMap::iterator require_entry(RequestId request_id);
  void increment(std::uint64_t& value, const char* description);
  std::uint64_t next_epoch() const;

  std::size_t max_active_requests_;
  std::set<WaitingKey, WaitingKeyLess> waiting_;
  EntryMap entries_;
  std::size_t running_count_{0};
  std::uint64_t epoch_{0};
  SchedulerStatistics statistics_;
};

}  // namespace llm_lab::serving
