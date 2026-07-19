#pragma once

#include "serving/backend.h"
#include "serving/event.h"
#include "serving/request.h"
#include "serving/scheduler.h"
#include "serving/simulation_clock.h"

#include <cstdint>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace llm_lab::serving {

namespace test {
struct SimulationEngineTestAccess;
}

struct SimulationEngineConfig {
  std::size_t max_active_requests{1};
};

class SimulationEngine {
 public:
  // The named lvalue backend is borrowed and must outlive the engine.
  // Its configuration is validated here. Rvalue backends are rejected to
  // prevent a temporary from leaving a dangling reference.
  explicit SimulationEngine(
      const Backend& backend,
      SimulationEngineConfig config = SimulationEngineConfig{});
  SimulationEngine(Backend&& backend) = delete;
  SimulationEngine(Backend&& backend, SimulationEngineConfig config) = delete;

  void submit_request(Request request);
  void cancel_request(RequestId request_id);
  bool run_next_timestamp();
  void run();

  const Request& request(RequestId request_id) const;
  const std::map<RequestId, Request>& requests() const;
  const std::vector<Event>& event_log() const;
  const SimulationClock& clock() const;
  const SchedulerStatistics& scheduler_statistics() const;
  std::size_t scheduler_waiting_count() const;
  std::size_t scheduler_running_count() const;
  bool failed() const noexcept { return failed_; }
  std::size_t processed_event_count() const noexcept { return event_log_.size(); }
  bool event_queue_empty() const noexcept { return events_.empty(); }
  std::uint64_t ignored_cancelled_event_count() const noexcept {
    return ignored_cancelled_event_count_;
  }

 private:
  struct PendingEvent {
    std::uint64_t event_sequence;
    EventType type;
    bool authorized_cancelled_ignore{false};
  };

  bool process_event(const Event& event);
  void start_next_if_idle();
  void schedule_request_event(std::int64_t timestamp_us,
                              RequestId request_id, EventType type);
  void schedule_after(std::int64_t cost_us, RequestId request_id,
                      EventType type);
  Request& mutable_request(RequestId request_id);
  void require_results_available() const;

  const Backend& backend_;
  std::unique_ptr<Scheduler> scheduler_;
  SimulationClock clock_;
  EventQueue events_;
  std::map<RequestId, Request> requests_;
  std::map<RequestId, PendingEvent> pending_events_;
  std::optional<RequestId> active_request_id_;
  std::vector<Event> event_log_;
  std::uint64_t ignored_cancelled_event_count_{0};
  bool failed_{false};

  friend struct test::SimulationEngineTestAccess;
};

}  // namespace llm_lab::serving
