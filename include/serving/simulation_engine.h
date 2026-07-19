#pragma once

#include "serving/backend.h"
#include "serving/event.h"
#include "serving/request.h"
#include "serving/simulation_clock.h"

#include <cstdint>
#include <cstddef>
#include <deque>
#include <map>
#include <optional>
#include <vector>

namespace llm_lab::serving {

class SimulationEngine {
 public:
  // The named lvalue backend is borrowed and must outlive the engine.
  // Its configuration is validated here. Rvalue backends are rejected to
  // prevent a temporary from leaving a dangling reference.
  explicit SimulationEngine(const Backend& backend);
  SimulationEngine(Backend&& backend) = delete;

  void submit_request(Request request);
  void run();

  const Request& request(RequestId request_id) const;
  const std::map<RequestId, Request>& requests() const;
  const std::vector<Event>& event_log() const;
  const SimulationClock& clock() const;
  bool failed() const noexcept { return failed_; }
  std::size_t processed_event_count() const noexcept { return event_log_.size(); }
  bool event_queue_empty() const noexcept { return events_.empty(); }

 private:
  void process_event(const Event& event);
  void start_next_if_idle();
  void schedule_after(std::int64_t cost_us, RequestId request_id,
                      EventType type);
  Request& mutable_request(RequestId request_id);
  void require_results_available() const;

  const Backend& backend_;
  SimulationClock clock_;
  EventQueue events_;
  std::map<RequestId, Request> requests_;
  std::deque<RequestId> waiting_;
  std::optional<RequestId> active_request_id_;
  std::vector<Event> event_log_;
  bool failed_{false};
};

}  // namespace llm_lab::serving
