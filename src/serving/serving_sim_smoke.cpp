#include "serving/request.h"
#include "serving/simulated_backend.h"
#include "serving/simulation_engine.h"

#include <iostream>

int main() {
  using namespace llm_lab::serving;

  const SimulatedBackend backend({10, 2, 1, 4, 1});
  SimulationEngine engine(backend);
  engine.submit_request(Request({101}, 0, 4, 2));
  engine.submit_request(Request({102}, 0, 0, 1));
  engine.submit_request(Request({103}, 3, 2, 0));
  engine.run_next_timestamp();
  engine.cancel_request({102});
  engine.run();

  std::cout << "SIMULATED request_id final_state arrival_time_us "
               "admitted_time_us first_token_time_us finish_time_us "
               "generated_token_count\n";
  for (const auto& entry : engine.requests()) {
    const Request& request = entry.second;
    std::cout << "SIMULATED " << request.request_id.value << ' '
              << to_string(request.state) << ' ' << request.arrival_time_us
              << ' ';
    if (request.admitted_time_us.has_value()) {
      std::cout << *request.admitted_time_us;
    } else {
      std::cout << "N/A";
    }
    std::cout << ' ';
    if (request.first_token_time_us.has_value()) {
      std::cout << *request.first_token_time_us;
    } else {
      std::cout << "N/A";
    }
    std::cout << ' ';
    if (request.finish_time_us.has_value()) {
      std::cout << *request.finish_time_us;
    } else {
      std::cout << "N/A";
    }
    std::cout << ' ' << request.generated_token_count << '\n';
  }

  const SchedulerStatistics& statistics = engine.scheduler_statistics();
  std::cout << "SIMULATED scheduler=FCFS arrived="
            << statistics.arrived_request_count
            << " admitted=" << statistics.admitted_request_count
            << " completed=" << statistics.completed_request_count
            << " cancelled=" << statistics.cancelled_request_count
            << " max_waiting=" << statistics.maximum_waiting_queue_depth
            << " total_queue_wait_us="
            << statistics.total_queue_wait_time_us
            << " waiting=" << engine.scheduler_waiting_count()
            << " running=" << engine.scheduler_running_count()
            << " ignored_cancelled_events="
            << engine.ignored_cancelled_event_count() << '\n';
}
