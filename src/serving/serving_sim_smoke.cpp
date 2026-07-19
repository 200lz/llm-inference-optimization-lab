#include "serving/request.h"
#include "serving/simulated_backend.h"
#include "serving/simulation_engine.h"

#include <iostream>

int main() {
  using namespace llm_lab::serving;

  const SimulatedBackend backend({10, 2, 1, 4, 1});
  SimulationEngine engine(backend);
  engine.submit_request(Request({101}, 0, 4, 2));
  engine.submit_request(Request({102}, 3, 0, 1));
  engine.submit_request(Request({103}, 3, 2, 0));
  engine.run();

  std::cout << "SIMULATED request_id arrival_time_us first_token_time_us "
               "finish_time_us generated_token_count final_state\n";
  for (const auto& entry : engine.requests()) {
    const Request& request = entry.second;
    std::cout << "SIMULATED " << request.request_id.value << ' '
              << request.arrival_time_us << ' ';
    if (request.first_token_time_us.has_value()) {
      std::cout << *request.first_token_time_us;
    } else {
      std::cout << "N/A";
    }
    std::cout << ' ' << *request.finish_time_us << ' '
              << request.generated_token_count << ' '
              << to_string(request.state) << '\n';
  }
}
