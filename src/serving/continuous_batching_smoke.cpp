#include "serving/continuous_batching.h"

#include <iostream>

namespace {

void print_ids(const std::vector<llm_lab::serving::RequestId>& ids) {
  std::cout << '[';
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << ids[index].value;
  }
  std::cout << ']';
}

void print_optional(const std::optional<std::int64_t>& value) {
  if (value.has_value()) {
    std::cout << *value;
  } else {
    std::cout << "N/A";
  }
}

}  // namespace

int main() {
  using namespace llm_lab::serving;

  const SimulatedBackend backend({10, 2, 1, 4, 1, 2, 1, 2, 1});
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(2, 4, SchedulingPolicy::DecodeFirst));
  engine.submit_request(Request({101}, 0, 2, 3));
  engine.submit_request(Request({102}, 0, 2, 2));
  engine.submit_request(Request({103}, 5, 1, 1));
  engine.run();

  std::cout << "SIMULATED CONTINUOUS BATCHING\n";
  std::cout << "policy=DecodeFirst (no threads; modeled time only)\n";
  std::cout << "iteration start_us end_us prefill_ids decode_ids scheduled_tokens\n";
  for (const BatchTraceEntry& entry : engine.plan_trace()) {
    std::cout << entry.iteration_number << ' ' << entry.start_timestamp_us
              << ' ' << entry.end_timestamp_us << ' ';
    print_ids(entry.plan.prefill_request_ids());
    std::cout << ' ';
    print_ids(entry.plan.decode_request_ids());
    std::cout << ' ' << entry.plan.total_scheduled_tokens() << '\n';
  }

  std::cout << "request_id state arrival_us admission_us first_token_us "
               "finish_us generated_tokens\n";
  for (const auto& entry : engine.requests()) {
    const Request& request = entry.second;
    std::cout << request.request_id.value << ' ' << to_string(request.state)
              << ' ' << request.arrival_time_us << ' ';
    print_optional(request.admitted_time_us);
    std::cout << ' ';
    print_optional(request.first_token_time_us);
    std::cout << ' ';
    print_optional(request.finish_time_us);
    std::cout << ' ' << request.generated_token_count << '\n';
  }

  const auto& stats = engine.statistics();
  std::cout << "SIMULATED batch_statistics iterations="
            << stats.scheduling_iteration_count
            << " nonempty=" << stats.nonempty_batch_count
            << " idle=" << stats.idle_iteration_count
            << " prefill_tokens=" << stats.total_prefill_tokens_scheduled
            << " decode_tokens=" << stats.total_decode_tokens_scheduled
            << " scheduled_sequences=" << stats.total_scheduled_sequences
            << " max_batch=" << stats.maximum_batch_size
            << " average_batch=";
  const auto average = stats.average_batch_size();
  if (average.has_value()) {
    std::cout << *average;
  } else {
    std::cout << "N/A";
  }
  std::cout << " max_tokens=" << stats.maximum_scheduled_tokens
            << " deferred=" << stats.deferred_request_count
            << " completed=" << stats.completed_request_count
            << " cancelled=" << stats.cancelled_request_count << '\n';
}
