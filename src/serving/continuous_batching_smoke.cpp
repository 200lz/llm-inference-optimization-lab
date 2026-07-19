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

void print_deferred(
    const std::vector<llm_lab::serving::DeferredRequest>& deferred) {
  std::cout << '[';
  for (std::size_t index = 0; index < deferred.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << deferred[index].request_id.value << ':'
              << llm_lab::serving::to_string(deferred[index].reason);
  }
  std::cout << ']';
}

void print_block_tables(const llm_lab::serving::BatchTraceEntry& entry) {
  std::cout << '[';
  std::size_t request_index = 0;
  for (const auto& table : entry.kv_block_tables) {
    if (request_index++ != 0) {
      std::cout << ',';
    }
    std::cout << table.first.value << "={";
    for (std::size_t index = 0; index < table.second.size(); ++index) {
      if (index != 0) {
        std::cout << ',';
      }
      std::cout << table.second[index];
    }
    std::cout << '}';
  }
  std::cout << ']';
}

}  // namespace

int main() {
  using namespace llm_lab::serving;

  const SimulatedBackend backend({10, 2, 1, 4, 1, 2, 1, 2, 1});
  ContinuousBatchingEngine engine(
      backend,
      ContinuousBatchingConfig(2, 8, SchedulingPolicy::DecodeFirst,
                               KVCacheConfig(3, 2)));
  engine.submit_request(Request({101}, 0, 4, 2));
  engine.submit_request(Request({102}, 0, 4, 1));
  const RunResult run_result = engine.run();

  std::cout << "SIMULATED BLOCK-BASED KV CACHE\n";
  std::cout << "metadata-only tokens; policy=DecodeFirst; blocks=3; block_tokens=2\n";
  std::cout << "run_result="
            << (run_result == RunResult::Completed ? "Completed" : "Stalled")
            << " (stalls are nonterminal simulated control flow)\n";
  std::cout << "iteration prefill decode deferred block_tables allocated/free "
               "represented fragmentation utilization\n";
  for (const BatchTraceEntry& entry : engine.plan_trace()) {
    std::cout << entry.iteration_number << ' ';
    print_ids(entry.plan.prefill_request_ids());
    std::cout << ' ';
    print_ids(entry.plan.decode_request_ids());
    std::cout << ' ';
    print_deferred(entry.plan.deferred_requests());
    std::cout << ' ';
    print_block_tables(entry);
    std::cout << ' ' << entry.allocated_kv_blocks << '/'
              << entry.free_kv_blocks << ' ' << entry.represented_kv_tokens
              << ' ' << entry.internal_fragmentation_tokens << ' '
              << entry.kv_block_utilization << '\n';
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
            << " stalled=" << stats.stalled_iteration_count
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
            << " kv_deferrals=" << stats.kv_capacity_deferral_count
            << " kv_allocated=" << stats.current_allocated_kv_blocks
            << " kv_peak=" << stats.peak_allocated_kv_blocks
            << " kv_utilization=" << stats.current_kv_block_utilization
            << " kv_peak_utilization=" << stats.peak_kv_block_utilization
            << " represented=" << stats.represented_kv_tokens
            << " fragmentation=" << stats.internal_fragmentation_tokens
            << " allocation_failures=" << stats.kv_allocation_failure_count
            << " completed=" << stats.completed_request_count
            << " cancelled=" << stats.cancelled_request_count << '\n';
}
