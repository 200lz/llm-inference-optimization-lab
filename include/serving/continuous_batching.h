#pragma once

#include "serving/kv_cache.h"
#include "serving/request.h"
#include "serving/simulated_backend.h"
#include "serving/simulation_clock.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <vector>

namespace llm_lab::serving {

namespace test {
struct ContinuousBatchingEngineTestAccess;
}

enum class SchedulingPolicy { DecodeFirst, FcfsMixed };
std::string_view to_string(SchedulingPolicy policy) noexcept;

class ContinuousBatchingConfig {
 public:
  explicit ContinuousBatchingConfig(
      std::size_t max_num_sequences = 8,
      std::uint64_t max_batched_tokens = 512,
      SchedulingPolicy scheduling_policy = SchedulingPolicy::DecodeFirst,
      KVCacheConfig kv_cache_config = KVCacheConfig(1024, 16));

  std::size_t max_num_sequences() const noexcept { return max_num_sequences_; }
  std::uint64_t max_batched_tokens() const noexcept {
    return max_batched_tokens_;
  }
  SchedulingPolicy scheduling_policy() const noexcept {
    return scheduling_policy_;
  }
  const KVCacheConfig& kv_cache_config() const noexcept {
    return kv_cache_config_;
  }

 private:
  std::size_t max_num_sequences_;
  std::uint64_t max_batched_tokens_;
  SchedulingPolicy scheduling_policy_;
  KVCacheConfig kv_cache_config_;
};

enum class DeferralReason { SequenceBudget, TokenBudget, KVCapacity };
std::string_view to_string(DeferralReason reason) noexcept;

enum class IterationOutcome { Progressed, Stalled, Complete };

class IterationResult {
 public:
  static IterationResult progressed() noexcept {
    return IterationResult(IterationOutcome::Progressed);
  }
  static IterationResult stalled() noexcept {
    return IterationResult(IterationOutcome::Stalled);
  }
  static IterationResult complete() noexcept {
    return IterationResult(IterationOutcome::Complete);
  }

  IterationOutcome outcome() const noexcept { return outcome_; }
  bool made_progress() const noexcept {
    return outcome_ == IterationOutcome::Progressed;
  }
  bool is_stalled() const noexcept {
    return outcome_ == IterationOutcome::Stalled;
  }
  operator bool() const noexcept {
    return outcome_ != IterationOutcome::Complete;
  }

 private:
  explicit IterationResult(IterationOutcome outcome) noexcept
      : outcome_(outcome) {}
  IterationOutcome outcome_;
};

enum class RunResult { Completed, Stalled };

struct DeferredRequest {
  RequestId request_id;
  DeferralReason reason;
};

struct PrefillWork {
  RequestId request_id;
  std::uint64_t prompt_token_count;
};

class BatchPlan {
 public:
  static BatchPlan create(std::uint64_t iteration_number,
                          std::vector<PrefillWork> prefill_work,
                          std::vector<RequestId> decode_request_ids,
                          std::size_t max_num_sequences,
                          std::uint64_t max_batched_tokens);
  // An explicit zero-work plan represents an interval in which no runnable
  // work fits (normally because the engine is waiting for the next arrival).
  static BatchPlan empty(std::uint64_t iteration_number);

  std::uint64_t iteration_number() const noexcept { return iteration_number_; }
  const std::vector<RequestId>& prefill_request_ids() const noexcept {
    return prefill_request_ids_;
  }
  const std::vector<RequestId>& decode_request_ids() const noexcept {
    return decode_request_ids_;
  }
  const std::vector<DeferredRequest>& deferred_requests() const noexcept {
    return deferred_requests_;
  }
  std::uint64_t total_prefill_tokens() const noexcept {
    return total_prefill_tokens_;
  }
  std::uint64_t total_decode_tokens() const noexcept {
    return total_decode_tokens_;
  }
  std::uint64_t total_scheduled_tokens() const noexcept {
    return total_scheduled_tokens_;
  }
  std::size_t scheduled_sequence_count() const noexcept {
    return prefill_request_ids_.size() + decode_request_ids_.size();
  }
  bool empty() const noexcept {
    return scheduled_sequence_count() == 0 && deferred_requests_.empty();
  }
  bool deferred_only() const noexcept {
    return scheduled_sequence_count() == 0 && !deferred_requests_.empty();
  }

 private:
  static BatchPlan create_with_deferred(
      std::uint64_t iteration_number,
      std::vector<PrefillWork> prefill_work,
      std::vector<RequestId> decode_request_ids,
      std::vector<DeferredRequest> deferred_requests,
      std::size_t max_num_sequences,
      std::uint64_t max_batched_tokens);
  BatchPlan(std::uint64_t iteration_number,
            std::vector<RequestId> prefill_request_ids,
            std::vector<RequestId> decode_request_ids,
            std::vector<DeferredRequest> deferred_requests,
            std::uint64_t total_prefill_tokens,
            std::uint64_t total_decode_tokens,
            std::uint64_t total_scheduled_tokens);

  std::uint64_t iteration_number_;
  std::vector<RequestId> prefill_request_ids_;
  std::vector<RequestId> decode_request_ids_;
  std::vector<DeferredRequest> deferred_requests_;
  std::uint64_t total_prefill_tokens_;
  std::uint64_t total_decode_tokens_;
  std::uint64_t total_scheduled_tokens_;

  friend class ContinuousBatchingEngine;
  friend struct test::ContinuousBatchingEngineTestAccess;
};

struct BatchTraceEntry {
  std::uint64_t iteration_number;
  std::int64_t start_timestamp_us;
  std::int64_t end_timestamp_us;
  SchedulingPolicy policy;
  BatchPlan plan;
  std::size_t allocated_kv_blocks;
  std::size_t free_kv_blocks;
  std::uint64_t represented_kv_tokens;
  std::uint64_t internal_fragmentation_tokens;
  double kv_block_utilization;
  std::map<RequestId, std::vector<std::size_t>> kv_block_tables;
};

struct ContinuousBatchingStatistics {
  std::uint64_t scheduling_iteration_count{0};
  std::uint64_t nonempty_batch_count{0};
  std::uint64_t idle_iteration_count{0};
  std::uint64_t stalled_iteration_count{0};
  std::uint64_t total_prefill_tokens_scheduled{0};
  std::uint64_t total_decode_tokens_scheduled{0};
  std::uint64_t total_scheduled_sequences{0};
  std::uint64_t maximum_batch_size{0};
  std::uint64_t maximum_scheduled_tokens{0};
  std::uint64_t deferred_request_count{0};
  std::uint64_t completed_request_count{0};
  std::uint64_t cancelled_request_count{0};
  std::size_t current_allocated_kv_blocks{0};
  std::size_t peak_allocated_kv_blocks{0};
  double current_kv_block_utilization{0.0};
  double peak_kv_block_utilization{0.0};
  std::uint64_t represented_kv_tokens{0};
  std::uint64_t internal_fragmentation_tokens{0};
  std::uint64_t kv_allocation_failure_count{0};
  // Counts one occurrence per request deferred for KV capacity per iteration.
  std::uint64_t kv_capacity_deferral_count{0};

  // Undefined when nonempty_batch_count is zero.
  std::optional<double> average_batch_size() const noexcept;
};

class ContinuousBatchingEngine {
 public:
  // The backend is borrowed and must outlive the engine.
  ContinuousBatchingEngine(const SimulatedBackend& backend,
                           ContinuousBatchingConfig config);
  ContinuousBatchingEngine(SimulatedBackend&& backend,
                           ContinuousBatchingConfig config) = delete;

  void submit_request(Request request);
  void cancel_request(RequestId request_id);
  IterationResult run_next_iteration();
  RunResult run();

  const Request& request(RequestId request_id) const;
  const std::map<RequestId, Request>& requests() const;
  const std::vector<BatchTraceEntry>& plan_trace() const;
  const ContinuousBatchingStatistics& statistics() const;
  const SimulationClock& clock() const;
  const KVCacheManager& kv_cache() const;
  bool failed() const noexcept { return failed_; }

 private:
  struct Candidate {
    RequestId request_id;
    std::int64_t arrival_time_us;
    bool is_prefill;
    std::uint64_t token_cost;
    std::size_t kv_blocks_required;
  };

  struct PreparedRequestUpdate {
    Request* target;
    Request value;
  };

  struct PreparedIteration {
    std::set<RequestId> arrived_requests;
    std::vector<PreparedRequestUpdate> request_updates;
    ContinuousBatchingStatistics statistics;
    BatchTraceEntry trace_entry;
    std::uint64_t next_iteration_number;
    SimulationClock clock;
    KVCacheManager kv_cache;
  };

  std::set<RequestId> arrivals_at_current_timestamp() const;
  BatchPlan build_plan(std::uint64_t iteration_number,
                       const std::set<RequestId>& arrived_requests) const;
  BatchPlan build_decode_first(
      std::uint64_t iteration_number,
      const std::set<RequestId>& arrived_requests) const;
  BatchPlan build_fcfs_mixed(
      std::uint64_t iteration_number,
      const std::set<RequestId>& arrived_requests) const;
  BatchPlan build_from_candidates(std::uint64_t iteration_number,
                                  const std::vector<Candidate>& candidates) const;
  static bool candidate_less(const Candidate& lhs,
                             const Candidate& rhs) noexcept;
  std::vector<PreparedRequestUpdate> prepare_request_updates(
      const BatchPlan& plan, std::int64_t end_timestamp_us,
      const std::set<RequestId>& arrived_requests,
      std::uint64_t& completed_request_count);
  ContinuousBatchingStatistics prepare_statistics(
      const BatchPlan& plan, std::uint64_t completed_request_count,
      const KVCacheManager& prepared_kv_cache) const;
  PreparedIteration prepare_iteration(
      BatchPlan plan, std::set<RequestId> arrived_requests,
      std::int64_t time_value_us, bool time_value_is_duration);
  void commit_iteration(PreparedIteration prepared) noexcept;
  std::optional<std::int64_t> next_future_arrival(
      const std::set<RequestId>& arrived_requests) const;
  bool has_runnable_work(
      const std::set<RequestId>& arrived_requests) const;
  void require_results_available() const;

  const SimulatedBackend& backend_;
  ContinuousBatchingConfig config_;
  SimulationClock clock_;
  std::map<RequestId, Request> requests_;
  std::set<RequestId> arrived_requests_;
  std::vector<BatchTraceEntry> plan_trace_;
  ContinuousBatchingStatistics statistics_;
  KVCacheManager kv_cache_;
  std::uint64_t next_iteration_number_{1};
  bool failed_{false};
  bool fail_trace_preparation_for_test_{false};

  friend struct test::ContinuousBatchingEngineTestAccess;
};

}  // namespace llm_lab::serving
