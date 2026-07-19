#pragma once

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
      SchedulingPolicy scheduling_policy = SchedulingPolicy::DecodeFirst);

  std::size_t max_num_sequences() const noexcept { return max_num_sequences_; }
  std::uint64_t max_batched_tokens() const noexcept {
    return max_batched_tokens_;
  }
  SchedulingPolicy scheduling_policy() const noexcept {
    return scheduling_policy_;
  }

 private:
  std::size_t max_num_sequences_;
  std::uint64_t max_batched_tokens_;
  SchedulingPolicy scheduling_policy_;
};

enum class DeferralReason { SequenceBudget, TokenBudget };
std::string_view to_string(DeferralReason reason) noexcept;

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
  bool empty() const noexcept { return scheduled_sequence_count() == 0; }

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
};

struct ContinuousBatchingStatistics {
  std::uint64_t scheduling_iteration_count{0};
  std::uint64_t nonempty_batch_count{0};
  std::uint64_t idle_iteration_count{0};
  std::uint64_t total_prefill_tokens_scheduled{0};
  std::uint64_t total_decode_tokens_scheduled{0};
  std::uint64_t total_scheduled_sequences{0};
  std::uint64_t maximum_batch_size{0};
  std::uint64_t maximum_scheduled_tokens{0};
  std::uint64_t deferred_request_count{0};
  std::uint64_t completed_request_count{0};
  std::uint64_t cancelled_request_count{0};

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
  bool run_next_iteration();
  void run();

  const Request& request(RequestId request_id) const;
  const std::map<RequestId, Request>& requests() const;
  const std::vector<BatchTraceEntry>& plan_trace() const;
  const ContinuousBatchingStatistics& statistics() const;
  const SimulationClock& clock() const;
  bool failed() const noexcept { return failed_; }

 private:
  struct Candidate {
    RequestId request_id;
    std::int64_t arrival_time_us;
    bool is_prefill;
    std::uint64_t token_cost;
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
      const BatchPlan& plan, std::uint64_t completed_request_count) const;
  PreparedIteration prepare_iteration(
      BatchPlan plan, std::set<RequestId> arrived_requests,
      std::int64_t end_timestamp_us);
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
  std::uint64_t next_iteration_number_{1};
  bool failed_{false};
  bool fail_trace_preparation_for_test_{false};

  friend struct test::ContinuousBatchingEngineTestAccess;
};

}  // namespace llm_lab::serving
