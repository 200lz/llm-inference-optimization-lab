#include "serving/continuous_batching.h"
#include "serving/simulation_engine.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace llm_lab::serving::test {

struct ContinuousBatchingEngineTestAccess {
  static void set_statistics(ContinuousBatchingEngine& engine,
                             ContinuousBatchingStatistics statistics) {
    engine.statistics_ = statistics;
  }
  static void set_next_iteration(ContinuousBatchingEngine& engine,
                                 std::uint64_t value) {
    engine.next_iteration_number_ = value;
  }
  static void set_trace_failure(ContinuousBatchingEngine& engine, bool value) {
    engine.fail_trace_preparation_for_test_ = value;
  }
  static BatchPlan create_with_deferred(
      std::uint64_t iteration_number,
      std::vector<PrefillWork> prefills,
      std::vector<RequestId> decodes,
      std::vector<DeferredRequest> deferred,
      std::size_t max_sequences,
      std::uint64_t max_tokens) {
    return BatchPlan::create_with_deferred(
        iteration_number, std::move(prefills), std::move(decodes),
        std::move(deferred), max_sequences, max_tokens);
  }
  static const std::map<RequestId, Request>& raw_requests(
      const ContinuousBatchingEngine& engine) {
    return engine.requests_;
  }
  static const std::set<RequestId>& raw_arrivals(
      const ContinuousBatchingEngine& engine) {
    return engine.arrived_requests_;
  }
  static const std::vector<BatchTraceEntry>& raw_trace(
      const ContinuousBatchingEngine& engine) {
    return engine.plan_trace_;
  }
  static const ContinuousBatchingStatistics& raw_statistics(
      const ContinuousBatchingEngine& engine) {
    return engine.statistics_;
  }
  static std::int64_t raw_clock_us(const ContinuousBatchingEngine& engine) {
    return engine.clock_.now_us();
  }
  static std::uint64_t raw_next_iteration(
      const ContinuousBatchingEngine& engine) {
    return engine.next_iteration_number_;
  }
  static const KVCacheManager& raw_kv_cache(
      const ContinuousBatchingEngine& engine) {
    return engine.kv_cache_;
  }
};

}  // namespace llm_lab::serving::test

namespace {

using namespace llm_lab::serving;
using llm_lab::serving::test::ContinuousBatchingEngineTestAccess;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
  try {
    callable();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

SimulatedBackend batch_backend() {
  return SimulatedBackend({0, 0, 0, 0, 0, 2, 1, 2, 1});
}

std::vector<std::uint64_t> values(const std::vector<RequestId>& ids) {
  std::vector<std::uint64_t> result;
  for (RequestId id : ids) {
    result.push_back(id.value);
  }
  return result;
}

void require_ids(const std::vector<RequestId>& actual,
                 const std::vector<std::uint64_t>& expected,
                 const std::string& message) {
  require(values(actual) == expected, message);
}

bool same_optional(const std::optional<std::int64_t>& lhs,
                   const std::optional<std::int64_t>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  return !lhs.has_value() || *lhs == *rhs;
}

bool same_request(const Request& lhs, const Request& rhs) {
  return lhs.request_id == rhs.request_id &&
         lhs.arrival_time_us == rhs.arrival_time_us &&
         lhs.prompt_length() == rhs.prompt_length() &&
         lhs.has_exact_prompt_tokens() == rhs.has_exact_prompt_tokens() &&
         (!lhs.has_exact_prompt_tokens() || lhs.prompt_tokens() == rhs.prompt_tokens()) &&
         lhs.max_new_tokens == rhs.max_new_tokens &&
         lhs.generated_token_count == rhs.generated_token_count &&
         lhs.state == rhs.state &&
         same_optional(lhs.admitted_time_us, rhs.admitted_time_us) &&
         same_optional(lhs.first_scheduled_time_us,
                       rhs.first_scheduled_time_us) &&
         same_optional(lhs.first_token_time_us, rhs.first_token_time_us) &&
         same_optional(lhs.finish_time_us, rhs.finish_time_us);
}

bool same_statistics(const ContinuousBatchingStatistics& lhs,
                     const ContinuousBatchingStatistics& rhs) {
  return lhs.scheduling_iteration_count == rhs.scheduling_iteration_count &&
         lhs.nonempty_batch_count == rhs.nonempty_batch_count &&
         lhs.idle_iteration_count == rhs.idle_iteration_count &&
         lhs.stalled_iteration_count == rhs.stalled_iteration_count &&
         lhs.total_prefill_tokens_scheduled ==
             rhs.total_prefill_tokens_scheduled &&
         lhs.total_decode_tokens_scheduled ==
             rhs.total_decode_tokens_scheduled &&
         lhs.total_scheduled_sequences == rhs.total_scheduled_sequences &&
         lhs.maximum_batch_size == rhs.maximum_batch_size &&
         lhs.maximum_scheduled_tokens == rhs.maximum_scheduled_tokens &&
         lhs.deferred_request_count == rhs.deferred_request_count &&
         lhs.completed_request_count == rhs.completed_request_count &&
         lhs.cancelled_request_count == rhs.cancelled_request_count &&
         lhs.current_allocated_kv_blocks == rhs.current_allocated_kv_blocks &&
         lhs.peak_allocated_kv_blocks == rhs.peak_allocated_kv_blocks &&
         lhs.current_kv_block_utilization ==
             rhs.current_kv_block_utilization &&
         lhs.peak_kv_block_utilization == rhs.peak_kv_block_utilization &&
         lhs.represented_kv_tokens == rhs.represented_kv_tokens &&
         lhs.internal_fragmentation_tokens ==
             rhs.internal_fragmentation_tokens &&
         lhs.kv_allocation_failure_count == rhs.kv_allocation_failure_count &&
         lhs.kv_capacity_deferral_count == rhs.kv_capacity_deferral_count &&
         lhs.total_original_prompt_tokens == rhs.total_original_prompt_tokens &&
         lhs.total_matched_prefix_tokens == rhs.total_matched_prefix_tokens &&
         lhs.saved_simulated_prefill_tokens == rhs.saved_simulated_prefill_tokens &&
         lhs.cache_lookup_count == rhs.cache_lookup_count &&
         lhs.cache_hit_lookup_count == rhs.cache_hit_lookup_count &&
         lhs.cache_miss_lookup_count == rhs.cache_miss_lookup_count &&
         lhs.cache_matched_block_count == rhs.cache_matched_block_count &&
         lhs.total_cache_eligible_prompt_tokens_looked_up ==
             rhs.total_cache_eligible_prompt_tokens_looked_up &&
         lhs.collision_verification_count == rhs.collision_verification_count &&
         lhs.prefix_cache_eviction_count == rhs.prefix_cache_eviction_count &&
         lhs.cached_kv_blocks == rhs.cached_kv_blocks &&
         lhs.referenced_shared_kv_blocks == rhs.referenced_shared_kv_blocks;
}

bool same_plan(const BatchPlan& lhs, const BatchPlan& rhs) {
  if (lhs.iteration_number() != rhs.iteration_number() ||
      values(lhs.prefill_request_ids()) != values(rhs.prefill_request_ids()) ||
      values(lhs.decode_request_ids()) != values(rhs.decode_request_ids()) ||
      lhs.total_prefill_tokens() != rhs.total_prefill_tokens() ||
      lhs.total_decode_tokens() != rhs.total_decode_tokens() ||
      lhs.total_scheduled_tokens() != rhs.total_scheduled_tokens() ||
      lhs.prefill_work().size() != rhs.prefill_work().size() ||
      lhs.decode_work().size() != rhs.decode_work().size() ||
      lhs.work_order().size() != rhs.work_order().size() ||
      lhs.planned_eviction_ids() != rhs.planned_eviction_ids() ||
      lhs.planned_allocation_ids() != rhs.planned_allocation_ids() ||
      lhs.deferred_requests().size() != rhs.deferred_requests().size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.prefill_work().size(); ++index) {
    const auto& a = lhs.prefill_work()[index];
    const auto& b = rhs.prefill_work()[index];
    if (a.request_id != b.request_id ||
        a.prompt_token_count != b.prompt_token_count ||
        a.original_prompt_token_count != b.original_prompt_token_count ||
        a.matched_prefix_token_count != b.matched_prefix_token_count ||
        a.matched_physical_block_ids != b.matched_physical_block_ids ||
        a.newly_allocated_block_ids != b.newly_allocated_block_ids ||
        a.evicted_block_ids != b.evicted_block_ids ||
        a.prefix_lookup_kind != b.prefix_lookup_kind) return false;
  }
  for (std::size_t index = 0; index < lhs.decode_work().size(); ++index) {
    const auto& a = lhs.decode_work()[index];
    const auto& b = rhs.decode_work()[index];
    if (a.request_id != b.request_id ||
        a.newly_allocated_block_id != b.newly_allocated_block_id ||
        a.evicted_block_ids != b.evicted_block_ids) return false;
  }
  for (std::size_t index = 0; index < lhs.work_order().size(); ++index) {
    if (lhs.work_order()[index].request_id != rhs.work_order()[index].request_id ||
        lhs.work_order()[index].is_prefill != rhs.work_order()[index].is_prefill)
      return false;
  }
  for (std::size_t index = 0; index < lhs.deferred_requests().size(); ++index) {
    if (lhs.deferred_requests()[index].request_id !=
            rhs.deferred_requests()[index].request_id ||
        lhs.deferred_requests()[index].reason !=
            rhs.deferred_requests()[index].reason) {
      return false;
    }
  }
  return true;
}

bool same_trace(const std::vector<BatchTraceEntry>& lhs,
                const std::vector<BatchTraceEntry>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index].iteration_number != rhs[index].iteration_number ||
        lhs[index].start_timestamp_us != rhs[index].start_timestamp_us ||
        lhs[index].end_timestamp_us != rhs[index].end_timestamp_us ||
        lhs[index].policy != rhs[index].policy ||
        !same_plan(lhs[index].plan, rhs[index].plan) ||
        lhs[index].allocated_kv_blocks != rhs[index].allocated_kv_blocks ||
        lhs[index].free_kv_blocks != rhs[index].free_kv_blocks ||
        lhs[index].represented_kv_tokens !=
            rhs[index].represented_kv_tokens ||
        lhs[index].internal_fragmentation_tokens !=
            rhs[index].internal_fragmentation_tokens ||
        lhs[index].kv_block_utilization !=
            rhs[index].kv_block_utilization ||
        lhs[index].kv_block_tables != rhs[index].kv_block_tables) {
      return false;
    }
  }
  return true;
}

struct EngineSnapshot {
  std::map<RequestId, Request> requests;
  std::set<RequestId> arrivals;
  std::vector<BatchTraceEntry> trace;
  ContinuousBatchingStatistics statistics;
  std::int64_t clock_us;
  std::uint64_t next_iteration;
  KVCacheManager kv_cache;
};

EngineSnapshot snapshot(const ContinuousBatchingEngine& engine) {
  return EngineSnapshot{
      ContinuousBatchingEngineTestAccess::raw_requests(engine),
      ContinuousBatchingEngineTestAccess::raw_arrivals(engine),
      ContinuousBatchingEngineTestAccess::raw_trace(engine),
      ContinuousBatchingEngineTestAccess::raw_statistics(engine),
      ContinuousBatchingEngineTestAccess::raw_clock_us(engine),
      ContinuousBatchingEngineTestAccess::raw_next_iteration(engine),
      ContinuousBatchingEngineTestAccess::raw_kv_cache(engine)};
}

bool unchanged(const ContinuousBatchingEngine& engine,
               const EngineSnapshot& before) {
  const auto& requests =
      ContinuousBatchingEngineTestAccess::raw_requests(engine);
  if (requests.size() != before.requests.size()) {
    return false;
  }
  auto lhs = requests.begin();
  auto rhs = before.requests.begin();
  for (; lhs != requests.end(); ++lhs, ++rhs) {
    if (lhs->first != rhs->first || !same_request(lhs->second, rhs->second)) {
      return false;
    }
  }
  return ContinuousBatchingEngineTestAccess::raw_arrivals(engine) ==
             before.arrivals &&
         same_trace(ContinuousBatchingEngineTestAccess::raw_trace(engine),
                    before.trace) &&
         same_statistics(
             ContinuousBatchingEngineTestAccess::raw_statistics(engine),
             before.statistics) &&
         ContinuousBatchingEngineTestAccess::raw_clock_us(engine) ==
             before.clock_us &&
         ContinuousBatchingEngineTestAccess::raw_next_iteration(engine) ==
             before.next_iteration &&
         ContinuousBatchingEngineTestAccess::raw_kv_cache(engine)
                 .physical_blocks() == before.kv_cache.physical_blocks() &&
         ContinuousBatchingEngineTestAccess::raw_kv_cache(engine)
                 .free_block_ids() == before.kv_cache.free_block_ids() &&
         ContinuousBatchingEngineTestAccess::raw_kv_cache(engine)
                 .block_tables() == before.kv_cache.block_tables() &&
         ContinuousBatchingEngineTestAccess::raw_kv_cache(engine)
                 .represented_token_count() ==
             before.kv_cache.represented_token_count() &&
         ContinuousBatchingEngineTestAccess::raw_kv_cache(engine)
                 .peak_allocated_block_count() ==
             before.kv_cache.peak_allocated_block_count() &&
         ContinuousBatchingEngineTestAccess::raw_kv_cache(engine)
                 .allocation_failure_count() ==
             before.kv_cache.allocation_failure_count() &&
         ContinuousBatchingEngineTestAccess::raw_kv_cache(engine)
                 .access_epoch() == before.kv_cache.access_epoch();
}

static_assert(!std::is_default_constructible_v<BatchPlan>);
static_assert(!std::is_aggregate_v<BatchPlan>);
using EmptyPlanFactory = BatchPlan (*)(std::uint64_t);
[[maybe_unused]] constexpr EmptyPlanFactory kEmptyPlanFactory =
    static_cast<EmptyPlanFactory>(&BatchPlan::empty);

void test_configuration_and_plan_validation() {
  require_throws<std::invalid_argument>(
      [] {
        ContinuousBatchingConfig config(0, 4,
                                        SchedulingPolicy::DecodeFirst);
        (void)config;
      },
      "zero sequence budget was accepted");
  require_throws<std::invalid_argument>(
      [] {
        ContinuousBatchingConfig config(2, 0,
                                        SchedulingPolicy::DecodeFirst);
        (void)config;
      },
      "zero token budget was accepted");
  require_throws<std::invalid_argument>(
      [] {
        ContinuousBatchingConfig config(
            2, 2, static_cast<SchedulingPolicy>(999));
        (void)config;
      },
      "invalid scheduling policy was accepted");

  require_throws<std::invalid_argument>(
      [] {
        (void)BatchPlan::create(1, {{RequestId{1}, 1}, {RequestId{1}, 1}},
                                {}, 2, 2);
      },
      "duplicate prefill request was accepted");
  require_throws<std::invalid_argument>(
      [] {
        (void)BatchPlan::create(1, {{RequestId{1}, 1}}, {RequestId{1}}, 2, 2);
      },
      "request in prefill and decode was accepted");
  require_throws<std::invalid_argument>(
      [] { (void)BatchPlan::create(1, {}, {{1}, {1}}, 2, 2); },
      "duplicate decode request was accepted");
  require_throws<std::invalid_argument>(
      [] {
        (void)BatchPlan::create(1, {{RequestId{1}, 1}}, {RequestId{2}}, 1, 2);
      },
      "sequence budget violation was accepted");
  require_throws<std::invalid_argument>(
      [] {
        (void)BatchPlan::create(1, {{RequestId{1}, 2}}, {RequestId{2}}, 2, 2);
      },
      "token budget violation was accepted");
  require_throws<std::invalid_argument>(
      [] { (void)BatchPlan::create(1, {}, {}, 1, 1); },
      "implicit empty plan was accepted");
  require_throws<std::invalid_argument>(
      [] {
        (void)ContinuousBatchingEngineTestAccess::create_with_deferred(
            1, {{RequestId{1}, 1}}, {},
            {{RequestId{2}, DeferralReason::TokenBudget},
             {RequestId{2}, DeferralReason::SequenceBudget}},
            1, 1);
      },
      "duplicate deferred request was accepted");
  require_throws<std::invalid_argument>(
      [] {
        (void)ContinuousBatchingEngineTestAccess::create_with_deferred(
            1, {{RequestId{1}, 1}}, {},
            {{RequestId{1}, DeferralReason::SequenceBudget}}, 1, 1);
      },
      "scheduled/deferred overlap was accepted");
  require_throws<std::invalid_argument>(
      [] {
        (void)ContinuousBatchingEngineTestAccess::create_with_deferred(
            1, {{RequestId{1}, 1}}, {},
            {{RequestId{2}, static_cast<DeferralReason>(999)}}, 1, 1);
      },
      "invalid deferral reason was accepted");

  const BatchPlan empty = BatchPlan::empty(7);
  require(empty.empty() && empty.iteration_number() == 7 &&
              empty.deferred_requests().empty() &&
              empty.total_prefill_tokens() == 0 &&
              empty.total_decode_tokens() == 0 &&
              empty.total_scheduled_tokens() == 0,
          "explicit empty plan is invalid");
  const BatchPlan plan = BatchPlan::create(
      8, {{RequestId{4}, 3}}, {RequestId{2}}, 2, 4);
  require_ids(plan.prefill_request_ids(), {4}, "prefill order changed");
  require_ids(plan.decode_request_ids(), {2}, "decode order changed");
  require(plan.deferred_requests().empty() &&
              plan.total_prefill_tokens() == 3 &&
              plan.total_decode_tokens() == 1 &&
              plan.total_scheduled_tokens() == 4 &&
              plan.scheduled_sequence_count() == 2,
          "BatchPlan totals are wrong");

  require_throws<std::invalid_argument>(
      [] { SimulatedBackend invalid({0, 0, 0, 0, 0, -1, 0, 0, 0}); },
      "negative mixed-batch cost was accepted");
  const auto backend = batch_backend();
  require_throws<std::invalid_argument>(
      [&] { (void)backend.estimate_batch_time_us(0, 0, 0); },
      "zero-sequence mixed batch was accepted");
  require_throws<std::invalid_argument>(
      [&] { (void)backend.estimate_batch_time_us(0, 2, 1); },
      "invalid mixed-batch decode count was accepted");
}

void test_oversized_prompt_rejected_synchronously() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(2, 3, SchedulingPolicy::DecodeFirst));
  require_throws<std::invalid_argument>(
      [&] { engine.submit_request(Request({1}, 0, 4, 1)); },
      "oversized prompt was accepted");
  require(engine.requests().empty() && engine.plan_trace().empty() &&
              engine.clock().now_us() == 0,
          "oversized prompt rejection mutated the engine");
}

void set_cancelled_count_to_maximum(ContinuousBatchingEngine& engine) {
  ContinuousBatchingStatistics statistics =
      ContinuousBatchingEngineTestAccess::raw_statistics(engine);
  statistics.cancelled_request_count =
      std::numeric_limits<std::uint64_t>::max();
  ContinuousBatchingEngineTestAccess::set_statistics(engine, statistics);
}

void require_cancellation_overflow_is_strong(ContinuousBatchingEngine& engine,
                                             RequestId request_id,
                                             const std::string& scenario) {
  set_cancelled_count_to_maximum(engine);
  const EngineSnapshot before = snapshot(engine);
  require_throws<std::overflow_error>(
      [&] { engine.cancel_request(request_id); },
      scenario + " cancellation did not overflow");
  require(!engine.failed(), scenario + " cancellation failed the engine");
  require(unchanged(engine, before),
          scenario + " cancellation overflow mutated engine state");
  require(engine.request(request_id).state ==
              before.requests.at(request_id).state,
          scenario + " cancellation overflow made results unavailable");
}

void test_cancellation_overflow_strong_guarantee() {
  const auto backend = batch_backend();

  ContinuousBatchingEngine prearrival(
      backend, ContinuousBatchingConfig(1, 2, SchedulingPolicy::DecodeFirst));
  prearrival.submit_request(Request({1}, 5, 1, 1));
  require_cancellation_overflow_is_strong(prearrival, {1}, "pre-arrival");

  ContinuousBatchingEngine waiting(
      backend, ContinuousBatchingConfig(1, 2, SchedulingPolicy::DecodeFirst));
  waiting.submit_request(Request({1}, 0, 1, 2));
  waiting.submit_request(Request({2}, 0, 1, 1));
  require(waiting.run_next_iteration(), "waiting setup did not prefill");
  require_cancellation_overflow_is_strong(waiting, {2}, "waiting");

  ContinuousBatchingEngine active(
      backend, ContinuousBatchingConfig(1, 2, SchedulingPolicy::DecodeFirst));
  active.submit_request(Request({1}, 0, 1, 2));
  require(active.run_next_iteration(), "active setup did not prefill");
  require_cancellation_overflow_is_strong(active, {1}, "active decode");

  ContinuousBatchingEngine usable(
      backend, ContinuousBatchingConfig(1, 2, SchedulingPolicy::DecodeFirst));
  usable.submit_request(Request({9}, 0, 1, 1));
  usable.cancel_request({9});
  usable.run();
  require(usable.request({9}).state == RequestState::Cancelled &&
              usable.statistics().cancelled_request_count == 1,
          "normal cancellation failed after overflow scenarios");
}

ContinuousBatchingEngine make_decode_first_engine(
    const SimulatedBackend& backend) {
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(2, 4, SchedulingPolicy::DecodeFirst));
  engine.submit_request(Request({2}, 0, 2, 2));
  engine.submit_request(Request({1}, 0, 2, 3));
  engine.submit_request(Request({3}, 5, 1, 1));
  return engine;
}

void check_decode_first_trace(const std::vector<BatchTraceEntry>& trace) {
  require(trace.size() == 5, "DecodeFirst trace length is wrong");
  const std::int64_t starts[] = {0, 8, 16, 24, 31};
  const std::int64_t ends[] = {8, 16, 24, 31, 36};
  const std::vector<std::vector<std::uint64_t>> prefills = {
      {1, 2}, {}, {}, {3}, {}};
  const std::vector<std::vector<std::uint64_t>> decodes = {
      {}, {1, 2}, {1, 2}, {1}, {3}};
  const std::uint64_t tokens[] = {4, 2, 2, 2, 1};
  for (std::size_t index = 0; index < trace.size(); ++index) {
    const auto& entry = trace[index];
    require(entry.iteration_number == index + 1 &&
                entry.start_timestamp_us == starts[index] &&
                entry.end_timestamp_us == ends[index] &&
                entry.policy == SchedulingPolicy::DecodeFirst,
            "DecodeFirst trace metadata differs");
    require_ids(entry.plan.prefill_request_ids(), prefills[index],
                "DecodeFirst prefill trace differs");
    require_ids(entry.plan.decode_request_ids(), decodes[index],
                "DecodeFirst decode trace differs");
    require(entry.plan.total_scheduled_tokens() == tokens[index],
            "DecodeFirst token trace differs");
  }
  require(trace[1].plan.deferred_requests().size() == 1 &&
              trace[1].plan.deferred_requests()[0].request_id == RequestId{3} &&
              trace[1].plan.deferred_requests()[0].reason ==
                  DeferralReason::SequenceBudget,
          "DecodeFirst did not defer prefill behind active decode");
}

void test_decode_first_lifecycle_trace_and_statistics() {
  const auto backend = batch_backend();
  auto engine = make_decode_first_engine(backend);
  engine.run();
  check_decode_first_trace(engine.plan_trace());

  const Request& first = engine.request({1});
  const Request& second = engine.request({2});
  const Request& arrival_during_decode = engine.request({3});
  require(first.state == RequestState::Finished &&
              first.generated_token_count == 3 &&
              first.first_token_time_us.has_value() &&
              *first.first_token_time_us == 16 &&
              first.finish_time_us.has_value() && *first.finish_time_us == 31,
          "long decode lifecycle is wrong");
  require(second.generated_token_count == 2 &&
              second.finish_time_us.has_value() &&
              *second.finish_time_us == 24,
          "second active decode lifecycle is wrong");
  require(arrival_during_decode.admitted_time_us.has_value() &&
              *arrival_during_decode.admitted_time_us == 24 &&
              arrival_during_decode.first_token_time_us.has_value() &&
              *arrival_during_decode.first_token_time_us == 36,
          "arrival during decode did not join later iterations");

  std::size_t first_prefills = 0;
  std::size_t first_decodes = 0;
  for (const auto& entry : engine.plan_trace()) {
    for (RequestId id : entry.plan.prefill_request_ids()) {
      first_prefills += id == RequestId{1} ? 1 : 0;
    }
    for (RequestId id : entry.plan.decode_request_ids()) {
      first_decodes += id == RequestId{1} ? 1 : 0;
    }
  }
  require(first_prefills == 1 && first_decodes == 3,
          "prefill/decode work cardinality is wrong");

  const auto& stats = engine.statistics();
  require(stats.scheduling_iteration_count == 5 &&
              stats.nonempty_batch_count == 5 &&
              stats.idle_iteration_count == 0 &&
              stats.total_prefill_tokens_scheduled == 5 &&
              stats.total_decode_tokens_scheduled == 6 &&
              stats.total_scheduled_sequences == 9 &&
              stats.maximum_batch_size == 2 &&
              stats.maximum_scheduled_tokens == 4 &&
              stats.deferred_request_count == 2 &&
              stats.completed_request_count == 3 &&
              stats.cancelled_request_count == 0,
          "DecodeFirst statistics are wrong");
  const auto average = stats.average_batch_size();
  require(average.has_value() && std::abs(*average - 1.8) < 1e-12,
          "average batch size is wrong");
}

void test_decode_first_token_budget_deferral() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(3, 2, SchedulingPolicy::DecodeFirst));
  engine.submit_request(Request({1}, 0, 1, 2));
  engine.submit_request(Request({2}, 0, 1, 2));
  require(engine.run_next_iteration(), "prefill iteration missing");
  engine.submit_request(Request({3}, engine.clock().now_us(), 1, 0));
  require(engine.run_next_iteration(), "decode iteration missing");
  const auto& deferred = engine.plan_trace().back().plan.deferred_requests();
  require(deferred.size() == 1 && deferred[0].request_id == RequestId{3} &&
              deferred[0].reason == DeferralReason::TokenBudget,
          "decode token use did not defer prefill by token budget");
  engine.run();
  require(engine.request({3}).state == RequestState::Finished,
          "finite DecodeFirst workload starved a prefill");
}

void test_fcfs_mixed_order_and_trace() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(2, 3, SchedulingPolicy::FcfsMixed));
  engine.submit_request(Request({3}, 0, 1, 1));
  engine.submit_request(Request({2}, 0, 3, 0));
  engine.submit_request(Request({1}, 0, 1, 0));
  engine.run();

  const auto& trace = engine.plan_trace();
  require(trace.size() == 3, "FcfsMixed trace length is wrong");
  require_ids(trace[0].plan.prefill_request_ids(), {1, 3},
              "same-time request ID tie-break failed");
  require_ids(trace[1].plan.prefill_request_ids(), {2},
              "global FCFS did not select older waiting prefill");
  require_ids(trace[1].plan.decode_request_ids(), {},
              "later decode bypassed earlier prefill");
  require(trace[1].plan.deferred_requests().size() == 1 &&
              trace[1].plan.deferred_requests()[0].request_id == RequestId{3} &&
              trace[1].plan.deferred_requests()[0].reason ==
                  DeferralReason::TokenBudget,
          "FcfsMixed prefill did not delay decode");
  require_ids(trace[2].plan.decode_request_ids(), {3},
              "deferred decode did not resume");
  require(trace[0].start_timestamp_us == 0 &&
              trace[0].end_timestamp_us == 6 &&
              trace[1].start_timestamp_us == 6 &&
              trace[1].end_timestamp_us == 12 &&
              trace[2].start_timestamp_us == 12 &&
              trace[2].end_timestamp_us == 17,
          "FcfsMixed exact timing trace differs");
}

void test_deferral_reason_precedence() {
  const auto backend = SimulatedBackend({0, 0, 0, 0, 0, 0, 0, 0, 0});
  const auto reason_for_second = [&](std::size_t sequences,
                                     std::uint64_t tokens) {
    ContinuousBatchingEngine engine(
        backend, ContinuousBatchingConfig(
                     sequences, tokens, SchedulingPolicy::FcfsMixed));
    engine.submit_request(Request({2}, 0, 1, 0));
    engine.submit_request(Request({1}, 0, 1, 0));
    require(engine.run_next_iteration(), "deferral scenario did not run");
    const auto& deferred = engine.plan_trace().back().plan.deferred_requests();
    require(deferred.size() == 1 &&
                deferred[0].request_id == RequestId{2},
            "deferral scenario selected the wrong request");
    return deferred[0].reason;
  };
  require(reason_for_second(1, 2) == DeferralReason::SequenceBudget,
          "sequence-only exhaustion has the wrong reason");
  require(reason_for_second(2, 1) == DeferralReason::TokenBudget,
          "token-only exhaustion has the wrong reason");
  require(reason_for_second(1, 1) == DeferralReason::TokenBudget,
          "both-budget exhaustion did not use token precedence");

  const auto kv_reason_for_second = [&](std::size_t sequences,
                                        std::uint64_t tokens) {
    ContinuousBatchingEngine engine(
        backend, ContinuousBatchingConfig(
                     sequences, tokens, SchedulingPolicy::FcfsMixed,
                     KVCacheConfig(1, 1)));
    engine.submit_request(Request({1}, 0, 1, 0));
    engine.submit_request(Request({2}, 0, 1, 0));
    require(engine.run_next_iteration().made_progress(),
            "combined KV precedence scenario did not run");
    const auto& deferred = engine.plan_trace().back().plan.deferred_requests();
    require(deferred.size() == 1 && deferred[0].request_id == RequestId{2},
            "combined KV precedence selected the wrong request");
    return deferred[0].reason;
  };
  require(kv_reason_for_second(2, 1) == DeferralReason::TokenBudget,
          "token and KV exhaustion did not use token precedence");
  require(kv_reason_for_second(1, 2) == DeferralReason::SequenceBudget,
          "sequence and KV exhaustion did not use sequence precedence");
  require(kv_reason_for_second(2, 2) == DeferralReason::KVCapacity,
          "KV-only exhaustion did not use KVCapacity");
}

void test_active_decode_deferral_reasons() {
  const auto backend = SimulatedBackend({0, 0, 0, 0, 0, 0, 0, 0, 0});
  const auto run_scenario = [&](std::size_t sequences,
                                std::uint64_t tokens) {
    ContinuousBatchingEngine engine(
        backend, ContinuousBatchingConfig(
                     sequences, tokens, SchedulingPolicy::FcfsMixed));
    engine.submit_request(Request({10}, 0, 0, 2));
    require(engine.run_next_iteration(), "active decode setup failed");
    engine.submit_request(Request({5}, 0, 1, 0));
    require(engine.run_next_iteration(), "active decode deferral failed");
    const auto& plan = engine.plan_trace().back().plan;
    require_ids(plan.prefill_request_ids(), {5},
                "older prefill was not selected");
    require(plan.deferred_requests().size() == 1 &&
                plan.deferred_requests()[0].request_id == RequestId{10},
            "active decode was not deferred");
    return plan.deferred_requests()[0].reason;
  };
  require(run_scenario(1, 2) == DeferralReason::SequenceBudget,
          "active decode sequence deferral reason is wrong");
  require(run_scenario(2, 1) == DeferralReason::TokenBudget,
          "active decode token deferral reason is wrong");
}

void test_zero_prompt_progress() {
  const auto backend = SimulatedBackend({0, 0, 0, 0, 0, 0, 0, 0, 0});
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(1, 1, SchedulingPolicy::DecodeFirst));
  engine.submit_request(Request({1}, 0, 0, 1));
  engine.submit_request(Request({2}, 0, 0, 0));
  engine.run();
  require(engine.plan_trace().size() == 3 &&
              engine.plan_trace()[0].plan.scheduled_sequence_count() == 1 &&
              engine.plan_trace()[0].plan.total_prefill_tokens() == 0 &&
              engine.clock().now_us() == 0,
          "zero-token prefill did not make zero-duration progress");
  require(engine.request({1}).state == RequestState::Finished &&
              engine.request({1}).generated_token_count == 1 &&
              engine.request({2}).state == RequestState::Finished &&
              !engine.request({2}).first_token_time_us.has_value() &&
              engine.statistics().completed_request_count == 2,
          "zero-prompt lifecycle is wrong");
}

void test_arrival_exactly_at_iteration_end() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(3, 4, SchedulingPolicy::DecodeFirst));
  engine.submit_request(Request({10}, 0, 1, 2));
  engine.submit_request(Request({3}, 4, 1, 0));
  engine.submit_request(Request({2}, 4, 1, 0));
  require(engine.run_next_iteration(), "initial iteration missing");
  require(engine.clock().now_us() == 4,
          "arrival boundary fixture has the wrong end time");
  require(engine.run_next_iteration(), "boundary arrival iteration missing");
  const auto& plan = engine.plan_trace().back().plan;
  require_ids(plan.decode_request_ids(), {10},
              "active decode missing at arrival boundary");
  require_ids(plan.prefill_request_ids(), {2, 3},
              "same-time boundary arrivals ignored request-ID order");
}

void test_policy_difference_on_same_workload() {
  const auto backend = batch_backend();
  const auto make_engine = [&](SchedulingPolicy policy) {
    ContinuousBatchingEngine engine(
        backend, ContinuousBatchingConfig(2, 3, policy));
    engine.submit_request(Request({3}, 0, 1, 1));
    engine.submit_request(Request({2}, 0, 3, 0));
    engine.submit_request(Request({1}, 0, 1, 0));
    return engine;
  };
  auto decode_first = make_engine(SchedulingPolicy::DecodeFirst);
  auto fcfs_mixed = make_engine(SchedulingPolicy::FcfsMixed);
  require(decode_first.run_next_iteration() &&
              fcfs_mixed.run_next_iteration(),
          "policy comparison setup failed");
  require(decode_first.run_next_iteration() &&
              fcfs_mixed.run_next_iteration(),
          "policy comparison second iteration failed");
  require_ids(decode_first.plan_trace()[1].plan.decode_request_ids(), {3},
              "DecodeFirst did not protect active decode");
  require_ids(decode_first.plan_trace()[1].plan.prefill_request_ids(), {},
              "DecodeFirst unexpectedly selected the older large prefill");
  require_ids(fcfs_mixed.plan_trace()[1].plan.prefill_request_ids(), {2},
              "FcfsMixed did not select the older fitting prefill");
  require_ids(fcfs_mixed.plan_trace()[1].plan.decode_request_ids(), {},
              "FcfsMixed did not defer the newer decode");
  decode_first.run();
  fcfs_mixed.run();
  for (RequestId id : {RequestId{1}, RequestId{2}, RequestId{3}}) {
    require(decode_first.request(id).state == RequestState::Finished &&
                fcfs_mixed.request(id).state == RequestState::Finished,
            "policy comparison workload did not drain");
  }
}

void test_per_iteration_sequence_limit_is_not_residency() {
  const auto backend = SimulatedBackend({0, 0, 0, 0, 0, 0, 0, 0, 0});
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(1, 1, SchedulingPolicy::FcfsMixed,
                                       KVCacheConfig(3, 2)));
  engine.submit_request(Request({10}, 0, 1, 2));
  require(engine.run_next_iteration(), "first resident decode setup failed");
  engine.submit_request(Request({5}, 0, 1, 1));
  require(engine.run_next_iteration(), "second resident decode setup failed");
  require(engine.request({5}).state == RequestState::Decoding &&
              engine.request({10}).state == RequestState::Decoding &&
              !engine.kv_cache().block_table({5}).empty() &&
              !engine.kv_cache().block_table({10}).empty() &&
              engine.kv_cache().allocated_block_count() == 2,
          "per-iteration limit was treated as resident capacity");
  for (const auto& entry : engine.plan_trace()) {
    require(entry.plan.scheduled_sequence_count() <= 1,
            "plan exceeded per-iteration sequence limit");
  }
  engine.run();
  require(engine.request({5}).state == RequestState::Finished &&
              engine.request({10}).state == RequestState::Finished,
          "deferred resident decode did not finish later");
}

void test_zero_output_idle_and_arrival_boundary() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(1, 2, SchedulingPolicy::DecodeFirst));
  engine.submit_request(Request({1}, 10, 2, 0));
  engine.run();
  const auto& trace = engine.plan_trace();
  require(trace.size() == 2 && trace[0].plan.empty() &&
              trace[0].start_timestamp_us == 0 &&
              trace[0].end_timestamp_us == 10 &&
              !trace[1].plan.empty(),
          "future arrival did not create an explicit idle plan");
  const Request& request = engine.request({1});
  require(request.state == RequestState::Finished &&
              request.admitted_time_us.has_value() &&
              *request.admitted_time_us == 10 &&
              !request.first_token_time_us.has_value() &&
              request.generated_token_count == 0,
          "zero-output request lifecycle is wrong");
  const auto& stats = engine.statistics();
  require(stats.scheduling_iteration_count == 2 &&
              stats.nonempty_batch_count == 1 &&
              stats.idle_iteration_count == 1,
          "idle statistics are wrong");

  ContinuousBatchingEngine empty(
      backend, ContinuousBatchingConfig(1, 2, SchedulingPolicy::DecodeFirst));
  empty.run();
  require(!empty.statistics().average_batch_size().has_value(),
          "empty average batch size was silently reported as zero");
}

void test_cancellation_stops_work() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine engine(
      backend, ContinuousBatchingConfig(2, 2, SchedulingPolicy::DecodeFirst));
  engine.submit_request(Request({1}, 0, 1, 3));
  engine.submit_request(Request({2}, 0, 1, 3));
  require(engine.run_next_iteration(), "initial prefill missing");
  require(engine.run_next_iteration(), "initial decode missing");
  engine.cancel_request({1});
  engine.run();
  const Request& cancelled = engine.request({1});
  require(cancelled.state == RequestState::Cancelled &&
              cancelled.generated_token_count == 1,
          "active cancellation generated more tokens");
  for (std::size_t index = 2; index < engine.plan_trace().size(); ++index) {
    require_ids(engine.plan_trace()[index].plan.decode_request_ids(), {2},
                "cancelled request re-entered a plan");
  }
  require(engine.statistics().cancelled_request_count == 1 &&
              engine.statistics().completed_request_count == 1,
          "cancellation/completion statistics are wrong");

  ContinuousBatchingEngine waiting(
      backend, ContinuousBatchingConfig(1, 2, SchedulingPolicy::DecodeFirst));
  waiting.submit_request(Request({9}, 5, 1, 1));
  waiting.cancel_request({9});
  waiting.run();
  require(waiting.plan_trace().empty() &&
              waiting.request({9}).state == RequestState::Cancelled,
          "pre-arrival cancellation created runnable work");

  ContinuousBatchingEngine queued(
      backend, ContinuousBatchingConfig(1, 1, SchedulingPolicy::DecodeFirst));
  queued.submit_request(Request({1}, 0, 1, 1));
  queued.submit_request(Request({2}, 0, 1, 1));
  require(queued.run_next_iteration(), "queued cancellation prefill missing");
  queued.cancel_request({2});
  queued.run();
  require(queued.request({2}).state == RequestState::Cancelled &&
              queued.request({2}).generated_token_count == 0 &&
              queued.statistics().cancelled_request_count == 1,
          "waiting cancellation did not remove queued work");
}

void test_determinism_and_checked_overflow() {
  const auto backend = batch_backend();
  auto lhs = make_decode_first_engine(backend);
  auto rhs = make_decode_first_engine(backend);
  lhs.run();
  rhs.run();
  require(same_trace(lhs.plan_trace(), rhs.plan_trace()),
          "complete deterministic traces differ");
  require(same_statistics(lhs.statistics(), rhs.statistics()),
          "deterministic final statistics differ");
  require(lhs.requests().size() == rhs.requests().size(),
          "deterministic final request counts differ");
  for (const auto& entry : lhs.requests()) {
    require(same_request(entry.second, rhs.request(entry.first)),
            "deterministic final request records differ");
  }

  const auto maximum = std::numeric_limits<std::int64_t>::max();
  const SimulatedBackend overflowing(
      {0, 0, 0, 0, 0, 0, maximum, 0, 0});
  ContinuousBatchingEngine cost_overflow(
      overflowing,
      ContinuousBatchingConfig(1, 2, SchedulingPolicy::DecodeFirst));
  cost_overflow.submit_request(Request({1}, 0, 2, 0));
  require_throws<std::overflow_error>([&] { cost_overflow.run(); },
                                      "mixed batch cost overflow was missed");
  require(cost_overflow.failed(), "cost overflow was not terminal");

  const SimulatedBackend unit_cost({0, 0, 0, 0, 0, 1, 0, 0, 0});
  ContinuousBatchingEngine timestamp_overflow(
      unit_cost,
      ContinuousBatchingConfig(1, 1, SchedulingPolicy::DecodeFirst));
  timestamp_overflow.submit_request(Request({1}, maximum, 1, 0));
  require(timestamp_overflow.run_next_iteration(), "idle jump missing");
  require_throws<std::overflow_error>(
      [&] { timestamp_overflow.run_next_iteration(); },
      "batch timestamp overflow was missed");
}

void require_terminal_failure_contract(ContinuousBatchingEngine& engine,
                                       RequestId existing_request) {
  require(engine.failed(), "failing iteration did not mark engine failed");
  require_throws<std::logic_error>([&] { engine.run(); },
                                   "run succeeded after terminal failure");
  require_throws<std::logic_error>([&] { engine.run_next_iteration(); },
                                   "iteration succeeded after terminal failure");
  require_throws<std::logic_error>(
      [&] { engine.submit_request(Request({999}, 0, 0, 0)); },
      "submission succeeded after terminal failure");
  require_throws<std::logic_error>(
      [&] { engine.cancel_request(existing_request); },
      "cancellation succeeded after terminal failure");
  require_throws<std::logic_error>([&] { (void)engine.request(existing_request); },
                                   "request result survived terminal failure");
  require_throws<std::logic_error>([&] { (void)engine.requests(); },
                                   "request table survived terminal failure");
  require_throws<std::logic_error>([&] { (void)engine.plan_trace(); },
                                   "trace survived terminal failure");
  require_throws<std::logic_error>([&] { (void)engine.statistics(); },
                                   "statistics survived terminal failure");
  require_throws<std::logic_error>([&] { (void)engine.clock(); },
                                   "clock survived terminal failure");
  require_throws<std::logic_error>([&] { (void)engine.kv_cache(); },
                                   "KV diagnostics survived terminal failure");
}

void require_iteration_failure_is_transactional(
    ContinuousBatchingEngine& engine, RequestId existing_request,
    const std::string& scenario) {
  const EngineSnapshot before = snapshot(engine);
  require_throws<std::exception>([&] { engine.run_next_iteration(); },
                                 scenario + " did not fail");
  require(unchanged(engine, before),
          scenario + " partially committed the failing iteration");
  require_terminal_failure_contract(engine, existing_request);
}

void test_iteration_two_phase_failure_guarantee() {
  const auto zero_backend = SimulatedBackend({0, 0, 0, 0, 0, 0, 0, 0, 0});
  const auto make_single = [&]() {
    ContinuousBatchingEngine engine(
        zero_backend,
        ContinuousBatchingConfig(1, 1, SchedulingPolicy::DecodeFirst));
    engine.submit_request(Request({1}, 0, 0, 0));
    return engine;
  };

  {
    auto engine = make_single();
    auto stats = ContinuousBatchingEngineTestAccess::raw_statistics(engine);
    stats.completed_request_count = std::numeric_limits<std::uint64_t>::max();
    ContinuousBatchingEngineTestAccess::set_statistics(engine, stats);
    require_iteration_failure_is_transactional(engine, {1},
                                                "completion overflow");
  }
  {
    auto engine = make_single();
    auto stats = ContinuousBatchingEngineTestAccess::raw_statistics(engine);
    stats.scheduling_iteration_count =
        std::numeric_limits<std::uint64_t>::max();
    ContinuousBatchingEngineTestAccess::set_statistics(engine, stats);
    require_iteration_failure_is_transactional(engine, {1},
                                                "iteration count overflow");
  }
  {
    auto engine = make_single();
    auto stats = ContinuousBatchingEngineTestAccess::raw_statistics(engine);
    stats.total_scheduled_sequences =
        std::numeric_limits<std::uint64_t>::max();
    ContinuousBatchingEngineTestAccess::set_statistics(engine, stats);
    require_iteration_failure_is_transactional(
        engine, {1}, "scheduled sequence overflow");
  }
  {
    ContinuousBatchingEngine engine(
        zero_backend,
        ContinuousBatchingConfig(2, 2, SchedulingPolicy::DecodeFirst,
                                 KVCacheConfig(1, 1)));
    engine.submit_request(Request({1}, 0, 1, 0));
    engine.submit_request(Request({2}, 0, 1, 0));
    auto stats = ContinuousBatchingEngineTestAccess::raw_statistics(engine);
    stats.deferred_request_count = std::numeric_limits<std::uint64_t>::max();
    ContinuousBatchingEngineTestAccess::set_statistics(engine, stats);
    require_iteration_failure_is_transactional(engine, {1},
                                                "deferred count overflow");
  }
  {
    ContinuousBatchingEngine engine(
        zero_backend,
        ContinuousBatchingConfig(2, 2, SchedulingPolicy::DecodeFirst,
                                 KVCacheConfig(1, 1)));
    engine.submit_request(Request({1}, 0, 1, 0));
    engine.submit_request(Request({2}, 0, 1, 0));
    auto stats = ContinuousBatchingEngineTestAccess::raw_statistics(engine);
    stats.kv_capacity_deferral_count =
        std::numeric_limits<std::uint64_t>::max();
    ContinuousBatchingEngineTestAccess::set_statistics(engine, stats);
    require_iteration_failure_is_transactional(engine, {1},
                                                "KV deferral count overflow");
  }
  {
    auto engine = make_single();
    ContinuousBatchingEngineTestAccess::set_next_iteration(
        engine, std::numeric_limits<std::uint64_t>::max());
    require_iteration_failure_is_transactional(engine, {1},
                                                "iteration number overflow");
  }
  {
    auto engine = make_single();
    ContinuousBatchingEngineTestAccess::set_trace_failure(engine, true);
    require_iteration_failure_is_transactional(engine, {1},
                                                "trace preparation failure");
  }
  {
    const auto unit_backend =
        SimulatedBackend({0, 0, 0, 0, 0, 1, 0, 0, 0});
    ContinuousBatchingEngine engine(
        unit_backend,
        ContinuousBatchingConfig(1, 1, SchedulingPolicy::DecodeFirst,
                                 KVCacheConfig(1, 1)));
    engine.submit_request(Request(
        {1}, std::numeric_limits<std::int64_t>::max(), 1, 0));
    require(engine.run_next_iteration(), "timestamp setup idle jump failed");
    require_iteration_failure_is_transactional(engine, {1},
                                                "timestamp overflow");
  }
  {
    ContinuousBatchingEngine engine(
        zero_backend,
        ContinuousBatchingConfig(1, 2, SchedulingPolicy::DecodeFirst,
                                 KVCacheConfig(2, 2)));
    engine.submit_request(Request({1}, 0, 1, 2));
    require(engine.run_next_iteration(), "resident KV setup failed");
    require(engine.kv_cache().allocated_block_count() == 1 &&
                !engine.kv_cache().block_table({1}).empty(),
            "resident KV failure fixture has no material allocation");
    auto stats = ContinuousBatchingEngineTestAccess::raw_statistics(engine);
    stats.scheduling_iteration_count =
        std::numeric_limits<std::uint64_t>::max();
    ContinuousBatchingEngineTestAccess::set_statistics(engine, stats);
    require_iteration_failure_is_transactional(engine, {1},
                                                "resident KV statistics overflow");
  }
}

ContinuousBatchingConfig kv_config(std::size_t sequences,
                                   std::uint64_t tokens,
                                   std::size_t blocks,
                                   std::uint64_t block_size) {
  return ContinuousBatchingConfig(sequences, tokens,
                                  SchedulingPolicy::DecodeFirst,
                                  KVCacheConfig(blocks, block_size));
}

void test_deferred_only_kv_stalls_are_nonterminal() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine prompt(backend, kv_config(1, 4, 1, 2));
  prompt.submit_request(Request({1}, 0, 3, 1));
  const auto prompt_before = snapshot(prompt);
  const IterationResult prompt_result = prompt.run_next_iteration();
  require(prompt_result.is_stalled() && !prompt.failed() &&
              prompt.clock().now_us() == prompt_before.clock_us &&
              same_request(prompt.request({1}), prompt_before.requests.at({1})) &&
              prompt.kv_cache().physical_blocks() ==
                  prompt_before.kv_cache.physical_blocks() &&
              prompt.kv_cache().free_block_ids() ==
                  prompt_before.kv_cache.free_block_ids(),
          "isolated prompt KV stall mutated work or failed the engine");
  const auto& prompt_trace = prompt.plan_trace();
  require(prompt_trace.size() == 1 && prompt_trace[0].plan.deferred_only() &&
              prompt_trace[0].start_timestamp_us == 0 &&
              prompt_trace[0].end_timestamp_us == 0 &&
              prompt_trace[0].plan.deferred_requests().size() == 1 &&
              prompt_trace[0].plan.deferred_requests()[0].request_id ==
                  RequestId{1} &&
              prompt_trace[0].plan.deferred_requests()[0].reason ==
                  DeferralReason::KVCapacity &&
              prompt.statistics().scheduling_iteration_count == 1 &&
              prompt.statistics().stalled_iteration_count == 1 &&
              prompt.statistics().nonempty_batch_count == 0 &&
              prompt.statistics().idle_iteration_count == 0 &&
              prompt.statistics().kv_capacity_deferral_count == 1 &&
              prompt.statistics().kv_allocation_failure_count == 0,
          "isolated prompt stall trace/statistics are wrong");

  ContinuousBatchingEngine permanent(backend, kv_config(1, 4, 1, 2));
  permanent.submit_request(Request({1}, 0, 3, 1));
  require(permanent.run() == RunResult::Stalled && !permanent.failed() &&
              permanent.plan_trace().size() == 1 &&
              same_trace(prompt.plan_trace(), permanent.plan_trace()) &&
              same_statistics(prompt.statistics(), permanent.statistics()),
          "run() did not return a deterministic nonterminal stall");

  ContinuousBatchingEngine decode(backend, kv_config(1, 2, 1, 2));
  decode.submit_request(Request({1}, 0, 2, 1));
  require(decode.run_next_iteration().made_progress(),
          "boundary stall setup prefill failed");
  const auto decode_before = snapshot(decode);
  const IterationResult decode_result = decode.run_next_iteration();
  require(decode_result.is_stalled() && !decode.failed() &&
              decode.request({1}).generated_token_count == 0 &&
              decode.clock().now_us() == decode_before.clock_us &&
              decode.kv_cache().physical_blocks() ==
                  decode_before.kv_cache.physical_blocks() &&
              decode.kv_cache().block_tables() ==
                  decode_before.kv_cache.block_tables(),
          "isolated boundary decode stall mutated request or KV state");
  decode.cancel_request({1});
  require(decode.kv_cache().allocated_block_count() == 0 && !decode.failed(),
          "cancellation after stall did not release resident blocks");
  decode.submit_request(Request({2}, decode.clock().now_us(), 2, 0));
  require(decode.run() == RunResult::Completed &&
              decode.request({2}).state == RequestState::Finished,
          "engine was not usable after cancelling a stalled decode");
}

void test_same_plan_completion_cannot_donate_blocks() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine prefill(backend, kv_config(2, 4, 2, 2));
  prefill.submit_request(Request({1}, 0, 2, 0));
  prefill.submit_request(Request({2}, 0, 2, 1));
  require(prefill.run_next_iteration().made_progress(),
          "same-plan prefill fixture did not run");
  const auto& first = prefill.plan_trace().back();
  require(first.kv_block_tables.count({1}) == 0 &&
              first.kv_block_tables.at({2}) ==
                  std::vector<std::size_t>({1}) &&
              first.free_kv_blocks == 1 &&
              prefill.kv_cache().free_block_ids().count(0) == 1,
          "zero-output prefill donated block 0 in the same plan");
  require(prefill.run_next_iteration().made_progress(),
          "following decode did not use released capacity");
  require(prefill.request({2}).state == RequestState::Finished &&
              prefill.kv_cache().allocated_block_count() == 0,
          "following iteration did not consume/release eligible block 0");

  ContinuousBatchingEngine decode(backend, kv_config(3, 5, 3, 2));
  decode.submit_request(Request({1}, 0, 1, 1));
  decode.submit_request(Request({2}, 0, 2, 4));
  decode.submit_request(Request({3}, 0, 2, 2));
  require(decode.run_next_iteration().made_progress(),
          "same-plan decode fixture prefill failed");
  decode.cancel_request({2});
  require(decode.kv_cache().free_block_ids() == std::set<std::size_t>({1}),
          "decode fixture did not expose block 1 at iteration start");
  require(decode.run_next_iteration().made_progress(),
          "same-plan decode iteration failed");
  const auto& decode_trace = decode.plan_trace().back();
  require(decode.request({1}).state == RequestState::Finished &&
              decode.request({3}).state == RequestState::Decoding &&
              decode_trace.kv_block_tables.at({3}) ==
                  std::vector<std::size_t>({2, 1}) &&
              decode.kv_cache().free_block_ids().count(0) == 1,
          "completing decode donated block 0 to boundary decode");
  decode.submit_request(Request({4}, decode.clock().now_us(), 1, 1));
  require(decode.run_next_iteration().made_progress(),
          "following plan did not run after decode release");
  require(decode.plan_trace().back().plan.prefill_request_ids() ==
              std::vector<RequestId>({RequestId{4}}) &&
              decode.plan_trace().back().kv_block_tables.at({4}) ==
                  std::vector<std::size_t>({0}),
          "released lowest ID was not eligible in the following plan");
}

void test_kv_capacity_lifecycle_and_determinism() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine engine(backend, kv_config(2, 8, 3, 2));
  engine.submit_request(Request({1}, 0, 4, 2));
  engine.submit_request(Request({2}, 0, 4, 1));
  engine.run();

  const auto& trace = engine.plan_trace();
  require(trace.size() == 5, "KV capacity scenario trace length is wrong");
  require_ids(trace[0].plan.prefill_request_ids(), {1},
              "request A prompt was not admitted");
  require(trace[0].plan.deferred_requests().size() == 1 &&
              trace[0].plan.deferred_requests()[0].request_id == RequestId{2} &&
              trace[0].plan.deferred_requests()[0].reason ==
                  DeferralReason::KVCapacity &&
              trace[0].allocated_kv_blocks == 2 &&
              trace[0].represented_kv_tokens == 4,
          "request B was not deferred for KV capacity");
  require(trace[1].allocated_kv_blocks == 3 &&
              trace[1].represented_kv_tokens == 5 &&
              trace[1].internal_fragmentation_tokens == 1,
          "decode boundary did not allocate exactly one block");
  require(trace[2].allocated_kv_blocks == 0 &&
              trace[2].free_kv_blocks == 3,
          "completion did not release before the next plan");
  require_ids(trace[3].plan.prefill_request_ids(), {2},
              "deferred request was not admitted after release");
  require(engine.request({1}).generated_token_count == 2 &&
              engine.request({2}).generated_token_count == 1 &&
              engine.kv_cache().allocated_block_count() == 0 &&
              engine.statistics().current_allocated_kv_blocks == 0 &&
              engine.statistics().peak_allocated_kv_blocks == 3 &&
              engine.statistics().current_kv_block_utilization == 0.0 &&
              engine.statistics().peak_kv_block_utilization == 1.0 &&
              engine.statistics().represented_kv_tokens == 0 &&
              engine.statistics().internal_fragmentation_tokens == 0 &&
              engine.statistics().kv_capacity_deferral_count == 3 &&
              engine.statistics().kv_allocation_failure_count == 0,
          "KV lifecycle statistics are wrong");

  ContinuousBatchingEngine repeat(backend, kv_config(2, 8, 3, 2));
  repeat.submit_request(Request({1}, 0, 4, 2));
  repeat.submit_request(Request({2}, 0, 4, 1));
  repeat.run();
  require(same_trace(engine.plan_trace(), repeat.plan_trace()),
          "KV block behavior is not deterministic across runs");
}

void test_kv_planner_reservation_and_decode_deferral() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine admission(backend, kv_config(3, 6, 2, 2));
  admission.submit_request(Request({1}, 0, 2, 1));
  admission.submit_request(Request({2}, 0, 2, 1));
  admission.submit_request(Request({3}, 0, 2, 1));
  require(admission.run_next_iteration(), "reservation plan did not run");
  const auto& first = admission.plan_trace().front();
  require_ids(first.plan.prefill_request_ids(), {1, 2},
              "planner did not reserve two available blocks");
  require(first.plan.deferred_requests().size() == 1 &&
              first.plan.deferred_requests()[0].request_id == RequestId{3} &&
              first.plan.deferred_requests()[0].reason ==
                  DeferralReason::KVCapacity &&
              first.allocated_kv_blocks == 2,
          "planner overcommitted the unchanged free pool");

  ContinuousBatchingEngine decode(backend, kv_config(2, 3, 2, 2));
  decode.submit_request(Request({1}, 0, 2, 1));
  decode.submit_request(Request({2}, 0, 1, 1));
  require(decode.run_next_iteration(), "decode setup prefill did not run");
  require(decode.kv_cache().allocated_block_count() == 2,
          "decode setup residency is wrong");
  require(decode.run_next_iteration(), "capacity-aware decode did not run");
  const auto& second = decode.plan_trace().back();
  require(second.plan.decode_request_ids() ==
              std::vector<RequestId>({RequestId{2}}) &&
              second.plan.deferred_requests().size() == 1 &&
              second.plan.deferred_requests()[0].request_id == RequestId{1} &&
              second.plan.deferred_requests()[0].reason ==
                  DeferralReason::KVCapacity &&
              decode.request({1}).generated_token_count == 0 &&
              decode.request({2}).generated_token_count == 1,
          "decode advanced without successful KV growth");
  require(decode.run_next_iteration(),
          "deferred boundary decode did not resume after release");
  require(decode.request({1}).generated_token_count == 1 &&
              decode.kv_cache().allocated_block_count() == 0,
          "deferred boundary decode did not finish and release");
}

void test_kv_cancellation_and_zero_output_release() {
  const auto backend = batch_backend();
  ContinuousBatchingEngine engine(backend, kv_config(2, 4, 3, 2));
  engine.submit_request(Request({1}, 0, 2, 3));
  engine.submit_request(Request({2}, 10, 2, 1));
  require(engine.run_next_iteration(), "active cancellation setup failed");
  require(engine.kv_cache().allocated_block_count() == 1,
          "active prompt KV was not resident");
  engine.cancel_request({2});
  require(engine.kv_cache().allocated_block_count() == 1,
          "waiting cancellation changed KV occupancy");
  engine.cancel_request({1});
  require(engine.kv_cache().allocated_block_count() == 0 &&
              engine.statistics().cancelled_request_count == 2,
          "active cancellation did not release KV");

  ContinuousBatchingEngine zero(backend, kv_config(1, 2, 1, 2));
  zero.submit_request(Request({9}, 0, 2, 0));
  zero.run();
  require(zero.request({9}).state == RequestState::Finished &&
              zero.kv_cache().allocated_block_count() == 0 &&
              zero.statistics().peak_allocated_kv_blocks == 1 &&
              zero.plan_trace().front().allocated_kv_blocks == 0,
          "zero-output prompt was not allocated then released atomically");
}

void test_simulated_throughput_comparison() {
  SimulatedBackendConfig config;
  config.prefill_base_us = 2;
  config.prefill_per_token_us = 1;
  config.prefill_per_active_sequence_us = 0;
  config.decode_base_us = 2;
  config.decode_per_active_sequence_us = 1;
  config.batch_base_us = 2;
  config.batch_prefill_per_token_us = 1;
  config.batch_decode_per_sequence_us = 1;
  config.batch_active_sequence_overhead_us = 0;
  const SimulatedBackend backend(config);
  SimulationEngine single(backend);
  single.submit_request(Request({1}, 0, 1, 2));
  single.submit_request(Request({2}, 0, 1, 2));
  single.run();

  ContinuousBatchingEngine continuous(
      backend, ContinuousBatchingConfig(2, 4, SchedulingPolicy::DecodeFirst));
  continuous.submit_request(Request({1}, 0, 1, 2));
  continuous.submit_request(Request({2}, 0, 1, 2));
  continuous.run();

  // SIMULATED educational cost-model comparison:
  // S2 performs two times (3 us prefill + two 3 us decode steps) = 18 us.
  // S3 performs one 4 us prefill batch + two 4 us decode batches = 12 us.
  require(continuous.clock().now_us() == 12 && single.clock().now_us() == 18 &&
              continuous.clock().now_us() < single.clock().now_us(),
          "SIMULATED educational cost-model comparison has wrong makespan");
  require(single.scheduler_statistics().completed_request_count == 2 &&
              continuous.statistics().completed_request_count == 2,
          "comparison completed different request counts");
  std::uint64_t single_tokens = 0;
  std::uint64_t continuous_tokens = 0;
  for (RequestId id : {RequestId{1}, RequestId{2}}) {
    const Request& single_request = single.request(id);
    const Request& continuous_request = continuous.request(id);
    require(single_request.arrival_time_us == continuous_request.arrival_time_us &&
                single_request.prompt_length() ==
                    continuous_request.prompt_length() &&
                single_request.max_new_tokens ==
                    continuous_request.max_new_tokens &&
                single_request.state == RequestState::Finished &&
                continuous_request.state == RequestState::Finished,
            "comparison workloads or final states differ");
    single_tokens += single_request.generated_token_count;
    continuous_tokens += continuous_request.generated_token_count;
  }
  require(single_tokens == 4 && continuous_tokens == 4,
          "comparison generated different modeled work");
}

void test_disabled_exact_count_s4_equivalence() {
  const SimulatedBackend backend = batch_backend();
  const auto config = ContinuousBatchingConfig(
      2, 4, SchedulingPolicy::DecodeFirst, KVCacheConfig(8, 2),
      PrefixCacheConfig(false));
  for (std::uint64_t length : {UINT64_C(0), UINT64_C(4)}) {
    ContinuousBatchingEngine count_engine(backend, config);
    ContinuousBatchingEngine exact_engine(backend, config);
    count_engine.submit_request(Request::count_only({1}, 0, length, 2));
    exact_engine.submit_request(Request::exact_tokens(
        {1}, 0, std::vector<std::int32_t>(static_cast<std::size_t>(length), 7), 2));
    require(count_engine.run() == RunResult::Completed &&
                exact_engine.run() == RunResult::Completed,
            "disabled exact/count workloads did not complete");
    const Request& count = count_engine.request({1});
    const Request& exact = exact_engine.request({1});
    require(count.request_id == exact.request_id &&
                count.arrival_time_us == exact.arrival_time_us &&
                count.prompt_length() == exact.prompt_length() &&
                count.max_new_tokens == exact.max_new_tokens &&
                count.generated_token_count == exact.generated_token_count &&
                count.state == exact.state &&
                same_optional(count.admitted_time_us, exact.admitted_time_us) &&
                same_optional(count.first_scheduled_time_us,
                              exact.first_scheduled_time_us) &&
                same_optional(count.first_token_time_us,
                              exact.first_token_time_us) &&
                same_optional(count.finish_time_us, exact.finish_time_us),
            "disabled exact/count request lifecycle differs");
    require(same_trace(count_engine.plan_trace(), exact_engine.plan_trace()),
            "disabled exact/count traces differ");
    require(same_statistics(count_engine.statistics(), exact_engine.statistics()),
            "disabled exact/count statistics differ");
    require(count_engine.kv_cache().physical_blocks() ==
                exact_engine.kv_cache().physical_blocks() &&
                count_engine.kv_cache().free_block_ids() ==
                    exact_engine.kv_cache().free_block_ids() &&
                count_engine.kv_cache().block_tables() ==
                    exact_engine.kv_cache().block_tables() &&
                count_engine.kv_cache().access_epoch() ==
                    exact_engine.kv_cache().access_epoch(),
            "disabled exact/count KV snapshots differ");
    require(exact_engine.statistics().cache_lookup_count == 0 &&
                exact_engine.statistics().prefix_cache_eviction_count == 0 &&
                exact_engine.statistics().cached_kv_blocks == 0 &&
                exact_engine.statistics().referenced_shared_kv_blocks == 0,
            "disabled exact request changed prefix-cache state or metrics");
  }

  ContinuousBatchingEngine count_oversized(backend, config);
  ContinuousBatchingEngine exact_oversized(backend, config);
  require_throws<std::invalid_argument>(
      [&] { count_oversized.submit_request(
          Request::count_only({2}, 0, 5, 1)); },
      "disabled oversized count-only prompt was accepted");
  require_throws<std::invalid_argument>(
      [&] { exact_oversized.submit_request(Request::exact_tokens(
          {2}, 0, std::vector<std::int32_t>(5, 7), 1)); },
      "disabled oversized exact prompt was accepted");
  require(ContinuousBatchingEngineTestAccess::raw_requests(count_oversized).empty() &&
              ContinuousBatchingEngineTestAccess::raw_requests(exact_oversized).empty(),
          "synchronous oversized rejection changed engine state");
}

void test_enabled_oversized_requires_committed_hit() {
  const SimulatedBackend backend = batch_backend();
  const PrefixCacheConfig cache(true, "salt", "model");
  const auto config = ContinuousBatchingConfig(
      2, 2, SchedulingPolicy::DecodeFirst, KVCacheConfig(4, 2), cache);
  ContinuousBatchingEngine hit_engine(backend, config);
  hit_engine.submit_request(Request::exact_tokens({1}, 0, {1, 2}, 0));
  require(hit_engine.run() == RunResult::Completed,
          "cache seed did not complete");
  hit_engine.submit_request(Request::exact_tokens(
      {2}, hit_engine.clock().now_us(), {1, 2, 3, 4}, 0));
  require(hit_engine.run() == RunResult::Completed,
          "oversized prompt with bounded unmatched suffix did not complete");
  const PrefillWork& hit = hit_engine.plan_trace().back().plan.prefill_work().front();
  require(hit.original_prompt_token_count == 4 &&
              hit.matched_prefix_token_count == 2 &&
              hit.prompt_token_count == 2,
          "oversized committed hit scheduled the wrong unmatched work");

  ContinuousBatchingEngine miss_engine(backend, config);
  miss_engine.submit_request(Request::exact_tokens({3}, 0, {9, 8, 7}, 1));
  require_throws<std::invalid_argument>(
      [&] { (void)miss_engine.run_next_iteration(); },
      "oversized unmatched prefill was not rejected deterministically");
  require(miss_engine.failed() &&
              ContinuousBatchingEngineTestAccess::raw_trace(miss_engine).empty(),
          "oversized unmatched prefill was represented as a stall or trace");
}

void test_cache_aware_complete_determinism() {
  const SimulatedBackend backend = batch_backend();
  const auto config = ContinuousBatchingConfig(
      2, 4, SchedulingPolicy::FcfsMixed, KVCacheConfig(4, 2),
      PrefixCacheConfig(true, "salt", "model"));
  ContinuousBatchingEngine lhs(backend, config);
  ContinuousBatchingEngine rhs(backend, config);
  for (ContinuousBatchingEngine* engine : {&lhs, &rhs}) {
    engine->submit_request(Request::exact_tokens({1}, 0, {1, 2, 3, 4}, 0));
    require(engine->run() == RunResult::Completed,
            "deterministic cache seed failed");
    const auto now = engine->clock().now_us();
    engine->submit_request(Request::exact_tokens({2}, now, {1, 2, 8}, 2));
    engine->submit_request(Request::exact_tokens({3}, now, {9}, 1));
    require(engine->run() == RunResult::Completed,
            "deterministic cache workload failed");
  }
  require(lhs.requests().size() == rhs.requests().size(),
          "cache-aware deterministic request counts differ");
  for (const auto& entry : lhs.requests())
    require(same_request(entry.second, rhs.request(entry.first)),
            "cache-aware deterministic request records differ");
  require(same_trace(lhs.plan_trace(), rhs.plan_trace()),
          "cache-aware complete traces differ");
  require(same_statistics(lhs.statistics(), rhs.statistics()),
          "cache-aware complete statistics differ");
  require(lhs.kv_cache().physical_blocks() == rhs.kv_cache().physical_blocks() &&
              lhs.kv_cache().free_block_ids() == rhs.kv_cache().free_block_ids() &&
              lhs.kv_cache().block_tables() == rhs.kv_cache().block_tables() &&
              lhs.kv_cache().access_epoch() == rhs.kv_cache().access_epoch(),
          "cache-aware deterministic KV snapshots differ");
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"configuration and BatchPlan", test_configuration_and_plan_validation},
      {"oversized prompt", test_oversized_prompt_rejected_synchronously},
      {"cancellation overflow strong guarantee",
       test_cancellation_overflow_strong_guarantee},
      {"DecodeFirst lifecycle trace statistics",
       test_decode_first_lifecycle_trace_and_statistics},
      {"DecodeFirst token deferral", test_decode_first_token_budget_deferral},
      {"FcfsMixed order and trace", test_fcfs_mixed_order_and_trace},
      {"deferral reason precedence", test_deferral_reason_precedence},
      {"active decode deferral reasons", test_active_decode_deferral_reasons},
      {"zero prompt progress", test_zero_prompt_progress},
      {"arrival exactly at iteration end",
       test_arrival_exactly_at_iteration_end},
      {"same-workload policy difference",
       test_policy_difference_on_same_workload},
      {"per-iteration sequence limit",
       test_per_iteration_sequence_limit_is_not_residency},
      {"zero output idle arrival", test_zero_output_idle_and_arrival_boundary},
      {"cancellation", test_cancellation_stops_work},
      {"determinism and overflow", test_determinism_and_checked_overflow},
      {"two-phase failure guarantee",
       test_iteration_two_phase_failure_guarantee},
      {"deferred-only KV stalls", test_deferred_only_kv_stalls_are_nonterminal},
      {"same-plan completion KV ordering",
       test_same_plan_completion_cannot_donate_blocks},
      {"KV capacity lifecycle and determinism",
       test_kv_capacity_lifecycle_and_determinism},
      {"KV planner reservation and decode deferral",
       test_kv_planner_reservation_and_decode_deferral},
      {"KV cancellation and zero-output release",
       test_kv_cancellation_and_zero_output_release},
      {"disabled exact/count S4 equivalence",
       test_disabled_exact_count_s4_equivalence},
      {"enabled oversized committed-hit policy",
       test_enabled_oversized_requires_committed_hit},
      {"cache-aware complete determinism",
       test_cache_aware_complete_determinism},
      {"SIMULATED educational cost-model comparison",
       test_simulated_throughput_comparison},
  };
  std::size_t failures = 0;
  for (const auto& test : tests) {
    try {
      test.second();
      std::cout << "PASS: " << test.first << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
    }
  }
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  return 0;
}
