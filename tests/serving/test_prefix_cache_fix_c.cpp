#include "serving/continuous_batching.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

namespace llm_lab::serving::test {

struct TokenKeyLess {
  bool operator()(const TokenBlockKey& a, const TokenBlockKey& b) const {
    return std::tie(a.parent_hash, a.token_ids, a.model_namespace,
                    a.cache_salt, a.block_hash) <
           std::tie(b.parent_hash, b.token_ids, b.model_namespace,
                    b.cache_salt, b.block_hash);
  }
};

using CanonicalMap = std::map<TokenBlockKey, std::size_t, TokenKeyLess>;

struct PromptProvenanceSnapshot {
  RequestId request_id;
  bool publication_eligible;
  std::optional<std::vector<std::int32_t>> exact_prompt_tokens;
  std::uint64_t total_prompt_token_count;
  std::size_t cache_eligible_full_prompt_blocks;
  std::size_t matched_shared_prefix_blocks;
  std::size_t computed_full_begin;
  std::size_t computed_full_end;
  std::optional<std::size_t> private_prompt_tail_position;
  std::size_t decode_block_start_position;
  bool publication_completed;
};

bool operator==(const PromptProvenanceSnapshot& a,
                const PromptProvenanceSnapshot& b) {
  return a.request_id == b.request_id &&
      a.publication_eligible == b.publication_eligible &&
      a.exact_prompt_tokens == b.exact_prompt_tokens &&
      a.total_prompt_token_count == b.total_prompt_token_count &&
      a.cache_eligible_full_prompt_blocks == b.cache_eligible_full_prompt_blocks &&
      a.matched_shared_prefix_blocks == b.matched_shared_prefix_blocks &&
      a.computed_full_begin == b.computed_full_begin &&
      a.computed_full_end == b.computed_full_end &&
      a.private_prompt_tail_position == b.private_prompt_tail_position &&
      a.decode_block_start_position == b.decode_block_start_position &&
      a.publication_completed == b.publication_completed;
}

struct ManagerSnapshot {
  std::size_t total_blocks;
  std::uint64_t block_size;
  std::uint64_t total_capacity;
  bool prefix_enabled;
  std::string salt;
  std::string model_namespace;
  PrefixCacheEvictionPolicy eviction_policy;
  std::vector<PhysicalKVBlock> blocks;
  std::set<std::size_t> free_ids;
  std::map<RequestId, std::vector<std::size_t>> tables;
  std::map<RequestId, PromptProvenanceSnapshot> provenance;
  std::map<std::uint64_t, std::vector<std::size_t>> buckets;
  CanonicalMap canonical;
  std::uint64_t represented_tokens;
  std::size_t allocated_blocks;
  std::size_t free_blocks;
  std::uint64_t fragmentation;
  double utilization;
  std::size_t peak_allocated;
  double peak_utilization;
  std::uint64_t allocation_failures;
  PrefixCacheMetrics metrics;
  std::size_t cached_blocks;
  std::size_t referenced_shared_blocks;
  std::optional<double> hit_rate;
  std::uint64_t access_epoch;
  std::vector<std::uint64_t> hash_probe_outputs;
  KVCacheFailurePoint failure_point;
};

bool same_metrics(const PrefixCacheMetrics& a, const PrefixCacheMetrics& b) {
  return a.cache_lookup_count == b.cache_lookup_count &&
      a.cache_hit_lookup_count == b.cache_hit_lookup_count &&
      a.cache_miss_lookup_count == b.cache_miss_lookup_count &&
      a.matched_block_count == b.matched_block_count &&
      a.matched_token_count == b.matched_token_count &&
      a.total_cache_eligible_prompt_tokens_looked_up ==
          b.total_cache_eligible_prompt_tokens_looked_up &&
      a.reused_request_count == b.reused_request_count &&
      a.saved_prefill_token_count == b.saved_prefill_token_count &&
      a.collision_verification_count == b.collision_verification_count &&
      a.eviction_count == b.eviction_count;
}

bool same_manager(const ManagerSnapshot& a, const ManagerSnapshot& b) {
  if (a.total_blocks != b.total_blocks || a.block_size != b.block_size ||
      a.total_capacity != b.total_capacity ||
      a.prefix_enabled != b.prefix_enabled || a.salt != b.salt ||
      a.model_namespace != b.model_namespace ||
      a.eviction_policy != b.eviction_policy || a.blocks != b.blocks ||
      a.free_ids != b.free_ids || a.tables != b.tables ||
      a.provenance != b.provenance || a.buckets != b.buckets ||
      a.canonical != b.canonical ||
      a.represented_tokens != b.represented_tokens ||
      a.allocated_blocks != b.allocated_blocks ||
      a.free_blocks != b.free_blocks || a.fragmentation != b.fragmentation ||
      a.utilization != b.utilization || a.peak_allocated != b.peak_allocated ||
      a.peak_utilization != b.peak_utilization ||
      a.allocation_failures != b.allocation_failures ||
      !same_metrics(a.metrics, b.metrics) ||
      a.cached_blocks != b.cached_blocks ||
      a.referenced_shared_blocks != b.referenced_shared_blocks ||
      a.hit_rate != b.hit_rate || a.access_epoch != b.access_epoch ||
      a.hash_probe_outputs != b.hash_probe_outputs ||
      a.failure_point != b.failure_point) return false;
  return true;
}

struct RequestSnapshot {
  RequestId id;
  std::int64_t arrival;
  bool exact;
  std::uint64_t prompt_length;
  std::optional<std::vector<std::int32_t>> tokens;
  std::uint64_t max_new;
  std::uint64_t generated;
  RequestState state;
  std::optional<std::int64_t> admitted;
  std::optional<std::int64_t> first_scheduled;
  std::optional<std::int64_t> first_token;
  std::optional<std::int64_t> finish;
};

bool operator==(const RequestSnapshot& a, const RequestSnapshot& b) {
  return a.id == b.id && a.arrival == b.arrival && a.exact == b.exact &&
      a.prompt_length == b.prompt_length && a.tokens == b.tokens &&
      a.max_new == b.max_new && a.generated == b.generated &&
      a.state == b.state && a.admitted == b.admitted &&
      a.first_scheduled == b.first_scheduled &&
      a.first_token == b.first_token && a.finish == b.finish;
}

bool same_prefill(const PrefillWork& a, const PrefillWork& b) {
  return a.request_id == b.request_id &&
      a.prompt_token_count == b.prompt_token_count &&
      a.original_prompt_token_count == b.original_prompt_token_count &&
      a.matched_prefix_token_count == b.matched_prefix_token_count &&
      a.matched_physical_block_ids == b.matched_physical_block_ids &&
      a.newly_allocated_block_ids == b.newly_allocated_block_ids &&
      a.evicted_block_ids == b.evicted_block_ids &&
      a.prefix_lookup_kind == b.prefix_lookup_kind;
}

bool same_plan(const BatchPlan& a, const BatchPlan& b) {
  if (a.iteration_number() != b.iteration_number() ||
      a.prefill_request_ids() != b.prefill_request_ids() ||
      a.decode_request_ids() != b.decode_request_ids() ||
      a.total_prefill_tokens() != b.total_prefill_tokens() ||
      a.total_decode_tokens() != b.total_decode_tokens() ||
      a.total_scheduled_tokens() != b.total_scheduled_tokens() ||
      a.planned_eviction_ids() != b.planned_eviction_ids() ||
      a.planned_allocation_ids() != b.planned_allocation_ids() ||
      a.prefill_work().size() != b.prefill_work().size() ||
      a.decode_work().size() != b.decode_work().size() ||
      a.work_order().size() != b.work_order().size() ||
      a.deferred_requests().size() != b.deferred_requests().size()) return false;
  for (std::size_t i = 0; i < a.prefill_work().size(); ++i)
    if (!same_prefill(a.prefill_work()[i], b.prefill_work()[i])) return false;
  for (std::size_t i = 0; i < a.decode_work().size(); ++i) {
    const auto& x = a.decode_work()[i]; const auto& y = b.decode_work()[i];
    if (x.request_id != y.request_id ||
        x.newly_allocated_block_id != y.newly_allocated_block_id ||
        x.evicted_block_ids != y.evicted_block_ids) return false;
  }
  for (std::size_t i = 0; i < a.work_order().size(); ++i)
    if (a.work_order()[i].request_id != b.work_order()[i].request_id ||
        a.work_order()[i].is_prefill != b.work_order()[i].is_prefill) return false;
  for (std::size_t i = 0; i < a.deferred_requests().size(); ++i)
    if (a.deferred_requests()[i].request_id != b.deferred_requests()[i].request_id ||
        a.deferred_requests()[i].reason != b.deferred_requests()[i].reason) return false;
  return true;
}

bool same_trace(const std::vector<BatchTraceEntry>& a,
                const std::vector<BatchTraceEntry>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].iteration_number != b[i].iteration_number ||
        a[i].start_timestamp_us != b[i].start_timestamp_us ||
        a[i].end_timestamp_us != b[i].end_timestamp_us ||
        a[i].policy != b[i].policy || !same_plan(a[i].plan, b[i].plan) ||
        a[i].allocated_kv_blocks != b[i].allocated_kv_blocks ||
        a[i].free_kv_blocks != b[i].free_kv_blocks ||
        a[i].represented_kv_tokens != b[i].represented_kv_tokens ||
        a[i].internal_fragmentation_tokens != b[i].internal_fragmentation_tokens ||
        a[i].cached_kv_blocks != b[i].cached_kv_blocks ||
        a[i].referenced_shared_kv_blocks != b[i].referenced_shared_kv_blocks ||
        a[i].kv_block_utilization != b[i].kv_block_utilization ||
        a[i].kv_block_tables != b[i].kv_block_tables) return false;
  }
  return true;
}

bool same_statistics(const ContinuousBatchingStatistics& a,
                     const ContinuousBatchingStatistics& b) {
  return std::tie(a.scheduling_iteration_count, a.nonempty_batch_count,
      a.idle_iteration_count, a.stalled_iteration_count,
      a.total_prefill_tokens_scheduled, a.total_decode_tokens_scheduled,
      a.total_scheduled_sequences, a.maximum_batch_size,
      a.maximum_scheduled_tokens, a.deferred_request_count,
      a.completed_request_count, a.cancelled_request_count,
      a.current_allocated_kv_blocks, a.peak_allocated_kv_blocks,
      a.current_kv_block_utilization, a.peak_kv_block_utilization,
      a.represented_kv_tokens, a.internal_fragmentation_tokens,
      a.kv_allocation_failure_count, a.kv_capacity_deferral_count,
      a.total_original_prompt_tokens, a.total_matched_prefix_tokens,
      a.saved_simulated_prefill_tokens, a.cache_lookup_count,
      a.cache_hit_lookup_count, a.cache_miss_lookup_count,
      a.cache_matched_block_count,
      a.total_cache_eligible_prompt_tokens_looked_up,
      a.collision_verification_count, a.prefix_cache_eviction_count,
      a.cached_kv_blocks, a.referenced_shared_kv_blocks) ==
    std::tie(b.scheduling_iteration_count, b.nonempty_batch_count,
      b.idle_iteration_count, b.stalled_iteration_count,
      b.total_prefill_tokens_scheduled, b.total_decode_tokens_scheduled,
      b.total_scheduled_sequences, b.maximum_batch_size,
      b.maximum_scheduled_tokens, b.deferred_request_count,
      b.completed_request_count, b.cancelled_request_count,
      b.current_allocated_kv_blocks, b.peak_allocated_kv_blocks,
      b.current_kv_block_utilization, b.peak_kv_block_utilization,
      b.represented_kv_tokens, b.internal_fragmentation_tokens,
      b.kv_allocation_failure_count, b.kv_capacity_deferral_count,
      b.total_original_prompt_tokens, b.total_matched_prefix_tokens,
      b.saved_simulated_prefill_tokens, b.cache_lookup_count,
      b.cache_hit_lookup_count, b.cache_miss_lookup_count,
      b.cache_matched_block_count,
      b.total_cache_eligible_prompt_tokens_looked_up,
      b.collision_verification_count, b.prefix_cache_eviction_count,
      b.cached_kv_blocks, b.referenced_shared_kv_blocks);
}

struct EngineSnapshot {
  std::size_t max_sequences;
  std::uint64_t max_tokens;
  SchedulingPolicy policy;
  std::size_t kv_blocks;
  std::uint64_t kv_block_size;
  bool prefix_enabled;
  std::string salt;
  std::string model_namespace;
  PrefixCacheEvictionPolicy prefix_policy;
  std::vector<RequestSnapshot> requests;
  std::set<RequestId> arrivals;
  std::int64_t clock;
  std::uint64_t next_iteration;
  std::vector<BatchTraceEntry> trace;
  ContinuousBatchingStatistics statistics;
  ManagerSnapshot manager;
  bool failed;
  bool trace_failure;
  bool reservation_failure;
  KVCacheFailurePoint prepared_failure;
};

bool same_engine(const EngineSnapshot& a, const EngineSnapshot& b,
                 bool compare_failed = true) {
  return a.max_sequences == b.max_sequences && a.max_tokens == b.max_tokens &&
      a.policy == b.policy && a.kv_blocks == b.kv_blocks &&
      a.kv_block_size == b.kv_block_size &&
      a.prefix_enabled == b.prefix_enabled && a.salt == b.salt &&
      a.model_namespace == b.model_namespace &&
      a.prefix_policy == b.prefix_policy && a.requests == b.requests &&
      a.arrivals == b.arrivals && a.clock == b.clock &&
      a.next_iteration == b.next_iteration && same_trace(a.trace, b.trace) &&
      same_statistics(a.statistics, b.statistics) &&
      same_manager(a.manager, b.manager) &&
      (!compare_failed || a.failed == b.failed) &&
      a.trace_failure == b.trace_failure &&
      a.reservation_failure == b.reservation_failure &&
      a.prepared_failure == b.prepared_failure;
}

struct FixCTestAccess {
  static ManagerSnapshot manager_snapshot(const KVCacheManager& manager) {
    ManagerSnapshot s;
    s.total_blocks = manager.config_.total_num_blocks();
    s.block_size = manager.config_.block_size_tokens();
    s.total_capacity = manager.config_.total_capacity_tokens();
    s.prefix_enabled = manager.prefix_config_.enabled();
    s.salt = manager.prefix_config_.cache_salt();
    s.model_namespace = manager.prefix_config_.model_namespace();
    s.eviction_policy = manager.prefix_config_.eviction_policy();
    s.blocks = manager.blocks_; s.free_ids = manager.free_blocks_;
    s.tables = manager.block_tables_; s.buckets = manager.prefix_index_;
    for (const auto& entry : manager.request_provenance_) {
      const auto& p = entry.second;
      s.provenance.emplace(entry.first, PromptProvenanceSnapshot{
          p.request_id, p.publication_eligible, p.exact_prompt_tokens,
          p.total_prompt_token_count, p.cache_eligible_full_prompt_blocks,
          p.matched_shared_prefix_blocks, p.computed_full_begin,
          p.computed_full_end, p.private_prompt_tail_position,
          p.decode_block_start_position, p.publication_completed});
    }
    for (const auto& block : manager.blocks_)
      if (block.prefix_key) s.canonical.emplace(*block.prefix_key, block.block_id);
    s.represented_tokens = manager.represented_token_count_;
    s.allocated_blocks = manager.allocated_block_count();
    s.free_blocks = manager.free_block_count();
    s.fragmentation = manager.internal_fragmentation_tokens();
    s.utilization = manager.utilization();
    s.peak_allocated = manager.peak_allocated_block_count_;
    s.peak_utilization = manager.peak_utilization();
    s.allocation_failures = manager.allocation_failure_count_;
    s.metrics = manager.prefix_metrics_;
    s.cached_blocks = manager.cached_block_count();
    s.referenced_shared_blocks = manager.referenced_shared_block_count();
    s.hit_rate = manager.prefix_metrics_.prefix_token_hit_rate();
    s.access_epoch = manager.access_epoch_;
    for (TokenBlockKey probe : std::vector<TokenBlockKey>{
             TokenBlockKey{kPrefixCacheRootHash, {}, "", "", 0},
             TokenBlockKey{kPrefixCacheRootHash, {-1, 0, 1}, "probe-model",
                           "probe-salt", 0},
             TokenBlockKey{UINT64_C(0x0123456789abcdef),
                           {10, 11, 12, 13}, "m", "s", 0}}) {
      s.hash_probe_outputs.push_back(manager.hash_function_(probe));
    }
    s.failure_point = manager.failure_point_for_test_;
    return s;
  }

  static EngineSnapshot engine_snapshot(const ContinuousBatchingEngine& engine) {
    const auto& c = engine.config_;
    EngineSnapshot s{c.max_num_sequences(), c.max_batched_tokens(),
        c.scheduling_policy(), c.kv_cache_config().total_num_blocks(),
        c.kv_cache_config().block_size_tokens(),
        c.prefix_cache_config().enabled(), c.prefix_cache_config().cache_salt(),
        c.prefix_cache_config().model_namespace(),
        c.prefix_cache_config().eviction_policy(), {}, engine.arrived_requests_,
        engine.clock_.now_us(), engine.next_iteration_number_, engine.plan_trace_,
        engine.statistics_, manager_snapshot(engine.kv_cache_), engine.failed_,
        engine.fail_trace_preparation_for_test_,
        engine.fail_reservation_match_for_test_,
        engine.prepared_kv_failure_for_test_};
    for (const auto& entry : engine.requests_) {
      const Request& r = entry.second;
      s.requests.push_back(RequestSnapshot{r.request_id, r.arrival_time_us,
          r.has_exact_prompt_tokens(), r.prompt_length(),
          r.has_exact_prompt_tokens()
              ? std::optional<std::vector<std::int32_t>>(r.prompt_tokens())
              : std::nullopt,
          r.max_new_tokens, r.generated_token_count, r.state,
          r.admitted_time_us, r.first_scheduled_time_us,
          r.first_token_time_us, r.finish_time_us});
    }
    return s;
  }

  static void set_failure(KVCacheManager& manager, KVCacheFailurePoint point) {
    manager.failure_point_for_test_ = point;
  }
  static void set_allocation_failures(KVCacheManager& manager,
                                      std::uint64_t value) {
    manager.allocation_failure_count_ = value;
  }
  static void replace_hash(KVCacheManager& manager,
                           PrefixHashFunction hash_function) {
    manager.hash_function_ = std::move(hash_function);
  }
  static void tie_cached_epochs(KVCacheManager& manager, std::uint64_t epoch) {
    for (auto& block : manager.blocks_)
      if (block.state == PhysicalKVBlockState::Cached)
        block.last_access_epoch = epoch;
    manager.access_epoch_ = epoch;
  }
  static void set_prepared_failure(ContinuousBatchingEngine& engine,
                                   KVCacheFailurePoint point) {
    engine.prepared_kv_failure_for_test_ = point;
  }
  static void set_trace_failure(ContinuousBatchingEngine& engine) {
    engine.fail_trace_preparation_for_test_ = true;
  }
  static void set_reservation_failure(ContinuousBatchingEngine& engine) {
    engine.fail_reservation_match_for_test_ = true;
  }
  static void set_next_iteration(ContinuousBatchingEngine& engine,
                                 std::uint64_t value) {
    engine.next_iteration_number_ = value;
  }
  static ContinuousBatchingStatistics& statistics(ContinuousBatchingEngine& e) {
    return e.statistics_;
  }
  static void advance_clock(ContinuousBatchingEngine& e, std::int64_t value) {
    e.clock_.advance_to(value);
  }
};

}  // namespace llm_lab::serving::test

namespace {
using namespace llm_lab::serving;
using llm_lab::serving::test::EngineSnapshot;
using llm_lab::serving::test::FixCTestAccess;
using llm_lab::serving::test::KVCacheFailurePoint;
using llm_lab::serving::test::ManagerSnapshot;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Exception, class Function>
void require_throws(Function&& function, const std::string& message) {
  try { function(); }
  catch (const Exception&) { return; }
  throw std::runtime_error(message);
}

template <class Exception, class Function>
bool throws_exactly(Function&& function) {
  try {
    function();
  } catch (const std::exception& error) {
    return typeid(error) == typeid(Exception);
  } catch (...) {
    return false;
  }
  return false;
}

template <class Exception, class Function>
void require_throws_exactly(Function&& function, const std::string& message) {
  if (!throws_exactly<Exception>(std::forward<Function>(function)))
    throw std::runtime_error(message);
}

PrefixCacheConfig enabled() {
  return PrefixCacheConfig(true, "fix-c-salt", "fix-c-model");
}

struct ShadowCache {
  struct Provenance {
    RequestId request_id;
    bool eligible{false};
    std::optional<std::vector<std::int32_t>> tokens;
    std::uint64_t total{0};
    std::size_t full{0};
    std::size_t matched{0};
    std::size_t computed_begin{0};
    std::size_t computed_end{0};
    std::optional<std::size_t> tail;
    std::size_t decode_start{0};
    bool published{false};
  };

  std::size_t count;
  std::uint64_t block_size;
  bool enabled_cache;
  std::string salt{"fix-c-salt"};
  std::string model{"fix-c-model"};
  bool constant_hash{false};
  std::vector<PhysicalKVBlock> blocks;
  std::set<std::size_t> free;
  std::map<RequestId, std::vector<std::size_t>> tables;
  std::map<RequestId, Provenance> provenance;
  std::map<std::uint64_t, std::vector<std::size_t>> buckets;
  llm_lab::serving::test::CanonicalMap canonical;
  PrefixCacheMetrics metrics;
  std::uint64_t represented{0};
  std::size_t peak{0};
  std::uint64_t failures{0};
  std::uint64_t epoch{0};
  std::size_t derived_allocated{0};
  std::size_t derived_free{0};
  double derived_utilization{0.0};
  double derived_peak_utilization{0.0};
  std::size_t derived_cached{0};
  std::size_t derived_shared{0};

  ShadowCache(std::size_t n, std::uint64_t bs, bool cache = true,
              bool collide = false)
      : count(n), block_size(bs), enabled_cache(cache), constant_hash(collide) {
    for (std::size_t id = 0; id < count; ++id) {
      blocks.push_back(PhysicalKVBlock{id, PhysicalKVBlockState::Free,
          std::nullopt, 0, 0, 0, std::nullopt, std::nullopt});
      free.insert(id);
    }
    refresh_derived();
  }

  void refresh_derived() {
    derived_free = free.size();
    derived_allocated = count - derived_free;
    derived_utilization = static_cast<double>(derived_allocated) /
                          static_cast<double>(count);
    derived_peak_utilization = static_cast<double>(peak) /
                               static_cast<double>(count);
    derived_cached = static_cast<std::size_t>(std::count_if(
        blocks.begin(), blocks.end(), [](const PhysicalKVBlock& block) {
          return block.state == PhysicalKVBlockState::Cached;
        }));
    derived_shared = static_cast<std::size_t>(std::count_if(
        blocks.begin(), blocks.end(), [](const PhysicalKVBlock& block) {
          return block.prefix_key.has_value() && block.reference_count > 0;
        }));
  }

  static void byte(std::uint64_t& h, std::uint8_t v) {
    h ^= v; h *= UINT64_C(1099511628211);
  }
  static void u64(std::uint64_t& h, std::uint64_t v) {
    for (unsigned i = 0; i < 8; ++i)
      byte(h, static_cast<std::uint8_t>(v >> (8U * i)));
  }
  static void text(std::uint64_t& h, const std::string& value) {
    u64(h, value.size());
    for (unsigned char v : value) byte(h, v);
  }
  std::uint64_t hash(const TokenBlockKey& key) const {
    if (constant_hash) return 7;
    std::uint64_t h = kPrefixCacheRootHash;
    byte(h, 1); u64(h, key.parent_hash);
    byte(h, 2); text(h, key.model_namespace);
    byte(h, 3); text(h, key.cache_salt);
    byte(h, 4); u64(h, key.token_ids.size());
    for (std::int32_t token : key.token_ids) {
      const std::uint32_t bits = static_cast<std::uint32_t>(token);
      for (unsigned i = 0; i < 4; ++i)
        byte(h, static_cast<std::uint8_t>(bits >> (8U * i)));
    }
    return h;
  }
  TokenBlockKey key(std::uint64_t parent,
                    const std::vector<std::int32_t>& tokens,
                    std::size_t offset) const {
    TokenBlockKey result;
    result.parent_hash = parent;
    result.token_ids.assign(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
        tokens.begin() + static_cast<std::ptrdiff_t>(offset + block_size));
    result.model_namespace = model; result.cache_salt = salt;
    result.block_hash = hash(result);
    return result;
  }
  static bool exact(const TokenBlockKey& a, const TokenBlockKey& b) {
    return a.parent_hash == b.parent_hash && a.token_ids == b.token_ids &&
        a.model_namespace == b.model_namespace && a.cache_salt == b.cache_salt;
  }
  std::size_t required(std::uint64_t tokens) const {
    return tokens == 0 ? 0 : static_cast<std::size_t>(
        tokens / block_size + (tokens % block_size != 0));
  }
  PrefixLookupResult lookup(const std::vector<std::int32_t>& tokens) const {
    PrefixLookupResult result;
    if (!enabled_cache) { result.kind = PrefixLookupKind::Disabled; return result; }
    const std::size_t full = tokens.size() / static_cast<std::size_t>(block_size);
    result.eligible_token_count = full * block_size;
    std::uint64_t parent = kPrefixCacheRootHash;
    for (std::size_t i = 0; i < full; ++i) {
      const TokenBlockKey wanted = key(parent, tokens, i * block_size);
      std::optional<std::size_t> found;
      const auto bucket = buckets.find(wanted.block_hash);
      if (bucket != buckets.end()) {
        for (std::size_t id : bucket->second) {
          if (blocks[id].prefix_key && exact(*blocks[id].prefix_key, wanted)) {
            found = id; break;
          }
          ++result.collision_verification_count;
        }
      }
      if (!found) break;
      result.physical_block_ids.push_back(*found);
      ++result.matched_block_count;
      result.matched_token_count += block_size;
      parent = wanted.block_hash;
    }
    result.next_parent_hash = parent;
    result.kind = result.matched_block_count == full && full != 0
        ? PrefixLookupKind::AllEligibleBlocksHit
        : result.matched_block_count != 0 ? PrefixLookupKind::PartialHit
                                          : PrefixLookupKind::Miss;
    return result;
  }
  std::vector<std::size_t> lru() const {
    std::vector<std::size_t> ids;
    for (const auto& block : blocks)
      if (block.state == PhysicalKVBlockState::Cached) ids.push_back(block.block_id);
    std::sort(ids.begin(), ids.end(), [&](std::size_t a, std::size_t b) {
      return std::tie(blocks[a].last_access_epoch, a) <
             std::tie(blocks[b].last_access_epoch, b);
    });
    return ids;
  }
  void evict_ids(const std::vector<std::size_t>& ids) {
    for (std::size_t id : ids) {
      auto& block = blocks[id];
      canonical.erase(*block.prefix_key);
      auto& bucket = buckets[block.prefix_key->block_hash];
      bucket.erase(std::find(bucket.begin(), bucket.end(), id));
      if (bucket.empty()) buckets.erase(block.prefix_key->block_hash);
      represented -= block.valid_token_count; ++metrics.eviction_count;
      block = PhysicalKVBlock{id, PhysicalKVBlockState::Free, std::nullopt,
                              0, 0, 0, std::nullopt, std::nullopt};
      free.insert(id);
    }
    refresh_derived();
  }
  std::pair<std::vector<std::size_t>, std::vector<std::size_t>> reserve(
      std::size_t needed, const std::vector<std::size_t>& protected_ids = {}) const {
    auto victims = lru();
    victims.erase(std::remove_if(victims.begin(), victims.end(),
        [&](std::size_t id) {
          return std::find(protected_ids.begin(), protected_ids.end(), id) !=
                 protected_ids.end();
        }), victims.end());
    const std::size_t evictions = needed > free.size() ? needed - free.size() : 0;
    if (evictions > victims.size()) throw std::runtime_error("shadow KV capacity");
    victims.resize(evictions);
    std::set<std::size_t> available = free;
    available.insert(victims.begin(), victims.end());
    std::vector<std::size_t> allocations;
    auto it = available.begin();
    for (std::size_t i = 0; i < needed; ++i, ++it) allocations.push_back(*it);
    return {victims, allocations};
  }
  void acquire(RequestId id, const std::vector<std::int32_t>& tokens,
               const PrefixLookupResult& supplied) {
    if (tables.count(id)) throw std::invalid_argument("shadow duplicate request");
    const PrefixLookupResult actual = lookup(tokens);
    if (actual.physical_block_ids != supplied.physical_block_ids)
      throw std::logic_error("shadow stale lookup");
    const std::uint64_t suffix = tokens.size() - actual.matched_token_count;
    const auto reservation = reserve(required(suffix), actual.physical_block_ids);
    ++metrics.cache_lookup_count;
    metrics.total_cache_eligible_prompt_tokens_looked_up += actual.eligible_token_count;
    metrics.collision_verification_count += actual.collision_verification_count;
    if (actual.matched_block_count) {
      ++metrics.cache_hit_lookup_count; ++metrics.reused_request_count;
      metrics.matched_block_count += actual.matched_block_count;
      metrics.matched_token_count += actual.matched_token_count;
      metrics.saved_prefill_token_count += actual.matched_token_count;
      ++epoch;
      for (std::size_t block_id : actual.physical_block_ids) {
        auto& block = blocks[block_id]; const bool cached = block.state == PhysicalKVBlockState::Cached;
        ++block.reference_count; block.state = PhysicalKVBlockState::InUse;
        block.owner.reset(); block.last_access_epoch = epoch;
        if (cached && block.provenance == KVBlockProvenance::ComputedPromptFull)
          block.provenance = KVBlockProvenance::SharedPromptFull;
      }
    } else ++metrics.cache_miss_lookup_count;
    evict_ids(reservation.first);
    std::vector<std::size_t> table = actual.physical_block_ids;
    if (!reservation.second.empty()) ++epoch;
    const std::size_t full = tokens.size() / static_cast<std::size_t>(block_size);
    for (std::size_t i = 0; i < reservation.second.size(); ++i) {
      const std::size_t block_id = reservation.second[i];
      auto& block = blocks[block_id]; block.state = PhysicalKVBlockState::InUse;
      block.owner = id; block.reference_count = 1;
      block.valid_token_count = std::min(block_size, suffix - i * block_size);
      block.last_access_epoch = epoch;
      block.provenance = actual.matched_block_count + i < full
          ? KVBlockProvenance::ComputedPromptFull
          : KVBlockProvenance::PrivatePromptTail;
      free.erase(block_id); table.push_back(block_id);
    }
    represented += suffix; peak = std::max(peak, count - free.size());
    tables.emplace(id, table);
    provenance.emplace(id, Provenance{id, true, tokens,
        static_cast<std::uint64_t>(tokens.size()), full,
        actual.matched_block_count, actual.matched_block_count, full,
        tokens.size() % block_size ? std::optional<std::size_t>(full) : std::nullopt,
        full + (tokens.size() % block_size ? 1U : 0U), false});
    refresh_derived();
  }
  void allocate_count(RequestId id, std::uint64_t tokens) {
    if (tables.count(id)) throw std::invalid_argument("shadow duplicate request");
    const auto reservation = reserve(required(tokens)); evict_ids(reservation.first);
    if (!reservation.second.empty()) ++epoch;
    for (std::size_t i = 0; i < reservation.second.size(); ++i) {
      auto& b = blocks[reservation.second[i]]; b.state = PhysicalKVBlockState::InUse;
      b.owner = id; b.reference_count = 1;
      b.valid_token_count = std::min(block_size, tokens - i * block_size);
      b.last_access_epoch = epoch;
      b.provenance = KVBlockProvenance::PrivatePromptCountOnly;
      free.erase(b.block_id);
    }
    tables.emplace(id, reservation.second);
    provenance.emplace(id, Provenance{id, false, std::nullopt, tokens, 0, 0, 0, 0,
                                      std::nullopt, reservation.second.size(), false});
    represented += tokens; peak = std::max(peak, count - free.size());
    refresh_derived();
  }
  void publish(RequestId id) {
    auto& p = provenance.at(id);
    if (!enabled_cache || !p.eligible || !p.tokens ||
        p.computed_begin == p.computed_end) throw std::logic_error("shadow ineligible publish");
    if (p.published) return;
    std::uint64_t parent = kPrefixCacheRootHash;
    for (std::size_t i = 0; i < p.full; ++i) {
      TokenBlockKey wanted = key(parent, *p.tokens, i * block_size);
      if (i >= p.computed_begin) {
        const bool canonical_exists = canonical.find(wanted) != canonical.end();
        auto& block = blocks[tables.at(id)[i]];
        if (!canonical_exists) {
          ++epoch; block.prefix_key = wanted; block.owner.reset();
          block.last_access_epoch = epoch;
          auto& ids = buckets[wanted.block_hash]; ids.push_back(block.block_id);
          std::sort(ids.begin(), ids.end());
          canonical.emplace(wanted, block.block_id);
        }
      }
      parent = wanted.block_hash;
    }
    p.published = true;
    refresh_derived();
  }
  void append(RequestId id) {
    auto& table = tables.at(id); bool fresh = table.empty();
    if (!fresh) {
      const auto& last = blocks[table.back()];
      fresh = last.prefix_key.has_value() || last.reference_count > 1 ||
              last.valid_token_count == block_size;
    }
    if (fresh) {
      const auto reservation = reserve(1); evict_ids(reservation.first);
      const std::size_t block_id = reservation.second.front();
      ++epoch; auto& b = blocks[block_id]; b.state = PhysicalKVBlockState::InUse;
      b.owner = id; b.reference_count = 1; b.valid_token_count = 1;
      b.last_access_epoch = epoch; b.provenance = KVBlockProvenance::PrivateDecode;
      free.erase(block_id); table.push_back(block_id);
    } else {
      ++epoch; ++blocks[table.back()].valid_token_count;
      blocks[table.back()].last_access_epoch = epoch;
    }
    ++represented; peak = std::max(peak, count - free.size());
    refresh_derived();
  }
  void release(RequestId id) {
    auto table = tables.at(id);
    for (std::size_t block_id : table) {
      auto& b = blocks[block_id]; --b.reference_count;
      if (b.reference_count) { b.owner.reset(); continue; }
      if (b.prefix_key) { b.state = PhysicalKVBlockState::Cached; b.owner.reset(); }
      else {
        represented -= b.valid_token_count;
        b = PhysicalKVBlock{block_id, PhysicalKVBlockState::Free,
                            std::nullopt, 0, 0, 0, std::nullopt, std::nullopt};
        free.insert(block_id);
      }
    }
    tables.erase(id); provenance.erase(id);
    refresh_derived();
  }
};

void compare_shadow(const KVCacheManager& manager, const ShadowCache& shadow,
                    const std::string& label) {
  const ManagerSnapshot actual = FixCTestAccess::manager_snapshot(manager);
  if (actual.blocks != shadow.blocks) {
    for (std::size_t i = 0; i < actual.blocks.size(); ++i) {
      if (!(actual.blocks[i] == shadow.blocks[i])) {
        throw std::runtime_error(label + ": physical block " +
            std::to_string(i) + " differs (epoch " +
            std::to_string(actual.blocks[i].last_access_epoch) + "/" +
            std::to_string(shadow.blocks[i].last_access_epoch) + ", refs " +
            std::to_string(actual.blocks[i].reference_count) + "/" +
            std::to_string(shadow.blocks[i].reference_count) + ", key " +
            std::to_string(actual.blocks[i].prefix_key.has_value()) + "/" +
            std::to_string(shadow.blocks[i].prefix_key.has_value()) +
            ", provenance " +
            std::to_string(actual.blocks[i].provenance
                ? static_cast<int>(*actual.blocks[i].provenance) : -1) + "/" +
            std::to_string(shadow.blocks[i].provenance
                ? static_cast<int>(*shadow.blocks[i].provenance) : -1) +
            ", keyeq " + std::to_string(
                actual.blocks[i].prefix_key && shadow.blocks[i].prefix_key &&
                *actual.blocks[i].prefix_key == *shadow.blocks[i].prefix_key) +
            ", parent " + std::to_string(actual.blocks[i].prefix_key
                ? actual.blocks[i].prefix_key->parent_hash : 0) + "/" +
            std::to_string(shadow.blocks[i].prefix_key
                ? shadow.blocks[i].prefix_key->parent_hash : 0) +
            ", hash " + std::to_string(actual.blocks[i].prefix_key
                ? actual.blocks[i].prefix_key->block_hash : 0) + "/" +
            std::to_string(shadow.blocks[i].prefix_key
                ? shadow.blocks[i].prefix_key->block_hash : 0) + ")");
      }
    }
  }
  require(actual.free_ids == shadow.free && actual.tables == shadow.tables,
          label + ": free IDs or request tables differ");
  require(actual.buckets == shadow.buckets, label + ": collision buckets differ");
  require(actual.canonical == shadow.canonical,
          label + ": canonical exact-key mapping differs");
  require(actual.allocated_blocks == shadow.derived_allocated &&
              actual.free_blocks == shadow.derived_free &&
              actual.represented_tokens == shadow.represented &&
              actual.fragmentation == shadow.derived_allocated * shadow.block_size -
                                          shadow.represented &&
              actual.utilization == shadow.derived_utilization &&
              actual.peak_allocated == shadow.peak &&
              actual.peak_utilization == shadow.derived_peak_utilization &&
              actual.cached_blocks == shadow.derived_cached &&
              actual.referenced_shared_blocks == shadow.derived_shared &&
              actual.allocation_failures == shadow.failures &&
              actual.access_epoch == shadow.epoch,
          label + ": physical accounting or epochs differ");
  require(llm_lab::serving::test::same_metrics(actual.metrics, shadow.metrics),
          label + ": prefix metrics differ");
  require(actual.metrics.matched_token_count == shadow.metrics.matched_token_count &&
              actual.metrics.total_cache_eligible_prompt_tokens_looked_up ==
                  shadow.metrics.total_cache_eligible_prompt_tokens_looked_up &&
              actual.hit_rate == shadow.metrics.prefix_token_hit_rate(),
          label + ": prefix hit-rate inputs or result differ");
  require(actual.provenance.size() == shadow.provenance.size(),
          label + ": provenance size differs");
  for (const auto& entry : shadow.provenance) {
    const auto found = actual.provenance.find(entry.first);
    require(found != actual.provenance.end(), label + ": missing provenance");
    const auto& a = found->second; const auto& p = entry.second;
    require(a.request_id == p.request_id &&
                a.publication_eligible == p.eligible &&
                a.exact_prompt_tokens == p.tokens &&
                a.total_prompt_token_count == p.total &&
                a.cache_eligible_full_prompt_blocks == p.full &&
                a.matched_shared_prefix_blocks == p.matched &&
                a.computed_full_begin == p.computed_begin &&
                a.computed_full_end == p.computed_end &&
                a.private_prompt_tail_position == p.tail &&
                a.decode_block_start_position == p.decode_start &&
                a.publication_completed == p.published,
            label + ": immutable prompt provenance differs");
  }
  require(manager.validate_invariants(), label + ": production invariants failed");
}

void compare_lookup(const PrefixLookupResult& actual,
                    const PrefixLookupResult& expected,
                    const std::string& label) {
  require(actual.kind == expected.kind &&
              actual.matched_block_count == expected.matched_block_count &&
              actual.matched_token_count == expected.matched_token_count &&
              actual.physical_block_ids == expected.physical_block_ids &&
              actual.next_parent_hash == expected.next_parent_hash &&
              actual.eligible_token_count == expected.eligible_token_count &&
              actual.collision_verification_count ==
                  expected.collision_verification_count,
          label + ": lookup result differs");
}

void exact_acquire(KVCacheManager& manager, ShadowCache& shadow, RequestId id,
                   const std::vector<std::int32_t>& tokens,
                   const std::string& label) {
  const PrefixLookupResult expected = shadow.lookup(tokens);
  const PrefixLookupResult actual = manager.find_longest_cached_prefix(tokens);
  compare_lookup(actual, expected, label);
  const auto reservation = shadow.reserve(
      shadow.required(tokens.size() - expected.matched_token_count),
      expected.physical_block_ids);
  const auto evictions_before = manager.prefix_cache_metrics().eviction_count;
  shadow.acquire(id, tokens, expected);
  manager.allocate_prompt_with_prefix(id, tokens, actual);
  const auto table = manager.block_table(id);
  require(std::vector<std::size_t>(
              table.begin() + static_cast<std::ptrdiff_t>(actual.matched_block_count),
              table.end()) == reservation.second &&
              manager.prefix_cache_metrics().eviction_count - evictions_before ==
                  reservation.first.size(),
          label + ": exact allocation or eviction prediction differs");
  compare_shadow(manager, shadow, label);
}

void publish_both(KVCacheManager& manager, ShadowCache& shadow, RequestId id,
                  const std::string& label) {
  shadow.publish(id); manager.insert_completed_prompt_blocks(id);
  compare_shadow(manager, shadow, label);
}

void release_both(KVCacheManager& manager, ShadowCache& shadow, RequestId id,
                  const std::string& label) {
  shadow.release(id); manager.release_request(id);
  compare_shadow(manager, shadow, label);
}

void append_both(KVCacheManager& manager, ShadowCache& shadow, RequestId id,
                 const std::string& label) {
  const auto reservation = shadow.reserve(
      shadow.tables.at(id).empty() ||
              shadow.blocks[shadow.tables.at(id).back()].prefix_key ||
              shadow.blocks[shadow.tables.at(id).back()].reference_count > 1 ||
              shadow.blocks[shadow.tables.at(id).back()].valid_token_count ==
                  shadow.block_size
          ? 1 : 0);
  const auto before = manager.prefix_cache_metrics().eviction_count;
  shadow.append(id); manager.append_decode_token(id);
  require(manager.prefix_cache_metrics().eviction_count - before ==
              reservation.first.size(),
          label + ": decode eviction prediction differs");
  compare_shadow(manager, shadow, label);
}

void test_independent_shadow_forced_operations() {
  KVCacheManager manager(KVCacheConfig(5, 2), enabled());
  ShadowCache shadow(5, 2);
  const std::vector<std::int32_t> chain{1, 2, 3, 4};
  exact_acquire(manager, shadow, {1}, chain, "initial exact miss");
  publish_both(manager, shadow, {1}, "initial canonical insertion");
  release_both(manager, shadow, {1}, "initial release to cached");
  FixCTestAccess::tie_cached_epochs(manager, manager.access_epoch());
  for (auto& block : shadow.blocks)
    if (block.state == PhysicalKVBlockState::Cached)
      block.last_access_epoch = shadow.epoch;
  require(manager.eligible_eviction_order() == shadow.lru() &&
              shadow.lru() == std::vector<std::size_t>({0, 1}),
          "shadow epoch tie did not use block ID tie-break");
  compare_shadow(manager, shadow, "epoch and block-ID LRU tie");

  exact_acquire(manager, shadow, {2}, chain, "single full hit");
  exact_acquire(manager, shadow, {3}, chain, "two-request shared hit");
  require(manager.physical_blocks()[0].reference_count == 2 &&
              shadow.blocks[0].reference_count == 2,
          "shared hit did not reach reference count two");
  release_both(manager, shadow, {2}, "release one shared reference");
  release_both(manager, shadow, {3}, "release final shared reference");

  exact_acquire(manager, shadow, {4}, {1, 2, 9}, "partial hit private tail");
  append_both(manager, shadow, {4}, "decode fills private tail");
  append_both(manager, shadow, {4}, "private decode allocation");
  append_both(manager, shadow, {4}, "decode-created full block remains private");
  require(!manager.physical_blocks()[manager.block_table({4}).back()].prefix_key &&
              manager.physical_blocks()[manager.block_table({4}).back()].provenance ==
                  KVBlockProvenance::PrivateDecode,
          "decode-created full block became shared or published");
  release_both(manager, shadow, {4}, "release partial/decode request");

  const std::vector<std::int32_t> duplicate{5, 6};
  const auto lookup5 = manager.find_longest_cached_prefix(duplicate);
  const auto lookup6 = manager.find_longest_cached_prefix(duplicate);
  const auto shadow5 = shadow.lookup(duplicate);
  const auto shadow6 = shadow.lookup(duplicate);
  compare_lookup(lookup5, shadow5, "first same-plan duplicate lookup");
  compare_lookup(lookup6, shadow6, "second same-plan duplicate lookup");
  shadow.acquire({5}, duplicate, shadow5);
  manager.allocate_prompt_with_prefix({5}, duplicate, lookup5);
  shadow.acquire({6}, duplicate, shadow6);
  manager.allocate_prompt_with_prefix({6}, duplicate, lookup6);
  compare_shadow(manager, shadow, "independent same-plan misses");
  publish_both(manager, shadow, {5}, "canonical duplicate winner");
  publish_both(manager, shadow, {6}, "noncanonical duplicate publication");
  const std::size_t duplicate_id = manager.block_table({6})[0];
  require(!manager.physical_blocks()[duplicate_id].prefix_key,
          "noncanonical duplicate did not remain private");
  release_both(manager, shadow, {6}, "private duplicate release to free");
  release_both(manager, shadow, {5}, "canonical duplicate release to cached");

  exact_acquire(manager, shadow, {7}, chain, "referenced eviction protection");
  const auto count_reservation = shadow.reserve(shadow.required(6));
  shadow.allocate_count({8}, 6); manager.allocate_prompt({8}, 6);
  const auto protected_table = manager.block_table({7});
  require(count_reservation.first.size() == 1 &&
              std::find(protected_table.begin(), protected_table.end(),
                        count_reservation.first[0]) == protected_table.end(),
          "referenced cache block was chosen for eviction");
  compare_shadow(manager, shadow, "deterministic LRU and lowest-ID reuse");
  release_both(manager, shadow, {8}, "count-only private release");
  release_both(manager, shadow, {7}, "protected shared release");

  KVCacheManager disabled(KVCacheConfig(3, 2), PrefixCacheConfig(false));
  ShadowCache disabled_shadow(3, 2, false);
  disabled_shadow.allocate_count({1}, 3);
  disabled.allocate_prompt({1}, 3);
  compare_shadow(disabled, disabled_shadow,
                 "disabled exact-token S4 private path");
}

void test_independent_collision_and_ancestor_shadow() {
  const auto constant = [](const TokenBlockKey&) { return UINT64_C(7); };
  KVCacheManager manager(KVCacheConfig(3, 2), enabled(), constant);
  ShadowCache shadow(3, 2, true, true);
  const std::vector<std::int32_t> chain{1, 2, 3, 4};
  exact_acquire(manager, shadow, {1}, chain, "collision chain miss");
  publish_both(manager, shadow, {1}, "collision chain publish");
  release_both(manager, shadow, {1}, "collision chain cached");
  exact_acquire(manager, shadow, {2}, {8, 9}, "collision mismatch acquire");
  publish_both(manager, shadow, {2}, "unrelated collision publish");
  release_both(manager, shadow, {2}, "unrelated collision cached");
  compare_lookup(manager.find_longest_cached_prefix({8, 9}),
                 shadow.lookup({8, 9}), "exact key among collision candidates");

  const auto victim = shadow.lru().front();
  shadow.evict_ids({victim});
  require(manager.evict_unused_blocks(1) == std::vector<std::size_t>{victim},
          "parent eviction ID differs from shadow");
  compare_shadow(manager, shadow, "parent evicted while child remains");
  const auto unreachable = shadow.lookup(chain);
  compare_lookup(manager.find_longest_cached_prefix(chain), unreachable,
                 "child unreachable after parent eviction");
  require(unreachable.matched_block_count == 0 &&
              shadow.blocks[1].state == PhysicalKVBlockState::Cached,
          "child was reused or removed with missing parent");
  exact_acquire(manager, shadow, {3}, {1, 2}, "exact parent recomputation");
  publish_both(manager, shadow, {3}, "stable parent republish");
  release_both(manager, shadow, {3}, "recomputed parent cached");
  const auto restored = shadow.lookup(chain);
  compare_lookup(manager.find_longest_cached_prefix(chain), restored,
                 "child reachable after parent recomputation");
  require(restored.physical_block_ids == std::vector<std::size_t>({0, 1}) &&
              manager.find_longest_cached_prefix({8, 9}).physical_block_ids ==
                  std::vector<std::size_t>{2},
          "collision cleanup corrupted child or unrelated exact key");
}

void require_shadow_mismatch(const KVCacheManager& manager,
                             const ShadowCache& shadow,
                             const std::string& label) {
  require_throws<std::runtime_error>(
      [&] { compare_shadow(manager, shadow, label); },
      label + ": comparison accepted a deliberate mismatch");
}

void test_shadow_comparison_sensitivity() {
  KVCacheManager manager(KVCacheConfig(3, 2), enabled());
  ShadowCache shadow(3, 2);
  exact_acquire(manager, shadow, {1}, {1, 2}, "sensitivity seed");
  publish_both(manager, shadow, {1}, "sensitivity publish");
  release_both(manager, shadow, {1}, "sensitivity cached");
  exact_acquire(manager, shadow, {2}, {1, 2}, "sensitivity shared");

  ShadowCache canonical = shadow;
  canonical.canonical.clear();
  require_shadow_mismatch(manager, canonical, "canonical-map sensitivity");

  ShadowCache provenance = shadow;
  provenance.provenance.at({2}).request_id = {999};
  require_shadow_mismatch(manager, provenance, "provenance-ID sensitivity");

  ShadowCache accounting = shadow;
  ++accounting.derived_allocated;
  require_shadow_mismatch(manager, accounting, "derived-accounting sensitivity");

  ShadowCache cached = shadow;
  ++cached.derived_cached;
  require_shadow_mismatch(manager, cached, "cached-count sensitivity");

  ShadowCache shared = shadow;
  ++shared.derived_shared;
  require_shadow_mismatch(manager, shared, "shared-count sensitivity");
}

void test_hash_function_snapshot_identity() {
  KVCacheManager stable(KVCacheConfig(2, 2), enabled());
  KVCacheManager constant(KVCacheConfig(2, 2), enabled(),
      [](const TokenBlockKey&) { return UINT64_C(7); });
  const ManagerSnapshot stable_snapshot = FixCTestAccess::manager_snapshot(stable);
  const ManagerSnapshot constant_snapshot = FixCTestAccess::manager_snapshot(constant);
  require(!llm_lab::serving::test::same_manager(stable_snapshot,
                                                constant_snapshot),
          "different hash behavior produced equal empty-manager snapshots");

  FixCTestAccess::replace_hash(stable,
      [](const TokenBlockKey&) { return UINT64_C(7); });
  require(!llm_lab::serving::test::same_manager(
              stable_snapshot, FixCTestAccess::manager_snapshot(stable)),
          "friend hash replacement did not change snapshot identity");
}

void test_prefix_capacity_failure_metric_and_taxonomy() {
  KVCacheManager manager(KVCacheConfig(2, 2), enabled());
  const std::vector<std::int32_t> prefix{1, 2};
  manager.allocate_prompt_with_prefix(
      {1}, prefix, manager.find_longest_cached_prefix(prefix));
  manager.insert_completed_prompt_blocks({1});
  manager.release_request({1});
  manager.allocate_prompt({2}, 2);
  manager.allocate_prompt_with_prefix(
      {3}, prefix, manager.find_longest_cached_prefix(prefix));

  const std::vector<std::int32_t> with_tail{1, 2, 3};
  const PrefixLookupResult hit = manager.find_longest_cached_prefix(with_tail);
  const ManagerSnapshot before = FixCTestAccess::manager_snapshot(manager);
  require_throws_exactly<KVCapacityError>(
      [&] { manager.allocate_prompt_with_prefix({4}, with_tail, hit); },
      "valid prefix-suffix exhaustion did not throw KVCapacityError");
  ManagerSnapshot expected = before;
  ++expected.allocation_failures;
  require(llm_lab::serving::test::same_manager(
              expected, FixCTestAccess::manager_snapshot(manager)),
          "prefix-suffix exhaustion changed state beyond its failure counter");
  require(manager.find_longest_cached_prefix(prefix).matched_block_count == 1 &&
              manager.validate_invariants(),
          "manager was unusable after prefix-suffix exhaustion");

  FixCTestAccess::set_allocation_failures(
      manager, std::numeric_limits<std::uint64_t>::max());
  const ManagerSnapshot overflow_before = FixCTestAccess::manager_snapshot(manager);
  require_throws_exactly<std::overflow_error>(
      [&] { manager.allocate_prompt_with_prefix({4}, with_tail, hit); },
      "prefix allocation-failure counter overflow had the wrong category");
  require(llm_lab::serving::test::same_manager(
              overflow_before, FixCTestAccess::manager_snapshot(manager)) &&
              manager.validate_invariants(),
          "allocation-failure counter overflow changed manager state");

  require(!throws_exactly<KVCapacityError>([] {
            throw std::overflow_error("not capacity");
          }),
          "exact exception helper accepted overflow as capacity");
  KVCacheManager full(KVCacheConfig(1, 2));
  full.allocate_prompt({10}, 2);
  require(throws_exactly<KVCapacityError>(
              [&] { full.allocate_prompt({11}, 1); }),
          "exact exception helper rejected a real capacity failure");

  SimulatedBackend backend(SimulatedBackendConfig{});
  for (SchedulingPolicy policy :
       {SchedulingPolicy::DecodeFirst, SchedulingPolicy::FcfsMixed}) {
    ContinuousBatchingEngine engine(backend, ContinuousBatchingConfig(
        1, 2, policy, KVCacheConfig(1, 2), enabled()));
    engine.submit_request(Request::count_only({1}, 0, 2, 4));
    require(engine.run_next_iteration().made_progress(),
            "planner-deferral capacity fixture failed");
    engine.submit_request(Request::exact_tokens(
        {2}, engine.clock().now_us(), {9}, 1));
    require(engine.run_next_iteration().is_stalled() &&
                engine.kv_cache().allocation_failure_count() == 0,
            "planner KVCapacity deferral counted as allocation failure");
  }
}

void test_independent_fixed_seed_sequence() {
  constexpr std::uint64_t seed = UINT64_C(0x5eedc0de);
  std::uint64_t random = seed;
  KVCacheManager manager(KVCacheConfig(12, 2), enabled());
  ShadowCache shadow(12, 2);
  std::set<RequestId> active;
  std::uint64_t next_id = 1;
  for (std::size_t step = 0; step < 96; ++step) {
    random = random * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    const unsigned op = static_cast<unsigned>((random >> 61U) & 3U);
    const std::string label = "fixed seed 0x5eedc0de step " + std::to_string(step);
    if ((op == 0 || active.empty()) && next_id <= 40) {
      const RequestId id{next_id++};
      const bool exact_prompt = ((random >> 17U) & 1U) != 0;
      const std::uint64_t length = 1 + ((random >> 20U) % 4U);
      if (exact_prompt) {
        std::vector<std::int32_t> tokens(static_cast<std::size_t>(length));
        for (std::size_t i = 0; i < tokens.size(); ++i)
          tokens[i] = static_cast<std::int32_t>((random >> (i * 7U)) & 7U);
        const auto expected = shadow.lookup(tokens);
        const auto actual = manager.find_longest_cached_prefix(tokens);
        compare_lookup(actual, expected, label + " lookup");
        auto evictable = shadow.lru();
        evictable.erase(std::remove_if(evictable.begin(), evictable.end(),
            [&](std::size_t block_id) {
              return std::find(expected.physical_block_ids.begin(),
                               expected.physical_block_ids.end(), block_id) !=
                     expected.physical_block_ids.end();
            }), evictable.end());
        const std::size_t needed = shadow.required(length - expected.matched_token_count);
        if (needed > shadow.free.size() + evictable.size()) {
          require_throws_exactly<KVCapacityError>(
              [&] { manager.allocate_prompt_with_prefix(id, tokens, actual); },
              label + ": predicted exact capacity failure had the wrong category");
          ++shadow.failures;
        } else {
          shadow.acquire(id, tokens, expected);
          manager.allocate_prompt_with_prefix(id, tokens, actual);
          if (tokens.size() >= 2 && ((random >> 13U) & 1U)) {
            shadow.publish(id); manager.insert_completed_prompt_blocks(id);
          }
          active.insert(id);
        }
      } else {
        const std::size_t needed = shadow.required(length);
        if (needed > shadow.free.size() + shadow.lru().size()) {
          require_throws_exactly<KVCapacityError>(
              [&] { manager.allocate_prompt(id, length); },
              label + ": predicted count capacity failure had the wrong category");
          ++shadow.failures;
        } else {
          shadow.allocate_count(id, length); manager.allocate_prompt(id, length);
          active.insert(id);
        }
      }
    } else if (op == 1 && !active.empty()) {
      const RequestId id = *std::next(active.begin(),
          static_cast<std::ptrdiff_t>((random >> 16U) % active.size()));
      const auto& table = shadow.tables.at(id);
      const bool needs_block = table.empty() ||
          shadow.blocks[table.back()].prefix_key.has_value() ||
          shadow.blocks[table.back()].reference_count > 1 ||
          shadow.blocks[table.back()].valid_token_count == shadow.block_size;
      if (needs_block && shadow.free.empty() && shadow.lru().empty()) {
        require_throws_exactly<KVCapacityError>(
            [&] { manager.append_decode_token(id); },
            label + ": predicted append capacity failure had the wrong category");
        ++shadow.failures;
      } else {
        shadow.append(id); manager.append_decode_token(id);
      }
    } else if (!active.empty()) {
      const RequestId id = *std::next(active.begin(),
          static_cast<std::ptrdiff_t>((random >> 16U) % active.size()));
      shadow.release(id); manager.release_request(id); active.erase(id);
    }
    compare_shadow(manager, shadow, label);
  }
  const ManagerSnapshot before = FixCTestAccess::manager_snapshot(manager);
  require_throws_exactly<std::out_of_range>(
      [&] { manager.release_request({999}); },
      "fixed-seed unknown release had the wrong category");
  require(llm_lab::serving::test::same_manager(
              before, FixCTestAccess::manager_snapshot(manager)),
          "fixed-seed invalid operation changed state");

  while (!active.empty()) {
    const RequestId id = *active.begin();
    shadow.release(id); manager.release_request(id); active.erase(id);
  }
  shadow.allocate_count({500}, 1); manager.allocate_prompt({500}, 1);
  const ManagerSnapshot invalid_before = FixCTestAccess::manager_snapshot(manager);
  require_throws_exactly<std::invalid_argument>(
      [&] { manager.allocate_prompt({500}, 1); },
      "fixed-seed duplicate request had the wrong category");
  require_throws_exactly<std::out_of_range>(
      [&] { manager.append_decode_token({999}); },
      "fixed-seed unknown append had the wrong category");
  require_throws_exactly<std::logic_error>(
      [&] { manager.insert_completed_prompt_blocks({500}); },
      "fixed-seed invalid publication had the wrong category");
  require(llm_lab::serving::test::same_manager(
              invalid_before, FixCTestAccess::manager_snapshot(manager)),
          "fixed-seed invalid operation matrix changed state");
  compare_shadow(manager, shadow, "fixed-seed final invalid operations");
}

KVCacheManager manager_fixture() {
  KVCacheManager manager(KVCacheConfig(6, 2), enabled());
  const std::vector<std::int32_t> prefix{1, 2};
  manager.allocate_prompt_with_prefix(
      {1}, prefix, manager.find_longest_cached_prefix(prefix));
  manager.insert_completed_prompt_blocks({1});
  manager.release_request({1});
  manager.allocate_prompt_with_prefix(
      {2}, {8, 9}, manager.find_longest_cached_prefix({8, 9}));
  manager.insert_completed_prompt_blocks({2});
  manager.release_request({2});
  const std::vector<std::int32_t> active{1, 2, 3};
  manager.allocate_prompt_with_prefix(
      {3}, active, manager.find_longest_cached_prefix(active));
  manager.append_decode_token({3});
  manager.append_decode_token({3});
  require(manager.validate_invariants(), "manager fixture is invalid");
  return manager;
}

template <class Exception, class Operation>
void verify_manager_failure(KVCacheFailurePoint point, Operation operation,
                            const std::string& label) {
  KVCacheManager manager = manager_fixture();
  FixCTestAccess::set_failure(manager, point);
  const ManagerSnapshot before = FixCTestAccess::manager_snapshot(manager);
  require_throws<Exception>([&] { operation(manager); },
                            label + ": wrong success/failure result");
  require(llm_lab::serving::test::same_manager(
              before, FixCTestAccess::manager_snapshot(manager)),
          label + ": complete manager snapshot changed");
  require(manager.validate_invariants(), label + ": invariants no longer pass");
  require(manager.find_longest_cached_prefix({1, 2}).matched_block_count == 1,
          label + ": manager is not usable after synchronous failure");
}

void test_complete_manager_failure_matrix() {
  const auto hit = [](KVCacheManager& m) {
    const std::vector<std::int32_t> tokens{1, 2, 7};
    m.allocate_prompt_with_prefix({20}, tokens,
                                  m.find_longest_cached_prefix(tokens));
  };
  const auto miss = [](KVCacheManager& m) {
    const std::vector<std::int32_t> tokens{7};
    m.allocate_prompt_with_prefix({20}, tokens,
                                  m.find_longest_cached_prefix(tokens));
  };
  for (KVCacheFailurePoint point : {
           KVCacheFailurePoint::CacheLookupCounter,
           KVCacheFailurePoint::CacheHitCounter,
           KVCacheFailurePoint::EligibleTokenCounter,
           KVCacheFailurePoint::MatchedBlockCounter,
           KVCacheFailurePoint::MatchedTokenCounter,
           KVCacheFailurePoint::ReusedRequestCounter,
           KVCacheFailurePoint::SavedPrefillCounter,
           KVCacheFailurePoint::ReferenceCounter,
           KVCacheFailurePoint::AccessEpoch,
           KVCacheFailurePoint::RepresentedTokenArithmetic})
    verify_manager_failure<std::overflow_error>(point, hit,
        "hit-side manager failure " + std::to_string(static_cast<int>(point)));
  for (KVCacheFailurePoint point : {
           KVCacheFailurePoint::CacheMissCounter,
           KVCacheFailurePoint::CollisionCounter})
    verify_manager_failure<std::overflow_error>(point, miss,
        "miss-side manager failure " + std::to_string(static_cast<int>(point)));
  verify_manager_failure<std::overflow_error>(
      KVCacheFailurePoint::EvictionCounter,
      [](KVCacheManager& m) { m.allocate_prompt({20}, 6); },
      "eviction counter overflow");
  verify_manager_failure<std::overflow_error>(
      KVCacheFailurePoint::AllocationFailureCounter,
      [](KVCacheManager& m) { m.allocate_prompt({20}, 100); },
      "allocation failure counter overflow");

  KVCacheManager invalid = manager_fixture();
  const ManagerSnapshot invalid_before = FixCTestAccess::manager_snapshot(invalid);
  require_throws<std::logic_error>(
      [&] { invalid.allocate_prompt_exact({30}, 1, {}, {99}); },
      "invalid planned physical ID was accepted");
  require(llm_lab::serving::test::same_manager(
              invalid_before, FixCTestAccess::manager_snapshot(invalid)) &&
              invalid.validate_invariants(),
          "invalid planned-ID rejection changed manager state");

  KVCacheManager conflict = manager_fixture();
  const std::vector<std::int32_t> fresh{5, 6};
  conflict.allocate_prompt_with_prefix(
      {40}, fresh, conflict.find_longest_cached_prefix(fresh));
  FixCTestAccess::set_failure(
      conflict, KVCacheFailurePoint::CanonicalIndexConflict);
  const ManagerSnapshot conflict_before = FixCTestAccess::manager_snapshot(conflict);
  require_throws<std::logic_error>(
      [&] { conflict.insert_completed_prompt_blocks({40}); },
      "canonical-index conflict was accepted");
  require(llm_lab::serving::test::same_manager(
              conflict_before, FixCTestAccess::manager_snapshot(conflict)) &&
              conflict.validate_invariants(),
          "canonical-index conflict changed manager state");

  KVCacheManager publication_epoch = manager_fixture();
  publication_epoch.allocate_prompt_with_prefix(
      {41}, fresh, publication_epoch.find_longest_cached_prefix(fresh));
  FixCTestAccess::set_failure(
      publication_epoch, KVCacheFailurePoint::PublicationAccessEpoch);
  const ManagerSnapshot publication_before =
      FixCTestAccess::manager_snapshot(publication_epoch);
  require_throws<std::overflow_error>(
      [&] { publication_epoch.insert_completed_prompt_blocks({41}); },
      "publication access-epoch overflow was missed");
  require(llm_lab::serving::test::same_manager(
              publication_before,
              FixCTestAccess::manager_snapshot(publication_epoch)) &&
              publication_epoch.validate_invariants(),
          "publication access-epoch overflow changed manager state");
}

SimulatedBackend engine_backend() {
  SimulatedBackendConfig c;
  c.batch_base_us = 1;
  return SimulatedBackend(c);
}

ContinuousBatchingEngine engine_fixture(const SimulatedBackend& backend) {
  ContinuousBatchingEngine engine(backend, ContinuousBatchingConfig(
      3, 6, SchedulingPolicy::DecodeFirst, KVCacheConfig(4, 2), enabled()));
  engine.submit_request(Request::exact_tokens({1}, 0, {1, 2}, 0));
  require(engine.run() == RunResult::Completed, "engine prefix seed failed");
  engine.submit_request(Request::exact_tokens(
      {2}, engine.clock().now_us(), {8, 9}, 0));
  require(engine.run() == RunResult::Completed, "engine eviction seed failed");
  engine.submit_request(Request::exact_tokens(
      {3}, engine.clock().now_us(), {1, 2, 3}, 6));
  require(engine.run_next_iteration().made_progress(), "active prefill failed");
  require(engine.run_next_iteration().made_progress(), "tail decode failed");
  require(engine.run_next_iteration().made_progress(), "private decode failed");
  engine.submit_request(Request::exact_tokens(
      {90}, engine.clock().now_us() + 100, {6}, 1));
  return engine;
}

void verify_post_failure_rejection(ContinuousBatchingEngine& engine,
                                   std::int64_t raw_clock) {
  require_throws<std::logic_error>([&] { (void)engine.run(); }, "run accepted failed engine");
  require_throws<std::logic_error>([&] { (void)engine.run_next_iteration(); }, "iteration accepted failed engine");
  require_throws<std::logic_error>([&] { engine.submit_request(Request::count_only({99}, raw_clock, 0, 0)); }, "submit accepted failed engine");
  require_throws<std::logic_error>([&] { engine.cancel_request({3}); }, "cancel accepted failed engine");
  require_throws<std::logic_error>([&] { (void)engine.request({3}); }, "request result accepted failed engine");
  require_throws<std::logic_error>([&] { (void)engine.requests(); }, "requests accepted failed engine");
  require_throws<std::logic_error>([&] { (void)engine.plan_trace(); }, "trace accepted failed engine");
  require_throws<std::logic_error>([&] { (void)engine.statistics(); }, "statistics accepted failed engine");
  require_throws<std::logic_error>([&] { (void)engine.clock(); }, "clock accepted failed engine");
  require_throws<std::logic_error>([&] { (void)engine.kv_cache(); }, "KV diagnostics accepted failed engine");
}

template <class Exception, class Configure, class Submit>
void verify_engine_failure(const std::string& label, Configure configure,
                           Submit submit) {
  const SimulatedBackend backend = engine_backend();
  ContinuousBatchingEngine engine = engine_fixture(backend);
  submit(engine);
  configure(engine);
  const EngineSnapshot before = FixCTestAccess::engine_snapshot(engine);
  require_throws<Exception>([&] { (void)engine.run_next_iteration(); },
                            label + ": wrong failure class");
  require(engine.failed(), label + ": engine did not become terminally failed");
  const EngineSnapshot after = FixCTestAccess::engine_snapshot(engine);
  require(llm_lab::serving::test::same_engine(before, after, false),
          label + ": live engine snapshot changed");
  require(after.trace.size() == before.trace.size(),
          label + ": failed trace was published");
  verify_post_failure_rejection(engine, before.clock);
}

void test_complete_engine_failure_matrix() {
  const auto hit_submit = [](ContinuousBatchingEngine& e) {
    e.submit_request(Request::exact_tokens({4}, e.clock().now_us(), {1, 2, 4}, 1));
  };
  for (KVCacheFailurePoint point : {
           KVCacheFailurePoint::CacheLookupCounter,
           KVCacheFailurePoint::CacheHitCounter,
           KVCacheFailurePoint::EligibleTokenCounter,
           KVCacheFailurePoint::MatchedBlockCounter,
           KVCacheFailurePoint::MatchedTokenCounter,
           KVCacheFailurePoint::ReusedRequestCounter,
           KVCacheFailurePoint::SavedPrefillCounter,
           KVCacheFailurePoint::ReferenceCounter,
           KVCacheFailurePoint::EvictionCounter,
           KVCacheFailurePoint::AccessEpoch,
           KVCacheFailurePoint::RepresentedTokenArithmetic}) {
    verify_engine_failure<std::overflow_error>(
        "prepared manager failure " + std::to_string(static_cast<int>(point)),
        [point](ContinuousBatchingEngine& e) {
          FixCTestAccess::set_prepared_failure(e, point);
        }, hit_submit);
  }
  for (KVCacheFailurePoint point : {
           KVCacheFailurePoint::CacheMissCounter,
           KVCacheFailurePoint::CollisionCounter}) {
    verify_engine_failure<std::overflow_error>(
        "prepared miss failure " + std::to_string(static_cast<int>(point)),
        [point](ContinuousBatchingEngine& e) {
          FixCTestAccess::set_prepared_failure(e, point);
        }, [](ContinuousBatchingEngine& e) {
          e.submit_request(Request::exact_tokens(
              {4}, e.clock().now_us(), {7}, 1));
        });
  }
  verify_engine_failure<std::logic_error>("canonical-index conflict",
      [](ContinuousBatchingEngine& e) {
        FixCTestAccess::set_prepared_failure(
            e, KVCacheFailurePoint::CanonicalIndexConflict);
      }, [](ContinuousBatchingEngine& e) {
        e.submit_request(Request::exact_tokens(
            {4}, e.clock().now_us(), {1, 2, 4, 5}, 1));
      });
  verify_engine_failure<std::overflow_error>("publication access epoch overflow",
      [](ContinuousBatchingEngine& e) {
        FixCTestAccess::set_prepared_failure(
            e, KVCacheFailurePoint::PublicationAccessEpoch);
      }, [](ContinuousBatchingEngine& e) {
        e.submit_request(Request::exact_tokens(
            {4}, e.clock().now_us(), {1, 2, 4, 5}, 1));
      });
  verify_engine_failure<std::overflow_error>("scheduling statistics overflow",
      [](ContinuousBatchingEngine& e) {
        FixCTestAccess::statistics(e).scheduling_iteration_count =
            std::numeric_limits<std::uint64_t>::max();
      }, hit_submit);
  verify_engine_failure<std::overflow_error>("completed request counter overflow",
      [](ContinuousBatchingEngine& e) {
        FixCTestAccess::statistics(e).completed_request_count =
            std::numeric_limits<std::uint64_t>::max();
      }, [](ContinuousBatchingEngine& e) {
        e.submit_request(Request::exact_tokens({4}, e.clock().now_us(), {1, 2}, 0));
      });
  verify_engine_failure<std::overflow_error>("KV deferral counter overflow",
      [](ContinuousBatchingEngine& e) {
        FixCTestAccess::statistics(e).kv_capacity_deferral_count =
            std::numeric_limits<std::uint64_t>::max();
      }, [](ContinuousBatchingEngine& e) {
        e.submit_request(Request::count_only({4}, e.clock().now_us(), 4, 1));
      });
  verify_engine_failure<std::overflow_error>("iteration number overflow",
      [](ContinuousBatchingEngine& e) {
        FixCTestAccess::set_next_iteration(
            e, std::numeric_limits<std::uint64_t>::max());
      }, hit_submit);
  verify_engine_failure<std::runtime_error>("trace preparation failure",
      [](ContinuousBatchingEngine& e) { FixCTestAccess::set_trace_failure(e); },
      hit_submit);
  verify_engine_failure<std::logic_error>("planned/actual reservation mismatch",
      [](ContinuousBatchingEngine& e) { FixCTestAccess::set_reservation_failure(e); },
      hit_submit);

  const SimulatedBackend backend = engine_backend();
  ContinuousBatchingEngine timestamp = engine_fixture(backend);
  hit_submit(timestamp);
  FixCTestAccess::advance_clock(
      timestamp, std::numeric_limits<std::int64_t>::max());
  const EngineSnapshot before = FixCTestAccess::engine_snapshot(timestamp);
  require_throws<std::overflow_error>(
      [&] { (void)timestamp.run_next_iteration(); },
      "timestamp overflow after prepared mutations was missed");
  require(timestamp.failed() && llm_lab::serving::test::same_engine(
              before, FixCTestAccess::engine_snapshot(timestamp), false),
          "timestamp overflow published prepared state");
  verify_post_failure_rejection(timestamp, before.clock);
}

std::vector<RunResult> run_deterministic_workload(
    ContinuousBatchingEngine& engine) {
  std::vector<RunResult> results;
  const std::vector<std::int32_t> chain{1, 2, 3, 4};
  engine.submit_request(Request::exact_tokens({1}, 0, chain, 0));
  results.push_back(engine.run());
  engine.submit_request(Request::exact_tokens(
      {4}, engine.clock().now_us(), chain, 0));
  results.push_back(engine.run());

  const auto same_plan_time = engine.clock().now_us();
  engine.submit_request(Request::exact_tokens({2}, same_plan_time, {7, 8}, 0));
  engine.submit_request(Request::exact_tokens({3}, same_plan_time, {7, 8}, 0));
  results.push_back(engine.run());

  engine.submit_request(Request::exact_tokens(
      {10}, engine.clock().now_us(), {1, 2, 3, 4, 5}, 6));
  require(engine.run_next_iteration().made_progress(),
          "deterministic full/partial hit prefill failed");
  require(engine.run_next_iteration().made_progress(),
          "deterministic tail decode failed");
  require(engine.run_next_iteration().made_progress(),
          "deterministic private decode allocation failed");
  engine.cancel_request({10});

  engine.submit_request(Request::exact_tokens(
      {11}, engine.clock().now_us(), {1, 2, 9}, 0));
  results.push_back(engine.run());
  engine.submit_request(Request::count_only(
      {12}, engine.clock().now_us(), 6, 0));
  results.push_back(engine.run());

  engine.submit_request(Request::count_only(
      {20}, engine.clock().now_us(), 6, 4));
  require(engine.run_next_iteration().made_progress(),
          "deterministic stall resident prefill failed");
  engine.submit_request(Request::exact_tokens(
      {21}, engine.clock().now_us(), {9}, 1));
  results.push_back(engine.run());
  require(results.back() == RunResult::Stalled,
          "deterministic workload did not expose a stall");
  engine.cancel_request({20});
  results.push_back(engine.run());
  require(results.back() == RunResult::Completed,
          "deterministic workload did not recover");
  return results;
}

void test_complete_repeated_run_determinism() {
  const SimulatedBackend backend = engine_backend();
  for (SchedulingPolicy policy :
       {SchedulingPolicy::DecodeFirst, SchedulingPolicy::FcfsMixed}) {
    const ContinuousBatchingConfig config(
        3, 6, policy, KVCacheConfig(4, 2), enabled());
    ContinuousBatchingEngine first(backend, config);
    ContinuousBatchingEngine second(backend, config);
    const auto first_results = run_deterministic_workload(first);
    const auto second_results = run_deterministic_workload(second);
    require(first_results == second_results,
            "repeated cache-aware RunResult sequence differs");
    const EngineSnapshot a = FixCTestAccess::engine_snapshot(first);
    const EngineSnapshot b = FixCTestAccess::engine_snapshot(second);
    require(llm_lab::serving::test::same_engine(a, b),
            "complete repeated cache-aware engine snapshot differs");
    bool saw_full_hit = false;
    bool saw_partial_hit = false;
    bool saw_eviction = false;
    bool saw_same_plan_misses = false;
    for (const auto& trace : a.trace) {
      std::size_t misses = 0;
      for (const auto& work : trace.plan.prefill_work()) {
        saw_full_hit = saw_full_hit ||
            (work.matched_prefix_token_count == work.original_prompt_token_count &&
             work.original_prompt_token_count != 0);
        saw_partial_hit = saw_partial_hit ||
            (work.matched_prefix_token_count != 0 &&
             work.matched_prefix_token_count < work.original_prompt_token_count);
        saw_eviction = saw_eviction || !work.evicted_block_ids.empty();
        if (work.prefix_lookup_kind == PrefixLookupKind::Miss) ++misses;
      }
      saw_same_plan_misses = saw_same_plan_misses || misses >= 2;
    }
    require(saw_full_hit && saw_partial_hit && saw_eviction && saw_same_plan_misses &&
                a.statistics.cancelled_request_count >= 2 &&
                a.statistics.stalled_iteration_count >= 1,
            "deterministic workload missed required cache-aware coverage");
  }
}

}  // namespace

int main() {
  try {
    test_independent_shadow_forced_operations();
    test_independent_collision_and_ancestor_shadow();
    test_shadow_comparison_sensitivity();
    test_hash_function_snapshot_identity();
    test_prefix_capacity_failure_metric_and_taxonomy();
    test_independent_fixed_seed_sequence();
    test_complete_manager_failure_matrix();
    test_complete_engine_failure_matrix();
    test_complete_repeated_run_determinism();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "prefix cache Fix C transaction tests passed\n";
  return EXIT_SUCCESS;
}
