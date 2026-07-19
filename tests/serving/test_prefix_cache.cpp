#include "serving/continuous_batching.h"
#include "serving/kv_cache.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

using namespace llm_lab::serving;

namespace llm_lab::serving::test {
struct KVCacheManagerTestAccess {
  static void set_prefix_metrics(KVCacheManager& manager,
                                 PrefixCacheMetrics metrics) {
    manager.prefix_metrics_ = metrics;
  }
  static void set_access_epoch(KVCacheManager& manager, std::uint64_t value) {
    manager.access_epoch_ = value;
  }
  static void tie_cached_epochs(KVCacheManager& manager,
                                std::uint64_t value) {
    for (auto& block : manager.blocks_) {
      if (block.state == PhysicalKVBlockState::Cached)
        block.last_access_epoch = value;
    }
    manager.access_epoch_ = value;
  }
  static bool same_full_state(const KVCacheManager& a, const KVCacheManager& b) {
    const auto same_metrics = [](const PrefixCacheMetrics& x,
                                 const PrefixCacheMetrics& y) {
      return x.cache_lookup_count == y.cache_lookup_count &&
          x.cache_hit_lookup_count == y.cache_hit_lookup_count &&
          x.cache_miss_lookup_count == y.cache_miss_lookup_count &&
          x.matched_block_count == y.matched_block_count &&
          x.matched_token_count == y.matched_token_count &&
          x.total_cache_eligible_prompt_tokens_looked_up ==
              y.total_cache_eligible_prompt_tokens_looked_up &&
          x.reused_request_count == y.reused_request_count &&
          x.saved_prefill_token_count == y.saved_prefill_token_count &&
          x.collision_verification_count == y.collision_verification_count &&
          x.eviction_count == y.eviction_count;
    };
    const auto same_request_provenance = [](const auto& x, const auto& y) {
      if (x.size() != y.size()) return false;
      auto xi = x.begin(); auto yi = y.begin();
      for (; xi != x.end(); ++xi, ++yi) {
        const auto& p = xi->second; const auto& q = yi->second;
        if (xi->first != yi->first || p.request_id != q.request_id ||
            p.publication_eligible != q.publication_eligible ||
            p.exact_prompt_tokens != q.exact_prompt_tokens ||
            p.total_prompt_token_count != q.total_prompt_token_count ||
            p.cache_eligible_full_prompt_blocks != q.cache_eligible_full_prompt_blocks ||
            p.matched_shared_prefix_blocks != q.matched_shared_prefix_blocks ||
            p.computed_full_begin != q.computed_full_begin ||
            p.computed_full_end != q.computed_full_end ||
            p.private_prompt_tail_position != q.private_prompt_tail_position ||
            p.decode_block_start_position != q.decode_block_start_position ||
            p.publication_completed != q.publication_completed) return false;
      }
      return true;
    };
    return a.blocks_ == b.blocks_ && a.free_blocks_ == b.free_blocks_ &&
        a.block_tables_ == b.block_tables_ &&
        same_request_provenance(a.request_provenance_, b.request_provenance_) &&
        a.prefix_index_ == b.prefix_index_ && same_metrics(a.prefix_metrics_, b.prefix_metrics_) &&
        a.represented_token_count_ == b.represented_token_count_ &&
        a.peak_allocated_block_count_ == b.peak_allocated_block_count_ &&
        a.allocation_failure_count_ == b.allocation_failure_count_ &&
        a.access_epoch_ == b.access_epoch_;
  }
  static void corrupt_hash(KVCacheManager& m) { ++m.blocks_[0].prefix_key->block_hash; }
  static void corrupt_namespace(KVCacheManager& m) { m.blocks_[0].prefix_key->model_namespace += "x"; }
  static void corrupt_salt(KVCacheManager& m) { m.blocks_[0].prefix_key->cache_salt += "x"; }
  static void corrupt_token_length(KVCacheManager& m) { m.blocks_[0].prefix_key->token_ids.pop_back(); }
  static void corrupt_block_provenance(KVCacheManager& m) {
    m.blocks_[0].provenance = static_cast<KVBlockProvenance>(255);
  }
  static void corrupt_physical_state(KVCacheManager& m) {
    m.blocks_[2].state = static_cast<PhysicalKVBlockState>(255);
  }
  static void add_empty_bucket(KVCacheManager& m) { m.prefix_index_[999] = {}; }
  static void add_conflicting_duplicate_key(KVCacheManager& m) {
    auto key = *m.blocks_[0].prefix_key;
    ++key.block_hash;
    m.blocks_[2].prefix_key = key;
    m.prefix_index_[key.block_hash].push_back(2);
  }
  static void index_decode_block(KVCacheManager& m) {
    auto key = *m.blocks_[0].prefix_key; key.parent_hash ^= 1; key.block_hash = m.hash_function_(key);
    m.blocks_[2].prefix_key = key; m.prefix_index_[key.block_hash].push_back(2);
  }
  static void index_prompt_tail(KVCacheManager& m) {
    auto key = *m.blocks_[0].prefix_key; key.parent_hash ^= 2; key.block_hash = m.hash_function_(key);
    m.blocks_[1].prefix_key = key; m.prefix_index_[key.block_hash].push_back(1);
  }
  static void remove_key_metadata(KVCacheManager& m) { m.blocks_[0].prefix_key.reset(); }
  static void corrupt_request_boundary(KVCacheManager& m) {
    m.request_provenance_.begin()->second.decode_block_start_position = 1;
  }
  static void mismatch_table_provenance(KVCacheManager& m) {
    m.blocks_[1].provenance = KVBlockProvenance::PrivateDecode;
  }
};
struct ContinuousBatchingEngineTestAccess {
  static KVCacheManager& raw_kv_cache(ContinuousBatchingEngine& engine) {
    return engine.kv_cache_;
  }
  static std::int64_t raw_clock(const ContinuousBatchingEngine& engine) {
    return engine.clock_.now_us();
  }
  static std::size_t raw_trace_size(const ContinuousBatchingEngine& engine) {
    return engine.plan_trace_.size();
  }
  static std::uint64_t raw_iteration(const ContinuousBatchingEngine& engine) {
    return engine.next_iteration_number_;
  }
  static const std::map<RequestId, Request>& raw_requests(
      const ContinuousBatchingEngine& engine) { return engine.requests_; }
};
}

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class Exception, class Function>
void require_throws(Function function, const char* message) {
  try { function(); } catch (const Exception&) { return; }
  throw std::runtime_error(message);
}
PrefixCacheConfig enabled(std::string salt = "salt", std::string ns = "model/rev/tokenizer") {
  return PrefixCacheConfig(true, std::move(salt), std::move(ns));
}

struct ManagerState {
  std::vector<PhysicalKVBlock> blocks;
  std::set<std::size_t> free;
  std::map<RequestId, std::vector<std::size_t>> tables;
  PrefixCacheMetrics metrics;
  std::uint64_t represented;
  std::uint64_t epoch;
};
ManagerState state(const KVCacheManager& manager) {
  return {manager.physical_blocks(), manager.free_block_ids(), manager.block_tables(),
          manager.prefix_cache_metrics(), manager.represented_token_count(),
          manager.access_epoch()};
}
bool same(const ManagerState& a, const ManagerState& b) {
  return a.blocks == b.blocks && a.free == b.free && a.tables == b.tables &&
      a.metrics.cache_lookup_count == b.metrics.cache_lookup_count &&
      a.metrics.eviction_count == b.metrics.eviction_count &&
      a.metrics.matched_token_count == b.metrics.matched_token_count &&
      a.represented == b.represented && a.epoch == b.epoch;
}

void test_key_hash_and_collision_verification() {
  TokenBlockKey key{kPrefixCacheRootHash, {10, 11, 12, 13}, "model", "salt", 0};
  key.block_hash = stable_token_block_hash(key);
  require(key.block_hash == UINT64_C(0xd5392f858a10326e), "stable known hash changed");
  TokenBlockKey child = key; child.parent_hash = key.block_hash;
  require(stable_token_block_hash(child) != key.block_hash, "parent did not affect child hash");
  child = key; child.model_namespace = "other";
  require(stable_token_block_hash(child) != key.block_hash, "namespace absent from hash");
  child = key; child.cache_salt = "other";
  require(stable_token_block_hash(child) != key.block_hash, "salt absent from hash");

  const auto constant_hash = [](const TokenBlockKey&) { return UINT64_C(7); };
  KVCacheManager manager(KVCacheConfig(4, 4), enabled(), constant_hash);
  manager.allocate_prompt_with_prefix({1}, {1, 2, 3, 4}, manager.find_longest_cached_prefix({1, 2, 3, 4}));
  manager.insert_completed_prompt_blocks({1});
  manager.release_request({1});
  const auto miss = manager.find_longest_cached_prefix({1, 2, 3, 9});
  require(miss.matched_block_count == 0 && miss.collision_verification_count == 1,
          "hash collision was accepted without exact verification");
  const auto hit = manager.find_longest_cached_prefix({1, 2, 3, 4});
  require(hit.matched_block_count == 1 && hit.physical_block_ids == std::vector<std::size_t>{0},
          "exact key did not survive collision bucket");
}

void test_request_prompt_representation() {
  Request count = Request::count_only({1}, 2, 7, 3);
  require(count.prompt_length() == 7 && !count.has_exact_prompt_tokens(),
          "count-only request representation failed");
  require_throws<std::logic_error>([&]{ (void)count.prompt_tokens(); },
                                   "count-only tokens were exposed");
  Request exact = Request::exact_tokens({2}, 3, {4,5,6}, 1);
  require(exact.prompt_length() == 3 && exact.has_exact_prompt_tokens() &&
          exact.prompt_tokens() == std::vector<std::int32_t>({4,5,6}),
          "exact request representation failed");
  Request empty = Request::exact_tokens({3}, 4, {}, 0);
  require(empty.prompt_length() == 0 && empty.has_exact_prompt_tokens() &&
          empty.prompt_tokens().empty(), "empty exact prompt lost its representation");
  Request copied = exact;
  Request moved = std::move(copied);
  require(moved.prompt_length() == moved.prompt_tokens().size(),
          "copied/moved exact request became inconsistent");

  SimulatedBackend backend(SimulatedBackendConfig{});
  ContinuousBatchingEngine engine(backend, ContinuousBatchingConfig(
      1, 8, SchedulingPolicy::DecodeFirst, KVCacheConfig(4,4), enabled()));
  engine.submit_request(exact);
  const auto& stored = llm_lab::serving::test::ContinuousBatchingEngineTestAccess::raw_requests(engine).at({2});
  require(stored.has_exact_prompt_tokens() &&
          stored.prompt_length() == stored.prompt_tokens().size(),
          "stored engine request lost the prompt representation invariant");
}

void test_longest_prefix_and_isolation() {
  KVCacheManager manager(KVCacheConfig(8, 4), enabled());
  const std::vector<std::int32_t> tokens{10,11,12,13,20,21,22,23,30};
  auto miss = manager.find_longest_cached_prefix(tokens);
  manager.allocate_prompt_with_prefix({1}, tokens, miss);
  manager.insert_completed_prompt_blocks({1});
  manager.release_request({1});
  auto full = manager.find_longest_cached_prefix(tokens);
  require(full.matched_block_count == 2 && full.matched_token_count == 8 &&
          full.physical_block_ids == std::vector<std::size_t>({0,1}) &&
          full.kind == PrefixLookupKind::AllEligibleBlocksHit, "multi-block lookup failed");
  require(tokens.size() % 4 != 0,
          "AllEligibleBlocksHit coverage unexpectedly has no prompt tail");
  auto partial = manager.find_longest_cached_prefix({10,11,12,13,99,98,97,96});
  require(partial.matched_block_count == 1 && partial.kind == PrefixLookupKind::PartialHit,
          "lookup did not stop at first divergence");
  require(manager.find_longest_cached_prefix({30,31,32}).matched_block_count == 0,
          "partial block was looked up");
  require(manager.find_longest_cached_prefix(tokens, "other", "salt").matched_block_count == 0,
          "namespace isolation failed");
  require(manager.find_longest_cached_prefix(tokens, "model/rev/tokenizer", "other").matched_block_count == 0,
          "salt isolation failed");
}

void test_publication_boundaries_and_idempotence() {
  KVCacheManager count_only(KVCacheConfig(4, 4), enabled());
  count_only.allocate_prompt({1}, 8);
  require(count_only.physical_blocks()[0].provenance ==
              KVBlockProvenance::PrivatePromptCountOnly &&
          count_only.physical_blocks()[1].provenance ==
              KVBlockProvenance::PrivatePromptCountOnly,
          "count-only full blocks were marked cacheable");
  const KVCacheManager count_snapshot = count_only;
  require_throws<std::logic_error>([&]{ count_only.insert_completed_prompt_blocks({1}); },
                                   "count-only request published a prefix");
  require(llm_lab::serving::test::KVCacheManagerTestAccess::same_full_state(
              count_only, count_snapshot),
          "rejected count-only publication changed manager state");

  KVCacheManager tail_only(KVCacheConfig(2, 4), enabled());
  const std::vector<std::int32_t> short_prompt{8,9,10};
  tail_only.allocate_prompt_with_prefix(
      {9}, short_prompt, tail_only.find_longest_cached_prefix(short_prompt));
  const KVCacheManager tail_snapshot = tail_only;
  require_throws<std::logic_error>([&]{ tail_only.insert_completed_prompt_blocks({9}); },
                                   "tail-only exact request published a prefix");
  require(llm_lab::serving::test::KVCacheManagerTestAccess::same_full_state(
              tail_only, tail_snapshot),
          "rejected tail-only publication changed manager state");

  KVCacheManager exact(KVCacheConfig(6, 4), enabled());
  const std::vector<std::int32_t> tokens{1,2,3,4,5};
  exact.allocate_prompt_with_prefix({2}, tokens,
                                    exact.find_longest_cached_prefix(tokens));
  exact.insert_completed_prompt_blocks({2});
  const auto first = exact.physical_blocks();
  require(first[0].prefix_key.has_value() && !first[1].prefix_key.has_value() &&
          first[1].provenance == KVBlockProvenance::PrivatePromptTail,
          "exact publication crossed the private prompt tail");
  const KVCacheManager published = exact;
  exact.insert_completed_prompt_blocks({2});
  require(llm_lab::serving::test::KVCacheManagerTestAccess::same_full_state(
              exact, published), "repeated publication was not idempotent");

  for (int i = 0; i < 7; ++i) exact.append_decode_token({2});
  const auto table = exact.block_table({2});
  require(table.size() == 3 &&
          exact.physical_blocks()[table[2]].provenance ==
              KVBlockProvenance::PrivateDecode &&
          exact.physical_blocks()[table[2]].valid_token_count == 4 &&
          !exact.physical_blocks()[table[2]].prefix_key.has_value(),
          "decode-created block became cache-publishable");
  const KVCacheManager decoded = exact;
  exact.insert_completed_prompt_blocks({2});
  require(llm_lab::serving::test::KVCacheManagerTestAccess::same_full_state(
              exact, decoded),
          "repeated publication changed the original prompt boundary after decode");
}

KVCacheManager corruption_fixture() {
  KVCacheManager manager(KVCacheConfig(8, 4), enabled());
  const std::vector<std::int32_t> tokens{1,2,3,4,5};
  manager.allocate_prompt_with_prefix({77}, tokens,
                                      manager.find_longest_cached_prefix(tokens));
  manager.insert_completed_prompt_blocks({77});
  for (int i = 0; i < 4; ++i) manager.append_decode_token({77});
  require(manager.block_table({77}).size() == 3 && manager.validate_invariants(),
          "corruption fixture is invalid");
  return manager;
}

template <class Corrupt>
void require_corruption_rejected(Corrupt corrupt, const char* message) {
  KVCacheManager manager = corruption_fixture();
  corrupt(manager);
  require(!manager.validate_invariants(), message);
  const KVCacheManager snapshot = manager;
  const auto unchanged = [&] {
    require(llm_lab::serving::test::KVCacheManagerTestAccess::same_full_state(
                manager, snapshot),
            "corrupt-state rejection was not transactional");
  };
  require_throws<std::logic_error>(
      [&]{ (void)manager.find_longest_cached_prefix({1,2,3,4}); }, message);
  unchanged();
  require_throws<std::logic_error>(
      [&]{ manager.insert_completed_prompt_blocks({77}); }, message);
  unchanged();
  require_throws<std::logic_error>([&]{ manager.release_request({77}); }, message);
  unchanged();
  require_throws<std::logic_error>([&]{ manager.evict_unused_blocks(1); }, message);
  unchanged();
}

void test_corruption_rejection_is_transactional() {
  using Access = llm_lab::serving::test::KVCacheManagerTestAccess;
  require_corruption_rejected(Access::corrupt_hash, "wrong hash was accepted");
  require_corruption_rejected(Access::corrupt_namespace, "wrong namespace was accepted");
  require_corruption_rejected(Access::corrupt_salt, "wrong salt was accepted");
  require_corruption_rejected(Access::corrupt_token_length, "wrong token length was accepted");
  require_corruption_rejected(Access::corrupt_block_provenance,
                              "invalid block provenance was accepted");
  require_corruption_rejected(Access::corrupt_physical_state,
                              "invalid physical state was accepted");
  require_corruption_rejected(Access::add_empty_bucket, "empty index bucket was accepted");
  require_corruption_rejected(Access::add_conflicting_duplicate_key,
                              "conflicting duplicate key was accepted");
  require_corruption_rejected(Access::index_decode_block, "indexed decode block was accepted");
  require_corruption_rejected(Access::index_prompt_tail, "indexed prompt tail was accepted");
  require_corruption_rejected(Access::remove_key_metadata,
                              "missing indexed key metadata was accepted");
  require_corruption_rejected(Access::corrupt_request_boundary,
                              "invalid request boundary was accepted");
  require_corruption_rejected(Access::mismatch_table_provenance,
                              "request/table provenance mismatch was accepted");
}

void test_platform_block_size_validation() {
  if constexpr (std::numeric_limits<std::size_t>::max() <
                std::numeric_limits<std::uint64_t>::max()) {
    require_throws<std::overflow_error>(
        [] { KVCacheConfig config(1,
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) + 1); },
        "block size wider than size_t was accepted");
  } else {
    KVCacheConfig config(1, std::numeric_limits<std::uint64_t>::max());
    require(config.block_size_tokens() == std::numeric_limits<std::uint64_t>::max(),
            "native-width block size was rejected");
  }
}

void test_sharing_release_and_canonical_duplicate() {
  KVCacheManager manager(KVCacheConfig(6, 4), enabled());
  const std::vector<std::int32_t> tokens{1,2,3,4,8,9};
  auto miss = manager.find_longest_cached_prefix(tokens);
  manager.allocate_prompt_with_prefix({1}, tokens, miss);
  manager.insert_completed_prompt_blocks({1});
  manager.release_request({1});
  auto hit = manager.find_longest_cached_prefix(tokens);
  manager.allocate_prompt_with_prefix({2}, tokens, hit);
  manager.allocate_prompt_with_prefix({3}, tokens, hit);
  require(manager.physical_blocks()[0].reference_count == 2 &&
          manager.represented_token_count() == 8, "shared physical accounting failed");
  require(manager.block_table({2}) == std::vector<std::size_t>({0,1}) &&
          manager.block_table({3}) == std::vector<std::size_t>({0,2}),
          "shared prefix/private tails are wrong");
  manager.release_request({2});
  require(manager.physical_blocks()[0].reference_count == 1, "first shared release failed");
  manager.release_request({3});
  require(manager.physical_blocks()[0].state == PhysicalKVBlockState::Cached &&
          manager.physical_blocks()[0].reference_count == 0, "last shared release failed");
  require_throws<std::out_of_range>([&]{ manager.release_request({3}); },
                                    "repeated release succeeded");

  KVCacheManager same_plan(KVCacheConfig(4, 4), enabled());
  const std::vector<std::int32_t> exact{7,7,7,7};
  auto first_miss = same_plan.find_longest_cached_prefix(exact);
  same_plan.allocate_prompt_with_prefix({10}, exact, first_miss);
  same_plan.allocate_prompt_with_prefix({11}, exact, first_miss);
  same_plan.insert_completed_prompt_blocks({10});
  same_plan.insert_completed_prompt_blocks({11});
  require(same_plan.block_table({10})[0] != same_plan.block_table({11})[0] &&
          same_plan.referenced_shared_block_count() == 1,
          "canonical duplicate policy published two keys");
}

void test_deterministic_lru_eviction() {
  KVCacheManager manager(KVCacheConfig(2, 4), enabled());
  for (std::uint64_t id = 1; id <= 2; ++id) {
    std::vector<std::int32_t> tokens(4, static_cast<std::int32_t>(id));
    auto lookup = manager.find_longest_cached_prefix(tokens);
    manager.allocate_prompt_with_prefix({id}, tokens, lookup);
    manager.insert_completed_prompt_blocks({id});
    manager.release_request({id});
  }
  require(manager.eligible_eviction_order() == std::vector<std::size_t>({0,1}),
          "LRU order is not epoch then block ID");
  llm_lab::serving::test::KVCacheManagerTestAccess::tie_cached_epochs(
      manager, manager.access_epoch());
  require(manager.validate_invariants() &&
              manager.eligible_eviction_order() ==
                  std::vector<std::size_t>({0, 1}),
          "equal-epoch LRU did not use block ID as its tie-break");
  manager.allocate_prompt({9}, 1);
  require(manager.block_table({9}) == std::vector<std::size_t>{0} &&
          !manager.physical_blocks()[0].prefix_key.has_value() &&
          manager.prefix_cache_metrics().eviction_count == 1,
          "minimum deterministic eviction/lowest-ID reuse failed");
  require(manager.physical_blocks()[1].state == PhysicalKVBlockState::Cached,
          "too many cached blocks were evicted");

  KVCacheManager protected_hit(KVCacheConfig(2, 4), enabled());
  for (std::uint64_t id = 1; id <= 2; ++id) {
    std::vector<std::int32_t> one_block(4, static_cast<std::int32_t>(id));
    auto lookup = protected_hit.find_longest_cached_prefix(one_block);
    protected_hit.allocate_prompt_with_prefix({id}, one_block, lookup);
    protected_hit.insert_completed_prompt_blocks({id});
    protected_hit.release_request({id});
  }
  const std::vector<std::int32_t> hit_with_tail{2,2,2,2,9};
  auto protected_lookup = protected_hit.find_longest_cached_prefix(hit_with_tail);
  protected_hit.allocate_prompt_with_prefix({8}, hit_with_tail, protected_lookup,
                                             {protected_lookup.physical_block_ids[0]});
  require(protected_hit.block_table({8}) == std::vector<std::size_t>({1,0}) &&
          protected_hit.prefix_cache_metrics().eviction_count == 1,
          "suffix allocation evicted its matched shared prefix");
}

void test_continuous_batching_visibility_and_work() {
  SimulatedBackend backend(SimulatedBackendConfig{});
  ContinuousBatchingEngine engine(backend, ContinuousBatchingConfig(
      2, 16, SchedulingPolicy::DecodeFirst, KVCacheConfig(5,4), enabled()));
  const std::vector<std::int32_t> tokens{1,2,3,4};
  engine.submit_request(Request({1}, 0, tokens, 0));
  engine.submit_request(Request({2}, 0, tokens, 0));
  require(engine.run_next_iteration().made_progress(), "same-plan prefill did not progress");
  const auto& first = engine.plan_trace().back().plan.prefill_work();
  require(first.size() == 2 && first[0].matched_prefix_token_count == 0 &&
          first[1].matched_prefix_token_count == 0 &&
          first[0].newly_allocated_block_ids != first[1].newly_allocated_block_ids,
          "new same-plan blocks became visible");
  engine.submit_request(Request({3}, engine.clock().now_us(), tokens, 1));
  require(engine.run_next_iteration().made_progress(), "later cached request did not progress");
  const auto& work = engine.plan_trace().back().plan.prefill_work().front();
  require(work.original_prompt_token_count == 4 && work.matched_prefix_token_count == 4 &&
          work.prompt_token_count == 0 && work.matched_physical_block_ids == std::vector<std::size_t>{0},
          "cache hit did not reduce scheduled prefill work to zero");
  require(engine.statistics().total_prefill_tokens_scheduled == 8,
          "logical scheduled prefill statistics are wrong");
  require(engine.kv_cache().prefix_cache_metrics().saved_prefill_token_count == 4,
          "saved simulated prefill work is wrong");
}

void test_overflow_transactions_and_terminal_engine_failure() {
  KVCacheManager manager(KVCacheConfig(2, 4), enabled());
  const std::vector<std::int32_t> tokens{1,2,3,4};
  manager.allocate_prompt_with_prefix({1}, tokens, manager.find_longest_cached_prefix(tokens));
  manager.insert_completed_prompt_blocks({1});
  manager.release_request({1});
  auto hit = manager.find_longest_cached_prefix(tokens);

  PrefixCacheMetrics metrics = manager.prefix_cache_metrics();
  metrics.cache_lookup_count = std::numeric_limits<std::uint64_t>::max();
  llm_lab::serving::test::KVCacheManagerTestAccess::set_prefix_metrics(manager, metrics);
  ManagerState before = state(manager);
  require_throws<std::overflow_error>(
      [&]{ manager.allocate_prompt_with_prefix({2}, tokens, hit); },
      "cache lookup metric overflow was missed");
  require(same(before, state(manager)), "lookup overflow changed manager state");

  metrics.cache_lookup_count = 0;
  metrics.eviction_count = std::numeric_limits<std::uint64_t>::max();
  llm_lab::serving::test::KVCacheManagerTestAccess::set_prefix_metrics(manager, metrics);
  before = state(manager);
  require_throws<std::overflow_error>([&]{ manager.allocate_prompt({3}, 8); },
                                      "eviction metric overflow was missed");
  require(same(before, state(manager)), "eviction overflow changed manager state");

  metrics.eviction_count = 0;
  llm_lab::serving::test::KVCacheManagerTestAccess::set_prefix_metrics(manager, metrics);
  llm_lab::serving::test::KVCacheManagerTestAccess::set_access_epoch(
      manager, std::numeric_limits<std::uint64_t>::max());
  before = state(manager);
  require_throws<std::overflow_error>(
      [&]{ manager.allocate_prompt_with_prefix({4}, tokens, hit); },
      "hit access-epoch overflow was missed");
  require(same(before, state(manager)), "access overflow changed manager state");

  SimulatedBackend backend(SimulatedBackendConfig{});
  ContinuousBatchingEngine engine(backend, ContinuousBatchingConfig(
      1, 8, SchedulingPolicy::DecodeFirst, KVCacheConfig(2,4), enabled()));
  engine.submit_request(Request({10}, 0, tokens, 0));
  engine.run_next_iteration();
  PrefixCacheMetrics engine_metrics = engine.kv_cache().prefix_cache_metrics();
  engine_metrics.cache_lookup_count = std::numeric_limits<std::uint64_t>::max();
  llm_lab::serving::test::KVCacheManagerTestAccess::set_prefix_metrics(
      llm_lab::serving::test::ContinuousBatchingEngineTestAccess::raw_kv_cache(engine),
      engine_metrics);
  const auto clock_before = engine.clock().now_us();
  engine.submit_request(Request({11}, clock_before, tokens, 1));
  const auto blocks_before = engine.kv_cache().physical_blocks();
  const auto requests_before =
      llm_lab::serving::test::ContinuousBatchingEngineTestAccess::raw_requests(engine);
  const auto trace_before = engine.plan_trace().size();
  const auto iteration_before =
      llm_lab::serving::test::ContinuousBatchingEngineTestAccess::raw_iteration(engine);
  require_throws<std::overflow_error>([&]{ engine.run_next_iteration(); },
                                      "engine cache metric overflow was missed");
  require(engine.failed(), "engine did not enter terminal Failed state");
  require_throws<std::logic_error>([&]{ (void)engine.request({11}); },
                                   "post-failure result access succeeded");
  // Test-only raw access verifies no prepared cache/clock/trace publication.
  auto& raw = llm_lab::serving::test::ContinuousBatchingEngineTestAccess::raw_kv_cache(engine);
  require(raw.physical_blocks() == blocks_before &&
          llm_lab::serving::test::ContinuousBatchingEngineTestAccess::raw_clock(engine) == clock_before &&
          llm_lab::serving::test::ContinuousBatchingEngineTestAccess::raw_trace_size(engine) == trace_before &&
          llm_lab::serving::test::ContinuousBatchingEngineTestAccess::raw_iteration(engine) == iteration_before &&
          llm_lab::serving::test::ContinuousBatchingEngineTestAccess::raw_requests(engine).size() == requests_before.size(),
          "failed engine published prepared cache mutation");
}

void test_ancestor_eviction_and_stable_reachability() {
  const auto constant_hash = [](const TokenBlockKey&) { return UINT64_C(7); };
  KVCacheManager manager(KVCacheConfig(3, 2), enabled(), constant_hash);
  const std::vector<std::int32_t> chain{1, 2, 3, 4};
  manager.allocate_prompt_with_prefix(
      {1}, chain, manager.find_longest_cached_prefix(chain));
  manager.insert_completed_prompt_blocks({1});
  manager.release_request({1});
  const std::vector<std::int32_t> unrelated{8, 9};
  manager.allocate_prompt_with_prefix(
      {2}, unrelated, manager.find_longest_cached_prefix(unrelated));
  manager.insert_completed_prompt_blocks({2});
  manager.release_request({2});

  require(manager.evict_unused_blocks(1) == std::vector<std::size_t>{0},
          "ancestor eviction did not choose the oldest parent");
  require(manager.physical_blocks()[1].state == PhysicalKVBlockState::Cached,
          "ancestor eviction incorrectly removed the child");
  require(manager.find_longest_cached_prefix(chain).matched_block_count == 0,
          "lookup crossed a missing parent");
  require(manager.find_longest_cached_prefix(unrelated).physical_block_ids ==
              std::vector<std::size_t>{2},
          "unrelated collision entry was damaged by ancestor eviction");

  const std::vector<std::int32_t> parent{1, 2};
  manager.allocate_prompt_with_prefix(
      {3}, parent, manager.find_longest_cached_prefix(parent));
  manager.insert_completed_prompt_blocks({3});
  manager.release_request({3});
  require(manager.find_longest_cached_prefix(chain).physical_block_ids ==
              std::vector<std::size_t>({0, 1}),
          "stable parent recomputation did not make the child reachable again");
  require(manager.find_longest_cached_prefix(unrelated).physical_block_ids ==
              std::vector<std::size_t>{2},
          "parent recomputation damaged an unrelated collision entry");
}

void test_sequential_selected_candidate_planning() {
  SimulatedBackend backend(SimulatedBackendConfig{});
  for (SchedulingPolicy policy :
       {SchedulingPolicy::DecodeFirst, SchedulingPolicy::FcfsMixed}) {
    ContinuousBatchingEngine engine(backend, ContinuousBatchingConfig(
        3, 2, policy, KVCacheConfig(2, 2), enabled()));
    engine.submit_request(Request::exact_tokens({1}, 0, {1, 2}, 0));
    engine.submit_request(Request::exact_tokens({2}, 0, {}, 3));
    require(engine.run_next_iteration().made_progress(),
            "sequential-planning fixture did not initialize");
    const auto before = engine.kv_cache().prefix_cache_metrics();
    const auto now = engine.clock().now_us();
    engine.submit_request(Request::exact_tokens({3}, now, {1, 2, 3, 4}, 1));
    engine.submit_request(Request::exact_tokens({4}, now, {9}, 1));
    require(engine.run_next_iteration().made_progress(),
            "later fitting candidate did not run");
    const BatchPlan& plan = engine.plan_trace().back().plan;
    require(plan.deferred_requests().size() == 1 &&
                plan.deferred_requests()[0].request_id == RequestId{3} &&
                plan.deferred_requests()[0].reason == DeferralReason::TokenBudget,
            "older hit candidate was not token-deferred");
    require(plan.prefill_work().size() == 1 &&
                plan.prefill_work()[0].request_id == RequestId{4} &&
                plan.prefill_work()[0].matched_physical_block_ids.empty() &&
                plan.prefill_work()[0].evicted_block_ids ==
                    std::vector<std::size_t>{0} &&
                plan.prefill_work()[0].newly_allocated_block_ids ==
                    std::vector<std::size_t>{0},
            "deferred hit protected its block from the later candidate");
    const auto after = engine.kv_cache().prefix_cache_metrics();
    require(after.cache_lookup_count == before.cache_lookup_count + 1 &&
                after.cache_hit_lookup_count == before.cache_hit_lookup_count &&
                after.cache_miss_lookup_count == before.cache_miss_lookup_count + 1 &&
                after.matched_token_count == before.matched_token_count &&
                after.eviction_count == before.eviction_count + 1,
            "unselected candidate committed lookup/reuse metrics");
  }
}

void test_selected_eviction_recomputes_later_match() {
  SimulatedBackend backend(SimulatedBackendConfig{});
  for (SchedulingPolicy policy :
       {SchedulingPolicy::DecodeFirst, SchedulingPolicy::FcfsMixed}) {
    ContinuousBatchingEngine engine(backend, ContinuousBatchingConfig(
        2, 5, policy, KVCacheConfig(3, 2), enabled()));
    const std::vector<std::int32_t> chain{1, 2, 3, 4};
    engine.submit_request(Request::exact_tokens({1}, 0, chain, 0));
    require(engine.run() == RunResult::Completed, "chain seed failed");
    engine.submit_request(Request::exact_tokens(
        {2}, engine.clock().now_us(), {8, 9}, 0));
    require(engine.run() == RunResult::Completed, "unrelated seed failed");
    const auto before = engine.kv_cache().prefix_cache_metrics();
    const auto now = engine.clock().now_us();
    engine.submit_request(Request::count_only({10}, now, 1, 1));
    engine.submit_request(Request::exact_tokens({11}, now, chain, 1));
    require(engine.run_next_iteration().made_progress(),
            "recomputed later candidate did not run");
    const BatchPlan& plan = engine.plan_trace().back().plan;
    require(plan.prefill_work().size() == 2 &&
                plan.prefill_work()[0].evicted_block_ids ==
                    std::vector<std::size_t>{0} &&
                plan.prefill_work()[1].matched_prefix_token_count == 0 &&
                plan.prefill_work()[1].prompt_token_count == 4 &&
                plan.prefill_work()[1].evicted_block_ids ==
                    std::vector<std::size_t>({1, 2}) &&
                plan.prefill_work()[1].newly_allocated_block_ids ==
                    std::vector<std::size_t>({1, 2}),
            "later prospective hit was not recomputed after parent eviction");
    require(plan.planned_eviction_ids() ==
                std::vector<std::size_t>({0, 1, 2}) &&
                plan.planned_allocation_ids() ==
                    std::vector<std::size_t>({0, 1, 2}) &&
                engine.kv_cache().block_table({10}) ==
                    std::vector<std::size_t>{0} &&
                engine.kv_cache().block_table({11}) ==
                    std::vector<std::size_t>({1, 2}),
            "planned, committed, and traced physical IDs differ");
    const auto after = engine.kv_cache().prefix_cache_metrics();
    require(after.cache_lookup_count == before.cache_lookup_count + 1 &&
                after.cache_hit_lookup_count == before.cache_hit_lookup_count &&
                after.cache_miss_lookup_count == before.cache_miss_lookup_count + 1,
            "recomputed later miss metrics are wrong");
  }
}

void test_decode_exact_eviction_and_cancellation() {
  SimulatedBackend backend(SimulatedBackendConfig{});
  for (SchedulingPolicy policy :
       {SchedulingPolicy::DecodeFirst, SchedulingPolicy::FcfsMixed}) {
    ContinuousBatchingEngine engine(backend, ContinuousBatchingConfig(
        1, 2, policy, KVCacheConfig(2, 2), enabled()));
    engine.submit_request(Request::exact_tokens({1}, 0, {1, 2}, 0));
    require(engine.run() == RunResult::Completed, "first decode fixture seed failed");
    engine.submit_request(Request::exact_tokens(
        {2}, engine.clock().now_us(), {8, 9}, 0));
    require(engine.run() == RunResult::Completed, "second decode fixture seed failed");
    engine.submit_request(Request::exact_tokens(
        {3}, engine.clock().now_us(), {1, 2}, 3));
    require(engine.run_next_iteration().made_progress(), "full-hit prefill failed");
    require(engine.run_next_iteration().made_progress(), "decode eviction failed");
    const BatchPlan& decode = engine.plan_trace().back().plan;
    require(decode.decode_work().size() == 1 &&
                decode.decode_work()[0].request_id == RequestId{3} &&
                decode.decode_work()[0].evicted_block_ids ==
                    std::vector<std::size_t>{1} &&
                decode.decode_work()[0].newly_allocated_block_id ==
                    std::optional<std::size_t>{1} &&
                decode.planned_eviction_ids() ==
                    std::vector<std::size_t>{1} &&
                decode.planned_allocation_ids() ==
                    std::vector<std::size_t>{1} &&
                engine.kv_cache().block_table({3}) ==
                    std::vector<std::size_t>({0, 1}) &&
                engine.kv_cache().physical_blocks()[1].provenance ==
                    KVBlockProvenance::PrivateDecode &&
                !engine.kv_cache().physical_blocks()[1].prefix_key.has_value(),
            "decode planned/actual/trace identity or privacy failed");
    engine.cancel_request({3});
    require(engine.kv_cache().physical_blocks()[0].state ==
                PhysicalKVBlockState::Cached &&
                engine.kv_cache().physical_blocks()[1].state ==
                    PhysicalKVBlockState::Free &&
                engine.kv_cache().referenced_shared_block_count() == 0,
            "active cancellation did not release shared prefix and private decode");
  }
}

void test_cache_aware_engine_integration_matrix() {
  SimulatedBackend backend(SimulatedBackendConfig{});
  for (SchedulingPolicy policy :
       {SchedulingPolicy::DecodeFirst, SchedulingPolicy::FcfsMixed}) {
    ContinuousBatchingEngine full(backend, ContinuousBatchingConfig(
        1, 2, policy, KVCacheConfig(1, 2), enabled()));
    full.submit_request(Request::exact_tokens({1}, 0, {4, 5}, 0));
    require(full.run() == RunResult::Completed, "full-hit seed failed");
    const auto before = full.kv_cache().prefix_cache_metrics();
    full.submit_request(Request::exact_tokens(
        {2}, full.clock().now_us(), {4, 5}, 0));
    require(full.run() == RunResult::Completed, "zero-output full hit failed");
    const auto& hit = full.plan_trace().back().plan.prefill_work().front();
    require(hit.matched_physical_block_ids == std::vector<std::size_t>{0} &&
                hit.newly_allocated_block_ids.empty() &&
                hit.prompt_token_count == 0 &&
                full.kv_cache().cached_block_count() == 1 &&
                full.kv_cache().referenced_shared_block_count() == 0 &&
                full.kv_cache().prefix_cache_metrics().cache_hit_lookup_count ==
                    before.cache_hit_lookup_count + 1,
            "full hit did not acquire/release its reference or avoid capacity deferral");

    ContinuousBatchingEngine partial(backend, ContinuousBatchingConfig(
        1, 3, policy, KVCacheConfig(2, 2), enabled()));
    partial.submit_request(Request::exact_tokens({1}, 0, {4, 5}, 0));
    require(partial.run() == RunResult::Completed, "partial-hit seed failed");
    partial.submit_request(Request::exact_tokens(
        {2}, partial.clock().now_us(), {4, 5, 6}, 2));
    require(partial.run_next_iteration().made_progress(), "partial hit failed");
    const auto& partial_work =
        partial.plan_trace().back().plan.prefill_work().front();
    require(partial_work.matched_physical_block_ids ==
                std::vector<std::size_t>{0} &&
                partial_work.newly_allocated_block_ids ==
                    std::vector<std::size_t>{1} &&
                partial_work.prompt_token_count == 1 &&
                partial.kv_cache().block_table({2}) ==
                    std::vector<std::size_t>({0, 1}),
            "partial hit did not reserve its exact private suffix");

    ContinuousBatchingEngine cancel(backend, ContinuousBatchingConfig(
        1, 3, policy, KVCacheConfig(3, 2), enabled()));
    cancel.submit_request(Request::exact_tokens({1}, 0, {1, 2}, 0));
    require(cancel.run() == RunResult::Completed, "cancellation seed failed");
    cancel.submit_request(Request::exact_tokens(
        {2}, cancel.clock().now_us(), {1, 2, 3}, 5));
    require(cancel.run_next_iteration().made_progress() &&
                cancel.run_next_iteration().made_progress() &&
                cancel.run_next_iteration().made_progress(),
            "shared/tail/decode cancellation fixture failed");
    const auto table = cancel.kv_cache().block_table({2});
    require(table.size() == 3 &&
                cancel.kv_cache().physical_blocks()[table[0]].prefix_key.has_value() &&
                cancel.kv_cache().physical_blocks()[table[1]].provenance ==
                    KVBlockProvenance::PrivatePromptTail &&
                cancel.kv_cache().physical_blocks()[table[2]].provenance ==
                    KVBlockProvenance::PrivateDecode,
            "cancellation fixture lacks shared prefix, tail, or decode block");
    cancel.cancel_request({2});
    require(cancel.kv_cache().physical_blocks()[table[0]].state ==
                PhysicalKVBlockState::Cached &&
                cancel.kv_cache().physical_blocks()[table[1]].state ==
                    PhysicalKVBlockState::Free &&
                cancel.kv_cache().physical_blocks()[table[2]].state ==
                    PhysicalKVBlockState::Free,
            "cancellation did not release every provenance class correctly");

    ContinuousBatchingEngine recovery(backend, ContinuousBatchingConfig(
        1, 2, policy, KVCacheConfig(1, 2), enabled()));
    recovery.submit_request(Request::count_only({1}, 0, 2, 4));
    require(recovery.run_next_iteration().made_progress(), "stall resident failed");
    recovery.submit_request(Request::exact_tokens(
        {2}, recovery.clock().now_us(), {9}, 1));
    const IterationResult stalled = recovery.run_next_iteration();
    require(stalled.is_stalled() &&
                recovery.plan_trace().back().plan.deferred_only() &&
                recovery.plan_trace().back().plan.deferred_requests()[0].reason ==
                    DeferralReason::KVCapacity,
            "cache-aware no-progress was not a KV-only stall");
    recovery.cancel_request({1});
    require(recovery.run() == RunResult::Completed &&
                recovery.request({2}).state == RequestState::Finished,
            "cache-aware stall did not recover after cancellation");
  }
}
}

int main() {
  try {
    test_request_prompt_representation();
    test_key_hash_and_collision_verification();
    test_longest_prefix_and_isolation();
    test_publication_boundaries_and_idempotence();
    test_corruption_rejection_is_transactional();
    test_platform_block_size_validation();
    test_sharing_release_and_canonical_duplicate();
    test_deterministic_lru_eviction();
    test_continuous_batching_visibility_and_work();
    test_overflow_transactions_and_terminal_engine_failure();
    test_ancestor_eviction_and_stable_reachability();
    test_sequential_selected_candidate_planning();
    test_selected_eviction_recomputes_later_match();
    test_decode_exact_eviction_and_cancellation();
    test_cache_aware_engine_integration_matrix();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return EXIT_FAILURE;
  }
  std::cout << "prefix cache tests passed\n"; return EXIT_SUCCESS;
}
