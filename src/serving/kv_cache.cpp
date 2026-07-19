#include "serving/kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llm_lab::serving {
namespace {
std::uint64_t inc(std::uint64_t value, const char* message) {
  if (value == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error(message);
  return value + 1;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b, const char* message) {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) throw std::overflow_error(message);
  return a + b;
}
void fnv_byte(std::uint64_t& hash, std::uint8_t byte) noexcept {
  hash ^= byte;
  hash *= UINT64_C(1099511628211);
}
void fnv_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
  for (unsigned i = 0; i < 8; ++i) fnv_byte(hash, static_cast<std::uint8_t>(value >> (8U * i)));
}
void fnv_string(std::uint64_t& hash, std::string_view value) noexcept {
  fnv_u64(hash, static_cast<std::uint64_t>(value.size()));
  for (unsigned char byte : value) fnv_byte(hash, byte);
}
bool valid_state(PhysicalKVBlockState state) noexcept {
  switch (state) {
    case PhysicalKVBlockState::Free:
    case PhysicalKVBlockState::InUse:
    case PhysicalKVBlockState::Cached: return true;
  }
  return false;
}
bool valid_provenance(KVBlockProvenance provenance) noexcept {
  switch (provenance) {
    case KVBlockProvenance::SharedPromptFull:
    case KVBlockProvenance::ComputedPromptFull:
    case KVBlockProvenance::PrivatePromptTail:
    case KVBlockProvenance::PrivatePromptCountOnly:
    case KVBlockProvenance::PrivateDecode: return true;
  }
  return false;
}
}

KVCacheConfig::KVCacheConfig(std::size_t blocks, std::uint64_t block_size)
    : total_num_blocks_(blocks), block_size_tokens_(block_size), total_capacity_tokens_(0) {
  if (blocks == 0) throw std::invalid_argument("total_num_blocks must be at least 1");
  if (block_size == 0) throw std::invalid_argument("block_size_tokens must be at least 1");
  if (block_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("block_size_tokens is not representable as size_t");
  if (blocks > std::numeric_limits<std::uint64_t>::max() / block_size)
    throw std::overflow_error("KV token capacity is not representable");
  total_capacity_tokens_ = static_cast<std::uint64_t>(blocks) * block_size;
}

PrefixCacheConfig::PrefixCacheConfig(bool enabled, std::string salt, std::string ns,
                                     PrefixCacheEvictionPolicy policy)
    : enabled_(enabled), cache_salt_(std::move(salt)), model_namespace_(std::move(ns)),
      eviction_policy_(policy) {
  if (policy != PrefixCacheEvictionPolicy::DeterministicLRU)
    throw std::invalid_argument("unsupported prefix-cache eviction policy");
}

bool exact_key_material_equal(const TokenBlockKey& a, const TokenBlockKey& b) noexcept {
  return a.parent_hash == b.parent_hash && a.token_ids == b.token_ids &&
         a.model_namespace == b.model_namespace && a.cache_salt == b.cache_salt;
}
bool operator==(const TokenBlockKey& a, const TokenBlockKey& b) noexcept {
  return a.block_hash == b.block_hash && exact_key_material_equal(a, b);
}

std::uint64_t stable_token_block_hash(const TokenBlockKey& key) noexcept {
  std::uint64_t hash = kPrefixCacheRootHash;
  fnv_byte(hash, 1); fnv_u64(hash, key.parent_hash);
  fnv_byte(hash, 2); fnv_string(hash, key.model_namespace);
  fnv_byte(hash, 3); fnv_string(hash, key.cache_salt);
  fnv_byte(hash, 4); fnv_u64(hash, static_cast<std::uint64_t>(key.token_ids.size()));
  for (std::int32_t token : key.token_ids) {
    const std::uint32_t bits = static_cast<std::uint32_t>(token);
    for (unsigned i = 0; i < 4; ++i) fnv_byte(hash, static_cast<std::uint8_t>(bits >> (8U * i)));
  }
  return hash;
}

std::optional<double> PrefixCacheMetrics::prefix_token_hit_rate() const noexcept {
  if (total_cache_eligible_prompt_tokens_looked_up == 0) return std::nullopt;
  return static_cast<double>(matched_token_count) /
         static_cast<double>(total_cache_eligible_prompt_tokens_looked_up);
}

KVCacheManager::KVCacheManager(KVCacheConfig config, PrefixCacheConfig prefix_config,
                               PrefixHashFunction hash_function)
    : config_(config), prefix_config_(std::move(prefix_config)),
      hash_function_(std::move(hash_function)) {
  if (!hash_function_) throw std::invalid_argument("prefix hash function must be callable");
  if (config_.total_num_blocks() > blocks_.max_size())
    throw std::length_error("KV block metadata count is not representable");
  blocks_.reserve(config_.total_num_blocks());
  for (std::size_t id = 0; id < config_.total_num_blocks(); ++id) {
    blocks_.push_back(PhysicalKVBlock{id, PhysicalKVBlockState::Free,
                                      std::nullopt, 0, 0, 0, std::nullopt,
                                      std::nullopt});
    free_blocks_.insert(id);
  }
}

std::size_t KVCacheManager::blocks_required(std::uint64_t tokens) const {
  if (tokens == 0) return 0;
  const auto required = tokens / config_.block_size_tokens() +
                        (tokens % config_.block_size_tokens() == 0 ? 0 : 1);
  if (required > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("required KV block count is not representable");
  return static_cast<std::size_t>(required);
}
bool KVCacheManager::can_allocate_blocks(std::size_t n) const noexcept { return n <= free_blocks_.size(); }
bool KVCacheManager::can_allocate_tokens(std::uint64_t n) const { return can_allocate_blocks(blocks_required(n)); }
std::uint64_t KVCacheManager::next_access_epoch() const {
  if (fail_for_test(test::KVCacheFailurePoint::AccessEpoch))
    throw std::overflow_error("injected KV access epoch overflow");
  return inc(access_epoch_, "KV access epoch overflow");
}
void KVCacheManager::note_allocation_failure() {
  if (fail_for_test(test::KVCacheFailurePoint::AllocationFailureCounter))
    throw std::overflow_error("injected KV allocation failure counter overflow");
  allocation_failure_count_ = inc(allocation_failure_count_, "KV allocation failure count overflow");
}

TokenBlockKey KVCacheManager::make_key(std::uint64_t parent,
    const std::vector<std::int32_t>& tokens, std::size_t offset,
    std::string_view ns, std::string_view salt) const {
  const auto size = static_cast<std::size_t>(config_.block_size_tokens());
  if (offset > tokens.size() || size > tokens.size() - offset)
    throw std::invalid_argument("prefix key requires one complete token block");
  TokenBlockKey key;
  key.parent_hash = parent;
  key.token_ids.assign(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                       tokens.begin() + static_cast<std::ptrdiff_t>(offset + size));
  key.model_namespace.assign(ns.data(), ns.size());
  key.cache_salt.assign(salt.data(), salt.size());
  key.block_hash = hash_function_(key);
  return key;
}

PrefixLookupResult KVCacheManager::find_longest_cached_prefix(
    const std::vector<std::int32_t>& tokens) const {
  return find_longest_cached_prefix(tokens, prefix_config_.model_namespace(),
                                    prefix_config_.cache_salt());
}
PrefixLookupResult KVCacheManager::find_longest_cached_prefix(
    const std::vector<std::int32_t>& tokens, std::string_view ns,
    std::string_view salt) const {
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state before prefix lookup");
  PrefixLookupResult result;
  if (!prefix_config_.enabled()) { result.kind = PrefixLookupKind::Disabled; return result; }
  const std::size_t bs = static_cast<std::size_t>(config_.block_size_tokens());
  const std::size_t full = tokens.size() / bs;
  result.eligible_token_count = static_cast<std::uint64_t>(full) * config_.block_size_tokens();
  std::uint64_t parent = kPrefixCacheRootHash;
  for (std::size_t i = 0; i < full; ++i) {
    TokenBlockKey key = make_key(parent, tokens, i * bs, ns, salt);
    const auto bucket = prefix_index_.find(key.block_hash);
    std::optional<std::size_t> match;
    if (bucket != prefix_index_.end()) {
      for (std::size_t id : bucket->second) {
        const auto& stored = blocks_.at(id).prefix_key;
        if (stored.has_value() && exact_key_material_equal(*stored, key)) {
          match = id; break;
        }
        result.collision_verification_count = inc(result.collision_verification_count,
                                                   "collision verification count overflow");
      }
    }
    if (!match.has_value()) break;
    result.physical_block_ids.push_back(*match);
    result.matched_block_count++;
    result.matched_token_count += config_.block_size_tokens();
    parent = key.block_hash;
  }
  result.next_parent_hash = parent;
  if (result.matched_block_count == full && full != 0)
    result.kind = PrefixLookupKind::AllEligibleBlocksHit;
  else if (result.matched_block_count != 0) result.kind = PrefixLookupKind::PartialHit;
  else result.kind = PrefixLookupKind::Miss;
  return result;
}

void KVCacheManager::allocate_prompt(RequestId id, std::uint64_t tokens) {
  if (!validate_invariants()) throw std::logic_error("invalid KV manager state before prompt allocation");
  if (block_tables_.count(id)) throw std::invalid_argument("duplicate KV request ID");
  const std::size_t required = blocks_required(tokens);
  const std::size_t evictable = prefix_config_.enabled() ? eligible_eviction_order().size() : 0;
  if (required > free_blocks_.size() + evictable) { note_allocation_failure(); throw KVCapacityError("insufficient KV capacity for prompt"); }
  const std::size_t eviction_count =
      required > free_blocks_.size() ? required - free_blocks_.size() : 0;
  auto evictions = eligible_eviction_order();
  evictions.resize(eviction_count);
  std::set<std::size_t> available = free_blocks_;
  available.insert(evictions.begin(), evictions.end());
  std::vector<std::size_t> allocations;
  auto available_it = available.begin();
  for (std::size_t i = 0; i < required; ++i, ++available_it)
    allocations.push_back(*available_it);
  allocate_prompt_exact(id, tokens, evictions, allocations);
}

void KVCacheManager::allocate_prompt_exact(
    RequestId id, std::uint64_t tokens,
    const std::vector<std::size_t>& eviction_ids,
    const std::vector<std::size_t>& allocation_ids) {
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state before exact prompt allocation");
  KVCacheManager copy(*this);
  copy.allocate_prompt_exact_in_place(id, tokens, eviction_ids, allocation_ids);
  swap(copy);
}

void KVCacheManager::allocate_prompt_exact_in_place(
    RequestId id, std::uint64_t tokens,
    const std::vector<std::size_t>& eviction_ids,
    const std::vector<std::size_t>& allocation_ids) {
  if (block_tables_.count(id))
    throw std::invalid_argument("duplicate KV request ID");
  const std::size_t required = blocks_required(tokens);
  auto order = eligible_eviction_order();
  const std::size_t needed =
      required > free_blocks_.size() ? required - free_blocks_.size() : 0;
  if (needed > order.size())
    throw KVCapacityError("insufficient KV capacity for prompt");
  order.resize(needed);
  if (eviction_ids != order)
    throw std::logic_error("planned prompt eviction IDs differ from actual IDs");
  std::set<std::size_t> available = free_blocks_;
  available.insert(eviction_ids.begin(), eviction_ids.end());
  std::vector<std::size_t> expected_allocations;
  auto available_it = available.begin();
  for (std::size_t i = 0; i < required; ++i, ++available_it)
    expected_allocations.push_back(*available_it);
  if (allocation_ids != expected_allocations)
    throw std::logic_error("planned prompt allocation IDs differ from actual IDs");
  evict_exact_blocks_in_place(eviction_ids);

  if (fail_for_test(test::KVCacheFailurePoint::RepresentedTokenArithmetic))
    throw std::overflow_error("injected represented-token arithmetic overflow");
  const auto represented = add(represented_token_count_, tokens,
                               "represented KV token count overflow");
  const std::uint64_t epoch = required ? next_access_epoch() : access_epoch_;
  block_tables_.emplace(id, allocation_ids);
  request_provenance_.emplace(id, RequestKVPromptProvenance{
      id, false, std::nullopt, tokens, 0, 0, 0, 0, std::nullopt,
      allocation_ids.size(), false});
  for (std::size_t i = 0; i < allocation_ids.size(); ++i) {
    auto& block = blocks_[allocation_ids[i]];
    block.state = PhysicalKVBlockState::InUse;
    block.owner = id;
    block.reference_count = 1;
    const auto used = static_cast<std::uint64_t>(i) * config_.block_size_tokens();
    block.valid_token_count =
        std::min(config_.block_size_tokens(), tokens - used);
    block.last_access_epoch = epoch;
    block.prefix_key.reset();
    block.provenance = KVBlockProvenance::PrivatePromptCountOnly;
    free_blocks_.erase(block.block_id);
  }
  represented_token_count_ = represented;
  if (required) access_epoch_ = epoch;
  update_peak();
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state after exact prompt allocation");
}
void KVCacheManager::allocate_prompt_with_prefix(RequestId id,
    const std::vector<std::int32_t>& tokens, const PrefixLookupResult& lookup,
    const std::set<std::size_t>& protected_blocks) {
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state before prefix acquisition");
  const PrefixLookupResult actual = find_longest_cached_prefix(tokens);
  if (actual.physical_block_ids != lookup.physical_block_ids ||
      actual.matched_token_count != lookup.matched_token_count)
    throw std::logic_error("stale prefix lookup snapshot");
  const std::uint64_t suffix =
      static_cast<std::uint64_t>(tokens.size()) - actual.matched_token_count;
  const std::size_t required = blocks_required(suffix);
  auto evictable = eligible_eviction_order();
  evictable.erase(std::remove_if(evictable.begin(), evictable.end(),
      [&](std::size_t block_id) {
        return protected_blocks.count(block_id) != 0 ||
               std::find(actual.physical_block_ids.begin(),
                         actual.physical_block_ids.end(), block_id) !=
                   actual.physical_block_ids.end();
      }), evictable.end());
  const std::size_t eviction_count =
      required > free_blocks_.size() ? required - free_blocks_.size() : 0;
  if (eviction_count > evictable.size()) {
    note_allocation_failure();
    throw KVCapacityError("insufficient KV capacity for prefix suffix");
  }
  evictable.resize(eviction_count);
  std::set<std::size_t> available = free_blocks_;
  available.insert(evictable.begin(), evictable.end());
  std::vector<std::size_t> allocations;
  auto available_it = available.begin();
  for (std::size_t i = 0; i < required; ++i, ++available_it)
    allocations.push_back(*available_it);
  allocate_prompt_with_prefix_exact(id, tokens, lookup, evictable, allocations);
}
void KVCacheManager::allocate_prompt_with_prefix_exact(
    RequestId id, const std::vector<std::int32_t>& tokens,
    const PrefixLookupResult& lookup,
    const std::vector<std::size_t>& eviction_ids,
    const std::vector<std::size_t>& allocation_ids) {
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state before exact prefix acquisition");
  KVCacheManager copy(*this);
  copy.allocate_prompt_with_prefix_exact_in_place(
      id, tokens, lookup, eviction_ids, allocation_ids);
  swap(copy);
}

void KVCacheManager::allocate_prompt_with_prefix_exact_in_place(
    RequestId id, const std::vector<std::int32_t>& tokens,
    const PrefixLookupResult& supplied,
    const std::vector<std::size_t>& eviction_ids,
    const std::vector<std::size_t>& allocation_ids) {
  if (block_tables_.count(id))
    throw std::invalid_argument("duplicate KV request ID");
  if (!prefix_config_.enabled())
    throw std::logic_error("exact prefix allocation requires an enabled cache");
  const PrefixLookupResult actual = find_longest_cached_prefix(tokens);
  if (actual.physical_block_ids != supplied.physical_block_ids ||
      actual.matched_token_count != supplied.matched_token_count ||
      actual.kind != supplied.kind)
    throw std::logic_error("stale prefix lookup snapshot");
  const std::uint64_t total = static_cast<std::uint64_t>(tokens.size());
  if (actual.matched_token_count > total)
    throw std::invalid_argument("invalid matched prefix length");
  const std::uint64_t suffix = total - actual.matched_token_count;
  const std::size_t required = blocks_required(suffix);
  auto evictable = eligible_eviction_order();
  evictable.erase(std::remove_if(evictable.begin(), evictable.end(),
      [&](std::size_t block_id) {
        return std::find(actual.physical_block_ids.begin(),
                         actual.physical_block_ids.end(), block_id) !=
               actual.physical_block_ids.end();
      }), evictable.end());
  const std::size_t needed =
      required > free_blocks_.size() ? required - free_blocks_.size() : 0;
  if (needed > evictable.size())
    throw KVCapacityError("insufficient KV capacity for prefix suffix");
  evictable.resize(needed);
  if (eviction_ids != evictable)
    throw std::logic_error("planned prefix eviction IDs differ from actual IDs");
  std::set<std::size_t> available = free_blocks_;
  available.insert(eviction_ids.begin(), eviction_ids.end());
  std::vector<std::size_t> expected_allocations;
  auto available_it = available.begin();
  for (std::size_t i = 0; i < required; ++i, ++available_it)
    expected_allocations.push_back(*available_it);
  if (allocation_ids != expected_allocations)
    throw std::logic_error("planned prefix allocation IDs differ from actual IDs");

  if (fail_for_test(test::KVCacheFailurePoint::RepresentedTokenArithmetic))
    throw std::overflow_error("injected represented-token arithmetic overflow");
  auto represented = add(represented_token_count_, suffix,
                         "represented KV token count overflow");
  PrefixCacheMetrics metrics = prefix_metrics_;
  if (fail_for_test(test::KVCacheFailurePoint::CacheLookupCounter))
    throw std::overflow_error("injected cache lookup counter overflow");
  metrics.cache_lookup_count = inc(metrics.cache_lookup_count, "cache lookup count overflow");
  if (fail_for_test(test::KVCacheFailurePoint::EligibleTokenCounter))
    throw std::overflow_error("injected eligible-token counter overflow");
  metrics.total_cache_eligible_prompt_tokens_looked_up = add(
      metrics.total_cache_eligible_prompt_tokens_looked_up,
      actual.eligible_token_count, "eligible prefix token count overflow");
  if (fail_for_test(test::KVCacheFailurePoint::CollisionCounter))
    throw std::overflow_error("injected collision counter overflow");
  metrics.collision_verification_count = add(
      metrics.collision_verification_count,
      actual.collision_verification_count,
      "collision verification count overflow");
  if (actual.matched_block_count) {
    if (fail_for_test(test::KVCacheFailurePoint::CacheHitCounter))
      throw std::overflow_error("injected cache hit counter overflow");
    metrics.cache_hit_lookup_count = inc(metrics.cache_hit_lookup_count,
                                         "cache hit count overflow");
    if (fail_for_test(test::KVCacheFailurePoint::ReusedRequestCounter))
      throw std::overflow_error("injected reused-request counter overflow");
    metrics.reused_request_count = inc(metrics.reused_request_count,
                                       "reused request count overflow");
    if (fail_for_test(test::KVCacheFailurePoint::MatchedBlockCounter))
      throw std::overflow_error("injected matched-block counter overflow");
    metrics.matched_block_count = add(metrics.matched_block_count,
        actual.matched_block_count, "matched block count overflow");
    if (fail_for_test(test::KVCacheFailurePoint::MatchedTokenCounter))
      throw std::overflow_error("injected matched-token counter overflow");
    metrics.matched_token_count = add(metrics.matched_token_count,
        actual.matched_token_count, "matched token count overflow");
    if (fail_for_test(test::KVCacheFailurePoint::SavedPrefillCounter))
      throw std::overflow_error("injected saved-prefill counter overflow");
    metrics.saved_prefill_token_count = add(metrics.saved_prefill_token_count,
        actual.matched_token_count, "saved prefill token count overflow");
  } else {
    if (fail_for_test(test::KVCacheFailurePoint::CacheMissCounter))
      throw std::overflow_error("injected cache miss counter overflow");
    metrics.cache_miss_lookup_count = inc(metrics.cache_miss_lookup_count,
                                           "cache miss count overflow");
  }
  if (fail_for_test(test::KVCacheFailurePoint::ReferenceCounter) &&
      !actual.physical_block_ids.empty())
    throw std::overflow_error("injected prefix reference counter overflow");
  for (std::size_t block_id : actual.physical_block_ids)
    if (blocks_[block_id].reference_count == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("prefix block reference count overflow");
  std::uint64_t hit_epoch = access_epoch_;
  if (!actual.physical_block_ids.empty()) hit_epoch = next_access_epoch();
  std::vector<std::size_t> table = actual.physical_block_ids;
  for (std::size_t block_id : table) {
    auto& block = blocks_[block_id];
    const bool was_cached = block.state == PhysicalKVBlockState::Cached;
    ++block.reference_count;
    block.state = PhysicalKVBlockState::InUse; block.owner.reset();
    block.last_access_epoch = hit_epoch;
    if (was_cached && block.provenance == KVBlockProvenance::ComputedPromptFull)
      block.provenance = KVBlockProvenance::SharedPromptFull;
  }
  if (!table.empty()) access_epoch_ = hit_epoch;
  evict_exact_blocks_in_place(eviction_ids);
  represented -= static_cast<std::uint64_t>(eviction_ids.size()) *
                 config_.block_size_tokens();
  metrics.eviction_count = prefix_metrics_.eviction_count;
  std::uint64_t allocation_epoch = access_epoch_;
  if (required) allocation_epoch = next_access_epoch();
  for (std::size_t i = 0; i < allocation_ids.size(); ++i) {
    auto& block = blocks_[allocation_ids[i]]; block.state = PhysicalKVBlockState::InUse;
    block.owner = id; block.reference_count = 1;
    const auto used = static_cast<std::uint64_t>(i) * config_.block_size_tokens();
    block.valid_token_count = std::min(config_.block_size_tokens(), suffix - used);
    block.last_access_epoch = allocation_epoch; block.prefix_key.reset();
    const std::size_t table_position = actual.matched_block_count + i;
    block.provenance = table_position < tokens.size() / static_cast<std::size_t>(config_.block_size_tokens())
                           ? KVBlockProvenance::ComputedPromptFull
                           : KVBlockProvenance::PrivatePromptTail;
    free_blocks_.erase(block.block_id); table.push_back(block.block_id);
  }
  if (required) access_epoch_ = allocation_epoch;
  const std::size_t full = tokens.size() / static_cast<std::size_t>(config_.block_size_tokens());
  const bool has_tail = tokens.size() % static_cast<std::size_t>(config_.block_size_tokens()) != 0;
  request_provenance_.emplace(id, RequestKVPromptProvenance{
      id, prefix_config_.enabled(), tokens, total, full,
      actual.matched_block_count, actual.matched_block_count, full,
      has_tail ? std::optional<std::size_t>(full) : std::nullopt,
      full + (has_tail ? 1U : 0U), false});
  block_tables_.emplace(id, std::move(table)); represented_token_count_ = represented;
  prefix_metrics_ = metrics; update_peak();
  if (!validate_invariants()) throw std::logic_error("invalid KV manager state after prefix acquisition");
}

void KVCacheManager::insert_completed_prompt_blocks(RequestId id) {
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state before prefix insertion");
  KVCacheManager copy(*this); copy.insert_completed_prompt_blocks_in_place(id); swap(copy);
}
void KVCacheManager::insert_completed_prompt_blocks_in_place(RequestId id) {
  auto found = block_tables_.find(id);
  if (found == block_tables_.end()) throw std::out_of_range("unknown KV request ID");
  auto provenance_it = request_provenance_.find(id);
  if (provenance_it == request_provenance_.end())
    throw std::logic_error("missing request prompt provenance");
  auto& provenance = provenance_it->second;
  if (!prefix_config_.enabled() || !provenance.publication_eligible ||
      !provenance.exact_prompt_tokens)
    throw std::logic_error("request prompt is ineligible for prefix publication");
  if (provenance.computed_full_begin == provenance.computed_full_end)
    throw std::logic_error("request has no computed full prompt blocks to publish");
  if (provenance.publication_completed) return;
  const auto& tokens = *provenance.exact_prompt_tokens;
  const std::size_t bs = static_cast<std::size_t>(config_.block_size_tokens());
  std::uint64_t parent = kPrefixCacheRootHash;
  for (std::size_t i = 0; i < provenance.cache_eligible_full_prompt_blocks; ++i) {
    TokenBlockKey key = make_key(parent, tokens, i * bs,
                                 prefix_config_.model_namespace(), prefix_config_.cache_salt());
    if (i < provenance.computed_full_begin) {
      const auto& block = blocks_[found->second[i]];
      if (!block.prefix_key || !exact_key_material_equal(*block.prefix_key, key))
        throw std::logic_error("shared prompt block does not match stored provenance");
      parent = key.block_hash;
      continue;
    }
    bool canonical_exists = false;
    auto bucket = prefix_index_.find(key.block_hash);
    if (bucket != prefix_index_.end()) {
      for (std::size_t candidate : bucket->second) {
        if (blocks_[candidate].prefix_key && exact_key_material_equal(*blocks_[candidate].prefix_key, key)) {
          canonical_exists = true; break;
        }
      }
    }
    auto& block = blocks_[found->second[i]];
    if (fail_for_test(test::KVCacheFailurePoint::CanonicalIndexConflict))
      throw std::logic_error("injected canonical index conflict");
    if (block.provenance != KVBlockProvenance::ComputedPromptFull)
      throw std::logic_error("non-computed prompt block cannot be published");
    if (!canonical_exists && !block.prefix_key.has_value()) {
      if (fail_for_test(test::KVCacheFailurePoint::PublicationAccessEpoch))
        throw std::overflow_error("injected publication access epoch overflow");
      const std::uint64_t epoch = next_access_epoch();
      block.prefix_key = key; block.owner.reset(); block.last_access_epoch = epoch;
      auto& ids = prefix_index_[key.block_hash]; ids.push_back(block.block_id);
      std::sort(ids.begin(), ids.end()); access_epoch_ = epoch;
    }
    parent = key.block_hash;
  }
  provenance.publication_completed = true;
  if (!validate_invariants()) throw std::logic_error("invalid KV manager state after prefix insertion");
}

bool KVCacheManager::decode_requires_new_block(RequestId id) const {
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state before decode capacity query");
  const auto found = block_tables_.find(id); if (found == block_tables_.end()) throw std::out_of_range("unknown KV request ID");
  if (found->second.empty()) return true;
  const auto& block = blocks_.at(found->second.back());
  if (block.state != PhysicalKVBlockState::InUse || block.reference_count == 0 ||
      block.valid_token_count == 0 || block.valid_token_count > config_.block_size_tokens())
    throw std::logic_error("invalid final KV block metadata");
  return block.prefix_key.has_value() || block.reference_count > 1 ||
         block.valid_token_count == config_.block_size_tokens();
}
void KVCacheManager::append_decode_token(RequestId id) {
  if (!validate_invariants()) throw std::logic_error("invalid KV manager state before decode append");
  const auto found = block_tables_.find(id);
  if (found == block_tables_.end()) throw std::out_of_range("unknown KV request ID");
  if (represented_token_count_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("represented KV token count overflow");
  if (fail_for_test(test::KVCacheFailurePoint::RepresentedTokenArithmetic))
    throw std::overflow_error("injected represented-token arithmetic overflow");
  (void)next_access_epoch();
  if (decode_requires_new_block(id) && free_blocks_.empty() && eligible_eviction_order().empty()) {
    note_allocation_failure();
    throw KVCapacityError("insufficient KV capacity for decode token");
  }
  const bool needs_block = decode_requires_new_block(id);
  std::vector<std::size_t> evictions;
  std::optional<std::size_t> allocation;
  if (needs_block) {
    if (free_blocks_.empty())
      evictions.push_back(eligible_eviction_order().front());
    std::set<std::size_t> available = free_blocks_;
    available.insert(evictions.begin(), evictions.end());
    allocation = *available.begin();
  }
  append_decode_token_exact(id, evictions, allocation);
}
void KVCacheManager::append_decode_token_exact(
    RequestId id, const std::vector<std::size_t>& eviction_ids,
    const std::optional<std::size_t>& allocation_id) {
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state before exact decode append");
  KVCacheManager copy(*this);
  copy.append_decode_token_exact_in_place(id, eviction_ids, allocation_id);
  swap(copy);
}

void KVCacheManager::append_decode_token_exact_in_place(
    RequestId id, const std::vector<std::size_t>& eviction_ids,
    const std::optional<std::size_t>& allocation_id) {
  auto found = block_tables_.find(id);
  if (found == block_tables_.end())
    throw std::out_of_range("unknown KV request ID");
  if (represented_token_count_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("represented KV token count overflow");
  const bool new_block = decode_requires_new_block(id);
  std::vector<std::size_t> expected_evictions;
  std::optional<std::size_t> expected_allocation;
  if (new_block) {
    if (free_blocks_.empty()) {
      const auto order = eligible_eviction_order();
      if (order.empty())
        throw KVCapacityError("insufficient KV capacity for decode token");
      expected_evictions.push_back(order.front());
    }
    std::set<std::size_t> available = free_blocks_;
    available.insert(expected_evictions.begin(), expected_evictions.end());
    expected_allocation = *available.begin();
  }
  if (eviction_ids != expected_evictions || allocation_id != expected_allocation)
    throw std::logic_error("planned decode physical IDs differ from actual IDs");
  evict_exact_blocks_in_place(eviction_ids);
  const std::uint64_t epoch = next_access_epoch();
  if (new_block) {
    const std::size_t block_id = *allocation_id;
    free_blocks_.erase(block_id);
    found->second.push_back(block_id); auto& block = blocks_[block_id];
    block.state = PhysicalKVBlockState::InUse; block.owner = id; block.reference_count = 1;
    block.valid_token_count = 1; block.last_access_epoch = epoch; block.prefix_key.reset();
    block.provenance = KVBlockProvenance::PrivateDecode;
  } else {
    auto& block = blocks_[found->second.back()]; ++block.valid_token_count; block.last_access_epoch = epoch;
  }
  ++represented_token_count_; access_epoch_ = epoch; update_peak();
  if (!validate_invariants()) throw std::logic_error("invalid KV manager state after decode append");
}

void KVCacheManager::release_request(RequestId id) {
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state before request release");
  KVCacheManager copy(*this); copy.release_request_in_place(id); swap(copy);
}
void KVCacheManager::release_request_in_place(RequestId id) {
  auto found = block_tables_.find(id); if (found == block_tables_.end()) throw std::out_of_range("unknown KV request ID");
  for (std::size_t block_id : found->second) {
    auto& block = blocks_[block_id];
    if (block.reference_count == 0) throw std::logic_error("KV reference count underflow");
    --block.reference_count;
    if (block.reference_count != 0) { block.state = PhysicalKVBlockState::InUse; block.owner.reset(); continue; }
    if (block.prefix_key.has_value()) { block.state = PhysicalKVBlockState::Cached; block.owner.reset(); continue; }
    if (block.valid_token_count > represented_token_count_) throw std::logic_error("represented KV token count underflow");
    represented_token_count_ -= block.valid_token_count;
    block.state = PhysicalKVBlockState::Free; block.owner.reset(); block.valid_token_count = 0;
    block.last_access_epoch = 0; block.provenance.reset(); free_blocks_.insert(block_id);
  }
  block_tables_.erase(found);
  request_provenance_.erase(id);
  if (!validate_invariants()) throw std::logic_error("invalid KV manager state after release");
}

std::vector<std::size_t> KVCacheManager::eligible_eviction_order() const {
  std::vector<std::size_t> result;
  for (const auto& block : blocks_) if (block.state == PhysicalKVBlockState::Cached) result.push_back(block.block_id);
  std::sort(result.begin(), result.end(), [this](std::size_t a, std::size_t b) {
    if (blocks_[a].last_access_epoch != blocks_[b].last_access_epoch)
      return blocks_[a].last_access_epoch < blocks_[b].last_access_epoch;
    return a < b;
  });
  return result;
}
std::vector<std::size_t> KVCacheManager::evict_unused_blocks(std::size_t count) {
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state before prefix eviction");
  KVCacheManager copy(*this); auto ids = copy.evict_unused_blocks_in_place(count); swap(copy); return ids;
}
std::vector<std::size_t> KVCacheManager::evict_unused_blocks_in_place(std::size_t count) {
  auto order = eligible_eviction_order();
  if (count > order.size()) throw std::runtime_error("insufficient evictable prefix-cache blocks");
  order.resize(count);
  evict_exact_blocks_in_place(order);
  if (!validate_invariants())
    throw std::logic_error("invalid KV manager state after eviction");
  return order;
}

void KVCacheManager::evict_exact_blocks_in_place(
    const std::vector<std::size_t>& ids) {
  auto order = eligible_eviction_order();
  if (ids.size() > order.size())
    throw std::runtime_error("insufficient evictable prefix-cache blocks");
  order.resize(ids.size());
  if (ids != order)
    throw std::logic_error("planned eviction IDs differ from deterministic LRU IDs");
  PrefixCacheMetrics metrics = prefix_metrics_;
  for (std::size_t id : ids) {
    if (fail_for_test(test::KVCacheFailurePoint::EvictionCounter))
      throw std::overflow_error("injected prefix eviction counter overflow");
    metrics.eviction_count = inc(metrics.eviction_count,
                                 "prefix eviction count overflow");
    auto& block = blocks_[id]; const auto key = *block.prefix_key;
    auto bucket = prefix_index_.find(key.block_hash);
    if (bucket == prefix_index_.end()) throw std::logic_error("missing prefix index entry");
    auto pos = std::find(bucket->second.begin(), bucket->second.end(), id);
    if (pos == bucket->second.end()) throw std::logic_error("missing prefix index block");
    bucket->second.erase(pos); if (bucket->second.empty()) prefix_index_.erase(bucket);
    if (block.valid_token_count > represented_token_count_) throw std::logic_error("represented KV token count underflow");
    represented_token_count_ -= block.valid_token_count;
    block.state = PhysicalKVBlockState::Free; block.owner.reset(); block.reference_count = 0;
    block.valid_token_count = 0; block.last_access_epoch = 0; block.prefix_key.reset();
    block.provenance.reset(); free_blocks_.insert(id);
  }
  prefix_metrics_ = metrics;
}

std::vector<std::size_t> KVCacheManager::block_table(RequestId id) const {
  const auto found = block_tables_.find(id); if (found == block_tables_.end()) throw std::out_of_range("unknown KV request ID"); return found->second;
}
std::size_t KVCacheManager::allocated_block_count() const noexcept { return blocks_.size() - free_blocks_.size(); }
std::size_t KVCacheManager::cached_block_count() const noexcept {
  return static_cast<std::size_t>(std::count_if(blocks_.begin(), blocks_.end(), [](const auto& b){ return b.state == PhysicalKVBlockState::Cached; }));
}
std::size_t KVCacheManager::referenced_shared_block_count() const noexcept {
  return static_cast<std::size_t>(std::count_if(blocks_.begin(), blocks_.end(), [](const auto& b){ return b.prefix_key.has_value() && b.reference_count > 0; }));
}
std::uint64_t KVCacheManager::internal_fragmentation_tokens() const noexcept {
  return static_cast<std::uint64_t>(allocated_block_count()) * config_.block_size_tokens() - represented_token_count_;
}
double KVCacheManager::utilization() const noexcept { return static_cast<double>(allocated_block_count()) / static_cast<double>(blocks_.size()); }
double KVCacheManager::peak_utilization() const noexcept { return static_cast<double>(peak_allocated_block_count_) / static_cast<double>(blocks_.size()); }
void KVCacheManager::update_peak() noexcept { peak_allocated_block_count_ = std::max(peak_allocated_block_count_, allocated_block_count()); }

bool KVCacheManager::validate_invariants() const noexcept {
  if (blocks_.size() != config_.total_num_blocks() ||
      peak_allocated_block_count_ > blocks_.size() ||
      block_tables_.size() != request_provenance_.size()) return false;
  std::vector<std::uint64_t> occurrences(blocks_.size(), 0);
  std::vector<std::optional<RequestId>> sole_request(blocks_.size());
  for (const auto& entry : block_tables_) {
    const auto provenance_it = request_provenance_.find(entry.first);
    if (provenance_it == request_provenance_.end()) return false;
    const auto& provenance = provenance_it->second;
    if (provenance.request_id != entry.first) return false;
    const std::size_t prompt_blocks = provenance.decode_block_start_position;
    if (prompt_blocks > entry.second.size() ||
        provenance.matched_shared_prefix_blocks >
            provenance.cache_eligible_full_prompt_blocks ||
        provenance.computed_full_begin != provenance.matched_shared_prefix_blocks ||
        provenance.computed_full_end != provenance.cache_eligible_full_prompt_blocks)
      return false;
    if (provenance.private_prompt_tail_position.has_value()) {
      if (*provenance.private_prompt_tail_position !=
              provenance.cache_eligible_full_prompt_blocks ||
          prompt_blocks != *provenance.private_prompt_tail_position + 1)
        return false;
    } else if (prompt_blocks != provenance.cache_eligible_full_prompt_blocks &&
               provenance.exact_prompt_tokens.has_value()) {
      return false;
    }
    if (provenance.exact_prompt_tokens) {
      if (provenance.total_prompt_token_count !=
              provenance.exact_prompt_tokens->size() ||
          provenance.cache_eligible_full_prompt_blocks !=
              provenance.exact_prompt_tokens->size() /
                  static_cast<std::size_t>(config_.block_size_tokens()) ||
          provenance.publication_eligible != prefix_config_.enabled())
        return false;
      const bool tail = provenance.exact_prompt_tokens->size() %
                            static_cast<std::size_t>(config_.block_size_tokens()) != 0;
      if (tail != provenance.private_prompt_tail_position.has_value()) return false;
      if (provenance.publication_completed &&
          (!provenance.publication_eligible ||
           provenance.computed_full_begin == provenance.computed_full_end))
        return false;
    } else {
      bool prompt_block_count_matches = false;
      try {
        prompt_block_count_matches =
            prompt_blocks == blocks_required(provenance.total_prompt_token_count);
      } catch (...) { return false; }
      if (provenance.publication_eligible || provenance.publication_completed ||
          provenance.cache_eligible_full_prompt_blocks != 0 ||
          provenance.matched_shared_prefix_blocks != 0 ||
          provenance.computed_full_begin != 0 || provenance.computed_full_end != 0 ||
          provenance.private_prompt_tail_position.has_value() ||
          !prompt_block_count_matches)
        return false;
    }
    std::set<std::size_t> unique;
    for (std::size_t position = 0; position < entry.second.size(); ++position) {
      const std::size_t id = entry.second[position];
      if (id >= blocks_.size() || !unique.insert(id).second) return false;
      if (!sole_request[id]) sole_request[id] = entry.first;
      ++occurrences[id];
      const auto block_provenance = blocks_[id].provenance;
      if (!block_provenance || !valid_provenance(*block_provenance)) return false;
      if (!provenance.exact_prompt_tokens) {
        if (position < prompt_blocks) {
          if (*block_provenance != KVBlockProvenance::PrivatePromptCountOnly)
            return false;
        } else if (*block_provenance != KVBlockProvenance::PrivateDecode) {
          return false;
        }
        if (position < prompt_blocks) {
          const std::uint64_t offset =
              static_cast<std::uint64_t>(position) * config_.block_size_tokens();
          const std::uint64_t base = std::min(
              config_.block_size_tokens(),
              provenance.total_prompt_token_count - offset);
          if (blocks_[id].valid_token_count < base) return false;
        }
      } else if (position < provenance.matched_shared_prefix_blocks) {
        if (*block_provenance != KVBlockProvenance::SharedPromptFull &&
            *block_provenance != KVBlockProvenance::ComputedPromptFull)
          return false;
      } else if (position < provenance.cache_eligible_full_prompt_blocks) {
        if (*block_provenance != KVBlockProvenance::ComputedPromptFull &&
            *block_provenance != KVBlockProvenance::SharedPromptFull)
          return false;
      } else if (position < prompt_blocks) {
        if (*block_provenance != KVBlockProvenance::PrivatePromptTail) return false;
        const std::uint64_t tail_tokens = provenance.total_prompt_token_count %
                                          config_.block_size_tokens();
        if (tail_tokens == 0 || blocks_[id].valid_token_count < tail_tokens)
          return false;
      } else if (*block_provenance != KVBlockProvenance::PrivateDecode) {
        return false;
      }
    }

    if (provenance.exact_prompt_tokens) {
      std::uint64_t parent = kPrefixCacheRootHash;
      for (std::size_t position = 0;
           position < provenance.cache_eligible_full_prompt_blocks; ++position) {
        TokenBlockKey expected;
        try {
          expected = make_key(parent, *provenance.exact_prompt_tokens,
                              position * static_cast<std::size_t>(config_.block_size_tokens()),
                              prefix_config_.model_namespace(),
                              prefix_config_.cache_salt());
        } catch (...) { return false; }
        const auto& block = blocks_[entry.second[position]];
        if (block.prefix_key &&
            (!exact_key_material_equal(*block.prefix_key, expected) ||
             block.prefix_key->block_hash != expected.block_hash))
          return false;
        if (position < provenance.matched_shared_prefix_blocks && !block.prefix_key)
          return false;
        if (!provenance.publication_completed &&
            position >= provenance.computed_full_begin && block.prefix_key)
          return false;
        if (provenance.publication_completed &&
            position >= provenance.computed_full_begin && !block.prefix_key) {
          bool canonical_exists = false;
          const auto bucket = prefix_index_.find(expected.block_hash);
          if (bucket != prefix_index_.end()) {
            for (std::size_t candidate : bucket->second) {
              if (candidate < blocks_.size() && blocks_[candidate].prefix_key &&
                  exact_key_material_equal(*blocks_[candidate].prefix_key, expected)) {
                canonical_exists = true;
                break;
              }
            }
          }
          if (!canonical_exists) return false;
        }
        parent = expected.block_hash;
      }
    }
  }
  std::uint64_t represented = 0;
  for (std::size_t id = 0; id < blocks_.size(); ++id) {
    const auto& b = blocks_[id];
    if (b.block_id != id || !valid_state(b.state) ||
        b.valid_token_count > config_.block_size_tokens() ||
        (b.provenance && !valid_provenance(*b.provenance))) return false;
    const bool free = free_blocks_.count(id) != 0;
    bool indexed = false;
    if (b.prefix_key) {
      const auto bucket = prefix_index_.find(b.prefix_key->block_hash);
      indexed = bucket != prefix_index_.end() && std::count(bucket->second.begin(), bucket->second.end(), id) == 1;
    }
    if (b.state == PhysicalKVBlockState::Free) {
      if (!free || occurrences[id] || b.owner || b.reference_count ||
          b.valid_token_count || b.last_access_epoch || b.prefix_key || indexed ||
          b.provenance) return false;
    } else {
      if (free || b.valid_token_count == 0 || b.last_access_epoch == 0 ||
          b.last_access_epoch > access_epoch_ || !b.provenance) return false;
      if (represented > std::numeric_limits<std::uint64_t>::max() - b.valid_token_count)
        return false;
      represented += b.valid_token_count;
      const bool cacheable = *b.provenance == KVBlockProvenance::SharedPromptFull ||
                             *b.provenance == KVBlockProvenance::ComputedPromptFull;
      if (cacheable && b.valid_token_count != config_.block_size_tokens()) return false;
      if (!cacheable && (b.prefix_key || indexed)) return false;
      if (b.state == PhysicalKVBlockState::Cached) {
        if (occurrences[id] || b.reference_count || b.owner || !b.prefix_key ||
            !indexed || !cacheable) return false;
      } else {
        if (b.reference_count == 0 || occurrences[id] != b.reference_count) return false;
        if (b.prefix_key) {
          if (!indexed || b.owner || !cacheable) return false;
        } else if (occurrences[id] != 1 || !b.owner ||
                   !sole_request[id] || *b.owner != *sole_request[id]) return false;
      }
    }
  }
  if (represented != represented_token_count_) return false;
  std::vector<std::size_t> all_indexed;
  for (const auto& bucket : prefix_index_) {
    if (bucket.second.empty()) return false;
    std::set<std::size_t> seen;
    for (std::size_t id : bucket.second) {
      if (id >= blocks_.size() || !seen.insert(id).second || !blocks_[id].prefix_key ||
          blocks_[id].prefix_key->block_hash != bucket.first || blocks_[id].state == PhysicalKVBlockState::Free) return false;
      const auto& key = *blocks_[id].prefix_key;
      bool hash_matches = false;
      try { hash_matches = hash_function_(key) == key.block_hash; }
      catch (...) { return false; }
      if (key.token_ids.size() != static_cast<std::size_t>(config_.block_size_tokens()) ||
          key.model_namespace != prefix_config_.model_namespace() ||
          key.cache_salt != prefix_config_.cache_salt() ||
          !hash_matches)
        return false;
      all_indexed.push_back(id);
    }
  }
  for (std::size_t i = 0; i < all_indexed.size(); ++i) {
    for (std::size_t j = i + 1; j < all_indexed.size(); ++j) {
      if (all_indexed[i] == all_indexed[j] ||
          exact_key_material_equal(*blocks_[all_indexed[i]].prefix_key,
                                   *blocks_[all_indexed[j]].prefix_key))
        return false;
    }
  }
  return true;
}

void KVCacheManager::swap(KVCacheManager& other) noexcept {
  using std::swap; swap(config_, other.config_); swap(prefix_config_, other.prefix_config_);
  swap(hash_function_, other.hash_function_); blocks_.swap(other.blocks_); free_blocks_.swap(other.free_blocks_);
  block_tables_.swap(other.block_tables_); request_provenance_.swap(other.request_provenance_);
  prefix_index_.swap(other.prefix_index_); swap(prefix_metrics_, other.prefix_metrics_);
  swap(represented_token_count_, other.represented_token_count_); swap(peak_allocated_block_count_, other.peak_allocated_block_count_);
  swap(allocation_failure_count_, other.allocation_failure_count_); swap(access_epoch_, other.access_epoch_);
}

}  // namespace llm_lab::serving
