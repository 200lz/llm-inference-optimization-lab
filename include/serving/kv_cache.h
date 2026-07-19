#pragma once

#include "serving/request.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llm_lab::serving {

class KVCapacityError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

namespace test {
struct KVCacheManagerTestAccess;
struct FixCTestAccess;
enum class KVCacheFailurePoint {
  None,
  CacheLookupCounter,
  CacheHitCounter,
  CacheMissCounter,
  EligibleTokenCounter,
  MatchedBlockCounter,
  MatchedTokenCounter,
  ReusedRequestCounter,
  SavedPrefillCounter,
  CollisionCounter,
  ReferenceCounter,
  EvictionCounter,
  AccessEpoch,
  PublicationAccessEpoch,
  AllocationFailureCounter,
  RepresentedTokenArithmetic,
  CanonicalIndexConflict,
};
}

class KVCacheConfig {
 public:
  explicit KVCacheConfig(std::size_t total_num_blocks,
                         std::uint64_t block_size_tokens = 16);
  std::size_t total_num_blocks() const noexcept { return total_num_blocks_; }
  std::uint64_t block_size_tokens() const noexcept { return block_size_tokens_; }
  std::uint64_t total_capacity_tokens() const noexcept { return total_capacity_tokens_; }
 private:
  std::size_t total_num_blocks_;
  std::uint64_t block_size_tokens_;
  std::uint64_t total_capacity_tokens_;
};

enum class PrefixCacheEvictionPolicy { DeterministicLRU };

class PrefixCacheConfig {
 public:
  explicit PrefixCacheConfig(bool enabled = false, std::string cache_salt = {},
      std::string model_namespace = {},
      PrefixCacheEvictionPolicy eviction_policy = PrefixCacheEvictionPolicy::DeterministicLRU);
  bool enabled() const noexcept { return enabled_; }
  const std::string& cache_salt() const noexcept { return cache_salt_; }
  const std::string& model_namespace() const noexcept { return model_namespace_; }
  PrefixCacheEvictionPolicy eviction_policy() const noexcept { return eviction_policy_; }
 private:
  bool enabled_;
  std::string cache_salt_;
  std::string model_namespace_;
  PrefixCacheEvictionPolicy eviction_policy_;
};

struct TokenBlockKey {
  std::uint64_t parent_hash{0};
  std::vector<std::int32_t> token_ids;
  std::string model_namespace;
  std::string cache_salt;
  std::uint64_t block_hash{0};
};

bool operator==(const TokenBlockKey& lhs, const TokenBlockKey& rhs) noexcept;
bool exact_key_material_equal(const TokenBlockKey& lhs,
                              const TokenBlockKey& rhs) noexcept;

// FNV-1a 64 over tagged, little-endian, length-prefixed key material.
std::uint64_t stable_token_block_hash(const TokenBlockKey& key) noexcept;
constexpr std::uint64_t kPrefixCacheRootHash = UINT64_C(0xcbf29ce484222325);
using PrefixHashFunction = std::function<std::uint64_t(const TokenBlockKey&)>;

enum class PrefixLookupKind { Disabled, Miss, PartialHit, AllEligibleBlocksHit };
struct PrefixLookupResult {
  std::size_t matched_block_count{0};
  std::uint64_t matched_token_count{0};
  std::vector<std::size_t> physical_block_ids;
  std::uint64_t next_parent_hash{kPrefixCacheRootHash};
  PrefixLookupKind kind{PrefixLookupKind::Miss};
  std::uint64_t eligible_token_count{0};
  std::uint64_t collision_verification_count{0};
};

struct PrefixCacheMetrics {
  std::uint64_t cache_lookup_count{0};
  std::uint64_t cache_hit_lookup_count{0};
  std::uint64_t cache_miss_lookup_count{0};
  std::uint64_t matched_block_count{0};
  std::uint64_t matched_token_count{0};
  std::uint64_t total_cache_eligible_prompt_tokens_looked_up{0};
  std::uint64_t reused_request_count{0};
  std::uint64_t saved_prefill_token_count{0};
  std::uint64_t collision_verification_count{0};
  std::uint64_t eviction_count{0};
  std::optional<double> prefix_token_hit_rate() const noexcept;
};

enum class PhysicalKVBlockState { Free, InUse, Cached };
enum class KVBlockProvenance {
  SharedPromptFull,
  ComputedPromptFull,
  PrivatePromptTail,
  PrivatePromptCountOnly,
  PrivateDecode,
};
struct PhysicalKVBlock {
  std::size_t block_id;
  PhysicalKVBlockState state{PhysicalKVBlockState::Free};
  std::optional<RequestId> owner;
  std::uint64_t reference_count{0};
  std::uint64_t valid_token_count{0};
  std::uint64_t last_access_epoch{0};
  std::optional<TokenBlockKey> prefix_key;
  std::optional<KVBlockProvenance> provenance;
};

inline bool operator==(const PhysicalKVBlock& lhs, const PhysicalKVBlock& rhs) noexcept {
  return lhs.block_id == rhs.block_id && lhs.state == rhs.state &&
    lhs.owner == rhs.owner && lhs.reference_count == rhs.reference_count &&
    lhs.valid_token_count == rhs.valid_token_count &&
    lhs.last_access_epoch == rhs.last_access_epoch && lhs.prefix_key == rhs.prefix_key &&
    lhs.provenance == rhs.provenance;
}

struct RequestKVPromptProvenance {
  RequestId request_id;
  bool publication_eligible{false};
  std::optional<std::vector<std::int32_t>> exact_prompt_tokens;
  std::uint64_t total_prompt_token_count{0};
  std::size_t cache_eligible_full_prompt_blocks{0};
  std::size_t matched_shared_prefix_blocks{0};
  std::size_t computed_full_begin{0};
  std::size_t computed_full_end{0};
  std::optional<std::size_t> private_prompt_tail_position;
  std::size_t decode_block_start_position{0};
  bool publication_completed{false};
};

class KVCacheManager {
 public:
  explicit KVCacheManager(KVCacheConfig config,
                          PrefixCacheConfig prefix_config = PrefixCacheConfig(),
                          PrefixHashFunction hash_function = stable_token_block_hash);
  std::size_t blocks_required(std::uint64_t token_count) const;
  bool can_allocate_blocks(std::size_t count) const noexcept;
  bool can_allocate_tokens(std::uint64_t token_count) const;
  bool decode_requires_new_block(RequestId request_id) const;

  void allocate_prompt(RequestId request_id, std::uint64_t prompt_token_count);
  void allocate_prompt_exact(RequestId request_id,
      std::uint64_t prompt_token_count,
      const std::vector<std::size_t>& eviction_ids,
      const std::vector<std::size_t>& allocation_ids);
  PrefixLookupResult find_longest_cached_prefix(const std::vector<std::int32_t>& prompt_tokens) const;
  PrefixLookupResult find_longest_cached_prefix(const std::vector<std::int32_t>& prompt_tokens,
      std::string_view model_namespace, std::string_view cache_salt) const;
  // Acquires exactly the immutable lookup snapshot and allocates its suffix;
  // publication is deliberately separate for same-plan visibility.
  void allocate_prompt_with_prefix(RequestId request_id,
      const std::vector<std::int32_t>& prompt_tokens,
      const PrefixLookupResult& lookup,
      const std::set<std::size_t>& protected_cached_blocks = {});
  void allocate_prompt_with_prefix_exact(RequestId request_id,
      const std::vector<std::int32_t>& prompt_tokens,
      const PrefixLookupResult& lookup,
      const std::vector<std::size_t>& eviction_ids,
      const std::vector<std::size_t>& allocation_ids);
  void insert_completed_prompt_blocks(RequestId request_id);
  void append_decode_token(RequestId request_id);
  void append_decode_token_exact(RequestId request_id,
      const std::vector<std::size_t>& eviction_ids,
      const std::optional<std::size_t>& allocation_id);
  void release_request(RequestId request_id);
  std::vector<std::size_t> evict_unused_blocks(std::size_t required_block_count);
  std::vector<std::size_t> eligible_eviction_order() const;

  std::vector<std::size_t> block_table(RequestId request_id) const;
  std::vector<PhysicalKVBlock> physical_blocks() const { return blocks_; }
  std::map<RequestId, std::vector<std::size_t>> block_tables() const { return block_tables_; }
  std::set<std::size_t> free_block_ids() const { return free_blocks_; }
  std::size_t allocated_block_count() const noexcept;
  std::size_t free_block_count() const noexcept { return free_blocks_.size(); }
  std::size_t cached_block_count() const noexcept;
  std::size_t referenced_shared_block_count() const noexcept;
  std::uint64_t total_capacity_tokens() const noexcept { return config_.total_capacity_tokens(); }
  std::uint64_t represented_token_count() const noexcept { return represented_token_count_; }
  std::uint64_t internal_fragmentation_tokens() const noexcept;
  double utilization() const noexcept;
  double peak_utilization() const noexcept;
  std::size_t peak_allocated_block_count() const noexcept { return peak_allocated_block_count_; }
  std::uint64_t allocation_failure_count() const noexcept { return allocation_failure_count_; }
  std::uint64_t access_epoch() const noexcept { return access_epoch_; }
  const KVCacheConfig& config() const noexcept { return config_; }
  const PrefixCacheConfig& prefix_cache_config() const noexcept { return prefix_config_; }
  const PrefixCacheMetrics& prefix_cache_metrics() const noexcept { return prefix_metrics_; }
  bool validate_invariants() const noexcept;
  void swap(KVCacheManager& other) noexcept;

 private:
  using PrefixIndex = std::map<std::uint64_t, std::vector<std::size_t>>;
  void allocate_prompt_exact_in_place(RequestId, std::uint64_t,
                                      const std::vector<std::size_t>&,
                                      const std::vector<std::size_t>&);
  void allocate_prompt_with_prefix_exact_in_place(RequestId,
      const std::vector<std::int32_t>&, const PrefixLookupResult&,
      const std::vector<std::size_t>&, const std::vector<std::size_t>&);
  void insert_completed_prompt_blocks_in_place(RequestId);
  void append_decode_token_exact_in_place(RequestId,
      const std::vector<std::size_t>&, const std::optional<std::size_t>&);
  void release_request_in_place(RequestId);
  std::vector<std::size_t> evict_unused_blocks_in_place(std::size_t);
  void evict_exact_blocks_in_place(const std::vector<std::size_t>&);
  TokenBlockKey make_key(std::uint64_t parent, const std::vector<std::int32_t>&,
                         std::size_t offset, std::string_view ns, std::string_view salt) const;
  void note_allocation_failure();
  std::uint64_t next_access_epoch() const;
  void update_peak() noexcept;
  bool fail_for_test(test::KVCacheFailurePoint point) const noexcept {
    return failure_point_for_test_ == point;
  }

  KVCacheConfig config_;
  PrefixCacheConfig prefix_config_;
  PrefixHashFunction hash_function_;
  std::vector<PhysicalKVBlock> blocks_;
  std::set<std::size_t> free_blocks_;
  std::map<RequestId, std::vector<std::size_t>> block_tables_;
  std::map<RequestId, RequestKVPromptProvenance> request_provenance_;
  PrefixIndex prefix_index_;
  PrefixCacheMetrics prefix_metrics_;
  std::uint64_t represented_token_count_{0};
  std::size_t peak_allocated_block_count_{0};
  std::uint64_t allocation_failure_count_{0};
  std::uint64_t access_epoch_{0};
  test::KVCacheFailurePoint failure_point_for_test_{
      test::KVCacheFailurePoint::None};
  friend struct test::KVCacheManagerTestAccess;
  friend struct test::FixCTestAccess;
  friend class ContinuousBatchingEngine;
};

}  // namespace llm_lab::serving
