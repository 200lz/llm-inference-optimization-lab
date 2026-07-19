#pragma once

#include "serving/request.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace llm_lab::serving {

namespace test {
struct KVCacheManagerTestAccess;
}

class KVCacheConfig {
 public:
  explicit KVCacheConfig(std::size_t total_num_blocks,
                         std::uint64_t block_size_tokens = 16);

  std::size_t total_num_blocks() const noexcept { return total_num_blocks_; }
  std::uint64_t block_size_tokens() const noexcept {
    return block_size_tokens_;
  }
  std::uint64_t total_capacity_tokens() const noexcept {
    return total_capacity_tokens_;
  }

 private:
  std::size_t total_num_blocks_;
  std::uint64_t block_size_tokens_;
  std::uint64_t total_capacity_tokens_;
};

enum class PhysicalKVBlockState { Free, InUse };

struct PhysicalKVBlock {
  std::size_t block_id;
  PhysicalKVBlockState state{PhysicalKVBlockState::Free};
  std::optional<RequestId> owner;
  std::uint64_t reference_count{0};
  std::uint64_t valid_token_count{0};
  std::uint64_t last_access_epoch{0};
};

inline bool operator==(const PhysicalKVBlock& lhs,
                       const PhysicalKVBlock& rhs) noexcept {
  return lhs.block_id == rhs.block_id && lhs.state == rhs.state &&
         lhs.owner == rhs.owner && lhs.reference_count == rhs.reference_count &&
         lhs.valid_token_count == rhs.valid_token_count &&
         lhs.last_access_epoch == rhs.last_access_epoch;
}

class KVCacheManager {
 public:
  explicit KVCacheManager(KVCacheConfig config);

  std::size_t blocks_required(std::uint64_t token_count) const;
  bool can_allocate_blocks(std::size_t count) const noexcept;
  bool can_allocate_tokens(std::uint64_t token_count) const;
  bool decode_requires_new_block(RequestId request_id) const;

  void allocate_prompt(RequestId request_id,
                       std::uint64_t prompt_token_count);
  void append_decode_token(RequestId request_id);
  void release_request(RequestId request_id);

  std::vector<std::size_t> block_table(RequestId request_id) const;
  std::vector<PhysicalKVBlock> physical_blocks() const {
    return blocks_;
  }
  std::map<RequestId, std::vector<std::size_t>> block_tables() const {
    return block_tables_;
  }
  std::set<std::size_t> free_block_ids() const { return free_blocks_; }
  std::size_t allocated_block_count() const noexcept;
  std::size_t free_block_count() const noexcept { return free_blocks_.size(); }
  std::uint64_t total_capacity_tokens() const noexcept {
    return config_.total_capacity_tokens();
  }
  std::uint64_t represented_token_count() const noexcept {
    return represented_token_count_;
  }
  std::uint64_t internal_fragmentation_tokens() const noexcept;
  double utilization() const noexcept;
  double peak_utilization() const noexcept;
  std::size_t peak_allocated_block_count() const noexcept {
    return peak_allocated_block_count_;
  }
  std::uint64_t allocation_failure_count() const noexcept {
    return allocation_failure_count_;
  }
  std::uint64_t access_epoch() const noexcept { return access_epoch_; }
  const KVCacheConfig& config() const noexcept { return config_; }

  bool validate_invariants() const noexcept;
  void swap(KVCacheManager& other) noexcept;

 private:
  void allocate_prompt_in_place(RequestId request_id,
                                std::uint64_t prompt_token_count);
  void append_decode_token_in_place(RequestId request_id);
  void release_request_in_place(RequestId request_id);
  void note_allocation_failure();
  std::uint64_t next_access_epoch() const;
  void update_peak() noexcept;

  KVCacheConfig config_;
  std::vector<PhysicalKVBlock> blocks_;
  std::set<std::size_t> free_blocks_;
  std::map<RequestId, std::vector<std::size_t>> block_tables_;
  std::uint64_t represented_token_count_{0};
  std::size_t peak_allocated_block_count_{0};
  std::uint64_t allocation_failure_count_{0};
  std::uint64_t access_epoch_{0};

  friend struct test::KVCacheManagerTestAccess;
};

}  // namespace llm_lab::serving
