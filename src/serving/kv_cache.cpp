#include "serving/kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llm_lab::serving {
namespace {

std::uint64_t checked_increment(std::uint64_t value, const char* message) {
  if (value == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error(message);
  }
  return value + 1;
}

}  // namespace

KVCacheConfig::KVCacheConfig(std::size_t total_num_blocks,
                             std::uint64_t block_size_tokens)
    : total_num_blocks_(total_num_blocks),
      block_size_tokens_(block_size_tokens),
      total_capacity_tokens_(0) {
  if (total_num_blocks_ == 0) {
    throw std::invalid_argument("total_num_blocks must be at least 1");
  }
  if (block_size_tokens_ == 0) {
    throw std::invalid_argument("block_size_tokens must be at least 1");
  }
  if (total_num_blocks_ >
      std::numeric_limits<std::uint64_t>::max() / block_size_tokens_) {
    throw std::overflow_error("KV token capacity is not representable");
  }
  total_capacity_tokens_ =
      static_cast<std::uint64_t>(total_num_blocks_) * block_size_tokens_;
}

KVCacheManager::KVCacheManager(KVCacheConfig config) : config_(config) {
  if (config_.total_num_blocks() > blocks_.max_size()) {
    throw std::length_error("KV block metadata count is not representable");
  }
  blocks_.reserve(config_.total_num_blocks());
  for (std::size_t id = 0; id < config_.total_num_blocks(); ++id) {
    blocks_.push_back(PhysicalKVBlock{id, PhysicalKVBlockState::Free,
                                      std::nullopt, 0, 0, 0});
    free_blocks_.insert(id);
  }
}

std::size_t KVCacheManager::blocks_required(
    std::uint64_t token_count) const {
  if (token_count == 0) {
    return 0;
  }
  const std::uint64_t quotient = token_count / config_.block_size_tokens();
  const std::uint64_t remainder = token_count % config_.block_size_tokens();
  const std::uint64_t required = quotient + (remainder == 0 ? 0 : 1);
  if (required > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("required KV block count is not representable");
  }
  return static_cast<std::size_t>(required);
}

bool KVCacheManager::can_allocate_blocks(std::size_t count) const noexcept {
  return count <= free_blocks_.size();
}

bool KVCacheManager::can_allocate_tokens(std::uint64_t token_count) const {
  return can_allocate_blocks(blocks_required(token_count));
}

bool KVCacheManager::decode_requires_new_block(RequestId request_id) const {
  const auto found = block_tables_.find(request_id);
  if (found == block_tables_.end()) {
    throw std::out_of_range("unknown KV request ID");
  }
  if (found->second.empty()) {
    return true;
  }
  const std::size_t id = found->second.back();
  if (id >= blocks_.size()) {
    throw std::logic_error("invalid KV block table");
  }
  const PhysicalKVBlock& block = blocks_[id];
  if (block.state != PhysicalKVBlockState::InUse ||
      block.owner != request_id || block.reference_count != 1 ||
      block.valid_token_count == 0 ||
      block.valid_token_count > config_.block_size_tokens()) {
    throw std::logic_error("invalid final KV block metadata");
  }
  return block.valid_token_count == config_.block_size_tokens();
}

void KVCacheManager::note_allocation_failure() {
  allocation_failure_count_ = checked_increment(
      allocation_failure_count_, "KV allocation failure count overflow");
}

std::uint64_t KVCacheManager::next_access_epoch() const {
  return checked_increment(access_epoch_, "KV access epoch overflow");
}

void KVCacheManager::allocate_prompt(RequestId request_id,
                                     std::uint64_t prompt_token_count) {
  if (!validate_invariants()) {
    throw std::logic_error("invalid KV manager state before prompt allocation");
  }
  if (block_tables_.find(request_id) != block_tables_.end()) {
    throw std::invalid_argument("duplicate KV request ID");
  }
  const std::size_t required = blocks_required(prompt_token_count);
  if (represented_token_count_ >
      std::numeric_limits<std::uint64_t>::max() - prompt_token_count) {
    throw std::overflow_error("represented KV token count overflow");
  }
  if (required != 0) {
    (void)next_access_epoch();
  }
  if (required > std::vector<std::size_t>().max_size() ||
      block_tables_.size() == block_tables_.max_size()) {
    throw std::length_error("KV request table capacity overflow");
  }
  if (!can_allocate_blocks(required)) {
    note_allocation_failure();
    throw std::runtime_error("insufficient KV capacity for prompt");
  }
  KVCacheManager prepared(*this);
  prepared.allocate_prompt_in_place(request_id, prompt_token_count);
  swap(prepared);
}

void KVCacheManager::allocate_prompt_in_place(
    RequestId request_id, std::uint64_t prompt_token_count) {
  if (!validate_invariants()) {
    throw std::logic_error("invalid KV manager state before prompt allocation");
  }
  if (block_tables_.find(request_id) != block_tables_.end()) {
    throw std::invalid_argument("duplicate KV request ID");
  }
  const std::size_t required = blocks_required(prompt_token_count);
  if (!can_allocate_blocks(required)) {
    throw std::runtime_error("insufficient KV capacity for prompt");
  }
  if (represented_token_count_ >
      std::numeric_limits<std::uint64_t>::max() - prompt_token_count) {
    throw std::overflow_error("represented KV token count overflow");
  }
  const std::uint64_t prepared_represented_token_count =
      represented_token_count_ + prompt_token_count;
  const std::uint64_t epoch =
      required == 0 ? access_epoch_ : next_access_epoch();
  std::vector<std::size_t> table;
  table.reserve(required);
  auto free_it = free_blocks_.begin();
  for (std::size_t index = 0; index < required; ++index, ++free_it) {
    table.push_back(*free_it);
  }
  if (block_tables_.size() == block_tables_.max_size()) {
    throw std::length_error("KV request table capacity overflow");
  }
  block_tables_.emplace(request_id, table);
  for (std::size_t index = 0; index < table.size(); ++index) {
    PhysicalKVBlock& block = blocks_[table[index]];
    block.state = PhysicalKVBlockState::InUse;
    block.owner = request_id;
    block.reference_count = 1;
    const std::uint64_t consumed =
        static_cast<std::uint64_t>(index) * config_.block_size_tokens();
    block.valid_token_count = std::min(
        config_.block_size_tokens(), prompt_token_count - consumed);
    block.last_access_epoch = epoch;
    free_blocks_.erase(table[index]);
  }
  represented_token_count_ = prepared_represented_token_count;
  if (required != 0) {
    access_epoch_ = epoch;
  }
  update_peak();
  if (!validate_invariants()) {
    throw std::logic_error("invalid KV manager state after prompt allocation");
  }
}

void KVCacheManager::append_decode_token(RequestId request_id) {
  if (!validate_invariants()) {
    throw std::logic_error("invalid KV manager state before decode append");
  }
  const auto found = block_tables_.find(request_id);
  if (found == block_tables_.end()) {
    throw std::out_of_range("unknown KV request ID");
  }
  if (represented_token_count_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("represented KV token count overflow");
  }
  (void)next_access_epoch();
  const bool requires_block = decode_requires_new_block(request_id);
  if (requires_block && found->second.size() == found->second.max_size()) {
    throw std::length_error("KV block table capacity overflow");
  }
  if (requires_block && free_blocks_.empty()) {
    note_allocation_failure();
    throw std::runtime_error("insufficient KV capacity for decode token");
  }
  KVCacheManager prepared(*this);
  prepared.append_decode_token_in_place(request_id);
  swap(prepared);
}

void KVCacheManager::append_decode_token_in_place(RequestId request_id) {
  if (!validate_invariants()) {
    throw std::logic_error("invalid KV manager state before decode append");
  }
  auto found = block_tables_.find(request_id);
  if (found == block_tables_.end()) {
    throw std::out_of_range("unknown KV request ID");
  }
  if (represented_token_count_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("represented KV token count overflow");
  }
  const std::uint64_t epoch = next_access_epoch();
  if (decode_requires_new_block(request_id)) {
    if (free_blocks_.empty()) {
      throw std::runtime_error("insufficient KV capacity for decode token");
    }
    if (found->second.size() == found->second.max_size()) {
      throw std::length_error("KV block table capacity overflow");
    }
    const std::size_t id = *free_blocks_.begin();
    found->second.push_back(id);
    PhysicalKVBlock& block = blocks_[id];
    block.state = PhysicalKVBlockState::InUse;
    block.owner = request_id;
    block.reference_count = 1;
    block.valid_token_count = 1;
    block.last_access_epoch = epoch;
    free_blocks_.erase(id);
  } else {
    PhysicalKVBlock& block = blocks_[found->second.back()];
    ++block.valid_token_count;
    block.last_access_epoch = epoch;
  }
  ++represented_token_count_;
  access_epoch_ = epoch;
  update_peak();
  if (!validate_invariants()) {
    throw std::logic_error("invalid KV manager state after decode append");
  }
}

void KVCacheManager::release_request(RequestId request_id) {
  KVCacheManager prepared(*this);
  prepared.release_request_in_place(request_id);
  swap(prepared);
}

void KVCacheManager::release_request_in_place(RequestId request_id) {
  if (!validate_invariants()) {
    throw std::logic_error("invalid KV manager state before release");
  }
  const auto found = block_tables_.find(request_id);
  if (found == block_tables_.end()) {
    throw std::out_of_range("unknown KV request ID");
  }
  std::uint64_t released_tokens = 0;
  for (std::size_t id : found->second) {
    const PhysicalKVBlock& block = blocks_[id];
    if (released_tokens > std::numeric_limits<std::uint64_t>::max() -
                              block.valid_token_count) {
      throw std::overflow_error("released KV token count overflow");
    }
    released_tokens += block.valid_token_count;
  }
  if (released_tokens > represented_token_count_) {
    throw std::logic_error("represented KV token count underflow");
  }
  for (std::size_t id : found->second) {
    PhysicalKVBlock& block = blocks_[id];
    block.state = PhysicalKVBlockState::Free;
    block.owner.reset();
    block.reference_count = 0;
    block.valid_token_count = 0;
    block.last_access_epoch = 0;
    free_blocks_.insert(id);
  }
  represented_token_count_ -= released_tokens;
  block_tables_.erase(found);
  if (!validate_invariants()) {
    throw std::logic_error("invalid KV manager state after release");
  }
}

std::vector<std::size_t> KVCacheManager::block_table(
    RequestId request_id) const {
  const auto found = block_tables_.find(request_id);
  if (found == block_tables_.end()) {
    throw std::out_of_range("unknown KV request ID");
  }
  return found->second;
}

std::size_t KVCacheManager::allocated_block_count() const noexcept {
  return blocks_.size() - free_blocks_.size();
}

std::uint64_t KVCacheManager::internal_fragmentation_tokens() const noexcept {
  return static_cast<std::uint64_t>(allocated_block_count()) *
             config_.block_size_tokens() -
         represented_token_count_;
}

double KVCacheManager::utilization() const noexcept {
  return static_cast<double>(allocated_block_count()) /
         static_cast<double>(config_.total_num_blocks());
}

double KVCacheManager::peak_utilization() const noexcept {
  return static_cast<double>(peak_allocated_block_count_) /
         static_cast<double>(config_.total_num_blocks());
}

void KVCacheManager::update_peak() noexcept {
  peak_allocated_block_count_ =
      std::max(peak_allocated_block_count_, allocated_block_count());
}

bool KVCacheManager::validate_invariants() const noexcept {
  if (blocks_.size() != config_.total_num_blocks() ||
      free_blocks_.size() > blocks_.size() ||
      peak_allocated_block_count_ > blocks_.size()) {
    return false;
  }
  std::uint64_t represented = 0;
  std::size_t owned_count = 0;
  for (std::size_t id = 0; id < blocks_.size(); ++id) {
    const PhysicalKVBlock& block = blocks_[id];
    if (block.block_id != id ||
        block.valid_token_count > config_.block_size_tokens()) {
      return false;
    }
    const bool in_free_pool = free_blocks_.count(id) == 1;
    std::size_t table_occurrences = 0;
    std::optional<RequestId> table_owner;
    for (const auto& entry : block_tables_) {
      for (std::size_t table_id : entry.second) {
        if (table_id == id) {
          ++table_occurrences;
          table_owner = entry.first;
        }
      }
    }
    if (block.state == PhysicalKVBlockState::Free) {
      if (!in_free_pool || table_occurrences != 0 || block.owner.has_value() ||
          block.reference_count != 0 || block.valid_token_count != 0 ||
          block.last_access_epoch != 0) {
        return false;
      }
    } else if (block.state == PhysicalKVBlockState::InUse) {
      if (in_free_pool || table_occurrences != 1 || !block.owner.has_value() ||
          table_owner != block.owner || block.reference_count != 1 ||
          block.valid_token_count == 0 || block.last_access_epoch == 0 ||
          block.last_access_epoch > access_epoch_) {
        return false;
      }
      ++owned_count;
      if (represented > std::numeric_limits<std::uint64_t>::max() -
                            block.valid_token_count) {
        return false;
      }
      represented += block.valid_token_count;
    } else {
      return false;
    }
  }
  for (const auto& entry : block_tables_) {
    for (std::size_t index = 0; index < entry.second.size(); ++index) {
      const std::size_t id = entry.second[index];
      if (id >= blocks_.size()) {
        return false;
      }
      if (index + 1 < entry.second.size() &&
          blocks_[id].valid_token_count != config_.block_size_tokens()) {
        return false;
      }
    }
  }
  return owned_count + free_blocks_.size() == blocks_.size() &&
         represented == represented_token_count_ &&
         represented_token_count_ <= config_.total_capacity_tokens();
}

void KVCacheManager::swap(KVCacheManager& other) noexcept {
  using std::swap;
  swap(config_, other.config_);
  blocks_.swap(other.blocks_);
  free_blocks_.swap(other.free_blocks_);
  block_tables_.swap(other.block_tables_);
  swap(represented_token_count_, other.represented_token_count_);
  swap(peak_allocated_block_count_, other.peak_allocated_block_count_);
  swap(allocation_failure_count_, other.allocation_failure_count_);
  swap(access_epoch_, other.access_epoch_);
}

}  // namespace llm_lab::serving
