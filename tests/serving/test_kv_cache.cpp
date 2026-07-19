#include "serving/kv_cache.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm_lab::serving::test {

struct KVCacheManagerTestAccess {
  static void set_failure_count(KVCacheManager& manager, std::uint64_t value) {
    manager.allocation_failure_count_ = value;
  }
  static void set_access_epoch(KVCacheManager& manager, std::uint64_t value) {
    manager.access_epoch_ = value;
  }
};

}  // namespace llm_lab::serving::test

namespace {

using namespace llm_lab::serving;
using llm_lab::serving::test::KVCacheManagerTestAccess;

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

struct ManagerSnapshot {
  std::size_t total_blocks;
  std::uint64_t block_size;
  std::uint64_t total_capacity;
  std::vector<PhysicalKVBlock> blocks;
  std::set<std::size_t> free_ids;
  std::map<RequestId, std::vector<std::size_t>> tables;
  std::uint64_t represented;
  std::size_t allocated;
  std::size_t free;
  std::uint64_t fragmentation;
  double utilization;
  std::size_t peak_allocated;
  double peak_utilization;
  std::uint64_t failures;
  std::uint64_t epoch;
};

ManagerSnapshot snapshot(const KVCacheManager& manager) {
  return ManagerSnapshot{
      manager.config().total_num_blocks(), manager.config().block_size_tokens(),
      manager.total_capacity_tokens(), manager.physical_blocks(),
      manager.free_block_ids(), manager.block_tables(),
      manager.represented_token_count(), manager.allocated_block_count(),
      manager.free_block_count(), manager.internal_fragmentation_tokens(),
      manager.utilization(), manager.peak_allocated_block_count(),
      manager.peak_utilization(), manager.allocation_failure_count(),
      manager.access_epoch()};
}

bool same_snapshot(const ManagerSnapshot& lhs, const ManagerSnapshot& rhs) {
  return lhs.total_blocks == rhs.total_blocks &&
         lhs.block_size == rhs.block_size &&
         lhs.total_capacity == rhs.total_capacity && lhs.blocks == rhs.blocks &&
         lhs.free_ids == rhs.free_ids && lhs.tables == rhs.tables &&
         lhs.represented == rhs.represented && lhs.allocated == rhs.allocated &&
         lhs.free == rhs.free && lhs.fragmentation == rhs.fragmentation &&
         lhs.utilization == rhs.utilization &&
         lhs.peak_allocated == rhs.peak_allocated &&
         lhs.peak_utilization == rhs.peak_utilization &&
         lhs.failures == rhs.failures && lhs.epoch == rhs.epoch;
}

void require_only_failure_incremented(const ManagerSnapshot& before,
                                      const ManagerSnapshot& after,
                                      const std::string& message) {
  ManagerSnapshot expected = before;
  ++expected.failures;
  require(same_snapshot(expected, after), message);
}

void test_configuration_and_required_blocks() {
  require_throws<std::invalid_argument>([] { (void)KVCacheConfig(0, 4); },
                                        "zero block count accepted");
  require_throws<std::invalid_argument>([] { (void)KVCacheConfig(4, 0); },
                                        "zero block size accepted");
  require_throws<std::overflow_error>(
      [] { (void)KVCacheConfig(2, std::numeric_limits<std::uint64_t>::max()); },
      "capacity overflow accepted");

  KVCacheManager manager(KVCacheConfig(4, 4));
  require(manager.blocks_required(0) == 0 &&
              manager.blocks_required(4) == 1 &&
              manager.blocks_required(5) == 2,
          "block rounding is wrong");
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t expected_maximum = maximum / 4 + 1;
  if (expected_maximum > std::numeric_limits<std::size_t>::max()) {
    require_throws<std::overflow_error>(
        [&] { (void)manager.blocks_required(maximum); },
        "unrepresentable maximum block count accepted");
  } else {
    require(manager.blocks_required(maximum) ==
                static_cast<std::size_t>(expected_maximum),
            "maximum token count wrapped during ceiling division");
  }
  require(manager.total_capacity_tokens() == 16 &&
              manager.validate_invariants(),
          "safe capacity setup is wrong");
}

void test_prompt_decode_release_metrics_and_epoch() {
  KVCacheManager manager(KVCacheConfig(4, 4));
  manager.allocate_prompt({10}, 0);
  require(manager.block_table({10}).empty() && manager.access_epoch() == 0 &&
              manager.validate_invariants(),
          "zero prompt registration or epoch is wrong");
  const auto retained_diagnostics = manager.block_tables();
  const auto retained_blocks = manager.physical_blocks();
  const auto retained_free_ids = manager.free_block_ids();

  manager.append_decode_token({10});
  require(manager.block_table({10}) == std::vector<std::size_t>({0}) &&
              manager.physical_blocks()[0].valid_token_count == 1 &&
              manager.physical_blocks()[0].last_access_epoch == 1 &&
              manager.access_epoch() == 1 && manager.validate_invariants(),
          "append from an empty table is wrong");
  require(retained_diagnostics.at({10}).empty(),
          "copied diagnostics changed after manager mutation");
  require(retained_blocks[0].state == PhysicalKVBlockState::Free &&
              retained_free_ids == std::set<std::size_t>({0, 1, 2, 3}),
          "retained physical/free diagnostics changed after mutation");
  manager.release_request({10});

  manager.allocate_prompt({1}, 5);
  require(manager.block_table({1}) == std::vector<std::size_t>({0, 1}) &&
              manager.represented_token_count() == 5 &&
              manager.internal_fragmentation_tokens() == 3 &&
              manager.utilization() == 0.5 && manager.access_epoch() == 2 &&
              manager.physical_blocks()[0].last_access_epoch == 2 &&
              manager.physical_blocks()[1].last_access_epoch == 2 &&
              manager.validate_invariants(),
          "prompt allocation accounting or epoch is wrong");
  manager.append_decode_token({1});
  require(manager.physical_blocks()[1].valid_token_count == 2 &&
              manager.physical_blocks()[1].last_access_epoch == 3,
          "partial-tail append epoch is wrong");
  manager.append_decode_token({1});
  manager.append_decode_token({1});
  require(manager.decode_requires_new_block({1}) &&
              manager.physical_blocks()[1].valid_token_count == 4 &&
              manager.access_epoch() == 5,
          "exact tail fill is wrong");
  manager.append_decode_token({1});
  require(manager.block_table({1}) == std::vector<std::size_t>({0, 1, 2}) &&
              manager.physical_blocks()[2].valid_token_count == 1 &&
              manager.physical_blocks()[2].last_access_epoch == 6 &&
              manager.represented_token_count() == 9 &&
              manager.peak_allocated_block_count() == 3 &&
              manager.peak_utilization() == 0.75 &&
              manager.validate_invariants(),
          "boundary append is wrong");
  manager.release_request({1});
  require(manager.allocated_block_count() == 0 &&
              manager.free_block_count() == 4 &&
              manager.internal_fragmentation_tokens() == 0 &&
              manager.utilization() == 0.0 &&
              manager.peak_allocated_block_count() == 3 &&
              manager.physical_blocks()[0].last_access_epoch == 0 &&
              manager.physical_blocks()[1].last_access_epoch == 0 &&
              manager.physical_blocks()[2].last_access_epoch == 0 &&
              manager.validate_invariants(),
          "release reset or peak preservation is wrong");
}

void test_failure_classification_and_strong_guarantees() {
  KVCacheManager manager(KVCacheConfig(2, 2));
  manager.allocate_prompt({1}, 4);

  ManagerSnapshot before = snapshot(manager);
  require_throws<std::invalid_argument>([&] { manager.allocate_prompt({1}, 0); },
                                        "duplicate prompt succeeded");
  require(same_snapshot(snapshot(manager), before),
          "duplicate prompt changed manager state or failure count");

  before = snapshot(manager);
  require_throws<std::out_of_range>([&] { manager.append_decode_token({99}); },
                                    "unknown append succeeded");
  require(same_snapshot(snapshot(manager), before),
          "unknown append changed manager state or failure count");

  before = snapshot(manager);
  require_throws<std::runtime_error>([&] { manager.allocate_prompt({2}, 1); },
                                     "over-capacity prompt succeeded");
  require_only_failure_incremented(before, snapshot(manager),
                                   "prompt capacity failure was not strong");

  before = snapshot(manager);
  require_throws<std::runtime_error>([&] { manager.append_decode_token({1}); },
                                     "full boundary append succeeded");
  require_only_failure_incremented(before, snapshot(manager),
                                   "decode capacity failure was not strong");

  KVCacheManager counter(KVCacheConfig(1, 1));
  counter.allocate_prompt({1}, 1);
  KVCacheManagerTestAccess::set_failure_count(
      counter, std::numeric_limits<std::uint64_t>::max());
  before = snapshot(counter);
  require_throws<std::overflow_error>([&] { counter.allocate_prompt({2}, 1); },
                                      "failure-counter overflow was missed");
  require(same_snapshot(snapshot(counter), before),
          "failure-counter overflow changed manager state");

  KVCacheManager epoch(KVCacheConfig(2, 2));
  epoch.allocate_prompt({1}, 1);
  KVCacheManagerTestAccess::set_access_epoch(
      epoch, std::numeric_limits<std::uint64_t>::max());
  before = snapshot(epoch);
  require_throws<std::overflow_error>([&] { epoch.append_decode_token({1}); },
                                      "epoch overflow was missed");
  require(same_snapshot(snapshot(epoch), before),
          "epoch overflow changed state or capacity-failure count");

  KVCacheManager represented(
      KVCacheConfig(1, std::numeric_limits<std::uint64_t>::max()));
  represented.allocate_prompt({1}, std::numeric_limits<std::uint64_t>::max());
  before = snapshot(represented);
  require_throws<std::overflow_error>(
      [&] { represented.append_decode_token({1}); },
      "represented-token overflow was missed");
  require(same_snapshot(snapshot(represented), before),
          "represented-token overflow changed state or failure count");

  KVCacheManager release(KVCacheConfig(2, 2));
  release.allocate_prompt({1}, 2);
  before = snapshot(release);
  require_throws<std::out_of_range>([&] { release.release_request({99}); },
                                    "unknown release succeeded");
  require(same_snapshot(snapshot(release), before),
          "unknown release changed manager state");
  release.release_request({1});
  before = snapshot(release);
  require_throws<std::out_of_range>([&] { release.release_request({1}); },
                                    "repeated release succeeded");
  require(same_snapshot(snapshot(release), before),
          "repeated release changed manager state");
}

struct ShadowRequest {
  std::vector<std::size_t> blocks;
  std::uint64_t tokens{0};
};

struct ShadowModel {
  std::size_t total_blocks;
  std::uint64_t block_size;
  std::set<std::size_t> free_ids;
  std::map<RequestId, ShadowRequest> requests;
  std::vector<std::uint64_t> valid_tokens;
  std::vector<std::uint64_t> epochs;
  std::uint64_t represented{0};
  std::uint64_t failures{0};
  std::uint64_t epoch{0};
  std::size_t peak{0};
};

ShadowModel make_shadow(std::size_t blocks, std::uint64_t block_size) {
  ShadowModel model{blocks, block_size, {}, {},
                    std::vector<std::uint64_t>(blocks, 0),
                    std::vector<std::uint64_t>(blocks, 0)};
  for (std::size_t id = 0; id < blocks; ++id) {
    model.free_ids.insert(id);
  }
  return model;
}

std::size_t shadow_required(const ShadowModel& model, std::uint64_t tokens) {
  return tokens == 0 ? 0 : static_cast<std::size_t>(
      tokens / model.block_size + (tokens % model.block_size != 0 ? 1 : 0));
}

void compare_shadow(const KVCacheManager& manager, const ShadowModel& model,
                    const std::string& context) {
  std::map<RequestId, std::vector<std::size_t>> expected_tables;
  for (const auto& entry : model.requests) {
    expected_tables.emplace(entry.first, entry.second.blocks);
  }
  require(manager.free_block_ids() == model.free_ids &&
              manager.block_tables() == expected_tables &&
              manager.represented_token_count() == model.represented &&
              manager.free_block_count() == model.free_ids.size() &&
              manager.allocated_block_count() ==
                  model.total_blocks - model.free_ids.size() &&
              manager.internal_fragmentation_tokens() ==
                  static_cast<std::uint64_t>(model.total_blocks -
                                             model.free_ids.size()) *
                          model.block_size -
                      model.represented &&
              manager.utilization() ==
                  static_cast<double>(model.total_blocks -
                                      model.free_ids.size()) /
                      static_cast<double>(model.total_blocks) &&
              manager.allocation_failure_count() == model.failures &&
              manager.access_epoch() == model.epoch &&
              manager.peak_allocated_block_count() == model.peak &&
              manager.peak_utilization() ==
                  static_cast<double>(model.peak) /
                      static_cast<double>(model.total_blocks),
          context + ": aggregate shadow state differs");
  const auto blocks = manager.physical_blocks();
  for (std::size_t id = 0; id < model.total_blocks; ++id) {
    const bool free = model.free_ids.count(id) != 0;
    require(blocks[id].block_id == id &&
                blocks[id].state == (free ? PhysicalKVBlockState::Free
                                          : PhysicalKVBlockState::InUse) &&
                blocks[id].reference_count == (free ? 0U : 1U) &&
                blocks[id].valid_token_count == model.valid_tokens[id] &&
                blocks[id].last_access_epoch == model.epochs[id],
            context + ": physical block differs from shadow");
    if (free) {
      require(!blocks[id].owner.has_value(),
              context + ": free shadow block has owner");
    } else {
      bool owner_matches = false;
      for (const auto& request : model.requests) {
        owner_matches = owner_matches ||
            (std::find(request.second.blocks.begin(), request.second.blocks.end(),
                       id) != request.second.blocks.end() &&
             blocks[id].owner == request.first);
      }
      require(owner_matches, context + ": owned shadow block owner differs");
    }
  }
  require(manager.validate_invariants(), context + ": invariants failed");
}

void shadow_allocate(KVCacheManager& manager, ShadowModel& model, RequestId id,
                     std::uint64_t tokens) {
  if (model.requests.count(id) != 0) {
    require_throws<std::invalid_argument>([&] { manager.allocate_prompt(id, tokens); },
                                          "shadow duplicate allocation succeeded");
    return;
  }
  const std::size_t required = shadow_required(model, tokens);
  if (required > model.free_ids.size()) {
    require_throws<std::runtime_error>([&] { manager.allocate_prompt(id, tokens); },
                                       "shadow capacity allocation succeeded");
    ++model.failures;
    return;
  }
  manager.allocate_prompt(id, tokens);
  ShadowRequest request;
  request.tokens = tokens;
  if (required != 0) {
    ++model.epoch;
  }
  for (std::size_t index = 0; index < required; ++index) {
    const std::size_t block_id = *model.free_ids.begin();
    model.free_ids.erase(model.free_ids.begin());
    request.blocks.push_back(block_id);
    const std::uint64_t consumed =
        static_cast<std::uint64_t>(index) * model.block_size;
    model.valid_tokens[block_id] =
        std::min(model.block_size, tokens - consumed);
    model.epochs[block_id] = model.epoch;
  }
  model.represented += tokens;
  model.requests.emplace(id, std::move(request));
  model.peak = std::max(model.peak, model.total_blocks - model.free_ids.size());
}

void shadow_append(KVCacheManager& manager, ShadowModel& model, RequestId id) {
  auto found = model.requests.find(id);
  if (found == model.requests.end()) {
    require_throws<std::out_of_range>([&] { manager.append_decode_token(id); },
                                      "shadow unknown append succeeded");
    return;
  }
  const bool needs_block = found->second.blocks.empty() ||
      model.valid_tokens[found->second.blocks.back()] == model.block_size;
  if (needs_block && model.free_ids.empty()) {
    require_throws<std::runtime_error>([&] { manager.append_decode_token(id); },
                                       "shadow boundary failure succeeded");
    ++model.failures;
    return;
  }
  manager.append_decode_token(id);
  ++model.epoch;
  if (needs_block) {
    const std::size_t block_id = *model.free_ids.begin();
    model.free_ids.erase(model.free_ids.begin());
    found->second.blocks.push_back(block_id);
    model.valid_tokens[block_id] = 1;
    model.epochs[block_id] = model.epoch;
  } else {
    const std::size_t block_id = found->second.blocks.back();
    ++model.valid_tokens[block_id];
    model.epochs[block_id] = model.epoch;
  }
  ++found->second.tokens;
  ++model.represented;
  model.peak = std::max(model.peak, model.total_blocks - model.free_ids.size());
}

void shadow_release(KVCacheManager& manager, ShadowModel& model, RequestId id) {
  auto found = model.requests.find(id);
  if (found == model.requests.end()) {
    require_throws<std::out_of_range>([&] { manager.release_request(id); },
                                      "shadow unknown release succeeded");
    return;
  }
  manager.release_request(id);
  for (std::size_t block_id : found->second.blocks) {
    model.free_ids.insert(block_id);
    model.valid_tokens[block_id] = 0;
    model.epochs[block_id] = 0;
  }
  model.represented -= found->second.tokens;
  model.requests.erase(found);
}

void test_fixed_seed_shadow_model() {
  KVCacheManager manager(KVCacheConfig(7, 3));
  ShadowModel model = make_shadow(7, 3);
  const auto check = [&](const std::string& context) {
    compare_shadow(manager, model, context);
  };

  shadow_allocate(manager, model, {1}, 0); check("zero registration");
  shadow_append(manager, model, {1}); check("append from empty");
  shadow_append(manager, model, {1}); check("partial append");
  shadow_append(manager, model, {1}); check("exact fill");
  shadow_append(manager, model, {1}); check("boundary allocation");
  shadow_allocate(manager, model, {2}, 15); check("fill capacity");
  shadow_append(manager, model, {1}); check("full partial tail");
  shadow_append(manager, model, {1}); check("full exact tail");
  shadow_append(manager, model, {1}); check("full boundary failure");
  shadow_allocate(manager, model, {2}, 0); check("duplicate registration");
  shadow_append(manager, model, {99}); check("unknown append");
  shadow_release(manager, model, {99}); check("unknown release");
  shadow_release(manager, model, {2}); check("release creates holes");
  shadow_allocate(manager, model, {3}, 6); check("lowest-ID hole reuse");

  std::uint32_t random = 0x5a17c9e3U;  // Fixed LCG seed.
  for (int step = 0; step < 500; ++step) {
    random = random * 1664525U + 1013904223U;
    const RequestId id{1U + ((random >> 8U) % 12U)};
    switch (random % 3U) {
      case 0:
        shadow_allocate(manager, model, id, (random >> 16U) % 8U);
        break;
      case 1:
        shadow_append(manager, model, id);
        break;
      default:
        shadow_release(manager, model, id);
        break;
    }
    check("fixed-seed step " + std::to_string(step));
  }
}

}  // namespace

int main() {
  try {
    test_configuration_and_required_blocks();
    test_prompt_decode_release_metrics_and_epoch();
    test_failure_classification_and_strong_guarantees();
    test_fixed_seed_shadow_model();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "KV cache tests passed\n";
  return 0;
}
