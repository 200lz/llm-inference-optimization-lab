#include "serving/kv_cache.h"

#include <iostream>
#include <vector>

using namespace llm_lab::serving;

namespace {
void print_ids(const std::vector<std::size_t>& ids) {
  std::cout << '[';
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << ids[i];
  }
  std::cout << ']';
}
}

int main() {
  std::cout << "SIMULATED PREFIX CACHE\n";
  KVCacheManager cache(KVCacheConfig(4, 4),
      PrefixCacheConfig(true, "smoke-salt", "educational-model/revision/tokenizer"));
  const std::vector<std::int32_t> system_and_tail{10,11,12,13,20,21};

  auto a_lookup = cache.find_longest_cached_prefix(system_and_tail);
  cache.allocate_prompt_with_prefix({1}, system_and_tail, a_lookup);
  cache.insert_completed_prompt_blocks({1});
  std::cout << "request ID=1 original prompt tokens=6 matched tokens=0 scheduled prefill tokens=6 shared block IDs=[] private block IDs=";
  print_ids(cache.block_table({1})); std::cout << '\n';
  cache.release_request({1});
  std::cout << "request ID=1 cache state=Cached reference count="
            << cache.physical_blocks()[0].reference_count << '\n';

  auto b_lookup = cache.find_longest_cached_prefix(system_and_tail);
  cache.allocate_prompt_with_prefix({2}, system_and_tail, b_lookup);
  auto c_lookup = cache.find_longest_cached_prefix({10,11,12,13});
  cache.allocate_prompt_with_prefix({3}, {10,11,12,13}, c_lookup);
  std::cout << "request ID=2 original prompt tokens=6 matched tokens=4 scheduled prefill tokens=2 shared block IDs=";
  print_ids(b_lookup.physical_block_ids); std::cout << " private block IDs=[1]\n";
  std::cout << "request ID=3 matched tokens=4 shared block IDs=";
  print_ids(c_lookup.physical_block_ids); std::cout << " reference counts="
            << cache.physical_blocks()[0].reference_count << '\n';
  cache.release_request({2});
  cache.release_request({3});

  cache.allocate_prompt({4}, 16);
  std::cout << "cache hits=" << cache.prefix_cache_metrics().cache_hit_lookup_count
            << " cache misses=" << cache.prefix_cache_metrics().cache_miss_lookup_count
            << " evicted IDs=[0] saved simulated prefill work="
            << cache.prefix_cache_metrics().saved_prefill_token_count << '\n';
  std::cout << "final physical utilization=" << cache.utilization()
            << " final cached block count=" << cache.cached_block_count() << '\n';
  return 0;
}
