#include "serving/continuous_batching.h"

#include "serving/checked_math.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace llm_lab::serving {
namespace {

std::uint64_t checked_incremented(std::uint64_t value, const char* message) {
  if (value == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error(message);
  }
  return value + 1;
}

std::uint64_t checked_add_unsigned(std::uint64_t lhs, std::uint64_t rhs,
                                   const char* message) {
  if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    throw std::overflow_error(message);
  }
  return lhs + rhs;
}

bool valid_deferral_reason(DeferralReason reason) noexcept {
  return reason == DeferralReason::SequenceBudget ||
         reason == DeferralReason::TokenBudget ||
         reason == DeferralReason::KVCapacity;
}

}  // namespace

std::string_view to_string(SchedulingPolicy policy) noexcept {
  switch (policy) {
    case SchedulingPolicy::DecodeFirst:
      return "DecodeFirst";
    case SchedulingPolicy::FcfsMixed:
      return "FcfsMixed";
  }
  return "Unknown";
}

std::string_view to_string(DeferralReason reason) noexcept {
  switch (reason) {
    case DeferralReason::SequenceBudget:
      return "sequence budget";
    case DeferralReason::TokenBudget:
      return "token budget";
    case DeferralReason::KVCapacity:
      return "KV capacity";
  }
  return "unknown";
}

ContinuousBatchingConfig::ContinuousBatchingConfig(
    std::size_t max_num_sequences, std::uint64_t max_batched_tokens,
    SchedulingPolicy scheduling_policy, KVCacheConfig kv_cache_config,
    PrefixCacheConfig prefix_cache_config)
    : max_num_sequences_(max_num_sequences),
      max_batched_tokens_(max_batched_tokens),
      scheduling_policy_(scheduling_policy),
      kv_cache_config_(kv_cache_config),
      prefix_cache_config_(std::move(prefix_cache_config)) {
  if (max_num_sequences_ == 0) {
    throw std::invalid_argument("max_num_sequences must be at least 1");
  }
  if (max_batched_tokens_ == 0) {
    throw std::invalid_argument("max_batched_tokens must be at least 1");
  }
  if (scheduling_policy_ != SchedulingPolicy::DecodeFirst &&
      scheduling_policy_ != SchedulingPolicy::FcfsMixed) {
    throw std::invalid_argument("unsupported continuous batching policy");
  }
}

BatchPlan::BatchPlan(std::uint64_t iteration_number,
                     std::vector<RequestId> prefill_request_ids,
                     std::vector<RequestId> decode_request_ids,
                     std::vector<DeferredRequest> deferred_requests,
                     std::uint64_t total_prefill_tokens,
                     std::uint64_t total_decode_tokens,
                     std::uint64_t total_scheduled_tokens)
    : iteration_number_(iteration_number),
      prefill_request_ids_(std::move(prefill_request_ids)),
      decode_request_ids_(std::move(decode_request_ids)),
      deferred_requests_(std::move(deferred_requests)),
      total_prefill_tokens_(total_prefill_tokens),
      total_decode_tokens_(total_decode_tokens),
      total_scheduled_tokens_(total_scheduled_tokens) {}

BatchPlan BatchPlan::create(
    std::uint64_t iteration_number, std::vector<PrefillWork> prefill_work,
    std::vector<RequestId> decode_request_ids,
    std::size_t max_num_sequences, std::uint64_t max_batched_tokens) {
  return create_with_deferred(iteration_number, std::move(prefill_work),
                              std::move(decode_request_ids), {},
                              max_num_sequences, max_batched_tokens);
}

BatchPlan BatchPlan::create_with_deferred(
    std::uint64_t iteration_number, std::vector<PrefillWork> prefill_work,
    std::vector<RequestId> decode_request_ids,
    std::vector<DeferredRequest> deferred_requests,
    std::size_t max_num_sequences, std::uint64_t max_batched_tokens) {
  if (max_num_sequences == 0 || max_batched_tokens == 0) {
    throw std::invalid_argument("BatchPlan budgets must be positive");
  }
  if (prefill_work.empty() && decode_request_ids.empty() &&
      deferred_requests.empty()) {
    throw std::invalid_argument("use BatchPlan::empty for a zero-work plan");
  }
  if (prefill_work.size() > max_num_sequences ||
      decode_request_ids.size() > max_num_sequences - prefill_work.size()) {
    throw std::invalid_argument("BatchPlan exceeds max_num_sequences");
  }

  std::set<RequestId> seen;
  std::vector<RequestId> prefill_request_ids;
  prefill_request_ids.reserve(prefill_work.size());
  std::uint64_t total_prefill_tokens = 0;
  for (const PrefillWork& work : prefill_work) {
    if (!seen.insert(work.request_id).second) {
      throw std::invalid_argument("request appears more than once in BatchPlan");
    }
    total_prefill_tokens = checked_add_unsigned(
        total_prefill_tokens, work.prompt_token_count,
        "BatchPlan prefill token total overflow");
    prefill_request_ids.push_back(work.request_id);
  }
  for (RequestId request_id : decode_request_ids) {
    if (!seen.insert(request_id).second) {
      throw std::invalid_argument(
          "request appears in both or more than once in BatchPlan");
    }
  }
  for (const DeferredRequest& deferred : deferred_requests) {
    if (!valid_deferral_reason(deferred.reason)) {
      throw std::invalid_argument("BatchPlan has an invalid deferral reason");
    }
    if (!seen.insert(deferred.request_id).second) {
      throw std::invalid_argument(
          "deferred request also appears elsewhere in BatchPlan");
    }
  }
  if (prefill_work.empty() && decode_request_ids.empty()) {
    for (const DeferredRequest& deferred : deferred_requests) {
      if (deferred.reason != DeferralReason::KVCapacity) {
        throw std::invalid_argument(
            "deferred-only BatchPlan requires KVCapacity reasons");
      }
    }
  }

  const auto total_decode_tokens =
      static_cast<std::uint64_t>(decode_request_ids.size());
  const auto total_scheduled_tokens = checked_add_unsigned(
      total_prefill_tokens, total_decode_tokens,
      "BatchPlan scheduled token total overflow");
  if (total_scheduled_tokens > max_batched_tokens) {
    throw std::invalid_argument("BatchPlan exceeds max_batched_tokens");
  }
  BatchPlan result(iteration_number, std::move(prefill_request_ids),
                   std::move(decode_request_ids),
                   std::move(deferred_requests), total_prefill_tokens,
                   total_decode_tokens, total_scheduled_tokens);
  result.prefill_work_ = std::move(prefill_work);
  return result;
}

BatchPlan BatchPlan::empty(std::uint64_t iteration_number) {
  return BatchPlan(iteration_number, {}, {}, {}, 0, 0, 0);
}

std::optional<double> ContinuousBatchingStatistics::average_batch_size()
    const noexcept {
  if (nonempty_batch_count == 0) {
    return std::nullopt;
  }
  return static_cast<double>(total_scheduled_sequences) /
         static_cast<double>(nonempty_batch_count);
}

std::optional<double> ContinuousBatchingStatistics::prefix_token_hit_rate()
    const noexcept {
  if (total_cache_eligible_prompt_tokens_looked_up == 0) return std::nullopt;
  return static_cast<double>(total_matched_prefix_tokens) /
         static_cast<double>(total_cache_eligible_prompt_tokens_looked_up);
}

ContinuousBatchingEngine::ContinuousBatchingEngine(
    const SimulatedBackend& backend, ContinuousBatchingConfig config)
    : backend_(backend), config_(config),
      kv_cache_(config.kv_cache_config(), config.prefix_cache_config()) {
  backend_.validate_configuration();
}

void ContinuousBatchingEngine::submit_request(Request request) {
  if (failed_) {
    throw std::logic_error("cannot submit to a failed continuous engine");
  }
  if (request.state != RequestState::Waiting ||
      request.generated_token_count != 0 ||
      request.admitted_time_us.has_value() ||
      request.first_scheduled_time_us.has_value() ||
      request.first_token_time_us.has_value() ||
      request.finish_time_us.has_value()) {
    throw std::invalid_argument("submitted request must be pristine");
  }
  if (request.arrival_time_us < clock_.now_us()) {
    throw std::invalid_argument(
        "request arrival precedes current simulation time");
  }
  if ((!config_.prefix_cache_config().enabled() ||
       !request.has_exact_prompt_tokens()) &&
      request.prompt_length() > config_.max_batched_tokens()) {
    throw std::invalid_argument(
        "prompt_token_count exceeds max_batched_tokens; chunked prefill is not implemented");
  }
  if (!requests_.emplace(request.request_id, std::move(request)).second) {
    throw std::invalid_argument("duplicate request ID");
  }
}

void ContinuousBatchingEngine::cancel_request(RequestId request_id) {
  if (failed_) {
    throw std::logic_error("cannot cancel in a failed continuous engine");
  }
  const auto found = requests_.find(request_id);
  if (found == requests_.end()) {
    throw std::out_of_range("unknown request ID");
  }
  Request& request = found->second;
  if (request.state == RequestState::Finished ||
      request.state == RequestState::Cancelled) {
    throw std::logic_error("terminal request cannot be cancelled");
  }
  if (request.state == RequestState::Preempted) {
    throw std::logic_error("preempted requests are not implemented");
  }

  ContinuousBatchingStatistics prepared_statistics = statistics_;
  prepared_statistics.cancelled_request_count = checked_incremented(
      statistics_.cancelled_request_count, "cancelled request count overflow");
  Request prepared_request = request;
  prepared_request.transition_to(RequestState::Cancelled);
  prepared_request.finish_time_us = clock_.now_us();
  std::set<RequestId> prepared_arrivals = arrived_requests_;
  prepared_arrivals.erase(request_id);
  KVCacheManager prepared_kv_cache = kv_cache_;
  if (request.state == RequestState::Decoding) {
    prepared_kv_cache.release_request(request_id);
  }
  prepared_statistics.current_allocated_kv_blocks =
      prepared_kv_cache.allocated_block_count();
  prepared_statistics.current_kv_block_utilization =
      prepared_kv_cache.utilization();
  prepared_statistics.represented_kv_tokens =
      prepared_kv_cache.represented_token_count();
  prepared_statistics.internal_fragmentation_tokens =
      prepared_kv_cache.internal_fragmentation_tokens();
  prepared_statistics.kv_allocation_failure_count =
      prepared_kv_cache.allocation_failure_count();
  prepared_statistics.cached_kv_blocks = prepared_kv_cache.cached_block_count();
  prepared_statistics.referenced_shared_kv_blocks =
      prepared_kv_cache.referenced_shared_block_count();
  const PrefixCacheMetrics& cancelled_prefix = prepared_kv_cache.prefix_cache_metrics();
  prepared_statistics.saved_simulated_prefill_tokens = cancelled_prefix.saved_prefill_token_count;
  prepared_statistics.cache_lookup_count = cancelled_prefix.cache_lookup_count;
  prepared_statistics.cache_hit_lookup_count = cancelled_prefix.cache_hit_lookup_count;
  prepared_statistics.cache_miss_lookup_count = cancelled_prefix.cache_miss_lookup_count;
  prepared_statistics.cache_matched_block_count = cancelled_prefix.matched_block_count;
  prepared_statistics.total_cache_eligible_prompt_tokens_looked_up =
      cancelled_prefix.total_cache_eligible_prompt_tokens_looked_up;
  prepared_statistics.collision_verification_count = cancelled_prefix.collision_verification_count;
  prepared_statistics.prefix_cache_eviction_count = cancelled_prefix.eviction_count;

  static_assert(std::is_nothrow_move_assignable_v<Request>);
  request = std::move(prepared_request);
  arrived_requests_.swap(prepared_arrivals);
  kv_cache_.swap(prepared_kv_cache);
  statistics_ = prepared_statistics;
}

std::set<RequestId>
ContinuousBatchingEngine::arrivals_at_current_timestamp() const {
  std::set<RequestId> prepared_arrivals = arrived_requests_;
  std::vector<const Request*> arrivals;
  for (const auto& entry : requests_) {
    const Request& request = entry.second;
    if (request.state == RequestState::Waiting &&
        prepared_arrivals.find(request.request_id) == prepared_arrivals.end() &&
        request.arrival_time_us <= clock_.now_us()) {
      arrivals.push_back(&request);
    }
  }
  std::sort(arrivals.begin(), arrivals.end(), [](const Request* lhs,
                                                  const Request* rhs) {
    if (lhs->arrival_time_us != rhs->arrival_time_us) {
      return lhs->arrival_time_us < rhs->arrival_time_us;
    }
    return lhs->request_id < rhs->request_id;
  });
  for (const Request* request : arrivals) {
    prepared_arrivals.insert(request->request_id);
  }
  return prepared_arrivals;
}

bool ContinuousBatchingEngine::has_runnable_work(
    const std::set<RequestId>& arrived_requests) const {
  for (const auto& entry : requests_) {
    const Request& request = entry.second;
    if (request.state == RequestState::Decoding ||
        (request.state == RequestState::Waiting &&
         arrived_requests.find(request.request_id) != arrived_requests.end())) {
      return true;
    }
  }
  return false;
}

std::optional<std::int64_t> ContinuousBatchingEngine::next_future_arrival(
    const std::set<RequestId>& arrived_requests) const {
  std::optional<std::int64_t> result;
  for (const auto& entry : requests_) {
    const Request& request = entry.second;
    if (request.state != RequestState::Waiting ||
        arrived_requests.find(request.request_id) != arrived_requests.end() ||
        request.arrival_time_us <= clock_.now_us()) {
      continue;
    }
    if (!result.has_value() || request.arrival_time_us < *result) {
      result = request.arrival_time_us;
    }
  }
  return result;
}

BatchPlan ContinuousBatchingEngine::build_from_candidates(
    std::uint64_t iteration_number,
    const std::vector<Candidate>& candidates) const {
  std::vector<PrefillWork> prefills;
  std::vector<DecodeWork> decode_work;
  std::vector<RequestId> decodes;
  std::vector<PlannedWorkItem> work_order;
  std::vector<DeferredRequest> deferred;
  std::vector<std::size_t> all_evictions;
  std::vector<std::size_t> all_allocations;
  std::uint64_t used_tokens = 0;
  KVCacheManager local = kv_cache_;
  for (const Candidate& candidate : candidates) {
    const Request& request = requests_.at(candidate.request_id);
    PrefixLookupResult lookup;
    std::uint64_t token_cost = 1;
    std::size_t blocks_required = 0;
    if (candidate.is_prefill) {
      if (config_.prefix_cache_config().enabled() &&
          request.has_exact_prompt_tokens()) {
        lookup = local.find_longest_cached_prefix(request.prompt_tokens());
      } else {
        lookup.kind = PrefixLookupKind::Disabled;
      }
      token_cost = request.prompt_length() - lookup.matched_token_count;
      if (request.prompt_length() > config_.max_batched_tokens()) {
        if (!config_.prefix_cache_config().enabled()) {
          throw std::invalid_argument(
              "prompt_token_count exceeds max_batched_tokens; chunked prefill is not implemented");
        }
        if (lookup.matched_token_count == 0 ||
            token_cost > config_.max_batched_tokens()) {
          throw std::invalid_argument("oversized unmatched prefill");
        }
      }
      blocks_required = local.blocks_required(token_cost);
    } else {
      blocks_required = local.decode_requires_new_block(candidate.request_id)
                            ? 1U
                            : 0U;
    }

    if (token_cost > config_.max_batched_tokens() - used_tokens) {
      deferred.push_back(
          DeferredRequest{candidate.request_id, DeferralReason::TokenBudget});
      continue;
    }
    if (prefills.size() + decodes.size() >= config_.max_num_sequences()) {
      deferred.push_back(
          DeferredRequest{candidate.request_id, DeferralReason::SequenceBudget});
      continue;
    }

    std::vector<std::size_t> evictable = local.eligible_eviction_order();
    if (candidate.is_prefill && !lookup.physical_block_ids.empty()) {
      evictable.erase(std::remove_if(evictable.begin(), evictable.end(),
          [&](std::size_t id) {
            return std::find(lookup.physical_block_ids.begin(),
                             lookup.physical_block_ids.end(), id) !=
                   lookup.physical_block_ids.end();
          }), evictable.end());
    }
    if (blocks_required > local.free_block_count() + evictable.size()) {
      deferred.push_back(
          DeferredRequest{candidate.request_id, DeferralReason::KVCapacity});
      continue;
    }

    const std::size_t eviction_count = blocks_required > local.free_block_count()
        ? blocks_required - local.free_block_count() : 0;
    std::vector<std::size_t> evicted;
    evicted.assign(evictable.begin(),
                   evictable.begin() + static_cast<std::ptrdiff_t>(eviction_count));
    std::set<std::size_t> available = local.free_block_ids();
    available.insert(evicted.begin(), evicted.end());
    std::vector<std::size_t> allocated;
    auto available_it = available.begin();
    for (std::size_t i = 0; i < blocks_required; ++i, ++available_it)
      allocated.push_back(*available_it);

    used_tokens += token_cost;
    if (candidate.is_prefill) {
      if (config_.prefix_cache_config().enabled() &&
          request.has_exact_prompt_tokens()) {
        local.allocate_prompt_with_prefix_exact(candidate.request_id,
            request.prompt_tokens(), lookup, evicted, allocated);
      } else {
        local.allocate_prompt_exact(candidate.request_id,
            request.prompt_length(), evicted, allocated);
      }
      prefills.push_back(PrefillWork{candidate.request_id, token_cost,
          request.prompt_length(), lookup.matched_token_count,
          lookup.physical_block_ids, allocated, evicted, lookup.kind});
    } else {
      const std::optional<std::size_t> allocation =
          allocated.empty() ? std::nullopt
                            : std::optional<std::size_t>(allocated.front());
      local.append_decode_token_exact(candidate.request_id, evicted, allocation);
      decodes.push_back(candidate.request_id);
      decode_work.push_back(DecodeWork{candidate.request_id, allocation, evicted});
    }
    work_order.push_back(PlannedWorkItem{candidate.request_id,
                                         candidate.is_prefill});
    all_evictions.insert(all_evictions.end(), evicted.begin(), evicted.end());
    all_allocations.insert(all_allocations.end(), allocated.begin(), allocated.end());
  }
  if (prefills.empty() && decodes.empty() && !deferred.empty()) {
    for (const DeferredRequest& item : deferred) {
      if (item.reason != DeferralReason::KVCapacity)
        throw std::logic_error(
            "token-only or sequence-only no-progress plan is invalid");
    }
    return BatchPlan::create_with_deferred(
        iteration_number, {}, {}, std::move(deferred),
        config_.max_num_sequences(), config_.max_batched_tokens());
  }
  const std::set<std::size_t> unique_evictions(all_evictions.begin(),
                                                all_evictions.end());
  const std::set<std::size_t> unique_allocations(all_allocations.begin(),
                                                  all_allocations.end());
  if (unique_evictions.size() != all_evictions.size())
    throw std::logic_error("physical block is evicted twice in one plan");
  if (unique_allocations.size() != all_allocations.size())
    throw std::logic_error("physical block is allocated twice in one plan");
  std::set<std::size_t> matched_ids;
  for (const PrefillWork& work : prefills)
    matched_ids.insert(work.matched_physical_block_ids.begin(),
                       work.matched_physical_block_ids.end());
  for (std::size_t id : matched_ids) {
    if (unique_evictions.count(id) != 0 || unique_allocations.count(id) != 0)
      throw std::logic_error("matched physical block has a conflicting plan role");
  }
  BatchPlan plan = BatchPlan::create_with_deferred(
      iteration_number, std::move(prefills), std::move(decodes),
      std::move(deferred), config_.max_num_sequences(),
      config_.max_batched_tokens());
  plan.decode_work_ = std::move(decode_work);
  plan.work_order_ = std::move(work_order);
  plan.planned_eviction_ids_ = std::move(all_evictions);
  plan.planned_allocation_ids_ = std::move(all_allocations);
  return plan;
}

bool ContinuousBatchingEngine::candidate_less(const Candidate& lhs,
                                               const Candidate& rhs) noexcept {
  if (lhs.arrival_time_us != rhs.arrival_time_us) {
    return lhs.arrival_time_us < rhs.arrival_time_us;
  }
  if (lhs.request_id != rhs.request_id) {
    return lhs.request_id < rhs.request_id;
  }
  // A request cannot normally have both kinds at once. Prefill wins if a
  // malformed snapshot ever creates that ambiguity.
  return lhs.is_prefill && !rhs.is_prefill;
}

BatchPlan ContinuousBatchingEngine::build_decode_first(
    std::uint64_t iteration_number,
    const std::set<RequestId>& arrived_requests) const {
  std::vector<Candidate> decodes;
  std::vector<Candidate> prefills;
  for (const auto& entry : requests_) {
    const Request& request = entry.second;
    if (request.state == RequestState::Decoding) {
      decodes.push_back(
          Candidate{request.request_id, request.arrival_time_us, false});
    } else if (request.state == RequestState::Waiting &&
               arrived_requests.find(request.request_id) !=
                   arrived_requests.end()) {
      prefills.push_back(
          Candidate{request.request_id, request.arrival_time_us, true});
    }
  }
  std::sort(decodes.begin(), decodes.end(), candidate_less);
  std::sort(prefills.begin(), prefills.end(), candidate_less);
  decodes.insert(decodes.end(), prefills.begin(), prefills.end());
  return build_from_candidates(iteration_number, decodes);
}

BatchPlan ContinuousBatchingEngine::build_fcfs_mixed(
    std::uint64_t iteration_number,
    const std::set<RequestId>& arrived_requests) const {
  std::vector<Candidate> candidates;
  for (const auto& entry : requests_) {
    const Request& request = entry.second;
    if (request.state == RequestState::Decoding) {
      candidates.push_back(
          Candidate{request.request_id, request.arrival_time_us, false});
    } else if (request.state == RequestState::Waiting &&
               arrived_requests.find(request.request_id) !=
                   arrived_requests.end()) {
      candidates.push_back(
          Candidate{request.request_id, request.arrival_time_us, true});
    }
  }
  std::sort(candidates.begin(), candidates.end(), candidate_less);
  return build_from_candidates(iteration_number, candidates);
}

BatchPlan ContinuousBatchingEngine::build_plan(
    std::uint64_t iteration_number,
    const std::set<RequestId>& arrived_requests) const {
  switch (config_.scheduling_policy()) {
    case SchedulingPolicy::DecodeFirst:
      return build_decode_first(iteration_number, arrived_requests);
    case SchedulingPolicy::FcfsMixed:
      return build_fcfs_mixed(iteration_number, arrived_requests);
  }
  throw std::logic_error("unsupported continuous batching policy");
}

std::vector<ContinuousBatchingEngine::PreparedRequestUpdate>
ContinuousBatchingEngine::prepare_request_updates(
    const BatchPlan& plan, std::int64_t end_timestamp_us,
    const std::set<RequestId>& arrived_requests,
    std::uint64_t& completed_request_count) {
  std::vector<PreparedRequestUpdate> updates;
  updates.reserve(plan.scheduled_sequence_count());
  completed_request_count = 0;
  for (RequestId request_id : plan.prefill_request_ids()) {
    Request& request = requests_.at(request_id);
    if (request.state != RequestState::Waiting ||
        arrived_requests.find(request_id) == arrived_requests.end()) {
      throw std::logic_error("BatchPlan prefill request is not runnable");
    }
    Request prepared = request;
    prepared.transition_to(RequestState::Prefilling);
    prepared.admitted_time_us = clock_.now_us();
    prepared.first_scheduled_time_us = clock_.now_us();
    if (prepared.max_new_tokens == 0) {
      prepared.transition_to(RequestState::Finished);
      prepared.finish_time_us = end_timestamp_us;
      completed_request_count = checked_incremented(
          completed_request_count, "iteration completion count overflow");
    } else {
      prepared.transition_to(RequestState::Decoding);
    }
    updates.push_back(PreparedRequestUpdate{&request, std::move(prepared)});
  }
  for (RequestId request_id : plan.decode_request_ids()) {
    Request& request = requests_.at(request_id);
    if (request.state != RequestState::Decoding ||
        request.generated_token_count >= request.max_new_tokens) {
      throw std::logic_error("BatchPlan decode request is not runnable");
    }
    Request prepared = request;
    ++prepared.generated_token_count;
    if (!prepared.first_token_time_us.has_value()) {
      prepared.first_token_time_us = end_timestamp_us;
    }
    if (prepared.generated_token_count == prepared.max_new_tokens) {
      prepared.transition_to(RequestState::Finished);
      prepared.finish_time_us = end_timestamp_us;
      completed_request_count = checked_incremented(
          completed_request_count, "iteration completion count overflow");
    }
    updates.push_back(PreparedRequestUpdate{&request, std::move(prepared)});
  }
  return updates;
}

ContinuousBatchingStatistics ContinuousBatchingEngine::prepare_statistics(
    const BatchPlan& plan, std::uint64_t completed_request_count,
    const KVCacheManager& prepared_kv_cache) const {
  ContinuousBatchingStatistics prepared = statistics_;
  prepared.scheduling_iteration_count = checked_incremented(
      prepared.scheduling_iteration_count,
      "scheduling iteration count overflow");
  if (plan.deferred_only()) {
    prepared.stalled_iteration_count = checked_incremented(
        prepared.stalled_iteration_count, "stalled iteration count overflow");
  } else if (plan.empty()) {
    prepared.idle_iteration_count = checked_incremented(
        prepared.idle_iteration_count, "idle iteration count overflow");
  } else {
    prepared.nonempty_batch_count = checked_incremented(
        prepared.nonempty_batch_count, "nonempty batch count overflow");
  }
  prepared.total_prefill_tokens_scheduled = checked_add_unsigned(
      prepared.total_prefill_tokens_scheduled,
      plan.total_prefill_tokens(), "prefill scheduled-token total overflow");
  prepared.total_decode_tokens_scheduled = checked_add_unsigned(
      prepared.total_decode_tokens_scheduled,
      plan.total_decode_tokens(), "decode scheduled-token total overflow");
  const auto sequences =
      static_cast<std::uint64_t>(plan.scheduled_sequence_count());
  prepared.total_scheduled_sequences = checked_add_unsigned(
      prepared.total_scheduled_sequences, sequences,
      "scheduled sequence total overflow");
  prepared.deferred_request_count = checked_add_unsigned(
      prepared.deferred_request_count,
      static_cast<std::uint64_t>(plan.deferred_requests().size()),
      "deferred request count overflow");
  std::uint64_t kv_deferrals = 0;
  for (const DeferredRequest& deferred : plan.deferred_requests()) {
    if (deferred.reason == DeferralReason::KVCapacity) {
      kv_deferrals = checked_incremented(
          kv_deferrals, "iteration KV deferral count overflow");
    }
  }
  prepared.kv_capacity_deferral_count = checked_add_unsigned(
      prepared.kv_capacity_deferral_count, kv_deferrals,
      "KV-capacity deferral count overflow");
  prepared.completed_request_count = checked_add_unsigned(
      prepared.completed_request_count, completed_request_count,
      "completed request count overflow");
  prepared.maximum_batch_size =
      std::max(prepared.maximum_batch_size, sequences);
  prepared.maximum_scheduled_tokens = std::max(
      prepared.maximum_scheduled_tokens, plan.total_scheduled_tokens());
  prepared.current_allocated_kv_blocks =
      prepared_kv_cache.allocated_block_count();
  prepared.peak_allocated_kv_blocks =
      prepared_kv_cache.peak_allocated_block_count();
  prepared.current_kv_block_utilization = prepared_kv_cache.utilization();
  prepared.peak_kv_block_utilization = prepared_kv_cache.peak_utilization();
  prepared.represented_kv_tokens =
      prepared_kv_cache.represented_token_count();
  prepared.internal_fragmentation_tokens =
      prepared_kv_cache.internal_fragmentation_tokens();
  prepared.kv_allocation_failure_count =
      prepared_kv_cache.allocation_failure_count();
  for (const PrefillWork& work : plan.prefill_work()) {
    prepared.total_original_prompt_tokens = checked_add_unsigned(
        prepared.total_original_prompt_tokens, work.original_prompt_token_count,
        "original prompt token total overflow");
    prepared.total_matched_prefix_tokens = checked_add_unsigned(
        prepared.total_matched_prefix_tokens, work.matched_prefix_token_count,
        "matched prefix token total overflow");
  }
  const PrefixCacheMetrics& prefix = prepared_kv_cache.prefix_cache_metrics();
  prepared.saved_simulated_prefill_tokens = prefix.saved_prefill_token_count;
  prepared.cache_lookup_count = prefix.cache_lookup_count;
  prepared.cache_hit_lookup_count = prefix.cache_hit_lookup_count;
  prepared.cache_miss_lookup_count = prefix.cache_miss_lookup_count;
  prepared.cache_matched_block_count = prefix.matched_block_count;
  prepared.total_cache_eligible_prompt_tokens_looked_up =
      prefix.total_cache_eligible_prompt_tokens_looked_up;
  prepared.collision_verification_count = prefix.collision_verification_count;
  prepared.prefix_cache_eviction_count = prefix.eviction_count;
  prepared.cached_kv_blocks = prepared_kv_cache.cached_block_count();
  prepared.referenced_shared_kv_blocks = prepared_kv_cache.referenced_shared_block_count();
  return prepared;
}

ContinuousBatchingEngine::PreparedIteration
ContinuousBatchingEngine::prepare_iteration(
    BatchPlan plan, std::set<RequestId> arrived_requests,
    std::int64_t time_value_us, bool time_value_is_duration) {
  KVCacheManager prepared_kv_cache = kv_cache_;
  prepared_kv_cache.failure_point_for_test_ =
      prepared_kv_failure_for_test_;
  if (fail_reservation_match_for_test_ && !plan.work_order_.empty()) {
    const PlannedWorkItem& last = plan.work_order_.back();
    if (last.is_prefill) {
      auto found = std::find_if(plan.prefill_work_.begin(),
          plan.prefill_work_.end(), [&](const PrefillWork& work) {
            return work.request_id == last.request_id;
          });
      if (found == plan.prefill_work_.end() ||
          found->newly_allocated_block_ids.empty())
        throw std::logic_error("reservation mismatch seam requires a prefill allocation");
      found->newly_allocated_block_ids.back() =
          prepared_kv_cache.config().total_num_blocks();
    } else {
      auto found = std::find_if(plan.decode_work_.begin(),
          plan.decode_work_.end(), [&](const DecodeWork& work) {
            return work.request_id == last.request_id;
          });
      if (found == plan.decode_work_.end() ||
          !found->newly_allocated_block_id.has_value())
        throw std::logic_error("reservation mismatch seam requires a decode allocation");
      found->newly_allocated_block_id =
          prepared_kv_cache.config().total_num_blocks();
    }
  }
  for (const PlannedWorkItem& item : plan.work_order()) {
    if (item.is_prefill) {
      const auto found = std::find_if(plan.prefill_work().begin(),
          plan.prefill_work().end(), [&](const PrefillWork& work) {
            return work.request_id == item.request_id;
          });
      if (found == plan.prefill_work().end())
        throw std::logic_error("planned prefill work is missing");
      const PrefillWork& work = *found;
      const Request& request = requests_.at(work.request_id);
      if (config_.prefix_cache_config().enabled() &&
          request.has_exact_prompt_tokens()) {
        PrefixLookupResult lookup;
        lookup.matched_block_count = work.matched_physical_block_ids.size();
        lookup.matched_token_count = work.matched_prefix_token_count;
        lookup.physical_block_ids = work.matched_physical_block_ids;
        lookup.kind = work.prefix_lookup_kind;
        lookup.eligible_token_count = static_cast<std::uint64_t>(
            request.prompt_tokens().size() /
            prepared_kv_cache.config().block_size_tokens()) *
            prepared_kv_cache.config().block_size_tokens();
        prepared_kv_cache.allocate_prompt_with_prefix_exact(
            work.request_id, request.prompt_tokens(), lookup,
            work.evicted_block_ids, work.newly_allocated_block_ids);
      } else {
        prepared_kv_cache.allocate_prompt_exact(work.request_id,
            request.prompt_length(), work.evicted_block_ids,
            work.newly_allocated_block_ids);
      }
    } else {
      const auto found = std::find_if(plan.decode_work().begin(),
          plan.decode_work().end(), [&](const DecodeWork& work) {
            return work.request_id == item.request_id;
          });
      if (found == plan.decode_work().end())
        throw std::logic_error("planned decode work is missing");
      prepared_kv_cache.append_decode_token_exact(
          found->request_id, found->evicted_block_ids,
          found->newly_allocated_block_id);
    }
  }
  // Publish only after every request has acquired its iteration-start snapshot
  // and allocated its private suffix: no same-plan insertion visibility.
  for (const PrefillWork& work : plan.prefill_work()) {
    const Request& request = requests_.at(work.request_id);
    const std::size_t eligible_blocks = request.has_exact_prompt_tokens()
        ? request.prompt_tokens().size() /
              static_cast<std::size_t>(prepared_kv_cache.config().block_size_tokens())
        : 0;
    if (prepared_kv_cache.prefix_cache_config().enabled() &&
        request.has_exact_prompt_tokens() &&
        work.matched_physical_block_ids.size() < eligible_blocks)
      prepared_kv_cache.insert_completed_prompt_blocks(work.request_id);
  }
  const std::int64_t end_timestamp_us = time_value_is_duration
      ? checked_add(clock_.now_us(), time_value_us)
      : time_value_us;
  if (end_timestamp_us < clock_.now_us()) {
    throw std::logic_error("prepared iteration moves time backwards");
  }
  SimulationClock prepared_clock = clock_;
  prepared_clock.advance_to(end_timestamp_us);
  std::uint64_t completed_request_count = 0;
  std::vector<PreparedRequestUpdate> request_updates;
  if (!plan.empty()) {
    request_updates = prepare_request_updates(
        plan, end_timestamp_us, arrived_requests, completed_request_count);
  }
  // Same-plan completions cannot donate capacity. Lifecycle results are ready,
  // and release occurs only after every planned allocation/append succeeded.
  for (RequestId request_id : plan.prefill_request_ids()) {
    const Request& request = requests_.at(request_id);
    if (request.max_new_tokens == 0) {
      prepared_kv_cache.release_request(request_id);
    }
  }
  for (RequestId request_id : plan.decode_request_ids()) {
    const Request& request = requests_.at(request_id);
    if (request.generated_token_count + 1 == request.max_new_tokens) {
      prepared_kv_cache.release_request(request_id);
    }
  }
  ContinuousBatchingStatistics prepared_statistics =
      prepare_statistics(plan, completed_request_count, prepared_kv_cache);
  const std::uint64_t prepared_next_iteration = checked_incremented(
      next_iteration_number_, "iteration number overflow");
  BatchTraceEntry trace_entry{plan.iteration_number(), clock_.now_us(),
                              end_timestamp_us, config_.scheduling_policy(),
                              std::move(plan),
                              prepared_kv_cache.allocated_block_count(),
                              prepared_kv_cache.free_block_count(),
                              prepared_kv_cache.represented_token_count(),
                              prepared_kv_cache.internal_fragmentation_tokens(),
                              prepared_kv_cache.utilization(),
                              prepared_kv_cache.block_tables()};
  if (fail_trace_preparation_for_test_) {
    throw std::runtime_error("injected trace preparation failure");
  }
  if (plan_trace_.size() == plan_trace_.max_size()) {
    throw std::length_error("plan trace capacity overflow");
  }
  plan_trace_.reserve(plan_trace_.size() + 1);
  return PreparedIteration{std::move(arrived_requests),
                           std::move(request_updates), prepared_statistics,
                           std::move(trace_entry), prepared_next_iteration,
                           prepared_clock, std::move(prepared_kv_cache)};
}

void ContinuousBatchingEngine::commit_iteration(
    PreparedIteration prepared) noexcept {
  static_assert(std::is_nothrow_move_assignable_v<Request>);
  static_assert(std::is_nothrow_move_constructible_v<BatchTraceEntry>);
  static_assert(std::is_nothrow_copy_assignable_v<SimulationClock>);
  clock_ = prepared.clock;
  arrived_requests_.swap(prepared.arrived_requests);
  for (PreparedRequestUpdate& update : prepared.request_updates) {
    *update.target = std::move(update.value);
  }
  statistics_ = prepared.statistics;
  kv_cache_.swap(prepared.kv_cache);
  next_iteration_number_ = prepared.next_iteration_number;
  plan_trace_.push_back(std::move(prepared.trace_entry));
}

IterationResult ContinuousBatchingEngine::run_next_iteration() {
  if (failed_) {
    throw std::logic_error("cannot run a failed continuous engine");
  }
  try {
    std::set<RequestId> prepared_arrivals = arrivals_at_current_timestamp();
    if (!has_runnable_work(prepared_arrivals)) {
      const auto next_arrival = next_future_arrival(prepared_arrivals);
      if (!next_arrival.has_value()) {
        return IterationResult::complete();
      }
      const BatchPlan plan = BatchPlan::empty(next_iteration_number_);
      PreparedIteration prepared = prepare_iteration(
          plan, std::move(prepared_arrivals), *next_arrival, false);
      commit_iteration(std::move(prepared));
      return IterationResult::progressed();
    }

    const BatchPlan plan =
        build_plan(next_iteration_number_, prepared_arrivals);
    if (plan.deferred_only()) {
      PreparedIteration prepared = prepare_iteration(
          plan, std::move(prepared_arrivals), clock_.now_us(), false);
      commit_iteration(std::move(prepared));
      return IterationResult::stalled();
    }
    const auto duration_us = backend_.estimate_batch_time_us(
        plan.total_prefill_tokens(), plan.total_decode_tokens(),
        static_cast<std::uint64_t>(plan.scheduled_sequence_count()));
    PreparedIteration prepared = prepare_iteration(
        plan, std::move(prepared_arrivals), duration_us, true);
    commit_iteration(std::move(prepared));
    return IterationResult::progressed();
  } catch (...) {
    failed_ = true;
    throw;
  }
}

RunResult ContinuousBatchingEngine::run() {
  while (true) {
    const IterationResult result = run_next_iteration();
    if (result.is_stalled()) {
      return RunResult::Stalled;
    }
    if (result.outcome() == IterationOutcome::Complete) {
      break;
    }
  }
  if (has_runnable_work(arrived_requests_) ||
      next_future_arrival(arrived_requests_).has_value()) {
    failed_ = true;
    throw std::logic_error("continuous simulation drained with unfinished work");
  }
  return RunResult::Completed;
}

void ContinuousBatchingEngine::require_results_available() const {
  if (failed_) {
    throw std::logic_error(
        "continuous simulation results are unavailable after failure");
  }
}

const Request& ContinuousBatchingEngine::request(RequestId request_id) const {
  require_results_available();
  const auto found = requests_.find(request_id);
  if (found == requests_.end()) {
    throw std::out_of_range("unknown request ID");
  }
  return found->second;
}

const std::map<RequestId, Request>& ContinuousBatchingEngine::requests() const {
  require_results_available();
  return requests_;
}

const std::vector<BatchTraceEntry>& ContinuousBatchingEngine::plan_trace()
    const {
  require_results_available();
  return plan_trace_;
}

const ContinuousBatchingStatistics&
ContinuousBatchingEngine::statistics() const {
  require_results_available();
  return statistics_;
}

const SimulationClock& ContinuousBatchingEngine::clock() const {
  require_results_available();
  return clock_;
}

const KVCacheManager& ContinuousBatchingEngine::kv_cache() const {
  require_results_available();
  return kv_cache_;
}

}  // namespace llm_lab::serving
