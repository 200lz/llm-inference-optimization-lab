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
         reason == DeferralReason::TokenBudget;
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
  }
  return "unknown";
}

ContinuousBatchingConfig::ContinuousBatchingConfig(
    std::size_t max_num_sequences, std::uint64_t max_batched_tokens,
    SchedulingPolicy scheduling_policy)
    : max_num_sequences_(max_num_sequences),
      max_batched_tokens_(max_batched_tokens),
      scheduling_policy_(scheduling_policy) {
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
  if (prefill_work.empty() && decode_request_ids.empty()) {
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

  const auto total_decode_tokens =
      static_cast<std::uint64_t>(decode_request_ids.size());
  const auto total_scheduled_tokens = checked_add_unsigned(
      total_prefill_tokens, total_decode_tokens,
      "BatchPlan scheduled token total overflow");
  if (total_scheduled_tokens > max_batched_tokens) {
    throw std::invalid_argument("BatchPlan exceeds max_batched_tokens");
  }
  return BatchPlan(iteration_number, std::move(prefill_request_ids),
                   std::move(decode_request_ids),
                   std::move(deferred_requests), total_prefill_tokens,
                   total_decode_tokens, total_scheduled_tokens);
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

ContinuousBatchingEngine::ContinuousBatchingEngine(
    const SimulatedBackend& backend, ContinuousBatchingConfig config)
    : backend_(backend), config_(config) {
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
  if (request.prompt_token_count > config_.max_batched_tokens()) {
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

  static_assert(std::is_nothrow_move_assignable_v<Request>);
  request = std::move(prepared_request);
  arrived_requests_.swap(prepared_arrivals);
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
  std::vector<RequestId> decodes;
  std::vector<DeferredRequest> deferred;
  std::uint64_t used_tokens = 0;
  for (const Candidate& candidate : candidates) {
    if (candidate.token_cost >
        config_.max_batched_tokens() - used_tokens) {
      deferred.push_back(
          DeferredRequest{candidate.request_id, DeferralReason::TokenBudget});
      continue;
    }
    if (prefills.size() + decodes.size() >= config_.max_num_sequences()) {
      deferred.push_back(
          DeferredRequest{candidate.request_id, DeferralReason::SequenceBudget});
      continue;
    }
    used_tokens += candidate.token_cost;
    if (candidate.is_prefill) {
      prefills.push_back(
          PrefillWork{candidate.request_id, candidate.token_cost});
    } else {
      decodes.push_back(candidate.request_id);
    }
  }
  if (prefills.empty() && decodes.empty()) {
    throw std::logic_error("runnable work cannot fit validated budgets");
  }
  return BatchPlan::create_with_deferred(
      iteration_number, std::move(prefills), std::move(decodes),
      std::move(deferred), config_.max_num_sequences(),
      config_.max_batched_tokens());
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
          Candidate{request.request_id, request.arrival_time_us, false, 1});
    } else if (request.state == RequestState::Waiting &&
               arrived_requests.find(request.request_id) !=
                   arrived_requests.end()) {
      prefills.push_back(Candidate{request.request_id, request.arrival_time_us,
                                   true, request.prompt_token_count});
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
          Candidate{request.request_id, request.arrival_time_us, false, 1});
    } else if (request.state == RequestState::Waiting &&
               arrived_requests.find(request.request_id) !=
                   arrived_requests.end()) {
      candidates.push_back(Candidate{request.request_id,
                                     request.arrival_time_us, true,
                                     request.prompt_token_count});
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
    const BatchPlan& plan, std::uint64_t completed_request_count) const {
  ContinuousBatchingStatistics prepared = statistics_;
  prepared.scheduling_iteration_count = checked_incremented(
      prepared.scheduling_iteration_count,
      "scheduling iteration count overflow");
  if (plan.empty()) {
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
  prepared.completed_request_count = checked_add_unsigned(
      prepared.completed_request_count, completed_request_count,
      "completed request count overflow");
  prepared.maximum_batch_size =
      std::max(prepared.maximum_batch_size, sequences);
  prepared.maximum_scheduled_tokens = std::max(
      prepared.maximum_scheduled_tokens, plan.total_scheduled_tokens());
  return prepared;
}

ContinuousBatchingEngine::PreparedIteration
ContinuousBatchingEngine::prepare_iteration(
    BatchPlan plan, std::set<RequestId> arrived_requests,
    std::int64_t end_timestamp_us) {
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
  ContinuousBatchingStatistics prepared_statistics =
      prepare_statistics(plan, completed_request_count);
  const std::uint64_t prepared_next_iteration = checked_incremented(
      next_iteration_number_, "iteration number overflow");
  BatchTraceEntry trace_entry{plan.iteration_number(), clock_.now_us(),
                              end_timestamp_us, config_.scheduling_policy(),
                              std::move(plan)};
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
                           prepared_clock};
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
  next_iteration_number_ = prepared.next_iteration_number;
  plan_trace_.push_back(std::move(prepared.trace_entry));
}

bool ContinuousBatchingEngine::run_next_iteration() {
  if (failed_) {
    throw std::logic_error("cannot run a failed continuous engine");
  }
  try {
    std::set<RequestId> prepared_arrivals = arrivals_at_current_timestamp();
    if (!has_runnable_work(prepared_arrivals)) {
      const auto next_arrival = next_future_arrival(prepared_arrivals);
      if (!next_arrival.has_value()) {
        return false;
      }
      const BatchPlan plan = BatchPlan::empty(next_iteration_number_);
      PreparedIteration prepared = prepare_iteration(
          plan, std::move(prepared_arrivals), *next_arrival);
      commit_iteration(std::move(prepared));
      return true;
    }

    const BatchPlan plan =
        build_plan(next_iteration_number_, prepared_arrivals);
    if (plan.empty()) {
      throw std::logic_error("runnable work cannot fit a validated budget");
    }
    const auto duration_us = backend_.estimate_batch_time_us(
        plan.total_prefill_tokens(), plan.total_decode_tokens(),
        static_cast<std::uint64_t>(plan.scheduled_sequence_count()));
    const auto end_timestamp_us = checked_add(clock_.now_us(), duration_us);
    PreparedIteration prepared = prepare_iteration(
        plan, std::move(prepared_arrivals), end_timestamp_us);
    commit_iteration(std::move(prepared));
    return true;
  } catch (...) {
    failed_ = true;
    throw;
  }
}

void ContinuousBatchingEngine::run() {
  while (run_next_iteration()) {
  }
  if (has_runnable_work(arrived_requests_) ||
      next_future_arrival(arrived_requests_).has_value()) {
    failed_ = true;
    throw std::logic_error("continuous simulation drained with unfinished work");
  }
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

}  // namespace llm_lab::serving
