#include "serving/continuous_batching.h"
#include "serving/event.h"
#include "serving/request.h"
#include "serving/simulated_backend.h"
#include "serving/simulation_engine.h"

#include <charconv>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace llm_lab::serving;

constexpr std::string_view kSchema = "serving-simulator-v2";

struct InputRequest {
  Request request;
  std::string external_id;
  std::string workload_class;
};

std::string json_escape(std::string_view text) {
  std::ostringstream out;
  for (unsigned char value : text) {
    switch (value) {
      case '\"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (value < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(value) << std::dec;
        } else {
          out << static_cast<char>(value);
        }
    }
  }
  return out.str();
}

template <typename T>
T parse_integer(std::string_view value, const char* name) {
  T parsed{};
  const char* begin = value.data();
  const char* end = begin + value.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " +
                                std::string(value));
  }
  return parsed;
}

std::vector<std::string> split(const std::string& value, char delimiter) {
  std::vector<std::string> fields;
  std::size_t begin = 0;
  for (;;) {
    const std::size_t end = value.find(delimiter, begin);
    fields.push_back(value.substr(begin, end - begin));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return fields;
}

std::map<std::string, std::string> parse_arguments(int argc, char** argv) {
  std::map<std::string, std::string> values;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc || std::string_view(argv[index]).rfind("--", 0) != 0)
      throw std::invalid_argument("arguments must be --name value pairs");
    const std::string name = std::string(argv[index]).substr(2);
    if (!values.emplace(name, argv[index + 1]).second)
      throw std::invalid_argument("duplicate argument --" + name);
  }
  return values;
}

const std::string& required(const std::map<std::string, std::string>& args,
                            const std::string& name) {
  const auto found = args.find(name);
  if (found == args.end()) throw std::invalid_argument("missing --" + name);
  return found->second;
}

std::vector<InputRequest> read_workload(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open workload TSV: " + path);
  std::string line;
  if (!std::getline(input, line) ||
      line != "internal_id\trequest_id\tworkload_class\tarrival_time_us\tprompt_token_count\tprompt_tokens\tmax_new_tokens")
    throw std::invalid_argument("invalid workload TSV header");
  std::vector<InputRequest> requests;
  std::uint64_t line_number = 1;
  std::map<std::uint64_t, bool> ids;
  std::map<std::string, bool> external_ids;
  while (std::getline(input, line)) {
    ++line_number;
    const auto fields = split(line, '\t');
    if (fields.size() != 7)
      throw std::invalid_argument("workload TSV line " +
          std::to_string(line_number) + ": expected 7 fields");
    const auto internal_id = parse_integer<std::uint64_t>(fields[0], "internal_id");
    const auto arrival = parse_integer<std::int64_t>(fields[3], "arrival_time_us");
    const auto prompt_count = parse_integer<std::uint64_t>(fields[4], "prompt_token_count");
    const auto output = parse_integer<std::uint64_t>(fields[6], "max_new_tokens");
    if (!ids.emplace(internal_id, true).second)
      throw std::invalid_argument("workload TSV line " +
          std::to_string(line_number) + ": duplicate internal_id");
    if (fields[1].empty() || !external_ids.emplace(fields[1], true).second)
      throw std::invalid_argument("workload TSV line " +
          std::to_string(line_number) + ": empty or duplicate request_id");
    if (fields[2].empty())
      throw std::invalid_argument("workload TSV line " +
          std::to_string(line_number) + ": empty workload_class");
    Request request = Request::count_only({internal_id}, arrival, prompt_count, output);
    if (fields[5] != "-") {
      std::vector<std::int32_t> tokens;
      if (!fields[5].empty()) {
        for (const std::string& token : split(fields[5], ','))
          tokens.push_back(parse_integer<std::int32_t>(token, "prompt token"));
      }
      if (tokens.size() != prompt_count)
        throw std::invalid_argument("workload TSV line " +
            std::to_string(line_number) + ": prompt count/token mismatch");
      request = Request::exact_tokens({internal_id}, arrival, std::move(tokens), output);
    }
    requests.push_back({std::move(request), fields[1], fields[2]});
  }
  return requests;
}

std::string ids_json(const std::vector<RequestId>& ids) {
  std::ostringstream out;
  out << '[';
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index) out << ',';
    out << ids[index].value;
  }
  out << ']';
  return out.str();
}

std::string deferrals_json(const std::vector<DeferredRequest>& items) {
  std::ostringstream out;
  out << '[';
  for (std::size_t index = 0; index < items.size(); ++index) {
    if (index) out << ',';
    out << "{\"request_id\":" << items[index].request_id.value
        << ",\"reason\":\"" << json_escape(to_string(items[index].reason)) << "\"}";
  }
  out << ']';
  return out.str();
}

void write_optional(std::ostream& out, const std::optional<std::int64_t>& value) {
  if (value) out << *value; else out << "null";
}

void write_request(std::ostream& out, const InputRequest& input,
                   const Request& request, std::uint64_t matched,
                   std::uint64_t scheduled, const std::vector<std::int64_t>& token_times) {
  out << "{\"record_type\":\"request\",\"schema_version\":\"" << kSchema
      << "\",\"evidence_type\":\"SIMULATED\",\"internal_id\":"
      << request.request_id.value << ",\"request_id\":\""
      << json_escape(input.external_id) << "\",\"workload_class\":\""
      << json_escape(input.workload_class) << "\",\"arrival_time_us\":"
      << request.arrival_time_us << ",\"admitted_time_us\":";
  write_optional(out, request.admitted_time_us);
  out << ",\"first_token_time_us\":"; write_optional(out, request.first_token_time_us);
  out << ",\"finish_time_us\":"; write_optional(out, request.finish_time_us);
  out << ",\"queue_delay_us\":";
  if (request.admitted_time_us)
    out << *request.admitted_time_us - request.arrival_time_us;
  else out << "null";
  out << ",\"ttft_us\":";
  if (request.generated_token_count > 0 && request.first_token_time_us)
    out << *request.first_token_time_us - request.arrival_time_us;
  else out << "null";
  out << ",\"end_to_end_latency_us\":";
  if (request.finish_time_us)
    out << *request.finish_time_us - request.arrival_time_us;
  else out << "null";
  out << ",\"tpot_us\":";
  if (request.generated_token_count >= 2 && request.first_token_time_us &&
      request.finish_time_us) {
    out << std::setprecision(17)
        << static_cast<double>(*request.finish_time_us -
                               *request.first_token_time_us) /
               static_cast<double>(request.generated_token_count - 1);
  } else {
    out << "null";
  }
  out << ",\"prompt_tokens_original\":" << request.prompt_length()
      << ",\"prompt_tokens_matched\":" << matched
      << ",\"prompt_tokens_scheduled\":" << scheduled
      << ",\"generated_tokens\":" << request.generated_token_count
      << ",\"max_new_tokens\":" << request.max_new_tokens
      << ",\"final_state\":\"" << to_string(request.state)
      << "\",\"reason\":null,\"decode_token_times_us\":[";
  for (std::size_t index = 0; index < token_times.size(); ++index) {
    if (index) out << ',';
    out << token_times[index];
  }
  out << "],\"inter_token_latencies_us\":[";
  for (std::size_t index = 1; index < token_times.size(); ++index) {
    if (index > 1) out << ',';
    out << token_times[index] - token_times[index - 1];
  }
  out << "]}\n";
}

int run_continuous(const std::map<std::string, std::string>& args,
                   const std::vector<InputRequest>& inputs, std::ostream& out,
                   const SimulatedBackend& backend) {
  const std::string policy_name = required(args, "policy");
  SchedulingPolicy policy;
  if (policy_name == "DecodeFirst") policy = SchedulingPolicy::DecodeFirst;
  else if (policy_name == "FcfsMixed") policy = SchedulingPolicy::FcfsMixed;
  else throw std::invalid_argument("unknown scheduling policy: " + policy_name);
  const bool prefix = parse_integer<int>(required(args, "prefix-cache"), "prefix-cache") != 0;
  ContinuousBatchingConfig config(
      parse_integer<std::size_t>(required(args, "max-num-sequences"), "max-num-sequences"),
      parse_integer<std::uint64_t>(required(args, "max-batched-tokens"), "max-batched-tokens"),
      policy,
      KVCacheConfig(parse_integer<std::size_t>(required(args, "kv-total-blocks"), "kv-total-blocks"),
                    parse_integer<std::uint64_t>(required(args, "kv-block-size"), "kv-block-size")),
      PrefixCacheConfig(prefix, required(args, "salt"), required(args, "namespace")));
  ContinuousBatchingEngine engine(backend, config);
  for (const auto& input : inputs) engine.submit_request(input.request);
  const RunResult run_result = engine.run();
  std::map<RequestId, std::uint64_t> matched;
  std::map<RequestId, std::uint64_t> scheduled;
  std::map<RequestId, std::vector<std::int64_t>> token_times;
  for (const BatchTraceEntry& trace : engine.plan_trace()) {
    const BatchPlan& plan = trace.plan;
    std::uint64_t iteration_hits = 0;
    std::uint64_t iteration_misses = 0;
    std::uint64_t iteration_matched_tokens = 0;
    for (const PrefillWork& work : plan.prefill_work()) {
      matched[work.request_id] += work.matched_prefix_token_count;
      scheduled[work.request_id] += work.prompt_token_count;
      iteration_matched_tokens += work.matched_prefix_token_count;
      if (work.prefix_lookup_kind != PrefixLookupKind::Disabled) {
        if (work.matched_prefix_token_count > 0) ++iteration_hits;
        else ++iteration_misses;
      }
    }
    for (RequestId id : plan.decode_request_ids()) token_times[id].push_back(trace.end_timestamp_us);
    out << "{\"record_type\":\"iteration\",\"schema_version\":\"" << kSchema
        << "\",\"evidence_type\":\"SIMULATED\",\"iteration_number\":"
        << trace.iteration_number << ",\"start_time_us\":" << trace.start_timestamp_us
        << ",\"end_time_us\":" << trace.end_timestamp_us << ",\"policy\":\""
        << to_string(trace.policy) << "\",\"prefill_ids\":"
        << ids_json(plan.prefill_request_ids()) << ",\"decode_ids\":"
        << ids_json(plan.decode_request_ids()) << ",\"deferred\":"
        << deferrals_json(plan.deferred_requests()) << ",\"scheduled_sequences\":"
        << plan.scheduled_sequence_count() << ",\"scheduled_prefill_tokens\":"
        << plan.total_prefill_tokens() << ",\"scheduled_decode_tokens\":"
        << plan.total_decode_tokens() << ",\"total_scheduled_tokens\":"
        << plan.total_scheduled_tokens() << ",\"kv_allocated\":"
        << trace.allocated_kv_blocks << ",\"kv_free\":" << trace.free_kv_blocks
        << ",\"kv_utilization\":" << std::setprecision(17)
        << trace.kv_block_utilization << ",\"internal_fragmentation_tokens\":"
        << trace.internal_fragmentation_tokens
        << ",\"represented_kv_tokens\":" << trace.represented_kv_tokens
        << ",\"cached_blocks\":" << trace.cached_kv_blocks
        << ",\"shared_referenced_blocks\":" << trace.referenced_shared_kv_blocks
        << ",\"prefix_hits\":" << iteration_hits
        << ",\"prefix_misses\":" << iteration_misses
        << ",\"matched_tokens\":" << iteration_matched_tokens
        << ",\"evicted_ids\":[";
    const auto& evicted = plan.planned_eviction_ids();
    for (std::size_t index = 0; index < evicted.size(); ++index) {
      if (index) out << ',';
      out << evicted[index];
    }
    out << "],\"stall\":" << (plan.deferred_only() ? "true" : "false") << "}\n";
  }
  for (const auto& input : inputs)
    write_request(out, input, engine.request(input.request.request_id),
                  matched[input.request.request_id], scheduled[input.request.request_id],
                  token_times[input.request.request_id]);
  const auto& stats = engine.statistics();
  std::uint64_t submitted_prompt_tokens = 0;
  std::uint64_t generated_tokens = 0;
  for (const auto& input : inputs) {
    submitted_prompt_tokens += input.request.prompt_length();
    generated_tokens += engine.request(input.request.request_id).generated_token_count;
  }
  out << "{\"record_type\":\"summary\",\"schema_version\":\"" << kSchema
      << "\",\"evidence_type\":\"SIMULATED\",\"execution_mode\":\"continuous_batching\""
      << ",\"scheduling_policy\":\"" << policy_name << "\",\"run_status\":\""
      << (run_result == RunResult::Completed ? "completed" : "stalled")
      << "\",\"submitted\":" << inputs.size()
      << ",\"completed\":" << stats.completed_request_count
      << ",\"cancelled\":" << stats.cancelled_request_count
      << ",\"rejected\":0,\"stalled\":" << stats.stalled_iteration_count
      << ",\"makespan_us\":" << engine.clock().now_us()
      << ",\"scheduling_iterations\":" << stats.scheduling_iteration_count
      << ",\"nonempty_batches\":" << stats.nonempty_batch_count
      << ",\"idle_iterations\":" << stats.idle_iteration_count
      << ",\"total_scheduled_sequences\":" << stats.total_scheduled_sequences
      << ",\"max_scheduled_tokens\":" << stats.maximum_scheduled_tokens
      << ",\"deferred_requests\":" << stats.deferred_request_count
      << ",\"average_batch_size\":";
  if (stats.average_batch_size()) out << *stats.average_batch_size(); else out << "null";
  out << ",\"max_batch_size\":" << stats.maximum_batch_size
      << ",\"current_allocated_blocks\":" << stats.current_allocated_kv_blocks
      << ",\"current_free_blocks\":" << engine.kv_cache().free_block_count()
      << ",\"kv_peak_utilization\":" << stats.peak_kv_block_utilization
      << ",\"kv_current_utilization\":" << stats.current_kv_block_utilization
      << ",\"prefix_token_hit_rate\":";
  if (stats.prefix_token_hit_rate()) out << *stats.prefix_token_hit_rate(); else out << "null";
  out << ",\"prefix_lookup_hits\":" << stats.cache_hit_lookup_count
      << ",\"prefix_lookup_misses\":" << stats.cache_miss_lookup_count
      << ",\"prefix_lookups\":" << stats.cache_lookup_count
      << ",\"matched_prefix_blocks\":" << stats.cache_matched_block_count
      << ",\"cache_eligible_prompt_tokens\":"
      << stats.total_cache_eligible_prompt_tokens_looked_up
      << ",\"collision_verifications\":" << stats.collision_verification_count
      << ",\"matched_prefix_tokens\":" << stats.total_matched_prefix_tokens
      << ",\"saved_simulated_prefill_tokens\":" << stats.saved_simulated_prefill_tokens
      << ",\"scheduled_prefill_tokens\":" << stats.total_prefill_tokens_scheduled
      << ",\"scheduled_decode_tokens\":" << stats.total_decode_tokens_scheduled
      << ",\"submitted_prompt_tokens\":" << submitted_prompt_tokens
      << ",\"generated_tokens\":" << generated_tokens
      << ",\"kv_capacity_deferrals\":" << stats.kv_capacity_deferral_count
      << ",\"kv_allocation_failures\":" << stats.kv_allocation_failure_count
      << ",\"eviction_count\":" << stats.prefix_cache_eviction_count
      << ",\"internal_fragmentation_tokens\":" << stats.internal_fragmentation_tokens
      << ",\"represented_kv_tokens\":" << stats.represented_kv_tokens
      << ",\"cached_blocks\":" << stats.cached_kv_blocks
      << ",\"shared_referenced_blocks\":" << stats.referenced_shared_kv_blocks
      << ",\"peak_allocated_blocks\":" << stats.peak_allocated_kv_blocks << "}\n";
  return 0;
}

int run_fcfs(const std::vector<InputRequest>& inputs, std::ostream& out,
             const SimulatedBackend& backend) {
  SimulationEngine engine(backend);
  for (const auto& input : inputs) engine.submit_request(input.request);
  engine.run();
  std::map<RequestId, std::vector<std::int64_t>> token_times;
  for (const Event& event : engine.event_log())
    if (event.type == EventType::DecodeStepComplete)
      token_times[event.request_id].push_back(event.timestamp_us);
  for (const auto& input : inputs)
    write_request(out, input, engine.request(input.request.request_id), 0,
                  input.request.prompt_length(), token_times[input.request.request_id]);
  std::uint64_t scheduled_prefill_tokens = 0;
  std::uint64_t scheduled_decode_tokens = 0;
  for (const auto& input : inputs) {
    scheduled_prefill_tokens += input.request.prompt_length();
    scheduled_decode_tokens +=
        engine.request(input.request.request_id).generated_token_count;
  }
  out << "{\"record_type\":\"summary\",\"schema_version\":\"" << kSchema
      << "\",\"evidence_type\":\"SIMULATED\",\"execution_mode\":\"single_active_fcfs\""
      << ",\"scheduling_policy\":null,\"run_status\":\"completed\""
      << ",\"submitted\":" << inputs.size()
      << ",\"completed\":" << inputs.size()
      << ",\"cancelled\":0,\"rejected\":0,\"stalled\":0,\"makespan_us\":"
      << engine.clock().now_us()
      << ",\"scheduling_iterations\":null"
      << ",\"nonempty_batches\":null"
      << ",\"idle_iterations\":null,\"total_scheduled_sequences\":null"
      << ",\"max_scheduled_tokens\":null,\"deferred_requests\":0"
      << ",\"average_batch_size\":" << (inputs.empty() ? "null" : "1")
      << ",\"max_batch_size\":" << (inputs.empty() ? 0 : 1)
      << ",\"current_allocated_blocks\":null,\"current_free_blocks\":null"
      << ",\"kv_peak_utilization\":null,\"kv_current_utilization\":null"
      << ",\"prefix_token_hit_rate\":null"
      << ",\"prefix_lookup_hits\":0,\"prefix_lookup_misses\":0"
      << ",\"prefix_lookups\":0,\"matched_prefix_blocks\":0"
      << ",\"cache_eligible_prompt_tokens\":0,\"collision_verifications\":0"
      << ",\"matched_prefix_tokens\":0,\"saved_simulated_prefill_tokens\":0"
      << ",\"scheduled_prefill_tokens\":" << scheduled_prefill_tokens
      << ",\"scheduled_decode_tokens\":" << scheduled_decode_tokens
      << ",\"submitted_prompt_tokens\":" << scheduled_prefill_tokens
      << ",\"generated_tokens\":" << scheduled_decode_tokens
      << ",\"kv_capacity_deferrals\":0,\"kv_allocation_failures\":0"
      << ",\"eviction_count\":0,\"internal_fragmentation_tokens\":null"
      << ",\"represented_kv_tokens\":null,\"cached_blocks\":null"
      << ",\"shared_referenced_blocks\":null,\"peak_allocated_blocks\":null}\n";
  return 0;
}

int main_impl(int argc, char** argv) {
  const auto args = parse_arguments(argc, argv);
  std::ifstream unused;
  const auto workload = read_workload(required(args, "input"));
  const SimulatedBackend backend({
      parse_integer<std::int64_t>(required(args, "prefill-base-us"), "prefill-base-us"),
      parse_integer<std::int64_t>(required(args, "prefill-per-token-us"), "prefill-per-token-us"),
      parse_integer<std::int64_t>(required(args, "prefill-per-sequence-us"), "prefill-per-sequence-us"),
      parse_integer<std::int64_t>(required(args, "decode-base-us"), "decode-base-us"),
      parse_integer<std::int64_t>(required(args, "decode-per-sequence-us"), "decode-per-sequence-us"),
      parse_integer<std::int64_t>(required(args, "batch-base-us"), "batch-base-us"),
      parse_integer<std::int64_t>(required(args, "batch-prefill-per-token-us"), "batch-prefill-per-token-us"),
      parse_integer<std::int64_t>(required(args, "batch-decode-per-sequence-us"), "batch-decode-per-sequence-us"),
      parse_integer<std::int64_t>(required(args, "batch-active-overhead-us"), "batch-active-overhead-us")});
  const std::string output_path = required(args, "output");
  std::ofstream output(output_path);
  if (!output) throw std::runtime_error("cannot open output: " + output_path);
  const std::string mode = required(args, "mode");
  if (mode == "continuous_batching") return run_continuous(args, workload, output, backend);
  if (mode == "single_active_fcfs") return run_fcfs(workload, output, backend);
  throw std::invalid_argument("unknown execution mode: " + mode);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "serving-benchmark-runner: " << error.what() << '\n';
    return 2;
  }
}
