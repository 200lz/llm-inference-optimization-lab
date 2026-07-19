#include "serving/simulated_backend.h"

#include "serving/checked_math.h"

#include <limits>
#include <stdexcept>

namespace llm_lab::serving {
namespace {

std::int64_t checked_count(std::uint64_t value, const char* description) {
  if (value > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(description);
  }
  return static_cast<std::int64_t>(value);
}

}  // namespace

std::int64_t checked_add(std::int64_t lhs, std::int64_t rhs) {
  if ((rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs)) {
    throw std::overflow_error("signed 64-bit addition overflow");
  }
  return lhs + rhs;
}

std::int64_t checked_multiply(std::int64_t lhs, std::int64_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if ((lhs == -1 && rhs == std::numeric_limits<std::int64_t>::min()) ||
      (rhs == -1 && lhs == std::numeric_limits<std::int64_t>::min())) {
    throw std::overflow_error("signed 64-bit multiplication overflow");
  }
  if (lhs > 0) {
    if ((rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() / rhs) ||
        (rhs < 0 && rhs < std::numeric_limits<std::int64_t>::min() / lhs)) {
      throw std::overflow_error("signed 64-bit multiplication overflow");
    }
  } else if ((rhs > 0 && lhs < std::numeric_limits<std::int64_t>::min() / rhs) ||
             (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::max() / rhs)) {
    throw std::overflow_error("signed 64-bit multiplication overflow");
  }
  return lhs * rhs;
}

SimulatedBackend::SimulatedBackend(SimulatedBackendConfig config)
    : config_(config) {
  validate_configuration();
}

void SimulatedBackend::validate_configuration() const {
  if (config_.prefill_base_us < 0 || config_.prefill_per_token_us < 0 ||
      config_.prefill_per_active_sequence_us < 0 ||
      config_.decode_base_us < 0 ||
      config_.decode_per_active_sequence_us < 0 ||
      config_.batch_base_us < 0 ||
      config_.batch_prefill_per_token_us < 0 ||
      config_.batch_decode_per_sequence_us < 0 ||
      config_.batch_active_sequence_overhead_us < 0) {
    throw std::invalid_argument("simulated backend costs must be non-negative");
  }
}

std::int64_t SimulatedBackend::estimate_batch_time_us(
    std::uint64_t total_prefill_tokens,
    std::uint64_t decode_sequence_count,
    std::uint64_t total_scheduled_sequences) const {
  if (total_scheduled_sequences == 0) {
    throw std::invalid_argument("mixed batch must schedule a sequence");
  }
  if (decode_sequence_count > total_scheduled_sequences) {
    throw std::invalid_argument(
        "decode sequence count exceeds scheduled sequence count");
  }
  const auto prefill = checked_count(total_prefill_tokens,
                                     "batch prefill token count overflow");
  const auto decode = checked_count(decode_sequence_count,
                                    "batch decode sequence count overflow");
  const auto sequences = checked_count(total_scheduled_sequences,
                                       "batch sequence count overflow");
  const auto prefill_cost = checked_multiply(
      config_.batch_prefill_per_token_us, prefill);
  const auto decode_cost = checked_multiply(
      config_.batch_decode_per_sequence_us, decode);
  const auto sequence_cost = checked_multiply(
      config_.batch_active_sequence_overhead_us, sequences);
  return checked_add(
      checked_add(checked_add(config_.batch_base_us, prefill_cost),
                  decode_cost),
      sequence_cost);
}

std::int64_t SimulatedBackend::estimate_prefill_us(
    std::uint64_t prompt_token_count,
    std::uint64_t active_batch_size) const {
  if (active_batch_size == 0) {
    throw std::invalid_argument("active prefill batch size must be positive");
  }
  const auto tokens = checked_count(prompt_token_count, "prompt count overflow");
  const auto active = checked_count(active_batch_size, "batch size overflow");
  const auto token_cost =
      checked_multiply(config_.prefill_per_token_us, tokens);
  const auto batch_cost =
      checked_multiply(config_.prefill_per_active_sequence_us, active);
  return checked_add(checked_add(config_.prefill_base_us, token_cost),
                     batch_cost);
}

std::int64_t SimulatedBackend::estimate_decode_step_us(
    std::uint64_t active_sequence_count) const {
  if (active_sequence_count == 0) {
    throw std::invalid_argument("active decode sequence count must be positive");
  }
  const auto active =
      checked_count(active_sequence_count, "active sequence count overflow");
  return checked_add(
      config_.decode_base_us,
      checked_multiply(config_.decode_per_active_sequence_us, active));
}

}  // namespace llm_lab::serving
