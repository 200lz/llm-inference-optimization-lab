#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace llm_lab::serving {

struct RequestId {
  std::uint64_t value{};
};

constexpr bool operator==(RequestId lhs, RequestId rhs) noexcept {
  return lhs.value == rhs.value;
}
constexpr bool operator!=(RequestId lhs, RequestId rhs) noexcept {
  return !(lhs == rhs);
}
constexpr bool operator<(RequestId lhs, RequestId rhs) noexcept {
  return lhs.value < rhs.value;
}

enum class RequestState {
  Waiting,
  Prefilling,
  Decoding,
  Finished,
  // Preempted remains reserved; Phase S2 implements explicit cancellation.
  Preempted,
  Cancelled,
};

std::string_view to_string(RequestState state) noexcept;
bool is_legal_transition(RequestState from, RequestState to) noexcept;

struct Request {
  RequestId request_id;
  std::int64_t arrival_time_us;
  std::uint64_t prompt_token_count;
  std::uint64_t max_new_tokens;
  std::uint64_t generated_token_count{0};
  RequestState state{RequestState::Waiting};
  std::optional<std::int64_t> admitted_time_us;
  std::optional<std::int64_t> first_scheduled_time_us;
  std::optional<std::int64_t> first_token_time_us;
  std::optional<std::int64_t> finish_time_us;

  Request(RequestId id, std::int64_t arrival_us,
          std::uint64_t prompt_tokens, std::uint64_t output_limit);

  // Throws std::logic_error for an illegal transition.
  void transition_to(RequestState next);
};

}  // namespace llm_lab::serving
