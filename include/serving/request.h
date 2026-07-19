#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

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
  std::uint64_t max_new_tokens;
  std::uint64_t generated_token_count{0};
  RequestState state{RequestState::Waiting};
  std::optional<std::int64_t> admitted_time_us;
  std::optional<std::int64_t> first_scheduled_time_us;
  std::optional<std::int64_t> first_token_time_us;
  std::optional<std::int64_t> finish_time_us;

  Request(RequestId id, std::int64_t arrival_us,
          std::uint64_t prompt_tokens, std::uint64_t output_limit);
  Request(RequestId id, std::int64_t arrival_us,
          std::vector<std::int32_t> exact_prompt_tokens,
          std::uint64_t output_limit);

  static Request count_only(RequestId id, std::int64_t arrival_us,
                            std::uint64_t prompt_length,
                            std::uint64_t output_limit);
  static Request exact_tokens(RequestId id, std::int64_t arrival_us,
                              std::vector<std::int32_t> prompt_tokens,
                              std::uint64_t output_limit);

  std::uint64_t prompt_length() const noexcept;
  bool has_exact_prompt_tokens() const noexcept;
  const std::vector<std::int32_t>& prompt_tokens() const;

  // Throws std::logic_error for an illegal transition.
  void transition_to(RequestState next);

 private:
  std::uint64_t count_only_prompt_length_{0};
  std::optional<std::vector<std::int32_t>> exact_prompt_tokens_;
};

}  // namespace llm_lab::serving
