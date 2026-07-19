#include "serving/request.h"

#include <stdexcept>

namespace llm_lab::serving {

std::string_view to_string(RequestState state) noexcept {
  switch (state) {
    case RequestState::Waiting:
      return "Waiting";
    case RequestState::Prefilling:
      return "Prefilling";
    case RequestState::Decoding:
      return "Decoding";
    case RequestState::Finished:
      return "Finished";
    case RequestState::Preempted:
      return "Preempted";
    case RequestState::Cancelled:
      return "Cancelled";
  }
  return "Unknown";
}

bool is_legal_transition(RequestState from, RequestState to) noexcept {
  switch (from) {
    case RequestState::Waiting:
      return to == RequestState::Prefilling ||
             to == RequestState::Cancelled;
    case RequestState::Prefilling:
      return to == RequestState::Decoding || to == RequestState::Finished ||
             to == RequestState::Preempted ||
             to == RequestState::Cancelled;
    case RequestState::Decoding:
      return to == RequestState::Finished ||
             to == RequestState::Preempted ||
             to == RequestState::Cancelled;
    case RequestState::Preempted:
      return to == RequestState::Waiting || to == RequestState::Cancelled;
    case RequestState::Finished:
    case RequestState::Cancelled:
      return false;
  }
  return false;
}

Request::Request(RequestId id, std::int64_t arrival_us,
                 std::uint64_t prompt_tokens, std::uint64_t output_limit)
    : request_id(id),
      arrival_time_us(arrival_us),
      prompt_token_count(prompt_tokens),
      max_new_tokens(output_limit) {
  if (arrival_time_us < 0) {
    throw std::invalid_argument("request arrival_time_us must be non-negative");
  }
}

void Request::transition_to(RequestState next) {
  if (!is_legal_transition(state, next)) {
    throw std::logic_error("illegal request state transition");
  }
  if (state == RequestState::Prefilling && next == RequestState::Finished &&
      max_new_tokens != 0) {
    throw std::logic_error(
        "Prefilling-to-Finished requires a zero output limit");
  }
  state = next;
}

}  // namespace llm_lab::serving
