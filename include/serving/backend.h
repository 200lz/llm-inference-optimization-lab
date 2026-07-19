#pragma once

#include <cstdint>

namespace llm_lab::serving {

class Backend {
 public:
  virtual ~Backend() = default;

  virtual void validate_configuration() const = 0;
  virtual std::int64_t estimate_prefill_us(
      std::uint64_t prompt_token_count,
      std::uint64_t active_batch_size) const = 0;
  virtual std::int64_t estimate_decode_step_us(
      std::uint64_t active_sequence_count) const = 0;
};

}  // namespace llm_lab::serving
