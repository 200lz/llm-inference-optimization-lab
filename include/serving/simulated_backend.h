#pragma once

#include "serving/backend.h"

#include <cstdint>

namespace llm_lab::serving {

struct SimulatedBackendConfig {
  std::int64_t prefill_base_us{0};
  std::int64_t prefill_per_token_us{0};
  std::int64_t prefill_per_active_sequence_us{0};
  std::int64_t decode_base_us{0};
  std::int64_t decode_per_active_sequence_us{0};
};

class SimulatedBackend final : public Backend {
 public:
  explicit SimulatedBackend(SimulatedBackendConfig config);

  void validate_configuration() const override;
  std::int64_t estimate_prefill_us(
      std::uint64_t prompt_token_count,
      std::uint64_t active_batch_size) const override;
  std::int64_t estimate_decode_step_us(
      std::uint64_t active_sequence_count) const override;

  const SimulatedBackendConfig& config() const noexcept { return config_; }

 private:
  SimulatedBackendConfig config_;
};

}  // namespace llm_lab::serving
