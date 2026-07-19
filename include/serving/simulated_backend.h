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
  // Phase S3 mixed-batch cost parameters. They are intentionally separate
  // from the S1/S2 APIs so existing single-active behavior is unchanged.
  std::int64_t batch_base_us{0};
  std::int64_t batch_prefill_per_token_us{0};
  std::int64_t batch_decode_per_sequence_us{0};
  std::int64_t batch_active_sequence_overhead_us{0};
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
  std::int64_t estimate_batch_time_us(
      std::uint64_t total_prefill_tokens,
      std::uint64_t decode_sequence_count,
      std::uint64_t total_scheduled_sequences) const;

  const SimulatedBackendConfig& config() const noexcept { return config_; }

 private:
  SimulatedBackendConfig config_;
};

}  // namespace llm_lab::serving
