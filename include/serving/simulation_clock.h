#pragma once

#include <cstdint>
#include <stdexcept>

namespace llm_lab::serving {

class SimulationClock {
 public:
  std::int64_t now_us() const noexcept { return now_us_; }

  void advance_to(std::int64_t timestamp_us) {
    if (timestamp_us < now_us_) {
      throw std::invalid_argument("simulation clock cannot move backwards");
    }
    now_us_ = timestamp_us;
  }

 private:
  std::int64_t now_us_{0};
};

}  // namespace llm_lab::serving
