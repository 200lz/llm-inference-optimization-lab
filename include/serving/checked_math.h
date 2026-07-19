#pragma once

#include <cstdint>

namespace llm_lab::serving {

// Throws std::overflow_error when the signed 64-bit result is not representable.
std::int64_t checked_add(std::int64_t lhs, std::int64_t rhs);
std::int64_t checked_multiply(std::int64_t lhs, std::int64_t rhs);

}  // namespace llm_lab::serving
