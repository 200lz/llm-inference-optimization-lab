#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace q8rn2 {

struct block_q8_0 {
    std::uint16_t d;
    std::int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34);

using tile_output = std::array<float, 8>;

float half_reference(std::uint16_t bits);
std::array<float, 2> scales_reference(std::uint16_t a, std::uint16_t b);
std::array<float, 2> scales_baseline(std::uint16_t a, std::uint16_t b);
std::array<float, 2> scales_optimized(std::uint16_t a, std::uint16_t b);

tile_output tile_baseline(const block_q8_0 * a_rows, const block_q8_0 * b_cols, std::size_t depth);
tile_output tile_optimized(const block_q8_0 * a_rows, const block_q8_0 * b_cols, std::size_t depth);
tile_output tile_reference(const block_q8_0 * a_rows, const block_q8_0 * b_cols, std::size_t depth);

}  // namespace q8rn2
