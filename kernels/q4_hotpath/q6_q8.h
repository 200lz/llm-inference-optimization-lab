#pragma once
#include <cstddef>
#include <cstdint>

namespace q4_hotpath {
constexpr int qk = 256;
#pragma pack(push, 1)
struct block_q6_k { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
#pragma pack(pop)
struct block_q8_k { float d; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(block_q6_k) == 210);
static_assert(sizeof(block_q8_k) == 292);

float fp16_to_fp32(uint16_t value);
float reference(const block_q6_k *, const block_q8_k *, std::size_t blocks);
float baseline(const block_q6_k *, const block_q8_k *, std::size_t blocks);
float optimized(const block_q6_k *, const block_q8_k *, std::size_t blocks);
float optimized_avx2(const block_q6_k *, const block_q8_k *, std::size_t blocks);
int decode_q6(const block_q6_k &, int index);
}
