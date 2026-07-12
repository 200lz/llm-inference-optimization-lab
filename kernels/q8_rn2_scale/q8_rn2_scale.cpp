#include "q8_rn2_scale.h"

#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace q8rn2 {
namespace {

float bits_float(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

__m256 dot8(const block_q8_0 & a, const block_q8_0 & b) {
    const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a.qs));
    const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b.qs));
    const __m256i abs_a = _mm256_sign_epi8(av, av);
    const __m256i signed_b = _mm256_sign_epi8(bv, av);
    const __m256i pairs = _mm256_maddubs_epi16(abs_a, signed_b);
    const __m256i sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    return _mm256_cvtepi32_ps(sums);
}

float hsum(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

__m256 splat_scale(__m256 scales, int lane) {
    switch (lane) {
        case 0: return _mm256_shuffle_ps(scales, scales, 0);
        case 1: return _mm256_shuffle_ps(scales, scales, 85);
        case 2: return _mm256_shuffle_ps(scales, scales, 170);
        default: return _mm256_shuffle_ps(scales, scales, 255);
    }
}

template <bool Packed>
tile_output tile_impl(const block_q8_0 * a, const block_q8_0 * b, std::size_t depth) {
    __m256 acc[2][4] = {};
    for (std::size_t l = 0; l < depth; ++l) {
        const std::uint64_t packed_a = std::uint64_t(a[3 * depth + l].d) << 48 |
                                       std::uint64_t(a[2 * depth + l].d) << 32 |
                                       std::uint64_t(a[1 * depth + l].d) << 16 |
                                       a[l].d;
        const __m128 da = _mm_cvtph_ps(_mm_cvtsi64_si128(static_cast<long long>(packed_a)));
        __m256 scales[2];
        if constexpr (Packed) {
            const std::uint32_t packed_b = b[l].d | (std::uint32_t(b[depth + l].d) << 16);
            const __m128 db = _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(packed_b)));
            const __m128 product0 = _mm_mul_ps(da, _mm_shuffle_ps(db, db, 0));
            const __m128 product1 = _mm_mul_ps(da, _mm_shuffle_ps(db, db, 85));
            scales[0] = _mm256_permute2f128_ps(_mm256_castps128_ps256(product0),
                                               _mm256_castps128_ps256(product0), 0);
            scales[1] = _mm256_permute2f128_ps(_mm256_castps128_ps256(product1),
                                               _mm256_castps128_ps256(product1), 0);
        } else {
            for (int j = 0; j < 2; ++j) {
                const __m128 db = _mm_set1_ps(half_reference(b[j * depth + l].d));
                const __m128 product = _mm_mul_ps(da, db);
                scales[j] = _mm256_permute2f128_ps(_mm256_castps128_ps256(product),
                                                   _mm256_castps128_ps256(product), 0);
            }
        }
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 4; ++i) {
                const __m256 scale = splat_scale(scales[j], i);
                acc[j][i] = _mm256_fmadd_ps(scale, dot8(a[i * depth + l], b[j * depth + l]), acc[j][i]);
            }
        }
    }
    tile_output out{};
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 4; ++i) out[j * 4 + i] = hsum(acc[j][i]);
    return out;
}

}  // namespace

float half_reference(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1fu;
    std::uint32_t fraction = h & 0x3ffu;
    if (exponent == 0) {
        if (fraction == 0) return bits_float(sign);
        int shift = 0;
        while ((fraction & 0x400u) == 0) { fraction <<= 1; ++shift; }
        fraction &= 0x3ffu;
        return bits_float(sign | ((127u - 14u - std::uint32_t(shift)) << 23) | (fraction << 13));
    }
    if (exponent == 31) return bits_float(sign | 0x7f800000u | (fraction << 13));
    return bits_float(sign | ((exponent + 112u) << 23) | (fraction << 13));
}

std::array<float, 2> scales_reference(std::uint16_t a, std::uint16_t b) { return {half_reference(a), half_reference(b)}; }
__attribute__((noinline)) std::array<float, 2> scales_baseline(std::uint16_t a, std::uint16_t b) { return {half_reference(a), half_reference(b)}; }
__attribute__((noinline)) std::array<float, 2> scales_optimized(std::uint16_t a, std::uint16_t b) {
    const std::uint32_t packed = a | (std::uint32_t(b) << 16);
    const __m128 values = _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(packed)));
    std::array<float, 2> out{};
    _mm_storel_pi(reinterpret_cast<__m64 *>(out.data()), values);
    return out;
}

__attribute__((noinline)) tile_output tile_baseline(const block_q8_0 * a, const block_q8_0 * b, std::size_t d) { return tile_impl<false>(a, b, d); }
__attribute__((noinline)) tile_output tile_optimized(const block_q8_0 * a, const block_q8_0 * b, std::size_t d) { return tile_impl<true>(a, b, d); }

tile_output tile_reference(const block_q8_0 * a, const block_q8_0 * b, std::size_t depth) {
    tile_output out{};
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 4; ++i) {
        double total = 0;
        for (std::size_t l = 0; l < depth; ++l) {
            int dot = 0;
            for (int q = 0; q < 32; ++q) dot += int(a[i * depth + l].qs[q]) * int(b[j * depth + l].qs[q]);
            total += double(half_reference(a[i * depth + l].d)) * half_reference(b[j * depth + l].d) * dot;
        }
        out[j * 4 + i] = static_cast<float>(total);
    }
    return out;
}

}  // namespace q8rn2
