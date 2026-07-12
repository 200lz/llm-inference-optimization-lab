#include "q8_rn2_scale.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
std::uint32_t bits(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return u; }

bool compare_pair(std::uint16_t a, std::uint16_t b, std::size_t & mismatch, std::size_t & nan_class,
                  float & max_abs, float & max_rel) {
    const auto r = q8rn2::scales_reference(a, b);
    const auto x = q8rn2::scales_baseline(a, b);
    const auto y = q8rn2::scales_optimized(a, b);
    bool ok = true;
    for (int i = 0; i < 2; ++i) {
        if (std::isnan(x[i]) || std::isnan(y[i])) {
            if (std::isnan(x[i]) != std::isnan(y[i])) { ++nan_class; ok = false; }
            continue;
        }
        if (bits(x[i]) != bits(y[i]) || bits(r[i]) != bits(y[i])) { ++mismatch; ok = false; }
        const float e = std::abs(x[i] - y[i]);
        max_abs = std::max(max_abs, e);
        if (x[i] != 0) max_rel = std::max(max_rel, e / std::abs(x[i]));
    }
    return ok;
}

bool tile_trials(std::size_t depth, std::mt19937 & rng) {
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_int_distribution<int> finite(0, 0x7bff);
    std::size_t mismatch = 0;
    double mean_abs = 0;
    float max_abs = 0, max_scaled_rel = 0;
    constexpr int trials = 2000;
    for (int t = 0; t < trials; ++t) {
        std::vector<q8rn2::block_q8_0> a(4 * depth), b(2 * depth);
        for (auto & block : a) {
            block.d = static_cast<std::uint16_t>(finite(rng));
            if (t == 0) block.d = 0;
            for (int i = 0; i < 32; ++i) block.qs[i] = static_cast<std::int8_t>(t == 0 ? 0 : t == 1 ? 1 : t == 2 ? (i & 1 ? -127 : 127) : q(rng));
        }
        for (auto & block : b) {
            block.d = static_cast<std::uint16_t>(finite(rng));
            if (t == 0) block.d = 0;
            for (int i = 0; i < 32; ++i) block.qs[i] = static_cast<std::int8_t>(t == 0 ? 0 : t == 1 ? -1 : t == 2 ? (i & 1 ? 127 : -127) : q(rng));
        }
        const auto x = q8rn2::tile_baseline(a.data(), b.data(), depth);
        const auto y = q8rn2::tile_optimized(a.data(), b.data(), depth);
        const auto r = q8rn2::tile_reference(a.data(), b.data(), depth);
        for (int i = 0; i < 8; ++i) {
            if (bits(x[i]) != bits(y[i])) ++mismatch;
            const float e = std::abs(x[i] - r[i]);
            max_abs = std::max(max_abs, e);
            max_scaled_rel = std::max(max_scaled_rel, e / (1.0f + std::abs(r[i])));
            mean_abs += e;
        }
    }
    std::cout << "tile depth=" << depth << " trials=" << trials << " mismatches=" << mismatch
              << " max_abs=" << max_abs << " max_scaled_rel=" << max_scaled_rel
              << " mean_abs=" << mean_abs / (trials * 8) << '\n';
    return mismatch == 0;
}
}  // namespace

int main() {
    const std::vector<std::pair<std::uint16_t, std::uint16_t>> adversarial = {
        {0x0000,0x0000}, {0x8000,0x0000}, {0x0001,0x03ff}, {0x0400,0x7bff},
        {0x3c00,0xbc00}, {0x7c00,0xfc00}, {0x7e01,0xfe55}, {0x3555,0x3555},
        {0x0400,0x7bff}, {0x0001,0x8001}
    };
    std::size_t mismatch = 0, nan_class = 0;
    float max_abs = 0, max_rel = 0;
    for (auto p : adversarial) compare_pair(p.first, p.second, mismatch, nan_class, max_abs, max_rel);
    std::mt19937 rng(0x8c2026u);
    std::uniform_int_distribution<int> bits16(0, 65535);
    for (int i = 0; i < 10000; ++i) compare_pair(bits16(rng), bits16(rng), mismatch, nan_class, max_abs, max_rel);
    std::cout << "scale trials=" << adversarial.size() + 10000 << " mismatches=" << mismatch
              << " nan_class_mismatches=" << nan_class << " max_abs=" << max_abs
              << " max_rel=" << max_rel << '\n';
    const bool ok = mismatch == 0 && nan_class == 0 && tile_trials(28, rng) && tile_trials(152, rng);
    return ok ? 0 : 1;
}
