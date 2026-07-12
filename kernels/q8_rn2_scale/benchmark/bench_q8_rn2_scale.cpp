#include "q8_rn2_scale.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

void fill_blocks(std::vector<q8rn2::block_q8_0> & blocks, std::mt19937 & rng) {
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_int_distribution<int> d(0x0400, 0x7bff);
    for (auto & block : blocks) {
        block.d = static_cast<std::uint16_t>(d(rng));
        for (auto & x : block.qs) {
            x = static_cast<std::int8_t>(q(rng));
        }
    }
}

template <class Fn>
double time_ns_per_tile(Fn fn,
                        const std::vector<q8rn2::block_q8_0> & a,
                        const std::vector<q8rn2::block_q8_0> & b,
                        std::size_t depth,
                        int iterations,
                        volatile float & sink) {
    const auto t0 = clock_type::now();
    for (int i = 0; i < iterations; ++i) {
        const auto out = fn(a.data(), b.data(), depth);
        sink += out[static_cast<std::size_t>(i) & 7u];
    }
    const auto t1 = clock_type::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iterations;
}

void run_depth(std::size_t depth) {
    std::mt19937 rng(static_cast<std::uint32_t>(0x8c2026u + depth));
    std::vector<q8rn2::block_q8_0> a(4 * depth);
    std::vector<q8rn2::block_q8_0> b(2 * depth);
    fill_blocks(a, rng);
    fill_blocks(b, rng);

    volatile float sink = 0.0f;
    constexpr int warmup = 2000;
    const int iterations = depth == 28 ? 60000 : 16000;
    const auto baseline = q8rn2::tile_baseline;
    const auto optimized = q8rn2::tile_optimized;
    for (int i = 0; i < warmup; ++i) {
        sink += baseline(a.data(), b.data(), depth)[0];
        sink += optimized(a.data(), b.data(), depth)[0];
    }

    std::vector<double> base_ns;
    std::vector<double> opt_ns;
    base_ns.reserve(11);
    opt_ns.reserve(11);
    for (int rep = 0; rep < 11; ++rep) {
        if ((rep & 1) == 0) {
            base_ns.push_back(time_ns_per_tile(baseline, a, b, depth, iterations, sink));
            opt_ns.push_back(time_ns_per_tile(optimized, a, b, depth, iterations, sink));
        } else {
            opt_ns.push_back(time_ns_per_tile(optimized, a, b, depth, iterations, sink));
            base_ns.push_back(time_ns_per_tile(baseline, a, b, depth, iterations, sink));
        }
    }
    std::sort(base_ns.begin(), base_ns.end());
    std::sort(opt_ns.begin(), opt_ns.end());

    const auto cv_percent = [](const std::vector<double> & xs) {
        double mean = 0.0;
        for (double x : xs) mean += x;
        mean /= xs.size();
        double var = 0.0;
        for (double x : xs) var += (x - mean) * (x - mean);
        return std::sqrt(var / (xs.size() - 1)) / mean * 100.0;
    };

    const double base = base_ns[base_ns.size() / 2];
    const double opt = opt_ns[opt_ns.size() / 2];
    std::cout << "depth=" << depth
              << " baseline_median_ns=" << std::fixed << std::setprecision(2) << base
              << " optimized_median_ns=" << opt
              << " speedup=" << base / opt
              << " baseline_cv_percent=" << cv_percent(base_ns)
              << " optimized_cv_percent=" << cv_percent(opt_ns)
              << " checksum=" << sink << '\n';
}

}  // namespace

int main() {
    run_depth(28);
    run_depth(152);
    return 0;
}
