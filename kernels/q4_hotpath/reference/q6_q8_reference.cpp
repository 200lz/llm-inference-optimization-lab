#include "q6_q8.h"
#include <cstring>
#include <immintrin.h>

namespace q4_hotpath {
float fp16_to_fp32(uint16_t value) { return _cvtsh_ss(value); }

int decode_q6(const block_q6_k & b, int k) {
    const int group = k / 128, lane = k % 128;
    const int chunk = lane / 32, i = lane % 32;
    const int low_index = group * 64 + (chunk & 1) * 32 + i;
    const int low = chunk < 2 ? (b.ql[low_index] & 15) : (b.ql[low_index] >> 4);
    const int high = (b.qh[group * 32 + i] >> (2 * chunk)) & 3;
    return low | (high << 4);
}

float reference(const block_q6_k * x, const block_q8_k * y, std::size_t blocks) {
    float lanes[8]{};
    for (std::size_t b = 0; b < blocks; ++b) {
        int32_t sums[8]{};
        for (int k = 0; k < qk; ++k) {
            const int q6 = decode_q6(x[b], k) - 32;
            sums[k & 7] += q6 * int(y[b].qs[k]) * int(x[b].scales[k / 16]);
        }
        const float d = fp16_to_fp32(x[b].d) * y[b].d;
        for (int lane = 0; lane < 8; ++lane) lanes[lane] += d * float(sums[lane]);
    }
    float result = 0;
    for (float lane : lanes) result += lane;
    return result;
}
}
