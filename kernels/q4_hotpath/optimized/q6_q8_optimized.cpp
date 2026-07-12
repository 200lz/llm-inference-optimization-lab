#include "q6_q8.h"
#include <immintrin.h>

namespace q4_hotpath {
float optimized(const block_q6_k * x, const block_q8_k * y, std::size_t blocks) {
    return optimized_avx2(x, y, blocks);
}
}
