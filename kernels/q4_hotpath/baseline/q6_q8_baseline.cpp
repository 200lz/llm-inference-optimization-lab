#include "q6_q8.h"
#include <immintrin.h>

namespace q4_hotpath {
namespace {
inline __m128i scale_shuffle(int i) {
    alignas(16) static const uint8_t masks[8][16] = {
        {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1}, {2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3},
        {4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5}, {6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7},
        {8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9}, {10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11},
        {12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13}, {14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15}};
    return _mm_load_si128(reinterpret_cast<const __m128i *>(masks[i]));
}
}
template <bool split> float avx2_kernel(const block_q6_k * x, const block_q8_k * y, std::size_t nb) {
    const __m256i m3 = _mm256_set1_epi8(3), m15 = _mm256_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    for (std::size_t i = 0; i < nb; ++i) {
        const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[i].scales));
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales);
        const __m256i bsums = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[i].bsums));
        const __m256i sub = _mm256_slli_epi32(_mm256_madd_epi16(bsums, scales16), 5);
        __m256i sumi = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        const uint8_t * q4=x[i].ql, *qh=x[i].qh; const int8_t * q8=y[i].qs;
        int is=0;
        for (int j=0;j<2;++j) {
            const __m256i l0=_mm256_loadu_si256((const __m256i*)q4); q4+=32;
            const __m256i l1=_mm256_loadu_si256((const __m256i*)q4); q4+=32;
            const __m256i h=_mm256_loadu_si256((const __m256i*)qh); qh+=32;
            __m256i v[4]={
                _mm256_or_si256(_mm256_and_si256(l0,m15),_mm256_slli_epi16(_mm256_and_si256(h,m3),4)),
                _mm256_or_si256(_mm256_and_si256(l1,m15),_mm256_slli_epi16(_mm256_and_si256(h,_mm256_set1_epi8(12)),2)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0,4),m15),_mm256_and_si256(h,_mm256_set1_epi8(48))),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1,4),m15),_mm256_srli_epi16(_mm256_and_si256(h,_mm256_set1_epi8(-64)),2))};
            __m256i p[4];
            for(int k=0;k<4;++k){p[k]=_mm256_maddubs_epi16(v[k],_mm256_loadu_si256((const __m256i*)q8));q8+=32;p[k]=_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales,scale_shuffle(is+k))),p[k]);}
            is+=4;
            sumi=_mm256_add_epi32(sumi,_mm256_add_epi32(p[0],p[1]));
            if constexpr (split) sumi2=_mm256_add_epi32(sumi2,_mm256_add_epi32(p[2],p[3]));
            else sumi=_mm256_add_epi32(sumi,_mm256_add_epi32(p[2],p[3]));
        }
        if constexpr (split) sumi=_mm256_add_epi32(sumi,sumi2);
        sumi=_mm256_sub_epi32(sumi,sub);
        const float d=y[i].d*fp16_to_fp32(x[i].d);
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(sumi),acc);
    }
    __m128 z=_mm_add_ps(_mm256_castps256_ps128(acc),_mm256_extractf128_ps(acc,1)); z=_mm_add_ps(z,_mm_movehl_ps(z,z)); z=_mm_add_ss(z,_mm_movehdup_ps(z)); return _mm_cvtss_f32(z);
}
float baseline(const block_q6_k * x, const block_q8_k * y, std::size_t nb) { return avx2_kernel<false>(x,y,nb); }
float optimized_avx2(const block_q6_k * x, const block_q8_k * y, std::size_t nb) { return avx2_kernel<true>(x,y,nb); }
}
