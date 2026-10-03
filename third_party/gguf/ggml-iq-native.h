#pragma once
#include "gguf.h"
#include <cassert>

extern float GGML_FP16_TO_FP32(ggml_half f);

// Ordinary IQ records: unpack only 32 integer values at a time. Keep the
// weights packed in RAM and share Q8_K activations with the NUMA MoE path.
namespace iq_native {
static constexpr int8_t iq4_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

template<class Block> inline int scale(const Block &x, int group);
template<> inline int scale(const block_iq3_s &x, int g) {
    return 1 + 2 * ((x.scales[g / 2] >> (4 * (g % 2))) & 15);
}
template<> inline int scale(const block_iq4_xs &x, int g) {
    return (((x.scales_l[g / 2] >> (4 * (g % 2))) & 15) |
            (((x.scales_h >> (2 * g)) & 3) << 4)) - 32;
}

#if defined(__AVX2__)
inline __m256i unpack(const block_iq3_s &x, int g) {
    const uint8_t *q = x.qs + 8 * g;
    const int hi = x.qh[g];
    const __m256i values = _mm256_setr_epi32(
        iq3s_grid[q[0] | ((hi << 8) & 256)], iq3s_grid[q[1] | ((hi << 7) & 256)],
        iq3s_grid[q[2] | ((hi << 6) & 256)], iq3s_grid[q[3] | ((hi << 5) & 256)],
        iq3s_grid[q[4] | ((hi << 4) & 256)], iq3s_grid[q[5] | ((hi << 3) & 256)],
        iq3s_grid[q[6] | ((hi << 2) & 256)], iq3s_grid[q[7] | ((hi << 1) & 256)]);
    const uint8_t *s = x.signs + 4 * g;
    const uint64_t repeat = 0x0101010101010101ULL;
    const __m256i signs = _mm256_setr_epi64x(s[0] * repeat, s[1] * repeat, s[2] * repeat, s[3] * repeat);
    const __m256i mask = _mm256_set1_epi64x(0x8040201008040201ULL);
    const __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(signs, mask), mask);
    return _mm256_sign_epi8(values, _mm256_or_si256(neg, _mm256_set1_epi8(1)));
}
inline __m256i unpack(const block_iq4_xs &x, int g) {
    const __m128i values = _mm_loadu_si128((const __m128i *)iq4_values);
    const __m128i bits = _mm_loadu_si128((const __m128i *)(x.qs + 16 * g));
    const __m128i mask = _mm_set1_epi8(15);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(
        _mm_shuffle_epi8(values, _mm_and_si128(bits, mask))),
        _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), mask)), 1);
}
#else
inline int value(const block_iq3_s &x, int g, int j) {
    const int i = j / 4;
    const int index = x.qs[8 * g + i] | (((x.qh[g] >> i) & 1) << 8);
    const int v = (iq3s_grid[index] >> (8 * (j % 4))) & 255;
    return x.signs[4 * g + j / 8] & (1 << (j % 8)) ? -v : v;
}
inline int value(const block_iq4_xs &x, int g, int j) {
    return iq4_values[(x.qs[16 * g + j % 16] >> (4 * (j / 16))) & 15];
}
#endif

template<class Block> inline float dot(int n, const Block *x, const block_q8_K *y) {
    assert(n % QK_K == 0);
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / QK_K; ++b) {
        __m256i total = _mm256_setzero_si256();
        for (int g = 0; g < 8; ++g) {
            const __m256i qx = unpack(x[b], g);
            const __m256i qy = _mm256_loadu_si256((const __m256i *)(y[b].qs + 32 * g));
            // Weight values never equal -128; applying Q8's sign to the
            // weight also handles the valid signed Q8 endpoint -128.
            const __m256i pair = _mm256_maddubs_epi16(_mm256_abs_epi8(qy), _mm256_sign_epi8(qx, qy));
            // The group scale fits in int16. Apply it in the widening
            // multiply-add instead of a second int32 vector multiply.
            total = _mm256_add_epi32(total,
                _mm256_madd_epi16(pair, _mm256_set1_epi16(scale(x[b], g))));
        }
#if defined(__F16C__)
        // Avoid spilling the accumulators around an external converter for
        // every block. Keep the table converter on CPUs without F16C.
        const float dx = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[b].d)));
        const __m256 d = _mm256_set1_ps(dx * y[b].d);
#else
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[b].d) * y[b].d);
#endif
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, _mm256_cvtepi32_ps(total)));
    }
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    return _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum)));
#else
    float result = 0;
    for (int b = 0; b < n / QK_K; ++b) {
        int total = 0;
        for (int g = 0; g < 8; ++g) {
            int sum = 0;
            for (int j = 0; j < 32; ++j) sum += value(x[b], g, j) * int(y[b].qs[g * 32 + j]);
            total += scale(x[b], g) * sum;
        }
        result += (GGML_FP16_TO_FP32(x[b].d) * y[b].d) * total;
    }
    return result;
#endif
}
} // namespace iq_native
