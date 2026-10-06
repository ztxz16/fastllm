#pragma once
#include "gguf.h"
#include <cassert>
#include <cstring>
#include <type_traits>

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
inline uint32_t iq3_xxs_metadata(const block_iq3_xxs &x, int g) {
    uint32_t bits;
    std::memcpy(&bits, x.qs + QK_K / 4 + 4 * g, sizeof(bits));
    return bits;
}
template<> inline int scale(const block_iq3_xxs &x, int g) {
    return 1 + 2 * (iq3_xxs_metadata(x, g) >> 28);
}
// IQ2_S has independent scales for the two halves of each 32-value group.
inline int iq2_s_scale(const block_iq2_s &x, int g, int half) {
    return 1 + 2 * ((x.scales[g] >> (4 * half)) & 15);
}
template<class Block> constexpr float block_scale() { return 1.f; }
template<> constexpr float block_scale<block_iq2_s>() { return .125f; }
template<> constexpr float block_scale<block_iq3_xxs>() { return .25f; }

#if defined(__AVX2__)
inline __m256i apply_signs(__m256i values, uint32_t bits) {
    const __m256i signs = _mm256_shuffle_epi8(_mm256_set1_epi32(bits),
        _mm256_setr_epi64x(0, 0x0101010101010101LL, 0x0202020202020202LL, 0x0303030303030303LL));
    const __m256i mask = _mm256_set1_epi64x(0x8040201008040201ULL);
    const __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(signs, mask), mask);
    return _mm256_sign_epi8(values, _mm256_or_si256(neg, _mm256_set1_epi8(1)));
}
inline __m256i unpack(const block_iq2_s &x, int g) {
    const uint8_t *q = x.qs + 4 * g;
    const unsigned h = x.qh[g];
    const __m256i values = _mm256_setr_epi64x(
        iq2s_grid[q[0] | ((h << 8) & 0x300)], iq2s_grid[q[1] | ((h << 6) & 0x300)],
        iq2s_grid[q[2] | ((h << 4) & 0x300)], iq2s_grid[q[3] | ((h << 2) & 0x300)]);
    uint32_t signs;
    std::memcpy(&signs, x.qs + QK_K / 8 + 4 * g, sizeof(signs));
    return apply_signs(values, signs);
}
inline __m256i unpack(const block_iq3_xxs &x, int g) {
    const uint8_t *q = x.qs + 8 * g;
    const __m256i values = _mm256_setr_epi32(
        iq3xxs_grid[q[0]], iq3xxs_grid[q[1]], iq3xxs_grid[q[2]], iq3xxs_grid[q[3]],
        iq3xxs_grid[q[4]], iq3xxs_grid[q[5]], iq3xxs_grid[q[6]], iq3xxs_grid[q[7]]);
    const uint32_t bits = iq3_xxs_metadata(x, g);
    const uint32_t signs = uint32_t(ksigns_iq2xs[bits & 127]) |
        (uint32_t(ksigns_iq2xs[(bits >> 7) & 127]) << 8) |
        (uint32_t(ksigns_iq2xs[(bits >> 14) & 127]) << 16) |
        (uint32_t(ksigns_iq2xs[(bits >> 21) & 127]) << 24);
    return apply_signs(values, signs);
}
inline __m256i unpack(const block_iq3_s &x, int g) {
    // Construct all eight 9-bit codebook indices together. Scalar shifts
    // for each index and 64-bit sign broadcasts dominate small CPU dots.
    const __m256i lo = _mm256_cvtepu8_epi32(
        _mm_loadl_epi64((const __m128i *)(x.qs + 8 * g)));
    const __m256i hi = _mm256_and_si256(
        _mm256_sllv_epi32(_mm256_set1_epi32(x.qh[g]),
                         _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1)),
        _mm256_set1_epi32(256));
    alignas(32) uint32_t index[8];
    _mm256_store_si256((__m256i *)index, _mm256_or_si256(lo, hi));
    // Scalar table loads avoid the expensive gather on older AVX2 CPUs.
    const __m256i values = _mm256_setr_epi32(
        iq3s_grid[index[0]], iq3s_grid[index[1]], iq3s_grid[index[2]], iq3s_grid[index[3]],
        iq3s_grid[index[4]], iq3s_grid[index[5]], iq3s_grid[index[6]], iq3s_grid[index[7]]);
    uint32_t bits;
    std::memcpy(&bits, x.signs + 4 * g, sizeof(bits));
    const __m256i signs = _mm256_shuffle_epi8(_mm256_set1_epi32(bits),
        _mm256_setr_epi64x(0, 0x0101010101010101LL, 0x0202020202020202LL, 0x0303030303030303LL));
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
inline int value(const block_iq2_s &x, int g, int j) {
    const int lane = j / 8;
    const int index = x.qs[4 * g + lane] | ((x.qh[g] << (8 - 2 * lane)) & 0x300);
    const int v = (iq2s_grid[index] >> (8 * (j % 8))) & 255;
    return x.qs[QK_K / 8 + 4 * g + lane] & (1 << (j % 8)) ? -v : v;
}
inline int value(const block_iq3_xxs &x, int g, int j) {
    const int v = (iq3xxs_grid[x.qs[8 * g + j / 4]] >> (8 * (j % 4))) & 255;
    const auto signs = ksigns_iq2xs[(iq3_xxs_metadata(x, g) >> (7 * (j / 8))) & 127];
    return signs & (1 << (j % 8)) ? -v : v;
}
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
            __m256i scales;
            if constexpr (std::is_same<Block, block_iq2_s>::value) {
                scales = _mm256_insertf128_si256(_mm256_castsi128_si256(
                    _mm_set1_epi16(iq2_s_scale(x[b], g, 0))),
                    _mm_set1_epi16(iq2_s_scale(x[b], g, 1)), 1);
            } else scales = _mm256_set1_epi16(scale(x[b], g));
            total = _mm256_add_epi32(total, _mm256_madd_epi16(pair, scales));
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
    return block_scale<Block>() * _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum)));
#else
    float result = 0;
    for (int b = 0; b < n / QK_K; ++b) {
        int total = 0;
        for (int g = 0; g < 8; ++g) {
            for (int j = 0; j < 32; ++j) {
                int s;
                if constexpr (std::is_same<Block, block_iq2_s>::value) s = iq2_s_scale(x[b], g, j / 16);
                else s = scale(x[b], g);
                total += s * value(x[b], g, j) * int(y[b].qs[g * 32 + j]);
            }
        }
        result += (GGML_FP16_TO_FP32(x[b].d) * y[b].d) * total;
    }
    return block_scale<Block>() * result;
#endif
}
} // namespace iq_native
