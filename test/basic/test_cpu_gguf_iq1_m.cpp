#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
namespace {
static const fastllm::FP16ToFP32Manager halfTable;
static inline float hsum_float_8(__m256 x) {
    __m128 a = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    return _mm_cvtss_f32(_mm_add_ss(a, _mm_movehdup_ps(a)));
}
static inline __m256i mul_add_epi8(__m256i x, __m256i y) {
    return _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
}
// Freeze the previous AVX2 arithmetic, including signed-byte wrapping,
// delta correction and floating-point accumulation order.
static void LegacyDot(int nb, float *s, const block_iq1_m *x, const block_q8_K *y) {
    iq1m_scale_t scale;


    const __m256i mask = _mm256_set1_epi16(0x7);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m256i q1b_1 = _mm256_set_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)],
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]
            );
            const __m256i q1b_2 = _mm256_set_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)],
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]
            );
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);

            const __m256i delta1 = _mm256_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m256i delta2 = _mm256_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);

            const __m256i dot3 = mul_add_epi8(delta1, q8b_1);
            const __m256i dot4 = mul_add_epi8(delta2, q8b_2);

            __m256i scale1 = _mm256_set_m128i(_mm_set1_epi16(sc[ib/2] >> 3), _mm_set1_epi16(sc[ib/2] >> 0));
            __m256i scale2 = _mm256_set_m128i(_mm_set1_epi16(sc[ib/2] >> 9), _mm_set1_epi16(sc[ib/2] >> 6));

            scale1 = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scale1, mask), 1), mone);
            scale2 = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scale2, mask), 1), mone);
            const __m256i p1 = _mm256_madd_epi16(dot1, scale1);
            const __m256i p2 = _mm256_madd_epi16(dot2, scale2);
            const __m256i p3 = _mm256_madd_epi16(dot3, scale1);
            const __m256i p4 = _mm256_madd_epi16(dot4, scale2);

            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p1, p2));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p3, p4));

            qs += 8; qh += 4;
        }

        const __m256 d = _mm256_set1_ps(y[i].d * halfTable.dict[scale.f16]);

        accum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1), accum1);
        accum2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2), accum2);
    }

    *s = hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2);


}
static void Verify(int nb, const block_iq1_m *x, const block_q8_K *y, bool scalar) {
    float reference, guard[] = {123.f, 0.f, -456.f};
    auto volatile legacy = &LegacyDot;
    legacy(nb, &reference, x, y);
    ggml_vec_dot_iq1_m_q8_K(nb * QK_K, guard + 1, 0, x, 0, y, 0, 1);
    if (guard[0] != 123.f || guard[2] != -456.f || memcmp(&reference, guard + 1, sizeof(float)))
        throw std::runtime_error("IQ1_M AVX2 bits or output boundary changed");
    if (!scalar) return;
    std::vector<float> plain(nb * QK_K);
    dequantize_row_iq1_m(x, plain.data(), plain.size());
    double sum = 0, magnitude = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
        double product = double(plain[i]) * y[i / QK_K].d * y[i / QK_K].qs[i % QK_K];
        sum += product; magnitude += std::abs(product);
    }
    if (!std::isfinite(guard[1]) || std::abs(guard[1] - sum) > 2e-5 + 3e-6 * magnitude)
        throw std::runtime_error("IQ1_M differs from dequantized FP64 reference");
}
static void Fill(std::vector<block_iq1_m> &x, std::vector<block_q8_K> &y,
                 std::mt19937 &rng, bool extreme) {
    const uint16_t scales[] = {0, 0x2800, 0xb000, 0x2c80, 0xa400, 0x0001, 0x8155};
    for (size_t i = 0; i < x.size(); ++i) {
        for (auto &q : x[i].qs) q = rng();
        for (auto &q : x[i].qh) q = rng();
        uint16_t words[4];
        for (int j = 0; j < 4; ++j)
            words[j] = (rng() & 0xfff) | (((scales[i % 7] >> (j * 4)) & 15) << 12);
        memcpy(x[i].scales, words, sizeof(words));
    }
    for (auto &v : y) {
        v.d = (int(rng() % 15) - 7) / 128.f; v.sum = 0;
        for (int i = 0; i < QK_K; ++i) {
            v.qs[i] = int(rng() % 255) - 127;
            if (i % 4 == 0) v.qs[i] = extreme ? -128 : -127;
            if (i % 4 == 1) v.qs[i] = 127;
            if (i % 4 == 2) v.qs[i] = 0;
            v.sum += v.qs[i];
        }
        for (int i = 0; i < QK_K / 16; ++i) {
            v.bsums[i] = 0;
            for (int j = 0; j < 16; ++j) v.bsums[i] += v.qs[16 * i + j];
        }
    }
}
static void Test() {
    std::mt19937 rng(7241);
    std::vector<block_iq1_m> x(1); std::vector<block_q8_K> y(1);
    Fill(x, y, rng, false);
    uint16_t scales[] = {0x0000, 0x0fff, 0x0a53, 0x0555};
    scales[2] |= 0x8000; scales[3] |= 0x2000; // global FP16 0x2800
    memcpy(x[0].scales, scales, sizeof(scales)); y[0].d = .03125f;
    // All paired metadata bytes cover every suffix and all four delta signs.
    for (int meta = 0; meta < 65536; ++meta) {
        for (int j = 0; j < 16; j += 2) {
            uint16_t value = meta + j * 8179;
            memcpy(x[0].qh + j, &value, sizeof(value));
        }
        for (int j = 0; j < 32; ++j) x[0].qs[j] = meta + j * 31;
        Verify(1, x.data(), y.data(), meta % 257 == 0);
    }
    for (int columns : {256, 512, 768, 2304, 5120}) for (bool extreme : {false, true}) {
        int nb = columns / QK_K;
        x.resize(nb * 257); y.resize(nb); Fill(x, y, rng, extreme);
        for (int row = 0; row < 257; ++row) Verify(nb, x.data() + row * nb, y.data(), !extreme);
    }
    // Independent activation pointers, odd tails, zero scales and -128 bytes.
    for (int cols : {256, 768, 2304, 5120}) for (bool extreme : {false, true}) {
        const int nb = cols / QK_K;
        x.resize(nb * 31); y.resize(nb * 8); Fill(x, y, rng, extreme);
        for (int rows = 1; rows <= 8; ++rows) for (int w = 0; w < 31; ++w) {
            const void *inputs[8];
            float actual[10], expected[8];
            for (float &v : actual) v = -12345.f;
            for (int r = 0; r < rows; ++r) {
                inputs[r] = y.data() + ((r * 3) % 8) * nb;
                auto volatile dot = &LegacyDot;
                dot(nb, &expected[r], x.data() + w * nb,
                    static_cast<const block_q8_K *>(inputs[r]));
            }
            ggml_vec_dot_iq1_q8_K_batch(GGML_TYPE_IQ1_M, cols,
                x.data() + w * nb, inputs, actual + 1, rows);
            if (actual[0] != -12345.f || actual[rows + 1] != -12345.f ||
                memcmp(expected, actual + 1, rows * sizeof(float)))
                throw std::runtime_error("IQ1 batched dot differs from frozen single-row oracle");
        }
    }
    puts("PASS: IQ1_M AVX2 legacy bits and dequantized reference");
}
}
#endif
int main() {
#if defined(__AVX2__)
    try { Test(); return 0; }
    catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
#else
    return 77;
#endif
}
