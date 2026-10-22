#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
namespace {
// Freeze the pre-optimization AVX2 arithmetic as a bitwise regression oracle.
static const fastllm::FP16ToFP32Manager halfTable;
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}
static void LegacyDot(int nb, float *s, const block_iq1_s *x, const block_q8_K *y) {

    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)],
                                                    iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)],
                                                    iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
            qs += 8;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * halfTable.dict[x[i].d];
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;

    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

}
static void CurrentDot(int nb, float *s, const block_iq1_s *x, const block_q8_K *y) {
    ggml_vec_dot_iq1_s_q8_K(nb*QK_K, s, 0, x, 0, y, 0, 1);
}
static void Fill(std::vector<block_iq1_s> &x, std::vector<block_q8_K> &y, std::mt19937 &rng, bool extreme) {
    const uint16_t scales[] = {0, 0x2800, 0xb000, 0x2c80, 0xa400, 0x0001, 0x8155};
    for (size_t i=0; i<x.size(); ++i) {
        x[i].d = scales[i%7];
        for (auto &q : x[i].qs) q = rng();
        for (auto &q : x[i].qh) q = rng();
    }
    for (auto &v : y) {
        v.d = (int(rng()%15)-7)/128.f;
        v.sum = 0;
        for (int i=0; i<QK_K; ++i) {
            v.qs[i] = int(rng()%255)-127;
            if (i%4 == 0) v.qs[i] = extreme ? -128 : -127;
            if (i%4 == 1) v.qs[i] = 127;
            if (i%4 == 2) v.qs[i] = 0;
            v.sum += v.qs[i];
        }
        for (int i=0; i<QK_K/16; ++i) {
            v.bsums[i] = 0;
            for (int j=0; j<16; ++j) v.bsums[i] += v.qs[i*16+j];
        }
    }
}
static void Verify(int nb, const block_iq1_s *x, const block_q8_K *y, bool scalar) {
    float legacy, guard[] = {123.f, 0.f, -456.f};
    // Keep the frozen oracle out of line: inlining it can change scalar
    // multiply-add contraction and defeat a bitwise comparison at -O3.
    auto volatile referenceDot = &LegacyDot;
    referenceDot(nb, &legacy, x, y);
    CurrentDot(nb, guard+1, x, y);
    if (guard[0] != 123.f || guard[2] != -456.f || std::memcmp(&legacy, guard+1, sizeof(float)))
        throw std::runtime_error("IQ1_S AVX2 bits or output boundary changed");
    if (!scalar) return;
    std::vector<float> decoded(nb*QK_K);
    dequantize_row_iq1_s(x, decoded.data(), decoded.size());
    double reference=0, magnitude=0;
    for (int i=0; i<nb*QK_K; ++i) {
        double p=double(decoded[i])*y[i/QK_K].d*y[i/QK_K].qs[i%QK_K];
        reference+=p; magnitude+=std::abs(p);
    }
    if (!std::isfinite(guard[1]) || std::abs(guard[1]-reference)>2e-5+3e-6*magnitude)
        throw std::runtime_error("IQ1_S differs from dequantized FP64 reference");
}
static void Test() {
    std::mt19937 rng(8742);
    std::vector<block_iq1_s> x(1); std::vector<block_q8_K> y(1);
    Fill(x,y,rng,false); x[0].d=0x2800; y[0].d=.03125f;
    // All index high bits, scales and signs; cycle low indices too.
    for (int meta=0; meta<65536; ++meta) {
        for (int j=0; j<8; ++j) x[0].qh[j]=uint16_t(meta+8191*j);
        for (int j=0; j<32; ++j) x[0].qs[j]=uint8_t(meta+31*j);
        Verify(1,x.data(),y.data(),meta%257==0);
    }
    for (int cols : {256,512,768,2304,5120}) for (bool extreme : {false,true}) {
        int nb=cols/QK_K;
        x.resize(nb*257); y.resize(nb); Fill(x,y,rng,extreme);
        for (int row=0; row<257; ++row) Verify(nb,x.data()+row*nb,y.data(),!extreme);
    }
    puts("PASS: IQ1_S AVX2 legacy bits and dequantized reference");
}

}
#endif
int main() {
#if defined(__AVX2__)
    try {
        Test();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1;
    }
#else
    return 77;
#endif
}
