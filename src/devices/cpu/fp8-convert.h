#pragma once
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __aarch64__
#include <arm_neon.h>
#endif
namespace fastllm {
#ifdef __AVX2__
    static inline __m256 FP8E4M3ToFloat_AVX2(__m128i bytes) {
        const __m256i words = _mm256_cvtepu8_epi32(bytes);
        const __m256i magnitude = _mm256_and_si256(words, _mm256_set1_epi32(0x7f));
        const __m256i sign = _mm256_slli_epi32(
            _mm256_and_si256(words, _mm256_set1_epi32(0x80)), 24);
        const __m256 normal = _mm256_castsi256_ps(_mm256_add_epi32(
            _mm256_slli_epi32(magnitude, 20), _mm256_set1_epi32(120 << 23)));
        const __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(magnitude), _mm256_set1_ps(1.0f / 512));
        const __m256 tiny = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(8), magnitude));
        return _mm256_or_ps(_mm256_blendv_ps(normal, subnormal, tiny), _mm256_castsi256_ps(sign));
    }
#endif
#ifdef __aarch64__
    static inline float32x4_t FP8E4M3ToFloat_NEON(uint32x4_t bytes) {
        const uint32x4_t magnitude = vandq_u32(bytes, vdupq_n_u32(0x7f));
        const uint32x4_t sign = vshlq_n_u32(vandq_u32(bytes, vdupq_n_u32(0x80)), 24);
        const uint32x4_t normal = vaddq_u32(vshlq_n_u32(magnitude, 20), vdupq_n_u32(120u << 23));
        const float32x4_t tiny = vmulq_n_f32(vcvtq_f32_u32(magnitude), 1.0f / 512);
        return vreinterpretq_f32_u32(vorrq_u32(sign, vbslq_u32(vcltq_u32(magnitude, vdupq_n_u32(8)),
                                                               vreinterpretq_u32_f32(tiny), normal)));
    }
#endif
}
