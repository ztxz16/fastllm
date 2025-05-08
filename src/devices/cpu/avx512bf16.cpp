//
// Created by huangyuyang on 5/8/25.
//

#include <cstdint>
#include "immintrin.h"

namespace fastllm {
    bool LinearBFloat16FP8E4M3_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, float *scales, 
                        int ks, int ms, float magicScale) {
        if (!(m % blockM == 0 && blockM % 32 == 0)) {
            return false;
        }
#ifdef __AVX512BF16__
        for (int i = 0; i < n; i++) {
            int j = st;
            __m256i v_a_mask_byte = _mm256_set1_epi8(0x80); 
            __m256i v_b_mask_byte = _mm256_set1_epi8(0x7F); 
            for (; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                __m512 last_sum = _mm512_setzero_ps(); // Accumulator for 16 parallel sums

                for (int midx = 0; midx < ms; midx++) {
                    float curScale = scales[j / blockK * ms + midx];
                    __m512 vScale = _mm512_set1_ps(curScale);

                    int l = midx * blockM;
                    __m512 v_sum = _mm512_setzero_ps(); // Accumulator for 16 parallel sums
                    for (; l + 31 < m && l + 31 < (midx + 1) * blockM; l += 32) {
                        // 1. Load 32 BF16 inputs
                        // Treat uint16_t* as __m512bh* - use loadu for unaligned access
                        __m512bh v_input_bf16 = (__m512bh)_mm512_loadu_si512((__m512i const*)(inputData + i * m + l));
                        // 2. Load 32 FP8 weights
                        __m256i va_bytes = _mm256_loadu_si256((__m256i*)&weightData[j * m + l]);

                        __m256i va_masked_bytes = _mm256_and_si256(va_bytes, v_a_mask_byte);
                        __m512i va_promoted_words = _mm512_cvtepu8_epi16(va_masked_bytes);
                        __m512i v_a_term_shifted = _mm512_slli_epi16(va_promoted_words, 8);

                        __m256i vb_masked_bytes = _mm256_and_si256(va_bytes, v_b_mask_byte);
                        __m512i vb_promoted_words = _mm512_cvtepu8_epi16(vb_masked_bytes);
                        __m512i v_b_term_shifted = _mm512_slli_epi16(vb_promoted_words, 4);

                        __m512i v_result = _mm512_or_si512(v_a_term_shifted, v_b_term_shifted);
                        __m512bh v_weights_bf16 = (__m512bh)v_result;
                        
                        // 3. Compute dot product: v_sum += v_input_bf16 * v_weights_bf16
                        v_sum = _mm512_dpbf16_ps(v_sum, v_input_bf16, v_weights_bf16);
                    }

                    last_sum = _mm512_fmadd_ps(v_sum, vScale, last_sum);
                }

                now += _mm512_reduce_add_ps(last_sum) * magicScale;
                outputData[i * k + j] = now;
            }
        }
        return true;
#endif
        return false;
    }
}