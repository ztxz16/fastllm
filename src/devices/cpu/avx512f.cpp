//
// Created by huangyuyang on 8/14/25.
//

#include <cstdint>

#ifdef __AVX2__
#include "immintrin.h"
#endif

#include "utils.h"

namespace fastllm {
    void AddBiasAVX512(float *outputData, float *biasData, int n, int k, int st, int end) {
#ifdef __AVX512F__
        if (biasData) {
            for (int i = 0; i < n; i++) {
                int j = st;
                for (; j + 15 < end; j += 16) {                    
                    _mm512_storeu_ps(outputData + i * k + j, 
                        _mm512_add_ps(_mm512_loadu_ps(outputData + i * k + j), _mm512_loadu_ps(biasData + j)));
                }
                for (; j < end; j++) {
                    outputData[i * k + j] += biasData[j];
                }
            }
        }
#endif
    }

    template <int BROW, int AROW>
    void mul_mat_f16_f32_direct_avx512(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const float* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        constexpr int SIMD_WIDTH = 16;  // AVX512 一次处理 16 个 float
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;

        // 累加器 - 注意这里的顺序
        __m512 acc[AROW * BROW];
        
        // 初始化
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm512_setzero_ps();
        }
        
        // 主循环
        for (int i = 0; i < nb; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                
                // 从 f16 转换到 f32 (AVX512)
                __m256i a_f16 = _mm256_loadu_si256((const __m256i*)(a_row + i * SIMD_WIDTH));
                __m512 a_vec = _mm512_cvtph_ps(a_f16);
                
                for (int iy = 0; iy < BROW; ++iy) {
                    const float* b_row = (const float*)((const char*)B + iy * stride_b);
                    __m512 b_vec = _mm512_loadu_ps(b_row + i * SIMD_WIDTH);
                    
                    int acc_idx = ix * BROW + iy;
                    acc[acc_idx] = _mm512_fmadd_ps(a_vec, b_vec, acc[acc_idx]);
                }
            }
        }
        
        // 处理剩余元素
        if (remainder > 0) {
            __mmask16 mask = (1 << remainder) - 1;
            
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                
                // 使用掩码加载
                __m256i a_f16 = _mm256_maskz_loadu_epi16(mask, a_row + nb * SIMD_WIDTH);
                __m512 a_vec = _mm512_cvtph_ps(a_f16);
                
                for (int iy = 0; iy < BROW; ++iy) {
                    const float* b_row = (const float*)((const char*)B + iy * stride_b);
                    __m512 b_vec = _mm512_maskz_loadu_ps(mask, b_row + nb * SIMD_WIDTH);
                    
                    int acc_idx = ix * BROW + iy;
                    acc[acc_idx] = _mm512_fmadd_ps(a_vec, b_vec, acc[acc_idx]);
                }
            }
        }
        
        // 水平求和并存储
        for (int ix = 0; ix < AROW; ++ix) {
            for (int iy = 0; iy < BROW; ++iy) {
                int acc_idx = ix * BROW + iy;
                float result = _mm512_reduce_add_ps(acc[acc_idx]);
                float* c_row = (float*)((char*)C + iy * stride_c);
                c_row[ix] = result;
            }
        }
#endif
    }

    template <int BRow>
    void LinearFloat32Float16_AVX512F_Row_Kernel(float *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int i, int m, int k, int st, int end) {
        int j = st;
        for (j = st; j + 4 < end; j += 5) {
            mul_mat_f16_f32_direct_avx512 <BRow, 5> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(float), outputData + i * k + j, k * sizeof(float));
        }
        switch (end - j) {
            case 0: break;
            case 1: mul_mat_f16_f32_direct_avx512 <BRow, 1> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(float), outputData + i * k + j, k * sizeof(float)); break;
            case 2: mul_mat_f16_f32_direct_avx512 <BRow, 2> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(float), outputData + i * k + j, k * sizeof(float)); break;
            case 3: mul_mat_f16_f32_direct_avx512 <BRow, 3> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(float), outputData + i * k + j, k * sizeof(float)); break;
            case 4: mul_mat_f16_f32_direct_avx512 <BRow, 4> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(float), outputData + i * k + j, k * sizeof(float)); break;
        }
    }

    bool LinearFloat32Float16_AVX512F_Kernel(float *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        int i = 0;
        for (; i + 4 < n; i += 5) {
            LinearFloat32Float16_AVX512F_Row_Kernel <5> (inputData, weightData, biasData, outputData, i, m, k, st, end);
        }
        switch (n - i) {
            case 0: break;
            case 1: LinearFloat32Float16_AVX512F_Row_Kernel <1> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 2: LinearFloat32Float16_AVX512F_Row_Kernel <2> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 3: LinearFloat32Float16_AVX512F_Row_Kernel <3> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 4: LinearFloat32Float16_AVX512F_Row_Kernel <4> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
        }
        AddBiasAVX512(outputData, biasData, n, k, st, end);
        return true;
    }
}