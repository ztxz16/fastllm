//
// Created by huangyuyang on 5/8/25.
//

#include <cstdint>

#ifdef __AVX2__
#include "immintrin.h"
#endif

#include "utils.h"
#include "fastllm.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

namespace fastllm {
    extern void AddBiasAVX512(float *outputData, float *biasData, int n, int k, int st, int end);
    
	template <int BROW, int AROW>
    void mul_mat_fp8e4m3_bf16_direct_avx512(
        int n,
        const uint8_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c,
        float *scales, 
        int stx, int blockK, int ms, int blockM, 
        float magicScale
    ) {
#ifdef __AVX512BF16__
        constexpr int SIMD_WIDTH = 32;  // AVX512 一次处理 16 个 float
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;

        // 累加器 - 注意这里的顺序
        __m512 acc[AROW * BROW];
        
        // 初始化
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm512_setzero_ps();
        }

        __m256i v_a_mask_byte = _mm256_set1_epi8(0x80); 
        __m256i v_b_mask_byte = _mm256_set1_epi8(0x7F); 
        
        // 主循环
        for (int i = 0; i < nb; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                float curScale = scales[(stx + ix) / blockK * ms + (i * SIMD_WIDTH) / blockM];
                __m512 vScale = _mm512_set1_ps(curScale);
                const uint8_t* a_row = (const uint8_t*)((const char*)A + ix * stride_a);
                
                // 从 fp8e4m3 转换到 bf16 (AVX512)
                __m256i va_bytes = _mm256_loadu_si256((const __m256i*)(a_row + i * SIMD_WIDTH));
                __m256i va_masked_bytes = _mm256_and_si256(va_bytes, v_a_mask_byte);
                __m512i va_promoted_words = _mm512_cvtepu8_epi16(va_masked_bytes);
                __m512i v_a_term_shifted = _mm512_slli_epi16(va_promoted_words, 8);

                __m256i vb_masked_bytes = _mm256_and_si256(va_bytes, v_b_mask_byte);
                __m512i vb_promoted_words = _mm512_cvtepu8_epi16(vb_masked_bytes);
                __m512i v_b_term_shifted = _mm512_slli_epi16(vb_promoted_words, 4);

                __m512i v_result = _mm512_or_si512(v_a_term_shifted, v_b_term_shifted);
                __m512bh v_weights_bf16 = (__m512bh)v_result;
                
                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    __m512bh v_input_bf16 = (__m512bh)_mm512_loadu_si512((__m512i const*)(b_row + i * SIMD_WIDTH));

                    // Compute dot product: v_sum += v_input_bf16 * v_weights_bf16
                    __m512 v_sum = _mm512_setzero_ps();
                    v_sum = _mm512_dpbf16_ps(v_sum, v_input_bf16, v_weights_bf16);
                    int acc_idx = ix * BROW + iy;
                    acc[acc_idx] = _mm512_fmadd_ps(v_sum, vScale, acc[acc_idx]);
                }
            }
        }

        // 水平求和并存储
        for (int ix = 0; ix < AROW; ++ix) {
            for (int iy = 0; iy < BROW; ++iy) {
                int acc_idx = ix * BROW + iy;
                float result = _mm512_reduce_add_ps(acc[acc_idx]);
                float* c_row = (float*)((char*)C + iy * stride_c);
                c_row[ix] = result * magicScale;
            }
        }
#endif
    }

    template <int BROW, int AROW>
    void mul_mat_bf16_f32_direct_avx512(
        int n,
        const uint16_t* A,  // BFloat16 以 uint16_t 存储
        size_t stride_a,
        const float* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#if defined(__AVX512BF16__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        constexpr int SIMD_WIDTH = 16;  // AVX512 一次处理 16 个 float
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;
    
        // 累加器
        __m512 acc[AROW * BROW];
        
        // 初始化
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm512_setzero_ps();
        }
        
        // 主循环
        for (int i = 0; i < nb; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                
                // 从 BFloat16 转换到 float32
                // BFloat16 存储在低16位，需要将其移到高16位来构成float32
                __m256i bf16_vec = _mm256_loadu_si256((const __m256i*)(a_row + i * SIMD_WIDTH));
                
                // 方法2：手动转换 - 将BFloat16左移16位得到float32
                __m512i bf16_expanded = _mm512_cvtepu16_epi32(bf16_vec);
                __m512i float_bits = _mm512_slli_epi32(bf16_expanded, 16);
                __m512 a_vec = _mm512_castsi512_ps(float_bits);
                
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
                
                // 使用掩码加载 BFloat16
                __m256i bf16_vec = _mm256_maskz_loadu_epi16(mask, a_row + nb * SIMD_WIDTH);
                __m512i bf16_expanded = _mm512_cvtepu16_epi32(bf16_vec);
                __m512i float_bits = _mm512_slli_epi32(bf16_expanded, 16);
                __m512 a_vec = _mm512_castsi512_ps(float_bits);
                
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
    
    template <int BROW, int AROW>
    void mul_mat_bf16_bf16_direct_avx512(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX512BF16__
        constexpr int SIMD_WIDTH = 32;  // AVX512 一次处理 32 个 bf16
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;
        if (remainder != 0) {
            printf("In mul_mat_bf16_bf16_direct_avx512, n %% 32 should be 0.");
            exit(0);
        }

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
                __m512bh a_vec = (__m512bh)_mm512_loadu_si512((__m512i const*)(a_row + i * SIMD_WIDTH));

                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    __m512bh b_vec = (__m512bh)_mm512_loadu_si512((__m512i const*)(b_row + i * SIMD_WIDTH));
                    
                    int acc_idx = ix * BROW + iy;
                    acc[acc_idx] = _mm512_dpbf16_ps(acc[acc_idx], a_vec, b_vec);
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
    void LinearBFloat16BFloat16_AVX512BF16_Row_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int i, int m, int k, int st, int end) {
        int j = st;
        for (j = st; j + 4 < end; j += 5) {
            mul_mat_bf16_bf16_direct_avx512 <BRow, 5> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float));
        }
        switch (end - j) {
            case 0: break;
            case 1: mul_mat_bf16_bf16_direct_avx512 <BRow, 1> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 2: mul_mat_bf16_bf16_direct_avx512 <BRow, 2> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 3: mul_mat_bf16_bf16_direct_avx512 <BRow, 3> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 4: mul_mat_bf16_bf16_direct_avx512 <BRow, 4> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
        }
    }

    bool LinearBFloat16BFloat16_AVX512BF16_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        int i = 0;
        for (; i + 4 < n; i += 5) {
            LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <5> (inputData, weightData, biasData, outputData, i, m, k, st, end);
        }
        switch (n - i) {
            case 0: break;
            case 1: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <1> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 2: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <2> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 3: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <3> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 4: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <4> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
        }
        AddBiasAVX512(outputData, biasData, n, k, st, end);
        return true;
    }

    bool LinearBFloat16FP8E4M3_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, float *scales, 
                        int ks, int ms, float magicScale) {
        if (!(m % blockM == 0 && blockM % 32 == 0)) {
            return false;
        }
 #ifdef __AVX512BF16__
        if (n > 31) {
            std::vector <uint16_t> tempBF16;
            tempBF16.resize((end - st) * m);
            static struct FP8E4M3ToFP32Manager fp8e4m3tofp32;
            for (int i = st; i < end; i++) {
                for (int midx = 0; midx < ms; midx++) {
                    float curScale = scales[i / blockK * ms + midx];
                    for (int l = midx * blockM; l < (midx + 1) * blockM; l++) {
                        float now = fp8e4m3tofp32.dict[weightData[i * m + l]] * curScale;
                        uint32_t val;
                        memcpy(&val, &now, sizeof(val));
                        tempBF16[(i - st) * m + l] = (uint16_t)(val >> 16);
                    }
                }
            }
            return LinearBFloat16BFloat16_AVX512BF16_Kernel(inputData, tempBF16.data(), biasData ? biasData + st : nullptr, 
                outputData + st, n, m, k, 0, end - st);
        }

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

    bool LinearBFloat16_FP8E4M3BLOCK128_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX512BF16__
        static int block_size = 128;
        size_t perRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, m);
        float magicScale = pow(2, 120);
        
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;

            int j = st;
            __m256i v_a_mask_byte = _mm256_set1_epi8(0x80); 
            __m256i v_b_mask_byte = _mm256_set1_epi8(0x7F); 
            
            for (; j < end; j++) {
                float now = 0.0f;
                __m512 last_sum = _mm512_setzero_ps(); // Accumulator for 16 parallel sums

                // 获取当前行的起始位置
                uint8_t *rowData = (uint8_t*)weightData + j * perRow;
                
                // 计算需要多少个block（每个block有128个FP8 + 1个float scale）
                const int blockM = 128;
                int numBlocks = (m + blockM - 1) / blockM;
                
                for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
                    // 计算当前block在rowData中的偏移
                    // 每个block占用 128 bytes (FP8) + 4 bytes (float scale)
                    size_t blockOffset = blockIdx * (blockM + sizeof(float));
                    
                    // 获取当前block的FP8数据和scale
                    uint8_t *fp8B = rowData + blockOffset;
                    
                    // 计算当前block处理的元素范围
                    int blockStart = blockIdx * blockM;
                    int blockEnd = std::min(blockStart + blockM, m);
                    
                    __m512 v_sum = _mm512_setzero_ps(); // Accumulator for 16 parallel sums
                    
                    // 处理当前block内的数据
                    int l = blockStart;
                    for (; l + 31 < blockEnd; l += 32) {
                        // 1. Load 32 BF16 inputs
                        __m512bh v_input_bf16 = (__m512bh)_mm512_loadu_si512((__m512i const*)(bf16A + l));
                        
                        // 2. Load 32 FP8 weights from current block
                        // 注意：fp8B指向当前block的开始，所以需要用 (l - blockStart) 作为偏移
                        __m256i va_bytes = _mm256_loadu_si256((__m256i*)(fp8B + (l - blockStart)));

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
                    
                    // 处理剩余的元素（如果有）
                    // TODO: 这里可能需要处理不足32个元素的情况
                    
                    float curScale = *(float*)(fp8B + blockM);  // scale在128个FP8之后
                    __m512 vScale = _mm512_set1_ps(curScale);
                    last_sum = _mm512_fmadd_ps(v_sum, vScale, last_sum);
                }
                
                now += _mm512_reduce_add_ps(last_sum) * magicScale;
                floatC[j] = now;
            }
        }
        return true;
#endif
        return false;
    }

    bool LinearBFloat16_FP8E4M3PERCHANNEL_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX512BF16__
        size_t perRow = GetDataBytes(DataType::FP8_E4M3_PERCHANNEL, 1, m);
        float magicScale = pow(2, 120);
        
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;

            int j = st;
            __m256i v_a_mask_byte = _mm256_set1_epi8(0x80); 
            __m256i v_b_mask_byte = _mm256_set1_epi8(0x7F); 
            
            for (; j < end; j++) {
                float now = 0.0f;
                __m512 last_sum = _mm512_setzero_ps(); // Accumulator for 16 parallel sums

                // 获取当前行的起始位置
                uint8_t *rowData = (uint8_t*)weightData + j * perRow;
                uint8_t *fp8B = rowData;
                    
                // 计算当前block处理的元素范围
                int blockStart = 0;
                int blockEnd = m;
                    
                __m512 v_sum = _mm512_setzero_ps(); // Accumulator for 16 parallel sums
                    
                // 处理当前block内的数据
                int l = blockStart;
                for (; l + 31 < blockEnd; l += 32) {
                    // 1. Load 32 BF16 inputs
                    __m512bh v_input_bf16 = (__m512bh)_mm512_loadu_si512((__m512i const*)(bf16A + l));
                        
                    // 2. Load 32 FP8 weights from current block
                    // 注意：fp8B指向当前block的开始，所以需要用 (l - blockStart) 作为偏移
                    __m256i va_bytes = _mm256_loadu_si256((__m256i*)(fp8B + (l - blockStart)));

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
                    
                // 处理剩余的元素（如果有）
                // TODO: 这里可能需要处理不足32个元素的情况
                    
                float curScale = *(float*)(fp8B + m);  // scale在m个FP8之后
                __m512 vScale = _mm512_set1_ps(curScale);
                last_sum = _mm512_fmadd_ps(v_sum, vScale, last_sum);

                now += _mm512_reduce_add_ps(last_sum) * magicScale;
                floatC[j] = now;
            }
        }
        return true;
#endif
        return false;
    }
}