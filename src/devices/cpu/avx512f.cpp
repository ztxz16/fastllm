//
// Created by huangyuyang on 8/14/25.
//

#include <cstdint>
#include <algorithm>
#include <cstring>

#if defined(__AVX2__) || defined(__AVX512F__)
#include "immintrin.h"
#endif

#include "utils.h"
#include "fastllm.h"

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

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    static inline __m512 NVFP4ToFloat32_AVX512(const uint8_t *packed) {
        __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed));
        const __m128i lowMask = _mm_set1_epi8(0x0F);
        __m128i low = _mm_and_si128(bytes, lowMask);
        __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), lowMask);
        __m128i interleaved = _mm_unpacklo_epi8(low, high);

        __m512i fp4 = _mm512_cvtepu8_epi32(interleaved);
        __m512i sign = _mm512_slli_epi32(_mm512_and_si512(fp4, _mm512_set1_epi32(0x8)), 28);
        __m512i body = _mm512_and_si512(fp4, _mm512_set1_epi32(0x7));

        __m512i exp = _mm512_slli_epi32(_mm512_add_epi32(_mm512_srli_epi32(body, 1), _mm512_set1_epi32(126)), 23);
        __m512i mant = _mm512_slli_epi32(_mm512_and_si512(body, _mm512_set1_epi32(1)), 22);
        mant = _mm512_maskz_mov_epi32(_mm512_cmpneq_epi32_mask(body, _mm512_set1_epi32(1)), mant);

        __m512i bits = _mm512_or_si512(sign, _mm512_or_si512(exp, mant));
        bits = _mm512_maskz_mov_epi32(_mm512_cmpneq_epi32_mask(body, _mm512_setzero_si512()), bits);
        return _mm512_castsi512_ps(bits);
    }

    static inline __m512 BFloat16ToFloat32_AVX512(const uint16_t *input) {
        __m256i bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
        __m512i fp32Bits = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16);
        return _mm512_castsi512_ps(fp32Bits);
    }

    static inline float NVFP4E8M0ScaleToFloatFast(uint8_t v) {
        uint32_t bits = v == 0 ? 0x00400000u : ((uint32_t)v << 23);
        float ret;
        memcpy(&ret, &bits, sizeof(ret));
        return ret;
    }

    static constexpr float NVFP4_E2M1_TABLE_AVX512F[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };

    template <int COLS, bool INPUT_BF16>
    static inline void LinearNVFP4E8M0Cols_AVX512F(
        const void *inputBase, const uint8_t *weightRows, const float *biasData, float *output,
        int outputCol, int m, int blockK, int blockM, const uint8_t *scaleBytes,
        int ms, int packedM
    ) {
        const float *inputF32 = reinterpret_cast<const float*>(inputBase);
        const uint16_t *inputBF16 = reinterpret_cast<const uint16_t*>(inputBase);

        __m512 acc[COLS];
        float now[COLS];
        for (int c = 0; c < COLS; c++) {
            acc[c] = _mm512_setzero_ps();
            now[c] = biasData ? biasData[outputCol + c] : 0.0f;
        }

        for (int midx = 0; midx < ms; midx++) {
            __m512 scaleVec[COLS];
            float scaleVal[COLS];
            if (outputCol / blockK == (outputCol + COLS - 1) / blockK) {
                float v = NVFP4E8M0ScaleToFloatFast(scaleBytes[(size_t)(outputCol / blockK) * ms + midx]);
                __m512 vv = _mm512_set1_ps(v);
                for (int c = 0; c < COLS; c++) {
                    scaleVal[c] = v;
                    scaleVec[c] = vv;
                }
            } else {
                for (int c = 0; c < COLS; c++) {
                    size_t scaleIdx = (size_t)((outputCol + c) / blockK) * ms + midx;
                    scaleVal[c] = NVFP4E8M0ScaleToFloatFast(scaleBytes[scaleIdx]);
                    scaleVec[c] = _mm512_set1_ps(scaleVal[c]);
                }
            }

            int l = midx * blockM;
            int blockEnd = std::min(m, (midx + 1) * blockM);
            __m512 blockAcc[COLS];
            for (int c = 0; c < COLS; c++) {
                blockAcc[c] = _mm512_setzero_ps();
            }
            for (; l + 15 < blockEnd; l += 16) {
                __m512 vi;
                if constexpr (INPUT_BF16) {
                    vi = BFloat16ToFloat32_AVX512(inputBF16 + l);
                } else {
                    vi = _mm512_loadu_ps(inputF32 + l);
                }
                for (int c = 0; c < COLS; c++) {
                    __m512 vw = NVFP4ToFloat32_AVX512(weightRows + (size_t)c * packedM + (l >> 1));
                    blockAcc[c] = _mm512_fmadd_ps(vi, vw, blockAcc[c]);
                }
            }
            for (int c = 0; c < COLS; c++) {
                acc[c] = _mm512_fmadd_ps(blockAcc[c], scaleVec[c], acc[c]);
            }

            for (; l < blockEnd; l++) {
                float x;
                if constexpr (INPUT_BF16) {
                    uint32_t inputBits = static_cast<uint32_t>(inputBF16[l]) << 16;
                    memcpy(&x, &inputBits, sizeof(x));
                } else {
                    x = inputF32[l];
                }
                uint8_t shift = (l & 1) ? 4 : 0;
                for (int c = 0; c < COLS; c++) {
                    uint8_t packed = weightRows[(size_t)c * packedM + (l >> 1)];
                    now[c] += scaleVal[c] * x * NVFP4_E2M1_TABLE_AVX512F[(packed >> shift) & 0xF];
                }
            }
        }

        for (int c = 0; c < COLS; c++) {
            output[c] = now[c] + _mm512_reduce_add_ps(acc[c]);
        }
    }

    template <bool INPUT_BF16>
    static inline bool LinearNVFP4E8M0_AVX512F_Run(
        const void *inputData, uint8_t *weightData, float *biasData, float *outputData,
        int n, int m, int k, int st, int end, int blockK, int blockM,
        const uint8_t *scaleBytes, int ms
    ) {
        int packedM = m >> 1;
        for (int i = 0; i < n; i++) {
            const void *input;
            if constexpr (INPUT_BF16) {
                input = reinterpret_cast<const uint16_t*>(inputData) + (size_t)i * m;
            } else {
                input = reinterpret_cast<const float*>(inputData) + (size_t)i * m;
            }

            int j = st;
            for (; j + 3 < end; j += 4) {
                LinearNVFP4E8M0Cols_AVX512F<4, INPUT_BF16>(
                    input, weightData + (size_t)j * packedM, biasData, outputData + (size_t)i * k + j,
                    j, m, blockK, blockM, scaleBytes, ms, packedM);
            }
            switch (end - j) {
                case 0: break;
                case 1:
                    LinearNVFP4E8M0Cols_AVX512F<1, INPUT_BF16>(
                        input, weightData + (size_t)j * packedM, biasData, outputData + (size_t)i * k + j,
                        j, m, blockK, blockM, scaleBytes, ms, packedM);
                    break;
                case 2:
                    LinearNVFP4E8M0Cols_AVX512F<2, INPUT_BF16>(
                        input, weightData + (size_t)j * packedM, biasData, outputData + (size_t)i * k + j,
                        j, m, blockK, blockM, scaleBytes, ms, packedM);
                    break;
                case 3:
                    LinearNVFP4E8M0Cols_AVX512F<3, INPUT_BF16>(
                        input, weightData + (size_t)j * packedM, biasData, outputData + (size_t)i * k + j,
                        j, m, blockK, blockM, scaleBytes, ms, packedM);
                    break;
            }
        }
        return true;
    }
#endif

    bool LinearFloat32NVFP4_AVX512F_Kernel(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, float *scales,
                        int ks, int ms) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        (void)ks;
        if (blockM % 16 != 0 || (m & 1)) {
            return false;
        }
        int packedM = m >> 1;
        for (int i = 0; i < n; i++) {
            const float *input = inputData + i * m;
            int j = st;
            for (; j + 3 < end; j += 4) {
                float now0 = biasData ? biasData[j + 0] : 0.0f;
                float now1 = biasData ? biasData[j + 1] : 0.0f;
                float now2 = biasData ? biasData[j + 2] : 0.0f;
                float now3 = biasData ? biasData[j + 3] : 0.0f;
                int blockK0 = (j + 0) / blockK;
                int blockK1 = (j + 1) / blockK;
                int blockK2 = (j + 2) / blockK;
                int blockK3 = (j + 3) / blockK;
                __m512 scaledSum0 = _mm512_setzero_ps();
                __m512 scaledSum1 = _mm512_setzero_ps();
                __m512 scaledSum2 = _mm512_setzero_ps();
                __m512 scaledSum3 = _mm512_setzero_ps();
                for (int midx = 0; midx < ms; midx++) {
                    __m512 scale0 = _mm512_set1_ps(scales[blockK0 * ms + midx]);
                    __m512 scale1 = _mm512_set1_ps(scales[blockK1 * ms + midx]);
                    __m512 scale2 = _mm512_set1_ps(scales[blockK2 * ms + midx]);
                    __m512 scale3 = _mm512_set1_ps(scales[blockK3 * ms + midx]);
                    int l = midx * blockM;
                    int blockEnd = std::min(m, (midx + 1) * blockM);
                    __m512 sum0 = _mm512_setzero_ps();
                    __m512 sum1 = _mm512_setzero_ps();
                    __m512 sum2 = _mm512_setzero_ps();
                    __m512 sum3 = _mm512_setzero_ps();
                    for (; l + 15 < blockEnd; l += 16) {
                        __m512 vi = _mm512_loadu_ps(input + l);
                        sum0 = _mm512_fmadd_ps(vi, NVFP4ToFloat32_AVX512(weightData + (j + 0) * packedM + (l >> 1)), sum0);
                        sum1 = _mm512_fmadd_ps(vi, NVFP4ToFloat32_AVX512(weightData + (j + 1) * packedM + (l >> 1)), sum1);
                        sum2 = _mm512_fmadd_ps(vi, NVFP4ToFloat32_AVX512(weightData + (j + 2) * packedM + (l >> 1)), sum2);
                        sum3 = _mm512_fmadd_ps(vi, NVFP4ToFloat32_AVX512(weightData + (j + 3) * packedM + (l >> 1)), sum3);
                    }
                    scaledSum0 = _mm512_fmadd_ps(sum0, scale0, scaledSum0);
                    scaledSum1 = _mm512_fmadd_ps(sum1, scale1, scaledSum1);
                    scaledSum2 = _mm512_fmadd_ps(sum2, scale2, scaledSum2);
                    scaledSum3 = _mm512_fmadd_ps(sum3, scale3, scaledSum3);
                    for (; l < blockEnd; l++) {
                        static const float table[16] = {
                            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
                        };
                        uint8_t packed0 = weightData[(j + 0) * packedM + (l >> 1)];
                        uint8_t packed1 = weightData[(j + 1) * packedM + (l >> 1)];
                        uint8_t packed2 = weightData[(j + 2) * packedM + (l >> 1)];
                        uint8_t packed3 = weightData[(j + 3) * packedM + (l >> 1)];
                        uint8_t shift = (l & 1) ? 4 : 0;
                        float x = input[l];
                        now0 += scales[blockK0 * ms + midx] * x * table[(packed0 >> shift) & 0xF];
                        now1 += scales[blockK1 * ms + midx] * x * table[(packed1 >> shift) & 0xF];
                        now2 += scales[blockK2 * ms + midx] * x * table[(packed2 >> shift) & 0xF];
                        now3 += scales[blockK3 * ms + midx] * x * table[(packed3 >> shift) & 0xF];
                    }
                }
                outputData[i * k + j + 0] = now0 + _mm512_reduce_add_ps(scaledSum0);
                outputData[i * k + j + 1] = now1 + _mm512_reduce_add_ps(scaledSum1);
                outputData[i * k + j + 2] = now2 + _mm512_reduce_add_ps(scaledSum2);
                outputData[i * k + j + 3] = now3 + _mm512_reduce_add_ps(scaledSum3);
            }
            for (; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int currentBlockK = j / blockK;
                __m512 scaledSum = _mm512_setzero_ps();
                for (int midx = 0; midx < ms; midx++) {
                    float curScale = scales[currentBlockK * ms + midx];
                    __m512 scale = _mm512_set1_ps(curScale);
                    int l = midx * blockM;
                    int blockEnd = std::min(m, (midx + 1) * blockM);
                    __m512 sum = _mm512_setzero_ps();
                    for (; l + 15 < blockEnd; l += 16) {
                        __m512 vi = _mm512_loadu_ps(input + l);
                        __m512 vw = NVFP4ToFloat32_AVX512(weightData + j * packedM + (l >> 1));
                        sum = _mm512_fmadd_ps(vi, vw, sum);
                    }
                    scaledSum = _mm512_fmadd_ps(sum, scale, scaledSum);
                    for (; l < blockEnd; l++) {
                        static const float table[16] = {
                            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
                        };
                        uint8_t packed = weightData[j * packedM + (l >> 1)];
                        uint8_t fp4 = (l & 1) ? (packed >> 4) : (packed & 0xF);
                        now += curScale * input[l] * table[fp4];
                    }
                }
                outputData[i * k + j] = now + _mm512_reduce_add_ps(scaledSum);
            }
        }
        return true;
#else
        return false;
#endif
    }

    bool LinearBFloat16NVFP4_AVX512F_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, float *scales,
                        int ks, int ms) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        (void)ks;
        if (blockM % 16 != 0 || (m & 1)) {
            return false;
        }
        int packedM = m >> 1;
        for (int i = 0; i < n; i++) {
            const uint16_t *input = inputData + i * m;
            int j = st;
            for (; j + 3 < end; j += 4) {
                float now0 = biasData ? biasData[j + 0] : 0.0f;
                float now1 = biasData ? biasData[j + 1] : 0.0f;
                float now2 = biasData ? biasData[j + 2] : 0.0f;
                float now3 = biasData ? biasData[j + 3] : 0.0f;
                int blockK0 = (j + 0) / blockK;
                int blockK1 = (j + 1) / blockK;
                int blockK2 = (j + 2) / blockK;
                int blockK3 = (j + 3) / blockK;
                __m512 scaledSum0 = _mm512_setzero_ps();
                __m512 scaledSum1 = _mm512_setzero_ps();
                __m512 scaledSum2 = _mm512_setzero_ps();
                __m512 scaledSum3 = _mm512_setzero_ps();
                for (int midx = 0; midx < ms; midx++) {
                    __m512 scale0 = _mm512_set1_ps(scales[blockK0 * ms + midx]);
                    __m512 scale1 = _mm512_set1_ps(scales[blockK1 * ms + midx]);
                    __m512 scale2 = _mm512_set1_ps(scales[blockK2 * ms + midx]);
                    __m512 scale3 = _mm512_set1_ps(scales[blockK3 * ms + midx]);
                    int l = midx * blockM;
                    int blockEnd = std::min(m, (midx + 1) * blockM);
                    __m512 sum0 = _mm512_setzero_ps();
                    __m512 sum1 = _mm512_setzero_ps();
                    __m512 sum2 = _mm512_setzero_ps();
                    __m512 sum3 = _mm512_setzero_ps();
                    for (; l + 15 < blockEnd; l += 16) {
                        __m512 vi = BFloat16ToFloat32_AVX512(input + l);
                        sum0 = _mm512_fmadd_ps(vi, NVFP4ToFloat32_AVX512(weightData + (j + 0) * packedM + (l >> 1)), sum0);
                        sum1 = _mm512_fmadd_ps(vi, NVFP4ToFloat32_AVX512(weightData + (j + 1) * packedM + (l >> 1)), sum1);
                        sum2 = _mm512_fmadd_ps(vi, NVFP4ToFloat32_AVX512(weightData + (j + 2) * packedM + (l >> 1)), sum2);
                        sum3 = _mm512_fmadd_ps(vi, NVFP4ToFloat32_AVX512(weightData + (j + 3) * packedM + (l >> 1)), sum3);
                    }
                    scaledSum0 = _mm512_fmadd_ps(sum0, scale0, scaledSum0);
                    scaledSum1 = _mm512_fmadd_ps(sum1, scale1, scaledSum1);
                    scaledSum2 = _mm512_fmadd_ps(sum2, scale2, scaledSum2);
                    scaledSum3 = _mm512_fmadd_ps(sum3, scale3, scaledSum3);
                    for (; l < blockEnd; l++) {
                        static const float table[16] = {
                            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
                        };
                        uint8_t packed0 = weightData[(j + 0) * packedM + (l >> 1)];
                        uint8_t packed1 = weightData[(j + 1) * packedM + (l >> 1)];
                        uint8_t packed2 = weightData[(j + 2) * packedM + (l >> 1)];
                        uint8_t packed3 = weightData[(j + 3) * packedM + (l >> 1)];
                        uint8_t shift = (l & 1) ? 4 : 0;
                        uint32_t inputBits = static_cast<uint32_t>(input[l]) << 16;
                        float x = *reinterpret_cast<float*>(&inputBits);
                        now0 += scales[blockK0 * ms + midx] * x * table[(packed0 >> shift) & 0xF];
                        now1 += scales[blockK1 * ms + midx] * x * table[(packed1 >> shift) & 0xF];
                        now2 += scales[blockK2 * ms + midx] * x * table[(packed2 >> shift) & 0xF];
                        now3 += scales[blockK3 * ms + midx] * x * table[(packed3 >> shift) & 0xF];
                    }
                }
                outputData[i * k + j + 0] = now0 + _mm512_reduce_add_ps(scaledSum0);
                outputData[i * k + j + 1] = now1 + _mm512_reduce_add_ps(scaledSum1);
                outputData[i * k + j + 2] = now2 + _mm512_reduce_add_ps(scaledSum2);
                outputData[i * k + j + 3] = now3 + _mm512_reduce_add_ps(scaledSum3);
            }
            for (; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int currentBlockK = j / blockK;
                __m512 scaledSum = _mm512_setzero_ps();
                for (int midx = 0; midx < ms; midx++) {
                    float curScale = scales[currentBlockK * ms + midx];
                    __m512 scale = _mm512_set1_ps(curScale);
                    int l = midx * blockM;
                    int blockEnd = std::min(m, (midx + 1) * blockM);
                    __m512 sum = _mm512_setzero_ps();
                    for (; l + 15 < blockEnd; l += 16) {
                        __m512 vi = BFloat16ToFloat32_AVX512(input + l);
                        __m512 vw = NVFP4ToFloat32_AVX512(weightData + j * packedM + (l >> 1));
                        sum = _mm512_fmadd_ps(vi, vw, sum);
                    }
                    scaledSum = _mm512_fmadd_ps(sum, scale, scaledSum);
                    for (; l < blockEnd; l++) {
                        static const float table[16] = {
                            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
                        };
                        uint8_t packed = weightData[j * packedM + (l >> 1)];
                        uint8_t fp4 = (l & 1) ? (packed >> 4) : (packed & 0xF);
                        uint32_t inputBits = static_cast<uint32_t>(input[l]) << 16;
                        float inputFloat = *reinterpret_cast<float*>(&inputBits);
                        now += curScale * inputFloat * table[fp4];
                    }
                }
                outputData[i * k + j] = now + _mm512_reduce_add_ps(scaledSum);
            }
        }
        return true;
#else
        return false;
#endif
    }

    bool LinearFloat32NVFP4E8M0_AVX512F_Kernel(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, uint8_t *scaleBytes,
                        int ks, int ms) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        (void)ks;
        if (blockM % 16 != 0 || (m & 1)) {
            return false;
        }
        return LinearNVFP4E8M0_AVX512F_Run<false>(inputData, weightData, biasData, outputData,
                                                  n, m, k, st, end, blockK, blockM, scaleBytes, ms);
#else
        return false;
#endif
    }

    bool LinearBFloat16NVFP4E8M0_AVX512F_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, uint8_t *scaleBytes,
                        int ks, int ms) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        (void)ks;
        if (blockM % 16 != 0 || (m & 1)) {
            return false;
        }
        return LinearNVFP4E8M0_AVX512F_Run<true>(inputData, weightData, biasData, outputData,
                                                 n, m, k, st, end, blockK, blockM, scaleBytes, ms);
#else
        return false;
#endif
    }
}
