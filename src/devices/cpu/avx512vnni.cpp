//
// Created by huangyuyang on 5/20/25.
//

#include <cstdint>
#include <algorithm>

#include "fastllm.h"
#include "utils/utils.h"
#ifdef __AVX2__
#include "immintrin.h"
#endif

namespace fastllm {
    int DotU4U8_AVX512VNNI(uint8_t *a, uint8_t *b, int n) {
#ifdef __AVX512VNNI__
        __m512i acc = _mm512_setzero_si512();
        int i = 0;
        int ans = 0;
        const __m512i lowMask = _mm512_set1_epi8(0xf);
        const __m512i ones = _mm512_set1_epi16(1);
        for (; i + 63 < n; i += 64) {
            __m256i orix = _mm256_loadu_si256((const __m256i *) (a + i / 2));
            __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
            __m512i bx = _mm512_and_si512(lowMask, bytex);
            __m512i by = _mm512_loadu_si512((const __m512i *) (b + i));
            acc = _mm512_dpbusd_epi32(acc, by, bx);
        }
        for (; i < n; i++) {
            ans += a[i] * b[i];
        }
        return ans + _mm512_reduce_add_epi32(acc);
#endif
        return 0;
    };

    // int8和int4的矩阵乘法  
    // a: [n * m]的int8矩阵
    // b: [k * m]的int4矩阵
    // c: [n * k]的float32的结果矩阵  
    bool MatMulInt8Int4_AVX512VNNI(uint8_t *a, uint8_t *b, float *c, int n, int m, int k) {
#ifdef __AVX512VNNI__
        if (m % 64 != 0) {
            return false;
        }
        float *values = c;

        int block = 0;
        for (; block + 3 < n; block += 4) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                uint8_t *a = weightWalk + (i * m) / 2;
                uint8_t *b = inputStart;

                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();

                const __m512i lowMask = _mm512_set1_epi8(0xf);
                const __m512i ones = _mm512_set1_epi16(1);
                int j = 0, ans = 0;
                for (; j + 63 < m; j += 64) {
                    __m256i orix = _mm256_loadu_si256((const __m256i *) (a + j / 2));
                    __m512i by0 = _mm512_loadu_si512((const __m512i *) (b + j));
                    __m512i by1 = _mm512_loadu_si512((const __m512i *) (b + m * 1 + j));
                    __m512i by2 = _mm512_loadu_si512((const __m512i *) (b + m * 2 + j));
                    __m512i by3 = _mm512_loadu_si512((const __m512i *) (b + m * 3 + j));

                    __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                    __m512i bx = _mm512_and_si512(lowMask, bytex);

                    acc0 = _mm512_dpbusd_epi32(acc0, by0, bx);
                    acc1 = _mm512_dpbusd_epi32(acc1, by1, bx);
                    acc2 = _mm512_dpbusd_epi32(acc2, by2, bx);
                    acc3 = _mm512_dpbusd_epi32(acc3, by3, bx);
                }

                values[block * k + i] = _mm512_reduce_add_epi32(acc0);
                values[(block + 1) * k + i] = _mm512_reduce_add_epi32(acc1);
                values[(block + 2) * k + i] = _mm512_reduce_add_epi32(acc2);
                values[(block + 3) * k + i] = _mm512_reduce_add_epi32(acc3);
            }
        }

        for (; block + 1 < n; block += 2) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                uint8_t *a = weightWalk + (i * m) / 2;
                uint8_t *b = inputStart;

                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();

                const __m512i lowMask = _mm512_set1_epi8(0xf);
                const __m512i ones = _mm512_set1_epi16(1);
                int j = 0, ans = 0;
                for (; j + 63 < m; j += 64) {
                    __m256i orix = _mm256_loadu_si256((const __m256i *) (a + j / 2));
                    __m512i by0 = _mm512_loadu_si512((const __m512i *) (b + j));
                    __m512i by1 = _mm512_loadu_si512((const __m512i *) (b + m * 1 + j));

                    __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                    __m512i bx = _mm512_and_si512(lowMask, bytex);

                    acc0 = _mm512_dpbusd_epi32(acc0, by0, bx);
                    acc1 = _mm512_dpbusd_epi32(acc1, by1, bx);
                }

                values[block * k + i] = _mm512_reduce_add_epi32(acc0);
                values[(block + 1) * k + i] = _mm512_reduce_add_epi32(acc1);
            }
        }

        for (; block < n; block++) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                uint8_t *a = weightWalk + (i * m) / 2;
                uint8_t *b = inputStart;

                __m512i acc = _mm512_setzero_si512();
                const __m512i lowMask = _mm512_set1_epi8(0xf);
                const __m512i ones = _mm512_set1_epi16(1);
                int j = 0, ans = 0;
                for (; j + 63 < m; j += 64) {
                    __m256i orix = _mm256_loadu_si256((const __m256i *) (a + j / 2));
                    __m512i by = _mm512_loadu_si512((const __m512i *) (b + j));

                    __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                    __m512i bx = _mm512_and_si512(lowMask, bytex);
                    acc = _mm512_dpbusd_epi32(acc, by, bx);
                }

                values[block * k + i] = _mm512_reduce_add_epi32(acc);
            }
        }
        return true;
#else
        return false;
#endif
    }

    // int8和int4的矩阵乘法  
    // a: [n * m]的int8矩阵
    // b: [k * m]的int4g矩阵
    // c: [n * k]的float32的结果矩阵  
    bool MatMulInt8Int4Group_AVX512VNNI(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, 
        int group, int realGroup, int groupCnt, float *iscales, float *scales, float *izeros, float *weightMins) {
#ifdef __AVX512VNNI__
        // Every quantization group must be representable by complete VNNI
        // blocks. Returning false selects the AVX2/scalar implementation with
        // the matching 32-value input layout.
        if (groupCnt <= 0 || groupCnt % 64 != 0 || m % 64 != 0) {
            return false;
        }
        float *values = c;
        int block = 0;

        for (; block + 3 < n; block += 4) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                uint8_t *a = weightWalk + (i * m) / 2;
                uint8_t *b = inputStart;

                __m512 lastSum0 = _mm512_setzero_ps();
                __m512 lastSum1 = _mm512_setzero_ps();
                __m512 lastSum2 = _mm512_setzero_ps();
                __m512 lastSum3 = _mm512_setzero_ps();

                for (int g = 0; g < realGroup; g++) {
                    const int iid0 = block * group + g;
                    const int iid1 = (block + 1) * group + g;
                    const int iid2 = (block + 2) * group + g;
                    const int iid3 = (block + 3) * group + g;
                    const int gid = i * group + g;
                    int st = g * groupCnt, end = std::min(m, (g + 1) * groupCnt);
                    
                    __m512i acc0 = _mm512_setzero_si512();
                    __m512i acc1 = _mm512_setzero_si512();
                    __m512i acc2 = _mm512_setzero_si512();
                    __m512i acc3 = _mm512_setzero_si512();

                    __m512i sub0 = _mm512_setzero_si512();
                    __m512i sub1 = _mm512_setzero_si512();
                    __m512i sub2 = _mm512_setzero_si512();
                    __m512i sub3 = _mm512_setzero_si512();

                    __m512i zeros0 = _mm512_set1_epi8((uint8_t)izeros[iid0]);
                    __m512i zeros1 = _mm512_set1_epi8((uint8_t)izeros[iid1]);
                    __m512i zeros2 = _mm512_set1_epi8((uint8_t)izeros[iid2]);
                    __m512i zeros3 = _mm512_set1_epi8((uint8_t)izeros[iid3]);

                    const __m512i lowMask = _mm512_set1_epi8(0xf);
                    const __m512i ones = _mm512_set1_epi16(1);
                    int j = st;
                    for (; j + 63 < end; j += 64) {
                        __m256i orix = _mm256_loadu_si256((const __m256i *) (a + j / 2));
                        __m512i by0 = _mm512_loadu_si512((const __m512i *) (b + j));
                        __m512i by1 = _mm512_loadu_si512((const __m512i *) (b + m * 1 + j));
                        __m512i by2 = _mm512_loadu_si512((const __m512i *) (b + m * 2 + j));
                        __m512i by3 = _mm512_loadu_si512((const __m512i *) (b + m * 3 + j));

                        __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                        __m512i bx = _mm512_and_si512(lowMask, bytex);

                        acc0 = _mm512_dpbusd_epi32(acc0, by0, bx);
                        acc1 = _mm512_dpbusd_epi32(acc1, by1, bx);
                        acc2 = _mm512_dpbusd_epi32(acc2, by2, bx);
                        acc3 = _mm512_dpbusd_epi32(acc3, by3, bx);

                        sub0 = _mm512_dpbusd_epi32(sub0, zeros0, bx);
                        sub1 = _mm512_dpbusd_epi32(sub1, zeros1, bx);
                        sub2 = _mm512_dpbusd_epi32(sub2, zeros2, bx);
                        sub3 = _mm512_dpbusd_epi32(sub3, zeros3, bx);
                    }

                    lastSum0 = _mm512_add_ps (
                        lastSum0, 
                        _mm512_mul_ps(
                            _mm512_cvtepi32_ps(_mm512_sub_epi32(acc0, sub0)), 
                            _mm512_set1_ps(iscales[iid0] * scales[gid])
                        ));
                    lastSum1 = _mm512_add_ps (
                        lastSum1, 
                        _mm512_mul_ps(
                            _mm512_cvtepi32_ps(_mm512_sub_epi32(acc1, sub1)), 
                            _mm512_set1_ps(iscales[iid1] * scales[gid])
                        ));
                    lastSum2 = _mm512_add_ps (
                        lastSum2, 
                        _mm512_mul_ps(
                            _mm512_cvtepi32_ps(_mm512_sub_epi32(acc2, sub2)), 
                            _mm512_set1_ps(iscales[iid2] * scales[gid])
                        ));
                    lastSum3 = _mm512_add_ps (
                        lastSum3, 
                        _mm512_mul_ps(
                            _mm512_cvtepi32_ps(_mm512_sub_epi32(acc3, sub3)), 
                            _mm512_set1_ps(iscales[iid3] * scales[gid])
                        ));
                }

                values[block * k + i] = _mm512_reduce_add_ps(lastSum0);
                values[(block + 1) * k + i] = _mm512_reduce_add_ps(lastSum1);
                values[(block + 2) * k + i] = _mm512_reduce_add_ps(lastSum2);
                values[(block + 3) * k + i] = _mm512_reduce_add_ps(lastSum3);
            }
        }

        for (; block + 1 < n; block += 2) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                uint8_t *a = weightWalk + (i * m) / 2;
                uint8_t *b = inputStart;

                __m512 lastSum0 = _mm512_setzero_ps();
                __m512 lastSum1 = _mm512_setzero_ps();

                for (int g = 0; g < realGroup; g++) {
                    const int iid0 = block * group + g;
                    const int iid1 = (block + 1) * group + g;
                    const int gid = i * group + g;
                    int st = g * groupCnt, end = std::min(m, (g + 1) * groupCnt);
                    
                    __m512i acc0 = _mm512_setzero_si512();
                    __m512i acc1 = _mm512_setzero_si512();

                    __m512i sub0 = _mm512_setzero_si512();
                    __m512i sub1 = _mm512_setzero_si512();

                    __m512i zeros0 = _mm512_set1_epi8((uint8_t)izeros[iid0]);
                    __m512i zeros1 = _mm512_set1_epi8((uint8_t)izeros[iid1]);

                    const __m512i lowMask = _mm512_set1_epi8(0xf);
                    int j = st;
                    for (; j + 63 < end; j += 64) {
                        __m256i orix = _mm256_loadu_si256((const __m256i *) (a + j / 2));
                        __m512i by0 = _mm512_loadu_si512((const __m512i *) (b + j));
                        __m512i by1 = _mm512_loadu_si512((const __m512i *) (b + m * 1 + j));

                        __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                        __m512i bx = _mm512_and_si512(lowMask, bytex);

                        acc0 = _mm512_dpbusd_epi32(acc0, by0, bx);
                        acc1 = _mm512_dpbusd_epi32(acc1, by1, bx);

                        sub0 = _mm512_dpbusd_epi32(sub0, zeros0, bx);
                        sub1 = _mm512_dpbusd_epi32(sub1, zeros1, bx);
                    }

                    lastSum0 = _mm512_add_ps (
                        lastSum0, 
                        _mm512_mul_ps(
                            _mm512_cvtepi32_ps(_mm512_sub_epi32(acc0, sub0)), 
                            _mm512_set1_ps(iscales[iid0] * scales[gid])
                        ));
                    lastSum1 = _mm512_add_ps (
                        lastSum1, 
                        _mm512_mul_ps(
                            _mm512_cvtepi32_ps(_mm512_sub_epi32(acc1, sub1)), 
                            _mm512_set1_ps(iscales[iid1] * scales[gid])
                        ));
                }

                values[block * k + i] = _mm512_reduce_add_ps(lastSum0);
                values[(block + 1) * k + i] = _mm512_reduce_add_ps(lastSum1);
            }
        }

        for (; block < n; block++) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                uint8_t *a = weightWalk + (i * m) / 2;
                uint8_t *b = inputStart;

                __m512 lastSum = _mm512_setzero_ps();
                for (int g = 0; g < realGroup; g++) {
                    const int iid = block * group + g;
                    const int gid = i * group + g;
                    int st = g * groupCnt, end = std::min(m, (g + 1) * groupCnt);
                    
                    __m512i acc = _mm512_setzero_si512();
                    __m512i sub = _mm512_setzero_si512();

                    const __m512i lowMask = _mm512_set1_epi8(0xf);
                    __m512i zeros = _mm512_set1_epi8((uint8_t)izeros[iid]);
                    
                    int j = st;
                    for (; j + 63 < end; j += 64) {
                        __m256i orix = _mm256_loadu_si256((const __m256i *) (a + j / 2));
                        __m512i by = _mm512_loadu_si512((const __m512i *) (b + j));

                        __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                        __m512i bx = _mm512_and_si512(lowMask, bytex);
                        acc = _mm512_dpbusd_epi32(acc, by, bx);
                        sub = _mm512_dpbusd_epi32(sub, zeros, bx);
                    }

                    lastSum = _mm512_add_ps (
                        lastSum, 
                        _mm512_mul_ps(
                            _mm512_cvtepi32_ps(_mm512_sub_epi32(acc, sub)), 
                            _mm512_set1_ps(iscales[iid] * scales[gid])
                        ));
                }

                values[block * k + i] = _mm512_reduce_add_ps(lastSum);
            }
        }
        return true;
#else
        return false;
#endif
    }

    extern void AddBiasAVX512(float *outputData, float *biasData, int n, int k, int st, int end);

    template <int BROW, int AROW>
    void mul_mat_int8_int8_direct_avx512vnni(
        int m,  // 内积维度
        const uint8_t* A,  // INT8_PERCHANNEL format (输入数据)
        size_t stride_a,
        const uint8_t* B,  // INT8_PERCHANNEL format (权重数据)
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX512VNNI__
        constexpr int SIMD_WIDTH = 64;  // AVX512 VNNI 一次处理 64 个 int8
        
        // 累加器 - 为每个输出元素准备一个累加器
        __m512i acc[AROW * BROW];
        
        // 初始化累加器
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm512_setzero_si512();
        }
        // 主循环 - 处理SIMD_WIDTH的倍数部分
        int nb = m / SIMD_WIDTH;
        for (int block = 0; block < nb; ++block) {
            for (int ia = 0; ia < AROW; ++ia) {
                // A矩阵的第ia行 (输入数据)
                const uint8_t* a_row = (const uint8_t*)((const char*)A + ia * stride_a);
                const int8_t* quantizedA = (const int8_t*)a_row;
                __m512i ay = _mm512_loadu_si512((const __m512i*)(quantizedA + block * SIMD_WIDTH));
                
                for (int ib = 0; ib < BROW; ++ib) {
                    // B矩阵的第ib行 (权重数据，INT8格式)
                    const uint8_t* b_row = (const uint8_t*)((const char*)B + ib * stride_b);
                    __m512i bx = _mm512_loadu_si512((const __m512i*)(b_row + block * SIMD_WIDTH));
                    int acc_idx = ia * BROW + ib;
                    acc[acc_idx] = _mm512_dpbusd_epi32(acc[acc_idx], bx, ay);
                }
            }
        }
        
        // 处理剩余部分
        int remainder = m % SIMD_WIDTH;
        int remainderSums[AROW * BROW] = {0};
        if (remainder > 0) {
            for (int ia = 0; ia < AROW; ++ia) {
                const uint8_t* a_row = (const uint8_t*)((const char*)A + ia * stride_a);
                const int8_t* quantizedA = (const int8_t*)a_row;
                
                for (int ib = 0; ib < BROW; ++ib) {
                    const uint8_t* b_row = (const uint8_t*)((const char*)B + ib * stride_b);
                    
                    for (int i = nb * SIMD_WIDTH; i < m; i++) {
                        int acc_idx = ia * BROW + ib;
                        remainderSums[acc_idx] += quantizedA[i] * b_row[i];
                    }
                }
            }
        }
        
        // 水平求和，应用缩放因子，然后存储结果
        for (int ia = 0; ia < AROW; ++ia) {
            const uint8_t* a_row = (const uint8_t*)((const char*)A + ia * stride_a);
            float scaleA = *(float*)(a_row + m);
            int sumA = *(int*)(a_row + m + sizeof(float));
            
            for (int ib = 0; ib < BROW; ++ib) {
                const uint8_t* b_row = (const uint8_t*)((const char*)B + ib * stride_b);
                float minB = *(float*)(b_row + m);
                float scaleB = *(float*)(b_row + m + sizeof(float));
                
                int acc_idx = ia * BROW + ib;
                int sum = _mm512_reduce_add_epi32(acc[acc_idx]) + remainderSums[acc_idx];
                
                // C[ia][ib] = sum * scaleA * scaleB + minB * scaleA * sumA
                float* c_ptr = (float*)((char*)C + ia * stride_c);
                c_ptr[ib] = sum * scaleA * scaleB + minB * scaleA * sumA;
            }
        }
#endif
    }

    template <int BRow>
    void LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX512VNNI_Row_Kernel(
        uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
        int i, int m, int k, int st, int end) 
    {
        size_t lda = GetDataBytes(DataType::INF_INT8_PERCHANNEL, 1, m);
        size_t ldb = GetDataBytes(DataType::INT8_PERCHANNEL, 1, m);
        size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
        
        int j = st;
        // 一次处理5列
        for (j = st; j + 4 < end; j += 5) {
            mul_mat_int8_int8_direct_avx512vnni<5, BRow>(
                m,
                inputData + i * lda, lda,           // A: 输入数据的第i行开始的BRow行
                weightData + j * ldb, ldb,          // B: 权重数据的第j行开始的5行
                outputData + i * k + j, ldc         // C: 输出位置
            );
        }
        
        // 处理剩余列
        switch (end - j) {
            case 0: break;
            case 1: 
                mul_mat_int8_int8_direct_avx512vnni<1, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
            case 2: 
                mul_mat_int8_int8_direct_avx512vnni<2, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
            case 3: 
                mul_mat_int8_int8_direct_avx512vnni<3, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
            case 4: 
                mul_mat_int8_int8_direct_avx512vnni<4, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
        }
    }

    bool LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX512VNNI_Kernel(
        uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
        int n, int m, int k, int st, int end) 
    {
        if (n == 1) {
#ifdef __AVX512VNNI__
            size_t lda = GetDataBytes(DataType::INF_INT8_PERCHANNEL, 1, m);
            size_t ldb = GetDataBytes(DataType::INT8_PERCHANNEL, 1, m);
            size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
            
            for (int i = 0; i < n; i++) {
                // A矩阵的第i行，InfInt8PerChannel格式
                uint8_t *infInt8A = (uint8_t*)inputData + i * lda;
                int8_t *quantizedA = (int8_t*)infInt8A;
                float scaleA = *(float*)(infInt8A + m);
                int sumA = *(int*)(infInt8A + m + sizeof(float));
                
                float *floatC = (float*)((uint8_t*)outputData + i * ldc);
                
                for (int j = st; j < end; j++) {
                    // B矩阵的第j行，INT8_PERCHANNEL格式
                    uint8_t *int8B = (uint8_t*)weightData + j * ldb;
                    float minB = *(float*)(int8B + m);
                    float scaleB = *(float*)(int8B + m + sizeof(float));
                    
                    int sum = 0;

                    __m512i acc = _mm512_setzero_si512();
                    const __m512i lowMask = _mm512_set1_epi8(0xf);
                    const __m512i ones = _mm512_set1_epi16(1);

                    int i = 0;
                    for (; i + 63 < m; i += 64) {
                        __m512i bx = _mm512_loadu_si512((const __m512i *) (int8B + i));
                        __m512i by = _mm512_loadu_si512((const __m512i *) (quantizedA + i));
                        acc = _mm512_dpbusd_epi32(acc, bx, by);
                    }
                    sum = _mm512_reduce_add_epi32(acc);

                    for (; i < m; i++) {
                        sum += quantizedA[i] * int8B[i];
                    }

                    floatC[j] = sum * scaleA * scaleB + minB * scaleA * sumA;
                }
            }
            AddBiasAVX512(outputData, biasData, n, k, st, end);
            return true;
#else
            return false;
#endif
        }
#ifdef __AVX512VNNI__
        int i = 0;
        // 一次处理5行
        for (; i + 4 < n; i += 5) {
            LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX512VNNI_Row_Kernel<5>(
                inputData, weightData, biasData, outputData, i, m, k, st, end
            );
        }
        
        // 处理剩余行
        switch (n - i) {
            case 0: break;
            case 1: 
                LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX512VNNI_Row_Kernel<1>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
            case 2: 
                LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX512VNNI_Row_Kernel<2>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
            case 3: 
                LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX512VNNI_Row_Kernel<3>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
            case 4: 
                LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX512VNNI_Row_Kernel<4>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
        }
        
        AddBiasAVX512(outputData, biasData, n, k, st, end);
        return true;
#else
        return false;
#endif
    }

    template <int BROW, int AROW>
    void mul_mat_int8_int4_direct_avx512vnni(
        int m,  // 内积维度
        const uint8_t* A,  // INT8_PERCHANNEL format (输入数据)
        size_t stride_a,
        const uint8_t* B,  // INT4_PERCHANNEL format (权重数据)
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX512VNNI__
        constexpr int SIMD_WIDTH = 64;  // AVX512 VNNI 一次处理 64 个 int8
        
        // 累加器 - 为每个输出元素准备一个累加器
        __m512i acc[AROW * BROW];
        
        // 初始化累加器
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm512_setzero_si512();
        }
        
        const __m512i lowMask = _mm512_set1_epi8(0xf);
        
        // 主循环 - 处理SIMD_WIDTH的倍数部分
        int nb = m / SIMD_WIDTH;
        for (int block = 0; block < nb; ++block) {
            for (int ia = 0; ia < AROW; ++ia) {
                // A矩阵的第ia行 (输入数据)
                const uint8_t* a_row = (const uint8_t*)((const char*)A + ia * stride_a);
                const int8_t* quantizedA = (const int8_t*)a_row;
                __m512i ay = _mm512_loadu_si512((const __m512i*)(quantizedA + block * SIMD_WIDTH));
                
                for (int ib = 0; ib < BROW; ++ib) {
                    // B矩阵的第ib行 (权重数据，INT4格式)
                    const uint8_t* b_row = (const uint8_t*)((const char*)B + ib * stride_b);
                    __m256i orix = _mm256_loadu_si256((const __m256i*)(b_row + block * SIMD_WIDTH / 2));
                    __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                    __m512i bx = _mm512_and_si512(lowMask, bytex);
                    
                    int acc_idx = ia * BROW + ib;
                    acc[acc_idx] = _mm512_dpbusd_epi32(acc[acc_idx], bx, ay);
                }
            }
        }
        
        // 处理剩余部分
        int remainder = m % SIMD_WIDTH;
        int remainderSums[AROW * BROW] = {0};
        if (remainder > 0) {
            for (int ia = 0; ia < AROW; ++ia) {
                const uint8_t* a_row = (const uint8_t*)((const char*)A + ia * stride_a);
                const int8_t* quantizedA = (const int8_t*)a_row;
                
                for (int ib = 0; ib < BROW; ++ib) {
                    const uint8_t* b_row = (const uint8_t*)((const char*)B + ib * stride_b);
                    
                    for (int i = nb * SIMD_WIDTH; i < m; i += 2) {
                        uint8_t packedValue = b_row[i / 2];
                        uint8_t int4Value0 = (packedValue >> 4) & 0x0F;
                        uint8_t int4Value1 = packedValue & 0x0F;
                        
                        int acc_idx = ia * BROW + ib;
                        remainderSums[acc_idx] += quantizedA[i] * int4Value0;
                        if (i + 1 < m) {
                            remainderSums[acc_idx] += quantizedA[i + 1] * int4Value1;
                        }
                    }
                }
            }
        }
        
        // 水平求和，应用缩放因子，然后存储结果
        for (int ia = 0; ia < AROW; ++ia) {
            const uint8_t* a_row = (const uint8_t*)((const char*)A + ia * stride_a);
            float scaleA = *(float*)(a_row + m);
            int sumA = *(int*)(a_row + m + sizeof(float));
            
            for (int ib = 0; ib < BROW; ++ib) {
                const uint8_t* b_row = (const uint8_t*)((const char*)B + ib * stride_b);
                float minB = *(float*)(b_row + (m + 1) / 2);
                float scaleB = *(float*)(b_row + (m + 1) / 2 + sizeof(float));
                
                int acc_idx = ia * BROW + ib;
                int sum = _mm512_reduce_add_epi32(acc[acc_idx]) + remainderSums[acc_idx];
                
                // C[ia][ib] = sum * scaleA * scaleB + minB * scaleA * sumA
                float* c_ptr = (float*)((char*)C + ia * stride_c);
                c_ptr[ib] = sum * scaleA * scaleB + minB * scaleA * sumA;
            }
        }
#endif
    }

    template <int BRow>
    void LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX512VNNI_Row_Kernel(
        uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
        int i, int m, int k, int st, int end) 
    {
        size_t lda = GetDataBytes(DataType::INF_INT8_PERCHANNEL, 1, m);
        size_t ldb = GetDataBytes(DataType::INT4_PERCHANNEL, 1, m);
        size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
        
        int j = st;
        // 一次处理5列
        for (j = st; j + 4 < end; j += 5) {
            mul_mat_int8_int4_direct_avx512vnni<5, BRow>(
                m,
                inputData + i * lda, lda,           // A: 输入数据的第i行开始的BRow行
                weightData + j * ldb, ldb,          // B: 权重数据的第j行开始的5行
                outputData + i * k + j, ldc         // C: 输出位置
            );
        }
        
        // 处理剩余列
        switch (end - j) {
            case 0: break;
            case 1: 
                mul_mat_int8_int4_direct_avx512vnni<1, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
            case 2: 
                mul_mat_int8_int4_direct_avx512vnni<2, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
            case 3: 
                mul_mat_int8_int4_direct_avx512vnni<3, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
            case 4: 
                mul_mat_int8_int4_direct_avx512vnni<4, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
        }
    }

    bool LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX512VNNI_Kernel(
        uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
        int n, int m, int k, int st, int end) 
    {
        if (n == 1) {
#ifdef __AVX512VNNI__
            size_t lda = GetDataBytes(DataType::INF_INT8_PERCHANNEL, 1, m);
            size_t ldb = GetDataBytes(DataType::INT4_PERCHANNEL, 1, m);
            size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
            
            for (int i = 0; i < n; i++) {
                // A矩阵的第i行，InfInt8PerChannel格式
                uint8_t *infInt8A = (uint8_t*)inputData + i * lda;
                int8_t *quantizedA = (int8_t*)infInt8A;
                float scaleA = *(float*)(infInt8A + m);
                int sumA = *(int*)(infInt8A + m + sizeof(float));
                
                float *floatC = (float*)((uint8_t*)outputData + i * ldc);
                
                for (int j = st; j < end; j++) {
                    // B矩阵的第j行，INT4_PERCHANNEL格式
                    uint8_t *int4B = (uint8_t*)weightData + j * ldb;
                    float minB = *(float*)(int4B + (m + 1) / 2);
                    float scaleB = *(float*)(int4B + (m + 1) / 2 + sizeof(float));
                    
                    int sum = 0;

                    __m512i acc = _mm512_setzero_si512();
                    const __m512i lowMask = _mm512_set1_epi8(0xf);
                    const __m512i ones = _mm512_set1_epi16(1);

                    int i = 0;
                    for (; i + 63 < m; i += 64) {
                        __m256i orix = _mm256_loadu_si256((const __m256i *) (int4B + i / 2));
                        __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                        __m512i bx = _mm512_and_si512(lowMask, bytex);
                        __m512i by = _mm512_loadu_si512((const __m512i *) (quantizedA + i));
                        acc = _mm512_dpbusd_epi32(acc, bx, by);
                    }
                    sum = _mm512_reduce_add_epi32(acc);

                    // 一次处理两个int4值（假设m是偶数）
                    for (; i < m; i += 2) {
                        uint8_t packedValue = int4B[i / 2];
                        // 提取高4位和低4位
                        uint8_t int4Value0 = (packedValue >> 4) & 0x0F;  // 第一个int4
                        uint8_t int4Value1 = packedValue & 0x0F;         // 第二个int4
                        
                        // 同时计算两个乘积并累加
                        sum += quantizedA[i] * int4Value0 + quantizedA[i + 1] * int4Value1;
                    }

                    floatC[j] = sum * scaleA * scaleB + minB * scaleA * sumA;
                }
            }
            AddBiasAVX512(outputData, biasData, n, k, st, end);
            return true;
#else
            return false;
#endif
        }
#ifdef __AVX512VNNI__
        int i = 0;
        // 一次处理5行
        for (; i + 4 < n; i += 5) {
            LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX512VNNI_Row_Kernel<5>(
                inputData, weightData, biasData, outputData, i, m, k, st, end
            );
        }
        
        // 处理剩余行
        switch (n - i) {
            case 0: break;
            case 1: 
                LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX512VNNI_Row_Kernel<1>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
            case 2: 
                LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX512VNNI_Row_Kernel<2>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
            case 3: 
                LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX512VNNI_Row_Kernel<3>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
            case 4: 
                LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX512VNNI_Row_Kernel<4>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
        }
        
        AddBiasAVX512(outputData, biasData, n, k, st, end);
        return true;
#else
        return false;
#endif
    }

    template <int BROW, int AROW>
    void mul_mat_int8_int4_group128_direct_avx512vnni(
        int m,  // 内积维度
        const uint8_t* A,  // INT8_GROUP128 format (输入数据)
        size_t stride_a,
        const uint8_t* B,  // INT4_GROUP128 format (权重数据)
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX512VNNI__
        constexpr int SIMD_WIDTH = 64;  // AVX512 VNNI 一次处理 64 个 int8
        constexpr int GROUP_SIZE = 128;  // GROUP128的组大小
        
        // 累加器 - 为每个输出元素准备一个累加器
        __m512i acc[AROW * BROW];
        
        // 初始化累加器
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm512_setzero_si512();
        }
        
        const __m512i lowMask = _mm512_set1_epi8(0xf);
        
        // 计算每行需要的组数
        int num_groups = (m + GROUP_SIZE - 1) / GROUP_SIZE;
        
        // 为每个输出元素准备group-wise的累加结果
        std::vector<float> group_results[AROW * BROW];
        for (int i = 0; i < AROW * BROW; ++i) {
            group_results[i].resize(num_groups, 0.0f);
        }
        
        // 按组处理
        for (int group = 0; group < num_groups; ++group) {
            int group_start = group * GROUP_SIZE;
            int group_end = std::min(group_start + GROUP_SIZE, m);
            int group_size = group_end - group_start;
            
            // 重置累加器
            for (int i = 0; i < AROW * BROW; ++i) {
                acc[i] = _mm512_setzero_si512();
            }
            
            // 主循环 - 处理当前组内SIMD_WIDTH的倍数部分
            int nb = group_size / SIMD_WIDTH;
            for (int block = 0; block < nb; ++block) {
                for (int ia = 0; ia < AROW; ++ia) {
                    // A矩阵的第ia行的当前组 (输入数据)
                    const uint8_t* a_row = (const uint8_t*)((const char*)A + ia * stride_a);
                    // 计算当前组的数据起始位置（考虑之前组的metadata）
                    const int8_t* quantizedA = (const int8_t*)(a_row + group * (GROUP_SIZE + 8)) + block * SIMD_WIDTH;
                    __m512i ay = _mm512_loadu_si512((const __m512i*)quantizedA);
                    
                    for (int ib = 0; ib < BROW; ++ib) {
                        // B矩阵的第ib行的当前组 (权重数据，INT4格式)
                        const uint8_t* b_row = (const uint8_t*)((const char*)B + ib * stride_b);
                        // 计算当前组的数据起始位置（INT4每128个值占64字节，加8字节metadata）
                        const uint8_t* int4B = b_row + group * (GROUP_SIZE / 2 + 8) + block * SIMD_WIDTH / 2;
                        __m256i orix = _mm256_loadu_si256((const __m256i*)int4B);
                        __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                        __m512i bx = _mm512_and_si512(lowMask, bytex);
                        
                        int acc_idx = ia * BROW + ib;
                        acc[acc_idx] = _mm512_dpbusd_epi32(acc[acc_idx], bx, ay);
                    }
                }
            }
            
            // 处理当前组内的剩余部分
            int remainder = group_size % SIMD_WIDTH;
            int remainderSums[AROW * BROW] = {0};
            if (remainder > 0) {
                for (int ia = 0; ia < AROW; ++ia) {
                    const uint8_t* a_row = (const uint8_t*)((const char*)A + ia * stride_a);
                    const int8_t* quantizedA = (const int8_t*)(a_row + group * (GROUP_SIZE + 8)) + nb * SIMD_WIDTH;
                    
                    for (int ib = 0; ib < BROW; ++ib) {
                        const uint8_t* b_row = (const uint8_t*)((const char*)B + ib * stride_b);
                        const uint8_t* int4B = b_row + group * (GROUP_SIZE / 2 + 8) + nb * SIMD_WIDTH / 2;
                        
                        for (int i = 0; i < remainder; i += 2) {
                            uint8_t packedValue = int4B[i / 2];
                            uint8_t int4Value0 = (packedValue >> 4) & 0x0F;
                            uint8_t int4Value1 = packedValue & 0x0F;
                            
                            int acc_idx = ia * BROW + ib;
                            remainderSums[acc_idx] += quantizedA[i] * int4Value0;
                            if (i + 1 < remainder) {
                                remainderSums[acc_idx] += quantizedA[i + 1] * int4Value1;
                            }
                        }
                    }
                }
            }
            
            // 对当前组应用scale和bias，存储到group_results中
            for (int ia = 0; ia < AROW; ++ia) {
                const uint8_t* a_row = (const uint8_t*)((const char*)A + ia * stride_a);
                // 当前组的metadata位置
                const uint8_t* a_metadata = a_row + group * (GROUP_SIZE + 8) + GROUP_SIZE;
                float scaleA = *(float*)a_metadata;
                int sumA = *(int*)(a_metadata + sizeof(float));
                
                for (int ib = 0; ib < BROW; ++ib) {
                    const uint8_t* b_row = (const uint8_t*)((const char*)B + ib * stride_b);
                    // 当前组的metadata位置
                    const uint8_t* b_metadata = b_row + group * (GROUP_SIZE / 2 + 8) + GROUP_SIZE / 2;
                    float minB = *(float*)b_metadata;
                    float scaleB = *(float*)(b_metadata + sizeof(float));
                    
                    int acc_idx = ia * BROW + ib;
                    int sum = _mm512_reduce_add_epi32(acc[acc_idx]) + remainderSums[acc_idx];
                    
                    // 存储当前组的结果
                    group_results[acc_idx][group] = sum * scaleA * scaleB + minB * scaleA * sumA;
                }
            }
        }
        
        // 汇总所有组的结果
        for (int ia = 0; ia < AROW; ++ia) {
            float* c_ptr = (float*)((char*)C + ia * stride_c);
            for (int ib = 0; ib < BROW; ++ib) {
                int acc_idx = ia * BROW + ib;
                float total = 0.0f;
                for (int group = 0; group < num_groups; ++group) {
                    total += group_results[acc_idx][group];
                }
                c_ptr[ib] = total;
            }
        }
#endif
    }

    template <int BRow>
    void LinearINT8GROUP128_INT4GROUP128_AVX512VNNI_Row_Kernel(
        uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
        int i, int m, int k, int st, int end) 
    {
        size_t lda = GetDataBytes(DataType::INF_INT8_GROUP128, 1, m);
        size_t ldb = GetDataBytes(DataType::INT4_GROUP128, 1, m);
        size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
        
        int j = st;
        // 一次处理5列
        for (j = st; j + 4 < end; j += 5) {
            mul_mat_int8_int4_group128_direct_avx512vnni<5, BRow>(
                m,
                inputData + i * lda, lda,           // A: 输入数据的第i行开始的BRow行
                weightData + j * ldb, ldb,          // B: 权重数据的第j行开始的5行
                outputData + i * k + j, ldc         // C: 输出位置
            );
        }
        
        // 处理剩余列
        switch (end - j) {
            case 0: break;
            case 1: 
                mul_mat_int8_int4_group128_direct_avx512vnni<1, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
            case 2: 
                mul_mat_int8_int4_group128_direct_avx512vnni<2, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
            case 3: 
                mul_mat_int8_int4_group128_direct_avx512vnni<3, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
            case 4: 
                mul_mat_int8_int4_group128_direct_avx512vnni<4, BRow>(
                    m, inputData + i * lda, lda, weightData + j * ldb, ldb, 
                    outputData + i * k + j, ldc); 
                break;
        }
    }

    bool LinearINT8GROUP128_INT4GROUP128_AVX512VNNI_Kernel(
        uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
        int n, int m, int k, int st, int end) 
    {
        if (n == 1 || true) {
#ifdef __AVX512VNNI__
            constexpr int GROUP_SIZE = 128;
            size_t lda = GetDataBytes(DataType::INF_INT8_GROUP128, 1, m);
            size_t ldb = GetDataBytes(DataType::INT4_GROUP128, 1, m);
            size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
            
            int num_groups = (m + GROUP_SIZE - 1) / GROUP_SIZE;
            
            for (int i = 0; i < n; i++) {
                // A矩阵的第i行，INT8_GROUP128格式
                uint8_t *int8A = (uint8_t*)inputData + i * lda;
                float *floatC = (float*)((uint8_t*)outputData + i * ldc);
                
                for (int j = st; j < end; j++) {
                    // B矩阵的第j行，INT4_GROUP128格式
                    uint8_t *int4B = (uint8_t*)weightData + j * ldb;
                    
                    float total = 0.0f;
                    
                    // 按组处理
                    for (int group = 0; group < num_groups; ++group) {
                        int group_start = group * GROUP_SIZE;
                        int group_end = std::min(group_start + GROUP_SIZE, m);
                        int group_size = group_end - group_start;
                        
                        // 当前组的数据和metadata位置
                        const int8_t* quantizedA = (const int8_t*)(int8A + group * (GROUP_SIZE + 8));
                        const uint8_t* a_metadata = int8A + group * (GROUP_SIZE + 8) + GROUP_SIZE;
                        float scaleA = *(float*)a_metadata;
                        int sumA = *(int*)(a_metadata + sizeof(float));
                        
                        const uint8_t* int4B_group = int4B + group * (GROUP_SIZE / 2 + 8);
                        const uint8_t* b_metadata = int4B_group + GROUP_SIZE / 2;
                        float minB = *(float*)b_metadata;
                        float scaleB = *(float*)(b_metadata + sizeof(float));
                        
                        int sum = 0;
                        __m512i acc = _mm512_setzero_si512();
                        const __m512i lowMask = _mm512_set1_epi8(0xf);
                        
                        int i = 0;
                        for (; i + 63 < group_size; i += 64) {
                            __m256i orix = _mm256_loadu_si256((const __m256i*)(int4B_group + i / 2));
                            __m512i bytex = _mm512_inserti64x4(_mm512_castsi256_si512(orix), _mm256_srli_epi16(orix, 4), 1);
                            __m512i bx = _mm512_and_si512(lowMask, bytex);
                            __m512i by = _mm512_loadu_si512((const __m512i*)(quantizedA + i));
                            acc = _mm512_dpbusd_epi32(acc, bx, by);
                        }
                        sum = _mm512_reduce_add_epi32(acc);
                        
                        // 处理剩余元素
                        for (; i < group_size; i += 2) {
                            uint8_t packedValue = int4B_group[i / 2];
                            uint8_t int4Value0 = (packedValue >> 4) & 0x0F;
                            uint8_t int4Value1 = packedValue & 0x0F;
                            
                            sum += quantizedA[i] * int4Value0;
                            if (i + 1 < group_size) {
                                sum += quantizedA[i + 1] * int4Value1;
                            }
                        }
                        
                        total += sum * scaleA * scaleB + minB * scaleA * sumA;
                    }
                    
                    floatC[j] = total;
                }
            }
            AddBiasAVX512(outputData, biasData, n, k, st, end);
            return true;
#else
            return false;
#endif
        }
#ifdef __AVX512VNNI__
        int i = 0;
        // 一次处理5行
        for (; i + 4 < n; i += 5) {
            LinearINT8GROUP128_INT4GROUP128_AVX512VNNI_Row_Kernel<5>(
                inputData, weightData, biasData, outputData, i, m, k, st, end
            );
        }
        
        // 处理剩余行
        switch (n - i) {
            case 0: break;
            case 1: 
                LinearINT8GROUP128_INT4GROUP128_AVX512VNNI_Row_Kernel<1>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
            case 2: 
                LinearINT8GROUP128_INT4GROUP128_AVX512VNNI_Row_Kernel<2>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
            case 3: 
                LinearINT8GROUP128_INT4GROUP128_AVX512VNNI_Row_Kernel<3>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
            case 4: 
                LinearINT8GROUP128_INT4GROUP128_AVX512VNNI_Row_Kernel<4>(
                    inputData, weightData, biasData, outputData, i, m, k, st, end); 
                break;
        }
        
        AddBiasAVX512(outputData, biasData, n, k, st, end);
        return true;
#else
        return false;
#endif
    }

#ifdef __AVX512VNNI__
    __attribute__((always_inline)) inline __m512i
    UnpackInt4Group32Pair_AVX512VNNI(const uint8_t *packedData) {
        // Two neighboring group-32 payloads are contiguous in every compact
        // four-group block. Unpack both with one 256-bit load and lane-local
        // shuffles, producing 64 unsigned nibble values in their original
        // order. This path is shared by decode and all batched row kernels.
        const __m256i packed = _mm256_loadu_si256(
            (const __m256i*)packedData);
        const __m256i nibbleMask = _mm256_set1_epi8(0x0f);
        const __m256i low = _mm256_and_si256(packed, nibbleMask);
        const __m256i high = _mm256_and_si256(
            _mm256_srli_epi16(packed, 4), nibbleMask);
        const __m256i firstHalf = _mm256_unpacklo_epi8(high, low);
        const __m256i secondHalf = _mm256_unpackhi_epi8(high, low);
        const __m256i group0 = _mm256_permute2x128_si256(
            firstHalf, secondHalf, 0x20);
        const __m256i group1 = _mm256_permute2x128_si256(
            firstHalf, secondHalf, 0x31);
        __m512i unpacked = _mm512_castsi256_si512(group0);
        return _mm512_inserti64x4(unpacked, group1, 1);
    }

    template <int ROWS>
    __attribute__((always_inline)) inline void
    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows(
            const uint8_t *inputData, const uint8_t *weightRow,
            float *biasData, float *outputData, int rowBase, int out,
            int groups, size_t inputRowBytes, int k) {
        constexpr int groupCnt = 32;
        constexpr size_t inputGroupBytes = groupCnt + sizeof(float) + sizeof(int);
        __m512 accum[ROWS];
        float correction[ROWS];
#pragma unroll
        for (int rowOffset = 0; rowOffset < ROWS; rowOffset++) {
            accum[rowOffset] = _mm512_setzero_ps();
            correction[rowOffset] = 0.0f;
        }

        int group = 0;
        for (; group + 1 < groups; group += 2) {
            const uint8_t *weightBlock0 = weightRow +
                GetInt4Group32DataOffset(group, groups);
            const __m512i weights =
                UnpackInt4Group32Pair_AVX512VNNI(weightBlock0);
            const size_t scaleOffset =
                GetInt4Group32ScaleOffset(group, groups);
            const float weightScale0 = BFloat16BitsToFloat32(
                *(const uint16_t*)(weightRow + scaleOffset));
            const float weightScale1 = BFloat16BitsToFloat32(
                *(const uint16_t*)(weightRow + scaleOffset +
                                   sizeof(uint16_t)));

#pragma unroll
            for (int rowOffset = 0; rowOffset < ROWS; rowOffset++) {
                const uint8_t *inputRow = inputData +
                    (size_t)(rowBase + rowOffset) * inputRowBytes;
                const uint8_t *input0 = inputRow +
                    (size_t)group * inputGroupBytes;
                const uint8_t *input1 = input0 + inputGroupBytes;
                const __m256i a0 = _mm256_loadu_si256(
                    (const __m256i*)input0);
                const __m256i a1 = _mm256_loadu_si256(
                    (const __m256i*)input1);
                __m512i activations = _mm512_castsi256_si512(a0);
                activations = _mm512_inserti64x4(activations, a1, 1);
                const __m512i dots = _mm512_dpbusd_epi32(
                    _mm512_setzero_si512(), weights, activations);

                const float scale0 =
                    *(const float*)(input0 + groupCnt) * weightScale0;
                const float scale1 =
                    *(const float*)(input1 + groupCnt) * weightScale1;
                const __m512 coefficients = _mm512_mask_blend_ps(
                    (__mmask16)0xff00, _mm512_set1_ps(scale0),
                    _mm512_set1_ps(scale1));
                accum[rowOffset] = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(dots), coefficients,
                    accum[rowOffset]);

                const int sum0 = *(const int*)(
                    input0 + groupCnt + sizeof(float));
                const int sum1 = *(const int*)(
                    input1 + groupCnt + sizeof(float));
                correction[rowOffset] -=
                    8.0f * ((float)sum0 * scale0 + (float)sum1 * scale1);
            }
        }

#pragma unroll
        for (int rowOffset = 0; rowOffset < ROWS; rowOffset++) {
            float total = _mm512_reduce_add_ps(accum[rowOffset]) +
                          correction[rowOffset];
            if (group < groups) {
                const uint8_t *inputRow = inputData +
                    (size_t)(rowBase + rowOffset) * inputRowBytes;
                const uint8_t *inputGroup = inputRow +
                    (size_t)group * inputGroupBytes;
                const int8_t *values = (const int8_t*)inputGroup;
                const uint8_t *packed = weightRow +
                    GetInt4Group32DataOffset(group, groups);
                int dot = 0;
#pragma unroll
                for (int i = 0; i < groupCnt; i += 2) {
                    dot += values[i] * (packed[i / 2] >> 4) +
                           values[i + 1] * (packed[i / 2] & 0x0f);
                }
                const float scale = *(const float*)(inputGroup + groupCnt) *
                    BFloat16BitsToFloat32(
                        *(const uint16_t*)(weightRow +
                            GetInt4Group32ScaleOffset(group, groups)));
                const int sum = *(const int*)(
                    inputGroup + groupCnt + sizeof(float));
                total += (dot - 8 * sum) * scale;
            }
            if (biasData != nullptr) {
                total += biasData[out];
            }
            outputData[(size_t)(rowBase + rowOffset) * k + out] = total;
        }
    }

    template <int ROWS>
    __attribute__((always_inline)) inline void
    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows2(
            const uint8_t *inputData, const uint8_t *weightRow0,
            const uint8_t *weightRow1, float *biasData, float *outputData,
            int rowBase, int out, int groups, size_t inputRowBytes, int k) {
        constexpr int groupCnt = 32;
        constexpr size_t inputGroupBytes = groupCnt + sizeof(float) + sizeof(int);
        __m512 accum0[ROWS], accum1[ROWS];
        float correction0[ROWS], correction1[ROWS];
#pragma unroll
        for (int rowOffset = 0; rowOffset < ROWS; rowOffset++) {
            accum0[rowOffset] = _mm512_setzero_ps();
            accum1[rowOffset] = _mm512_setzero_ps();
            correction0[rowOffset] = 0.0f;
            correction1[rowOffset] = 0.0f;
        }

        // The two-output specialization is used only for even group counts;
        // Laguna's 2048/3072-wide expert matrices satisfy this naturally.
        for (int group = 0; group < groups; group += 2) {
            const size_t dataOffset0 =
                GetInt4Group32DataOffset(group, groups);
            const __m512i weights0 =
                UnpackInt4Group32Pair_AVX512VNNI(weightRow0 + dataOffset0);
            const __m512i weights1 =
                UnpackInt4Group32Pair_AVX512VNNI(weightRow1 + dataOffset0);

            const size_t scaleOffset0 =
                GetInt4Group32ScaleOffset(group, groups);
            const size_t scaleOffset1 = scaleOffset0 + sizeof(uint16_t);
            const float weightScale00 = BFloat16BitsToFloat32(
                *(const uint16_t*)(weightRow0 + scaleOffset0));
            const float weightScale01 = BFloat16BitsToFloat32(
                *(const uint16_t*)(weightRow0 + scaleOffset1));
            const float weightScale10 = BFloat16BitsToFloat32(
                *(const uint16_t*)(weightRow1 + scaleOffset0));
            const float weightScale11 = BFloat16BitsToFloat32(
                *(const uint16_t*)(weightRow1 + scaleOffset1));

#pragma unroll
            for (int rowOffset = 0; rowOffset < ROWS; rowOffset++) {
                const uint8_t *inputRow = inputData +
                    (size_t)(rowBase + rowOffset) * inputRowBytes;
                const uint8_t *input0 = inputRow +
                    (size_t)group * inputGroupBytes;
                const uint8_t *input1 = input0 + inputGroupBytes;
                const __m256i a0 = _mm256_loadu_si256(
                    (const __m256i*)input0);
                const __m256i a1 = _mm256_loadu_si256(
                    (const __m256i*)input1);
                __m512i activations = _mm512_castsi256_si512(a0);
                activations = _mm512_inserti64x4(activations, a1, 1);
                const __m512i dots0 = _mm512_dpbusd_epi32(
                    _mm512_setzero_si512(), weights0, activations);
                const __m512i dots1 = _mm512_dpbusd_epi32(
                    _mm512_setzero_si512(), weights1, activations);

                const float inputScale0 = *(const float*)(input0 + groupCnt);
                const float inputScale1 = *(const float*)(input1 + groupCnt);
                const float scale00 = inputScale0 * weightScale00;
                const float scale01 = inputScale1 * weightScale01;
                const float scale10 = inputScale0 * weightScale10;
                const float scale11 = inputScale1 * weightScale11;
                const __m512 coefficients0 = _mm512_mask_blend_ps(
                    (__mmask16)0xff00, _mm512_set1_ps(scale00),
                    _mm512_set1_ps(scale01));
                const __m512 coefficients1 = _mm512_mask_blend_ps(
                    (__mmask16)0xff00, _mm512_set1_ps(scale10),
                    _mm512_set1_ps(scale11));
                accum0[rowOffset] = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(dots0), coefficients0,
                    accum0[rowOffset]);
                accum1[rowOffset] = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(dots1), coefficients1,
                    accum1[rowOffset]);

                const int sum0 = *(const int*)(
                    input0 + groupCnt + sizeof(float));
                const int sum1 = *(const int*)(
                    input1 + groupCnt + sizeof(float));
                correction0[rowOffset] -=
                    8.0f * ((float)sum0 * scale00 + (float)sum1 * scale01);
                correction1[rowOffset] -=
                    8.0f * ((float)sum0 * scale10 + (float)sum1 * scale11);
            }
        }

        const float bias0 = biasData == nullptr ? 0.0f : biasData[out];
        const float bias1 = biasData == nullptr ? 0.0f : biasData[out + 1];
#pragma unroll
        for (int rowOffset = 0; rowOffset < ROWS; rowOffset++) {
            outputData[(size_t)(rowBase + rowOffset) * k + out] =
                _mm512_reduce_add_ps(accum0[rowOffset]) +
                correction0[rowOffset] + bias0;
            outputData[(size_t)(rowBase + rowOffset) * k + out + 1] =
                _mm512_reduce_add_ps(accum1[rowOffset]) +
                correction1[rowOffset] + bias1;
        }
    }
#endif

    bool LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Kernel(
        uint8_t *inputData, uint8_t *weightData, float *biasData,
        float *outputData, int n, int m, int k, int st, int end) {
#ifdef __AVX512VNNI__
        if (m <= 0 || m % 32 != 0) {
            return false;
        }
        constexpr int groupCnt = 32;
        constexpr size_t inputGroupBytes = groupCnt + sizeof(float) + sizeof(int);
        const int groups = m / groupCnt;
        const size_t inputRowBytes = GetDataBytes(DataType::INF_INT8_GROUP32, 1, m);
        const size_t weightRowBytes = GetDataBytes(DataType::INT4_GROUP32, 1, m);
        if (n == 1) {
            for (int out = st; out < end; out++) {
                const uint8_t *weightRow =
                    weightData + (size_t)out * weightRowBytes;
                LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows<1>(
                    inputData, weightRow, biasData, outputData, 0, out,
                    groups, inputRowBytes, k);
            }
            return true;
        }

        // For batched rows, block two output rows together so both dot products
        // reuse activation loads, scales and sums. Both this path and the
        // dedicated n == 1 path above share the compact pair-unpack helper.
        constexpr int inputBlock = 8;
        int out = st;
        if ((groups & 1) == 0) {
            for (; out + 1 < end; out += 2) {
                const uint8_t *weightRow0 =
                    weightData + (size_t)out * weightRowBytes;
                const uint8_t *weightRow1 = weightRow0 + weightRowBytes;
                int rowBase = 0;
                for (; rowBase + inputBlock <= n; rowBase += inputBlock) {
                    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows2<inputBlock>(
                        inputData, weightRow0, weightRow1, biasData, outputData,
                        rowBase, out, groups, inputRowBytes, k);
                }
                switch (n - rowBase) {
                    case 0: break;
                    case 1:
                        LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows2<1>(
                            inputData, weightRow0, weightRow1, biasData,
                            outputData, rowBase, out, groups, inputRowBytes, k);
                        break;
                    case 2:
                        LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows2<2>(
                            inputData, weightRow0, weightRow1, biasData,
                            outputData, rowBase, out, groups, inputRowBytes, k);
                        break;
                    case 3:
                        LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows2<3>(
                            inputData, weightRow0, weightRow1, biasData,
                            outputData, rowBase, out, groups, inputRowBytes, k);
                        break;
                    case 4:
                        LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows2<4>(
                            inputData, weightRow0, weightRow1, biasData,
                            outputData, rowBase, out, groups, inputRowBytes, k);
                        break;
                    case 5:
                        LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows2<5>(
                            inputData, weightRow0, weightRow1, biasData,
                            outputData, rowBase, out, groups, inputRowBytes, k);
                        break;
                    case 6:
                        LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows2<6>(
                            inputData, weightRow0, weightRow1, biasData,
                            outputData, rowBase, out, groups, inputRowBytes, k);
                        break;
                    case 7:
                        LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows2<7>(
                            inputData, weightRow0, weightRow1, biasData,
                            outputData, rowBase, out, groups, inputRowBytes, k);
                        break;
                }
            }
        }
        for (; out < end; out++) {
            const uint8_t *weightRow = weightData + (size_t)out * weightRowBytes;
            int rowBase = 0;
            for (; rowBase + inputBlock <= n; rowBase += inputBlock) {
                LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows<inputBlock>(
                    inputData, weightRow, biasData, outputData, rowBase, out,
                    groups, inputRowBytes, k);
            }
            switch (n - rowBase) {
                case 0: break;
                case 1:
                    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows<1>(
                        inputData, weightRow, biasData, outputData, rowBase, out,
                        groups, inputRowBytes, k);
                    break;
                case 2:
                    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows<2>(
                        inputData, weightRow, biasData, outputData, rowBase, out,
                        groups, inputRowBytes, k);
                    break;
                case 3:
                    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows<3>(
                        inputData, weightRow, biasData, outputData, rowBase, out,
                        groups, inputRowBytes, k);
                    break;
                case 4:
                    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows<4>(
                        inputData, weightRow, biasData, outputData, rowBase, out,
                        groups, inputRowBytes, k);
                    break;
                case 5:
                    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows<5>(
                        inputData, weightRow, biasData, outputData, rowBase, out,
                        groups, inputRowBytes, k);
                    break;
                case 6:
                    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows<6>(
                        inputData, weightRow, biasData, outputData, rowBase, out,
                        groups, inputRowBytes, k);
                    break;
                case 7:
                    LinearINT8GROUP32_INT4GROUP32_AVX512VNNI_Rows<7>(
                        inputData, weightRow, biasData, outputData, rowBase, out,
                        groups, inputRowBytes, k);
                    break;
            }
        }
        return true;
#else
        return false;
#endif
    }

}
