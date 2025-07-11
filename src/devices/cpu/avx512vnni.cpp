//
// Created by huangyuyang on 5/20/25.
//

#include <cstdint>
#include <algorithm>

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
}