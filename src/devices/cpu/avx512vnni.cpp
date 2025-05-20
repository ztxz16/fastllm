//
// Created by huangyuyang on 5/20/25.
//

#include <cstdint>
#include "immintrin.h"

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
#else
        printf("Wrong: need AVX512VNNI.\n");
        exit(0);
#endif
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
}