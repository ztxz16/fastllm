//
// Created by huangyuyang on 8/14/24.
//

#include "devices/cpu/computeutils.h"

#include <cstring>
#include <thread>
#include <cfloat>
#include <cmath>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#include "utils.h"

namespace fastllm {
    extern FP16ToFP32Manager fp16tofp32;
    extern void Float16ToFloat32(uint16_t *float16, float *float32, int len);
    extern void Float32ToFloat16(float *float32, uint16_t *float16, int len);

#ifdef __AVX2__
    int DotU8U8(uint8_t *a, uint8_t *b, int n) {
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        int ans = 0;
        const __m256i lowMask = _mm256_set1_epi8(0xf);
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i ones8 = _mm256_set1_epi8(1);
        const __m256i xors = _mm256_set1_epi8(-128);
        for (; i + 31 < n; i += 32) {
            __m256i bx = _mm256_loadu_si256((const __m256i *) (a + i));
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));

            by = _mm256_xor_si256(by, xors);
            by = _mm256_add_epi8(by, _mm256_and_si256(_mm256_cmpeq_epi8(by, xors), ones8));

            by = _mm256_sign_epi8(by, bx);
            bx = _mm256_sign_epi8(bx, bx);

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(bx, by), ones));
        }
        for (; i < n; i++) {
            ans += ((int8_t*)a)[i] * ((int)b[i] - 128);
        }

        return ans + I32sum(acc);
    };
#endif

    void MatMulFloat16Float16(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData, 
                            int n, int m, int k, int st, int end) {
#ifdef __aarch64__
        if (n > 1 && n < 8) {
            const int BLOCKA = 7;
            float16x8_t va[BLOCKA], vb, vc[BLOCKA];
        
            for (int i = 0; i < n; i += BLOCKA) {
                int cur = std::min(BLOCKA, n - i);
                for (int j = st; j < end; j++) {
                    for (int l1 = 0; l1 < cur; l1++) {
                        vc[l1] = vdupq_n_f16(0.0f);
                    }
                        
                    for (int k = 0; k < m; k += 8) {
                        for (int l = 0; l < cur; l++) {
                            va[l] = vld1q_f16((float16_t*)inputData + (i + l) * m + k);
                        }
                        vb = vld1q_f16((float16_t*)weightData + j * m + k);
                                    
                        for (int l1 = 0; l1 < cur; l1++) {
                            vc[l1] = vfmaq_f16(vc[l1], va[l1], vb);
                        }
                    }

                    for (int l0 = 0; l0 < cur; l0++) {
                        float now = vc[l0][0] + vc[l0][1] + vc[l0][2] + vc[l0][3] + 
                                    vc[l0][4] + vc[l0][5] + vc[l0][6] + vc[l0][7];
                        if (biasData != nullptr) {
                            now += biasData[j];
                        }
                        outputData[(i + l0) * k + j] = float_to_half(now);
                    }
                }
            }
        } else if (n > 1) {
            const int BN = 64, BM = 64, BK = 64;
            uint16_t *a = new uint16_t[BN * BM];
            uint16_t *b = new uint16_t[BK * BM];
            uint16_t *c = new uint16_t[BN * BK];
            if (biasData == nullptr) {
                for (int i = 0; i < n; i++) {
                    memset(outputData + st, 0, (end - st) * sizeof(uint16_t));
                }
            } else {
                for (int i = 0; i < n; i++) {
                    for (int j = st; j < end; j++) {
                        outputData[i * k + j] = float_to_half(biasData[j]);
                    }
                }
            }

            for (int mst = 0; mst < m; mst += BM) {
                int mend = std::min(mst + BM, m);
                memset(c, 0, BN * BK * sizeof(uint16_t));
                for (int nst = 0; nst < n; nst += BN) {
                    int nend = std::min(nst + BN, n);
                    for (int i = nst; i < nend; i++) {
                        memcpy(a + (i - nst) * BM, inputData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                    }
                    for (int kst = st; kst < end; kst += BK) {
                        int kend = std::min(kst + BK, end);
                        for (int i = kst; i < kend; i++) {
                            memcpy(b + (i - kst) * BM, weightData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                        }

                        const int BLOCKB = 8;
                        float16x8_t va, vb[BLOCKB], vc[BLOCKB];

                        for (int i = 0; i < BN && i < (nend - nst); i++) {
                            for (int j = 0; j < BK; j += BLOCKB) {
                                for (int l1 = 0; l1 < BLOCKB; l1++) {
                                    vc[l1] = vdupq_n_f16(0.0f);
                                }

                                for (int k = 0; k < BM; k += 8) {
                                    va = vld1q_f16((float16_t*)a + i * BM + k);
                                    for (int l = 0; l < BLOCKB; l++) {
                                        vb[l] = vld1q_f16((float16_t*)b + (j + l) * BM + k);
                                    }
                                    
                                    for (int l1 = 0; l1 < BLOCKB; l1++) {
                                        vc[l1] = vfmaq_f16(vc[l1], va, vb[l1]);
                                    }
                                }

                                float16x8x2_t temp0 = vtrnq_f16(vc[0], vc[1]);
                                float16x8x2_t temp1 = vtrnq_f16(vc[2], vc[3]);
                                float16x8x2_t temp2 = vtrnq_f16(vc[4], vc[5]);
                                float16x8x2_t temp3 = vtrnq_f16(vc[6], vc[7]);

                                vc[0] = vaddq_f16(temp0.val[0], temp0.val[1]);
                                vc[1] = vaddq_f16(temp1.val[0], temp1.val[1]);
                                vc[2] = vaddq_f16(temp2.val[0], temp2.val[1]);
                                vc[3] = vaddq_f16(temp3.val[0], temp3.val[1]);

                                float32x4x2_t temp4 = vtrnq_f32(vreinterpretq_f32_f16(vc[0]), vreinterpretq_f32_f16(vc[1]));
                                float32x4x2_t temp5 = vtrnq_f32(vreinterpretq_f32_f16(vc[2]), vreinterpretq_f32_f16(vc[3]));

                                vc[0] = vaddq_f16(vreinterpretq_f16_f32(temp4.val[0]), vreinterpretq_f16_f32(temp4.val[1]));
                                vc[1] = vaddq_f16(vreinterpretq_f16_f32(temp5.val[0]), vreinterpretq_f16_f32(temp5.val[1]));

                                vst1q_f16((float16_t*)c + i * BK + j, vcombine_f16(vadd_f16(vget_high_f16(vc[0]), vget_low_f16(vc[0])), vadd_f16(vget_high_f16(vc[1]), vget_low_f16(vc[1]))));

                                /*for (int l0 = 0; l0 < BLOCKA; l0++) {
                                        for (int l1 = 0; l1 < BLOCKB; l1++) {
                                            float now = vc[l0][l1][0] + vc[l0][l1][1] + vc[l0][l1][2] + vc[l0][l1][3] + 
                                                vc[l0][l1][4] + vc[l0][l1][5] + vc[l0][l1][6] + vc[l0][l1][7];
                                            c[(i + l0) * block + (j + l1)] = float_to_half(now);
                                        }
                                }*/
                            }
                        }
/*
                            for (int i = 0; i < block && i < (nend - nst); i++) {
                                for (int j = 0; j < block; j++) {
                                    float now = 0.0;
                                    int k = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                    float16x8_t sum = {0, 0, 0, 0, 0, 0, 0, 0};
                                    for (; k + 7 < block; k += 8) {
                                        sum = vfmaq_f16(sum, vld1q_f16((float16_t*)a + i * block + k),
                                                            vld1q_f16((float16_t*)b + j * block + k));
                                    }
                                    now += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#endif
                                    for (; k < block; k++) {
                                        now += fp16tofp32.dict[a[i * block + k]] * fp16tofp32.dict[b[j * block + k]];
                                    }
                                    c[i * block + j] = float_to_half(now);
                                }
                            }
*/

                        for (int i = nst; i < nend; i++) {
                            int j = kst;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                            for (; j + 7 < kend; j += 8) {
                                vst1q_f16((float16_t*)outputData + i * k + j, vaddq_f16(vld1q_f16((float16_t*)outputData + i * k + j), vld1q_f16((float16_t*)c + (i - nst) * BK + (j - kst))));
                            }
#endif
                            for (; j < kend; j++) {
                                outputData[i * k + j] = (float_to_half)(fp16tofp32.dict[outputData[i * k + j]] + fp16tofp32.dict[c[(i - nst) * BK + (j - kst)]]);
                            }
                        }
                    }
                }
            }
            delete[] a;
            delete[] b;
            delete[] c;
/*
                for (int i = 0; i < n; i++) {
                    for (int j = st; j < end; j++) {
                        float now = biasData ? biasData[j] : 0.0f;
                        for (int l = 0; l < m; l++) {
                            now += fp16tofp32.dict[inputData[i * m + l]] * fp16tofp32.dict[weightData[j * m + l]];
                        }
{
    float a = half_to_float(outputData[i * k + j]);
    if (st == 0 && fabs(a - now) > 1e-1) {
        printf("wrong %d %d %f %f\n", i, j, a, now);
        exit(0);
    }
}
                        outputData[i * k + j] = float_to_half(now);
                    }
                }
*/
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float16x8_t sum = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (; l + 7 < m; l += 8) {
                        sum = vfmaq_f16(sum, vld1q_f16((float16_t*)inputData + i * m + l),
                                            vld1q_f16((float16_t*)weightData + j * m + l));
                    }
                    now += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#endif
                    for (; l < m; l++) {
                        now += fp16tofp32.dict[inputData[i * m + l]] * fp16tofp32.dict[weightData[j * m + l]];
                    }
                    outputData[i * k + j] = float_to_half(now);
                }
            }
        }
#elif defined(__AVX2__)
        if (n > 8) {
            const int BN = 64, BM = 128, BK = 64;
            uint16_t *a = new uint16_t[BN * BM];
            uint16_t *b = new uint16_t[BK * BM];
            uint16_t *c = new uint16_t[BN * BK];
            if (biasData == nullptr) {
                for (int i = 0; i < n; i++) {
                    memset(outputData + st, 0, (end - st) * sizeof(uint16_t));
                }
            } else {
                for (int i = 0; i < n; i++) {
                    for (int j = st; j < end; j++) {
                        outputData[i * k + j] = float_to_half(biasData[j]);
                    }
                }
            }

            for (int mst = 0; mst < m; mst += BM) {
                int mend = std::min(mst + BM, m);
                memset(c, 0, BN * BK * sizeof(uint16_t));
                for (int nst = 0; nst < n; nst += BN) {
                    int nend = std::min(nst + BN, n);
                    for (int i = nst; i < nend; i++) {
                        memcpy(a + (i - nst) * BM, inputData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                    }
                    for (int kst = st; kst < end; kst += BK) {
                        int kend = std::min(kst + BK, end);
                        for (int i = kst; i < kend; i++) {
                            memcpy(b + (i - kst) * BM, weightData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                        }

                        const int BLOCKB = 8;
                        __m256 va, vb[BLOCKB], vc[BLOCKB];

                        for (int i = 0; i < BN && i < (nend - nst); i++) {
                            for (int j = 0; j < BK; j += BLOCKB) {
                                for (int l1 = 0; l1 < BLOCKB; l1++) {
                                    vc[l1] = _mm256_setzero_ps();
                                }

                                for (int k = 0; k < BM; k += 8) {
                                    va = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (a + i * BM + k)));
                                    for (int l = 0; l < BLOCKB; l++) {
                                        vb[l] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (b + (j + l) * BM + k)));
                                    }
                                    
                                    for (int l1 = 0; l1 < BLOCKB; l1++) {
                                        vc[l1] = _mm256_fmadd_ps(va, vb[l1], vc[l1]);
                                    }
                                }

                                __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
                                __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;

                                __t0 = _mm256_unpacklo_ps(vc[0], vc[1]);
                                __t1 = _mm256_unpackhi_ps(vc[0], vc[1]);
                                __t2 = _mm256_unpacklo_ps(vc[2], vc[3]);
                                __t3 = _mm256_unpackhi_ps(vc[2], vc[3]);
                                __t4 = _mm256_unpacklo_ps(vc[4], vc[5]);
                                __t5 = _mm256_unpackhi_ps(vc[4], vc[5]);
                                __t6 = _mm256_unpacklo_ps(vc[6], vc[7]);
                                __t7 = _mm256_unpackhi_ps(vc[6], vc[7]);
                                __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
                                __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
                                __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
                                __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
                                __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
                                __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
                                __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
                                __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
                                
                                __m256 sum = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt1, __tt5, 0x20));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt2, __tt6, 0x20));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt3, __tt7, 0x20));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt0, __tt4, 0x31));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt1, __tt5, 0x31));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt2, __tt6, 0x31));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt3, __tt7, 0x31));
                                
                                _mm_storeu_si128 ((__m128i*)(c + i * BK + j), _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
                            }
                        }

                        for (int i = nst; i < nend; i++) {
                            int j = kst;
                            for (; j + 7 < kend; j += 8) {
                                __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (outputData + i * k + j)));
                                __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (c + (i - nst) * BK + (j - kst))));
                                __m256 vsum = _mm256_add_ps(va, vb);
                                _mm_storeu_si128 ((__m128i*)(outputData + i * k + j), _mm256_cvtps_ph(vsum, _MM_FROUND_TO_NEAREST_INT));
                            }
                            for (; j < kend; j++) {
                                outputData[i * k + j] = (float_to_half)(fp16tofp32.dict[outputData[i * k + j]] + fp16tofp32.dict[c[(i - nst) * BK + (j - kst)]]);
                            }
                        }
                    }
                }
            }
            delete[] a;
            delete[] b;
            delete[] c;
        } else if (n > 0) {
            const int BLOCKA = 8;
            __m256 va[BLOCKA], vb, vc[BLOCKA];
        
            for (int i = 0; i < n; i += BLOCKA) {
                int cur = std::min(BLOCKA, n - i);
                for (int j = st; j < end; j++) {
                    for (int l1 = 0; l1 < cur; l1++) {
                        vc[l1] = _mm256_setzero_ps();
                    }
                        
                    for (int k = 0; k < m; k += 8) {
                        for (int l = 0; l < cur; l++) {
                            va[l] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (inputData + (i + l) * m + k)));
                        }
                        vb = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (weightData + j * m + k)));
                                    
                        for (int l1 = 0; l1 < cur; l1++) {
                            vc[l1] = _mm256_fmadd_ps(va[l1], vb, vc[l1]);
                        }
                    }

                    for (int l0 = 0; l0 < cur; l0++) {
                        float now = Floatsum(vc[l0]);
                        if (biasData != nullptr) {
                            now += biasData[j];
                        }
                        outputData[(i + l0) * k + j] = float_to_half(now);
                    }
                }
            }
        }
#else
        if (n > 3) {
            const int BN = 64, BM = 64, BK = 64;
            float *a = new float[BN * BM];
            float *b = new float[BK * BM];
            float *c = new float[BN * BK];
            if (biasData == nullptr) {
                for (int i = 0; i < n; i++) {
                    memset(outputData + st, 0, (end - st) * sizeof(uint16_t));
                }
            } else {
                for (int i = 0; i < n; i++) {
                    for (int j = st; j < end; j++) {
                        outputData[i * k + j] = float_to_half(biasData[j]);
                    }
                }
            }

            for (int mst = 0; mst < m; mst += BM) {
                int mend = std::min(mst + BM, m);
                memset(c, 0, BN * BK * sizeof(float));
                for (int nst = 0; nst < n; nst += BN) {
                    int nend = std::min(nst + BN, n);
                    for (int i = nst; i < nend; i++) {
                        Float16ToFloat32(inputData + i * m + mst, a + (i - nst) * BM, mend - mst);
                    }
                    for (int kst = st; kst < end; kst += BK) {
                        int kend = std::min(kst + BK, end);
                        for (int i = kst; i < kend; i++) {
                            Float16ToFloat32(weightData + i * m + mst, b + (i - kst) * BM, mend - mst);
                        }
                        
                        for (int i = 0; i < BN && i < (nend - nst); i++) {
                            for (int j = 0; j < BK; j++) {
                                float now = 0.0;
                                for (int k = 0; k < BM; k++) {
                                    now += a[i * BM + k] * b[j * BM + k];
                                }
                                c[i * BK + j] = now;
                            }
                        }

                        for (int i = nst; i < nend; i++) {
                            int j = kst;
                            for (; j < kend; j++) {
                                outputData[i * k + j] = (float_to_half)(fp16tofp32.dict[outputData[i * k + j]] + c[(i - nst) * BK + (j - kst)]);
                            }
                        }
                    }
                }
            }
            delete[] a;
            delete[] b;
            delete[] c;
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
                    for (; l < m; l++) {
                        now += fp16tofp32.dict[inputData[i * m + l]] * fp16tofp32.dict[weightData[j * m + l]];
                    }
                    outputData[i * k + j] = float_to_half(now);
                }
            }
        }
#endif
    }

#ifdef __ARM_FEATURE_DOTPROD
        inline static void MatMulInt8Int8RunSomeBlock(uint8_t *weightWalk, uint8_t *inputStart, int32_t *c, 
                            int curBlock, uint32x4_t *sum, uint8x16_t *vi, 
                            int block, int k, int m, int kstride) {
                for (int i = 0; i < k; i++) {
                    std::vector <int> values = std::vector <int> (curBlock, 0);
                    uint8_t *inputWalk = inputStart;
                    int j = 0;

                    for (int j = 0; j < curBlock; j++) {
                        sum[j][0] = sum[j][1] = sum[j][2] = sum[j][3] = 0;
                    }
                    for (; j + 15 < m; j += 16) {
                        for (int x = 0; x < curBlock; x++) {
                            vi[x] = vld1q_u8(inputWalk + m * x);
                        }
                        uint8x16_t vw = vld1q_u8(weightWalk);
                        for (int x = 0; x < curBlock; x++) {
                            sum[x] = vdotq_u32(sum[x], vi[x], vw);
                        }
                        inputWalk += 16;
                        weightWalk += 16;
                    }
                    for (int x = 0; x < curBlock; x++) {
                        values[x] += sum[x][0] + sum[x][1] + sum[x][2] + sum[x][3];
                    }

                    for (; j < m; j++) {
                        int curWeight = (int)(*(weightWalk++));
                        for (int x = 0; x < curBlock; x++) {
                            values[x] += curWeight * (*(inputWalk + x * m));
                        }
                        inputWalk++;
                    }

                    for (int x = 0; x < curBlock; x++) {
                        c[(block + x) * kstride + i] = values[x];
                    }
                }
        }
#endif

    void MatMulInt8Int8(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride) {
#ifdef __ARM_FEATURE_DOTPROD
#define RUNBLOCK(x) for (; block + (x - 1) < n; block += (x)) MatMulInt8Int8RunSomeBlock(b, a + block * m, c, (x), sum, vi, block, k, m, kstride);
            int block = 0;
            uint32x4_t sum[16];
            uint8x16_t vi[16];
            RUNBLOCK(16);
            RUNBLOCK(8);RUNBLOCK(7);RUNBLOCK(6);RUNBLOCK(5);
            RUNBLOCK(4);RUNBLOCK(3);RUNBLOCK(2);RUNBLOCK(1);
#undef RUNBLOCK
#elif defined(__aarch64__)
            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;

                    int per = 64;
                    int cnt = m / per;
                    int sur = m % per;

                    uint32x4_t sum = {0};
                    uint16x8_t temp = {0};
                    uint16x8_t temp1 = {0};
                    uint16x8_t temp2 = {0};
                    uint16x8_t temp3 = {0};
                    uint16x8_t temp4 = {0};
                    uint16x8_t temp5 = {0};
                    uint16x8_t temp6 = {0};
                    uint16x8_t temp7 = {0};

                    while (cnt--) {
                        temp = vmull_u8(vld1_u8(inputWalk), vld1_u8(weightWalk));
                        temp1 = vmull_u8(vld1_u8(inputWalk + 8), vld1_u8(weightWalk + 8));
                        temp2 = vmull_u8(vld1_u8(inputWalk + 16), vld1_u8(weightWalk + 16));
                        temp3 = vmull_u8(vld1_u8(inputWalk + 24), vld1_u8(weightWalk + 24));
                        temp4 = vmull_u8(vld1_u8(inputWalk + 32), vld1_u8(weightWalk + 32));
                        temp5 = vmull_u8(vld1_u8(inputWalk + 40), vld1_u8(weightWalk + 40));
                        temp6 = vmull_u8(vld1_u8(inputWalk + 48), vld1_u8(weightWalk + 48));
                        temp7 = vmull_u8(vld1_u8(inputWalk + 56), vld1_u8(weightWalk + 56));

                        sum = vpadalq_u16(sum, temp);
                        sum = vpadalq_u16(sum, temp1);
                        sum = vpadalq_u16(sum, temp2);
                        sum = vpadalq_u16(sum, temp3);
                        sum = vpadalq_u16(sum, temp4);
                        sum = vpadalq_u16(sum, temp5);
                        sum = vpadalq_u16(sum, temp6);
                        sum = vpadalq_u16(sum, temp7);

                        inputWalk += per;
                        weightWalk += per;
                    }

                    value += (sum[0] + sum[1] + sum[2] + sum[3]);
                    while (sur--) {
                        value += (int)(*(weightWalk++)) * (*(inputWalk++));
                    }

                    c[block * kstride + i] = value;
                }
            }
#elif defined(__AVX2__)
            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    uint8_t *inputWalk = inputStart;

                    c[block * kstride + i] = DotU8U8(inputWalk, weightWalk, m);
                    weightWalk += m;
                }
            }
#else
            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;
                    for (int j = 0; j < m; j++) {
                        value += (int)(*(weightWalk++)) * (*(inputWalk++));
                    }

                    c[block * kstride + i] = value;
                }
            }
#endif
    }
}