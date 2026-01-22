//
// Created by huangyuyang on 1/21/26.
//

#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#ifdef USE_ROCM
#include "fastllm-hip.h"
#endif

// FlashInfer includes
#include "attention_impl.cuh"
#include "attention/default_prefill_params.cuh"
#include "attention/variants.cuh"
#include "attention/mask.cuh"
#include "pos_enc.cuh"
#include "utils.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
#include "mma.h"
using namespace nvcuda;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
#define CUDA_NO_TENSOR_CORE
#endif

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)
extern void showError(cudaError_t result, char const* const message, const char* const file, int const line);

extern void *FastllmCudaPrepareInput(const fastllm::Data &input);
extern void *FastllmCudaPrepareOutput(fastllm::Data &output);
extern void FastllmCudaFinishInput(const fastllm::Data &input, void *data);
extern void FastllmCudaFinishOutput(fastllm::Data &output, void *data);
extern cublasHandle_t getFastllmCublasHandle();

template <int BN, int BM, int BK>
__global__ void HalfFC(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int N, const int M, const int K,
    half scale, const int base) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int wid = tid >> 5;

    int stN = bx * BN;
    int stK = by * BK;
    int wrap0 = wid >> 1;
    int wrap1 = wid & 1;

    if (base + stN + BN <= stK) {
        return;
    }

    __shared__ half cur[BN][BK];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4][8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[4][8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(frag_a[i][j], &a[(stN + wrap0 * 64 + i * 16) * M + j * 16], M);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(frag_b[i][j], &b[(stK + wrap1 * 64 + i * 16) * M + j * 16], M);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                wmma::mma_sync(frag_c[i][j], frag_a[i][k], frag_b[j][k], frag_c[i][j]);
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&cur[(wrap0 * 64 + i * 16)][(wrap1 * 64 + j * 16)], frag_c[i][j], BK, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = 0; i < BN; i++) {
        if (base + stN + i < stK + tid) {
            cur[i][tid] = (half)0;
        }
    }

    for (int i = 0; i < BN; i++) {
        c[(stN + i) * K + stK + tid] = __hmul(cur[i][tid], scale);
    }
#endif
}

void GpuQK(half *q, half *k, half *qk, int qlen, int klen, int dim, float scale, int base) {    
    const int BQ = 128, BK = 128, DIM = 128;
    dim3 blockDim(128);
    int BX = (qlen + BQ - 1) / BQ;
    int BY = (klen + BK - 1) / BK;
    dim3 gridDim(BX, BY);
    HalfFC <BQ, DIM, BK> <<<gridDim, blockDim>>> (q, k, qk, qlen, dim, klen, (half)scale, base);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmHalfMatMulTransBBatchKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    half *input0 = (half*)pointer[id * 8 + 0];
    half *input1 = (half*)pointer[id * 8 + 1];
    half *output = (half*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 
    if (m == 128) {
        int wid = tid >> 5;
        int perN = 8, perK = 128;

        const int BN = 8, BK = 128;
        __shared__ float curC[BN][BK];
        half hscale = (half)alpha;

        for (int stN = 0; stN < n; stN += perN) {
            int endN = min(n, stN + perN);
            for (int stK = 0; stK < k; stK += perK) {
                int endK = min(k, stK + perK);
                wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[8];
                wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b[8];
                wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_c;

                wmma::fill_fragment(frag_c, 0.0);
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_a[j], &input0[(stN) * input0Stride + j * 16], input0Stride);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_b[j], &input1[(stK + wid * 32) * input1Stride + j * 16], input1Stride);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::mma_sync(frag_c, frag_a[j], frag_b[j], frag_c);
                }
                __syncthreads();

                wmma::store_matrix_sync(&curC[0][wid * 32], frag_c, BK, wmma::mem_row_major);
                __syncthreads();

                if (stK + tid < endK) {
                    for (int i = 0; stN + i < endN; i++) {
                        output[(stN + i) * k + stK + tid] = (half)(curC[i][tid] * alpha);
                    }
                }
                __syncthreads();
            }
        }
        return;
    }
#endif
    int pera = 4, perb = 4;
    half cura[4][4], curb[4][4];
    float curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                curc[i][j] = 0.0f;
            }
        }
        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
                FETCH_FLOAT2(cura[a - taska * pera]) = FETCH_FLOAT2(input0[a * input0Stride + l]);
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
                FETCH_FLOAT2(curb[b - taskb * perb]) = FETCH_FLOAT2(input1[b * input1Stride + l]);
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += (float)cura[i][k] * (float)curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        }
    }
/*
    int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        half *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            half *curInput1 = input1 + j * input1Stride;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += (float)curInput0[l] * (float)curInput1[l];
            }
            output[i * k + j] = (half)(sum * alpha);
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMatMulTransBBatchKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    float *input0 = (float*)pointer[id * 8 + 0];
    float *input1 = (float*)pointer[id * 8 + 1];
    float *output = (float*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = input0[a * input0Stride + l + x];
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = input1[b * input1Stride + l + x];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        }
    }

/*
    int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        float *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            float *curInput1 = input1 + j * input1Stride;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += curInput0[l] * curInput1[l];
            }
            output[i * k + j] = sum * alpha;
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmHalfMatMulKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    half *input0 = (half*)pointer[id * 8 + 0];
    half *input1 = (half*)pointer[id * 8 + 1];
    half *output = (half*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);
    int tid = threadIdx.x;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 
    if (k == 128) {
        int wid = tid >> 5;
        int perN = 8, perM = 128;
        for (int i = 0; i < n; i++) {
            output[i * k + tid] = (half)0;
        }

        __shared__ half curA[8][128];
        __shared__ float curC[8][128];

        for (int stN = 0; stN < n; stN += perN) {
            int endN = min(stN + perN, n);
            wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_c;
            wmma::fill_fragment(frag_c, 0.0);

            for (int stM = 0; stM < m; stM += perM) {
                int endM = min(stM + perM, m);
                if (stM + tid < m) {
                    for (int i = 0; stN + i < endN; i++) {
                        curA[i][tid] = input0[(stN + i) * input0Stride + stM + tid];
                    }
                } else {
                    for (int i = 0; stN + i < endN; i++) {
                        curA[i][tid] = (half)0.0;
                    }
                }

                wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[8];
                wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[8];
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_a[j], &curA[0][16 * j], 128);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_b[j], &input1[(stM + 16 * j) * input1Stride + wid * 32], input1Stride);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::mma_sync(frag_c, frag_a[j], frag_b[j], frag_c);
                }
                __syncthreads();
            }
            wmma::store_matrix_sync(&curC[0][wid * 32], frag_c, 128, wmma::mem_row_major);
            __syncthreads();

            for (int i = 0; stN + i < endN; i++) {
                output[(stN + i) * k + tid] = (half)((float)output[(stN + i) * k + tid] + (float)curC[i][tid] * alpha);
            }
            __syncthreads();
        }
        return;
    }
#endif
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = (l + x < m ? (float)input0[a * input0Stride + l + x] : 0.f);
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = (l + x < m ? (float)input1[(l + x) * input1Stride + b] : 0.f);
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        }
    }
/*
    for (int i = 0; i < n; i++) {
        half *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            half *curInput1 = input1 + j;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += (float)curInput0[l] * (float)curInput1[l * input1Stride];
            }
            output[i * k + j] = (half)(sum * alpha);
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMatMulKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    float *input0 = (float*)pointer[id * 8 + 0];
    float *input1 = (float*)pointer[id * 8 + 1];
    float *output = (float*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = l + x < m ? input0[a * input0Stride + l + x] : 0;
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = l + x < m ? input1[(l + x) * input1Stride + b] : 0;
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        }
    }

/*
    //int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        float *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            float *curInput1 = input1 + j;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += curInput0[l] * curInput1[l * input1Stride];
            }
            output[i * k + j] = sum * alpha;
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void SimpleMask(float* a, float *b, float maskValue, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        if (b[i] > 0.99) {
            a[i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void SimpleMask(half* a, half *b, half maskValue, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        if (__half2float(b[i]) > 0.99) {
            a[i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionMaskKernel(float* a, float *b, float maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (b[on * spatial + i] > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionMaskKernel(half *a, half *b, half maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (__half2float(b[on * spatial + i]) > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void CausalMask(T* a, T maskValue, int q, int k, int base) {
    a += blockIdx.x * k;
    for (int i = base + blockIdx.x + threadIdx.x + 1; i < k; i += THREAD_PER_BLOCK) {
        a[i] = maskValue;
    }
}

__global__ void InitBlockAtten(float *sum0, float *max0, float *sum1, float *max1, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        sum0[i] = sum1[i] = 0.0f;
        max0[i] = max1[i] = -10000.0f;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void AttnBlockUpdate(half *data, int n, int m, float *lastMax, float *lastSum, float *curMax, float *curSum) {
    __shared__ float scale;
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    if (tid == 0) {
        float diff = fminf(lastMax[bid] - curMax[bid], 0.f);
        float oldSum = lastSum[bid] * expf(diff);
        scale = (curSum[bid] > 1e-10f) ? (oldSum / curSum[bid]) : 0.0f;

        lastSum[bid] = curSum[bid];
        lastMax[bid] = curMax[bid];
    }
    __syncthreads();

    for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
        data[bid * m + i] = (half)((float)data[bid * m + i] * scale);
    }
}

template <int THREAD_PER_BLOCK>
__device__ void FastllmSoftmaxKernelInner1Func(float *input, float *output, int channels, float *maxp, float *sump) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float maxV;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float maxValue = -1e100;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        maxValue = max(maxValue, input[i]);
    }
    sdata[tid] = maxValue;
    __syncthreads();

    // 2. 求max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 3. 记录max
    if (tid == 0) {
        maxV = sdata[0];
        if (maxp != nullptr) {
            maxp[0] = sdata[0];
        }
    }
    __syncthreads();

    // 4. 求和
    float sum = 0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = exp(input[i] - maxV);
        sum += output[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (fabs(sdata[0]) < 1e-6) {
            sdata[0] = 0.0001;
        }
        if (sump != nullptr) {
            sump[0] = sdata[0];
        }
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] /= sdata[0];
    }
}

__device__ half FastllmHalfMaxFunc(const __half a, const __half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
    return __half2float(a) >= __half2float(b) ? a : b;
#else
#if defined(CUDART_VERSION) && CUDART_VERSION > 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hmax(a, b);
#else
    return __hge(a, b) ? a : b;
#endif
#endif
}

template <int THREAD_PER_BLOCK>
__device__ void FastllmSoftmaxKernelInner1Func(half *input, half *output, int channels, float *maxp, float *sump) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float maxValue = -1e10;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        maxValue = max(maxValue, (float)input[i]);
    }
    sdata[tid] = maxValue;
    __syncthreads();

    // 2. 求max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 3. 记录max
    if (tid == 0) {
        if (maxp != nullptr) {
            sdata[0] = max(maxp[0], sdata[0]);
        }
    }
    __syncthreads();
    float maxV = sdata[0];
    __syncthreads();

    // 4. 求和
    float sum = 0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        sum = sum + exp((float)input[i] - maxV);
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (fabs(sdata[0]) < 1e-6) {
            sdata[0] = 0.0001;
        }
        if (sump != nullptr) {
            sump[0] = sump[0] * exp(maxp[0] - maxV) + sdata[0];
            sdata[0] = sump[0];
            maxp[0] = maxV;
        }
    }
    __syncthreads();

    float scale = 1.0 / sdata[0];
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (half)(exp((float)input[i] - maxV) * scale);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(float* input, float *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(half* input, half *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(half* input, half *output, int outer, int channels, float *maxp, float *sump) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels, maxp + o, sump + o);
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmSoftmaxKernelInner1WithCausalMask(T* input, T *output, int outer, int channels, int base) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, o + base + 1, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmSoftmaxKernelInner1WithCausalMask(T* input, T *output, int outer, int channels, int base, float *maxp, float *sump) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, min(channels, o + base + 1), maxp + o, sump + o);
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelBatchInner1(uint8_t** pointer) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> ((T*)pointer[o * 3], (T*)pointer[o * 3 + 1],
                                                       (int)((size_t)pointer[o * 3 + 2]), nullptr, nullptr);
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelBatchInner1(uint8_t** pointer, int outer) {
    int o = blockIdx.x;
    int channels = (int)((size_t)pointer[o / outer * 2 + 1]);
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> ((T*)pointer[o / outer * 2] + (o % outer) * channels, (T*)pointer[o / outer * 2] + (o % outer) * channels,
                                                       channels, nullptr, nullptr);
}

bool FastllmCudaSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.Count(axis + 1);
    if (inner == 1) {
        if (input.dataType == fastllm::DataType::FLOAT32) {
            if (channels < 8) {
                FastllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> (cudaInput, cudaOutput, outer, channels);
            } else if (channels < 64) {
                FastllmSoftmaxKernelInner1 <8> <<< outer, 8 >>> (cudaInput, cudaOutput, outer, channels);
            } else if (channels < 512) {
                FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (cudaInput, cudaOutput, outer, channels);
            } else {
                FastllmSoftmaxKernelInner1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, outer, channels);
            }
        } else {
            if (channels < 8) {
                FastllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            } else if (channels < 64) {
                FastllmSoftmaxKernelInner1 <8> <<< outer, 8 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            } else if (channels < 512) {
                FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            } else {
                FastllmSoftmaxKernelInner1 <256> <<< outer, 256 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            }
        }
    } else {
        printf("softmax error.\n");
        exit(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSoftmaxBatch(fastllm::Data **inputs, fastllm::Data **outputs, int axis, int batch) {
    int total = 0;
    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        total += outer;
    }
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * total * 3);
    uint8_t ** cpuPointers = new uint8_t*[total * 3];
    int cur = 0;

    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        auto &output = *outputs[b];
        float *cudaInput = (float *) input.cudaData;
        float *cudaOutput = (float *) output.cudaData;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        if (inner == 1) {
            for (int o = 0; o < outer; o++) {
                cpuPointers[cur * 3 + 0] = (uint8_t*)(cudaInput + o * channels);
                cpuPointers[cur * 3 + 1] = (uint8_t*)(cudaOutput + o * channels);
                cpuPointers[cur * 3 + 2] = (uint8_t*)((size_t)channels);
                cur++;
            }
        } else {
            printf("softmax error.\n");
            exit(0);
        }
    }

    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * total * 3, cudaMemcpyHostToDevice);
    FastllmSoftmaxKernelBatchInner1 <float, 256> <<<total, 256>>> (pointers);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

extern bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis);

bool FastllmCudaHalfAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                              const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType) {
    using namespace flashinfer;
    
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];
    half *qd = (half*)q.cudaData;
    half *kd = (half*)k.cudaData;
    half *vd = (half*)v.cudaData;
    half *maskd = mask.dims.size() > 0 ? (half*)mask.cudaData : nullptr;
    half *od = (half*)output.cudaData;
    int batch = (mask.dims.size() == 3) ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));

    // 使用 FlashInfer 实现 attention
    
    uint32_t num_kv_heads = k0;          // KV heads 数（所有 batch 共享）

    // 最可能的情况：group 就是 group_size（GQA 的 group size）
    // 所以 num_qo_heads（每个 batch）= group * num_kv_heads
    uint32_t num_qo_heads = group * num_kv_heads;  // 每个 batch 的 Q heads 数 = group_size * num_kv_heads
    uint32_t actual_batch = q0 / num_qo_heads;     // batch 数 = q0 / (每个 batch 的 Q heads 数)
    
    // 验证参数有效性，避免除零错误
    if (num_kv_heads == 0) {
        printf("Error: num_kv_heads is 0 (k0=%d)\n", k0);
        return false;
    }
    if (num_qo_heads == 0) {
        printf("Error: num_qo_heads is 0 (group=%d, num_kv_heads=%u)\n", group, num_kv_heads);
        return false;
    }
    if (num_qo_heads % num_kv_heads != 0) {
        printf("Error: num_qo_heads (%u) is not divisible by num_kv_heads (%u), group=%d\n", 
               num_qo_heads, num_kv_heads, group);
        return false;
    }
    if (actual_batch == 0) {
        printf("Error: actual_batch is 0 (q0=%d, num_qo_heads=%u)\n", q0, num_qo_heads);
        return false;
    }
    if (q0 % num_qo_heads != 0) {
        printf("Error: q0 (%d) is not divisible by num_qo_heads (%u)\n", q0, num_qo_heads);
        return false;
    }
    uint32_t qo_len = q1;                // query 序列长度
    uint32_t kv_len = k1;                // key/value 序列长度
    uint32_t head_dim_qk = q2;           // QK head dimension
    uint32_t head_dim_vo = v2;           // VO head dimension
    
    // 确定 mask mode - FlashInfer 的 custom mask 需要 bit-packed 格式，暂时不支持
    MaskMode mask_mode = MaskMode::kNone;
    bool use_custom_mask = (maskd != nullptr);
// printf("maskType = %d, use_custom_mask = %d, batch = %d\n", maskType, use_custom_mask, batch);
    if (maskType == 0 && !use_custom_mask && batch == 1) {
        mask_mode = MaskMode::kCausal;
    }
mask_mode = MaskMode::kCausal;
    // 注意：FlashInfer 的 custom mask 格式与 fastllm 不同，暂时禁用
    if (use_custom_mask) {
        // Fallback 到原始实现，因为 mask 格式不兼容
        use_custom_mask = false;
    }
    
    // FlashInfer 支持 HND 布局，使用 HND 布局实现
    // fastllm 的数据布局是 HND: [num_heads, seq_len, head_dim]
    // 对于 HND 布局：
    // - stride_n (token 之间的 stride) = head_dim
    // - stride_h (head 之间的 stride) = seq_len * head_dim
    bool use_flashinfer = (head_dim_qk == 128 && head_dim_vo == 128 && !use_custom_mask);
// use_flashinfer = false;
    // 调试信息：打印参数值
    if (use_flashinfer) {
        // printf("FlashInfer params: q0=%d, q1=%d, q2=%d, k0=%d, k1=%d, v2=%d, group=%d, batch=%d\n", q0, q1, q2, k0, k1, v2, group, batch);
        // printf("  num_kv_heads=%u, num_qo_heads=%u, actual_batch=%u, qo_len=%u, kv_len=%u\n", num_kv_heads, num_qo_heads, actual_batch, qo_len, kv_len);
    }
    
    if (use_flashinfer) {
        // 为每个 batch item 调用 FlashInfer
        // q0 = batch * num_qo_heads，所以需要按 batch 循环
        for (int batch_idx = 0; batch_idx < actual_batch; batch_idx++) {
            // 准备参数
            // q 的布局: [batch*num_qo_heads, seq_len, head_dim] (HND)
            // 对于单个 batch，需要 group 个 heads 的数据
            // cur_q 指向 batch_idx * group 个 heads 的起始位置
            half *cur_q = qd + batch_idx * group * q.Count(1);
            half *cur_k = kd + batch_idx * k.Count(1);
            half *cur_v = vd + batch_idx * v.Count(1);
            half *cur_o = od + batch_idx * group * output.Count(1);
            
            // 对于 HND 布局 [num_heads, seq_len, head_dim]:
            // - stride_n (token 之间的 stride) = head_dim
            // - stride_h (head 之间的 stride) = seq_len * head_dim
            uint32_t q_stride_n = q.strides[1];    // head_dim (token 之间的 stride)
            uint32_t q_stride_h = q.Count(1);      // seq_len * head_dim (head 之间的 stride)
            
            // k/v 也是 HND 布局: [num_kv_heads, kv_len, head_dim]
            uint32_t kv_stride_n = k.strides[1];   // head_dim (token 之间的 stride)
            uint32_t kv_stride_h = k.Count(1);     // kv_len * head_dim (head 之间的 stride)

            // 验证 stride 值
            if (q_stride_n == 0 || q_stride_h == 0 || kv_stride_n == 0 || kv_stride_h == 0) {
                printf("Error: Invalid stride values: q_stride_n=%u, q_stride_h=%u, kv_stride_n=%u, kv_stride_h=%u\n",
                       q_stride_n, q_stride_h, kv_stride_n, kv_stride_h);
                use_flashinfer = false;
                break;
            }
            
            // 验证序列长度
            if (qo_len == 0 || kv_len == 0) {
                printf("Error: Invalid sequence lengths: qo_len=%u, kv_len=%u\n", qo_len, kv_len);
                use_flashinfer = false;
                break;
            }
            
            // 创建 SinglePrefillParams (使用 HND 布局)
            // 注意：FlashInfer 内部会计算 group_size = num_qo_heads / num_kv_heads
            // 所以我们需要确保 num_qo_heads 是 num_kv_heads 的倍数
            // 由于我们已经验证了 num_qo_heads % num_kv_heads == 0，这应该是安全的
            
            // 再次验证，避免运行时除零
            uint32_t expected_group_size = num_qo_heads / num_kv_heads;
            if (expected_group_size == 0) {
                printf("Error: expected_group_size is 0 (num_qo_heads=%u, num_kv_heads=%u)\n",
                       num_qo_heads, num_kv_heads);
                use_flashinfer = false;
                break;
            }
            
            SinglePrefillParams<half, half, half> params(
                cur_q, cur_k, cur_v, nullptr,  // q, k, v, custom_mask (暂时不支持)
                cur_o, nullptr, nullptr,        // o, lse, alibi_slopes
                num_qo_heads,                   // num_qo_heads (每个 batch 的 Q heads 数)
                num_kv_heads,                   // num_kv_heads (KV heads 数)
                qo_len,                         // qo_len
                kv_len,                         // kv_len
                q_stride_n,                     // q_stride_n (token stride for HND = head_dim)
                q_stride_h,                     // q_stride_h (head stride for HND = seq_len * head_dim)
                kv_stride_n,                    // k_stride_n (token stride for HND = head_dim)
                kv_stride_h,                    // k_stride_h (head stride for HND = kv_len * head_dim)
                head_dim_qk,                    // head_dim
                -1,                             // window_left (-1 means no sliding window)
                0.0f,                           // logits_soft_cap
                scale,                          // sm_scale
                1.0f,                           // rope_scale (不使用 RoPE)
                10000.0f                        // rope_theta (不使用 RoPE)
            );
            
            // 分配临时缓冲区（如果需要 partition-kv）
            half *tmp = nullptr;
            // 暂时不分配
            /* if (kv_len > 8192) {
                uint32_t num_chunks = (kv_len + 8191) / 8192;
                size_t tmp_size = num_chunks * qo_len * num_qo_heads * head_dim_vo * sizeof(half) + 
                                 num_chunks * qo_len * num_qo_heads * sizeof(float);
                tmp = (half*)FastllmCudaMalloc(tmp_size);
            } */ 
            
            // 调用 FlashInfer，根据 mask_mode 选择不同的 variant
            cudaError_t status = cudaSuccess;
            cudaStream_t stream = nullptr;
            
            if (mask_mode == MaskMode::kCausal) {
                status = SinglePrefillWithKVCacheDispatched<128, 128, PosEncodingMode::kNone, false, MaskMode::kCausal, DefaultAttention<false, false, false, false>>(
                    params, tmp, stream);
            } else {
                status = SinglePrefillWithKVCacheDispatched<128, 128, PosEncodingMode::kNone, false, MaskMode::kNone, DefaultAttention<false, false, false, false>>(
                    params, tmp, stream);
            }
            
            if (tmp != nullptr) {
                FastllmCudaFree(tmp);
            }
            
            if (status != cudaSuccess) {
                printf("FlashInfer error: %s\n", cudaGetErrorString(status));
                // Fallback 到原始实现
                use_flashinfer = false;
                break;
            }
((fastllm::Data*)&output)->Resize({output.dims[1], output.dims[0], output.dims[2]});
FastllmCudaPermute(*((fastllm::Data*)&output), {1, 0, 2});
        }
        
        if (use_flashinfer) {
            DeviceSync();
            return true;
        }
    }
    
    // Fallback 到原始实现
    half beta = __float2half_rn(0.0f), one = __float2half_rn(1.0f), hscale = __float2half_rn(scale);
    if (q1 >= 1024 || (q1 > 1 && q1 != k1 && k1 >= 1024)) {
        int alignQ1 = q1, alignK1 = k1;
        int part = alignK1;
        bool useFastAttn = getCudaInfos()->hasTensorCore && batch == 1 && (q2 == 128 && v2 == 128) && maskType == 0;
        useFastAttn &= (q1 % 1024 == 0 && k1 % 1024 == 0);

        if (useFastAttn) {
            alignQ1 = ((q1 - 1) / 128 + 1) * 128;
            alignK1 = ((k1 - 1) / 128 + 1) * 128;
            part = (alignK1 > 8192 ? 8192 : alignK1);
        }
        half *qk = (half *) FastllmCudaMalloc(alignQ1 * part * sizeof(half));

        cudaMemset(qk, 0, alignQ1 * part * sizeof(half));
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;
        for (int i = 0; i < q0; i++) {
//DeviceSync();
//auto st = std::chrono::system_clock::now();
            if (useFastAttn) { 
                if (alignK1 > 8192) {
                    float *lastSum = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *lastMax = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *currentSum = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *currentMax = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));

                    int threadPerBlock = std::min(256, alignQ1);
                    InitBlockAtten <<< (alignQ1 - 1) / threadPerBlock + 1, threadPerBlock>>> (lastSum, lastMax, currentSum, currentMax, alignQ1);

                    int part = 8192;
                    for (int st = 0; st < alignK1; st += part) {
                        int len = std::min(part, alignK1 - st);
                        status = cublasHgemm(fastllmCublasHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            len, alignQ1, q2, &hscale,
                                            kd + (i / group) * k.Count(1) + st * k.strides[1], k.strides[1],
                                            qd + i * q.Count(1), q.strides[1],
                                            &beta, 
                                            qk, len);
                        CausalMask<256, half> <<<q1, 256>>>(qk, __float2half_rn(0.0f), alignQ1, len, k1 - q1 - st);
                        FastllmSoftmaxKernelInner1WithCausalMask<256> <<< q1, 256 >>>(qk, qk, alignQ1, len, k1 - q1 - st, currentMax, currentSum);
                        if (st > 0) {
                            AttnBlockUpdate <128> <<< alignQ1, 128 >>> (od + i * v2 * q1, alignQ1, v2, lastMax, lastSum, currentMax, currentSum);
                        } else {
                            cudaMemcpy(lastMax, currentMax, alignQ1 * sizeof(float), cudaMemcpyDeviceToDevice);
                            cudaMemcpy(lastSum, currentSum, alignQ1 * sizeof(float), cudaMemcpyDeviceToDevice);
                        }
                        half currentScale = __float2half_rn(st > 0 ? 1.0f : 0.0f);
                        status = cublasHgemm(fastllmCublasHandle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            v2, alignQ1, len, &one,
                                            vd + (i / group) * v.Count(1) + st * v.strides[1], v.strides[1],
                                            qk, len,
                                            &currentScale,
                                            od + i * v2 * q1, v2);
                    }

                    FastllmCudaFree(lastSum);
                    FastllmCudaFree(lastMax);
                    FastllmCudaFree(currentSum);
                    FastllmCudaFree(currentMax);
                } else {
                    GpuQK(qd + i * q.Count(1), kd + (i / group) * k.Count(1), qk, alignQ1, alignK1, q2, scale, k1 - q1);
                    FastllmSoftmaxKernelInner1WithCausalMask<128> <<< q1, 128 >>>(qk, qk, q1, alignK1, k1 - q1);
                    status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N,
                                                v2, q1, alignK1, &one,
                                                vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                                qk, alignK1, alignK1 * alignQ1,
                                                &beta,
                                                od + i * v2 * q1, v2, v2 * q1, 1);
                }
            } else {
                status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                k1, q1, q2, &hscale,
                                                kd + (i / group) * k.Count(1), k.strides[1], k.Count(1),
                                                qd + i * q.Count(1), q.strides[1], q.Count(1),
                                                &beta,
                                                qk, k1, k1 * q1, 1);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("status = %d\n", (int) status);
                    printf("Error: cublas error during MatMulTransB in Attention operator.\n");
                    throw ("cublas error");
                    exit(0);
                }

                if (batch == 1 && maskd == nullptr && maskType == 0) {
                    CausalMask<256, half> <<<q1, 256>>>(qk, __float2half_rn(0), q1, k1, k1 - q1);
                    FastllmSoftmaxKernelInner1WithCausalMask<128> <<< q1, 128 >>>(qk, qk, q1, k1, k1 - q1);
                } else {
                    if (maskd != nullptr) {
                        SimpleMask<256> <<< (q1 * k1 / 256) + 1, 256>>>(qk, maskd + (i / (q0 / batch)) * maskStride, __float2half_rn(-10000), q1 * k1);
                    }

                    int outer = q1;
                    if (k1 < 8) {
                        FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, qk, outer, k1);
                    } else if (k1 < 64) {
                        FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, qk, outer, k1);
                    } else if (k1 < 512) {
                        FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, qk, outer, k1);
                    } else {
                        FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, qk, outer, k1);
                    }
                }

                status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                               v2, q1, k1, &one,
                                               vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                               qk, k1, k1 * q1,
                                               &beta,
                                               od + i * v2 * q1, v2, v2 * q1, 1);
            }

//DeviceSync(); printf("softmax spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
/*DeviceSync();
int n = k1, m = q1, k = q2;
float spend = GetSpan(st, std::chrono::system_clock::now());
float gops = (float)n * m * k * 4 / spend / 1e9;
printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);*/
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error during MatMul in Attention operator.\n");
                throw ("cublas error");
                exit(0);
            }
        }

        FastllmCudaFree(qk);
        DeviceSync();
        return true;
    }

    if (true) {
        half *qk = (half *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(half));
        half *temp = (half *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(half));
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &hscale,
                                           kd, k.strides[1], k.Count(1),
                                           qd, q.strides[1], q.Count(1) * group,
                                           &beta,
                                           qk, k1, k1 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        if (maskd) {
            int spatial = q1 * k1, n = batch, m = q0 / batch;
            FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(qk, maskd, __float2half_rn(-10000), n, m, spatial);
        }

        int outer = q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           vd, v.strides[1], v.Count(1),
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        DeviceSync();
        return true;
    }
    return true;
}

bool FastllmCudaAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType) {
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];
    float *qd = (float*)q.cudaData;
    float *kd = (float*)k.cudaData;
    float *vd = (float*)v.cudaData;
    float *maskd = mask.dims.size() > 0 ? (float*)mask.cudaData : nullptr;
    float *od = (float*)output.cudaData;
    int batch = (mask.dims.size() == 3) ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));

    if (q1 >= 1024 || (q1 > 1 && q1 != k1 && k1 >= 1024)) {
        float *qk = (float *) FastllmCudaMalloc(q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;


        for (int i = 0; i < q0; i++) {
            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_T, CUBLAS_OP_N,
                                               k1, q1, q2, &scale,
                                               kd + (i / group) * k.Count(1), k.strides[1], k.Count(1),
                                               qd + i * q.Count(1), q.strides[1], q.Count(1),
                                               &beta,
                                               qk, k1, k1 * q1, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error during MatMulTransB in Attention operator.\n");
                throw ("cublas error");
                exit(0);
            }

            if (batch == 1 && maskd == nullptr && maskType == 0) {
                CausalMask<256, float> <<<q1, 256>>>(qk, 0, q1, k1, k1 - q1);
                FastllmSoftmaxKernelInner1WithCausalMask<128> <<< q1, 128 >>>(qk, qk, q1, k1, k1 - q1);
            } else {
                if (maskd) {
                    SimpleMask<256> <<< (q1 * k1 / 256) + 1, 256>>>(qk, maskd + (i / (q0 / batch)) * maskStride, -10000, q1 * k1);
                }
                int outer = q1;
                if (k1 < 8) {
                    FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, qk, outer, k1);
                } else if (k1 < 64) {
                    FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, qk, outer, k1);
                } else if (k1 < 512) {
                    FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, qk, outer, k1);
                } else {
                    FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, qk, outer, k1);
                }
            }

            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                               v2, q1, k1, &one,
                                               vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                               qk, k1, k1 * q1,
                                               &beta,
                                               od + i * v2 * q1, v2, v2 * q1, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error during MatMul in Attention operator.\n");
                throw ("cublas error");
                exit(0);
            }
        }

        FastllmCudaFree(qk);
        DeviceSync();
        return true;
    }

    if (true) {
        float *qk = (float *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        float *temp = (float *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &scale,
                                           kd, k.strides[1], k.Count(1),
                                           qd, q.strides[1], q.Count(1) * group,
                                           &beta,
                                           qk, k1, k1 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        if (maskd) {
            int spatial = q1 * k1, n = batch, m = q0 / batch;
            FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(qk, maskd, -10000, n, m, spatial);
        }

        int outer = q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           vd, v.strides[1], v.Count(1),
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        DeviceSync();
        return true;
    }
    return true;
}

template <typename T>
bool DoFastllmCudaAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                               fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch) {
    if (false) {
        half beta = __float2half_rn(0.0f), one = __float2half_rn(1.0f), hscale = __float2half_rn(scale);
        int q0 = q[0]->dims[0], q1 = q[0]->dims[1], q2 = q[0]->dims[2], k0 = k[0]->dims[0], k1 = k[0]->dims[1], v2 = v[0]->dims[2];
        for (int i = 0; i < batch; i++) {
            q1 = max(q1, q[i]->dims[1]);
            k1 = max(k1, k[i]->dims[1]);
        }

        half *allKeys = (half*) FastllmCudaMalloc(batch * k0 * k1 * q2 * sizeof(half));
        half *allValues = (half*) FastllmCudaMalloc(batch * k0 * k1 * v2 * sizeof(half));

        std::vector <void*> dsts, srcs;
        std::vector <size_t> dpitchs, spitchs, widths, heights;
        for (int i = 0; i < batch; i++) {
            dsts.push_back((uint8_t *) (allKeys + i * k0 * k1 * q2));
            dpitchs.push_back(k1 * q2 * sizeof(half));
            srcs.push_back(k[i]->cudaData);
            spitchs.push_back(k[i]->strides[0] * sizeof(half));
            widths.push_back(k[i]->dims[1] * q2 * sizeof(half));
            heights.push_back(k0);

            dsts.push_back((uint8_t *) (allValues + i * k0 * k1 * v2));
            dpitchs.push_back(k1 * v2 * sizeof(half));
            srcs.push_back(v[i]->cudaData);
            spitchs.push_back(v[i]->strides[0] * sizeof(half));
            widths.push_back(v[i]->dims[1] * v2 * sizeof(half));
            heights.push_back(k0);
        }
        FastllmCudaMemcpy2DDeviceToDeviceBatch(dsts.data(), dpitchs.data(), srcs.data(), spitchs.data(), widths.data(), heights.data(), dsts.size());
/*
        for (int i = 0; i < batch; i++) {
            cudaMemcpy2D(
                allKeys + i * k0 * k1 * q2, k1 * q2 * sizeof(half), 
                k[i]->cudaData, k[i]->strides[0] * sizeof(half), 
                k[i]->dims[1] * q2 * sizeof(half), k0, 
                cudaMemcpyDeviceToDevice
            );
            cudaMemcpy2D(
                allValues + i * k0 * k1 * v2, k1 * v2 * sizeof(half), 
                v[i]->cudaData, v[i]->strides[0] * sizeof(half), 
                v[i]->dims[1] * v2 * sizeof(half), k0, 
                cudaMemcpyDeviceToDevice
            );
        }
*/
        half *qd = (half*)q[0]->cudaData;
        half *od = (half*)output[0]->cudaData;
        half *qk = (half *) FastllmCudaMalloc(batch * q0 * q1 * k1 * sizeof(half));
        half *temp = (half *) FastllmCudaMalloc(batch * q0 * q1 * k1 * sizeof(half));
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &hscale,
                                           allKeys, q2, k1 * q2,
                                           qd, q2, group * q1 * q2,
                                           &beta,
                                           qk, k1, k1 * q1 * group, batch * q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        int outer = batch * q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           allValues, v2, k1 * v2,
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, batch * q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        FastllmCudaFree(allKeys);
        FastllmCudaFree(allValues);
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        DeviceSync();
        return true;
    }

    int k0 = k[0]->dims[0];
    size_t memSum = 0;
    for (int b = 0; b < batch; b++) {
        memSum += q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
    }
    T *mem = (T*) FastllmCudaMalloc(memSum * sizeof(T));
    T **qk = new T*[batch];
    memSum = 0;
    for (int b = 0; b < batch; b++) {
        int s = q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
        qk[b] = mem + memSum;
        memSum += s;
    }

    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * k0 * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * k0 * 8];
    if (true) {
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) q[b]->cudaData + i * group * q[b]->dims[1] * q[b]->dims[2] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) k[b]->cudaData + i * k[b]->strides[0] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) q[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) q[b]->strides[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) k[b]->strides[1];
            }
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, cudaMemcpyHostToDevice);
        if (typeid(T) == typeid(half)) {
            FastllmHalfMatMulTransBBatchKernel <128> <<<batch * k0, 128>>> (pointers, scale);
        } else {
            FastllmMatMulTransBBatchKernel <128> <<<batch * k0, 128>>> (pointers, scale);
        }
    }

    if (true) {
        int outer = q[0]->dims[0] * q[0]->dims[1];
        int maxChannels = 0;
        for (int b = 0; b < batch; b++) {
            int outer = q[b]->dims[0] * q[b]->dims[1];
            int channels = k[b]->dims[1];
            cpuPointers[b * 2 + 0] = (uint8_t*)(qk[b]);
            cpuPointers[b * 2 + 1] = (uint8_t*)((size_t)channels);
            maxChannels = max(maxChannels, channels);
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 2, cudaMemcpyHostToDevice);
        if (maxChannels < 128) {
            FastllmSoftmaxKernelBatchInner1 <T, 32> <<<batch * outer, 32>>> (pointers, outer);
        } else if (maxChannels < 512) {
            FastllmSoftmaxKernelBatchInner1 <T, 64> <<<batch * outer, 64>>> (pointers, outer);
        } else {
            FastllmSoftmaxKernelBatchInner1 <T, 128> <<<batch * outer, 128>>> (pointers, outer);
        }
    }

    if (true) {
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) v[b]->cudaData + i * v[b]->strides[0] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) output[b]->cudaData + i * group * q[b]->dims[1] * v[b]->dims[2] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) v[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) v[b]->strides[1];
            }
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, cudaMemcpyHostToDevice);
        
        if (typeid(T) == typeid(half)) {
            FastllmHalfMatMulKernel <128> <<<batch * k0, 128>>> (pointers, 1.0f);
        } else {
            FastllmMatMulKernel <128> <<<batch * k0, 128>>> (pointers, 1.0f);
        }
    }

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    FastllmCudaFree(mem);
    delete[] qk;
    
    DeviceSync();
    return true;
}

bool FastllmCudaAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                               fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch) {
    if (q[0]->dataType == fastllm::DataType::FLOAT32) {
        return DoFastllmCudaAttentionBatch <float> (q, k, v, mask, output, group, scale, batch);
    } else if (q[0]->dataType == fastllm::DataType::FLOAT16) {
        return DoFastllmCudaAttentionBatch <half> (q, k, v, mask, output, group, scale, batch);
    } else {
        printf("Error: attention datatype error.\n");
        throw ("Error: attention datatype error.");
        exit(0);
    }
}

bool FastllmCudaAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue) {
    int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];
    float *cudaData = (float *) FastllmCudaPrepareInput(input);
    float *maskData = (float *) FastllmCudaPrepareInput(mask);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(cudaData, maskData, maskValue,
                                                       n, m, spatial);
    } else {
        FastllmAttentionMaskKernel <256> <<< n * m, 256>>>((half*)cudaData, (half*)maskData, __float2half(maskValue),
                                                        n, m, spatial);
    }
    FastllmCudaFinishInput(mask, maskData);
    FastllmCudaFinishOutput(input, cudaData);
    return true;
}

bool FastllmCudaMLA(const fastllm::Data &qNope, const fastllm::Data &qPe, const fastllm::Data &kvCache, const fastllm::Data &peCache, 
    fastllm::Data &ss, fastllm::Data &output, float softmaxScale) {
    int b = qPe.dims[0], s = qPe.dims[1], h = qPe.dims[2], c = qNope.dims.back(), t = kvCache.dims[1], r = qPe.dims[3];
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    if (qNope.dataType == fastllm::DataType::FLOAT32) {
        float *score = (float*)FastllmCudaMalloc(b * s * h * t * sizeof(float));
        float alpha = softmaxScale, beta0 = 0.0f, beta1 = 1.0f;
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, c, &alpha,
            (float*)peCache.cudaData, c, t * c,
            (float*)qNope.cudaData, c, h * c,
            &beta0,
            score, t, t * h, 1);
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, r, &alpha,
            (float*)kvCache.cudaData, r, t * r,
            (float*)qPe.cudaData, r, h * r,
            &beta1,
            score, t, t * h, 1);        
        int outer = b * s * h, channels = t;
        FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (score, score, outer, channels);
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    c, b * s * h, t, &beta1,
                    (float*)peCache.cudaData, c, t * c,
                    score, t, b * s * h * t,
                    &beta0,
                    (float*)output.cudaData, c, c * b * s * h, 1);
        FastllmCudaFree(score);
    } else if (qNope.dataType == fastllm::DataType::FLOAT16) {
        half *score = (half*)FastllmCudaMalloc(b * s * h * t * sizeof(half));
        half alpha = __float2half_rn(softmaxScale), beta0 = __float2half_rn(0.0f), beta1 = __float2half_rn(1.0f);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, c, &alpha,
            (half*)peCache.cudaData, c, t * c,
            (half*)qNope.cudaData, c, h * c,
            &beta0,
            score, t, t * h, 1);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, r, &alpha,
            (half*)kvCache.cudaData, r, t * r,
            (half*)qPe.cudaData, r, h * r,
            &beta1,
            score, t, t * h, 1);        
        int outer = b * s * h, channels = t;
        FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (score, score, outer, channels);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    c, b * s * h, t, &beta1,
                    (half*)peCache.cudaData, c, t * c,
                    score, t, b * s * h * t,
                    &beta0,
                    (half*)output.cudaData, c, c * b * s * h, 1);
        FastllmCudaFree(score);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int) status);
        printf("Error: cublas error during MatMul in MLA operator.\n");
        throw("cublas error");
    }

    DeviceSync();
    return true;
}

bool FastllmCudaBatchMatMulTransBBatch(void **i0s, void **i1s, void **os,
                                       int *ns, int *ms, int *ks,
                                       int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *) i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *) i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *) os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *) (size_t) ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *) (size_t) ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *) (size_t) ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *) (size_t) i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *) (size_t) i1Strides[i];
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 8, cudaMemcpyHostToDevice);
    FastllmMatMulTransBBatchKernel <128> <<<batch, 128>>> (pointers, alpha);
    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

bool FastllmCudaBatchMatMulBatch(void **i0s, void **i1s, void **os,
                                 int *ns, int *ms, int *ks,
                                 int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *) i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *) i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *) os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *) (size_t) ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *) (size_t) ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *) (size_t) ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *) (size_t) i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *) (size_t) i1Strides[i];
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 8, cudaMemcpyHostToDevice);
    FastllmMatMulKernel <128> <<<batch, 128>>> (pointers, alpha);
    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}
