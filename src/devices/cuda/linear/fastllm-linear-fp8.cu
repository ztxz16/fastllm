//
// Created by huangyuyang on 2/6/26.
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#ifdef __CUDACC__
#include <cuda_bf16.h>
#endif

typedef union __align__(16) _union_bf16_4_fp8 {
    uint2 in;
    __nv_bfloat16 out[4];
    __nv_bfloat162 out2[2];
    __device__ _union_bf16_4_fp8() {
      // Do nothing
    }
} union_bf16_4_fp8;

__global__ void FastllmCudaFP8E4M3BLOCK1282HalfKernel(uint8_t* a, half *b) {
    unsigned int tid = threadIdx.x;
    unsigned int st = blockIdx.x;

    a += st * (128 + sizeof(float));
    b += st * 128;
    b[tid] = __float2half((float)__ushort_as_half(((a[tid] & 0x80) << 8) | ((a[tid] & 0x7F) << 7)) * *(float*)(a + 128));
}

__global__ void FastllmCudaFP8E4M32HalfKernel(uint8_t* a, float *scales, half *b, int k, int m, int blockK, int blockM) {
    unsigned int tid = threadIdx.x;
    unsigned int st = blockIdx.x;

    int ms = (m - 1) / blockM + 1;
    scales += (st / blockK) * ms;

    for (int i = tid * 4; i < m; i += blockDim.x * 4) {
        float curScale = scales[i / blockM];
        uint32_t ori = *(uint32_t*)(a + st * m + i);
        half bf0 = __ushort_as_half( (((ori >> 0) & 0x80) << 8) | (((ori >> 0) & 0x7F) << 7) );
        half bf1 = __ushort_as_half( (((ori >> 8) & 0x80) << 8) | (((ori >> 8) & 0x7F) << 7) );
        half bf2 = __ushort_as_half( (((ori >> 16) & 0x80) << 8) | (((ori >> 16) & 0x7F) << 7) );
        half bf3 = __ushort_as_half( (((ori >> 24) & 0x80) << 8) | (((ori >> 24) & 0x7F) << 7) );

        b[st * m + i + 0] = __float2half((float)bf0 * curScale);
        b[st * m + i + 1] = __float2half((float)bf1 * curScale);
        b[st * m + i + 2] = __float2half((float)bf2 * curScale);
        b[st * m + i + 3] = __float2half((float)bf3 * curScale);
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFP8E4M3Kernel1MultiRow(float *A, uint8_t *B, float *C,
                                                    float *bias, float *scales,
                                                    int m, int k, int blockM, int blockK) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(120.0f);
    scales += (st / blockK) * ms;

    const uint8_t *baseB = (uint8_t*)B + st * m;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        float curScale = scales[i / blockM];
        uint32_t bb = ((uint32_t*)(baseB + i))[0];
        float bf0 = __uint_as_float( (((bb >> 0) & 0x80) << 24) | (((bb >> 0) & 0x7F) << 20) ) * curScale;
        float bf1 = __uint_as_float( (((bb >> 8) & 0x80) << 24) | (((bb >> 8) & 0x7F) << 20) ) * curScale;
        float bf2 = __uint_as_float( (((bb >> 16) & 0x80) << 24) | (((bb >> 16) & 0x7F) << 20) ) * curScale;
        float bf3 = __uint_as_float( (((bb >> 24) & 0x80) << 24) | (((bb >> 24) & 0x7F) << 20) ) * curScale;

        // float bf0 = (float)baseB[i + 0] * curScale;
        // float bf1 = (float)baseB[i + 1] * curScale;
        // float bf2 = (float)baseB[i + 2] * curScale;
        // float bf3 = (float)baseB[i + 3] * curScale;
#pragma unroll
        for (int x = 0; x < PART; x++) {
            float4 aBuffer = FETCH_FLOAT4(A[i + x * m]);

            sdata[x][tid] += aBuffer.x * bf0;
            sdata[x][tid] += aBuffer.y * bf1;
            sdata[x][tid] += aBuffer.z * bf2;
            sdata[x][tid] += aBuffer.w * bf3;
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[st + k * x] = sdata[x][0] * magicScaleConstant;
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = sdata[x][0] * magicScaleConstant + bias[st];
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfFP8E4M3Kernel1MultiRow(half *A, uint8_t *B, half *C,
                                                    half *bias, float *scales,
                                                    int m, int k, int blockM, int blockK) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    scales += (st / blockK) * ms;

    const uint8_t *baseB = (uint8_t*)B + st * m;
    union_half4 regA;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        float curScale = scales[i / blockM];
        uint32_t bb = ((uint32_t*)(baseB + i))[0];
        __half2 B01 = make_half2(__short_as_half( (((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 7) ), 
                                __short_as_half( (((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 7) ));
        __half2 B23 = make_half2(__short_as_half( (((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7) ), 
                                __short_as_half( (((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7) ));        
#pragma unroll
        for (int x = 0; x < PART; x++) {
            regA.in = *reinterpret_cast<const uint2 *>(A + x * m + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
            sdata[x][tid] += ((float)regA.out[0] * (float)B01.x + 
                                (float)regA.out[1] * (float)B01.y +
                                (float)regA.out[2] * (float)B23.x +
                                (float)regA.out[3] * (float)B23.y) * curScale;
#else
            __half2 p01 = __hmul2(regA.out2[0], B01); // {a0b0, a1b1}
            __half2 p23 = __hmul2(regA.out2[1], B23); // {a2b2, a3b3}
            __half2 sum_halves_vec = __hadd2(p01, p23); // {a0b0+a2b2, a1b1+a3b3}
            __half sum_h = __hadd(sum_halves_vec.x, sum_halves_vec.y); // (a0b0+a2b2) + (a1b1+a3b3)
            sdata[x][tid] += __half2float(sum_h) * curScale;
#endif
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] * magicScaleConstant);
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] * magicScaleConstant + (float)bias[st]);
        }
    }
    __syncthreads();
}

void LaunchFastllmGemmFp32FP8E4M3(float *input, uint8_t *weight, float *output, float *bias, float *scales, int n, int m, int k, int blockM, int blockK) {
    if (n == 1) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 2) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 3) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 4) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 5) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 6) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 7) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else {
        int i = 0; 
        for (; i + 7 < n; i += 8) {
            FastllmGemvFP8E4M3Kernel1MultiRow<64, 8> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i < n; i++) {
            FastllmGemvFP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        return;
    }
}

void LaunchFastllmGemmFp16FP8E4M3(half *input, uint8_t *weight, half *output, half *bias, float *scales, int n, int m, int k, int blockM, int blockK) {
    if (n == 1) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 2) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 3) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 4) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 5) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 6) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 7) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 8) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 8> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 9) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 9> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 10) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 10> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 11) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 11> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 12) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 12> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 13) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 13> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 14) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 14> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 15) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 15> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else {
        int i = 0; 
        for (; i + 15 < n; i += 16) {
            FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 16> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i + 7 < n; i += 8) {
            FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 8> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i + 3 < n; i += 4) {
            FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 4> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i < n; i++) {
            FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        return;
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow(half *A, uint8_t *B, half *C,
                                                    half *bias,
                                                    int m, int k, int perRow) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
    const int block_size = 128;
    const float magicScaleConstant = exp2f(8.0f);

    const uint8_t *baseB = B + st * perRow;
    int numBlocks = (m - 1) / block_size + 1;
    union_half4 regA;

    for (int blk = 0; blk < numBlocks; blk++) {
        int blkStart = blk * block_size;
        int blkEnd = min(blkStart + block_size, m);
        // 数据布局: [fp8_0..fp8_127][float_scale_0][fp8_128..fp8_255][float_scale_1]...
        // 每 (128 + 4) 字节为一个 block
        const uint8_t *blkData = baseB + blk * (block_size + sizeof(float));
        float blkScale = *(float*)(blkData + block_size);

        for (int i = blkStart + tid * 4; i < blkEnd; i += THREAD_PER_BLOCK * 4) {
            int localIdx = i - blkStart;
            int remaining = blkEnd - i;
            
            if (remaining >= 4) {
                uint32_t bb = *(uint32_t*)(blkData + localIdx);
                __half2 B01 = make_half2(__short_as_half( (((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 7) ), 
                                        __short_as_half( (((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 7) ));
                __half2 B23 = make_half2(__short_as_half( (((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7) ), 
                                        __short_as_half( (((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7) ));
#pragma unroll
                for (int x = 0; x < PART; x++) {
                    regA.in = *reinterpret_cast<const uint2 *>(A + x * m + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
                    sdata[x][tid] += ((float)regA.out[0] * (float)B01.x + 
                                        (float)regA.out[1] * (float)B01.y +
                                        (float)regA.out[2] * (float)B23.x +
                                        (float)regA.out[3] * (float)B23.y) * blkScale;
#else
                    __half2 p01 = __hmul2(regA.out2[0], B01);
                    __half2 p23 = __hmul2(regA.out2[1], B23);
                    __half2 sum_halves_vec = __hadd2(p01, p23);
                    __half sum_h = __hadd(sum_halves_vec.x, sum_halves_vec.y);
                    sdata[x][tid] += __half2float(sum_h) * blkScale;
#endif
                }
            } else {
                for (int j = 0; j < remaining; j++) {
                    half bVal = __float2half((float)__ushort_as_half(((blkData[localIdx + j] & 0x80) << 8) | ((blkData[localIdx + j] & 0x7F) << 7)) * blkScale);
#pragma unroll
                    for (int x = 0; x < PART; x++) {
                        sdata[x][tid] += __half2float(A[x * m + i + j]) * __half2float(bVal);
                    }
                }
            }
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] * magicScaleConstant);
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] * magicScaleConstant + (float)bias[st]);
        }
    }
    __syncthreads();
}

void LaunchFastllmGemmFp16FP8E4M3Block128(half *input, uint8_t *weight, half *output, half *bias, int n, int m, int k, int perRow) {
    if (n == 1) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 2) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 3) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 4) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 5) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 6) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 7) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 8) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 8> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 9) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 9> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 10) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 10> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 11) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 11> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 12) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 12> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 13) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 13> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 14) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 14> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 15) {
        FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 15> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else {
        int i = 0; 
        for (; i + 15 < n; i += 16) {
            FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 16> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, m, k, perRow);
        }
        for (; i + 7 < n; i += 8) {
            FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 8> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, m, k, perRow);
        }
        for (; i + 3 < n; i += 4) {
            FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 4> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, m, k, perRow);
        }
        for (; i < n; i++) {
            FastllmGemvHalfFP8E4M3Block128Kernel1MultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, m, k, perRow);
        }
        return;
    }
}

static void FastllmCudaFP8E4M3Block128EnsureHalfBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr ||
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *) cudaBiasData);
    }
}

static void FastllmCudaFP8E4M3Block128EnsureBFloat16BiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr ||
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        __nv_bfloat16 *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(__nv_bfloat16));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2Bf16Kernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(__nv_bfloat16));
        }
        checkCudaErrors("Error: CUDA error when moving bf16 bias to device!", state);
        weight.extraCudaHalfData.push_back((void *) cudaBiasData);
    }
}

static void FastllmCudaFP8E4M3Block128EnsureBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        cudaError_t state = cudaSuccess;
        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }
}

bool FastllmCudaMatMulFloatFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP8E4M3Block128EnsureBiasOnDevice(weight, bias, k);

    float *cudaBiasData = (float*)weight.extraCudaData[0];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n >= 0) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output;

        cudaFp16Input = (half *) FastllmCudaMalloc((size_t)n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc((size_t)n * k * sizeof(half));

        size_t wsBytes = 0;
        bool ownScratch = false;
        half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(half);
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

        __half h_alpha = __float2half_rn(exp2f(8.0f)), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        const size_t fp8BlockBytes = 128 + sizeof(float);
        const int blocksPerRow = m / 128;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);

            FastllmCudaFP8E4M3BLOCK1282HalfKernel <<< kc * blocksPerRow, 128 >>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * blocksPerRow * fp8BlockBytes,
                cudaFp16Weight);

            status = cublasGemmEx(fastllmCublasHandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  kc, n, m,
                                  &h_alpha, cudaFp16Weight, AType,
                                  m, cudaFp16Input, BType,
                                  m, &h_beta,
                                  cudaFp16Output + kOff, CType,
                                  k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error.\n");
                throw("cublas error");
                exit(0);
            }
        }

        len = n * k;
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

static void FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, weight.scales.size() * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), weight.scales.size() * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }
}

static void FastllmCudaFP8E4M3EnsureHalfBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr ||
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *) cudaBiasData);
    }
}

bool FastllmCudaMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, k);

    float *cudaScales = (float*)weight.extraCudaData[0];
    float *cudaBiasData = (float*)weight.extraCudaData[1];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n >= 16) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output;

        cudaFp16Input = (half *) FastllmCudaMalloc((size_t)n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc((size_t)n * k * sizeof(half));

        size_t wsBytes = 0;
        bool ownScratch = false;
        half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(half);
        int blockK = weight.blockK;
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));
        // FP8 per-blockK scales 要求 chunk 起点对齐 blockK；最后一个 chunk 大小可不对齐
        if (blockK > 0) {
            maxRowsPerChunk = (maxRowsPerChunk / blockK) * blockK;
            if (maxRowsPerChunk < blockK) {
                // workspace 太小，回退到 bigBuffer 一次性放下
                FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
                cudaFp16Weight = (half *) FastllmCudaMalloc((size_t)k * m * sizeof(half));
                ownScratch = true;
                maxRowsPerChunk = k;
            }
        }
        int ms = (m - 1) / weight.blockM + 1;

        __half h_alpha = __float2half_rn(exp2f(8.0f)), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            FastllmCudaFP8E4M32HalfKernel <<< kc, 256 >>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * m,
                cudaScales + (size_t)(kOff / blockK) * ms,
                cudaFp16Weight, kc, m, weight.blockK, weight.blockM);
            status = cublasGemmEx(fastllmCublasHandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  kc, n, m,
                                  &h_alpha, cudaFp16Weight, AType,
                                  m, cudaFp16Input, BType,
                                  m, &h_beta,
                                  cudaFp16Output + kOff, CType,
                                  k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error.\n");
                throw("cublas error");
                exit(0);
            }
        }

        len = n * k;
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    } else {
        LaunchFastllmGemmFp32FP8E4M3(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, n, m, k, weight.blockM, weight.blockK);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, k);
    FastllmCudaFP8E4M3EnsureHalfBiasOnDevice(weight, bias, k);

    float *cudaScales = (float*)weight.extraCudaData[0];
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    if (n >= 32) {
        auto fastllmCublasHandle = getFastllmCublasHandle();

        size_t wsBytes = 0;
        bool ownScratch = false;
        half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(half);
        int blockK = weight.blockK;
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));
        if (blockK > 0) {
            maxRowsPerChunk = (maxRowsPerChunk / blockK) * blockK;
            if (maxRowsPerChunk < blockK) {
                FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
                cudaFp16Weight = (half *) FastllmCudaMalloc((size_t)k * m * sizeof(half));
                ownScratch = true;
                maxRowsPerChunk = k;
            }
        }
        int ms = (m - 1) / weight.blockM + 1;

        __half h_alpha = __float2half_rn(exp2f(8.0f));
        __half h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            FastllmCudaFP8E4M32HalfKernel <<< kc, 256 >>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * m,
                cudaScales + (size_t)(kOff / blockK) * ms,
                cudaFp16Weight, kc, m, weight.blockK, weight.blockM);

            status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    kc, n, m,
                                    &h_alpha, cudaFp16Weight, AType,
                                    m, cudaInput, BType,
                                    m, &h_beta,
                                    cudaOutput + kOff, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error.\n");
                throw("cublas error");
                exit(0);
            }
        }
        if (bias.dims.size() > 0) {
            half *cudaBiasDataFp16 = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasDataFp16, k);
        }

        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    } else {
        LaunchFastllmGemmFp16FP8E4M3(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, n, m, k, weight.blockM, weight.blockK);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// ==================== BFloat16 x FP8_E4M3 ====================

__global__ void FastllmCudaFP8E4M32BF16Kernel(uint8_t* a, float *scales, __nv_bfloat16 *b, int k, int m, int blockK, int blockM) {
    unsigned int tid = threadIdx.x;
    unsigned int st = blockIdx.x;

    int ms = (m - 1) / blockM + 1;
    scales += (st / blockK) * ms;

    for (int i = tid * 4; i < m; i += blockDim.x * 4) {
        float curScale = scales[i / blockM];
        uint32_t ori = *(uint32_t*)(a + st * m + i);
        uint16_t b0_bits = (((ori >> 0) & 0x80) << 8) | (((ori >> 0) & 0x7F) << 4);
        uint16_t b1_bits = (((ori >> 8) & 0x80) << 8) | (((ori >> 8) & 0x7F) << 4);
        uint16_t b2_bits = (((ori >> 16) & 0x80) << 8) | (((ori >> 16) & 0x7F) << 4);
        uint16_t b3_bits = (((ori >> 24) & 0x80) << 8) | (((ori >> 24) & 0x7F) << 4);

        b[st * m + i + 0] = __float2bfloat16_rn(__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b0_bits)) * curScale);
        b[st * m + i + 1] = __float2bfloat16_rn(__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b1_bits)) * curScale);
        b[st * m + i + 2] = __float2bfloat16_rn(__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b2_bits)) * curScale);
        b[st * m + i + 3] = __float2bfloat16_rn(__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b3_bits)) * curScale);
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvBF16FP8E4M3Kernel1MultiRow(__nv_bfloat16 *A, uint8_t *B, __nv_bfloat16 *C,
                                                    __nv_bfloat16 *bias, float *scales,
                                                    int m, int k, int blockM, int blockK) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int st = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(120.0f);
    scales += (st / blockK) * ms;

    const uint8_t *baseB = (uint8_t*)B + st * m;
    union_bf16_4_fp8 regA;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        float curScale = scales[i / blockM];
        uint32_t bb = ((uint32_t*)(baseB + i))[0];
        uint16_t b0_bits = (((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 4);
        uint16_t b1_bits = (((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 4);
        uint16_t b2_bits = (((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 4);
        uint16_t b3_bits = (((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 4);
        float bf0 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b0_bits));
        float bf1 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b1_bits));
        float bf2 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b2_bits));
        float bf3 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b3_bits));

#pragma unroll
        for (int x = 0; x < PART; x++) {
            regA.in = *reinterpret_cast<const uint2 *>(A + x * m + i);
            sdata[x][tid] += (__bfloat162float(regA.out[0]) * bf0 +
                              __bfloat162float(regA.out[1]) * bf1 +
                              __bfloat162float(regA.out[2]) * bf2 +
                              __bfloat162float(regA.out[3]) * bf3) * curScale;
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[st + k * x] = __float2bfloat16_rn(sdata[x][0] * magicScaleConstant);
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = __float2bfloat16_rn(sdata[x][0] * magicScaleConstant + __bfloat162float(bias[st]));
        }
    }
    __syncthreads();
}

void LaunchFastllmGemmBF16FP8E4M3(__nv_bfloat16 *input, uint8_t *weight, __nv_bfloat16 *output, __nv_bfloat16 *bias, float *scales, int n, int m, int k, int blockM, int blockK) {
    if (n == 1) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 2) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 3) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 4) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 5) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 6) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 7) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 8) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 8> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 9) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 9> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 10) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 10> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 11) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 11> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 12) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 12> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 13) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 13> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 14) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 14> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 15) {
        FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 15> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else {
        int i = 0;
        for (; i + 15 < n; i += 16) {
            FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 16> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i + 7 < n; i += 8) {
            FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 8> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i + 3 < n; i += 4) {
            FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 4> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i < n; i++) {
            FastllmGemvBF16FP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        return;
    }
}

static void FastllmCudaFP8E4M3EnsureBFloat16BiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr ||
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        __nv_bfloat16 *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(__nv_bfloat16));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2Bf16Kernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(__nv_bfloat16));
        }
        checkCudaErrors("Error: CUDA error when moving bf16 bias to device!", state);
        weight.extraCudaHalfData.push_back((void *) cudaBiasData);
    }
}

bool FastllmCudaBFloat16MatMulFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, k);
    FastllmCudaFP8E4M3EnsureBFloat16BiasOnDevice(weight, bias, k);

    float *cudaScales = (float*)weight.extraCudaData[0];
    __nv_bfloat16 *cudaBF16Bias = bias.dims.size() == 0 ? nullptr : (__nv_bfloat16 *) weight.extraCudaHalfData[0];
    __nv_bfloat16 *cudaInput = (__nv_bfloat16*)FastllmCudaPrepareInput(input);
    __nv_bfloat16 *cudaOutput = (__nv_bfloat16*)FastllmCudaPrepareOutput(output);

    if (n >= 32) {
        auto fastllmCublasHandle = getFastllmCublasHandle();

        size_t wsBytes = 0;
        bool ownScratch = false;
        __nv_bfloat16 *cudaBF16Weight = (__nv_bfloat16 *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(__nv_bfloat16), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(__nv_bfloat16);
        int blockK = weight.blockK;
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));
        if (blockK > 0) {
            maxRowsPerChunk = (maxRowsPerChunk / blockK) * blockK;
            if (maxRowsPerChunk < blockK) {
                FastllmReleaseDequantScratch(cudaBF16Weight, ownScratch);
                cudaBF16Weight = (__nv_bfloat16 *) FastllmCudaMalloc((size_t)k * m * sizeof(__nv_bfloat16));
                ownScratch = true;
                maxRowsPerChunk = k;
            }
        }
        int ms = (m - 1) / weight.blockM + 1;

        float h_alpha = exp2f(120.0f), h_beta = 0.0f;
        cudaDataType_t AType = CUDA_R_16BF, BType = CUDA_R_16BF, CType = CUDA_R_16BF, ComputeType = CUDA_R_32F;
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            FastllmCudaFP8E4M32BF16Kernel <<< kc, 256 >>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * m,
                cudaScales + (size_t)(kOff / blockK) * ms,
                cudaBF16Weight, kc, m, weight.blockK, weight.blockM);

            status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    kc, n, m,
                                    &h_alpha, cudaBF16Weight, AType,
                                    m, cudaInput, BType,
                                    m, &h_beta,
                                    cudaOutput + kOff, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error (BFloat16MatMulFP8E4M3).\n");
                throw("cublas error");
                exit(0);
            }
        }

        if (cudaBF16Bias != nullptr) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBF16Bias, k);
        }

        FastllmReleaseDequantScratch(cudaBF16Weight, ownScratch);
    } else {
        LaunchFastllmGemmBF16FP8E4M3(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBF16Bias, cudaScales, n, m, k, weight.blockM, weight.blockK);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloatFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP8E4M3Block128EnsureHalfBiasOnDevice(weight, bias, k);

    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    size_t perRow = m + ((m - 1) / 128 + 1) * sizeof(float);

    if (n >= 32) {
        auto fastllmCublasHandle = getFastllmCublasHandle();

        size_t wsBytes = 0;
        bool ownScratch = false;
        half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(half);
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

        const size_t fp8BlockBytes = 128 + sizeof(float);
        const int blocksPerRow = m / 128;

        __half h_alpha = __float2half_rn(exp2f(8.0f));
        __half h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            FastllmCudaFP8E4M3BLOCK1282HalfKernel <<< kc * blocksPerRow, 128 >>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * blocksPerRow * fp8BlockBytes,
                cudaFp16Weight);

            status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    kc, n, m,
                                    &h_alpha, cudaFp16Weight, AType,
                                    m, cudaInput, BType,
                                    m, &h_beta,
                                    cudaOutput + kOff, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error.\n");
                throw("cublas error");
                exit(0);
            }
        }
        if (bias.dims.size() > 0) {
            half *cudaBiasDataHalf = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasDataHalf, k);
        }

        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    } else {
        LaunchFastllmGemmFp16FP8E4M3Block128(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, n, m, k, perRow);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// ==================== BFloat16 x FP8_E4M3_BLOCK_128 ====================

// FP8 BLOCK_128 -> BF16 转换 kernel
// 数据布局: 每个block = 128字节fp8 + 4字节float scale
// FP8 -> BF16 位操作: sign: (byte & 0x80) << 8, mantissa: (byte & 0x7F) << 4
__global__ void FastllmCudaFP8E4M3BLOCK1282BF16Kernel(uint8_t* a, __nv_bfloat16 *b) {
    unsigned int tid = threadIdx.x;
    unsigned int st = blockIdx.x;

    a += st * (128 + sizeof(float));
    b += st * 128;
    uint16_t bf16_bits = ((a[tid] & 0x80) << 8) | ((a[tid] & 0x7F) << 4);
    float val = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&bf16_bits)) * *(float*)(a + 128);
    b[tid] = __float2bfloat16_rn(val);
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow(__nv_bfloat16 *A, uint8_t *B, __nv_bfloat16 *C,
                                                    __nv_bfloat16 *bias,
                                                    int m, int k, int perRow) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int st = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
    const int block_size = 128;
    const float magicScaleConstant = exp2f(120.0f);

    const uint8_t *baseB = B + st * perRow;
    int numBlocks = (m - 1) / block_size + 1;
    union_bf16_4_fp8 regA;

    for (int blk = 0; blk < numBlocks; blk++) {
        int blkStart = blk * block_size;
        int blkEnd = min(blkStart + block_size, m);
        // 数据布局: [fp8_0..fp8_127][float_scale_0][fp8_128..fp8_255][float_scale_1]...
        const uint8_t *blkData = baseB + blk * (block_size + sizeof(float));
        float blkScale = *(float*)(blkData + block_size);

        for (int i = blkStart + tid * 4; i < blkEnd; i += THREAD_PER_BLOCK * 4) {
            int localIdx = i - blkStart;
            int remaining = blkEnd - i;

            if (remaining >= 4) {
                uint32_t bb = *(uint32_t*)(blkData + localIdx);
                // FP8 -> BF16 位操作: sign << 8, mantissa << 4
                uint16_t b0_bits = (((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 4);
                uint16_t b1_bits = (((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 4);
                uint16_t b2_bits = (((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 4);
                uint16_t b3_bits = (((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 4);
                float bf0 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b0_bits));
                float bf1 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b1_bits));
                float bf2 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b2_bits));
                float bf3 = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&b3_bits));
#pragma unroll
                for (int x = 0; x < PART; x++) {
                    regA.in = *reinterpret_cast<const uint2 *>(A + x * m + i);
                    sdata[x][tid] += (__bfloat162float(regA.out[0]) * bf0 +
                                      __bfloat162float(regA.out[1]) * bf1 +
                                      __bfloat162float(regA.out[2]) * bf2 +
                                      __bfloat162float(regA.out[3]) * bf3) * blkScale;
                }
            } else {
                for (int j = 0; j < remaining; j++) {
                    uint16_t bj_bits = ((blkData[localIdx + j] & 0x80) << 8) | ((blkData[localIdx + j] & 0x7F) << 4);
                    float bVal = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&bj_bits)) * blkScale;
#pragma unroll
                    for (int x = 0; x < PART; x++) {
                        sdata[x][tid] += __bfloat162float(A[x * m + i + j]) * bVal;
                    }
                }
            }
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[st + k * x] = __float2bfloat16_rn(sdata[x][0] * magicScaleConstant);
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = __float2bfloat16_rn(sdata[x][0] * magicScaleConstant + __bfloat162float(bias[st]));
        }
    }
    __syncthreads();
}

void LaunchFastllmGemmBF16FP8E4M3Block128(__nv_bfloat16 *input, uint8_t *weight, __nv_bfloat16 *output, __nv_bfloat16 *bias, int n, int m, int k, int perRow) {
    if (n == 1) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 2) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 3) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 4) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 5) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 6) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 7) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 8) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 8> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 9) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 9> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 10) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 10> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 11) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 11> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 12) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 12> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 13) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 13> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 14) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 14> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else if (n == 15) {
        FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 15> <<< k, 64 >>>(input, weight, output, bias, m, k, perRow);
    } else {
        int i = 0;
        for (; i + 15 < n; i += 16) {
            FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 16> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, m, k, perRow);
        }
        for (; i + 7 < n; i += 8) {
            FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 8> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, m, k, perRow);
        }
        for (; i + 3 < n; i += 4) {
            FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 4> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, m, k, perRow);
        }
        for (; i < n; i++) {
            FastllmGemvBF16FP8E4M3Block128Kernel1MultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, m, k, perRow);
        }
        return;
    }
}

bool FastllmCudaBFloat16MatMulFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP8E4M3Block128EnsureBFloat16BiasOnDevice(weight, bias, k);

    __nv_bfloat16 *cudaBF16Bias = bias.dims.size() == 0 ? nullptr : (__nv_bfloat16 *) weight.extraCudaHalfData[0];

    __nv_bfloat16 *cudaInput = (__nv_bfloat16*)FastllmCudaPrepareInput(input);
    __nv_bfloat16 *cudaOutput = (__nv_bfloat16*)FastllmCudaPrepareOutput(output);

    size_t perRow = m + ((m - 1) / 128 + 1) * sizeof(float);

    if (n >= 32) {
        auto fastllmCublasHandle = getFastllmCublasHandle();

        size_t wsBytes = 0;
        bool ownScratch = false;
        __nv_bfloat16 *cudaBF16Weight = (__nv_bfloat16 *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(__nv_bfloat16), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(__nv_bfloat16);
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

        const size_t fp8BlockBytes = 128 + sizeof(float);
        const int blocksPerRow = m / 128;

        float h_alpha = exp2f(120.0), h_beta = 0.0f;
        cudaDataType_t AType = CUDA_R_16BF, BType = CUDA_R_16BF, CType = CUDA_R_16BF, ComputeType = CUDA_R_32F;
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            FastllmCudaFP8E4M3BLOCK1282BF16Kernel <<< kc * blocksPerRow, 128 >>>(
                (uint8_t*)weight.cudaData + (size_t)kOff * blocksPerRow * fp8BlockBytes,
                cudaBF16Weight);

            status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    kc, n, m,
                                    &h_alpha, cudaBF16Weight, AType,
                                    m, cudaInput, BType,
                                    m, &h_beta,
                                    cudaOutput + kOff, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error (BFloat16MatMulFP8E4M3Block128).\n");
                throw("cublas error");
                exit(0);
            }
        }

        if (cudaBF16Bias != nullptr) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBF16Bias, k);
        }

        FastllmReleaseDequantScratch(cudaBF16Weight, ownScratch);
    } else {
        LaunchFastllmGemmBF16FP8E4M3Block128(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBF16Bias, n, m, k, perRow);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// ==================== NVFP4_BLOCK_16 ====================

__device__ __forceinline__ float FastllmCudaNVFP4E2M1ToFloat(uint8_t v) {
    float value = 0.0f;
    switch (v & 0x7) {
        case 0: value = 0.0f; break;
        case 1: value = 0.5f; break;
        case 2: value = 1.0f; break;
        case 3: value = 1.5f; break;
        case 4: value = 2.0f; break;
        case 5: value = 3.0f; break;
        case 6: value = 4.0f; break;
        default: value = 6.0f; break;
    }
    return (v & 0x8) ? -value : value;
}

__global__ void FastllmCudaNVFP4Block162HalfKernel(uint8_t *a, half *b, int m, int perRow) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    uint8_t *rowData = a + (size_t)row * perRow;
    half *rowOut = b + (size_t)row * m;
    for (int i = tid; i < m; i += blockDim.x) {
        int block = i >> 4;
        int offset = i & 15;
        uint8_t *blockData = rowData + block * (8 + sizeof(float));
        float scale = *(float*)(blockData + 8);
        uint8_t packed = blockData[offset >> 1];
        uint8_t fp4 = (offset & 1) ? (packed >> 4) : (packed & 0xF);
        rowOut[i] = __float2half_rn(FastllmCudaNVFP4E2M1ToFloat(fp4) * scale);
    }
}

__global__ void FastllmCudaNVFP4Block162BFloat16Kernel(uint8_t *a, __nv_bfloat16 *b, int m, int perRow) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    uint8_t *rowData = a + (size_t)row * perRow;
    __nv_bfloat16 *rowOut = b + (size_t)row * m;
    for (int i = tid; i < m; i += blockDim.x) {
        int block = i >> 4;
        int offset = i & 15;
        uint8_t *blockData = rowData + block * (8 + sizeof(float));
        float scale = *(float*)(blockData + 8);
        uint8_t packed = blockData[offset >> 1];
        uint8_t fp4 = (offset & 1) ? (packed >> 4) : (packed & 0xF);
        rowOut[i] = __float2bfloat16_rn(FastllmCudaNVFP4E2M1ToFloat(fp4) * scale);
    }
}

static inline size_t FastllmCudaNVFP4Block16BytesPerRow(int m) {
    return (size_t)((m - 1) / 16 + 1) * (8 + sizeof(float));
}

bool FastllmCudaMatMulFloatNVFP4Block16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP8E4M3Block128EnsureBiasOnDevice(weight, bias, k);

    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    auto fastllmCublasHandle = getFastllmCublasHandle();
    half *cudaFp16Input = (half *) FastllmCudaMalloc((size_t)n * m * sizeof(half));
    half *cudaFp16Output = (half *) FastllmCudaMalloc((size_t)n * k * sizeof(half));

    size_t wsBytes = 0;
    bool ownScratch = false;
    half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
    size_t bytesPerRow = (size_t)m * sizeof(half);
    int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

    int len = n * m;
    int threadPerBlock = std::min(256, len);
    FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaInput, cudaFp16Input, len);

    const size_t packedBytesPerRow = FastllmCudaNVFP4Block16BytesPerRow(m);
    const __half h_alpha = __float2half_rn(1.0f);
    const __half h_beta = __float2half_rn(0.0f);
    cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    int dequantThreads = std::min(256, m);

    for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
        int kc = std::min(maxRowsPerChunk, k - kOff);
        FastllmCudaNVFP4Block162HalfKernel <<< kc, dequantThreads >>>(
            (uint8_t*)weight.cudaData + (size_t)kOff * packedBytesPerRow,
            cudaFp16Weight, m, packedBytesPerRow);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              kc, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output + kOff, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error (MatMulFloatNVFP4Block16).\n");
            throw("cublas error");
            exit(0);
        }
    }

    len = n * k;
    threadPerBlock = std::min(256, len);
    FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput, len);
    if (bias.dims.size() > 0) {
        FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
    }

    FastllmCudaFree(cudaFp16Input);
    FastllmCudaFree(cudaFp16Output);
    FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloatNVFP4Block16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP8E4M3Block128EnsureHalfBiasOnDevice(weight, bias, k);

    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    auto fastllmCublasHandle = getFastllmCublasHandle();
    size_t wsBytes = 0;
    bool ownScratch = false;
    half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
    size_t bytesPerRow = (size_t)m * sizeof(half);
    int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

    const size_t packedBytesPerRow = FastllmCudaNVFP4Block16BytesPerRow(m);
    const __half h_alpha = __float2half_rn(1.0f);
    const __half h_beta = __float2half_rn(0.0f);
    cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    int dequantThreads = std::min(256, m);

    for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
        int kc = std::min(maxRowsPerChunk, k - kOff);
        FastllmCudaNVFP4Block162HalfKernel <<< kc, dequantThreads >>>(
            (uint8_t*)weight.cudaData + (size_t)kOff * packedBytesPerRow,
            cudaFp16Weight, m, packedBytesPerRow);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              kc, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaInput, BType,
                              m, &h_beta,
                              cudaOutput + kOff, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error (HalfMatMulFloatNVFP4Block16).\n");
            throw("cublas error");
            exit(0);
        }
    }

    if (cudaBiasData != nullptr) {
        FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
    }

    FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaBFloat16MatMulNVFP4Block16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFP8E4M3Block128EnsureBFloat16BiasOnDevice(weight, bias, k);

    __nv_bfloat16 *cudaBiasData = bias.dims.size() == 0 ? nullptr : (__nv_bfloat16 *) weight.extraCudaHalfData[0];
    __nv_bfloat16 *cudaInput = (__nv_bfloat16*)FastllmCudaPrepareInput(input);
    __nv_bfloat16 *cudaOutput = (__nv_bfloat16*)FastllmCudaPrepareOutput(output);

    auto fastllmCublasHandle = getFastllmCublasHandle();
    size_t wsBytes = 0;
    bool ownScratch = false;
    __nv_bfloat16 *cudaBF16Weight = (__nv_bfloat16 *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(__nv_bfloat16), &wsBytes, &ownScratch);
    size_t bytesPerRow = (size_t)m * sizeof(__nv_bfloat16);
    int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

    const size_t packedBytesPerRow = FastllmCudaNVFP4Block16BytesPerRow(m);
    float h_alpha = 1.0f, h_beta = 0.0f;
    cudaDataType_t AType = CUDA_R_16BF, BType = CUDA_R_16BF, CType = CUDA_R_16BF, ComputeType = CUDA_R_32F;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    int dequantThreads = std::min(256, m);

    for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
        int kc = std::min(maxRowsPerChunk, k - kOff);
        FastllmCudaNVFP4Block162BFloat16Kernel <<< kc, dequantThreads >>>(
            (uint8_t*)weight.cudaData + (size_t)kOff * packedBytesPerRow,
            cudaBF16Weight, m, packedBytesPerRow);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              kc, n, m,
                              &h_alpha, cudaBF16Weight, AType,
                              m, cudaInput, BType,
                              m, &h_beta,
                              cudaOutput + kOff, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error (BFloat16MatMulNVFP4Block16).\n");
            throw("cublas error");
            exit(0);
        }
    }

    if (cudaBiasData != nullptr) {
        FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
    }

    FastllmReleaseDequantScratch(cudaBF16Weight, ownScratch);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
