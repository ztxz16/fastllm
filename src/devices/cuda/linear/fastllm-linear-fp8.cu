//
// Created by huangyuyang on 2/6/26.
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

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
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;

        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(exp2f(8.0f)), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);
        FastllmCudaFP8E4M3BLOCK1282HalfKernel <<< k * (m / 128), 128 >>>((uint8_t*)weight.cudaData, cudaFp16Weight);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
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
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;

        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(exp2f(8.0f)), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        len = k * m;
        FastllmCudaFP8E4M32HalfKernel <<< k, 256 >>>((uint8_t*)weight.cudaData, cudaScales, cudaFp16Weight, k, m, weight.blockK, weight.blockM);
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
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
        half *cudaFp16Weight;

        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(exp2f(8.0f)); // fp8 -> fp16的转换系数  
        __half h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);

        len = k * m;

        FastllmCudaFP8E4M32HalfKernel <<< k, 256 >>>((uint8_t*)weight.cudaData, cudaScales, cudaFp16Weight, k, m, weight.blockK, weight.blockM);

        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaOutput, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Weight);
    } else {
        LaunchFastllmGemmFp16FP8E4M3(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, n, m, k, weight.blockM, weight.blockK);
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
        half *cudaFp16Weight;

        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(exp2f(8.0f));
        __half h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        FastllmCudaFP8E4M3BLOCK1282HalfKernel <<< k * (m / 128), 128 >>>((uint8_t*)weight.cudaData, cudaFp16Weight);

        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaOutput, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }
        if (bias.dims.size() > 0) {
            half *cudaBiasDataHalf = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasDataHalf, k);
        }

        FastllmCudaFree(cudaFp16Weight);
    } else {
        LaunchFastllmGemmFp16FP8E4M3Block128(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, n, m, k, perRow);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
