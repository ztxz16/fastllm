//
// FP8 MoE CUDA kernels and runtime helpers.
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#ifdef __CUDACC__
#include <cuda_fp8.h>
#endif


template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3SwigluKernel(half *A, uint8_t *B, half *C, float *scales, int m, int k, int blockM, int blockK) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;
    union_half4 regA;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            uint32_t gateBytes = *(uint32_t*)(baseGate + i);
            uint32_t upBytes = *(uint32_t*)(baseUp + i);
            __half2 gate01 = make_half2(__short_as_half((((gateBytes >> 0) & 0x80) << 8) | (((gateBytes >> 0) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 8) & 0x80) << 8) | (((gateBytes >> 8) & 0x7F) << 7)));
            __half2 gate23 = make_half2(__short_as_half((((gateBytes >> 16) & 0x80) << 8) | (((gateBytes >> 16) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 24) & 0x80) << 8) | (((gateBytes >> 24) & 0x7F) << 7)));
            __half2 up01 = make_half2(__short_as_half((((upBytes >> 0) & 0x80) << 8) | (((upBytes >> 0) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 8) & 0x80) << 8) | (((upBytes >> 8) & 0x7F) << 7)));
            __half2 up23 = make_half2(__short_as_half((((upBytes >> 16) & 0x80) << 8) | (((upBytes >> 16) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 24) & 0x80) << 8) | (((upBytes >> 24) & 0x7F) << 7)));
            regA.in = *reinterpret_cast<const uint2 *>(A + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
            sdataGate[tid] += ((float)regA.out[0] * (float)gate01.x +
                               (float)regA.out[1] * (float)gate01.y +
                               (float)regA.out[2] * (float)gate23.x +
                               (float)regA.out[3] * (float)gate23.y) * gateScale;
            sdataUp[tid] += ((float)regA.out[0] * (float)up01.x +
                             (float)regA.out[1] * (float)up01.y +
                             (float)regA.out[2] * (float)up23.x +
                             (float)regA.out[3] * (float)up23.y) * upScale;
#else
            __half2 gateProd01 = __hmul2(regA.out2[0], gate01);
            __half2 gateProd23 = __hmul2(regA.out2[1], gate23);
            __half2 gatePair = __hadd2(gateProd01, gateProd23);
            __half gateSum = __hadd(gatePair.x, gatePair.y);
            sdataGate[tid] += __half2float(gateSum) * gateScale;

            __half2 upProd01 = __hmul2(regA.out2[0], up01);
            __half2 upProd23 = __hmul2(regA.out2[1], up23);
            __half2 upPair = __hadd2(upProd01, upProd23);
            __half upSum = __hadd(upPair.x, upPair.y);
            sdataUp[tid] += __half2float(upSum) * upScale;
#endif
        } else {
            for (int j = 0; j < remaining; j++) {
                half gateVal = __float2half((float)__ushort_as_half(((baseGate[i + j] & 0x80) << 8) | ((baseGate[i + j] & 0x7F) << 7)) * gateScale);
                half upVal = __float2half((float)__ushort_as_half(((baseUp[i + j] & 0x80) << 8) | ((baseUp[i + j] & 0x7F) << 7)) * upScale);
                float aVal = __half2float(A[i + j]);
                sdataGate[tid] += aVal * __half2float(gateVal);
                sdataUp[tid] += aVal * __half2float(upVal);
            }
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdataGate[tid] += sdataGate[tid + s];
            sdataUp[tid] += sdataUp[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float gate = (float)(half)(sdataGate[0] * magicScaleConstant);
        float up = (float)(half)(sdataUp[0] * magicScaleConstant);
        C[p] = (half)((gate / (1.0f + expf(-gate))) * up);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3AddToKernel(half *A, uint8_t *B, half *C, float *scales, float alpha, bool overwrite, int m, int k, int blockM, int blockK) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    const uint8_t *baseB = B + (size_t)st * m;
    float *rowScales = scales + (st / blockK) * ms;
    union_half4 regA;

    sdata[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float curScale = rowScales[i / blockM];
        if (remaining >= 4) {
            uint32_t bb = *(uint32_t*)(baseB + i);
            __half2 B01 = make_half2(__short_as_half((((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 7)),
                                     __short_as_half((((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 7)));
            __half2 B23 = make_half2(__short_as_half((((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7)),
                                     __short_as_half((((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7)));
            regA.in = *reinterpret_cast<const uint2 *>(A + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
            sdata[tid] += ((float)regA.out[0] * (float)B01.x +
                           (float)regA.out[1] * (float)B01.y +
                           (float)regA.out[2] * (float)B23.x +
                           (float)regA.out[3] * (float)B23.y) * curScale;
#else
            __half2 p01 = __hmul2(regA.out2[0], B01);
            __half2 p23 = __hmul2(regA.out2[1], B23);
            __half2 sumHalvesVec = __hadd2(p01, p23);
            __half sumH = __hadd(sumHalvesVec.x, sumHalvesVec.y);
            sdata[tid] += __half2float(sumH) * curScale;
#endif
        } else {
            for (int j = 0; j < remaining; j++) {
                half bVal = __float2half((float)__ushort_as_half(((baseB[i + j] & 0x80) << 8) | ((baseB[i + j] & 0x7F) << 7)) * curScale);
                sdata[tid] += __half2float(A[i + j]) * __half2float(bVal);
            }
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float value = (float)(half)(sdata[0] * magicScaleConstant) * alpha;
        if (!overwrite) {
            value += __half2float(C[st]);
        }
        C[st] = (half)value;
    }
}

void LaunchFastllmGemmFp16FP8E4M3Swiglu(half *input, uint8_t *weight, half *output, float *scales, int m, int k, int blockM, int blockK) {
    FastllmGemvHalfFP8E4M3SwigluKernel<64> <<< k, 64 >>>(input, weight, output, scales, m, k, blockM, blockK);
}

void LaunchFastllmGemmFp16FP8E4M3AddTo(half *input, uint8_t *weight, half *output, float *scales, float alpha, bool overwrite, int m, int k, int blockM, int blockK) {
    FastllmGemvHalfFP8E4M3AddToKernel<64> <<< k, 64 >>>(input, weight, output, scales, alpha, overwrite, m, k, blockM, blockK);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3TopKSwigluKernel(half *A, uint8_t **weights, float **scalesPtrs,
                                                       half *C, int m, int k, int blockM, int blockK) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int expert = blockIdx.y;
    uint8_t *B = weights[expert];
    float *scales = scalesPtrs[expert];
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;
    union_half4 regA;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            uint32_t gateBytes = *(uint32_t*)(baseGate + i);
            uint32_t upBytes = *(uint32_t*)(baseUp + i);
            __half2 gate01 = make_half2(__short_as_half((((gateBytes >> 0) & 0x80) << 8) | (((gateBytes >> 0) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 8) & 0x80) << 8) | (((gateBytes >> 8) & 0x7F) << 7)));
            __half2 gate23 = make_half2(__short_as_half((((gateBytes >> 16) & 0x80) << 8) | (((gateBytes >> 16) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 24) & 0x80) << 8) | (((gateBytes >> 24) & 0x7F) << 7)));
            __half2 up01 = make_half2(__short_as_half((((upBytes >> 0) & 0x80) << 8) | (((upBytes >> 0) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 8) & 0x80) << 8) | (((upBytes >> 8) & 0x7F) << 7)));
            __half2 up23 = make_half2(__short_as_half((((upBytes >> 16) & 0x80) << 8) | (((upBytes >> 16) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 24) & 0x80) << 8) | (((upBytes >> 24) & 0x7F) << 7)));
            regA.in = *reinterpret_cast<const uint2 *>(A + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
            sdataGate[tid] += ((float)regA.out[0] * (float)gate01.x +
                               (float)regA.out[1] * (float)gate01.y +
                               (float)regA.out[2] * (float)gate23.x +
                               (float)regA.out[3] * (float)gate23.y) * gateScale;
            sdataUp[tid] += ((float)regA.out[0] * (float)up01.x +
                             (float)regA.out[1] * (float)up01.y +
                             (float)regA.out[2] * (float)up23.x +
                             (float)regA.out[3] * (float)up23.y) * upScale;
#else
            __half2 gateProd01 = __hmul2(regA.out2[0], gate01);
            __half2 gateProd23 = __hmul2(regA.out2[1], gate23);
            __half2 gatePair = __hadd2(gateProd01, gateProd23);
            __half gateSum = __hadd(gatePair.x, gatePair.y);
            sdataGate[tid] += __half2float(gateSum) * gateScale;

            __half2 upProd01 = __hmul2(regA.out2[0], up01);
            __half2 upProd23 = __hmul2(regA.out2[1], up23);
            __half2 upPair = __hadd2(upProd01, upProd23);
            __half upSum = __hadd(upPair.x, upPair.y);
            sdataUp[tid] += __half2float(upSum) * upScale;
#endif
        } else {
            for (int j = 0; j < remaining; j++) {
                half gateVal = __float2half((float)__ushort_as_half(((baseGate[i + j] & 0x80) << 8) | ((baseGate[i + j] & 0x7F) << 7)) * gateScale);
                half upVal = __float2half((float)__ushort_as_half(((baseUp[i + j] & 0x80) << 8) | ((baseUp[i + j] & 0x7F) << 7)) * upScale);
                float aVal = __half2float(A[i + j]);
                sdataGate[tid] += aVal * __half2float(gateVal);
                sdataUp[tid] += aVal * __half2float(upVal);
            }
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdataGate[tid] += sdataGate[tid + s];
            sdataUp[tid] += sdataUp[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float gate = (float)(half)(sdataGate[0] * magicScaleConstant);
        float up = (float)(half)(sdataUp[0] * magicScaleConstant);
        C[(size_t)expert * k + p] = (half)((gate / (1.0f + expf(-gate))) * up);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3TopKDownReduceKernel(half *A, uint8_t **weights, float **scalesPtrs,
                                                           half *C, float *scores, int topk,
                                                           int m, int k, int blockM, int blockK) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    union_half4 regA;

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int expert = 0; expert < topk; expert++) {
        uint8_t *B = weights[expert];
        float *scales = scalesPtrs[expert];
        const uint8_t *baseB = B + (size_t)st * m;
        float *rowScales = scales + (st / blockK) * ms;
        half *expertInput = A + (size_t)expert * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            int remaining = m - i;
            float curScale = rowScales[i / blockM];
            if (remaining >= 4) {
                uint32_t bb = *(uint32_t*)(baseB + i);
                __half2 B01 = make_half2(__short_as_half((((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 7)),
                                         __short_as_half((((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 7)));
                __half2 B23 = make_half2(__short_as_half((((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7)),
                                         __short_as_half((((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7)));
                regA.in = *reinterpret_cast<const uint2 *>(expertInput + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
                sdata[tid] += ((float)regA.out[0] * (float)B01.x +
                               (float)regA.out[1] * (float)B01.y +
                               (float)regA.out[2] * (float)B23.x +
                               (float)regA.out[3] * (float)B23.y) * curScale;
#else
                __half2 p01 = __hmul2(regA.out2[0], B01);
                __half2 p23 = __hmul2(regA.out2[1], B23);
                __half2 sumHalvesVec = __hadd2(p01, p23);
                __half sumH = __hadd(sumHalvesVec.x, sumHalvesVec.y);
                sdata[tid] += __half2float(sumH) * curScale;
#endif
            } else {
                for (int j = 0; j < remaining; j++) {
                    half bVal = __float2half((float)__ushort_as_half(((baseB[i + j] & 0x80) << 8) | ((baseB[i + j] & 0x7F) << 7)) * curScale);
                    sdata[tid] += __half2float(expertInput[i + j]) * __half2float(bVal);
                }
            }
        }
        __syncthreads();

        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out += (float)(half)(sdata[0] * magicScaleConstant) * scores[expert];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[st] = (half)out;
    }
}

void LaunchFastllmGemmFp16FP8E4M3TopKSwiglu(half *input, uint8_t **weights, float **scales, half *output,
                                            int topk, int m, int k, int blockM, int blockK) {
    dim3 grid(k, topk);
    FastllmGemvHalfFP8E4M3TopKSwigluKernel<64> <<< grid, 64 >>>(input, weights, scales, output, m, k, blockM, blockK);
}

void LaunchFastllmGemmFp16FP8E4M3TopKDownReduce(half *input, uint8_t **weights, float **scales, half *output, float *scores,
                                                int topk, int m, int k, int blockM, int blockK) {
    FastllmGemvHalfFP8E4M3TopKDownReduceKernel<64> <<< k, 64 >>>(input, weights, scales, output, scores, topk, m, k, blockM, blockK);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3TopKSwigluIndexedKernel(half *A, const int32_t *indices,
                                                              uint8_t **weights, float **scalesPtrs,
                                                              half *C, int topk, int m, int k,
                                                              int blockM, int blockK) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int topkSlot = blockIdx.y;
    if (topkSlot >= topk) {
        return;
    }
    int expertIdx = indices[topkSlot];
    uint8_t *B = weights[expertIdx];
    float *scales = scalesPtrs[expertIdx];
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;
    union_half4 regA;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            uint32_t gateBytes = *(uint32_t*)(baseGate + i);
            uint32_t upBytes = *(uint32_t*)(baseUp + i);
            __half2 gate01 = make_half2(__short_as_half((((gateBytes >> 0) & 0x80) << 8) | (((gateBytes >> 0) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 8) & 0x80) << 8) | (((gateBytes >> 8) & 0x7F) << 7)));
            __half2 gate23 = make_half2(__short_as_half((((gateBytes >> 16) & 0x80) << 8) | (((gateBytes >> 16) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 24) & 0x80) << 8) | (((gateBytes >> 24) & 0x7F) << 7)));
            __half2 up01 = make_half2(__short_as_half((((upBytes >> 0) & 0x80) << 8) | (((upBytes >> 0) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 8) & 0x80) << 8) | (((upBytes >> 8) & 0x7F) << 7)));
            __half2 up23 = make_half2(__short_as_half((((upBytes >> 16) & 0x80) << 8) | (((upBytes >> 16) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 24) & 0x80) << 8) | (((upBytes >> 24) & 0x7F) << 7)));
            regA.in = *reinterpret_cast<const uint2 *>(A + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
            sdataGate[tid] += ((float)regA.out[0] * (float)gate01.x +
                               (float)regA.out[1] * (float)gate01.y +
                               (float)regA.out[2] * (float)gate23.x +
                               (float)regA.out[3] * (float)gate23.y) * gateScale;
            sdataUp[tid] += ((float)regA.out[0] * (float)up01.x +
                             (float)regA.out[1] * (float)up01.y +
                             (float)regA.out[2] * (float)up23.x +
                             (float)regA.out[3] * (float)up23.y) * upScale;
#else
            __half2 gateProd01 = __hmul2(regA.out2[0], gate01);
            __half2 gateProd23 = __hmul2(regA.out2[1], gate23);
            __half2 gatePair = __hadd2(gateProd01, gateProd23);
            __half gateSum = __hadd(gatePair.x, gatePair.y);
            sdataGate[tid] += __half2float(gateSum) * gateScale;

            __half2 upProd01 = __hmul2(regA.out2[0], up01);
            __half2 upProd23 = __hmul2(regA.out2[1], up23);
            __half2 upPair = __hadd2(upProd01, upProd23);
            __half upSum = __hadd(upPair.x, upPair.y);
            sdataUp[tid] += __half2float(upSum) * upScale;
#endif
        } else {
            for (int j = 0; j < remaining; j++) {
                half gateVal = __float2half((float)__ushort_as_half(((baseGate[i + j] & 0x80) << 8) | ((baseGate[i + j] & 0x7F) << 7)) * gateScale);
                half upVal = __float2half((float)__ushort_as_half(((baseUp[i + j] & 0x80) << 8) | ((baseUp[i + j] & 0x7F) << 7)) * upScale);
                float aVal = __half2float(A[i + j]);
                sdataGate[tid] += aVal * __half2float(gateVal);
                sdataUp[tid] += aVal * __half2float(upVal);
            }
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdataGate[tid] += sdataGate[tid + s];
            sdataUp[tid] += sdataUp[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float gate = (float)(half)(sdataGate[0] * magicScaleConstant);
        float up = (float)(half)(sdataUp[0] * magicScaleConstant);
        C[(size_t)topkSlot * k + p] = (half)((gate / (1.0f + expf(-gate))) * up);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3TopKDownReduceIndexedKernel(half *A, const int32_t *indices,
                                                                  uint8_t **weights, float **scalesPtrs,
                                                                  half *C, const float *scores, int topk,
                                                                  int m, int k, int blockM, int blockK) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    union_half4 regA;

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int expertIdx = indices[topkSlot];
        uint8_t *B = weights[expertIdx];
        float *scales = scalesPtrs[expertIdx];
        const uint8_t *baseB = B + (size_t)st * m;
        float *rowScales = scales + (st / blockK) * ms;
        half *expertInput = A + (size_t)topkSlot * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            int remaining = m - i;
            float curScale = rowScales[i / blockM];
            if (remaining >= 4) {
                uint32_t bb = *(uint32_t*)(baseB + i);
                __half2 B01 = make_half2(__short_as_half((((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 7)),
                                         __short_as_half((((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 7)));
                __half2 B23 = make_half2(__short_as_half((((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7)),
                                         __short_as_half((((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7)));
                regA.in = *reinterpret_cast<const uint2 *>(expertInput + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
                sdata[tid] += ((float)regA.out[0] * (float)B01.x +
                               (float)regA.out[1] * (float)B01.y +
                               (float)regA.out[2] * (float)B23.x +
                               (float)regA.out[3] * (float)B23.y) * curScale;
#else
                __half2 p01 = __hmul2(regA.out2[0], B01);
                __half2 p23 = __hmul2(regA.out2[1], B23);
                __half2 sumHalvesVec = __hadd2(p01, p23);
                __half sumH = __hadd(sumHalvesVec.x, sumHalvesVec.y);
                sdata[tid] += __half2float(sumH) * curScale;
#endif
            } else {
                for (int j = 0; j < remaining; j++) {
                    half bVal = __float2half((float)__ushort_as_half(((baseB[i + j] & 0x80) << 8) | ((baseB[i + j] & 0x7F) << 7)) * curScale);
                    sdata[tid] += __half2float(expertInput[i + j]) * __half2float(bVal);
                }
            }
        }
        __syncthreads();

        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out += (float)(half)(sdata[0] * magicScaleConstant) * scores[topkSlot];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[st] = (half)out;
    }
}

void LaunchFastllmGemmFp16FP8E4M3TopKSwigluIndexed(half *input, const int32_t *indices,
                                                   uint8_t **weights, float **scales, half *output,
                                                   int topk, int m, int k, int blockM, int blockK) {
    dim3 grid(k, topk);
    FastllmGemvHalfFP8E4M3TopKSwigluIndexedKernel<64> <<< grid, 64 >>>(input, indices, weights, scales, output, topk, m, k, blockM, blockK);
}

void LaunchFastllmGemmFp16FP8E4M3TopKDownReduceIndexed(half *input, const int32_t *indices,
                                                       uint8_t **weights, float **scales, half *output, const float *scores,
                                                       int topk, int m, int k, int blockM, int blockK) {
    FastllmGemvHalfFP8E4M3TopKDownReduceIndexedKernel<64> <<< k, 64 >>>(input, indices, weights, scales, output, scores, topk, m, k, blockM, blockK);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3SmallBatchTopKSwigluIndexedKernel(half *A, const int32_t *indices,
                                                                        uint8_t **weights, float **scalesPtrs,
                                                                        half *C, int batch, int topk, int m, int k,
                                                                        int blockM, int blockK) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int task = blockIdx.y;
    int token = task / topk;
    if (token >= batch) {
        return;
    }

    int expertIdx = indices[task];
    uint8_t *B = weights[expertIdx];
    float *scales = scalesPtrs[expertIdx];
    half *tokenInput = A + (size_t)token * m;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;
    union_half4 regA;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            uint32_t gateBytes = *(uint32_t*)(baseGate + i);
            uint32_t upBytes = *(uint32_t*)(baseUp + i);
            __half2 gate01 = make_half2(__short_as_half((((gateBytes >> 0) & 0x80) << 8) | (((gateBytes >> 0) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 8) & 0x80) << 8) | (((gateBytes >> 8) & 0x7F) << 7)));
            __half2 gate23 = make_half2(__short_as_half((((gateBytes >> 16) & 0x80) << 8) | (((gateBytes >> 16) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 24) & 0x80) << 8) | (((gateBytes >> 24) & 0x7F) << 7)));
            __half2 up01 = make_half2(__short_as_half((((upBytes >> 0) & 0x80) << 8) | (((upBytes >> 0) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 8) & 0x80) << 8) | (((upBytes >> 8) & 0x7F) << 7)));
            __half2 up23 = make_half2(__short_as_half((((upBytes >> 16) & 0x80) << 8) | (((upBytes >> 16) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 24) & 0x80) << 8) | (((upBytes >> 24) & 0x7F) << 7)));
            regA.in = *reinterpret_cast<const uint2 *>(tokenInput + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
            sdataGate[tid] += ((float)regA.out[0] * (float)gate01.x +
                               (float)regA.out[1] * (float)gate01.y +
                               (float)regA.out[2] * (float)gate23.x +
                               (float)regA.out[3] * (float)gate23.y) * gateScale;
            sdataUp[tid] += ((float)regA.out[0] * (float)up01.x +
                             (float)regA.out[1] * (float)up01.y +
                             (float)regA.out[2] * (float)up23.x +
                             (float)regA.out[3] * (float)up23.y) * upScale;
#else
            __half2 gateProd01 = __hmul2(regA.out2[0], gate01);
            __half2 gateProd23 = __hmul2(regA.out2[1], gate23);
            __half2 gatePair = __hadd2(gateProd01, gateProd23);
            __half gateSum = __hadd(gatePair.x, gatePair.y);
            sdataGate[tid] += __half2float(gateSum) * gateScale;

            __half2 upProd01 = __hmul2(regA.out2[0], up01);
            __half2 upProd23 = __hmul2(regA.out2[1], up23);
            __half2 upPair = __hadd2(upProd01, upProd23);
            __half upSum = __hadd(upPair.x, upPair.y);
            sdataUp[tid] += __half2float(upSum) * upScale;
#endif
        } else {
            for (int j = 0; j < remaining; j++) {
                half gateVal = __float2half((float)__ushort_as_half(((baseGate[i + j] & 0x80) << 8) | ((baseGate[i + j] & 0x7F) << 7)) * gateScale);
                half upVal = __float2half((float)__ushort_as_half(((baseUp[i + j] & 0x80) << 8) | ((baseUp[i + j] & 0x7F) << 7)) * upScale);
                float aVal = __half2float(tokenInput[i + j]);
                sdataGate[tid] += aVal * __half2float(gateVal);
                sdataUp[tid] += aVal * __half2float(upVal);
            }
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdataGate[tid] += sdataGate[tid + s];
            sdataUp[tid] += sdataUp[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float gate = (float)(half)(sdataGate[0] * magicScaleConstant);
        float up = (float)(half)(sdataUp[0] * magicScaleConstant);
        C[(size_t)task * k + p] = (half)((gate / (1.0f + expf(-gate))) * up);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3SmallBatchTopKDownReduceIndexedKernel(half *A, const int32_t *indices,
                                                                            uint8_t **weights, float **scalesPtrs,
                                                                            half *C, const float *scores, int batch, int topk,
                                                                            int m, int k, int blockM, int blockK) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int token = blockIdx.y;
    if (token >= batch) {
        return;
    }

    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    union_half4 regA;

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int task = token * topk + topkSlot;
        int expertIdx = indices[task];
        uint8_t *B = weights[expertIdx];
        float *scales = scalesPtrs[expertIdx];
        const uint8_t *baseB = B + (size_t)st * m;
        float *rowScales = scales + (st / blockK) * ms;
        half *expertInput = A + (size_t)task * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            int remaining = m - i;
            float curScale = rowScales[i / blockM];
            if (remaining >= 4) {
                uint32_t bb = *(uint32_t*)(baseB + i);
                __half2 B01 = make_half2(__short_as_half((((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 7)),
                                         __short_as_half((((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 7)));
                __half2 B23 = make_half2(__short_as_half((((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7)),
                                         __short_as_half((((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7)));
                regA.in = *reinterpret_cast<const uint2 *>(expertInput + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
                sdata[tid] += ((float)regA.out[0] * (float)B01.x +
                               (float)regA.out[1] * (float)B01.y +
                               (float)regA.out[2] * (float)B23.x +
                               (float)regA.out[3] * (float)B23.y) * curScale;
#else
                __half2 p01 = __hmul2(regA.out2[0], B01);
                __half2 p23 = __hmul2(regA.out2[1], B23);
                __half2 sumHalvesVec = __hadd2(p01, p23);
                __half sumH = __hadd(sumHalvesVec.x, sumHalvesVec.y);
                sdata[tid] += __half2float(sumH) * curScale;
#endif
            } else {
                for (int j = 0; j < remaining; j++) {
                    half bVal = __float2half((float)__ushort_as_half(((baseB[i + j] & 0x80) << 8) | ((baseB[i + j] & 0x7F) << 7)) * curScale);
                    sdata[tid] += __half2float(expertInput[i + j]) * __half2float(bVal);
                }
            }
        }
        __syncthreads();

        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out += (float)(half)(sdata[0] * magicScaleConstant) * scores[task];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[(size_t)token * k + st] = (half)out;
    }
}

void LaunchFastllmGemmFp16FP8E4M3SmallBatchTopKSwigluIndexed(half *input, const int32_t *indices,
                                                             uint8_t **weights, float **scales, half *output,
                                                             int batch, int topk, int m, int k, int blockM, int blockK) {
    dim3 grid(k, batch * topk);
    FastllmGemvHalfFP8E4M3SmallBatchTopKSwigluIndexedKernel<64> <<< grid, 64 >>>(
        input, indices, weights, scales, output, batch, topk, m, k, blockM, blockK);
}

void LaunchFastllmGemmFp16FP8E4M3SmallBatchTopKDownReduceIndexed(half *input, const int32_t *indices,
                                                                 uint8_t **weights, float **scales, half *output, const float *scores,
                                                                 int batch, int topk, int m, int k, int blockM, int blockK) {
    dim3 grid(k, batch);
    FastllmGemvHalfFP8E4M3SmallBatchTopKDownReduceIndexedKernel<64> <<< grid, 64 >>>(
        input, indices, weights, scales, output, scores, batch, topk, m, k, blockM, blockK);
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfFP8E4M3GroupedTopKSwigluIndexedKernel(half *A, const int *routeRows,
                                                                     const int *expertStarts, const int *expertCounts,
                                                                     uint8_t **weights, float **scalesPtrs,
                                                                     half *C, int maxChunks, int m, int k,
                                                                     int blockM, int blockK) {
    __shared__ float sdataGate[PART][THREAD_PER_BLOCK];
    __shared__ float sdataUp[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int expert = blockIdx.y;
    int chunk = blockIdx.z;
    int count = expertCounts[expert];
    int localBase = chunk * PART;
    if (chunk >= maxChunks || localBase >= count) {
        return;
    }

    int start = expertStarts[expert];
    int rows[PART];
    bool active[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) {
        int local = localBase + x;
        active[x] = local < count;
        rows[x] = active[x] ? routeRows[start + local] : 0;
        sdataGate[x][tid] = 0.0f;
        sdataUp[x][tid] = 0.0f;
    }

    uint8_t *B = weights[expert];
    float *scales = scalesPtrs[expert];
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;
    union_half4 regA;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            uint32_t gateBytes = *(uint32_t*)(baseGate + i);
            uint32_t upBytes = *(uint32_t*)(baseUp + i);
            __half2 gate01 = make_half2(__short_as_half((((gateBytes >> 0) & 0x80) << 8) | (((gateBytes >> 0) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 8) & 0x80) << 8) | (((gateBytes >> 8) & 0x7F) << 7)));
            __half2 gate23 = make_half2(__short_as_half((((gateBytes >> 16) & 0x80) << 8) | (((gateBytes >> 16) & 0x7F) << 7)),
                                        __short_as_half((((gateBytes >> 24) & 0x80) << 8) | (((gateBytes >> 24) & 0x7F) << 7)));
            __half2 up01 = make_half2(__short_as_half((((upBytes >> 0) & 0x80) << 8) | (((upBytes >> 0) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 8) & 0x80) << 8) | (((upBytes >> 8) & 0x7F) << 7)));
            __half2 up23 = make_half2(__short_as_half((((upBytes >> 16) & 0x80) << 8) | (((upBytes >> 16) & 0x7F) << 7)),
                                      __short_as_half((((upBytes >> 24) & 0x80) << 8) | (((upBytes >> 24) & 0x7F) << 7)));
#pragma unroll
            for (int x = 0; x < PART; x++) {
                if (!active[x]) {
                    continue;
                }
                regA.in = *reinterpret_cast<const uint2 *>(A + (size_t)rows[x] * m + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
                sdataGate[x][tid] += ((float)regA.out[0] * (float)gate01.x +
                                      (float)regA.out[1] * (float)gate01.y +
                                      (float)regA.out[2] * (float)gate23.x +
                                      (float)regA.out[3] * (float)gate23.y) * gateScale;
                sdataUp[x][tid] += ((float)regA.out[0] * (float)up01.x +
                                    (float)regA.out[1] * (float)up01.y +
                                    (float)regA.out[2] * (float)up23.x +
                                    (float)regA.out[3] * (float)up23.y) * upScale;
#else
                __half2 gateProd01 = __hmul2(regA.out2[0], gate01);
                __half2 gateProd23 = __hmul2(regA.out2[1], gate23);
                __half2 gatePair = __hadd2(gateProd01, gateProd23);
                __half gateSum = __hadd(gatePair.x, gatePair.y);
                sdataGate[x][tid] += __half2float(gateSum) * gateScale;

                __half2 upProd01 = __hmul2(regA.out2[0], up01);
                __half2 upProd23 = __hmul2(regA.out2[1], up23);
                __half2 upPair = __hadd2(upProd01, upProd23);
                __half upSum = __hadd(upPair.x, upPair.y);
                sdataUp[x][tid] += __half2float(upSum) * upScale;
#endif
            }
        } else {
            for (int j = 0; j < remaining; j++) {
                half gateVal = __float2half((float)__ushort_as_half(((baseGate[i + j] & 0x80) << 8) | ((baseGate[i + j] & 0x7F) << 7)) * gateScale);
                half upVal = __float2half((float)__ushort_as_half(((baseUp[i + j] & 0x80) << 8) | ((baseUp[i + j] & 0x7F) << 7)) * upScale);
#pragma unroll
                for (int x = 0; x < PART; x++) {
                    if (!active[x]) {
                        continue;
                    }
                    float aVal = __half2float(A[(size_t)rows[x] * m + i + j]);
                    sdataGate[x][tid] += aVal * __half2float(gateVal);
                    sdataUp[x][tid] += aVal * __half2float(upVal);
                }
            }
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdataGate[x][tid] += sdataGate[x][tid + s];
                sdataUp[x][tid] += sdataUp[x][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            if (active[x]) {
                float gate = (float)(half)(sdataGate[x][0] * magicScaleConstant);
                float up = (float)(half)(sdataUp[x][0] * magicScaleConstant);
                C[(size_t)(start + localBase + x) * k + p] = (half)((gate / (1.0f + expf(-gate))) * up);
            }
        }
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfFP8E4M3GroupedTopKDownScatterIndexedKernel(half *A, const int *routeRows,
                                                                          const float *routeScales,
                                                                          const int *expertStarts, const int *expertCounts,
                                                                          uint8_t **weights, float **scalesPtrs,
                                                                          half *C, int maxChunks, int m, int k,
                                                                          int blockM, int blockK) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int expert = blockIdx.y;
    int chunk = blockIdx.z;
    int count = expertCounts[expert];
    int localBase = chunk * PART;
    if (chunk >= maxChunks || localBase >= count) {
        return;
    }

    int start = expertStarts[expert];
    float alphas[PART];
    bool active[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) {
        int local = localBase + x;
        active[x] = local < count;
        alphas[x] = active[x] ? routeScales[start + local] : 0.0f;
        sdata[x][tid] = 0.0f;
    }

    uint8_t *B = weights[expert];
    float *scales = scalesPtrs[expert];
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    const uint8_t *baseB = B + (size_t)st * m;
    float *rowScales = scales + (st / blockK) * ms;
    union_half4 regA;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float curScale = rowScales[i / blockM];
        if (remaining >= 4) {
            uint32_t bb = *(uint32_t*)(baseB + i);
            __half2 B01 = make_half2(__short_as_half((((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 7)),
                                     __short_as_half((((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 7)));
            __half2 B23 = make_half2(__short_as_half((((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7)),
                                     __short_as_half((((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7)));
#pragma unroll
            for (int x = 0; x < PART; x++) {
                if (!active[x]) {
                    continue;
                }
                regA.in = *reinterpret_cast<const uint2 *>(A + (size_t)(start + localBase + x) * m + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
                sdata[x][tid] += ((float)regA.out[0] * (float)B01.x +
                                  (float)regA.out[1] * (float)B01.y +
                                  (float)regA.out[2] * (float)B23.x +
                                  (float)regA.out[3] * (float)B23.y) * curScale;
#else
                __half2 p01 = __hmul2(regA.out2[0], B01);
                __half2 p23 = __hmul2(regA.out2[1], B23);
                __half2 sumHalvesVec = __hadd2(p01, p23);
                __half sumH = __hadd(sumHalvesVec.x, sumHalvesVec.y);
                sdata[x][tid] += __half2float(sumH) * curScale;
#endif
            }
        } else {
            for (int j = 0; j < remaining; j++) {
                half bVal = __float2half((float)__ushort_as_half(((baseB[i + j] & 0x80) << 8) | ((baseB[i + j] & 0x7F) << 7)) * curScale);
#pragma unroll
                for (int x = 0; x < PART; x++) {
                    if (!active[x]) {
                        continue;
                    }
                    sdata[x][tid] += __half2float(A[(size_t)(start + localBase + x) * m + i + j]) * __half2float(bVal);
                }
            }
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += sdata[x][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            if (active[x]) {
                float value = (float)(half)(sdata[x][0] * magicScaleConstant) * alphas[x];
                C[(size_t)(start + localBase + x) * k + st] = __float2half(value);
            }
        }
    }
}

__global__ void FastllmGroupedMoeReduceOutputKernel(half *partOutput, const int *routePositions,
                                                    half *output, int batch, int topk, int hidden) {
    int st = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;
    if (token >= batch || st >= hidden) {
        return;
    }

    float value = 0.0f;
    for (int slot = 0; slot < topk; slot++) {
        int pos = routePositions[token * topk + slot];
        value += __half2float(partOutput[(size_t)pos * hidden + st]);
    }
    output[(size_t)token * hidden + st] = __float2half(value);
}

template <int PART>
void LaunchFastllmGemmFp16FP8E4M3GroupedTopKSwigluIndexed(half *input, const int *routeRows,
                                                          const int *expertStarts, const int *expertCounts,
                                                          uint8_t **weights, float **scales, half *output,
                                                          int experts, int maxExpertTasks, int m, int k,
                                                          int blockM, int blockK) {
    int maxChunks = (maxExpertTasks + PART - 1) / PART;
    dim3 grid(k, experts, maxChunks);
    FastllmGemvHalfFP8E4M3GroupedTopKSwigluIndexedKernel<64, PART> <<< grid, 64 >>>(
        input, routeRows, expertStarts, expertCounts, weights, scales, output, maxChunks, m, k, blockM, blockK);
}

template <int PART>
void LaunchFastllmGemmFp16FP8E4M3GroupedTopKDownScatterIndexed(half *input, const int *routeRows, const float *routeScales,
                                                               const int *expertStarts, const int *expertCounts,
                                                               uint8_t **weights, float **scales, half *output,
                                                               int experts, int maxExpertTasks, int m, int k,
                                                               int blockM, int blockK) {
    int maxChunks = (maxExpertTasks + PART - 1) / PART;
    dim3 grid(k, experts, maxChunks);
    FastllmGemvHalfFP8E4M3GroupedTopKDownScatterIndexedKernel<64, PART> <<< grid, 64 >>>(
        input, routeRows, routeScales, expertStarts, expertCounts, weights, scales, output, maxChunks, m, k, blockM, blockK);
}

void LaunchFastllmGroupedMoeReduceOutput(half *partOutput, const int *routePositions,
                                         half *output, int batch, int topk, int hidden) {
    dim3 block(256);
    dim3 grid((hidden + block.x - 1) / block.x, batch);
    FastllmGroupedMoeReduceOutputKernel <<< grid, block >>>(partOutput, routePositions, output, batch, topk, hidden);
}

__device__ __forceinline__ uint8_t FastllmFloatToFp8E4M3Byte(float value) {
#if (CUDART_VERSION >= 12000)
    __nv_fp8_e4m3 fp8(value);
    return *reinterpret_cast<uint8_t*>(&fp8);
#else
    return 0;
#endif
}

void LaunchFastllmGemmFp16FP8E4M3Block128Swiglu(half *input, uint8_t *weight, half *output, int m, int k, int perRow) {
    FastllmGemvHalfFP8E4M3Block128SwigluKernel<64> <<< k, 64 >>>(input, weight, output, m, k, perRow);
}

void LaunchFastllmGemmFp16FP8E4M3Block128AddTo(half *input, uint8_t *weight, half *output, float alpha, bool overwrite, int m, int k, int perRow) {
    FastllmGemvHalfFP8E4M3Block128AddToKernel<64> <<< k, 64 >>>(input, weight, output, alpha, overwrite, m, k, perRow);
}

bool FastllmCudaHalfMatMulFloatFP8E4M3Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (n != 1 || bias.dims.size() > 0 || weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.blockM <= 0 || weight.blockK <= 0) {
        return false;
    }
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, k);
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        output.dataDevice = fastllm::DataDevice::CUDA;
        output.dataDeviceIds = input.dataDeviceIds;
    }
    output.Allocate(false);
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);
    float *cudaScales = (float*)weight.extraCudaData[0];
    LaunchFastllmGemmFp16FP8E4M3Swiglu(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaScales, m, k, weight.blockM, weight.blockK);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloatFP8E4M3AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k) {
    if (n != 1 || weight.dataType != fastllm::DataType::FP8_E4M3 || output.dataType != fastllm::DataType::FLOAT16 ||
        weight.blockM <= 0 || weight.blockK <= 0) {
        return false;
    }
    fastllm::Data emptyBias;
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, emptyBias, k);
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        output.dataDevice = fastllm::DataDevice::CUDA;
        output.dataDeviceIds = input.dataDeviceIds;
    }
    output.Allocate(false);
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);
    float *cudaScales = (float*)weight.extraCudaData[0];
    LaunchFastllmGemmFp16FP8E4M3AddTo(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaScales, alpha, overwrite, m, k, weight.blockM, weight.blockK);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

struct FastllmMoeFp8Batch1Scratch {
    int capacity = 0;
    uint8_t **gateWeights = nullptr;
    float **gateScales = nullptr;
    uint8_t **downWeights = nullptr;
    float **downScales = nullptr;
    float *scores = nullptr;
};

static std::map<int, FastllmMoeFp8Batch1Scratch> fastllmMoeFp8Batch1Scratch;

static FastllmMoeFp8Batch1Scratch &FastllmGetMoeFp8Batch1Scratch(int topk) {
    int deviceId = FastllmCudaGetDevice();
    FastllmMoeFp8Batch1Scratch &scratch = fastllmMoeFp8Batch1Scratch[deviceId];
    if (scratch.capacity < topk) {
        scratch.capacity = topk;
        size_t ptrBytes = (size_t)topk * sizeof(void*);
        scratch.gateWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
        scratch.gateScales = (float**)FastllmCudaMalloc(ptrBytes);
        scratch.downWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
        scratch.downScales = (float**)FastllmCudaMalloc(ptrBytes);
        scratch.scores = (float*)FastllmCudaMalloc((size_t)topk * sizeof(float));
    }
    return scratch;
}

struct FastllmMoeFp8ExpertTable {
    bool inited = false;
    int experts = 0;
    int hidden = 0;
    int inter = 0;
    int gateBlockM = 0;
    int gateBlockK = 0;
    int downBlockM = 0;
    int downBlockK = 0;
    uint8_t **gateWeights = nullptr;
    float **gateScales = nullptr;
    uint8_t **downWeights = nullptr;
    float **downScales = nullptr;
};

static std::map<std::pair<int, const void*>, FastllmMoeFp8ExpertTable> fastllmMoeFp8ExpertTables;

static bool FastllmGetMoeFp8ExpertTable(fastllm::Data **weights, int weightsBatch, int hidden, int inter,
                                        FastllmMoeFp8ExpertTable *&table) {
    if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1)) {
        return false;
    }
    int experts = weightsBatch / 2 - 1;
    if (experts <= 0) {
        return false;
    }

    int deviceId = FastllmCudaGetDevice();
    auto key = std::make_pair(deviceId, (const void*)weights[2]);
    FastllmMoeFp8ExpertTable &cached = fastllmMoeFp8ExpertTables[key];
    if (cached.inited) {
        if (cached.experts != experts || cached.hidden != hidden || cached.inter != inter) {
            return false;
        }
        table = &cached;
        return true;
    }

    fastllm::Data emptyBias;
    std::vector<uint8_t*> hGateWeights(experts), hDownWeights(experts);
    std::vector<float*> hGateScales(experts), hDownScales(experts);
    int gateBlockM = -1, gateBlockK = -1, downBlockM = -1, downBlockK = -1;

    for (int e = 0; e < experts; e++) {
        int idx = (e + 1) * 2;
        fastllm::Data *gateup = weights[idx];
        fastllm::Data *down = weights[idx + 1];
        if (gateup == nullptr || down == nullptr ||
            gateup->dataType != fastllm::DataType::FP8_E4M3 || down->dataType != fastllm::DataType::FP8_E4M3 ||
            gateup->dims.size() != 2 || down->dims.size() != 2 ||
            gateup->dims[1] != hidden || gateup->dims[0] != inter * 2 ||
            down->dims[1] != inter || down->dims[0] != hidden ||
            gateup->blockM <= 0 || gateup->blockK <= 0 ||
            down->blockM <= 0 || down->blockK <= 0 ||
            gateup->cudaData == nullptr || down->cudaData == nullptr) {
            return false;
        }
        if (gateBlockM < 0) {
            gateBlockM = gateup->blockM;
            gateBlockK = gateup->blockK;
            downBlockM = down->blockM;
            downBlockK = down->blockK;
        } else if (gateBlockM != gateup->blockM || gateBlockK != gateup->blockK ||
                   downBlockM != down->blockM || downBlockK != down->blockK) {
            return false;
        }

        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(*gateup, emptyBias, inter);
        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(*down, emptyBias, hidden);
        if (gateup->extraCudaData.size() == 0 || down->extraCudaData.size() == 0) {
            return false;
        }
        hGateWeights[e] = (uint8_t*)gateup->cudaData;
        hDownWeights[e] = (uint8_t*)down->cudaData;
        hGateScales[e] = (float*)gateup->extraCudaData[0];
        hDownScales[e] = (float*)down->extraCudaData[0];
    }

    size_t ptrBytes = (size_t)experts * sizeof(void*);
    cached.gateWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
    cached.gateScales = (float**)FastllmCudaMalloc(ptrBytes);
    cached.downWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
    cached.downScales = (float**)FastllmCudaMalloc(ptrBytes);

    cudaError_t state = cudaSuccess;
    state = cudaMemcpyAsync(cached.gateWeights, hGateWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when caching MoE gate pointer table!", state);
    state = cudaMemcpyAsync(cached.gateScales, hGateScales.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when caching MoE gate scale table!", state);
    state = cudaMemcpyAsync(cached.downWeights, hDownWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when caching MoE down pointer table!", state);
    state = cudaMemcpyAsync(cached.downScales, hDownScales.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when caching MoE down scale table!", state);

    cached.inited = true;
    cached.experts = experts;
    cached.hidden = hidden;
    cached.inter = inter;
    cached.gateBlockM = gateBlockM;
    cached.gateBlockK = gateBlockK;
    cached.downBlockM = downBlockM;
    cached.downBlockK = downBlockK;
    table = &cached;
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                 fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                 const float *scores, int topk, int hidden, int inter) {
    if (topk <= 0 || hidden <= 0 || inter <= 0 || indices == nullptr || scores == nullptr ||
        input.dataType != fastllm::DataType::FLOAT16 || input.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }

    FastllmMoeFp8ExpertTable *table = nullptr;
    if (!FastllmGetMoeFp8ExpertTable(weights, weightsBatch, hidden, inter, table)) {
        return false;
    }

    w1.dataDevice = input.dataDevice;
    w1.dataDeviceIds = input.dataDeviceIds;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = fastllm::DataType::FLOAT16;
    w1.Resize({topk, inter});
    output.dataType = fastllm::DataType::FLOAT16;
    output.Resize({1, hidden});
    w1.Allocate(false);
    output.Allocate(false);

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaW1 = (half*)FastllmCudaPrepareOutput(w1);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    LaunchFastllmGemmFp16FP8E4M3TopKSwigluIndexed(cudaInput, indices, table->gateWeights, table->gateScales, cudaW1,
                                                  topk, hidden, inter, table->gateBlockM, table->gateBlockK);
    LaunchFastllmGemmFp16FP8E4M3TopKDownReduceIndexed(cudaW1, indices, table->downWeights, table->downScales, cudaOutput, scores,
                                                      topk, inter, hidden, table->downBlockM, table->downBlockK);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                     fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                     const float *scores, int batch, int topk, int hidden, int inter) {
    if (batch <= 0 || batch > 64 || topk <= 0 || hidden <= 0 || inter <= 0 || indices == nullptr || scores == nullptr ||
        input.dataType != fastllm::DataType::FLOAT16 || input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dims.size() == 0 || input.dims[0] != batch) {
        return false;
    }

    FastllmMoeFp8ExpertTable *table = nullptr;
    if (!FastllmGetMoeFp8ExpertTable(weights, weightsBatch, hidden, inter, table)) {
        return false;
    }

    w1.dataDevice = input.dataDevice;
    w1.dataDeviceIds = input.dataDeviceIds;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = fastllm::DataType::FLOAT16;
    w1.Resize({batch * topk, inter});
    output.dataType = fastllm::DataType::FLOAT16;
    output.Resize({batch, hidden});
    w1.Allocate(false);
    output.Allocate(false);

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaW1 = (half*)FastllmCudaPrepareOutput(w1);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    LaunchFastllmGemmFp16FP8E4M3SmallBatchTopKSwigluIndexed(cudaInput, indices, table->gateWeights, table->gateScales, cudaW1,
                                                            batch, topk, hidden, inter, table->gateBlockM, table->gateBlockK);
    LaunchFastllmGemmFp16FP8E4M3SmallBatchTopKDownReduceIndexed(cudaW1, indices, table->downWeights, table->downScales, cudaOutput, scores,
                                                                batch, topk, inter, hidden, table->downBlockM, table->downBlockK);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                  fastllm::Data **weights, int weightsBatch,
                                                  const int *routeRows, const float *routeScales,
                                                  const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                  int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter) {
    if (batch <= 0 || topk <= 0 || totalTasks <= 0 || maxExpertTasks <= 0 || hidden <= 0 || inter <= 0 ||
        routeRows == nullptr || routeScales == nullptr || routePositions == nullptr ||
        expertStarts == nullptr || expertCounts == nullptr ||
        input.dataType != fastllm::DataType::FLOAT16 || input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dims.size() == 0 || input.dims[0] != batch) {
        return false;
    }

    FastllmMoeFp8ExpertTable *table = nullptr;
    if (!FastllmGetMoeFp8ExpertTable(weights, weightsBatch, hidden, inter, table)) {
        return false;
    }

    int experts = table->experts;
    w1.dataDevice = input.dataDevice;
    w1.dataDeviceIds = input.dataDeviceIds;
    w2.dataDevice = input.dataDevice;
    w2.dataDeviceIds = input.dataDeviceIds;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = fastllm::DataType::FLOAT16;
    w1.Resize({totalTasks, inter});
    w2.dataType = fastllm::DataType::FLOAT16;
    w2.Resize({totalTasks, hidden});
    output.dataType = fastllm::DataType::FLOAT16;
    output.Resize({batch, hidden});
    w1.Allocate(false);
    w2.Allocate(false);
    output.Allocate(false);

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaW1 = (half*)FastllmCudaPrepareOutput(w1);
    half *cudaW2 = (half*)FastllmCudaPrepareOutput(w2);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    int *cudaRouteRows = (int*)FastllmCudaMalloc((size_t)totalTasks * sizeof(int));
    float *cudaRouteScales = (float*)FastllmCudaMalloc((size_t)totalTasks * sizeof(float));
    int *cudaRoutePositions = (int*)FastllmCudaMalloc((size_t)batch * topk * sizeof(int));
    int *cudaExpertStarts = (int*)FastllmCudaMalloc((size_t)experts * sizeof(int));
    int *cudaExpertCounts = (int*)FastllmCudaMalloc((size_t)experts * sizeof(int));

    cudaError_t state = cudaSuccess;
    state = cudaMemcpyAsync(cudaRouteRows, routeRows, (size_t)totalTasks * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copying grouped MoE route rows!", state);
    state = cudaMemcpyAsync(cudaRouteScales, routeScales, (size_t)totalTasks * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copying grouped MoE route scales!", state);
    state = cudaMemcpyAsync(cudaRoutePositions, routePositions, (size_t)batch * topk * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copying grouped MoE route positions!", state);
    state = cudaMemcpyAsync(cudaExpertStarts, expertStarts, (size_t)experts * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copying grouped MoE expert starts!", state);
    state = cudaMemcpyAsync(cudaExpertCounts, expertCounts, (size_t)experts * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copying grouped MoE expert counts!", state);

    if (maxExpertTasks <= 8) {
        LaunchFastllmGemmFp16FP8E4M3GroupedTopKSwigluIndexed<8>(
            cudaInput, cudaRouteRows, cudaExpertStarts, cudaExpertCounts,
            table->gateWeights, table->gateScales, cudaW1,
            experts, maxExpertTasks, hidden, inter, table->gateBlockM, table->gateBlockK);
        LaunchFastllmGemmFp16FP8E4M3GroupedTopKDownScatterIndexed<8>(
            cudaW1, cudaRouteRows, cudaRouteScales, cudaExpertStarts, cudaExpertCounts,
            table->downWeights, table->downScales, cudaW2,
            experts, maxExpertTasks, inter, hidden, table->downBlockM, table->downBlockK);
    } else {
        LaunchFastllmGemmFp16FP8E4M3GroupedTopKSwigluIndexed<16>(
            cudaInput, cudaRouteRows, cudaExpertStarts, cudaExpertCounts,
            table->gateWeights, table->gateScales, cudaW1,
            experts, maxExpertTasks, hidden, inter, table->gateBlockM, table->gateBlockK);
        LaunchFastllmGemmFp16FP8E4M3GroupedTopKDownScatterIndexed<16>(
            cudaW1, cudaRouteRows, cudaRouteScales, cudaExpertStarts, cudaExpertCounts,
            table->downWeights, table->downScales, cudaW2,
            experts, maxExpertTasks, inter, hidden, table->downBlockM, table->downBlockK);
    }
    LaunchFastllmGroupedMoeReduceOutput(cudaW2, cudaRoutePositions, cudaOutput, batch, topk, hidden);

    FastllmCudaFree(cudaRouteRows);
    FastllmCudaFree(cudaRouteScales);
    FastllmCudaFree(cudaRoutePositions);
    FastllmCudaFree(cudaExpertStarts);
    FastllmCudaFree(cudaExpertCounts);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                          fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                          bool scoresOnCuda, int topk, int hidden, int inter) {
    if (topk <= 0 || hidden <= 0 || inter <= 0 ||
        input.dataType != fastllm::DataType::FLOAT16 || input.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }

    int gateBlockM = gateups[0]->blockM, gateBlockK = gateups[0]->blockK;
    int downBlockM = downs[0]->blockM, downBlockK = downs[0]->blockK;
    if (gateBlockM <= 0 || gateBlockK <= 0 || downBlockM <= 0 || downBlockK <= 0) {
        return false;
    }

    fastllm::Data emptyBias;
    std::vector<uint8_t*> hGateWeights(topk), hDownWeights(topk);
    std::vector<float*> hGateScales(topk), hDownScales(topk);
    for (int i = 0; i < topk; i++) {
        fastllm::Data *gateup = gateups[i];
        fastllm::Data *down = downs[i];
        if (gateup == nullptr || down == nullptr ||
            gateup->dataType != fastllm::DataType::FP8_E4M3 || down->dataType != fastllm::DataType::FP8_E4M3 ||
            gateup->blockM != gateBlockM || gateup->blockK != gateBlockK ||
            down->blockM != downBlockM || down->blockK != downBlockK ||
            gateup->dims.size() != 2 || down->dims.size() != 2 ||
            gateup->dims[1] != hidden || gateup->dims[0] != inter * 2 ||
            down->dims[1] != inter || down->dims[0] != hidden) {
            return false;
        }
        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(*gateup, emptyBias, inter);
        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(*down, emptyBias, hidden);
        hGateWeights[i] = (uint8_t*)gateup->cudaData;
        hDownWeights[i] = (uint8_t*)down->cudaData;
        hGateScales[i] = (float*)gateup->extraCudaData[0];
        hDownScales[i] = (float*)down->extraCudaData[0];
    }

    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        w1.dataDevice = fastllm::DataDevice::CUDA;
        w1.dataDeviceIds = input.dataDeviceIds;
        output.dataDevice = fastllm::DataDevice::CUDA;
        output.dataDeviceIds = input.dataDeviceIds;
    }
    w1.dataType = fastllm::DataType::FLOAT16;
    w1.Resize({topk, inter});
    output.dataType = fastllm::DataType::FLOAT16;
    output.Resize({1, hidden});
    w1.Allocate(false);
    output.Allocate(false);

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaW1 = (half*)FastllmCudaPrepareOutput(w1);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    FastllmMoeFp8Batch1Scratch &scratch = FastllmGetMoeFp8Batch1Scratch(topk);
    size_t ptrBytes = (size_t)topk * sizeof(void*);

    cudaError_t state = cudaSuccess;
    state = cudaMemcpyAsync(scratch.gateWeights, hGateWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when moving MoE gate pointers to device!", state);
    state = cudaMemcpyAsync(scratch.gateScales, hGateScales.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when moving MoE gate scales to device!", state);
    state = cudaMemcpyAsync(scratch.downWeights, hDownWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when moving MoE down pointers to device!", state);
    state = cudaMemcpyAsync(scratch.downScales, hDownScales.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when moving MoE down scales to device!", state);
    float *cudaScores = scratch.scores;
    if (scoresOnCuda) {
        cudaScores = (float*)scores;
    } else {
        state = cudaMemcpyAsync(scratch.scores, scores, (size_t)topk * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaErrors("Error: CUDA error when moving MoE scores to device!", state);
    }

    LaunchFastllmGemmFp16FP8E4M3TopKSwiglu(cudaInput, scratch.gateWeights, scratch.gateScales, cudaW1,
                                           topk, hidden, inter, gateBlockM, gateBlockK);
    LaunchFastllmGemmFp16FP8E4M3TopKDownReduce(cudaW1, scratch.downWeights, scratch.downScales, cudaOutput, cudaScores,
                                               topk, inter, hidden, downBlockM, downBlockK);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMatMulFloatFP8E4M3Block128Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (n != 1 || bias.dims.size() > 0 || weight.dataType != fastllm::DataType::FP8_E4M3_BLOCK_128) {
        return false;
    }
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        output.dataDevice = fastllm::DataDevice::CUDA;
        output.dataDeviceIds = input.dataDeviceIds;
    }
    output.Allocate(false);
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);
    size_t perRow = m + ((m - 1) / 128 + 1) * sizeof(float);
    LaunchFastllmGemmFp16FP8E4M3Block128Swiglu(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, m, k, perRow);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloatFP8E4M3Block128AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k) {
    if (n != 1 || weight.dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 || output.dataType != fastllm::DataType::FLOAT16) {
        return false;
    }
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        output.dataDevice = fastllm::DataDevice::CUDA;
        output.dataDeviceIds = input.dataDeviceIds;
    }
    output.Allocate(false);
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);
    size_t perRow = m + ((m - 1) / 128 + 1) * sizeof(float);
    LaunchFastllmGemmFp16FP8E4M3Block128AddTo(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, alpha, overwrite, m, k, perRow);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
