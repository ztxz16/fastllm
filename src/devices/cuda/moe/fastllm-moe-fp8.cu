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
#include <type_traits>
#include <vector>

#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#endif

template <typename T>
struct FastllmMoeFp8Traits;

template <>
struct FastllmMoeFp8Traits<half> {
    static constexpr fastllm::DataType dataType = fastllm::DataType::FLOAT16;

    __device__ __forceinline__ static float toFloat(half value) {
        return __half2float(value);
    }

    __device__ __forceinline__ static half fromFloat(float value) {
        return __float2half(value);
    }

    __device__ __forceinline__ static float fp8ToFloat(uint8_t value) {
        return __half2float(__ushort_as_half(((value & 0x80) << 8) | ((value & 0x7F) << 7)));
    }

    __device__ __forceinline__ static float magicScale() {
        return exp2f(8.0f);
    }
};

template <>
struct FastllmMoeFp8Traits<__nv_bfloat16> {
    static constexpr fastllm::DataType dataType = fastllm::DataType::BFLOAT16;

    __device__ __forceinline__ static float toFloat(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }

    __device__ __forceinline__ static __nv_bfloat16 fromFloat(float value) {
        return __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ static float fp8ToFloat(uint8_t value) {
        uint16_t bits = ((value & 0x80) << 8) | ((value & 0x7F) << 4);
        return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&bits));
    }

    __device__ __forceinline__ static float magicScale() {
        return exp2f(120.0f);
    }
};

template <typename T>
__device__ __forceinline__ float FastllmMoeFp8Round(float value) {
    return FastllmMoeFp8Traits<T>::toFloat(FastllmMoeFp8Traits<T>::fromFloat(value));
}

template <typename T>
__device__ __forceinline__ void FastllmMoeFp8Accumulate4(const T *A, int offset, uint32_t bytes, float scale, float &sum) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
        sum += FastllmMoeFp8Traits<T>::toFloat(A[offset + j]) *
               FastllmMoeFp8Traits<T>::fp8ToFloat((uint8_t)(bytes >> (j * 8))) * scale;
    }
}

template <typename T>
__device__ __forceinline__ void FastllmMoeFp8AccumulateRemainder(const T *A, const uint8_t *B, int offset, int remaining,
                                                                 float scale, float &sum) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j < remaining) {
            sum += FastllmMoeFp8Traits<T>::toFloat(A[offset + j]) *
                   FastllmMoeFp8Traits<T>::fp8ToFloat(B[offset + j]) * scale;
        }
    }
}


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

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3Block128SwigluKernel(half *A, uint8_t *B, half *C, int m, int k, int perRow) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    const int blockSize = 128;
    const float magicScaleConstant = exp2f(8.0f);
    const uint8_t *baseGate = B + (size_t)p * perRow;
    const uint8_t *baseUp = B + (size_t)(p + k) * perRow;
    int numBlocks = (m - 1) / blockSize + 1;
    union_half4 regA;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;

    for (int blk = 0; blk < numBlocks; blk++) {
        int blkStart = blk * blockSize;
        int blkEnd = min(blkStart + blockSize, m);
        const uint8_t *gateBlock = baseGate + blk * (blockSize + sizeof(float));
        const uint8_t *upBlock = baseUp + blk * (blockSize + sizeof(float));
        float gateScale = *(float*)(gateBlock + blockSize);
        float upScale = *(float*)(upBlock + blockSize);

        for (int i = blkStart + tid * 4; i < blkEnd; i += THREAD_PER_BLOCK * 4) {
            int localIdx = i - blkStart;
            int remaining = blkEnd - i;
            if (remaining >= 4) {
                uint32_t gateBytes = *(uint32_t*)(gateBlock + localIdx);
                uint32_t upBytes = *(uint32_t*)(upBlock + localIdx);
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
                    half gateVal = __float2half((float)__ushort_as_half(((gateBlock[localIdx + j] & 0x80) << 8) | ((gateBlock[localIdx + j] & 0x7F) << 7)) * gateScale);
                    half upVal = __float2half((float)__ushort_as_half(((upBlock[localIdx + j] & 0x80) << 8) | ((upBlock[localIdx + j] & 0x7F) << 7)) * upScale);
                    float aVal = __half2float(A[i + j]);
                    sdataGate[tid] += aVal * __half2float(gateVal);
                    sdataUp[tid] += aVal * __half2float(upVal);
                }
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
__global__ void FastllmGemvHalfFP8E4M3Block128AddToKernel(half *A, uint8_t *B, half *C, float alpha, bool overwrite, int m, int k, int perRow) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    const int blockSize = 128;
    const float magicScaleConstant = exp2f(8.0f);
    const uint8_t *baseB = B + (size_t)st * perRow;
    int numBlocks = (m - 1) / blockSize + 1;
    union_half4 regA;

    sdata[tid] = 0.0f;
    for (int blk = 0; blk < numBlocks; blk++) {
        int blkStart = blk * blockSize;
        int blkEnd = min(blkStart + blockSize, m);
        const uint8_t *blkData = baseB + blk * (blockSize + sizeof(float));
        float blkScale = *(float*)(blkData + blockSize);

        for (int i = blkStart + tid * 4; i < blkEnd; i += THREAD_PER_BLOCK * 4) {
            int localIdx = i - blkStart;
            int remaining = blkEnd - i;
            if (remaining >= 4) {
                uint32_t bb = *(uint32_t*)(blkData + localIdx);
                __half2 B01 = make_half2(__short_as_half((((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 7)),
                                         __short_as_half((((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 7)));
                __half2 B23 = make_half2(__short_as_half((((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7)),
                                         __short_as_half((((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7)));
                regA.in = *reinterpret_cast<const uint2 *>(A + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
                sdata[tid] += ((float)regA.out[0] * (float)B01.x +
                               (float)regA.out[1] * (float)B01.y +
                               (float)regA.out[2] * (float)B23.x +
                               (float)regA.out[3] * (float)B23.y) * blkScale;
#else
                __half2 p01 = __hmul2(regA.out2[0], B01);
                __half2 p23 = __hmul2(regA.out2[1], B23);
                __half2 sumHalvesVec = __hadd2(p01, p23);
                __half sumH = __hadd(sumHalvesVec.x, sumHalvesVec.y);
                sdata[tid] += __half2float(sumH) * blkScale;
#endif
            } else {
                for (int j = 0; j < remaining; j++) {
                    half bVal = __float2half((float)__ushort_as_half(((blkData[localIdx + j] & 0x80) << 8) | ((blkData[localIdx + j] & 0x7F) << 7)) * blkScale);
                    sdata[tid] += __half2float(A[i + j]) * __half2float(bVal);
                }
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

void LaunchFastllmGemmFp16FP8E4M3Block128Swiglu(half *input, uint8_t *weight, half *output, int m, int k, int perRow) {
    FastllmGemvHalfFP8E4M3Block128SwigluKernel<64> <<< k, 64 >>>(input, weight, output, m, k, perRow);
}

void LaunchFastllmGemmFp16FP8E4M3Block128AddTo(half *input, uint8_t *weight, half *output, float alpha, bool overwrite, int m, int k, int perRow) {
    FastllmGemvHalfFP8E4M3Block128AddToKernel<64> <<< k, 64 >>>(input, weight, output, alpha, overwrite, m, k, perRow);
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3SwigluKernel(T *A, uint8_t *B, T *C, float *scales,
                                                    int m, int k, int blockM, int blockK) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(baseGate + i), gateScale, sdataGate[tid]);
            FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(baseUp + i), upScale, sdataUp[tid]);
        } else {
            FastllmMoeFp8AccumulateRemainder(A, baseGate, i, remaining, gateScale, sdataGate[tid]);
            FastllmMoeFp8AccumulateRemainder(A, baseUp, i, remaining, upScale, sdataUp[tid]);
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
        float gate = FastllmMoeFp8Round<T>(sdataGate[0] * magicScaleConstant);
        float up = FastllmMoeFp8Round<T>(sdataUp[0] * magicScaleConstant);
        C[p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3AddToKernel(T *A, uint8_t *B, T *C, float *scales,
                                                   float alpha, bool overwrite, int m, int k,
                                                   int blockM, int blockK) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const uint8_t *baseB = B + (size_t)st * m;
    float *rowScales = scales + (st / blockK) * ms;

    sdata[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float curScale = rowScales[i / blockM];
        if (remaining >= 4) {
            FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(baseB + i), curScale, sdata[tid]);
        } else {
            FastllmMoeFp8AccumulateRemainder(A, baseB, i, remaining, curScale, sdata[tid]);
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
        float value = FastllmMoeFp8Round<T>(sdata[0] * magicScaleConstant) * alpha;
        if (!overwrite) {
            value += FastllmMoeFp8Traits<T>::toFloat(C[st]);
        }
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(value);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3TopKSwigluKernel(T *A, uint8_t **weights, float **scalesPtrs,
                                                        T *C, int m, int k, int blockM, int blockK) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int expert = blockIdx.y;
    uint8_t *B = weights[expert];
    float *scales = scalesPtrs[expert];
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(baseGate + i), gateScale, sdataGate[tid]);
            FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(baseUp + i), upScale, sdataUp[tid]);
        } else {
            FastllmMoeFp8AccumulateRemainder(A, baseGate, i, remaining, gateScale, sdataGate[tid]);
            FastllmMoeFp8AccumulateRemainder(A, baseUp, i, remaining, upScale, sdataUp[tid]);
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
        float gate = FastllmMoeFp8Round<T>(sdataGate[0] * magicScaleConstant);
        float up = FastllmMoeFp8Round<T>(sdataUp[0] * magicScaleConstant);
        C[(size_t)expert * k + p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3TopKDownReduceKernel(T *A, uint8_t **weights, float **scalesPtrs,
                                                            T *C, float *scores, int topk,
                                                            int m, int k, int blockM, int blockK) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int expert = 0; expert < topk; expert++) {
        uint8_t *B = weights[expert];
        float *scales = scalesPtrs[expert];
        const uint8_t *baseB = B + (size_t)st * m;
        float *rowScales = scales + (st / blockK) * ms;
        T *expertInput = A + (size_t)expert * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            int remaining = m - i;
            float curScale = rowScales[i / blockM];
            if (remaining >= 4) {
                FastllmMoeFp8Accumulate4(expertInput, i, *(uint32_t*)(baseB + i), curScale, sdata[tid]);
            } else {
                FastllmMoeFp8AccumulateRemainder(expertInput, baseB, i, remaining, curScale, sdata[tid]);
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
            out += FastllmMoeFp8Round<T>(sdata[0] * magicScaleConstant) * scores[expert];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3TopKSwigluIndexedKernel(T *A, const int32_t *indices,
                                                               uint8_t **weights, float **scalesPtrs,
                                                               T *C, int topk, int m, int k,
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
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(baseGate + i), gateScale, sdataGate[tid]);
            FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(baseUp + i), upScale, sdataUp[tid]);
        } else {
            FastllmMoeFp8AccumulateRemainder(A, baseGate, i, remaining, gateScale, sdataGate[tid]);
            FastllmMoeFp8AccumulateRemainder(A, baseUp, i, remaining, upScale, sdataUp[tid]);
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
        float gate = FastllmMoeFp8Round<T>(sdataGate[0] * magicScaleConstant);
        float up = FastllmMoeFp8Round<T>(sdataUp[0] * magicScaleConstant);
        C[(size_t)topkSlot * k + p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3TopKDownReduceIndexedKernel(T *A, const int32_t *indices,
                                                                   uint8_t **weights, float **scalesPtrs,
                                                                   T *C, const float *scores, int topk,
                                                                   int m, int k, int blockM, int blockK) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();

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
        T *expertInput = A + (size_t)topkSlot * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            int remaining = m - i;
            float curScale = rowScales[i / blockM];
            if (remaining >= 4) {
                FastllmMoeFp8Accumulate4(expertInput, i, *(uint32_t*)(baseB + i), curScale, sdata[tid]);
            } else {
                FastllmMoeFp8AccumulateRemainder(expertInput, baseB, i, remaining, curScale, sdata[tid]);
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
            out += FastllmMoeFp8Round<T>(sdata[0] * magicScaleConstant) * scores[topkSlot];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3SmallBatchTopKSwigluIndexedKernel(T *A, const int32_t *indices,
                                                                         uint8_t **weights, float **scalesPtrs,
                                                                         T *C, int batch, int topk, int m, int k,
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
    T *tokenInput = A + (size_t)token * m;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            FastllmMoeFp8Accumulate4(tokenInput, i, *(uint32_t*)(baseGate + i), gateScale, sdataGate[tid]);
            FastllmMoeFp8Accumulate4(tokenInput, i, *(uint32_t*)(baseUp + i), upScale, sdataUp[tid]);
        } else {
            FastllmMoeFp8AccumulateRemainder(tokenInput, baseGate, i, remaining, gateScale, sdataGate[tid]);
            FastllmMoeFp8AccumulateRemainder(tokenInput, baseUp, i, remaining, upScale, sdataUp[tid]);
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
        float gate = FastllmMoeFp8Round<T>(sdataGate[0] * magicScaleConstant);
        float up = FastllmMoeFp8Round<T>(sdataUp[0] * magicScaleConstant);
        C[(size_t)task * k + p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3SmallBatchTopKDownReduceIndexedKernel(T *A, const int32_t *indices,
                                                                             uint8_t **weights, float **scalesPtrs,
                                                                             T *C, const float *scores, int batch, int topk,
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
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();

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
        T *expertInput = A + (size_t)task * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            int remaining = m - i;
            float curScale = rowScales[i / blockM];
            if (remaining >= 4) {
                FastllmMoeFp8Accumulate4(expertInput, i, *(uint32_t*)(baseB + i), curScale, sdata[tid]);
            } else {
                FastllmMoeFp8AccumulateRemainder(expertInput, baseB, i, remaining, curScale, sdata[tid]);
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
            out += FastllmMoeFp8Round<T>(sdata[0] * magicScaleConstant) * scores[task];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[(size_t)token * k + st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <typename T, int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvTypedFP8E4M3GroupedTopKSwigluIndexedKernel(T *A, const int *routeRows,
                                                                      const int *expertStarts, const int *expertCounts,
                                                                      uint8_t **weights, float **scalesPtrs,
                                                                      T *C, int maxChunks, int m, int k,
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
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const uint8_t *baseGate = B + (size_t)p * m;
    const uint8_t *baseUp = B + (size_t)(p + k) * m;
    float *gateScales = scales + (p / blockK) * ms;
    float *upScales = scales + ((p + k) / blockK) * ms;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float gateScale = gateScales[i / blockM];
        float upScale = upScales[i / blockM];
        if (remaining >= 4) {
            uint32_t gateBytes = *(uint32_t*)(baseGate + i);
            uint32_t upBytes = *(uint32_t*)(baseUp + i);
#pragma unroll
            for (int x = 0; x < PART; x++) {
                if (active[x]) {
                    T *rowInput = A + (size_t)rows[x] * m;
                    FastllmMoeFp8Accumulate4(rowInput, i, gateBytes, gateScale, sdataGate[x][tid]);
                    FastllmMoeFp8Accumulate4(rowInput, i, upBytes, upScale, sdataUp[x][tid]);
                }
            }
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                if (active[x]) {
                    T *rowInput = A + (size_t)rows[x] * m;
                    FastllmMoeFp8AccumulateRemainder(rowInput, baseGate, i, remaining, gateScale, sdataGate[x][tid]);
                    FastllmMoeFp8AccumulateRemainder(rowInput, baseUp, i, remaining, upScale, sdataUp[x][tid]);
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
                float gate = FastllmMoeFp8Round<T>(sdataGate[x][0] * magicScaleConstant);
                float up = FastllmMoeFp8Round<T>(sdataUp[x][0] * magicScaleConstant);
                C[(size_t)(start + localBase + x) * k + p] =
                    FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
            }
        }
    }
}

template <typename T, int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvTypedFP8E4M3GroupedTopKDownScatterIndexedKernel(T *A, const int *routeRows,
                                                                           const float *routeScales,
                                                                           const int *expertStarts, const int *expertCounts,
                                                                           uint8_t **weights, float **scalesPtrs,
                                                                           T *C, int maxChunks, int m, int k,
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

    (void)routeRows;
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
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const uint8_t *baseB = B + (size_t)st * m;
    float *rowScales = scales + (st / blockK) * ms;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        int remaining = m - i;
        float curScale = rowScales[i / blockM];
        if (remaining >= 4) {
            uint32_t bytes = *(uint32_t*)(baseB + i);
#pragma unroll
            for (int x = 0; x < PART; x++) {
                if (active[x]) {
                    T *rowInput = A + (size_t)(start + localBase + x) * m;
                    FastllmMoeFp8Accumulate4(rowInput, i, bytes, curScale, sdata[x][tid]);
                }
            }
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                if (active[x]) {
                    T *rowInput = A + (size_t)(start + localBase + x) * m;
                    FastllmMoeFp8AccumulateRemainder(rowInput, baseB, i, remaining, curScale, sdata[x][tid]);
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
                float value = FastllmMoeFp8Round<T>(sdata[x][0] * magicScaleConstant) * alphas[x];
                C[(size_t)(start + localBase + x) * k + st] = FastllmMoeFp8Traits<T>::fromFloat(value);
            }
        }
    }
}

template <typename T>
__global__ void FastllmGroupedMoeReduceOutputTypedKernel(T *partOutput, const int *routePositions,
                                                         T *output, int batch, int topk, int hidden) {
    int st = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;
    if (token >= batch || st >= hidden) {
        return;
    }

    float value = 0.0f;
    for (int slot = 0; slot < topk; slot++) {
        int pos = routePositions[token * topk + slot];
        value += FastllmMoeFp8Traits<T>::toFloat(partOutput[(size_t)pos * hidden + st]);
    }
    output[(size_t)token * hidden + st] = FastllmMoeFp8Traits<T>::fromFloat(value);
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3Block128SwigluKernel(T *A, uint8_t *B, T *C,
                                                            int m, int k, int perRow) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    const int blockSize = 128;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const uint8_t *baseGate = B + (size_t)p * perRow;
    const uint8_t *baseUp = B + (size_t)(p + k) * perRow;
    int numBlocks = (m - 1) / blockSize + 1;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int blk = 0; blk < numBlocks; blk++) {
        int blkStart = blk * blockSize;
        int blkEnd = min(blkStart + blockSize, m);
        const uint8_t *gateBlock = baseGate + blk * (blockSize + sizeof(float));
        const uint8_t *upBlock = baseUp + blk * (blockSize + sizeof(float));
        float gateScale = *(float*)(gateBlock + blockSize);
        float upScale = *(float*)(upBlock + blockSize);

        for (int i = blkStart + tid * 4; i < blkEnd; i += THREAD_PER_BLOCK * 4) {
            int localIdx = i - blkStart;
            int remaining = blkEnd - i;
            if (remaining >= 4) {
                FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(gateBlock + localIdx), gateScale, sdataGate[tid]);
                FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(upBlock + localIdx), upScale, sdataUp[tid]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (j < remaining) {
                        float aVal = FastllmMoeFp8Traits<T>::toFloat(A[i + j]);
                        sdataGate[tid] += aVal * FastllmMoeFp8Traits<T>::fp8ToFloat(gateBlock[localIdx + j]) * gateScale;
                        sdataUp[tid] += aVal * FastllmMoeFp8Traits<T>::fp8ToFloat(upBlock[localIdx + j]) * upScale;
                    }
                }
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
        float gate = FastllmMoeFp8Round<T>(sdataGate[0] * magicScaleConstant);
        float up = FastllmMoeFp8Round<T>(sdataUp[0] * magicScaleConstant);
        C[p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3Block128AddToKernel(T *A, uint8_t *B, T *C,
                                                           float alpha, bool overwrite,
                                                           int m, int k, int perRow) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    const int blockSize = 128;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const uint8_t *baseB = B + (size_t)st * perRow;
    int numBlocks = (m - 1) / blockSize + 1;

    sdata[tid] = 0.0f;
    for (int blk = 0; blk < numBlocks; blk++) {
        int blkStart = blk * blockSize;
        int blkEnd = min(blkStart + blockSize, m);
        const uint8_t *blkData = baseB + blk * (blockSize + sizeof(float));
        float blkScale = *(float*)(blkData + blockSize);

        for (int i = blkStart + tid * 4; i < blkEnd; i += THREAD_PER_BLOCK * 4) {
            int localIdx = i - blkStart;
            int remaining = blkEnd - i;
            if (remaining >= 4) {
                FastllmMoeFp8Accumulate4(A, i, *(uint32_t*)(blkData + localIdx), blkScale, sdata[tid]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (j < remaining) {
                        sdata[tid] += FastllmMoeFp8Traits<T>::toFloat(A[i + j]) *
                                      FastllmMoeFp8Traits<T>::fp8ToFloat(blkData[localIdx + j]) * blkScale;
                    }
                }
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
        float value = FastllmMoeFp8Round<T>(sdata[0] * magicScaleConstant) * alpha;
        if (!overwrite) {
            value += FastllmMoeFp8Traits<T>::toFloat(C[st]);
        }
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(value);
    }
}

void LaunchFastllmGemmBF16FP8E4M3Swiglu(__nv_bfloat16 *input, uint8_t *weight, __nv_bfloat16 *output,
                                        float *scales, int m, int k, int blockM, int blockK) {
    FastllmGemvTypedFP8E4M3SwigluKernel<__nv_bfloat16, 64> <<< k, 64 >>>(input, weight, output, scales, m, k, blockM, blockK);
}

void LaunchFastllmGemmBF16FP8E4M3AddTo(__nv_bfloat16 *input, uint8_t *weight, __nv_bfloat16 *output,
                                       float *scales, float alpha, bool overwrite, int m, int k,
                                       int blockM, int blockK) {
    FastllmGemvTypedFP8E4M3AddToKernel<__nv_bfloat16, 64> <<< k, 64 >>>(
        input, weight, output, scales, alpha, overwrite, m, k, blockM, blockK);
}

void LaunchFastllmGemmBF16FP8E4M3TopKSwiglu(__nv_bfloat16 *input, uint8_t **weights, float **scales,
                                            __nv_bfloat16 *output, int topk, int m, int k,
                                            int blockM, int blockK) {
    dim3 grid(k, topk);
    FastllmGemvTypedFP8E4M3TopKSwigluKernel<__nv_bfloat16, 64> <<< grid, 64 >>>(
        input, weights, scales, output, m, k, blockM, blockK);
}

void LaunchFastllmGemmBF16FP8E4M3TopKDownReduce(__nv_bfloat16 *input, uint8_t **weights, float **scales,
                                                __nv_bfloat16 *output, float *scores, int topk,
                                                int m, int k, int blockM, int blockK) {
    FastllmGemvTypedFP8E4M3TopKDownReduceKernel<__nv_bfloat16, 64> <<< k, 64 >>>(
        input, weights, scales, output, scores, topk, m, k, blockM, blockK);
}

void LaunchFastllmGemmBF16FP8E4M3TopKSwigluIndexed(__nv_bfloat16 *input, const int32_t *indices,
                                                   uint8_t **weights, float **scales, __nv_bfloat16 *output,
                                                   int topk, int m, int k, int blockM, int blockK) {
    dim3 grid(k, topk);
    FastllmGemvTypedFP8E4M3TopKSwigluIndexedKernel<__nv_bfloat16, 64> <<< grid, 64 >>>(
        input, indices, weights, scales, output, topk, m, k, blockM, blockK);
}

void LaunchFastllmGemmBF16FP8E4M3TopKDownReduceIndexed(__nv_bfloat16 *input, const int32_t *indices,
                                                       uint8_t **weights, float **scales, __nv_bfloat16 *output,
                                                       const float *scores, int topk,
                                                       int m, int k, int blockM, int blockK) {
    FastllmGemvTypedFP8E4M3TopKDownReduceIndexedKernel<__nv_bfloat16, 64> <<< k, 64 >>>(
        input, indices, weights, scales, output, scores, topk, m, k, blockM, blockK);
}

void LaunchFastllmGemmBF16FP8E4M3SmallBatchTopKSwigluIndexed(__nv_bfloat16 *input, const int32_t *indices,
                                                             uint8_t **weights, float **scales,
                                                             __nv_bfloat16 *output, int batch, int topk,
                                                             int m, int k, int blockM, int blockK) {
    dim3 grid(k, batch * topk);
    FastllmGemvTypedFP8E4M3SmallBatchTopKSwigluIndexedKernel<__nv_bfloat16, 64> <<< grid, 64 >>>(
        input, indices, weights, scales, output, batch, topk, m, k, blockM, blockK);
}

void LaunchFastllmGemmBF16FP8E4M3SmallBatchTopKDownReduceIndexed(__nv_bfloat16 *input, const int32_t *indices,
                                                                 uint8_t **weights, float **scales,
                                                                 __nv_bfloat16 *output, const float *scores,
                                                                 int batch, int topk, int m, int k,
                                                                 int blockM, int blockK) {
    dim3 grid(k, batch);
    FastllmGemvTypedFP8E4M3SmallBatchTopKDownReduceIndexedKernel<__nv_bfloat16, 64> <<< grid, 64 >>>(
        input, indices, weights, scales, output, scores, batch, topk, m, k, blockM, blockK);
}

template <int PART>
void LaunchFastllmGemmBF16FP8E4M3GroupedTopKSwigluIndexed(__nv_bfloat16 *input, const int *routeRows,
                                                          const int *expertStarts, const int *expertCounts,
                                                          uint8_t **weights, float **scales, __nv_bfloat16 *output,
                                                          int experts, int maxExpertTasks, int m, int k,
                                                          int blockM, int blockK) {
    int maxChunks = (maxExpertTasks + PART - 1) / PART;
    dim3 grid(k, experts, maxChunks);
    FastllmGemvTypedFP8E4M3GroupedTopKSwigluIndexedKernel<__nv_bfloat16, 64, PART> <<< grid, 64 >>>(
        input, routeRows, expertStarts, expertCounts, weights, scales, output, maxChunks, m, k, blockM, blockK);
}

template <int PART>
void LaunchFastllmGemmBF16FP8E4M3GroupedTopKDownScatterIndexed(__nv_bfloat16 *input, const int *routeRows,
                                                               const float *routeScales,
                                                               const int *expertStarts, const int *expertCounts,
                                                               uint8_t **weights, float **scales, __nv_bfloat16 *output,
                                                               int experts, int maxExpertTasks, int m, int k,
                                                               int blockM, int blockK) {
    int maxChunks = (maxExpertTasks + PART - 1) / PART;
    dim3 grid(k, experts, maxChunks);
    FastllmGemvTypedFP8E4M3GroupedTopKDownScatterIndexedKernel<__nv_bfloat16, 64, PART> <<< grid, 64 >>>(
        input, routeRows, routeScales, expertStarts, expertCounts, weights, scales, output, maxChunks, m, k, blockM, blockK);
}

void LaunchFastllmGroupedMoeReduceOutputBF16(__nv_bfloat16 *partOutput, const int *routePositions,
                                             __nv_bfloat16 *output, int batch, int topk, int hidden) {
    dim3 block(256);
    dim3 grid((hidden + block.x - 1) / block.x, batch);
    FastllmGroupedMoeReduceOutputTypedKernel<__nv_bfloat16> <<< grid, block >>>(
        partOutput, routePositions, output, batch, topk, hidden);
}

void LaunchFastllmGemmBF16FP8E4M3Block128Swiglu(__nv_bfloat16 *input, uint8_t *weight,
                                                __nv_bfloat16 *output, int m, int k, int perRow) {
    FastllmGemvTypedFP8E4M3Block128SwigluKernel<__nv_bfloat16, 64> <<< k, 64 >>>(
        input, weight, output, m, k, perRow);
}

void LaunchFastllmGemmBF16FP8E4M3Block128AddTo(__nv_bfloat16 *input, uint8_t *weight,
                                               __nv_bfloat16 *output, float alpha, bool overwrite,
                                               int m, int k, int perRow) {
    FastllmGemvTypedFP8E4M3Block128AddToKernel<__nv_bfloat16, 64> <<< k, 64 >>>(
        input, weight, output, alpha, overwrite, m, k, perRow);
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3Swiglu(T *input, uint8_t *weight, T *output, float *scales,
                                                int m, int k, int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3Swiglu(input, weight, output, scales, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3Swiglu(input, weight, output, scales, m, k, blockM, blockK);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3AddTo(T *input, uint8_t *weight, T *output, float *scales,
                                               float alpha, bool overwrite, int m, int k,
                                               int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3AddTo(input, weight, output, scales, alpha, overwrite, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3AddTo(input, weight, output, scales, alpha, overwrite, m, k, blockM, blockK);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3TopKSwiglu(T *input, uint8_t **weights, float **scales,
                                                    T *output, int topk, int m, int k,
                                                    int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3TopKSwiglu(input, weights, scales, output, topk, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3TopKSwiglu(input, weights, scales, output, topk, m, k, blockM, blockK);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3TopKDownReduce(T *input, uint8_t **weights, float **scales,
                                                        T *output, float *scores, int topk, int m, int k,
                                                        int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3TopKDownReduce(input, weights, scales, output, scores, topk, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3TopKDownReduce(input, weights, scales, output, scores, topk, m, k, blockM, blockK);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3TopKSwigluIndexed(T *input, const int32_t *indices,
                                                           uint8_t **weights, float **scales, T *output,
                                                           int topk, int m, int k, int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3TopKSwigluIndexed(input, indices, weights, scales, output, topk, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3TopKSwigluIndexed(input, indices, weights, scales, output, topk, m, k, blockM, blockK);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3TopKDownReduceIndexed(T *input, const int32_t *indices,
                                                               uint8_t **weights, float **scales, T *output,
                                                               const float *scores, int topk, int m, int k,
                                                               int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3TopKDownReduceIndexed(input, indices, weights, scales, output, scores, topk, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3TopKDownReduceIndexed(input, indices, weights, scales, output, scores, topk, m, k, blockM, blockK);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3SmallBatchTopKSwigluIndexed(T *input, const int32_t *indices,
                                                                     uint8_t **weights, float **scales, T *output,
                                                                     int batch, int topk, int m, int k,
                                                                     int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3SmallBatchTopKSwigluIndexed(input, indices, weights, scales, output, batch, topk, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3SmallBatchTopKSwigluIndexed(input, indices, weights, scales, output, batch, topk, m, k, blockM, blockK);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3SmallBatchTopKDownReduceIndexed(T *input, const int32_t *indices,
                                                                         uint8_t **weights, float **scales, T *output,
                                                                         const float *scores, int batch, int topk,
                                                                         int m, int k, int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3SmallBatchTopKDownReduceIndexed(input, indices, weights, scales, output, scores, batch, topk, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3SmallBatchTopKDownReduceIndexed(input, indices, weights, scales, output, scores, batch, topk, m, k, blockM, blockK);
    }
}

template <typename T, int PART>
static void LaunchFastllmGemmTypedFP8E4M3GroupedTopKSwigluIndexed(T *input, const int *routeRows,
                                                                  const int *expertStarts, const int *expertCounts,
                                                                  uint8_t **weights, float **scales, T *output,
                                                                  int experts, int maxExpertTasks, int m, int k,
                                                                  int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3GroupedTopKSwigluIndexed<PART>(
            input, routeRows, expertStarts, expertCounts, weights, scales, output,
            experts, maxExpertTasks, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3GroupedTopKSwigluIndexed<PART>(
            input, routeRows, expertStarts, expertCounts, weights, scales, output,
            experts, maxExpertTasks, m, k, blockM, blockK);
    }
}

template <typename T, int PART>
static void LaunchFastllmGemmTypedFP8E4M3GroupedTopKDownScatterIndexed(T *input, const int *routeRows,
                                                                       const float *routeScales,
                                                                       const int *expertStarts, const int *expertCounts,
                                                                       uint8_t **weights, float **scales, T *output,
                                                                       int experts, int maxExpertTasks, int m, int k,
                                                                       int blockM, int blockK) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3GroupedTopKDownScatterIndexed<PART>(
            input, routeRows, routeScales, expertStarts, expertCounts, weights, scales, output,
            experts, maxExpertTasks, m, k, blockM, blockK);
    } else {
        LaunchFastllmGemmBF16FP8E4M3GroupedTopKDownScatterIndexed<PART>(
            input, routeRows, routeScales, expertStarts, expertCounts, weights, scales, output,
            experts, maxExpertTasks, m, k, blockM, blockK);
    }
}

template <typename T>
static void LaunchFastllmGroupedMoeReduceOutputTyped(T *partOutput, const int *routePositions,
                                                     T *output, int batch, int topk, int hidden) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGroupedMoeReduceOutput(partOutput, routePositions, output, batch, topk, hidden);
    } else {
        LaunchFastllmGroupedMoeReduceOutputBF16(partOutput, routePositions, output, batch, topk, hidden);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3Block128Swiglu(T *input, uint8_t *weight, T *output,
                                                        int m, int k, int perRow) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3Block128Swiglu(input, weight, output, m, k, perRow);
    } else {
        LaunchFastllmGemmBF16FP8E4M3Block128Swiglu(input, weight, output, m, k, perRow);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3Block128AddTo(T *input, uint8_t *weight, T *output,
                                                       float alpha, bool overwrite, int m, int k, int perRow) {
    if constexpr (std::is_same_v<T, half>) {
        LaunchFastllmGemmFp16FP8E4M3Block128AddTo(input, weight, output, alpha, overwrite, m, k, perRow);
    } else {
        LaunchFastllmGemmBF16FP8E4M3Block128AddTo(input, weight, output, alpha, overwrite, m, k, perRow);
    }
}

template <typename T>
static bool FastllmCudaTypedMatMulFP8E4M3Swiglu(const fastllm::Data &input, fastllm::Data &weight,
                                                const fastllm::Data &bias, fastllm::Data &output,
                                                int n, int m, int k) {
    if (n != 1 || bias.dims.size() > 0 || weight.dataType != fastllm::DataType::FP8_E4M3 ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType || weight.blockM <= 0 || weight.blockK <= 0) {
        return false;
    }
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, k);
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        output.dataDevice = fastllm::DataDevice::CUDA;
        output.dataDeviceIds = input.dataDeviceIds;
    }
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Allocate(false);
    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);
    float *cudaScales = (float*)weight.extraCudaData[0];
    LaunchFastllmGemmTypedFP8E4M3Swiglu(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaScales, m, k, weight.blockM, weight.blockK);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

template <typename T>
static bool FastllmCudaTypedMatMulFP8E4M3AddTo(const fastllm::Data &input, fastllm::Data &weight,
                                               fastllm::Data &output, float alpha, bool overwrite,
                                               int n, int m, int k) {
    if (n != 1 || weight.dataType != fastllm::DataType::FP8_E4M3 ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType ||
        output.dataType != FastllmMoeFp8Traits<T>::dataType ||
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
    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);
    float *cudaScales = (float*)weight.extraCudaData[0];
    LaunchFastllmGemmTypedFP8E4M3AddTo(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaScales, alpha, overwrite, m, k, weight.blockM, weight.blockK);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

template <typename T>
static bool FastllmCudaTypedMatMulFP8E4M3Block128Swiglu(const fastllm::Data &input, fastllm::Data &weight,
                                                        const fastllm::Data &bias, fastllm::Data &output,
                                                        int n, int m, int k) {
    if (n != 1 || bias.dims.size() > 0 || weight.dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType) {
        return false;
    }
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        output.dataDevice = fastllm::DataDevice::CUDA;
        output.dataDeviceIds = input.dataDeviceIds;
    }
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Allocate(false);
    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);
    size_t perRow = m + ((m - 1) / 128 + 1) * sizeof(float);
    LaunchFastllmGemmTypedFP8E4M3Block128Swiglu(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, m, k, perRow);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

template <typename T>
static bool FastllmCudaTypedMatMulFP8E4M3Block128AddTo(const fastllm::Data &input, fastllm::Data &weight,
                                                       fastllm::Data &output, float alpha, bool overwrite,
                                                       int n, int m, int k) {
    if (n != 1 || weight.dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType ||
        output.dataType != FastllmMoeFp8Traits<T>::dataType) {
        return false;
    }
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        output.dataDevice = fastllm::DataDevice::CUDA;
        output.dataDeviceIds = input.dataDeviceIds;
    }
    output.Allocate(false);
    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);
    size_t perRow = m + ((m - 1) / 128 + 1) * sizeof(float);
    LaunchFastllmGemmTypedFP8E4M3Block128AddTo(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, alpha, overwrite, m, k, perRow);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloatFP8E4M3Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaTypedMatMulFP8E4M3Swiglu<half>(input, weight, bias, output, n, m, k);
}

bool FastllmCudaHalfMatMulFloatFP8E4M3AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k) {
    return FastllmCudaTypedMatMulFP8E4M3AddTo<half>(input, weight, output, alpha, overwrite, n, m, k);
}

bool FastllmCudaBFloat16MatMulFP8E4M3Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaTypedMatMulFP8E4M3Swiglu<__nv_bfloat16>(input, weight, bias, output, n, m, k);
}

bool FastllmCudaBFloat16MatMulFP8E4M3AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k) {
    return FastllmCudaTypedMatMulFP8E4M3AddTo<__nv_bfloat16>(input, weight, output, alpha, overwrite, n, m, k);
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

template <typename T>
static bool FastllmCudaTypedMergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                         fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                         const float *scores, int topk, int hidden, int inter) {
    if (topk <= 0 || hidden <= 0 || inter <= 0 || indices == nullptr || scores == nullptr ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType || input.dataDevice != fastllm::DataDevice::CUDA) {
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
    w1.dataType = FastllmMoeFp8Traits<T>::dataType;
    w1.Resize({topk, inter});
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Resize({1, hidden});
    w1.Allocate(false);
    output.Allocate(false);

    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaW1 = (T*)FastllmCudaPrepareOutput(w1);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);

    LaunchFastllmGemmTypedFP8E4M3TopKSwigluIndexed(cudaInput, indices, table->gateWeights, table->gateScales, cudaW1,
                                                   topk, hidden, inter, table->gateBlockM, table->gateBlockK);
    LaunchFastllmGemmTypedFP8E4M3TopKDownReduceIndexed(cudaW1, indices, table->downWeights, table->downScales, cudaOutput, scores,
                                                       topk, inter, hidden, table->downBlockM, table->downBlockK);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                 fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                 const float *scores, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3Batch1Indexed<half>(
        input, w1, output, weights, weightsBatch, indices, scores, topk, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                     fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                     const float *scores, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3Batch1Indexed<__nv_bfloat16>(
        input, w1, output, weights, weightsBatch, indices, scores, topk, hidden, inter);
}

template <typename T>
static bool FastllmCudaTypedMergeMOEFP8E4M3SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                             fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                             const float *scores, int batch, int topk, int hidden, int inter) {
    if (batch <= 0 || batch > 64 || topk <= 0 || hidden <= 0 || inter <= 0 || indices == nullptr || scores == nullptr ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType || input.dataDevice != fastllm::DataDevice::CUDA ||
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
    w1.dataType = FastllmMoeFp8Traits<T>::dataType;
    w1.Resize({batch * topk, inter});
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Resize({batch, hidden});
    w1.Allocate(false);
    output.Allocate(false);

    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaW1 = (T*)FastllmCudaPrepareOutput(w1);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);

    LaunchFastllmGemmTypedFP8E4M3SmallBatchTopKSwigluIndexed(cudaInput, indices, table->gateWeights, table->gateScales, cudaW1,
                                                             batch, topk, hidden, inter, table->gateBlockM, table->gateBlockK);
    LaunchFastllmGemmTypedFP8E4M3SmallBatchTopKDownReduceIndexed(cudaW1, indices, table->downWeights, table->downScales, cudaOutput, scores,
                                                                 batch, topk, inter, hidden, table->downBlockM, table->downBlockK);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                     fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                     const float *scores, int batch, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3SmallBatchIndexed<half>(
        input, w1, output, weights, weightsBatch, indices, scores, batch, topk, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOEFP8E4M3SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                         fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                         const float *scores, int batch, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3SmallBatchIndexed<__nv_bfloat16>(
        input, w1, output, weights, weightsBatch, indices, scores, batch, topk, hidden, inter);
}

template <typename T>
static bool FastllmCudaTypedMergeMOEFP8E4M3GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                          fastllm::Data **weights, int weightsBatch,
                                                          const int *routeRows, const float *routeScales,
                                                          const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                          int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter) {
    if (batch <= 0 || topk <= 0 || totalTasks <= 0 || maxExpertTasks <= 0 || hidden <= 0 || inter <= 0 ||
        routeRows == nullptr || routeScales == nullptr || routePositions == nullptr ||
        expertStarts == nullptr || expertCounts == nullptr ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType || input.dataDevice != fastllm::DataDevice::CUDA ||
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
    w1.dataType = FastllmMoeFp8Traits<T>::dataType;
    w1.Resize({totalTasks, inter});
    w2.dataType = FastllmMoeFp8Traits<T>::dataType;
    w2.Resize({totalTasks, hidden});
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Resize({batch, hidden});
    w1.Allocate(false);
    w2.Allocate(false);
    output.Allocate(false);

    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaW1 = (T*)FastllmCudaPrepareOutput(w1);
    T *cudaW2 = (T*)FastllmCudaPrepareOutput(w2);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);

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
        LaunchFastllmGemmTypedFP8E4M3GroupedTopKSwigluIndexed<T, 8>(
            cudaInput, cudaRouteRows, cudaExpertStarts, cudaExpertCounts,
            table->gateWeights, table->gateScales, cudaW1,
            experts, maxExpertTasks, hidden, inter, table->gateBlockM, table->gateBlockK);
        LaunchFastllmGemmTypedFP8E4M3GroupedTopKDownScatterIndexed<T, 8>(
            cudaW1, cudaRouteRows, cudaRouteScales, cudaExpertStarts, cudaExpertCounts,
            table->downWeights, table->downScales, cudaW2,
            experts, maxExpertTasks, inter, hidden, table->downBlockM, table->downBlockK);
    } else {
        LaunchFastllmGemmTypedFP8E4M3GroupedTopKSwigluIndexed<T, 16>(
            cudaInput, cudaRouteRows, cudaExpertStarts, cudaExpertCounts,
            table->gateWeights, table->gateScales, cudaW1,
            experts, maxExpertTasks, hidden, inter, table->gateBlockM, table->gateBlockK);
        LaunchFastllmGemmTypedFP8E4M3GroupedTopKDownScatterIndexed<T, 16>(
            cudaW1, cudaRouteRows, cudaRouteScales, cudaExpertStarts, cudaExpertCounts,
            table->downWeights, table->downScales, cudaW2,
            experts, maxExpertTasks, inter, hidden, table->downBlockM, table->downBlockK);
    }
    LaunchFastllmGroupedMoeReduceOutputTyped(cudaW2, cudaRoutePositions, cudaOutput, batch, topk, hidden);

    FastllmCudaFree(cudaRouteRows);
    FastllmCudaFree(cudaRouteScales);
    FastllmCudaFree(cudaRoutePositions);
    FastllmCudaFree(cudaExpertStarts);
    FastllmCudaFree(cudaExpertCounts);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                  fastllm::Data **weights, int weightsBatch,
                                                  const int *routeRows, const float *routeScales,
                                                  const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                  int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3GroupedIndexed<half>(
        input, w1, w2, output, weights, weightsBatch,
        routeRows, routeScales, routePositions, expertStarts, expertCounts,
        batch, topk, totalTasks, maxExpertTasks, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOEFP8E4M3GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                      fastllm::Data **weights, int weightsBatch,
                                                      const int *routeRows, const float *routeScales,
                                                      const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                      int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3GroupedIndexed<__nv_bfloat16>(
        input, w1, w2, output, weights, weightsBatch,
        routeRows, routeScales, routePositions, expertStarts, expertCounts,
        batch, topk, totalTasks, maxExpertTasks, hidden, inter);
}

template <typename T>
static bool FastllmCudaTypedMergeMOEFP8E4M3Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                  fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                                  bool scoresOnCuda, int topk, int hidden, int inter) {
    if (topk <= 0 || hidden <= 0 || inter <= 0 ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType || input.dataDevice != fastllm::DataDevice::CUDA) {
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
    w1.dataType = FastllmMoeFp8Traits<T>::dataType;
    w1.Resize({topk, inter});
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Resize({1, hidden});
    w1.Allocate(false);
    output.Allocate(false);

    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaW1 = (T*)FastllmCudaPrepareOutput(w1);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);

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

    LaunchFastllmGemmTypedFP8E4M3TopKSwiglu(cudaInput, scratch.gateWeights, scratch.gateScales, cudaW1,
                                            topk, hidden, inter, gateBlockM, gateBlockK);
    LaunchFastllmGemmTypedFP8E4M3TopKDownReduce(cudaW1, scratch.downWeights, scratch.downScales, cudaOutput, cudaScores,
                                                topk, inter, hidden, downBlockM, downBlockK);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                          fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                          bool scoresOnCuda, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3Batch1<half>(
        input, w1, output, gateups, downs, scores, scoresOnCuda, topk, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOEFP8E4M3Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                              fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                              bool scoresOnCuda, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3Batch1<__nv_bfloat16>(
        input, w1, output, gateups, downs, scores, scoresOnCuda, topk, hidden, inter);
}

bool FastllmCudaHalfMatMulFloatFP8E4M3Block128Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaTypedMatMulFP8E4M3Block128Swiglu<half>(input, weight, bias, output, n, m, k);
}

bool FastllmCudaHalfMatMulFloatFP8E4M3Block128AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k) {
    return FastllmCudaTypedMatMulFP8E4M3Block128AddTo<half>(input, weight, output, alpha, overwrite, n, m, k);
}

bool FastllmCudaBFloat16MatMulFP8E4M3Block128Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaTypedMatMulFP8E4M3Block128Swiglu<__nv_bfloat16>(input, weight, bias, output, n, m, k);
}

bool FastllmCudaBFloat16MatMulFP8E4M3Block128AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k) {
    return FastllmCudaTypedMatMulFP8E4M3Block128AddTo<__nv_bfloat16>(input, weight, output, alpha, overwrite, n, m, k);
}
