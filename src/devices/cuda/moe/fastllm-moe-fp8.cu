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
#include <mutex>
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
__device__ __forceinline__ void FastllmMoeFp8Accumulate4Legacy(
        const T *A, int offset, uint32_t bytes, float scale, float &sum) {
    FastllmMoeFp8Accumulate4(A, offset, bytes, scale, sum);
}

template <>
__device__ __forceinline__ void FastllmMoeFp8Accumulate4Legacy<half>(
        const half *A, int offset, uint32_t bytes, float scale, float &sum) {
    __half2 values01 = make_half2(
        __short_as_half((((bytes >> 0) & 0x80) << 8) | (((bytes >> 0) & 0x7F) << 7)),
        __short_as_half((((bytes >> 8) & 0x80) << 8) | (((bytes >> 8) & 0x7F) << 7)));
    __half2 values23 = make_half2(
        __short_as_half((((bytes >> 16) & 0x80) << 8) | (((bytes >> 16) & 0x7F) << 7)),
        __short_as_half((((bytes >> 24) & 0x80) << 8) | (((bytes >> 24) & 0x7F) << 7)));
    const __half2 *input = reinterpret_cast<const __half2*>(A + offset);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
    sum += (__half2float(input[0].x) * __half2float(values01.x) +
            __half2float(input[0].y) * __half2float(values01.y) +
            __half2float(input[1].x) * __half2float(values23.x) +
            __half2float(input[1].y) * __half2float(values23.y)) * scale;
#else
    __half2 product01 = __hmul2(input[0], values01);
    __half2 product23 = __hmul2(input[1], values23);
    __half2 pair = __hadd2(product01, product23);
    __half value = __hadd(pair.x, pair.y);
    sum += __half2float(value) * scale;
#endif
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

__device__ __forceinline__ float FastllmMoeNVFP4MagicScale() {
    return __uint_as_float(253u << 23);
}

__device__ __forceinline__ float FastllmMoeNVFP4E8M0ToFloat(uint8_t v) {
    uint32_t bits = v == 0 ? 0x00400000u : ((uint32_t)v << 23);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float FastllmMoeNVFP4E8M0ToMagicScale(uint8_t v) {
    return __uint_as_float(((uint32_t)v + 126u) << 23);
}

__device__ __forceinline__ float FastllmMoeNVFP4PseudoBFloat16ToFloat(uint8_t v) {
    uint32_t bits = ((uint32_t)(v & 0x8) << 28) | ((uint32_t)(v & 0x7) << 22);
    return __uint_as_float(bits);
}

template <bool SCALE_E8M0>
__device__ __forceinline__ float FastllmMoeNVFP4ApplyScale(float value, const uint8_t *blockData) {
    if constexpr (SCALE_E8M0) {
        uint8_t scaleByte = blockData[8];
        if (scaleByte <= 128) {
            return value * FastllmMoeNVFP4E8M0ToMagicScale(scaleByte);
        }
        return (value * FastllmMoeNVFP4MagicScale()) * FastllmMoeNVFP4E8M0ToFloat(scaleByte);
    } else {
        return (value * FastllmMoeNVFP4MagicScale()) * (*(float*)(blockData + 8));
    }
}

template <bool SCALE_E8M0, typename T>
__device__ __forceinline__ void FastllmMoeNVFP4Block16Accumulate4(const T *A, int offset,
                                                                  const uint8_t *rowData, int m,
                                                                  float &sum) {
    const int blockBytes = SCALE_E8M0 ? 9 : (8 + (int)sizeof(float));
    int block = offset >> 4;
    int blockStart = block << 4;
    int blockEnd = min(blockStart + 16, m);
    const uint8_t *blockData = rowData + block * blockBytes;
    int local = offset - blockStart;
    int remaining = min(4, blockEnd - offset);
    float blockSum = 0.0f;
    if (remaining == 4) {
        uint8_t packed01 = blockData[local >> 1];
        uint8_t packed23 = blockData[(local + 2) >> 1];
        blockSum += FastllmMoeFp8Traits<T>::toFloat(A[offset + 0]) * FastllmMoeNVFP4PseudoBFloat16ToFloat(packed01 & 0xF);
        blockSum += FastllmMoeFp8Traits<T>::toFloat(A[offset + 1]) * FastllmMoeNVFP4PseudoBFloat16ToFloat(packed01 >> 4);
        blockSum += FastllmMoeFp8Traits<T>::toFloat(A[offset + 2]) * FastllmMoeNVFP4PseudoBFloat16ToFloat(packed23 & 0xF);
        blockSum += FastllmMoeFp8Traits<T>::toFloat(A[offset + 3]) * FastllmMoeNVFP4PseudoBFloat16ToFloat(packed23 >> 4);
    } else {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j < remaining) {
                int col = offset + j;
                int localCol = col - blockStart;
                uint8_t packed = blockData[localCol >> 1];
                uint8_t fp4 = (localCol & 1) ? (packed >> 4) : (packed & 0xF);
                blockSum += FastllmMoeFp8Traits<T>::toFloat(A[col]) * FastllmMoeNVFP4PseudoBFloat16ToFloat(fp4);
            }
        }
    }
    sum += FastllmMoeNVFP4ApplyScale<SCALE_E8M0>(blockSum, blockData);
}

template <typename T>
__device__ __forceinline__ void FastllmMoeNVFP4CompactAccumulate4(const T *A, int offset,
                                                                  const uint8_t *weightData, int row,
                                                                  int rows, int m, int blockK, int blockM,
                                                                  int scaleCols, float &sum) {
    int remaining = min(4, m - offset);
    if (remaining <= 0) {
        return;
    }
    int packedPerRow = (m + 1) >> 1;
    const uint8_t *rowData = weightData + (size_t)row * packedPerRow;
    const uint8_t *scaleData = weightData + (size_t)rows * packedPerRow;
    int scaleRow = row / blockK;
    if (remaining == 4 && offset / blockM == (offset + 3) / blockM) {
        uint8_t packed01 = rowData[offset >> 1];
        uint8_t packed23 = rowData[(offset + 2) >> 1];
        float scaleMagic = FastllmMoeNVFP4E8M0ToMagicScale(scaleData[(size_t)scaleRow * scaleCols + offset / blockM]);
        float blockSum =
            FastllmMoeFp8Traits<T>::toFloat(A[offset + 0]) * FastllmMoeNVFP4PseudoBFloat16ToFloat(packed01 & 0xF) +
            FastllmMoeFp8Traits<T>::toFloat(A[offset + 1]) * FastllmMoeNVFP4PseudoBFloat16ToFloat(packed01 >> 4) +
            FastllmMoeFp8Traits<T>::toFloat(A[offset + 2]) * FastllmMoeNVFP4PseudoBFloat16ToFloat(packed23 & 0xF) +
            FastllmMoeFp8Traits<T>::toFloat(A[offset + 3]) * FastllmMoeNVFP4PseudoBFloat16ToFloat(packed23 >> 4);
        sum += blockSum * scaleMagic;
        return;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j < remaining) {
            int col = offset + j;
            uint8_t packed = rowData[col >> 1];
            uint8_t fp4 = (col & 1) ? (packed >> 4) : (packed & 0xF);
            float scaleMagic = FastllmMoeNVFP4E8M0ToMagicScale(scaleData[(size_t)scaleRow * scaleCols + col / blockM]);
            sum += FastllmMoeFp8Traits<T>::toFloat(A[col]) *
                   FastllmMoeNVFP4PseudoBFloat16ToFloat(fp4) * scaleMagic;
        }
    }
}

static inline size_t FastllmMoeNVFP4Block16BytesPerRow(int m, bool scaleE8M0) {
    return (size_t)((m - 1) / 16 + 1) * (scaleE8M0 ? 9 : (8 + (int)sizeof(float)));
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

// Batch-1 hidden=2048/inter=256/topk=8 specialization for the indexed-weight
// path: one warp computes one output and one CTA computes eight outputs. Each
// lane reproduces the work of two lanes from the legacy 64-thread kernel, then
// performs the old stride-32 merge before the warp shuffle reduction. This
// keeps the floating-point reduction tree intact while removing block-wide
// barriers and cutting the number of launched CTAs.
template <typename T>
__global__ __launch_bounds__(256) void FastllmGemvTypedFP8E4M3TopKSwigluIndexedWarpKernel(
        const T *A, const int32_t *indices, uint8_t **weights, float **scalesPtrs,
        T *C) {
    constexpr int HIDDEN = 2048;
    constexpr int INTER = 256;
    constexpr int TOPK = 8;
    constexpr int BLOCK = 128;
    constexpr int LEGACY_THREADS = 64;
    constexpr int OUTPUTS_PER_CTA = 8;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int p = blockIdx.x * OUTPUTS_PER_CTA + warp;
    int topkSlot = blockIdx.y;
    if (p >= INTER || topkSlot >= TOPK) {
        return;
    }

    int expertIdx = indices[topkSlot];
    const uint8_t *weight = weights[expertIdx];
    const float *scales = scalesPtrs[expertIdx];
    const uint8_t *baseGate = weight + (size_t)p * HIDDEN;
    const uint8_t *baseUp = weight + (size_t)(p + INTER) * HIDDEN;
    const float *gateScales = scales + (p / BLOCK) * (HIDDEN / BLOCK);
    const float *upScales = scales + ((p + INTER) / BLOCK) * (HIDDEN / BLOCK);

    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
#pragma unroll
    for (int part = 0; part < HIDDEN / (LEGACY_THREADS * 4); part++) {
        int offset0 = lane * 4 + part * LEGACY_THREADS * 4;
        int offset1 = (lane + 32) * 4 + part * LEGACY_THREADS * 4;
        float gateScale0 = gateScales[offset0 / BLOCK];
        float gateScale1 = gateScales[offset1 / BLOCK];
        float upScale0 = upScales[offset0 / BLOCK];
        float upScale1 = upScales[offset1 / BLOCK];
        FastllmMoeFp8Accumulate4Legacy(A, offset0, *(const uint32_t*)(baseGate + offset0), gateScale0, gate0);
        FastllmMoeFp8Accumulate4Legacy(A, offset1, *(const uint32_t*)(baseGate + offset1), gateScale1, gate1);
        FastllmMoeFp8Accumulate4Legacy(A, offset0, *(const uint32_t*)(baseUp + offset0), upScale0, up0);
        FastllmMoeFp8Accumulate4Legacy(A, offset1, *(const uint32_t*)(baseUp + offset1), upScale1, up1);
    }

    float gate = gate0 + gate1;
    float up = up0 + up1;
    constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(FULL_WARP_MASK, gate, offset);
        up += __shfl_down_sync(FULL_WARP_MASK, up, offset);
    }
    if (lane == 0) {
        const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
        gate = FastllmMoeFp8Round<T>(gate * magicScaleConstant);
        up = FastllmMoeFp8Round<T>(up * magicScaleConstant);
        C[(size_t)topkSlot * INTER + p] =
            FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T>
__global__ __launch_bounds__(256) void FastllmGemvTypedFP8E4M3TopKDownReduceIndexedWarpKernel(
        const T *A, const int32_t *indices, uint8_t **weights, float **scalesPtrs,
        T *C, const float *scores) {
    constexpr int INTER = 256;
    constexpr int HIDDEN = 2048;
    constexpr int TOPK = 8;
    constexpr int BLOCK = 128;
    constexpr int OUTPUTS_PER_CTA = 8;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int st = blockIdx.x * OUTPUTS_PER_CTA + warp;
    if (st >= HIDDEN) {
        return;
    }

    float out = 0.0f;
    constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;
#pragma unroll
    for (int topkSlot = 0; topkSlot < TOPK; topkSlot++) {
        int expertIdx = indices[topkSlot];
        const uint8_t *weight = weights[expertIdx];
        const float *scales = scalesPtrs[expertIdx];
        const uint8_t *baseWeight = weight + (size_t)st * INTER;
        const float *rowScales = scales + (st / BLOCK) * (INTER / BLOCK);
        const T *expertInput = A + (size_t)topkSlot * INTER;

        int offset0 = lane * 4;
        int offset1 = (lane + 32) * 4;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        FastllmMoeFp8Accumulate4Legacy(expertInput, offset0,
                                       *(const uint32_t*)(baseWeight + offset0),
                                       rowScales[offset0 / BLOCK], sum0);
        FastllmMoeFp8Accumulate4Legacy(expertInput, offset1,
                                       *(const uint32_t*)(baseWeight + offset1),
                                       rowScales[offset1 / BLOCK], sum1);
        float sum = sum0 + sum1;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
        }
        if (lane == 0) {
            out += FastllmMoeFp8Round<T>(sum * FastllmMoeFp8Traits<T>::magicScale()) *
                   scores[topkSlot];
        }
    }
    if (lane == 0) {
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

template <bool SCALE_E8M0, typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4Block16TopKSwigluKernel(T *A, uint8_t **weights,
                                                             T *C, int m, int k, int perRow) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int expert = blockIdx.y;
    uint8_t *B = weights[expert];
    const uint8_t *baseGate = B + (size_t)p * perRow;
    const uint8_t *baseUp = B + (size_t)(p + k) * perRow;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(A, i, baseGate, m, sdataGate[tid]);
        FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(A, i, baseUp, m, sdataUp[tid]);
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
        float gate = FastllmMoeFp8Round<T>(sdataGate[0]);
        float up = FastllmMoeFp8Round<T>(sdataUp[0]);
        C[(size_t)expert * k + p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <bool SCALE_E8M0, typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4Block16TopKDownReduceKernel(T *A, uint8_t **weights,
                                                                 T *C, float *scores, int topk,
                                                                 int m, int k, int perRow) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int expert = 0; expert < topk; expert++) {
        uint8_t *B = weights[expert];
        const uint8_t *baseB = B + (size_t)st * perRow;
        T *expertInput = A + (size_t)expert * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(expertInput, i, baseB, m, sdata[tid]);
        }
        __syncthreads();

        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out += FastllmMoeFp8Round<T>(sdata[0]) * scores[expert];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <bool SCALE_E8M0, typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4Block16TopKSwigluIndexedKernel(T *A, const int32_t *indices,
                                                                    uint8_t **weights, T *C,
                                                                    int topk, int m, int k, int perRow) {
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
    const uint8_t *baseGate = B + (size_t)p * perRow;
    const uint8_t *baseUp = B + (size_t)(p + k) * perRow;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(A, i, baseGate, m, sdataGate[tid]);
        FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(A, i, baseUp, m, sdataUp[tid]);
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
        float gate = FastllmMoeFp8Round<T>(sdataGate[0]);
        float up = FastllmMoeFp8Round<T>(sdataUp[0]);
        C[(size_t)topkSlot * k + p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <bool SCALE_E8M0, typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4Block16TopKDownReduceIndexedKernel(T *A, const int32_t *indices,
                                                                        uint8_t **weights, T *C,
                                                                        const float *scores, int topk,
                                                                        int m, int k, int perRow) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int expertIdx = indices[topkSlot];
        uint8_t *B = weights[expertIdx];
        const uint8_t *baseB = B + (size_t)st * perRow;
        T *expertInput = A + (size_t)topkSlot * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(expertInput, i, baseB, m, sdata[tid]);
        }
        __syncthreads();

        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out += FastllmMoeFp8Round<T>(sdata[0]) * scores[topkSlot];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

// Decode uses only a few routed experts.  The original kernel evaluated them
// serially inside one 64-thread block, paying a full block reduction for every
// expert.  Assign one 64-thread group to each expert so all expert dot products
// run concurrently while preserving the original per-expert reduction order.
template <bool SCALE_E8M0, typename T, int GROUP_THREADS, int MAX_TOPK>
__global__ void FastllmGemvTypedNVFP4Block16TopKDownReduceIndexedParallelKernel(
        T *A, const int32_t *indices, uint8_t **weights, T *C,
        const float *scores, int topk, int m, int k, int perRow) {
    __shared__ float partials[MAX_TOPK * GROUP_THREADS];
    __shared__ float expertOutputs[MAX_TOPK];
    int group = threadIdx.x / GROUP_THREADS;
    int local = threadIdx.x % GROUP_THREADS;
    int st = blockIdx.x;

    int expertIdx = indices[group];
    uint8_t *B = weights[expertIdx];
    const uint8_t *baseB = B + (size_t)st * perRow;
    T *expertInput = A + (size_t)group * m;
    float value = 0.0f;
    for (int i = local * 4; i < m; i += GROUP_THREADS * 4) {
        FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(expertInput, i, baseB, m, value);
    }
    int sharedIdx = group * GROUP_THREADS + local;
    partials[sharedIdx] = value;
    __syncthreads();

    for (int stride = GROUP_THREADS / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            partials[sharedIdx] += partials[sharedIdx + stride];
        }
        __syncthreads();
    }
    if (local == 0) {
        expertOutputs[group] = FastllmMoeFp8Round<T>(partials[group * GROUP_THREADS]) * scores[group];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float output = 0.0f;
        for (int slot = 0; slot < topk; slot++) {
            output += expertOutputs[slot];
        }
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(output);
    }
}

template <bool SCALE_E8M0, typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4Block16SmallBatchTopKSwigluIndexedKernel(T *A, const int32_t *indices,
                                                                              uint8_t **weights, T *C,
                                                                              int batch, int topk, int m, int k,
                                                                              int perRow) {
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
    T *tokenInput = A + (size_t)token * m;
    const uint8_t *baseGate = B + (size_t)p * perRow;
    const uint8_t *baseUp = B + (size_t)(p + k) * perRow;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(tokenInput, i, baseGate, m, sdataGate[tid]);
        FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(tokenInput, i, baseUp, m, sdataUp[tid]);
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
        float gate = FastllmMoeFp8Round<T>(sdataGate[0]);
        float up = FastllmMoeFp8Round<T>(sdataUp[0]);
        C[(size_t)task * k + p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <bool SCALE_E8M0, typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4Block16SmallBatchTopKDownReduceIndexedKernel(T *A, const int32_t *indices,
                                                                                  uint8_t **weights, T *C,
                                                                                  const float *scores, int batch, int topk,
                                                                                  int m, int k, int perRow) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int token = blockIdx.y;
    if (token >= batch) {
        return;
    }

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int task = token * topk + topkSlot;
        int expertIdx = indices[task];
        uint8_t *B = weights[expertIdx];
        const uint8_t *baseB = B + (size_t)st * perRow;
        T *expertInput = A + (size_t)task * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(expertInput, i, baseB, m, sdata[tid]);
        }
        __syncthreads();

        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out += FastllmMoeFp8Round<T>(sdata[0]) * scores[task];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[(size_t)token * k + st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4CompactTopKSwigluIndexedKernel(T *A, const int32_t *indices,
                                                                    uint8_t **weights, T *C,
                                                                    int topk, int m, int k,
                                                                    int blockK, int blockM, int scaleCols) {
    static_assert(THREAD_PER_BLOCK == 64, "NVFP4 compact GEMV reduction expects two warps");
    __shared__ float warpGate[2];
    __shared__ float warpUp[2];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int topkSlot = blockIdx.y;
    if (topkSlot >= topk) {
        return;
    }
    int expertIdx = indices[topkSlot];
    uint8_t *B = weights[expertIdx];
    int rows = k * 2;

    float gateSum = 0.0f;
    float upSum = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        FastllmMoeNVFP4CompactAccumulate4(A, i, B, p, rows, m, blockK, blockM, scaleCols, gateSum);
        FastllmMoeNVFP4CompactAccumulate4(A, i, B, p + k, rows, m, blockK, blockM, scaleCols, upSum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gateSum += __shfl_down_sync(0xffffffffu, gateSum, offset);
        upSum += __shfl_down_sync(0xffffffffu, upSum, offset);
    }
    if ((tid & 31) == 0) {
        warpGate[tid >> 5] = gateSum;
        warpUp[tid >> 5] = upSum;
    }
    __syncthreads();

    if (tid == 0) {
        float gate = FastllmMoeFp8Round<T>(warpGate[0] + warpGate[1]);
        float up = FastllmMoeFp8Round<T>(warpUp[0] + warpUp[1]);
        C[(size_t)topkSlot * k + p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4CompactTopKDownReduceIndexedKernel(T *A, const int32_t *indices,
                                                                        uint8_t **weights, T *C,
                                                                        const float *scores, int topk,
                                                                        int m, int k,
                                                                        int blockK, int blockM, int scaleCols) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int expertIdx = indices[topkSlot];
        uint8_t *B = weights[expertIdx];
        T *expertInput = A + (size_t)topkSlot * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            FastllmMoeNVFP4CompactAccumulate4(expertInput, i, B, st, k, m, blockK, blockM, scaleCols, sdata[tid]);
        }
        __syncthreads();

        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out += FastllmMoeFp8Round<T>(sdata[0]) * scores[topkSlot];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <typename T, int GROUP_THREADS, int MAX_TOPK>
__global__ void FastllmGemvTypedNVFP4CompactTopKDownReduceIndexedParallelKernel(
        T *A, const int32_t *indices, uint8_t **weights, T *C,
        const float *scores, int topk, int m, int k,
        int blockK, int blockM, int scaleCols) {
    static_assert(GROUP_THREADS == 64, "NVFP4 compact GEMV reduction expects two warps per route");
    __shared__ float warpPartials[MAX_TOPK * 2];
    __shared__ float expertOutputs[MAX_TOPK];
    int group = threadIdx.x / GROUP_THREADS;
    int local = threadIdx.x % GROUP_THREADS;
    int st = blockIdx.x;

    int expertIdx = indices[group];
    uint8_t *B = weights[expertIdx];
    T *expertInput = A + (size_t)group * m;
    float value = 0.0f;
    for (int i = local * 4; i < m; i += GROUP_THREADS * 4) {
        FastllmMoeNVFP4CompactAccumulate4(
            expertInput, i, B, st, k, m, blockK, blockM, scaleCols, value);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if ((local & 31) == 0) {
        warpPartials[group * 2 + (local >> 5)] = value;
    }
    __syncthreads();
    if (local == 0) {
        expertOutputs[group] = FastllmMoeFp8Round<T>(warpPartials[group * 2] +
                                                      warpPartials[group * 2 + 1]) * scores[group];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float output = 0.0f;
        for (int slot = 0; slot < topk; slot++) {
            output += expertOutputs[slot];
        }
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(output);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4CompactExpertParallelSwigluKernel(
        T *A, const int32_t *globalIndices, uint8_t **localWeights, T *C,
        int topk, int ownerRank, int ownerCount, int m, int k,
        int blockK, int blockM, int scaleCols) {
    static_assert(THREAD_PER_BLOCK == 64, "NVFP4 compact GEMV reduction expects two warps");
    __shared__ float warpGate[2];
    __shared__ float warpUp[2];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int topkSlot = blockIdx.y;
    if (topkSlot >= topk) {
        return;
    }
    int globalExpert = globalIndices[topkSlot];
    if (ownerRank < 0 || ownerCount <= 0 || globalExpert < 0 ||
        globalExpert % ownerCount != ownerRank) {
        return;
    }
    int localExpert = globalExpert / ownerCount;
    uint8_t *B = localWeights[localExpert];
    int rows = k * 2;

    float gateSum = 0.0f;
    float upSum = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        FastllmMoeNVFP4CompactAccumulate4(A, i, B, p, rows, m,
                                           blockK, blockM, scaleCols, gateSum);
        FastllmMoeNVFP4CompactAccumulate4(A, i, B, p + k, rows, m,
                                           blockK, blockM, scaleCols, upSum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gateSum += __shfl_down_sync(0xffffffffu, gateSum, offset);
        upSum += __shfl_down_sync(0xffffffffu, upSum, offset);
    }
    if ((tid & 31) == 0) {
        warpGate[tid >> 5] = gateSum;
        warpUp[tid >> 5] = upSum;
    }
    __syncthreads();
    if (tid == 0) {
        float gate = FastllmMoeFp8Round<T>(warpGate[0] + warpGate[1]);
        float up = FastllmMoeFp8Round<T>(warpUp[0] + warpUp[1]);
        C[(size_t)topkSlot * k + p] =
            FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4CompactExpertParallelDownReduceSerialKernel(
        T *A, const int32_t *globalIndices, uint8_t **localWeights, T *C,
        const float *scores, int topk, int ownerRank, int ownerCount,
        int m, int k, int blockK, int blockM, int scaleCols) {
    static_assert(THREAD_PER_BLOCK == 64, "NVFP4 compact GEMV reduction expects two warps");
    __shared__ float warpPartials[2];
    int tid = threadIdx.x;
    int st = blockIdx.x;
    float result = 0.0f;

    for (int slot = 0; slot < topk; ++slot) {
        int globalExpert = globalIndices[slot];
        bool active = ownerRank >= 0 && ownerCount > 0 && globalExpert >= 0 &&
                      globalExpert % ownerCount == ownerRank;
        float value = 0.0f;
        if (active) {
            int localExpert = globalExpert / ownerCount;
            uint8_t *B = localWeights[localExpert];
            T *expertInput = A + (size_t)slot * m;
            for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
                FastllmMoeNVFP4CompactAccumulate4(
                    expertInput, i, B, st, k, m, blockK, blockM, scaleCols, value);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if ((tid & 31) == 0) {
            warpPartials[tid >> 5] = value;
        }
        __syncthreads();
        if (tid == 0 && active) {
            result += FastllmMoeFp8Round<T>(warpPartials[0] + warpPartials[1]) *
                      scores[slot];
        }
        __syncthreads();
    }
    if (tid == 0) {
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(result);
    }
}

// DeepSeek V4 Flash stores routed experts in compact NVFP4, while its shared
// expert stays in block-scaled FP8 E4M3.  Decode previously launched a separate
// pair of GEMV kernels (and, under tensor parallelism, a separate all-reduce)
// for the shared expert.  Treat the shared expert as one extra fixed route so
// routed and shared experts share the same two kernel launches.
template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4CompactSharedFP8TopKSwigluIndexedKernel(
        T *A, const int32_t *indices, uint8_t **routedWeights,
        const uint8_t *sharedWeight, const float *sharedScales, T *C,
        int topk, int m, int k,
        int routedBlockK, int routedBlockM, int routedScaleCols,
        int sharedBlockK, int sharedBlockM) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int slot = blockIdx.y;

    float gateValue = 0.0f;
    float upValue = 0.0f;
    if (slot < topk) {
        int expertIdx = indices[slot];
        uint8_t *weight = routedWeights[expertIdx];
        int rows = k * 2;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            FastllmMoeNVFP4CompactAccumulate4(
                A, i, weight, p, rows, m,
                routedBlockK, routedBlockM, routedScaleCols, gateValue);
            FastllmMoeNVFP4CompactAccumulate4(
                A, i, weight, p + k, rows, m,
                routedBlockK, routedBlockM, routedScaleCols, upValue);
        }
    } else {
        int scaleCols = (m - 1) / sharedBlockM + 1;
        const uint8_t *baseGate = sharedWeight + (size_t)p * m;
        const uint8_t *baseUp = sharedWeight + (size_t)(p + k) * m;
        const float *gateScales = sharedScales + (p / sharedBlockK) * scaleCols;
        const float *upScales = sharedScales + ((p + k) / sharedBlockK) * scaleCols;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            int remaining = m - i;
            float gateScale = gateScales[i / sharedBlockM];
            float upScale = upScales[i / sharedBlockM];
            if (remaining >= 4) {
                FastllmMoeFp8Accumulate4(A, i, *(const uint32_t*)(baseGate + i), gateScale, gateValue);
                FastllmMoeFp8Accumulate4(A, i, *(const uint32_t*)(baseUp + i), upScale, upValue);
            } else {
                FastllmMoeFp8AccumulateRemainder(A, baseGate, i, remaining, gateScale, gateValue);
                FastllmMoeFp8AccumulateRemainder(A, baseUp, i, remaining, upScale, upValue);
            }
        }
        float magicScale = FastllmMoeFp8Traits<T>::magicScale();
        gateValue *= magicScale;
        upValue *= magicScale;
    }

    sdataGate[tid] = gateValue;
    sdataUp[tid] = upValue;
    __syncthreads();
    for (int stride = THREAD_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdataGate[tid] += sdataGate[tid + stride];
            sdataUp[tid] += sdataUp[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float gate = FastllmMoeFp8Round<T>(sdataGate[0]);
        float up = FastllmMoeFp8Round<T>(sdataUp[0]);
        C[(size_t)slot * k + p] =
            FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int GROUP_THREADS, int MAX_ROUTES>
__global__ void FastllmGemvTypedNVFP4CompactSharedFP8TopKDownReduceIndexedParallelKernel(
        T *A, const int32_t *indices, uint8_t **routedWeights,
        const uint8_t *sharedWeight, const float *sharedScales, T *C,
        const float *scores, float sharedScale, int topk, int m, int k,
        int routedBlockK, int routedBlockM, int routedScaleCols,
        int sharedBlockK, int sharedBlockM) {
    __shared__ float partials[MAX_ROUTES * GROUP_THREADS];
    __shared__ float routeOutputs[MAX_ROUTES];
    int group = threadIdx.x / GROUP_THREADS;
    int local = threadIdx.x % GROUP_THREADS;
    int st = blockIdx.x;
    bool shared = group == topk;

    T *routeInput = A + (size_t)group * m;
    float value = 0.0f;
    if (!shared) {
        int expertIdx = indices[group];
        uint8_t *weight = routedWeights[expertIdx];
        for (int i = local * 4; i < m; i += GROUP_THREADS * 4) {
            FastllmMoeNVFP4CompactAccumulate4(
                routeInput, i, weight, st, k, m,
                routedBlockK, routedBlockM, routedScaleCols, value);
        }
    } else {
        int scaleCols = (m - 1) / sharedBlockM + 1;
        const uint8_t *baseB = sharedWeight + (size_t)st * m;
        const float *rowScales = sharedScales + (st / sharedBlockK) * scaleCols;
        for (int i = local * 4; i < m; i += GROUP_THREADS * 4) {
            int remaining = m - i;
            float scale = rowScales[i / sharedBlockM];
            if (remaining >= 4) {
                FastllmMoeFp8Accumulate4(A + (size_t)group * m, i,
                                          *(const uint32_t*)(baseB + i), scale, value);
            } else {
                FastllmMoeFp8AccumulateRemainder(A + (size_t)group * m, baseB, i,
                                                  remaining, scale, value);
            }
        }
        value *= FastllmMoeFp8Traits<T>::magicScale();
    }

    int partialIdx = group * GROUP_THREADS + local;
    partials[partialIdx] = value;
    __syncthreads();
    for (int stride = GROUP_THREADS / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            partials[partialIdx] += partials[partialIdx + stride];
        }
        __syncthreads();
    }
    if (local == 0) {
        float alpha = shared ? sharedScale : scores[group];
        routeOutputs[group] = FastllmMoeFp8Round<T>(partials[group * GROUP_THREADS]) * alpha;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float output = 0.0f;
        for (int slot = 0; slot <= topk; slot++) {
            output += routeOutputs[slot];
        }
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(output);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4CompactSmallBatchTopKSwigluIndexedKernel(T *A, const int32_t *indices,
                                                                              uint8_t **weights, T *C,
                                                                              int batch, int topk, int m, int k,
                                                                              int blockK, int blockM, int scaleCols) {
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
    T *tokenInput = A + (size_t)token * m;
    int rows = k * 2;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        FastllmMoeNVFP4CompactAccumulate4(tokenInput, i, B, p, rows, m, blockK, blockM, scaleCols, sdataGate[tid]);
        FastllmMoeNVFP4CompactAccumulate4(tokenInput, i, B, p + k, rows, m, blockK, blockM, scaleCols, sdataUp[tid]);
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
        float gate = FastllmMoeFp8Round<T>(sdataGate[0]);
        float up = FastllmMoeFp8Round<T>(sdataUp[0]);
        C[(size_t)task * k + p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedNVFP4CompactSmallBatchTopKDownReduceIndexedKernel(T *A, const int32_t *indices,
                                                                                  uint8_t **weights, T *C,
                                                                                  const float *scores, int batch, int topk,
                                                                                  int m, int k,
                                                                                  int blockK, int blockM, int scaleCols) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int token = blockIdx.y;
    if (token >= batch) {
        return;
    }

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int task = token * topk + topkSlot;
        int expertIdx = indices[task];
        uint8_t *B = weights[expertIdx];
        T *expertInput = A + (size_t)task * m;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
            FastllmMoeNVFP4CompactAccumulate4(expertInput, i, B, st, k, m, blockK, blockM, scaleCols, sdata[tid]);
        }
        __syncthreads();

        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out += FastllmMoeFp8Round<T>(sdata[0]) * scores[task];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[(size_t)token * k + st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <bool SCALE_E8M0, typename T, int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvTypedNVFP4Block16GroupedTopKSwigluIndexedKernel(T *A, const int *routeRows,
                                                                           const int *expertStarts, const int *expertCounts,
                                                                           uint8_t **weights, T *C, int maxChunks,
                                                                           int m, int k, int perRow) {
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
    const uint8_t *baseGate = B + (size_t)p * perRow;
    const uint8_t *baseUp = B + (size_t)(p + k) * perRow;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            if (active[x]) {
                T *rowInput = A + (size_t)rows[x] * m;
                FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(rowInput, i, baseGate, m, sdataGate[x][tid]);
                FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(rowInput, i, baseUp, m, sdataUp[x][tid]);
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
                float gate = FastllmMoeFp8Round<T>(sdataGate[x][0]);
                float up = FastllmMoeFp8Round<T>(sdataUp[x][0]);
                C[(size_t)(start + localBase + x) * k + p] =
                    FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
            }
        }
    }
}

template <bool SCALE_E8M0, typename T, int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvTypedNVFP4Block16GroupedTopKDownScatterIndexedKernel(T *A, const int *routeRows,
                                                                                const float *routeScales,
                                                                                const int *expertStarts,
                                                                                const int *expertCounts,
                                                                                uint8_t **weights, T *C,
                                                                                int maxChunks, int m, int k,
                                                                                int perRow) {
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
    const uint8_t *baseB = B + (size_t)st * perRow;

    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            if (active[x]) {
                T *rowInput = A + (size_t)(start + localBase + x) * m;
                FastllmMoeNVFP4Block16Accumulate4<SCALE_E8M0>(rowInput, i, baseB, m, sdata[x][tid]);
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
                float value = FastllmMoeFp8Round<T>(sdata[x][0]) * alphas[x];
                C[(size_t)(start + localBase + x) * k + st] = FastllmMoeFp8Traits<T>::fromFloat(value);
            }
        }
    }
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

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3Block128TopKSwigluIndexedKernel(
        T *A, const int32_t *indices, uint8_t **weights, T *C,
        int topk, int m, int k, int perRow) {
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
        C[(size_t)topkSlot * k + p] = FastllmMoeFp8Traits<T>::fromFloat((gate / (1.0f + expf(-gate))) * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3Block128TopKDownReduceIndexedKernel(
        T *A, const int32_t *indices, uint8_t **weights, T *C, const float *scores,
        int topk, int m, int k, int perRow) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    const int blockSize = 128;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    int numBlocks = (m - 1) / blockSize + 1;

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int expertIdx = indices[topkSlot];
        uint8_t *B = weights[expertIdx];
        const uint8_t *baseB = B + (size_t)st * perRow;
        T *expertInput = A + (size_t)topkSlot * m;

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
                    FastllmMoeFp8Accumulate4(expertInput, i, *(uint32_t*)(blkData + localIdx), blkScale, sdata[tid]);
                } else {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        if (j < remaining) {
                            sdata[tid] += FastllmMoeFp8Traits<T>::toFloat(expertInput[i + j]) *
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
            out += FastllmMoeFp8Round<T>(sdata[0] * magicScaleConstant) * scores[topkSlot];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3FusedTopKSwigluKernel(
        const T *A, const int32_t *indices, const uint8_t *gateWeight, const uint8_t *upWeight,
        const float *gateScales, const float *upScales, T *C,
        int batch, int topk, int hidden, int inter, int experts, int blockM, int blockK, float swigluLimit) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int topkSlot = blockIdx.y;
    int token = blockIdx.z;
    if (token >= batch || topkSlot >= topk) {
        return;
    }

    int task = token * topk + topkSlot;
    int expertIdx = indices[task];
    if (expertIdx < 0 || expertIdx >= experts) {
        if (tid == 0) {
            C[(size_t)task * inter + p] = FastllmMoeFp8Traits<T>::fromFloat(0.0f);
        }
        return;
    }

    int ms = (hidden - 1) / blockM + 1;
    int ks = (inter - 1) / blockK + 1;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const T *tokenInput = A + (size_t)token * hidden;
    const uint8_t *baseGate = gateWeight + ((size_t)expertIdx * inter + p) * hidden;
    const uint8_t *baseUp = upWeight + ((size_t)expertIdx * inter + p) * hidden;
    const float *gateRowScales = gateScales + ((size_t)expertIdx * ks + p / blockK) * ms;
    const float *upRowScales = upScales + ((size_t)expertIdx * ks + p / blockK) * ms;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < hidden; i += THREAD_PER_BLOCK * 4) {
        int remaining = hidden - i;
        float gateScale = gateRowScales[i / blockM];
        float upScale = upRowScales[i / blockM];
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
        float gateAct = gate / (1.0f + expf(-gate));
        if (swigluLimit > 0.0f) {
            gateAct = gateAct > swigluLimit ? swigluLimit : gateAct;
            up = up < -swigluLimit ? -swigluLimit : (up > swigluLimit ? swigluLimit : up);
        }
        C[(size_t)task * inter + p] = FastllmMoeFp8Traits<T>::fromFloat(gateAct * up);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3FusedTopKSwigluKernel(
        half *A, const int32_t *indices, uint8_t *gateWeight, uint8_t *upWeight,
        float *gateScales, float *upScales, half *C,
        int batch, int topk, int hidden, int inter, int experts, int blockM, int blockK, float swigluLimit) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int topkSlot = blockIdx.y;
    int token = blockIdx.z;
    if (token >= batch || topkSlot >= topk) {
        return;
    }

    int task = token * topk + topkSlot;
    int expertIdx = indices[task];
    if (expertIdx < 0 || expertIdx >= experts) {
        if (tid == 0) {
            C[(size_t)task * inter + p] = (half)0.0f;
        }
        return;
    }

    int ms = (hidden - 1) / blockM + 1;
    int ks = (inter - 1) / blockK + 1;
    const float magicScaleConstant = exp2f(8.0f);
    half *tokenInput = A + (size_t)token * hidden;
    const uint8_t *baseGate = gateWeight + ((size_t)expertIdx * inter + p) * hidden;
    const uint8_t *baseUp = upWeight + ((size_t)expertIdx * inter + p) * hidden;
    float *gateRowScales = gateScales + ((size_t)expertIdx * ks + p / blockK) * ms;
    float *upRowScales = upScales + ((size_t)expertIdx * ks + p / blockK) * ms;
    union_half4 regA;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int i = tid * 4; i < hidden; i += THREAD_PER_BLOCK * 4) {
        int remaining = hidden - i;
        float gateScale = gateRowScales[i / blockM];
        float upScale = upRowScales[i / blockM];
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
        float gateAct = gate / (1.0f + expf(-gate));
        if (swigluLimit > 0.0f) {
            gateAct = gateAct > swigluLimit ? swigluLimit : gateAct;
            up = up < -swigluLimit ? -swigluLimit : (up > swigluLimit ? swigluLimit : up);
        }
        C[(size_t)task * inter + p] = (half)(gateAct * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3FusedTopKDownReduceKernel(
        const T *A, const int32_t *indices, const uint8_t *downWeight, const float *downScales,
        T *C, const float *scores, int batch, int topk, int inter, int hidden, int experts,
        int blockM, int blockK) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int token = blockIdx.y;
    if (token >= batch) {
        return;
    }

    int ms = (inter - 1) / blockM + 1;
    int ks = (hidden - 1) / blockK + 1;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int task = token * topk + topkSlot;
        int expertIdx = indices[task];
        if (expertIdx < 0 || expertIdx >= experts) {
            continue;
        }
        const uint8_t *baseB = downWeight + ((size_t)expertIdx * hidden + st) * inter;
        const float *rowScales = downScales + ((size_t)expertIdx * ks + st / blockK) * ms;
        const T *expertInput = A + (size_t)task * inter;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < inter; i += THREAD_PER_BLOCK * 4) {
            int remaining = inter - i;
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
        C[(size_t)token * hidden + st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvHalfFP8E4M3FusedTopKDownReduceKernel(
        half *A, const int32_t *indices, uint8_t *downWeight, float *downScales,
        half *C, const float *scores, int batch, int topk, int inter, int hidden, int experts,
        int blockM, int blockK) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int token = blockIdx.y;
    if (token >= batch) {
        return;
    }

    int ms = (inter - 1) / blockM + 1;
    int ks = (hidden - 1) / blockK + 1;
    const float magicScaleConstant = exp2f(8.0f);
    union_half4 regA;

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int task = token * topk + topkSlot;
        int expertIdx = indices[task];
        if (expertIdx < 0 || expertIdx >= experts) {
            continue;
        }
        const uint8_t *baseB = downWeight + ((size_t)expertIdx * hidden + st) * inter;
        float *rowScales = downScales + ((size_t)expertIdx * ks + st / blockK) * ms;
        half *expertInput = A + (size_t)task * inter;

        sdata[tid] = 0.0f;
        for (int i = tid * 4; i < inter; i += THREAD_PER_BLOCK * 4) {
            int remaining = inter - i;
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
        C[(size_t)token * hidden + st] = (half)out;
    }
}

// Qwen3.5 TP2 fused-weight specialization. The lane mapping intentionally
// matches the legacy 64-thread reduction tree so that enabling the default
// fast path does not change FP16/BF16 results.
template <typename T>
__global__ __launch_bounds__(256) void FastllmGemvTypedFP8E4M3FusedTopKSwigluWarpKernel(
        const T *A, const int32_t *indices,
        const uint8_t *gateWeight, const uint8_t *upWeight,
        const float *gateScales, const float *upScales,
        T *C, int experts, float swigluLimit) {
    constexpr int HIDDEN = 2048;
    constexpr int INTER = 256;
    constexpr int TOPK = 8;
    constexpr int BLOCK = 128;
    constexpr int LEGACY_THREADS = 64;
    constexpr int OUTPUTS_PER_CTA = 8;
    constexpr int SCALE_COLS = HIDDEN / BLOCK;
    constexpr int SCALE_ROWS = INTER / BLOCK;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int p = blockIdx.x * OUTPUTS_PER_CTA + warp;
    int topkSlot = blockIdx.y;
    if (p >= INTER || topkSlot >= TOPK) {
        return;
    }

    int expertIdx = indices[topkSlot];
    if (expertIdx < 0 || expertIdx >= experts) {
        if (lane == 0) {
            C[(size_t)topkSlot * INTER + p] = FastllmMoeFp8Traits<T>::fromFloat(0.0f);
        }
        return;
    }

    const uint8_t *baseGate = gateWeight + ((size_t)expertIdx * INTER + p) * HIDDEN;
    const uint8_t *baseUp = upWeight + ((size_t)expertIdx * INTER + p) * HIDDEN;
    const float *gateRowScales = gateScales +
        ((size_t)expertIdx * SCALE_ROWS + p / BLOCK) * SCALE_COLS;
    const float *upRowScales = upScales +
        ((size_t)expertIdx * SCALE_ROWS + p / BLOCK) * SCALE_COLS;

    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
#pragma unroll
    for (int part = 0; part < HIDDEN / (LEGACY_THREADS * 4); part++) {
        int offset0 = lane * 4 + part * LEGACY_THREADS * 4;
        int offset1 = (lane + 32) * 4 + part * LEGACY_THREADS * 4;
        FastllmMoeFp8Accumulate4Legacy(A, offset0,
            *(const uint32_t*)(baseGate + offset0), gateRowScales[offset0 / BLOCK], gate0);
        FastllmMoeFp8Accumulate4Legacy(A, offset1,
            *(const uint32_t*)(baseGate + offset1), gateRowScales[offset1 / BLOCK], gate1);
        FastllmMoeFp8Accumulate4Legacy(A, offset0,
            *(const uint32_t*)(baseUp + offset0), upRowScales[offset0 / BLOCK], up0);
        FastllmMoeFp8Accumulate4Legacy(A, offset1,
            *(const uint32_t*)(baseUp + offset1), upRowScales[offset1 / BLOCK], up1);
    }

    float gate = gate0 + gate1;
    float up = up0 + up1;
    constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(FULL_WARP_MASK, gate, offset);
        up += __shfl_down_sync(FULL_WARP_MASK, up, offset);
    }
    if (lane == 0) {
        const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
        gate = FastllmMoeFp8Round<T>(gate * magicScaleConstant);
        up = FastllmMoeFp8Round<T>(up * magicScaleConstant);
        float gateAct = gate / (1.0f + expf(-gate));
        if (swigluLimit > 0.0f) {
            gateAct = gateAct > swigluLimit ? swigluLimit : gateAct;
            up = up < -swigluLimit ? -swigluLimit : (up > swigluLimit ? swigluLimit : up);
        }
        C[(size_t)topkSlot * INTER + p] = FastllmMoeFp8Traits<T>::fromFloat(gateAct * up);
    }
}

template <typename T>
__global__ __launch_bounds__(256) void FastllmGemvTypedFP8E4M3FusedTopKDownReduceWarpKernel(
        const T *A, const int32_t *indices,
        const uint8_t *downWeight, const float *downScales,
        T *C, const float *scores, int experts) {
    constexpr int INTER = 256;
    constexpr int HIDDEN = 2048;
    constexpr int TOPK = 8;
    constexpr int BLOCK = 128;
    constexpr int OUTPUTS_PER_CTA = 8;
    constexpr int SCALE_COLS = INTER / BLOCK;
    constexpr int SCALE_ROWS = HIDDEN / BLOCK;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int st = blockIdx.x * OUTPUTS_PER_CTA + warp;
    if (st >= HIDDEN) {
        return;
    }

    float out = 0.0f;
    constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;
#pragma unroll
    for (int topkSlot = 0; topkSlot < TOPK; topkSlot++) {
        int expertIdx = indices[topkSlot];
        if (expertIdx < 0 || expertIdx >= experts) {
            continue;
        }
        const uint8_t *baseWeight = downWeight + ((size_t)expertIdx * HIDDEN + st) * INTER;
        const float *rowScales = downScales +
            ((size_t)expertIdx * SCALE_ROWS + st / BLOCK) * SCALE_COLS;
        const T *expertInput = A + (size_t)topkSlot * INTER;

        int offset0 = lane * 4;
        int offset1 = (lane + 32) * 4;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        FastllmMoeFp8Accumulate4Legacy(expertInput, offset0,
            *(const uint32_t*)(baseWeight + offset0), rowScales[offset0 / BLOCK], sum0);
        FastllmMoeFp8Accumulate4Legacy(expertInput, offset1,
            *(const uint32_t*)(baseWeight + offset1), rowScales[offset1 / BLOCK], sum1);
        float sum = sum0 + sum1;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
        }
        if (lane == 0) {
            out += FastllmMoeFp8Round<T>(sum * FastllmMoeFp8Traits<T>::magicScale()) *
                   scores[topkSlot];
        }
    }
    if (lane == 0) {
        C[st] = FastllmMoeFp8Traits<T>::fromFloat(out);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3Block128FusedTopKSwigluKernel(
        const T *A, const int32_t *indices, const uint8_t *gateWeight, const uint8_t *upWeight,
        T *C, int batch, int topk, int hidden, int inter, int experts, int perRow, float swigluLimit) {
    __shared__ float sdataGate[THREAD_PER_BLOCK];
    __shared__ float sdataUp[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    int topkSlot = blockIdx.y;
    int token = blockIdx.z;
    if (token >= batch || topkSlot >= topk) {
        return;
    }

    int task = token * topk + topkSlot;
    int expertIdx = indices[task];
    if (expertIdx < 0 || expertIdx >= experts) {
        if (tid == 0) {
            C[(size_t)task * inter + p] = FastllmMoeFp8Traits<T>::fromFloat(0.0f);
        }
        return;
    }

    const int blockSize = 128;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    const T *tokenInput = A + (size_t)token * hidden;
    const uint8_t *baseGate = gateWeight + ((size_t)expertIdx * inter + p) * perRow;
    const uint8_t *baseUp = upWeight + ((size_t)expertIdx * inter + p) * perRow;
    int numBlocks = (hidden - 1) / blockSize + 1;

    sdataGate[tid] = 0.0f;
    sdataUp[tid] = 0.0f;
    for (int blk = 0; blk < numBlocks; blk++) {
        int blkStart = blk * blockSize;
        int blkEnd = min(blkStart + blockSize, hidden);
        const uint8_t *gateBlock = baseGate + blk * (blockSize + sizeof(float));
        const uint8_t *upBlock = baseUp + blk * (blockSize + sizeof(float));
        float gateScale = *(float*)(gateBlock + blockSize);
        float upScale = *(float*)(upBlock + blockSize);

        for (int i = blkStart + tid * 4; i < blkEnd; i += THREAD_PER_BLOCK * 4) {
            int localIdx = i - blkStart;
            int remaining = blkEnd - i;
            if (remaining >= 4) {
                FastllmMoeFp8Accumulate4(tokenInput, i, *(uint32_t*)(gateBlock + localIdx), gateScale, sdataGate[tid]);
                FastllmMoeFp8Accumulate4(tokenInput, i, *(uint32_t*)(upBlock + localIdx), upScale, sdataUp[tid]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (j < remaining) {
                        float aVal = FastllmMoeFp8Traits<T>::toFloat(tokenInput[i + j]);
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
        float gateAct = gate / (1.0f + expf(-gate));
        if (swigluLimit > 0.0f) {
            gateAct = gateAct > swigluLimit ? swigluLimit : gateAct;
            up = up < -swigluLimit ? -swigluLimit : (up > swigluLimit ? swigluLimit : up);
        }
        C[(size_t)task * inter + p] = FastllmMoeFp8Traits<T>::fromFloat(gateAct * up);
    }
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmGemvTypedFP8E4M3Block128FusedTopKDownReduceKernel(
        const T *A, const int32_t *indices, const uint8_t *downWeight, T *C, const float *scores,
        int batch, int topk, int inter, int hidden, int experts, int perRow) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float out;
    unsigned int tid = threadIdx.x;
    int st = blockIdx.x;
    int token = blockIdx.y;
    if (token >= batch) {
        return;
    }

    const int blockSize = 128;
    const float magicScaleConstant = FastllmMoeFp8Traits<T>::magicScale();
    int numBlocks = (inter - 1) / blockSize + 1;

    if (tid == 0) {
        out = 0.0f;
    }
    __syncthreads();

    for (int topkSlot = 0; topkSlot < topk; topkSlot++) {
        int task = token * topk + topkSlot;
        int expertIdx = indices[task];
        if (expertIdx < 0 || expertIdx >= experts) {
            continue;
        }
        const uint8_t *baseB = downWeight + ((size_t)expertIdx * hidden + st) * perRow;
        const T *expertInput = A + (size_t)task * inter;

        sdata[tid] = 0.0f;
        for (int blk = 0; blk < numBlocks; blk++) {
            int blkStart = blk * blockSize;
            int blkEnd = min(blkStart + blockSize, inter);
            const uint8_t *blkData = baseB + blk * (blockSize + sizeof(float));
            float blkScale = *(float*)(blkData + blockSize);

            for (int i = blkStart + tid * 4; i < blkEnd; i += THREAD_PER_BLOCK * 4) {
                int localIdx = i - blkStart;
                int remaining = blkEnd - i;
                if (remaining >= 4) {
                    FastllmMoeFp8Accumulate4(expertInput, i, *(uint32_t*)(blkData + localIdx), blkScale, sdata[tid]);
                } else {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        if (j < remaining) {
                            sdata[tid] += FastllmMoeFp8Traits<T>::toFloat(expertInput[i + j]) *
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
            out += FastllmMoeFp8Round<T>(sdata[0] * magicScaleConstant) * scores[task];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[(size_t)token * hidden + st] = FastllmMoeFp8Traits<T>::fromFloat(out);
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
static void LaunchFastllmGemmTypedFP8E4M3Block128TopKSwigluIndexed(
        T *input, const int32_t *indices, uint8_t **weights, T *output,
        int topk, int m, int k, int perRow) {
    dim3 grid(k, topk);
    FastllmGemvTypedFP8E4M3Block128TopKSwigluIndexedKernel<T, 64> <<< grid, 64 >>>(
        input, indices, weights, output, topk, m, k, perRow);
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3Block128TopKDownReduceIndexed(
        T *input, const int32_t *indices, uint8_t **weights, T *output, const float *scores,
        int topk, int m, int k, int perRow) {
    FastllmGemvTypedFP8E4M3Block128TopKDownReduceIndexedKernel<T, 64> <<< k, 64 >>>(
        input, indices, weights, output, scores, topk, m, k, perRow);
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3FusedTopKSwiglu(
        T *input, const int32_t *indices, uint8_t *gateWeight, uint8_t *upWeight,
        float *gateScales, float *upScales, T *output,
        int batch, int topk, int hidden, int inter, int experts, int blockM, int blockK,
        float swigluLimit, bool allowWarpSpecialization) {
    if (allowWarpSpecialization && batch == 1 && topk == 8 && hidden == 2048 && inter == 256 &&
        blockM == 128 && blockK == 128) {
        dim3 grid(inter / 8, topk);
        FastllmGemvTypedFP8E4M3FusedTopKSwigluWarpKernel<T><<<grid, 256>>>(
            input, indices, gateWeight, upWeight, gateScales, upScales,
            output, experts, swigluLimit);
        return;
    }
    dim3 grid(inter, topk, batch);
    if constexpr (std::is_same_v<T, half>) {
        FastllmGemvHalfFP8E4M3FusedTopKSwigluKernel<64> <<< grid, 64 >>>(
            input, indices, gateWeight, upWeight, gateScales, upScales, output,
            batch, topk, hidden, inter, experts, blockM, blockK, swigluLimit);
    } else {
        FastllmGemvTypedFP8E4M3FusedTopKSwigluKernel<T, 64> <<< grid, 64 >>>(
            input, indices, gateWeight, upWeight, gateScales, upScales, output,
            batch, topk, hidden, inter, experts, blockM, blockK, swigluLimit);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3FusedTopKDownReduce(
        T *input, const int32_t *indices, uint8_t *downWeight, float *downScales,
        T *output, const float *scores, int batch, int topk, int inter, int hidden, int experts,
        int blockM, int blockK, bool allowWarpSpecialization) {
    if (allowWarpSpecialization && batch == 1 && topk == 8 && inter == 256 && hidden == 2048 &&
        blockM == 128 && blockK == 128) {
        FastllmGemvTypedFP8E4M3FusedTopKDownReduceWarpKernel<T><<<hidden / 8, 256>>>(
            input, indices, downWeight, downScales, output, scores, experts);
        return;
    }
    dim3 grid(hidden, batch);
    if constexpr (std::is_same_v<T, half>) {
        FastllmGemvHalfFP8E4M3FusedTopKDownReduceKernel<64> <<< grid, 64 >>>(
            input, indices, downWeight, downScales, output, scores,
            batch, topk, inter, hidden, experts, blockM, blockK);
    } else {
        FastllmGemvTypedFP8E4M3FusedTopKDownReduceKernel<T, 64> <<< grid, 64 >>>(
            input, indices, downWeight, downScales, output, scores,
            batch, topk, inter, hidden, experts, blockM, blockK);
    }
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3Block128FusedTopKSwiglu(
        const T *input, const int32_t *indices, const uint8_t *gateWeight, const uint8_t *upWeight,
        T *output, int batch, int topk, int hidden, int inter, int experts, int perRow, float swigluLimit) {
    dim3 grid(inter, topk, batch);
    FastllmGemvTypedFP8E4M3Block128FusedTopKSwigluKernel<T, 64> <<< grid, 64 >>>(
        input, indices, gateWeight, upWeight, output, batch, topk, hidden, inter, experts, perRow, swigluLimit);
}

template <typename T>
static void LaunchFastllmGemmTypedFP8E4M3Block128FusedTopKDownReduce(
        const T *input, const int32_t *indices, const uint8_t *downWeight, T *output, const float *scores,
        int batch, int topk, int inter, int hidden, int experts, int perRow) {
    dim3 grid(hidden, batch);
    FastllmGemvTypedFP8E4M3Block128FusedTopKDownReduceKernel<T, 64> <<< grid, 64 >>>(
        input, indices, downWeight, output, scores, batch, topk, inter, hidden, experts, perRow);
}

template <bool SCALE_E8M0, typename T>
static void LaunchFastllmGemmTypedNVFP4TopKSwiglu(T *input, uint8_t **weights, T *output,
                                                  int topk, int m, int k, int perRow) {
    dim3 grid(k, topk);
    FastllmGemvTypedNVFP4Block16TopKSwigluKernel<SCALE_E8M0, T, 64> <<< grid, 64 >>>(
        input, weights, output, m, k, perRow);
}

template <bool SCALE_E8M0, typename T>
static void LaunchFastllmGemmTypedNVFP4TopKDownReduce(T *input, uint8_t **weights, T *output,
                                                      float *scores, int topk, int m, int k, int perRow) {
    FastllmGemvTypedNVFP4Block16TopKDownReduceKernel<SCALE_E8M0, T, 64> <<< k, 64 >>>(
        input, weights, output, scores, topk, m, k, perRow);
}

template <bool SCALE_E8M0, typename T>
static void LaunchFastllmGemmTypedNVFP4TopKSwigluIndexed(T *input, const int32_t *indices,
                                                         uint8_t **weights, T *output,
                                                         int topk, int m, int k, int perRow) {
    dim3 grid(k, topk);
    FastllmGemvTypedNVFP4Block16TopKSwigluIndexedKernel<SCALE_E8M0, T, 64> <<< grid, 64 >>>(
        input, indices, weights, output, topk, m, k, perRow);
}

template <bool SCALE_E8M0, typename T>
static void LaunchFastllmGemmTypedNVFP4TopKDownReduceIndexed(T *input, const int32_t *indices,
                                                             uint8_t **weights, T *output,
                                                             const float *scores, int topk, int m, int k,
                                                             int perRow) {
    if (topk > 0 && topk <= 16) {
        constexpr int groupThreads = 64;
        FastllmGemvTypedNVFP4Block16TopKDownReduceIndexedParallelKernel<
            SCALE_E8M0, T, groupThreads, 16><<<k, topk * groupThreads>>>(
                input, indices, weights, output, scores, topk, m, k, perRow);
        return;
    }
    FastllmGemvTypedNVFP4Block16TopKDownReduceIndexedKernel<SCALE_E8M0, T, 64> <<< k, 64 >>>(
        input, indices, weights, output, scores, topk, m, k, perRow);
}

template <bool SCALE_E8M0, typename T>
static void LaunchFastllmGemmTypedNVFP4SmallBatchTopKSwigluIndexed(T *input, const int32_t *indices,
                                                                   uint8_t **weights, T *output,
                                                                   int batch, int topk, int m, int k,
                                                                   int perRow) {
    dim3 grid(k, batch * topk);
    FastllmGemvTypedNVFP4Block16SmallBatchTopKSwigluIndexedKernel<SCALE_E8M0, T, 64> <<< grid, 64 >>>(
        input, indices, weights, output, batch, topk, m, k, perRow);
}

template <bool SCALE_E8M0, typename T>
static void LaunchFastllmGemmTypedNVFP4SmallBatchTopKDownReduceIndexed(T *input, const int32_t *indices,
                                                                       uint8_t **weights, T *output,
                                                                       const float *scores, int batch, int topk,
                                                                       int m, int k, int perRow) {
    dim3 grid(k, batch);
    FastllmGemvTypedNVFP4Block16SmallBatchTopKDownReduceIndexedKernel<SCALE_E8M0, T, 64> <<< grid, 64 >>>(
        input, indices, weights, output, scores, batch, topk, m, k, perRow);
}

template <typename T>
static void LaunchFastllmGemmTypedNVFP4CompactTopKSwigluIndexed(T *input, const int32_t *indices,
                                                                uint8_t **weights, T *output,
                                                                int topk, int m, int k,
                                                                int blockK, int blockM, int scaleCols) {
    dim3 grid(k, topk);
    FastllmGemvTypedNVFP4CompactTopKSwigluIndexedKernel<T, 64> <<< grid, 64 >>>(
        input, indices, weights, output, topk, m, k, blockK, blockM, scaleCols);
}

template <typename T>
static void LaunchFastllmGemmTypedNVFP4CompactTopKDownReduceIndexed(T *input, const int32_t *indices,
                                                                    uint8_t **weights, T *output,
                                                                    const float *scores, int topk, int m, int k,
                                                                    int blockK, int blockM, int scaleCols) {
    if (topk > 0 && topk <= 16) {
        constexpr int groupThreads = 64;
        FastllmGemvTypedNVFP4CompactTopKDownReduceIndexedParallelKernel<
            T, groupThreads, 16><<<k, topk * groupThreads>>>(
                input, indices, weights, output, scores, topk, m, k,
                blockK, blockM, scaleCols);
        return;
    }
    FastllmGemvTypedNVFP4CompactTopKDownReduceIndexedKernel<T, 64> <<< k, 64 >>>(
        input, indices, weights, output, scores, topk, m, k, blockK, blockM, scaleCols);
}

template <typename T>
static void LaunchFastllmGemmTypedNVFP4CompactExpertParallel(
        T *input, const int32_t *globalIndices, uint8_t **gateWeights,
        uint8_t **downWeights, T *intermediate, T *output, const float *scores,
        int topk, int ownerRank, int ownerCount, int hidden, int inter,
        int gateBlockK, int gateBlockM, int gateScaleCols,
        int downBlockK, int downBlockM, int downScaleCols) {
    dim3 gateGrid(inter, topk);
    FastllmGemvTypedNVFP4CompactExpertParallelSwigluKernel<T, 64><<<gateGrid, 64>>>(
        input, globalIndices, gateWeights, intermediate, topk, ownerRank, ownerCount,
        hidden, inter, gateBlockK, gateBlockM, gateScaleCols);
    FastllmGemvTypedNVFP4CompactExpertParallelDownReduceSerialKernel<T, 64>
        <<<hidden, 64>>>(intermediate, globalIndices, downWeights, output, scores,
                         topk, ownerRank, ownerCount, inter, hidden,
                         downBlockK, downBlockM, downScaleCols);
}

template <typename T>
static void LaunchFastllmGemmTypedNVFP4CompactSharedFP8TopKSwigluIndexed(
        T *input, const int32_t *indices, uint8_t **routedWeights,
        const uint8_t *sharedWeight, const float *sharedScales, T *output,
        int topk, int m, int k,
        int routedBlockK, int routedBlockM, int routedScaleCols,
        int sharedBlockK, int sharedBlockM) {
    dim3 grid(k, topk + 1);
    FastllmGemvTypedNVFP4CompactSharedFP8TopKSwigluIndexedKernel<T, 64><<<grid, 64>>>(
        input, indices, routedWeights, sharedWeight, sharedScales, output,
        topk, m, k, routedBlockK, routedBlockM, routedScaleCols,
        sharedBlockK, sharedBlockM);
}

template <typename T>
static void LaunchFastllmGemmTypedNVFP4CompactSharedFP8TopKDownReduceIndexed(
        T *input, const int32_t *indices, uint8_t **routedWeights,
        const uint8_t *sharedWeight, const float *sharedScales, T *output,
        const float *scores, float sharedScale, int topk, int m, int k,
        int routedBlockK, int routedBlockM, int routedScaleCols,
        int sharedBlockK, int sharedBlockM) {
    constexpr int groupThreads = 64;
    constexpr int maxRoutes = 16;
    FastllmGemvTypedNVFP4CompactSharedFP8TopKDownReduceIndexedParallelKernel<
        T, groupThreads, maxRoutes><<<k, (topk + 1) * groupThreads>>>(
            input, indices, routedWeights, sharedWeight, sharedScales, output,
            scores, sharedScale, topk, m, k,
            routedBlockK, routedBlockM, routedScaleCols,
            sharedBlockK, sharedBlockM);
}

template <typename T>
static void LaunchFastllmGemmTypedNVFP4CompactSmallBatchTopKSwigluIndexed(T *input, const int32_t *indices,
                                                                          uint8_t **weights, T *output,
                                                                          int batch, int topk, int m, int k,
                                                                          int blockK, int blockM, int scaleCols) {
    dim3 grid(k, batch * topk);
    FastllmGemvTypedNVFP4CompactSmallBatchTopKSwigluIndexedKernel<T, 64> <<< grid, 64 >>>(
        input, indices, weights, output, batch, topk, m, k, blockK, blockM, scaleCols);
}

template <typename T>
static void LaunchFastllmGemmTypedNVFP4CompactSmallBatchTopKDownReduceIndexed(T *input, const int32_t *indices,
                                                                              uint8_t **weights, T *output,
                                                                              const float *scores, int batch, int topk,
                                                                              int m, int k,
                                                                              int blockK, int blockM, int scaleCols) {
    dim3 grid(k, batch);
    FastllmGemvTypedNVFP4CompactSmallBatchTopKDownReduceIndexedKernel<T, 64> <<< grid, 64 >>>(
        input, indices, weights, output, scores, batch, topk, m, k, blockK, blockM, scaleCols);
}

template <bool SCALE_E8M0, typename T, int PART>
static void LaunchFastllmGemmTypedNVFP4GroupedTopKSwigluIndexed(T *input, const int *routeRows,
                                                                const int *expertStarts, const int *expertCounts,
                                                                uint8_t **weights, T *output,
                                                                int experts, int maxExpertTasks, int m, int k,
                                                                int perRow) {
    int maxChunks = (maxExpertTasks + PART - 1) / PART;
    dim3 grid(k, experts, maxChunks);
    FastllmGemvTypedNVFP4Block16GroupedTopKSwigluIndexedKernel<SCALE_E8M0, T, 64, PART> <<< grid, 64 >>>(
        input, routeRows, expertStarts, expertCounts, weights, output, maxChunks, m, k, perRow);
}

template <bool SCALE_E8M0, typename T, int PART>
static void LaunchFastllmGemmTypedNVFP4GroupedTopKDownScatterIndexed(T *input, const int *routeRows,
                                                                     const float *routeScales,
                                                                     const int *expertStarts, const int *expertCounts,
                                                                     uint8_t **weights, T *output,
                                                                     int experts, int maxExpertTasks, int m, int k,
                                                                     int perRow) {
    int maxChunks = (maxExpertTasks + PART - 1) / PART;
    dim3 grid(k, experts, maxChunks);
    FastllmGemvTypedNVFP4Block16GroupedTopKDownScatterIndexedKernel<SCALE_E8M0, T, 64, PART> <<< grid, 64 >>>(
        input, routeRows, routeScales, expertStarts, expertCounts, weights, output, maxChunks, m, k, perRow);
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
                                                           int topk, int m, int k, int blockM, int blockK,
                                                           bool allowWarpSpecialization) {
    if (allowWarpSpecialization && topk == 8 && m == 2048 && k == 256 &&
        blockM == 128 && blockK == 128) {
        dim3 grid(k / 8, topk);
        FastllmGemvTypedFP8E4M3TopKSwigluIndexedWarpKernel<T><<<grid, 256>>>(
            input, indices, weights, scales, output);
        return;
    }
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
                                                               int blockM, int blockK,
                                                               bool allowWarpSpecialization) {
    if (allowWarpSpecialization && topk == 8 && m == 256 && k == 2048 &&
        blockM == 128 && blockK == 128) {
        FastllmGemvTypedFP8E4M3TopKDownReduceIndexedWarpKernel<T><<<k / 8, 256>>>(
            input, indices, weights, scales, output, scores);
        return;
    }
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
static std::mutex fastllmMoeFp8Batch1ScratchMutex;

static FastllmMoeFp8Batch1Scratch &FastllmGetMoeFp8Batch1Scratch(int topk) {
    int deviceId = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(fastllmMoeFp8Batch1ScratchMutex);
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
static std::mutex fastllmMoeFp8ExpertTablesMutex;

bool FastllmCudaRegisterMoeFp8ExpertTableFromPacked(fastllm::Data **weights, int weightsBatch, int hidden, int inter,
                                                    void *packedGateWeights, void *packedGateScales,
                                                    void *packedDownWeights, void *packedDownScales,
                                                    int gateBlockM, int gateBlockK, int downBlockM, int downBlockK) {
    if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) ||
        hidden <= 0 || inter <= 0 || gateBlockM <= 0 || gateBlockK <= 0 || downBlockM <= 0 || downBlockK <= 0 ||
        packedGateWeights == nullptr || packedGateScales == nullptr ||
        packedDownWeights == nullptr || packedDownScales == nullptr) {
        return false;
    }
    int experts = weightsBatch / 2 - 1;
    if (experts <= 0 || weights[2] == nullptr) {
        return false;
    }

    int gateScaleRows = (inter * 2 + gateBlockK - 1) / gateBlockK;
    int gateScaleCols = (hidden + gateBlockM - 1) / gateBlockM;
    int downScaleRows = (hidden + downBlockK - 1) / downBlockK;
    int downScaleCols = (inter + downBlockM - 1) / downBlockM;
    size_t gateWeightBytes = (size_t)inter * 2 * hidden;
    size_t downWeightBytes = (size_t)hidden * inter;
    size_t gateScaleElements = (size_t)gateScaleRows * gateScaleCols;
    size_t downScaleElements = (size_t)downScaleRows * downScaleCols;

    std::vector<uint8_t*> hGateWeights(experts), hDownWeights(experts);
    std::vector<float*> hGateScales(experts), hDownScales(experts);
    for (int e = 0; e < experts; e++) {
        hGateWeights[e] = (uint8_t*)packedGateWeights + (size_t)e * gateWeightBytes;
        hDownWeights[e] = (uint8_t*)packedDownWeights + (size_t)e * downWeightBytes;
        hGateScales[e] = (float*)packedGateScales + (size_t)e * gateScaleElements;
        hDownScales[e] = (float*)packedDownScales + (size_t)e * downScaleElements;
    }

    int deviceId = FastllmCudaGetDevice();
    auto key = std::make_pair(deviceId, (const void*)weights[2]);
    std::lock_guard<std::mutex> guard(fastllmMoeFp8ExpertTablesMutex);
    FastllmMoeFp8ExpertTable &cached = fastllmMoeFp8ExpertTables[key];
    if (cached.inited &&
        (cached.experts != experts || cached.hidden != hidden || cached.inter != inter)) {
        return false;
    }

    size_t ptrBytes = (size_t)experts * sizeof(void*);
    if (cached.gateWeights == nullptr) {
        cached.gateWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
    }
    if (cached.gateScales == nullptr) {
        cached.gateScales = (float**)FastllmCudaMalloc(ptrBytes);
    }
    if (cached.downWeights == nullptr) {
        cached.downWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
    }
    if (cached.downScales == nullptr) {
        cached.downScales = (float**)FastllmCudaMalloc(ptrBytes);
    }
    if (cached.gateWeights == nullptr || cached.gateScales == nullptr ||
        cached.downWeights == nullptr || cached.downScales == nullptr) {
        return false;
    }

    cudaError_t state = cudaSuccess;
    state = cudaMemcpyAsync(cached.gateWeights, hGateWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when registering packed MoE gate pointer table!", state);
    state = cudaMemcpyAsync(cached.gateScales, hGateScales.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when registering packed MoE gate scale table!", state);
    state = cudaMemcpyAsync(cached.downWeights, hDownWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when registering packed MoE down pointer table!", state);
    state = cudaMemcpyAsync(cached.downScales, hDownScales.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when registering packed MoE down scale table!", state);

    cached.inited = true;
    cached.experts = experts;
    cached.hidden = hidden;
    cached.inter = inter;
    cached.gateBlockM = gateBlockM;
    cached.gateBlockK = gateBlockK;
    cached.downBlockM = downBlockM;
    cached.downBlockK = downBlockK;
    return true;
}

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
    std::lock_guard<std::mutex> guard(fastllmMoeFp8ExpertTablesMutex);
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

struct FastllmMoeFp8Block128ExpertTable {
    bool inited = false;
    int experts = 0;
    int hidden = 0;
    int inter = 0;
    int gatePerRow = 0;
    int downPerRow = 0;
    uint8_t **gateWeights = nullptr;
    uint8_t **downWeights = nullptr;
};

static std::map<std::pair<int, const void*>, FastllmMoeFp8Block128ExpertTable> fastllmMoeFp8Block128ExpertTables;
static std::mutex fastllmMoeFp8Block128ExpertTablesMutex;

static bool FastllmGetMoeFp8Block128ExpertTable(fastllm::Data **weights, int weightsBatch, int hidden, int inter,
                                                FastllmMoeFp8Block128ExpertTable *&table) {
    if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) ||
        hidden <= 0 || inter <= 0) {
        return false;
    }
    int experts = weightsBatch / 2 - 1;
    if (experts <= 0) {
        return false;
    }

    int deviceId = FastllmCudaGetDevice();
    auto key = std::make_pair(deviceId, (const void*)weights[2]);
    std::lock_guard<std::mutex> guard(fastllmMoeFp8Block128ExpertTablesMutex);
    FastllmMoeFp8Block128ExpertTable &cached = fastllmMoeFp8Block128ExpertTables[key];
    if (cached.inited) {
        if (cached.experts != experts || cached.hidden != hidden || cached.inter != inter) {
            return false;
        }
        table = &cached;
        return true;
    }

    std::vector<uint8_t*> hGateWeights(experts), hDownWeights(experts);
    int gatePerRow = hidden + ((hidden - 1) / 128 + 1) * (int)sizeof(float);
    int downPerRow = inter + ((inter - 1) / 128 + 1) * (int)sizeof(float);
    for (int e = 0; e < experts; e++) {
        int idx = (e + 1) * 2;
        fastllm::Data *gateup = weights[idx];
        fastllm::Data *down = weights[idx + 1];
        if (gateup == nullptr || down == nullptr ||
            gateup->dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 ||
            down->dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 ||
            gateup->dims.size() != 2 || down->dims.size() != 2 ||
            gateup->dims[1] != hidden || gateup->dims[0] != inter * 2 ||
            down->dims[1] != inter || down->dims[0] != hidden ||
            gateup->cudaData == nullptr || down->cudaData == nullptr) {
            return false;
        }
        hGateWeights[e] = (uint8_t*)gateup->cudaData;
        hDownWeights[e] = (uint8_t*)down->cudaData;
    }

    size_t ptrBytes = (size_t)experts * sizeof(void*);
    cached.gateWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
    cached.downWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);

    cudaError_t state = cudaMemcpyAsync(cached.gateWeights, hGateWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when caching block128 MoE gate pointer table!", state);
    state = cudaMemcpyAsync(cached.downWeights, hDownWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when caching block128 MoE down pointer table!", state);

    cached.inited = true;
    cached.experts = experts;
    cached.hidden = hidden;
    cached.inter = inter;
    cached.gatePerRow = gatePerRow;
    cached.downPerRow = downPerRow;
    table = &cached;
    return true;
}

static inline bool FastllmMoeNVFP4IsWeightType(fastllm::DataType type) {
    return type == fastllm::DataType::NVFP4 ||
           type == fastllm::DataType::NVFP4_BLOCK_16 ||
           type == fastllm::DataType::NVFP4_BLOCK_16_E8M0;
}

static inline bool FastllmMoeNVFP4ScaleE8M0(fastllm::DataType type) {
    return type == fastllm::DataType::NVFP4 ||
           type == fastllm::DataType::NVFP4_BLOCK_16_E8M0;
}

struct FastllmMoeNVFP4Batch1Scratch {
    int capacity = 0;
    uint8_t **gateWeights = nullptr;
    uint8_t **downWeights = nullptr;
    float *scores = nullptr;
};

static std::map<int, FastllmMoeNVFP4Batch1Scratch> fastllmMoeNVFP4Batch1Scratch;
static std::mutex fastllmMoeNVFP4Batch1ScratchMutex;

static FastllmMoeNVFP4Batch1Scratch &FastllmGetMoeNVFP4Batch1Scratch(int topk) {
    int deviceId = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(fastllmMoeNVFP4Batch1ScratchMutex);
    FastllmMoeNVFP4Batch1Scratch &scratch = fastllmMoeNVFP4Batch1Scratch[deviceId];
    if (scratch.capacity < topk) {
        scratch.capacity = topk;
        size_t ptrBytes = (size_t)topk * sizeof(void*);
        scratch.gateWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
        scratch.downWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
        scratch.scores = (float*)FastllmCudaMalloc((size_t)topk * sizeof(float));
    }
    return scratch;
}

struct FastllmMoeNVFP4ExpertTable {
    bool inited = false;
    bool compact = false;
    bool scaleE8M0 = false;
    int experts = 0;
    int hidden = 0;
    int inter = 0;
    int gatePerRow = 0;
    int downPerRow = 0;
    int gateBlockK = 0;
    int gateBlockM = 0;
    int gateScaleCols = 0;
    int downBlockK = 0;
    int downBlockM = 0;
    int downScaleCols = 0;
    uint8_t **gateWeights = nullptr;
    uint8_t **downWeights = nullptr;
};

static std::map<std::pair<int, const void*>, FastllmMoeNVFP4ExpertTable> fastllmMoeNVFP4ExpertTables;
static std::mutex fastllmMoeNVFP4ExpertTablesMutex;

static bool FastllmGetMoeNVFP4ExpertTable(fastllm::Data **weights, int weightsBatch, int hidden, int inter,
                                          FastllmMoeNVFP4ExpertTable *&table) {
    if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1)) {
        return false;
    }
    int experts = weightsBatch / 2 - 1;
    if (experts <= 0) {
        return false;
    }

    fastllm::Data *firstGate = weights[2];
    fastllm::Data *firstDown = weights[3];
    if (firstGate == nullptr || firstDown == nullptr ||
        !FastllmMoeNVFP4IsWeightType(firstGate->dataType) ||
        firstGate->dataType != firstDown->dataType) {
        return false;
    }
    fastllm::DataType weightType = firstGate->dataType;
    bool compact = weightType == fastllm::DataType::NVFP4;
    bool scaleE8M0 = FastllmMoeNVFP4ScaleE8M0(weightType);
    int gateBlockK = compact ? firstGate->blockK : 0;
    int gateBlockM = compact ? firstGate->blockM : 0;
    int downBlockK = compact ? firstDown->blockK : 0;
    int downBlockM = compact ? firstDown->blockM : 0;
    if (compact && (gateBlockK <= 0 || gateBlockM <= 0 || downBlockK <= 0 || downBlockM <= 0 ||
                    !firstGate->scales.empty() || !firstDown->scales.empty())) {
        return false;
    }
    int gateScaleCols = compact ? ((hidden - 1) / gateBlockM + 1) : 0;
    int downScaleCols = compact ? ((inter - 1) / downBlockM + 1) : 0;

    int deviceId = FastllmCudaGetDevice();
    auto key = std::make_pair(deviceId, (const void*)weights[2]);
    std::lock_guard<std::mutex> guard(fastllmMoeNVFP4ExpertTablesMutex);
    FastllmMoeNVFP4ExpertTable &cached = fastllmMoeNVFP4ExpertTables[key];
    if (cached.inited) {
        if (cached.experts != experts || cached.hidden != hidden || cached.inter != inter ||
            cached.compact != compact || cached.scaleE8M0 != scaleE8M0 ||
            cached.gateBlockK != gateBlockK || cached.gateBlockM != gateBlockM ||
            cached.downBlockK != downBlockK || cached.downBlockM != downBlockM) {
            return false;
        }
        table = &cached;
        return true;
    }

    std::vector<uint8_t*> hGateWeights(experts), hDownWeights(experts);
    for (int e = 0; e < experts; e++) {
        int idx = (e + 1) * 2;
        fastllm::Data *gateup = weights[idx];
        fastllm::Data *down = weights[idx + 1];
        if (gateup == nullptr || down == nullptr ||
            gateup->dataType != weightType || down->dataType != weightType ||
            gateup->dims.size() != 2 || down->dims.size() != 2 ||
            gateup->dims[1] != hidden || gateup->dims[0] != inter * 2 ||
            down->dims[1] != inter || down->dims[0] != hidden ||
            gateup->cudaData == nullptr || down->cudaData == nullptr) {
            return false;
        }
        if (compact && (gateup->blockK != gateBlockK || gateup->blockM != gateBlockM ||
                        down->blockK != downBlockK || down->blockM != downBlockM ||
                        !gateup->scales.empty() || !down->scales.empty())) {
            return false;
        }
        hGateWeights[e] = (uint8_t*)gateup->cudaData;
        hDownWeights[e] = (uint8_t*)down->cudaData;
    }

    size_t ptrBytes = (size_t)experts * sizeof(void*);
    cached.gateWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
    cached.downWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);

    cudaError_t state = cudaSuccess;
    state = cudaMemcpyAsync(cached.gateWeights, hGateWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when caching NVFP4 MoE gate pointer table!", state);
    state = cudaMemcpyAsync(cached.downWeights, hDownWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when caching NVFP4 MoE down pointer table!", state);

    cached.inited = true;
    cached.compact = compact;
    cached.scaleE8M0 = scaleE8M0;
    cached.experts = experts;
    cached.hidden = hidden;
    cached.inter = inter;
    cached.gatePerRow = (int)FastllmMoeNVFP4Block16BytesPerRow(hidden, scaleE8M0);
    cached.downPerRow = (int)FastllmMoeNVFP4Block16BytesPerRow(inter, scaleE8M0);
    cached.gateBlockK = gateBlockK;
    cached.gateBlockM = gateBlockM;
    cached.gateScaleCols = gateScaleCols;
    cached.downBlockK = downBlockK;
    cached.downBlockM = downBlockM;
    cached.downScaleCols = downScaleCols;
    table = &cached;
    return true;
}

template <typename T>
static bool FastllmCudaTypedMergeMOENVFP4Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                       fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                       const float *scores, int topk, int hidden, int inter) {
    if (topk <= 0 || hidden <= 0 || inter <= 0 || indices == nullptr || scores == nullptr ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType || input.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }

    FastllmMoeNVFP4ExpertTable *table = nullptr;
    if (!FastllmGetMoeNVFP4ExpertTable(weights, weightsBatch, hidden, inter, table)) {
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

    if (table->compact) {
        LaunchFastllmGemmTypedNVFP4CompactTopKSwigluIndexed(cudaInput, indices, table->gateWeights, cudaW1,
                                                            topk, hidden, inter,
                                                            table->gateBlockK, table->gateBlockM, table->gateScaleCols);
        LaunchFastllmGemmTypedNVFP4CompactTopKDownReduceIndexed(cudaW1, indices, table->downWeights, cudaOutput, scores,
                                                                topk, inter, hidden,
                                                                table->downBlockK, table->downBlockM, table->downScaleCols);
    } else if (table->scaleE8M0) {
        LaunchFastllmGemmTypedNVFP4TopKSwigluIndexed<true>(cudaInput, indices, table->gateWeights, cudaW1,
                                                           topk, hidden, inter, table->gatePerRow);
        LaunchFastllmGemmTypedNVFP4TopKDownReduceIndexed<true>(cudaW1, indices, table->downWeights, cudaOutput, scores,
                                                               topk, inter, hidden, table->downPerRow);
    } else {
        LaunchFastllmGemmTypedNVFP4TopKSwigluIndexed<false>(cudaInput, indices, table->gateWeights, cudaW1,
                                                            topk, hidden, inter, table->gatePerRow);
        LaunchFastllmGemmTypedNVFP4TopKDownReduceIndexed<false>(cudaW1, indices, table->downWeights, cudaOutput, scores,
                                                                topk, inter, hidden, table->downPerRow);
    }

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

template <typename T>
static bool FastllmCudaTypedMergeMOENVFP4Batch1IndexedSharedFP8(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *indices,
        const float *scores, float sharedScale, int topk, int hidden, int inter) {
    if (topk <= 0 || topk + 1 > 16 || hidden <= 0 || inter <= 0 ||
        indices == nullptr || scores == nullptr || sharedScale == 0.0f ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        weights == nullptr || weightsBatch < 4) {
        return false;
    }

    fastllm::Data *sharedGateup = weights[0];
    fastllm::Data *sharedDown = weights[1];
    if (sharedGateup == nullptr || sharedDown == nullptr ||
        sharedGateup->dataType != fastllm::DataType::FP8_E4M3 ||
        sharedDown->dataType != fastllm::DataType::FP8_E4M3 ||
        sharedGateup->dims.size() != 2 || sharedDown->dims.size() != 2 ||
        sharedGateup->dims[0] != inter * 2 || sharedGateup->dims[1] != hidden ||
        sharedDown->dims[0] != hidden || sharedDown->dims[1] != inter ||
        sharedGateup->blockK <= 0 || sharedGateup->blockM <= 0 ||
        sharedDown->blockK <= 0 || sharedDown->blockM <= 0 ||
        sharedGateup->cudaData == nullptr || sharedDown->cudaData == nullptr) {
        return false;
    }

    FastllmMoeNVFP4ExpertTable *table = nullptr;
    if (!FastllmGetMoeNVFP4ExpertTable(weights, weightsBatch, hidden, inter, table) ||
        table == nullptr || !table->compact) {
        return false;
    }

    fastllm::Data emptyBias;
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(*sharedGateup, emptyBias, inter);
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(*sharedDown, emptyBias, hidden);
    if (sharedGateup->extraCudaData.empty() || sharedGateup->extraCudaData[0] == nullptr ||
        sharedDown->extraCudaData.empty() || sharedDown->extraCudaData[0] == nullptr) {
        return false;
    }

    w1.dataDevice = input.dataDevice;
    w1.dataDeviceIds = input.dataDeviceIds;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = FastllmMoeFp8Traits<T>::dataType;
    w1.Resize({topk + 1, inter});
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Resize({1, hidden});
    w1.Allocate(false);
    output.Allocate(false);

    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaW1 = (T*)FastllmCudaPrepareOutput(w1);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);
    LaunchFastllmGemmTypedNVFP4CompactSharedFP8TopKSwigluIndexed(
        cudaInput, indices, table->gateWeights,
        (const uint8_t*)sharedGateup->cudaData,
        (const float*)sharedGateup->extraCudaData[0], cudaW1,
        topk, hidden, inter,
        table->gateBlockK, table->gateBlockM, table->gateScaleCols,
        sharedGateup->blockK, sharedGateup->blockM);
    LaunchFastllmGemmTypedNVFP4CompactSharedFP8TopKDownReduceIndexed(
        cudaW1, indices, table->downWeights,
        (const uint8_t*)sharedDown->cudaData,
        (const float*)sharedDown->extraCudaData[0], cudaOutput,
        scores, sharedScale, topk, inter, hidden,
        table->downBlockK, table->downBlockM, table->downScaleCols,
        sharedDown->blockK, sharedDown->blockM);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOENVFP4Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                               fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                               const float *scores, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4Batch1Indexed<half>(
        input, w1, output, weights, weightsBatch, indices, scores, topk, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOENVFP4Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                   fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                   const float *scores, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4Batch1Indexed<__nv_bfloat16>(
        input, w1, output, weights, weightsBatch, indices, scores, topk, hidden, inter);
}

template <typename T>
static bool FastllmCudaTypedMergeMOENVFP4Batch1ExpertParallel(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *globalIndices,
        const float *scores, int topk, int ownerRank, int ownerCount) {
    if (topk <= 0 || topk > 16 || ownerCount <= 0 || globalIndices == nullptr || scores == nullptr ||
        input.dims.empty() || input.dataType != FastllmMoeFp8Traits<T>::dataType ||
        input.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    int hidden = input.dims.back();
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Resize({1, hidden});
    output.Allocate(false);
    if (ownerRank < 0) {
        cudaError_t state = cudaMemsetAsync(output.cudaData, 0, output.GetBytes(), cudaStreamPerThread);
        checkCudaErrors("Error: CUDA error when zeroing idle EP output!", state);
        return true;
    }

    FastllmMoeNVFP4ExpertTable *table = nullptr;
    if (!FastllmGetMoeNVFP4ExpertTable(weights, weightsBatch, hidden,
                                       weights[3] == nullptr ? 0 : weights[3]->dims[1], table) ||
        table == nullptr || !table->compact) {
        return false;
    }
    int inter = table->inter;
    w1.dataDevice = input.dataDevice;
    w1.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = FastllmMoeFp8Traits<T>::dataType;
    w1.Resize({topk, inter});
    w1.Allocate(false);

    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaW1 = (T*)FastllmCudaPrepareOutput(w1);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);
    LaunchFastllmGemmTypedNVFP4CompactExpertParallel(
        cudaInput, globalIndices, table->gateWeights, table->downWeights,
        cudaW1, cudaOutput, scores, topk, ownerRank, ownerCount, hidden, inter,
        table->gateBlockK, table->gateBlockM, table->gateScaleCols,
        table->downBlockK, table->downBlockM, table->downScaleCols);
    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOENVFP4Batch1ExpertParallel(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *globalIndices,
        const float *scores, int topk, int ownerRank, int ownerCount) {
    return FastllmCudaTypedMergeMOENVFP4Batch1ExpertParallel<half>(
        input, w1, output, weights, weightsBatch, globalIndices, scores,
        topk, ownerRank, ownerCount);
}

bool FastllmCudaBFloat16MergeMOENVFP4Batch1ExpertParallel(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *globalIndices,
        const float *scores, int topk, int ownerRank, int ownerCount) {
    return FastllmCudaTypedMergeMOENVFP4Batch1ExpertParallel<__nv_bfloat16>(
        input, w1, output, weights, weightsBatch, globalIndices, scores,
        topk, ownerRank, ownerCount);
}

bool FastllmCudaHalfMergeMOENVFP4Batch1IndexedSharedFP8(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *indices,
        const float *scores, float sharedScale, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4Batch1IndexedSharedFP8<half>(
        input, w1, output, weights, weightsBatch, indices, scores,
        sharedScale, topk, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOENVFP4Batch1IndexedSharedFP8(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *indices,
        const float *scores, float sharedScale, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4Batch1IndexedSharedFP8<__nv_bfloat16>(
        input, w1, output, weights, weightsBatch, indices, scores,
        sharedScale, topk, hidden, inter);
}

template <typename T>
static bool FastllmCudaTypedMergeMOENVFP4SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                           fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                           const float *scores, int batch, int topk, int hidden, int inter) {
    if (batch <= 0 || batch > 64 || topk <= 0 || hidden <= 0 || inter <= 0 || indices == nullptr || scores == nullptr ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType || input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dims.size() == 0 || input.dims[0] != batch) {
        return false;
    }

    FastllmMoeNVFP4ExpertTable *table = nullptr;
    if (!FastllmGetMoeNVFP4ExpertTable(weights, weightsBatch, hidden, inter, table)) {
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

    if (table->compact) {
        LaunchFastllmGemmTypedNVFP4CompactSmallBatchTopKSwigluIndexed(cudaInput, indices, table->gateWeights, cudaW1,
                                                                      batch, topk, hidden, inter,
                                                                      table->gateBlockK, table->gateBlockM, table->gateScaleCols);
        LaunchFastllmGemmTypedNVFP4CompactSmallBatchTopKDownReduceIndexed(cudaW1, indices, table->downWeights, cudaOutput, scores,
                                                                          batch, topk, inter, hidden,
                                                                          table->downBlockK, table->downBlockM, table->downScaleCols);
    } else if (table->scaleE8M0) {
        LaunchFastllmGemmTypedNVFP4SmallBatchTopKSwigluIndexed<true>(cudaInput, indices, table->gateWeights, cudaW1,
                                                                     batch, topk, hidden, inter, table->gatePerRow);
        LaunchFastllmGemmTypedNVFP4SmallBatchTopKDownReduceIndexed<true>(cudaW1, indices, table->downWeights, cudaOutput, scores,
                                                                         batch, topk, inter, hidden, table->downPerRow);
    } else {
        LaunchFastllmGemmTypedNVFP4SmallBatchTopKSwigluIndexed<false>(cudaInput, indices, table->gateWeights, cudaW1,
                                                                      batch, topk, hidden, inter, table->gatePerRow);
        LaunchFastllmGemmTypedNVFP4SmallBatchTopKDownReduceIndexed<false>(cudaW1, indices, table->downWeights, cudaOutput, scores,
                                                                          batch, topk, inter, hidden, table->downPerRow);
    }

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOENVFP4SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                   fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                   const float *scores, int batch, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4SmallBatchIndexed<half>(
        input, w1, output, weights, weightsBatch, indices, scores, batch, topk, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOENVFP4SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                       fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                       const float *scores, int batch, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4SmallBatchIndexed<__nv_bfloat16>(
        input, w1, output, weights, weightsBatch, indices, scores, batch, topk, hidden, inter);
}

template <typename T>
static bool FastllmCudaTypedMergeMOENVFP4GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
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

    FastllmMoeNVFP4ExpertTable *table = nullptr;
    if (!FastllmGetMoeNVFP4ExpertTable(weights, weightsBatch, hidden, inter, table)) {
        return false;
    }
    if (table->compact) {
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
    checkCudaErrors("Error: CUDA error when copying NVFP4 grouped MoE route rows!", state);
    state = cudaMemcpyAsync(cudaRouteScales, routeScales, (size_t)totalTasks * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copying NVFP4 grouped MoE route scales!", state);
    state = cudaMemcpyAsync(cudaRoutePositions, routePositions, (size_t)batch * topk * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copying NVFP4 grouped MoE route positions!", state);
    state = cudaMemcpyAsync(cudaExpertStarts, expertStarts, (size_t)experts * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copying NVFP4 grouped MoE expert starts!", state);
    state = cudaMemcpyAsync(cudaExpertCounts, expertCounts, (size_t)experts * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copying NVFP4 grouped MoE expert counts!", state);

    if (table->scaleE8M0) {
        if (maxExpertTasks <= 8) {
            LaunchFastllmGemmTypedNVFP4GroupedTopKSwigluIndexed<true, T, 8>(
                cudaInput, cudaRouteRows, cudaExpertStarts, cudaExpertCounts,
                table->gateWeights, cudaW1, experts, maxExpertTasks, hidden, inter, table->gatePerRow);
            LaunchFastllmGemmTypedNVFP4GroupedTopKDownScatterIndexed<true, T, 8>(
                cudaW1, cudaRouteRows, cudaRouteScales, cudaExpertStarts, cudaExpertCounts,
                table->downWeights, cudaW2, experts, maxExpertTasks, inter, hidden, table->downPerRow);
        } else {
            LaunchFastllmGemmTypedNVFP4GroupedTopKSwigluIndexed<true, T, 16>(
                cudaInput, cudaRouteRows, cudaExpertStarts, cudaExpertCounts,
                table->gateWeights, cudaW1, experts, maxExpertTasks, hidden, inter, table->gatePerRow);
            LaunchFastllmGemmTypedNVFP4GroupedTopKDownScatterIndexed<true, T, 16>(
                cudaW1, cudaRouteRows, cudaRouteScales, cudaExpertStarts, cudaExpertCounts,
                table->downWeights, cudaW2, experts, maxExpertTasks, inter, hidden, table->downPerRow);
        }
    } else {
        if (maxExpertTasks <= 8) {
            LaunchFastllmGemmTypedNVFP4GroupedTopKSwigluIndexed<false, T, 8>(
                cudaInput, cudaRouteRows, cudaExpertStarts, cudaExpertCounts,
                table->gateWeights, cudaW1, experts, maxExpertTasks, hidden, inter, table->gatePerRow);
            LaunchFastllmGemmTypedNVFP4GroupedTopKDownScatterIndexed<false, T, 8>(
                cudaW1, cudaRouteRows, cudaRouteScales, cudaExpertStarts, cudaExpertCounts,
                table->downWeights, cudaW2, experts, maxExpertTasks, inter, hidden, table->downPerRow);
        } else {
            LaunchFastllmGemmTypedNVFP4GroupedTopKSwigluIndexed<false, T, 16>(
                cudaInput, cudaRouteRows, cudaExpertStarts, cudaExpertCounts,
                table->gateWeights, cudaW1, experts, maxExpertTasks, hidden, inter, table->gatePerRow);
            LaunchFastllmGemmTypedNVFP4GroupedTopKDownScatterIndexed<false, T, 16>(
                cudaW1, cudaRouteRows, cudaRouteScales, cudaExpertStarts, cudaExpertCounts,
                table->downWeights, cudaW2, experts, maxExpertTasks, inter, hidden, table->downPerRow);
        }
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

bool FastllmCudaHalfMergeMOENVFP4GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                fastllm::Data **weights, int weightsBatch,
                                                const int *routeRows, const float *routeScales,
                                                const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4GroupedIndexed<half>(
        input, w1, w2, output, weights, weightsBatch,
        routeRows, routeScales, routePositions, expertStarts, expertCounts,
        batch, topk, totalTasks, maxExpertTasks, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOENVFP4GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                    fastllm::Data **weights, int weightsBatch,
                                                    const int *routeRows, const float *routeScales,
                                                    const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                    int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4GroupedIndexed<__nv_bfloat16>(
        input, w1, w2, output, weights, weightsBatch,
        routeRows, routeScales, routePositions, expertStarts, expertCounts,
        batch, topk, totalTasks, maxExpertTasks, hidden, inter);
}

template <typename T>
static bool FastllmCudaTypedMergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                         fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                         const float *scores, int topk, int hidden, int inter,
                                                         bool allowWarpSpecialization) {
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
                                                   topk, hidden, inter, table->gateBlockM, table->gateBlockK,
                                                   allowWarpSpecialization);
    LaunchFastllmGemmTypedFP8E4M3TopKDownReduceIndexed(cudaW1, indices, table->downWeights, table->downScales, cudaOutput, scores,
                                                       topk, inter, hidden, table->downBlockM, table->downBlockK,
                                                       allowWarpSpecialization);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                 fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                 const float *scores, int topk, int hidden, int inter,
                                                 bool allowWarpSpecialization) {
    return FastllmCudaTypedMergeMOEFP8E4M3Batch1Indexed<half>(
        input, w1, output, weights, weightsBatch, indices, scores, topk, hidden, inter,
        allowWarpSpecialization);
}

bool FastllmCudaBFloat16MergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                     fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                     const float *scores, int topk, int hidden, int inter,
                                                     bool allowWarpSpecialization) {
    return FastllmCudaTypedMergeMOEFP8E4M3Batch1Indexed<__nv_bfloat16>(
        input, w1, output, weights, weightsBatch, indices, scores, topk, hidden, inter,
        allowWarpSpecialization);
}

template <typename T>
static bool FastllmCudaTypedMergeMOEFP8E4M3Block128Batch1Indexed(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *indices,
        const float *scores, int topk, int hidden, int inter) {
    if (topk <= 0 || hidden <= 0 || inter <= 0 || indices == nullptr || scores == nullptr ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType ||
        input.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }

    FastllmMoeFp8Block128ExpertTable *table = nullptr;
    if (!FastllmGetMoeFp8Block128ExpertTable(weights, weightsBatch, hidden, inter, table)) {
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

    LaunchFastllmGemmTypedFP8E4M3Block128TopKSwigluIndexed(
        cudaInput, indices, table->gateWeights, cudaW1, topk, hidden, inter, table->gatePerRow);
    LaunchFastllmGemmTypedFP8E4M3Block128TopKDownReduceIndexed(
        cudaW1, indices, table->downWeights, cudaOutput, scores, topk, inter, hidden, table->downPerRow);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOEFP8E4M3Block128Batch1Indexed(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *indices,
        const float *scores, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3Block128Batch1Indexed<half>(
        input, w1, output, weights, weightsBatch, indices, scores, topk, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOEFP8E4M3Block128Batch1Indexed(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *indices,
        const float *scores, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEFP8E4M3Block128Batch1Indexed<__nv_bfloat16>(
        input, w1, output, weights, weightsBatch, indices, scores, topk, hidden, inter);
}

template <typename T>
static bool FastllmCudaTypedFusedMOEFP8E4M3(
        const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up, fastllm::Data &down,
        const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &w1, fastllm::Data &output,
        int batch, int topk, int hidden, int inter, int experts, float swigluLimit,
        bool allowWarpSpecialization) {
    if (batch <= 0 || topk <= 0 || hidden <= 0 || inter <= 0 || experts <= 0 ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        index.dataDevice != fastllm::DataDevice::CUDA || index.dataType != fastllm::DataType::INT32 ||
        score.dataDevice != fastllm::DataDevice::CUDA || score.dataType != fastllm::DataType::FLOAT32 ||
        gate.dataDevice != fastllm::DataDevice::CUDA ||
        up.dataDevice != fastllm::DataDevice::CUDA ||
        down.dataDevice != fastllm::DataDevice::CUDA ||
        gate.dataType != fastllm::DataType::FP8_E4M3 ||
        up.dataType != fastllm::DataType::FP8_E4M3 ||
        down.dataType != fastllm::DataType::FP8_E4M3 ||
        gate.cudaData == nullptr || up.cudaData == nullptr || down.cudaData == nullptr ||
        index.cudaData == nullptr || score.cudaData == nullptr ||
        gate.extraCudaData.empty() || up.extraCudaData.empty() || down.extraCudaData.empty()) {
        return false;
    }
    if (gate.dims.size() != 3 || up.dims.size() != 3 || down.dims.size() != 3 ||
        gate.dims[0] != experts || gate.dims[1] != inter || gate.dims[2] != hidden ||
        up.dims[0] != experts || up.dims[1] != inter || up.dims[2] != hidden ||
        down.dims[0] != experts || down.dims[1] != hidden || down.dims[2] != inter ||
        gate.blockM <= 0 || gate.blockK <= 0 ||
        up.blockM != gate.blockM || up.blockK != gate.blockK ||
        down.blockM <= 0 || down.blockK <= 0) {
        return false;
    }

    w1.dataDevice = input.dataDevice;
    w1.dataDeviceIds = input.dataDeviceIds;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = FastllmMoeFp8Traits<T>::dataType;
    w1.Resize({batch * topk, inter});
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Resize(input.dims);
    w1.Allocate(false);
    output.Allocate(false);

    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaW1 = (T*)FastllmCudaPrepareOutput(w1);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);
    const int32_t *cudaIndex = (const int32_t*)index.cudaData;
    const float *cudaScore = (const float*)score.cudaData;
    uint8_t *cudaGate = (uint8_t*)gate.cudaData;
    uint8_t *cudaUp = (uint8_t*)up.cudaData;
    uint8_t *cudaDown = (uint8_t*)down.cudaData;
    float *cudaGateScales = (float*)gate.extraCudaData[0];
    float *cudaUpScales = (float*)up.extraCudaData[0];
    float *cudaDownScales = (float*)down.extraCudaData[0];

    LaunchFastllmGemmTypedFP8E4M3FusedTopKSwiglu(
        cudaInput, cudaIndex, cudaGate, cudaUp, cudaGateScales, cudaUpScales, cudaW1,
        batch, topk, hidden, inter, experts, gate.blockM, gate.blockK, swigluLimit,
        allowWarpSpecialization);
    LaunchFastllmGemmTypedFP8E4M3FusedTopKDownReduce(
        cudaW1, cudaIndex, cudaDown, cudaDownScales, cudaOutput, cudaScore,
        batch, topk, inter, hidden, experts, down.blockM, down.blockK,
        allowWarpSpecialization);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfFusedMOEFP8E4M3(
        const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up,
        fastllm::Data &down, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &w1, fastllm::Data &output,
        int batch, int topk, int hidden, int inter, int experts, float swigluLimit,
        bool allowWarpSpecialization) {
    return FastllmCudaTypedFusedMOEFP8E4M3<half>(
        input, gate, up, down, index, score, w1, output,
        batch, topk, hidden, inter, experts, swigluLimit, allowWarpSpecialization);
}

bool FastllmCudaBFloat16FusedMOEFP8E4M3(
        const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up,
        fastllm::Data &down, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &w1, fastllm::Data &output,
        int batch, int topk, int hidden, int inter, int experts, float swigluLimit,
        bool allowWarpSpecialization) {
    return FastllmCudaTypedFusedMOEFP8E4M3<__nv_bfloat16>(
        input, gate, up, down, index, score, w1, output,
        batch, topk, hidden, inter, experts, swigluLimit, allowWarpSpecialization);
}

template <typename T>
static bool FastllmCudaTypedFusedMOEFP8E4M3Block128(
        const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up, fastllm::Data &down,
        const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &w1, fastllm::Data &output,
        int batch, int topk, int hidden, int inter, int experts, float swigluLimit) {
    if (batch <= 0 || topk <= 0 || hidden <= 0 || inter <= 0 || experts <= 0 ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        index.dataDevice != fastllm::DataDevice::CUDA || index.dataType != fastllm::DataType::INT32 ||
        score.dataDevice != fastllm::DataDevice::CUDA || score.dataType != fastllm::DataType::FLOAT32 ||
        gate.dataDevice != fastllm::DataDevice::CUDA ||
        up.dataDevice != fastllm::DataDevice::CUDA ||
        down.dataDevice != fastllm::DataDevice::CUDA ||
        gate.dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 ||
        up.dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 ||
        down.dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 ||
        gate.cudaData == nullptr || up.cudaData == nullptr || down.cudaData == nullptr ||
        index.cudaData == nullptr || score.cudaData == nullptr) {
        return false;
    }
    if (gate.dims.size() != 3 || up.dims.size() != 3 || down.dims.size() != 3 ||
        gate.dims[0] != experts || gate.dims[1] != inter || gate.dims[2] != hidden ||
        up.dims[0] != experts || up.dims[1] != inter || up.dims[2] != hidden ||
        down.dims[0] != experts || down.dims[1] != hidden || down.dims[2] != inter) {
        return false;
    }

    w1.dataDevice = input.dataDevice;
    w1.dataDeviceIds = input.dataDeviceIds;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = FastllmMoeFp8Traits<T>::dataType;
    w1.Resize({batch * topk, inter});
    output.dataType = FastllmMoeFp8Traits<T>::dataType;
    output.Resize(input.dims);
    w1.Allocate(false);
    output.Allocate(false);

    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaW1 = (T*)FastllmCudaPrepareOutput(w1);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);
    const int32_t *cudaIndex = (const int32_t*)index.cudaData;
    const float *cudaScore = (const float*)score.cudaData;
    const uint8_t *cudaGate = (const uint8_t*)gate.cudaData;
    const uint8_t *cudaUp = (const uint8_t*)up.cudaData;
    const uint8_t *cudaDown = (const uint8_t*)down.cudaData;

    int gatePerRow = hidden + ((hidden - 1) / 128 + 1) * (int)sizeof(float);
    int downPerRow = inter + ((inter - 1) / 128 + 1) * (int)sizeof(float);
    LaunchFastllmGemmTypedFP8E4M3Block128FusedTopKSwiglu(
        cudaInput, cudaIndex, cudaGate, cudaUp, cudaW1, batch, topk, hidden, inter, experts, gatePerRow, swigluLimit);
    LaunchFastllmGemmTypedFP8E4M3Block128FusedTopKDownReduce(
        cudaW1, cudaIndex, cudaDown, cudaOutput, cudaScore, batch, topk, inter, hidden, experts, downPerRow);

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfFusedMOEFP8E4M3Block128(
        const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up,
        fastllm::Data &down, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &w1, fastllm::Data &output,
        int batch, int topk, int hidden, int inter, int experts, float swigluLimit) {
    return FastllmCudaTypedFusedMOEFP8E4M3Block128<half>(
        input, gate, up, down, index, score, w1, output, batch, topk, hidden, inter, experts, swigluLimit);
}

bool FastllmCudaBFloat16FusedMOEFP8E4M3Block128(
        const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up,
        fastllm::Data &down, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &w1, fastllm::Data &output,
        int batch, int topk, int hidden, int inter, int experts, float swigluLimit) {
    return FastllmCudaTypedFusedMOEFP8E4M3Block128<__nv_bfloat16>(
        input, gate, up, down, index, score, w1, output, batch, topk, hidden, inter, experts, swigluLimit);
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

template <typename T>
static bool FastllmCudaTypedMergeMOENVFP4Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                                bool scoresOnCuda, int topk, int hidden, int inter) {
    if (topk <= 0 || hidden <= 0 || inter <= 0 || scores == nullptr ||
        input.dataType != FastllmMoeFp8Traits<T>::dataType || input.dataDevice != fastllm::DataDevice::CUDA ||
        gateups == nullptr || downs == nullptr || gateups[0] == nullptr || downs[0] == nullptr ||
        !FastllmMoeNVFP4IsWeightType(gateups[0]->dataType) || gateups[0]->dataType != downs[0]->dataType ||
        gateups[0]->dataType == fastllm::DataType::NVFP4) {
        return false;
    }

    fastllm::DataType weightType = gateups[0]->dataType;
    bool scaleE8M0 = FastllmMoeNVFP4ScaleE8M0(weightType);
    int gatePerRow = (int)FastllmMoeNVFP4Block16BytesPerRow(hidden, scaleE8M0);
    int downPerRow = (int)FastllmMoeNVFP4Block16BytesPerRow(inter, scaleE8M0);

    std::vector<uint8_t*> hGateWeights(topk), hDownWeights(topk);
    for (int i = 0; i < topk; i++) {
        fastllm::Data *gateup = gateups[i];
        fastllm::Data *down = downs[i];
        if (gateup == nullptr || down == nullptr ||
            gateup->dataType != weightType || down->dataType != weightType ||
            gateup->dims.size() != 2 || down->dims.size() != 2 ||
            gateup->dims[1] != hidden || gateup->dims[0] != inter * 2 ||
            down->dims[1] != inter || down->dims[0] != hidden ||
            gateup->cudaData == nullptr || down->cudaData == nullptr) {
            return false;
        }
        hGateWeights[i] = (uint8_t*)gateup->cudaData;
        hDownWeights[i] = (uint8_t*)down->cudaData;
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

    FastllmMoeNVFP4Batch1Scratch &scratch = FastllmGetMoeNVFP4Batch1Scratch(topk);
    size_t ptrBytes = (size_t)topk * sizeof(void*);

    cudaError_t state = cudaSuccess;
    state = cudaMemcpyAsync(scratch.gateWeights, hGateWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when moving NVFP4 MoE gate pointers to device!", state);
    state = cudaMemcpyAsync(scratch.downWeights, hDownWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when moving NVFP4 MoE down pointers to device!", state);

    float *cudaScores = scratch.scores;
    if (scoresOnCuda) {
        cudaScores = (float*)scores;
    } else {
        state = cudaMemcpyAsync(scratch.scores, scores, (size_t)topk * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaErrors("Error: CUDA error when moving NVFP4 MoE scores to device!", state);
    }

    if (scaleE8M0) {
        LaunchFastllmGemmTypedNVFP4TopKSwiglu<true>(cudaInput, scratch.gateWeights, cudaW1,
                                                    topk, hidden, inter, gatePerRow);
        LaunchFastllmGemmTypedNVFP4TopKDownReduce<true>(cudaW1, scratch.downWeights, cudaOutput, cudaScores,
                                                        topk, inter, hidden, downPerRow);
    } else {
        LaunchFastllmGemmTypedNVFP4TopKSwiglu<false>(cudaInput, scratch.gateWeights, cudaW1,
                                                     topk, hidden, inter, gatePerRow);
        LaunchFastllmGemmTypedNVFP4TopKDownReduce<false>(cudaW1, scratch.downWeights, cudaOutput, cudaScores,
                                                         topk, inter, hidden, downPerRow);
    }

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

bool FastllmCudaHalfMergeMOENVFP4Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                        fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                        bool scoresOnCuda, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4Batch1<half>(
        input, w1, output, gateups, downs, scores, scoresOnCuda, topk, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOENVFP4Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                            fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                            bool scoresOnCuda, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOENVFP4Batch1<__nv_bfloat16>(
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
