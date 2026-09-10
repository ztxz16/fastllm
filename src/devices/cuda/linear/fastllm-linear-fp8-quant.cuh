#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace fastllm_cuda_cutlass_fp8_sm89 {

template <typename T>
__device__ __forceinline__ float FastllmSm89ToFloat(T value);

template <>
__device__ __forceinline__ float FastllmSm89ToFloat(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float FastllmSm89ToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T, bool Vectorized = false>
__global__ void __launch_bounds__(256) FastllmSm89QuantPerRowKernel(
    const T *__restrict__ input, uint8_t *__restrict__ quant,
    float *__restrict__ scales, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    constexpr int kWarps = 8;
    __shared__ float warpMax[kWarps];
    __shared__ float rowScale;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const T *rowInput = input + (size_t)row * cols;
    uint8_t *rowQuant = quant + (size_t)row * cols;

    float maxAbs = 0.0f;
    if constexpr (Vectorized) {
        for (int group = threadIdx.x; group < cols / 4; group += blockDim.x) {
            union { uint2 packed; T values[4]; } value;
            value.packed = reinterpret_cast<const uint2 *>(rowInput)[group];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                maxAbs = fmaxf(maxAbs, fabsf(FastllmSm89ToFloat(value.values[i])));
            }
        }
    } else {
        for (int col = threadIdx.x; col < cols; col += blockDim.x) {
            maxAbs = fmaxf(maxAbs, fabsf(FastllmSm89ToFloat(rowInput[col])));
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maxAbs = fmaxf(maxAbs,
                       __shfl_down_sync(0xffffffffu, maxAbs, offset));
    }
    if (lane == 0) {
        warpMax[warp] = maxAbs;
    }
    __syncthreads();
    if (warp == 0) {
        maxAbs = lane < kWarps ? warpMax[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maxAbs = fmaxf(maxAbs,
                           __shfl_down_sync(0xffffffffu, maxAbs, offset));
        }
        if (lane == 0) {
            rowScale = maxAbs > 0.0f ? maxAbs * (1.0f / 448.0f) : 1.0f;
            scales[row] = rowScale;
        }
    }
    __syncthreads();
    float invScale = 1.0f / rowScale;
    if constexpr (Vectorized) {
        for (int group = threadIdx.x; group < cols / 4; group += blockDim.x) {
            union { uint2 packed; T values[4]; } value;
            value.packed = reinterpret_cast<const uint2 *>(rowInput)[group];
            float converted[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                converted[i] = FastllmSm89ToFloat(value.values[i]) * invScale;
            }
            uint32_t low = __nv_cvt_float2_to_fp8x2(
                make_float2(converted[0], converted[1]), __NV_SATFINITE, __NV_E4M3);
            uint32_t high = __nv_cvt_float2_to_fp8x2(
                make_float2(converted[2], converted[3]), __NV_SATFINITE, __NV_E4M3);
            reinterpret_cast<uint32_t *>(rowQuant)[group] = low | (high << 16);
        }
    } else {
        for (int col = threadIdx.x; col < cols; col += blockDim.x) {
            float value = FastllmSm89ToFloat(rowInput[col]) * invScale;
            rowQuant[col] = (uint8_t)__nv_cvt_float_to_fp8(
                value, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

template <typename T>
inline void FastllmSm89LaunchQuantPerRow(
    const T *input, uint8_t *quant, float *scales,
    int rows, int cols, cudaStream_t stream) {
    // Four adjacent elements per thread reduce load/store and conversion
    // instructions without changing the scale or FP8 rounding. Keep enough
    // work for all 256 threads; narrow or unaligned rows use the scalar path.
    bool vectorized = cols >= 256 * 4 && cols % 4 == 0 &&
                      (reinterpret_cast<uintptr_t>(input) % alignof(uint2)) == 0 &&
                      (reinterpret_cast<uintptr_t>(quant) % alignof(uint32_t)) == 0;
    if (vectorized) {
        FastllmSm89QuantPerRowKernel<T, true><<<rows, 256, 0, stream>>>(
            input, quant, scales, rows, cols);
    } else {
        FastllmSm89QuantPerRowKernel<T, false><<<rows, 256, 0, stream>>>(
            input, quant, scales, rows, cols);
    }
}

} // namespace fastllm_cuda_cutlass_fp8_sm89
