#pragma once
#include "fastllm-cuda-launch.cuh"
#include "fastllm-cuda-ordered-reduce.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <cstdint>

namespace fastllm { namespace cuda_norm {
using cuda_launch::DependencyStart;
template<class T> __device__ T Cast(float x) { return T(x); }
template<class T> __device__ float Float(T x) { return float(x); }
template<class T> using Pair = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
template<class T> __device__ Pair<T> Cast2(float x, float y) {
    if constexpr (std::is_same_v<T, __half>) return __floats2half2_rn(x, y);
    else return __floats2bfloat162_rn(x, y);
}
__device__ inline float2 Float2(__half2 x) { return __half22float2(x); }
__device__ inline float2 Float2(__nv_bfloat162 x) { return __bfloat1622float2(x); }
using fastllm::cuda::WarpSum;
using fastllm::cuda::WarpFirstBlockSum256;
// Odd row strides are not aligned for float4/half2/bfloat162 accesses. Scalar
// loads preserve arbitrary row alignment; the general even-width path stays
// unchanged. No tensor-core instructions are required.
template <typename T, bool Fused = false>
__global__ void RMSNorm(const T *input, const float *weight,
        T *output, int channels, float eps,
        const T *gate = nullptr, bool zeroCentered = false) {
    if constexpr (Fused) DependencyStart();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    input += (uint64_t)blockIdx.x * channels;
    output += (uint64_t)blockIdx.x * channels;
    __shared__ float partial[8], scale;
    float sum = 0.0f;
    for (int i = tid; i < channels; i += 256) {
        float v = (float)input[i]; sum += v * v;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? partial[lane] : 0.0f;
        for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) scale = rsqrtf(sum / channels + eps);
    }
    __syncthreads();
    for (int i = tid; i < channels; i += 256) {
        // The plain Data path preserves signed-zero weights as well.
        float affine = weight[i];
        if constexpr (Fused) affine += zeroCentered ? 1.f : 0.f;
        float x = (float)input[i] * scale * affine;
        if constexpr (Fused) {
            if (gate) {
                float g = (float)gate[(size_t)blockIdx.x * channels + i];
                x *= g / (1.f + expf(-g));
            }
        }
        output[i] = (T)x;
    }
}

// Preserve the original 256-thread RMS reduction and 16-bit residual boundary.
// The common hidden width keeps rounded values in registers, avoiding a
// global read-after-write dependency before the normalized output store.
template<typename T, int Width, bool RoundNormalized = false, bool PDL = true>
__global__ void AddRMSNorm(T *residual, const T *input,
        const float *weight, T *output, int width, float eps, bool zeroCentered) {
    if constexpr (PDL) DependencyStart();
    __shared__ float scratch[8];
    const int columns = Width ? Width : width;
    const size_t offset = (size_t)blockIdx.x * columns;
    float values[Width ? Width / 256 : 1], affine[Width && RoundNormalized ? Width / 256 : 1], sum = 0.f;
#pragma unroll
    for (int d = threadIdx.x, j = 0; d < columns; d += 256, ++j) {
        auto rounded = Cast<T>(Float(residual[offset + d]) +
                                    Float(input[offset + d]));
        float x = Float(rounded);
        if constexpr (Width) {
            values[j] = x;
            if constexpr (RoundNormalized) affine[j] = weight[d];
        } else residual[offset + d] = rounded;
        sum += x * x;
    }
    float inv = rsqrtf(WarpFirstBlockSum256(sum, scratch) / columns + eps);
#pragma unroll
    for (int d = threadIdx.x, j = 0; d < columns; d += 256, ++j) {
        float x;
        if constexpr (Width) {
            x = values[j];
            residual[offset + d] = Cast<T>(x);
        } else x = Float(residual[offset + d]);
        if constexpr (RoundNormalized) {
            float w = Width ? affine[j] : weight[d];
            output[offset + d] = Cast<T>(Float(Cast<T>(x * inv)) * w);
        } else output[offset + d] = Cast<T>(x * inv * (weight[d] + (zeroCentered ? 1.f : 0.f)));
    }
}
// Two adjacent columns per lane preserve the original 256-thread RMS
// reduction while using vectorized 16-bit accesses and only one CTA barrier.
template<typename T, bool Add>
__global__ void HiddenRMSNorm(const T *input, T *residual,
        const float *weight, const T *gate, T *output,
        float eps, bool zeroCentered) {
    DependencyStart();
    const int tid = threadIdx.x, lane = tid % 32;
    const size_t offset = (size_t)blockIdx.x * 2048;
    float2 values[8];
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = tid * 2 + j * 256;
        float2 x = Float2(*reinterpret_cast<const Pair<T> *>(input + offset + d));
        if constexpr (Add) {
            float2 h = Float2(*reinterpret_cast<const Pair<T> *>(residual + offset + d));
            x = Float2(Cast2<T>(x.x + h.x, x.y + h.y));
        }
        values[j] = x;
        a += x.x * x.x; b += x.y * x.y;
    }
#pragma unroll
    for (int step = 8; step; step >>= 1) {
        a += __shfl_down_sync(0xffffffff, a, step, 16);
        b += __shfl_down_sync(0xffffffff, b, step, 16);
    }
    __shared__ float partial[8];
    if (lane % 16 == 0) partial[tid / 16] = a + b;
    __syncthreads();
    const float inv = rsqrtf(WarpSum(lane < 8 ? partial[lane] : 0.f) / 2048.f + eps);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = tid * 2 + j * 256;
        const float2 x = values[j];
        if constexpr (Add)
            *reinterpret_cast<Pair<T> *>(residual + offset + d) = Cast2<T>(x.x, x.y);
        float u = x.x * inv * (weight[d] + (zeroCentered ? 1.f : 0.f));
        float v = x.y * inv * (weight[d + 1] + (zeroCentered ? 1.f : 0.f));
        if (gate) {
            float2 z = Float2(*reinterpret_cast<const Pair<T> *>(gate + offset + d));
            u *= z.x / (1.f + expf(-z.x)); v *= z.y / (1.f + expf(-z.y));
        }
        *reinterpret_cast<Pair<T> *>(output + offset + d) = Cast2<T>(u, v);
    }
}
// Small normalization widths use one warp per row for bulk prefill. Each
// 32-element partial and its reduction order match RMSNorm<T>'s CTA tree.
template<typename T, int Width>
__global__ void WarpRMSNorm(const T *input, const float *weight,
        const T *gate, T *output, int rows, float eps, bool zeroCentered) {
    DependencyStart();
    const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    constexpr int Parts = Width / 32;
    const size_t offset = (size_t)row * Width;
    float x[Parts], sums[Parts];
#pragma unroll
    for (int j = 0; j < Parts; ++j) {
        x[j] = Float(input[offset + j * 32 + lane]);
        sums[j] = WarpSum(x[j] * x[j]);
    }
#pragma unroll
    for (int step = Parts / 2; step; step /= 2)
        for (int j = 0; j < step; ++j) sums[j] += sums[j + step];
    float inv = rsqrtf(sums[0] / Width + eps);
#pragma unroll
    for (int j = 0; j < Parts; ++j) {
        float y = x[j] * inv * (weight[j * 32 + lane] + (zeroCentered ? 1.f : 0.f));
        if (gate) {
            float z = Float(gate[offset + j * 32 + lane]);
            y *= z / (1.f + expf(-z));
        }
        output[offset + j * 32 + lane] = Cast<T>(y);
    }
}
}} // namespace fastllm::cuda_norm
