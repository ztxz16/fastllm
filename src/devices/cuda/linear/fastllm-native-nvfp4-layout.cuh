#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
namespace fastllm_native_layout {
__host__ __device__ inline size_t ScaleIndex(int row, int group, int groups) {
    return (size_t(row / 128) * (groups / 4) + group / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 +
           group % 4;
}
__device__ inline float Scale(uint8_t b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return float(v);
}
__device__ inline float2 Pair(uint8_t b) {
    __nv_fp4x2_e2m1 v;
    v.__x = b;
    return static_cast<float2>(v);
}
static __global__ void Pack(const uint8_t *raw, uint8_t *codes, uint8_t *scales, int N, int K, float global,
                            int *bad) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, groups = K / 16;
    if (i >= size_t(N) * groups)
        return;
    auto src = raw + i * 12;
    float s = *reinterpret_cast<const float *>(src + 8);
    __nv_fp8_e4m3 q(s / global);
    float restored = float(q) * global;
    if (!isfinite(s) || s < 0 || !isfinite(restored) || fabsf(restored - s) > fmaxf(1e-12f, fabsf(s) * 1e-5f))
        atomicExch(bad, 1);
    *reinterpret_cast<uint2 *>(codes + i * 8) =
        make_uint2(*reinterpret_cast<const uint32_t *>(src), *reinterpret_cast<const uint32_t *>(src + 4));
    scales[ScaleIndex(i / groups, i % groups, groups)] = q.__x;
}
static __global__ void Restore(const uint8_t *codes, const uint8_t *scales, const float *global, uint8_t *raw,
                               int N, int K) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, groups = K / 16;
    if (i >= size_t(N) * groups)
        return;
    auto dst = raw + i * 12;
    uint2 q = *reinterpret_cast<const uint2 *>(codes + i * 8);
    // Interleaved raw blocks are only four-byte aligned.
    *reinterpret_cast<uint32_t *>(dst) = q.x;
    *reinterpret_cast<uint32_t *>(dst + 4) = q.y;
    *reinterpret_cast<float *>(dst + 8) =
        Scale(scales[ScaleIndex(i / groups, i % groups, groups)]) * (*global / 128.f);
}
template <int Mode, int Warps = 4, int FixedK = 0, int Chains = 1>
static __global__ void Gemv(const half *__restrict__ x, const uint8_t *__restrict__ w,
                            const uint8_t *__restrict__ scales, const float *__restrict__ global,
                            const half *__restrict__ bias, half *__restrict__ y, int M, int N, int K) {
    if constexpr (FixedK)
        K = FixedK;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, width = Mode == 1 ? N / 2 : N;
    int row = (blockIdx.x / (128 / Warps)) * 128 + (blockIdx.x % (128 / Warps)) * (Warps / 4) +
              (warp % 4) * 32 + warp / 4;
    if (row >= width)
        return;
    int token = blockIdx.y;
    float sum = 0, up = 0;
#pragma unroll 2
    for (int group = lane; group < K / 16; group += 32) {
        uint2 code = *reinterpret_cast<const uint2 *>(w + size_t(row) * K / 2 + group * 8);
        uint2 codeUp{};
        if constexpr (Mode == 1)
            codeUp = *reinterpret_cast<const uint2 *>(w + size_t(row + width) * K / 2 + group * 8);
        float dot[Chains] = {}, dotUp[Chains] = {};
#pragma unroll
        for (int p = 0; p < 8; p++) {
            float2 a =
                __half22float2(*reinterpret_cast<const half2 *>(x + size_t(token) * K + group * 16 + p * 2));
            float2 b = Pair(uint8_t((p < 4 ? code.x : code.y) >> ((p % 4) * 8)));
            dot[(p * 2) % Chains] = fmaf(a.x, b.x, dot[(p * 2) % Chains]);
            dot[(p * 2 + 1) % Chains] = fmaf(a.y, b.y, dot[(p * 2 + 1) % Chains]);
            if constexpr (Mode == 1) {
                float2 c = Pair(uint8_t((p < 4 ? codeUp.x : codeUp.y) >> ((p % 4) * 8)));
                dotUp[(p * 2) % Chains] = fmaf(a.x, c.x, dotUp[(p * 2) % Chains]);
                dotUp[(p * 2 + 1) % Chains] = fmaf(a.y, c.y, dotUp[(p * 2 + 1) % Chains]);
            }
        }
        float total = 0, totalUp = 0;
#pragma unroll
        for (int c = 0; c < Chains; c++) {
            total += dot[c];
            if constexpr (Mode == 1)
                totalUp += dotUp[c];
        }
        sum = fmaf(total, Scale(scales[ScaleIndex(row, group, K / 16)]), sum);
        if constexpr (Mode == 1)
            up = fmaf(totalUp, Scale(scales[ScaleIndex(row + width, group, K / 16)]), up);
    }
#pragma unroll
    for (int d = 16; d; d /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, d);
        if constexpr (Mode == 1)
            up += __shfl_down_sync(0xffffffff, up, d);
    }
    if (lane == 0) {
        half v = __float2half(sum * (*global / 128.f));
        if (bias)
            v = __hadd(v, bias[row]);
        if constexpr (Mode == 1) {
            half u = __float2half(up * (*global / 128.f));
            if (bias)
                u = __hadd(u, bias[row + width]);
            v = __hmul(__hdiv(v, __hadd(__float2half(1.f), hexp(__hneg(v)))), u);
        }
        if constexpr (Mode == 2)
            v = __hadd(v, y[size_t(token) * width + row]);
        y[size_t(token) * width + row] = v;
    }
}
static __global__ void Dequant(const uint8_t *w, const uint8_t *scales, const float *global, half *out,
                               int first, int rows, int K) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * K / 2)
        return;
    int row = first + i / (K / 2), pair = i % (K / 2), group = pair / 8;
    float2 v = Pair(w[size_t(row) * K / 2 + pair]);
    float s = Scale(scales[ScaleIndex(row, group, K / 16)]) * (*global / 128.f);
    reinterpret_cast<half2 *>(out)[i] = __floats2half2_rn(v.x * s, v.y * s);
}
} // namespace fastllm_native_layout
