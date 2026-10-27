#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace fastllm {
namespace rmssmall {
// RMSNorm followed by a small dense projection. Normalized values are also
// published for consumers such as a second, larger projection. No model layout
// or gating convention is embedded in this kernel.
template <class T, int D, int B = 256, int R = 2>
__global__ __launch_bounds__(B) void Kernel(const T *input, const float *norm, const T *weight,
                                            const float *bias, T *normalized, T *output, int N, float eps) {
    constexpr int P = D / (B * 2), W = B / 32;
    static_assert(__is_same(T, half) || __is_same(T, __nv_bfloat16));
    static_assert(B >= 32 && B <= 1024 && B % 32 == 0);
    static_assert(D > 0 && D % (B * 2) == 0 && R > 0 && R <= B);
    int tid = threadIdx.x, lane = tid % 32, row = blockIdx.x * R;
    input += size_t(blockIdx.y) * D;
    normalized += size_t(blockIdx.y) * D;
    output += size_t(blockIdx.y) * N;
    float2 x[P], g[P], w[R][P];
    float sq = 0, dot[R] = {};
#pragma unroll
    for (int j = 0; j < P; ++j) {
        int pair = tid + j * B;
        uint32_t bits = reinterpret_cast<const uint32_t *>(input)[pair];
        if constexpr (__is_same(T, half))
            x[j] = __half22float2(*reinterpret_cast<half2 *>(&bits));
        else
            x[j] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&bits));
        g[j] = reinterpret_cast<const float2 *>(norm)[pair];
        sq = fmaf(x[j].x, x[j].x, sq);
        sq = fmaf(x[j].y, x[j].y, sq);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            uint32_t wb =
                row + r < N ? reinterpret_cast<const uint32_t *>(weight + size_t(row + r) * D)[pair] : 0;
            if constexpr (__is_same(T, half))
                w[r][j] = __half22float2(*reinterpret_cast<half2 *>(&wb));
            else
                w[r][j] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&wb));
        }
    }
#pragma unroll
    for (int d = 16; d; d >>= 1)
        sq += __shfl_down_sync(0xffffffff, sq, d);
    __shared__ float partial[R + 1][W];
    if (lane == 0)
        partial[0][tid / 32] = sq;
    __syncthreads();
    float total = 0;
#pragma unroll
    for (int i = 0; i < W; ++i)
        total += partial[0][i];
    float inv = rsqrtf(total / D + eps);
#pragma unroll
    for (int j = 0; j < P; ++j) {
        int pair = tid + j * B;
        T a = T((x[j].x * inv) * g[j].x), b = T((x[j].y * inv) * g[j].y);
        if (pair % gridDim.x == blockIdx.x) {
            normalized[pair * 2] = a;
            normalized[pair * 2 + 1] = b;
        }
        {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                dot[r] = fmaf(float(a), w[r][j].x, dot[r]);
                dot[r] = fmaf(float(b), w[r][j].y, dot[r]);
            }
        }
    }
    {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int d = 16; d; d >>= 1)
                dot[r] += __shfl_down_sync(0xffffffff, dot[r], d);
            if (lane == 0)
                partial[r + 1][tid / 32] = dot[r];
        }
        __syncthreads();
    }
    if (tid < R && row + tid < N) {
        float v = 0;
#pragma unroll
        for (int i = 0; i < W; ++i)
            v += partial[tid + 1][i];
        output[row + tid] = T(v + (bias ? bias[row + tid] : 0.f));
    }
}
} // namespace rmssmall
} // namespace fastllm
