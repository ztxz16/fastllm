#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace fastllm {
namespace normdecode {
// Single-row decode specialization. FP32 norm weights remain FP32; only the
// reduction order changes. Cache every input/weight before the one block barrier,
// so input == output is safe. All warps then read the small shared partial sum
// array themselves, avoiding a second block barrier and a warp-0 bottleneck.
template <class T, int D = 5120, int B = 512>
__global__ __launch_bounds__(B) void Kernel(const T *input, const float *weight, T *output, float eps) {
    static_assert(__is_same(T, half) || __is_same(T, __nv_bfloat16));
    static_assert(D % (2 * B) == 0 && B % 32 == 0);
    constexpr int Pairs = D / (2 * B), Warps = B / 32;
    int tid = threadIdx.x;
    uint32_t values[Pairs];
    float2 weights[Pairs];
    float sum0 = 0, sum1 = 0;
#pragma unroll
    for (int j = 0; j < Pairs; ++j) {
        int pair = tid + j * B;
        values[j] = reinterpret_cast<const uint32_t *>(input)[pair];
        weights[j] = reinterpret_cast<const float2 *>(weight)[pair];
        float2 value;
        if constexpr (__is_same(T, half))
            value = __half22float2(*reinterpret_cast<half2 *>(&values[j]));
        else
            value = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&values[j]));
        sum0 = fmaf(value.x, value.x, sum0);
        sum1 = fmaf(value.y, value.y, sum1);
    }
    float sum = sum0 + sum1;
#pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    __shared__ float partial[Warps];
    if ((tid & 31) == 0)
        partial[tid / 32] = sum;
    __syncthreads();
    float total = 0;
#pragma unroll
    for (int w = 0; w < Warps; ++w)
        total += partial[w];
    float scale = rsqrtf(total / D + eps);
#pragma unroll
    for (int j = 0; j < Pairs; ++j) {
        float2 value;
        if constexpr (__is_same(T, half))
            value = __half22float2(*reinterpret_cast<half2 *>(&values[j]));
        else
            value = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&values[j]));
        float a = (value.x * scale) * weights[j].x;
        float b = (value.y * scale) * weights[j].y;
        if constexpr (__is_same(T, half))
            reinterpret_cast<half2 *>(output)[tid + j * B] = __floats2half2_rn(a, b);
        else
            reinterpret_cast<__nv_bfloat162 *>(output)[tid + j * B] = __floats2bfloat162_rn(a, b);
    }
}
} // namespace normdecode
} // namespace fastllm
