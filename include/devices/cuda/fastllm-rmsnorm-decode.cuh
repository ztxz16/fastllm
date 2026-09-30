#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>

namespace fastllm {
namespace normdecode {
// Keep the generic 1024-thread kernel's reduction tree while using 512 physical
// threads. Cache inputs and weights before the barrier so input == output is
// safe. Each warp repeats the final reduction, avoiding a second block barrier.
template <class T, int D = 5120, int B = 512>
__global__ __launch_bounds__(B) void Kernel(const T *input, const float *weight, T *output, float eps) {
    static_assert(__is_same(T, half) || __is_same(T, __nv_bfloat16));
    static_assert(D == 5120 && B == 512);
    constexpr int LegacyThreads = 1024, Groups = LegacyThreads / B;
    constexpr int Pairs = (D / 2 + LegacyThreads - 1) / LegacyThreads;
    input += size_t(blockIdx.x) * D;
    output += size_t(blockIdx.x) * D;
    int tid = threadIdx.x;
    uint32_t values[Groups][Pairs];
    float2 weights[Groups][Pairs];
    float sums[Groups] = {};
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int j = 0; j < Pairs; ++j) {
            int pair = tid + g * B + j * LegacyThreads;
            if (pair < D / 2) {
                values[g][j] = reinterpret_cast<const uint32_t *>(input)[pair];
                weights[g][j] = reinterpret_cast<const float2 *>(weight)[pair];
                float2 value;
                if constexpr (__is_same(T, half))
                    value = __half22float2(*reinterpret_cast<half2 *>(&values[g][j]));
                else
                    value = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&values[g][j]));
                sums[g] += value.x * value.x + value.y * value.y;
            }
        }
    }
#pragma unroll
    for (int delta = 16; delta; delta >>= 1) {
#pragma unroll
        for (int g = 0; g < Groups; ++g)
            sums[g] += __shfl_down_sync(0xffffffff, sums[g], delta);
    }
    __shared__ float partial[LegacyThreads / 32];
    if ((tid & 31) == 0) {
#pragma unroll
        for (int g = 0; g < Groups; ++g)
            partial[g * (B / 32) + tid / 32] = sums[g];
    }
    __syncthreads();
    float total = partial[tid & 31];
#pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        total += __shfl_down_sync(0xffffffff, total, delta);
    total = __shfl_sync(0xffffffff, total, 0);
    // The generic kernel divides by a runtime channel count. Preserve that
    // rounding instead of contracting a constant reciprocal with epsilon.
    float scale = rsqrtf(__fdiv_rn(total, float(D)) + eps);
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int j = 0; j < Pairs; ++j) {
            int pair = tid + g * B + j * LegacyThreads;
            if (pair < D / 2) {
                float2 value;
                if constexpr (__is_same(T, half))
                    value = __half22float2(*reinterpret_cast<half2 *>(&values[g][j]));
                else
                    value = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&values[g][j]));
                float a = (value.x * scale) * weights[g][j].x;
                float b = (value.y * scale) * weights[g][j].y;
                if constexpr (__is_same(T, half))
                    reinterpret_cast<half2 *>(output)[pair] = __floats2half2_rn(a, b);
                else
                    reinterpret_cast<__nv_bfloat162 *>(output)[pair] = __floats2bfloat162_rn(a, b);
            }
        }
    }
}
} // namespace normdecode
} // namespace fastllm
