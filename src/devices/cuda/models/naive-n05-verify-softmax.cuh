#pragma once
#include <cuda_bf16.h>
namespace naive_dsa_decode {
__device__ __forceinline__ float Round(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float WarpSum(float x) {
    for (int s = 16; s; s >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, s);
    return x;
}
// Preserve the original 256-lane reduction tree and BF16 probability rounding.
__device__ __forceinline__ void Normalize(const float *input, float *output, const float *sink, int head,
                                          int count, float *scratch, float &maximum, float &denominator,
                                          int t) {
    float logits[8];
    float mx = sink ? sink[head] : -INFINITY;
    if (t < 256) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            logits[i] = t + i * 256 < count ? input[t + i * 256] : -INFINITY;
            mx = fmaxf(mx, logits[i]);
        }
        scratch[t] = mx;
    }
    __syncthreads();
    if (t < 32) {
        float a = fmaxf(scratch[t], scratch[t + 128]), b = fmaxf(scratch[t + 64], scratch[t + 192]);
        float c = fmaxf(scratch[t + 32], scratch[t + 160]), d = fmaxf(scratch[t + 96], scratch[t + 224]);
        float top = fmaxf(fmaxf(a, b), fmaxf(c, d));
        for (int s = 16; s; s >>= 1)
            top = fmaxf(top, __shfl_down_sync(0xffffffffu, top, s));
        if (!t)
            maximum = top;
    }
    __syncthreads();
    if (t < 256) {
        float sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            logits[i] = t + i * 256 < count ? expf(logits[i] - maximum) : 0;
            sum += logits[i];
        }
        if (t == 0 && sink)
            sum += expf(sink[head] - maximum);
        scratch[t] = sum;
    }
    __syncthreads();
    if (t < 32) {
        float a = scratch[t] + scratch[t + 128], b = scratch[t + 64] + scratch[t + 192],
              c = scratch[t + 32] + scratch[t + 160], d = scratch[t + 96] + scratch[t + 224];
        float total = WarpSum((a + b) + (c + d));
        if (!t)
            denominator = total;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (t < 256 && t + i * 256 < count)
            output[t + i * 256] = denominator > 0 ? Round(logits[i] / denominator) : 0;
}
} // namespace naive_dsa_decode
