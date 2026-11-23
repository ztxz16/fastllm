#pragma once
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include "devices/cuda/fastllm-mxfp8-activation.cuh"

namespace fastllm_mxfp8_projection {
struct WeightFragment {
    uint2 low, high;
    float lowScale, highScale;
};
__device__ __forceinline__ WeightFragment LoadWeights(const void *weights, const float *scales,
                                                      int stride, int row0, int row1, int k) {
    const auto *w = static_cast<const uint8_t *>(weights);
    const uint2 a = *reinterpret_cast<const uint2 *>(w + size_t(row0) * stride + k);
    const uint2 b = *reinterpret_cast<const uint2 *>(w + size_t(row1) * stride + k);
    const float s0 = scales[(size_t(row0) * stride + k) / 32];
    const float s1 = scales[(size_t(row1) * stride + k) / 32];
    return {a, b, s0, s1};
}
struct ActivationFragment {
    uint2 values;
    float scale;
};
__device__ __forceinline__ ActivationFragment Quantize(uint4 activation) {
    uint32_t exponent;
    const uint2 packed = fastllm_mxfp8::PackActivation(
        reinterpret_cast<const __nv_bfloat162 *>(&activation), exponent, 0xffffffffu);
    return {packed, __uint_as_float(exponent << 23)};
}
__device__ __forceinline__ void Accumulate(float (&acc)[4], WeightFragment weight,
                                           uint2 q, float x0, float x1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const uint2 a = weight.low, b = weight.high;
    const float s0 = weight.lowScale, s1 = weight.highScale;
    float d[4]; float zero = 0.f;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};"
        : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        : "r"(a.x),"r"(b.x),"r"(a.y),"r"(b.y),"r"(q.x),"r"(q.y),"f"(zero));
    acc[0]=fmaf(d[0],s0*x0,acc[0]); acc[1]=fmaf(d[1],s0*x1,acc[1]);
    acc[2]=fmaf(d[2],s1*x0,acc[2]); acc[3]=fmaf(d[3],s1*x1,acc[3]);
#endif
}
__device__ __forceinline__ void Step(float (&acc)[4], WeightFragment weight, uint4 activation) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const int quad = threadIdx.x % 4;
    const auto x = Quantize(activation);
    Accumulate(acc, weight, x.values, __shfl_sync(0xffffffffu, x.scale, quad * 8),
               __shfl_sync(0xffffffffu, x.scale, quad * 8 + 4));
#endif
}
__device__ __forceinline__ void Step(float (&acc)[4], const void *weights, const float *scales,
                                    int stride, int row0, int row1, int k, uint4 activation) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    Step(acc, LoadWeights(weights, scales, stride, row0, row1, k), activation);
#endif
}
} // namespace fastllm_mxfp8_projection
