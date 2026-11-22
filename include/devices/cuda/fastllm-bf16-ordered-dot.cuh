#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace fastllm { namespace cuda {
union alignas(16) Bf16OrderedDotPacket { uint4 bits; __nv_bfloat16 values[8]; };

// Same eight products and compensated 256-lane tree as the existing __nv_bfloat16
// GEMV. The logical lanes are held in registers by one warp per output.
template<int K>
__device__ __forceinline__ __nv_bfloat16 Bf16OrderedWarpDot(const __nv_bfloat16 *x, const __nv_bfloat16 *w) {
    int lane = threadIdx.x % 32;
    float values[8] = {}, correction[8] = {};
#pragma unroll
    for (int part = 0; part < 8; ++part) {
#pragma unroll
        for (int col = (lane + part * 32) * 8; col < K; col += 2048) {
            Bf16OrderedDotPacket a, b;
            a.bits = *reinterpret_cast<const uint4 *>(x + col);
            b.bits = *reinterpret_cast<const uint4 *>(w + col);
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                sum += float(a.values[i]) * float(b.values[i]);
            values[part] += sum;
        }
    }
#pragma unroll
    for (int stride = 4; stride; stride >>= 1) {
#pragma unroll
        for (int p = 0; p < stride; ++p) {
            float other = values[p + stride] - correction[p];
            float sum = values[p] + other;
            correction[p] = (sum - values[p]) - other;
            values[p] = sum;
        }
    }
    float value = values[0], diff = correction[0];
#pragma unroll
    for (int stride = 16; stride; stride >>= 1) {
        float peer = __shfl_down_sync(0xffffffffu, value, stride);
        if (lane < stride) {
            float other = peer - diff;
            float sum = value + other;
            diff = (sum - value) - other;
            value = sum;
        }
    }
    return __float2bfloat16_rn(value);
}

}} // namespace fastllm::cuda
