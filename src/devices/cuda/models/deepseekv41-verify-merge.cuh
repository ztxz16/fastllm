#pragma once
#include <cuda_bf16.h>
#include "devices/cuda/fastllm-v41-full-decode.h"

namespace fastllm_v41_verify {
__device__ __forceinline__ float VerifyExp2(float x) {
    // FlashInfer uses --use_fast_math. Keep the same exponent in both adapters.
    float value;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(value) : "f"(x));
    return value;
}

// FlashInfer compiles its MMA with FTZ. The verifier's BF16 current prefix
// also accepts subnormal inputs; preserve the original FP32 accumulator so
// that rounding a negative tiny value to FP8 retains its sign.
template<bool PreserveDenormals>
__device__ __forceinline__ float VerifyFma(float a, float b, float c) {
    if constexpr (PreserveDenormals) {
        float result;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
        return result;
    }
    return fmaf(a, b, c);
}

// Keep the numerical body shared by the portable two-launch implementation
// and the two four-warp head groups inside the cooperative verifier.
template<bool HalfCta>
__device__ __forceinline__ void MergeBarrier() {
    if constexpr (HalfCta) {
        if (threadIdx.x < 128) asm volatile("bar.sync 1, 128;" ::: "memory");
        else asm volatile("bar.sync 2, 128;" ::: "memory");
    } else __syncthreads();
}

template<bool HalfCta>
__device__ __forceinline__ uint2 MergeHead(const __nv_bfloat16 *q, const __nv_bfloat16 *chunk,
        const __nv_bfloat16 *partial, const float *lse, float sink, float scale, int splits,
        int token, float *prefixWeights, float *splitWeights) {
    const int t = threadIdx.x & 127, lane = t & 31, warp = t / 32;
    // The model has a 128-token sliding window, independent of batch/chunk size.
    const int first = max(0, token - 127), count = token - first + 1;
    for (int local = warp; local < count; local += 4) {
        float dot = 0.f;
#pragma unroll
        for (int d = lane; d < 512; d += 32)
            dot = VerifyFma<HalfCta>(float(q[d]), float(chunk[size_t(first + local) * 512 + d]), dot);
        for (int s = 16; s; s >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, s);
        if (!lane)
            prefixWeights[local] = dot * scale * 1.4426950408889634f;
    }
    MergeBarrier<HalfCta>();
    if (t < 32) {
        const float sinkScore = sink * 1.4426950408889634f;
        float maximum = sinkScore;
        for (int row = t; row < count; row += 32)
            maximum = fmaxf(maximum, prefixWeights[row]);
        for (int i = t; i < splits; i += 32)
            maximum = fmaxf(maximum, lse[i]);
        for (int s = 16; s; s >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
        float denominator = 0.f;
        for (int row = t; row < count; row += 32) {
            const float w = VerifyExp2(prefixWeights[row] - maximum);
            prefixWeights[row] = w;
            denominator += w;
        }
        for (int i = t; i < splits; i += 32) {
            const float w = lse[i] > -1e29f ? VerifyExp2(lse[i] - maximum) : 0.f;
            splitWeights[i] = w;
            denominator += w;
        }
        for (int s = 16; s; s >>= 1)
            denominator += __shfl_xor_sync(0xffffffff, denominator, s);
        const float inv = __fdividef(1.f, denominator + VerifyExp2(sinkScore - maximum));
        for (int row = t; row < count; row += 32)
            prefixWeights[row] *= inv;
        for (int i = t; i < splits; i += 32)
            splitWeights[i] *= inv;
    }
    MergeBarrier<HalfCta>();
    float acc[4] = {};
    for (int row = 0; row < count; ++row) {
#pragma unroll
        for (int d = 0; d < 4; ++d)
            acc[d] = VerifyFma<HalfCta>(prefixWeights[row], float(chunk[size_t(first + row) * 512 + t * 4 + d]), acc[d]);
    }
    for (int i = 0; i < splits; ++i) {
        if (splitWeights[i] == 0.f)
            continue;
#pragma unroll
        for (int d = 0; d < 4; ++d)
            acc[d] = VerifyFma<HalfCta>(splitWeights[i], float(partial[size_t(i) * 512 + t * 4 + d]), acc[d]);
    }
    uint2 result;
    auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&result);
    pairs[0] = __floats2bfloat162_rn(acc[0], acc[1]);
    pairs[1] = __floats2bfloat162_rn(acc[2], acc[3]);
    return result;
}

__device__ __forceinline__ uint2 MergeValues(const __nv_bfloat16 *q, const __nv_bfloat16 *chunk,
        const __nv_bfloat16 *partial, const float *lse, const float *sink, float scale,
        int heads, int scratchHeads, int splits, int length,
        const FastllmV41FullDecodeRequest *requests, int rowOffset) {
    const int h = blockIdx.x, row = rowOffset + blockIdx.y;
    const int token = requests ? row - requests[row].chunkBegin : row % length;
    const size_t split = (size_t(row) * scratchHeads + h) * splits;
    __shared__ float prefixWeights[128], splitWeights[18];
    return MergeHead<false>(q + (size_t(row) * heads + h) * 512, chunk + size_t(row - token) * 512,
        partial + split * 512, lse + split, sink[h], scale, splits, token, prefixWeights, splitWeights);
}
} // namespace fastllm_v41_verify
