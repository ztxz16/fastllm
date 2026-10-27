#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace fastllm {
namespace rmssmall {
// Match normdecode::Kernel<half,5120,512> and the FP16 x BF16 GEMV's
// accumulation tree. Each CTA produces two rows and shares their norm; CTA zero publishes
// the rounded FP16 activation for the large GGUF projections that follow.
// The B8 fallback also rounds the dot input/output through BF16; retain that
// boundary unless the caller requests the existing exact-row GEMV route.
template <bool Bf16Dot = false>
__global__
__launch_bounds__(512) void Mixed5120(const half *__restrict__ input, const float *__restrict__ norm,
                                      const __nv_bfloat16 *__restrict__ weight, half *__restrict__ normalized,
                                      half *__restrict__ output, int outputs, float eps) {
    constexpr int D = 5120, B = 512, DotThreads = 256, Legacy = 1024, Groups = Legacy / B;
    const int tid = threadIdx.x, lane = tid & 31;
    input += size_t(blockIdx.y) * D;
    normalized += size_t(blockIdx.y) * D;
    const int localRow = tid / DotThreads, dotTid = tid % DotThreads;
    const int row = blockIdx.x * 2 + localRow;
    weight += size_t(row) * D;
    float sums[Groups] = {};
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int pair = tid + g * B + j * Legacy;
            if (pair < D / 2) {
                const float2 x = __half22float2(reinterpret_cast<const half2 *>(input)[pair]);
                sums[g] += x.x * x.x + x.y * x.y;
            }
        }
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) {
#pragma unroll
        for (int g = 0; g < Groups; ++g)
            sums[g] += __shfl_down_sync(0xffffffff, sums[g], d);
    }
    __shared__ float partial[32], dots[2][DotThreads];
    float *dot = dots[localRow];
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < Groups; ++g)
            partial[g * (B / 32) + tid / 32] = sums[g];
    }
    __syncthreads();
    float total = partial[lane];
#pragma unroll
    for (int d = 16; d; d >>= 1)
        total += __shfl_down_sync(0xffffffff, total, d);
    total = __shfl_sync(0xffffffff, total, 0);
    const float scale = rsqrtf(__fdiv_rn(total, float(D)) + eps);
    float value = 0.0f;
    constexpr int Width = Bf16Dot ? 8 : 4;
    union Packed {
        uint4 u4;
        uint2 u2;
        half h[8];
        __nv_bfloat16 b[8];
    };
#pragma unroll
    for (int i = dotTid * Width; i < D; i += DotThreads * Width) {
        Packed x, w;
        float4 g[Width / 4];
        if constexpr (Bf16Dot) {
            x.u4 = *reinterpret_cast<const uint4 *>(input + i);
            w.u4 = row < outputs ? *reinterpret_cast<const uint4 *>(weight + i) : uint4{};
        } else {
            x.u2 = *reinterpret_cast<const uint2 *>(input + i);
            w.u2 = row < outputs ? *reinterpret_cast<const uint2 *>(weight + i) : uint2{};
        }
#pragma unroll
        for (int j = 0; j < Width / 4; ++j)
            g[j] = *reinterpret_cast<const float4 *>(norm + i + 4 * j);
        // Keep the native four/eight-element partial sum and its FP16/BF16
        // materialization boundaries before the compensated block reduction.
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < Width; ++j) {
            const float gamma = reinterpret_cast<const float *>(g)[j];
            const half h = __float2half_rn((__half2float(x.h[j]) * scale) * gamma);
            if (blockIdx.x == 0 && localRow == 0)
                normalized[i + j] = h;
            float a = __half2float(h);
            if constexpr (Bf16Dot)
                a = __bfloat162float(__float2bfloat16_rn(a));
            sum += a * __bfloat162float(w.b[j]);
        }
        value += sum;
    }
    dot[dotTid] = value;
    __syncthreads();
    float diff = 0.0f;
    for (int s = DotThreads / 2; s >= 32; s >>= 1) {
        if (dotTid < s) {
            const float other = dot[dotTid + s] - diff;
            const float sum = dot[dotTid] + other;
            diff = (sum - dot[dotTid]) - other;
            dot[dotTid] = sum;
        }
        __syncthreads();
    }
    // The last five compensated steps stay inside one warp. Preserve the
    // same partner values and per-lane compensation without block barriers.
    if (dotTid < 32) {
        float result = dot[dotTid];
#pragma unroll
        for (int s = 16; s; s >>= 1) {
            const float partner = __shfl_down_sync(0xffffffff, result, s);
            if (dotTid < s) {
                const float other = partner - diff;
                const float sum = result + other;
                diff = (sum - result) - other;
                result = sum;
            }
        }
        if (dotTid == 0 && row < outputs) {
            if constexpr (Bf16Dot)
                result = __bfloat162float(__float2bfloat16_rn(result));
            output[size_t(blockIdx.y) * outputs + row] = __float2half_rn(result);
        }
    }
}
} // namespace rmssmall
} // namespace fastllm
