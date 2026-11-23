// Six-row BF16 x MXFP8 joint Q-A/KV projection on B300.
#pragma once

#define MARLIN_NAMESPACE_NAME fastllm_verify_qkv_marlin
#include "marlin_dense_fp8/dequant.h"
#undef MARLIN_NAMESPACE_NAME

namespace fastllm_verify_qkv {
namespace marlin = fastllm_verify_qkv_marlin;
using BFloat = marlin::MarlinScalarType<vllm::kBFloat16.id()>;
constexpr int Rows = 6, Columns = 1792, Inner = 5120;
constexpr int Panel = 16, Stripes = 10, GroupsPerStripe = 16, Threads = Stripes * 2 * 32;
constexpr size_t WeightBytes = size_t(Columns) * Inner;

// A warp reads both K16 fragments for its K32 group in one uint4.
// Keep the regular Marlin packing available to the other batch shapes.
__global__ void Pack(const uint2 *packed, uint4 *panels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = i % 32, step = (i / 32) % 8, half = (i / 256) % 2;
    const int stripe = (i / 512) % Stripes, panel = i / (512 * Stripes);
    const int group = stripe * GroupsPerStripe + step * 2 + half;
    const int offset = ((group * 2 * Columns + (panel / 4) * 64 + 2 * lane) * 2) + panel % 4;
    const uint2 lo = packed[offset], hi = packed[offset + Columns * 2];
    panels[i] = make_uint4(lo.x, lo.y, hi.x, hi.y);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1030
__device__ __forceinline__ void Accumulate(uint2 code, __nv_bfloat162 scale, uint2 activation, BFloat::FragC &sum) {
    BFloat::FragB first, second;
    marlin::dequant<__nv_bfloat162, vllm::kFE4M3fn.id(), false>(code.x, &first[0]);
    marlin::dequant<__nv_bfloat162, vllm::kFE4M3fn.id(), false>(code.y, &second[0]);
    const auto lo = __halves2bfloat162(scale.x, scale.x), hi = __halves2bfloat162(scale.y, scale.y);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        first[i] = __hmul2(first[i], lo);
        second[i] = __hmul2(second[i], hi);
    }
    const auto *l = reinterpret_cast<const uint32_t *>(&first);
    const auto *h = reinterpret_cast<const uint32_t *>(&second);
    // All dependencies are explicit. Independent panel loads can move across this MMA.
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3])
        : "r"(l[0]), "r"(h[0]), "r"(l[1]), "r"(h[1]), "r"(activation.x), "r"(activation.y));
}
#endif

__global__ void VerifyQkvPanel(const uint32_t *input, const uint4 *weight, const uint2 *scales,
                               __nv_bfloat16 *output) {
#if __CUDA_ARCH__ == 1030
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, row = lane / 4;
    const int panel = blockIdx.x, stripe = warp / 2, half = warp % 2;
    BFloat::FragC sum{};
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const int group = stripe * GroupsPerStripe + 2 * step + half;
        const uint4 w = weight[((panel * Stripes * 2 + warp) * 8 + step) * 32 + lane];
        const uint2 q = scales[(group * Columns / 64 + panel / 4) * 8 + row];
        __nv_bfloat162 pairs[2];
        marlin::dequant_fp8_scales<__nv_bfloat162, vllm::kFE8M0fnu.id()>(panel & 2 ? q.y : q.x, pairs);
        const auto scale = pairs[panel % 2];
        uint32_t x[4]{};
        if (row < Rows) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) x[k] = input[row * (Inner / 2) + group * 16 + k * 4 + lane % 4];
        }
        Accumulate(make_uint2(w.x, w.y), scale, make_uint2(x[0], x[1]), sum);
        Accumulate(make_uint2(w.z, w.w), scale, make_uint2(x[2], x[3]), sum);
    }
    // Match Marlin's two warp partials per stripe, followed by stripes 9..0.
    __shared__ float4 partial[Stripes][2][32];
    partial[stripe][half][lane] = make_float4(sum[0], sum[1], sum[2], sum[3]);
    __syncthreads();
    if (warp == 0) {
        BFloat::FragC result{};
        #pragma unroll
        for (int s = Stripes - 1; s >= 0; --s) {
            const float4 a = partial[s][0][lane], b = partial[s][1][lane];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float value = reinterpret_cast<const float *>(&a)[j] + reinterpret_cast<const float *>(&b)[j];
                result[j] = s == Stripes - 1 ? value : __fadd_rn(value, result[j]);
            }
        }
        const int token = (lane % 4) * 2, col = panel * Panel + row;
        if (token >= Rows) return;
        output[token * Columns + col] = __float2bfloat16_rn(result[0]);
        output[(token + 1) * Columns + col] = __float2bfloat16_rn(result[1]);
        output[token * Columns + col + 8] = __float2bfloat16_rn(result[2]);
        output[(token + 1) * Columns + col + 8] = __float2bfloat16_rn(result[3]);
    }
#endif
}

inline void *Prepare(const void *packed) {
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, VerifyQkvPanel) != cudaSuccess || attr.ptxVersion != 103 ||
        attr.maxThreadsPerBlock < Threads) {
        cudaGetLastError();
        return nullptr;
    }
    void *panel = nullptr;
    if (FastllmCudaTryDirectMalloc(&panel, WeightBytes) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return nullptr;
    constexpr int threads = 256;
    Pack<<<WeightBytes / sizeof(uint4) / threads, threads, 0, cudaStreamPerThread>>>(
        static_cast<const uint2 *>(packed), static_cast<uint4 *>(panel));
    if (cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess) return panel;
    FastllmCudaDirectFree(panel);
    return nullptr;
}

inline bool Run(const void *input, const void *weight, const void *scales, void *output) {
    if (!input || !output || ((reinterpret_cast<uintptr_t>(input) & 3) | (reinterpret_cast<uintptr_t>(output) & 1)))
        return false;
    VerifyQkvPanel<<<Columns / Panel, Threads, 0, cudaStreamPerThread>>>(static_cast<const uint32_t *>(input),
        static_cast<const uint4 *>(weight), static_cast<const uint2 *>(scales), static_cast<__nv_bfloat16 *>(output));
    return cudaGetLastError() == cudaSuccess;
}
} // namespace fastllm_verify_qkv
