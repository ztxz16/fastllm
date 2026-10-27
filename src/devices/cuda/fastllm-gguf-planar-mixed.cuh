#pragma once
#include "fastllm-gguf-planar-dot.cuh"

namespace fastllm_gguf_planar_mixed {
using namespace fastllm_gguf_planar;
// Larger IQ1_M tiles benefit from spreading the compact codebook load over
// eight warps and capping registers for two resident blocks. Small tiles
// keep four warps to avoid excess threads in memory-bound projections.
template <ggml_type Gate, ggml_type Up, int T>
inline constexpr int BlockWarps = (Gate == GGML_TYPE_IQ1_M || Up == GGML_TYPE_IQ1_M) && T >= 4 ? 8 : 4;
// Different quantization formats share activation loads and one rounded
// SwiGLU writeback. The two codebooks are loaded once per thread block.
template <ggml_type Gate, ggml_type Up, int T>
__global__ __launch_bounds__(32 * BlockWarps<Gate, Up, T>, BlockWarps<Gate, Up, T> == 8 ? 2 : 1) void Project(
    const void *__restrict__ gate, const void *__restrict__ up, const int8_t *__restrict__ qs,
    const half2 *__restrict__ ds, const short2 *__restrict__ sums, half *__restrict__ output, int k, int n,
    int stride) {
    constexpr int Warps = BlockWarps<Gate, Up, T>;
    __shared__ __align__(16) uint32_t gTable[PlanarTableWords<Gate>], uTable[PlanarTableWords<Up>];
    if constexpr (Gate != GGML_TYPE_Q4_K && Gate != GGML_TYPE_Q2_K && Gate != GGML_TYPE_IQ4_XS)
        LoadPlanarTable<Gate, Warps>(gTable);
    if constexpr (Up != GGML_TYPE_Q4_K && Up != GGML_TYPE_Q2_K && Up != GGML_TYPE_IQ4_XS)
        LoadPlanarTable<Up, Warps>(uTable);
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int row = blockIdx.x * Warps + warp; row < n; row += gridDim.x * Warps) {
        float gSum[T] = {}, uSum[T] = {};
#pragma unroll 1
        for (int it = 0; it < (k + 1023) / 1024; ++it) {
            const int slice = it * 32 + lane;
            if (slice >= k / 32)
                continue;
            int a[T][8], loSum[T] = {}, hiSum[T] = {};
            float dx[T];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int4 *p = reinterpret_cast<const int4 *>(qs + t * k + slice * 32);
                const int4 lo = p[0], hi = p[1];
                a[t][0] = lo.x;
                a[t][1] = lo.y;
                a[t][2] = lo.z;
                a[t][3] = lo.w;
                a[t][4] = hi.x;
                a[t][5] = hi.y;
                a[t][6] = hi.z;
                a[t][7] = hi.w;
                dx[t] = __low2float(ds[t * (k / 32) + slice]);
                if constexpr (Gate == GGML_TYPE_Q4_K || Gate == GGML_TYPE_Q2_K || Up == GGML_TYPE_Q4_K ||
                              Up == GGML_TYPE_Q2_K) {
                    const short2 v = sums[t * (k / 32) + slice];
                    loSum[t] = v.x;
                    hiSum[t] = v.y;
                }
            }
            const int block = row * (k / 256) + slice / 8, group = lane % 8;
            Dot<Gate, T>(gate, block, group, gTable, a, dx, loSum, hiSum, gSum);
            Dot<Up, T>(up, block, group, uTable, a, dx, loSum, hiSum, uSum);
        }
        float gValue = 0, uValue = 0;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float g = warp_reduce_sum(gSum[t]), u = warp_reduce_sum(uSum[t]);
            if (lane == t) {
                gValue = g;
                uValue = u;
            }
        }
        if (lane < T) {
            const half g = __float2half_rn(gValue), u = __float2half_rn(uValue);
            const half act = __hdiv(g, __hadd(__float2half(1), hexp(-g)));
            output[lane * stride + row] = __hmul(act, u);
        }
    }
}
} // namespace fastllm_gguf_planar_mixed
