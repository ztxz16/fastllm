#pragma once
#include "fastllm-gguf-planar-dot.cuh"

namespace fastllm_gguf_planar_mixed {
using namespace fastllm_gguf_planar;
// Larger IQ1_M tiles benefit from spreading the compact codebook load over
// eight warps and capping registers for two resident blocks. Small tiles
// keep four warps to avoid excess threads in memory-bound projections.
template <ggml_type First, ggml_type Second, int T>
inline constexpr int BlockWarps = (First == GGML_TYPE_IQ1_M || Second == GGML_TYPE_IQ1_M) && T >= 4 ? 8 : 4;
// Different quantization formats share activation loads and one rounded
// SwiGLU writeback. The two codebooks are loaded once per thread block.
template <ggml_type First, ggml_type Second, int T>
__global__ __launch_bounds__(32 * BlockWarps<First, Second, T>, BlockWarps<First, Second, T> == 8 ? 2 : 1) void Project(
    const void *__restrict__ firstWeight, const void *__restrict__ secondWeight, const int8_t *__restrict__ qs,
    const half2 *__restrict__ ds, const short2 *__restrict__ sums, half *__restrict__ output, int k, int n,
    int stride, bool gateIsSecond) {
    constexpr int Warps = BlockWarps<First, Second, T>;
    __shared__ __align__(16) uint32_t firstTable[PlanarTableWords<First>], secondTable[PlanarTableWords<Second>];
    if constexpr (First != GGML_TYPE_Q4_K && First != GGML_TYPE_Q2_K && First != GGML_TYPE_IQ4_XS)
        LoadPlanarTable<First, Warps>(firstTable);
    if constexpr (Second != GGML_TYPE_Q4_K && Second != GGML_TYPE_Q2_K && Second != GGML_TYPE_IQ4_XS)
        LoadPlanarTable<Second, Warps>(secondTable);
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int row = blockIdx.x * Warps + warp; row < n; row += gridDim.x * Warps) {
        float firstSum[T] = {}, secondSum[T] = {};
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
                if constexpr (First == GGML_TYPE_Q4_K || First == GGML_TYPE_Q2_K || Second == GGML_TYPE_Q4_K ||
                              Second == GGML_TYPE_Q2_K) {
                    const short2 v = sums[t * (k / 32) + slice];
                    loSum[t] = v.x;
                    hiSum[t] = v.y;
                }
            }
            const int block = row * (k / 256) + slice / 8, group = lane % 8;
            Dot<First, T>(firstWeight, block, group, firstTable, a, dx, loSum, hiSum, firstSum);
            Dot<Second, T>(secondWeight, block, group, secondTable, a, dx, loSum, hiSum, secondSum);
        }
        float firstValue = 0, secondValue = 0;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float first = warp_reduce_sum(firstSum[t]), second = warp_reduce_sum(secondSum[t]);
            if (lane == t) {
                firstValue = first;
                secondValue = second;
            }
        }
        if (lane < T) {
            const half gate = __float2half_rn(gateIsSecond ? secondValue : firstValue);
            const half up = __float2half_rn(gateIsSecond ? firstValue : secondValue);
            const half act = __hdiv(gate, __hadd(__float2half(1), hexp(-gate)));
            output[lane * stride + row] = __hmul(act, up);
        }
    }
}
} // namespace fastllm_gguf_planar_mixed
