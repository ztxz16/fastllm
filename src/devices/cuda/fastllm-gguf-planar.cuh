#pragma once

#include "fastllm-gguf-planar-dot.cuh"

// Preserve the integer arithmetic and FP16 boundaries of small-mmvq.
// Planar activations allow aligned vector loads. Keep the K loop rolled
// to avoid register spills and excessive instruction size for wide models.
namespace fastllm_gguf_planar {
using namespace fastllm_gguf_small_mmvq;
template <ggml_type Type, int T, int Mode>
inline constexpr int BlockWarps = Type == GGML_TYPE_IQ1_M && T >= 5      ? 8
                                  : Mode <= 1 && Type == GGML_TYPE_IQ2_S ? (T == 1   ? 8
                                                                            : T == 7 ? 2
                                                                                     : 4)
                                                                         : 4;
template <ggml_type Type, int T, int Mode>
inline constexpr int OutputRows =
    Mode == 3                                                                     ? 2
    : Type == GGML_TYPE_IQ2_S && T == 6                                           ? 4
    : (T == 2 || (T == 3 && (Type == GGML_TYPE_Q2_K || Type == GGML_TYPE_IQ2_S))) ? 1
                                                                                  : 2;

// Keep the original XOR16,8,4,2,1 sum tree, but progressively assign each
// output to its destination lane. Exchanging the opposite register halves
// avoids computing every final result in all 32 lanes. Constant indices let
// the compiler keep the shrinking arrays in registers.
template <int Mask, int Count>
__device__ __forceinline__ float ReduceOutputLanes(float (&acc)[Count], int lane) {
    if constexpr (Mask == 0) {
        return acc[0];
    } else if constexpr (Count <= Mask) {
#pragma unroll
        for (int j = 0; j < Count; ++j)
            acc[j] += __shfl_xor_sync(0xffffffff, acc[j], Mask);
        return ReduceOutputLanes<Mask / 2>(acc, lane);
    } else {
        static_assert(Count == Mask * 2);
        float next[Mask];
#pragma unroll
        for (int j = 0; j < Mask; ++j) {
            const float own = lane & Mask ? acc[j + Mask] : acc[j];
            const float send = lane & Mask ? acc[j] : acc[j + Mask];
            next[j] = own + __shfl_xor_sync(0xffffffff, send, Mask);
        }
        return ReduceOutputLanes<Mask / 2>(next, lane);
    }
}

// A single input row leaves enough registers to decode gate and up in
// separate half warps. Keep two K partials so their first addition matches
// the original XOR16 reduction before reducing within each half warp.
template <ggml_type Type>
__global__ __launch_bounds__(128, 1) void ProjectGateUpSingle(const void *__restrict__ gate,
                                                              const void *__restrict__ up,
                                                              const int8_t *__restrict__ qs,
                                                              const half2 *__restrict__ ds,
                                                              half *__restrict__ output, int k, int n) {
    static_assert(Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_IQ3_XXS);
    __shared__ __align__(16) uint32_t table[PlanarTableWords<Type>];
    LoadPlanarTable<Type, 4>(table);
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const void *weight = lane < 16 ? gate : up;
    for (int row = blockIdx.x * 4 + warp; row < n; row += gridDim.x * 4) {
        float sum[2][1] = {};
#pragma unroll 1
        for (int it = 0; it < (k + 1023) / 1024; ++it) {
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                const int slice = it * 32 + lane % 16 + 16 * part;
                if (slice >= k / 32)
                    continue;
                const int4 *p = reinterpret_cast<const int4 *>(qs + slice * 32);
                const int4 lo = p[0], hi = p[1];
                const int a[1][8] = {{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w}};
                const float dx[1] = {__low2float(ds[slice])};
                const int unused[1] = {};
                Dot<Type, 1, 3>(weight, row * (k / 256) + slice / 8, slice % 8, table, a, dx, unused, unused,
                             sum[part]);
            }
        }
        float value = sum[0][0] + sum[1][0];
#pragma unroll
        for (int offset = 8; offset; offset >>= 1)
            value += __shfl_xor_sync(0xffffffff, value, offset, 16);
        const float upValue = __shfl_sync(0xffffffff, value, 16);
        if (lane == 0) {
            const half g = __float2half_rn(value), u = __float2half_rn(upValue);
            output[row] = __hmul(__hdiv(g, __hadd(__float2half(1), hexp(-g))), u);
        }
    }
}

template <ggml_type Type, int T, int KS, int Mode, bool StreamingWeights = false>
__global__
__launch_bounds__(32 * BlockWarps<Type, T, Mode>, BlockWarps<Type, T, Mode> == 8 ? 2 : 1) void Project(
    const void *__restrict__ weights, const void *__restrict__ upWeights, const int8_t *__restrict__ qs,
    const half2 *__restrict__ ds, const short2 *__restrict__ sums, half *__restrict__ output, int n,
    int outputStride, int runtimeK) {
    constexpr int Warps = BlockWarps<Type, T, Mode>;
    const int K = KS ? KS : runtimeK;
    static_assert(T >= 1 && T <= 8, "planar tile must have 1..8 rows");
    __shared__ __align__(16) uint32_t table[PlanarTableWords<Type>];
    if constexpr (Type != GGML_TYPE_Q4_K && Type != GGML_TYPE_Q2_K && Type != GGML_TYPE_IQ4_XS) {
        LoadPlanarTable<Type, Warps>(table);
        __syncthreads();
    }
    // Small tiles can gain occupancy by keeping one output channel per warp.
    constexpr int Rows = OutputRows<Type, T, Mode>;
    constexpr int step = Mode == 3 ? 1 : Rows;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int row = (blockIdx.x * Warps + warp) * step; row < n; row += gridDim.x * Warps * step) {
        float sum[Rows][T] = {};
#pragma unroll 1
        for (int it = 0; it < (K + 1023) / 1024; ++it) {
            const int slice = it * 32 + lane;
            if (slice >= K / 32)
                continue;
            int a[T][8], loSum[T], hiSum[T];
            float dx[T];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int4 *p = reinterpret_cast<const int4 *>(qs + t * K + slice * 32);
                const int4 lo = p[0], hi = p[1];
                a[t][0] = lo.x;
                a[t][1] = lo.y;
                a[t][2] = lo.z;
                a[t][3] = lo.w;
                a[t][4] = hi.x;
                a[t][5] = hi.y;
                a[t][6] = hi.z;
                a[t][7] = hi.w;
                dx[t] = __low2float(ds[t * (K / 32) + slice]);
                if constexpr (Type == GGML_TYPE_Q4_K || Type == GGML_TYPE_Q2_K) {
                    const short2 v = sums[t * (K / 32) + slice];
                    loSum[t] = v.x;
                    hiSum[t] = v.y;
                }
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const void *w = Mode == 3 && r == 1 ? upWeights : weights;
                const int outputRow = Mode == 3 ? row : min(row + r, n - 1);
                const int block = outputRow * (K / 256) + slice / 8, group = lane % 8;
                Dot<Type, T, Mode, StreamingWeights>(w, block, group, table, a, dx, loSum, hiSum, sum[r]);
            }
        }
        // Each lane receives one final result. The half-precision SiLU/divide
        // runs once with 8 or 16 active lanes, rather than serially per token
        // on lane zero. Every reduction and intermediate rounding is unchanged.
        if constexpr (Rows == 2 && T >= 7 &&
                      (Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_IQ3_XXS || Type == GGML_TYPE_IQ4_XS)) {
            // With 14 or 16 outputs, distributed reduction saves enough
            // shuffles to offset its register selection. Smaller tiles and
            // the other quantizers keep their existing register schedules.
            float partials[16] = {};
#pragma unroll
            for (int i = 0; i < Rows * T; ++i)
                partials[i] = sum[i / T][i % T];
            const float value = ReduceOutputLanes<16>(partials, lane);
            if constexpr (Mode == 3) {
                const float upValue = __shfl_sync(0xffffffff, value, (T + lane) & 31);
                if (lane < T) {
                    const half gate = __float2half_rn(value), up = __float2half_rn(upValue);
                    const half act = __hdiv(gate, __hadd(__float2half(1.0f), hexp(-gate)));
                    output[lane * outputStride + row] = __hmul(act, up);
                }
            } else if (lane < Rows * T && row + lane / T < n)
                FastllmGgufStore<Mode>(output + (lane % T) * outputStride + row + lane / T, value);
        } else if constexpr (Rows > 2) {
            static_assert(Mode != 3 && Rows * T <= 32);
            float value = 0;
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float v = warp_reduce_sum(sum[r][t]);
                    if (lane == r * T + t)
                        value = v;
                }
            }
            if (lane < Rows * T && row + lane / T < n)
                FastllmGgufStore<Mode>(output + (lane % T) * outputStride + row + lane / T, value);
        } else {
            float value = 0, gateValue = 0;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float x = warp_reduce_sum(sum[0][t]);
                float y = 0;
                if constexpr (Rows > 1)
                    y = warp_reduce_sum(sum[1][t]);
                if constexpr (Mode == 3) {
                    if (lane == t) {
                        gateValue = x;
                        value = y;
                    }
                } else {
                    if (lane == t)
                        value = x;
                    if (lane == t + T)
                        value = y;
                }
            }
            if constexpr (Mode == 3) {
                if (lane < T) {
                    const half gate = __float2half_rn(gateValue), up = __float2half_rn(value);
                    const half act = __hdiv(gate, __hadd(__float2half(1.0f), hexp(-gate)));
                    output[lane * outputStride + row] = __hmul(act, up);
                }
            } else if (lane < Rows * T && row + lane / T < n)
                FastllmGgufStore<Mode>(output + (lane % T) * outputStride + row + lane / T, value);
        }
    }
}

} // namespace fastllm_gguf_planar
