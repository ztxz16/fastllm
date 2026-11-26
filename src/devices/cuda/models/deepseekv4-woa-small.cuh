#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace fastllm { namespace cuda { namespace woa_small {

// The compact checkpoint path rounds scaled E4M3 to BF16. The original
// generic projection instead scales the intermediate half encoding and rounds
// to FP16. Keep both contracts, including their treatment of E4M3 NaNs.
template<bool Compact>
__device__ __forceinline__ float WeightValue(unsigned code, float scale) {
    unsigned short bits = ((code & 128) << 8) | ((code & 127) << 7);
    float pseudoHalf = __half2float(__ushort_as_half(bits));
    if constexpr (Compact) {
        if ((code & 127) == 127) return __int_as_float(0x7fffffff);
        return __bfloat162float(__float2bfloat16_rn((pseudoHalf * 256.0f) * scale));
    } else {
        return __half2float(__float2half_rn((pseudoHalf * scale) * 256.0f));
    }
}

template<int Rows, bool Compact>
__device__ __forceinline__ void WarpDot(const __nv_bfloat16 *input, const uint8_t *weight,
        const float *scales, __nv_bfloat16 *output, int tokens, int groups, int rank,
        int hidden, int row) {
    const int group = blockIdx.y, first = blockIdx.z * Rows, lane = threadIdx.x % 32;
    const size_t weightRow = size_t(group) * rank + row;
    float sums[Rows] = {};
    for (int k = lane * 4; k < hidden; k += 128) {
        uint32_t packed = *reinterpret_cast<const uint32_t *>(weight + weightRow * hidden + k);
        float scale = scales[(weightRow / 32) * (hidden / 32) + k / 32];
        float values[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) values[j] = WeightValue<Compact>((packed >> (8 * j)) & 255, scale);
#pragma unroll
        for (int t = 0; t < Rows; ++t) {
            if (first + t >= tokens) continue;
            const auto *src = input + (size_t(first + t) * groups + group) * hidden + k;
            uint2 pair = *reinterpret_cast<const uint2 *>(src);
            const auto *activation = reinterpret_cast<const __nv_bfloat16 *>(&pair);
#pragma unroll
            for (int j = 0; j < 4; ++j) sums[t] = fmaf(__bfloat162float(activation[j]), values[j], sums[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < Rows; ++t) {
#pragma unroll
        for (int d = 16; d; d /= 2) sums[t] += __shfl_down_sync(0xffffffff, sums[t], d);
        if (lane == 0 && first + t < tokens)
            output[(size_t(first + t) * groups + group) * rank + row] = __float2bfloat16_rn(sums[t]);
    }
}

template<int Rows, bool Compact>
__global__ void WoaWarpRows(const __nv_bfloat16 *input, const uint8_t *weight, const float *scales,
        __nv_bfloat16 *output, int tokens, int groups, int rank, int hidden) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    WarpDot<Rows, Compact>(input, weight, scales, output, tokens, groups, rank, hidden, row);
}

__device__ __forceinline__ unsigned PackedBf16(const uint8_t *weight, float scale) {
    unsigned codes = *reinterpret_cast<const uint16_t *>(weight);
    auto pair = __floats2bfloat162_rn(WeightValue<true>(codes & 255, scale), WeightValue<true>(codes >> 8, scale));
    return (unsigned(__bfloat16_as_ushort(pair.y)) << 16) | __bfloat16_as_ushort(pair.x);
}

// A warp computes 16 output features x 8 token rows. Warps share the K
// reduction, decoding the compact weight fragment directly into registers.
template<bool Compact, int Warps>
__global__ void WoaMmaRows(const __nv_bfloat16 *input, const uint8_t *weight, const float *scales,
        __nv_bfloat16 *output, int tokens, int groups, int rank, int hidden) {
    const int row = blockIdx.x * 16, warp = threadIdx.x / 32;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(CUDA_NO_TENSOR_CORE)
    const int lane = threadIdx.x % 32, r = lane / 4, c = lane % 4;
    const int first = blockIdx.z * 8, group = blockIdx.y;
    const size_t weightRow = size_t(group) * rank + row;
    float sum[4] = {};
    constexpr int step = Compact ? 16 : 8;
    for (int tile = warp * 32; tile < hidden; tile += Warps * 32) {
        float scale = scales[(weightRow / 32) * (hidden / 32) + tile / 32];
#pragma unroll
        for (int offset = 0; offset < 32; offset += step) {
            int k = tile + offset + (Compact ? 2 * c : c);
            const uint8_t *w = weight + (weightRow + r) * hidden + k;
            unsigned b0 = 0, b1 = 0;
            if constexpr (Compact) {
                if (first + r < tokens) {
                    const auto *x = input + (size_t(first + r) * groups + group) * hidden + k;
                    b0 = *reinterpret_cast<const uint32_t *>(x);
                    b1 = *reinterpret_cast<const uint32_t *>(x + 8);
                }
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3])
                    : "r"(PackedBf16(w, scale)), "r"(PackedBf16(w + 8 * size_t(hidden), scale)),
                      "r"(PackedBf16(w + 8, scale)), "r"(PackedBf16(w + 8 * size_t(hidden) + 8, scale)),
                      "r"(b0), "r"(b1));
            } else {
                // BF16 activations and FP16-rounded weights both fit TF32 exactly.
                if (first + r < tokens) {
                    const auto *x = input + (size_t(first + r) * groups + group) * hidden + k;
                    b0 = __float_as_uint(__bfloat162float(x[0]));
                    b1 = __float_as_uint(__bfloat162float(x[4]));
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3])
                    : "r"(__float_as_uint(WeightValue<false>(w[0], scale))),
                      "r"(__float_as_uint(WeightValue<false>(w[8 * size_t(hidden)], scale))),
                      "r"(__float_as_uint(WeightValue<false>(w[4], scale))),
                      "r"(__float_as_uint(WeightValue<false>(w[8 * size_t(hidden) + 4], scale))), "r"(b0), "r"(b1));
            }
        }
    }
    __shared__ float partial[Warps][4][32];
#pragma unroll
    for (int j = 0; j < 4; ++j) partial[warp][j][lane] = sum[j];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float value = partial[0][j][lane];
#pragma unroll
            for (int i = 1; i < Warps; ++i) value += partial[i][j][lane];
            int token = first + 2 * c + (j & 1);
            if (token < tokens)
                output[(size_t(token) * groups + group) * rank + row + r + (j / 2) * 8] = __float2bfloat16_rn(value);
        }
    }
#else
    // A lower-target PTX image may be JIT-loaded on a newer device.
    for (int i = warp; i < 16; i += Warps)
        WarpDot<8, Compact>(input, weight, scales, output, tokens, groups, rank, hidden, row + i);
#endif
}

inline bool CanLaunch(const void *input, const void *weight, int tokens, int groups,
                      int rank, int hidden, int arch) {
    return arch >= 75 && tokens >= 1 && tokens <= 16 && groups > 0 && groups <= 65535 &&
           rank > 0 && rank % 32 == 0 && hidden > 0 && hidden % 32 == 0 &&
           reinterpret_cast<uintptr_t>(input) % 8 == 0 && reinterpret_cast<uintptr_t>(weight) % 4 == 0;
}

template<bool Compact>
inline bool Launch(const __nv_bfloat16 *input, const uint8_t *weight, const float *scales, __nv_bfloat16 *output,
        int tokens, int groups, int rank, int hidden, int arch, cudaStream_t stream) {
    if (tokens == 1) {
        WoaWarpRows<1, Compact><<<dim3(rank / 8, groups), 256, 0, stream>>>(input, weight, scales, output, tokens, groups, rank, hidden);
    } else if (arch < 80) {
        WoaWarpRows<8, Compact><<<dim3(rank / 8, groups, (tokens + 7) / 8), 256, 0, stream>>>(input, weight, scales, output, tokens, groups, rank, hidden);
    } else {
        constexpr int warps = Compact ? 4 : 8;
        WoaMmaRows<Compact, warps><<<dim3(rank / 16, groups, (tokens + 7) / 8), warps * 32, 0, stream>>>(input, weight, scales, output, tokens, groups, rank, hidden);
    }
    return cudaGetLastError() == cudaSuccess;
}
}}}
