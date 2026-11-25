#pragma once
#include "fastllm-cuda-ordered-reduce.cuh"
#include "fastllm-naive-flex-moe.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace fastllm_norm_rows {
using BF16 = __nv_bfloat16;
constexpr int Width = 4096;
// Multiple experts reuse each normalized two-row tile. Keep the per-row
// reduction order and both BF16 materializations identical to NormRowsRouter.
// Preserve the existing MXFP8/32 result, then rebase four adjacent scales.
// This optional image saves a separate conversion launch before block128 gate.
template<int Capacity = 8>
static __device__ __forceinline__ void StoreGateBlock128(
        uint8_t *dest, int row, int col, uint2 q, unsigned e) {
    unsigned coarse = max(e, __shfl_xor_sync(0xffffffffu, e, 4));
    coarse = max(coarse, __shfl_xor_sync(0xffffffffu, coarse, 8));
    const float ratio = __uint_as_float(unsigned(max(0, 127 + int(e) - int(coarse))) << 23);
    uint2 output{};
    const auto *src = reinterpret_cast<const unsigned short *>(&q);
    auto *dst = reinterpret_cast<unsigned short *>(&output);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __half2_raw bits = __nv_cvt_fp8x2_to_halfraw2(src[j], __NV_E4M3);
        const float2 v = __half22float2(__half2(bits));
        dst[j] = __nv_cvt_float2_to_fp8x2(make_float2(v.x * ratio, v.y * ratio), __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2 *>(dest + row * 4112 + col) = output;
    if (col % 128 == 0)
        reinterpret_cast<float *>(dest + Capacity * 4112)[col / 128 * Capacity + row] = __uint_as_float(coarse << 23);
    if (col == 0) *reinterpret_cast<uint4 *>(dest + row * 4112 + 4096) = make_uint4(0,0,0,0);
}

// Only CTA (0,0) clears rows not covered by any producer tile. The last
// partial tile writes its own padding, so these writes never race it.
template<int Rows>
static __device__ __forceinline__ void ClearGateBlock128Tail(uint8_t *dest, int rows) {
    if (!dest) return;
    for (int row = (rows+Rows-1)/Rows*Rows; row < 8; ++row) {
        for (int col = int(threadIdx.x)*16; col < 4112; col += int(blockDim.x)*16)
            *reinterpret_cast<uint4 *>(dest+row*4112+col) = make_uint4(0,0,0,0);
        for (int k = threadIdx.x; k < 32; k += blockDim.x)
            reinterpret_cast<float *>(dest+32896)[k*8+row] = __uint_as_float(105u<<23);
    }
}

// Marlin's verification reference assigns strided columns to each dot
// lane. The block-scaled path uses contiguous vectors for its packed output.
template<bool Strided, typename Vector, typename Scalar>
static __device__ __forceinline__ Vector ReadRouterTile(const Scalar *chunk, int thread, int element = 0) {
    if constexpr (Strided) {
        Vector result;
#pragma unroll
        for (int j = 0; j < sizeof(Vector) / sizeof(Scalar); ++j)
            reinterpret_cast<Scalar *>(&result)[j] = chunk[thread + (element + j) * 256];
        return result;
    } else {
        return *reinterpret_cast<const Vector *>(chunk + thread * 8 + element);
    }
}

// Reuse router weights across a row tile while consuming each rounded normalization
// vector in registers. Batch the independent norm reductions at one barrier.
// The BF16 normalization and FP8 preparation remain bitwise unchanged;
// FP32 dot accumulation has a separate operator-L2 validation contract.
// Four unprepared rows share the same weights without allocating an FP8 image.
// With a residual branch, stage the rounded sum until every router tile has read the input.
template<int Experts, int Rows = 2, bool Prepare = true, int Capacity = 8, bool Residual = false, bool WaitForSwa = false, bool MarlinOrder = false, class Epilogue = void>
static __device__ __forceinline__ void NormRowsRouterBody(
        const BF16 *input, const float *weight, float *output,
        const float *gamma, BF16 *normed, float eps, int rows,
        FastllmMoePreparedInput prepared, const BF16 *branch = nullptr, BF16 *sumOutput = nullptr) {
#if CUDART_VERSION >= 12000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (Prepare && !Residual && !WaitForSwa && Experts == 8 && Rows == 2 && Capacity == 8)
        if (prepared.overlapVerifyGate) cudaTriggerProgrammaticLaunchCompletion();
#endif
    constexpr int Chunks = 2, Warps = 8;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int expert = blockIdx.x * Experts, first = blockIdx.y * Rows;
    static_assert(!MarlinOrder || (!Prepare && !Residual));
    constexpr int NormWarps = MarlinOrder ? 2 * Warps : Warps;
    __shared__ float normSums[Rows][NormWarps], dotSums[Experts * Rows][Warps];

    float4 affine[Chunks * 2];
#pragma unroll
    for (int i = 0; i < Chunks * 2; ++i) {
        const int chunk = (i / 2) * 2048;
        affine[i] = ReadRouterTile<MarlinOrder, float4>(gamma + chunk, t, (i % 2) * 4);
    }
    float4 cached[Experts][Chunks * 2];
#pragma unroll
    for (int e = 0; e < Experts; ++e) {
#pragma unroll
        for (int i = 0; i < Chunks * 2; ++i) {
            const int chunk = (i / 2) * 2048;
            cached[e][i] = ReadRouterTile<MarlinOrder, float4>(weight + size_t(expert + e) * Width + chunk, t, (i % 2) * 4);
        }
    }
#if CUDART_VERSION >= 12000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (WaitForSwa) {
        // Only immutable router weights and gamma can be loaded before the SWA residual is ready.
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    float partial[Experts][Rows]{};
    if constexpr (Prepare) if (blockIdx.x == 0 && blockIdx.y == 0) {
        // A custom epilogue initializes its own scheduling counters.
        if constexpr (std::is_void_v<Epilogue>) prepared.counters[t] = 0;
        if constexpr (!Residual) if (t == 0) ++*prepared.epoch;
        if constexpr (Capacity == 8 && !Residual) ClearGateBlock128Tail<Rows>(prepared.block128,rows);
    }
    uint4 raw[Rows][Chunks];
    float inv[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if constexpr (MarlinOrder) {
#pragma unroll
            for (int c = 0; c < Chunks; ++c) {
                uint4 packet{};
                if (first + row < rows)
                    packet = *reinterpret_cast<const uint4 *>(input + (first + row) * Width + c * 2048 + t * 8);
                float square = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float v = float(reinterpret_cast<const BF16 *>(&packet)[j]);
                    square += v * v;
                }
                square = fastllm::cuda::WarpSum(square);
                if (!lane) normSums[row][c * Warps + warp] = square;
            }
        } else {
            float square = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int index = (first + row) * Width + t + i * 256;
                float v = 0.f;
                if (first + row < rows) {
                    BF16 x = input[index];
                    if constexpr (Residual) x = __hadd(x, branch[index]);
                    v = float(x);
                }
                square += v * v;
            }
            square = fastllm::cuda::WarpSum(square);
            if (lane == 0) normSums[row][warp] = square;
        }
#pragma unroll
        for (int c = 0; c < Chunks; ++c) {
            const int offset = (first + row) * Width + t * 8 + c * 2048;
            uint4 x{};
            if (first + row < rows) {
                x = ReadRouterTile<MarlinOrder, uint4>(input + (first + row) * Width + c * 2048, t);
                if constexpr (Residual) {
                    const uint4 y = *reinterpret_cast<const uint4 *>(branch + offset);
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        reinterpret_cast<__nv_bfloat162 *>(&x)[j] = __hadd2(
                            reinterpret_cast<const __nv_bfloat162 *>(&x)[j],
                            reinterpret_cast<const __nv_bfloat162 *>(&y)[j]);
                    if (expert == 0) *reinterpret_cast<uint4 *>(sumOutput + offset) = x;
                }
            }
            raw[row][c] = x;
        }
    }
    __syncthreads();
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        const float sum = fastllm::cuda::WarpSum(lane < NormWarps ? normSums[row][lane] : 0.f);
        inv[row] = rsqrtf(sum / Width + eps);
    }
#pragma unroll
    for (int c = 0; c < Chunks; ++c) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float v[8], mx = 0.f;
            uint4 rounded;
            auto *dest = reinterpret_cast<__nv_bfloat162 *>(&rounded);
            const auto *src = reinterpret_cast<const __nv_bfloat162 *>(&raw[row][c]);
            const float *gm = reinterpret_cast<const float *>(&affine[c * 2]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 x = __bfloat1622float2(src[j]);
                const float2 n = __bfloat1622float2(__floats2bfloat162_rn(x.x * inv[row], x.y * inv[row]));
                dest[j] = __floats2bfloat162_rn(n.x * gm[j * 2], n.y * gm[j * 2 + 1]);
                const float2 z = __bfloat1622float2(dest[j]);
                v[j * 2] = z.x; v[j * 2 + 1] = z.y;
                mx = fmaxf(mx, fmaxf(fabsf(z.x), fabsf(z.y)));
            }
            const int col = t * 8 + c * 2048;
#pragma unroll
            for (int e = 0; e < Experts; ++e) {
                const float *w = reinterpret_cast<const float *>(&cached[e][c * 2]);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    partial[e][row] = __fmaf_rn(v[j], w[j], partial[e][row]);
            }
            if (expert == 0 && first + row < rows) {
                BF16 *dst = normed + (first + row) * Width + c * 2048;
                if constexpr (MarlinOrder) {
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        dst[t + j * 256] = reinterpret_cast<const BF16 *>(&rounded)[j];
                } else {
                    *reinterpret_cast<uint4 *>(dst + t * 8) = rounded;
                }
            }
            if constexpr (Prepare) if (blockIdx.x == warp + c * Warps) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
                const unsigned e = max((__float_as_uint(mx) + 0x7fffffu - 0x600000u) >> 23, 113u) - 8;
                const float scale = __uint_as_float((254 - e) << 23);
                uint2 q{};
                auto *halves = reinterpret_cast<unsigned short *>(&q);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    halves[j] = __nv_cvt_float2_to_fp8x2(make_float2(v[j * 2] * scale, v[j * 2 + 1] * scale),
                        __NV_SATFINITE, __NV_E4M3);
                const size_t offset = size_t(first + row) * Width + col;
                if constexpr (Capacity == 8) if (first + row < rows) {
                    if constexpr (!std::is_void_v<Epilogue>) {
                        Epilogue::store(prepared,first+row,col,q,e);
                    } else {
                        *reinterpret_cast<uint2 *>(prepared.values + offset) = q;
                        if (t % 4 == 0) prepared.scales[offset / 32] = e;
                    }
                }
                if constexpr (!Residual) if (prepared.block128) StoreGateBlock128<Capacity>(prepared.block128, first + row, col, q, e);
            }
        }
    }
#pragma unroll
    for (int e = 0; e < Experts; ++e) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
#pragma unroll
            for (int d = 1; d < 32; d *= 2)
                partial[e][row] += __shfl_down_sync(0xffffffff, partial[e][row], d);
            if (lane == 0) dotSums[e * Rows + row][warp] = partial[e][row];
        }
    }
    __syncthreads();
    for (int task = t / 8; task < Experts * Rows; task += 32) {
        float value = dotSums[task][t % 8];
#pragma unroll
        for (int d = 1; d < 8; d *= 2) value += __shfl_down_sync(0xffffffff, value, d, 8);
        if (t % 8 == 0 && first + task % Rows < rows)
            output[(first + task % Rows) * 256 + expert + task / Rows] = 1.0 / (1.0 + expf(-(value + 0.f)));
    }
    if constexpr (!std::is_void_v<Epilogue>) Epilogue::complete(prepared,output);
}

template<int Experts, int Rows = 2, bool Prepare = true, int Capacity = 8, bool Residual = false, bool WaitForSwa = false, class Epilogue = void>
static __global__ void NormRowsRouterDirect(
        const BF16 *input, const float *weight, float *output,
        const float *gamma, BF16 *normed, float eps, int rows,
        FastllmMoePreparedInput prepared, const BF16 *branch = nullptr, BF16 *sumOutput = nullptr) {
    NormRowsRouterBody<Experts, Rows, Prepare, Capacity, Residual, WaitForSwa, false, Epilogue>(
        input, weight, output, gamma, normed, eps, rows, prepared, branch, sumOutput);
}

} // namespace fastllm_norm_rows
