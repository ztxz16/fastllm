#pragma once
#include "fastllm-cuda-ordered-reduce.cuh"
#include "fastllm-mxfp8-activation.cuh"

namespace fastllm_swa_qkv {
// Copy four consecutive K32 exponents per word; transpose the row and K128
// indices with a bank-conflict-free shared tile. No persistent packed copy.
template <int TileRows, int TileGroups, int Threads>
__device__ __forceinline__ void PackWeightTile(const float4 *src, uint32_t *dst, int n,
                                               int k, int tilesPerRow, int tileIndex) {
    __shared__ uint32_t tile[TileRows][TileGroups + 1];
    const int groups = k / 128, firstGroup = tileIndex % tilesPerRow * TileGroups,
              firstRow = tileIndex / tilesPerRow * TileRows;
    for (int i = threadIdx.x; i < TileRows * TileGroups; i += Threads) {
        const int r = i / TileGroups, g = i % TileGroups;
        if (firstRow + r < n && firstGroup + g < groups) {
            const float4 v = src[size_t(firstRow + r) * groups + firstGroup + g];
            tile[r][g] = (__float_as_uint(v.x) >> 23) | ((__float_as_uint(v.y) >> 23) << 8) |
                         ((__float_as_uint(v.z) >> 23) << 16) | ((__float_as_uint(v.w) >> 23) << 24);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TileGroups * TileRows; i += Threads) {
        const int g = i / TileRows, r = i % TileRows;
        if (firstRow + r < n && firstGroup + g < groups)
            dst[size_t(firstGroup + g) * n + firstRow + r] = tile[r][g];
    }
}

// One CTA per verification row preserves AddRMSNorm's 256-lane reduction and
// both BF16 rounding points. Backends without cached scales also launch packing CTAs.
template<bool AfterExpertReduce, bool Sm120 = false>
__global__ void PrepareSwaQkv(__nv_bfloat16 *hidden, const __nv_bfloat16 *branch,
                              const float *gamma, float eps, const float4 *weightScales,
                              uint32_t *packed, uint8_t *qx, uint8_t *sx, int rows, int pitch,
                              const float *positions, float theta, __nv_bfloat162 *rotary,
                              const __nv_bfloat16 *normalized, __nv_bfloat16 *feature) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // The GEMM can initialize its descriptors and tensor memory while these
    // CTAs prepare the operands. It waits before issuing its first TMA read.
    cudaTriggerProgrammaticLaunchCompletion();
    // Scale tiles only touch private scratch. Residual and rotary tiles need the previous output.
    if constexpr (AfterExpertReduce) {
        if (blockIdx.x < rows || blockIdx.x == gridDim.x - 1)
            cudaGridDependencySynchronize();
    }
    constexpr int width = 4096, channels = 1856, threads = 512;
    const int row = blockIdx.x, t = threadIdx.x;
    if (row == int(gridDim.x) - 1) {
        if (t < rows * 32) {
            const float angle = positions[t / 32] * powf(theta, -2.f * (t % 32) / 64.f);
            rotary[t] = __floats2bfloat162_rn(cosf(angle), sinf(angle));
        }
        return;
    }
    if constexpr (!Sm120) {
        if (row >= rows) {
            PackWeightTile<16, 32, threads>(weightScales, packed, channels, width, 1, row - rows);
            return;
        }
    }
    using Pair = __nv_bfloat162;
    union Vector {
        uint4 words;
        Pair pairs[4];
        __device__ Vector() {}
    };
    const int col = t * 8, index = row * width + col;
    FastllmMxfp8Activation activation{qx, sx, rows, width, pitch, Sm120 ? 0 : -1};
    if constexpr (Sm120) {
        // An already materialized input can enter here after a generic FFN or fallback.
        if (normalized) {
            fastllm_mxfp8::Store8(activation, row, col, *reinterpret_cast<const uint4 *>(normalized + index));
            return;
        }
    }
    Vector residual, addend;
    residual.words = *reinterpret_cast<const uint4 *>(hidden + index);
    addend.words = *reinterpret_cast<const uint4 *>(branch + index);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        residual.pairs[j] = __hadd2(residual.pairs[j], addend.pairs[j]);
    *reinterpret_cast<uint4 *>(hidden + index) = residual.words;
    if constexpr (Sm120) {
        if (feature) *reinterpret_cast<uint4 *>(feature + index) = residual.words;
    }

    __shared__ __align__(16) __nv_bfloat16 values[width];
    __shared__ float partial[8];
    *reinterpret_cast<uint4 *>(values + col) = residual.words;
    __syncthreads();
    float square = 0.f;
    if (t < 256) {
#pragma unroll
        for (int j = 0; j < width / 256; ++j) {
            const float v = float(values[t + j * 256]);
            square += v * v;
        }
        square = fastllm::cuda::WarpSum(square);
        if ((t & 31) == 0) partial[t / 32] = square;
    }
    __syncthreads();
    if (t < 32) {
        const float total = fastllm::cuda::WarpSum(t < 8 ? partial[t] : 0.f);
        if (t == 0) partial[0] = rsqrtf(total / width + eps);
    }
    __syncthreads();
    const float inv = partial[0];
    const float4 g0 = *reinterpret_cast<const float4 *>(gamma + col);
    const float4 g1 = *reinterpret_cast<const float4 *>(gamma + col + 4);
    const float gains[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 v = __bfloat1622float2(residual.pairs[j]);
        const auto scaled = __floats2bfloat162_rn(v.x * inv, v.y * inv);
        const float2 z = __bfloat1622float2(scaled);
        residual.pairs[j] = __floats2bfloat162_rn(z.x * gains[j * 2], z.y * gains[j * 2 + 1]);
    }
    fastllm_mxfp8::Store8(activation, row, col, residual.words);
#endif
}

} // namespace fastllm_swa_qkv
