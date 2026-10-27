#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace fastllm_fp8_marlin {

static __device__ __forceinline__ void DecodeFour(uint32_t q, half2 &a, half2 &b) {
    uint32_t q01 = __byte_perm(q, 0u, 0x1404);
    uint32_t q23 = __byte_perm(q, 0u, 0x3424);
    uint32_t o01 = (q01 & 0x80008000u) | ((q01 & 0x7F007F00u) >> 1);
    uint32_t o23 = (q23 & 0x80008000u) | ((q23 & 0x7F007F00u) >> 1);
    a = *reinterpret_cast<half2 *>(&o01);
    b = *reinterpret_cast<half2 *>(&o23);
}

// Reuse each packed FP8 word across Rows tokens. The K split, half products,
// FP32 scale/reduction and rounding match the existing one-token layout GEMV.
// Requires sizeN % 64 == 0, sizeK % 128 == 0 and 128x128 block scales.
template<int Rows>
static __global__ void MultiRowGemv(
        const half *__restrict__ input, const uint32_t *__restrict__ weight,
        const float *__restrict__ scales, const half *__restrict__ bias,
        half *__restrict__ output, int sizeN, int sizeK) {
    static_assert(Rows == 2 || Rows == 3, "two or three token GEMV only");
    constexpr int Warps = 8;
    __shared__ float partial[Rows][Warps][8];
    const int computeWarp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int storageWarp = blockIdx.x & 7;
    const int nTile = blockIdx.x >> 3;
    const int word = storageWarp * 32 + lane;
    const int rowPair = lane >> 3;
    const int slot = lane & 7;
    const int outLocal = (slot >> 1) * 16 + storageWarp + (slot & 1) * 8;
    const int out = nTile * 64 + outLocal;
    const int nTiles = sizeN >> 6;
    const int scaleCols = sizeK >> 7;
    float acc[Rows] = {};
    for (int group = computeWarp; group < scaleCols; group += Warps) {
        float groupAcc[Rows] = {};
        const size_t tileStride = size_t(nTiles) * 256;
        size_t offset = (size_t(group * 8) * nTiles + nTile) * 256 + word;
        uint32_t packed = weight[offset];
#pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            half2 w01, w23;
            DecodeFour(packed, w01, w23);
            offset += tileStride;
            if (tile + 1 < 8) packed = weight[offset];
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const int in = group * 128 + tile * 16 + rowPair * 2;
                const half *a = input + size_t(row) * sizeK + in;
                const half2 p0 = *reinterpret_cast<const half2 *>(a);
                const half2 p8 = *reinterpret_cast<const half2 *>(a + 8);
                const half2 a01 = __halves2half2(__low2half(p0), __low2half(p8));
                const half2 a23 = __halves2half2(__high2half(p0), __high2half(p8));
                const half2 sum = __hadd2(__hmul2(a01, w01), __hmul2(a23, w23));
                groupAcc[row] += __half2float(__low2half(sum)) +
                                 __half2float(__high2half(sum));
            }
        }
        const float scale = scales[size_t(nTile >> 1) * scaleCols + group] * 256.0f;
#pragma unroll
        for (int row = 0; row < Rows; ++row) acc[row] += groupAcc[row] * scale;
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        acc[row] += __shfl_down_sync(0xffffffffu, acc[row], 16);
        acc[row] += __shfl_down_sync(0xffffffffu, acc[row], 8);
        if (lane < 8) partial[row][computeWarp][lane] = acc[row];
    }
    __syncthreads();
    if (computeWarp == 0 && lane < 8) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float value = partial[row][0][lane];
#pragma unroll
            for (int warp = 1; warp < Warps; ++warp) value += partial[row][warp][lane];
            if (bias) value += __half2float(bias[out]);
            output[size_t(row) * sizeN + out] = __float2half_rn(value);
        }
    }
}

} // namespace fastllm_fp8_marlin
