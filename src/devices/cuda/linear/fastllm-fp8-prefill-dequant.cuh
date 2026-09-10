#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Decode the packed Marlin FP8 bytes with its rounded block128 half scales.
// N slices start and end on 64-row boundaries; K is a multiple of 128.
__device__ __forceinline__ void Fp8PrefillDecodePairs(
        uint32_t packed, half2 scale, half2 &low, half2 &high) {
    uint32_t x = (packed & 0x00ff00ffu) << 8;
    uint32_t y = packed & 0xff00ff00u;
    x = (x & 0x80008000u) | ((x & 0x7f007f00u) >> 1);
    y = (y & 0x80008000u) | ((y & 0x7f007f00u) >> 1);
    low = __hmul2(*reinterpret_cast<half2 *>(&x), scale);
    high = __hmul2(*reinterpret_cast<half2 *>(&y), scale);
}

// Keep packed reads coalesced, transpose a 64 x 256 tile in shared memory,
// then write eight consecutive FP16 values per thread for the cuBLAS operand.
__global__ void Fp8PrefillDequantKernel(
        const uint32_t *weight, const half *scales, half *output,
        int sizeN, int sizeK, int offsetN) {
    constexpr int kTile = 256;
    __shared__ half2 tile[64 * (kTile / 2)];
    int tileN = blockIdx.x, startK = blockIdx.y * kTile, word = threadIdx.x;
    int lane = word & 31, slot = lane & 7, rowPair = lane >> 3;
    int localN = (slot >> 1) * 16 + (word >> 5) + (slot & 1) * 8;
    int globalN = offsetN + tileN * 64 + localN;
    int swizzle = (localN >> 3) * 4;
#pragma unroll
    for (int t = 0; t < kTile / 16; ++t) {
        int tileK = startK / 16 + t;
        half2 low = __float2half2_rn(0), high = low;
        if (tileK * 16 < sizeK) {
            half scale = scales[size_t(tileK / 8) * sizeN + (globalN / 64) * 64];
            Fp8PrefillDecodePairs(
                weight[(size_t(tileK) * (sizeN / 64) + globalN / 64) * 256 + word],
                __halves2half2(scale, scale), low, high);
        }
        int pairK = t * 8 + rowPair;
        tile[localN * (kTile / 2) + (pairK ^ swizzle)] = low;
        tile[localN * (kTile / 2) + ((pairK + 4) ^ swizzle)] = high;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * kTile / 8; i += blockDim.x) {
        int row = i / (kTile / 8), vectorK = i % (kTile / 8);
        int pairK = vectorK * 4, readSwizzle = (row >> 3) * 4;
        if (startK + vectorK * 8 < sizeK) {
            reinterpret_cast<uint4 *>(output)[size_t(tileN * 64 + row) * (sizeK / 8) + startK / 8 + vectorK] =
                *reinterpret_cast<uint4 *>(&tile[row * (kTile / 2) + (pairK ^ readSwizzle)]);
        }
    }
}
