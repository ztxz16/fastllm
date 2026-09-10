#pragma once

// Reverse the Marlin S0E5M3 scale permutation and preserve its normalized FP16
// weight exactly. The caller applies the FP32 tensor scale after accumulation;
// it is never folded into these FP16 weights.
__device__ __forceinline__ int Nvfp4PrefillScaleIndex(int n) {
    int transposed = ((n & 7) << 3) | ((n & 63) >> 3);
    int low = transposed & 3;
    return (n & ~63) + (transposed & ~3) + ((low & 1) << 1) + (low >> 1);
}

__device__ __forceinline__ half2 Nvfp4PrefillDecodePair(uint32_t word, half2 scale) {
    uint32_t bits = ((word & 0x00080008u) << 12) | ((word & 0x00070007u) << 9);
    return __hmul2(*reinterpret_cast<half2 *>(&bits), scale);
}

// Coalesced packed reads and 16-byte output stores through a shared-memory
// XOR transpose. K is a multiple of 64; the last 256-wide tile and logical N
// may be partial. Padded Marlin rows are read but never written to the output.
__global__ void Nvfp4PrefillDequantKernel(const uint32_t *weight, const uint8_t *scales,
        half *output, int packedN, int sizeK, int countN) {
    constexpr int TileK = 256;
    __shared__ half2 tile[64 * (TileK / 2)];
    int word = threadIdx.x & 127;
    int localN = (word & 3) * 16 + (word >> 4);
    int tileN = blockIdx.x;
    int globalN = tileN * 64 + localN;
    int startK = blockIdx.y * TileK;
#pragma unroll
    for (int t = 0; t < TileK / 32; ++t) {
        int groupInTile = t * 2 + (threadIdx.x >> 7);
        int group = startK / 16 + groupInTile;
        half2 a = __float2half2_rn(0), b = a, c = a, d = a;
        if (group < sizeK / 16) {
            uint32_t q = weight[(size_t(group) * (packedN / 64) + globalN / 64) * 128 + word];
            half2 s0 = __half2half2(__ushort_as_half(uint16_t(scales[size_t(group) * packedN + Nvfp4PrefillScaleIndex(globalN)]) << 7));
            half2 s1 = __half2half2(__ushort_as_half(uint16_t(scales[size_t(group) * packedN + Nvfp4PrefillScaleIndex(globalN + 8)]) << 7));
            a = Nvfp4PrefillDecodePair(q, s0);
            b = Nvfp4PrefillDecodePair(q >> 4, s0);
            c = Nvfp4PrefillDecodePair(q >> 8, s1);
            d = Nvfp4PrefillDecodePair(q >> 12, s1);
        }
        int pairK = groupInTile * 8 + ((word >> 2) & 3);
        int swizzle0 = (localN >> 3) * 4;
        int swizzle1 = ((localN + 8) >> 3) * 4;
        tile[localN * (TileK / 2) + (pairK ^ swizzle0)] = a;
        tile[localN * (TileK / 2) + ((pairK + 4) ^ swizzle0)] = b;
        tile[(localN + 8) * (TileK / 2) + (pairK ^ swizzle1)] = c;
        tile[(localN + 8) * (TileK / 2) + ((pairK + 4) ^ swizzle1)] = d;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * TileK / 8; i += blockDim.x) {
        int row = i / (TileK / 8), vecK = i % (TileK / 8);
        int pairK = vecK * 4, swizzle = (row >> 3) * 4;
        if (tileN * 64 + row < countN && startK + vecK * 8 < sizeK) {
            reinterpret_cast<uint4 *>(output)[size_t(tileN * 64 + row) * (sizeK / 8) + startK / 8 + vecK] =
                *reinterpret_cast<uint4 *>(&tile[row * (TileK / 2) + (pairK ^ swizzle)]);
        }
    }
}
