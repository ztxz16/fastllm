#pragma once

#include <cuda_fp8.h>

namespace fastllm_prefill {

// Recreate the existing NUMA input bytes using only GPU-resident records.
// Cache storage may use [gate, up] rows while the host operator consumes
// interleaved pairs. No dequantization or projection arithmetic is changed.
__global__ void RestorePackedRows(const uint8_t *source, uint8_t *output,
                                 int rows, size_t pitch, bool cross) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size_t(rows) * pitch; i += size_t(gridDim.x) * blockDim.x) {
        const int row = i / pitch;
        const int original = cross ? row / 2 + (row % 2) * (rows / 2) : row;
        output[i] = source[size_t(original) * pitch + i % pitch];
    }
}

__global__ void RestoreFp8(const uint8_t *source, const float *scales, uint8_t *output,
                          int rows, int columns, int blockK, int blockM,
                          size_t pitch, bool cross) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size_t(rows) * pitch; i += size_t(gridDim.x) * blockDim.x) {
        const int row = i / pitch;
        const int original = cross ? row / 2 + (row % 2) * (rows / 2) : row;
        const int col = i % pitch, block = col / (blockM + 4), within = col % (blockM + 4);
        const int count = min(blockM, columns - block * blockM);
        if (within < count) output[i] = source[size_t(original) * columns + block * blockM + within];
        else {
            const float scale = scales[size_t(original / blockK) * ((columns + blockM - 1) / blockM) + block];
            output[i] = reinterpret_cast<const uint8_t *>(&scale)[within - count];
        }
    }
}

__global__ void RestoreNvfp4(const uint8_t *source, const uint8_t *scales,
                            const float *globalScales, uint8_t *output,
                            int rows, int columns, size_t pitch, bool cross,
                            bool compact, bool planar) {
    const int blocks = (columns + 15) / 16;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size_t(rows) * blocks; i += size_t(gridDim.x) * blockDim.x) {
        const int row = i / blocks, block = i % blocks;
        const int original = cross ? row / 2 + (row % 2) * (rows / 2) : row;
        const float global = globalScales[cross ? original / (rows / 2) : 0];
        const uint8_t scaleByte = scales[size_t(original) * blocks + block];
        constexpr int tile = fastllm::NVFP4_PLANAR_TILE_ROWS;
        const size_t tileBase = size_t(row / tile) * tile * blocks * 12;
        uint8_t *dest = output + (planar ? tileBase + size_t(row % tile) * blocks * 8 + block * 8 :
            size_t(row) * pitch + (compact ? 4 + block * 9 : block * 12));
        const int count = min(8, (columns - block * 16 + 1) / 2);
        for (int byte = 0; byte < 8; ++byte)
            dest[byte] = byte < count ? source[size_t(original) * ((columns + 1) / 2) + block * 8 + byte] : 0;
        if (compact) {
            dest[8] = scaleByte;
            if (block == 0) {
                const auto *value = reinterpret_cast<const uint8_t *>(&global);
                for (int byte = 0; byte < 4; ++byte) output[size_t(row) * pitch + byte] = value[byte];
            }
            if (block == blocks - 1)
                for (size_t byte = 4 + size_t(blocks) * 9; byte < pitch; ++byte)
                    output[size_t(row) * pitch + byte] = 0;
        } else {
            __nv_fp8_e4m3 value; value.__x = scaleByte;
            const float scale = __fmul_rn(float(value), global);
            uint8_t *scaleDest = planar ? output + tileBase + size_t(tile) * blocks * 8 +
                (size_t(row % tile) * blocks + block) * 4 : dest + 8;
            const auto *bytes = reinterpret_cast<const uint8_t *>(&scale);
            for (int byte = 0; byte < 4; ++byte) scaleDest[byte] = bytes[byte];
        }
    }
}

} // namespace fastllm_prefill
