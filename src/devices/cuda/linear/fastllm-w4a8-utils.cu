#include "fastllm-w4a8-utils.cuh"

#include <algorithm>
#include <array>
#include <cstdio>

#include <cuda_runtime.h>

namespace fastllm::cuda::w4a8 {

__constant__ uint8_t kInt4EncodeLut[256];

__global__ void EncodeInt4Kernel(const uint8_t *input, uint8_t *output,
                                 size_t byteCount) {
    constexpr size_t vectorBytes = sizeof(uint4);
    size_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    size_t threadCount = size_t(gridDim.x) * blockDim.x;
    size_t vectorCount = byteCount / vectorBytes;

    for (size_t index = thread; index < vectorCount; index += threadCount) {
        uint4 value = reinterpret_cast<const uint4 *>(input)[index];
        uint8_t *bytes = reinterpret_cast<uint8_t *>(&value);
#pragma unroll
        for (int i = 0; i < int(vectorBytes); ++i) {
            bytes[i] = kInt4EncodeLut[bytes[i]];
        }
        reinterpret_cast<uint4 *>(output)[index] = value;
    }
}

static bool UploadInt4EncodeLut() {
    std::array<uint8_t, 256> lut{};
    auto encodeNibble = [](uint8_t value) -> uint8_t {
        return (value == 0 || (value & 0x8)) ? value : uint8_t(8 - value);
    };
    for (int byte = 0; byte < 256; ++byte) {
        uint8_t low = byte & 0xF;
        uint8_t high = (byte >> 4) & 0xF;
        lut[byte] = uint8_t((encodeNibble(high) << 4) | encodeNibble(low));
    }
    return cudaMemcpyToSymbol(kInt4EncodeLut, lut.data(), lut.size(), 0,
                              cudaMemcpyHostToDevice) == cudaSuccess;
}

bool EncodeInt4ForCutlass(const cutlass::int4b_t *input,
                          cutlass::int4b_t *output,
                          size_t elementCount) {
    if (!UploadInt4EncodeLut()) {
        return false;
    }

    static_assert(sizeof(cutlass::int4b_t::Storage) == 1,
                  "int4 storage must be one byte");
    size_t byteCount = elementCount / 2;
    constexpr int blockSize = 256;
    size_t vectorCount = byteCount / sizeof(uint4);
    int gridSize = int((vectorCount + blockSize - 1) / blockSize);
    gridSize = std::max(gridSize, 1);

    EncodeInt4Kernel<<<gridSize, blockSize>>>(
        reinterpret_cast<const uint8_t *>(input),
        reinterpret_cast<uint8_t *>(output), byteCount);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::printf("EncodeInt4Kernel launch error: %s (%d)\n",
                    cudaGetErrorString(error), error);
        return false;
    }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::printf("EncodeInt4Kernel runtime error: %s (%d)\n",
                    cudaGetErrorString(error), error);
        return false;
    }
    return true;
}

} // namespace fastllm::cuda::w4a8
