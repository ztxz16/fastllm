#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__global__ void V41CommitVerifiedPrefix(const FastllmV41VerifyCommitLayer *layers, int accepted, int rowOffset) {
    const auto layer = layers[blockIdx.x];
    const auto request = layer.requests[rowOffset];
    const auto *input = static_cast<const uint8_t *>(layer.savedKV) + (size_t)rowOffset * layer.ringRowBytes;
    auto *window = static_cast<uint8_t *>(request.windowKV);
    // Only the last window rows survive if a future configuration uses a
    // smaller ring than the accepted prefix; each destination is written once.
    const int firstRow = accepted > layer.window ? accepted - layer.window : 0;
    const int bytes = layer.ringRowBytes;
    for (int i = firstRow * bytes + threadIdx.x; i < accepted * bytes; i += blockDim.x) {
        const int row = i / bytes, column = i % bytes;
        window[((request.position + row) % layer.window) * bytes + column] = input[i];
    }
    // Full-verify plans only use compression ratios 1 and 2. The saved raw
    // stream includes the preceding uncompressed row when the start is odd.
    if (layer.ratio == 2 && (request.position + accepted) % 2) {
        const int row = 2 * request.chunkBegin + request.position % 2 + accepted - 1;
        const auto *kv = static_cast<const uint32_t *>(layer.savedRawKV) + row * 512;
        const auto *score = static_cast<const uint32_t *>(layer.savedRawScore) + row * 512;
        auto *tail = static_cast<uint32_t *>(request.rawTailKV);
        auto *tailScore = static_cast<uint32_t *>(request.rawTailScore);
        for (int column = threadIdx.x; column < 512; column += blockDim.x) {
            tail[column] = kv[column];
            tailScore[column] = score[column];
        }
    }
}
} // namespace

bool FastllmCudaV41CommitVerifiedPrefix(const FastllmV41VerifyCommitLayer *layers, int layerCount,
                                        int accepted, int forwarded, int rowOffset) {
    if (!layers || layerCount <= 0 || forwarded < 1 || forwarded > FastllmV41FullDecodeMaxBatch ||
        accepted < 1 || accepted > forwarded || rowOffset < 0 ||
        rowOffset > FastllmV41FullDecodeMaxBatch - forwarded)
        return false;
    V41CommitVerifiedPrefix<<<layerCount, 256, 0, cudaStreamPerThread>>>(layers, accepted, rowOffset);
    return cudaGetLastError() == cudaSuccess;
}
