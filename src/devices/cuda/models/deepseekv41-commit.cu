#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__global__ void V41CommitVerifiedPrefix(const FastllmV41VerifyCommitLayer *layers, int accepted) {
    const auto layer = layers[blockIdx.x];
    const auto request = layer.requests[0];
    const auto *input = static_cast<const uint16_t *>(layer.savedKV);
    auto *window = static_cast<uint16_t *>(request.windowKV);
    // Only the last window rows survive if a future configuration uses a
    // smaller ring than the accepted prefix; each destination is written once.
    const int firstRow = accepted > layer.window ? accepted - layer.window : 0;
    for (int i = firstRow * 512 + threadIdx.x; i < accepted * 512; i += blockDim.x) {
        const int row = i / 512, column = i % 512;
        window[((request.position + row) % layer.window) * 512 + column] = input[i];
    }
    // Full-verify plans only use compression ratios 1 and 2. The saved raw
    // stream includes the preceding uncompressed row when the start is odd.
    if (layer.ratio == 2 && (request.position + accepted) % 2) {
        const int row = request.position % 2 + accepted - 1;
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
                                        int accepted, int forwarded) {
    if (!layers || layerCount <= 0 || forwarded < 1 || forwarded > 8 || accepted < 1 || accepted > forwarded)
        return false;
    V41CommitVerifiedPrefix<<<layerCount, 256, 0, cudaStreamPerThread>>>(layers, accepted);
    return cudaGetLastError() == cudaSuccess;
}
