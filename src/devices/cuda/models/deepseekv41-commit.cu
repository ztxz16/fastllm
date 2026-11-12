#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__device__ void CommitPrefix(const FastllmV41VerifyCommitLayer &layer, int accepted, int rowOffset) {
    const auto request = layer.requests[rowOffset];
    const auto *input = static_cast<const uint8_t *>(layer.savedKV) + (size_t)rowOffset * layer.ringRowBytes;
    auto *window = static_cast<uint8_t *>(request.windowKV);
    // Only the last window rows survive if a future configuration uses a
    // smaller ring than the accepted prefix; each destination is written once.
    const int firstRow = accepted > layer.window ? accepted - layer.window : 0;
    const int bytes = layer.ringRowBytes;
    for (size_t i = size_t(firstRow) * bytes + threadIdx.x; i < size_t(accepted) * bytes; i += blockDim.x) {
        const int row = i / bytes, column = i % bytes;
        window[((int64_t(request.position) + row) % layer.window) * bytes + column] = input[i];
    }
    // Full-verify plans only use compression ratios 1 and 2. The saved raw
    // stream includes the preceding uncompressed row when the start is odd.
    if (layer.ratio == 2 && (int64_t(request.position) + accepted) % 2) {
        const size_t row = size_t(2) * request.chunkBegin + request.position % 2 + accepted - 1;
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
__global__ void V41CommitVerifiedPrefix(const FastllmV41VerifyCommitLayer *layers, int accepted, int rowOffset) {
    CommitPrefix(layers[blockIdx.x], accepted, rowOffset);
}
__global__ void V41CommitVerifiedBatch(const FastllmV41VerifyCommitLayer *layers,
        int layerCount, const int *spans, int batchRows) {
    const int request = blockIdx.x / layerCount;
    const int accepted = spans[2 * request], offset = spans[2 * request + 1];
    if (accepted > 0 && offset >= 0 && offset <= batchRows - accepted)
        CommitPrefix(layers[blockIdx.x % layerCount], accepted, offset);
}
} // namespace

bool FastllmCudaV41CommitVerifiedPrefix(const FastllmV41VerifyCommitLayer *layers, int layerCount,
                                        int accepted, int forwarded, int rowOffset, int batchRows) {
    if (batchRows == 0) batchRows = forwarded; // legacy single-request call
    if (!layers || layerCount <= 0 || batchRows < 1 || forwarded < 1 || forwarded > batchRows ||
        accepted < 1 || accepted > forwarded || rowOffset < 0 || rowOffset > batchRows - forwarded)
        return false;
    V41CommitVerifiedPrefix<<<layerCount, 256, 0, cudaStreamPerThread>>>(layers, accepted, rowOffset);
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaV41CommitVerifiedBatch(const FastllmV41VerifyCommitLayer *layers,
        int layerCount, const int *spans, int requests, int batchRows) {
    if (!layers || !spans || layerCount <= 0 || requests <= 0 || batchRows < requests ||
        uint64_t(layerCount) * requests > INT32_MAX) return false;
    V41CommitVerifiedBatch<<<layerCount * requests, 256, 0, cudaStreamPerThread>>>(
        layers, layerCount, spans, batchRows);
    return cudaGetLastError() == cudaSuccess;
}
