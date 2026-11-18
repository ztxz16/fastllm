#pragma once
#include <math_constants.h>

namespace fastllm_sharded_logits {
constexpr int Capacity = FastllmCudaLogitsTopKCapacity;
constexpr int RowTile = 128;
struct Destinations {
    const float *logits[RowTile];
    FastllmCudaTopKShard *peer[RowTile];
};

// One coalesced peer write per shard and request. Transmit original logits:
// scaling determines local membership, while the owner retains the original
// rounding, global token tie order and seeded FlashInfer sampling behaviour.
__global__ void Dispatch(Destinations rows, const FastllmCudaTopKShard *selected, int offset) {
    const int row = blockIdx.x, col = threadIdx.x;
    const auto local = selected[row].candidates[col];
    // FlashInfer samples in vocabulary order. Preserve that order in the
    // compact vocabulary, including deterministic token-ID ties at top-k.
    int ids[1] = {local.token < 0 ? INT_MAX : local.token + offset};
    float values[1] = {local.token < 0 || local.score == -INFINITY ? -INFINITY : rows.logits[row][local.token]};
    using Sort = cub::BlockRadixSort<int, Capacity, 1, float>;
    __shared__ typename Sort::TempStorage scratch;
    Sort(scratch).Sort(ids, values);
    if (col == 0) rows.peer[row]->nanCount = selected[row].nanCount;
    rows.peer[row]->candidates[col] = {values[0], ids[0] == INT_MAX ? -1 : ids[0]};
}

__global__ void Unpack(const FastllmCudaTopKShard *packed, float *logits, int shards) {
    const int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    const int numericWidth = shards * Capacity, width = numericWidth + Capacity;
    if (col >= width) return;
    const auto *parts = packed + size_t(row) * shards;
    float score = -INFINITY;
    if (col < numericWidth) {
        const auto candidate = parts[col / Capacity].candidates[col % Capacity];
        if (candidate.token >= 0) score = candidate.score;
    } else {
        // FlashInfer ranks scaled NaNs above numeric scores in radix top-k,
        // then rejects them in the floating-point cutoff comparison. Retain
        // their multiplicity without displacing numeric candidates at ties.
        int nanCount = 0;
        for (int i = 0; i < shards; ++i) nanCount = min(Capacity, nanCount + max(0, parts[i].nanCount));
        if (col - numericWidth < nanCount) score = CUDART_NAN_F;
    }
    logits[size_t(row) * width + col] = score;
}

__global__ void Remap(const FastllmCudaTopKShard *packed, const float *probs,
        int *tokens, float *floatTokens, int rows, int shards) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int col = tokens[row], numericWidth = shards * Capacity;
    // FlashInfer returns index zero when a row has no sampleable mass. That
    // is the global token zero, even when the first compact candidate is not.
    int id = 0;
    if (unsigned(col) < unsigned(numericWidth) &&
        probs[size_t(row) * (numericWidth + Capacity) + col] > 0.f) {
        id = packed[size_t(row) * shards + col / Capacity].candidates[col % Capacity].token;
    }
    tokens[row] = id;
    floatTokens[row] = float(id);
}
}

bool FastllmCudaDispatchLogitsTopK(const FastllmCudaTopKRow *rows,
        FastllmCudaTopKShard *const *destinations, int batch, int vocab, int offset,
        FastllmCudaTopKShard *selected, void *workspace, size_t workspaceBytes) {
    using namespace fastllm_sharded_logits;
    if (!destinations || !selected || offset < 0 || int64_t(offset) + vocab > INT_MAX ||
        !SelectLogitsTopK(rows, batch, vocab, nullptr, workspace, workspaceBytes, nullptr, selected)) return false;
    for (int start = 0; start < batch;) {
        const int n = std::min(RowTile, batch - start);
        Destinations tile{};
        for (int i = 0; i < n; ++i) {
            tile.logits[i] = rows[start + i].logits;
            tile.peer[i] = destinations[start + i];
        }
        Dispatch<<<n, Capacity, 0, cudaStreamPerThread>>>(tile, selected + start, offset);
        start += n;
    }
    return cudaGetLastError() == cudaSuccess;
}
bool FastllmCudaUnpackLogitsTopK(const FastllmCudaTopKShard *packed,
        float *logits, int rows, int shards) {
    if (!packed || !logits || rows <= 0 || shards <= 0) return false;
    const int width = (shards + 1) * FastllmCudaLogitsTopKCapacity;
    fastllm_sharded_logits::Unpack<<<dim3((width + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
        packed, logits, shards);
    return cudaGetLastError() == cudaSuccess;
}
bool FastllmCudaRemapTopKTokens(const FastllmCudaTopKShard *packed, const float *probs,
        int *tokens, float *floatTokens, int rows, int shards) {
    if (!packed || !probs || !tokens || !floatTokens || rows <= 0 || shards <= 0) return false;
    fastllm_sharded_logits::Remap<<<(rows + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        packed, probs, tokens, floatTokens, rows, shards);
    return cudaGetLastError() == cudaSuccess;
}
