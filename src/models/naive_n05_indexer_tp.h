#pragma once
#include "utils/utils.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"

namespace fastllm {
namespace {
// Every rank scores the same number of rows for AllGather. Overlap only the
// last shard's excess rows; each shard remains above the prefill dispatch gate.
int RaggedIndexerRows(int length, int ranks) { return length / ranks + (length % ranks != 0); }
void ScoreRaggedIndexer(const Data &query, const Data &weights, const Data &key, int past, int topK, bool fp8, int rank,
                        int ranks, Data &gathered) {
    const int length = query.dims[1], rows = RaggedIndexerRows(length, ranks);
    const int heads = weights.dims.back(), width = query.dims.back();
    AssertInFastLLM(rows <= length && rank >= 0 && rank < ranks, "Invalid ragged Indexer partition.");
    const int first = std::min(rank * rows, length - rows);
    gathered.dataType = INT32;
    gathered.Resize({ranks * rows, topK});
    gathered.ToDevice(query.dataDevice, query.dataDeviceIds, false);
    gathered.Allocate(false);
    Data localQ, localWeights, localIndices;
    localQ.Resize({1, rows, width});
    localQ.FakeFrom(query, (size_t)first * width * query.unitSize);
    localWeights.Resize({1, rows, heads});
    localWeights.FakeFrom(weights, (size_t)first * heads * weights.unitSize);
    localIndices.Resize({rows, topK});
    localIndices.FakeFrom(gathered, (size_t)rank * rows * topK * sizeof(int));
    // The chronological key view is local, including context-sharded caches.
    FastllmCudaNaiveIndexer(localQ, localWeights, key, heads, width / heads, past + first, topK, fp8, localIndices);
}
void CompactRaggedIndexer(const Data &gathered, int length, int ranks, Data &indices) {
    const int rows = RaggedIndexerRows(length, ranks), topK = gathered.dims.back();
    const size_t rowBytes = (size_t)topK * sizeof(int);
    indices.dataType = INT32;
    indices.Resize({length, topK});
    indices.ToDevice(gathered.dataDevice, gathered.dataDeviceIds, false);
    indices.Allocate(false);
    std::vector<void *> dsts;
    std::vector<const void *> srcs;
    std::vector<size_t> sizes;
    for (int rank = 0; rank < ranks; ++rank) {
        const int first = rank * rows, count = std::min(rows, length - first);
        if (count <= 0)
            break;
        const int offset = first - std::min(first, length - rows);
        dsts.push_back((uint8_t *)indices.cudaData + first * rowBytes);
        srcs.push_back((const uint8_t *)gathered.cudaData + (rank * rows + offset) * rowBytes);
        sizes.push_back(count * rowBytes);
    }
    AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dsts.data(), srcs.data(), sizes.data(),
                                                                             (int)sizes.size()),
                    "Ragged Indexer copy failed.");
}
} // namespace
} // namespace fastllm
