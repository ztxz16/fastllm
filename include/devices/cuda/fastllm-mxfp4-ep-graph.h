#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// CUDA-only runtime. Kernel compilation/export is a separate developer tool.
struct FastllmMxfp4EpNativeState;
struct FastllmMxfp4EpGraph {
    void *graph = nullptr, *graphExec = nullptr, *input = nullptr, *indices = nullptr;
    void *scores = nullptr, *output = nullptr;
    int capacity = 0;
};
bool FastllmCudaMxfp4EpGraphSupported(const std::vector<int> &devices);
std::shared_ptr<FastllmMxfp4EpNativeState> FastllmCudaCreateMxfp4EpNative(
    const std::vector<int> &devices,
    const std::vector<std::vector<std::vector<uint64_t>>> &weights);
// Copies a layer into the native layout and synchronizes all source reads.
// The caller can release that layer's source tensors when this returns.
void FastllmCudaBindMxfp4EpNativeWeights(FastllmMxfp4EpNativeState &state, int layer);
std::array<void *, 2> FastllmCudaMxfp4EpPackedWeights(const FastllmMxfp4EpNativeState &state, int rank, int layer);
std::vector<FastllmMxfp4EpGraph> FastllmCudaPrepareMxfp4EpNative(
    FastllmMxfp4EpNativeState &state, int layer, int rows);
bool FastllmCudaReplayMxfp4EpGraph(const FastllmMxfp4EpGraph &graph,
    const void *input, const int32_t *indices, const float *scores, void *output,
    int rows, int hidden, int topk, const int *liveKeys);
bool FastllmCudaMxfp4EpGraphCapturing();
