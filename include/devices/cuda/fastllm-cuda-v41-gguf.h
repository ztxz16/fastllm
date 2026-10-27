#pragma once
#include "fastllm.h"
#include <functional>
#include <future>

// Resident native V4.1 GGUF MoE. GPU routes and immutable weight pointers;
// indexed GEMV for <=32 rows, grouped MMQ for supported larger batches.
// Both preserve block-32 activation quantization and FP32 expert accumulation.
bool FastllmCudaMergeMOEV41GGUFResident(
    const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &activation,
    fastllm::Data &workspace, fastllm::Data &output, fastllm::Data **weights,
    int weightsBatch, const fastllm::Data &indices, const fastllm::Data &scores,
    float sharedScale, float swigluLimit, bool compactEpRoutes = false);

// Read-only admission for a previously warmed resident decode shape. The
// model owns and validates graph workspace addresses; weight retirement
// invalidates the resident pointer table and this readiness check together.
bool FastllmCudaV41GGUFGraphReady(const fastllm::Data *layerKey, int rows, int topk);

// Internal launcher; caller validates layouts and owns all scratch tensors.
bool FastllmCudaRunV41GGUF(
    const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &activation,
    fastllm::Data &workspace, fastllm::Data &output, const void *weightPointers,
    const int32_t *indices, const float *scores, int gateType, int downType,
    int hidden, int inter, int experts, int topk, float swigluLimit,
    bool compactEpRoutes = false);

// Native CUDA activation semantics (Q8_1), distinct from NUMA's Q8_K path.
size_t FastllmCudaV41GGUFGroupedWorkspaceBytes(
    int gateType, int downType, int rows, int hidden, int inter, int experts, int topk);
bool FastllmCudaV41GGUFGroupedProjection(
    bool gate, int type, const void *inputQ8, void *outputBF16,
    const void *weightPointers, const int32_t *indices, void *workspace,
    int rows, int hidden, int inter, int experts, int topk);

// Canonical packed records restored from NUMA shards, with per-layer slot offsets.
struct FastllmCudaMoeGGUFCacheView;
bool FastllmCudaMoeV41CanonicalGGUFCacheCompute(
    const fastllm::Data &input, fastllm::Data &activation,
    const FastllmCudaMoeGGUFCacheView &view, const float *scores,
    int topk, float swigluLimit, float *perExpert);
size_t FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(int hidden, int inter);
bool FastllmCudaV41GGUFSupported(int gateType, int downType);

// Resident CUDA pointers remain live until return. Read jobs fill bounded,
// page-aligned pinned slots and return the canonical payload within each slot.
// The caller waits for DMA before reusing a host slot, and drains read jobs on
// failure before releasing their storage. Cache updates run on the caller.
struct FastllmCudaV41DiskStaging {
    void *data = nullptr;
    size_t bytes = 0;
    ~FastllmCudaV41DiskStaging();
};
struct FastllmCudaV41DiskExperts {
    FastllmCudaV41DiskStaging *staging = nullptr;
    std::vector<const void *> residents; // gate/down pairs, without shared slot
    std::vector<bool> releaseFirst; // consume cold RAM records before recycling their slots
    std::function<std::future<const uint8_t *>(int, int, uint8_t *, size_t)> read;
    std::function<void(int)> release;
};
bool FastllmCudaMergeMOEV41GGUFDisk(const fastllm::Data &input,
    fastllm::Data &workspace, fastllm::Data &output, fastllm::Data **weights,
    int experts, const int32_t *indices, const float *scores, int topk,
    float swigluLimit, const FastllmCudaV41DiskExperts &source);
