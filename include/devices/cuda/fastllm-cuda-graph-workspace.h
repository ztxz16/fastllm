#ifndef FASTLLM_CUDA_GRAPH_WORKSPACE_H
#define FASTLLM_CUDA_GRAPH_WORKSPACE_H
#include <cstddef>
#include <vector>

// Record one eager warmup's pooled allocation requests. Nested scopes restore
// the previous recorder; recording never changes allocation or free behavior.
std::vector<size_t> *FastllmCudaSetAllocationRecorder(std::vector<size_t> *requests);
class FastllmCudaAllocationRecorder {
    std::vector<size_t> *previous;
public:
    explicit FastllmCudaAllocationRecorder(std::vector<size_t> *requests)
        : previous(FastllmCudaSetAllocationRecorder(requests)) {}
    ~FastllmCudaAllocationRecorder() { FastllmCudaSetAllocationRecorder(previous); }
    FastllmCudaAllocationRecorder(const FastllmCudaAllocationRecorder &) = delete;
    FastllmCudaAllocationRecorder &operator=(const FastllmCudaAllocationRecorder &) = delete;
};
// Outside capture, reserve all requested blocks simultaneously, then return
// them to the pool. Capture can subsequently pin its actual scratch addresses.
// A capacity failure preserves live allocations and CUDA error flags.
bool FastllmCudaReserveGraphWorkspace(const std::vector<size_t> &requests);
#endif
