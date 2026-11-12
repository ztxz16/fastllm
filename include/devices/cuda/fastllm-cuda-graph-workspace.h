#ifndef FASTLLM_CUDA_GRAPH_WORKSPACE_H
#define FASTLLM_CUDA_GRAPH_WORKSPACE_H
#include <cstddef>
#include <vector>

// Record one eager warmup's distinct physical blocks, keeping the largest
// request for each block reused during that scope. Nested scopes restore the
// previous recorder; recording never changes allocation or free behavior.
void *FastllmCudaBeginAllocationRecording(std::vector<size_t> *requests);
void FastllmCudaEndAllocationRecording(void *recording);
class FastllmCudaAllocationRecorder {
    void *recording;
public:
    explicit FastllmCudaAllocationRecorder(std::vector<size_t> *requests)
        : recording(FastllmCudaBeginAllocationRecording(requests)) {}
    ~FastllmCudaAllocationRecorder() { FastllmCudaEndAllocationRecording(recording); }
    FastllmCudaAllocationRecorder(const FastllmCudaAllocationRecorder &) = delete;
    FastllmCudaAllocationRecorder &operator=(const FastllmCudaAllocationRecorder &) = delete;
};
// Warmup must exercise the same eligible implementations as capture. This
// scope excludes eager-only kernels without suppressing allocation, tuning or
// immutable-weight preparation. Nested scopes preserve the caller's policy.
bool FastllmCudaGraphPreparationActive();
bool FastllmCudaSetGraphPreparation(bool active);
class FastllmCudaGraphPreparation {
    bool previous;
public:
    explicit FastllmCudaGraphPreparation(bool active = true)
        : previous(FastllmCudaSetGraphPreparation(active || FastllmCudaGraphPreparationActive())) {}
    ~FastllmCudaGraphPreparation() { FastllmCudaSetGraphPreparation(previous); }
    FastllmCudaGraphPreparation(const FastllmCudaGraphPreparation &) = delete;
    FastllmCudaGraphPreparation &operator=(const FastllmCudaGraphPreparation &) = delete;
};
// Outside capture, reserve all requested blocks simultaneously, then return
// them to the pool. Capture can subsequently pin its actual scratch addresses.
// Between segments this may reuse idle blocks protected by the current graph
// transaction: reservation never reads/writes them. Other graph owners stay pinned.
// A capacity failure preserves live allocations and CUDA error flags.
bool FastllmCudaReserveGraphWorkspace(const std::vector<size_t> &requests);
// Captures in one group may alias transient scratch. The caller must execute
// those graphs serially (or establish stream dependencies), prepare persistent
// outputs before capture, and keep the identity alive until all graphs die.
// Only reservation/capture can borrow pinned scratch; eager/prefill allocations
// and live Data storage remain isolated. Nested scopes restore the prior group.
const void *FastllmCudaSetGraphScratchGroup(const void *group);
class FastllmCudaGraphScratchGroup {
    const void *previous;
public:
    explicit FastllmCudaGraphScratchGroup(const void *group)
        : previous(FastllmCudaSetGraphScratchGroup(group)) {}
    ~FastllmCudaGraphScratchGroup() { FastllmCudaSetGraphScratchGroup(previous); }
    FastllmCudaGraphScratchGroup(const FastllmCudaGraphScratchGroup &) = delete;
    FastllmCudaGraphScratchGroup &operator=(const FastllmCudaGraphScratchGroup &) = delete;
};
#endif
