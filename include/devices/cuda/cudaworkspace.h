#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>

namespace fastllm {
// A bounded CUDA arena for serial, non-graph work. Outstanding allocations
// retain their owner, so tensors may safely outlive an allocation scope.
class CudaWorkspace {
public:
    CudaWorkspace(int device, size_t bytes);
    ~CudaWorkspace();
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace &operator=(const CudaWorkspace&) = delete;
    int Device() const { return device; }
    size_t Capacity() const { return capacity; }
    size_t PeakBytes() const;
    size_t LiveBytes() const;
    void *Allocate(size_t bytes);
    void Free(void *pointer);
private:
    int device;
    size_t capacity;
    void *base = nullptr;
    mutable std::mutex mutex;
    std::map<size_t, size_t> available, allocated;
    size_t liveBytes = 0, peakBytes = 0;
};

std::shared_ptr<CudaWorkspace> SetCudaWorkspace(std::shared_ptr<CudaWorkspace> workspace);
bool TryAllocateCudaWorkspace(int device, size_t bytes, void **pointer);
bool IsCudaWorkspacePointer(void *pointer);
bool TryFreeCudaWorkspace(void *pointer);

class CudaWorkspaceScope {
public:
    explicit CudaWorkspaceScope(std::shared_ptr<CudaWorkspace> workspace)
        : previous(SetCudaWorkspace(std::move(workspace))) {}
    ~CudaWorkspaceScope() { SetCudaWorkspace(std::move(previous)); }
    CudaWorkspaceScope(const CudaWorkspaceScope&) = delete;
    CudaWorkspaceScope &operator=(const CudaWorkspaceScope&) = delete;
private:
    std::shared_ptr<CudaWorkspace> previous;
};
}
