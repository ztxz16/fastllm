#pragma once

#include <cstddef>
#include <functional>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace fastllm {
class CudaWorkspaceExhausted : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
// A bounded CUDA arena. Allocate/release outside graph capture and order device
// work before reusing freed storage. Outstanding allocations retain their owner.
class CudaWorkspace {
public:
    CudaWorkspace(int device, size_t bytes);
    // A growable pool can reduce a new slab under memory pressure, down to
    // the actual allocation size. Fixed-budget arenas keep the two-arg API.
    CudaWorkspace(int device, size_t bytes, size_t minimumBytes);
    struct Region { void *data; size_t bytes; };
    // Borrow immutable-address storage. The release callback restores its
    // owner only after every allocation and CUDA completion fence retires.
    CudaWorkspace(int device, std::vector<Region> regions, std::function<void()> release);
    ~CudaWorkspace();
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace &operator=(const CudaWorkspace&) = delete;
    int Device() const { return device; }
    size_t Capacity() const { return capacity; }
    size_t PeakBytes() const;
    size_t LiveBytes() const;
    void *Allocate(size_t bytes);
    void Free(void *pointer);
    void FreeAfterStream(void *pointer, void *stream, bool currentThread = false);
    bool Borrowed() const { return bool(release); }
private:
    int device;
    size_t capacity;
    void *base = nullptr;
    std::function<void()> release;
    std::vector<std::pair<void *, void *>> pending;
    void FreeLocked(void *pointer);
    mutable std::mutex mutex;
    std::map<size_t, size_t> available, allocated;
    size_t liveBytes = 0, peakBytes = 0;
};

std::shared_ptr<CudaWorkspace> SetCudaWorkspace(std::shared_ptr<CudaWorkspace> workspace);
bool TryAllocateCudaWorkspace(int device, size_t bytes, void **pointer);
// Explicit allocation for persistent storage, without redirecting scratch
// allocations on the calling thread. The allocation retains its arena owner.
void *AllocateCudaWorkspace(const std::shared_ptr<CudaWorkspace> &workspace, size_t bytes);
void *AllocateCudaWorkspace(const std::shared_ptr<CudaWorkspace> &workspace, size_t bytes,
                            bool reportExhaustion);
bool IsCudaWorkspacePointer(void *pointer);
bool TryFreeCudaWorkspace(void *pointer);
bool TryFreeCudaWorkspaceAfterStream(void *pointer, void *stream);

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
