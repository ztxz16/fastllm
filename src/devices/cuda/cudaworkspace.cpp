#include "devices/cuda/cudaworkspace.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fastllm {
namespace {
struct WorkspaceOwners {
    std::mutex mutex;
    std::map<void*, std::shared_ptr<CudaWorkspace>> allocations;
};
WorkspaceOwners &Owners() {
    // CUDA tensors in other translation units can be destroyed during process
    // shutdown. Keep the registry available regardless of static teardown order.
    static auto *owners = new WorkspaceOwners;
    return *owners;
}
thread_local std::shared_ptr<CudaWorkspace> currentWorkspace;
std::atomic<size_t> workspacePointers{0};
constexpr size_t alignment = 256;
}

CudaWorkspace::CudaWorkspace(int device, size_t bytes)
    : device(device), capacity(bytes - bytes % alignment) {
    if (capacity == 0) throw std::invalid_argument("CUDA workspace must contain at least 256 bytes");
    const int previous = FastllmCudaGetDevice();
    FastllmCudaSetDevice(device);
    base = FastllmCudaDirectMalloc(capacity);
    FastllmCudaSetDevice(previous);
    if (!base) throw std::runtime_error("Cannot allocate multimodal CUDA workspace before KV cache sizing");
    available.emplace(0, capacity);
}

CudaWorkspace::~CudaWorkspace() {
    const int previous = FastllmCudaGetDevice();
    FastllmCudaSetDevice(device);
    ForceDeviceSync();
    FastllmCudaDirectFree(base);
    FastllmCudaSetDevice(previous);
}

size_t CudaWorkspace::PeakBytes() const {
    std::lock_guard<std::mutex> guard(mutex);
    return peakBytes;
}
size_t CudaWorkspace::LiveBytes() const {
    std::lock_guard<std::mutex> guard(mutex);
    return liveBytes;
}

void *CudaWorkspace::Allocate(size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max() - alignment + 1) return nullptr;
    bytes = std::max(alignment, (bytes + alignment - 1) / alignment * alignment);
    std::lock_guard<std::mutex> guard(mutex);
    auto best = available.end();
    for (auto it = available.begin(); it != available.end(); ++it) {
        if (it->second >= bytes && (best == available.end() || it->second < best->second)) best = it;
    }
    if (best == available.end()) return nullptr;
    const size_t offset = best->first, remaining = best->second - bytes;
    allocated.emplace(offset, bytes);
    available.erase(best);
    if (remaining) available.emplace(offset + bytes, remaining);
    liveBytes += bytes;
    peakBytes = std::max(peakBytes, liveBytes);
    return static_cast<unsigned char*>(base) + offset;
}

void CudaWorkspace::Free(void *pointer) {
    std::lock_guard<std::mutex> guard(mutex);
    size_t offset = static_cast<unsigned char*>(pointer) - static_cast<unsigned char*>(base);
    auto live = allocated.find(offset);
    if (live == allocated.end()) throw std::logic_error("Invalid CUDA workspace release");
    size_t bytes = live->second;
    liveBytes -= bytes;
    allocated.erase(live);
    auto next = available.lower_bound(offset);
    if (next != available.begin()) {
        auto previous = std::prev(next);
        if (previous->first + previous->second == offset) {
            offset = previous->first;
            bytes += previous->second;
            available.erase(previous);
        }
    }
    if (next != available.end() && offset + bytes == next->first) {
        bytes += next->second;
        available.erase(next);
    }
    available.emplace(offset, bytes);
}

std::shared_ptr<CudaWorkspace> SetCudaWorkspace(std::shared_ptr<CudaWorkspace> workspace) {
    auto previous = std::move(currentWorkspace);
    currentWorkspace = std::move(workspace);
    return previous;
}

bool TryAllocateCudaWorkspace(int device, size_t bytes, void **pointer) {
    auto owner = currentWorkspace;
    if (!owner || owner->Device() != device) return false;
    *pointer = owner->Allocate(bytes);
    if (*pointer) {
        auto &owners = Owners();
        std::lock_guard<std::mutex> guard(owners.mutex);
        owners.allocations.emplace(*pointer, std::move(owner));
        workspacePointers.fetch_add(1, std::memory_order_release);
    } else {
        fprintf(stderr, "[Vision] CUDA workspace exhausted: request %.2f MiB, live %.2f MiB, capacity %.2f MiB.\n",
                bytes / 1048576.0, owner->LiveBytes() / 1048576.0, owner->Capacity() / 1048576.0);
    }
    return true;
}

bool IsCudaWorkspacePointer(void *pointer) {
    if (workspacePointers.load(std::memory_order_acquire) == 0) return false;
    auto &owners = Owners();
    std::lock_guard<std::mutex> guard(owners.mutex);
    return owners.allocations.count(pointer) != 0;
}

bool TryFreeCudaWorkspace(void *pointer) {
    if (workspacePointers.load(std::memory_order_acquire) == 0) return false;
    std::shared_ptr<CudaWorkspace> owner;
    {
        auto &owners = Owners();
        std::lock_guard<std::mutex> guard(owners.mutex);
        auto it = owners.allocations.find(pointer);
        if (it == owners.allocations.end()) return false;
        owner = std::move(it->second);
        owners.allocations.erase(it);
        workspacePointers.fetch_sub(1, std::memory_order_release);
    }
    owner->Free(pointer);
    return true;
}
}
