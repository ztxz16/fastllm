#include "devices/cuda/cudaworkspace.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace fastllm {
namespace {
struct WorkspaceOwners {
    std::mutex mutex;
    std::unordered_map<void*, std::shared_ptr<CudaWorkspace>> allocations;
};
WorkspaceOwners &Owners(const void *pointer) {
    // CUDA tensors in other translation units can be destroyed during process
    // shutdown. Keep the registry available regardless of static teardown order.
    static auto *owners = new std::array<WorkspaceOwners, 64>;
    // Every ordinary CUDA free also checks this registry. A single mutex
    // serializes TP ranks even when none of their scratch tensors are arena
    // slices. Mix high address bits: CUDA allocations are heavily aligned.
    uint64_t bits = reinterpret_cast<std::uintptr_t>(pointer);
    bits ^= bits >> 33;
    bits *= 0xff51afd7ed558ccdULL;
    bits ^= bits >> 33;
    return (*owners)[bits % owners->size()];
}
thread_local std::shared_ptr<CudaWorkspace> currentWorkspace;
std::atomic<size_t> workspacePointers{0};
constexpr size_t alignment = 256;
}

CudaWorkspace::CudaWorkspace(int device, size_t bytes)
    : CudaWorkspace(device, bytes, 0) {}

CudaWorkspace::CudaWorkspace(int device, size_t bytes, size_t minimumBytes)
    : device(device), capacity(bytes - bytes % alignment) {
    if (capacity == 0) throw std::invalid_argument("CUDA workspace must contain at least 256 bytes");
    if (minimumBytes > capacity) throw std::invalid_argument("Invalid CUDA workspace minimum");
    const size_t minimum = minimumBytes ? (minimumBytes + alignment - 1) / alignment * alignment : capacity;
    const int previous = FastllmCudaGetDevice();
    FastllmCudaSetDevice(device);
    if (minimumBytes) {
        for (;;) {
            const auto result = FastllmCudaTryDirectMalloc(&base, capacity);
            if (result != FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE || capacity == minimum) break;
            capacity = std::max(minimum, capacity / 2 / alignment * alignment);
        }
    } else {
        base = FastllmCudaDirectMalloc(capacity);
    }
    FastllmCudaSetDevice(previous);
    if (!base) throw std::runtime_error("Cannot allocate CUDA workspace");
    available.emplace(reinterpret_cast<std::uintptr_t>(base), capacity);
}

CudaWorkspace::CudaWorkspace(int device, std::vector<Region> regions,
                             std::function<void()> release)
    : device(device), capacity(0), release(std::move(release)) {
    if (!this->release || regions.empty()) throw std::invalid_argument("invalid borrowed CUDA workspace");
    for (const auto &region : regions) {
        const auto address = reinterpret_cast<std::uintptr_t>(region.data);
        if (!address || address % alignment || !region.bytes || region.bytes % alignment ||
            region.bytes > std::numeric_limits<size_t>::max() - address)
            throw std::invalid_argument("unaligned borrowed CUDA workspace");
        auto next = available.lower_bound(address);
        if ((next != available.end() && address + region.bytes > next->first) ||
            (next != available.begin() && std::prev(next)->first + std::prev(next)->second > address))
            throw std::invalid_argument("overlapping borrowed CUDA workspace");
        available.emplace(address, region.bytes);
        capacity += region.bytes;
    }
}

CudaWorkspace::~CudaWorkspace() {
    const int previous = FastllmCudaGetDevice();
    FastllmCudaSetDevice(device);
    if (release) {
        if (GetFastllmEnv().printProfile) std::fprintf(stderr,
            "[Fastllm] Borrowed workspace cuda:%d: peak %.2f MiB / %.2f MiB.\n",
            device, peakBytes / 1048576.0, capacity / 1048576.0);
        for (auto &item : pending) {
            FastllmCudaEventSynchronize(item.second);
            FastllmCudaEventDestroy(item.second);
        }
        release();
    } else {
        ForceDeviceSync();
        FastllmCudaDirectFree(base);
    }
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
    // Reuse across host/GPU worker streams with device-side dependencies;
    // do not put a host synchronization on every temporary tensor release.
    size_t pendingCount = 0;
    for (auto item : pending) {
        if (item.first) {
            FreeLocked(item.first);
            item.first = nullptr;
        }
        if (FastllmCudaEventComplete(item.second)) {
            FastllmCudaEventDestroy(item.second);
        } else {
            // A later allocation may use a different stream and another slice
            // of this released range. Keep the fence until it completes.
            FastllmCudaCurrentThreadStreamWaitEvent(item.second);
            pending[pendingCount++] = item;
        }
    }
    pending.resize(pendingCount);
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
    return reinterpret_cast<void *>(offset);
}

void CudaWorkspace::Free(void *pointer) {
    std::lock_guard<std::mutex> guard(mutex);
    FreeLocked(pointer);
}

void CudaWorkspace::FreeAfterStream(void *pointer, void *stream, bool currentThread) {
    void *event = FastllmCudaEventCreate();
    if (currentThread) FastllmCudaEventRecordCurrentThread(event);
    else FastllmCudaEventRecord(event, stream);
    std::lock_guard<std::mutex> guard(mutex);
    pending.emplace_back(pointer, event);
}

void CudaWorkspace::FreeLocked(void *pointer) {
    size_t offset = reinterpret_cast<std::uintptr_t>(pointer);
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
    *pointer = AllocateCudaWorkspace(owner, bytes);
    return true;
}

void *AllocateCudaWorkspace(const std::shared_ptr<CudaWorkspace> &owner, size_t bytes) {
    return AllocateCudaWorkspace(owner, bytes, true);
}

void *AllocateCudaWorkspace(const std::shared_ptr<CudaWorkspace> &owner, size_t bytes,
                            bool reportExhaustion) {
    void *pointer = owner->Allocate(bytes);
    if (pointer) {
        auto &owners = Owners(pointer);
        std::lock_guard<std::mutex> guard(owners.mutex);
        owners.allocations.emplace(pointer, owner);
        workspacePointers.fetch_add(1, std::memory_order_release);
    } else if (reportExhaustion) {
        fprintf(stderr, "[Fastllm] CUDA arena exhausted: request %.2f MiB, live %.2f MiB, capacity %.2f MiB.\n",
                bytes / 1048576.0, owner->LiveBytes() / 1048576.0, owner->Capacity() / 1048576.0);
    }
    return pointer;
}

bool IsCudaWorkspacePointer(void *pointer) {
    if (workspacePointers.load(std::memory_order_acquire) == 0) return false;
    auto &owners = Owners(pointer);
    std::lock_guard<std::mutex> guard(owners.mutex);
    return owners.allocations.count(pointer) != 0;
}

static bool ReleaseWorkspacePointer(void *pointer, void *stream, bool afterStream) {
    if (workspacePointers.load(std::memory_order_acquire) == 0) return false;
    std::shared_ptr<CudaWorkspace> owner;
    {
        auto &owners = Owners(pointer);
        std::lock_guard<std::mutex> guard(owners.mutex);
        auto it = owners.allocations.find(pointer);
        if (it == owners.allocations.end()) return false;
        owner = std::move(it->second);
        owners.allocations.erase(it);
        workspacePointers.fetch_sub(1, std::memory_order_release);
    }
    if (owner->Borrowed()) owner->FreeAfterStream(pointer, stream, !afterStream);
    else {
        if (afterStream) FastllmCudaStreamSynchronize(stream);
        owner->Free(pointer);
    }
    return true;
}
bool TryFreeCudaWorkspace(void *pointer) {
    return ReleaseWorkspacePointer(pointer, nullptr, false);
}
bool TryFreeCudaWorkspaceAfterStream(void *pointer, void *stream) {
    return ReleaseWorkspacePointer(pointer, stream, true);
}
}
