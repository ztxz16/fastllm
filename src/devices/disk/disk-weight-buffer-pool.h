#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace fastllm {
struct DiskWeightMappingCounters {
    inline static std::atomic<uint64_t> maps{0}, unmaps{0};
};

struct DiskWeightDeleter {
    size_t mappedBytes = 0;
    void operator()(uint8_t *p) const {
        if (mappedBytes) {
#ifdef _WIN32
            VirtualFree(p, 0, MEM_RELEASE);
#else
            munmap(p, mappedBytes);
#endif
            DiskWeightMappingCounters::unmaps.fetch_add(1, std::memory_order_relaxed);
        } else delete[] p;
    }
};
using DiskWeightAllocation = std::unique_ptr<uint8_t[], DiskWeightDeleter>;

// Private to disk expert records. Splits are page-aligned, independently
// unmap-able subranges of anonymous mappings; no file mapping is borrowed.
// The byte budget is unchanged. More slots keep small split remainders useful.
class DiskWeightBufferPool {
#ifdef _WIN32
    // VirtualFree(MEM_RELEASE) requires the original allocation base. Unlike
    // mmap, independent subranges and adjacent reservations cannot be released
    // as one allocation. Retain whole reservations with exact-size reuse.
    static constexpr bool splitMappings = false;
#else
    static constexpr bool splitMappings = true;
#endif
    struct Buffer { DiskWeightAllocation data; size_t bytes = 0; uint64_t age = 0; };
    std::array<Buffer, 64> buffers;
    std::mutex mutex;
    uint64_t (*budget)();
    size_t bytes = 0;
    uint64_t retiredHeapBytes = 0, reuses = 0, allocations = 0, clock = 0;
    uint64_t splits = 0, coalesces = 0;
    void Drop(Buffer &b) {
        bytes -= b.bytes;
        if (!b.data.get_deleter().mappedBytes) retiredHeapBytes += b.bytes;
        b.data.reset(); b.bytes = 0; b.age = 0;
    }
    void TrimTo(uint64_t limit) {
        while (bytes > limit) {
            Buffer *oldest = nullptr;
            for (auto &b : buffers)
                if (b.data && (!oldest || b.age < oldest->age)) oldest = &b;
            Drop(*oldest);
        }
    }
public:
    explicit DiskWeightBufferPool(uint64_t (*budget)()) : budget(budget) {}
    DiskWeightAllocation Acquire(size_t required, bool mapped = false) {
        std::lock_guard<std::mutex> lock(mutex);
        Buffer *best = nullptr;
        for (auto &b : buffers) {
            if (!b.data || bool(b.data.get_deleter().mappedBytes) != mapped ||
                    (mapped && splitMappings ? b.bytes < required : b.bytes != required)) continue;
            if (!best || b.bytes < best->bytes) best = &b;
        }
        if (!best) { ++allocations; return nullptr; }
        ++reuses;
        bytes -= required;
        if (best->bytes == required) {
            auto data = std::move(best->data);
            best->bytes = 0; best->age = 0;
            return data;
        }
        // Mapped requests are host-page multiples (DiskGGUFAllocationBytes).
        // Keep the suffix instead of wasting capacity in a smaller RAM expert.
        auto *base = best->data.release();
        best->bytes -= required;
        best->data = DiskWeightAllocation(base + required, DiskWeightDeleter{best->bytes});
        ++splits;
        return DiskWeightAllocation(base, DiskWeightDeleter{required});
    }
    bool Retain(DiskWeightAllocation &data, size_t allocated) {
        const uint64_t limit = budget();
        if (!data || !allocated || allocated > (16ULL << 20) || allocated > limit) return false;
        std::lock_guard<std::mutex> lock(mutex);
        TrimTo(limit - allocated);
        if (splitMappings && data.get_deleter().mappedBytes) {
            // Join only free adjacent ranges. Live readers never participate.
            // Bound a coalesced eviction unit to the original per-buffer cap.
            bool joined;
            do {
                joined = false;
                for (auto &b : buffers) {
                    if (!b.data || !b.data.get_deleter().mappedBytes ||
                            b.bytes + allocated > (16ULL << 20)) continue;
                    uintptr_t a = reinterpret_cast<uintptr_t>(data.get());
                    uintptr_t other = reinterpret_cast<uintptr_t>(b.data.get());
                    if (a + allocated != other && other + b.bytes != a) continue;
                    data.release(); b.data.release();
                    const auto start = std::min(a, other);
                    allocated += b.bytes;
                    bytes -= b.bytes; b.bytes = 0; b.age = 0;
                    data = DiskWeightAllocation(reinterpret_cast<uint8_t *>(start), DiskWeightDeleter{allocated});
                    ++coalesces; joined = true;
                }
            } while (joined);
        }
        auto &slot = *std::min_element(buffers.begin(), buffers.end(),
            [](const Buffer &a, const Buffer &b) { return a.age < b.age; });
        if (slot.data) Drop(slot);
        slot.data = std::move(data); slot.bytes = allocated; slot.age = ++clock;
        bytes += allocated;
        return true;
    }
    uint64_t Clear() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto &b : buffers) Drop(b);
        const uint64_t retired = retiredHeapBytes;
        retiredHeapBytes = 0;
        return retired;
    }
    uint64_t Trim() {
        std::lock_guard<std::mutex> lock(mutex);
        TrimTo(budget());
        const uint64_t retired = retiredHeapBytes;
        retiredHeapBytes = 0;
        return retired;
    }
    template<class T> void Stats(T &s) {
        std::lock_guard<std::mutex> lock(mutex);
        s.cpuWorkspaceBytes = bytes; s.cpuBufferReuses = reuses;
        s.cpuBufferAllocations = allocations;
        s.cpuBufferSplits = splits; s.cpuBufferCoalesces = coalesces;
        s.cpuBufferMmaps = DiskWeightMappingCounters::maps.load(std::memory_order_relaxed);
        s.cpuBufferMunmaps = DiskWeightMappingCounters::unmaps.load(std::memory_order_relaxed);
    }
};
} // namespace fastllm
