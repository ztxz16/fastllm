#include "internal.h"
#include <algorithm>

namespace fastkv {
struct BufferPool::State {
    std::mutex mutex;
    std::vector<Buffer> idle;
    std::condition_variable changed;
    size_t limit, allocated = 0;
    BufferAllocator allocator;
    std::map<int, size_t> deviceLimits, deviceAllocated;
    std::map<int, size_t> preferred;
    bool peerAccessible = false;
    size_t checkedOut = 0;
    bool closed = false;
    static bool Whole(const Buffer &b) {
        return b.Size() == b.AllocationSize();
    }
    void MergeIdle(Buffer &buffer) {
        for (auto it = idle.begin(); it != idle.end();) {
            if (it->Id() != buffer.Id() ||
                (it->Offset() + it->Size() != buffer.Offset() && buffer.Offset() + buffer.Size() != it->Offset())) {
                ++it;
                continue;
            }
            buffer.offset = std::min(buffer.Offset(), it->Offset());
            buffer.length += it->Size();
            it = idle.erase(it);
        }
    }
    explicit State(size_t bytes) : limit(bytes) {}
    void TrimIdle() {
        std::vector<Buffer> released;
        std::map<int, size_t> bytes;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto it = idle.begin(); it != idle.end();) {
                if (!preferred.empty() && !Whole(*it)) { ++it; continue; }
                bytes[it->Device()] += it->Size();
                released.push_back(std::move(*it));
                it = idle.erase(it);
            }
        }
        // Keep capacity charged until CUDA storage and MR attachments retire.
        released.clear();
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const auto &item : bytes) {
                allocated -= item.second;
                deviceAllocated[item.first] -= item.second;
            }
        }
        changed.notify_all();
    }
    void Recycle(Buffer buffer, bool returnedLease = true) noexcept {
        const size_t bytes = buffer.Size();
        const int device = buffer.Device();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (returnedLease) --checkedOut;
            if (!closed) {
                try {
                    if (!preferred.empty()) MergeIdle(buffer);
                    idle.push_back(std::move(buffer));
                    changed.notify_all();
                    return;
                } catch (...) {}
            }
        }
        // GPU deallocation and MR destruction must never hold the pool mutex.
        // Keep their capacity charged until the physical allocation is gone.
        buffer = {};
        {
            std::lock_guard<std::mutex> lock(mutex);
            allocated -= bytes;
            deviceAllocated[device] -= bytes;
        }
        changed.notify_all();
    }
};
BufferPool::BufferPool(size_t limit, std::map<int, size_t> limits, BufferAllocator allocator)
    : state(std::make_shared<State>(limit)) {
    state->deviceLimits = std::move(limits);
    state->allocator = allocator ? std::move(allocator) : Buffer::Allocate;
}
size_t BufferPool::Limit(int device) const {
    auto found = state->deviceLimits.find(device);
    return found == state->deviceLimits.end() ? state->limit : std::min(state->limit, found->second);
}
static size_t AllocationAlignment(size_t bytes) {
    return bytes < (1 << 20) ? 4096 : (2 << 20);
}
static size_t Rounded(size_t bytes, size_t limit) {
    const size_t alignment = AllocationAlignment(bytes);
    return std::min(CheckedAdd(bytes, alignment - 1) / alignment * alignment, limit);
}
void BufferPool::Preallocate(const std::vector<int> &devices, bool peerAccessible) {
    Require(!devices.empty() && state->preferred.empty(), "FastKV: invalid pool preallocation");
    // A byte limit is an upper bound. If another process has consumed device
    // memory since the free-memory query, warm a smaller share and retain the
    // normal on-demand path for larger messages.
    for (size_t share = state->limit / devices.size(); share >= 4096; share /= 2) {
        std::map<int, size_t> preferred;
        std::vector<std::pair<int, size_t>> allocations;
        for (int device : devices) {
            size_t bytes = std::min(Limit(device), share);
            const size_t alignment = AllocationAlignment(bytes);
            bytes -= bytes % alignment;
            if (!bytes) continue;
            if (preferred.emplace(device, bytes).second) allocations.emplace_back(device, bytes);
        }
        if (allocations.empty()) continue;
        try {
            auto buffers = AcquireBatch(allocations);
            state->idle.reserve(state->idle.size() + 2 * buffers.size());
            for (auto &buffer : buffers) buffer.MarkPreallocated();
            state->preferred.swap(preferred);
            state->peerAccessible = peerAccessible && state->preferred.size() > 1;
            return;
        } catch (const CapacityBusy &) {
            // AcquireBatch has rolled back every device in this attempt.
        }
    }
}
void BufferPool::CheckCapacity(const std::vector<std::pair<int, size_t>> &allocations) const {
    size_t total = 0;
    std::map<int, size_t> devices;
    for (const auto &item : allocations) {
        const size_t limit = Limit(item.first);
        Require(item.second <= limit, "FastKV: payload exceeds device buffer budget");
        const size_t bytes = Rounded(item.second, limit);
        total = CheckedAdd(total, bytes);
        devices[item.first] = CheckedAdd(devices[item.first], bytes);
        Require(devices[item.first] <= limit, "FastKV: message exceeds device buffer budget");
    }
    Require(total <= state->limit, "FastKV: message exceeds buffer budget");
}
void BufferPool::WaitForRelease() {
    std::unique_lock<std::mutex> lock(state->mutex);
    Require(!state->closed, "FastKV: buffer pool stopped");
    state->changed.wait_for(lock, std::chrono::milliseconds(20));
}
BufferPool::~BufferPool() { Clear(); }
void BufferPool::Clear() {
    std::vector<Buffer> released;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->closed = true;
        released.swap(state->idle);
    }
    for (auto &buffer : released) state->Recycle(std::move(buffer), false);
    state->changed.notify_all();
}
Buffer BufferPool::Acquire(size_t bytes, int device, std::shared_ptr<void> credit) {
    if (!bytes) return Buffer::Allocate(0, device);
    return std::move(AcquireBatch({{device, bytes}}, std::move(credit)).front());
}
std::vector<Buffer> BufferPool::AcquirePacked(const std::vector<std::pair<int, size_t>> &allocations,
                                             size_t alignment, std::shared_ptr<void> credit) {
    return AcquirePackedLayout(PackedLayout(allocations, alignment), std::move(credit));
}
BufferPool::PackedLayout::PackedLayout(const std::vector<std::pair<int, size_t>> &allocations, size_t alignment)
    : alignment(alignment), slices(allocations.size()) {
    Require(alignment && !(alignment & (alignment - 1)), "FastKV: invalid buffer alignment");
    std::map<int, size_t> devices;
    for (size_t i = 0; i < allocations.size(); ++i) {
        const auto &item = allocations[i];
        if (!item.second) continue;
        auto found = devices.emplace(item.first, arenas.size());
        if (found.second) arenas.emplace_back(item.first, 0);
        const size_t index = found.first->second;
        const size_t offset = CheckedAdd(arenas[index].second, alignment - 1) & ~(alignment - 1);
        arenas[index].second = CheckedAdd(offset, item.second);
        slices[i] = {index, offset, item.second};
    }
}
bool BufferPool::PackedLayout::PlaceOnPeers(const std::vector<std::pair<int, size_t>> &space,
                                          PackedLayout &out) const {
    out.arenas.clear();
    out.slices.assign(slices.size(), {});
    std::vector<size_t> indices(space.size(), SIZE_MAX);
    for (size_t i = 0; i < slices.size(); ++i) {
        const size_t bytes = slices[i].bytes;
        if (!bytes) continue;
        bool placed = false;
        for (size_t j = 0; j < space.size(); ++j) {
            const size_t used = indices[j] == SIZE_MAX ? 0 : out.arenas[indices[j]].second;
            const size_t offset = CheckedAdd(used, alignment - 1) & ~(alignment - 1);
            if (offset > space[j].second || bytes > space[j].second - offset) continue;
            if (indices[j] == SIZE_MAX) {
                indices[j] = out.arenas.size();
                out.arenas.emplace_back(space[j].first, 0);
            }
            out.slices[i] = {indices[j], offset, bytes};
            out.arenas[indices[j]].second = offset + bytes;
            placed = true;
            break;
        }
        if (!placed) return false;
    }
    return true;
}
std::vector<Buffer> BufferPool::AcquirePeerPackedLayout(const PackedLayout &layout) {
    if (!state->peerAccessible || layout.arenas.size() != 1) return AcquirePackedLayout(layout);
    const auto &request = layout.arenas.front();
    auto local = state->preferred.find(request.first);
    if (local == state->preferred.end() || request.second <= local->second) return AcquirePackedLayout(layout);

    CheckCapacity(layout.arenas);
    // Prefer the producer's own slab. Spill whole tensors onto warm peers;
    // packing kernels and RDMA both use the final physical address directly.
    std::vector<std::pair<int, size_t>> full{*local};
    for (const auto &entry : state->preferred) if (entry.first != request.first) full.push_back(entry);
    auto available = full;
    bool intact = true;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        Require(!state->closed, "FastKV: buffer pool stopped");
        for (auto &entry : available) {
            intact &= state->deviceAllocated.at(entry.first) == entry.second;
            size_t bytes = 0;
            for (const auto &b : state->idle)
                if (b.Device() == entry.first && b.Preallocated()) bytes = std::max(bytes, b.Size());
            // AcquireBatch rounds a lease to 4 KiB or 2 MiB. A leftover
            // range can be smaller than the rounded size of a large lease.
            if (bytes >= (1 << 20)) bytes = std::max(size_t((1 << 20) - 4096), bytes & ~size_t((2 << 20) - 1));
            entry.second = bytes;
        }
    }
    std::stable_sort(available.begin() + 1, available.end(),
        [](const auto &a, const auto &b) { return a.second > b.second; });
    PackedLayout selected({}, layout.alignment);
    if (layout.PlaceOnPeers(available, selected))
        return selected.SliceArenas(AcquireBatchImpl(selected.arenas, {}, true));
    if (intact && layout.PlaceOnPeers(full, selected))
        throw CapacityBusy("FastKV: waiting for peer export ranges");
    // Some layouts cannot fit the warm geometry (e.g. one indivisible
    // tensor exceeds every slab). Preserve the full on-demand budget.
    return AcquirePackedLayout(layout);
}
std::vector<Buffer> BufferPool::AcquirePackedLayout(const PackedLayout &layout, std::shared_ptr<void> credit) {
    return layout.SliceArenas(AcquireBatch(layout.arenas, std::move(credit)));
}
std::vector<Buffer> BufferPool::PackedLayout::SliceArenas(const std::vector<Buffer> &buffers) const {
    std::vector<Buffer> result(slices.size());
    for (size_t i = 0; i < slices.size(); ++i) {
        const auto &slice = slices[i];
        if (slice.bytes) result[i] = buffers[slice.arena].Slice(slice.offset, slice.bytes);
    }
    return result;
}
std::vector<Buffer> BufferPool::AcquireBatch(const std::vector<std::pair<int, size_t>> &allocations,
                                            std::shared_ptr<void> credit) {
    return AcquireBatchImpl(allocations, std::move(credit), false);
}
std::vector<Buffer> BufferPool::AcquireBatchImpl(const std::vector<std::pair<int, size_t>> &allocations,
                                                std::shared_ptr<void> credit, bool warmOnly) {
    CheckCapacity(allocations);
    std::vector<size_t> capacities;
    for (const auto &item : allocations) capacities.push_back(item.second ? Rounded(item.second, Limit(item.first)) : 0);
    auto physical = capacities;
    if (allocations.size() == 1 && capacities[0]) {
        const auto preferred = state->preferred.find(allocations[0].first);
        if (preferred != state->preferred.end()) physical[0] = std::max(capacities[0], preferred->second);
    }
    std::vector<Buffer> arenas(allocations.size());
    std::vector<Buffer> result(allocations.size());
    std::vector<bool> fresh(allocations.size());
    for (;;) {
        std::vector<Buffer> released;
        std::map<int, size_t> releasing;
        size_t releasedBytes = 0;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            Require(!state->closed, "FastKV: buffer pool stopped");
            released.reserve(state->idle.size());
            for (const auto &item : state->deviceAllocated) releasing.emplace(item.first, 0);
            std::vector<int> selected(allocations.size(), -1);
            std::vector<bool> used(state->idle.size(), false);
            // A checkout can return its tail and its lease independently.
            // Reserve their list slots before admission so coalescing and
            // returning a range cannot lose it under host memory pressure.
            if (!state->preferred.empty())
                state->idle.reserve(state->idle.size() + 2 * (state->checkedOut + allocations.size()));
            size_t needed = 0;
            std::map<int, size_t> deviceNeeded;
            for (size_t i = 0; i < allocations.size(); ++i) {
                if (!capacities[i]) continue;
                // Best fit preserves registered larger arenas as lengths vary.
                for (size_t j = 0; j < state->idle.size(); ++j) {
                    const auto &b = state->idle[j];
                    if (!used[j] && (!warmOnly || b.Preallocated()) &&
                        b.Device() == allocations[i].first && b.Size() >= capacities[i] &&
                        (selected[i] < 0 || b.Size() < state->idle[selected[i]].Size())) selected[i] = int(j);
                }
                if (warmOnly && selected[i] < 0) throw CapacityBusy("FastKV: peer export range in use");
                if (selected[i] >= 0) used[selected[i]] = true;
                else { needed = CheckedAdd(needed, physical[i]); deviceNeeded[allocations[i].first] += physical[i]; }
            }
            auto fits = [&] {
                if (needed > state->limit - (state->allocated - releasedBytes)) return false;
                for (const auto &item : deviceNeeded)
                    if (item.second > Limit(item.first) - (state->deviceAllocated[item.first] - releasing[item.first])) return false;
                return true;
            };
            if (!fits()) {
                if (!state->preferred.empty() && allocations.size() == 1) {
                    const int device = allocations[0].first;
                    size_t idleBytes = 0;
                    for (const auto &b : state->idle) if (b.Device() == device) {
                        idleBytes += b.Size();
                        // Another slice pins this allocation. Its contiguous
                        // capacity returns when those transfers complete.
                        if (!State::Whole(b) && b.AllocationSize() >= capacities[0])
                            throw CapacityBusy("FastKV: export arena in use");
                    }
                    const auto pref = state->preferred.find(device);
                    if (pref != state->preferred.end() && capacities[0] <= pref->second &&
                        state->deviceAllocated[device] > idleBytes && state->deviceAllocated[device] >= pref->second)
                        throw CapacityBusy("FastKV: export arena in use");
                }
                // Choose victims before committing evictions. A range only
                // releases physical budget when its entire allocation is idle.
                std::vector<size_t> victims;
                auto choose = [&](size_t j) {
                    const auto &b = state->idle[j];
                    if (!state->preferred.empty() && !State::Whole(b)) return;
                    victims.push_back(j);
                    releasedBytes += b.Size();
                    releasing[b.Device()] += b.Size();
                };
                for (size_t j = 0; j < state->idle.size() && !fits(); ++j)
                    if (!used[j]) choose(j);
                if (victims.empty()) {
                    // Oversized reuse may prevent an otherwise feasible batch.
                    for (size_t i = 0; i < selected.size(); ++i) if (selected[i] >= 0 &&
                            state->idle[selected[i]].Size() > capacities[i]) choose(selected[i]);
                }
                if (victims.empty() || (!state->preferred.empty() && allocations.size() == 1 && !fits()))
                    throw CapacityBusy("FastKV: buffer pool byte budget exhausted");
                for (size_t j : victims) released.push_back(std::move(state->idle[j]));
                state->idle.erase(std::remove_if(state->idle.begin(), state->idle.end(),
                    [](const Buffer &b) { return !b.Data(); }), state->idle.end());
            } else {
                for (size_t i = 0; i < allocations.size(); ++i) {
                    if (capacities[i]) ++state->checkedOut;
                    fresh[i] = capacities[i] && selected[i] < 0;
                    if (selected[i] >= 0) arenas[i] = std::move(state->idle[selected[i]]);
                }
                state->idle.erase(std::remove_if(state->idle.begin(), state->idle.end(),
                    [](const Buffer &b) { return !b.Data(); }), state->idle.end());
                state->allocated += needed;
                for (const auto &item : deviceNeeded) state->deviceAllocated[item.first] += item.second;
                break;
            }
        }
        released.clear();
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->allocated -= releasedBytes;
            for (const auto &item : releasing) state->deviceAllocated[item.first] -= item.second;
        }
        state->changed.notify_all();
    }
    auto rollback = [&] {
        for (size_t i = 0; i < arenas.size(); ++i) {
            if (arenas[i].Size()) state->Recycle(std::move(arenas[i]));
            else if (fresh[i]) {
                std::lock_guard<std::mutex> lock(state->mutex);
                --state->checkedOut;
                state->allocated -= physical[i];
                state->deviceAllocated[allocations[i].first] -= physical[i];
            }
        }
        state->changed.notify_all();
    };
    try {
        for (size_t i = 0; i < arenas.size(); ++i) if (fresh[i]) {
            auto buffer = state->allocator(physical[i], allocations[i].first);
            Require(buffer.Size() == physical[i] && buffer.Data() &&
                        buffer.Device() == allocations[i].first && !buffer.PoolBacked() &&
                        buffer.Kind() == (allocations[i].first < 0 ? MemoryKind::Host : MemoryKind::Cuda),
                    "FastKV: allocator returned an incompatible buffer");
            // Rebuilding a standard arena after an oversized export must
            // restore the reusable RDMA registration path as well.
            const auto preferred = state->preferred.find(buffer.Device());
            if (preferred != state->preferred.end() && physical[i] == preferred->second)
                buffer.MarkPreallocated();
            arenas[i] = std::move(buffer);
        }
    } catch (const CapacityBusy &) {
        rollback();
        // A budget is an upper bound, not a reservation in the embedding
        // process. Retire idle size classes on physical pressure too. The
        // scheduler/transport retries with cancellation and deadline checks;
        // never wait here while holding part of a multi-device message.
        state->TrimIdle();
        throw;
    } catch (...) {
        rollback();
        throw;
    }
    try {
        for (size_t i = 0; i < arenas.size(); ++i) {
            if (!capacities[i]) continue;
            if (!state->preferred.empty() && arenas[i].Size() > capacities[i]) {
                auto tail = arenas[i].Slice(capacities[i], arenas[i].Size() - capacities[i]);
                arenas[i].length = capacities[i];
                state->Recycle(std::move(tail), false);
            }
            result[i] = arenas[i].Slice(0, allocations[i].second);
            auto *address = arenas[i].MutableData();
            result[i].lease = std::shared_ptr<void>(address,
                [pool = state, arena = std::move(arenas[i]), credit](void *) mutable {
                    pool->Recycle(std::move(arena));
                });
        }
    } catch (...) {
        for (auto &arena : arenas) if (arena.Data()) state->Recycle(std::move(arena));
        throw;
    }
    return result;
}
} // namespace fastkv
