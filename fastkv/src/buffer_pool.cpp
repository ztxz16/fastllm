#include "internal.h"
#include <algorithm>

namespace fastkv {
struct BufferPool::State {
    std::mutex mutex;
    std::vector<Buffer> idle;
    std::condition_variable changed;
    size_t limit, allocated = 0;
    std::map<int, size_t> deviceLimits, deviceAllocated;
    bool closed = false;
    explicit State(size_t bytes) : limit(bytes) {}
    void Recycle(Buffer buffer) noexcept {
        const size_t bytes = buffer.Size();
        const int device = buffer.Device();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!closed) {
                try {
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
BufferPool::BufferPool(size_t limit, std::map<int, size_t> limits) : state(std::make_shared<State>(limit)) {
    state->deviceLimits = std::move(limits);
}
size_t BufferPool::Limit(int device) const {
    auto found = state->deviceLimits.find(device);
    return found == state->deviceLimits.end() ? state->limit : std::min(state->limit, found->second);
}
static size_t Rounded(size_t bytes, size_t limit) {
    const size_t alignment = bytes < (1 << 20) ? 4096 : (2 << 20);
    return std::min(CheckedAdd(bytes, alignment - 1) / alignment * alignment, limit);
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
    for (auto &buffer : released) state->Recycle(std::move(buffer));
    state->changed.notify_all();
}
Buffer BufferPool::Acquire(size_t bytes, int device, std::shared_ptr<void> credit) {
    if (!bytes) return Buffer::Allocate(0, device);
    return std::move(AcquireBatch({{device, bytes}}, std::move(credit)).front());
}
std::vector<Buffer> BufferPool::AcquirePacked(const std::vector<std::pair<int, size_t>> &allocations,
                                             size_t alignment, std::shared_ptr<void> credit) {
    Require(alignment && !(alignment & (alignment - 1)), "FastKV: invalid buffer alignment");
    std::vector<std::pair<int, size_t>> plan;
    std::map<int, size_t> devices;
    std::vector<std::pair<size_t, size_t>> slices(allocations.size());
    for (size_t i = 0; i < allocations.size(); ++i) {
        const auto &item = allocations[i];
        if (!item.second) continue;
        auto found = devices.emplace(item.first, plan.size());
        if (found.second) plan.emplace_back(item.first, 0);
        const size_t index = found.first->second;
        const size_t offset = CheckedAdd(plan[index].second, alignment - 1) & ~(alignment - 1);
        plan[index].second = CheckedAdd(offset, item.second);
        slices[i] = {index, offset};
    }
    // Reserve every device together: callers never hold partial message
    // allocations while waiting for another message's lease to retire.
    auto arenas = AcquireBatch(plan, std::move(credit));
    std::vector<Buffer> result(allocations.size());
    for (size_t i = 0; i < allocations.size(); ++i)
        if (allocations[i].second)
            result[i] = arenas[slices[i].first].Slice(slices[i].second, allocations[i].second);
    return result;
}
std::vector<Buffer> BufferPool::AcquireBatch(const std::vector<std::pair<int, size_t>> &allocations,
                                            std::shared_ptr<void> credit) {
    CheckCapacity(allocations);
    std::vector<size_t> capacities;
    for (const auto &item : allocations) capacities.push_back(item.second ? Rounded(item.second, Limit(item.first)) : 0);
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
            size_t needed = 0;
            std::map<int, size_t> deviceNeeded;
            for (size_t i = 0; i < allocations.size(); ++i) {
                if (!capacities[i]) continue;
                // Best fit preserves registered larger arenas as lengths vary.
                for (size_t j = 0; j < state->idle.size(); ++j) {
                    const auto &b = state->idle[j];
                    if (!used[j] && b.Device() == allocations[i].first && b.Size() >= capacities[i] &&
                        (selected[i] < 0 || b.Size() < state->idle[selected[i]].Size())) selected[i] = int(j);
                }
                if (selected[i] >= 0) used[selected[i]] = true;
                else { needed += capacities[i]; deviceNeeded[allocations[i].first] += capacities[i]; }
            }
            auto fits = [&] {
                if (needed > state->limit - (state->allocated - releasedBytes)) return false;
                for (const auto &item : deviceNeeded)
                    if (item.second > Limit(item.first) - (state->deviceAllocated[item.first] - releasing[item.first])) return false;
                return true;
            };
            if (!fits()) {
                // Retire unused arenas in release order. Their bytes remain
                // charged while MR destruction runs outside the mutex.
                for (size_t j = 0; j < state->idle.size() && !fits(); ++j) if (!used[j]) {
                    auto &b = state->idle[j];
                    releasedBytes += b.Size(); releasing[b.Device()] += b.Size();
                    released.push_back(std::move(b));
                }
                if (released.empty()) {
                    // Oversized reuse can prevent an otherwise feasible plan.
                    // Drop such idle arenas and retry with the minimal sizes.
                    for (size_t i = 0; i < selected.size(); ++i) if (selected[i] >= 0 &&
                            state->idle[selected[i]].Size() > capacities[i]) {
                        auto &b = state->idle[selected[i]];
                        releasedBytes += b.Size(); releasing[b.Device()] += b.Size();
                        released.push_back(std::move(b));
                    }
                }
                if (released.empty()) throw CapacityBusy("FastKV: buffer pool byte budget exhausted");
                state->idle.erase(std::remove_if(state->idle.begin(), state->idle.end(),
                    [](const Buffer &b) { return !b.Data(); }), state->idle.end());
            } else {
                for (size_t i = 0; i < allocations.size(); ++i) {
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
    try {
        for (size_t i = 0; i < arenas.size(); ++i)
            if (fresh[i]) arenas[i] = Buffer::Allocate(capacities[i], allocations[i].first);
    } catch (...) {
        for (size_t i = 0; i < arenas.size(); ++i) {
            if (arenas[i].Size()) state->Recycle(std::move(arenas[i]));
            else if (fresh[i]) {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->allocated -= capacities[i];
                state->deviceAllocated[allocations[i].first] -= capacities[i];
            }
        }
        state->changed.notify_all();
        throw;
    }
    try {
        for (size_t i = 0; i < arenas.size(); ++i) {
            if (!capacities[i]) continue;
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
