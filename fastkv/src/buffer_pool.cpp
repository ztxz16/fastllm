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
        for (const auto &buffer : state->idle) {
            state->allocated -= buffer.Size();
            state->deviceAllocated[buffer.Device()] -= buffer.Size();
        }
        released.swap(state->idle);
    }
    state->changed.notify_all();
}
Buffer BufferPool::Acquire(size_t bytes, int device, std::shared_ptr<void> credit) {
    if (!bytes) return Buffer::Allocate(0, device);
    // Bounded classes allow neighbouring token lengths to reuse registrations.
    const size_t deviceLimit = Limit(device);
    const size_t capacity = Rounded(bytes, deviceLimit);
    Require(capacity >= bytes, "FastKV: arena exceeds byte budget");
    Buffer arena;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        Require(!state->closed, "FastKV: buffer pool stopped");
        auto found = std::find_if(state->idle.begin(), state->idle.end(), [&](const Buffer &b) {
            return b.Size() == capacity && b.Device() == device;
        });
        if (found != state->idle.end()) {
            arena = std::move(*found);
            state->idle.erase(found);
        } else {
            // Evict only unleased arenas. MR destruction precedes the next
            // allocation, keeping resident pool memory within the byte limit.
            auto fits = [&] { return capacity <= state->limit - state->allocated &&
                                     capacity <= deviceLimit - state->deviceAllocated[device]; };
            while (!fits() && !state->idle.empty()) {
                state->allocated -= state->idle.back().Size();
                state->deviceAllocated[state->idle.back().Device()] -= state->idle.back().Size();
                state->idle.pop_back();
            }
            if (!fits()) throw CapacityBusy("FastKV: buffer pool byte budget exhausted");
            state->allocated += capacity;
            state->deviceAllocated[device] += capacity;
        }
    }
    try {
        if (!arena.Size()) arena = Buffer::Allocate(capacity, device);
    } catch (...) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->allocated -= capacity;
        state->deviceAllocated[device] -= capacity;
        state->changed.notify_all();
        throw;
    }
    auto result = arena.Slice(0, bytes);
    // The returned Buffer shares Storage (and MRs), but has a per-use lease.
    // The idle pool owns only the raw Buffer, so there is no ownership cycle.
    auto *address = arena.MutableData();
    result.lease = std::shared_ptr<void>(address,
        [pool = state, arena = std::move(arena), credit = std::move(credit)](void *) mutable {
            std::lock_guard<std::mutex> lock(pool->mutex);
            if (pool->closed) {
                pool->allocated -= arena.Size();
                pool->deviceAllocated[arena.Device()] -= arena.Size();
            }
            else {
                try { pool->idle.push_back(std::move(arena)); }
                catch (...) {
                    pool->allocated -= arena.Size();
                    pool->deviceAllocated[arena.Device()] -= arena.Size();
                }
            }
            pool->changed.notify_all();
        });
    return result;
}
} // namespace fastkv
