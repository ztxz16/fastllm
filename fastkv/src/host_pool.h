#pragma once
#include "internal.h"
#include <iterator>
#include <set>

namespace fastkv {
// Private byte allocator for L2 storage. A slab, including its holes,
// owns one budget credit and one NUMA allocation. Leases may outlive the cache.
class HostPool {
    struct Segment { size_t bytes; bool used = false; };
    struct Slab {
        std::shared_ptr<void> credit;
        Buffer memory;
        int node;
        size_t live = 0;
        std::map<size_t, Segment> segments;
        // Used segments keep a zero-size index node for allocation-free release.
        std::set<std::pair<size_t, size_t>> sizes;
        Slab(Buffer memory, std::shared_ptr<void> credit, int node)
            : credit(std::move(credit)), memory(std::move(memory)), node(node) {
            segments.emplace(0, Segment{this->memory.Size()});
            sizes.emplace(this->memory.Size(), size_t(0));
        }
        size_t Largest() const {
            return sizes.rbegin()->first;
        }
    };
    using Key = std::pair<size_t, uint64_t>; // largest hole, stable slab identity
    using Slabs = std::map<Key, std::shared_ptr<Slab>>;
    struct State {
        std::mutex mutex;
        std::map<int, Slabs> nodes;
        bool retainPinned = true;
        size_t idlePinned = 0;
        void Release(const std::shared_ptr<Slab> &slab, size_t offset) {
            // Reuse existing map nodes when freeing/coalescing. Destruction of
            // a lease must not allocate, including under host memory pressure.
            Slabs::node_type removed;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto &slabs = nodes.at(slab->node);
                removed = slabs.extract({slab->Largest(), slab->memory.Id()});
                auto part = slab->segments.find(offset);
                auto range = slab->sizes.extract({0, offset});
                part->second.used = false;
                auto next = std::next(part);
                if (next != slab->segments.end() && !next->second.used) {
                    slab->sizes.erase({next->second.bytes, next->first});
                    part->second.bytes += next->second.bytes;
                    slab->segments.erase(next);
                }
                if (part != slab->segments.begin()) {
                    auto previous = std::prev(part);
                    if (!previous->second.used) {
                        slab->sizes.erase({previous->second.bytes, previous->first});
                        previous->second.bytes += part->second.bytes;
                        slab->segments.erase(part);
                        part = previous;
                    }
                }
                range.value() = {part->second.bytes, part->first};
                slab->sizes.insert(std::move(range));
                --slab->live;
                if (slab->live || (retainPinned && slab->memory.HostPinned())) {
                    removed.key().first = slab->Largest();
                    slabs.insert(std::move(removed));
                    if (!slab->live) ++idlePinned;
                }
            }
        }
    };
    struct Lease {
        std::shared_ptr<State> state;
        std::shared_ptr<Slab> slab;
        size_t offset = 0;
        bool active = false;
        ~Lease() { if (active) state->Release(slab, offset); }
    };
    std::shared_ptr<State> state = std::make_shared<State>();
    // Caller holds State::mutex. Prepare throwing allocations before marking
    // the range used; a failed allocation leaves the existing slab unchanged.
    Buffer Take(Slabs &slabs, Slabs::iterator found, size_t bytes) {
        auto slab = found->second;
        auto range = slab->sizes.lower_bound({bytes, 0});
        auto part = slab->segments.find(range->second);
        auto lease = std::make_shared<Lease>();
        lease->state = state; lease->slab = slab; lease->offset = part->first;
        auto result = slab->memory.Slice(part->first, bytes).WithLease(lease);
        if (part->second.bytes > bytes) {
            const size_t offset = part->first + bytes, remaining = part->second.bytes - bytes;
            auto tail = slab->segments.emplace(offset, Segment{remaining}).first;
            try { slab->sizes.emplace(remaining, offset); }
            catch (...) { slab->segments.erase(tail); throw; }
        }
        auto sizeEntry = slab->sizes.extract(range);
        sizeEntry.value().first = 0;
        slab->sizes.insert(std::move(sizeEntry));
        auto entry = slabs.extract(found);
        part->second = {bytes, true}; ++slab->live;
        entry.key().first = slab->Largest();
        slabs.insert(std::move(entry));
        lease->active = true;
        return result;
    }
  public:
    HostPool() = default;
    HostPool(const HostPool &) = delete;
    HostPool &operator=(const HostPool &) = delete;
    ~HostPool() {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->retainPinned = false;
        for (auto &node : state->nodes) {
            for (auto it = node.second.begin(); it != node.second.end();) {
                if (!it->second->live) it = node.second.erase(it);
                else ++it;
            }
        }
    }
    // An empty registered slab can be reused immediately after eviction.
    // Surrender its physical credit when another allocation needs the space.
    bool ReleaseIdle() {
        Slabs::node_type released;
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->idlePinned) return false;
        for (auto &node : state->nodes) {
            auto &slabs = node.second;
            for (auto it = slabs.rbegin(); it != slabs.rend(); ++it) {
                if (!it->second->live) {
                    released = slabs.extract(std::prev(it.base()));
                    --state->idlePinned;
                    return true;
                }
            }
        }
        return false;
    }
    Buffer TryAllocate(size_t bytes, int node) {
        Require(bytes > 0, "FastKV: empty host allocation");
        std::lock_guard<std::mutex> lock(state->mutex);
        auto group = state->nodes.find(node);
        if (group == state->nodes.end()) return {};
        auto found = group->second.lower_bound({bytes, 0});
        if (found == group->second.end()) return {};
        const bool idle = !found->second->live;
        auto result = Take(group->second, found, bytes);
        if (idle) --state->idlePinned;
        return result;
    }
    Buffer Add(Buffer memory, std::shared_ptr<void> credit, int node, size_t bytes) {
        Require(bytes && bytes <= memory.Size() && memory.Kind() == MemoryKind::Host,
                "FastKV: invalid host slab");
        auto slab = std::make_shared<Slab>(std::move(memory), std::move(credit), node);
        std::lock_guard<std::mutex> lock(state->mutex);
        auto &slabs = state->nodes[node];
        auto found = slabs.emplace(Key{slab->memory.Size(), slab->memory.Id()}, slab).first;
        try { return Take(slabs, found, bytes); }
        catch (...) { slabs.erase(found); throw; }
    }
};
} // namespace fastkv
