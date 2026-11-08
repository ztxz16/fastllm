#ifndef FASTLLM_QWEN35_STATE_STAGING_H
#define FASTLLM_QWEN35_STATE_STAGING_H

#include "utils/qwen35_host_credit.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace fastllm { namespace qwen35_state_stage {

enum class Result { Granted, Retry, TooLarge, PoolFull, Closed, Failed, PermanentCapacity };
enum class State { Reserved, Enqueued, Ready, Materialized, Published, Cancelled, Failed };

struct Piece {
    int device = -1;
    size_t bytes = 0;
    // compactSource stages strided rows into one contiguous device buffer.
    size_t sourceStrideBytes = 0;
    size_t rowBytes = 0;
    int rows = 1;
    bool compactSource = false;
    const void *source = nullptr;
    void *destination = nullptr;
    void *validFlag = nullptr;
    // deviceBuffer is arena base + arenaOffset. The arena is preallocated by the
    // budget, so a capture never calls cudaMalloc on the compute path.
    void *deviceBuffer = nullptr;
    size_t arenaOffset = 0;
    size_t arenaReservedBytes = 0;
    int dataType = 0;
    std::vector<int> dims;
    std::vector<int> expansionDims;
    bool linear = false;
    bool transposed = false;
    long long cacheUid = 0;
    std::string name;
};

struct Identity {
    uint64_t requestId = 0;
    uint64_t requestGeneration = 0;
    uint64_t copyGeneration = 0;
    std::string cacheIdentity;
    bool Matches(uint64_t request, uint64_t requestGen, uint64_t copyGen,
                 const std::string &cache) const {
        return requestId == request && requestGeneration == requestGen &&
               copyGeneration == copyGen && cacheIdentity == cache;
    }
};

struct Ticket {
    Identity identity;
    std::vector<Piece> pieces;
    std::vector<std::shared_ptr<qwen35_host_credit::Credit> > credits;
    std::shared_ptr<void> owner;
    State state = State::Reserved;
    bool published = false;
    bool fallbackSync = false;
    std::map<int, size_t> stageBytes;
    // device -> (arena offset, reserved bytes). Returned to the arena by the
    // staging CUDA owner only after that device's copies have completed.
    std::map<int, std::pair<size_t, size_t> > arenaSpans;
    uint64_t d2dEnqueueNs = 0;
    uint64_t readyWaitNs = 0;
    uint64_t materializeNs = 0;
    uint64_t fallbackSyncCount = 0;
    std::string fallbackReason;
};

inline bool MarkEnqueued(const std::shared_ptr<Ticket> &ticket) {
    if (!ticket || ticket->state != State::Reserved) return false;
    ticket->state = State::Enqueued; return true;
}
inline bool MarkReady(const std::shared_ptr<Ticket> &ticket) {
    if (!ticket || ticket->state != State::Enqueued) return false;
    ticket->state = State::Ready; return true;
}
inline bool MarkMaterialized(const std::shared_ptr<Ticket> &ticket) {
    if (!ticket || ticket->state != State::Ready) return false;
    ticket->state = State::Materialized; return true;
}
inline bool CanPublish(const std::shared_ptr<Ticket> &ticket) {
    return ticket && ticket->state == State::Materialized && !ticket->published;
}
inline bool Publish(const std::shared_ptr<Ticket> &ticket) {
    if (!CanPublish(ticket)) return false;
    ticket->state = State::Published; ticket->published = true; return true;
}
inline void Fail(const std::shared_ptr<Ticket> &ticket) {
    if (ticket && ticket->state != State::Published) ticket->state = State::Failed;
}
inline void Cancel(const std::shared_ptr<Ticket> &ticket) {
    if (ticket && ticket->state != State::Published) ticket->state = State::Cancelled;
}

// Fixed, preallocated per-device arena for staged captures. The budget owns the
// device memory: a capture only sub-allocates it, so no cudaMalloc runs per
// piece and a full arena is a refusal (fall back to the sync path), never a
// dropped save.
class Arena {
    struct Device {
        size_t capacity = 0;
        void *base = nullptr;
        std::vector<std::pair<size_t, size_t> > used;   // sorted by offset
    };
    mutable std::mutex mutex;
    std::map<int, Device> devices;
    uint64_t refusals = 0;
public:
    // cudaMalloc memory is 256-byte aligned and only whole 256-byte units are
    // handed out, so every offset stays suitably aligned for any vector type.
    static size_t Align(size_t bytes) { return (bytes + 255u) & ~(size_t)255u; }
    void Adopt(int device, void *base, size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex);
        Device &d = devices[device];
        d.base = base; d.capacity = capacity; d.used.clear();
    }
    bool HasBase(int device) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = devices.find(device);
        return it != devices.end() && it->second.base != nullptr;
    }
    void *Base(int device) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = devices.find(device);
        return it == devices.end() ? nullptr : it->second.base;
    }
    size_t Capacity(int device) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = devices.find(device);
        return it == devices.end() ? 0 : it->second.capacity;
    }
    // bytes must already be Align()-ed by the caller.
    bool Acquire(int device, size_t bytes, size_t &offset) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = devices.find(device);
        if (it == devices.end() || it->second.base == nullptr || bytes == 0) return false;
        Device &d = it->second;
        size_t cursor = 0;
        for (size_t i = 0; i <= d.used.size(); ++i) {
            const size_t end = i < d.used.size() ? d.used[i].first : d.capacity;
            if (end >= cursor && end - cursor >= bytes) {
                d.used.insert(d.used.begin() + i, std::make_pair(cursor, bytes));
                offset = cursor;
                return true;
            }
            if (i < d.used.size())
                cursor = std::max(cursor, d.used[i].first + d.used[i].second);
        }
        ++refusals;
        return false;
    }
    void Release(int device, size_t offset, size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = devices.find(device);
        if (it == devices.end()) return;
        auto &used = it->second.used;
        for (size_t i = 0; i < used.size(); ++i) {
            if (used[i].first == offset && used[i].second == bytes) {
                used.erase(used.begin() + i);
                return;
            }
        }
    }
    size_t UsedBytes(int device) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = devices.find(device);
        if (it == devices.end()) return 0;
        size_t total = 0;
        for (const auto &span : it->second.used) total += span.second;
        return total;
    }
    uint64_t Refusals() const {
        std::lock_guard<std::mutex> lock(mutex); return refusals;
    }
};

// One event per device, recorded on that device's producer stream after its last
// capture copy. All copies for a device are FIFO on one stream, so a single
// event covers a whole capture: the export lane waits on it before the D2H, and
// the linear slot reattach path waits on it before it rewrites a slot.
class ReaderGate {
    mutable std::mutex mutex;
    std::map<int, void *> events;
public:
    void *Adopt(int device, void *candidate) {
        std::lock_guard<std::mutex> lock(mutex);
        void *&slot = events[device];
        if (slot == nullptr) slot = candidate;
        return slot;
    }
    void *For(int device) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = events.find(device);
        return it == events.end() ? nullptr : it->second;
    }
};

inline ReaderGate &GlobalReaderGate() { static ReaderGate gate; return gate; }

class Pool {
    mutable std::mutex mutex;
    std::map<int, std::shared_ptr<qwen35_host_credit::Ledger> > ledgers;
    std::map<int, size_t> limits;
    std::shared_ptr<Arena> arena = std::make_shared<Arena>();
    bool closed = false;
public:
    std::shared_ptr<Arena> ArenaRef() const { return arena; }
    void Configure(const std::map<int, size_t> &newLimits) {
        std::lock_guard<std::mutex> lock(mutex);
        // Merge, do not replace: a device that first owns staged state after
        // warmup (the standalone DFlash draft GPU) must gain a limit without
        // dropping the limits already granted to the TP forward devices.
        for (const auto &entry : newLimits) {
            if (!limits.count(entry.first)) limits[entry.first] = entry.second;
            if (!ledgers.count(entry.first))
                ledgers[entry.first] = std::make_shared<qwen35_host_credit::Ledger>();
        }
    }
    bool HasLimits() const {
        std::lock_guard<std::mutex> lock(mutex); return !limits.empty();
    }
    bool HasLimit(int device) const {
        std::lock_guard<std::mutex> lock(mutex); return limits.count(device) != 0;
    }
    // needs[device] is already Arena::Align()-ed. The arena hands out one span
    // per device so every piece of one capture is contiguous, which lets the
    // export lane D2H the whole device span in one go.
    Result TryReserve(const std::map<int, size_t> &needs, const Identity &identity,
                      std::shared_ptr<Ticket> &result) {
        result.reset();
        std::lock_guard<std::mutex> lock(mutex);
        if (closed) return Result::Closed;
        if (needs.empty()) return Result::Failed;
        std::vector<std::shared_ptr<qwen35_host_credit::Credit> > credits;
        std::map<int, std::pair<size_t, size_t> > spans;
        auto rollback = [&]() {
            for (const auto &span : spans)
                arena->Release(span.first, span.second.first, span.second.second);
        };
        for (const auto &entry : needs) {
            auto limit = limits.find(entry.first);
            auto ledger = ledgers.find(entry.first);
            if (limit == limits.end() || ledger == ledgers.end()) {
                rollback(); return Result::TooLarge;
            }
            std::shared_ptr<qwen35_host_credit::Credit> credit;
            const auto r = ledger->second->TryReserve(entry.second, limit->second, 0, credit);
            if (r != qwen35_host_credit::Result::Granted) {
                rollback();
                switch (r) {
                    case qwen35_host_credit::Result::Retry: return Result::Retry;
                    case qwen35_host_credit::Result::TooLarge: return Result::TooLarge;
                    case qwen35_host_credit::Result::Closed: return Result::Closed;
                    case qwen35_host_credit::Result::PermanentCapacity: return Result::PermanentCapacity;
                    default: return Result::Failed;
                }
            }
            credits.push_back(std::move(credit));
            size_t offset = 0;
            if (!arena->Acquire(entry.first, entry.second, offset)) {
                rollback(); return Result::PoolFull;
            }
            spans[entry.first] = std::make_pair(offset, entry.second);
        }
        auto ticket = std::make_shared<Ticket>();
        ticket->identity = identity; ticket->credits = std::move(credits);
        ticket->arenaSpans = std::move(spans);
        result = std::move(ticket); return Result::Granted;
    }
    void Close() {
        std::lock_guard<std::mutex> lock(mutex);
        closed = true; for (auto &entry : ledgers) entry.second->Close();
    }
};

} }
#endif
