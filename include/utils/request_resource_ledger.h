#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <map>
#include <stdexcept>
#include <utility>

namespace fastllm {

// Request-scoped ownership accounting. Resident model/cache owners are kept
// outside this object; the terminal event reports them separately.
struct RequestResourceSnapshot {
    size_t transientOwners = 0;
    size_t temporaryLeases = 0;
    size_t pendingOperations = 0;
    size_t temporaryReaders = 0;
    size_t residentOwners = 0;
    size_t residentGenerations = 0;
    bool residentObserved = false;
    bool consistencyError = false;
    std::set<uint64_t> liveGenerations;
};

struct RequestResourceTransition {
    uint64_t nativeHandle, nativeGeneration, holderId, heldGeneration;
    const char *action, *role;
    size_t owners, leases, pending, readers;
};

class RequestResourceLedger {
    mutable std::mutex mutex;
    size_t transientOwners = 1; // ResponseContext itself; released after member destruction.
    size_t temporaryLeases = 0;
    size_t pendingOperations = 0;
    size_t temporaryReaders = 0;
    size_t residentOwners = 0;
    size_t residentGenerations = 0;
    bool residentObserved = false;
    std::atomic<bool> consistencyError{false};
    bool contextOwnerHeld = true;
    struct Hold { uint64_t generation; bool reader; };
    std::map<uint64_t, Hold> holders;
    uint64_t nextToken = 1; // Zero is the inert token and is never issued.
    using Trace = void (*)(const RequestResourceTransition &) noexcept;
    Trace trace = nullptr;
    uint64_t nativeHandle = 0, nativeGeneration = 0;
    void Notify(const char *action, const char *role, uint64_t token,
                uint64_t heldGeneration) const noexcept {
        if (trace) trace({nativeHandle, nativeGeneration, token, heldGeneration,
            action, role, transientOwners, temporaryLeases, pendingOperations,
            temporaryReaders});
    }

    uint64_t AcquireToken(uint64_t generation, bool reader) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!nextToken || nextToken == std::numeric_limits<uint64_t>::max() ||
            transientOwners == std::numeric_limits<size_t>::max() ||
            (reader && (temporaryReaders == std::numeric_limits<size_t>::max() ||
                        temporaryLeases == std::numeric_limits<size_t>::max())) ||
            (!reader && pendingOperations == std::numeric_limits<size_t>::max())) {
            consistencyError = true;
            throw std::overflow_error("Request resource ledger token/count overflow");
        }
        // Allocate the holder entry before changing any balance or token cursor.
        try {
            holders.emplace(nextToken, Hold{generation, reader});
        } catch (...) {
            consistencyError = true;
            throw;
        }
        const uint64_t token = nextToken++;
        ++transientOwners;
        if (reader) { ++temporaryReaders; ++temporaryLeases; }
        else { ++pendingOperations; }
        Notify("acquire", reader ? "reader" : "pending", token, generation);
        return token;
    }

public:
    void SetTrace(uint64_t handle, uint64_t generation, Trace observer) {
        std::lock_guard<std::mutex> lock(mutex);
        nativeHandle = handle; nativeGeneration = generation; trace = observer;
        Notify("acquire", "context", 0, generation);
    }
    void RecordConsistencyError() noexcept {
        consistencyError.store(true, std::memory_order_relaxed);
    }
    void ReleaseContextOwner() noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (!contextOwnerHeld || transientOwners == 0) { consistencyError = true; return; }
            contextOwnerHeld = false; --transientOwners;
            Notify("release", "context", 0, nativeGeneration);
        } catch (...) { consistencyError.store(true, std::memory_order_relaxed); }
    }
    uint64_t AcquireReaderToken(uint64_t generation) { return AcquireToken(generation, true); }
    uint64_t BeginPendingToken(uint64_t generation) { return AcquireToken(generation, false); }
    void ReleaseToken(uint64_t token) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = holders.find(token);
            if (it == holders.end() || !transientOwners ||
                (it->second.reader && (!temporaryReaders || !temporaryLeases)) ||
                (!it->second.reader && !pendingOperations)) {
                consistencyError = true;
                return;
            }
            const Hold held = it->second;
            if (it->second.reader) { --temporaryReaders; --temporaryLeases; }
            else { --pendingOperations; }
            --transientOwners;
            holders.erase(it);
            Notify("release", held.reader ? "reader" : "pending", token, held.generation);
        } catch (...) { consistencyError.store(true, std::memory_order_relaxed); }
    }
    // Call only with an actual model/cache sample; an absent sample stays unknown.
    void ObserveResident(size_t owners, size_t generations, bool observed = true) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            residentOwners = owners; residentGenerations = generations; residentObserved = observed;
        } catch (...) { consistencyError.store(true, std::memory_order_relaxed); }
    }
    RequestResourceSnapshot Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        RequestResourceSnapshot result;
        result.transientOwners = transientOwners;
        result.temporaryLeases = temporaryLeases;
        result.pendingOperations = pendingOperations;
        result.temporaryReaders = temporaryReaders;
        result.residentOwners = residentOwners;
        result.residentGenerations = residentGenerations;
        result.residentObserved = residentObserved;
        result.consistencyError = consistencyError.load(std::memory_order_relaxed);
        for (const auto &entry : holders) result.liveGenerations.insert(entry.second.generation);
        return result;
    }
};

// One token per real holder. Null ledger is the default-off fast path and does
// not allocate, acquire a lock, or retain a request ledger.
class RequestResourceHold {
    std::shared_ptr<RequestResourceLedger> ledger;
    uint64_t token = 0;

    RequestResourceHold(std::shared_ptr<RequestResourceLedger> owner,
                        uint64_t generation, bool reader) : ledger(std::move(owner)) {
        if (ledger) token = reader ? ledger->AcquireReaderToken(generation) :
                                    ledger->BeginPendingToken(generation);
    }

public:
    RequestResourceHold() noexcept = default;
    RequestResourceHold(const RequestResourceHold &) = delete;
    RequestResourceHold &operator=(const RequestResourceHold &) = delete;
    RequestResourceHold(RequestResourceHold &&other) noexcept :
        ledger(std::move(other.ledger)), token(std::exchange(other.token, 0)) {}
    RequestResourceHold &operator=(RequestResourceHold &&other) noexcept {
        if (this != &other) {
            Reset();
            ledger = std::move(other.ledger);
            token = std::exchange(other.token, 0);
        }
        return *this;
    }
    ~RequestResourceHold() { Reset(); }
    static RequestResourceHold Reader(std::shared_ptr<RequestResourceLedger> owner,
                                      uint64_t generation) {
        return RequestResourceHold(std::move(owner), generation, true);
    }
    static RequestResourceHold Pending(std::shared_ptr<RequestResourceLedger> owner,
                                       uint64_t generation) {
        return RequestResourceHold(std::move(owner), generation, false);
    }
    void Reset() noexcept {
        auto owner = std::move(ledger);
        const uint64_t held = std::exchange(token, 0);
        if (owner && held) owner->ReleaseToken(held);
    }
    explicit operator bool() const noexcept { return token != 0; }
    uint64_t TokenId() const noexcept { return token; } // Unique within this ledger.
};

} // namespace fastllm
