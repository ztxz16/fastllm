//
// CPU-only draft write fence / generation lease protocol.
//
// The DFlash draft cache publishes one immutable generation at a time. A prefix
// snapshot leases the current generation under the model mutex and reads it
// after the lock is released, so the writer must never mutate a generation that
// still has a second owner. This header contains the whole protocol: a metadata
// probe taken with the mutex held, the copy-on-write clone taken with no lock
// held, and the publish that must share one critical section with the write.
//
// Nothing here includes CUDA or model headers: the storage type is a template
// parameter, so the CPU regression exercises exactly the code the model runs
// (probe, clone, publish, observed mutex) with a CPU storage and no CUDA call.
//
// Observation is disabled by default: every guarded region pays one relaxed
// atomic load. The marks are emitted by the objects that really hold the mutex
// or really perform the device work, so a caller-reported "under_lock" flag can
// never fabricate a passing lock oracle.
#ifndef FASTLLM_P0_DRAFT_LEASE_H
#define FASTLLM_P0_DRAFT_LEASE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace p0lease {

enum class LockEvent : int {
    LockEnter = 0,
    LockExit = 1,
    DeviceWorkBegin = 2,
    DeviceWorkEnd = 3,
    Publish = 4,
    WriteBegin = 5,
    WriteEnd = 6,
    LeaseAcquire = 7,
    LeaseRelease = 8
};

struct LockMark {
    LockEvent event = LockEvent::LockEnter;
    // Owned string: a caller's stack buffer must not dangle before inspection.
    std::string tag;
    int depth = 0;
    uint64_t seq = 0;
};

class Observer {
public:
    static bool Enabled() noexcept {
        return EnabledRef().load(std::memory_order_acquire);
    }
    static void SetEnabled(bool enabled) noexcept {
        EnabledRef().store(enabled, std::memory_order_release);
    }
    // Lock depth is per thread: device work in thread A is not "under the lock"
    // merely because thread B happens to hold it.
    static int Depth() noexcept {
        return DepthRef();
    }
    // Records an event with the depth sampled at the call. Never throws: an
    // observation failure must not change model behaviour or leak the mutex.
    static void Note(LockEvent event, const char *tag) noexcept {
        try {
            if (!EnabledRef().load(std::memory_order_relaxed)) {
                return;
            }
            std::lock_guard<std::mutex> guard(MutexRef());
            LockMark mark;
            mark.event = event;
            mark.tag = tag ? tag : "";
            mark.depth = DepthRef();
            mark.seq = SeqRef().fetch_add(1, std::memory_order_relaxed) + 1;
            MarksRef().push_back(mark);
        } catch (...) {
        }
    }
    // Depth changes only here, so every observer (the model and the CPU
    // regression) reads the same counter that ObservedLock maintains.
    static void Enter(const char *tag) noexcept {
        ++DepthRef();
        Note(LockEvent::LockEnter, tag);
    }
    static void Leave(const char *tag) noexcept {
        Note(LockEvent::LockExit, tag);
        --DepthRef();
    }
    static std::vector<LockMark> Drain() noexcept {
        try {
            std::lock_guard<std::mutex> guard(MutexRef());
            std::vector<LockMark> out;
            out.swap(MarksRef());
            return out;
        } catch (...) {
            return std::vector<LockMark>();
        }
    }
    static void Reset() noexcept {
        try {
            std::lock_guard<std::mutex> guard(MutexRef());
            MarksRef().clear();
            SeqRef().store(0, std::memory_order_relaxed);
        } catch (...) {
        }
        DepthRef() = 0;
    }

private:
    static std::atomic<bool> &EnabledRef() {
        static std::atomic<bool> enabled(false);
        return enabled;
    }
    static int &DepthRef() {
        static thread_local int depth = 0;
        return depth;
    }
    static std::atomic<uint64_t> &SeqRef() {
        static std::atomic<uint64_t> seq(0);
        return seq;
    }
    static std::mutex &MutexRef() {
        static std::mutex mutex;
        return mutex;
    }
    static std::vector<LockMark> &MarksRef() {
        static std::vector<LockMark> marks;
        return marks;
    }
};

class ObservedLock {
public:
    ObservedLock(std::mutex &mutex, const char *tag) : mutex_(mutex), tag_(tag) {
        mutex_.lock();
        held_ = true;
        Observer::Enter(tag_);
    }
    ~ObservedLock() {
        Unlock();
    }
    ObservedLock(const ObservedLock &) = delete;
    ObservedLock &operator=(const ObservedLock &) = delete;
    void Unlock() {
        if (!held_) {
            return;
        }
        Observer::Leave(tag_);
        mutex_.unlock();
        held_ = false;
    }
    bool Held() const {
        return held_;
    }

private:
    std::mutex &mutex_;
    const char *tag_;
    bool held_ = false;
};

// Marks a region that performs device work on behalf of the draft cache: clone,
// allocation, D2D/D2H copy or a stream wait. The lock oracle requires depth 0
// for both marks, so an indirect CUDA call reached from Clone() cannot hide.
class DeviceWorkScope {
public:
    explicit DeviceWorkScope(const char *tag) : tag_(tag) {
        Observer::Note(LockEvent::DeviceWorkBegin, tag_);
    }
    ~DeviceWorkScope() {
        Observer::Note(LockEvent::DeviceWorkEnd, tag_);
    }
    DeviceWorkScope(const DeviceWorkScope &) = delete;
    DeviceWorkScope &operator=(const DeviceWorkScope &) = delete;

private:
    const char *tag_;
};

// One prepared write: source keeps the observed generation alive, prepared is
// the private clone created while no lock was held.
template <class Storage>
struct WriteTicket {
    std::shared_ptr<Storage> source;
    std::shared_ptr<Storage> prepared;
    bool copyNeeded = false;
};

// Call with the protocol mutex held. Metadata only: copies one shared_ptr and
// samples its use count. Never allocates device memory.
template <class Storage>
WriteTicket<Storage> BeginWrite(const std::shared_ptr<Storage> &owned) {
    WriteTicket<Storage> ticket;
    if (!owned) {
        return ticket;
    }
    // The count is the owning container plus any outstanding lease. Sample it
    // before this ticket takes its own reference.
    ticket.copyNeeded = owned.use_count() > 1;
    ticket.source = owned;
    Observer::Note(LockEvent::LeaseAcquire, "draft_begin");
    return ticket;
}

// Call with no lock held. The clone is the only place that performs device work
// for the fence.
template <class Storage>
void CloneForWrite(WriteTicket<Storage> &ticket) {
    if (!ticket.copyNeeded || !ticket.source) {
        return;
    }
    DeviceWorkScope scope("draft_clone");
    ticket.prepared = ticket.source->Clone();
}

// Call with the protocol mutex held, in the same critical section as the write.
// Returns false when the generation moved or the clone is unavailable; the
// caller must then drop the cache instead of writing a leased generation.
template <class Storage>
bool PublishWrite(std::shared_ptr<Storage> &draft, WriteTicket<Storage> &ticket) {
    if (!draft) {
        return false;
    }
    if (!ticket.source) {
        // The probe saw no entry: the container created this generation itself
        // and nothing can have leased it yet.
        return draft.use_count() == 1;
    }
    if (draft.get() != ticket.source.get()) {
        return false;
    }
    if (ticket.copyNeeded) {
        if (!ticket.prepared) {
            return false;
        }
        draft = std::move(ticket.prepared);
        Observer::Note(LockEvent::Publish, "draft_publish");
        return true;
    }
    // A lease may have appeared between the probe and this critical section. The
    // caller must re-probe (and clone) instead of writing a generation that is
    // already shared. The ticket holds one reference of its own, so only
    // owners beyond it count as leases.
    const long owners = draft.use_count() - (ticket.source.get() == draft.get() ? 1 : 0);
    return owners == 1;
}

}  // namespace p0lease

#endif  // FASTLLM_P0_DRAFT_LEASE_H
