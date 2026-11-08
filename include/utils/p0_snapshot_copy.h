#pragma once
#include "fastllm.h"
#include "utils/p0_snapshot_observation.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fastllm { namespace p0snapshot {
constexpr size_t kProcessBytes = size_t(128) << 20;
constexpr size_t kLeaseBytes = size_t(64) << 20;
constexpr size_t kLeaseCount = 2;
inline bool ParseEnabled(const char *s) {
    return s && (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0 ||
                 std::strcmp(s, "on") == 0);
}
inline bool Enabled() {
    static const bool enabled = ParseEnabled(std::getenv("FASTLLM_SNAPSHOT_COPY_BATCHED_ASYNC"));
    return enabled;
}
inline bool Mul(size_t a, size_t b, size_t &out) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return false;
    out = a * b; return true;
}
inline bool Add(size_t a, size_t b, size_t &out) {
    if (a > std::numeric_limits<size_t>::max() - b) return false;
    out = a + b; return true;
}
struct Source {
    const Data *data = nullptr;
    // Owns the real cache exclusion lock (production) or tensor (CPU regression).
    // It is never a no-op shared_ptr around a borrowed Data pointer.
    std::shared_ptr<const void> owner;
    int device = -1;
    std::shared_ptr<void> producer;
};
struct Tensor {
    DataType dtype;
    std::vector<int> dims;
    std::vector<uint64_t> strides;
    size_t offset = 0, bytes = 0, sourceExtent = 0, sourceAllocationBytes = 0;
    bool transposed = false;
    std::string name;
    long long cacheUid = 0;
};
struct Head {
    size_t tensor = 0, head = 0, sourceOffset = 0, strideBytes = 0;
    size_t bytes = 0, destinationOffset = 0;
    const uint8_t *source = nullptr;
    int device = -1;
};
struct Plan {
    std::vector<Source> sources;
    std::vector<Tensor> tensors;
    std::vector<Head> heads;
    size_t bytes = 0;
};
inline bool Describe(const std::vector<Source> &sources, Plan &out, std::string &reason) {
    Plan p;
    std::map<int, void *> producers;
    for (const auto &s : sources) {
        if (s.device>=0) {
            const auto found=producers.find(s.device);
            if (found!=producers.end() && found->second!=s.producer.get()) {
                reason="conflicting_device_producers"; return false;
            }
            producers[s.device]=s.producer.get();
        }
        if (!s.data || !s.owner) { reason = "missing_source_owner"; return false; }
        const Data &d = *s.data;
        if (d.multiDeviceData || d.isPagedKVCache || d.dataType != DataType::FLOAT16 ||
            d.unitSize != 2 || d.unitSizeDiv != 1 || d.dims.size() != 3 ||
            d.strides.size() != 3 || d.strides[2] != 1 || d.dims[0] <= 0 ||
            d.dims[1] <= 0 || d.dims[2] <= 0 || d.strides[1] != uint64_t(d.dims[2])) {
            reason = "unsupported_dtype_or_layout"; return false;
        }
        const uint8_t *base = d.dataDevice == DataDevice::CPU ? d.cpuData :
            (d.dataDevice == DataDevice::CUDA ? static_cast<const uint8_t *>(d.cudaData) : nullptr);
        if (!base || (d.dataDevice == DataDevice::CUDA && (s.device < 0 || !s.producer))) {
            reason = "missing_pointer_or_producer"; return false;
        }
        size_t elements, row, stride, bytes, last, extent, end;
        if (!Mul(size_t(d.dims[1]), size_t(d.dims[2]), elements) || !Mul(elements, 2, row) ||
            !Mul(size_t(d.strides[0]), 2, stride) || stride < row ||
            !Mul(size_t(d.dims[0]), row, bytes) || !Mul(size_t(d.dims[0]-1), stride, last) ||
            !Add(last, row, extent) || !Add(p.bytes, bytes, end)) {
            reason = "layout_overflow"; return false;
        }
        // Data::GetBytes is its strided logical extent; expansionBytes is the
        // real retained allocation when expanded. Never copy expansion padding.
        if (extent > (d.expansionBytes ? d.expansionBytes : d.GetBytes()) ||
            uintptr_t(base) > std::numeric_limits<uintptr_t>::max() - extent ||
            size_t(d.dims[0]) > 65536-p.heads.size()) {
            reason = "source_extent_out_of_bounds"; return false;
        }
        Tensor t; t.dtype=d.dataType; t.dims=d.dims; t.strides=d.strides;
        t.offset=p.bytes; t.bytes=bytes; t.sourceExtent=extent;
        t.sourceAllocationBytes=d.expansionBytes ? d.expansionBytes : d.GetBytes();
        t.transposed=d.isLinearAttentionTransposed;
        t.name=d.name; t.cacheUid=d.cacheUid;
        for (int h=0; h<d.dims[0]; ++h) p.heads.push_back({p.tensors.size(), size_t(h),
            size_t(h)*stride, stride, row, p.bytes+size_t(h)*row, base+size_t(h)*stride, s.device});
        p.tensors.push_back(std::move(t)); p.sources.push_back(s); p.bytes=end;
    }
    if (p.heads.empty()) { reason="empty_generation"; return false; }
    out=std::move(p); return true;
}
struct Budget {
    std::mutex mutex;
    size_t bytes=0, leases=0, peakBytes=0, peakLeases=0;
    std::function<void *(size_t)> allocate;
    std::function<void(void *)> free;
};
struct BudgetObservation { size_t bytes, leases, peakBytes, peakLeases; };
class Lease {
    friend class Arena;
    std::shared_ptr<Budget> state;
    Lease(std::shared_ptr<Budget> s, size_t n, uint64_t g):state(std::move(s)),bytes(n),generation(g) {}
public:
    void *base=nullptr;
    const size_t bytes;
    size_t stagingBytes=0;
    const uint64_t generation;
    Lease(const Lease &)=delete;
    Lease &operator=(const Lease &)=delete;
    ~Lease() {
        // Caller must hold the Lease through every terminal acknowledgement and
        // CPU pack. There is one matching allocator-specific deleter, no slice.
        if (base) state->free(base);
        std::lock_guard<std::mutex> lock(state->mutex);
        state->bytes-=bytes; --state->leases;
    }
};
class Arena {
    std::shared_ptr<Budget> state=std::make_shared<Budget>();
public:
    Arena(std::function<void *(size_t)> alloc, std::function<void(void *)> free) {
        state->allocate=std::move(alloc); state->free=std::move(free);
    }
    std::unique_ptr<Lease> Acquire(size_t stagingBytes, uint64_t generation, std::string &reason,
                                   size_t retainedBytes=0) {
        size_t bytes=0;
        if (!stagingBytes || !Add(stagingBytes,retainedBytes,bytes) || bytes>kLeaseBytes) {
            reason="lease_budget"; return {};
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->leases>=kLeaseCount || bytes>kProcessBytes-state->bytes) {
                reason="no_slot_or_process_budget"; return {};
            }
            state->bytes+=bytes; ++state->leases;
            state->peakBytes=std::max(state->peakBytes,state->bytes);
            state->peakLeases=std::max(state->peakLeases,state->leases);
        }
        // Allocate the owner before pinning so exceptions cannot leak a base.
        std::unique_ptr<Lease> lease;
        try { lease.reset(new Lease(state,bytes,generation)); }
        catch (...) {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->bytes-=bytes; --state->leases; throw;
        }
        lease->stagingBytes=stagingBytes;
        try { lease->base=state->allocate(stagingBytes); } catch (...) { reason="pin_allocation_exception"; return {}; }
        if (!lease->base) { reason="pin_allocation_failed"; return {}; }
        return lease;
    }
    BudgetObservation Observe() const {
        std::lock_guard<std::mutex> lock(state->mutex);
        return {state->bytes,state->leases,state->peakBytes,state->peakLeases};
    }
};
enum class Completion { Pending, Success, Failure };
struct Channel { virtual ~Channel() = default; };
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::unique_ptr<Channel> Prepare(int device, const std::shared_ptr<void> &producer)=0;
    virtual bool Enqueue(Channel &, void *dst, const void *src, size_t bytes) noexcept=0;
    virtual bool Seal(Channel &) noexcept=0;
    // Pending is never terminal. Wait must not throw or release its resources.
    // A fatal CUDA state without provable completion must fail-stop, not unwind.
    virtual Completion Wait(Channel &) noexcept=0;
};
struct Stats {
    uint64_t generation=0;
    size_t tensors=0, heads=0, bytes=0, accepted=0, joinedDevices=0, pendingObservations=0;
    size_t terminalCopies=0;
    Observer observer;
    bool contentDiagnostics=false;
    bool observationError=false;
    BudgetObservation lastBudget{0,0,0,0};
    const Arena *arena=nullptr;
    size_t observationSeq=0;
    bool ready=false;
    std::string reason;
    double enqueueMs=0, joinMs=0, cpuPackMs=0;
    void Emit(Observation o) noexcept {
        if (!observer) return;
        try {
            o.generation=generation; o.seq=++observationSeq;
            o.monoNs=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            o.accepted=accepted; o.terminal=terminalCopies;
            o.pending=accepted >= terminalCopies ? accepted-terminalCopies : 0;
            if (arena) lastBudget=arena->Observe();
            o.currentBytes=lastBudget.bytes; o.currentLeases=lastBudget.leases;
            o.peakBytes=lastBudget.peakBytes; o.peakLeases=lastBudget.peakLeases;
            observer(o);
        } catch (...) { observationError=true; }
    }
};
struct DeviceJob { int device; std::unique_ptr<Channel> channel; bool sealed=false, terminal=false; size_t accepted=0; };
class JoinGuard {
    std::vector<DeviceJob> &jobs; Backend &backend; Stats &stats;
public:
    bool success=true;
    JoinGuard(std::vector<DeviceJob> &j, Backend &b, Stats &s):jobs(j),backend(b),stats(s) {}
    void Join() noexcept {
        for (auto &job: jobs) {
            if (job.terminal) continue;
            if (!job.sealed) { job.sealed=true; if (!backend.Seal(*job.channel)) success=false; }
            Completion status;
            while ((status=backend.Wait(*job.channel))==Completion::Pending) {
                ++stats.pendingObservations; std::this_thread::yield();
            }
            job.terminal=true; ++stats.joinedDevices;
            stats.terminalCopies+=job.accepted;
            if (stats.observer) try { Observation o; o.event="copy_terminal"; o.device=job.device;
                o.deviceAccepted=job.accepted; o.deviceTerminal=job.accepted; o.devicePending=0;
                o.success=status==Completion::Success; stats.Emit(o); } catch (...) { stats.observationError=true; }
            if (status!=Completion::Success) success=false;
        }
    }
    ~JoinGuard() { Join(); }
};
enum class Result { Success, Fallback, Failed };
inline uint64_t PublishedCopyGeneration(Result result,const Stats &stats) noexcept {
    return result==Result::Success && stats.ready ? stats.generation : 0;
}

inline double Milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
inline Result Copy(Plan plan, const std::vector<Data *> &destinations,
                   Arena &arena, Backend &backend, uint64_t generation, Stats &stats,
                   const std::function<bool()> &cancelled={}, Lease *preparedLease=nullptr) {
    stats.arena=&arena;
    auto arenaReset = ScopeExit([&stats] { stats.arena=nullptr; });
    stats.generation=generation; stats.tensors=plan.tensors.size(); stats.heads=plan.heads.size(); stats.bytes=plan.bytes;
    if (!generation || destinations.size()!=plan.tensors.size()) { stats.reason="destination_count"; return Result::Failed; }
    for (Data *d:destinations) if (!d || d->cpuData || d->cudaData || d->isFake || d->multiDeviceData) {
        stats.reason="destination_not_empty_owned"; return Result::Failed;
    }
    // Destruction order: join -> channels -> staging -> source refs -> cleanup.
    // This measures helper-owned temporary references, not the context owner.
    ScopeExit cleanup([&] {
        plan.sources.clear();
        if (stats.observer) {
            try { Observation o; o.event="helper_cleanup"; o.sourceRefs=plan.sources.size();
                stats.Emit(o); } catch (...) { stats.observationError=true; }
        }
    });
    auto ownedLease=preparedLease ? std::unique_ptr<Lease>() : arena.Acquire(plan.bytes,generation,stats.reason);
    Lease *lease=preparedLease ? preparedLease : ownedLease.get();
    if (lease && (lease->generation!=generation || lease->stagingBytes<plan.bytes)) {
        stats.reason="prepared_lease_mismatch"; return Result::Failed;
    }
    if (!lease) return Result::Fallback; // no producer wait / enqueue has happened
    if (stats.observer) { try { Observation o; o.event="lease_acquired";
      o.bytes=lease->bytes; stats.Emit(o); } catch (...) { stats.observationError=true; } }
    std::vector<DeviceJob> jobs;
    JoinGuard join(jobs,backend,stats); // destroyed BEFORE lease and source owners
    auto start=std::chrono::steady_clock::now();
    try {
        std::map<int,size_t> byDevice;
        // Prepare every device before first copy, retaining each real producer.
        for (const auto &s:plan.sources) if (s.device>=0 && !byDevice.count(s.device)) {
            auto channel=backend.Prepare(s.device,s.producer);
            if (!channel) { stats.reason="prepare_failed"; return Result::Failed; }
            if (stats.observer) { try { Observation o; o.event="copy_wait_submitted"; o.device=s.device; o.producerId=reinterpret_cast<uintptr_t>(s.producer.get()); stats.Emit(o); } catch (...) { stats.observationError=true; } }
            byDevice[s.device]=jobs.size(); jobs.push_back({s.device,std::move(channel),false,false});
        }
        for (const auto &h:plan.heads) {
            if (cancelled && cancelled()) { stats.reason="abort"; join.success=false; break; }
            auto *dst=static_cast<uint8_t *>(lease->base)+h.destinationOffset;
            if (h.device<0) std::memcpy(dst,h.source,h.bytes);
            else if (!backend.Enqueue(*jobs[byDevice.at(h.device)].channel,dst,h.source,h.bytes)) {
                stats.reason="partial_enqueue_failure"; join.success=false; break;
            }
            ++stats.accepted;
            if (h.device>=0) ++jobs[byDevice.at(h.device)].accepted;
            else ++stats.terminalCopies;
            if (stats.observer) { try { Observation o; o.event="copy_accepted"; o.device=h.device;
                o.tensor=h.tensor; o.head=h.head; o.bytes=h.bytes;
                o.deviceAccepted=h.device>=0 ? jobs[byDevice.at(h.device)].accepted : stats.accepted;
                o.deviceTerminal=h.device<0 ? o.deviceAccepted : 0;
                o.devicePending=o.deviceAccepted-o.deviceTerminal; stats.Emit(o); } catch (...) { stats.observationError=true; } }
        }
        stats.enqueueMs=Milliseconds(start); start=std::chrono::steady_clock::now();
        join.Join(); stats.joinMs=Milliseconds(start);
        if (!join.success || stats.accepted!=plan.heads.size() || (cancelled && cancelled())) {
            if (stats.reason.empty()) stats.reason="copy_failure_or_abort_after_terminal";
            return Result::Failed;
        }
        // Content diagnostics are opt-in and run only after every real terminal.
        if (stats.observer && stats.contentDiagnostics) for (const auto &h:plan.heads) {
            const auto &t=plan.tensors[h.tensor]; try { Observation o;
            o.event="staging_head"; o.device=h.device; o.tensor=h.tensor; o.head=h.head;
            o.tokens=t.dims[1]; o.width=t.dims[2]; o.dtype=int(t.dtype);
            o.tensorHeads=t.dims[0]; o.sourceExtent=t.sourceExtent; o.sourceAllocationBytes=t.sourceAllocationBytes;
            o.strideBytes=h.strideBytes; o.sourceOffset=h.sourceOffset;
            o.destinationOffset=h.destinationOffset; o.bytes=h.bytes;
            o.digest=ContentDigest(static_cast<uint8_t *>(lease->base)+h.destinationOffset,h.bytes);
            stats.Emit(o); } catch (...) { stats.observationError=true; }
        }
        // All devices are terminal before ANY snapshot CPU Data allocation.
        start=std::chrono::steady_clock::now();
        for (size_t i=0;i<plan.tensors.size();++i) {
            const auto &t=plan.tensors[i]; Data &d=*destinations[i];
            d.dataDevice=DataDevice::CPU; d.dataDeviceIds.clear(); d.dataType=t.dtype;
            d.UpdateUnitSize(); d.Resize(t.dims); d.Allocate(false);
            std::memcpy(d.cpuData,static_cast<uint8_t *>(lease->base)+t.offset,t.bytes);
            d.name=t.name; d.cacheUid=t.cacheUid; d.isKVCache=true;
            d.isLinearAttention=false; d.isLinearAttentionTransposed=t.transposed;
            d.isPagedKVCache=false; d.pagedKVCacheData=nullptr; d.pageIndex.clear();
            d.lastPageLen=0; d.multiDeviceData=false; d.ClearTensorParallelLayout();
        }
        if (stats.observer && stats.contentDiagnostics) for (const auto &h:plan.heads) {
            const auto &t=plan.tensors[h.tensor]; const Data &d=*destinations[h.tensor];
            try { Observation o; o.event="cpu_head"; o.device=h.device; o.tensor=h.tensor; o.head=h.head;
            o.tokens=t.dims[1]; o.width=t.dims[2]; o.dtype=int(d.dataType);
            o.tensorHeads=t.dims[0]; o.sourceExtent=t.sourceExtent; o.sourceAllocationBytes=t.sourceAllocationBytes;
            o.strideBytes=h.strideBytes; o.sourceOffset=h.sourceOffset;
            o.destinationOffset=h.destinationOffset; o.bytes=h.bytes;
            o.digest=ContentDigest(d.cpuData+size_t(h.head)*d.strides[0]*d.unitSize,h.bytes);
            const auto staged=ContentDigest(static_cast<uint8_t *>(lease->base)+h.destinationOffset,h.bytes);
            o.success=o.digest==staged; stats.Emit(o);
            if (!o.success) { stats.reason="diagnostic_content_mismatch"; return Result::Failed; }
            } catch (...) { stats.observationError=true; }
        }
        stats.cpuPackMs=Milliseconds(start); stats.ready=true;
        if (stats.observer) { try { Observation o; o.event="ready"; stats.Emit(o); } catch (...) { stats.observationError=true; } }
        return Result::Success;
    } catch (...) {
        // Includes CPU allocation failure after terminal. No partially constructed
        // generation is publishable; caller's private snapshot owns cleanup.
        join.Join(); stats.reason="copy_or_cpu_pack_exception"; return Result::Failed;
    }
}
// Pinned staging only: existing 64 MiB lease / 128 MiB process / two slots.
// Full destination/shadow bytes are charged by the caller to existing
// HostLimit/ExportState/GPU snapshot reserve, before this helper is called.
// No raw ResponseContext may occur in plan/owners/observer captures.
struct CopyWave {
    Plan plan;
    std::vector<Data *> destinations;
};
inline bool MakeCopyWaves(const std::vector<Source> &sources,
        const std::vector<Data *> &destinations,std::vector<CopyWave> &out,
        size_t &maxWaveBytes,std::string &reason) {
    if (sources.size()!=destinations.size()) { reason="wave_destination_count"; return false; }
    out.clear(); CopyWave current; maxWaveBytes=0;
    for (size_t i=0;i<sources.size();++i) {
        Plan one; if (!Describe({sources[i]},one,reason)) return false;
        if (one.bytes>kLeaseBytes) { reason="tensor_exceeds_wave_budget"; return false; }
        bool conflictingProducer = false;
        for (const auto &source : current.plan.sources)
            if (source.device >= 0 && source.device == sources[i].device &&
                source.producer.get() != sources[i].producer.get()) conflictingProducer = true;
        if (current.plan.bytes && (one.bytes>kLeaseBytes-current.plan.bytes || conflictingProducer)) {
            maxWaveBytes=std::max(maxWaveBytes,current.plan.bytes);
            out.push_back(std::move(current)); current=CopyWave();
        }
        const size_t tensorIndex=current.plan.tensors.size();
        for (auto head:one.heads) {
            head.tensor=tensorIndex;
            head.destinationOffset += current.plan.bytes;
            current.plan.heads.push_back(head);
        }
        one.tensors.front().offset=current.plan.bytes;
        current.plan.bytes += one.bytes;
        current.plan.sources.push_back(std::move(one.sources.front()));
        current.plan.tensors.push_back(std::move(one.tensors.front()));
        current.destinations.push_back(destinations[i]);
    }
    if (current.plan.bytes) {
        maxWaveBytes=std::max(maxWaveBytes,current.plan.bytes);
        out.push_back(std::move(current));
    }
    if (out.empty()) { reason="empty_generation"; return false; }
    return true;
}
struct PendingIdentity {
    uint64_t request=0, nativeGeneration=0, copyGeneration=0;
    int cachedLen=0;
    // The legacy three counters above remain available to the CPU regression
    // fixture. Production asynchronous work may opt into the complete identity
    // tuple below; once any extended field is present, every field is mandatory
    // and terminal publication must compare the tuple as a whole.
    uint64_t requestGeneration=0, pageGeneration=0;
    std::string cacheIdentity;
    std::shared_ptr<const void> ownerLease;
    bool HasCompleteIdentity() const noexcept {
        return request != 0 && nativeGeneration != 0 && copyGeneration != 0 &&
               requestGeneration != 0 && pageGeneration != 0 &&
               !cacheIdentity.empty() && ownerLease != nullptr;
    }
    bool HasExtendedIdentity() const noexcept {
        return requestGeneration != 0 || pageGeneration != 0 ||
               !cacheIdentity.empty() || ownerLease != nullptr;
    }
    bool SameExtendedIdentity(const PendingIdentity &other) const noexcept {
        return request == other.request &&
               nativeGeneration == other.nativeGeneration &&
               copyGeneration == other.copyGeneration &&
               requestGeneration == other.requestGeneration &&
               pageGeneration == other.pageGeneration &&
               cacheIdentity == other.cacheIdentity &&
               ownerLease.get() == other.ownerLease.get();
    }
};

template<class Snapshot>
inline bool SamePrefixSnapshotIdentity(const Snapshot &a,const Snapshot &b) {
    if (a.cachedLen!=b.cachedLen || a.tokens!=b.tokens) return false;
    const auto &x=a.matchIdentity; const auto &y=b.matchIdentity;
    if (x.pages<=0 || x.pages!=y.pages || x.textPages!=y.textPages ||
        x.textPages<0 || x.textPages>x.pages ||
        x.mediaBaseSizes.size()!=size_t(x.pages-x.textPages) ||
        x.provenance!=y.provenance || x.mediaBaseSizes!=y.mediaBaseSizes) return false;
    // Text uses no media bytes. An empty raw-key vector and N empty text keys
    // retain the existing SnapshotPrefixIdentityMatches equivalence.
    for (int page=x.textPages;page<x.pages;++page) {
        const size_t index=size_t(page-x.textPages);
        if (index>=x.mediaBaseSizes.size() || size_t(page)>=a.mediaPageKeys.size() ||
            size_t(page)>=b.mediaPageKeys.size()) return false;
        const size_t bytes=x.mediaBaseSizes[index];
        if (a.mediaPageKeys[page].size()<bytes || b.mediaPageKeys[page].size()<bytes ||
            a.mediaPageKeys[page].compare(0,bytes,b.mediaPageKeys[page],0,bytes)!=0) return false;
    }
    return true;
}
template<class Snapshot> struct SnapshotPublication {
    std::shared_ptr<Snapshot> selected;
    std::vector<std::shared_ptr<Snapshot>> removed;
};
// Caller holds its queue mutex. The canonical charge is independent of whether
// a prepared payload was applied before cancellation/Stop/early worker failure.
inline bool RebindPreparedCharge(size_t &total,size_t &charged,size_t replacement,
                                 size_t limit,uint64_t storeInflight) noexcept {
    if (replacement>charged) {
        const size_t extra=replacement-charged;
        if (extra>limit || total>limit-extra || storeInflight>limit-extra-total) return false;
        total+=extra;
    } else {
        const size_t released=charged-replacement;
        if (released>total) return false;
        total-=released;
    }
    charged=replacement; return true;
}
inline bool ReleasePreparedCharge(size_t &total,size_t charged) noexcept {
    if (charged>total) return false;
    total-=charged; return true;
}
// Caller holds only its registry lock; returned retired owners are destroyed
// after that lock is released. This is publish-order recency, not read-touch LRU.
template<class Snapshot>
inline SnapshotPublication<Snapshot> PublishPrefixSnapshot(
        std::vector<std::shared_ptr<Snapshot>> &items,std::shared_ptr<Snapshot> snapshot,
        int perRequest,int global,long long &timestamp) {
    SnapshotPublication<Snapshot> result;
    const int request=snapshot->requestId;
    for (auto it=items.begin();it!=items.end();) {
        auto old=*it;
        if (old && SamePrefixSnapshotIdentity(*old,*snapshot)) {
            if ((old->mtpValid&&!snapshot->mtpValid) || (old->dflashValid&&!snapshot->dflashValid))
                snapshot=old;
            result.removed.push_back(std::move(*it)); it=items.erase(it);
        } else ++it;
    }
    snapshot->requestId=request; snapshot->timestamp=++timestamp;
    items.push_back(snapshot);
    size_t count=0;
    for (const auto &entry:items) if (entry && entry->requestId==request) ++count;
    while (count>size_t(std::max(1,perRequest))) {
        auto oldest=items.end();
        for (auto it=items.begin();it!=items.end();++it)
            if (*it && (*it)->requestId==request &&
                (oldest==items.end() || (*it)->timestamp<(*oldest)->timestamp)) oldest=it;
        if (oldest==items.end()) break;
        result.removed.push_back(std::move(*oldest)); items.erase(oldest); --count;
    }
    while (items.size()>size_t(std::max(1,global))) {
        result.removed.push_back(std::move(items.front())); items.erase(items.begin());
    }
    result.selected=std::move(snapshot);
    return result;
}
class PendingSnapshotHandle {
    // Destroyed last: charges only pinned staging until the scheduler has
    // published/discarded. Shadow/destination budgets belong to the caller.
    std::unique_ptr<Lease> reservation;
    std::shared_ptr<const void> destinationOwner;
    std::shared_ptr<const void> lifetimeOwner;
    std::vector<CopyWave> waves;
    std::unique_ptr<Backend> backend;
    Arena *arena;
    Stats stats;
    PendingIdentity identity;
    std::atomic<bool> cancelled{false}, done{false};
    std::function<void()> completed;
    Result result=Result::Failed;
    std::thread worker;
    PendingSnapshotHandle(std::unique_ptr<Lease> charge,
        std::shared_ptr<const void> dstOwner,std::shared_ptr<const void> keepAlive,
        std::vector<CopyWave> prepared,std::unique_ptr<Backend> copyBackend,
        Arena &pool,PendingIdentity key,Stats observations) :
        reservation(std::move(charge)),destinationOwner(std::move(dstOwner)),
        lifetimeOwner(std::move(keepAlive)),waves(std::move(prepared)),
        backend(std::move(copyBackend)),arena(&pool),
        stats(std::move(observations)),identity(key) {}
public:
    PendingSnapshotHandle(const PendingSnapshotHandle &)=delete;
    PendingSnapshotHandle &operator=(const PendingSnapshotHandle &)=delete;
    static std::unique_ptr<PendingSnapshotHandle> Submit(
        std::unique_ptr<Lease> reservation,std::shared_ptr<const void> destinationOwner,
        std::shared_ptr<const void> lifetimeOwner,std::vector<CopyWave> prepared,
        std::unique_ptr<Backend> backend,
        Arena &arena,PendingIdentity identity,Stats observations,
        std::function<void()> completed = {}) {
        if (!reservation || !destinationOwner || !lifetimeOwner || !backend ||
            !identity.nativeGeneration || !identity.copyGeneration ||
            (identity.HasExtendedIdentity() && !identity.HasCompleteIdentity()) ||
            reservation->generation!=identity.copyGeneration || prepared.empty())
            throw std::invalid_argument("invalid pending snapshot ownership");
        for (const auto &wave:prepared)
            if (wave.plan.bytes>reservation->stagingBytes ||
                wave.destinations.size()!=wave.plan.tensors.size())
                throw std::invalid_argument("invalid pending snapshot wave");
        auto job=std::unique_ptr<PendingSnapshotHandle>(new PendingSnapshotHandle(
            std::move(reservation),std::move(destinationOwner),std::move(lifetimeOwner),
            std::move(prepared),std::move(backend),arena,
            identity,std::move(observations)));
        job->completed = std::move(completed);
        // If thread construction throws, job destruction releases owners and
        // charge; no copy has been submitted and caller uses synchronous fallback.
        job->worker=std::thread([ptr=job.get()] {
            try {
                ptr->result=Result::Success;
                for (auto &wave:ptr->waves) {
                    Stats current;
                    current.contentDiagnostics=ptr->stats.contentDiagnostics;
                    if (ptr->stats.observer) current.observer=[ptr](const Observation &input) {
                        Observation output=input;
                        output.seq=++ptr->stats.observationSeq;
                        output.tensor+=ptr->stats.tensors;
                        output.accepted+=ptr->stats.accepted;
                        output.terminal+=ptr->stats.terminalCopies;
                        if (output.event=="ready") output.event="wave_ready";
                        if (output.event=="helper_cleanup") output.event="wave_cleanup";
                        ptr->stats.observer(output);
                    };
                    const auto waveResult=Copy(std::move(wave.plan),wave.destinations,
                        *ptr->arena,*ptr->backend,ptr->identity.copyGeneration,current,
                        [ptr] { return ptr->cancelled.load(std::memory_order_acquire); },
                        ptr->reservation.get());
                    ptr->stats.accepted += current.accepted;
                    ptr->stats.terminalCopies += current.terminalCopies;
                    ptr->stats.bytes += current.bytes;
                    ptr->stats.heads += current.heads;
                    ptr->stats.tensors += current.tensors;
                    ptr->stats.cpuPackMs += current.cpuPackMs;
                    ptr->stats.enqueueMs += current.enqueueMs;
                    ptr->stats.joinMs += current.joinMs;
                    ptr->stats.observationError=ptr->stats.observationError||current.observationError;
                    if (waveResult!=Result::Success) { ptr->result=waveResult; break; }
                }
                ptr->stats.generation=ptr->identity.copyGeneration;
                ptr->stats.ready=ptr->result==Result::Success &&
                    !ptr->cancelled.load(std::memory_order_acquire);
                ptr->stats.lastBudget=ptr->arena->Observe();
                if (ptr->stats.ready && ptr->stats.observer) {
                    Observation ready; ready.event="ready"; ptr->stats.Emit(ready);
                }
            } catch (...) { ptr->stats.reason="pending_worker_exception"; ptr->result=Result::Failed; }
            ptr->done.store(true,std::memory_order_release);
            if (ptr->completed) try { ptr->completed(); } catch (...) {}
        });
        return job;
    }
    void Cancel() noexcept { cancelled.store(true,std::memory_order_release); }
    bool Ready() const noexcept { return done.load(std::memory_order_acquire); }
    void Join() noexcept {
        if (worker.joinable()) {
            // Handles are scheduler/model-owned and never destroyed by their worker.
            if (worker.get_id()==std::this_thread::get_id()) std::terminate();
            worker.join();
        }
    }
    Result Finish() noexcept {
        Join();
        return cancelled.load(std::memory_order_acquire) ? Result::Failed : result;
    }
    const Stats &TerminalStats() const {
        if (!Ready()) throw std::logic_error("pending snapshot has no terminal stats");
        return stats;
    }
    const PendingIdentity &Identity() const noexcept { return identity; }
    ~PendingSnapshotHandle() { Cancel(); Join(); }
};
}} // namespace fastllm::p0snapshot
