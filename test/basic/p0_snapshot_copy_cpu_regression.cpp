#include "utils/p0_draft_lease.h"
#include "utils/p0_snapshot_copy.h"
#include "utils/request_resource_ledger.h"
#include "utils/snapshot_prefix_identity.h"
#include <cstdio>
#include <map>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>

using namespace fastllm;
namespace S = fastllm::p0snapshot;
extern "C" unsigned long long P0SnapshotCudaRejectedCalls();
static void Need(bool ok,const char *message) { if (!ok) throw std::runtime_error(message); }
struct HostPins {
    size_t allocated=0,freed=0;
    bool fail=false;
    std::set<void *> live;
    S::Arena arena;
    HostPins():arena([this](size_t n)->void * {
        if (fail) return nullptr;
        void *p=std::malloc(n); if (p) { ++allocated; live.insert(p); } return p;
    },[this](void *p) { Need(live.erase(p)==1,"wrong staging deleter/base or double free"); ++freed; std::free(p); }) {}
    void Zero() {
        auto b=arena.Observe(); Need(b.bytes==0 && b.leases==0,"lease not released");
        Need(live.empty() && allocated==freed,"pinned owner not freed exactly once");
        Need(b.peakBytes<=S::kProcessBytes && b.peakLeases<=S::kLeaseCount,"budget exceeded");
    }
};
struct FakeChannel: S::Channel {
    int device; int polls=0; bool sealed=false;
    struct Op { void *dst; const void *src; size_t bytes; };
    std::vector<Op> ops;
    explicit FakeChannel(int d):device(d) {}
};
struct FakeBackend: S::Backend {
    HostPins &pins;
    int failAfter=-1,prepareFail=-1;
    bool terminalFailure=false,sealFailure=false,mutateStaged=false,invariants=true;
    size_t enqueues=0,prepares=0,waits=0;
    int pendingPolls=0;
    std::vector<int> terminalOrder;
    std::vector<Data *> outputs;
    std::vector<std::weak_ptr<const void>> sourceOwners;
    explicit FakeBackend(HostPins &p):pins(p) {}
    std::unique_ptr<S::Channel> Prepare(int device,const std::shared_ptr<void> &producer) override {
        ++prepares; if (device==prepareFail || !producer) return {};
        return std::unique_ptr<S::Channel>(new FakeChannel(device));
    }
    bool Enqueue(S::Channel &base,void *dst,const void *src,size_t n) noexcept override {
        if (failAfter>=0 && int(enqueues)==failAfter) return false;
        try { static_cast<FakeChannel &>(base).ops.push_back({dst,src,n}); ++enqueues;
            std::cout<<"event=copy_accepted copy_id="<<enqueues<<" device="<<static_cast<FakeChannel &>(base).device<<" bytes="<<n<<"\n"; return true; }
        catch (...) { return false; }
    }
    bool Seal(S::Channel &base) noexcept override {
        static_cast<FakeChannel &>(base).sealed=true; return !sealFailure;
    }
    S::Completion Wait(S::Channel &base) noexcept override {
        auto &ch=static_cast<FakeChannel &>(base); ++waits;
        auto b=pins.arena.Observe();
        invariants=invariants && ch.sealed && b.bytes>0 && b.leases==1;
        for (auto &w:sourceOwners) invariants=invariants && !w.expired();
        for (auto *d:outputs) invariants=invariants && d->cpuData==nullptr;
        if (ch.polls++<pendingPolls) {
            std::cout<<"event=wait_pending device="<<ch.device<<" pinned="<<b.bytes<<" leases="<<b.leases<<" source_retained="<<invariants<<"\n";
            return S::Completion::Pending;
        }
        for (auto &op:ch.ops) { std::memcpy(op.dst,op.src,op.bytes); if (mutateStaged && op.bytes) *static_cast<unsigned char *>(op.dst) ^= 1; }
        try { terminalOrder.push_back(ch.device); } catch (...) { std::terminate(); }
        std::cout<<"event=copy_terminal device="<<ch.device<<" success="<<(!terminalFailure)<<" pinned="<<b.bytes<<"\n";
        return terminalFailure?S::Completion::Failure:S::Completion::Success;
    }
};
// The lock oracle is the production RAII wrapper itself: p0lease::ObservedLock
// takes the real mutex of the guarded region and marks enter/exit from the
// object that holds it, so neither the test nor the model can fabricate a pass
// by calling an oracle entry point manually. The backend samples the depth that
// wrapper maintains at every terminal boundary.
struct ObservedBackend: FakeBackend {
    int maxDepth=0; size_t samples=0;
    explicit ObservedBackend(HostPins &p):FakeBackend(p) {}
    void Sample() noexcept { ++samples; const int d=p0lease::Observer::Depth(); if (d>maxDepth) maxDepth=d; }
    std::unique_ptr<S::Channel> Prepare(int device,const std::shared_ptr<void> &producer) override {
        Sample(); return FakeBackend::Prepare(device,producer);
    }
    bool Enqueue(S::Channel &c,void *d,const void *s,size_t n) noexcept override {
        Sample(); return FakeBackend::Enqueue(c,d,s,n);
    }
    bool Seal(S::Channel &c) noexcept override { Sample(); return FakeBackend::Seal(c); }
    S::Completion Wait(S::Channel &c) noexcept override { Sample(); return FakeBackend::Wait(c); }
};

static std::shared_ptr<Data> Tensor(int heads,int tokens,int width,int capacity,int seed=1) {
    auto d=std::make_shared<Data>(DataType::FLOAT16,std::vector<int>{heads,tokens,width});
    d->Allocate();
    if (capacity>tokens) d->Expansion({heads,capacity,width});
    std::memset(d->cpuData,0xEE,d->expansionBytes?d->expansionBytes:d->GetBytes());
    for (int h=0;h<heads;++h) for (int t=0;t<tokens;++t) for (int w=0;w<width;++w) {
        const size_t at=(size_t(h)*d->strides[0]+size_t(t)*d->strides[1]+w)*2;
        const uint16_t v=uint16_t(seed+h*173+t*13+w);
        std::memcpy(d->cpuData+at,&v,2);
    }
    return d;
}
struct Fixture {
    S::Plan plan;
    std::vector<std::unique_ptr<Data>> ownedOutputs;
    std::vector<Data *> outputs;
    std::vector<std::vector<uint8_t>> expected;
    std::vector<std::weak_ptr<const void>> weakOwners;
};
static Fixture Build(size_t tensors=10,int heads=8,int tokens=17,int capacity=32,bool multi=false) {
    Fixture f; std::vector<S::Source> sources;
    auto producer=std::make_shared<int>(7);
    for (size_t i=0;i<tensors;++i) {
        auto d=Tensor(heads,tokens,7,capacity,int(i)+3);
        sources.push_back({d.get(),d,multi?int(i%2):0,producer}); f.weakOwners.push_back(d);
        std::vector<uint8_t> gold;
        for (int h=0;h<heads;++h) gold.insert(gold.end(),d->cpuData+size_t(h)*d->strides[0]*2,
            d->cpuData+size_t(h)*d->strides[0]*2+size_t(tokens)*7*2);
        f.expected.push_back(std::move(gold));
        f.ownedOutputs.emplace_back(new Data()); f.outputs.push_back(f.ownedOutputs.back().get());
    }
    std::string reason; Need(S::Describe(sources,f.plan,reason),reason.c_str()); return f;
}
static uint64_t Hash(const Data &d) {
    uint64_t value=1469598103934665603ull;
    for (size_t i=0;i<d.GetBytes();++i) value=(value^d.cpuData[i])*1099511628211ull;
    return value;
}
static void Equal(const Fixture &f) {
    for (size_t i=0;i<f.outputs.size();++i) {
        const auto &d=*f.outputs[i];
        Need(d.dataDevice==DataDevice::CPU && !d.isFake && !d.isPinned,"snapshot must own ordinary CPU storage");
        Need(d.dataType==DataType::FLOAT16 && d.GetBytes()==f.expected[i].size(),"dtype/compact byte count mismatch");
        Need(std::memcmp(d.cpuData,f.expected[i].data(),f.expected[i].size())==0,"head stride padding copied or tail lost");
    }
}
static void Layout80() {
    auto f=Build(); Need(f.plan.heads.size()==80 && f.plan.tensors.size()==10,"10 tensor / 80 head geometry missing");
    size_t next=0;
    for (const auto &h:f.plan.heads) {
        Need(h.destinationOffset==next && h.bytes==17*7*2 && h.strideBytes==32*7*2,"descriptor geometry mismatch"); next+=h.bytes;
    }
    Need(next==f.plan.bytes,"layout sum mismatch");
}
static void ValidCopy(bool multi=false,int polls=0) {
    auto f=Build(10,8,17,32,multi); HostPins pins; FakeBackend b(pins);
    b.pendingPolls=polls; b.outputs=f.outputs; b.sourceOwners=f.weakOwners; S::Stats stats;
    Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,b,1,stats)==S::Result::Success,"valid production helper copy failed");
    Need(S::PublishedCopyGeneration(S::Result::Success,stats)==1,"successful helper lost publication generation");
    Need(stats.ready && stats.accepted==80 && b.invariants,"Ready before all terminals or source/lease freed early");
    Need(stats.joinedDevices==(multi?2u:1u),"not exactly one terminal join per device");
    Need(stats.pendingObservations==size_t(polls)*(multi?2u:1u),"pending observation hidden");
    for (auto &w:f.weakOwners) Need(w.expired(),"source owner retained beyond synchronous return");
    Equal(f); pins.Zero();
}
static void Failure(const std::string &mode) {
    auto f=Build(10,8,3,19,true); HostPins pins; FakeBackend b(pins);
    b.outputs=f.outputs; b.sourceOwners=f.weakOwners; S::Stats stats;
    if (mode=="partial") b.failAfter=9;
    if (mode=="copy") b.terminalFailure=true;
    if (mode=="seal") b.sealFailure=true;
    if (mode=="prepare") b.prepareFail=1;
    size_t cancellationChecks=0;
    auto cancel=[&] { ++cancellationChecks; return mode=="abort" && cancellationChecks>4; };
    const auto result=S::Copy(std::move(f.plan),f.outputs,pins.arena,b,9,stats,cancel);
    Need(result==S::Result::Failed && !stats.ready && b.invariants,"failure published or released live resources");
    for (auto *d:f.outputs) Need(d->cpuData==nullptr,"partial generation was materialized");
    Need(stats.joinedDevices==b.prepares-(mode=="prepare"?1:0),"prepared device not drained");
    pins.Zero();
}
static void ObservationBoundaryIsolation() {
    int ownership=1,errors=0;
    const bool logged=S::ObserveNoThrow([] { throw std::runtime_error("native_event_allocation"); },[&] { ++errors; });
    Need(!logged && errors==1 && ownership==1,"observation exception changed ownership or escaped");
    Need(!S::ObserveNoThrow([] { throw 1; },[] { throw 2; }),"error sink exception escaped");
    Need(S::ObserveNoThrow([&] { ++ownership; },[&] { ++errors; }) && ownership==2 && errors==1,"positive observation boundary failed");
}
static void RequestResourceLedgerContract() {
    RequestResourceLedger ledger;
    auto initial = ledger.Snapshot();
    Need(initial.transientOwners==1 && initial.temporaryReaders==0 &&
         initial.pendingOperations==0 && initial.liveGenerations.empty(),
         "request ledger did not start with context owner only");
    const uint64_t reader = ledger.AcquireReaderToken(101);
    const uint64_t pending = ledger.BeginPendingToken(101);
    auto active = ledger.Snapshot();
    Need(active.transientOwners==3 && active.temporaryReaders==1 &&
         active.temporaryLeases==1 && active.pendingOperations==1 &&
         active.liveGenerations.count(101)==1,
         "request ledger lost transient row ownership");
    ledger.ReleaseToken(pending);
    auto between = ledger.Snapshot();
    Need(between.transientOwners==2 && between.temporaryReaders==1 &&
         between.pendingOperations==0 && between.liveGenerations.count(101)==1,
         "request ledger erased generation while reader was still held");
    ledger.ReleaseToken(reader);
    ledger.ReleaseContextOwner();
    auto terminal = ledger.Snapshot();
    Need(terminal.transientOwners==0 && terminal.temporaryReaders==0 &&
         terminal.temporaryLeases==0 && terminal.pendingOperations==0 &&
         terminal.liveGenerations.empty(),
         "request ledger terminal zero was not fail-closed");
}
static void PinFallback() {
    auto f=Build(); HostPins pins; pins.fail=true; FakeBackend b(pins); S::Stats stats;
    Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,b,1,stats)==S::Result::Fallback,"pin failure did not fallback");
    Need(stats.generation==1 && S::PublishedCopyGeneration(S::Result::Fallback,stats)==0,"pin fallback exposed async publication generation");
    Need(b.prepares==0 && b.enqueues==0,"fallback happened after enqueue"); pins.Zero();
}
static void BudgetFallback() {
    HostPins pins; std::string why;
    auto a=pins.arena.Acquire(S::kLeaseBytes,1,why),b=pins.arena.Acquire(S::kLeaseBytes,2,why);
    Need(a && b && pins.arena.Observe().bytes==S::kProcessBytes,"global budget not shared");
    auto f=Build(); FakeBackend transport(pins); S::Stats stats;
    Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,transport,3,stats)==S::Result::Fallback,"third lease admitted");
    Need(stats.generation==3 && S::PublishedCopyGeneration(S::Result::Fallback,stats)==0,"budget fallback exposed async publication generation");
    Need(transport.prepares==0,"no-slot fallback after stream creation"); a.reset(); b.reset(); pins.Zero();
    Need(!pins.arena.Acquire(S::kLeaseBytes+1,4,why),"per-lease cap raised");
}
static void Reject(const std::string &mode) {
    auto d=Tensor(8,17,7,32); auto producer=std::make_shared<int>(2);
    std::shared_ptr<const void> owner=d;
    if (mode=="dtype") { d->dataType=DataType::BFLOAT16; d->UpdateUnitSize(); }
    if (mode=="stride") d->strides[0]=1;
    if (mode=="overflow") d->strides[0]=std::numeric_limits<uint64_t>::max();
    if (mode=="owner") owner.reset();
    if (mode=="extent") d->expansionBytes=1;
    S::Plan p; std::string why;
    Need(!S::Describe({{d.get(),owner,0,producer}},p,why) && !why.empty(),"bad descriptor accepted");
}
static void DataLifecycle() {
    auto f=Build(1,8,13,61); HostPins pins; FakeBackend backend(pins); S::Stats stats;
    Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,backend,5,stats)==S::Result::Success,"lifecycle snapshot failed");
    Data &d=*f.outputs[0]; const auto hash=Hash(d);
    Data restored; restored.CopyFrom(d); Need(Hash(restored)==hash,"restore CopyFrom mismatch");
    restored.Resize(d.dims); Need(Hash(restored)==hash,"Resize changed content");
    restored.Expansion({8,37,7});
    for (int h=0;h<8;++h) Need(std::memcmp(restored.cpuData+size_t(h)*restored.strides[0]*2,
        f.expected[0].data()+size_t(h)*13*7*2,13*7*2)==0,"Expansion corrupted noncompact heads");
    restored.FreeSpace(); Need(restored.cpuData==nullptr,"FreeSpace failed");
    // Recreate a fresh ordinary Data after FreeSpace and exercise its destructor.
    { Data again; again.CopyFrom(d); Need(Hash(again)==hash,"Data reconstruction/destructor contract"); }
    pins.Zero();
}
static void EightGenerations() {
    HostPins pins; std::vector<std::unique_ptr<Data>> old; std::vector<uint64_t> hashes;
    for (int g=1;g<=16;++g) {
        auto f=Build(1,8,g+1,g+9); FakeBackend backend(pins); S::Stats stats;
        Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,backend,g,stats)==S::Result::Success,"generation failed");
        if (g<=8) { hashes.push_back(Hash(*f.outputs[0])); old.push_back(std::move(f.ownedOutputs[0])); }
        for (size_t i=0;i<old.size();++i) Need(Hash(*old[i])==hashes[i],"old generation aliases staging/new owner");
        pins.Zero();
    }
    Need(std::set<uint64_t>(hashes.begin(),hashes.end()).size()==8,"generation hashes not independent");
    old.clear(); pins.Zero();
}
static void WrongReadyNegative() {
    auto f=Build(1); HostPins pins; FakeBackend backend(pins); backend.outputs=f.outputs;
    backend.sourceOwners=f.weakOwners; backend.pendingPolls=3; S::Stats stats;
    Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,backend,1,stats)==S::Result::Success && backend.invariants,
        "early Ready/source release negative oracle not exercised"); pins.Zero();
}
static void ProducerMismatch() {
    auto a=Tensor(8,3,7,19),b=Tensor(8,3,7,19);
    auto first=std::make_shared<int>(1),second=std::make_shared<int>(2);
    S::Plan p; std::string reason;
    Need(!S::Describe({{a.get(),a,0,first},{b.get(),b,0,second}},p,reason) &&
        reason=="conflicting_device_producers","mixed same-device producer dependencies accepted");
}
static void TailLengths() {
    for (int n:{1,3,19,33}) {
        auto f=Build(2,8,n,n+17); HostPins pins; FakeBackend b(pins); S::Stats stats;
        Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,b,n,stats)==S::Result::Success,"tail copy failed");
        Equal(f); pins.Zero();
    }
}
static void PinException() {
    S::Arena arena([](size_t)->void * { throw std::bad_alloc(); },[](void *) { std::terminate(); });
    std::string reason; Need(!arena.Acquire(4096,1,reason),"throwing pin allocator did not fallback");
    auto b=arena.Observe(); Need(b.bytes==0 && b.leases==0,"pin exception leaked reservation");
}
static void LockScopeDraftBatch() {
    auto f=Build(10,8,17,32,true); HostPins pins; ObservedBackend backend(pins);
    backend.outputs=f.outputs; backend.sourceOwners=f.weakOwners; S::Stats stats;
    std::mutex mtpCacheMutex; std::shared_ptr<const void> lease;
    p0lease::Observer::Reset(); p0lease::Observer::SetEnabled(true);
    {
        // Production order (qwen3_5.cpp F1a region): metadata validation and the
        // lease are taken under the real mutex, everything else after unlock.
        p0lease::ObservedLock guard(mtpCacheMutex,"f1a_validate");
        Need(p0lease::Observer::Depth()==1,"observed mutex did not report its own depth");
        lease=f.plan.sources.front().owner;
        Need(lease && !lease.owner_before(f.plan.sources.front().owner) &&
             !f.plan.sources.front().owner.owner_before(lease),
             "snapshot lease is not the same immutable source owner");
    }
    Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,backend,21,stats)==S::Result::Success,
        "leased draft batch copy failed");
    Need(backend.samples>=4,"backend boundaries were not observed");
    Need(backend.maxDepth==0,"prepare/enqueue/seal/wait observed inside the mtp lock");
    Need(p0lease::Observer::Depth()==0,"lock depth is not balanced after the copy");
    auto marks=p0lease::Observer::Drain();
    size_t enters=0,exits=0;
    for (const auto &m:marks) {
        if (m.event==p0lease::LockEvent::LockEnter) ++enters;
        if (m.event==p0lease::LockEvent::LockExit) ++exits;
    }
    Need(enters==1 && exits==1,"real mutex enter/exit not observed exactly once");
    Equal(f);
    for (auto *d:f.outputs) Need(d->cpuData!=nullptr,"CPU pack missing after unlock");
    lease.reset(); pins.Zero();
    p0lease::Observer::SetEnabled(false); p0lease::Observer::Reset();
}
static void LockScopeReverseMutation() {
    auto f=Build(2,8,5,13,false); HostPins pins; ObservedBackend backend(pins); S::Stats stats;
    std::mutex mtpCacheMutex;
    p0lease::Observer::Reset(); p0lease::Observer::SetEnabled(true);
    {
        // Negative control: the rejected revision-1 shape, helper (and its
        // wait/join/CPU pack) invoked while the mtp lock is still held.
        p0lease::ObservedLock guard(mtpCacheMutex,"reverse_mutation");
        Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,backend,22,stats)==S::Result::Success,
            "reverse-mutation copy failed");
    }
    Need(backend.maxDepth>=1,"lock-depth oracle missed an under-lock prepare/enqueue/seal/wait");
    p0lease::Observer::SetEnabled(false); p0lease::Observer::Reset();
    pins.Zero();
}
// ---------------------------------------------------------------- draft fence
// Exercises the same p0lease protocol the model runs (probe under the real
// mutex, clone with no lock, publish in the write's critical section) with a
// CPU storage that emits its own device-work/write marks.
struct LeaseStorage {
    static std::atomic<uint64_t> &Next() { static std::atomic<uint64_t> n(5000); return n; }
    static unsigned char Pattern(size_t i) { return (unsigned char)(i*7u+3u); }
    uint64_t generation;
    std::vector<unsigned char> bytes;
    LeaseStorage():generation(Next().fetch_add(1)+1) {}
    std::shared_ptr<LeaseStorage> Clone() const {
        auto copy=std::make_shared<LeaseStorage>();
        copy->bytes.assign(bytes.begin(),bytes.end());
        return copy;
    }
    void Append() { bytes.push_back(Pattern(bytes.size())); }
    bool Valid() const {
        for (size_t i=0;i<bytes.size();++i) if (bytes[i]!=Pattern(i)) return false;
        return true;
    }
};
struct LeaseBoard {
    std::mutex mutex;
    std::map<int,std::shared_ptr<LeaseStorage>> entries;
};
struct LeaseViolations {
    std::vector<std::string> items;
    bool Has(const char *prefix) const {
        for (const auto &v:items) if (v.rfind(prefix,0)==0) return true;
        return false;
    }
};
static LeaseViolations LeaseInspect(const std::vector<p0lease::LockMark> &marks) {
    LeaseViolations out;
    for (const auto &m:marks) {
        if ((m.event==p0lease::LockEvent::DeviceWorkBegin||m.event==p0lease::LockEvent::DeviceWorkEnd) && m.depth!=0)
            out.items.push_back(std::string("device_work_under_lock:")+m.tag);
        if (m.event==p0lease::LockEvent::WriteBegin) {
            long owners=-1;
            if (std::sscanf(m.tag.c_str(),"owners=%ld",&owners)!=1) owners=-1;
            if (owners!=1) out.items.push_back(std::string("write_without_exclusive_owner:")+m.tag);
        }
    }
    if (p0lease::Observer::Depth()!=0) out.items.push_back("unbalanced_lock_depth");
    return out;
}
static void WriteFenced(const std::shared_ptr<LeaseStorage> &draft,
                        p0lease::WriteTicket<LeaseStorage> &ticket);
static void LeaseWriter(LeaseBoard &board,int key) {
    // A lease can appear between the probe and the publish; the protocol then
    // requires a re-probe with a fresh clone instead of an in-place write.
    for (int attempt=0;attempt<8;++attempt) {
        p0lease::WriteTicket<LeaseStorage> ticket;
        {
            p0lease::ObservedLock probe(board.mutex,"draft_fence_probe");
            auto probeIt=board.entries.find(key);
            if (probeIt!=board.entries.end()) ticket=p0lease::BeginWrite(probeIt->second);
        }
        p0lease::CloneForWrite(ticket);
        p0lease::ObservedLock lock(board.mutex,"draft_append");
        auto found=board.entries.find(key);
        Need(found!=board.entries.end(),"writer lost its board entry");
        if (!p0lease::PublishWrite(found->second,ticket)) continue;
        WriteFenced(found->second,ticket);
        return;
    }
    Need(false,"writer could not publish its fence after re-probing");
}
static void WriteFenced(const std::shared_ptr<LeaseStorage> &draft,
                        p0lease::WriteTicket<LeaseStorage> &ticket) {
    // The ticket is not a lease: release it before sampling the owner count so
    // the oracle sees only real leases.
    ticket.source.reset();
    ticket.prepared.reset();
    char tag[96];
    std::snprintf(tag,sizeof(tag),"owners=%ld",(long)draft.use_count());
    p0lease::Observer::Note(p0lease::LockEvent::WriteBegin,tag);
    draft->Append();
    p0lease::Observer::Note(p0lease::LockEvent::WriteEnd,tag);
}
static void LeaseReader(LeaseBoard &board,int key) {
    std::shared_ptr<const LeaseStorage> lease;
    {
        p0lease::ObservedLock lock(board.mutex,"snapshot_lease");
        auto it=board.entries.find(key);
        if (it!=board.entries.end()) lease=it->second;
    }
    if (!lease) return;
    std::vector<unsigned char> copy=lease->bytes;
    lease.reset();
    for (size_t i=0;i<copy.size();++i) Need(copy[i]==LeaseStorage::Pattern(i),"leased generation mutated while copied");
}
static void DraftLeaseConcurrency() {
    LeaseBoard board; board.entries[1]=std::make_shared<LeaseStorage>();
    for (int i=0;i<4;++i) board.entries[1]->Append();
    p0lease::Observer::Reset(); p0lease::Observer::SetEnabled(true);
    std::atomic<int> failures(0);
    std::mutex errorMutex;
    std::string firstError;
    auto record=[&](const char *who,const std::exception &e) {
        std::lock_guard<std::mutex> lock(errorMutex);
        if (firstError.empty()) firstError=std::string(who)+": "+e.what();
        ++failures;
    };
    std::shared_ptr<LeaseStorage> heldReader;
    { p0lease::ObservedLock lock(board.mutex,"deterministic_reader_hold"); heldReader=board.entries[1]; }
    std::thread writer([&] { try { for (int i=0;i<300;++i) LeaseWriter(board,1); }
                               catch (const std::exception &e) { record("writer",e); } });
    std::thread reader([&] { try { for (int i=0;i<400;++i) LeaseReader(board,1); }
                               catch (const std::exception &e) { record("reader",e); } });
    writer.join(); reader.join();
    Need(heldReader->Valid(),"held reader generation mutated during clone");
    heldReader.reset();
    auto marks=p0lease::Observer::Drain();
    const LeaseViolations bad=LeaseInspect(marks);
    size_t deviceWork=0;
    for (const auto &m:marks) if (m.event==p0lease::LockEvent::DeviceWorkBegin) ++deviceWork;
    Need(failures==0,("concurrent fence run reported a failure: "+firstError).c_str());
    const std::string first = bad.items.empty()? std::string() : bad.items.front();
    Need(bad.items.empty(),("the production fence protocol violated the lock oracle: "+first).c_str());
    Need(deviceWork>0,"no copy-on-write clone happened, the concurrent lease path was not exercised");
    Need(board.entries[1]->Valid(),"draft payload corrupted by concurrent writes");
    Need(p0lease::Observer::Depth()==0,"lock depth unbalanced after the concurrent run");
    p0lease::Observer::SetEnabled(false); p0lease::Observer::Reset();
}
static void DraftLeaseNegativeCloneUnderLock() {
    LeaseBoard board; board.entries[1]=std::make_shared<LeaseStorage>();
    board.entries[1]->Append();
    std::shared_ptr<LeaseStorage> lease=board.entries[1];
    p0lease::Observer::Reset(); p0lease::Observer::SetEnabled(true);
    {
        // Negative control: device work (clone) performed inside the mutex.
        p0lease::ObservedLock lock(board.mutex,"negative_clone_under_lock");
        p0lease::WriteTicket<LeaseStorage> ticket=p0lease::BeginWrite(board.entries[1]);
        Need(ticket.copyNeeded,"negative control did not take a shared generation");
        p0lease::CloneForWrite(ticket);
    }
    const LeaseViolations bad=LeaseInspect(p0lease::Observer::Drain());
    Need(bad.Has("device_work_under_lock"),"oracle missed device work performed under the observed mutex");
    p0lease::Observer::SetEnabled(false); p0lease::Observer::Reset();
    lease.reset();
}
static void DraftLeaseNegativeWriteBeforeFence() {
    LeaseBoard board; board.entries[1]=std::make_shared<LeaseStorage>();
    board.entries[1]->Append();
    std::shared_ptr<LeaseStorage> lease;
    {
        p0lease::ObservedLock lock(board.mutex,"negative_probe");
        lease=board.entries[1];
    }
    p0lease::Observer::Reset(); p0lease::Observer::SetEnabled(true);
    {
        // Negative control: mutate the leased generation without any fence.
        p0lease::ObservedLock lock(board.mutex,"negative_write_before_fence");
        char tag[96];
        std::snprintf(tag,sizeof(tag),"owners=%ld",(long)board.entries[1].use_count());
        p0lease::Observer::Note(p0lease::LockEvent::WriteBegin,tag);
        board.entries[1]->Append();
        p0lease::Observer::Note(p0lease::LockEvent::WriteEnd,tag);
    }
    const LeaseViolations bad=LeaseInspect(p0lease::Observer::Drain());
    Need(bad.Has("write_without_exclusive_owner"),"oracle missed a write to a leased generation before the fence");
    p0lease::Observer::SetEnabled(false); p0lease::Observer::Reset();
    lease.reset();
}
static void DraftLeaseEraseDurability() {
    LeaseBoard board; board.entries[1]=std::make_shared<LeaseStorage>();
    for (int i=0;i<9;++i) board.entries[1]->Append();
    std::shared_ptr<LeaseStorage> lease;
    {
        p0lease::ObservedLock lock(board.mutex,"erase_probe");
        lease=board.entries[1];
    }
    {
        // Erase/replace while the snapshot leases the erased generation.
        p0lease::ObservedLock lock(board.mutex,"erase_replace");
        board.entries.erase(1);
        board.entries[1]=std::make_shared<LeaseStorage>();
        board.entries[1]->Append();
    }
    Need(lease->Valid(),"leased generation was corrupted by erase/replace");
    std::vector<unsigned char> copy=lease->bytes;
    lease.reset();
    for (size_t i=0;i<copy.size();++i) Need(copy[i]==LeaseStorage::Pattern(i),"erased generation storage was recycled");
    Need(board.entries[1]->Valid(),"replacement generation is not independently valid");
}
static void ObserverThrowIsolation() {
    auto f=Build(2,2,3,4,false); HostPins pins; FakeBackend backend(pins);
    backend.outputs=f.outputs; backend.sourceOwners=f.weakOwners; backend.pendingPolls=1;
    S::Stats stats; stats.contentDiagnostics=true;
    size_t cleanup=0,terminal=0;
    stats.observer=[&](const S::Observation &o) {
        if (o.event=="helper_cleanup") ++cleanup;
        if (o.event=="copy_terminal") ++terminal;
        throw std::runtime_error("observer_throw");
    };
    auto result=S::Copy(std::move(f.plan),f.outputs,pins.arena,backend,991,stats);
    Need(result==S::Result::Success && stats.observationError && stats.ready,
        "observer exception changed copy result or lost error fact");
    Need(backend.invariants && stats.terminalCopies==stats.accepted && terminal==1 && cleanup==1 && !stats.arena,
        "observer exception prevented terminal/drain/cleanup");
    Equal(f); pins.Zero();
}
static void ObservedContentAndPublicationNegative() {
    auto f=Build(2,2,3,4,false); HostPins pins; FakeBackend backend(pins);
    backend.outputs=f.outputs; backend.sourceOwners=f.weakOwners; backend.pendingPolls=1;
    S::Stats stats; stats.contentDiagnostics=true;
    std::vector<S::Observation> events; bool injected=false;
    stats.observer=[&](const S::Observation &o) {
        events.push_back(o);
        if (!injected && o.event=="cpu_head" && f.outputs[1]->cpuData) {
            f.outputs[1]->cpuData[0]^=1; injected=true;
        }
    };
    auto result=S::Copy(std::move(f.plan),f.outputs,pins.arena,backend,992,stats);
    Need(result==S::Result::Failed && !stats.ready,"real helper content mismatch reached Ready");
    Need(injected,"destination content mutation was not injected");
    pins.Zero();
    size_t terminal=0,ready=0; for (const auto &o:events) { if (o.event=="copy_terminal") ++terminal; if (o.event=="ready") ++ready; }
    Need(terminal==1 && ready==0 && stats.reason=="diagnostic_content_mismatch" && !stats.observationError,
        "terminal/ready ordering invalid or mismatch branch not exercised");
    Need(backend.invariants && events.back().event=="helper_cleanup" &&
         events.back().pending==0 && events.back().currentBytes==0 && events.back().currentLeases==0 &&
         events.back().sourceRefs==0 && !stats.arena,"mismatch cleanup incomplete");
    for (auto &w:f.weakOwners) Need(w.expired(),"mismatch retained helper source owner");
}

static void ObservedProductionCopy() {
    auto f=Build(10,8,17,32,true); HostPins pins; FakeBackend backend(pins);
    backend.outputs=f.outputs; backend.sourceOwners=f.weakOwners; backend.pendingPolls=2;
    S::Stats stats; stats.contentDiagnostics=true;
    std::vector<S::Observation> observations;
    stats.observer=[&](const S::Observation &o) {
        observations.push_back(o);
        std::cout<<"OBS_JSON {\"event\":\""<<o.event<<"\",\"copy_generation\":"<<o.generation
            <<",\"helper_seq\":"<<o.seq<<",\"helper_monotonic_ns\":"<<o.monoNs
            <<",\"tensor\":"<<o.tensor<<",\"head\":"<<o.head<<",\"effective_tokens\":"<<o.tokens
            <<",\"width\":"<<o.width<<",\"dtype\":"<<o.dtype<<",\"stride_bytes\":"<<o.strideBytes
            <<",\"tensor_heads\":"<<o.tensorHeads<<",\"source_extent\":"<<o.sourceExtent
            <<",\"source_allocation_bytes\":"<<o.sourceAllocationBytes
            <<",\"device_accepted\":"<<o.deviceAccepted<<",\"device_terminal\":"<<o.deviceTerminal
            <<",\"device_pending\":"<<o.devicePending
            <<",\"source_offset\":"<<o.sourceOffset<<",\"destination_offset\":"<<o.destinationOffset
            <<",\"bytes\":"<<o.bytes<<",\"device\":"<<o.device<<",\"content_fnv1a64\":\""<<std::hex<<o.digest<<std::dec
            <<"\",\"accepted\":"<<o.accepted<<",\"terminal\":"<<o.terminal<<",\"pending\":"<<o.pending
            <<",\"source_refs\":"<<o.sourceRefs<<",\"current_bytes\":"<<o.currentBytes
            <<",\"current_leases\":"<<o.currentLeases<<",\"peak_bytes\":"<<o.peakBytes
            <<",\"peak_leases\":"<<o.peakLeases<<",\"success\":"<<(o.success?1:0)<<"}\n";
    };
    Need(S::Copy(std::move(f.plan),f.outputs,pins.arena,backend,701,stats)==S::Result::Success,"observed actual helper failed");
    Equal(f); pins.Zero();
    Need(stats.pendingObservations==4,"cumulative poll count lost");
    Need(observations.back().event=="helper_cleanup" && observations.back().pending==0 &&
        observations.back().currentBytes==0 && observations.back().currentLeases==0 &&
        observations.back().sourceRefs==0,"actual cleanup balance not observed");
    Need(!stats.observationError && !stats.arena,"observer failed or retained Arena pointer");
    size_t staging=0,cpu=0,expectedSeq=0,deviceTerminals=0;
    std::map<int,size_t> acceptedByDevice;
    uint64_t lastMono=0;
    for (const auto &o:observations) { staging+=o.event=="staging_head"; cpu+=o.event=="cpu_head";
        Need(o.seq==++expectedSeq && o.monoNs>=lastMono,"raw helper timeline reordered"); lastMono=o.monoNs;
        Need(o.peakBytes>=o.currentBytes && o.peakLeases>=o.currentLeases,"invalid live budget peaks");
        if (o.event!="helper_cleanup")
            Need(o.currentBytes==19040 && o.currentLeases==1 && o.peakBytes==19040 && o.peakLeases==1,
                 "per-event live Arena counters lost");
        if (o.event=="copy_accepted") {
            Need(o.deviceAccepted==++acceptedByDevice[o.device] && o.deviceTerminal==0 &&
                 o.devicePending==o.deviceAccepted,"accepted per-device counters invalid");
        }
        if (o.event=="copy_terminal") {
            ++deviceTerminals;
            Need(o.deviceAccepted==40 && o.deviceTerminal==40 && o.devicePending==0,
                 "terminal per-device summary invalid");
        }
        if (o.event=="staging_head" || o.event=="cpu_head")
            Need(o.tensorHeads==8 && o.sourceExtent==3374 && o.sourceAllocationBytes>=o.sourceExtent &&
                 o.strideBytes==448 && o.bytes==238,"actual Data descriptor missing");
        if (o.event=="staging_head" || o.event=="cpu_head" || o.event=="ready")
            Need(o.accepted==80 && o.terminal==80 && o.pending==0,"content/ready observed before real terminal");
    }
    Need(staging==80 && cpu==80 && deviceTerminals==2,"missing actual content/device observations");
}

struct PendingGate {
    std::atomic<bool> entered{false}, release{false};
};
struct PendingBackend : FakeBackend {
    std::shared_ptr<PendingGate> gate;
    PendingBackend(HostPins &pins, std::shared_ptr<PendingGate> gate)
        : FakeBackend(pins), gate(std::move(gate)) {}
    S::Completion Wait(S::Channel &channel) noexcept override {
        gate->entered.store(true, std::memory_order_release);
        while (!gate->release.load(std::memory_order_acquire)) std::this_thread::yield();
        return FakeBackend::Wait(channel);
    }
};
static void PendingOwnership(bool cancel, bool multiwave) {
    auto fixture = std::make_shared<Fixture>(Build(2,2,3,3));
    std::vector<S::CopyWave> waves;
    size_t maxBytes = 0;
    std::string reason;
    if (multiwave) {
        for (size_t i=0;i<fixture->plan.sources.size();++i) {
            S::Plan one;
            Need(S::Describe({fixture->plan.sources[i]},one,reason),reason.c_str());
            maxBytes=std::max(maxBytes,one.bytes);
            waves.push_back({std::move(one),{fixture->outputs[i]}});
        }
    } else Need(S::MakeCopyWaves(fixture->plan.sources,fixture->outputs,waves,maxBytes,reason),reason.c_str());
    fixture->plan.sources.clear();
    HostPins pins;
    auto reservation=pins.arena.Acquire(maxBytes,901,reason);
    Need(bool(reservation),"pending staging reservation");
    auto gate=std::make_shared<PendingGate>();
    auto backend=std::unique_ptr<PendingBackend>(new PendingBackend(pins,gate));
    backend->outputs=fixture->outputs;
    auto handle=S::PendingSnapshotHandle::Submit(std::move(reservation),fixture,fixture,
        std::move(waves),std::unique_ptr<S::Backend>(backend.release()),pins.arena,
        {0,23,901,128},S::Stats{});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while (!gate->entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline)
        std::this_thread::yield();
    if (!gate->entered.load(std::memory_order_acquire)) {
        gate->release=true; handle->Join(); Need(false,"pending backend did not start");
    }
    Need(!handle->Ready(),"pending generation published before terminal");
    Need(pins.arena.Observe().leases==1,"pending job lost bounded staging owner");
    for (auto *output:fixture->outputs) Need(output->cpuData==nullptr,"pending destination visible early");
    if (cancel) handle->Cancel();
    gate->release.store(true,std::memory_order_release);
    const auto result=handle->Finish();
    Need(result==(cancel?S::Result::Failed:S::Result::Success),"pending terminal result mismatch");
    if (!cancel) Equal(*fixture);
    Need(handle->Ready(),"pending done not published");
    handle.reset(); pins.Zero();
}

struct RegistrySnapshot {
    int requestId=0,cachedLen=128;
    long long timestamp=0;
    std::vector<int> tokens;
    std::vector<std::string> mediaPageKeys;
    SnapshotPrefixIdentity matchIdentity;
    bool mtpValid=false,dflashValid=false;
};
static std::shared_ptr<RegistrySnapshot> RegistryValue(int request,int token) {
    auto value=std::make_shared<RegistrySnapshot>();
    value->requestId=request; value->tokens={token};
    value->matchIdentity.pages=1; value->matchIdentity.textPages=1;
    return value;
}
static void SnapshotPublicationPolicy() {
    std::vector<std::shared_ptr<RegistrySnapshot>> items;
    long long clock=0;
    for (int token=100;token<104;++token) S::PublishPrefixSnapshot(items,RegistryValue(2,token),4,8,clock);
    for (int token=0;token<4;++token) S::PublishPrefixSnapshot(items,RegistryValue(1,token),4,8,clock);
    for (int token=4;token<8;++token) S::PublishPrefixSnapshot(items,RegistryValue(1,token),4,8,clock);
    size_t a=0,b=0;
    for (const auto &value:items) { a+=value->requestId==1; b+=value->requestId==2; }
    Need(items.size()==8 && a==4 && b==4,"A continuation evicted peer B beyond original per-request cap");
    std::vector<int> oldGlobalOnly;
    for (int token=100;token<104;++token) oldGlobalOnly.push_back(token);
    for (int token=0;token<8;++token) {
        oldGlobalOnly.push_back(token);
        if (oldGlobalOnly.size()>8) oldGlobalOnly.erase(oldGlobalOnly.begin());
    }
    Need(std::none_of(oldGlobalOnly.begin(),oldGlobalOnly.end(),[](int token){return token>=100;}),
         "retention negative control did not expose missing per-request cap");
    auto oldest=items.front(); std::weak_ptr<RegistrySnapshot> retired=oldest; oldest.reset();
    auto publication=S::PublishPrefixSnapshot(items,RegistryValue(3,200),4,8,clock);
    Need(items.size()==8 && !publication.removed.empty() && !retired.expired(),"registry retired owner released inside caller lock");
    publication.removed.clear(); Need(retired.expired(),"retired registry owner retained after caller release");

    items.clear();
    auto richer=RegistryValue(7,42); richer->dflashValid=true;
    S::PublishPrefixSnapshot(items,richer,4,8,clock);
    auto poorer=RegistryValue(8,42); poorer->mediaPageKeys={""};
    publication=S::PublishPrefixSnapshot(items,poorer,4,8,clock);
    Need(publication.selected==richer && items.size()==1 && richer->requestId==8 &&
        richer->dflashValid,"equivalent text prefix lost richer DFlash state");
    auto mtp=RegistryValue(9,43); mtp->mtpValid=true;
    S::PublishPrefixSnapshot(items,mtp,4,8,clock);
    publication=S::PublishPrefixSnapshot(items,RegistryValue(10,43),4,8,clock);
    Need(publication.selected==mtp && mtp->mtpValid,"richer MTP state lost");

    auto pictureA=RegistryValue(11,44),pictureB=RegistryValue(11,44);
    std::vector<std::string> baseA{"image-A/mrope-1"},baseB{"image-B/mrope-1"};
    pictureA->mediaPageKeys=baseA; pictureB->mediaPageKeys=baseB;
    Need(BuildSnapshotPrefixIdentity(baseA,&baseA,"",1,pictureA->matchIdentity) &&
        BuildSnapshotPrefixIdentity(baseB,&baseB,"",1,pictureB->matchIdentity),"fixture media identity");
    Need(!S::SamePrefixSnapshotIdentity(*pictureA,*pictureB),"different image content deduplicated");
    pictureB->mediaPageKeys={"image-A/mrope-2"};
    Need(!S::SamePrefixSnapshotIdentity(*pictureA,*pictureB),"different MRoPE identity deduplicated");
    *pictureB=*pictureA;
    pictureB->matchIdentity.provenance="other-adapter";
    Need(!S::SamePrefixSnapshotIdentity(*pictureA,*pictureB),"different cache provenance deduplicated");
}

static void PreparedCanonicalCharge() {
    for (const size_t rebound : {size_t(150),size_t(40)}) {
        size_t total=100,canonical=100;
        Need(S::RebindPreparedCharge(total,canonical,rebound,1000,0),"charge rebind rejected");
        Need(total==rebound && canonical==rebound,"canonical rebind charge mismatch");
        // A canceled/stopped/early-failed worker has not applied replacement
        // payload. Terminal refund must still use this canonical gate charge.
        Need(S::ReleasePreparedCharge(total,canonical) && total==0,
             "cancel/Stop/early-failure refunded stale original charge");
    }
    size_t total=100,canonical=100;
    Need(!S::RebindPreparedCharge(total,canonical,150,120,0) && total==100 && canonical==100,
         "failed rebind changed budget");
    Need(!S::ReleasePreparedCharge(total,101) && total==100,"charge underflow forged zero");
}
int main() {
    struct Case { const char *id; std::function<void()> run; };
    std::vector<Case> cases={
        {"CPP-PREPARED-CANONICAL-CHARGE",PreparedCanonicalCharge},
        {"CPP-SNAPSHOT-PUBLICATION-POLICY",SnapshotPublicationPolicy},
        {"CPP-PENDING-OWNS-UNTIL-TERMINAL",[]{PendingOwnership(false,false);}},
        {"CPP-PENDING-CANCEL-DRAINS",[]{PendingOwnership(true,false);}},
        {"CPP-PENDING-MULTIWAVE",[]{PendingOwnership(false,true);}},
        {"CPP-OBSERVED-PRODUCTION-CONTENT",ObservedProductionCopy},
        {"CPP-OBSERVER-THROW-ISOLATION",ObserverThrowIsolation},
        {"CPP-NATIVE-EVENT-THROW-ISOLATION",ObservationBoundaryIsolation},
        {"CPP-REQUEST-RESOURCE-LEDGER",RequestResourceLedgerContract},
        {"CPP-OBSERVED-CONTENT-PUBLISH-NEGATIVE",ObservedContentAndPublicationNegative},
        {"CPP-DEFAULT-OFF",[]{Need(!S::ParseEnabled(nullptr) && !S::ParseEnabled("0") && S::ParseEnabled("1"),"default-off parser");}},
        {"CPP-DESCRIPTORS-80",Layout80}, {"CPP-STRIDE-TAIL",[]{ValidCopy();}},
        {"CPP-MULTIDEVICE-JOIN",[]{ValidCopy(true);}}, {"CPP-TIMEOUT-HOLDS",[]{ValidCopy(true,2);}},
        {"CPP-PARTIAL-ENQUEUE",[]{Failure("partial");}}, {"CPP-COPY-FAILURE",[]{Failure("copy");}},
        {"CPP-COMPLETE-RECORD-FAILURE",[]{Failure("seal");}}, {"CPP-PREPARE-FAILURE",[]{Failure("prepare");}},
        {"CPP-ABORT-DRAINS",[]{Failure("abort");}}, {"CPP-PIN-FALLBACK",PinFallback},
        {"CPP-NO-SLOT-BUDGET",BudgetFallback}, {"CPP-REJECT-DTYPE",[]{Reject("dtype");}},
        {"CPP-REJECT-STRIDE",[]{Reject("stride");}}, {"CPP-REJECT-OVERFLOW",[]{Reject("overflow");}},
        {"CPP-REJECT-OWNER",[]{Reject("owner");}}, {"CPP-REJECT-EXTENT",[]{Reject("extent");}},
        {"CPP-DATA-LIFECYCLE-RESTORE",DataLifecycle}, {"CPP-EIGHT-GENERATIONS",EightGenerations},
        {"CPP-EARLY-READY-NEGATIVE-ORACLE",WrongReadyNegative},
        {"CPP-PRODUCER-MISMATCH",ProducerMismatch}, {"CPP-ACTUAL-TAIL-LENGTHS",TailLengths},
        {"CPP-PIN-EXCEPTION",PinException},
        {"CPP-CUDA-INIT-REJECTED",[]{Need(P0SnapshotCudaRejectedCalls()==0,"CPU test attempted CUDA initialization/allocation/launch");}},
        {"CPP-LOCK-SCOPE-DRAFT-BATCH",LockScopeDraftBatch},
        {"CPP-LOCK-SCOPE-REVERSE-MUTATION",LockScopeReverseMutation},
        {"CPP-DRAFT-LEASE-COW-CONCURRENCY",DraftLeaseConcurrency},
        {"CPP-DRAFT-LEASE-NEGATIVE-CLONE-UNDER-LOCK",DraftLeaseNegativeCloneUnderLock},
        {"CPP-DRAFT-LEASE-NEGATIVE-WRITE-BEFORE-FENCE",DraftLeaseNegativeWriteBeforeFence},
        {"CPP-DRAFT-LEASE-ERASE-DURABILITY",DraftLeaseEraseDurability}
    };
    size_t failed=0;
    for (const auto &c:cases) {
        try { c.run(); std::cout<<c.id<<" expected=contract_satisfied actual=PASS exit=0\n"; }
        catch (const std::exception &e) { ++failed; std::cerr<<c.id<<" expected=contract_satisfied actual=FAIL exit=1 error="<<e.what()<<"\n"; }
    }
    std::cout<<"P0SnapshotCopyCpuRegression cases="<<cases.size()<<" failed="<<failed
             <<" cuda_initialization_calls="<<P0SnapshotCudaRejectedCalls()<<"\n";
    return failed || P0SnapshotCudaRejectedCalls()!=0 ? 1:0;
}
