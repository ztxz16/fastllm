// Diagnostics use the existing raw event identity/sequence/timestamp source.
// Content hashing is a separate opt-in correctness configuration, bounded by
// the request/checkpoint budget frozen in S04; it is off in performance A/B.
static bool Qwen35SnapshotDiagnostics() {
    static const bool enabled=p0snapshot::ParseEnabled(std::getenv("FASTLLM_SNAPSHOT_COPY_DIAGNOSTICS"));
    return enabled && GetFastllmObservabilityMode()!=FastllmObservabilityMode::Off;
}
static bool Qwen35SnapshotTrace() {
    static const bool enabled=p0snapshot::ParseEnabled(std::getenv("FASTLLM_SNAPSHOT_COPY_TRACE"));
    return enabled && GetFastllmObservabilityMode()!=FastllmObservabilityMode::Off;
}
template<class... Args> static bool Qwen35SnapshotEventNoThrow(
        uint64_t request,const char *event,const char *format,Args... args) noexcept {
    return p0snapshot::ObserveNoThrow([&] {
        FastllmEvent("snapshot",request,event,format,args...);
    },[&] {
        // Allocation-free failure fact. Missing raw events and this marker are
        // evidence failure, while the copy/ownership result remains unchanged.
        std::fprintf(stderr,"[SnapshotCopy] observation_error=1 native_handle=%llu event=%s\n",
            (unsigned long long)request,event?event:"unknown");
    });
}
static p0snapshot::Observer Qwen35SnapshotObserver(uint64_t request,uint64_t draftGeneration,int cachedLen) {
    if (!Qwen35SnapshotTrace()) return {};
    return [=](const p0snapshot::Observation &o) {
        if (!Qwen35SnapshotEventNoThrow(request,o.event.c_str(),
            "copy_generation=%llu draft_generation=%llu cached_len=%d helper_seq=%zu helper_monotonic_ns=%llu "
            "producer_event=%llu tensor=%zu head=%zu effective_tokens=%zu width=%zu dtype=%d stride_bytes=%zu source_offset=%zu "
            "destination_offset=%zu bytes=%zu device=%d content_fnv1a64=%016llx "
            "accepted=%zu terminal=%zu pending=%zu source_refs=%zu current_bytes=%zu current_leases=%zu "
            "peak_bytes=%zu peak_leases=%zu tensor_heads=%zu source_extent=%zu source_allocation_bytes=%zu "
            "device_accepted=%zu device_terminal=%zu device_pending=%zu success=%d",
            (unsigned long long)o.generation,(unsigned long long)draftGeneration,cachedLen,o.seq,
            (unsigned long long)o.monoNs,(unsigned long long)o.producerId,o.tensor,o.head,o.tokens,o.width,o.dtype,o.strideBytes,
            o.sourceOffset,o.destinationOffset,o.bytes,o.device,(unsigned long long)o.digest,
            o.accepted,o.terminal,o.pending,o.sourceRefs,o.currentBytes,o.currentLeases,o.peakBytes,o.peakLeases,o.tensorHeads,o.sourceExtent,o.sourceAllocationBytes,
            o.deviceAccepted,o.deviceTerminal,o.devicePending,o.success?1:0))
            throw std::runtime_error("snapshot observation failed");
    };
}
// F1a checked adapter. Included inside fastllm's anonymous namespace.
// No CUDA call in this file is executed unless the opt-in path is selected.
class Qwen35SnapshotCudaDeviceScope {
    int old=-1;
public:
    explicit Qwen35SnapshotCudaDeviceScope(int device) {
        if (cudaGetDevice(&old)!=cudaSuccess || cudaSetDevice(device)!=cudaSuccess)
            throw std::runtime_error("snapshot copy device selection failed");
    }
    ~Qwen35SnapshotCudaDeviceScope() { if (old>=0 && cudaSetDevice(old)!=cudaSuccess) std::terminate(); }
};
static std::shared_ptr<void> Qwen35SnapshotRecordDraftProducer(int device,uint64_t draftGeneration,int cachedLen) {
    Qwen35SnapshotCudaDeviceScope scope(device);
    cudaEvent_t event=nullptr;
    const auto created=cudaEventCreateWithFlags(&event,cudaEventDisableTiming);
    if (created==cudaErrorMemoryAllocation) return {};
    if (created!=cudaSuccess) throw std::runtime_error("snapshot producer event creation failed");
    if (cudaEventRecord(event,cudaStreamPerThread)!=cudaSuccess) {
        cudaEventDestroy(event);
        throw std::runtime_error("snapshot producer event record failed");
    }
    auto owner=std::shared_ptr<void>(reinterpret_cast<void *>(event),[device](void *ptr) {
        Qwen35SnapshotCudaDeviceScope scope(device);
        if (cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ptr))!=cudaSuccess) std::terminate();
    });
    if (Qwen35SnapshotTrace())
        Qwen35SnapshotEventNoThrow(0,"producer_recorded",
            "draft_generation=%llu cached_len=%d device=%d producer_event=%llu",
            (unsigned long long)draftGeneration,cachedLen,device,
            (unsigned long long)reinterpret_cast<uintptr_t>(event));
    return owner;
}
class Qwen35SnapshotCudaChannel final: public p0snapshot::Channel {
public:
    int device;
    cudaStream_t stream=nullptr;
    cudaEvent_t done=nullptr;
    bool recorded=false, terminal=false;
    std::shared_ptr<void> producer;
    explicit Qwen35SnapshotCudaChannel(int d):device(d) {}
    ~Qwen35SnapshotCudaChannel() override {
        Qwen35SnapshotCudaDeviceScope scope(device);
        // Even an exception between prepare and enqueue cannot release a stream
        // that might still hold a copy. Never trust the void logging wrappers.
        if (stream && !terminal && cudaStreamSynchronize(stream)!=cudaSuccess) std::terminate();
        if (done && cudaEventDestroy(done)!=cudaSuccess) std::terminate();
        if (stream && cudaStreamDestroy(stream)!=cudaSuccess) std::terminate();
    }
};
class Qwen35SnapshotCudaBackend final: public p0snapshot::Backend {
    uint64_t request = 0, generation = 0;
public:
    explicit Qwen35SnapshotCudaBackend(uint64_t request = 0, uint64_t generation = 0)
        : request(request), generation(generation) {}
    std::unique_ptr<p0snapshot::Channel> Prepare(int device,const std::shared_ptr<void> &producer) override {
        if (!producer) return {};
        Qwen35SnapshotCudaDeviceScope scope(device);
        auto ch=std::unique_ptr<Qwen35SnapshotCudaChannel>(new Qwen35SnapshotCudaChannel(device));
        ch->producer=producer;
        if (cudaStreamCreateWithFlags(&ch->stream,cudaStreamNonBlocking)!=cudaSuccess ||
            cudaEventCreateWithFlags(&ch->done,cudaEventDisableTiming)!=cudaSuccess ||
            cudaStreamWaitEvent(ch->stream,reinterpret_cast<cudaEvent_t>(producer.get()),0)!=cudaSuccess) return {};
        return std::unique_ptr<p0snapshot::Channel>(ch.release());
    }
    bool Enqueue(p0snapshot::Channel &base,void *dst,const void *src,size_t bytes) noexcept override {
        auto &ch=static_cast<Qwen35SnapshotCudaChannel &>(base);
        Qwen35SnapshotCudaDeviceScope scope(ch.device);
        const uint64_t begin = Qwen35SnapshotTrace() ? FastllmObservationTimestampNs() : 0;
        const bool success = cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToHost,ch.stream)==cudaSuccess;
        if (begin) Qwen35SnapshotEventNoThrow(request,"snapshot_copy_api",
            "copy_generation=%llu host_tid=%llu enqueue_begin_ns=%llu enqueue_end_ns=%llu device=%d bytes=%zu success=%d",
            (unsigned long long)generation,(unsigned long long)RequestDiagnosticThreadId(),
            (unsigned long long)begin,(unsigned long long)FastllmObservationTimestampNs(),ch.device,bytes,success?1:0);
        return success;
    }
    bool Seal(p0snapshot::Channel &base) noexcept override {
        auto &ch=static_cast<Qwen35SnapshotCudaChannel &>(base);
        Qwen35SnapshotCudaDeviceScope scope(ch.device);
        ch.recorded=cudaEventRecord(ch.done,ch.stream)==cudaSuccess;
        return ch.recorded;
    }
    p0snapshot::Completion Wait(p0snapshot::Channel &base) noexcept override {
        auto &ch=static_cast<Qwen35SnapshotCudaChannel &>(base);
        Qwen35SnapshotCudaDeviceScope scope(ch.device);
        if (ch.recorded && cudaEventSynchronize(ch.done)==cudaSuccess) {
            ch.terminal=true; return p0snapshot::Completion::Success;
        }
        // A missing/failed event is NOT a terminal ack. Drain the actual stream.
        // A poisoned device with no provable drain is fail-stop: do not unwind
        // and release the caller's cache lock or staging while DMA is unknown.
        if (cudaStreamSynchronize(ch.stream)!=cudaSuccess) {
            std::fprintf(stderr,"[SnapshotCopy] terminal unproven; refusing source release\n");
            std::terminate();
        }
        ch.terminal=true; return p0snapshot::Completion::Failure;
    }
};
static p0snapshot::Arena &Qwen35SnapshotProcessArena() {
    static p0snapshot::Arena arena([](size_t bytes)->void * {
        void *base=nullptr;
        // Portable staging allows one lease to cover several source devices.
        const auto status=cudaHostAlloc(&base,bytes,cudaHostAllocPortable);
        if (status==cudaSuccess) return base;
        // Only local pin-resource exhaustion is eligible for synchronous retry.
        // A previous asynchronous/device error must not be masked by fallback.
        if (status==cudaErrorMemoryAllocation || status==cudaErrorNotSupported) return nullptr;
        std::fprintf(stderr,"[SnapshotCopy] pin allocation reported device error; refusing fallback\n");
        std::terminate();
    },[](void *base) {
        if (cudaFreeHost(base)!=cudaSuccess) std::terminate();
    });
    return arena;
}
// One independently owned linear tensor. Original pooled/borrowed Data is read
// only while the scheduler's actual producer callback owns the current boundary.
// Flattened transport retains original metadata for scheduler-side restoration.
struct Qwen35FrozenSnapshotTensor {
    Data storage;
    Data view;
    std::vector<int> originalDims;
    DataType originalType=DataType::FLOAT16;
    bool linear=false,transposed=false;
    std::shared_ptr<void> freezeDone;
};
static std::shared_ptr<Qwen35FrozenSnapshotTensor> Qwen35FreezeSnapshotTensorOnProducer(
        const Data &source,int device,uint64_t generation,int cachedLen) {
    // Actual producer-thread entry only. Unsupported layouts use the unchanged
    // synchronous path; no guessing/padding reads or implicit Data assignment.
    if (source.multiDeviceData ||
        (source.dataType!=DataType::FLOAT16 && source.dataType!=DataType::BFLOAT16) ||
        source.unitSize!=2 || source.unitSizeDiv!=1 || source.dims.empty() ||
        source.dims.size()!=source.strides.size() ||
        (source.isPagedKVCache && !source.isLinearAttention)) return {};
    size_t count=1;
    for (size_t i=source.dims.size();i-->0;) {
        if (source.dims[i]<=0 || source.strides[i]!=count ||
            !p0snapshot::Mul(count,size_t(source.dims[i]),count)) return {};
    }
    if (!count || count>size_t(std::numeric_limits<int>::max())) return {};
    size_t bytes=0;
    if (!p0snapshot::Mul(count,size_t(2),bytes) ||
        bytes>(source.expansionBytes?source.expansionBytes:source.GetBytes())) return {};
    auto frozen=std::make_shared<Qwen35FrozenSnapshotTensor>();
    frozen->originalDims=source.dims;
    frozen->originalType=source.dataType;
    frozen->linear=source.isLinearAttention;
    frozen->transposed=source.isLinearAttentionTransposed;
    Data &dst=frozen->storage;
    // Opaque 16-bit transport. BF16 is bit-copied, never numerically converted.
    // Scheduler must restore originalType/dims before publishing CPU snapshot.
    dst.dataType=DataType::FLOAT16;dst.UpdateUnitSize();dst.Resize({1,1,int(count)});
    dst.name=source.name;dst.cacheUid=source.cacheUid;
    if (source.dataDevice==DataDevice::CPU) {
        if (!source.cpuData) return {};
        dst.Allocate(false);std::memcpy(dst.cpuData,source.cpuData,bytes);
    } else if (source.dataDevice==DataDevice::CUDA) {
        if (device<0 || !source.cudaData) return {};
        Qwen35SnapshotCudaDeviceScope scope(device);
        dst.dataDevice=DataDevice::CUDA;dst.dataDeviceIds={device};dst.Allocate(false);
        if (!dst.cudaData) return {};
        const auto freezeStart=std::chrono::steady_clock::now();
        if (Qwen35SnapshotTrace()) Qwen35SnapshotEventNoThrow(0,"linear_freeze_begin",
            "copy_generation=%llu cached_len=%d device=%d bytes=%zu",
            (unsigned long long)generation,cachedLen,device,bytes);
        // Ordered on the ACTUAL producer stream, after its writes. A synchronous
        // freeze boundary protects original Data even when next forward changes
        // thread/stream, releases the context or reuses a pooled slot. Only this
        // bounded D2D is joined here; D2H+CPU pack run on PendingSnapshotHandle.
        const auto copied=cudaMemcpyAsync(dst.cudaData,source.cudaData,bytes,
            cudaMemcpyDeviceToDevice,cudaStreamPerThread);
        if (copied!=cudaSuccess) {
            if (cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess) std::terminate();
            throw std::runtime_error("snapshot linear freeze enqueue failed");
        }
        cudaEvent_t doneEvent=nullptr;
        if (cudaEventCreateWithFlags(&doneEvent,cudaEventDisableTiming)!=cudaSuccess ||
            cudaEventRecord(doneEvent,cudaStreamPerThread)!=cudaSuccess) {
            if (doneEvent) cudaEventDestroy(doneEvent);
            if (cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess) std::terminate();
            return {};
        }
        frozen->freezeDone=std::shared_ptr<void>(reinterpret_cast<void*>(doneEvent),
            [device](void *ptr) {
                Qwen35SnapshotCudaDeviceScope eventScope(device);
                if (cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ptr))!=cudaSuccess)
                    std::terminate();
            });
        if (cudaEventSynchronize(doneEvent)!=cudaSuccess) {
            if (cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess) std::terminate();
            throw std::runtime_error("snapshot linear freeze terminal failed");
        }
        if (Qwen35SnapshotTrace()) Qwen35SnapshotEventNoThrow(0,"linear_freeze_terminal",
            "copy_generation=%llu cached_len=%d device=%d bytes=%zu freeze_ms=%.6f",
            (unsigned long long)generation,cachedLen,device,bytes,
            p0snapshot::Milliseconds(freezeStart));
    } else return {};
    // Fake view borrows ONLY our owned immutable storage; Source.owner below
    // is the actual shared owner, never a no-op shared_ptr around a borrowed ptr.
    frozen->view.FakeFrom(dst,0);frozen->view.Resize({1,1,int(count)});
    frozen->view.name=source.name;frozen->view.cacheUid=source.cacheUid;
    return frozen;
}
static p0snapshot::Source Qwen35FrozenSnapshotSource(
        const std::shared_ptr<Qwen35FrozenSnapshotTensor> &frozen,int device,
        const std::shared_ptr<void> &groupProducer) {
    return {&frozen->view,frozen,
        frozen->storage.dataDevice==DataDevice::CPU?-1:device,groupProducer};
}
static bool Qwen35RestoreFrozenSnapshotMetadata(
        const Qwen35FrozenSnapshotTensor &frozen,Data &destination) {
    if (destination.dataDevice!=DataDevice::CPU || !destination.cpuData ||
        destination.dataType!=DataType::FLOAT16 ||
        destination.Count(0)!=frozen.view.Count(0)) return false;
    destination.Reshape(frozen.originalDims);
    destination.dataType=frozen.originalType;destination.UpdateUnitSize();
    destination.isLinearAttention=frozen.linear;
    destination.isLinearAttentionTransposed=frozen.transposed;
    return true;
}
static p0snapshot::Result Qwen35SnapshotCopyDraftBatch(
        const std::vector<p0snapshot::Source> &sources,const std::vector<Data *> &destinations,
        const void *model,int request,int cachedLen,uint64_t draftGeneration,uint64_t nativeHandle,
        uint64_t *copyGenerationOut) {
    p0snapshot::Plan plan; std::string reason;
    if (!p0snapshot::Describe(sources,plan,reason)) {
        std::fprintf(stderr,"[SnapshotCopy] draft rejected reason=%s\n",reason.c_str());
        return p0snapshot::Result::Failed;
    }
    static std::atomic<uint64_t> generation{0};
    p0snapshot::Stats stats; Qwen35SnapshotCudaBackend backend;
    p0snapshot::ObserveNoThrow([&] {
        stats.observer=Qwen35SnapshotObserver(nativeHandle,draftGeneration,cachedLen);
    },[&] {
        stats.observationError=true;
        std::fprintf(stderr,"[SnapshotCopy] observation_error=1 native_handle=%llu event=observer_factory\n",
            (unsigned long long)nativeHandle);
    });
    stats.contentDiagnostics=Qwen35SnapshotDiagnostics();
    const auto result=p0snapshot::Copy(std::move(plan),destinations,
        Qwen35SnapshotProcessArena(),backend,++generation,stats);
    if (copyGenerationOut) *copyGenerationOut=p0snapshot::PublishedCopyGeneration(result,stats);
    // Emitted after Copy returns: cleanup observers have also completed. A failed
    // observer cannot alter the copy result, but must prevent evidence acceptance.
    if (Qwen35SnapshotTrace())
        Qwen35SnapshotEventNoThrow(nativeHandle,"copy_summary",
            "copy_generation=%llu draft_generation=%llu cached_len=%d accepted=%zu terminal=%zu pending=%zu "
            "ready=%d observation_error=%d result=%s",
            (unsigned long long)stats.generation,(unsigned long long)draftGeneration,cachedLen,
            stats.accepted,stats.terminalCopies,stats.accepted-stats.terminalCopies,
            stats.ready?1:0,stats.observationError?1:0,
            result==p0snapshot::Result::Success?"success":result==p0snapshot::Result::Fallback?"fallback":"failed");
    const auto budget=Qwen35SnapshotProcessArena().Observe();
    std::fprintf(stderr,"[SnapshotCopy] model=%p request=%d generation=%llu cached_len=%d "
        "draft_generation=%llu tensors=%zu heads=%zu bytes=%zu accepted=%zu joined_devices=%zu producer=recorded "
        "enqueue_ms=%.3f join_ms=%.3f cpu_pack_ms=%.3f pinned_peak=%zu reason=%s ready=%d observation_error=%d\n",
        model,request,(unsigned long long)stats.generation,cachedLen,
        (unsigned long long)draftGeneration,stats.tensors,stats.heads,
        stats.bytes,stats.accepted,stats.joinedDevices,stats.enqueueMs,stats.joinMs,stats.cpuPackMs,
        budget.peakBytes,stats.reason.empty()?"none":stats.reason.c_str(),stats.ready?1:0,stats.observationError?1:0);
    return result;
}
