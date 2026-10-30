#pragma once

// Native, single-GPU decode handoff. A CPU dispatcher owns NUMA execution;
// the caller only submits graphs. Stream memory operations wait without
// occupying an SM. Every layer owns its activation/result/control buffers.
#include <deque>

namespace {
#if defined(USE_NUMAS) && !defined(USE_ROCM) && CUDART_VERSION >= 11070
struct NativeMoeControl {
    alignas(64) unsigned published = 0;
    alignas(64) unsigned computed = 0;
    int stagedCount = 0;
    alignas(64) int stagedByHits[kMaxTopK + 1];
    alignas(64) int32_t ids[kMaxTopK], resident[kMaxTopK], staged[kMaxTopK], gpu[kMaxTopK];
    float scores[kMaxTopK];
};

__global__ void PublishNativeMoe(NativeMoeControl *box, float *hostInput,
        const __nv_bfloat16 *input, const int32_t *indices, const float *scores,
        const int32_t *keys, const int32_t *slotKeys, int32_t *slots,
        unsigned long long *hits, unsigned long long *misses,
        int base, int experts, int hidden, int topk, bool publish = false) {
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        hostInput[c] = __bfloat162float(input[c]);
    if (threadIdx.x < 32) {
        const int r = threadIdx.x, e = r < topk ? indices[r] : -1;
        const int key = base + e;
        const int slot = e >= 0 && e < experts ? keys[key] : -1;
        const bool resident = slot >= 0 && slotKeys[slot] == key;
        const unsigned mask = __ballot_sync(0xffffffff, r < topk && resident);
        if (!r) { *hits += __popc(mask); *misses += topk - __popc(mask); }
        if (r < topk) {
            box->ids[r] = e; box->scores[r] = scores[r];
            box->resident[r] = slots[r] = resident ? slot : -1;
            if (publish) {
                const int selected = box->stagedByHits[__popc(mask)];
                const int at = fastllm::cuda::dsv41_cache::NativeStagedSlot(mask, r, selected);
                box->staged[r] = slots[topk + r] = at;
                box->gpu[r] = slots[2 * topk + r] = resident || at >= 0 ? e : -1;
                if (at >= 0) {
                    slots[3 * topk + at] = e;
                    slots[4 * topk + at] = at;
                }
                if (!r) box->stagedCount = slots[5 * topk] = selected;
            }
        }
    }
    __threadfence_system();
    if (publish) {
        __syncthreads();
        if (!threadIdx.x) atomicExch_system(&box->published, 1u);
    }
}

__global__ void ReadNativeMoePlan(const NativeMoeControl *box, int32_t *staged, int32_t *gpu, int topk) {
    if (threadIdx.x < topk) {
        staged[threadIdx.x] = box->staged[threadIdx.x];
        gpu[threadIdx.x] = box->gpu[threadIdx.x];
    }
}

struct NativeMoeMemOps {
    decltype(&cuCtxGetCurrent) context = nullptr;
    decltype(&cuGraphAddBatchMemOpNode) add = nullptr;
    bool ready = false;
    NativeMoeMemOps() {
        ready = cudaGetDriverEntryPoint("cuCtxGetCurrent", reinterpret_cast<void **>(&context),
                    cudaEnableDefault) == cudaSuccess && context &&
                cudaGetDriverEntryPoint("cuGraphAddBatchMemOpNode", reinterpret_cast<void **>(&add),
                    cudaEnableDefault) == cudaSuccess && add;
        if (!ready) (void)cudaGetLastError();
    }
    bool Append(unsigned *address, bool wait, unsigned value) const {
        cudaStreamCaptureStatus status;
        cudaGraph_t graph = nullptr;
        const cudaGraphNode_t *deps = nullptr;
        size_t count = 0;
        CUcontext ctx = nullptr;
        if (!ready || context(&ctx) != CUDA_SUCCESS ||
            cudaStreamGetCaptureInfo(cudaStreamPerThread, &status, nullptr, &graph, &deps, &count) != cudaSuccess ||
            status != cudaStreamCaptureStatusActive) return false;
        CUstreamBatchMemOpParams op{};
        if (wait) {
            op.waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
            op.waitValue.address = reinterpret_cast<CUdeviceptr>(address);
            op.waitValue.value = value; op.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
        } else {
            op.writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
            op.writeValue.address = reinterpret_cast<CUdeviceptr>(address);
            op.writeValue.value = value;
        }
        CUDA_BATCH_MEM_OP_NODE_PARAMS params{ctx, 1, &op, 0};
        CUgraphNode node = nullptr;
        if (add(&node, reinterpret_cast<CUgraph>(graph),
                reinterpret_cast<const CUgraphNode *>(deps), count, &params) != CUDA_SUCCESS) return false;
        auto next = reinterpret_cast<cudaGraphNode_t>(node);
        return cudaStreamUpdateCaptureDependencies(cudaStreamPerThread, &next, 1,
                    cudaStreamSetCaptureDependencies) == cudaSuccess;
    }
};

struct NativeMoeMailbox {
    struct Layer {
        NativeMoeControl *host = nullptr, *mapped = nullptr;
        float *hostData = nullptr, *mappedData = nullptr;
        float *partial = nullptr;
        int32_t *slots = nullptr; // resident, staged, GPU route IDs
        fastllm::Data quantized, gate;
        cudaGraphExec_t exec = nullptr, finishExec = nullptr;
        cudaGraph_t fullGraph = nullptr;
        NativeMoeMailbox *owner = nullptr;
        int table = -1, modelLayer = -1;
        std::vector<fastllm::Data *> weights;
        cudaEvent_t uploadBegin = nullptr, uploadDone = nullptr;
        cudaEvent_t begin = nullptr, residentDone = nullptr, stagedBegin = nullptr, done = nullptr, completed = nullptr;
        std::array<const void *, 4> pointers{};
        fastllm::MoeDecodeOverlapScheduler cost;
        int hidden = 0, topk = 0, hits = 0, staged = 0;
        bool timed = true;
        std::atomic<bool> pending{false}, submitted{false};
        ~Layer() {
            if (pending && completed) cudaEventSynchronize(completed);
            if (fullGraph) cudaGraphDestroy(fullGraph);
            if (exec) cudaGraphExecDestroy(exec);
            if (finishExec) cudaGraphExecDestroy(finishExec);
            for (auto e : {begin, residentDone, stagedBegin, done, completed, uploadBegin, uploadDone}) if (e) cudaEventDestroy(e);
            cudaFreeHost(host); cudaFreeHost(hostData); cudaFree(partial); cudaFree(slots);
        }
        bool Init(const OffloadLayout &layout, int routes, int device) {
            hidden = layout.hidden; topk = routes;
            if (cudaHostAlloc(&host, sizeof(*host), cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(&mapped, host, 0) != cudaSuccess ||
                cudaHostAlloc(&hostData, (topk + 1) * hidden * sizeof(float), cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(&mappedData, hostData, 0) != cudaSuccess ||
                cudaMalloc(&partial, topk * hidden * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&slots, (5 * topk + 1) * sizeof(int32_t)) != cudaSuccess) return false;
            new (host) NativeMoeControl{};
            std::fill_n(hostData, (topk + 1) * hidden, 0.0f);
            for (auto *e : {&begin, &residentDone, &stagedBegin, &done, &completed, &uploadBegin, &uploadDone})
                if (cudaEventCreate(e) != cudaSuccess) return false;
            for (auto *d : {&quantized, &gate}) {
                d->dataType = fastllm::BFLOAT16;
                d->Resize(d == &quantized ? std::vector<int>{1, hidden} : std::vector<int>{topk, layout.inter});
                d->ToDevice(fastllm::CUDA, {device}, false); d->Allocate(false);
            }
            return true;
        }
    };
    struct Request {
        Layer *work;
        FastllmMoeRoutePhase phase;
    };
    OffloadGroup &group;
    DeviceCache &cache;
    NativeMoeMemOps memops;
    std::unique_ptr<DecodeOverlapWorkspace> copy;
    std::map<int, std::unique_ptr<Layer>> layers, fullLayers;
    std::thread dispatcher;
    std::mutex mutex;
    std::condition_variable wake, drained;
    std::deque<Request> queue;
    bool busy = false, stopping = false;
    std::exception_ptr error;
    Layer *previous = nullptr, *fullLast = nullptr;
    bool failed = false;

    NativeMoeMailbox(OffloadGroup &g, DeviceCache &c) : group(g), cache(c) {}
    void Finish(bool observe) {
        std::unique_lock<std::mutex> lock(mutex);
        drained.wait(lock, [&] { return queue.empty() && !busy; });
        auto failure = std::exchange(error, nullptr);
        lock.unlock();
        if (observe && previous) { Observe(*previous); previous = nullptr; }
        if (observe && fullLast) {
            // The full graph orders all routed reductions before the last one.
            // One completion protects every CPU result and timing sample.
            checkCudaErrors("Native full completion", cudaEventSynchronize(fullLast->completed));
            for (auto &entry : fullLayers) Observe(*entry.second, false);
            fullLast = nullptr;
        }
        if (failure) std::rethrow_exception(failure);
    }
    ~NativeMoeMailbox() {
        try { Finish(true); } catch (...) {}
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_one();
        if (dispatcher.joinable()) dispatcher.join();
        cudaSetDevice(cache.device);
        layers.clear(); fullLayers.clear(); copy.reset();
    }
    void Observe(Layer &w, bool synchronize = true) {
        if (!w.pending) return;
        if (synchronize) checkCudaErrors("Native mailbox completion", cudaEventSynchronize(w.completed));
        if (!w.timed) { w.pending = false; return; }
        float ms = 0;
        if (w.hits) {
            checkCudaErrors("Native resident timing", cudaEventElapsedTime(&ms, w.begin, w.residentDone));
            w.cost.residentExpert.Observe(ms * 1000 / w.hits);
        }
        if (w.staged) {
            checkCudaErrors("Native upload timing", cudaEventElapsedTime(&ms, w.fullGraph ? w.uploadBegin : copy->copyStart,
                w.fullGraph ? w.uploadDone : copy->copyDone));
            w.cost.copiedExpert.Observe(ms * 1000 / w.staged);
            checkCudaErrors("Native staged timing", cudaEventElapsedTime(&ms, w.fullGraph ? w.uploadDone : w.stagedBegin, w.done));
            w.cost.stagedExpert.Observe(ms * 1000 / w.staged);
        }
        w.pending = false;
    }
    void Run() {
        cudaSetDevice(cache.device);
        for (;;) {
            Request r;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) return;
                r = std::move(queue.front()); queue.pop_front(); busy = true;
            }
            auto &w = *r.work;
            auto &box = *w.host;
            const bool full = w.fullGraph != nullptr;
            // Publication is downstream of the preceding layer's reduction,
            // so its DMA scratch and timing events can now be reused.
            try {
                unsigned polls = 0;
                const double deadline = HybridNowUs() + 30000000;
                while (!__atomic_load_n(&box.published, __ATOMIC_ACQUIRE)) {
                    if ((++polls & 65535) == 0 && HybridNowUs() > deadline)
                        throw std::runtime_error("Native mailbox GPU publication timed out.");
#if defined(__x86_64__)
                __builtin_ia32_pause();
#else
                std::this_thread::yield();
#endif
                }
                if (!full) {
                    if (previous) Observe(*previous);
                    previous = &w;
                }
                const auto &layout = group.LayerLayout(w.table);
                int hits = 0;
                std::array<int, kMaxTopK> order{};
                for (int i = 0; i < w.topk; ++i) {
                    fastllm::AssertInFastLLM(box.ids[i] >= 0 && box.ids[i] < layout.experts,
                                           "Native mailbox invalid expert ID.");
                    if (box.resident[i] >= 0) {
                        if (!full) order[hits] = i;
                        ++hits;
                    }
                    if (!full) {
                        box.gpu[i] = box.resident[i] >= 0 ? box.ids[i] : -1;
                        box.staged[i] = -1;
                    }
                }
                int next = hits;
                for (int i = 0; !full && i < w.topk; ++i)
                    if (box.resident[i] < 0) order[next++] = i;
                cache.frequency->Observe(w.table * layout.experts, box.ids, w.topk);
                const int staged = full ? box.stagedCount : w.cost.SelectDecodeMisses(w.topk - hits, hits);
                for (int i = 0; !full && i < staged; ++i) {
                    const int route = order[hits + i];
                    box.staged[route] = i; box.gpu[route] = box.ids[route];
                }
                w.hits = hits; w.staged = staged;
                double cpuUs = 0;
                auto upload = [&] {
                    if (full) return;
                    const double start = HybridNowUs();
                    for (int i = 0; i < staged; ++i)
                        copy->CopyExpert(group, w.table, box.ids[order[hits + i]], i, staged);
                    w.cost.dispatch.Observe(HybridNowUs() - start);
                    // Record a CUDA-visible dependency before letting the
                    // caller enqueue its result graph. A graph memory wait
                    // must never be the only ordering edge to external DMA.
                    ReadNativeMoePlan<<<1, 32, 0, copy->copyStream>>>(w.mapped,
                        w.slots+w.topk, w.slots+2*w.topk, w.topk);
                    if (staged) {
                        checkCudaErrors("Native mailbox input", cudaStreamWaitEvent(copy->copyStream, w.begin, 0));
                        checkCudaErrors("Native staged begin", cudaEventRecord(w.stagedBegin, copy->copyStream));
                        Compute(w, layout, copy->records, w.slots+w.topk,
                            static_cast<const float *>(w.pointers[2]), copy->copyStream);
                    }
                    checkCudaErrors("Native staged done", cudaEventRecord(w.done, copy->copyStream));
                    w.submitted.store(true, std::memory_order_release);
                };
                fastllm::NumasMoeDecodeExpertsWithOverlap(w.hostData, w.hostData + w.hidden,
                    w.weights.data(), box.ids, box.gpu, w.topk, w.modelLayer, upload,
                    box.scores, layout.swigluLimit, 32, &cpuUs);
                w.cost.ObserveDecodeCpu(w.topk - hits - staged, cpuUs);
                ++w.cost.calls;
                {
                    const uint64_t d[8] = {1, uint64_t(w.topk), uint64_t(hits), uint64_t(w.topk-hits),
                        uint64_t(hits+staged), uint64_t(w.topk-hits-staged), uint64_t(hits), 0};
                    for (int i = 0; i < 8; ++i) {
                        cache.cooperativeStats[i] += d[i];
                        if (r.phase != FastllmMoeRoutePhase::Unspecified) cache.phaseStats[int(r.phase)][i] += d[i];
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!error) error = std::current_exception();
                // Release the graph before reporting the failure on the main
                // thread. Invalid output is never returned as a successful step.
                std::fill_n(w.hostData + w.hidden, w.topk * w.hidden, 0.0f);
                if (!full) {
                    std::fill_n(box.gpu, w.topk, -1); std::fill_n(box.staged, w.topk, -1);
                    ReadNativeMoePlan<<<1, 32, 0, copy->copyStream>>>(w.mapped,
                        w.slots+w.topk, w.slots+2*w.topk, w.topk);
                    cudaEventRecord(w.done, copy->copyStream);
                    w.submitted.store(true, std::memory_order_release);
                }
            }
            __atomic_store_n(&box.computed, 1u, __ATOMIC_RELEASE);
            { std::lock_guard<std::mutex> lock(mutex); busy = false; }
            drained.notify_all();
        }
    }
    static void Compute(Layer &w, const OffloadLayout &layout, const uint8_t *records,
            const int32_t *slots, const float *scores, cudaStream_t stream, bool resident = false) {
        if (resident && w.hidden % 256 == 0 && layout.inter % 32 == 0 && FastllmCudaRuntimeArch() == 120) {
            fastllm::cuda::dsv41_cache::GateResident<<<dim3(layout.inter / 32, w.topk), 512, 0, stream>>>(
                static_cast<const __nv_bfloat16 *>(w.quantized.cudaData), slots, records, scores,
                static_cast<__nv_bfloat16 *>(w.gate.cudaData), w.hidden, layout.inter,
                layout.recordStride, layout.swigluLimit);
        } else fastllm::cuda::dsv41_cache::Gate<<<dim3(layout.inter / 32, w.topk), 512, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(w.quantized.cudaData), slots, records,
            scores, static_cast<__nv_bfloat16 *>(w.gate.cudaData),
            w.hidden, layout.inter, layout.recordStride, layout.swigluLimit);
        fastllm::cuda::dsv41_cache::Down<<<dim3(w.hidden / 8, w.topk), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(w.gate.cudaData), slots, records, w.partial,
            w.hidden, layout.inter, layout.recordStride, layout.downOffset);
    }
    bool Capture(Layer &w, int table, const fastllm::Data &input,
            const fastllm::Data &index, const fastllm::Data &score, fastllm::Data &output) {
        for (auto *e : {&w.exec, &w.finishExec}) {
            if (*e) cudaGraphExecDestroy(*e);
            *e = nullptr;
        }
        const auto &layout = group.LayerLayout(table);
        auto capture = [&](cudaGraphExec_t &exec, const std::function<bool()> &body) {
            if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
            const bool ok = body();
            cudaGraph_t graph = nullptr;
            const auto end = cudaStreamEndCapture(cudaStreamPerThread, &graph);
            const bool ready = ok && end == cudaSuccess && graph &&
                cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess;
            if (graph) cudaGraphDestroy(graph);
            if (!ready) (void)cudaGetLastError();
            return ready;
        };
        if (!capture(w.exec, [&] {
            PublishNativeMoe<<<1, 256, 0, cudaStreamPerThread>>>(w.mapped, w.mappedData,
                static_cast<const __nv_bfloat16 *>(input.cudaData), static_cast<const int32_t *>(index.cudaData),
                static_cast<const float *>(score.cudaData), cache.keyToSlot, cache.slotKeys, w.slots,
                cache.hitCount, cache.totalMissCount, table * layout.experts, layout.experts, w.hidden, w.topk);
            if (!memops.Append(&w.mapped->published, false, 1)) return false;
            if (!FastllmCudaDeepSeekV41QuantizeActivation(input, w.quantized)) return false;
            if (cudaEventRecordWithFlags(w.begin, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            Compute(w, layout, cache.records, w.slots, static_cast<const float *>(score.cudaData), cudaStreamPerThread, true);
            return cudaGetLastError() == cudaSuccess &&
                cudaEventRecordWithFlags(w.residentDone, cudaStreamPerThread, cudaEventRecordExternal) == cudaSuccess;
        })) return false;
        if (!capture(w.finishExec, [&] {
            // External GPU work is joined by an event before this graph is
            // launched. Only CPU arithmetic can still be pending here.
            if (!memops.Append(&w.mapped->computed, true, 1)) return false;
            fastllm::cuda::dsv41_cache::Reduce<<<(w.hidden+255)/256, 256, 0, cudaStreamPerThread>>>(
                w.mappedData+w.hidden, w.partial, w.slots+2*w.topk, static_cast<const int32_t *>(index.cudaData),
                static_cast<__nv_bfloat16 *>(output.cudaData), w.hidden, w.topk);
            const bool reset = memops.Append(&w.mapped->published, false, 0) &&
                memops.Append(&w.mapped->computed, false, 0) && cudaGetLastError() == cudaSuccess;
            return reset && cudaEventRecordWithFlags(w.completed, cudaStreamPerThread, cudaEventRecordExternal) == cudaSuccess;
        })) return false;
        w.pointers = {input.cudaData, index.cudaData, score.cudaData, output.cudaData};
        return true;
    }
    Layer *CreateFull(int table, const fastllm::Data &input, const fastllm::Data &index,
            const fastllm::Data &score, fastllm::Data &output, fastllm::Data **weights,
            int weightsBatch, int layer, cudaGraph_t sharedGraph) {
        if (failed || !memops.ready || !sharedGraph) return nullptr;
        Finish(true);
        const auto &layout = group.LayerLayout(table);
        const int topk = index.dims.back();
        if (!copy) copy = MakeDecodeOverlap(group, group.tableKeys.size(), topk);
        if (!copy) return nullptr;
        auto fresh = std::make_unique<Layer>();
        if (!fresh->Init(layout, topk, cache.device)) return nullptr;
        auto &w = *fresh;
        w.owner = this; w.table = table; w.modelLayer = layer;
        w.weights.assign(weights, weights + weightsBatch);
        auto measured = layers.find(table);
        if (measured != layers.end()) w.cost = measured->second->cost;
        output.dataType = input.dataType; output.Resize({1, w.hidden});
        output.ToDevice(fastllm::CUDA, {cache.device}, false); output.Allocate(false);
        w.pointers = {input.cudaData, index.cudaData, score.cudaData, output.cudaData};
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, cache.device) != cudaSuccess) return nullptr;
        const auto launch = fastllm::cuda::RecordCopyConfiguration(layout.recordStride, topk, prop);
        if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return nullptr;
        bool ok = [&] {
            PublishNativeMoe<<<1, 256, 0, cudaStreamPerThread>>>(w.mapped, w.mappedData,
                static_cast<const __nv_bfloat16 *>(input.cudaData), static_cast<const int32_t *>(index.cudaData),
                static_cast<const float *>(score.cudaData), cache.keyToSlot, cache.slotKeys, w.slots,
                cache.hitCount, cache.totalMissCount, table * layout.experts, layout.experts, w.hidden, topk, true);
            // Shared and routed experts read the same immutable activation.
            // Keep publication first in node order, but fork shared work from
            // the graph root. The child completion joins both branches.
            cudaStreamCaptureStatus status;
            cudaGraph_t graph = nullptr;
            const cudaGraphNode_t *deps = nullptr;
            size_t count = 0;
            cudaGraphNode_t shared = nullptr;
            if (cudaStreamGetCaptureInfo(cudaStreamPerThread, &status, nullptr, &graph, &deps, &count) != cudaSuccess ||
                cudaGraphAddChildGraphNode(&shared, graph, nullptr, 0, sharedGraph) != cudaSuccess) return false;
            if (!FastllmCudaDeepSeekV41QuantizeActivation(input, w.quantized)) return false;
            if (cudaEventRecordWithFlags(w.begin, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            Compute(w, layout, cache.records, w.slots, static_cast<const float *>(score.cudaData), cudaStreamPerThread, true);
            if (cudaEventRecordWithFlags(w.residentDone, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            if (cudaEventRecordWithFlags(w.uploadBegin, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            // GPU reads the selected pinned records using device-side IDs.
            // No CUDA work is submitted by the CPU dispatcher during replay.
            if (group.sharedLayout.shards) {
                const auto &shared = group.sharedLayout;
                auto *pointers = cache.numaPointers + size_t(table) * layout.experts * 2 * shared.shards;
                const auto *auxiliary = shared.auxiliaryBytes ? group.deviceHostRecords +
                    size_t(table) * layout.experts * shared.auxiliaryBytes : nullptr;
                if (!fastllm::cuda::CopySharedExpertRecords(shared, pointers, auxiliary, copy->records,
                        w.slots + 3 * topk, w.slots + 4 * topk, w.slots + 5 * topk, topk,
                        launch, cudaStreamPerThread)) return false;
            } else {
                fastllm::cuda::RecordCopyView view{group.deviceHostRecords + group.layerHostOffsets[table],
                    copy->records, layout.recordStride, layout.recordStride, layout.recordStride};
                if (!fastllm::cuda::CopyRecords(view, w.slots + 3 * topk, w.slots + 4 * topk,
                        w.slots + 5 * topk, topk, launch, cudaStreamPerThread)) return false;
            }
            if (cudaEventRecordWithFlags(w.uploadDone, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            Compute(w, layout, copy->records, w.slots + topk, static_cast<const float *>(score.cudaData), cudaStreamPerThread);
            if (cudaEventRecordWithFlags(w.done, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            if (!memops.Append(&w.mapped->computed, true, 1)) return false;
            fastllm::cuda::dsv41_cache::Reduce<<<(w.hidden+255)/256, 256, 0, cudaStreamPerThread>>>(
                w.mappedData+w.hidden, w.partial, w.slots+2*topk, static_cast<const int32_t *>(index.cudaData),
                static_cast<__nv_bfloat16 *>(output.cudaData), w.hidden, topk,
                &w.mapped->published, &w.mapped->computed);
            if (cudaGetLastError() != cudaSuccess) return false;
            return cudaEventRecordWithFlags(w.completed, cudaStreamPerThread, cudaEventRecordExternal) == cudaSuccess;
        }();
        const auto end = cudaStreamEndCapture(cudaStreamPerThread, &w.fullGraph);
        if (!ok || end != cudaSuccess || !w.fullGraph) { cudaGetLastError(); return nullptr; }
        auto *handle = fresh.get();
        fullLayers[table] = std::move(fresh);
        return handle;
    }
    static bool StripTiming(cudaGraph_t graph, const std::vector<cudaEvent_t> &events) {
        // Batch-memop nodes have no CUDA Runtime node-type counterpart.
        // Query the Driver type while keeping the runtime graph handles.
        decltype(&cuGraphNodeGetType) nodeType = nullptr;
        if (cudaGetDriverEntryPoint("cuGraphNodeGetType", reinterpret_cast<void **>(&nodeType),
                cudaEnableDefault) != cudaSuccess || !nodeType) return false;
        size_t count = 0;
        if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) return false;
        std::vector<cudaGraphNode_t> nodes(count);
        if (cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) return false;
        for (auto node : nodes) {
            CUgraphNodeType type;
            if (nodeType(reinterpret_cast<CUgraphNode>(node), &type) != CUDA_SUCCESS) return false;
            if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
                cudaGraph_t child = nullptr;
                if (cudaGraphChildGraphNodeGetGraph(node, &child) != cudaSuccess ||
                    !StripTiming(child, events)) return false;
            } else if (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
                cudaEvent_t event = nullptr;
                if (cudaGraphEventRecordNodeGetEvent(node, &event) != cudaSuccess) return false;
                if (std::find(events.begin(), events.end(), event) == events.end()) continue;
                size_t beforeCount = 0, afterCount = 0;
                if (cudaGraphNodeGetDependencies(node, nullptr, &beforeCount) != cudaSuccess ||
                    cudaGraphNodeGetDependentNodes(node, nullptr, &afterCount) != cudaSuccess) return false;
                std::vector<cudaGraphNode_t> before(beforeCount), after(afterCount);
                if (cudaGraphNodeGetDependencies(node, before.data(), &beforeCount) != cudaSuccess ||
                    cudaGraphNodeGetDependentNodes(node, after.data(), &afterCount) != cudaSuccess) return false;
                for (auto next : after) {
                    size_t n = 0;
                    if (cudaGraphNodeGetDependencies(next, nullptr, &n) != cudaSuccess) return false;
                    std::vector<cudaGraphNode_t> existing(n);
                    if (cudaGraphNodeGetDependencies(next, existing.data(), &n) != cudaSuccess) return false;
                    for (auto prev : before)
                        if (std::find(existing.begin(), existing.end(), prev) == existing.end() &&
                            cudaGraphAddDependencies(graph, &prev, &next, 1) != cudaSuccess) return false;
                }
                if (cudaGraphDestroyNode(node) != cudaSuccess) return false;
            }
        }
        return true;
    }
    bool CreateReplay(cudaGraph_t source, void *const *handles, int count, void **result) {
        std::vector<cudaEvent_t> events;
        for (int i = 0; i < count; ++i) {
            auto *w = static_cast<Layer *>(handles[i]);
            if (!w || w->owner != this || !w->fullGraph) return false;
            for (auto e : {w->begin, w->residentDone, w->uploadBegin, w->uploadDone, w->done})
                events.push_back(e);
            if (i + 1 < count) events.push_back(w->completed);
        }
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        bool ok = cudaGraphClone(&graph, source) == cudaSuccess && StripTiming(graph, events) &&
            cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess;
        if (graph) cudaGraphDestroy(graph);
        if (!ok) { if (exec) cudaGraphExecDestroy(exec); cudaGetLastError(); return false; }
        *result = exec;
        return true;
    }
    bool LaunchFull(void *exec, void *const *handles, int count, bool timed) {
        Finish(true);
        if (!cache.frequencyActive) return false;
        for (int i = 0; i < count; ++i) {
            auto *w = static_cast<Layer *>(handles[i]);
            if (!w || w->owner != this || !w->fullGraph) return false;
        }
        // Prepare owned requests before launch; allocation failures must
        // not expose a partial queue to the dispatcher.
        std::deque<Request> requests;
        for (int i = 0; i < count; ++i)
            requests.push_back({static_cast<Layer *>(handles[i]), FastllmMoeRoutePhase::Decode});
        if (!dispatcher.joinable()) dispatcher = std::thread([this] { Run(); });
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (int i = 0; i < count; ++i) {
                auto &w = *static_cast<Layer *>(handles[i]);
                // Costs are immutable for this layer until the dispatcher
                // finishes its current request. Only residency is GPU-dynamic.
                for (int hits = 0; hits <= w.topk; ++hits)
                    w.host->stagedByHits[hits] = w.cost.SelectDecodeMisses(w.topk - hits, hits);
            }
            queue.swap(requests);
            if (!FastllmCudaGraphLaunch(exec)) {
                queue.clear();
                // The FastLLM error flag still reports the rejected launch;
                // consume the runtime error so cleanup/explicit retry cannot
                // mistake it for a failure of the next admission kernel.
                cudaGetLastError();
                return false;
            }
            for (int i = 0; i < count; ++i) {
                auto &w = *static_cast<Layer *>(handles[i]);
                w.pending = true; w.timed = timed;
            }
            fullLast = static_cast<Layer *>(handles[count - 1]);
        }
        wake.notify_one();
        return true;
    }
    bool Submit(int table, const fastllm::Data &input, const fastllm::Data &index,
            const fastllm::Data &score, fastllm::Data &output, fastllm::Data **weights,
            int weightsBatch, int layer, const std::function<void()> &parallel) {
        if (failed || !memops.ready) return false;
        const int topk = index.dims[1];
        if (!copy) {
            copy = MakeDecodeOverlap(group, group.tableKeys.size(), topk);
            if (!copy) { failed = true; return false; }
            // Load the DMA restore kernel before any graph can wait for it.
            // CUDA lazy module loading may synchronize the whole context.
            copy->CopyExpert(group, table, 0, 0, 1);
            checkCudaErrors("Native mailbox upload warmup", cudaEventSynchronize(copy->copyDone));
        }
        auto &owned = layers[table];
        if (!owned) {
            Finish(true);
            auto fresh = std::make_unique<Layer>();
            if (!fresh->Init(group.LayerLayout(table), topk, cache.device)) { failed = true; return false; }
            owned = std::move(fresh);
        }
        auto &w = *owned;
        if (w.pending) Finish(true);
        if (w.topk != topk) return false;
        w.table = table; w.modelLayer = layer;
        w.weights.assign(weights, weights + weightsBatch);
        output.dataType = input.dataType; output.Resize({1, w.hidden});
        output.ToDevice(fastllm::CUDA, {cache.device}, false); output.Allocate(false);
        const std::array<const void *, 4> pointers{input.cudaData,index.cudaData,score.cudaData,output.cudaData};
        if (!w.exec || pointers != w.pointers) {
            Finish(true);
            checkCudaErrors("Native mailbox recapture", cudaStreamSynchronize(cudaStreamPerThread));
            if (!Capture(w, table, input, index, score, output)) { failed = true; return false; }
        }
        if (!dispatcher.joinable()) {
            dispatcher = std::thread([this] { Run(); });
            std::fprintf(stderr, "[Fastllm] Native MoE graph mailbox active on cuda:%d.\n", cache.device);
        }
        if (parallel) { parallel(); checkCudaErrors("Native mailbox shared device", cudaSetDevice(cache.device)); }
        // Allocate the owned request before launch. Keep the queue locked
        // until launch succeeds so rejection cannot strand a CPU waiter.
        cudaError_t launched;
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push_back({&w, routePhase});
            w.submitted.store(false, std::memory_order_release);
            launched = cudaGraphLaunch(w.exec, cudaStreamPerThread);
            if (launched == cudaSuccess) w.pending = true;
            else queue.pop_back();
        }
        checkCudaErrors("Native mailbox replay", launched);
        wake.notify_one();
        // Wait only for routing and CUDA submission, never the NUMA result.
        // Resident and shared GPU work are already running during this handoff.
        while (!w.submitted.load(std::memory_order_acquire)) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#else
            std::this_thread::yield();
#endif
        }
        checkCudaErrors("Native staged join", cudaStreamWaitEvent(cudaStreamPerThread, w.done, 0));
        checkCudaErrors("Native mailbox reduction", cudaGraphLaunch(w.finishExec, cudaStreamPerThread));
        return true;
    }
};

bool TryNativeMoeMailbox(OffloadGroup &group, DeviceCache &cache, int table,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch, int layer,
        const std::function<void()> &parallel, const std::vector<int> *devices) {
    const auto &layout = group.LayerLayout(table);
    // Model capture can instantiate a shared-expert graph in parallel().
    // Keep its allocation/capture phase synchronous; asynchronous replay
    // starts only after that model graph and its workspace are established.
    if (!fastllm::GetFastllmEnv().cudaGraph || FastllmCudaGraphMemoryPoolActive() || !cache.frequencyActive ||
        (devices && devices->size() != 1) || !group.expertDevices.empty() || group.deviceCaches.size() != 1 ||
        std::any_of(group.cacheDevices.begin(), group.cacheDevices.end(),
            [&](int d) { return d >= 0 && d != cache.device; }) ||
        !layout.deepSeekV41 || layout.weightType != fastllm::NVFP4_BLOCK_32_E8M0 ||
        input.dataType != fastllm::BFLOAT16 || input.dims[0] != 1 ||
        (routePhase != FastllmMoeRoutePhase::Decode && routePhase != FastllmMoeRoutePhase::Unspecified)) return false;
    const char *mode = std::getenv("FASTLLM_DSV41_MOE_CACHE_MODE");
    if (mode && std::strcmp(mode, "gpu") == 0) return false;
    if (!cache.nativeMailbox) cache.nativeMailbox = std::make_shared<NativeMoeMailbox>(group, cache);
    return cache.nativeMailbox->Submit(table, input, index, score, output, weights, weightsBatch, layer, parallel);
}
void FinishNativeMoeMailbox(DeviceCache &cache, bool observe) {
    if (cache.nativeMailbox) cache.nativeMailbox->Finish(observe);
}
#else
void FinishNativeMoeMailbox(DeviceCache &, bool) {}
#ifdef USE_NUMAS
bool TryNativeMoeMailbox(OffloadGroup &, DeviceCache &, int,
        const fastllm::Data &, const fastllm::Data &, const fastllm::Data &, fastllm::Data &,
        fastllm::Data **, int, int, const std::function<void()> &, const std::vector<int> *) { return false; }
#endif
#endif
} // namespace
