#pragma once
#include "../fastllm-cuda-graph-compat.cuh"

// Native, single-GPU decode handoff. A CPU dispatcher owns NUMA execution;
// the caller only submits graphs. Full replay uses an explicit one-thread
// acquire wait; the split-layer path retains stream memory waits. Every layer
// owns its activation/result/control buffers.
#include <deque>

namespace {
#if defined(USE_NUMAS) && !defined(USE_ROCM) && CUDART_VERSION >= 11070
constexpr int kNativeMoeMaxRows = 32;
constexpr int kNativeMoeMaxRoutes = kNativeMoeMaxRows * kMaxTopK;
struct NativeMoeControl {
    alignas(64) unsigned published = 0;
    alignas(64) unsigned computed = 0;
    alignas(64) unsigned planned = 0;
    int stagedCount = 0;
    alignas(64) int stagedByHits[kMaxTopK + 1];
    alignas(64) int32_t ids[kNativeMoeMaxRoutes], resident[kNativeMoeMaxRoutes], staged[kNativeMoeMaxRoutes], gpu[kNativeMoeMaxRoutes];
    float scores[kNativeMoeMaxRoutes];
    int32_t stagedExperts[kMaxTopK], stagedSlots[kMaxTopK];
};

__global__ void PublishNativeMoe(NativeMoeControl *box, float *hostInput,
        const __nv_bfloat16 *input, const int32_t *indices, const float *scores,
        const int32_t *keys, const int32_t *slotKeys, int32_t *slots,
        unsigned long long *hits, unsigned long long *misses,
        int base, int experts, int hidden, int topk, bool publish = false) {
    // Publish adjacent values together to amortize mapped-host store traffic.
    // Keep the per-thread system fences and block-wide publication ordering.
    if ((reinterpret_cast<uintptr_t>(input) & 7u) == 0 &&
        (reinterpret_cast<uintptr_t>(hostInput) & 15u) == 0) {
        for (int c = threadIdx.x * 4; c < hidden; c += blockDim.x * 4) {
            if (c + 3 < hidden) {
                const uint2 packed = *reinterpret_cast<const uint2 *>(input + c);
                const auto *p = reinterpret_cast<const __nv_bfloat162 *>(&packed);
                const float2 a = __bfloat1622float2(p[0]), b = __bfloat1622float2(p[1]);
                *reinterpret_cast<float4 *>(hostInput + c) = make_float4(a.x, a.y, b.x, b.y);
            } else {
                for (int j = c; j < hidden; ++j) hostInput[j] = __bfloat162float(input[j]);
            }
        }
    } else {
        for (int c = threadIdx.x; c < hidden; c += blockDim.x)
            hostInput[c] = __bfloat162float(input[c]);
    }
    if (threadIdx.x < 32) {
        const int r = threadIdx.x, e = r < topk ? indices[r] : -1;
        const int key = base + e;
        const int slot = e >= 0 && e < experts ?
            fastllm::cuda::moe_admission::LoadAdmission(keys + key) : -1;
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

// Route order is local to an eight-row compute tile, matching both the
// grouped CUDA implementation and its exact native fallback.
__device__ int NativeBatchRouteRank(const int32_t *indices, int r, int rows, int topk) {
    const int first = r / (8 * topk) * (8 * topk);
    const int end = min(rows * topk, first + 8 * topk);
    int rank = first;
    for (int j = first; j < end; ++j)
        rank += indices[j] < indices[r] || (indices[j] == indices[r] && j < r);
    return rank;
}
__global__ void PublishNativeBatchMoe(NativeMoeControl *box, uint16_t *hostInput,
        const uint16_t *input, const int32_t *indices, const float *scores,
        const int32_t *keys, const int32_t *slotKeys, int32_t *slots, int32_t *routeMap,
        unsigned long long *hits, unsigned long long *misses,
        int base, int experts, int hidden, int rows, int topk) {
    for (int c = threadIdx.x; c < rows * hidden; c += blockDim.x) hostInput[c] = input[c];
    for (int r = threadIdx.x; r < rows * topk; r += blockDim.x) {
        const int expert = indices[r], key = base + expert;
        int slot = expert >= 0 && expert < experts ?
            fastllm::cuda::moe_admission::LoadAdmission(keys + key) : -1;
        if (slot >= 0 && slotKeys[slot] != key) slot = -1;
        const int at = NativeBatchRouteRank(indices, r, rows, topk);
        slots[at] = slot;
        routeMap[at] = r % (8 * topk);
        box->ids[r] = expert; box->resident[r] = slot; box->scores[r] = scores[r];
        atomicAdd(slot >= 0 ? hits : misses, 1ULL);
    }
    __threadfence_system(); __syncthreads();
    if (!threadIdx.x) atomicExch_system(&box->published, 1u);
}
__global__ void ReadNativeBatchPlan(const NativeMoeControl *box, const int32_t *indices,
        int32_t *staged, int32_t *gpu, int rows, int topk) {
    for (int r = threadIdx.x; r < rows * topk; r += blockDim.x) {
        staged[NativeBatchRouteRank(indices, r, rows, topk)] = box->staged[r];
        gpu[r] = box->gpu[r];
    }
}

__global__ void ReadNativeMoePlan(const NativeMoeControl *box, int32_t *staged, int32_t *gpu, int topk) {
    if (threadIdx.x < topk) {
        staged[threadIdx.x] = box->staged[threadIdx.x];
        gpu[threadIdx.x] = box->gpu[threadIdx.x];
    }
}

// Full replay waits only for CPU arithmetic: its producer submits no CUDA.
// One thread avoids issuing the same host-memory acquire from every Reduce CTA.
// Keep this as an explicit graph node so profiling separates waiting from math.
__global__ void WaitNativeMoe(const unsigned *ready) {
    unsigned value;
    do {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(ready) : "memory");
        if (!value) __nanosleep(64);
#else
        value = *reinterpret_cast<const volatile unsigned *>(ready);
        __threadfence_system();
#endif
    } while (!value);
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
            fastllm_cuda_graph_compat::CaptureInfo(cudaStreamPerThread, &status, nullptr, &graph, &deps, &count) != cudaSuccess ||
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
        return fastllm_cuda_graph_compat::UpdateCaptureDependencies(cudaStreamPerThread, &next, 1,
                    cudaStreamSetCaptureDependencies) == cudaSuccess;
    }
};

struct NativeMoeMailbox {
    struct Layer {
        NativeMoeControl *host = nullptr, *mapped = nullptr;
        float *hostData = nullptr, *mappedData = nullptr;
        float *partial = nullptr;
        int32_t *slots = nullptr; // resident, staged, GPU route IDs
        int32_t *routeMap = nullptr;
        fastllm::Data quantized, gate;
        fastllm::Data ggufWorkspace[2], ggufActivation[2];
        cudaGraphExec_t exec = nullptr, finishExec = nullptr;
        cudaGraph_t fullGraph = nullptr;
        NativeMoeMailbox *owner = nullptr;
        int table = -1, modelLayer = -1;
        std::vector<fastllm::Data *> weights;
        cudaEvent_t uploadBegin = nullptr, uploadDone = nullptr;
        cudaEvent_t begin = nullptr, residentDone = nullptr, stagedBegin = nullptr, done = nullptr, completed = nullptr;
        std::array<const void *, 4> pointers{};
        fastllm::MoeDecodeOverlapScheduler cost;
        int hidden = 0, topk = 0, rows = 1, routes = 0, hits = 0, staged = 0, stagedRoutes = 0;
        bool timed = true;
        std::atomic<bool> pending{false}, submitted{false};
        ~Layer() {
            if (pending && completed) cudaEventSynchronize(completed);
            if (fullGraph) cudaGraphDestroy(fullGraph);
            if (exec) cudaGraphExecDestroy(exec);
            if (finishExec) cudaGraphExecDestroy(finishExec);
            for (auto e : {begin, residentDone, stagedBegin, done, completed, uploadBegin, uploadDone}) if (e) cudaEventDestroy(e);
            cudaFreeHost(host); cudaFreeHost(hostData); cudaFree(partial); cudaFree(slots); cudaFree(routeMap);
        }
        bool Init(const OffloadLayout &layout, int selected, int device, int inputRows = 1) {
            hidden = layout.hidden; topk = selected; rows = inputRows; routes = rows * topk;
            if (cudaHostAlloc(&host, sizeof(*host), cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(&mapped, host, 0) != cudaSuccess ||
                cudaHostAlloc(&hostData, (routes + rows) * hidden * sizeof(float), cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(&mappedData, hostData, 0) != cudaSuccess ||
                cudaMalloc(&partial, routes * hidden * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&slots, (rows > 1 ? 3 * routes : 5 * routes + 1) * sizeof(int32_t)) != cudaSuccess ||
                (rows > 1 && cudaMalloc(&routeMap, routes * sizeof(int32_t)) != cudaSuccess)) return false;
            new (host) NativeMoeControl{};
            std::fill_n(hostData, (routes + rows) * hidden, 0.0f);
            for (auto *e : {&begin, &residentDone, &stagedBegin, &done, &completed, &uploadBegin, &uploadDone})
                if (cudaEventCreate(e) != cudaSuccess) return false;
            quantized.dataType = fastllm::BFLOAT16;
            quantized.Resize({rows, hidden});
            quantized.ToDevice(fastllm::CUDA, {device}, false); quantized.Allocate(false);
            // GGUF owns separate resident/staged projection scratch below.
            // Only native experts consume this Gate buffer.
            if (layout.weightType != fastllm::DATA_GGUF_FORMAT) {
                gate.dataType = fastllm::BFLOAT16;
                gate.Resize({routes, layout.inter});
                gate.ToDevice(fastllm::CUDA, {device}, false); gate.Allocate(false);
            }
            if (layout.weightType == fastllm::DATA_GGUF_FORMAT) {
                const size_t bytes = FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(hidden, layout.inter);
                if (!bytes) return false;
                // Resident and staged branches execute concurrently. Neither
                // quantization scratch nor activation storage may alias.
                for (int branch = 0; branch < 2; ++branch) {
                    auto &scratch = ggufWorkspace[branch];
                    scratch.dataType = fastllm::INT8; scratch.Resize({int(bytes)});
                    scratch.ToDevice(fastllm::CUDA, {device}, false); scratch.Allocate(false);
                    auto &activation = ggufActivation[branch];
                    activation.dataType = fastllm::BFLOAT16;
                    activation.Resize({std::min(rows, 8) * topk, layout.inter});
                    activation.ToDevice(fastllm::CUDA, {device}, false); activation.Allocate(false);
                }
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
    std::unique_ptr<DecodeOverlapWorkspace> copy, fullCopy;
    std::map<int, std::unique_ptr<Layer>> layers;
    // Graphs for different verifier lengths coexist; never replace buffers
    // still referenced by an older captured graph.
    std::vector<std::unique_ptr<Layer>> fullLayers;
    std::thread dispatcher;
    std::vector<int> dispatcherCpus;
    std::mutex mutex;
    std::condition_variable wake, drained;
    std::deque<Request> queue;
    bool busy = false, stopping = false;
    std::atomic<bool> launchCanceled{false};
    std::exception_ptr error;
    Layer *previous = nullptr, *fullLast = nullptr;
    bool failed = false;

    NativeMoeMailbox(OffloadGroup &g, DeviceCache &c) : group(g), cache(c) {
        fastllm::HostMoePlacementScope placement(g.tableKeys.empty() ? nullptr : g.tableKeys.front());
        const auto cpus = fastllm::GetNumasCudaWorkerCpuSets({c.device});
        if (cpus.size() == 1) dispatcherCpus = cpus.front();
    }
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
            for (auto &entry : fullLayers) Observe(*entry, false);
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
        layers.clear(); fullLayers.clear(); fullCopy.reset(); copy.reset();
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
            w.cost.stagedExpert.Observe(ms * 1000 / (w.rows > 1 ? w.stagedRoutes : w.staged));
        }
        w.pending = false;
    }
    void Run() {
        // Excludes expert worker cores and respects the caller's allowed mask.
        fastllm::BindNumasWorkerCpuSet(dispatcherCpus);
        cudaSetDevice(cache.device);
        for (;;) {
            Request r;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) return;
                r = std::move(queue.front()); queue.pop_front(); busy = true;
            }
            // Cancellation and normal/error completion release the dispatcher
            // through one path, so a rejected launch cannot strand Finish().
            struct CompleteRequest {
                NativeMoeMailbox &mailbox;
                ~CompleteRequest() {
                    { std::lock_guard<std::mutex> lock(mailbox.mutex); mailbox.busy = false; }
                    mailbox.drained.notify_all();
                }
            } completion{*this};
            auto &w = *r.work;
            auto &box = *w.host;
            const bool full = w.fullGraph != nullptr;
            // Publication is downstream of the preceding layer's reduction,
            // so its DMA scratch and timing events can now be reused.
            try {
                unsigned polls = 0;
                const double deadline = HybridNowUs() + 30000000;
                bool canceled = false;
                while (!__atomic_load_n(&box.published, __ATOMIC_ACQUIRE)) {
                    if (full && launchCanceled.load(std::memory_order_acquire)) {
                        canceled = true;
                        break;
                    }
                    if ((++polls & 65535) == 0 && HybridNowUs() > deadline)
                        throw std::runtime_error("Native mailbox GPU publication timed out.");
#if defined(__x86_64__)
                __builtin_ia32_pause();
#else
                std::this_thread::yield();
#endif
                }
                if (canceled) continue;
                if (!full) {
                    if (previous) Observe(*previous);
                    previous = &w;
                }
                const auto &layout = group.LayerLayout(w.table);
                if (w.rows > 1) {
                    RunBatch(w, r.phase, layout);
                    __atomic_store_n(&box.computed, 1u, __ATOMIC_RELEASE);
                    continue;
                }
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
                std::fill_n(w.hostData + w.rows * w.hidden, w.routes * w.hidden, 0.0f);
                if (w.rows > 1 && !__atomic_load_n(&box.planned, __ATOMIC_ACQUIRE)) {
                    std::fill_n(box.staged, w.routes, -1); std::fill_n(box.gpu, w.routes, -1);
                    box.stagedCount = 0;
                    __atomic_store_n(&box.planned, 1u, __ATOMIC_RELEASE);
                }
                if (!full) {
                    std::fill_n(box.gpu, w.topk, -1); std::fill_n(box.staged, w.topk, -1);
                    ReadNativeMoePlan<<<1, 32, 0, copy->copyStream>>>(w.mapped,
                        w.slots+w.topk, w.slots+2*w.topk, w.topk);
                    cudaEventRecord(w.done, copy->copyStream);
                    w.submitted.store(true, std::memory_order_release);
                }
            }
            __atomic_store_n(&box.computed, 1u, __ATOMIC_RELEASE);
        }
    }
    void RunBatch(Layer &w, FastllmMoeRoutePhase phase, const OffloadLayout &layout) {
        auto &box = *w.host;
        std::vector<int> missed, reuse;
        w.hits = 0;
        for (int r = 0; r < w.routes; ++r) {
            fastllm::AssertInFastLLM(box.ids[r] >= 0 && box.ids[r] < layout.experts,
                                   "Native verifier mailbox invalid expert ID.");
            box.staged[r] = -1;
            box.gpu[r] = box.resident[r] >= 0 ? box.ids[r] : -1;
            if (box.resident[r] >= 0) { ++w.hits; continue; }
            auto found = std::find(missed.begin(), missed.end(), box.ids[r]);
            if (found == missed.end()) { missed.push_back(box.ids[r]); reuse.push_back(1); }
            else ++reuse[found - missed.begin()];
        }
        for (size_t i = 0; i < missed.size(); ++i) for (size_t j = i + 1; j < missed.size(); ++j)
            if (reuse[j] > reuse[i]) { std::swap(reuse[i], reuse[j]); std::swap(missed[i], missed[j]); }
        cache.frequency->Observe(w.table * layout.experts, box.ids, w.routes);
        w.staged = std::min(kMaxTopK, w.cost.SelectMisses(missed.size(), w.hits, reuse.data()));
        w.stagedRoutes = 0;
        for (int i = 0; i < w.staged; ++i) {
            box.stagedExperts[i] = missed[i]; box.stagedSlots[i] = i;
            for (int r = 0; r < w.routes; ++r) if (box.ids[r] == missed[i]) {
                box.staged[r] = i; box.gpu[r] = box.ids[r]; ++w.stagedRoutes;
            }
        }
        box.stagedCount = w.staged;
        // The CPU producer publishes a complete immutable plan and submits no
        // CUDA work. The graph can already execute shared and resident experts.
        __atomic_store_n(&box.planned, 1u, __ATOMIC_RELEASE);
        const int cpuRoutes = w.routes - w.hits - w.stagedRoutes;
        if (cpuRoutes > 0) {
            const double begin = HybridNowUs();
            // Cache preparation already validated every immutable weight.
            // Reuse that result, as the segmented verifier does, instead of
            // scanning every expert again on each layer's critical path.
            fastllm::NumasMoeVerifyExpertsWithOverlap(
                reinterpret_cast<const uint16_t *>(w.hostData),
                w.hostData + w.rows * w.hidden, w.rows, w.weights.data(), w.weights.size(),
                box.ids, box.gpu, box.scores, w.topk, w.modelLayer, layout.swigluLimit,
                true, 32, {}, nullptr, group.cpuDecodeReady);
            w.cost.cpuExpert.Observe((HybridNowUs() - begin) / cpuRoutes);
        }
        ++w.cost.calls;
        const uint64_t d[8] = {1, uint64_t(w.routes), uint64_t(w.hits), uint64_t(w.routes-w.hits),
            uint64_t(w.hits+w.stagedRoutes), uint64_t(cpuRoutes), uint64_t(w.hits), 0};
        for (int i = 0; i < 8; ++i) {
            cache.cooperativeStats[i] += d[i];
            if (phase != FastllmMoeRoutePhase::Unspecified) cache.phaseStats[int(phase)][i] += d[i];
        }
    }
    static bool ComputeBatch(Layer &w, const OffloadLayout &layout, const uint8_t *records,
            const int32_t *slots, const float *scores) {
        for (int first = 0; first < w.rows; first += 8) {
            const int rows = std::min(8, w.rows - first), base = first * w.topk;
            fastllm::Data input, gate, dummy;
            input.FakeFrom(w.quantized, size_t(first) * w.hidden * 2); input.Resize({rows, w.hidden});
            gate.FakeFrom(w.gate, size_t(base) * layout.inter * 2); gate.Resize({rows*w.topk, layout.inter});
            FastllmCudaMoeGGUFCacheView view{};
            view.records = records; view.routeSlots = const_cast<int32_t *>(slots + base);
            view.routeMap = w.routeMap ? w.routeMap + base : nullptr; view.routeCount = rows * w.topk;
            if (!ComputeBatchExperts(input, gate, dummy, layout, view, scores + base,
                    w.topk, w.partial + size_t(base) * w.hidden, true)) return false;
        }
        return true;
    }
    bool ComputeGGUF(Layer &w, const OffloadLayout &layout, const uint8_t *records,
                     const int32_t *slots, const float *scores, bool resident) {
        const int branch = resident ? 0 : 1;
        for (int first = 0; first < w.rows; first += 8) {
            const int rows = std::min(8, w.rows - first), base = first * w.topk;
            fastllm::Data input;
            input.FakeFrom(w.quantized, size_t(first) * w.hidden * sizeof(uint16_t));
            input.Resize({rows, w.hidden});
            FastllmCudaMoeGGUFCacheView view{records, const_cast<int32_t *>(slots + base),
                layout.recordStride, layout.downOffset, layout.gateGgmlType, layout.downGgmlType,
                layout.hidden, layout.inter, w.ggufWorkspace[branch].cudaData,
                size_t(w.ggufWorkspace[branch].GetBytes()), resident ? cache.slotOffsets : nullptr};
            view.routeMap = w.routeMap ? w.routeMap + base : nullptr;
            view.routeCount = rows * w.topk;
            // Missing routes belong to the other branch (or CPU). Never
            // overwrite their partials while publishing this branch's rows.
            view.preserveMissing = true;
            if (!FastllmCudaMoeV41CanonicalGGUFCacheCompute(input, w.ggufActivation[branch],
                    view, scores + base, w.topk, layout.swigluLimit,
                    w.partial + size_t(base) * w.hidden)) return false;
        }
        return true;
    }
    static void Compute(Layer &w, const OffloadLayout &layout, const uint8_t *records,
            const int32_t *slots, const float *scores, cudaStream_t stream, bool resident = false) {
        if (resident && w.hidden % 256 == 0 && layout.inter % 32 == 0 && FastllmCudaRuntimeArch() == 120) {
            if (layout.planarMxfp4)
                fastllm::cuda::dsv41_cache::GateResident<true><<<dim3(layout.inter / 32, w.topk), 512, 0, stream>>>(
                    static_cast<const __nv_bfloat16 *>(w.quantized.cudaData), slots, records, scores,
                    static_cast<__nv_bfloat16 *>(w.gate.cudaData), w.hidden, layout.inter,
                    layout.recordStride, layout.swigluLimit);
            else
                fastllm::cuda::dsv41_cache::GateResident<false><<<dim3(layout.inter / 32, w.topk), 512, 0, stream>>>(
                    static_cast<const __nv_bfloat16 *>(w.quantized.cudaData), slots, records, scores,
                    static_cast<__nv_bfloat16 *>(w.gate.cudaData), w.hidden, layout.inter,
                    layout.recordStride, layout.swigluLimit);
        } else fastllm::cuda::dsv41_cache::Gate<<<dim3(layout.inter / 32, w.topk), 512, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(w.quantized.cudaData), slots, records,
            scores, static_cast<__nv_bfloat16 *>(w.gate.cudaData),
            w.hidden, layout.inter, layout.recordStride, layout.swigluLimit, 1, nullptr, layout.planarMxfp4);
        fastllm::cuda::dsv41_cache::Down<<<dim3(w.hidden / 8, w.topk), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(w.gate.cudaData), slots, records, w.partial,
            w.hidden, layout.inter, layout.recordStride, layout.downOffset, nullptr, layout.planarMxfp4);
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
            PublishNativeMoe<<<1, w.hidden >= 4096 ? 512 : 256, 0, cudaStreamPerThread>>>(w.mapped, w.mappedData,
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
        if (failed) return nullptr;
        Finish(true);
        const auto &layout = group.LayerLayout(table);
        const int topk = index.dims.back();
        if (!fullCopy) fullCopy = MakeDecodeOverlap(group, group.tableKeys.size(), kMaxTopK);
        if (!fullCopy) return nullptr;
        auto fresh = std::make_unique<Layer>();
        if (!fresh->Init(layout, topk, cache.device, input.dims[0])) return nullptr;
        auto &w = *fresh;
        w.owner = this; w.table = table; w.modelLayer = layer;
        w.weights.assign(weights, weights + weightsBatch);
        auto measured = layers.find(table);
        if (measured != layers.end()) w.cost = measured->second->cost;
        output.dataType = input.dataType; output.Resize({w.rows, w.hidden});
        output.ToDevice(fastllm::CUDA, {cache.device}, false); output.Allocate(false);
        w.pointers = {input.cudaData, index.cudaData, score.cudaData, output.cudaData};
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, cache.device) != cudaSuccess) return nullptr;
        const auto launch = fastllm::cuda::RecordCopyConfiguration(layout.recordStride, kMaxTopK, prop);
        if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return nullptr;
        bool ok = [&] {
            if (w.rows > 1) PublishNativeBatchMoe<<<1, 256, 0, cudaStreamPerThread>>>(w.mapped,
                reinterpret_cast<uint16_t *>(w.mappedData), static_cast<const uint16_t *>(input.cudaData),
                static_cast<const int32_t *>(index.cudaData), static_cast<const float *>(score.cudaData),
                cache.keyToSlot, cache.slotKeys, w.slots, w.routeMap, cache.hitCount, cache.totalMissCount,
                table * layout.experts, layout.experts, w.hidden, w.rows, topk);
            else PublishNativeMoe<<<1, w.hidden >= 4096 ? 512 : 256, 0, cudaStreamPerThread>>>(w.mapped, w.mappedData,
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
            if (sharedGraph &&
                (fastllm_cuda_graph_compat::CaptureInfo(cudaStreamPerThread, &status, nullptr, &graph, &deps, &count) != cudaSuccess ||
                 cudaGraphAddChildGraphNode(&shared, graph, nullptr, 0, sharedGraph) != cudaSuccess)) return false;
            if (!FastllmCudaDeepSeekV41QuantizeActivation(input, w.quantized)) return false;
            if (cudaEventRecordWithFlags(w.begin, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            if (fastllm_cuda_graph_compat::CaptureInfo(cudaStreamPerThread, &status, nullptr, &graph, &deps, &count) != cudaSuccess) return false;
            std::vector<cudaGraphNode_t> fork(deps, deps + count);
            if (layout.weightType == fastllm::DATA_GGUF_FORMAT) {
                if (!ComputeGGUF(w, layout, cache.records, w.slots,
                        static_cast<const float *>(score.cudaData), true)) return false;
            } else if (w.rows > 1 || layout.weightType == fastllm::FP8_E4M3_BLOCK_128) {
                if (!ComputeBatch(w, layout, cache.records, w.slots, static_cast<const float *>(score.cudaData))) return false;
            } else Compute(w, layout, cache.records, w.slots, static_cast<const float *>(score.cudaData), cudaStreamPerThread, true);
            if (cudaEventRecordWithFlags(w.residentDone, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            if (fastllm_cuda_graph_compat::CaptureInfo(cudaStreamPerThread, &status, nullptr, &graph, &deps, &count) != cudaSuccess) return false;
            std::vector<cudaGraphNode_t> residentTail(deps, deps + count);
            // Resident and staged routes write disjoint gate/partial rows.
            // Both depend on publication and quantization, and join at Reduce.
            if (fastllm_cuda_graph_compat::UpdateCaptureDependencies(cudaStreamPerThread, fork.data(), fork.size(),
                    cudaStreamSetCaptureDependencies) != cudaSuccess) return false;
            if (w.rows > 1) {
                WaitNativeMoe<<<1, 1, 0, cudaStreamPerThread>>>(&w.mapped->planned);
                ReadNativeBatchPlan<<<1, 256, 0, cudaStreamPerThread>>>(w.mapped,
                    static_cast<const int32_t *>(index.cudaData), w.slots + w.routes,
                    w.slots + 2 * w.routes, w.rows, topk);
            }
            if (cudaEventRecordWithFlags(w.uploadBegin, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            // GPU reads the selected pinned records using device-side IDs.
            // No CUDA work is submitted by the CPU dispatcher during replay.
            const int32_t *stagedExperts = w.rows > 1 ? w.mapped->stagedExperts : w.slots + 3 * topk;
            const int32_t *stagedSlots = w.rows > 1 ? w.mapped->stagedSlots : w.slots + 4 * topk;
            const int32_t *stagedCount = w.rows > 1 ? &w.mapped->stagedCount : w.slots + 5 * topk;
            const int maxStaged = w.rows > 1 ? kMaxTopK : topk;
            if (!group.ggufSources.empty()) {
                const auto &source = group.ggufSources[table];
                auto *pointers = cache.numaPointers + size_t(table) * layout.experts * 2 * source.shards;
                // Device IDs select immutable mapped NUMA shards. Restore the
                // canonical record losslessly, including per-layer IQ/R4 types.
                if (fastllm_gguf_restore::CopyRecords(source, pointers, fullCopy->records,
                        cudaStreamPerThread, stagedExperts, stagedSlots, stagedCount) != cudaSuccess) return false;
            } else if (group.sharedLayout.shards) {
                const auto &shared = group.sharedLayout;
                auto *pointers = cache.numaPointers + size_t(table) * layout.experts * 2 * shared.shards;
                const auto *auxiliary = shared.auxiliaryBytes ? group.deviceHostRecords +
                    size_t(table) * layout.experts * shared.auxiliaryBytes : nullptr;
                if (!fastllm::cuda::CopySharedExpertRecords(shared, pointers, auxiliary, fullCopy->records,
                        stagedExperts, stagedSlots, stagedCount, maxStaged, launch, cudaStreamPerThread)) return false;
            } else {
                fastllm::cuda::RecordCopyView view{group.deviceHostRecords + group.layerHostOffsets[table],
                    fullCopy->records, layout.recordStride, layout.recordStride, layout.recordStride};
                if (!fastllm::cuda::CopyRecords(view, stagedExperts, stagedSlots, stagedCount,
                        maxStaged, launch, cudaStreamPerThread)) return false;
            }
            if (cudaEventRecordWithFlags(w.uploadDone, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            if (layout.weightType == fastllm::DATA_GGUF_FORMAT) {
                if (!ComputeGGUF(w, layout, fullCopy->records, w.slots + w.routes,
                        static_cast<const float *>(score.cudaData), false)) return false;
            } else if (w.rows > 1 || layout.weightType == fastllm::FP8_E4M3_BLOCK_128) {
                if (!ComputeBatch(w, layout, fullCopy->records, w.slots + w.routes, static_cast<const float *>(score.cudaData))) return false;
            } else Compute(w, layout, fullCopy->records, w.slots + topk, static_cast<const float *>(score.cudaData), cudaStreamPerThread);
            if (cudaEventRecordWithFlags(w.done, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            if (fastllm_cuda_graph_compat::UpdateCaptureDependencies(cudaStreamPerThread, residentTail.data(), residentTail.size(),
                    cudaStreamAddCaptureDependencies) != cudaSuccess) return false;
            WaitNativeMoe<<<1, 1, 0, cudaStreamPerThread>>>(&w.mapped->computed);
            fastllm::cuda::dsv41_cache::Reduce<<<dim3((w.hidden+255)/256, w.rows), 256, 0, cudaStreamPerThread>>>(
                w.mappedData+w.rows*w.hidden, w.partial, w.slots+2*w.routes, static_cast<const int32_t *>(index.cudaData),
                static_cast<__nv_bfloat16 *>(output.cudaData), w.hidden, topk);
            if (cudaGetLastError() != cudaSuccess) return false;
            return cudaEventRecordWithFlags(w.completed, cudaStreamPerThread, cudaEventRecordExternal) == cudaSuccess;
        }();
        const auto end = cudaStreamEndCapture(cudaStreamPerThread, &w.fullGraph);
        if (!ok || end != cudaSuccess || !w.fullGraph) { cudaGetLastError(); return nullptr; }
        auto *handle = fresh.get();
        fullLayers.push_back(std::move(fresh));
        return handle;
    }
    void ReleaseFull(Layer *handle) {
        Finish(true);
        auto it = std::find_if(fullLayers.begin(), fullLayers.end(),
            [handle](const std::unique_ptr<Layer> &layer) { return layer.get() == handle; });
        if (it != fullLayers.end()) fullLayers.erase(it);
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
                if (fastllm_cuda_graph_compat::GetDependencies(node, nullptr, &beforeCount) != cudaSuccess ||
                    fastllm_cuda_graph_compat::GetDependentNodes(node, nullptr, &afterCount) != cudaSuccess) return false;
                std::vector<cudaGraphNode_t> before(beforeCount), after(afterCount);
                if (fastllm_cuda_graph_compat::GetDependencies(node, before.data(), &beforeCount) != cudaSuccess ||
                    fastllm_cuda_graph_compat::GetDependentNodes(node, after.data(), &afterCount) != cudaSuccess) return false;
                for (auto next : after) {
                    size_t n = 0;
                    if (fastllm_cuda_graph_compat::GetDependencies(next, nullptr, &n) != cudaSuccess) return false;
                    std::vector<cudaGraphNode_t> existing(n);
                    if (fastllm_cuda_graph_compat::GetDependencies(next, existing.data(), &n) != cudaSuccess) return false;
                    for (auto prev : before)
                        if (std::find(existing.begin(), existing.end(), prev) == existing.end() &&
                            fastllm_cuda_graph_compat::AddDependencies(graph, &prev, &next, 1) != cudaSuccess) return false;
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
        std::deque<Request> requests;
        for (int i = 0; i < count; ++i)
            requests.push_back({static_cast<Layer *>(handles[i]),
                routePhase != FastllmMoeRoutePhase::Unspecified ? routePhase :
                (static_cast<Layer *>(handles[i])->rows > 1 ? FastllmMoeRoutePhase::Verify : FastllmMoeRoutePhase::Decode)});
        if (!dispatcher.joinable()) dispatcher = std::thread([this] { Run(); });
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (int i = 0; i < count; ++i) {
                auto &w = *static_cast<Layer *>(handles[i]);
                // Finish(true) retired every reduction. Reset before publishing
                // the new CPU queue, so no dispatcher can consume an old flag.
                __atomic_store_n(&w.host->published, 0u, __ATOMIC_RELEASE);
                __atomic_store_n(&w.host->computed, 0u, __ATOMIC_RELEASE);
                if (w.rows > 1) {
                    __atomic_store_n(&w.host->planned, 0u, __ATOMIC_RELEASE);
                } else {
                    for (int hits = 0; hits <= w.topk; ++hits)
                        w.host->stagedByHits[hits] = w.cost.SelectDecodeMisses(w.topk - hits, hits);
                }
            }
            launchCanceled.store(false, std::memory_order_release);
            queue.swap(requests);
        }
        // The first GPU publication can precede cudaGraphLaunch's return.
        // Let the dispatcher consume it without waiting for the caller's lock.
        wake.notify_one();
        if (!FastllmCudaGraphLaunch(exec)) {
            launchCanceled.store(true, std::memory_order_release);
            std::unique_lock<std::mutex> lock(mutex);
            queue.clear();
            drained.wait(lock, [&] { return !busy; });
            // Preserve the FastLLM error flag while allowing an explicit retry.
            cudaGetLastError();
            return false;
        }
        for (int i = 0; i < count; ++i) {
            auto &w = *static_cast<Layer *>(handles[i]);
            w.pending = true; w.timed = timed;
        }
        fullLast = static_cast<Layer *>(handles[count - 1]);
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
        (devices && devices->size() != 1) || !NativeMailboxOwnsLayer(group, cache, table) ||
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
