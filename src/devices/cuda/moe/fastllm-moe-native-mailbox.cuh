#pragma once

// Native, single-GPU decode handoff. A CPU dispatcher owns NUMA execution;
// the caller only submits graphs. Full replay uses an explicit one-thread
// acquire wait; the split-layer path retains stream memory waits. Every layer
// owns its activation/result/control buffers.
#include <deque>

namespace {
#if defined(USE_NUMAS) && !defined(USE_ROCM) && CUDART_VERSION >= 11070
struct NativeMoeControl {
    alignas(64) unsigned published = 0;
    alignas(64) unsigned computed = 0;
    alignas(64) unsigned planned = 0;
    int stagedCount = 0;
    alignas(64) int stagedByHits[kMaxTopK + 1];
    // Route storage follows the control header, sized for this graph's batch.
    // Derive addresses from the local host/device mapping, without assuming UVA
    // maps pinned storage to the same virtual address on both processors.
    int routes = 0;
    __host__ __device__ int32_t *ids() const { return reinterpret_cast<int32_t *>(const_cast<NativeMoeControl *>(this + 1)); }
    __host__ __device__ int32_t *resident() const { return ids() + routes; }
    __host__ __device__ int32_t *staged() const { return ids() + 2 * routes; }
    __host__ __device__ int32_t *gpu() const { return ids() + 3 * routes; }
    __host__ __device__ float *scores() const { return reinterpret_cast<float *>(ids() + 4 * routes); }
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
            box->ids()[r] = e; box->scores()[r] = scores[r];
            box->resident()[r] = slots[r] = resident ? slot : -1;
            if (publish) {
                const int selected = box->stagedByHits[__popc(mask)];
                const int at = fastllm::cuda::dsv41_cache::NativeStagedSlot(mask, r, selected);
                box->staged()[r] = slots[topk + r] = at;
                box->gpu()[r] = slots[2 * topk + r] = resident || at >= 0 ? e : -1;
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

// Standard MoE publishes FP32 activations and retains unweighted expert
// outputs. Scheduling/control ownership is shared with the V4.1 mailbox.
template <typename T>
__global__ void PublishStandardMoe(NativeMoeControl *box, float *hostInput, float *deviceInput,
        const T *input, const int32_t *indices, const float *scores,
        const int32_t *keys, const int32_t *slotKeys, int32_t *slots,
        unsigned long long *hits, unsigned long long *misses,
        int base, int experts, int hidden, int rows, int topk) {
    for (int c = threadIdx.x; c < rows * hidden; c += blockDim.x)
        hostInput[c] = deviceInput[c] = ActivationTraits<T>::ToFloat(input[c]);
    if (rows == 1) {
        if (threadIdx.x < 32) {
            const int r = threadIdx.x, e = r < topk ? indices[r] : -1, key = base + e;
            const int slot = e >= 0 && e < experts ?
                fastllm::cuda::moe_admission::LoadAdmission(keys + key) : -1;
            const bool resident = slot >= 0 && slotKeys[slot] == key;
            const unsigned mask = __ballot_sync(0xffffffffu, r < topk && resident);
            if (!r) { *hits += __popc(mask); *misses += topk - __popc(mask); }
            if (r < topk) {
                box->ids()[r] = e; box->scores()[r] = scores[r];
                box->resident()[r] = slots[r] = resident ? slot : -1;
                const int selected = box->stagedByHits[__popc(mask)];
                const int at = fastllm::cuda::dsv41_cache::NativeStagedSlot(mask, r, selected);
                box->staged()[r] = slots[topk + r] = at;
                box->gpu()[r] = slots[2 * topk + r] = resident || at >= 0 ? e : -1;
                if (at >= 0) { slots[3 * topk + at] = e; slots[4 * topk + at] = at; }
                if (!r) box->stagedCount = slots[5 * topk] = selected;
            }
        }
    } else for (int route = threadIdx.x; route < rows * topk; route += blockDim.x) {
        const int e = indices[route], key = base + e;
        int slot = e >= 0 && e < experts ?
            fastllm::cuda::moe_admission::LoadAdmission(keys + key) : -1;
        if (slot >= 0 && slotKeys[slot] != key) slot = -1;
        box->ids()[route] = e; box->scores()[route] = scores[route];
        box->resident()[route] = slots[route] = slot;
        atomicAdd(slot >= 0 ? hits : misses, 1ULL);
    }
    __threadfence_system(); __syncthreads();
    if (!threadIdx.x) atomicExch_system(&box->published, 1u);
}

__global__ void ReduceStandardMailbox(const float *cpu, const float *resident, const float *staged,
        const int32_t *residentSlots, const int32_t *gpu, const float *scores,
        float *output, int hidden, int topk, bool fused) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (col >= hidden) return;
    float sum = 0;
    for (int k = 0; k < topk; ++k) {
        const int route = row * topk + k;
        const float *source = residentSlots[route] >= 0 ? resident : gpu[route] >= 0 ? staged : cpu;
        const float value = source[size_t(route) * hidden + col];
        sum = fused ? __fmaf_rn(value, scores[route], sum) : __fadd_rn(sum, __fmul_rn(value, scores[route]));
    }
    output[row * hidden + col] = sum;
}

// Scored formats keep independent resident/staged output buffers. Their
// kernels may clear inactive routes, and the mailbox chooses exactly one
// producer for each route before the model's BF16 boundary.
__global__ void ReduceNativeScoredMailbox(const float *cpu, const float *resident,
        const float *staged, const int32_t *residentSlots, const int32_t *gpu,
        const int32_t *indices, __nv_bfloat16 *output, int hidden, int topk) {
    const int row = blockIdx.y, base = row * topk;
    cpu += size_t(base) * hidden; resident += size_t(base) * hidden; staged += size_t(base) * hidden;
    residentSlots += base; gpu += base; indices += base; output += size_t(row) * hidden;
    __shared__ int order[16];
    if(threadIdx.x<topk) {
        const int r=threadIdx.x; int rank=0;
        for(int j=0;j<topk;++j) rank+=indices[j]<indices[r] || (indices[j]==indices[r] && j<r);
        order[rank]=r;
    }
    __syncthreads();
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=hidden) return;
    float sum=0;
    for(int at=0;at<topk;++at) {
        const int r=order[at];
        const float *source=gpu[r]<0?cpu:residentSlots[r]>=0?resident:staged;
        sum=__fadd_rn(sum,source[size_t(r)*hidden+c]);
    }
    output[c]=__float2bfloat16_rn(sum);
}

// Route ordering follows the arithmetic backend's capacity, independently
// of request boundaries. Native block-FP4 kernels accept the complete batch;
// adapters with bounded scratch retain their existing compute tiles.
static int NativeBatchTileRows(const OffloadLayout &layout, int rows) {
    // The route grid uses CUDA's y dimension. Keep its limit independent of
    // the request count and of the number of rows in a speculative proposal.
    if (layout.weightType == fastllm::EXL3 || layout.weightType == fastllm::DATA_GGUF_FORMAT) return rows;
    return std::min(rows, layout.weightType == fastllm::NVFP4_BLOCK_32_E8M0 ? 65535 / kMaxTopK : 8);
}
static bool NativeBatchSortRoutes(const OffloadLayout &layout) {
    // Compressed backends that group record identities themselves consume
    // original route IDs; do not repeat a quadratic ranking at publication.
    return layout.weightType != fastllm::EXL3 && layout.weightType != fastllm::DATA_GGUF_FORMAT;
}
__device__ int NativeBatchRouteRank(const int32_t *indices, int r, int rows, int topk, int tileRows = 8) {
    const int first = r / (tileRows * topk) * (tileRows * topk);
    const int end = min(rows * topk, first + tileRows * topk);
    int rank = first;
    for (int j = first; j < end; ++j)
        rank += indices[j] < indices[r] || (indices[j] == indices[r] && j < r);
    return rank;
}
__global__ void PublishNativeBatchMoe(NativeMoeControl *box, uint16_t *hostInput,
        const uint16_t *input, const int32_t *indices, const float *scores,
        const int32_t *keys, const int32_t *slotKeys, int32_t *slots, int32_t *routeMap,
        unsigned long long *hits, unsigned long long *misses,
        int base, int experts, int hidden, int rows, int topk, int tileRows = 8, bool sorted = true) {
    for (int c = threadIdx.x; c < rows * hidden; c += blockDim.x) hostInput[c] = input[c];
    for (int r = threadIdx.x; r < rows * topk; r += blockDim.x) {
        const int expert = indices[r], key = base + expert;
        int slot = expert >= 0 && expert < experts ?
            fastllm::cuda::moe_admission::LoadAdmission(keys + key) : -1;
        if (slot >= 0 && slotKeys[slot] != key) slot = -1;
        const int at = sorted ? NativeBatchRouteRank(indices, r, rows, topk, tileRows) : r;
        slots[at] = slot;
        routeMap[at] = sorted ? r % (tileRows * topk) : r;
        box->ids()[r] = expert; box->resident()[r] = slot; box->scores()[r] = scores[r];
        atomicAdd(slot >= 0 ? hits : misses, 1ULL);
    }
    __threadfence_system(); __syncthreads();
    if (!threadIdx.x) atomicExch_system(&box->published, 1u);
}
__global__ void ReadNativeBatchPlan(const NativeMoeControl *box, const int32_t *indices,
        int32_t *staged, int32_t *gpu, int rows, int topk, bool sorted = true, int tileRows = 8) {
    for (int r = threadIdx.x; r < rows * topk; r += blockDim.x) {
        staged[sorted ? NativeBatchRouteRank(indices, r, rows, topk, tileRows) : r] = box->staged()[r];
        gpu[r] = box->gpu()[r];
    }
}

__global__ void ReadNativeMoePlan(const NativeMoeControl *box, int32_t *staged, int32_t *gpu, int topk) {
    if (threadIdx.x < topk) {
        staged[threadIdx.x] = box->staged()[threadIdx.x];
        gpu[threadIdx.x] = box->gpu()[threadIdx.x];
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
            FastllmCudaStreamGetCaptureDependencies(cudaStreamPerThread, &status, &graph, &deps, &count) != cudaSuccess ||
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
        return FastllmCudaStreamUpdateCaptureDependencies(cudaStreamPerThread, &next, 1,
                    cudaStreamSetCaptureDependencies) == cudaSuccess;
    }
};

struct NativeMoeMailbox {
    // A full graph orders each routed reduction before the next layer's
    // publication. Arithmetic scratch can be shared by compatible layers;
    // control, routing and CPU result mailboxes remain private. Keep each
    // shape alive until its last graph handle is released.
    struct ComputeScratch {
        float *partial = nullptr;
        fastllm::Data quantized, gate;
        fastllm::Data ggufWorkspace[2];
        ~ComputeScratch() { cudaFree(partial); }
        bool Init(const OffloadLayout &layout, int topk, int device, int rows) {
            const bool standard = !layout.deepSeekV41 && !layout.glm5;
            const int hidden = layout.hidden, routes = rows * topk;
            if (cudaMalloc(&partial, routes * hidden * sizeof(float) * (standard || layout.weightType==fastllm::EXL3 ? 2 : 1)) != cudaSuccess)
                return false;
            quantized.dataType = standard ? fastllm::FLOAT32 : fastllm::BFLOAT16;
            quantized.Resize({rows, hidden});
            quantized.ToDevice(fastllm::CUDA, {device}, false); quantized.Allocate(false);
            // GGUF owns separate resident/staged projection scratch below.
            // Only native experts consume this Gate buffer.
            if (standard || layout.weightType != fastllm::DATA_GGUF_FORMAT) {
                gate.dataType = standard ? fastllm::FLOAT32 : fastllm::BFLOAT16;
                // Standard GGUF kernels clear inactive routes. Parallel
                // resident/staged branches must not share activation scratch.
                gate.Resize({routes * (standard ? 2 : 1), layout.inter});
                gate.ToDevice(fastllm::CUDA, {device}, false); gate.Allocate(false);
            }
            if (layout.weightType == fastllm::DATA_GGUF_FORMAT) {
                const size_t bytes = standard ? FastllmCudaMoeGGUFCacheBatchWorkspaceBytes(hidden, layout.inter, rows, topk) :
                    FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(hidden, layout.inter, rows, topk);
                if (!bytes || bytes>INT_MAX) return false;
                // Resident and staged branches execute concurrently. Neither
                // quantization scratch nor activation storage may alias.
                for (int branch = 0; branch < 2; ++branch) {
                    auto &scratch = ggufWorkspace[branch];
                    scratch.dataType = fastllm::INT8; scratch.Resize({int(bytes)});
                    scratch.ToDevice(fastllm::CUDA, {device}, false); scratch.Allocate(false);
                }
            }
            return true;
        }
    };
    struct Layer {
        NativeMoeControl *host = nullptr, *mapped = nullptr;
        float *hostData = nullptr, *mappedData = nullptr;
        std::shared_ptr<ComputeScratch> compute;
        int32_t *slots = nullptr; // resident, staged, GPU route IDs
        int32_t *routeMap = nullptr;
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
        bool timed = true, standard = false, sharedQ8Input = false;
        std::atomic<bool> pending{false}, submitted{false};
        ~Layer() {
            if (pending && completed) cudaEventSynchronize(completed);
            if (fullGraph) cudaGraphDestroy(fullGraph);
            if (exec) cudaGraphExecDestroy(exec);
            if (finishExec) cudaGraphExecDestroy(finishExec);
            for (auto e : {begin, residentDone, stagedBegin, done, completed, uploadBegin, uploadDone}) if (e) cudaEventDestroy(e);
            cudaFreeHost(host); cudaFreeHost(hostData); cudaFree(slots); cudaFree(routeMap);
        }
        bool Init(const OffloadLayout &layout, int selected, int device, int inputRows = 1, std::shared_ptr<ComputeScratch> shared = {}) {
            standard = !layout.deepSeekV41 && !layout.glm5;
            hidden = layout.hidden; topk = selected; rows = inputRows; routes = rows * topk;
            if (cudaHostAlloc(&host, sizeof(*host) + size_t(routes) * 5 * sizeof(int32_t), cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(&mapped, host, 0) != cudaSuccess ||
                cudaHostAlloc(&hostData, (routes + rows) * hidden * sizeof(float), cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(&mappedData, hostData, 0) != cudaSuccess ||
                cudaMalloc(&slots, (rows > 1 ? 3 * routes : 5 * routes + 1) * sizeof(int32_t)) != cudaSuccess ||
                (!standard && rows > 1 && cudaMalloc(&routeMap, routes * sizeof(int32_t)) != cudaSuccess)) return false;
            new (host) NativeMoeControl{};
            host->routes = routes;
            std::fill_n(host->ids(), size_t(routes) * 5, int32_t(0));
            std::fill_n(hostData, (routes + rows) * hidden, 0.0f);
            for (auto *e : {&begin, &residentDone, &stagedBegin, &done, &completed, &uploadBegin, &uploadDone})
                if (cudaEventCreate(e) != cudaSuccess) return false;
            compute = std::move(shared);
            if (!compute) {
                compute = std::make_shared<ComputeScratch>();
                if (!compute->Init(layout, topk, device, rows)) return false;
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
    using ScratchKey = std::array<int, 7>;
    std::map<ScratchKey, std::weak_ptr<ComputeScratch>> fullScratch;
    std::thread dispatcher;
    std::vector<int> dispatcherCpus;
    // Request cleanup may release a draft graph on a consumer thread while
    // the executor creates or replays another graph owned by this mailbox.
    // Serialize handle ownership and timing retirement, separately from the
    // dispatch queue lock. Finish() is also called by the owning operations.
    std::recursive_mutex lifecycleMutex;
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
        std::lock_guard<std::recursive_mutex> lifecycle(lifecycleMutex);
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
        if (cache.pageableSource) cache.pageableSource->Check();
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
                    fastllm::AssertInFastLLM(box.ids()[i] >= 0 && box.ids()[i] < layout.experts,
                                           "Native mailbox invalid expert ID.");
                    if (box.resident()[i] >= 0) {
                        if (!full) order[hits] = i;
                        ++hits;
                    }
                    if (!full) {
                        box.gpu()[i] = box.resident()[i] >= 0 ? box.ids()[i] : -1;
                        box.staged()[i] = -1;
                    }
                }
                int next = hits;
                for (int i = 0; !full && i < w.topk; ++i)
                    if (box.resident()[i] < 0) order[next++] = i;
                cache.frequency->Observe(w.table * layout.experts, box.ids(), w.topk);
                const int staged = full ? box.stagedCount : w.cost.SelectDecodeMisses(w.topk - hits, hits);
                for (int i = 0; !full && i < staged; ++i) {
                    const int route = order[hits + i];
                    box.staged()[route] = i; box.gpu()[route] = box.ids()[route];
                }
                w.hits = hits; w.staged = staged;
                double cpuUs = 0;
                auto upload = [&] {
                    if (full) return;
                    const double start = HybridNowUs();
                    for (int i = 0; i < staged; ++i)
                        copy->CopyExpert(group, w.table, box.ids()[order[hits + i]], i, staged);
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
                    w.weights.data(), box.ids(), box.gpu(), w.topk, w.modelLayer, upload,
                    w.standard ? nullptr : box.scores(), layout.swigluLimit, 32, &cpuUs);
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
                    std::fill_n(box.staged(), w.routes, -1); std::fill_n(box.gpu(), w.routes, -1);
                    box.stagedCount = 0;
                    __atomic_store_n(&box.planned, 1u, __ATOMIC_RELEASE);
                }
                if (!full) {
                    std::fill_n(box.gpu(), w.topk, -1); std::fill_n(box.staged(), w.topk, -1);
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
            fastllm::AssertInFastLLM(box.ids()[r] >= 0 && box.ids()[r] < layout.experts,
                                   "Native verifier mailbox invalid expert ID.");
            box.staged()[r] = -1;
            box.gpu()[r] = box.resident()[r] >= 0 ? box.ids()[r] : -1;
            if (box.resident()[r] >= 0) { ++w.hits; continue; }
            auto found = std::find(missed.begin(), missed.end(), box.ids()[r]);
            if (found == missed.end()) { missed.push_back(box.ids()[r]); reuse.push_back(1); }
            else ++reuse[found - missed.begin()];
        }
        for (size_t i = 0; i < missed.size(); ++i) for (size_t j = i + 1; j < missed.size(); ++j)
            if (reuse[j] > reuse[i]) { std::swap(reuse[i], reuse[j]); std::swap(missed[i], missed[j]); }
        cache.frequency->Observe(w.table * layout.experts, box.ids(), w.routes);
        w.staged = std::min(kMaxTopK, w.cost.SelectMisses(missed.size(), w.hits, reuse.data()));
        w.stagedRoutes = 0;
        for (int i = 0; i < w.staged; ++i) {
            box.stagedExperts[i] = missed[i]; box.stagedSlots[i] = i;
            for (int r = 0; r < w.routes; ++r) if (box.ids()[r] == missed[i]) {
                box.staged()[r] = i; box.gpu()[r] = box.ids()[r]; ++w.stagedRoutes;
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
            if (w.standard) fastllm::NumasMoeDecodeExpertsBatch(w.hostData,
                w.hostData + w.rows * w.hidden, w.rows, w.weights.data(), w.weights.size(),
                box.ids(), box.gpu(), box.scores(), w.topk, w.modelLayer);
            else fastllm::NumasMoeVerifyExpertsWithOverlap(
                reinterpret_cast<const uint16_t *>(w.hostData),
                w.hostData + w.rows * w.hidden, w.rows, w.weights.data(), w.weights.size(),
                box.ids(), box.gpu(), box.scores(), w.topk, w.modelLayer, layout.swigluLimit,
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
    bool ComputeStandard(Layer &w, const OffloadLayout &layout, const uint8_t *records,
            const int32_t *slots, const float *scores, bool resident) {
        const int branch = resident ? 0 : 1;
        if (layout.weightType == fastllm::NVFP4_BLOCK_16_E4M3 && w.rows > 1 &&
                SupportsNVFP4WideDecode(layout)) {
            fastllm::Data gate;
            gate.FakeFrom(w.compute->gate, size_t(branch) * w.routes * layout.inter * sizeof(float));
            gate.Resize({w.routes, layout.inter});
            FastllmCudaMoeGGUFCacheView view{};
            view.records = records; view.routeSlots = slots;
            view.recordStride = layout.recordStride; view.downOffset = layout.downOffset;
            view.hidden = w.hidden; view.inter = layout.inter; view.routeCount = w.routes;
            view.slotOffsets = resident ? cache.slotOffsets : nullptr;
            return ComputeNVFP4Batch(w.compute->quantized, gate, layout, view, w.topk,
                w.compute->partial + size_t(branch) * w.routes * w.hidden);
        }
        if (layout.weightType == fastllm::DATA_GGUF_FORMAT && w.rows > 1) {
            // Keep verifier rows in the same projection launch. Splitting
            // them into single-row graphs repeats quantization and dispatch
            // work that the ordinary hybrid batch backend already shares.
            fastllm::Data gate, dummy;
            gate.FakeFrom(w.compute->gate, size_t(branch) * w.routes * layout.inter * sizeof(float));
            gate.Resize({w.routes, layout.inter});
            FastllmCudaMoeGGUFCacheView view{records, slots,
                layout.recordStride, layout.downOffset, layout.gateGgmlType, layout.downGgmlType,
                w.hidden, layout.inter, w.compute->ggufWorkspace[branch].cudaData,
                size_t(w.compute->ggufWorkspace[branch].GetBytes()), resident ? cache.slotOffsets : nullptr};
            SetStandardGGUFLayout(group, w.table, view);
            view.exactSubwarp = group.qwen38FlashNext && w.rows > 1;
            if (w.sharedQ8Input) view.preparedQ8Input = w.compute->ggufWorkspace[0].cudaData;
            return ComputeGGUFExperts(w.compute->quantized, gate, dummy, layout, view, scores,
                w.topk, w.compute->partial + size_t(branch) * w.routes * w.hidden);
        }
        for (int row = 0; row < w.rows; ++row) {
            fastllm::Data input, gate, dummy;
            input.FakeFrom(w.compute->quantized, size_t(row) * w.hidden * sizeof(float));
            input.Resize({1, w.hidden});
            gate.FakeFrom(w.compute->gate, (size_t(branch) * w.routes + row * w.topk) * layout.inter * sizeof(float));
            gate.Resize({w.topk, layout.inter});
            float *partial = w.compute->partial + (size_t(branch) * w.routes + row * w.topk) * w.hidden;
            if (layout.weightType == fastllm::DATA_GGUF_FORMAT) {
                FastllmCudaMoeGGUFCacheView view{records, slots + row * w.topk,
                    layout.recordStride, layout.downOffset, layout.gateGgmlType, layout.downGgmlType,
                    w.hidden, layout.inter, w.compute->ggufWorkspace[branch].cudaData,
                    size_t(w.compute->ggufWorkspace[branch].GetBytes()), resident ? cache.slotOffsets : nullptr};
                SetStandardGGUFLayout(group, w.table, view);
            view.exactSubwarp = group.qwen38FlashNext && w.rows > 1;
                if (w.sharedQ8Input) view.preparedQ8Input = w.compute->ggufWorkspace[0].cudaData;
                if (!ComputeGGUFExperts(input, gate, dummy, layout, view,
                        scores + row * w.topk, w.topk, partial)) return false;
            } else {
                DeviceCache view;
                view.records = const_cast<uint8_t *>(records); view.routeSlots = const_cast<int32_t *>(slots + row * w.topk);
                if (!ComputeNVFP4Cache(input, gate, dummy, layout, view,
                        scores + row * w.topk, w.topk, partial)) return false;
            }
        }
        return true;
    }
    bool PublishStandard(Layer &w, const OffloadLayout &layout, const fastllm::Data &input,
            const fastllm::Data &index, const fastllm::Data &score) {
        auto launch = [&](auto *typed) {
            PublishStandardMoe<<<1, 256, 0, cudaStreamPerThread>>>(w.mapped, w.mappedData,
                static_cast<float *>(w.compute->quantized.cudaData), typed,
                static_cast<const int32_t *>(index.cudaData), static_cast<const float *>(score.cudaData),
                cache.keyToSlot, cache.slotKeys, w.slots, cache.hitCount, cache.totalMissCount,
                w.table * layout.experts, layout.experts, w.hidden, w.rows, w.topk);
        };
        switch (input.dataType) {
        case fastllm::FLOAT16: launch(static_cast<const half *>(input.cudaData)); break;
        case fastllm::BFLOAT16: launch(static_cast<const __nv_bfloat16 *>(input.cudaData)); break;
        case fastllm::FLOAT32: launch(static_cast<const float *>(input.cudaData)); break;
        default: return false;
        }
        return cudaGetLastError() == cudaSuccess;
    }
    static bool ComputeBatch(Layer &w, const OffloadLayout &layout, const uint8_t *records,
            const int32_t *slots, const float *scores) {
        const int tileRows = NativeBatchTileRows(layout, w.rows);
        for (int first = 0; first < w.rows; first += tileRows) {
            const int rows = std::min(tileRows, w.rows - first), base = first * w.topk;
            fastllm::Data input, gate, dummy;
            input.FakeFrom(w.compute->quantized, size_t(first) * w.hidden * 2); input.Resize({rows, w.hidden});
            gate.FakeFrom(w.compute->gate, size_t(base) * layout.inter * 2); gate.Resize({rows*w.topk, layout.inter});
            FastllmCudaMoeGGUFCacheView view{};
            view.records = records; view.routeSlots = const_cast<int32_t *>(slots + base);
            view.routeMap = w.routeMap ? w.routeMap + base : nullptr; view.routeCount = rows * w.topk;
            if (!ComputeBatchExperts(input, gate, dummy, layout, view, scores + base,
                    w.topk, w.compute->partial + size_t(base) * w.hidden, true)) return false;
        }
        return true;
    }
    bool ComputeGGUF(Layer &w, const OffloadLayout &layout, const uint8_t *records,
                     const int32_t *slots, const float *scores, bool resident) {
        const int branch = resident ? 0 : 1;
        FastllmCudaMoeGGUFCacheView view{records, const_cast<int32_t *>(slots),
            layout.recordStride, layout.downOffset, layout.gateGgmlType, layout.downGgmlType,
            layout.hidden, layout.inter, w.compute->ggufWorkspace[branch].cudaData,
            size_t(w.compute->ggufWorkspace[branch].GetBytes()), resident ? cache.slotOffsets : nullptr};
        view.routeMap = w.routeMap; view.routeCount = w.routes;
        // The resident, staged and CPU branches own disjoint routes.
        view.preserveMissing = true;
        return FastllmCudaMoeV41CanonicalGGUFCacheCompute(w.compute->quantized,
            view, scores, w.topk, layout.swigluLimit, w.compute->partial);
    }
    static void Compute(Layer &w, const OffloadLayout &layout, const uint8_t *records,
            const int32_t *slots, const float *scores, cudaStream_t stream, bool resident = false) {
        if (layout.weightType==fastllm::EXL3) {
            // Resident, staged and shared experts have independent scratch;
            // their captured branches can execute concurrently.
            fastllm::AssertInFastLLM(stream==cudaStreamPerThread && w.owner,
                "EXL3 mailbox requires the full-graph compute stream");
            fastllm::AssertInFastLLM(FastllmCudaExl3Cache(w.compute->quantized, records, slots,
                resident ? w.owner->cache.slotOffsets : nullptr, layout.recordStride, layout.downOffset,
                layout.inter, layout.gateBlockK, layout.downBlockK, scores, w.topk, layout.swigluLimit,
                w.compute->partial + (resident ? 0 : size_t(w.routes) * w.hidden),
                resident ? 0 : 1, w.routeMap, w.routes), "EXL3 graph expert compute failed");
            return;
        }
        if (resident && w.hidden % 256 == 0 && layout.inter % 32 == 0 && FastllmCudaRuntimeArch() == 120) {
            if (layout.planarMxfp4)
                fastllm::cuda::dsv41_cache::GateResident<true><<<dim3(layout.inter / 32, w.topk), 512, 0, stream>>>(
                    static_cast<const __nv_bfloat16 *>(w.compute->quantized.cudaData), slots, records, scores,
                    static_cast<__nv_bfloat16 *>(w.compute->gate.cudaData), w.hidden, layout.inter,
                    layout.recordStride, layout.swigluLimit);
            else
                fastllm::cuda::dsv41_cache::GateResident<false><<<dim3(layout.inter / 32, w.topk), 512, 0, stream>>>(
                    static_cast<const __nv_bfloat16 *>(w.compute->quantized.cudaData), slots, records, scores,
                    static_cast<__nv_bfloat16 *>(w.compute->gate.cudaData), w.hidden, layout.inter,
                    layout.recordStride, layout.swigluLimit);
        } else fastllm::cuda::dsv41_cache::Gate<<<dim3(layout.inter / 32, w.topk), 512, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(w.compute->quantized.cudaData), slots, records,
            scores, static_cast<__nv_bfloat16 *>(w.compute->gate.cudaData),
            w.hidden, layout.inter, layout.recordStride, layout.swigluLimit, 1, nullptr, layout.planarMxfp4);
        fastllm::cuda::dsv41_cache::Down<<<dim3(w.hidden / 8, w.topk), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(w.compute->gate.cudaData), slots, records, w.compute->partial,
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
            if (!FastllmCudaDeepSeekV41QuantizeActivation(input, w.compute->quantized)) return false;
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
                w.mappedData+w.hidden, w.compute->partial, w.slots+2*w.topk, static_cast<const int32_t *>(index.cudaData),
                static_cast<__nv_bfloat16 *>(output.cudaData), w.hidden, w.topk);
            const bool reset = memops.Append(&w.mapped->published, false, 0) &&
                memops.Append(&w.mapped->computed, false, 0) && cudaGetLastError() == cudaSuccess;
            return reset && cudaEventRecordWithFlags(w.completed, cudaStreamPerThread, cudaEventRecordExternal) == cudaSuccess;
        })) return false;
        w.pointers = {input.cudaData, index.cudaData, score.cudaData, output.cudaData};
        return true;
    }
    void WarmupGGUF(Layer &w,const OffloadLayout &layout) {
        if (!w.standard || layout.weightType!=fastllm::DATA_GGUF_FORMAT || w.rows<2) return;
        FastllmCudaMoeGGUFCacheView view{cache.records,w.slots,layout.recordStride,layout.downOffset,
            layout.gateGgmlType,layout.downGgmlType,w.hidden,layout.inter,
            w.compute->ggufWorkspace[0].cudaData,size_t(w.compute->ggufWorkspace[0].GetBytes()),cache.slotOffsets};
        SetStandardGGUFLayout(group,w.table,view);view.exactSubwarp=group.qwen38FlashNext;
        auto &input=w.compute->quantized;
        if (!FastllmCudaMoeGGUFCacheNeedsPreparation(input,view,w.topk)) return;
        // Benchmark only committed records of this layer: mixed-format GGUF
        // layers can share a cache but not each other's byte interpretation.
        WaitCacheAdmission(cache);
        std::vector<int> keys(cache.slots),available;
        fastllm::kernels::CudaCheck(cudaMemcpyAsync(keys.data(),cache.slotKeys,keys.size()*sizeof(int),
            cudaMemcpyDeviceToHost,cudaStreamPerThread));
        fastllm::kernels::CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
        for (int i=0;i<cache.slots && available.size()<size_t(w.routes);++i)
            if (keys[i]>=w.table*layout.experts && keys[i]<(w.table+1)*layout.experts) available.push_back(i);
        if (available.empty()) return;
        std::vector<int> slots(w.routes);
        const int unique=std::min<int>(available.size(),std::max(w.topk,(w.routes+1)/2));
        for (int i=0;i<w.routes;++i) slots[i]=available[i%unique];
        std::vector<float> x(size_t(w.rows)*w.hidden),scores(w.routes,1.f/w.topk);
        for (size_t i=0;i<x.size();++i)x[i]=float(int((i*73)%257)-128)/64;
        fastllm::Data score(fastllm::FLOAT32,{w.rows,w.topk}),gate,dummy;
        score.ToDevice(fastllm::CUDA,{cache.device},false);score.Allocate(false);
        gate.FakeFrom(w.compute->gate,0);gate.Resize({w.routes,layout.inter});
        auto upload=[&](void *dst,const void *src,size_t bytes) {
            fastllm::kernels::CudaCheck(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,cudaStreamPerThread));
        };
        upload(input.cudaData,x.data(),x.size()*sizeof(float));
        upload(w.slots,slots.data(),slots.size()*sizeof(int));
        upload(score.cudaData,scores.data(),scores.size()*sizeof(float));
        if (!ComputeGGUFExperts(input,gate,dummy,layout,view,static_cast<float *>(score.cudaData),
                w.topk,w.compute->partial)) throw std::runtime_error("GGUF kernel warmup rejected");
        fastllm::kernels::CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
    }

    Layer *CreateFull(int table, const fastllm::Data &input, const fastllm::Data &index,
            const fastllm::Data &score, fastllm::Data &output, fastllm::Data **weights,
            int weightsBatch, int layer, cudaGraph_t sharedGraph) {
        std::lock_guard<std::recursive_mutex> lifecycle(lifecycleMutex);
        if (failed) return nullptr;
        Finish(true);
        const auto &layout = group.LayerLayout(table);
        const int topk = index.dims.back();
        if (!fullCopy) fullCopy = MakeDecodeOverlap(group, group.tableKeys.size(), kMaxTopK);
        if (!fullCopy) return nullptr;
        const ScratchKey scratchKey{layout.hidden, layout.inter, input.dims[0], topk,
            int(layout.weightType), int(layout.deepSeekV41), int(layout.glm5)};
        auto fresh = std::make_unique<Layer>();
        if (!fresh->Init(layout, topk, cache.device, input.dims[0], fullScratch[scratchKey].lock()))
            return nullptr;
        fullScratch[scratchKey] = fresh->compute;
        auto &w = *fresh;
        w.sharedQ8Input = group.qwen38FlashNext && w.standard &&
            layout.weightType == fastllm::DATA_GGUF_FORMAT &&
            FastllmCudaMoeGGUFCacheQ8Supported(layout.gateGgmlType, layout.downGgmlType, w.hidden, layout.inter);
        w.owner = this; w.table = table; w.modelLayer = layer;
        w.weights.assign(weights, weights + weightsBatch);
        auto measured = layers.find(table);
        if (measured != layers.end()) w.cost = measured->second->cost;
        output.dataType = w.standard ? fastllm::FLOAT32 : input.dataType; output.Resize({w.rows, w.hidden});
        output.ToDevice(fastllm::CUDA, {cache.device}, false); output.Allocate(false);
        w.pointers = {input.cudaData, index.cudaData, score.cudaData, output.cudaData};
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, cache.device) != cudaSuccess) return nullptr;
        if (layout.weightType == fastllm::NVFP4_BLOCK_32_E8M0) (void)Mxfp4BatchMmaSupported();
        WarmupMxfp4KernelPlan(layout,cache,w.rows,topk);
        WarmupGGUF(w,layout);
        const auto launch = fastllm::cuda::RecordCopyConfiguration(layout.recordStride, kMaxTopK, prop);
        if (layout.weightType==fastllm::EXL3) {
            if (!PrepareCacheInput(layout,input,w.compute->quantized)) return nullptr;
            // Allocate/load both adapter paths before any captured node can
            // wait on host publication. Inactive routes do not read records.
            if(cudaMemsetAsync(w.slots,255,(w.rows>1?3*w.routes:5*w.routes+1)*sizeof(int32_t),cudaStreamPerThread)!=cudaSuccess) return nullptr;
            if(w.routeMap && cudaMemsetAsync(w.routeMap,255,w.routes*sizeof(int32_t),cudaStreamPerThread)!=cudaSuccess) return nullptr;
            Compute(w,layout,cache.records,w.slots,static_cast<const float *>(score.cudaData),cudaStreamPerThread,true);
            Compute(w,layout,fullCopy->records,w.slots+w.routes,static_cast<const float *>(score.cudaData),cudaStreamPerThread,false);
            if(cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess) return nullptr;
        }
        if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return nullptr;
        bool ok = [&] {
            if (w.standard) { if (!PublishStandard(w, layout, input, index, score)) return false; }
            else if (w.rows > 1) PublishNativeBatchMoe<<<1, 256, 0, cudaStreamPerThread>>>(w.mapped,
                reinterpret_cast<uint16_t *>(w.mappedData), static_cast<const uint16_t *>(input.cudaData),
                static_cast<const int32_t *>(index.cudaData), static_cast<const float *>(score.cudaData),
                cache.keyToSlot, cache.slotKeys, w.slots, w.routeMap, cache.hitCount, cache.totalMissCount,
                table * layout.experts, layout.experts, w.hidden, w.rows, topk, NativeBatchTileRows(layout, w.rows), NativeBatchSortRoutes(layout));
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
                (FastllmCudaStreamGetCaptureDependencies(cudaStreamPerThread, &status, &graph, &deps, &count) != cudaSuccess ||
                 cudaGraphAddChildGraphNode(&shared, graph, nullptr, 0, sharedGraph) != cudaSuccess)) return false;
            if (!w.standard && !PrepareCacheInput(layout, input, w.compute->quantized)) return false;
            if (cudaEventRecordWithFlags(w.begin, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            // Both graph branches consume one immutable quantized activation.
            // Record it before saving the fork dependencies so staged misses
            // cannot read a previous token while resident kernels are running.
            if (w.sharedQ8Input && !FastllmCudaMoeGGUFPrepareQ8Input(w.compute->quantized,
                    w.compute->ggufWorkspace[0].cudaData, w.compute->ggufWorkspace[0].GetBytes())) return false;
            if (FastllmCudaStreamGetCaptureDependencies(cudaStreamPerThread, &status, &graph, &deps, &count) != cudaSuccess) return false;
            std::vector<cudaGraphNode_t> fork(deps, deps + count);
            std::vector<cudaGraphNode_t> residentFork = fork;
            // Resolve the small batch plan before forking the two GPU paths.
            // Otherwise the plan reader itself can queue behind resident CTA
            // waves, leaving the entire copy branch on the completion tail.
            if (w.rows > 1) {
                WaitNativeMoe<<<1, 1, 0, cudaStreamPerThread>>>(&w.mapped->planned);
                ReadNativeBatchPlan<<<1, 256, 0, cudaStreamPerThread>>>(w.mapped,
                    static_cast<const int32_t *>(index.cudaData), w.slots + w.routes,
                    w.slots + 2 * w.routes, w.rows, topk, !w.standard && NativeBatchSortRoutes(layout), NativeBatchTileRows(layout, w.rows));
                if (FastllmCudaStreamGetCaptureDependencies(cudaStreamPerThread, &status, &graph, &deps, &count) != cudaSuccess) return false;
                residentFork.assign(deps, deps + count);
            }
            if (cudaEventRecordWithFlags(w.uploadBegin, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            // GPU reads the selected pinned records using device-side IDs.
            // No CUDA work is submitted by the CPU dispatcher during replay.
            const int32_t *stagedExperts = w.rows > 1 ? w.mapped->stagedExperts : w.slots + 3 * topk;
            const int32_t *stagedSlots = w.rows > 1 ? w.mapped->stagedSlots : w.slots + 4 * topk;
            const int32_t *stagedCount = w.rows > 1 ? &w.mapped->stagedCount : w.slots + 5 * topk;
            const int maxStaged = w.rows > 1 ? kMaxTopK : topk;
            if (cache.pageableSource) {
                if (!cache.pageableSource->Copy(table, fullCopy->records, stagedExperts, stagedSlots,
                        stagedCount, nullptr, maxStaged, launch, cudaStreamPerThread, true)) return false;
            } else if (!group.ggufSources.empty()) {
                const auto &source = group.ggufSources[table];
                auto *pointers = cache.numaPointers + size_t(table) * layout.experts * 2 * source.shards;
                // Device IDs select immutable mapped NUMA shards. Restore the
                // canonical record losslessly, including per-layer IQ/R4 types.
                const auto copied = NativeStandardGGUFRecords(group)
                    ? fastllm_gguf_restore::CopyNativeRecords(source, pointers, fullCopy->records,
                        cudaStreamPerThread, stagedExperts, stagedSlots, stagedCount)
                    : fastllm_gguf_restore::CopyRecords(source, pointers, fullCopy->records,
                        cudaStreamPerThread, stagedExperts, stagedSlots, stagedCount);
                if (copied != cudaSuccess) return false;
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
            if (w.standard) {
                if (!ComputeStandard(w, layout, fullCopy->records, w.slots + w.routes,
                        static_cast<const float *>(score.cudaData), false)) return false;
            } else if (layout.weightType == fastllm::DATA_GGUF_FORMAT) {
                if (!ComputeGGUF(w, layout, fullCopy->records, w.slots + w.routes,
                        static_cast<const float *>(score.cudaData), false)) return false;
            } else if (layout.weightType != fastllm::EXL3 && (w.rows > 1 || layout.weightType == fastllm::FP8_E4M3_BLOCK_128)) {
                if (!ComputeBatch(w, layout, fullCopy->records, w.slots + w.routes, static_cast<const float *>(score.cudaData))) return false;
            } else Compute(w, layout, fullCopy->records, w.slots + w.routes, static_cast<const float *>(score.cudaData), cudaStreamPerThread);
            if (cudaEventRecordWithFlags(w.done, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            if (FastllmCudaStreamGetCaptureDependencies(cudaStreamPerThread, &status, &graph, &deps, &count) != cudaSuccess) return false;
            std::vector<cudaGraphNode_t> stagedTail(deps, deps + count);
            // The resident branch consumes the same immutable activation and
            // writes disjoint route rows. Join both branches before reducing.
            if (FastllmCudaStreamUpdateCaptureDependencies(cudaStreamPerThread, residentFork.data(), residentFork.size(),
                    cudaStreamSetCaptureDependencies) != cudaSuccess) return false;
            if (w.standard) {
                if (!ComputeStandard(w, layout, cache.records, w.slots,
                        static_cast<const float *>(score.cudaData), true)) return false;
            } else if (layout.weightType == fastllm::DATA_GGUF_FORMAT) {
                if (!ComputeGGUF(w, layout, cache.records, w.slots,
                        static_cast<const float *>(score.cudaData), true)) return false;
            } else if (layout.weightType != fastllm::EXL3 && (w.rows > 1 || layout.weightType == fastllm::FP8_E4M3_BLOCK_128)) {
                if (!ComputeBatch(w, layout, cache.records, w.slots, static_cast<const float *>(score.cudaData))) return false;
            } else Compute(w, layout, cache.records, w.slots, static_cast<const float *>(score.cudaData), cudaStreamPerThread, true);
            if (cudaEventRecordWithFlags(w.residentDone, cudaStreamPerThread, cudaEventRecordExternal) != cudaSuccess) return false;
            if (FastllmCudaStreamUpdateCaptureDependencies(cudaStreamPerThread, stagedTail.data(), stagedTail.size(),
                    cudaStreamAddCaptureDependencies) != cudaSuccess) return false;
            WaitNativeMoe<<<1, 1, 0, cudaStreamPerThread>>>(&w.mapped->computed);
            if (layout.weightType==fastllm::EXL3) ReduceNativeScoredMailbox<<<dim3((w.hidden+255)/256,w.rows),256,0,cudaStreamPerThread>>>(
                w.mappedData+w.rows*w.hidden,w.compute->partial,w.compute->partial+size_t(w.routes)*w.hidden,
                w.rows>1?w.mapped->resident():w.slots,w.slots+2*w.routes,static_cast<const int32_t *>(index.cudaData),
                static_cast<__nv_bfloat16 *>(output.cudaData),w.hidden,topk);
            else if (w.standard) ReduceStandardMailbox<<<dim3((w.hidden+255)/256, w.rows), 256, 0, cudaStreamPerThread>>>(
                w.mappedData + w.rows * w.hidden, w.compute->partial, w.compute->partial + w.routes * w.hidden,
                w.slots, w.slots + 2 * w.routes, static_cast<const float *>(score.cudaData),
                static_cast<float *>(output.cudaData), w.hidden, topk, w.rows > 1);
            else fastllm::cuda::dsv41_cache::Reduce<<<dim3((w.hidden+255)/256, w.rows), 256, 0, cudaStreamPerThread>>>(
                w.mappedData+w.rows*w.hidden, w.compute->partial, w.slots+2*w.routes, static_cast<const int32_t *>(index.cudaData),
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
        std::lock_guard<std::recursive_mutex> lifecycle(lifecycleMutex);
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
                if (FastllmCudaGraphNodeGetDependencies(node, nullptr, &beforeCount) != cudaSuccess ||
                    FastllmCudaGraphNodeGetDependentNodes(node, nullptr, &afterCount) != cudaSuccess) return false;
                std::vector<cudaGraphNode_t> before(beforeCount), after(afterCount);
                if (FastllmCudaGraphNodeGetDependencies(node, before.data(), &beforeCount) != cudaSuccess ||
                    FastllmCudaGraphNodeGetDependentNodes(node, after.data(), &afterCount) != cudaSuccess) return false;
                for (auto next : after) {
                    size_t n = 0;
                    if (FastllmCudaGraphNodeGetDependencies(next, nullptr, &n) != cudaSuccess) return false;
                    std::vector<cudaGraphNode_t> existing(n);
                    if (FastllmCudaGraphNodeGetDependencies(next, existing.data(), &n) != cudaSuccess) return false;
                    for (auto prev : before)
                        if (std::find(existing.begin(), existing.end(), prev) == existing.end() &&
                            FastllmCudaGraphAddDependencies(graph, &prev, &next, 1) != cudaSuccess) return false;
                }
                if (cudaGraphDestroyNode(node) != cudaSuccess) return false;
            }
        }
        return true;
    }
    bool CreateReplay(cudaGraph_t source, void *const *handles, int count, void **result) {
        std::lock_guard<std::recursive_mutex> lifecycle(lifecycleMutex);
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
        std::lock_guard<std::recursive_mutex> lifecycle(lifecycleMutex);
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
        std::lock_guard<std::recursive_mutex> lifecycle(lifecycleMutex);
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
