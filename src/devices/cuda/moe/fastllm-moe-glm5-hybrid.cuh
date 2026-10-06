#pragma once

// Included after the shared cache and batched workspace definitions.
namespace {
struct Glm5MultiGpuVerify {
    using Rank = FastllmCudaMoeExpertParallel::Rank;
    using Estimate = fastllm::MoeDecodeScheduler::Estimate;
    struct DeviceWork {
        Rank rank;
        fastllm::Data input, received;
        cudaEvent_t inputStart = nullptr, inputEnd = nullptr;
        cudaEvent_t returnStart = nullptr, returnEnd = nullptr;
        cudaEvent_t mergeStart = nullptr, mergeEnd = nullptr;
        Estimate inputPerRow, returnPerRow, mergePerRow;
        int previousRows = 0, previousPeers = 0;
        bool sent = false, merged = false;
        std::vector<int> offsets;

        ~DeviceWork() {
            int previous = 0;
            cudaGetDevice(&previous);
            if (rank.cudaDevice >= 0) cudaSetDevice(rank.cudaDevice);
            if (rank.pending) cudaEventSynchronize(rank.done);
            input.FreeSpace(); received.FreeSpace();
            for (auto event : {inputStart, inputEnd, returnStart, returnEnd, mergeStart, mergeEnd})
                if (event) cudaEventDestroy(event);
            cudaSetDevice(previous);
        }
        bool Prepare(int device, int hidden) {
            rank.cudaDevice = device;
            if (!rank.Prepare(hidden)) return false;
            float ms = 0;
            if (sent) {
                checkCudaErrors("GLM input timing", cudaEventElapsedTime(&ms, inputStart, inputEnd));
                inputPerRow.Observe(ms * 1000 / previousRows);
                checkCudaErrors("GLM result timing", cudaEventElapsedTime(&ms, returnStart, returnEnd));
                returnPerRow.Observe(ms * 1000 / previousRows);
            }
            if (merged) {
                checkCudaErrors("GLM merge timing", cudaEventElapsedTime(&ms, mergeStart, mergeEnd));
                mergePerRow.Observe(ms * 1000 / (previousRows * previousPeers));
            }
            sent = merged = false;
            for (auto *event : {&inputStart, &inputEnd, &returnStart, &returnEnd, &mergeStart, &mergeEnd})
                if (!*event && cudaEventCreate(event) != cudaSuccess) return false;
            return true;
        }
    };
    std::map<int, std::unique_ptr<DeviceWork>> devices;
    std::vector<Estimate> cpu;
    std::vector<uint64_t> calls;
    std::mutex mutex;
    DeviceWork *previousRoot = nullptr;

    ~Glm5MultiGpuVerify() {
        // A peer's pinned result may still be read by the origin GPU's H2D.
        if (previousRoot) cudaEventSynchronize(previousRoot->rank.done);
    }
};

// Preserve each expert's FP32 result until the existing ordered BF16 reduction.
// Summing GPU-local partial outputs first would change GLM's rounding order.
__global__ void CopyOwnedGlm5Routes(float *destination, const float *source,
        const int32_t *owners, int owner, int hidden, int routes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden * routes && owners[i / hidden] == owner) destination[i] = source[i];
}

bool TryGlm5MultiGpuVerify(OffloadGroup &group, DeviceCache &origin, int table,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch, int layer,
        const std::function<void()> &launchParallel) {
    using namespace fastllm;
    using Scheduler = MoeDecodeOverlapScheduler;
    const auto &layout = group.LayerLayout(table);
    const int rows = input.dims[0], hidden = layout.hidden;
    if (!origin.frequencyActive || !origin.hostKeyToSlot || !NativeSharedRecords(group) || input.dataType != BFLOAT16 ||
        rows <= 1 || input.dims[1] != hidden || !PackedCacheRows(index) || !PackedCacheRows(score) ||
        index.dims[0] != rows || index.dims != score.dims || index.dims[1] <= 0 ||
        index.dims[1] > kMaxTopK || index.dataType != INT32 || score.dataType != FLOAT32 ||
        index.dataDevice != CUDA || score.dataDevice != CUDA || !index.cudaData || !score.cudaData) return false;
    cudaStreamCaptureStatus originCapture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &originCapture) != cudaSuccess ||
        originCapture != cudaStreamCaptureStatusNone) return false;
    const int topk = index.dims[1], routes = rows * topk, base = table * layout.experts;
    std::vector<DeviceCache *> caches;
    std::shared_ptr<Glm5MultiGpuVerify> shared;
    {
        std::lock_guard<std::mutex> lock(group.mutex);
        caches.push_back(&origin);
        for (auto &entry : group.deviceCaches)
            if (entry.second && entry.second.get() != &origin && entry.second->ready &&
                entry.second->hostKeyToSlot) caches.push_back(entry.second.get());
        if (caches.size() < 2) return false;
        std::sort(caches.begin() + 1, caches.end(), [](auto *a, auto *b) { return a->device < b->device; });
        if (!group.cooperativeVerify) group.cooperativeVerify = std::make_shared<Glm5MultiGpuVerify>();
        shared = group.cooperativeVerify;
    }
    auto &state = *shared;
    std::lock_guard<std::mutex> lock(state.mutex);
    struct RestoreDevice {
        int device;
        ~RestoreDevice() { cudaSetDevice(device); }
    } restore{origin.device};
    if (state.previousRoot) checkCudaErrors("GLM previous gather", cudaEventSynchronize(state.previousRoot->rank.done));
    std::vector<Glm5MultiGpuVerify::DeviceWork *> work;
    const int timingLayer = (rows - 1) * group.tableKeys.size() + table;
    const int timingLayers = FASTLLM_CUDA_MOE_CACHE_MAX_BATCH * group.tableKeys.size();
    state.cpu.resize(timingLayers); state.calls.resize(timingLayers);
    for (auto *cache : caches) {
        checkCudaErrors("GLM cooperative device", cudaSetDevice(cache->device));
        cudaStreamCaptureStatus capture;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) return false;
        auto &entry = state.devices[cache->device];
        if (!entry) entry = std::make_unique<Glm5MultiGpuVerify::DeviceWork>();
        if (!entry->Prepare(cache->device, hidden)) { (void)cudaGetLastError(); return false; }
        auto &w = entry->rank;
        w.cache = cache; w.group = &group; w.table = table; w.rows = rows; w.topk = topk;
        if (cache->admissionDone) checkCudaErrors("GLM peer residency", cudaEventSynchronize(cache->admissionDone));
        const int capacity = std::min(routes, layout.experts);
        if (!w.overlap || int(w.overlap->expertReady.size()) < capacity) {
            auto next = std::make_unique<DecodeOverlapWorkspace>();
            if (next->Init(layout.recordStride, timingLayers, capacity)) {
                if (w.overlap) next->layers = std::move(w.overlap->layers);
                w.overlap = std::move(next);
            } else (void)cudaGetLastError(); // Resident/CPU execution remains available.
        }
        if (w.overlap) w.overlap->ConfigureLayout(group, table, rows);
        work.push_back(entry.get());
    }
    auto &root = *work[0];
    auto &r = root.rank;
    checkCudaErrors("GLM origin", cudaSetDevice(origin.device));
    checkCudaErrors("GLM routing ids", cudaMemcpyAsync(r.Indices(), index.cudaData,
        routes * sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("GLM routing scores", cudaMemcpyAsync(r.Scores(), score.cudaData,
        routes * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("GLM routing input", cudaMemcpyAsync(r.host, input.cudaData,
        size_t(rows) * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("GLM routing ready", cudaStreamSynchronize(cudaStreamPerThread));
    for (int i = 0; i < routes; ++i) if (r.Indices()[i] < 0 || r.Indices()[i] >= layout.experts) return false;

    std::vector<int> hits(work.size(), 0), gpuRoutes(work.size(), 0);
    std::vector<int> misses, reuse;
    for (auto *device : work) {
        auto &w = device->rank;
        w.missedExperts.clear(); w.stagedSlots.fill(-1);
        std::copy_n(r.Indices(), routes, w.Indices());
        std::copy_n(r.Scores(), routes, w.Scores());
    }
    for (int i = 0; i < routes; ++i) {
        r.Owners()[i] = -1;
        for (size_t d = 0; d < work.size(); ++d) {
            auto &w = work[d]->rank;
            w.Resident()[i] = w.cache->hostKeyToSlot[base + r.Indices()[i]];
            if (r.Owners()[i] < 0 && w.Resident()[i] >= 0) { r.Owners()[i] = d; ++hits[d]; }
        }
        if (r.Owners()[i] >= 0) continue;
        const int expert = r.Indices()[i];
        auto found = std::find(misses.begin(), misses.end(), expert);
        if (found == misses.end()) { misses.push_back(expert); reuse.push_back(1); }
        else ++reuse[found - misses.begin()];
    }
    for (size_t i = 0; i < reuse.size(); ++i) for (size_t j = i + 1; j < reuse.size(); ++j)
        if (reuse[j] > reuse[i]) { std::swap(reuse[i], reuse[j]); std::swap(misses[i], misses[j]); }
    std::vector<Scheduler::SharedRankPlan> plans;
    for (size_t d = 0; d < work.size(); ++d) {
        auto &w = *work[d];
        auto *overlap = w.rank.overlap.get();
        plans.push_back({overlap ? &overlap->layers[timingLayer] : nullptr, hits[d],
            overlap ? int(overlap->expertReady.size()) : 0,
            d ? rows * (w.inputPerRow.us + w.returnPerRow.us + root.mergePerRow.us) : 0});
    }
    const auto owners = Scheduler::AssignSharedMisses(plans, state.cpu[timingLayer], reuse, state.calls[timingLayer]++);
    for (size_t i = 0; i < misses.size(); ++i) if (owners[i] >= 0) {
        auto &w = work[owners[i]]->rank;
        const int slot = w.missedExperts.size();
        w.missedExperts.push_back(misses[i]);
        for (int route = 0; route < routes; ++route) if (r.Indices()[route] == misses[i]) {
            r.Owners()[route] = owners[i]; w.stagedSlots[route] = slot;
        }
    }
    const int cpuRoutes = std::count(r.Owners(), r.Owners() + routes, -1);
    auto allocate = [&](Data &data, DataType type, std::vector<int> dims, int device) {
        data.dataType = type; data.Resize(dims);
        data.ToDevice(CUDA, {device}, false); data.Allocate(false);
    };
    for (size_t d = 0; d < work.size(); ++d) {
        auto &w = work[d]->rank;
        checkCudaErrors("GLM prepare device", cudaSetDevice(w.cudaDevice));
        int at = 0;
        for (int i = 0; i < routes; ++i)
            if (r.Owners()[i] == int(d) && w.stagedSlots[i] < 0) w.BatchRoutes()[at++] = i;
        std::sort(w.BatchRoutes(), w.BatchRoutes() + at, [&](int a, int b) {
            return w.Resident()[a] != w.Resident()[b] ? w.Resident()[a] < w.Resident()[b] : a < b;
        });
        for (int i = 0; i < at; ++i) w.BatchSlots()[i] = w.Resident()[w.BatchRoutes()[i]];
        auto &offsets = work[d]->offsets;
        offsets.assign(1, at);
        for (int e = 0; e < int(w.missedExperts.size()); ++e) {
            for (int i = 0; i < routes; ++i) if (w.stagedSlots[i] == e) {
                w.BatchSlots()[at] = e; w.BatchRoutes()[at++] = i;
            }
            offsets.push_back(at);
        }
        gpuRoutes[d] = at;
        // All allocations precede CPU/GPU dispatch; cudaMalloc can synchronize.
        if (at) {
            allocate(w.batchSlots, INT32, {routes}, w.cudaDevice);
            allocate(w.batchRoutes, INT32, {routes}, w.cudaDevice);
            allocate(w.scores, FLOAT32, {rows, topk}, w.cudaDevice);
            allocate(w.gate, BFLOAT16, {routes, layout.inter}, w.cudaDevice);
            if (d) allocate(work[d]->input, BFLOAT16, {rows, hidden}, w.cudaDevice);
        }
    }
    checkCudaErrors("GLM output device", cudaSetDevice(origin.device));
    allocate(r.ids, INT32, {rows, topk}, origin.device);
    allocate(r.owners, INT32, {rows, topk}, origin.device);
    allocate(root.received, FLOAT32, {routes, hidden}, origin.device);
    allocate(output, BFLOAT16, {rows, hidden}, origin.device);
    // Cache admission still observes the origin layer exactly once. A helper
    // uses existing residents/temporary slots without changing cache placement.
    origin.frequency->Observe(base, r.Indices(), routes);
    if (launchParallel) launchParallel();
    checkCudaErrors("GLM restore origin", cudaSetDevice(origin.device));
    auto submitGpu = [&] {
        for (size_t d = 0; d < work.size(); ++d) {
            auto &w = work[d]->rank;
            auto *overlap = w.overlap.get();
            const int staged = w.missedExperts.size(), count = gpuRoutes[d];
            if (d && !count) continue;
            const double dispatchStart = HybridNowUs();
            checkCudaErrors("GLM submit device", cudaSetDevice(w.cudaDevice));
            if (d && count) {
                checkCudaErrors("GLM input start", cudaEventRecord(work[d]->inputStart, cudaStreamPerThread));
                checkCudaErrors("GLM peer input", cudaMemcpyAsync(work[d]->input.cudaData, r.host,
                    size_t(rows) * hidden * sizeof(uint16_t), cudaMemcpyHostToDevice, cudaStreamPerThread));
                checkCudaErrors("GLM input end", cudaEventRecord(work[d]->inputEnd, cudaStreamPerThread));
            }
            if (count) {
                checkCudaErrors("GLM route slots", cudaMemcpyAsync(w.batchSlots.cudaData, w.BatchSlots(),
                    count * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
                checkCudaErrors("GLM route map", cudaMemcpyAsync(w.batchRoutes.cudaData, w.BatchRoutes(),
                    count * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
                checkCudaErrors("GLM route scores", cudaMemcpyAsync(w.scores.cudaData, w.Scores(),
                    routes * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
            }
            TouchResidentRoutes<<<1, 256, 0, cudaStreamPerThread>>>(
                static_cast<int32_t *>(w.batchSlots.cudaData), static_cast<int32_t *>(w.batchRoutes.cudaData),
                w.cache->lastUsed, w.cache->step, w.cache->hitCount, hits[d], rows, topk,
                w.cache->totalMissCount, count - hits[d] + (d == 0 ? cpuRoutes : 0));
            if (staged) overlap->PrepareCopy(0);
            const auto &activation = d ? work[d]->input : input;
            if (hits[d]) {
                if (overlap) checkCudaErrors("GLM resident start", cudaEventRecord(overlap->residentStart, cudaStreamPerThread));
                AssertInFastLLM(ComputeGlm5Experts(activation, w.gate, layout, w.cache->records,
                    static_cast<int32_t *>(w.batchSlots.cudaData), static_cast<float *>(w.scores.cudaData),
                    topk, w.GpuOutput(), static_cast<int32_t *>(w.batchRoutes.cudaData), hits[d]), "GLM peer resident failed");
                if (overlap) checkCudaErrors("GLM resident done", cudaEventRecord(overlap->residentDone, cudaStreamPerThread));
            }
            for (int e = 0; e < staged; ++e) {
                overlap->CopyExpert(group, table, w.missedExperts[e], e, staged);
                checkCudaErrors("GLM peer expert ready", cudaStreamWaitEvent(cudaStreamPerThread, overlap->expertReady[e], 0));
                checkCudaErrors("GLM staged start", cudaEventRecord(overlap->stagedStart[e], cudaStreamPerThread));
                const int start = work[d]->offsets[e], count = work[d]->offsets[e + 1] - start;
                AssertInFastLLM(ComputeGlm5Experts(activation, w.gate, layout, overlap->records,
                    static_cast<int32_t *>(w.batchSlots.cudaData) + start, static_cast<float *>(w.scores.cudaData),
                    topk, w.GpuOutput(), static_cast<int32_t *>(w.batchRoutes.cudaData) + start, count), "GLM peer staged failed");
                checkCudaErrors("GLM staged done", cudaEventRecord(overlap->stagedDone[e], cudaStreamPerThread));
            }
            if (overlap) {
                if (staged) overlap->layers[timingLayer].dispatch.Observe(HybridNowUs() - dispatchStart);
                overlap->previousLayer = timingLayer; overlap->previousHits = hits[d];
                overlap->previousMisses = staged; overlap->previousStagedRoutes = count - hits[d];
                ++overlap->layers[timingLayer].calls;
            }
            if (d && count) {
                checkCudaErrors("GLM result start", cudaEventRecord(work[d]->returnStart, cudaStreamPerThread));
                checkCudaErrors("GLM peer result", cudaMemcpyAsync(w.CpuOutput(), w.GpuOutput(),
                    size_t(routes) * hidden * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
                checkCudaErrors("GLM result end", cudaEventRecord(work[d]->returnEnd, cudaStreamPerThread));
                work[d]->sent = true;
            }
            checkCudaErrors("GLM GPU submitted", cudaEventRecord(w.done, cudaStreamPerThread));
            w.pending = true; work[d]->previousRows = rows;
        }
        checkCudaErrors("GLM CPU device restore", cudaSetDevice(origin.device));
    };
    const double cpuStart = HybridNowUs();
    if (cpuRoutes) {
        NumasMoeVerifyExpertsWithOverlap(reinterpret_cast<const uint16_t *>(r.host), r.CpuOutput(), rows,
            weights, weightsBatch, r.Indices(), r.Owners(), r.Scores(), topk, layer,
            layout.swigluLimit, true, 128, submitGpu);
        state.cpu[timingLayer].Observe((HybridNowUs() - cpuStart) / cpuRoutes);
        checkCudaErrors("GLM CPU results", cudaMemcpyAsync(r.device, r.CpuOutput(),
            size_t(routes) * hidden * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
    } else submitGpu();
    checkCudaErrors("GLM owners", cudaMemcpyAsync(r.owners.cudaData, r.Owners(),
        routes * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
    checkCudaErrors("GLM ids", cudaMemcpyAsync(r.ids.cudaData, r.Indices(),
        routes * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
    int peers = 0;
    for (size_t d = 1; d < work.size(); ++d) if (gpuRoutes[d]) {
        auto &w = work[d]->rank;
        checkCudaErrors("GLM result ready", cudaEventSynchronize(w.done));
        if (!peers) checkCudaErrors("GLM merge start", cudaEventRecord(root.mergeStart, cudaStreamPerThread));
        checkCudaErrors("GLM gather result", cudaMemcpyAsync(root.received.cudaData, w.CpuOutput(),
            size_t(routes) * hidden * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        CopyOwnedGlm5Routes<<<(routes * hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            r.GpuOutput(), static_cast<float *>(root.received.cudaData), static_cast<int32_t *>(r.owners.cudaData), d, hidden, routes);
        ++peers;
    }
    if (peers) {
        checkCudaErrors("GLM merge end", cudaEventRecord(root.mergeEnd, cudaStreamPerThread));
        root.merged = true; root.previousPeers = peers;
    }
    fastllm::cuda::dsv41_cache::Reduce<<<dim3((hidden + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
        r.device, r.GpuOutput(), static_cast<int32_t *>(r.owners.cudaData), static_cast<int32_t *>(r.ids.cudaData),
        static_cast<__nv_bfloat16 *>(output.cudaData), hidden, topk);
    checkCudaErrors("GLM gathered reduction", cudaGetLastError());
    checkCudaErrors("GLM gather done", cudaEventRecord(r.done, cudaStreamPerThread));
    state.previousRoot = &root;
    for (size_t d = 0; d < work.size(); ++d) {
        auto &stats = caches[d]->cooperativeStats;
        const int cpu = d == 0 ? cpuRoutes : 0, total = gpuRoutes[d] + cpu;
        stats[0] += d == 0; stats[1] += total; stats[2] += hits[d];
        stats[3] += total - hits[d]; stats[4] += gpuRoutes[d]; stats[5] += cpu; stats[6] += hits[d];
    }
    return true;
}
} // namespace
