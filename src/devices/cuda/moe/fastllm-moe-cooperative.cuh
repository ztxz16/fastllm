#pragma once

// Included after the shared cache and batched workspace definitions.
namespace {
bool SupportsCooperativeMoe(const OffloadLayout &layout) {
    return layout.glm5 || (layout.deepSeekV41 && layout.weightType == fastllm::DATA_GGUF_FORMAT);
}

struct CooperativeMoe {
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
                checkCudaErrors("Cooperative MoE input timing", cudaEventElapsedTime(&ms, inputStart, inputEnd));
                inputPerRow.Observe(ms * 1000 / previousRows);
                checkCudaErrors("Cooperative MoE result timing", cudaEventElapsedTime(&ms, returnStart, returnEnd));
                returnPerRow.Observe(ms * 1000 / previousRows);
            }
            if (merged) {
                checkCudaErrors("Cooperative MoE merge timing", cudaEventElapsedTime(&ms, mergeStart, mergeEnd));
                mergePerRow.Observe(ms * 1000 / (previousRows * previousPeers));
            }
            sent = merged = false;
            for (auto *event : {&inputStart, &inputEnd, &returnStart, &returnEnd, &mergeStart, &mergeEnd})
                if (!*event && cudaEventCreate(event) != cudaSuccess) return false;
            return true;
        }
    };
    std::map<int, std::unique_ptr<DeviceWork>> devices;
    std::vector<fastllm::MoeDecodeOverlapScheduler> cpu;
    std::vector<float> cpuInput;
    std::mutex mutex;
    DeviceWork *previousRoot = nullptr;

    ~CooperativeMoe() {
        // A peer's pinned result may still be read by the origin GPU's H2D.
        if (previousRoot) cudaEventSynchronize(previousRoot->rank.done);
    }
};

// Preserve each expert's FP32 result until the existing ordered BF16 reduction.
// Summing GPU-local partial outputs first would change the rounding order.
__global__ void CopyOwnedMoeRoutes(float *destination, const float *source,
        const int32_t *owners, int owner, int hidden, int routes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden * routes && owners[i / hidden] == owner) destination[i] = source[i];
}

bool TryCooperativeMoe(OffloadGroup &group, DeviceCache &origin, int table,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch, int layer,
        const std::function<void()> &launchParallel,
        const std::vector<int> *allowedDevices = nullptr) {
    using namespace fastllm;
    using Scheduler = MoeDecodeOverlapScheduler;
    const auto &layout = group.LayerLayout(table);
    const int rows = input.dims[0], hidden = layout.hidden;
    const bool layerAdmission = layout.deepSeekV41 && layout.weightType == DATA_GGUF_FORMAT &&
        rows > 1 && allowedDevices && allowedDevices->size() > 1;
    if (!SupportsCooperativeMoe(layout) || (!origin.frequencyActive && !layerAdmission) ||
        !origin.hostKeyToSlot || input.dataType != BFLOAT16 ||
        !(NativeSharedRecords(group) || (layout.weightType == DATA_GGUF_FORMAT &&
          group.sharedLayout.shards == 0 && GGUFHybridShape(layout))) ||
        rows < 1 || rows > FASTLLM_CUDA_MOE_CACHE_MAX_BATCH || input.dims[1] != hidden ||
        !PackedCacheRows(index) || !PackedCacheRows(score) ||
        index.dims[0] != rows || index.dims != score.dims || index.dims[1] <= 0 ||
        index.dims[1] > kMaxTopK || index.dataType != INT32 || score.dataType != FLOAT32 ||
        index.dataDevice != CUDA || score.dataDevice != CUDA || !index.cudaData || !score.cudaData) return false;
    cudaStreamCaptureStatus originCapture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &originCapture) != cudaSuccess ||
        originCapture != cudaStreamCaptureStatusNone) return false;
    struct RestoreDevice {
        int device;
        ~RestoreDevice() { cudaSetDevice(device); }
    } restore{origin.device};
    if (allowedDevices) {
        std::vector<int> unprepared;
        {
            std::lock_guard<std::mutex> lock(group.mutex);
            for (int device : *allowedDevices) {
                const auto it = group.deviceCaches.find(device);
                if (it == group.deviceCaches.end() || !it->second || !it->second->attempted)
                    unprepared.push_back(device);
            }
        }
        // Prepare lazily, without extra device switches/dependencies on each
        // layer after warmup. The residency fence below protects existing caches.
        for (int device : unprepared) {
            checkCudaErrors("Cooperative MoE helper device", cudaSetDevice(device));
            cudaStreamCaptureStatus capture;
            if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
                capture != cudaStreamCaptureStatusNone) return false;
            GetDeviceCache(group);
        }
    }
    const int topk = index.dims[1], routes = rows * topk, base = table * layout.experts;
    std::vector<DeviceCache *> caches;
    std::shared_ptr<CooperativeMoe> shared;
    {
        std::lock_guard<std::mutex> lock(group.mutex);
        caches.push_back(&origin);
        for (auto &entry : group.deviceCaches)
            if (entry.second && entry.second.get() != &origin && entry.second->ready &&
                entry.second->hostKeyToSlot && (!allowedDevices ||
                std::find(allowedDevices->begin(), allowedDevices->end(), entry.first) != allowedDevices->end()))
                caches.push_back(entry.second.get());
        // GGUF verification shares the resident/miss scheduler even with one
        // GPU. Keep the existing single-row and compact NVFP4 dispatch paths.
        if (caches.size() < 2 && (rows == 1 || layout.weightType != DATA_GGUF_FORMAT)) return false;
        if (!group.cooperativeHybrid) group.cooperativeHybrid = std::make_shared<CooperativeMoe>();
        shared = group.cooperativeHybrid;
    }
    auto &state = *shared;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.previousRoot) checkCudaErrors("Cooperative MoE previous gather", cudaEventSynchronize(state.previousRoot->rank.done));
    std::vector<CooperativeMoe::DeviceWork *> work;
    const int timingLayer = (rows - 1) * group.tableKeys.size() + table;
    const int timingLayers = FASTLLM_CUDA_MOE_CACHE_MAX_BATCH * group.tableKeys.size();
    state.cpu.resize(timingLayers);
    for (auto *cache : caches) {
        checkCudaErrors("Cooperative MoE device", cudaSetDevice(cache->device));
        cudaStreamCaptureStatus capture;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) return false;
        auto &entry = state.devices[cache->device];
        if (!entry) entry = std::make_unique<CooperativeMoe::DeviceWork>();
        if (!entry->Prepare(cache->device, hidden)) { (void)cudaGetLastError(); return false; }
        auto &w = entry->rank;
        w.admitted = false;
        w.cache = cache; w.group = &group; w.table = table; w.rows = rows; w.topk = topk;
        if (cache->admissionDone) checkCudaErrors("Cooperative MoE peer residency", cudaEventSynchronize(cache->admissionDone));
        const int capacity = std::min(routes, layout.experts);
        if (!w.overlap || int(w.overlap->expertReady.size()) < capacity) {
            auto next = std::make_unique<DecodeOverlapWorkspace>();
            size_t stride = layout.recordStride;
            for (const auto &l : group.layerLayouts) stride = std::max(stride, l.recordStride);
            if (next->Init(stride, timingLayers, capacity, !group.ggufSources.empty())) {
                if (w.overlap) next->layers = std::move(w.overlap->layers);
                w.overlap = std::move(next);
            } else (void)cudaGetLastError(); // Resident/CPU execution remains available.
        }
        if (w.overlap) w.overlap->ConfigureLayout(group, table, rows);
        work.push_back(entry.get());
    }
    auto &root = *work[0];
    auto &r = root.rank;
    checkCudaErrors("Cooperative MoE origin", cudaSetDevice(origin.device));
    checkCudaErrors("Cooperative MoE routing ids", cudaMemcpyAsync(r.Indices(), index.cudaData,
        routes * sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("Cooperative MoE routing scores", cudaMemcpyAsync(r.Scores(), score.cudaData,
        routes * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("Cooperative MoE routing input", cudaMemcpyAsync(r.host, input.cudaData,
        size_t(rows) * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("Cooperative MoE routing ready", cudaStreamSynchronize(cudaStreamPerThread));
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
            d ? rows * ((rows == 1 && overlap &&
                overlap->layers[timingLayer].decodeLaunch.initialized ? 0 : w.inputPerRow.us) +
                w.returnPerRow.us + root.mergePerRow.us) : 0});
    }
    const auto owners = Scheduler::AssignSharedMisses(plans, state.cpu[timingLayer].cpuExpert, reuse,
        state.cpu[timingLayer].calls++, rows == 1 ? &state.cpu[timingLayer] : nullptr);
    for (size_t i = 0; i < misses.size(); ++i) if (owners[i] >= 0) {
        auto &w = work[owners[i]]->rank;
        const int slot = w.missedExperts.size();
        w.missedExperts.push_back(misses[i]);
        for (int route = 0; route < routes; ++route) if (r.Indices()[route] == misses[i]) {
            r.Owners()[route] = owners[i]; w.stagedSlots[route] = slot;
        }
    }
    const int cpuRoutes = std::count(r.Owners(), r.Owners() + routes, -1);
    // Preserve V4.1 verify's bounded admission policy: one reused miss per
    // layer, on its permanent PP owner. Enabling another execution device
    // must not turn a small verify into a global cache-refill burst.
    int candidate = -1;
    if (layerAdmission && origin.slots > 0) {
        if (r.heat.size() != group.totalRecords) r.heat.assign(group.totalRecords, 0);
        unsigned bestHeat = 0;
        for (int i = 0; i < routes; ++i) {
            auto &heat = r.heat[base + r.Indices()[i]];
            heat = std::min(heat + 1, 65535u);
            if (r.Resident()[i] < 0 && heat >= 2 && heat > bestHeat) {
                bestHeat = heat; candidate = i;
            }
        }
        int interval = 31;
        while (std::gcd(interval, int(group.tableKeys.size())) != 1) ++interval;
        const double cpuBudget = cpuRoutes * state.cpu[timingLayer].cpuExpert.us;
        if (r.refill.initialized && cpuBudget > 0 && r.refill.us > cpuBudget &&
            r.calls % interval != 0) candidate = -1;
    }
    auto allocate = [&](Data &data, DataType type, std::vector<int> dims, int device) {
        data.dataType = type; data.Resize(dims);
        data.ToDevice(CUDA, {device}, false); data.Allocate(false);
    };
    for (size_t d = 0; d < work.size(); ++d) {
        auto &w = work[d]->rank;
        checkCudaErrors("Cooperative MoE prepare device", cudaSetDevice(w.cudaDevice));
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
            if (layout.weightType == DATA_GGUF_FORMAT)
                allocate(w.batchWorkspace, INT8,
                    {int(layout.deepSeekV41 ?
                        FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(hidden, layout.inter) :
                        size_t(rows) * (hidden + topk * layout.inter) * sizeof(float))}, w.cudaDevice);
            if (d) allocate(work[d]->input, BFLOAT16, {rows, hidden}, w.cudaDevice);
        }
    }
    checkCudaErrors("Cooperative MoE output device", cudaSetDevice(origin.device));
    allocate(r.ids, INT32, {rows, topk}, origin.device);
    allocate(r.owners, INT32, {rows, topk}, origin.device);
    if (candidate >= 0) allocate(r.lookup, INT32, {2 * FastllmCudaMoeExpertParallel::maxRoutes}, origin.device);
    if (work.size() > 1) allocate(root.received, FLOAT32, {routes, hidden}, origin.device);
    allocate(output, BFLOAT16, {rows, hidden}, origin.device);
    // Only the expert's permanent owner updates its frequency. Temporary
    // miss offload may still choose any device independently of cache placement.
    for (auto *device : work) {
        auto &cache = *device->rank.cache;
        if (layerAdmission || !cache.frequencyActive || !cache.frequency) continue;
        if (group.expertDevices.empty() && &cache != &origin) continue;
        std::array<int, kMaxTopK> owned;
        for (int row = 0; row < rows; ++row) {
            for (int k = 0; k < topk; ++k) {
                const int e = r.Indices()[row * topk + k];
                owned[k] = group.expertDevices.empty() ||
                    e % cache.prefillOwnerCount == cache.prefillOwnerRank ? e : -1;
            }
            cache.frequency->Observe(base, owned.data(), topk);
        }
    }
    auto launchShared = [&] {
        checkCudaErrors("Cooperative MoE shared origin", cudaSetDevice(origin.device));
        if (launchParallel) launchParallel();
        checkCudaErrors("Cooperative MoE restore origin", cudaSetDevice(origin.device));
    };
    // Decode reports worker-only CPU timing and can overlap shared work.
    // Keep verify's existing ordering outside its wall-clock CPU estimate.
    if (rows > 1) launchShared();
    auto submitGpu = [&] {
        const double submitStart = HybridNowUs();
        // Start every link before submitting projection kernels. Interleaving
        // a whole device's kernels before the next device's DMA delays helpers
        // and prevents short decode projections from overlapping.
        for (size_t d = 0; d < work.size(); ++d) {
            auto &w = work[d]->rank;
            auto *overlap = w.overlap.get();
            const int staged = w.missedExperts.size(), count = gpuRoutes[d];
            if (d && !count) continue;
            checkCudaErrors("Cooperative MoE submit device", cudaSetDevice(w.cudaDevice));
            if (staged && rows == 1) {
                checkCudaErrors("Cooperative MoE launch timing", cudaEventRecord(overlap->decodeSubmit, cudaStreamPerThread));
                overlap->decodeSubmitHostUs = HybridNowUs() - submitStart;
            }
            if (d && count) {
                checkCudaErrors("Cooperative MoE input start", cudaEventRecord(work[d]->inputStart, cudaStreamPerThread));
                checkCudaErrors("Cooperative MoE peer input", cudaMemcpyAsync(work[d]->input.cudaData, r.host,
                    size_t(rows) * hidden * sizeof(uint16_t), cudaMemcpyHostToDevice, cudaStreamPerThread));
                checkCudaErrors("Cooperative MoE input end", cudaEventRecord(work[d]->inputEnd, cudaStreamPerThread));
            }
            if (count) {
                checkCudaErrors("Cooperative MoE route slots", cudaMemcpyAsync(w.batchSlots.cudaData, w.BatchSlots(),
                    count * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
                checkCudaErrors("Cooperative MoE route map", cudaMemcpyAsync(w.batchRoutes.cudaData, w.BatchRoutes(),
                    count * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
                checkCudaErrors("Cooperative MoE route scores", cudaMemcpyAsync(w.scores.cudaData, w.Scores(),
                    routes * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
            }
            TouchResidentRoutes<<<1, 256, 0, cudaStreamPerThread>>>(
                static_cast<int32_t *>(w.batchSlots.cudaData), static_cast<int32_t *>(w.batchRoutes.cudaData),
                w.cache->lastUsed, w.cache->step, w.cache->hitCount, hits[d], rows, topk,
                w.cache->totalMissCount, count - hits[d] + (d == 0 ? cpuRoutes : 0));
            if (staged) overlap->CopyExperts(group, table, w.missedExperts.data(), staged, 0);
        }
        if (rows == 1) launchShared();
        for (size_t d = 0; d < work.size(); ++d) {
            auto &w = work[d]->rank;
            auto *overlap = w.overlap.get();
            const int staged = w.missedExperts.size(), count = gpuRoutes[d];
            if (d && !count) continue;
            checkCudaErrors("Cooperative MoE compute device", cudaSetDevice(w.cudaDevice));
            const auto &activation = d ? work[d]->input : input;
            bool inputPrepared = false;
            auto compute = [&](const uint8_t *records, int start, int count, bool resident) {
                auto *slots = static_cast<int32_t *>(w.batchSlots.cudaData) + start;
                auto *map = static_cast<int32_t *>(w.batchRoutes.cudaData) + start;
                if (layout.weightType == NVFP4_BLOCK_16_E4M3_PACKED ||
                    layout.weightType == FP8_E4M3_BLOCK_128)
                    return ComputeGlm5Experts(activation, w.gate, layout, records, slots,
                        static_cast<float *>(w.scores.cudaData), topk, w.GpuOutput(), map, count);
                FastllmCudaMoeGGUFCacheView view{records, slots, layout.recordStride, layout.downOffset,
                    layout.gateGgmlType, layout.downGgmlType, hidden, layout.inter,
                    w.batchWorkspace.cudaData, size_t(w.batchWorkspace.GetBytes()),
                    resident ? w.cache->slotOffsets : nullptr};
                view.routeMap = map; view.routeCount = count; view.q8InputPrepared = inputPrepared;
                if (!resident && overlap) overlap->SetLayout(view);
                if (resident && NativeGlmGGUFRecords(group)) {
                    view.numaGateType = group.ggufSources[table].weights[0].type;
                    view.numaDownType = group.ggufSources[table].weights[1].type;
                }
                // Preserve the full input layout and sparse route map: all
                // rows sharing a staged expert use one upload and one batch,
                // and resident/staged batches reuse the input quantization.
                const bool ok = ComputeGGUFExperts(activation, w.gate, output, layout, view,
                    static_cast<float *>(w.scores.cudaData), topk, w.GpuOutput());
                inputPrepared = true;
                return ok;
            };
            if (hits[d]) {
                if (overlap) checkCudaErrors("Cooperative MoE resident start", cudaEventRecord(overlap->residentStart, cudaStreamPerThread));
                AssertInFastLLM(compute(w.cache->records, 0, hits[d], true), "Cooperative MoE peer resident failed");
                if (overlap) checkCudaErrors("Cooperative MoE resident done", cudaEventRecord(overlap->residentDone, cudaStreamPerThread));
            }
            for (int e = 0; e < staged; ++e) {
                checkCudaErrors("Cooperative MoE peer expert ready", cudaStreamWaitEvent(cudaStreamPerThread, overlap->expertReady[e], 0));
                checkCudaErrors("Cooperative MoE staged start", cudaEventRecord(overlap->stagedStart[e], cudaStreamPerThread));
                const int start = work[d]->offsets[e], count = work[d]->offsets[e + 1] - start;
                AssertInFastLLM(compute(overlap->records, start, count, false), "Cooperative MoE peer staged failed");
                checkCudaErrors("Cooperative MoE staged done", cudaEventRecord(overlap->stagedDone[e], cudaStreamPerThread));
            }
            if (overlap) {
                if (staged) overlap->layers[timingLayer].dispatch.Observe(HybridNowUs() - submitStart);
                overlap->previousLayer = timingLayer; overlap->previousHits = hits[d];
                overlap->previousMisses = staged; overlap->previousStagedRoutes = count - hits[d];
                ++overlap->layers[timingLayer].calls;
            }
            if (d && count) {
                checkCudaErrors("Cooperative MoE result start", cudaEventRecord(work[d]->returnStart, cudaStreamPerThread));
                checkCudaErrors("Cooperative MoE peer result", cudaMemcpyAsync(w.CpuOutput(), w.GpuOutput(),
                    size_t(routes) * hidden * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
                checkCudaErrors("Cooperative MoE result end", cudaEventRecord(work[d]->returnEnd, cudaStreamPerThread));
                work[d]->sent = true;
            }
            checkCudaErrors("Cooperative MoE GPU submitted", cudaEventRecord(w.done, cudaStreamPerThread));
            w.pending = true; work[d]->previousRows = rows;
        }
        checkCudaErrors("Cooperative MoE CPU device restore", cudaSetDevice(origin.device));
    };
    const double cpuStart = HybridNowUs();
    if (cpuRoutes) {
        if (rows == 1) {
            state.cpuInput.resize(hidden);
            const auto *bf = reinterpret_cast<const uint16_t *>(r.host);
            for (int c = 0; c < hidden; ++c) state.cpuInput[c] = BFloat16BitsToFloat32(bf[c]);
            double elapsed = 0;
            NumasMoeDecodeExpertsWithOverlap(state.cpuInput.data(), r.CpuOutput(), weights,
                r.Indices(), r.Owners(), topk, layer, submitGpu, r.Scores(), layout.swigluLimit,
                layout.deepSeekV41 ? 32 : 128, &elapsed);
            state.cpu[timingLayer].ObserveDecodeCpu(cpuRoutes, elapsed);
        } else {
            double elapsed = 0;
            NumasMoeVerifyExpertsWithOverlap(reinterpret_cast<const uint16_t *>(r.host), r.CpuOutput(), rows,
                weights, weightsBatch, r.Indices(), r.Owners(), r.Scores(), topk, layer,
                layout.swigluLimit, true, layout.deepSeekV41 ? 32 : 128,
                submitGpu, &elapsed, group.cpuDecodeReady);
            state.cpu[timingLayer].cpuExpert.Observe(
                (elapsed > 0 ? elapsed : HybridNowUs() - cpuStart) / cpuRoutes);
        }
        checkCudaErrors("Cooperative MoE CPU results", cudaMemcpyAsync(r.device, r.CpuOutput(),
            size_t(routes) * hidden * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
    } else submitGpu();
    checkCudaErrors("Cooperative MoE owners", cudaMemcpyAsync(r.owners.cudaData, r.Owners(),
        routes * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
    checkCudaErrors("Cooperative MoE ids", cudaMemcpyAsync(r.ids.cudaData, r.Indices(),
        routes * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
    int peers = 0;
    for (size_t d = 1; d < work.size(); ++d) if (gpuRoutes[d]) {
        auto &w = work[d]->rank;
        checkCudaErrors("Cooperative MoE result ready", cudaEventSynchronize(w.done));
        if (!peers) checkCudaErrors("Cooperative MoE merge start", cudaEventRecord(root.mergeStart, cudaStreamPerThread));
        checkCudaErrors("Cooperative MoE gather result", cudaMemcpyAsync(root.received.cudaData, w.CpuOutput(),
            size_t(routes) * hidden * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        CopyOwnedMoeRoutes<<<(routes * hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            r.GpuOutput(), static_cast<float *>(root.received.cudaData), static_cast<int32_t *>(r.owners.cudaData), d, hidden, routes);
        ++peers;
    }
    if (peers) {
        checkCudaErrors("Cooperative MoE merge end", cudaEventRecord(root.mergeEnd, cudaStreamPerThread));
        root.merged = true; root.previousPeers = peers;
    }
    fastllm::cuda::dsv41_cache::Reduce<<<dim3((hidden + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
        r.device, r.GpuOutput(), static_cast<int32_t *>(r.owners.cudaData), static_cast<int32_t *>(r.ids.cudaData),
        static_cast<__nv_bfloat16 *>(output.cudaData), hidden, topk);
    checkCudaErrors("Cooperative MoE gathered reduction", cudaGetLastError());
    checkCudaErrors("Cooperative MoE gather done", cudaEventRecord(r.done, cudaStreamPerThread));
    r.admitted = candidate >= 0 && !misses.empty();
    if (r.admitted) {
        if (!origin.admissionStream) checkCudaErrors("Cooperative MoE admission stream",
            cudaStreamCreateWithFlags(&origin.admissionStream, cudaStreamNonBlocking));
        const auto stream = origin.admissionStream;
        checkCudaErrors("Cooperative MoE admission readers", cudaStreamWaitEvent(stream, r.done, 0));
        checkCudaErrors("Cooperative MoE admission start", cudaEventRecord(r.copyStart, stream));
        BuildHybridPrefetchRoutes<<<1, 32, 0, stream>>>(
            static_cast<int32_t *>(r.ids.cudaData) + (candidate / topk) * topk,
            origin.keyToSlot, origin.slotKeys, static_cast<int32_t *>(r.lookup.cudaData),
            base, layout.experts, topk, candidate % topk);
        const int staged = r.stagedSlots[candidate];
        const uint8_t *record = staged >= 0
            ? r.overlap->records + size_t(staged) * layout.recordStride : nullptr;
        AssertInFastLLM(EnsureCachedExperts(&group, &origin, table,
            static_cast<int32_t *>(r.lookup.cudaData), topk, stream, record,
            r.Indices()[candidate]), "Cooperative MoE admission failed");
        checkCudaErrors("Cooperative MoE admission done", cudaEventRecord(r.copyEnd, stream));
        checkCudaErrors("Cooperative MoE admission fence", cudaEventRecord(origin.admissionDone, stream));
    }
    ++r.calls;
    state.previousRoot = &root;
    for (size_t d = 0; d < work.size(); ++d) {
        auto &stats = caches[d]->cooperativeStats;
        const int cpu = d == 0 ? cpuRoutes : 0, total = gpuRoutes[d] + cpu;
        stats[0] += d == 0; stats[1] += total; stats[2] += hits[d];
        stats[3] += total - hits[d]; stats[4] += gpuRoutes[d]; stats[5] += cpu; stats[6] += hits[d];
        if (d == 0) stats[7] += r.admitted;
    }
    return true;
}
} // namespace
