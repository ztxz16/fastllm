#pragma once

// GLM TP2 expert cache: paired output-row shards with shared CPU fallback.
namespace {
struct Glm5TensorParallel {
    static constexpr int maxRows = fastllm::kNumasMoePrefetchMaxRows;
    static constexpr int maxRoutes = maxRows * kMaxTopK;
    struct Rank {
        fastllm::Data input, activation, scratch, result, slots, map, scores, ids, owners, cpu;
        uint8_t *records = nullptr;
        uint64_t *offsets = nullptr;
        cudaEvent_t gate = nullptr, down = nullptr;
    } rank[2];
    DeviceCache scope;
    OffloadGroup &group;
    fastllm::cuda::CacheSlotPlan plan;
    std::unique_ptr<fastllm::MoeFrequencyPolicy> policy;
    std::vector<fastllm::MoeDecodeOverlapScheduler> timing;
    std::vector<float> cpuInput;
    float *host = nullptr;
    void **pointers = nullptr;
    uint8_t *canonical = nullptr;
    cudaStream_t copy = nullptr;
    cudaEvent_t ready = nullptr, copyStart = nullptr, copied = nullptr, computeStart = nullptr, done = nullptr;
    int references = 0, previousTable = -1, previousGpu = 0, previousStaged = 0;
    size_t maxStride = 0;
    bool pending = false;
    uint64_t calls = 0, residentRoutes = 0, streamedRoutes = 0, cpuRoutes = 0, admissions = 0;

    static void Allocate(fastllm::Data &data, fastllm::DataType type, std::vector<int> shape, int device) {
        data.dataType = type; data.Resize(shape); data.ToDevice(fastllm::CUDA, {device}, false); data.Allocate(false);
    }
    int *Ids() { return reinterpret_cast<int *>(host + (maxRoutes + maxRows) * 4096); }
    int *Owners() { return Ids() + maxRoutes; }
    int *Slots() { return Owners() + maxRoutes; }
    int *Map() { return Slots() + maxRoutes; }
    float *Scores() { return reinterpret_cast<float *>(Map() + maxRoutes); }
    float *CpuOutput() { return host + maxRows * 4096; }

    explicit Glm5TensorParallel(OffloadGroup &owner) : group(owner) {
        using namespace fastllm;
        int original = 0; cudaGetDevice(&original);
        // Preparation has already checked the weight layouts and SM75 peers.
        std::vector<size_t> strides;
        for (const auto &l : group.layerLayouts) {
            strides.push_back(l.gateBytes / 2 + l.downBytes / 2);
            maxStride = std::max(maxStride, strides.back());
        }
        uint64_t budget = UINT64_MAX;
        for (int d = 0; d < 2; ++d) {
            checkCudaErrors("TP device", cudaSetDevice(d));
            const auto enabled = cudaDeviceEnablePeerAccess(1-d, 0);
            if (enabled == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
            else checkCudaErrors("TP enable peer", enabled);
            size_t free = 0, total = 0; checkCudaErrors("TP memory", cudaMemGetInfo(&free, &total));
            const size_t reserve = DeviceMemoryReserveBytes(total) + 64 * maxStride;
            budget = std::min(budget, std::min(DeviceCacheBudgetBytes(d), uint64_t(free > reserve ? free - reserve : 0)));
        }
        plan = cuda::PlanCacheSlots(strides, group.layout.experts, budget, kMaxTopK);
        AssertInFastLLM(plan.offsets.size() >= kMaxTopK, "Insufficient cache budget for MoE TP.");
        const size_t residentSlots = plan.offsets.size();
        std::vector<int> keyParts(group.totalRecords), slotParts(residentSlots);
        std::vector<uint64_t> bytes(group.totalRecords);
        std::map<int, int> partitions;
        for (int t = 0; t < int(strides.size()); ++t) {
            const auto span = plan.layers[t];
            const int p = partitions.emplace(span.begin, partitions.size()).first->second;
            std::fill_n(keyParts.begin() + t * group.layout.experts, group.layout.experts, p);
            std::fill_n(bytes.begin() + t * group.layout.experts, group.layout.experts, 2 * strides[t]);
            std::fill_n(slotParts.begin() + span.begin, span.count, p);
        }
        policy = std::make_unique<MoeFrequencyPolicy>(keyParts, slotParts, strides.size(), GetMoeCacheConfig(), bytes);
        timing.resize(strides.size() * maxRows); cpuInput.resize(4096);
        std::vector<uint64_t> offsets = plan.offsets;
        for (int e = 0; e < kMaxTopK; ++e) offsets.push_back(plan.bytes + e * maxStride);
        for (int d = 0; d < 2; ++d) {
            checkCudaErrors("TP allocation device", cudaSetDevice(d));
            auto &r = rank[d];
            checkCudaErrors("TP records", cudaMalloc(&r.records, plan.bytes + kMaxTopK * maxStride));
            checkCudaErrors("TP offsets", cudaMalloc(&r.offsets, offsets.size() * sizeof(uint64_t)));
            checkCudaErrors("TP offsets upload", cudaMemcpy(r.offsets, offsets.data(), offsets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
            Allocate(r.input, BFLOAT16, {maxRows,4096}, d);
            Allocate(r.activation, BFLOAT16, {maxRoutes,2048}, d);
            Allocate(r.scratch, FLOAT32, {maxRoutes,2048}, d);
            Allocate(r.result, FLOAT32, {maxRoutes,4096}, d);
            Allocate(r.cpu, FLOAT32, {maxRoutes,4096}, d);
            for (auto *meta : {&r.slots, &r.map, &r.ids, &r.owners}) Allocate(*meta, INT32, {maxRoutes}, d);
            Allocate(r.scores, FLOAT32, {maxRoutes}, d);
            checkCudaErrors("TP gate event", cudaEventCreateWithFlags(&r.gate, cudaEventDisableTiming));
            checkCudaErrors("TP down event", cudaEventCreate(&r.down));
        }
        checkCudaErrors("TP allocation root", cudaSetDevice(0));
        checkCudaErrors("TP host scratch", cudaMallocHost(&host, (size_t(maxRoutes+maxRows)*4096 + 5*maxRoutes) * sizeof(float)));
        checkCudaErrors("TP canonical scratch", cudaMalloc(&canonical, 2 * maxStride + 256));
        checkCudaErrors("TP NUMA pointers", cudaMalloc(&pointers, group.numaPointers.size() * sizeof(void *)));
        checkCudaErrors("TP NUMA pointers upload", cudaMemcpy(pointers, group.numaPointers.data(), group.numaPointers.size() * sizeof(void *), cudaMemcpyHostToDevice));
        checkCudaErrors("TP copy stream", cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        for (auto *e : {&ready, &copyStart, &copied, &computeStart, &done}) checkCudaErrors("TP event", cudaEventCreate(e));
        scope.tensorParallelOwner = &group;
        std::fprintf(stderr, "[GLM TP] expert cache per GPU: %zu bytes, %zu paired slots\n",
            size_t(plan.bytes), residentSlots);
        checkCudaErrors("TP allocation restore", cudaSetDevice(original));
    }

    ~Glm5TensorParallel() {
        int original = 0; cudaGetDevice(&original);
        if (pending) cudaEventSynchronize(done);
        cudaSetDevice(0); if (copy) cudaStreamSynchronize(copy);
        for (int d = 0; d < 2; ++d) {
            cudaSetDevice(d); auto &r = rank[d];
            cudaFree(r.records); cudaFree(r.offsets);
            cudaEventDestroy(r.gate); cudaEventDestroy(r.down);
        }
        cudaSetDevice(0); cudaFree(canonical); cudaFree(pointers); cudaFreeHost(host);
        for (auto e : {ready, copyStart, copied, computeStart, done}) cudaEventDestroy(e);
        if (copy) cudaStreamDestroy(copy);
        cudaSetDevice(original);
    }

    void Observe() {
        if (!pending) return;
        checkCudaErrors("TP previous gather", cudaEventSynchronize(done));
        if (previousGpu) {
            float ms = 0;
            checkCudaErrors("TP compute timing", cudaEventElapsedTime(&ms, computeStart, rank[0].down));
            auto &t = timing[previousTable];
            if (!previousStaged) t.residentExpert.Observe(ms * 1000 / previousGpu);
            else {
                t.stagedExpert.Observe(ms * 1000 / previousGpu);
                checkCudaErrors("TP copy timing", cudaEventElapsedTime(&ms, copyStart, copied));
                t.copiedExpert.Observe(ms * 1000 / previousStaged);
            }
        }
        pending = false;
    }

    void CopyExpert(int table, int expert, int slot) {
        const auto &l = group.LayerLayout(table);
        const auto &source = group.ggufSources[table];
        checkCudaErrors("TP canonical restore", fastllm_gguf_restore::CopyRecords(source,
            pointers + size_t(table) * l.experts * 2 * source.shards, canonical, copy,
            nullptr, nullptr, nullptr, nullptr, expert));
        const size_t offset = slot < int(plan.offsets.size()) ? plan.offsets[slot] :
            plan.bytes + size_t(slot - plan.offsets.size()) * maxStride;
        const size_t gateQuarter = l.gateBytes / 4, downHalf = l.downBytes / 2;
        for (int d = 0; d < 2; ++d) {
            auto *target = rank[d].records + offset;
            for (int part = 0; part < 2; ++part)
                checkCudaErrors("TP gate shard", cudaMemcpyAsync(target + part * gateQuarter,
                    canonical + part * l.gateBytes / 2 + d * gateQuarter, gateQuarter, cudaMemcpyDefault, copy));
            checkCudaErrors("TP down shard", cudaMemcpyAsync(target + l.gateBytes / 2,
                canonical + l.downOffset + d * downHalf, downHalf, cudaMemcpyDefault, copy));
        }
    }
};

void *BeginGlm5Tp(OffloadGroup &group) {
    if (!group.tensorParallel) group.tensorParallel = std::make_shared<Glm5TensorParallel>(group);
    auto &s = *group.tensorParallel;
    if (s.references++ == 0) s.policy->BeginStep();
    return &s.scope;
}

void EndGlm5Tp(OffloadGroup &group) {
    auto &s = *group.tensorParallel;
    if (--s.references) return;
    const auto plan = s.policy->EndStep();
    if (plan.empty()) return;
    int original = 0; cudaGetDevice(&original);
    checkCudaErrors("TP admission device", cudaSetDevice(0));
    if (s.pending) checkCudaErrors("TP admission wait", cudaStreamWaitEvent(s.copy, s.done, 0));
    for (const auto &a : plan) s.CopyExpert(a.key / group.layout.experts, a.key % group.layout.experts, a.slot);
    checkCudaErrors("TP admissions ready", cudaEventRecord(s.ready, s.copy));
    s.admissions += plan.size();
    checkCudaErrors("TP admission restore", cudaSetDevice(original));
}

bool RunGlm5Tp(OffloadGroup &group, int table, const fastllm::Data &input,
        const fastllm::Data &index, const fastllm::Data &score, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, int layer, const std::function<void()> &launchParallel) {
    using namespace fastllm;
    const int rows = input.dims.size() == 2 ? input.dims[0] : 0;
    if (rows < 1 || rows > Glm5TensorParallel::maxRows || input.dims[1] != 4096 ||
        input.dataType != BFLOAT16 || input.dataDevice != CUDA || !input.cudaData ||
        !group.tensorParallel) return false;
    auto &s = *group.tensorParallel;
    AssertInFastLLM(s.references > 0, "MoE TP decode scope missing.");
    const int topk = index.dims.size() == 2 ? index.dims.back() : 0;
    const int origin = FastllmCudaGetDevice();
    const int routes = rows * topk;
    AssertInFastLLM(topk > 0 && topk <= kMaxTopK && index.dims == std::vector<int>({rows,topk}) &&
        score.dims == index.dims && index.dataType == INT32 && score.dataType == FLOAT32 &&
        index.cudaData && score.cudaData,
        "Invalid MoE TP routes.");
    s.Observe();
    const auto &l = group.LayerLayout(table);
    checkCudaErrors("TP IDs", cudaMemcpyAsync(s.Ids(), index.cudaData, routes*sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("TP scores", cudaMemcpyAsync(s.Scores(), score.cudaData, routes*sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("TP input", cudaMemcpyAsync(s.host, input.cudaData, size_t(rows)*4096*2, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("TP routing ready", cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<int> missed;
    int count = 0, hits = 0;
    for (int r = 0; r < routes; ++r) {
        AssertInFastLLM(s.Ids()[r] >= 0 && s.Ids()[r] < l.experts, "Invalid TP expert ID.");
        s.Owners()[r] = -1;
        const int slot = s.policy->Slot(table*l.experts + s.Ids()[r]);
        if (slot >= 0) { s.Slots()[count] = slot; s.Map()[count++] = r; s.Owners()[r] = s.Ids()[r]; ++hits; }
        else missed.push_back(r);
    }
    const int timingIndex = table * Glm5TensorParallel::maxRows + rows - 1;
    auto &timing = s.timing[timingIndex];
    // Keep the existing bounded streaming reserve; extra misses stay on CPU.
    const int staged = std::min(kMaxTopK, timing.SelectDecodeMisses(missed.size(), hits));
    for (int i = 0; i < staged; ++i) {
        const int r = missed[i];
        s.Slots()[count] = s.plan.offsets.size() + i; s.Map()[count++] = r; s.Owners()[r] = s.Ids()[r];
    }
    s.policy->Observe(table*l.experts, s.Ids(), routes);
    Glm5TensorParallel::Allocate(output, BFLOAT16, {rows,4096}, origin);
    auto launchStage = [&](int device, bool down) {
        auto &r = s.rank[device];
        FastllmCudaMoeGGUFCacheView view{r.records, static_cast<int32_t *>(r.slots.cudaData),
            l.gateBytes/2+l.downBytes/2, l.gateBytes/2, l.gateGgmlType, l.downGgmlType,4096,2048,
            r.scratch.cudaData,size_t(r.scratch.GetBytes()),r.offsets};
        view.routeMap = static_cast<int32_t *>(r.map.cudaData); view.routeCount = count;
        AssertInFastLLM(FastllmCudaMoeGlm5GGUFTpStage(down,device,r.input,r.activation,view,
            static_cast<float *>(r.scores.cudaData),topk,l.swigluLimit,static_cast<float *>(r.result.cudaData)),
            down ? "TP down rejected." : "TP gate rejected.");
    };
    auto submit = [&] {
        const double start = HybridNowUs();
        if (launchParallel) launchParallel();
        checkCudaErrors("TP submit root", cudaSetDevice(0));
        if (staged) {
            checkCudaErrors("TP staged start", cudaEventRecord(s.copyStart, s.copy));
            for (int i = 0; i < staged; ++i) s.CopyExpert(table, s.Ids()[missed[i]], s.plan.offsets.size()+i);
            checkCudaErrors("TP staged timing", cudaEventRecord(s.copied, s.copy));
            checkCudaErrors("TP staged ready", cudaEventRecord(s.ready, s.copy));
        }
        for (int d = 0; d < 2 && count; ++d) {
            checkCudaErrors("TP gate device", cudaSetDevice(d)); auto &r = s.rank[d];
            r.input.Resize({rows,4096});
            checkCudaErrors("TP admission fence", cudaStreamWaitEvent(cudaStreamPerThread, s.ready, 0));
            checkCudaErrors("TP activation broadcast", cudaMemcpyAsync(r.input.cudaData, input.cudaData, size_t(rows)*4096*2, cudaMemcpyDefault, cudaStreamPerThread));
            checkCudaErrors("TP slots upload", cudaMemcpyAsync(r.slots.cudaData, s.Slots(), count*4, cudaMemcpyHostToDevice, cudaStreamPerThread));
            checkCudaErrors("TP map upload", cudaMemcpyAsync(r.map.cudaData, s.Map(), count*4, cudaMemcpyHostToDevice, cudaStreamPerThread));
            checkCudaErrors("TP scores upload", cudaMemcpyAsync(r.scores.cudaData, s.Scores(), routes*4, cudaMemcpyHostToDevice, cudaStreamPerThread));
            if (d == 0) checkCudaErrors("TP compute start", cudaEventRecord(s.computeStart, cudaStreamPerThread));
            launchStage(d, false);
            checkCudaErrors("TP gate ready", cudaEventRecord(r.gate, cudaStreamPerThread));
        }
        for (int d = 0; d < 2 && count; ++d) {
            checkCudaErrors("TP down device", cudaSetDevice(d)); auto &r = s.rank[d]; auto &peer = s.rank[1-d];
            checkCudaErrors("TP activation fence", cudaStreamWaitEvent(cudaStreamPerThread, peer.gate, 0));
            checkCudaErrors("TP activation allgather", cudaMemcpy2DAsync(
                static_cast<uint8_t *>(r.activation.cudaData)+(1-d)*2048,4096,
                static_cast<uint8_t *>(peer.activation.cudaData)+(1-d)*2048,4096,2048,count,cudaMemcpyDefault,cudaStreamPerThread));
            launchStage(d, true);
            checkCudaErrors("TP down ready", cudaEventRecord(r.down, cudaStreamPerThread));
        }
        checkCudaErrors("TP restore output device", cudaSetDevice(origin));
        if (staged) timing.dispatch.Observe(HybridNowUs()-start);
    };
    const int cpu = routes-count;
    if (cpu) {
        const auto *bf = reinterpret_cast<const uint16_t *>(s.host);
        double elapsed = 0;
        if (rows == 1) {
            for (int c = 0; c < 4096; ++c) s.cpuInput[c] = BFloat16BitsToFloat32(bf[c]);
            NumasMoeDecodeExpertsWithOverlap(s.cpuInput.data(),s.CpuOutput(),weights,s.Ids(),s.Owners(),topk,
                layer,submit,s.Scores(),l.swigluLimit,128,&elapsed);
        } else {
            NumasMoeVerifyExpertsWithOverlap(bf,s.CpuOutput(),rows,weights,weightsBatch,
                s.Ids(),s.Owners(),s.Scores(),topk,layer,l.swigluLimit,true,128,submit,&elapsed,group.cpuDecodeReady);
        }
        timing.ObserveDecodeCpu(cpu,elapsed);
    } else submit();
    auto &root = s.rank[origin];
    if (cpu) checkCudaErrors("TP CPU result", cudaMemcpyAsync(root.cpu.cudaData,s.CpuOutput(),size_t(routes)*4096*4,cudaMemcpyHostToDevice,cudaStreamPerThread));
    if (count) {
        checkCudaErrors("TP output fence", cudaStreamWaitEvent(cudaStreamPerThread,s.rank[1-origin].down,0));
        checkCudaErrors("TP output gather", cudaMemcpy2DAsync(static_cast<uint8_t *>(root.result.cudaData)+(1-origin)*8192,16384,
            static_cast<uint8_t *>(s.rank[1-origin].result.cudaData)+(1-origin)*8192,16384,8192,routes,cudaMemcpyDefault,cudaStreamPerThread));
    }
    checkCudaErrors("TP reduce IDs", cudaMemcpyAsync(root.ids.cudaData,s.Ids(),routes*4,cudaMemcpyHostToDevice,cudaStreamPerThread));
    checkCudaErrors("TP reduce owners", cudaMemcpyAsync(root.owners.cudaData,s.Owners(),routes*4,cudaMemcpyHostToDevice,cudaStreamPerThread));
    cuda::dsv41_cache::Reduce<<<dim3(16,rows),256,0,cudaStreamPerThread>>>(static_cast<float *>(root.cpu.cudaData),
        static_cast<float *>(root.result.cudaData),static_cast<int32_t *>(root.owners.cudaData),static_cast<int32_t *>(root.ids.cudaData),
        static_cast<__nv_bfloat16 *>(output.cudaData),4096,topk);
    checkCudaErrors("TP ordered reduction",cudaGetLastError());
    // done belongs to GPU 0, so record it on that device after a cross-device
    // event from the output owner. The next call observes one common fence.
    checkCudaErrors("TP root completion",cudaEventRecord(root.gate,cudaStreamPerThread));
    checkCudaErrors("TP completion device",cudaSetDevice(0));
    checkCudaErrors("TP completion wait",cudaStreamWaitEvent(cudaStreamPerThread,root.gate,0));
    checkCudaErrors("TP completed",cudaEventRecord(s.done,cudaStreamPerThread));
    checkCudaErrors("TP completion restore",cudaSetDevice(origin));
    s.pending=true; s.previousTable=timingIndex; s.previousGpu=count; s.previousStaged=staged;
    ++s.calls; ++timing.calls; s.residentRoutes+=hits; s.streamedRoutes+=staged; s.cpuRoutes+=cpu;
    return true;
}
} // namespace
