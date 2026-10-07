#pragma once

// A single frequency policy owns logical experts across all TP devices.
// Gate/up and down OUTPUT rows are sharded. Exchanging the small BF16
// activation between projections preserves the full dot-product reduction
// and Q8_K blocks instead of rounding and summing partial down projections.
namespace {
struct Glm5TensorCache {
    struct Rank {
        int device = -1;
        uint8_t *records = nullptr;
        cudaStream_t admission = nullptr;
        cudaEvent_t admitted = nullptr, computed = nullptr;
        fastllm::Data offsets, input, slots, routes, scores, gate, activation, scratch, output;
        uint16_t *hostGate = nullptr;
        float *hostOutput = nullptr;
        size_t hostGateBytes = 0, hostOutputBytes = 0;
        uint64_t computedRoutes = 0, uploadedBytes = 0;
        ~Rank() {
            int old = 0; cudaGetDevice(&old); cudaSetDevice(device);
            if (admitted) cudaEventSynchronize(admitted);
            if (computed) cudaEventSynchronize(computed);
            for (auto *data : {&offsets, &input, &slots, &routes, &scores, &gate, &activation, &scratch, &output}) data->FreeSpace();
            cudaFree(records); cudaFreeHost(hostGate); cudaFreeHost(hostOutput);
            if (admission) cudaStreamDestroy(admission);
            if (admitted) cudaEventDestroy(admitted);
            if (computed) cudaEventDestroy(computed);
            cudaSetDevice(old);
        }
    };
    OffloadGroup &group;
    std::vector<std::unique_ptr<Rank>> ranks;
    fastllm::cuda::CacheSlotPlan plan;
    std::unique_ptr<fastllm::MoeFrequencyPolicy> frequency;
    std::vector<fastllm::MoeDecodeScheduler::Estimate> cost;
    std::vector<int> selectedSlots, selectedRoutes;
    uint16_t *hostActivation = nullptr;
    float *hostOutput = nullptr;
    size_t activationBytes = 0, outputBytes = 0;
    bool active = false;
    explicit Glm5TensorCache(OffloadGroup &g) : group(g) {}
    ~Glm5TensorCache() {
        Wait(); cudaFreeHost(hostActivation); cudaFreeHost(hostOutput);
    }
    static void Allocate(fastllm::Data &x, fastllm::DataType type,
                         std::vector<int> dims, int device) {
        x.dataType = type; x.Resize(dims); x.ToDevice(fastllm::CUDA, {device}, false); x.Allocate(false);
    }
    template<class T> static void Pinned(T *&ptr, size_t &capacity, size_t bytes) {
        if (capacity >= bytes) return;
        cudaFreeHost(ptr); ptr = nullptr; capacity = 0;
        checkCudaErrors("TP expert host workspace", cudaMallocHost(&ptr, bytes)); capacity = bytes;
    }
    void Wait() {
        int old = 0; cudaGetDevice(&old);
        for (auto &r : ranks) {
            cudaSetDevice(r->device);
            if (r->admitted) checkCudaErrors("TP expert admission ready", cudaEventSynchronize(r->admitted));
            if (r->computed) checkCudaErrors("TP expert previous computation", cudaEventSynchronize(r->computed));
        }
        cudaSetDevice(old);
    }
    bool Init(const std::vector<int> &devices) {
        const int count = devices.size();
        if (count < 2 || group.layerLayouts.empty() ||
            group.layout.weightType != fastllm::DATA_GGUF_FORMAT || !group.layout.glm5 ||
            !group.cpuDecodeReady) return false;
        std::vector<size_t> strides;
        for (const auto &l : group.layerLayouts) {
            if (!l.glm5 || l.inter % count || l.hidden % count ||
                (l.inter / count) % 2 || (l.hidden / count) % 4 ||
                l.gateBytes % count || l.downBytes % count ||
                !FastllmCudaMoeGlm5GGUFCacheSupported(l.gateGgmlType, l.downGgmlType, l.hidden, l.inter)) return false;
            strides.push_back((l.gateBytes / count + l.downBytes / count + 127) / 128 * 128);
        }
        for (const auto &source : group.ggufSources) {
            if (!FastllmCudaMoeGlm5GGUFCacheNumaSupported(source.weights[0].type,
                    source.weights[1].type, source.weights[0].columns, source.weights[1].columns)) return false;
            // Native R4 slices start and end at complete four-row blocks.
            if ((source.weights[0].rows / count) % 4 || (source.weights[1].rows / count) % 4) return false;
        }
        int old = 0; cudaGetDevice(&old);
        struct Restore { int device; ~Restore(){cudaSetDevice(device);} } restore{old};
        uint64_t budget = UINT64_MAX;
        for (int device : devices) {
            if (!DeviceCacheBudgetBytes(device)) return false;
            auto found = group.deviceCaches.find(device);
            if (found != group.deviceCaches.end() && found->second && found->second->slots) return false;
            cudaSetDevice(device);
            size_t free = 0, total = 0;
            if (cudaMemGetInfo(&free, &total) != cudaSuccess) return false;
            // Retain runtime reserve plus small metadata and transfer workspaces.
            const size_t reserve = DeviceMemoryReserveBytes(total) + (32ULL << 20);
            budget = std::min(budget, std::min(DeviceCacheBudgetBytes(device),
                uint64_t(free > reserve ? free - reserve : 0)));
        }
        plan = fastllm::cuda::PlanCacheSlots(strides, group.layout.experts, budget, kMaxTopK);
        if (plan.offsets.size() < kMaxTopK) return false;
        for (int device : devices) {
            auto r = std::make_unique<Rank>(); r->device = device; cudaSetDevice(device);
            if (cudaMalloc(&r->records, plan.bytes) != cudaSuccess ||
                cudaStreamCreateWithFlags(&r->admission, cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreateWithFlags(&r->admitted, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&r->computed, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError(); return false;
            }
            Allocate(r->offsets, fastllm::INT8, {int(plan.offsets.size() * sizeof(uint64_t))}, device);
            checkCudaErrors("TP expert slot offsets", cudaMemcpy(r->offsets.cudaData, plan.offsets.data(),
                plan.offsets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors("TP expert admission init", cudaEventRecord(r->admitted, cudaStreamPerThread));
            checkCudaErrors("TP expert compute init", cudaEventRecord(r->computed, cudaStreamPerThread));
            ranks.push_back(std::move(r));
        }
        std::vector<int> keys(group.totalRecords), slots(plan.offsets.size());
        std::vector<uint64_t> bytes(group.totalRecords);
        std::map<int,int> partitions;
        for (int t = 0; t < int(strides.size()); ++t) {
            const auto span = plan.layers[t];
            const int id = partitions.emplace(span.begin, partitions.size()).first->second;
            std::fill_n(keys.begin() + t * group.layout.experts, group.layout.experts, id);
            std::fill_n(bytes.begin() + t * group.layout.experts, group.layout.experts,
                group.LayerLayout(t).gateBytes + group.LayerLayout(t).downBytes);
            std::fill_n(slots.begin() + span.begin, span.count, id);
        }
        frequency = std::make_unique<fastllm::MoeFrequencyPolicy>(std::move(keys), std::move(slots),
            strides.size(), fastllm::GetMoeCacheConfig(), std::move(bytes));
        cost.resize(FASTLLM_CUDA_MOE_CACHE_MAX_BATCH * strides.size());
        std::fprintf(stderr, "[Fastllm] GLM GGUF TP expert cache: %zu devices, %zu shared expert slots, "
            "%.3f GiB/device, %.3f GiB total; shared frequency policy, row-sharded projections.\n",
            ranks.size(), plan.offsets.size(), double(plan.bytes)/(1ULL<<30), double(plan.bytes)*ranks.size()/(1ULL<<30));
        return true;
    }
    void Begin() { Wait(); frequency->BeginStep(); active = true; }
    int Lookup(int table, const int32_t *ids, int routes) {
        selectedSlots.clear(); selectedRoutes.clear();
        for (int i = 0; i < routes; ++i) {
            const int slot = frequency->Slot(table * group.layout.experts + ids[i]);
            if (slot >= 0) { selectedSlots.push_back(slot); selectedRoutes.push_back(i); }
        }
        return selectedRoutes.size();
    }
    void Observe(int table, const int32_t *ids, int rows, int topk) {
        for (int row = 0; row < rows; ++row) frequency->Observe(table * group.layout.experts, ids + row * topk, topk);
    }
    void CopyNumaRows(int table, int expert, int part, int first, int count,
                      uint8_t *destination, cudaStream_t stream) {
        const auto &source = group.ggufSources[table];
        const auto &matrix = source.weights[part];
        const int perNode = matrix.rows / source.shards;
        const size_t base = (size_t(table) * group.layout.experts + expert) * 2 * source.shards + part * source.shards;
        while (count) {
            const int node = first / perNode, row = first % perNode, n = std::min(count, perNode - row);
            checkCudaErrors("TP expert shard upload", cudaMemcpyAsync(destination,
                static_cast<const uint8_t *>(group.numaPointers[base + node]) + size_t(row) * matrix.rowBytes,
                size_t(n) * matrix.rowBytes, cudaMemcpyHostToDevice, stream));
            destination += size_t(n) * matrix.rowBytes; first += n; count -= n;
        }
    }
    void End() {
        if (!active) return;
        active = false;
        const auto admissions = frequency->EndStep();
        if (admissions.empty()) return;
        int old = 0; cudaGetDevice(&old);
        struct Restore { int device; ~Restore(){cudaSetDevice(device);} } restore{old};
        const int count = ranks.size();
        for (int rank = 0; rank < count; ++rank) {
            auto &r = *ranks[rank]; cudaSetDevice(r.device);
            checkCudaErrors("TP expert previous admission", cudaEventSynchronize(r.admitted));
            checkCudaErrors("TP expert readers", cudaEventRecord(r.computed, cudaStreamPerThread));
            checkCudaErrors("TP expert admission readers", cudaStreamWaitEvent(r.admission, r.computed, 0));
            for (const auto &a : admissions) {
                const int table = a.key / group.layout.experts, expert = a.key % group.layout.experts;
                const auto &l = group.LayerLayout(table);
                uint8_t *dst = r.records + plan.offsets[a.slot];
                const size_t gate = l.gateBytes / count, down = l.downBytes / count;
                if (!group.ggufSources.empty()) {
                    CopyNumaRows(table, expert, 0, rank * (2*l.inter/count), 2*l.inter/count, dst, r.admission);
                    CopyNumaRows(table, expert, 1, rank * (l.hidden/count), l.hidden/count, dst + gate, r.admission);
                } else {
                    const auto *src = group.hostRecords + group.layerHostOffsets[table] + size_t(expert)*l.recordStride;
                    for (int part = 0; part < 2; ++part)
                        checkCudaErrors("TP expert canonical gate", cudaMemcpyAsync(dst + part*gate/2,
                            src + part*l.gateBytes/2 + rank*gate/2, gate/2, cudaMemcpyHostToDevice, r.admission));
                    checkCudaErrors("TP expert canonical down", cudaMemcpyAsync(dst + gate,
                        src + l.downOffset + rank*down, down, cudaMemcpyHostToDevice, r.admission));
                }
                r.uploadedBytes += gate + down;
            }
            checkCudaErrors("TP expert shard publication", cudaEventRecord(r.admitted, r.admission));
        }
    }
    double EstimateUs(int table, int rows) const {
        return selectedRoutes.size() * cost[(rows-1)*group.tableKeys.size()+table].us;
    }
    void Compute(int table, int rows, int topk, const uint16_t *input,
                  const float *scores, float *destination, int origin) {
        const int hits = selectedRoutes.size();
        if (!hits) return;
        const auto &l = group.LayerLayout(table);
        const int count = ranks.size(), inter = l.inter/count, hidden = l.hidden/count, routes = rows*topk;
        const double begin = HybridNowUs();
        Wait();
        Pinned(hostActivation, activationBytes, size_t(hits)*l.inter*sizeof(uint16_t));
        Pinned(hostOutput, outputBytes, size_t(routes)*l.hidden*sizeof(float));
        for (auto &ptr : ranks) {
            auto &r = *ptr; cudaSetDevice(r.device);
            Allocate(r.input, fastllm::BFLOAT16, {rows,l.hidden}, r.device);
            Allocate(r.slots, fastllm::INT32, {hits}, r.device);
            Allocate(r.routes, fastllm::INT32, {hits}, r.device);
            Allocate(r.scores, fastllm::FLOAT32, {routes}, r.device);
            Allocate(r.activation, fastllm::BFLOAT16, {hits,l.inter}, r.device);
            Allocate(r.scratch, fastllm::FLOAT32, {std::max(rows*l.hidden,hits*l.inter)}, r.device);
            Allocate(r.output, fastllm::FLOAT32, {hits,hidden}, r.device);
            Pinned(r.hostGate, r.hostGateBytes, size_t(hits)*inter*sizeof(uint16_t));
            Pinned(r.hostOutput, r.hostOutputBytes, size_t(hits)*hidden*sizeof(float));
            checkCudaErrors("TP expert input", cudaMemcpyAsync(r.input.cudaData,input,size_t(rows)*l.hidden*2,cudaMemcpyHostToDevice,cudaStreamPerThread));
            checkCudaErrors("TP expert slots", cudaMemcpyAsync(r.slots.cudaData,selectedSlots.data(),hits*sizeof(int),cudaMemcpyHostToDevice,cudaStreamPerThread));
            checkCudaErrors("TP expert routes", cudaMemcpyAsync(r.routes.cudaData,selectedRoutes.data(),hits*sizeof(int),cudaMemcpyHostToDevice,cudaStreamPerThread));
            checkCudaErrors("TP expert scores", cudaMemcpyAsync(r.scores.cudaData,scores,routes*sizeof(float),cudaMemcpyHostToDevice,cudaStreamPerThread));
            FastllmCudaMoeGGUFCacheView v{r.records,static_cast<int32_t *>(r.slots.cudaData),0,l.gateBytes/count,
                l.gateGgmlType,l.downGgmlType,l.hidden,inter,r.scratch.cudaData,size_t(r.scratch.GetBytes()),
                static_cast<uint64_t *>(r.offsets.cudaData)};
            v.routeMap=static_cast<int32_t *>(r.routes.cudaData);v.routeCount=hits;
            if (!group.ggufSources.empty()) v.numaGateType=group.ggufSources[table].weights[0].type;
            fastllm::AssertInFastLLM(FastllmCudaMoeGlm5GGUFCacheGate(r.input,r.gate,v,
                static_cast<float *>(r.scores.cudaData),topk,l.swigluLimit),"TP expert gate rejected");
            checkCudaErrors("TP expert gate gather",cudaMemcpyAsync(r.hostGate,r.gate.cudaData,size_t(hits)*inter*2,cudaMemcpyDeviceToHost,cudaStreamPerThread));
            checkCudaErrors("TP expert gate ready",cudaEventRecord(r.computed,cudaStreamPerThread));
        }
        for (int rank=0;rank<count;++rank) {
            auto &r=*ranks[rank];cudaSetDevice(r.device);
            checkCudaErrors("TP expert gate wait",cudaEventSynchronize(r.computed));
            for(int i=0;i<hits;++i) std::memcpy(hostActivation+size_t(i)*l.inter+rank*inter,r.hostGate+size_t(i)*inter,inter*2);
        }
        for (auto &ptr:ranks) {
            auto &r=*ptr;cudaSetDevice(r.device);
            checkCudaErrors("TP expert activation broadcast",cudaMemcpyAsync(r.activation.cudaData,hostActivation,size_t(hits)*l.inter*2,cudaMemcpyHostToDevice,cudaStreamPerThread));
            FastllmCudaMoeGGUFCacheView v{r.records,static_cast<int32_t *>(r.slots.cudaData),0,l.gateBytes/count,
                l.gateGgmlType,l.downGgmlType,hidden,l.inter,r.scratch.cudaData,size_t(r.scratch.GetBytes()),
                static_cast<uint64_t *>(r.offsets.cudaData)};
            if (!group.ggufSources.empty()) v.numaDownType=group.ggufSources[table].weights[1].type;
            fastllm::AssertInFastLLM(FastllmCudaMoeGlm5GGUFCacheDown(r.activation,v,static_cast<float *>(r.output.cudaData)),"TP expert down rejected");
            checkCudaErrors("TP expert output gather",cudaMemcpyAsync(r.hostOutput,r.output.cudaData,size_t(hits)*hidden*4,cudaMemcpyDeviceToHost,cudaStreamPerThread));
            checkCudaErrors("TP expert output ready",cudaEventRecord(r.computed,cudaStreamPerThread));
            r.computedRoutes+=hits;
        }
        std::fill_n(hostOutput,size_t(routes)*l.hidden,0.f);
        for(int rank=0;rank<count;++rank) {
            auto &r=*ranks[rank];cudaSetDevice(r.device);
            checkCudaErrors("TP expert output wait",cudaEventSynchronize(r.computed));
            for(int i=0;i<hits;++i) std::memcpy(hostOutput+size_t(selectedRoutes[i])*l.hidden+rank*hidden,r.hostOutput+size_t(i)*hidden,hidden*4);
        }
        cudaSetDevice(origin);
        checkCudaErrors("TP expert gathered output",cudaMemcpyAsync(destination,hostOutput,size_t(routes)*l.hidden*4,cudaMemcpyHostToDevice,cudaStreamPerThread));
        for(auto &r:ranks) if(r->device==origin) checkCudaErrors("TP expert staging lifetime",cudaEventRecord(r->computed,cudaStreamPerThread));
        cost[(rows-1)*group.tableKeys.size()+table].Observe((HybridNowUs()-begin)/hits);
    }
};

bool PrepareGlm5TensorCache(OffloadGroup &group, const std::vector<int> &devices) {
    if(group.tensorCache) {
        if(group.tensorCache->ranks.size()!=devices.size()) return false;
        for(size_t i=0;i<devices.size();++i) if(group.tensorCache->ranks[i]->device!=devices[i]) return false;
        return true;
    }
    auto cache=std::make_shared<Glm5TensorCache>(group);
    if(!cache->Init(devices)) return false;
    group.tensorCache=std::move(cache);return true;
}
} // namespace
