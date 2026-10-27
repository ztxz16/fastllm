// Included by diskdevice.cpp: bounded compact NVFP4 disk/RAM/CUDA hierarchy.
// Checkpoint metadata is immutable; only leased records contain payloads.
#ifdef USE_NUMAS
struct DiskCompactRecord {
    std::unique_ptr<Data> gate, down;
    int device = -1;
    uint64_t Bytes() const { return gate->GetBytes() + down->GetBytes(); }
    ~DiskCompactRecord() {
        // NUMA views borrow the contiguous owner, never separate allocations.
        if (gate) gate->numasData.clear();
        if (down) down->numasData.clear();
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (device >= 0) {
            const int previous = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            gate.reset(); down.reset();
            FastllmCudaSetDevice(previous);
        }
#endif
    }
};
using DiskCompactLease = std::shared_ptr<DiskCompactRecord>;

static void SetDiskCompactNumaViews(Data &data) {
    const int nodes = GetNumasMoeShardCount();
    AssertInFastLLM(data.dims[0] % nodes == 0, "Disk expert cannot split NUMA rows.\n");
    const size_t shardBytes = data.GetBytes() / nodes;
    for (int node = 0; node < nodes; ++node)
        data.numasData.push_back(data.cpuData + node * shardBytes);
}

static DiskCompactLease PackDiskCompact(std::unique_ptr<DiskCachedExpert> source) {
    auto record = std::make_shared<DiskCompactRecord>();
    for (int part = 0; part < 2; ++part) {
        const Data &src = part ? *source->down : *source->gate;
        auto dst = std::make_unique<Data>(NVFP4_BLOCK_16_E4M3_PACKED, src.dims);
        dst->blockK = 1; dst->blockM = 16;
        dst->Allocate(false);
        PackCompactE4M3NVFP4Block16Rows(src.dims[0], src.dims[1], src.cpuData,
            GetNVFP4ScaleData(src), src.scales, src.blockK, src.blockM,
            dst->cpuData, 0, src.dims[0], part == 0, false, true);
        SetDiskCompactNumaViews(*dst);
        (part ? record->down : record->gate) = std::move(dst);
    }
    return record;
}

#if defined(USE_CUDA) && !defined(USE_ROCM)
struct DiskCompactDeviceScope {
    const int previous = FastllmCudaGetDevice();
    explicit DiskCompactDeviceScope(int device) { FastllmCudaSetDevice(device); }
    ~DiskCompactDeviceScope() { FastllmCudaSetDevice(previous); }
};
static DiskCompactLease DownloadDiskCompact(const DiskCompactLease &deviceRecord) {
    DiskCompactDeviceScope device(deviceRecord->device);
    auto record = std::make_shared<DiskCompactRecord>();
    for (int part = 0; part < 2; ++part) {
        const Data &src = part ? *deviceRecord->down : *deviceRecord->gate;
        auto dst = std::make_unique<Data>(src.dataType, src.dims);
        dst->blockK = 1; dst->blockM = 16; dst->Allocate(false);
        FastllmCudaCopyFromDeviceToHost(dst->cpuData, src.cudaData, src.GetBytes());
        SetDiskCompactNumaViews(*dst);
        (part ? record->down : record->gate) = std::move(dst);
    }
    return record;
}
static DiskCompactLease UploadDiskCompact(const DiskCompactLease &host, int device) {
    DiskCompactDeviceScope scope(device);
    auto record = std::make_shared<DiskCompactRecord>();
    record->device = device;
    for (int part = 0; part < 2; ++part) {
        const Data &src = part ? *host->down : *host->gate;
        auto dst = std::make_unique<Data>(src.dataType, src.dims);
        dst->blockK = 1; dst->blockM = 16; dst->directMemory = true;
        dst->cudaData = FastllmCudaDirectMalloc(src.GetBytes());
        dst->expansionBytes = src.GetBytes(); dst->expansionSize = src.Count(0);
        dst->dataDevice = CUDA; dst->dataDeviceIds = {device};
        FastllmCudaCopyFromHostToDevice(dst->cudaData, src.cpuData, src.GetBytes());
        (part ? record->down : record->gate) = std::move(dst);
    }
    return record;
}
struct DiskCompactGpuJob {
    Data input, scores, activation, output;
    std::vector<int> routes;
    DiskCompactLease record;
    void *start = nullptr, *end = nullptr;
    bool queued = false, completed = false;
    DiskCompactGpuJob(const Data &x, const float *score, std::vector<int> selected,
                      int topk, int device) : routes(std::move(selected)) {
        input.dataType = BFLOAT16; input.Resize({int(routes.size()), x.dims[1]}); input.Allocate(false);
        scores.dataType = FLOAT32; scores.Resize({int(routes.size()), 1}); scores.Allocate(false);
        for (size_t i = 0; i < routes.size(); ++i) {
            memcpy(input.cpuData + i * x.dims[1] * 2,
                x.cpuData + size_t(routes[i] / topk) * x.dims[1] * 2, x.dims[1] * 2);
            ((float*)scores.cpuData)[i] = score[routes[i]];
        }
        input.ToDevice(CUDA, {device}, true); scores.ToDevice(CUDA, {device}, true);
        start = FastllmCudaEventCreateTiming(); end = FastllmCudaEventCreateTiming();
    }
    ~DiskCompactGpuJob() {
        // A disk read can fail after resident kernels were queued. Drain them
        // before their input/output buffers return to the shared CUDA pool.
        if (queued && !completed) FastllmCudaEventSynchronize(end);
        FastllmCudaEventDestroy(start); FastllmCudaEventDestroy(end);
    }
    void Run(const DiskCompactLease &weight, float limit) {
        record = weight;
        FastllmCudaEventRecordCurrentThread(start);
        AssertInFastLLM(FastllmCudaGlm5DiskExpert(input, scores, *record->gate, *record->down,
            activation, output, limit), "Disk compact CUDA expert unsupported.\n");
        FastllmCudaEventRecordCurrentThread(end);
        queued = true;
    }
    void Gather(float *perRoute, int hidden) {
        output.ToDevice(CPU);
        completed = true;
        for (size_t i = 0; i < routes.size(); ++i)
            memcpy(perRoute + size_t(routes[i]) * hidden,
                output.cpuData + i * hidden * sizeof(float), hidden * sizeof(float));
    }
};
#endif

struct DiskCompactTier {
    MoeFrequencyPolicy policy;
    std::vector<DiskCompactLease> slots;
    std::vector<int> owners;
    explicit DiskCompactTier(int records, int capacity, int layers, uint64_t stride)
        : policy(std::vector<int>(records, 0), std::vector<int>(capacity, 0), layers,
                 GetMoeCacheConfig(), std::vector<uint64_t>(records, stride)),
          slots(capacity), owners(capacity, -1) {
        for (int key = 0; key < records; ++key) policy.SetCandidateEligible(key, false);
    }
    DiskCompactLease Find(int key) const {
        int slot = policy.Slot(key);
        return slot >= 0 && owners[slot] == key ? slots[slot] : nullptr;
    }
    bool Install(MoeFrequencyPolicy::Admission a, DiskCompactLease record) {
        bool eviction = bool(slots[a.slot]);
        slots[a.slot] = std::move(record); owners[a.slot] = a.key;
        return eviction;
    }
    uint64_t Bytes() const {
        uint64_t bytes = 0; for (const auto &r : slots) if (r) bytes += r->Bytes(); return bytes;
    }
};
struct DiskCompactGroup {
    std::vector<std::vector<Data*>> layers;
    int experts = 0, hidden = 0, inter = 0;
    uint64_t stride = 0, ramBudget = 0, gpuBudget = 0, stagingLimit = 0;
    bool stepActive = false;
    std::unique_ptr<DiskCompactTier> ram;
    std::map<int, std::unique_ptr<DiskCompactTier>> gpus;
    std::map<int, std::vector<MoeDecodeOverlapScheduler>> timing;
    std::map<int, DiskCompactLease> staging;
    std::deque<int> stagingOrder;
    std::vector<int> gpuReferences;
    std::map<int, DiskCompactLease> demotions;
    DiskMoeCacheStats stats;
    DiskExpertPrefetch prefetch;

    explicit DiskCompactGroup(std::vector<std::vector<Data*>> tables) : layers(std::move(tables)) {
        experts = layers[0].size() / 2 - 1;
        gpuReferences.resize(experts * layers.size(), 0);
        hidden = layers[0][2]->dims[1]; inter = layers[0][3]->dims[1];
        stride = GetDataBytes(NVFP4_BLOCK_16_E4M3_PACKED, inter * 2, hidden) +
                 GetDataBytes(NVFP4_BLOCK_16_E4M3_PACKED, hidden, inter);
        ramBudget = GetMoeCpuCacheBytes(); gpuBudget = GetMoeCudaCacheBytes();
        // Staging is part of the requested RAM budget. Additional active
        // records are bounded by rows * topK (or streamed during prefill),
        // never an entire model.
        stagingLimit = ramBudget >= stride ? std::min<uint64_t>(
            ramBudget, std::max<uint64_t>(stride, std::min<uint64_t>(256ULL << 20, ramBudget / 16))) : 0;
        int capacity = std::min<uint64_t>(experts * layers.size(), (ramBudget - stagingLimit) / stride);
        ram = std::make_unique<DiskCompactTier>(experts * layers.size(), capacity, layers.size(), stride);
        std::fprintf(stderr, "[Fastllm] Disk compact frequency: %zu layers, %d experts/layer, %.3f MiB/expert; RAM %d slots + %.1f MiB staging.\n",
            layers.size(), experts, double(stride) / (1 << 20), capacity, double(stagingLimit) / (1 << 20));
    }
    bool CanHoldHost() const { return !ram->slots.empty() || stagingLimit >= stride; }
    DiskCompactTier *Gpu(int device) {
        if (device < 0 || gpuBudget < stride) return nullptr;
        auto &p = gpus[device];
        if (!p) {
            int capacity = std::min<uint64_t>(experts * layers.size(), gpuBudget / stride);
            p = std::make_unique<DiskCompactTier>(experts * layers.size(), capacity, layers.size(), stride);
            if (stepActive) p->policy.BeginStep();
            timing[device].resize(layers.size());
            // A zero-sized RAM tier cannot retain candidates across layers;
            // preserve GPU cache adaptation through bounded admission reads.
            if (!CanHoldHost())
                for (int key = 0; key < int(gpuReferences.size()); ++key) p->policy.SetCandidateEligible(key, true);
            for (int key : ram->owners) if (key >= 0) RefreshAvailability(key);
            for (const auto &entry : staging) RefreshAvailability(entry.first);
        }
        return p.get();
    }
    void Begin() {
        if (stepActive) return;
        stepActive = true; ram->policy.BeginStep();
        for (auto &gpu : gpus) gpu.second->policy.BeginStep();
    }
    DiskCompactLease FindHost(int key) {
        if (auto r = ram->Find(key)) return r;
        auto it = staging.find(key); return it == staging.end() ? nullptr : it->second;
    }
    void RefreshAvailability(int key) {
        const bool host = bool(FindHost(key));
        ram->policy.SetCandidateEligible(key, !gpuReferences[key] && (host || demotions.count(key)));
        for (auto &gpu : gpus) gpu.second->policy.SetCandidateEligible(key, host || !CanHoldHost());
    }
    void Unstage(int key) {
        if (!staging.erase(key)) return;
        stagingOrder.erase(std::remove(stagingOrder.begin(), stagingOrder.end(), key), stagingOrder.end());
        RefreshAvailability(key);
    }
    void DropRam(int key) {
        const int slot = ram->policy.Evict(key);
        if (slot < 0) return;
        if (ram->slots[slot]) ++stats.cpuEvictions;
        ram->slots[slot].reset(); ram->owners[slot] = -1;
        RefreshAvailability(key);
    }
    void InstallRam(MoeFrequencyPolicy::Admission a, DiskCompactLease record) {
        const int old = ram->owners[a.slot];
        if (ram->Install(a, std::move(record))) ++stats.cpuEvictions;
        Unstage(a.key);
        if (old >= 0) RefreshAvailability(old);
        RefreshAvailability(a.key);
    }
    void Stage(int key, const DiskCompactLease &r) {
        if (stagingLimit < stride || staging.count(key) || ram->Find(key) || gpuReferences[key]) return;
        while ((staging.size() + 1) * stride > stagingLimit && !stagingOrder.empty()) {
            // Retain the hottest already-read admission candidates across
            // layers, rather than only the final layers' most recent misses.
            int victim = stagingOrder.front();
            for (int candidate : stagingOrder)
                if (ram->policy.Score(candidate) < ram->policy.Score(victim)) victim = candidate;
            if (ram->policy.Score(key) < ram->policy.Score(victim)) return;
            Unstage(victim);
        }
        staging[key] = r; stagingOrder.push_back(key);
        RefreshAvailability(key);
    }
    std::unique_ptr<DiskCachedExpert> Read(int key) {
        auto &table = layers[key / experts]; int e = key % experts;
        for (int p = 0; p < 2; ++p)
            for (auto &part : table[2 + e * 2 + p]->diskWeightParts) stats.diskBytes += part.bytes;
        return ReadDiskExpert(table[2 + e * 2], table[3 + e * 2], false);
    }
    DiskCompactLease Host(int key) {
        if (auto r = FindHost(key)) return r;
        auto r = PackDiskCompact(Read(key)); Stage(key, r); return r;
    }
    // Two bounded reads hide SSD request latency behind packing and the
    // consumer's CPU/GPU work while serving uncached demand.
    template<class Consumer>
    void StreamHosts(const std::vector<int> &keys, Consumer consume) {
        std::map<int, std::future<std::unique_ptr<DiskCachedExpert>>> reads;
        struct Drain {
            decltype(reads) &pending;
            ~Drain() { for (auto &p : pending) if (p.second.valid()) p.second.wait(); }
        } drain{reads};
        auto queue = [&](size_t first) {
            for (size_t i = first; i < keys.size() && reads.size() < 2; ++i) {
                int key = keys[i];
                if (FindHost(key) || reads.count(key)) continue;
                auto &table = layers[key / experts]; int e = key % experts;
                for (int part = 0; part < 2; ++part)
                    for (auto &p : table[2 + e * 2 + part]->diskWeightParts) stats.diskBytes += p.bytes;
                auto pending = reads.emplace(key, std::future<std::unique_ptr<DiskCachedExpert>>{});
                pending.first->second = prefetch.Read(table[2 + e * 2], table[3 + e * 2], false, 2);
            }
        };
        queue(0);
        for (size_t i = 0; i < keys.size(); ++i) {
            const int key = keys[i];
            auto record = FindHost(key);
            auto pending = reads.find(key);
            if (pending != reads.end()) {
                auto source = pending->second.get(); reads.erase(pending);
                if (!record) record = PackDiskCompact(std::move(source));
            }
            if (!record) record = PackDiskCompact(Read(key));
            Stage(key, record);
            queue(i + 1);
            consume(key, std::move(record));
        }
    }
    void End() {
        if (!stepActive) return;
        stepActive = false;
#if defined(USE_CUDA) && !defined(USE_ROCM)
        struct Promotion {
            DiskCompactTier *tier;
            int device, victim;
            MoeFrequencyPolicy::Admission admission;
            DiskCompactLease host;
        };
        std::vector<Promotion> promotions;
        for (auto &gpu : gpus) {
            for (auto a : gpu.second->policy.EndStep()) {
                auto host = FindHost(a.key);
                AssertInFastLLM(bool(host) || !CanHoldHost(), "Disk GPU admission lost its host record.\n");
                const int old = gpu.second->owners[a.slot];
                promotions.push_back({gpu.second.get(), gpu.first, old, a, std::move(host)});
                if (old >= 0) {
                    --gpuReferences[old];
                    demotions[old] = gpu.second->slots[a.slot];
                }
                ++gpuReferences[a.key];
            }
        }
        // The plans pin their upload sources. RAM can now give those slots to
        // different experts, and rank GPU victims before deciding to copy them.
        for (const auto &p : promotions) {
            DropRam(p.admission.key); Unstage(p.admission.key);
            RefreshAvailability(p.admission.key);
            if (p.victim >= 0) RefreshAvailability(p.victim);
        }
#endif
        const auto admissions = ram->policy.EndStep();
        for (auto a : admissions) {
            auto record = FindHost(a.key);
#if defined(USE_CUDA) && !defined(USE_ROCM)
            if (!record && demotions.count(a.key)) {
                // Release the old host payload before allocating its replacement.
                if (ram->slots[a.slot]) ++stats.cpuEvictions;
                const int old = ram->owners[a.slot];
                ram->slots[a.slot].reset(); ram->owners[a.slot] = -1;
                if (old >= 0) RefreshAvailability(old);
                record = DownloadDiskCompact(demotions.at(a.key));
                ++stats.cudaDemotions; stats.cudaDemotionBytes += record->Bytes();
            }
#endif
            AssertInFastLLM(bool(record), "Disk RAM admission lost its available record.\n");
            InstallRam(a, std::move(record));
        }
#if defined(USE_CUDA) && !defined(USE_ROCM)
        for (const auto &p : promotions) {
            if (p.tier->slots[p.admission.slot]) ++stats.cudaEvictions;
            p.tier->slots[p.admission.slot].reset(); p.tier->owners[p.admission.slot] = -1;
        }
        demotions.clear(); // Free all victims before allocating CUDA replacements.
        for (const auto &p : promotions) {
            auto host = p.host ? p.host : Host(p.admission.key);
            p.tier->Install(p.admission, UploadDiskCompact(host, p.device));
            ++stats.uploads;
            if (p.victim >= 0) RefreshAvailability(p.victim);
            RefreshAvailability(p.admission.key);
        }
#endif
#if defined(__linux__) && defined(__GLIBC__)
        malloc_trim(0);
#endif
    }
    void PrefillAdmit(DiskCompactTier &tier, int key, int layer, DiskCompactLease r, bool gpu) {
        if (!gpu && gpuReferences[key]) return;
        tier.policy.SetCandidateEligible(key, true); // The caller holds the payload.
        auto a = tier.policy.SelectPrefill(key, layer * experts, {});
        if (a.key < 0) { RefreshAvailability(key); return; }
        if (!gpu) {
            tier.policy.Admit(layer, a); InstallRam(a, std::move(r));
            return;
        }
#if defined(USE_CUDA) && !defined(USE_ROCM)
        const int old = tier.owners[a.slot];
        auto evicted = tier.slots[a.slot];
        if (old >= 0) --gpuReferences[old];
        ++gpuReferences[key]; DropRam(key); Unstage(key);
        tier.policy.Admit(layer, a);
        if (tier.Install(a, std::move(r))) ++stats.cudaEvictions;
        RefreshAvailability(key);
        if (old >= 0 && !gpuReferences[old]) {
            ram->policy.SetCandidateEligible(old, true);
            auto hostAdmission = ram->policy.SelectPrefill(old, old / experts * experts, {});
            if (hostAdmission.key >= 0) {
                auto host = DownloadDiskCompact(evicted);
                ++stats.cudaDemotions; stats.cudaDemotionBytes += host->Bytes();
                ram->policy.Admit(old / experts, hostAdmission);
                InstallRam(hostAdmission, std::move(host));
            }
            RefreshAvailability(old);
        }
#endif
    }
};
static std::mutex &DiskCompactMutex() { static auto *m = new std::mutex; return *m; }
static std::unordered_map<const Data*, std::pair<std::shared_ptr<DiskCompactGroup>, int>> &DiskCompactRegistry() {
    static auto *r = new std::unordered_map<const Data*, std::pair<std::shared_ptr<DiskCompactGroup>, int>>;
    return *r;
}
static void ReleaseDiskCompact(const Data *weight) {
    std::lock_guard<std::mutex> lock(DiskCompactMutex());
    auto &r = DiskCompactRegistry(); auto it = r.find(weight); if (it == r.end()) return;
    auto group = it->second.first;
    for (auto i = r.begin(); i != r.end();) {
        if (i->second.first == group) i = r.erase(i); else ++i;
    }
}
static void AddDiskCompactStats(DiskMoeCacheStats &s) {
    std::lock_guard<std::mutex> lock(DiskCompactMutex());
    std::set<DiskCompactGroup*> seen;
    for (const auto &entry : DiskCompactRegistry()) {
        auto &g = *entry.second.first; if (!seen.insert(&g).second) continue;
        s.cpuBytes += g.ram->Bytes() + g.staging.size() * g.stride;
        for (auto &gpu : g.gpus) s.cudaBytes += gpu.second->Bytes();
        s.cpuHits += g.stats.cpuHits; s.cudaHits += g.stats.cudaHits; s.misses += g.stats.misses;
        s.diskBytes += g.stats.diskBytes; s.uploads += g.stats.uploads;
        s.cpuEvictions += g.stats.cpuEvictions; s.cudaEvictions += g.stats.cudaEvictions;
        s.cudaDemotions += g.stats.cudaDemotions; s.cudaDemotionBytes += g.stats.cudaDemotionBytes;
        for (int key : g.ram->owners)
            if (key >= 0 && g.gpuReferences[key]) s.cpuCudaOverlapBytes += g.stride;
        for (const auto &record : g.staging)
            if (g.gpuReferences[record.first]) s.cpuCudaOverlapBytes += g.stride;
    }
}
static void TrimDiskCompact() {
    std::lock_guard<std::mutex> lock(DiskCompactMutex());
    std::map<DiskCompactGroup*, std::shared_ptr<DiskCompactGroup>> replacements;
    for (auto &entry : DiskCompactRegistry()) {
        auto &g = *entry.second.first;
        if (g.ramBudget == GetMoeCpuCacheBytes() && g.gpuBudget == GetMoeCudaCacheBytes()) continue;
        auto &next = replacements[&g];
        if (!next) next = std::make_shared<DiskCompactGroup>(g.layers);
        entry.second.first = next;
    }
#if defined(__linux__) && defined(__GLIBC__)
    malloc_trim(0);
#endif
}
#endif

bool PrepareDiskMoeCache(const std::vector<std::vector<Data*>> &layers) {
#ifdef USE_NUMAS
    std::vector<std::vector<Data*>> tables;
    for (const auto &table : layers) {
        if (table.size() < 4 || !table[2] || !table[2]->isDiskWeight) continue;
        if (table[0] || table[1]) return false;
        for (size_t i = 2; i < table.size(); ++i) {
            const auto *w = table[i];
            if (!w || !w->isDiskWeight || w->dataType != NVFP4_BLOCK_16_E4M3 ||
                w->dims.size() != 2 || w->blockM != 16 || w->blockK <= 0 || w->scales.empty()) return false;
            if (w->dims != table[2 + i % 2]->dims || w->dims[0] % 128 || w->dims[1] % 128) return false;
        }
        if (table[2]->dims[0] != 2 * table[3]->dims[1] || table[2]->dims[1] != table[3]->dims[0]) return false;
        if (!tables.empty() && (table.size() != tables[0].size() || table[2]->dims != tables[0][2]->dims ||
            table[3]->dims != tables[0][3]->dims)) return false;
        tables.push_back(table);
    }
    if (tables.empty()) return false;
    std::lock_guard<std::mutex> lock(DiskCompactMutex());
    auto &r = DiskCompactRegistry();
    if (r.count(tables[0][2])) return true;
    auto group = std::make_shared<DiskCompactGroup>(std::move(tables));
    for (size_t i = 0; i < group->layers.size(); ++i)
        r.emplace(group->layers[i][2], std::make_pair(group, int(i)));
    return true;
#else
    return false;
#endif
}

static bool RunDiskCompact(const DataDict &datas, const FloatDict &fp, const IntDict &ip) {
#ifndef USE_NUMAS
    return false;
#else
    Data &input = *datas.at("input"), &output = *datas.at("output");
    Data &index = *datas.at("index"), &score = *datas.at("score");
    Data **weights = (Data**)datas.at("weights");
    if (input.dataType != BFLOAT16 || !ip.count("deepSeekV4Mode") || !ip.at("deepSeekV4Mode") ||
        (ip.count("activationQuantBlock") && ip.at("activationQuantBlock") != 128) ||
        (ip.count("gateType") && ip.at("gateType") != MoeGateSwiglu)) return false;
    std::lock_guard<std::mutex> lock(DiskCompactMutex());
    auto entry = DiskCompactRegistry().find(weights[2]);
    if (entry == DiskCompactRegistry().end()) return false;
    auto &g = *entry->second.first;
    int layer = entry->second.second, rows = input.dims[0], topk = index.dims[1], hidden = g.hidden;
    AssertInFastLLM(ip.at("weights___batch") == int(g.layers[layer].size()), "Disk expert table changed.\n");
    int device = -1;
#if defined(USE_CUDA) && !defined(USE_ROCM)
    device = FastllmCudaGetDevice();
#endif
    auto *gpu = g.Gpu(device);
    const bool prefill = rows > 9;
    if (!prefill) g.Begin();
    auto *ids = (int32_t*)index.cpuData;
    ToDataType(score, FLOAT32); auto *scores = (float*)score.cpuData;
    float limit = fp.count("swigluLimit") ? fp.at("swigluLimit") : 0;
    std::vector<std::vector<int>> routes(g.experts);
    std::vector<int> histogram(g.experts, 0);
    std::vector<DiskCompactLease> residents(g.experts), host(g.experts);
    for (int r = 0; r < rows * topk; ++r) {
        AssertInFastLLM(ids[r] >= 0 && ids[r] < g.experts, "Disk expert route out of range.\n");
        routes[ids[r]].push_back(r); ++histogram[ids[r]];
    }
    for (int e = 0; e < g.experts; ++e) if (!routes[e].empty()) {
        int key = layer * g.experts + e;
        residents[e] = gpu ? gpu->Find(key) : nullptr;
        if (residents[e]) g.stats.cudaHits += routes[e].size();
        else {
            host[e] = g.FindHost(key);
            (host[e] ? g.stats.cpuHits : g.stats.misses) += routes[e].size();
        }
    }
    if (prefill) {
        g.ram->policy.ObservePrefill(layer * g.experts, histogram, rows);
        if (gpu) gpu->policy.ObservePrefill(layer * g.experts, histogram, rows);
    } else for (int row = 0; row < rows; ++row) {
        if (gpu) gpu->policy.Observe(layer * g.experts, ids + row * topk, topk);
        g.ram->policy.Observe(layer * g.experts, ids + row * topk, topk);
    }
    std::vector<float> perRoute(size_t(rows) * topk * hidden, 0);
    if (prefill) {
        // At most one expert's arithmetic and two checkpoint reads in flight.
        // Do not retain a 288-expert layer merely to run its first token.
        std::vector<int> active;
        for (int e = 0; e < g.experts; ++e) if (!routes[e].empty()) active.push_back(e);
        std::map<int, std::future<std::unique_ptr<DiskCachedExpert>>> reads;
        struct Drain { decltype(reads) &r; ~Drain() { for (auto &p : r) if (p.second.valid()) p.second.wait(); } } drain{reads};
        auto queue = [&](size_t start) {
            for (size_t i = start; i < active.size() && reads.size() < 2; ++i) {
                int e = active[i]; if (host[e] || residents[e] || reads.count(e)) continue;
                auto &table = g.layers[layer];
                for (int p = 0; p < 2; ++p) for (auto &part : table[2 + e * 2 + p]->diskWeightParts) g.stats.diskBytes += part.bytes;
                auto pending = reads.emplace(e, std::future<std::unique_ptr<DiskCachedExpert>>{});
                pending.first->second = g.prefetch.Read(table[2 + e * 2], table[3 + e * 2], false, 2);
            }
        };
        queue(0);
        for (size_t i = 0; i < active.size(); ++i) {
            int e = active[i], key = layer * g.experts + e;
            if (!residents[e] && !host[e]) {
                auto pending = reads.find(e);
                if (pending != reads.end()) { host[e] = PackDiskCompact(pending->second.get()); reads.erase(pending); }
                else host[e] = g.Host(key);
            }
            queue(i + 1);
#if defined(USE_CUDA) && !defined(USE_ROCM)
            if (gpu) {
                auto record = residents[e] ? residents[e] : UploadDiskCompact(host[e], device);
                DiskCompactGpuJob job(input, scores, routes[e], topk, device);
                job.Run(record, limit); job.Gather(perRoute.data(), hidden);
                if (!residents[e]) {
                    g.PrefillAdmit(*gpu, key, layer, record, true);
                    ++g.stats.uploads;
                }
            } else
#endif
            {
                Data x(BFLOAT16, {int(routes[e].size()), hidden}); x.Allocate(false);
                std::vector<int> localIds(routes[e].size(), 0), localGpu(routes[e].size(), -1);
                std::vector<float> localScores, result(size_t(routes[e].size()) * hidden);
                for (size_t j = 0; j < routes[e].size(); ++j) {
                    memcpy(x.cpuData + j * hidden * 2, input.cpuData + size_t(routes[e][j] / topk) * hidden * 2, hidden * 2);
                    localScores.push_back(scores[routes[e][j]]);
                }
                Data *table[] = {nullptr, nullptr, host[e]->gate.get(), host[e]->down.get()};
                // Keep the exact decode activation/weight rounding boundary.
                // Large CPU GEMM preparation can otherwise switch arithmetic.
                for (size_t first = 0; first < routes[e].size(); first += 8) {
                    int count = std::min<size_t>(8, routes[e].size() - first);
                    NumasMoeVerifyExperts((uint16_t*)x.cpuData + first * hidden,
                        result.data() + first * hidden, count, table, 4,
                        localIds.data() + first, localGpu.data() + first,
                        localScores.data() + first, 1, layer, limit, true, 128);
                }
                for (size_t j = 0; j < routes[e].size(); ++j)
                    memcpy(perRoute.data() + size_t(routes[e][j]) * hidden, result.data() + j * hidden, hidden * sizeof(float));
            }
            if (host[e]) g.PrefillAdmit(*g.ram, key, layer, host[e], false);
            host[e].reset(); residents[e].reset();
        }
    } else {
        std::vector<int> misses, reuse;
        int hits = 0;
        for (int e = 0; e < g.experts; ++e) if (!routes[e].empty()) {
            if (residents[e]) hits += routes[e].size(); else misses.push_back(e);
        }
        std::stable_sort(misses.begin(), misses.end(), [&](int a, int b) { return routes[a].size() > routes[b].size(); });
        for (int e : misses) reuse.push_back(routes[e].size());
        auto &timing = g.timing[device]; if (timing.empty()) timing.resize(g.layers.size());
        auto &t = timing[layer];
        int staged = gpu ? t.SelectMisses(misses.size(), hits, reuse.data()) : 0;
        std::vector<int> gpuIndices(rows * topk, -1);
#if defined(USE_CUDA) && !defined(USE_ROCM)
        std::vector<std::unique_ptr<DiskCompactGpuJob>> jobs;
        // Resident work starts before waiting for SSD demand reads.
        for (int e = 0; e < g.experts; ++e) if (residents[e]) {
            auto job = std::make_unique<DiskCompactGpuJob>(input, scores, routes[e], topk, device);
            job->Run(residents[e], limit); jobs.push_back(std::move(job));
            for (int r : routes[e]) gpuIndices[r] = e;
        }
#endif
        std::vector<int> demand;
        for (int e : misses) if (!host[e]) demand.push_back(layer * g.experts + e);
        g.StreamHosts(demand, [&](int key, DiskCompactLease record) { host[key % g.experts] = std::move(record); });
        for (int i = 0; i < staged; ++i) for (int r : routes[misses[i]]) gpuIndices[r] = misses[i];
        std::vector<Data*> table(g.layers[layer].size(), nullptr);
        for (int e : misses) { table[2 + 2 * e] = host[e]->gate.get(); table[3 + 2 * e] = host[e]->down.get(); }
        if (!misses.empty() && !table[2]) { table[2] = host[misses[0]]->gate.get(); table[3] = host[misses[0]]->down.get(); }
        auto submit = [&] {
#if defined(USE_CUDA) && !defined(USE_ROCM)
            for (int i = 0; i < staged; ++i) {
                int e = misses[i]; double start = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
                auto record = UploadDiskCompact(host[e], device);
                double copied = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
                t.copiedExpert.Observe(copied - start);
                auto job = std::make_unique<DiskCompactGpuJob>(input, scores, routes[e], topk, device);
                job->Run(record, limit);
                // A timed sample includes this temporary expert's GPU execution;
                // CPU gate/up workers remain active throughout the callback.
                job->Gather(perRoute.data(), hidden);
                double end = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
                t.stagedExpert.Observe((end - copied) / routes[e].size());
            }
#endif
        };
        int cpuRoutes = std::count(gpuIndices.begin(), gpuIndices.end(), -1);
        auto start = std::chrono::steady_clock::now();
        if (cpuRoutes) {
            if (rows == 1) {
                std::vector<float> x(hidden);
                for (int i = 0; i < hidden; ++i) x[i] = ReadDiskFloatValue(input.cpuData, BFLOAT16, i);
                NumasMoeDecodeExpertsWithOverlap(x.data(), perRoute.data(), table.data(), ids, gpuIndices.data(),
                    topk, layer, submit, scores, limit, 128);
            } else NumasMoeVerifyExpertsWithOverlap((uint16_t*)input.cpuData, perRoute.data(), rows, table.data(), table.size(),
                ids, gpuIndices.data(), scores, topk, layer, limit, true, 128, submit);
            t.cpuExpert.Observe(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / cpuRoutes);
        } else submit();
#if defined(USE_CUDA) && !defined(USE_ROCM)
        for (auto &job : jobs) {
            job->Gather(perRoute.data(), hidden);
            t.residentExpert.Observe(FastllmCudaEventElapsedTime(job->start, job->end) * 1000 / job->routes.size());
        }
#endif
        ++t.calls;
    }
    output.Allocate(false);
    for (int row = 0; row < rows; ++row) for (int c = 0; c < hidden; ++c) {
        float sum = 0;
        for (int j = 0; j < topk; ++j) sum += perRoute[(size_t(row) * topk + j) * hidden + c];
        WriteDiskFloatValue(output.cpuData, output.dataType, size_t(row) * hidden + c, RoundFloat32ToBFloat16RNE(sum));
    }
    if (!prefill && layer + 1 == int(g.layers.size())) g.End();
    return true;
#endif
}
