// Included after ForwardWorkspace: reuses the ordinary model's arithmetic and
// captured projection/MoE graphs, replacing only dynamic decode orchestration.
namespace {
// Keep the hybrid graph coordinator near its GPU without changing the NUMA
// expert pool or the serving thread's affinity outside this operation.
struct V41FullCallerAffinity {
#if defined(USE_NUMAS) && defined(__linux__)
    cpu_set_t original;
    bool restore = false;
    explicit V41FullCallerAffinity(const std::vector<int> &cpus) {
        if (!cpus.empty() && sched_getaffinity(0, sizeof(original), &original) == 0)
            restore = fastllm::BindNumasWorkerCpuSet(cpus);
    }
    ~V41FullCallerAffinity() {
        if (restore) sched_setaffinity(0, sizeof(original), &original);
    }
#else
    explicit V41FullCallerAffinity(const std::vector<int> &) {}
#endif
};

Data &V41FullLocal(Data &d, int device) {
    if (!d.multiDeviceData) return d;
    return *d.multiDeviceDatas.at(device);
}
const Data &V41FullLocal(const Data &d, int device) {
    if (!d.multiDeviceData) return d;
    return *d.multiDeviceDatas.at(device);
}
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
struct V41OwnerFfn {
    Data residual, post, comb, input, logits, ids, scores, mixScratch;
    void *mixGemm = nullptr, *clusterMix = nullptr;
    std::vector<void *> mixGraphs;
    ~V41OwnerFfn() {
        if (mixGraphs.empty() && !mixGemm && !clusterMix) return;
        const int previous = FastllmCudaGetDevice();
        FastllmCudaSetDevice(mixScratch.dataDeviceIds.front());
        for (void *graph : mixGraphs) FastllmCudaGraphDestroy(graph);
        FastllmCudaV41DestroyOwnerMixGemm(mixGemm);
        FastllmCudaV41DestroyOwnerStream(clusterMix);
        FastllmCudaSetDevice(previous);
    }
};
#endif
struct V41FullLayer {
    FastllmV41FullDecodeConfig config;
    int source = -1;
    Data joined, scores, latents, keys, savedKV, savedRawKV, savedRawScore;
    std::vector<void *> plans;
    std::shared_ptr<void> outputProjection;
};
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
struct V41OwnerInputPacket {
    Data device;
    void *host = nullptr;
    size_t positionOffset = 0;
    std::vector<int> order;
    ~V41OwnerInputPacket() { if (host) FastllmCudaHostFree(host); }
};
struct V41OwnerEngram {
    size_t rowOffset = 0, maskOffset = 0;
    Data packetMask;
    Data rows, mask, gathered, kv, workspace, quantized;
    bool nativeProjection = false;
};
#endif
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
// Graph-owned divisors allow exact signed modulo for every Engram hash without
// a host integer divide per head. The table range is checked once at capture.
struct V41EngramBucket {
    uint64_t prime, offset, reciprocal;
    int64_t Row(uint64_t hash) const {
        const bool negative = (hash >> 63) != 0;
        const uint64_t magnitude = negative ? uint64_t(0) - hash : hash;
#ifdef __SIZEOF_INT128__
        const uint64_t quotient = uint64_t((__uint128_t(magnitude) * reciprocal) >> 64);
        uint64_t rem = magnitude - quotient * prime;
        if (rem >= prime) rem -= prime;
#else
        const uint64_t rem = magnitude % prime;
#endif
        return int64_t(offset + (negative && rem ? prime - rem : rem));
    }
};
#endif
struct V41FullEngram {
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    std::vector<std::unique_ptr<V41OwnerEngram>> owners;
    std::vector<V41EngramBucket> buckets;
#endif
    int index = 0;
    std::shared_ptr<V41EngramTable> table;
    Data rows, mask, gathered, hostGathered, kv, quantized, reduction;
    std::vector<int64_t> hostRows;
    std::vector<float> hostMask;
    void *asyncGraph = nullptr, *asyncData = nullptr;
    std::atomic<unsigned> *asyncReady = nullptr;
    ~V41FullEngram() {
        if (asyncGraph) FastllmCudaGraphDestroy(asyncGraph);
        if (asyncReady) FastllmCudaHostFree(asyncReady);
    }
};
// A single persistent CPU producer owns conversion and I/O tickets. It never
// enters the shared NUMA worker pool or submits CUDA while the graph runs.
struct V41FullEngramWorker {
    std::vector<V41FullEngram *> entries;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable wake, done;
    bool pending = false, stopping = false;
    std::exception_ptr error;
    explicit V41FullEngramWorker(std::vector<V41FullEngram *> entries) : entries(std::move(entries)) {
        worker = std::thread([this] { Run(); });
    }
    void Start() {
        std::lock_guard<std::mutex> lock(mutex);
        AssertInFastLLM(!pending, "full graph Engram producer still pending");
        for (auto *e : entries) e->asyncReady->store(0u, std::memory_order_release);
        pending = true;
        wake.notify_one();
    }
    void Finish() {
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [&] { return !pending; });
        auto failure = std::exchange(error, nullptr);
        if (failure) std::rethrow_exception(failure);
    }
    void Run() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || pending; });
                if (stopping) return;
            }
            try {
                for (auto *e : entries) {
                    auto &t = *e->table;
                    auto *dst = static_cast<uint16_t *>(e->asyncData);
                    if (t.ggufWeight) {
                        V41GatherGgufEngramRows(t, e->hostRows, dst);
                    } else {
                        v41_engram_detail::GatherNativeRows(t.data, t.scale,
                            t.diskValueReader.get(), t.diskScaleReader.get(), t.rows,
                            t.dim, t.scaleBlock, e->hostRows, dst);
                    }
                    e->asyncReady->store(1u, std::memory_order_release);
                }
            } catch (...) {
                error = std::current_exception();
                // Release every consumer before reporting failure. No output
                // from a failed producer is returned as successful inference.
                for (auto *e : entries) {
                    if (!e->asyncReady->load(std::memory_order_acquire)) {
                        std::memset(e->asyncData, 0, e->gathered.GetBytes());
                        e->asyncReady->store(1u, std::memory_order_release);
                    }
                }
            }
            { std::lock_guard<std::mutex> lock(mutex); pending = false; }
            done.notify_one();
        }
    }
    ~V41FullEngramWorker() {
        try { Finish(); } catch (...) {}
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_one();
        worker.join();
    }
};
struct V41FullGraph {
    std::vector<int> devices, coordinatorCpus;
    std::vector<std::vector<int>> rankCoordinatorCpus;
    int batch = 1, maxKeys = 0;
    int firstLayer = 0, endLayer = 0;
    bool verifying = false, capturesInput = false, hybrid = false, tensorParallel = true;
    // Rank-local attention uses the same dynamic KV plans as ordinary TP.
    // Its children are interleaved with global Engram/MoE by the DP parent.
    bool attentionOnly = false;
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    std::vector<std::unique_ptr<V41OwnerFfn>> ownerFfn;
    std::vector<std::vector<FastllmDsv41MegaLayer *>> ownerExperts;
    std::vector<std::unique_ptr<V41OwnerInputPacket>> inputPackets;
#endif
    std::vector<std::vector<const void *>> attentionLayers;
    // Alternate children share all workspaces and keep the ordinary graph for long or mixed requests.
    std::map<const void *, std::pair<int, const void *>> allIndexChildren;
    std::vector<void *> allIndexExecs[2];
    int allIndexLimit[2] = {};
    std::vector<void *> splitAttentionChildren;
    void *attentionMixGemm = nullptr;
    parallel::CudaRuntime *attentionRuntime = nullptr;
    std::unique_ptr<parallel::CudaRowTransfer> rowTransfer;
    std::vector<std::shared_ptr<DeepSeekV41CudaGraphState>> attentionStates;
    std::vector<std::shared_ptr<V41FullGraph>> attentionGraphs;
    // Private warmup requests supply independent KV for padding. They never
    // enter the scheduler or prefix cache and never alias a serving request.
    std::vector<DeepSeekV41Segment> padding;
    std::vector<void *> moeHandles;
    std::vector<std::vector<void *>> rankMoeHandles;
    std::vector<void *> rankUntimedExecs;
    std::vector<int> rankMoeFirstLayer;
    Data routedOutput;
    std::unique_ptr<DiskMoeV41Graph> diskMoe;
    Data inputIds, embedding, deferredHcMix, indexHeadWeights;
    std::vector<std::unique_ptr<Data>> ownerInputIds;
    std::vector<std::vector<int>> ownerHostIds;
    const void *embeddingWeightAddress = nullptr;
    std::vector<float> hostTokens;
    std::vector<std::unique_ptr<V41FullLayer>> layers;
    std::map<int, std::unique_ptr<V41FullEngram>> engrams;
    std::unique_ptr<V41FullEngramWorker> engramWorker;
    Data topk, candidates, head, norm, normFloat, headLogits, logits, requests, gatheredLogits;
    Data candidateBlocks, candidateCounts;
    Data greedyIds, greedyScores;
    std::map<int, void *> greedyReady;
    std::map<int, void *> greedyHost, greedyMapped;
    bool captureGreedy = false;
    std::map<int, Data> mainHidden;
    Data packedMainHidden;
    std::vector<void *> ready;
    void *lastVerifyReader = nullptr;
    std::vector<Data> commitDescriptors;
    Data commitSpans;
    std::vector<int> hostCommitSpans;
    std::vector<std::vector<FastllmV41FullDecodeRequest>> hostRequests;
    std::vector<std::vector<void *>> hostPlans;
    struct FixedRow {
        std::weak_ptr<void> lease;
        const DeepSeekV41RequestState *state;
        uint64_t revision;
        int position;
        int64_t limit;
    };
    std::vector<FixedRow> fixedRows;
    bool trackPositions = false, haveFixedUpload = false, advancePositions = false;
    std::shared_ptr<DeepSeekV41CudaGraphState> pieces;
    std::vector<void *> graphs, execs;
    void *untimedExec = nullptr;
    uint64_t replays = 0;
    ~V41FullGraph() {
        engramWorker.reset();
        const int old = FastllmCudaGetDevice();
        for (int d : devices)
            FastllmCudaSyncDevice(d);
        for (size_t rank = 0; rank < devices.size(); ++rank) {
            FastllmCudaSetDevice(devices[rank]);
            for (auto &variant : allIndexExecs)
                if (rank < variant.size() && variant[rank]) FastllmCudaGraphExecDestroy(variant[rank]);
            if (rank < execs.size() && execs[rank])
                FastllmCudaGraphExecDestroy(execs[rank]);
            if (rank < graphs.size() && graphs[rank])
                FastllmCudaGraphDestroy(graphs[rank]);
            auto pinned = greedyHost.find(devices[rank]);
            if (pinned != greedyHost.end()) FastllmCudaHostFree(pinned->second);
            if (rank < ready.size() && ready[rank])
                FastllmCudaEventDestroy(ready[rank]);
        }
        if (lastVerifyReader) {
            FastllmCudaSetDevice(devices.front());
            FastllmCudaEventDestroy(lastVerifyReader);
        }
        if (untimedExec) FastllmCudaGraphExecDestroy(untimedExec);
        for (size_t rank = 0; rank < rankUntimedExecs.size(); ++rank) {
            FastllmCudaSetDevice(devices[rank]);
            if (rankUntimedExecs[rank]) FastllmCudaGraphExecDestroy(rankUntimedExecs[rank]);
        }
        for (void *handle : moeHandles) FastllmCudaReleaseNativeMoeFullGraph(handle);
        for (size_t rank = 0; rank < rankMoeHandles.size(); ++rank) {
            FastllmCudaSetDevice(devices[rank]);
            for (void *handle : rankMoeHandles[rank]) FastllmCudaReleaseNativeMoeFullGraph(handle);
        }
        diskMoe.reset();
        rowTransfer.reset();
        if (!splitAttentionChildren.empty() || attentionMixGemm) {
            FastllmCudaSetDevice(devices.front());
            for (auto clone : splitAttentionChildren) FastllmCudaGraphDestroy(clone);
            FastllmCudaV41DestroyOwnerMixGemm(attentionMixGemm);
        }
        attentionGraphs.clear();
        attentionStates.clear();
        pieces.reset();
        for (auto &layer : layers) if (layer)
            for (void *plan : layer->plans)
                FastllmCudaV41DestroyFullDecode(plan);
        FastllmCudaSetDevice(old);
    }
};

// Match the captured row counts, but not request identities or their order.
// KV pointers and positions are uploaded anew; the persistent permutation is
// updated when requests move within the batch. Unknown layouts stay eager.
bool V41FullPartition(V41FullGraph &g, const std::vector<DeepSeekV41Segment> &segments,
                      std::vector<std::vector<DeepSeekV41Segment>> &local,
                      parallel::RowPartition &rows) {
    const auto &topology = g.attentionRuntime->Layout();
    std::vector<int> owners;
    local.assign(topology.Size(), {});
    if (g.verifying && segments.size() > 1 && !V41VerifyOwnerFfnEnabled()) return false;
    for (const auto &s : segments) {
        const bool mode = g.verifying ? s.spec && s.spec->deferWindow && s.spec->wantAllTokens &&
            s.seqlen > 0 : !s.spec && s.seqlen == 1;
        if (!s.state || !mode || s.startPos < 1 ||
            s.offset != int(owners.size()) || s.seqlen > g.batch - s.offset ||
            s.state->attentionGroup < 0 || s.state->attentionGroup >= topology.attentionDP ||
            s.state->attentionRanks.size() != size_t(topology.attentionTP)) return false;
        owners.insert(owners.end(), s.seqlen, s.state->attentionGroup);
        if (g.verifying) V41PrepareAttentionSpec(*s.spec, topology.attentionTP, s.state->layers.size());
        for (int t = 0; t < topology.attentionTP; ++t) {
            const int rank = s.state->attentionGroup * topology.attentionTP + t;
            auto part = s;
            part.state = s.state->attentionRanks[t];
            if (g.verifying) part.spec = s.spec->attentionRanks[t].get();
            if (!part.state) return false;
            auto &parts = local[rank];
            part.offset = parts.empty() ? 0 : parts.back().offset + parts.back().seqlen;
            parts.push_back(std::move(part));
        }
    }
    if (int(owners.size()) != g.batch) return false;
    rows = parallel::RowPartition(topology.attentionDP, owners);
    if (!g.attentionGraphs.empty()) for (int r = 0; r < topology.Size(); ++r) {
        const auto &graph = g.attentionGraphs[r];
        int count = 0;
        for (const auto &part : local[r]) count += part.seqlen;
        if (count != (graph ? graph->batch : 0)) return false;
        if (graph && g.verifying && graph->layers[graph->firstLayer]->config.segmentedVerify !=
                (local[r].size() > 1)) return false;
        if (graph && (!g.attentionStates[r]->captured || g.attentionStates[r]->disabled ||
            g.attentionStates[r]->fullDecode != graph)) return false;
    }
    return true;
}

bool V41FullDispatch(const V41FullGraph &g, const std::function<void(int, int)> &body) {
    if (g.attentionRuntime) { g.attentionRuntime->Enqueue(body); return true; }
    if (g.tensorParallel) return MultiCudaRunDeviceCallbacks(g.devices, [&](int rank, int device) {
        const std::vector<int> empty;
        V41FullCallerAffinity affinity(rank < int(g.rankCoordinatorCpus.size())
            ? g.rankCoordinatorCpus[rank] : empty);
        body(rank, device);
    });
    FastllmCudaSetDevice(g.devices.front());
    body(0, g.devices.front());
    return true;
}
void V41FullReserveRows(V41FullGraph &g, Data &data, DataType type, int width, int rows) {
    if (g.tensorParallel) { V41ReserveDecodeRows(data, type, width, rows, g.devices); return; }
    if (data.dims.empty()) {
        data.dataType = type; data.UpdateUnitSize(); data.Resize({1, 0, width});
        data.dataDevice = DataDevice::CUDA; data.dataDeviceIds = g.devices;
    }
    data.ToDevice(DataDevice::CUDA, g.devices);
    if (V41CacheRowCapacity(data) < rows) {
        int capacity = std::max(rows, V41CacheRowCapacity(data) * 3 / 2);
        if (rows > 2) capacity = (capacity + 255) / 256 * 256;
        data.Expansion({1, capacity, width});
    }
}

// All allocations happen before graph launch. Moving a cache is safe because
// graph kernels load its current pointer from request metadata on every replay.
bool V41FullReserve(V41FullGraph &g, const std::vector<DeepSeekV41Segment> &segments,
                    bool fixedOnly = false, bool checkOwnedVerify = false) {
    g.advancePositions = false;
    if (g.trackPositions && fixedOnly && g.haveFixedUpload && !segments.empty() &&
        segments.size() == g.fixedRows.size() && int(segments.size()) == g.batch) {
        bool same = true;
        for (size_t i = 0; same && i < segments.size(); ++i) {
            const auto &s = segments[i]; const auto &row = g.fixedRows[i];
            // A weak ownership comparison detects a recycled slot even if its
            // lease and state are allocated at their predecessor's addresses.
            same = s.state && row.state == s.state.get() && s.state->storageLease &&
                !row.lease.owner_before(s.state->storageLease) && !s.state->storageLease.owner_before(row.lease) &&
                row.revision == s.state->storageRevision && !s.spec && s.seqlen == 1 && s.offset == int(i) &&
                int64_t(s.startPos) == int64_t(row.position) + 1 && s.startPos <= row.limit;
        }
        if (same) { g.advancePositions = true; return true; }
    }
    g.haveFixedUpload = false;
    g.fixedRows.clear();
    if (fixedOnly && (g.tensorParallel || g.devices.size() != 1)) return false;
    int totalRows = 0;
    for (const auto &s : segments) {
        if (s.offset != totalRows || s.seqlen < 1) return false;
        totalRows += s.seqlen;
    }
    if (totalRows != g.batch || (!g.verifying && (int)segments.size() != g.batch)) return false;
    for (const auto &s : segments) {
        const bool validMode = g.verifying
            ? s.spec && s.spec->deferWindow && s.spec->wantAllTokens
            : !s.spec && s.seqlen == 1;
        if (!s.state || !validMode || s.startPos < 1 ||
            (int)s.state->layers.size() != (int)g.layers.size())
            return false;
        if (fixedOnly && !s.state->storageLease && !checkOwnedVerify) return false;
        int64_t limit = INT_MAX - 1;
        for (int l = g.firstLayer; l < g.endLayer; ++l) {
            const auto &cfg = g.layers[l]->config;
            auto &cache = s.state->layers[l];
            if (cfg.ratio && (s.startPos + (long long)s.seqlen) / cfg.ratio > g.maxKeys)
                return false;
            if (cache.windowKV.dataType != (cfg.ringMode ? DataType::INT8 : DataType::BFLOAT16) ||
                cache.windowKV.dims.size() != 3 || cache.windowKV.dims[0] != 1 ||
                cache.windowKV.dims[1] != cfg.window || cache.windowKV.dims[2] != (cfg.ringMode ? cfg.ringRowBytes : 512) ||
                (g.tensorParallel && (!cache.windowKV.multiDeviceData ||
                !cache.windowKV.IsTensorParallelReplicated())))
                return false;
            auto validRows = [](const Data &data, int mode, int width) {
                return data.dims.empty() || (data.dataType == (mode ? DataType::INT8 : DataType::BFLOAT16) &&
                    data.dims.size() == 3 && data.dims[0] == 1 && data.dims[2] == width);
            };
            if (cfg.kvSource && (!validRows(cache.compressedKV, cfg.cmpMode, cfg.cmpMode ? cfg.cmpRowBytes : 512) ||
                (cfg.indexSource && !validRows(cache.indexK, cfg.indexMode, cfg.indexMode ? cfg.indexRowBytes : 128)))) return false;
            if (fixedOnly) {
                // A staged verifier can also use request-owned CUDA allocations.
                // Check their current capacities before accepting the cached addresses.
                auto resident = [&](const Data &d, int rows) {
                    return (d.isFake || checkOwnedVerify) && d.cudaData && !d.multiDeviceData &&
                        d.dataDevice == CUDA && d.dataDeviceIds == g.devices &&
                        V41CacheRowCapacity(d) >= rows;
                };
                if (!resident(cache.windowKV, cfg.window)) return false;
                if (g.trackPositions && cfg.ratio) {
                    int cap = g.maxKeys;
                    if (cfg.kvSource) {
                        cap = std::min(cap, V41CacheRowCapacity(cache.compressedKV));
                        if (cfg.indexSource) cap = std::min(cap, V41CacheRowCapacity(cache.indexK));
                    }
                    limit = std::min(limit, (int64_t(cap) + 1) * cfg.ratio - 2);
                }
                if (cfg.kvSource) {
                    const int rows = (s.startPos + s.seqlen) / cfg.ratio;
                    if (!resident(cache.compressedKV, rows) || (cfg.indexSource && !resident(cache.indexK, rows)) ||
                        (cfg.ratio == 2 && (!resident(cache.rawTailKV, 2) || !resident(cache.rawTailScore, 2))))
                        return false;
                }
            }
            for (int device : g.devices) {
                if (!V41FullLocal(cache.windowKV, device).cudaData)
                    return false;
            }
        }
        if (fixedOnly && g.trackPositions)
            g.fixedRows.push_back({s.state->storageLease,s.state.get(),s.state->storageRevision,s.startPos,limit});
    }
    if (fixedOnly) return true;
    for (const auto &s : segments)
        for (int l = g.firstLayer; l < g.endLayer; ++l) {
            const auto &cfg = g.layers[l]->config;
            if (!cfg.kvSource)
                continue;
            auto &cache = s.state->layers[l];
            const int rows = (s.startPos + s.seqlen) / cfg.ratio;
            V41FullReserveRows(g, cache.compressedKV, cfg.cmpMode ? DataType::INT8 : DataType::BFLOAT16,
                                 cfg.cmpMode ? cfg.cmpRowBytes : 512, rows);
            if (cfg.indexSource)
                V41FullReserveRows(g, cache.indexK, cfg.indexMode ? DataType::INT8 : DataType::BFLOAT16,
                                     cfg.indexMode ? cfg.indexRowBytes : 128, rows);
            if (cfg.ratio == 2) {
                V41FullReserveRows(g, cache.rawTailKV, DataType::FLOAT32, 512, 2);
                V41FullReserveRows(g, cache.rawTailScore, DataType::FLOAT32, 512, 2);
            }
        }
    return true;
}

bool V41FullUploadRank(V41FullGraph &g, const std::vector<DeepSeekV41Segment> &segments, int rank,
                       int device) {
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    if (!g.inputPackets.empty()) return true; // The rank graph copies its packed owner inputs.
#endif
    if (g.advancePositions && g.trackPositions && g.engrams.empty() && !g.capturesInput) {
        const size_t begin = size_t(g.firstLayer) * g.batch;
        auto *storage = static_cast<FastllmV41FullDecodeRequest *>(V41FullLocal(g.requests,device).cudaData);
        if (!FastllmCudaV41AdvancePositions(storage + begin, size_t(g.endLayer - g.firstLayer) * g.batch))
            return false;
        for (auto &row : g.fixedRows) ++row.position;
        return true;
    }
    auto &requests = g.hostRequests[rank];
    auto &plans = g.hostPlans[rank];
    if (!g.attentionRuntime) {
        requests.resize(g.layers.size() * g.batch);
        plans.resize(g.layers.size(), nullptr);
    }
    auto ptr = [device](Data &d) -> void * {
        return V41FullLocal(d, device).cudaData;
    };
    for (int l = g.firstLayer; l < g.endLayer; ++l) {
        if (!g.layers[l]) continue;
        const auto &layer = *g.layers[l];
        if (layer.plans.empty())
            continue;
        size_t segmentIndex = 0;
        for (int i = 0; i < g.batch; ++i) {
            while (i >= segments[segmentIndex].offset + segments[segmentIndex].seqlen) ++segmentIndex;
            const auto &segment = segments[segmentIndex];
            auto &cache = segment.state->layers[l];
            auto &source = segment.state->layers[layer.source < 0 ? l : layer.source];
            requests[l * g.batch + i] = {ptr(cache.windowKV), ptr(source.compressedKV), ptr(source.indexK),
                ptr(cache.rawTailKV), ptr(cache.rawTailScore), segment.startPos + i - segment.offset,
                g.verifying ? segment.offset : 0, g.verifying ? segment.seqlen : 0};
        }
        plans[l] = layer.plans[rank];
    }
    std::vector<FastllmV41HostCopy> copies;
    copies.reserve(2 * g.engrams.size() + 1);
    for (auto &entry : g.engrams) {
        auto &e = *entry.second;
        if (e.table->gpuTable)
            copies.push_back({ptr(e.rows), e.hostRows.data(), e.hostRows.size() * sizeof(int64_t)});
        else if (!e.asyncGraph)
            copies.push_back({ptr(e.gathered), e.hostGathered.cpuData, e.hostGathered.GetBytes()});
    }
    for (auto &entry : g.engrams) {
        auto &e = *entry.second;
        copies.push_back({ptr(e.mask), e.hostMask.data(), e.hostMask.size() * sizeof(float)});
    }
    if (!g.ownerInputIds.empty() && g.ownerInputIds[rank])
        copies.push_back({g.ownerInputIds[rank]->cudaData, g.ownerHostIds[rank].data(),
                          g.ownerHostIds[rank].size() * sizeof(int)});
    if (g.capturesInput && g.ownerInputIds.empty() && rank == 0)
        copies.push_back({ptr(g.inputIds), g.hostTokens.data(), g.hostTokens.size() * sizeof(float)});
    if (g.attentionRuntime)
        return FastllmCudaV41UploadAuxiliaryInputs(copies.data(), copies.size()) && !FastllmCudaGetThreadError();
    if (!FastllmCudaV41UploadFullDecodeInputs(ptr(g.requests), plans.data(), plans.size(),
            requests.data(), g.batch, copies.data(), copies.size())) return false;
    if (g.trackPositions && !g.advancePositions && g.engrams.empty() && !g.capturesInput)
        g.haveFixedUpload = int(g.fixedRows.size()) == g.batch;
    return !FastllmCudaGetThreadError();
}

void V41FullAdvance(V41FullGraph &g, const std::vector<DeepSeekV41Segment> &segments) {
    auto resize = [&](Data &data, int rows) {
        if (data.dims.empty() || (!g.tensorParallel && data.dims[1] == rows))
            return;
        // Slot views have fixed physical expansion strides. Updating their
        // logical row count has no allocation or TP layout side effect.
        if (g.trackPositions && data.isFake && !data.multiDeviceData && data.dims.size() == 3 &&
            data.expansionDims.size() == 3 && rows >= 0 && rows <= data.expansionDims[1]) {
            data.dims[1] = rows;
            return;
        }
        auto dims = data.dims;
        dims[1] = rows;
        for (int d : g.devices) V41FullLocal(data, d).Resize(dims);
        if (g.tensorParallel) V41SyncRootFromReplica(data, g.devices);
    };
    for (const auto &s : segments)
        for (int l = g.firstLayer; l < g.endLayer; ++l) {
            auto &cache = s.state->layers[l];
            cache.totalLen = s.startPos + s.seqlen;
            const auto &cfg = g.layers[l]->config;
            if (!cfg.kvSource)
                continue;
            cache.compressedBlocks = (s.startPos + s.seqlen) / cfg.ratio;
            cache.rawTail = (s.startPos + s.seqlen) % cfg.ratio;
            resize(cache.compressedKV, cache.compressedBlocks);
            if (cfg.indexSource)
                resize(cache.indexK, cache.compressedBlocks);
            if (!g.verifying) {
                resize(cache.rawTailKV, cache.rawTail);
                resize(cache.rawTailScore, cache.rawTail);
            }
        }
}
struct V41RequestVerifyPages {
    std::weak_ptr<V41FullGraph> graph;
    std::weak_ptr<DeepSeekV41RequestState> state;
    std::weak_ptr<void> lease;
    uint64_t revision = 0;
    std::vector<FastllmV41FullDecodeRequest> addresses;
    std::shared_ptr<void> slot;

    static FastllmV41FullDecodeRequest Addresses(const V41FullGraph &g, const DeepSeekV41RequestState &s, int l) {
        const auto &cache = s.layers[l];
        const auto &source = s.layers[g.layers[l]->source < 0 ? l : g.layers[l]->source];
        return {cache.windowKV.cudaData, source.compressedKV.cudaData, source.indexK.cudaData,
                cache.rawTailKV.cudaData, cache.rawTailScore.cudaData};
    }
    bool Matches(const V41FullGraph &g, const DeepSeekV41RequestState &s) const {
        if (graph.lock().get() != &g || state.lock().get() != &s ||
            lease.owner_before(s.storageLease) || s.storageLease.owner_before(lease) ||
            revision != s.storageRevision || addresses.size() != g.layers.size()) return false;
        for (size_t l = 0; l < addresses.size(); ++l) {
            const auto a = Addresses(g,s,l); const auto &b = addresses[l];
            if (a.windowKV != b.windowKV || a.compressedKV != b.compressedKV || a.indexK != b.indexK ||
                a.rawTailKV != b.rawTailKV || a.rawTailScore != b.rawTailScore) return false;
        }
        return true;
    }
};
void V41FullBindVerify(const std::shared_ptr<V41FullGraph> &owner, const DeepSeekV41Segment &segment,
                      const std::vector<int> &targetLayers) {
    auto &g = *owner;
    // Scratch views borrow graph buffers until commit/draft completes,
    // before the next target forward can overwrite them.
    auto borrow = [&](Data &view, const Data &source, int rows, int offset) {
        if (g.tensorParallel) {
            AssertInFastLLM(V41DecodeSliceView(view, source, offset, g.devices, rows),
                            "full verify buffer view failed");
        } else {
            AssertInFastLLM(!source.multiDeviceData && source.cudaData &&
                source.dims.size() >= 3 && source.dims[0] == 1 && source.dims[1] >= offset + rows,
                "full verify local buffer view failed");
            view.FakeFrom(source, (size_t)offset * source.Count(2) * source.unitSize);
            auto dims = source.dims; dims[1] = rows;
            view.Resize(dims);
        }
    };
    auto &scratch = *segment.spec;
    const int layers = g.layers.size();
    scratch.prevRawTail.resize(layers);
    scratch.prevBlocks.resize(layers);
    const bool pipeline = g.firstLayer > 0 || g.endLayer < layers;
    if (pipeline) {
        scratch.windowKV.resize(layers);
        scratch.rawKV.resize(layers);
        scratch.rawScore.resize(layers);
    }
    for (int l = g.firstLayer; l < g.endLayer; ++l) {
        if (!g.layers[l]) continue;
        const auto &layer = *g.layers[l];
        if (pipeline) borrow(scratch.windowKV[l], layer.savedKV, segment.seqlen, segment.offset);
        if (!layer.config.kvSource) continue;
        if (pipeline) {
            const int rawRows = segment.startPos % layer.config.ratio + segment.seqlen;
            borrow(scratch.rawKV[l], layer.savedRawKV, rawRows, 2 * segment.offset);
            if (layer.config.ratio == 2) borrow(scratch.rawScore[l], layer.savedRawScore, rawRows, 2 * segment.offset);
        }
        scratch.prevRawTail[l] = segment.startPos % layer.config.ratio;
        scratch.prevBlocks[l] = segment.startPos / layer.config.ratio;
    }
    if (!g.attentionRuntime) scratch.commitKV = [owner, offset = segment.offset, rows = segment.seqlen, previous = std::move(scratch.commitKV)](int accepted) {
        if (previous) previous(accepted);
        const auto &graph = *owner;
        std::vector<int> ok(graph.devices.size(), 0);
        const bool submitted = V41FullDispatch(graph, [&](int rank, int) {
            ok[rank] = FastllmCudaV41CommitVerifiedPrefix(
                static_cast<const FastllmV41VerifyCommitLayer *>(graph.commitDescriptors[rank].cudaData),
                graph.endLayer - graph.firstLayer, accepted, rows, offset, graph.batch);
        });
        AssertInFastLLM(submitted && std::all_of(ok.begin(), ok.end(), [](int v) { return v != 0; }),
                        "full verifier KV commit failed");
    };
    if (!g.attentionRuntime && !g.tensorParallel && g.devices.size() == 1) {
        if (segment.offset == 0 && segment.seqlen == g.batch &&
            g.batch == 6 && g.firstLayer == 0 && g.endLayer == int(g.layers.size()) &&
            std::all_of(g.layers.begin(),g.layers.end(),[](const auto &layer){return bool(layer);})) {
            scratch.stageNextPages = [owner,state=segment.state](std::shared_ptr<void> &storage, int pos) {
                if (owner->devices.front() != FastllmCudaGetDevice()) return false;
                auto pages = std::static_pointer_cast<V41RequestVerifyPages>(storage);
                if (!pages || !pages->Matches(*owner,*state)) {
                    pages = std::make_shared<V41RequestVerifyPages>();
                    pages->graph = owner; pages->state = state; pages->lease = state->storageLease;
                    pages->revision = state->storageRevision;
                    for (int l = 0; l < owner->endLayer; ++l)
                        pages->addresses.push_back(V41RequestVerifyPages::Addresses(*owner,*state,l));
                    void *slot = FastllmCudaV41CreateVerifyPages(pages->addresses.data(),owner->endLayer,owner->batch);
                    if (!slot) return false;
                    pages->slot.reset(slot,FastllmCudaV41DestroyVerifyPages); storage = pages;
                }
                return FastllmCudaV41StageVerifyPages(pages->slot.get(),pos);
            };
        }
        scratch.commitDeviceSpan = [owner](int device, const int *span) {
            if (device != owner->devices.front() || device != FastllmCudaGetDevice()) return false;
            return FastllmCudaV41CommitVerifiedBatch(
                static_cast<const FastllmV41VerifyCommitLayer *>(owner->commitDescriptors.front().cudaData),
                owner->endLayer - owner->firstLayer, span, 1, owner->batch);
        };
        scratch.commitGroup = owner.get();
        scratch.commitBatchKV = [owner, previous = std::move(scratch.commitBatchKV)](const std::vector<std::pair<int, int>> &spans) {
            if (previous) previous(spans);
            auto &graph = *owner;
            FastllmCudaSetDevice(graph.devices.front());
            graph.hostCommitSpans.resize(2 * spans.size());
            for (size_t i = 0; i < spans.size(); ++i) {
                graph.hostCommitSpans[2 * i] = spans[i].first;
                graph.hostCommitSpans[2 * i + 1] = spans[i].second;
            }
            FastllmCudaCopyFromHostToDevice(graph.commitSpans.cudaData,
                graph.hostCommitSpans.data(), graph.hostCommitSpans.size() * sizeof(int));
            AssertInFastLLM(FastllmCudaV41CommitVerifiedBatch(
                static_cast<const FastllmV41VerifyCommitLayer *>(graph.commitDescriptors.front().cudaData),
                graph.endLayer - graph.firstLayer, static_cast<const int *>(graph.commitSpans.cudaData), spans.size(), graph.batch),
                "full verifier batched KV commit failed");
        };
    }
    if (scratch.captureMain && g.packedMainHidden.cudaData) {
        scratch.packedMainOwner = owner.get();
        auto &view = scratch.packedMainHidden;
        const size_t width = g.packedMainHidden.dims.back();
        view.FakeFrom(g.packedMainHidden, size_t(segment.offset) * width * sizeof(uint16_t));
        view.Resize({1, segment.seqlen, int(width)});
    } else if (scratch.captureMain) {
        scratch.mainHidden.resize(targetLayers.size());
        for (size_t i = 0; i < targetLayers.size(); ++i)
            if (targetLayers[i] >= g.firstLayer && targetLayers[i] < g.endLayer)
                borrow(scratch.mainHidden[i], g.mainHidden.at(targetLayers[i]), segment.seqlen, segment.offset);
    }
}
} // namespace

void DeepSeekV41Model::BuildFullDecodeGraph(
    int batch, const std::vector<DeepSeekV41Segment> &segments, int firstLayer, int endLayer) {
    const bool distributedAttention = v41AttentionDP > 1;
    const bool attentionOnly = v41AttentionRankRuntime != nullptr;
    const bool verifying = !segments.empty() && std::all_of(segments.begin(), segments.end(),
        [](const DeepSeekV41Segment &s) { return s.spec && s.spec->deferWindow && s.spec->wantAllTokens; });
    const auto capturesDraftInput = [&](int layer) {
        return verifying && std::find(v41DsparkTargetLayerIds.begin(), v41DsparkTargetLayerIds.end(),
                                      layer) != v41DsparkTargetLayerIds.end();
    };
    const bool independentDecode = int(segments.size()) == batch &&
        std::all_of(segments.begin(), segments.end(),
            [](const DeepSeekV41Segment &s) { return s.seqlen == 1 && !s.spec; });
    if (firstLayer < 0 && executionMode == ExecutionMode::Hybrid &&
        V41DeviceMapCudaDeviceCount(deviceMap) > 1 && !V41DeviceMapUsesMultiCuda(deviceMap)) {
        std::vector<CudaPrefillStage> stages;
        if ((verifying || independentDecode) && BuildCudaPrefillStages(deviceMap, block_cnt, stages))
            for (const auto &stage : stages)
                BuildFullDecodeGraph(batch, segments, stage.firstLayer, stage.endLayer);
        return;
    }
    const bool pipeline = firstLayer >= 0;
    const int64_t graphKey = pipeline ? FastllmV41PipelineGraphKey(firstLayer, batch, verifying, segments.size() > 1) : attentionOnly ?
        batch * 3 + (v41AttentionLogicalRows == 1 ? 0 : v41AttentionLogicalRows <= 8 ? 1 : 2) : batch;
    if (!pipeline) { firstLayer = 0; endLayer = block_cnt; }
    const bool hybrid = executionMode == ExecutionMode::Hybrid &&
        (verifying || independentDecode) && FastllmCudaMoeCacheRequested();
    // Decode and segmented verification share the attention backend's batch
    // capacity, including packed FP4/FP8 KV. Causality comes from each request's
    // chunk metadata, not a smaller verification-only scheduling limit.
    if (batch < 1 || (!hybrid && executionMode != ExecutionMode::Gpu) ||
        (!verifying && std::any_of(segments.begin(), segments.end(),
            [](const DeepSeekV41Segment &seg) { return seg.spec != nullptr; })) ||
        (!hybrid && !distributedAttention && !attentionOnly && verifying &&
            (kvCacheDataType == DataType::FP4_E2M1 || kvCacheDataType == DataType::FP8_E4M3)) ||
        (!verifying && !v41CudaGraphWarmingUp) || head_dim_full != 512 || hc_mult != 4 || index_head_dim != 128)
        return;
    // Explicit reference/debug modes must retain their selected arithmetic.
    if (V41EnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT") ||
        V41EnvFlag("FASTLLM_DSV41_LEGACY_ATTN") || V41EnvFlag("FASTLLM_DSV41_LEGACY_INDEXER") ||
        V41EnvFlag("FASTLLM_DSV41_LEGACY_TOPK") || V41EnvFlag("FASTLLM_DSV41_LEGACY_ROTARY"))
        return;
    auto slot = v41CudaGraphSlots.find(verifying && !pipeline ? FastllmV41VerificationGraphKey(batch, segments.size() > 1,
        attentionOnly ? v41AttentionLogicalRows : 0) : graphKey);
    if (slot == v41CudaGraphSlots.end())
        return;
    auto state = V41GetCudaGraphState(slot->second);
    if (!state->captured || (!hybrid && !attentionOnly && !state->tensorParallel) ||
        (hybrid && !state->tensorParallel && state->devices.size() != 1) || state->fullDecode ||
        state->segments.size() != (size_t)block_cnt * 4)
        return;
    for (int i = firstLayer * 4; i < endLayer * 4; ++i) {
        if ((distributedAttention && i % 4 == 0) || (attentionOnly && i % 4 >= 2)) continue;
        if (state->segments[i].passthrough || state->segments[i].graphs.size() != state->devices.size())
            return;
    }
    for (int layer = firstLayer; layer < endLayer; ++layer) {
        if (attentionOnly) continue;
        const bool capturedMoe = state->segments[layer * 4 + 3].containsGpuMoe;
        const auto moeDevice = SelectMoeDeviceForLayer(layer);
        const bool hostMoe = V41DeviceSpecUsesType(moeDevice, "numa") || moeDevice == "cpu" ||
            (V41DeviceSpecUsesType(moeDevice, "disk") && DiskMoeV41Graph::Supports(weights[layer]));
        // A GPU expert placed on a different PP device is not a host-cache
        // layer. Reject before warming any child or changing captured operands.
        // Its ordinary peer transfer and remote compute remain segmented.
        if ((!hybrid && !capturedMoe) ||
            (hybrid && !capturedMoe && !hostMoe) ||
            (!distributedAttention && state->segments[layer * 4].kvOutputs.size() != state->devices.size())) return;
    }
    if (!hybrid) for (auto &table : engramTables)
        if (!std::static_pointer_cast<V41EngramTable>(table)->gpuTable) return;
    auto g = std::make_shared<V41FullGraph>();
    g->devices = state->devices;
    g->firstLayer = firstLayer; g->endLayer = endLayer;
    if (pipeline) {
        ApplyDeviceMap(deviceMap, firstLayer + 1, block_cnt);
        FastllmCudaSetDevice(g->devices.front());
    }
    g->hybrid = hybrid; g->tensorParallel = state->tensorParallel;
    g->rankMoeHandles.resize(g->devices.size());
    g->rankUntimedExecs.resize(g->devices.size(), nullptr);
    g->rankMoeFirstLayer.resize(g->devices.size(), -1);
    g->batch = batch; g->verifying = verifying;
    g->attentionOnly = attentionOnly;
    g->trackPositions = attentionOnly && !verifying && g->devices.size() == 1 &&
        V41EnvFlag("FASTLLM_DSV41_DEVICE_POSITIONS");
    if (attentionOnly) g->attentionLayers.resize(block_cnt);
    if (distributedAttention) {
        if (!v41AttentionParallel) return;
        auto &parallel = *v41AttentionParallel;
        auto &runtime = parallel.runtime;
        g->attentionRuntime = &runtime;
        std::vector<std::vector<DeepSeekV41Segment>> local;
        parallel::RowPartition rows(v41AttentionDP, {});
        if (!V41FullPartition(*g, segments, local, rows)) return;
        g->attentionStates.resize(g->devices.size());
        g->attentionGraphs.resize(g->devices.size());
        // The allocation pinning transaction is process-wide. Build rank
        // compute pieces sequentially, always on that GPU's fixed worker.
        for (int r = 0; r < int(g->devices.size()); ++r) {
            if (local[r].empty()) continue;
            runtime.Run([&](int rank, int) {
                if (rank != r) return;
                auto &model = *parallel.models[rank];
                model.v41AttentionLogicalRows = batch;
                const int count = rows.localToGlobal[runtime.Layout().AttentionOwner(rank)].size();
                model.BuildFullDecodeGraph(count, local[rank]);
                const int64_t key = verifying ? FastllmV41VerificationGraphKey(count, local[rank].size() > 1, batch) : int64_t(count) * 3 + (batch == 1 ? 0 : batch <= 8 ? 1 : 2);
                auto found = model.v41CudaGraphSlots.find(key);
                if (found == model.v41CudaGraphSlots.end()) return;
                auto state = V41GetCudaGraphState(found->second);
                if (!state->fullDecode) return;
                g->attentionStates[rank] = state;
                g->attentionGraphs[rank] = std::static_pointer_cast<V41FullGraph>(state->fullDecode);
            });
            if (!g->attentionGraphs[r]) return;
        }
        size_t transferWidth = size_t(embed_dim) * sizeof(uint16_t) +
            size_t(2 * hc_mult + hc_mult * hc_mult) * sizeof(float);
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
        if (!hybrid && !pipeline && (!verifying || V41VerifyOwnerFfnEnabled()) &&
            runtime.Layout().attentionTP == 1 && g->devices.size() == 8 &&
            V41EnvFlag("FASTLLM_DSV41_LOCAL_EP_GRAPH") &&
            norm_topk_prob && num_experts_per_tok == 6 && embed_dim == 5120 &&
            std::all_of(g->attentionGraphs.begin(), g->attentionGraphs.end(), [](auto &x) {
                return !x || x->batch <= FastllmDsv41MegaMaxRows / 8;
            })) {
            g->ownerExperts.resize(block_cnt);
            for (int l = firstLayer; l < endLayer; ++l) {
                g->ownerExperts[l] = MultiCudaDsv41SharedLayers(weights[l][2], g->devices);
                if (g->ownerExperts[l].size() != g->devices.size()) {
                    g->ownerExperts.clear();
                    break;
                }
                for (const char *suffix : {".hc_ffn_fn", ".hc_ffn_scale", ".hc_ffn_base",
                        ".ffn_norm.weight", ".ffn.gate.weight", ".ffn.gate.bias"}) {
                    auto &w = weight["layers." + std::to_string(l) + suffix];
                    if (!w.IsTensorParallelReplicated()) { g->ownerExperts.clear(); break; }
                }
                if (g->ownerExperts.empty()) break;
            }
            if (!g->ownerExperts.empty()) {
                g->ownerFfn.resize(g->devices.size());
                runtime.Run([&](int rank, int device) {
                    if (!g->attentionGraphs[rank]) return;
                    auto a = std::make_unique<V41OwnerFfn>();
                    const int m = g->attentionGraphs[rank]->batch;
                    if (m >= 16) for (int l = firstLayer; l < endLayer; ++l)
                        FastllmCudaHcSplitWeight(V41FullLocal(weight["layers." + std::to_string(l) + ".hc_ffn_fn"], device));
                    auto reserve = [&](Data &d, DataType type, std::vector<int> dims) {
                        d.dataType = type; d.UpdateUnitSize(); d.Resize(dims);
                        d.ToDevice(CUDA, {device}, false); d.Allocate(false);
                    };
                    reserve(a->residual, BFLOAT16, {1,m,4,embed_dim});
                    reserve(a->post, FLOAT32, {1,m,4});
                    reserve(a->comb, FLOAT32, {1,m,4,4}); reserve(a->input, BFLOAT16, {1,m,embed_dim});
                    reserve(a->logits, FLOAT32, {m,384}); reserve(a->ids, INT32, {m,6});
                    reserve(a->scores, FLOAT32, {m,6});
                    const bool useCluster = FastllmV41HcEnabledAtRows("FASTLLM_DSV41_HC_CLUSTER", m);
                    if (useCluster || FastllmV41HcEnabledAtRows("FASTLLM_DSV41_HC_OVERLAP", m))
                        reserve(a->mixScratch, FLOAT32, {m, 8, 25});
                    if (a->mixScratch.cudaData && embed_dim == 5120 && hc_mult == 4 &&
                        useCluster && (!verifying || m == 6))
                        a->clusterMix = FastllmCudaV41CreateOwnerStream(m);
                    g->ownerFfn[rank] = std::move(a);
                });
                transferWidth = size_t(embed_dim*hc_mult)*sizeof(uint16_t) + hc_mult*sizeof(float);
            }
        }
#endif
        g->rowTransfer.reset(new parallel::CudaRowTransfer(runtime, rows));
        g->rowTransfer->Reserve(transferWidth);
        if (!verifying) {
            g->padding = segments;
            // Retained scratch must not count as live KV load when placing
            // subsequent warmup batches or real requests on attention groups.
            auto &requests = parallel.requests;
            requests.erase(std::remove_if(requests.begin(), requests.end(), [&](const auto &weak) {
                const auto request = weak.lock();
                return !request || std::any_of(segments.begin(), segments.end(),
                    [&](const auto &s) { return s.state == request; });
            }), requests.end());
        }
    }
#ifdef USE_NUMAS
    if (hybrid && !weights.empty() && weights[0].size() > 2 && weights[0][2]) {
        HostMoePlacementScope placement(weights[0][2]);
        g->rankCoordinatorCpus = GetNumasCudaWorkerCpuSets(g->devices);
        if (!g->rankCoordinatorCpus.empty()) g->coordinatorCpus = g->rankCoordinatorCpus.front();
    }
#endif
    V41FullCallerAffinity affinity(g->coordinatorCpus);
    g->hostRequests.resize(g->devices.size());
    g->hostPlans.resize(g->devices.size());
    // Reserve the configured context before capture. Growing these operands
    // during a live request would invalidate graph addresses and require a
    // warm replay that can modify its KV state. Both short and long contexts
    // retain the same graph throughout the configured serving window.
    int contextCapacity = max_positions;
    if (tokensLimit > 0 && (contextCapacity <= 0 || tokensLimit < contextCapacity))
        contextCapacity = tokensLimit;
    // Indexer launch grids and workspace sizes follow this bound. Replay
    // validates each request before touching KV and can use the eager path.
    g->maxKeys = std::max(index_topk, contextCapacity > 0 ? contextCapacity : 131072);
    g->pieces = std::make_shared<DeepSeekV41CudaGraphState>();
    g->pieces->PrepareDevices(g->devices, g->tensorParallel);
    auto &ws = *state->workspace;
    ScopedTpDispatch dispatch(g->tensorParallel, g->devices);
    const int original = FastllmCudaGetDevice();
    auto alloc = [&](Data &data, DataType dtype, const std::vector<int> &dims) {
        V41AllocLike(data, dtype, dims, ws.hiddenStates, g->devices);
    };

#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    // Hidden features preceding an Engram must include that layer's embedding.
    // The ordinary gather handles such checkpoints and partial pipeline graphs.
    const bool ownerFeatures = verifying && !g->ownerFfn.empty() && hc_mult == 4 &&
        firstLayer == 0 && endLayer == block_cnt && !v41DsparkTpDevices.empty() &&
        v41DsparkTpDevices.front() == g->devices.front() && !v41DsparkTargetLayerIds.empty() &&
        std::all_of(v41DsparkTargetLayerIds.begin(), v41DsparkTargetLayerIds.end(), [&](int l) {
            return l > firstLayer && l < endLayer &&
                std::count(v41DsparkTargetLayerIds.begin(), v41DsparkTargetLayerIds.end(), l) == 1 &&
                std::find(engram_layer_ids.begin(), engram_layer_ids.end(), l) == engram_layer_ids.end();
        }) && FastllmCudaPeerAccessInit(g->devices);
    if (ownerFeatures) {
        FastllmCudaSetDevice(g->devices.front());
        auto &packed = g->packedMainHidden;
        packed.dataType = BFLOAT16; packed.UpdateUnitSize();
        packed.Resize({1, batch, embed_dim * int(v41DsparkTargetLayerIds.size())});
        packed.ToDevice(CUDA, {g->devices.front()}, false); packed.Allocate(false);
    }
#endif
    // LLMSamplingBlock can merge per-rank candidates with one host read.
    // Other sampling policies and logit requests still consume the full logits.
    g->captureGreedy = !attentionOnly && !verifying && distributedAttention && g->tensorParallel &&
        endLayer == block_cnt && V41EnvFlag("FASTLLM_DSV41_CAPTURE_GREEDY") &&
        ws.hiddenStates.multiDeviceData && ws.hiddenStates.IsTensorParallelReplicated();
    if (g->captureGreedy) {
        alloc(g->greedyIds, INT32, {batch});
        alloc(g->greedyScores, FLOAT32, {batch});
        if (V41EnvFlag("FASTLLM_DSV41_PUBLISH_GREEDY")) {
            for (int device : g->devices) {
                FastllmCudaSetDevice(device);
                void *host = FastllmCudaHostMalloc(size_t(batch) * (sizeof(int) + sizeof(float)));
                if (!host) break;
                g->greedyHost.emplace(device, host);
                if (void *mapped = FastllmCudaHostDevicePointer(host)) g->greedyMapped.emplace(device, mapped);
                else break;
            }
            FastllmCudaSetDevice(original);
            if (g->greedyMapped.size() != g->devices.size()) g->greedyMapped.clear();
        }
    }

    // Keep BF16 embedding outputs and token operands at fixed addresses. The
    // ordinary/input-embedding and other precision paths retain their frontend.
    auto &embeddingWeight = weight["embed.weight"];
    g->capturesInput = !attentionOnly && firstLayer == 0 && (verifying || (hybrid && !g->tensorParallel)) && embeddingWeight.dataType == DataType::BFLOAT16 &&
        embeddingWeight.dataDevice == DataDevice::CUDA && embeddingWeight.cudaData &&
        !embeddingWeight.multiDeviceData && embeddingWeight.dims.size() == 2 &&
        embeddingWeight.dims[1] == embed_dim &&
        GetPointerDeviceId(embeddingWeight.cudaData) == g->devices.front();
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    // The local expert graph already established all peer access. Each
    // request owner reads the immutable embedding and builds its four HC
    // channels directly, including the first layer's constant pre-mix.
    const bool multiVerify = verifying && segments.size() > 1;
    if ((!verifying || multiVerify) && firstLayer == 0 && !g->ownerFfn.empty() && hc_mult == 4 &&
        embed_dim % 4 == 0 && V41EnvFlag("FASTLLM_DSV41_OWNER_EMBEDDING") &&
        std::find(engram_layer_ids.begin(),engram_layer_ids.end(),0) == engram_layer_ids.end() &&
        GetCudaEmbedding() && !GetLowMemMode() && !embeddingWeight.lockInCPU &&
        (embeddingWeight.dataType == BFLOAT16 || embeddingWeight.dataType == FLOAT32) &&
        embeddingWeight.dataDevice == CUDA && !embeddingWeight.multiDeviceData && embeddingWeight.cudaData &&
        embeddingWeight.dims.size() == 2 && embeddingWeight.dims[1] == embed_dim &&
        GetPointerDeviceId(embeddingWeight.cudaData) == g->devices.front()) {
        g->capturesInput = true;
        g->embeddingWeightAddress = embeddingWeight.cudaData;
        g->ownerInputIds.resize(g->devices.size()); g->ownerHostIds.resize(g->devices.size());
        g->attentionRuntime->Run([&](int rank, int device) {
            if (!g->attentionGraphs[rank]) return;
            const int rows = g->attentionGraphs[rank]->batch;
            auto ids = std::make_unique<Data>(INT32, std::vector<int>{rows});
            // Capture runs one warm execution before request metadata is uploaded.
            ids->ToDevice(CUDA,{device},false); ids->Allocate(true);
            g->ownerInputIds[rank] = std::move(ids); g->ownerHostIds[rank].resize(rows);
        });
    }
#endif
    if (g->capturesInput && g->ownerInputIds.empty()) {
        g->embeddingWeightAddress = embeddingWeight.cudaData;
        alloc(g->embedding, DataType::BFLOAT16, {1, batch, 1, embed_dim});
        g->hostTokens.assign(batch, 0.f);
        if (!verifying) alloc(g->inputIds, DataType::FLOAT32, {1, batch});
    }
    if (hybrid && g->tensorParallel) {
        alloc(g->routedOutput, DataType::BFLOAT16, {batch, embed_dim});
    }
    if (hybrid && !g->tensorParallel && batch == 1) alloc(g->deferredHcMix, DataType::FLOAT32, {2 * block_cnt, 32});
    std::vector<int> hcStart, hcJoin;
    const int blockSize = std::max(1, candidate_block_size);
    g->layers.resize(block_cnt);
    if (!distributedAttention) {
    alloc(g->topk, DataType::INT32, {1, batch, index_topk});
    alloc(g->candidates, DataType::INT8, {1, batch, (g->maxKeys + blockSize - 1) / blockSize});
    alloc(g->candidateBlocks, DataType::INT32, {batch, (g->maxKeys + blockSize - 1) / blockSize});
    alloc(g->candidateCounts, DataType::INT32, {batch});
    for (int l = firstLayer; l < endLayer; ++l) {
        auto layer = std::make_unique<V41FullLayer>();
        auto &c = layer->config;
        c.batch = batch;
        if (kvCacheDataType == DataType::FP4_E2M1 || kvCacheDataType == DataType::FP8_E4M3) {
            const bool fp4 = kvCacheDataType == DataType::FP4_E2M1;
            c.ringMode = 1; c.ringRowBytes = 528;
            c.cmpMode = fp4 ? 3 : 1; c.cmpRowBytes = fp4 ? 288 : 528;
            c.indexMode = fp4 ? 2 : 1; c.indexRowBytes = fp4 ? 68 : 132;
        }
        c.verifying = verifying;
        c.segmentedVerify = verifying && segments.size() > 1;
        c.heads = num_attention_heads / g->devices.size();
        c.indexHeads = index_n_heads;
        c.ratio = compress_ratios[l];
        c.window = window_size;
        c.maxKeys = g->maxKeys;
        c.topK = index_topk;
        c.blockSize = blockSize;
        c.candidateTopK = std::max(1, candidate_topk_blocks);
        c.kvSource = c.ratio && isKvSource[l];
        c.indexSource = c.ratio && isIndexSource[l];
        c.writeCandidates =
            c.indexSource && l == candidate_source_layer_id && candidate_topk_blocks > 0;
        c.useCandidates =
            c.indexSource && candidate_source_layer_id >= 0 && l > candidate_source_layer_id;
        c.ropeDim = qk_rope_head_dim;
        c.ropeBase = c.ratio ? compress_rope_theta : rope_base;
        c.originalSeqLen = c.ratio ? rope_scaling_original_max_position_embeddings : 0;
        c.ropeFactor = rope_factor;
        c.betaFast = rope_scaling_beta_fast;
        c.betaSlow = rope_scaling_beta_slow;
        c.normEps = rms_norm_eps;
        c.softmaxScale = 1.f / std::sqrt(float(head_dim_full));
        layer->source = kvSourceOf[l];
        if (verifying || c.ringMode) {
            alloc(layer->savedKV, c.ringMode ? DataType::INT8 : DataType::BFLOAT16,
                  {1, batch, c.ringMode ? c.ringRowBytes : 512});
            if (verifying && c.kvSource) {
                alloc(layer->savedRawKV, DataType::FLOAT32, {1, 2 * batch, 512});
                if (c.ratio == 2)
                    alloc(layer->savedRawScore, DataType::FLOAT32, {1, 2 * batch, 512});
            }
        }
        if (c.kvSource) {
            alloc(layer->joined, DataType::FLOAT32, {batch, c.ratio, 512});
            if (c.ratio == 2)
                alloc(layer->scores, DataType::FLOAT32, {batch, c.ratio, 512});
            alloc(layer->latents, DataType::BFLOAT16, {1, batch, 512});
            if (c.indexSource)
                alloc(layer->keys, DataType::BFLOAT16, {1, batch, 128});
        }
        g->layers[l] = std::move(layer);
    }
    if (!V41FullReserve(*g, segments))
        return;
    }
    const auto snapshots = V41SnapshotSegments(segments, engram_max_ngram_size);
    for (int i = 0; i < (int)engram_layer_ids.size(); ++i) {
        if (engram_layer_ids[i] < firstLayer || engram_layer_ids[i] >= endLayer) continue;
        auto e = std::make_unique<V41FullEngram>();
        e->index = i;
        e->table = std::static_pointer_cast<V41EngramTable>(engramTables[i]);
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
        if (!g->verifying && e->table->gpuTable && V41EnvFlag("FASTLLM_DSV41_DIRECT_HASH")) {
            const auto &primes = engramMeta.primes[i], &offsets = engramMeta.offsets[i];
            for (size_t col = 0; col < primes.size(); ++col) {
                if (primes[col] < 2 || offsets[col] < 0 || primes[col] > e->table->rows ||
                    offsets[col] > e->table->rows - primes[col]) { e->buckets.clear(); break; }
                const uint64_t prime = uint64_t(primes[col]);
                uint64_t reciprocal = 0;
#ifdef __SIZEOF_INT128__
                reciprocal = uint64_t((__uint128_t(1) << 64) / prime);
#endif
                e->buckets.push_back({prime, uint64_t(offsets[col]), reciprocal});
            }
        }
#endif
        bool dead = false;
        V41BuildEngramInputs(engramMeta, i, engram_max_ngram_size, engram_n_heads, snapshots, batch,
                             e->hostRows, e->hostMask, dead);
        if (!g->verifying) {
            alloc(e->rows, DataType::INT32, {1, (int)e->hostRows.size() * 2});
            alloc(e->mask, DataType::FLOAT32, {1, batch});
        }
        alloc(e->gathered, DataType::BFLOAT16,
              {1, batch, (int)e->hostRows.size() / batch * e->table->dim});
        if (!e->table->gpuTable)
            GatherEngramRows(engram_layer_ids[i], e->hostRows, batch, e->hostGathered);
        const auto &t = *e->table;
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
        // The N-gram table is peer-readable and the projection keeps a full
        // weight replica on each rank. Select each owner's request metadata
        // before the lookup, without publishing hidden states at these layers.
        if (!g->ownerFfn.empty() && t.gpuTable && engram_layer_ids[i] > firstLayer &&
            !capturesDraftInput(engram_layer_ids[i]) &&
            !V41EnvFlag("FASTLLM_DSV41_DISABLE_OWNER_ENGRAM")) {
            const auto prefix = "layers." + std::to_string(engram_layer_ids[i]) + ".engram.";
            auto &w = weight[prefix + "wkv.weight"];
            const int heads = e->hostRows.size() / batch;
            const int k = heads * t.dim;
            if (quantizedLinearNames.count(prefix + "wkv.weight") && w.IsTensorParallelReplicated() &&
                w.dims == std::vector<int>{(hc_mult + 1) * embed_dim, k} &&
                w.dataType == FP8_E4M3 && w.blockK == 32 && w.blockM == 32 && k % 128 == 0 &&
                weight[prefix + "q_weight"].IsTensorParallelReplicated() &&
                weight[prefix + "k_weight"].IsTensorParallelReplicated()) {
                e->owners.resize(g->devices.size());
                g->attentionRuntime->Run([&](int rank, int device) {
                    if (!g->attentionGraphs[rank]) return;
                    const int m = g->attentionGraphs[rank]->batch;
                    auto o = std::make_unique<V41OwnerEngram>();
                    auto reserve = [&](Data &d, DataType type, std::vector<int> dims) {
                        d.dataType = type; d.UpdateUnitSize(); d.Resize(dims);
                        d.ToDevice(CUDA, {device}, false); d.Allocate(false);
                    };
                    reserve(o->rows, INT32, {m, heads * 2});
                    reserve(o->mask, FLOAT32, {1, m});
                    reserve(o->gathered, BFLOAT16, {1, m, k});
                    reserve(o->kv, BFLOAT16, {1, m, w.dims[0]});
                    reserve(o->quantized, BFLOAT16, {1, m, k});
                    e->owners[rank] = std::move(o);
                });
            }
        }
#endif
        if (hybrid && V41CanPrefetchEngramRows(t)) {
            FastllmCudaSetDevice(g->devices.front());
            // Unsupported devices keep the existing synchronous gather path.
            if (FastllmCudaV41CreateHostCopy(V41FullLocal(e->gathered, g->devices.front()).cudaData, e->gathered.GetBytes(),
                    &e->asyncData, &e->asyncReady, &e->asyncGraph)) {
                std::memcpy(e->asyncData, e->hostGathered.cpuData, e->gathered.GetBytes());
                FastllmCudaCopyFromHostToDevice(V41FullLocal(e->gathered, g->devices.front()).cudaData, e->asyncData, e->gathered.GetBytes());
            }
        }
        g->engrams[engram_layer_ids[i]] = std::move(e);
    }
    size_t metadataBytes = block_cnt * batch * sizeof(FastllmV41FullDecodeRequest);
    if (g->verifying) {
        for (const auto &entry : g->engrams)
            metadataBytes += entry.second->hostRows.size() * sizeof(int64_t) + batch * sizeof(float);
        if (g->capturesInput && g->ownerInputIds.empty()) metadataBytes += batch * sizeof(float);
    }
    alloc(g->requests, DataType::INT8, {1, (int)metadataBytes});
    if (g->verifying) {
        size_t offset = block_cnt * batch * sizeof(FastllmV41FullDecodeRequest);
        auto view = [&](Data &data, DataType dtype, const std::vector<int> &dims) {
            auto bind = [&](Data &local, int device) {
                local.FakeFrom(V41FullLocal(g->requests, device), offset);
                local.dataType = dtype;
                local.UpdateUnitSize();
                local.Resize(dims);
                local.expansionDims = dims;
                local.expansionSize = local.Count(0);
                local.expansionBytes = local.GetBytes();
            };
            if (g->tensorParallel) {
                data.multiDeviceData = true;
                for (int device : g->devices) {
                    auto *local = new Data();
                    bind(*local, device);
                    data.multiDeviceDatas[device] = local;
                }
                V41SyncRootFromReplica(data, g->devices);
            } else bind(data, g->devices.front());
            offset += data.GetBytes();
        };
        // All int64 rows precede float masks, preserving alignment for odd M.
        for (auto &entry : g->engrams)
            view(entry.second->rows, DataType::INT32, {1, (int)entry.second->hostRows.size() * 2});
        for (auto &entry : g->engrams)
            view(entry.second->mask, DataType::FLOAT32, {1, batch});
        if (g->capturesInput && g->ownerInputIds.empty()) view(g->inputIds, DataType::FLOAT32, {1, batch});
        AssertInFastLLM(offset == metadataBytes, "full verify metadata layout mismatch");
    }
    if (g->capturesInput && g->ownerInputIds.empty()) {
        FastllmCudaSetDevice(g->devices.front());
        FastllmCudaCopyFromHostToDevice(V41FullLocal(g->inputIds, g->devices.front()).cudaData,
                                      g->hostTokens.data(), batch * sizeof(float));
    }
    if (verifying && ws.dsparkPreMix.dims != std::vector<int>({1, batch, hc_mult})) {
        alloc(ws.dsparkPreMix, DataType::FLOAT32, {1, batch, hc_mult});
        std::vector<float> uniform(size_t(batch) * hc_mult, 1.0f / hc_mult);
        V41FullDispatch(*g, [&](int, int device) {
            FastllmCudaCopyFromHostToDevice(V41FullLocal(ws.dsparkPreMix, device).cudaData,
                uniform.data(), uniform.size() * sizeof(float));
        });
    }
    // Lazy verification capture runs after this call's embedding. The warm
    // Engram children modify hiddenStates; restore its contents (not address)
    // before replay, including the fallback path.
    Data initialHidden;
    if (verifying) V41Assign(initialHidden, ws.hiddenStates, g->devices);
    bool poolOpen = false;
    try {
        // Expert children are built between captured attention segments. Their
        // private benchmark must finish before the shared allocation transaction
        // opens; it is intentionally forbidden while that pool is active.
        if (hybrid && !attentionOnly) {
            for (int l = firstLayer; l < endLayer; ++l) {
                if (state->segments[l * 4 + 3].containsGpuMoe || weights[l][2]->isDiskWeight) continue;
                const size_t owner = g->tensorParallel ? size_t(l) * g->devices.size() / block_cnt : 0;
                FastllmCudaSetDevice(g->devices[owner]);
                auto routed = weights[l]; routed[0] = routed[1] = nullptr;
                FastllmCudaPrepareNativeMoeKernelPlan(routed.data(), routed.size(), batch, num_experts_per_tok);
            }
            FastllmCudaSetDevice(g->devices.front());
        }
        std::vector<std::vector<const void *>> children(g->devices.size());
        auto capture = [&](const std::function<void()> &body) {
            std::vector<size_t> allocations;
            {
                FastllmCudaAllocationRecorder recorder(g->hybrid ? &allocations : nullptr);
                body(); // Warm GEMM plans and persistent operands before capture.
            }
            for (int d : g->devices)
                FastllmCudaSyncDevice(d);
            // GGUF GEMM releases its dequantization scratch through an event.
            // Retire that event and reserve capture-safe temporary storage.
            if (g->hybrid && !FastllmCudaReserveGraphWorkspace(allocations))
                throw std::runtime_error("full graph temporary workspace");
            if (!poolOpen) {
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                if (!FastllmCudaGraphMemoryPoolBegin())
                    throw std::runtime_error("full graph memory pool");
                poolOpen = true;
            }
            if (!V41CaptureGraphSegment(*g->pieces, body))
                throw std::runtime_error("full graph child capture " + std::to_string(g->pieces->segments.size()) +
                    (attentionOnly ? " (attention)" : " (main)"));
            const auto &s = g->pieces->segments.back();
            for (size_t rank = 0; rank < children.size(); ++rank)
                children[rank].push_back(s.graphs[rank]);
        };
        if (!g->ownerInputIds.empty()) {
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
            // Each rank consumes its own token IDs and private Engram rows. The
            // preceding sampling fence protects the pinned source across replays.
            if ((multiVerify || !g->greedyMapped.empty()) &&
                std::all_of(g->engrams.begin(), g->engrams.end(), [&](const auto &entry) {
                    return entry.second->owners.size() == g->devices.size();
                })) {
                g->inputPackets.resize(g->devices.size());
                for (size_t r = 0; r < g->devices.size(); ++r) if (g->ownerInputIds[r]) {
                    FastllmCudaSetDevice(g->devices[r]);
                    auto packet = std::make_unique<V41OwnerInputPacket>();
                    auto &p = *packet;
                    size_t bytes = g->ownerInputIds[r]->GetBytes();
                    auto &owner = *g->attentionGraphs[r];
                    if (owner.trackPositions && !owner.capturesInput && owner.engrams.empty() &&
                        V41FullLocal(owner.requests, g->devices[r]).cudaData &&
                        V41EnvFlag("FASTLLM_DSV41_PACKET_POSITIONS")) {
                        p.positionOffset = bytes;
                        bytes += owner.batch * sizeof(int);
                    }
                    for (auto &entry : g->engrams) {
                        auto &o = *entry.second->owners[r];
                        o.rowOffset = (bytes + 7) & ~size_t(7);
                        o.maskOffset = o.rowOffset + o.rows.GetBytes();
                        bytes = o.maskOffset + o.mask.GetBytes();
                    }
                    p.device.dataType = FLOAT32; p.device.UpdateUnitSize(); p.device.Resize({int(bytes / 4)});
                    p.device.ToDevice(CUDA, {g->devices[r]}, false); p.device.Allocate();
                    p.host = FastllmCudaHostMalloc(bytes);
                    if (!p.host) { g->inputPackets.clear(); break; }
                    std::memset(p.host, 0, bytes);
                    for (auto &entry : g->engrams) {
                        auto &o = *entry.second->owners[r];
                        o.packetMask.FakeFrom(p.device, o.maskOffset);
                        o.packetMask.Resize(o.mask.dims);
                    }
                    g->inputPackets[r] = std::move(packet);
                }
                FastllmCudaSetDevice(original);
            }
#endif
            capture([&] {
                g->attentionRuntime->Enqueue([&](int rank,int) {
                    if (!g->ownerInputIds[rank]) return;
                    auto &local = *g->attentionStates[rank]->workspace;
                    const int *ids = static_cast<const int *>(g->ownerInputIds[rank]->cudaData);
                    const int *positions = nullptr;
                    FastllmV41FullDecodeRequest *requests = nullptr;
                    int layers = 0;
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                    if (!g->inputPackets.empty()) {
                        auto &p = *g->inputPackets[rank];
                        AssertInFastLLM(FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
                            p.device.cudaData, p.host, p.device.GetBytes()), "owner input packet copy failed");
                        ids = static_cast<const int *>(p.device.cudaData);
                        if (p.positionOffset) {
                            auto &owner = *g->attentionGraphs[rank];
                            positions = ids + p.positionOffset / sizeof(int);
                            requests = static_cast<FastllmV41FullDecodeRequest *>(
                                V41FullLocal(owner.requests, g->devices[rank]).cudaData) + size_t(owner.firstLayer) * owner.batch;
                            layers = owner.endLayer - owner.firstLayer;
                        }
                    }
#endif
                    AssertInFastLLM(FastllmCudaV41OwnerEmbedding(embeddingWeight.cudaData,
                        embeddingWeight.dataType == FLOAT32, ids,
                        local.hiddenStates.cudaData, static_cast<float *>(local.preMix.cudaData),
                        g->attentionGraphs[rank]->batch, embed_dim, embeddingWeight.dims[0], positions, requests, layers), "owner embedding capture failed");
                });
            });
        } else if (g->capturesInput) {
            capture([&] {
                V41FullDispatch(*g, [&](int rank, int device) {
                    auto &local = V41FullLocal(g->embedding, device);
                    if (rank == 0)
                        AssertInFastLLM(FastllmCudaEmbeddingDirect(
                            V41FullLocal(g->inputIds, device), embeddingWeight, local),
                            "full verify embedding failed");
                    if (g->tensorParallel) FastllmNcclBroadcastFrom(
                        local.cudaData, local.cudaData, batch * embed_dim, DataType::BFLOAT16,
                        g->devices.front(), device);
                    const int rowBytes = embed_dim * sizeof(uint16_t);
                    FastllmCudaRepeat(local.cudaData, V41FullLocal(ws.hiddenStates, device).cudaData,
                        batch, hc_mult, rowBytes, hc_mult * rowBytes, rowBytes, rowBytes);
                });
            });
        }
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
        std::vector<std::vector<std::pair<int, int>>> engramForks(children.size());
        std::vector<int> previousAttentionInput(children.size(), -1);
#endif
        int candidateWorkspaceRatio = -1;
        std::vector<std::map<bool, void *>> attentionScratch(g->devices.size());
        for (int l = firstLayer; l < endLayer; ++l) {
            const size_t attentionStart = children[0].size();
            size_t attentionInputChild = attentionStart;
            const std::string prefix = "layers." + std::to_string(l);
            auto engram = g->engrams.find(l);
            if (engram != g->engrams.end()) {
                auto &e = *engram->second;
                V41FullDispatch(*g, [&](int rank, int device) {
                    V41FullUploadRank(*g, segments, rank, device);
                });
                if (e.asyncGraph) children[0].push_back(e.asyncGraph);
                capture([&] {
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                    if (!e.owners.empty()) {
                        g->attentionRuntime->Enqueue([&](int rank, int device) {
                            if (!e.owners[rank]) return;
                            auto &o = *e.owners[rank];
                            auto &w = V41FullLocal(weight[prefix + ".engram.wkv.weight"], device);
                            const size_t heads = e.hostRows.size() / batch;
                            const int64_t *rows = static_cast<const int64_t *>(o.rows.cudaData);
                            Data *mask = &o.mask;
                            if (g->inputPackets.empty()) {
                                g->rowTransfer->Select(rank, V41FullLocal(e.rows, device).cudaData, o.rows.cudaData,
                                                       heads * sizeof(int64_t));
                                g->rowTransfer->Select(rank, V41FullLocal(e.mask, device).cudaData, o.mask.cudaData, sizeof(float));
                            } else {
                                auto *data = static_cast<const uint8_t *>(g->inputPackets[rank]->device.cudaData);
                                rows = reinterpret_cast<const int64_t *>(data + o.rowOffset);
                                mask = &o.packetMask;
                            }
                            AssertInFastLLM(FastllmCudaV41GatherEngramDevice(e.table->gpuTable,
                                rows, o.rows.Count(0) / 2, o.gathered),
                                "owner-local Engram lookup failed");
                            o.nativeProjection = FastllmCudaTryBlock32Mxfp8Sm103(o.gathered, w, o.kv, o.workspace);
                            if (!o.nativeProjection) {
                                AssertInFastLLM(FastllmCudaDeepSeekV41QuantizeActivation(o.gathered, o.quantized) &&
                                    FastllmCudaBFloat16FP8ReplicatedRowSlice(o.quantized, w, o.kv, 0, w.dims[0]),
                                    "owner-local Engram projection failed");
                            }
                            auto &hidden = g->attentionStates[rank]->workspace->hiddenStates;
                            AssertInFastLLM(FastllmCudaDeepSeekV41EngramApply(hidden, o.kv,
                                V41FullLocal(weight[prefix + ".engram.q_weight"], device),
                                V41FullLocal(weight[prefix + ".engram.k_weight"], device), mask,
                                rms_norm_eps, 1e-6f), "owner-local Engram gate failed");
                        });
                        return;
                    }
#endif
                    if (e.asyncGraph && g->tensorParallel) V41FullDispatch(*g, [&](int, int device) {
                        auto &local = V41FullLocal(e.gathered, device);
                        FastllmNcclBroadcastFrom(local.cudaData, local.cudaData, local.Count(0),
                            local.dataType, g->devices.front(), device);
                    });
                    if (e.table->gpuTable) V41FullDispatch(*g, [&](int, int device) {
                        bool ok = FastllmCudaV41GatherEngramDevice(
                            e.table->gpuTable,
                            static_cast<const int64_t *>(
                                V41FullLocal(e.rows, device).cudaData),
                            e.hostRows.size(), V41FullLocal(e.gathered, device));
                        AssertInFastLLM(ok, "full graph Engram gather failed");
                    });
                    LinearWithActivationQuant(e.gathered, prefix + ".engram.wkv.weight", e.kv, true,
                                              &e.quantized, false,
                                              &e.reduction);
                    V41EngramApply(ws.hiddenStates, e.kv, weight[prefix + ".engram.q_weight"],
                                   weight[prefix + ".engram.k_weight"], &e.mask, rms_norm_eps);
                });
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                // The selected rows and layer-owned WKV buffers are independent of the residual.
                if (verifying && !e.asyncGraph && e.owners.size() == children.size())
                    for (size_t rank = 0; rank < children.size(); ++rank)
                        if (e.owners[rank] && g->attentionGraphs[rank]->batch == 6 &&
                            e.owners[rank]->nativeProjection && previousAttentionInput[rank] >= 0)
                            engramForks[rank].emplace_back(int(children[rank].size()) - 1, previousAttentionInput[rank]);
#endif
            }
            if (capturesDraftInput(l) && !g->packedMainHidden.cudaData) {
                capture([&] {
                    V41HcApplyPre(ws.hiddenStates, ws.dsparkPreMix, g->mainHidden[l]);
                });
            }
            // Delayed-pre normalization consumes the preceding layer's pre.
            // This attention mix is first read by the attention residual below.
            if (hybrid) { hcStart.push_back(children[0].size()); hcJoin.push_back(-1); }
            if (distributedAttention) {
                auto &runtime = *g->attentionRuntime;
                const auto &topology = runtime.Layout();
                bool selectInputs = true;
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                selectInputs = g->ownerFfn.empty() || (l == firstLayer && g->ownerInputIds.empty()) ||
                    (engram != g->engrams.end() && engram->second->owners.empty());
#endif
                if (selectInputs) capture([&] {
                    runtime.Enqueue([&](int rank, int device) {
                        if (!g->attentionGraphs[rank]) return;
                        auto &local = *g->attentionStates[rank]->workspace;
                        g->rowTransfer->Select(rank, V41FullLocal(ws.hiddenStates, device).cudaData,
                            local.hiddenStates.cudaData, size_t(embed_dim) * hc_mult * sizeof(uint16_t));
                        g->rowTransfer->Select(rank, V41FullLocal(l ? ws.ffnPre : ws.preMix, device).cudaData,
                            local.preMix.cudaData, hc_mult * sizeof(float));
                    });
                });
                for (size_t rank = 0; rank < children.size(); ++rank) if (g->attentionGraphs[rank]) {
                    const auto &parts = g->attentionGraphs[rank]->attentionLayers.at(l);
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                    auto &list = children[rank];
                    previousAttentionInput[rank] = int(list.size()) - 1;
                    auto *owner = g->ownerFfn.empty() ? nullptr : g->ownerFfn[rank].get();
                    const int rows = g->attentionGraphs[rank]->batch;
                    if (owner && l > firstLayer && engram == g->engrams.end() && !selectInputs &&
                        !list.empty() && parts.size() == 1 && FastllmV41HcEnabledAtRows("FASTLLM_DSV41_EXPERT_BOUNDARY", rows)) {
                        auto &alternates = g->attentionGraphs[rank]->allIndexChildren;
                        const auto alternate = alternates.find(parts[0]);
                        const void *shortGraph = alternate == alternates.end() ? nullptr : alternate->second.second;
                        void *fused[3] = {};
                        FastllmCudaSetDevice(g->devices[rank]);
                        if (FastllmCudaV41FuseExpertBoundary(list.back(), parts[0], shortGraph,
                                FastllmDsv41OwnerPostKernel(), rows, !isKvSource[l] && !isIndexSource[l], fused)) {
                            list.back() = fused[0]; list.push_back(fused[1]);
                            for (auto handle : fused) if (handle) owner->mixGraphs.push_back(handle);
                            if (shortGraph) g->allIndexChildren[fused[1]] = {alternate->second.first, fused[2]};
                            continue;
                        }
                        FastllmCudaClearGraphError(); FastllmCudaClearThreadError();
                    }
#endif
                    children[rank].insert(children[rank].end(), parts.begin(), parts.end());
                }
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                if (!g->ownerFfn.empty()) {
                    // DSpark reads pre-attention hidden states in global token order.
                    // Publish the previous layer before any of its features are sampled.
                    const auto nextEngram = g->engrams.find(l + 1);
                    const auto feature = std::find(v41DsparkTargetLayerIds.begin(), v41DsparkTargetLayerIds.end(), l + 1);
                    const bool saveFeature = ownerFeatures && feature != v41DsparkTargetLayerIds.end();
                    const bool publish = (capturesDraftInput(l + 1) && !saveFeature) || l + 1 == endLayer ||
                        (nextEngram != g->engrams.end() && nextEngram->second->owners.empty());
                    capture([&] {
                        runtime.Enqueue([&](int rank, int device) {
                            auto *a = g->ownerFfn[rank].get();
                            auto *local = a ? g->attentionStates[rank]->workspace.get() : nullptr;
                            const int m = a ? g->attentionGraphs[rank]->batch : 0;
                            void *mixReady = nullptr, *packetReady = nullptr;
                            FastllmDsv41InputPacket packet{};
                            bool routesReady = false;
                            if (a) {
                                auto w = [&](const char *suffix) -> Data & {
                                    return V41FullLocal(weight[prefix + suffix], device);
                                };
                                Data &fn = w(".hc_ffn_fn"), &scale = w(".hc_ffn_scale"), &base = w(".hc_ffn_base");
                                Data &norm = w(".ffn_norm.weight");
                                if (a->clusterMix) {
                                    packet = FastllmDsv41OwnerInputPacket(g->ownerExperts[l][rank], m);
                                    mixReady = FastllmCudaV41OwnerClusterPacket(a->clusterMix, m,
                                        static_cast<float *>(a->mixScratch.cudaData), local->attnProj,
                                        local->hiddenStates, local->attnPre, local->attnPost, local->attnComb,
                                        fn, scale, base, norm, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                                        a->residual, local->preMix, a->post, a->comb, a->input, &packet);
                                }
                                bool fused = mixReady || FastllmCudaDeepSeekV41HcPostMixPreNorm(local->attnProj,
                                    local->hiddenStates, local->attnPre, local->attnPost, local->attnComb,
                                    fn, scale, base, norm, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                                    a->residual, local->preMix, a->post, a->comb, a->input);
                                if (!fused) {
                                    AssertInFastLLM(FastllmCudaDeepSeekV41HcPost(local->attnProj,
                                        local->hiddenStates, local->attnPost, local->attnComb, a->residual),
                                        "owner-local attention residual failed");
                                    if (!FastllmCudaDeepSeekV41HcMixPreNorm(a->residual, local->attnPre, fn,
                                            scale, base, norm, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                                            local->preMix, a->post, a->comb, a->input, nullptr))
                                        AssertInFastLLM(FastllmCudaDeepSeekV41HcMix(a->residual, fn, scale, base, hc_mult,
                                            hc_sinkhorn_iters, hc_eps, rms_norm_eps, local->preMix, a->post, a->comb) &&
                                        FastllmCudaDeepSeekV41HcPreNorm(a->residual, local->attnPre, norm,
                                            rms_norm_eps, a->input), "owner-local FFN normalization failed");
                                }
                                if (!mixReady || !packet.codes)
                                    packetReady = FastllmDsv41PackLocalInput(g->ownerExperts[l][rank], a->input.cudaData, m);
                                auto &rw = w(".ffn.gate.weight"), &bias = w(".ffn.gate.bias");
                                if (packetReady) packet = FastllmDsv41OwnerInputPacket(g->ownerExperts[l][rank], m);
                                if ((packetReady || mixReady) && packet.routeIds && (m == 6 || m >= 16))
                                    routesReady = FastllmCudaV41OwnerRouterPacket(a->input,rw,bias,gate_temp,
                                        routed_scaling_factor,a->logits,a->ids,a->scores,packet);
                                if (!routesReady && !FastllmCudaV41OwnerRouterSelect(a->input,rw,bias,gate_temp,routed_scaling_factor,
                                        a->logits,a->ids,a->scores)) {
                                    AssertInFastLLM(FastllmCudaV41RouterScores(a->input,rw,gate_temp,a->logits),
                                                    "owner-local FFN router projection failed");
                                    if (mixReady && packet.routeIds)
                                        routesReady = FastllmCudaV41SelectAndPublish((float *)a->logits.cudaData,
                                            (const float *)bias.cudaData, (int *)a->ids.cudaData, (float *)a->scores.cudaData,
                                            m, routed_scaling_factor, packet);
                                    AssertInFastLLM(routesReady || FastllmCudaSelectExpert(a->logits,&bias,a->ids,a->scores,
                                            6,true,routed_scaling_factor), "owner-local FFN selection failed");
                                }
                            }
                            FastllmCudaGraphRecordCollective();
                            FastllmDsv41MegaRunLocalPacket(g->ownerExperts[l][rank], a ? a->input.cudaData : nullptr,
                                a ? static_cast<int32_t *>(a->ids.cudaData) : nullptr,
                                a ? static_cast<float *>(a->scores.cudaData) : nullptr,
                                a ? a->residual.cudaData : nullptr,
                                a ? static_cast<float *>(a->post.cudaData) : nullptr,
                                a ? static_cast<float *>(a->comb.cudaData) : nullptr,
                                a ? local->hiddenStates.cudaData : nullptr, m, nullptr, mixReady, packetReady, mixReady && packet.codes, routesReady);
                            if (saveFeature && a)
                                AssertInFastLLM(FastllmCudaV41PublishDraftFeature(local->hiddenStates.cudaData,
                                    g->packedMainHidden.cudaData, g->rowTransfer->LocalToGlobal(rank), m, embed_dim,
                                    int(feature - v41DsparkTargetLayerIds.begin()), int(v41DsparkTargetLayerIds.size())),
                                    "owner-local draft feature publication failed");
                            if (publish) {
                                parallel::CudaRowTransfer::Field fields[] = {
                                    {a ? local->hiddenStates.cudaData : nullptr,
                                        V41FullLocal(ws.hiddenStates,device).cudaData,
                                        size_t(embed_dim*hc_mult)*sizeof(uint16_t)},
                                    {a ? local->preMix.cudaData : nullptr, V41FullLocal(ws.ffnPre,device).cudaData,
                                        size_t(hc_mult)*sizeof(float)}
                                };
                                g->rowTransfer->Collect(rank, fields, 2);
                            }
                        });
                    });
                    for (size_t rank = 0; rank < children.size(); ++rank) {
                        auto *owner = g->ownerFfn[rank].get();
                        if (!owner || !owner->mixScratch.cudaData || owner->clusterMix ||
                            !FastllmV41HcEnabledAtRows("FASTLLM_DSV41_HC_OVERLAP", g->attentionGraphs[rank]->batch)) continue;
                        FastllmCudaSetDevice(g->devices[rank]);
                        auto &child = children[rank].back();
                        if (void *fork = FastllmCudaV41ForkOwnerMix(&child, 1, g->attentionGraphs[rank]->batch,
                                static_cast<float *>(owner->mixScratch.cudaData), FastllmDsv41OwnerPostKernel(), FastllmDsv41PrepareKernel(), &owner->mixGemm)) {
                            owner->mixGraphs.push_back(fork);
                            child = fork;
                        }
                    }
                    continue;
                }
#endif
                capture([&] {
                    runtime.Enqueue([&](int rank, int device) {
                        auto *local = g->attentionStates[rank] ? g->attentionStates[rank]->workspace.get() : nullptr;
                        if (local && topology.attentionTP > 1) {
                            auto &projection = local->attnProj;
                            auto &sum = local->reductionBuffers["attention_tp"];
                            const auto type = projection.dataType;
                            ToDataType(projection, sum, FLOAT32);
                            runtime.AllReduce(topology.attention[topology.AttentionOwner(rank)],
                                rank, sum.cudaData, sum.Count(0), parallel::ScalarType::Float32);
                            ToDataType(sum, projection, type);
                        }
                        parallel::CudaRowTransfer::Field fields[4];
                        Data *globals[] = {&ws.attnProj, &ws.attnPre, &ws.attnPost, &ws.attnComb};
                        Data *locals[] = {local ? &local->attnProj : nullptr, local ? &local->attnPre : nullptr,
                            local ? &local->attnPost : nullptr, local ? &local->attnComb : nullptr};
                        for (int i = 0; i < 4; ++i)
                            fields[i] = {locals[i] ? locals[i]->cudaData : nullptr,
                                V41FullLocal(*globals[i], device).cudaData, globals[i]->GetBytes() / batch};
                        g->rowTransfer->Collect(rank, fields, 4);
                    });
                });
            } else {
            attentionInputChild = children[0].size();
            for (size_t rank = 0; rank < children.size(); ++rank) {
                void *child = state->segments[l * 4].graphs[rank];
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                if (attentionOnly && batch == 1 && g->devices.size() == 1 &&
                    v41AttentionRankRuntime && V41EnvFlag("FASTLLM_DSV41_LOCAL_EP_GRAPH") &&
                    !V41EnvFlag("FASTLLM_DSV41_DISABLE_SPLIT_ATTENTION_MIX")) {
                    FastllmCudaSetDevice(g->devices.front());
                    if (g->deferredHcMix.dims.empty()) alloc(g->deferredHcMix, FLOAT32, {8 * 25});
                    if (auto clone = FastllmCudaV41CloneSingleAttentionMix(child,
                            static_cast<float *>(g->deferredHcMix.cudaData))) {
                        g->splitAttentionChildren.push_back(clone);
                        child = clone;
                    }
                }
#endif
                children[rank].push_back(child);
            }
            auto &layer = *g->layers[l];
            // Every layer receives the same per-query positions. Only share
            // lists with matching compression ratios; a new mask invalidates
            // the previous producer even when its addresses stay unchanged.
            if (layer.config.writeCandidates) candidateWorkspaceRatio = -1;
            const bool shareCandidates = layer.config.indexSource && layer.config.useCandidates &&
                !layer.config.writeCandidates &&
                (candidateWorkspaceRatio < 0 || candidateWorkspaceRatio == layer.config.ratio);
            const bool reuseCandidates = shareCandidates && candidateWorkspaceRatio == layer.config.ratio;
            if (shareCandidates) candidateWorkspaceRatio = layer.config.ratio;
            if (layer.config.kvSource) {
                auto &norm = weight[prefix + ".attn.compressor.norm.weight"];
                if (g->tensorParallel) PrepareMultiCudaReplicatedData(norm, g->devices, true);
                else norm.ToDevice(DataDevice::CUDA, g->devices);
            }
            bool plansOk = true;
            layer.plans.resize(g->devices.size(), nullptr);
            V41FullDispatch(*g, [&](int rank, int device) {
                auto ptr = [device](Data &d) -> void * {
                    return V41FullLocal(d, device).cudaData;
                };
                FastllmV41FullDecodeBuffers b;
                b.requests = static_cast<FastllmV41FullDecodeRequest *>(ptr(g->requests)) + l * batch;
                b.q = ptr(ws.q);
                // Consume the address written by this layer's captured pre
                // graph, even if another layer later replaced ws.kv storage.
                b.kv = state->segments[l * 4].kvOutputs.at(rank);
                b.output = ptr(ws.attnOut);
                const auto &projection = state->segments[l * 4];
                if (layer.config.kvSource) {
                    b.rawKV = projection.rawKVOutputs.at(rank);
                    if (layer.config.ratio > 1) b.rawScore = projection.rawScoreOutputs.at(rank);
                }
                b.joined = ptr(layer.joined);
                b.scores = ptr(layer.scores);
                b.latents = ptr(layer.latents);
                b.newIndexKeys = ptr(layer.keys);
                b.indexQ = ptr(ws.qIdxAll);
                b.indexWeights = ptr(ws.idxWeightsAll);
                b.topk = ptr(g->topk);
                b.candidates = ptr(g->candidates);
                if (shareCandidates) {
                    b.candidateBlocks = ptr(g->candidateBlocks);
                    b.candidateCounts = ptr(g->candidateCounts);
                    b.reuseCandidates = reuseCandidates;
                }
                b.sink = ptr(weight[prefix + ".attn.attn_sink"]);
                if (verifying || layer.config.ringMode) {
                    b.savedKV = ptr(layer.savedKV);
                    b.savedRawKV = ptr(layer.savedRawKV);
                    b.savedRawScore = ptr(layer.savedRawScore);
                }
                if (layer.config.kvSource)
                    b.compressNorm = ptr(weight[prefix + ".attn.compressor.norm.weight"]);
                if (layer.config.kvSource && layer.config.indexSource) {
                    auto &norm = weight[prefix + ".attn.indexer.k_norm.weight"];
                    if (norm.dataType == DataType::FLOAT32 && norm.Count(0) == 128)
                        b.indexNorm = static_cast<const float *>(ptr(norm));
                }
                layer.plans[rank] = FastllmCudaV41CreateFullDecode(layer.config, b);
                // Hidden-state dependencies serialize every layer's attention,
                // including when independent mHC/MoE branches overlap.
                auto &prior = attentionScratch[rank][layer.config.ratio != 0];
                if (prior) FastllmCudaV41ShareAttentionWorkspace(layer.plans[rank], prior);
                else prior = layer.plans[rank];
            });
            for (void *plan : layer.plans)
                plansOk &= plan != nullptr;
            if (!plansOk)
                throw std::runtime_error("full graph dynamic plan rejected");
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
            if (attentionOnly && g->devices.size() == 1 && (!verifying || batch == 6) &&
                V41EnvFlag("FASTLLM_DSV41_MLA_OUTPUT")) {
                void *codes = nullptr, *scales = nullptr;
                int pitch = 0;
                bool headMajor = false;
                if (FastllmCudaV41MlaOutput(layer.plans[0], codes, scales, pitch, headMajor)) {
                    const int device = g->devices.front();
                    FastllmCudaSetDevice(device);
                    auto inserted = v41OutputWeights.try_emplace(std::make_pair(l, headMajor));
                    if (inserted.second)
                        inserted.first->second = FastllmCudaV41PackOutputWeight(
                            V41FullLocal(weight[prefix + ".attn.wo_a.weight"], device), headMajor);
                    layer.outputProjection = FastllmCudaV41PlanOutput(inserted.first->second,
                        V41FullLocal(weight[prefix + ".attn.wo_b.weight"], device),
                        V41FullLocal(ws.attnProj, device), batch, pitch, codes, scales);
                    if (layer.outputProjection) FastllmCudaV41UseMlaOutput(layer.plans[0]);
                }
            }
#endif
            V41FullDispatch(*g, [&](int rank, int device) {
                AssertInFastLLM(V41FullUploadRank(*g, segments, rank, device),
                                "full graph initial metadata rejected");
            });
            auto captureAttention = [&](bool allIndices) {
                const bool publish = (allIndices || verifying) &&
                    std::all_of(layer.plans.begin(), layer.plans.end(), FastllmCudaV41CanPublishSource);
                capture([&] {
                    V41FullDispatch(*g, [&](int rank, int) {
                        auto *plan = layer.plans[rank];
                        const bool prepared = publish ? FastllmCudaV41PrepareSource(plan, allIndices) :
                            allIndices ? FastllmCudaV41PrepareAllIndices(plan, false) : FastllmCudaV41PrepareFullDecode(plan);
                        AssertInFastLLM(prepared, "full graph prepare failed");
                    });
                    if (layer.config.kvSource && layer.config.indexSource) {
                        LinearWithActivationQuant(layer.latents, prefix + ".attn.indexer.wk.weight", layer.keys, true);
                        if (!publish)
                            V41RMSNormBF16(layer.keys, weight[prefix + ".attn.indexer.k_norm.weight"], rms_norm_eps, layer.keys);
                    }
                    V41FullDispatch(*g, [&](int rank, int) {
                        auto *plan = layer.plans[rank];
                        const bool finished = publish ? FastllmCudaV41FinishSource(plan, allIndices) :
                            allIndices ? FastllmCudaV41FinishAllIndices(plan) : FastllmCudaV41FinishFullDecode(plan);
                        AssertInFastLLM(finished, "full graph attention failed");
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                        if (layer.outputProjection)
                            AssertInFastLLM(FastllmCudaV41RunOutput(layer.outputProjection),
                                            "full graph output projection failed");
#endif
                    });
                });
            };
            captureAttention(false);
            if (attentionOnly && (!verifying || batch == 6) && layer.config.indexSource && V41EnvFlag("FASTLLM_DSV41_SHORT_INDEX")) {
                const auto ordinary = children[0].back();
                captureAttention(true);
                auto selected = children[0].back();
                if (FastllmCudaV41CanPublishSource(layer.plans[0])) {
                    if (auto fork = FastllmCudaV41ForkSourceStore(selected)) {
                        g->splitAttentionChildren.push_back(fork);
                        selected = fork;
                    }
                }
                g->allIndexChildren[ordinary] = {layer.config.ratio, selected};
                children[0].pop_back();
            }
            } // dynamic local/world attention
            if (attentionOnly) {
                if (!g->layers[l]->outputProjection)
                    children[0].push_back(state->segments[l * 4 + 1].graphs[0]);
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
                if ((!verifying || batch <= 8) && g->devices.size() == 1 && V41EnvFlag("FASTLLM_DSV41_LOCAL_EP_GRAPH") &&
                    v41AttentionRankRuntime && FastllmV41HcEnabledAtRows("FASTLLM_DSV41_HC_OVERLAP", batch)) {
                    // The existing B1 optimization may already have cloned the
                    // prelude. Start from its captured, unmodified template.
                    auto &parts = children[0];
                    std::vector<const void *> input(parts.begin() + attentionInputChild, parts.end());
                    const auto &prelude = state->segments[l * 4];
                    input.front() = prelude.graphs[0];
                    if ((!verifying || batch == 6) && prelude.beforeCompressor && prelude.compressorInput && prelude.beforeIndexProjection) {
                        if (auto branched = FastllmCudaV41GraphPrefix(prelude.graphs[0], prelude.beforeIndexProjection,
                                prelude.compressorInput, prelude.beforeCompressor, false)) {
                            input.front() = branched;
                            g->splitAttentionChildren.push_back(branched);
                        }
                    }
                    if (g->deferredHcMix.dims.empty())
                        alloc(g->deferredHcMix, FLOAT32, {batch * 8 * 25});
                    FastllmCudaSetDevice(g->devices.front());
                    auto &layer = *g->layers[l];
                    auto *mix = static_cast<float *>(g->deferredHcMix.cudaData);
                    void *fork = nullptr;
                    if (verifying && batch == 6 && layer.config.indexSource && layer.config.indexHeads == 32) {
                        if (g->indexHeadWeights.dims.empty())
                            alloc(g->indexHeadWeights, BFLOAT16, {batch, layer.config.indexHeads});
                        fork = FastllmCudaV41ForkIndexProjections(input.data(), input.size(), prelude.graphs[0],
                            prelude.indexProjectionNodes, layer.plans[0], g->indexHeadWeights.cudaData, mix, &g->attentionMixGemm);
                    }
                    if (!fork)
                        fork = FastllmCudaV41ForkOwnerMixQuery(input.data(), input.size(), batch, mix, nullptr, nullptr,
                                                              &g->attentionMixGemm, layer.config.indexSource);
                    if (fork) {
                        g->splitAttentionChildren.push_back(fork);
                        // Both versions must have exactly the same mHC dependency and FP32 reduction.
                        if (input.size() > 1 && g->allIndexChildren.count(input[1])) {
                            auto alternative = input;
                            auto replacement = g->allIndexChildren.at(input[1]);
                            alternative[1] = replacement.second;
                            void *prefixGraph = nullptr;
                            if (verifying && prelude.compressorInput && prelude.beforeCompressor)
                                prefixGraph = FastllmCudaV41GraphPrefix(prelude.graphs[0], prelude.beforeIndexProjection,
                                    prelude.compressorInput, prelude.beforeCompressor);
                            if (!prefixGraph && !prelude.indexProjectionNodes.empty())
                                prefixGraph = FastllmCudaV41WithoutIndexProjections(prelude.graphs[0], prelude.indexProjectionNodes);
                            if (V41DecodeCudaGraphVerbose() && batch >= 16 && g->devices.front() == 0)
                                printf("[Fastllm] short index prelude layer=%d rows=%d projection_nodes=%zu trimmed=%d frontier=%d\n",
                                    l, batch, prelude.indexProjectionNodes.size(), int(prefixGraph != nullptr),
                                    int(prelude.beforeIndexProjection != nullptr));
                            if (!prefixGraph)
                                prefixGraph = FastllmCudaV41GraphPrefix(prelude.graphs[0], prelude.beforeIndexProjection,
                                    prelude.compressorInput, prelude.beforeCompressor);
                            if (prefixGraph) {
                                alternative[0] = prefixGraph;
                                if (prelude.beforeCompressor && prelude.compressorInput &&
                                    FastllmV41HcEnabledAtRows("FASTLLM_DSV41_EARLY_COMPRESS", batch)) {
                                    if (auto core = FastllmCudaV41MoveSourceToPrelude(prefixGraph,
                                            prelude.beforeIndexProjection, alternative[1])) {
                                        alternative[1] = core;
                                        g->splitAttentionChildren.push_back(core);
                                    }
                                }
                                g->splitAttentionChildren.push_back(prefixGraph);
                            }
                            if (auto shortFork = FastllmCudaV41ForkOwnerMixQuery(alternative.data(), alternative.size(), batch,
                                    static_cast<float *>(g->deferredHcMix.cudaData), nullptr, nullptr, &g->attentionMixGemm, !prefixGraph)) {
                                g->splitAttentionChildren.push_back(shortFork);
                                g->allIndexChildren[fork] = {replacement.first, shortFork};
                            }
                        }
                        parts.resize(attentionInputChild);
                        parts.push_back(fork);
                    }
                }
#endif
                g->attentionLayers[l].assign(children[0].begin() + attentionStart, children[0].end());
                continue;
            }
            const bool hostMoe = hybrid && !state->segments[l * 4 + 3].containsGpuMoe;
            if (hybrid) {
                hcJoin.back() = children[0].size();
                if (hostMoe) { hcStart.push_back(children[0].size()); hcJoin.push_back(-1); }
            }
            for (int piece = 1; piece < (hostMoe ? 3 : 4); ++piece)
                for (size_t rank = 0; rank < children.size(); ++rank)
                    children[rank].push_back(state->segments[l * 4 + piece].graphs[rank]);
            if (hostMoe && g->tensorParallel) {
                // TP shards attention; each complete routed layer retains one
                // cache owner. Broadcast its result directly inside the graph.
                const size_t owner = size_t(l) * g->devices.size() / block_cnt;
                const int device = g->devices[owner];
                FastllmCudaSetDevice(device);
                auto routed = weights[l]; routed[0] = routed[1] = nullptr;
                Data input, index, score;
                input.FakeFrom(V41FullLocal(ws.ffnInput, device), 0);
                input.Resize({batch, embed_dim});
                index.FakeFrom(V41FullLocal(ws.expertIndex, device), 0);
                index.Resize(state->segments[l * 4 + 2].expertIndexDims);
                score.FakeFrom(V41FullLocal(ws.expertScore, device), 0);
                score.Resize(state->segments[l * 4 + 2].expertScoreDims);
                void *moeGraph = nullptr;
                auto &output = V41FullLocal(g->routedOutput, device);
                // Shared and routed experts only read ffnInput. On the owner,
                // let the native mailbox fork them and join before broadcast.
                // The other ranks compute their shared result independently.
                for (size_t rank = 0; rank < children.size(); ++rank)
                    if (rank != owner || routed[2]->isDiskWeight)
                        children[rank].push_back(state->segments[l * 4 + 3].graphs[rank]);
                if (routed[2]->isDiskWeight) {
                    if (!g->diskMoe) g->diskMoe = std::make_unique<DiskMoeV41Graph>();
                    moeGraph = g->diskMoe->Add(input, index, score, output, routed, swiglu_limit);
                } else {
                    // Segmented TP has no model-wide admission scope. Prepare
                    // this owner's policy before creating its mailbox children.
                    HybridMoeDecodeStep preparation(true);
                    preparation.Begin(routed.data(), routed.size(), num_experts_per_tok);
                    void *handle = FastllmCudaCreateNativeMoeFullGraph(input, index, score, output,
                        routed.data(), routed.size(), l, state->segments[l * 4 + 3].graphs[owner], &moeGraph);
                    if (handle) {
                        g->rankMoeHandles[owner].push_back(handle);
                        if (g->rankMoeFirstLayer[owner] < 0) g->rankMoeFirstLayer[owner] = l;
                    }
                }
                if (!moeGraph) throw std::runtime_error("TP MoE full graph rejected at layer " + std::to_string(l));
                children[owner].push_back(moeGraph);
                capture([&] {
                    V41FullDispatch(*g, [&](int, int rankDevice) {
                        auto local = [&](Data &data) -> Data & { return V41FullLocal(data, rankDevice); };
                        auto &result = local(g->routedOutput);
                        FastllmNcclBroadcastFrom(result.cudaData, result.cudaData, batch * embed_dim,
                            result.dataType, device, rankDevice);
                        AssertInFastLLM(FastllmCudaDeepSeekV41MoeHcPost(result, local(ws.sharedExpertOut),
                            local(ws.hiddenTemp), local(ws.ffnPost), local(ws.ffnComb), local(ws.hiddenStates)),
                            "full TP hybrid MoE residual failed");
                    });
                });
            } else if (hostMoe) {
                auto routed = weights[l]; routed[0] = routed[1] = nullptr;
                Data input;
                input.FakeFrom(ws.ffnInput, 0); input.Resize({batch, embed_dim});
                ws.expertIndex.Reshape(state->segments[l * 4 + 2].expertIndexDims);
                ws.expertScore.Reshape(state->segments[l * 4 + 2].expertScoreDims);
                void *moeGraph = nullptr;
                if (routed[2]->isDiskWeight) {
                    if (!g->diskMoe) g->diskMoe = std::make_unique<DiskMoeV41Graph>();
                    moeGraph = g->diskMoe->Add(input, ws.expertIndex, ws.expertScore,
                        ws.ffnOut, routed, swiglu_limit);
                    children[0].push_back(state->segments[l * 4 + 3].graphs[0]);
                } else {
                    void *handle = FastllmCudaCreateNativeMoeFullGraph(input, ws.expertIndex,
                        ws.expertScore, ws.ffnOut, routed.data(), routed.size(), l,
                        state->segments[l * 4 + 3].graphs[0], &moeGraph);
                    if (handle) g->moeHandles.push_back(handle);
                }
                if (!moeGraph) throw std::runtime_error("MoE full graph rejected at layer " + std::to_string(l));
                children[0].push_back(moeGraph);
                // Router and expert computation read ffnInput, not ffnPre /
                // ffnPost / ffnComb. Join before the residual consumes them.
                hcJoin.back() = children[0].size();
                capture([&] {
                    AssertInFastLLM(FastllmCudaDeepSeekV41MoeHcPost(ws.ffnOut, ws.sharedExpertOut,
                        ws.hiddenTemp, ws.ffnPost, ws.ffnComb, ws.hiddenStates),
                        "full hybrid MoE residual failed");
                });
            }
        }
        if (!attentionOnly && endLayer == block_cnt) capture([&] {
            V41HcApplyPre(ws.hiddenStates, ws.ffnPre, g->head);
            weight["head.weight"].tpLinearType = TP_LINEAR_ROW;
            const bool compactHead = weight["head.weight"].dataType == DataType::FLOAT32 ||
                (weight["head.weight"].dataType == DataType::BFLOAT16 &&
                 !quantizedLinearNames.count("head.weight"));
            if (verifying || !compactHead) {
                // Preserve the ordinary quantized-head / verifier boundary.
                // mHC may produce FP32; forcing BF16 here changes GGUF logits.
                RMSNorm(g->head, weight["norm.weight"], rms_norm_eps, g->norm);
                if (verifying) {
                    Data *headInput = &g->norm;
                    // Keep the FP32 head cast at a stable address for every
                    // verification shape; worker-local temporaries cannot be
                    // reserved by the caller's allocation recorder.
                    if (weight["head.weight"].dataType == DataType::FLOAT32) {
                        ToDataType(g->norm, g->normFloat, DataType::FLOAT32);
                        headInput = &g->normFloat;
                    }
                    DsparkProjectHead(*headInput, g->headLogits, false, g->tensorParallel);
                }
                else Linear(g->norm, weight["head.weight"], *GetEmptyData(), g->headLogits);
                ToDataType(g->headLogits, g->logits, DataType::FLOAT32);
            } else {
                V41RMSNormBF16(g->head, weight["norm.weight"], rms_norm_eps, g->norm);
                if (!TryBFloat16HeadLinear(g->norm, g->logits)) {
                    ToDataType(g->norm, g->normFloat, DataType::FLOAT32);
                    Linear(g->normFloat, weight["head.weight"], *GetEmptyData(), g->logits);
                }
            }
            if (g->captureGreedy) {
                if (!g->logits.multiDeviceData || !g->logits.IsTensorParallelSharded() ||
                    g->logits.dataType != FLOAT32 || g->logits.multiDeviceDatas.size() != g->devices.size()) {
                    g->captureGreedy = false;
                } else {
                    V41FullDispatch(*g, [&](int, int device) {
                        const auto &logits = V41FullLocal(g->logits, device);
                        int *ids = static_cast<int *>(V41FullLocal(g->greedyIds, device).cudaData);
                        float *scores = static_cast<float *>(V41FullLocal(g->greedyScores, device).cudaData);
                        if (!g->greedyMapped.empty()) {
                            ids = static_cast<int *>(g->greedyMapped.at(device));
                            scores = reinterpret_cast<float *>(ids + batch);
                        }
                        AssertInFastLLM(FastllmCudaGreedySamplingWithScores(
                            static_cast<float *>(logits.cudaData), ids, scores,
                            batch, logits.dims.back()), "full graph greedy candidates failed");
                    });
                }
            }
            if (verifying && !g->tensorParallel) {
                g->gatheredLogits.CopyFrom(g->logits);
            } else if (verifying) {
                if (g->gatheredLogits.dims.empty()) {
                    FastllmCudaSetDevice(g->devices.front());
                    g->gatheredLogits.dataType = DataType::FLOAT32;
                    g->gatheredLogits.dataDevice = DataDevice::CUDA;
                    g->gatheredLogits.dataDeviceIds = {g->devices.front()};
                    g->gatheredLogits.Resize(g->logits.dims);
                    g->gatheredLogits.Allocate();
                }
                V41FullDispatch(*g, [&](int, int device) {
                    const auto &local = V41FullLocal(g->logits, device);
                    const int vocab = g->logits.dims.back(), width = local.dims.back();
                    int offset = 0;
                    for (const auto &range : g->logits.tpRanges.at(device)) {
                        AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(
                            static_cast<float *>(g->gatheredLogits.cudaData) + range.first,
                            vocab * sizeof(float), static_cast<const float *>(local.cudaData) + offset,
                            width * sizeof(float), (range.second - range.first) * sizeof(float), batch),
                            "full verify logits gather failed");
                        offset += range.second - range.first;
                    }
                });
            }
        });
        if (!FastllmCudaGraphMemoryPoolEnd(g->pieces->reservedPointers))
            throw std::runtime_error("full graph pool finish");
        poolOpen = false;
        g->graphs.resize(g->devices.size(), nullptr);
        g->execs.resize(g->devices.size(), nullptr);
        if (attentionOnly && verifying && batch == 6 && g->devices.size() == 1) {
            FastllmCudaSetDevice(g->devices.front());
            g->lastVerifyReader = FastllmCudaEventCreate();
        }
        if (verifying || g->captureGreedy) g->ready.resize(g->devices.size(), nullptr);
        auto compose = [&](const std::vector<const void *> &parts, size_t rank, void **graph, void **exec) {
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
            const bool forked = !engramForks[rank].empty() &&
                FastllmCudaV41ComposeEngramForks(parts, engramForks[rank], graph, exec);
            if (!forked && !FastllmCudaGraphCompose(parts.data(), parts.size(), graph, exec)) return false;
            if (multiVerify && g->attentionRuntime && !g->ownerFfn.empty() && g->devices.size() == 8) {
                if (void *flat = FastllmCudaV41FlatVerifyExec(*graph)) {
                    FastllmCudaGraphExecDestroy(*exec);
                    *exec = flat;
                }
            }
            return true;
#else
            return FastllmCudaGraphCompose(parts.data(), parts.size(), graph, exec);
#endif
        };
        for (size_t rank = 0; !attentionOnly && rank < g->devices.size(); ++rank) {
            FastllmCudaSetDevice(g->devices[rank]);
            if (verifying || g->captureGreedy) {
                g->ready[rank] = FastllmCudaEventCreate();
                if (!g->ready[rank]) throw std::runtime_error("full graph completion event");
                if (g->captureGreedy) g->greedyReady.emplace(g->devices[rank], g->ready[rank]);
            }
            const bool deferred = hybrid && !g->tensorParallel && batch == 1 && FastllmCudaV41ComposeHcGraph(
                children[rank].data(), children[rank].size(), hcStart.data(), hcJoin.data(),
                hcStart.size(), static_cast<float *>(g->deferredHcMix.cudaData),
                &g->graphs[rank], &g->execs[rank]);
            if (!deferred && !compose(children[rank], rank, &g->graphs[rank], &g->execs[rank]))
                throw std::runtime_error("full graph composition failed");
        }
        // Only replace complete rank-local source layers. A generic graph is always executable.
        const bool sixRowsPerOwner = verifying && std::all_of(g->attentionGraphs.begin(), g->attentionGraphs.end(),
            [](const auto &owner) { return !owner || (owner->batch == 6 && !owner->layers[owner->firstLayer]->config.segmentedVerify); });
        if (g->attentionRuntime && (!verifying || sixRowsPerOwner) && !hybrid && candidate_source_layer_id >= 0 &&
            candidate_source_layer_id < block_cnt && compress_ratios[candidate_source_layer_id] == 1 &&
            std::all_of(compress_ratios.begin(), compress_ratios.end(), [](int r) { return r >= 0 && r <= 2; }) &&
            int64_t(candidate_block_size) * candidate_topk_blocks >= int64_t(index_topk) * 2 &&
            V41EnvFlag("FASTLLM_DSV41_SHORT_INDEX")) {
            for (int mode = 0; mode < 2; ++mode) {
                g->allIndexExecs[mode].resize(g->devices.size(), nullptr);
                bool complete = true;
                for (size_t rank = 0; complete && rank < g->devices.size(); ++rank) {
                    auto local = g->attentionGraphs[rank];
                    if (!local) continue; // Empty EP peers use their ordinary graph.
                    auto parts = children[rank];
                    size_t replaced = 0, required = 0;
                    for (int l = 0; l < block_cnt; ++l)
                        if (isIndexSource[l] && compress_ratios[l] >= mode + 1) ++required;
                    for (auto &part : parts) {
                        const auto &choices = g->allIndexChildren.count(part) ? g->allIndexChildren : local->allIndexChildren;
                        auto it = choices.find(part);
                        if (it != choices.end() && it->second.first >= mode + 1) {
                            part = it->second.second; ++replaced;
                        }
                    }
                    FastllmCudaSetDevice(g->devices[rank]);
                    void *composed = nullptr;
                    complete = replaced == required && required &&
                        compose(parts, rank, &composed, &g->allIndexExecs[mode][rank]);
                    if (composed) FastllmCudaGraphDestroy(composed);
                    if (!complete) { FastllmCudaClearGraphError(); FastllmCudaClearThreadError(); }
                }
                if (complete) g->allIndexLimit[mode] = index_topk * (mode + 1);
            }
        }
        if (!g->moeHandles.empty() && !FastllmCudaCreateNativeMoeReplayGraph(g->graphs[0], g->moeHandles.data(),
                g->moeHandles.size(), &g->untimedExec))
            throw std::runtime_error("full graph timing removal failed");
        for (size_t rank = 0; rank < g->devices.size(); ++rank) {
            auto &handles = g->rankMoeHandles[rank];
            FastllmCudaSetDevice(g->devices[rank]);
            if (!handles.empty() && !FastllmCudaCreateNativeMoeReplayGraph(g->graphs[rank],
                    handles.data(), handles.size(), &g->rankUntimedExecs[rank]))
                throw std::runtime_error("TP full graph timing removal failed");
        }
        if (verifying && !distributedAttention) {
            if (!g->tensorParallel && g->devices.size() == 1) {
                g->commitSpans.dataType = DataType::INT32;
                g->commitSpans.UpdateUnitSize();
                g->commitSpans.Resize({g->batch, 2});
                g->commitSpans.ToDevice(DataDevice::CUDA, g->devices, false);
                g->commitSpans.Allocate();
            }
            g->commitDescriptors.resize(g->devices.size());
            for (size_t rank = 0; rank < g->devices.size(); ++rank) {
                const int device = g->devices[rank];
                FastllmCudaSetDevice(device);
                auto ptr = [device](const Data &data) -> void * {
                    return V41FullLocal(data, device).cudaData;
                };
                const auto *requests = static_cast<const FastllmV41FullDecodeRequest *>(ptr(g->requests));
                std::vector<FastllmV41VerifyCommitLayer> descriptors;
                descriptors.reserve(g->endLayer - g->firstLayer);
                for (int layer = g->firstLayer; layer < g->endLayer; ++layer) {
                    const auto &l = *g->layers[layer];
                    descriptors.push_back({
                        requests + layer * g->batch,
                        ptr(l.savedKV), ptr(l.savedRawKV), ptr(l.savedRawScore),
                        l.config.window, l.config.kvSource ? l.config.ratio : 0, l.config.ringRowBytes});
                }
                Data &data = g->commitDescriptors[rank];
                data.dataType = DataType::INT8;
                data.UpdateUnitSize();
                data.Resize({(int)(descriptors.size() * sizeof(FastllmV41VerifyCommitLayer))});
                data.ToDevice(DataDevice::CUDA, {device}, false);
                data.Allocate();
                FastllmCudaCopyFromHostToDevice(data.cudaData, descriptors.data(), data.GetBytes());
            }
        }
        std::vector<V41FullEngram *> asyncEntries;
        for (auto &entry : g->engrams)
            if (entry.second->asyncGraph) asyncEntries.push_back(entry.second.get());
        if (!asyncEntries.empty())
            g->engramWorker.reset(new V41FullEngramWorker(std::move(asyncEntries)));
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
        if (!g->ownerFfn.empty())
            fprintf(stderr, "[Fastllm] owner-local EP8 full %s batch=%d layers=%d\n",
                    verifying ? "verify" : "decode", batch, endLayer-firstLayer);
#endif
        state->fullDecode = g;
        if (!attentionOnly) fprintf(stderr,
                "[Fastllm] DeepSeek-V4.1 full %s graph ready: %s=%d, ranks=%zu, one graph "
                "per rank/step, layers=[%d,%d), maxKeys=%d\n",
                distributedAttention ? (verifying ? "DP verify" : "DP decode") : verifying ? "verify" : "decode",
                verifying ? "tokens" : "batch", batch, g->devices.size(), firstLayer, endLayer, g->maxKeys);
    } catch (const std::exception &error) {
        if (poolOpen)
            FastllmCudaGraphMemoryPoolAbort();
        FastllmCudaClearThreadError();
        FastllmCudaClearGraphError();
        fprintf(stderr,
                "[Fastllm] DeepSeek-V4.1 full graph unavailable: %s; retaining segmented graphs\n",
                error.what());
    }
    if (verifying) {
        V41FullDispatch(*g, [&](int, int device) {
            FastllmCudaCopyFromDeviceToDevice(V41FullLocal(ws.hiddenStates, device).cudaData,
                V41FullLocal(initialHidden, device).cudaData,
                V41FullLocal(initialHidden, device).GetBytes());
        });
        for (int device : g->devices) FastllmCudaSyncDevice(device);
    }
    FastllmCudaSetDevice(original);
}

bool DeepSeekV41Model::TryPaddedDecode(const std::vector<DeepSeekV41Segment> &segments, const Data &ids,
        const std::vector<GenerationConfig> &configs, const LastTokensManager &last,
        std::vector<std::vector<float>*> *logits, std::vector<std::pair<Data*, Data*>> &past,
        std::vector<int> &result) {
    if (executionMode != ExecutionMode::Gpu || v41AttentionDP <= 1 || v41CudaGraphWarmingUp ||
        !GetFastllmEnv().cudaGraph || segments.size() <= 8) return false;
    // Small batches use a separate arithmetic regime also shared by DSpark.
    // Keep their existing path rather than padding across that boundary.
    std::vector<int> counts(v41AttentionDP, 0);
    for (const auto &s : segments) {
        if (!s.state || s.spec || s.seqlen != 1 || s.startPos < 1 ||
            s.state->attentionGroup < 0 || s.state->attentionGroup >= v41AttentionDP) return false;
        ++counts[s.state->attentionGroup];
    }
    std::shared_ptr<V41FullGraph> graph;
    for (auto &slot : v41CudaGraphSlots) {
        if (slot.first < int(segments.size())) continue;
        auto state = V41GetCudaGraphState(slot.second);
        if (!state->captured || state->disabled || !state->fullDecode) continue;
        auto g = std::static_pointer_cast<V41FullGraph>(state->fullDecode);
        if (!g->attentionRuntime || g->verifying || g->padding.size() != size_t(g->batch)) continue;
        bool fits = true;
        for (int group = 0; group < v41AttentionDP; ++group) {
            const auto &local = g->attentionGraphs[group * g->attentionRuntime->Layout().attentionTP];
            fits &= counts[group] <= (local ? local->batch : 0);
        }
        if (!fits) continue;
        // An exact layout already takes the ordinary full-graph path.
        if (g->batch == int(segments.size())) return false;
        graph = std::move(g);
        break;
    }
    if (!graph) return false;
    auto padded = segments;
    std::vector<int> used(v41AttentionDP, 0);
    for (const auto &s : graph->padding) {
        const int group = s.state->attentionGroup;
        if (used[group]++ < counts[group]) continue;
        auto dummy = s;
        dummy.startPos = dummy.state->totalLen;
        dummy.offset = padded.size(); dummy.seqlen = 1; dummy.spec = nullptr;
        padded.push_back(std::move(dummy));
    }
    if (padded.size() != size_t(graph->batch)) return false;
    // Rewind only private scratch metadata. Kernels may overwrite its next KV
    // position; no real request ever reads these values. Bound scratch context
    // and allocation across arbitrarily many replays, including eager fallback.
    struct Rewind {
        struct State { DeepSeekV41RequestState *p; int length; size_t history; };
        struct Layer { DeepSeekV41LayerCache *p; int length, blocks, tail; };
        std::vector<State> states;
        std::vector<Layer> layers;
        std::vector<std::pair<Data*, int>> rows;
        void Save(DeepSeekV41RequestState &s) {
            states.push_back({&s, s.totalLen, s.engramHistory.size()});
            for (auto &c : s.layers) {
                layers.push_back({&c, c.totalLen, c.compressedBlocks, c.rawTail});
                for (Data *d : {&c.compressedKV, &c.indexK, &c.rawTailKV, &c.rawTailScore}) {
                    if (d->dims.size() == 3) rows.emplace_back(d, d->dims[1]);
                    for (auto &replica : d->multiDeviceDatas)
                        if (replica.second->dims.size() == 3) rows.emplace_back(replica.second, replica.second->dims[1]);
                }
            }
            for (auto &rank : s.attentionRanks) if (rank) Save(*rank);
        }
        ~Rewind() {
            for (auto &s : states) { s.p->totalLen = s.length; s.p->engramHistory.resize(s.history); }
            for (auto &c : layers) { c.p->totalLen = c.length; c.p->compressedBlocks = c.blocks; c.p->rawTail = c.tail; }
            for (auto &entry : rows) if (entry.first->dims.size() == 3 && entry.first->dims[1] != entry.second) {
                auto dims = entry.first->dims; dims[1] = entry.second; entry.first->Resize(dims);
            }
        }
    } rewind;
    for (size_t i = segments.size(); i < padded.size(); ++i) rewind.Save(*padded[i].state);
    Data host; host.CopyFrom(ids); host.ToDevice(CPU);
    const auto *values = reinterpret_cast<const float *>(host.cpuData);
    std::vector<float> tokens(values, values + ids.Count(0));
    tokens.resize(padded.size(), 1.f);
    Data paddedIds(FLOAT32, {1, int(tokens.size())}, tokens);
    auto paddedConfigs = configs;
    paddedConfigs.resize(padded.size());
    for (size_t i = configs.size(); i < padded.size(); ++i) {
        paddedConfigs[i].do_sample = false; paddedConfigs[i].top_k = 1;
    }
    auto paddedLast = last;
    paddedLast.units.resize(padded.size());
    auto paddedLogits = logits ? *logits : std::vector<std::vector<float>*>();
    paddedLogits.resize(padded.size(), nullptr);
    std::vector<Data> storage((padded.size() - segments.size()) * block_cnt * 2);
    auto paddedPast = past;
    for (size_t i = 0; i < storage.size(); i += 2) paddedPast.emplace_back(&storage[i], &storage[i+1]);
    result = ForwardGpu(padded, paddedIds, nullptr, nullptr, paddedConfigs, paddedLast,
        logits ? &paddedLogits : nullptr, paddedPast);
    result.resize(segments.size());
    return true;
}

bool DeepSeekV41Model::TryFullDecodeGraph(ForwardWorkspace &f, bool inputsPrepared) {
    // Verification warmup uses the same composition/replay path as requests.
    // Ordinary decode still builds its full graph explicitly after warmup.
    // A graph captured for ordinary decode has no DSpark target features.
    // Keep plain DSpark steps on their feature-producing segmented path.
    if ((!f.dsparkVerify && std::any_of(f.inputSegments.begin(), f.inputSegments.end(),
            [](const DeepSeekV41Segment &seg) { return seg.spec != nullptr; })) ||
        !f.graphReplay || !f.graphState ||
        (v41CudaGraphWarmingUp && !f.dsparkVerify) ||
        (f.quantKV && executionMode != ExecutionMode::Hybrid && f.dsparkVerify && v41AttentionDP == 1) ||
        f.hasImageTokens || f.inputEmbeds)
        return false;
    if (f.graphState->fullDecode) {
        const auto &graph = *std::static_pointer_cast<V41FullGraph>(f.graphState->fullDecode);
        // With no explicit embedding policy, disabling graphs moves this
        // weight back to CPU. Re-enabling can allocate a new CUDA address.
        // Only the composed input child owns that address; layer segments
        // remain valid and can be reused after the frontend uploads it again.
        if (graph.capturesInput && graph.embeddingWeightAddress != weight["embed.weight"].cudaData) {
            f.graphState->fullDecode.reset();
            f.graphState->fullVerifyAttempted = false;
        }
    }
    // Building a graph warms its children and therefore still needs the eager
    // input on its first call. Only an existing graph may bypass that frontend.
    if (!inputsPrepared && (!f.graphState->fullDecode ||
            !std::static_pointer_cast<V41FullGraph>(f.graphState->fullDecode)->capturesInput))
        return false;
    if (!inputsPrepared) {
        // Preserve the address checks normally performed by PrepareInputs.
        std::vector<const void *> current;
        CollectForwardGraphPointers(f, current);
        if (current != f.graphState->boundaryPointers) return false;
    }
    if (f.dsparkVerify && !f.graphState->fullVerifyAttempted) {
        f.graphState->fullVerifyAttempted = true;
        BuildFullDecodeGraph(f.seqlen, f.inputSegments,
            f.pipeline ? f.pipeline->firstLayer : -1, f.pipeline ? f.pipeline->endLayer : -1);
    }
    if (!f.graphState->fullDecode) return false;
    auto g = std::static_pointer_cast<V41FullGraph>(f.graphState->fullDecode);
    if (g->attentionOnly) return false;
    if (!g->moeHandles.empty() && (!f.moeStep || !*f.moeStep)) return false;
    V41FullCallerAffinity affinity(g->coordinatorCpus);
    const auto *staged = f.inputSegments.size() == 1 ? f.inputSegments.front().spec : nullptr;
    if (staged && !staged->resolveDraftIds) staged = nullptr;
    if (staged) {
        const size_t offset = size_t(block_cnt) * g->batch * sizeof(FastllmV41FullDecodeRequest);
        if (!g->verifying || !g->capturesInput || !g->attentionRuntime || !g->ownerInputIds.empty() ||
            g->engramWorker || staged->verifyProducer != g->devices.front() ||
            g->requests.GetBytes() != offset + staged->verifyPacketBytes || g->engrams.size() != engram_layer_ids.size() ||
            std::any_of(g->engrams.begin(), g->engrams.end(), [](const auto &e) { return !e.second->table->gpuTable; })) return false;
    }
    if (g->capturesInput && !staged) {
        if ((int)f.tokenIds.size() != g->batch) return false;
        const int vocab = weight["embed.weight"].dims[0];
        for (int i = 0; i < g->batch; ++i) {
            if (f.tokenIds[i] < 0 || f.tokenIds[i] >= vocab) return false;
            if (g->ownerInputIds.empty()) g->hostTokens[i] = f.tokenIds[i];
        }
    }
    std::vector<std::vector<DeepSeekV41Segment>> attentionSegments;
    std::shared_ptr<V41RequestVerifyPages> pages;
    if (staged) {
        const auto &s = f.inputSegments.front();
        if (s.state->dspark->nextVerifyPosition == s.startPos)
            pages = std::static_pointer_cast<V41RequestVerifyPages>(s.state->dspark->nextVerifyPages);
    }
    bool residentInputs = false;
    if (g->attentionRuntime) {
        parallel::RowPartition rows(v41AttentionDP, {});
        if (!V41FullPartition(*g, f.inputSegments, attentionSegments, rows)) return false;
        bool fixed = true;
        for (size_t rank = 0; fixed && rank < g->attentionGraphs.size(); ++rank)
            if (g->attentionGraphs[rank]) {
                const auto &parts = attentionSegments[rank];
                auto &local = *g->attentionGraphs[rank];
                const bool validated = pages && parts.size() == 1 && parts.front().seqlen == local.batch &&
                    pages->Matches(local, *parts.front().state);
                fixed = V41FullReserve(local, parts, true, validated);
            }
        if (!fixed) {
            std::vector<int> reserved(g->devices.size(), 1);
            g->attentionRuntime->Run([&](int rank, int) {
                if (g->attentionGraphs[rank])
                    reserved[rank] = V41FullReserve(*g->attentionGraphs[rank], attentionSegments[rank]);
            });
            if (std::find(reserved.begin(), reserved.end(), 0) != reserved.end()) return false;
        }
        for (size_t r = 0; r < g->ownerInputIds.size(); ++r) {
            const auto &order = rows.localToGlobal[g->attentionRuntime->Layout().AttentionOwner(r)];
            if (order.size() != g->ownerHostIds[r].size()) return false;
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
            if (!g->inputPackets.empty() && g->inputPackets[r]) g->inputPackets[r]->order = order;
#endif
            for (size_t i = 0; i < order.size(); ++i) g->ownerHostIds[r][i] = f.tokenIds[order[i]];
        }
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
        residentInputs = !g->verifying && fixed && !g->inputPackets.empty();
        for (size_t r = 0; residentInputs && r < g->devices.size(); ++r) {
            const auto &owner = g->attentionGraphs[r];
            residentInputs = !owner || (g->inputPackets[r] && g->inputPackets[r]->positionOffset &&
                                        owner->advancePositions);
        }
#endif
        g->rowTransfer->Reset(std::move(rows));
    } else if (!V41FullReserve(*g, f.inputSegments)) return false;
    bool directHash = false;
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    // Packet graphs consume only the rows belonging to this rank. Reading the
    // active request's compressed history here also handles migration and padding.
    directHash = !g->verifying && !g->inputPackets.empty() && !g->engramWorker &&
        std::all_of(g->engrams.begin(), g->engrams.end(), [&](const auto &entry) {
            return entry.second->buckets.size() == size_t((engram_max_ngram_size - 1) * engram_n_heads) &&
                   entry.second->table->gpuTable && !entry.second->asyncGraph;
        });
#endif
    if (!directHash && !staged) {
        const auto snapshots = V41SnapshotSegments(f.inputSegments, engram_max_ngram_size);
        for (auto &entry : g->engrams) {
            auto &e = *entry.second;
            bool dead = false;
            V41BuildEngramInputs(engramMeta, e.index, engram_max_ngram_size, engram_n_heads, snapshots,
                                 g->batch, e.hostRows, e.hostMask, dead);
            if (e.hostRows.size() * 2 != e.rows.Count(0) || e.hostMask.size() != (size_t)g->batch ||
                std::any_of(e.hostRows.begin(), e.hostRows.end(),
                            [&](int64_t row) { return row < 0 || row >= e.table->rows; }))
                return false;
        }
    }
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    for (size_t r = 0; r < g->inputPackets.size(); ++r) if (g->inputPackets[r]) {
        auto &p = *g->inputPackets[r];
        auto *data = static_cast<uint8_t *>(p.host);
        const auto &ids = g->ownerHostIds[r];
        std::memcpy(data, ids.data(), ids.size() * sizeof(int));
        if (p.positionOffset) {
            auto *positions = reinterpret_cast<int *>(data + p.positionOffset);
            for (size_t i = 0; i < p.order.size(); ++i)
                positions[i] = f.inputSegments[p.order[i]].startPos;
        }
        for (const auto &entry : g->engrams) {
            const auto &e = *entry.second;
            const auto &o = *e.owners[r];
            const size_t heads = e.hostRows.size() / g->batch;
            auto *rows = reinterpret_cast<int64_t *>(data + o.rowOffset);
            auto *mask = reinterpret_cast<float *>(data + o.maskOffset);
            for (size_t i = 0; i < p.order.size(); ++i) {
                const int index = p.order[i];
                auto *dst = rows + i * heads;
                if (!directHash) {
                    std::memcpy(dst, e.hostRows.data() + size_t(index) * heads, heads * sizeof(int64_t));
                    mask[i] = e.hostMask[index];
                    continue;
                }
                const auto &seg = f.inputSegments[index];
                const auto &history = seg.state->engramHistory;
                if (seg.seqlen != 1 || seg.startPos < 0 || size_t(seg.startPos) >= history.size()) return false;
                mask[i] = history[seg.startPos] < 0 ? 0.f : 1.f;
                const auto &multipliers = engramMeta.multipliers[e.index];
                uint64_t rolling = 0;
                bool pad = false;
                for (int j = 0; j < engram_max_ngram_size; ++j) {
                    pad = pad || seg.startPos < j || history[seg.startPos - j] < 0;
                    const int token = pad ? engramMeta.padCompressedId : history[seg.startPos - j];
                    rolling ^= uint64_t(token) * uint64_t(multipliers[j]);
                    if (j) for (int h = 0; h < engram_n_heads; ++h) {
                        const size_t col = size_t(j - 1) * engram_n_heads + h;
                        dst[col] = e.buckets[col].Row(rolling);
                    }
                }
            }
        }
    }
#endif
    // CPU rows are prepared alongside layer 0. Each Engram consumer
    // waits on its own publication, so late I/O cannot expose stale rows.
    // Finish on exceptional launch paths before hostRows can be reused.
    struct JoinEngram {
        V41FullEngramWorker *worker;
        void Finish() { if (worker) std::exchange(worker, nullptr)->Finish(); }
        ~JoinEngram() { try { Finish(); } catch (...) {} }
    } joinEngram{g->engramWorker.get()};
    if (g->engramWorker) g->engramWorker->Start();
    for (auto &entry : g->engrams) {
        auto &e = *entry.second;
        if (!e.table->gpuTable && !e.asyncGraph)
            GatherEngramRows(entry.first, e.hostRows, g->batch, e.hostGathered);
    }
    // Recalibrate initially and periodically without putting timing nodes on
    // every replay. CPU cost observations continue on every token.
    const bool timed = !g->untimedExec || g->replays < 8 || (g->replays % 32 == 0);
    std::vector<int> uploaded(g->devices.size(), 0), ok(g->devices.size(), 0);
    std::atomic<int> ready{0};
    int indexVariant = -1;
    if (g->allIndexLimit[0] || g->allIndexLimit[1]) {
        int64_t longest = 0;
        for (const auto &s : f.inputSegments) longest = std::max(longest, int64_t(s.startPos) + s.seqlen);
        if (longest <= g->allIndexLimit[0]) indexVariant = 0;
        else if (longest <= g->allIndexLimit[1]) indexVariant = 1;
    }
    auto dispatchReplay = [&] { return V41FullDispatch(*g, [&](int rank, int device) {
        if (!residentInputs) {
            uploaded[rank] = staged || V41FullUploadRank(*g, f.inputSegments, rank, device);
            auto *packetDestination = staged ? static_cast<uint8_t *>(V41FullLocal(g->requests, device).cudaData) +
                size_t(block_cnt) * g->batch * sizeof(FastllmV41FullDecodeRequest) : nullptr;
            bool published = false;
            if (g->attentionRuntime && g->attentionGraphs[rank]) {
                auto &local = *g->attentionGraphs[rank];
                const auto &segments = attentionSegments[rank];
                const bool valid = pages && segments.size() == 1 && segments.front().offset == 0 &&
                    segments.front().seqlen == local.batch && pages->Matches(local, *segments.front().state);
                auto *destination = V41FullLocal(local.requests, device).cudaData;
                if (valid && staged && device == staged->verifyProducer && local.lastVerifyReader)
                    published = FastllmCudaV41PublishVerifyInputs(pages->slot.get(), destination, packetDestination,
                        staged->verifyPacket, staged->verifyPacketBytes, staged->verifyReady, local.lastVerifyReader,
                        g->ready[rank]);
                uploaded[rank] &= published || (valid && FastllmCudaV41ConsumeVerifyPages(pages->slot.get(), destination)) ||
                    V41FullUploadRank(local, segments, 0, device);
            }
            if (staged && uploaded[rank] && !published) {
                FastllmCudaCurrentThreadStreamWaitEvent(staged->verifyReady);
                uploaded[rank] = FastllmCudaMemcpyPeerAsyncCurrentThread(device, packetDestination, staged->verifyProducer,
                    staged->verifyPacket, staged->verifyPacketBytes);
            }
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (ready.load(std::memory_order_acquire) < (int)g->devices.size())
                std::this_thread::yield();
        } else if (const auto &owner = g->attentionGraphs[rank]) {
            // Reserve verified the same request, lease, addresses and next position.
            // The embedding prologue publishes the absolute row positions before attention.
            for (auto &row : owner->fixedRows) ++row.position;
        }
        if (residentInputs || std::all_of(uploaded.begin(), uploaded.end(), [](int v) { return v == 1; })) {
            const auto &handles = g->rankMoeHandles[rank];
            HybridMoeDecodeStep rankStep(true);
            if (!handles.empty()) {
                auto &table = weights[g->rankMoeFirstLayer[rank]];
                rankStep.Begin(table.data(), table.size(), num_experts_per_tok);
                AssertInFastLLM(bool(rankStep), "TP full graph cache policy unavailable");
            }
            const bool rankTimed = g->replays < 8 || g->replays % 32 == 0 || !g->rankUntimedExecs[rank];
            void *exec = g->tensorParallel
                ? (rankTimed ? g->execs[rank] : g->rankUntimedExecs[rank])
                : (timed ? g->execs[rank] : g->untimedExec);
            if (indexVariant >= 0 && g->allIndexExecs[indexVariant][rank])
                exec = g->allIndexExecs[indexVariant][rank];
            auto replay = [&] {
                if (!handles.empty()) return FastllmCudaLaunchNativeMoeFullGraph(
                    exec, handles.data(), handles.size(), rankTimed);
                return !g->moeHandles.empty()
                    ? FastllmCudaLaunchNativeMoeFullGraph(exec, g->moeHandles.data(), g->moeHandles.size(), timed)
                    : FastllmCudaGraphLaunch(exec);
            };
            const bool success = replay();
            ok[rank] = success ? 1 : -1;
            if (g->verifying || g->captureGreedy) FastllmCudaEventRecordCurrentThread(g->ready[rank]);
            if (g->attentionRuntime && g->attentionGraphs[rank] && g->attentionGraphs[rank]->lastVerifyReader)
                FastllmCudaEventRecordCurrentThread(g->attentionGraphs[rank]->lastVerifyReader);
        }
    }); };
    const bool launched = g->diskMoe ? g->diskMoe->Launch(dispatchReplay) : dispatchReplay();
    joinEngram.Finish();
    AssertInFastLLM(launched && std::all_of(ok.begin(), ok.end(), [](int v) { return v == 1; }),
                    "full decode graph replay failed");
    if (g->verifying) {
        // Root sampling reads peer-written vocabulary shards after every rank.
        FastllmCudaSetDevice(g->devices.front());
        for (void *event : g->ready) FastllmCudaCurrentThreadStreamWaitEvent(event);
        for (const auto &segment : f.inputSegments)
            V41FullBindVerify(g, segment, v41DsparkTargetLayerIds);
        // Rank-local verify views and their commit closures are host metadata.
        // Their GPU producers are already joined on the sampling stream above.
        for (size_t rank = 0; rank < attentionSegments.size(); ++rank)
            for (const auto &segment : attentionSegments[rank])
                V41FullBindVerify(g->attentionGraphs[rank], segment, {});
        if (f.pipeline) f.pipeline->fullVerify = true;
    }
    ++g->replays;
    if (V41DecodeCudaGraphVerbose() && g->replays <= 2)
        fprintf(stderr, "[Fastllm] DeepSeek-V4.1 full graph replay: batch=%d attention_dp=%d ranks=%zu\n",
            g->batch, v41AttentionDP, g->devices.size());
    if (g->attentionRuntime) {
        for (size_t r = 0; r < g->attentionGraphs.size(); ++r)
            if (g->attentionGraphs[r]) V41FullAdvance(*g->attentionGraphs[r], attentionSegments[r]);
    } else V41FullAdvance(*g, f.inputSegments);
    f.curHidden = &f.ws->hiddenStates;
    f.preMixPtr = &f.ws->ffnPre;
    if (!inputsPrepared) ++f.graphState->replayCount;
    if (g->captureGreedy) {
        if (g->greedyMapped.empty()) {
            f.fullGraphGreedyIds = &g->greedyIds;
            f.fullGraphGreedyScores = &g->greedyScores;
        } else f.fullGraphGreedyHost = &g->greedyHost;
        f.fullGraphGreedyEvents = &g->greedyReady;
    }
    f.fullGraphHead = g->endLayer == block_cnt ? &g->head : nullptr;
    f.fullGraphLogits = g->endLayer != block_cnt ? nullptr : g->verifying ? &g->gatheredLogits : &g->logits;
    return true;
}
