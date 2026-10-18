// Included after ForwardWorkspace: reuses the ordinary model's arithmetic and
// captured projection/MoE graphs, replacing only dynamic decode orchestration.
namespace {
struct V41FullLayer {
    FastllmV41FullDecodeConfig config;
    int source = -1;
    Data joined, scores, latents, keys;
    std::vector<void *> plans;
};
struct V41FullEngram {
    int index = 0;
    std::shared_ptr<V41EngramTable> table;
    Data rows, mask, gathered, kv, quantized;
    std::vector<int64_t> hostRows;
    std::vector<float> hostMask;
};
struct V41FullGraph {
    std::vector<int> devices;
    int batch = 1, maxKeys = 131072;
    std::vector<std::unique_ptr<V41FullLayer>> layers;
    std::map<int, std::unique_ptr<V41FullEngram>> engrams;
    Data topk, candidates, head, norm, normFloat, logits;
    std::shared_ptr<DeepSeekV41CudaGraphState> pieces;
    std::vector<void *> graphs, execs;
    ~V41FullGraph() {
        const int old = FastllmCudaGetDevice();
        for (int d : devices)
            FastllmCudaSyncDevice(d);
        for (size_t rank = 0; rank < devices.size(); ++rank) {
            FastllmCudaSetDevice(devices[rank]);
            if (rank < execs.size() && execs[rank])
                FastllmCudaGraphExecDestroy(execs[rank]);
            if (rank < graphs.size() && graphs[rank])
                FastllmCudaGraphDestroy(graphs[rank]);
        }
        pieces.reset();
        for (auto &layer : layers)
            for (void *plan : layer->plans)
                FastllmCudaV41DestroyFullDecode(plan);
        FastllmCudaSetDevice(old);
    }
};

// All allocations happen before graph launch. Moving a cache is safe because
// graph kernels load its current pointer from request metadata on every replay.
bool V41FullReserve(V41FullGraph &g, const std::vector<DeepSeekV41Segment> &segments) {
    if ((int)segments.size() != g.batch)
        return false;
    for (const auto &s : segments) {
        if (!s.state || s.spec || s.seqlen != 1 || s.startPos < 1 ||
            (int)s.state->layers.size() != (int)g.layers.size())
            return false;
        for (size_t l = 0; l < g.layers.size(); ++l) {
            const auto &cfg = g.layers[l]->config;
            auto &cache = s.state->layers[l];
            if (cfg.ratio && (s.startPos + 1LL) / cfg.ratio > g.maxKeys)
                return false;
            if (cache.windowKV.dataType != DataType::BFLOAT16 || !cache.windowKV.multiDeviceData ||
                !cache.windowKV.IsTensorParallelReplicated())
                return false;
            for (int device : g.devices) {
                auto it = cache.windowKV.multiDeviceDatas.find(device);
                if (it == cache.windowKV.multiDeviceDatas.end() || !it->second ||
                    !it->second->cudaData)
                    return false;
            }
        }
    }
    for (const auto &s : segments)
        for (size_t l = 0; l < g.layers.size(); ++l) {
            const auto &cfg = g.layers[l]->config;
            if (!cfg.kvSource)
                continue;
            auto &cache = s.state->layers[l];
            const int rows = (s.startPos + 1) / cfg.ratio;
            V41ReserveDecodeRows(cache.compressedKV, DataType::BFLOAT16, 512, rows, g.devices);
            if (cfg.indexSource)
                V41ReserveDecodeRows(cache.indexK, DataType::BFLOAT16, 128, rows, g.devices);
            if (cfg.ratio == 2) {
                V41ReserveDecodeRows(cache.rawTailKV, DataType::FLOAT32, 512, 2, g.devices);
                V41ReserveDecodeRows(cache.rawTailScore, DataType::FLOAT32, 512, 2, g.devices);
            }
        }
    return true;
}

bool V41FullUploadRank(V41FullGraph &g, const std::vector<DeepSeekV41Segment> &segments, int rank,
                       int device) {
    std::vector<FastllmV41FullDecodeRequest> requests(g.batch);
    auto ptr = [device](Data &d) -> void * {
        auto it = d.multiDeviceDatas.find(device);
        return it == d.multiDeviceDatas.end() || !it->second ? nullptr : it->second->cudaData;
    };
    for (size_t l = 0; l < g.layers.size(); ++l) {
        const auto &layer = *g.layers[l];
        if (layer.plans.empty())
            continue;
        for (int i = 0; i < g.batch; ++i) {
            auto &cache = segments[i].state->layers[l];
            auto &source = segments[i].state->layers[layer.source < 0 ? l : layer.source];
            requests[i] = {ptr(cache.windowKV),  ptr(source.compressedKV), ptr(source.indexK),
                           ptr(cache.rawTailKV), ptr(cache.rawTailScore),  segments[i].startPos};
        }
        if (!FastllmCudaV41UploadFullDecode(layer.plans[rank], requests.data(), requests.size()))
            return false;
    }
    for (auto &entry : g.engrams) {
        auto &e = *entry.second;
        FastllmCudaCopyFromHostToDevice(ptr(e.rows), e.hostRows.data(),
                                        e.hostRows.size() * sizeof(int64_t));
        FastllmCudaCopyFromHostToDevice(ptr(e.mask), e.hostMask.data(),
                                        e.hostMask.size() * sizeof(float));
    }
    return !FastllmCudaGetThreadError();
}

void V41FullAdvance(V41FullGraph &g, const std::vector<DeepSeekV41Segment> &segments) {
    auto resize = [&](Data &data, int rows) {
        if (data.dims.empty())
            return;
        auto dims = data.dims;
        dims[1] = rows;
        for (int d : g.devices)
            data.multiDeviceDatas.at(d)->Resize(dims);
        V41SyncRootFromReplica(data, g.devices);
    };
    for (const auto &s : segments)
        for (size_t l = 0; l < g.layers.size(); ++l) {
            auto &cache = s.state->layers[l];
            cache.totalLen = s.startPos + 1;
            const auto &cfg = g.layers[l]->config;
            if (!cfg.kvSource)
                continue;
            cache.compressedBlocks = (s.startPos + 1) / cfg.ratio;
            cache.rawTail = (s.startPos + 1) % cfg.ratio;
            resize(cache.compressedKV, cache.compressedBlocks);
            if (cfg.indexSource)
                resize(cache.indexK, cache.compressedBlocks);
            resize(cache.rawTailKV, cache.rawTail);
            resize(cache.rawTailScore, cache.rawTail);
        }
}
} // namespace

void DeepSeekV41Model::BuildFullDecodeGraph(
    int batch, const std::vector<std::shared_ptr<DeepSeekV41RequestState>> &states) {
    // First enable ordinary BF16-KV GPU decode. Packed KV, speculative verify,
    // other execution modes and unsupported shapes retain the existing path.
    if (batch != 1 || executionMode != ExecutionMode::Gpu || v41DsparkEnabled ||
        (kvCacheDataType == DataType::FP8_E4M3 || kvCacheDataType == DataType::FP4_E2M1) ||
        !v41CudaGraphWarmingUp || head_dim_full != 512 || hc_mult != 4 || index_head_dim != 128 ||
        weight["head.weight"].dataType != DataType::FLOAT32)
        return;
    // Explicit reference/debug modes must retain their selected arithmetic.
    if (V41DecodeCudaGraphReplayMask() != 15 || V41EnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT") ||
        V41EnvFlag("FASTLLM_DSV41_LEGACY_ATTN") || V41EnvFlag("FASTLLM_DSV41_LEGACY_INDEXER") ||
        V41EnvFlag("FASTLLM_DSV41_LEGACY_TOPK") || V41EnvFlag("FASTLLM_DSV41_LEGACY_ROTARY"))
        return;
    auto slot = v41CudaGraphSlots.find(batch);
    if (slot == v41CudaGraphSlots.end())
        return;
    auto state = V41GetCudaGraphState(slot->second);
    if (!state->captured || !state->tensorParallel || state->fullDecode ||
        state->segments.size() != (size_t)block_cnt * 4)
        return;
    for (const auto &s : state->segments)
        if (s.passthrough || s.graphs.size() != state->devices.size())
            return;
    for (auto &table : engramTables)
        if (!std::static_pointer_cast<V41EngramTable>(table)->gpuTable)
            return;
    auto g = std::make_shared<V41FullGraph>();
    g->devices = state->devices;
    g->batch = batch;
    g->pieces = std::make_shared<DeepSeekV41CudaGraphState>();
    g->pieces->PrepareDevices(g->devices, true);
    auto &ws = *state->workspace;
    ScopedTpDispatch dispatch(true, g->devices);
    const int original = FastllmCudaGetDevice();
    std::vector<DeepSeekV41Segment> segments(batch);
    for (int i = 0; i < batch; ++i) {
        segments[i].state = states[i];
        segments[i].offset = i;
        segments[i].seqlen = 1;
        segments[i].startPos = states[i]->totalLen - 1;
    }
    auto alloc = [&](Data &data, DataType dtype, const std::vector<int> &dims) {
        V41AllocLike(data, dtype, dims, ws.hiddenStates, g->devices);
    };
    const int blockSize = std::max(1, candidate_block_size);
    alloc(g->topk, DataType::INT32, {1, batch, index_topk});
    alloc(g->candidates, DataType::INT8, {1, batch, (g->maxKeys + blockSize - 1) / blockSize});
    for (int l = 0; l < block_cnt; ++l) {
        auto layer = std::make_unique<V41FullLayer>();
        auto &c = layer->config;
        c.batch = batch;
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
        if (c.kvSource) {
            alloc(layer->joined, DataType::FLOAT32, {batch, c.ratio, 512});
            if (c.ratio == 2)
                alloc(layer->scores, DataType::FLOAT32, {batch, c.ratio, 512});
            alloc(layer->latents, DataType::BFLOAT16, {1, batch, 512});
            if (c.indexSource)
                alloc(layer->keys, DataType::BFLOAT16, {1, batch, 128});
        }
        g->layers.emplace_back(std::move(layer));
    }
    if (!V41FullReserve(*g, segments))
        return;
    const auto snapshots = V41SnapshotSegments(segments, engram_max_ngram_size);
    for (int i = 0; i < (int)engram_layer_ids.size(); ++i) {
        auto e = std::make_unique<V41FullEngram>();
        e->index = i;
        e->table = std::static_pointer_cast<V41EngramTable>(engramTables[i]);
        bool dead = false;
        V41BuildEngramInputs(engramMeta, i, engram_max_ngram_size, engram_n_heads, snapshots, batch,
                             e->hostRows, e->hostMask, dead);
        alloc(e->rows, DataType::INT32, {1, (int)e->hostRows.size() * 2});
        alloc(e->mask, DataType::FLOAT32, {1, batch});
        alloc(e->gathered, DataType::BFLOAT16,
              {1, batch, (int)e->hostRows.size() / batch * e->table->dim});
        g->engrams[engram_layer_ids[i]] = std::move(e);
    }
    bool poolOpen = false;
    try {
        std::vector<std::vector<const void *>> children(g->devices.size());
        auto capture = [&](const std::function<void()> &body) {
            body(); // Warm GEMM plans and allocate persistent operands before capture.
            for (int d : g->devices)
                FastllmCudaSyncDevice(d);
            if (!poolOpen) {
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                if (!FastllmCudaGraphMemoryPoolBegin())
                    throw std::runtime_error("full graph memory pool");
                poolOpen = true;
            }
            if (!V41CaptureGraphSegment(*g->pieces, body))
                throw std::runtime_error("full graph child capture");
            const auto &s = g->pieces->segments.back();
            for (size_t rank = 0; rank < children.size(); ++rank)
                children[rank].push_back(s.graphs[rank]);
        };
        for (int l = 0; l < block_cnt; ++l) {
            const std::string prefix = "layers." + std::to_string(l);
            auto engram = g->engrams.find(l);
            if (engram != g->engrams.end()) {
                auto &e = *engram->second;
                MultiCudaRunDeviceCallbacks(g->devices, [&](int rank, int device) {
                    V41FullUploadRank(*g, segments, rank, device);
                });
                capture([&] {
                    MultiCudaRunDeviceCallbacks(g->devices, [&](int, int device) {
                        bool ok = FastllmCudaV41GatherEngramDevice(
                            e.table->gpuTable,
                            static_cast<const int64_t *>(
                                e.rows.multiDeviceDatas.at(device)->cudaData),
                            e.hostRows.size(), *e.gathered.multiDeviceDatas.at(device));
                        AssertInFastLLM(ok, "full graph Engram gather failed");
                    });
                    LinearWithActivationQuant(e.gathered, prefix + ".engram.wkv.weight", e.kv, true,
                                              &e.quantized);
                    V41EngramApply(ws.hiddenStates, e.kv, weight[prefix + ".engram.q_weight"],
                                   weight[prefix + ".engram.k_weight"], &e.mask, rms_norm_eps);
                });
            }
            for (size_t rank = 0; rank < children.size(); ++rank)
                children[rank].push_back(state->segments[l * 4].graphs[rank]);
            auto &layer = *g->layers[l];
            if (layer.config.kvSource)
                PrepareMultiCudaReplicatedData(weight[prefix + ".attn.compressor.norm.weight"],
                                               g->devices, true);
            bool plansOk = true;
            layer.plans.resize(g->devices.size(), nullptr);
            MultiCudaRunDeviceCallbacks(g->devices, [&](int rank, int device) {
                auto ptr = [device](Data &d) -> void * {
                    auto it = d.multiDeviceDatas.find(device);
                    return it == d.multiDeviceDatas.end() || !it->second ? nullptr
                                                                         : it->second->cudaData;
                };
                FastllmV41FullDecodeBuffers b;
                b.q = ptr(ws.q);
                b.kv = ptr(ws.kv);
                b.output = ptr(ws.attnOut);
                b.rawKV = ptr(ws.rawKVAll);
                b.rawScore = ptr(ws.rawScoreAll);
                b.joined = ptr(layer.joined);
                b.scores = ptr(layer.scores);
                b.latents = ptr(layer.latents);
                b.newIndexKeys = ptr(layer.keys);
                b.indexQ = ptr(ws.qIdxAll);
                b.indexWeights = ptr(ws.idxWeightsAll);
                b.topk = ptr(g->topk);
                b.candidates = ptr(g->candidates);
                b.sink = ptr(weight[prefix + ".attn.attn_sink"]);
                if (layer.config.kvSource)
                    b.compressNorm = ptr(weight[prefix + ".attn.compressor.norm.weight"]);
                layer.plans[rank] = FastllmCudaV41CreateFullDecode(layer.config, b);
            });
            for (void *plan : layer.plans)
                plansOk &= plan != nullptr;
            if (!plansOk)
                throw std::runtime_error("full graph dynamic plan rejected");
            MultiCudaRunDeviceCallbacks(g->devices, [&](int rank, int device) {
                AssertInFastLLM(V41FullUploadRank(*g, segments, rank, device),
                                "full graph initial metadata rejected");
            });
            capture([&] {
                MultiCudaRunDeviceCallbacks(g->devices, [&](int rank, int) {
                    AssertInFastLLM(FastllmCudaV41PrepareFullDecode(layer.plans[rank]),
                                    "full graph prepare failed");
                });
                if (layer.config.kvSource && layer.config.indexSource) {
                    LinearWithActivationQuant(layer.latents, prefix + ".attn.indexer.wk.weight",
                                              layer.keys, true);
                    V41RMSNormBF16(layer.keys, weight[prefix + ".attn.indexer.k_norm.weight"],
                                   rms_norm_eps, layer.keys);
                }
                MultiCudaRunDeviceCallbacks(g->devices, [&](int rank, int) {
                    AssertInFastLLM(FastllmCudaV41FinishFullDecode(layer.plans[rank]),
                                    "full graph attention failed");
                });
            });
            for (int piece = 1; piece < 4; ++piece)
                for (size_t rank = 0; rank < children.size(); ++rank)
                    children[rank].push_back(state->segments[l * 4 + piece].graphs[rank]);
        }
        capture([&] {
            V41HcApplyPre(ws.hiddenStates, ws.ffnPre, g->head);
            V41RMSNormBF16(g->head, weight["norm.weight"], rms_norm_eps, g->norm);
            ToDataType(g->norm, g->normFloat, DataType::FLOAT32);
            weight["head.weight"].tpLinearType = TP_LINEAR_ROW;
            Linear(g->normFloat, weight["head.weight"], *GetEmptyData(), g->logits);
        });
        if (!FastllmCudaGraphMemoryPoolEnd(g->pieces->reservedPointers))
            throw std::runtime_error("full graph pool finish");
        poolOpen = false;
        g->graphs.resize(g->devices.size(), nullptr);
        g->execs.resize(g->devices.size(), nullptr);
        for (size_t rank = 0; rank < g->devices.size(); ++rank) {
            FastllmCudaSetDevice(g->devices[rank]);
            if (!FastllmCudaGraphCompose(children[rank].data(), children[rank].size(),
                                         &g->graphs[rank], &g->execs[rank]))
                throw std::runtime_error("full graph composition failed");
        }
        state->fullDecode = g;
        fprintf(stderr,
                "[Fastllm] DeepSeek-V4.1 full decode graph ready: batch=%d, ranks=%zu, one graph "
                "per rank/step\n",
                batch, g->devices.size());
    } catch (const std::exception &error) {
        if (poolOpen)
            FastllmCudaGraphMemoryPoolAbort();
        FastllmCudaClearThreadError();
        FastllmCudaClearGraphError();
        fprintf(stderr,
                "[Fastllm] DeepSeek-V4.1 full graph unavailable: %s; retaining segmented graphs\n",
                error.what());
    }
    FastllmCudaSetDevice(original);
}

bool DeepSeekV41Model::TryFullDecodeGraph(ForwardWorkspace &f) {
    if (!f.graphReplay || !f.graphState || !f.graphState->fullDecode || f.dsparkVerify ||
        v41CudaGraphWarmingUp || f.quantKV || f.hasImageTokens || f.inputEmbeds)
        return false;
    auto g = std::static_pointer_cast<V41FullGraph>(f.graphState->fullDecode);
    if (!V41FullReserve(*g, f.inputSegments))
        return false;
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
    std::vector<int> uploaded(g->devices.size(), 0), ok(g->devices.size(), 0);
    std::atomic<int> ready{0};
    const bool launched = MultiCudaRunDeviceCallbacks(g->devices, [&](int rank, int device) {
        uploaded[rank] = V41FullUploadRank(*g, f.inputSegments, rank, device) ? 1 : 0;
        ready.fetch_add(1, std::memory_order_release);
        while (ready.load(std::memory_order_acquire) < (int)g->devices.size())
            std::this_thread::yield();
        if (std::all_of(uploaded.begin(), uploaded.end(), [](int v) { return v == 1; }))
            ok[rank] = FastllmCudaGraphLaunch(g->execs[rank]) ? 1 : -1;
    });
    AssertInFastLLM(launched && std::all_of(ok.begin(), ok.end(), [](int v) { return v == 1; }),
                    "full decode graph replay failed");
    V41FullAdvance(*g, f.inputSegments);
    f.curHidden = &f.ws->hiddenStates;
    f.preMixPtr = &f.ws->ffnPre;
    f.fullGraphHead = &g->head;
    f.fullGraphLogits = &g->logits;
    return true;
}
