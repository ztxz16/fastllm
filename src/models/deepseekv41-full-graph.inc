// Included after ForwardWorkspace: reuses the ordinary model's arithmetic and
// captured projection/MoE graphs, replacing only dynamic decode orchestration.
namespace {
struct V41FullLayer {
    FastllmV41FullDecodeConfig config;
    int source = -1;
    Data joined, scores, latents, keys, savedKV, savedRawKV, savedRawScore;
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
    bool verifying = false;
    std::vector<std::unique_ptr<V41FullLayer>> layers;
    std::map<int, std::unique_ptr<V41FullEngram>> engrams;
    Data topk, candidates, head, norm, normFloat, logits, requests, gatheredLogits;
    std::map<int, Data> mainHidden;
    std::vector<void *> ready;
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
            if (rank < ready.size() && ready[rank])
                FastllmCudaEventDestroy(ready[rank]);
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
    if ((int)segments.size() != (g.verifying ? 1 : g.batch))
        return false;
    for (const auto &s : segments) {
        const bool validMode = g.verifying
            ? s.spec && s.spec->deferWindow && s.spec->wantAllTokens && s.seqlen == g.batch
            : !s.spec && s.seqlen == 1;
        if (!s.state || !validMode || s.startPos < 1 ||
            (int)s.state->layers.size() != (int)g.layers.size())
            return false;
        for (size_t l = 0; l < g.layers.size(); ++l) {
            const auto &cfg = g.layers[l]->config;
            auto &cache = s.state->layers[l];
            if (cfg.ratio && (s.startPos + (long long)s.seqlen) / cfg.ratio > g.maxKeys)
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
            const int rows = (s.startPos + s.seqlen) / cfg.ratio;
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
    std::vector<FastllmV41FullDecodeRequest> requests(g.layers.size() * g.batch);
    std::vector<void *> plans(g.layers.size(), nullptr);
    auto ptr = [device](Data &d) -> void * {
        auto it = d.multiDeviceDatas.find(device);
        return it == d.multiDeviceDatas.end() || !it->second ? nullptr : it->second->cudaData;
    };
    for (size_t l = 0; l < g.layers.size(); ++l) {
        const auto &layer = *g.layers[l];
        if (layer.plans.empty())
            continue;
        for (int i = 0; i < g.batch; ++i) {
            const auto &segment = segments[g.verifying ? 0 : i];
            auto &cache = segment.state->layers[l];
            auto &source = segment.state->layers[layer.source < 0 ? l : layer.source];
            requests[l * g.batch + i] = {ptr(cache.windowKV), ptr(source.compressedKV), ptr(source.indexK),
                ptr(cache.rawTailKV), ptr(cache.rawTailScore), segment.startPos + (g.verifying ? i : 0)};
        }
        plans[l] = layer.plans[rank];
    }
    if (!FastllmCudaV41UploadFullDecodeBatch(ptr(g.requests), plans.data(), plans.size(),
                                           requests.data(), g.batch)) return false;
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
void V41FullBindVerify(V41FullGraph &g, const DeepSeekV41Segment &segment,
                      const std::vector<int> &targetLayers) {
    // Scratch views borrow graph buffers until commit/draft completes,
    // before the next target forward can overwrite them.
    auto borrow = [&](Data &view, const Data &source, int rows) {
        AssertInFastLLM(V41DecodeSliceView(view, source, 0, g.devices, rows),
                        "full verify buffer view failed");
    };
    auto &scratch = *segment.spec;
    const int layers = g.layers.size();
    scratch.windowKV.resize(layers);
    scratch.rawKV.resize(layers);
    scratch.rawScore.resize(layers);
    scratch.prevRawTail.resize(layers);
    scratch.prevBlocks.resize(layers);
    for (int l = 0; l < layers; ++l) {
        const auto &layer = *g.layers[l];
        borrow(scratch.windowKV[l], layer.savedKV, g.batch);
        if (!layer.config.kvSource) continue;
        scratch.prevRawTail[l] = segment.startPos % layer.config.ratio;
        scratch.prevBlocks[l] = segment.startPos / layer.config.ratio;
        const int rows = scratch.prevRawTail[l] + segment.seqlen;
        borrow(scratch.rawKV[l], layer.savedRawKV, rows);
        if (layer.config.ratio > 1)
            borrow(scratch.rawScore[l], layer.savedRawScore, rows);
    }
    if (scratch.captureMain) {
        scratch.mainHidden.resize(targetLayers.size());
        for (size_t i = 0; i < targetLayers.size(); ++i)
            borrow(scratch.mainHidden[i], g.mainHidden.at(targetLayers[i]), g.batch);
    }
}
} // namespace

void DeepSeekV41Model::BuildFullDecodeGraph(
    int batch, const std::vector<DeepSeekV41Segment> &segments) {
    const bool verifying = segments.size() == 1 && segments[0].spec && segments[0].spec->deferWindow;
    // Small BF16-KV target chunks share decode's fixed operands and dynamic
    // request descriptors. Other shapes keep the segmented/eager path.
    if (batch < 1 || batch > (verifying ? 8 : 16) || executionMode != ExecutionMode::Gpu ||
        (!verifying && v41DsparkEnabled) ||
        (kvCacheDataType == DataType::FP8_E4M3 || kvCacheDataType == DataType::FP4_E2M1) ||
        (!verifying && !v41CudaGraphWarmingUp) || head_dim_full != 512 || hc_mult != 4 || index_head_dim != 128 ||
        weight["head.weight"].dataType != DataType::FLOAT32)
        return;
    // Explicit reference/debug modes must retain their selected arithmetic.
    if (V41DecodeCudaGraphReplayMask() != 15 || V41EnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT") ||
        V41EnvFlag("FASTLLM_DSV41_LEGACY_ATTN") || V41EnvFlag("FASTLLM_DSV41_LEGACY_INDEXER") ||
        V41EnvFlag("FASTLLM_DSV41_LEGACY_TOPK") || V41EnvFlag("FASTLLM_DSV41_LEGACY_ROTARY"))
        return;
    auto slot = v41CudaGraphSlots.find(verifying ? -batch : batch);
    if (slot == v41CudaGraphSlots.end())
        return;
    auto state = V41GetCudaGraphState(slot->second);
    if (!state->captured || !state->tensorParallel || state->fullDecode ||
        state->segments.size() != (size_t)block_cnt * 4)
        return;
    for (const auto &s : state->segments)
        if (s.passthrough || s.graphs.size() != state->devices.size())
            return;
    for (int layer = 0; layer < block_cnt; ++layer)
        if (!state->segments[layer * 4 + 3].containsGpuMoe) return;
    for (auto &table : engramTables)
        if (!std::static_pointer_cast<V41EngramTable>(table)->gpuTable)
            return;
    auto g = std::make_shared<V41FullGraph>();
    g->devices = state->devices;
    g->batch = batch;
    g->verifying = verifying;
    g->pieces = std::make_shared<DeepSeekV41CudaGraphState>();
    g->pieces->PrepareDevices(g->devices, true);
    auto &ws = *state->workspace;
    ScopedTpDispatch dispatch(true, g->devices);
    const int original = FastllmCudaGetDevice();
    auto alloc = [&](Data &data, DataType dtype, const std::vector<int> &dims) {
        V41AllocLike(data, dtype, dims, ws.hiddenStates, g->devices);
    };
    alloc(g->requests, DataType::INT8,
          {1, int(block_cnt * batch * sizeof(FastllmV41FullDecodeRequest))});
    const int blockSize = std::max(1, candidate_block_size);
    alloc(g->topk, DataType::INT32, {1, batch, index_topk});
    alloc(g->candidates, DataType::INT8, {1, batch, (g->maxKeys + blockSize - 1) / blockSize});
    for (int l = 0; l < block_cnt; ++l) {
        auto layer = std::make_unique<V41FullLayer>();
        auto &c = layer->config;
        c.batch = batch;
        c.verifying = verifying;
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
        if (verifying) {
            alloc(layer->savedKV, DataType::BFLOAT16, {1, batch, 512});
            if (c.kvSource) {
                alloc(layer->savedRawKV, DataType::FLOAT32, {1, batch + c.ratio - 1, 512});
                if (c.ratio == 2)
                    alloc(layer->savedRawScore, DataType::FLOAT32, {1, batch + c.ratio - 1, 512});
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
    // Lazy verification capture runs after this call's embedding. The warm
    // Engram children modify hiddenStates; restore its contents (not address)
    // before replay, including the fallback path.
    Data initialHidden;
    if (verifying) V41Assign(initialHidden, ws.hiddenStates, g->devices);
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
            if (verifying && std::find(v41DsparkTargetLayerIds.begin(),
                    v41DsparkTargetLayerIds.end(), l) != v41DsparkTargetLayerIds.end()) {
                capture([&] {
                    V41HcApplyPre(ws.hiddenStates, ws.dsparkPreMix, g->mainHidden[l]);
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
                b.requests = static_cast<FastllmV41FullDecodeRequest *>(ptr(g->requests)) + l * batch;
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
                if (verifying) {
                    b.savedKV = ptr(layer.savedKV);
                    b.savedRawKV = ptr(layer.savedRawKV);
                    b.savedRawScore = ptr(layer.savedRawScore);
                }
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
            if (verifying) {
                if (g->gatheredLogits.dims.empty()) {
                    FastllmCudaSetDevice(g->devices.front());
                    g->gatheredLogits.dataType = DataType::FLOAT32;
                    g->gatheredLogits.dataDevice = DataDevice::CUDA;
                    g->gatheredLogits.dataDeviceIds = {g->devices.front()};
                    g->gatheredLogits.Resize(g->logits.dims);
                    g->gatheredLogits.Allocate();
                }
                MultiCudaRunDeviceCallbacks(g->devices, [&](int, int device) {
                    const auto &local = *g->logits.multiDeviceDatas.at(device);
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
        if (verifying) g->ready.resize(g->devices.size(), nullptr);
        for (size_t rank = 0; rank < g->devices.size(); ++rank) {
            FastllmCudaSetDevice(g->devices[rank]);
            if (verifying) {
                g->ready[rank] = FastllmCudaEventCreate();
                if (!g->ready[rank]) throw std::runtime_error("full verify completion event");
            }
            if (!FastllmCudaGraphCompose(children[rank].data(), children[rank].size(),
                                         &g->graphs[rank], &g->execs[rank]))
                throw std::runtime_error("full graph composition failed");
        }
        state->fullDecode = g;
        fprintf(stderr,
                "[Fastllm] DeepSeek-V4.1 full %s graph ready: %s=%d, ranks=%zu, one graph "
                "per rank/step\n",
                verifying ? "verify" : "decode", verifying ? "tokens" : "batch", batch, g->devices.size());
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
        MultiCudaRunDeviceCallbacks(g->devices, [&](int, int device) {
            FastllmCudaCopyFromDeviceToDevice(ws.hiddenStates.multiDeviceDatas.at(device)->cudaData,
                initialHidden.multiDeviceDatas.at(device)->cudaData,
                initialHidden.multiDeviceDatas.at(device)->GetBytes());
        });
        for (int device : g->devices) FastllmCudaSyncDevice(device);
    }
    FastllmCudaSetDevice(original);
}

bool DeepSeekV41Model::TryFullDecodeGraph(ForwardWorkspace &f) {
    // Verification warmup uses the same composition/replay path as requests.
    // Ordinary decode still builds its full graph explicitly after warmup.
    if (!f.graphReplay || !f.graphState || (v41CudaGraphWarmingUp && !f.dsparkVerify) || f.quantKV ||
        f.hasImageTokens || f.inputEmbeds)
        return false;
    if (f.dsparkVerify && !f.graphState->fullVerifyAttempted) {
        f.graphState->fullVerifyAttempted = true;
        BuildFullDecodeGraph(f.seqlen, f.inputSegments);
    }
    if (!f.graphState->fullDecode) return false;
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
        if (std::all_of(uploaded.begin(), uploaded.end(), [](int v) { return v == 1; })) {
            ok[rank] = FastllmCudaGraphLaunch(g->execs[rank]) ? 1 : -1;
            if (g->verifying) FastllmCudaEventRecordCurrentThread(g->ready[rank]);
        }
    });
    AssertInFastLLM(launched && std::all_of(ok.begin(), ok.end(), [](int v) { return v == 1; }),
                    "full decode graph replay failed");
    if (g->verifying) {
        // Root sampling reads peer-written vocabulary shards after every rank.
        FastllmCudaSetDevice(g->devices.front());
        for (void *event : g->ready) FastllmCudaCurrentThreadStreamWaitEvent(event);
        V41FullBindVerify(*g, f.inputSegments[0], v41DsparkTargetLayerIds);
    }
    V41FullAdvance(*g, f.inputSegments);
    f.curHidden = &f.ws->hiddenStates;
    f.preMixPtr = &f.ws->ffnPre;
    f.fullGraphHead = &g->head;
    f.fullGraphLogits = g->verifying ? &g->gatheredLogits : &g->logits;
    return true;
}
