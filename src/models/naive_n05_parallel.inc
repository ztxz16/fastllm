// Naive adapter for the shared group runtime. Included within namespace fastllm.
void NaiveN05FlashModel::ConfigureAttentionParallel(int dp) {
    if (dp < 1) throw std::invalid_argument("attention DP must be positive");
    if (tpPrepared || !draftContexts.empty() || !responseContextDict.dicts.empty())
        throw std::invalid_argument("configure attention DP before warmup or requests");
    if (dp == 1) { attentionDP = 1; return; }
#ifdef USE_CUDA
    if (indexerLayerSharding)
        throw std::invalid_argument("Naive attention DP supports replicated or context-sharded Indexer storage");
    parallel::Topology topology(tpDevices, dp, tpEp8Mega ? tpDevices.size() : 1);
    if (GetKVCacheInCPU() || dataType != BFLOAT16 || !FastllmNaiveKVType(kvCacheDataType))
        throw std::invalid_argument("Naive attention DP requires BF16 compute and BF16/FP8 GPU KV");
    for (auto cfg : {full, sliding})
        if (cfg.heads % topology.attentionTP ||
            (cfg.kvHeads % topology.attentionTP && topology.attentionTP % cfg.kvHeads))
            throw std::invalid_argument("Naive attention TP requires complete Q and KV head groups");
    const int previous = attentionDP;
    attentionDP = dp;
    try { if (kvMemConfig.enabled) ConfigureKvMem(kvMemConfig); }
    catch (...) { attentionDP = previous; throw; }
#else
    throw std::invalid_argument("Naive attention DP requires CUDA");
#endif
}

void NaiveN05FlashModel::InitAttentionParallelCache(
        const std::vector<std::pair<Data *, Data *>> &roots, int group) {
#ifdef USE_CUDA
    if (attentionDP == 1) return;
    AssertInFastLLM(attentionParallel && roots.size() == (size_t)block_cnt,
        "Naive attention DP cache requires initialized topology");
    if (group < 0) {
        if (roots.front().first->multiDeviceData) return;
        group = attentionParallel->nextGroup++ % attentionDP;
    }
    const auto &topology = attentionParallel->runtime.Layout();
    std::vector<int> devices;
    for (int rank : topology.groups[topology.attention.at(group)].ranks) devices.push_back(tpDevices[rank]);
    for (int layer = 0; layer < block_cnt; ++layer)
        for (Data *root : {roots[layer].first, roots[layer].second}) {
            if (root->multiDeviceData) {
                AssertInFastLLM(root->dataDeviceIds == devices && root->multiDeviceDatas.size() == devices.size(),
                    "Naive attention DP cache group changed during a request");
                continue;
            }
            AssertInFastLLM(root->dims.empty(), "Naive attention DP cannot import an unpartitioned cache");
            root->kvCacheFp8Indexer = !HasSeparateIndexer() && !slidingLayers[layer] &&
                root == roots[layer].first && fp8IndexerCache;
            root->multiDeviceData = true; root->dataDevice = CUDA;
            root->dataDeviceIds = devices; root->isKVCache = true;
            for (int device : devices) {
                auto *local = new Data(LayerKVType(layer));
                local->kvCacheFp8Indexer = root->kvCacheFp8Indexer;
                local->isKVCache = true; local->dataDevice = CUDA; local->dataDeviceIds = {device};
                root->multiDeviceDatas[device] = local;
            }
        }
#endif
}

Data NaiveN05FlashModel::ForwardAttentionParallel(const Data &inputIds, const Data &positions,
        std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config,
        TargetCapture *capture, LogitsSelection *selection, const TargetBatch *batch) {
#ifndef USE_CUDA
    throw std::invalid_argument("Naive attention DP requires CUDA");
#else
    // Reject unsupported layouts rather than replicating request KV across groups.
    AssertInFastLLM(!indexerLayerSharding,
        "Naive attention DP: use replicated or context-sharded Indexer storage.");
    auto &state = *attentionParallel;
    const auto &topology = state.runtime.Layout();
    const int sequences = batch ? batch->Size() : 1;
    const int total = inputIds.dims.at(1);
    const bool verifying = capture && capture->verifying;
    const int outputRows = verifying ? total : sequences;
    const int vocab = weight["lm_head.weight"].dims[0];
    auto key = [&](int seq, int layer) -> Data & { return batch ? batch->Key(seq, layer) : kv[layer].first; };
    auto value = [&](int seq, int layer) -> Data & { return batch ? batch->Value(seq, layer) : kv[layer].second; };
    Data hostPositions(positions); hostPositions.ToDevice(CPU);
    AssertInFastLLM(hostPositions.dataType == FLOAT32 && hostPositions.Count(0) == (size_t)total,
        "Naive attention DP requires one position per input row");
    const bool ownedDecode = batch && batch->Decode() && !verifying && !sparseKv &&
        mainLoop && mainLoop->get_id() == std::this_thread::get_id();
    auto matchesPlan = [&] {
        std::lock_guard<std::mutex> guard(tpRequestMutex);
        return state.plan.Matches(*batch, tpCacheEpoch.load(std::memory_order_relaxed), tpRequestIds);
    };
    const bool reusePlan = ownedDecode && state.planValid &&
        state.planKVType == kvCacheDataType && matchesPlan();
    // Do not publish a partially rebuilt plan after an exception.
    state.planValid = false;
    state.reuseBindings = false;
    if (reusePlan) {
        AssertInFastLLM(total == sequences, "Naive attention DP decode row count changed");
        const float *pos = (const float *)hostPositions.cpuData;
        for (auto &group : state.groups)
            for (size_t row = 0; row < group.sequenceIndices.size(); ++row) {
                const int seq = group.sequenceIndices[row];
                // Sampling/grammar constraints and positions can change without
                // changing KV ownership. Never cache these request values.
                group.configs[row] = batch->configs[seq];
                group.positions[row] = pos[seq];
            }
    } else {
        for (auto &group : state.groups) {
            group.batch.reset(); group.kv.clear(); group.configs.clear();
            group.lengths.clear(); group.positions.clear(); group.sequenceIndices.clear();
        }
        std::vector<int> owners;
        int offset = 0;
        for (int seq = 0; seq < sequences; ++seq) {
            Data &first = key(seq, 0);
            const int count = batch ? batch->lengths[seq] : total;
            int group;
            if (first.multiDeviceData) {
                auto it = std::find(tpDevices.begin(), tpDevices.end(), first.dataDeviceIds.at(0));
                AssertInFastLLM(it != tpDevices.end(), "Naive attention DP cache belongs to a different device domain");
                group = topology.AttentionOwner(it - tpDevices.begin());
            } else {
                AssertInFastLLM(first.dims.empty(), "Naive attention DP cannot import an unpartitioned cache");
                group = state.nextGroup++ % attentionDP;
            }
            std::vector<std::pair<Data *, Data *>> roots;
            for (int layer = 0; layer < block_cnt; ++layer) roots.emplace_back(&key(seq, layer), &value(seq, layer));
            InitAttentionParallelCache(roots, group);
            if (sparseKv) AttachSparseKv(roots);
            auto &local = state.groups[group];
            local.kv.insert(local.kv.end(), roots.begin(), roots.end());
            local.lengths.push_back(count);
            local.sequenceIndices.push_back(seq);
            local.configs.push_back(batch ? batch->configs[seq] : config);
            const float *pos = (const float *)hostPositions.cpuData;
            local.positions.insert(local.positions.end(), pos + offset, pos + offset + count);
            owners.insert(owners.end(), count, group);
            offset += count;
        }
        AssertInFastLLM(offset == total, "Naive attention DP row lengths do not match the input");
        parallel::RowPartition nextRows(attentionDP, owners);
        if (state.rows.localToGlobal != nextRows.localToGlobal) {
            // Row-transfer allocations and collective counts are graph operands.
            // Drop the old graph before changing either; shape misses stay eager.
            tpDecodeState.reset();
        }
        state.rows = std::move(nextRows);
        if (!state.transfer) state.transfer = std::make_unique<parallel::CudaRowTransfer>(state.runtime, state.rows);
        else state.transfer->Reset(state.rows);
        state.transfer->Reserve(size_t(embed_dim) * sizeof(uint16_t));
        for (auto &group : state.groups)
            group.batch = std::make_unique<TargetBatch>(TargetBatch{group.kv, group.configs, group.lengths, block_cnt});
    }
    if (!tpPagedCaches.empty()) {
        // Reserve on every owner before any attention/world collective. Pool
        // exhaustion must fail the step before peers enter NCCL.
        std::vector<std::exception_ptr> errors(tpDevices.size());
        tpWorkers.Run(tpDevices, [&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            const auto &group = state.groups[topology.AttentionOwner(rank)];
            std::vector<Data *> pending;
            try {
                for (size_t row = 0; row < group.lengths.size(); ++row) {
                    const int seq = group.sequenceIndices[row], count = group.lengths[row];
                    for (int layer = 0; layer < block_cnt; ++layer) {
                        const auto &cfg = slidingLayers[layer] ? sliding : full;
                        for (int part = 0; part < 2; ++part) {
                            Data &root = part ? value(seq, layer) : key(seq, layer);
                            Data &local = *root.multiDeviceDatas.at(tpDevices[rank]);
                            if (slidingLayers[layer]) {
                                if (sparseKv) continue;
                                const int capacity = tpSwaCacheCapacity;
                                ReserveCache(local, capacity, std::max(1, cfg.kvHeads / AttentionTPSize()) *
                                    (part ? cfg.valueDim : cfg.headDim), true, &key(seq, 0));
                            } else {
                                const int past = root.dims.empty() ? 0 : root.dims[1];
                                PreparePagedCache(local, *tpPagedCaches[rank][layer * 2 + part],
                                    past + count, &pending, &key(seq, 0));
                            }
                        }
                    }
                }
            } catch (...) { UploadPagedCacheTables(pending); throw; }
            UploadPagedCacheTables(pending);
        }, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
    }
    if (!tpSwaPrefillWorkspaces.empty() && (!batch || !batch->Decode())) {
        const int length = batch ? *std::max_element(batch->lengths.begin(), batch->lengths.end()) : total;
        state.runtime.Run([&](int rank, int) { PrepareSwaPrefillWorkspace(rank, length); });
    }
    Data embedding;
    if (!GetCudaEmbedding() || GetLowMemMode()) Embedding(inputIds, weight["model.embed_tokens.weight"], embedding);
    Data &joined = tpSamplingLogits;
    if (!isIntermediateChunkedPrefill) {
        FastllmCudaSetDevice(tpDevices.front());
        joined.dataType = FLOAT32; joined.UpdateUnitSize(); joined.Resize({outputRows, vocab});
        joined.ToDevice(CUDA, {tpDevices.front()}, false); joined.Allocate(false);
    }
    std::vector<PrefixSnapshots> prefixes(sequences);
    bool recording = false;
    if (saveHistoryChat && !verifying && (!draftEnabled ||
        (capture && capture->collectHidden && total > 1))) {
        for (int seq = 0; seq < sequences; ++seq) {
            const Data &first = key(seq, 0);
            prefixes[seq] = BeginPrefixSnapshots(first, first.dims.empty() ? 0 : first.dims[1],
                batch ? batch->lengths[seq] : total);
            recording = recording || !prefixes[seq].empty();
        }
    }
    std::vector<TargetCapture> captures(tpDevices.size());
    for (auto &feature : captures) {
        feature.collectHidden = capture && capture->collectHidden;
        feature.verifying = verifying;
        feature.borrowOutputs = capture && capture->borrowOutputs;
        if (recording) feature.prefixes = prefixes;
    }
    std::vector<int *> cacheLengths;
    LogitsSelection fullLogits;
    // Page growth and shared-tail COW can invalidate a binding even when the
    // row partition was reusable. Check identities again after reservation.
    const bool bindingsUnchanged = reusePlan && matchesPlan();
    state.reuseBindings = bindingsUnchanged;
    bool graphDecode = PrepareTensorParallelDecode(inputIds, kv, verifying, fullLogits,
        cacheLengths, batch, capture && capture->collectHidden);
    state.reuseBindings = false;
    // Ring candidates use the same batched SWA/commit kernels as world TP.
    // A cache without the candidate tail retains chronological verification;
    // CommitTargetCaches restores its ring after accepting the chosen rows.
    bool ringVerify = verifying && tpSwaRingCache &&
        (batch ? std::all_of(batch->lengths.begin(), batch->lengths.end(),
            [](int count) { return count > 0 && count <= 8; }) : total > 0 && total <= 8);
    for (int seq = 0; ringVerify && seq < sequences; ++seq)
        for (int layer = 0; layer < block_cnt; ++layer) if (slidingLayers[layer])
            for (const Data *root : {&key(seq, layer), &value(seq, layer)})
                for (const auto &entry : root->multiDeviceDatas)
                    ringVerify = ringVerify && entry.second->expansionDims.size() == 3 &&
                        entry.second->expansionDims[1] >= window + (batch ? batch->lengths[seq] : total);
    if (verifying && tpSwaRingCache && !ringVerify)
        state.runtime.Run([&](int rank, int device) {
            for (int seq : state.groups[topology.AttentionOwner(rank)].sequenceIndices) {
                const int past = key(seq, 0).dims.empty() ? 0 : key(seq, 0).dims[1];
                for (int layer = 0; layer < block_cnt; ++layer) if (slidingLayers[layer])
                    FastllmCudaNaiveSwaCacheLayout(*key(seq, layer).multiDeviceDatas.at(device),
                        *value(seq, layer).multiDeviceDatas.at(device), past, false);
            }
        });
    for (auto &feature : captures) feature.ringVerify = ringVerify;
    if (graphDecode && verifying)
        for (auto &rank : tpDecodeState->ranks) rank->features.ringVerify = ringVerify;
    auto forwardRank = [&](int rank, int device) {
        const int group = topology.AttentionOwner(rank);
        auto &local = state.ranks[rank];
        Data logits;
        if (graphDecode) {
            logits = ForwardTensorParallelDecode(rank, inputIds, positions, kv, config,
                nullptr, &captures[rank], nullptr, batch, true);
            if (tpDecodeState->mode == TPDecodeState::Prepare ||
                tpDecodeState->mode == TPDecodeState::Capture) return;
        } else {
            local.positions.CopyFrom(Data(FLOAT32, {1, (int)state.groups[group].positions.size()}, state.groups[group].positions));
            Data ids(inputIds), pos(positions);
            logits = RunTarget(ids, pos, kv, config, &captures[rank], rank,
                embedding.dims.empty() ? nullptr : &embedding, &local.workspace, batch);
        }
        if (!isIntermediateChunkedPrefill) {
            const auto range = tpVocabRanges[rank];
            const size_t bytes = size_t(range.second - range.first) * sizeof(float);
            AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(
                (float *)joined.cudaData + range.first, size_t(vocab) * sizeof(float),
                logits.cudaData, bytes, bytes, outputRows), "Naive attention DP logits gather failed");
            FastllmCudaSyncCurrentThreadStream();
        }
    };
    if (graphDecode && tpDecodeState->warmed && !tpDecodeState->captured && !tpDecodeState->disabled) {
        auto &graph = *tpDecodeState;
        graph.mode = TPDecodeState::Prepare;
        state.runtime.Run(forwardRank);
        state.runtime.Run([&](int rank, int) {
            graph.ranks[rank]->ok = FastllmCudaGraphPrepareCaptureDevice();
        });
        bool ok = std::all_of(graph.ranks.begin(), graph.ranks.end(), [](const auto &r) { return r->ok; });
        bool pool = ok && FastllmCudaGraphMemoryPoolBegin();
        if (pool) {
            graph.mode = TPDecodeState::Capture;
            // Begin and end on each fixed worker within one dispatch. Runtime
            // completion events must never join an unfinished rank capture.
            std::mutex barrierMutex;
            std::condition_variable barrier;
            int arrived = 0;
            std::vector<int> begun(tpDevices.size(), 0);
            state.runtime.Run([&](int rank, int device) {
                FastllmCudaClearThreadError();
                begun[rank] = FastllmCudaGraphBeginCapture();
                {
                    std::unique_lock<std::mutex> lock(barrierMutex);
                    if (++arrived == (int)tpDevices.size()) barrier.notify_all();
                    else barrier.wait(lock, [&] { return arrived == (int)tpDevices.size(); });
                }
                // No rank may enter NCCL unless every participant captured.
                if (std::all_of(begun.begin(), begun.end(), [](int value) { return value; }))
                    forwardRank(rank, device);
                else {
                    if (begun[rank]) FastllmCudaGraphEndCapture(&graph.ranks[rank]->graph);
                    graph.ranks[rank]->ok = false;
                }
                FastllmCudaClearThreadError(); FastllmCudaClearLastError();
            });
            ok = FastllmCudaGraphMemoryPoolEnd(graph.reservedPointers);
            ok = ok && std::all_of(graph.ranks.begin(), graph.ranks.end(),
                [](const auto &r) { return r->ok && r->graph; });
            if (ok) {
                state.runtime.Run([&](int rank, int) {
                    auto &r = *graph.ranks[rank];
                    r.ok = FastllmCudaGraphInstantiate(r.graph, &r.exec);
                });
                ok = std::all_of(graph.ranks.begin(), graph.ranks.end(),
                    [](const auto &r) { return r->ok && r->exec; });
            }
        } else ok = false;
        if (!ok) {
            graph.ClearGraphs(); graph.disabled = true;
            graphDecode = false;
            fprintf(stderr, "[Fastllm] Naive attention DP graph unavailable; using eager.\n");
        } else graph.captured = true;
    }
    if (graphDecode)
        tpDecodeState->mode = tpDecodeState->captured ? TPDecodeState::Replay : TPDecodeState::Warm;
    state.runtime.Run(forwardRank);
    if (graphDecode) {
        tpDecodeState->warmed = true;
        if (!std::all_of(tpDecodeState->ranks.begin(), tpDecodeState->ranks.end(),
            [](const auto &r) { return r->ok; })) tpDecodeState->disabled = true;
    }
    // Commit local and root metadata together, before recording prefix windows.
    // Each worker owns whole requests; no root can race another rank's update.
    ForSequenceRanges(tpWorkers, tpDevices, sequences, graphDecode, [&](int begin, int end) {
        for (int seq = begin; seq < end; ++seq) {
            const int oldLength = graphDecode ? key(seq, 0).dims[1] : 0;
            const int count = batch ? batch->lengths[seq] : total;
            for (int layer = 0; layer < block_cnt; ++layer) {
                // A saturated sliding window keeps its descriptor shape. Only
                // skip it when the owned plan survived reservation unchanged.
                if (graphDecode && bindingsUnchanged && slidingLayers[layer] && oldLength >= window - 1)
                    continue;
                const auto &cfg = slidingLayers[layer] ? sliding : full;
                const int length = slidingLayers[layer] ?
                    (verifying ? std::min(oldLength, window - 1) + count :
                        std::min(oldLength + count, window - 1)) : oldLength + count;
                for (int part = 0; part < 2; ++part) {
                    Data &root = part ? value(seq, layer) : key(seq, layer);
                    if (graphDecode) for (auto &entry : root.multiDeviceDatas) {
                        FastllmNaiveResizeKVLength(*entry.second, length, entry.second->dims[2]);
                        if (entry.second->isPagedKVCache)
                            entry.second->lastPageLen = length ? (length - 1) % entry.second->pageLen + 1 : 0;
                    }
                    const auto &local = *root.multiDeviceDatas.at(root.dataDeviceIds.front());
                    const int width = cfg.kvHeads * (part ? cfg.valueDim : cfg.headDim) +
                        (!part && !slidingLayers[layer] ? PackedIndexerWidth() : 0);
                    FastllmNaiveResizeKVLength(root, local.dims[1], width);
                    if (root.expansionDims.size() == 3) {
                        root.expansionDims[0] = 1;
                        root.expansionDims[1] = local.expansionDims[1];
                        root.expansionDims[2] = width;
                    } else root.expansionDims = {1, local.expansionDims[1], width};
                }
            }
        }
    });
    if (graphDecode && recording) {
        state.runtime.Run([&](int rank, int device) {
            PrefixWindowCopies copies;
            for (int seq : state.groups[topology.AttentionOwner(rank)].sequenceIndices)
                for (const auto &snapshot : prefixes[seq])
                    for (int layer = 0; layer < block_cnt; ++layer) if (slidingLayers[layer]) {
                        Data &k = *key(seq, layer).multiDeviceDatas.at(device);
                        Data &v = *value(seq, layer).multiDeviceDatas.at(device);
                        int rows = std::min(snapshot->length, window - 1);
                        AppendPrefixWindowCopies(copies, *snapshot, rank, layer, k, v,
                            tpSwaRingCache ? snapshot->length - rows : k.dims[1] - rows, tpSwaRingCache);
                    }
            copies.Submit();
            FastllmCudaSyncCurrentThreadStream();
        });
    }
    if (recording) for (int seq = 0; seq < sequences; ++seq)
        FinishPrefixSnapshots(key(seq, 0), prefixes[seq], (batch ? batch->lengths[seq] : total) > 1);
    if (ownedDecode) {
        std::lock_guard<std::mutex> guard(tpRequestMutex);
        state.planKVType = kvCacheDataType;
        state.planValid = state.plan.Remember(*batch,
            tpCacheEpoch.load(std::memory_order_relaxed), tpRequestIds);
        state.validatedGraph = graphDecode && tpDecodeState->captured && !tpDecodeState->disabled
            ? tpDecodeState : std::shared_ptr<TPDecodeState>();
    }
    if (capture) {
        capture->ringVerify = ringVerify;
        capture->hidden = std::move(captures.front().hidden);
        capture->prefixes = std::move(prefixes);
    }
    if (isIntermediateChunkedPrefill) return Data();
    FastllmCudaSetDevice(tpDevices.front());
    if (selection && selection->cudaSampling && !selection->cudaLogits) {
        selection->sampledTokens = SampleCudaLogits(joined, batch ? batch->configs : std::vector<GenerationConfig>{config});
        return Data();
    }
    Data output(joined);
    if (!(selection && selection->cudaLogits)) output.ToDevice(CPU);
    return output;
#endif
}
