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
        group = AllocatePrefillGroup();
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

void NaiveN05FlashModel::PrepareAttentionParallelSampling() {
#ifdef USE_CUDA
    if (!attentionParallel || draftEnabled || kvMemConfig.enabled ||
        !FastllmCudaPeerAccessInit(tpDevices)) return;
    auto &state = *attentionParallel;
    const auto &topology = state.runtime.Layout();
    const int rows = maxBatch > 0 ? maxBatch : 256;
    state.samplingLogits.resize(attentionDP);
    const int shards = tpDevices.size();
    constexpr int shardWords = sizeof(FastllmCudaTopKShard) / sizeof(int);
    size_t scratch = 0;
#ifndef USE_ROCM
    scratch = FastllmCudaLogitsTopKWorkspaceBytes(rows,
        (weight["lm_head.weight"].dims[0] + int64_t(shards) - 1) / shards);
    if (scratch && scratch <= INT_MAX && int64_t(rows) * shards * shardWords <= INT_MAX &&
        weight["lm_head.weight"].dims[0] / shards > 2 * FastllmCudaLogitsTopKCapacity)
        state.samplingShards.resize(shards);
#endif
    state.runtime.Run([&](int rank, int device) {
        if (!state.samplingShards.empty()) {
            auto &s = state.samplingShards[rank];
            auto reserve = [&](Data &d, DataType type, std::vector<int> dims) {
                d.dataType = type; d.UpdateUnitSize(); d.Resize(dims);
                d.ToDevice(CUDA, {device}, false); d.Allocate(false);
            };
            reserve(s.workspace, INT8, {int(scratch)});
            reserve(s.selected, INT32, {rows, shardWords});
            if (topology.AttentionRank(rank) == 0) {
                reserve(s.received, INT32, {rows, shards, shardWords});
                // Padding rows initially map to token zero and have no NaNs.
                FastllmCudaMemset0(s.received.cudaData, s.received.GetBytes());
            }
        }
        if (topology.AttentionRank(rank) != 0) return;
        auto &logits = state.samplingLogits[topology.AttentionOwner(rank)];
        logits.dataType = FLOAT32; logits.Resize({rows, weight["lm_head.weight"].dims[0]});
        logits.ToDevice(CUDA, {device}, false); logits.Allocate(true);
        // Reserve mask and sampler scratch on the same persistent worker that
        // serves this group, before sizing KV pools or accepting requests.
        auto mask = std::make_shared<std::vector<uint32_t>>((logits.dims.back() + 31) / 32, 0);
        (*mask)[0] = 1;
        GenerationConfig config; config.top_k = 1; config.tool_call_token_mask = mask;
        SampleCudaLogits(logits, std::vector<GenerationConfig>(rows, config));
    });
    state.samplingCapacity = rows;
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
    std::vector<std::pair<Data *, Data *>> singleKV;
    std::vector<GenerationConfig> singleConfigs;
    std::vector<int> singleLengths;
    TargetBatch singleBatch{singleKV, singleConfigs, singleLengths, block_cnt};
    if (tpServingState && !batch && ((!capture && inputIds.dims == std::vector<int>({1, 1})) ||
        (tpServingState->verifyWidth && capture && capture->verifying))) {
        for (auto &pair : kv) singleKV.emplace_back(&pair.first, &pair.second);
        singleConfigs.push_back(config); singleLengths.push_back(inputIds.dims.at(1)); batch = &singleBatch;
    }
    if (!capture || capture->verifying || !capture->collectHidden) {
        Data servingOutput;
        if (ForwardAttentionParallelServing(inputIds, positions, kv, config, selection, batch, servingOutput, capture))
            return servingOutput;
        AssertInFastLLM(state.servingPadded || !tpServingState || tpServingState->pipelineOutputSlot < 0,
            "Naive DP pipeline lost its admitted serving graph");
    }
    // An intervening prefill, verification or eager fallback may change cache
    // descriptors or shared graph storage. Revalidate on the next serving step.
    if (!state.servingPadded) {
        if (batch && batch->proposals) batch->proposals->materialize();
        state.servingPlan.valid = false;
    }
    const auto &topology = state.runtime.Layout();
    const int sequences = batch ? batch->Size() : 1;
    const int total = inputIds.dims.at(1);
    const bool verifying = capture && capture->verifying;
    const int outputRows = verifying ? total : sequences;
    const int vocab = weight["lm_head.weight"].dims[0];
    auto key = [&](int seq, int layer) -> Data & { return batch ? batch->Key(seq, layer) : kv[layer].first; };
    auto value = [&](int seq, int layer) -> Data & { return batch ? batch->Value(seq, layer) : kv[layer].second; };
    Data downloadedPositions;
    if (positions.dataDevice != CPU) {
        downloadedPositions.CopyFrom(positions);
        downloadedPositions.ToDevice(CPU);
    }
    const Data &hostPositions = positions.dataDevice == CPU ? positions : downloadedPositions;
    AssertInFastLLM(hostPositions.dataType == FLOAT32 && hostPositions.Count(0) == (size_t)total,
        "Naive attention DP requires one position per input row");
    const int liveRows = state.servingPadded ? state.servingLiveRows : sequences;
    const bool ownedVerify = verifying && state.servingPadded && state.verifyLengths;
    const bool ownedDecode = batch && (ownedVerify || (batch->Decode() && !verifying)) && !sparseKv &&
        !(tpServingState && tpServingState->preparing) &&
        mainLoop && mainLoop->get_id() == std::this_thread::get_id();
    auto matchesPlan = [&] {
        std::lock_guard<std::mutex> guard(tpRequestMutex);
        return state.plan.Matches(*batch, tpCacheEpoch.load(std::memory_order_relaxed), tpRequestIds, liveRows);
    };
    const bool compatiblePlan = ownedDecode && state.plan.valid && state.plan.kvType == kvCacheDataType &&
        (!state.servingPadded || state.plan.graph.lock() == tpDecodeState);
    const bool reusePlan = compatiblePlan && matchesPlan();
    const bool reserveCached = reusePlan && state.servingPadded && state.plan.CanAppend(state.verifyLengths);
    // Do not publish a partially rebuilt plan after an exception.
    state.plan.valid = false;
    std::vector<int> retained;
    state.previousBindingRows.clear();
    if (compatiblePlan && state.servingPadded && !reserveCached) {
        std::lock_guard<std::mutex> guard(tpRequestMutex);
        if (state.plan.epoch == tpCacheEpoch.load(std::memory_order_relaxed)) {
            std::vector<int> localRows(state.plan.requests.size(), -1);
            for (const auto &group : state.groups)
                for (size_t i = 0; i < group.sequenceIndices.size(); ++i) {
                    const int seq = group.sequenceIndices[i];
                    if (seq < (int)localRows.size()) localRows[seq] = i;
                }
            // Recheck under this lock before retaining cached descriptor pointers.
            if (reusePlan && !state.plan.Matches(*batch, state.plan.epoch, tpRequestIds, liveRows))
                throw AttentionParallelState::UnsupportedServingLayout{};
            retained = reusePlan ? state.plan.RetainInOrder(state.verifyLengths) :
                state.plan.Reorder(*batch, liveRows, tpRequestIds, state.verifyLengths);
            state.previousBindingRows = retained;
            for (int &row : state.previousBindingRows) if (row >= 0) row = localRows[row];
        }
    }
    state.reuseBindings = false;
    if (reusePlan) {
        const int width = ownedVerify ? tpServingState->verifyWidth : 1;
        AssertInFastLLM(total == sequences * width, "Naive attention DP decode row count changed");
        const float *pos = (const float *)hostPositions.cpuData;
        for (auto &group : state.groups)
            for (size_t row = 0; row < group.sequenceIndices.size(); ++row) {
                const int seq = group.sequenceIndices[row];
                // Sampling/grammar constraints and positions can change without
                // changing KV ownership. Never cache these request values.
                group.configs[row] = batch->configs[seq];
                std::copy_n(pos + seq * width, width, group.positions.begin() + row * width);
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
                group = AllocatePrefillGroup();
            }
            auto &local = state.groups[group];
            if (!retained.empty() && (seq >= liveRows || retained[seq] >= 0)) {
                auto begin = batch->kv.begin() + (size_t)seq * block_cnt;
                local.kv.insert(local.kv.end(), begin, begin + block_cnt);
            } else {
                std::vector<std::pair<Data *, Data *>> roots;
                for (int layer = 0; layer < block_cnt; ++layer) roots.emplace_back(&key(seq, layer), &value(seq, layer));
                InitAttentionParallelCache(roots, group);
                if (sparseKv) AttachSparseKv(roots);
                local.kv.insert(local.kv.end(), roots.begin(), roots.end());
            }
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
        if (!state.servingPadded && state.rows.localToGlobal != nextRows.localToGlobal) {
            // Row-transfer allocations and collective counts are graph operands.
            // Drop the old graph before changing either; shape misses stay eager.
            tpDecodeState.reset();
        }
        state.rows = std::move(nextRows);
        if (!state.transfer) {
            state.transfer = std::make_unique<parallel::CudaRowTransfer>(state.runtime, state.rows);
            if (state.servingPadded)
                state.transfer->ReserveFixed(tpServingState->batchBuckets.back() * std::max(1, tpServingState->verifyWidth), size_t(embed_dim) * 2);
        } else state.transfer->Reset(state.rows);
        state.transfer->Reserve(size_t(embed_dim) * sizeof(uint16_t));
        for (auto &group : state.groups)
            group.batch = std::make_unique<TargetBatch>(TargetBatch{group.kv, group.configs, group.lengths, block_cnt});
    }
    if (tpEp8MegaState) {
        const int ranks = tpDevices.size();
        int maxRows = total < ranks ? total : int((int64_t(total) + ranks - 1) / ranks);
        for (const auto &local : state.rows.localToGlobal)
            maxRows = std::max(maxRows, (int)local.size());
        FastllmCudaPrepareMimoEp8Mega(*tpEp8MegaState, maxRows);
    }
    // Stable owned requests reuse physical capacity after checking shared tails.
    // Identity/epoch checks cover rebinding and rollback; capacity bounds cover
    // page growth. Shared history tails still trigger copy-on-write; direct
    // forwards retain full reservation checks.
    std::vector<int> reservationLimits;
    if (!tpPagedCaches.empty() && !(tpServingState && tpServingState->preparing) && !reserveCached) {
        // Reserve on every owner before any attention/world collective. Pool
        // exhaustion must fail the step before peers enter NCCL.
        std::vector<std::exception_ptr> errors(tpDevices.size());
        tpWorkers.Run(tpDevices, [&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            const auto &group = state.groups[topology.AttentionOwner(rank)];
            std::vector<PageTableUpdate> pending;
            try {
                for (size_t row = 0; row < group.lengths.size(); ++row) {
                    const int seq = group.sequenceIndices[row];
                    if ((state.servingPadded && seq >= liveRows) ||
                        (!retained.empty() && retained[seq] >= 0)) continue;
                    const int count = state.verifyLengths ? state.verifyLengths->at(seq) : group.lengths[row];
                    bool invalidated = false;
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
                                    past + count, &pending, &key(seq, 0), ownedDecode ? &invalidated : nullptr);
                            }
                        }
                    }
                }
            } catch (...) { UploadPagedCacheTables(pending); throw; }
            UploadPagedCacheTables(pending);
        }, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
        if (ownedDecode && state.servingPadded) {
            reservationLimits.assign(liveRows, max_positions);
            if (retained.empty()) state.plan.pageTails.clear();
            for (int seq = 0; seq < liveRows; ++seq) {
                if (!retained.empty() && retained[seq] >= 0) {
                    reservationLimits[seq] = state.plan.requests[seq].reservedUntil;
                    continue;
                }
                for (int layer = 0; layer < block_cnt; ++layer) if (!slidingLayers[layer])
                    for (const Data *root : {&key(seq, layer), &value(seq, layer)})
                        for (const auto &entry : root->multiDeviceDatas) {
                            const auto &local = *entry.second;
                            const int64_t capacity = local.isPagedKVCache ?
                                int64_t(local.pageIndex.size()) * local.pageLen : 0;
                            reservationLimits[seq] = std::min<int64_t>(reservationLimits[seq],
                                std::min<int64_t>(capacity, local.expansionDims.at(1)));
                            if (local.isPagedKVCache && (!reusePlan || retained.empty()))
                                state.plan.pageTails[local.pagedKVCacheData].emplace_back(&local, seq);
                        }
            }
        }
    }
    if (!verifying && !tpSwaPrefillWorkspaces.empty() && (!batch || !batch->Decode())) {
        const int length = batch ? *std::max_element(batch->lengths.begin(), batch->lengths.end()) : total;
        state.runtime.Run([&](int rank, int) { PrepareSwaPrefillWorkspace(rank, length); });
    }
    Data embedding;
    if (!GetCudaEmbedding() || GetLowMemMode()) Embedding(inputIds, weight["model.embed_tokens.weight"], embedding);
    const bool distributedSampling = !isIntermediateChunkedPrefill && !capture && !sparseKv &&
        !(tpServingState && tpServingState->preparing) && state.samplingCapacity >= liveRows &&
        selection && selection->cudaSampling && !selection->cudaLogits;
    const bool preparingVerify = verifying && tpServingState && tpServingState->preparing && state.servingPadded;
    const bool distributedVerification = verifying && capture && !capture->logitsDestinations.empty();
    if (distributedVerification) {
        AssertInFastLLM(batch && capture->logitsDestinations.size() == (size_t)liveRows,
            "Invalid distributed verification destinations.");
        for (int seq = 0; seq < liveRows; ++seq) {
            const Data *out = capture->logitsDestinations[seq];
            AssertInFastLLM(out && out->dataDevice == CUDA && out->dataType == FLOAT32 && out->cudaData &&
                out->dims.size() == 2 && out->dims[0] == (state.verifyLengths ? state.verifyLengths->at(seq) : batch->lengths[seq]) &&
                out->dims[1] == vocab, "Invalid local verification logits.");
        }
    }
    std::vector<std::vector<int>> samplingRows;
    if (distributedSampling) {
        samplingRows.resize(attentionDP);
        for (int group = 0; group < attentionDP; ++group) {
            for (int seq : state.groups[group].sequenceIndices)
                if (seq < liveRows) samplingRows[group].push_back(seq);
            state.samplingLogits[group].Resize({(int)samplingRows[group].size(), vocab});
        }
    }
    Data &joined = tpSamplingLogits;
    if (!isIntermediateChunkedPrefill && !distributedSampling && !distributedVerification && !preparingVerify) {
        FastllmCudaSetDevice(tpDevices.front());
        joined.dataType = FLOAT32; joined.UpdateUnitSize(); joined.Resize({outputRows, vocab});
        joined.ToDevice(CUDA, {tpDevices.front()}, false); joined.Allocate(false);
    }
    std::vector<PrefixSnapshots> prefixes(sequences);
    bool recording = false;
    if (saveHistoryChat && !verifying && (!draftEnabled ||
        (capture && capture->collectHidden && total > 1))) {
        for (int seq = 0; seq < sequences; ++seq) {
            if (state.servingPadded && seq >= state.servingLiveRows) continue;
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
    // Other rows can grow/COW during reservation. Never reuse a descriptor
    // whose request identity was invalidated while reserving those rows.
    if (!retained.empty()) {
        std::lock_guard<std::mutex> guard(tpRequestMutex);
        for (int seq = 0; seq < liveRows; ++seq) if (retained[seq] >= 0) {
            const auto &r = state.plan.requests[seq];
            auto id = tpRequestIds.find(r.key);
            if (state.plan.epoch != tpCacheEpoch.load(std::memory_order_relaxed) ||
                id == tpRequestIds.end() || id->second != r.identity)
                throw AttentionParallelState::UnsupportedServingLayout{};
        }
    }
    bool graphDecode = PrepareTensorParallelDecode(inputIds, kv, verifying, fullLogits,
        cacheLengths, batch, capture && capture->collectHidden);
    state.reuseBindings = false;
    if (state.servingPadded && !graphDecode)
        throw AttentionParallelState::UnsupportedServingLayout{};
    if (!graphDecode && tpServingState && tpServingState->preparing) return Data();
    std::vector<std::vector<CudaDecodeSampling::Request>> samplingRequests;
    bool asyncSampling = graphDecode && state.servingPadded && ownedDecode &&
        distributedSampling && tpServingState && tpServingState->pipelineReady &&
        state.sampling.size() == (size_t)attentionDP;
    if (asyncSampling) {
        samplingRequests.resize(attentionDP);
        std::lock_guard<std::mutex> guard(tpRequestMutex);
        for (int group = 0; group < attentionDP && asyncSampling; ++group) {
            if (samplingRows[group].empty()) continue;
            const bool constrained = std::any_of(samplingRows[group].begin(), samplingRows[group].end(),
                [&](int seq) { return bool(batch->configs[seq].tool_call_grammar); });
            if (!state.sampling[group] || !state.sampling[group]->Supports(samplingRows[group].size(), constrained)) {
                asyncSampling = false; break;
            }
            for (int seq : samplingRows[group]) {
                const auto &cfg = batch->configs[seq];
                auto identity = tpSamplingRequestIds.find(&key(seq, 0));
                if (identity == tpSamplingRequestIds.end() || (cfg.tool_call_token_mask && !cfg.tool_call_grammar) ||
                    cfg.tool_call_name_constraint_enabled || cfg.tool_call_parameter_name_constraint_enabled ||
                    !cfg.tool_call_allowed_token_ids.empty()) {
                    asyncSampling = false; break;
                }
                samplingRequests[group].push_back({identity->second, (uint64_t)key(seq, 0).dims[1] + 1,
                    cfg.top_k <= 1 ? 1.0f : cfg.temperature, cfg.top_p, std::max(1, cfg.top_k)});
                if (cfg.tool_call_grammar) {
                    const auto compiled = cfg.tool_call_grammar->Compile(cfg.tool_call_grammar_state);
                    samplingRequests[group].back().constraint = compiled.first;
                    samplingRequests[group].back().constraintState = compiled.second;
                }
            }
        }
    }
    bool shardedSampling = asyncSampling && !state.samplingShards.empty();
    for (int group = 0; group < attentionDP && shardedSampling; ++group) {
        const auto &requests = samplingRequests[group];
        if (!requests.empty()) shardedSampling = state.sampling[group]->SupportsShardedTopK(requests.size()) &&
            std::all_of(requests.begin(), requests.end(), [](const CudaDecodeSampling::Request &r) {
                return !r.constraint && r.topK > 0 && r.topK <= FastllmCudaLogitsTopKCapacity &&
                    r.temperature > 0.f && std::isfinite(1.f / r.temperature);
            });
    }
    const int samplingSlot = std::max(0, tpServingState ? tpServingState->pipelineOutputSlot : -1);
    const bool pipelined = tpServingState && tpServingState->pipelineOutputSlot >= 0;
    AssertInFastLLM(!pipelined || asyncSampling,
        "Naive DP pipeline sampling admission changed");
    // Ring candidates use the same batched SWA/commit kernels as world TP.
    // A cache without the candidate tail retains chronological verification;
    // CommitTargetCaches restores its ring after accepting the chosen rows.
    bool ringVerify = verifying && tpSwaRingCache &&
        (batch ? std::all_of(batch->lengths.begin(), batch->lengths.end(),
            [](int count) { return count > 0 && count <= 8; }) : total > 0 && total <= 8);
    for (int seq = 0; !state.servingPadded && ringVerify && seq < liveRows; ++seq)
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
    std::vector<FastllmCudaLogitShard> logitShards;
    std::vector<std::vector<FastllmCudaLogitRows>> gatherRows;
    if (batch && batch->proposals && state.servingPadded &&
        (batch->proposals->rows.empty() || !graphDecode || !tpDecodeState->captured))
        batch->proposals->materialize();
    auto forwardRank = [&](int rank, int device) {
        const int group = topology.AttentionOwner(rank);
        auto &local = state.ranks[rank];
        Data logits;
        if (graphDecode) {
            std::optional<FastllmCudaAllocationRecorder> recorder;
            if (state.servingPadded && tpServingState->preparing && tpDecodeState->mode == TPDecodeState::Warm) {
                auto &requests = tpDecodeState->ranks[rank]->captureAllocations;
                requests.clear();
                recorder.emplace(&requests);
            }
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
        if (preparingVerify) {
            FastllmCudaSyncCurrentThreadStream();
            return;
        }
        if (!isIntermediateChunkedPrefill) {
            const auto range = tpVocabRanges[rank];
            const size_t bytes = size_t(range.second - range.first) * sizeof(float);
            if (distributedVerification) {
                if (!gatherRows.empty()) {
                    // Each consumer waits for all producers before reading peer logits.
                    FastllmCudaEventRecordCurrentThread(tpDecodeState->ranks[rank]->storage->gatherReady);
                    return;
                }
                std::vector<void *> dst(liveRows);
                std::vector<const void *> src(liveRows);
                std::vector<size_t> dstPitch(liveRows, size_t(vocab) * sizeof(float));
                std::vector<size_t> widths(liveRows, bytes), heights(liveRows);
                int row = 0;
                for (int seq = 0; seq < liveRows; ++seq) {
                    const auto &out = *capture->logitsDestinations[seq];
                    dst[seq] = static_cast<float *>(out.cudaData) + range.first;
                    src[seq] = static_cast<const char *>(logits.cudaData) + size_t(row) * bytes;
                    heights[seq] = out.dims[0];
                    row += batch->lengths[seq];
                }
                AssertInFastLLM(FastllmCudaBatchCopy2DAsyncCurrentThread(dst.data(), src.data(),
                    dstPitch.data(), widths.data(), widths.data(), heights.data(), liveRows),
                    "Naive verification logits redistribution failed");
            } else if (shardedSampling) {
                const int shards = tpDevices.size();
                std::vector<FastllmCudaTopKRow> inputs;
                std::vector<FastllmCudaTopKShard *> destinations;
                inputs.reserve(liveRows); destinations.reserve(liveRows);
                for (int owner = 0; owner < attentionDP; ++owner) {
                    const int leader = topology.groups[topology.attention.at(owner)].ranks.front();
                    auto *inbox = (FastllmCudaTopKShard *)state.samplingShards[leader].received.cudaData;
                    for (size_t row = 0; row < samplingRows[owner].size(); ++row) {
                        inputs.push_back({(float *)logits.cudaData + size_t(samplingRows[owner][row]) *
                            (range.second - range.first), 1.f / samplingRequests[owner][row].temperature});
                        destinations.push_back(inbox + row * shards + rank);
                    }
                }
                auto &s = state.samplingShards[rank];
                AssertInFastLLM(FastllmCudaDispatchLogitsTopK(inputs.data(), destinations.data(), inputs.size(),
                    range.second - range.first, range.first, (FastllmCudaTopKShard *)s.selected.cudaData,
                    s.workspace.cudaData, s.workspace.GetBytes()), "Naive sharded top-k dispatch failed");
            } else if (distributedSampling) {
                std::vector<void *> destinations;
                std::vector<const void *> sources;
                std::vector<size_t> sizes;
                for (int owner = 0; owner < attentionDP; ++owner)
                    for (size_t row = 0; row < samplingRows[owner].size(); ++row) {
                        destinations.push_back((float *)state.samplingLogits[owner].cudaData + row * vocab + range.first);
                        sources.push_back((char *)logits.cudaData + size_t(samplingRows[owner][row]) * bytes);
                        sizes.push_back(bytes);
                    }
                AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                    destinations.data(), sources.data(), sizes.data(), sizes.size()),
                    "Naive attention DP logits redistribution failed");
            } else {
                AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(
                    (float *)joined.cudaData + range.first, size_t(vocab) * sizeof(float),
                    logits.cudaData, bytes, bytes, outputRows), "Naive attention DP logits gather failed");
            }
            if (asyncSampling)
                FastllmCudaEventRecordCurrentThread(tpDecodeState->ranks[rank]->storage->gatherReady);
            else FastllmCudaSyncCurrentThreadStream();
        }
    };
    if (graphDecode && tpDecodeState->warmed && !tpDecodeState->captured && !tpDecodeState->disabled &&
        (!tpServingState || !tpServingState->preparing || tpServingState->capturing)) {
        auto &graph = *tpDecodeState;
        // These prewarmed variants share rank buffers and replay serially. Use
        // the same family identity for reservation, capture and pool pinning.
        const void *family = state.servingPadded ? graph.ranks.front()->storage.get() : nullptr;
        FastllmCudaGraphScratchGroup group(family);
        graph.mode = TPDecodeState::Prepare;
        state.runtime.Run(forwardRank);
        state.runtime.Run([&](int rank, int) {
            FastllmCudaGraphScratchGroup group(family);
            auto &r = *graph.ranks[rank];
            r.ok = FastllmCudaGraphPrepareCaptureDevice() &&
                (r.captureAllocations.empty() || FastllmCudaReserveGraphWorkspace(r.captureAllocations));
            r.captureAllocations.clear();
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
                FastllmCudaGraphScratchGroup group(family);
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
    if (state.servingPadded && !graphDecode)
        throw AttentionParallelState::UnsupportedServingLayout{};
    if (graphDecode)
        tpDecodeState->mode = tpDecodeState->captured ? TPDecodeState::Replay : TPDecodeState::Warm;
    if (auto *trace = inference_trace::CurrentForward()) {
        trace->mode = graphDecode && !tpDecodeState->preparedEager
            ? (tpDecodeState->captured ? "cuda_graph" : "graph_warmup") : "eager";
        trace->Mark("graph_capture_reserve_ms");
    }
    struct PendingSampling {
        const std::vector<int> &devices;
        const std::vector<std::shared_ptr<CudaDecodeSampling>> &samplers;
        bool active = false;
        ~PendingSampling() {
            // A partially submitted forward must finish before the scheduler
            // can retire its KV or reuse logits after an exception.
            if (active) {
                for (int device : devices) { FastllmCudaSetDevice(device); ForceDeviceSync(); }
                for (const auto &sampler : samplers) if (sampler) sampler->Discard();
            }
        }
    } pendingSampling{tpDevices, state.sampling};
    // Graph-bound heads stay live through sampling. Pull each BF16 shard from its
    // producer into the owner's FP32 rows, retaining the FP32 scatter for other layouts.
    if (distributedVerification && graphDecode && tpDecodeState->captured && !preparingVerify &&
        FastllmCudaPeerAccessInit(tpDevices)) {
        for (size_t rank = 0; rank < tpDevices.size(); ++rank) {
            const auto &storage = *tpDecodeState->ranks[rank]->storage;
            const auto &head = storage.buffers.logitsBf16;
            const auto range = tpVocabRanges[rank];
            const int width = range.second - range.first;
            if (!storage.gatherReady || head.dataType != BFLOAT16 || !head.cudaData ||
                head.expansionBytes < size_t(total) * width * sizeof(uint16_t)) break;
            logitShards.push_back({static_cast<const uint16_t *>(head.cudaData), range.first, width});
        }
        if (logitShards.size() == tpDevices.size()) {
            gatherRows.resize(tpDevices.size());
            int row = 0;
            for (int seq = 0; seq < liveRows; ++seq) {
                const auto &dst = *capture->logitsDestinations[seq];
                const auto owner = dst.dataDeviceIds.size() == 1
                    ? std::find(tpDevices.begin(), tpDevices.end(), dst.dataDeviceIds.front()) : tpDevices.end();
                if (owner == tpDevices.end()) { gatherRows.clear(); break; }
                gatherRows[owner - tpDevices.begin()].push_back({static_cast<float *>(dst.cudaData), row, dst.dims[0]});
                row += batch->lengths[seq];
            }
        }
    }
    if (asyncSampling) {
        pendingSampling.active = true;
        state.runtime.Enqueue(forwardRank);
        state.runtime.Enqueue([&](int rank, int) {
            if (topology.AttentionRank(rank) != 0) return;
            const int group = topology.AttentionOwner(rank);
            if (samplingRows[group].empty()) return;
            // All producers must finish their dense or candidate writes before the owner samples.
            // Waiting only for this owner's model would race peer stores.
            for (const auto &producer : tpDecodeState->ranks)
                FastllmCudaCurrentThreadStreamWaitEvent(producer->storage->gatherReady);
            AssertInFastLLM(state.sampling[group]->Submit(samplingRequests[group], samplingSlot, shardedSampling),
                "Naive attention DP sampling admission changed after submission");
        });
    } else if (!gatherRows.empty()) {
        state.runtime.Enqueue(forwardRank);
        state.runtime.Run([&](int rank, int) {
            const auto &rows = gatherRows[rank];
            if (rows.empty()) return;
            for (const auto &producer : tpDecodeState->ranks)
                FastllmCudaCurrentThreadStreamWaitEvent(producer->storage->gatherReady);
            AssertInFastLLM(FastllmCudaGatherBf16Logits(logitShards.data(), logitShards.size(),
                rows.data(), rows.size(), vocab), "Naive verification logits gather failed");
        });
    } else state.runtime.Run(forwardRank);
    if (graphDecode) {
        tpDecodeState->warmed = true;
        if (!std::all_of(tpDecodeState->ranks.begin(), tpDecodeState->ranks.end(),
            [](const auto &r) { return r->ok; })) tpDecodeState->disabled = true;
    }
    // Commit local and root metadata together, before recording prefix windows.
    // Each worker owns whole requests; no root can race another rank's update.
    // A graph verification writes candidate payload only. Its consumer publishes
    // the accepted length once, after the acceptance decision.
    const bool deferredVerify = ownedVerify && graphDecode && ringVerify && !preparingVerify;
    const bool committedMetadata = deferredVerify || (!verifying && graphDecode && bindingsUnchanged &&
        state.plan.AdvanceMetadata(window));
    if (!committedMetadata) ForSequenceRanges(tpWorkers, tpDevices, sequences, graphDecode, [&](int begin, int end) {
        for (int seq = begin; seq < end; ++seq) {
            if (state.servingPadded && seq >= state.servingLiveRows) continue;
            if (!verifying && graphDecode && !retained.empty() && retained[seq] >= 0 &&
                state.plan.requests[seq].length >= window - 1 && !state.plan.lengthFields[seq].empty()) {
                state.plan.AdvanceRow(seq);
                continue;
            }
            const int oldLength = graphDecode ? key(seq, 0).dims[1] : 0;
            const int count = state.verifyLengths ? state.verifyLengths->at(seq) : batch ? batch->lengths[seq] : total;
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
    if (recording) for (int seq = 0; seq < liveRows; ++seq)
        FinishPrefixSnapshots(key(seq, 0), prefixes[seq], (batch ? batch->lengths[seq] : total) > 1);
    if (ownedDecode) {
        if ((!bindingsUnchanged && retained.empty()) || !graphDecode) state.plan.lengthFields.clear();
        if (graphDecode && (!verifying || deferredVerify) && (state.plan.lengthFields.empty() || !retained.empty())) {
            state.plan.lengthFields.resize(liveRows);
            bool safe = true;
            for (int seq = 0; seq < liveRows; ++seq) {
                if (!state.plan.lengthFields[seq].empty()) continue;
                for (int layer = 0; layer < block_cnt; ++layer) if (!slidingLayers[layer])
                    for (Data *root : {&key(seq, layer), &value(seq, layer)}) {
                        auto remember = [&](Data &data) {
                            if (data.dims.size() != 3 ||
                                !FastllmNaiveCanResizeKVLength(data, data.dims[1], data.dims[2]) ||
                                (data.isPagedKVCache && data.pageLen <= 0)) { safe = false; return; }
                            state.plan.lengthFields[seq].push_back({&data.dims[1],
                                data.isPagedKVCache ? &data.lastPageLen : nullptr, data.pageLen});
                        };
                        remember(*root);
                        for (auto &entry : root->multiDeviceDatas) remember(*entry.second);
                    }
            }
            if (!safe) state.plan.lengthFields.clear();
        }
        std::lock_guard<std::mutex> guard(tpRequestMutex);
        state.plan.kvType = kvCacheDataType;
        state.plan.valid = state.plan.Remember(*batch,
            tpCacheEpoch.load(std::memory_order_relaxed), tpRequestIds, liveRows,
            reservationLimits.empty() ? nullptr : &reservationLimits);
        state.plan.graph = graphDecode && tpDecodeState->captured && !tpDecodeState->disabled
            ? tpDecodeState : std::shared_ptr<TPDecodeState>();
    }
    if (capture) {
        capture->ringVerify = ringVerify;
        capture->hidden = std::move(captures.front().hidden);
        capture->prefixes = std::move(prefixes);
    }
    if (isIntermediateChunkedPrefill || distributedVerification || preparingVerify) return Data();
    if (distributedSampling) {
        if (asyncSampling) {
            if (!pipelined) {
                selection->sampledTokens.resize(liveRows);
                for (int group = 0; group < attentionDP; ++group) {
                    if (samplingRows[group].empty()) continue;
                    auto tokens = state.sampling[group]->Collect(samplingSlot);
                    for (size_t row = 0; row < tokens.size(); ++row)
                        selection->sampledTokens[samplingRows[group][row]] = tokens[row];
                }
            }
            // In pipeline mode TargetDecodePipeline owns collection/draining;
            // do not allocate an unused host result on each submitted step.
            pendingSampling.active = false;
            return Data();
        }
        selection->sampledTokens.resize(liveRows);
        state.runtime.Run([&](int rank, int) {
            if (topology.AttentionRank(rank) != 0) return;
            const int group = topology.AttentionOwner(rank);
            const auto &rows = samplingRows[group];
            if (rows.empty()) return;
            std::vector<GenerationConfig> configs;
            for (int seq : rows) configs.push_back(batch ? batch->configs[seq] : config);
            auto tokens = SampleCudaLogits(state.samplingLogits[group], configs);
            for (size_t row = 0; row < rows.size(); ++row) selection->sampledTokens[rows[row]] = tokens[row];
        });
        return Data();
    }
    FastllmCudaSetDevice(tpDevices.front());
    if (selection && selection->cudaSampling && !selection->cudaLogits) {
        Data logits;
        logits.FakeFrom(joined, 0); logits.Resize({liveRows, vocab});
        selection->sampledTokens = SampleCudaLogits(logits, batch
            ? std::vector<GenerationConfig>(batch->configs.begin(), batch->configs.begin() + liveRows)
            : std::vector<GenerationConfig>{config});
        return Data();
    }
    Data output;
    if (state.servingPadded) {
        output.FakeFrom(joined, 0); output.Resize(joined.dims);
    } else output.CopyFrom(joined);
    if (!(selection && selection->cudaLogits)) output.ToDevice(CPU);
    return output;
#endif
}
