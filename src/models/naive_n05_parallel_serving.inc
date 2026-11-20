#ifdef USE_CUDA
std::shared_ptr<NaiveN05FlashModel::TPDecodeState> NaiveN05FlashModel::SelectAttentionServingGraph(
        const TargetBatch &batch, std::vector<int> &counts, std::vector<std::vector<int>> *samplingRows) const {
    const auto &serving = *tpServingState;
    const auto &topology = attentionParallel->runtime.Layout();
    counts.assign(attentionDP, 0);
    if (samplingRows) samplingRows->assign(attentionDP, {});
    int longest = 0;
    for (int row = 0; row < batch.Size(); ++row) {
        const Data &key = batch.Key(row, 0);
        if (key.dims.size() != 3 || !key.multiDeviceData || key.dataDeviceIds.empty()) return nullptr;
        auto device = std::find(tpDevices.begin(), tpDevices.end(), key.dataDeviceIds.front());
        if (device == tpDevices.end()) return nullptr;
        const int owner = topology.AttentionOwner(device - tpDevices.begin());
        ++counts[owner];
        if (samplingRows) (*samplingRows)[owner].push_back(row);
        const int length = key.dims[1] + batch.lengths[row];
        if (length <= 1 || length > max_positions) return nullptr;
        longest = std::max(longest, length);
    }
    const auto shape = DecodeGraphBatch::Select(
        *std::max_element(counts.begin(), counts.end()) * attentionDP, serving.batchBuckets);
    if (!shape) return nullptr;
    // Captured kernels read each row's live length. A long-capacity graph also
    // handles shorter rows (Indexer selects all valid keys when length <= K),
    // including the exact dense-to-sparse boundary and mixed-length batches.
    // Ordinary decode starts its sparse graph range at capacity 4096.
    const int minCapacity = !serving.verifyWidth && longest >= 2048 ? std::max(longest, 4096) : longest;
    const auto found = serving.preparing
        ? serving.graphs.find({shape.graphRows, serving.prepareCapacity})
        : serving.graphs.lower_bound({shape.graphRows, minCapacity});
    if (found == serving.graphs.end() || found->first.first != shape.graphRows || found->second->disabled ||
        (!serving.preparing && !found->second->captured)) return nullptr;
    return found->second;
}
#endif

// Ordinary DP decode uses the same prewarmed graph family as world TP. Each
// attention group has a fixed number of slots; inactive slots have live length
// zero and null KV bindings. Request order and cache ownership remain unchanged.
bool NaiveN05FlashModel::ForwardAttentionParallelServing(const Data &inputIds,
        const Data &positions, std::vector<std::pair<Data, Data>> &kv,
        const GenerationConfig &config, LogitsSelection *selection,
        const TargetBatch *batch, Data &output, TargetCapture *capture) {
#ifdef USE_CUDA
    auto &dp = *attentionParallel;
    const bool verify = capture && capture->verifying;
    if (!tpServingState || dp.servingPadded || !batch || verify != bool(tpServingState->verifyWidth) ||
        (!verify && !batch->Decode()) ||
        !GetFastllmEnv().cudaGraph || isIntermediateChunkedPrefill || sparseKv ||
        inputIds.dataDevice != CPU || positions.dataDevice != CPU ||
        inputIds.dataType != FLOAT32 || positions.dataType != FLOAT32 ||
        inputIds.dims.size() != 2 || inputIds.dims[0] != 1 || positions.Count(0) != inputIds.Count(0) ||
        inputIds.dims[1] != std::accumulate(batch->lengths.begin(), batch->lengths.end(), 0)) return false;
    auto &serving = *tpServingState;
    const int width = std::max(1, serving.verifyWidth);
    if (verify && std::any_of(batch->lengths.begin(), batch->lengths.end(),
            [width](int n) { return n < 1 || n > width; })) return false;
    if (dataType != BFLOAT16 || kvCacheDataType != serving.kvType || moeAtype != serving.moeType ||
        !GetCudaEmbedding() || GetLowMemMode() || serving.generation != FastllmGetNcclGeneration()) {
        serving.sealed = false;
        return false;
    }
    if (!serving.preparing && (!serving.sealed || (maxBatch > 0 && batch->Size() > maxBatch))) return false;
    const auto &topology = dp.runtime.Layout();
    std::vector<int> counts;
    auto graph = SelectAttentionServingGraph(*batch, counts);
    if (!graph) return false;
    if (dp.padding.empty()) {
        dp.padding.resize((size_t)attentionDP * block_cnt);
        for (int group = 0; group < attentionDP; ++group)
        for (int layer = 0; layer < block_cnt; ++layer)
        for (Data *root : {&dp.padding[group * block_cnt + layer].first,
                           &dp.padding[group * block_cnt + layer].second}) {
            root->multiDeviceData = true; root->dataDevice = CUDA;
            root->Resize({1, 2, 1});
            for (int rank : topology.groups[topology.attention[group]].ranks) {
                root->dataDeviceIds.push_back(tpDevices[rank]);
                root->multiDeviceDatas[tpDevices[rank]] = new Data(LayerKVType(layer));
            }
        }
    }
    const int slots = graph->rows / width;
    const int liveTokens = inputIds.dims[1];
    struct Scope {
        AttentionParallelState &dp;
        std::shared_ptr<TPDecodeState> &current, previous;
        void SwapScratch() {
            dp.ranks.swap(dp.servingRanks); dp.transfer.swap(dp.servingTransfer);
            dp.groups.swap(dp.servingGroups); std::swap(dp.rows, dp.servingRows);
            std::swap(dp.plan, dp.servingPlan);
        }
        Scope(AttentionParallelState &dp, std::shared_ptr<TPDecodeState> &current, int live)
            : dp(dp), current(current), previous(current) {
            dp.servingPadded = true; dp.servingLiveRows = live;
            SwapScratch();
        }
        ~Scope() {
            SwapScratch(); dp.reuseBindings = false;
            dp.servingPadded = false; dp.verifyLengths = nullptr; current = previous;
        }
    } scope(dp, tpDecodeState, batch->Size());
    tpDecodeState = graph;
    if (verify) dp.verifyLengths = &batch->lengths;
    LogitsSelection all;
    all.cudaSampling = selection && selection->cudaSampling && !selection->cudaLogits;
    all.cudaLogits = !all.cudaSampling;
    try {
        auto forward = [&]() -> Data {
            // Every request has at most width rows. Filling every slot and row
            // means the caller already has the exact captured input layout.
            if (batch->Size() == slots && liveTokens == graph->rows)
                return ForwardAttentionParallel(inputIds, positions, kv, config, capture, &all, batch);
            std::vector<std::pair<Data *, Data *>> pointers = batch->kv;
            std::vector<GenerationConfig> configs = batch->configs;
            std::vector<int> lengths(slots, width);
            std::vector<float> ids(graph->rows, 0), pos(graph->rows, -1);
            int source = 0;
            for (int seq = 0; seq < batch->Size(); ++seq) {
                const int count = batch->lengths[seq];
                std::copy_n((const float *)inputIds.cpuData + source, count, ids.begin() + seq * width);
                std::copy_n((const float *)positions.cpuData + source, count, pos.begin() + seq * width);
                source += count;
            }
            for (int group = 0; group < attentionDP; ++group)
            for (int row = counts[group]; row < slots / attentionDP; ++row) {
                for (int layer = 0; layer < block_cnt; ++layer) {
                    auto &pair = dp.padding[group * block_cnt + layer];
                    pointers.emplace_back(&pair.first, &pair.second);
                }
                configs.push_back(config);
            }
            TargetBatch padded{pointers, configs, lengths, block_cnt};
            return ForwardAttentionParallel(Data(FLOAT32, {1, graph->rows}, ids),
                Data(FLOAT32, {1, graph->rows}, pos), kv, config, capture, &all, &padded);
        };
        Data logits = forward();
        if (verify) {
            if (serving.preparing) return true;
            FastllmCudaSetDevice(tpDevices.front());
            const bool contiguous = liveTokens == batch->Size() * width;
            auto copyRagged = [&](const Data &paddedRows, Data &packed, const std::vector<int> &shape) {
                packed.dataType = paddedRows.dataType; packed.UpdateUnitSize(); packed.Resize(shape);
                packed.ToDevice(CUDA, {tpDevices.front()}, false); packed.Allocate(false);
                const size_t bytes = size_t(shape.back()) * paddedRows.unitSize;
                std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> sizes;
                int offset = 0;
                for (int seq = 0; seq < batch->Size(); ++seq) {
                    dst.push_back((char *)packed.cudaData + size_t(offset) * bytes);
                    src.push_back((char *)paddedRows.cudaData + size_t(seq) * width * bytes);
                    sizes.push_back(size_t(batch->lengths[seq]) * bytes);
                    offset += batch->lengths[seq];
                }
                AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                    dst.data(), src.data(), sizes.data(), dst.size()), "Cannot compact verify rows");
                FastllmCudaSyncCurrentThreadStream();
            };
            if (contiguous) {
                for (auto &item : capture->hidden) item.second.Resize({1, liveTokens, embed_dim});
            } else {
                auto paddedHidden = std::move(capture->hidden);
                for (auto &item : paddedHidden) copyRagged(item.second, capture->hidden[item.first], {1, liveTokens, embed_dim});
            }
            if (!logits.dims.empty()) {
                const std::vector<int> shape{liveTokens, weight["lm_head.weight"].dims[0]};
                if (contiguous) {
                    Data live;
                    live.FakeFrom(logits, 0);
                    live.Resize(shape);
                    output.CopyFrom(live);
                } else {
                    copyRagged(logits, output, shape);
                }
                if (!(selection && selection->cudaLogits)) output.ToDevice(CPU);
            }
            return true;
        }
        if (all.cudaSampling) selection->sampledTokens = std::move(all.sampledTokens);
        else {
            logits.Resize({batch->Size(), weight["lm_head.weight"].dims[0]});
            output.CopyFrom(logits);
            if (!(selection && selection->cudaLogits)) output.ToDevice(CPU);
        }
        return true;
    } catch (const AttentionParallelState::UnsupportedServingLayout &) {
        // Reservation may have grown pages, but no model kernels ran and no
        // request lengths were committed. The ordinary path can safely retry.
        return false;
    }
#else
    return false;
#endif
}

bool NaiveN05FlashModel::PrepareAttentionParallelServingDecode(const TargetBatch *batch) {
#ifdef USE_CUDA
    auto &dp = *attentionParallel;
    if (!dp.servingPadded || !batch || !tpDecodeState) return false;
    auto &graph = *tpDecodeState;
    const bool preparing = tpServingState->preparing;
    // The owned scheduler already checked live request identities, order and
    // lengths both before and after page reservation. A graph change, COW or
    // allocation invalidates this path; padding has no independent KV state.
    if (!preparing && dp.reuseBindings && dp.plan.graph.lock() == tpDecodeState)
        return graph.active = true;
    auto prepareRank = [&](int rank) {
        const auto &group = dp.groups[dp.runtime.Layout().AttentionOwner(rank)];
        auto &storage = *graph.ranks[rank]->storage;
        auto &buf = storage.buffers;
        const int count = group.sequenceIndices.size();
        const auto &previous = dp.previousBindingRows;
        bool moved = false;
        std::vector<int> changedRows;
        for (int row = 0; row < buf.tableRows; ++row) {
            const int seq = row < count ? group.sequenceIndices[row] : dp.servingLiveRows;
            if (seq < dp.servingLiveRows && !previous.empty() && previous[seq] >= 0) {
                if (previous[seq] == row) continue;
                moved = true;
            }
            changedRows.push_back(row);
        }
        // A compaction can overwrite a later row's source. Snapshot the old
        // table only when survivors actually move between physical slots.
        const auto oldTable = moved ? storage.hostCache : std::vector<void *>{};
        for (int layer = 0; layer < block_cnt; ++layer)
        for (int row : changedRows) {
            const bool live = row < count && group.sequenceIndices[row] < dp.servingLiveRows;
            const int seq = live ? group.sequenceIndices[row] : -1;
            const int stride = HasSeparateIndexer() && !slidingLayers[layer] ? 8 : 4;
            const size_t slot = (size_t)layer * buf.tableRows * (HasSeparateIndexer() ? 8 : 4) + row * stride;
            if (live && !previous.empty() && previous[seq] >= 0) {
                auto src = oldTable.begin() + (slot - row * stride) + previous[seq] * stride;
                std::copy_n(src, stride, storage.hostCache.begin() + slot);
                storage.cacheDirty = true;
                continue;
            }
            void *bindings[8] = {};
            if (live) for (int part = 0; part < 2; ++part) {
                Data &root = part ? batch->Value(seq, layer) : batch->Key(seq, layer);
                auto it = root.multiDeviceDatas.find(tpDevices[rank]);
                if (it == root.multiDeviceDatas.end() || !it->second) return false;
                const Data &local = *it->second;
                const auto &cfg = slidingLayers[layer] ? sliding : full;
                const int width = std::max(1, cfg.kvHeads / AttentionTPSize()) *
                    (part ? cfg.valueDim : cfg.headDim) + (!part ? LocalIndexerCacheWidth(layer, rank) : 0);
                const int past = batch->Key(seq, 0).dims[1];
                const int count = dp.verifyLengths ? dp.verifyLengths->at(seq) : 1;
                const int required = slidingLayers[layer] ? (graph.verifying ?
                    std::min(past, window - 1) + count : std::min(past + count, window)) : past + count;
                if (graph.verifying && slidingLayers[layer] && (local.expansionDims.size() != 3 ||
                    local.expansionDims[1] < window + batch->lengths[seq])) return false;
                const int rootWidth = cfg.kvHeads * (part ? cfg.valueDim : cfg.headDim) +
                    (!part && !slidingLayers[layer] ? PackedIndexerWidth() : 0);
                // A DP root is metadata only and may retain the caller's
                // default dtype. Captured reads/writes use the local payload.
                if (root.dims.size() != 3 || root.dims[0] != 1 || root.dims[2] != rootWidth ||
                    root.expansionDims.size() != 3 || root.expansionDims[1] < required) return false;
                // Graph descriptors retain dtype, strides and page geometry.
                // Only payload/page-table addresses are dynamically rebound.
                const bool paged = !slidingLayers[layer] && !tpPagedCaches.empty();
                if (local.isPagedKVCache != paged || (paged &&
                    (local.pagedKVCacheData != tpPagedCaches[rank][layer * 2 + part].get() ||
                     local.pageLen != local.pagedKVCacheData->pageLen))) return false;
                if (!local.cudaData || local.dataDevice != CUDA || local.dataType != LayerKVType(layer) ||
                    (uintptr_t)local.cudaData % 16 || local.strides.size() != 3 ||
                    local.strides[1] != width || local.strides[2] != 1 ||
                    !FastllmNaiveCanResizeKVLength(local, required, width) ||
                    (!preparing && local.isPagedKVCache &&
                     (int64_t)local.pageIndex.size() * local.pageLen < required)) return false;
                bindings[part] = local.isPagedKVCache ? local.pagedKVCacheData->cudaData : local.cudaData;
                bindings[part + 2] = local.isPagedKVCache ? local.cudaData : nullptr;
                if (!part && stride == 8 && HasLocalIndexer(layer, rank)) {
                    bindings[4] = IndexerStorage(layer, rank).cudaData;
                    bindings[6] = bindings[2];
                }
            }
            for (int part = 0; part < stride; ++part)
                if (storage.hostCache[slot + part] != bindings[part]) {
                    storage.hostCache[slot + part] = bindings[part]; storage.cacheDirty = true;
                }
        }
        if (preparing) {
            const int width = std::max(1, tpServingState->verifyWidth);
            if (indexerContextSharding) {
                buf.indexContextExchange.FakeFrom(storage.indexExchange, 0);
                buf.indexContextExchange.Resize({count * width * AttentionTPSize() * indexTopK * 2});
            }
            if (graph.capacity > 2048) {
                buf.decode.indexScores.FakeFrom(storage.indexScores, 0);
                buf.decode.indexScores.Resize({count * width * graph.capacity});
                for (int row = 0; row < count; ++row) {
                    auto &part = *buf.sequences[row];
                    part.decode.indexScores.FakeFrom(storage.indexScores, (size_t)row * width * graph.capacity * sizeof(int));
                    part.decode.indexScores.Resize({width * graph.capacity});
                }
            }
        }
        return true;
    };
    if (preparing) {
        for (int rank = 0; rank < (int)tpDevices.size(); ++rank)
            if (!prepareRank(rank)) return false;
    } else {
        // Each rank owns its graph storage. Rebind changed batches on the
        // existing rank workers before any rank is allowed to launch.
        std::vector<char> ready(tpDevices.size());
        dp.runtime.Run([&](int rank, int) { ready[rank] = prepareRank(rank); });
        if (std::find(ready.begin(), ready.end(), 0) != ready.end()) return false;
    }
    return graph.active = true;
#else
    return false;
#endif
}
