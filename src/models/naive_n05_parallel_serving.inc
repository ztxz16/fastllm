// Ordinary DP decode uses the same prewarmed graph family as world TP. Each
// attention group has a fixed number of slots; inactive slots have live length
// zero and null KV bindings. Request order and cache ownership remain unchanged.
bool NaiveN05FlashModel::ForwardAttentionParallelServing(const Data &inputIds,
        const Data &positions, std::vector<std::pair<Data, Data>> &kv,
        const GenerationConfig &config, LogitsSelection *selection,
        const TargetBatch *batch, Data &output) {
#ifdef USE_CUDA
    auto &dp = *attentionParallel;
    if (!tpServingState || dp.servingPadded || !batch || !batch->Decode() ||
        !GetFastllmEnv().cudaGraph || isIntermediateChunkedPrefill || sparseKv ||
        inputIds.dataDevice != CPU || positions.dataDevice != CPU ||
        inputIds.dataType != FLOAT32 || positions.dataType != FLOAT32 ||
        inputIds.dims != std::vector<int>({1, batch->Size()}) ||
        positions.Count(0) != (size_t)batch->Size()) return false;
    auto &serving = *tpServingState;
    if (dataType != BFLOAT16 || kvCacheDataType != serving.kvType || moeAtype != serving.moeType ||
        !GetCudaEmbedding() || GetLowMemMode() || serving.generation != FastllmGetNcclGeneration()) {
        serving.sealed = false;
        return false;
    }
    if (!serving.preparing && (!serving.sealed || (maxBatch > 0 && batch->Size() > maxBatch))) return false;
    const auto &topology = dp.runtime.Layout();
    std::vector<int> counts(attentionDP);
    int region = -1, longest = 0;
    for (int row = 0; row < batch->Size(); ++row) {
        const Data &key = batch->Key(row, 0);
        if (key.dims.size() != 3 || !key.multiDeviceData || key.dataDeviceIds.empty()) return false;
        auto device = std::find(tpDevices.begin(), tpDevices.end(), key.dataDeviceIds.front());
        if (device == tpDevices.end()) return false;
        ++counts[topology.AttentionOwner(device - tpDevices.begin())];
        const int length = key.dims[1] + 1;
        if (length <= 1 || length > max_positions || length == 2048) return false;
        const int current = length <= 256 ? 0 : length < 2048 ? 1 : 2;
        if (region >= 0 && current != region && !serving.preparing) return false;
        region = current; longest = std::max(longest, length);
    }
    const auto shape = DecodeGraphBatch::Select(
        *std::max_element(counts.begin(), counts.end()) * attentionDP, serving.batchBuckets);
    if (!shape) return false;
    int capacity = region == 0 ? 256 : region == 1 ? 2047 : 4096;
    while (capacity < longest && capacity < max_positions)
        capacity = (int)std::min<int64_t>((int64_t)capacity * 2, max_positions);
    if (serving.preparing) capacity = serving.prepareCapacity;
    auto found = serving.graphs.find({shape.graphRows, capacity});
    if (found == serving.graphs.end() || found->second->disabled ||
        (!serving.preparing && !found->second->captured)) return false;
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
    std::vector<std::pair<Data *, Data *>> pointers = batch->kv;
    std::vector<GenerationConfig> configs = batch->configs;
    std::vector<int> lengths(shape.graphRows, 1);
    std::vector<float> ids((float *)inputIds.cpuData, (float *)inputIds.cpuData + batch->Size());
    std::vector<float> pos((float *)positions.cpuData, (float *)positions.cpuData + batch->Size());
    for (int group = 0; group < attentionDP; ++group)
    for (int row = counts[group]; row < shape.graphRows / attentionDP; ++row) {
        for (int layer = 0; layer < block_cnt; ++layer) {
            auto &pair = dp.padding[group * block_cnt + layer];
            pointers.emplace_back(&pair.first, &pair.second);
        }
        configs.push_back(config); ids.push_back(0); pos.push_back(-1);
    }
    TargetBatch padded{pointers, configs, lengths, block_cnt};
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
            dp.servingPadded = false; current = previous;
        }
    } scope(dp, tpDecodeState, batch->Size());
    tpDecodeState = found->second;
    LogitsSelection all; all.cudaLogits = true;
    try {
        Data logits = ForwardAttentionParallel(Data(FLOAT32, {1, shape.graphRows}, ids),
            Data(FLOAT32, {1, shape.graphRows}, pos), kv, config, nullptr, &all, &padded);
        logits.Resize({batch->Size(), weight["lm_head.weight"].dims[0]});
        if (selection && selection->cudaSampling && !selection->cudaLogits) {
            selection->sampledTokens = SampleCudaLogits(logits, batch->configs);
        } else {
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
    for (int rank = 0; rank < (int)tpDevices.size(); ++rank) {
        const auto &group = dp.groups[dp.runtime.Layout().AttentionOwner(rank)];
        auto &storage = *graph.ranks[rank]->storage;
        auto &buf = storage.buffers;
        const int count = group.sequenceIndices.size();
        for (int layer = 0; layer < block_cnt; ++layer)
        for (int row = 0; row < buf.tableRows; ++row) {
            const bool live = row < count && group.sequenceIndices[row] < dp.servingLiveRows;
            const int seq = live ? group.sequenceIndices[row] : -1;
            const int stride = HasSeparateIndexer() && !slidingLayers[layer] ? 8 : 4;
            const size_t slot = (size_t)layer * buf.tableRows * (HasSeparateIndexer() ? 8 : 4) + row * stride;
            void *bindings[8] = {};
            if (live) for (int part = 0; part < 2; ++part) {
                Data &root = part ? batch->Value(seq, layer) : batch->Key(seq, layer);
                auto it = root.multiDeviceDatas.find(tpDevices[rank]);
                if (it == root.multiDeviceDatas.end() || !it->second) return false;
                const Data &local = *it->second;
                const auto &cfg = slidingLayers[layer] ? sliding : full;
                const int width = std::max(1, cfg.kvHeads / AttentionTPSize()) *
                    (part ? cfg.valueDim : cfg.headDim) + (!part ? LocalIndexerCacheWidth(layer, rank) : 0);
                const int next = batch->Key(seq, 0).dims[1] + 1;
                const int required = slidingLayers[layer] ? std::min(next, window) : next;
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
            if (indexerContextSharding) {
                buf.indexContextExchange.FakeFrom(storage.indexExchange, 0);
                buf.indexContextExchange.Resize({count * AttentionTPSize() * indexTopK * 2});
            }
            if (graph.capacity > 2048) {
                buf.decode.indexScores.FakeFrom(storage.indexScores, 0);
                buf.decode.indexScores.Resize({count * graph.capacity});
                for (int row = 0; row < count; ++row) {
                    auto &part = *buf.sequences[row];
                    part.decode.indexScores.FakeFrom(storage.indexScores, (size_t)row * graph.capacity * sizeof(int));
                    part.decode.indexScores.Resize({graph.capacity});
                }
            }
        }
    }
    return graph.active = true;
#else
    return false;
#endif
}
