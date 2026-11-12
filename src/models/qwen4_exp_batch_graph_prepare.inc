#ifdef USE_CUDA
bool Qwen4ExpModel::PrepareBatchGraph(std::shared_ptr<BatchGraph> &owned,
        const std::vector<ForwardState*> &frames, const std::vector<BatchedRequest*> &drafts,
        const std::vector<Data*> &hidden) {
    if (!GetFastllmEnv().cudaGraph || GetKVCacheInCPU() || threadTpState || threadTpRank >= 0 ||
        kvMemConfig.enabled || !Qwen4CudaOnlyDeviceMap(deviceMap) || hidden.empty()) return false;
    const char *hybridGraph = std::getenv("FASTLLM_QWEN38_HYBRID_GRAPH");
    if (executionMode == ExecutionMode::Hybrid && hybridGraph && std::strcmp(hybridGraph, "1") != 0)
        return false;
    const int device = FastllmCudaGetDevice();
    for (int layer = 0; layer < block_cnt; ++layer) {
        auto name = SelectDeviceFromMap(deviceMap, layer + 1, block_cnt);
        if (name != "cuda" && name != "cuda:" + std::to_string(device)) return false;
    }
    const bool draft = !drafts.empty();
    const size_t count = draft ? drafts.size() : frames.size();
    AssertInFastLLM(hidden.size() == count, "Qwen batch graph input count mismatch");
    for (size_t i = 0; i < count; ++i) {
        const auto &part = *hidden[i];
        if (part.dataDevice != CUDA || part.dims.size() != 3 || part.dims[0] != 1 ||
            part.dims[1] <= 0 || part.dims.back() != hidden.front()->dims.back() ||
            part.dataType != hidden.front()->dataType || !part.cudaData ||
            (!part.dataDeviceIds.empty() && part.dataDeviceIds != std::vector<int>{device})) return false;
        if (!draft && !frames[i]->work->qsaDeviceCompatibleMask) return false;
        const int previous = draft ? drafts[i]->state->mtpState->key.dims[1] : frames[i]->work->qsaPreviousLength;
        // Dense graph padding changes the reference attention reduction. Keep
        // short contexts on the established eager path until sparse QSA starts.
        if (previous < indexerBudget) return false;
    }
    if (!owned) owned = std::make_shared<BatchGraph>();
    auto &g = *owned;
    if (g.requests.size() != count) {
        g.Reset(); g.requests.clear(); g.signature.clear();
        for (size_t i = 0; i < count; ++i) g.requests.emplace_back(new BatchGraph::Request);
    }
    g.device = device; g.recorder.device = device;
    std::vector<uint64_t> signature{uint64_t(device), uint64_t(count), uint64_t(draft)};
    auto address = [&](const void *p) { signature.push_back(uint64_t(uintptr_t(p))); };
    std::vector<void*> tailDestinations;
    std::vector<const void*> tailSources;
    std::vector<size_t> tailSizes;
    for (size_t i = 0; i < count; ++i) {
        auto &a = *g.requests[i]; auto &graph = a.attention;
        RequestState &state = draft ? drafts[i]->state->mtpState->attentionState : *frames[i]->work->requestState;
        const Data &position = draft ? drafts[i]->position : frames[i]->work->positionIds;
        a.validRows = hidden[i]->dims[1];
        a.rows = draft ? std::max(a.validRows, drafts[i]->state->mtpState->proposals.empty()
            ? Qwen4MtpDraftsPerStep() + 1 : 1) : a.validRows;
        a.previous = draft ? drafts[i]->state->mtpState->key.dims[1] : frames[i]->work->qsaPreviousLength;
        a.width = indexerBudget + indexerCompressRatio - 1;
        address(&state); signature.push_back(a.rows);
        graph.verification.linearConvInputs.resize(block_cnt);
        graph.verification.linearAlphas.resize(block_cnt);
        graph.verification.linearBetas.resize(block_cnt);
        graph.verification.linearUpdates.resize(block_cnt);
        // Ordinary decode has no speculative rollback. Keep QSA graph storage
        // in both modes, but only attach checkpoints for actual verification.
        auto *capture = draft ? nullptr : frames[i]->work->verificationCapture;
        signature.push_back(capture != nullptr);
        graph.verification.runtimeCheckpoint = capture ? capture->runtimeCheckpoint : nullptr;
        for (int layer = draft ? block_cnt : 0; layer < (draft ? block_cnt + 1 : block_cnt); ++layer) {
            auto &key = draft ? drafts[i]->state->mtpState->key : frames[i]->work->pastKeyValues[layer].first;
            auto &value = draft ? drafts[i]->state->mtpState->value : frames[i]->work->pastKeyValues[layer].second;
            if (!draft && IsLinearAttentionLayer(layer)) {
                address(key.cudaData); address(value.cudaData);
                if (graph.verification.runtimeCheckpoint) address(graph.verification.runtimeCheckpoint->linearSecond[layer].cudaData);
                continue;
            }
            if (!PrepareFullAttentionGraphStorage(layer, key, value, state, a.rows, device,
                    &graph.verification, signature)) return false;
            graph.verification.qsaRawKeys[layer].Resize({a.rows, indexerHeadDim});
            const int capacity = Qwen4AxisCapacity(*state.indexerBlockKeyTensors[layer], 0);
            if (!Qwen4PrepareDecodeGraphWorkspace(graph.qsaScoreWorkspaces[layer], FLOAT32, {a.rows, capacity}, device) ||
                !Qwen4PrepareDecodeGraphWorkspace(graph.qsaSelectedWorkspaces[layer], INT32,
                    {a.rows, indexerBudget / indexerCompressRatio}, device)) return false;
            const int tail = a.previous % indexerCompressRatio;
            // Only completed blocks are copied into the host mirror after
            // replay. An incomplete tail remains in its original device
            // storage, so snapshotting it every token serves no reader.
            if (tail && a.validRows >= indexerCompressRatio - tail) {
                auto &keys = graph.qsaTailKeySnapshots[layer], &positions = graph.qsaTailPositionSnapshots[layer];
                Qwen4PrepareDecodeGraphWorkspace(keys, FLOAT32, {indexerCompressRatio, indexerHeadDim}, device);
                Qwen4PrepareDecodeGraphWorkspace(positions, FLOAT32, {indexerCompressRatio}, device);
                tailDestinations.push_back(keys.cudaData);
                tailSources.push_back(state.indexerTailKeyTensors[layer]->cudaData);
                tailSizes.push_back(size_t(tail) * indexerHeadDim * sizeof(float));
                tailDestinations.push_back(positions.cudaData);
                tailSources.push_back(state.indexerTailPositionTensors[layer]->cudaData);
                tailSizes.push_back(size_t(tail) * sizeof(float));
            }
        }
        if (!draft && pleLayer >= 0 && pleLayer < block_cnt) {
            // Rollback can select either history buffer. Normalize the active
            // address so accepted prefixes do not invalidate the whole graph.
            if (state.pleConvHistoryIndex == 1 && state.pleConvHistoryTensors[0] && state.pleConvHistoryTensors[1]) {
                auto &dst = *state.pleConvHistoryTensors[0], &src = *state.pleConvHistoryTensors[1];
                AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(dst.cudaData, src.cudaData, src.GetBytes()),
                    "Qwen batch PLE history staging failed");
                state.pleConvHistoryIndex = 0;
            }
            for (auto &history : state.pleConvHistoryTensors) address(history ? history->cudaData : nullptr);
            signature.push_back(state.pleConvHistoryIndex);
        }
        if (position.dataDevice == CPU && position.dataType == FLOAT32 &&
            position.cpuData && !position.multiDeviceData &&
            !position.dims.empty() && !position.strides.empty() && position.strides.back() == 1 &&
            position.Count(0) == position.dims.back() && position.Count(0) > 0 && position.Count(0) <= a.rows) {
            // Upload into the stable graph input directly. A transient CUDA
            // tensor followed by D2D adds no ownership or ordering protection.
            if (!Qwen4PrepareDecodeGraphWorkspace(graph.positionIds, FLOAT32, {1, a.rows}, device)) return false;
            std::vector<float> values(a.rows);
            const auto *source = reinterpret_cast<const float*>(position.cpuData);
            const int available = position.Count(0);
            for (int row = 0; row < a.rows; ++row) values[row] = source[std::min(row, available - 1)];
            FastllmCudaCopyFromHostToDevice(graph.positionIds.cudaData, values.data(), values.size() * sizeof(float));
        } else {
            Data devicePosition;
            if (draft) {
                auto values = Qwen4HostInts(position);
                values.resize(a.rows, values.back());
                devicePosition.CopyFrom(Data(FLOAT32, {1, a.rows}, std::vector<float>(values.begin(), values.end())));
            } else devicePosition.CopyFrom(position);
            devicePosition.ToDevice(CUDA, std::vector<int>{device});
            if (!Qwen4PrepareDecodeGraphTensor(graph.positionIds, devicePosition, device)) return false;
        }
        if (!Qwen4PrepareDecodeGraphWorkspace(graph.decodeMeta, INT32, {2}, device)) return false;
        if (!graph.pinnedDecodeMeta) graph.pinnedDecodeMeta = (int32_t*)FastllmCudaHostMalloc(2 * sizeof(int32_t));
        if (!graph.pinnedDecodeMeta) return false;
        graph.pinnedDecodeMeta[0] = a.previous;
        graph.pinnedDecodeMeta[1] = a.validRows;
        FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(graph.decodeMeta.cudaData, graph.pinnedDecodeMeta, 2 * sizeof(int32_t));
    }
    if (!tailDestinations.empty() &&
        !FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
            tailDestinations.data(), tailSources.data(), tailSizes.data(),
            int(tailDestinations.size()))) return false;
    if (signature != g.signature) { g.Reset(); g.signature = signature; }
    int totalRows = 0;
    for (auto &request : g.requests) {
        AssertInFastLLM(totalRows <= std::numeric_limits<int>::max() - request->rows,
            "Qwen batch graph row count exceeds tensor indexing capacity");
        totalRows += request->rows;
    }
    AssertInFastLLM(Qwen4PrepareDecodeGraphWorkspace(g.hidden, hidden.front()->dataType,
        {1, totalRows, hidden.front()->dims.back()}, device), "Qwen batch hidden staging failed");
    size_t offset = 0;
    for (size_t i = 0; i < hidden.size(); ++i) {
        auto *part = hidden[i];
        const size_t rowBytes = size_t(part->dims.back()) * part->unitSize;
        const size_t bytes = size_t(part->dims[1]) * part->dims.back() * part->unitSize;
        AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
            static_cast<uint8_t*>(g.hidden.cudaData) + offset, part->cudaData, bytes),
            "Qwen batch hidden upload failed");
        // Padding is graph storage, not additional logical tokens. Copy a
        // finite input row; only valid rows may update persistent QSA state.
        for (int row = part->dims[1]; row < g.requests[i]->rows; ++row)
            AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                static_cast<uint8_t*>(g.hidden.cudaData) + offset + row * rowBytes,
                static_cast<uint8_t*>(part->cudaData) + bytes - rowBytes, rowBytes),
                "Qwen batch padding upload failed");
        offset += g.requests[i]->rows * rowBytes;
    }
    for (size_t i = 0; !draft && i < count; ++i) {
        auto &f = *frames[i]->work;
        if (pleLayer >= 0 && pleLayer < block_cnt) {
            Data &embedding = g.requests[i]->ple;
            AssertInFastLLM(Qwen4PrepareDecodeGraphWorkspace(embedding, FLOAT32,
                {1, f.inputIds.dims[1], ngramHeads * ngramHeadDim}, device),
                "Qwen batch PLE staging allocation failed");
            PreparePLEEmbedding(*hidden[i], f.inputIds, *f.requestState, embedding,
                f.hostInputTokens, &f.plePrefetch);
        }
    }
    return true;
}

void Qwen4ExpModel::FinishBatchGraph(BatchGraph &g, const std::vector<ForwardState*> &frames,
                                    const std::vector<BatchedRequest*> &drafts) {
    const bool draft = !drafts.empty();
    QsaHostMirrorTransfer::Batch mirrorBatch;
    for (size_t i = 0; i < g.requests.size(); ++i) {
        auto &a = *g.requests[i]; auto &graph = a.attention;
        auto &state = draft ? drafts[i]->state->mtpState->attentionState : *frames[i]->work->requestState;
        const int length = a.previous + a.validRows, oldTail = a.previous % indexerCompressRatio;
        const int completed = (oldTail + a.validRows) / indexerCompressRatio * indexerCompressRatio;
        for (int layer = draft ? block_cnt : 0; layer < (draft ? block_cnt + 1 : block_cnt); ++layer) {
            if (!draft && IsLinearAttentionLayer(layer)) continue;
            auto &key = draft ? drafts[i]->state->mtpState->key : frames[i]->work->pastKeyValues[layer].first;
            auto &value = draft ? drafts[i]->state->mtpState->value : frames[i]->work->pastKeyValues[layer].second;
            if (completed) {
                int queued = a.previous - oldTail;
                auto &transfer = state.indexerHostMirrorTransfers[layer];
                if (oldTail) {
                    Data keys, positions;
                    keys.FakeFrom(graph.qsaTailKeySnapshots[layer], 0); keys.Resize({oldTail, indexerHeadDim});
                    positions.FakeFrom(graph.qsaTailPositionSnapshots[layer], 0); positions.Resize({oldTail});
                    transfer->Queue(keys, positions, queued, oldTail, indexerHeadDim, &mirrorBatch); queued += oldTail;
                }
                if (completed > oldTail) {
                    Data keys, positions;
                    keys.FakeFrom(graph.verification.qsaRawKeys[layer], 0); keys.Resize({completed - oldTail, indexerHeadDim});
                    positions.FakeFrom(graph.positionIds, 0); positions.Resize({completed - oldTail});
                    transfer->Queue(keys, positions, queued, completed - oldTail, indexerHeadDim, &mirrorBatch);
                }
            }
            Qwen4ResizeKVCache(key, length); Qwen4ResizeKVCache(value, length);
            state.geometricCacheGrowthReadyLayers.insert(layer);
            state.indexerTailKeyTensors[layer]->Resize({length % indexerCompressRatio, indexerHeadDim});
            state.indexerTailPositionTensors[layer]->Resize({length % indexerCompressRatio});
            state.indexerBlockKeyTensors[layer]->Resize({length / indexerCompressRatio, indexerHeadDim});
        }
        if (!draft && frames[i]->work->verificationCapture) {
            auto &dst = *frames[i]->work->verificationCapture;
            // A request may leave the shared graph while retaining its last
            // verification capture. Give rollback outputs request ownership.
            auto copy = [](Data &dst, const Data &src) {
                if (dst.cudaDataBorrowed) dst.FreeSpace();
                dst.CopyFrom(src);
            };
            if (pleLayer >= 0 && pleLayer < block_cnt) {
                copy(dst.pleInput, graph.verification.pleInput); copy(dst.pleNormalized, graph.verification.pleNormalized);
            }
            for (int layer = 0; layer < block_cnt; ++layer) {
                if (IsLinearAttentionLayer(layer)) {
                    copy(dst.linearConvInputs[layer], graph.verification.linearConvInputs[layer]);
                    copy(dst.linearAlphas[layer], graph.verification.linearAlphas[layer]);
                    copy(dst.linearBetas[layer], graph.verification.linearBetas[layer]);
                    if (graph.verification.linearUpdates[layer].dims.size() == 4)
                        copy(dst.linearUpdates[layer], graph.verification.linearUpdates[layer]);
                    else dst.linearUpdates[layer].Resize({0});
                } else copy(dst.qsaRawKeys[layer], graph.verification.qsaRawKeys[layer]);
            }
        }
    }
}
#endif
