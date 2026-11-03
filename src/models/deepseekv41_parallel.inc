// Included after ForwardWorkspace: the adapter reuses the model's local
// attention stages. Threading, groups and row transport live in parallel/.
#ifdef USE_CUDA
struct DeepSeekV41Model::AttentionParallel {
    parallel::CudaRuntime runtime;
    std::vector<std::unique_ptr<DeepSeekV41Model>> models;
    std::vector<std::weak_ptr<DeepSeekV41RequestState>> requests;
    std::unique_ptr<parallel::CudaRowTransfer> transfer;
    parallel::CudaRowTransfer &Transfer(const parallel::RowPartition &rows) {
        if (!transfer) transfer = std::make_unique<parallel::CudaRowTransfer>(runtime, rows);
        else transfer->Reset(rows);
        return *transfer;
    }
    explicit AttentionParallel(parallel::Topology topology) : runtime(std::move(topology)) {}
    int Place(const std::shared_ptr<DeepSeekV41RequestState> &state) {
        if (state->attentionGroup >= 0) return state->attentionGroup;
        const int groups = runtime.Layout().attentionDP;
        std::vector<uint64_t> loads(groups, 0);
        for (auto it = requests.begin(); it != requests.end();) {
            auto request = it->lock();
            if (request && request->attentionGroup >= 0) {
                loads.at(request->attentionGroup) += uint64_t(request->totalLen) + 1;
                ++it;
            } else it = requests.erase(it);
        }
        state->attentionGroup = std::min_element(loads.begin(), loads.end()) - loads.begin();
        state->attentionRanks.resize(runtime.Layout().attentionTP);
        requests.push_back(state);
        return state->attentionGroup;
    }
};

struct DeepSeekV41Model::AttentionParallelFrame {
    struct RankStep {
        std::vector<DeepSeekV41Segment> segments;
        std::vector<GenerationConfig> configs;
        Data ids, embeddings;
        LastTokensManager last;
        std::vector<std::pair<Data*, Data*>> past;
        std::unique_ptr<ForwardWorkspace> work;
    };
    AttentionParallel &parallel;
    parallel::RowPartition rows;
    parallel::CudaRowTransfer &transfer;
    std::vector<std::unique_ptr<RankStep>> ranks;
    static parallel::RowPartition Partition(AttentionParallel &parallel, ForwardWorkspace &parent) {
        std::vector<int> owners;
        for (const auto &seg : parent.boundedPrefill ? parent.tailSegments : parent.inputSegments) {
            int group = parallel.Place(seg.state);
            owners.insert(owners.end(), seg.seqlen, group);
        }
        return parallel::RowPartition(parallel.runtime.Layout().attentionDP, owners);
    }
    AttentionParallelFrame(AttentionParallel &parallel, ForwardWorkspace &parent)
        : parallel(parallel), rows(Partition(parallel, parent)), transfer(parallel.Transfer(rows)) {
        auto &runtime = parallel.runtime;
        const auto &topology = runtime.Layout();
        transfer.Reserve(size_t(parent.dim) * sizeof(uint16_t) +
            size_t(2 * parent.model.hc_mult + parent.model.hc_mult * parent.model.hc_mult) * sizeof(float));
        for (int rank = 0; rank < topology.Size(); ++rank) ranks.emplace_back(new RankStep);
        runtime.Run([&](int rank, int device) {
            auto &step = *ranks[rank];
            auto &model = *parallel.models[rank];
            model.v41AttentionLogicalRows = parent.seqlen;
            model.v41CudaGraphWarmingUp = parent.model.v41CudaGraphWarmingUp;
            int offset = 0;
            std::vector<float> tokens;
            for (int i = 0; i < parent.numSegments; ++i) {
                const auto &seg = parent.inputSegments[i];
                if (seg.state->attentionGroup != topology.AttentionOwner(rank)) continue;
                auto &local = seg.state->attentionRanks[topology.AttentionRank(rank)];
                if (!local) {
                    local = std::make_shared<DeepSeekV41RequestState>();
                    local->layers.resize(model.block_cnt);
                    local->totalLen = seg.state->totalLen;
                    if (local->totalLen) {
                        // Prefix restore is a logical CPU snapshot, independent
                        // of the group selected for this new request.
                        for (int layer = 0; layer < model.block_cnt; ++layer) {
                            const auto &source = seg.state->layers.at(layer);
                            auto &dest = local->layers[layer];
                            dest.totalLen = source.totalLen;
                            dest.compressedBlocks = source.compressedBlocks;
                            dest.rawTail = source.rawTail;
                            V41RestoreTensor(dest.windowKV, source.windowKV);
                            V41RestoreTensor(dest.compressedKV, source.compressedKV);
                            V41RestoreTensor(dest.indexK, source.indexK);
                            V41RestoreTensor(dest.rawTailKV, source.rawTailKV);
                            V41RestoreTensor(dest.rawTailScore, source.rawTailScore);
                        }
                    }
                }
                AssertInFastLLM(local->totalLen == seg.startPos, "Attention DP request position mismatch");
                DeepSeekV41Segment segment = seg;
                segment.state = local; segment.offset = offset;
                step.segments.push_back(segment);
                step.configs.push_back(parent.generationConfigsIn[i]);
                for (int j = 0; j < seg.seqlen; ++j) tokens.push_back(parent.tokenIds[seg.offset + j]);
                offset += seg.seqlen;
            }
            if (!offset) return;
            step.ids.CopyFrom(Data(FLOAT32, {1, offset}, tokens));
            // Embedding/Engram remain global stages. Prepare only local shape
            // metadata here; the real hidden and HC pre arrive before each layer.
            step.embeddings.CopyFrom(Data(BFLOAT16, {1, offset, parent.dim}, std::vector<float>(size_t(offset) * parent.dim, 0)));
            step.work.reset(new ForwardWorkspace(model, step.segments, step.ids, &step.embeddings,
                nullptr, step.configs, step.last, nullptr, step.past));
            // Placeholder embeddings must not override the parent's decision
            // for text-only bounded prefill, including mixed request lengths.
            step.work->boundedPrefill = parent.boundedPrefill;
            step.work->attentionGraphParent = parent.graphState;
            step.work->attentionParentCapturing = parent.graphCapture;
            model.PrepareForward(*step.work);
            model.PrepareForwardInputs(*step.work);
        });
    }
    void FinishGraphs(ForwardWorkspace &parent) {
        parallel.runtime.Run([&](int rank, int) {
            auto &step = *ranks[rank];
            if (!step.work) return;
            auto &f = *step.work;
            if (f.graphCapture && !parent.graphCapture)
                f.model.GiveUpForwardGraph(f, "attention parent capture failed");
            else f.model.FinishForwardGraphs(f);
        });
    }
    ~AttentionParallelFrame() {
        // ForwardWorkspace temporary allocations are released on their owner.
        // All successful runtime calls joined device completion.
        try { parallel.runtime.Run([&](int rank, int) { ranks[rank]->work.reset(); }); }
        catch (...) {}
    }
};

void DeepSeekV41Model::ConfigureAttentionParallel(int dp) {
    if (dp < 1) throw std::invalid_argument("attention DP must be positive");
    if (v41AttentionParallel || !v41States.empty() || !v41CudaGraphSlots.empty())
        throw std::invalid_argument("configure attention DP before warmup or requests");
    if (dp == 1) { v41AttentionDP = 1; return; }
    ApplyDeviceMap(deviceMap, 1, block_cnt);
    auto devices = V41TpDevices(deviceMap);
    parallel::Topology topology(devices, dp, devices.size());
    if (executionMode != ExecutionMode::Gpu || v41DsparkEnabled)
        throw std::invalid_argument("attention DP requires GPU experts and no DSpark");
    if (o_groups % topology.attentionTP || num_attention_heads % topology.attentionTP)
        throw std::invalid_argument("attention heads/output groups are not divisible by attention TP");
    for (int layer = 0; layer < block_cnt; ++layer)
        if (SelectMoeDeviceForLayer(layer) != SelectDeviceFromMap(deviceMap, layer + 1, block_cnt))
            throw std::invalid_argument("attention DP currently requires world EP on the same GPU devices");
    v41AttentionDP = dp;
    v41MoeCudaGraphCompatible = true;
}

void DeepSeekV41Model::PrepareAttentionParallel() {
    if (v41AttentionDP == 1 || v41AttentionParallel) return;
    ApplyDeviceMap(deviceMap, 1, block_cnt);
    auto devices = V41TpDevices(deviceMap);
    auto state = std::make_shared<AttentionParallel>(parallel::Topology(devices, v41AttentionDP, devices.size()));
    const auto &topology = state->runtime.Layout();
    // Reuse the generic FP8/BF16 weight splitter. Attention ranges repeat across
    // DP groups; expert weights remain in the original EP backend.
    for (int rank = 0; rank < topology.Size(); ++rank) {
        auto local = std::make_unique<DeepSeekV41Model>();
        local->weight.dicts = weight.dicts;
        local->InitParams();
        // basellm::InitParams imports the process default device maps. Override
        // them afterwards: a rank adapter must never dispatch world MultiCUDA.
        local->deviceMap = {{"cuda:" + std::to_string(devices[rank]), 1}};
        local->moeDeviceMap = local->deviceMap;
        local->layeredMoeDeviceMap.clear();
        local->executionMode = ExecutionMode::Gpu;
        local->dataType = dataType; local->kvCacheDataType = kvCacheDataType;
        local->v41MoeCudaGraphCompatible = true;
        local->engram_layer_ids.clear(); local->engramTables.clear();
        local->v41DsparkEnabled = false; local->decoderSwaTailLayer = decoderSwaTailLayer;
        local->num_attention_heads /= topology.attentionTP;
        local->o_groups /= topology.attentionTP;
        local->v41AttentionRankRuntime = &state->runtime;
        local->v41AttentionRank = rank;
        local->quantizedLinearNames = quantizedLinearNames;
        state->models.push_back(std::move(local));
    }
    for (auto &item : weight.weight) {
        const auto &name = item.first;
        auto &source = item.second;
        if (!V41StartsWith(name, "layers.") ||
            (name.find(".attn.") == std::string::npos && name.find(".attn_norm.") == std::string::npos &&
             name.find(".hc_attn_") == std::string::npos)) continue;
        if (source.isDiskWeight || (source.dataType != FLOAT32 && source.dataType != FLOAT16 &&
            source.dataType != BFLOAT16 && source.dataType != FP8_E4M3))
            throw std::invalid_argument("unsupported attention DP weight storage: " + name);
        int axis = -1;
        if (V41EndsWith(name, ".attn.wq_b.weight") || V41EndsWith(name, ".attn.wo_a.weight")) axis = 0;
        if (V41EndsWith(name, ".attn.wo_b.weight")) axis = 1;
        DivisionScheme ranges;
        if (axis >= 0 || V41EndsWith(name, ".attn.attn_sink")) {
            const int length = source.dims.at(std::max(0, axis));
            if (length % topology.attentionTP) throw std::invalid_argument("unaligned attention weight: " + name);
            for (int rank = 0; rank < topology.Size(); ++rank) {
                const int begin = topology.AttentionRank(rank) * (length / topology.attentionTP);
                ranges[devices[rank]] = {{begin, begin + length / topology.attentionTP}};
            }
            if (source.multiDeviceData) parallel::RestoreWeightOnHost(source);
            Data empty;
            bool ok = axis >= 0 ? SplitMultiCudaWeight(source, empty, devices, ranges, axis, true, true) :
                SplitMultiCudaWeight1D(source, devices, ranges);
            AssertInFastLLM(ok, "attention DP weight split failed");
        } else PrepareMultiCudaReplicatedData(source, devices, true);
        state->runtime.Run([&](int rank, int) {
            const Data &shard = *source.multiDeviceDatas.at(devices[rank]);
            Data &dest = state->models[rank]->weight[name];
            dest.CopyFrom(shard);
            dest.blockK = shard.blockK; dest.blockM = shard.blockM; dest.scales = shard.scales;
            dest.weightType = shard.weightType; dest.isModelWeight = true;
            dest.ClearTensorParallelLayout(); dest.tpLinearType = TP_LINEAR_NONE;
        });
        // The rank models now own the payloads. Retaining the temporary
        // splitter replicas would duplicate every attention weight on GPU.
        V41ResetMultiDevice(source);
        source.FreeSpace();
    }
    printf("[parallel] world=%d attention_dp=%d attention_tp=%d moe_ep=%d moe_tp=1; fixed GPU workers\n",
        topology.Size(), topology.attentionDP, topology.attentionTP, topology.Size());
    v41AttentionParallel = std::move(state);
}

void DeepSeekV41Model::RunAttentionParallel(ForwardWorkspace &f, int layer) {
    if (!f.attentionParallel) f.attentionParallel = std::make_shared<AttentionParallelFrame>(*v41AttentionParallel, f);
    auto &frame = *f.attentionParallel;
    auto &runtime = v41AttentionParallel->runtime;
    auto &topology = runtime.Layout();
    if (frame.rows.Rows() != f.seqlen) {
        frame.rows = AttentionParallelFrame::Partition(*v41AttentionParallel, f);
        frame.transfer.Reset(frame.rows);
    }
    auto ensureGlobal = [&](Data &data, DataType type, std::vector<int> dims) {
        data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
        data.dataDevice = CUDA; data.dataDeviceIds = topology.devices;
        PrepareMultiCudaReplicatedData(data, topology.devices, false);
    };
    ensureGlobal(f.ws->attnProj, BFLOAT16, {1, f.seqlen, f.dim});
    ensureGlobal(f.ws->attnPre, FLOAT32, {1, f.seqlen, hc_mult});
    ensureGlobal(f.ws->attnPost, FLOAT32, {1, f.seqlen, hc_mult});
    ensureGlobal(f.ws->attnComb, FLOAT32, {1, f.seqlen, hc_mult, hc_mult});
    PrepareMultiCudaReplicatedData(*f.curHidden, topology.devices, true);
    PrepareMultiCudaReplicatedData(*f.preMixPtr, topology.devices, true);
    // Allocate collectively only on a cold/growing workspace. Host callbacks
    // have already finished, so updating shape metadata cannot race a worker.
    bool allocate = false;
    for (Data *global : {&f.ws->attnProj, &f.ws->attnPre, &f.ws->attnPost, &f.ws->attnComb})
        for (int device : topology.devices) {
            auto &local = *global->multiDeviceDatas.at(device);
            local.Resize(global->dims);
            allocate |= !local.cudaData || local.expansionSize < local.Count(0);
        }
    if (allocate) runtime.Enqueue([&](int, int device) {
        for (Data *global : {&f.ws->attnProj, &f.ws->attnPre, &f.ws->attnPost, &f.ws->attnComb})
            global->multiDeviceDatas.at(device)->Allocate();
    });
    runtime.Enqueue([&](int rank, int device) {
        auto &step = *frame.ranks[rank];
        if (step.work) {
            auto &local = *step.work;
            local.model.v41AttentionLogicalRows = f.seqlen;
            local.model.TrimForwardPrefill(local, layer, true);
            auto select = [&](Data &dest, const Data &source) {
                auto dims = source.dims; dims[1] = local.seqlen;
                dest.dataType = source.dataType; dest.UpdateUnitSize(); dest.Resize(dims);
                dest.ToDevice(CUDA, {device}, false); dest.Allocate();
                frame.transfer.Select(rank, source.multiDeviceDatas.at(device)->cudaData, dest.cudaData,
                    source.GetBytes() / f.seqlen);
            };
            select(*local.curHidden, *f.curHidden);
            select(local.ws->preMix, *f.preMixPtr);
            local.preMixPtr = &local.ws->preMix;
            if (local.graphCapture && !f.graphCapture)
                local.model.GiveUpForwardGraph(local, "attention parent capture failed");
            local.model.BeginForwardLayer(local, layer);
        }
        parallel::CudaRowTransfer::Field fields[4];
        Data *globals[] = {&f.ws->attnProj, &f.ws->attnPre, &f.ws->attnPost, &f.ws->attnComb};
        Data *locals[] = {step.work ? &step.work->ws->attnProj : nullptr,
            step.work ? &step.work->ws->attnPre : nullptr,
            step.work ? &step.work->ws->attnPost : nullptr,
            step.work ? &step.work->ws->attnComb : nullptr};
        for (int i = 0; i < 4; ++i)
            fields[i] = {locals[i] ? locals[i]->cudaData : nullptr,
                globals[i]->multiDeviceDatas.at(device)->cudaData, globals[i]->GetBytes() / f.seqlen};
        frame.transfer.Collect(rank, fields, 4);
    });
}
#else
void DeepSeekV41Model::ConfigureAttentionParallel(int dp) {
    if (dp != 1) throw std::invalid_argument("attention DP requires CUDA");
}
void DeepSeekV41Model::PrepareAttentionParallel() {}
void DeepSeekV41Model::RunAttentionParallel(ForwardWorkspace &, int) {}
#endif
