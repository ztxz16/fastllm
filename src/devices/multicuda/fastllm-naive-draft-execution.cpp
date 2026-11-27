#include "naive-draft-state.h"
#include "../../models/naive_n05_tp.h"
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
#include "devices/cuda/naive-n05-draft-fused.h"
#endif

namespace fastllm {
using namespace naive_draft_detail;

#ifdef USE_CUDA
namespace {
    template<class Weight, class LinearOp, class FinishOp, class Scratch>
    void RunDenseDraftMlp(const DraftLayerParallel &parallel, int requests, bool packed, const std::string &layer,
            Weight &weight, LinearOp &linear, FinishOp finish, Data &normalized, Data &hidden,
            Data &gamma, float eps, Scratch &l, Data &output, const DraftCollectiveLayer *collective = nullptr) {
        Data &down = weight(layer + "mlp.down_proj.weight");
        if (packed) {
            Data &gateUp = weight(layer + "mlp.gateup_proj.weight");
            if (TryCudaDraftMlp(parallel, requests, normalized, {&gateUp, down}, hidden, gamma,
                    eps, l.gate, l.output, output, collective)) return;
            linear(normalized, gateUp, l.gateUp);
            FastllmCudaNaiveDraftSwiGLU(l.gateUp, l.gate);
        } else {
            linear(normalized, weight(layer + "mlp.gate_proj.weight"), l.gate);
            linear(normalized, weight(layer + "mlp.up_proj.weight"), l.up);
            Silu(l.gate, l.gate);
            MulTo(l.gate, l.up);
        }
        linear(l.gate, down, l.output);
        finish();
    }
}
#endif

#ifdef USE_CUDA
NaiveN05FlashModel::DraftWorkspace::~DraftWorkspace() {
    int previous = FastllmCudaGetDevice();
    if (device >= 0) FastllmCudaSetDevice(device);
    DrainProjectedHead();
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    JoinLookahead();
    if (prefetchSubmitted) FastllmCudaEventSynchronize(prefetchReady);
    if (prefetchStream) FastllmCudaStreamDestroy(prefetchStream);
    if (prefetchReady) FastllmCudaEventDestroy(prefetchReady);
#endif
    for (auto &graph : graphs) graph.Clear();
    proposal.graph.Clear();
    proposals.clear(); bucketProposals.clear(); batchGraphs.clear();
    sampling.Clear();
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    FastllmDraftFusedDestroy(fused);
    if (pinnedHeadOutput) FastllmCudaHostFree(pinnedHeadOutput);
#endif
    if (verifyLogitsReady) FastllmCudaEventDestroy(verifyLogitsReady);
    if (ownerPacket) FastllmCudaHostFree(ownerPacket);
    if (gatheredHeadReady) FastllmCudaEventDestroy(gatheredHeadReady);
    if (readyStream) FastllmCudaStreamDestroy(readyStream);
    if (readySource) FastllmCudaEventDestroy(readySource);
    if (ready) FastllmCudaEventDestroy(ready);
    if (done) FastllmCudaEventDestroy(done);
    if (inputUploaded) {
        if (inputInFlight) FastllmCudaEventSynchronize(inputUploaded);
        FastllmCudaEventDestroy(inputUploaded);
    }
    if (pinnedInput) FastllmCudaHostFree(pinnedInput);
    // Data frees use their recorded device; restore the caller's selection.
    FastllmCudaSetDevice(previous);
}

#endif

// All draft graph implementations bind the same serving operations. Feature placement is
// chosen before sharding the FC weights and stays fixed across eager execution and graph replay.
struct NaiveN05FlashModel::DraftExecution {
    decltype(&NaiveN05FlashModel::RunLayerDraftBatch) body;
    decltype(&NaiveN05FlashModel::TryDraftHeadProposal) beforeBody;
    decltype(&NaiveN05FlashModel::PrepareFusedDraftProposal) afterBody;
    decltype(&NaiveN05FlashModel::TryCommitDraftContexts) commit;
    decltype(&NaiveN05FlashModel::PrepareLayerDraftServingMemory) warm;
    bool rankFeatures, deviceVerification;

    static const DraftExecution &Select(bool rankFeatures) {
        using M = NaiveN05FlashModel;
        static const DraftExecution layered{&M::RunLayerDraftBatch, &M::TryDraftHeadProposal, nullptr, &M::CommitRootDraftRows,
                                            &M::PrepareLayerDraftServingMemory, false, false};
        static const DraftExecution grouped{&M::RunDraftLowLatencyBatch, nullptr, &M::PrepareFusedDraftProposal, &M::CommitRankDraftRows,
                                            &M::PrepareLowLatencyDraftServingMemory, true, true};
        return rankFeatures ? grouped : layered;
    }
};
const NaiveN05FlashModel::DraftExecution &NaiveN05FlashModel::DraftOps() const {
    return draftExecution ? *draftExecution : DraftExecution::Select(false);
}
bool NaiveN05FlashModel::RankDraftFeatures() const { return DraftOps().rankFeatures; }

#ifdef USE_CUDA
bool NaiveN05FlashModel::PrepareDraftTP() {
    if (draftTP) return true;
    const char *option = draftParallelLayout.empty() ? std::getenv("FASTLLM_DSPARK_TP") : draftParallelLayout.c_str();
    const bool automatic = !option || !*option;
    const std::string setting = automatic ? (draftExperts ? "dp:" : "") +
        std::to_string(std::max<size_t>(1, tpDevices.size())) : option;
    auto singleDevice = [&]() {
        if (!draftSingleDeviceLogged) {
            printf("[Naive draft] backbone_tp=1 attention_tp=1 mode=%s\n",
                   automatic ? "auto-fallback" : "explicit");
            draftSingleDeviceLogged = true;
        }
        return false;
    };
    const bool oneDevice = setting == "1" || setting == "dp:1";
    if (oneDevice && (tpDevices.size() != 1 || !GetFastllmEnv().cudaGraph || draftExperts || draftKVOnly))
        return singleDevice();
    const bool mlpOnly = setting.compare(0, 4, "mlp:") == 0;
    const bool replica = setting.compare(0, 3, "dp:") == 0;
    const bool dataParallel = replica || oneDevice;
    const std::string count = replica ? setting.substr(3) : mlpOnly ? setting.substr(4) : setting;
    const int available = FastllmCudaGetDeviceCount();
    AssertInFastLLM(!count.empty() && count.size() <= 9 &&
        std::all_of(count.begin(), count.end(), [](char c) { return c >= '0' && c <= '9'; }),
        "FASTLLM_DSPARK_TP must be a rank count, mlp:<rank count> or dp:<rank count>.");
    const int ranks = std::stoi(count);
    AssertInFastLLM((ranks > 1 || oneDevice) && ranks <= available, "Invalid number of draft CUDA devices.");
    const bool supported = GetCudaEmbedding() && !GetLowMemMode() && draftHeads > 0 &&
        draftKvHeads > 0 && draftHeads % draftKvHeads == 0 && draftHeadDim > 0 &&
        draftHeadDim <= 256 && draftHeadDim % 4 == 0 && draftBlock > 1 && draftBlock < 32 &&
        draftLayers > 0 && embed_dim > 0 && draftWindow > 1 &&
        (int64_t)draftWindow + draftBlock <= INT_MAX &&
        (draftHeads + 2LL * draftKvHeads) * draftHeadDim <= INT_MAX &&
        (dataParallel || mlpOnly || (draftHeads % ranks == 0 && (draftKvHeads % ranks == 0 || ranks % draftKvHeads == 0)));
    if ((automatic || oneDevice) && !supported) return singleDevice();
    AssertInFastLLM(supported,
        "Draft TP requires dense BF16 weights, supported Q/KV heads and CUDA embedding.");
    auto state = std::make_shared<DraftTPState>();
    state->dataParallel = dataParallel;
    state->attentionDivisor = dataParallel || mlpOnly ? 1 : ranks;
    AssertInFastLLM(tpDevices.empty() || tpDevices.size() >= (size_t)ranks,
                    "Draft TP needs enough target devices.");
    if (!tpDevices.empty()) {
        state->devices.assign(tpDevices.begin(), tpDevices.begin() + ranks);
    } else {
        ApplyDraftDevice();
        state->devices.push_back(FastllmCudaGetDevice());
        for (int device = 0; device < available && (int)state->devices.size() < ranks; ++device)
            if (device != state->devices.front()) state->devices.push_back(device);
    }
    state->executors.resize(ranks);
    // Reject unsupported projections before transferring ownership of any weight.
    auto projection = [&](const std::string &name, const std::vector<int> &dims) {
        auto it = weight.weight.find(name);
        const bool valid = it != weight.weight.end() && it->second.dataType == BFLOAT16 &&
            !it->second.multiDeviceData && it->second.dims == dims;
        if (automatic || oneDevice) return valid;
        AssertInFastLLM(valid,
            "Draft TP needs a dense BF16 projection with matching shape: " + name);
        return true;
    };
    const int kvWidth = draftKvHeads * draftHeadDim;
    const std::vector<int> widths = draftKVOnly ? std::vector<int>{kvWidth, kvWidth} :
        std::vector<int>{draftHeads * draftHeadDim, kvWidth, kvWidth};
    for (int i = 0; i < draftLayers; ++i) {
        const std::string p = "dspark.layers." + std::to_string(i) + ".";
        if (!projection(p + "self_attn." + DraftKVProjectionName(),
                        {(DraftContextQueryHeads() + 2 * draftKvHeads) * draftHeadDim, embed_dim})) return singleDevice();
        if (draftKVOnly) continue;
        const std::string mlp = p + (draftExperts ? "mlp.shared_experts." : "mlp.");
        auto down = weight.weight.find(mlp + "down_proj.weight");
        const bool valid = down != weight.weight.end() && down->second.dims.size() == 2 &&
            down->second.dims[1] > 0 && down->second.dims[1] <= INT_MAX / 2 &&
            (draftExperts || dataParallel || down->second.dims[1] % ranks == 0);
        if ((automatic || oneDevice) && !valid) return singleDevice();
        AssertInFastLLM(valid, "Draft MLP width must be divisible by its TP size.");
        const int mid = down->second.dims[1];
        if (!projection(p + "self_attn.o_proj.weight", {embed_dim, draftHeads * draftHeadDim}) ||
            !projection(mlp + "gateup_proj.weight", {2 * mid, embed_dim}) ||
            !projection(mlp + "down_proj.weight", {embed_dim, mid})) return singleDevice();
    }
    const bool peerAccess = FastllmCudaPeerAccessInit(state->devices);
    if (automatic && !peerAccess) return singleDevice();
    AssertInFastLLM(peerAccess, "Draft TP requires peer access.");
    auto split = [&](const std::string &name, int axis, const std::vector<int> &widths, bool replicated = false, bool qkv = false) {
        Data &w = weight[name], bias;
        AssertInFastLLM(w.dataType == BFLOAT16 && w.dims.size() == 2 && !w.multiDeviceData,
                        "Draft TP needs unsharded BF16 projection " + name);
        replicated = replicated || dataParallel;
        DivisionScheme scheme;
        int offset = 0;
        for (size_t part = 0; part < widths.size(); ++part) {
            const int width = widths[part];
            // When TP exceeds the KV-head count, neighboring query shards
            // share one whole KV head. Never split a head's RoPE channels.
            const int copies = qkv && part + 2 >= widths.size() && !replicated ? std::max(1, ranks / draftKvHeads) : 1;
            const int divisions = ranks / copies;
            AssertInFastLLM(width > 0 && (replicated || width % divisions == 0), "Draft TP projection cannot be split.");
            for (int rank = 0; rank < ranks; ++rank) {
                const int shard = rank / copies;
                scheme[state->devices[rank]].push_back(replicated ? std::make_pair(offset, offset + width) :
                    std::make_pair(offset + shard * width / divisions, offset + (shard + 1) * width / divisions));
            }
            offset += width;
        }
        AssertInFastLLM(offset == w.dims[axis], "Draft TP projection shape mismatch.");
        w.tpLinearType = replicated ? TP_LINEAR_NONE : axis == 0 ? TP_LINEAR_ROW : TP_LINEAR_COLUMN;
        AssertInFastLLM(SplitMultiCudaWeight(w, bias, state->devices, scheme, axis, true), "Draft TP split failed.");
        AssertInFastLLM(!w.cudaData && !w.cpuData, "Draft TP retained an unsplit weight copy.");
        if (replicated) {
            w.tpLayout = TP_LAYOUT_REPLICATED;
            w.tpAxis = -1;
            w.tpRanges.clear();
        }
    };
    auto replicate = [&](const std::string &name) {
        Data &w = weight[name];
        if (w.multiDeviceData) {
            AssertInFastLLM(w.IsTensorParallelReplicated(), "Draft input must be replicated: " + name);
            for (int device : state->devices) AssertInFastLLM(w.multiDeviceDatas.count(device), "Missing replicated draft input.");
        } else PrepareMultiCudaReplicatedData(w, state->devices, true);
    };
    bool nativeMoe = false;
    if (draftExperts && dataParallel && draftExperts % ranks == 0) {
        FastllmCudaSetDevice(state->devices.front());
        const auto &first = weight["dspark.layers.0.mlp.experts.0.down_proj.weight"];
        const bool sm90Shape = embed_dim == 4096 && draftExpertWidth == 2048 &&
                               draftExperts == 128 && draftTopK == 3;
        nativeMoe = FastllmCudaMimoIsMxfp4Expert(first) && embed_dim % 128 == 0 &&
            draftExpertWidth % 128 == 0 && (FastllmGenericEpSupported(state->devices) ||
                (sm90Shape && FastllmCudaRuntimeArch() == 90 && FastllmCudaMimoEp8MegaSupported(state->devices)));
    }
    std::vector<std::vector<std::vector<Data *>>> tables(nativeMoe ? ranks : 0);
    for (auto &rank : tables) rank.resize(draftLayers);
    for (int i = 0; i < draftLayers; ++i) {
        const std::string p = "dspark.layers." + std::to_string(i) + ".";
        // MLP-only TP replicates attention using the same ownership transfer
        // as shards; neither strategy retains a second source weight payload.
        split(p + "self_attn." + DraftKVProjectionName(), 0, widths, mlpOnly, true);
        if (draftKVOnly) {
            replicate(p + "self_attn.k_norm.weight");
            continue;
        }
        split(p + "self_attn.o_proj.weight", mlpOnly ? 0 : 1,
              {mlpOnly ? embed_dim : draftHeads * draftHeadDim}, mlpOnly);
        const std::string mlp = p + (draftExperts ? "mlp.shared_experts." : "mlp.");
        const int mid = weight[mlp + "down_proj.weight"].dims.at(1);
        split(mlp + "gateup_proj.weight", 0, {mid, mid}, draftExperts != 0);
        split(mlp + "down_proj.weight", 1, {mid}, draftExperts != 0);
        if (draftExperts) {
            replicate(p + "mlp.gate.weight");
            for (auto &rank : tables) rank[i].resize(2 + 2 * draftExperts);
            for (int e = 0; e < draftExperts; ++e) {
                const auto expert = p + "mlp.experts." + std::to_string(e) + ".";
                for (int part = 0; part < 2; ++part) {
                    const auto name = expert + (part ? "down_proj.weight" : "gateup_proj.weight");
                    if (!nativeMoe) { replicate(name); continue; }
                    auto &w = weight[name];
                    const int rank = e / (draftExperts / ranks), gpu = state->devices[rank];
                    w.directMemory = true;
                    AssertInFastLLM(PlaceMultiCudaWeightOnDevice(w, state->devices, gpu),
                                    "Cannot place DSpark expert on its owner.");
                    tables[rank][i][2 + 2 * e + part] = w.multiDeviceDatas.at(gpu);
                }
            }
        }
        for (const char *name : {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"}) replicate(p + name);
    }
    if (!draftKVOnly) {
        replicate("dspark.norm.weight");
        replicate("dspark.mask_embedding");
        replicate("model.embed_tokens.weight");
    }
    // Context rows are independent. Replicate this dense projection
    // so high-concurrency commits can divide their rows across draft workers.
    auto fc = weight.weight.find("dspark.fc.weight");
    auto contextNorm = weight.weight.find("dspark.hidden_norm.weight");
    const bool contextWeights = fc != weight.weight.end() && fc->second.dataType == BFLOAT16 &&
        fc->second.dims == std::vector<int>{embed_dim, (int)draftTargetLayers.size() * embed_dim} &&
        contextNorm != weight.weight.end() && contextNorm->second.dims == std::vector<int>{embed_dim};
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    // The fused context partitions one source matrix itself. Replicating it
    // would disable that path and its dependent next-round draft submission.
    const bool rankFeatures = !draftExperts && !draftKVOnly && !dataParallel && attentionDP == 1 &&
        !mlpOnly && state->devices == tpDevices && contextWeights &&
        CudaDraftCanShardContext(state->devices, draftTargetLayers.size(), draftLayers, embed_dim,
            draftHeadDim, draftWindow, fc->second, contextNorm->second);
#else
    constexpr bool rankFeatures = false;
#endif
    draftExecution = &DraftExecution::Select(rankFeatures);
    // Other layouts keep the replicated projection for parallel context rows.
    if (contextWeights && !RankDraftFeatures()) {
        if (draftKVOnly) split("dspark.fc.weight", 0, {embed_dim}, true);
        else replicate("dspark.fc.weight");
        replicate("dspark.hidden_norm.weight");
    }
    if (nativeMoe) state->moe = FastllmCudaCreateMimoEp8Mega(state->devices, tables, draftTopK);
    if (ranks > 1) {
        state->group = FastllmCudaNaiveDraftTPCreate(state->devices);
        AssertInFastLLM(state->group, "Draft TP communicator creation failed.");
    }
    draftTP = std::move(state);
    printf("[Naive draft] backbone_tp=%d backbone_dp=%d attention_tp=%d mode=%s\n",
           dataParallel ? 1 : ranks, dataParallel ? ranks : 1,
           draftTP->attentionDivisor, automatic ? "auto" : "explicit");
    ApplyDraftDevice();
    return true;
}
#endif

#ifdef USE_CUDA
bool NaiveN05FlashModel::AppendDraftKVOnRank(Data &input, int start, DraftContext &context, int rank, DraftWorkspace *body) {
    auto &r = *draftContextStorage->tpRanks.at(rank);
    const int heads = DraftContextQueryHeads() / draftTP->attentionDivisor;
    const int kvHeads = std::max(1, draftKvHeads / draftTP->attentionDivisor);
    const int rows = input.dims[1], width = kvHeads * draftHeadDim;
    std::vector<Data> views(draftLayers), parts(draftLayers);
    std::vector<const Data *> raw, norms, weights;
    for (int i = 0; i < draftLayers; ++i) {
        const std::string p = "dspark.layers." + std::to_string(i) + ".self_attn.";
        Data &qkv = *weight[p + DraftKVProjectionName()].multiDeviceDatas.at(r.device);
        views[i].FakeFrom(qkv, (size_t)heads * draftHeadDim * embed_dim * sizeof(uint16_t));
        views[i].Resize({2 * width, embed_dim});
        weights.push_back(&views[i]);
        norms.push_back(weight[p + "k_norm.weight"].multiDeviceDatas.at(r.device));
    }
    // Batch the K/V row views without packing or copying the weights.
    bool batched = FastllmCudaNaiveDraftKVProject(input, weights, r.contextRaw, r.contextPointers);
    for (int i = 0; i < draftLayers; ++i) {
        if (batched) {
            parts[i].FakeFrom(r.contextRaw, (size_t)i * rows * 2 * width * sizeof(uint16_t));
            parts[i].Resize({1, rows, 2 * width});
            raw.push_back(&parts[i]);
        } else {
            DraftLinear(input, views[i], r.contextLayers[i].raw);
            raw.push_back(&r.contextLayers[i].raw);
        }
    }
    auto &kv = rank && !draftTP->dataParallel ? context.workspace->tpRanks[rank]->tpKV : context.kv;
    bool submitted = false;
    AssertInFastLLM(FastllmCudaNaiveDraftKVWithBody(raw, norms, start, kv, kvHeads,
        draftHeadDim, draftWindow, draftWindow + draftBlock, draftEps, draftTheta,
        body ? &body->graphs[0].kvBody : nullptr, submitted), "Draft TP KV append failed.");
    return submitted;
}
#endif

#ifdef USE_CUDA
bool NaiveN05FlashModel::AppendDraftContextTP(Data &hidden, int start, DraftContext &context,
        int nextAnchor, bool projectFeatures, bool onOwner) {
    if (!PrepareDraftTP()) return false;
    auto &tp = *draftTP;
    if (!onOwner) ApplyDraftDevice();
    PrepareDraftContextTP(context);
    if (!draftContextStorage) {
        draftContextStorage = std::make_shared<DraftWorkspace>();
        draftContextStorage->device = tp.devices.front();
        draftContextStorage->done = FastllmCudaEventCreate();
        for (int device : tp.devices) {
            auto rank = std::make_shared<DraftWorkspace>(); rank->device = device;
            rank->contextLayers.resize(draftLayers);
            FastllmCudaSetDevice(device); rank->done = FastllmCudaEventCreate();
            draftContextStorage->tpRanks.push_back(std::move(rank));
        }
        FastllmCudaSetDevice(tp.devices.front());
    }
    auto &ws = *draftContextStorage;
    const int begin = std::max(0, hidden.dims[1] - draftWindow + 1);
    // A peer must finish consuming the root input before its storage can be reused.
    Data *selected = &hidden;
    if (!projectFeatures && (begin || hidden.dataDevice != DataDevice::CUDA ||
        hidden.dataDeviceIds != std::vector<int>{tp.devices.front()} || hidden.multiDeviceData ||
        hidden.strides != std::vector<uint64_t>{(uint64_t)hidden.dims[1] * hidden.dims[2], (uint64_t)hidden.dims[2], 1})) {
        Split(hidden, 1, begin, hidden.dims[1], ws.contextSelected);
        ws.contextSelected.ToDevice(DataDevice::CUDA, std::vector<int>{tp.devices.front()});
        selected = &ws.contextSelected;
    }
    // Publish the root PTDS writes before another device reads the selected rows.
    if (!onOwner && (!tp.dataParallel || context.deviceRank != 0)) ws.RecordPeerEvent(ws.ready);
    auto append = [&](int rank) {
        if (tp.dataParallel && rank != context.deviceRank) return;
        auto &r = *ws.tpRanks[rank];
        Data *input = selected;
        if (rank && !onOwner) FastllmCudaCurrentThreadStreamWaitEvent(ws.ready);
        if (projectFeatures) {
            DraftLinear(*selected, *weight["dspark.fc.weight"].multiDeviceDatas.at(r.device), r.projected);
            KimiK3RMSNorm(r.projected, *weight["dspark.hidden_norm.weight"].multiDeviceDatas.at(r.device),
                          draftEps, r.committedHidden);
            input = &r.committedHidden;
        } else if (rank) {
            r.contextSelected.dataType = BFLOAT16;
            r.contextSelected.UpdateUnitSize();
            r.contextSelected.Resize(selected->dims);
            r.contextSelected.ToDevice(DataDevice::CUDA, std::vector<int>{r.device}, false);
            r.contextSelected.Allocate(false);
            AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(r.device, r.contextSelected.cudaData,
                tp.devices.front(), selected->cudaData, selected->GetBytes()), "Draft TP context transfer failed.");
            input = &r.contextSelected;
        }
        auto *bucket = nextAnchor >= 0 ? draftBatchStorage->tpRanks[rank].get() : nullptr;
        const int length = start + hidden.dims[1] + 1;
        std::shared_ptr<DraftWorkspace> entry;
        if (bucket) {
            auto it = bucket->batchGraphs.find(1);
            const int width = std::max(1, draftKvHeads / tp.attentionDivisor) * draftHeadDim;
            auto stable = [&](const Data &kv) {
                return kv.isKVCache && !kv.isFake && !kv.isPagedKVCache &&
                    FastllmNaiveCanResizeKVLength(kv, draftWindow + draftBlock, width);
            };
            if (it != bucket->batchGraphs.end() && it->second->graphs[0].kvBody.exec &&
                std::all_of(context.kv.begin(), context.kv.end(),
                    [&](const auto &pair) { return stable(pair.first) && stable(pair.second); }) &&
                it->second->UploadOwner(context, nextAnchor, length, bucket->batchKV)) {
                entry = it->second;
            }
        }
        if (AppendDraftKVOnRank(*input, start + begin, context, rank, entry.get())) {
            auto &queued = bucket->ownerReplay;
            try {
                bucket->RecordPeerEvent(entry->done);
            } catch (...) {
                FastllmCudaSyncCurrentThreadStream();
                throw;
            }
            queued.bucket = entry;
            queued.slots.front().length = length;
            // Finish the public KV commit and release its inputs while the
            // private draft body continues. Cancellation joins the latter.
            FastllmCudaEventSynchronize(entry->graphs[0].kvBody.kvReady);
            return;
        }
        if (!tp.dataParallel) FastllmCudaEventRecordCurrentThread(r.done);
        else if (rank) FastllmCudaSyncCurrentThreadStream();
    };
    if (onOwner) append(context.deviceRank);
    else tp.Run(append);
    if (!tp.dataParallel)
        for (size_t rank = 1; rank < tp.devices.size(); ++rank)
            FastllmCudaCurrentThreadStreamWaitEvent(ws.tpRanks[rank]->done);
    context.committed = start + hidden.dims[1];
    return true;
}
#endif

#ifdef USE_CUDA
bool NaiveN05FlashModel::CommitPackedDraftContext(TargetCapture &capture, int tokens,
        DraftContext &context, std::vector<std::pair<Data, Data>> &kv, int nextAnchor, bool onOwner) {
    if (!capture.verifying || capture.packedRows.size() != 1) return false;
    const auto &feature = capture.packedRows.front();
    if (!feature.data || tokens <= 0) return false;
    const bool local = draftTP && draftTP->dataParallel && context.deviceRank >= 0 &&
        context.deviceRank < int(draftTP->devices.size()) && draftTP->devices[context.deviceRank] == feature.device &&
        tokens < draftWindow && weight["dspark.fc.weight"].IsTensorParallelReplicated() &&
        weight["dspark.hidden_norm.weight"].IsTensorParallelReplicated();
    const int width = embed_dim * draftTargetLayers.size();
    const int start = context.committed;
    context.continuationSerial = 0;
    Data features(BFLOAT16, {1, tokens, width});
    if (!local) {
        const int device = FastllmCudaGetDevice();
        features.ToDevice(CUDA, {device}, false);
        features.Allocate(false);
        AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(device, features.cudaData, feature.device,
            feature.data, size_t(tokens) * width * sizeof(uint16_t)), "Cannot transfer accepted draft features");
        Data projected, hidden;
        ProjectDraftContext(features, projected, hidden);
        AppendDraftContext(hidden, start, context);
        FastllmCudaSyncCurrentThreadStream();
    } else {
        features.cudaDataBorrowed = true;
        features.cudaData = const_cast<uint16_t *>(feature.data);
        features.dataDevice = CUDA;
        features.dataDeviceIds = {feature.device};
        DraftWorkspace::OwnerReplay *replay = nullptr;
        std::unique_lock<std::mutex> ownerGuard;
        if (nextAnchor >= 0 && draftBatchStorage && !preparingGraphs && GetFastllmEnv().cudaGraph) {
            std::lock_guard<std::mutex> history(historyMutex);
            auto owner = draftContexts.find(&kv);
            if (owner != draftContexts.end() && owner->second.get() == &context) {
                replay = &draftBatchStorage->tpRanks.at(context.deviceRank)->ownerReplay;
                ownerGuard = std::unique_lock<std::mutex>(replay->mutex);
                replay->Join();
                replay->slots.push_back({owner->second, nextAnchor, 0});
            }
        }
        AssertInFastLLM(AppendDraftContextTP(features, start, context, replay ? nextAnchor : -1, true, onOwner),
                        "Cannot append owner-local draft features");
        const bool queued = replay && replay->bucket;
        if (ownerGuard.owns_lock()) {
            if (!queued) replay->slots.clear();
            ownerGuard.unlock();
        }
        if (!queued) FastllmCudaSyncCurrentThreadStream();
    }
    ClearDraftPrefixHidden(context.prefixHidden);
    return true;
}
#endif

Data *NaiveN05FlashModel::DraftOwnerLogits(const DraftRound &round, int steps) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
    // All shards can write disjoint columns directly into the fixed proposal arena.
    // On the first call or after growth the ordinary gather initializes that arena.
    if (steps > 1 && steps == std::min(draftTokens, draftBlock) &&
        GetFastllmEnv().cudaGraph && !round.config.tool_call_grammar &&
        !round.config.IsSimpleGreedy() && weight["lm_head.weight"].dataType == BFLOAT16) {
        const int owner = round.context.deviceRank;
        if (owner >= 0 && owner < (int)draftSamplingRanks.size() && draftSamplingRanks[owner]) {
            auto &arena = draftSamplingRanks[owner]->sampling;
            if (arena.rows >= std::max(1, maxBatch > 0 ? maxBatch : 512) && arena.steps >= draftTokens &&
                arena.verifyProposal.cudaData && arena.verifyProposal.dims.back() == weight["lm_head.weight"].dims[0])
                return &arena.verifyProposal;
        }
    }
#endif
    return nullptr;
}

bool NaiveN05FlashModel::AcceptDraftOnOwners(std::vector<std::unique_ptr<DraftRound>> &rounds, Data &logits,
        const std::vector<int> *verifyLengths, TargetCapture *capture) {
#ifdef USE_CUDA
    if (draftTP && draftTP->dataParallel && !rounds.empty() &&
        std::all_of(rounds.begin(), rounds.end(), [](const auto &r) { return r->samplingRank >= 0; })) {
        const int ranks = draftTP->devices.size(), vocab = rounds.front()->vocab;
        std::vector<std::vector<DraftRound *>> groups(ranks);
        std::vector<std::vector<int>> offsets(ranks), indices(ranks);
        int offset = 0;
        for (size_t i = 0; i < rounds.size(); ++i) {
            auto &round = rounds[i];
            indices[round->samplingRank].push_back(i);
            groups[round->samplingRank].push_back(round.get());
            offsets[round->samplingRank].push_back(offset); offset += round->proposed.size() + 1;
        }
        // Attention DP writes directly to these destinations. Other target
        // layouts retain their normal gather and redistribute only at this boundary.
        if (!logits.dims.empty()) { ApplyDraftDevice(); FastllmCudaSyncCurrentThreadStream(); }
        // Replay the owner's existing graph after its masked KV write and before
        // the scheduler publishes target cache lengths. No sampling is advanced.
        std::vector<void *> bodyDone(ranks);
        auto queueNextBody = [&](int rank) {
            const auto &group = groups[rank];
            const int count = group.size();
            if (count <= 1 || !draftBatchStorage || preparingGraphs || draftExperts || !GetFastllmEnv().cudaGraph)
                return false;
            auto &worker = *draftBatchStorage->tpRanks[rank];
            auto found = worker.batchGraphs.find(count);
            if (found == worker.batchGraphs.end() || !found->second->ownerPacket || !found->second->graphs[0].exec)
                return false;
            std::vector<DraftWorkspace::OwnerReplay::Slot> slots;
            slots.reserve(count);
            // Retirement takes these locks in the same order. Hold the replay lock
            // until its completion event protects every request in this group.
            std::unique_lock<std::mutex> history(historyMutex);
            for (auto *r : group) {
                if (!r->draftCommitted || !r->targetPayloadWritten || r->context.deviceRank != rank ||
                    r->context.committed != r->oldLength + r->sampledAccepted + 1) return false;
                auto it = draftContexts.find(r->bodyOwnerKey);
                if (it == draftContexts.end() || it->second.get() != &r->context) return false;
                int anchor = r->NextBackboneAnchor(r->sampledAccepted, r->sampledNext);
                if (anchor < 0) return false;
                slots.push_back({it->second, anchor, r->context.committed + 1});
            }
            auto &queued = worker.ownerReplay;
            std::lock_guard<std::mutex> guard(queued.mutex);
            history.unlock();
            queued.Join();
            auto &entry = *found->second;
            auto *bindings = static_cast<uint64_t *>(entry.ownerPacket);
            const size_t pointerCount = size_t(draftLayers) * count * 2;
            auto *tail = reinterpret_cast<unsigned char *>(bindings + pointerCount);
            for (int row = 0; row < count; ++row) {
                const auto &ctx = *slots[row].owner;
                for (int layer = 0; layer < draftLayers; ++layer) {
                    const size_t slot = (size_t(layer) * count + row) * 2;
                    bindings[slot] = reinterpret_cast<uintptr_t>(ctx.kv[layer].first.cudaData);
                    bindings[slot + 1] = reinterpret_cast<uintptr_t>(ctx.kv[layer].second.cudaData);
                }
                const float token = slots[row].anchor;
                const int length = slots[row].length;
                std::memcpy(tail + row * sizeof(float), &token, sizeof(token));
                std::memcpy(tail + count * sizeof(float) + row * sizeof(int), &length, sizeof(length));
            }
            queued.slots = std::move(slots);
            try {
                AssertInFastLLM(FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(worker.batchKV.cudaData,
                    bindings, (pointerCount + count) * sizeof(uint64_t)), "Cannot bind next owner draft body.");
                AssertInFastLLM(FastllmCudaGraphLaunch(entry.graphs[0].exec), "Cannot replay next owner draft body.");
                worker.RecordPeerEvent(entry.done);
                queued.bucket = found->second;
                bodyDone[rank] = entry.done;
            } catch (...) {
                FastllmCudaSyncCurrentThreadStream();
                queued.Join();
                throw;
            }
            return true;
        };
        draftTP->Run([&](int rank) {
            if (groups[rank].empty()) return;
            int rows = 0;
            std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
            for (size_t i = 0; i < groups[rank].size(); ++i) {
                auto &r = *groups[rank][i]; rows += r.proposed.size() + 1;
                if (!logits.dims.empty()) {
                    dst.push_back(r.targetLogits->cudaData);
                    src.push_back((const float *)logits.cudaData + (size_t)offsets[rank][i] * vocab);
                    bytes.push_back(r.targetLogits->GetBytes());
                }
            }
            AssertInFastLLM(dst.empty() || FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                dst.data(), src.data(), bytes.data(), dst.size()), "Cannot distribute target verification logits.");
            Data local;
            local.FakeFrom(draftSamplingRanks[rank]->samplingTarget, 0); local.Resize({rows, vocab});
            if (capture && !capture->ownerLogitsReady.empty())
                FastllmCudaCurrentThreadStreamWaitEvent(capture->ownerLogitsReady.at(rank));
            else if (capture && rounds.size() == 1)
                for (void *event : capture->outputReady) FastllmCudaCurrentThreadStreamWaitEvent(event);
            PrepareDraftAcceptanceLocal(groups[rank], local, rank);
            for (auto *r : groups[rank]) AssertInFastLLM(r->sampledAccepted >= 0, "Local rejection sampling incomplete.");
            if (queueNextBody(rank)) return;
            auto &first = *groups[rank].front();
            if (first.ownerCommitKV && !first.draftCommitted) {
                TargetCapture localFeatures;
                localFeatures.verifying = true;
                localFeatures.packedRows.push_back(first.feature);
                CommitDraftContext(localFeatures, first.sampledAccepted + 1, first.context, *first.ownerCommitKV,
                    first.NextBackboneAnchor(first.sampledAccepted, first.sampledNext), true);
                first.draftCommitted = true;
            } else {
                // Each owner already has the accepted lengths. Batch its live rows on
                // the sampler stream while the other owners finish independently.
                if (capture && rounds.size() > 1 && capture->packedRows.size() == rounds.size()) {
                    TargetCapture accepted;
                    accepted.verifying = true;
                    std::vector<DraftContext *> contexts;
                    std::vector<int> lengths, counts;
                    for (int i : indices[rank]) {
                        auto &r = *rounds[i];
                        if (r.draftCommitted) continue;
                        accepted.packedRows.push_back(capture->packedRows[i]);
                        contexts.push_back(&r.context);
                        lengths.push_back(r.proposed.size() + 1);
                        counts.push_back(r.sampledAccepted + 1);
                    }
                    if (!contexts.empty() && CommitDraftContexts(accepted, lengths, counts, contexts, -1, rank)) {
                        for (auto *r : groups[rank]) r->draftCommitted = true;
                        return;
                    }
                }
                FastllmCudaSyncCurrentThreadStream();
            }
        });
        if (CanShardDraftHead() && std::all_of(bodyDone.begin(), bodyDone.end(), [](void *event) { return event; })) {
            std::lock_guard<std::mutex> guard(historyMutex);
            // Retirement can drain a body between worker publication and this join.
            for (const auto &worker : draftBatchStorage->tpRanks)
                if (!worker->ownerReplay.bucket) return true;
            const std::vector<int> limits(rounds.size(), draftTokens);
            DraftHeadShards head{limits, tpDraftHeadInputs, tpDraftHeadLogits};
            const int total = rounds.size() * draftTokens;
            if (head.inputs.size() != ranks || head.logits.size() != ranks) return true;
            for (int rank = 0; rank < ranks; ++rank) {
                const size_t columns = tpVocabRanges[rank].second - tpVocabRanges[rank].first;
                if (!head.inputs[rank].cudaData || head.inputs[rank].expansionBytes < size_t(total) * embed_dim * 2 ||
                    !head.logits[rank].cudaData || head.logits[rank].expansionBytes < size_t(total) * columns * 2)
                    return true;
            }
            try {
                RunDraftShardedHead(head, indices, bodyDone, true);
                // The previous rejection is complete on all owners. Reuse each
                // owner's existing full-width workspace up to the RNG boundary.
                std::vector<const Data *> shards;
                for (const auto &logits : head.logits) shards.push_back(&logits);
                std::vector<const void *> buffers(ranks);
                draftTP->Run([&](int rank) {
                    auto &output = draftSamplingRanks[rank]->sampling.verifyProposal;
                    const int rows = indices[rank].size() * draftTokens;
                    if (!output.cudaData || output.expansionBytes < size_t(rows) * vocab * 2) return;
                    Data view;
                    view.FakeFrom(output, 0); view.dataType = BFLOAT16; view.UpdateUnitSize();
                    view.Resize({1, rows, vocab});
                    std::vector<int> selected;
                    selected.reserve(rows);
                    for (int row : indices[rank])
                        for (int step = 0; step < draftTokens; ++step) selected.push_back(row * draftTokens + step);
                    for (const auto &worker : draftBatchStorage->tpRanks)
                        FastllmCudaCurrentThreadStreamWaitEvent(worker->done);
                    if (!FastllmCudaNaiveGatherDraftLogits(shards, tpDevices, total, selected, view)) return;
                    auto &event = draftBatchStorage->tpRanks[rank]->gatheredHeadReady;
                    if (!event) event = FastllmCudaEventCreate();
                    FastllmCudaEventRecordCurrentThread(event);
                    buffers[rank] = view.cudaData;
                });
                draftBatchStorage->gatheredHeadBuffers = std::move(buffers);
                draftBatchStorage->projectedHeadRows = std::move(indices);
            } catch (...) {
                for (int device : draftTP->devices) { FastllmCudaSetDevice(device); ForceDeviceSync(); }
                FastllmCudaSetDevice(draftTP->devices.front());
                throw;
            }
        }
        return true;
    }
#endif
    return false;
}

void NaiveN05FlashModel::DrainPendingDraftContext() {
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (!draftBatchStorage || !draftBatchStorage->pendingContext) return;
    draftBatchStorage->DrainPendingBackbone();
    const int previous = FastllmCudaGetDevice();
    for (size_t rank = 0; rank < draftTP->devices.size(); ++rank) {
        FastllmCudaSetDevice(draftTP->devices[rank]);
        FastllmCudaEventSynchronize(draftContextStorage->tpRanks[rank]->done);
    }
    FastllmCudaSetDevice(previous);
    draftBatchStorage->pendingContext = nullptr;
    draftBatchStorage->pendingContextOwners.clear();
#endif
}

void NaiveN05FlashModel::JoinDraftHeadLookahead() {
#ifdef USE_CUDA
    std::lock_guard<std::mutex> guard(historyMutex);
    if (draftBatchStorage) draftBatchStorage->DrainProjectedHead();
#endif
}

void NaiveN05FlashModel::JoinDraftLookahead(DraftContext *context) {
#ifdef USE_CUDA
#ifndef FASTLLM_ENABLE_DRAFT_FUSED
    if (!draftTP || !draftTP->dataParallel) return;
#endif
    // Request removal already owns historyMutex. Other entry points acquire
    // it here so removal cannot race publication/retirement of lookahead.
    std::unique_lock<std::mutex> guard(historyMutex,std::defer_lock);
    if (!context) guard.lock();
    if (draftBatchStorage && draftTP && draftTP->dataParallel) {
        draftBatchStorage->DrainProjectedHead();
        for (auto &rank : draftBatchStorage->tpRanks) {
            auto &replay = rank->ownerReplay;
            std::lock_guard<std::mutex> lock(replay.mutex);
            if (!context || replay.Contains(context)) replay.Join();
        }
    }
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (draftBatchStorage && draftBatchStorage->pendingContext &&
        (!context || std::find(draftBatchStorage->pendingContextOwners.begin(),
            draftBatchStorage->pendingContextOwners.end(),context) != draftBatchStorage->pendingContextOwners.end()))
        DrainPendingDraftContext();
    if (!draftBatchStorage || !draftBatchStorage->nextInFlight) return;
    if (context && std::none_of(draftBatchStorage->nextRetained.begin(),
            draftBatchStorage->nextRetained.end(),
            [&](const auto &owner) { return owner.get() == context; })) return;
    draftBatchStorage->JoinLookahead();
#endif
#endif
}

bool NaiveN05FlashModel::CommitSpeculativeCaches(TargetCapture &capture,
        const std::vector<int> &lengths, const std::vector<int> &counts,
        const std::vector<DraftContext *> &contexts,
        const std::vector<std::vector<std::pair<Data, Data>> *> &caches, const std::vector<int> &past,
        const std::vector<int> &nextAnchors) {
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    std::lock_guard<std::mutex> lookaheadGuard(historyMutex);
    auto *storage = draftBatchStorage.get();
    // A short-tail fallback may commit another request after a preceding
    // subgroup queued lookahead. Acceptance only joins the body consumed by
    // this exact group; never discard an unrelated in-flight producer.
    if (storage && storage->nextInFlight) {
        bool consumed = storage->nextConsumed && storage->nextRetained.size() == contexts.size();
        for (size_t row = 0; consumed && row < contexts.size(); ++row)
            consumed = storage->nextRetained[row].get() == contexts[row];
        if (!consumed) storage->JoinLookahead();
    }
    if (!capture.ringVerify || !draftTP || draftTP->devices != tpDevices ||
        caches.size() != contexts.size() || past.size() != contexts.size() || counts.size() != contexts.size()) return false;
    for (size_t row = 0; row < contexts.size(); ++row)
        if (!contexts[row] || !caches[row] || contexts[row]->committed != past[row]) return false;
    auto graph = storage ? storage->nextGraph : nullptr;
    bool prepareNext = !preparingGraphs && GetFastllmEnv().cudaGraph && storage &&
        nextAnchors.size() == contexts.size() && storage->CanPrefetch(contexts.size());
    std::vector<int> nextLengths;
    std::vector<std::weak_ptr<DraftContext>> nextOwners;
    std::vector<std::shared_ptr<DraftContext>> nextRetained;
    if (prepareNext) {
        for (size_t row = 0; row < contexts.size(); ++row) {
            auto owner = draftContexts.find(caches[row]);
            const int committed = past[row] + counts[row];
            if (nextAnchors[row] < 0 || owner == draftContexts.end() || owner->second.get() != contexts[row]) {
                prepareNext = false; break;
            }
            nextLengths.push_back(committed+1); nextOwners.push_back(owner->second);
            nextRetained.push_back(owner->second);
        }
    }
    const bool prefetched = prepareNext && storage->pendingBackbone &&
        storage->pendingContext == &capture && storage->pendingContextOwners == contexts;
    if (storage && !prefetched) storage->DrainPendingBackbone();
    if (storage) {
        // Acceptance has already joined the previous head and its backbone.
        // No host event wait is needed on this normal consumption path.
        if (!prefetched) storage->nextProposed = false;
        storage->nextReady = storage->nextInFlight = storage->nextConsumed = false;
        storage->nextRetained.clear();
        for (auto &rank : storage->tpRanks) rank->prefetchSubmitted = false;
    }
    std::function<void(int)> enqueueNext;
    if (prepareNext) {
        storage->nextRetained = std::move(nextRetained);
        storage->nextInFlight = true;
        if (prefetched) {
            // The sampling callback queued exactly these owners' token rows.
            // The host has now checked stop tokens, lengths and the next budget.
            storage->pendingBackbone = false;
            for (auto &local : storage->tpRanks) local->prefetchSubmitted = true;
        } else enqueueNext = [&](int rank) {
            auto &local = *storage->tpRanks[rank];
            local.UploadInputs(nextAnchors,nextLengths,maxBatch > 0 ? maxBatch : 512);
            // inputUploaded follows this rank's context/KV commit on its
            // ordinary stream. The public commit only waits for that stream.
            FastllmCudaStreamWaitEvent(local.prefetchStream,local.inputUploaded);
            AssertInFastLLM(FastllmCudaGraphLaunchOnStream(graph->tpRanks[rank]->graphs[0].exec,
                local.prefetchStream),"Next draft backbone launch failed.");
            FastllmCudaEventRecord(local.prefetchReady,local.prefetchStream);
            local.prefetchSubmitted = true;
            local.backboneReady = local.prefetchReady;
        };
    }
    // The common context capability check runs before enqueueTarget is invoked.
    // Each rank queues both KV updates on its existing stream; the context
    // done event covers both, while the next backbone may still be running.
    const bool committed = TryCommitDraftContexts(capture, lengths, counts, contexts, -1, nullptr,
        [&](int rank) { SubmitTargetCacheUpdates(caches, past, counts, tpDevices[rank]); }, enqueueNext);
    if (committed) PublishTargetCacheLengths(caches, past, counts);
    if (committed && prepareNext) {
        storage->nextAnchors = nextAnchors;
        storage->nextLengths = std::move(nextLengths);
        storage->nextOwners = std::move(nextOwners);
        storage->nextReady = true;
    } else if (storage && prepareNext) storage->JoinLookahead();
    return committed;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::TryCommitDraftContexts(TargetCapture &capture,
        const std::vector<int> &lengths, const std::vector<int> &counts,
        const std::vector<DraftContext *> &contexts, int nextAnchor, const int *accepted,
        const std::function<void(int)> &enqueueTarget, const std::function<void(int)> &enqueueNext) {
    return (this->*DraftOps().commit)(capture, lengths, counts, contexts, nextAnchor, accepted, enqueueTarget, enqueueNext);
}

bool NaiveN05FlashModel::CommitRootDraftRows(TargetCapture &capture,
        const std::vector<int> &lengths, const std::vector<int> &counts,
        const std::vector<DraftContext *> &contexts, int nextAnchor, const int *accepted,
        const std::function<void(int)> &enqueueTarget, const std::function<void(int)> &enqueueNext) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
    if (accepted || enqueueTarget || enqueueNext || contexts.size() != 1 || !contexts.front() ||
        counts.size() != 1 || lengths.size() != 1 || counts[0] > lengths[0]) return false;
    auto &context = *contexts.front();
    const int count = counts.front();
    const auto issued = context.continuationSerial;
    context.continuationSerial = 0;
    if (!capture.verifying || !draftTP || draftTP->dataParallel || draftTP->devices != tpDevices ||
        draftExperts || draftKVOnly || !draftContextStorage || !context.workspace ||
        draftTargetLayers.size() != tpDevices.size() || context.workspace->tpRanks.size() != tpDevices.size() ||
        draftContextStorage->tpRanks.size() != tpDevices.size()) return false;
    auto &state = *draftContextStorage;
    if (!state.contextKernel.handle && !preparingGraphs) return false;
    FastllmDraftContinuation next{draftBatchStorage && maxBatch == 1 && !preparingGraphs ? &draftBatchStorage->replay : nullptr,
        issued, context.continuationSerial, draftProposalStorage ? draftProposalStorage->proposalHead.handle.get() : nullptr,
        draftProposalStorage ? draftProposalStorage->sampling.probabilities.cudaData : nullptr};
    std::vector<FastllmDraftContextRank> ranks;
    for (size_t r = 0; r < tpDevices.size(); ++r) {
        auto it = capture.hidden.find(draftTargetLayers[r]);
        ranks.push_back({it == capture.hidden.end() ? nullptr : &it->second,
                         r ? &context.workspace->tpRanks[r]->tpKV : &context.kv, state.tpRanks[r]->done});
    }
    ApplyDraftDevice();
    if (!FastllmCudaCommitDraftFeatures(tpDevices, ranks, weight, state.contextKernel,
            {draftLayers, int(draftTargetLayers.size()), embed_dim, draftHeadDim, draftBlock, draftWindow,
             draftTP->attentionDivisor}, preparingGraphs, count, context.committed, draftEps, draftTheta,
            nextAnchor, next, [&](const auto &body) { draftTP->Run(body); },
            [&] { state.RecordPeerEvent(state.ready); return state.ready; })) return false;
    context.committed += count;
    ClearDraftPrefixHidden(context.prefixHidden);
    return true;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::CommitRankDraftRows(TargetCapture &capture,
        const std::vector<int> &lengths, const std::vector<int> &counts,
        const std::vector<DraftContext *> &contexts, int, const int *deviceAccepted,
        const std::function<void(int)> &enqueueTarget, const std::function<void(int)> &enqueueNext) {
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    const int count = contexts.size();
    if (deviceAccepted && preparingGraphs) return false;
    if (attentionDP > 1 || !capture.verifying || count < 1 || count > 64 || lengths.size() != contexts.size() ||
        counts.size() != contexts.size() || !draftTP || !draftBatchStorage || !draftBatchStorage->fused ||
        !draftBatchStorage->fusedContextReady || !draftContextStorage || draftContextStorage->tpRanks.size() != 8 ||
        capture.rankHidden.size() != 8 || draftBatchStorage->fusedContexts.size() != contexts.size()) return false;
    const bool queued = draftBatchStorage->pendingContext != nullptr;
    if (queued) AssertInFastLLM(!deviceAccepted && draftBatchStorage->pendingContext == &capture &&
        draftBatchStorage->pendingContextOwners == contexts,"Pending draft context belongs to another batch.");
    std::vector<int> old(count), offsets(count);
    int captured = 0;
    for (int row = 0; row < count; ++row) {
        auto *ctx = contexts[row];
        if (!ctx || draftBatchStorage->fusedContexts[row] != ctx || counts[row] < 1 || counts[row] > 8 ||
            counts[row] > lengths[row] || ctx->committed < 0 || ctx->kv.size() != 5 ||
            ctx->kv[0].first.dims.size() != 3 || !ctx->workspace || ctx->workspace->tpRanks.size() != 8) return false;
        old[row] = ctx->kv[0].first.dims[1]; offsets[row] = captured; captured += lengths[row];
        if (old[row] < 0 || old[row] > 1023) return false;
    }
    std::vector<const Data *> features(8);
    // A queued update already validated these stable bindings. Only its
    // accepted lengths are new when the host returns to publish the commit.
    if (!queued) for (int rank = 0; rank < 8; ++rank) {
        auto &map = rank ? capture.rankHidden[rank] : capture.hidden;
        auto it = map.find(draftTargetLayers[rank]);
        if (it == map.end()) return false;
        const Data &x = it->second;
        if (x.dataType != BFLOAT16 || x.dataDevice != DataDevice::CUDA || !x.cudaData ||
            x.dataDeviceIds != std::vector<int>{draftTP->devices[rank]} ||
            x.dims.size() != 3 || x.dims[0] != 1 || x.dims[1] < captured || x.dims[2] != 4096 ||
            x.strides.size() != 3 || x.strides[1] != 4096 || x.strides[2] != 1) return false;
        features[rank] = &x;
        for (int row = 0; row < count; ++row) {
            auto &cache = rank ? contexts[row]->workspace->tpRanks[rank]->tpKV : contexts[row]->kv;
            if (cache.size() != 5) return false;
            for (auto &pair : cache) for (Data *t : {&pair.first,&pair.second})
                if (!t->cudaData || t->dataType != BFLOAT16 || t->isPagedKVCache ||
                    t->dims != std::vector<int>({1,old[row],128}) || t->expansionDims.size() != 3 ||
                    t->expansionDims[1] < 1031 || t->strides.size() != 3 || t->strides[1] != 128 ||
                    t->dataDeviceIds != std::vector<int>{draftTP->devices[rank]}) return false;
        }
    }
    ApplyDraftDevice();
    // Acceptance is ordered after target completion. Device-count submissions
    // retain borrowed features until the matching host commit or scope fence.
    const bool shared = FastllmDraftFusedContextSharedSupported(draftBatchStorage->fused,count);
    if (deviceAccepted && !shared) return false;
    std::vector<FastllmDraftContextRequest> requests;
    if (shared && !queued) requests.reserve(8*count);
    if (shared && !queued) for (int rank = 0; rank < 8; ++rank) {
        auto &scratch = *draftBatchStorage->tpRanks[rank];
        for (int row = 0; row < count; ++row)
            requests.push_back({(const uint16_t*)features[rank]->cudaData+(size_t)offsets[row]*4096,
                (const void *const *)scratch.cacheTable.cudaData,counts[row],contexts[row]->committed,
                old[row],count,row});
    }
    auto &storage = *draftBatchStorage;
    const int *sampled = nullptr;
    if (deviceAccepted && GetFastllmEnv().cudaGraph &&
        (!storage.nextInFlight || storage.nextConsumed) && storage.CanPrefetch(count) &&
        std::all_of(lengths.begin(),lengths.end(),[&](int n) { return n == draftTokens+1; }) &&
        std::all_of(contexts.begin(),contexts.end(),[&](const auto *c) {
            return c->committed < max_positions - draftTokens - 2;
        })) {
        sampled = FastllmCudaMtpSmallResultRows(deviceAccepted,count,draftTokens);
    }
    const bool propose = sampled && count==1 && draftTokens==7 && storage.lastHeadTemperature>0 &&
        std::isfinite(storage.lastHeadTemperature) &&
        std::all_of(storage.nextGraph->tpRanks.begin(),storage.nextGraph->tpRanks.end(),
            [](const auto &rank) { return rank->graphs[0].acceptedExec != nullptr; });
    if (propose) {
        auto random=contexts.front()->random;
        storage.predictedSeed=random();  // Commit the RNG only on actual consumption.
        storage.predictedTemperature=storage.lastHeadTemperature;
    }
    const bool targetFirst = enqueueTarget && !draftBatchStorage->targetCommitReady;
    if (deviceAccepted) {
        draftBatchStorage->pendingContext = &capture;
        draftBatchStorage->pendingContextOwners = contexts;
    }
    try {
        if (shared && !queued) {
            bool uploaded;
            if (sampled) {
                FastllmDraftNextInputs next{};
                next.tokens = sampled;
                next.stride = draftTokens + 1;
                for (int rank = 0; rank < 8; ++rank) {
                    next.anchors[rank] = static_cast<float *>(storage.tpRanks[rank]->id.cudaData);
                    next.lengths[rank] = static_cast<int *>(storage.tpRanks[rank]->live.cudaData);
                }
                if (propose) {
                    next.seed = storage.predictedSeed;
                    next.temperature = storage.predictedTemperature;
                }
                uploaded = FastllmDraftFusedContextUploadNext(storage.fused,requests.data(),count,deviceAccepted,next,nullptr);
            } else if (deviceAccepted)
                uploaded = FastllmDraftFusedContextSharedUploadAccepted(storage.fused,requests.data(),count,deviceAccepted,nullptr);
            else
                uploaded = FastllmDraftFusedContextSharedUpload(storage.fused,requests.data(),count);
            AssertInFastLLM(uploaded,"Shared context descriptor upload failed.");
        }
        draftTP->Run([&](int rank) {
            // Preserve first-use loading before any peer can enter a resident
            // collective. Subsequent commits may overlap host preparation.
            if (targetFirst) enqueueTarget(rank);
            auto &scratch = *draftBatchStorage->tpRanks[rank];
            if (propose) {
                auto &body=storage.nextGraph->tpRanks[rank]->graphs[0];
                auto *kvDone=draftContextStorage->tpRanks[rank]->done;
                FastllmCudaEventRecordCurrentThread(kvDone);
                FastllmCudaStreamWaitEvent(scratch.prefetchStream,kvDone);
                AssertInFastLLM(FastllmDraftFusedAcceptedBodyRun(storage.fused,body.acceptedExec,scratch.prefetchStream),
                    "Cannot replay accepted context and draft body.");
                FastllmCudaEventRecord(scratch.prefetchReady,scratch.prefetchStream);
                FastllmCudaCurrentThreadStreamWaitEvent(kvDone);
                scratch.backboneReady=scratch.prefetchReady;
                return;
            }
            if (queued) {
                // This rank's stream already owns the accepted context update.
                // Append KV commit/lookahead without submitting it twice.
            } else if (shared) {
                AssertInFastLLM(FastllmDraftFusedContextSharedRun(draftBatchStorage->fused,rank,count,nullptr),
                               "Shared draft context launch failed.");
            } else for (int row = 0; row < count; ++row)
                AssertInFastLLM(FastllmDraftFusedContextRun(draftBatchStorage->fused,rank,row,
                    (const uint16_t*)features[rank]->cudaData+(size_t)offsets[row]*4096,
                    counts[row],contexts[row]->committed,old[row],
                    (const void *const *)scratch.cacheTable.cudaData,count,row,nullptr),
                    "Fused draft context launch failed.");
            // The draft body reads its own KV. Release it before submitting the
            // independent target cache update; the public event still covers both.
            if (enqueueNext) enqueueNext(rank);
            if (enqueueTarget && !targetFirst) enqueueTarget(rank);
            FastllmCudaEventRecordCurrentThread(draftContextStorage->tpRanks[rank]->done);
            if (sampled) {
                auto &body = storage.nextGraph->tpRanks[rank]->graphs[0];
                FastllmCudaStreamWaitEvent(scratch.prefetchStream,draftContextStorage->tpRanks[rank]->done);
                AssertInFastLLM(FastllmCudaGraphLaunchOnStream(body.exec,scratch.prefetchStream),
                    "Cannot launch sampled draft backbone.");
                FastllmCudaEventRecord(scratch.prefetchReady,scratch.prefetchStream);
                scratch.backboneReady = scratch.prefetchReady;
            }
        });
    } catch (...) {
        // A partial submission cannot fall back. Retain borrowed request/cache
        // storage until every producer stream has drained, then propagate.
        draftTP->Run([&](int) { FastllmCudaSyncCurrentThreadStream(); });
        if (sampled) draftTP->Run([&](int rank) {
            auto &local = *storage.tpRanks[rank];
            FastllmCudaEventRecord(local.prefetchReady,local.prefetchStream);
            FastllmCudaEventSynchronize(local.prefetchReady);
        });
        // The lookahead callback is only supplied by CommitSpeculativeCaches,
        // which holds historyMutex across submission and exception cleanup.
        if (enqueueNext) draftBatchStorage->JoinLookahead();
        throw;
    }
    if (deviceAccepted) {
        storage.pendingBackbone = sampled != nullptr;
        storage.nextProposed = propose;
        return true;
    }
    storage.DrainPendingBackbone();
    for (int rank = 1; rank < 8; ++rank)
        FastllmCudaCurrentThreadStreamWaitEvent(draftContextStorage->tpRanks[rank]->done);
    FastllmCudaSyncCurrentThreadStream();
    if (enqueueTarget) draftBatchStorage->targetCommitReady = true;
    draftBatchStorage->pendingContext = nullptr;
    draftBatchStorage->pendingContextOwners.clear();
    for (int row = 0; row < count; ++row) {
        const int length = std::min(old[row],1023-counts[row])+counts[row];
        for (int rank = 0; rank < 8; ++rank) {
            auto &cache = rank ? contexts[row]->workspace->tpRanks[rank]->tpKV : contexts[row]->kv;
            for (auto &pair : cache) for (Data *t : {&pair.first,&pair.second}) FastllmNaiveResizeKVLength(*t,length,128);
        }
        contexts[row]->committed += counts[row];
        ClearDraftPrefixHidden(contexts[row]->prefixHidden);
    }
    return true;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::RunDraftBatch(const std::vector<int> &anchors,
        const std::vector<DraftContext *> &contexts, Data &normalized, int reserveRows,
        DraftHeadShards *headShards, const FastllmDraftHeadCall *headCall) {
#ifdef USE_CUDA
    if (draftBatchStorage) draftBatchStorage->replay.Begin(headCall != nullptr);
#endif
    const auto run = headShards ? &NaiveN05FlashModel::RunLayerDraftBatch : DraftOps().body;
    return (this->*run)(anchors, contexts, normalized, reserveRows, headShards, headCall);
}

bool NaiveN05FlashModel::CanRunSingleOwnerProposal() const {
    return attentionDP > 1 && CanShardDraftHead() && !draftConfidenceThreshold &&
        GetCudaEmbedding() && !GetLowMemMode() && GetFastllmEnv().cudaGraph;
}

#ifdef USE_CUDA
void NaiveN05FlashModel::RunDraftShardedHead(DraftHeadShards &head,
        const std::vector<std::vector<int>> &rankRequests, const std::vector<void *> &done, bool waitForOwner) {
    const auto &devices = draftTP->devices;
    const int batch = head.limits.size(), ranks = devices.size();
    auto scratch = [&](int rank) -> DraftWorkspace & { return *draftBatchStorage->tpRanks[rank]; };
    // Each vocabulary shard reads the live prefix directly from each DP owner.
    // Source rows are owner-local and can be reordered or padded by the backbone graph.
    std::vector<int> offsets(batch + 1);
    std::partial_sum(head.limits.begin(), head.limits.end(), offsets.begin() + 1);
    std::vector<const void *> sources(batch);
    std::vector<size_t> bytes(batch);
    for (int rank = 0; rank < ranks; ++rank) {
        const auto &requests = rankRequests[rank];
        const auto *source = static_cast<const uint16_t *>(scratch(rank).normalized.cudaData);
        for (size_t row = 0; row < requests.size(); ++row) {
            const int i = requests[row];
            sources[i] = source + row * draftBlock * embed_dim;
            bytes[i] = size_t(head.limits[i]) * embed_dim * sizeof(uint16_t);
        }
    }
    head.inputs.resize(ranks);
    head.logits.resize(ranks);
    draftTP->Run([&](int rank) {
        for (int peer = 0; peer < ranks; ++peer) if ((waitForOwner || peer != rank) && !rankRequests[peer].empty())
            FastllmCudaCurrentThreadStreamWaitEvent(done[peer]);
        auto &projection = *weight["lm_head.weight"].multiDeviceDatas.at(devices[rank]);
        ProjectCudaDraftHeadShard({devices[rank], tpVocabRanges[rank], projection, head.inputs[rank],
            head.logits[rank], head.ownerLogits}, scratch(rank).normalized, embed_dim, sources, offsets, bytes);
        // Separate from the backbone event: a slower peer can still be enqueueing its input reads.
        scratch(rank).RecordPeerEvent(scratch(rank).done);
    });
}

#endif

bool NaiveN05FlashModel::RunLayerDraftBatch(const std::vector<int> &anchors,
        const std::vector<DraftContext *> &contexts, Data &normalized, int reserveRows,
        DraftHeadShards *headShards, const FastllmDraftHeadCall *headCall) {
#ifdef USE_CUDA
    GraphCaptureScope capturePolicy(preparingGraphs);

    if (contexts.size() < (draftTP ? 1 : 2) || anchors.size() != contexts.size() ||
        !GetCudaEmbedding() || GetLowMemMode() || draftBlock < 2 || draftBlock >= 32 ||
        draftHeadDim % 4 || draftHeadDim > 256) return false;
    ApplyDraftDevice();
    const std::vector<int> devices = draftTP ? draftTP->devices : std::vector<int>{FastllmCudaGetDevice()};
    const int ranks = devices.size(), device = devices.front();
    const bool dataParallel = draftTP && draftTP->dataParallel;
    const int divisor = draftTP ? draftTP->attentionDivisor : 1;
    const int batch = contexts.size();
    // Keep each request's KV on a stable DP owner, including cancellation,
    // replacement and reordered batches. Only feature rows cross devices.
    std::vector<std::vector<int>> rankRequests(ranks);
    for (size_t i = 0; i < contexts.size(); ++i) {
        if (!contexts[i]) return false;
        if (dataParallel) {
            const int rank = contexts[i]->deviceRank;
            if (rank < 0 || rank >= ranks) return false;
            rankRequests[rank].push_back(i);
        } else for (auto &requests : rankRequests) requests.push_back(i);
    }
    if (headShards && (!CanShardDraftHead() || headShards->limits.size() != contexts.size() ||
        std::any_of(headShards->limits.begin(), headShards->limits.end(), [&](int n) { return n < 1 || n > draftBlock; })))
        return false;
    if (dataParallel && draftBatchStorage && !preparingGraphs) {
        std::vector<void *> done(ranks);
        bool ready = headShards != nullptr;
        {
            std::lock_guard<std::mutex> guard(historyMutex);
            for (int rank = 0; rank < ranks; ++rank) {
                const auto &requests = rankRequests[rank];
                if (requests.empty()) continue;
                auto &queued = draftBatchStorage->tpRanks[rank]->ownerReplay;
                std::lock_guard<std::mutex> lock(queued.mutex);
                if (reserveRows <= static_cast<int>(requests.size()) && queued.Matches(requests, contexts, anchors))
                    done[rank] = queued.bucket->done;
                else ready = false;
            }
            if (ready && !headShards->ownerLogits && draftBatchStorage->projectedHeadRows == rankRequests &&
                std::all_of(headShards->limits.begin(), headShards->limits.end(), [&](int n) { return n == draftTokens; })) {
                headShards->gatheredBuffers = std::move(draftBatchStorage->gatheredHeadBuffers);
                draftBatchStorage->projectedHeadRows.clear();
            } else draftBatchStorage->DrainProjectedHead();
            for (int rank = 0; rank < ranks; ++rank) {
                auto &queued = draftBatchStorage->tpRanks[rank]->ownerReplay;
                std::lock_guard<std::mutex> lock(queued.mutex);
                if (ready && done[rank]) queued.consumed = true;
                else queued.Join();
            }
        }
        if (ready) {
            headShards->ownerBodies = std::all_of(rankRequests.begin(), rankRequests.end(),
                [](const auto &rows) { return rows.size() <= 1; });
            if (headShards->gatheredBuffers.empty()) RunDraftShardedHead(*headShards, rankRequests, done, true);
            return true;
        }
    }
    const int heads = draftHeads / divisor, kvHeads = std::max(1, draftKvHeads / divisor);
#ifndef USE_ROCM
    const auto fusion = draftTP && !dataParallel && devices == tpDevices
        ? FastllmCudaDraftLayerSupport(ranks, batch, draftBlock, heads, kvHeads, draftHeadDim, draftWindow)
        : FastllmDraftLayerSupport{};
#endif
    auto weightAt = [&](const std::string &name, int rank) -> Data & {
        return draftTP ? *weight[name].multiDeviceDatas.at(devices[rank]) : DraftWeight(name);
    };
    auto cacheAt = [&](DraftContext &context, int rank, int layer) -> std::pair<Data, Data> & {
        return rank && !dataParallel ? context.workspace->tpRanks[rank]->tpKV[layer] : context.kv[layer];
    };
    Data &embedding = weightAt("model.embed_tokens.weight", 0), &mask = weightAt("dspark.mask_embedding", 0);
    if (embedding.dataType != BFLOAT16 || mask.dataType != BFLOAT16 || mask.Count(0) != embed_dim) return false;
    bool bindableCaches = true;
    for (auto *context : contexts) {
        if (context->kv.size() != (size_t)draftLayers ||
            (draftTP && (!context->workspace || context->workspace->tpRanks.size() != devices.size()))) return false;
        for (int rank = 0; rank < ranks; ++rank) if (!dataParallel || rank == context->deviceRank)
        for (int layer = 0; layer < draftLayers; ++layer)
        for (Data *cache : {&cacheAt(*context, rank, layer).first, &cacheAt(*context, rank, layer).second}) {
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds.size() != 1 || cache->dataDeviceIds.front() != devices[rank] ||
                cache->dims.size() != 3 ||
                cache->dims[2] != kvHeads * draftHeadDim || cache->expansionDims.size() != 3 ||
                cache->expansionDims[1] < draftWindow + draftBlock) return false;
            if (!cache->cudaData || cache->isPagedKVCache || cache->multiDeviceData ||
                cache->dims[0] != 1 || cache->expansionDims[0] != 1 ||
                cache->expansionDims[2] != kvHeads * draftHeadDim || cache->strides.size() != 3 ||
                cache->strides[1] != kvHeads * draftHeadDim || cache->strides[2] != 1)
                bindableCaches = false;
        }
    }
    embedding.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    mask.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    for (auto *context : contexts) if (!context->workspace) {
        context->workspace = std::make_shared<DraftWorkspace>();
        auto &state = *context->workspace;
        state.device = device; state.inputs = {embedding.cudaData, mask.cudaData};
        for (auto &pair : context->kv) for (Data *cache : {&pair.first, &pair.second}) state.inputs.push_back(cache->cudaData);
        state.layers.resize(draftLayers); state.contextLayers.resize(draftLayers);
    }
    // The fused kernel also accepts null KV bindings for padded slots.
    bool dynamicKV = bindableCaches;
    std::vector<int> rankRows;
    for (int rank = 0; rank < ranks; ++rank) {
        rankRows.push_back(rankRequests[rank].size());
        if (!rankRows.back()) continue;
        FastllmCudaSetDevice(devices[rank]);
        dynamicKV = (dataParallel || rankRows.back() >= 8) &&
            FastllmCudaNaiveDraftCacheBindingsSupported(rankRows.back(), draftBlock,
            heads, kvHeads, draftHeadDim, draftWindow) && dynamicKV;
    }
    FastllmCudaSetDevice(device);
    if (!draftBatchStorage || draftBatchStorage->device != device) {
        draftBatchStorage = std::make_shared<DraftWorkspace>();
        draftBatchStorage->device = device;
        for (int gpu : devices) {
            auto scratch = std::make_shared<DraftWorkspace>(); scratch->device = gpu;
            if (draftExperts) scratch->moe = std::make_unique<DraftMoeScratch>();
            draftBatchStorage->tpRanks.push_back(std::move(scratch));
        }
    }
    auto scratch = [&](int rank) -> DraftWorkspace & { return *draftBatchStorage->tpRanks.at(rank); };
    const bool collective = draftTP && draftTP->moe;
    int collectiveRows = 0;
    if (collective) {
        int rows = *std::max_element(rankRows.begin(), rankRows.end());
        if (dynamicKV && GetFastllmEnv().cudaGraph) {
            const int bound = WindowRequestsPerGroup(maxBatch > 0 ? maxBatch : 512, ranks);
            rows = DraftGraphBucket(std::max(rows, reserveRows), bound);
            for (auto &n : rankRows) if (n) n = rows;
        }
        collectiveRows = draftBlock * rows;
        FastllmCudaPrepareMimoEp8Mega(*draftTP->moe, collectiveRows);
    }
    const bool graphEnabled = GetFastllmEnv().cudaGraph && bindableCaches && (!draftExperts || collective);
    const bool bucketedDP = dataParallel && dynamicKV && graphEnabled && !collective;
    std::shared_ptr<DraftWorkspace> state;
    if (bucketedDP) {
        const int maxLocal = WindowRequestsPerGroup(maxBatch > 0 ? maxBatch : 512, ranks);
        for (int rank = 0; rank < ranks; ++rank) {
            if (!rankRows[rank]) continue;
            rankRows[rank] = DraftGraphBucket(std::max(rankRows[rank], reserveRows), maxLocal);
            auto &entry = scratch(rank).batchGraphs[rankRows[rank]];
            if (entry) continue;
            FastllmCudaSetDevice(devices[rank]);
            entry = std::make_shared<DraftWorkspace>();
            entry->device = devices[rank];
            entry->done = FastllmCudaEventCreate();
            entry->batchDescriptors.resize(draftLayers);
        }
    } else {
        std::vector<void *> inputs{embedding.cudaData, mask.cudaData};
        if (headCall) inputs.push_back(headCall->state);
        if (graphEnabled) {
            auto found = std::find_if(draftBatchWorkspaces.begin(), draftBatchWorkspaces.end(),
                [&](const std::shared_ptr<DraftWorkspace> &entry) {
                    return entry->device == device && entry->inputs == inputs && !entry->lowLatencyBody &&
                        entry->rankRows == rankRows;
                });
            if (found != draftBatchWorkspaces.end()) {
                state = *found;
                draftBatchWorkspaces.erase(found);
            }
        }
        if (!state) {
            state = std::make_shared<DraftWorkspace>();
            state->device = device;
            state->inputs = std::move(inputs);
            state->rankRows = rankRows;
            if (draftTP) for (int gpu : devices) {
                FastllmCudaSetDevice(gpu);
                auto local = std::make_shared<DraftWorkspace>();
                local->device = gpu;
                local->done = FastllmCudaEventCreate();
                state->tpRanks.push_back(std::move(local));
            }
        }
        if (graphEnabled) {
            draftBatchWorkspaces.push_back(state);
            if (draftBatchWorkspaces.size() > 8) draftBatchWorkspaces.erase(draftBatchWorkspaces.begin());
        }
    }
    FastllmCudaSetDevice(device);
    auto localState = [&](int rank) -> DraftWorkspace & {
        if (bucketedDP) return *scratch(rank).batchGraphs.at(rankRows[rank]);
        return draftTP ? *state->tpRanks[rank] : *state;
    };
    // Install map entries on the caller before rank workers read them.
    for (int gpu : devices) draftLinearGraphWorkspaces[gpu];
    auto run = [&](const std::function<void(int)> &fn) {
        if (draftTP) draftTP->Run(fn); else fn(0);
    };
    // Store token IDs and live lengths after this bucket's KV bindings when
    // the existing descriptor allocation has room. Views depend only on the
    // captured bucket; each replay uploads fresh request data, including padding.
    auto packedInputs = [&](int count) {
        return headCall || (bucketedDP && size_t(count) * (2 * draftLayers + 1) <=
            size_t(std::max(maxBatch, count)) * 2 * draftLayers + 1);
    };
    auto upload = [&](int rank) {
        auto &r = scratch(rank);
        const auto &requests = rankRequests[rank];
        if (requests.empty()) return;
        const int count = rankRows[rank];
        auto copy = [&](Data &target, DataType type, const void *host, const std::vector<int> &dims) {
            target.dataType = type; target.UpdateUnitSize(); target.Resize(dims);
            target.ToDevice(DataDevice::CUDA, {devices[rank]}, false); target.Allocate(false);
            FastllmCudaCopyFromHostToDevice(target.cudaData, const_cast<void *>(host), target.GetBytes());
        };
        std::vector<float> localTokens(count);
        std::vector<int> localLengths(count);
        for (size_t row = 0; row < requests.size(); ++row) {
            const int seq = requests[row];
            localTokens[row] = anchors[seq];
            localLengths[row] = contexts[seq]->committed + 1;
        }
        const bool packed = packedInputs(count);
        if (!packed) {
            copy(r.id, FLOAT32, localTokens.data(), {1, count});
            copy(r.live, INT32, localLengths.data(), {count});
        }
        if (dynamicKV || graphEnabled || headCall) {
            const size_t slots = size_t(draftLayers) * count * 2;
            std::vector<uint64_t> bindings(slots + (packed ? count : 0));
            if (headCall && rank == 0) headCall->request.AppendTo(bindings);
            for (int layer = 0; layer < draftLayers; ++layer) for (size_t row = 0; row < requests.size(); ++row) {
                auto &cache = cacheAt(*contexts[requests[row]], rank, layer);
                const size_t slot = (size_t(layer) * count + row) * 2;
                bindings[slot] = (uint64_t)cache.first.cudaData;
                bindings[slot + 1] = (uint64_t)cache.second.cudaData;
            }
            if (packed) {
                auto *tail = reinterpret_cast<unsigned char *>(bindings.data() + slots);
                std::memcpy(tail, localTokens.data(), count * sizeof(float));
                std::memcpy(tail + count * sizeof(float), localLengths.data(), count * sizeof(int));
            }
            r.batchKV.dataType = INT32; r.batchKV.UpdateUnitSize();
            if (headCall) r.batchKV.Resize({int(bindings.size() * 2)});
            else r.batchKV.Resize({draftLayers * std::max(maxBatch, count) * 4 + (bucketedDP ? 2 : 0)});
            r.batchKV.ToDevice(DataDevice::CUDA, {devices[rank]}, false); r.batchKV.Allocate(false);
            FastllmCudaCopyFromHostToDevice(r.batchKV.cudaData, bindings.data(), bindings.size() * sizeof(uint64_t));
        }
    };
    auto body = [&](int rank, bool reduce = true) {
        const auto &selected = rankRequests[rank];
        if (selected.empty()) {
            if (collective) {
                Data empty(BFLOAT16, {0, embed_dim}), unused;
                for (int layer = 0; layer < draftLayers; ++layer)
                    RunDraftMoe(layer, rank, empty, unused, *scratch(rank).moe, collectiveRows, reduce);
            }
            return;
        }
        std::vector<DraftContext *> localContexts;
        for (int i : selected) localContexts.push_back(contexts[i]);
        // Only geometry is captured. Padded slots bind null in the replay table.
        if (bucketedDP || collective) localContexts.resize(rankRows[rank], localContexts.front());
#ifndef USE_ROCM
        FastllmCudaLinearGraphScope linearScope(draftLinearGraphWorkspaces.at(devices[rank]));
#endif
        auto w = [&](const std::string &name) -> Data & { return weightAt(name, rank); };
        auto linear = [&](Data &input, Data &weight, Data &output) {
            // TP uses the existing cached Lt/GEMV dispatch. The strided-GEMM
            // API can lazily change CUDA kernel attributes between collectives.
            if (!draftTP || (dataParallel && PreferCudaDraftReplicaMatMul(input, weight))) DraftLinear(input, weight, output);
            else Linear(input, weight, Data(), output);
        };
        auto allReduce = [&](Data &value, Data &reduced) {
            if (!draftTP || dataParallel) return;
            // Keep both typed allocations stable across graph variants.
            // In-place conversion replaces the shared BF16 pointer during
            // capture/replay transitions and defeats workspace reuse.
            ToDataType(value, reduced, FLOAT32);
            if (reduce) FastllmCudaNaiveDraftTPReduce(draftTP->group, rank, reduced);
            ToDataType(reduced, value, BFLOAT16);
        };
        auto &r = scratch(rank);
        if (r.layers.empty()) r.layers.resize(draftLayers);
        Data packedIds, packedLive;
        const int count = rankRows[rank];
        const bool packed = packedInputs(count);
        if (packed) {
            const size_t tail = size_t(draftLayers) * count * 2 * sizeof(uint64_t);
            packedIds.FakeFrom(r.batchKV, tail);
            packedIds.dataType = FLOAT32; packedIds.UpdateUnitSize(); packedIds.Resize({1, count});
            packedLive.FakeFrom(r.batchKV, tail + count * sizeof(float)); packedLive.Resize({count});
        }
        Data &ids = packed ? packedIds : r.id, &live = packed ? packedLive : r.live;
        Data &result = r.normalized;
        auto addNorm = [&](Data &hidden, Data &branch, Data &gamma, Data &output) {
            gamma.ToDevice(DataDevice::CUDA, std::vector<int>{devices[rank]});
            if (hidden.dataType == BFLOAT16 && branch.dataType == BFLOAT16 &&
                    gamma.dataType == FLOAT32 && gamma.Count(0) == hidden.dims.back()) {
                FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, gamma, draftEps, output);
            } else {
                AddTo(hidden, branch);
                KimiK3RMSNorm(hidden, gamma, draftEps, output);
            }
        };
        auto reduceAddNorm = [&](Data &hidden, Data &branch, Data &reduced, Data &gamma,
                                 Data &output, bool needsReduce) {
#ifndef USE_ROCM
            if (needsReduce && reduce && fusion.reduceNorm &&
                FastllmCudaTryDraftReduceAddNorm(devices[rank], draftEps, hidden, branch, gamma, output))
                return;
#endif
            if (needsReduce) allReduce(branch, reduced);
            addNorm(hidden, branch, gamma, output);
        };
        Data &embedding = w("model.embed_tokens.weight"), &mask = w("dspark.mask_embedding");
        struct Request { Data positions, live; };
        std::vector<Request> requests(localContexts.size());
        auto append = [](Data &batch, Data &part, bool first) {
            if (first) Copy(part, batch);
            else { Data joined; Cat(batch, part, 1, joined); Copy(joined, batch); }
        };
        Data &hidden = r.hidden, &positions = r.positions, &normed = r.combined;
        bool preparedEmbedding = false;
#ifndef USE_ROCM
        if (headCall)
            preparedEmbedding = FastllmCudaPrepareDraftEmbedding(
                ids, live, embedding, mask, w("dspark.layers.0.input_layernorm.weight"), hidden, positions,
                normed, draftEps, devices[rank]);
#endif
        if (!preparedEmbedding)
            FastllmCudaNaiveDraftInput(ids, embedding, mask, live, draftBlock, hidden, positions);
        for (size_t row = 0; row < localContexts.size(); ++row) {
            requests[row].positions.Resize({1, draftBlock});
            requests[row].positions.FakeFrom(positions, row * draftBlock * sizeof(float));
            requests[row].live.Resize({1}); requests[row].live.FakeFrom(live, row * sizeof(int));
        }
        if (!preparedEmbedding) KimiK3RMSNorm(hidden, w("dspark.layers.0.input_layernorm.weight"), draftEps, normed);
        for (int layer = 0; layer < draftLayers; ++layer) {
            const std::string p = "dspark.layers." + std::to_string(layer) + ".";
            auto &l = r.layers[layer];
            Data &qkv = l.qkv, &qs = l.q, &ks = l.k, &vs = l.v, &attention = l.attention;
            Data &output = l.output;
            const bool merged = weight.weight.count(p + "self_attn.mergeqkv.weight");
            std::vector<FastllmNaiveDraftRow> descriptors;
            for (size_t row = 0; row < localContexts.size(); ++row) {
                auto *context = localContexts[row];
                auto &cache = cacheAt(*context, rank, layer);
                const auto *table = static_cast<const void *const *>(r.batchKV.cudaData);
                const auto *pair = graphEnabled && !dynamicKV ? table + 2 * (layer * localContexts.size() + row) : nullptr;
                descriptors.push_back({&cache.first, &cache.second, &requests[row].positions,
                    &requests[row].live, !dynamicKV && std::min(context->committed, draftWindow - 1) + draftBlock <= 256, pair});
            }
            Data &packed = l.packed, &query = l.query, &scores = l.scores;
            Data bindings(INT32, {(int)localContexts.size(), 4});
            if (dynamicKV) bindings.FakeFrom(r.batchKV, (size_t)layer * localContexts.size() * 2 * sizeof(uint64_t));
            DraftCollectiveLayer collectiveAttention{};
#ifndef USE_ROCM
            collectiveAttention = {fusion.attention && reduce && bindableCaches && (dynamicKV || graphEnabled),
                &l.attentionProjection, &live, &r.batchKV, layer};
#endif
            const DraftLayerParallel attentionPlacement{devices, rank, divisor};
            bool fusedAttention = false;
            if (merged) {
                const DraftAttentionWeights weights{&w(p + "self_attn.mergeqkv.weight"),
                    w(p + "self_attn.o_proj.weight"), w(p + "self_attn.q_norm.weight"),
                    w(p + "self_attn.k_norm.weight"), w(p + "post_attention_layernorm.weight")};
                fusedAttention = TryCudaDraftAttention(attentionPlacement,
                    {heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta}, descriptors,
                    dynamicKV ? &bindings : nullptr, normed, weights, hidden, qkv, attention, output, r.layers.front().scores, &collectiveAttention);
            }
            if (!fusedAttention) {
                if (merged) linear(normed, w(p + "self_attn.mergeqkv.weight"), qkv);
                else {
                    linear(normed, w(p + "self_attn.q_proj.weight"), qs);
                    linear(normed, w(p + "self_attn.k_proj.weight"), ks);
                    linear(normed, w(p + "self_attn.v_proj.weight"), vs);
                    Data qk;
                    Cat(qs, ks, -1, qk); Cat(qk, vs, -1, packed);
                }
                const bool batchedAttention = FastllmCudaNaiveBatchDraftQKV(merged ? qkv : packed,
                        w(p + "self_attn.q_norm.weight"), w(p + "self_attn.k_norm.weight"),
                        descriptors, query, heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta,
                        &localState(rank).batchDescriptors[layer], dynamicKV ? &bindings : nullptr) &&
                    FastllmCudaNaiveBatchDraftAttention(query, descriptors, heads, kvHeads,
                        draftHeadDim, draftWindow, scores, attention, dynamicKV ? &bindings : nullptr);
                if (!batchedAttention) {
                    // Scalar attention embeds the current request's KV addresses.
                    localState(rank).disabled = true;
                    Data part;
                    for (size_t row = 0; row < selected.size(); ++row) {
                        auto &r = requests[row]; auto &cache = cacheAt(*localContexts[row], rank, layer);
                        Data raw, q, k, v, scores;
                        const int begin = row * draftBlock, end = begin + draftBlock;
                        if (merged) Split(qkv, 1, begin, end, raw);
                        if (!merged || !FastllmCudaNaiveDraftQKV(raw,
                                w(p + "self_attn.q_norm.weight"), w(p + "self_attn.k_norm.weight"),
                                r.positions, r.live, cache.first, cache.second, q,
                                heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta)) {
                            if (merged) {
                                const int qw = heads * draftHeadDim, kw = kvHeads * draftHeadDim;
                                Split(raw, -1, 0, qw, q); Split(raw, -1, qw, qw + kw, k);
                                Split(raw, -1, qw + kw, qw + 2 * kw, v);
                            } else {
                                Split(qs, 1, begin, end, q); Split(ks, 1, begin, end, k); Split(vs, 1, begin, end, v);
                            }
                            q.Reshape({1, draftBlock * heads, draftHeadDim});
                            k.Reshape({1, draftBlock * kvHeads, draftHeadDim});
                            KimiK3RMSNorm(q, w(p + "self_attn.q_norm.weight"), draftEps, q);
                            KimiK3RMSNorm(k, w(p + "self_attn.k_norm.weight"), draftEps, k);
                            q.Reshape({1, draftBlock, heads * draftHeadDim});
                            k.Reshape({1, draftBlock, kvHeads * draftHeadDim});
                            FastllmCudaNaiveRope(q, r.positions, heads, draftHeadDim, draftHeadDim, draftTheta);
                            FastllmCudaNaiveRope(k, r.positions, kvHeads, draftHeadDim, draftHeadDim, draftTheta);
                            FastllmCudaNaiveAppendVerifyCache(cache.first, cache.second, k, v, r.live, draftWindow);
                        }
                        const bool shortAttention = std::min(localContexts[row]->committed, draftWindow - 1) + draftBlock <= 256;
                        FastllmCudaNaiveDraftAttention(q, cache.first, cache.second, r.live,
                            heads, kvHeads, draftHeadDim, draftWindow, shortAttention, scores, part);
                        append(attention, part, row == 0);
                    }
                    // Padding has no KV owner. Only extend the projection input.
                    for (size_t row = selected.size(); row < localContexts.size(); ++row)
                        append(attention, part, false);
                }
                linear(attention, w(p + "self_attn.o_proj.weight"), output);
                reduceAddNorm(hidden, output, l.reduced, w(p + "post_attention_layernorm.weight"), normed,
                              draftTP && draftTP->attentionDivisor > 1);
            }
            const bool last = layer + 1 == draftLayers;
            Data &gamma = last ? w("dspark.norm.weight") :
                w("dspark.layers." + std::to_string(layer + 1) + ".input_layernorm.weight");
            Data &next = last ? result : normed;
            if (draftExperts) {
                RunDraftMoe(layer, rank, normed, output, *r.moe, collectiveRows, reduce);
                reduceAddNorm(hidden, output, l.reduced, gamma, next, false);
            } else {
                DraftCollectiveLayer collectiveMlp{};
#ifndef USE_ROCM
                collectiveMlp = {fusion.mlp && reduce, &l.mlpProjection};
#endif
                auto finish = [&] { reduceAddNorm(hidden, output, l.reduced, gamma, next, true); };
                RunDenseDraftMlp({devices, rank, dataParallel ? 1 : ranks}, count,
                    weight.weight.count(p + "mlp.gateup_proj.weight"), p, w, linear, finish,
                    normed, hidden, gamma, draftEps, l, next, &collectiveMlp);
            }
        }
#ifndef USE_ROCM
        if (headCall && reduce)
            FastllmCudaLaunchDraftProposal(headCall->state, rank, result, r.batchKV, draftLayers);
#endif
    };
    if (bucketedDP) {
        run([&](int rank) {
            if (rankRequests[rank].empty()) return;
            auto &entry = localState(rank);
            auto &graph = entry.graphs[0];
            const bool warm = !graph.exec;
            const auto before = warm ? scratch(rank).Pointers() : std::vector<void *>{};
            upload(rank);
            if (warm) {
                body(rank);
                FastllmCudaSyncCurrentThreadStream();
                if (before != scratch(rank).Pointers())
                    for (auto &other : scratch(rank).batchGraphs)
                        if (other.second.get() != &entry) other.second->graphs[0].Clear();
                if (preparingGraphs && !entry.disabled && graph.Capture([&] { body(rank); }, entry.disabled) &&
                    !draftExperts && packedInputs(rankRows[rank]) &&
                    (rankRows[rank] > 1 || FastllmCudaNaivePrepareKVBody(graph.graph, draftLayers, graph.kvBody)) &&
                    !entry.ownerPacket)
                    entry.ownerPacket = FastllmCudaHostMalloc(size_t(rankRows[rank]) * (2 * draftLayers + 1) * sizeof(uint64_t));
            }
            if (graph.exec) AssertInFastLLM(FastllmCudaGraphLaunch(graph.exec), "Cannot replay draft DP graph");
            if (ranks > 1) scratch(rank).RecordPeerEvent(entry.done);
            else FastllmCudaEventRecordCurrentThread(entry.done);
        });
        if (!headShards) for (int rank = 1; rank < ranks; ++rank) if (!rankRequests[rank].empty())
            FastllmCudaCurrentThreadStreamWaitEvent(localState(rank).done);
    } else {
        for (int rank = 0; rank < ranks; ++rank) localState(rank).batchDescriptors.resize(draftLayers);
        std::vector<std::vector<void *>> before;
        for (int rank = 0; rank < ranks; ++rank) before.push_back(scratch(rank).Pointers());
        auto invalidateMoved = [&]() {
            bool changed = false;
            for (int rank = 0; rank < ranks; ++rank) changed = changed || before[rank] != scratch(rank).Pointers();
            if (!changed) return;
            for (auto &other : draftBatchWorkspaces) if (other != state) {
                other->graphs[0].Clear();
                for (auto &rank : other->tpRanks) rank->graphs[0].Clear();
            }
        };
        run(upload);
        if (draftTP && !state->warmed) {
            // Initialize every GEMM plan/kernel before any rank enters NCCL.
            // Only speculative scratch is written; committed KV is unchanged.
            // A CUDA lazy-load/attribute update while another rank is already
            // waiting in a collective can otherwise deadlock the driver.
            run([&](int rank) { body(rank, false); ForceDeviceSync(); });
            state->warmed = true;
        }
        bool useGraph = graphEnabled && !state->disabled &&
            (preparingGraphs || localState(0).graphs[0].exec);
        if (useGraph && !localState(0).graphs[0].exec) {
            run([&](int rank) { body(rank); ForceDeviceSync(); });
            invalidateMoved();
            if (std::any_of(state->tpRanks.begin(), state->tpRanks.end(),
                            [](const auto &rank) { return rank->disabled; }) || state->disabled) {
                state->disabled = true;
                useGraph = false;
            } else if (!draftTP) useGraph = state->graphs[0].Capture([&] { body(0); }, state->disabled);
            else {
                std::vector<int> ready(ranks);
                auto allReady = [&] { return std::all_of(ready.begin(), ready.end(), [](int n) { return n != 0; }); };
                run([&](int rank) { ready[rank] = FastllmCudaGraphPrepareCaptureDevice(); });
                bool pool = allReady() && FastllmCudaGraphMemoryPoolBegin(), ok = pool;
                if (pool) {
                    run([&](int rank) { FastllmCudaClearThreadError(); ready[rank] = FastllmCudaGraphBeginCapture(); });
                    ok = allReady();
                    run([&](int rank) {
                        if (!ready[rank]) return;
                        if (ok) body(rank);
                        ready[rank] = FastllmCudaGraphEndCapture(&localState(rank).graphs[0].graph) &&
                                      !FastllmCudaGetThreadError() && !localState(rank).disabled;
                    });
                    ok = FastllmCudaGraphMemoryPoolEnd(state->graphs[0].reserved) && ok && allReady();
                    if (ok) {
                        run([&](int rank) { auto &g = localState(rank).graphs[0]; ready[rank] = FastllmCudaGraphInstantiate(g.graph, &g.exec); });
                        ok = allReady();
                    }
                }
                if (!ok) {
                    run([&](int rank) { localState(rank).graphs[0].Clear(); FastllmCudaClearLastError(); FastllmCudaClearThreadError(); });
                    state->graphs[0].Clear(); state->disabled = true; useGraph = false;
                }
            }
        }
        // Capture can also materialize operator-owned buffers after warmup.
        // Retire other variants before replay if any shared address moved.
        invalidateMoved();
        run([&](int rank) {
            auto &r = localState(rank);
#ifndef USE_ROCM
            auto &g=r.graphs[0];
            if (headCall && preparingGraphs && maxBatch==1 && useGraph && !g.canContinue)
                g.canContinue = FastllmCudaCanContinueDraftGraph(g.graph);
#endif
            if (useGraph) AssertInFastLLM(FastllmCudaGraphLaunch(r.graphs[0].exec), "Naive batch draft graph launch failed.");
            else body(rank);
            if (ranks > 1) scratch(rank).RecordPeerEvent(r.done);
            else if (draftTP) FastllmCudaEventRecordCurrentThread(r.done);
        });
#ifndef USE_ROCM
        if (headCall && useGraph && maxBatch == 1) {
            draftBatchStorage->replay.Bind(state, headCall->state,
                [graph = state.get(), scratch = draftBatchStorage.get()](int rank) {
                    auto &g = graph->tpRanks[rank]->graphs[0];
                    return FastllmDraftGraphRank{g.canContinue ? g.exec : nullptr, scratch->tpRanks[rank]->batchKV.cudaData};
                });
        }
#endif
        if (!useGraph) invalidateMoved();
        if (draftTP && !headShards && !headCall) for (int rank = 1; rank < ranks; ++rank)
            FastllmCudaCurrentThreadStreamWaitEvent(localState(rank).done);
        if (!useGraph) invalidateMoved();
    }
    if (headCall) {
        FastllmCudaCurrentThreadStreamWaitEvent(localState(0).done);
        return true;
    }
    if (headShards) {
        headShards->collectiveDynamicKV = collective && dynamicKV;
        headShards->ownerBodies = bucketedDP;
        for (int rank = 0; rank < ranks && headShards->ownerBodies; ++rank)
            if (!rankRequests[rank].empty())
                headShards->ownerBodies = rankRows[rank] == 1 && localState(rank).graphs[0].kvBody.exec;
        std::vector<void *> done(ranks);
        for (int rank = 0; rank < ranks; ++rank)
            if (!rankRequests[rank].empty()) done[rank] = localState(rank).done;
        RunDraftShardedHead(*headShards, rankRequests, done, false);
        return true;
    }
    if (dataParallel) {
        normalized.dataType = BFLOAT16; normalized.UpdateUnitSize();
        normalized.Resize({1, batch * draftBlock, embed_dim});
        normalized.ToDevice(DataDevice::CUDA, {device}, false); normalized.Allocate(false);
        std::vector<void *> destinations;
        std::vector<const void *> sources;
        std::vector<size_t> sizes;
        const size_t width = size_t(draftBlock) * embed_dim;
        for (int rank = 0; rank < ranks; ++rank) {
            const auto &requests = rankRequests[rank];
            for (size_t row = 0; row < requests.size(); ++row) {
                destinations.push_back((uint16_t *)normalized.cudaData + size_t(requests[row]) * width);
                sources.push_back((const uint16_t *)scratch(rank).normalized.cudaData + row * width);
                sizes.push_back(width * sizeof(uint16_t));
            }
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(destinations.data(),
            sources.data(), sizes.data(), sizes.size()), "Draft DP output gathering failed.");
        // Workers may reuse their scratch on the next round. Complete the
        // root-stream peer reads before returning ownership to the scheduler.
        FastllmCudaSyncCurrentThreadStream();
    } else {
        scratch(0).normalized.Resize({1, batch * draftBlock, embed_dim});
        Copy(scratch(0).normalized, normalized);
    }
    return true;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::BorrowDraftLookahead(const std::vector<int> &anchors,
        const std::vector<DraftContext *> &contexts, Data &hidden) {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_DRAFT_FUSED)
    std::lock_guard<std::mutex> guard(historyMutex);
    if (!draftBatchStorage || !draftBatchStorage->ConsumeLookahead(anchors, contexts)) return false;
    ApplyDraftDevice();
    // The view is consumed within this forward. Keep the cross-rank join for
    // consumers which fall back after inspecting the request's sampling mode.
    for (const auto &rank : draftBatchStorage->tpRanks)
        FastllmCudaCurrentThreadStreamWaitEvent(rank->prefetchReady);
    const auto &source = draftBatchStorage->tpRanks.front()->normalized;
    hidden.FakeFrom(source, 0);
    hidden.Resize(source.dims);
    return true;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::RunDraftLowLatencyBatch(const std::vector<int> &anchors,
        const std::vector<DraftContext *> &contexts, Data &normalized, int, DraftHeadShards *, const FastllmDraftHeadCall *) {
    GraphCaptureScope capturePolicy(preparingGraphs);
#ifdef USE_CUDA
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    std::lock_guard<std::mutex> lookaheadGuard(historyMutex);

    if (draftBatchStorage) {
        auto &storage = *draftBatchStorage;
        if (storage.ConsumeLookahead(anchors, contexts)) {
            ApplyDraftDevice();
            for (const auto &rank : storage.tpRanks)
                FastllmCudaCurrentThreadStreamWaitEvent(rank->prefetchReady);
            Copy(storage.tpRanks.front()->normalized, normalized);
            return true;
        }
        storage.nextReady = false;
        storage.JoinLookahead();
        storage.nextGraph.reset(); storage.nextOwners.clear();
        storage.fusedContexts.clear();
    }
#endif
    if (contexts.empty() || anchors.size() != contexts.size() ||
        !GetCudaEmbedding() || GetLowMemMode() || draftBlock < 2 || draftBlock >= 32 ||
        draftHeadDim % 4 || draftHeadDim > 256) return false;
    ApplyDraftDevice();
    const std::vector<int> devices = draftTP ? draftTP->devices : std::vector<int>{FastllmCudaGetDevice()};
    const int ranks = devices.size(), device = devices.front();
    const bool dataParallel = draftTP && draftTP->dataParallel;
    const int divisor = draftTP ? draftTP->attentionDivisor : 1;
    const int batch = contexts.size();
    // Keep each request's KV on a stable DP owner, including cancellation,
    // replacement and reordered batches. Only feature rows cross devices.
    std::vector<std::vector<int>> rankRequests(ranks);
    for (size_t i = 0; i < contexts.size(); ++i) {
        if (!contexts[i]) return false;
        if (dataParallel) {
            const int rank = contexts[i]->deviceRank;
            if (rank < 0 || rank >= ranks) return false;
            rankRequests[rank].push_back(i);
        } else for (auto &requests : rankRequests) requests.push_back(i);
    }
    const int heads = draftHeads / divisor, kvHeads = std::max(1, draftKvHeads / divisor);
    auto weightAt = [&](const std::string &name, int rank) -> Data & {
        return draftTP ? *weight[name].multiDeviceDatas.at(devices[rank]) : DraftWeight(name);
    };
    auto cacheAt = [&](DraftContext &context, int rank, int layer) -> std::pair<Data, Data> & {
        return rank && !dataParallel ? context.workspace->tpRanks[rank]->tpKV[layer] : context.kv[layer];
    };
    Data &embedding = weightAt("model.embed_tokens.weight", 0), &mask = weightAt("dspark.mask_embedding", 0);
    if (embedding.dataType != BFLOAT16 || mask.dataType != BFLOAT16 || mask.Count(0) != embed_dim) return false;
    for (auto *context : contexts) {
        if (!context || context->kv.size() != (size_t)draftLayers ||
            (draftTP && (!context->workspace || context->workspace->tpRanks.size() != devices.size()))) return false;
        for (int rank = 0; rank < ranks; ++rank) if (!dataParallel || rank == context->deviceRank)
        for (int layer = 0; layer < draftLayers; ++layer)
        for (Data *cache : {&cacheAt(*context, rank, layer).first, &cacheAt(*context, rank, layer).second})
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds != std::vector<int>{devices[rank]} || !cache->cudaData ||
                cache->isPagedKVCache || cache->multiDeviceData || cache->dims.size() != 3 ||
                cache->dims[0] != 1 || cache->dims[2] != kvHeads * draftHeadDim ||
                cache->expansionDims.size() != 3 || cache->expansionDims[0] != 1 ||
                cache->expansionDims[1] < draftWindow + draftBlock ||
                cache->expansionDims[2] != kvHeads * draftHeadDim || cache->strides.size() != 3 ||
                cache->strides[1] != kvHeads * draftHeadDim || cache->strides[2] != 1) return false;
    }
    embedding.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    mask.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    for (auto *context : contexts) if (!context->workspace) {
        context->workspace = std::make_shared<DraftWorkspace>();
        auto &state = *context->workspace;
        state.device = device; state.inputs = {embedding.cudaData, mask.cudaData};
        for (auto &pair : context->kv) for (Data *cache : {&pair.first, &pair.second}) state.inputs.push_back(cache->cudaData);
        state.layers.resize(draftLayers); state.contextLayers.resize(draftLayers);
    }
    // A stable device table decouples graph lifetime from request KV lifetime.
    // Graph-capable attention reads each row's live length and KV pointers on device.
    // Scalar attention with embedded request addresses is rejected before capture.
    bool dynamicKV = true;
    std::vector<int> rankRows;
    for (int rank = 0; rank < ranks; ++rank) {
        rankRows.push_back(rankRequests[rank].size());
        if (!rankRows.back()) continue;
        FastllmCudaSetDevice(devices[rank]);
        dynamicKV = FastllmCudaNaiveDraftCacheBindingsSupported(rankRows.back(), draftBlock,
            heads, kvHeads, draftHeadDim, draftWindow) && dynamicKV;
    }
    FastllmCudaSetDevice(device);
    std::vector<void *> inputs{embedding.cudaData, mask.cudaData};
    std::shared_ptr<DraftWorkspace> state;
    if (GetFastllmEnv().cudaGraph) {
        auto found = std::find_if(draftBatchWorkspaces.begin(), draftBatchWorkspaces.end(),
            [&](const std::shared_ptr<DraftWorkspace> &entry) {
                return entry->device == device && entry->inputs == inputs && entry->lowLatencyBody &&
                    entry->rankRows == rankRows;
            });
        if (found != draftBatchWorkspaces.end()) {
            state = *found; draftBatchWorkspaces.erase(found);
        }
    }
    if (!state) {
        state = std::make_shared<DraftWorkspace>();
        state->device = device; state->inputs = std::move(inputs); state->lowLatencyBody = true;
        // Dynamic KV bindings may reuse a graph across owners, but each rank's
        // row count still determines its captured projection/attention shapes.
        state->rankRows = std::move(rankRows);
        if (draftTP) for (int gpu : devices) {
            auto local = std::make_shared<DraftWorkspace>(); local->device = gpu;
            FastllmCudaSetDevice(gpu); local->done = FastllmCudaEventCreate();
            state->tpRanks.push_back(std::move(local));
        }
        FastllmCudaSetDevice(device);
    }
    if (GetFastllmEnv().cudaGraph) {
        draftBatchWorkspaces.push_back(state);
        // Keep a small eager cache as well: unsupported ragged shapes must not
        // repeat GEMM preparation on every round. Never evict a prepared graph.
        if (!preparingGraphs) {
            auto eager = [](const auto &entry) {
                return !entry->graphs[0].exec && std::none_of(entry->tpRanks.begin(), entry->tpRanks.end(),
                    [](const auto &rank) { return rank->graphs[0].exec != nullptr; });
            };
            if (std::count_if(draftBatchWorkspaces.begin(), draftBatchWorkspaces.end(), eager) > 8)
                draftBatchWorkspaces.erase(std::find_if(draftBatchWorkspaces.begin(), draftBatchWorkspaces.end(), eager));
        }
    }
    auto localState = [&](int rank) -> DraftWorkspace & { return draftTP ? *state->tpRanks[rank] : *state; };
    if (!draftBatchStorage || draftBatchStorage->device != device) {
        draftBatchStorage = std::make_shared<DraftWorkspace>();
        draftBatchStorage->device = device;
        for (int gpu : devices) {
            auto scratch = std::make_shared<DraftWorkspace>(); scratch->device = gpu;
            draftBatchStorage->tpRanks.push_back(std::move(scratch));
        }
    }
    auto scratch = [&](int rank) -> DraftWorkspace & { return *draftBatchStorage->tpRanks.at(rank); };
    // Install map entries on the caller before rank workers read them.
    for (int gpu : devices) draftLinearGraphWorkspaces[gpu];
    auto run = [&](const std::function<void(int)> &fn) {
        if (draftTP) draftTP->Run(fn); else fn(0);
    };
    std::vector<int> lengths(contexts.size());
    for (size_t row = 0; row < contexts.size(); ++row) lengths[row] = contexts[row]->committed + 1;
    auto upload = [&](int rank) {
        auto &r = scratch(rank);
        const auto &requests = rankRequests[rank];
        if (requests.empty()) return;
        std::vector<int> localAnchors, localLengths;
        for (int row : requests) {
            localAnchors.push_back(anchors[row]);
            localLengths.push_back(lengths[row]);
        }
        r.UploadInputs(localAnchors, localLengths, maxBatch > 0 ? maxBatch : 512);
        std::vector<const void *> bindings;
        for (int layer = 0; layer < draftLayers; ++layer)
        for (int row : requests) {
            auto &pair = cacheAt(*contexts[row], rank, layer);
            bindings.push_back(pair.first.cudaData); bindings.push_back(pair.second.cudaData);
        }
        r.cacheTable.dataType = INT32; r.cacheTable.UpdateUnitSize();
        r.cacheTable.Resize({(int)(bindings.size() * sizeof(void *) / sizeof(int))});
        r.cacheTable.ToDevice(DataDevice::CUDA, {devices[rank]}, false); r.cacheTable.Allocate(false);
        if (bindings != r.uploadedCacheBindings || r.cacheTable.cudaData != r.uploadedCacheTable) {
            FastllmCudaCopyFromHostToDevice(r.cacheTable.cudaData, bindings.data(), r.cacheTable.GetBytes());
            r.uploadedCacheBindings = bindings; r.uploadedCacheTable = r.cacheTable.cudaData;
        }

        };
    std::vector<int> rebindable(ranks, 1);
    bool fusedBackbone = false;
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    // Capability dispatch precedes every collective. All ranks either use
    // this path or the original one, including unsupported batch shapes.
    const DraftBodyShape shape{divisor, heads, kvHeads, draftHeadDim, draftBlock, embed_dim, draftLayers, draftWindow};
    if (draftTP && !dataParallel && !draftBatchStorage->fusedUnavailable &&
        CudaDraftBodySupportsShape(ranks, shape, contexts.size())) {
        if (!draftBatchStorage->fused) {
            std::vector<DraftBodyWeights> layers;
            layers.reserve(ranks * draftLayers);
            for (int gpu : devices) {
                for (int layer = 0; layer < draftLayers; ++layer) {
                    const std::string p = "dspark.layers." + std::to_string(layer) + ".";
                    auto tensor = [&](const char *name) -> const Data * {
                        auto it = weight.weight.find(p + name);
                        if (it == weight.weight.end()) return nullptr;
                        auto shard = it->second.multiDeviceDatas.find(gpu);
                        return shard == it->second.multiDeviceDatas.end() ? nullptr : shard->second;
                    };
                    layers.push_back({tensor("self_attn.mergeqkv.weight"), tensor("self_attn.o_proj.weight"),
                        tensor("input_layernorm.weight"), tensor("self_attn.q_norm.weight"),
                        tensor("self_attn.k_norm.weight"), tensor("mlp.gateup_proj.weight"),
                        tensor("mlp.down_proj.weight"), tensor("post_attention_layernorm.weight")});
                }
            }
            draftBatchStorage->fused = CreateCudaDraftBody(devices, shape, layers, draftEps, draftTheta);
            draftBatchStorage->fusedUnavailable = !draftBatchStorage->fused;
            if (draftBatchStorage->fused && RankDraftFeatures()) {
                auto &fc = weight["dspark.fc.weight"], &norm = weight["dspark.hidden_norm.weight"];
                draftBatchStorage->fusedContextReady = PrepareCudaDraftContext(draftBatchStorage->fused, device, fc, norm);
            }
            FastllmCudaSetDevice(device);
        }
        if (draftBatchStorage->fused) {
            // Reserve the configured serving capacity on first use, including the
            // single-request bootstrap that can precede serving graph preparation.
            fusedBackbone = FastllmDraftFusedPrepare(draftBatchStorage->fused,
                CudaDraftBodyCapacity(contexts.size(), maxBatch));
        }
        if (!fusedBackbone && draftBatchStorage->fused) {
            // Existing graphs borrow this domain. Retire them before freeing
            // a partially grown workspace and selecting the original path.
            run([&](int rank) { ForceDeviceSync(); });
            for (auto &entry : draftBatchWorkspaces) {
                entry->graphs[0].Clear();
                for (auto &local : entry->tpRanks) local->graphs[0].Clear();
            }
            FastllmDraftFusedDestroy(draftBatchStorage->fused);
            draftBatchStorage->fused = nullptr;
            draftBatchStorage->fusedContextReady = false;
            draftBatchStorage->fusedHeadReady = false;
            draftBatchStorage->fusedHeadUnavailable = true;
            draftBatchStorage->fusedContexts.clear();
            draftBatchStorage->fusedUnavailable = true;
            FastllmCudaSetDevice(device);
        }
    }
#endif
    auto body = [&](int rank, bool reduce = true) {
        const auto &selected = rankRequests[rank];
        if (selected.empty()) return;
        std::vector<DraftContext *> localContexts;
        for (int i : selected) localContexts.push_back(contexts[i]);
#ifndef USE_ROCM
        FastllmCudaLinearGraphScope linearScope(draftLinearGraphWorkspaces.at(devices[rank]));
#endif
        auto w = [&](const std::string &name) -> Data & { return weightAt(name, rank); };
        auto linear = [&](Data &input, Data &weight, Data &output) {
            // TP uses the existing cached Lt/GEMV dispatch. The strided-GEMM
            // API can lazily change CUDA kernel attributes between collectives.
            if (draftTP) Linear(input, weight, Data(), output);
            else DraftLinear(input, weight, output);
        };
        auto allReduce = [&](Data &value, Data &reduced) {
            if (!draftTP || dataParallel) return;
            // Keep both typed allocations stable across graph variants.
            // In-place conversion replaces the shared BF16 pointer during
            // capture/replay transitions and defeats workspace reuse.
            ToDataType(value, reduced, FLOAT32);
            if (reduce) {
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
                if (fusedBackbone && !CudaDraftUseBody(contexts.size()) &&
                    FastllmDraftFusedReduceSupported(draftBatchStorage->fused,contexts.size())) {
                    AssertInFastLLM(FastllmDraftFusedReduce(draftBatchStorage->fused,rank,contexts.size(),
                        reduced.cudaData,reduced.Count(0),nullptr),"Batched draft reduction failed.");
                } else
#endif
                    FastllmCudaNaiveDraftTPReduce(draftTP->group, rank, reduced);
            }
            ToDataType(reduced, value, BFLOAT16);
        };
        auto &r = scratch(rank);
        if (r.layers.empty()) r.layers.resize(draftLayers);
        Data &ids = r.id, &live = r.live, &result = r.normalized;
        auto addNorm = [&](Data &hidden, Data &branch, Data &gamma, Data &output) {
            gamma.ToDevice(DataDevice::CUDA, std::vector<int>{devices[rank]});
            if (hidden.dataType == BFLOAT16 && branch.dataType == BFLOAT16 &&
                    gamma.dataType == FLOAT32 && gamma.Count(0) == hidden.dims.back()) {
                FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, gamma, draftEps, output);
            } else {
                AddTo(hidden, branch);
                KimiK3RMSNorm(hidden, gamma, draftEps, output);
            }
        };
        Data &embedding = w("model.embed_tokens.weight"), &mask = w("dspark.mask_embedding");
        auto append = [](Data &batch, Data &part, bool first) {
            if (first) Copy(part, batch);
            else { Data joined; Cat(batch, part, 1, joined); Copy(joined, batch); }
        };
        Data &hidden = r.hidden, &positions = r.positions, &normed = r.combined;
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
        if (fusedBackbone && CudaDraftUseBody(contexts.size())) {
            // Configure/pack has already completed on every GPU. The legacy
            // no-collective warmup must not enter a fused peer wait.
            if (!reduce) return;
            RunPreparedCudaDraftBody(draftBatchStorage->fused, rank, devices[rank], localContexts.size(),
                draftBlock, ids, embedding, mask, live, r.cacheTable, w("dspark.norm.weight"), draftEps,
                hidden, positions, result);
            return;
        }
#endif
        struct Request { Data positions, live; };
        std::vector<Request> requests(localContexts.size());
        FastllmCudaNaiveDraftInput(ids, embedding, mask, live, draftBlock, hidden, positions);
        for (size_t row = 0; row < localContexts.size(); ++row) {
            requests[row].positions.Resize({1, draftBlock});
            requests[row].positions.FakeFrom(positions, row * draftBlock * sizeof(float));
            requests[row].live.Resize({1}); requests[row].live.FakeFrom(live, row * sizeof(int));
        }
        KimiK3RMSNorm(hidden, w("dspark.layers.0.input_layernorm.weight"), draftEps, normed);
        for (int layer = 0; layer < draftLayers; ++layer) {
            const std::string p = "dspark.layers." + std::to_string(layer) + ".";
            // Layers execute sequentially on this stream; reuse their matrix scratch.
            auto &l = r.layers.front();
            Data &qkv = l.qkv, &qs = l.q, &ks = l.k, &vs = l.v, &attention = l.attention;
            Data &output = l.output;
            const bool merged = weight.weight.count(p + "self_attn.mergeqkv.weight");
            std::vector<FastllmNaiveDraftRow> descriptors;
            for (size_t row = 0; row < localContexts.size(); ++row) {
                auto *context = localContexts[row];
                auto &cache = cacheAt(*context, rank, layer);
                descriptors.push_back({&cache.first, &cache.second, &requests[row].positions,
                    &requests[row].live, std::min(context->committed, draftWindow - 1) + draftBlock <= 256,
                    (const void *const *)r.cacheTable.cudaData + ((size_t)layer * localContexts.size() + row) * 2});
            }
            Data &packed = l.packed, &query = l.query, &scores = l.scores;
            Data bindings(INT32, {(int)localContexts.size(), 4});
            if (dynamicKV) bindings.FakeFrom(r.cacheTable, (size_t)layer * localContexts.size() * 2 * sizeof(uint64_t));
            const DraftAttentionWeights weights{merged ? &w(p + "self_attn.mergeqkv.weight") : nullptr,
                w(p + "self_attn.o_proj.weight"),
                w(p + "self_attn.q_norm.weight"), w(p + "self_attn.k_norm.weight"), w(p + "post_attention_layernorm.weight")};
            if (!TryCudaDraftAttention({devices, rank, divisor},
                    {heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta}, descriptors,
                    dynamicKV ? &bindings : nullptr, normed, weights, hidden, qkv, attention, output, r.layers.front().scores)) {
                if (merged) linear(normed, *weights.qkv, qkv);
                else {
                    linear(normed, w(p + "self_attn.q_proj.weight"), qs);
                    linear(normed, w(p + "self_attn.k_proj.weight"), ks);
                    linear(normed, w(p + "self_attn.v_proj.weight"), vs);
                    Data qk;
                    Cat(qs, ks, -1, qk); Cat(qk, vs, -1, packed);
                }
                const bool batchedAttention = FastllmCudaNaiveBatchDraftQKV(merged ? qkv : packed,
                        weights.qNorm, weights.kNorm,
                        descriptors, query, heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta,
                        &localState(rank).batchDescriptors[layer], dynamicKV ? &bindings : nullptr) &&
                    FastllmCudaNaiveBatchDraftAttention(query, descriptors, heads, kvHeads,
                        draftHeadDim, draftWindow, scores, attention, dynamicKV ? &bindings : nullptr);
                if (!batchedAttention) {
                    // This fallback embeds request KV addresses. It may execute
                    // eagerly, but must never enter a request-independent graph.
                    rebindable[rank] = 0;
                    for (size_t row = 0; row < localContexts.size(); ++row) {
                        auto &r = requests[row]; auto &cache = cacheAt(*localContexts[row], rank, layer);
                        Data raw, q, k, v, scores, part;
                        const int begin = row * draftBlock, end = begin + draftBlock;
                        if (merged) Split(qkv, 1, begin, end, raw);
                        if (!merged || !FastllmCudaNaiveDraftQKV(raw,
                                weights.qNorm, weights.kNorm,
                                r.positions, r.live, cache.first, cache.second, q,
                                heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta)) {
                            if (merged) {
                                const int qw = heads * draftHeadDim, kw = kvHeads * draftHeadDim;
                                Split(raw, -1, 0, qw, q); Split(raw, -1, qw, qw + kw, k);
                                Split(raw, -1, qw + kw, qw + 2 * kw, v);
                            } else {
                                Split(qs, 1, begin, end, q); Split(ks, 1, begin, end, k); Split(vs, 1, begin, end, v);
                            }
                            q.Reshape({1, draftBlock * heads, draftHeadDim});
                            k.Reshape({1, draftBlock * kvHeads, draftHeadDim});
                            KimiK3RMSNorm(q, weights.qNorm, draftEps, q);
                            KimiK3RMSNorm(k, weights.kNorm, draftEps, k);
                            q.Reshape({1, draftBlock, heads * draftHeadDim});
                            k.Reshape({1, draftBlock, kvHeads * draftHeadDim});
                            FastllmCudaNaiveRope(q, r.positions, heads, draftHeadDim, draftHeadDim, draftTheta);
                            FastllmCudaNaiveRope(k, r.positions, kvHeads, draftHeadDim, draftHeadDim, draftTheta);
                            FastllmCudaNaiveAppendVerifyCache(cache.first, cache.second, k, v, r.live, draftWindow);
                        }
                        const bool shortAttention = std::min(localContexts[row]->committed, draftWindow - 1) + draftBlock <= 256;
                        FastllmCudaNaiveDraftAttention(q, cache.first, cache.second, r.live,
                            heads, kvHeads, draftHeadDim, draftWindow, shortAttention, scores, part);
                        append(attention, part, row == 0);
                    }
                }
                linear(attention, weights.o, output);
                if (draftTP && draftTP->attentionDivisor > 1) allReduce(output, l.reduced);
                addNorm(hidden, output, weights.postNorm, normed);
            }
            Data &nextGamma = w(layer + 1 == draftLayers ? "dspark.norm.weight" :
                "dspark.layers." + std::to_string(layer + 1) + ".input_layernorm.weight");
            Data &nextInput = layer + 1 == draftLayers ? result : normed;
            auto finish = [&] {
                allReduce(l.output, l.reduced);
                addNorm(hidden, l.output, nextGamma, nextInput);
            };
            RunDenseDraftMlp({devices, rank, dataParallel ? 1 : ranks}, int(localContexts.size()),
                weight.weight.count(p + "mlp.gateup_proj.weight"), p, w, linear, finish,
                normed, hidden, nextGamma, draftEps, l, nextInput);
        }

    };
    for (int rank = 0; rank < ranks; ++rank) localState(rank).batchDescriptors.resize(draftLayers);
    std::vector<std::vector<void *>> before;
    for (int rank = 0; rank < ranks; ++rank) before.push_back(scratch(rank).Pointers());
    auto invalidateMoved = [&]() {
        bool changed = false;
        for (int rank = 0; rank < ranks; ++rank) changed = changed || before[rank] != scratch(rank).Pointers();
        if (!changed) return;
        for (auto &other : draftBatchWorkspaces) if (other != state) {
            other->graphs[0].Clear();
            for (auto &rank : other->tpRanks) rank->graphs[0].Clear();
        }
    };
    const bool uploaded = preparingGraphs || !state->warmed;
    if (uploaded) run(upload);
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (fusedBackbone) draftBatchStorage->fusedContexts.assign(contexts.begin(), contexts.end());
#endif
    if (draftTP && !state->warmed) {
        // Initialize every GEMM plan/kernel before any rank enters NCCL.
        // Only speculative scratch is written; committed KV is unchanged.
        // A CUDA lazy-load/attribute update while another rank is already
        // waiting in a collective can otherwise deadlock the driver.
        run([&](int rank) { body(rank, false); ForceDeviceSync(); });
        state->warmed = true;
    }
    bool useGraph = GetFastllmEnv().cudaGraph && !state->disabled;
    if (useGraph && !localState(0).graphs[0].exec && !draftCapturing) useGraph = false;
    if (useGraph && !localState(0).graphs[0].exec) {
        run([&](int rank) { body(rank); ForceDeviceSync(); });
        invalidateMoved();
        if (std::any_of(rebindable.begin(), rebindable.end(), [](int value) { return !value; })) {
            state->disabled = true; useGraph = false;
        } else if (!draftTP) useGraph = state->graphs[0].Capture([&] { body(0); }, state->disabled);
        else {
            std::vector<int> ready(ranks);
            auto allReady = [&] { return std::all_of(ready.begin(), ready.end(), [](int n) { return n != 0; }); };
            run([&](int rank) { ready[rank] = FastllmCudaGraphPrepareCaptureDevice(); });
            bool pool = allReady() && FastllmCudaGraphMemoryPoolBegin(), ok = pool;
            if (pool) {
                run([&](int rank) { FastllmCudaClearThreadError(); ready[rank] = FastllmCudaGraphBeginCapture(); });
                ok = allReady();
                run([&](int rank) {
                    if (!ready[rank]) return;
                    if (ok) body(rank);
                    ready[rank] = FastllmCudaGraphEndCapture(&localState(rank).graphs[0].graph) && !FastllmCudaGetThreadError();
                });
                ok = FastllmCudaGraphMemoryPoolEnd(state->graphs[0].reserved) && ok && allReady();
                if (ok) {
                    run([&](int rank) { auto &g = localState(rank).graphs[0]; ready[rank] = FastllmCudaGraphInstantiate(g.graph, &g.exec); });
                    ok = allReady();
                }
            }
            if (!ok) {
                run([&](int rank) { localState(rank).graphs[0].Clear(); FastllmCudaClearLastError(); FastllmCudaClearThreadError(); });
                state->graphs[0].Clear(); state->disabled = true; useGraph = false;
            }
        }
    }
    // Capture can also materialize operator-owned buffers after warmup.
    // Retire other variants before replay if any shared address moved.
    invalidateMoved();
    run([&](int rank) {
        if (!uploaded) upload(rank);
        auto &r = localState(rank);
        if (useGraph) AssertInFastLLM(FastllmCudaGraphLaunch(r.graphs[0].exec), "Naive batch draft graph launch failed.");
        else body(rank);
        if (draftTP) FastllmCudaEventRecordCurrentThread(r.done);
    });
    if (draftTP) for (int rank = 1; rank < ranks; ++rank)
        FastllmCudaCurrentThreadStreamWaitEvent(localState(rank).done);
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (draftTP) for (int rank=0; rank<ranks; ++rank) scratch(rank).backboneReady=localState(rank).done;
#endif
    if (!useGraph) invalidateMoved();
    if (dataParallel) {
        normalized.dataType = BFLOAT16; normalized.UpdateUnitSize();
        normalized.Resize({1, batch * draftBlock, embed_dim});
        normalized.ToDevice(DataDevice::CUDA, {device}, false); normalized.Allocate(false);
        std::vector<void *> destinations;
        std::vector<const void *> sources;
        std::vector<size_t> sizes;
        const size_t width = size_t(draftBlock) * embed_dim;
        for (int rank = 0; rank < ranks; ++rank) {
            const auto &requests = rankRequests[rank];
            for (size_t row = 0; row < requests.size(); ++row) {
                destinations.push_back((uint16_t *)normalized.cudaData + size_t(requests[row]) * width);
                sources.push_back((const uint16_t *)scratch(rank).normalized.cudaData + row * width);
                sizes.push_back(width * sizeof(uint16_t));
            }
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(destinations.data(),
            sources.data(), sizes.data(), sizes.size()), "Draft DP output gathering failed.");
        // Workers may reuse their scratch on the next round. Complete the
        // root-stream peer reads before returning ownership to the scheduler.
        FastllmCudaSyncCurrentThreadStream();
    } else {
        // Graph replay updates device values, not Data's host shape. Publish the
        // actual dense result on every rank before a sharded downstream consumer.
        const std::vector<int> resultDims{1, (int)contexts.size() * draftBlock, embed_dim};
        for (int rank = 0; rank < ranks; ++rank)
            if (scratch(rank).normalized.dims != resultDims) scratch(rank).normalized.Resize(resultDims);
        Copy(scratch(0).normalized, normalized);
    }
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (!preparingGraphs && useGraph && fusedBackbone) draftBatchStorage->nextGraph = state;
#endif
    return true;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::PrepareFusedDraftProposal(
        const std::vector<DraftRound *> &rounds, const Data &hidden) {
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (attentionDP > 1) return false;
    const int count = rounds.size(), ranks = tpDevices.size();
    auto &head = weight["lm_head.weight"];
    if (!count || !ranks || !draftTP || draftTP->devices != tpDevices ||
        !draftBatchStorage || !draftBatchStorage->fused || draftBatchStorage->fusedHeadUnavailable ||
        draftBatchStorage->tpRanks.size() != tpDevices.size() || head.dims.empty() ||
        draftBatchStorage->fusedContexts.size() != rounds.size() ||
        !CudaDraftHeadSupportsShard(hidden, tpDevices.front(), ranks, count, draftBlock, embed_dim, head.dims[0]))
        return false;
    const int vocab = head.dims[0];
    for (int row = 0; row < count; ++row) {
        const auto &r = *rounds[row];
        if (draftBatchStorage->fusedContexts[row] != &r.context || r.limit < 1 || r.limit > draftBlock ||
            r.anchor < 0 || r.anchor >= vocab || r.config.IsSimpleGreedy() || r.config.output_token_least > 0 ||
            r.config.tool_call_grammar || r.config.tool_call_token_mask || !SelectLogits(r.config).cudaSampling ||
            !std::isfinite(r.config.temperature) || r.config.temperature <= 0) return false;
    }
    for (int rank = 0; rank < ranks; ++rank) {
        const auto &x = draftBatchStorage->tpRanks[rank]->normalized;
        if (!CudaDraftHeadSupportsShard(x, tpDevices[rank], ranks, count, draftBlock, embed_dim, vocab)) return false;
    }
    ApplyDraftDevice();
    const int owner = tpDevices[0];
    if (!draftBatchStorage->fusedHeadReady) {
        // Packing and the immutable bias table are startup work. Requests only
        // reuse this domain; unsupported geometry or memory keeps the old path.
        if (!preparingGraphs) return false;
        // Give the configured KV capacity first claim on device memory. The
        // early workspace warmup will retry after the physical pool exists.
        if (!GetKVCacheInCPU() && tpKVPool.empty() && !sparseKv) return false;
        auto &w1 = weight["dspark.markov_head.markov_w1.weight"], &w2 = weight["dspark.markov_head.markov_w2.weight"];
        auto &cw = weight["dspark.confidence_head.proj.weight"], &cb = weight["dspark.confidence_head.proj.bias"];
        const bool supported = PrepareCudaDraftShardedHead(draftBatchStorage->fused, tpDevices, tpVocabRanges,
            head, w1, w2, cw, cb, GetGpuMemRatio());
        draftBatchStorage->fusedHeadReady = supported;
        draftBatchStorage->fusedHeadUnavailable = !supported;
        if (!supported) { FastllmCudaSetDevice(owner); return false; }
        const int capacity = CudaDraftBodyCapacity(count, maxBatch > 0 ? maxBatch : 64);
        draftBatchStorage->pinnedHeadOutput = FastllmCudaHostMalloc(size_t(capacity) * draftBlock * (sizeof(int64_t) + sizeof(float)));
        draftBatchStorage->pinnedHeadCapacity = capacity;
        for (int rank = 0; rank < ranks; ++rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            auto &local = *draftBatchStorage->tpRanks[rank];
            if (!local.done) local.done = FastllmCudaEventCreate();
        }
        FastllmCudaSetDevice(owner);
    }
    if (!draftBatchStorage->pinnedHeadOutput || count > draftBatchStorage->pinnedHeadCapacity) return false;
    PrepareDraftSamplingMemory(count, draftBlock, vocab, weight["dspark.markov_head.markov_w1.weight"].dims.back());
    auto &probabilities = draftProposalStorage->sampling.probabilities;
    std::vector<uint64_t> seeds;
    for (auto *r : rounds) seeds.push_back(r->context.random());
    auto &draftStorage=*draftBatchStorage;
    const bool hasProposal=count==1 && rounds.front()->limit==draftBlock && draftStorage.nextInFlight &&
        draftStorage.nextConsumed && draftStorage.nextOwners.size()==1 &&
        draftStorage.nextOwners.front().lock().get()==&rounds.front()->context &&
        draftStorage.nextLengths.front()==rounds.front()->oldLength+1 &&
        draftStorage.nextAnchors.front()==rounds.front()->anchor &&
        draftStorage.predictedSeed==seeds.front() && draftStorage.predictedTemperature==rounds.front()->config.temperature &&
        draftStorage.nextProposed;
    if (count==1) draftStorage.lastHeadTemperature=rounds.front()->config.temperature;
    auto launchHead = [&]() { draftTP->Run([&](int rank) {
        auto &local = *draftBatchStorage->tpRanks[rank];
        if (local.backboneReady) FastllmCudaCurrentThreadStreamWaitEvent(local.backboneReady);
        if (!hasProposal) for (int row = 0; row < count; ++row)
            AssertInFastLLM(TryCudaDraftHead(DraftShardedHead{draftBatchStorage->fused, rank, row, rounds[row]->anchor,
                (const uint16_t*)local.normalized.cudaData + size_t(row) * draftBlock * embed_dim,
                seeds[row], rounds[row]->config.temperature}, rounds[row]->limit),
                "Fused draft proposal launch failed.");
        FastllmCudaEventRecordCurrentThread(local.done);
    }); };
    for (int row = 0; row < count; ++row) {
        auto &r = *rounds[row];
        r.vocab = vocab; r.greedy = false;
        r.selection = SelectLogits(r.config,true); r.selection.count = 0; r.selection.cudaLogits = true;
        r.proposalOwner = draftProposalStorage;
        r.proposalOffset = size_t(row) * draftBlock * vocab;
    }
    try {
    bool submitted=false;
    if (rounds.front()->enqueueVerification && HasVerificationGraph(count,draftTokens+1) &&
        std::all_of(rounds.begin(),rounds.end(),[&](const auto *r) { return r->limit == draftTokens; })) {
        std::vector<TargetCapture::DraftInput> inputs;
        bool supported = true;
        for (int rank = 0; rank < ranks; ++rank) for (int row = 0; row < count; ++row) {
            const auto *ids = FastllmDraftFusedHeadTokens(draftBatchStorage->fused,rank,row);
            supported = supported && ids;
            inputs.push_back({ids,draftBatchStorage->tpRanks[rank]->done,rounds[row]->oldLength,
                rounds[row]->anchor,seeds[row],rounds[row]->config.temperature,
                draftBatchStorage->tpRanks[rank]->normalized.cudaData,
                draftBatchStorage->tpRanks[rank]->backboneReady,draftBatchStorage->fused,rounds[row]->limit+1,
                hasProposal});
        }
        if (supported) {
            rounds.front()->enqueueVerification(inputs,launchHead); submitted=true;
        }
        ApplyDraftDevice();
    }
    if (!submitted) launchHead();
    // Composite graphs publish this event after their head/input prefix. It
    // covers rank zero too: its producer is now the target worker's stream.
    for (int rank = 0; rank < ranks; ++rank)
        FastllmCudaCurrentThreadStreamWaitEvent(draftBatchStorage->tpRanks[rank]->done);
    auto *allTokens = static_cast<int64_t *>(draftBatchStorage->pinnedHeadOutput);
    auto *allConfidence = reinterpret_cast<float *>(allTokens + draftBatchStorage->pinnedHeadCapacity * draftBlock);
    std::vector<float> temperatures;
    std::vector<int> limits;
    for (const auto *round : rounds) {
        temperatures.push_back(round->config.temperature);
        limits.push_back(round->limit);
    }
    AssertInFastLLM(ExportCudaDraftShardedHead(draftBatchStorage->fused, hidden, draftBlock, vocab,
        temperatures, limits, probabilities, allTokens, draftConfidenceThreshold > 0 ? allConfidence : nullptr),
        "Fused draft probability export failed.");
    // One host completion boundary per batch; each request retains its own
    // probabilities, confidence, seed and accepted prefix.
    FastllmCudaSyncCurrentThreadStream();
    for (int row = 0; row < count; ++row) {
        auto &r = *rounds[row];
        const int64_t *tokens = allTokens + row * draftBlock;
        const float *confidence = allConfidence + row * draftBlock;
        for (int step = 0; step < r.limit; ++step) {
            if (step && draftConfidenceThreshold > 0 &&
                1.f/(1.f+std::exp(-confidence[step])) < draftConfidenceThreshold) break;
            AssertInFastLLM(tokens[step] >= 0 && tokens[step] < vocab,"Invalid fused draft token.");
            r.proposed.push_back(tokens[step]);
        }
    }
    return true;
    } catch (...) {
        // A submitted target borrows request caches and graph outputs. Drain
        // both producers before any caller can retire them on an error path.
        if (rounds.front()->earlyCapture) {
            try {
                std::vector<std::exception_ptr> errors(tpDevices.size());
                tpWorkers.Run(tpDevices,[&](int rank) {
                    FastllmCudaSetDevice(tpDevices[rank]); FastllmCudaSyncCurrentThreadStream();
                },errors);
                ApplyDraftDevice(); FastllmCudaSyncCurrentThreadStream();
            } catch (...) {}
        }
        throw;
    }
#else
    return false;
#endif
}

bool NaiveN05FlashModel::TryDraftHeadProposal(DraftRound &round) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
    if (!draftTP || draftTP->dataParallel || draftTP->devices != tpDevices ||
        draftExperts || draftConfidenceThreshold != 0 || !GetCudaEmbedding() || !GetFastllmEnv().cudaGraph ||
        GetLowMemMode() || round.config.output_token_least > 0 || round.config.tool_call_grammar ||
        round.config.tool_call_token_mask || !SupportsCudaRejectionSampling(round.config) || round.config.IsSimpleGreedy())
        return false;
    ApplyDraftDevice();
    if (!draftProposalStorage) {
        if (!preparingGraphs) return false;
        draftProposalStorage = std::make_shared<DraftWorkspace>();
        draftProposalStorage->device = FastllmCudaGetDevice();
    }
    if (!FastllmCudaProposeDraft(tpDevices, tpVocabRanges, weight, draftProposalStorage->proposalHead,
            embed_dim, draftBlock, round.limit, round.anchor, round.config.temperature, preparingGraphs,
            round.context.continuationSerial, round.context.committed, round.vocab, round.proposed,
            [&](int steps, int vocab, int latent) {
                PrepareDraftSamplingMemory(1, steps, vocab, latent);
                return static_cast<float *>(draftProposalStorage->sampling.probabilities.cudaData);
            }, [&] { return draftBatchStorage && maxBatch == 1 ? &draftBatchStorage->replay : nullptr; },
            [&](const FastllmDraftHeadCall &call) {
                Data unused;
                return RunDraftBatch({round.anchor}, {&round.context}, unused, 0, nullptr, &call);
            })) return false;
    round.greedy = false;
    round.selection = SelectLogits(round.config, true);
    round.selection.count = 0;
    round.selection.cudaLogits = true;
    round.proposalOwner = draftProposalStorage;
    round.proposalOffset = 0;
    return true;
#else
    return false;
#endif
}

void NaiveN05FlashModel::PrepareDraftServingMemory(bool history) {
#ifdef USE_CUDA
    if (draftEnabled && PrepareDraftTP()) (this->*DraftOps().warm)(history);
#endif
}

void NaiveN05FlashModel::PrepareLayerDraftServingMemory(bool history) {
#ifdef USE_CUDA
    const int batch = tpDevices.size() == 1 ? 1 : (maxBatch > 0 ? maxBatch : 512);
    const int steps = std::max(1, std::min(draftTokens, draftBlock)), verifyRows = steps + 1;
    ApplyDraftDevice();
    Data hidden(BFLOAT16, {1, draftWindow - 1, embed_dim});
    hidden.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); hidden.Allocate(true);
    std::vector<std::shared_ptr<DraftContext>> owners;
    std::vector<DraftContext *> contexts;
    for (int i = 0; i < batch; ++i) {
        auto context = CreateDraftContext();
        AppendDraftContext(hidden, 0, *context);
        if (history) {
            auto &tail = context->workspace->prefixStorage;
            tail.dataType = BFLOAT16; tail.Resize({1, draftWindow + GetPageLen() - 1, embed_dim});
            tail.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); tail.Allocate(false);
        }
        contexts.push_back(context.get()); owners.push_back(std::move(context));
    }
    auto warmPrefillCommit = [&]() {
        TargetCapture capture;
        for (int layer : draftTargetLayers) {
            Data &feature = capture.hidden[layer];
            feature.dataType = BFLOAT16; feature.Resize({1, draftWindow + GetPageLen() - 1, embed_dim});
            feature.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); feature.Allocate(true);
        }
        std::vector<std::pair<Data, Data>> unused;
        CommitDraftContext(capture, draftWindow + GetPageLen() - 1, *contexts.front(), unused);
    };
    if (pdPrefillOnly) {
        warmPrefillCommit();
        if (history && draftTP->dataParallel && draftContextStorage && !GetLowMemMode() &&
            FastllmCudaPeerAccessInit(draftTP->devices) && weight["dspark.fc.weight"].dataType == BFLOAT16 &&
            weight["dspark.fc.weight"].IsTensorParallelReplicated() &&
            weight["dspark.hidden_norm.weight"].IsTensorParallelReplicated()) {
            const int keep = draftWindow + GetPageLen() - 1;
            draftTP->Run([&](int rank) {
                auto &r = *draftContextStorage->tpRanks[rank];
                r.combined.dataType = BFLOAT16; r.combined.UpdateUnitSize();
                r.combined.Resize({1, keep, int(draftTargetLayers.size()) * embed_dim});
                r.combined.ToDevice(CUDA, {r.device}, false); r.combined.Allocate(true);
                DraftLinear(r.combined, *weight["dspark.fc.weight"].multiDeviceDatas.at(r.device), r.projected);
                KimiK3RMSNorm(r.projected, *weight["dspark.hidden_norm.weight"].multiDeviceDatas.at(r.device),
                              draftEps, r.committedHidden);
                FastllmCudaSyncCurrentThreadStream();
            });
        }
        idleDraftContexts = std::move(owners);
        return;
    }
    const int ranks = draftTP->devices.size();
    const int localLimit = draftTP->dataParallel ? WindowRequestsPerGroup(batch, ranks) : 0;
    const auto buckets = localLimit && GetFastllmEnv().cudaGraph ? SpeculativeGraphBuckets(localLimit) : std::vector<int>{};
    bool backboneReady = false;
    const int vocab = weight["lm_head.weight"].dims[0];
    Data head;
    std::vector<GenerationConfig> configs(batch);
    std::vector<std::unique_ptr<DraftRound>> rounds;
    for (int i = 0; i < batch; ++i) {
        configs[i].temperature = 1.0f; configs[i].top_p = 0.95f;
        configs[i].top_k = vocab;
        rounds.emplace_back(new DraftRound(*this, *contexts[i], configs[i], 1, steps));
    }
    if (PrepareDraftBatch(rounds, head, localLimit, &backboneReady)) {
        tpSamplingLogits.Resize({batch * verifyRows, vocab});
        tpSamplingLogits.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false);
        tpSamplingLogits.Allocate(false);
        FastllmCudaMemset0(tpSamplingLogits.cudaData, tpSamplingLogits.GetBytes());
        PrepareDraftAcceptance(rounds, tpSamplingLogits);
    }
    if (backboneReady && (!draftTP->dataParallel || ranks == 1) && GetFastllmEnv().cudaGraph &&
        head.dims == std::vector<int>{1, batch * steps, vocab} && !draftConfidenceThreshold) {
        // The standalone path stages its logits into this graph's stable buffer.
        Data first;
        first.FakeFrom(head, 0); first.Resize({1, steps, vocab});
        DraftRound single(*this, *contexts.front(), configs.front(), 1, steps);
        PrepareLowLatencyProposalGraph({&single}, first, nullptr);
    }
    if (backboneReady) {
        if (!buckets.empty() && !draftConfidenceThreshold && !draftSamplingRanks.empty()) {
            // Materialize each owner's stable proposal graphs while the weights
            // and pools are warm. Subsequent admission only uploads live inputs.
            std::vector<std::vector<DraftRound *>> groups(ranks);
            for (auto &round : rounds) groups.at(round->context.deviceRank).push_back(round.get());
            draftTP->Run([&](int rank) {
                auto &rows = groups[rank];
                // A later request can land on any owner even with max_batch=1.
                // Proposal capture only needs a sampler config and a dummy anchor.
                DraftRound warm(*this, *contexts.front(), configs.front(), 1, steps);
                if (rows.empty()) rows.push_back(&warm);
                auto &storage = *draftSamplingRanks[rank];
                for (int bucket : buckets) {
                    const int n = std::min<int>(rows.size(), bucket);
                    std::vector<DraftRound *> selected(rows.begin(), rows.begin() + n);
                    Data staged;
                    staged.FakeFrom(storage.samplingInput, 0); staged.Resize({1, n * steps, vocab});
                    FastllmCudaMemset0(staged.cudaData, staged.GetBytes());
                    PrepareDraftBatchGraph(selected, staged, rank, bucket);
                    PrepareDraftBatchGraph(selected, staged, rank, bucket, true);
                }
            });
            ApplyDraftDevice();
        }
        TargetCapture capture;
        capture.verifying = true;
        for (int layer : draftTargetLayers) {
            Data &feature = capture.hidden[layer];
            feature.dataType = BFLOAT16; feature.Resize({1, batch * verifyRows, embed_dim});
            feature.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); feature.Allocate(true);
        }
        CommitDraftContexts(capture, std::vector<int>(batch, verifyRows), std::vector<int>(batch, verifyRows), contexts);
        capture.hidden.clear();
        warmPrefillCommit();
    }
    if (!buckets.empty()) {
        std::vector<std::vector<DraftContext *>> groups(ranks);
        for (auto *context : contexts) groups.at(context->deviceRank).push_back(context);
        // A single live request can move to any attention owner. Capture the
        // missing owner buckets at startup using transient KV. Replays bind
        // their own request pointers through the existing device table.
        if (!draftExperts && backboneReady && draftBatchStorage &&
            std::any_of(draftBatchStorage->tpRanks.begin(), draftBatchStorage->tpRanks.end(),
                [](const auto &rank) { return !rank->batchGraphs.empty(); })) {
            for (int rank = 0; rank < ranks; ++rank) {
                if (!groups[rank].empty()) continue;
                DraftContext warm;
                warm.deviceRank = rank;
                Data token;
                token.FakeFrom(hidden, 0); token.Resize({1, 1, embed_dim});
                AppendDraftContext(token, 0, warm);
                Data result;
                for (auto it = buckets.rbegin(); it != buckets.rend(); ++it)
                    RunDraftBatch({1}, {&warm}, result, *it);
            }
        }
        for (int bound : buckets) {
            if (backboneReady && bound == localLimit) break; // Already warmed by the full backbone run.
            std::vector<DraftContext *> sample;
            for (const auto &group : groups)
                sample.insert(sample.end(), group.begin(), group.begin() + std::min<size_t>(bound, group.size()));
            if (!sample.empty()) {
                Data result;
                RunDraftBatch(std::vector<int>(sample.size(), 1), sample, result);
            }
        }
    }
    idleDraftContexts = std::move(owners);
#endif
}

void NaiveN05FlashModel::PrepareLowLatencyDraftServingMemory(bool history) {
#ifdef USE_CUDA
    const int batch = maxBatch > 0 ? maxBatch : 512;
    if (!GetCudaEmbedding() || GetLowMemMode() ||
        draftBlock < 2 || draftBlock >= 32 || draftHeadDim % 4 || draftHeadDim > 256) return;
    const int owner = draftTP->devices.front();
    const int steps = std::max(1, std::min(draftTokens, draftBlock)), verifyRows = steps + 1;
    ApplyDraftDevice();
    Data hidden(BFLOAT16, {1, draftWindow - 1, embed_dim});
    hidden.ToDevice(DataDevice::CUDA, {owner}, false); hidden.Allocate(true);
    std::vector<std::shared_ptr<DraftContext>> owners;
    std::vector<DraftContext *> contexts;
    std::vector<int> anchors(batch, 1);
    for (int i = 0; i < batch; ++i) {
        auto context = CreateDraftContext();
        AppendDraftContext(hidden, 0, *context);
        contexts.push_back(context.get()); owners.push_back(std::move(context));
    }
    Data normalized;
    if (RunDraftBatch(anchors, contexts, normalized)) {
        // The backbone always keeps its trained block. Only the proposed
        // prefix and its verification distributions need vocabulary buffers.
        Data headInput;
        GatherDraftHeadInput(normalized, std::vector<int>(batch, steps), draftBlock, headInput);
        Data head = RunDraftHead(headInput);
        std::vector<GenerationConfig> configs(batch);
        std::vector<std::unique_ptr<DraftRound>> rounds;
        for (int i = 0; i < batch; ++i) {
            configs[i].temperature = 1.0f; configs[i].top_p = 0.95f;
            configs[i].top_k = head.dims.back();
            rounds.emplace_back(new DraftRound(*this, *contexts[i], configs[i], 1, steps));
        }
        std::vector<DraftRound *> fusedRounds;
        for (auto &round : rounds) fusedRounds.push_back(round.get());
        TryDraftProposal(fusedRounds, normalized);
        for (auto &round : rounds) round->proposed.clear();
        if (PrepareLowLatencyProposalGraph(fusedRounds, head, &normalized)) {
            tpSamplingLogits.Resize({batch * verifyRows, head.dims.back()});
            tpSamplingLogits.ToDevice(DataDevice::CUDA, {owner}, false);
            tpSamplingLogits.Allocate(false);
            FastllmCudaMemset0(tpSamplingLogits.cudaData, tpSamplingLogits.GetBytes());
            PrepareDraftAcceptance(rounds, tpSamplingLogits);
        }
        TargetCapture capture;
        capture.verifying = true;
        for (int layer : draftTargetLayers) {
            Data &feature = capture.hidden[layer];
            feature.dataType = BFLOAT16; feature.Resize({1, batch * verifyRows, embed_dim});
            feature.ToDevice(DataDevice::CUDA, {owner}, false); feature.Allocate(true);
        }
        CommitDraftContexts(capture, std::vector<int>(batch, verifyRows), std::vector<int>(batch, verifyRows), contexts);
        capture.hidden.clear(); capture.verifying = false;
        for (int layer : draftTargetLayers) {
            Data &feature = capture.hidden[layer];
            feature.dataType = BFLOAT16; feature.Resize({1, draftWindow + GetPageLen() - 1, embed_dim});
            feature.ToDevice(DataDevice::CUDA, {owner}, false); feature.Allocate(true);
        }
        std::vector<std::pair<Data, Data>> unused;
        CommitDraftContext(capture, draftWindow + GetPageLen() - 1, *contexts.front(), unused);
    }
    if (GetFastllmEnv().cudaGraph && preparingGraphs) {
        // Warm the largest activation shapes before capturing any variant.
        // Exercise both eager attention paths; one captured shape reads all lengths.
        auto buckets = DecodeGraphBatchBuckets(batch, 8, 8);
        buckets.back() = batch;
        const bool finalCapture = draftCapturing;
        for (int pass = 0; pass < (finalCapture ? 2 : 1); ++pass) {
            draftCapturing = finalCapture && pass == 1;
            for (auto it = buckets.rbegin(); it != buckets.rend(); ++it) {
                const int count = *it;
                std::vector<DraftContext *> selected(contexts.begin(), contexts.begin() + count);
                std::vector<int> ids(count, 1);
                for (bool shortAttention : {false, true}) {
                    if (shortAttention && draftCapturing) break;
                    for (auto *ctx : selected) ctx->committed = shortAttention ? 2 : draftWindow - 1;
                    Data features;
                    if (!RunDraftBatch(ids, selected, features)) continue;
                    // Full blocks serve ordinary rounds; single-request tails
                    // prepare every legal output/context-limit length as well.
                    for (int limit = steps; limit >= (count == 1 ? 1 : steps); --limit) {
                        Data input; GatherDraftHeadInput(features, std::vector<int>(count, limit), draftBlock, input);
                        Data logits = RunDraftHead(input);
                        std::vector<GenerationConfig> configs(count);
                        std::vector<std::unique_ptr<DraftRound>> rounds;
                        std::vector<DraftRound *> views;
                        for (int row = 0; row < count; ++row) {
                            configs[row].temperature = 1.f; configs[row].top_p = .95f;
                            configs[row].top_k = logits.dims.back();
                            rounds.emplace_back(new DraftRound(*this, *selected[row], configs[row], 1, limit));
                            views.push_back(rounds.back().get());
                        }
                        PrepareLowLatencyProposalGraph(views, logits, &features);
                    }
                }
            }
        }
        draftCapturing = finalCapture;
    }
    for (auto *context : contexts) if (history && context->workspace) {
        auto &tail = context->workspace->prefixStorage;
        tail.dataType = BFLOAT16; tail.Resize({1, draftWindow + GetPageLen() - 1, embed_dim});
        tail.ToDevice(DataDevice::CUDA, {owner}, false); tail.Allocate(false);
    }
    idleDraftContexts = std::move(owners);
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (draftBatchStorage && draftBatchStorage->fusedHeadReady) {
        std::vector<const void *> hidden; std::vector<void *> ready;
        for (const auto &rank : draftBatchStorage->tpRanks) {
            FastllmCudaSetDevice(rank->device);
            if (!rank->prefetchStream) rank->prefetchStream = FastllmCudaStreamCreate();
            if (!rank->prefetchReady) rank->prefetchReady = FastllmCudaEventCreate();
            hidden.push_back(rank->normalized.cudaData); ready.push_back(rank->done);
        }
        ApplyDraftDevice();
        if (draftContextStorage && draftContextStorage->tpRanks.size() == 8) {
            for (auto &state : draftBatchWorkspaces) {
                if (!state->lowLatencyBody || state->rankRows != std::vector<int>(8,1) ||
                    state->tpRanks.size() != 8) continue;
                for (int rank=0; rank<8; ++rank) {
                    auto &local = *state->tpRanks[rank];
                    auto &body = local.graphs[0];
                    if (!body.exec || body.acceptedExec) continue;
                    FastllmCudaSetDevice(local.device);
                    body.acceptedExec = FastllmDraftFusedAcceptedBodyCreate(draftBatchStorage->fused,rank,
                        body.graph,draftContextStorage->tpRanks[rank]->done);
                }
            }
            ApplyDraftDevice();
        }
        PrepareDraftVerificationGraphs(draftBatchStorage->fused,hidden,ready);
    }
#endif
#endif
}

bool NaiveN05FlashModel::TryDraftProposal(DraftRound &round) {
    auto propose = DraftOps().beforeBody;
    return propose && (this->*propose)(round);
}

bool NaiveN05FlashModel::TryDraftProposal(const std::vector<DraftRound *> &rounds, const Data &normalized) {
    auto propose = DraftOps().afterBody;
    return propose && (this->*propose)(rounds, normalized);
}

bool NaiveN05FlashModel::TryDraftProposal(const std::vector<std::unique_ptr<DraftRound>> &rounds,
        const Data &normalized) {
    auto propose = DraftOps().afterBody;
    if (!propose) return false;
    std::vector<DraftRound *> views;
    views.reserve(rounds.size());
    for (const auto &round : rounds) views.push_back(round.get());
    return (this->*propose)(views, normalized);
}

bool NaiveN05FlashModel::DraftDeviceVerification() const {
    return DraftOps().deviceVerification;
}
void NaiveN05FlashModel::UploadDraftVerifyInputs(const TargetCapture &capture, void *graph, Data &storage,
        int stride, int rows) {
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (capture.compositeDraft) {
        std::vector<FastllmDraftHeadRequest> requests;
        for (const auto &input : capture.draftInputs)
            requests.push_back({input.anchor, input.position, input.seed, input.temperature});
        AssertInFastLLM(!requests.empty() && FastllmDraftFusedVerificationUpload(graph, requests.data(),
            requests.size(), nullptr), "Cannot upload composite draft verification inputs.");
        FastllmCudaCurrentThreadStreamWaitEvent(capture.draftInputs.front().inputReady);
        return;
    }
    int total = 0;
    for (const auto &input : capture.draftInputs) {
        AssertInFastLLM(input.rows >= 2 && input.rows <= 8, "Invalid draft verification prefix.");
        total += input.rows;
    }
    AssertInFastLLM(total == rows && !storage.dims.empty(), "Invalid device draft verification inputs.");
    int offset = 0;
    for (size_t row = 0; row < capture.draftInputs.size(); ++row) {
        const auto &input = capture.draftInputs[row];
        if (!row || input.ready != capture.draftInputs[row - 1].ready)
            FastllmCudaCurrentThreadStreamWaitEvent(input.ready);
        AssertInFastLLM(FastllmDraftFusedVerificationInput(input.tokens, storage.cudaData, stride, offset,
                        input.position, input.rows, nullptr), "Cannot prepare device verification inputs.");
        offset += input.rows;
    }
#else
    AssertInFastLLM(false, "Device verification input requires a prepared draft backend.");
#endif
}
void *NaiveN05FlashModel::DraftVerificationExec(void *graph, const TargetCapture *capture) {
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    return FastllmDraftFusedVerificationExecReady(graph, capture && capture->compositeDraft,
        capture && !capture->draftInputs.empty() && capture->draftInputs.front().proposed);
#else
    return nullptr;
#endif
}

void NaiveN05FlashModel::PrepareDraftVerificationGraphs(void *fused,
        const std::vector<const void *> &hidden,const std::vector<void *> &ready) {
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (!preparingGraphs || !draftCapturing || !fused || hidden.size()!=tpDevices.size() || ready.size()!=tpDevices.size()) return;
    const int verifyRows = draftTokens+1;
    if (verifyRows < 2 || verifyRows > draftBlock+1 || verifyRows > 8) return;
    for (auto &state : tpBatchVerifyGraphs) {
        const int count=state->sequenceRows.size();
        if (!state->captured || state->disabled || count<1 || count>64 || state->rows!=count*verifyRows ||
            !std::all_of(state->sequenceRows.begin(),state->sequenceRows.end(),[&](int rows) { return rows==verifyRows; })) continue;
        bool complete=true;
        std::vector<std::shared_ptr<void>> plans(tpDevices.size());
        for (size_t rank=0; rank<tpDevices.size(); ++rank) {
            auto &r=*state->ranks[rank]; auto &storage=*r.storage;
            FastllmCudaSetDevice(tpDevices[rank]);
            const int stride=storage.maxRows?storage.maxRows:state->rows;
            void *plan=FastllmDraftFusedVerificationCreate(fused,rank,count,hidden[rank],
                storage.inputStorage.cudaData,stride,ready[rank],r.graph,verifyRows);
            plans[rank]=std::shared_ptr<void>(plan,FastllmDraftFusedVerificationDestroy);
            complete=complete && plan;
        }
        if (complete) for (size_t rank=0;rank<tpDevices.size();++rank) {
            auto &r=*state->ranks[rank];
            FastllmCudaSetDevice(tpDevices[rank]);
            r.draftVerification=std::move(plans[rank]);
            r.draftFused=fused;r.draftHidden=hidden[rank];
            // The prefix can be disabled on this same executable for fallback.
            // Do not retain a second copy of the complete target execution plan.
            FastllmCudaGraphExecDestroy(r.exec);r.exec=nullptr;
        }
    }
    ApplyDraftDevice();
#endif
}

}
