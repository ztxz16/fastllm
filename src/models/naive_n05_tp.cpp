#include "models/naive_n05_flash.h"
#include "naive_n05_tp.h"
#include "executor.h"
#include "utils.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {

bool NaiveN05FlashModel::CanReuseTensorParallelCache(const ResponseContext *context) const {
#ifdef USE_CUDA
    // With a single live handle, removal happens after its last forward and
    // creation precedes its first forward. Never touch another request's KV or
    // the worker-owned decode state from these callbacks.
    return !isFree && tpDevices.size() > 1 && maxBatch == 1 &&
        GetFastllmEnv().cudaGraph && !saveHistoryChat && !GetKVCacheInCPU() &&
        kvCacheDataType == BFLOAT16 && context->multimodalInput.empty() &&
        context->cacheLen == 0 && responseContextDict.dicts.size() == 1 &&
        responseContextDict.dicts.begin()->second == context;
#else
    return false;
#endif
}

void NaiveN05FlashModel::RestoreTensorParallelCache(ResponseContext *context) {
    if (tpIdleCache.empty()) return;
    // Release unused storage on every return path, before a new prefill can
    // allocate another set. A successful swap leaves only empty descriptors.
    std::vector<std::pair<Data, Data>> idle;
    idle.swap(tpIdleCache);
    if (!CanReuseTensorParallelCache(context) ||
        (int)context->pastKeyValues.size() != block_cnt || (int)idle.size() != block_cnt) return;
    const int reserve = CacheReserveCapacity(context->generationConfig);
    if (reserve <= 0) return;
    const int64_t capacity = ((int64_t)reserve + 127) / 128 * 128;
    for (int layer = 0; layer < block_cnt; ++layer) {
        for (Data *root : {&context->pastKeyValues[layer].first,
                           &context->pastKeyValues[layer].second})
            if (!root->dims.empty() || root->multiDeviceData || root->cudaData || root->cpuData)
                return;
        if (slidingLayers[layer]) continue;
        for (Data *root : {&idle[layer].first, &idle[layer].second})
            for (int device : tpDevices)
                if (root->multiDeviceDatas.at(device)->expansionDims[1] != capacity)
                    return;
    }
    context->pastKeyValues.swap(idle);
    // The scheduler identifies a pending prompt by EMPTY root capacity.
    // Keeping the previous root capacity would consume an active slot and
    // could prevent this fresh request from ever being admitted.
    for (auto &layer : context->pastKeyValues) {
        for (Data *root : {&layer.first, &layer.second}) {
            root->dims.clear();
            root->strides.clear();
            root->expansionDims.clear();
            for (auto &item : root->multiDeviceDatas) {
                Data &local = *item.second;
                local.Resize({1, 0, local.dims[2]});
            }
        }
    }
}

void NaiveN05FlashModel::RecycleTensorParallelCache(ResponseContext *context) {
    if (!CanReuseTensorParallelCache(context) ||
        (!context->isEnding && !context->isAbort) ||
        context->error != ResponseContextErrorNone ||
        CacheReserveCapacity(context->generationConfig) <= 0 ||
        (int)context->pastKeyValues.size() != block_cnt) return;
    for (const auto &layer : context->pastKeyValues) {
        for (const Data *root : {&layer.first, &layer.second}) {
            if (!root->multiDeviceData || root->isPagedKVCache || root->isFake ||
                root->cudaData || root->cpuData || root->dataDeviceIds != tpDevices ||
                root->multiDeviceDatas.size() != tpDevices.size()) return;
            for (int device : tpDevices) {
                auto it = root->multiDeviceDatas.find(device);
                if (it == root->multiDeviceDatas.end() || !it->second) return;
                const Data &local = *it->second;
                if (local.isFake || local.cudaDataBorrowed || local.isPagedKVCache ||
                    local.dataDevice != DataDevice::CUDA || local.dataType != BFLOAT16 ||
                    !local.cudaData || local.dims.size() != 3 || local.dims[1] <= 0 ||
                    local.expansionDims.size() != 3 || local.expansionDims[1] < local.dims[1]) return;
            }
        }
    }
    // Bound retention to one completed request's existing allocation set;
    // ownership is transferred, never copied or shared with another handle.
    tpIdleCache.clear();
    tpIdleCache.swap(context->pastKeyValues);
}

bool NaiveN05FlashModel::InitTensorParallel() {
#ifdef USE_CUDA
    // Use the existing --tp setting; ordinary lists of layer devices retain
    // the serial path. All ranks in this implementation have equal widths.
    const char *value = std::getenv("FASTLLM_TP");
    std::string spec = value ? value : "";
    if (spec.empty() || spec == "false" || spec == "off" || spec == "none") return false;
    tpDevices.clear();
    std::map<int, int> ratios;
    if (spec == "auto" || spec == "true" || spec == "on" ||
        std::all_of(spec.begin(), spec.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        int count = spec == "auto" || spec == "true" || spec == "on"
            ? FastllmCudaGetDeviceCount() : std::max(1, std::stoi(spec));
        for (int i = 0; i < count; ++i) tpDevices.push_back(i);
    } else {
        if (spec.find(':') == std::string::npos) spec = "cuda:" + spec;
        tpDevices = ParseDeviceIds(spec, StartWith(spec, "multicuda") ? "multicuda" : "cuda", ratios);
    }
    AssertInFastLLM(!tpDevices.empty(), "Naive TP: invalid device list.");
    std::set<int> unique;
    for (int device : tpDevices) {
        AssertInFastLLM(device >= 0 && device < FastllmCudaGetDeviceCount() && unique.insert(device).second,
                        "Naive TP: device is unavailable or repeated.");
        AssertInFastLLM(ratios.empty() || ratios[device] == ratios[tpDevices[0]],
                        "Naive TP requires equal device ratios.");
    }
    if (tpDevices.size() <= 1) { tpDevices.clear(); return false; }
    const int ranks = tpDevices.size();
    for (auto cfg : {full, sliding}) {
        AssertInFastLLM(cfg.heads % ranks == 0 &&
            (cfg.kvHeads % ranks == 0 || ranks % cfg.kvHeads == 0),
            "Naive TP requires complete Q heads and divisible or replicated KV groups.");
    }
    return true;
#else
    return false;
#endif
}

void NaiveN05FlashModel::PrepareTensorParallel() {
#ifdef USE_CUDA
    if (tpPrepared) return;
    const int ranks = tpDevices.size();
    AssertInFastLLM(ranks > 1 && !GetKVCacheInCPU(), "Naive TP requires CUDA KV cache.");
    AssertInFastLLM(FastllmInitNccl(tpDevices), "Naive TP: NCCL initialization failed.");
    auto rangeScheme = [&](int total, int copies = 1) {
        AssertInFastLLM(total % ranks == 0, "Naive TP: projection width must divide the rank count.");
        DivisionScheme scheme;
        for (int r = 0; r < ranks; ++r)
            for (int c = 0; c < copies; ++c)
                scheme[tpDevices[r]].push_back({c * total + r * (total / ranks),
                                                c * total + (r + 1) * (total / ranks)});
        return scheme;
    };
    auto split = [&](const std::string &name, DivisionScheme scheme, int axis,
                     const std::string &biasName = "") {
        Data emptyBias;
        weight[name].tpLinearType = axis == 0 ? TP_LINEAR_ROW : TP_LINEAR_COLUMN;
        Data &bias = biasName.empty() ? emptyBias : weight[biasName];
        if (!bias.dims.empty()) ToDataType(bias, FLOAT32);
        AssertInFastLLM(SplitMultiCudaWeight(weight[name], bias, tpDevices, scheme, axis, true),
                        "Naive TP: cannot split " + name);
    };
    auto replicate = [&](const std::string &name) {
        auto it = weight.weight.find(name);
        if (it != weight.weight.end() && !it->second.dims.empty())
            PrepareMultiCudaReplicatedData(it->second, tpDevices, true);
    };
    if (GetCudaEmbedding() && !GetLowMemMode()) replicate("model.embed_tokens.weight");
    replicate("model.norm.weight");
    for (int layer = 0; layer < block_cnt; ++layer) {
        std::string prefix = "model.layers." + std::to_string(layer);
        std::string ap = prefix + ".self_attn.";
        const auto &cfg = slidingLayers[layer] ? sliding : full;
        replicate(prefix + ".input_layernorm.weight");
        replicate(prefix + ".post_attention_layernorm.weight");
        split(ap + "q_proj.weight", rangeScheme(cfg.heads * cfg.headDim), 0, ap + "q_proj.bias");
        for (auto item : {std::make_pair("k", cfg.headDim), std::make_pair("v", cfg.valueDim)}) {
            DivisionScheme scheme;
            for (int r = 0; r < ranks; ++r) {
                // When TP exceeds KV heads, adjacent ranks share one KV head.
                int begin = r * cfg.kvHeads / ranks;
                int count = std::max(1, cfg.kvHeads / ranks);
                scheme[tpDevices[r]] = {{begin * item.second, (begin + count) * item.second}};
            }
            split(ap + item.first + "_proj.weight", scheme, 0, ap + item.first + "_proj.bias");
        }
        split(ap + "o_proj.weight", rangeScheme(cfg.heads * cfg.valueDim), 1);
        auto sinkIt = weight.weight.find(ap + "attention_sink_bias");
        if (sinkIt != weight.weight.end() && !sinkIt->second.dims.empty()) {
            ToDataType(sinkIt->second, FLOAT32);
            AssertInFastLLM(SplitMultiCudaWeight1D(sinkIt->second, tpDevices, rangeScheme(cfg.heads)),
                            "Naive TP: cannot split attention sinks.");
        }
        if (!slidingLayers[layer]) {
            for (const char *name : {"wq.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias"})
                replicate(ap + "indexer." + name);
        }
        if (moeLayers[layer]) {
            replicate(prefix + ".mlp.gate.weight");
            replicate(prefix + ".mlp.gate.e_score_correction_bias");
            for (int e = 0; e < num_experts; ++e) {
                std::string base = prefix + ".mlp.experts." + std::to_string(e) + ".";
                int mid = weight[base + "down_proj.weight"].dims[1];
                split(base + "gateup_proj.weight", rangeScheme(mid, 2), 0);
                split(base + "down_proj.weight", rangeScheme(mid), 1);
            }
        } else {
            int mid = weight[prefix + ".mlp.down_proj.weight"].dims[1];
            auto scheme = rangeScheme(mid);
            split(prefix + ".mlp.gate_proj.weight", scheme, 0);
            split(prefix + ".mlp.up_proj.weight", scheme, 0);
            split(prefix + ".mlp.down_proj.weight", scheme, 1);
        }
    }
    // Logits are gathered on the host for the existing sampler, without a
    // full-vocabulary all-reduce or GPU broadcast.
    int vocab = weight["lm_head.weight"].dims[0];
    DivisionScheme vocabScheme;
    tpVocabRanges.clear();
    for (int r = 0; r < ranks; ++r) {
        tpVocabRanges.push_back({(int)((int64_t)vocab * r / ranks), (int)((int64_t)vocab * (r + 1) / ranks)});
        vocabScheme[tpDevices[r]] = {tpVocabRanges.back()};
    }
    split("lm_head.weight", vocabScheme, 0);
    tpMoeWeights.resize(ranks); tpMoeBiases.resize(ranks);
    for (int r = 0; r < ranks; ++r) {
        tpMoeWeights[r].resize(block_cnt); tpMoeBiases[r].resize(block_cnt);
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (!moeLayers[layer]) continue;
            auto &weights = tpMoeWeights[r][layer];
            weights = {nullptr, nullptr};
            for (int e = 0; e < num_experts; ++e) {
                std::string base = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
                for (const char *name : {"gateup_proj.weight", "down_proj.weight"})
                    weights.push_back(weight[base + name].multiDeviceDatas.at(tpDevices[r]));
            }
            tpMoeBiases[r][layer].resize(weights.size(), nullptr);
        }
    }
    tpPrepared = true;
#endif
}

#ifdef USE_CUDA
void NaiveN05FlashModel::TPDecodeState::ClearGraphs() {
    for (size_t rank = 0; rank < ranks.size(); ++rank) {
        FastllmCudaSetDevice(devices[rank]);
        FastllmCudaGraphExecDestroy(ranks[rank]->exec);
        FastllmCudaGraphDestroy(ranks[rank]->graph);
        ranks[rank]->exec = ranks[rank]->graph = nullptr;
        ranks[rank]->ok = true;
    }
    FastllmCudaGraphMemoryPoolRelease(reservedPointers);
    reservedPointers.clear();
    warmed = captured = disabled = active = false;
}

NaiveN05FlashModel::TPDecodeState::~TPDecodeState() { ClearGraphs(); }

bool NaiveN05FlashModel::PrepareTensorParallelDecode(const Data &inputIds,
        std::vector<std::pair<Data, Data>> &kv, bool verifying, const LogitsSelection &selection) {
    // Prefill has its own scratch. Keep decode and registered communication
    // buffers alive across requests, but only replay after validating all KV
    // addresses/capacities and the collective generation below.
    if (tpDecodeState) tpDecodeState->active = false;
    const int rows = verifying ? 8 : 1;
    if (!GetFastllmEnv().cudaGraph || inputIds.dims != std::vector<int>({1, rows}) ||
        inputIds.dataType != FLOAT32 || isIntermediateChunkedPrefill ||
        !GetCudaEmbedding() || GetLowMemMode() || dataType != BFLOAT16 ||
        kvCacheDataType != BFLOAT16 || moeAtype != BFLOAT16 ||
        indexHeads != 16 || indexDim != 128 || indexTopK != 2048 ||
        window < 2 || window > 128 || kv[0].first.dims.size() != 3)
        return false;
    const int firstLength = kv[0].first.dims[1] + 1;
    const int nextLength = firstLength + rows - 1;
    // At 2048 keys eager attention changes its PV reduction tree. Execute that
    // boundary token eagerly and capture the sparse tree from the next token.
    if (firstLength <= 1 || nextLength > max_positions ||
        (firstLength <= indexTopK && nextLength >= indexTopK) ||
        (verifying && firstLength <= 256 && nextLength > 256))
        return false;
    int region = nextLength <= 256 ? 0 : nextLength < indexTopK ? 1 : 2;
    int capacity = region == 0 ? 256 : region == 1 ? indexTopK - 1 :
        (int)std::min<int64_t>(max_positions, ((int64_t)nextLength + 4095) / 4096 * 4096);
    const int ranks = tpDevices.size();
    // Retain the existing FlashInfer path on larger shards until a graph-safe
    // dynamic-length equivalent is available.
    if (sliding.heads / ranks >= 64 && sliding.heads / sliding.kvHeads == 8 &&
        window == 128 && sliding.headDim == 192 && sliding.valueDim == 128)
        return false;
    std::vector<void *> pointers;
    std::vector<int> capacities;
    for (int layer = 0; layer < block_cnt; ++layer) {
        for (int device : tpDevices) {
            for (Data *root : {&kv[layer].first, &kv[layer].second}) {
                Data &local = *root->multiDeviceDatas.at(device);
                int required = slidingLayers[layer]
                    ? (verifying ? std::min(firstLength - 1, window - 1) + rows
                                 : std::min(nextLength, window)) : nextLength;
                if (local.dims.size() != 3 || local.expansionDims.size() != 3 ||
                    local.expansionDims[1] < required || !local.cudaData ||
                    local.dataDevice != DataDevice::CUDA || local.dataType != BFLOAT16)
                    return false;
                pointers.push_back(local.cudaData);
                capacities.push_back(local.expansionDims[1]);
                if (!slidingLayers[layer]) capacity = std::min(capacity, local.expansionDims[1]);
            }
        }
    }
    uint64_t generation = FastllmGetNcclGeneration();
    if (tpDecodeState && tpDecodeState->cachePointers == pointers &&
        tpDecodeState->cacheCapacities == capacities && tpDecodeState->region == region &&
        tpDecodeState->rows == rows && tpDecodeState->verifying == verifying &&
        tpDecodeState->selection.greedy == selection.greedy &&
        tpDecodeState->selection.count == selection.count &&
        tpDecodeState->selection.invTemperature == selection.invTemperature &&
        tpDecodeState->capacity >= nextLength && tpDecodeState->ncclGeneration == generation)
        return tpDecodeState->active = !tpDecodeState->disabled;
    auto state = tpDecodeState ? tpDecodeState : std::make_shared<TPDecodeState>();
    state->ClearGraphs();
    state->disabled = true;
    state->devices = tpDevices;
    state->region = region;
    state->rows = rows;
    state->verifying = verifying;
    state->selection.greedy = selection.greedy;
    state->selection.count = selection.count;
    state->selection.invTemperature = selection.invTemperature;
    state->capacity = capacity;
    state->cachePointers = std::move(pointers);
    state->cacheCapacities = std::move(capacities);
    state->ncclGeneration = generation;
    for (int rank = 0; rank < ranks; ++rank) {
        FastllmCudaSetDevice(tpDevices[rank]);
        if (!FastllmCudaNaiveDecodeGraphSupported()) return false;
        auto &embedding = weight["model.embed_tokens.weight"];
        auto it = embedding.multiDeviceDatas.find(tpDevices[rank]);
        if (it == embedding.multiDeviceDatas.end() || !it->second->cudaData ||
            it->second->dataType != BFLOAT16) return false;
        // Other MoE implementations can read routing results on the host. The
        // packed BF16 backend keeps routes and intermediate storage on device.
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (!moeLayers[layer]) continue;
            if (!FastllmCudaNVFP4E4M3GroupedMoeSupported(tpDevices[rank])) return false;
            auto &weights = tpMoeWeights[rank][layer];
            for (Data *w : weights)
                if (w && w->dataType != NVFP4_BLOCK_16_E4M3_PACKED) return false;
            if (!FastllmCudaPrepareNVFP4E4M3Moe(weights.data(), weights.size())) return false;
        }
        if (rank >= (int)state->ranks.size())
            state->ranks.emplace_back(new TPDecodeState::Rank());
        state->ranks[rank]->buffers.capacity = capacity;
        state->ranks[rank]->features.verifying = verifying;
        state->ranks[rank]->features.collectHidden = verifying && rank == 0;
    }
    state->disabled = false;
    state->active = true;
    tpDecodeState = std::move(state);
    return true;
}

Data NaiveN05FlashModel::ForwardTensorParallelDecode(int rank, const Data &inputIds,
        const Data &positions, std::vector<std::pair<Data, Data>> &kv,
        const GenerationConfig &config, const Data *embedding, TargetCapture *capture,
        float *candidateResult) {
    auto &state = *tpDecodeState;
    auto &r = *state.ranks.at(rank);
    auto &buf = r.buffers;
    if (state.mode != TPDecodeState::Capture) {
        // Keep these allocations and their CUDA addresses across all replays.
        auto copyInput = [&](Data &dst, const Data &src) {
            if (dst.dims.empty()) {
                dst.dataType = src.dataType;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {tpDevices[rank]};
                dst.Resize(src.dims);
                dst.Allocate();
            }
            if (src.dataDevice == DataDevice::CUDA)
                FastllmCudaCopyFromDeviceToDevice(dst.cudaData, src.cudaData, src.GetBytes());
            else FastllmCudaCopyFromHostToDevice(dst.cudaData, src.cpuData, src.GetBytes());
        };
        copyInput(buf.inputIds, inputIds);
        Data localPositions(positions);
        ToDataType(localPositions, FLOAT32);
        copyInput(buf.positions, localPositions);
        int lengths[8];
        for (int row = 0; row < state.rows; ++row) lengths[row] = kv[0].first.dims[1] + row + 1;
        if (buf.liveKeys.dims.empty()) {
            buf.liveKeys.dataType = INT32;
            buf.liveKeys.UpdateUnitSize();
            buf.liveKeys.dataDevice = DataDevice::CUDA;
            buf.liveKeys.dataDeviceIds = {tpDevices[rank]};
            buf.liveKeys.Resize({state.rows});
            buf.liveKeys.Allocate();
        }
        FastllmCudaCopyFromHostToDevice(buf.liveKeys.cudaData, lengths, state.rows * sizeof(int));
        if (state.mode == TPDecodeState::Prepare) return Data();
    }
    if (state.mode == TPDecodeState::Replay) {
        AssertInFastLLM(FastllmCudaGraphLaunch(r.exec), "Naive TP decode graph launch failed.");
    } else {
        if (state.mode == TPDecodeState::Warm) r.communicationPointers.clear();
        RunTarget(buf.inputIds, buf.positions, kv, config,
                  state.verifying ? &r.features : nullptr, rank, embedding, &buf);
        if (state.selection.count)
            FastllmCudaNaiveLogitsSelect(buf.logits, tpVocabRanges[rank].first,
                state.selection.count, state.selection.greedy, state.selection.invTemperature,
                r.logitsPartial, r.logitsCandidates);
        if (state.mode == TPDecodeState::Capture) {
            bool clean = !FastllmCudaGetThreadError();
            r.ok = FastllmCudaGraphEndCapture(&r.graph) && clean && r.ok;
            if (!r.ok) fprintf(stderr, "[Fastllm] Naive TP graph rank %d: %s\n", rank,
                               FastllmCudaGraphLastError());
            return Data();
        }
    }
    if (capture && capture->collectHidden)
        for (auto &feature : r.features.hidden) Copy(feature.second, capture->hidden[feature.first]);
    if (state.selection.count) {
        AssertInFastLLM(candidateResult != nullptr, "Naive TP candidate output is missing.");
        FastllmCudaCopyFromDeviceToHost(candidateResult, r.logitsCandidates.cudaData, r.logitsCandidates.GetBytes());
        return Data();
    }
    // ToDevice(CPU) would release the graph's persistent output allocation.
    Data output(FLOAT32, buf.logits.dims);
    output.Allocate();
    FastllmCudaCopyFromDeviceToHost(output.cpuData, buf.logits.cudaData, buf.logits.GetBytes());
    return output;
}
#endif

Data NaiveN05FlashModel::ForwardSingleGPU(int rank, const Data &inputIds, const Data &positions,
        std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config, const Data *embedding,
        TargetCapture *capture, float *candidateResult, const LogitsSelection *selection) {
#ifdef USE_CUDA
    int device = tpDevices.at(rank);
    FastllmCudaSetDevice(device);
    static thread_local std::unique_ptr<Executor> executor;
    if (!executor) executor.reset(new Executor());
    executor->SetFirstDevice("cuda:" + std::to_string(device));
    struct RestoreExecutor {
        void *previous;
        ~RestoreExecutor() { SetCurrentThreadExecutor(previous); }
    } restore{GetExecutor()};
    SetCurrentThreadExecutor(executor.get());
    if (tpDecodeState && tpDecodeState->active && !tpDecodeState->disabled)
        return ForwardTensorParallelDecode(rank, inputIds, positions, kv, config, embedding, capture, candidateResult);
    // Generic operators can move their inputs. Give every worker its own IDs.
    Data localIds(inputIds), localPositions(positions);
    TargetWorkspace *workspace = capture && capture->verifying
        ? tpVerifyWorkspaces.at(rank).get() : nullptr;
    Data logits = RunTarget(localIds, localPositions, kv, config, capture, rank, embedding, workspace);
    if (candidateResult && !isIntermediateChunkedPrefill) {
        Data partial, top;
        FastllmCudaNaiveLogitsSelect(logits, tpVocabRanges[rank].first,
            selection->count, selection->greedy, selection->invTemperature, partial, top);
        FastllmCudaCopyFromDeviceToHost(candidateResult, top.cudaData, top.GetBytes());
        ForceDeviceSync();
        return Data();
    }
    if (!isIntermediateChunkedPrefill) logits.ToDevice(DataDevice::CPU);
    ForceDeviceSync();
    return logits;
#else
    ErrorInFastLLM("Naive ForwardSingleGPU requires CUDA.");
    return Data();
#endif
}

Data NaiveN05FlashModel::ForwardTensorParallel(const Data &inputIds, const Data &positions,
        std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config, TargetCapture *capture,
        LogitsSelection *selection) {
#ifdef USE_CUDA
    PrepareTensorParallel();
    const int vocab = weight["lm_head.weight"].dims[0];
    const bool compact = selection && selection->count > 0 && !isIntermediateChunkedPrefill &&
        std::all_of(tpVocabRanges.begin(), tpVocabRanges.end(), [&](const auto &range) {
            return FastllmNaiveCanSelectLogits(range.second - range.first, range.first,
                                              selection->count, selection->greedy);
        });
    if (compact) selection->count = std::min(selection->count, vocab);
    const int count = compact ? selection->count : 0;
    LogitsSelection fullLogits;
    const int rows = capture && capture->verifying ? inputIds.dims[1] : 1;
    AssertInFastLLM(!saveHistoryChat, "Naive TP currently requires --cache_history false.");
    AssertInFastLLM((int)kv.size() == block_cnt, "Naive TP: incomplete KV cache.");
    for (auto &layer : kv) {
        for (Data *cache : {&layer.first, &layer.second}) {
            if (cache->multiDeviceData) continue;
            AssertInFastLLM(cache->dims.empty(), "Naive TP cannot reuse a serial KV cache.");
            cache->multiDeviceData = true;
            cache->dataDevice = DataDevice::CUDA;
            cache->dataDeviceIds = tpDevices;
            cache->isKVCache = true;
            for (int device : tpDevices) {
                Data *local = new Data(kvCacheDataType);
                local->dataDevice = DataDevice::CUDA;
                local->dataDeviceIds = {device};
                local->isKVCache = true;
                cache->multiDeviceDatas[device] = local;
            }
        }
    }
    // CPU embedding uses the shared CPU worker pool, so compute it once on
    // the caller instead of concurrently entering that pool from eight ranks.
    Data embedding;
    if (!GetCudaEmbedding() || GetLowMemMode())
        Embedding(inputIds, weight["model.embed_tokens.weight"], embedding);
    // One bounded graph state for full verification blocks, separate from
    // ordinary decode. Smaller blocks and region boundaries remain eager.
    const bool verify = capture && capture->verifying;
    struct RestoreState {
        std::shared_ptr<TPDecodeState> &active, &verifyState;
        bool swapped;
        ~RestoreState() { if (swapped) active.swap(verifyState); }
    } restore{tpDecodeState, tpVerifyState, verify};
    if (verify) tpDecodeState.swap(tpVerifyState);
    if (capture && tpDecodeState) tpDecodeState->active = false;
    bool graphDecode = (!capture || verify) && PrepareTensorParallelDecode(inputIds, kv, verify, compact ? *selection : fullLogits);
    // Retain each rank's verification scratch and communication addresses.
    // The existing collective registration still validates the complete tuple.
    if (capture && capture->verifying && tpVerifyWorkspaces.empty()) {
        for (size_t rank = 0; rank < tpDevices.size(); ++rank)
            tpVerifyWorkspaces.emplace_back(new TargetWorkspace());
    }
    std::vector<TargetCapture> captures(capture ? tpDevices.size() : 0);
    if (capture) {
        for (int rank = 0; rank < (int)tpDevices.size(); ++rank) {
            captures[rank].verifying = capture->verifying;
            captures[rank].collectHidden = capture->collectHidden && rank == 0;
        }
    }
    std::vector<Data> logits(compact ? 0 : tpDevices.size());
    std::vector<float> candidates(compact ? tpDevices.size() * rows * count * 2 : 0);
    std::vector<std::exception_ptr> errors(tpDevices.size());
    auto runRanks = [&](const std::function<void(int)> &task) {
        std::fill(errors.begin(), errors.end(), nullptr);
        tpWorkers.Run(tpDevices, task, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
    };
    auto forwardRanks = [&]() {
        runRanks([&](int rank) {
            Data local = ForwardSingleGPU(rank, inputIds, positions, kv, config,
                                          embedding.dims.empty() ? nullptr : &embedding,
                                          capture ? &captures[rank] : nullptr,
                                          compact ? candidates.data() + (size_t)rank * rows * count * 2 : nullptr,
                                          compact ? selection : nullptr);
            if (!compact) logits[rank].CopyFrom(local);
        });
    };
    if (graphDecode && tpDecodeState->warmed && !tpDecodeState->captured) {
        auto &state = *tpDecodeState;
        state.mode = TPDecodeState::Prepare;
        forwardRanks();
        // Resolve sentinels on all devices before beginning any capture.
        runRanks([&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            state.ranks[rank]->ok = FastllmCudaGraphPrepareCaptureDevice();
        });
        bool ok = std::all_of(state.ranks.begin(), state.ranks.end(),
                             [](const auto &r) { return r->ok; });
        bool pool = ok && FastllmCudaGraphMemoryPoolBegin();
        if (pool) {
            runRanks([&](int rank) {
                FastllmCudaSetDevice(tpDevices[rank]);
                FastllmCudaClearThreadError();
                state.ranks[rank]->ok = FastllmCudaGraphBeginCapture();
            });
            ok = std::all_of(state.ranks.begin(), state.ranks.end(),
                             [](const auto &r) { return r->ok; });
            if (ok) {
                state.mode = TPDecodeState::Capture;
                forwardRanks();
            } else {
                // Every successful begin is ended on its owning worker.
                runRanks([&](int rank) {
                    if (state.ranks[rank]->ok) {
                        FastllmCudaSetDevice(tpDevices[rank]);
                        FastllmCudaGraphEndCapture(&state.ranks[rank]->graph);
                    }
                    state.ranks[rank]->ok = false;
                });
            }
            ok = FastllmCudaGraphMemoryPoolEnd(state.reservedPointers) && ok;
            ok = ok && std::all_of(state.ranks.begin(), state.ranks.end(),
                                   [](const auto &r) { return r->ok && r->graph; });
            if (ok) {
                runRanks([&](int rank) {
                    FastllmCudaSetDevice(tpDevices[rank]);
                    auto &r = *state.ranks[rank];
                    r.ok = FastllmCudaGraphInstantiate(r.graph, &r.exec);
                });
                ok = std::all_of(state.ranks.begin(), state.ranks.end(),
                                 [](const auto &r) { return r->ok && r->exec; });
            }
        } else ok = false;
        if (!ok) {
            // Capture has not executed the token or committed KV metadata.
            // All ranks agree on fallback before any graph is launched.
            state.disabled = true;
            runRanks([&](int rank) {
                FastllmCudaSetDevice(tpDevices[rank]);
                FastllmCudaClearLastError();
                FastllmCudaClearThreadError();
            });
            fprintf(stderr, "[Fastllm] Naive TP decode graph capture failed; using eager for this cache.\n");
            graphDecode = false;
        } else state.captured = true;
    }
    if (graphDecode)
        tpDecodeState->mode = tpDecodeState->captured ? TPDecodeState::Replay : TPDecodeState::Warm;
    forwardRanks();
    if (graphDecode) {
        tpDecodeState->warmed = true;
        int oldLength = kv[0].first.dims[1];
        int nextLength = oldLength + tpDecodeState->rows;
        // Graph kernels append/trim on the device. Commit host metadata once,
        // only after all ranks finish a successful warmup or replay.
        for (int layer = 0; layer < block_cnt; ++layer) {
            int length = slidingLayers[layer]
                ? (verify ? std::min(oldLength, window - 1) + tpDecodeState->rows
                          : std::min(nextLength, window - 1)) : nextLength;
            for (int device : tpDevices)
                for (Data *root : {&kv[layer].first, &kv[layer].second}) {
                    Data &local = *root->multiDeviceDatas.at(device);
                    local.Resize({1, length, local.dims[2]});
                }
        }
    }
    for (int layer = 0; layer < block_cnt; ++layer) {
        const auto &cfg = slidingLayers[layer] ? sliding : full;
        auto syncMeta = [&](Data &root, int width) {
            const Data &local = *root.multiDeviceDatas.at(tpDevices[0]);
            root.Resize({1, local.dims[1], width});
            root.expansionDims = {1, local.expansionDims[1], width};
        };
        syncMeta(kv[layer].first, cfg.kvHeads * cfg.headDim + (slidingLayers[layer] ? 0 : indexDim));
        syncMeta(kv[layer].second, cfg.kvHeads * cfg.valueDim);
    }
    if (capture) capture->hidden = std::move(captures[0].hidden);
    if (isIntermediateChunkedPrefill) return Data();
    if (compact) {
        auto &output = selection->candidates;
        output.dataType = FLOAT32;
        output.Resize({rows, count * 2});
        output.Allocate();
        std::vector<std::pair<float, int>> choices;
        choices.reserve(tpDevices.size() * count);
        for (int row = 0; row < rows; ++row) {
            choices.clear();
            for (size_t rank = 0; rank < tpDevices.size(); ++rank) {
                const float *local = candidates.data() + (rank * rows + row) * count * 2;
                for (int k = 0; k < count; ++k)
                    if (local[k * 2] >= 0) choices.emplace_back(local[k * 2 + 1], (int)local[k * 2]);
            }
            auto better = [&](const auto &a, const auto &b) {
                return selection->greedy ? FastllmNaiveTop1Better(a.first, a.second, b.first, b.second) :
                    (a.first > b.first || (a.first == b.first && a.second < b.second));
            };
            int kept = std::min<int>(count, choices.size());
            std::partial_sort(choices.begin(), choices.begin() + kept, choices.end(), better);
            float *dst = (float *)output.cpuData + (size_t)row * count * 2;
            for (int k = 0; k < count; ++k) {
                dst[k * 2] = k < kept ? choices[k].second : 0;
                dst[k * 2 + 1] = k < kept ? choices[k].first : -INFINITY;
            }
        }
        return Data();
    }
    Data output(FLOAT32, {1, rows, vocab});
    output.Allocate();
    for (int r = 0; r < (int)tpDevices.size(); ++r) {
        auto range = tpVocabRanges[r];
        const int localVocab = range.second - range.first;
        for (int row = 0; row < rows; ++row)
            std::memcpy((float *)output.cpuData + (size_t)row * vocab + range.first,
                        (float *)logits[r].cpuData + (size_t)row * localVocab,
                        localVocab * sizeof(float));
    }
    return output;
#else
    return Data();
#endif
}

bool NaiveN05FlashModel::HasVerificationGraph() const {
#ifdef USE_CUDA
    return GetFastllmEnv().cudaGraph && tpVerifyState &&
        tpVerifyState->warmed && !tpVerifyState->disabled;
#else
    return false;
#endif
}

Data NaiveN05FlashModel::RunDraftHead(Data &hidden) {
    if (tpDevices.empty()) {
        Data output;
        if (hidden.dims[1] > 1) MatMulTransB(hidden, weight["lm_head.weight"], output);
        else Linear(hidden, weight["lm_head.weight"], Data(), output);
        return output;
    }
#ifdef USE_CUDA
    const int rows = hidden.dims[1], vocab = weight["lm_head.weight"].dims[0];
    const int owner = tpDevices.front();
    if (hidden.dataType == BFLOAT16 && hidden.dataDevice == DataDevice::CUDA &&
        hidden.dataDeviceIds == std::vector<int>{owner} && FastllmCudaPeerAccessInit(tpDevices)) {
        // Keep the shared vocabulary shards on their ranks. The owner publishes
        // its hidden rows once, and ranks copy their logits directly into disjoint
        // columns of the owner's result, without a full CPU round trip.
        FastllmCudaSetDevice(owner);
        if (tpDraftHeadInputs.empty()) {
            tpDraftHeadInputs.resize(tpDevices.size());
            tpDraftHeadLogits.resize(tpDevices.size());
        }
        tpDraftHeadOutput.dataType = BFLOAT16;
        tpDraftHeadOutput.UpdateUnitSize();
        tpDraftHeadOutput.Resize({1, rows, vocab});
        tpDraftHeadOutput.ToDevice(DataDevice::CUDA, std::vector<int>{owner}, false);
        tpDraftHeadOutput.Allocate(false);
        struct ReadyEvent {
            void *event = FastllmCudaEventCreate();
            ~ReadyEvent() { FastllmCudaEventDestroy(event); }
        } ready;
        FastllmCudaEventRecordCurrentThread(ready.event);
        std::vector<std::exception_ptr> errors(tpDevices.size());
        tpWorkers.Run(tpDevices, [&](int rank) {
            const int device = tpDevices[rank];
            FastllmCudaSetDevice(device);
            Executor executor;
            executor.SetFirstDevice("cuda:" + std::to_string(device));
            struct Restore {
                void *previous;
                ~Restore() { SetCurrentThreadExecutor(previous); }
            } restore{GetExecutor()};
            SetCurrentThreadExecutor(&executor);
            Data &input = tpDraftHeadInputs[rank], &logits = tpDraftHeadLogits[rank];
            input.dataType = BFLOAT16;
            input.UpdateUnitSize();
            input.Resize(hidden.dims);
            input.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
            input.Allocate(false);
            FastllmCudaCurrentThreadStreamWaitEvent(ready.event);
            AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(device, input.cudaData,
                owner, hidden.cudaData, hidden.GetBytes()), "Naive draft head input copy failed.");
            Data &head = *weight["lm_head.weight"].multiDeviceDatas.at(device);
            if (rows > 1) MatMulTransB(input, head, logits);
            else Linear(input, head, Data(), logits);
            const auto range = tpVocabRanges[rank];
            const size_t bytes = (range.second - range.first) * sizeof(uint16_t);
            AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(
                (uint16_t *)tpDraftHeadOutput.cudaData + range.first, vocab * sizeof(uint16_t),
                logits.cudaData, bytes, bytes, rows), "Naive draft head logits copy failed.");
            // All peer writes finish before the owner consumes the joined head.
            // This also keeps input/logits storage alive for asynchronous copies.
            ForceDeviceSync();
        }, errors);
        FastllmCudaSetDevice(owner);
        for (auto error : errors) if (error) std::rethrow_exception(error);
        Data output;
        Copy(tpDraftHeadOutput, output);
        return output;
    }
    // Copy once before workers consume the owner stream's result. Each worker
    // owns its inputs and the existing vocabulary shard; no extra full head.
    Data host(hidden);
    host.ToDevice(DataDevice::CPU);
    std::vector<Data> logits(tpDevices.size());
    std::vector<std::exception_ptr> errors(tpDevices.size());
    tpWorkers.Run(tpDevices, [&](int rank) {
        const int device = tpDevices[rank];
        FastllmCudaSetDevice(device);
        Executor executor;
        executor.SetFirstDevice("cuda:" + std::to_string(device));
        struct Restore {
            void *previous;
            ~Restore() { SetCurrentThreadExecutor(previous); }
        } restore{GetExecutor()};
        SetCurrentThreadExecutor(&executor);
        Data input(host);
        input.ToDevice(DataDevice::CUDA, std::vector<int>{device});
        Data &head = *weight["lm_head.weight"].multiDeviceDatas.at(device);
        if (rows > 1) MatMulTransB(input, head, logits[rank]);
        else Linear(input, head, Data(), logits[rank]);
        logits[rank].ToDevice(DataDevice::CPU);
    }, errors);
    for (auto error : errors) if (error) std::rethrow_exception(error);
    Data output(BFLOAT16, {1, rows, vocab});
    output.Allocate();
    for (int rank = 0; rank < (int)tpDevices.size(); ++rank) {
        const auto range = tpVocabRanges[rank];
        const int localVocab = range.second - range.first;
        for (int row = 0; row < rows; ++row)
            std::memcpy((uint16_t *)output.cpuData + (size_t)row * vocab + range.first,
                        (uint16_t *)logits[rank].cpuData + (size_t)row * localVocab,
                        localVocab * sizeof(uint16_t));
    }
    ApplyDraftDevice();
    output.ToDevice(DataDevice::CUDA, std::vector<int>{tpDevices.front()});
    return output;
#else
    return Data();
#endif
}

}
