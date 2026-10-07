#include "naive_n05_flash.h"
#include "naive_n05_tp.h"
#include "blocks/baseblock.h"
#include "executor.h"
#include "json11.hpp"
#include "utils.h"
#include <cmath>
#include <climits>
#include "devices/cuda/naive-n05-cuda.cuh"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
namespace {
    float ConfigNumber(const WeightMap &weight, const std::string &key, float fallback) {
        auto it = weight.dicts.find(key);
        return it == weight.dicts.end() || it->second == "null"
                   ? fallback : std::stof(it->second);
    }
    std::vector<int> ConfigList(const WeightMap &weight, const std::string &key, int size) {
        auto it = weight.dicts.find(key);
        AssertInFastLLM(it != weight.dicts.end(), "Naive-N0.5 requires " + key);
        std::string error;
        auto values = json11::Json::parse(it->second, error);
        AssertInFastLLM(error.empty() && (int)values.array_items().size() == size,
                        "Invalid Naive-N0.5 " + key);
        std::vector<int> result;
        for (auto &value : values.array_items()) result.push_back(value.int_value());
        return result;
    }
}

int NaiveN05FlashModel::CacheReserveCapacity(const GenerationConfig &config) const {
    // Like ChatGLM, reserve the request's prompt and bounded output in one
    // allocation. Chunked prefill must use the full prompt length supplied by
    // GenerationConfig, not the current chunk. Warmup/direct calls without
    // this metadata keep the existing incremental allocation path.
    if (config.input_token_length <= 0) return 0;
    int64_t capacity = config.input_token_length;
    if (config.output_token_limit > 0) capacity += (int64_t)config.output_token_limit - 1;
    int limit = max_positions;
    if (tokensLimit > 0) limit = std::min(limit, tokensLimit);
    if (GetMaxTokens() > 0) limit = std::min(limit, GetMaxTokens());
    return (int)std::min<int64_t>(capacity, limit);
}

void NaiveN05FlashModel::AppendCache(Data &cache, Data &input, int reserveCapacity) {
    int oldLength = cache.dims.empty() ? 0 : cache.dims[1];
    int length = oldLength + input.dims[1];
    cache.dataType = input.dataType;
    cache.UpdateUnitSize();
    cache.ToDevice(input.dataDevice, input.dataDeviceIds);
    const int64_t wanted = std::max(length, reserveCapacity);
    if (cache.expansionDims.empty() || cache.expansionDims[1] < wanted) {
        const int capacity = (int)std::min<int64_t>(INT_MAX, (wanted + 127) / 128 * 128);
        cache.Expansion({1, capacity, input.dims[2]});
    }
    CatDirect(cache, input, 1);
    cache.isKVCache = true;
}

void NaiveN05FlashModel::TrimCache(Data &cache, int length) {
    if (cache.dims[1] <= length) return;
    Data suffix;
    Split(cache, 1, cache.dims[1] - length, cache.dims[1], suffix);
    cache.CopyFrom(suffix);
    cache.isKVCache = true;
}

NaiveN05FlashModel::NaiveN05FlashModel() {
    model_type = model_struct = "naive_n05_flash";
    canDoBatchForward = false;
    // Feed enough rows to each of the 256 experts to reuse its weights.
    // 2048 also matches the sparse attention top-k boundary.
    defaultChunkedPrefillSize = 2048;
    dataType = kvCacheDataType = moeAtype = DataType::BFLOAT16;
    pre_prompt = user_role = bot_role = history_sep = "";
    weight.embeddingNames.insert("model.embed_tokens.weight");
    weight.linearNames = {
        "lm_head.weight", "model.layers.*.self_attn.*_proj.weight",
        "model.layers.*.self_attn.indexer.wq.weight",
        "model.layers.*.self_attn.indexer.wk.weight",
        "model.layers.*.mlp.gate.weight",
        "model.layers.*.mlp*gate_proj.weight",
        "model.layers.*.mlp*up_proj.weight",
        "model.layers.*.mlp*down_proj.weight"
    };
}

NaiveN05FlashModel::~NaiveN05FlashModel() {
    ShutdownRuntime();
    tpWorkers.Stop();
    tpDecodeState.reset();
    tpVerifyState.reset();
    tpBatchState.reset();
}

void NaiveN05FlashModel::InitParams() {
    basellm::InitParams();
    auto number = [&](const std::string &name, float fallback) {
        return ConfigNumber(weight, name, fallback);
    };
    auto attention = [&](const std::string &prefix) {
        return AttentionConfig{
            (int)number(prefix + "num_attention_heads", 64),
            (int)number(prefix + "num_key_value_heads", prefix.empty() ? 4 : 8),
            (int)number(prefix + "head_dim", 192),
            (int)number(prefix + "v_head_dim", 128),
            number(prefix + "rope_theta", prefix.empty() ? 1e7f : 1e4f)};
    };
    full = attention("");
    sliding = attention("swa_");
    head_dim = full.headDim;
    num_key_value_heads = full.kvHeads;
    rotary_dim = (int)(head_dim * number("partial_rotary_factor", 0.334f));
    slidingLayers = ConfigList(weight, "hybrid_layer_pattern", block_cnt);
    moeLayers = ConfigList(weight, "moe_layer_freq", block_cnt);
    window = number("sliding_window", 128);
    indexHeads = number("index_n_heads", 16);
    indexDim = number("index_head_dim", 128);
    indexTopK = number("index_top_k", 2048);
    partialRotary = number("partial_rotary_factor", 0.334f);
    valueScale = number("attention_value_scale", 1.0f);
    rms_norm_eps = number("layernorm_epsilon", 1e-5f);
    num_experts = number("n_routed_experts", 256);
    num_experts_per_tok = number("num_experts_per_tok", 8);
    routed_scaling_factor = number("routed_scaling_factor", 1.0f);
    norm_topk_prob = weight.dicts["norm_topk_prob"] != "false";
    max_positions = number("max_position_embeddings", 1048576);
    indexFp8 = weight.dicts["indexer_activation_dtype"] != "bf16";
    InitDraft();
    InitTensorParallel();
    canDoBatchForward = !draftEnabled && tpDevices.size() > 1;
    historyBytesPerToken = draftEnabled ? embed_dim * sizeof(uint16_t) : 0;
    for (int layer = 0; layer < block_cnt; ++layer) {
        const auto &cfg = slidingLayers[layer] ? sliding : full;
        historyBytesPerToken += sizeof(uint16_t) *
            (cfg.kvHeads * (cfg.headDim + cfg.valueDim) + (slidingLayers[layer] ? 0 : indexDim));
    }
    AssertInFastLLM(window > 0 && indexDim == 128 && indexTopK > 0 &&
                    full.heads % full.kvHeads == 0 && sliding.heads % sliding.kvHeads == 0 &&
                    (int)(full.headDim * partialRotary) <= indexDim &&
                    weight.dicts["scoring_func"] == "sigmoid" &&
                    number("n_group", 1) == 1 && number("topk_group", 1) == 1,
                    "Unsupported Naive-N0.5 attention/indexer/router configuration.");
    if (!useCustomMoeAtype) moeAtype = DataType::BFLOAT16;
    for (int layer = 0; layer < block_cnt; layer++) {
        const std::string ap = "model.layers." + std::to_string(layer) + ".self_attn.";
        weightMergeRules.push_back(WeightMergeRule({
            WeightMergeRuleSingle({ap + "q_proj.weight", ap + "k_proj.weight", ap + "v_proj.weight"},
                                  ap + "mergeqkv.weight", "linear"),
            WeightMergeRuleSingle({ap + "q_proj.bias", ap + "k_proj.bias", ap + "v_proj.bias"},
                                  ap + "mergeqkv.bias", "bias")}));
        if (!moeLayers[layer]) continue;
        for (int expert = 0; expert < num_experts; expert++) {
            std::string base = "model.layers." + std::to_string(layer) +
                               ".mlp.experts." + std::to_string(expert) + ".";
            weightMergeRules.push_back(WeightMergeRule({WeightMergeRuleSingle(
                {base + "gate_proj.weight", base + "up_proj.weight"},
                base + "gateup_proj.weight", "linearSwiglu")}));
            AddSpecialWeight(base + "gateup_proj.weight", "linearSwiglu", layer);
            AddSpecialWeight(base + "down_proj.weight", "linearColumn", layer);
            for (auto kind : {"gate", "up", "down"})
                moeLinears.insert(base + kind + "_proj.weight");
        }
    }
}

std::map<std::string, std::vector<std::pair<std::string, DataType>>>
NaiveN05FlashModel::GetTensorMap(const std::vector<std::string> &names) {
    auto result = basellm::GetTensorMap(names);
    const bool compactNvfp4 = std::any_of(names.begin(), names.end(), [](const std::string &name) {
        return name.find(".mlp.experts.") != std::string::npos &&
               (StringEndWith(name, ".weight_scale") || StringEndWith(name, ".weight_scale_2"));
    });
    auto usesCuda = [](const std::map<std::string, int> &devices) {
        return std::any_of(devices.begin(), devices.end(), [](const auto &device) {
            return device.first.rfind("cuda", 0) == 0 ||
                   device.first.rfind("multicuda", 0) == 0;
        });
    };
    // Keep original E4M3 scales in the packed CUDA source layout. Grouped
    // Marlin validates and replaces it once during warmup; unsupported
    // layouts retain the native kernels and independent gate/up globals.
    // CPU/NUMA keeps the original compact checkpoint layout.
    const bool cudaExperts = usesCuda(moeDeviceMap.empty() ? deviceMap : moeDeviceMap) ||
                            (moeDeviceLayers >= 0 && usesCuda(layeredMoeDeviceMap));
    const DataType nvfp4Type = cudaExperts ? DataType::NVFP4_BLOCK_16_E4M3_PACKED :
                                          DataType::NVFP4_BLOCK_16_E4M3;
    for (auto &name : names) {
        if (draftEnabled && (name.rfind("layers.", 0) == 0 ||
                name.rfind("markov_head.", 0) == 0 || name.rfind("confidence_head.", 0) == 0 ||
                name == "fc.weight" || name == "hidden_norm.weight" ||
                name == "norm.weight" || name == "mask_embedding")) {
            auto type = (name.find("norm.weight") != std::string::npos ||
                         name.rfind("confidence_head.", 0) == 0) ? DataType::FLOAT32 : DataType::BFLOAT16;
            result[name] = {{"dspark." + name, type}};
        } else if (compactNvfp4 && moeLinears.count(name)) {
            result[name] = {{name, nvfp4Type}};
        } else if (name.find(".mlp.gate.") != std::string::npos) {
            result[name] = {{name, DataType::FLOAT32}};
        } else if (name.find(".mlp.experts.") == std::string::npos &&
                   (weight.GetWeightType(name) == WeightType::LINEAR ||
                    weight.GetWeightType(name) == WeightType::EMBEDDING)) {
            result[name] = {{name, DataType::BFLOAT16}};
        }
    }
    return result;
}

int NaiveN05FlashModel::GetKVCacheRetainedTokens(int layer) const {
    return slidingLayers.at(layer) ? window - 1 : -1;
}

int NaiveN05FlashModel::Forward(
        const Data &inputIds, const Data &attentionMask, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &pastKeyValues,
        const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
        std::vector<float> *retLogits) {
    if (draftEnabled)
        return ForwardDraft(inputIds, positionIds, pastKeyValues, generationConfig, lastTokens, retLogits);
    auto selection = SelectLogits(generationConfig);
    Data logits = tpDevices.size() > 1
        ? ForwardTensorParallel(inputIds, positionIds, pastKeyValues, generationConfig, nullptr, &selection)
        : RunTarget(inputIds, positionIds, pastKeyValues, generationConfig, nullptr);
    if (isIntermediateChunkedPrefill) return 0;
    return SampleTarget(logits, pastKeyValues, generationConfig, lastTokens, retLogits, &selection);
}

std::vector<int> NaiveN05FlashModel::ForwardBatch(int batch, const Data &inputIds,
        const std::vector<Data *> &attentionMask, const std::vector<Data *> &positionIds,
        const std::vector<int> &seqLens, std::vector<std::pair<Data *, Data *>> &pastKeyValues,
        const std::vector<GenerationConfig> &configs, const LastTokensManager &lastTokens,
        std::vector<std::vector<float> *> *retLogits) {
    AssertInFastLLM(!draftEnabled && tpDevices.size() > 1 && batch > 0 &&
        (int)seqLens.size() == batch && (int)configs.size() == batch &&
        (int)positionIds.size() == batch && (int)pastKeyValues.size() == batch * block_cnt,
        "Naive batch forward requires ordinary TP inference and complete request descriptors.");
    TargetBatch requests{pastKeyValues, configs, seqLens, block_cnt};
    std::vector<float> packedPositions;
    auto selection = SelectLogits(configs[0]);
    for (int b = 0; b < batch; ++b) {
        AssertInFastLLM(seqLens[b] > 0 && positionIds[b] &&
            (b >= (int)attentionMask.size() || !attentionMask[b] || attentionMask[b]->dims.empty()),
            "Naive batch expects unpadded sequences with explicit positions.");
        for (int layer = 0; layer < block_cnt; ++layer)
            AssertInFastLLM(pastKeyValues[b * block_cnt + layer].first &&
                pastKeyValues[b * block_cnt + layer].second, "Naive batch has a null KV descriptor.");
        const auto &key = requests.Key(b, 0);
        const int past = key.dims.empty() ? 0 : key.dims[1];
        AssertInFastLLM((int64_t)past + seqLens[b] <= max_positions,
            "Naive batch request exceeds the context window.");
        Data positions(*positionIds[b]);
        ToDataType(positions, FLOAT32);
        positions.ToDevice(DataDevice::CPU);
        AssertInFastLLM(positions.Count(0) == seqLens[b], "Naive batch position count mismatch.");
        const float *values = (const float *)positions.cpuData;
        packedPositions.insert(packedPositions.end(), values, values + seqLens[b]);
        auto other = SelectLogits(configs[b]);
        if (selection.count != other.count || selection.greedy != other.greedy ||
            selection.invTemperature != other.invTemperature) selection.count = 0;
    }
    AssertInFastLLM(inputIds.dims == std::vector<int>({1, (int)packedPositions.size()}) &&
        inputIds.dataType == FLOAT32, "Naive batch input must pack the unpadded sequences in order.");
    Data positions(FLOAT32, inputIds.dims, packedPositions);
    // The batch descriptor references response-owned KV. No KV payload or
    // ownership is copied into the legacy single-sequence argument.
    std::vector<std::pair<Data, Data>> unused;
    Data logits = ForwardTensorParallel(inputIds, positions, unused, configs[0],
                                        nullptr, &selection, &requests);
    std::vector<int> tokens(batch);
    if (!selection.candidates.dims.empty()) {
        for (int b = 0; b < batch; ++b)
            tokens[b] = selection.greedy
                ? (int)((float *)selection.candidates.cpuData)[b * 2]
                : LLMSamplingOnly(selection.candidates, b, configs[b]);
        return tokens;
    }
    const int vocab = logits.dims.back();
    for (int b = 0; b < batch; ++b)
        if (configs[b].output_logits && retLogits && b < (int)retLogits->size() && (*retLogits)[b]) {
            const float *row = (const float *)logits.cpuData + (size_t)b * vocab;
            (*retLogits)[b]->assign(row, row + vocab);
        }
    ResetLogitsOfEOS(batch, &logits, pastKeyValues, configs);
    for (int b = 0; b < batch; ++b) {
        if (configs[b].IsSimpleGreedy()) {
            const float *row = (const float *)logits.cpuData + (size_t)b * vocab;
            // The owning TP result is already on CPU. Preserve CUDA Top1's
            // tie order without asking the executor to move a borrowed view.
            float best = -INFINITY;
            for (int id = 0; id < vocab; ++id)
                if (row[id] > -INFINITY && FastllmNaiveTop1Better(row[id], id, best, tokens[b])) {
                    best = row[id];
                    tokens[b] = id;
                }
        } else {
            LastTokensUnit empty;
            tokens[b] = LLMSampling(logits, b, configs[b],
                b < (int)lastTokens.units.size() ? lastTokens.units[b] : empty);
        }
    }
    return tokens;
}

Data NaiveN05FlashModel::RunTarget(
        const Data &inputIds, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &pastKeyValues, const GenerationConfig &config,
        TargetCapture *capture, int tpRank, const Data *embedding, TargetWorkspace *workspace,
        const TargetBatch *batch) {
#ifndef USE_CUDA
    ErrorInFastLLM("Naive-N0.5 currently requires the CUDA backend for attention.");
    return Data();
#else
    AssertInFastLLM(dataType == DataType::BFLOAT16 && kvCacheDataType == DataType::BFLOAT16,
                    "Naive-N0.5 requires BF16 activations and KV cache (use auto or bfloat16).");
    AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1 &&
                    (batch || (int)pastKeyValues.size() == block_cnt),
                    "Naive-N0.5 expects one unpadded sequence and a complete KV cache.");
    const bool tensorParallel = tpRank >= 0;
    const bool decodeWorkspace = workspace && workspace->capacity > 0;
    const bool graphVerify = decodeWorkspace && capture && capture->verifying;
    const int gpu = tensorParallel ? tpDevices.at(tpRank) : -1;
    Data emptyWeight;
    auto localWeight = [&](const std::string &name) -> Data & {
        if (!tensorParallel) return weight[name];
        auto it = weight.weight.find(name);
        if (it == weight.weight.end() || it->second.dims.empty()) return emptyWeight;
        return *it->second.multiDeviceDatas.at(gpu);
    };
    auto &moeWeights = tensorParallel ? tpMoeWeights.at(tpRank) : this->moeWeights;
    auto &moeBiases = tensorParallel ? tpMoeBiases.at(tpRank) : this->moeBiases;
    size_t communication = 0;
    auto reduce = [&](Data &data) {
        if (decodeWorkspace) {
            auto &state = *tpDecodeState;
            auto &rank = *state.ranks.at(tpRank);
            if (state.mode == TPDecodeState::Warm)
                rank.communicationPointers.push_back(data.cudaData);
            else if (communication >= rank.communicationPointers.size() ||
                     rank.communicationPointers[communication] != data.cudaData)
                rank.ok = false;
            ++communication;
        }
        if (!tensorParallel) return;
        if (((batch && batch->Decode()) || (capture && capture->verifying)) && inputIds.dims[1] > 1) {
            if (!FastllmCudaCustomAllReduceRows(data.cudaData, data.cudaData,
                    data.Count(0), embed_dim, data.dataType, gpu)) {
                // An unsupported/disabled custom path must retain ordinary
                // single-token NCCL arithmetic, including its message size.
                for (int row = 0; row < inputIds.dims[1]; ++row) {
                    void *ptr = (uint8_t *)data.cudaData + (size_t)row * embed_dim * data.unitSize;
                    FastllmNcclAllReduce(ptr, ptr, embed_dim, data.dataType, gpu);
                }
            }
        } else FastllmNcclAllReduce(data.cudaData, data.cudaData, data.Count(0), data.dataType, gpu);
    };
    int length = inputIds.dims[1];
    const int reserveCapacity = CacheReserveCapacity(config);
    const int previousExactThreshold = FastllmCudaGetLinearExactBatchThreshold();
    struct RestoreExactThreshold {
        int value;
        ~RestoreExactThreshold() { FastllmCudaSetLinearExactBatchThreshold(value); }
    } restoreExactThreshold{previousExactThreshold};
    // Keep the decode reduction order for paths honoring this threshold,
    // especially the FP32 router where rounding can change expert selection.
    // BF16-to-BF16 batches of eight or more rows use cuBLAS and allow bounded
    // floating-point differences from independent single-row decoding.
    if ((batch && batch->Decode()) || (capture && capture->verifying))
        FastllmCudaSetLinearExactBatchThreshold(std::max(previousExactThreshold, length + 1));
    Data &firstKey = batch ? batch->Key(0, 0) : pastKeyValues[0].first;
    int pastLength = firstKey.dims.empty() ? 0 : firstKey.dims[1];
    auto historyChunk = tensorParallel ? nullptr : BeginHistoryChunk(pastKeyValues, pastLength, length);
    if (capture) capture->history = historyChunk;
    AssertInFastLLM(!slidingLayers[0] && (batch || pastLength + length <= max_positions),
                    "Naive-N0.5 requires a DSA first layer and input within the context window.");
    if (moeWeights.empty()) {
        moeWeights.resize(block_cnt);
        moeBiases.resize(block_cnt);
        for (int layer = 0; layer < block_cnt; layer++) {
            if (!moeLayers[layer]) continue;
            // MergeMOE reserves the first pair for a shared expert.
            moeWeights[layer] = {nullptr, nullptr};
            for (int expert = 0; expert < num_experts; expert++) {
                std::string base = "model.layers." + std::to_string(layer) +
                                   ".mlp.experts." + std::to_string(expert) + ".";
                moeWeights[layer].push_back(&localWeight(base + "gateup_proj.weight"));
                moeWeights[layer].push_back(&localWeight(base + "down_proj.weight"));
            }
            moeBiases[layer].resize(moeWeights[layer].size(), nullptr);
        }
    }
    if (!tensorParallel) ApplyDeviceMap(deviceMap, 1, block_cnt);
    TargetWorkspace temporary;
    TargetWorkspace &buf = workspace ? *workspace : temporary;
    Data &hidden = buf.hidden;
    Data &normed = buf.normed;
    Data &q = buf.q;
    Data &k = buf.k;
    Data &v = buf.v;
    Data &qkv = buf.qkv;
    Data &attn = buf.attn;
    Data &projected = buf.projected;
    Data &routerInput = buf.routerInput;
    Data &router = buf.router;
    Data &expertIndex = buf.expertIndex;
    Data &expertScore = buf.expertScore;
    Data &w1 = buf.w1;
    Data &w2 = buf.w2;
    Data &w3 = buf.w3;
    Data &tempInput = buf.tempInput;
    Data &tempOutput = buf.tempOutput;
    Data &moeOutput = buf.moeOutput;
    Data &moeInputTemp = buf.moeInputTemp;
    Data &moeOutputTemp = buf.moeOutputTemp;
    Data &positions = buf.positions;
    if (!decodeWorkspace) {
        positions.CopyFrom(positionIds);
        ToDataType(positions, DataType::FLOAT32);
    }
    if (embedding) hidden.CopyFrom(*embedding);
    // A persistent verification workspace keeps BF16 storage. Embedding's
    // FP32 intermediate cannot reuse that allocation at the same element count.
    else if (decodeWorkspace || (workspace && GetCudaEmbedding() && !GetLowMemMode()))
        EmbeddingDirect(inputIds, localWeight("model.embed_tokens.weight"), hidden);
    else Embedding(inputIds, localWeight("model.embed_tokens.weight"), hidden);
    ToDataType(hidden, DataType::BFLOAT16);
    auto norm = [&](Data &input, Data &weight, Data &output) {
        if (!decodeWorkspace) {
            KimiK3RMSNorm(input, weight, rms_norm_eps, output);
            return;
        }
        output.dataType = DataType::BFLOAT16;
        output.Resize(input.dims);
        output.ToDevice(DataDevice::CUDA, {gpu}, false);
        output.Allocate(false);
        if (!FastllmCudaKimiK3RMSNorm(input, weight, output, rms_norm_eps)) {
            if (FastllmCudaGraphIsCapturingFast()) FastllmCudaSetThreadError();
            else AssertInFastLLM(false, "Naive TP RMSNorm launch failed.");
        }
    };
    for (int layer = 0; layer < block_cnt; layer++) {
        if (!tensorParallel) ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
        std::string prefix = "model.layers." + std::to_string(layer);
        std::string ap = prefix + ".self_attn.";
        auto cfg = slidingLayers[layer] ? sliding : full;
        if (tensorParallel) {
            cfg.heads /= tpDevices.size();
            cfg.kvHeads = std::max(1, cfg.kvHeads / (int)tpDevices.size());
        }
        int rotaryDim = (int)(cfg.headDim * partialRotary);
        // This RMSNorm rounds the normalized activation before multiplying
        // the affine weight, matching the checkpoint's LlamaRMSNorm.
        if (!decodeWorkspace || layer == 0)
            norm(hidden, localWeight(prefix + ".input_layernorm.weight"), normed);
        const auto merged = weight.weight.find(ap + "mergeqkv.weight");
        const bool mergedQkv = merged != weight.weight.end() && !merged->second.dims.empty();
        if (mergedQkv) {
            Linear(normed, localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"), qkv);
            // Q/K and V have different head widths. Read packed projection rows
            // directly in the RoPE/cache kernel instead of launching three Splits.
            for (auto item : {std::make_pair(&q, cfg.heads * cfg.headDim),
                              std::make_pair(&k, cfg.kvHeads * cfg.headDim),
                              std::make_pair(&v, cfg.kvHeads * cfg.valueDim)}) {
                item.first->dataType = qkv.dataType;
                item.first->Resize({1, length, item.second});
                item.first->ToDevice(qkv.dataDevice, qkv.dataDeviceIds, false);
                item.first->Allocate(false);
            }
        } else {
            Linear(normed, localWeight(ap + "q_proj.weight"), localWeight(ap + "q_proj.bias"), q);
            Linear(normed, localWeight(ap + "k_proj.weight"), localWeight(ap + "k_proj.bias"), k);
            Linear(normed, localWeight(ap + "v_proj.weight"), localWeight(ap + "v_proj.bias"), v);
        }
        AssertInFastLLM(q.dataDevice == DataDevice::CUDA,
                        "Naive-N0.5 attention requires --device cuda.");
        positions.ToDevice(q.dataDevice, q.dataDeviceIds);
        auto attend = [&](Data &normed, Data &q, Data &k, Data &v, Data &qkv,
                          Data &positions, Data &pastKey, Data &pastValue,
                          TargetWorkspace &buf, int length, int reserveCapacity, Data &attn) {
            Data &packed = buf.packed, &indexKey = buf.indexKey, &indexQ = buf.indexQ;
            Data &indexWeights = buf.indexWeights, &indices = buf.indices;
            int localPast = pastKey.dims.empty() ? 0 : pastKey.dims[1];
            // Sliding layers retain only window-1 rows between chunks. Keep their
            // reservation bounded even when the full request is very long.
            const int layerCapacity = slidingLayers[layer]
                ? (int)std::min<int64_t>(reserveCapacity, (int64_t)window - 1 + length)
                : reserveCapacity;
            if (!slidingLayers[layer]) {
                std::string ip = ap + "indexer.";
                Linear(normed, localWeight(ip + "wk.weight"), Data(), indexKey);
                // LayerNorm accumulates in FP32; the generic CUDA operation does
                // not accept BF16 storage, so round only its final result.
                if (decodeWorkspace) {
                    ToDataType(indexKey, buf.indexKeyFloat, DataType::FLOAT32);
                    LayerNorm(buf.indexKeyFloat, localWeight(ip + "k_norm.weight"),
                              localWeight(ip + "k_norm.bias"), -1, buf.indexKeyFloat);
                    ToDataType(buf.indexKeyFloat, indexKey, DataType::BFLOAT16);
                } else {
                    ToDataType(indexKey, DataType::FLOAT32);
                    LayerNorm(indexKey, localWeight(ip + "k_norm.weight"), localWeight(ip + "k_norm.bias"), -1, indexKey);
                    ToDataType(indexKey, DataType::BFLOAT16);
                }
            }
            Data noIndexKey, noLiveKeys;
            bool fusedCache = !historyChunk &&
                (decodeWorkspace || (!tensorParallel && length <= 8 && localPast > 0)) &&
                FastllmCudaNaiveRopeAppendCache(q, k, v, slidingLayers[layer] ? noIndexKey : indexKey,
                    positions, pastKey, pastValue, decodeWorkspace ? buf.liveKeys : noLiveKeys, cfg.heads, cfg.kvHeads,
                    cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale,
                    slidingLayers[layer] ? window : 0, mergedQkv ? &qkv : nullptr);
            if (fusedCache && !decodeWorkspace) {
                pastKey.Resize({1, localPast + length, pastKey.dims[2]});
                pastValue.Resize({1, localPast + length, pastValue.dims[2]});
            }
            if (!fusedCache) {
                FastllmCudaNaiveRopeQKScaleV(q, k, v, positions, cfg.heads, cfg.kvHeads,
                    cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale, mergedQkv ? &qkv : nullptr);
                if (!slidingLayers[layer]) {
                    FastllmCudaNaiveRope(indexKey, positions, 1, indexDim, rotaryDim, cfg.theta);
                    Cat(k, indexKey, 2, packed);
                } else {
                    packed.CopyFrom(k);
                }
                if (historyChunk) {
                    CopyHistoryTensor(packed, historyChunk->layers[layer].first, historyChunk->length);
                    CopyHistoryTensor(v, historyChunk->layers[layer].second, historyChunk->length);
                }
                if (decodeWorkspace) {
                    if (graphVerify) FastllmCudaNaiveAppendVerifyCache(pastKey, pastValue, packed, v,
                        buf.liveKeys, slidingLayers[layer] ? window : 0);
                    else FastllmCudaNaiveAppendDecodeCache(pastKey, pastValue, packed, v,
                        buf.liveKeys, slidingLayers[layer] ? window : 0);
                } else {
                    AppendCache(pastKey, packed, layerCapacity);
                    AppendCache(pastValue, v, layerCapacity);
                }
            }
            Data noIndices;
            Data *selected = &noIndices;
            if (!slidingLayers[layer] && (decodeWorkspace ? buf.capacity : pastKey.dims[1]) > indexTopK) {
                std::string ip = ap + "indexer.";
                Linear(normed, localWeight(ip + "wq.weight"), Data(), indexQ);
                FastllmCudaNaiveRope(indexQ, positions, indexHeads, indexDim, rotaryDim, cfg.theta);
                Linear(normed, localWeight(ip + "weights_proj.weight"), Data(), indexWeights);
                Mul(indexWeights, 1.0f / std::sqrt((float)indexHeads), indexWeights);
                if (decodeWorkspace) {
                    if (graphVerify) FastllmCudaNaiveGraphVerifyIndexer(indexQ, indexWeights, pastKey,
                        buf.liveKeys, buf.capacity, indexFp8, buf.decode, indices);
                    else FastllmCudaNaiveDecodeIndexer(indexQ, indexWeights, pastKey,
                        buf.liveKeys, buf.capacity, indexFp8, buf.decode, indices);
                } else {
                    if (capture && capture->verifying && length <= 8)
                        FastllmCudaNaiveVerifyIndexer(indexQ, indexWeights, pastKey, indexHeads,
                            indexDim, localPast, indexTopK, indexFp8, indices);
                    else FastllmCudaNaiveIndexer(indexQ, indexWeights, pastKey, indexHeads, indexDim,
                                                localPast, indexTopK, indexFp8, indices);
                }
                selected = &indices;
            }
            Data &sink = localWeight(ap + "attention_sink_bias");
            if (!sink.dims.empty()) {
                ToDataType(sink, DataType::FLOAT32);
                sink.ToDevice(q.dataDevice, q.dataDeviceIds);
            }
            if (decodeWorkspace) {
                if (graphVerify) FastllmCudaNaiveGraphVerifyAttention(q, pastKey, pastValue, *selected, sink,
                    buf.liveKeys, buf.capacity, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                    slidingLayers[layer] ? window : 0, buf.decode, attn);
                else FastllmCudaNaiveDecodeAttention(q, pastKey, pastValue, *selected, sink,
                    buf.liveKeys, buf.capacity, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                    slidingLayers[layer] ? window : 0, buf.decode, attn);
                if (slidingLayers[layer] && !graphVerify)
                    FastllmCudaNaiveTrimDecodeCache(pastKey, pastValue, buf.liveKeys, window);
            } else {
                if (capture && capture->verifying && length <= 8)
                    FastllmCudaNaiveVerifyAttention(q, pastKey, pastValue, *selected, sink,
                        cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                        localPast, slidingLayers[layer] ? window : 0, attn);
                else FastllmCudaNaiveAttention(q, pastKey, pastValue, *selected, sink,
                                               cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                                               localPast, slidingLayers[layer] ? window : 0, attn);
                if (slidingLayers[layer] && (!capture || !capture->verifying))
                    FastllmCudaNaiveTrimCache(pastKey, pastValue, window - 1);
            }
        };
        if (batch) {
            while ((int)buf.sequences.size() < batch->Size())
                buf.sequences.emplace_back(new TargetWorkspace());
            attn.dataType = BFLOAT16;
            attn.Resize({1, length, cfg.heads * cfg.valueDim});
            attn.ToDevice(q.dataDevice, q.dataDeviceIds, false);
            attn.Allocate(false);
            int offset = 0;
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                const int count = batch->lengths[sequence];
                auto view = [&](Data &dst, Data &src, int width) {
                    dst.Resize({1, count, width});
                    dst.FakeFrom(src, (size_t)offset * width * src.unitSize);
                };
                Data rowNorm, rowQ, rowK, rowV, rowQkv, rowPositions, rowAttention;
                view(rowNorm, normed, embed_dim);
                view(rowQ, q, cfg.heads * cfg.headDim);
                view(rowK, k, cfg.kvHeads * cfg.headDim);
                view(rowV, v, cfg.kvHeads * cfg.valueDim);
                if (mergedQkv) view(rowQkv, qkv, qkv.dims.back());
                view(rowAttention, attn, cfg.heads * cfg.valueDim);
                rowPositions.Resize({1, count});
                rowPositions.FakeFrom(positions, (size_t)offset * sizeof(float));
                auto &part = *buf.sequences[sequence];
                if (decodeWorkspace) {
                    part.liveKeys.Resize({1});
                    part.liveKeys.FakeFrom(buf.liveKeys, sequence * sizeof(int));
                }
                attend(rowNorm, rowQ, rowK, rowV, rowQkv, rowPositions,
                       *batch->Key(sequence, layer).multiDeviceDatas.at(gpu),
                       *batch->Value(sequence, layer).multiDeviceDatas.at(gpu), part, count,
                       CacheReserveCapacity(batch->configs[sequence]), rowAttention);
                offset += count;
            }
        } else {
            auto &pastKey = tensorParallel ? *pastKeyValues[layer].first.multiDeviceDatas.at(gpu) : pastKeyValues[layer].first;
            auto &pastValue = tensorParallel ? *pastKeyValues[layer].second.multiDeviceDatas.at(gpu) : pastKeyValues[layer].second;
            attend(normed, q, k, v, qkv, positions, pastKey, pastValue, buf, length, reserveCapacity, attn);
        }
        Linear(attn, localWeight(ap + "o_proj.weight"), Data(), projected);
        reduce(projected);
        if (decodeWorkspace) {
            FastllmCudaNaiveAddDecodeRMSNorm(hidden, projected,
                localWeight(prefix + ".post_attention_layernorm.weight"), rms_norm_eps, normed);
        } else {
            AddTo(hidden, projected);
            norm(hidden, localWeight(prefix + ".post_attention_layernorm.weight"), normed);
        }
        auto addMlpResidual = [&](Data &branch) {
            if (!decodeWorkspace) { AddTo(hidden, branch); return; }
            const bool lastLayer = layer + 1 == block_cnt;
            const std::string name = lastLayer ? "model.norm.weight" :
                "model.layers." + std::to_string(layer + 1) + ".input_layernorm.weight";
            FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, localWeight(name),
                rms_norm_eps, lastLayer ? buf.last : normed);
        };
        if (!moeLayers[layer]) {
            // Dense and routed-expert scratch have independent lifetimes.
            Data &w1 = workspace ? buf.denseGate : buf.w1;
            Data &w2 = workspace ? buf.denseDown : buf.w2;
            Data &w3 = workspace ? buf.denseUp : buf.w3;
            Linear(normed, localWeight(prefix + ".mlp.gate_proj.weight"), Data(), w1);
            Linear(normed, localWeight(prefix + ".mlp.up_proj.weight"), Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, localWeight(prefix + ".mlp.down_proj.weight"), Data(), w2);
            reduce(w2);
            addMlpResidual(w2);
        } else {
            Data &routerWeight = localWeight(prefix + ".mlp.gate.weight");
            bool hasRouterProbabilities = length == 1 &&
                FastllmCudaNaiveRouterSigmoid(normed, routerWeight, router);
            if (!hasRouterProbabilities &&
                !FastllmCudaNaiveRouterVerify(normed, routerWeight, router)) {
                ToDataType(normed, routerInput, DataType::FLOAT32);
                Linear(routerInput, routerWeight, Data(), router);
            }
            Data &routerBias = localWeight(prefix + ".mlp.gate.e_score_correction_bias");
            auto &executor = *(Executor *)GetExecutor();
            bool fusedRouter = false;
            // The warp-fused sigmoid amortizes selection for multiple rows;
            // a single row already has sigmoid fused into its router projection.
            if (length > 1 && router.dataDevice == DataDevice::CUDA) {
                DataDict routerData = {{"logits", &router}, {"index", &expertIndex},
                                       {"score", &expertScore}, {"gateBias", &routerBias}};
                FloatDict routerFloats = {{"routeScale", routed_scaling_factor}};
                IntDict routerInts = {{"topk", num_experts_per_tok}, {"needNorm", norm_topk_prob ? 1 : 0}};
                fusedRouter = executor.CanRunOnFirstDevice(
                    "FusedSigmoidSelectExpert", routerData, routerFloats, routerInts);
                if (fusedRouter)
                    executor.Run("FusedSigmoidSelectExpert", routerData, routerFloats, routerInts);
            }
            if (!fusedRouter) {
                if (!hasRouterProbabilities) Sigmoid(router, router);
                SelectExpert(router, expertIndex, expertScore, num_experts_per_tok,
                             norm_topk_prob, routed_scaling_factor, &routerBias);
            }
            normed.Reshape({length, embed_dim});
            if (!tensorParallel) ApplyMoeDeviceMapForLayer(layer);
            if (executor.firstDevice.find("numa") == 0 &&
                (moeWeights[layer][2]->dataType == DataType::FP8_E4M3 ||
                 moeWeights[layer][2]->dataType == DataType::FP8_E4M3_BLOCK_128)) {
                // The released checkpoint uses W8A8 and rounds each expert
                // projection/activation/route product to BF16. Request that
                // arithmetic explicitly; ordinary NUMA MoE uses W8A16.
                executor.Run("MergeMOE", {
                    {"input", &normed}, {"index", &expertIndex}, {"score", &expertScore},
                    {"weights", (Data *)moeWeights[layer].data()},
                    {"biass", (Data *)moeBiases[layer].data()},
                    {"w1", &w1}, {"w2", &w2}, {"w3", &w3},
                    {"curInput", &tempInput}, {"curOutput", &tempOutput}, {"output", &moeOutput}
                }, {{"sharedScale", 0.0f}}, {
                    {"weights___batch", (int)moeWeights[layer].size()},
                    {"biass___batch", (int)moeBiases[layer].size()},
                    {"layer", layer}, {"gateType", (int)MoeGateSwiglu}, {"fp8EagerMode", 1}
                });
            } else if (((batch && batch->Decode()) || (capture && capture->verifying)) && length > 1 &&
                       (moeWeights[layer][2]->dataType == DataType::BFLOAT16 ||
                        moeWeights[layer][2]->dataType == DataType::NVFP4_BLOCK_16_E4M3_PACKED) &&
                       normed.dataDevice == DataDevice::CUDA) {
                // Expert batching can change BF16 accumulation order, including
                // Marlin's split-K schedule. Verify with decode arithmetic;
                // the following TP reduction still covers the entire block.
                moeOutput.dataType = BFLOAT16;
                moeOutput.Resize({length, embed_dim});
                moeOutput.ToDevice(normed.dataDevice, normed.dataDeviceIds);
                moeOutput.Allocate(false);
                const bool residentRows =
                    moeWeights[layer][2]->dataType == DataType::NVFP4_BLOCK_16_E4M3_PACKED &&
                    FastllmCudaNVFP4E4M3GroupedMoeSupported(normed.dataDeviceIds.at(0)) &&
                    FastllmCudaPrepareNVFP4E4M3Moe(moeWeights[layer].data(), moeWeights[layer].size());
                // The verify kernel uses full-K expert tiles. Ordinary decode
                // may split K, so keep its per-request reduction order here.
                const bool mergedRows = !batch && residentRows &&
                    FastllmCudaMergeMOENVFP4E4M3MarlinRows(normed, w1, w2, moeOutput,
                        moeWeights[layer].data(), moeWeights[layer].size(),
                        (const int32_t *)expertIndex.cudaData, (const float *)expertScore.cudaData,
                        length, num_experts_per_tok);
                for (int row = 0; !mergedRows && row < length; ++row) {
                    Data rowInput(BFLOAT16, {1, embed_dim});
                    Data rowIndex, rowScore, rowOutput;
                    rowInput.FakeFrom(normed, (size_t)row * embed_dim * 2);
                    if (residentRows) {
                        rowIndex.Resize({1, expertIndex.dims[1]});
                        rowScore.Resize({1, expertScore.dims[1]});
                        rowOutput.Resize({1, embed_dim});
                        rowIndex.FakeFrom(expertIndex, (size_t)row * expertIndex.GetBytes() / length);
                        rowScore.FakeFrom(expertScore, (size_t)row * expertScore.GetBytes() / length);
                        rowOutput.FakeFrom(moeOutput, (size_t)row * embed_dim * 2);
                    } else {
                        // Generic expert backends may move routing to CPU;
                        // those need owning tensors instead of device views.
                        Split(expertIndex, 0, row, row + 1, rowIndex);
                        Split(expertScore, 0, row, row + 1, rowScore);
                    }
                    MergeMOEBlock(&rowInput, &rowIndex, &rowScore, &moeWeights[layer],
                        &moeBiases[layer], &w1, &w2, &w3, &tempInput, &tempOutput,
                        0.0f, &rowOutput, layer, DataType::BFLOAT16, moeAtype,
                        &moeInputTemp, &moeOutputTemp);
                    if (!residentRows)
                        FastllmCudaCopyFromDeviceToDevice((uint16_t *)moeOutput.cudaData + (size_t)row * embed_dim,
                                                         rowOutput.cudaData, (size_t)embed_dim * 2);
                }
            } else {
                MergeMOEBlock(&normed, &expertIndex, &expertScore, &moeWeights[layer],
                              &moeBiases[layer], &w1, &w2, &w3, &tempInput, &tempOutput,
                              0.0f, &moeOutput, layer, DataType::BFLOAT16, moeAtype,
                              &moeInputTemp, &moeOutputTemp);
            }
            if (!tensorParallel) ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
            moeOutput.Reshape(hidden.dims);
            reduce(moeOutput);
            addMlpResidual(moeOutput);
        }
        if (capture && capture->collectHidden &&
            std::find(draftTargetLayers.begin(), draftTargetLayers.end(), layer) != draftTargetLayers.end())
            Copy(hidden, capture->hidden[layer]);
    }
    if (!tensorParallel && !capture) FinishHistoryChunk(pastKeyValues, historyChunk);
    if (isIntermediateChunkedPrefill) return Data();
    Data localLogits;
    Data &last = buf.last, &logits = decodeWorkspace ? buf.logits : localLogits;
    if (!decodeWorkspace) {
        if (batch) {
            last.dataType = BFLOAT16;
            last.Resize({1, batch->Size(), embed_dim});
            last.ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false);
            last.Allocate(false);
            int offset = 0;
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                offset += batch->lengths[sequence];
                FastllmCudaCopyFromDeviceToDevice((uint16_t *)last.cudaData + (size_t)sequence * embed_dim,
                    (uint16_t *)hidden.cudaData + (size_t)(offset - 1) * embed_dim, (size_t)embed_dim * sizeof(uint16_t));
            }
        } else if (capture && capture->verifying) Copy(hidden, last);
        else Split(hidden, 1, length - 1, length, last);
        norm(last, localWeight("model.norm.weight"), last);
    }
    if (decodeWorkspace) {
        // Conversions must not free addresses retained by the graph.
        Linear(last, localWeight("lm_head.weight"), Data(), buf.logitsBf16);
        ToDataType(buf.logitsBf16, logits, DataType::FLOAT32);
        return Data();
    }
    Linear(last, localWeight("lm_head.weight"), Data(), logits);
    ToDataType(logits, DataType::FLOAT32);
    return localLogits;
#endif
}

NaiveN05FlashModel::LogitsSelection NaiveN05FlashModel::SelectLogits(
        const GenerationConfig &config, bool speculative) const {
    LogitsSelection selection;
    // These transforms must happen before selection. Keep the established
    // full-logits path until their branch-local GPU equivalents are available.
    if (config.output_logits || config.output_token_least > 0 ||
        std::abs(config.repeat_penalty - 1.0f) > 1e-8f ||
        !config.tool_call_allowed_token_ids.empty()) return selection;
    selection.greedy = config.IsSimpleGreedy();
    if (selection.greedy) selection.count = 1;
    else if (config.top_k > 1 && config.top_k <= kNaiveLogitsMaxTopK &&
             std::isfinite(config.temperature) && config.temperature > 0 &&
             std::isfinite(1.0f / config.temperature)) {
        selection.count = config.top_k;
        // Ordinary LLMSampling ranks scaled scores; DSpark ranks raw scores
        // and applies temperature after subtracting the largest score.
        selection.invTemperature = speculative ? 1.0f : 1.0f / config.temperature;
    }
    return selection;
}

int NaiveN05FlashModel::SampleTarget(
        Data &logits, std::vector<std::pair<Data, Data>> &pastKeyValues,
        const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
        std::vector<float> *retLogits, LogitsSelection *selection) {
    if (selection && !selection->candidates.dims.empty()) {
        if (selection->greedy) return (int)((float *)selection->candidates.cpuData)[0];
        return LLMSamplingOnly(selection->candidates, 0, generationConfig);
    }
    Data top;
    if (generationConfig.output_logits && retLogits) {
        logits.ToDevice(DataDevice::CPU);
        retLogits->assign((float *)logits.cpuData, (float *)logits.cpuData + logits.Count(0));
    }
    ResetLogitsOfEOS(1, &logits, pastKeyValues, generationConfig);
    if (generationConfig.IsSimpleGreedy()) {
        TopK(logits, top, 1);
        top.ToDevice(DataDevice::CPU);
        return (int)(((float *)top.cpuData)[0] + 1e-3f);
    }
    LastTokensUnit empty;
    return LLMSampling(logits, 0, generationConfig,
                       lastTokens.units.empty() ? empty : lastTokens.units[0]);
}

void NaiveN05FlashModel::WarmUp() {
    Data ids(DataType::FLOAT32, {1, 1}, {1.0f});
    Data positions(DataType::FLOAT32, {1, 1}, {0.0f});
    std::vector<std::pair<Data, Data>> cache;
    for (int layer = 0; layer < block_cnt; layer++)
        cache.emplace_back(Data(kvCacheDataType), Data(kvCacheDataType));
    Forward(ids, Data(), positions, cache);
    if (draftEnabled) {
        auto context = draftContexts.at(&cache);
        RunDraft(1, *context);
        draftContexts.erase(&cache);
    }
    elementsInKVCachePerToken = 0;
    for (int layer = 0; layer < block_cnt; layer++) {
        if (slidingLayers[layer]) continue;
        if (tpDevices.size() > 1) {
            // Count physical storage, including replicated KV heads and the
            // replicated Indexer key, for the scheduler's token budget.
            for (int device : tpDevices)
                elementsInKVCachePerToken += cache[layer].first.multiDeviceDatas.at(device)->dims[2] +
                                            cache[layer].second.multiDeviceDatas.at(device)->dims[2];
        } else {
            elementsInKVCachePerToken += cache[layer].first.dims[2] + cache[layer].second.dims[2];
        }
    }
}
}
