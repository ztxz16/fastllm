#include "naive_n05_flash.h"
#include "blocks/baseblock.h"
#include "executor.h"
#include "json11.hpp"
#include "utils.h"
#include <cmath>
#include <climits>
#ifdef USE_CUDA
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
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
    Data logits = RunTarget(inputIds, positionIds, pastKeyValues, generationConfig, nullptr);
    if (isIntermediateChunkedPrefill) return 0;
    return SampleTarget(logits, pastKeyValues, generationConfig, lastTokens, retLogits);
}

Data NaiveN05FlashModel::RunTarget(
        const Data &inputIds, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &pastKeyValues, const GenerationConfig &config,
        TargetCapture *capture) {
#ifndef USE_CUDA
    ErrorInFastLLM("Naive-N0.5 currently requires the CUDA backend for attention.");
    return Data();
#else
    AssertInFastLLM(dataType == DataType::BFLOAT16 && kvCacheDataType == DataType::BFLOAT16,
                    "Naive-N0.5 requires BF16 activations and KV cache (use auto or bfloat16).");
    AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1 &&
                    (int)pastKeyValues.size() == block_cnt,
                    "Naive-N0.5 expects one unpadded sequence and a complete KV cache.");
    int length = inputIds.dims[1];
    const int reserveCapacity = CacheReserveCapacity(config);
    const int previousExactThreshold = FastllmCudaGetLinearExactBatchThreshold();
    struct RestoreExactThreshold {
        int value;
        ~RestoreExactThreshold() { FastllmCudaSetLinearExactBatchThreshold(value); }
    } restoreExactThreshold{previousExactThreshold};
    // Verification must use the decode reduction tree, including the FP32
    // router. Different GEMM rounding can change expert selection and amplify
    // logit differences even when every cache and attention mask is correct.
    if (capture && capture->verifying)
        FastllmCudaSetLinearExactBatchThreshold(std::max(previousExactThreshold, length + 1));
    int pastLength = pastKeyValues[0].first.dims.empty() ? 0 : pastKeyValues[0].first.dims[1];
    auto historyChunk = BeginHistoryChunk(pastKeyValues, pastLength, length);
    if (capture) capture->history = historyChunk;
    AssertInFastLLM(!slidingLayers[0] && pastLength + length <= max_positions,
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
                moeWeights[layer].push_back(&weight[base + "gateup_proj.weight"]);
                moeWeights[layer].push_back(&weight[base + "down_proj.weight"]);
            }
            moeBiases[layer].resize(moeWeights[layer].size(), nullptr);
        }
    }
    ApplyDeviceMap(deviceMap, 1, block_cnt);
    Data hidden, normed, q, k, v, packed, attn, projected;
    Data routerInput, router, expertIndex, expertScore;
    Data w1, w2, w3, tempInput, tempOutput, moeOutput, moeInputTemp, moeOutputTemp;
    Data indexQ, indexKey, indexWeights, indices, positions;
    positions.CopyFrom(positionIds);
    ToDataType(positions, DataType::FLOAT32);
    Embedding(inputIds, weight["model.embed_tokens.weight"], hidden);
    ToDataType(hidden, DataType::BFLOAT16);
    for (int layer = 0; layer < block_cnt; layer++) {
        ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
        std::string prefix = "model.layers." + std::to_string(layer);
        std::string ap = prefix + ".self_attn.";
        auto &cfg = slidingLayers[layer] ? sliding : full;
        int rotaryDim = (int)(cfg.headDim * partialRotary);
        // This RMSNorm rounds the normalized activation before multiplying
        // the affine weight, matching the checkpoint's LlamaRMSNorm.
        KimiK3RMSNorm(hidden, weight[prefix + ".input_layernorm.weight"], rms_norm_eps, normed);
        Linear(normed, weight[ap + "q_proj.weight"], weight[ap + "q_proj.bias"], q);
        Linear(normed, weight[ap + "k_proj.weight"], weight[ap + "k_proj.bias"], k);
        Linear(normed, weight[ap + "v_proj.weight"], weight[ap + "v_proj.bias"], v);
        AssertInFastLLM(q.dataDevice == DataDevice::CUDA,
                        "Naive-N0.5 attention requires --device cuda.");
        positions.ToDevice(q.dataDevice, q.dataDeviceIds);
        FastllmCudaNaiveRope(q, positions, cfg.heads, cfg.headDim, rotaryDim, cfg.theta);
        FastllmCudaNaiveRope(k, positions, cfg.kvHeads, cfg.headDim, rotaryDim, cfg.theta);
        Mul(v, valueScale, v);
        if (!slidingLayers[layer]) {
            std::string ip = ap + "indexer.";
            Linear(normed, weight[ip + "wk.weight"], Data(), indexKey);
            // LayerNorm accumulates in FP32; the generic CUDA operation does
            // not accept BF16 storage, so round only its final result.
            ToDataType(indexKey, DataType::FLOAT32);
            LayerNorm(indexKey, weight[ip + "k_norm.weight"], weight[ip + "k_norm.bias"], -1, indexKey);
            ToDataType(indexKey, DataType::BFLOAT16);
            FastllmCudaNaiveRope(indexKey, positions, 1, indexDim, rotaryDim, cfg.theta);
            Cat(k, indexKey, 2, packed);
        } else {
            packed.CopyFrom(k);
        }
        if (historyChunk) {
            CopyHistoryTensor(packed, historyChunk->layers[layer].first, historyChunk->length);
            CopyHistoryTensor(v, historyChunk->layers[layer].second, historyChunk->length);
        }
        auto &pastKey = pastKeyValues[layer].first;
        auto &pastValue = pastKeyValues[layer].second;
        int localPast = pastKey.dims.empty() ? 0 : pastKey.dims[1];
        // Sliding layers retain only window-1 rows between chunks. Keep their
        // reservation bounded even when the full request is very long.
        const int layerCapacity = slidingLayers[layer]
            ? (int)std::min<int64_t>(reserveCapacity, (int64_t)window - 1 + length)
            : reserveCapacity;
        AppendCache(pastKey, packed, layerCapacity);
        AppendCache(pastValue, v, layerCapacity);
        Data noIndices;
        Data *selected = &noIndices;
        if (!slidingLayers[layer] && pastKey.dims[1] > indexTopK) {
            std::string ip = ap + "indexer.";
            Linear(normed, weight[ip + "wq.weight"], Data(), indexQ);
            FastllmCudaNaiveRope(indexQ, positions, indexHeads, indexDim, rotaryDim, cfg.theta);
            Linear(normed, weight[ip + "weights_proj.weight"], Data(), indexWeights);
            Mul(indexWeights, 1.0f / std::sqrt((float)indexHeads), indexWeights);
            FastllmCudaNaiveIndexer(indexQ, indexWeights, pastKey, indexHeads, indexDim,
                                    localPast, indexTopK, indexFp8, indices);
            selected = &indices;
        }
        Data &sink = weight[ap + "attention_sink_bias"];
        if (!sink.dims.empty()) {
            ToDataType(sink, DataType::FLOAT32);
            sink.ToDevice(q.dataDevice, q.dataDeviceIds);
        }
        FastllmCudaNaiveAttention(q, pastKey, pastValue, *selected, sink,
                                  cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                                  localPast, slidingLayers[layer] ? window : 0, attn);
        if (slidingLayers[layer] && (!capture || !capture->verifying)) {
            FastllmCudaNaiveTrimCache(pastKey, pastValue, window - 1);
        }
        Linear(attn, weight[ap + "o_proj.weight"], Data(), projected);
        AddTo(hidden, projected);
        KimiK3RMSNorm(hidden, weight[prefix + ".post_attention_layernorm.weight"], rms_norm_eps, normed);
        if (!moeLayers[layer]) {
            Linear(normed, weight[prefix + ".mlp.gate_proj.weight"], Data(), w1);
            Linear(normed, weight[prefix + ".mlp.up_proj.weight"], Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight[prefix + ".mlp.down_proj.weight"], Data(), w2);
            AddTo(hidden, w2);
        } else {
            ToDataType(normed, routerInput, DataType::FLOAT32);
            Linear(routerInput, weight[prefix + ".mlp.gate.weight"], Data(), router);
            Sigmoid(router, router);
            SelectExpert(router, expertIndex, expertScore, num_experts_per_tok,
                         norm_topk_prob, routed_scaling_factor,
                         &weight[prefix + ".mlp.gate.e_score_correction_bias"]);
            normed.Reshape({length, embed_dim});
            ApplyMoeDeviceMapForLayer(layer);
            auto &executor = *(Executor *)GetExecutor();
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
            } else {
                MergeMOEBlock(&normed, &expertIndex, &expertScore, &moeWeights[layer],
                              &moeBiases[layer], &w1, &w2, &w3, &tempInput, &tempOutput,
                              0.0f, &moeOutput, layer, DataType::BFLOAT16, moeAtype,
                              &moeInputTemp, &moeOutputTemp);
            }
            ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
            moeOutput.Reshape(hidden.dims);
            AddTo(hidden, moeOutput);
        }
        if (capture && std::find(draftTargetLayers.begin(), draftTargetLayers.end(), layer) != draftTargetLayers.end())
            Copy(hidden, capture->hidden[layer]);
    }
    if (!capture) FinishHistoryChunk(pastKeyValues, historyChunk);
    if (isIntermediateChunkedPrefill) return Data();
    Data last, logits;
    if (capture && capture->verifying) Copy(hidden, last);
    else Split(hidden, 1, length - 1, length, last);
    KimiK3RMSNorm(last, weight["model.norm.weight"], rms_norm_eps, last);
    Linear(last, weight["lm_head.weight"], Data(), logits);
    ToDataType(logits, DataType::FLOAT32);
    return logits;
#endif
}

int NaiveN05FlashModel::SampleTarget(
        Data &logits, std::vector<std::pair<Data, Data>> &pastKeyValues,
        const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
        std::vector<float> *retLogits) {
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
        if (!slidingLayers[layer])
            elementsInKVCachePerToken += cache[layer].first.dims[2] + cache[layer].second.dims[2];
    }
}
}
