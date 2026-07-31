#include "kimi_k3.h"

#include "utils.h"
#include "gguf.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

namespace fastllm {
    const std::string KimiK3Model::languagePrefix = "language_model.model.";

    namespace {
        void FlattenTextConfig(std::map<std::string, std::string> &dicts) {
            std::map<std::string, std::string> extra;
            const std::string prefix = "text_config.";
            for (const auto &item : dicts) {
                if (item.first.rfind(prefix, 0) != 0) {
                    continue;
                }
                std::string stripped = item.first.substr(prefix.size());
                if (!stripped.empty() && dicts.find(stripped) == dicts.end()) {
                    extra[stripped] = item.second;
                }
            }
            dicts.insert(extra.begin(), extra.end());
        }


        void UnpackKdaConvCache(
                Data &packed, int history, int channels,
                Data &qCache, Data &kCache, Data &vCache) {
            if (packed.dims.empty()) {
                return;
            }
            AssertInFastLLM(
                packed.dataType == DataType::BFLOAT16 &&
                packed.dims == std::vector<int>({3, history, channels}),
                "Kimi-K3 packed convolution cache has an invalid shape.");
            Split(packed, 0, 0, 1, qCache);
            Split(packed, 0, 1, 2, kCache);
            Split(packed, 0, 2, 3, vCache);
        }

        void PackKdaConvCache(
                Data &qCache, Data &kCache, Data &vCache, Data &packed) {
            Data qkCache, combined;
            Cat(qCache, kCache, 0, qkCache);
            Cat(qkCache, vCache, 0, combined);
            Copy(combined, packed);
        }

        std::vector<int> ParseIntegerList(const std::string &text) {
            std::vector<int> values;
            for (size_t position = 0; position < text.size();) {
                if (!std::isdigit((unsigned char)text[position]) &&
                    text[position] != '-') {
                    position++;
                    continue;
                }
                bool negative = text[position] == '-';
                if (negative) {
                    position++;
                }
                if (position >= text.size() ||
                    !std::isdigit((unsigned char)text[position])) {
                    continue;
                }
                int value = 0;
                while (position < text.size() &&
                       std::isdigit((unsigned char)text[position])) {
                    value = value * 10 + (text[position] - '0');
                    position++;
                }
                values.push_back(negative ? -value : value);
            }
            return values;
        }

        void AppendSequenceCache(Data &cache, const Data &current,
                                 int allocationUnit = 64) {
            AssertInFastLLM(current.dims.size() == 3,
                            "Sequence cache input must be [heads, tokens, dim].");
            if (cache.dims.empty() && cache.expansionDims.empty()) {
                cache.dataType = current.dataType;
                cache.UpdateUnitSize();
                cache.dataDevice = current.dataDevice;
                cache.dataDeviceIds = current.dataDeviceIds;
            }
            AssertInFastLLM(cache.dataType == current.dataType,
                            "Sequence cache dtype mismatch.");
            const int appendTokens = current.dims[1];
            while ((cache.dims.empty() &&
                    (cache.expansionDims.size() < 2 ||
                     appendTokens > cache.expansionDims[1])) ||
                   (!cache.dims.empty() &&
                    (cache.expansionDims.size() < 2 ||
                     cache.dims[1] + appendTokens >
                         cache.expansionDims[1]))) {
                std::vector<int> expanded =
                    cache.dims.empty() ?
                        std::vector<int>({
                            current.dims[0],
                            ((appendTokens - 1) / allocationUnit + 1) *
                                allocationUnit,
                            current.dims[2]}) :
                        (cache.expansionDims.size() == cache.dims.size() ?
                             cache.expansionDims : cache.dims);
                if (!cache.dims.empty()) {
                    expanded[1] +=
                        ((appendTokens - 1) / allocationUnit + 1) *
                        allocationUnit;
                }
                cache.Expansion(expanded);
            }
            CatDirect(cache, current, 1);
        }

        void ResizeSequenceCache(Data &cache, int tokens) {
            AssertInFastLLM(cache.dims.size() == 3 && tokens >= 0 &&
                            tokens <= cache.dims[1],
                            "Invalid sequence cache trim.");
            if (cache.isPagedKVCache) {
                AssertInFastLLM(
                    cache.pagedKVCacheData != nullptr && cache.pageLen > 0,
                    "Paged sequence cache is missing its manager.");
                const int neededPages = tokens == 0 ? 0 :
                    (tokens + cache.pageLen - 1) / cache.pageLen;
                if (neededPages < (int)cache.pageIndex.size()) {
                    std::vector<int> released(
                        cache.pageIndex.begin() + neededPages,
                        cache.pageIndex.end());
                    cache.pagedKVCacheData->ReleasePageIndices(released);
                    cache.pageIndex.resize(neededPages);
                }
                cache.lastPageLen = tokens == 0 ? 0 :
                    (tokens - 1) % cache.pageLen + 1;
            }
            cache.Resize({cache.dims[0], tokens, cache.dims[2]});
        }

        bool DeviceMapUsesCuda(const std::map<std::string, int> &deviceMap) {
            for (const auto &item : deviceMap) {
                if (item.second > 0 && item.first.rfind("cuda", 0) == 0) {
                    return true;
                }
            }
            return false;
        }

        void CreateKimiMlaAbsorbedWeights(
                Data &source, int heads, int nopeDim, int valueDim,
                int latentDim, Data &keyWeight, Data &valueWeight) {
            AssertInFastLLM(
                source.dataDevice == DataDevice::CPU &&
                source.dims == std::vector<int>({
                    heads * (nopeDim + valueDim), latentDim}),
                "Kimi-K3 kv_b_proj has an invalid shape or device.");

            auto initialize = [](Data &target, const std::vector<int> &dims,
                                 const std::string &name) {
                target.FreeSpace();
                target.dataType = DataType::BFLOAT16;
                target.UpdateUnitSize();
                target.Resize(dims);
                target.Allocate(false);
                target.name = name;
                target.isModelWeight = true;
                target.weightType = WeightType::LINEAR;
            };
            initialize(keyWeight, {heads, nopeDim, latentDim},
                       source.name + "__0");
            initialize(valueWeight, {heads, valueDim, latentDim},
                       source.name + "__1");

            const int rowsPerHead = nopeDim + valueDim;
            const int valuesPerHead = rowsPerHead * latentDim;
            std::vector<float> floatRows(valuesPerHead);
            uint16_t *keyData = (uint16_t*)keyWeight.cpuData;
            uint16_t *valueData = (uint16_t*)valueWeight.cpuData;

            auto copyHead = [&](int head, const float *rows) {
                uint16_t *keyHead = keyData +
                    (uint64_t)head * nopeDim * latentDim;
                uint16_t *valueHead = valueData +
                    (uint64_t)head * valueDim * latentDim;
                for (int i = 0; i < nopeDim * latentDim; i++) {
                    keyHead[i] = Float32ToBFloat16RNEBits(rows[i]);
                }
                for (int i = 0; i < valueDim * latentDim; i++) {
                    valueHead[i] = Float32ToBFloat16RNEBits(
                        rows[(uint64_t)nopeDim * latentDim + i]);
                }
            };

            if (source.dataType == DataType::DATA_GGUF_FORMAT) {
                const ggml_type type = (ggml_type)source.ggmlType;
                auto toFloat = ggml_type_to_float(type);
                AssertInFastLLM(
                    toFloat != nullptr,
                    "Kimi-K3 kv_b_proj GGUF type cannot be dequantized.");
                const size_t rowBytes = ggml_row_size(type, latentDim);
                for (int head = 0; head < heads; head++) {
                    const uint8_t *headData = source.cpuData +
                        (uint64_t)head * rowsPerHead * rowBytes;
                    toFloat(headData, floatRows.data(), valuesPerHead);
                    copyHead(head, floatRows.data());
                }
            } else {
                AssertInFastLLM(
                    source.dataType == DataType::FLOAT32 ||
                    source.dataType == DataType::FLOAT16 ||
                    source.dataType == DataType::BFLOAT16,
                    "Kimi-K3 MLA absorption requires a float or GGUF "
                    "kv_b_proj weight.");
                for (int head = 0; head < heads; head++) {
                    const uint64_t base =
                        (uint64_t)head * valuesPerHead;
                    for (int i = 0; i < valuesPerHead; i++) {
                        if (source.dataType == DataType::FLOAT32) {
                            floatRows[i] = ((float*)source.cpuData)[base + i];
                        } else if (source.dataType == DataType::FLOAT16) {
                            floatRows[i] = half_to_float(
                                ((uint16_t*)source.cpuData)[base + i]);
                        } else {
                            uint32_t bits = (uint32_t)
                                ((uint16_t*)source.cpuData)[base + i] << 16;
                            std::memcpy(&floatRows[i], &bits, sizeof(bits));
                        }
                    }
                    copyHead(head, floatRows.data());
                }
            }
        }

    }

    KimiK3Model::KimiK3Model() {
        model_type = "kimi_k3";
        model_struct = "kimi_k3";
        canDoBatchForward = false;
        dataType = DataType::BFLOAT16;
        // Kimi-K3 attention kernels consume BF16 Q/K/V.  Leaving the generic
        // default (FP32) here doubles every MLA cache and makes the automatic
        // context-window calculation report half of the useful capacity.
        kvCacheDataType = DataType::BFLOAT16;
        defaultChunkedPrefillSize = 2048;

        weight.embeddingNames.insert(languagePrefix + "embed_tokens.weight");
        weight.linearNames = {
            "language_model.lm_head.weight",
            languagePrefix + "layers.*.self_attn.q_proj.weight",
            languagePrefix + "layers.*.self_attn.k_proj.weight",
            languagePrefix + "layers.*.self_attn.v_proj.weight",
            languagePrefix + "layers.*.self_attn.f_a_proj.weight",
            languagePrefix + "layers.*.self_attn.f_b_proj.weight",
            languagePrefix + "layers.*.self_attn.b_proj.weight",
            languagePrefix + "layers.*.self_attn.g_proj.weight",
            languagePrefix + "layers.*.self_attn.o_proj.weight",
            languagePrefix + "layers.*.mlp.gate_proj.weight",
            languagePrefix + "layers.*.mlp.up_proj.weight",
            languagePrefix + "layers.*.mlp.down_proj.weight",
            languagePrefix + "layers.*.self_attn.q_a_proj.weight",
            languagePrefix + "layers.*.self_attn.q_b_proj.weight",
            languagePrefix + "layers.*.self_attn.kv_a_proj_with_mqa.weight",
            languagePrefix + "layers.*.self_attn.kv_b_proj.weight",
            languagePrefix + "layers.*.block_sparse_moe.gate.weight",
            languagePrefix + "layers.*.block_sparse_moe.routed_expert_down_proj.weight",
            languagePrefix + "layers.*.block_sparse_moe.routed_expert_up_proj.weight",
            languagePrefix + "layers.*.block_sparse_moe.shared_experts.gate_proj.weight",
            languagePrefix + "layers.*.block_sparse_moe.shared_experts.up_proj.weight",
            languagePrefix + "layers.*.block_sparse_moe.shared_experts.down_proj.weight",
            languagePrefix + "layers.*.block_sparse_moe.experts.*.w1.weight",
            languagePrefix + "layers.*.block_sparse_moe.experts.*.w2.weight",
            languagePrefix + "layers.*.block_sparse_moe.experts.*.w3.weight",
        };
    }

    KimiK3Model::~KimiK3Model() {
        ShutdownRuntime();
    }

    void KimiK3Model::InitParams() {
        FlattenTextConfig(weight.dicts);
        basellm::InitParams();
        auto requiredInt = [&](const std::string &key) {
            auto it = weight.dicts.find(key);
            AssertInFastLLM(it != weight.dicts.end(),
                            "Kimi-K3 config is missing " + key);
            return atoi(it->second.c_str());
        };
        auto optionalFloat = [&](const std::string &key, float fallback) {
            auto it = weight.dicts.find(key);
            return it == weight.dicts.end() ? fallback :
                   (float)atof(it->second.c_str());
        };
        kdaHeads = requiredInt("linear_attn_config.num_heads");
        kdaHeadDim = requiredInt("linear_attn_config.head_dim");
        shortConvKernel =
            requiredInt("linear_attn_config.short_conv_kernel_size");
        attnResBlockSize = requiredInt("attn_res_block_size");
        expertCount = requiredInt("num_experts");
        expertsPerToken = requiredInt("num_experts_per_token");
        routedExpertHiddenSize = requiredInt("routed_expert_hidden_size");
        moeIntermediateSize = requiredInt("moe_intermediate_size");
        qLoraRank = requiredInt("q_lora_rank");
        kvLoraRank = requiredInt("kv_lora_rank");
        qkNopeHeadDim = requiredInt("qk_nope_head_dim");
        qkRopeHeadDim = requiredInt("qk_rope_head_dim");
        vHeadDim = requiredInt("v_head_dim");
        gateLowerBound = optionalFloat(
            "linear_attn_config.gate_lower_bound", -5.0f);
        situBeta = optionalFloat("activation_situ_beta", 4.0f);
        situLinearBeta = optionalFloat("activation_situ_linear_beta", 25.0f);
        rms_norm_eps = optionalFloat("rms_norm_eps", 1e-5f);
        max_positions = requiredInt("max_position_embeddings");
        head_dim = kdaHeadDim;
        kdaLayers.assign(block_cnt, false);
        auto kdaIt = weight.dicts.find("linear_attn_config.kda_layers");
        AssertInFastLLM(kdaIt != weight.dicts.end(),
                        "Kimi-K3 config is missing linear_attn_config.kda_layers");
        const std::string &kdaList = kdaIt->second;
        for (size_t position = 0; position < kdaList.size();) {
            if (!std::isdigit((unsigned char)kdaList[position])) {
                position++;
                continue;
            }
            int layerNumber = 0;
            while (position < kdaList.size() &&
                   std::isdigit((unsigned char)kdaList[position])) {
                layerNumber = layerNumber * 10 + (kdaList[position] - '0');
                position++;
            }
            if (layerNumber >= 1 && layerNumber <= block_cnt) {
                kdaLayers[layerNumber - 1] = true;
            }
        }
        // KDA caches contain fixed convolution/recurrent state. MLA stores
        // one 512-d latent and one 64-d positional vector per token/layer;
        // kvCacheId points at the first such paged cache for length tracking.
        kvCacheId = -1;
        elementsInKVCachePerToken = 0;
        for (int layerIndex = 0; layerIndex < block_cnt; layerIndex++) {
            if (!kdaLayers[layerIndex]) {
                if (kvCacheId < 0) {
                    kvCacheId = layerIndex;
                }
                elementsInKVCachePerToken +=
                    kvLoraRank + qkRopeHeadDim;
            }
        }
        AssertInFastLLM(kvCacheId >= 0,
                        "Kimi-K3 requires at least one MLA layer.");
        AssertInFastLLM(embed_dim == 7168 && kdaHeads == 96 &&
                        kdaHeadDim == 128 && shortConvKernel == 4 &&
                        expertCount == 896 && expertsPerToken == 16 &&
                        routedExpertHiddenSize == 3584 &&
                        moeIntermediateSize == 3072 && qLoraRank == 1536 &&
                        kvLoraRank == 512 && qkNopeHeadDim == 128 &&
                        qkRopeHeadDim == 64 && vHeadDim == 128 &&
                        block_cnt == 93 && attnResBlockSize == 12,
                        "The Kimi-K3 implementation only supports the "
                        "published K3 text configuration.");

        dsparkEnabled = weight.dicts.find("dspark.model_path") !=
                        weight.dicts.end();
        if (dsparkEnabled) {
            auto dsparkRequiredInt = [&](const std::string &key) {
                const std::string full = "dspark." + key;
                auto it = weight.dicts.find(full);
                AssertInFastLLM(it != weight.dicts.end(),
                                "DSpark config is missing " + key);
                return atoi(it->second.c_str());
            };
            auto dsparkOptionalFloat = [&](const std::string &key,
                                           float fallback) {
                auto it = weight.dicts.find("dspark." + key);
                return it == weight.dicts.end() ? fallback :
                       (float)atof(it->second.c_str());
            };
            dsparkBlockSize = dsparkRequiredInt("block_size");
            dsparkLayers = dsparkRequiredInt("num_hidden_layers");
            dsparkHeads = dsparkRequiredInt("num_attention_heads");
            dsparkKvHeads = dsparkRequiredInt("num_key_value_heads");
            dsparkHeadDim = dsparkRequiredInt("head_dim");
            dsparkIntermediateSize =
                dsparkRequiredInt("intermediate_size");
            dsparkMaskTokenId =
                dsparkRequiredInt("dflash_config.mask_token_id");
            dsparkMarkovRank = dsparkRequiredInt("markov_rank");
            dsparkRmsNormEps = dsparkOptionalFloat(
                "rms_norm_eps", 1e-5f);
            auto targetLayers = weight.dicts.find(
                "dspark.dflash_config.target_layer_ids");
            AssertInFastLLM(targetLayers != weight.dicts.end(),
                            "DSpark config is missing target_layer_ids.");
            dsparkTargetLayerIds = ParseIntegerList(targetLayers->second);

            int targetVocab = requiredInt("vocab_size");
            AssertInFastLLM(
                dsparkBlockSize == 7 && dsparkLayers == 5 &&
                dsparkHeads == 64 && dsparkKvHeads == 16 &&
                dsparkHeadDim == 64 && dsparkIntermediateSize == 14336 &&
                dsparkMarkovRank == 256 &&
                dsparkRequiredInt("hidden_size") == embed_dim &&
                dsparkRequiredInt("vocab_size") == targetVocab &&
                dsparkRequiredInt("num_target_layers") == block_cnt &&
                dsparkTargetLayerIds ==
                    std::vector<int>({7, 23, 51, 67, 83}),
                "The loaded DSpark checkpoint is incompatible with the "
                "published Kimi-K3 DSpark architecture.");
            const char *confidenceThresholdEnv = std::getenv(
                "FASTLLM_DSPARK_CONFIDENCE_THRESHOLD");
            if (confidenceThresholdEnv != nullptr &&
                confidenceThresholdEnv[0] != '\0') {
                char *end = nullptr;
                dsparkConfidenceThreshold = std::strtof(
                    confidenceThresholdEnv, &end);
                AssertInFastLLM(
                    end != confidenceThresholdEnv && *end == '\0' &&
                    std::isfinite(dsparkConfidenceThreshold) &&
                    dsparkConfidenceThreshold >= 0.0f &&
                    dsparkConfidenceThreshold <= 1.0f,
                    "FASTLLM_DSPARK_CONFIDENCE_THRESHOLD must be in [0, 1].");
            }
            weight.embeddingNames.insert(
                "dspark.markov_head.markov_w1.weight");
            for (const std::string &name : {
                    "dspark.fc.weight",
                    "dspark.markov_head.markov_w2.weight",
                    "dspark.confidence_head.proj.weight"}) {
                weight.linearNames.insert(name);
            }
            for (int layer = 0; layer < dsparkLayers; layer++) {
                const std::string prefix = "dspark.layers." +
                    std::to_string(layer) + ".";
                for (const std::string &suffix : {
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                        "self_attn.o_proj.weight",
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight",
                        "mlp.down_proj.weight"}) {
                    weight.linearNames.insert(prefix + suffix);
                }
            }
            // DSpark keeps five ordinary FP16 GQA K/V caches on the final
            // pipeline device in addition to the target's compressed MLA.
            elementsInKVCachePerToken +=
                dsparkLayers * 2 * dsparkKvHeads * dsparkHeadDim;
        }
        // Export/load policy uses FastLLM's ordinary MoE registry.  Source
        // MXFP4 tensors map to these logical names, and dtype_config can then
        // convert them to GGML Q2_K without checkpoint-specific exporter code.
        for (int layerIndex = 1; layerIndex < block_cnt; layerIndex++) {
            const std::string experts = languagePrefix + "layers." +
                std::to_string(layerIndex) + ".block_sparse_moe.experts.";
            for (int expert = 0; expert < expertCount; expert++) {
                const std::string prefix = experts + std::to_string(expert) + ".";
                moeLinears.insert(prefix + "w1.weight");
                moeLinears.insert(prefix + "w2.weight");
                moeLinears.insert(prefix + "w3.weight");
            }
        }
    }

    std::map<std::string, std::vector<std::pair<std::string, DataType>>>
    KimiK3Model::GetTensorMap(const std::vector<std::string> &tensorNames) {
        std::map<std::string, std::vector<std::pair<std::string, DataType>>> result;
        // Keep every matrix consumed by Linear on FastLLM's ordinary
        // DATA_AUTO_LINEAR path.  This preserves the source behaviour when
        // loading with BF16 while allowing dtype_config to select GGML
        // quantization per weight during export/load.
        static const std::set<std::string> quantizableLinearSuffixes = {
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.f_a_proj.weight",
            "self_attn.f_b_proj.weight",
            "self_attn.b_proj.weight",
            "self_attn.g_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "block_sparse_moe.gate.weight",
            "block_sparse_moe.routed_expert_down_proj.weight",
            "block_sparse_moe.routed_expert_up_proj.weight",
            "block_sparse_moe.shared_experts.gate_proj.weight",
            "block_sparse_moe.shared_experts.up_proj.weight",
            "block_sparse_moe.shared_experts.down_proj.weight",
        };
        static const std::set<std::string> fp32Suffixes = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attention_res_norm.weight",
            "self_attention_res_proj.weight",
            "mlp_res_norm.weight",
            "mlp_res_proj.weight",
            "self_attn.q_conv1d.weight",
            "self_attn.k_conv1d.weight",
            "self_attn.v_conv1d.weight",
            "self_attn.A_log",
            "self_attn.dt_bias",
            "self_attn.o_norm.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.kv_a_layernorm.weight",
            "block_sparse_moe.gate.e_score_correction_bias",
            "block_sparse_moe.routed_expert_norm.weight",
        };
        for (const std::string &name : tensorNames) {
            if (dsparkEnabled) {
                const std::string target = "dspark." + name;
                if (name == "fc.weight" ||
                    name == "markov_head.markov_w1.weight" ||
                    name == "markov_head.markov_w2.weight") {
                    result[name].emplace_back(target, DataType::BFLOAT16);
                    continue;
                }
                if (name == "hidden_norm.weight" ||
                    name == "norm.weight" ||
                    name == "confidence_head.proj.weight" ||
                    name == "confidence_head.proj.bias") {
                    result[name].emplace_back(target, DataType::FLOAT32);
                    continue;
                }
                bool draftMatched = false;
                for (int layerIndex = 0; layerIndex < dsparkLayers;
                     layerIndex++) {
                    const std::string layer = "layers." +
                        std::to_string(layerIndex) + ".";
                    if (name.rfind(layer, 0) != 0) {
                        continue;
                    }
                    std::string suffix = name.substr(layer.size());
                    static const std::set<std::string> draftLinearSuffixes = {
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                        "self_attn.o_proj.weight",
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight",
                        "mlp.down_proj.weight",
                    };
                    static const std::set<std::string> draftNormSuffixes = {
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                        "self_attn.q_norm.weight",
                        "self_attn.k_norm.weight",
                    };
                    if (draftLinearSuffixes.find(suffix) !=
                        draftLinearSuffixes.end()) {
                        result[name].emplace_back(
                            target, DataType::BFLOAT16);
                        draftMatched = true;
                    } else if (draftNormSuffixes.find(suffix) !=
                               draftNormSuffixes.end()) {
                        result[name].emplace_back(
                            target, DataType::FLOAT32);
                        draftMatched = true;
                    }
                    break;
                }
                if (draftMatched) {
                    continue;
                }
            }
            if (name == languagePrefix + "embed_tokens.weight") {
                result[name].emplace_back(name, DataType::BFLOAT16);
                continue;
            }
            if (name == languagePrefix + "norm.weight" ||
                name == languagePrefix + "output_attn_res_norm.weight" ||
                name == languagePrefix + "output_attn_res_proj.weight") {
                result[name].emplace_back(name, DataType::FLOAT32);
                continue;
            }
            if (name == "language_model.lm_head.weight") {
                result[name].emplace_back(
                    name, DataType::DATA_AUTO_LINEAR);
                continue;
            }
            for (int layerIndex = 0; layerIndex < block_cnt;
                 layerIndex++) {
                const std::string layer = languagePrefix + "layers." +
                    std::to_string(layerIndex) + ".";
                if (name.rfind(layer, 0) != 0) {
                    continue;
                }
                std::string suffix = name.substr(layer.size());
                if (quantizableLinearSuffixes.find(suffix) !=
                    quantizableLinearSuffixes.end()) {
                    result[name].emplace_back(
                        name, DataType::DATA_AUTO_LINEAR);
                } else if (fp32Suffixes.find(suffix) != fp32Suffixes.end()) {
                    result[name].emplace_back(name, DataType::FLOAT32);
                } else if (suffix.rfind("block_sparse_moe.experts.", 0) == 0 &&
                           StringEndWith(name, ".weight_packed")) {
                    std::string target = name.substr(
                        0, name.size() - std::string("weight_packed").size()) +
                        "weight";
                    result[name].emplace_back(target, DataType::DATA_AUTO_LINEAR);
                }
                break;
            }
        }
        return result;
    }

    void KimiK3Model::OnModelWeightsLoaded() {
        auto require = [&](const std::string &name) -> Data& {
            auto it = weight.weight.find(name);
            AssertInFastLLM(it != weight.weight.end() && !it->second.dims.empty(),
                            "Kimi-K3 checkpoint is missing " + name);
            return it->second;
        };
        require(languagePrefix + "embed_tokens.weight");
        expertW1s.assign(block_cnt, {});
        expertW2s.assign(block_cnt, {});
        expertW3s.assign(block_cnt, {});
#ifdef USE_CUDA
        const bool createCompressedMlaWeights = DeviceMapUsesCuda(deviceMap);
#else
        const bool createCompressedMlaWeights = false;
#endif

        for (int layerIndex = 0; layerIndex < block_cnt; layerIndex++) {
            const std::string layer = languagePrefix + "layers." +
                std::to_string(layerIndex) + ".";
            for (const std::string &suffix : {
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "mlp_res_norm.weight",
                    "mlp_res_proj.weight"}) {
                require(layer + suffix);
            }
            if (layerIndex > 0) {
                require(layer + "self_attention_res_norm.weight");
                require(layer + "self_attention_res_proj.weight");
            }

            if (kdaLayers[layerIndex]) {
                for (const std::string &suffix : {
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                        "self_attn.q_conv1d.weight",
                        "self_attn.k_conv1d.weight",
                        "self_attn.v_conv1d.weight",
                        "self_attn.f_a_proj.weight",
                        "self_attn.f_b_proj.weight",
                        "self_attn.b_proj.weight",
                        "self_attn.A_log",
                        "self_attn.dt_bias",
                        "self_attn.g_proj.weight",
                        "self_attn.o_norm.weight",
                        "self_attn.o_proj.weight"}) {
                    require(layer + suffix);
                }
                Data &aLog = require(layer + "self_attn.A_log");
                if (aLog.Count(0) == (uint64_t)kdaHeadDim) {
                    if (layerIndex == 0) {
                        std::cerr
                            << "[Kimi-K3] WARNING: checkpoint A_log has "
                            << aLog.Count(0)
                            << " values; broadcasting it over head_dim. The "
                               "bundled Transformers code and technical report "
                               "describe "
                            << kdaHeads << " per-head values.\n";
                    }
                } else {
                    AssertInFastLLM(aLog.Count(0) == (uint64_t)kdaHeads,
                                    "Kimi-K3 A_log has an unsupported shape.");
                }
            } else {
                for (const std::string &suffix : {
                        "self_attn.q_a_proj.weight",
                        "self_attn.q_a_layernorm.weight",
                        "self_attn.q_b_proj.weight",
                        "self_attn.kv_a_proj_with_mqa.weight",
                        "self_attn.kv_a_layernorm.weight",
                        "self_attn.kv_b_proj.weight",
                        "self_attn.g_proj.weight",
                        "self_attn.o_proj.weight"}) {
                    require(layer + suffix);
                }
                if (createCompressedMlaWeights) {
                    const std::string kvWeightName =
                        layer + "self_attn.kv_b_proj.weight";
                    Data &combinedKvWeight = require(kvWeightName);
                    CreateKimiMlaAbsorbedWeights(
                        combinedKvWeight, kdaHeads, qkNopeHeadDim, vHeadDim,
                        kvLoraRank, weight.weight[kvWeightName + "__0"],
                        weight.weight[kvWeightName + "__1"]);
                }
            }

            if (layerIndex == 0) {
                require(layer + "mlp.gate_proj.weight");
                require(layer + "mlp.up_proj.weight");
                require(layer + "mlp.down_proj.weight");
                continue;
            }

            const std::string moe = layer + "block_sparse_moe.";
            for (const std::string &suffix : {
                    "gate.weight",
                    "gate.e_score_correction_bias",
                    "routed_expert_down_proj.weight",
                    "routed_expert_norm.weight",
                    "routed_expert_up_proj.weight",
                    "shared_experts.gate_proj.weight",
                    "shared_experts.up_proj.weight",
                    "shared_experts.down_proj.weight"}) {
                require(moe + suffix);
            }
            expertW1s[layerIndex].reserve(expertCount);
            expertW2s[layerIndex].reserve(expertCount);
            expertW3s[layerIndex].reserve(expertCount);
            for (int expert = 0; expert < expertCount; expert++) {
                const std::string expertPrefix = moe + "experts." +
                    std::to_string(expert) + ".";
                Data &w1 = require(expertPrefix + "w1.weight");
                Data &w2 = require(expertPrefix + "w2.weight");
                Data &w3 = require(expertPrefix + "w3.weight");
                AssertInFastLLM(
                    w1.dims.size() == 2 && w2.dims.size() == 2 &&
                    w3.dims.size() == 2,
                    "Kimi-K3 routed expert weights must be matrices.");
#ifdef USE_NUMAS
                AssertInFastLLM(
                    IsNumasLinearWeightSupported(&w1) &&
                    IsNumasLinearWeightSupported(&w2) &&
                    IsNumasLinearWeightSupported(&w3),
                    "Kimi-K3 routed expert uses a format unsupported by the "
                    "NUMA Linear/MergeMOE conversion path.");
#endif
                expertW1s[layerIndex].push_back(&w1);
                expertW2s[layerIndex].push_back(&w2);
                expertW3s[layerIndex].push_back(&w3);
            }
        }
        require(languagePrefix + "output_attn_res_norm.weight");
        require(languagePrefix + "output_attn_res_proj.weight");
        require(languagePrefix + "norm.weight");
        require("language_model.lm_head.weight");

        if (dsparkEnabled) {
            const int targetVocab = atoi(weight.dicts.at("vocab_size").c_str());
            auto requireShape = [&](const std::string &name,
                                    const std::vector<int> &shape) -> Data& {
                Data &data = require("dspark." + name);
                AssertInFastLLM(
                    data.dims == shape,
                    "DSpark weight has an invalid shape: " + name);
                return data;
            };
            requireShape("fc.weight", {embed_dim, 5 * embed_dim});
            requireShape("hidden_norm.weight", {embed_dim});
            requireShape("norm.weight", {embed_dim});
            requireShape("markov_head.markov_w1.weight",
                         {targetVocab, dsparkMarkovRank});
            requireShape("markov_head.markov_w2.weight",
                         {targetVocab, dsparkMarkovRank});
            requireShape("confidence_head.proj.weight",
                         {1, embed_dim + dsparkMarkovRank});
            requireShape("confidence_head.proj.bias", {1});
            for (int layerIndex = 0; layerIndex < dsparkLayers;
                 layerIndex++) {
                const std::string layer = "layers." +
                    std::to_string(layerIndex) + ".";
                requireShape(layer + "input_layernorm.weight", {embed_dim});
                requireShape(layer + "post_attention_layernorm.weight",
                             {embed_dim});
                requireShape(layer + "self_attn.q_norm.weight",
                             {dsparkHeadDim});
                requireShape(layer + "self_attn.k_norm.weight",
                             {dsparkHeadDim});
                requireShape(layer + "self_attn.q_proj.weight",
                             {dsparkHeads * dsparkHeadDim, embed_dim});
                requireShape(layer + "self_attn.k_proj.weight",
                             {dsparkKvHeads * dsparkHeadDim, embed_dim});
                requireShape(layer + "self_attn.v_proj.weight",
                             {dsparkKvHeads * dsparkHeadDim, embed_dim});
                requireShape(layer + "self_attn.o_proj.weight",
                             {embed_dim, dsparkHeads * dsparkHeadDim});
                requireShape(layer + "mlp.gate_proj.weight",
                             {dsparkIntermediateSize, embed_dim});
                requireShape(layer + "mlp.up_proj.weight",
                             {dsparkIntermediateSize, embed_dim});
                requireShape(layer + "mlp.down_proj.weight",
                             {embed_dim, dsparkIntermediateSize});
            }
            std::cout << "[Kimi-K3] DSpark enabled: 5-layer BF16 draft, "
                      << "block=" << dsparkBlockSize
                      << ", target hidden layers=[7,23,51,67,83].\n";
            std::cout << "[Kimi-K3] Reusable history cache uses aligned "
                      << "target + DSpark prefill snapshots.\n";
        }
    }

    void KimiK3Model::WarmUp() {
#ifdef USE_NUMAS
        size_t totalWeights = 0;
        for (int layerIndex = 1; layerIndex < block_cnt;
             layerIndex++) {
            const std::string selectedDevice =
                SelectMoeDeviceForLayer(layerIndex);
            if (selectedDevice == "numa" ||
                selectedDevice.rfind("numa:", 0) == 0) {
                totalWeights += expertW1s[layerIndex].size() * 3;
            }
        }
        if (totalWeights > 0) {
            std::cout << "[Kimi-K3] NUMA warmup: converting and registering "
                      << totalWeights << " routed-expert weights.\n";
            auto start = std::chrono::steady_clock::now();
            size_t completed = 0;
            int lastProgress = -1;
            for (int layerIndex = 1; layerIndex < block_cnt;
                 layerIndex++) {
                const std::string selectedDevice =
                    SelectMoeDeviceForLayer(layerIndex);
                if (selectedDevice != "numa" &&
                    selectedDevice.rfind("numa:", 0) != 0) {
                    continue;
                }
                std::vector<Data*> layerWeights;
                layerWeights.reserve(expertW1s[layerIndex].size() * 3);
                for (int expert = 0;
                     expert < (int)expertW1s[layerIndex].size(); expert++) {
                    layerWeights.push_back(expertW1s[layerIndex][expert]);
                    layerWeights.push_back(expertW2s[layerIndex][expert]);
                    layerWeights.push_back(expertW3s[layerIndex][expert]);
                }
                RegisterNumasLinearWeightBatch(layerWeights);
                for (Data *weight : layerWeights) {
                    AssertInFastLLM(
                        weight != nullptr && weight->cpuData == nullptr &&
                        IsNumasLinearWeightRegistered(weight),
                        "Kimi-K3 NUMA warmup failed to register an expert "
                        "weight through the Linear/MergeMOE format path.");
                }
                completed += layerWeights.size();
                int progress = (int)(completed * 100 / totalWeights);
                if (progress != lastProgress) {
                    std::cout << "\r[Kimi-K3] NUMA warmup: " << progress << "%"
                              << std::flush;
                    lastProgress = progress;
                }
            }
            double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << "\n[Kimi-K3] NUMA warmup complete: " << completed
                      << " expert weights in " << std::fixed
                      << std::setprecision(2) << seconds << " s.\n";
        }
#endif

#ifdef USE_CUDA
        if (!DeviceMapUsesCuda(deviceMap)) {
            return;
        }
        AssertInFastLLM(
            dataType == DataType::BFLOAT16,
            "Kimi-K3 CUDA compressed MLA requires BF16 activations.");

        // The old scheduler sizes KV cache when the first request arrives.
        // Kimi-K3 weights used to migrate lazily to CUDA during that request,
        // so the scheduler mistook weight memory for free KV capacity and a
        // long prefill later OOMed.  Exercise both target and DSpark paths now,
        // then release their temporary caches before capacity is measured.
        std::cout << "[Kimi-K3] CUDA weight warmup...\n";
        std::vector<std::pair<Data, Data>> warmCaches;
        warmCaches.reserve(block_cnt);
        for (int layerIndex = 0; layerIndex < block_cnt; layerIndex++) {
            warmCaches.emplace_back(
                Data(kvCacheDataType), Data(kvCacheDataType));
            warmCaches.back().first.SetKVCache();
            warmCaches.back().second.SetKVCache();
        }
        GenerationConfig warmConfig;
        LastTokensManager warmLastTokens;
        Data warmInput(DataType::FLOAT32, {1, 1}, {1.0f});
        cudaWeightWarmupRunning = true;
        int nextToken = Forward(
            warmInput, Data(), Data(), warmCaches,
            warmConfig, warmLastTokens, nullptr);
        if (dsparkEnabled) {
            Data draftInput(
                DataType::FLOAT32, {1, 1}, {(float)nextToken});
            Forward(draftInput, Data(), Data(), warmCaches,
                    warmConfig, warmLastTokens, nullptr);
            std::lock_guard<std::mutex> guard(dsparkContextMutex);
            dsparkContexts.erase(&warmCaches);
        }
        cudaWeightWarmupRunning = false;
        warmCaches.clear();
        ClearAllPagedCacheManagers();
        FastllmCudaClearBigBuffer();

        // Publish the usable capacity before the Python server builds
        // /v1/models metadata.  The old scheduler otherwise performs this
        // calculation lazily on the first request, so clients are briefly
        // told that the full architectural context window is available.
        const auto freeSizes = FastllmCudaGetFreeSizes();
        const auto totalSizes = FastllmCudaGetTotalSizes();
        std::set<int> deviceIds;
        std::map<int, int> ratios;
        for (const auto &item : deviceMap) {
            if (StartWith(item.first, "cuda")) {
                for (int deviceId :
                     ParseDeviceIds(item.first, "cuda", ratios)) {
                    deviceIds.insert(deviceId);
                }
            }
        }
        if (deviceIds.empty()) {
            deviceIds.insert(0);
        }
        std::map<int, long long> cacheBytesPerToken;
        auto addCacheBytes = [&](const std::string &device,
                                 long long bytes) {
            std::map<int, int> localRatios;
            std::vector<int> ids = StartWith(device, "cuda") ?
                ParseDeviceIds(device, "cuda", localRatios) :
                std::vector<int>();
            if (ids.empty()) {
                ids.assign(deviceIds.begin(), deviceIds.end());
            }
            // A multi-CUDA layer shards its cache; pipeline CUDA entries have
            // one id and therefore receive the complete per-layer charge.
            const long long perDevice =
                (bytes + (long long)ids.size() - 1) / ids.size();
            for (int id : ids) {
                cacheBytesPerToken[id] += perDevice;
            }
        };
        const long long targetLayerBytes = GetDataBytes(
            kvCacheDataType, 1, kvLoraRank + qkRopeHeadDim);
        for (int layerIndex = 0; layerIndex < block_cnt; layerIndex++) {
            if (!kdaLayers[layerIndex]) {
                addCacheBytes(
                    SelectDeviceFromMap(
                        deviceMap, layerIndex, block_cnt),
                    targetLayerBytes);
            }
        }
        if (dsparkEnabled) {
            const long long draftBytes = GetDataBytes(
                DataType::FLOAT16, 1,
                dsparkLayers * 2 * dsparkKvHeads * dsparkHeadDim);
            addCacheBytes(
                SelectDeviceFromMap(deviceMap, block_cnt, block_cnt),
                draftBytes);
        }

        long long capacity = max_positions;
        const int prefillChunkSize = GetChunkedPrefillSize();
        const int extraPrefillTokens = std::max(
            0, prefillChunkSize - defaultChunkedPrefillSize);
        const long long activationBytesPerExtraToken =
            (long long)GetDataBytes(
                dataType, 1, embed_dim) * 9;
        const long long linearPrefillReserve =
            (long long)extraPrefillTokens *
            activationBytesPerExtraToken;
        // Larger chunks also grow direct CUDA workspaces and leave a less
        // reusable mix of pooled buffers.  That high-water mark is not fully
        // described by the live hidden-state rows above: a 4096-token Kimi
        // prefill still exhausted the generic allocator headroom late in a
        // long request.  Keep one allocator-growth margin whenever the chunk
        // exceeds the default; explicit context/KV limits remain untouched.
        const long long allocatorGrowthMargin =
            extraPrefillTokens > 0 ? (768LL << 20) : 0;
        const long long prefillRuntimeReserve =
            linearPrefillReserve + allocatorGrowthMargin;
        if (prefillRuntimeReserve > 0 && kvCacheLimit <= 0 &&
            GetMaxTokens() <= 0) {
            std::cout << "[Kimi-K3] Auto context reserves "
                      << std::fixed << std::setprecision(2)
                      << (double)prefillRuntimeReserve /
                             (1024.0 * 1024.0)
                      << " MiB per GPU for chunked prefill "
                      << prefillChunkSize << ".\n";
        }
        for (int deviceId : deviceIds) {
            if (deviceId >= 0 && deviceId < (int)freeSizes.size()) {
                long long usable = std::max(
                    freeSizes[deviceId] * 3 / 4,
                    freeSizes[deviceId] - (2LL << 30));
                if (kvCacheLimit <= 0 && GetMaxTokens() <= 0) {
                    if (deviceId < (int)totalSizes.size()) {
                        const long long ratioReserve = (long long)(
                            totalSizes[deviceId] *
                            (1.0 - GetGpuMemRatio()));
                        usable = std::min(
                            usable,
                            std::max(
                                0LL,
                                freeSizes[deviceId] - ratioReserve));
                    }
                    usable = std::max(
                        0LL, usable - prefillRuntimeReserve);
                }
                auto bytes = cacheBytesPerToken.find(deviceId);
                if (bytes != cacheBytesPerToken.end() && bytes->second > 0) {
                    capacity = std::min(capacity, usable / bytes->second);
                }
            }
        }
        if (kvCacheLimit > 0) {
            const long long totalBytesPerToken = GetDataBytes(
                kvCacheDataType, 1, elementsInKVCachePerToken);
            capacity = totalBytesPerToken > 0 ?
                kvCacheLimit / totalBytesPerToken : 0;
        }
        if (GetMaxTokens() > 0) {
            capacity = GetMaxTokens();
        }
        if (capacity > 0) {
            tokensLimit = (int)std::min<long long>(
                capacity, std::numeric_limits<int>::max());
            promptLimit = tokensLimit * 3 / 4;
            std::cout << "[Kimi-K3] KV cache capacity after CUDA warmup: "
                      << tokensLimit << " tokens (prompt limit "
                      << std::min(max_positions, promptLimit) << ").\n";
        }
        std::cout << "[Kimi-K3] CUDA weight warmup complete.\n";
#endif
    }

    void KimiK3Model::RunKdaAttentionImpl(
            int layerIndex, Data &normalized, int sequence,
            std::vector<std::pair<Data, Data>> *pastKeyValues,
            TargetRunCapture *capture, Data &attention) {
        AssertInFastLLM(
            normalized.dims.size() >= 2 &&
            normalized.dims[normalized.dims.size() - 2] == sequence,
            "Kimi-K3 KDA input has an invalid sequence dimension.");
        const std::string layer = languagePrefix + "layers." +
            std::to_string(layerIndex) + ".";

        Data qCache, kCache, vCache;
        if (pastKeyValues != nullptr) {
            UnpackKdaConvCache(
                (*pastKeyValues)[layerIndex].first,
                shortConvKernel - 1, kdaHeads * kdaHeadDim,
                qCache, kCache, vCache);
        }
        Data localState;
        Data &state = pastKeyValues == nullptr ?
            localState : (*pastKeyValues)[layerIndex].second;

        auto runChunk = [&](Data &chunkInput, int chunkSequence,
                            Data &chunkAttention) {
            Data qProjected, kProjected, vProjected;
            Linear(chunkInput, weight[layer + "self_attn.q_proj.weight"],
                   Data(), qProjected);
            Linear(chunkInput, weight[layer + "self_attn.k_proj.weight"],
                   Data(), kProjected);
            Linear(chunkInput, weight[layer + "self_attn.v_proj.weight"],
                   Data(), vProjected);
            if (capture != nullptr && capture->captureKda) {
                capture->kda[layerIndex].qProjected.CopyFrom(qProjected);
                capture->kda[layerIndex].kProjected.CopyFrom(kProjected);
                capture->kda[layerIndex].vProjected.CopyFrom(vProjected);
            }

            Data qConv, kConv, vConv;
            if (pastKeyValues != nullptr) {
                KimiK3CausalConv1D(
                    qProjected,
                    weight[layer + "self_attn.q_conv1d.weight"],
                    shortConvKernel, qCache, qConv);
                KimiK3CausalConv1D(
                    kProjected,
                    weight[layer + "self_attn.k_conv1d.weight"],
                    shortConvKernel, kCache, kConv);
                KimiK3CausalConv1D(
                    vProjected,
                    weight[layer + "self_attn.v_conv1d.weight"],
                    shortConvKernel, vCache, vConv);
            } else {
                KimiK3CausalConv1D(
                    qProjected,
                    weight[layer + "self_attn.q_conv1d.weight"],
                    shortConvKernel, qConv);
                KimiK3CausalConv1D(
                    kProjected,
                    weight[layer + "self_attn.k_conv1d.weight"],
                    shortConvKernel, kConv);
                KimiK3CausalConv1D(
                    vProjected,
                    weight[layer + "self_attn.v_conv1d.weight"],
                    shortConvKernel, vConv);
            }
            qConv.Reshape({1, chunkSequence, kdaHeads, kdaHeadDim});
            kConv.Reshape({1, chunkSequence, kdaHeads, kdaHeadDim});
            vConv.Reshape({1, chunkSequence, kdaHeads, kdaHeadDim});
            Data q, k;
            KimiK3L2Norm(qConv, 1e-6f, q);
            KimiK3L2Norm(kConv, 1e-6f, k);

            Data gateLowRank, rawGate;
            Linear(chunkInput, weight[layer + "self_attn.f_a_proj.weight"],
                   Data(), gateLowRank);
            Linear(gateLowRank, weight[layer + "self_attn.f_b_proj.weight"],
                   Data(), rawGate);
            rawGate.Reshape(
                {1, chunkSequence, kdaHeads, kdaHeadDim});
            Data rawBetaBf16, rawBeta;
            Linear(chunkInput, weight[layer + "self_attn.b_proj.weight"],
                   Data(), rawBetaBf16);
            ToDataType(rawBetaBf16, rawBeta, DataType::FLOAT32);
            if (capture != nullptr && capture->captureKda) {
                capture->kda[layerIndex].k.CopyFrom(k);
                capture->kda[layerIndex].v.CopyFrom(vConv);
                capture->kda[layerIndex].rawGate.CopyFrom(rawGate);
                capture->kda[layerIndex].rawBeta.CopyFrom(rawBeta);
            }

            Data kdaOutput;
            KimiK3RecurrentKDAOutputOnly(
                q, k, vConv, rawGate, rawBeta,
                weight[layer + "self_attn.A_log"],
                weight[layer + "self_attn.dt_bias"], gateLowerBound,
                state, kdaOutput);

            Data outputGate;
            Linear(chunkInput, weight[layer + "self_attn.g_proj.weight"],
                   Data(), outputGate);
            outputGate.Reshape(
                {1, chunkSequence, kdaHeads, kdaHeadDim});
            Data gatedAttention;
            KimiK3RMSNormSigmoidGate(
                kdaOutput, outputGate,
                weight[layer + "self_attn.o_norm.weight"],
                rms_norm_eps, gatedAttention);
            gatedAttention.Reshape(
                {1, chunkSequence, kdaHeads * kdaHeadDim});
            Linear(gatedAttention, weight[layer + "self_attn.o_proj.weight"],
                   Data(), chunkAttention);
        };

        const int sequenceAxis = (int)normalized.dims.size() - 2;
        const int kdaChunkSize = 1024;
        const bool chunkPrefill =
            normalized.dataDevice == DataDevice::CUDA &&
            pastKeyValues != nullptr &&
            (capture == nullptr || !capture->captureKda) &&
            sequence > kdaChunkSize;
        if (!chunkPrefill) {
            runChunk(normalized, sequence, attention);
        } else {
            for (int start = 0; start < sequence; start += kdaChunkSize) {
                const int end = std::min(sequence, start + kdaChunkSize);
                Data chunkInput, chunkAttention;
                Split(normalized, sequenceAxis, start, end, chunkInput);
                runChunk(chunkInput, end - start, chunkAttention);
                if (start == 0) {
                    Copy(chunkAttention, attention);
                    attention.Expansion(normalized.dims);
                } else {
                    CatDirect(attention, chunkAttention, sequenceAxis);
                }
            }
            attention.expansionDims.clear();
        }

        if (pastKeyValues != nullptr) {
            PackKdaConvCache(
                qCache, kCache, vCache,
                (*pastKeyValues)[layerIndex].first);
        }
    }

    Data KimiK3Model::RunFirstLayerImpl(
            const std::vector<int> &tokenIds,
            std::vector<std::pair<Data, Data>> *pastKeyValues,
            TargetRunCapture *capture) {
        AssertInFastLLM(!tokenIds.empty(),
                        "Kimi-K3 requires at least one input token.");
        ApplyDeviceMap(deviceMap, 0, 1);
        const std::string layer = languagePrefix + "layers.0.";
        int sequence = (int)tokenIds.size();
        std::vector<float> ids(sequence);
        for (int i = 0; i < sequence; i++) {
            ids[i] = (float)tokenIds[i];
        }
        Data inputIds(DataType::FLOAT32, {1, sequence}, ids);
        Data embedding;
        Embedding(inputIds, weight[languagePrefix + "embed_tokens.weight"], embedding);
        ToDataType(embedding, DataType::BFLOAT16);

        Data normalized;
        KimiK3RMSNorm(
            embedding, weight[layer + "input_layernorm.weight"],
            rms_norm_eps, normalized);

        Data attention;
        RunKdaAttentionImpl(
            0, normalized, sequence, pastKeyValues, capture, attention);

        Data mlpInput;
        embedding.Reshape({sequence, 1, embed_dim});
        KimiK3AttnRes(
            attention, embedding,
            weight[layer + "mlp_res_proj.weight"],
            weight[layer + "mlp_res_norm.weight"],
            rms_norm_eps, mlpInput);
        Data mlpNormalized;
        KimiK3RMSNorm(
            mlpInput, weight[layer + "post_attention_layernorm.weight"],
            rms_norm_eps, mlpNormalized);

        Data mlpGate, mlpUp;
        Linear(mlpNormalized, weight[layer + "mlp.gate_proj.weight"],
               Data(), mlpGate);
        Linear(mlpNormalized, weight[layer + "mlp.up_proj.weight"],
               Data(), mlpUp);
        Data mlpActivated;
        KimiK3SiTUAndMul(
            mlpGate, mlpUp, situBeta, situLinearBeta, mlpActivated);
        Data mlpOutput;
        Linear(mlpActivated, weight[layer + "mlp.down_proj.weight"],
               Data(), mlpOutput);
        Data layerOutput;
        Copy(attention, layerOutput);
        AddTo(layerOutput, mlpOutput);
        return layerOutput;
    }

    Data KimiK3Model::RunLayersImpl(
            const std::vector<int> &tokenIds, int layerCount,
            std::vector<std::pair<Data, Data>> *pastKeyValues,
            TargetRunCapture *capture) {
        AssertInFastLLM(layerCount >= 1 && layerCount <= block_cnt,
                        "Kimi-K3 requested layer count is out of range.");
        if (pastKeyValues != nullptr) {
            AssertInFastLLM((int)pastKeyValues->size() >= layerCount,
                            "Kimi-K3 cache table is shorter than the decoder.");
        }
        if (capture != nullptr && capture->captureKda &&
            (int)capture->kda.size() < layerCount) {
            capture->kda.resize(layerCount);
        }
        Data prefixSum = RunFirstLayerImpl(
            tokenIds, pastKeyValues, capture);
        if (layerCount == 1) {
            return prefixSum;
        }

        const int sequence = (int)tokenIds.size();
        std::vector<float> ids(sequence);
        for (int i = 0; i < sequence; i++) {
            ids[i] = (float)tokenIds[i];
        }
        Data inputIds(DataType::FLOAT32, {1, sequence}, ids);
        Data blockResidual;
        Embedding(inputIds, weight[languagePrefix + "embed_tokens.weight"],
                  blockResidual);
        ToDataType(blockResidual, DataType::BFLOAT16);
        blockResidual.Reshape({sequence, 1, embed_dim});

        const bool cudaSharedExpert = GetCudaSharedExpert();
        static const std::map<std::string, int> cpuMoeSupportDeviceMap = {
            {"cpu", 1},
        };
        static const std::map<std::string, int> diskMoeSupportDeviceMap = {
            {"disk", 1},
        };
        auto isCpuMoeLayer = [&](int layerIndex) {
            const std::string selected = SelectMoeDeviceForLayer(layerIndex);
            return selected == "cpu" || selected == "numa" ||
                   selected.rfind("cpu:", 0) == 0 ||
                   selected.rfind("numa:", 0) == 0;
        };
        auto applyRoutedProjectionDevice = [&](int layerIndex) {
            if (!cudaSharedExpert && isCpuMoeLayer(layerIndex)) {
                ApplyDeviceMap(cpuMoeSupportDeviceMap, 1, 1);
            } else {
                ApplyDeviceMap(deviceMap, layerIndex, layerCount);
            }
        };
        auto applySharedExpertDevice = [&](int layerIndex) {
            if (!cudaSharedExpert) {
                const std::string selected =
                    SelectMoeDeviceForLayer(layerIndex);
                if (selected == "disk" ||
                    selected.rfind("disk:", 0) == 0) {
                    ApplyDeviceMap(diskMoeSupportDeviceMap, 1, 1);
                } else {
                    ApplyDeviceMap(cpuMoeSupportDeviceMap, 1, 1);
                }
            } else {
                ApplyDeviceMap(deviceMap, layerIndex, layerCount);
            }
        };

        auto runKdaAttention = [&](int layerIndex, Data &normalized,
                                   Data &attention) {
            RunKdaAttentionImpl(
                layerIndex, normalized, sequence, pastKeyValues,
                capture, attention);
        };

        auto runMlaAttention = [&](int layerIndex, Data &normalized,
                                   Data &attention) {
            const std::string layer = languagePrefix + "layers." +
                std::to_string(layerIndex) + ".";
            const int qHeadDimension = qkNopeHeadDim + qkRopeHeadDim;

            Data qLowRank, qLowRankNorm, qStates;
            Linear(normalized, weight[layer + "self_attn.q_a_proj.weight"],
                   Data(), qLowRank);
            KimiK3RMSNorm(
                qLowRank, weight[layer + "self_attn.q_a_layernorm.weight"],
                rms_norm_eps, qLowRankNorm);
            Linear(qLowRankNorm, weight[layer + "self_attn.q_b_proj.weight"],
                   Data(), qStates);
            qStates.Reshape({1, sequence, kdaHeads, qHeadDimension});
            PermuteSelf(qStates, {0, 2, 1, 3});
            Data qNope, qRot;
            Split(qStates, -1, 0, qkNopeHeadDim, qNope);
            Split(qStates, -1, qkNopeHeadDim, qHeadDimension, qRot);

            Data compressedKv, kvLatent, kRot;
            Linear(normalized,
                   weight[layer + "self_attn.kv_a_proj_with_mqa.weight"],
                   Data(), compressedKv);
            Split(compressedKv, -1, 0, kvLoraRank, kvLatent);
            Split(compressedKv, -1, kvLoraRank,
                  kvLoraRank + qkRopeHeadDim, kRot);
            Data kvLatentNorm, kvStates;
            KimiK3RMSNorm(
                kvLatent,
                weight[layer + "self_attn.kv_a_layernorm.weight"],
                rms_norm_eps, kvLatentNorm);

#ifdef USE_CUDA
            const bool useCompressedMla =
                pastKeyValues != nullptr && !GetKVCacheInCPU() &&
                kvLatentNorm.dataDevice == DataDevice::CUDA;
            if (useCompressedMla) {
                const std::string kvWeightName =
                    layer + "self_attn.kv_b_proj.weight";
                Data &absorbedKeyWeight = weight[kvWeightName + "__0"];
                Data &absorbedValueWeight = weight[kvWeightName + "__1"];
                AssertInFastLLM(
                    absorbedKeyWeight.dataType == DataType::BFLOAT16 &&
                    absorbedValueWeight.dataType == DataType::BFLOAT16,
                    "Kimi-K3 absorbed MLA weights must be BF16.");

                // Store only the normalized 512-d latent and the model's
                // 64-d positional partition. Kimi-K3 has mla_use_nope=true,
                // so unlike DeepSeekV2 no rotary transform is applied here.
                kvLatentNorm.Reshape({1, sequence, kvLoraRank});
                kRot.Reshape({1, sequence, qkRopeHeadDim});
                Data &keyCache = (*pastKeyValues)[layerIndex].first;
                Data &valueCache = (*pastKeyValues)[layerIndex].second;
                auto preparePagedDescriptor = [](Data &cache,
                                                  const Data &current) {
                    if (cache.dims.empty() && cache.pageIndex.empty()) {
                        cache.dataType = current.dataType;
                        cache.UpdateUnitSize();
                        cache.dataDevice = current.dataDevice;
                        cache.dataDeviceIds = current.dataDeviceIds;
                    }
                    cache.isKVCache = true;
                    AssertInFastLLM(
                        cache.dataType == current.dataType,
                        "Kimi-K3 compressed MLA cache dtype mismatch.");
                };
                preparePagedDescriptor(keyCache, kRot);
                preparePagedDescriptor(valueCache, kvLatentNorm);

                const int pageLen = GetPageLen();
                const int maxPages = cudaWeightWarmupRunning ? 1 :
                    std::max(1, (tokensLimit + pageLen - 1) / pageLen);
                PagedCacheManager *keyManager = AllocatePagedCacheManager(
                    layerIndex * 2,
                    PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_MLP_CACHE,
                    kRot, pageLen, maxPages);
                PagedCacheManager *valueManager = AllocatePagedCacheManager(
                    layerIndex * 2 + 1,
                    PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_MLP_CACHE,
                    kvLatentNorm, pageLen, maxPages);
                keyCache.ToDevice(
                    kRot.dataDevice, kRot.dataDeviceIds, false);
                valueCache.ToDevice(
                    kvLatentNorm.dataDevice,
                    kvLatentNorm.dataDeviceIds, false);
                AppendPagedCache(*keyManager, keyCache, kRot);
                AppendPagedCache(*valueManager, valueCache, kvLatentNorm);

                PermuteSelf(qRot, {0, 2, 1, 3});
                PermuteSelf(qNope, {0, 2, 1, 3});
                const int batch = qNope.dims[0];
                const int queryLength = qNope.dims[1];
                const int heads = qNope.dims[2];
                const int cacheLength = keyCache.pageIndex.empty() ? 0 :
                    ((int)keyCache.pageIndex.size() - 1) * pageLen +
                    keyCache.lastPageLen;
                const int historyLength = cacheLength - queryLength;
                AssertInFastLLM(
                    batch == 1 && historyLength >= 0,
                    "Kimi-K3 compressed MLA expects one contiguous request.");

                Data attentionHeads;
                const int mlaQueryChunkSize = 1024;
                auto runMlaChunk = [&](Data &qNopeChunk,
                                       Data &qRotChunk,
                                       int chunkLength, int kvLength,
                                       Data &chunkAttentionHeads) {
                    PermuteSelf(qNopeChunk, {2, 0, 1, 3});
                    qNopeChunk.Reshape(
                        {heads, batch * chunkLength, qkNopeHeadDim});

                    Data absorbedQuery, latentAttention;
                    MatMul(qNopeChunk, absorbedKeyWeight, absorbedQuery);
                    MergeMLAPaged(
                        absorbedQuery, qRotChunk,
                        keyCache, valueCache, latentAttention,
                        1.0f / std::sqrt((float)qHeadDimension),
                        kvLength);
                    MatMulTransB(
                        latentAttention, absorbedValueWeight,
                        chunkAttentionHeads);
                    chunkAttentionHeads.Reshape(
                        {heads, batch, chunkLength, vHeadDim});
                    PermuteSelf(chunkAttentionHeads, {1, 2, 0, 3});
                    chunkAttentionHeads.Reshape(
                        {batch, chunkLength, heads * vHeadDim});
                };
                if (queryLength <= mlaQueryChunkSize) {
                    runMlaChunk(
                        qNope, qRot, queryLength,
                        cacheLength, attentionHeads);
                } else {
                    for (int start = 0; start < queryLength;
                         start += mlaQueryChunkSize) {
                        const int end = std::min(
                            queryLength, start + mlaQueryChunkSize);
                        const int chunkLength = end - start;
                        Data qNopeChunk, qRotChunk;
                        Split(qNope, 1, start, end, qNopeChunk);
                        Split(qRot, 1, start, end, qRotChunk);
                        Data chunkAttentionHeads;
                        runMlaChunk(
                            qNopeChunk, qRotChunk, chunkLength,
                            historyLength + end, chunkAttentionHeads);
                        if (start == 0) {
                            Copy(chunkAttentionHeads, attentionHeads);
                            attentionHeads.Expansion(
                                {batch, queryLength,
                                 heads * vHeadDim});
                        } else {
                            CatDirect(
                                attentionHeads, chunkAttentionHeads, 1);
                        }
                    }
                    attentionHeads.expansionDims.clear();
                }

                Data outputGate;
                Linear(normalized,
                       weight[layer + "self_attn.g_proj.weight"],
                       Data(), outputGate);
                Sigmoid(outputGate, outputGate);
                MulTo(attentionHeads, outputGate);
                Linear(attentionHeads,
                       weight[layer + "self_attn.o_proj.weight"],
                       Data(), attention);
                return;
            }
#endif
            Linear(kvLatentNorm, weight[layer + "self_attn.kv_b_proj.weight"],
                   Data(), kvStates);
            kvStates.Reshape(
                {1, sequence, kdaHeads, qkNopeHeadDim + vHeadDim});
            PermuteSelf(kvStates, {0, 2, 1, 3});
            Data kNope, valueStates;
            Split(kvStates, -1, 0, qkNopeHeadDim, kNope);
            Split(kvStates, -1, qkNopeHeadDim,
                  qkNopeHeadDim + vHeadDim, valueStates);

            kRot.Reshape({1, sequence, 1, qkRopeHeadDim});
            PermuteSelf(kRot, {0, 2, 1, 3});
            Data kRotRepeated;
            Repeat(kRot, 1, kdaHeads, kRotRepeated);
            Data queryStates, keyStates;
            Cat(qNope, qRot, -1, queryStates);
            Cat(kNope, kRotRepeated, -1, keyStates);

            Data *attentionKeys = &keyStates;
            Data *attentionValues = &valueStates;
            Data *keyCache = nullptr;
            Data *valueCache = nullptr;
            if (pastKeyValues != nullptr) {
                keyCache = &(*pastKeyValues)[layerIndex].first;
                valueCache = &(*pastKeyValues)[layerIndex].second;
                keyStates.Reshape(
                    {kdaHeads, sequence, qHeadDimension});
                valueStates.Reshape(
                    {kdaHeads, sequence, vHeadDim});
                auto prepareEmptyCache = [](Data &cache,
                                            const Data &current) {
                    // ResponseContext normally constructs every descriptor
                    // with kvCacheDataType.  Keep this helper robust when a
                    // caller provides default Data objects whose unit size
                    // has not been initialized yet.
                    if (cache.dims.empty() &&
                        cache.expansionDims.empty() &&
                        cache.expansionSize == 0) {
                        cache.dataType = current.dataType;
                        cache.UpdateUnitSize();
                        cache.dataDevice = current.dataDevice;
                    }
                    cache.isKVCache = true;
                    AssertInFastLLM(
                        cache.dataType == current.dataType,
                        "Kimi-K3 MLA cache dtype must match current K/V states.");
                };
                prepareEmptyCache(*keyCache, keyStates);
                prepareEmptyCache(*valueCache, valueStates);
                const int cacheUnit = 64;
                while ((keyCache->dims.empty() &&
                        (keyCache->expansionDims.empty() ||
                         sequence > keyCache->expansionDims[1])) ||
                       (!keyCache->dims.empty() &&
                        (keyCache->expansionDims.size() < 2 ||
                         keyCache->dims[1] + sequence >
                             keyCache->expansionDims[1]))) {
                    std::vector<int> expanded = keyCache->dims.empty() ?
                        std::vector<int>({
                            kdaHeads,
                            ((sequence - 1) / cacheUnit + 1) * cacheUnit,
                            qHeadDimension}) : keyCache->expansionDims;
                    if (!keyCache->dims.empty()) {
                        expanded[1] +=
                            ((sequence - 1) / cacheUnit + 1) * cacheUnit;
                    }
                    keyCache->Expansion(expanded);
                }
                while ((valueCache->dims.empty() &&
                        (valueCache->expansionDims.empty() ||
                         sequence > valueCache->expansionDims[1])) ||
                       (!valueCache->dims.empty() &&
                        (valueCache->expansionDims.size() < 2 ||
                         valueCache->dims[1] + sequence >
                             valueCache->expansionDims[1]))) {
                    std::vector<int> expanded = valueCache->dims.empty() ?
                        std::vector<int>({
                            kdaHeads,
                            ((sequence - 1) / cacheUnit + 1) * cacheUnit,
                            vHeadDim}) : valueCache->expansionDims;
                    if (!valueCache->dims.empty()) {
                        expanded[1] +=
                            ((sequence - 1) / cacheUnit + 1) * cacheUnit;
                    }
                    valueCache->Expansion(expanded);
                }
                CatDirect(*keyCache, keyStates, 1);
                CatDirect(*valueCache, valueStates, 1);
                const int cachedSequence = keyCache->dims[1];
                AssertInFastLLM(
                    valueCache->dims[1] == cachedSequence,
                    "Kimi-K3 MLA key/value cache length mismatch.");
                attentionKeys = keyCache;
                attentionValues = valueCache;
            }
            Data attentionHeads;
            KimiK3CausalAttention(
                queryStates, *attentionKeys, *attentionValues,
                1.0f / std::sqrt((float)qHeadDimension), attentionHeads);
            PermuteSelf(attentionHeads, {0, 2, 1, 3});
            attentionHeads.Reshape(
                {1, sequence, kdaHeads * vHeadDim});
            Data outputGate;
            Linear(normalized, weight[layer + "self_attn.g_proj.weight"],
                   Data(), outputGate);
            Sigmoid(outputGate, outputGate);
            MulTo(attentionHeads, outputGate);
            Linear(attentionHeads, weight[layer + "self_attn.o_proj.weight"],
                   Data(), attention);
        };

        auto runSparseMoe = [&](int layerIndex, Data &input,
                                Data &moeOutput) {
            const std::string layer = languagePrefix + "layers." +
                std::to_string(layerIndex) + ".";
            const std::string moe = layer + "block_sparse_moe.";
            ApplyDeviceMap(deviceMap, layerIndex, layerCount);
            Data routerInput;
            ToDataType(input, routerInput, DataType::FLOAT32);
            Data routerScores;
            Linear(routerInput, weight[moe + "gate.weight"],
                   Data(), routerScores);
            Sigmoid(routerScores, routerScores);
            Data expertIndex, expertScore;
            SelectExpert(
                routerScores, expertIndex, expertScore,
                expertsPerToken, true, 1.0f,
                &weight[moe + "gate.e_score_correction_bias"]);

            applyRoutedProjectionDevice(layerIndex);
            Data routedInput;
            Linear(input, weight[moe + "routed_expert_down_proj.weight"],
                   Data(), routedInput);
            routedInput.Reshape({sequence, routedExpertHiddenSize});

            Data routedLatent;
            ApplyMoeDeviceMapForLayer(layerIndex);
            KimiK3RoutedExperts(
                routedInput, expertIndex, expertScore,
                expertW1s[layerIndex], expertW2s[layerIndex],
                expertW3s[layerIndex], situBeta, situLinearBeta,
                routedLatent);
            routedLatent.Reshape({1, sequence, routedExpertHiddenSize});
            applyRoutedProjectionDevice(layerIndex);
            Data routedNormalized;
            KimiK3RMSNorm(
                routedLatent, weight[moe + "routed_expert_norm.weight"],
                rms_norm_eps, routedNormalized);
            Data routedOutput;
            Linear(routedNormalized,
                   weight[moe + "routed_expert_up_proj.weight"],
                   Data(), routedOutput);

            // Match the cuda_shared_expert policy used by the other MoE
            // models.  Keep the unfused shared branch on CPU when CUDA shared
            // experts are disabled; with a CPU/NUMA MoE map, keep the adjacent
            // latent projections there as well.  The router remains on the
            // layer device to preserve its original numerical path.  Restore
            // that device before combining both branches so the decoder state
            // stays on its normal CUDA/pipeline device.
            Data sharedGate, sharedUp, sharedActivated, sharedOutput;
            applySharedExpertDevice(layerIndex);
            Linear(input, weight[moe + "shared_experts.gate_proj.weight"],
                   Data(), sharedGate);
            Linear(input, weight[moe + "shared_experts.up_proj.weight"],
                   Data(), sharedUp);
            KimiK3SiTUAndMul(
                sharedGate, sharedUp, situBeta, situLinearBeta,
                sharedActivated);
            Linear(sharedActivated,
                   weight[moe + "shared_experts.down_proj.weight"],
                   Data(), sharedOutput);
            ApplyDeviceMap(deviceMap, layerIndex, layerCount);
            Copy(routedOutput, moeOutput);
            AddTo(moeOutput, sharedOutput);
        };

        for (int layerIndex = 1; layerIndex < layerCount; layerIndex++) {
            ApplyDeviceMap(deviceMap, layerIndex, layerCount);
            const std::string layer = languagePrefix + "layers." +
                std::to_string(layerIndex) + ".";
            Data attentionInput;
            KimiK3AttnRes(
                prefixSum, blockResidual,
                weight[layer + "self_attention_res_proj.weight"],
                weight[layer + "self_attention_res_norm.weight"],
                rms_norm_eps, attentionInput);
            Data normalized;
            KimiK3RMSNorm(
                attentionInput, weight[layer + "input_layernorm.weight"],
                rms_norm_eps, normalized);

            bool startsResidualBlock =
                layerIndex % attnResBlockSize == 0;
            if (startsResidualBlock) {
                Data residualCandidate, expandedResiduals;
                Copy(prefixSum, residualCandidate);
                residualCandidate.Reshape({sequence, 1, embed_dim});
                Cat(blockResidual, residualCandidate, 1, expandedResiduals);
                Copy(expandedResiduals, blockResidual);
            }

            Data attention;
            if (kdaLayers[layerIndex]) {
                runKdaAttention(layerIndex, normalized, attention);
            } else {
                runMlaAttention(layerIndex, normalized, attention);
            }
            if (startsResidualBlock) {
                Copy(attention, prefixSum);
            } else {
                AddTo(prefixSum, attention);
            }

            Data mlpInput;
            KimiK3AttnRes(
                prefixSum, blockResidual,
                weight[layer + "mlp_res_proj.weight"],
                weight[layer + "mlp_res_norm.weight"],
                rms_norm_eps, mlpInput);
            Data mlpNormalized;
            KimiK3RMSNorm(
                mlpInput, weight[layer + "post_attention_layernorm.weight"],
                rms_norm_eps, mlpNormalized);
            Data moeOutput;
            runSparseMoe(layerIndex, mlpNormalized, moeOutput);
            AddTo(prefixSum, moeOutput);
            if (capture != nullptr &&
                std::find(dsparkTargetLayerIds.begin(),
                          dsparkTargetLayerIds.end(), layerIndex) !=
                    dsparkTargetLayerIds.end()) {
                capture->targetHidden[layerIndex].CopyFrom(prefixSum);
            }
        }
        if (layerCount == block_cnt) {
            Data outputResidual, finalOutput;
            KimiK3AttnRes(
                prefixSum, blockResidual,
                weight[languagePrefix + "output_attn_res_proj.weight"],
                weight[languagePrefix + "output_attn_res_norm.weight"],
                rms_norm_eps, outputResidual);
            KimiK3RMSNorm(
                outputResidual, weight[languagePrefix + "norm.weight"],
                rms_norm_eps, finalOutput);
            return finalOutput;
        }
        return prefixSum;
    }

    void KimiK3Model::PrepareToolCallConstraint(
            ResponseContext *context,
            GenerationConfig &generationConfig) {
        basellm::PrepareToolCallConstraint(context, generationConfig);
        if (context == nullptr ||
            !generationConfig.tool_call_content_sampling_enabled) {
            return;
        }

        int &samplingStarted =
                context->intParams["kimi_k3_tool_content_sampling_started"];
        int state = context->intParams["kimi_k3_tool_content_state"];
        int contentTokens =
                context->intParams["kimi_k3_tool_content_tokens"];
        const int minContentSamplingTokens = 64;
        bool shouldSample =
                state == 1 && contentTokens >= minContentSamplingTokens;
        if (samplingStarted == 0 && shouldSample) {
            // DSpark commits accepted draft tokens into target KV before the
            // scheduler consumes them. Switching to ordinary sampling while
            // those tokens are pending would append one of them twice.
            bool hasPending = false;
            if (dsparkEnabled) {
                std::lock_guard<std::mutex> guard(dsparkContextMutex);
                auto it = dsparkContexts.find(&context->pastKeyValues);
                hasPending = it != dsparkContexts.end() &&
                             it->second != nullptr &&
                             !it->second->pending.empty();
            }
            if (!hasPending) {
                samplingStarted = 1;
            }
        }
        if (samplingStarted == 0) {
            return;
        }

        // Remember the transition for the rest of the response. ForwardDspark
        // samples from the target logits that it already computes for a draft
        // block, stopping at the first sampled token that differs from the
        // draft. This preserves target sampling without falling back to slow
        // one-token full-model steps.
        generationConfig.tool_call_content_sampling_active = true;
        if (shouldSample) {
            generationConfig.do_sample = true;
            generationConfig.top_k =
                    generationConfig.tool_call_content_top_k;
            generationConfig.top_p =
                    generationConfig.tool_call_content_top_p;
            generationConfig.temperature =
                    generationConfig.tool_call_content_temperature;
            // Keep code generation deterministic while gently discouraging
            // the exact long-range loops seen with K3's pure greedy default.
            // A top-k of one applies this penalty before selecting the token,
            // avoiding the semantic drift caused by random code sampling.
            generationConfig.repeat_penalty = std::max(
                generationConfig.repeat_penalty, 1.05f);
        }
    }

    void KimiK3Model::UpdateToolCallConstraintState(
            ResponseContext *context, int tokenId) {
        bool baseTracksText = context != nullptr &&
                (context->generationConfig.tool_call_name_constraint_enabled ||
                 context->generationConfig
                         .tool_call_parameter_name_constraint_enabled);
        basellm::UpdateToolCallConstraintState(context, tokenId);
        if (context == nullptr || tokenId < 0 ||
            !context->generationConfig
                    .tool_call_content_sampling_enabled) {
            return;
        }

        std::string tokenText = this->weight.tokenizer.DecodeTokens(
                std::vector<int>{tokenId});
        // The generic name/parameter constraint already appended this token.
        // Without those constraints K3 still needs its own rolling text
        // buffer in order to recognize XTML string arguments.
        if (!baseTracksText) {
            context->toolCallConstraintGeneratedText += tokenText;
        }
        std::string &text = context->toolCallConstraintGeneratedText;
        int &state = context->intParams["kimi_k3_tool_content_state"];
        int &contentTokens =
                context->intParams["kimi_k3_tool_content_tokens"];
        if (state == 1) {
            contentTokens++;
        }
        if (state == 1 && tokenText.find("<|close|>") != std::string::npos) {
            // The close marker itself is sampled as the end of the long value;
            // render the remaining argument/call structure greedily.
            state = 2;
        }
        if (state == 2) {
            auto openPos = text.rfind("<|open|>argument");
            auto closePos = text.rfind(
                    "<|close|>argument<|sep|>");
            if (closePos != std::string::npos &&
                (openPos == std::string::npos || closePos > openPos)) {
                state = 0;
            }
        }
        if (state == 0) {
            auto openPos = text.rfind("<|open|>argument");
            auto closePos = text.rfind(
                    "<|close|>argument<|sep|>");
            if (openPos != std::string::npos &&
                (closePos == std::string::npos || openPos > closePos)) {
                auto headerEnd = text.find("<|sep|>", openPos);
                if (headerEnd != std::string::npos) {
                    std::string header = text.substr(
                            openPos, headerEnd + 7 - openPos);
                    bool longString =
                            header.find("type=\"string\"") !=
                                    std::string::npos &&
                            (header.find("key=\"content\"") !=
                                     std::string::npos ||
                             header.find("key=\"patch\"") !=
                                     std::string::npos ||
                             header.find("key=\"command\"") !=
                                     std::string::npos);
                    if (longString) {
                        state = 1;
                        contentTokens = 0;
                    }
                }
            }
        }
        const size_t maxTrackedBytes = 8192;
        if (text.size() > maxTrackedBytes) {
            text.erase(0, text.size() - maxTrackedBytes);
        }
    }

    void KimiK3Model::OnResponseContextCreated(ResponseContext *context) {
        if (context == nullptr || !dsparkEnabled) {
            return;
        }
        std::shared_ptr<DsparkHistoryCacheMemory> pending;
        {
            std::lock_guard<std::mutex> guard(dsparkHistoryCacheMutex);
            pending.swap(pendingDsparkHistoryCache);
        }
        if (pending != nullptr) {
            std::lock_guard<std::mutex> forwardGuard(forwardLocker);
            RestoreDsparkHistoryCache(*pending, context);
            return;
        }
        std::lock_guard<std::mutex> guard(dsparkContextMutex);
        dsparkContexts.erase(&context->pastKeyValues);
    }

    void KimiK3Model::OnResponseContextRemoved(ResponseContext *context) {
        if (context == nullptr || !dsparkEnabled) {
            return;
        }
        std::lock_guard<std::mutex> guard(dsparkContextMutex);
        dsparkContexts.erase(&context->pastKeyValues);
    }

    bool KimiK3Model::TryRestoreHistoryCache(
            std::vector<int> &inputTokens, int &cacheLen) {
        if (!dsparkEnabled || !saveHistoryChat || inputTokens.size() <= 1) {
            return false;
        }
        std::shared_ptr<DsparkHistoryCacheMemory> best;
        size_t bestLength = 0;
        {
            std::lock_guard<std::mutex> guard(dsparkHistoryCacheMutex);
            for (auto &item : dsparkHistoryCache) {
                const std::vector<int> &cachedTokens = item.first;
                if (cachedTokens.size() <= bestLength ||
                    cachedTokens.size() >= inputTokens.size() ||
                    !std::equal(cachedTokens.begin(), cachedTokens.end(),
                                inputTokens.begin())) {
                    continue;
                }
                best = item.second;
                bestLength = cachedTokens.size();
            }
            if (best != nullptr) {
                best->flushTime = ++dsparkHistoryCacheFlushTime;
                pendingDsparkHistoryCache = best;
            }
        }
        if (best == nullptr) {
            return false;
        }
        inputTokens.erase(inputTokens.begin(),
                          inputTokens.begin() + bestLength);
        cacheLen = (int)bestLength;
        return true;
    }

    KimiK3Model::DsparkContext &KimiK3Model::GetDsparkContext(
            std::vector<std::pair<Data, Data>> &pastKeyValues) {
        std::lock_guard<std::mutex> guard(dsparkContextMutex);
        auto *key = &pastKeyValues;
        auto &entry = dsparkContexts[key];
        if (entry == nullptr) {
            entry.reset(new DsparkContext());
            entry->adaptiveDraftLimit = dsparkBlockSize;
            entry->draftKeyValues.resize(dsparkLayers);
            entry->kdaSnapshots.resize(block_cnt);
            entry->replay.resize(block_cnt);
        }
        return *entry;
    }

    void KimiK3Model::RecordDsparkHistoryCache(
            const std::vector<int> &inputTokens,
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            const DsparkContext &context) {
        if (!saveHistoryChat || inputTokens.size() <= 1 ||
            context.committedTokens != (int)inputTokens.size() ||
            !context.pending.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> guard(dsparkHistoryCacheMutex);
            auto existing = dsparkHistoryCache.find(inputTokens);
            if (existing != dsparkHistoryCache.end()) {
                existing->second->flushTime =
                    ++dsparkHistoryCacheFlushTime;
                return;
            }
        }

        auto memory = std::make_shared<DsparkHistoryCacheMemory>();
        memory->inputTokens = inputTokens;
        memory->adaptiveDraftLimit = context.adaptiveDraftLimit;
        const bool moveToCpu = GetHistoryCacheInCPU();
        auto copyTensor = [&](const Data &source, Data &target) {
            if (source.isPagedKVCache) {
                AssertInFastLLM(
                    source.pagedKVCacheData != nullptr &&
                    !source.pageIndex.empty(),
                    "DSpark paged history cache descriptor is incomplete.");
                target.name = source.name;
                target.cacheUid = source.cacheUid;
                target.isKVCache = source.isKVCache;
                target.isPagedKVCache = true;
                target.pagedKVCacheData = source.pagedKVCacheData;
                target.pageLen = source.pageLen;
                target.pageIndex = source.pageIndex;
                target.lastPageLen = source.lastPageLen;
                target.dataType = source.dataType;
                target.UpdateUnitSize();
                target.dims = source.dims;
                target.strides = source.strides;
                target.dataDevice = source.dataDevice;
                target.dataDeviceIds = source.dataDeviceIds;
                target.pagedKVCacheData->Pick(target.pageIndex);
                return;
            }
            if (moveToCpu) {
                target.name = source.name;
                target.isKVCache = source.isKVCache;
                target.isLinearAttention = source.isLinearAttention;
                target.isLinearAttentionTransposed =
                    source.isLinearAttentionTransposed;
                target.cacheUid = source.cacheUid;
                target.dataType = source.dataType;
                target.UpdateUnitSize();
                if (source.dims.empty()) {
                    target.lockInCPU = true;
                    return;
                }
                target.dataDevice = DataDevice::CPU;
                if (!source.expansionDims.empty() &&
                    source.expansionDims != source.dims) {
                    target.Expansion(source.expansionDims);
                    target.Resize(source.dims);
                    target.Allocate();
                } else {
                    target.Resize(source.dims);
                    target.Allocate();
                }
                const size_t bytes = source.GetBytes();
                AssertInFastLLM(
                    target.cpuData != nullptr &&
                    bytes <= target.expansionBytes,
                    "DSpark CPU history cache allocation failed.");
                if (source.dataDevice == DataDevice::CPU) {
                    AssertInFastLLM(
                        source.cpuData != nullptr,
                        "DSpark history cache source has no CPU data.");
                    std::memcpy(target.cpuData, source.cpuData, bytes);
                } else {
#ifdef USE_CUDA
                    AssertInFastLLM(
                        source.cudaData != nullptr,
                        "DSpark history cache source has no CUDA data.");
                    const int originalDevice = FastllmCudaGetDevice();
                    int sourceDevice = GetPointerDeviceId(source.cudaData);
                    if (sourceDevice < 0 &&
                        !source.dataDeviceIds.empty()) {
                        sourceDevice = source.dataDeviceIds[0];
                    }
                    AssertInFastLLM(
                        sourceDevice >= 0,
                        "DSpark history cache source GPU is unknown.");
                    FastllmCudaSetDevice(sourceDevice);
                    FastllmCudaCopyFromDeviceToHost(
                        target.cpuData, source.cudaData, bytes);
                    FastllmCudaSetDevice(originalDevice);
#else
                    ErrorInFastLLM(
                        "DSpark CUDA cache cannot be copied in a CPU build.");
#endif
                }
                target.lockInCPU = true;
                return;
            }

#ifdef USE_CUDA
            int originalDevice = -1;
            if (source.dataDevice == DataDevice::CUDA &&
                source.cudaData != nullptr) {
                originalDevice = FastllmCudaGetDevice();
                int sourceDevice = GetPointerDeviceId(source.cudaData);
                if (sourceDevice < 0 && !source.dataDeviceIds.empty()) {
                    sourceDevice = source.dataDeviceIds[0];
                }
                if (sourceDevice >= 0) {
                    FastllmCudaSetDevice(sourceDevice);
                }
            }
#endif
            target.CopyFrom(source);
#ifdef USE_CUDA
            if (originalDevice >= 0) {
                FastllmCudaSetDevice(originalDevice);
            }
#endif
        };
        auto copyCache = [&](const std::vector<std::pair<Data, Data>> &source,
                             std::vector<std::pair<Data, Data>> &target) {
            target.resize(source.size());
            for (int index = 0; index < (int)source.size(); index++) {
                copyTensor(source[index].first, target[index].first);
                copyTensor(source[index].second, target[index].second);
            }
        };
        copyCache(pastKeyValues, memory->targetKeyValues);
        copyCache(context.draftKeyValues, memory->draftKeyValues);

        std::lock_guard<std::mutex> guard(dsparkHistoryCacheMutex);
        auto existing = dsparkHistoryCache.find(inputTokens);
        if (existing != dsparkHistoryCache.end()) {
            existing->second->flushTime = ++dsparkHistoryCacheFlushTime;
            return;
        }
        while ((int)dsparkHistoryCache.size() >=
               dsparkHistoryCacheMaxRecords) {
            auto oldest = dsparkHistoryCache.end();
            for (auto it = dsparkHistoryCache.begin();
                 it != dsparkHistoryCache.end(); ++it) {
                if (oldest == dsparkHistoryCache.end() ||
                    it->second->flushTime < oldest->second->flushTime) {
                    oldest = it;
                }
            }
            if (oldest == dsparkHistoryCache.end()) {
                break;
            }
            dsparkHistoryCache.erase(oldest);
        }
        memory->flushTime = ++dsparkHistoryCacheFlushTime;
        dsparkHistoryCache[inputTokens] = std::move(memory);
    }

    void KimiK3Model::RestoreDsparkHistoryCache(
            const DsparkHistoryCacheMemory &memory,
            ResponseContext *responseContext) {
        AssertInFastLLM(responseContext != nullptr &&
                        memory.inputTokens.size() > 1 &&
                        memory.targetKeyValues.size() ==
                            (size_t)block_cnt &&
                        memory.draftKeyValues.size() ==
                            (size_t)dsparkLayers,
                        "DSpark history cache snapshot is incomplete.");
        responseContext->pastKeyValues.resize(block_cnt);
        auto restoreTensor = [](const Data &source, Data &target) {
            if (!source.isPagedKVCache) {
#ifdef USE_CUDA
                const int originalDevice = FastllmCudaGetDevice();
                if (source.dataDevice == DataDevice::CUDA &&
                    source.cudaData != nullptr) {
                    int sourceDevice = GetPointerDeviceId(source.cudaData);
                    if (sourceDevice < 0 &&
                        !source.dataDeviceIds.empty()) {
                        sourceDevice = source.dataDeviceIds[0];
                    }
                    AssertInFastLLM(
                        sourceDevice >= 0,
                        "DSpark history cache source GPU is unknown.");
                    FastllmCudaSetDevice(sourceDevice);
                }
#endif
                target.CopyFrom(source);
#ifdef USE_CUDA
                FastllmCudaSetDevice(originalDevice);
#endif
                return;
            }
            AssertInFastLLM(
                source.pagedKVCacheData != nullptr &&
                !source.pageIndex.empty() &&
                source.pageLen > 0 &&
                source.lastPageLen > 0 &&
                source.lastPageLen <= source.pageLen,
                "DSpark paged history cache snapshot is incomplete.");
            target.name = source.name;
            target.cacheUid = source.cacheUid;
            target.isKVCache = source.isKVCache;
            target.isPagedKVCache = true;
            target.pagedKVCacheData = source.pagedKVCacheData;
            target.pageLen = source.pageLen;
            target.pageIndex = source.pageIndex;
            target.lastPageLen = source.lastPageLen;
            target.dataType = source.dataType;
            target.UpdateUnitSize();
            target.dims = source.dims;
            target.strides = source.strides;
            target.dataDevice = source.dataDevice;
            target.dataDeviceIds = source.dataDeviceIds;
            target.pagedKVCacheData->Pick(target.pageIndex);

            // Full pages are immutable and may be shared. The final partial
            // page must be private because the restored request appends its
            // uncached suffix into that page.
            if (target.lastPageLen == target.pageLen) {
                return;
            }
            PagedCacheManager *manager = target.pagedKVCacheData;
            AssertInFastLLM(
                !manager->dims.empty() && manager->dims[0] > 0,
                "DSpark paged history cache manager has an invalid shape.");
            const size_t pageBytes =
                manager->GetBytes() / (size_t)manager->dims[0];
            const int sharedPage = target.pageIndex.back();
            const int privatePage = manager->GetUnusedPageIndex(true);
            if (manager->dataDevice == DataDevice::CPU) {
                AssertInFastLLM(
                    manager->cpuData != nullptr,
                    "DSpark paged history cache manager has no CPU data.");
                std::memcpy(
                    manager->cpuData + (size_t)privatePage * pageBytes,
                    manager->cpuData + (size_t)sharedPage * pageBytes,
                    pageBytes);
            } else {
#ifdef USE_CUDA
                AssertInFastLLM(
                    manager->cudaData != nullptr,
                    "DSpark paged history cache manager has no CUDA data.");
                const int originalDevice = FastllmCudaGetDevice();
                int cacheDevice = GetPointerDeviceId(manager->cudaData);
                if (cacheDevice < 0 &&
                    !manager->dataDeviceIds.empty()) {
                    cacheDevice = manager->dataDeviceIds[0];
                }
                AssertInFastLLM(
                    cacheDevice >= 0,
                    "DSpark paged history cache GPU is unknown.");
                FastllmCudaSetDevice(cacheDevice);
                FastllmCudaCopyFromDeviceToDevice(
                    (uint8_t*)manager->cudaData +
                        (size_t)privatePage * pageBytes,
                    (uint8_t*)manager->cudaData +
                        (size_t)sharedPage * pageBytes,
                    pageBytes);
                FastllmCudaSetDevice(originalDevice);
#else
                ErrorInFastLLM(
                    "DSpark CUDA paged cache cannot be restored in a CPU build.");
#endif
            }
            manager->ReleasePageIndex(sharedPage);
            target.pageIndex.back() = privatePage;
        };
        for (int index = 0; index < block_cnt; index++) {
            restoreTensor(
                memory.targetKeyValues[index].first,
                responseContext->pastKeyValues[index].first);
            restoreTensor(
                memory.targetKeyValues[index].second,
                responseContext->pastKeyValues[index].second);
        }

        auto restored = std::make_unique<DsparkContext>();
        restored->initialized = true;
        restored->committedTokens = (int)memory.inputTokens.size();
        restored->adaptiveDraftLimit = std::max(
            1, std::min(dsparkBlockSize, memory.adaptiveDraftLimit));
        restored->draftKeyValues.resize(dsparkLayers);
        for (int index = 0; index < dsparkLayers; index++) {
            restored->draftKeyValues[index].first.CopyFrom(
                memory.draftKeyValues[index].first);
            restored->draftKeyValues[index].second.CopyFrom(
                memory.draftKeyValues[index].second);
        }
        restored->kdaSnapshots.resize(block_cnt);
        restored->replay.resize(block_cnt);
        restored->historyTokens = memory.inputTokens;

        std::lock_guard<std::mutex> guard(dsparkContextMutex);
        dsparkContexts[&responseContext->pastKeyValues] =
            std::move(restored);
    }

    void KimiK3Model::EnsureDsparkRotary(int positions) {
        if (positions <= dsparkRotaryCapacity) {
            return;
        }
        int capacity = std::max(4096, dsparkRotaryCapacity);
        while (capacity < positions) {
            capacity = std::min(max_positions,
                                std::max(capacity + 1, capacity * 2));
            AssertInFastLLM(capacity >= positions || capacity < max_positions,
                            "DSpark position exceeds the target context window.");
        }
        std::vector<float> sinValues(
            (size_t)capacity * dsparkHeadDim, 0.0f);
        std::vector<float> cosValues(
            (size_t)capacity * dsparkHeadDim, 0.0f);
        std::vector<float> inverseFrequency;
        inverseFrequency.reserve(dsparkHeadDim / 2);
        for (int channel = 0; channel < dsparkHeadDim; channel += 2) {
            inverseFrequency.push_back(
                1.0f / std::pow(10000.0f,
                                (float)channel / dsparkHeadDim));
        }
        for (int position = 0; position < capacity; position++) {
            for (int channel = 0;
                 channel < (int)inverseFrequency.size(); channel++) {
                float angle = position * inverseFrequency[channel];
                sinValues[(size_t)position * dsparkHeadDim + channel] =
                    std::sin(angle);
                cosValues[(size_t)position * dsparkHeadDim + channel] =
                    std::cos(angle);
            }
        }
        Data newSin(DataType::FLOAT32,
                    {capacity, dsparkHeadDim}, sinValues);
        Data newCos(DataType::FLOAT32,
                    {capacity, dsparkHeadDim}, cosValues);
        dsparkSinData.CopyFrom(newSin);
        dsparkCosData.CopyFrom(newCos);
        dsparkRotaryCapacity = capacity;
    }

    int KimiK3Model::SampleTargetHidden(
            Data &hiddenStates,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *logits) {
        AssertInFastLLM(hiddenStates.dims.size() == 3 &&
                        hiddenStates.dims[0] == 1,
                        "Kimi-K3 target hidden shape is invalid.");
        int sequence = hiddenStates.dims[1];
        Data lastHidden;
        if (sequence > 1) {
            Split(hiddenStates, 1, sequence - 1, sequence, lastHidden);
        } else {
            Copy(hiddenStates, lastHidden);
        }
        Data outputLogits;
        Linear(lastHidden, weight["language_model.lm_head.weight"],
               Data(), outputLogits);
        ToDataType(outputLogits, DataType::FLOAT32);
        outputLogits.ToDevice(DataDevice::CPU);
        if (generationConfig.output_logits && logits != nullptr) {
            int vocabulary = outputLogits.dims.back();
            logits->resize(vocabulary);
            std::memcpy(logits->data(), outputLogits.cpuData,
                        (size_t)vocabulary * sizeof(float));
        }
        if (generationConfig.IsSimpleGreedy()) {
            Data topk;
            TopK(outputLogits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            return (int)(((float*)topk.cpuData)[0] + 1e-3f);
        }
        if (!lastTokens.units.empty()) {
            return LLMSampling(outputLogits, 0, generationConfig,
                               lastTokens.units[0]);
        }
        return LLMSamplingOnly(outputLogits, 0, generationConfig);
    }

    void KimiK3Model::AppendDsparkTargetHidden(
            const TargetRunCapture &capture, int tokens,
            DsparkContext &context) {
        AssertInFastLLM(tokens > 0,
                        "DSpark must append at least one target token.");
        ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
        Data combined;
        for (int feature = 0;
             feature < (int)dsparkTargetLayerIds.size(); feature++) {
            int layer = dsparkTargetLayerIds[feature];
            auto it = capture.targetHidden.find(layer);
            AssertInFastLLM(it != capture.targetHidden.end(),
                            "DSpark target hidden capture is incomplete.");
            Data selected;
            AssertInFastLLM(it->second.dims.size() == 3 &&
                            it->second.dims[1] >= tokens,
                            "DSpark captured target hidden has an invalid shape.");
            if (it->second.dims[1] == tokens) {
                Copy(it->second, selected);
            } else {
                Split(it->second, 1, 0, tokens, selected);
            }
            if (feature == 0) {
                Copy(selected, combined);
            } else {
                Data joined;
                Cat(combined, selected, -1, joined);
                Copy(joined, combined);
            }
        }

        Data projected, contextHidden;
        Linear(combined, weight["dspark.fc.weight"], Data(), projected);
        RMSNorm(projected, weight["dspark.hidden_norm.weight"],
                dsparkRmsNormEps, contextHidden);

        const int startPosition = context.committedTokens;
        EnsureDsparkRotary(startPosition + tokens);
        std::vector<float> positionValues(tokens);
        for (int token = 0; token < tokens; token++) {
            positionValues[token] = (float)(startPosition + token);
        }
        Data positions(DataType::FLOAT32, {1, tokens}, positionValues);
        for (int layerIndex = 0; layerIndex < dsparkLayers;
             layerIndex++) {
            ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
            const std::string layer = "dspark.layers." +
                std::to_string(layerIndex) + ".";
            Data key, value;
            Linear(contextHidden, weight[layer + "self_attn.k_proj.weight"],
                   Data(), key);
            Linear(contextHidden, weight[layer + "self_attn.v_proj.weight"],
                   Data(), value);
            key.Reshape({1, tokens, dsparkKvHeads, dsparkHeadDim});
            value.Reshape({1, tokens, dsparkKvHeads, dsparkHeadDim});
            RMSNorm(key, weight[layer + "self_attn.k_norm.weight"],
                    dsparkRmsNormEps, key);
            LlamaRotatePosition2D(key, positions, dsparkSinData,
                                  dsparkCosData, dsparkHeadDim);
            PermuteSelf(key, {0, 2, 1, 3});
            PermuteSelf(value, {0, 2, 1, 3});
            key.Reshape({dsparkKvHeads, tokens, dsparkHeadDim});
            value.Reshape({dsparkKvHeads, tokens, dsparkHeadDim});
            // FastLLM's non-paged CUDA Attention kernel currently supports
            // FP16/FP32, while the published DSpark checkpoint is BF16.
            // Keep the transformer in BF16 and store only the draft attention
            // K/V boundary in FP16. Draft logits are always verified by the
            // BF16 target model, so this cannot alter the committed greedy
            // token stream; it may only affect the speculative acceptance rate.
            ToDataType(key, DataType::FLOAT16);
            ToDataType(value, DataType::FLOAT16);
            AppendSequenceCache(context.draftKeyValues[layerIndex].first,
                                key);
            AppendSequenceCache(context.draftKeyValues[layerIndex].second,
                                value);
        }
        context.committedTokens += tokens;
    }

    KimiK3Model::DsparkDraftProposal KimiK3Model::RunDsparkDraft(
            int anchorToken, DsparkContext &context) {
        AssertInFastLLM(context.initialized &&
                        (int)context.draftKeyValues.size() == dsparkLayers,
                        "DSpark context is not initialized.");
        ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
        EnsureDsparkRotary(context.committedTokens + dsparkBlockSize);

        std::vector<float> tokenValues(dsparkBlockSize,
                                       (float)dsparkMaskTokenId);
        tokenValues[0] = (float)anchorToken;
        Data inputIds(DataType::FLOAT32,
                      {1, dsparkBlockSize}, tokenValues);
        Data hiddenStates;
        Embedding(inputIds,
                  weight[languagePrefix + "embed_tokens.weight"],
                  hiddenStates);
        ToDataType(hiddenStates, DataType::BFLOAT16);

        std::vector<float> positionValues(dsparkBlockSize);
        for (int token = 0; token < dsparkBlockSize; token++) {
            positionValues[token] =
                (float)(context.committedTokens + token);
        }
        Data positions(DataType::FLOAT32,
                       {1, dsparkBlockSize}, positionValues);

        for (int layerIndex = 0; layerIndex < dsparkLayers;
             layerIndex++) {
            ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
            const std::string layer = "dspark.layers." +
                std::to_string(layerIndex) + ".";
            Data normalized;
            RMSNorm(hiddenStates,
                    weight[layer + "input_layernorm.weight"],
                    dsparkRmsNormEps, normalized);

            Data query, key, value;
            Linear(normalized, weight[layer + "self_attn.q_proj.weight"],
                   Data(), query);
            Linear(normalized, weight[layer + "self_attn.k_proj.weight"],
                   Data(), key);
            Linear(normalized, weight[layer + "self_attn.v_proj.weight"],
                   Data(), value);
            query.Reshape(
                {1, dsparkBlockSize, dsparkHeads, dsparkHeadDim});
            key.Reshape(
                {1, dsparkBlockSize, dsparkKvHeads, dsparkHeadDim});
            value.Reshape(
                {1, dsparkBlockSize, dsparkKvHeads, dsparkHeadDim});
            RMSNorm(query, weight[layer + "self_attn.q_norm.weight"],
                    dsparkRmsNormEps, query);
            RMSNorm(key, weight[layer + "self_attn.k_norm.weight"],
                    dsparkRmsNormEps, key);
            LlamaRotatePosition2D(query, positions, dsparkSinData,
                                  dsparkCosData, dsparkHeadDim);
            LlamaRotatePosition2D(key, positions, dsparkSinData,
                                  dsparkCosData, dsparkHeadDim);
            PermuteSelf(query, {0, 2, 1, 3});
            PermuteSelf(key, {0, 2, 1, 3});
            PermuteSelf(value, {0, 2, 1, 3});
            query.Reshape(
                {dsparkHeads, dsparkBlockSize, dsparkHeadDim});
            key.Reshape(
                {dsparkKvHeads, dsparkBlockSize, dsparkHeadDim});
            value.Reshape(
                {dsparkKvHeads, dsparkBlockSize, dsparkHeadDim});
            ToDataType(query, DataType::FLOAT16);
            ToDataType(key, DataType::FLOAT16);
            ToDataType(value, DataType::FLOAT16);

            Data &keyCache = context.draftKeyValues[layerIndex].first;
            Data &valueCache = context.draftKeyValues[layerIndex].second;
            AssertInFastLLM(
                keyCache.dims.size() == 3 &&
                valueCache.dims.size() == 3 &&
                keyCache.dims[1] == context.committedTokens &&
                valueCache.dims[1] == context.committedTokens,
                "DSpark draft KV cache is out of sync with the target.");
            const int oldKeyTokens = keyCache.dims[1];
            const int oldValueTokens = valueCache.dims[1];
            AppendSequenceCache(keyCache, key);
            AppendSequenceCache(valueCache, value);
            Data attentionHeads;
            Attention(query, keyCache, valueCache, *GetEmptyData(),
                      attentionHeads, dsparkHeads / dsparkKvHeads,
                      1.0f / std::sqrt((float)dsparkHeadDim), 2);
            ResizeSequenceCache(keyCache, oldKeyTokens);
            ResizeSequenceCache(valueCache, oldValueTokens);

            PermuteSelf(attentionHeads, {1, 0, 2});
            attentionHeads.Reshape(
                {1, dsparkBlockSize, dsparkHeads * dsparkHeadDim});
            ToDataType(attentionHeads, DataType::BFLOAT16);
            Data attentionOutput;
            Linear(attentionHeads,
                   weight[layer + "self_attn.o_proj.weight"],
                   Data(), attentionOutput);
            AddTo(hiddenStates, attentionOutput);

            RMSNorm(hiddenStates,
                    weight[layer + "post_attention_layernorm.weight"],
                    dsparkRmsNormEps, normalized);
            Data gate, up;
            Linear(normalized, weight[layer + "mlp.gate_proj.weight"],
                   Data(), gate);
            Linear(normalized, weight[layer + "mlp.up_proj.weight"],
                   Data(), up);
            // The generic CUDA SiLU/Mul kernels have the same FP16/FP32
            // boundary as Attention. Keep the surrounding draft projections
            // and residual stream in BF16.
            ToDataType(gate, DataType::FLOAT16);
            ToDataType(up, DataType::FLOAT16);
            Silu(gate, gate);
            MulTo(gate, up);
            ToDataType(gate, DataType::BFLOAT16);
            Data mlpOutput;
            Linear(gate, weight[layer + "mlp.down_proj.weight"],
                   Data(), mlpOutput);
            AddTo(hiddenStates, mlpOutput);
        }

        Data normalizedDraft;
        RMSNorm(hiddenStates, weight["dspark.norm.weight"],
                dsparkRmsNormEps, normalizedDraft);
        Data baseLogits;
        Linear(normalizedDraft, weight["language_model.lm_head.weight"],
               Data(), baseLogits);

        DsparkDraftProposal proposal;
        proposal.tokens.reserve(dsparkBlockSize);
        std::vector<float> previousTokens;
        previousTokens.reserve(dsparkBlockSize);
        int previousToken = anchorToken;
        for (int step = 0; step < dsparkBlockSize; step++) {
            previousTokens.push_back((float)previousToken);
            Data stepLogits;
            Split(baseLogits, 1, step, step + 1, stepLogits);
            Data previousIds(
                DataType::FLOAT32, {1, 1}, {(float)previousToken});
            Data markovLatent, markovBias;
            Embedding(previousIds,
                      weight["dspark.markov_head.markov_w1.weight"],
                      markovLatent);
            Linear(markovLatent,
                   weight["dspark.markov_head.markov_w2.weight"],
                   Data(), markovBias);
            ToDataType(stepLogits, DataType::FLOAT32);
            ToDataType(markovBias, DataType::FLOAT32);
            AddTo(stepLogits, markovBias);
            Data topk;
            TopK(stepLogits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            previousToken =
                (int)(((float*)topk.cpuData)[0] + 1e-3f);
            proposal.tokens.push_back(previousToken);
        }

        if (dsparkConfidenceThreshold <= 0.0f) {
            proposal.confidence.assign(dsparkBlockSize, 1.0f);
            return proposal;
        }

        // The checkpoint predicts the conditional acceptance probability for
        // every draft position from the normalized draft hidden state and the
        // Markov embedding of [anchor, draft[:-1]].  Compute all seven values
        // in one small linear call so adaptive scheduling adds only one device
        // synchronization, rather than one synchronization per draft token.
        Data previousIds(DataType::FLOAT32,
                         {1, dsparkBlockSize}, previousTokens);
        Data markovEmbeddings;
        Embedding(previousIds,
                  weight["dspark.markov_head.markov_w1.weight"],
                  markovEmbeddings);
        Data confidenceHidden, confidenceMarkov;
        Copy(normalizedDraft, confidenceHidden);
        Copy(markovEmbeddings, confidenceMarkov);
        ToDataType(confidenceHidden, DataType::FLOAT32);
        ToDataType(confidenceMarkov, DataType::FLOAT32);
        Data confidenceFeatures;
        Cat(confidenceHidden, confidenceMarkov, -1, confidenceFeatures);
        Data confidenceLogits;
        Linear(confidenceFeatures,
               weight["dspark.confidence_head.proj.weight"],
               weight["dspark.confidence_head.proj.bias"],
               confidenceLogits);
        ToDataType(confidenceLogits, DataType::FLOAT32);
        confidenceLogits.ToDevice(DataDevice::CPU);
        AssertInFastLLM(
            confidenceLogits.dims == std::vector<int>({1, dsparkBlockSize, 1}),
            "DSpark confidence head output has an invalid shape.");
        proposal.confidence.resize(dsparkBlockSize);
        const float *confidenceData =
            (const float*)confidenceLogits.cpuData;
        for (int step = 0; step < dsparkBlockSize; step++) {
            const float value = confidenceData[step];
            const float probability = value >= 0.0f ?
                1.0f / (1.0f + std::exp(-value)) :
                std::exp(value) / (1.0f + std::exp(value));
            AssertInFastLLM(std::isfinite(probability),
                            "DSpark confidence head produced NaN or Inf.");
            proposal.confidence[step] = probability;
        }
        return proposal;
    }

    int KimiK3Model::SelectDsparkVerifyDrafts(
            const DsparkDraftProposal &proposal,
            const DsparkContext &context) const {
        AssertInFastLLM(
            (int)proposal.tokens.size() == dsparkBlockSize &&
            (int)proposal.confidence.size() == dsparkBlockSize,
            "DSpark proposal has an invalid length.");
        // A zero threshold is the explicit fixed-block fallback.  Besides
        // making the policy configurable, this gives correctness/performance
        // regressions a byte-for-byte equivalent of the original path.
        if (dsparkConfidenceThreshold <= 0.0f) {
            return dsparkBlockSize;
        }

        int confidenceLimit = 0;
        while (confidenceLimit < dsparkBlockSize &&
               proposal.confidence[confidenceLimit] >=
                   dsparkConfidenceThreshold) {
            confidenceLimit++;
        }
        const int rollingLimit = std::max(
            1, std::min(dsparkBlockSize, context.adaptiveDraftLimit));
        return std::min(confidenceLimit, rollingLimit);
    }

    void KimiK3Model::UpdateDsparkAdaptiveLimit(
            int verifyDrafts, int acceptedDrafts,
            DsparkContext &context) const {
        if (dsparkConfidenceThreshold <= 0.0f || verifyDrafts == 0) {
            return;
        }
        AssertInFastLLM(
            acceptedDrafts >= 0 && acceptedDrafts <= verifyDrafts,
            "DSpark accepted draft count is invalid.");
        if (acceptedDrafts < verifyDrafts) {
            // Keep one probe after the last known-good position.  This reacts
            // immediately to task/domain shifts while retaining a path for the
            // limit to grow again when later blocks improve.
            context.adaptiveDraftLimit = std::max(1, acceptedDrafts + 1);
        } else if (verifyDrafts >= context.adaptiveDraftLimit) {
            context.adaptiveDraftLimit = std::min(
                dsparkBlockSize, context.adaptiveDraftLimit + 1);
        }
    }

    void KimiK3Model::SnapshotKdaCaches(
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            DsparkContext &context) {
        AssertInFastLLM((int)pastKeyValues.size() >= block_cnt,
                        "Kimi-K3 target cache table is incomplete.");
        if ((int)context.kdaSnapshots.size() < block_cnt) {
            context.kdaSnapshots.resize(block_cnt);
        }
        for (int layerIndex = 0; layerIndex < block_cnt; layerIndex++) {
            if (!kdaLayers[layerIndex]) {
                continue;
            }
            AssertInFastLLM(
                !pastKeyValues[layerIndex].first.dims.empty() &&
                !pastKeyValues[layerIndex].second.dims.empty(),
                "DSpark requires initialized KDA caches before verification.");
            context.kdaSnapshots[layerIndex].first.CopyFrom(
                pastKeyValues[layerIndex].first);
            context.kdaSnapshots[layerIndex].second.CopyFrom(
                pastKeyValues[layerIndex].second);
        }
    }

    void KimiK3Model::CommitTargetVerification(
            int oldTokens, int commitTokens,
            int verifyTokens,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            DsparkContext &context) {
        AssertInFastLLM(commitTokens >= 1 &&
                        commitTokens <= verifyTokens &&
                        verifyTokens >= 1 &&
                        verifyTokens <= dsparkBlockSize + 1,
                        "DSpark target commit length is invalid.");
        if (commitTokens < verifyTokens) {
            AssertInFastLLM((int)context.replay.size() >= block_cnt,
                            "DSpark KDA replay capture is incomplete.");
            for (int layerIndex = 0; layerIndex < block_cnt;
                 layerIndex++) {
                if (!kdaLayers[layerIndex]) {
                    continue;
                }
                ApplyDeviceMap(deviceMap, layerIndex, block_cnt);
                pastKeyValues[layerIndex].first.CopyFrom(
                    context.kdaSnapshots[layerIndex].first);
                pastKeyValues[layerIndex].second.CopyFrom(
                    context.kdaSnapshots[layerIndex].second);

                KdaReplayCapture &replay = context.replay[layerIndex];
                AssertInFastLLM(
                    replay.qProjected.dims.size() >= 2 &&
                    replay.qProjected.dims[1] == verifyTokens,
                    "DSpark KDA projection replay data is incomplete.");
                KimiK3UpdatePackedConvCache(
                    replay.qProjected, replay.kProjected,
                    replay.vProjected, shortConvKernel - 1,
                    commitTokens, pastKeyValues[layerIndex].first);
                const std::string layer = languagePrefix + "layers." +
                    std::to_string(layerIndex) + ".";
                KimiK3RecurrentKDAUpdateState(
                    replay.k, replay.v,
                    replay.rawGate, replay.rawBeta,
                    weight[layer + "self_attn.A_log"],
                    weight[layer + "self_attn.dt_bias"], gateLowerBound,
                    commitTokens, pastKeyValues[layerIndex].second);
            }
        }

        const int committedLength = oldTokens + commitTokens;
        for (int layerIndex = 0; layerIndex < block_cnt; layerIndex++) {
            if (kdaLayers[layerIndex]) {
                continue;
            }
            Data &key = pastKeyValues[layerIndex].first;
            Data &value = pastKeyValues[layerIndex].second;
            AssertInFastLLM(
                key.dims.size() == 3 && value.dims.size() == 3 &&
                key.dims[1] == oldTokens + verifyTokens &&
                value.dims[1] == oldTokens + verifyTokens,
                "DSpark MLA verification cache length is invalid.");
            if (commitTokens < verifyTokens) {
                ResizeSequenceCache(key, committedLength);
                ResizeSequenceCache(value, committedLength);
            }
        }
        ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
    }

    int KimiK3Model::ForwardDspark(
            const std::vector<int> &tokenIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *logits) {
        DsparkContext &context = GetDsparkContext(pastKeyValues);
        if (!context.pending.empty()) {
            AssertInFastLLM(
                tokenIds.size() == 1 &&
                tokenIds[0] == context.pending.front().expectedInput,
                "DSpark pending token stream is out of sync with the scheduler.");
            int output = context.pending.front().outputToken;
            context.pending.pop_front();
            return output;
        }

        if (!context.initialized || tokenIds.size() != 1) {
            AssertInFastLLM(
                context.historyTokens.empty() ||
                context.committedTokens ==
                    (int)context.historyTokens.size(),
                "DSpark history token state is out of sync.");
            TargetRunCapture capture;
            Data hiddenStates = RunLayersImpl(
                tokenIds, block_cnt, &pastKeyValues, &capture);
            AppendDsparkTargetHidden(
                capture, (int)tokenIds.size(), context);
            context.historyTokens.insert(
                context.historyTokens.end(),
                tokenIds.begin(), tokenIds.end());
            context.initialized = true;
            RecordDsparkHistoryCache(
                context.historyTokens, pastKeyValues, context);
            return SampleTargetHidden(hiddenStates, generationConfig,
                                      lastTokens, logits);
        }

        const int anchorToken = tokenIds[0];
        const int oldTokens = context.committedTokens;
        DsparkDraftProposal proposal = RunDsparkDraft(anchorToken, context);
        const int verifyDrafts = SelectDsparkVerifyDrafts(proposal, context);
        const int verifyTokens = verifyDrafts + 1;

        if (verifyDrafts > 0) {
            SnapshotKdaCaches(pastKeyValues, context);
        }
        std::vector<int> verifyIds;
        verifyIds.reserve(verifyTokens);
        verifyIds.push_back(anchorToken);
        verifyIds.insert(verifyIds.end(), proposal.tokens.begin(),
                         proposal.tokens.begin() + verifyDrafts);
        TargetRunCapture capture;
        capture.captureKda = verifyDrafts > 0;
        if (capture.captureKda) {
            capture.kda.swap(context.replay);
        }
        Data hiddenStates = RunLayersImpl(
            verifyIds, block_cnt, &pastKeyValues, &capture);
        if (capture.captureKda) {
            capture.kda.swap(context.replay);
        }

        Data targetLogits;
        Linear(hiddenStates, weight["language_model.lm_head.weight"],
               Data(), targetLogits);
        ToDataType(targetLogits, DataType::FLOAT32);
        const bool simpleGreedy = generationConfig.IsSimpleGreedy();
        int accepted = 0;
        int nextToken = -1;
        if (simpleGreedy) {
            Data targetTopK;
            TopK(targetLogits, targetTopK, 1);
            targetTopK.ToDevice(DataDevice::CPU);
            AssertInFastLLM(
                targetTopK.dims.size() == 3 &&
                targetTopK.dims[1] == verifyTokens &&
                targetTopK.dims.back() == 2,
                "DSpark target verification TopK shape is invalid.");
            const float *topKData = (const float*)targetTopK.cpuData;
            while (accepted < verifyDrafts) {
                int targetToken = (int)(
                    topKData[(size_t)accepted * 2] + 1e-3f);
                if (proposal.tokens[accepted] != targetToken) {
                    nextToken = targetToken;
                    break;
                }
                accepted++;
            }
            if (nextToken < 0) {
                nextToken = (int)(
                    topKData[(size_t)accepted * 2] + 1e-3f);
            }
        } else {
            AssertInFastLLM(
                targetLogits.dims.size() == 3 &&
                targetLogits.dims[1] == verifyTokens,
                "DSpark target sampling logits shape is invalid.");
            LastTokensUnit samplingTokens;
            if (!lastTokens.units.empty()) {
                samplingTokens = lastTokens.units[0];
            }
            while (accepted < verifyDrafts) {
                int sampled = LLMSampling(
                    targetLogits, accepted, generationConfig,
                    samplingTokens);
                if (proposal.tokens[accepted] != sampled) {
                    nextToken = sampled;
                    break;
                }
                samplingTokens.Push(sampled);
                accepted++;
            }
            if (nextToken < 0) {
                nextToken = LLMSampling(
                    targetLogits, accepted, generationConfig,
                    samplingTokens);
            }
        }
        const int commitTokens = accepted + 1;
        CommitTargetVerification(oldTokens, commitTokens, verifyTokens,
                                 pastKeyValues, context);
        AppendDsparkTargetHidden(capture, commitTokens, context);
        if (simpleGreedy) {
            UpdateDsparkAdaptiveLimit(verifyDrafts, accepted, context);
        }

        std::vector<int> outputs;
        outputs.reserve(commitTokens);
        outputs.insert(outputs.end(), proposal.tokens.begin(),
                       proposal.tokens.begin() + accepted);
        outputs.push_back(nextToken);
        AssertInFastLLM((int)outputs.size() == commitTokens,
                        "DSpark output/commit alignment failed.");
        for (int index = 1; index < (int)outputs.size(); index++) {
            context.pending.push_back(
                {outputs[index - 1], outputs[index]});
        }

        return outputs[0];
    }

    int KimiK3Model::Forward(
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *logits) {
        (void)attentionMask;
        (void)positionIds;
        AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
                        "Kimi-K3 Forward currently supports batch size 1.");
        if ((int)pastKeyValues.size() < block_cnt) {
            pastKeyValues.resize(block_cnt);
        }

        std::vector<int> tokenIds;
        Data currentIds;
        currentIds.CopyFrom(inputIds);
        currentIds.ToDevice(DataDevice::CPU);
        AssertInFastLLM(currentIds.dataType == DataType::FLOAT32,
                        "Kimi-K3 input ids must be float32 token indices.");
        const float *currentData = (const float*)currentIds.cpuData;
        for (uint64_t i = 0; i < currentIds.Count(0); i++) {
            tokenIds.push_back((int)(currentData[i] + 1e-3f));
        }
        if (dsparkEnabled &&
            (generationConfig.IsSimpleGreedy() ||
             generationConfig.tool_call_content_sampling_active) &&
            !generationConfig.output_logits) {
            return ForwardDspark(tokenIds, pastKeyValues, generationConfig,
                                 lastTokens, logits);
        }
        Data hiddenStates = RunLayersImpl(
            tokenIds, block_cnt, &pastKeyValues);
        return SampleTargetHidden(hiddenStates, generationConfig,
                                  lastTokens, logits);
    }

    bool KimiK3Model::NeedAttentionMask(int qlen, int klen) {
        (void)qlen;
        (void)klen;
        return false;
    }

    std::string KimiK3Model::MakeInput(
            const std::string &history, int round, const std::string &input) {
        (void)round;
        return history +
            "<|open|>message role=\"user\"<|sep|>" + input +
            "<|close|>message<|sep|><|end_of_msg|>"
            "<|open|>message role=\"assistant\"<|sep|>"
            "<|open|>response<|sep|>";
    }

    std::string KimiK3Model::MakeHistory(
            const std::string &history, int round,
            const std::string &input, const std::string &output) {
        (void)round;
        return history +
            "<|open|>message role=\"user\"<|sep|>" + input +
            "<|close|>message<|sep|><|end_of_msg|>"
            "<|open|>message role=\"assistant\"<|sep|>"
            "<|open|>response<|sep|>" + output +
            "<|close|>response<|sep|><|close|>message<|sep|>"
            "<|end_of_msg|>";
    }
}
