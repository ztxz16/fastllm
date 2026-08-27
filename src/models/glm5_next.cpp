#include "glm5_next.h"

#include "blocks/baseblock.h"
#include "gguf.h"
#include "utils.h"

#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>

namespace fastllm {
    const std::string Glm5NextModel::languagePrefix =
        "model.language_model.";

    namespace {
        void FlattenGlm5NextTextConfig(
                std::map<std::string, std::string> &dicts) {
            const std::string prefix = "text_config.";
            std::map<std::string, std::string> flattened;
            for (const auto &item : dicts) {
                if (item.first.rfind(prefix, 0) != 0) {
                    continue;
                }
                const std::string key = item.first.substr(prefix.size());
                if (!key.empty() && dicts.find(key) == dicts.end()) {
                    flattened[key] = item.second;
                }
            }
            dicts.insert(flattened.begin(), flattened.end());
        }

        std::vector<int> ParseGlm5NextIntegerList(
                const std::string &text) {
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
                    value = value * 10 + text[position] - '0';
                    position++;
                }
                values.push_back(negative ? -value : value);
            }
            return values;
        }

        void UnpackGlm5NextConvCache(
                Data &packed, int history, int channels,
                Data &qCache, Data &kCache, Data &vCache) {
            if (packed.dims.empty()) {
                return;
            }
            AssertInFastLLM(
                packed.dataType == DataType::BFLOAT16 &&
                packed.dims == std::vector<int>({3, history, channels}),
                "GLM-5.3 packed KDA convolution cache has an invalid shape.");
            Split(packed, 0, 0, 1, qCache);
            Split(packed, 0, 1, 2, kCache);
            Split(packed, 0, 2, 3, vCache);
        }

        void PackGlm5NextConvCache(
                Data &qCache, Data &kCache, Data &vCache, Data &packed) {
            Data qkCache, combined;
            Cat(qCache, kCache, 0, qkCache);
            Cat(qkCache, vCache, 0, combined);
            Copy(combined, packed);
            packed.SetKVCache();
            // This is a fixed-size convolution history, not a token-growing
            // attention cache.  Generic history-cache restore must copy it as
            // a whole instead of slicing dimension 1 by the matched prefix.
            packed.isLinearAttention = true;
        }

        void AppendGlm5NextSequenceCache(
                Data &cache, const Data &current, int allocationUnit = 64) {
            AssertInFastLLM(
                current.dims.size() == 3,
                "GLM-5.3 attention cache input must be [heads,tokens,dim].");
            if (cache.dims.empty() && cache.expansionDims.empty() &&
                cache.expansionSize == 0) {
                cache.dataType = current.dataType;
                cache.UpdateUnitSize();
                cache.dataDevice = current.dataDevice;
                cache.dataDeviceIds = current.dataDeviceIds;
            }
            cache.SetKVCache();
            AssertInFastLLM(
                cache.dataType == current.dataType,
                "GLM-5.3 attention cache dtype mismatch.");
            const int appendTokens = current.dims[1];
            while ((cache.dims.empty() &&
                    (cache.expansionDims.size() < 2 ||
                     appendTokens > cache.expansionDims[1])) ||
                   (!cache.dims.empty() &&
                    (cache.expansionDims.size() < 2 ||
                     cache.dims[1] + appendTokens >
                         cache.expansionDims[1]))) {
                std::vector<int> expanded;
                if (cache.dims.empty()) {
                    expanded = {
                        current.dims[0],
                        ((appendTokens - 1) / allocationUnit + 1) *
                            allocationUnit,
                        current.dims[2],
                    };
                } else {
                    expanded = cache.expansionDims.size() ==
                                       cache.dims.size() ?
                        cache.expansionDims : cache.dims;
                    expanded[1] +=
                        ((appendTokens - 1) / allocationUnit + 1) *
                            allocationUnit;
                }
                cache.Expansion(expanded);
            }
            CatDirect(cache, current, 1);
        }

        float Glm5NextReadFloat(const Data &data, uint64_t index) {
            if (data.dataType == DataType::FLOAT32) {
                return ((const float*)data.cpuData)[index];
            }
            if (data.dataType == DataType::FLOAT16) {
                return half_to_float(((const uint16_t*)data.cpuData)[index]);
            }
            if (data.dataType == DataType::BFLOAT16) {
                uint32_t bits =
                    (uint32_t)((const uint16_t*)data.cpuData)[index] << 16;
                float value;
                std::memcpy(&value, &bits, sizeof(value));
                return value;
            }
            ErrorInFastLLM("GLM-5.3 clamp received an unsupported dtype.");
            return 0.0f;
        }

        uint16_t Glm5NextFloatToBfloat16(float value) {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            const uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
            return (uint16_t)((bits + rounding) >> 16);
        }

        void Glm5NextWriteFloat(Data &data, uint64_t index, float value) {
            if (data.dataType == DataType::FLOAT32) {
                ((float*)data.cpuData)[index] = value;
            } else if (data.dataType == DataType::FLOAT16) {
                ((uint16_t*)data.cpuData)[index] = float_to_half(value);
            } else if (data.dataType == DataType::BFLOAT16) {
                ((uint16_t*)data.cpuData)[index] =
                    Glm5NextFloatToBfloat16(value);
            } else {
                ErrorInFastLLM(
                    "GLM-5.3 clamp received an unsupported dtype.");
            }
        }

        void Glm5NextClamp(
                Data &data, bool hasMin, float minValue,
                bool hasMax, float maxValue) {
            if (!hasMin && !hasMax) {
                return;
            }
#ifdef USE_CUDA
            if (data.dataDevice == DataDevice::CUDA &&
                data.cudaData != nullptr &&
                FastllmCudaClamp(
                    data, hasMin, minValue, hasMax, maxValue)) {
                return;
            }
#endif
            const DataDevice originalDevice = data.dataDevice;
            const std::vector<int> originalDeviceIds = data.dataDeviceIds;
            data.ToDevice(DataDevice::CPU);
            const uint64_t count = data.Count(0);
            for (uint64_t index = 0; index < count; index++) {
                float value = Glm5NextReadFloat(data, index);
                if (hasMin) {
                    value = std::max(value, minValue);
                }
                if (hasMax) {
                    value = std::min(value, maxValue);
                }
                Glm5NextWriteFloat(data, index, value);
            }
            data.ToDevice(originalDevice, originalDeviceIds);
        }

        void Glm5NextHcMean(const Data &input, Data &output) {
            AssertInFastLLM(
                input.dims.size() == 4 && input.dims[2] > 0,
                "GLM-5.3 final HC input must be [batch,sequence,hc,hidden].");
            const int hc = input.dims[2];
            Data inputFloat, meanFloat;
            ToDataType(input, inputFloat, DataType::FLOAT32);
            for (int index = 0; index < hc; index++) {
                Data selected;
                Split(inputFloat, 2, index, index + 1, selected);
                selected.Reshape(
                    {input.dims[0], input.dims[1], input.dims[3]});
                if (index == 0) {
                    Mul(selected, 1.0f / hc, meanFloat);
                } else {
                    AddTo(meanFloat, selected, 1.0f / hc);
                }
            }
            ToDataType(meanFloat, output, DataType::BFLOAT16);
        }

        bool Glm5NextEndsWith(
                const std::string &text, const std::string &suffix) {
            return text.size() >= suffix.size() &&
                   text.compare(text.size() - suffix.size(),
                                suffix.size(), suffix) == 0;
        }
    }

    Glm5NextModel::Glm5NextModel() {
        model_type = "glm5_next";
        model_struct = "glm5_next";
        canDoBatchForward = false;
        dataType = DataType::BFLOAT16;
        kvCacheDataType = DataType::BFLOAT16;
        moeAtype = DataType::BFLOAT16;
        defaultChunkedPrefillSize = 1024;

        weight.embeddingNames.insert(
            languagePrefix + "embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight",
            languagePrefix + "layers.*.self_attn.q_proj.weight",
            languagePrefix + "layers.*.self_attn.k_proj.weight",
            languagePrefix + "layers.*.self_attn.v_proj.weight",
            languagePrefix + "layers.*.self_attn.f_a_proj.weight",
            languagePrefix + "layers.*.self_attn.f_b_proj.weight",
            languagePrefix + "layers.*.self_attn.b_proj.weight",
            languagePrefix + "layers.*.self_attn.g_a_proj.weight",
            languagePrefix + "layers.*.self_attn.g_b_proj.weight",
            languagePrefix + "layers.*.self_attn.o_proj.weight",
            languagePrefix + "layers.*.self_attn.q_a_proj.weight",
            languagePrefix + "layers.*.self_attn.q_b_proj.weight",
            languagePrefix + "layers.*.self_attn.kv_a_proj_with_mqa.weight",
            languagePrefix + "layers.*.self_attn.kv_b_proj.weight",
            languagePrefix + "layers.*.mlp.gate.weight",
            languagePrefix + "layers.*.mlp.gate_proj.weight",
            languagePrefix + "layers.*.mlp.up_proj.weight",
            languagePrefix + "layers.*.mlp.gateup_proj.weight",
            languagePrefix + "layers.*.mlp.down_proj.weight",
            languagePrefix + "layers.*.mlp.shared_experts.gate_proj.weight",
            languagePrefix + "layers.*.mlp.shared_experts.up_proj.weight",
            languagePrefix + "layers.*.mlp.shared_experts.gateup_proj.weight",
            languagePrefix + "layers.*.mlp.shared_experts.down_proj.weight",
            languagePrefix + "layers.*.mlp.experts.*.gate_proj.weight",
            languagePrefix + "layers.*.mlp.experts.*.up_proj.weight",
            languagePrefix + "layers.*.mlp.experts.*.gateup_proj.weight",
            languagePrefix + "layers.*.mlp.experts.*.down_proj.weight",
        };
    }

    Glm5NextModel::~Glm5NextModel() {
        ShutdownRuntime();
    }

    void Glm5NextModel::SetDataType(DataType dataType) {
        if (dataType != DataType::BFLOAT16) {
            basellm::SetDataType(dataType);
            return;
        }
        this->dataType = dataType;
        if (!this->useCustomKVCacheDataType) {
            this->kvCacheDataType = dataType;
        }
    }

    void Glm5NextModel::InitParams() {
        FlattenGlm5NextTextConfig(weight.dicts);
        basellm::InitParams();

        auto requiredInt = [&](const std::string &key) {
            auto it = weight.dicts.find(key);
            AssertInFastLLM(
                it != weight.dicts.end(),
                "GLM-5.3 config is missing " + key + ".");
            return std::atoi(it->second.c_str());
        };
        auto requiredFloat = [&](const std::string &key) {
            auto it = weight.dicts.find(key);
            AssertInFastLLM(
                it != weight.dicts.end(),
                "GLM-5.3 config is missing " + key + ".");
            return (float)std::atof(it->second.c_str());
        };

        kdaHeads = requiredInt("linear_attn_config.num_heads");
        kdaHeadDim = requiredInt("linear_attn_config.head_dim");
        shortConvKernel =
            requiredInt("linear_attn_config.short_conv_kernel_size");
        gateLowerBound =
            requiredFloat("linear_attn_config.gate_lower_bound");

        qLoraRank = requiredInt("q_lora_rank");
        kvLoraRank = requiredInt("kv_lora_rank");
        qkNopeHeadDim = requiredInt("qk_nope_head_dim");
        qkRopeHeadDim = requiredInt("qk_rope_head_dim");
        qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        valueHeadDim = requiredInt("v_head_dim");

        denseIntermediateSize = requiredInt("intermediate_size");
        moeIntermediateSize = requiredInt("moe_intermediate_size");
        firstDenseLayers = requiredInt("first_k_dense_replace");
        swigluLimit = requiredFloat("swiglu_limit");

        hcMult = requiredInt("hc_mult");
        hcSinkhornIters = requiredInt("hc_sinkhorn_iters");
        hcEps = requiredFloat("hc_eps");
        rms_norm_eps = requiredFloat("rms_norm_eps");

        indexTopK = requiredInt("index_topk");
        exactSparseAttentionLimit = indexTopK;
        max_positions = std::min(
            requiredInt("max_position_embeddings"),
            exactSparseAttentionLimit);

        num_experts = requiredInt("n_routed_experts");
        num_experts_per_tok = requiredInt("num_experts_per_tok");
        n_shared_experts = requiredInt("n_shared_experts");
        routed_scaling_factor = requiredFloat("routed_scaling_factor");
        norm_topk_prob = weight.dicts["norm_topk_prob"] == "true";

        kdaLayers.assign(block_cnt, false);
        auto kdaConfig = weight.dicts.find(
            "linear_attn_config.kda_layers");
        AssertInFastLLM(
            kdaConfig != weight.dicts.end(),
            "GLM-5.3 config is missing linear_attn_config.kda_layers.");
        for (int layer : ParseGlm5NextIntegerList(kdaConfig->second)) {
            AssertInFastLLM(
                layer >= 0 && layer < block_cnt,
                "GLM-5.3 KDA layer index is out of range.");
            kdaLayers[layer] = true;
        }
        denseMlpLayers.assign(block_cnt, false);
        for (int layer = 0; layer < block_cnt; layer++) {
            denseMlpLayers[layer] = layer < firstDenseLayers;
        }

        int kdaCount = (int)std::count(
            kdaLayers.begin(), kdaLayers.end(), true);
        AssertInFastLLM(
            embed_dim == 4096 && block_cnt == 45 &&
            num_attention_heads == 64 && kdaHeads == 64 &&
            kdaHeadDim == 128 && shortConvKernel == 4 &&
            qLoraRank == 1536 && kvLoraRank == 512 &&
            qkNopeHeadDim == 256 && qkRopeHeadDim == 0 &&
            valueHeadDim == 256 && denseIntermediateSize == 12288 &&
            moeIntermediateSize == 2048 && firstDenseLayers == 3 &&
            num_experts == 288 && num_experts_per_tok == 8 &&
            n_shared_experts == 1 && hcMult == 4 &&
            hcSinkhornIters == 20 && indexTopK == 2048 &&
            kdaCount == 34,
            "The GLM-5.3 implementation currently supports the published "
            "GLM-5.3-Flash text configuration only.");

        kvCacheId = 3;
        elementsInKVCachePerToken = 0;
        for (int layer = 0; layer < block_cnt; layer++) {
            if (!kdaLayers[layer]) {
                elementsInKVCachePerToken +=
                    (long long)num_attention_heads *
                    (qkHeadDim + valueHeadDim);
            }
        }

        for (int layer = 0; layer < block_cnt; layer++) {
            const std::string prefix = languagePrefix + "layers." +
                std::to_string(layer) + ".mlp.";
            if (denseMlpLayers[layer]) {
                const std::string gate = prefix + "gate_proj.weight";
                const std::string up = prefix + "up_proj.weight";
                const std::string gateUp = prefix +
                    "gateup_proj.weight";
                weightMergeRules.push_back(WeightMergeRule({
                    WeightMergeRuleSingle(
                        {gate, up}, gateUp, "linearSwiglu"),
                }));
                continue;
            }

            for (int expert = -1; expert < num_experts; expert++) {
                std::string expertPrefix;
                if (expert < 0) {
                    expertPrefix = prefix + "shared_experts.";
                } else {
                    expertPrefix = prefix + "experts." +
                        std::to_string(expert) + ".";
                }
                const std::string gate =
                    expertPrefix + "gate_proj.weight";
                const std::string up =
                    expertPrefix + "up_proj.weight";
                const std::string gateUp =
                    expertPrefix + "gateup_proj.weight";
                const std::string down =
                    expertPrefix + "down_proj.weight";
                weightMergeRules.push_back(WeightMergeRule({
                    WeightMergeRuleSingle(
                        {gate, up}, gateUp, "linearSwiglu"),
                }));
                if (expert >= 0 || !GetCudaSharedExpert()) {
                    AddSpecialWeight(gateUp, "linearSwiglu", layer);
                    AddSpecialWeight(down, "linearColumn", layer);
                }
                moeLinears.insert(gate);
                moeLinears.insert(up);
                moeLinears.insert(down);
            }
        }

        std::cout
            << "[GLM-5.3] Hybrid text model: 34 KDA + 11 DSA layers, "
            << "42 MoE layers, BF16 activations.\n"
            << "[GLM-5.3] DSA currently uses the exact all-visible path "
            << "through " << exactSparseAttentionLimit
            << " cached tokens.\n";
    }

    std::map<std::string,
             std::vector<std::pair<std::string, DataType>>>
    Glm5NextModel::GetTensorMap(
            const std::vector<std::string> &tensorNames) {
        std::map<std::string,
                 std::vector<std::pair<std::string, DataType>>> result;
        const std::string layersPrefix = languagePrefix + "layers.";

        for (const std::string &name : tensorNames) {
            if (name == languagePrefix + "embed_tokens.weight") {
                result[name].emplace_back(name, DataType::BFLOAT16);
                continue;
            }
            if (name == languagePrefix + "norm.weight") {
                result[name].emplace_back(name, DataType::FLOAT32);
                continue;
            }
            if (name == "lm_head.weight") {
                result[name].emplace_back(name, DataType::BFLOAT16);
                continue;
            }
            if (name.rfind(layersPrefix, 0) != 0) {
                continue;
            }

            size_t position = layersPrefix.size();
            int layer = 0;
            bool hasLayer = false;
            while (position < name.size() &&
                   std::isdigit((unsigned char)name[position])) {
                hasLayer = true;
                layer = layer * 10 + name[position] - '0';
                position++;
            }
            if (!hasLayer || layer < 0 || layer >= block_cnt ||
                position >= name.size() || name[position] != '.') {
                continue;
            }
            const std::string suffix = name.substr(position + 1);

            if (suffix.rfind("self_attn.indexer.", 0) == 0 ||
                Glm5NextEndsWith(suffix, ".weight_scale_inv")) {
                continue;
            }
            if (suffix == "hc_attn_fn" || suffix == "hc_ffn_fn") {
                result[name].emplace_back(name, DataType::BFLOAT16);
                continue;
            }
            if (suffix == "hc_attn_base" ||
                suffix == "hc_attn_scale" ||
                suffix == "hc_ffn_base" ||
                suffix == "hc_ffn_scale" ||
                suffix == "mlp.gate.e_score_correction_bias") {
                result[name].emplace_back(name, DataType::FLOAT32);
                continue;
            }
            if (suffix == "input_layernorm.weight" ||
                suffix == "post_attention_layernorm.weight" ||
                suffix == "self_attn.q_a_layernorm.weight" ||
                suffix == "self_attn.kv_a_layernorm.weight" ||
                suffix == "self_attn.o_norm.weight" ||
                suffix == "self_attn.A_log" ||
                suffix == "self_attn.dt_bias" ||
                suffix == "self_attn.q_conv1d.weight" ||
                suffix == "self_attn.k_conv1d.weight" ||
                suffix == "self_attn.v_conv1d.weight" ||
                suffix == "mlp.gate.weight") {
                result[name].emplace_back(name, DataType::FLOAT32);
                continue;
            }

            if (suffix == "mlp.gate_proj.weight" ||
                suffix == "mlp.up_proj.weight" ||
                suffix == "mlp.down_proj.weight" ||
                (suffix.rfind("mlp.shared_experts.", 0) == 0 &&
                 Glm5NextEndsWith(suffix, ".weight")) ||
                (suffix.rfind("mlp.experts.", 0) == 0 &&
                 Glm5NextEndsWith(suffix, ".weight"))) {
                result[name].emplace_back(
                    name, DataType::DATA_AUTO_LINEAR);
                continue;
            }

            if (kdaLayers[layer]) {
                static const std::set<std::string> kdaBfloat16 = {
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.f_a_proj.weight",
                    "self_attn.f_b_proj.weight",
                    "self_attn.b_proj.weight",
                    "self_attn.g_a_proj.weight",
                    "self_attn.g_b_proj.weight",
                    "self_attn.o_proj.weight",
                };
                if (kdaBfloat16.find(suffix) != kdaBfloat16.end()) {
                    result[name].emplace_back(name, DataType::BFLOAT16);
                }
            } else if (suffix == "self_attn.kv_b_proj.weight") {
                result[name].emplace_back(name, DataType::BFLOAT16);
            } else if (suffix == "self_attn.q_a_proj.weight" ||
                       suffix == "self_attn.q_b_proj.weight" ||
                       suffix == "self_attn.kv_a_proj_with_mqa.weight" ||
                       suffix == "self_attn.o_proj.weight") {
                result[name].emplace_back(
                    name, DataType::DATA_AUTO_LINEAR);
            }
        }
        return result;
    }

    void Glm5NextModel::OnModelWeightsLoaded() {
        auto require = [&](const std::string &name) -> Data& {
            auto it = weight.weight.find(name);
            AssertInFastLLM(
                it != weight.weight.end() && !it->second.dims.empty(),
                "GLM-5.3 checkpoint is missing " + name + ".");
            return it->second;
        };

        require(languagePrefix + "embed_tokens.weight");
        require(languagePrefix + "norm.weight");
        require("lm_head.weight");

        expertWeights.assign(block_cnt, {});
        expertBiases.assign(block_cnt, {});
        for (int layer = 0; layer < block_cnt; layer++) {
            const std::string prefix = languagePrefix + "layers." +
                std::to_string(layer) + ".";
            for (const std::string &suffix : {
                    "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
                    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight"}) {
                require(prefix + suffix);
            }

            if (kdaLayers[layer]) {
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
                        "self_attn.A_log", "self_attn.dt_bias",
                        "self_attn.g_a_proj.weight",
                        "self_attn.g_b_proj.weight",
                        "self_attn.o_norm.weight",
                        "self_attn.o_proj.weight"}) {
                    require(prefix + suffix);
                }
            } else {
                for (const std::string &suffix : {
                        "self_attn.q_a_proj.weight",
                        "self_attn.q_a_layernorm.weight",
                        "self_attn.q_b_proj.weight",
                        "self_attn.kv_a_proj_with_mqa.weight",
                        "self_attn.kv_a_layernorm.weight",
                        "self_attn.kv_b_proj.weight",
                        "self_attn.o_proj.weight"}) {
                    require(prefix + suffix);
                }
            }

            const std::string mlp = prefix + "mlp.";
            if (denseMlpLayers[layer]) {
                require(mlp + "gateup_proj.weight");
                require(mlp + "down_proj.weight");
                continue;
            }

            require(mlp + "gate.weight");
            require(mlp + "gate.e_score_correction_bias");
            require(mlp + "shared_experts.gateup_proj.weight");
            require(mlp + "shared_experts.down_proj.weight");

            expertWeights[layer].reserve(2 * (num_experts + 1));
            expertBiases[layer].reserve(2 * (num_experts + 1));
            expertWeights[layer].push_back(nullptr);
            expertWeights[layer].push_back(nullptr);
            expertBiases[layer].push_back(nullptr);
            expertBiases[layer].push_back(nullptr);
            for (int expert = 0; expert < num_experts; expert++) {
                const std::string expertPrefix = mlp + "experts." +
                    std::to_string(expert) + ".";
                expertWeights[layer].push_back(
                    &require(expertPrefix + "gateup_proj.weight"));
                expertWeights[layer].push_back(
                    &require(expertPrefix + "down_proj.weight"));
                expertBiases[layer].push_back(nullptr);
                expertBiases[layer].push_back(nullptr);
            }
        }
    }

    void Glm5NextModel::RunKdaAttention(
            int layerIndex, Data &input, int sequence,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            Data &output) {
        const std::string prefix = languagePrefix + "layers." +
            std::to_string(layerIndex) + ".self_attn.";
        Data qCache, kCache, vCache;
        UnpackGlm5NextConvCache(
            pastKeyValues[layerIndex].first,
            shortConvKernel - 1, kdaHeads * kdaHeadDim,
            qCache, kCache, vCache);
        Data &state = pastKeyValues[layerIndex].second;

        auto runChunk = [&](Data &chunkInput, int chunkSequence,
                            Data &chunkOutput) {
            Data qProjected, kProjected, vProjected;
            Linear(chunkInput, weight[prefix + "q_proj.weight"],
                   Data(), qProjected);
            Linear(chunkInput, weight[prefix + "k_proj.weight"],
                   Data(), kProjected);
            Linear(chunkInput, weight[prefix + "v_proj.weight"],
                   Data(), vProjected);

            Data q, k, v;
            KimiK3CausalConv1D(
                qProjected, weight[prefix + "q_conv1d.weight"],
                shortConvKernel, qCache, q);
            KimiK3CausalConv1D(
                kProjected, weight[prefix + "k_conv1d.weight"],
                shortConvKernel, kCache, k);
            KimiK3CausalConv1D(
                vProjected, weight[prefix + "v_conv1d.weight"],
                shortConvKernel, vCache, v);
            q.Reshape({1, chunkSequence, kdaHeads, kdaHeadDim});
            k.Reshape({1, chunkSequence, kdaHeads, kdaHeadDim});
            v.Reshape({1, chunkSequence, kdaHeads, kdaHeadDim});

            Data gateLowRank, rawGate;
            Linear(chunkInput, weight[prefix + "f_a_proj.weight"],
                   Data(), gateLowRank);
            Linear(gateLowRank, weight[prefix + "f_b_proj.weight"],
                   Data(), rawGate);
            rawGate.Reshape(
                {1, chunkSequence, kdaHeads, kdaHeadDim});

            Data rawBetaBfloat16, rawBeta;
            Linear(chunkInput, weight[prefix + "b_proj.weight"],
                   Data(), rawBetaBfloat16);
            ToDataType(rawBetaBfloat16, rawBeta, DataType::FLOAT32);

            Data attention;
            KimiK3RecurrentKDAOutputOnly(
                q, k, v, rawGate, rawBeta,
                weight[prefix + "A_log"],
                weight[prefix + "dt_bias"], gateLowerBound,
                state, attention, true, true);

            Data gateLow, gate;
            Linear(chunkInput, weight[prefix + "g_a_proj.weight"],
                   Data(), gateLow);
            Linear(gateLow, weight[prefix + "g_b_proj.weight"],
                   Data(), gate);
            gate.Reshape(
                {1, chunkSequence, kdaHeads, kdaHeadDim});
            Data gatedAttention;
            KimiK3RMSNormSigmoidGate(
                attention, gate, weight[prefix + "o_norm.weight"],
                rms_norm_eps, gatedAttention);
            gatedAttention.Reshape(
                {1, chunkSequence, kdaHeads * kdaHeadDim});
            Linear(gatedAttention, weight[prefix + "o_proj.weight"],
                   Data(), chunkOutput);
        };

        const int chunkSize = 1024;
        if (sequence <= chunkSize) {
            runChunk(input, sequence, output);
        } else {
            for (int start = 0; start < sequence; start += chunkSize) {
                const int end = std::min(sequence, start + chunkSize);
                Data chunkInput, chunkOutput;
                Split(input, 1, start, end, chunkInput);
                runChunk(chunkInput, end - start, chunkOutput);
                if (start == 0) {
                    Copy(chunkOutput, output);
                    output.Expansion(input.dims);
                } else {
                    CatDirect(output, chunkOutput, 1);
                }
            }
            output.expansionDims.clear();
        }
        PackGlm5NextConvCache(
            qCache, kCache, vCache,
            pastKeyValues[layerIndex].first);
        state.SetKVCache();
        // The KDA recurrent matrix is also a fixed-size state.  Keeping both
        // members marked makes schedulers and cache accounting consistently
        // skip token-length handling for KDA layers.
        state.isLinearAttention = true;
    }

    void Glm5NextModel::RunSparseAttention(
            int layerIndex, Data &input, int sequence,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            Data &output) {
        const std::string prefix = languagePrefix + "layers." +
            std::to_string(layerIndex) + ".self_attn.";

        Data qResidual, qNormalized, query;
        Linear(input, weight[prefix + "q_a_proj.weight"],
               Data(), qResidual);
        KimiK3RMSNorm(
            qResidual, weight[prefix + "q_a_layernorm.weight"],
            rms_norm_eps, qNormalized);
        Linear(qNormalized, weight[prefix + "q_b_proj.weight"],
               Data(), query);
        query.Reshape(
            {1, sequence, num_attention_heads, qkHeadDim});
        PermuteSelf(query, {0, 2, 1, 3});

        Data compressedKv, kvNormalized, expandedKv;
        Linear(input, weight[prefix + "kv_a_proj_with_mqa.weight"],
               Data(), compressedKv);
        KimiK3RMSNorm(
            compressedKv, weight[prefix + "kv_a_layernorm.weight"],
            rms_norm_eps, kvNormalized);
        Linear(kvNormalized, weight[prefix + "kv_b_proj.weight"],
               Data(), expandedKv);
        expandedKv.Reshape(
            {1, sequence, num_attention_heads,
             qkNopeHeadDim + valueHeadDim});
        PermuteSelf(expandedKv, {0, 2, 1, 3});

        Data key, value;
        Split(expandedKv, -1, 0, qkNopeHeadDim, key);
        Split(expandedKv, -1, qkNopeHeadDim,
              qkNopeHeadDim + valueHeadDim, value);
        key.Reshape(
            {num_attention_heads, sequence, qkHeadDim});
        value.Reshape(
            {num_attention_heads, sequence, valueHeadDim});

        Data &keyCache = pastKeyValues[layerIndex].first;
        Data &valueCache = pastKeyValues[layerIndex].second;
        AppendGlm5NextSequenceCache(keyCache, key);
        AppendGlm5NextSequenceCache(valueCache, value);
        AssertInFastLLM(
            keyCache.dims[1] == valueCache.dims[1] &&
            keyCache.dims[1] <= exactSparseAttentionLimit,
            "GLM-5.3 DSA exact all-visible path supports at most " +
            std::to_string(exactSparseAttentionLimit) +
            " cached tokens.");

        Data attentionHeads;
        KimiK3CausalAttention(
            query, keyCache, valueCache,
            1.0f / std::sqrt((float)qkHeadDim), attentionHeads);
        PermuteSelf(attentionHeads, {0, 2, 1, 3});
        attentionHeads.Reshape(
            {1, sequence, num_attention_heads * valueHeadDim});
        Linear(attentionHeads, weight[prefix + "o_proj.weight"],
               Data(), output);
    }

    void Glm5NextModel::RunClampedMlp(
            Data &input, Data &gateUpWeight, Data &downWeight,
            Data &output) {
        Data gateUp, gate, up;
        Linear(input, gateUpWeight, Data(), gateUp);
        AssertInFastLLM(
            !gateUp.dims.empty() && gateUp.dims.back() % 2 == 0,
            "GLM-5.3 merged gate/up output has an invalid shape.");
        const int intermediate = gateUp.dims.back() / 2;
        Split(gateUp, -1, 0, intermediate, gate);
        Split(gateUp, -1, intermediate, 2 * intermediate, up);
        Glm5NextClamp(gate, false, 0.0f, true, swigluLimit);
        Glm5NextClamp(up, true, -swigluLimit, true, swigluLimit);
        Silu(gate, gate);
        MulTo(gate, up);
        Linear(gate, downWeight, Data(), output);
    }

    void Glm5NextModel::RunMoe(
            int layerIndex, Data &input, int sequence, Data &output) {
        const std::string mlp = languagePrefix + "layers." +
            std::to_string(layerIndex) + ".mlp.";
        const std::vector<int> outputDims = input.dims;
        Data flatInput;
        Copy(input, flatInput);
        flatInput.Reshape({sequence, embed_dim});

        ApplyDeviceMap(deviceMap, layerIndex + 1, block_cnt);
        Data routerInput, routerScores;
        ToDataType(flatInput, routerInput, DataType::FLOAT32);
        Linear(routerInput, weight[mlp + "gate.weight"],
               Data(), routerScores);
        Sigmoid(routerScores, routerScores);
        Data expertIndex, expertScore;
        SelectExpert(
            routerScores, expertIndex, expertScore,
            num_experts_per_tok, norm_topk_prob,
            routed_scaling_factor,
            &weight[mlp + "gate.e_score_correction_bias"]);

        Data sharedOutput;
        if (GetCudaSharedExpert()) {
            ApplyDeviceMap(deviceMap, layerIndex + 1, block_cnt);
        } else {
            ApplyMoeDeviceMapForLayer(layerIndex);
        }
        RunClampedMlp(
            flatInput,
            weight[mlp + "shared_experts.gateup_proj.weight"],
            weight[mlp + "shared_experts.down_proj.weight"],
            sharedOutput);

        Data w1, w2, w3, tempInput, tempOutput;
        Data moeInputTemp, moeOutputTemp, routedOutput;
        ApplyMoeDeviceMapForLayer(layerIndex);
        MergeMOEBlock(
            &flatInput, &expertIndex, &expertScore,
            &expertWeights[layerIndex], &expertBiases[layerIndex],
            &w1, &w2, &w3, &tempInput, &tempOutput,
            0.0f, &routedOutput, layerIndex,
            flatInput.dataType, moeAtype,
            &moeInputTemp, &moeOutputTemp,
            MoeGateSwiglu, false, swigluLimit, true);

        ApplyDeviceMap(deviceMap, layerIndex + 1, block_cnt);
        Copy(routedOutput, output);
        AddTo(output, sharedOutput);
        output.Reshape(outputDims);
    }

    int Glm5NextModel::Sample(
            Data &hiddenStates,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *logits) {
        AssertInFastLLM(
            hiddenStates.dims.size() == 3 &&
            hiddenStates.dims[0] == 1,
            "GLM-5.3 final hidden state has an invalid shape.");
        Data lastHidden;
        const int sequence = hiddenStates.dims[1];
        if (sequence > 1) {
            Split(hiddenStates, 1, sequence - 1, sequence, lastHidden);
        } else {
            Copy(hiddenStates, lastHidden);
        }
        Data outputLogits;
        Linear(lastHidden, weight["lm_head.weight"],
               Data(), outputLogits);
        ToDataType(outputLogits, DataType::FLOAT32);
        outputLogits.ToDevice(DataDevice::CPU);
        if (generationConfig.output_logits && logits != nullptr) {
            const int vocabulary = outputLogits.dims.back();
            logits->resize(vocabulary);
            std::memcpy(
                logits->data(), outputLogits.cpuData,
                (size_t)vocabulary * sizeof(float));
        }
        if (generationConfig.IsSimpleGreedy()) {
            Data topk;
            TopK(outputLogits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            return (int)(((float*)topk.cpuData)[0] + 1e-3f);
        }
        if (!lastTokens.units.empty()) {
            return LLMSampling(
                outputLogits, 0, generationConfig,
                lastTokens.units[0]);
        }
        return LLMSamplingOnly(outputLogits, 0, generationConfig);
    }

    int Glm5NextModel::Forward(
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *logits) {
        (void)attentionMask;
        (void)positionIds;
        AssertInFastLLM(
            inputIds.dims.size() == 2 && inputIds.dims[0] == 1 &&
            inputIds.dims[1] > 0,
            "GLM-5.3 Forward currently supports one non-empty request.");
        if ((int)pastKeyValues.size() < block_cnt) {
            pastKeyValues.resize(block_cnt);
        }
        const int sequence = inputIds.dims[1];

        ApplyDeviceMap(deviceMap, 0, block_cnt);
        Data embedding, hiddenStates, hiddenStatesTemp;
        Embedding(
            inputIds, weight[languagePrefix + "embed_tokens.weight"],
            embedding);
        ToDataType(embedding, DataType::BFLOAT16);
        embedding.Reshape({1, sequence, 1, embed_dim});
        Repeat(embedding, 2, hcMult, hiddenStates);

        Data *current = &hiddenStates;
        Data *next = &hiddenStatesTemp;
        for (int layer = 0; layer < block_cnt; layer++) {
            ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
            const std::string prefix = languagePrefix + "layers." +
                std::to_string(layer) + ".";

            Data attentionInput, attentionPost, attentionComb;
            DeepSeekV4HcPre(
                *current, weight[prefix + "hc_attn_fn"],
                weight[prefix + "hc_attn_scale"],
                weight[prefix + "hc_attn_base"], hcMult,
                hcSinkhornIters, hcEps, rms_norm_eps,
                attentionInput, attentionPost, attentionComb);
            Data normalizedAttention;
            KimiK3RMSNorm(
                attentionInput,
                weight[prefix + "input_layernorm.weight"],
                rms_norm_eps, normalizedAttention);
            Data attentionOutput;
            if (kdaLayers[layer]) {
                RunKdaAttention(
                    layer, normalizedAttention, sequence,
                    pastKeyValues, attentionOutput);
            } else {
                RunSparseAttention(
                    layer, normalizedAttention, sequence,
                    pastKeyValues, attentionOutput);
            }
            DeepSeekV4HcPost(
                attentionOutput, *current,
                attentionPost, attentionComb, *next);
            std::swap(current, next);

            Data ffnInput, ffnPost, ffnComb;
            DeepSeekV4HcPre(
                *current, weight[prefix + "hc_ffn_fn"],
                weight[prefix + "hc_ffn_scale"],
                weight[prefix + "hc_ffn_base"], hcMult,
                hcSinkhornIters, hcEps, rms_norm_eps,
                ffnInput, ffnPost, ffnComb);
            Data normalizedFfn;
            KimiK3RMSNorm(
                ffnInput,
                weight[prefix + "post_attention_layernorm.weight"],
                rms_norm_eps, normalizedFfn);
            Data ffnOutput;
            if (denseMlpLayers[layer]) {
                RunClampedMlp(
                    normalizedFfn,
                    weight[prefix + "mlp.gateup_proj.weight"],
                    weight[prefix + "mlp.down_proj.weight"],
                    ffnOutput);
            } else {
                RunMoe(layer, normalizedFfn, sequence, ffnOutput);
            }
            DeepSeekV4HcPost(
                ffnOutput, *current, ffnPost, ffnComb, *next);
            std::swap(current, next);
        }

        ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
        Data collapsed, finalHidden;
        Glm5NextHcMean(*current, collapsed);
        KimiK3RMSNorm(
            collapsed, weight[languagePrefix + "norm.weight"],
            rms_norm_eps, finalHidden);
        return Sample(
            finalHidden, generationConfig, lastTokens, logits);
    }

    bool Glm5NextModel::NeedAttentionMask(int qlen, int klen) {
        (void)qlen;
        (void)klen;
        return false;
    }

    std::string Glm5NextModel::MakeInput(
            const std::string &history, int round,
            const std::string &input) {
        (void)round;
        return history + input;
    }

    std::string Glm5NextModel::MakeHistory(
            const std::string &history, int round,
            const std::string &input, const std::string &output) {
        (void)round;
        return history + input + output;
    }
}
