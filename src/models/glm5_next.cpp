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
#include <cstdlib>
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

        void BindGlm5NextConvCacheViews(
                Data &packed, const Data &reference,
                int history, int channels,
                Data &qCache, Data &kCache, Data &vCache) {
            if (packed.dims.empty()) {
                packed.dataType = DataType::BFLOAT16;
                packed.UpdateUnitSize();
                packed.dataDevice = reference.dataDevice;
                packed.dataDeviceIds = reference.dataDeviceIds;
                packed.Resize({3, history, channels});
                packed.Allocate();
            }
            AssertInFastLLM(
                packed.dataType == DataType::BFLOAT16 &&
                packed.dims == std::vector<int>({3, history, channels}),
                "GLM-5.3 packed KDA convolution cache has an invalid shape.");
            if (!packed.lockInCPU) {
                packed.ToDevice(
                    reference.dataDevice, reference.dataDeviceIds);
            }
            const size_t sliceBytes = packed.GetBytes() / 3;
            auto bind = [&](Data &view, int index) {
                view.FakeFrom(packed, (size_t)index * sliceBytes);
                view.dataDeviceIds = packed.dataDeviceIds;
                view.lockInCPU = packed.lockInCPU;
                view.Resize({1, history, channels});
            };
            bind(qCache, 0);
            bind(kCache, 1);
            bind(vCache, 2);
            packed.SetKVCache();
            // This is a fixed-size convolution history, not a token-growing
            // attention cache.  Generic history-cache restore must copy it as
            // a whole instead of slicing dimension 1 by the matched prefix.
            packed.isLinearAttention = true;
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

        bool Glm5NextHistoryCacheDebugEnabled() {
            static const bool enabled = []() {
                const char *value = std::getenv(
                    "FASTLLM_GLM5_NEXT_HISTORY_CACHE_DEBUG");
                return value != nullptr && value[0] != '\0' &&
                       std::strcmp(value, "0") != 0 &&
                       std::strcmp(value, "false") != 0 &&
                       std::strcmp(value, "off") != 0;
            }();
            return enabled;
        }

        int Glm5NextPagedCacheLength(const Data &cache) {
            if (!cache.isPagedKVCache ||
                cache.pagedKVCacheData == nullptr ||
                cache.pageLen <= 0 || cache.pageIndex.empty() ||
                cache.lastPageLen <= 0 ||
                cache.lastPageLen > cache.pageLen) {
                return -1;
            }
            return ((int)cache.pageIndex.size() - 1) * cache.pageLen +
                   cache.lastPageLen;
        }

        uint64_t Glm5NextPagedCacheBytes(const Data &cache) {
            if (Glm5NextPagedCacheLength(cache) <= 0) {
                return 0;
            }
            const PagedCacheManager *manager = cache.pagedKVCacheData;
            if (manager->maxPages <= 0 || manager->dims.size() != 4) {
                return 0;
            }
            return manager->GetBytes() / (uint64_t)manager->maxPages *
                   (uint64_t)cache.pageIndex.size();
        }

        std::string Glm5NextDimsText(const std::vector<int> &dims) {
            std::string text = "[";
            for (size_t i = 0; i < dims.size(); i++) {
                if (i > 0) {
                    text += ",";
                }
                text += std::to_string(dims[i]);
            }
            return text + "]";
        }

        void DebugGlm5NextCacheDescriptor(
                int layer, const char *slot, const Data &cache) {
            if (!Glm5NextHistoryCacheDebugEnabled()) {
                return;
            }
            const std::string dims = Glm5NextDimsText(cache.dims);
            const std::string managerDims =
                cache.pagedKVCacheData == nullptr ? "[]" :
                Glm5NextDimsText(cache.pagedKVCacheData->dims);
            fprintf(stderr,
                    "[glm5-next-history] descriptor layer=%d slot=%s "
                    "dtype=%d dims=%s kv=%d linear=%d transposed=%d "
                    "paged=%d pages=%zu page_len=%d last_page_len=%d "
                    "manager_dims=%s manager_max_pages=%d\n",
                    layer, slot, (int)cache.dataType, dims.c_str(),
                    cache.isKVCache ? 1 : 0,
                    cache.isLinearAttention ? 1 : 0,
                    cache.isLinearAttentionTransposed ? 1 : 0,
                    cache.isPagedKVCache ? 1 : 0,
                    cache.pageIndex.size(), cache.pageLen,
                    cache.lastPageLen, managerDims.c_str(),
                    cache.pagedKVCacheData == nullptr ? -1 :
                        cache.pagedKVCacheData->maxPages);
        }

        void ShareGlm5NextPagedCache(
                const Data &source, Data &target) {
            AssertInFastLLM(
                Glm5NextPagedCacheLength(source) > 0 &&
                source.pagedKVCacheData->dims.size() == 4 &&
                target.pageIndex.empty() &&
                target.pagedKVCacheData == nullptr &&
                target.cpuData == nullptr && target.cudaData == nullptr,
                "GLM-5.3 cannot share an invalid or non-empty paged cache.");
            target.name = source.name;
            target.cacheUid = source.cacheUid;
            target.isKVCache = true;
            target.isPagedKVCache = true;
            target.isLinearAttention = false;
            target.isLinearAttentionTransposed = false;
            target.dataType = source.dataType;
            target.UpdateUnitSize();
            target.dataDevice = source.dataDevice;
            target.dataDeviceIds = source.dataDeviceIds;
            target.Resize(source.dims);
            target.pageLen = source.pageLen;
            target.pagedKVCacheData = source.pagedKVCacheData;
            target.pageIndex = source.pageIndex;
            target.lastPageLen = source.lastPageLen;
            target.pagedKVCacheData->Pick(target.pageIndex);
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
        {
            std::lock_guard<std::mutex> guard(historyCacheMutex);
            pendingHistoryCache.reset();
            historyCache.clear();
            historyCacheBytes = 0;
        }
        {
            std::lock_guard<std::mutex> guard(responseContextsMutex);
            responseContexts.clear();
        }
        ClearAllPagedCacheManagers();
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
        max_positions = requiredInt("max_position_embeddings");
        AssertInFastLLM(
            max_positions >= indexTopK,
            "GLM-5.3 max_position_embeddings is smaller than index_topk.");
        if (const char *gpuCacheMb = std::getenv(
                "FASTLLM_GLM5_NEXT_HISTORY_CACHE_GPU_MB")) {
            char *end = nullptr;
            unsigned long long value = std::strtoull(
                gpuCacheMb, &end, 10);
            AssertInFastLLM(
                end != gpuCacheMb && *end == '\0' && value <= 65536,
                "FASTLLM_GLM5_NEXT_HISTORY_CACHE_GPU_MB must be an "
                "integer in [0, 65536].");
            historyCacheGpuStateLimitBytes =
                (uint64_t)value * 1024ULL * 1024ULL;
        }
        if (const char *cacheBudgetGb = std::getenv(
                "FASTLLM_GLM5_NEXT_HISTORY_CACHE_MAX_GB")) {
            char *end = nullptr;
            unsigned long long value = std::strtoull(
                cacheBudgetGb, &end, 10);
            AssertInFastLLM(
                end != cacheBudgetGb && *end == '\0' &&
                value >= 1 && value <= 1024,
                "FASTLLM_GLM5_NEXT_HISTORY_CACHE_MAX_GB must be an "
                "integer in [1, 1024].");
            historyCacheMaxBytes =
                (uint64_t)value * 1024ULL * 1024ULL * 1024ULL;
        }

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
            << "[GLM-5.3] Context window restored to " << max_positions
            << " tokens. DSA above Top-" << indexTopK
            << " currently uses exact paged attention with "
            << fastllm::GetPageLen() << "-token cache pages.\n";
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

    int Glm5NextModel::GetHistoryCacheSequenceLength(
            const std::vector<std::pair<Data, Data>>
                &pastKeyValues) const {
        if ((int)pastKeyValues.size() < block_cnt ||
            (int)kdaLayers.size() < block_cnt) {
            return -1;
        }
        int sequenceLength = -1;
        int sparseLayers = 0;
        int populatedSparseLayers = 0;
        for (int layer = 0; layer < block_cnt; layer++) {
            if (kdaLayers[layer]) {
                continue;
            }
            sparseLayers++;
            const Data &key = pastKeyValues[layer].first;
            const Data &value = pastKeyValues[layer].second;
            if (key.dims.empty() && value.dims.empty() &&
                !key.isPagedKVCache && !value.isPagedKVCache) {
                continue;
            }
            populatedSparseLayers++;
            const int keyLength = Glm5NextPagedCacheLength(key);
            const int valueLength = Glm5NextPagedCacheLength(value);
            if (key.dims.size() != 3 || value.dims.size() != 3 ||
                keyLength <= 0 || keyLength != valueLength ||
                key.dims[1] != keyLength ||
                value.dims[1] != valueLength) {
                return -1;
            }
            if (sequenceLength < 0) {
                sequenceLength = keyLength;
            } else if (sequenceLength != keyLength) {
                return -1;
            }
        }
        if (populatedSparseLayers > 0 &&
            populatedSparseLayers != sparseLayers) {
            return -1;
        }
        return std::max(0, sequenceLength);
    }

    bool Glm5NextModel::CanRestoreHistoryCache(
            const HistoryCacheMemory &memory) const {
        if (memory.sequenceLength <= 0 ||
            memory.sequenceLength != (int)memory.tokens.size() ||
            (int)memory.pastKeyValues.size() < block_cnt ||
            (int)kdaLayers.size() < block_cnt) {
            return false;
        }
        for (int layer = 0; layer < block_cnt; layer++) {
            const Data &first = memory.pastKeyValues[layer].first;
            const Data &second = memory.pastKeyValues[layer].second;
            if (first.multiDeviceData || second.multiDeviceData ||
                !first.multiDeviceDatas.empty() ||
                !second.multiDeviceDatas.empty()) {
                return false;
            }
            if (kdaLayers[layer]) {
                if (first.dataType != DataType::BFLOAT16 ||
                    first.dims != std::vector<int>({
                        3, shortConvKernel - 1,
                        kdaHeads * kdaHeadDim}) ||
                    second.dataType != DataType::FLOAT32 ||
                    second.dims != std::vector<int>({
                        1, kdaHeads, kdaHeadDim, kdaHeadDim}) ||
                    !first.isLinearAttention ||
                    !second.isLinearAttention ||
                    first.isPagedKVCache || second.isPagedKVCache) {
                    return false;
                }
                continue;
            }
            auto validPaged = [&](const Data &cache, int headDim) {
                if (cache.dataType != DataType::BFLOAT16 ||
                    cache.dims != std::vector<int>({
                        num_attention_heads, memory.sequenceLength,
                        headDim}) ||
                    cache.isLinearAttention ||
                    Glm5NextPagedCacheLength(cache) !=
                        memory.sequenceLength ||
                    cache.lastPageLen != cache.pageLen ||
                    cache.pagedKVCacheData == nullptr ||
                    cache.pagedKVCacheData->dataType !=
                        DataType::BFLOAT16 ||
                    cache.pagedKVCacheData->dims.size() != 4 ||
                    cache.pagedKVCacheData->dims[1] != cache.pageLen ||
                    cache.pagedKVCacheData->dims[2] !=
                        num_attention_heads ||
                    cache.pagedKVCacheData->dims[3] != headDim) {
                    return false;
                }
                for (int page : cache.pageIndex) {
                    if (page < 0 ||
                        page >= cache.pagedKVCacheData->maxPages) {
                        return false;
                    }
                }
                return true;
            };
            if (!validPaged(first, qkHeadDim) ||
                !validPaged(second, valueHeadDim) ||
                first.pageLen != second.pageLen ||
                first.pageIndex.size() != second.pageIndex.size()) {
                return false;
            }
        }
        return true;
    }

    void Glm5NextModel::OnResponseContextCreated(
            ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        std::shared_ptr<HistoryCacheMemory> pending;
        {
            std::lock_guard<std::mutex> guard(historyCacheMutex);
            pending.swap(pendingHistoryCache);
        }
        if (pending != nullptr) {
            std::lock_guard<std::mutex> forwardGuard(forwardLocker);
            RestoreHistoryCache(*pending, context);
            context->preTokens = context->cacheLen;
            context->intParams["add_special_tokens"] = 0;
            context->intParams["promptLen"] =
                context->cacheLen + (int)context->currentTokens.size();
            context->intParams["index"] = -1;
        }
        std::lock_guard<std::mutex> guard(responseContextsMutex);
        responseContexts[&context->pastKeyValues] = context;
    }

    void Glm5NextModel::OnResponseContextRemoved(
            ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> guard(responseContextsMutex);
        responseContexts.erase(&context->pastKeyValues);
    }

    bool Glm5NextModel::TryRestoreHistoryCache(
            std::vector<int> &inputTokens, int &cacheLen) {
        if (!saveHistoryChat || inputTokens.size() <= 1) {
            return false;
        }
        std::shared_ptr<HistoryCacheMemory> best;
        size_t bestLength = 0;
        size_t recordCount = 0;
        {
            std::lock_guard<std::mutex> guard(historyCacheMutex);
            pendingHistoryCache.reset();
            for (auto it = historyCache.begin();
                 it != historyCache.end();) {
                auto current = it++;
                if (!CanRestoreHistoryCache(*current->second)) {
                    historyCacheBytes -= std::min(
                        historyCacheBytes, current->second->bytes);
                    historyCache.erase(current);
                    continue;
                }
                const std::vector<int> &cachedTokens = current->first;
                if (cachedTokens.size() <= bestLength ||
                    cachedTokens.size() >= inputTokens.size() ||
                    !std::equal(cachedTokens.begin(), cachedTokens.end(),
                                inputTokens.begin())) {
                    continue;
                }
                best = current->second;
                bestLength = cachedTokens.size();
            }

            // Custom snapshots retain pages outside the generic paged trie.
            // Reclaim unrelated LRU snapshots before scheduler admission so
            // their references cannot leave a prefill waiting forever for a
            // fixed-size page pool. Keep a small decode reserve and never
            // evict the snapshot selected for this request.
            std::set<PagedCacheManager*> managers;
            for (const auto &item : historyCache) {
                const auto &values = item.second->pastKeyValues;
                for (int layer = 0;
                     layer < block_cnt && layer < (int)values.size();
                     layer++) {
                    if (kdaLayers[layer]) {
                        continue;
                    }
                    if (values[layer].first.pagedKVCacheData != nullptr) {
                        managers.insert(
                            values[layer].first.pagedKVCacheData);
                    }
                    if (values[layer].second.pagedKVCacheData != nullptr) {
                        managers.insert(
                            values[layer].second.pagedKVCacheData);
                    }
                }
            }
            const int remainingTokens =
                (int)inputTokens.size() - (int)bestLength;
            auto hasPageReserve = [&]() {
                for (PagedCacheManager *manager : managers) {
                    if (manager == nullptr || manager->pageLen <= 0 ||
                        manager->maxPages <= 0) {
                        continue;
                    }
                    const int requiredPages =
                        (remainingTokens + manager->pageLen - 1) /
                        manager->pageLen;
                    const int reservePages =
                        std::max(1, manager->maxPages / 8);
                    const int desiredFreePages = std::min(
                        manager->maxPages,
                        requiredPages + reservePages);
                    std::lock_guard<std::mutex> pageGuard(
                        manager->pageIndexLocker);
                    if (manager->FreePageCount() < desiredFreePages) {
                        return false;
                    }
                }
                return true;
            };
            while (!historyCache.empty() && !hasPageReserve()) {
                auto oldest = historyCache.end();
                for (auto it = historyCache.begin();
                     it != historyCache.end(); ++it) {
                    if (it->second == best) {
                        continue;
                    }
                    if (oldest == historyCache.end() ||
                        it->second->flushTime <
                            oldest->second->flushTime) {
                        oldest = it;
                    }
                }
                if (oldest == historyCache.end()) {
                    break;
                }
                historyCacheBytes -= std::min(
                    historyCacheBytes, oldest->second->bytes);
                historyCache.erase(oldest);
            }
            recordCount = historyCache.size();
            if (best != nullptr) {
                best->flushTime = ++historyCacheFlushTime;
                pendingHistoryCache = best;
            }
        }
        if (best == nullptr) {
            if (Glm5NextHistoryCacheDebugEnabled()) {
                fprintf(stderr,
                        "[glm5-next-history] miss input=%zu records=%zu\n",
                        inputTokens.size(), recordCount);
            }
            return false;
        }
        inputTokens.erase(
            inputTokens.begin(), inputTokens.begin() + bestLength);
        cacheLen = (int)bestLength;
        if (Glm5NextHistoryCacheDebugEnabled()) {
            fprintf(stderr,
                    "[glm5-next-history] hit input=%zu restore=%zu "
                    "state=%s dsa=shared-pages\n",
                    inputTokens.size() + bestLength, bestLength,
                    best->recurrentStateOnCpu ? "cpu" : "gpu");
        }
        return true;
    }

    void Glm5NextModel::RecordHistoryCache(
            const std::vector<int> &tokens,
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            int sequenceLength) {
        if (!saveHistoryChat || sequenceLength <= 0 ||
            sequenceLength != (int)tokens.size() ||
            GetHistoryCacheSequenceLength(pastKeyValues) !=
                sequenceLength) {
            return;
        }
        {
            std::lock_guard<std::mutex> guard(historyCacheMutex);
            auto existing = historyCache.find(tokens);
            if (existing != historyCache.end()) {
                existing->second->flushTime = ++historyCacheFlushTime;
                return;
            }
        }

        uint64_t estimatedBytes = 0;
        uint64_t recurrentBytes = 0;
        bool recurrentSourcesOnCuda = true;
        for (int layer = 0; layer < block_cnt; layer++) {
            for (const Data *source : {
                    &pastKeyValues[layer].first,
                    &pastKeyValues[layer].second}) {
                if (source->dims.empty() || source->multiDeviceData ||
                    !source->multiDeviceDatas.empty()) {
                    return;
                }
                if (kdaLayers[layer]) {
                    if (source->isPagedKVCache) {
                        return;
                    }
                    const uint64_t bytes =
                        source->expansionBytes > 0 ?
                        source->expansionBytes : source->GetBytes();
                    recurrentBytes += bytes;
                    estimatedBytes += bytes;
                    recurrentSourcesOnCuda &=
                        source->dataDevice == DataDevice::CUDA;
                } else {
                    if (Glm5NextPagedCacheLength(*source) !=
                            sequenceLength ||
                        source->lastPageLen != source->pageLen) {
                        return;
                    }
                    const uint64_t bytes =
                        Glm5NextPagedCacheBytes(*source);
                    if (bytes == 0) {
                        return;
                    }
                    estimatedBytes += bytes;
                }
            }
        }

        // DSA pages remain in their runtime pools and are retained by reference;
        // only the fixed-size KDA recurrent state needs a physical snapshot.
        bool storeOnCpu = GetHistoryCacheInCPU() ||
            !recurrentSourcesOnCuda ||
            recurrentBytes > historyCacheGpuStateLimitBytes;
#ifdef USE_CUDA
        if (!storeOnCpu) {
            const uint64_t reserveBytes =
                512ULL * 1024ULL * 1024ULL;
            const long long freeBytes = FastllmCudaGetFreeSize();
            storeOnCpu = freeBytes <= 0 ||
                recurrentBytes + reserveBytes > (uint64_t)freeBytes;
        }
#else
        storeOnCpu = true;
#endif

        auto evictOldestLocked = [&]() -> bool {
            auto oldest = historyCache.end();
            for (auto it = historyCache.begin();
                 it != historyCache.end(); ++it) {
                if (oldest == historyCache.end() ||
                    it->second->flushTime < oldest->second->flushTime) {
                    oldest = it;
                }
            }
            if (oldest == historyCache.end()) {
                return false;
            }
            historyCacheBytes -= std::min(
                historyCacheBytes, oldest->second->bytes);
            historyCache.erase(oldest);
            return true;
        };
        auto overBudget = [&](uint64_t incoming) {
            return incoming > historyCacheMaxBytes ||
                historyCacheBytes > historyCacheMaxBytes - incoming;
        };
        {
            std::lock_guard<std::mutex> guard(historyCacheMutex);
            while (!historyCache.empty() &&
                   ((int)historyCache.size() >= historyCacheMaxRecords ||
                    overBudget(estimatedBytes))) {
                if (!evictOldestLocked()) {
                    break;
                }
            }
        }

        auto memory = std::make_shared<HistoryCacheMemory>();
        memory->tokens = tokens;
        memory->sequenceLength = sequenceLength;
        memory->recurrentStateOnCpu = storeOnCpu;
        memory->pastKeyValues.resize(block_cnt);
        const bool lockCpuCache = GetHistoryCacheInCPU();
        auto copyTensor = [&](const Data &source, Data &target) {
            if (!storeOnCpu) {
#ifdef USE_CUDA
                const int originalDevice = FastllmCudaGetDevice();
                int sourceDevice = GetPointerDeviceId(source.cudaData);
                if (sourceDevice < 0 && !source.dataDeviceIds.empty()) {
                    sourceDevice = source.dataDeviceIds[0];
                }
                AssertInFastLLM(
                    sourceDevice >= 0,
                    "GLM-5.3 history cache source GPU is unknown.");
                FastllmCudaSetDevice(sourceDevice);
                target.CopyFrom(source);
                target.dataDeviceIds = source.dataDeviceIds;
                FastllmCudaSetDevice(originalDevice);
#else
                ErrorInFastLLM(
                    "GLM-5.3 CUDA history cache requires a CUDA build.");
#endif
                return;
            }

            target.name = source.name;
            target.isKVCache = source.isKVCache;
            target.isLinearAttention = source.isLinearAttention;
            target.isLinearAttentionTransposed =
                source.isLinearAttentionTransposed;
            target.cacheUid = source.cacheUid;
            target.dataType = source.dataType;
            target.UpdateUnitSize();
            target.dataDevice = DataDevice::CPU;
            target.dataDeviceIds = source.dataDeviceIds;
            target.Resize(source.dims);
            if (!source.expansionDims.empty() &&
                source.expansionDims != source.dims) {
                target.Expansion(source.expansionDims);
                target.Resize(source.dims);
            } else {
                target.Allocate(false);
            }
            const size_t bytes = source.expansionBytes > 0 ?
                source.expansionBytes : source.GetBytes();
            AssertInFastLLM(
                target.cpuData != nullptr &&
                bytes <= target.expansionBytes,
                "GLM-5.3 CPU history cache allocation failed.");
            if (source.dataDevice == DataDevice::CPU) {
                AssertInFastLLM(
                    source.cpuData != nullptr,
                    "GLM-5.3 history cache source has no CPU data.");
                std::memcpy(target.cpuData, source.cpuData, bytes);
            } else {
#ifdef USE_CUDA
                AssertInFastLLM(
                    source.cudaData != nullptr,
                    "GLM-5.3 history cache source has no CUDA data.");
                const int originalDevice = FastllmCudaGetDevice();
                int sourceDevice = GetPointerDeviceId(source.cudaData);
                if (sourceDevice < 0 && !source.dataDeviceIds.empty()) {
                    sourceDevice = source.dataDeviceIds[0];
                }
                AssertInFastLLM(
                    sourceDevice >= 0,
                    "GLM-5.3 history cache source GPU is unknown.");
                FastllmCudaSetDevice(sourceDevice);
                FastllmCudaCopyFromDeviceToHost(
                    target.cpuData, source.cudaData, bytes);
                FastllmCudaSetDevice(originalDevice);
#else
                ErrorInFastLLM(
                    "GLM-5.3 CUDA history cache requires a CUDA build.");
#endif
            }
            target.lockInCPU = lockCpuCache;
        };

        for (int layer = 0; layer < block_cnt; layer++) {
            if (kdaLayers[layer]) {
                copyTensor(pastKeyValues[layer].first,
                           memory->pastKeyValues[layer].first);
                copyTensor(pastKeyValues[layer].second,
                           memory->pastKeyValues[layer].second);
                memory->bytes +=
                    memory->pastKeyValues[layer].first.expansionBytes;
                memory->bytes +=
                    memory->pastKeyValues[layer].second.expansionBytes;
            } else {
                ShareGlm5NextPagedCache(
                    pastKeyValues[layer].first,
                    memory->pastKeyValues[layer].first);
                ShareGlm5NextPagedCache(
                    pastKeyValues[layer].second,
                    memory->pastKeyValues[layer].second);
                memory->bytes += Glm5NextPagedCacheBytes(
                    memory->pastKeyValues[layer].first);
                memory->bytes += Glm5NextPagedCacheBytes(
                    memory->pastKeyValues[layer].second);
            }
        }
        if (!CanRestoreHistoryCache(*memory)) {
            for (int layer = 0; layer < block_cnt; layer++) {
                DebugGlm5NextCacheDescriptor(
                    layer, "first",
                    memory->pastKeyValues[layer].first);
                DebugGlm5NextCacheDescriptor(
                    layer, "second",
                    memory->pastKeyValues[layer].second);
            }
            fprintf(stderr,
                    "[glm5-next-history] skipped incomplete snapshot\n");
            return;
        }

        std::lock_guard<std::mutex> guard(historyCacheMutex);
        auto existing = historyCache.find(tokens);
        if (existing != historyCache.end()) {
            existing->second->flushTime = ++historyCacheFlushTime;
            return;
        }
        while (!historyCache.empty() &&
               ((int)historyCache.size() >= historyCacheMaxRecords ||
                overBudget(memory->bytes))) {
            if (!evictOldestLocked()) {
                break;
            }
        }
        memory->flushTime = ++historyCacheFlushTime;
        historyCacheBytes += memory->bytes;
        historyCache[tokens] = std::move(memory);
        if (Glm5NextHistoryCacheDebugEnabled()) {
            const auto &stored = historyCache[tokens];
            fprintf(stderr,
                    "[glm5-next-history] record tokens=%zu bytes=%.2fGiB "
                    "state=%s dsa=shared-pages records=%zu total=%.2fGiB\n",
                    tokens.size(),
                    stored->bytes / (1024.0 * 1024.0 * 1024.0),
                    stored->recurrentStateOnCpu ? "cpu" : "gpu",
                    historyCache.size(),
                    historyCacheBytes /
                        (1024.0 * 1024.0 * 1024.0));
        }
    }

    void Glm5NextModel::RestoreHistoryCache(
            const HistoryCacheMemory &memory,
            ResponseContext *context) {
        AssertInFastLLM(
            context != nullptr &&
            context->cacheLen == memory.sequenceLength &&
            CanRestoreHistoryCache(memory),
            "GLM-5.3 history cache snapshot is incomplete.");
        context->pastKeyValues.resize(block_cnt);
        auto restoreTensor = [](const Data &source, Data &target) {
#ifdef USE_CUDA
            const int originalDevice = FastllmCudaGetDevice();
            if (source.dataDevice == DataDevice::CUDA &&
                source.cudaData != nullptr) {
                int sourceDevice = GetPointerDeviceId(source.cudaData);
                if (sourceDevice < 0 && !source.dataDeviceIds.empty()) {
                    sourceDevice = source.dataDeviceIds[0];
                }
                AssertInFastLLM(
                    sourceDevice >= 0,
                    "GLM-5.3 history cache source GPU is unknown.");
                FastllmCudaSetDevice(sourceDevice);
            }
#endif
            target.CopyFrom(source);
            target.lockInCPU = source.lockInCPU;
            target.dataDeviceIds = source.dataDeviceIds;
#ifdef USE_CUDA
            if (source.dataDevice == DataDevice::CPU &&
                !source.lockInCPU && !source.dataDeviceIds.empty()) {
                target.ToDevice(
                    DataDevice::CUDA, source.dataDeviceIds);
            }
            FastllmCudaSetDevice(originalDevice);
#endif
        };
        for (int layer = 0; layer < block_cnt; layer++) {
            if (kdaLayers[layer]) {
                restoreTensor(memory.pastKeyValues[layer].first,
                              context->pastKeyValues[layer].first);
                restoreTensor(memory.pastKeyValues[layer].second,
                              context->pastKeyValues[layer].second);
            } else {
                ShareGlm5NextPagedCache(
                    memory.pastKeyValues[layer].first,
                    context->pastKeyValues[layer].first);
                ShareGlm5NextPagedCache(
                    memory.pastKeyValues[layer].second,
                    context->pastKeyValues[layer].second);
            }
        }
        AssertInFastLLM(
            GetHistoryCacheSequenceLength(context->pastKeyValues) ==
                memory.sequenceLength,
            "GLM-5.3 restored DSA caches are out of sync.");
        if (Glm5NextHistoryCacheDebugEnabled()) {
            fprintf(stderr,
                    "[glm5-next-history] restored=%d state=%s "
                    "dsa=shared-pages\n",
                    memory.sequenceLength,
                    memory.recurrentStateOnCpu ? "cpu" : "gpu");
        }
    }

    void Glm5NextModel::RunKdaAttention(
            int layerIndex, Data &input, int sequence,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            Data &output) {
        const std::string prefix = languagePrefix + "layers." +
            std::to_string(layerIndex) + ".self_attn.";
        Data qCache, kCache, vCache;
        BindGlm5NextConvCacheViews(
            pastKeyValues[layerIndex].first,
            input,
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
        qResidual.FreeSpace();
        qNormalized.FreeSpace();
        // The generic Linear path can materialize these projections as FP32.
        // Paged CUDA attention supports half/bfloat16 queries and cache pages;
        // keep GLM-5.3's advertised BF16 activation/cache contract explicit.
        ToDataType(query, DataType::BFLOAT16);
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
        compressedKv.FreeSpace();
        kvNormalized.FreeSpace();
        ToDataType(expandedKv, DataType::BFLOAT16);
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
        auto initializePagedDescriptor = [](Data &cache,
                                             const Data &current) {
            if (cache.dims.empty() && cache.pageIndex.empty() &&
                cache.pagedKVCacheData == nullptr) {
                cache.dataType = current.dataType;
                cache.UpdateUnitSize();
                cache.dataDevice = current.dataDevice;
                cache.dataDeviceIds = current.dataDeviceIds;
                cache.SetKVCache();
            }
            AssertInFastLLM(
                cache.dataType == current.dataType,
                "GLM-5.3 DSA paged-cache descriptor dtype mismatch.");
        };
        initializePagedDescriptor(keyCache, key);
        initializePagedDescriptor(valueCache, value);
        PagedCacheManager *keyManager = AllocatePagedCacheManager(
            layerIndex * 2,
            PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
            key);
        PagedCacheManager *valueManager = AllocatePagedCacheManager(
            layerIndex * 2 + 1,
            PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
            value);
        AssertInFastLLM(
            keyManager != nullptr && valueManager != nullptr,
            "GLM-5.3 failed to allocate DSA paged-cache managers.");
        AppendPagedCache(*keyManager, keyCache, key);
        AppendPagedCache(*valueManager, valueCache, value);
        key.FreeSpace();
        value.FreeSpace();
        AssertInFastLLM(
            Glm5NextPagedCacheLength(keyCache) == keyCache.dims[1] &&
            Glm5NextPagedCacheLength(valueCache) == valueCache.dims[1] &&
            keyCache.dims[1] == valueCache.dims[1],
            "GLM-5.3 DSA key/value caches are out of sync.");

        // Paged attention reads the shared page pool directly. expandedKv is
        // twice as wide as the result, so reuse its now-dead K/V projection
        // storage for the attention output.
        query.Reshape(
            {num_attention_heads, sequence, qkHeadDim});
        Data attentionHeads;
        attentionHeads.FakeFrom(expandedKv, 0);
        attentionHeads.dataDeviceIds = expandedKv.dataDeviceIds;
        AttentionPaged(
            query, keyCache, valueCache, attentionHeads, 1,
            1.0f / std::sqrt((float)qkHeadDim), 1,
            layerIndex > kvCacheId);
        query.FreeSpace();
        PermuteSelf(attentionHeads, {1, 0, 2});
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
        input.Reshape({sequence, embed_dim});

        ApplyDeviceMap(deviceMap, layerIndex + 1, block_cnt);
        Data routerInput, routerScores;
        ToDataType(input, routerInput, DataType::FLOAT32);
        Linear(routerInput, weight[mlp + "gate.weight"],
               Data(), routerScores);
        Sigmoid(routerScores, routerScores);
        Data expertIndex, expertScore;
        SelectExpert(
            routerScores, expertIndex, expertScore,
            num_experts_per_tok, norm_topk_prob,
            routed_scaling_factor,
            &weight[mlp + "gate.e_score_correction_bias"]);

        if (GetCudaSharedExpert()) {
            ApplyDeviceMap(deviceMap, layerIndex + 1, block_cnt);
        } else {
            ApplyMoeDeviceMapForLayer(layerIndex);
        }
        RunClampedMlp(
            input,
            weight[mlp + "shared_experts.gateup_proj.weight"],
            weight[mlp + "shared_experts.down_proj.weight"],
            output);

        Data w1, w2, w3, tempInput, tempOutput;
        Data moeInputTemp, moeOutputTemp, routedOutput;
        ApplyMoeDeviceMapForLayer(layerIndex);
        MergeMOEBlock(
            &input, &expertIndex, &expertScore,
            &expertWeights[layerIndex], &expertBiases[layerIndex],
            &w1, &w2, &w3, &tempInput, &tempOutput,
            0.0f, &routedOutput, layerIndex,
            input.dataType, moeAtype,
            &moeInputTemp, &moeOutputTemp,
            MoeGateSwiglu, false, swigluLimit, true);

        ApplyDeviceMap(deviceMap, layerIndex + 1, block_cnt);
        AddTo(output, routedOutput);
        input.Reshape(outputDims);
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
        Data lastHiddenView;
        Data *lastHidden = &hiddenStates;
        const int sequence = hiddenStates.dims[1];
        if (sequence > 1) {
            Split(
                hiddenStates, 1, sequence - 1, sequence,
                lastHiddenView);
            lastHidden = &lastHiddenView;
        }
        Data outputLogits;
        Linear(*lastHidden, weight["lm_head.weight"],
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
        AssertInFastLLM(
            inputIds.dims.size() == 2 && inputIds.dims[0] == 1 &&
            inputIds.dims[1] > 0,
            "GLM-5.3 Forward currently supports one non-empty request.");
        const int sequence = inputIds.dims[1];
        ResponseContext *context = nullptr;
        if (saveHistoryChat) {
            std::lock_guard<std::mutex> guard(responseContextsMutex);
            auto it = responseContexts.find(&pastKeyValues);
            if (it != responseContexts.end()) {
                context = it->second;
            }
        }
        const int cachedLength =
            GetHistoryCacheSequenceLength(pastKeyValues);
        const bool isFinalPromptChunk =
            context != nullptr && cachedLength >= 0 &&
            cachedLength + sequence == context->inputTokens &&
            (int)context->allTokens.size() >= context->inputTokens;
        if (!isFinalPromptChunk) {
            return ForwardImpl(
                inputIds, attentionMask, positionIds, pastKeyValues,
                generationConfig, lastTokens, logits, true);
        }

        // KDA stores a recurrent state after every consumed token and cannot
        // be sliced back like DSA K/V. Keep the atomic checkpoint strictly
        // before prompt N and align it to a complete DSA page. Restored
        // requests therefore append into a new page instead of modifying a
        // page that is still shared with history.
        const int pageLen = fastllm::GetPageLen();
        AssertInFastLLM(
            pageLen > 0,
            "GLM-5.3 history cache requires a positive page length.");
        const int checkpointLength =
            (context->inputTokens - 1) / pageLen * pageLen;
        const int prefixLength = checkpointLength - cachedLength;
        AssertInFastLLM(
            prefixLength >= 0 && prefixLength < sequence,
            "GLM-5.3 prompt checkpoint is outside the current chunk.");

        Data suffixInput, suffixPositionIds;
        if (prefixLength > 0) {
            Data prefixInput, prefixPositionIds;
            Split(inputIds, 1, 0, prefixLength, prefixInput);
            Split(inputIds, 1, prefixLength, sequence, suffixInput);
            if (positionIds.dims.size() == 2 &&
                positionIds.dims[0] == 1 &&
                positionIds.dims[1] == sequence) {
                Split(positionIds, 1, 0, prefixLength,
                      prefixPositionIds);
                Split(positionIds, 1, prefixLength, sequence,
                      suffixPositionIds);
            }
            ForwardImpl(
                prefixInput, Data(), prefixPositionIds, pastKeyValues,
                generationConfig, lastTokens, nullptr, false);
        }
        AssertInFastLLM(
            GetHistoryCacheSequenceLength(pastKeyValues) ==
                checkpointLength,
            "GLM-5.3 prompt checkpoint is not page-aligned.");
        if (checkpointLength > 0) {
            std::vector<int> checkpointTokens(
                context->allTokens.begin(),
                context->allTokens.begin() + checkpointLength);
            RecordHistoryCache(
                checkpointTokens, pastKeyValues, checkpointLength);
        }
        if (prefixLength > 0) {
            return ForwardImpl(
                suffixInput, Data(), suffixPositionIds, pastKeyValues,
                generationConfig, lastTokens, logits, true);
        }
        return ForwardImpl(
            inputIds, attentionMask, positionIds, pastKeyValues,
            generationConfig, lastTokens, logits, true);
    }

    int Glm5NextModel::ForwardImpl(
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *logits,
            bool sampleOutput) {
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

        if (!sampleOutput) {
            return 0;
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
