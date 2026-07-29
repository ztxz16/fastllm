#include "kimi_k3.h"

#include "utils.h"
#include "gguf.h"
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

    }

    KimiK3Model::KimiK3Model() {
        model_type = "kimi_k3";
        model_struct = "kimi_k3";
        canDoBatchForward = false;
        dataType = DataType::BFLOAT16;
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
        // FastLLM's scheduler uses one ordinary attention cache to account
        // for active sequence length and reserved KV capacity.  KDA caches
        // contain fixed convolution/recurrent state, so point kvCacheId at
        // the first MLA layer and expose every MLA cache in FastLLM's normal
        // [heads, sequence, dim] Expansion/CatDirect layout.
        kvCacheId = -1;
        elementsInKVCachePerToken = 0;
        for (int layerIndex = 0; layerIndex < block_cnt; layerIndex++) {
            if (!kdaLayers[layerIndex]) {
                if (kvCacheId < 0) {
                    kvCacheId = layerIndex;
                }
                elementsInKVCachePerToken +=
                    kdaHeads * (qkNopeHeadDim + qkRopeHeadDim + vHeadDim);
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
        if (totalWeights == 0) {
            return;
        }

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
#endif
    }

    Data KimiK3Model::RunFirstLayerImpl(
            const std::vector<int> &tokenIds,
            std::vector<std::pair<Data, Data>> *pastKeyValues) {
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

        Data qProjected, kProjected, vProjected;
        Linear(normalized, weight[layer + "self_attn.q_proj.weight"],
               Data(), qProjected);
        Linear(normalized, weight[layer + "self_attn.k_proj.weight"],
               Data(), kProjected);
        Linear(normalized, weight[layer + "self_attn.v_proj.weight"],
               Data(), vProjected);

        Data qConv, kConv, vConv;
        Data qCache, kCache, vCache;
        if (pastKeyValues != nullptr) {
            UnpackKdaConvCache(
                (*pastKeyValues)[0].first, shortConvKernel - 1,
                kdaHeads * kdaHeadDim, qCache, kCache, vCache);
            KimiK3CausalConv1D(
                qProjected, weight[layer + "self_attn.q_conv1d.weight"],
                shortConvKernel, qCache, qConv);
            KimiK3CausalConv1D(
                kProjected, weight[layer + "self_attn.k_conv1d.weight"],
                shortConvKernel, kCache, kConv);
            KimiK3CausalConv1D(
                vProjected, weight[layer + "self_attn.v_conv1d.weight"],
                shortConvKernel, vCache, vConv);
            PackKdaConvCache(
                qCache, kCache, vCache, (*pastKeyValues)[0].first);
        } else {
            KimiK3CausalConv1D(
                qProjected, weight[layer + "self_attn.q_conv1d.weight"],
                shortConvKernel, qConv);
            KimiK3CausalConv1D(
                kProjected, weight[layer + "self_attn.k_conv1d.weight"],
                shortConvKernel, kConv);
            KimiK3CausalConv1D(
                vProjected, weight[layer + "self_attn.v_conv1d.weight"],
                shortConvKernel, vConv);
        }

        qConv.Reshape({1, sequence, kdaHeads, kdaHeadDim});
        kConv.Reshape({1, sequence, kdaHeads, kdaHeadDim});
        vConv.Reshape({1, sequence, kdaHeads, kdaHeadDim});
        Data q, k;
        KimiK3L2Norm(qConv, 1e-6f, q);
        KimiK3L2Norm(kConv, 1e-6f, k);

        Data gateLowRank, rawGate;
        Linear(normalized, weight[layer + "self_attn.f_a_proj.weight"],
               Data(), gateLowRank);
        Linear(gateLowRank, weight[layer + "self_attn.f_b_proj.weight"],
               Data(), rawGate);
        rawGate.Reshape({1, sequence, kdaHeads, kdaHeadDim});
        Data rawBetaBf16, rawBeta;
        Linear(normalized, weight[layer + "self_attn.b_proj.weight"],
               Data(), rawBetaBf16);
        ToDataType(rawBetaBf16, rawBeta, DataType::FLOAT32);

        Data localKdaState, kdaOutput, kdaDecay, kdaBeta;
        Data &kdaState = pastKeyValues == nullptr ?
            localKdaState : (*pastKeyValues)[0].second;
        KimiK3RecurrentKDA(
            q, k, vConv, rawGate, rawBeta,
            weight[layer + "self_attn.A_log"],
            weight[layer + "self_attn.dt_bias"], gateLowerBound,
            kdaState, kdaOutput, kdaDecay, kdaBeta);

        Data outputGate;
        Linear(normalized, weight[layer + "self_attn.g_proj.weight"],
               Data(), outputGate);
        outputGate.Reshape({1, sequence, kdaHeads, kdaHeadDim});
        Data gatedAttention;
        KimiK3RMSNormSigmoidGate(
            kdaOutput, outputGate,
            weight[layer + "self_attn.o_norm.weight"],
            rms_norm_eps, gatedAttention);
        gatedAttention.Reshape({1, sequence, kdaHeads * kdaHeadDim});
        Data attention;
        Linear(gatedAttention, weight[layer + "self_attn.o_proj.weight"],
               Data(), attention);

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
            std::vector<std::pair<Data, Data>> *pastKeyValues) {
        AssertInFastLLM(layerCount >= 1 && layerCount <= block_cnt,
                        "Kimi-K3 requested layer count is out of range.");
        if (pastKeyValues != nullptr) {
            AssertInFastLLM((int)pastKeyValues->size() >= layerCount,
                            "Kimi-K3 cache table is shorter than the decoder.");
        }
        Data prefixSum = RunFirstLayerImpl(
            tokenIds, pastKeyValues);
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

        auto runKdaAttention = [&](int layerIndex, Data &normalized,
                                   Data &attention) {
            const std::string layer = languagePrefix + "layers." +
                std::to_string(layerIndex) + ".";
            Data qProjected, kProjected, vProjected;
            Linear(normalized, weight[layer + "self_attn.q_proj.weight"],
                   Data(), qProjected);
            Linear(normalized, weight[layer + "self_attn.k_proj.weight"],
                   Data(), kProjected);
            Linear(normalized, weight[layer + "self_attn.v_proj.weight"],
                   Data(), vProjected);

            Data qConv, kConv, vConv;
            Data qCache, kCache, vCache;
            if (pastKeyValues != nullptr) {
                UnpackKdaConvCache(
                    (*pastKeyValues)[layerIndex].first,
                    shortConvKernel - 1, kdaHeads * kdaHeadDim,
                    qCache, kCache, vCache);
                KimiK3CausalConv1D(
                    qProjected, weight[layer + "self_attn.q_conv1d.weight"],
                    shortConvKernel, qCache, qConv);
                KimiK3CausalConv1D(
                    kProjected, weight[layer + "self_attn.k_conv1d.weight"],
                    shortConvKernel, kCache, kConv);
                KimiK3CausalConv1D(
                    vProjected, weight[layer + "self_attn.v_conv1d.weight"],
                    shortConvKernel, vCache, vConv);
                PackKdaConvCache(
                    qCache, kCache, vCache,
                    (*pastKeyValues)[layerIndex].first);
            } else {
                KimiK3CausalConv1D(
                    qProjected, weight[layer + "self_attn.q_conv1d.weight"],
                    shortConvKernel, qConv);
                KimiK3CausalConv1D(
                    kProjected, weight[layer + "self_attn.k_conv1d.weight"],
                    shortConvKernel, kConv);
                KimiK3CausalConv1D(
                    vProjected, weight[layer + "self_attn.v_conv1d.weight"],
                    shortConvKernel, vConv);
            }
            qConv.Reshape({1, sequence, kdaHeads, kdaHeadDim});
            kConv.Reshape({1, sequence, kdaHeads, kdaHeadDim});
            vConv.Reshape({1, sequence, kdaHeads, kdaHeadDim});
            Data q, k;
            KimiK3L2Norm(qConv, 1e-6f, q);
            KimiK3L2Norm(kConv, 1e-6f, k);

            Data gateLowRank, rawGate;
            Linear(normalized, weight[layer + "self_attn.f_a_proj.weight"],
                   Data(), gateLowRank);
            Linear(gateLowRank, weight[layer + "self_attn.f_b_proj.weight"],
                   Data(), rawGate);
            rawGate.Reshape({1, sequence, kdaHeads, kdaHeadDim});
            Data rawBetaBf16, rawBeta;
            Linear(normalized, weight[layer + "self_attn.b_proj.weight"],
                   Data(), rawBetaBf16);
            ToDataType(rawBetaBf16, rawBeta, DataType::FLOAT32);

            Data localState, kdaOutput, decay, activatedBeta;
            Data &state = pastKeyValues == nullptr ?
                localState : (*pastKeyValues)[layerIndex].second;
            KimiK3RecurrentKDA(
                q, k, vConv, rawGate, rawBeta,
                weight[layer + "self_attn.A_log"],
                weight[layer + "self_attn.dt_bias"], gateLowerBound,
                state, kdaOutput, decay, activatedBeta);

            Data outputGate;
            Linear(normalized, weight[layer + "self_attn.g_proj.weight"],
                   Data(), outputGate);
            outputGate.Reshape({1, sequence, kdaHeads, kdaHeadDim});
            Data gatedAttention;
            KimiK3RMSNormSigmoidGate(
                kdaOutput, outputGate,
                weight[layer + "self_attn.o_norm.weight"],
                rms_norm_eps, gatedAttention);
            gatedAttention.Reshape(
                {1, sequence, kdaHeads * kdaHeadDim});
            Linear(gatedAttention, weight[layer + "self_attn.o_proj.weight"],
                   Data(), attention);
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

            Data routedInput;
            Linear(input, weight[moe + "routed_expert_down_proj.weight"],
                   Data(), routedInput);
            routedInput.Reshape({sequence, routedExpertHiddenSize});
            Data routedLatent;
            KimiK3RoutedExperts(
                routedInput, expertIndex, expertScore,
                expertW1s[layerIndex], expertW2s[layerIndex],
                expertW3s[layerIndex], situBeta, situLinearBeta,
                routedLatent);
            routedLatent.Reshape({1, sequence, routedExpertHiddenSize});
            Data routedNormalized;
            KimiK3RMSNorm(
                routedLatent, weight[moe + "routed_expert_norm.weight"],
                rms_norm_eps, routedNormalized);
            Data routedOutput;
            Linear(routedNormalized,
                   weight[moe + "routed_expert_up_proj.weight"],
                   Data(), routedOutput);

            Data sharedGate, sharedUp, sharedActivated, sharedOutput;
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
            ApplyMoeDeviceMapForLayer(layerIndex);
            runSparseMoe(layerIndex, mlpNormalized, moeOutput);
            ApplyDeviceMap(deviceMap, layerIndex, layerCount);
            AddTo(prefixSum, moeOutput);
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
        Data hiddenStates = RunLayersImpl(
            tokenIds, block_cnt, &pastKeyValues);
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
