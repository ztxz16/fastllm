//
// Created by huangyuyang on 2/19/26.
//

#include "utils.h"

#include "qwen3_5.h"

#include <algorithm>
#include <sstream>

#include <unordered_map>

#include <cctype>
#include <cstring>
#include <cstdlib>
#include <cmath>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

#ifdef USE_TFACC
#include "fastllm-tfacc.h"
#endif

#ifdef USE_NUMA
#include "fastllm-numa.h"
#endif

namespace fastllm {
#ifdef USE_NUMAS
    extern void RegisterNumas(fastllm::Data *data, std::string weightType);
#endif

    static void Add1(Data &input) {
        if (input.dims.size() == 0) {
            return;
        }
        float *v = (float*)input.cpuData;
        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            v[i] += 1.0f;
        }
    }

    static bool IsTrueString(const std::string &value) {
        std::string lowered = value;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return lowered == "1" || lowered == "true" || lowered == "on";
    }

    static bool IsFloatLikeType(DataType dataType) {
        return dataType == DataType::FLOAT32 ||
               dataType == DataType::FLOAT16 ||
               dataType == DataType::BFLOAT16;
    }

    static void CheckAddInputType(const Data &data, const std::string &name, int layer) {
        if (!IsFloatLikeType(data.dataType)) {
            ErrorInFastLLM("Qwen3.5 layer " + std::to_string(layer) + " AddTo input `" + name +
                           "` has unsupported data type " + std::to_string((int)data.dataType) + ".");
        }
    }

    static bool TryGetFusedMoeLayerPrefix(const std::string &weightName, std::string &layerPrefix) {
        static const std::string gateupSuffix = "experts.gate_up_proj";
        static const std::string downSuffix = "experts.down_proj";
        if (StringEndWith(weightName, gateupSuffix)) {
            layerPrefix = weightName.substr(0, weightName.size() - gateupSuffix.size());
            return true;
        }
        if (StringEndWith(weightName, downSuffix)) {
            layerPrefix = weightName.substr(0, weightName.size() - downSuffix.size());
            return true;
        }
        return false;
    }

    static void SplitExpertLinearWeight(Data &dst, const Data &src, const std::string &name, int expertIndex) {
        AssertInFastLLM(src.dims.size() == 3, "Qwen3.5 MoE fused expert weight should be 3D.");
        AssertInFastLLM(expertIndex >= 0 && expertIndex < src.dims[0], "Qwen3.5 MoE expert index out of range.");
        AssertInFastLLM(src.dataType == DataType::FLOAT16 || src.dataType == DataType::BFLOAT16 ||
                        src.dataType == DataType::FLOAT32,
                        "Qwen3.5 MoE fused expert slicing currently supports float16/bfloat16/float32 weights only.");
        AssertInFastLLM(src.dataDevice == DataDevice::CPU && src.cpuData != nullptr,
                        "Qwen3.5 MoE fused expert slicing expects CPU weight data during load.");

        dst = Data(src.dataType, {src.dims[1], src.dims[2]});
        dst.Allocate();
        const uint64_t bytesPerExpert = src.GetBytes() / src.dims[0];
        memcpy(dst.cpuData, src.cpuData + bytesPerExpert * expertIndex, bytesPerExpert);
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
    }

    static void RegisterExpertLinearWeight(Data &data, const std::string &weightType) {
        if (!GetFastllmEnv().activateNuma) {
            return;
        }

        data.CalcWeightSum();

#if defined(USE_TFACC) || defined(USE_NUMA)
        data.weightSum.resize(1);
        RegisterFastllmData(&data, weightType);
#endif

#if defined(USE_NUMAS)
        RegisterNumas(&data, weightType);
#endif
    }

    const std::string Qwen3_5Model::language_prefix = "model.language_model.";
    const std::string Qwen3_5Model::visual_prefix = "model.visual.";

    int Qwen3_5Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        Data attentionMaskCopy(attentionMask), positionIdsCopy(positionIds);
        std::vector <Data*> attentionMasks = {&attentionMaskCopy};
        std::vector <Data*> positionIdsVec = {&positionIdsCopy};
        std::vector <int> seqLens = {(int)inputIds.dims[1]};
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        std::vector <std::pair <Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
    }

    Qwen3_5Model::Qwen3_5Model() {
        this->model_struct = "qwen3_5";
        this->model_type = "qwen3_5";
        this->use_new_engine = true;
        this->num_experts = 0;
        this->num_experts_per_tok = 0;
        this->norm_topk_prob = true;

        weight.embeddingNames.insert(language_prefix + "embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight",
            language_prefix + "layers.*.mlp.down_proj.weight", language_prefix + "layers.*.mlp.up_proj.weight",
            language_prefix + "layers.*.mlp.gate_proj.weight", language_prefix + "layers.*.mlp.gate_proj.weight",
            language_prefix + "layers.*.mlp.gateup_proj.weight",
            language_prefix + "layers.*.mlp.gate.weight",
            language_prefix + "layers.*.mlp.shared_expert_gate.weight",
            language_prefix + "layers.*.mlp.shared_expert.gate_proj.weight",
            language_prefix + "layers.*.mlp.shared_expert.up_proj.weight",
            language_prefix + "layers.*.mlp.shared_expert.down_proj.weight",
            language_prefix + "layers.*.mlp.shared_expert.gateup_proj.weight",
            language_prefix + "layers.*.mlp.experts.gate_up_proj",
            language_prefix + "layers.*.mlp.experts.down_proj",
            language_prefix + "layers.*.mlp.experts.*.gate_proj.weight",
            language_prefix + "layers.*.mlp.experts.*.up_proj.weight",
            language_prefix + "layers.*.mlp.experts.*.down_proj.weight",
            language_prefix + "layers.*.mlp.experts.*.gateup_proj.weight",
            language_prefix + "layers.*.self_attn.o_proj.weight", language_prefix + "layers.*.self_attn.q_proj.weight",
            language_prefix + "layers.*.self_attn.k_proj.weight",
            language_prefix + "layers.*.self_attn.v_proj.weight", language_prefix + "layers.*.self_attn.mergeqkv.weight",
            language_prefix + "layers.*.self_attn.W_pack.weight",
            language_prefix + "layers.*.linear_attn.in_proj_qkvz.weight",
            language_prefix + "layers.*.linear_attn.in_proj_ba.weight",
            language_prefix + "layers.*.linear_attn.out_proj.weight"
        };
    }

    void Qwen3_5Model::InitParams() {
        std::map<std::string, std::string> extra;
        for (auto &it : this->weight.dicts) {
            std::string key = it.first;
            if (key.substr(0, 12) == "text_config.") {
                std::string stripped = key.substr(12);
                if (stripped.substr(0, 16) == "rope_parameters.") {
                    stripped = stripped.substr(16);
                }
                extra[stripped] = it.second;
            }
        }
        for (auto &it : extra) {
            if (this->weight.dicts.find(it.first) == this->weight.dicts.end()) {
                this->weight.dicts[it.first] = it.second;
            }
        }
        basellm::InitParams();
        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("linear_num_key_heads") != this->weight.dicts.end()) {
            num_k_heads = atoi(this->weight.dicts["linear_num_key_heads"].c_str());
        }
        if (this->weight.dicts.find("linear_num_value_heads") != this->weight.dicts.end()) {
            num_v_heads = atoi(this->weight.dicts["linear_num_value_heads"].c_str());
        }
        if (this->weight.dicts.find("linear_key_head_dim") != this->weight.dicts.end()) {
            head_k_dim = atoi(this->weight.dicts["linear_key_head_dim"].c_str());
        }
        if (this->weight.dicts.find("linear_value_head_dim") != this->weight.dicts.end()) {
            head_v_dim = atoi(this->weight.dicts["linear_value_head_dim"].c_str());
        }
        head_dim = embed_dim / num_attention_heads;
        if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
            head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        }
        if (this->weight.dicts.find("partial_rotary_factor") != this->weight.dicts.end()) {
            rotary_dim = (int)(head_dim * atof(this->weight.dicts["partial_rotary_factor"].c_str()) + 1e-5);
        } else {
            rotary_dim = (int)(head_dim * 0.25 + 1e-5); // qwen3.5的默认值
        }
        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.type") != this->weight.dicts.end()) {
            std::string type = this->weight.dicts["rope_scaling.type"];
            if (type == "linear")
               rope_type = RoPEType::LINEAR_SCALE;
            else if (type == "dynamic")
               rope_type = RoPEType::DYMAMIC_NTK;
        }
        if (this->weight.dicts.find("rope_theta") != this->weight.dicts.end()) {
            rope_base = atof(this->weight.dicts["rope_theta"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }
        num_experts = 0;
        if (this->weight.dicts.find("num_experts") != this->weight.dicts.end()) {
            num_experts = atoi(this->weight.dicts["num_experts"].c_str());
        }
        num_experts_per_tok = 0;
        if (this->weight.dicts.find("num_experts_per_tok") != this->weight.dicts.end()) {
            num_experts_per_tok = atoi(this->weight.dicts["num_experts_per_tok"].c_str());
        }
        if (this->weight.dicts.find("norm_topk_prob") != this->weight.dicts.end()) {
            norm_topk_prob = IsTrueString(this->weight.dicts["norm_topk_prob"]);
        } else {
            norm_topk_prob = true;
        }
        n_shared_experts = 0;
        if (this->weight.dicts.find("shared_expert_intermediate_size") != this->weight.dicts.end() &&
            atoi(this->weight.dicts["shared_expert_intermediate_size"].c_str()) > 0) {
            n_shared_experts = 1;
        }
        weights.clear();
        biass.clear();
        moeWeightsPrepared = false;

        for (int i = 0; i < block_cnt; i++) {
            std::string w1WeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string w3WeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string swigluWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
            );

            if (num_experts > 0) {
                std::string sharedGateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gate_proj.weight";
                std::string sharedUpWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.up_proj.weight";
                std::string sharedGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gateup_proj.weight";
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({sharedGateWeightName, sharedUpWeightName}, sharedGateupWeightName, std::string("linearSwiglu"))})
                );

                for (int j = 0; j < num_experts; j++) {
                    std::string expertGateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gate_proj.weight";
                    std::string expertUpWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".up_proj.weight";
                    std::string expertGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                    std::string expertDownWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                    this->weightMergeRules.push_back(
                        WeightMergeRule({WeightMergeRuleSingle({expertGateWeightName, expertUpWeightName}, expertGateupWeightName, std::string("linearSwiglu"))})
                    );
                    this->specialWeights[expertGateupWeightName] = "linearSwiglu";
                    this->specialWeights[expertDownWeightName] = "linearColumn";
                    this->moeLinears.insert(expertGateWeightName);
                    this->moeLinears.insert(expertUpWeightName);
                    this->moeLinears.insert(expertDownWeightName);
                }
            }

            std::string qWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                                 WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            );

            // Merge GDN linear projections: qkv + z -> qkvz, b + a -> ba
            std::string qkvWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkv.weight";
            std::string zWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_z.weight";
            std::string qkvzWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkvz.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qkvWeightName, zWeightName}, qkvzWeightName, std::string("linear"))})
            );

            std::string bWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_b.weight";
            std::string aWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_a.weight";
            std::string baWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_ba.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({bWeightName, aWeightName}, baWeightName, std::string("linear"))})
            );
        }

        float inv_scale = pow((float)head_k_dim, -0.5);
        std::vector <float> v_inv_scale(head_k_dim, inv_scale);
        Data temp(DataType::FLOAT32, std::vector<int>{head_k_dim}, v_inv_scale);
        inv_scale_data.CopyFrom(temp);
    }

    void Qwen3_5Model::SplitFusedMoeWeightsIfNeeded(const std::string &layerPrefix) {
        const std::string fusedGateupName = layerPrefix + "experts.gate_up_proj";
        const std::string fusedDownName = layerPrefix + "experts.down_proj";
        const std::string firstExpertGateupName = layerPrefix + "experts.0.gateup_proj.weight";
        if (this->weight.weight.find(firstExpertGateupName) != this->weight.weight.end()) {
            return;
        }

        auto fusedGateupIt = this->weight.weight.find(fusedGateupName);
        if (fusedGateupIt == this->weight.weight.end()) {
            return;
        }

        auto fusedDownIt = this->weight.weight.find(fusedDownName);
        AssertInFastLLM(fusedDownIt != this->weight.weight.end(), "Qwen3.5 fused MoE weights are incomplete.");
        Data &fusedGateup = fusedGateupIt->second;
        Data &fusedDown = fusedDownIt->second;
        AssertInFastLLM(fusedGateup.dims.size() == 3 && fusedDown.dims.size() == 3,
                        "Qwen3.5 MoE fused expert weights should be 3D.");
        AssertInFastLLM(fusedGateup.dims[0] == num_experts && fusedDown.dims[0] == num_experts,
                        "Qwen3.5 MoE fused expert count mismatch.");

        for (int j = 0; j < num_experts; j++) {
            const std::string expertGateupName = layerPrefix + "experts." + std::to_string(j) + ".gateup_proj.weight";
            const std::string expertDownName = layerPrefix + "experts." + std::to_string(j) + ".down_proj.weight";
            SplitExpertLinearWeight(this->weight.weight[expertGateupName], fusedGateup, expertGateupName, j);
            SplitExpertLinearWeight(this->weight.weight[expertDownName], fusedDown, expertDownName, j);
            RegisterExpertLinearWeight(this->weight.weight[expertGateupName], "linearSwiglu");
            RegisterExpertLinearWeight(this->weight.weight[expertDownName], "linearColumn");
        }

        this->weight.weight.erase(fusedGateupName);
        this->weight.weight.erase(fusedDownName);
    }

    void Qwen3_5Model::OnWeightLoaded(const std::string &weightName, const std::set<std::string> &finishedWeightNames) {
        if (num_experts <= 0) {
            return;
        }

        std::string layerPrefix;
        if (!TryGetFusedMoeLayerPrefix(weightName, layerPrefix) ||
            !StartWith(layerPrefix, language_prefix + "layers.")) {
            return;
        }

        const std::string fusedGateupName = layerPrefix + "experts.gate_up_proj";
        const std::string fusedDownName = layerPrefix + "experts.down_proj";
        if (finishedWeightNames.find(fusedGateupName) == finishedWeightNames.end() ||
            finishedWeightNames.find(fusedDownName) == finishedWeightNames.end()) {
            return;
        }

        SplitFusedMoeWeightsIfNeeded(layerPrefix);
    }

    void Qwen3_5Model::PrepareMoeWeights() {
        if (moeWeightsPrepared || num_experts <= 0) {
            moeWeightsPrepared = true;
            return;
        }

        weights.clear();
        biass.clear();
        weights.resize(block_cnt);
        biass.resize(block_cnt);

        for (int i = 0; i < block_cnt; i++) {
            const std::string layerPrefix = language_prefix + "layers." + std::to_string(i) + ".mlp.";
            const std::string routerWeightName = layerPrefix + "gate.weight";
            const std::string fusedGateupName = layerPrefix + "experts.gate_up_proj";
            const std::string fusedDownName = layerPrefix + "experts.down_proj";
            const std::string firstExpertGateupName = layerPrefix + "experts.0.gateup_proj.weight";

            const bool hasMoeLayer =
                this->weight.weight.find(routerWeightName) != this->weight.weight.end() ||
                this->weight.weight.find(fusedGateupName) != this->weight.weight.end() ||
                this->weight.weight.find(firstExpertGateupName) != this->weight.weight.end();
            if (!hasMoeLayer) {
                continue;
            }

            SplitFusedMoeWeightsIfNeeded(layerPrefix);

            weights[i].push_back(nullptr);
            weights[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            for (int j = 0; j < num_experts; j++) {
                const std::string expertGateupName = layerPrefix + "experts." + std::to_string(j) + ".gateup_proj.weight";
                const std::string expertDownName = layerPrefix + "experts." + std::to_string(j) + ".down_proj.weight";
                AssertInFastLLM(this->weight.weight.find(expertGateupName) != this->weight.weight.end() &&
                                this->weight.weight.find(expertDownName) != this->weight.weight.end(),
                                "Qwen3.5 MoE expert weights are incomplete.");
                weights[i].push_back(&this->weight[expertGateupName]);
                weights[i].push_back(&this->weight[expertDownName]);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
            }
        }

        moeWeightsPrepared = true;
    }

    std::vector <int> Qwen3_5Model::ForwardV2(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        int seqLen = inputIds.dims[1];

        Data qkv;
        // Data &qkv = this->forwardDataManager.GetData("qkv");
        Data q;
        // Data &q = this->forwardDataManager.GetData("q");
        Data k;
        // Data &k = this->forwardDataManager.GetData("k");
        Data v;
        // Data &v = this->forwardDataManager.GetData("v");
        Data embeddingResult;
        // Data &embeddingResult = this->forwardDataManager.GetData("embeddingResult");
        Data hiddenStates;
        // Data &hiddenStates = this->forwardDataManager.GetData("hiddenStates");
        Data attenInput;
        // Data &attenInput = this->forwardDataManager.GetData("attenInput");
        Data attenLastOutput;
        // Data &attenLastOutput = this->forwardDataManager.GetData("attenLastOutput");
        std::vector <Data*> pointersK;
        pointersK.resize(batch);


        std::vector<Data*> batchPastKeys;
        std::vector<Data*> batchPastValues;
        batchPastKeys.resize(batch);
        batchPastValues.resize(batch);

        Data allPositionIds;
        // Data &allPositionIds = this->forwardDataManager.GetData("allPositionIds");
        Data qSizes;
        // Data &qSizes = this->forwardDataManager.GetData("qSizes");
        Data pageSizes;
        // Data &pageSizes = this->forwardDataManager.GetData("pageSizes");
        Data pageIndexs;
        // Data &pageIndexs = this->forwardDataManager.GetData("pageIndexs");
        Data lastPageLens;
        // Data &lastPageLens = this->forwardDataManager.GetData("lastPageLens");
        Data insertIndexs;
        // Data &insertIndexs = this->forwardDataManager.GetData("insertIndexs");
        Data insertPositions;
        // Data &insertPositions = this->forwardDataManager.GetData("insertPositions");
        Data attenOutput;
        // Data &attenOutput = this->forwardDataManager.GetData("attenOutput");
        Data w1, w2, w3;
        Data routerLogits;
        Data sharedGate;
        Data moeFinal, moeFinal2;
        Data tempInput, tempOutput;
        Data attenPart, moePart;
        bool generatedBatchDecodeParams = false;
        bool generatedAppendPagedCacheBatchParams = false;

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;            
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            std::vector <float> vPositionIds;            
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*)positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));
        }

        PrepareMoeWeights();

        if (!initialized_add1) {
            for (int i = 0; i < block_cnt; i++) {
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".input_layernorm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".post_attention_layernorm.weight"]);
            }
            Add1(this->weight[language_prefix + "norm.weight"]);
            initialized_add1 = true;
        }

        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            // 如果tie weight，那么embedding在cuda上处理
            SetCudaEmbedding(true);
        }
        Embedding(inputIds, this->weight[language_prefix + "embed_tokens.weight"], embeddingResult);

        ToDataType(embeddingResult, hiddenStates, this->dataType);
        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string inputRmsName = language_prefix + "layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string qNormName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight";
            std::string kNormName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight";
            std::string oWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string postRmsName = language_prefix + "layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string swigluWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            std::string downWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.down_proj.weight";

            RMSNorm(hiddenStates, this->weight[inputRmsName], rms_norm_eps, attenInput);
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            Data &pastKey = *pastKeyValues[i].first, &pastValue = *pastKeyValues[i].second;

            if (weight.weight.find(language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight") != weight.weight.end()) {
                // Gate Attention Block
                std::string qWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
                std::string qBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.bias";
                std::string kWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.weight";
                std::string kBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.bias";
                std::string vWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.weight";
                std::string vBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.bias";
                std::string oWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                std::string oBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

                Data qgate, q, gate, k, v, mergedQkv;
                if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], mergedQkv);

                    int qgateDim = num_attention_heads * this->head_dim * 2;
                    int kvDim = num_key_value_heads * this->head_dim;
                    Split(mergedQkv, -1, 0, qgateDim, qgate);
                    Split(mergedQkv, -1, qgateDim, qgateDim + kvDim, k);
                    Split(mergedQkv, -1, qgateDim + kvDim, qgateDim + kvDim * 2, v);
                } else {
                    Linear(attenInput, weight[qWeightName], weight[qBiasName], qgate);
                    Linear(attenInput, weight[kWeightName], weight[kBiasName], k);
                    Linear(attenInput, weight[vWeightName], weight[vBiasName], v);
                }

                qgate.Reshape({bsz, seqlen, -1, this->head_dim * 2});
                Split(qgate, -1, 0, this->head_dim, q);
                Split(qgate, -1, this->head_dim, qgate.dims.back(), gate);
                gate.Reshape({bsz, seqlen, -1});

                k.Reshape({bsz, seqlen, -1, this->head_dim});
                v.Reshape({bsz, seqlen, -1, this->head_dim});

                RMSNorm(q, this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);
                {
                    float ropeScale = (rope_type == RoPEType::LINEAR_SCALE) ? rope_factor : 1.0f;
                    fastllm::RopeEncoding(q, allPositionIds, rotary_dim, rope_base, ropeScale);
                    fastllm::RopeEncoding(k, allPositionIds, rotary_dim, rope_base, ropeScale);
                }

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});
                std::vector <int> qkvSize = {-1, seqlen, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                {
                    // Paged Attention
                    PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                        i * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, k);
                    PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                        i * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, v);
                    AppendPagedCache(*pagedCacheKManager, pastKey, k);
                    AppendPagedCache(*pagedCacheVManager, pastValue, v);
                    AttentionPaged(q, pastKey, pastValue, qkv, q.dims[0] / k.dims[0], 1.0 / sqrt(head_dim), 1, i > 0);
                }

                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});

                Sigmoid(gate, gate);
                MulTo(qkv, gate);
                
                Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
                Linear(qkv, weight[oWeightName], oBias, attenInput);
            } else {
                // Gated Delta Net Block
                Data &pastKey = *pastKeyValues[i].first, &pastValue = *pastKeyValues[i].second;
                pastKey.isLinearAttention = pastValue.isLinearAttention = true;
                std::string qkvzWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkvz.weight";
                std::string baWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_ba.weight";
                std::string conv1dWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.conv1d.weight";
                std::string conv1dBiasName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.conv1d.bias";
                std::string aLogName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.A_log";
                std::string dtBiasName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.dt_bias";

                int kd = num_k_heads * head_k_dim, vd = num_v_heads * head_v_dim;

                // Optimization 1: Merge 4 Linear calls into 2
                // qkv+z fused into one Linear, b+a fused into one Linear
                Data mixed_qkvz, ba_merged, mixed_qkv, z, b, a, g;
                Linear(attenInput, weight[qkvzWeightName], Data(), mixed_qkvz);
                Linear(attenInput, weight[baWeightName], Data(), ba_merged);

                // Split qkvz -> mixed_qkv + z
                int qkvz_dim = kd * 2 + vd;
                Split(mixed_qkvz, -1, 0, qkvz_dim, mixed_qkv);
                Split(mixed_qkvz, -1, qkvz_dim, qkvz_dim + vd, z);

                // Split ba -> b + a (note: b and a have dim num_v_heads, not vd)
                Split(ba_merged, -1, 0, num_v_heads, b);
                Split(ba_merged, -1, num_v_heads, num_v_heads * 2, a);

                // mixed_qkv: (bsz, seqlen, key_dim*2+value_dim) -> transpose to (bsz, key_dim*2+value_dim, seqlen)
                PermuteSelf(mixed_qkv, {0, 2, 1});

                z.Reshape({bsz, seqlen, -1, head_v_dim});
                Data conv;
                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    Data hidden_states_new;
                    Cat(pastKey, mixed_qkv, -1, hidden_states_new);
                    Split(hidden_states_new, -1, hidden_states_new.dims.back() - 4, hidden_states_new.dims.back(), pastKey);
                    Conv1DPerChannel(
                        hidden_states_new, weight[conv1dWeightName], weight[conv1dBiasName], 
                        hidden_states_new.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 0, 
                        conv
                    );
                    Split(conv, -1, conv.dims.back() - 1, conv.dims.back(), mixed_qkv);
                    Silu(mixed_qkv, mixed_qkv);
                } else {
                    if (mixed_qkv.dims.back() >= 4) {
                        Split(mixed_qkv, -1, mixed_qkv.dims.back() - 4, mixed_qkv.dims.back(), pastKey);
                        pastKey.expansionDims = pastKey.dims;
                    } else {
                        Data temp;
                        Mul(mixed_qkv, 1.0f, temp);
                        Repeat(temp, -1, 4, mixed_qkv);
                        // ErrorInFastLLM("mixed_qkv.dims.back() < 4");
                    }

                    Conv1DPerChannel(
                        mixed_qkv, weight[conv1dWeightName], weight[conv1dBiasName], 
                        mixed_qkv.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 3, 
                        conv
                    );
                    Split(conv, -1, 0, seqlen, mixed_qkv);
                    Silu(mixed_qkv, mixed_qkv);
                }

                // mixed_qkv: (bsz, conv_dim, seqlen) -> transpose back to (bsz, seqlen, conv_dim)
                PermuteSelf(mixed_qkv, {0, 2, 1});

                Split(mixed_qkv, -1, 0, kd, q);
                Split(mixed_qkv, -1, kd, kd + kd, k);
                Split(mixed_qkv, -1, kd + kd, kd + kd + vd, v);
                
                q.Reshape({q.dims[0], q.dims[1], -1, head_k_dim});
                k.Reshape({k.dims[0], k.dims[1], -1, head_k_dim});
                v.Reshape({v.dims[0], v.dims[1], -1, head_v_dim});

                Sigmoid(b, b);
                MambaSoftplus(a, weight[aLogName], weight[dtBiasName], g);


                Data &last_recurrent_state = pastValue;

                Data core_attn_out, core_attn_out_temp;
                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    // torch_recurrent_gated_delta_rule
                    {
                        RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                        RMSNorm(k, inv_scale_data, rms_norm_eps, k);
                    }

                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});
                    PermuteSelf(b, {0, 2, 1});
                    PermuteSelf(g, {0, 2, 1});

                    int key_batch_size = k.dims[0], key_sequence_length = k.dims[1], key_num_heads = k.dims[2], key_k_head_dim = k.dims[3];
                    int v_head_dim_local = v.dims.back();
                    float scale = 1.0f / pow(q.dims.back(), 0.5);
                    Mul(q, scale, q);

                    RecurrentGatedDeltaRule (
                        q, k, v, g, b,
                        last_recurrent_state, 
                        core_attn_out
                    ); 
                    PermuteSelf(core_attn_out, {0, 2, 1, 3});
                } else {
                    // torch_chunk_gated_delta_rule
                    if (num_v_heads / num_k_heads > 1) {
                        Data qrepeat, krepeat;
                        Mul(q, 1.0f, qrepeat);
                        Mul(k, 1.0f, krepeat);

                        qrepeat.Resize({q.dims[0], q.dims[1], q.dims[2], 1, q.dims[3]});
                        krepeat.Resize({k.dims[0], k.dims[1], k.dims[2], 1, k.dims[3]});

                        Repeat(qrepeat, 3, num_v_heads / num_k_heads, q);
                        Repeat(krepeat, 3, num_v_heads / num_k_heads, k);

                        q.Reshape({q.dims[0], q.dims[1], -1, q.dims.back()});
                        k.Reshape({k.dims[0], k.dims[1], -1, k.dims.back()});
                    }

                    {
                        RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                        RMSNorm(k, inv_scale_data, rms_norm_eps, k);
                    }

                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});
                    PermuteSelf(b, {0, 2, 1});
                    PermuteSelf(g, {0, 2, 1});
                    
                    int key_batch_size = k.dims[0], key_sequence_length = k.dims[1], key_num_heads = k.dims[2], key_k_head_dim = k.dims[3];

                    int chunk_size = 64;
                    int v_head_dim_local = v.dims.back();
                    int seq = k.dims[2];
                    int pad_size = (chunk_size - seq % chunk_size) % chunk_size;

                    Data qq, kk, vv, bb, gg, decayMask;
                    Data qq_pad, kk_pad, vv_pad, bb_pad, gg_pad; // used only when pad_size > 0
                    Data *pkk, *pvv, *pbb, *pgg;
                    if (pad_size > 0) {
                        Data qtemp;
                        Pad(q, 2, pad_size, qtemp);
                        Pad(k, 2, pad_size, kk_pad);
                        Pad(v, 2, pad_size, vv_pad);
                        Pad(b, 2, pad_size, bb_pad);
                        Pad(g, 2, pad_size, gg_pad);
                        float scale = 1.0f / pow(qtemp.dims.back(), 0.5);
                        Mul(qtemp, scale, qq);
                        pkk = &kk_pad; pvv = &vv_pad; pbb = &bb_pad; pgg = &gg_pad;
                    } else {
                        // Avoid 5 Mul(x, 1.0f, y) copies when no padding needed
                        float scale = 1.0f / pow(q.dims.back(), 0.5);
                        Mul(q, scale, qq);
                        pkk = &k; pvv = &v; pbb = &b; pgg = &g;
                    }

                    int tot_heads = seq + pad_size;

                    pbb->Resize({(*pbb).dims[0], (*pbb).dims[1], (*pbb).dims[2], 1});
                    Data k_beta, v_beta;
                    Mul(*pkk, 1.0f, k_beta);
                    Mul(*pvv, 1.0f, v_beta);
                    MulTo(k_beta, *pbb);
                    MulTo(v_beta, *pbb);

                    qq.Reshape({qq.dims[0], qq.dims[1], -1, chunk_size, qq.dims.back()});
                    pkk->Reshape({(*pkk).dims[0], (*pkk).dims[1], -1, chunk_size, (*pkk).dims.back()});
                    k_beta.Reshape({k_beta.dims[0], k_beta.dims[1], -1, chunk_size, k_beta.dims.back()});
                    v_beta.Reshape({v_beta.dims[0], v_beta.dims[1], -1, chunk_size, v_beta.dims.back()});
                    pgg->Reshape({(*pgg).dims[0], (*pgg).dims[1], -1, chunk_size});

                    CumSumLastDim(*pgg);
                    MakeDecayMask(*pgg, decayMask);

                    Data at, attn;
                    MatMulTransB(k_beta, *pkk, at);
                    Mul(at, -1.0f, attn);
                    MulTo(attn, decayMask);

                    CausalMask(attn, 0, 0.0f);
                    TransferAttn(attn);
                    MatMul(attn, v_beta, vv);
                    Data k_cumdecay, g_exp;                    
                    Exp(*pgg, g_exp);

                    // Optimization: avoid k_temp copy - MulTo k_beta directly since k_beta is not used after this
                    MulTo(k_beta, g_exp);
                    MatMul(attn, k_beta, k_cumdecay);

                    MatMulTransB(qq, *pkk, attn);
                    MulTo(attn, decayMask);
                    CausalMask(attn, 1, 0.0f);

                    if (last_recurrent_state.dims.size() == 0) {
#ifdef USE_CUDA
                        if (qq.dataDevice == DataDevice::CUDA) {
                            last_recurrent_state.dataDevice = qq.dataDevice;
                            last_recurrent_state.dataDeviceIds = qq.dataDeviceIds;
                        }
#endif
                        last_recurrent_state.Resize({key_batch_size, key_sequence_length, key_k_head_dim, v_head_dim_local});
                        last_recurrent_state.Allocate(0.0f);
                    }

                    auto runChunkPrefillReference = [&](Data &state, Data &out) {
                        auto makeChunk4D = [](Data &src, int idx, Data &dst) {
                            dst.dims = {src.dims[1], src.dims[2], src.dims[3], src.dims[4]};
                            dst.strides = {src.strides[1], src.strides[2], src.strides[3], src.strides[4]};
                            dst.FakeFrom(src, (size_t) idx * src.strides[0] * src.unitSize);
                        };
                        auto makeChunk3D = [](Data &src, int idx, Data &dst) {
                            dst.dims = {src.dims[1], src.dims[2], src.dims[3]};
                            dst.strides = {src.strides[1], src.strides[2], src.strides[3]};
                            dst.FakeFrom(src, (size_t) idx * src.strides[0] * src.unitSize);
                        };

                        for (int ci = 0; ci < tot_heads / chunk_size; ci++) {
                            Data q_i, k_i, v_i, attn_i, k_cumdecay_i;
                            makeChunk4D(qq, ci, q_i);
                            makeChunk4D(*pkk, ci, k_i);
                            makeChunk4D(vv, ci, v_i);
                            makeChunk4D(attn, ci, attn_i);
                            makeChunk4D(k_cumdecay, ci, k_cumdecay_i);

                            Data v_prime, v_new;
                            MatMul(k_cumdecay_i, state, v_prime);
                            Mul(v_prime, -1.0f, v_new);
                            AddTo(v_new, v_i);

                            Data attn_inter, g_i, g_i_exp;
                            makeChunk3D(*pgg, ci, g_i);
                            makeChunk3D(g_exp, ci, g_i_exp);
                            g_i_exp.Resize({g_i_exp.dims[0], g_i_exp.dims[1], g_i_exp.dims[2], 1});
                            MulTo(q_i, g_i_exp);

                            MatMul(q_i, state, attn_inter);
                            Data atv;
                            MatMul(attn_i, v_new, atv);
                            AddTo(atv, attn_inter);
                            atv.Resize({atv.dims[0], atv.dims[1], 1, atv.dims[2], atv.dims[3]});
                            if (ci == 0) {
                                Mul(atv, 1.0f, out);
                            } else {
                                Mul(out, 1.0f, core_attn_out_temp);
                                Cat(core_attn_out_temp, atv, 3, out);
                            }

                            Data g_i_last, g_i_last_repeat, g_i_delta, g_i_scale;
                            Split(g_i, 2, g_i.dims[2] - 1, g_i.dims[2], g_i_last);
                            Repeat(g_i_last, 2, g_i.dims[2], g_i_last_repeat);
                            Mul(g_i, -1.0f, g_i_delta);
                            AddTo(g_i_last_repeat, g_i_delta);
                            Exp(g_i_last_repeat, g_i_scale);
                            g_i_scale.Resize({g_i_scale.dims[0], g_i_scale.dims[1], g_i_scale.dims[2], 1});
                            MulTo(k_i, g_i_scale);

                            Data k_i_v_new;
                            PermuteSelf(k_i, {0, 1, 3, 2});
                            MatMul(k_i, v_new, k_i_v_new);

                            Data g_i_exp_last;
                            g_i_exp_last.dims = {g_i_exp.dims[0], g_i_exp.dims[1], 1, g_i_exp.dims[3]};
                            g_i_exp_last.strides = {g_i_exp.strides[0], g_i_exp.strides[1], g_i_exp.strides[2], g_i_exp.strides[3]};
                            g_i_exp_last.FakeFrom(g_i_exp, (size_t)(g_i_exp.dims[2] - 1) * g_i_exp.strides[2] * g_i_exp.unitSize);
                            MulTo(state, g_i_exp_last);
                            AddTo(state, k_i_v_new);
                        }
                    };

                    bool useFusedChunkPrefill = false;
#ifdef USE_CUDA
                    if (qq.dataDevice == DataDevice::CUDA &&
                        pkk->dataDevice == DataDevice::CUDA &&
                        vv.dataDevice == DataDevice::CUDA &&
                        pgg->dataDevice == DataDevice::CUDA &&
                        attn.dataDevice == DataDevice::CUDA &&
                        k_cumdecay.dataDevice == DataDevice::CUDA &&
                        last_recurrent_state.dataDevice == DataDevice::CUDA) {
                        useFusedChunkPrefill = GetFastllmEnv().useFusedGdnPrefill;
                    }
#endif

                    if (useFusedChunkPrefill) {
#ifdef USE_CUDA
                        ChunkGatedDeltaRulePrefill(
                            qq, *pkk, vv, *pgg, attn, k_cumdecay,
                            last_recurrent_state, core_attn_out
                        );
#endif
                    } else {
                        PermuteSelf(qq, {2, 0, 1, 3, 4});
                        PermuteSelf(*pkk, {2, 0, 1, 3, 4});
                        PermuteSelf(vv, {2, 0, 1, 3, 4});
                        PermuteSelf(attn, {2, 0, 1, 3, 4});
                        PermuteSelf(k_cumdecay, {2, 0, 1, 3, 4});
                        PermuteSelf(*pgg, {2, 0, 1, 3});
                        runChunkPrefillReference(last_recurrent_state, core_attn_out);
                    }

                    core_attn_out.Reshape({core_attn_out.dims[0], core_attn_out.dims[1], -1, core_attn_out.dims.back()});
                    if (pad_size > 0) {
                        Split(core_attn_out, 2, 0, seq, core_attn_out_temp);
                        PermuteSelf(core_attn_out_temp, {0, 2, 1, 3});
                        Mul(core_attn_out_temp, 1.0f, core_attn_out);
                    } else {
                        PermuteSelf(core_attn_out, {0, 2, 1, 3});
                    }
                }

                {
                    std::vector <int> zShape = z.dims;
                    core_attn_out.Reshape({-1, core_attn_out.dims.back()});
                    z.Reshape({-1, z.dims.back()});

                    RMSNorm(core_attn_out, this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.norm.weight"], rms_norm_eps, core_attn_out);
                    Silu(z, z);
                    MulTo(core_attn_out, z);

                    core_attn_out.Reshape({zShape[0], zShape[1], -1});
                    Linear(core_attn_out, 
                        this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.out_proj.weight"], 
                        Data(), attenInput);
                }
            }

            CheckAddInputType(hiddenStates, "hiddenStates_after_attn", i);
            AddTo(hiddenStates, attenInput);
            RMSNorm(hiddenStates, this->weight[postRmsName], rms_norm_eps, attenInput);
            if (weight.weight.find(swigluWeightName) != weight.weight.end() &&
                weight.weight.find(downWeightName) != weight.weight.end()) {
                MLPBlock(&attenInput, &weight[swigluWeightName], &weight[downWeightName], &v, &q, &hiddenStates);
                continue;
            }

            std::string gateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gate.weight";
            if (weight.weight.find(gateWeightName) == weight.weight.end()) {
                ErrorInFastLLM("Qwen3.5 layer " + std::to_string(i) + " has neither dense MLP nor MoE weights.");
            }

            std::string gateBiasName = language_prefix + "layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";
            std::string firstExpertGateupName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts.0.gateup_proj.weight";
            std::string sharedGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gateup_proj.weight";
            std::string sharedGateProjWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gate_proj.weight";
            std::string sharedUpProjWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.up_proj.weight";
            std::string sharedDownWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.down_proj.weight";
            std::string sharedExpertGateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert_gate.weight";
            Data *gateBiasData = weight.weight.find(gateBiasName) != weight.weight.end() ? &weight[gateBiasName] : nullptr;

            int flatBatch = attenInput.dims[0];
            int flatLen = attenInput.dims[1];
            attenInput.Reshape({flatBatch * flatLen, attenInput.dims[2]});
            moeFinal = Data();
            moeFinal2 = Data();
            sharedGate = Data();

            Linear(attenInput, weight[gateWeightName], Data(), routerLogits);
            ToDataType(routerLogits, DataType::FLOAT32);
            if (gateBiasData != nullptr) {
                ToDataType(*gateBiasData, DataType::FLOAT32);
            }
            Softmax(routerLogits, routerLogits, -1);

            if (weight.weight.find(sharedDownWeightName) != weight.weight.end()) {
                if (weight.weight.find(sharedGateupWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[sharedGateupWeightName], Data(), w3);
                    Swiglu(w3, w1);
                } else if (weight.weight.find(sharedGateProjWeightName) != weight.weight.end() &&
                           weight.weight.find(sharedUpProjWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[sharedGateProjWeightName], Data(), w1);
                    Silu(w1, w1);
                    Linear(attenInput, weight[sharedUpProjWeightName], Data(), w3);
                    MulTo(w1, w3);
                }
                if (w1.dims.size() != 0) {
                    Linear(w1, weight[sharedDownWeightName], Data(), moeFinal2);
                    if (weight.weight.find(sharedExpertGateWeightName) != weight.weight.end()) {
                        Linear(attenInput, weight[sharedExpertGateWeightName], Data(), sharedGate);
                        Sigmoid(sharedGate, sharedGate);
                        MulTo(moeFinal2, sharedGate);
                    }
                }
            }

            bool useMergeMoe = weight.weight.find(firstExpertGateupName) != weight.weight.end() &&
                               !weights[i].empty() && CanRunMergeMOE(attenInput, biass[i]);
            if (useMergeMoe) {
                Data expertIndex, expertScore;
                SelectExpert(routerLogits, expertIndex, expertScore, this->num_experts_per_tok, this->norm_topk_prob,
                             this->routed_scaling_factor, gateBiasData);
                ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
                MergeMOE(
                    attenInput, expertIndex, expertScore,
                    weights[i], biass[i],
                    w1, w2, w3, tempInput, tempOutput,
                    1.0f,
                    moeFinal, i
                );
            } else {
                routerLogits.ToDevice(DataDevice::CPU);
                float *cpuRouterLogits = (float*)routerLogits.cpuData;
                float *cpuBias = nullptr;
                if (gateBiasData != nullptr) {
                    gateBiasData->ToDevice(DataDevice::CPU);
                    cpuBias = (float*)gateBiasData->cpuData;
                }
                int expertCount = routerLogits.dims.back();

                moeFinal.dataType = hiddenStates.dataType;
                moeFinal.dataDevice = attenInput.dataDevice;
                moeFinal.dataDeviceIds = attenInput.dataDeviceIds;
                moeFinal.UpdateUnitSize();
                moeFinal.Resize({0, attenInput.dims[1]});
                moeFinal.Expansion(attenInput.dims);
                for (int b = 0; b < flatBatch * flatLen; b++) {
                    float *cur = cpuRouterLogits + b * expertCount;
                    std::vector <std::pair <float, int> > candidates;
                    candidates.reserve(expertCount);
                    for (int j = 0; j < expertCount; j++) {
                        float score = cur[j];
                        if (cpuBias != nullptr) {
                            score += cpuBias[j];
                        }
                        candidates.push_back(std::make_pair(-score, j));
                    }
                    std::sort(candidates.begin(), candidates.end());

                    Data *currentData = &attenInput;
                    if (flatBatch * flatLen != 1) {
                        Split(attenInput, 0, b, b + 1, attenPart);
                        currentData = &attenPart;
                    }
                    moePart.dataType = hiddenStates.dataType;
                    moePart.dataDevice = currentData->dataDevice;
                    moePart.dataDeviceIds = currentData->dataDeviceIds;
                    moePart.UpdateUnitSize();
                    moePart.Resize(currentData->dims);
                    moePart.Allocate(0.0f);

                    float sum = 0.0f;
                    for (int j = 0; j < this->num_experts_per_tok; j++) {
                        sum += cur[candidates[j].second];
                    }
                    if (!this->norm_topk_prob) {
                        sum = 1.0f;
                    }

                    for (int j = 0; j < this->num_experts_per_tok; j++) {
                        int idx = candidates[j].second;
                        float value = cur[idx];
                        if (sum != 0.0f) {
                            value /= sum;
                        }
                        value *= this->routed_scaling_factor;

                        std::string expertGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight";
                        std::string expertDownWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".down_proj.weight";
                        AssertInFastLLM(weight.weight.find(expertGateupWeightName) != weight.weight.end() &&
                                        weight.weight.find(expertDownWeightName) != weight.weight.end(),
                                        "Qwen3.5 MoE expert weights are incomplete.");
                        Linear(*currentData, weight[expertGateupWeightName], Data(), w3);
                        Swiglu(w3, w1);
                        Linear(w1, weight[expertDownWeightName], Data(), w2);
                        if (w2.dataType != moePart.dataType) {
                            ToDataType(w2, moePart.dataType);
                        }
                        CheckAddInputType(moePart, "moePart", i);
                        AddTo(moePart, w2, value);
                    }
                    if (moePart.dataType != moeFinal.dataType) {
                        ToDataType(moePart, moeFinal.dataType);
                    }
                    CatDirect(moeFinal, moePart, 0);
                }
                moeFinal.expansionDims.clear();
            }

            moeFinal.Reshape(hiddenStates.dims);
            Data tempMoeFinal;
            tempMoeFinal.CopyFrom(moeFinal);
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            if (tempMoeFinal.dataType != hiddenStates.dataType) {
                ToDataType(tempMoeFinal, hiddenStates.dataType);
            }
            CheckAddInputType(hiddenStates, "hiddenStates_after_moe", i);
            AddTo(hiddenStates, tempMoeFinal);
            if (moeFinal2.dims.size() != 0) {
                moeFinal2.Reshape(hiddenStates.dims);
                if (moeFinal2.dataType != hiddenStates.dataType) {
                    ToDataType(moeFinal2, hiddenStates.dataType);
                }
                CheckAddInputType(hiddenStates, "hiddenStates_after_shared_expert", i);
                AddTo(hiddenStates, moeFinal2);
            }
        }

        std::string lmHeadWeightName = "lm_head.weight";
        if (this->weight.weight.find(lmHeadWeightName) == this->weight.weight.end()) {
            lmHeadWeightName = language_prefix + "embed_tokens.weight";
        }
        std::vector <int> lastRet;
        LLMSamplingBlock(
            this, &hiddenStates,
            &weight[language_prefix + "norm.weight"], &weight[lmHeadWeightName],
            rms_norm_eps, batch, all1, seqLens,
            pastKeyValues, generationConfigs, lastTokens,
            retLogits, lastRet
        );
        return lastRet;
    }

    bool Qwen3_5Model::NeedAttentionMask(int qlen, int klen) {
        return false;
    }

    std::string Qwen3_5Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Qwen3_5Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void Qwen3_5Model::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(this->dataType, {1, 1}, {0});
        Data positionIds = Data(this->dataType, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            this->weight["lm_head.weight"] = Data();
            this->weight["lm_head.weight"].CopyFrom(this->weight[language_prefix + "embed_tokens.weight"]);
            ToDataType(this->weight["lm_head.weight"], this->dataType);
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        this->kvCacheId = 0;
        elementsInKVCachePerToken = 0;
        bool foundTokenGrowingCache = false;
        for (int i = 0; i < block_cnt; i++) {
            if (pastKeyValues[i].first.isLinearAttention || pastKeyValues[i].second.isLinearAttention) {
                continue;
            }
            if (pastKeyValues[i].first.dims.size() < 3 || pastKeyValues[i].second.dims.size() < 3) {
                continue;
            }
            if (!foundTokenGrowingCache) {
                this->kvCacheId = i;
                foundTokenGrowingCache = true;
            }
            elementsInKVCachePerToken +=
                (long long)pastKeyValues[i].first.dims[0] * pastKeyValues[i].first.dims[2] +
                (long long)pastKeyValues[i].second.dims[0] * pastKeyValues[i].second.dims[2];
        }
        printf("finish.\n");
    }
}
