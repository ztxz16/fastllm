//
// Created by huangyuyang on 10/9/25.
//

#include "utils.h"

#include "gpt_oss.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    extern std::vector <float> GetInterLeavePowerOf2(int n);
    extern std::vector <float> GetInterleave(int n);

    GptOssModel::GptOssModel() {
        this->model_type = "gpt_oss";
        this->model_struct = "gpt_oss";

        // 默认使用alpaca的提示词和instruction
        this->pre_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";
        this->user_role = "### Instruction:\n";
        this->bot_role = "\n\n### Response:";
        this->history_sep = "</s>";

        block_cnt = 32;
        rotary_dim = 128;

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight", "model.layers.*.down_proj.weight", "model.layers.*.up_proj.weight",
            "model.layers.*.gate_proj.weight",  "model.layers.*.gate_proj.weight", "model.layers.*.gateup_proj.weight",
            "model.layers.*.self_attn.o_proj.weight", "model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight", "model.layers.*.self_attn.mergeqkv.weight", "model.layers.*.self_attn.W_pack.weight",
            "model.layers.*.mlp.*.weight"
        };
    }

    void GptOssModel::InitParams() {
        basellm::InitParams();
        rope_scaling_type = this->weight.dicts["rope_scaling.rope_type"];
        AssertInFastLLM(rope_scaling_type == "yarn", "Fastllm.GptOSS: Only support ropescaling.type = yarn.");

        if (this->weight.dicts.find("rope_scaling.beta_fast") != this->weight.dicts.end()) {
            rope_scaling_beta_fast = atoi(this->weight.dicts["rope_scaling.beta_fast"].c_str());
        } else {
            rope_scaling_beta_fast = 32;
        }
        if (this->weight.dicts.find("rope_scaling.beta_slow") != this->weight.dicts.end()) {
            rope_scaling_beta_slow = atoi(this->weight.dicts["rope_scaling.beta_slow"].c_str());
        } else {
            rope_scaling_beta_slow = 1;
        }
        if (this->weight.dicts.find("rope_scaling.original_max_position_embeddings") != this->weight.dicts.end()) {
            rope_scaling_original_max_position_embeddings = atoi(this->weight.dicts["rope_scaling.original_max_position_embeddings"].c_str());
        } else {
            rope_scaling_original_max_position_embeddings = 4096;
        }
        if (this->weight.dicts.find("rope_scaling.mscale") != this->weight.dicts.end()) {
            rope_scaling_mscale = atof(this->weight.dicts["rope_scaling.mscale"].c_str());
        } else {
            rope_scaling_mscale = 0.707;
        }
        if (this->weight.dicts.find("rope_scaling.mscale_all_dim") != this->weight.dicts.end()) {
            rope_scaling_mscale_all_dim = atof(this->weight.dicts["rope_scaling.mscale_all_dim"].c_str());
        } else {
            rope_scaling_mscale_all_dim = 0.707;
        }

        num_experts = atoi(this->weight.dicts["n_routed_experts"].c_str());
        num_experts_per_tok = atoi(this->weight.dicts["num_experts_per_tok"].c_str());
        n_shared_experts = atoi(this->weight.dicts["n_shared_experts"].c_str());
        norm_topk_prob = (this->weight.dicts["norm_topk_prob"] == "true");

        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
            head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        }
        embed_dim = head_dim * num_attention_heads;
        rotary_dim = head_dim;

        if (this->weight.dicts.find("partial_rotary_factor") != this->weight.dicts.end()) {
            rotary_dim = (int)(rotary_dim * atof(this->weight.dicts["partial_rotary_factor"].c_str()) + 1e-5);
        }

        if (this->weight.dicts.find("routed_scaling_factor") != this->weight.dicts.end()) {
            routed_scaling_factor = atof(this->weight.dicts["routed_scaling_factor"].c_str());
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
        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(rope_base, rope_factor);
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));

        for (int i = 0; i < block_cnt; i++) {
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                                 WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            );
        }

        for (int i = 0; i < block_cnt; i++) {
            for (int j = -1; j < this->num_experts; j++) {
                std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gate_proj.weight";
                std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".up_proj.weight";
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                if (j == -1) {
                    w1WeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.gate_proj.weight";
                    w3WeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.up_proj.weight";
                    swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight";
                    downWeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.down_proj.weight";
                }
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
                );

                if (j != -1 || !GetCudaSharedExpert()) {
                    this->specialWeights[swigluWeightName] = "linearSwiglu";
                    this->specialWeights[downWeightName] = "linearColumn";
                }
                
                this->moeLinears.insert(w1WeightName);
                this->moeLinears.insert(w3WeightName);
                this->moeLinears.insert(downWeightName);
            }
        }
    }

    extern float yarn_find_correction_dim(int num_rotations, int dim, float base, int max_position_embeddings);
    extern void yarn_find_correction_range(int low_rot, int high_rot, int dim, float base, int max_position_embeddings, int &low, int &high);
    extern float yarn_get_mscale(float scale, float mscale);
    extern std::vector <float> yarn_linear_ramp_mask(float min, float max, int dim);

    std::pair<std::vector<float>, std::vector<float>> GptOssModel::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
        int dim = rotary_dim;
        std::vector <float> freqExtra, freqInter;
        for (int i = 0; i < dim; i += 2) {
            freqExtra.push_back(1.0 / pow(base, (float)i / rotary_dim));
            freqInter.push_back(1.0 / (rope_factor * pow(base, (float)i / rotary_dim)));
        }

        int low, high;
        yarn_find_correction_range (
            rope_scaling_beta_fast,
            rope_scaling_beta_slow,
            dim,
            base,
            rope_scaling_original_max_position_embeddings,
            low, high
        );

        std::vector <float> invFreqMask = yarn_linear_ramp_mask(low, high, dim / 2);
        for (int i = 0; i < invFreqMask.size(); i++) {
            invFreqMask[i] = 1.0 - invFreqMask[i];
        }
        
        std::vector <float> invFreq;
        for (int i = 0; i < freqInter.size(); i++) {
            invFreq.push_back(freqInter[i] * (1.0 - invFreqMask[i]) + freqExtra[i] * invFreqMask[i]);
        }

        // float _mscale = yarn_get_mscale(rope_factor, rope_scaling_mscale) / yarn_get_mscale(rope_factor, rope_scaling_mscale_all_dim);
        float _mscale = yarn_get_mscale(rope_factor, 1.0f);
        int positions = std::max(max_positions, seqLen);
        sin.resize(positions);
        cos.resize(positions);

        for (int i = 0; i < positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size() * 2; j++) {
                sin[i][j] = ::sin((float)i * invFreq[j % invFreq.size()]) * _mscale;
                cos[i][j] = ::cos((float)i * invFreq[j % invFreq.size()]) * _mscale;
            }
        }

        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
        }
        for (int i = 0; i < cos.size(); i++) {
            fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    int GptOssModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> GptOssModel::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv, curInput, curOutput;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3, routerLogits, gate, attenPart, moePart, moeFinal, sharedGate;
        Data tempInput, tempOutput;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;
        Data ww1, ww2, ww3, moeFinal2;

        std::string scoring_func = "sigmoid";
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);

        int seqlen = hiddenStates.dims[1];
        if (weights.size() == 0) {
            weights.resize(block_cnt);
            biass.resize(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight"]);
                weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.down_proj.weight"]);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                for (int j = 0; j < this->num_experts; j++) {
                    weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight"]);
                    weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight"]);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                }
            }
        }
        
        Data attenInputTemp;
        bool cudaSe = GetCudaSharedExpert();
        
        for (int i = 0; i < block_cnt; i++) {
            bool canRunExSilu = CanRunLinearEx(LinearExType::ExSilu);
            bool canRunExSwiglu = CanRunLinearEx(LinearExType::ExSwiglu);

            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            ToDataType(attenInput, attenInputTemp, this->dataType);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string sinkWeightName = "model.layers." + std::to_string(i) + ".self_attn.sinks";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];

            if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()
                && CanRunMergeAttention()
                && false) {
                std::vector <Data*> keys, values, masks;
                keys.push_back(&pastKeyValues[i].first);
                values.push_back(&pastKeyValues[i].second);
                masks.push_back((Data*)&attentionMask);
                MergeAttention (
                    attenInputTemp, 
                    weight[mergeQkvWeightName], weight[mergeQkvBiasName], 
                    weight[oWeightName], weight[oBiasName],
                    qkv, q, k, v, curInput, curOutput,
                    num_attention_heads, num_key_value_heads, head_dim, rotary_dim, 1.0 / sqrt(head_dim),
                    positionIds, *sinDataPtr, *cosDataPtr, 
                    keys, values, masks, w1
                );
                AddTo(hiddenStates, w1);
            } else {
                if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                    Linear(attenInputTemp, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                    int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                    int qdim = per * (num_attention_heads / num_key_value_heads);

                    Split(qkv, -1, 0, qdim, q);
                    Split(qkv, -1, qdim, qdim + per, k);
                    Split(qkv, -1, qdim + per, qdim + per * 2, v);
                } else {
                    Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                    Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                    Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                    Linear(attenInputTemp, weight[qWeightName], qBias, q);
                    Linear(attenInputTemp, weight[kWeightName], kBias, k);
                    Linear(attenInputTemp, weight[vWeightName], vBias, v);
                }

                std::vector <int> qkvSize = {bsz, seqlen, -1, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(k.dataDevice);
                    pastValue.ToDevice(k.dataDevice);
                }
                int targetSeqLength = (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqlen : seqlen;
                if (i == 0 && targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                    float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                    float newbase = rope_base * scale;
                    std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                    sinDataPtr = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                    cosDataPtr = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
                }

                fastllm::LlamaRotatePosition2D(q, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
                fastllm::LlamaRotatePosition2D(k, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});

                qkvSize = {-1, seqlen, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                int unitLen = 64;
    #ifdef USE_CUDA
                unitLen = 128;
    #endif
                while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                    || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                    || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }

                CatDirect(pastKey, k, 1);
                CatDirect(pastValue, v, 1);

                // 1.2 Attention
                if (false) {
                    Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                } else {
                    MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim), q.dims[0] / pastKey.dims[0]);
                    attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
                    AttentionMask(attenWeights, attentionMask, -10000);

                    Data curSinks0, curSinks, combinedLogits;
                    weight[sinkWeightName].Reshape({1, -1, 1, 1});
                    Repeat(weight[sinkWeightName], 0, bsz, curSinks0);
                    Repeat(curSinks0, 2, seqlen, curSinks);

                    Cat(attenWeights, curSinks, -1, combinedLogits);
                    Softmax(combinedLogits, combinedLogits, -1);
                    Split(combinedLogits, -1, 0, combinedLogits.dims.back() - 1, attenWeights);
                    MatMul(attenWeights, pastValue, qkv, 1.f, attenWeights.dims[1] / pastValue.dims[0]);
                    qkv.Reshape({qkv.dims[1], qkv.dims[2], qkv.dims[3]});
                }

                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});

                Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
                Linear(qkv, weight[oWeightName], oBias, attenInput);
                ToDataType(attenInput, DataType::FLOAT32);
                AddTo(hiddenStates, attenInput);
            }

            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);           

            {
                std::string gateUpWeightName = "model.layers." + std::to_string(i) + ".mlp.experts.gate_up_proj";
                std::string gateUpBiasName = "model.layers." + std::to_string(i) + ".mlp.experts.gate_up_proj_bias";
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.experts.down_proj";
                std::string downBiasName = "model.layers." + std::to_string(i) + ".mlp.experts.down_proj_bias";

                if (weight.weight.find(gateUpWeightName) != weight.weight.end()) {
                    int num_experts = weight[gateUpWeightName].dims[0];
                    for (int id = 0; id < num_experts; id++) {
                        std::string curGateUpWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(id) + ".gateup_proj.weight";
                        std::string curGateUpBiasName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(id) + ".gateup_proj.bias";
                        std::string curDownWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(id) + ".down_proj.weight";
                        std::string curDownBiasName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(id) + ".down_proj.bias";

                        Split(weight[gateUpWeightName], 0, id, id + 1, weight[curGateUpWeightName]);
                        Split(weight[gateUpBiasName], 0, id, id + 1, weight[curGateUpBiasName]);
                        Split(weight[downWeightName], 0, id, id + 1, weight[curDownWeightName]);
                        Split(weight[downBiasName], 0, id, id + 1, weight[curDownBiasName]);

                        weight[curGateUpWeightName].Resize({weight[curGateUpWeightName].dims[1], weight[curGateUpWeightName].dims[2]});
                        PermuteSelf(weight[curGateUpWeightName], {1, 0});
                        weight[curGateUpBiasName].Resize({weight[curGateUpBiasName].dims[1]});
                        weight[curDownWeightName].Resize({weight[curDownWeightName].dims[1], weight[curDownWeightName].dims[2]});
                        PermuteSelf(weight[curDownWeightName], {1, 0});
                        weight[curDownBiasName].Resize({weight[curDownBiasName].dims[1]});
                    }

                    weight.weight.erase(gateUpWeightName);
                    weight.weight.erase(gateUpBiasName);
                    weight.weight.erase(downWeightName);
                    weight.weight.erase(downBiasName);
                }

                // 这里是moe mlp
                std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.router.weight";
                std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.router.bias";

                int batch = attenInput.dims[0], len = attenInput.dims[1];
                attenInput.Reshape({batch * len, attenInput.dims[2]});
                Linear(attenInput, weight[gateWeightName], weight[gateBiasName], routerLogits);
                ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);

                /* moe */ {
                    ToDataType(routerLogits, DataType::FLOAT32);
                    routerLogits.ToDevice(DataDevice::CPU);
                    float *cpuRouterLogits = (float*)routerLogits.cpuData;
                    int m = routerLogits.dims.back();

                    moeFinal = Data();
                    moeFinal.Resize({0, attenInput.dims[1]});
                    moeFinal.Expansion(attenInput.dims);

                    for (int b = 0; b < batch * len; b++) {
                        float *cur = cpuRouterLogits + b * m;
                        std::vector <std::pair <float, int> > v; // (value, idx)
                        for (int i = 0; i < m; i++) {
                            v.push_back(std::make_pair(-cur[i], i));
                        }
                        sort(v.begin(), v.end());
                        Data *currentData = &attenInput;
                        if (batch * len != 1) {
                            Split(attenInput, 0, b, b + 1, attenPart);
                            currentData = &attenPart;
                        }
                        moePart.Resize(currentData->dims);
                        moePart.Allocate(0.0f);

                        float sum = 0.0, maxv = -1e9;
                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            maxv = std::max(maxv, cur[v[j].second]);
                        }
                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            sum += exp(cur[v[j].second] - maxv);
                        }
                        
                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            int idx = v[j].second;
                            float value = exp(cur[idx] - maxv) / sum;
                            Linear(*currentData, 
                                weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight"], 
                                weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.bias"], 
                                w3);
                            SwigluGptOss(w3, w1);
                            Linear(w1, 
                                weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".down_proj.weight"], 
                                weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".down_proj.bias"], 
                                w2);
                            AddTo(moePart, w2, value);
                        }

                        CatDirect(moeFinal, moePart, 0);
                    }
                    moeFinal.expansionDims.clear();
                }

                moeFinal.Reshape(hiddenStates.dims);
                Data tempMoeFinal;
                tempMoeFinal.CopyFrom(moeFinal);
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                AddTo(hiddenStates, tempMoeFinal);
            }
        }

        Data logits, topk;
        Data tempHiddenStates;
        Data *lastHiddenStates;
        if (maxLen > 1) {
            Split(hiddenStates, 1, maxLen - 1, maxLen, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        } else {
            lastHiddenStates = &hiddenStates;
        }

        if (generationConfig.top_k <= 1) {
            // 禁用simple greedy
            ((GenerationConfig*)&generationConfig)->top_k = 5;
            ((GenerationConfig*)&generationConfig)->top_p = 0.95;
            if (fabs(generationConfig.temperature - 1.0f) < 1e-9) {
                ((GenerationConfig*)&generationConfig)->temperature = 0.6;
            }
        }

        std::vector <int> lastRet;
        {
            auto &hiddenStates = *lastHiddenStates;
            RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);

            if (generationConfig.output_logits && retLogits != nullptr) {
                int size = logits.dims.back();
                logits.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    (*retLogits)[b]->resize(size);
                    memcpy((float*)(*retLogits)[b]->data(), 
                        ((float*)logits.cpuData) + ((b + 1) * logits.dims[1] - 1) * size, 
                        size * logits.unitSize);
                }
            }

            if (generationConfig.IsSimpleGreedy()) {
                TopK(logits, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    int base = b;
                    lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
                }
            } else if (generationConfig.top_k <= 50 && !generationConfig.output_logits) {
                if ((generationConfig.repeat_penalty - 1.0f) > 1e-9) {
                    int maxTokenSetSize = 0;
                    for (int b = 0; b < batch; b++) {
                        maxTokenSetSize = std::max(maxTokenSetSize, (int)lastTokens.units[b].tokenSet.size());
                    }
                    std::vector <float> penaltyData = std::vector <float> (batch * maxTokenSetSize, -100.0f);
                    std::vector <float> penaltyScaleData = std::vector <float> (batch, 1.0f);
                    for (int b = 0; b < batch; b++) {
                        int curId = 0;
                        for (int i : lastTokens.units[b].tokenSet) {
                            penaltyData[b * maxTokenSetSize + curId] = i;
                            curId++;
                        }
                        penaltyScaleData[b] = generationConfig.repeat_penalty;
                    }
                    Data penalty, penaltyScale;
                    penalty.CopyFrom(Data(DataType::FLOAT32, {batch, maxTokenSetSize}, penaltyData));
                    penaltyScale.CopyFrom(Data(DataType::FLOAT32, {batch}, penaltyScaleData));
                    RepeatPenalty(logits, penalty, penaltyScale);
                }

                Data topk;
                TopK(logits, topk, generationConfig.top_k);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    lastRet.push_back(LLMSamplingOnly(topk, b, generationConfig));
                }
            } else {
                for (int b = 0; b < batch; b++) {
                    int base = b * logits.dims[1] + logits.dims[1] - 1;
                    lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
                }
            }
        }
        if (sinDataPtr != &sinData)
            delete sinDataPtr;
        if (cosDataPtr != &cosData)
            delete cosDataPtr;

        return lastRet;
    }

    std::vector <int> GptOssModel::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        int seqLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, curAttenOutput;
        Data attenLastOutput;
        Data w1, w2, w3, routerLogits, gate, attenPart, moePart, moeFinal, sharedGate;
        Data tempInput, tempOutput;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;
        std::vector <Data> curContextLayer;
        curContextLayer.resize(batch);
        std::vector <Data> curKs, curVs, curQs;
        curKs.resize(batch);
        curVs.resize(batch);
        curQs.resize(batch);
        std::vector <Data*> pointersK, pointersV, pointersQ;
        pointersK.resize(batch);
        pointersV.resize(batch);
        pointersQ.resize(batch);
        std::vector <Data*> keys, values, qs, attns, masks, contexts;
        keys.resize(batch);
        values.resize(batch);
        qs.resize(batch);
        attns.resize(batch);
        masks.resize(batch);
        contexts.resize(batch);
        Data allPositionIds;
        Data ww1, ww2, ww3, moeFinal2;
        std::string scoring_func = "sigmoid";

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
            allPositionIds.CopyFrom(*(Data*)positionIds[0]);
            allPositionIds.Expansion({1, seqLen});
            for (int i = 1; i < batch; i++) {
                CatDirect(allPositionIds, *(Data*)positionIds[i], 1);
            }
        }

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);

        int seqlen = hiddenStates.dims[1];
        Data attenInputTemp;
        bool cudaSe = GetCudaSharedExpert();

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            
            ToDataType(attenInput, attenInputTemp, this->dataType);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                Linear(attenInputTemp, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                int qdim = per * (num_attention_heads / num_key_value_heads);

                Split(qkv, -1, 0, qdim, q);
                Split(qkv, -1, qdim, qdim + per, k);
                Split(qkv, -1, qdim + per, qdim + per * 2, v);
            } else {
                Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                Linear(attenInputTemp, weight[qWeightName], qBias, q);
                Linear(attenInputTemp, weight[kWeightName], kBias, k);
                Linear(attenInputTemp, weight[vWeightName], vBias, v);
            }

            q.Reshape({q.dims[0], q.dims[1], -1, head_dim});
            k.Reshape({k.dims[0], k.dims[1], -1, head_dim});
            v.Reshape({v.dims[0], v.dims[1], -1, head_dim});

            int cacheOuter = k.dims[2], cacheInner = k.dims[3];
            int targetSeqLength = 0;
            for (int b = 0; b < batch; b++) {
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                    if (GetKVCacheInCPU()) {
                        pastKey.lockInCPU = true;
                        pastValue.lockInCPU = true;
                    } else {
                        pastKey.ToDevice(k.dataDevice);
                        pastValue.ToDevice(k.dataDevice);
                    }
                    targetSeqLength = std::max(targetSeqLength, (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqLens[b] : seqLens[b]);
            }

            if (targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                    float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                    float newbase = rope_base * scale;
                    std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                    sinDataPtr = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                    cosDataPtr = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
            }

            for (int b = 0; b < batch; b++) {
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                int curLen = seqLens[b];
                
                int unitLen = 64;
#ifdef USE_CUDA
                unitLen = 128;
#endif
                while ((pastKey.dims.size() == 0 &&
                        (pastKey.expansionDims.size() == 0 || curLen > pastKey.expansionDims[1]))
                       || (pastKey.dims.size() > 0 && pastKey.dims[1] + curLen > pastKey.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector<int> {cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 &&
                        (pastValue.expansionDims.size() == 0 || curLen > pastValue.expansionDims[1]))
                       || (pastValue.dims.size() > 0 && pastValue.dims[1] + curLen > pastValue.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector<int>{cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
            }

            {
                int dim = q.dims.back();
                Data qRot, qPass, kRot, kPass;
                Split(q, -1, 0, dim / 2, qRot);
                Split(q, -1, dim / 2, dim, qPass);
                Split(k, -1, 0, dim / 2, kRot);
                Split(k, -1, dim / 2, dim, kPass);

                fastllm::LlamaRotatePosition2D(qRot, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
                fastllm::LlamaRotatePosition2D(kRot, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);

                Cat(qRot, qPass, -1, q);
                Cat(kRot, kPass, -1, k);
            }
            

            Data attenOutput = Data(this->dataType);
            int total = 0;

            if (false) {
                
            } else {
                if (all1 && batch > 1) {
                    q.Reshape({-1, q.dims[2], q.dims[3]});
                    k.Reshape({-1, k.dims[2], k.dims[3]});
                    v.Reshape({-1, v.dims[2], v.dims[3]});

                    std::vector <int> qdims = {q.dims[1], 1, q.dims[2]};
                    std::vector <uint64_t> qstrides = {(uint64_t)q.dims[2], (uint64_t)q.dims[2], 1};
                    std::vector <int> kdims = {k.dims[1], 1, k.dims[2]};
                    std::vector <uint64_t> kstrides = {(uint64_t)k.dims[2], (uint64_t)k.dims[2], 1};
                    std::vector <int> vdims = {v.dims[1], 1, v.dims[2]};
                    std::vector <uint64_t> vstrides = {(uint64_t)v.dims[2], (uint64_t)v.dims[2], 1};
                    for (int b = 0; b < batch; b++) {
                        curQs[b].dims = qdims;
                        curQs[b].strides = qstrides;
                        curQs[b].FakeFrom(q, b * q.strides[0] * q.unitSize);
                        curKs[b].dims = kdims;
                        curKs[b].strides = kstrides;
                        curKs[b].FakeFrom(k, b * k.strides[0] * k.unitSize);
                        curVs[b].dims = vdims;
                        curVs[b].strides = vstrides;
                        curVs[b].FakeFrom(v, b * v.strides[0] * v.unitSize);
                    }

                    total = batch;
                } else {
                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});

                    std::vector<int> qkvSize = {-1, seqlen, head_dim};
                    q.Reshape(qkvSize);
                    k.Reshape(qkvSize);
                    v.Reshape(qkvSize);

                    for (int b = 0; b < batch; b++) {
                        Split(k, 1, total, total + seqLens[b], curKs[b]);
                        Split(v, 1, total, total + seqLens[b], curVs[b]);
                        Split(q, 1, total, total + seqLens[b], curQs[b]);
                        total += seqLens[b];
                    }
                }

                for (int b = 0; b < batch; b++) {
                    keys[b] = (pastKeyValues[b * block_cnt + i].first);
                    values[b] = (pastKeyValues[b * block_cnt + i].second);
                    pointersK[b] = (&curKs[b]);
                    pointersV[b] = (&curVs[b]);
                }
                CatDirectBatch(keys, pointersK, 1);
                CatDirectBatch(values, pointersV, 1);
            }

            if (all1 && batch > 1) {
                attenOutput.ToDevice(q.dataDevice);
                attenOutput.Resize({1, batch, embed_dim});
                attenOutput.Allocate();
                for (int b = 0; b < batch; b++) {
                    qs[b] = (&curQs[b]);
                    keys[b] = (pastKeyValues[b * block_cnt + i].first);
                    values[b] = (pastKeyValues[b * block_cnt + i].second);
                    masks[b] = attentionMask[b];
                    curContextLayer[b].FakeFrom(attenOutput, b * embed_dim * attenOutput.unitSize);
                    contexts[b] = (&curContextLayer[b]);
                }
                AttentionBatch(qs, keys, values, masks, contexts, qs[0]->dims[0] / values[0]->dims[0], 1.0 / scale_attn, 1);
            } else {
                attenOutput.ToDevice(curQs[0].dataDevice);
                attenOutput.Resize({1, total, embed_dim});
                attenOutput.Allocate();
                int curLen = 0;
                for (int b = 0; b < batch; b++) {
                    auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                    curAttenOutput.FakeFrom(attenOutput, curLen * embed_dim * attenOutput.unitSize);
                    curLen += seqLens[b];

                    // 1.2 Attention
                    if (attentionMask[b] == nullptr) {
                        Attention(q, pastKey, pastValue, Data(), curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                    } else {
                        Attention(q, pastKey, pastValue, *attentionMask[b], curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                    }
                    PermuteSelf(curAttenOutput, {1, 0, 2});
                }
            }

            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(attenOutput, weight[oWeightName], oBias, attenLastOutput);
            ToDataType(attenLastOutput, DataType::FLOAT32);
            AddTo(hiddenStates, attenLastOutput);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);
            
            // 2. moe mlp
            if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight") != weight.weight.end()) {
                if (CanRunLinearEx(LinearExType::ExSilu)) {
                    LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1, LinearExType::ExSilu);
                } else {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
                    Silu(w1, w1);
                }
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
                MulTo(w1, w3);
                Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
                AddTo(hiddenStates, w2);
            } else {
                // 这里是moe mlp
                std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate.weight";
                std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";

                int batch = attenInput.dims[0], len = attenInput.dims[1];
                attenInput.Reshape({batch * len, attenInput.dims[2]});

                if (cudaSe) {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight"], Data(), ww3);
                    Swiglu(ww3, ww1);
                    Linear(ww1, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.down_proj.weight"], Data(), moeFinal2);
                    weights[i][0] = weights[i][1] = nullptr;
                }

                Linear(attenInput, weight[gateWeightName], Data(), routerLogits);

                bool needNorm = false;
                if (scoring_func == "sigmoid") {
                    Sigmoid(routerLogits, routerLogits);
                    needNorm = true;
                } else {
                    Softmax(routerLogits, routerLogits, -1);
                }

                ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
                if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.experts.0.gateup_proj.weight") != weight.weight.end() 
                    && CanRunMergeMOE(attenInput, biass[i])) {
                    MergeMOE (
                        attenInput, routerLogits, weight[gateBiasName],
                        weights[i], biass[i],
                        w1, w2, w3, tempInput, tempOutput, 
                        this->routed_scaling_factor, 1.0f,
                        this->num_experts_per_tok, needNorm,
                        moeFinal
                    );
                } else {
                    Data &bias = weight[gateBiasName];                  
                    ToDataType(routerLogits, DataType::FLOAT32);
                    routerLogits.ToDevice(DataDevice::CPU);
                    float *cpuRouterLogits = (float*)routerLogits.cpuData;
                    int m = routerLogits.dims.back();

                    moeFinal = Data();
                    moeFinal.Resize({0, attenInput.dims[1]});
                    moeFinal.Expansion(attenInput.dims);

                    for (int b = 0; b < batch * len; b++) {
                        float *cur = cpuRouterLogits + b * m;
                        std::vector <std::pair <float, int> > v; // (value, idx)
                        for (int i = 0; i < m; i++) {
                            v.push_back(std::make_pair(-cur[i], i));
                        }
                        if (bias.dims.size() > 0) {
                            ToDataType(bias, DataType::FLOAT32);
                            bias.ToDevice(DataDevice::CPU);
                            float *cpuBias = (float*)bias.cpuData;
                            for (int i = 0; i < m; i++) {
                                v[i].first -= cpuBias[i];
                            }
                        }

                        sort(v.begin(), v.end());
                        Data *currentData = &attenInput;
                        if (batch * len != 1) {
                            Split(attenInput, 0, b, b + 1, attenPart);
                            currentData = &attenPart;
                        }
                        moePart.Resize(currentData->dims);
                        moePart.Allocate(0.0f);

                        float sum = 0.0;
                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            float value = cur[v[j].second];
                            sum += value;
                        }
                        if (!needNorm) {
                            sum = 1.0;
                        }

                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            int idx = v[j].second;
                            float value = cur[idx];

                            value /= sum;
                            value *= routed_scaling_factor;
                            if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight") != weight.weight.end()) {
                                if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                                    LinearEx(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight"], Data(), w1, LinearExType::ExSwiglu);
                                } else {
                                    Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight"], Data(), w3);
                                    Swiglu(w3, w1);
                                }
                            } else {
                                if (CanRunLinearEx(LinearExType::ExSilu)) {
                                    LinearEx(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gate_proj.weight"], Data(), w1, LinearExType::ExSilu);
                                } else {
                                    Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gate_proj.weight"], Data(), w1);
                                    Silu(w1, w1);
                                }
                                Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".up_proj.weight"], Data(), w3);
                                MulTo(w1, w3);
                            }
                            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".down_proj.weight"], Data(), w2);
                            AddTo(moePart, w2, value);
                        }

                        if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight") != weight.weight.end()) {
                            if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                                LinearEx(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight"], Data(), w1, LinearExType::ExSwiglu);
                            } else {
                                Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight"], Data(), w3);
                                Swiglu(w3, w1);
                            }
                        } else {
                            if (CanRunLinearEx(LinearExType::ExSilu)) {
                                LinearEx(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gate_proj.weight"], Data(), w1, LinearExType::ExSilu);
                            } else {
                                Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gate_proj.weight"], Data(), w1);
                                Silu(w1, w1);
                            }
                            Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.up_proj.weight"], Data(), w3);
                            MulTo(w1, w3);
                        }
                        Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.down_proj.weight"], Data(), w2);
                        AddTo(moePart, w2);

                        CatDirect(moeFinal, moePart, 0);
                    }
                    moeFinal.expansionDims.clear();
                }

                moeFinal.Reshape(hiddenStates.dims);

                Data tempMoeFinal;
                tempMoeFinal.CopyFrom(moeFinal);
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                AddTo(hiddenStates, tempMoeFinal);

                if (cudaSe) {
                    moeFinal2.Reshape(hiddenStates.dims);
                    AddTo(hiddenStates, moeFinal2);
                }
            }
        }

        Data logits, curLogit;
        RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        std::vector <int> lastRet;
        int total = 0;

        for (int b = 0; b < batch; b++) {
            if (generationConfigs[b].top_k <= 1) {
                // 禁用simple greedy
                ((GenerationConfig*)&generationConfigs[b])->top_k = 5;
                ((GenerationConfig*)&generationConfigs[b])->top_p = 0.95;
                if (fabs(generationConfigs[b].temperature - 1.0f) < 1e-9) {
                    ((GenerationConfig*)&generationConfigs[b])->temperature = 0.6;
                }
            }
        }
        
        for (int b = 0; b < batch; b++) {
            Split(logits, 1, total + seqLens[b] - 1, total + seqLens[b], curLogit);
            if (generationConfigs[b].output_logits && retLogits != nullptr && (*retLogits)[b] != nullptr) {
                curLogit.ToDevice(DataDevice::CPU);
                (*retLogits)[b]->resize(curLogit.Count(0));
                memcpy((float*)(*retLogits)[b]->data(), (float*)curLogit.cpuData, curLogit.GetBytes());
            }
            if (generationConfigs[b].IsSimpleGreedy()) {
                Data topk;
                TopK(curLogit, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                lastRet.push_back((int) (((float *) topk.cpuData)[0] + 1e-3));
            } else {
                lastRet.push_back(LLMSampling(curLogit, 0, generationConfigs[b], lastTokens.units[b]));
            }
            total += seqLens[b];
        }

        return lastRet;
    }

    bool GptOssModel::NeedAttentionMask(int qlen, int klen) {
        if (((qlen == 1) || (qlen >= 1024))) {
            return false;
        }
        return true;
    }

    void GptOssModel::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            std::vector <int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
                seqLens[i] = (int)inputTokens[i].size();
            }

            std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = tokens.size(), base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                }
                for (int j = 0; j < len; j++) {
                    vpids[i * maxLen + base + j] = j;
                }

                std::fill(vmask.data() + i * maxLen * maxLen,
                        vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                for (int j = maxLen - len; j < maxLen; j++) {
                    std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                            vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                }
                for (int j = 0; j < len; j++) {
                    for (int k = j + 1; k < len; k++) {
                        vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                    }
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, vpids));
        } else {
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen, maxLen);
                pids[i] = promptLen + index - 1;
            }
            maxLen += index;
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int curLen = params[i].find("promptLen")->second + index;
                for (int j = 0; j < maxLen - curLen; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
        }
    }

    std::string GptOssModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string GptOssModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void GptOssModel::WarmUp() {
return;
        printf("Warmup...\n");
        int oldTopk = this->num_experts_per_tok;
        this->num_experts_per_tok = this->num_experts;

        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        this->num_experts_per_tok = oldTopk;
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
    }
}
