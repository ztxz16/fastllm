//
// Created by huangyuyang on 9/12/25.
//

#include "utils.h"

#include "qwen3_next.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    extern std::vector <float> GetInterLeavePowerOf2(int n);
    extern std::vector <float> GetInterleave(int n);

    Qwen3NextModel::Qwen3NextModel() {
        this->canDoBatchForward = false;
        this->model_type = "qwen3_next";
        this->model_struct = "qwen3_next";

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
            "model.layers.*.mlp.*.weight",
            "model.layers.*.linear_attn.in_proj_ba.weight",
            "model.layers.*.linear_attn.in_proj_qkvz.weight",
            "model.layers.*.linear_attn.out_proj.weight",
            "mtp.layers.*.gate_proj.weight", "mtp.layers.*.up_proj.weight", "mtp.layers.*.down_proj.weight",
            "mtp.layers.*.self_attn.q_proj.weight", "mtp.layers.*.self_attn.k_proj.weight", "mtp.layers.*.self_attn.v_proj.weight", "mtp.layers.*.self_attn.o_proj.weight"
        };
    }

    void Qwen3NextModel::InitParams() {
        basellm::InitParams();
        num_experts = atoi(this->weight.dicts["num_experts"].c_str());
        num_experts_per_tok = atoi(this->weight.dicts["num_experts_per_tok"].c_str());
        norm_topk_prob = (this->weight.dicts["norm_topk_prob"] == "true");

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
            // this->weightMergeRules.push_back(
               // WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
               //                  WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            // );
        }

        for (int i = 0; i < block_cnt; i++) {
            for (int j = -1; j < this->num_experts; j++) {
                std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gate_proj.weight";
                std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".up_proj.weight";
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight";

                if (j == -1) {
                    w1WeightName = "model.layers." + std::to_string(i) + ".mlp.shared_expert.gate_proj.weight";
                    w3WeightName = "model.layers." + std::to_string(i) + ".mlp.shared_expert.up_proj.weight";
                    swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.shared_expert.gateup_proj.weight";
                    downWeightName = "model.layers." + std::to_string(i) + ".mlp.shared_expert.down_proj.weight";
                }
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
                );

                if (j != -1) {
                    this->specialWeights[swigluWeightName] = "linearSwiglu";
                    this->specialWeights[downWeightName] = "linearColumn";
                }
                
                if (j != -1) {
                    this->moeLinears.insert(w1WeightName);
                    this->moeLinears.insert(w3WeightName);
                    this->moeLinears.insert(downWeightName);
                }
            }
        }
    }

    std::pair<std::vector<float>, std::vector<float>> Qwen3NextModel::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
        int positions = std::max(max_positions, seqLen);
        sin.resize(positions);
        cos.resize(positions);
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(base, (float)i / rotary_dim));
        }

        float scale = rope_type == RoPEType::LINEAR_SCALE ? factor : 1.0;
        for (int i = 0; i < positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size(); j++) {
                sin[i][j] = ::sin((float)i / scale * invFreq[j]);
                cos[i][j] = ::cos((float)i / scale * invFreq[j]);
            }
        }
        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
            fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    int Qwen3NextModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    void FakePad(Data &input, Data &output, int axis, int dim) {
        if (dim == 0) {
            Mul(input, 1.0f, output);
            return;
        }
        Data temp;
        std::vector <int> dims = input.dims;
        dims[axis] = dim;
        temp.Resize(dims);
        temp.Allocate(0.0f);
        ToDataType(temp, input.dataType);
        Cat(input, temp, axis, output);
    }

    void Add1(Data &input) {
        if (input.dims.size() == 0) {
            return;
        }
        float *v = (float*)input.cpuData;
        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            v[i] += 1.0f;
        }
    }

    std::vector <int> Qwen3NextModel::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv, curInput, curOutput;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3, routerLogits, gate, attenPart, moePart, moeFinal, moeFinal2, sharedGate;
        Data tempInput, tempOutput;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);

        int seqlen = hiddenStates.dims[1];
        if (weights.size() == 0) {
            weights.resize(block_cnt);
            biass.resize(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                weights[i].push_back(nullptr);
                weights[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                for (int j = 0; j < this->num_experts; j++) {
                    weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight"]);
                    weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight"]);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                }
            }

            for (int i = 0; i < block_cnt; i++) {
                Add1(this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"]);
                Add1(this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"]);
                Add1(this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"]);
                Add1(this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"]);
            }
            Add1(weight["model.norm.weight"]);
        }
        
        Data attenInputTemp;
        Data hidden_states_new;
        Data mixed_qkvz, mixed_ba, z, b, a, g;
        Data core_attn_out, core_attn_out_temp;
        Data sharedGateRepeat;

        float inv_scale = pow((float)head_k_dim, -0.5);
        std::vector <float> v_inv_scale = std::vector <float> (head_k_dim, inv_scale);
        Data inv_scale_data = Data(DataType::FLOAT32, {head_k_dim}, v_inv_scale);

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);

            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            ToDataType(attenInput, attenInputTemp, this->dataType);
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } 

            if (weight.weight.find("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight") != weight.weight.end()) {
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

                Data qgate, q, gate, k, v;
                Linear(attenInputTemp, weight[qWeightName], weight[qBiasName], qgate);
                qgate.Reshape({bsz, seqlen, -1, this->head_dim * 2});
                Split(qgate, -1, 0, this->head_dim, q);
                Split(qgate, -1, this->head_dim, qgate.dims.back(), gate);
                gate.Reshape({bsz, seqlen, -1});

                Linear(attenInputTemp, weight[kWeightName], weight[kBiasName], k);
                Linear(attenInputTemp, weight[vWeightName], weight[vBiasName], v);

                k.Reshape({bsz, seqlen, -1, this->head_dim});
                v.Reshape({bsz, seqlen, -1, this->head_dim});

                RMSNorm(q, this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);

                if (false) {
                    int dim = cosDataPtr->dims.back();
                    Data qRot, qPass, kRot, kPass;
                    Split(q, -1, 0, dim, qRot);
                    Split(q, -1, dim, q.dims.back(), qPass);
                    Split(k, -1, 0, dim, kRot);
                    Split(k, -1, dim, k.dims.back(), kPass);
                    fastllm::LlamaRotatePosition2D(qRot, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
                    fastllm::LlamaRotatePosition2D(kRot, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);

                    Cat(qRot, qPass, -1, q);
                    Cat(kRot, kPass, -1, k);
                } else {
                    int dim = cosDataPtr->dims.back();
                    fastllm::LlamaRotatePosition2DPart(q, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim, dim);
                    fastllm::LlamaRotatePosition2DPart(k, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim, dim);
                }

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});

                std::vector <int> qkvSize = {-1, seqlen, head_dim};
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
                Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});

                Sigmoid(gate, gate);
                MulTo(qkv, gate);
                
                Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
                Linear(qkv, weight[oWeightName], oBias, attenInput);
            } else {
                Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
                pastKey.isLinearAttention = pastValue.isLinearAttention = true;
                std::string qkvzWeightName = "model.layers." + std::to_string(i) + ".linear_attn.in_proj_qkvz.weight";
                std::string qkvzBiasName = "model.layers." + std::to_string(i) + ".linear_attn.in_proj_qkvz.bias";
                std::string baWeightName = "model.layers." + std::to_string(i) + ".linear_attn.in_proj_ba.weight";
                std::string baBiasName = "model.layers." + std::to_string(i) + ".linear_attn.in_proj_ba.bias";
                std::string conv1dWeightName = "model.layers." + std::to_string(i) + ".linear_attn.conv1d.weight";
                std::string conv1dBiasName = "model.layers." + std::to_string(i) + ".linear_attn.conv1d.bias";
                std::string aLogName = "model.layers." + std::to_string(i) + ".linear_attn.A_log";
                std::string dtBiasName = "model.layers." + std::to_string(i) + ".linear_attn.dt_bias";

                Linear(attenInputTemp, weight[qkvzWeightName], Data(), mixed_qkvz);
                Linear(attenInputTemp, weight[baWeightName], Data(), mixed_ba);

                mixed_qkvz.Resize({bsz, seqlen, num_k_heads, 2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads});
                mixed_ba.Resize({bsz, seqlen, num_k_heads, 2 * num_v_heads / num_k_heads});

                std::vector <int> qkvz = {head_k_dim, head_k_dim, num_v_heads / num_k_heads * head_v_dim, num_v_heads / num_k_heads * head_v_dim};
                for (int i = 1; i < qkvz.size(); i++) {
                    qkvz[i] += qkvz[i - 1];
                }
                int per = mixed_qkvz.dims.back() / qkvz.back();
                Split(mixed_qkvz, -1, 0, qkvz[0] * per, q);
                Split(mixed_qkvz, -1, qkvz[0] * per, qkvz[1] * per, k);
                Split(mixed_qkvz, -1, qkvz[1] * per, qkvz[2] * per, v);
                Split(mixed_qkvz, -1, qkvz[2] * per, qkvz[3] * per, z);

                per = mixed_ba.dims.back() / 2;
                Split(mixed_ba, -1, 0, per, b);
                Split(mixed_ba, -1, per, per + per, a);

                v.Reshape({v.dims[0], v.dims[1], -1});
                z.Reshape({z.dims[0], z.dims[1], -1, head_v_dim});
                b.Reshape({b.dims[0], b.dims[1], -1});
                a.Reshape({a.dims[0], a.dims[1], -1});

                q.Reshape({q.dims[0], q.dims[1], -1});
                k.Reshape({k.dims[0], k.dims[1], -1});
                v.Reshape({v.dims[0], v.dims[1], -1});

                Data qk, qkv, conv;
                Cat(q, k, -1, qk);
                Cat(qk, v, -1, qkv);
                PermuteSelf(qkv, {0, 2, 1});

                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    Data hidden_states_new;
                    Cat(pastKey, qkv, -1, hidden_states_new);
                    Split(hidden_states_new, -1, hidden_states_new.dims.back() - 4, hidden_states_new.dims.back(), pastKey);
                    Conv1DPerChannel(
                        hidden_states_new, weight[conv1dWeightName], weight[conv1dBiasName], 
                        hidden_states_new.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 0, 
                        conv
                    );
                    Split(conv, -1, conv.dims.back() - 1, conv.dims.back(), qkv);
                    Silu(qkv, qkv);
                } else {
                    if (qkv.dims.back() >= 4) {
                        Split(qkv, -1, qkv.dims.back() - 4, qkv.dims.back(), pastKey);
                        // PermuteSelf(pastKey, {0, 2, 1});
                        pastKey.expansionDims = pastKey.dims;
                    } else {
                        ErrorInFastLLM("qkv.dims.back() < 4");
                    }

                    Conv1DPerChannel(
                        qkv, weight[conv1dWeightName], weight[conv1dBiasName], 
                        qkv.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 3, 
                        conv
                    );
                    Split(conv, -1, 0, seqlen, qkv);
                    Silu(qkv, qkv);
                }

                int kd = num_k_heads * head_k_dim, vd = num_v_heads * head_v_dim;
                PermuteSelf(qkv, {0, 2, 1});

                Split(qkv, -1, 0, kd, q);
                Split(qkv, -1, kd, kd + kd, k);
                Split(qkv, -1, kd + kd, kd + kd + vd, v);
                
                q.Reshape({q.dims[0], q.dims[1], -1, head_k_dim});
                k.Reshape({k.dims[0], k.dims[1], -1, head_k_dim});
                v.Reshape({v.dims[0], v.dims[1], -1, head_v_dim});

                if (!(bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0)) {
                    Data qtemp, ktemp;
                    Mul(q, 1.0f, qtemp);
                    Mul(k, 1.0f, ktemp);

                    qtemp.Resize({q.dims[0], q.dims[1], q.dims[2], 1, q.dims[3]});
                    ktemp.Resize({k.dims[0], k.dims[1], k.dims[2], 1, k.dims[3]});

                    Repeat(qtemp, 3, 2, q);
                    Repeat(ktemp, 3, 2, k);

                    q.Reshape({q.dims[0], q.dims[1], -1, q.dims.back()});
                    k.Reshape({k.dims[0], k.dims[1], -1, k.dims.back()});
                }
                
                Sigmoid(b, b); // beta = b.sigmoid()
                MambaSoftplus(a, weight[aLogName], weight[dtBiasName], g); // g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
                Data &last_recurrent_state = pastValue;
                Data core_attn_out, core_attn_out_temp;
                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    // torch_recurrent_gated_delta_rule
                    {
                        // rms norm
                        RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                        RMSNorm(k, inv_scale_data, rms_norm_eps, k);
                    }

                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});
                    PermuteSelf(b, {0, 2, 1});
                    PermuteSelf(g, {0, 2, 1});

                    // batch_size, sequence_length, num_heads, k_head_dim = key.shape, 这里num_heads才是真正的序列长度
                    int key_batch_size = k.dims[0], key_sequence_length = k.dims[1], key_num_heads = k.dims[2], key_k_head_dim = k.dims[3];
                    int v_head_dim = v.dims.back();
                    float scale = 1.0f / pow(q.dims.back(), 0.5);
                    Mul(q, scale, q); // query = query * scale

                    RecurrentGatedDeltaRule (
                        q, k, v, g, b,
                        last_recurrent_state, 
                        core_attn_out
                    ); 
                    PermuteSelf(core_attn_out, {0, 2, 1, 3});
                } else {
                    // torch_chunk_gated_delta_rule
                    {
                        // rms norm
                        float inv_scale = pow((float)head_k_dim, -0.5);
                        std::vector <float> v_inv_scale = std::vector <float> (head_k_dim, inv_scale);
                        Data inv_scale_data = Data(DataType::FLOAT32, {head_k_dim}, v_inv_scale);

                        RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                        RMSNorm(k, inv_scale_data, rms_norm_eps, k);
                    }

                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});
                    PermuteSelf(b, {0, 2, 1});
                    PermuteSelf(g, {0, 2, 1});
                    
                    // batch_size, sequence_length, num_heads, k_head_dim = key.shape, 这里num_heads才是真正的序列长度
                    int key_batch_size = k.dims[0], key_sequence_length = k.dims[1], key_num_heads = k.dims[2], key_k_head_dim = k.dims[3];

                    int chunk_size = 64;
                    int v_head_dim = v.dims.back();
                    int seq = k.dims[2];
                    int pad_size = (chunk_size - seq % chunk_size) % chunk_size;

                    Data qtemp, qq, kk, vv, bb, gg, decayMask;
                    {
                        // pad 
                        FakePad(q, qtemp, 2, pad_size); // query = F.pad(query, (0, 0, 0, pad_size))
                        FakePad(k, kk, 2, pad_size); // key = F.pad(key, (0, 0, 0, pad_size))
                        FakePad(v, vv, 2, pad_size); // value = F.pad(value, (0, 0, 0, pad_size))
                        FakePad(b, bb, 2, pad_size); // beta = F.pad(beta, (0, pad_size))
                        FakePad(g, gg, 2, pad_size); // g = F.pad(g, (0, pad_size))
                    }

                    int tot_heads = seq + pad_size;
                    float scale = 1.0f / pow(qtemp.dims.back(), 0.5);
                    Mul(qtemp, scale, qq); // query = query * scale

                    bb.Resize({bb.dims[0], bb.dims[1], bb.dims[2], 1});
                    Data k_beta, v_beta;
                    Mul(kk, 1.0f, k_beta);
                    Mul(vv, 1.0f, v_beta);
                    MulTo(k_beta, bb);
                    MulTo(v_beta, bb);

                    qq.Reshape({qq.dims[0], qq.dims[1], -1, chunk_size, qq.dims.back()});
                    kk.Reshape({kk.dims[0], kk.dims[1], -1, chunk_size, kk.dims.back()});
                    k_beta.Reshape({k_beta.dims[0], k_beta.dims[1], -1, chunk_size, k_beta.dims.back()});
                    v_beta.Reshape({v_beta.dims[0], v_beta.dims[1], -1, chunk_size, v_beta.dims.back()});
                    gg.Reshape({gg.dims[0], gg.dims[1], -1, chunk_size});

                    CumSumLastDim(gg);
                    MakeDecayMask(gg, decayMask);

                    Data at, attn;
                    MatMulTransB(k_beta, kk, at);
                    Mul(at, -1.0f, attn);
                    MulTo(attn, decayMask);
                    CausalMask(attn, 0, 0.0f);

                    TransferAttn(attn);
                    MatMul(attn, v_beta, vv);
                    
                    Data k_temp, k_cumdecay;                    
                    Exp(gg, g);

                    Mul(k_beta, 1.0f, k_temp);
                    MulTo(k_temp, g);
                    MatMul(attn, k_temp, k_cumdecay);

                    if (last_recurrent_state.dims.size() == 0) {
                        last_recurrent_state.Resize({key_batch_size, key_sequence_length, key_k_head_dim, v_head_dim});
                        last_recurrent_state.Allocate(0.0f);
                    }

                    for (int i = 0; i < tot_heads / chunk_size; i++) {
                        Data q_i, k_i, v_i, decay_mask_i, k_cumdecay_i;
                        Split(qq, 2, i, i + 1, q_i);
                        Split(kk, 2, i, i + 1, k_i);
                        Split(vv, 2, i, i + 1, v_i);

                        q_i.Resize({q_i.dims[0], q_i.dims[1], q_i.dims[3], q_i.dims[4]});
                        k_i.Resize({k_i.dims[0], k_i.dims[1], k_i.dims[3], k_i.dims[4]});
                        v_i.Resize({v_i.dims[0], v_i.dims[1], v_i.dims[3], v_i.dims[4]});
                        
                        Split(decayMask, 2, i, i + 1, decay_mask_i);
                        decay_mask_i.Resize({decay_mask_i.dims[0], decay_mask_i.dims[1], decay_mask_i.dims[3], decay_mask_i.dims[4]});

                        MatMulTransB(q_i, k_i, attn);
                        MulTo(attn, decay_mask_i);
                        CausalMask(attn, 1, 0.0f);

                        Split(k_cumdecay, 2, i, i + 1, k_cumdecay_i);
                        k_cumdecay_i.Resize({k_cumdecay_i.dims[0], k_cumdecay_i.dims[1], k_cumdecay_i.dims[3], k_cumdecay_i.dims[4]});

                        Data v_prime, v_new;
                        MatMul(k_cumdecay_i, last_recurrent_state, v_prime);
                        Mul(v_prime, -1.0f, v_new);
                        AddTo(v_new, v_i);

                        Data attn_inter, g_i, g_i_exp, q_i_temp;
                        Split(gg, 2, i, i + 1, g_i);
                        g_i.Resize({g_i.dims[0], g_i.dims[1], g_i.dims[3], 1});
                        Exp(g_i, g_i_exp);
                        Mul(q_i, 1.0f, q_i_temp);
                        MulTo(q_i_temp, g_i_exp);

                        MatMul(q_i_temp, last_recurrent_state, attn_inter);
                        Data atv;
                        MatMul(attn, v_new, atv);
                        AddTo(atv, attn_inter);
                        atv.Resize({atv.dims[0], atv.dims[1], 1, atv.dims[2], atv.dims[3]});
                        if (i == 0) {
                            Mul(atv, 1.0f, core_attn_out);
                        } else {
                            Mul(core_attn_out, 1.0f, core_attn_out_temp);
                            Cat(core_attn_out_temp, atv, 3, core_attn_out);
                        }

                        g_i.Resize({g_i.dims[0], g_i.dims[1], g_i.dims[2]});
                        Data g_i_last, g_i_last_repeat, g_i_l_temp;
                        Split(g_i, -1, g_i.dims.back() - 1, g_i.dims.back(), g_i_last);
                        Repeat(g_i_last, -1, g_i.dims.back(), g_i_last_repeat);
                        Mul(g_i, -1.0f, g_i_l_temp);
                        AddTo(g_i_l_temp, g_i_last_repeat);
                        Exp(g_i_l_temp, g_i_l_temp);
                        g_i_l_temp.Resize({g_i_l_temp.dims[0], g_i_l_temp.dims[1], g_i_l_temp.dims[2], 1});
                        MulTo(k_i, g_i_l_temp);
                        PermuteSelf(k_i, {0, 1, 3, 2});

                        Data k_i_v_new;
                        MatMul(k_i, v_new, k_i_v_new);

                        Data g_i_exp_last;
                        Split(g_i_exp, 2, g_i_exp.dims[2] - 1, g_i_exp.dims[2], g_i_exp_last);
                        MulTo(last_recurrent_state, g_i_exp_last);
                        AddTo(last_recurrent_state, k_i_v_new);
                    }

                    core_attn_out.Reshape({core_attn_out.dims[0], core_attn_out.dims[1], -1, core_attn_out.dims.back()});
                    Split(core_attn_out, 2, 0, seq, core_attn_out_temp);
                    PermuteSelf(core_attn_out_temp, {0, 2, 1, 3});
                    Mul(core_attn_out_temp, 1.0f, core_attn_out);
                }

                {
                    // z_shape_og = z.shape
                    // core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
                    // z = z.reshape(-1, z.shape[-1])
                    // core_attn_out = self.norm(core_attn_out, z)
                    // core_attn_out = core_attn_out.reshape(z_shape_og)
                    // core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
                    // output = self.out_proj(core_attn_out)
                    
                    std::vector <int> zShape = z.dims;
                    core_attn_out.Reshape({-1, core_attn_out.dims.back()});
                    z.Reshape({-1, z.dims.back()});

                    RMSNorm(core_attn_out, this->weight["model.layers." + std::to_string(i) + ".linear_attn.norm.weight"], rms_norm_eps, core_attn_out);
                    Silu(z, z);
                    MulTo(core_attn_out, z);

                    core_attn_out.Reshape({zShape[0], zShape[1], -1});
                    Linear(core_attn_out, 
                        this->weight["model.layers." + std::to_string(i) + ".linear_attn.out_proj.weight"], 
                        this->weight["model.layers." + std::to_string(i) + ".linear_attn.out_proj.bias"], attenInput);
                }
            }

            ToDataType(attenInput, DataType::FLOAT32);
            AddTo(hiddenStates, attenInput);

            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);           

            // 2. moe mlp
            {
                // 这里是moe mlp
                std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate.weight";
                std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";

                int batch = attenInput.dims[0], len = attenInput.dims[1];
                attenInput.Reshape({batch * len, attenInput.dims[2]});
                Data sharedGate, sharedGateRepeat;
                Linear(attenInput, weight[gateWeightName], Data(), routerLogits);
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.shared_expert.gateup_proj.weight"], Data(), w3);
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.shared_expert_gate.weight"], Data(), sharedGate);
                bool needNorm = true;
                Softmax(routerLogits, routerLogits, -1);

                Swiglu(w3, w1);
                Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.shared_expert.down_proj.weight"], Data(), moeFinal2);
                Sigmoid(sharedGate, sharedGate);
                
                if (sharedGate.Count(0) == 1) {
                    MulTo(moeFinal2, sharedGate);
                } else {
                    MulTo(moeFinal2, sharedGate);
                }
                
                ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
                MergeMOE (
                        attenInput, routerLogits, weight[gateBiasName],
                        weights[i], biass[i],
                        w1, w2, w3, tempInput, tempOutput,
                        this->routed_scaling_factor, 1.0f,
                        this->num_experts_per_tok, needNorm,
                        moeFinal
                );

                moeFinal.Reshape(hiddenStates.dims);
                Data tempMoeFinal;
                tempMoeFinal.CopyFrom(moeFinal);
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                AddTo(hiddenStates, tempMoeFinal);
                moeFinal2.Reshape(hiddenStates.dims);
                AddTo(hiddenStates, moeFinal2);
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

            // logits.Print();
            // exit(0);

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

    std::vector <int> Qwen3NextModel::ForwardBatch(int batch,
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
        Data q, k, v, qkv, qgate;
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
            for (int b = 0; b < batch; b++) {
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                }
            }

            if (weight.weight.find("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight") != weight.weight.end()) {
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
                
                Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                
                Linear(attenInputTemp, weight[qWeightName], qBias, qgate);

                Linear(attenInputTemp, weight[kWeightName], kBias, k);
                Linear(attenInputTemp, weight[vWeightName], vBias, v);

                qgate.Reshape({qgate.dims[0], qgate.dims[1], -1, head_dim * 2});
                k.Reshape({k.dims[0], k.dims[1], -1, head_dim});
                v.Reshape({v.dims[0], v.dims[1], -1, head_dim});

                Split(qgate, -1, 0, this->head_dim, q);
                Split(qgate, -1, this->head_dim, qgate.dims.back(), gate);
                gate.Reshape({gate.dims[0], gate.dims[1], -1});

                RMSNorm(q, this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);

                int cacheOuter = k.dims[2], cacheInner = k.dims[3];
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
                    int dim = cosDataPtr->dims.back();
                    Data qRot, qPass, kRot, kPass;
                    Split(q, -1, 0, dim, qRot);
                    Split(q, -1, dim, q.dims.back(), qPass);
                    Split(k, -1, 0, dim, kRot);
                    Split(k, -1, dim, k.dims.back(), kPass);

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
                    if (true) {
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
                }

                Sigmoid(gate, gate);
                MulTo(attenOutput, gate);

                Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
                Linear(attenOutput, weight[oWeightName], oBias, attenLastOutput);
            } else {

            }

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
                Linear(attenInput, weight[gateWeightName], Data(), routerLogits);

                bool needNorm = true;
                Softmax(routerLogits, routerLogits, -1);

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

    bool Qwen3NextModel::NeedAttentionMask(int qlen, int klen) {
        if (((qlen == 1) || (qlen >= 1024))) {
            return false;
        }
        return true;
    }

    void Qwen3NextModel::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
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

    std::string Qwen3NextModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Qwen3NextModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void Qwen3NextModel::WarmUp() {
        printf("Warmup...\n");
        int oldTopk = this->num_experts_per_tok;
        this->num_experts_per_tok = this->num_experts;

        Data inputIds = Data(DataType::FLOAT32, {1, 4}, {0, 1, 2, 3});
        Data attentionMask = Data(DataType::FLOAT32, {4, 4});
        Data positionIds = Data(DataType::FLOAT32, {1, 4}, {0, 1, 2, 3});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        this->num_experts_per_tok = oldTopk;
        elementsInKVCachePerToken = 0;
        for (int i = 0; i < block_cnt; i++) {
            if (!pastKeyValues[i].first.isLinearAttention) {
                if (this->kvCacheId == 0) {
                    this->kvCacheId = i;
                }

                elementsInKVCachePerToken += 
                    (pastKeyValues[i].first.dims[0] * pastKeyValues[i].first.dims[2] + 
                    pastKeyValues[i].second.dims[0] * pastKeyValues[i].second.dims[2]);
            }
        }
        printf("finish.\n");
    }
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            