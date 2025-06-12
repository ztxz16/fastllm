//
// Created by huangyuyang on 4/29/25.
//

#include "utils.h"

#include "qwen3.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    extern std::vector <float> GetInterLeavePowerOf2(int n);
    extern std::vector <float> GetInterleave(int n);

    Qwen3Model::Qwen3Model() {
        this->model_struct = "llama";
        this->model_type = "qwen3";

        // 默认使用 llama3 的提示词和instruction
        this->pre_prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>";
        this->user_role="<|start_header_id|>user<|end_header_id|>\n";
        this->bot_role="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
        this->history_sep="<|eot_id|>\n";

        block_cnt = 32;
        rotary_dim = 128;

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight", "model.layers.*.mlp.down_proj.weight", "model.layers.*.mlp.up_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",  "model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.gateup_proj.weight",
            "model.layers.*.self_attn.o_proj.weight", "model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight", "model.layers.*.self_attn.mergeqkv.weight", "model.layers.*.self_attn.W_pack.weight"
        };
    }

    void Qwen3Model::InitParams() {
        basellm::InitParams();
        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
            head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        }
        embed_dim = head_dim * num_attention_heads;
        rotary_dim = head_dim;
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
        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(rope_base, rope_factor, std::max(max_positions, 16384));
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));
        for (int i = 0; i < block_cnt; i++) {
            std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
            );

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
    }

    std::pair<std::vector<float>, std::vector<float>> Qwen3Model::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
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

    int Qwen3Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> Qwen3Model::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv, curInput, curOutput;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
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

            if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()
                && CanRunMergeAttention()
                && false) {
                // MLP(attenInput, weight[swigluWeightName], Data(), weight[downWeightName], Data(), k);
                // printf("n_head = %d, %d\n", num_attention_heads, num_key_value_heads);
                std::vector <Data*> keys, values, masks;
                keys.push_back(&pastKeyValues[i].first);
                values.push_back(&pastKeyValues[i].second);
                masks.push_back((Data*)&attentionMask);
                MergeAttention (
                    attenInput, 
                    weight[mergeQkvWeightName], weight[mergeQkvBiasName], 
                    weight[oWeightName], weight[oBiasName],
                    qkv, q, k, v, curInput, curOutput,
                    num_attention_heads, num_key_value_heads, head_dim, rotary_dim, 1.0 / sqrt(head_dim),
                    positionIds, *sinDataPtr, *cosDataPtr, 
                    keys, values, masks, w1
                );
                AddTo(hiddenStates, w1);
            } else {
                if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                    int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                    int qdim = per * (num_attention_heads / num_key_value_heads);
                    Split(qkv, -1, 0, qdim, q);
                    Split(qkv, -1, qdim, qdim + per, k);
                    Split(qkv, -1, qdim + per, qdim + per * 2, v);
                } else {
                    if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                        Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                        int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                        int qdim = per * (num_attention_heads / num_key_value_heads);

                        Split(qkv, -1, 0, qdim, q);
                        Split(qkv, -1, qdim, qdim + per, k);
                        Split(qkv, -1, qdim + per, qdim + per * 2, v);
                    } else {
                        Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                        Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                        Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                        Linear(attenInput, weight[qWeightName], qBias, q);
                        Linear(attenInput, weight[kWeightName], kBias, k);
                        Linear(attenInput, weight[vWeightName], vBias, v);
                    }
                }

                std::vector <int> qkvSize = {bsz, seqlen, -1, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                RMSNorm(q, this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);

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
                Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);

                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});

                Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
                Linear(qkv, weight[oWeightName], oBias, attenInput);
                AddTo(hiddenStates, attenInput);
            }

            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);

            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            if (weight.weight.find(swigluWeightName) != weight.weight.end() && CanRunMLP()) {
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
                MLP(attenInput, weight[swigluWeightName], Data(), weight[downWeightName], Data(), w1, w2, w3, k);
                AddTo(hiddenStates, k);
            } else {
                if (weight.weight.find(swigluWeightName) != weight.weight.end()) {
                    if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                        LinearEx(attenInput, weight[swigluWeightName], Data(), q, LinearExType::ExSwiglu);
                    } else {
                        Linear(attenInput, weight[swigluWeightName], Data(), v);
                        Swiglu(v, q);
                    }
                } else {
                    if (CanRunLinearEx(LinearExType::ExSilu)) {
                        LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), q, LinearExType::ExSilu);
                    } else {
                        Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), q);
                        Silu(q, q);
                    }
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), v);
                    MulTo(q, v);
                }
                Linear(q, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), k);
                AddTo(hiddenStates, k);
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

        std::vector <int> lastRet;
        {
            auto &hiddenStates = *lastHiddenStates;
            RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
            ToDataType(logits, DataType::FLOAT32);
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

            if (generationConfig.top_k <= 1) {
                // 禁用simple greedy
                ((GenerationConfig*)&generationConfig)->top_k = 5;
                ((GenerationConfig*)&generationConfig)->top_p = 0.95;
                if (fabs(generationConfig.temperature - 1.0f) < 1e-9) {
                    ((GenerationConfig*)&generationConfig)->temperature = 0.6;
                }
            }

            ResetLogitsOfEOS(batch, &logits, pastKeyValues, generationConfig);
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

    std::vector <int> Qwen3Model::ForwardBatch(int batch,
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
        Data w1, w2, w3;
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
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
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
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                int qdim = per * (num_attention_heads / num_key_value_heads);
                Split(qkv, -1, 0, qdim, q);
                Split(qkv, -1, qdim, qdim + per, k);
                Split(qkv, -1, qdim + per, qdim + per * 2, v);
            } else {
                if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                    int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                    int qdim = per * (num_attention_heads / num_key_value_heads);

                    Split(qkv, -1, 0, qdim, q);
                    Split(qkv, -1, qdim, qdim + per, k);
                    Split(qkv, -1, qdim + per, qdim + per * 2, v);
                } else {
                    Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                    Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                    Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                    Linear(attenInput, weight[qWeightName], qBias, q);
                    Linear(attenInput, weight[kWeightName], kBias, k);
                    Linear(attenInput, weight[vWeightName], vBias, v);
                }
            }

            q.Reshape({q.dims[0], q.dims[1], -1, head_dim});
            k.Reshape({k.dims[0], k.dims[1], -1, head_dim});
            v.Reshape({v.dims[0], v.dims[1], -1, head_dim});

            RMSNorm(q, this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
            RMSNorm(k, this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);

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

            fastllm::LlamaRotatePosition2D(q, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            fastllm::LlamaRotatePosition2D(k, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);

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
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);

            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            if (weight.weight.find(swigluWeightName) != weight.weight.end() && CanRunMLP()) {
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
                MLP(attenInput, weight[swigluWeightName], Data(), weight[downWeightName], Data(), w1, w2, w3, k);
                AddTo(hiddenStates, k);
            } else {
                if (weight.weight.find(swigluWeightName) != weight.weight.end()) {
                    if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                        LinearEx(attenInput, weight[swigluWeightName], Data(), w1, LinearExType::ExSwiglu);
                    } else {
                        Linear(attenInput, weight[swigluWeightName], Data(), w3);
                        Swiglu(w3, w1);
                    }
                } else {
                    if (CanRunLinearEx(LinearExType::ExSilu)) {
                        LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1, LinearExType::ExSilu);
                    } else {
                        Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
                        Silu(w1, w1);
                    }
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
                    MulTo(w1, w3);
                }

                Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
                AddTo(hiddenStates, w2);
            }
        }

        Data logits;
        std::vector <Data> curLogits;
        curLogits.resize(batch);

        if (batch > 1 && !all1) {
            int total = 0;
            std::vector <Data> lastTokens;
            std::vector <Data*> lastTokenPointers;
            lastTokens.resize(seqLens.size());
            for (int b = 0; b < seqLens.size(); b++) {
                Split(hiddenStates, 1, total + seqLens[b] - 1, total + seqLens[b], lastTokens[b]);
                total += seqLens[b];
                lastTokenPointers.push_back(&lastTokens[b]);
            }
            CatBatch(lastTokenPointers, 1, hiddenStates);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        ToDataType(logits, DataType::FLOAT32);
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

        bool allSimple = true, needLogits = false;
        int maxTopK = 1;
        for (int b = 0; b < batch; b++) {
            if (!generationConfigs[b].IsSimpleGreedy()) {
                allSimple = false;
                break;
            }
        }
        for (int b = 0; b < batch; b++) {
            needLogits |= generationConfigs[b].output_logits;
            maxTopK = std::max(maxTopK, generationConfigs[b].top_k);
        }

        ResetLogitsOfEOS(batch, &logits, pastKeyValues, generationConfigs);
        if (batch > 1 && allSimple) {
            Data topk;
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            float *topkData = (float*)topk.cpuData;
            for (int b = 0; b < batch; b++) {
                lastRet.push_back((int) (topkData[0] + 1e-3));
                topkData += topk.Count(2);
            }
        } else if (batch > 1 && maxTopK <= 50 && !needLogits) {
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
                penaltyScaleData[b] = generationConfigs[b].repeat_penalty;
            }
            Data penalty, penaltyScale;
            penalty.CopyFrom(Data(DataType::FLOAT32, {batch, maxTokenSetSize}, penaltyData));
            penaltyScale.CopyFrom(Data(DataType::FLOAT32, {batch}, penaltyScaleData));
            RepeatPenalty(logits, penalty, penaltyScale);
            Data topk;
            TopK(logits, topk, maxTopK);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                lastRet.push_back(LLMSamplingOnly(topk, b, generationConfigs[b]));
            }
        } else {
            for (int b = 0; b < batch; b++) {
                pointersK[b] = (&curLogits[b]);
            }
            SplitBatch(logits, 1, batch, pointersK);

            for (int b = 0; b < batch; b++) {
                Data &curLogit = curLogits[b];
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
            }
        }
        if (sinDataPtr != &sinData)
            delete sinDataPtr;
        if (cosDataPtr != &cosData)
            delete cosDataPtr;
        return lastRet;
    }

    bool Qwen3Model::NeedAttentionMask(int qlen, int klen) {
        if (((qlen == 1) || (qlen >= 1024))) {
            return false;
        }
        return true;
    }

    void Qwen3Model::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
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

    std::string Qwen3Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Qwen3Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void Qwen3Model::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            this->weight["lm_head.weight"] = Data();
            this->weight["lm_head.weight"].CopyFrom(this->weight["model.embed_tokens.weight"]);
            ToDataType(this->weight["lm_head.weight"], DataType::FLOAT16);
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
    }
}
