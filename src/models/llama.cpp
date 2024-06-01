//
// Created by huangyuyang on 6/1/23.
//

#include "utils.h"

#include "llama.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    std::vector <float> GetInterLeavePowerOf2(int n) {
        float start = powf(2, -powf(2, -(log2f(n) - 3)));
        float ratio = start;
        std::vector <float> ret;
        for (int i = 0; i < n; i++) {
            ret.push_back(start * powf(ratio, i));
        }
        return ret;
    }
    std::vector <float> GetInterleave(int n) {
        int base = 1;
        while (base < n) {
            base <<= 1;
        }
        if (base == n) {
            return GetInterLeavePowerOf2(n);
        } else {
            std::vector <float> ret = GetInterLeavePowerOf2(base / 2);
            std::vector <float> part2 = GetInterLeavePowerOf2(base);
            for (int i = 0; i < n - base / 2; i++) {
                ret.push_back(part2[i * 2]);
            }
            return ret;
        }
    }

    LlamaModel::LlamaModel() {
        this->model_type = "llama";

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

    void LlamaModel::InitParams() {
        basellm::InitParams();
        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        head_dim = embed_dim / num_attention_heads;
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
        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(rope_base, rope_factor);
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));
    }

    std::pair<std::vector<float>, std::vector<float>> LlamaModel::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
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

    int LlamaModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> LlamaModel::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        if (!mergeQKV) {
            bool canMerge = true;
            for (int i = 0; i < block_cnt; i++) {
                std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
                
                std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
                std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
                std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
                std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
                std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
                std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
                std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

                if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                    mergeQKV = true;
                    break;
                } else {
                    Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                    Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                    Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();

                    Data &q = weight.weight[qWeightName];
                    Data &k = weight.weight[kWeightName];
                    Data &v = weight.weight[vWeightName];

                    if ((q.dataType == DataType::INT4_GROUP && q.dims[1] % q.groupCnt != 0) || 
                        (k.dataType == DataType::INT4_GROUP && k.dims[1] % k.groupCnt != 0) ||
                        (v.dataType == DataType::INT4_GROUP && v.dims[1] % v.groupCnt != 0)) {
                        canMerge = false;
                        break;
                    }

                    if (weight.weight.find(qBiasName) != weight.weight.end()) {
                        Data middle;
                        Cat(qBias, kBias, -1, middle);
                        Cat(middle, vBias, -1, weight.weight[mergeQkvBiasName]);
                        weight.weight[mergeQkvBiasName].name = mergeQkvBiasName;
                    } else {
                        weight.weight[mergeQkvBiasName] = Data();
                    }

                    weight.weight[mergeQkvWeightName] = Data(q.dataType, {q.dims[0] + k.dims[0] + v.dims[0], q.dims[1]});
                    Data &mergeQKV = weight.weight[mergeQkvWeightName];

                    mergeQKV.name = mergeQkvWeightName;
                    mergeQKV.Allocate();
                    memcpy(mergeQKV.cpuData, q.cpuData, q.GetBytes());
                    memcpy(mergeQKV.cpuData + q.GetBytes(), k.cpuData, k.GetBytes());
                    memcpy(mergeQKV.cpuData + q.GetBytes() + k.GetBytes(), v.cpuData, v.GetBytes());
                    mergeQKV.group = q.group;
                    mergeQKV.groupCnt = q.groupCnt;
                    mergeQKV.perChannelAxis = q.perChannelAxis;
                    mergeQKV.perChannelsConfigs = AppendVector(q.perChannelsConfigs, AppendVector(k.perChannelsConfigs, v.perChannelsConfigs));
                    mergeQKV.zeros = AppendVector(q.zeros, AppendVector(k.zeros, v.zeros));
                    mergeQKV.scales = AppendVector(q.scales, AppendVector(k.scales, v.scales));
                    mergeQKV.mins = AppendVector(q.mins, AppendVector(k.mins, v.mins));

                    weight.weight.erase(qWeightName);
                    weight.weight.erase(kWeightName);
                    weight.weight.erase(vWeightName);
                    weight.weight.erase(qBiasName);
                    weight.weight.erase(kBiasName);
                    weight.weight.erase(vBiasName);
                }
            }

            this->mergeQKV = canMerge;
        }

        if (!mergeSwiglu) {
            bool canMerge = true;
            for (int i = 0; i < block_cnt; i++) {
                std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
                std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.up_proj.weight";
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";

                Data &w1 = weight.weight[w1WeightName], &w3 = weight.weight[w3WeightName];
                if ((w1.dataType == DataType::INT4_GROUP && w1.dims[1] % w1.groupCnt != 0) || 
                    (w3.dataType == DataType::INT4_GROUP && w3.dims[1] % w3.groupCnt != 0)) {
                    canMerge = false;
                    break;
                }

                weight.weight[swigluWeightName] = Data(w1.dataType, {w1.dims[0] + w3.dims[0], w1.dims[1]});
                Data &swiglu = weight.weight[swigluWeightName];
                swiglu.name = swigluWeightName;
                swiglu.Allocate();
                memcpy(swiglu.cpuData, w1.cpuData, w1.GetBytes());
                memcpy(swiglu.cpuData + w1.GetBytes(), w3.cpuData, w3.GetBytes());
                    
                swiglu.perChannelAxis = w1.perChannelAxis;
                swiglu.group = w1.group;
                swiglu.groupCnt = w1.groupCnt;
                swiglu.perChannelsConfigs = AppendVector(w1.perChannelsConfigs, w3.perChannelsConfigs);
                swiglu.zeros = AppendVector(w1.zeros, w3.zeros);
                swiglu.scales = AppendVector(w1.scales, w3.scales);
                swiglu.mins = AppendVector(w1.mins, w3.mins);

                weight.weight.erase(w1WeightName);
                weight.weight.erase(w3WeightName);
            }

            this->mergeSwiglu = canMerge;            
        }
        
        Data alibiData;
        if (this->weight.dicts["use_alibi"] == "1") {
            std::vector<float> alibi = GetInterleave(num_attention_heads);
            alibiData.CopyFrom(Data(DataType::FLOAT32, {(int) alibi.size()}, alibi));
        }

        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
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

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(DataDevice::CUDA);
                pastValue.ToDevice(DataDevice::CUDA);
            }
            int targetSeqLength = (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqlen : seqlen;
            if (i == 0 && targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                float newbase = rope_base * scale;
                std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                sinDataPtr = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                cosDataPtr = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
            }

            if (alibiData.dims.size() == 0) {
                fastllm::LlamaRotatePosition2D(q, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
                fastllm::LlamaRotatePosition2D(k, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            }

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
            // 1.2.0 q * k^T
            if (alibiData.dims.size() == 0) {
                Attention(q, pastKey, pastValue, attentionMask, attenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
            } else {
                MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim), q.dims[0] / pastKey.dims[0]);
                attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
                if (alibiData.dims.size() != 0) {
                    attenWeights.Reshape({-1, num_attention_heads, attenWeights.dims[2], attenWeights.dims[3]});
                    AlibiMask(attenWeights, alibiData, -10000);
                    attenWeights.Reshape({1, -1, attenWeights.dims[2], attenWeights.dims[3]});
                } else if (attentionMask.dims.size() != 0) {
                    AttentionMask(attenWeights, attentionMask, -10000);
                }

                Softmax(attenWeights, attenWeights, -1);
                MatMul(attenWeights, pastValue, attenOutput, 1.f, attenWeights.dims[1] / pastValue.dims[0]);
                attenOutput.Reshape({attenOutput.dims[1], attenOutput.dims[2], attenOutput.dims[3]});
            }

            PermuteSelf(attenOutput, {1, 0, 2});
            attenOutput.Reshape({seqlen, bsz, -1});
            PermuteSelf(attenOutput, {1, 0, 2});

            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(attenOutput, weight[oWeightName], oBias, attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);
            if (this->mergeSwiglu) {
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
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

            if (generationConfig.IsSimpleGreedy()) {
                TopK(logits, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    int base = b;
                    lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
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

    std::vector <int> LlamaModel::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        int seqLen = inputIds.dims[1];
        Data alibiData;
        if (this->weight.dicts["use_alibi"] == "1") {
            std::vector<float> alibi = GetInterleave(num_attention_heads);
            alibiData.CopyFrom(Data(DataType::FLOAT32, {(int) alibi.size()}, alibi));
        }

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
        std::vector <std::vector <int> > outputSizes;
        outputSizes.resize(batch);
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

        if (all1) {
            for (int b = 0; b < batch; b++) {
                contexts[b] = positionIds[b];
            }
            CatBatch(contexts, 1, allPositionIds);
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

            int targetSeqLength = 0;
            for (int b = 0; b < batch; b++) {
                auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(DataDevice::CUDA);
                    pastValue.ToDevice(DataDevice::CUDA);
                }
                targetSeqLength = std::max(targetSeqLength, (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqLens[b] : seqLens[b]);
            }

            if (i == 0 && targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                float newbase = rope_base * scale;
                std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                sinDataPtr = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                cosDataPtr = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
            }

            if (alibiData.dims.size() == 0) {
                fastllm::LlamaRotatePosition2D(q, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
                fastllm::LlamaRotatePosition2D(k, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            }

            Data attenOutput = Data(this->dataType);
            int total = 0;
            if (all1 && batch > 1) {
                q.Reshape({-1, q.dims[2], q.dims[3]});
                k.Reshape({-1, k.dims[2], k.dims[3]});
                v.Reshape({-1, v.dims[2], v.dims[3]});

                for (int b = 0; b < batch; b++) {
                    curQs[b].Resize({q.dims[1], 1, q.dims[2]});
                    curQs[b].FakeFrom(q, b * q.strides[0] * q.unitSize);
                    curKs[b].Resize({k.dims[1], 1, k.dims[2]});
                    curKs[b].FakeFrom(k, b * k.strides[0] * k.unitSize);
                    curVs[b].Resize({v.dims[1], 1, v.dims[2]});
                    curVs[b].FakeFrom(v, b * v.strides[0] * v.unitSize);
                }

                total = batch;
            } else {
                for (int b = 0; b < batch; b++) {
                    Split(k, 1, total, total + seqLens[b], curKs[b]);
                    Split(v, 1, total, total + seqLens[b], curVs[b]);
                    Split(q, 1, total, total + seqLens[b], curQs[b]);
                    total += seqLens[b];
                }

                for (int b = 0; b < batch; b++) {
                    auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});

                    std::vector<int> qkvSize = {-1, seqLens[b], head_dim};
                    q.Reshape(qkvSize);
                    k.Reshape(qkvSize);
                    v.Reshape(qkvSize);
                }
            }

            for (int b = 0; b < batch; b++) {
                auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                
                int unitLen = 64;
#ifdef USE_CUDA
                unitLen = 128;
#endif
                while ((pastKey.dims.size() == 0 &&
                        (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                       || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector<int>{k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 &&
                        (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                       || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector<int>{v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
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

            if (alibiData.dims.size() == 0 && all1 && batch > 1) {
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

                    outputSizes[b] = {1, qs[b]->dims[0], qs[b]->dims[1], keys[b]->dims[1]};
                }
                AttentionBatch(qs, keys, values, masks, contexts, qs[0]->dims[0] / values[0]->dims[0], 1.0 / scale_attn, 1);
            } else {
                for (int b = 0; b < batch; b++) {
                    auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;

                    // 1.2 Attention
                    // 1.2.0 q * k^T
                    if (alibiData.dims.size() == 0) {
                        if (attentionMask[b] == nullptr) {
                            Attention(q, pastKey, pastValue, Data(), curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                        } else {
                            Attention(q, pastKey, pastValue, *attentionMask[b], curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                        }
                    } else {
                        MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim), q.dims[0] / pastKey.dims[0]);
                        attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
                        if (alibiData.dims.size() != 0) {
                            AlibiMask(attenWeights, alibiData, -10000);
                        } else if (attentionMask[b] != nullptr) {
                            AttentionMask(attenWeights, *attentionMask[b], -10000);
                        }

                        Softmax(attenWeights, attenWeights, -1);
                        MatMul(attenWeights, pastValue, curAttenOutput, 1.f, attenWeights.dims[1] / pastValue.dims[0]);
                        curAttenOutput.Reshape({curAttenOutput.dims[1], curAttenOutput.dims[2], curAttenOutput.dims[3]});
                    }

                    PermuteSelf(curAttenOutput, {1, 0, 2});
                    curAttenOutput.Reshape({seqLens[b], bsz, -1});
                    PermuteSelf(curAttenOutput, {1, 0, 2});
                    if (attenOutput.dims.size() == 0) {
                        std::vector <int> dims = curAttenOutput.dims;
                        dims[1] = total;
                        attenOutput.Expansion(dims);
                        attenOutput.ToDevice(q.dataDevice);
                    }
                    CatDirect(attenOutput, curAttenOutput, 1);
                }
            }

            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(attenOutput, weight[oWeightName], oBias, attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);
            if (this->mergeSwiglu) {
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
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

        Data logits;
        std::vector <Data> curLogits;
        curLogits.resize(batch);

        RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        ToDataType(logits, DataType::FLOAT32);
        std::vector <int> lastRet;
        int total = 0;

        if (all1 && batch > 1) {
            for (int b = 0; b < batch; b++) {
                pointersK[b] = (&curLogits[b]);
            }
            SplitBatch(logits, 1, batch, pointersK);
        } else {
            for (int b = 0; b < batch; b++) {
                Split(logits, 1, total + seqLens[b] - 1, total + seqLens[b], curLogits[b]);
            }
        }

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
            total += seqLens[b];
        }
        if (sinDataPtr != &sinData)
            delete sinDataPtr;
        if (cosDataPtr != &cosData)
            delete cosDataPtr;
        return lastRet;
    }

    void LlamaModel::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
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

    std::string LlamaModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string LlamaModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void LlamaModel::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        printf("finish.\n");
    }
}
