//
// Created by huangyuyang on 6/1/23.
//

#include "utils.h"

#include "internlm2.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    Internlm2Model::Internlm2Model()
        : LlamaModel() {
        this->model_type = "internlm";
        rotary_dim = 128;
        weight.embeddingNames.insert("model.tok_embeddings.weight");
        weight.linearNames = {"model.layers.*.attention.wq.weight", "model.layers.*.attention.wk.weight", "model.layers.*.attention.wv.weight", 
            "model.layers.*.attention.wqkv.weight", "model.layers.*.attention.wo.weight",
            "model.layers.*.feed_forward.w1.weight", "model.layers.*.feed_forward.w2.weight", "model.layers.*.feed_forward.w3.weight",
            "output.weight"};
    }

    void Internlm2Model::InitParams() {
        LlamaModel::InitParams();
        weight.tokenizer.SetSpecialTokens({{"</s>", 2}, {"<s>", 1}, {"<unk>", 0}, {"<|im_start|>", 92543}, {"<|im_end|>", 92542}, 
                                          {"<|action_start|>", 92541}, {"<|action_end|>", 92540}, {"<|interpreter|>", 92539}, {"<|plugin|>", 92538}});
    }

    int Internlm2Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                                const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                                std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> Internlm2Model::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                                const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                                std::vector <std::vector <float>*> *retLogits) {
        if (!mergeSwiglu) {
            bool canMerge = true;
            for (int i = 0; i < block_cnt; i++) {
                std::string w1WeightName = "model.layers." + std::to_string(i) + ".feed_forward.w1.weight";
                std::string w3WeightName = "model.layers." + std::to_string(i) + ".feed_forward.w3.weight";
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";

                if (weight.weight.find(swigluWeightName) != weight.weight.end()) {
                    mergeQKV = true;
                    break;
                }
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

        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;

        Embedding(inputIds, this->weight["model.tok_embeddings.weight"], hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".attention_norm.weight"],
                    rms_norm_eps, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".attention.wq.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".attention.wq.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".attention.wk.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".attention.wk.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".attention.wv.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".attention.wv.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".attention.wqkv.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".attention.wo.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".attention.wo.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
/*
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                int qdim = per * (num_attention_heads / num_key_value_heads);
                Split(qkv, -1, 0, qdim, q);
                Split(qkv, -1, qdim, qdim + per, k);
                Split(qkv, -1, qdim + per, qdim + per * 2, v);
*/
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int qdim = num_attention_heads / num_key_value_heads;
                qkv.Reshape({-1, (num_attention_heads / num_key_value_heads + 2), head_dim});
                Split(qkv, -2, 0, qdim, q);
                Split(qkv, -2, qdim, qdim + 1, k);
                Split(qkv, -2, qdim + 1, qdim + 2, v);
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
            // 1.2.0 q * k^T
            Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
            PermuteSelf(qkv, {1, 0, 2});
            qkv.Reshape({seqlen, bsz, -1});
            PermuteSelf(qkv, {1, 0, 2});

            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(qkv, weight[oWeightName], oBias, attenInput);
            AddTo(hiddenStates, attenInput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".ffn_norm.weight"], rms_norm_eps, attenInput);
            if (this->mergeSwiglu) {
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                    LinearEx(attenInput, weight[swigluWeightName], Data(), q, LinearExType::ExSwiglu);
                } else {
                    Linear(attenInput, weight[swigluWeightName], Data(), v);
                    Swiglu(v, q);
                }
            } else {
                if (CanRunLinearEx(LinearExType::ExSilu)) {
                    LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w1.weight"], Data(), q, LinearExType::ExSilu);
                } else {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w1.weight"], Data(), q);
                    Silu(q, q);
                }
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w3.weight"], Data(), v);
                MulTo(q, v);
            }
            Linear(q, weight["model.layers." + std::to_string(i) + ".feed_forward.w2.weight"], Data(), k);
            AddTo(hiddenStates, k);
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
            Linear(hiddenStates, weight["output.weight"], Data(), logits);
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

    std::vector <int> Internlm2Model::ForwardBatch(int batch,
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

        Embedding(inputIds, this->weight["model.tok_embeddings.weight"], hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".attention_norm.weight"],
                    rms_norm_eps, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".attention.wq.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".attention.wq.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".attention.wk.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".attention.wk.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".attention.wv.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".attention.wv.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".attention.wqkv.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".attention.wo.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".attention.wo.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".attention.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".attention.mergeqkv.bias";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int qdim = num_attention_heads / num_key_value_heads;
                qkv.Reshape({-1, (num_attention_heads / num_key_value_heads + 2), head_dim});
                Split(qkv, -2, 0, qdim, q);
                Split(qkv, -2, qdim, qdim + 1, k);
                Split(qkv, -2, qdim + 1, qdim + 2, v);
                q.Reshape({bsz, -1, embed_dim});
                k.Reshape({bsz, -1, head_dim * num_key_value_heads});
                v.Reshape({bsz, -1, head_dim * num_key_value_heads});
/*
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                int qdim = per * (num_attention_heads / num_key_value_heads);
                Split(qkv, -1, 0, qdim, q);
                Split(qkv, -1, qdim, qdim + per, k);
                Split(qkv, -1, qdim + per, qdim + per * 2, v);
*/
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
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".ffn_norm.weight"], rms_norm_eps, attenInput);
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
                    LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w1.weight"], Data(), w1, LinearExType::ExSilu);
                } else {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w1.weight"], Data(), w1);
                    Silu(w1, w1);
                }
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w3.weight"], Data(), w3);
                MulTo(w1, w3);
            }

            Linear(w1, weight["model.layers." + std::to_string(i) + ".feed_forward.w2.weight"], Data(), w2);
            AddTo(hiddenStates, w2);
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
        Linear(hiddenStates, weight["output.weight"], Data(), logits);
        ToDataType(logits, DataType::FLOAT32);
        std::vector <int> lastRet;
        int total = 0;

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
}
