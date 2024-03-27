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

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int qdim = num_attention_heads / num_key_value_heads;
                qkv.Reshape({-1, (num_attention_heads / num_key_value_heads + 2), head_dim});
                Split(qkv, -2, 0, qdim, q);
                Split(qkv, -2, qdim, qdim + 1, k);
                Split(qkv, -2, qdim + 1, qdim + 2, v);
            } else {
                Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                Linear(attenInput, weight[qWeightName], qBias, q);
                Linear(attenInput, weight[kWeightName], kBias, k);
                Linear(attenInput, weight[vWeightName], vBias, v);
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
            MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim), q.dims[0] / pastKey.dims[0]);
            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attenWeights, attentionMask, -10000);
            }
            Softmax(attenWeights, attenWeights, -1);
            MatMul(attenWeights, pastValue, attenOutput, 1.f, attenWeights.dims[1] / pastValue.dims[0]);

            attenOutput.Reshape({attenOutput.dims[1], attenOutput.dims[2], attenOutput.dims[3]});
            PermuteSelf(attenOutput, {1, 0, 2});
            attenOutput.Reshape({seqlen, bsz, -1});
            PermuteSelf(attenOutput, {1, 0, 2});

            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(attenOutput, weight[oWeightName], oBias, attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".ffn_norm.weight"], rms_norm_eps, attenInput);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w1.weight"], Data(), w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w3.weight"], Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".feed_forward.w2.weight"], Data(), w2);
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
            Linear(hiddenStates, weight["output.weight"], Data(), logits);
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

        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, curAttenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        std::vector <Data*> sinDataPtrList(batch, &sinData);
        std::vector <Data*> cosDataPtrList(batch, &cosData);

        Embedding(inputIds, this->weight["model.tok_embeddings.weight"], hiddenStates);
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
            } else {
                Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                Linear(attenInput, weight[qWeightName], qBias, q);
                Linear(attenInput, weight[kWeightName], kBias, k);
                Linear(attenInput, weight[vWeightName], vBias, v);
            }

            Data attenOutput = Data(DataType::FLOAT32);
            int total = 0;
            std::vector <Data> curKs, curVs, curQs;
            curKs.resize(batch);
            curVs.resize(batch);
            curQs.resize(batch);
            for (int b = 0; b < batch; b++) {
                Split(k, 1, total, total + seqLens[b], curKs[b]);
                Split(v, 1, total, total + seqLens[b], curVs[b]);
                Split(q, 1, total, total + seqLens[b], curQs[b]);
                total += seqLens[b];
            }

            for (int b = 0; b < batch; b++) {
                auto &q = curQs[b], &k = curKs[b], &v = curVs[b];

                std::vector<int> qkvSize = {bsz, seqLens[b], -1, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(DataDevice::CUDA);
                    pastValue.ToDevice(DataDevice::CUDA);
                }
                int targetSeqLength = (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqLens[b] : seqLens[b];
                if (i == 0 && targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                    float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                    float newbase = rope_base * scale;
                    std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                    sinDataPtrList[b] = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                    cosDataPtrList[b] = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
                }

                fastllm::LlamaRotatePosition2D(q, *positionIds[b], *sinDataPtrList[b], *cosDataPtrList[b], rotary_dim);
                fastllm::LlamaRotatePosition2D(k, *positionIds[b], *sinDataPtrList[b], *cosDataPtrList[b], rotary_dim);

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});

                qkvSize = {-1, seqLens[b], head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);
                
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

                CatDirect(pastKey, k, 1);
                CatDirect(pastValue, v, 1);

                // 1.2 Attention
                // 1.2.0 q * k^T
                MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim), q.dims[0] / pastKey.dims[0]);
                attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
                if (attentionMask[b] != nullptr) {
                    AttentionMask(attenWeights, *attentionMask[b], -10000);
                }

                Softmax(attenWeights, attenWeights, -1);
                MatMul(attenWeights, pastValue, curAttenOutput, 1.f, attenWeights.dims[1] / pastValue.dims[0]);
                curAttenOutput.Reshape({curAttenOutput.dims[1], curAttenOutput.dims[2], curAttenOutput.dims[3]});
                PermuteSelf(curAttenOutput, {1, 0, 2});
                curAttenOutput.Reshape({seqLens[b], bsz, -1});
                PermuteSelf(curAttenOutput, {1, 0, 2});
                if (attenOutput.dims.size() == 0) {
                    std::vector <int> dims = curAttenOutput.dims;
                    dims[1] = total;
                    attenOutput.Expansion(dims);
                }
                CatDirect(attenOutput, curAttenOutput, 1);
            }

            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(attenOutput, weight[oWeightName], oBias, attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".ffn_norm.weight"], rms_norm_eps, attenInput);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w1.weight"], Data(), w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".feed_forward.w3.weight"], Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".feed_forward.w2.weight"], Data(), w2);
            AddTo(hiddenStates, w2);
        }

        Data logits, curLogit;
        RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
        Linear(hiddenStates, weight["output.weight"], Data(), logits);
        std::vector <int> lastRet;
        int total = 0;
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
        for (Data* sinPtr : sinDataPtrList)
            if (sinPtr != &sinData)
                delete sinPtr;
        for (Data* cosPtr : cosDataPtrList)
            if (cosPtr != &cosData)
                delete cosPtr;
        return lastRet;
    }

}
