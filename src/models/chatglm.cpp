//
// Created by huangyuyang on 5/11/23.
//

#include "utils.h"

#include "chatglm.h"

#include <cmath>

#include <chrono>

#include <algorithm>

#include <map>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    ChatGLMModel::ChatGLMModel() {
        this->model_type = "chatglm";

        this->bos_token_id = 130004;
        this->eos_token_id = 130005;

        sin.resize(max_positions);
        cos.resize(max_positions);
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(10000, (float)i / rotary_dim));
        }
        for (int i = 0; i < max_positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size(); j++) {
                sin[i][j] = ::sin((float)i * invFreq[j]);
                cos[i][j] = ::cos((float)i * invFreq[j]);
            }
        }

        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            for (int j = 0; j < sin[0].size(); j++) {
                fsin.push_back(sin[i][j]);
                fcos.push_back(cos[i][j]);
            }
        }
        sinData.CopyFrom(Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, fsin));
        cosData.CopyFrom(Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, fcos));
        if (GetVersion() == 1) {
            weight.embeddingNames.insert("transformer.word_embeddings.weight");
        } else if (GetVersion() == 2) {
            weight.embeddingNames.insert("transformer.embedding.word_embeddings.weight");
        }
    }

    int ChatGLMModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                              const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                              const GenerationConfig &generationConfig, const LastTokensManager &lastTokens) {
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens)[0];
    }

    std::vector <int> ChatGLMModel::ForwardBatch(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector <std::pair <Data, Data> > &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens) {
        int maxLen = inputIds.dims[1];
        Data inputEmbeddings;
        Data attenInput;
        Data qkv, q, k, v;
        Data attnProbs;
        Data attnOutput;
        Data contextLayer;
        Data mlpInput;
        Data middle, middle2;
        Data temp;
        std::vector<int> lastRet;
        // ChatGLMBlock
        int version = GetVersion();
        std::string weightPre, weightMiddle;
        if (version == 1) {
            weightPre = "transformer.layers.";
            weightMiddle = ".attention";
        } else if (version == 2) {
            weightPre = "transformer.encoder.layers.";
            weightMiddle = ".self_attention";
        }

        // ChatGLM2
        Data inputIdsPermute;
        Permute(inputIds, {1, 0}, inputIdsPermute);
        Embedding(inputIdsPermute, this->weight["transformer" + std::string((version == 2 ? ".embedding" : "")) +
                                                ".word_embeddings.weight"], inputEmbeddings);
        Data &hiddenStates = inputEmbeddings;
        for (int i = 0; i < block_cnt; i++) {
            if (version == 1) {
                std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
                std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
                LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
            } else if (version == 2) {
                std::string inputRMSWeightName =
                        "transformer.encoder.layers." + std::to_string(i) + ".input_layernorm.weight";
                RMSNorm(hiddenStates, weight[inputRMSWeightName], 1e-5, attenInput);
            }
            std::string qkvWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.weight";
            std::string qkvBiasName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.bias";
            Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);

            if (version == 1) {
                qkv.Reshape({qkv.dims[0], qkv.dims[1], num_attention_heads, -1});
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
                fastllm::RotatePosition2D(q, positionIds, sinData, cosData, rotary_dim);
                fastllm::RotatePosition2D(k, positionIds, sinData, cosData, rotary_dim);
            } else if (version == 2) {
                int qLen = embed_dim, kvLen = (qkv.dims.back() - embed_dim) / 2;
                Split(qkv, -1, 0, qLen, q);
                Split(qkv, -1, qLen, qLen + kvLen, k);
                Split(qkv, -1, qLen + kvLen, qLen + kvLen + kvLen, v);
                q.Reshape({q.dims[0], q.dims[1], -1, embed_dim / num_attention_heads});
                k.Reshape({k.dims[0], k.dims[1], -1, embed_dim / num_attention_heads});
                v.Reshape({v.dims[0], v.dims[1], -1, embed_dim / num_attention_heads});
                fastllm::NearlyRotatePosition2D(q, positionIds, sinData, cosData, rotary_dim);
                fastllm::NearlyRotatePosition2D(k, positionIds, sinData, cosData, rotary_dim);
            }

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(DataDevice::CUDA);
                pastValue.ToDevice(DataDevice::CUDA);
            };

            k.Resize({k.dims[0], k.dims[1] * k.dims[2], k.dims[3]});
            v.Resize({v.dims[0], v.dims[1] * v.dims[2], v.dims[3]});

            PermuteSelf(k, {1, 0, 2});
            PermuteSelf(v, {1, 0, 2});

            int unitLen = 64;
#ifdef USE_CUDA
            unitLen = 128;
#endif
            while ((pastKey.dims.size() == 0 &&
                    (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                   || (pastKey.dims.size() > 0 && (pastKey.expansionDims.size() == 0 ||
                                                   pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1]))) {
                std::vector<int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector<int>{k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                    if (generationConfig.output_token_limit > 0) {
                        newDims[1] = std::min(newDims[1], k.dims[1] + generationConfig.output_token_limit);
                    }
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }

            while ((pastValue.dims.size() == 0 &&
                    (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                   || (pastValue.dims.size() > 0 && (pastValue.expansionDims.size() == 0 ||
                                                     pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1]))) {
                std::vector<int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector<int>{v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                    if (generationConfig.output_token_limit > 0) {
                        newDims[1] = std::min(newDims[1], k.dims[1] + generationConfig.output_token_limit);
                    }
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }
            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 1);
            std::vector<int> outputSize = {q.dims[1], q.dims[2], q.dims[0], pastKey.dims[1]};

            q.Reshape({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});
            PermuteSelf(q, {1, 0, 2});

            // 1.2 Attention
            // 1.2.0 q * k^T
            q.Reshape({pastKey.dims[0], -1, q.dims[2]});
            MatMulTransB(q, pastKey, attnProbs, 1.0 / (scale_attn * (i + 1)));
            attnProbs.Reshape(outputSize);

            // 1.2.1 Mask
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attnProbs, attentionMask, -10000);
            }

            // 1.2.2 softmax
            Mul(attnProbs, i + 1, attnProbs);
            Softmax(attnProbs, attnProbs, -1);
            outputSize = {1, pastValue.dims[0], q.dims[1], pastValue.dims[1]};
            attnProbs.Reshape({outputSize[0] * outputSize[1], outputSize[2], -1});
            // 1.2.3 prob * v

            attnProbs.Reshape({pastValue.dims[0], -1, attnProbs.dims[2]});
            MatMul(attnProbs, pastValue, contextLayer);
            contextLayer.Reshape({batch, num_attention_heads, maxLen, -1});
            PermuteSelf(contextLayer, {2, 0, 1, 3});
            contextLayer.Reshape({contextLayer.dims[0], contextLayer.dims[1], embed_dim});

            // 1.2.4 dense
            std::string denseWeightName = weightPre + std::to_string(i) + weightMiddle + ".dense.weight";
            std::string denseBiasName = weightPre + std::to_string(i) + weightMiddle + ".dense.bias";
            Linear(contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);

            // 1.3
            if (GetVersion() == 1) {
                float alpha = sqrt(2 * block_cnt);
                Mul(attenInput, alpha, hiddenStates);
                AddTo(hiddenStates, attnOutput);
                std::string postLNWeightName =
                        "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                std::string postLNBiasName =
                        "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.bias";
                LayerNorm(hiddenStates, weight[postLNWeightName], weight[postLNBiasName], -1, mlpInput);
                // 1.4 MLP
                std::string fcInKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                std::string fcOutKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
                Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                GeluNew(middle, middle);
                Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
                AddTo(hiddenStates, mlpInput, alpha);
            } else {
                AddTo(hiddenStates, attnOutput);
                std::string postRMSWeightName =
                        "transformer.encoder.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                Mul(hiddenStates, 1.0, temp);
                RMSNorm(hiddenStates, weight[postRMSWeightName], 1e-5, mlpInput);
                // 1.4 MLP
                std::string fcInKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                std::string fcOutKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
                Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                Swiglu(middle, middle2);
                Linear(middle2, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
                AddTo(hiddenStates, temp);
            }
        }

        Data logits, topk;
        if (version == 1) {
            LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"],
                      weight["transformer.final_layernorm.bias"], -1, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        } else {
            RMSNorm(hiddenStates, weight["transformer.encoder.final_layernorm.weight"], 1e-5, hiddenStates);
            Linear(hiddenStates, weight["transformer.output_layer.weight"], Data(), logits);
        }

        if (generationConfig.IsSimpleGreedy()) {
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b;
                lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
            }
        } else {
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b;
                lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
            }
        }

        return lastRet;
    }

    std::vector <int> ChatGLMModel::ForwardBatch(
            int batch,
            const Data &inputIds,
            const std::vector <Data*> &attentionMask,
            const std::vector <Data*> &positionIds,
            const std::vector <int> &seqLens,
            std::vector <std::pair <Data*, Data*> > &pastKeyValues,
            const std::vector <GenerationConfig> &generationConfigs,
            const LastTokensManager &lastTokens) {
        int seqLen = inputIds.dims[1];
        sinData.ToDevice(DataDevice::CUDA);
        cosData.ToDevice(DataDevice::CUDA);
        int version = GetVersion();
        std::string weightPre, weightMiddle;
        if (version == 1) {
            weightPre = "transformer.layers.";
            weightMiddle = ".attention";
        } else if (version == 2) {
            weightPre = "transformer.encoder.layers.";
            weightMiddle = ".self_attention";
        }

        Data inputEmbeddings;
        Data inputIdsPermute;
        Permute(inputIds, {1, 0}, inputIdsPermute);
        Embedding(inputIdsPermute, this->weight["transformer" + std::string((version == 2 ? ".embedding" : "")) +
                                                ".word_embeddings.weight"], inputEmbeddings);
        Data &hiddenStates = inputEmbeddings;
        hiddenStates.ToDevice(DataDevice::CUDA);

        Data attenInput;
        Data qkv, q, k, v;
        Data attnOutput;
        Data mlpInput, middle, middle2;
        std::vector <Data> attnProbs;
        std::vector <Data> curContextLayer;
        std::vector <Data> curKs, curVs, curQs;
        attnProbs.resize(batch);
        curContextLayer.resize(batch);
        curKs.resize(batch);
        curVs.resize(batch);
        curQs.resize(batch);

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }

        if (batch > 1) {
            positionIds[0]->Expansion({2, seqLen});
            for (int i = 1; i < batch; i++) {
                CatDirect(*(Data*)positionIds[0], *(Data*)positionIds[i], 1);
            }
        }

        std::vector <std::vector <int> > outputSizes;
        outputSizes.resize(batch);
        for (int i = 0; i < block_cnt; i++) {
            if (version == 1) {
                std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
                std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
                LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
            } else if (version == 2) {
                std::string inputRMSWeightName =
                        "transformer.encoder.layers." + std::to_string(i) + ".input_layernorm.weight";
                RMSNorm(hiddenStates, weight[inputRMSWeightName], 1e-5, attenInput);
            }

            std::string qkvWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.weight";
            std::string qkvBiasName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.bias";
            Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);

            if (version == 1) {
                qkv.Reshape({qkv.dims[0], qkv.dims[1], num_attention_heads, -1});
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
            } else if (version == 2) {
                int qLen = embed_dim, kvLen = (qkv.dims.back() - embed_dim) / 2;
                Split(qkv, -1, 0, qLen, q);
                Split(qkv, -1, qLen, qLen + kvLen, k);
                Split(qkv, -1, qLen + kvLen, qLen + kvLen + kvLen, v);
                q.Reshape({q.dims[0], q.dims[1], -1, embed_dim / num_attention_heads});
                k.Reshape({k.dims[0], k.dims[1], -1, embed_dim / num_attention_heads});
                v.Reshape({v.dims[0], v.dims[1], -1, embed_dim / num_attention_heads});
            }

            if (version == 1) {
                fastllm::RotatePosition2D(q, *positionIds[0], sinData, cosData, rotary_dim);
                fastllm::RotatePosition2D(k, *positionIds[0], sinData, cosData, rotary_dim);
            } else if (version == 2) {
                fastllm::NearlyRotatePosition2D(q, *positionIds[0], sinData, cosData, rotary_dim);
                fastllm::NearlyRotatePosition2D(k, *positionIds[0], sinData, cosData, rotary_dim);
            }

            k.Resize({k.dims[0], k.dims[1] * k.dims[2], k.dims[3]});
            v.Resize({v.dims[0], v.dims[1] * v.dims[2], v.dims[3]});
            q.Resize({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});

            Data contextLayer = Data(DataType::FLOAT32);
            int total = 0;

            if (all1 && batch > 1) {
                std::vector <Data*> pointersK, pointersV, pointersQ;
                for (int b = 0; b < batch; b++) {
                    pointersK.push_back(&curKs[b]);
                    pointersV.push_back(&curVs[b]);
                    pointersQ.push_back(&curQs[b]);
                }
                SplitBatch(k, 0, batch, pointersK);
                SplitBatch(v, 0, batch, pointersV);
                SplitBatch(q, 0, batch, pointersQ);
                total = batch;
            } else {
                if (batch > 1) {
                    for (int b = 0; b < batch; b++) {
                        Split(k, 0, total, total + seqLens[b], curKs[b]);
                        Split(v, 0, total, total + seqLens[b], curVs[b]);
                        Split(q, 0, total, total + seqLens[b], curQs[b]);
                        total += seqLens[b];
                    }
                }
            }

            for (int b = 0; b < batch; b++) {
                auto pq = &(batch == 1 ? q : curQs[b]);
                auto pk = &(batch == 1 ? k : curKs[b]);
                auto pv = &(batch == 1 ? v : curVs[b]);
                auto &q = *pq, &k = *pk, &v = *pv;
                if (all1) {
                    k.Reshape({k.dims[1], k.dims[0], k.dims[2]});
                    v.Reshape({v.dims[1], v.dims[0], v.dims[2]});
                    q.Reshape({q.dims[1], q.dims[0], q.dims[2]});
                } else {
                    PermuteSelf(k, {1, 0, 2});
                    PermuteSelf(v, {1, 0, 2});
                    PermuteSelf(q, {1, 0, 2});
                }

                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt +
                                                                                                     i].second;
                pastKey.ToDevice(DataDevice::CUDA);
                pastValue.ToDevice(DataDevice::CUDA);

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
                        if (generationConfigs[b].output_token_limit > 0) {
                            newDims[1] = std::min(newDims[1], k.dims[1] + generationConfigs[b].output_token_limit);
                        }
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
                        if (generationConfigs[b].output_token_limit > 0) {
                            newDims[1] = std::min(newDims[1], k.dims[1] + generationConfigs[b].output_token_limit);
                        }
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
            }

            if (batch == 1) {
                CatDirect(*pastKeyValues[i].first, k, 1);
                CatDirect(*pastKeyValues[i].second, v, 1);
            } else {
                std::vector <Data*> keys, values;
                std::vector<Data *> pointersK, pointersV;
                for (int b = 0; b < batch; b++) {
                    keys.push_back(pastKeyValues[b * block_cnt + i].first);
                    values.push_back(pastKeyValues[b * block_cnt + i].second);
                    pointersK.push_back(&curKs[b]);
                    pointersV.push_back(&curVs[b]);
                }
                CatDirectBatch(keys, pointersK, 1);
                CatDirectBatch(values, pointersV, 1);
            }

            for (int b = 0; b < batch; b++) {
                auto pq = &(batch == 1 ? q : curQs[b]);
                auto &q = *pq;
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first;
                outputSizes[b] = {1, q.dims[0], q.dims[1], pastKey.dims[1]};
                q.Reshape({pastKey.dims[0], -1, q.dims[2]});
            }

            // 1.2 Attention
            // 1.2.0 q * k^T
            if (all1 && batch > 1) {
                std::vector <Data*> qs, keys, attns;
                for (int b = 0; b < batch; b++) {
                    qs.push_back(&curQs[b]);
                    keys.push_back(pastKeyValues[b * block_cnt + i].first);
                    attns.push_back(&attnProbs[b]);
                }
                MatMulTransBBatch(qs, keys, attns, 1.0 / (scale_attn * (i + 1)));
            } else {
                for (int b = 0; b < batch; b++) {
                    auto pq = &(batch == 1 ? q : curQs[b]);
                    auto &q = *pq;
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first;
                    MatMulTransB(q, pastKey, attnProbs[b], 1.0 / (scale_attn * (i + 1)));
                }
            }

            for (int b = 0; b < batch; b++) {
                attnProbs[b].Reshape(outputSizes[b]);
                // 1.2.1 Mask
                if (attentionMask[b] != nullptr) {
                    AttentionMask(attnProbs[b], *attentionMask[b], -10000);
                }
            }

            // 1.2.2 softmax
            std::vector <Data*> attns;
            for (int i = 0; i < attnProbs.size(); i++) {
                attns.push_back(&attnProbs[i]);
            }
            MulBatch(attns, i + 1, attns);
            SoftmaxBatch(attns, attns, -1);

            for (int b = 0; b < batch; b++) {
                Data &pastValue = *pastKeyValues[b * block_cnt + i].second;
                outputSizes[b] = {1, num_attention_heads, -1, pastValue.dims[2]};
                attnProbs[b].Reshape({pastValue.dims[0], -1, attnProbs[b].dims[3]});
            }

            // 1.2.3 prob * v
            if (all1 && batch > 1) {
                std::vector <Data*> attns, values, contexts;
                for (int b = 0; b < batch; b++) {
                    attns.push_back(&attnProbs[b]);
                    values.push_back(pastKeyValues[b * block_cnt + i].second);
                    contexts.push_back(&curContextLayer[b]);
                }
                MatMulBatch(attns, values, contexts);
            } else {
                for (int b = 0; b < batch; b++) {
                    Data &pastValue = *pastKeyValues[b * block_cnt + i].second;
                    MatMul(attnProbs[b], pastValue, curContextLayer[b]);
                }
            }

            for (int b = 0; b < batch; b++) {
                curContextLayer[b].Reshape(outputSizes[b]);
                PermuteSelf(curContextLayer[b], {2, 0, 1, 3});
                curContextLayer[b].Reshape({curContextLayer[b].dims[0], curContextLayer[b].dims[1], embed_dim});
            }

            if (all1 && batch > 1) {
                std::vector <Data*> contexts;
                for (int b = 0; b < batch; b++) {
                    contexts.push_back(&curContextLayer[b]);
                }
                CatBatch(contexts, 0, contextLayer);
            } else {
                if (batch > 1) {
                    for (int b = 0; b < batch; b++) {
                        if (contextLayer.dims.size() == 0) {
                            std::vector<int> dims = curContextLayer[b].dims;
                            dims[0] = total;
                            contextLayer.Expansion(dims);
                        }
                        contextLayer.ToDevice(DataDevice::CUDA);
                        CatDirect(contextLayer, curContextLayer[b], 0);
                    }
                }
            }

            // 1.2.4 dense
            std::string denseWeightName = weightPre + std::to_string(i) + weightMiddle + ".dense.weight";
            std::string denseBiasName = weightPre + std::to_string(i) + weightMiddle + ".dense.bias";
            Linear(batch == 1 ? curContextLayer[0] : contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);
            if (GetVersion() == 1) {
                float alpha = sqrt(2 * block_cnt);
                Mul(attenInput, alpha, hiddenStates);
                AddTo(hiddenStates, attnOutput);
                std::string postLNWeightName =
                        "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                std::string postLNBiasName =
                        "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.bias";
                LayerNorm(hiddenStates, weight[postLNWeightName], weight[postLNBiasName], -1, mlpInput);
                // 1.4 MLP
                std::string fcInKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                std::string fcOutKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
                Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                GeluNew(middle, middle);
                Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
                AddTo(hiddenStates, mlpInput, alpha);
            } else {
                AddTo(hiddenStates, attnOutput);
                std::string postRMSWeightName =
                        "transformer.encoder.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                Data temp;
                Mul(hiddenStates, 1.0, temp);
                RMSNorm(hiddenStates, weight[postRMSWeightName], 1e-5, mlpInput);
                // 1.4 MLP
                std::string fcInKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                std::string fcOutKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
                Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                Swiglu(middle, middle2);
                Linear(middle2, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
                AddTo(hiddenStates, temp);
            }
        }
        Data logits;
        if (version == 1) {
            LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"],
                      weight["transformer.final_layernorm.bias"], -1, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        } else {
            RMSNorm(hiddenStates, weight["transformer.encoder.final_layernorm.weight"], 1e-5, hiddenStates);
            Linear(hiddenStates, weight["transformer.output_layer.weight"], Data(), logits);
        }
        std::vector <int> lastRet;
        int total = 0;
        for (int b = 0; b < batch; b++) {
            Data curLogit;
            Split(logits, 0, total + seqLens[b] - 1, total + seqLens[b], curLogit);
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

    std::string ChatGLMModel::Response(const std::string& input, RuntimeResult retCb,
                                       const GenerationConfig &generationConfig) {
        int gmask_token_id = this->weight.dicts.find("gmask_token_id") != this->weight.dicts.end() ?
                             atoi(this->weight.dicts["gmask_token_id"].c_str()) : 130001;
#ifdef USE_CUDA
        FastllmCudaClearBigBuffer();
#endif
        Data inputIds = this->weight.tokenizer.Encode(input);
        std::vector <float> ids;
        for (int i = 0; i < inputIds.Count(0); i++) {
            ids.push_back(((float*)inputIds.cpuData)[i]);
        }
        if (GetVersion() == 1) {
            ids.push_back(gmask_token_id);
            ids.push_back(bos_token_id);
        } else if (GetVersion() == 2) {
            ids.insert(ids.begin(), 64792);
            ids.insert(ids.begin(), 64790);
        }

        int seqLen = ids.size();
        inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, ids));

        std::vector <float> vmask = std::vector <float> (seqLen * seqLen, 0);
        std::vector <float> vpids = std::vector <float> (seqLen * 2, 0);
        for (int i = 0; i < seqLen - 1; i++) {
            vmask[i * seqLen + seqLen - 1] = 1;
            vpids[i] = i;
        }
        vpids[seqLen - 1] = seqLen - 2;
        vpids[seqLen * 2 - 1] = 1;

        if (GetVersion() == 2) {
            for (int i = 0; i < seqLen; i++) {
                vpids[i] = i;
                for (int j = i + 1; j < seqLen; j++) {
                    vmask[i * seqLen + j] = 1;
                }
            }
        }
        Data attentionMask = Data(DataType::FLOAT32, {seqLen, seqLen}, vmask);
        Data positionIds = Data(DataType::FLOAT32, {2, seqLen}, vpids);

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }

        std::string retString = "";
        int len = 1, maskIds = -1;
        std::vector <float> results;
		int index = 0;
        LastTokensManager tokens (1, generationConfig.last_n);
        while (true) {
            auto st = std::chrono::system_clock::now();
            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
            tokens.units[0].Push(ret);
            if (ret == eos_token_id) {
                break;
            }

            results.push_back(ret);
            std::string curString = weight.tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results)).c_str();
            retString += curString;
			if (retCb)
#ifdef PY_API
				retCb(index, pybind11::bytes(retString));
#else
				retCb(index, curString.c_str());
#endif
            index++;
            fflush(stdout);
            results.clear();

            len++;
            if (maskIds == -1) {
                maskIds = (int)ids.size() - (GetVersion() == 1 ? 2 : 0);
            }

            attentionMask.ToDevice(DataDevice::CPU);
            positionIds.ToDevice(DataDevice::CPU);
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)ret}));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {2, 1}, {(float)maskIds, (float)(len)}));

            if (GetVersion() == 2) {
                maskIds++;
            }

            if (index == generationConfig.output_token_limit) {
                break;
            }
             // printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));
        }
		if (retCb)
#ifdef PY_API
			retCb(-1, pybind11::bytes(retString));
#else
			retCb(-1, retString.c_str());
#endif
        return retString;
    }

    void ChatGLMModel::ResponseBatch(const std::vector <std::string> &inputs,
                               std::vector <std::string> &outputs,
                               RuntimeResultBatch retCb,
                               const GenerationConfig &generationConfig) {
#ifdef USE_CUDA
        FastllmCudaClearBigBuffer();
#endif
        int gmask_token_id = this->weight.dicts.find("gmask_token_id") != this->weight.dicts.end() ?
                             atoi(this->weight.dicts["gmask_token_id"].c_str()) : 130001;
        // 1. first
        int batch = inputs.size();
        outputs.clear();
        outputs.resize(batch, "");

        std::vector <Data> inputTokens;
        std::vector <int> seqLens;
        inputTokens.resize(batch);
        seqLens.resize(batch);
        int maxLen = 0;
        for (int i = 0; i < batch; i++) {
            inputTokens[i].CopyFrom(this->weight.tokenizer.Encode(inputs[i]));
            maxLen = std::max(maxLen, (int)inputTokens[i].Count(0) + 2);
            seqLens[i] = (int)inputTokens[i].Count(0);
        }

        std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
        std::vector <float> vpids = std::vector <float> (batch * 2 * maxLen, 0);
        std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
        for (int i = 0; i < batch; i++) {
            if (GetVersion() == 1) {
                Data &tokens = inputTokens[i];
                int len = tokens.Count(0), base = maxLen - 2 - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = ((float *) tokens.cpuData)[j];
                }
                ids[i * maxLen + base + len] = gmask_token_id;
                ids[i * maxLen + base + len + 1] = bos_token_id;
                len += 2;
                for (int j = 0; j < len - 1; j++) {
                    vpids[i * 2 * maxLen + base + j] = j;
                }
                vpids[i * 2 * maxLen + base + len - 1] = len - 2;
                vpids[i * 2 * maxLen + maxLen + base + len - 1] = 1;
                std::fill(vmask.data() + i * maxLen * maxLen,
                          vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                for (int j = maxLen - len; j < maxLen; j++) {
                    std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                              vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                }
                for (int j = 0; j < len - 1; j++) {
                    vmask[i * maxLen * maxLen + (base + j) * maxLen + base + len - 1] = 1;
                }
            } else {
                Data &tokens = inputTokens[i];
                int len = tokens.Count(0), base = maxLen - 2 - len;
                ids[i * maxLen + base] = 64790;
                ids[i * maxLen + base + 1] = 64792;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + 2 + j] = ((float*)tokens.cpuData)[j];
                }
                len += 2;
                for (int j = 0; j < len; j++) {
                    vpids[i * 2 * maxLen + base + j] = j;
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
        }

        Data inputIds = Data(DataType::FLOAT32, {batch, maxLen}, ids);
        Data attentionMask = Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask);
        Data positionIds = Data(DataType::FLOAT32, {batch * 2, maxLen}, vpids);

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }

        int len = 1;
        std::vector <int> maskIds = std::vector <int> (batch, -1);
        std::vector <bool> isEnding = std::vector <bool> (batch, false);
        int index = 0;
        LastTokensManager tokensManager (batch, generationConfig.last_n);
        while (true) {
            auto st = std::chrono::system_clock::now();
            //ClearProfiler();
            std::vector <int> ret = ForwardBatch(batch, inputIds, attentionMask, positionIds, pastKeyValues,
                                                 generationConfig, tokensManager);
            //PrintProfiler();
            for (int i = 0; i < batch; i++) {
                tokensManager.units[i].Push(ret[i]);
            }
            std::vector <float> fret;
            std::vector <float> results;
            int endingCount = 0;
            std::vector <std::string> curStrings;
            for (int i = 0; i < batch; i++) {
                fret.push_back(ret[i]);
                if (ret[i] == eos_token_id) {
                    isEnding[i] = true;
                }
                if (isEnding[i]) {
                    curStrings.push_back("");
                    endingCount++;
                    continue;
                }
                results.push_back(ret[i]);
                std::string curString = weight.tokenizer.Decode(
                        Data(DataType::FLOAT32, {(int) results.size()}, results)).c_str();
                outputs[i] += curString;
                curStrings.push_back(curString);
                results.clear();

                if (maskIds[i] == -1) {
                    maskIds[i] = seqLens[i] + (GetVersion() == 1 ? 0 : 2);
                }
            }

            if (endingCount == batch) {
                break;
            }
            if (retCb)
                retCb(index, curStrings);
            index++;
            len++;
            std::vector <float> pids = std::vector <float> (batch * 2);
            for (int i = 0; i < batch; i++) {
                pids[i * 2] = maskIds[i];
                pids[i * 2 + 1] = len;

                if (GetVersion() == 2) {
                    maskIds[i]++;
                }
            }
            maxLen++;
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                seqLens[i]++;
                for (int j = 0; j < maxLen - seqLens[i] - 2; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }
            positionIds.ToDevice(DataDevice::CPU);
            attentionMask.ToDevice(DataDevice::CPU);
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch * 2, 1}, pids));

            // printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));

            if (index == generationConfig.output_token_limit) {
                break;
            }
        }

        if (retCb)
            retCb(-1, outputs);
    }

    void ChatGLMModel::WarmUp() {
    	printf("Warmup...\n");
	    Data inputIds = Data(DataType::FLOAT32, {1, 1}, {(float)bos_token_id});
	    Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
	    Data positionIds = Data(DataType::FLOAT32, {2, 1}, {0, 0});

	    std::vector <std::pair <Data, Data> > pastKeyValues;
	    for (int i = 0; i < block_cnt; i++) {
		    pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
		                                           Data(DataType::FLOAT32)));
	    }
	    Forward(inputIds, attentionMask, positionIds, pastKeyValues);
	    printf("finish.\n");
    }

    std::string ChatGLMModel::MakeInput(const std::string &history, int round, const std::string &input) {
        if (round == 0 && GetVersion() == 1) {
            return input;
        } else {
            return history + ("[Round " + std::to_string(round) + "]\n问：" + input + "\n答：");
        }
    }

    std::string ChatGLMModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (history + ("[Round " + std::to_string(round) + "]\n问：" + input + "\n答：" + output + "\n"));
    }

    int ChatGLMModel::LaunchResponseTokens(const std::vector<int> &inputTokens,
                                           const GenerationConfig &generationConfig) {
        mainLoopLocker.lock();
        if (mainLoop == nullptr) {
            if (mainLoop == nullptr) {
                mainLoop = new std::thread([](ChatGLMModel *model) {
                    while (true) {
                        std::vector <Data*> attentionMasks;
                        std::vector <Data*> positionIds;
                        std::vector <std::pair <Data*, Data*> > pastKeyValues;
                        std::vector <float> ids;
                        std::vector <int> seqLens;
                        std::vector <int> handles;
                        std::vector <GenerationConfig> generationConfigs;
                        LastTokensManager tokensManager;
                        model->dictLocker.lock();
                        for (auto &it: model->responseContextDict.dicts) {
                            if (it.second->isEnding) {
                                continue;
                            }
                            generationConfigs.push_back(it.second->generationConfig);
                            tokensManager.units.push_back(it.second->tokens);
                            handles.push_back(it.first);
                            for (int i = 0; i < it.second->currentTokens.size(); i++) {
                                ids.push_back(it.second->currentTokens[i]);
                            }
                            if (it.second->preTokens == 0) {
                                int seqLen = it.second->currentTokens.size();
                                if (model->GetVersion() == 1) {
                                    int gmask_token_id =
                                            model->weight.dicts.find("gmask_token_id") != model->weight.dicts.end() ?
                                            atoi(model->weight.dicts["gmask_token_id"].c_str()) : 130001;
                                    if (it.second->currentTokens.size() < 2 ||
                                        it.second->currentTokens.back() != model->bos_token_id) {
                                        ids.push_back(gmask_token_id);
                                        ids.push_back(model->bos_token_id);
                                        seqLen += 2;
                                    }
                                } else {
                                    if (it.second->currentTokens.size() < 2 ||
                                        it.second->currentTokens[0] != 64790) {
                                        ids.insert(ids.begin() + (ids.size() - it.second->currentTokens.size()), 64790);
                                        ids.insert(ids.begin() + (ids.size() - it.second->currentTokens.size()), 64792);
                                        seqLen += 2;
                                    }
                                }

                                seqLens.push_back(seqLen);
                                std::vector<float> vmask = std::vector<float>(seqLen * seqLen, 0);
                                std::vector<float> vpids = std::vector<float>(seqLen * 2, 0);
                                for (int i = 0; i < seqLen - 1; i++) {
                                    vmask[i * seqLen + seqLen - 1] = 1;
                                    vpids[i] = i;
                                }
                                vpids[seqLen - 1] = seqLen - 2;
                                vpids[seqLen * 2 - 1] = 1;

                                if (model->GetVersion() == 2) {
                                    for (int i = 0; i < seqLen; i++) {
                                        vpids[i] = i;
                                        for (int j = i + 1; j < seqLen; j++) {
                                            vmask[i * seqLen + j] = 1;
                                        }
                                    }
                                }

                                it.second->intParams["maskIds"] = seqLen - (model->GetVersion() == 1 ?  2 : 0);
                                it.second->intParams["len"] = 1;

                                attentionMasks.push_back(new Data(DataType::FLOAT32, {seqLen, seqLen}, vmask));
                                positionIds.push_back(new Data(DataType::FLOAT32, {2, seqLen}, vpids));
                            } else {
                                seqLens.push_back(1);
                                it.second->intParams["len"]++;
                                attentionMasks.push_back(nullptr);
                                positionIds.push_back(new Data(DataType::FLOAT32, {2, 1}, {(float)it.second->intParams["maskIds"], (float)(it.second->intParams["len"])}));
                                if (model->GetVersion() == 2) {
                                    it.second->intParams["maskIds"]++;
                                }
                            }

                            it.second->preTokens += seqLens.back();
                            for (int i = 0; i < model->block_cnt; i++) {
                                pastKeyValues.push_back(std::make_pair(&it.second->pastKeyValues[i].first,
                                                                       &it.second->pastKeyValues[i].second));
                            }
                        }

                        if (seqLens.size() > 0) {
                            std::vector <std::pair <Data, Data> > *pastKeyValue1;
                            if (seqLens.size() == 1) {
                                pastKeyValue1 = &model->responseContextDict.dicts[handles[0]]->pastKeyValues;
                            }
                            model->dictLocker.unlock();
#ifdef USE_CUDA
                            FastllmCudaClearBigBuffer();
#endif
                            Data inputIds = Data(DataType::FLOAT32, {1, (int) ids.size()}, ids);
//auto st = std::chrono::system_clock::now();
//ClearProfiler();
                            std::vector<int> ret;
                            if (seqLens.size() > 1) {
                                ret = model->ForwardBatch(seqLens.size(), inputIds, attentionMasks,
                                                          positionIds, seqLens, pastKeyValues, generationConfigs,
                                                          tokensManager);
                            } else {
                                ret = std::vector <int> {model->Forward(inputIds,
                                                         attentionMasks[0] == nullptr ? Data() : *attentionMasks[0],
                                                         *positionIds[0],
                                                         *pastKeyValue1, generationConfigs[0], tokensManager)};
                            }
//PrintProfiler();
//printf("%d spend %f\n", ids.size(), GetSpan(st, std::chrono::system_clock::now()));
                            model->dictLocker.lock();
                            for (int i = 0; i < handles.size(); i++) {
                                auto &it = *model->responseContextDict.dicts.find(handles[i]);
                                int curRet = ret[i];
                                if (curRet == model->eos_token_id) {
                                    it.second->isEnding = true;
                                } else {
                                    it.second->currentTokens = std::vector<int>{curRet};
                                    it.second->resultTokenQueue.push(curRet);
                                    it.second->tokens.Push(curRet);
                                    it.second->curTokens++;
                                    if (it.second->curTokens == it.second->generationConfig.output_token_limit) {
                                        it.second->isEnding = true;
                                    }
                                }
                            }
                        }

                        for (int i = 0; i < attentionMasks.size(); i++) {
                            delete attentionMasks[i];
                        }
                        for (int i = 0; i < positionIds.size(); i++) {
                            delete positionIds[i];
                        }

                        model->dictLocker.unlock();
                        MySleep(0);
                    }
                }, this);
            }
        }
        mainLoopLocker.unlock();

        dictLocker.lock();
        int handleId = responseContextDict.CreateHandle();
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        context->Init(this->block_cnt);
        context->currentTokens = inputTokens;
        context->generationConfig = generationConfig;
        context->tokens = LastTokensUnit(generationConfig.last_n);
        dictLocker.unlock();
        return handleId;
    }

    int ChatGLMModel::FetchResponseTokens(int handleId) {
        dictLocker.lock();
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            dictLocker.unlock();
            return -1;
        } else {
            while (true) {
                if (context->resultTokenQueue.size() > 0) {
                    int ret = context->resultTokenQueue.front();
                    context->resultTokenQueue.pop();
                    dictLocker.unlock();
                    return ret;
                } else {
                    if (context->isEnding) {
                        responseContextDict.RemoveHandle(handleId);
                        dictLocker.unlock();
                        return -1;
                    }
                }
                dictLocker.unlock();
                MySleep(0);
                dictLocker.lock();
            }
        }
    }

    int ChatGLMModel::GetVersion() {
        if (this->weight.weight.find("transformer.embedding.word_embeddings.weight") != this->weight.weight.end()) {
            return 2;
        } else {
            return 1;
        }
    }
}
