//
// Created by huangyuyang on 5/11/23.
//

#include "utils.h"

#include "chatglm.h"

#include <cmath>

#include <chrono>

#include <algorithm>

#include <map>

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    void ChatGLMModel::UpdateRotaryPosEmb(float rope_factor) {
        if (rope_factor == this->rope_factor) {
            return;
        }
        this->rope_factor = rope_factor;
        sin.resize(max_positions);
        cos.resize(max_positions);
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            int base = this->bot_role.empty() ? 10000 : 10000 * this->rope_factor;
            invFreq.push_back(1.0 / pow(base, (float)i / rotary_dim));
        }
        for (int i = 0; i < max_positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size(); j++) {
                float scale = this->bot_role.empty() ? rope_factor : 1.0f;
                sin[i][j] = ::sin((float)i / scale * invFreq[j]);
                cos[i][j] = ::cos((float)i / scale * invFreq[j]);
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
    }

    ChatGLMModel::ChatGLMModel() {
        this->model_type = "chatglm";

        this->bos_token_id = 130004;    // V1 后期版本 bos token，可通过 config.json 覆盖
        this->eos_token_id = 130005;    // V1 后期版本 eos token，可通过 config.json 覆盖
        this->gmask_token_id= 150001;   // V1最初版本, 150528 tokens，部分 config.json 没有 gmask_token_id，因此取默认值。

        this->rope_factor = -1.0;
        this->UpdateRotaryPosEmb(1.0f);
        weight.embeddingNames.insert("transformer.word_embeddings.weight");
        weight.embeddingNames.insert("transformer.embedding.word_embeddings.weight");

        weight.linearNames = {
            "*.query_key_value.weight", "*.dense.weight",
            "*.mlp.dense_h_to_4h.weight", "*.mlp.dense_4h_to_h.weight",
            "lm_head.weight", "transformer.output_layer.weight"
        };
    }

    void ChatGLMModel::InitParams() {
        basellm::InitParams();
        if (GetVersion() == 1) {
            if (this->weight.dicts.find("gmask_token_id") != this->weight.dicts.end()) {
                this->gmask_token_id = atoi(this->weight.dicts["gmask_token_id"].c_str());
            }
        } else if (GetVersion() == 2 && this->tokenizerClass != "ChatGLM4Tokenizer") {
            this->gmask_token_id = 64790;
            this->bos_token_id = 64792;
        }
        if (this->weight.dicts.find("rope_ratio") != this->weight.dicts.end()) {            
            UpdateRotaryPosEmb(atof(this->weight.dicts["rope_ratio"].c_str()));
        }
        if (this->weight.dicts.find("layernorm_epsilon") != this->weight.dicts.end()) {
            this->layernorm_epsilon = atof(this->weight.dicts["layernorm_epsilon"].c_str());
        }
        if (this->weight.dicts.find("seq_length") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["seq_length"].c_str());
        }
    }

    int ChatGLMModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                              const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                              const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                              std::vector <float> *logits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(logits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> ChatGLMModel::ForwardBatch(
            int batch,
            const Data &inputIds,
            const Data &attentionMaskOri,
            const Data &positionIdsOri,
            std::vector <std::pair <Data, Data> > &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
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
        Data inputIdsPermute, curInputIdsPermute;
        Permute(inputIds, {1, 0}, inputIdsPermute);

        int seqLen = inputIdsPermute.dims[0];
        int perSeqLen = std::min(seqLen, 1024);

        for (int seqStart = 0; seqStart < seqLen; seqStart += perSeqLen) {
            Data inputEmbeddings;
            int curSeqLen = std::min(perSeqLen, seqLen - seqStart);
            Data *positionIdsPointer, *attentionMaskPointer;
            Data positionIdsNew, attentionMaskNew;
            if (curSeqLen == seqLen) {
                Embedding(inputIdsPermute, this->weight["transformer" + std::string((version == 2 ? ".embedding" : "")) +
                                                    ".word_embeddings.weight"], inputEmbeddings);
                attentionMaskPointer = (Data*)&attentionMaskOri;
                positionIdsPointer = (Data*)&positionIdsOri;
            } else {
                std::vector <float> curMasks = std::vector <float> (curSeqLen * (seqStart + curSeqLen));
                int idx = 0;
                for (int i = 0; i < curSeqLen; i++) {
                    for (int j = 0; j < seqStart + curSeqLen; j++) {
                        curMasks[idx++] = (seqStart + i) < j;
                    }
                }
                attentionMaskNew.CopyFrom(Data(FLOAT32, {curSeqLen, seqStart + curSeqLen}, curMasks));
                ToDataType(attentionMaskNew, this->dataType);
                Split(positionIdsOri, -1, seqStart, seqStart + curSeqLen, positionIdsNew);
                positionIdsPointer = &positionIdsNew;
                attentionMaskPointer = &attentionMaskNew;

                Split(inputIdsPermute, 0, seqStart, seqStart + curSeqLen, curInputIdsPermute);
                Embedding(curInputIdsPermute, this->weight["transformer" + std::string((version == 2 ? ".embedding" : "")) +
                                                    ".word_embeddings.weight"], inputEmbeddings);
            }

            Data &positionIds = *positionIdsPointer;
            Data &attentionMask = *attentionMaskPointer;

            ToDataType(inputEmbeddings, this->dataType);

            Data &hiddenStates = inputEmbeddings;
            for (int i = 0; i < block_cnt; i++) {
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                if (version == 1) {
                    std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
                    std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
                    LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
                } else if (version == 2) {
                    std::string inputRMSWeightName =
                            "transformer.encoder.layers." + std::to_string(i) + ".input_layernorm.weight";
                    RMSNorm(hiddenStates, weight[inputRMSWeightName], layernorm_epsilon, attenInput);
                }
                std::string qkvWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.weight";
                std::string qkvBiasName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.bias";
                if (!adapterName.empty()) {
                    std::string peftType = weight.peftDict[adapterName]["peft_type"];
                    if (peftType == "LORA") {
                        std::string loraAWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.lora_A." + adapterName + ".weight";
                        std::string loraBWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.lora_B." + adapterName + ".weight";
                        LoraLayer(attenInput, weight[qkvWeightName], weight[loraAWeightName], weight[loraBWeightName], weight[qkvBiasName], qkv, weight.peftDict[adapterName]);
                    } else if (peftType == "IA3") {
                        std::string ia3WeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.ia3_l" + adapterName + ".weight";
                        IA3Layer(attenInput, weight[qkvWeightName], weight[ia3WeightName], weight[qkvBiasName], qkv, weight.peftDict[adapterName]);
                    }
                } else {
                    Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);
                }
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
                            newDims[1] = k.dims[1] + generationConfig.output_token_limit;
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
                            newDims[1] = k.dims[1] + generationConfig.output_token_limit;
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
                Attention(q, pastKey, pastValue, attentionMask, contextLayer, q.dims[0] / pastKey.dims[0], 1.0 / scale_attn, 1);
    /*
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
    */

                contextLayer.Reshape({batch, num_attention_heads, curSeqLen, -1});
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
                    RMSNorm(hiddenStates, weight[postRMSWeightName], this->layernorm_epsilon, mlpInput);
                    // 1.4 MLP
                    std::string fcInKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                    std::string fcOutKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";

                    if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                        LinearEx(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle2, LinearExType::ExSwiglu);
                    } else {
                        Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                        Swiglu(middle, middle2);
                    }

                    Linear(middle2, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
                    AddTo(hiddenStates, temp);
                }
            }
            
            if (seqStart + curSeqLen < seqLen) {
                continue;
            }

            Data logits, topk;
            Data tempHiddenStates;
            Data *lastHiddenStates;
            if (curSeqLen > 1) {
                Split(hiddenStates, 0, curSeqLen - 1, curSeqLen, tempHiddenStates);
                lastHiddenStates = &tempHiddenStates;
            } else {
                lastHiddenStates = &hiddenStates;
            }

            {
                auto &hiddenStates = *lastHiddenStates;
                if (version == 1) {
                    LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"],
                            weight["transformer.final_layernorm.bias"], -1, hiddenStates);
                    Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
                } else {
                    RMSNorm(hiddenStates, weight["transformer.encoder.final_layernorm.weight"], this->layernorm_epsilon, hiddenStates);
                    Linear(hiddenStates, weight["transformer.output_layer.weight"], Data(), logits);
                }

                ToDataType(logits, DataType::FLOAT32);
//logits.ToDevice(DataDevice::CPU);
//logits.Print();
                if (generationConfig.output_logits && retLogits != nullptr) {
                    int size = logits.dims.back();
                    logits.ToDevice(DataDevice::CPU);
                    for (int b = 0; b < batch; b++) {
                        int base = b;
                        (*retLogits)[b]->resize(size);
                        memcpy((float *) (*retLogits)[b]->data(), ((float *) logits.cpuData) + base * size,
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
                } else if (!lastTokens.units.empty()) {
                    for (int b = 0; b < batch; b++) {
                        int base = b;
                        lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
                    }
                }
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
            const LastTokensManager &lastTokens,
            std::vector <std::vector <float>*> *retLogits) {
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
        ToDataType(inputEmbeddings, this->dataType);
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
        std::vector <Data*> keys, values, qs, attns, masks, contexts;
        keys.resize(batch);
        values.resize(batch);
        qs.resize(batch);
        attns.resize(batch);
        masks.resize(batch);
        contexts.resize(batch);

        std::vector <Data*> pointersK, pointersV, pointersQ;
        pointersK.resize(batch);
        pointersV.resize(batch);
        pointersQ.resize(batch);

        std::vector <std::vector <int> > outputSizes;
        outputSizes.resize(batch);
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            if (version == 1) {
                std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
                std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
                LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
            } else if (version == 2) {
                std::string inputRMSWeightName =
                        "transformer.encoder.layers." + std::to_string(i) + ".input_layernorm.weight";
                RMSNorm(hiddenStates, weight[inputRMSWeightName], this->layernorm_epsilon, attenInput);
            }

            std::string qkvWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.weight";
            std::string qkvBiasName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.bias";
            if (!adapterName.empty()) {
                std::string peftType = weight.peftDict[adapterName]["peft_type"];
                if (peftType == "LORA") {
                    std::string loraAWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.lora_A." + adapterName + ".weight";
                    std::string loraBWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.lora_B." + adapterName + ".weight";
                    LoraLayer(attenInput, weight[qkvWeightName], weight[loraAWeightName], weight[loraBWeightName], weight[qkvBiasName], qkv, weight.peftDict[adapterName]);
                } else if (peftType == "IA3") {
                    std::string ia3WeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.ia3_l" + adapterName + ".weight";
                    IA3Layer(attenInput, weight[qkvWeightName], weight[ia3WeightName], weight[qkvBiasName], qkv, weight.peftDict[adapterName]);
                }
            } else {
                Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);
            }

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

            Data contextLayer = Data(this->dataType);
            int total = 0;
            if (all1 && batch > 1) {
                for (int b = 0; b < batch; b++) {
                    curQs[b].Resize({q.dims[1], 1, q.dims[2]});
                    curQs[b].FakeFrom(q, b * q.strides[0] * q.unitSize);
                    curKs[b].Resize({k.dims[1], 1, k.dims[2]});
                    curKs[b].FakeFrom(k, b * k.strides[0] * k.unitSize);
                    curVs[b].Resize({v.dims[1], 1, v.dims[2]});
                    curVs[b].FakeFrom(v, b * v.strides[0] * v.unitSize);
                }
            } else {
                PermuteSelf(k, {1, 0, 2});
                PermuteSelf(v, {1, 0, 2});
                PermuteSelf(q, {1, 0, 2});
                for (int b = 0; b < batch; b++) {
                    Split(k, 1, total, total + seqLens[b], curKs[b]);
                    Split(v, 1, total, total + seqLens[b], curVs[b]);
                    Split(q, 1, total, total + seqLens[b], curQs[b]);
                    total += seqLens[b];
                }
            }

            for (int b = 0; b < batch; b++) {
                auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt +
                                                                                                     i].second;
                if (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] <= pastKey.expansionDims[1]) {
                    continue;
                }

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
            for (int b = 0; b < batch; b++) {
                keys[b] = (pastKeyValues[b * block_cnt + i].first);
                values[b] = (pastKeyValues[b * block_cnt + i].second);
                pointersK[b] = (&curKs[b]);
                pointersV[b] = (&curVs[b]);
            }
            CatDirectBatch(keys, pointersK, 1);
            CatDirectBatch(values, pointersV, 1);
            if (all1 && batch > 1) {
                for (int b = 0; b < batch; b++) {
                    qs[b] = (&curQs[b]);
                    keys[b] = (pastKeyValues[b * block_cnt + i].first);
                    values[b] = (pastKeyValues[b * block_cnt + i].second);
                    masks[b] = attentionMask[b];
                    contexts[b] = (&curContextLayer[b]);

                    outputSizes[b] = {1, qs[b]->dims[0], qs[b]->dims[1], keys[b]->dims[1]};
                }
                AttentionBatch(qs, keys, values, masks, contexts, qs[0]->dims[0] / values[0]->dims[0], 1.0 / scale_attn, 1);
            } else {
                for (int b = 0; b < batch; b++) {
                    auto &q = curQs[b];
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first;
                    outputSizes[b] = {1, q.dims[0], q.dims[1], pastKey.dims[1]};
                    q.Reshape({pastKey.dims[0], -1, q.dims[2]});
                }

                // 1.2 Attention
                // 1.2.0 q * k^T
                if (all1 && batch > 1) {
                    for (int b = 0; b < batch; b++) {
                        qs[b] = (&curQs[b]);
                        keys[b] = (pastKeyValues[b * block_cnt + i].first);
                        attns[b] = (&attnProbs[b]);
                    }
                    MatMulTransBBatch(qs, keys, attns, 1.0 / (scale_attn * (i + 1)));
                } else {
                    for (int b = 0; b < batch; b++) {
                        auto &q = curQs[b];
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
                for (int i = 0; i < attnProbs.size(); i++) {
                    attns[i] = (&attnProbs[i]);
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
                    for (int b = 0; b < batch; b++) {
                        attns[b] = (&attnProbs[b]);
                        values[b] = (pastKeyValues[b * block_cnt + i].second);
                        contexts[b] = (&curContextLayer[b]);
                    }
                    MatMulBatch(attns, values, contexts);
                } else {
                    for (int b = 0; b < batch; b++) {
                        Data &pastValue = *pastKeyValues[b * block_cnt + i].second;
                        MatMul(attnProbs[b], pastValue, curContextLayer[b]);
                    }
                }
            }
            if (all1) {
                for (int b = 0; b < batch; b++) {
                    curContextLayer[b].dims[0] = outputSizes[b][2];
                    curContextLayer[b].dims[1] = outputSizes[b][0];
                    curContextLayer[b].dims[2] = embed_dim;
                    curContextLayer[b].strides[0] = curContextLayer[b].dims[1] * curContextLayer[b].dims[2];
                    curContextLayer[b].strides[1] = curContextLayer[b].dims[2];
                    curContextLayer[b].strides[2] = 1;
                }
            } else {
                for (int b = 0; b < batch; b++) {
                    curContextLayer[b].Reshape(outputSizes[b]);
                    PermuteSelf(curContextLayer[b], {2, 0, 1, 3});
                    curContextLayer[b].Reshape({curContextLayer[b].dims[0], curContextLayer[b].dims[1], embed_dim});
                }
            }

            if (all1 && batch > 1) {
                for (int b = 0; b < batch; b++) {
                    contexts[b] = (&curContextLayer[b]);
                }
                CatBatch(contexts, 0, contextLayer);
            } else {
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
            // 1.2.4 dense
            std::string denseWeightName = weightPre + std::to_string(i) + weightMiddle + ".dense.weight";
            std::string denseBiasName = weightPre + std::to_string(i) + weightMiddle + ".dense.bias";
            Linear(contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);
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
                RMSNorm(hiddenStates, weight[postRMSWeightName], this->layernorm_epsilon, mlpInput);
                // 1.4 MLP
                std::string fcInKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
                std::string fcOutKeyName = "transformer.encoder.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
                if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                    LinearEx(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle2, LinearExType::ExSwiglu);
                } else {
                    Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
                    Swiglu(middle, middle2);
                }

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
            RMSNorm(hiddenStates, weight["transformer.encoder.final_layernorm.weight"], this->layernorm_epsilon, hiddenStates);
            Linear(hiddenStates, weight["transformer.output_layer.weight"], Data(), logits);
        }
        ToDataType(logits, DataType::FLOAT32);
        std::vector <int> lastRet;
        int total = 0;
        Data curLogit;
        for (int b = 0; b < batch; b++) {
            Split(logits, 0, total + seqLens[b] - 1, total + seqLens[b], curLogit);
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

    void ChatGLMModel::FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                     const std::map <std::string, int> &params,
                                     Data &inputIds, Data &attentionMask, Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int index = params.find("index")->second;
        int promptLen = params.find("promptLen")->second;
        bool add_special_tokens = params.find("add_special_tokens")->second == 0? false: true;

        if (index == 0) {
            if (add_special_tokens) {
                for (auto &ids: inputTokens) {
                    if (GetVersion() == 1) {
                        ids.push_back(gmask_token_id);
                        ids.push_back(bos_token_id);
                    } else if (GetVersion() == 2) {
                        if (ids.size() < 2 || ids[0] != this->gmask_token_id || ids[1] != this->bos_token_id) {
                            ids.insert(ids.begin(), this->bos_token_id);
                            ids.insert(ids.begin(), this->gmask_token_id);
                        }
                    }
                }
            }


            int seqLen = inputTokens[0].size();
            std::vector<float> vpids = std::vector<float>(seqLen * 2, 0);
            for (int i = 0; i < seqLen - 1; i++) {
                vpids[i] = i;
            }
            vpids[seqLen - 1] = seqLen - 2;
            vpids[seqLen * 2 - 1] = 1;

            if (GetVersion() == 2) {
                for (int i = 0; i < seqLen; i++) {
                    vpids[i] = i;
                }
            }

            if (seqLen <= 4096) {
                std::vector<float> vmask = std::vector<float>(seqLen * seqLen, 0);
                for (int i = 0; i < seqLen - 1; i++) {
                    vmask[i * seqLen + seqLen - 1] = 1;
                }
                if (GetVersion() == 2) {
                    for (int i = 0; i < seqLen; i++) {
                        for (int j = i + 1; j < seqLen; j++) {
                            vmask[i * seqLen + j] = 1;
                        }
                    }
                }
                attentionMask.CopyFrom(Data(DataType::FLOAT32, {seqLen, seqLen}, vmask));
            } else {
                attentionMask = Data();
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, inputTokens[0]));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {2, seqLen}, vpids));
        } else {
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, inputTokens[0]));
            attentionMask = Data();
            if (GetVersion() == 1) {
                positionIds.CopyFrom(Data(DataType::FLOAT32, {2, 1}, {(float) promptLen, (float) (index + 1)}));
            } else {
                positionIds.CopyFrom(Data(DataType::FLOAT32, {2, 1}, {(float) promptLen + index + 1, (float) (index + 1)}));
            }
        }
    }

    void ChatGLMModel::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        bool add_special_tokens = params[0].find("add_special_tokens")->second == 0? false: true;
        int special_tokens_offset = 0;
        if (add_special_tokens) {
            special_tokens_offset = 2;
        }
        if (index == 0) {
            std::vector<int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int) inputTokens[i].size() + special_tokens_offset);
                seqLens[i] = (int) inputTokens[i].size();
            }

            std::vector<float> ids = std::vector<float>(batch * maxLen, 0);
            std::vector<float> vpids = std::vector<float>(batch * 2 * maxLen, 0);
            std::vector<float> vmask = std::vector<float>(batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                if (GetVersion() == 1) {
                    auto &tokens = inputTokens[i];
                    int len = tokens.size(), base = maxLen - special_tokens_offset - len;
                    for (int j = 0; j < len; j++) {
                        ids[i * maxLen + base + j] = tokens[j];
                    }
                    if (add_special_tokens) {
                        ids[i * maxLen + base + len] = gmask_token_id;
                        ids[i * maxLen + base + len + 1] = bos_token_id;
                    }
                    len += special_tokens_offset;
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
                    auto &tokens = inputTokens[i];
                    int len = tokens.size(), base = maxLen - special_tokens_offset - len;
                    if (add_special_tokens) {
                        ids[i * maxLen + base] = gmask_token_id;
                        ids[i * maxLen + base + 1] = bos_token_id;
                    }
                    for (int j = 0; j < len; j++) {
                        ids[i * maxLen + base + special_tokens_offset + j] = tokens[j];
                    }
                    len += special_tokens_offset;
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

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch * 2, maxLen}, vpids));
        } else {
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            std::vector <float> pids = std::vector<float>(batch * 2);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen + 2, maxLen);
                pids[i * 2 + 1] = index + 1;
                if (GetVersion() == 1) {
                    pids[i * 2] = promptLen;
                } else {
                    pids[i * 2] = promptLen + index + 1;
                }
            }
            maxLen += index;
            std::vector<float> vmasks = std::vector<float>(batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                for (int j = 0; j < maxLen - index - promptLen - 2; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch * 2, 1}, pids));
        }
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
        if (this->bot_role != "") {
            return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
        } else {
            if (GetVersion() == 2)
                round++;
            if (round == 0 && GetVersion() == 1) {
                return input;
            } else {
#if defined(_WIN32) or defined(_WIN64)
                return history + ("[Round " + std::to_string(round) + u8"]\n\n问：" + input + u8"\n\n答：");
#else
                return history + ("[Round " + std::to_string(round) + "]\n\n问：" + input + "\n\n答：");
#endif
            }
        }
    }

    std::string ChatGLMModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        if (this->bot_role != "") {
            return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
        }

		if (GetVersion() == 2)
			round++;
#if defined(_WIN32) or defined(_WIN64)
        return (history + ("[Round " + std::to_string(round) + u8"]\n\n问：" + input + u8"\n\n答：" + output + "\n"));
#else
        return (history + ("[Round " + std::to_string(round) + "]\n\n问：" + input + "\n\n答：" + output + "\n\n"));
#endif
    }

    int ChatGLMModel::GetVersion() {
        if (this->weight.weight.find("transformer.embedding.word_embeddings.weight") != this->weight.weight.end()) {
            return 2;
        } else {
            return 1;
        }
    }
}
