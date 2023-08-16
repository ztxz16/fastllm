//
// Created by siemon on 8/9/23.
//

#include "utils.h"

#include "qwen.h"

#include <cmath>

#include <chrono>

#include <algorithm>

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    extern double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2);

    QWenModel::QWenModel() {
        this->model_type = "qwen";
        this->pre_prompt = "You are a helpful assistant.";
        this->user_role = "user";
        this->bot_role = "assistant";

        embed_dim = 4096;
		num_attention_heads = 32;
		head_dim = embed_dim / num_attention_heads;
		block_cnt = 32;
        rotary_dim = 128;
        seq_length = 2048;
        use_log_attn = true;

        ntk_alpha = 1.f;
        UpdateRotaryPosEmb(ntk_alpha);

        if (use_log_attn) {
            logn_list = Data(DataType::FLOAT32);
            logn_list.Resize({1, max_positions, 1, 1});
            logn_list.Allocate();
            float *logn = (float *) logn_list.cpuData;
            for (int i = 0; i < seq_length; i++) {
                logn[i] = 1;
            }
            for (int i = seq_length; i < max_positions; i++) {
                logn[i] = std::log(i) / std::log(seq_length);
            }
        }

        weight.embeddingNames.insert("transformer.wte.weight");
    }

    int QWenModel::Forward(const Data &inputIds,
                           const Data &attentionMask,
                           const Data &positionIds,
                           std::vector <std::pair <Data, Data> > &pastKeyValues,
                           const GenerationConfig &generationConfig,
                           const LastTokensManager &lastTokens,
                           std::vector <float> *logits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(logits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> QWenModel::ForwardBatch(int batch,
                                              const Data &inputIds,
                                              const Data &attentionMask,
                                              const Data &positionIds,
                                              std::vector <std::pair <Data, Data> > &pastKeyValues,
                                              const GenerationConfig &generationConfig,
                                              const LastTokensManager &lastTokens,
                                              std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];                                        
        Data hiddenStates;
        Data attnInput, attnOutput;
        Data query, key, value;
        Data attnWeights, attnLastOutput;
        Data a1, a2, mlpOutput;

        // printf("input id: ");
        // for (int i = 0; i < inputIds.Count(0); i++) {
        //     printf("%d ", (int )((float *) inputIds.cpuData)[i]);
        // }
        // printf("\n");

        Embedding(inputIds, this->weight["transformer.wte.weight"], hiddenStates);
        for (int i = 0; i < this->block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            int seqlen = hiddenStates.dims[1];

            std::string ln_1_name = "transformer.h." + std::to_string(i) + ".ln_1.weight";
            std::string attn_weight_name = "transformer.h." + std::to_string(i) + ".attn.c_attn.weight";
            std::string attn_bias_name = "transformer.h." + std::to_string(i) + ".attn.c_attn.bias";

            RMSNorm(hiddenStates, weight[ln_1_name], 1e-6, attnInput);
            Linear(attnInput, weight[attn_weight_name], weight[attn_bias_name], attnOutput); // attnOutput [batch, seqlen, embed_dim * 3]
            Split(attnOutput, 2, 0, embed_dim, query);
            Split(attnOutput, 2, embed_dim, 2 * embed_dim, key);
            Split(attnOutput, 2, embed_dim * 2, embed_dim * 3, value);

            query.Reshape({query.dims[0], query.dims[1], num_attention_heads, head_dim});
            key.Reshape({key.dims[0], key.dims[1], num_attention_heads, head_dim});
            value.Reshape({value.dims[0], value.dims[1], num_attention_heads, head_dim});

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (pastKey.dims.empty()) {
                // 计算new_ntk_alpha
                float context_value = std::log2((float) seqlen / seq_length) + 1;
                float new_ntk_alpha = std::max(std::pow(2, std::ceil(context_value) - 1), 1.);
                if (new_ntk_alpha != ntk_alpha) {
                    UpdateRotaryPosEmb(new_ntk_alpha);
                }
            }

            LlamaRotatePosition2D(query, positionIds, sinData, cosData, rotary_dim);
            LlamaRotatePosition2D(key, positionIds, sinData, cosData, rotary_dim);

            if (use_log_attn) {
                ApplyLognAttn(query, logn_list, positionIds);
            }

            PermuteSelf(query, {0, 2, 1, 3});
            PermuteSelf(key, {0, 2, 1, 3});
            PermuteSelf(value, {0, 2, 1, 3});

            std::vector<int> qkvSize = {batch * num_attention_heads, seqlen, -1};
            query.Reshape(qkvSize);
            key.Reshape(qkvSize);
            value.Reshape(qkvSize);

            int unitLen = 64;
#ifdef USE_CUDA
            unitLen = 128;
#endif
            while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || key.dims[1] > pastKey.expansionDims[1]))
                   || (pastKey.dims.size() > 0 && pastKey.dims[1] + key.dims[1] > pastKey.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector <int> {key.dims[0], ((key.dims[1] - 1) / unitLen + 1) * unitLen, key.dims[2]};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((key.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || value.dims[1] > pastValue.expansionDims[1]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[1] + value.dims[1] > pastValue.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {value.dims[0], ((value.dims[1] - 1) / unitLen + 1) * unitLen, value.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((value.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }
            CatDirect(pastKey, key, 1);
            CatDirect(pastValue, value, 1);

            // Attention
            MatMulTransB(query, pastKey, attnWeights, 1.0 / sqrt(head_dim));
            attnWeights.Reshape({1, attnWeights.dims[0], attnWeights.dims[1], attnWeights.dims[2]});
            if (!attentionMask.dims.empty()) {
                AttentionMask(attnWeights, attentionMask, -10000);
            }

            Softmax(attnWeights, attnWeights, -1);
            MatMul(attnWeights, pastValue, attnOutput);

            attnOutput.Reshape({attnOutput.dims[1], attnOutput.dims[2], attnOutput.dims[3]});
            PermuteSelf(attnOutput, {1, 0, 2});
            attnOutput.Reshape({seqlen, batch, -1});
            PermuteSelf(attnOutput, {1, 0, 2});

            std::string proj_weight_name = "transformer.h." + std::to_string(i) + ".attn.c_proj.weight";
            Linear(attnOutput, weight[proj_weight_name], Data(), attnLastOutput);
            AddTo(hiddenStates, attnLastOutput);

            std::string ln_2_name = "transformer.h." + std::to_string(i) + ".ln_2.weight";
            RMSNorm(hiddenStates, weight[ln_2_name], 1e-6, attnInput);

            std::string mlp_w1_weight_name = "transformer.h." + std::to_string(i) + ".mlp.w1.weight";
            std::string mlp_w2_weight_name = "transformer.h." + std::to_string(i) + ".mlp.w2.weight";
            std::string mlp_proj_weight_name = "transformer.h." + std::to_string(i) + ".mlp.c_proj.weight";
            Linear(attnInput, weight[mlp_w1_weight_name], Data(), a1);
            Linear(attnInput, weight[mlp_w2_weight_name], Data(), a2);
            Silu(a2, a2);
            MulTo(a1, a2);
            Linear(a1, weight[mlp_proj_weight_name], Data(), mlpOutput);
            AddTo(hiddenStates, mlpOutput);
        }

        RMSNorm(hiddenStates, weight["transformer.ln_f.weight"], 1e-6, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        logits.ToDevice(DataDevice::CPU);

        if (generationConfig.output_logits && retLogits != nullptr) {
            int size = logits.dims.back();
            logits.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b;
                (*retLogits)[b]->resize(size);
                memcpy((float*)(*retLogits)[b]->data(), ((float*)logits.cpuData) + base * size, size * logits.unitSize);
            }
        }

        std::vector <int> lastRet;
        if (generationConfig.IsSimpleGreedy()) {
            for (int b = 0; b < batch; b++) {
                int base = b * logits.dims[1] + logits.dims[1] - 1;
                std::pair <float, int> ret = std::make_pair(-1e9, -1);
                for (int i = 0; i < logits.dims.back(); i++) {
                    ret = max(ret, std::make_pair(((float *) logits.cpuData)[base * logits.dims.back() + i], i));
                }
                lastRet.push_back(ret.second);
            }
        } else {
            for (int b = 0; b < batch; b++) {
                int base = b * logits.dims[1] + logits.dims[1] - 1;
                lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
            }
        }
        
        return lastRet;
    }
    
    std::vector <int> QWenModel::ForwardBatch(int batch,
                                              const Data &inputIds,
                                              const std::vector <Data*> &attentionMask,
                                              const std::vector <Data*> &positionIds,
                                              const std::vector <int> &seqLens,
                                              std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                              const std::vector <GenerationConfig> &generationConfigs,
                                              const LastTokensManager &lastTokens,
                                              std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attnInput, attnOutput;
        Data query, key, value;
        Data attnWeights, attnLastOutput;
        Data a1, a2, mlpOutput;

        Embedding(inputIds, this->weight["transformer.wte.weight"], hiddenStates);
        for (int i = 0; i < this->block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);

            std::string ln_1_name = "transformer.h." + std::to_string(i) + ".ln_1.weight";
            std::string attn_weight_name = "transformer.h." + std::to_string(i) + ".attn.c_attn.weight";
            std::string attn_bias_name = "transformer.h." + std::to_string(i) + ".attn.c_attn.bias";

            RMSNorm(hiddenStates, weight[ln_1_name], 1e-6, attnInput);
            Linear(attnInput, weight[attn_weight_name], weight[attn_bias_name], attnOutput); // attnOutput [batch, seqlen, embed_dim * 3]
            Split(attnOutput, 2, 0, embed_dim, query);
            Split(attnOutput, 2, embed_dim, 2 * embed_dim, key);
            Split(attnOutput, 2, embed_dim * 2, embed_dim * 3, value);

            std::vector<Data> curKs, curVs, curQs;
            curKs.resize(batch);
            curVs.resize(batch);
            curQs.resize(batch);
            int total = 0;
            for (int b = 0; b < batch; b++) {
                Split(query, 1, total, total + seqLens[b], curQs[b]);
                Split(key, 1, total, total + seqLens[b], curKs[b]);
                Split(value, 1, total, total + seqLens[b], curVs[b]);
                total += seqLens[b];
            }

            Data attnOutputAll = Data(DataType::FLOAT32);
            for (int b = 0; b < batch; b++) {
                // in this loop, batch = 1
                auto &query = curQs[b];
                auto &key = curKs[b];
                auto &value = curVs[b];

                query.Reshape({1, seqLens[b], num_attention_heads, head_dim});
                key.Reshape({1, seqLens[b], num_attention_heads, head_dim});
                value.Reshape({1, seqLens[b], num_attention_heads, head_dim});

                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                if (pastKey.dims.empty()) {
                    // 计算new_ntk_alpha
                    float context_value = std::log2((float) seqLens[b] / seq_length) + 1;
                    float new_ntk_alpha = std::max(std::pow(2, std::ceil(context_value) - 1), 1.);
                    if (new_ntk_alpha != ntk_alpha) {
                        UpdateRotaryPosEmb(new_ntk_alpha);
                    }
                }

                LlamaRotatePosition2D(query, *positionIds[b], sinData, cosData, rotary_dim);
                LlamaRotatePosition2D(key, *positionIds[b], sinData, cosData, rotary_dim);

                if (use_log_attn) {
                    ApplyLognAttn(query, logn_list, *positionIds[b]);
                }

                PermuteSelf(query, {0, 2, 1, 3});
                PermuteSelf(key, {0, 2, 1, 3});
                PermuteSelf(value, {0, 2, 1, 3});

                std::vector<int> qkvSize = {num_attention_heads, seqLens[b], -1};
                query.Reshape(qkvSize);
                key.Reshape(qkvSize);
                value.Reshape(qkvSize);

                int unitLen = 64;
    #ifdef USE_CUDA
                unitLen = 128;
    #endif
                while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || key.dims[1] > pastKey.expansionDims[1]))
                    || (pastKey.dims.size() > 0 && pastKey.dims[1] + key.dims[1] > pastKey.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector <int> {key.dims[0], ((key.dims[1] - 1) / unitLen + 1) * unitLen, key.dims[2]};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((key.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || value.dims[1] > pastValue.expansionDims[1]))
                    || (pastValue.dims.size() > 0 && pastValue.dims[1] + value.dims[1] > pastValue.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector <int> {value.dims[0], ((value.dims[1] - 1) / unitLen + 1) * unitLen, value.dims[2]};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((value.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
                CatDirect(pastKey, key, 1);
                CatDirect(pastValue, value, 1);


                MatMulTransB(query, pastKey, attnWeights, 1.0 / sqrt(head_dim));
                attnWeights.Reshape({1, attnWeights.dims[0], attnWeights.dims[1], attnWeights.dims[2]});
                if (attentionMask[b]) {
                    AttentionMask(attnWeights, *attentionMask[b], -10000);
                }

                Softmax(attnWeights, attnWeights, -1);
                MatMul(attnWeights, pastValue, attnOutput);

                attnOutput.Reshape({attnOutput.dims[1], attnOutput.dims[2], attnOutput.dims[3]});
                PermuteSelf(attnOutput, {1, 0, 2});
                attnOutput.Reshape({seqLens[b], 1, -1});
                PermuteSelf(attnOutput, {1, 0, 2});


                if (attnOutputAll.dims.size() == 0) {
                    std::vector <int> dims = attnOutput.dims;
                    dims[1] = total;
                    attnOutputAll.Expansion(dims);
                }
                CatDirect(attnOutputAll, attnOutput, 1);
            }

            std::string proj_weight_name = "transformer.h." + std::to_string(i) + ".attn.c_proj.weight";
            Linear(attnOutputAll, weight[proj_weight_name], Data(), attnLastOutput);
            AddTo(hiddenStates, attnLastOutput);

            std::string ln_2_name = "transformer.h." + std::to_string(i) + ".ln_2.weight";
            RMSNorm(hiddenStates, weight[ln_2_name], 1e-6, attnInput);

            std::string mlp_w1_weight_name = "transformer.h." + std::to_string(i) + ".mlp.w1.weight";
            std::string mlp_w2_weight_name = "transformer.h." + std::to_string(i) + ".mlp.w2.weight";
            std::string mlp_proj_weight_name = "transformer.h." + std::to_string(i) + ".mlp.c_proj.weight";
            Linear(attnInput, weight[mlp_w1_weight_name], Data(), a1);
            Linear(attnInput, weight[mlp_w2_weight_name], Data(), a2);
            Silu(a2, a2);
            MulTo(a1, a2);
            Linear(a1, weight[mlp_proj_weight_name], Data(), mlpOutput);
            AddTo(hiddenStates, mlpOutput);
        }

        RMSNorm(hiddenStates, weight["transformer.ln_f.weight"], 1e-6, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        logits.ToDevice(DataDevice::CPU);

        std::vector <int> lastRet;
        int total = 0;
        for (int b = 0; b < batch; b++) {
            if (generationConfigs[b].output_logits && retLogits != nullptr && (*retLogits)[b] != nullptr) {
                int base = (total + seqLens[b] - 1);
                (*retLogits)[b]->resize(logits.dims.back());
                memcpy((float*)(*retLogits)[b]->data(), (float*)(logits.cpuData + base * logits.dims.back() * logits.unitSize), logits.dims.back() * logits.unitSize);
            }
            if (generationConfigs[b].IsSimpleGreedy()) {
                std::pair<float, int> ret = std::make_pair(-1e9, -1);
                int base = (total + seqLens[b] - 1);
                total += seqLens[b];
                for (int i = 0; i < logits.dims.back(); i++) {
                    ret = max(ret, std::make_pair(((float *) logits.cpuData)[base * logits.dims.back() + i], i));
                }
                lastRet.push_back(ret.second);
            } else {
                int base = (total + seqLens[b] - 1);
                total += seqLens[b];
                lastRet.push_back(LLMSampling(logits, base, generationConfigs[b], lastTokens.units[b]));
            }
        }
        return lastRet;
    }

    std::string QWenModel::MakeInput(const std::string &history, int round, const std::string &input) {
        if (weight.dicts["chat_format"] == "chatml") {
            return (round == 0 ? im_start + "system" + "\n" + pre_prompt + im_end : history) + 
                "\n" + im_start + user_role + "\n" + input + im_end + "\n" + im_start + bot_role + "\n";
        } else if (weight.dicts["chat_format"] == "raw") {
            return history + input;
        } else {
            ErrorInFastLLM("Unknown char_format for QWen: " + weight.dicts["chat_format"]);
            return "";
        }
    }

    std::string QWenModel::MakeHistory(const std::string &history, int round, 
                                       const std::string &input, const std::string &output) {
        if (weight.dicts["chat_format"] == "chatml") {
            return (round == 0 ? im_start + "system" + "\n" + pre_prompt + im_end : history) + 
                "\n" + im_start + user_role + "\n" + input + im_end + "\n" + im_start + bot_role + "\n" + output + im_end;
        } else if (weight.dicts["chat_format"] == "raw") {
            return history + input + output;
        } else {
            ErrorInFastLLM("Unknown char_format for QWen: " + weight.dicts["chat_format"]);
            return "";
        }
    }

    void QWenModel::FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                  const std::map <std::string, int> &params,
                                  Data &inputIds, Data &attentionMask, Data &positionIds) {
        int index = params.find("index")->second;
        int promptLen = params.find("promptLen")->second;
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);
        if (index == 0) {
            int seqLen = inputTokens[0].size();
            std::vector <float> vmask = std::vector <float> (seqLen * seqLen, 0);
            std::vector<float> vpids = std::vector<float>(seqLen, 0);
            for (int i = 0; i < seqLen; i++) {
                vpids[i] = i;
                for (int j = i + 1; j < seqLen; j++) {
                    vmask[i * seqLen + j] = 1;
                }
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, inputTokens[0]));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {seqLen, seqLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vpids));
        } else {
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, inputTokens[0]));
            attentionMask.CopyFrom(Data());
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) (promptLen + index - 1)}));
        }
    }

    void QWenModel::FillLLMInputsBatch(std::vector <std::vector <float> > &inputTokens,
                                       const std::vector <std::map <std::string, int> > &params,
                                       Data &inputIds, Data &attentionMask, Data &positionIds) {
        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        int promptLen = params[0].find("promptLen")->second;

        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);
        
        if (index == 0) {
            int seqLen = inputTokens[0].size();
            std::vector<float> ids = std::vector<float>(batch * seqLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * seqLen * seqLen, 0);
            std::vector<float> vpids = std::vector<float>(batch * seqLen, 0);
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < seqLen; i++) {
                    ids[b * seqLen + i] = inputTokens[b][i];
                }
            }
            for (int i = 0; i < seqLen; i++) {
                vpids[i] = i;
                for (int j = i + 1; j < seqLen; j++) {
                    vmask[i * seqLen + j] = 1;
                }
            }
            for (int b = 1; b < batch; b++) {
                memcpy(vmask.data() + b * seqLen * seqLen, vmask.data(), seqLen * seqLen * sizeof(float));
                memcpy(vpids.data() + b * seqLen, vpids.data(), seqLen * sizeof(float));
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, seqLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, seqLen, seqLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, seqLen}, vpids));
        } else {
            std::vector<float> ids = std::vector<float>(batch * 1, 0);
            std::vector<float> vpids = std::vector<float>(batch * 1, 0);
            for (int b = 0; b < batch; b++) {
                ids[b] = inputTokens[b][0];
                vpids[b] = (float) (promptLen + index - 1);
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, ids));
            attentionMask.CopyFrom(Data());
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, vpids));
        }
    }

    void QWenModel::WarmUp() {
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
#ifdef USE_TFACC40T
        FastllmTfaccReleaseTempMemory();
#endif
        printf("finish.\n");
    }

    void QWenModel::UpdateRotaryPosEmb(float ntk_alpha) {
        float base = 10000 * pow(ntk_alpha, (float) rotary_dim / (rotary_dim - 2));

        if (sin.empty() || cos.empty()) {
            sin.resize(max_positions);
            cos.resize(max_positions);
        }
        
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(base, (float)i / rotary_dim));
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
    }
}