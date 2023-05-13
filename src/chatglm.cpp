//
// Created by huangyuyang on 5/11/23.
//

#include "chatglm.h"

#include <cmath>

#include <chrono>

#include <algorithm>

namespace fastllm {
    extern double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2);

    ChatGLMModel::ChatGLMModel() {
        sin.resize(max_positions);
        cos.resize(max_positions);
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(10000, (float)i / rotary_dim));
        }
        for (int i = 0; i < max_positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < rotary_dim; j++) {
                sin[i][j] = ::sin((float)i * invFreq[j]);
                cos[i][j] = ::cos((float)i * invFreq[j]);
            }
        }
    }

    void ChatGLMModel::RotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds) {
        // ChatGLM的Rotate，比较神奇，把key和value切成两半，分别和positionIds[0]和positionIds[1]旋转
        int outer = data.dims[0] * data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        for (int o = 0; o < outer; o++) {
            for (int part = 0; part < 2; part++) {
                int index = (int) ((float *) positionIds.cpuData)[part * positionIds.dims.back() + o];
                std::vector<float> &sin = this->sin[index];
                std::vector<float> &cos = this->cos[index];
                float *d = (float *) data.cpuData + o * spatial + part * m / 2;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < rotary_dim && j < m / 4; j++) {
                        float a = d[j], b = d[j + m / 4];
                        d[j] = a * cos[j] - b * sin[j];
                        d[j + m / 4] = a * sin[j] + b * cos[j];
                    }

                    d += m;
                }
            }
        }
    }

    void ChatGLMModel::LoadFromFile(const std::string &fileName) {
        this->weight.LoadFromFile(fileName);
    }

    int ChatGLMModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                              const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues) {
        Data inputEmbeddings;
        Embedding(inputIds, this->weight["transformer.word_embeddings.weight"], inputEmbeddings);
        Data hiddenStates = inputEmbeddings;
        hiddenStates.Permute({1, 0, 2});

        // ChatGLMBlock
        for (int i = 0; i < block_cnt; i++) {
//auto st = std::chrono::system_clock::now();
            std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
            Data attenInput;
            LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
//printf("input.ln %f\n", GetSpan(st, std::chrono::system_clock::now())); st = std::chrono::system_clock::now();
            std::string qkvWeightName = "transformer.layers." + std::to_string(i) + ".attention.query_key_value.weight";
            std::string qkvBiasName = "transformer.layers." + std::to_string(i) + ".attention.query_key_value.bias";
            Data qkv, q, k, v;
            Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);
//printf("qkv %f\n", GetSpan(st, std::chrono::system_clock::now())); st = std::chrono::system_clock::now();
            qkv.Reshape({qkv.dims[0], qkv.dims[1], num_attention_heads, -1});
            int per = qkv.dims.back() / 3;
            Split(qkv, -1, 0, per, q);
            Split(qkv, -1, per, per * 2, k);
            Split(qkv, -1, per * 2, per * 3, v);

            RotatePosition2D(q, positionIds);
            RotatePosition2D(k, positionIds);
//printf("rot %f\n", GetSpan(st, std::chrono::system_clock::now())); st = std::chrono::system_clock::now();
            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (pastKey.Count(0) + k.Count(0) > pastKey.expansionSize) {
                pastKey.Expansion(pastKey.Count(0) + k.Count(1) * 100);
            }
            if (pastValue.Count(0) + v.Count(0) > pastValue.expansionSize) {
                pastValue.Expansion(pastValue.Count(0) + v.Count(1) * 100);
            }
            CatDirectAxis0(pastKey, k);
            CatDirectAxis0(pastValue, v);

//printf("cat %f\n", GetSpan(st, std::chrono::system_clock::now())); st = std::chrono::system_clock::now();
            std::vector <int> outputSize = {q.dims[1], q.dims[2], q.dims[0], pastKeyValues[i].first.dims[0]};

            q.Reshape({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});
            q.Permute({1, 0, 2});

            std::vector <int> tempDims = pastKeyValues[i].first.dims;
            pastKeyValues[i].first.Reshape({pastKeyValues[i].first.dims[0],
                       pastKeyValues[i].first.dims[1] * pastKeyValues[i].first.dims[2],
                       pastKeyValues[i].first.dims[3]});
            Permute(pastKeyValues[i].first, {1, 0, 2}, k);
            pastKeyValues[i].first.Reshape(tempDims);

            // 1.2 Attention
            // 1.2.0 q * k^T
            Data attnProbs;
//printf("qk0 %f\n", GetSpan(st, std::chrono::system_clock::now())); st = std::chrono::system_clock::now();
            MatMulTransB(q, k, attnProbs, 1.0 / (scale_attn * (i + 1)));
            attnProbs.Reshape(outputSize);
//printf("qk1 %f\n", GetSpan(st, std::chrono::system_clock::now())); st = std::chrono::system_clock::now();
            // 1.2.1 Mask
            if (attentionMask.dims.size() != 0) {
                float *maskData = (float *) attentionMask.cpuData;
                float *attnData = (float *) attnProbs.cpuData;
                int spatial = attnProbs.Count(2), outer = attnProbs.Count(0) / spatial;;
                for (int o = 0; o < outer; o++) {
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[i] > 0.99) {
                            attnData[o * spatial + i] = -10000;
                        }
                    }
                }
            }
            // 1.2.2 softmax
            Mul(attnProbs, i + 1, attnProbs);
            Softmax(attnProbs, attnProbs, -1);

            outputSize = {pastKeyValues[i].second.dims[1],
                          pastKeyValues[i].second.dims[2],
                          q.dims[1],
                          pastKeyValues[i].second.dims[3]};

            tempDims = pastKeyValues[i].second.dims;
            pastKeyValues[i].second.Reshape({pastKeyValues[i].second.dims[0], outputSize[0] * outputSize[1], -1});
            Permute(pastKeyValues[i].second, {1, 2, 0}, v);
            pastKeyValues[i].second.Reshape(tempDims);

            attnProbs.Reshape({outputSize[0] * outputSize[1], outputSize[2], -1});
            // 1.2.3 prob * v
            Data tempCL, contextLayer;
            MatMulTransB(attnProbs, v, tempCL);
            tempCL.Reshape(outputSize);
            Permute(tempCL, {2, 0, 1, 3}, contextLayer);
            contextLayer.Reshape({contextLayer.dims[0], contextLayer.dims[1], embed_dim});
//printf("qkv %f\n", GetSpan(st, std::chrono::system_clock::now())); st = std::chrono::system_clock::now();
            // 1.2.4 dense
            std::string denseWeightName = "transformer.layers." + std::to_string(i) + ".attention.dense.weight";
            std::string denseBiasName = "transformer.layers." + std::to_string(i) + ".attention.dense.bias";
            Data attnOutput;
            Linear(contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);
//printf("dense %f\n", GetSpan(st, std::chrono::system_clock::now())); st = std::chrono::system_clock::now();
            // 1.3
            float alpha = sqrt(2 * block_cnt);
            Mul(attenInput, alpha, hiddenStates);
            AddTo(hiddenStates, attnOutput);
            std::string postLNWeightName = "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string postLNBiasName = "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.bias";
            Data mlpInput;
            LayerNorm(hiddenStates, weight[postLNWeightName], weight[postLNBiasName], -1, mlpInput);
//printf("postln %f\n", GetSpan(st, std::chrono::system_clock::now())); st = std::chrono::system_clock::now();
            // 1.4 MLP
            std::string fcInKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
            std::string fcOutKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
            Data middle, mlpOutput;
            Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
            GeluNew(middle, middle);
            Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], mlpOutput);
            AddTo(mlpOutput, mlpInput, alpha);
            hiddenStates.CopyFrom(mlpOutput);
//printf("mlp %f\n", GetSpan(st, std::chrono::system_clock::now()));
        }

        LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"], weight["transformer.final_layernorm.bias"], -1, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);

        std::vector <std::pair <float, int> > v;
        int base = logits.dims[0] - 1;
        for (int i = 0; i < logits.dims.back(); i++) {
            v.push_back(std::make_pair(((float*)logits.cpuData)[base * logits.dims.back() + i], i));
        }
        std::sort(v.begin(), v.end());
        std::reverse(v.begin(), v.end());

        return v[0].second;
    }

    std::string ChatGLMModel::Response(const std::string &input) {
        Data inputIds = this->weight.tokenizer.Encode(input);
        std::vector <float> ids;
        for (int i = 0; i < inputIds.Count(0); i++) {
            ids.push_back(((float*)inputIds.cpuData)[i]);
        }
        ids.push_back(130001);
        ids.push_back(130004);
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
        while (true) {
            auto st = std::chrono::system_clock::now();

            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues);
            if (ret == 130005) {
                break;
            }

            results.push_back(ret);
            std::string curString = weight.tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results)).c_str();
            retString += curString;
            printf("%s", curString.c_str());
            fflush(stdout);
            results.clear();

            len++;
            if (maskIds == -1) {
                maskIds = (int)ids.size() - 2;
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)ret}));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {2, 1}, {(float)maskIds, (float)(len)}));

            //printf("spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        }

        printf("\n");
        return retString;
    }

    void ChatGLMModel::SaveLowBitModel(const std::string &fileName, int bit) {
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {130004});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {2, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        this->weight.SaveLowBitModel(fileName, bit);
    }
}
