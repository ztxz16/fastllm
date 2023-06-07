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
#include "fastllm-cuda.h"
#endif

namespace fastllm {
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
        weight.embeddingNames.insert("transformer.word_embeddings.weight");
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
TimeRecord timeRecord;
//timeRecord.Clear();
//timeRecord.Record();
        sinData.ToDevice(DataDevice::CUDA);
        cosData.ToDevice(DataDevice::CUDA);
        Data inputEmbeddings;
        Embedding(inputIds, this->weight["transformer.word_embeddings.weight"], inputEmbeddings);
        Data hiddenStates = inputEmbeddings;
        hiddenStates.Permute({1, 0, 2});
        hiddenStates.ToDevice(DataDevice::CUDA);
        //timeRecord.Record("embedding");
        // ChatGLMBlock
        for (int i = 0; i < block_cnt; i++) {
//timeRecord.Record("next block");
            std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
            Data attenInput;
            LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
//timeRecord.Record("layernorm");
            std::string qkvWeightName = "transformer.layers." + std::to_string(i) + ".attention.query_key_value.weight";
            std::string qkvBiasName = "transformer.layers." + std::to_string(i) + ".attention.query_key_value.bias";
            Data qkv, q, k, v;
            Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);
//timeRecord.Record("linear");
            qkv.Reshape({qkv.dims[0], qkv.dims[1], num_attention_heads, -1});
            int per = qkv.dims.back() / 3;
            Split(qkv, -1, 0, per, q);
            Split(qkv, -1, per, per * 2, k);
            Split(qkv, -1, per * 2, per * 3, v);

//timeRecord.Record("split");
#ifdef USE_CUDA
            FastllmCudaRotatePosition2D(q, positionIds, sinData, cosData, rotary_dim);
            FastllmCudaRotatePosition2D(k, positionIds, sinData, cosData, rotary_dim);
#else
            RotatePosition2D(q, positionIds);
            RotatePosition2D(k, positionIds);
#endif
//timeRecord.Record("rot");
            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;

            pastKey.ToDevice(DataDevice::CUDA);
            pastValue.ToDevice(DataDevice::CUDA);

            k.Resize({k.dims[0], k.dims[1] * k.dims[2], k.dims[3]});
            v.Resize({v.dims[0], v.dims[1] * v.dims[2], v.dims[3]});

            k.Permute({1, 0, 2});
            v.Permute({1, 2, 0});

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

            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[2] > pastValue.expansionDims[2]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[2] + v.dims[2] > pastValue.expansionDims[2])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], v.dims[1], ((v.dims[2] - 1) / unitLen + 1) * unitLen};
                } else {
                    newDims = pastValue.dims;
                    newDims[2] += ((v.dims[2] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }

//timeRecord.Record("cat");
            CatDirect(pastKey, k, 1);
//timeRecord.Record("catk");
            CatDirect(pastValue, v, 2);
//timeRecord.Record("catv");
            std::vector <int> outputSize = {q.dims[1], q.dims[2], q.dims[0], pastKey.dims[1]};

            q.Reshape({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});
            q.Permute({1, 0, 2});

            // 1.2 Attention
            // 1.2.0 q * k^T
            Data attnProbs;
//timeRecord.Record("qk0");

            MatMulTransB(q, pastKey, attnProbs, 1.0 / (scale_attn * (i + 1)));
            attnProbs.Reshape(outputSize);
//timeRecord.Record("qk1");
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
            Data contextLayer;
//timeRecord.Record("qkv prepare");
            MatMulTransB(attnProbs, pastValue, contextLayer);
//timeRecord.Record("MatMulTransB");
            contextLayer.Reshape(outputSize);
            contextLayer.Permute({2, 0, 1, 3});

            contextLayer.Reshape({contextLayer.dims[0], contextLayer.dims[1], embed_dim});
            // 1.2.4 dense
            std::string denseWeightName = "transformer.layers." + std::to_string(i) + ".attention.dense.weight";
            std::string denseBiasName = "transformer.layers." + std::to_string(i) + ".attention.dense.bias";
            Data attnOutput;
//timeRecord.Record("qkv");
            Linear(contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);
//timeRecord.Record("linear");
            // 1.3
            float alpha = sqrt(2 * block_cnt);
            Mul(attenInput, alpha, hiddenStates);
            AddTo(hiddenStates, attnOutput);
            std::string postLNWeightName = "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string postLNBiasName = "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.bias";
            Data mlpInput;
            LayerNorm(hiddenStates, weight[postLNWeightName], weight[postLNBiasName], -1, mlpInput);
            // 1.4 MLP
            std::string fcInKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
            std::string fcOutKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
            Data middle;
//timeRecord.Record("post ln");
            Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
//timeRecord.Record("linear");
            GeluNew(middle, middle);
//timeRecord.Record("gelu");
            Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
//timeRecord.Record("linear");
            AddTo(hiddenStates, mlpInput, alpha);
//timeRecord.Record("mlp");
        }

        LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"], weight["transformer.final_layernorm.bias"], -1, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        logits.ToDevice(DataDevice::CPU);
        //timeRecord.Record("logits");
        std::pair <float, int> ret = std::make_pair(-1e9, -1);
        int base = logits.dims[0] - 1;
        for (int i = 0; i < logits.dims.back(); i++) {
            ret = max(ret, std::make_pair(((float*)logits.cpuData)[base * logits.dims.back() + i], i));
        }
//timeRecord.Record("get max");

//timeRecord.Print();
        return ret.second;
    }

    std::vector <int> ChatGLMModel::ForwardBatch(
            int batch,
            const Data &inputIds,
            const std::vector <Data> &attentionMask,
            const Data &positionIds,
            const std::vector <int> &seqLens,
            std::vector <std::pair <Data, Data> > &pastKeyValues) {
        sinData.ToDevice(DataDevice::CUDA);
        cosData.ToDevice(DataDevice::CUDA);
        Data inputEmbeddings;
        Embedding(inputIds, this->weight["transformer.word_embeddings.weight"], inputEmbeddings);
        Data hiddenStates = inputEmbeddings;
        hiddenStates.Permute({1, 0, 2});
        hiddenStates.ToDevice(DataDevice::CUDA);
        for (int i = 0; i < block_cnt; i++) {
            std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
            Data attenInput;
            LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
            std::string qkvWeightName = "transformer.layers." + std::to_string(i) + ".attention.query_key_value.weight";
            std::string qkvBiasName = "transformer.layers." + std::to_string(i) + ".attention.query_key_value.bias";
            Data qkv, q, k, v;
            Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);
            qkv.Reshape({qkv.dims[0], qkv.dims[1], num_attention_heads, -1});
            int per = qkv.dims.back() / 3;
            Split(qkv, -1, 0, per, q);
            Split(qkv, -1, per, per * 2, k);
            Split(qkv, -1, per * 2, per * 3, v);

#ifdef USE_CUDA
            FastllmCudaRotatePosition2D(q, positionIds, sinData, cosData, rotary_dim);
            FastllmCudaRotatePosition2D(k, positionIds, sinData, cosData, rotary_dim);
#else
            RotatePosition2D(q, positionIds);
            RotatePosition2D(k, positionIds);
#endif
            k.Resize({k.dims[0], k.dims[1] * k.dims[2], k.dims[3]});
            v.Resize({v.dims[0], v.dims[1] * v.dims[2], v.dims[3]});
            q.Reshape({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});

            Data contextLayer = Data(DataType::FLOAT32);

            int totalLen = 0;
            for (int b = 0; b < batch; b++) {
                totalLen += seqLens[b];
            }

            int total = 0;
            std::vector <Data> curKs, curVs, curQs;
            curKs.resize(batch);
            curVs.resize(batch);
            curQs.resize(batch);
            for (int b = 0; b < batch; b++) {
                Split(k, 0, total, total + seqLens[b], curKs[b]);
                Split(v, 0, total, total + seqLens[b], curVs[b]);
                Split(q, 0, total, total + seqLens[b], curQs[b]);
                total += seqLens[b];
            }
            for (int b = 0; b < batch; b++) {
                curKs[b].Permute({1, 0, 2});
                curVs[b].Permute({1, 2, 0});
                curQs[b].Permute({1, 0, 2});
            }

            for (int b = 0; b < batch ; b++) {
                Data &k = curKs[b], &v = curVs[b], &q = curQs[b];
                Data &pastKey = pastKeyValues[b * block_cnt + i].first, &pastValue = pastKeyValues[b * block_cnt + i].second;

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
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }

                while ((pastValue.dims.size() == 0 &&
                        (pastValue.expansionDims.size() == 0 || v.dims[2] > pastValue.expansionDims[2]))
                       || (pastValue.dims.size() > 0 && pastValue.dims[2] + v.dims[2] > pastValue.expansionDims[2])) {
                    std::vector<int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector<int>{v.dims[0], v.dims[1], ((v.dims[2] - 1) / unitLen + 1) * unitLen};
                    } else {
                        newDims = pastValue.dims;
                        newDims[2] += ((v.dims[2] - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }

                CatDirect(pastKey, k, 1);
                CatDirect(pastValue, v, 2);

                std::vector<int> outputSize = {1, q.dims[0], q.dims[1], pastKey.dims[1]};

                // 1.2 Attention
                // 1.2.0 q * k^T
                Data attnProbs;
                MatMulTransB(q, pastKey, attnProbs, 1.0 / (scale_attn * (i + 1)));
                attnProbs.Reshape(outputSize);

                // 1.2.1 Mask
                if (attentionMask.size() != 0) {
                    AttentionMask(attnProbs, attentionMask[b], -10000);
                }
                // 1.2.2 softmax
                Mul(attnProbs, i + 1, attnProbs);
                Softmax(attnProbs, attnProbs, -1);

                outputSize = {1, pastValue.dims[0], q.dims[1], pastValue.dims[1]};
                attnProbs.Reshape({outputSize[0] * outputSize[1], outputSize[2], -1});
                // 1.2.3 prob * v
                Data curContextLayer;
                MatMulTransB(attnProbs, pastValue, curContextLayer);
                curContextLayer.Reshape(outputSize);
                curContextLayer.Permute({2, 0, 1, 3});
                curContextLayer.Reshape({curContextLayer.dims[0], curContextLayer.dims[1], embed_dim});

                if (contextLayer.dims.size() == 0) {
                    std::vector <int> dims = curContextLayer.dims;
                    dims[0] = totalLen;
                    contextLayer.Expansion(dims);
                }
                contextLayer.ToDevice(DataDevice::CUDA);
                CatDirect(contextLayer, curContextLayer, 0);
            }

            // 1.2.4 dense
            std::string denseWeightName = "transformer.layers." + std::to_string(i) + ".attention.dense.weight";
            std::string denseBiasName = "transformer.layers." + std::to_string(i) + ".attention.dense.bias";
            Data attnOutput;
            Linear(contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);

            // 1.3
            float alpha = sqrt(2 * block_cnt);
            Mul(attenInput, alpha, hiddenStates);
            AddTo(hiddenStates, attnOutput);
            std::string postLNWeightName = "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string postLNBiasName = "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.bias";
            Data mlpInput;
            LayerNorm(hiddenStates, weight[postLNWeightName], weight[postLNBiasName], -1, mlpInput);
            // 1.4 MLP
            std::string fcInKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
            std::string fcOutKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
            Data middle;
            Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
            GeluNew(middle, middle);
            Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
            AddTo(hiddenStates, mlpInput, alpha);
        }

        LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"], weight["transformer.final_layernorm.bias"], -1, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        logits.ToDevice(DataDevice::CPU);

        std::vector <int> lastRet;
        int total = 0;
        for (int b = 0; b < batch; b++) {
            std::pair<float, int> ret = std::make_pair(-1e9, -1);
            int base = (total + seqLens[b] - 1);
            total += seqLens[b];
            for (int i = 0; i < logits.dims.back(); i++) {
                ret = max(ret, std::make_pair(((float *) logits.cpuData)[base * logits.dims.back() + i], i));
            }
            lastRet.push_back(ret.second);
        }

        return lastRet;
    }

    std::string ChatGLMModel::Response(const std::string& input, RuntimeResult retCb) {
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
		int index = 0;
        while (true) {
            auto st = std::chrono::system_clock::now();

            attentionMask.ToDevice(DataDevice::CUDA);
            positionIds.ToDevice(DataDevice::CUDA);
            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues);
            if (ret == 130005) {
                break;
            }

            results.push_back(ret);
            std::string curString = weight.tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results)).c_str();
            retString += curString;
			if (retCb)
				retCb(index++, curString.c_str());
            fflush(stdout);
            results.clear();

            len++;
            if (maskIds == -1) {
                maskIds = (int)ids.size() - 2;
            }

            attentionMask.ToDevice(DataDevice::CPU);
            positionIds.ToDevice(DataDevice::CPU);
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)ret}));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {2, 1}, {(float)maskIds, (float)(len)}));

            //printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));
        }
		if (retCb)
			retCb(-1, retString.c_str());

        return retString;
    }

    void ChatGLMModel::ResponseBatch(const std::vector <std::string> &inputs,
                               std::vector <std::string> &outputs,
                               RuntimeResult retCb) {
        // 1. first
        int batch = inputs.size();
        outputs.clear();
        outputs.resize(batch, "");

        std::vector <float> ids;
        std::vector <int> seqLens;
        for (int i = 0; i < inputs.size(); i++) {
            Data inputIds = this->weight.tokenizer.Encode(inputs[i]);
            seqLens.push_back(inputIds.Count(0) + 2);
            for (int i = 0; i < inputIds.Count(0); i++) {
                ids.push_back(((float*)inputIds.cpuData)[i]);
            }
            ids.push_back(130001);
            ids.push_back(130004);
        }
        Data inputIds = Data(DataType::FLOAT32, {1, (int)ids.size()}, ids);

        std::vector <float> vpids = std::vector <float> (2 * (int)ids.size(), 0);
        std::vector <Data> attentionMasks;
        attentionMasks.resize(seqLens.size());
        int total = 0;
        for (int i = 0; i < seqLens.size(); i++) {
            int curLen = seqLens[i];

            std::vector <float> vmask = std::vector <float> (curLen * curLen, 0);
            for (int i = 0; i < curLen - 1; i++) {
                vmask[i * curLen + curLen - 1] = 1;
            }
            attentionMasks[i].CopyFrom(Data(DataType::FLOAT32, {curLen, curLen}, vmask));

            for (int j = 0; j < curLen - 1; j++) {
                vpids[total + j] = j;
            }
            vpids[total + curLen - 1] = curLen - 2;
            vpids[(int)ids.size() + total + curLen - 1] = 1;
            total += curLen;
        }
        Data positionIds = Data(DataType::FLOAT32, {2, (int)ids.size()}, vpids);

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < batch * block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }

        int len = 1;
        std::vector <int> maskIds = std::vector <int> (batch, -1);
        std::vector <float> results;
        std::vector <bool> isEnding = std::vector <bool> (batch, false);
        int index = 0;
        while (true) {
            auto st = std::chrono::system_clock::now();

            for (int i = 0; i < attentionMasks.size(); i++) {
                attentionMasks[i].ToDevice(DataDevice::CUDA);
            }
            positionIds.ToDevice(DataDevice::CUDA);
            std::vector <int> ret = ForwardBatch(batch, inputIds, attentionMasks, positionIds,
                                                          seqLens, pastKeyValues);
            std::vector <float> fret;
            int endingCount = 0;
            for (int i = 0; i < batch; i++) {
                fret.push_back(ret[i]);
                if (ret[i] == 130005) {
                    isEnding[i] = true;
                }
                if (isEnding[i]) {
                    endingCount++;
                    continue;
                }
                results.push_back(ret[i]);
                std::string curString = weight.tokenizer.Decode(
                        Data(DataType::FLOAT32, {(int) results.size()}, results)).c_str();
                printf("%d %s\n", i, curString.c_str());
                outputs[i] += curString;
                results.clear();

                if (maskIds[i] == -1) {
                    maskIds[i] = seqLens[i] - 2;
                }

                /*
                if (retCb)
                    retCb(index++, curString.c_str());
                fflush(stdout);
*/
            }

            if (endingCount == batch) {
                break;
            }

            len++;

            std::vector <float> pids = std::vector <float> (2 * batch);
            for (int i = 0; i < batch; i++) {
                pids[i] = maskIds[i];
                pids[i + batch] = len;
                seqLens[i] = 1;
            }
            positionIds.ToDevice(DataDevice::CPU);
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, batch}, fret));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {2, batch}, pids));
            attentionMasks.clear();

            printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));

            if (len == 8) {
                break;
            }
        }
/*
        if (retCb)
            retCb(-1, retString.c_str());
*/
    }

    void ChatGLMModel::WarmUp() {
    	printf("Warmup...\n");
	    Data inputIds = Data(DataType::FLOAT32, {1, 1}, {130004});
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

    void ChatGLMModel::SaveLowBitModel(const std::string &fileName, int bit) {
        WarmUp();
        this->weight.SaveLowBitModel(fileName, bit);
    }
}
