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
        int len = data.dims[0], bs = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        for (int l = 0; l < len; l++) {
            for (int b = 0; b < bs; b++) {
                for (int part = 0; part < 2; part++) {
                    int index = (int) ((float *) positionIds.cpuData)[(b * 2 + part) * positionIds.dims.back() + l];
                    std::vector<float> &sin = this->sin[index];
                    std::vector<float> &cos = this->cos[index];
                    float *d = (float *) data.cpuData + (l * bs + b) * spatial + part * m / 2;
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
        Data logits, topk;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        TopK(logits, topk, 1);
        topk.ToDevice(DataDevice::CPU);
//timeRecord.Record("get max");
//timeRecord.Print();
        return (int)(((float *) topk.cpuData)[(topk.dims[0] - 1) * 2] + 1e-3);
    }

    std::vector <int> ChatGLMModel::ForwardBatch(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector <std::pair <Data, Data> > &pastKeyValues) {
TimeRecord batchRecord;
//batchrecord.Clear();
//batchrecord.Record();
        int maxLen = inputIds.dims[1];
        sinData.ToDevice(DataDevice::CUDA);
        cosData.ToDevice(DataDevice::CUDA);
        Data inputEmbeddings;
        Embedding(inputIds, this->weight["transformer.word_embeddings.weight"], inputEmbeddings);
        Data hiddenStates = inputEmbeddings;
        hiddenStates.Permute({1, 0, 2});
        hiddenStates.ToDevice(DataDevice::CUDA);
        // ChatGLMBlock
//batchrecord.Record("Pre");
        for (int i = 0; i < block_cnt; i++) {
            std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
            Data attenInput;
            LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
            std::string qkvWeightName = "transformer.layers." + std::to_string(i) + ".attention.query_key_value.weight";
            std::string qkvBiasName = "transformer.layers." + std::to_string(i) + ".attention.query_key_value.bias";
            Data qkv, q, k, v;
//batchrecord.Record("LayerNorm");
            Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);
//batchrecord.Record("Linear");
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

            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 2);
            std::vector <int> outputSize = {q.dims[1], q.dims[2], q.dims[0], pastKey.dims[1]};

            q.Reshape({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});
            q.Permute({1, 0, 2});

            // 1.2 Attention
            // 1.2.0 q * k^T
            Data attnProbs;
//batchrecord.Record("GetQKV");
            MatMulTransB(q, pastKey, attnProbs, 1.0 / (scale_attn * (i + 1)));
//batchrecord.Record("MatMulTransB");
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
            Data contextLayer;
//batchrecord.Record("Softmax");
            MatMulTransB(attnProbs, pastValue, contextLayer);
//batchrecord.Record("MatMulTransB");
            contextLayer.Reshape({batch, num_attention_heads, maxLen, -1});
            contextLayer.Permute({2, 0, 1, 3});

            contextLayer.Reshape({contextLayer.dims[0], contextLayer.dims[1], embed_dim});
            // 1.2.4 dense
            std::string denseWeightName = "transformer.layers." + std::to_string(i) + ".attention.dense.weight";
            std::string denseBiasName = "transformer.layers." + std::to_string(i) + ".attention.dense.bias";
            Data attnOutput;
//batchrecord.Record("contextLayer");
            Linear(contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);
//batchrecord.Record("Linear");
            // 1.3
            float alpha = sqrt(2 * block_cnt);
            Mul(attenInput, alpha, hiddenStates);
            AddTo(hiddenStates, attnOutput);
//batchrecord.Record("Add");
            std::string postLNWeightName = "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string postLNBiasName = "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.bias";
            Data mlpInput;
            LayerNorm(hiddenStates, weight[postLNWeightName], weight[postLNBiasName], -1, mlpInput);
            // 1.4 MLP
            std::string fcInKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
            std::string fcOutKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
            Data middle;
//batchrecord.Record("LayerNorm");
            Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
//batchrecord.Record("Linear");
            GeluNew(middle, middle);
//batchrecord.Record("Gelu");
            Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);
//batchrecord.Record("Linear");
            AddTo(hiddenStates, mlpInput, alpha);
//batchrecord.Record("Add");
        }
        LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"], weight["transformer.final_layernorm.bias"], -1, hiddenStates);
        Data logits, topk;
//batchrecord.Record("LayerNorm");
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
//batchrecord.Record("Linear");
        TopK(logits, topk, 1);
        topk.ToDevice(DataDevice::CPU);
//batchrecord.Record("logit to cpu");
        std::vector <int> lastRet;
        for (int b = 0; b < batch; b++) {
            int base = (maxLen - 1) * batch + b;
            lastRet.push_back((int)(((float *) topk.cpuData)[base * 2] + 1e-3));
        }
//batchrecord.Record("last");
//batchrecord.Print();
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
                               RuntimeResultBatch retCb) {
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
            Data &tokens = inputTokens[i];
            int len = tokens.Count(0), base = maxLen - 2 - len;
            for (int j = 0; j < len; j++) {
                ids[i * maxLen + base + j] = ((float*)tokens.cpuData)[j];
            }
            ids[i * maxLen + base + len] = 130001;
            ids[i * maxLen + base + len + 1] = 130004;
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
        while (true) {
            auto st = std::chrono::system_clock::now();

            attentionMask.ToDevice(DataDevice::CUDA);
            positionIds.ToDevice(DataDevice::CUDA);
            std::vector <int> ret = ForwardBatch(batch, inputIds, attentionMask, positionIds, pastKeyValues);
            std::vector <float> fret;
            std::vector <float> results;
            int endingCount = 0;
            std::vector <std::string> curStrings;
            for (int i = 0; i < batch; i++) {
                fret.push_back(ret[i]);
                if (ret[i] == 130005) {
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
                    maskIds[i] = seqLens[i];
                }
            }

            if (endingCount == batch) {
                break;
            }
            if (retCb)
                retCb(index++, curStrings);

            len++;
            std::vector <float> pids = std::vector <float> (batch * 2);
            for (int i = 0; i < batch; i++) {
                pids[i * 2] = maskIds[i];
                pids[i * 2 + 1] = len;
            }
            positionIds.ToDevice(DataDevice::CPU);
            attentionMask.ToDevice(DataDevice::CPU);
            attentionMask = Data();
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch * 2, 1}, pids));

            //printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));
        }

        if (retCb)
            retCb(-1, outputs);
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
