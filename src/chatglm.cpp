//
// Created by huangyuyang on 5/11/23.
//

#include "chatglm.h"

#include <cmath>

#include <chrono>

#include <algorithm>

#include <map>

namespace fastllm {
    extern double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2);

    struct TimeRecord {
        std::map <std::string, float> v;
        std::chrono::system_clock::time_point t;

        void Clear() {
            v.clear();
        }

        void Record() {
            t = std::chrono::system_clock::now();
        }

        void Record(const std::string &key) {
            auto now = std::chrono::system_clock::now();
            v[key] += GetSpan(t, now);
            t = now;
        }

        void Print() {
            float s = 0;
            for (auto &it : v) {
                printf("%s: %f s.\n", it.first.c_str(), it.second);
                s += it.second;
            }
            printf("Total: %f s.\n", s);
        }
    };

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

            if (q.dims[0] != 1) {
                v.ToDevice(DataDevice::CPU);
            }
            q.ToDevice(DataDevice::CPU);
            k.ToDevice(DataDevice::CPU);

            RotatePosition2D(q, positionIds);
            RotatePosition2D(k, positionIds);
//timeRecord.Record("rot");
            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;

            pastKey.ToDevice(DataDevice::CUDA);
            pastValue.ToDevice(DataDevice::CUDA);

            k.Resize({k.dims[0], k.dims[1] * k.dims[2], k.dims[3]});
            k.Permute({1, 0, 2});
            v.Resize({v.dims[0], v.dims[1] * v.dims[2], v.dims[3]});
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

            k.ToDevice(DataDevice::CUDA);
            v.ToDevice(DataDevice::CUDA);
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

            q.ToDevice(DataDevice::CUDA);
            MatMulTransB(q, pastKey, attnProbs, 1.0 / (scale_attn * (i + 1)));
            attnProbs.Reshape(outputSize);
//timeRecord.Record("qk1");
            // 1.2.1 Mask
            if (attentionMask.dims.size() != 0) {
                attnProbs.ToDevice(DataDevice::CPU);

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

            outputSize = {1, pastValue.dims[0], q.dims[1], pastValue.dims[1]};
            attnProbs.Reshape({outputSize[0] * outputSize[1], outputSize[2], -1});
            // 1.2.3 prob * v
            Data contextLayer;
//timeRecord.Record("qkv prepare");
            attnProbs.ToDevice(DataDevice::CUDA);
            MatMulTransB(attnProbs, pastValue, contextLayer);
            if (contextLayer.dims[2] != 1) {
                contextLayer.ToDevice(DataDevice::CPU);
            }
//timeRecord.Record("MatMulTransB");
            contextLayer.Reshape(outputSize);
            contextLayer.Permute({2, 0, 1, 3});
            contextLayer.Reshape({contextLayer.dims[0], contextLayer.dims[1], embed_dim});
            // 1.2.4 dense
            std::string denseWeightName = "transformer.layers." + std::to_string(i) + ".attention.dense.weight";
            std::string denseBiasName = "transformer.layers." + std::to_string(i) + ".attention.dense.bias";
            Data attnOutput;
//timeRecord.Record("qkv");
            contextLayer.ToDevice(DataDevice::CUDA);
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
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)ret}));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {2, 1}, {(float)maskIds, (float)(len)}));

            //printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));
        }
		if (retCb)
			retCb(-1, retString.c_str());

        return retString;
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
