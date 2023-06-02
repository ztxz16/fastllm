//
// Created by huangyuyang on 5/12/23.
//

#include "utils.h"

#include "moss.h"

#include <cmath>

#include <chrono>

#include <algorithm>

namespace fastllm {
    extern double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2);

    MOSSModel::MOSSModel() {
        // 初始化sin, cos
		embed_dim = 6144;
		num_attention_heads = 24;
		head_dim = embed_dim / num_attention_heads;
		block_cnt = 34;

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
        this->weight.embeddingNames.insert("transformer.wte.weight");
    }

    void MOSSModel::LoadFromFile(const std::string &fileName) {
        this->weight.LoadFromFile(fileName);
    }

    void MOSSModel::CausalMask(Data &data, int start) {
        int outer = data.dims[0] * data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        for (int o = 0; o < outer; o++) {
            float *d = (float*)data.cpuData + o * spatial;
            for (int i = 0; i < n; i++) {
                if (i + start + 1 < m) {
                    std::fill(d + i * m + i + start + 1, d + (i + 1) * m, -std::numeric_limits<float>::max());
                }
            }
        }
    }

    void MOSSModel::RotatePosition2D(Data &data, const Data &positionIds) {
        int outer = data.dims[0] * data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        for (int o = 0; o < outer; o++) {
            int index = (int)((float*)positionIds.cpuData)[o];
            std::vector <float> &sin = this->sin[index];
            std::vector <float> &cos = this->cos[index];
            float *d = (float*)data.cpuData + o * spatial;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j + 1 < rotary_dim && j + 1 < m; j += 2) {
                    float a = d[j], b = d[j + 1];
                    d[j] = a * cos[j / 2] - b * sin[j / 2];
                    d[j + 1] = a * sin[j / 2] + b * cos[j / 2];
                }

                d += m;
            }
        }
    }

    int MOSSModel::Forward(const Data &inputIds, const Data &attentionMask,
                            const Data &positionIds, std::vector <std::pair <Data, Data> > &pastKeyValues) {
        auto st = std::chrono::system_clock::now();

        Data inputEmbeddings;
        Embedding(inputIds, this->weight["transformer.wte.weight"], inputEmbeddings);
        Data hiddenStates = inputEmbeddings;

        // MossBlock
        for (int i = 0; i < block_cnt; i++) {
            // 1.0 LayerNorm
            Data residual = hiddenStates;
            std::string lnWeightName = "transformer.h." + std::to_string(i) + ".ln_1.weight";
            std::string lnBiasName = "transformer.h." + std::to_string(i) + ".ln_1.bias";
            LayerNorm(residual, weight[lnWeightName], weight[lnBiasName], -1, hiddenStates);

            // 1.1 Get query, key, value
            std::string qkvProjName = "transformer.h." + std::to_string(i) + ".attn.qkv_proj.weight";
            Data qkv, q, k, v;
            Linear(hiddenStates, weight[qkvProjName], Data(), qkv);

            qkv.Reshape({qkv.dims[0], qkv.dims[1], 4, -1});
            int per = qkv.dims.back() / 3;
            Split(qkv, -1, 0, per, q);
            Split(qkv, -1, per, per * 2, v);
            Split(qkv, -1, per * 2, per * 3, k);

            q.Reshape({q.dims[0], q.dims[1], -1, head_dim});
            k.Reshape({k.dims[0], k.dims[1], -1, head_dim});
            v.Reshape({v.dims[0], v.dims[1], -1, head_dim});

            RotatePosition2D(q, positionIds);
            RotatePosition2D(k, positionIds);

            q.Permute({0, 2, 1, 3});
            k.Permute({0, 2, 1, 3});
            v.Permute({0, 2, 1, 3});

            Data pastKey = pastKeyValues[i].first, pastValue = pastKeyValues[i].second;
            Cat(pastKey, k, -2, pastKeyValues[i].first);
            Cat(pastValue, v, -2, pastKeyValues[i].second);

            k.CopyFrom(pastKeyValues[i].first);
            v.CopyFrom(pastKeyValues[i].second);

            // 1.2 Attention
            // 1.2.0 q * k^T
            Data attnWeights;
            MatMulTransB(q, k, attnWeights, 1.0 / scale_attn);

            // 1.2.1 causal_mask
            CausalMask(attnWeights, k.dims[2] - q.dims[2]);

            // 1.2.2 attentionMask
            // TODO: attentionMask, 这里似乎都是1, 暂且跳过了

            // 1.2.3 softmax
            Softmax(attnWeights, attnWeights, -1);

            // 1.2.4 headMask
            // TODO: headMask, 这里似乎都是None, 暂且跳过了

            // 1.2.5 attention_weights * v
            Data attnOutput;
            v.Permute({0, 1, 3, 2});
            MatMulTransB(attnWeights, v, attnOutput);

            // 1.3
            attnOutput.Permute({0, 2, 1, 3});
            attnOutput.Reshape({attnOutput.dims[0], attnOutput.dims[1], -1});
            std::string outProjName = "transformer.h." + std::to_string(i) + ".attn.out_proj.weight";
            Data realOutput;
            Linear(attnOutput, weight[outProjName], Data(), realOutput);

            // 1.4 MLP
            std::string fcInKeyName = "transformer.h." + std::to_string(i) + ".mlp.fc_in";
            std::string fcOutKeyName = "transformer.h." + std::to_string(i) + ".mlp.fc_out";
            Data middle;
            Linear(hiddenStates, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
            GeluNew(middle, middle);
            Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], hiddenStates);

            AddTo(hiddenStates, residual);
            AddTo(hiddenStates, realOutput);
        }

        LayerNorm(hiddenStates, weight["transformer.ln_f.weight"], weight["transformer.ln_f.bias"], -1, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], weight["lm_head.bias"], logits);

        std::vector <std::pair <float, int> > v;
        int base = logits.dims[logits.dims.size() - 2] - 1;
        for (int i = 0; i < logits.dims.back(); i++) {
            v.push_back(std::make_pair(((float*)logits.cpuData)[base * logits.dims.back() + i], i));
        }
        std::sort(v.begin(), v.end());
        std::reverse(v.begin(), v.end());

        float spend = GetSpan(st, std::chrono::system_clock::now());
        //printf("forward spend %f s.\n", spend);
        return v[0].second;
    }

    std::string MOSSModel::Response(const std::string &input, RuntimeResult retCb) {
        Data inputIds = this->weight.tokenizer.Encode(input);
        Data attentionMask = inputIds;
        Data positionIds = inputIds;
        std::vector<std::pair<Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(), Data()));
        }

        int len = inputIds.dims[1];
        for (int i = 0; i < len; i++) {
            ((float *) attentionMask.cpuData)[i] = 1;
            ((float *) positionIds.cpuData)[i] = i;
        }

        std::vector<float> results;
        std::string retString = "";
		int index = 0;
        while (true) {
            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues);
            if (ret == 106068) {
                break;
            }

            results.push_back(ret);
            std::string current = weight.tokenizer.Decode(
                    Data(DataType::FLOAT32, {(int) results.size()}, results)).c_str();
            retString += current;
			if (retCb)
				retCb(index++, current.c_str());
            fflush(stdout);
            results.clear();

            len++;
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) ret}));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {1, len}, std::vector<float>(len, 1.0f)));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) (len - 1)}));
        }

		if (retCb)
			retCb(index++, retString.c_str());
        // printf("%s\n", weight.tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results)).c_str());
        return retString;
    }

    void MOSSModel::SaveLowBitModel(const std::string &fileName, int bit) {
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {(float) 1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, std::vector<float>(1, 1.0f));
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {(float) (0)});
        std::vector<std::pair<Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(), Data()));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        this->weight.SaveLowBitModel(fileName, bit);
    }
}