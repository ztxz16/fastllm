//
// Created by huangyuyang on 5/12/23.
//

#include "utils.h"

#include "moss.h"

#include <cmath>

#include <chrono>

#include <algorithm>

#include <sstream>

#include <unordered_map>

namespace fastllm {
    extern double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2);

    MOSSModel::MOSSModel() {
        this->model_type = "moss";
        this->pre_prompt = "You are an AI assistant whose name is MOSS. ";
        this->user_role = "<|Human|>: ";
        this->bot_role = "<eoh>";
        this->history_sep = "";

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
                            const Data &positionIds, std::vector <std::pair <Data, Data> > &pastKeyValues,
                           const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                           std::vector <float> *retLogits) {
        auto st = std::chrono::system_clock::now();

        Data inputEmbeddings;
        Embedding(inputIds, this->weight["transformer.wte.weight"], inputEmbeddings);
        Data hiddenStates = inputEmbeddings;

        // MossBlock
        for (int i = 0; i < block_cnt; i++) {
            // 1.0 LayerNorm
            Data residual;
            Mul(hiddenStates, 1.0, residual);
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

            q.ToDevice(DataDevice::CPU);
            k.ToDevice(DataDevice::CPU);
            RotatePosition2D(q, positionIds);
            RotatePosition2D(k, positionIds);
            q.ToDevice(DataDevice::CUDA);
            k.ToDevice(DataDevice::CUDA);

            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

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
            attnWeights.ToDevice(DataDevice::CPU);
            CausalMask(attnWeights, k.dims[2] - q.dims[2]);
            attnWeights.ToDevice(DataDevice::CUDA);

            // 1.2.2 attentionMask
            // TODO: attentionMask, 这里似乎都是1, 暂且跳过了

            // 1.2.3 softmax
            Softmax(attnWeights, attnWeights, -1);

            // 1.2.4 headMask
            // TODO: headMask, 这里似乎都是None, 暂且跳过了

            // 1.2.5 attention_weights * v
            Data attnOutput;
            PermuteSelf(v, {0, 1, 3, 2});
            MatMulTransB(attnWeights, v, attnOutput);

            // 1.3
            PermuteSelf(attnOutput, {0, 2, 1, 3});
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

        logits.ToDevice(DataDevice::CPU);
        int ret = -1;
        if (generationConfig.IsSimpleGreedy()) {
            std::vector<std::pair<float, int> > v;
            int base = logits.dims[logits.dims.size() - 2] - 1;
            for (int i = 0; i < logits.dims.back(); i++) {
                v.push_back(std::make_pair(((float *) logits.cpuData)[base * logits.dims.back() + i], i));
            }
            std::sort(v.begin(), v.end());
            std::reverse(v.begin(), v.end());
            ret = v[0].second;
        } else if (!lastTokens.units.empty()) {
            ret = LLMSampling(logits, logits.dims[logits.dims.size() - 2] - 1, generationConfig, lastTokens.units[0]);
        }

        float spend = GetSpan(st, std::chrono::system_clock::now());
        //printf("forward spend %f s.\n", spend);
        return ret;
    }

    std::string MOSSModel::Response(const std::string &input,
                                    RuntimeResult retCb,
                                    const GenerationConfig &generationConfig) {
#ifdef PY_API
		size_t pos = input.rfind("time_stamp:");
		std::string prompt = (generationConfig.enable_hash_id && pos != -1)?  input.substr(0, pos):input;
		size_t hash_id = std::hash<std::string>{}(input);
        Data inputIds = this->weight.tokenizer.Encode(prompt);
#else
        Data inputIds = this->weight.tokenizer.Encode(input);
#endif
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
        LastTokensManager tokens (1, generationConfig.last_n);
        while (true) {
            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
            tokens.units[0].Push(ret);
            if (ret == 106068) {
                break;
            }

            results.push_back(ret);
            std::string current = weight.tokenizer.Decode(
                    Data(DataType::FLOAT32, {(int) results.size()}, results)).c_str();
            retString += current;
			if (retCb)
#ifdef PY_API
			{
				if(generationConfig.enable_hash_id){
					std::stringstream ss;
					ss << retString << "hash_id:"<<hash_id;
					retCb(index, pybind11::bytes(ss.str()));
				}else{
					retCb(index, pybind11::bytes(retString));
				}
			}
#else
				retCb(index, current.c_str());
#endif
            index++;
            fflush(stdout);
            results.clear();

            len++;

            inputIds.ToDevice(DataDevice::CPU);
            attentionMask.ToDevice(DataDevice::CPU);
            positionIds.ToDevice(DataDevice::CPU);
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) ret}));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {1, len}, std::vector<float>(len, 1.0f)));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) (len - 1)}));

            if (index == generationConfig.output_token_limit) {
                break;
            }
        }

		if (retCb)
#ifdef PY_API
		{
			if(generationConfig.enable_hash_id){
				std::stringstream ss;
				ss << retString << "hash_id:"<<hash_id;
				retCb(-1, pybind11::bytes(ss.str()));
			}else{
				retCb(-1, pybind11::bytes(retString));
			}
		}
#else
			retCb(-1, retString.c_str());
#endif
        return retString;
    }

    std::string MOSSModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string MOSSModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void MOSSModel::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {(float)bos_token_id});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        printf("finish.\n");
    }

    void
    MOSSModel::FillLLMInputs(std::vector<std::vector<float>> &inputTokens, const std::map<std::string, int> &params,
                             fastllm::Data &inputIds, fastllm::Data &attentionMask, fastllm::Data &positionIds) {
        int index = params.find("index")->second;
        int promptLen = params.find("promptLen")->second;
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);
        if (index == 0) {
            int seqLen = inputTokens[0].size();
            std::vector<float> vmask = std::vector<float>(seqLen, 1);
            std::vector<float> vpids = std::vector<float>(seqLen, 0);
            for (int i = 0; i < seqLen; i++) {
                vpids[i] = i;
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, inputTokens[0]));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vpids));
        } else {
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, inputTokens[0]));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {1, promptLen + index}, std::vector<float>(promptLen + index, 1.0f)));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) (promptLen + index - 1)}));
        }
    }
}
