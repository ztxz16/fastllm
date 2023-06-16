//
// Created by huangyuyang on 6/1/23.
//

#include "utils.h"

#include "vicuna.h"

namespace fastllm {
    VicunaModel::VicunaModel() {
        block_cnt = 32;
        rotary_dim = 128;

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
        weight.embeddingNames.insert("model.embed_tokens.weight");
    }

    void VicunaModel::RotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds) {
        int outer = data.dims[0] * data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        for (int o = 0; o < outer; o++) {
            int index = (int)((float*)positionIds.cpuData)[o];
            std::vector <float> &sin = this->sin[index];
            std::vector <float> &cos = this->cos[index];
            float *d = (float*)data.cpuData + o * spatial;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < rotary_dim && j < m / 2; j++) {
                    float a = d[j], b = d[j + m / 2];
                    d[j] = a * cos[j] - b * sin[j];
                    d[j + m / 2] = a * sin[j] + b * cos[j];
                }

                d += m;
            }
        }
    }

    void VicunaModel::LoadFromFile(const std::string &fileName) {
        this->weight.LoadFromFile(fileName);
    }

    int VicunaModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                              const fastllm::Data &positionIds, const Data &penaltyFactor,
                              std::vector<std::pair<Data, Data>> &pastKeyValues) {
TimeRecord timeRecord;
timeRecord.Clear();
timeRecord.Record();
        Data hiddenStates;
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        for (int i = 0; i < block_cnt; i++) {
            Data attenInput;
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    1e-6, attenInput);
timeRecord.Record("rms");
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            Data q, k, v;
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];

            Linear(attenInput, weight[qWeightName], Data(), q);
            Linear(attenInput, weight[kWeightName], Data(), k);
            Linear(attenInput, weight[vWeightName], Data(), v);

            std::vector <int> qkvSize = {bsz, seqlen, num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);
timeRecord.Record("qkv");

            q.ToDevice(DataDevice::CPU);
            k.ToDevice(DataDevice::CPU);
            RotatePosition2D(q, positionIds);
            RotatePosition2D(k, positionIds);
            q.ToDevice(DataDevice::CUDA);
            k.ToDevice(DataDevice::CUDA);

            qkvSize = {bsz * seqlen, num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            PermuteSelf(q, {1, 0, 2});
            PermuteSelf(k, {1, 0, 2});
            PermuteSelf(v, {1, 2, 0});

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
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

            // 1.2 Attention
            // 1.2.0 q * k^T
            Data attenWeights, attenOutput;
            MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim));
            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attenWeights, attentionMask, -1e100);
            }
            Softmax(attenWeights, attenWeights, -1);
            MatMulTransB(attenWeights, pastValue, attenOutput);

            attenOutput.Reshape({attenOutput.dims[1], attenOutput.dims[2], attenOutput.dims[3]});
            PermuteSelf(attenOutput, {1, 0, 2});
            attenOutput.Reshape({bsz, seqlen, -1});

            Data attenLastOutput;
            Linear(attenOutput, weight[oWeightName], Data(), attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
timeRecord.Record("attn");
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], 1e-6, attenInput);
timeRecord.Record("rms");
            Data w1, w2, w3;
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
timeRecord.Record("mlp linerar");
            Silu(w1, w1);
timeRecord.Record("mlp silu");
            MulTo(w1, w3);
timeRecord.Record("mlp mul");
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
timeRecord.Record("mlp linerar");
            AddTo(hiddenStates, w2);
timeRecord.Record("mlp add");
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], 1e-6, hiddenStates);
timeRecord.Record("rms");
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        logits.ToDevice(DataDevice::CPU);
timeRecord.Record("logits");
//timeRecord.Print();
        std::pair <float, int> ret = std::make_pair(-1e9, -1);
        int base = logits.dims[1] - 1;
        for (int i = 0; i < logits.dims.back(); i++) {
            ret = max(ret, std::make_pair(((float*)logits.cpuData)[base * logits.dims.back() + i], i));
        }
        return ret.second;
    }

    std::string VicunaModel::Response(const std::string& input, RuntimeResult retCb) {
        int bos = atoi(this->weight.dicts["bos"].c_str());
        int eos = atoi(this->weight.dicts["eos"].c_str());

        Data inputIds = this->weight.tokenizer.Encode(input);
        std::vector <float> ids;
        ids.push_back(bos);
        for (int i = 0; i < inputIds.Count(0); i++) {
            ids.push_back(((float*)inputIds.cpuData)[i]);
        }

        int seqLen = ids.size();
        inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, ids));

        std::vector <float> vmask = std::vector <float> (seqLen * seqLen, 0);
        std::vector <float> vpids = std::vector <float> (seqLen, 0);
        for (int i = 0; i < seqLen; i++) {
            vpids[i] = i;
            for (int j = i + 1; j < seqLen; j++) {
                vmask[i * seqLen + j] = 1;
            }
        }

        Data attentionMask = Data(DataType::FLOAT32, {seqLen, seqLen}, vmask);
        Data positionIds = Data(DataType::FLOAT32, {1, seqLen}, vpids);

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }

        std::string retString = "";
        int len = seqLen;
        std::vector <float> results;
        int index = 0;
        while (true) {
            auto st = std::chrono::system_clock::now();

            int ret = Forward(inputIds, attentionMask, positionIds, Data(), pastKeyValues);
            if (ret == eos) {
                break;
            }

            results.push_back(ret);
            std::string curString = weight.tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results)).c_str();
            retString += curString;
            if (retCb)
                retCb(index++, curString.c_str());
            fflush(stdout);
            results.clear();

            attentionMask.ToDevice(DataDevice::CPU);
            positionIds.ToDevice(DataDevice::CPU);
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)ret}));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)len}));
            len++;

            //printf("spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        }
        if (retCb)
            retCb(-1, retString.c_str());

        return retString;
    }

    void VicunaModel::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, Data(), pastKeyValues);
        printf("finish.\n");
    }

    void VicunaModel::SaveLowBitModel(const std::string &fileName, int bit) {
        WarmUp();
        this->weight.SaveLowBitModel(fileName, bit);
    }
}