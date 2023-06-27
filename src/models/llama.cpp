//
// Created by huangyuyang on 6/1/23.
//

#include "utils.h"

#include "llama.h"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    LlamaModel::LlamaModel() {
        this->model_type = "llama";

        // 默认使用alpaca的提示词和instruction
        this->pre_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";
        this->user_role = "### Instruction:\n";
        this->bot_role = "\n\n### Response:";
        this->history_sep = "</s>";

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
        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            for (int j = 0; j < sin[0].size(); j++) {
                fsin.push_back(sin[i][j]);
                fcos.push_back(cos[i][j]);
            }
        }
        sinData.CopyFrom(Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, fsin));
        cosData.CopyFrom(Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, fcos));
        weight.embeddingNames.insert("model.embed_tokens.weight");
    }

    int LlamaModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                              const fastllm::Data &positionIds, const Data &penaltyFactor,
                              std::vector<std::pair<Data, Data>> &pastKeyValues) {
        Data hiddenStates;
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        for (int i = 0; i < block_cnt; i++) {
            Data attenInput;
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    1e-6, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            Data q, k, v, qkv;
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];

            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
            } else {
                Linear(attenInput, weight[qWeightName], Data(), q);
                Linear(attenInput, weight[kWeightName], Data(), k);
                Linear(attenInput, weight[vWeightName], Data(), v);
            }

            std::vector <int> qkvSize = {bsz, seqlen, num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            fastllm::LlamaRotatePosition2D(q, positionIds, sinData, cosData, rotary_dim);
            fastllm::LlamaRotatePosition2D(k, positionIds, sinData, cosData, rotary_dim);

            qkvSize = {bsz * seqlen, num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            PermuteSelf(q, {1, 0, 2});
            PermuteSelf(k, {1, 0, 2});
            PermuteSelf(v, {1, 0, 2});

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
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }

            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 1);

            // 1.2 Attention
            // 1.2.0 q * k^T
            Data attenWeights, attenOutput;
            MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim));
            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attenWeights, attentionMask, -10000);
            }
            Softmax(attenWeights, attenWeights, -1);
            MatMul(attenWeights, pastValue, attenOutput);

            attenOutput.Reshape({attenOutput.dims[1], attenOutput.dims[2], attenOutput.dims[3]});
            PermuteSelf(attenOutput, {1, 0, 2});
            attenOutput.Reshape({bsz, seqlen, -1});

            Data attenLastOutput;
            Linear(attenOutput, weight[oWeightName], Data(), attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], 1e-6, attenInput);
            Data w1, w2, w3;
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
            AddTo(hiddenStates, w2);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], 1e-6, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        logits.ToDevice(DataDevice::CPU);

        if (this->do_sample && penaltyFactor.dims == logits.dims) {
            RepeatPenalty(logits, penaltyFactor);
        }

        std::pair <float, int> ret = std::make_pair(-1e9, -1);
        int base = logits.dims[1] - 1;
        for (int i = 0; i < logits.dims.back(); i++) {
            ret = max(ret, std::make_pair(((float*)logits.cpuData)[base * logits.dims.back() + i], i));
        }
        return ret.second;
    }

    std::vector <int> LlamaModel::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, const Data &penaltyFactor,
                            std::vector<std::pair<Data, Data>> &pastKeyValues) {
        Data hiddenStates;
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            Data attenInput;
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    1e-6, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            Data q, k, v, qkv;
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
            } else {
                Linear(attenInput, weight[qWeightName], Data(), q);
                Linear(attenInput, weight[kWeightName], Data(), k);
                Linear(attenInput, weight[vWeightName], Data(), v);
            }

            std::vector <int> qkvSize = {bsz, seqlen, num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            fastllm::LlamaRotatePosition2D(q, positionIds, sinData, cosData, rotary_dim);
            fastllm::LlamaRotatePosition2D(k, positionIds, sinData, cosData, rotary_dim);
            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            qkvSize = {bsz * num_attention_heads, seqlen, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

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
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }

            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 1);

            // 1.2 Attention
            // 1.2.0 q * k^T
            Data attenWeights, attenOutput;
            MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim));
            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attenWeights, attentionMask, -10000);
            }
            Softmax(attenWeights, attenWeights, -1);
            MatMul(attenWeights, pastValue, attenOutput);

            attenOutput.Reshape({attenOutput.dims[1], attenOutput.dims[2], attenOutput.dims[3]});
            PermuteSelf(attenOutput, {1, 0, 2});
            attenOutput.Reshape({seqlen, bsz, -1});
            PermuteSelf(attenOutput, {1, 0, 2});

            Data attenLastOutput;
            Linear(attenOutput, weight[oWeightName], Data(), attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], 1e-6, attenInput);
            Data w1, w2, w3;
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
            AddTo(hiddenStates, w2);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], 1e-6, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        logits.ToDevice(DataDevice::CPU);
        if (this->do_sample && penaltyFactor.dims == logits.dims) {
            // RepeatPenalty(logits, penaltyFactor);
        }

        std::vector <int> lastRet;
        for (int b = 0; b < batch; b++) {
            int base = b * logits.dims[1] + logits.dims[1] - 1;
            std::pair <float, int> ret = std::make_pair(-1e9, -1);
            for (int i = 0; i < logits.dims.back(); i++) {
                ret = max(ret, std::make_pair(((float *) logits.cpuData)[base * logits.dims.back() + i], i));
            }
            lastRet.push_back(ret.second);
        }
        return lastRet;
    }

    std::vector <int> LlamaModel::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <Data*> &penaltyFactor,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues) {
        Data hiddenStates;
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            Data attenInput;
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    1e-6, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            Data q, k, v, qkv;
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
            } else {
                Linear(attenInput, weight[qWeightName], Data(), q);
                Linear(attenInput, weight[kWeightName], Data(), k);
                Linear(attenInput, weight[vWeightName], Data(), v);
            }

            Data attenOutput = Data(DataType::FLOAT32);
            int total = 0;
            std::vector <Data> curKs, curVs, curQs;
            curKs.resize(batch);
            curVs.resize(batch);
            curQs.resize(batch);
            for (int b = 0; b < batch; b++) {
                Split(k, 1, total, total + seqLens[b], curKs[b]);
                Split(v, 1, total, total + seqLens[b], curVs[b]);
                Split(q, 1, total, total + seqLens[b], curQs[b]);
                total += seqLens[b];
            }

            for (int b = 0; b < batch; b++) {
                auto &q = curQs[b], &k = curKs[b], &v = curVs[b];

                std::vector<int> qkvSize = {bsz, seqLens[b], num_attention_heads, -1};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                fastllm::LlamaRotatePosition2D(q, *positionIds[b], sinData, cosData, rotary_dim);
                fastllm::LlamaRotatePosition2D(k, *positionIds[b], sinData, cosData, rotary_dim);
                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});

                qkvSize = {bsz * num_attention_heads, seqLens[b], -1};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
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
                        (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                       || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector<int>{v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }

                CatDirect(pastKey, k, 1);
                CatDirect(pastValue, v, 1);

                // 1.2 Attention
                // 1.2.0 q * k^T
                Data attenWeights, curAttenOutput;
                MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim));
                attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
                if (attentionMask[b] != nullptr) {
                    AttentionMask(attenWeights, *attentionMask[b], -10000);
                }
                Softmax(attenWeights, attenWeights, -1);
                MatMul(attenWeights, pastValue, curAttenOutput);
                curAttenOutput.Reshape({curAttenOutput.dims[1], curAttenOutput.dims[2], curAttenOutput.dims[3]});
                PermuteSelf(curAttenOutput, {1, 0, 2});
                curAttenOutput.Reshape({seqLens[b], bsz, -1});
                PermuteSelf(curAttenOutput, {1, 0, 2});
                if (attenOutput.dims.size() == 0) {
                    std::vector <int> dims = curAttenOutput.dims;
                    dims[1] = total;
                    attenOutput.Expansion(dims);
                }
                CatDirect(attenOutput, curAttenOutput, 1);
            }

            Data attenLastOutput;
            Linear(attenOutput, weight[oWeightName], Data(), attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], 1e-6, attenInput);
            Data w1, w2, w3;
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
            AddTo(hiddenStates, w2);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], 1e-6, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        logits.ToDevice(DataDevice::CPU);
        //if (this->do_sample && penaltyFactor.dims == logits.dims) {
            // RepeatPenalty(logits, penaltyFactor);
        //}

        std::vector <int> lastRet;
        int total = 0;
        for (int b = 0; b < batch; b++) {
            int base = (total + seqLens[b] - 1);
            std::pair <float, int> ret = std::make_pair(-1e9, -1);
            for (int i = 0; i < logits.dims.back(); i++) {
                ret = max(ret, std::make_pair(((float *) logits.cpuData)[base * logits.dims.back() + i], i));
            }
            lastRet.push_back(ret.second);
        }
        return lastRet;
    }

    std::string LlamaModel::Response(const std::string& input, RuntimeResult retCb) {
//auto st = std::chrono::system_clock::now();
        Data inputIds = this->weight.tokenizer.Encode(input);
        std::vector <float> ids;
        ids.push_back(bos_token_id);
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

        int vocabSize = this->weight.tokenizer.tokenToStringDict.size();
        TokenPenaltyManager tokenPenaltyManager;
        if (this->do_sample) {
            tokenPenaltyManager.Init(vocabSize, this->last_n, this->repeat_penalty);
            /*for (int i = std::max(0, (int)ids.size() - this->last_n); i < ids.size(); i++) {
                tokenPenaltyManager.InsertToken((int)(ids[i] + 1e-6));
            }*/
        }

        while (true) {
            auto st = std::chrono::system_clock::now();

            int ret = Forward(inputIds, attentionMask, positionIds, tokenPenaltyManager.penalty, pastKeyValues);
            if (ret == eos_token_id) {
                break;
            }

            results.push_back(ret);
            std::string curString = weight.tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results)).c_str();
            retString += curString;
            if (retCb)
#ifdef PY_API
				retCb(index++, pybind11::bytes(retString));
#else
                retCb(index++, curString.c_str());
#endif

            if (index == this->output_token_limit) {
                break;
            }
            results.clear();

            attentionMask.ToDevice(DataDevice::CPU);
            positionIds.ToDevice(DataDevice::CPU);
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)ret}));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)len}));
            if (do_sample) {
                tokenPenaltyManager.InsertToken(ret);
            }
            len++;

            //printf("spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        }
        if (retCb)
#ifdef PY_API
			retCb(-1, pybind11::bytes(retString));
#else
            retCb(-1, retString.c_str());
#endif

        return retString;
    }

    void LlamaModel::ResponseBatch(const std::vector<std::string> &inputs, std::vector<std::string> &outputs,
                                   RuntimeResultBatch retCb) {
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
            maxLen = std::max(maxLen, (int)inputTokens[i].Count(0) + 1);
            seqLens[i] = (int)inputTokens[i].Count(0) + 1;
        }

        std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
        std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
        std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
        for (int i = 0; i < batch; i++) {
            Data &tokens = inputTokens[i];
            int len = tokens.Count(0), base = maxLen - 1 - len;
            ids[i * maxLen + base] = bos_token_id;
            for (int j = 0; j < len; j++) {
                ids[i * maxLen + base + 1 + j] = ((float*)tokens.cpuData)[j];
            }
            len += 1;

            for (int j = 0; j < len; j++) {
                vpids[i * maxLen + base + j] = j;
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

        Data inputIds = Data(DataType::FLOAT32, {batch, maxLen}, ids);
        Data attentionMask = Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask);
        Data positionIds = Data(DataType::FLOAT32, {batch, maxLen}, vpids);

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }

        std::string retString = "";
        std::vector <int> lens = seqLens;
        std::vector <bool> isEnding = std::vector <bool> (batch, false);
        std::vector <float> results;
        int index = 0;

        int vocabSize = this->weight.tokenizer.tokenToStringDict.size();
        TokenPenaltyManager tokenPenaltyManager;
        this->do_sample = false;
        if (this->do_sample) {
            tokenPenaltyManager.Init(vocabSize, this->last_n, this->repeat_penalty);
            /*for (int i = std::max(0, (int)ids.size() - this->last_n); i < ids.size(); i++) {
                tokenPenaltyManager.InsertToken((int)(ids[i] + 1e-6));
            }*/
        }

        while (true) {
            auto st = std::chrono::system_clock::now();
            std::vector <int> ret = ForwardBatch(batch, inputIds, attentionMask, positionIds, Data(), pastKeyValues);
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
            }

            if (endingCount == batch) {
                break;
            }
            if (retCb) {
                retCb(index++, curStrings);
            }

            maxLen++;
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                pids[i] = lens[i];
                lens[i]++;
                for (int j = 0; j < maxLen - lens[i]; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }
            positionIds.ToDevice(DataDevice::CPU);
            attentionMask.ToDevice(DataDevice::CPU);
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
            if (index == this->output_token_limit) {
                break;
            }
            if (do_sample) {
                //tokenPenaltyManager.InsertToken(ret);
            }

            //printf("spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        }
        if (retCb)
            retCb(-1, outputs);
    }

    std::string LlamaModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string LlamaModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void LlamaModel::WarmUp() {
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

    int LlamaModel::LaunchResponseTokens(const std::vector<int> &inputTokens) {
        mainLoopLocker.lock();
        if (mainLoop == nullptr) {
            if (mainLoop == nullptr) {
                mainLoop = new std::thread([](LlamaModel *model) {
                    while (true) {
                        std::vector <Data*> attentionMasks;
                        std::vector <Data*> positionIds;
                        std::vector <std::pair <Data*, Data*> > pastKeyValues;
                        std::vector<float> ids;
                        std::vector<int> seqLens;
                        model->dictLocker.lock();
                        for (auto &it: model->responseContextDict.dicts) {
                            if (it.second->isEnding) {
                                continue;
                            }
                            if (it.second->preTokens == 0) {
                                ids.push_back(model->bos_token_id);
                                for (int i = 0; i < it.second->currentTokens.size(); i++) {
                                    ids.push_back(it.second->currentTokens[i]);
                                }

                                int seqLen = it.second->currentTokens.size() + 1;
                                seqLens.push_back(seqLen);

                                std::vector <float> vmask = std::vector <float> (seqLen * seqLen, 0);
                                std::vector <float> vpids = std::vector <float> (seqLen, 0);
                                for (int i = 0; i < seqLen; i++) {
                                    vpids[i] = i;
                                    for (int j = i + 1; j < seqLen; j++) {
                                        vmask[i * seqLen + j] = 1;
                                    }
                                }
                                it.second->intParams["len"] = seqLen;

                                attentionMasks.push_back(new Data(DataType::FLOAT32, {seqLen, seqLen}, vmask));
                                positionIds.push_back(new Data(DataType::FLOAT32, {2, seqLen}, vpids));
                            } else {
                                int ret = it.second->currentTokens[0];
                                if (model->do_sample) {
                                    //it.second->tokenPenaltyManager.InsertToken(ret);
                                }

                                seqLens.push_back(1);
                                ids.push_back(ret);
                                attentionMasks.push_back(nullptr);
                                positionIds.push_back(new Data(DataType::FLOAT32, {1, 1}, {(float)it.second->intParams["len"]}));
                                it.second->intParams["len"]++;
                            }

                            it.second->preTokens += seqLens.back();
                            for (int i = 0; i < model->block_cnt; i++) {
                                pastKeyValues.push_back(std::make_pair(&it.second->pastKeyValues[i].first,
                                                                       &it.second->pastKeyValues[i].second));
                            }
                        }

                        if (seqLens.size() > 0) {
#ifdef USE_CUDA
                            FastllmCudaClearBigBuffer();
#endif
                            Data inputIds = Data(DataType::FLOAT32, {1, (int) ids.size()}, ids);
                            std::vector<int> ret = model->ForwardBatch(seqLens.size(), inputIds, attentionMasks,
                                                                       positionIds, std::vector <Data*> (),
                                                                       seqLens, pastKeyValues);
                            int idx = 0;
                            for (auto &it: model->responseContextDict.dicts) {
                                if (it.second->isEnding) {
                                    continue;
                                }
                                int curRet = ret[idx++];
                                if (curRet == model->eos_token_id) {
                                    it.second->isEnding = true;
                                } else {
                                    it.second->currentTokens = std::vector<int>{curRet};
                                    it.second->resultTokenQueue.push(curRet);
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
                        pthread_yield();
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
        dictLocker.unlock();
        return handleId;
    }

    std::pair<bool, std::vector<int>> LlamaModel::FetchResponseTokens(int handleId) {
        dictLocker.lock();
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            dictLocker.unlock();
            return std::make_pair(false, std::vector <int> ());
        } else {
            std::vector <int> ret;
            while (context->resultTokenQueue.size() > 0) {
                ret.push_back(context->resultTokenQueue.front());
                context->resultTokenQueue.pop();
            }
            bool remain = (!context->isEnding || ret.size() > 0);
            if (!remain) {
                responseContextDict.RemoveHandle(handleId);
            }
            dictLocker.unlock();
            return std::make_pair(remain, ret);
        }
    }
}
