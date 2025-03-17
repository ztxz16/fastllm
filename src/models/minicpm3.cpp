//
// Created by huangyuyang on 6/1/23.
//

#include "utils.h"

#include "minicpm3.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    
    MiniCpm3Model::MiniCpm3Model() {
        this->model_type = "minicpm3";

        this->history_sep = "";
        this->pre_prompt = "";
        this->user_role = "";
        this->bot_role = "";

        block_cnt = 40;
        rotary_dim = 32; // todo 这部分理论上应该放在 InitParams 之后

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

    void MiniCpm3Model::InitParams() {
        basellm::InitParams();
        if (this->weight.dicts.find("scale_emb") != this->weight.dicts.end()) {
            this->embed_scale = std::stof(this->weight.dicts["scale_emb"]);
        }
        if (this->weight.dicts.find("scale_depth") != this->weight.dicts.end()) {
            float scale_depth = std::stof(this->weight.dicts["scale_depth"]);
            this->attention_scale = scale_depth / std::sqrt(block_cnt);
        }
        if (this->weight.dicts.find("dim_model_base") != this->weight.dicts.end()) {
            int32_t dim_model_base = std::stoi(this->weight.dicts["dim_model_base"]);
            this->rms_scale = 1.f / (this->embed_dim / dim_model_base);
        }
        if (this->weight.dicts.find("hidden_size") != this->weight.dicts.end()) {
            this->hidden_size = std::stoi(this->weight.dicts["hidden_size"]);
        }
        if (this->weight.dicts.find("qk_nope_head_dim") != this->weight.dicts.end()) {
            this->qk_nope_head_dim = std::stoi(this->weight.dicts["qk_nope_head_dim"]);
        }
        if (this->weight.dicts.find("qk_rope_head_dim") != this->weight.dicts.end()) {
            this->qk_rope_head_dim = std::stoi(this->weight.dicts["qk_rope_head_dim"]);
        }
        if (this->weight.dicts.find("kv_lora_rank") != this->weight.dicts.end()) {
            this->kv_lora_rank = std::stoi(this->weight.dicts["kv_lora_rank"]);
        }
        weight.tokenizer.SetSpecialTokens({{"<s>", 2}, {"<s>", 1}, {"<unk>", 0}, {"<|im_start|>", 73441}, {"<|im_end|>", 73440}, {"<|tool_call|>", 73442}, 
                                          {"<|execute_start|>", 73443}, {"<|execute_end|>", 73444}, {"<|fim_prefix|>", 73445}, {"<|fim_middle|>", 73446}, {"<|fim_suffix|>", 73447}});
    }

    int MiniCpm3Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> MiniCpm3Model::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {

        int maxLen = inputIds.dims[1];
        int v_head_dim = this->hidden_size / this->num_attention_heads;
        Data hiddenStates;
        Data attenInput;
        Data qa, qa_norm, qb, q_nope, q_rope;
        Data kva, compressed_kv, k_rope, kv_norm, kvb;
        Data k_nope, k_rope_expand, value_states, query_states, key_states;
        Data attenWeights, attenOutput, attenLastOutput;
        Data w1, w2, w3;
        
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        Mul(hiddenStates, embed_scale, hiddenStates);
        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    1e-5, attenInput);
            std::string qaWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_a_proj.weight";
            std::string qbWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_b_proj.weight";
            std::string kvaWeightName = "model.layers." + std::to_string(i) + ".self_attn.kv_a_proj_with_mqa.weight";
            std::string kvbWeightName = "model.layers." + std::to_string(i) + ".self_attn.kv_b_proj.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            Linear(attenInput, weight[qaWeightName], Data(), qa);
            RMSNorm(qa, this->weight["model.layers." + std::to_string(i) + ".self_attn.q_a_layernorm.weight"], 
                1e-5, qa_norm);
            Linear(qa_norm, weight[qbWeightName], Data(), qb);
            qb.Reshape({bsz, seqlen, num_attention_heads, -1});
            PermuteSelf(qb, {0, 2, 1, 3});
            Split(qb, -1, 0, this->qk_nope_head_dim, q_nope);
            Split(qb, -1, this->qk_nope_head_dim, this->qk_nope_head_dim + this->qk_rope_head_dim, q_rope);

            Linear(attenInput, weight[kvaWeightName], Data(), kva);
            Split(kva, -1, 0, this->kv_lora_rank, compressed_kv);
            Split(kva, -1, this->kv_lora_rank, this->kv_lora_rank + this->qk_rope_head_dim, k_rope);
            k_rope.Reshape({bsz, 1, seqlen, this->qk_rope_head_dim});
            RMSNorm(compressed_kv, this->weight["model.layers." + std::to_string(i) + ".self_attn.kv_a_layernorm.weight"], 
                1e-5, kv_norm);
            Linear(kv_norm, weight[kvbWeightName], Data(), kvb);
            kvb.Reshape({bsz, seqlen, num_attention_heads, qk_nope_head_dim + v_head_dim});
            PermuteSelf(kvb, {0, 2, 1, 3});
            Split(kvb, -1, 0, qk_nope_head_dim, k_nope);
            Split(kvb, -1, qk_nope_head_dim, qk_nope_head_dim + v_head_dim, value_states);

            PermuteSelf(q_rope, {0, 2, 1, 3});
            PermuteSelf(k_rope, {0, 2, 1, 3});
            fastllm::LlamaRotatePosition2D(q_rope, positionIds, sinData, cosData, rotary_dim);
            fastllm::LlamaRotatePosition2D(k_rope, positionIds, sinData, cosData, rotary_dim);
            PermuteSelf(q_rope, {0, 2, 1, 3});
            PermuteSelf(k_rope, {0, 2, 1, 3});
            Cat(q_nope, q_rope, -1, query_states);

            k_rope.Reshape({bsz, seqlen * qk_rope_head_dim});
            k_rope_expand.CopyFrom(k_rope);
            k_rope_expand.Expansion({bsz, num_attention_heads * seqlen * qk_rope_head_dim});
            for (int i = 1; i < num_attention_heads; i++)
                CatDirect(k_rope_expand, k_rope, 1);
            k_rope_expand.expansionDims.clear();
            k_rope_expand.Reshape({bsz, num_attention_heads, seqlen, qk_rope_head_dim});
            Cat(k_nope, k_rope_expand, -1, key_states);
            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(k_nope.dataDevice);
                pastValue.ToDevice(k_nope.dataDevice);
            }
            key_states.Reshape({bsz * num_attention_heads, seqlen, -1});
            value_states.Reshape({bsz * num_attention_heads, seqlen, -1});

            int key_unitLen = 96;
#ifdef USE_CUDA
            key_unitLen = 192;
#endif
            while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || key_states.dims[1] > pastKey.expansionDims[1]))
                   || (pastKey.dims.size() > 0 && pastKey.dims[1] + key_states.dims[1] > pastKey.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector <int> {key_states.dims[0], ((key_states.dims[1] - 1) / key_unitLen + 1) * key_unitLen, key_states.dims[2]};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((key_states.dims[1] - 1) / key_unitLen + 1) * key_unitLen;
                }
                pastKey.Expansion(newDims);
            }
            int value_unitLen = 64;
#ifdef USE_CUDA
            value_unitLen = 128;
#endif
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || value_states.dims[1] > pastValue.expansionDims[1]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[1] + value_states.dims[1] > pastValue.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {value_states.dims[0], ((value_states.dims[1] - 1) / value_unitLen + 1) * value_unitLen, value_states.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((value_states.dims[1] - 1) / value_unitLen + 1) * value_unitLen;
                }
                pastValue.Expansion(newDims);
            }
            CatDirect(pastKey, key_states, 1);
            CatDirect(pastValue, value_states, 1);
            
            // 1.2 Attention
            // 1.2.0 q * k^T
            query_states.Reshape({bsz * num_attention_heads, seqlen, -1});
            MatMulTransB(query_states, pastKey, attenWeights, 1.0 / sqrt(v_head_dim));
            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
            if (seqlen > 1) {
                int promptLen = pastKey.dims[1];
                std::vector <float> vmask = std::vector <float> (seqlen * promptLen, 0);
                for (int i = 0; i < seqlen; i++)
                    for (int j = i + 1; j < seqlen; j++)
                        vmask[i * promptLen + (promptLen - seqlen + j)] = 1;
                AttentionMask(attenWeights, Data(DataType::FLOAT32, {seqlen, promptLen}, vmask), -10000);
            }
            Softmax(attenWeights, attenWeights, -1);
            MatMul(attenWeights, pastValue, attenOutput);
            attenOutput.Reshape({bsz, num_attention_heads, seqlen, v_head_dim});
            PermuteSelf(attenOutput, {0, 2, 1, 3});
            attenOutput.Reshape({bsz, seqlen, num_attention_heads * v_head_dim});
            Linear(attenOutput, weight[oWeightName], Data(), attenLastOutput);
            AddTo(hiddenStates, attenLastOutput, this->attention_scale);
            
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], 1e-5, attenInput);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);

            AddTo(hiddenStates, w2, this->attention_scale);
        }

        Data logits, topk;
        Data tempHiddenStates;
        Data *lastHiddenStates;
        if (maxLen > 1) {
            Split(hiddenStates, 1, maxLen - 1, maxLen, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        } else {
            lastHiddenStates = &hiddenStates;
        }

        std::vector <int> lastRet;
        {
            auto &hiddenStates = *lastHiddenStates;
            RMSNorm(hiddenStates, weight["model.norm.weight"], 1e-5, hiddenStates);
            Mul(hiddenStates, this->rms_scale, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
            if (generationConfig.IsSimpleGreedy()) {
                TopK(logits, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    int base = b;
                    lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
                }
            } else {
                for (int b = 0; b < batch; b++) {
                    int base = b * logits.dims[1] + logits.dims[1] - 1;
                    lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
                }
            }
        }
        return lastRet;
    }

    std::vector <int> MiniCpm3Model::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {

        int v_head_dim = this->hidden_size / this->num_attention_heads;
        Data hiddenStates;
        Data attenInput;
        Data qa, qa_norm, qb, batch_q_nope, batch_q_rope;
        Data kva, compressed_kv, batch_k_rope, kv_norm, kvb;
        Data batch_k_nope, k_rope_expand, batch_value_states, query_states, key_states;
        Data attenWeights, curAttenOutput, attenLastOutput;
        Data w1, w2, w3;
        
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        Mul(hiddenStates, embed_scale, hiddenStates);
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    1e-5, attenInput);
            std::string qaWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_a_proj.weight";
            std::string qbWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_b_proj.weight";
            std::string kvaWeightName = "model.layers." + std::to_string(i) + ".self_attn.kv_a_proj_with_mqa.weight";
            std::string kvbWeightName = "model.layers." + std::to_string(i) + ".self_attn.kv_b_proj.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], b_seqlen = attenInput.dims[1];
            Linear(attenInput, weight[qaWeightName], Data(), qa);
            RMSNorm(qa, this->weight["model.layers." + std::to_string(i) + ".self_attn.q_a_layernorm.weight"], 
                1e-5, qa_norm);
            Linear(qa_norm, weight[qbWeightName], Data(), qb);
            qb.Reshape({bsz, b_seqlen, num_attention_heads, -1});
            PermuteSelf(qb, {0, 2, 1, 3});
            Split(qb, -1, 0, this->qk_nope_head_dim, batch_q_nope);
            Split(qb, -1, this->qk_nope_head_dim, this->qk_nope_head_dim + this->qk_rope_head_dim, batch_q_rope);

            Linear(attenInput, weight[kvaWeightName], Data(), kva);
            Split(kva, -1, 0, this->kv_lora_rank, compressed_kv);
            Split(kva, -1, this->kv_lora_rank, this->kv_lora_rank + this->qk_rope_head_dim, batch_k_rope);
            batch_k_rope.Reshape({bsz, 1, b_seqlen, this->qk_rope_head_dim});
            RMSNorm(compressed_kv, this->weight["model.layers." + std::to_string(i) + ".self_attn.kv_a_layernorm.weight"], 
                1e-5, kv_norm);
            Linear(kv_norm, weight[kvbWeightName], Data(), kvb);
            kvb.Reshape({bsz, b_seqlen, num_attention_heads, qk_nope_head_dim + v_head_dim});
            PermuteSelf(kvb, {0, 2, 1, 3});
            Split(kvb, -1, 0, qk_nope_head_dim, batch_k_nope);
            Split(kvb, -1, qk_nope_head_dim, qk_nope_head_dim + v_head_dim, batch_value_states);

            Data attenOutput = Data(DataType::FLOAT32);
            int total = 0;
            std::vector <Data> curQNs, curQRs, curKNs, curKRs, curVs;
            curQNs.resize(batch);
            curQRs.resize(batch);
            curKNs.resize(batch);
            curKRs.resize(batch);
            curVs.resize(batch);
            for (int b = 0; b < batch; b++) {
                Split(batch_q_nope, 2, total, total + seqLens[b], curQNs[b]);
                Split(batch_q_rope, 2, total, total + seqLens[b], curQRs[b]);
                Split(batch_k_nope, 2, total, total + seqLens[b], curKNs[b]);
                Split(batch_k_rope, 2, total, total + seqLens[b], curKRs[b]);
                Split(batch_value_states, 2, total, total + seqLens[b], curVs[b]);
                total += seqLens[b];
            }

            for (int b = 0; b < batch; b++) {
                int seqlen = seqLens[b];
                auto &q_nope = curQNs[b], &q_rope = curQRs[b];
                auto &k_nope = curKNs[b], &k_rope = curKRs[b], &value_states = curVs[b];

                PermuteSelf(q_rope, {0, 2, 1, 3});
                PermuteSelf(k_rope, {0, 2, 1, 3});
                fastllm::LlamaRotatePosition2D(q_rope, *positionIds[b], sinData, cosData, rotary_dim);
                fastllm::LlamaRotatePosition2D(k_rope, *positionIds[b], sinData, cosData, rotary_dim);
                PermuteSelf(q_rope, {0, 2, 1, 3});
                PermuteSelf(k_rope, {0, 2, 1, 3});
                Cat(q_nope, q_rope, -1, query_states);

                k_rope.Reshape({bsz, seqlen * qk_rope_head_dim});
                k_rope_expand.CopyFrom(k_rope);
                k_rope_expand.Expansion({bsz, num_attention_heads * seqlen * qk_rope_head_dim});
                for (int i = 1; i < num_attention_heads; i++)
                    CatDirect(k_rope_expand, k_rope, 1);
                k_rope_expand.expansionDims.clear();
                k_rope_expand.Reshape({bsz, num_attention_heads, seqlen, qk_rope_head_dim});
                Cat(k_nope, k_rope_expand, -1, key_states);

                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(k_nope.dataDevice);
                    pastValue.ToDevice(k_nope.dataDevice);
                }
                key_states.Reshape({bsz * num_attention_heads, seqlen, -1});
                value_states.Reshape({bsz * num_attention_heads, seqlen, -1});

                int key_unitLen = 96;
    #ifdef USE_CUDA
                key_unitLen = 192;
    #endif
                while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || key_states.dims[1] > pastKey.expansionDims[1]))
                       || (pastKey.dims.size() > 0 && pastKey.dims[1] + key_states.dims[1] > pastKey.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector <int> {key_states.dims[0], ((key_states.dims[1] - 1) / key_unitLen + 1) * key_unitLen, key_states.dims[2]};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((key_states.dims[1] - 1) / key_unitLen + 1) * key_unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                int value_unitLen = 64;
    #ifdef USE_CUDA
                value_unitLen = 128;
    #endif
                while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || value_states.dims[1] > pastValue.expansionDims[1]))
                       || (pastValue.dims.size() > 0 && pastValue.dims[1] + value_states.dims[1] > pastValue.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector <int> {value_states.dims[0], ((value_states.dims[1] - 1) / value_unitLen + 1) * value_unitLen, value_states.dims[2]};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((value_states.dims[1] - 1) / value_unitLen + 1) * value_unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
                CatDirect(pastKey, key_states, 1);
                CatDirect(pastValue, value_states, 1);
                
                // 1.2 Attention
                // 1.2.0 q * k^T
                query_states.Reshape({bsz * num_attention_heads, seqlen, -1});
                MatMulTransB(query_states, pastKey, attenWeights, 1.0 / sqrt(v_head_dim));
                attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
                if (seqlen > 1) {
                    int promptLen = pastKey.dims[1];
                    std::vector <float> vmask = std::vector <float> (seqlen * promptLen, 0);
                    for (int i = 0; i < seqlen; i++)
                        for (int j = i + 1; j < seqlen; j++)
                            vmask[i * promptLen + (promptLen - seqlen + j)] = 1;
                    AttentionMask(attenWeights, Data(DataType::FLOAT32, {seqlen, promptLen}, vmask), -10000);
                }
                Softmax(attenWeights, attenWeights, -1);
                MatMul(attenWeights, pastValue, curAttenOutput);
                curAttenOutput.Reshape({bsz, num_attention_heads, seqlen, v_head_dim});
                PermuteSelf(curAttenOutput, {0, 2, 1, 3});
                curAttenOutput.Reshape({bsz, seqlen, num_attention_heads * v_head_dim});
                if (attenOutput.dims.size() == 0) {
                    std::vector <int> dims = curAttenOutput.dims;
                    dims[1] = total;
                    attenOutput.Expansion(dims);
                }
                CatDirect(attenOutput, curAttenOutput, 1);
            }
            Linear(attenOutput, weight[oWeightName], Data(), attenLastOutput);
            AddTo(hiddenStates, attenLastOutput, this->attention_scale);
            
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], 1e-5, attenInput);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);

            AddTo(hiddenStates, w2, this->attention_scale);
        }

        Data logits, curLogit;
        RMSNorm(hiddenStates, weight["model.norm.weight"], 1e-5, hiddenStates);
        Mul(hiddenStates, this->rms_scale, hiddenStates);
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        std::vector <int> lastRet;
        int total = 0;
        for (int b = 0; b < batch; b++) {
            Split(logits, 1, total + seqLens[b] - 1, total + seqLens[b], curLogit);
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

}
