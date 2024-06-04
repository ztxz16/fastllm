//
// Created by huangyuyang on 5/11/24.
//

#include "utils.h"

#include "deepseekv2.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

#include "json11.hpp"

namespace fastllm {
    extern std::vector <float> GetInterLeavePowerOf2(int n);
    extern std::vector <float> GetInterleave(int n);

    DeepSeekV2Model::DeepSeekV2Model() {
        this->model_type = "deepseek_v2";

        // 默认使用alpaca的提示词和instruction
        this->pre_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";
        this->user_role = "### Instruction:\n";
        this->bot_role = "\n\n### Response:";
        this->history_sep = "</s>";

        weight.embeddingNames.insert("model.embed_tokens.weight");
    }

    void DeepSeekV2Model::InitParams() {
        basellm::InitParams();
        rope_scaling_type = this->weight.dicts["rope_scaling.type"];
        AssertInFastLLM(rope_scaling_type == "yarn", "Fastllm.DeepSeekV2: Only support ropescaling.type = yarn.");

        if (this->weight.dicts.find("rope_scaling.beta_fast") != this->weight.dicts.end()) {
            rope_scaling_beta_fast = atoi(this->weight.dicts["rope_scaling.beta_fast"].c_str());
        } else {
            rope_scaling_beta_fast = 32;
        }
        if (this->weight.dicts.find("rope_scaling.beta_slow") != this->weight.dicts.end()) {
            rope_scaling_beta_slow = atoi(this->weight.dicts["rope_scaling.beta_slow"].c_str());
        } else {
            rope_scaling_beta_slow = 1;
        }
        if (this->weight.dicts.find("rope_scaling.original_max_position_embeddings") != this->weight.dicts.end()) {
            rope_scaling_original_max_position_embeddings = atoi(this->weight.dicts["rope_scaling.original_max_position_embeddings"].c_str());
        } else {
            rope_scaling_original_max_position_embeddings = 4096;
        }
        if (this->weight.dicts.find("rope_scaling.mscale") != this->weight.dicts.end()) {
            rope_scaling_mscale = atof(this->weight.dicts["rope_scaling.mscale"].c_str());
        } else {
            rope_scaling_mscale = 0.707;
        }
        if (this->weight.dicts.find("rope_scaling.mscale_all_dim") != this->weight.dicts.end()) {
            rope_scaling_mscale_all_dim = atof(this->weight.dicts["rope_scaling.mscale_all_dim"].c_str());
        } else {
            rope_scaling_mscale_all_dim = 0.707;
        }

        routed_scaling_factor = atof(this->weight.dicts["routed_scaling_factor"].c_str());
        max_position_embeddings = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        rope_theta = atoi(this->weight.dicts["rope_theta"].c_str());
        q_lora_rank = atoi(this->weight.dicts["q_lora_rank"].c_str());
        qk_rope_head_dim = atoi(this->weight.dicts["qk_rope_head_dim"].c_str());
        kv_lora_rank = atoi(this->weight.dicts["kv_lora_rank"].c_str());
        v_head_dim = atoi(this->weight.dicts["v_head_dim"].c_str());
        qk_nope_head_dim = atoi(this->weight.dicts["qk_nope_head_dim"].c_str());
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        num_experts = atoi(this->weight.dicts["n_routed_experts"].c_str());
        num_experts_per_tok = atoi(this->weight.dicts["num_experts_per_tok"].c_str());
        norm_topk_prob = (this->weight.dicts["norm_topk_prob"] == "true");

        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        head_dim = embed_dim / num_attention_heads;
        rotary_dim = qk_rope_head_dim;
        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.type") != this->weight.dicts.end()) {
            std::string type = this->weight.dicts["rope_scaling.type"];
            if (type == "linear")
               rope_type = RoPEType::LINEAR_SCALE;
            else if (type == "dynamic")
               rope_type = RoPEType::DYMAMIC_NTK;
        }
        if (this->weight.dicts.find("rope_theta") != this->weight.dicts.end()) {
            rope_base = atof(this->weight.dicts["rope_theta"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }
        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(rope_base, rope_factor);
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));
    }

    float yarn_find_correction_dim(int num_rotations, int dim, float base, int max_position_embeddings) {
        return (dim * log(max_position_embeddings / (num_rotations * 2 * M_PI))) / (2 * log(base));
    }
    
    void yarn_find_correction_range(int low_rot, int high_rot, int dim, float base, int max_position_embeddings, int &low, int &high) {
        low = (int)(floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)) + 1e-5);
        high = (int)(ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)) + 1e-5);
        low = std::max(low, 0);
        high = std::min(high, dim - 1);
    }

    float yarn_get_mscale(float scale, float mscale) {
        if (scale <= 1) {
            return 1.0;
        }
        return 0.1 * mscale * log(scale) + 1.0;
    }

    std::vector <float> yarn_linear_ramp_mask(float min, float max, int dim) {
        max = std::max(min + 0.001f, max);
        std::vector <float> ret;
        for (int i = 0; i < dim; i++) {
            float x = (i - min) / (max - min);
            x = std::max(x, 0.0f);
            x = std::min(x, 1.0f);
            ret.push_back(x);
        }
        return ret;
    }

    std::pair<std::vector<float>, std::vector<float>> DeepSeekV2Model::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
        int dim = rotary_dim;
        std::vector <float> freqExtra, freqInter;
        for (int i = 0; i < dim; i += 2) {
            freqExtra.push_back(1.0 / pow(base, (float)i / rotary_dim));
            freqInter.push_back(1.0 / (rope_factor * pow(base, (float)i / rotary_dim)));
        }

        int low, high;
        yarn_find_correction_range (
            rope_scaling_beta_fast,
            rope_scaling_beta_slow,
            dim,
            base,
            rope_scaling_original_max_position_embeddings,
            low, high
        );

        std::vector <float> invFreqMask = yarn_linear_ramp_mask(low, high, dim / 2);
        for (int i = 0; i < invFreqMask.size(); i++) {
            invFreqMask[i] = 1.0 - invFreqMask[i];
        }
        
        std::vector <float> invFreq;
        for (int i = 0; i < freqInter.size(); i++) {
            invFreq.push_back(freqInter[i] * (1.0 - invFreqMask[i]) + freqExtra[i] * invFreqMask[i]);
        }

        float _mscale = yarn_get_mscale(rope_factor, rope_scaling_mscale) / yarn_get_mscale(rope_factor, rope_scaling_mscale_all_dim);

        int positions = std::max(max_positions, seqLen);
        sin.resize(positions);
        cos.resize(positions);

        for (int i = 0; i < positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size() * 2; j++) {
                sin[i][j] = ::sin((float)i * invFreq[j % invFreq.size()]) * _mscale;
                cos[i][j] = ::cos((float)i * invFreq[j % invFreq.size()]) * _mscale;
            }
        }

        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
        }
        for (int i = 0; i < cos.size(); i++) {
            fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    int DeepSeekV2Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> DeepSeekV2Model::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        if (!mergeSwiglu) {
            bool canMerge = true;
            for (int i = 0; i < block_cnt; i++) {
                for (int j = -1; j < this->num_experts; j++) {
                    std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gate_proj.weight";
                    std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".up_proj.weight";
                    std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                    if (j == -1) {
                        w1WeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.gate_proj.weight";
                        w3WeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.up_proj.weight";
                        swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight";
                    }

                    if (weight.weight.find(w1WeightName) == weight.weight.end()) {
                        continue;
                    }

                    Data &w1 = weight.weight[w1WeightName], &w3 = weight.weight[w3WeightName];
                    if ((w1.dataType == DataType::INT4_GROUP && w1.dims[1] % w1.groupCnt != 0) || 
                        (w3.dataType == DataType::INT4_GROUP && w3.dims[1] % w3.groupCnt != 0)) {
                        canMerge = false;
                        break;
                    }

                    weight.weight[swigluWeightName] = Data(w1.dataType, {w1.dims[0] + w3.dims[0], w1.dims[1]});
                    Data &swiglu = weight.weight[swigluWeightName];
                    swiglu.name = swigluWeightName;
                    swiglu.Allocate();
                    memcpy(swiglu.cpuData, w1.cpuData, w1.GetBytes());
                    memcpy(swiglu.cpuData + w1.GetBytes(), w3.cpuData, w3.GetBytes());
                        
                    swiglu.perChannelAxis = w1.perChannelAxis;
                    swiglu.group = w1.group;
                    swiglu.groupCnt = w1.groupCnt;
                    swiglu.perChannelsConfigs = AppendVector(w1.perChannelsConfigs, w3.perChannelsConfigs);
                    swiglu.zeros = AppendVector(w1.zeros, w3.zeros);
                    swiglu.scales = AppendVector(w1.scales, w3.scales);
                    swiglu.mins = AppendVector(w1.mins, w3.mins);

                    weight.weight.erase(w1WeightName);
                    weight.weight.erase(w3WeightName);
                }

                this->mergeSwiglu = canMerge;
            }           
        }

        Data alibiData;
        if (this->weight.dicts["use_alibi"] == "1") {
            std::vector<float> alibi = GetInterleave(num_attention_heads);
            alibiData.CopyFrom(Data(DataType::FLOAT32, {(int) alibi.size()}, alibi));
        }

        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data qa, q, q_nope, q_pe, compressed_kv_ori, compressed_kv, k_pe, k_pe_repeat, kv_ln, kv, k_nope, k, v, qkv;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3, routerLogits, gate, attenPart, moePart, moeFinal, sharedGate;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        int seqlen = hiddenStates.dims[1];

        if (weights.size() == 0) {
            weights.resize(block_cnt);
            biass.resize(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight"]);
                weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.down_proj.weight"]);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                for (int j = 0; j < this->num_experts; j++) {
                    weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight"]);
                    weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight"]);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                }
            }
        }
                    
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);

            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string qaWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_a_proj.weight";
            std::string qaBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_a_proj.bias";
            std::string qRmsNormName = "model.layers." + std::to_string(i) + ".self_attn.q_a_layernorm.weight";
            std::string qbWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_b_proj.weight";
            std::string qbBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_b_proj.bias";
            std::string compressedKvWeightName = "model.layers." + std::to_string(i) + ".self_attn.kv_a_proj_with_mqa.weight";
            std::string compressedKvBiasName = "model.layers." + std::to_string(i) + ".self_attn.kv_a_proj_with_mqa.bias";

            std::string kvRmsNormName = "model.layers." + std::to_string(i) + ".self_attn.kv_a_layernorm.weight";
            std::string kvWeightName = "model.layers." + std::to_string(i) + ".self_attn.kv_b_proj.weight";
            std::string kvBiasName = "model.layers." + std::to_string(i) + ".self_attn.kv_b_proj.bias";

            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (this->weight.weight.find(qaWeightName) != this->weight.weight.end()) { 
                Linear(attenInput, this->weight[qaWeightName], this->weight[qaBiasName], qa);
                RMSNorm(qa, this->weight[qRmsNormName], this->rms_norm_eps, qa);
                Linear(qa, this->weight[qbWeightName], this->weight[qbBiasName], q);
            } else {
                Linear(attenInput, this->weight[qWeightName], this->weight[qBiasName], q);
            }
            
            q.Reshape({bsz, seqlen, -1, q_head_dim});
            PermuteSelf(q, {0, 2, 1, 3});
            Split(q, -1, 0, qk_nope_head_dim, q_nope);
            Split(q, -1, qk_nope_head_dim, q_head_dim, q_pe);
            Linear(attenInput, this->weight[compressedKvWeightName], this->weight[compressedKvBiasName], compressed_kv_ori);
            Split(compressed_kv_ori, -1, 0, kv_lora_rank, compressed_kv);
            Split(compressed_kv_ori, -1, kv_lora_rank, kv_lora_rank + qk_rope_head_dim, k_pe);
            k_pe.Reshape({bsz, seqlen, 1, qk_rope_head_dim});
            RMSNorm(compressed_kv, this->weight[kvRmsNormName], this->rms_norm_eps, kv_ln);
            Linear(kv_ln, this->weight[kvWeightName], this->weight[kvBiasName], kv);
            kv.Reshape({bsz, seqlen, num_attention_heads, qk_nope_head_dim + v_head_dim});
            PermuteSelf(kv, {0, 2, 1, 3});
            Split(kv, -1, 0, qk_nope_head_dim, k_nope);
            Split(kv, -1, qk_nope_head_dim, qk_nope_head_dim + v_head_dim, v);

            PermuteSelf(q_pe, {0, 2, 1, 3});
            q_pe.Reshape({q_pe.dims[0], q_pe.dims[1], q_pe.dims[2], -1, 2});
            PermuteSelf(q_pe, {0, 1, 2, 4, 3});
            q_pe.Reshape({q_pe.dims[0], q_pe.dims[1], q_pe.dims[2], -1});
            fastllm::LlamaRotatePosition2D(q_pe, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            PermuteSelf(q_pe, {0, 2, 1, 3});

            k_pe.Reshape({bsz, seqlen, 1, qk_rope_head_dim / 2, 2});
            PermuteSelf(k_pe, {0, 1, 2, 4, 3});
            k_pe.Reshape({bsz, seqlen, 1, qk_rope_head_dim});
            fastllm::LlamaRotatePosition2D(k_pe, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            PermuteSelf(k_pe, {0, 2, 1, 3});

            Cat(q_nope, q_pe, -1, q);

            Repeat(k_pe, 1, k_nope.dims[1], k_pe_repeat);
            Cat(k_nope, k_pe_repeat, -1, k);

            PermuteSelf(q, {1, 0, 2, 3});
            PermuteSelf(k, {1, 0, 2, 3});
            PermuteSelf(v, {1, 0, 2, 3});
            q.Reshape({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});
            k.Reshape({k.dims[0], k.dims[1] * k.dims[2], k.dims[3]});
            v.Reshape({v.dims[0], v.dims[1] * v.dims[2], v.dims[3]});

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(DataDevice::CUDA);
                pastValue.ToDevice(DataDevice::CUDA);
            }

            int unitLen = 128;
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
            Attention(q, pastKey, pastValue, attentionMask, attenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(q_head_dim), 1);
            PermuteSelf(attenOutput, {1, 0, 2});
            attenOutput.Reshape({seqlen, bsz, -1});
            PermuteSelf(attenOutput, {1, 0, 2});

            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(attenOutput, weight[oWeightName], oBias, attenLastOutput);

            AddTo(hiddenStates, attenLastOutput);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);            

            // 2. moe mlp
            if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight") != weight.weight.end()) {
                if (CanRunLinearEx(LinearExType::ExSilu)) {
                    LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1, LinearExType::ExSilu);
                } else {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
                    Silu(w1, w1);
                }
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
                MulTo(w1, w3);
                Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
                AddTo(hiddenStates, w2);
            } else {
                // 这里是moe mlp
                std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate.weight";
                int batch = attenInput.dims[0], len = attenInput.dims[1];
                attenInput.Reshape({batch * len, attenInput.dims[2]});
                Linear(attenInput, weight[gateWeightName], Data(), routerLogits);
                Softmax(routerLogits, routerLogits, -1);

                if (this->mergeSwiglu && CanRunMergeMOE()) {
                    MergeMOE (
                        attenInput, routerLogits,
                        weights[i], biass[i],
                        this->routed_scaling_factor, 1.0f,
                        this->num_experts_per_tok,
                        moeFinal
                    );
                } else {
                    TopK(routerLogits, gate, this->num_experts_per_tok);
                    gate.ToDevice(DataDevice::CPU);
                    float *gateData = (float*)gate.cpuData;
                    /// TODO: 这里是greedy topk， 需要实现group limited topk

                    moeFinal = Data();
                    moeFinal.Resize({0, attenInput.dims[1]});
                    moeFinal.Expansion(attenInput.dims);

                    for (int b = 0; b < batch * len; b++) {
                        Data *currentData = &attenInput;
                        if (batch * len != 1) {
                            Split(attenInput, 0, b, b + 1, attenPart);
                            currentData = &attenPart;
                        }
                        moePart.Resize(currentData->dims);
                        moePart.Allocate(0.0f);

                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            int idx = (int)(gateData[(b * this->num_experts_per_tok + j) * 2] + 1e-1);
                            float value = gateData[(b * this->num_experts_per_tok + j) * 2 + 1];
                            value *= routed_scaling_factor;
                            if (this->mergeSwiglu) {
                                if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                                    LinearEx(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight"], Data(), w1, LinearExType::ExSwiglu);
                                } else {
                                    Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight"], Data(), w3);
                                    Swiglu(w3, w1);
                                }
                            } else {
                                if (CanRunLinearEx(LinearExType::ExSilu)) {
                                    LinearEx(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gate_proj.weight"], Data(), w1, LinearExType::ExSilu);
                                } else {
                                    Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gate_proj.weight"], Data(), w1);
                                    Silu(w1, w1);
                                }
                                Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".up_proj.weight"], Data(), w3);
                                MulTo(w1, w3);
                            }
                            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".down_proj.weight"], Data(), w2);
                            AddTo(moePart, w2, value);
                        }

                        if (this->mergeSwiglu) {
                            if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                                LinearEx(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight"], Data(), w1, LinearExType::ExSwiglu);
                            } else {
                                Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight"], Data(), w3);
                                Swiglu(w3, w1);
                            }
                        } else {
                            if (CanRunLinearEx(LinearExType::ExSilu)) {
                                LinearEx(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gate_proj.weight"], Data(), w1, LinearExType::ExSilu);
                            } else {
                                Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gate_proj.weight"], Data(), w1);
                                Silu(w1, w1);
                            }
                            Linear(*currentData, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.up_proj.weight"], Data(), w3);
                            MulTo(w1, w3);
                        }
                        Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.down_proj.weight"], Data(), w2);
                        AddTo(moePart, w2);

                        CatDirect(moeFinal, moePart, 0);
                    }
                }

                moeFinal.Reshape(hiddenStates.dims);
                AddTo(hiddenStates, moeFinal);
            }
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
            RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);

            if (generationConfig.output_logits && retLogits != nullptr) {
                int size = logits.dims.back();
                logits.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    (*retLogits)[b]->resize(size);
                    memcpy((float*)(*retLogits)[b]->data(), 
                        ((float*)logits.cpuData) + ((b + 1) * logits.dims[1] - 1) * size, 
                        size * logits.unitSize);
                }
            }

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
/*
        if (sinDataPtr != &sinData)
            delete sinDataPtr;
        if (cosDataPtr != &cosData)
            delete cosDataPtr;
*/
        return lastRet;
    }

    std::vector <int> DeepSeekV2Model::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        Data alibiData;
        if (this->weight.dicts["use_alibi"] == "1") {
            std::vector<float> alibi = GetInterleave(num_attention_heads);
            alibiData.CopyFrom(Data(DataType::FLOAT32, {(int) alibi.size()}, alibi));
        }

        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, curAttenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        std::vector <Data*> sinDataPtrList(batch, &sinData);
        std::vector <Data*> cosDataPtrList(batch, &cosData);

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                int qdim = per * (num_attention_heads / num_key_value_heads);
                Split(qkv, -1, 0, qdim, q);
                Split(qkv, -1, qdim, qdim + per, k);
                Split(qkv, -1, qdim + per, qdim + per * 2, v);
            } else {
                if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                    int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                    int qdim = per * (num_attention_heads / num_key_value_heads);

                    Split(qkv, -1, 0, qdim, q);
                    Split(qkv, -1, qdim, qdim + per, k);
                    Split(qkv, -1, qdim + per, qdim + per * 2, v);
                } else {
                    Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                    Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                    Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                    Linear(attenInput, weight[qWeightName], qBias, q);
                    Linear(attenInput, weight[kWeightName], kBias, k);
                    Linear(attenInput, weight[vWeightName], vBias, v);
                }
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

                std::vector<int> qkvSize = {bsz, seqLens[b], -1, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(DataDevice::CUDA);
                    pastValue.ToDevice(DataDevice::CUDA);
                }
                int targetSeqLength = (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqLens[b] : seqLens[b];
                if (i == 0 && targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                    float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                    float newbase = rope_base * scale;
                    std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                    sinDataPtrList[b] = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                    cosDataPtrList[b] = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
                }

                if (alibiData.dims.size() == 0) {
                    fastllm::LlamaRotatePosition2D(q, *positionIds[b], *sinDataPtrList[b], *cosDataPtrList[b], rotary_dim);
                    fastllm::LlamaRotatePosition2D(k, *positionIds[b], *sinDataPtrList[b], *cosDataPtrList[b], rotary_dim);
                }

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});

                qkvSize = {-1, seqLens[b], head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);
                
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
                if (alibiData.dims.size() == 0) {
                    Attention(q, pastKey, pastValue, attentionMask[b] == nullptr ? Data() : *attentionMask[b], curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                } else {
                    MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim), q.dims[0] / pastKey.dims[0]);
                    attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
                    if (alibiData.dims.size() != 0) {
                        AlibiMask(attenWeights, alibiData, -10000);
                    } else if (attentionMask[b] != nullptr) {
                        AttentionMask(attenWeights, *attentionMask[b], -10000);
                    }

                    Softmax(attenWeights, attenWeights, -1);
                    MatMul(attenWeights, pastValue, curAttenOutput, 1.f, attenWeights.dims[1] / pastValue.dims[0]);
                    curAttenOutput.Reshape({curAttenOutput.dims[1], curAttenOutput.dims[2], curAttenOutput.dims[3]});
                }

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

            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(attenOutput, weight[oWeightName], oBias, attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);
            if (true) {
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                    LinearEx(attenInput, weight[swigluWeightName], Data(), w1, LinearExType::ExSwiglu);
                } else {
                    Linear(attenInput, weight[swigluWeightName], Data(), w3);
                    Swiglu(w3, w1);
                }
            } else {
                if (CanRunLinearEx(LinearExType::ExSilu)) {
                    LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1, LinearExType::ExSilu);
                } else {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
                    Silu(w1, w1);
                }
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
                MulTo(w1, w3);
            }

            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
            AddTo(hiddenStates, w2);
        }

        Data logits, curLogit;
        RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
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
        for (Data* sinPtr : sinDataPtrList)
            if (sinPtr != &sinData)
                delete sinPtr;
        for (Data* cosPtr : cosDataPtrList)
            if (cosPtr != &cosData)
                delete cosPtr;
        return lastRet;
    }

    void DeepSeekV2Model::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            std::vector <int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
                seqLens[i] = (int)inputTokens[i].size();
            }

            std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = tokens.size(), base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                }
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

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, vpids));
        } else {
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen, maxLen);
                pids[i] = promptLen + index - 1;
            }
            maxLen += index;
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int curLen = params[i].find("promptLen")->second + index;
                for (int j = 0; j < maxLen - curLen; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
        }
    }

    std::string DeepSeekV2Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string DeepSeekV2Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void DeepSeekV2Model::WarmUp() {
        printf("Warmup...\n");
        int oldTopk = this->num_experts_per_tok;
        this->num_experts_per_tok = this->num_experts;

        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        printf("finish.\n");
        this->num_experts_per_tok = oldTopk;
    }
}
