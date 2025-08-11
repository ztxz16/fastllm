//
// Created by huangyuyang on 5/11/24.
//

#include "deepseekv2.h"

#include "executor.h"

#include "utils.h"

#include <sstream>

#include <random>

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
        this->model_struct = "deepseek_v2";

        // 默认使用alpaca的提示词和instruction
        this->pre_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";
        this->user_role = "### Instruction:\n";
        this->bot_role = "\n\n### Response:";
        this->history_sep = "</s>";

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight", "model.layers.*.mlp*down_proj.weight", "model.layers.*.mlp*up_proj.weight",
            "model.layers.*.mlp*gate_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.q_a_proj.weight",
            "model.layers.*.self_attn.q_b_proj.weight",
            "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.*.self_attn.kv_b_proj.weight",
            "model.layers.*.self_attn.o_proj.weight", "model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight", "model.layers.*.self_attn.mergeqkv.weight", "model.layers.*.self_attn.W_pack.weight",
            "model.layers.*.mlp.gate.weight"
        };
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

        n_shared_experts = atoi(this->weight.dicts["n_shared_experts"].c_str());
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

        for (int i = 0; i < block_cnt; i++) {
            for (int j = -1; j < this->num_experts; j++) {
                std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gate_proj.weight";
                std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".up_proj.weight";
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                if (j == -1) {
                    w1WeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.gate_proj.weight";
                    w3WeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.up_proj.weight";
                    swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight";
                    downWeightName = "model.layers." + std::to_string(i) + ".mlp.shared_experts.down_proj.weight";
                }
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
                );

                if (j != -1 || !GetCudaSharedExpert()) {
                    this->specialWeights[swigluWeightName] = "linearSwiglu";
                    this->specialWeights[downWeightName] = "linearColumn";
                }
                
                this->moeLinears.insert(w1WeightName);
                this->moeLinears.insert(w3WeightName);
                this->moeLinears.insert(downWeightName);
            }

            this->cantQuantLinears.insert("model.layers." + std::to_string(i) + ".self_attn.kv_b_proj.weight");
        }
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

    std::vector <int> GumbelMaxTrick(Data &logits, int batch) {
        std::vector <int> ret;
        Mul(logits, 1 / 0.6, logits);
        Softmax(logits, logits, -1);
        ToDataType(logits, DataType::FLOAT32);
        logits.ToDevice(DataDevice::CPU);
        int vocabSize = logits.dims.back();
        static std::random_device rd;  // 用于获取随机种子
        static std::mt19937 gen(rd()); // 使用Mersenne Twister引擎
        // 设置指数分布，lambda = 1
        static std::exponential_distribution<> exp_dist(1.0);

        for (int b = 0; b < batch; b++) {
            float *base = ((float*)logits.cpuData) + b * vocabSize;
            int selId = -1;
            float maxValue = -1e10;
            for (int i = 0; i < vocabSize; i++) {
                float nowValue = base[i] / exp_dist(gen);
                if (nowValue > maxValue) {
                    maxValue = nowValue;
                    selId = i;
                }
            }
            ret.push_back(selId);
        }

        return ret;
    }

    std::vector <int> DeepSeekV2Model::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        Executor &excutor = *((Executor*)GetExecutor());

        std::string scoring_func = "softmax";
        if (this->weight.dicts.find("scoring_func") != this->weight.dicts.end()) {
            scoring_func = this->weight.dicts["scoring_func"];
        }

        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data qa, q, q_nope, q_pe, compressed_kv_ori, compressed_kv, k_pe, k_pe_repeat, kv_ln, kv, k_nope, k, v, qkv;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3, routerLogits, gate, attenPart, moePart, moeFinal, sharedGate;
        Data curInput, curOutput;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;
        Data ww1, ww2, ww3, moeFinal2;

        Data resultTemp, qpeTemp, qnopeTemp, kTemp, vTemp;
//inputIds.Print();
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        // ToDataType(hiddenStates, this->dataType);

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

        float softmax_scale = 1.0 / sqrt(q_head_dim);
        float mscale = 0.1 * rope_scaling_mscale * log(rope_factor) + 1.0;
        softmax_scale = softmax_scale * mscale * mscale;
        
        Data attenInputTemp, x, result, score0, score1;
        bool cudaSe = GetCudaSharedExpert();
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            UpdateRotaryPtr(&sinDataPtr, &cosDataPtr, excutor.firstDevice);

            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            
            ToDataType(attenInput, attenInputTemp, this->dataType);

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
                Linear(attenInputTemp, this->weight[qaWeightName], this->weight[qaBiasName], qa);
                RMSNorm(qa, this->weight[qRmsNormName], this->rms_norm_eps, qa);
                Linear(qa, this->weight[qbWeightName], this->weight[qbBiasName], q);
            } else {
                Linear(attenInputTemp, this->weight[qWeightName], this->weight[qBiasName], q);
            }
            
            q.Reshape({bsz, seqlen, -1, q_head_dim});
            PermuteSelf(q, {0, 2, 1, 3});
            Split(q, -1, 0, qk_nope_head_dim, q_nope);
            Split(q, -1, qk_nope_head_dim, q_head_dim, q_pe);
            Linear(attenInputTemp, this->weight[compressedKvWeightName], this->weight[compressedKvBiasName], compressed_kv_ori);
            Split(compressed_kv_ori, -1, 0, kv_lora_rank, compressed_kv);
            Split(compressed_kv_ori, -1, kv_lora_rank, kv_lora_rank + qk_rope_head_dim, k_pe);
            RMSNorm(compressed_kv, this->weight[kvRmsNormName], this->rms_norm_eps, kv_ln);

            PermuteSelf(q_pe, {0, 2, 1, 3});
            PermuteSelf(q_pe, {1, 0, 2, 3});
            fastllm::NearlyRotatePosition2D(q_pe, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            PermuteSelf(q_pe, {1, 0, 2, 3});
            PermuteSelf(q_pe, {0, 2, 1, 3});

            k_pe.Reshape({bsz, seqlen, 1, qk_rope_head_dim});
            PermuteSelf(k_pe, {1, 0, 2, 3});
            fastllm::NearlyRotatePosition2D(k_pe, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            PermuteSelf(k_pe, {1, 0, 2, 3});
            PermuteSelf(k_pe, {0, 2, 1, 3});

            if (excutor.GetFirstDeviceType() == "cuda" || excutor.GetFirstDeviceType() == "multicuda") {
                auto &k = k_pe, &v = kv_ln;
                k_pe.Reshape({k_pe.dims[0], k_pe.dims[2], k_pe.dims[3]});
                Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(k.dataDevice);
                    pastValue.ToDevice(v.dataDevice);
                }
                int unitLen = 128;
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

                // ToDataType(k, kTemp, this->dataType);
                // ToDataType(v, vTemp, this->dataType);
                CatDirect(pastKey, k, 1);
                CatDirect(pastValue, v, 1);
                
// printf("matmul catdirect spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // absorb
                PermuteSelf(q_pe, {0, 2, 1, 3});
                PermuteSelf(q_nope, {0, 2, 1, 3});
// printf("matmul permuteself spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                int b = q_nope.dims[0], s = q_nope.dims[1], h = q_nope.dims[2], d = q_nope.dims[3];
                PermuteSelf(q_nope, {2, 0, 1, 3});
                q_nope.Reshape({q_nope.dims[0], -1, q_nope.dims[3]});

                std::string kv0Name = kvWeightName + "__0", kv1Name = kvWeightName + "__1";
                if (this->weight.weight.find(kvWeightName) != this->weight.weight.end()) {
                    this->weight[kvWeightName].Reshape({num_attention_heads, -1, kv_lora_rank});
                    Split(this->weight[kvWeightName], 1, 0, qk_nope_head_dim, this->weight[kv0Name]);
                    Split(this->weight[kvWeightName], 1, qk_nope_head_dim, qk_nope_head_dim + v_head_dim, this->weight[kv1Name]);
                    this->weight.weight.erase(kvWeightName);
                }

                Data &kv0 = this->weight[kv0Name];
                Data &kv1 = this->weight[kv1Name];

                if (kv0.dims != kv1.dims) {
                    PermuteSelf(kv0, {0, 2, 1});
                }
                
                // ToDataType(q_nope, qnopeTemp, this->dataType);
                MatMul(q_nope, kv0, result);
// printf("matmul0 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                int c = result.dims.back(), t = pastValue.dims[1];
                if (attentionMask.dims.size() == 0 && b == 1 && s > 1 && t > 1) {
                    std::vector <float> vmasks = std::vector <float> (s * t, 0.0f);
                    for (int i = 0; i < s; i++) {
                        for (int j = t - s + i + 1; j < t; j++) {
                            vmasks[i * t + j] = 1.0;
                        }
                    }
                    ((Data*)&attentionMask)->CopyFrom(Data(DataType::FLOAT32, { s, t }, vmasks));
                }

                PermuteSelf(result, {1, 0, 2});
                if (true) {
                    ToDataType(attentionMask, this->dataType);
                    // ToDataType(q_pe, qpeTemp, this->dataType);
                    MergeMLA(result, q_pe, pastKey, pastValue, *((Data*)&attentionMask), x, softmax_scale);
                } else {
                    result.Reshape({b, s * h, c});
                    MatMulTransB(result, pastValue, score0);
                    score0.Reshape({b, s, h, t});

                    q_pe.Reshape({q_pe.dims[0], -1, q_pe.dims[3]});
                    MatMulTransB(q_pe, pastKey, score1);
                    score1.Reshape({b, s, h, t});

                    AddTo(score1, score0);
                    Mul(score1, softmax_scale, score0);

                    if (attentionMask.dims.size() > 0) {
                        score0.Reshape({b * s, h, t});
                        ToDataType(attentionMask, this->dataType);
                        AttentionMask(score0, attentionMask, -10000);
                    }

                    Softmax(score0, score0, -1);
                    score0.Reshape({b, s * h, t});
                    MatMul(score0, pastValue, x);
                }
                x.Reshape({b, s, h, c});
                PermuteSelf(x, {2, 0, 1, 3});
                x.Reshape({h, b * s, c});
                MatMulTransB(x, kv1, attenOutput);
                attenOutput.Reshape({h, b, s, -1});
                PermuteSelf(attenOutput, {1, 2, 0, 3});
                attenOutput.Reshape({seqlen, bsz, -1});
                PermuteSelf(attenOutput, {1, 0, 2});

                ToDataType(attenOutput, DataType::FLOAT32);
            } else {
                Linear(kv_ln, this->weight[kvWeightName], this->weight[kvBiasName], kv);
                kv.Reshape({bsz, seqlen, num_attention_heads, qk_nope_head_dim + v_head_dim});
                PermuteSelf(kv, {0, 2, 1, 3});
                Split(kv, -1, 0, qk_nope_head_dim, k_nope);
                Split(kv, -1, qk_nope_head_dim, qk_nope_head_dim + v_head_dim, v);
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
                    pastKey.ToDevice(k.dataDevice);
                    pastValue.ToDevice(v.dataDevice);
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
                Attention(q, pastKey, pastValue, attentionMask, attenOutput, q.dims[0] / pastKey.dims[0], softmax_scale, 1);
                PermuteSelf(attenOutput, {1, 0, 2});
                attenOutput.Reshape({seqlen, bsz, -1});
                PermuteSelf(attenOutput, {1, 0, 2});

                ToDataType(attenOutput, DataType::FLOAT32);
            }

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
                std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";

                int batch = attenInput.dims[0], len = attenInput.dims[1];
                attenInput.Reshape({batch * len, attenInput.dims[2]});

                if (cudaSe) {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight"], Data(), ww3);
                    Swiglu(ww3, ww1);
                    Linear(ww1, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.down_proj.weight"], Data(), moeFinal2);
                    weights[i][0] = weights[i][1] = nullptr;
                }

                Linear(attenInput, weight[gateWeightName], Data(), routerLogits);

                bool needNorm = false;
                if (scoring_func == "sigmoid") {
                    Sigmoid(routerLogits, routerLogits);
                    needNorm = true;
                } else {
                    Softmax(routerLogits, routerLogits, -1);
                }

                ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
                if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.experts.0.gateup_proj.weight") != weight.weight.end() 
                    && CanRunMergeMOE(attenInput, biass[i])) {
                    MergeMOE (
                        attenInput, routerLogits, weight[gateBiasName],
                        weights[i], biass[i],
                        w1, w2, w3, curInput, curOutput, 
                        this->routed_scaling_factor, 1.0f,
                        this->num_experts_per_tok, needNorm,
                        moeFinal
                    );
                } else {
                    Data &bias = weight[gateBiasName];                  
                    ToDataType(routerLogits, DataType::FLOAT32);
                    routerLogits.ToDevice(DataDevice::CPU);
                    float *cpuRouterLogits = (float*)routerLogits.cpuData;
                    int m = routerLogits.dims.back();

                    moeFinal = Data();
                    moeFinal.Resize({0, attenInput.dims[1]});
                    moeFinal.Expansion(attenInput.dims);

                    for (int b = 0; b < batch * len; b++) {
                        float *cur = cpuRouterLogits + b * m;
                        std::vector <std::pair <float, int> > v; // (value, idx)
                        for (int i = 0; i < m; i++) {
                            v.push_back(std::make_pair(-cur[i], i));
                        }
                        if (bias.dims.size() > 0) {
                            ToDataType(bias, DataType::FLOAT32);
                            bias.ToDevice(DataDevice::CPU);
                            float *cpuBias = (float*)bias.cpuData;
                            for (int i = 0; i < m; i++) {
                                v[i].first -= cpuBias[i];
                            }
                        }

                        sort(v.begin(), v.end());
                        Data *currentData = &attenInput;
                        if (batch * len != 1) {
                            Split(attenInput, 0, b, b + 1, attenPart);
                            currentData = &attenPart;
                        }
                        moePart.Resize(currentData->dims);
                        moePart.Allocate(0.0f);

                        float sum = 0.0;
                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            float value = cur[v[j].second];
                            sum += value;
                        }
                        if (!needNorm) {
                            sum = 1.0;
                        }

                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            int idx = v[j].second;
                            float value = cur[idx];

                            value /= sum;
                            value *= routed_scaling_factor;
                            if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight") != weight.weight.end()) {
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

                        if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight") != weight.weight.end()) {
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
                    moeFinal.expansionDims.clear();
                }

                moeFinal.Reshape(hiddenStates.dims);

                Data tempMoeFinal;
                tempMoeFinal.CopyFrom(moeFinal);
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                AddTo(hiddenStates, tempMoeFinal);

                if (cudaSe) {
                    moeFinal2.Reshape(hiddenStates.dims);
                    AddTo(hiddenStates, moeFinal2);
                }
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
            ToDataType(logits, DataType::FLOAT32);
            ResetLogitsOfEOS(batch, &logits, pastKeyValues, generationConfig);
            if (this->weight.dicts["model_type"] != "deepseek_v2") {
                if (((GenerationConfig*)&generationConfig)->top_k <= 1) { 
                    ((GenerationConfig*)&generationConfig)->top_k = 10;
                }

                ((GenerationConfig*)&generationConfig)->top_p = 0.95;
                if (fabs(generationConfig.temperature - 1.0f) < 1e-6) {
                    ((GenerationConfig*)&generationConfig)->temperature = 0.6;
                } 
            }
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
            } else if (generationConfig.top_k <= 50 && fabs(generationConfig.repeat_penalty - 1.0f) < 1e-5) {
                /*int maxTokenSetSize = 0;
                for (int b = 0; b < batch; b++) {
                    maxTokenSetSize = std::max(maxTokenSetSize, (int)lastTokens.units[b].tokenSet.size());
                }
                std::vector <float> penaltyData = std::vector <float> (batch * maxTokenSetSize, -100.0f);
                std::vector <float> penaltyScaleData = std::vector <float> (batch, 1.0f);
                for (int b = 0; b < batch; b++) {
                    int curId = 0;
                    for (int i : lastTokens.units[b].tokenSet) {
                        penaltyData[b * maxTokenSetSize + curId] = i;
                        curId++;
                    }
                    penaltyScaleData[b] = generationConfigs[b].repeat_penalty;
                }
                Data penalty, penaltyScale;
                penalty.CopyFrom(Data(DataType::FLOAT32, {batch, maxTokenSetSize}, penaltyData));
                penaltyScale.CopyFrom(Data(DataType::FLOAT32, {batch}, penaltyScaleData));
                RepeatPenalty(logits, penalty, penaltyScale);*/
                Data topk;
                TopK(logits, topk, generationConfig.top_k);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    lastRet.push_back(LLMSamplingOnly(topk, b, generationConfig));
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

    std::vector <int> DeepSeekV2Model::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        Executor &excutor = *((Executor*)GetExecutor());
        int seqLen = inputIds.dims[1];
        std::string scoring_func = "softmax";
        if (this->weight.dicts.find("scoring_func") != this->weight.dicts.end()) {
            scoring_func = this->weight.dicts["scoring_func"];
        }

        Data hiddenStates;
        Data attenInput;
        Data qa, q, q_nope, q_pe, compressed_kv_ori, compressed_kv, k_pe, k_pe_repeat, kv_ln, kv, k_nope, k, v, qkv;
        Data lastx, allResult, cur_q_pe, result, x;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3, routerLogits, gate, attenPart, moePart, moeFinal, sharedGate;
        Data curInput, curOutput;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;
        Data  curAttenOutput;
        Data ww1, ww2, ww3, moeFinal2;

        std::vector <Data> curContextLayer;
        curContextLayer.resize(batch);
        std::vector <Data> curKs, curVs, curQs;
        curKs.resize(batch);
        curVs.resize(batch);
        curQs.resize(batch);
        std::vector <Data*> pointersK, pointersV, pointersQ;
        pointersK.resize(batch);
        pointersV.resize(batch);
        pointersQ.resize(batch);
        std::vector <Data*> keys, values, qs, attns, masks, contexts;
        keys.resize(batch);
        values.resize(batch);
        qs.resize(batch);
        attns.resize(batch);
        masks.resize(batch);
        contexts.resize(batch);

        Data allPositionIds;
        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;            
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            allPositionIds.CopyFrom(*(Data*)positionIds[0]);
            allPositionIds.Expansion({1, seqLen});
            for (int i = 1; i < batch; i++) {
                CatDirect(allPositionIds, *(Data*)positionIds[i], 1);
            }
        }

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        // ToDataType(hiddenStates, this->dataType);
        int seqlen = hiddenStates.dims[1];
        float softmax_scale = 1.0 / sqrt(q_head_dim);
        float mscale = 0.1 * rope_scaling_mscale * log(rope_factor) + 1.0;
        softmax_scale = softmax_scale * mscale * mscale;

        Data attenInputTemp;
        bool cudaSe = GetCudaSharedExpert();

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            UpdateRotaryPtr(&sinDataPtr, &cosDataPtr, excutor.firstDevice);
            
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            
            ToDataType(attenInput, attenInputTemp, this->dataType);
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
                Linear(attenInputTemp, this->weight[qaWeightName], this->weight[qaBiasName], qa);
                RMSNorm(qa, this->weight[qRmsNormName], this->rms_norm_eps, qa);
                Linear(qa, this->weight[qbWeightName], this->weight[qbBiasName], q);
            } else {
                Linear(attenInputTemp, this->weight[qWeightName], this->weight[qBiasName], q);
            }

            q.Reshape({bsz, seqlen, -1, q_head_dim});
            PermuteSelf(q, {0, 2, 1, 3});
            Split(q, -1, 0, qk_nope_head_dim, q_nope);
            Split(q, -1, qk_nope_head_dim, q_head_dim, q_pe);
            Linear(attenInputTemp, this->weight[compressedKvWeightName], this->weight[compressedKvBiasName], compressed_kv_ori);
            Split(compressed_kv_ori, -1, 0, kv_lora_rank, compressed_kv);
            Split(compressed_kv_ori, -1, kv_lora_rank, kv_lora_rank + qk_rope_head_dim, k_pe);
            RMSNorm(compressed_kv, this->weight[kvRmsNormName], this->rms_norm_eps, kv_ln);

            PermuteSelf(q_pe, {0, 2, 1, 3});
            PermuteSelf(q_pe, {1, 0, 2, 3});
            fastllm::NearlyRotatePosition2D(q_pe, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            PermuteSelf(q_pe, {1, 0, 2, 3});
            PermuteSelf(q_pe, {0, 2, 1, 3});

            k_pe.Reshape({bsz, seqlen, 1, qk_rope_head_dim});
            PermuteSelf(k_pe, {1, 0, 2, 3});
            fastllm::NearlyRotatePosition2D(k_pe, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            PermuteSelf(k_pe, {1, 0, 2, 3});
            PermuteSelf(k_pe, {0, 2, 1, 3});

            Data attenOutput = Data(this->dataType);
            if (excutor.GetFirstDeviceType() == "cuda" || excutor.GetFirstDeviceType() == "multicuda") {
                k_pe.Reshape({k_pe.dims[0], k_pe.dims[2], k_pe.dims[3]});
                std::string kv0Name = kvWeightName + "__0", kv1Name = kvWeightName + "__1";
                Data &kv0 = this->weight[kv0Name];
                Data &kv1 = this->weight[kv1Name];

                PermuteSelf(q_nope, {0, 2, 1, 3});
                int b = q_nope.dims[0], s = q_nope.dims[1], h = q_nope.dims[2], d = q_nope.dims[3];
                PermuteSelf(q_nope, {2, 0, 1, 3});
                q_nope.Reshape({q_nope.dims[0], -1, q_nope.dims[3]});
                MatMul(q_nope, kv0, allResult);
                PermuteSelf(allResult, {1, 0, 2});
                int c = allResult.dims.back(), r = q_pe.dims.back();

                std::vector <int> kdims = {k_pe.dims[0], 1, k_pe.dims[2]};
                std::vector <uint64_t> kstrides = {(uint64_t)k_pe.dims[2], (uint64_t)k_pe.dims[2], 1};
                k.dims = kdims;
                k.strides = kstrides;

                std::vector <int> vdims = {kv_ln.dims[0], 1, kv_ln.dims[2]};
                std::vector <uint64_t> vstrides = {(uint64_t)kv_ln.dims[2], (uint64_t)kv_ln.dims[2], 1};
                v.dims = vdims;
                v.strides = vstrides;

                for (int bid = 0; bid < batch; bid++) {
                    k.FakeFrom(k_pe, bid * kstrides[1] * k_pe.unitSize);
                    v.FakeFrom(kv_ln, bid * vstrides[1] * kv_ln.unitSize);
                    Data &pastKey = *pastKeyValues[bid * block_cnt + i].first, &pastValue = *pastKeyValues[bid * block_cnt + i].second;
                    if (GetKVCacheInCPU()) {
                        pastKey.lockInCPU = true;
                        pastValue.lockInCPU = true;
                    } else {
                        pastKey.ToDevice(k.dataDevice);
                        pastValue.ToDevice(v.dataDevice);
                    }
                    int unitLen = 128;
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
                }

                lastx.ToDevice(allResult.dataDevice);
                lastx.dataType = allResult.dataType;
                lastx.Resize({batch, h, c});
                lastx.Allocate();

                std::vector <int> xdims = {1, h, c};
                std::vector <uint64_t> xstrides = {(uint64_t)h * c, (uint64_t)c, 1};
                x.dims = xdims;
                x.strides = xstrides;

                PermuteSelf(q_pe, {0, 2, 1, 3});
                std::vector <int> qpedims = {1, 1, h, r};
                std::vector <uint64_t> qpestrides = {(uint64_t)h * r, (uint64_t)h * r, (uint64_t)r, 1};

                for (int bid = 0; bid < batch; bid++) {
                    Data &pastKey = *pastKeyValues[bid * block_cnt + i].first, &pastValue = *pastKeyValues[bid * block_cnt + i].second;
                    cur_q_pe.dims = qpedims;
                    cur_q_pe.strides = qpestrides;

                    result.dims = xdims;
                    result.strides = xstrides;

                    cur_q_pe.FakeFrom(q_pe, bid * h * r * q_pe.unitSize);
                    result.FakeFrom(allResult, bid * h * c * allResult.unitSize);
                    x.FakeFrom(lastx, bid * xstrides[0] * lastx.unitSize);
                    MergeMLA(result, cur_q_pe, pastKey, pastValue, Data(), x, softmax_scale);
                }

                lastx.Reshape({b, s, h, -1});
                PermuteSelf(lastx, {2, 0, 1, 3});
                lastx.Reshape({h, b * s, -1});
                MatMulTransB(lastx, kv1, attenOutput);

                attenOutput.Reshape({h, b, s, -1});
                PermuteSelf(attenOutput, {1, 2, 0, 3});
                attenOutput.Reshape({seqlen, bsz, -1});
                PermuteSelf(attenOutput, {1, 0, 2});
                ToDataType(attenOutput, DataType::FLOAT32);
            } else {
                Linear(kv_ln, this->weight[kvWeightName], this->weight[kvBiasName], kv);
                kv.Reshape({bsz, seqlen, num_attention_heads, qk_nope_head_dim + v_head_dim});
                PermuteSelf(kv, {0, 2, 1, 3});
                Split(kv, -1, 0, qk_nope_head_dim, k_nope);
                Split(kv, -1, qk_nope_head_dim, qk_nope_head_dim + v_head_dim, v);
                Cat(q_nope, q_pe, -1, q);

                Repeat(k_pe, 1, k_nope.dims[1], k_pe_repeat);
                Cat(k_nope, k_pe_repeat, -1, k);

                PermuteSelf(q, {1, 0, 2, 3});
                PermuteSelf(k, {1, 0, 2, 3});
                PermuteSelf(v, {1, 0, 2, 3});
                q.Reshape({q.dims[0], q.dims[1] * q.dims[2], q.dims[3]});
                k.Reshape({k.dims[0], k.dims[1] * k.dims[2], k.dims[3]});
                v.Reshape({v.dims[0], v.dims[1] * v.dims[2], v.dims[3]});

                PermuteSelf(q, {1, 0, 2});
                q.Reshape({1, q.dims[0], q.dims[1], q.dims[2]});
                PermuteSelf(k, {1, 0, 2});
                k.Reshape({1, k.dims[0], k.dims[1], k.dims[2]});
                PermuteSelf(v, {1, 0, 2});
                v.Reshape({1, v.dims[0], v.dims[1], v.dims[2]});
                int targetSeqLength = 0;
                for (int b = 0; b < batch; b++) {
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                    if (GetKVCacheInCPU()) {
                        pastKey.lockInCPU = true;
                        pastValue.lockInCPU = true;
                    } else {
                        pastKey.ToDevice(k.dataDevice);
                        pastValue.ToDevice(v.dataDevice);
                    }
                }

                for (int b = 0; b < batch; b++) {
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                    int curLen = seqLens[b];
                    
                    int unitLen = 64;
    #ifdef USE_CUDA
                    unitLen = 128;
    #endif
                    int cacheOuter = k.dims[2], cacheInner = k.dims[3];
                    while ((pastKey.dims.size() == 0 &&
                            (pastKey.expansionDims.size() == 0 || curLen > pastKey.expansionDims[1]))
                        || (pastKey.dims.size() > 0 && pastKey.dims[1] + curLen > pastKey.expansionDims[1])) {
                        std::vector<int> newDims;
                        if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                            newDims = std::vector<int> {cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                        } else {
                            newDims = pastKey.dims;
                            newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                        }
                        pastKey.Expansion(newDims);
                    }

                    cacheOuter = v.dims[2], cacheInner = v.dims[3];
                    while ((pastValue.dims.size() == 0 &&
                            (pastValue.expansionDims.size() == 0 || curLen > pastValue.expansionDims[1]))
                        || (pastValue.dims.size() > 0 && pastValue.dims[1] + curLen > pastValue.expansionDims[1])) {
                        std::vector<int> newDims;
                        if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                            newDims = std::vector<int>{cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                        } else {
                            newDims = pastValue.dims;
                            newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                        }
                        pastValue.Expansion(newDims);
                    }
                }

                int total = 0;
                q.Reshape({-1, q.dims[2], q.dims[3]});
                k.Reshape({-1, k.dims[2], k.dims[3]});
                v.Reshape({-1, v.dims[2], v.dims[3]});

                std::vector <int> qdims = {q.dims[1], 1, q.dims[2]};
                std::vector <uint64_t> qstrides = {(uint64_t)q.dims[2], (uint64_t)q.dims[2], 1};
                std::vector <int> kdims = {k.dims[1], 1, k.dims[2]};
                std::vector <uint64_t> kstrides = {(uint64_t)k.dims[2], (uint64_t)k.dims[2], 1};
                std::vector <int> vdims = {v.dims[1], 1, v.dims[2]};
                std::vector <uint64_t> vstrides = {(uint64_t)v.dims[2], (uint64_t)v.dims[2], 1};
                for (int b = 0; b < batch; b++) {
                    curQs[b].dims = qdims;
                    curQs[b].strides = qstrides;
                    curQs[b].FakeFrom(q, b * q.strides[0] * q.unitSize);
                    curKs[b].dims = kdims;
                    curKs[b].strides = kstrides;
                    curKs[b].FakeFrom(k, b * k.strides[0] * k.unitSize);
                    curVs[b].dims = vdims;
                    curVs[b].strides = vstrides;
                    curVs[b].FakeFrom(v, b * v.strides[0] * v.unitSize);
                }

                total = batch;
                for (int b = 0; b < batch; b++) {
                    keys[b] = (pastKeyValues[b * block_cnt + i].first);
                    values[b] = (pastKeyValues[b * block_cnt + i].second);
                    pointersK[b] = (&curKs[b]);
                    pointersV[b] = (&curVs[b]);
                }
                CatDirectBatch(keys, pointersK, 1);
                CatDirectBatch(values, pointersV, 1);

                int attnOutputDim = weight[oWeightName].dims[1];
                attenOutput.ToDevice(q.dataDevice);
                attenOutput.Resize({1, batch, attnOutputDim});
                attenOutput.Allocate();
                for (int b = 0; b < batch; b++) {
                    qs[b] = (&curQs[b]);
                    keys[b] = (pastKeyValues[b * block_cnt + i].first);
                    values[b] = (pastKeyValues[b * block_cnt + i].second);
                    masks[b] = attentionMask[b];
                    curContextLayer[b].FakeFrom(attenOutput, b * attnOutputDim * attenOutput.unitSize);
                    contexts[b] = (&curContextLayer[b]);
                }
                AttentionBatch(qs, keys, values, masks, contexts, qs[0]->dims[0] / values[0]->dims[0], 1.0 / scale_attn, 1);

                ToDataType(attenOutput, DataType::FLOAT32);
            }

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
                std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";

                int batch = attenInput.dims[0], len = attenInput.dims[1];
                attenInput.Reshape({batch * len, attenInput.dims[2]});

                if (cudaSe) {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight"], Data(), ww3);
                    Swiglu(ww3, ww1);
                    Linear(ww1, weight["model.layers." + std::to_string(i) + ".mlp.shared_experts.down_proj.weight"], Data(), moeFinal2);
                    weights[i][0] = weights[i][1] = nullptr;
                }

                Linear(attenInput, weight[gateWeightName], Data(), routerLogits);

                bool needNorm = false;
                if (scoring_func == "sigmoid") {
                    Sigmoid(routerLogits, routerLogits);
                    needNorm = true;
                } else {
                    Softmax(routerLogits, routerLogits, -1);
                }

                ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
                if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.experts.0.gateup_proj.weight") != weight.weight.end() 
                    && CanRunMergeMOE(attenInput, biass[i])) {
                    MergeMOE (
                        attenInput, routerLogits, weight[gateBiasName],
                        weights[i], biass[i],
                        w1, w2, w3, curInput, curOutput,
                        this->routed_scaling_factor, 1.0f,
                        this->num_experts_per_tok, needNorm,
                        moeFinal
                    );
                } else {
                    Data &bias = weight[gateBiasName];                  
                    ToDataType(routerLogits, DataType::FLOAT32);
                    routerLogits.ToDevice(DataDevice::CPU);
                    float *cpuRouterLogits = (float*)routerLogits.cpuData;
                    int m = routerLogits.dims.back();

                    moeFinal = Data();
                    moeFinal.Resize({0, attenInput.dims[1]});
                    moeFinal.Expansion(attenInput.dims);

                    for (int b = 0; b < batch * len; b++) {
                        float *cur = cpuRouterLogits + b * m;
                        std::vector <std::pair <float, int> > v; // (value, idx)
                        for (int i = 0; i < m; i++) {
                            v.push_back(std::make_pair(-cur[i], i));
                        }
                        if (bias.dims.size() > 0) {
                            ToDataType(bias, DataType::FLOAT32);
                            bias.ToDevice(DataDevice::CPU);
                            float *cpuBias = (float*)bias.cpuData;
                            for (int i = 0; i < m; i++) {
                                v[i].first -= cpuBias[i];
                            }
                        }

                        sort(v.begin(), v.end());
                        Data *currentData = &attenInput;
                        if (batch * len != 1) {
                            Split(attenInput, 0, b, b + 1, attenPart);
                            currentData = &attenPart;
                        }
                        moePart.Resize(currentData->dims);
                        moePart.Allocate(0.0f);

                        float sum = 0.0;
                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            float value = cur[v[j].second];
                            sum += value;
                        }
                        if (!needNorm) {
                            sum = 1.0;
                        }

                        for (int j = 0; j < this->num_experts_per_tok; j++) {
                            int idx = v[j].second;
                            float value = cur[idx];

                            value /= sum;
                            value *= routed_scaling_factor;
                            if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight") != weight.weight.end()) {
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

                        if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.shared_experts.gateup_proj.weight") != weight.weight.end()) {
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
                    moeFinal.expansionDims.clear();
                }

                moeFinal.Reshape(hiddenStates.dims);

                Data tempMoeFinal;
                tempMoeFinal.CopyFrom(moeFinal);
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                AddTo(hiddenStates, tempMoeFinal);

                if (cudaSe) {
                    moeFinal2.Reshape(hiddenStates.dims);
                    AddTo(hiddenStates, moeFinal2);
                }
            }
        }

        Data logits;
        std::vector <Data> curLogits;
        curLogits.resize(batch);

        if (batch > 1 && !all1) {
            int total = 0;
            std::vector <Data> lastTokens;
            std::vector <Data*> lastTokenPointers;
            lastTokens.resize(seqLens.size());
            for (int b = 0; b < seqLens.size(); b++) {
                Split(hiddenStates, 1, total + seqLens[b] - 1, total + seqLens[b], lastTokens[b]);
                total += seqLens[b];
                lastTokenPointers.push_back(&lastTokens[b]);
            }
            CatBatch(lastTokenPointers, 1, hiddenStates);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        ToDataType(logits, DataType::FLOAT32);
        std::vector <int> lastRet;

        ResetLogitsOfEOS(batch, &logits, pastKeyValues, generationConfigs);
        if (this->weight.dicts["model_type"] != "deepseek_v2") {
            for (auto &generationConfig : generationConfigs) {
                if (((GenerationConfig*)&generationConfig)->top_k <= 1) { 
                    ((GenerationConfig*)&generationConfig)->top_k = 10;
                }

                ((GenerationConfig*)&generationConfig)->top_p = 0.95;
                if (fabs(generationConfig.temperature - 1.0f) < 1e-6) {
                    ((GenerationConfig*)&generationConfig)->temperature = 0.6;
                } 
            }
        }

        int total = 0;

        bool allSimple = true, needLogits = false;
        int maxTopK = 1;
        for (int b = 0; b < batch; b++) {
            if (!generationConfigs[b].IsSimpleGreedy()) {
                allSimple = false;
                break;
            }
        }
        for (int b = 0; b < batch; b++) {
            needLogits |= generationConfigs[b].output_logits;
            maxTopK = std::max(maxTopK, generationConfigs[b].top_k);
        }

        if (batch > 1 && allSimple) {
            Data topk;
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            float *topkData = (float*)topk.cpuData;
            for (int b = 0; b < batch; b++) {
                lastRet.push_back((int) (topkData[0] + 1e-3));
                topkData += topk.Count(2);
            }
        } else if (batch > 1 && maxTopK <= 50 && !needLogits) {
            int maxTokenSetSize = 0;
            for (int b = 0; b < batch; b++) {
                maxTokenSetSize = std::max(maxTokenSetSize, (int)lastTokens.units[b].tokenSet.size());
            }
            std::vector <float> penaltyData = std::vector <float> (batch * maxTokenSetSize, -100.0f);
            std::vector <float> penaltyScaleData = std::vector <float> (batch, 1.0f);
            for (int b = 0; b < batch; b++) {
                int curId = 0;
                for (int i : lastTokens.units[b].tokenSet) {
                    penaltyData[b * maxTokenSetSize + curId] = i;
                    curId++;
                }
                penaltyScaleData[b] = generationConfigs[b].repeat_penalty;
            }
            Data penalty, penaltyScale;
            penalty.CopyFrom(Data(DataType::FLOAT32, {batch, maxTokenSetSize}, penaltyData));
            penaltyScale.CopyFrom(Data(DataType::FLOAT32, {batch}, penaltyScaleData));
            RepeatPenalty(logits, penalty, penaltyScale);
            Data topk;
            TopK(logits, topk, maxTopK);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                lastRet.push_back(LLMSamplingOnly(topk, b, generationConfigs[b]));
            }
        } else {
            for (int b = 0; b < batch; b++) {
                pointersK[b] = (&curLogits[b]);
            }
            SplitBatch(logits, 1, batch, pointersK);

            for (int b = 0; b < batch; b++) {
                Data &curLogit = curLogits[b];
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
            }
        }
        return lastRet;
    }

    bool DeepSeekV2Model::NeedAttentionMask(int qlen, int klen) {
        if (((qlen == 1) || (qlen >= 8192))) {
            return false;
        }
        return true;
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
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
        this->num_experts_per_tok = oldTopk;
    }
}
