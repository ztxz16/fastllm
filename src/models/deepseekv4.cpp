//
// Created by huangyuyang on 4/24/26.
//
// DeepSeek-V4 系列模型的 fastllm 适配框架。
//
// 当前文件提供：
//   - 模型类型 / 权重前缀的注册；
//   - 全部超参解析（包括 HC、Indexer、Compress、MTP 等 V4 特有字段）；
//   - YaRN RoPE（主分支与 compress 分支）的预计算；
//   - Forward / ForwardBatch 等接口的占位实现（暂未支持完整推理，
//     直接调用会抛出未实现错误，便于后续按层逐步填充）。
//
// 完整 forward 涉及 Hyper-Connections / 稀疏 attention / hash gate / MTP，
// 这些算子在 fastllm 中尚无对应实现，将会作为后续 PR 单独提交。
//

#include "deepseekv4.h"

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
    // 复用 deepseekv2.cpp 中的 yarn 工具函数
    extern float yarn_find_correction_dim(int num_rotations, int dim, float base, int max_position_embeddings);
    extern void yarn_find_correction_range(int low_rot, int high_rot, int dim, float base, int max_position_embeddings, int &low, int &high);
    extern float yarn_get_mscale(float scale, float mscale);
    extern std::vector <float> yarn_linear_ramp_mask(float min, float max, int dim);

    DeepSeekV4Model::DeepSeekV4Model() {
        this->model_type = "deepseek_v4";
        this->model_struct = "deepseek_v4";
        this->defaultChunkedPrefillSize = 2048;

        // V4 推荐 thinking 模式，需配合外部 chat_template；这里给一份最小默认值
        this->pre_prompt = "";
        this->user_role = "<|User|>";
        this->bot_role = "<|Assistant|>";
        this->history_sep = "";

        // 与 model.py 对齐：embed -> layers.X.attn / ffn -> head -> mtp.Z.*
        // 注意 V4 ckpt 的命名前缀直接是 layers / mtp / embed / head（无 model. 前缀）
        weight.embeddingNames.insert("embed.weight");
        weight.linearNames = {
            "head.weight",
            // attention 主权重
            "layers.*.attn.wq_a.weight", "layers.*.attn.wq_b.weight",
            "layers.*.attn.wkv.weight",
            "layers.*.attn.wo_a.weight", "layers.*.attn.wo_b.weight",
            // indexer / compressor 子权重
            "layers.*.attn.indexer.wq_b.weight",
            "layers.*.attn.indexer.weights_proj.weight",
            "layers.*.attn.indexer.compressor.wkv.weight",
            "layers.*.attn.indexer.compressor.wgate.weight",
            "layers.*.attn.compressor.wkv.weight",
            "layers.*.attn.compressor.wgate.weight",
            // moe gate / experts
            "layers.*.ffn.gate.weight",
            "layers.*.ffn.experts.*.w1.weight",
            "layers.*.ffn.experts.*.w2.weight",
            "layers.*.ffn.experts.*.w3.weight",
            "layers.*.ffn.shared_experts.w1.weight",
            "layers.*.ffn.shared_experts.w2.weight",
            "layers.*.ffn.shared_experts.w3.weight",
            // mtp 同构权重
            "mtp.*.attn.wq_a.weight", "mtp.*.attn.wq_b.weight",
            "mtp.*.attn.wkv.weight",
            "mtp.*.attn.wo_a.weight", "mtp.*.attn.wo_b.weight",
            "mtp.*.ffn.gate.weight",
            "mtp.*.ffn.experts.*.w1.weight",
            "mtp.*.ffn.experts.*.w2.weight",
            "mtp.*.ffn.experts.*.w3.weight",
            "mtp.*.ffn.shared_experts.w1.weight",
            "mtp.*.ffn.shared_experts.w2.weight",
            "mtp.*.ffn.shared_experts.w3.weight",
            "mtp.*.e_proj.weight", "mtp.*.h_proj.weight",
        };
    }

    void DeepSeekV4Model::InitParams() {
        basellm::InitParams();

        // -------- 基础尺寸 --------
        max_position_embeddings = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }

        // num_attention_heads / num_key_value_heads / embed_dim / block_cnt 已由 basellm 解析
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }

        // -------- Attention 维度 --------
        q_lora_rank = atoi(this->weight.dicts["q_lora_rank"].c_str());
        if (this->weight.dicts.find("o_lora_rank") != this->weight.dicts.end()) {
            o_lora_rank = atoi(this->weight.dicts["o_lora_rank"].c_str());
        }
        if (this->weight.dicts.find("o_groups") != this->weight.dicts.end()) {
            o_groups = atoi(this->weight.dicts["o_groups"].c_str());
        }
        if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
            head_dim_full = atoi(this->weight.dicts["head_dim"].c_str());
        }
        qk_rope_head_dim = atoi(this->weight.dicts["qk_rope_head_dim"].c_str());
        qk_nope_head_dim = head_dim_full - qk_rope_head_dim;
        head_dim = head_dim_full;
        rotary_dim = qk_rope_head_dim;

        if (this->weight.dicts.find("sliding_window") != this->weight.dicts.end()) {
            window_size = atoi(this->weight.dicts["sliding_window"].c_str());
        }

        // -------- Indexer --------
        if (this->weight.dicts.find("index_n_heads") != this->weight.dicts.end()) {
            index_n_heads = atoi(this->weight.dicts["index_n_heads"].c_str());
        }
        if (this->weight.dicts.find("index_head_dim") != this->weight.dicts.end()) {
            index_head_dim = atoi(this->weight.dicts["index_head_dim"].c_str());
        }
        if (this->weight.dicts.find("index_topk") != this->weight.dicts.end()) {
            index_topk = atoi(this->weight.dicts["index_topk"].c_str());
        }

        // -------- compress_ratios（数组） --------
        compress_ratios.clear();
        if (this->weight.dicts.find("compress_ratios") != this->weight.dicts.end()) {
            std::string err;
            auto j = json11::Json::parse(this->weight.dicts["compress_ratios"], err);
            if (j.is_array()) {
                for (auto &v : j.array_items()) {
                    compress_ratios.push_back(v.int_value());
                }
            }
        }
        if ((int)compress_ratios.size() < block_cnt) {
            compress_ratios.resize(block_cnt, 0);
        }

        // -------- MoE --------
        if (this->weight.dicts.find("moe_intermediate_size") != this->weight.dicts.end()) {
            moe_intermediate_size = atoi(this->weight.dicts["moe_intermediate_size"].c_str());
        }
        if (this->weight.dicts.find("n_shared_experts") != this->weight.dicts.end()) {
            n_shared_experts = atoi(this->weight.dicts["n_shared_experts"].c_str());
        }
        if (this->weight.dicts.find("n_routed_experts") != this->weight.dicts.end()) {
            num_experts = atoi(this->weight.dicts["n_routed_experts"].c_str());
        }
        if (this->weight.dicts.find("num_experts_per_tok") != this->weight.dicts.end()) {
            num_experts_per_tok = atoi(this->weight.dicts["num_experts_per_tok"].c_str());
        }
        norm_topk_prob = (this->weight.dicts.find("norm_topk_prob") != this->weight.dicts.end() &&
                          this->weight.dicts["norm_topk_prob"] == "true");
        if (this->weight.dicts.find("routed_scaling_factor") != this->weight.dicts.end()) {
            routed_scaling_factor = atof(this->weight.dicts["routed_scaling_factor"].c_str());
        }
        if (this->weight.dicts.find("scoring_func") != this->weight.dicts.end()) {
            scoring_func = this->weight.dicts["scoring_func"];
        }
        if (this->weight.dicts.find("topk_method") != this->weight.dicts.end()) {
            topk_method = this->weight.dicts["topk_method"];
        }
        if (this->weight.dicts.find("swiglu_limit") != this->weight.dicts.end()) {
            swiglu_limit = atof(this->weight.dicts["swiglu_limit"].c_str());
        }

        // -------- Hash 路由 / MTP --------
        if (this->weight.dicts.find("num_hash_layers") != this->weight.dicts.end()) {
            num_hash_layers = atoi(this->weight.dicts["num_hash_layers"].c_str());
        }
        if (this->weight.dicts.find("num_nextn_predict_layers") != this->weight.dicts.end()) {
            num_nextn_predict_layers = atoi(this->weight.dicts["num_nextn_predict_layers"].c_str());
        }

        // -------- Hyper-Connections --------
        if (this->weight.dicts.find("hc_mult") != this->weight.dicts.end()) {
            hc_mult = atoi(this->weight.dicts["hc_mult"].c_str());
        }
        if (this->weight.dicts.find("hc_sinkhorn_iters") != this->weight.dicts.end()) {
            hc_sinkhorn_iters = atoi(this->weight.dicts["hc_sinkhorn_iters"].c_str());
        }
        if (this->weight.dicts.find("hc_eps") != this->weight.dicts.end()) {
            hc_eps = atof(this->weight.dicts["hc_eps"].c_str());
        }

        // -------- RoPE / YaRN --------
        if (this->weight.dicts.find("rope_theta") != this->weight.dicts.end()) {
            rope_base = atof(this->weight.dicts["rope_theta"].c_str());
        }
        if (this->weight.dicts.find("compress_rope_theta") != this->weight.dicts.end()) {
            compress_rope_theta = atof(this->weight.dicts["compress_rope_theta"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.type") != this->weight.dicts.end()) {
            rope_scaling_type = this->weight.dicts["rope_scaling.type"];
            if (rope_scaling_type == "yarn") {
                rope_type = RoPEType::YARN;
            } else if (rope_scaling_type == "linear") {
                rope_type = RoPEType::LINEAR_SCALE;
            } else if (rope_scaling_type == "dynamic") {
                rope_type = RoPEType::DYMAMIC_NTK;
            }
        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.beta_fast") != this->weight.dicts.end()) {
            rope_scaling_beta_fast = atoi(this->weight.dicts["rope_scaling.beta_fast"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.beta_slow") != this->weight.dicts.end()) {
            rope_scaling_beta_slow = atoi(this->weight.dicts["rope_scaling.beta_slow"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.original_max_position_embeddings") != this->weight.dicts.end()) {
            rope_scaling_original_max_position_embeddings = atof(this->weight.dicts["rope_scaling.original_max_position_embeddings"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.mscale") != this->weight.dicts.end()) {
            rope_scaling_mscale = atof(this->weight.dicts["rope_scaling.mscale"].c_str());
        } else {
            rope_scaling_mscale = 1.0f;
        }
        if (this->weight.dicts.find("rope_scaling.mscale_all_dim") != this->weight.dicts.end()) {
            rope_scaling_mscale_all_dim = atof(this->weight.dicts["rope_scaling.mscale_all_dim"].c_str());
        } else {
            rope_scaling_mscale_all_dim = rope_scaling_mscale;
        }

        // 预计算 RoPE：主分支 (rope_theta) + compress 分支 (compress_rope_theta)
        auto pair = this->UpdateRotaryPosEmb(rope_base, rope_factor);
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));

        auto cpair = this->UpdateCompressRotaryPosEmb(compress_rope_theta, rope_factor);
        compressSinData.ToDevice(DataDevice::CPU);
        compressCosData.ToDevice(DataDevice::CPU);
        compressSinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->compressSin.size(), (int)this->compressSin[0].size() }, cpair.first));
        compressCosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->compressCos.size(), (int)this->compressCos[0].size() }, cpair.second));

        // -------- 注册 expert merge / 特殊层（与 V2 类似，用 V4 的命名） --------
        for (int i = 0; i < block_cnt; i++) {
            for (int j = -1; j < this->num_experts; j++) {
                std::string w1Name, w3Name, swigluName, downName;
                if (j == -1) {
                    w1Name = "layers." + std::to_string(i) + ".ffn.shared_experts.w1.weight";
                    w3Name = "layers." + std::to_string(i) + ".ffn.shared_experts.w3.weight";
                    swigluName = "layers." + std::to_string(i) + ".ffn.shared_experts.gateup.weight";
                    downName = "layers." + std::to_string(i) + ".ffn.shared_experts.w2.weight";
                } else {
                    w1Name = "layers." + std::to_string(i) + ".ffn.experts." + std::to_string(j) + ".w1.weight";
                    w3Name = "layers." + std::to_string(i) + ".ffn.experts." + std::to_string(j) + ".w3.weight";
                    swigluName = "layers." + std::to_string(i) + ".ffn.experts." + std::to_string(j) + ".gateup.weight";
                    downName = "layers." + std::to_string(i) + ".ffn.experts." + std::to_string(j) + ".w2.weight";
                }
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({w1Name, w3Name}, swigluName, std::string("linearSwiglu"))})
                );
                if (j != -1 || !GetCudaSharedExpert()) {
                    this->specialWeights[swigluName] = "linearSwiglu";
                    this->specialWeights[downName] = "linearColumn";
                }
                this->moeLinears.insert(w1Name);
                this->moeLinears.insert(w3Name);
                this->moeLinears.insert(downName);
            }
            // wkv / wo_a / indexer 的 latent 投影需保持高精度
            this->cantQuantLinears.insert("layers." + std::to_string(i) + ".attn.wkv.weight");
            this->cantQuantLinears.insert("layers." + std::to_string(i) + ".attn.wo_a.weight");
        }
    }

    std::pair<std::vector<float>, std::vector<float>> DeepSeekV4Model::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
        int dim = rotary_dim;
        std::vector <float> freqExtra, freqInter;
        for (int i = 0; i < dim; i += 2) {
            freqExtra.push_back(1.0 / pow(base, (float)i / rotary_dim));
            freqInter.push_back(1.0 / (rope_factor * pow(base, (float)i / rotary_dim)));
        }

        int low, high;
        yarn_find_correction_range(
            rope_scaling_beta_fast,
            rope_scaling_beta_slow,
            dim, base,
            (int)rope_scaling_original_max_position_embeddings,
            low, high
        );
        std::vector <float> invFreqMask = yarn_linear_ramp_mask(low, high, dim / 2);
        for (size_t i = 0; i < invFreqMask.size(); i++) {
            invFreqMask[i] = 1.0 - invFreqMask[i];
        }
        std::vector <float> invFreq;
        for (size_t i = 0; i < freqInter.size(); i++) {
            invFreq.push_back(freqInter[i] * (1.0 - invFreqMask[i]) + freqExtra[i] * invFreqMask[i]);
        }

        float _mscale = yarn_get_mscale(rope_factor, rope_scaling_mscale) /
                        yarn_get_mscale(rope_factor, rope_scaling_mscale_all_dim);

        int positions = std::max(max_positions, seqLen);
        sin.resize(positions);
        cos.resize(positions);
        for (int i = 0; i < positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < (int)invFreq.size() * 2; j++) {
                sin[i][j] = ::sin((float)i * invFreq[j % invFreq.size()]) * _mscale;
                cos[i][j] = ::cos((float)i * invFreq[j % invFreq.size()]) * _mscale;
            }
        }
        std::vector <float> fsin, fcos;
        for (size_t i = 0; i < sin.size(); i++) {
            fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
        }
        for (size_t i = 0; i < cos.size(); i++) {
            fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    std::pair<std::vector<float>, std::vector<float>> DeepSeekV4Model::UpdateCompressRotaryPosEmb(float base, float factor, int seqLen) {
        // compress 分支不做 YaRN 插值，对应 model.py 中 original_seq_len > 0 才开启的逻辑
        // 这里只生成纯 RoPE 频率（base 使用 compress_rope_theta）
        int dim = rotary_dim;
        std::vector <float> invFreq;
        for (int i = 0; i < dim; i += 2) {
            invFreq.push_back(1.0 / pow(base, (float)i / rotary_dim));
        }

        int positions = std::max(max_positions, seqLen);
        compressSin.resize(positions);
        compressCos.resize(positions);
        for (int i = 0; i < positions; i++) {
            compressSin[i].resize(rotary_dim);
            compressCos[i].resize(rotary_dim);
            for (int j = 0; j < (int)invFreq.size() * 2; j++) {
                compressSin[i][j] = ::sin((float)i * invFreq[j % invFreq.size()]);
                compressCos[i][j] = ::cos((float)i * invFreq[j % invFreq.size()]);
            }
        }
        std::vector <float> fsin, fcos;
        for (size_t i = 0; i < compressSin.size(); i++) {
            fsin.insert(fsin.end(), compressSin[i].begin(), compressSin[i].end());
        }
        for (size_t i = 0; i < compressCos.size(); i++) {
            fcos.insert(fcos.end(), compressCos[i].begin(), compressCos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    int DeepSeekV4Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                 const fastllm::Data &positionIds,
                                 std::vector<std::pair<Data, Data>> &pastKeyValues,
                                 const GenerationConfig &generationConfig,
                                 const LastTokensManager &lastTokens,
                                 std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues,
                            generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> DeepSeekV4Model::ForwardBatch(int batch,
                                                   const fastllm::Data &inputIds,
                                                   const fastllm::Data &attentionMask,
                                                   const fastllm::Data &positionIds,
                                                   std::vector<std::pair<Data, Data>> &pastKeyValues,
                                                   const GenerationConfig &generationConfig,
                                                   const LastTokensManager &lastTokens,
                                                   std::vector <std::vector <float>*> *retLogits) {
        inputIds.Print();
        return std::vector<int>(batch, 0);
    }

    std::vector <int> DeepSeekV4Model::ForwardBatch(int batch,
                                                   const Data &inputIds,
                                                   const std::vector <Data*> &attentionMask,
                                                   const std::vector <Data*> &positionIds,
                                                   const std::vector <int> &seqLens,
                                                   std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                                   const std::vector <GenerationConfig> &generationConfigs,
                                                   const LastTokensManager &lastTokens,
                                                   std::vector <std::vector <float>*> *retLogits) {
        ErrorInFastLLM("DeepSeekV4Model::ForwardBatch (multi-prompt) is not implemented yet.");
        return std::vector<int>(batch, 0);
    }

    bool DeepSeekV4Model::NeedAttentionMask(int qlen, int klen) {
        // 滑窗 + sparse 索引下，mask 由 sparse_attn 内部处理
        return false;
    }

    void DeepSeekV4Model::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                             const std::vector<std::map<std::string, int>> &params,
                                             fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                             fastllm::Data &positionIds) {
        // 先复用 DeepSeekV2 的 batch 填充逻辑：左填充 + 因果 mask + 顺序 positionIds
        // 后续若 hash gate 需要原始 input_ids，可以保持当前 inputIds 直接被消费
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = (int)inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
            }
            std::vector <float> ids(batch * maxLen, 0);
            std::vector <float> vpids(batch * maxLen, 0);
            std::vector <float> vmask(batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = (int)tokens.size();
                int base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
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
            std::vector <float> pids(batch);
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
            std::vector <float> vmasks(batch * maxLen, 0.0f);
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

    void DeepSeekV4Model::WarmUp() {
        // forward 尚未实现，warmup 暂时只打印提示，保持构造期间不崩溃
        printf("DeepSeekV4Model warmup skipped: forward not implemented yet.\n");
    }

    std::string DeepSeekV4Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string DeepSeekV4Model::MakeHistory(const std::string &history, int round,
                                             const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }
}
