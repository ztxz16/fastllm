//
// Created by huangyuyang on 4/24/26.
//
// DeepSeek-V4 系列模型（DeepSeek-V4-Pro / DeepSeek-V4-Flash）。
//
// 架构要点（参考 hfmodels/DeepSeek-V4-Flash/inference/model.py）：
//   1. Hyper-Connections (HC)：用 hc_mult 份隐藏状态副本替代普通残差，
//      通过 hc_pre / hc_post 在每个 attn / ffn 模块前后做加权混合，
//      其中 hc_pre 还会借助 Sinkhorn 计算 pre / post / comb 权重。
//   2. 混合注意力 = 滑动窗口注意力 + 可选的 Compressed Sparse Attention (CSA)
//      与 Heavily Compressed Attention (HCA)。每层的 compress_ratio 由 config
//      中的 compress_ratios 数组指定（0 表示纯滑窗，4 表示 CSA + Indexer，
//      其它值如 128 表示 HCA）。
//   3. MQA：q 走 LoRA（wq_a + q_norm + wq_b），kv 用单头共享（num_key_value_heads=1）；
//      o 走 grouped low-rank（wo_a + wo_b，按 o_groups 分组）。
//   4. MoE：前 num_hash_layers 个 MoE 层使用 hash 路由（gate.tid2eid 决定专家），
//      其余使用打分函数 sqrtsoftplus + noaux_tc top-k；num_experts_per_tok 个路由专家
//      + n_shared_experts 个共享专家，激活带 swiglu_limit。
//   5. 多 Token 预测（MTP）：在主 N 层之外再额外维护 num_nextn_predict_layers 层，
//      复用 embed / head，但有独立 e_proj / h_proj / enorm / hnorm / norm。
//   6. 量化：权重以 FP8_E4M3 + ue8m0 scale 存储，部分 expert 使用 FP4_E2M1FN_X2。
//
// 当前文件先给出类骨架与数据成员定义，便于后续逐步实现。
//

#ifndef FASTLLM_DEEPSEEKV4_H
#define FASTLLM_DEEPSEEKV4_H

#include "basellm.h"
#include "deepseekv2.h"

#include "cmath"

#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>

namespace fastllm {
    struct DeepSeekV4DecodeLayerCache {
        bool initialized = false;
        int bsz = 0;
        int totalLen = 0;
        int headDim = 0;
        int windowSize = 0;
        int compressRatio = 0;
        int compressorWideDim = 0;
        std::vector<float> windowKV;
        Data windowKVData;
        std::vector<float> compressorKVRaw;
        std::vector<float> compressorScoreRaw;
        Data compressedKV;
        Data compressedKVCuda;
        int compressedBlocks = 0;
        int compressedTokenBase = 0;
        int rawTailStartPos = 0;
        std::vector<float> compressorTailKV;
        std::vector<float> compressorTailScore;
    };

    struct DeepSeekV4HistoryLayerCache {
        bool initialized = false;
        int bsz = 0;
        int totalLen = 0;
        int headDim = 0;
        int windowSize = 0;
        int compressRatio = 0;
        int compressorWideDim = 0;
        std::vector<float> windowKV;
        std::vector<float> compressorKVRaw;
        std::vector<float> compressorScoreRaw;
        Data compressedKV;
        int compressedBlocks = 0;
        int compressedTokenBase = 0;
        int rawTailStartPos = 0;
        std::vector<float> compressorTailKV;
        std::vector<float> compressorTailScore;
    };

    struct DeepSeekV4HistoryCacheMemory {
        std::vector<int> inputToken;
        int tokens = 0;
        int blockCount = 0;
        uint64_t blockHash = 0;
        int recordTimes = 0;
        long long flushTime = 0;
        std::vector<DeepSeekV4HistoryLayerCache> layers;
    };

    struct DeepSeekV4HistoryCacheManager {
        std::mutex locker;
        int logicalBlockSize = 256;
        int maxRecordNum = 8;
        long long flushTime = 0;
        std::map<std::vector<int>, DeepSeekV4HistoryCacheMemory> memorys;
        std::map<uint64_t, std::vector<int> > blockIndex;

        void SetMaxRecordNum(int maxRecordNum);
        void Record(const DeepSeekV4HistoryCacheMemory &memory);
        bool Get(const std::vector<int> &inputToken, DeepSeekV4HistoryCacheMemory &memory, int &hitLen);
    };

    class DeepSeekV4Model : public basellm {
    public:
        DeepSeekV4Model(); // 构造函数

        virtual void InitParams(); // 初始化参数信息

        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr);

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr);

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr);

        // 是否需要生成 AttentionMask
        virtual bool NeedAttentionMask(int qlen, int klen);

        // 根据输入的 tokens 生成 LLM 推理的输入
        virtual void FillLLMInputsBatch(std::vector <std::vector <float> > &inputTokens,
                                        const std::vector <std::map <std::string, int> > &params,
                                        Data &inputIds, Data &attentionMask, Data &positionIds);

        virtual void WarmUp(); // 预热

        virtual bool TryRestoreHistoryCache(std::vector<int> &inputTokens, int &cacheLen) override;

        virtual void TryRecordHistoryCache(const std::vector<int> &allTokens) override;

        virtual bool UseGenericHistoryCache() const override { return false; }

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成 prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新 history

        // 计算 RoPE（YaRN，含 mscale），与 DeepSeekV2 接口保持一致
        std::pair<std::vector<float>, std::vector<float>> UpdateRotaryPosEmb(float base, float factor, int seqLen = 0);

        // 计算 compress 注意力使用的 RoPE（compress_rope_theta 对应原始上下文长度）
        std::pair<std::vector<float>, std::vector<float>> UpdateCompressRotaryPosEmb(float base, float factor, int seqLen = 0);

    protected:
        // -------- 通用 RoPE / 归一化参数 --------
        RoPEType rope_type = RoPEType::YARN;
        float rope_base = 10000.f;
        float rope_factor = 1.f;
        float rms_norm_eps = 1e-6;

        int max_position_embeddings = 1048576;
        float compress_rope_theta = 160000.f;

        // -------- YaRN 相关 --------
        int rope_scaling_beta_fast = 32;
        int rope_scaling_beta_slow = 1;
        float rope_scaling_mscale = 1.0f;
        float rope_scaling_mscale_all_dim = 1.0f;
        float rope_scaling_original_max_position_embeddings = 65536;
        std::string rope_scaling_type = "yarn";

        // 用于 compress 注意力分支的 sin/cos
        Data compressSinData, compressCosData;
        std::vector<std::vector<float> > compressSin, compressCos;

        // -------- Attention 维度 --------
        // V4 的 wq_a/wq_b 一律存在（必有 q_lora_rank），无独立 q_proj
        int q_lora_rank = 1024;
        int o_lora_rank = 1024;
        int o_groups = 8;        // o 投影按头分组的组数
        int head_dim_full = 512; // V4 中 attention head 的总维度（包含 nope + rope）
        int qk_rope_head_dim = 64;
        int qk_nope_head_dim = 0; // = head_dim_full - qk_rope_head_dim
        // 兼容父类基础属性
        // num_attention_heads / num_key_value_heads 沿用基类
        int window_size = 128; // 滑动窗口大小

        // -------- Indexer / Compressor --------
        int index_n_heads = 64;
        int index_head_dim = 128;
        int index_topk = 512;

        // 每层的 compress_ratio：0 表示纯滑窗，4 表示 CSA(+Indexer)，其它（如 128）表示 HCA
        std::vector <int> compress_ratios;

        // -------- MoE --------
        int num_hash_layers = 0;            // 前若干层使用 hash 路由
        int num_nextn_predict_layers = 0;   // MTP 层数（n_mtp_layers）
        int moe_intermediate_size = 0;      // expert 内部维度
        std::string scoring_func = "sqrtsoftplus"; // softmax / sigmoid / sqrtsoftplus
        std::string topk_method = "noaux_tc";
        float swiglu_limit = 0.f;           // SwiGLU 截断
        bool mergeSwiglu = false;

        // -------- Hyper-Connections --------
        int hc_mult = 4;
        int hc_sinkhorn_iters = 20;
        float hc_eps = 1e-6f;

        // -------- 缓存 --------
        std::vector <std::vector <Data*> > weights;
        std::vector <std::vector <Data*> > biass;

        // 调试对齐用：decode 阶段保存已生成 token，并可选择完整重算上下文。
        std::vector<int> debugFullRecomputeTokens;
        int debugGeneratedTokens = 0;

        // 单请求 decode cache。当前 ForwardBatch(batch=1) 路径使用，后续可迁移到 paged cache。
        std::vector<DeepSeekV4DecodeLayerCache> decodeLayerCaches;

        DeepSeekV4HistoryCacheManager deepseekV4HistoryCacheManager;
        std::vector<int> deepseekV4HistoryTokens;

        bool RestoreHistoryCacheMemory(const DeepSeekV4HistoryCacheMemory &memory);
        void RecordHistorySnapshot(const std::vector<int> &tokens, int totalLen);
    };
}

#endif //FASTLLM_DEEPSEEKV4_H
