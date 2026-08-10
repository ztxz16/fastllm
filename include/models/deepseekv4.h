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

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>

namespace fastllm {
    struct DeepSeekV4DecodeLayerCache {
        DeepSeekV4DecodeLayerCache() = default;
        DeepSeekV4DecodeLayerCache(const DeepSeekV4DecodeLayerCache &other);
        DeepSeekV4DecodeLayerCache &operator=(const DeepSeekV4DecodeLayerCache &other);

        bool initialized = false;
        int bsz = 0;
        int totalLen = 0;
        int headDim = 0;
        int windowSize = 0;
        int compressRatio = 0;
        int compressorWideDim = 0;
        Data windowKV;
        Data compressorKVRaw;
        Data compressorScoreRaw;
        int compressorRawTokenBase = 0;
        Data compressedKV;
        int compressedBlocks = 0;
        int compressedTokenBase = 0;
        int rawTailStartPos = 0;
        Data compressorTailKV;
        Data compressorTailScore;

        // C4 attention has a second, 128-wide compressor that feeds the
        // learned sparse-attention indexer.  Its lifetime mirrors the main
        // 512-wide attention compressor, but the resulting rows are used only
        // for scoring/selecting the compressed-attention top-k.
        int indexerCompressorWideDim = 0;
        Data indexerCompressorKVRaw;
        Data indexerCompressorScoreRaw;
        int indexerCompressorRawTokenBase = 0;
        Data indexerCompressedKV;
        int indexerCompressedBlocks = 0;

        // 单 token CUDA Graph 使用固定地址的原始 compressor ring 和压缩 KV。
        // 这些是运行时派生缓存，不进入 history/prefix cache 的拷贝。
        bool cudaGraphCacheReady = false;
        int cudaGraphRawCapacity = 0;
        int cudaGraphCompressedCapacity = 0;
        Data cudaGraphCompressorKVRing;
        Data cudaGraphCompressorScoreRing;
        Data cudaGraphApe;
        Data cudaGraphNormWeight;
        int cudaGraphIndexerRawCapacity = 0;
        int cudaGraphIndexerCompressedCapacity = 0;
        Data cudaGraphIndexerCompressorKVRing;
        Data cudaGraphIndexerCompressorScoreRing;
        Data cudaGraphIndexerApe;
        Data cudaGraphIndexerNormWeight;

        // Optional SM120 cache mirrors. FlashInfer's sparse MLA kernel reads
        // 64-token pages with a 584-byte logical token ABI (FP8 NoPE, BF16
        // RoPE, and UE8M0 footer scales). The generic FLOAT32/BF16 caches
        // above remain authoritative and provide the fallback on older GPUs.
        int cudaGraphPackedWindowCapacity = 0;
        int cudaGraphPackedCompressedCapacity = 0;
        Data cudaGraphPackedWindowKV;
        Data cudaGraphPackedCompressedKV;

        // FP8 indexer K rows use the same scalar power-of-two scale contract
        // as vLLM/DeepGEMM: 128 E4M3 bytes plus one FP32 scale per compressed
        // token.  The BF16 cache above remains authoritative for the generic
        // CUDA fallback and cache rollback.
        Data cudaGraphIndexerFp8KV;
        Data cudaGraphIndexerFp8Scale;
        Data cudaGraphIndexerIndices;
        Data cudaGraphIndexerLengths;
    };

    struct DeepSeekV4HistoryLayerCache {
        DeepSeekV4HistoryLayerCache() = default;
        DeepSeekV4HistoryLayerCache(const DeepSeekV4HistoryLayerCache &other);
        DeepSeekV4HistoryLayerCache &operator=(const DeepSeekV4HistoryLayerCache &other);

        bool initialized = false;
        int bsz = 0;
        int totalLen = 0;
        int headDim = 0;
        int windowSize = 0;
        int compressRatio = 0;
        int compressorWideDim = 0;
        Data windowKV;
        Data compressorKVRaw;
        Data compressorScoreRaw;
        int compressorRawTokenBase = 0;
        Data compressedKV;
        int compressedBlocks = 0;
        int compressedTokenBase = 0;
        int rawTailStartPos = 0;
        Data compressorTailKV;
        Data compressorTailScore;
        int indexerCompressorWideDim = 0;
        Data indexerCompressorKVRaw;
        Data indexerCompressorScoreRaw;
        int indexerCompressorRawTokenBase = 0;
        Data indexerCompressedKV;
        int indexerCompressedBlocks = 0;
    };

    struct DeepSeekV4HistoryCacheMemory {
        std::vector<int> inputToken;
        int tokens = 0;
        int blockCount = 0;
        uint64_t blockHash = 0;
        int recordTimes = 0;
        long long flushTime = 0;
        std::vector<DeepSeekV4HistoryLayerCache> layers;
        // DSpark derives a separate three-stage rolling window from target
        // hidden states.  A target-only prefix cannot resume speculative
        // decoding, so keep the committed draft context with the target cache.
        bool dsparkValid = false;
        int dsparkCommittedTokens = 0;
        std::vector<int> dsparkHistoryTokens;
        std::vector<Data> dsparkMainWindowKV;
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
        bool Get(const std::vector<int> &inputToken,
                 DeepSeekV4HistoryCacheMemory &memory, int &hitLen,
                 bool requireDspark = false);
    };

    struct DeepSeekV4DsparkPendingStep {
        int expectedInput = -1;
        int outputToken = -1;
    };

    struct DeepSeekV4DsparkTargetCapture {
        std::map<int, Data> targetHidden;
        Data headInput;
        // A steady-state CUDA graph also produces the sharded verifier logits
        // and each TP rank's local top-1 candidates.  These non-owning pointers
        // refer to the request's graph workspace; generic/eager paths leave
        // samplingReady false and keep using the established sampler.
        Data *samplingLogitsFloat = nullptr;
        Data *samplingGreedyIds = nullptr;
        Data *samplingGreedyScores = nullptr;
        std::map<int, void*> samplingReadyEvents;
        bool samplingReady = false;
        bool samplingDevicesDrained = false;
        // The steady verifier graph also precomputes the fixed eight-row
        // DSpark context projection.  AppendDsparkTargetHidden only commits
        // the dynamically accepted prefix into the rolling windows.
        std::vector<Data*> contextStageKV;
        int contextRows = 0;
        bool contextReady = false;
    };

    struct DeepSeekV4DsparkProposal {
        std::vector<int> tokens;
        // Conditional probability that each draft survives verification
        // after every preceding draft in the block has been accepted.
        std::vector<float> confidence;
        // A steady-state CUDA-graph draft can leave its proposal on the root
        // GPU.  ForwardDspark then feeds it directly into the target graph and
        // materializes tokens only after target sampling has synchronized the
        // round.  CPU, lower-SM and eager paths keep using tokens directly.
        Data *gpuTokens = nullptr;
        Data *gpuReadySignal = nullptr;
        Data *gpuReadySeen = nullptr;
        bool gpuDeferred = false;
    };

    struct DeepSeekV4DsparkContext {
        bool initialized = false;
        int committedTokens = 0;
        // DSpark 的三个 attention block 只缓存由目标模型 main hidden
        // 生成的滑窗 KV；draft token 自身的 KV 只在一次 proposal 内使用。
        std::vector<Data> mainWindowKV;
        std::deque<DeepSeekV4DsparkPendingStep> pending;
        std::vector<int> historyTokens;
        uint64_t proposedTokens = 0;
        uint64_t acceptedTokens = 0;
        uint64_t verifyRounds = 0;
        // An optional target-only cooldown can skip draft construction after
        // an unprofitable confidence probe.  It defaults to zero because
        // proposal confidence changes quickly between prose and code.
        int targetOnlyRoundsRemaining = 0;
        // Two-round request-local throughput window.  Confidence calibration
        // predicts whether a prefix should pay off; this window closes the
        // loop with the acceptance and wall time actually observed.
        double speculativeWindowMs = 0.0;
        uint64_t speculativeWindowCommitted = 0;
        int speculativeWindowRounds = 0;
        // CUDA Graph replays write to the addresses captured during warmup.
        // Keep verification outputs request-local and alive for the full graph
        // lifetime instead of publishing into an invocation-local temporary.
        DeepSeekV4DsparkTargetCapture targetCapture;
        // Retain the fixed-shape verification projection workspace per request.
        // Destroying these MultiCUDA tensors after every accepted block forced
        // allocator synchronization between otherwise asynchronous graphs.
        Data targetCombinedTemp;
        Data targetCombined;
        Data targetProjected;
        Data targetMainHidden;
        std::vector<Data> targetStageKV;
        std::vector<Data> targetCommittedKV;
        std::vector<Data> targetAppendedKV;
        // LLMSamplingBlock mutates its input, so keep a reusable copy separate
        // from the persistent target CUDA-graph capture.
        Data targetSamplingInput;
        Data targetSamplingLogits;
        Data targetSamplingLogitsFloat;
        // SM120 verifier postprocess workspace.  Local top-1 candidates are
        // gathered to the target root, reduced and rejection-sampled on GPU;
        // the result also drives the next draft's dynamic context commit.
        Data targetAcceptanceCandidateIds;
        Data targetAcceptanceCandidateScores;
        Data targetAcceptanceGlobalOffsets;
        Data targetAcceptanceResult;
        Data targetAcceptanceSignal;
        Data targetAcceptanceSeen;
        std::shared_ptr<void> targetAcceptanceHost;
        std::vector<int> targetAcceptanceDevices;
        std::vector<int> targetAcceptanceOffsets;
        int targetAcceptanceRootDevice = -1;
        bool targetAcceptanceReady = false;
        // vLLM proposes the next draft at the end of the current verifier
        // iteration.  Keep the same one-round lookahead here so the async
        // draft graph can overlap server output and scheduler bookkeeping.
        DeepSeekV4DsparkProposal prefetchedProposal;
        int prefetchedAnchorToken = -1;
        int prefetchedCommittedTokens = -1;
        bool prefetchedProposalReady = false;
        // Declared last so the draft graph executable and its pinned workspace
        // are released before the request-local cache tensors it references.
        std::shared_ptr<void> cudaGraphState;
    };

    struct DeepSeekV4RequestState {
        std::vector<DeepSeekV4DecodeLayerCache> decodeLayerCaches;
        std::vector<int> historyTokens;
        bool restoredHistoryCache = false;
        std::shared_ptr<DeepSeekV4DsparkContext> dspark;
        // Declared after dspark so graph executables are destroyed before the
        // persistent capture buffers referenced by their kernel nodes.
        std::shared_ptr<void> cudaGraphState;
    };

    class DeepSeekV4Model : public basellm {
    public:
        DeepSeekV4Model(); // 构造函数

        ~DeepSeekV4Model() override;

        virtual void InitParams(); // 初始化参数信息

        virtual std::map<std::string,
                         std::vector<std::pair<std::string, DataType> > >
                GetTensorMap(
                    const std::vector<std::string> &tensorNames) override;

        virtual std::string SelectSpecialWeightDevice(
                const std::string &weightName, int layerId) const override;

        virtual void OnModelWeightsLoaded() override;

        int GetTensorParallelAttentionSplitUnit() const {
            return o_groups > 0 && num_attention_heads % o_groups == 0 ?
                head_dim_full * (num_attention_heads / o_groups) : 0;
        }

        int GetTensorParallelOutputGroupSplitUnit() const {
            return o_lora_rank;
        }

        bool UseTensorParallelRoutedExperts() const {
            return dsparkEnabled;
        }

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

        virtual void TryRecordResponseContext(ResponseContext *context) override;

        virtual void OnResponseContextCreated(ResponseContext *context) override;

        virtual void OnResponseContextRemoved(ResponseContext *context) override;

        virtual bool UseGenericHistoryCache() const override { return false; }

        virtual bool UseModelSpecificScheduler() const override { return true; }

        virtual void RunModelSpecificScheduler() override;

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

        // -------- Embedded DSpark（DeepSeek-V4 的 mtp.0/1/2） --------
        bool dsparkEnabled = false;
        int dsparkTokens = 0;
        int dsparkLayers = 0;
        int dsparkNoiseTokenId = -1;
        int dsparkMarkovRank = 0;
        float dsparkConfidenceThreshold = 0.5f;
        std::vector<int> dsparkTargetLayerIds;
        std::vector<std::vector<Data*> > dsparkWeights;
        std::vector<std::vector<Data*> > dsparkBiass;
        std::atomic<bool> dsparkLogPrinted{false};
        std::atomic<long long> dsparkValidationCount{0};
        std::atomic<long long> dsparkProposedTokenCount{0};
        std::atomic<long long> dsparkAcceptedTokenCount{0};
        // Minimum steady latency observed for the complete draft and each
        // verifier prefix length, in microseconds.  The confidence scheduler
        // uses this model-wide curve across requests and ignores cold outliers
        // naturally by retaining minima.
        std::atomic<long long> dsparkDraftBestUs{0};
        std::array<std::atomic<long long>, 65> dsparkVerifyBestUs{};
        std::array<std::atomic<long long>, 64>
            dsparkDraftPositionAttempts{};
        std::array<std::atomic<long long>, 64>
            dsparkDraftPositionAccepts{};
        // Cumulative prefix-survival confidence, scaled by 1e6.  Comparing
        // this with observed survival supplies the online calibration that is
        // not shipped as metadata with the standalone checkpoint.
        std::array<std::atomic<long long>, 64>
            dsparkDraftPositionPredictedSurvival{};

        // 调试对齐用：decode 阶段保存已生成 token，并可选择完整重算上下文。
        std::vector<int> debugFullRecomputeTokens;
        int debugGeneratedTokens = 0;

        // 单请求 decode cache。当前 ForwardBatch(batch=1) 路径使用，后续可迁移到 paged cache。
        std::vector<DeepSeekV4DecodeLayerCache> decodeLayerCaches;

        DeepSeekV4HistoryCacheManager deepseekV4HistoryCacheManager;
        std::vector<int> deepseekV4HistoryTokens;
        std::mutex requestStateMutex;
        std::map<const void*, std::shared_ptr<DeepSeekV4RequestState> > requestStates;
        std::map<const void*, std::shared_ptr<DeepSeekV4RequestState> > requestStatesByFirstKey;
        std::shared_ptr<DeepSeekV4RequestState> pendingRequestState;
        std::shared_ptr<void> fallbackCudaGraphState;

        bool RestoreHistoryCacheMemory(const DeepSeekV4HistoryCacheMemory &memory);
        bool RestoreHistoryCacheMemory(const DeepSeekV4HistoryCacheMemory &memory,
                                       DeepSeekV4RequestState &state);
        void RecordHistorySnapshot(const std::vector<int> &tokens, int totalLen);
        void RecordHistorySnapshot(const std::vector<int> &tokens, int totalLen,
                                   const std::vector<DeepSeekV4DecodeLayerCache> &decodeCaches,
                                   const DeepSeekV4DsparkContext *dsparkContext = nullptr);
        std::shared_ptr<DeepSeekV4RequestState> GetRequestState(std::vector<std::pair<Data, Data> > &pastKeyValues);
        std::shared_ptr<DeepSeekV4RequestState> GetRequestStateByFirstKey(const Data *firstPastKey);

        std::vector<int> RunDsparkTarget(
                const std::vector<int> &tokenIds, int startPos,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *retLogits,
                DeepSeekV4DsparkTargetCapture *capture);

        void AppendDsparkTargetHidden(
                const DeepSeekV4DsparkTargetCapture &capture, int tokens,
                DeepSeekV4DsparkContext &context);

        DeepSeekV4DsparkProposal RunDsparkDraft(
            int anchorToken, DeepSeekV4DsparkContext &context,
            bool forceEager = false, bool deferGpuCopy = false);

        int SelectDsparkVerifyDrafts(
                const DeepSeekV4DsparkProposal &proposal,
                const DeepSeekV4DsparkContext &context,
                bool *preferTargetOnly = nullptr) const;

        std::vector<int> SampleDsparkTargetRows(
                Data &headInput,
                DeepSeekV4DsparkContext *persistentContext = nullptr);

        int ForwardDspark(
                const Data &inputIds,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *retLogits);
    };
}

#endif //FASTLLM_DEEPSEEKV4_H
