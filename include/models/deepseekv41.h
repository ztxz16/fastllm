//
// DeepSeek-V4.1 系列模型（DeepSeek-V4.1-Flash）。
//
// 相对 DeepSeek-V4 的架构变化（参考 hfmodels/DeepSeek-V4.1-Flash/inference/model.py）：
//   1. Hyper-Connections 的 pre/post/comb 系数由"上一个子层"计算、"下一个子层"使用：
//      attention 使用上一层 FFN 产出的 pre，FFN 使用本层 attention 产出的 pre，
//      最后一层 FFN 产出的 pre 用于 lm_head 前的 hc_pre。第一层 attention 使用 one-hot pre。
//   2. 跨层共享压缩 KV：compress_ratios 取值 0 / 1 / 2。只有 kv_source_layer_ids 中的层
//      拥有 compressor 与 compress_kv_cache，其后（直到下一个 source 之前）的层直接读取
//      该 cache；index_source_layer_ids 中的层运行 indexer，其它层复用最近 index source
//      的 top-k 结果。ratio 1 的 compressor 是纯投影（无 gate），ratio 2 做 softmax 池化。
//   3. 两级 indexer：candidate_source_layer_id 层先按 candidate_block_size 分块选出
//      candidate_topk_blocks 个块，之后的 index source 只在这些块内部做 top-k。
//      indexer 的 key 由 compressor 的 latent 经 wk + k_norm 派生，不再有独立 compressor。
//   4. Engram：在 engram_layer_ids 层之前，对 residual stream 做 n-gram 哈希查表
//      （embed 为 FP8 + 逐行 32 列一组的 UE8M0 scale），经 wkv 得到 key/value 并做门控写回。
//      哈希基于 tokenizer 归一化后的压缩 token id（engram_token_map，由 Python 侧生成）。
//   5. 无 hash 路由层；gate 使用 bias（noaux_tc），图像 token 使用 bias_vl。
//   6. 稠密权重 FP8 块大小 32x32，专家 FP4（沿 K 每 32 个一组 UE8M0 scale）。
//   7. mtp.* 为 DSpark 草稿模型（本实现暂不加载）；vision.* / aligner.* / image_* 为视觉编码器
//      （ViT + 3x3 下采样 aligner，实现见 src/models/deepseekv41_vision.cpp），图像 token 的嵌入
//      由 ForwardMultimodal 写入 input_ids 中 image_token_id 的位置。
//
// 当前实现目标：在通用 CUDA/CPU 路径上跑通文本 / 图文推理（单请求 prefill + decode），
// 专家与 Engram 表放在 CPU 内存，注意力层放在 GPU。
//

#ifndef FASTLLM_DEEPSEEKV41_H
#define FASTLLM_DEEPSEEKV41_H

#include "deepseekv4.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace fastllm {
    // 单层的推理缓存
    struct DeepSeekV41LayerCache {
        int totalLen = 0;             // 已经进入本层的 token 数
        Data windowKV;                // [windowSize, headDim] BF16 环形滑窗 KV，row = pos % windowSize

        // 以下仅 kv source 层使用
        Data compressedKV;            // [capacity, headDim] BF16，已加 RoPE 与 FP4 伪量化
        int compressedBlocks = 0;     // 已经写入的压缩块数
        Data indexK;                  // [capacity, indexHeadDim] BF16，indexer 的 key
        Data rawTailKV;               // FP32 [rawTail, headDim]，尚未凑满一组的 compressor 原始输入
        Data rawTailScore;            // FP32 [rawTail, headDim]
        int rawTail = 0;
    };

    // 一张图在 prompt 中占据的 span（含 image_start / image_newline / image_end）及其嵌入
    struct DeepSeekV41ImageSpan {
        int start = 0;
        int length = 0;
        Data embeds;                      // CPU FLOAT32 [length, dim]
    };

    // ==================== DSpark 投机解码 ====================
    //
    // mtp.0/1/2 是三个与主干同构的草稿层（compress_ratio 0，纯滑窗注意力，128 专家 top-3）。
    // 它们的滑窗 KV 不来自草稿 token 自身，而来自目标模型 dspark_target_layer_ids 各层
    // attention 输入（对 hc 份取均值）拼接后经 main_proj / main_norm 得到的 main_x，
    // 因此每个已提交位置在草稿侧只有一行 KV。一次 proposal 用 noise token 填满
    // block_size 个位置，逐位置产出候选 token（markov head 做 bigram 修正）与置信度。
    //
    // 草稿模型只影响接受率，不影响输出：所有候选都由目标模型逐个贪心比对，
    // 第一个不匹配处截断，因此开启 DSpark 与关闭时的贪心输出完全一致。

    // 草稿层的每层缓存
    struct DeepSeekV41DsparkLayerCache {
        Data windowKV;                // [1, windowSize, headDim] BF16 环形缓冲，row = pos % windowSize
    };

    struct DeepSeekV41DsparkState {
        std::vector<DeepSeekV41DsparkLayerCache> layers;
        int committed = 0;            // 已写入草稿滑窗的位置数（== 目标模型的 totalLen）
        int filled = 0;               // 滑窗中连续有效的位置数（前缀缓存恢复后从 0 开始重新累积）
        bool disabled = false;        // 该请求不适合投机（非贪心 / 图文 / 出错）
        // 下一轮的候选：drafts[j] 是位置 committed + 1 + j 的候选 token，
        // 只有下一次前向的起始位置为 committed 且首个 token 为 anchor 时才可用
        std::vector<int> drafts;
        std::vector<float> confidence;
        int anchor = -1;
        int anchorPos = -1;
        // 已经校验通过、等待调度器逐个取走的 token：(期望的输入 token, 应返回的 token)
        std::deque<std::pair<int, int> > pending;
        uint64_t rounds = 0, proposed = 0, accepted = 0;
    };

    // 一次"校验前向"里需要额外记录的信息：延后的滑窗写入、压缩缓存的回滚点、
    // 目标层的 main hidden，以及所有位置的贪心 token
    struct DeepSeekV41SpecScratch {
        bool captureMain = false;     // 采集 dspark_target_layer_ids 各层的 main hidden
        bool deferWindow = false;     // 滑窗写入延后到接受长度确定之后
        bool wantAllGreedy = false;   // head 对本片段的每个位置都出贪心 token
        std::vector<Data> mainHidden;         // [目标层数]，每个 [1, seqlen, dim]
        std::vector<Data> windowKV;           // [block_cnt]，本次前向的滑窗 KV（延后写入）
        std::vector<Data> rawKV, rawScore;    // kv source 层：压缩器的原始输入流（含旧 rawTail）
        std::vector<int> prevRawTail;         // kv source 层：前向之前的 rawTail 行数
        std::vector<int> prevBlocks;          // kv source 层：前向之前的压缩块数
        std::vector<int> greedy;              // wantAllGreedy 时每个位置的贪心 token
    };

    struct DeepSeekV41RequestState {
        std::vector<DeepSeekV41LayerCache> layers;
        std::vector<int> engramHistory;   // 每个已处理 token 的压缩 id（图像 token 为 -1）
        int totalLen = 0;
        int restoredLen = 0;              // 由前缀缓存恢复的 token 数（0 表示全新请求）
        // 图文请求：由 OnResponseContextCreated / ForwardMultimodal 记录，首个 prefill 块编码为 imageSpans
        const std::map <std::string, std::vector <Data*> > *pendingMultimodal = nullptr;
        bool imagesEncoded = false;
        std::vector<DeepSeekV41ImageSpan> imageSpans;
        // DSpark 草稿状态（首次 decode 时创建），随请求一起释放
        std::shared_ptr<DeepSeekV41DsparkState> dspark;
    };

    // 前缀缓存的一条记录：某段 token 序列处理完后的完整请求状态（张量放在 CPU）
    struct DeepSeekV41HistoryMemory {
        std::vector<int> tokens;          // 已经进入模型的 token（长度 == totalLen）
        int totalLen = 0;
        std::vector<DeepSeekV41LayerCache> layers;
        std::vector<int> engramHistory;
        long long flushTime = 0;
        int recordTimes = 0;
    };

    struct DeepSeekV41HistoryCacheManager {
        std::mutex locker;
        int maxRecordNum = 8;
        long long flushTime = 0;
        // Data 没有深拷贝赋值，记录一律通过 shared_ptr 持有，避免隐式拷贝造成别名
        std::map<std::vector<int>, std::shared_ptr<DeepSeekV41HistoryMemory> > memorys;

        void Record(const std::shared_ptr<DeepSeekV41HistoryMemory> &memory);
        // 按公共前缀长度从长到短列出候选（相同长度时记录更短的在前，更容易满足截断约束）；
        // 可截断性由模型侧检查
        std::vector<std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> > GetCandidates(
                const std::vector<int> &inputTokens);
    };

    // 一次前向中的一个序列片段：属于哪个请求、从哪个位置开始、多少个 token、在拼接输入中的偏移
    struct DeepSeekV41Segment {
        std::shared_ptr<DeepSeekV41RequestState> state;
        int startPos = 0;
        int seqlen = 0;
        int offset = 0;
        // 非空时本片段处于 DSpark 校验 / 采集模式（见 DeepSeekV41SpecScratch）
        DeepSeekV41SpecScratch *spec = nullptr;
    };

    // Engram 哈希元数据（由 tokenizer 归一化派生，Python 侧生成 JSON，此处加载）
    struct DeepSeekV41EngramMeta {
        bool loaded = false;
        std::vector<int> tokenMap;                       // 原始 token id -> 压缩 id
        int compressedVocabSize = 0;
        int padCompressedId = 0;                         // engram_pad_token_id 映射后的压缩 id
        std::vector<std::vector<int64_t> > multipliers;  // [engramLayer][maxNgram]
        std::vector<std::vector<int64_t> > primes;       // [engramLayer][(maxNgram-1)*heads]
        std::vector<std::vector<int64_t> > offsets;      // [engramLayer][(maxNgram-1)*heads]
    };

    class DeepSeekV41Model : public DeepSeekV4Model {
    public:
        DeepSeekV41Model();

        ~DeepSeekV41Model() override;

        void InitParams() override;

        std::map<std::string, std::vector<std::pair<std::string, DataType> > >
                GetTensorMap(const std::vector<std::string> &tensorNames) override;

        void OnModelWeightsLoaded() override;

        // V4.1 的 mtp.* 由本类自行解析（结构与 DeepSeek-V4 Flash 的内置 DSpark 不同）
        bool UsesEmbeddedV4Dspark() const override { return false; }

        int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr) override;

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr) override;

        // 图文前向：multimodalInput 由 Python 侧的 deepseek_v41_multimodal.py 构造：
        //   "pixel_values": 每张图一个 FLOAT32 [nPatches, 3 * patch * patch]
        //   "image_grid":   INT32 [numImages, 3]，每行 (span 起始位置, nVitH, nVitW)
        // 图像 span 的嵌入在请求的第一个 prefill 块编码并缓存在请求状态里，之后每个块（无论由调度器
        // 还是本函数切分）按位置重叠写入，因此 span 可以跨块；decode 步骤退化为普通 Forward。
        std::vector <int> ForwardMultimodal(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const std::map <std::string, std::vector <Data*> > &multimodalInput,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr) override;

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr) override;

        void WarmUp() override;

        // 每个 token 的长期 KV 缓存字节数（压缩 KV + indexer key），随 kvCacheDataType 变化
        long long KVCacheBytesPerToken() const;

        bool TryRestoreHistoryCache(std::vector<int> &inputTokens, int &cacheLen) override;
        void TryRecordHistoryCache(const std::vector<int> &allTokens) override;
        void TryRecordResponseContext(ResponseContext *context) override;
        void OnResponseContextCreated(ResponseContext *context) override;
        void OnResponseContextRemoved(ResponseContext *context) override;
        bool UseGenericHistoryCache() const override { return false; }
        bool UseModelSpecificScheduler() const override { return false; }

    protected:
        // -------- 跨层共享 --------
        std::vector<int> kv_source_layer_ids;
        std::vector<int> index_source_layer_ids;
        int candidate_source_layer_id = -1;
        int candidate_topk_blocks = 0;
        int candidate_block_size = 0;
        float gate_temp = 1.0f;
        int image_token_id = -1;

        // -------- 视觉编码器（deepseekv41_vision.cpp）--------
        int vision_n_layers = 0;          // 0 表示没有视觉塔
        int vision_dim = 1024;
        int vision_n_heads = 16;
        int vision_inter_dim = 2816;
        int vision_patch_size = 14;
        int vision_downsample_ratio = 3;
        float vision_rope_theta = 10000.0f;
        float vision_norm_eps = 1e-6f;
        bool VisionEnabled() const { return vision_n_layers > 0; }
        void InitVisionParams();
        bool IsVisionTensor(const std::string &name) const;
        // ViT + aligner：patches FLOAT32 [nVitH * nVitW, 3 * patch * patch]，
        // 输出 CPU FLOAT32 [ceil(nVitH / r) * ceil(nVitW / r), dim]
        void EncodeImage(const Data &patches, int nVitH, int nVitW, Data &output, const std::string &dumpPrefix = "");
        // 编码 multimodalInput 中的全部图像，得到每个 span 的嵌入（分隔符 + aligner 输出）并存入 state
        void EncodeImageSpans(const std::map <std::string, std::vector <Data*> > &multimodalInput,
                              DeepSeekV41RequestState &state);
        // 若本块 [startPos, startPos + seqlen) 与某个图像 span 重叠：embeds = 文本嵌入并写入图像嵌入
        //（CPU FLOAT32 [1, seqlen, dim]），imageMask[i] = 1 表示图像 token；返回是否有重叠
        bool PrepareImageEmbeds(const Data &inputIds, int startPos, DeepSeekV41RequestState &state,
                                Data &embeds, std::vector<int> &imageMask);

        // 每层派生信息
        std::vector<int> kvSourceOf;      // 本层读取哪一层的压缩 KV（-1 表示纯滑窗）
        std::vector<int> indexSourceOf;   // 本层复用哪一层的 top-k
        std::vector<char> isKvSource;
        std::vector<char> isIndexSource;

        // -------- Engram --------
        std::vector<int> engram_layer_ids;
        std::vector<int64_t> engram_num_embeddings;
        int engram_max_ngram_size = 4;
        int engram_vocab_size = 0;
        int engram_n_heads = 0;
        int engram_head_dim = 0;
        int engram_pad_token_id = 2;
        int engram_compressed_vocab_size = 0;
        DeepSeekV41EngramMeta engramMeta;
        std::vector<std::shared_ptr<void> > engramTables;   // 每个 engram 层一张表（实现见 cpp）
        // 跨层预取：两个 engram 层相距很远（默认层 1 与层 14），而哈希只依赖 token 历史、
        // 不依赖中间激活，所以下一层的行号与表行可以在本层计算时后台算好。
        // 由 FASTLLM_DSV41_ENGRAM_PREFETCH 打开，默认关闭（实现见 cpp）。
        std::shared_ptr<void> engramPrefetch;

        // -------- 请求状态 --------
        // 状态同时按 &pastKeyValues（单请求 Forward）与 &pastKeyValues[0].first（调度器的多请求
        // ForwardBatch 只传每层 Data 指针）两把 key 索引；两者指向同一个 shared_ptr。
        std::mutex v41StateMutex;
        std::map<const void*, std::shared_ptr<DeepSeekV41RequestState> > v41States;
        std::map<const void*, std::shared_ptr<DeepSeekV41RequestState> > v41StatesByFirstKey;
        std::shared_ptr<DeepSeekV41RequestState> v41PendingRestoredState;   // TryRestoreHistoryCache 产生，
                                                                             // OnResponseContextCreated 接管
        DeepSeekV41HistoryCacheManager v41HistoryCache;

        // -------- 单 token decode 的 CUDA Graph --------
        // 捕获的两段（见 ForwardSegments 里的说明）只读写权重与解码工作区，不碰任何
        // 请求私有的 KV 缓存，因此整个模型共用一份图；状态自带互斥量，抢不到锁的并发
        // 前向直接退回逐算子执行。
        std::shared_ptr<void> v41CudaGraphSlot;

        std::shared_ptr<DeepSeekV41RequestState> GetOrCreateState(
                std::vector<std::pair<Data, Data> > &pastKeyValues, bool reset);
        std::shared_ptr<DeepSeekV41RequestState> GetStateByFirstKey(const Data *firstKey);
        void RegisterState(const void *vectorKey, const void *firstKey,
                           const std::shared_ptr<DeepSeekV41RequestState> &state);

        // 前缀缓存：把请求状态快照到 CPU / 从快照恢复前 hitLen 个 token 的状态
        std::shared_ptr<DeepSeekV41HistoryMemory> SnapshotState(const DeepSeekV41RequestState &state,
                                                                const std::vector<int> &allTokens);
        std::shared_ptr<DeepSeekV41RequestState> RestoreState(const DeepSeekV41HistoryMemory &memory, int hitLen);
        // 检查快照能否截断到 len 个 token（滑窗环形缓存与压缩尾块的约束）
        bool CanTruncateHistory(const DeepSeekV41HistoryMemory &memory, int len) const;

        // 实际的前向：多个序列片段拼接成一个 token 流，Linear / MoE / Engram 查表按整批执行，
        // RoPE、压缩、indexer、稀疏注意力按片段分别执行。
        // inputEmbeds 非空时直接作为嵌入（[1, tokens, dim]，供视觉输入使用）；
        // imageMask 非空时标记每个 token 是否为图像 token（Engram 历史置 -1，路由改用 gate.bias_vl）。
        std::vector<int> ForwardSegments(
                std::vector<DeepSeekV41Segment> &segments,
                const Data &inputIds,
                const Data *inputEmbeds,
                const std::vector<int> *imageMask,
                const std::vector<GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector<std::vector<float>*> *retLogits,
                std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues);

        // ForwardSegments 的 DSpark 包装：打开 main hidden 采集，单片段且 allowSpeculate
        // 时把候选拼进输入做一次校验，返回后按接受长度提交 / 回滚，并生成下一轮候选。
        // DSpark 关闭（或请求不支持）时等价于直接调用 ForwardSegments。
        std::vector<int> ForwardSegmentsWithDspark(
                std::vector<DeepSeekV41Segment> &segments,
                const Data &inputIds,
                const Data *inputEmbeds,
                const std::vector<int> *imageMask,
                const std::vector<GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector<std::vector<float>*> *retLogits,
                std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues,
                bool allowSpeculate);

        // 单序列前向：构造单个片段调用 ForwardSegments。inputEmbeds 非空时代替 embedding 查表，
        // imageMask 非空时（长度 seqlen，1 = 图像 token）Engram 置 -1 且路由改用 gate.bias_vl；
        // 两者为空且请求带多模态输入时，在此按位置编码 / 写入图像嵌入。
        std::vector <int> ForwardSingle(
                const Data &inputIds,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector <std::vector <float>*> *logits,
                const Data *inputEmbeds,
                const std::vector<int> *imageMask);

        // -------- DSpark 投机解码（src/models/deepseekv41_dspark.cpp）--------
        bool v41DsparkEnabled = false;
        int v41DsparkTokens = 0;              // 每轮最多校验的 draft token 数（<= block size）
        int v41DsparkBlockSize = 0;           // checkpoint 训练时的 block size
        int v41DsparkLayers = 0;              // mtp.* 的层数（num_nextn_predict_layers）
        int v41DsparkNoiseTokenId = -1;
        int v41DsparkMarkovRank = 0;
        int v41DsparkExperts = 0;             // dspark_n_routed_experts
        int v41DsparkTopk = 0;                // dspark_num_experts_per_tok
        float v41DsparkConfidenceThreshold = 0.0f;
        std::vector<int> v41DsparkTargetLayerIds;
        std::vector<char> v41IsDsparkTarget;  // [block_cnt]
        std::vector<std::vector<Data*> > v41DsparkMoeWeights, v41DsparkMoeBiass;
        std::atomic<long long> v41DsparkRounds{0};
        std::atomic<long long> v41DsparkProposedTokens{0};
        std::atomic<long long> v41DsparkAcceptedTokens{0};
        std::atomic<long long> v41DsparkVerifyRounds{0};

        void InitDsparkParams();
        bool DsparkTensorNeeded(const std::string &name) const;
        // 请求是否可以做投机解码（贪心、无 logits 输出、无工具约束、非图文）
        bool DsparkSupportsRequest(const GenerationConfig &config,
                                   const DeepSeekV41RequestState &state) const;
        std::shared_ptr<DeepSeekV41DsparkState> GetOrCreateDsparkState(
                DeepSeekV41RequestState &state, int startPos);
        // 若队首的候选与本次输入一致，直接返回已经校验过的 token（不做前向），否则返回 -1
        int DsparkTakePending(DeepSeekV41RequestState &state, const Data &inputIds, int seqlen);
        // 若 state 有可用的候选，把 inputIds 扩展成 [anchor, draft...]，返回 draft 个数
        int DsparkBuildVerifyInput(DeepSeekV41RequestState &state, const Data &inputIds,
                                   int startPos, Data &verifyIds, std::vector<int> &drafts);
        // 把本次前向的前 accept 个位置提交进各层缓存，其余回滚
        void DsparkCommitPrefix(DeepSeekV41RequestState &state, DeepSeekV41SpecScratch &scratch,
                                int startPos, int accept, int forwarded);
        // 用 main hidden 更新草稿滑窗，并为下一轮生成候选
        void DsparkAdvance(DeepSeekV41RequestState &state, DeepSeekV41SpecScratch &scratch,
                           int startPos, int committed, int anchorToken);
        // 三层草稿前向 + markov head + confidence head
        void DsparkRunDraft(DeepSeekV41DsparkState &dspark, int anchorToken,
                            std::vector<int> &tokens, std::vector<float> &confidence);
        void DsparkBuildMoeWeights();
        // 调试：把各层缓存长度写到 FASTLLM_DSV41_DEBUG_STATE 指定的文件
        void DsparkDebugDumpState(const DeepSeekV41RequestState &state, const char *tag);
        void DsparkReportStats();

        void LoadEngramMeta();
        void BuildEngramPrimes();

        // 计算 [tokens, (maxNgram-1)*heads] 的哈希行号
        void ComputeEngramHashes(int engramLayerIndex,
                                 const std::vector<int> &history, int startPos, int seqlen,
                                 std::vector<int64_t> &rows) const;

        // 从 FP8 表中取行，输出 BF16 [tokens, cols * headDim]
        // prepMs 非空时回填"准备输出 Data"那一段的耗时（计时用，见 FASTLLM_DSV41_ENGRAM_PROFILE）
        void GatherEngramRows(int layer, const std::vector<int64_t> &rows, int tokens, Data &output,
                              double *prepMs = nullptr);

        // 对一批片段做 Engram：各片段分别算哈希行号，查表 / wkv / 门控按整批执行
        void RunEngram(int layer, int engramLayerIndex, const std::vector<DeepSeekV41Segment> &segments,
                       Data &hiddenStates);
    };
}

#endif //FASTLLM_DEEPSEEKV41_H
