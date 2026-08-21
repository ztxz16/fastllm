#include "dots3_note.h"

#include "executor.h"
#include "utils.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace fastllm {
    namespace {
        constexpr int kDefaultChunkedPrefillSize = 4096;
#ifdef USE_CUDA
        constexpr long long kLongKvFreeReserveBytes =
            7LL * 1024LL * 1024LL * 1024LL;
        constexpr long long kLongKvResidentBudgetBytes =
            3LL * 1024LL * 1024LL * 1024LL;
        constexpr int kSparsePrefillMaxKeys = 16384;
#endif

        int GetIntConfig(const WeightMap &weight, const std::string &name, int fallback) {
            auto it = weight.dicts.find(name);
            return it == weight.dicts.end() ? fallback : atoi(it->second.c_str());
        }

        float GetFloatConfig(const WeightMap &weight, const std::string &name, float fallback) {
            auto it = weight.dicts.find(name);
            return it == weight.dicts.end() ? fallback : atof(it->second.c_str());
        }

        bool StartsWith(const std::string &value, const std::string &prefix) {
            return value.size() >= prefix.size() &&
                   value.compare(0, prefix.size(), prefix) == 0;
        }

        bool UsesCudaIndexer(const Executor &executor) {
#ifdef USE_CUDA
            return executor.firstDevice == "cuda" ||
                   executor.firstDevice.rfind("cuda:", 0) == 0 ||
                   executor.firstDevice == "multicuda" ||
                   executor.firstDevice.rfind("multicuda:", 0) == 0;
#else
            (void)executor;
            return false;
#endif
        }

        bool EnvFlagEnabled(const char *name) {
            const char *value = std::getenv(name);
            return value != nullptr && value[0] != '\0' &&
                   std::strcmp(value, "0") != 0 &&
                   std::strcmp(value, "false") != 0 &&
                   std::strcmp(value, "FALSE") != 0 &&
                   std::strcmp(value, "off") != 0 &&
                   std::strcmp(value, "OFF") != 0;
        }

        bool HistoryCacheDebugEnabled() {
            return EnvFlagEnabled("FASTLLM_DOTS3_NOTE_HISTORY_DEBUG");
        }

        double ProfileNowMs() {
            using Clock = std::chrono::steady_clock;
            return std::chrono::duration<double, std::milli>(
                Clock::now().time_since_epoch()).count();
        }

        struct ScopedExecutorProfiler {
            std::string opType;
            double startMs;
            float startProfile;
            Executor *executor;

            explicit ScopedExecutorProfiler(const std::string &opType)
                : opType(opType), startMs(ProfileNowMs()),
                  startProfile(0.0f),
                  executor((Executor *)GetExecutor()) {
                if (executor != nullptr) {
                    startProfile = executor->GetProfilerTotal();
                }
            }

            ~ScopedExecutorProfiler() {
                if (executor == nullptr) {
                    return;
                }
                float elapsed =
                    (float)((ProfileNowMs() - startMs) * 0.001);
                float nested = executor->GetProfilerTotal() - startProfile;
                if (elapsed - nested > 1.0e-7f) {
                    executor->AddProfiler(opType, elapsed - nested);
                }
            }
        };

    }

    Dots3NoteModel::Dots3NoteModel() {
        this->model_type = "dots3_note";
        this->model_struct = "dots3_note";
        this->canDoBatchForward = false;
        this->defaultChunkedPrefillSize = kDefaultChunkedPrefillSize;

        this->pre_prompt = "";
        this->user_role = "";
        this->bot_role = "";
        this->history_sep = "";

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight",
            "model.layers.*.self_attn.q_a_proj.weight",
            "model.layers.*.self_attn.q_b_proj.weight",
            "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.*.self_attn.kv_b_proj.weight",
            "model.layers.*.self_attn.g_proj.weight",
            "model.layers.*.self_attn.o_proj.weight",
            "model.layers.*.self_attn.indexer.wq_b.weight",
            "model.layers.*.self_attn.indexer.wk.weight",
            "model.layers.*.self_attn.indexer.weights_proj.weight",
            "model.layers.*.mlp.gate.weight",
            "model.layers.*.mlp*gate_proj.weight",
            "model.layers.*.mlp*up_proj.weight",
            "model.layers.*.mlp*down_proj.weight"
        };
    }

    Dots3NoteModel::~Dots3NoteModel() {
        ShutdownRuntime();
        {
            std::lock_guard<std::mutex> guard(historyCacheMutex);
            pendingHistoryCache.reset();
            historyCache.clear();
        }
        {
            std::lock_guard<std::mutex> guard(responseContextsMutex);
            responseContexts.clear();
        }
        {
            std::lock_guard<std::mutex> guard(indexerCacheMutex);
            indexerCaches.clear();
        }
    }

    std::map<std::string, std::vector<std::pair<std::string, DataType>>>
    Dots3NoteModel::GetTensorMap(const std::vector<std::string> &tensorNames) {
        auto ret = basellm::GetTensorMap(tensorNames);
        for (const std::string &name : tensorNames) {
            bool skip = StartsWith(name, "vision_encoder.") ||
                        StartsWith(name, "audio_encoder.") ||
                        StartsWith(name, "model.layers.46.") ||
                        StartsWith(name, "model.mtp.");
            if (skip) {
                ret[name].clear();
            } else if (name.find(".self_attn.indexer.k_norm.") !=
                       std::string::npos) {
                // Transformers promotes both affine parameters to FP32 before
                // the index-key LayerNorm.
                ret[name] = {{name, DataType::FLOAT32}};
            }
        }
        return ret;
    }

    void Dots3NoteModel::InitParams() {
        basellm::InitParams();

        AssertInFastLLM(block_cnt == 46,
                        "FastLLM Dots3-Note currently expects 46 text layers.\n");

        rms_norm_eps = GetFloatConfig(weight, "rms_norm_eps", 1.0e-5f);
        num_experts = GetIntConfig(weight, "n_routed_experts", 256);
        n_shared_experts = GetIntConfig(weight, "n_shared_experts", 1);
        num_experts_per_tok = GetIntConfig(weight, "num_experts_per_tok", 8);
        routed_scaling_factor = GetFloatConfig(weight, "routed_scaling_factor", 1.0f);
        norm_topk_prob = true;

        fullNumHeads = GetIntConfig(weight, "num_attention_heads", 128);
        fullQLoraRank = GetIntConfig(weight, "q_lora_rank", 1024);
        fullKVLoraRank = GetIntConfig(weight, "kv_lora_rank", 512);
        fullQKNopeHeadDim = GetIntConfig(weight, "qk_nope_head_dim", 128);
        fullQKRopeHeadDim = GetIntConfig(weight, "qk_rope_head_dim", 64);
        fullVHeadDim = GetIntConfig(weight, "v_head_dim", 128);
        fullRopeTheta = GetFloatConfig(weight, "rope_theta", 80000000.0f);

        swaNumHeads = GetIntConfig(weight, "swa_num_attention_heads", 64);
        swaQLoraRank = GetIntConfig(weight, "swa_q_lora_rank", 1024);
        swaKVLoraRank = GetIntConfig(weight, "swa_kv_lora_rank", 1024);
        swaQKNopeHeadDim = GetIntConfig(weight, "swa_qk_nope_head_dim", 192);
        swaQKRopeHeadDim = GetIntConfig(weight, "swa_qk_rope_head_dim", 64);
        swaVHeadDim = GetIntConfig(weight, "swa_v_head_dim", 128);
        swaRopeTheta = GetFloatConfig(weight, "swa_rope_theta", 50000.0f);

        shortContextLimit = GetIntConfig(weight, "sliding_window_size", 513);
        AssertInFastLLM(shortContextLimit > 0,
                        "FastLLM Dots3-Note requires a positive sliding window.\n");
        indexHeads = GetIntConfig(weight, "index_n_heads", 64);
        indexHeadDim = GetIntConfig(weight, "index_head_dim", 128);
        indexTopK = GetIntConfig(weight, "index_topk", 2048);
        AssertInFastLLM(indexHeads == 64 && indexHeadDim == 128 &&
                        indexTopK == 2048,
                        "FastLLM Dots3-Note currently expects a 64x128 "
                        "Top-2048 DSA indexer.\n");
        int configuredMaxPositions =
            GetIntConfig(weight, "max_position_embeddings", 524288);
        max_positions = configuredMaxPositions;
        AssertInFastLLM(max_positions >= shortContextLimit,
                        "FastLLM Dots3-Note max positions must cover its sliding window.\n");
        AssertInFastLLM(fullQKRopeHeadDim == swaQKRopeHeadDim,
                        "FastLLM Dots3-Note requires matching full/SWA RoPE dimensions.\n");
        rotary_dim = fullQKRopeHeadDim;

        EnsureRotaryTableCapacity(std::min(shortContextLimit, max_positions));

        // The checkpoint and the Transformers reference both use BF16 activations
        // for the routed experts.  The CLI may still override this explicitly.
        if (!useCustomMoeAtype) {
            moeAtype = DataType::BFLOAT16;
        }

        for (int layer = 1; layer < block_cnt; layer++) {
            for (int expert = -1; expert < num_experts; expert++) {
                std::string base = "model.layers." + std::to_string(layer) + ".mlp.";
                if (expert < 0) {
                    base += "shared_experts.";
                } else {
                    base += "experts." + std::to_string(expert) + ".";
                }

                std::string gate = base + "gate_proj.weight";
                std::string up = base + "up_proj.weight";
                std::string gateUp = base + "gateup_proj.weight";
                std::string down = base + "down_proj.weight";

                weightMergeRules.push_back(WeightMergeRule({
                    WeightMergeRuleSingle({gate, up}, gateUp, "linearSwiglu")
                }));
                if (expert >= 0 || !GetCudaSharedExpert()) {
                    AddSpecialWeight(gateUp, "linearSwiglu", layer);
                    AddSpecialWeight(down, "linearColumn", layer);
                }

                moeLinears.insert(gate);
                moeLinears.insert(up);
                moeLinears.insert(down);
            }
        }
    }

    Dots3NoteModel::AttentionConfig
    Dots3NoteModel::GetAttentionConfig(int layer) const {
        // The checkpoint's layer_types are:
        // 0, 1, 5, 9, ... 45 = full_attention; all other layers = sliding_attention.
        bool full = layer == 0 || layer % 4 == 1;
        if (full) {
            return {fullNumHeads, fullQLoraRank, fullKVLoraRank,
                    fullQKNopeHeadDim, fullQKRopeHeadDim,
                    fullVHeadDim, fullRopeTheta};
        }
        return {swaNumHeads, swaQLoraRank, swaKVLoraRank,
                swaQKNopeHeadDim, swaQKRopeHeadDim,
                swaVHeadDim, swaRopeTheta};
    }

    bool Dots3NoteModel::IsFullAttentionLayer(int layer) const {
        return layer == 0 || layer % 4 == 1;
    }

    void Dots3NoteModel::BuildRotaryTable(float theta, int positions,
                                          Data &sinTable, Data &cosTable) {
        int tableWidth = rotary_dim / 2;
        std::vector<float> sinValues((size_t)positions * tableWidth);
        std::vector<float> cosValues((size_t)positions * tableWidth);
        for (int position = 0; position < positions; position++) {
            for (int i = 0; i < tableWidth; i++) {
                float exponent = (float)(2 * i) / (float)rotary_dim;
                float inverseFrequency = 1.0f / std::pow(theta, exponent);
                float angle = (float)position * inverseFrequency;
                // With DSA enabled Transformers builds RoPE tables from an
                // FP32 rotary input.  Main attention rounds only after the
                // rotation; the indexer Q path stays FP32 through quantization.
                sinValues[position * tableWidth + i] = std::sin(angle);
                cosValues[position * tableWidth + i] = std::cos(angle);
            }
        }
        sinTable.CopyFrom(Data(DataType::FLOAT32,
                               {positions, tableWidth}, sinValues));
        cosTable.CopyFrom(Data(DataType::FLOAT32,
                               {positions, tableWidth}, cosValues));
    }

    void Dots3NoteModel::EnsureRotaryTableCapacity(int positions) {
        AssertInFastLLM(positions > 0 && positions <= max_positions,
                        "FastLLM Dots3-Note RoPE position is out of range.\n");
        if (positions <= rotaryCapacity) {
            return;
        }
        int newCapacity = std::max(1, rotaryCapacity);
        while (newCapacity < positions) {
            newCapacity = std::min(max_positions, newCapacity * 2);
        }
        BuildRotaryTable(fullRopeTheta, newCapacity,
                         fullSinData, fullCosData);
        BuildRotaryTable(swaRopeTheta, newCapacity,
                         swaSinData, swaCosData);
        deviceFullSinData.clear();
        deviceFullCosData.clear();
        deviceSwaSinData.clear();
        deviceSwaCosData.clear();
        rotaryCapacity = newCapacity;
    }

    void Dots3NoteModel::GetRotaryTableForDevice(bool fullAttention,
                                                  const std::string &device,
                                                  Data *&sinTable,
                                                  Data *&cosTable) {
        Data &sourceSin = fullAttention ? fullSinData : swaSinData;
        Data &sourceCos = fullAttention ? fullCosData : swaCosData;
        auto &sinMap = fullAttention ? deviceFullSinData : deviceSwaSinData;
        auto &cosMap = fullAttention ? deviceFullCosData : deviceSwaCosData;
        Data &deviceSin = sinMap[device];
        Data &deviceCos = cosMap[device];
        if (deviceSin.dims.empty()) {
            Mul(sourceSin, 1.0f, deviceSin);
            Mul(sourceCos, 1.0f, deviceCos);
        }
        sinTable = &deviceSin;
        cosTable = &deviceCos;
    }

    void Dots3NoteModel::InitMoeWeightViews() {
        if (!moeWeights.empty()) {
            return;
        }
        moeWeights.resize(block_cnt);
        moeBiass.resize(block_cnt);
        for (int layer = 1; layer < block_cnt; layer++) {
            std::string shared = "model.layers." + std::to_string(layer) +
                                 ".mlp.shared_experts.";
            moeWeights[layer].push_back(&weight[shared + "gateup_proj.weight"]);
            moeWeights[layer].push_back(&weight[shared + "down_proj.weight"]);
            moeBiass[layer].push_back(nullptr);
            moeBiass[layer].push_back(nullptr);
            for (int expert = 0; expert < num_experts; expert++) {
                std::string base = "model.layers." + std::to_string(layer) +
                                   ".mlp.experts." + std::to_string(expert) + ".";
                moeWeights[layer].push_back(&weight[base + "gateup_proj.weight"]);
                moeWeights[layer].push_back(&weight[base + "down_proj.weight"]);
                moeBiass[layer].push_back(nullptr);
                moeBiass[layer].push_back(nullptr);
            }
        }
    }

    std::shared_ptr<Dots3NoteModel::IndexerCacheMemory>
    Dots3NoteModel::GetOrCreateIndexerCache(
            const std::vector<std::pair<Data, Data>> *pastKeyValues) {
        std::lock_guard<std::mutex> guard(indexerCacheMutex);
        auto &cache = indexerCaches[pastKeyValues];
        if (cache == nullptr) {
            cache = std::make_shared<IndexerCacheMemory>();
            cache->layers.resize(block_cnt);
        }
        return cache;
    }

    int Dots3NoteModel::Forward(
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *retLogits) {
        (void)attentionMask;
        AssertInFastLLM(dataType == DataType::BFLOAT16,
                        "FastLLM Dots3-Note correctness path requires BF16 activations. "
                        "Use --atype bfloat16 (auto selects it for this model).\n");
        AssertInFastLLM(kvCacheDataType == DataType::BFLOAT16,
                        "FastLLM Dots3-Note correctness path requires a BF16 KV cache.\n");
        AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
                        "FastLLM Dots3-Note currently supports batch size 1.\n");
        AssertInFastLLM((int)pastKeyValues.size() >= block_cnt,
                        "FastLLM Dots3-Note received an incomplete KV cache.\n");

        int seqlen = inputIds.dims[1];
        int sequencePastLen = pastKeyValues[0].first.dims.size() > 1
                                  ? pastKeyValues[0].first.dims[1]
                                  : 0;
        if (HistoryCacheDebugEnabled()) {
            int slidingPastLen = pastKeyValues.size() > 2 &&
                                 pastKeyValues[2].first.dims.size() > 1
                                     ? pastKeyValues[2].first.dims[1] : 0;
            fprintf(stderr,
                    "[dots-history] forward q=%d full_past=%d sliding_past=%d\n",
                    seqlen, sequencePastLen, slidingPastLen);
        }
        AssertInFastLLM(sequencePastLen + seqlen <= max_positions,
                        "FastLLM Dots3-Note input exceeds the model context window.\n");
        EnsureRotaryTableCapacity(sequencePastLen + seqlen);

        Executor &executor = *((Executor *)GetExecutor());
        InitMoeWeightViews();
        bool rotateLongFullKvCache = false;
        bool parkFreshSlidingCaches = false;
        std::vector<uint8_t> keepFullKvOnCuda;
#ifdef USE_CUDA
        bool fullKvIsCuda = !pastKeyValues.empty() &&
            pastKeyValues[0].first.dataDevice == DataDevice::CUDA &&
            pastKeyValues[0].first.cudaData != nullptr &&
            !pastKeyValues[0].first.multiDeviceData;
        bool fullKvWasStaged = !GetKVCacheInCPU() &&
            !pastKeyValues.empty() &&
            pastKeyValues[0].first.dataDevice == DataDevice::CPU &&
            pastKeyValues[0].first.isKVCache &&
            pastKeyValues[0].first.cpuData != nullptr &&
            !pastKeyValues[0].first.dataDeviceIds.empty() &&
            !pastKeyValues[0].first.multiDeviceData;
        bool longPrefillAppend =
            sequencePastLen >= 8192 && seqlen > 1;
        bool longDecode =
            sequencePastLen >= 16384 && seqlen == 1;
        if ((longPrefillAppend || longDecode) &&
            (fullKvIsCuda || fullKvWasStaged)) {
            std::vector<long long> totalSizes =
                FastllmCudaGetTotalSizes();
            int deviceId = pastKeyValues[0].first.dataDeviceIds.empty()
                ? FastllmCudaGetDevice()
                : pastKeyValues[0].first.dataDeviceIds[0];
            if (deviceId >= 0 && deviceId < (int)totalSizes.size()) {
                rotateLongFullKvCache =
                    totalSizes[deviceId] <=
                    32LL * 1024LL * 1024LL * 1024LL;
            }
            if (EnvFlagEnabled(
                    "FASTLLM_DOTS3_NOTE_KEEP_FULL_KV_GPU")) {
                rotateLongFullKvCache = false;
            }
        }
        if (HistoryCacheDebugEnabled()) {
            fprintf(stderr, "[dots-history] rotate_full_kv=%d\n",
                    rotateLongFullKvCache ? 1 : 0);
        }
        if (sequencePastLen == 0 && seqlen > indexTopK &&
            !GetKVCacheInCPU()) {
            std::vector<long long> totalSizes =
                FastllmCudaGetTotalSizes();
            int deviceId = FastllmCudaGetDevice();
            parkFreshSlidingCaches =
                deviceId >= 0 && deviceId < (int)totalSizes.size() &&
                totalSizes[deviceId] <=
                    32LL * 1024LL * 1024LL * 1024LL;
        }
#endif
        std::shared_ptr<IndexerCacheMemory> indexerCache =
            GetOrCreateIndexerCache(&pastKeyValues);
        if (sequencePastLen == 0) {
            auto fresh = std::make_shared<IndexerCacheMemory>();
            fresh->layers.resize(block_cnt);
            {
                std::lock_guard<std::mutex> guard(indexerCacheMutex);
                indexerCaches[&pastKeyValues] = fresh;
            }
            indexerCache = std::move(fresh);
        }
#ifdef USE_CUDA
        if (rotateLongFullKvCache) {
            keepFullKvOnCuda.assign(block_cnt, 0);
            const long long freeReserveBytes = kLongKvFreeReserveBytes;
            const long long residentBudgetBytes =
                kLongKvResidentBudgetBytes;

            // Keep a small suffix resident across chunks. Its projected size
            // includes the current append, so the budget remains bounded as
            // context grows.
            auto projectedBytesAfterAppend = [&](const Data &cache) {
                uint64_t bytes = cache.expansionBytes;
                if (bytes == 0 || cache.dims.size() <= 1 ||
                    cache.expansionDims.size() <= 1 ||
                    cache.expansionDims[1] <= 0) {
                    return bytes;
                }
                int capacity = cache.expansionDims[1];
                int target = cache.dims[1] + seqlen;
                int appendBlock = ((seqlen - 1) / 128 + 1) * 128;
                int projectedCapacity = capacity;
                while (projectedCapacity < target) {
                    projectedCapacity += appendBlock;
                }
                return bytes * (uint64_t)projectedCapacity /
                       (uint64_t)capacity;
            };
            uint64_t residentBytes = 0;
            int residentLayers = 0;
            for (int layer = block_cnt - 1; layer >= 0; --layer) {
                if (!IsFullAttentionLayer(layer)) {
                    continue;
                }
                uint64_t projectedBytes =
                    projectedBytesAfterAppend(
                        pastKeyValues[layer].first) +
                    projectedBytesAfterAppend(
                        pastKeyValues[layer].second);
                if (projectedBytes <=
                    (uint64_t)std::max(
                        0LL, residentBudgetBytes -
                        (long long)residentBytes)) {
                    keepFullKvOnCuda[layer] = 1;
                    residentBytes += projectedBytes;
                    residentLayers++;
                } else {
                    break;
                }
            }

            // Full-attention KV dominates the 24 GiB footprint. Previously
            // all 13 caches were evicted once the context reached 8K, even
            // though only part of that space is needed by the active layer.
            // Release idle pool blocks first, then spill a suffix only until
            // there is enough headroom for cache growth and attention scratch.
            FastllmCudaClearBigBuffer();
            long long freeBefore = FastllmCudaGetFreeSize();
            uint64_t stagedBytes = 0;
            int stagedLayers = 0;
            {
                ScopedExecutorProfiler kvSpillProfile(
                    "Dots3NoteKvInitialSpill");
                for (int layer = block_cnt - 1;
                     layer >= 0 &&
                     freeBefore + (long long)stagedBytes < freeReserveBytes;
                     --layer) {
                    if (!IsFullAttentionLayer(layer)) {
                        continue;
                    }
                    Data &pastKey = pastKeyValues[layer].first;
                    Data &pastValue = pastKeyValues[layer].second;
                    bool keyOnCuda =
                        pastKey.dataDevice == DataDevice::CUDA &&
                        pastKey.cudaData != nullptr;
                    bool valueOnCuda =
                        pastValue.dataDevice == DataDevice::CUDA &&
                        pastValue.cudaData != nullptr;
                    if (!keyOnCuda && !valueOnCuda) {
                        continue;
                    }
                    if (keyOnCuda) {
                        stagedBytes += pastKey.expansionBytes;
                        pastKey.ToDevice(DataDevice::CPU);
                    }
                    if (valueOnCuda) {
                        stagedBytes += pastValue.expansionBytes;
                        pastValue.ToDevice(DataDevice::CPU);
                    }
                    stagedLayers++;
                }
            }
            FastllmCudaClearBigBuffer();
            if (HistoryCacheDebugEnabled()) {
                fprintf(stderr,
                        "[dots-history] initial_spill layers=%d bytes_mb=%llu "
                        "resident_layers=%d resident_budget_mb=%lld "
                        "reserve_mb=%lld free_before_mb=%lld free_after_mb=%lld\n",
                        stagedLayers,
                        (unsigned long long)(stagedBytes >> 20),
                        residentLayers, residentBudgetBytes >> 20,
                        freeReserveBytes >> 20, freeBefore >> 20,
                        FastllmCudaGetFreeSize() >> 20);
            }
        }
#endif

        Data hiddenStates;
        // MergeMOE reuses these NUMA-side workspaces across layers; keep them
        // outside the loop while releasing the CUDA activation tensors below.
        Data moeInputTemp, moeOutputTemp;
        Data moeW1, moeW2, moeW3, moeCurInput, moeCurOutput;
        bool cudaSharedExpert = GetCudaSharedExpert();
        const bool useCpuLmHead = EnvFlagEnabled(
            "FASTLLM_DOTS3_NOTE_CPU_LM_HEAD");
        const bool prefetchCudaWeights = seqlen > 1 &&
            EnvFlagEnabled("FASTLLM_DOTS3_NOTE_PREFETCH_WEIGHTS");
        const bool profilePrefetch = std::getenv(
            "FASTLLM_PROFILE_DOTS3_NOTE_PREFETCH") != nullptr;
        int lmHeadPrefetchLayer = block_cnt - 2;
        const char *lmHeadPrefetchLayerEnv = std::getenv(
            "FASTLLM_DOTS3_NOTE_PREFETCH_LM_HEAD_LAYER");
        if (lmHeadPrefetchLayerEnv != nullptr &&
            lmHeadPrefetchLayerEnv[0] != '\0') {
            lmHeadPrefetchLayer = std::max(
                1, std::min(block_cnt - 1,
                            std::atoi(lmHeadPrefetchLayerEnv)));
        }

        auto collectCudaPrefetchWeights = [&](int nextLayer) {
            std::vector<Data *> ret;
            auto append = [&](const std::string &name) {
                auto it = weight.weight.find(name);
                if (it != weight.weight.end()) {
                    ret.push_back(&it->second);
                }
            };
            if (nextLayer >= block_cnt) {
                append("model.norm.weight");
                append("lm_head.weight");
                return ret;
            }

            std::string prefix = "model.layers." +
                                 std::to_string(nextLayer);
            std::string attentionPrefix = prefix + ".self_attn.";
            append(prefix + ".input_layernorm.weight");
            append(attentionPrefix + "q_a_proj.weight");
            append(attentionPrefix + "q_a_layernorm.weight");
            append(attentionPrefix + "q_b_proj.weight");
            append(attentionPrefix + "kv_a_proj_with_mqa.weight");
            append(attentionPrefix + "kv_a_layernorm.weight");
            append(attentionPrefix + "k_rope_only_layernorm.weight");
            append(attentionPrefix + "kv_b_proj.weight");
            append(attentionPrefix + "g_proj.weight");
            append(attentionPrefix + "o_proj.weight");
            append(attentionPrefix + "indexer.wq_b.weight");
            append(attentionPrefix + "indexer.wk.weight");
            append(attentionPrefix + "indexer.k_norm.weight");
            append(attentionPrefix + "indexer.k_norm.bias");
            append(attentionPrefix + "indexer.weights_proj.weight");
            append(prefix + ".post_attention_layernorm.weight");
            append(prefix + ".mlp.gate.weight");
            append(prefix + ".mlp.gate.e_score_correction_bias");
            if (cudaSharedExpert) {
                append(prefix + ".mlp.shared_experts.gateup_proj.weight");
                append(prefix + ".mlp.shared_experts.down_proj.weight");
            }
            return ret;
        };

        Embedding(inputIds, weight["model.embed_tokens.weight"], hiddenStates);
        ToDataType(hiddenStates, DataType::BFLOAT16);

        std::thread lmHeadPrefetchThread;
        bool lmHeadPrefetchStarted = false;
        auto lmHeadPrefetchStart = std::chrono::system_clock::now();
        for (int layer = 0; layer < block_cnt; layer++) {
            ApplyDeviceMap(deviceMap, layer + 1, block_cnt);

            // These tensors are consumed entirely within one transformer
            // layer.  Releasing them at the iteration boundary makes their
            // CUDA buffers available to the next layer instead of pinning the
            // peak allocation of every distinct intermediate for all 46
            // layers of a long prefill.
            Data attenInput;
            Data qa, q, qNope, qPe;
            Data indexTopKIndices;
            Data compressedKvAll, compressedKv, kPe, normalizedKv;
            Data kv, kNope, value, kPeRepeat, key;
            Data attentionScores, attentionProbFloat, attentionProb;
            Data slidingAttentionMask;
            int slidingMaskPastLen = -1;
            int slidingMaskKeyLen = -1;
            Data attentionOutput, attentionGate, attentionProjected;
            Data w1, w2, w3;
            Data routerLogits, expertIndex, expertScore;
            Data moeOutput, moeOutputCopy;
            Data sharedGateUp, sharedGate, sharedUp, sharedOutput;
            Data combinedMoe;

            AttentionConfig config = GetAttentionConfig(layer);
            bool fullAttention = IsFullAttentionLayer(layer);
            Data *sinTable = nullptr, *cosTable = nullptr;
            GetRotaryTableForDevice(fullAttention, executor.firstDevice,
                                    sinTable, cosTable);

            std::string prefix = "model.layers." + std::to_string(layer);
            std::string attentionPrefix = prefix + ".self_attn.";

            RMSNorm(hiddenStates, weight[prefix + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            ToDataType(attenInput, DataType::BFLOAT16);

            Linear(attenInput, weight[attentionPrefix + "q_a_proj.weight"],
                   Data(), qa);
            RMSNorm(qa, weight[attentionPrefix + "q_a_layernorm.weight"],
                    rms_norm_eps, qa);
            Mul(qa, std::sqrt((float)embed_dim / (float)config.qLoraRank), qa);

            if (fullAttention) {
#ifdef USE_CUDA
                const bool useCudaIndexer = UsesCudaIndexer(executor);
                if (useCudaIndexer) {
                    // Keep only the compact FP8 tensors alive for cache append
                    // and TopK. Split Q while it is still BF16, then convert
                    // each half independently; this avoids a full 256 MiB FP32
                    // Q tensor at 8K. Ending each stage's scope also lets the
                    // quantizer and TopK reuse the retired buffers.
                    Data indexQFp8, indexFoldedWeights;
                    Data indexKFp8, indexKScales;
                    const std::string indexerPrefix =
                        attentionPrefix + "indexer.";
                    {
                        Data indexQPe, indexQNope;
                        {
                            {
                                Data indexQPeBf16, indexQNopeBf16;
                                {
                                    Data indexQ;
                                    Linear(
                                        qa,
                                        weight[indexerPrefix + "wq_b.weight"],
                                        Data(), indexQ);
                                    indexQ.Reshape(
                                        {1, seqlen, indexHeads,
                                         indexHeadDim});
                                    Split(indexQ, -1, 0,
                                          config.qkRopeHeadDim,
                                          indexQPeBf16);
                                    Split(indexQ, -1,
                                          config.qkRopeHeadDim,
                                          indexHeadDim, indexQNopeBf16);
                                }
                                ToDataType(indexQPeBf16, indexQPe,
                                           DataType::FLOAT32);
                                ToDataType(indexQNopeBf16, indexQNope,
                                           DataType::FLOAT32);
                            }
                            // NearlyRotatePosition2D consumes a
                            // sequence-major tensor.
                            PermuteSelf(indexQPe, {1, 0, 2, 3});
                            NearlyRotatePosition2D(
                                indexQPe, positionIds, *sinTable, *cosTable,
                                config.qkRopeHeadDim);
                            PermuteSelf(indexQPe, {1, 0, 2, 3});
                        }

                        Data indexKPrepared;
                        {
                            Data indexK, indexKFloat, indexKNorm;
                            Data indexKPe, indexKNope;
                            Linear(attenInput,
                                   weight[indexerPrefix + "wk.weight"],
                                   Data(), indexK);
                            ToDataType(indexK, indexKFloat,
                                       DataType::FLOAT32);
                            LayerNorm(
                                indexKFloat,
                                weight[indexerPrefix + "k_norm.weight"],
                                weight[indexerPrefix + "k_norm.bias"], -1,
                                indexKNorm);
                            indexKNorm.Reshape(
                                {1, seqlen, 1, indexHeadDim});
                            Split(indexKNorm, -1, 0,
                                  config.qkRopeHeadDim, indexKPe);
                            Split(indexKNorm, -1,
                                  config.qkRopeHeadDim, indexHeadDim,
                                  indexKNope);
                            PermuteSelf(indexKPe, {1, 0, 2, 3});
                            NearlyRotatePosition2D(
                                indexKPe, positionIds, *sinTable, *cosTable,
                                config.qkRopeHeadDim);
                            PermuteSelf(indexKPe, {1, 0, 2, 3});
                            // The fused reference rounds K once after FP32
                            // LayerNorm and RoPE, immediately before the
                            // dynamic E4M3 quantizer.
                            AssertInFastLLM(
                                FastllmCudaDots3NotePackIndexerKey(
                                    indexKPe, indexKNope,
                                    indexKPrepared),
                                "FastLLM Dots3-Note failed to pack the index K tensor.\n");
                        }

                        Data indexWeights;
                        Linear(
                            attenInput,
                            weight[indexerPrefix + "weights_proj.weight"],
                            Data(), indexWeights);
                        indexWeights.Reshape(
                            {1, seqlen, indexHeads});
                        AssertInFastLLM(
                            FastllmCudaDots3NoteQuantizeIndexer(
                                indexQPe, indexQNope, indexKPrepared,
                                indexWeights, indexQFp8,
                                indexFoldedWeights, indexKFp8,
                                indexKScales),
                            "FastLLM Dots3-Note failed to quantize the DSA indexer.\n");
                    }

                    IndexerLayerCache &cache = indexerCache->layers[layer];
                    int cachedLength = cache.keys.dims.size() == 3
                                           ? cache.keys.dims[1] : 0;
                    AssertInFastLLM(
                        cachedLength == sequencePastLen,
                        "FastLLM Dots3-Note index-key cache is out of sync.\n");
                    auto appendIndexer = [&](Data &target,
                                             const Data &current) {
                        target.ToDevice(current.dataDevice);
                        const int cacheBlock = 128;
                        while ((target.dims.empty() &&
                                (target.expansionDims.empty() ||
                                 current.dims[1] >
                                     target.expansionDims[1])) ||
                               (!target.dims.empty() &&
                                target.dims[1] + current.dims[1] >
                                    target.expansionDims[1])) {
                            std::vector<int> dims;
                            if (target.dims.empty() || target.Count(0) == 0) {
                                dims = current.dims;
                                dims[1] =
                                    ((current.dims[1] - 1) / cacheBlock + 1) *
                                    cacheBlock;
                            } else {
                                dims = target.dims;
                                dims[1] +=
                                    ((current.dims[1] - 1) / cacheBlock + 1) *
                                    cacheBlock;
                            }
                            target.Expansion(dims);
                        }
                        CatDirect(target, current, 1);
                        target.SetKVCache();
                    };
                    appendIndexer(cache.keys, indexKFp8);
                    appendIndexer(cache.scales, indexKScales);
                    if (sequencePastLen + seqlen > indexTopK) {
                        ScopedExecutorProfiler indexerProfile(
                            "Dots3NoteIndexerTopK");
                        AssertInFastLLM(
                            FastllmCudaDots3NoteIndexerTopK(
                                indexQFp8, indexFoldedWeights,
                                cache.keys, cache.scales,
                                sequencePastLen, indexTopK,
                                indexTopKIndices),
                            "FastLLM Dots3-Note DSA Top-2048 failed.\n");
                    }
                } else {
                    AssertInFastLLM(
                        sequencePastLen + seqlen <= indexTopK,
                        "Dots3-Note contexts above 2048 tokens require "
                        "a CUDA DSA indexer.\n");
                }
#else
                AssertInFastLLM(sequencePastLen + seqlen <= indexTopK,
                                "Dots3-Note contexts above 2048 tokens "
                                "require a CUDA DSA indexer.\n");
#endif
            }

            Data &pastKey = pastKeyValues[layer].first;
            Data &pastValue = pastKeyValues[layer].second;
            int layerPastLen = pastKey.dims.size() > 1
                                   ? pastKey.dims[1] : 0;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                bool needsKvStageIn = rotateLongFullKvCache &&
                    fullAttention &&
                    (pastKey.dataDevice != attenInput.dataDevice ||
                     pastValue.dataDevice != attenInput.dataDevice);
                if (needsKvStageIn) {
                    ScopedExecutorProfiler kvStageInProfile(
                        "Dots3NoteKvStageIn");
                    pastKey.ToDevice(attenInput.dataDevice);
                    pastValue.ToDevice(attenInput.dataDevice);
                } else {
                    pastKey.ToDevice(attenInput.dataDevice);
                    pastValue.ToDevice(attenInput.dataDevice);
                }
            }

            bool directFreshFullPrefill = false;
#ifdef USE_CUDA
            directFreshFullPrefill =
                fullAttention && !GetKVCacheInCPU() &&
                layerPastLen == 0 && seqlen >= 16 &&
                attenInput.dataDevice == DataDevice::CUDA;
#endif

            auto prepareFreshPooledCache = [](
                    Data &cache, const std::vector<int> &dims,
                    const std::vector<int> &capacity) {
                cache.FreeSpace();
                cache.directMemory = false;
                cache.expansionDims.clear();
                cache.Resize(capacity);
                cache.Allocate(false);
                cache.expansionDims = capacity;
                cache.Resize(dims);
            };

            // KV-A is small enough to keep while Q is built. Once both low-rank
            // projections have consumed attenInput, retire that 80 MiB 8K
            // activation at the Q allocation high-watermark. The identical
            // input RMSNorm is recomputed below immediately before g_proj.
            Linear(attenInput,
                   weight[attentionPrefix + "kv_a_proj_with_mqa.weight"],
                   Data(), compressedKvAll);
            Split(compressedKvAll, -1, 0, config.kvLoraRank, compressedKv);
            Split(compressedKvAll, -1, config.kvLoraRank,
                  config.kvLoraRank + config.qkRopeHeadDim, kPe);
            compressedKvAll.FreeSpace();
            bool recomputeAttentionInput =
                sequencePastLen == 0 && seqlen > indexTopK;
            if (recomputeAttentionInput) {
                attenInput.FreeSpace();
            }

            Linear(qa, weight[attentionPrefix + "q_b_proj.weight"], Data(), q);
            qa.FreeSpace();

            q.Reshape({1, seqlen, config.numHeads,
                       config.qkNopeHeadDim + config.qkRopeHeadDim});
            PermuteSelf(q, {0, 2, 1, 3});
            Split(q, -1, 0, config.qkNopeHeadDim, qNope);
            Split(q, -1, config.qkNopeHeadDim,
                  config.qkNopeHeadDim + config.qkRopeHeadDim, qPe);

            PermuteSelf(qPe, {0, 2, 1, 3});
            PermuteSelf(qPe, {1, 0, 2, 3});
            NearlyRotatePosition2D(qPe, positionIds, *sinTable, *cosTable,
                                   config.qkRopeHeadDim);
            PermuteSelf(qPe, {1, 0, 2, 3});
            PermuteSelf(qPe, {0, 2, 1, 3});

            Cat(qNope, qPe, -1, q);
            qNope.FreeSpace();
            qPe.FreeSpace();

            RMSNorm(compressedKv,
                    weight[attentionPrefix + "kv_a_layernorm.weight"],
                    rms_norm_eps, normalizedKv);
            compressedKv.FreeSpace();
            Mul(normalizedKv,
                std::sqrt((float)embed_dim / (float)config.kvLoraRank),
                normalizedKv);
            RMSNorm(kPe,
                    weight[attentionPrefix + "k_rope_only_layernorm.weight"],
                    rms_norm_eps, kPe);
            Linear(normalizedKv, weight[attentionPrefix + "kv_b_proj.weight"],
                   Data(), kv);
            normalizedKv.FreeSpace();
            kv.Reshape({1, seqlen, config.numHeads,
                        config.qkNopeHeadDim + config.vHeadDim});
            PermuteSelf(kv, {0, 2, 1, 3});
            kv.Reshape({config.numHeads, seqlen,
                        config.qkNopeHeadDim + config.vHeadDim});
            Data &currentValue = directFreshFullPrefill
                                     ? pastValue : value;
            if (directFreshFullPrefill) {
                int capacity = ((seqlen - 1) / 128 + 1) * 128;
                prepareFreshPooledCache(
                    pastValue,
                    {config.numHeads, seqlen, config.vHeadDim},
                    {config.numHeads, capacity, config.vHeadDim});
            } else {
                Split(kv, -1, 0, config.qkNopeHeadDim, kNope);
            }
            Split(kv, -1, config.qkNopeHeadDim,
                  config.qkNopeHeadDim + config.vHeadDim, currentValue);
            if (!directFreshFullPrefill) {
                kv.FreeSpace();
            }

            kPe.Reshape({1, seqlen, 1, config.qkRopeHeadDim});
            PermuteSelf(kPe, {1, 0, 2, 3});
            NearlyRotatePosition2D(kPe, positionIds, *sinTable, *cosTable,
                                   config.qkRopeHeadDim);
            PermuteSelf(kPe, {1, 0, 2, 3});
            PermuteSelf(kPe, {0, 2, 1, 3});

            Data &currentKey = directFreshFullPrefill ? pastKey : key;
            bool reuseKvForSparseAttention =
                directFreshFullPrefill && seqlen > indexTopK;
            if (directFreshFullPrefill) {
                int capacity = ((seqlen - 1) / 128 + 1) * 128;
                prepareFreshPooledCache(
                    pastKey,
                    {config.numHeads, seqlen,
                     config.qkNopeHeadDim + config.qkRopeHeadDim},
                    {config.numHeads, capacity,
                     config.qkNopeHeadDim + config.qkRopeHeadDim});
                kPe.Reshape({seqlen, config.qkRopeHeadDim});
#ifdef USE_CUDA
                AssertInFastLLM(
                    FastllmCudaDots3NotePackAttentionKey(
                        kv, kPe, config.qkNopeHeadDim, pastKey),
                    "FastLLM Dots3-Note failed to pack the attention K tensor.\n");
#endif
                if (!reuseKvForSparseAttention) {
                    kv.FreeSpace();
                }
            } else {
                Repeat(kPe, 1, config.numHeads, kPeRepeat);
                kPeRepeat.Reshape(
                    {config.numHeads, seqlen,
                     config.qkRopeHeadDim});
                Cat(kNope, kPeRepeat, -1, currentKey);
            }
            kPe.FreeSpace();
            kNope.FreeSpace();
            kPeRepeat.FreeSpace();

            int qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim;
            q.Reshape({config.numHeads, seqlen, qHeadDim});
            if (directFreshFullPrefill) {
                pastKey.SetKVCache();
                pastValue.SetKVCache();
            }

            const int cacheBlock = 128;
            bool directFreshSlidingPrefill = false;
#ifdef USE_CUDA
            directFreshSlidingPrefill =
                !fullAttention && !GetKVCacheInCPU() &&
                layerPastLen == 0 && seqlen >= 16 &&
                seqlen > shortContextLimit &&
                q.dataDevice == DataDevice::CUDA &&
                key.dataDevice == DataDevice::CUDA &&
                value.dataDevice == DataDevice::CUDA;
#endif
            if (!directFreshSlidingPrefill &&
                !directFreshFullPrefill) {
                while ((pastKey.dims.empty() &&
                        (pastKey.expansionDims.empty() ||
                         key.dims[1] > pastKey.expansionDims[1])) ||
                       (!pastKey.dims.empty() &&
                        pastKey.dims[1] + key.dims[1] >
                            pastKey.expansionDims[1])) {
                    std::vector<int> dims;
                    if (pastKey.dims.empty() || pastKey.Count(0) == 0) {
                        dims = {
                            key.dims[0],
                            ((key.dims[1] - 1) / cacheBlock + 1) *
                                cacheBlock,
                            key.dims[2]};
                    } else {
                        dims = pastKey.dims;
                        dims[1] +=
                            ((key.dims[1] - 1) / cacheBlock + 1) *
                            cacheBlock;
                    }
                    pastKey.Expansion(dims);
                }
                while ((pastValue.dims.empty() &&
                        (pastValue.expansionDims.empty() ||
                         value.dims[1] > pastValue.expansionDims[1])) ||
                       (!pastValue.dims.empty() &&
                        pastValue.dims[1] + value.dims[1] >
                            pastValue.expansionDims[1])) {
                    std::vector<int> dims;
                    if (pastValue.dims.empty() ||
                        pastValue.Count(0) == 0) {
                        dims = {
                            value.dims[0],
                            ((value.dims[1] - 1) / cacheBlock + 1) *
                                cacheBlock,
                            value.dims[2]};
                    } else {
                        dims = pastValue.dims;
                        dims[1] +=
                            ((value.dims[1] - 1) / cacheBlock + 1) *
                            cacheBlock;
                    }
                    pastValue.Expansion(dims);
                }
                CatDirect(pastKey, key, 1);
                CatDirect(pastValue, value, 1);
            }

            int keyLen = directFreshSlidingPrefill
                             ? key.dims[1] : pastKey.dims[1];
            bool usedSlidingPrefill = false;
#ifdef USE_CUDA
            if (directFreshSlidingPrefill) {
                ScopedExecutorProfiler slidingProfile(
                    "Dots3NoteSlidingAttention");
                usedSlidingPrefill =
                    FastllmCudaDots3NoteSlidingAttentionPrefill(
                        q, key, value, 0, shortContextLimit,
                        1.0f / std::sqrt((float)qHeadDim),
                        attentionOutput);
                AssertInFastLLM(
                    usedSlidingPrefill,
                    "FastLLM Dots3-Note sliding attention failed.\n");
            } else if (!fullAttention && seqlen >= 16 &&
                keyLen > shortContextLimit &&
                q.dataDevice == DataDevice::CUDA) {
                ScopedExecutorProfiler slidingProfile(
                    "Dots3NoteSlidingAttention");
                usedSlidingPrefill =
                    FastllmCudaDots3NoteSlidingAttentionPrefill(
                        q, pastKey, pastValue, layerPastLen,
                        shortContextLimit,
                        1.0f / std::sqrt((float)qHeadDim),
                        attentionOutput);
                AssertInFastLLM(
                    usedSlidingPrefill,
                    "FastLLM Dots3-Note sliding attention failed.\n");
            }
#endif
            if (!usedSlidingPrefill && fullAttention && keyLen > indexTopK) {
#ifdef USE_CUDA
                ScopedExecutorProfiler sparseProfile(
                    "Dots3NoteSparseAttention");
                // Small decode-shaped calls remain on the row kernel; bounded
                // prefill blocks amortize the dense Tensor Core QK/AV path.
                bool useCublasPrefill =
                    seqlen >= 16 && keyLen <= kSparsePrefillMaxKeys &&
                    !EnvFlagEnabled(
                        "FASTLLM_DOTS3_NOTE_DISABLE_CUBLAS_SPARSE_PREFILL");
                void *sparseScratch = nullptr;
                size_t sparseScratchBytes = 0;
                if (reuseKvForSparseAttention) {
                    size_t attentionOutputBytes =
                        (size_t)config.numHeads * seqlen *
                        config.vHeadDim * sizeof(uint16_t);
                    size_t minimumScratchBytes =
                        (size_t)config.numHeads * keyLen *
                        sizeof(uint16_t);
                    AssertInFastLLM(
                        kv.expansionBytes >=
                            attentionOutputBytes + minimumScratchBytes,
                        "FastLLM Dots3-Note KV projection is too small to "
                        "back sparse attention.\n");
                    attentionOutput.FakeFrom(kv, 0);
                    attentionOutput.Resize(
                        {config.numHeads, seqlen, config.vHeadDim});
                    sparseScratch =
                        (uint8_t *)kv.cudaData + attentionOutputBytes;
                    sparseScratchBytes =
                        kv.expansionBytes - attentionOutputBytes;
                }
                bool sparseOk = useCublasPrefill &&
                    FastllmCudaDots3NoteSparseAttentionPrefill(
                        q, pastKey, pastValue, indexTopKIndices,
                        layerPastLen,
                        1.0f / std::sqrt((float)qHeadDim),
                        attentionOutput, sparseScratch,
                        sparseScratchBytes);
                if (!sparseOk) {
                    sparseOk = FastllmCudaDots3NoteSparseAttention(
                        q, pastKey, pastValue, indexTopKIndices,
                        layerPastLen,
                        1.0f / std::sqrt((float)qHeadDim),
                        attentionOutput);
                }
                AssertInFastLLM(sparseOk,
                                "FastLLM Dots3-Note sparse MLA failed.\n");
#else
                ErrorInFastLLM(
                    "Dots3-Note sparse MLA requires a CUDA build.\n");
#endif
            } else if (!usedSlidingPrefill) {
                MatMulTransB(q, pastKey, attentionScores);
                Mul(attentionScores, 1.0f / std::sqrt((float)qHeadDim),
                    attentionScores);
                CausalMask(attentionScores, layerPastLen + 1, -10000.0f);
                ToDataType(attentionScores, attentionProbFloat,
                           DataType::FLOAT32);
                if (!fullAttention && keyLen > shortContextLimit) {
                    if (slidingMaskPastLen != layerPastLen ||
                        slidingMaskKeyLen != keyLen) {
                        std::vector<float> mask(
                            (size_t)seqlen * keyLen, 0.0f);
                        for (int query = 0; query < seqlen; query++) {
                            int absoluteQuery = layerPastLen + query;
                            int firstVisible = std::max(
                                0, absoluteQuery - shortContextLimit + 1);
                            for (int keyIndex = 0;
                                 keyIndex < firstVisible; keyIndex++) {
                                mask[(size_t)query * keyLen + keyIndex] = 1.0f;
                            }
                        }
                        slidingAttentionMask.CopyFrom(Data(
                            DataType::FLOAT32, {1, seqlen, keyLen}, mask));
                        slidingMaskPastLen = layerPastLen;
                        slidingMaskKeyLen = keyLen;
                    }
                    attentionProbFloat.Reshape(
                        {1, config.numHeads, seqlen, keyLen});
                    AttentionMask(attentionProbFloat, slidingAttentionMask,
                                  -10000.0f);
                    attentionProbFloat.Reshape(
                        {config.numHeads, seqlen, keyLen});
                }
                Softmax(attentionProbFloat, attentionProbFloat, -1);
                ToDataType(attentionProbFloat, attentionProb,
                           DataType::BFLOAT16);
                MatMul(attentionProb, pastValue, attentionOutput);
            }

            if (rotateLongFullKvCache && fullAttention &&
                !keepFullKvOnCuda[layer]) {
                // The blocking D2H copies also order after the attention
                // kernels that consumed these buffers. Release the active
                // CUDA storage only after its host copy is complete.
                {
                    ScopedExecutorProfiler kvStageOutProfile(
                        "Dots3NoteKvStageOut");
                    pastKey.ToDevice(DataDevice::CPU);
                    pastValue.ToDevice(DataDevice::CPU);
                }
            }

            if (!fullAttention) {
                // A query with a window of N needs only the previous N - 1
                // cached tokens; the current token supplies the final slot.
                // Keep one append slot instead of a whole 128-token block so
                // every sliding layer has a bounded persistent footprint.
                int slidingCacheTokens = std::max(
                    0, shortContextLimit - 1);
                int slidingCacheCapacity = std::max(
                    1, shortContextLimit);
                if (directFreshSlidingPrefill) {
                    int cachedTokens = key.dims[1];
                    int start = std::max(
                        0, cachedTokens - slidingCacheTokens);
                    Data compactKey, compactValue;
                    Split(key, 1, start, cachedTokens, compactKey);
                    Split(value, 1, start, cachedTokens, compactValue);
                    std::vector<int> keyCapacity = compactKey.dims;
                    std::vector<int> valueCapacity = compactValue.dims;
                    keyCapacity[1] = slidingCacheCapacity;
                    valueCapacity[1] = slidingCacheCapacity;
                    if (pastKey.expansionDims.empty() ||
                        pastKey.expansionDims[1] < slidingCacheCapacity) {
                        pastKey.Expansion(keyCapacity);
                    }
                    if (pastValue.expansionDims.empty() ||
                        pastValue.expansionDims[1] < slidingCacheCapacity) {
                        pastValue.Expansion(valueCapacity);
                    }
                    CatDirect(pastKey, compactKey, 1);
                    CatDirect(pastValue, compactValue, 1);
                    pastKey.SetKVCache();
                    pastValue.SetKVCache();
                } else {
                    int cachedTokens = pastKey.dims[1];
                    if (cachedTokens > slidingCacheTokens) {
                        int start = cachedTokens - slidingCacheTokens;
                        Data compactKey, compactValue;
                        Split(pastKey, 1, start, cachedTokens, compactKey);
                        Split(pastValue, 1, start, cachedTokens,
                              compactValue);
                        pastKey.CopyFrom(compactKey);
                        pastValue.CopyFrom(compactValue);
                        // CopyFrom intentionally drops the source's spare
                        // capacity. CatDirect requires expansionDims to
                        // describe writable storage, otherwise the first token
                        // appended after compaction writes past the compact
                        // allocation. Recreate one token of headroom.
                        std::vector<int> keyCapacity = pastKey.dims;
                        std::vector<int> valueCapacity = pastValue.dims;
                        keyCapacity[1] = slidingCacheCapacity;
                        valueCapacity[1] = slidingCacheCapacity;
                        pastKey.Expansion(keyCapacity);
                        pastValue.Expansion(valueCapacity);
                        pastKey.SetKVCache();
                        pastValue.SetKVCache();
                    }
                }
            }

            if (parkFreshSlidingCaches && (layer == 2 || layer == 3)) {
                // On 24/32 GiB cards, the final long-prefill layers otherwise
                // miss a 384 MiB Q allocation by only a few MiB after CUDA
                // shared-expert workspaces have warmed up. Parking two of the
                // earliest already-bounded SWA caches moves only about 80 MiB
                // (instead of all 7.5 GiB of full-attention KV at 8K). They
                // are staged back automatically if this request continues
                // into decode.
                pastKey.ToDevice(DataDevice::CPU);
                pastValue.ToDevice(DataDevice::CPU);
            }

            // Attention has consumed Q/K/V and any dense probability
            // intermediates. Sliding layers have also compacted their cache by
            // this point, so these buffers can back the gate/projection/MoE
            // stages that follow.
            q.FreeSpace();
            key.FreeSpace();
            value.FreeSpace();
            indexTopKIndices.FreeSpace();
            attentionScores.FreeSpace();
            attentionProbFloat.FreeSpace();
            attentionProb.FreeSpace();
            slidingAttentionMask.FreeSpace();

            attentionOutput.Reshape(
                {1, config.numHeads, seqlen, config.vHeadDim});
            PermuteSelf(attentionOutput, {0, 2, 1, 3});

            if (recomputeAttentionInput) {
                RMSNorm(
                    hiddenStates,
                    weight[prefix + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
                ToDataType(attenInput, DataType::BFLOAT16);
            }
            Linear(attenInput, weight[attentionPrefix + "g_proj.weight"],
                   Data(), attentionGate);
            Sigmoid(attentionGate, attentionGate);
            attentionGate.Reshape({1, seqlen, config.numHeads, 1});
            MulTo(attentionOutput, attentionGate);
            attentionOutput.Reshape(
                {1, seqlen, config.numHeads * config.vHeadDim});
            Linear(attentionOutput, weight[attentionPrefix + "o_proj.weight"],
                   Data(), attentionProjected);
            AddTo(hiddenStates, attentionProjected);
            attentionOutput.FreeSpace();
            attentionGate.FreeSpace();
            attentionProjected.FreeSpace();
            kv.FreeSpace();

            RMSNorm(hiddenStates,
                    weight[prefix + ".post_attention_layernorm.weight"],
                    rms_norm_eps, attenInput);
            ToDataType(attenInput, DataType::BFLOAT16);

            if (layer == 0) {
                Linear(attenInput, weight[prefix + ".mlp.gate_proj.weight"],
                       Data(), w1);
                Silu(w1, w1);
                Linear(attenInput, weight[prefix + ".mlp.up_proj.weight"],
                       Data(), w3);
                MulTo(w1, w3);
                Linear(w1, weight[prefix + ".mlp.down_proj.weight"],
                       Data(), w2);
                AddTo(hiddenStates, w2);
            } else {
                int tokens = attenInput.Count(0) / attenInput.dims.back();
                attenInput.Reshape({tokens, embed_dim});
                std::string gateName = prefix + ".mlp.gate.weight";
                std::string gateBiasName =
                    prefix + ".mlp.gate.e_score_correction_bias";

                Linear(attenInput, weight[gateName], Data(), routerLogits);
                Sigmoid(routerLogits, routerLogits);
                ToDataType(routerLogits, DataType::FLOAT32);
                Data *gateBias = weight.weight.find(gateBiasName) !=
                                         weight.weight.end()
                                     ? &weight[gateBiasName]
                                     : nullptr;
                SelectExpert(routerLogits, expertIndex, expertScore,
                             num_experts_per_tok, true,
                             routed_scaling_factor, gateBias);

                std::thread weightPrefetchThread;
                bool weightPrefetchStarted = false;
                size_t prefetchBytes = 0;

                if (cudaSharedExpert) {
                    std::string sharedPrefix =
                        prefix + ".mlp.shared_experts.";
                    bool sharedOutputReady = false;
                    if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                        // Fuse the gate/up projection with SwiGLU so long
                        // prefills keep only the half-width activated result.
                        // At 8K this avoids a transient 512 MiB BF16 tensor.
                        LinearEx(
                            attenInput,
                            weight[sharedPrefix + "gateup_proj.weight"],
                            Data(), sharedGate,
                            LinearExType::ExSwiglu);
                    } else if (rotateLongFullKvCache && tokens > 2048) {
                        // Appending an 8K chunk to a long cache can leave less
                        // than the 512 MiB needed by the full-width gate/up
                        // projection. Compute independent token rows in 2K
                        // slices and write each down-projection directly into
                        // its place in the final shared-expert output.
                        const int sharedChunkTokens = 2048;
                        sharedOutput.dataType = attenInput.dataType;
                        sharedOutput.UpdateUnitSize();
                        sharedOutput.dataDevice = attenInput.dataDevice;
                        sharedOutput.dataDeviceIds =
                            attenInput.dataDeviceIds;
                        sharedOutput.Resize({tokens, embed_dim});
                        sharedOutput.Allocate(false);
                        for (int start = 0; start < tokens;
                             start += sharedChunkTokens) {
                            int end = std::min(
                                tokens, start + sharedChunkTokens);
                            int chunkTokens = end - start;
                            Data chunkInput, chunkGateUp, chunkGate,
                                 chunkUp, chunkOutput;
                            chunkInput.Resize(
                                {chunkTokens, embed_dim});
                            chunkInput.FakeFrom(
                                attenInput,
                                (size_t)start * attenInput.strides[0] *
                                    attenInput.unitSize);
                            chunkInput.dataDeviceIds =
                                attenInput.dataDeviceIds;
                            Linear(
                                chunkInput,
                                weight[sharedPrefix +
                                       "gateup_proj.weight"],
                                Data(), chunkGateUp);
                            int sharedIntermediate =
                                chunkGateUp.dims.back() / 2;
                            Split(chunkGateUp, -1, 0,
                                  sharedIntermediate, chunkGate);
                            Split(chunkGateUp, -1,
                                  sharedIntermediate,
                                  sharedIntermediate * 2, chunkUp);
                            chunkGateUp.FreeSpace();
                            Silu(chunkGate, chunkGate);
                            MulTo(chunkGate, chunkUp);
                            chunkUp.FreeSpace();
                            chunkOutput.Resize(
                                {chunkTokens, embed_dim});
                            chunkOutput.FakeFrom(
                                sharedOutput,
                                (size_t)start *
                                    sharedOutput.strides[0] *
                                    sharedOutput.unitSize);
                            chunkOutput.dataDeviceIds =
                                sharedOutput.dataDeviceIds;
                            Linear(
                                chunkGate,
                                weight[sharedPrefix +
                                       "down_proj.weight"],
                                Data(), chunkOutput);
                        }
                        sharedOutputReady = true;
                    } else {
                        Linear(attenInput,
                               weight[sharedPrefix + "gateup_proj.weight"],
                               Data(), sharedGateUp);
                        int sharedIntermediate =
                            sharedGateUp.dims.back() / 2;
                        Split(sharedGateUp, -1, 0, sharedIntermediate,
                              sharedGate);
                        Split(sharedGateUp, -1, sharedIntermediate,
                              sharedIntermediate * 2, sharedUp);
                        Silu(sharedGate, sharedGate);
                        MulTo(sharedGate, sharedUp);
                    }
                    if (!sharedOutputReady) {
                        Linear(sharedGate,
                               weight[sharedPrefix + "down_proj.weight"],
                               Data(), sharedOutput);
                    }
                    moeWeights[layer][0] = nullptr;
                    moeWeights[layer][1] = nullptr;
                }

                auto prefetchStart = std::chrono::system_clock::now();
                if (prefetchCudaWeights &&
                    layer + 1 < block_cnt &&
                    attenInput.dataDevice == DataDevice::CUDA) {
                    std::vector<Data *> prefetchWeights =
                        collectCudaPrefetchWeights(layer + 1);
                    for (Data *data : prefetchWeights) {
                        if (data->dataDevice != DataDevice::CUDA) {
                            prefetchBytes += data->GetBytes();
                        }
                    }
                    std::vector<int> prefetchDeviceIds =
                        attenInput.dataDeviceIds;
                    weightPrefetchThread = std::thread(
                        [prefetchWeights = std::move(prefetchWeights),
                         prefetchDeviceIds = std::move(prefetchDeviceIds)]() {
                            for (Data *data : prefetchWeights) {
                                data->ToDevice(DataDevice::CUDA,
                                               prefetchDeviceIds);
                            }
                        });
                    weightPrefetchStarted = true;
                }

                ApplyMoeDeviceMapForLayer(layer);
                AssertInFastLLM(CanRunMergeMOE(attenInput,
                                               moeWeights[layer],
                                               moeBiass[layer]),
                                "FastLLM Dots3-Note requires MergeMOE "
                                "support on the selected MoE device.\n");
                MergeMOEBlock(&attenInput, &expertIndex, &expertScore,
                              &moeWeights[layer], &moeBiass[layer],
                              &moeW1, &moeW2, &moeW3,
                              &moeCurInput, &moeCurOutput,
                              1.0f, &moeOutput, layer,
                              attenInput.dataType, moeAtype,
                              &moeInputTemp, &moeOutputTemp,
                              MoeGateSwiglu, false, 0.0f, false,
                              nullptr);
                auto prefetchJoinStart = std::chrono::system_clock::now();
                if (weightPrefetchThread.joinable()) {
                    weightPrefetchThread.join();
                }
                if (profilePrefetch && weightPrefetchStarted) {
                    auto prefetchEnd = std::chrono::system_clock::now();
                    printf("[fastllm-profile-dots3-prefetch] layer=%d bytes=%zu elapsed_ms=%.3f join_ms=%.3f\n",
                           layer, prefetchBytes,
                           GetSpan(prefetchStart, prefetchEnd) * 1000.0,
                           GetSpan(prefetchJoinStart, prefetchEnd) * 1000.0);
                    fflush(stdout);
                }
                if (prefetchCudaWeights && !useCpuLmHead &&
                    layer == lmHeadPrefetchLayer) {
                    std::vector<Data *> finalWeights =
                        collectCudaPrefetchWeights(block_cnt);
                    std::vector<int> prefetchDeviceIds =
                        moeOutput.dataDeviceIds;
                    if (prefetchDeviceIds.empty()) {
                        prefetchDeviceIds = attenInput.dataDeviceIds;
                    }
                    lmHeadPrefetchStart = std::chrono::system_clock::now();
                    lmHeadPrefetchThread = std::thread(
                        [finalWeights = std::move(finalWeights),
                         prefetchDeviceIds = std::move(prefetchDeviceIds)]() {
                            for (Data *data : finalWeights) {
                                data->ToDevice(DataDevice::CUDA,
                                               prefetchDeviceIds);
                            }
                        });
                    lmHeadPrefetchStarted = true;
                }
                moeOutput.Reshape(hiddenStates.dims);
                moeOutputCopy.CopyFrom(moeOutput);
                ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
                if (cudaSharedExpert) {
                    sharedOutput.Reshape(hiddenStates.dims);
                    combinedMoe.CopyFrom(sharedOutput);
                    AddTo(combinedMoe, moeOutputCopy);
                } else {
                    combinedMoe.CopyFrom(moeOutputCopy);
                }
                AddTo(hiddenStates, combinedMoe);
            }
        }

        auto lmHeadPrefetchJoinStart = std::chrono::system_clock::now();
        if (lmHeadPrefetchThread.joinable()) {
            lmHeadPrefetchThread.join();
        }
        if (profilePrefetch && lmHeadPrefetchStarted) {
            auto lmHeadPrefetchEnd = std::chrono::system_clock::now();
            printf("[fastllm-profile-dots3-prefetch-head] layer=%d elapsed_ms=%.3f join_ms=%.3f\n",
                   lmHeadPrefetchLayer,
                   GetSpan(lmHeadPrefetchStart,
                           lmHeadPrefetchEnd) * 1000.0,
                   GetSpan(lmHeadPrefetchJoinStart,
                           lmHeadPrefetchEnd) * 1000.0);
            fflush(stdout);
        }

        MaybeRecordPromptHistoryCache(pastKeyValues);

        Data lastHiddenStates, logits, topk;
        if (seqlen > 1) {
            Split(hiddenStates, 1, seqlen - 1, seqlen, lastHiddenStates);
        } else {
            lastHiddenStates.CopyFrom(hiddenStates);
        }
        bool runCpuLmHead = useCpuLmHead;
#ifdef USE_CUDA
        Data &lmHeadWeight = weight["lm_head.weight"];
        if (!runCpuLmHead &&
            lastHiddenStates.dataDevice == DataDevice::CUDA &&
            lmHeadWeight.dataDevice != DataDevice::CUDA) {
            // Loading the 1.45 GiB vocabulary projection after a long prefill
            // must not turn an otherwise valid KV cache into an OOM. Keep the
            // warmed attention workspaces in the pool for the next request,
            // and leave the projection on CPU when genuinely free device
            // memory cannot hold it with a safety margin.
            const long long freeBytes = FastllmCudaGetFreeSize();
            const long long requiredBytes =
                (long long)lmHeadWeight.GetBytes() +
                256LL * 1024LL * 1024LL;
            runCpuLmHead = freeBytes > 0 && freeBytes < requiredBytes;
        }
#endif
        if (runCpuLmHead) {
            executor.SetFirstDevice("cpu");
        }
        RMSNorm(lastHiddenStates, weight["model.norm.weight"],
                rms_norm_eps, lastHiddenStates);
        Linear(lastHiddenStates, weight["lm_head.weight"], Data(), logits);
        ToDataType(logits, DataType::FLOAT32);

        if (generationConfig.output_logits && retLogits != nullptr) {
            int size = logits.dims.back();
            logits.ToDevice(DataDevice::CPU);
            retLogits->resize(size);
            memcpy(retLogits->data(),
                   ((float *)logits.cpuData) +
                       (logits.Count(0) / size - 1) * size,
                   size * sizeof(float));
        }

        ResetLogitsOfEOS(1, &logits, pastKeyValues, generationConfig);
        if (generationConfig.IsSimpleGreedy()) {
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            int token = (int)(((float *)topk.cpuData)[0] + 1.0e-3f);
            if (HistoryCacheDebugEnabled()) {
                fprintf(stderr,
                        "[dots-history] next_token=%d full_cache=%d sliding_cache=%d\n",
                        token,
                        pastKeyValues[0].first.dims.size() > 1
                            ? pastKeyValues[0].first.dims[1] : 0,
                        pastKeyValues[2].first.dims.size() > 1
                            ? pastKeyValues[2].first.dims[1] : 0);
            }
            return token;
        }

        LastTokensUnit emptyLastTokens;
        const LastTokensUnit &last = lastTokens.units.empty()
                                         ? emptyLastTokens
                                         : lastTokens.units[0];
        return LLMSampling(logits, logits.Count(0) / logits.dims.back() - 1,
                           generationConfig, last);
    }

    bool Dots3NoteModel::NeedAttentionMask(int qlen, int klen) {
        (void)qlen;
        (void)klen;
        return false;
    }

    int Dots3NoteModel::GetKVCacheRetainedTokens(int layer) const {
        return IsFullAttentionLayer(layer) ? -1 : shortContextLimit;
    }

    void Dots3NoteModel::WarmUp() {
        printf("Warmup Dots3-Note...\n");
        Data inputIds(DataType::FLOAT32, {1, 1}, {1.0f});
        Data positionIds(DataType::FLOAT32, {1, 1}, {0.0f});
        std::vector<std::pair<Data, Data>> pastKeyValues;
        pastKeyValues.reserve(block_cnt);
        for (int layer = 0; layer < block_cnt; layer++) {
            pastKeyValues.push_back({Data(kvCacheDataType),
                                     Data(kvCacheDataType)});
        }
        Forward(inputIds, Data(), positionIds, pastKeyValues);
        {
            std::lock_guard<std::mutex> guard(indexerCacheMutex);
            indexerCaches.erase(&pastKeyValues);
        }

        elementsInKVCachePerToken = 0;
        for (int layer = 0; layer < (int)pastKeyValues.size(); layer++) {
            if (!IsFullAttentionLayer(layer)) {
                continue;
            }
            const auto &cache = pastKeyValues[layer];
            if (cache.first.dims.size() == 3) {
                elementsInKVCachePerToken +=
                    (long long)cache.first.dims[0] * cache.first.dims[2];
            }
            if (cache.second.dims.size() == 3) {
                elementsInKVCachePerToken +=
                    (long long)cache.second.dims[0] * cache.second.dims[2];
            }
        }
        printf("finish.\n");
    }

    void Dots3NoteModel::OnResponseContextCreated(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        std::shared_ptr<HistoryCacheMemory> pending;
        {
            std::lock_guard<std::mutex> guard(historyCacheMutex);
            pending.swap(pendingHistoryCache);
        }
        if (pending != nullptr) {
            std::lock_guard<std::mutex> forwardGuard(forwardLocker);
            RestoreHistoryCache(*pending, context->cacheLen, context);
            // A restored non-paged cache is already an active sequence.  The
            // legacy scheduler otherwise counts its allocated KV as occupying
            // maxBatch while still classifying it as an unscheduled prefill
            // (preTokens == 0), and terminates it without calling Forward.
            // Seed the normal decode bookkeeping so the uncached suffix is
            // evaluated at the same absolute positions as a full prefill.
            context->preTokens = context->cacheLen;
            context->intParams["add_special_tokens"] = 0;
            context->intParams["promptLen"] =
                context->cacheLen + (int)context->currentTokens.size();
            context->intParams["index"] = -1;
        }
        std::lock_guard<std::mutex> guard(responseContextsMutex);
        responseContexts[&context->pastKeyValues] = context;
    }

    void Dots3NoteModel::OnResponseContextRemoved(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        {
            std::lock_guard<std::mutex> guard(responseContextsMutex);
            responseContexts.erase(&context->pastKeyValues);
        }
        {
            std::lock_guard<std::mutex> guard(indexerCacheMutex);
            indexerCaches.erase(&context->pastKeyValues);
        }
    }

    bool Dots3NoteModel::TryRestoreHistoryCache(
            std::vector<int> &inputTokens, int &cacheLen) {
        if (!saveHistoryChat || inputTokens.size() <= 1) {
            return false;
        }
        std::shared_ptr<HistoryCacheMemory> best;
        int bestRestoreLength = 0;
        size_t recordCount = 0;
        {
            std::lock_guard<std::mutex> guard(historyCacheMutex);
            pendingHistoryCache.reset();
            recordCount = historyCache.size();
            for (auto &item : historyCache) {
                const std::vector<int> &cachedTokens = item.first;
                if (cachedTokens.empty() ||
                    cachedTokens.size() > inputTokens.size() ||
                    !std::equal(cachedTokens.begin(), cachedTokens.end(),
                                inputTokens.begin())) {
                    continue;
                }
                int restoreLength = (int)cachedTokens.size();
                if (cachedTokens.size() == inputTokens.size()) {
                    restoreLength--;
                }
                if (restoreLength <= bestRestoreLength ||
                    !CanRestoreHistoryCache(*item.second, restoreLength)) {
                    continue;
                }
                best = item.second;
                bestRestoreLength = restoreLength;
            }
            if (best != nullptr) {
                best->flushTime = ++historyCacheFlushTime;
                pendingHistoryCache = best;
            }
        }
        if (best == nullptr || bestRestoreLength <= 0) {
            if (HistoryCacheDebugEnabled()) {
                fprintf(stderr,
                        "[dots-history] miss input=%zu records=%zu\n",
                        inputTokens.size(), recordCount);
            }
            return false;
        }
        if (HistoryCacheDebugEnabled()) {
            fprintf(stderr,
                    "[dots-history] hit input=%zu snapshot=%d restore=%d\n",
                    inputTokens.size(), best->sequenceLength,
                    bestRestoreLength);
        }
        inputTokens.erase(inputTokens.begin(),
                          inputTokens.begin() + bestRestoreLength);
        cacheLen = bestRestoreLength;
        return true;
    }

    void Dots3NoteModel::TryRecordResponseContext(ResponseContext *context) {
        if (context == nullptr || !saveHistoryChat ||
            context->pastKeyValues.empty() || context->allTokens.empty()) {
            return;
        }
        int sequenceLength =
            context->pastKeyValues[0].first.dims.size() > 1
                ? context->pastKeyValues[0].first.dims[1] : 0;
        if (sequenceLength <= 0 ||
            sequenceLength > (int)context->allTokens.size()) {
            return;
        }
        std::vector<int> tokens(context->allTokens.begin(),
                                context->allTokens.begin() + sequenceLength);
        RecordHistoryCache(tokens, context->pastKeyValues, sequenceLength);
    }

    void Dots3NoteModel::MaybeRecordPromptHistoryCache(
            std::vector<std::pair<Data, Data>> &pastKeyValues) {
        if (!saveHistoryChat || pastKeyValues.empty()) {
            return;
        }
        ResponseContext *context = nullptr;
        {
            std::lock_guard<std::mutex> guard(responseContextsMutex);
            auto it = responseContexts.find(&pastKeyValues);
            if (it != responseContexts.end()) {
                context = it->second;
            }
        }
        if (context == nullptr) {
            return;
        }
        int sequenceLength = pastKeyValues[0].first.dims.size() > 1
                                 ? pastKeyValues[0].first.dims[1] : 0;
        if (sequenceLength <= 0 ||
            sequenceLength != context->inputTokens ||
            sequenceLength > (int)context->allTokens.size()) {
            return;
        }
        std::vector<int> tokens(context->allTokens.begin(),
                                context->allTokens.begin() + sequenceLength);
        RecordHistoryCache(tokens, pastKeyValues, sequenceLength);
    }

    void Dots3NoteModel::RecordHistoryCache(
            const std::vector<int> &tokens,
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            int sequenceLength) {
        if (!saveHistoryChat || sequenceLength <= 0 ||
            sequenceLength != (int)tokens.size() ||
            (int)pastKeyValues.size() < block_cnt) {
            return;
        }
        {
            std::lock_guard<std::mutex> guard(historyCacheMutex);
            auto existing = historyCache.find(tokens);
            if (existing != historyCache.end()) {
                existing->second->flushTime = ++historyCacheFlushTime;
                return;
            }
        }

        auto memory = std::make_shared<HistoryCacheMemory>();
        memory->tokens = tokens;
        memory->sequenceLength = sequenceLength;
        memory->pastKeyValues.resize(block_cnt);
        std::shared_ptr<IndexerCacheMemory> activeIndexerCache;
        {
            std::lock_guard<std::mutex> guard(indexerCacheMutex);
            auto it = indexerCaches.find(&pastKeyValues);
            if (it != indexerCaches.end()) {
                activeIndexerCache = it->second;
            }
        }
        const bool moveToCpu = GetHistoryCacheInCPU();
        auto copyTensor = [&](const Data &source, Data &target) {
            AssertInFastLLM(!source.multiDeviceData &&
                            source.multiDeviceDatas.empty() &&
                            !source.isPagedKVCache,
                            "Dots3-Note history cache expects a contiguous tensor.\n");
            if (!moveToCpu || source.dataDevice == DataDevice::CPU) {
                target.CopyFrom(source);
                if (moveToCpu) {
                    target.lockInCPU = true;
                }
                return;
            }

            target.name = source.name;
            target.isKVCache = source.isKVCache;
            target.cacheUid = source.cacheUid;
            target.dataType = source.dataType;
            target.UpdateUnitSize();
            target.dataDevice = DataDevice::CPU;
            if (source.dims.empty()) {
                target.lockInCPU = true;
                return;
            }
            if (!source.expansionDims.empty() &&
                source.expansionDims != source.dims) {
                target.Expansion(source.expansionDims);
                target.Resize(source.dims);
                target.Allocate();
            } else {
                target.Resize(source.dims);
                target.Allocate();
            }
            const size_t bytes = source.GetBytes();
            AssertInFastLLM(target.cpuData != nullptr &&
                            bytes <= target.expansionBytes,
                            "Dots3-Note CPU history cache allocation failed.\n");
#ifdef USE_CUDA
            AssertInFastLLM(source.cudaData != nullptr,
                            "Dots3-Note history cache source has no CUDA data.\n");
            int originalDevice = FastllmCudaGetDevice();
            int sourceDevice = GetPointerDeviceId(source.cudaData);
            if (sourceDevice < 0 && !source.dataDeviceIds.empty()) {
                sourceDevice = source.dataDeviceIds[0];
            }
            AssertInFastLLM(sourceDevice >= 0,
                            "Dots3-Note history cache source GPU is unknown.\n");
            FastllmCudaSetDevice(sourceDevice);
            FastllmCudaCopyFromDeviceToHost(
                target.cpuData, source.cudaData, bytes);
            FastllmCudaSetDevice(originalDevice);
#else
            ErrorInFastLLM(
                "Dots3-Note CUDA cache cannot be copied in a CPU build.\n");
#endif
            target.lockInCPU = true;
        };

        for (int layer = 0; layer < block_cnt; layer++) {
            copyTensor(pastKeyValues[layer].first,
                       memory->pastKeyValues[layer].first);
            copyTensor(pastKeyValues[layer].second,
                       memory->pastKeyValues[layer].second);
        }
        Executor &executor = *((Executor *)GetExecutor());
        if (UsesCudaIndexer(executor) && activeIndexerCache != nullptr &&
            (int)activeIndexerCache->layers.size() >= block_cnt) {
            memory->indexerCache = std::make_shared<IndexerCacheMemory>();
            memory->indexerCache->layers.resize(block_cnt);
            for (int layer = 0; layer < block_cnt; layer++) {
                if (!IsFullAttentionLayer(layer)) {
                    continue;
                }
                copyTensor(activeIndexerCache->layers[layer].keys,
                           memory->indexerCache->layers[layer].keys);
                copyTensor(activeIndexerCache->layers[layer].scales,
                           memory->indexerCache->layers[layer].scales);
            }
        }

        std::lock_guard<std::mutex> guard(historyCacheMutex);
        auto existing = historyCache.find(tokens);
        if (existing != historyCache.end()) {
            existing->second->flushTime = ++historyCacheFlushTime;
            return;
        }
        while ((int)historyCache.size() >= historyCacheMaxRecords) {
            auto oldest = historyCache.end();
            for (auto it = historyCache.begin();
                 it != historyCache.end(); ++it) {
                if (oldest == historyCache.end() ||
                    it->second->flushTime < oldest->second->flushTime) {
                    oldest = it;
                }
            }
            if (oldest == historyCache.end()) {
                break;
            }
            historyCache.erase(oldest);
        }
        memory->flushTime = ++historyCacheFlushTime;
        historyCache[tokens] = std::move(memory);
        if (HistoryCacheDebugEnabled()) {
            const auto &stored = historyCache[tokens];
            fprintf(stderr,
                    "[dots-history] record tokens=%zu full=%d sliding=%d records=%zu\n",
                    tokens.size(),
                    stored->pastKeyValues[0].first.dims[1],
                    stored->pastKeyValues[2].first.dims[1],
                    historyCache.size());
        }
    }

    bool Dots3NoteModel::CanRestoreHistoryCache(
            const HistoryCacheMemory &memory, int restoreLength) const {
        Executor &executor = *((Executor *)GetExecutor());
        const bool requiresIndexerCache = UsesCudaIndexer(executor);
        if (restoreLength <= 0 ||
            restoreLength > memory.sequenceLength ||
            memory.sequenceLength != (int)memory.tokens.size() ||
            (int)memory.pastKeyValues.size() < block_cnt ||
            (requiresIndexerCache &&
             (memory.indexerCache == nullptr ||
              (int)memory.indexerCache->layers.size() < block_cnt))) {
            return false;
        }
        for (int layer = 0; layer < block_cnt; layer++) {
            const Data &key = memory.pastKeyValues[layer].first;
            const Data &value = memory.pastKeyValues[layer].second;
            if (key.dims.size() != 3 || value.dims.size() != 3 ||
                key.dims[1] != value.dims[1]) {
                return false;
            }
            int sourceLength = key.dims[1];
            if (IsFullAttentionLayer(layer)) {
                if (sourceLength < restoreLength) {
                    return false;
                }
                if (requiresIndexerCache) {
                    const IndexerLayerCache &indexer =
                        memory.indexerCache->layers[layer];
                    if (indexer.keys.dims.size() != 3 ||
                        indexer.scales.dims.size() != 3 ||
                        indexer.keys.dims[1] < restoreLength ||
                        indexer.scales.dims[1] < restoreLength) {
                        return false;
                    }
                }
                continue;
            }
            int desiredLength = std::min(
                restoreLength, std::max(0, shortContextLimit - 1));
            int desiredStart = restoreLength - desiredLength;
            int sourceStart = memory.sequenceLength - sourceLength;
            if (sourceStart > desiredStart ||
                desiredStart + desiredLength >
                    sourceStart + sourceLength) {
                return false;
            }
        }
        return true;
    }

    void Dots3NoteModel::RestoreHistoryCache(
            const HistoryCacheMemory &memory, int restoreLength,
            ResponseContext *context) {
        AssertInFastLLM(context != nullptr &&
                        CanRestoreHistoryCache(memory, restoreLength),
                        "Dots3-Note history cache snapshot is incomplete.\n");
        context->pastKeyValues.resize(block_cnt);
        for (int layer = 0; layer < block_cnt; layer++) {
            const Data &sourceKey = memory.pastKeyValues[layer].first;
            const Data &sourceValue = memory.pastKeyValues[layer].second;
            int start = 0;
            int end = restoreLength;
            if (!IsFullAttentionLayer(layer)) {
                int desiredLength = std::min(
                    restoreLength, std::max(0, shortContextLimit - 1));
                int desiredStart = restoreLength - desiredLength;
                int sourceStart = memory.sequenceLength - sourceKey.dims[1];
                start = desiredStart - sourceStart;
                end = start + desiredLength;
            }
            Data &targetKey = context->pastKeyValues[layer].first;
            Data &targetValue = context->pastKeyValues[layer].second;
            Split(sourceKey, 1, start, end, targetKey);
            Split(sourceValue, 1, start, end, targetValue);
            targetKey.SetKVCache();
            targetValue.SetKVCache();
            if (targetKey.dims.size() == 3 && targetKey.dims[1] > 0) {
                std::vector<int> keyCapacity = targetKey.dims;
                std::vector<int> valueCapacity = targetValue.dims;
                // Expansion needs a strictly larger shape.  An exact history
                // hit restores window_size - 1 tokens (512 for this model),
                // which is already a multiple of the cache block; rounding
                // up to the same value leaves Expansion without a valid
                // growth axis.  Always reserve the following block.
                keyCapacity[1] = (keyCapacity[1] / 128 + 1) * 128;
                valueCapacity[1] = (valueCapacity[1] / 128 + 1) * 128;
                targetKey.Expansion(keyCapacity);
                targetValue.Expansion(valueCapacity);
                targetKey.SetKVCache();
                targetValue.SetKVCache();
            }
        }
        Executor &executor = *((Executor *)GetExecutor());
        if (UsesCudaIndexer(executor)) {
            auto restoredIndexer = std::make_shared<IndexerCacheMemory>();
            restoredIndexer->layers.resize(block_cnt);
            for (int layer = 0; layer < block_cnt; layer++) {
                if (!IsFullAttentionLayer(layer)) {
                    continue;
                }
                const IndexerLayerCache &source =
                    memory.indexerCache->layers[layer];
                IndexerLayerCache &target = restoredIndexer->layers[layer];
                Split(source.keys, 1, 0, restoreLength, target.keys);
                Split(source.scales, 1, 0, restoreLength, target.scales);
                std::vector<int> keyCapacity = target.keys.dims;
                std::vector<int> scaleCapacity = target.scales.dims;
                keyCapacity[1] = (restoreLength / 128 + 1) * 128;
                scaleCapacity[1] = keyCapacity[1];
                target.keys.Expansion(keyCapacity);
                target.scales.Expansion(scaleCapacity);
                target.keys.SetKVCache();
                target.scales.SetKVCache();
            }
            {
                std::lock_guard<std::mutex> guard(indexerCacheMutex);
                indexerCaches[&context->pastKeyValues] = restoredIndexer;
            }
        }
        if (HistoryCacheDebugEnabled()) {
            fprintf(stderr,
                    "[dots-history] restored=%d full=%d sliding=%d\n",
                    restoreLength,
                    context->pastKeyValues[0].first.dims[1],
                    context->pastKeyValues[2].first.dims[1]);
        }
    }

    std::string Dots3NoteModel::MakeInput(const std::string &history,
                                           int round,
                                           const std::string &input) {
        const std::string prefix = round == 0
            ? "<|system|>You are a helpful assistant.<|endofsystem|>"
            : history;
        const std::string noThink =
            input.size() >= 10 &&
            input.compare(input.size() - 10, 10, "<no_think>") == 0
                ? ""
                : "<no_think>";
        return prefix + "<|user|>" + input + noThink +
               "<|endofuser|><|assistant|><think>\n\n</think>\n\n";
    }

    std::string Dots3NoteModel::MakeHistory(const std::string &history,
                                             int round,
                                             const std::string &input,
                                             const std::string &output) {
        return MakeInput(history, round, input) + output +
               "<|endofassistant|>";
    }
}
