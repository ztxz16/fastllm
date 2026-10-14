// Independent hybrid forward; tensor blocks and CUDA kernels remain shared.
#include "qwen4_exp_execution.h"
#include "models/qwen4_kvmem.h"
#include "models/qwen4_paged_cache.h"
#include "utils.h"

#include <cstdio>
#include <limits>
#include <stdexcept>
namespace fastllm {
    void Qwen4ExpModel::ConfigureKvMem(const KvMemConfig &config) {
        if (mainLoop != nullptr || autoWarmupRunning.load() || kvMemLocked ||
            GetPagedKVCacheManager(kvCacheId, true) != nullptr)
            throw std::invalid_argument("KVMem must be configured before warmup or inference");
        if (!config.enabled) {
            kvMemConfig = config;
            return;
        }
#ifndef USE_CUDA
        throw std::invalid_argument("KVMem requires a CUDA build");
#else
        if (maxBatch > 1) throw std::invalid_argument("KVMem currently requires max_batch=1");
        if (config.maxTokens > max_positions)
            throw std::invalid_argument("KVMem logical context exceeds the model's configured context");
        if (GetFastllmEnv().skipWarmup) throw std::invalid_argument("KVMem requires warmup");
        ValidateKvMemModel(config);
        kvMemConfig = config;
        maxBatch = 1;
        saveHistoryChat = false;
        tokensLimit = promptLimit = config.maxTokens;
#endif
    }

    void Qwen4ExpModel::ValidateKvMemModel(const KvMemConfig &config) const {
        if (!config.enabled) return;
        if (!SupportsKvMem())
            throw std::invalid_argument("Qwen4 KVMem requires the Hybrid execution path");
        if (GetKVCacheInCPU() || deviceMap.empty() || dataType != FLOAT16 ||
            kvCacheDataType != FLOAT16 || (head_dim != 128 && head_dim != 256) ||
            indexerCompressRatio != 4 || indexerBudget <= 0)
            throw std::invalid_argument("Qwen4 KVMem requires CUDA FP16 QSA (head_dim 128/256, compression ratio 4)");
        for (const auto &entry : deviceMap) {
            if (entry.first != "cuda" && entry.first.rfind("cuda:", 0) != 0)
                throw std::invalid_argument("Qwen4 KVMem requires CUDA attention devices");
        }
        if (config.maxTokens < 2 || config.maxTokens > std::numeric_limits<int>::max() - 128 ||
            config.residentTokens < 4 || config.residentTokens % 4 ||
            config.prefillTokens < 1 || config.prefillTokens > config.maxTokens)
            throw std::invalid_argument("Qwen4 KVMem requires positive capacities and resident_tokens divisible by 4");
        // Include a possible MTP attention layer and its provisional suffix.
        // Rank-local caches split KV heads; this validation runs on the root.
        uint64_t layers = 0;
        for (int layer = 0; layer < block_cnt; ++layer) layers += !IsLinearAttentionLayer(layer);
        if (!layers) throw std::invalid_argument("Qwen4 KVMem requires full-attention layers");
        ++layers;
        const uint64_t tokens = (uint64_t(config.maxTokens) + Qwen4MtpDraftsPerStep() + 4) / 4 * 4;
        if (tokens * layers * uint64_t(num_key_value_heads) * head_dim * 4 > config.hostBytes)
            throw std::invalid_argument("Qwen4 KVMem host budget cannot back target and MTP KV");
    }

    void Qwen4ExpModel::PrepareHybridKvMemCache(int layer, const Data &key, int heads,
                                               Data &pastKey, Data &pastValue) {
#ifdef USE_CUDA
        Qwen4CudaDeviceGuard deviceGuard(key.dataDeviceIds);
        if (auto *existing = Qwen4GetKvMem(pastKey)) {
            existing->Reset();
            return;
        }
        AssertInFastLLM(key.dataDevice == DataDevice::CUDA && key.dataDeviceIds.size() == 1,
                        "Qwen4 KVMem requires a rank-local CUDA cache.");
        std::shared_ptr<Qwen4KvMem> storage;
        if (servingCache) {
            const auto found = servingCache->kvMemStorage.find(layer);
            if (found != servingCache->kvMemStorage.end())
                storage = std::dynamic_pointer_cast<Qwen4KvMem>(found->second);
        }
        if (storage) storage->Reset();
        else {
            KvMemConfig storageConfig = kvMemConfig;
            storageConfig.maxTokens += Qwen4MtpDraftsPerStep();
            storage = std::make_shared<Qwen4KvMem>(storageConfig, key.dataType,
                heads, head_dim, key.dataDeviceIds.front());
        }
        storage->Attach(pastKey, pastValue, storage);
#endif
    }

    size_t Qwen4ExpModel::RetainHybridKvMemStorage(
            const std::vector<std::pair<Data, Data>> &cache,
            const std::shared_ptr<KvMemCache> &mtp, ServingCache &reserved) const {
        size_t bytes = 0;
#ifdef USE_CUDA
        for (int layer = 0; layer <= block_cnt; ++layer) {
            auto owner = layer == block_cnt ? mtp : cache[layer].first.kvMemCache;
            if (!owner) continue;
            auto storage = std::dynamic_pointer_cast<Qwen4KvMem>(owner);
            AssertInFastLLM(storage != nullptr, "Qwen4 Hybrid KV storage type mismatch.");
            bytes += storage->gpuBytes;
            reserved.kvMemStorage.emplace(layer, std::move(owner));
        }
#endif
        return bytes;
    }

    void Qwen4ExpModel::FinishHybridKvMemWarmup() {
        if (kvMemConfig.enabled) {
            kvMemLocked = true;
            std::printf("[Fastllm] Qwen4 KVMem: logical=%d, resident=%d tokens/layer; exact QSA selection, host backing, CUDA Graph enabled when available.\n",
                kvMemConfig.maxTokens, kvMemConfig.residentTokens);
        }
    }

    std::vector<int> Qwen4ExpModel::ForwardHybrid(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<std::vector<float> *> *retLogits,
            Data *targetMultiHidden,
            std::vector<int> *allVerificationTokens,
            std::vector<unsigned char> *verificationAccepted,
            TargetVerificationCapture *verificationCapture,
            bool recordPrefixSnapshot,
            bool allowDecodeCudaGraph,
            bool decodeEquivalentGdn,
            const std::vector<int> *hostInputTokens,
            bool materializeCausalMaskOnGraphFallback,
            const Data *precomputedEmbedding, PrefillStageContext *prefill) {
        ForwardState frame(*this, batch, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig,
            lastTokens, retLogits, targetMultiHidden, allVerificationTokens, verificationAccepted,
            verificationCapture, recordPrefixSnapshot, allowDecodeCudaGraph, decodeEquivalentGdn,
            hostInputTokens, materializeCausalMaskOnGraphFallback, precomputedEmbedding, prefill);
        frame.BeginHybridStep();
        frame.Prepare();
        frame.PrefetchExperts(frame.FirstLayer());
        for (int layer = frame.FirstLayer(); layer < frame.EndLayer(); ++layer) {
            const int action = frame.BeginLayer(layer);
            if (action == ForwardState::Complete) break;
            if (action == ForwardState::Skip) continue;
            frame.HybridExperts(layer);
            frame.PrefetchExperts(layer + 1);
            frame.EndLayer(layer);
        }
        return frame.Finish();
    }
}
