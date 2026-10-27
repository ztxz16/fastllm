#include "models/basellm.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace fastllm {
    void basellm::ConfigureKvMem(const KvMemConfig &config) {
        if (mainLoop != nullptr || autoWarmupRunning.load() || kvMemLocked ||
            GetPagedKVCacheManager(kvCacheId, true) != nullptr) {
            throw std::invalid_argument("KVMem must be configured before warmup or inference");
        }
        config.Validate(GetPageLen());
        if (!config.enabled) {
            kvMemConfig = config;
            return;
        }
#ifndef USE_CUDA
        throw std::invalid_argument("KVMem requires a CUDA build");
#else
        if (!SupportsKvMem() || !CanUseGPUForward() || !use_new_engine) {
            throw std::invalid_argument("KVMem is not supported by this model/backend");
        }
        if (deviceMap.size() != 1 ||
            (deviceMap.begin()->first != "cuda" && deviceMap.begin()->first.find("cuda:") != 0) ||
            deviceMap.begin()->first.find(',') != std::string::npos || GetKVCacheInCPU()) {
            throw std::invalid_argument("KVMem currently requires one CUDA device");
        }
        if ((dataType != DataType::FLOAT16 && dataType != DataType::BFLOAT16) || kvCacheDataType != dataType) {
            throw std::invalid_argument("KVMem requires matching FP16/BF16 compute and KV dtypes");
        }
        if (head_dim != 128 && head_dim != 256) {
            throw std::invalid_argument("KVMem: current model adapters require head_dim 128 or 256");
        }
        if (maxBatch > 1) throw std::invalid_argument("KVMem currently requires max_batch=1");
        if (config.maxTokens > max_positions) {
            throw std::invalid_argument("KVMem logical context exceeds the model's configured context");
        }
        if (const char *oldEngine = std::getenv("USE_OLD_ENGINE")) {
            std::string value(oldEngine);
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            if (value == "1" || value == "on") throw std::invalid_argument("KVMem requires the new GPU engine");
        }
        if (GetFastllmEnv().skipWarmup) throw std::invalid_argument("KVMem requires warmup");
        ValidateKvMemModel(config);
        uint64_t logicalTokens = (((uint64_t)config.maxTokens + GetPageLen() - 1) / GetPageLen()) * GetPageLen();
        uint64_t bytesPerToken = 0, limit = config.hostBytes / logicalTokens;
        for (int i = 0; i < block_cnt; ++i) {
            if (!KvMemLayerEligible(i)) continue;
            uint64_t bytes = KvMemLayerBytesPerToken(i);
            if (!bytes || bytes > limit - bytesPerToken) {
                throw std::invalid_argument("KVMem host budget cannot back all full-attention layers at max_tokens");
            }
            bytesPerToken += bytes;
        }
        if (!bytesPerToken) throw std::invalid_argument("KVMem: model has no eligible KV layers");
        kvMemConfig = config;
        maxBatch = 1;
        saveHistoryChat = false;
        tokensLimit = promptLimit = config.maxTokens;
#endif
    }

    void basellm::PrepareKvMemCaches(int batch, int devices,
                                    const std::vector<std::pair<Data*, Data*>> &caches) {
        if (!kvMemConfig.enabled) return;
        if (batch != 1 || devices != 1 || (int)caches.size() != block_cnt) {
            throw std::invalid_argument("KVMem currently requires one request on one CUDA device");
        }
        ValidateKvMemModel(kvMemConfig);
        if (maxBatch > 1 || kvCacheDataType != dataType || saveHistoryChat) {
            throw std::invalid_argument("KVMem configuration changed after setup");
        }
        uint64_t logicalTokens = (((uint64_t)kvMemConfig.maxTokens + GetPageLen() - 1) / GetPageLen()) * GetPageLen();
        for (int i = 0; i < block_cnt; ++i) {
            if (!KvMemLayerEligible(i)) continue;
            std::shared_ptr<KvMemConfig> config;
            for (Data *cache : {caches[i].first, caches[i].second}) {
                if (cache->kvMemConfig) continue;
                if (!cache->dims.empty() || !cache->pageIndex.empty()) {
                    throw std::invalid_argument("KVMem must start with an empty request cache");
                }
                if (!config) {
                    config = std::make_shared<KvMemConfig>(kvMemConfig);
                    // Reserve each layer's exact worst-case backing budget.
                    // Their sum was checked against the request limit above.
                    config->hostBytes = logicalTokens * KvMemLayerBytesPerToken(i);
                }
                cache->kvMemConfig = config;
            }
        }
    }

    void basellm::WarmupKvMem() {
        ResponseContext context;
        context.Init(block_cnt, dataType, kvCacheDataType);
        std::vector<std::pair<Data*, Data*>> caches;
        for (auto &kv : context.pastKeyValues) caches.emplace_back(&kv.first, &kv.second);
        Data input(DataType::FLOAT32, {1, 2}, std::vector<float>{1, 1});
        Data positions(DataType::FLOAT32, {1, 2}, std::vector<float>{0, 1});
        std::vector<Data*> masks{nullptr}, positionIds{&positions};
        std::vector<int> lengths{2};
        std::vector<GenerationConfig> configs{GenerationConfig()};
        LastTokensManager lastTokens;
        ForwardGPU(1, input, masks, positionIds, lengths, caches, configs, lastTokens);
        tokensLimit = promptLimit = kvMemConfig.maxTokens;
        maxBatch = 1;
        kvMemLocked = true;
        printf("[Fastllm] KVMem enabled: logical=%d, resident=%d, host=%llu MiB; single-request eager decoding.\n",
               kvMemConfig.maxTokens, kvMemConfig.residentTokens,
               (unsigned long long)(kvMemConfig.hostBytes >> 20));
    }
}
