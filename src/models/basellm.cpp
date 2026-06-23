//
// Created by huangyuyang on 6/25/23.
//

#include "basellm.h"
#include "utils.h"
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <chrono>
#include <set>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    namespace {
        static bool NeedRepeatPenalty(const GenerationConfig &config) {
            float diff = config.repeat_penalty - 1.0f;
            return diff > 1e-6f || diff < -1e-6f;
        }

        static std::vector<float>* CreatePendingResultLogits(const GenerationConfig &config) {
            return config.output_logits ? new std::vector<float>() : nullptr;
        }

        static void QueueGeneratedResultLogits(ResponseContext *ctx,
                                               std::vector<std::vector<float>*> &logits,
                                               int index) {
            if (ctx == nullptr || index < 0 || index >= (int)logits.size() || logits[index] == nullptr) {
                return;
            }
            ctx->resultLogits.push(logits[index]);
            logits[index] = nullptr;
        }

        static void ReleasePendingResultLogits(std::vector<std::vector<float>*> &logits) {
            for (auto *&item : logits) {
                delete item;
                item = nullptr;
            }
        }

        static void ReleasePagedCachePages(Data &cache, bool clearDims = false) {
            std::set<std::pair<PagedCacheManager*, int> > releasedPages;
            auto releaseUnique = [&](Data &pagedCache) {
                if (!pagedCache.isPagedKVCache || pagedCache.pagedKVCacheData == nullptr ||
                    pagedCache.pageIndex.empty()) {
                    return;
                }
                std::vector<int> uniquePages;
                for (int page : pagedCache.pageIndex) {
                    std::pair<PagedCacheManager*, int> key = {pagedCache.pagedKVCacheData, page};
                    if (releasedPages.insert(key).second) {
                        uniquePages.push_back(page);
                    }
                }
                if (!uniquePages.empty()) {
                    pagedCache.pagedKVCacheData->ReleasePageIndices(uniquePages);
                }
                pagedCache.pageIndex.clear();
                pagedCache.lastPageLen = 0;
            };
            releaseUnique(cache);
            if (cache.multiDeviceData) {
                for (auto &it : cache.multiDeviceDatas) {
                    if (it.second != nullptr) {
                        releaseUnique(*it.second);
                        if (clearDims) {
                            it.second->dims.clear();
                        }
                    }
                }
            }
            if (clearDims) {
                cache.dims.clear();
            }
        }

        static bool ToolCallConstraintStartsWith(const std::string &text, const std::string &prefix) {
            return text.size() >= prefix.size() &&
                   memcmp(text.data(), prefix.data(), prefix.size()) == 0;
        }

        static bool FindLastPrefix(const std::string &text,
                                   const std::vector<std::string> &prefixes,
                                   std::string::size_type &bestPos,
                                   const std::string **bestPrefix) {
            bestPos = std::string::npos;
            *bestPrefix = nullptr;
            for (const auto &prefix : prefixes) {
                if (prefix.empty()) {
                    continue;
                }
                auto pos = text.rfind(prefix);
                if (pos == std::string::npos) {
                    continue;
                }
                if (bestPos == std::string::npos || pos > bestPos) {
                    bestPos = pos;
                    *bestPrefix = &prefix;
                }
            }
            return *bestPrefix != nullptr;
        }

        static bool FindLastPrefixBefore(const std::string &text,
                                         const std::vector<std::string> &prefixes,
                                         std::string::size_type exclusiveEnd,
                                         std::string::size_type &bestPos) {
            bestPos = std::string::npos;
            if (exclusiveEnd == std::string::npos || exclusiveEnd > text.size()) {
                exclusiveEnd = text.size();
            }
            for (const auto &prefix : prefixes) {
                if (prefix.empty() || exclusiveEnd < prefix.size()) {
                    continue;
                }
                auto pos = text.rfind(prefix, exclusiveEnd - prefix.size());
                if (pos == std::string::npos) {
                    continue;
                }
                if (bestPos == std::string::npos || pos > bestPos) {
                    bestPos = pos;
                }
            }
            return bestPos != std::string::npos;
        }

        static std::string::size_type FindLastNeedleBefore(
                const std::string &text,
                const std::vector<std::string> &needles,
                std::string::size_type exclusiveEnd) {
            std::string::size_type bestPos = std::string::npos;
            if (exclusiveEnd == std::string::npos || exclusiveEnd > text.size()) {
                exclusiveEnd = text.size();
            }
            for (const auto &needle : needles) {
                if (needle.empty() || exclusiveEnd < needle.size()) {
                    continue;
                }
                auto pos = text.rfind(needle, exclusiveEnd - needle.size());
                if (pos == std::string::npos) {
                    continue;
                }
                if (bestPos == std::string::npos || pos > bestPos) {
                    bestPos = pos;
                }
            }
            return bestPos;
        }

        static bool HasUnclosedToolCallParameterBefore(
                const std::string &text,
                const GenerationConfig &config,
                std::string::size_type invokePos,
                std::string::size_type parameterPos) {
            std::string::size_type previousParameterPos = std::string::npos;
            if (!FindLastPrefixBefore(text, config.tool_call_parameter_name_prefixes,
                                      parameterPos, previousParameterPos) ||
                previousParameterPos < invokePos) {
                return false;
            }

            const std::vector<std::string> parameterCloseTags = {
                    "</｜DSML｜parameter>",
                    "</\\DSML\\parameter>",
            };
            auto closePos = FindLastNeedleBefore(text, parameterCloseTags, parameterPos);
            if (closePos != std::string::npos && closePos > previousParameterPos) {
                return false;
            }
            return true;
        }

        static bool FindActiveToolCallNamePartial(const std::string &text,
                                                  const GenerationConfig &config,
                                                  std::string &partial) {
            if (!config.tool_call_name_constraint_enabled ||
                config.tool_call_allowed_names.empty() ||
                config.tool_call_invoke_name_prefixes.empty()) {
                return false;
            }
            const std::string terminator =
                    config.tool_call_name_terminator.empty() ? "\"" : config.tool_call_name_terminator;
            std::string::size_type bestPos = std::string::npos;
            const std::string *bestPrefix = nullptr;
            if (!FindLastPrefix(text, config.tool_call_invoke_name_prefixes,
                                bestPos, &bestPrefix)) {
                return false;
            }
            auto nameStart = bestPos + bestPrefix->size();
            if (text.find(terminator, nameStart) != std::string::npos) {
                return false;
            }
            partial = text.substr(nameStart);
            return true;
        }

        static bool FindActiveToolCallInvokeName(const std::string &text,
                                                 const GenerationConfig &config,
                                                 std::string &toolName,
                                                 std::string::size_type &invokePos) {
            if (config.tool_call_invoke_name_prefixes.empty()) {
                return false;
            }
            const std::string terminator =
                    config.tool_call_name_terminator.empty() ? "\"" : config.tool_call_name_terminator;
            const std::string standardClose = "</｜DSML｜invoke>";
            const std::string alternateClose = "</\\DSML\\invoke>";
            std::string::size_type bestPos = std::string::npos;
            const std::string *bestPrefix = nullptr;
            if (!FindLastPrefix(text, config.tool_call_invoke_name_prefixes,
                                bestPos, &bestPrefix)) {
                return false;
            }
            auto nameStart = bestPos + bestPrefix->size();
            auto terminatorPos = text.find(terminator, nameStart);
            if (terminatorPos == std::string::npos) {
                return false;
            }
            auto standardClosePos = text.rfind(standardClose);
            auto alternateClosePos = text.rfind(alternateClose);
            auto closePos = standardClosePos;
            if (closePos == std::string::npos ||
                (alternateClosePos != std::string::npos &&
                 alternateClosePos > closePos)) {
                closePos = alternateClosePos;
            }
            if (closePos != std::string::npos && closePos > bestPos) {
                return false;
            }
            toolName = text.substr(nameStart, terminatorPos - nameStart);
            invokePos = bestPos;
            return !toolName.empty();
        }

        static bool FindActiveToolCallParameterNamePartial(
                const std::string &text,
                const GenerationConfig &config,
                std::string &partial,
                std::vector<std::string> &allowedNames) {
            if (!config.tool_call_parameter_name_constraint_enabled ||
                config.tool_call_allowed_parameter_names.empty() ||
                config.tool_call_parameter_name_prefixes.empty()) {
                return false;
            }
            std::string toolName;
            std::string::size_type invokePos = std::string::npos;
            if (!FindActiveToolCallInvokeName(text, config, toolName, invokePos)) {
                return false;
            }
            auto allowedIt = config.tool_call_allowed_parameter_names.find(toolName);
            if (allowedIt == config.tool_call_allowed_parameter_names.end() ||
                allowedIt->second.empty()) {
                return false;
            }
            std::string::size_type parameterPos = std::string::npos;
            const std::string *parameterPrefix = nullptr;
            if (!FindLastPrefix(text, config.tool_call_parameter_name_prefixes,
                                parameterPos, &parameterPrefix)) {
                return false;
            }
            if (parameterPos == std::string::npos || parameterPos < invokePos) {
                return false;
            }
            if (HasUnclosedToolCallParameterBefore(text, config,
                                                   invokePos, parameterPos)) {
                return false;
            }
            const std::string terminator =
                    config.tool_call_name_terminator.empty() ? "\"" : config.tool_call_name_terminator;
            auto nameStart = parameterPos + parameterPrefix->size();
            if (text.find(terminator, nameStart) != std::string::npos) {
                return false;
            }
            partial = text.substr(nameStart);
            allowedNames = allowedIt->second;
            return true;
        }

        static bool IsToolCallEnumTokenAllowed(const std::string &partial,
                                               const std::string &tokenText,
                                               const std::vector<std::string> &allowedValues,
                                               const std::string &terminator) {
            if (tokenText.empty()) {
                return false;
            }
            std::string combined = partial + tokenText;
            auto terminatorPos = combined.find(terminator);
            std::string namePart = terminatorPos == std::string::npos ?
                                   combined : combined.substr(0, terminatorPos);
            for (const auto &name : allowedValues) {
                if (terminatorPos != std::string::npos) {
                    if (namePart == name) {
                        return true;
                    }
                    continue;
                }
                if (ToolCallConstraintStartsWith(name, namePart)) {
                    return true;
                }
            }
            return false;
        }

        static bool IsPureGpuDeviceSpec(const std::string &device) {
#ifndef USE_CUDA
            return false;
#else
            std::map<int, int> ratios;
            std::vector<int> devices;
            if (device == "cuda") {
                return true;
            }
            if (device == "multicuda") {
                return true;
            }
            if (device.rfind("cuda:", 0) == 0) {
                devices = ParseDeviceIds(device, "cuda", ratios);
            } else if (device.rfind("multicuda:", 0) == 0) {
                devices = ParseDeviceIds(device, "multicuda", ratios);
            } else {
                return false;
            }
            if (devices.empty()) {
                return false;
            }
            for (int id : devices) {
                if (id == 99999) {
                    return false;
                }
            }
            return true;
#endif
        }

        static bool IsPureGpuDeviceMap(const std::map<std::string, int> &deviceMap) {
            bool hasGpuDevice = false;
            for (auto &it : deviceMap) {
                if (it.second <= 0) {
                    continue;
                }
                if (!IsPureGpuDeviceSpec(it.first)) {
                    return false;
                }
                hasGpuDevice = true;
            }
            return hasGpuDevice;
        }

        static bool IsPureGpuMode(const basellm *model) {
            if (model == nullptr || !IsPureGpuDeviceMap(model->deviceMap)) {
                return false;
            }
            return (model->moeDeviceMap.empty() || IsPureGpuDeviceMap(model->moeDeviceMap)) &&
                   (model->moeDeviceLayers < 0 || model->layeredMoeDeviceMap.empty() ||
                    IsPureGpuDeviceMap(model->layeredMoeDeviceMap));
        }

        static void PrintLoopProfile(const char *loopName,
                                     const std::vector<int> &seqLens,
                                     int outputTokens,
                                     const std::chrono::system_clock::time_point &startTime) {
            PrintProfiler();
            int total = 0;
            for (int len : seqLens) {
                total += len;
            }
            float spend = GetSpan(startTime, std::chrono::system_clock::now());
            float tokenPerSecond = spend > 0.0f ? (float)total / spend : 0.0f;
            printf("[fastllm-profile] loop = %s, batch = %d, input tokens = %d, output tokens = %d, spend = %f s, tokens / s = %f\n",
                   loopName, (int)seqLens.size(), total, outputTokens, spend, tokenPerSecond);
        }
    }

    static int NormalizeMaxBatchByModelCapability(basellm *model, int maxBatch) {
        if (model != nullptr && !model->canDoBatchForward && !model->canDoConcurrentForward) {
            model->maxBatch = 1;
            return 1;
        }
        return maxBatch;
    }

    int ResponseContextDict::CreateHandle() {
        locker.lock();
        int newId = 0;
        while (dicts.find(newId) != dicts.end()) {
            newId++;
        }
        dicts[newId] = new ResponseContext();
        locker.unlock();
        return newId;
    }

    ResponseContext *ResponseContextDict::GetHandle(int handleId) {
        locker.lock();
        ResponseContext *ret = dicts.find(handleId) != dicts.end() ? dicts[handleId] : nullptr;
        locker.unlock();
        return ret;
    }

    void ResponseContextDict::RemoveHandle(int handleId) {
        locker.lock();
        if (dicts.find(handleId) != dicts.end()) {
            delete dicts[handleId];
            dicts.erase(handleId);
        }
        locker.unlock();
    }

    ResponseContext::~ResponseContext() {
        for (auto &item : multimodalInput) {
            for (auto *data : item.second) {
                delete data;
            }
        }
        multimodalInput.clear();

        while (!resultLogits.empty()) {
            delete resultLogits.front();
            resultLogits.pop();
        }
    }

    void ResponseContext::Init(int blocks, DataType, DataType kvCacheDataType) {
        pastKeyValues.clear();
        for (int i = 0; i < blocks; i++) {
            pastKeyValues.push_back(std::make_pair(Data(kvCacheDataType),
                                                   Data(kvCacheDataType)));
            pastKeyValues.back().first.SetKVCache();
            pastKeyValues.back().second.SetKVCache();
        }
        intParams.clear();
        toolCallConstraintGeneratedText.clear();
        currentTokens.clear();
        allTokens.clear();
        while (resultTokenQueue.size() > 0){
            resultTokenQueue.pop();
        }
        isEnding = false;
        preTokens = 0;
    }

    void ResponseContext::TryRecord(basellm *model) {
        model->TryRecordResponseContext(this);
    }

    void basellm::TryRecordResponseContext(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        this->TryRecordHistoryCache(context->allTokens);
        if (this->saveHistoryChat && this->UseGenericHistoryCache()) {
            this->pastKVCacheManager.Record(context->allTokens, context->allTokens.size(), &context->pastKeyValues);
        }
    }

    void basellm::PrepareToolCallConstraint(ResponseContext *context, GenerationConfig &generationConfig) {
        generationConfig.tool_call_allowed_token_ids.clear();
        if (context == nullptr ||
            (!generationConfig.tool_call_name_constraint_enabled &&
             !generationConfig.tool_call_parameter_name_constraint_enabled)) {
            return;
        }
        std::string partial;
        std::vector<std::string> allowedValues;
        if (!FindActiveToolCallParameterNamePartial(
                    context->toolCallConstraintGeneratedText,
                    generationConfig,
                    partial,
                    allowedValues)) {
            if (!FindActiveToolCallNamePartial(
                        context->toolCallConstraintGeneratedText,
                        generationConfig,
                        partial)) {
                return;
            }
            allowedValues = generationConfig.tool_call_allowed_names;
        }
        if (allowedValues.empty()) {
            return;
        }

        std::vector<int> allowedIds;
        allowedIds.reserve(allowedValues.size() * 4);
        const std::string terminator =
                generationConfig.tool_call_name_terminator.empty()
                ? "\"" : generationConfig.tool_call_name_terminator;
        for (const auto &item : this->weight.tokenizer.tokenToStringDict) {
            int tokenId = item.first;
            std::string tokenText = this->weight.tokenizer.DecodeTokens(std::vector<int>{tokenId});
            if (IsToolCallEnumTokenAllowed(partial, tokenText,
                                           allowedValues, terminator)) {
                allowedIds.push_back(tokenId);
            }
        }
        std::sort(allowedIds.begin(), allowedIds.end());
        allowedIds.erase(std::unique(allowedIds.begin(), allowedIds.end()), allowedIds.end());
        generationConfig.tool_call_allowed_token_ids = std::move(allowedIds);
    }

    void basellm::UpdateToolCallConstraintState(ResponseContext *context, int tokenId) {
        if (context == nullptr ||
            (!context->generationConfig.tool_call_name_constraint_enabled &&
             !context->generationConfig.tool_call_parameter_name_constraint_enabled) ||
            tokenId < 0) {
            return;
        }
        context->toolCallConstraintGeneratedText += this->weight.tokenizer.DecodeTokens(std::vector<int>{tokenId});
        const size_t maxTrackedBytes = 8192;
        if (context->toolCallConstraintGeneratedText.size() > maxTrackedBytes) {
            context->toolCallConstraintGeneratedText.erase(
                    0, context->toolCallConstraintGeneratedText.size() - maxTrackedBytes);
        }
    }

    void basellm::RemoveResponseContext(int handleId) {
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context != nullptr) {
            this->OnResponseContextRemoved(context);
        }
        responseContextDict.RemoveHandle(handleId);
    }

    void ResponseContext::TryRecordPagedCache(basellm *model) {
        bool hasLinearAttentionCache = false;
        for (int i = 0; i < (int)this->pastKeyValues.size(); i++) {
            auto &kvFirst = this->pastKeyValues[i].first;
            auto &kvSecond = this->pastKeyValues[i].second;
            if (kvFirst.isLinearAttention || kvSecond.isLinearAttention) {
                hasLinearAttentionCache = true;
                break;
            }
        }
        if (hasLinearAttentionCache &&
            (model == nullptr || !model->TryRecordPagedPrefixCacheExtra(this))) {
            return;
        }
        std::function<void(Data&)> recordPagedCache = [&](Data &cache) {
            if (cache.multiDeviceData && !cache.multiDeviceDatas.empty()) {
                bool recordedLocal = false;
                for (auto &it : cache.multiDeviceDatas) {
                    if (it.second != nullptr) {
                        recordPagedCache(*it.second);
                        recordedLocal = true;
                    }
                }
                if (recordedLocal) {
                    return;
                }
            }
            if (cache.pagedKVCacheData != nullptr && !cache.pageIndex.empty() &&
                cache.pagedKVCacheData->type == PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE) {
                cache.pagedKVCacheData->Record(this->allTokens, cache.pageIndex);
            }
        };
        for (int i = 0; i < (int)this->pastKeyValues.size(); i++) {
            auto &kvFirst = this->pastKeyValues[i].first;
            auto &kvSecond = this->pastKeyValues[i].second;
            recordPagedCache(kvFirst);
            recordPagedCache(kvSecond);
        }
    }

    PagedCacheManager* basellm::GetPagedKVCacheManager(int layerIndex, bool isKey) const {
        if (layerIndex < 0) {
            return nullptr;
        }
        return GetPagedCacheManager(layerIndex * 2 + (isKey ? 0 : 1));
    }

    std::vector<std::pair<int, PagedCacheManager*> > basellm::GetPagedKVCacheManagers(int layerIndex, bool isKey) const {
        std::vector<std::pair<int, PagedCacheManager*> > ret;
        PagedCacheManager *manager = this->GetPagedKVCacheManager(layerIndex, isKey);
        if (manager != nullptr) {
            int device = -1;
            Data *managerData = (Data*)manager;
            if (!managerData->dataDeviceIds.empty()) {
                device = managerData->dataDeviceIds[0];
            }
            ret.push_back(std::make_pair(device, manager));
        }
        return ret;
    }

    bool basellm::TryRecordPagedPrefixCacheExtra(ResponseContext *context) {
        (void)context;
        return false;
    }

    int basellm::QueryPagedPrefixCacheExtra(ResponseContext *context, int maxCachedLen) const {
        (void)context;
        return maxCachedLen;
    }

    bool basellm::RestorePagedPrefixCacheExtra(ResponseContext *context, int cachedLen) const {
        (void)context;
        (void)cachedLen;
        return true;
    }

    PastKVCacheMemory::PastKVCacheMemory(const std::vector <int> &inputToken, int tokens, long long flushTime, std::vector<std::pair<Data, Data> > *kv) {
        this->inputToken = inputToken;
        this->tokens = tokens;
        this->flushTime = flushTime;
        this->recordTimes = 1;
        auto dataType = (*kv)[0].first.dataType;
        for (int i = 0; i < kv->size(); i++) {
            this->kv.push_back(std::make_pair(Data(dataType), Data(dataType)));
        }
        for (int i = 0; i < kv->size(); i++) {
            this->kv[i].first.CopyFrom((*kv)[i].first);
            this->kv[i].second.CopyFrom((*kv)[i].second);

            if (GetHistoryCacheInCPU()) {
                this->kv[i].first.ToDevice(DataDevice::CPU);
                this->kv[i].first.lockInCPU = true;
                this->kv[i].second.ToDevice(DataDevice::CPU);
                this->kv[i].second.lockInCPU = true;
            }
        }
    }

    void PastKVCacheManager::SetMaxRecordNum(int maxRecordNum) {
        std::lock_guard <std::mutex> lock(this->locker);
        this->maxRecordNum = maxRecordNum;
    }

    void PastKVCacheManager::Record(const std::vector <int> &inputToken, int tokens, std::vector<std::pair<Data, Data> > *kv) {
        bool isLinear = false;
        for (int i = 0; i < kv->size(); i++) {
            if ((*kv)[i].first.isLinearAttention) {
                isLinear = true;
                break;
            }
        }
        std::lock_guard <std::mutex> lock(this->locker);
        if (this->memorys.find(inputToken) != this->memorys.end()) {
            this->memorys[inputToken]->recordTimes++;
            this->memorys[inputToken]->flushTime = ++flushTime;
            return;
        }

        std::vector <int> replaceCache;
        if (!isLinear) {
            for (auto &it : this->memorys) {
                // 如果当前inputToken覆盖了某一个cahce 90%以上的前缀，那么直接替换掉
                int lcp = 0;
                for (int i = 0; i < it.first.size() && i < inputToken.size(); i++) {
                    if (it.first[i] == inputToken[i]) {
                        lcp++;
                    } else {
                        break;
                    }
                }
                if (lcp > (int)it.first.size() * 9 / 10) {
                    replaceCache = it.first;
                }
            }
        }
        if (replaceCache.size() > 0) {
            delete this->memorys[replaceCache];
            this->memorys.erase(this->memorys.find(replaceCache));
        }
        if (this->memorys.size() >= this->maxRecordNum) {
            std::vector <int> eraseToken;
            long long minFlushTime = (1LL << 60);
            for (auto &it : this->memorys) {
                if (it.second->flushTime < minFlushTime) {
                    minFlushTime = it.second->flushTime;
                    eraseToken = it.first;
                }
            }
            delete this->memorys[eraseToken];
            this->memorys.erase(this->memorys.find(eraseToken));
        }

        this->memorys[inputToken] = new PastKVCacheMemory(inputToken, tokens, ++flushTime, kv);
    }

    void PastKVCacheManager::Remove(const std::vector <int> &inputToken) {
        std::lock_guard <std::mutex> lock(this->locker);
        if (this->memorys.find(inputToken) != this->memorys.end()) {
            if ((--this->memorys[inputToken]->recordTimes) <= 0) {
                delete this->memorys[inputToken];
                this->memorys.erase(this->memorys.find(inputToken));
            }
        }
    }

    std::pair <PastKVCacheMemory*, int> PastKVCacheManager::Get(const std::vector <int> &inputToken) {
        bool isLinear = false;
        if (this->memorys.size() > 0) {
            auto &kv = this->memorys.begin()->second->kv;
            for (int i = 0; i < kv.size(); i++) {
                if (kv[i].first.isLinearAttention) {
                    isLinear = true;
                    break;
                }
            }
        }
        std::lock_guard <std::mutex> lock(this->locker);
        int maxPrefixToken = 0;
        PastKVCacheMemory *ret = nullptr;
        for (auto &it : this->memorys) {
            const std::vector <int> &cur = it.first;
            int match = 0;
            for (int i = 0; i < cur.size() && i < inputToken.size(); i++) {
                if (inputToken[i] == cur[i]) {
                    match = i + 1;
                } else {
                    break;
                }
            }
            if (isLinear && match != cur.size()) {
                continue;
            }
            if (match > maxPrefixToken) {
                maxPrefixToken = match;
                ret = it.second;
            }
        }
        if (ret != nullptr) {
            ret->flushTime = ++this->flushTime;
        }
        maxPrefixToken = std::min(maxPrefixToken, (int)inputToken.size() - 1);
        return std::make_pair(ret, maxPrefixToken);
    }

    void PastKVCacheManager::Unlock() {
        locker.unlock();
    }

    basellm::~basellm() {
        {
            std::lock_guard<std::mutex> guard(dictLocker);
            this->isFree = true;
        }
        dictCV.notify_all();

        std::thread *loop = nullptr;
        {
            std::lock_guard<std::mutex> guard(mainLoopLocker);
            loop = this->mainLoop;
            this->mainLoop = nullptr;
        }
        if (loop != nullptr) {
            if (loop->joinable()) {
                loop->join();
            }
            delete loop;
        }

        {
            std::lock_guard<std::mutex> guard(responseContextDict.locker);
            for (auto &item : responseContextDict.dicts) {
                this->OnResponseContextRemoved(item.second);
                delete item.second;
            }
            responseContextDict.dicts.clear();
        }

        this->weight.ReleaseWeight();
    }

    std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                basellm::GetTensorMap(const std::vector <std::string> &tensorNames) {
        std::map <std::string, std::vector <std::pair <std::string, DataType> > > ret;
        for (auto &name : tensorNames) {
            std::string realName = name;
            if (StringEndWith(name, ".qweight")) {
                realName = name.substr(0, name.size() - 7) + "weight";
            }
            WeightType weightType = this->weight.GetWeightType(realName);
            DataType dataType = DataType::DATA_AUTO_NONE;
            if (weightType == WeightType::LINEAR) {
                dataType = DataType::DATA_AUTO_LINEAR;
                if (this->cantQuantLinears.find(realName) != this->cantQuantLinears.end()) {
                    dataType = DataType::FLOAT16;
                }
            } else if (weightType == WeightType::EMBEDDING) {
                dataType = DataType::DATA_AUTO_EMBEDDING;
            }
            ret[name].push_back(std::make_pair(realName, dataType));
        }
        return ret;
    }

    std::string basellm::Response(const std::string &oriInput, RuntimeResult retCb,
                                  const fastllm::GenerationConfig &generationConfig) {
        std::string input = oriInput;
        if (this->saveHistoryChat) {
            if (lastKeyValues != nullptr) {
                if (input.size() < lastPrompt.size() || (input.substr(0, lastPrompt.size()) != lastPrompt)) {
                    lastPrompt = "";
                    lastPromptTokens = 0;
                    delete lastKeyValues;
                    lastKeyValues = nullptr;
                } else {
                    input = input.substr(lastPrompt.size());
                }
            }
        } else {
            lastPrompt = "";
            lastPromptTokens = 0;
            delete lastKeyValues;
            lastKeyValues = nullptr;
        }

        //printf("lastPrompt = %s\n", lastPrompt.c_str());
        //printf("input = %s\n", input.c_str());

#ifdef USE_CUDA
        FastllmCudaClearBigBuffer();
#endif
        std::string prompt = input;
#ifdef PY_API
        size_t pos = input.rfind("time_stamp:");
        prompt = (generationConfig.enable_hash_id && pos != -1) ? input.substr(0, pos) : input;
        size_t hash_id = std::hash<std::string>{}(input);
#endif
        Data inputIds, attentionMask, positionIds;

        Data inputTokenData = this->weight.tokenizer.Encode(prompt);
        LastTokensManager tokens(1, generationConfig.last_n <= 0 ? max_positions : generationConfig.last_n);
        std::vector<std::vector<float> > inputTokens;
        inputTokens.resize(1);
        for (int i = 0; i < inputTokenData.Count(0); i++) {
            inputTokens[0].push_back(((float *) inputTokenData.cpuData)[i]);
            if (generationConfig.last_n <= 0)
                tokens.units[0].Push((int) ((float *) inputTokenData.cpuData)[i]);
        }

        if (lastKeyValues == nullptr) {
            lastKeyValues = new std::vector<std::pair<Data, Data> >();
            for (int i = 0; i < block_cnt; i++) {
                lastKeyValues->push_back(std::make_pair(Data(this->kvCacheDataType), Data(this->kvCacheDataType)));
                lastKeyValues->back().first.SetKVCache();
                lastKeyValues->back().second.SetKVCache();
            }
        }

        std::vector<std::pair<Data, Data> > &pastKeyValues = (*lastKeyValues);
        std::string retString = "";
        std::vector<float> results;
        // LastTokensManager tokens(1, generationConfig.last_n);
        int promptLen = lastPromptTokens + inputTokens[0].size(), index = 0;
        int add_special_tokens = generationConfig.add_special_tokens? 1: 0;
        FillLLMInputs(inputTokens, {{"promptLen", promptLen}, {"index", index}, {"add_special_tokens", add_special_tokens}},
                      inputIds, attentionMask, positionIds);
        ToDataType(attentionMask, this->dataType);
        while (true) {
            auto st = std::chrono::system_clock::now();
            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
            tokens.units[0].Push(ret);
            if (ret == eos_token_id
                || generationConfig.stop_token_ids.find(ret) != generationConfig.stop_token_ids.end()
                || eos_token_ids.find(ret) != eos_token_ids.end()) {
                break;
            }

            results.push_back(ret);
            std::string curString = weight.tokenizer.Decode(
                    Data(DataType::FLOAT32, {(int) results.size()}, results)).c_str();
            retString += curString;
            if (retCb)
#ifdef PY_API
            {
                if (generationConfig.enable_hash_id) {
                    std::stringstream ss;
                    ss << retString << "hash_id:"<<hash_id;
                    retCb(index, pybind11::bytes(ss.str()));
                } else {
                    retCb(index, pybind11::bytes(retString));
                }
            }
#else
                retCb(index, curString.c_str());
#endif
            index++;
            fflush(stdout);
            results.clear();

            inputTokens[0] = std::vector<float> {(float)ret};
            FillLLMInputs(inputTokens, {{"promptLen", promptLen}, {"index", index}, {"add_special_tokens", add_special_tokens}},
                          inputIds, attentionMask, positionIds);
            ToDataType(attentionMask, this->dataType);
            if (index == generationConfig.output_token_limit) {
                break;
            }
            // printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));
        }
        if (retCb)
#ifdef PY_API
        {
            if(generationConfig.enable_hash_id){
                std::stringstream ss;
                ss << retString << "hash_id:"<<hash_id;
                retCb(-1, pybind11::bytes(ss.str()));
            }else{
                retCb(-1, pybind11::bytes(retString));
            }
        }
#else
            retCb(-1, retString.c_str());
#endif

        lastPrompt += (input + retString);
        lastPromptTokens = promptLen + index;
        return retString;
    }

    void basellm::ResponseBatch(const std::vector<std::string> &inputs, std::vector<std::string> &outputs,
                                RuntimeResultBatch retCb, const fastllm::GenerationConfig &generationConfig) {
#ifdef USE_CUDA
        FastllmCudaClearBigBuffer();
#endif
#ifdef PY_API
        std::vector<std::string> prompts;
        std::vector < size_t > hash_ids;
        for (auto _input: inputs){
            size_t hash_id = std::hash<std::string>{}(_input);
            hash_ids.push_back(hash_id);

            size_t pos = _input.rfind("time_stamp:");
            std::string prompt = (generationConfig.enable_hash_id && pos != -1) ? _input.substr(0, pos) : _input;
            prompts.push_back(prompt);
        }
#else
        std::vector<std::string> prompts = inputs;
#endif
        // 1. first
        Data inputIds, attentionMask, positionIds;

        int batch = prompts.size();
        outputs.clear();
        outputs.resize(batch, "");

        LastTokensManager tokensManager(batch, generationConfig.last_n <= 0 ? max_positions : generationConfig.last_n);
        std::vector<std::vector<float> > inputTokens;
        inputTokens.resize(batch);

        for (int i = 0; i < batch; i++) {
            Data now = this->weight.tokenizer.Encode(prompts[i]);
            for (int j = 0; j < now.Count(0); j++) {
                inputTokens[i].push_back(((float *) now.cpuData)[j]);
                if (generationConfig.last_n <= 0)
                    tokensManager.units[i].Push((int) ((float *)now.cpuData)[j]);
            }
        }

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->kvCacheDataType),
                                                   Data(this->kvCacheDataType)));
            pastKeyValues.back().first.SetKVCache();
            pastKeyValues.back().second.SetKVCache();
        }

        std::vector <std::map <std::string, int> > params;
        params.resize(batch);
        for (int i = 0; i < batch; i++) {
            params[i]["promptLen"] = (int)inputTokens[i].size();
        }
        params[0]["index"] = 0;
        int index = 0;
        params[0]["add_special_tokens"] = generationConfig.add_special_tokens? 1: 0;

        // LastTokensManager tokensManager (batch, generationConfig.last_n);
        std::vector <bool> isEnding = std::vector <bool> (batch, false);
        FillLLMInputsBatch(inputTokens, params, inputIds, attentionMask, positionIds);
        ToDataType(attentionMask, this->dataType);
        while (true) {
            auto st = std::chrono::system_clock::now();
// ClearProfiler();
            std::vector <int> ret = ForwardBatch(batch, inputIds, attentionMask, positionIds, pastKeyValues,
                                                 generationConfig, tokensManager);
// PrintProfiler();
            for (int i = 0; i < batch; i++) {
                tokensManager.units[i].Push(ret[i]);
            }
            std::vector <float> fret;
            std::vector <float> results;
            int endingCount = 0;
            std::vector <std::string> curStrings;
            for (int i = 0; i < batch; i++) {
                fret.push_back(ret[i]);
                inputTokens[i] = std::vector <float> {(float)ret[i]};
                if (ret[i] == eos_token_id || eos_token_ids.find(ret[i]) != eos_token_ids.end()) {
                    isEnding[i] = true;
                } else {
                    auto itStopTk = generationConfig.stop_token_ids.find(ret[i]);
                    if (itStopTk != generationConfig.stop_token_ids.end()) {
                        isEnding[i] = true;
                    }
                }
                if (isEnding[i]) {
                    curStrings.push_back("");
                    endingCount++;
                    continue;
                }
                results.push_back(ret[i]);
                std::string curString = weight.tokenizer.Decode(
                        Data(DataType::FLOAT32, {(int) results.size()}, results)).c_str();
                outputs[i] += curString;
                curStrings.push_back(curString);
                results.clear();
            }

            if (endingCount == batch) {
                break;
            }
            if (retCb)
#ifdef PY_API
            {
                if (generationConfig.enable_hash_id) {
                    std::vector<pybind11::bytes> rtnStrings;
                    for (size_t i=0; i<batch; i++){
                        std::stringstream ss;
                        ss << curStrings[i] << "hash_id:" << hash_ids[i];
                        rtnStrings.push_back(pybind11::bytes(ss.str()));
                    }
                    retCb(index, rtnStrings);
                } else {
                    std::vector<pybind11::bytes> rtnStrings;
                    for (size_t i=0; i<batch; i++){
                        std::stringstream ss;
                        ss << curStrings[i];
                        rtnStrings.push_back(pybind11::bytes(ss.str()));
                    }
                    retCb(index, rtnStrings);
                }
            }
#else
                retCb(index, curStrings);
#endif
            index++;
            params[0]["index"] = index;
            FillLLMInputsBatch(inputTokens, params, inputIds, attentionMask, positionIds);
            ToDataType(attentionMask, this->dataType);
            // printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));

            if (index == generationConfig.output_token_limit) {
                break;
            }
        }
        if (retCb)
#ifdef PY_API
        {
            if (generationConfig.enable_hash_id) {
                std::vector<pybind11::bytes> rtnStrings;
                for (size_t i=0; i<batch; i++){
                    std::stringstream ss;
                    ss << outputs[i] << "hash_id:" << hash_ids[i];
                    rtnStrings.push_back(pybind11::bytes(ss.str()));
                }
                retCb(-1, rtnStrings);
            } else {
                std::vector<pybind11::bytes> rtnStrings;
                for (size_t i=0; i<batch; i++){
                    std::stringstream ss;
                    ss << outputs[i];
                    rtnStrings.push_back(pybind11::bytes(ss.str()));
                }
                retCb(-1, rtnStrings);
            }
        }
#else
            retCb(-1, outputs);
#endif
    }

    int basellm::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                         const fastllm::Data &positionIds,
                         std::vector<std::pair<Data, Data>> &pastKeyValues,
                         const fastllm::GenerationConfig &generationConfig,
                         const fastllm::LastTokensManager &lastTokens,
                         std::vector <float> *logits) {
        printf("Unsupport forward.\n");
        exit(0);
    }

    std::vector<int> basellm::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                           const fastllm::Data &positionIds,
                                           std::vector<std::pair<Data, Data>> &pastKeyValues,
                                           const fastllm::GenerationConfig &generationConfig,
                                           const fastllm::LastTokensManager &lastTokens,
                                           std::vector <std::vector <float>*> *retLogits) {
        printf("Unsupport forward batch.\n");
        exit(0);
    }

    std::vector<int> basellm::ForwardBatch(int batch, const fastllm::Data &inputIds,
                                           const std::vector<Data *> &attentionMask,
                          const std::vector<Data *> &positionIds, const std::vector<int> &seqLens,
                          std::vector<std::pair<Data *, Data *>> &pastKeyValues,
                          const std::vector<GenerationConfig> &generationConfigs,
                          const fastllm::LastTokensManager &lastTokens,
                          std::vector <std::vector <float>*> *logits) {
        std::vector <int> ret;
        int cur = 0;
        for (int i = 0; i < batch; i++) {
            std::vector<std::pair<Data, Data> > curKV;
            curKV.resize(this->block_cnt);
            for (int j = 0; j < this->block_cnt; j++) {
                Mul(*pastKeyValues[i * this->block_cnt + j].first, 1.0, curKV[j].first);
                Mul(*pastKeyValues[i * this->block_cnt + j].second, 1.0, curKV[j].second);
            }
            Data curInput;
            Split(inputIds, 1, cur, cur + seqLens[i], curInput);
            LastTokensManager curTokens;
            curTokens.units.push_back(lastTokens.units[i]);
            ret.push_back(this->Forward(curInput, *attentionMask[i], *positionIds[i], curKV, generationConfigs[i], curTokens));
            for (int j = 0; j < this->block_cnt; j++) {
                Mul(curKV[j].first, 1.0, *pastKeyValues[i * this->block_cnt + j].first);
                Mul(curKV[j].second, 1.0, *pastKeyValues[i * this->block_cnt + j].second);
            }
        }
        return ret;
    }

    std::vector<int> basellm::ForwardV2(int batch, const fastllm::Data &inputIds,
                                           const std::vector<Data *> &attentionMask,
                          const std::vector<Data *> &positionIds, const std::vector<int> &seqLens,
                          std::vector<std::pair<Data *, Data *>> &pastKeyValues,
                          const std::vector<GenerationConfig> &generationConfigs,
                          const fastllm::LastTokensManager &lastTokens,
                          std::vector <std::vector <float>*> *logits) {
        printf("Unsupport ForwardV2.\n");
        exit(0);
    }

    std::vector<int> basellm::ForwardGPU(int batch, const fastllm::Data &inputIds,
                                           const std::vector<Data *> &attentionMask,
                          const std::vector<Data *> &positionIds, const std::vector<int> &seqLens,
                          std::vector<std::pair<Data *, Data *>> &pastKeyValues,
                          const std::vector<GenerationConfig> &generationConfigs,
                          const fastllm::LastTokensManager &lastTokens,
                          std::vector <std::vector <float>*> *logits) {
        return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                         pastKeyValues, generationConfigs, lastTokens, logits);
    }

    std::vector <int> basellm::ForwardMultimodal(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const std::map <std::string, std::vector <Data*> > &multimodalInput,
                const GenerationConfig &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector <std::vector <float>*> *logits) {
        printf("Unsupport multi modal forward.\n");
        exit(0);
    }

    void basellm::NewMainLoop() {
        RunNewMainLoop(false);
    }

    void basellm::GPUMainLoop() {
        RunNewMainLoop(true);
    }

    void basellm::RunNewMainLoop(bool useGPUForward) {
        basellm *model = this;
        int maxTotalLens = 0;
        int totalPages = 0;
        int pagesLimit = 0; // 默认为totalPages的80%，Prefill不会让已用分页超过此限制
        int pageLen = fastllm::GetPageLen();

        int maxBatch = 512;
        if (model->maxBatch > 0) {
            maxBatch = model->maxBatch;
        }
        maxBatch = NormalizeMaxBatchByModelCapability(model, maxBatch);

        int prefillChunkSize = model->GetChunkedPrefillSize();

        // 辅助lambda：释放一个请求占用的所有KV Cache分页，并以allTokens重新初始化为pending prefill状态
        auto releaseAndReinitRequest = [&](ResponseContext *ctx) {
            for (int i = 0; i < model->block_cnt; i++) {
                auto &kvFirst = ctx->pastKeyValues[i].first;
                auto &kvSecond = ctx->pastKeyValues[i].second;
                ReleasePagedCachePages(kvFirst, true);
                ReleasePagedCachePages(kvSecond, true);
            }
            ctx->currentTokens = ctx->allTokens;
            ctx->preTokens = 0;
            ctx->cacheLen = 0;
            ctx->intParams.clear();
        };

        auto getPagedManagerFromCache = [](Data &cache) -> PagedCacheManager* {
            if (cache.multiDeviceData) {
                for (auto &it : cache.multiDeviceDatas) {
                    if (it.second != nullptr && it.second->pagedKVCacheData != nullptr) {
                        return it.second->pagedKVCacheData;
                    }
                }
            }
            return cache.pagedKVCacheData;
        };

        auto findRuntimePagedManager = [&]() -> PagedCacheManager* {
            if (model->kvCacheId >= 0) {
                for (auto &it : model->responseContextDict.dicts) {
                    ResponseContext *ctx = it.second;
                    if (ctx == nullptr || model->kvCacheId >= (int)ctx->pastKeyValues.size()) {
                        continue;
                    }
                    PagedCacheManager *manager =
                        getPagedManagerFromCache(ctx->pastKeyValues[model->kvCacheId].first);
                    if (manager != nullptr) {
                        return manager;
                    }
                }
            }
            return model->GetPagedKVCacheManager(model->kvCacheId, true);
        };

        auto collectDecodePageNeeds =
                [&](const std::vector<ResponseContext*> &contexts) -> std::map<PagedCacheManager*, int> {
            std::map<PagedCacheManager*, int> needs;
            std::function<void(Data&)> addCacheNeed = [&](Data &cache) {
                if (cache.multiDeviceData && !cache.multiDeviceDatas.empty()) {
                    bool usedLocal = false;
                    for (auto &it : cache.multiDeviceDatas) {
                        if (it.second != nullptr) {
                            addCacheNeed(*it.second);
                            usedLocal = true;
                        }
                    }
                    if (usedLocal) {
                        return;
                    }
                }
                if (!cache.isPagedKVCache || cache.pagedKVCacheData == nullptr) {
                    return;
                }
                if (cache.pagedKVCacheData->type != PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE) {
                    return;
                }
                if (cache.pageIndex.empty() || cache.lastPageLen >= cache.pageLen) {
                    needs[cache.pagedKVCacheData]++;
                }
            };
            for (auto *ctx : contexts) {
                if (ctx == nullptr) {
                    continue;
                }
                for (int i = 0; i < model->block_cnt && i < (int)ctx->pastKeyValues.size(); i++) {
                    addCacheNeed(ctx->pastKeyValues[i].first);
                    addCacheNeed(ctx->pastKeyValues[i].second);
                }
            }
            return needs;
        };

        auto hasPagedManagerShortage = [](const std::map<PagedCacheManager*, int> &needs) -> bool {
            for (auto &it : needs) {
                PagedCacheManager *manager = it.first;
                if (manager == nullptr || it.second <= 0) {
                    continue;
                }
                int freePages = 0;
                {
                    std::lock_guard<std::mutex> guard(manager->pageIndexLocker);
                    freePages = manager->FreePageCount();
                }
                if (freePages < it.second) {
                    return true;
                }
            }
            return false;
        };

        struct PageNeedState {
            std::map<PagedCacheManager*, int> needs;
            bool impossible = false;
        };

        auto mergePageNeeds = [](std::map<PagedCacheManager*, int> &dst,
                                 const std::map<PagedCacheManager*, int> &src) {
            for (auto &it : src) {
                if (it.first != nullptr && it.second > 0) {
                    dst[it.first] += it.second;
                }
            }
        };

        auto addManagerPageNeed = [](PagedCacheManager *manager, int currentTokens,
                                     int currentPages, int appendTokens,
                                     PageNeedState &state) {
            if (manager == nullptr || appendTokens <= 0 ||
                manager->type != PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE ||
                manager->pageLen <= 0) {
                return;
            }
            currentTokens = std::max(0, currentTokens);
            currentPages = std::max(0, currentPages);
            int totalTokens = currentTokens + appendTokens;
            int totalPages = (totalTokens + manager->pageLen - 1) / manager->pageLen;
            if (totalPages > manager->maxPages) {
                state.impossible = true;
                return;
            }
            int newPages = totalPages - currentPages;
            if (newPages > 0) {
                state.needs[manager] += newPages;
            }
        };

        auto collectPrefillPageNeeds = [&](ResponseContext *ctx, int appendTokens) -> PageNeedState {
            PageNeedState state;
            if (ctx == nullptr || appendTokens <= 0) {
                return state;
            }

            std::function<bool(Data&, int, PageNeedState&)> addExistingCacheNeed =
                    [&](Data &cache, int tokens, PageNeedState &out) -> bool {
                if (cache.multiDeviceData && !cache.multiDeviceDatas.empty()) {
                    bool usedLocal = false;
                    for (auto &it : cache.multiDeviceDatas) {
                        if (it.second != nullptr) {
                            usedLocal = addExistingCacheNeed(*it.second, tokens, out) || usedLocal;
                        }
                    }
                    if (usedLocal) {
                        return true;
                    }
                }
                if (!cache.isPagedKVCache || cache.pagedKVCacheData == nullptr) {
                    return false;
                }
                int currentPages = (int)cache.pageIndex.size();
                int cachePageLen = cache.pageLen > 0 ? cache.pageLen : cache.pagedKVCacheData->pageLen;
                int currentTokens = currentPages > 0 ?
                        (currentPages - 1) * cachePageLen + cache.lastPageLen : 0;
                addManagerPageNeed(cache.pagedKVCacheData, currentTokens, currentPages, tokens, out);
                return true;
            };

            for (int li = 0; li < model->block_cnt && li < (int)ctx->pastKeyValues.size(); li++) {
                for (int keyFlag = 0; keyFlag < 2; keyFlag++) {
                    bool isKey = keyFlag == 0;
                    Data &cache = isKey ? ctx->pastKeyValues[li].first : ctx->pastKeyValues[li].second;
                    if (addExistingCacheNeed(cache, appendTokens, state)) {
                        continue;
                    }
                    auto refs = model->GetPagedKVCacheManagers(li, isKey);
                    for (auto &ref : refs) {
                        addManagerPageNeed(ref.second, 0, 0, appendTokens, state);
                    }
                }
            }
            return state;
        };

        auto *pcm = model->GetPagedKVCacheManager(model->kvCacheId, true);
        if (pcm != nullptr) {
            totalPages = pcm->maxPages;
            pageLen = pcm->pageLen;
            maxTotalLens = totalPages * pageLen;
            pagesLimit = totalPages * 4 / 5;
            maxBatch = std::max(1, std::min(maxBatch, maxTotalLens / 128));
            model->tokensLimit = maxTotalLens;
            model->promptLimit = pagesLimit * pageLen;
            if (model->verbose) {
                printf("Fastllm KV Cache Token limit: %d tokens (totalPages=%d, pageLen=%d).\n", maxTotalLens, totalPages, pageLen);
                printf("Fastllm AddPrefill Pages limit: %d pages (80%% of %d).\n", pagesLimit, totalPages);
                printf("Fastllm Batch limit: %d.\n", maxBatch);
            }
        } else if (model->tokensLimit > 0) {
            maxTotalLens = model->tokensLimit;
            totalPages = std::max(1, maxTotalLens / pageLen);
            pagesLimit = model->promptLimit > 0 ?
                std::max(1, (model->promptLimit + pageLen - 1) / pageLen) :
                totalPages * 4 / 5;
            maxBatch = std::max(1, std::min(maxBatch, maxTotalLens / 128));
            if (model->promptLimit <= 0) {
                model->promptLimit = pagesLimit * pageLen;
            }
            if (model->verbose) {
                printf("Fastllm KV Cache Token limit: %d tokens (pageLen=%d).\n", maxTotalLens, pageLen);
                printf("Fastllm AddPrefill Pages limit: %d pages.\n", pagesLimit);
                printf("Fastllm Batch limit: %d.\n", maxBatch);
            }
        }

        auto lastRecordTime = std::chrono::system_clock::now();
        long long genTokens = 0;
        const bool canUseFastDecodeInput = (model->model_type == "qwen3");
        const bool printProfile = GetFastllmEnv().printProfile;
        while (true) {
            if (model->isFree) {
                break;
            }
            std::vector <Data*> attentionMasks;
            std::vector <Data*> positionIds;
            std::vector <Data*> ownedAttentionMasks;
            std::vector <Data*> ownedPositionIds;
            std::vector <std::pair <Data*, Data*> > pastKeyValues;
            std::vector <float> ids;
            std::vector <int> seqLens;
            std::vector <int> handles;
            std::vector <GenerationConfig> generationConfigs;
            LastTokensManager tokensManager;
            std::vector <ResponseContext*> tokenContexts;
            std::vector <std::vector <float>* > logits;
            std::vector <float> decodePositionValues;
            std::vector <Data> decodePositionIds;
            static const std::vector<int> decodeScalarDims = {1, 1};
            const int reserveBatch = std::max(1, maxBatch);
            bool selectedNeedLastTokens = false;
            attentionMasks.reserve(reserveBatch);
            positionIds.reserve(reserveBatch);
            ownedAttentionMasks.reserve(reserveBatch);
            ownedPositionIds.reserve(reserveBatch);
            pastKeyValues.reserve(reserveBatch * model->block_cnt);
            ids.reserve(reserveBatch);
            seqLens.reserve(reserveBatch);
            handles.reserve(reserveBatch);
            generationConfigs.reserve(reserveBatch);
            tokenContexts.reserve(reserveBatch);
            logits.reserve(reserveBatch);
            decodePositionValues.reserve(reserveBatch);
            decodePositionIds.reserve(reserveBatch);

            std::unique_lock<std::mutex> dictLocker(model->dictLocker);
            auto &forwardLocker = model->forwardLocker;

            // 单次遍历：处理abort、释放isEnding的KV cache、统计alive、构建orders、检测hasPrefill
            std::vector <int> abortHandles;
            int busyPages = 0, currentActivate = 0;
            bool hasPrefill = false;
            struct DecodeOrder {
                int sortKey;
                int handle;
                ResponseContext *context;
            };
            std::vector <DecodeOrder> orders;
            orders.reserve(model->responseContextDict.dicts.size());
            int limit = (maxTotalLens > 0) ? maxTotalLens : 999999999;

            for (auto &it: model->responseContextDict.dicts) {
                if (it.second->isAbort) {
                    it.second->TryRecordPagedCache(model);
                    abortHandles.push_back(it.first);
                    continue;
                }
                if (it.second->isEnding) {
                    for (int i = 0; i < model->block_cnt; i++) {
                        auto &kvFirst = it.second->pastKeyValues[i].first;
                        auto &kvSecond = it.second->pastKeyValues[i].second;
                        ReleasePagedCachePages(kvFirst);
                        ReleasePagedCachePages(kvSecond);
                    }
                    continue;
                }
                if (it.second->preTokens > 0) {
                    currentActivate++;
                }
                if (it.second->preTokens == 0) {
                    hasPrefill = true;
                }
                orders.push_back({-(int)it.second->currentTokens.size(), it.first, it.second});
            }
            for (auto &it : abortHandles) {
                model->RemoveResponseContext(it);
            }
            sort(orders.begin(), orders.end(), [](const DecodeOrder &a, const DecodeOrder &b) {
                if (a.sortKey != b.sortKey) {
                    return a.sortKey < b.sortKey;
                }
                return a.handle < b.handle;
            });

            // 通过PagedCacheManager获取实际使用的物理页数（复用的页只算一次）
            if (totalPages > 0) {
                PagedCacheManager *probeManager = findRuntimePagedManager();
                if (probeManager != nullptr) {
                    std::lock_guard<std::mutex> guard(probeManager->pageIndexLocker);
                    busyPages = probeManager->maxPages - probeManager->FreePageCount();
                    totalPages = probeManager->maxPages;
                    pageLen = probeManager->pageLen;
                    maxTotalLens = totalPages * pageLen;
                    pagesLimit = totalPages * 4 / 5;
                }
            }

            // 当busyPages未超过pagesLimit时可以开启新的Prefill；超过时只做Decode
            bool canAddPrefill = (pagesLimit > 0) ? (busyPages < pagesLimit) : true;

            for (int isPrompt = 1; isPrompt >= 0; isPrompt--) {
                if (isPrompt == 0 && seqLens.size() > 0) {
                    continue;
                }
                // 超过阈值时跳过Prefill
                if (isPrompt == 1 && !canAddPrefill) {
                    continue;
                }
                // 未超过阈值且有pending的prefill请求时，优先尝试prefill；但如果prefill阶段没收集到任何请求，回退做decode
                if (isPrompt == 0 && hasPrefill && canAddPrefill && seqLens.size() > 0) {
                    continue;
                }

                int prefillTokenCount = 0;
                int curBusyPages = busyPages;
                int pendingNewPages = 0;
                bool selectedMultimodal = false;
                std::map<PagedCacheManager*, int> selectedPrefillPageNeeds;

                for (auto &ii : orders) {
                    ResponseContext *ctx = ii.context;

                    if (ctx->isEnding) {
                        continue;
                    }
                    if (isPrompt && ctx->preTokens != 0) {
                        continue;
                    }
                    if (!isPrompt && ctx->preTokens == 0) {
                        continue;
                    }

                    bool isMultimodal = !ctx->multimodalInput.empty();
                    if (selectedMultimodal && !isMultimodal) {
                        continue;
                    }
                    if (isMultimodal && seqLens.size() > 0) {
                        continue;
                    }

                    int contextTokens = isPrompt ? ctx->cacheLen + (int)ctx->currentTokens.size() :
                                                    (int)ctx->allTokens.size();
                    if ((maxTotalLens > 0 && contextTokens > maxTotalLens) ||
                        contextTokens > model->max_positions) {
                        ctx->isEnding = true;
                        ctx->error = ResponseContextErrorPromptTooLong;
                        // printf("[Handle %d] Finished. Reason: prompt too long (cacheLen=%d, currentTokens=%d, maxTotalLens=%d, max_positions=%d).\n",
                               // it.first, it.second->cacheLen, (int)it.second->currentTokens.size(), maxTotalLens, model->max_positions);
                        continue;
                    }
                    if (!isPrompt && maxTotalLens > 0 && contextTokens >= maxTotalLens) {
                        ctx->isEnding = true;
                        ctx->TryRecordPagedCache(model);
                        continue;
                    }

                    if (isPrompt) {
                        if (ctx->cacheLen == 0) {
                            auto probeRefs = model->GetPagedKVCacheManagers(model->kvCacheId, true);
                            PagedCacheManager *probeManager = nullptr;
                            for (auto &ref : probeRefs) {
                                if (ref.second != nullptr) {
                                    probeManager = ref.second;
                                    break;
                                }
                            }
                            if (probeManager != nullptr) {
                                std::map<PagedCacheManager*, std::vector<int> > queriedPages;
                                auto queryManager = [&](PagedCacheManager *manager) -> std::vector<int>& {
                                    auto it = queriedPages.find(manager);
                                    if (it == queriedPages.end()) {
                                        std::vector<int> pages;
                                        manager->Query(ctx->currentTokens, pages);
                                        it = queriedPages.insert(std::make_pair(manager, std::move(pages))).first;
                                    }
                                    return it->second;
                                };

                                int minCachedPages = (int)queryManager(probeManager).size();
                                if (minCachedPages > 0) {
                                    for (int li = 0; li < model->block_cnt; li++) {
                                        for (int keyFlag = 0; keyFlag < 2; keyFlag++) {
                                            bool isKey = keyFlag == 0;
                                            auto refs = model->GetPagedKVCacheManagers(li, isKey);
                                            for (auto &ref : refs) {
                                                PagedCacheManager *manager = ref.second;
                                                if (manager == nullptr || manager->pageLen != probeManager->pageLen) {
                                                    continue;
                                                }
                                                minCachedPages = std::min(minCachedPages,
                                                        (int)queryManager(manager).size());
                                            }
                                        }
                                    }
                                }

                                if (minCachedPages > 0) {
                                    int cachedLen = minCachedPages * probeManager->pageLen;
                                    if (cachedLen >= (int)ctx->currentTokens.size()) {
                                        minCachedPages--;
                                        cachedLen = minCachedPages * probeManager->pageLen;
                                    }
                                }
                                if (minCachedPages > 0) {
                                    int cachedLen = minCachedPages * probeManager->pageLen;
                                    int extraCachedLen = model->QueryPagedPrefixCacheExtra(ctx, cachedLen);
                                    extraCachedLen = std::max(0, std::min(extraCachedLen, cachedLen));
                                    minCachedPages = extraCachedLen / probeManager->pageLen;
                                }
                                if (minCachedPages > 0) {
                                    int cachedLen = minCachedPages * probeManager->pageLen;
                                    if (!model->RestorePagedPrefixCacheExtra(ctx, cachedLen)) {
                                        continue;
                                    }
                                    auto managerDevice = [](PagedCacheManager *manager) {
                                        if (manager == nullptr) {
                                            return -1;
                                        }
                                        Data *managerData = (Data*)manager;
                                        if (managerData->dataDeviceIds.empty()) {
                                            return -1;
                                        }
                                        return managerData->dataDeviceIds[0];
                                    };
                                    auto restoreOne = [&](Data &cache,
                                                          PagedCacheManager *manager,
                                                          const std::vector<int> &pages) {
                                        if (manager == nullptr || (int)pages.size() < minCachedPages) {
                                            return;
                                        }
                                        Data *managerData = (Data*)manager;
                                        if (managerData->dims.size() < 4) {
                                            return;
                                        }
                                        cache.isKVCache = true;
                                        cache.isPagedKVCache = true;
                                        cache.pagedKVCacheData = manager;
                                        cache.pageLen = manager->pageLen;
                                        cache.pageIndex.assign(pages.begin(), pages.begin() + minCachedPages);
                                        manager->Pick(cache.pageIndex);
                                        cache.lastPageLen = manager->pageLen;
                                        cache.dataType = managerData->dataType;
                                        cache.UpdateUnitSize();
                                        cache.dataDevice = managerData->dataDevice;
                                        cache.dataDeviceIds = managerData->dataDeviceIds;
                                        int numHeads = managerData->dims[2];
                                        int headDim = managerData->dims[3];
                                        cache.Resize({numHeads, minCachedPages * manager->pageLen, headDim});
                                    };
                                    auto restorePagedCache = [&](Data &cache,
                                                                 const std::vector<std::pair<int, PagedCacheManager*> > &refs) {
                                        std::vector<std::pair<int, PagedCacheManager*> > validRefs;
                                        for (auto ref : refs) {
                                            if (ref.second == nullptr || ref.second->pageLen != probeManager->pageLen) {
                                                continue;
                                            }
                                            if ((int)queryManager(ref.second).size() < minCachedPages) {
                                                continue;
                                            }
                                            validRefs.push_back(ref);
                                        }
                                        if (validRefs.empty()) {
                                            return;
                                        }
                                        if (validRefs.size() == 1) {
                                            restoreOne(cache, validRefs[0].second, queryManager(validRefs[0].second));
                                            return;
                                        }

                                        cache.multiDeviceData = true;
                                        cache.dataDevice = DataDevice::CUDA;
                                        cache.dataDeviceIds.clear();
                                        cache.isKVCache = true;
                                        cache.isPagedKVCache = true;
                                        Data *firstLocal = nullptr;
                                        for (auto &ref : validRefs) {
                                            int device = ref.first >= 0 ? ref.first : managerDevice(ref.second);
                                            if (device < 0) {
                                                continue;
                                            }
                                            cache.dataDeviceIds.push_back(device);
                                            Data *managerData = (Data*)ref.second;
                                            Data *&local = cache.multiDeviceDatas[device];
                                            if (local == nullptr) {
                                                local = new Data(managerData->dataType);
                                                local->SetKVCache();
                                                local->cacheUid = cache.cacheUid;
                                            }
                                            restoreOne(*local, ref.second, queryManager(ref.second));
                                            if (firstLocal == nullptr) {
                                                firstLocal = local;
                                            }
                                        }
                                        if (firstLocal == nullptr) {
                                            cache.multiDeviceData = false;
                                            cache.isPagedKVCache = false;
                                            cache.pagedKVCacheData = nullptr;
                                            cache.pageIndex.clear();
                                            return;
                                        }
                                        cache.dataType = firstLocal->dataType;
                                        cache.UpdateUnitSize();
                                        cache.cudaData = nullptr;
                                        cache.pageLen = firstLocal->pageLen;
                                        cache.pageIndex = firstLocal->pageIndex;
                                        cache.lastPageLen = firstLocal->lastPageLen;
                                        cache.pagedKVCacheData = firstLocal->pagedKVCacheData;
                                        cache.dims = firstLocal->dims;
                                    };
                                    for (int li = 0; li < model->block_cnt; li++) {
                                        auto &kvFirst = ctx->pastKeyValues[li].first;
                                        auto &kvSecond = ctx->pastKeyValues[li].second;
                                        restorePagedCache(kvFirst, model->GetPagedKVCacheManagers(li, true));
                                        restorePagedCache(kvSecond, model->GetPagedKVCacheManagers(li, false));
                                    }
                                    ctx->currentTokens.erase(ctx->currentTokens.begin(), ctx->currentTokens.begin() + cachedLen);
                                    ctx->cacheLen = cachedLen;
                                    {
                                        std::lock_guard<std::mutex> guard(probeManager->pageIndexLocker);
                                        curBusyPages = probeManager->maxPages - probeManager->FreePageCount() + pendingNewPages;
                                    }
                                }
                            }
                        }

                        if (currentActivate + (int)seqLens.size() >= maxBatch) {
                            continue;
                        }

                        int thisLen = (int)ctx->currentTokens.size();
                        int thisPages = (thisLen + pageLen - 1) / pageLen;

                        PageNeedState pageNeed = collectPrefillPageNeeds(ctx, thisLen);
                        if (pageNeed.impossible) {
                            ctx->isEnding = true;
                            ctx->error = ResponseContextErrorPromptTooLong;
                            continue;
                        }
                        std::map<PagedCacheManager*, int> combinedPrefillPageNeeds = selectedPrefillPageNeeds;
                        mergePageNeeds(combinedPrefillPageNeeds, pageNeed.needs);
                        if (hasPagedManagerShortage(combinedPrefillPageNeeds)) {
                            continue;
                        }

                        // Prefill后已用分页不能超过pagesLimit（除非单个请求就超过了）
                        if (pagesLimit > 0 && curBusyPages + thisPages > pagesLimit) {
                            bool noActiveRequests = currentActivate == 0 && seqLens.empty();
                            // pagesLimit is a soft prefill throttle. Do not let cached or
                            // stale page accounting leave pending requests unscheduled forever.
                            if (!noActiveRequests) {
                                if (seqLens.size() > 0 || thisPages <= pagesLimit) {
                                    continue;
                                }
                                if (currentActivate > 0) {
                                    continue;
                                }
                            }
                        }

                        if (thisLen > prefillChunkSize) {
                            if (seqLens.size() > 0) {
                                continue;
                            }
                        } else {
                            if (prefillTokenCount + thisLen > prefillChunkSize && seqLens.size() > 0) {
                                continue;
                            }
                        }
                        prefillTokenCount += thisLen;
                        curBusyPages += thisPages;
                        pendingNewPages += thisPages;
                        selectedPrefillPageNeeds.swap(combinedPrefillPageNeeds);
                        currentActivate++;
                    } else {
                        // Decode阶段：不在这里限制分页，由后续驱逐逻辑统一处理
                    }

                    generationConfigs.push_back(ctx->generationConfig);
                    model->PrepareToolCallConstraint(ctx, generationConfigs.back());
                    bool ctxNeedRepeatPenalty = NeedRepeatPenalty(generationConfigs.back());
                    selectedNeedLastTokens |= ctxNeedRepeatPenalty ||
                                              (generationConfigs.back().output_logits &&
                                               !generationConfigs.back().IsSimpleGreedy()) ||
                                              !generationConfigs.back().tool_call_allowed_token_ids.empty();
                    logits.push_back(CreatePendingResultLogits(ctx->generationConfig));

                    tokenContexts.push_back(ctx);
                    handles.push_back(ii.handle);
                    if (isMultimodal) {
                        selectedMultimodal = true;
                    }

                    bool fastDecodeInput = canUseFastDecodeInput && !isPrompt && !isMultimodal && ctx->currentTokens.size() == 1;
                    if (fastDecodeInput) {
                        ids.push_back((float)ctx->currentTokens[0]);
                        seqLens.push_back(1);
                        attentionMasks.push_back(nullptr);

                        float position = ctx->allTokens.empty() ? 0.0f : (float)(ctx->allTokens.size() - 1);
                        decodePositionValues.push_back(position);
                        decodePositionIds.emplace_back(DataType::FLOAT32, decodeScalarDims,
                                                       DataDevice::CPU, (void*)&decodePositionValues.back());
                        positionIds.push_back(&decodePositionIds.back());
                        ctx->preTokens += 1;
                    } else {
                        if (ctx->preTokens == 0) {
                            ctx->intParams["add_special_tokens"] = ctx->cacheLen > 0 ? false : ctx->generationConfig.add_special_tokens;
                            ctx->intParams["promptLen"] = ctx->cacheLen + ctx->currentTokens.size();
                            ctx->intParams["index"] = 0;
                        } else {
                            ctx->intParams["index"]++;
                        }
                        Data inputIds, attentionMask, curPositionIds;
                        std::vector<std::vector<float> > tokens;
                        tokens.resize(1);
                        tokens[0].reserve(ctx->currentTokens.size());
                        for (int i: ctx->currentTokens) {
                            tokens[0].push_back(i);
                        }
                        model->FillLLMInputs(tokens, ctx->intParams, inputIds, attentionMask, curPositionIds);
                        ToDataType(attentionMask, model->dataType);

                        seqLens.push_back(inputIds.Count(0));
                        for (int i = 0; i < inputIds.Count(0); i++) {
                            ids.push_back(((float *) inputIds.cpuData)[i]);
                        }
                        if (attentionMask.dims.size() == 0) {
                            attentionMasks.push_back(nullptr);
                        } else {
                            attentionMasks.push_back(new Data());
                            ownedAttentionMasks.push_back(attentionMasks.back());
                            attentionMask.ToDevice(DataDevice::CPU);
                            attentionMasks.back()->CopyFrom(attentionMask);
                        }
                        if (curPositionIds.dims.size() == 0) {
                            positionIds.push_back(nullptr);
                        } else {
                            positionIds.push_back(new Data());
                            ownedPositionIds.push_back(positionIds.back());
                            positionIds.back()->CopyFrom(curPositionIds);
                        }
                        ctx->preTokens += seqLens.back();
                    }
                    for (int i = 0; i < model->block_cnt; i++) {
                        pastKeyValues.push_back(std::make_pair(&ctx->pastKeyValues[i].first,
                                                               &ctx->pastKeyValues[i].second));
                    }

                    if (selectedMultimodal) {
                        break;
                    }
                    if (seqLens.size() >= maxBatch || (totalPages > 0 && curBusyPages >= totalPages)) {
                        break;
                    }
                }
            }

            if (selectedNeedLastTokens) {
                tokensManager.units.reserve(tokenContexts.size());
                for (auto *ctx : tokenContexts) {
                    tokensManager.units.push_back(ctx->tokens);
                }
            }

            // Decode阶段：检查空闲分页是否足够，不够时释放资源
            if (seqLens.size() > 0 && seqLens[0] == 1) {
                auto pageNeeds = collectDecodePageNeeds(tokenContexts);
                if (!pageNeeds.empty()) {
                    while (hasPagedManagerShortage(pageNeeds)) {
                        // 空闲分页不够，从本轮decode批次中选择上下文最长的请求驱逐
                        int maxLen = -1, evictIdx = -1;
                        for (int i = 0; i < (int)handles.size(); i++) {
                            auto &ctx = *tokenContexts[i];
                            auto &kvFirst = ctx.pastKeyValues[model->kvCacheId].first;
                            if (kvFirst.pageIndex.size() > 0) {
                                int curLen = (kvFirst.pageIndex.size() - 1) * kvFirst.pageLen + kvFirst.lastPageLen;
                                if (curLen > maxLen) {
                                    maxLen = curLen;
                                    evictIdx = i;
                                }
                            }
                        }
                        if (evictIdx == -1) {
                            break;
                        }
                        int evictHandle = handles[evictIdx];
                        auto *evictCtx = model->responseContextDict.dicts[evictHandle];
                        int evictedPages = (int)evictCtx->pastKeyValues[model->kvCacheId].first.pageIndex.size();
                        // printf("[Handle %d] Evicting for recompute: releasing %d pages (contextLen=%d).\n", evictHandle, evictedPages, maxLen);
                        releaseAndReinitRequest(evictCtx);

                        // 从本轮decode批次中移除被驱逐的请求
                        handles.erase(handles.begin() + evictIdx);
                        tokenContexts.erase(tokenContexts.begin() + evictIdx);
                        seqLens.erase(seqLens.begin() + evictIdx);
                        generationConfigs.erase(generationConfigs.begin() + evictIdx);
                        if (!tokensManager.units.empty()) {
                            tokensManager.units.erase(tokensManager.units.begin() + evictIdx);
                        }
                        if (logits[evictIdx] != nullptr) {
                            delete logits[evictIdx];
                        }
                        logits.erase(logits.begin() + evictIdx);
                        attentionMasks.erase(attentionMasks.begin() + evictIdx);
                        positionIds.erase(positionIds.begin() + evictIdx);
                        pastKeyValues.erase(pastKeyValues.begin() + evictIdx * model->block_cnt,
                                            pastKeyValues.begin() + (evictIdx + 1) * model->block_cnt);

                        // 重新统计所有层 K/V manager 的需求。CUDA graph 会在进入
                        // forward 前为所有层预分配 page，单看代表层不够。
                        pageNeeds = collectDecodePageNeeds(tokenContexts);
                    }
                    if (!pageNeeds.empty() && !hasPagedManagerShortage(pageNeeds)) {
                        // 重新计算ids
                        ids.clear();
                        for (int i = 0; i < (int)tokenContexts.size(); i++) {
                            auto &ctx = *tokenContexts[i];
                            for (int t : ctx.currentTokens) {
                                ids.push_back((float)t);
                            }
                        }
                    } else if (hasPagedManagerShortage(pageNeeds)) {
                        for (auto *ctx : tokenContexts) {
                            if (ctx != nullptr) {
                                releaseAndReinitRequest(ctx);
                            }
                        }
                        handles.clear();
                        tokenContexts.clear();
                        seqLens.clear();
                        generationConfigs.clear();
                        if (!tokensManager.units.empty()) {
                            tokensManager.units.clear();
                        }
                        ReleasePendingResultLogits(logits);
                        logits.clear();
                        attentionMasks.clear();
                        positionIds.clear();
                        pastKeyValues.clear();
                        ids.clear();
                    }
                }
                if (handles.empty()) {
                    seqLens.clear();
                }
            }

            if (seqLens.size() > 0) {
                ResponseContext *singleContext = nullptr;
                bool isSingleMultimodal = false;
                if (handles.size() == 1) {
                    auto contextIt = model->responseContextDict.dicts.find(handles[0]);
                    if (contextIt != model->responseContextDict.dicts.end()) {
                        singleContext = contextIt->second;
                        isSingleMultimodal = !singleContext->multimodalInput.empty();
                    }
                }
                dictLocker.unlock();
                forwardLocker.lock();
#ifdef USE_CUDA
                // FastllmCudaClearBigBuffer();
#endif
                Data inputIds = Data(DataType::FLOAT32, {1, (int) ids.size()}, ids);
                std::vector<int> ret;
                std::chrono::system_clock::time_point profileStartTime;
                if (printProfile) {
                    profileStartTime = std::chrono::system_clock::now();
                    ClearProfiler();
                }
                if (isSingleMultimodal) {
                    ret = model->ForwardMultimodal(
                        inputIds,
                        attentionMasks[0] == nullptr ? Data() : *attentionMasks[0],
                        positionIds[0] == nullptr ? Data() : *positionIds[0],
                        singleContext->pastKeyValues,
                        singleContext->multimodalInput,
                        generationConfigs[0],
                        tokensManager,
                        &logits
                    );
                } else if (seqLens.size() == 1 && seqLens[0] > prefillChunkSize) {
                    int len = seqLens[0];
                    std::vector <std::pair <Data, Data> > *pastKeyValue1;
                    dictLocker.lock();
                    pastKeyValue1 = &model->responseContextDict.dicts[handles[0]]->pastKeyValues;
                    dictLocker.unlock();
                    auto prefillStartTime = std::chrono::system_clock::now();
                    for (int st = 0; st < len; ) {
                        int curLen = std::min(prefillChunkSize, len - st);
                        auto chunkStartTime = std::chrono::system_clock::now();
                        Data curInput, curPositionIds;
                        Split(inputIds, 1, st, st + curLen, curInput);
                        {
                            curPositionIds.dataType = positionIds[0]->dataType;
                            curPositionIds.Resize({1, curLen});
                            curPositionIds.Allocate();
                            int unitSize = curPositionIds.unitSize;
                            memcpy(curPositionIds.cpuData,
                                   positionIds[0]->cpuData + st * unitSize,
                                   curLen * unitSize);
                        }

                        std::vector <int> curSeqLens = {curLen};
                        std::vector <Data*> curAttentionMasks = {nullptr};
                        std::vector <Data*> curPositionIdsVec = {&curPositionIds};
                        std::vector <std::pair <Data*, Data*> > curPastKeyValues;
                        for (int i = 0; i < model->block_cnt; i++) {
                            curPastKeyValues.push_back(std::make_pair(&(*pastKeyValue1)[i].first,
                                                                      &(*pastKeyValue1)[i].second));
                        }
                        if (useGPUForward) {
                            ret = model->ForwardGPU(1, curInput, curAttentionMasks,
                                                     curPositionIdsVec, curSeqLens, curPastKeyValues, generationConfigs,
                                                     tokensManager, &logits);
                        } else {
                            ret = model->ForwardV2(1, curInput, curAttentionMasks,
                                                   curPositionIdsVec, curSeqLens, curPastKeyValues, generationConfigs,
                                                   tokensManager, &logits);
                        }
                        st += curLen;
                        if (st < len) {
                            dictLocker.lock();
                            auto contextIt = model->responseContextDict.dicts.find(handles[0]);
                            if (contextIt != model->responseContextDict.dicts.end() &&
                                (int)contextIt->second->allTokens.size() >= pageLen) {
                                contextIt->second->TryRecordPagedCache(model);
                            }
                            dictLocker.unlock();
                        }
                        if (model->verbose) {
                            auto chunkEndTime = std::chrono::system_clock::now();
                            float chunkSpend = GetSpan(chunkStartTime, chunkEndTime);
                            float totalSpend = GetSpan(prefillStartTime, chunkEndTime);
                            float chunkSpeed = chunkSpend > 0 ? curLen / chunkSpend : 0;
                            float avgSpeed = totalSpend > 0 ? st / totalSpend : 0;
                            printf("[Prompt] Long Prefill ... (%d/%d, %d%%). Speed: %f tokens / s.\n",
                                   st, len, st * 100 / len, chunkSpeed);
                        }
                    }
                } else {
                    auto batchStartTime = std::chrono::system_clock::now();
                    if (useGPUForward) {
                        ret = model->ForwardGPU(seqLens.size(), inputIds, attentionMasks,
                                                positionIds, seqLens, pastKeyValues, generationConfigs,
                                                tokensManager, &logits);
                    } else {
                        ret = model->ForwardV2(seqLens.size(), inputIds, attentionMasks,
                                               positionIds, seqLens, pastKeyValues, generationConfigs,
                                               tokensManager, &logits);
                    }
                    if (model->verbose) {
                        int prefillTokens = 0;
                        for (int i = 0; i < seqLens.size(); i++) {
                            if (seqLens[i] > 1) {
                                prefillTokens += seqLens[i];
                            }
                        }
                        if (prefillTokens > 0) {
                            auto batchEndTime = std::chrono::system_clock::now();
                            float batchSpend = GetSpan(batchStartTime, batchEndTime);
                            float prefillSpeed = batchSpend > 0 ? prefillTokens / batchSpend : 0;
                            printf("[Prompt] %d Tokens. Speed: %f tokens / s.\n", prefillTokens, prefillSpeed);
                        }
                    }
                }
                if (printProfile) {
                    PrintLoopProfile("new", seqLens, (int)ret.size(), profileStartTime);
                }

                forwardLocker.unlock();
                dictLocker.lock();

                // Prefill完成后立即Record，使其他请求可以尽早命中Prefix Cache
                for (int i = 0; i < (int)handles.size(); i++) {
                    if (seqLens[i] > 1) {
                        auto &ctx = *model->responseContextDict.dicts[handles[i]];
                        if ((int)ctx.allTokens.size() >= pageLen) {
                            ctx.TryRecordPagedCache(model);
                        }
                    }
                }

                if (model->verbose) {
                    genTokens += seqLens.size();
                    auto nowTime = std::chrono::system_clock::now();
                    float spend = GetSpan(lastRecordTime, nowTime);
                    if (spend > 1) {
                        int logPending = (int)orders.size() - currentActivate;
                        float kvUsage = totalPages > 0 ? busyPages * 100.0f / totalPages : 0;
                        printf("[Decode] alive = %d, pending = %d, context usages: %.1f%%, Speed: %f tokens / s.\n", currentActivate, logPending, kvUsage, (float)genTokens / spend);
                        lastRecordTime = nowTime;
                        genTokens = 0;
                    }
                }

                for (int i = 0; i < handles.size(); i++) {
                    ResponseContext *ctx = tokenContexts[i];
                    int curRet = ret[i];
                    if (curRet == model->eos_token_id || model->eos_token_ids.find(curRet) != model->eos_token_ids.end()) {
                        ctx->isEnding = true;
                        ctx->TryRecordPagedCache(model);
                        // printf("[Handle %d] Finished. Reason: eos token (token_id=%d), total tokens: %d.\n", handles[i], curRet, it.second->curTokens);
                    } else {
                        auto itStopTk = ctx->generationConfig.stop_token_ids.find(curRet);
                        if (itStopTk != ctx->generationConfig.stop_token_ids.end()) {
                            ctx->isEnding = true;
                            ctx->TryRecordPagedCache(model);
                            // printf("[Handle %d] Finished. Reason: stop token (token_id=%d), total tokens: %d.\n", handles[i], curRet, it.second->curTokens);
                        }
                    }
                    if (ctx->isEnding == false) {
                        model->UpdateToolCallConstraintState(ctx, curRet);
                        if (ctx->currentTokens.size() == 1) {
                            ctx->currentTokens[0] = curRet;
                        } else {
                            ctx->currentTokens.assign(1, curRet);
                        }
                        ctx->resultTokenQueue.push(curRet);
                        QueueGeneratedResultLogits(ctx, logits, i);
                        ctx->allTokens.push_back(curRet);
                        if (NeedRepeatPenalty(ctx->generationConfig)) {
                            ctx->tokens.Push(curRet);
                        }
                        ctx->curTokens++;
                        if (ctx->curTokens == ctx->generationConfig.output_token_limit) {
                            ctx->isEnding = true;
                            ctx->TryRecordPagedCache(model);
                            // printf("[Handle %d] Finished. Reason: output token limit reached (curTokens=%d, limit=%d).\n",
                                   // handles[i], it.second->curTokens, it.second->generationConfig.output_token_limit);
                        } else if ((maxTotalLens > 0 && (int)ctx->allTokens.size() >= maxTotalLens) ||
                                   ctx->allTokens.size() >= model->max_positions) {
                            ctx->isEnding = true;
                            ctx->TryRecordPagedCache(model);
                            // printf("[Handle %d] Finished. Reason: max positions reached (allTokens=%d, max_positions=%d).\n",
                                   //handles[i], (int)it.second->allTokens.size(), model->max_positions);
                        }
                    }
                }
                ReleasePendingResultLogits(logits);
            } else {
                // 没有任何请求可以调度时，等待新请求
            }

            for (int i = 0; i < ownedAttentionMasks.size(); i++) {
                delete ownedAttentionMasks[i];
            }
            for (int i = 0; i < ownedPositionIds.size(); i++) {
                delete ownedPositionIds[i];
            }

            if (seqLens.size() == 0) {
                if (!orders.empty()) {
                    model->dictCV.wait_for(dictLocker, std::chrono::milliseconds(10));
                } else {
                    model->dictCV.wait(dictLocker);
                }
            }
        }
    }

    int basellm::LaunchResponseTokens(const std::vector<int> &inputTokens,
                                      const fastllm::GenerationConfig &generationConfig,
                                      const std::map <std::string, std::vector <Data*> > &multimodalInput) {
        mainLoopLocker.lock();
        if (mainLoop == nullptr) {
            if (mainLoop == nullptr) {
                if (this->UseModelSpecificScheduler()) {
                    mainLoop = new std::thread([](basellm *model) {
                        model->RunModelSpecificScheduler();
                    }, this);
                } else {
                    bool useNewEngine = this->use_new_engine;
                    const char *envVal = std::getenv("USE_OLD_ENGINE");
                    if (envVal != nullptr) {
                        std::string val(envVal);
                        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
                        if (val == "on" || val == "1") {
                            useNewEngine = false;
                        }
                    }
                    if (useNewEngine) {
                        if (IsPureGpuMode(this)) {
                            mainLoop = new std::thread([](basellm *model) {
                                model->GPUMainLoop();
                            }, this);
                        } else {
                            mainLoop = new std::thread([](basellm *model) {
                                model->NewMainLoop();
                            }, this);
                        }
                    } else {
                mainLoop = new std::thread([](basellm *model) {
                    long long kvCacheLimit = 16LL << 30;
#ifdef USE_CUDA
                    auto freeSizes = FastllmCudaGetFreeSizes();
                    auto dmap = GetDeviceMap();
                    std::set <int> deviceIds;
                    std::map <int, int> ratios;
                    for (auto &it : dmap) {
                        if (StartWith(it.first, "cuda")) {
                            for (int id : ParseDeviceIds(it.first, "cuda", ratios)) {
                                deviceIds.insert(id);
                            }
                        }
                    }
                    if (deviceIds.size() == 0) {
                        deviceIds.insert(0);
                    }
                    kvCacheLimit = 0;
                    for (int id : deviceIds) {
                        if (id < freeSizes.size()) {
                            kvCacheLimit += std::max(freeSizes[id] * 3 / 4, freeSizes[id] - (2LL << 30));
                        }
                    }
                    if (kvCacheLimit == 0) {
                        kvCacheLimit = 16LL << 30;
                    }
#endif
                    if (model->kvCacheLimit > 0) {
                        kvCacheLimit = model->kvCacheLimit;
                    }

                    int maxTotalLens = kvCacheLimit / 1024 / 1024;
                    if (model->elementsInKVCachePerToken > 0) {
                        long long bytesPerToken = GetDataBytes(model->kvCacheDataType, 1, model->elementsInKVCachePerToken);
                        if (bytesPerToken > 0) {
                            maxTotalLens = kvCacheLimit / bytesPerToken;
                        }
                    } else {
                        maxTotalLens = kvCacheLimit / 1024 / 1024;
                    }
                    if (model->tokensLimit > 0) {
                        maxTotalLens = model->tokensLimit;
                    }

                    int maxBatch = std::max(1, std::min(512, maxTotalLens / 128));
                    if (model->maxBatch > 0) {
                        maxBatch = model->maxBatch;
                    }
                    maxBatch = NormalizeMaxBatchByModelCapability(model, maxBatch);

                    model->tokensLimit = maxTotalLens;
                    int limit = maxTotalLens;
                    model->promptLimit = limit * 3 / 4;

                    if (model->verbose) {
                        printf("Fastllm KV Cache Limit: %f MB.\n", (double)kvCacheLimit / 1e6);
                        printf("Fastllm KV Cache Token limit: %d tokens.\n", maxTotalLens);
                        printf("Fastllm Prompt Token limit: %d tokens.\n", std::min(model->max_positions, model->promptLimit));
                        printf("Fastllm Batch limit: %d.\n", maxBatch);
                    }

                    auto lastRecordTime = std::chrono::system_clock::now();
                    long long genTokens = 0;
                    const bool printProfile = GetFastllmEnv().printProfile;
                    while (true) {
                        if (model->isFree) {
                            break;
                        }
                        std::vector <Data*> attentionMasks;
                        std::vector <Data*> positionIds;
                        std::vector <std::pair <Data*, Data*> > pastKeyValues;
                        std::vector <float> ids;
                        std::vector <int> seqLens;
                        std::vector <int> handles;
                        std::vector <GenerationConfig> generationConfigs;
                        LastTokensManager tokensManager;
                        std::vector <std::vector <float>* > logits;

                        std::unique_lock<std::mutex> dictLocker(model->dictLocker);
                        auto &forwardLocker = model->forwardLocker;

                        // 首先把已经abort的请求删除掉
                        std::set <int> abortHandles;
                        for (auto &it: model->responseContextDict.dicts) {
                            if (it.second->isAbort) {
                                it.second->TryRecord(model);
                                abortHandles.insert(it.first);
                            }
                        }
                        for (auto &it : abortHandles) {
                            model->RemoveResponseContext(it);
                        }

                        int limit = maxTotalLens;
                        int promptLimit = model->promptLimit;

                        int lenSum = 0, currentActivate = 0;
                        for (auto &it: model->responseContextDict.dicts) {
                            if (it.second->pastKeyValues[model->kvCacheId].first.expansionDims.size() > 0) {
                                lenSum += it.second->pastKeyValues[model->kvCacheId].first.expansionDims[1];
                                currentActivate++;
                            }
                        }
                        std::vector <std::pair <int, int> > orders;
                        for (auto &it : model->responseContextDict.dicts) {
                            orders.push_back(std::make_pair(-(int)it.second->currentTokens.size(), it.first));
                        }
                        sort(orders.begin(), orders.end());

                        for (int isPrompt = 1; isPrompt >= 0; isPrompt--) {
                            int cnt = 0;
                            if (isPrompt == 0 && seqLens.size() > 0) {
                                continue;
                            }
/*
                            if (lenSum >= promptLimit && isPrompt) {
                                continue;
                            }
*/

                            int currentMaxLen = 0;

                            // for (auto &it: model->responseContextDict.dicts) {
                            for (auto &ii : orders) {
                                auto &it = *model->responseContextDict.dicts.find(ii.second);
                                // if (model->model_struct == "deepseek_v2" && isPrompt && seqLens.size() > 0) {
                                if (isPrompt && seqLens.size() > 0) {
                                    // TODO: multicuda支持多prompt一起推理
                                    continue;
                                }
                                if (isPrompt) {
                                    int alive = 0;
                                    for (auto &it: model->responseContextDict.dicts) {
                                        if (it.second->isEnding) {
                                            continue;
                                        }
                                        if (it.second->pastKeyValues[model->kvCacheId].first.expansionDims.size() > 0) {
                                            alive++;
                                        }
                                    }
                                    if (alive >= maxBatch) {
                                        continue;
                                    }
                                }

                                if (it.second->isEnding) {
                                    continue;
                                }
                                if (isPrompt && it.second->preTokens != 0) {
                                    continue;
                                }
                                if (!isPrompt && it.second->preTokens == 0) {
                                    continue;
                                }

                                if (it.second->cacheLen + it.second->currentTokens.size() > maxTotalLens ||
                                    it.second->cacheLen + it.second->currentTokens.size() > model->max_positions) {
                                    it.second->isEnding = true;
                                    it.second->error = ResponseContextErrorPromptTooLong;
                                    continue;
                                }

                                int outputLimit = it.second->generationConfig.output_token_limit;
                                outputLimit = (outputLimit < 0 ? 128 : outputLimit);
/*
                                if (isPrompt && lenSum + it.second->currentTokens.size() > promptLimit) {
                                    continue;
                                }
*/
                                if (isPrompt && lenSum + it.second->currentTokens.size() + (currentActivate + 1) * 256 > maxTotalLens) {
                                    continue;
                                }

                                if (!isPrompt) {
                                    if (it.second->pastKeyValues[model->kvCacheId].first.isPagedKVCache) {
                                        if (it.second->pastKeyValues[model->kvCacheId].first.pageLen == it.second->pastKeyValues[model->kvCacheId].first.lastPageLen) {
                                            int sur = it.second->generationConfig.output_token_limit - it.second->curTokens;
                                            int predictLen = 256;
                                            if (sur > 0) {
                                                predictLen = std::min(predictLen, ((sur - 1) / 128 + 1) * 128);
                                            }
                                            if (lenSum + predictLen > limit) {
                                                continue;
                                            }
                                            lenSum += predictLen;
                                        }
                                    } else {
                                        if (it.second->pastKeyValues[model->kvCacheId].first.expansionDims[1] == it.second->pastKeyValues[0].first.dims[1]) {
                                            int sur = it.second->generationConfig.output_token_limit - it.second->curTokens;
                                            int predictLen = 256;
                                            if (sur > 0) {
                                                predictLen = std::min(predictLen, ((sur - 1) / 128 + 1) * 128);
                                            }
                                            if (lenSum + predictLen > limit) {
                                                continue;
                                            }
                                            lenSum += predictLen;
                                        }
                                    }
                                } else {
                                    if (it.second->currentTokens.size() * 2 < currentMaxLen) {
                                        continue;
                                    }
                                    currentMaxLen = std::max(currentMaxLen, (int)it.second->currentTokens.size());
                                    lenSum += it.second->currentTokens.size();
                                    currentActivate++;
                                }

                                generationConfigs.push_back(it.second->generationConfig);
                                model->PrepareToolCallConstraint(it.second, generationConfigs.back());
                                logits.push_back(CreatePendingResultLogits(it.second->generationConfig));

                                tokensManager.units.push_back(it.second->tokens);
                                handles.push_back(it.first);

                                if (it.second->preTokens == 0) {
                                    it.second->intParams["add_special_tokens"] = it.second->cacheLen > 0 ? false : it.second->generationConfig.add_special_tokens;
                                    it.second->intParams["promptLen"] = it.second->cacheLen + it.second->currentTokens.size();
                                    it.second->intParams["index"] = 0;
                                } else {
                                    it.second->intParams["index"]++;
                                }
                                Data inputIds, attentionMask, curPositionIds;
                                std::vector<std::vector<float> > tokens;
                                tokens.resize(1);
                                for (int i: it.second->currentTokens) {
                                    tokens[0].push_back(i);
                                }
                                model->FillLLMInputs(tokens, it.second->intParams, inputIds, attentionMask, curPositionIds);
                                ToDataType(attentionMask, model->dataType);

                                seqLens.push_back(inputIds.Count(0));
                                for (int i = 0; i < inputIds.Count(0); i++) {
                                    ids.push_back(((float *) inputIds.cpuData)[i]);
                                }
                                if (attentionMask.dims.size() == 0) {
                                    attentionMasks.push_back(nullptr);
                                } else {
                                    attentionMasks.push_back(new Data());
                                    attentionMask.ToDevice(DataDevice::CPU);
                                    attentionMasks.back()->CopyFrom(attentionMask);
                                }
                                if (curPositionIds.dims.size() == 0) {
                                    positionIds.push_back(nullptr);
                                } else {
                                    positionIds.push_back(new Data());
                                    positionIds.back()->CopyFrom(curPositionIds);
                                }
                                it.second->preTokens += seqLens.back();
                                for (int i = 0; i < model->block_cnt; i++) {
                                    pastKeyValues.push_back(std::make_pair(&it.second->pastKeyValues[i].first,
                                                                           &it.second->pastKeyValues[i].second));
                                }
                                if (isPrompt) {
                                    cnt += it.second->currentTokens.size();

                                    if (cnt > 1024) {
                                        break;
                                    }
                                    // break;
                                }

                                if (seqLens.size() >= maxBatch || lenSum + seqLens.size() * 128 > limit) {
                                    break;
                                }
                            }
                        }
                        if (seqLens.size() > 0) {
                            std::vector <std::pair <Data, Data> > *pastKeyValue1;
                            if (seqLens.size() == 1) {
                                pastKeyValue1 = &model->responseContextDict.dicts[handles[0]]->pastKeyValues;
                            }
                            dictLocker.unlock();
                            forwardLocker.lock();
#ifdef USE_CUDA
                            FastllmCudaClearBigBuffer();
#endif
                            Data inputIds = Data(DataType::FLOAT32, {1, (int) ids.size()}, ids);
                            std::vector<int> ret;
                            std::chrono::system_clock::time_point profileStartTime;
                            if (printProfile) {
                                profileStartTime = std::chrono::system_clock::now();
                                ClearProfiler();
                            }
                            if (seqLens.size() > 1) {
                                if (!model->canDoBatchForward) {
                                    dictLocker.lock();
                                    for (int i = 0; i < handles.size(); i++) {
                                        Data inputIdNow = Data(DataType::FLOAT32, {1, 1}, {ids[i]});
                                        LastTokensManager singleTokens;
                                        singleTokens.units.push_back(tokensManager.units[i]);
                                        ret.push_back(model->Forward(inputIdNow,
                                                            attentionMasks[i] == nullptr ? Data() : *attentionMasks[i],
                                                            *positionIds[i],
                                                            model->responseContextDict.dicts[handles[i]]->pastKeyValues,
                                                            generationConfigs[i], singleTokens, logits[i]));
                                    }
                                    dictLocker.unlock();
                                } else {
                                    ret = model->ForwardBatch(seqLens.size(), inputIds, attentionMasks,
                                                            positionIds, seqLens, pastKeyValues, generationConfigs,
                                                            tokensManager, &logits);
                                }
                            } else {
                                int first, part;
                                first = part = model->GetChunkedPrefillSize();
                                if (seqLens[0] > first) {
                                    int len = seqLens[0];
                                    for (int st = 0; st < len; ) {
                                        if (model->verbose) {
                                            genTokens += seqLens.size();
                                            auto nowTime = std::chrono::system_clock::now();
                                            float spend = GetSpan(lastRecordTime, nowTime);
                                            if (spend > 1) {
                                                printf("Long Prefill ... (%d%%)\n", st * 100 / len);
                                                lastRecordTime = nowTime;
                                            }
                                        }
                                        int curLen = std::min(st == 0 ? first : part, len - st);
                                        Data curInput, curPositionIds;
                                        Split(inputIds, 1, st, st + curLen, curInput);
                                        Split(*positionIds[0], 1, st, st + curLen, curPositionIds);

                                        ret = std::vector <int> {model->Forward(curInput, Data(), curPositionIds,
                                            *pastKeyValue1, generationConfigs[0], tokensManager, logits[0])};
                                        st += curLen;
                                    }
                                } else {
                                    auto context = model->responseContextDict.dicts.begin()->second;
                                    if (context->multimodalInput.size() > 0) {
                                        ret = model->ForwardMultimodal(inputIds,
                                                            attentionMasks[0] == nullptr ? Data() : *attentionMasks[0],
                                                            *positionIds[0], *pastKeyValue1, context->multimodalInput,
                                                           context->generationConfig, tokensManager, &logits);
                                    } else {
                                        ret = std::vector <int> {model->Forward(inputIds,
                                                                        attentionMasks[0] == nullptr ? Data() : *attentionMasks[0],
                                                                        *positionIds[0],
                                                                        *pastKeyValue1, generationConfigs[0], tokensManager, logits[0])};
                                    }
                                }
                            }
                            if (printProfile) {
                                PrintLoopProfile("old", seqLens, (int)ret.size(), profileStartTime);
                            }
                            forwardLocker.unlock();
                            dictLocker.lock();

                            if (model->verbose) {
                                genTokens += seqLens.size();
                                auto nowTime = std::chrono::system_clock::now();
                                float spend = GetSpan(lastRecordTime, nowTime);
                                if (spend > 1) {
                                    int total = 0, alive = 0, aliveLen = 0, pending = 0;
                                    for (auto &it: model->responseContextDict.dicts) {
                                        if (it.second->isEnding) {
                                            continue;
                                        }
                                        if (it.second->pastKeyValues[model->kvCacheId].first.expansionDims.size() > 0) {
                                            alive++;
                                            aliveLen += it.second->pastKeyValues[model->kvCacheId].first.expansionDims[1];
                                        } else {
                                            pending++;
                                        }
                                    }
                                    printf("[Decode] alive = %d, pending = %d, contextLen = %d, Speed: %f tokens / s.\n", alive, pending, aliveLen, (float)genTokens / spend);
                                    lastRecordTime = nowTime;
                                    genTokens = 0;
                                }
                            }

                            for (int i = 0; i < handles.size(); i++) {
                                auto &it = *model->responseContextDict.dicts.find(handles[i]);
                                int curRet = ret[i];
                                if (curRet == model->eos_token_id || model->eos_token_ids.find(curRet) != model->eos_token_ids.end()) {
                                    it.second->isEnding = true;
                                    it.second->TryRecord(model);
                                } else {
                                    auto itStopTk = it.second->generationConfig.stop_token_ids.find(curRet);
                                    if (itStopTk != it.second->generationConfig.stop_token_ids.end()) {
                                        it.second->isEnding = true;
                                        it.second->TryRecord(model);
                                    }
                                }
                                if (it.second->isEnding == false) {
                                    model->UpdateToolCallConstraintState(it.second, curRet);
                                    it.second->currentTokens = std::vector<int>{curRet};
                                    it.second->resultTokenQueue.push(curRet);
                                    QueueGeneratedResultLogits(it.second, logits, i);
                                    it.second->allTokens.push_back(curRet);
                                    it.second->tokens.Push(curRet);
                                    it.second->curTokens++;
                                    if (it.second->curTokens == it.second->generationConfig.output_token_limit
                                        || it.second->allTokens.size() >= model->max_positions) {
                                        it.second->isEnding = true;
                                        it.second->TryRecord(model);
                                    }
                                }
                            }
                            ReleasePendingResultLogits(logits);
                        } else {
                            int maxLen = -1, select = -1;
                            for (auto &it: model->responseContextDict.dicts) {
                                if (it.second->isEnding) {
                                    continue;
                                }
                                if (it.second->pastKeyValues[model->kvCacheId].first.expansionDims.size() > 0) {
                                    int curLen = it.second->pastKeyValues[model->kvCacheId].first.expansionDims[1];
                                    if (curLen > maxLen) {
                                        maxLen = curLen;
                                        select = it.first;
                                    }
                                }
                            }
                            if (select != -1) {
                                model->responseContextDict.dicts[select]->isEnding = true;
                                continue;
                            }
                        }

                        for (int i = 0; i < attentionMasks.size(); i++) {
                            delete attentionMasks[i];
                        }
                        for (int i = 0; i < positionIds.size(); i++) {
                            delete positionIds[i];
                        }

                        if (seqLens.size() == 0) {
                            model->dictCV.wait(dictLocker);
                        }
                    }
                }, this);
                    } // end of else (old engine)
                }
            }
        }
        mainLoopLocker.unlock();

        dictLocker.lock();
        int handleId = responseContextDict.CreateHandle();
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        context->Init(this->block_cnt, this->dataType, this->kvCacheDataType);
        context->currentTokens = inputTokens;
        context->allTokens = inputTokens;
        context->generationConfig = generationConfig;
        context->multimodalInput = multimodalInput;
        context->tokens = LastTokensUnit(generationConfig.last_n);

        bool restoredNativeHistory = this->TryRestoreHistoryCache(context->currentTokens, context->cacheLen);

        auto cache = restoredNativeHistory || !this->UseGenericHistoryCache() ?
                     std::make_pair((PastKVCacheMemory*)nullptr, 0) :
                     pastKVCacheManager.Get(inputTokens);
        if (cache.first != nullptr && cache.second > 0) {
            int len = cache.second;

            forwardLocker.lock();
            for (int i = 0; i < this->block_cnt; i++) {
                if (cache.first->kv[i].first.isLinearAttention) {
                    context->pastKeyValues[i].first.CopyFrom(cache.first->kv[i].first);
                    context->pastKeyValues[i].second.CopyFrom(cache.first->kv[i].second);
                } else {
                    Split(cache.first->kv[i].first, 1, 0, len, context->pastKeyValues[i].first);
                    Split(cache.first->kv[i].second, 1, 0, len, context->pastKeyValues[i].second);
                    auto kdims = context->pastKeyValues[i].first.dims, vdims = context->pastKeyValues[i].second.dims;
                    kdims[1] = ((kdims[1] - 1) / 128 + 1) * 128;
                    vdims[1] = ((vdims[1] - 1) / 128 + 1) * 128;
                    context->pastKeyValues[i].first.Expansion(kdims);
                    context->pastKeyValues[i].second.Expansion(vdims);
                    // context->pastKeyValues[i].first.CopyFrom(cache.first->kv[i].first);
                    // context->pastKeyValues[i].second.CopyFrom(cache.first->kv[i].second);
                }
            }
            forwardLocker.unlock();
            context->currentTokens.erase(context->currentTokens.begin(), context->currentTokens.begin() + len);
            context->cacheLen = len;
        }

        this->OnResponseContextCreated(context);

        dictLocker.unlock();
        dictCV.notify_one();
        return handleId;
    }

    bool basellm::CanFetchResponse(int handleId) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            return true;
        } else {
            return (context->resultTokenQueue.size() > 0 || context->isEnding);
        }
    }

    void basellm::AbortResponse(int handleId) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        ResponseContext *context = responseContextDict.GetHandle(handleId);

        if (context == nullptr) {
            return;
        } else {
            context->isAbort = true;
        }
    }

    int basellm::FetchResponseTokens(int handleId) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            return -1;
        } else {
            while (true) {
                if (context->resultTokenQueue.size() > 0) {
                    int ret = context->resultTokenQueue.front();
                    context->resultTokenQueue.pop();
                    return ret;
                } else {
                    if (context->isEnding) {
                        ResponseContextError err = context->error;
                        RemoveResponseContext(handleId);
                        dictLocker.unlock();
                        dictCV.notify_one();
                        if (err == ResponseContextErrorPromptTooLong) {
                            return -2;
                        }
                        return -1;
                    }
                }
                dictLocker.unlock();
                MySleep(0);
                dictLocker.lock();
            }
        }
    }

    int basellm::FetchResponseLogits(int handleId, std::vector<float> &logits) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            return -1;
        } else {
            while (true) {
                if (context->resultTokenQueue.size() > 0) {
                    int ret = context->resultTokenQueue.front();
                    context->resultTokenQueue.pop();
                    if (!context->resultLogits.empty()) {
                        logits = *context->resultLogits.front();
                        delete context->resultLogits.front();
                        context->resultLogits.pop();
                    }
                    return ret;
                } else {
                    if (context->isEnding) {
                        RemoveResponseContext(handleId);
                        dictLocker.unlock();
                        dictCV.notify_one();
                        return -1;
                    }
                }
                dictLocker.unlock();
                MySleep(0);
                dictLocker.lock();
            }
        }
    }

    void basellm::AddPromptCache(const std::vector <int> &inputTokens) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        auto cache = pastKVCacheManager.Get(inputTokens);
        if (cache.first != nullptr && cache.first->inputToken.size() == inputTokens.size()) {
            return;
        }
        Data inputIds, attentionMask, positionIds;
        std::vector<std::pair<Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->kvCacheDataType), Data(this->kvCacheDataType)));
            pastKeyValues.back().first.SetKVCache();
            pastKeyValues.back().second.SetKVCache();
        }

        int promptLen = inputTokens.size(), index = 0;
        int add_special_tokens = false;
        std::vector <std::vector <float> > fInputTokens;
        fInputTokens.resize(1);
        for (int i = 0; i < inputTokens.size(); i++) {
            fInputTokens[0].push_back(inputTokens[i]);
        }
        FillLLMInputs(fInputTokens, {{"promptLen", promptLen}, {"index", index}, {"add_special_tokens", add_special_tokens}},
                      inputIds, attentionMask, positionIds);
        ToDataType(attentionMask, this->dataType);
        int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        pastKVCacheManager.Record(inputTokens, inputTokens.size(), &pastKeyValues);
    }

    bool basellm::NeedAttentionMask(int qlen, int klen) {
        return true;
    }

    // 根据输入的tokens生成LLM推理的输入
    void basellm::FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                               const std::map <std::string, int> &params,
                               Data &inputIds, Data &attentionMask, Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int index = params.find("index")->second;
        int promptLen = params.find("promptLen")->second;

        if (inputTokens[0].size() > 1) {
            int seqLen = inputTokens[0].size();
            std::vector <float> vpids = std::vector <float> (seqLen, 0);
            for (int i = 0; i < seqLen; i++) {
                vpids[i] = promptLen - seqLen + i;
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, inputTokens[0]));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vpids));

            if (NeedAttentionMask(seqLen, promptLen)) {
                std::vector <float> vmask = std::vector <float> (seqLen * promptLen, 0);
                for (int i = 0; i < seqLen; i++) {
                    vpids[i] = promptLen - seqLen + i;
                    for (int j = i + 1; j < seqLen; j++) {
                        vmask[i * promptLen + (promptLen - seqLen + j)] = 1;
                    }
                }
                attentionMask.CopyFrom(Data(DataType::FLOAT32, {seqLen, promptLen}, vmask));
            } else {
                attentionMask = Data();
            }
        } else {
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, inputTokens[0]));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) promptLen + index - 1}));
        }
    }

    // 根据输入的tokens生成LLM推理的输入
    void basellm::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                     const std::vector<std::map<std::string, int>> &params, fastllm::Data &inputIds,
                                     fastllm::Data &attentionMask, fastllm::Data &positionIds) {
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

    void basellm::SetAdapter(const std::string &name) {
        if (weight.peftDict.find(name) == weight.peftDict.end()) {
            ErrorInFastLLM("Can`t find adapter name: " + name);
        }
        adapterName = name;
    }

    void basellm::DisableAdapter() {
        adapterName = "";
    }

    bool basellm::SetSaveHistoryChat(bool save) {
        this->saveHistoryChat = save;
            return true;
    }

    void basellm::SetMoeExperts(int experts) {
        this->num_experts_per_tok = std::max(1, experts - this->n_shared_experts);
        return;
    }

    void basellm::SetChunkedPrefillSize(int size) {
        this->chunkedPrefillSize = size;
    }

    int basellm::GetChunkedPrefillSize() {
        if (this->chunkedPrefillSize >= 0) {
            return this->chunkedPrefillSize;
        }
        return this->defaultChunkedPrefillSize;
    }

    void basellm::SetDataType(DataType dataType) {
        if (dataType == DataType::FLOAT32) {

        } else if (dataType == DataType::FLOAT16) {
            AssertInFastLLM(this->use_new_engine ||
                            this->model_struct == "chatglm" ||
                            this->model_struct == "llama" ||
                            this->model_struct == "graph" ||
                            this->model_struct == "cogvlm" ||
                            this->model_struct == "deepseek_v2" ||
                            this->model_struct == "deepseek_v4" ||
                            this->model_struct == "qwen3_moe" ||
                            this->model_struct == "minimax_m2" ||
                            this->model_struct == "hunyuan" ||
                            this->model_struct == "ernie4_5" ||
                            this->model_struct == "pangu_moe" ||
                            this->model_struct == "glm4_moe" ||
                            this->model_struct == "qwen3_next" ||
                            this->model_struct == "gemma4",
                            this->model_struct + " doesn't support float16");
        } else if (dataType == DataType::BFLOAT16) {
            AssertInFastLLM(this->use_new_engine ||
                            this->model_struct == "chatglm" ||
                            this->model_struct == "llama" ||
                            this->model_struct == "graph" ||
                            this->model_struct == "cogvlm" ||
                            this->model_struct == "deepseek_v2" ||
                            this->model_struct == "deepseek_v4" ||
                            this->model_struct == "qwen3_moe" ||
                            this->model_struct == "minimax_m2" ||
                            this->model_struct == "hunyuan" ||
                            this->model_struct == "ernie4_5" ||
                            this->model_struct == "pangu_moe" ||
                            this->model_struct == "glm4_moe" ||
                            this->model_struct == "qwen3_next" ||
                            this->model_struct == "gemma4",
                            this->model_struct + " doesn't support bfloat16");
        } else {
            ErrorInFastLLM("SetDataType Error: datatype should be float32, float16 or bfloat16");
        }
        this->dataType = dataType;
        if (!this->useCustomKVCacheDataType) {
            this->kvCacheDataType = dataType;
        }
    }

    void basellm::SetKVCacheDataType(DataType dataType) {
#ifndef USE_CUDA
        if (dataType == DataType::FP8_E4M3) {
            ErrorInFastLLM("SetKVCacheDataType Error: fp8_e4m3 kv cache requires CUDA support.");
        }
#endif
        if (dataType == DataType::FLOAT32 ||
            dataType == DataType::FLOAT16 ||
            dataType == DataType::BFLOAT16 ||
            dataType == DataType::FP8_E4M3) {
            this->kvCacheDataType = dataType;
            this->useCustomKVCacheDataType = true;
        } else {
            ErrorInFastLLM("SetKVCacheDataType Error: datatype should be float32, float16, bfloat16 or fp8_e4m3");
        }
    }

    void basellm::SetMoeAtype(DataType type) {
        if (type == DataType::FLOAT32 || type == DataType::FLOAT16 || type == DataType::BFLOAT16) {
            this->moeAtype = type;
        } else {
            ErrorInFastLLM("SetMoeAtype Error: moe_atype should be float32, float16 or bfloat16");
        }
    }

    void basellm::UpdateRotaryPtr(Data **sinDataPtr, Data **cosDataPtr, const std::string &device) {
        if (this->deviceSinDatas.find(device) == this->deviceSinDatas.end()) {
            this->deviceSinDatas[device] = new Data();
            this->deviceCosDatas[device] = new Data();
            Mul(sinData, 1.0f, *this->deviceSinDatas[device]);
            Mul(cosData, 1.0f, *this->deviceCosDatas[device]);
        }

        *sinDataPtr = this->deviceSinDatas[device];
        *cosDataPtr = this->deviceCosDatas[device];
    }

    JinjaVar ChatMessagesToJinjaVar(const ChatMessages &messages) {
        JinjaVar ret = {{"messages", fastllm::JinjaArray {}}};
        for (auto &message : messages) {
            ret["messages"].arrayValue.push_back({
                {"role", message.first},
                {"content", message.second}
            });
        }
        ret["add_generation_prompt"] = fastllm::JinjaVar{1};
        ret["tools"] = fastllm::JinjaVar{std::vector <JinjaVar>()};
        return ret;
    }

    std::string basellm::ApplyChatTemplate(const ChatMessages &messages) {
        if (this->weight.tokenizer.chatTemplate == "") {
            std::string ret = "";
            std::string user = "";
            int round = 0;
            for (auto &message : messages) {
                if (message.first == "user") {
                    user = message.second;
                } else if (message.first == "assistant") {
                    ret = MakeHistory(ret, round++, user, message.second);
                }
            }
            ret = MakeInput(ret, round, user);
            return ret;
        }
        return ApplyChatTemplate(ChatMessagesToJinjaVar(messages));
    }

    std::vector <int> basellm::ApplyChatTemplateToTokens(const ChatMessages &messages) {
        auto prompt = this->ApplyChatTemplate(messages);
        auto input = this->weight.tokenizer.Encode(prompt);
        std::vector<int> tokens;
        for (int i = 0; i < input.Count(0); i++) {
            tokens.push_back(((float *) input.cpuData)[i]);
        }
        return tokens;
    }

    std::string basellm::ApplyChatTemplate(const JinjaVar &var) {
        AssertInFastLLM(this->weight.tokenizer.chatTemplate != "",
                        "ApplyChatTemplate error: model doesn't has chat_template.");
        JinjaVar local = var;
        for (auto &it : this->weight.tokenizer.tokenizerConfig.object_items()) {
            if (it.first != "messages" && it.second.is_string()) {
                local[it.first] = it.second.string_value();
            } else if (it.first.find_last_of("_token") != std::string::npos && it.second.is_object()) {
                local[it.first] = it.second["content"].string_value();
            }
        }
        JinjaTemplate temp = JinjaTemplate(this->weight.tokenizer.chatTemplate);
        return temp.Apply(local);
    }

    std::vector <int> basellm::ApplyChatTemplateToTokens(const JinjaVar &var) {
        auto prompt = this->ApplyChatTemplate(var);
        auto input = this->weight.tokenizer.Encode(prompt);
        std::vector<int> tokens;
        for (int i = 0; i < input.Count(0); i++) {
            tokens.push_back(((float *) input.cpuData)[i]);
        }
        return tokens;
    }

    static int GetCacheLen(const Data &cache) {
        if (cache.isPagedKVCache) {
            if (cache.pageIndex.empty()) {
                return 0;
            }
            return (cache.pageIndex.size() - 1) * cache.pageLen + cache.lastPageLen;
        }
        if (cache.dims.size() > 1) {
            return cache.dims[1];
        }
        if (cache.expansionDims.size() > 1) {
            return cache.expansionDims[1];
        }
        return 0;
    }

    static int GetTokenGrowingCacheLen(const basellm *model, std::vector <std::pair <Data, Data> > &pastKeyValues) {
        auto tryGetLen = [&](int idx) {
            if (idx < 0 || idx >= (int)pastKeyValues.size()) {
                return 0;
            }
            auto &pastKey = pastKeyValues[idx].first;
            auto &pastValue = pastKeyValues[idx].second;
            if (pastKey.isLinearAttention || pastValue.isLinearAttention) {
                return 0;
            }
            int len = GetCacheLen(pastKey);
            if (len > 0) {
                return len;
            }
            return GetCacheLen(pastValue);
        };

        int len = tryGetLen(model->kvCacheId);
        if (len > 0) {
            return len;
        }
        for (int i = 0; i < (int)pastKeyValues.size(); i++) {
            len = tryGetLen(i);
            if (len > 0) {
                return len;
            }
        }
        return 0;
    }

    static int GetTokenGrowingCacheLen(const basellm *model, std::vector <std::pair <Data*, Data*> > &pastKeyValues) {
        auto tryGetLen = [&](int idx) {
            if (idx < 0 || idx >= (int)pastKeyValues.size()) {
                return 0;
            }
            auto *pastKey = pastKeyValues[idx].first;
            auto *pastValue = pastKeyValues[idx].second;
            if (pastKey == nullptr || pastValue == nullptr) {
                return 0;
            }
            if (pastKey->isLinearAttention || pastValue->isLinearAttention) {
                return 0;
            }
            int len = GetCacheLen(*pastKey);
            if (len > 0) {
                return len;
            }
            return GetCacheLen(*pastValue);
        };

        int len = tryGetLen(model->kvCacheId);
        if (len > 0) {
            return len;
        }
        for (int i = 0; i < (int)pastKeyValues.size(); i++) {
            len = tryGetLen(i);
            if (len > 0) {
                return len;
            }
        }
        return 0;
    }

    void basellm::ResetLogitsOfEOS(int batch, Data *logits, std::vector <std::pair <Data, Data> > &pastKeyValues,
            const GenerationConfig &generationConfig) {
        auto &config = generationConfig;
        if (config.output_token_least <= 0) {
            return;
        }
        int cacheLen = GetTokenGrowingCacheLen(this, pastKeyValues);
        if (logits->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            bool need_reset = false;
            std::vector<int> common_eos_ids;
            common_eos_ids.push_back(this->eos_token_id);
            for (auto id: this->eos_token_ids) {
                common_eos_ids.push_back(id);
            }
            for (auto id: config.stop_token_ids) {
                common_eos_ids.push_back(id);
            }
            for (int b = 0; b < batch; b++) {
                need_reset |= config.output_token_least - cacheLen + config.input_token_length > 0;
            }
            if (need_reset) {
                ToDataType(*logits, DataType::FLOAT32);
                FastllmResetLogitsOfEOSAll(batch, logits, common_eos_ids);
            }
#endif
        } else {
            for (int b = 0; b < batch; b++) {
                if (config.output_token_least > cacheLen - config.input_token_length) {
                    ToDataType(*logits, DataType::FLOAT32);
                    float *logit = ((float*)logits->cpuData) + logits->Count(0) / batch * b;
                    logit[this->eos_token_id] = 0;
                    for (auto id: this->eos_token_ids)
                        logit[id] = 0;
                    for (auto id: config.stop_token_ids)
                        logit[id] = 0;
                }
            }
        }
        return;
    }

    void basellm::ResetLogitsOfEOS(int batch, Data *logits, std::vector <std::pair <Data*, Data*> > &pastKeyValues,
            const std::vector <GenerationConfig> &generationConfigs) {
        bool hasMinOutputLength = false;
        for (int b = 0; b < batch; b++) {
            if (generationConfigs[b].output_token_least > 0) {
                hasMinOutputLength = true;
                break;
            }
        }
        if (!hasMinOutputLength) {
            return;
        }
        int cacheLen = GetTokenGrowingCacheLen(this, pastKeyValues);
        if (logits->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            auto buildEosIds = [&](const GenerationConfig &config) {
                std::vector<int> ids;
                ids.push_back(this->eos_token_id);
                for (auto id: this->eos_token_ids) {
                    ids.push_back(id);
                }
                for (auto id: config.stop_token_ids) {
                    ids.push_back(id);
                }
                return ids;
            };
            bool need_reset = false, all_need_reset = true, same_eos_ids = true;
            std::vector<int> common_eos_ids;
            for (int b = 0; b < batch; b++) {
                auto &config = generationConfigs[b];
                bool curNeedReset = config.output_token_least - cacheLen + config.input_token_length > 0;
                need_reset |= curNeedReset;
                all_need_reset &= curNeedReset;
                std::vector<int> cur_eos_ids = buildEosIds(config);
                if (b == 0) {
                    common_eos_ids = std::move(cur_eos_ids);
                } else if (cur_eos_ids != common_eos_ids) {
                    same_eos_ids = false;
                }
            }
            if (need_reset) {
                ToDataType(*logits, DataType::FLOAT32);
                if (all_need_reset && same_eos_ids) {
                    FastllmResetLogitsOfEOSAll(batch, logits, common_eos_ids);
                } else {
                    std::vector<int> res_lens, eos_nums, eos_ids;
                    for (int b = 0; b < batch; b++) {
                        auto &config = generationConfigs[b];
                        res_lens.push_back(config.output_token_least - cacheLen + config.input_token_length);
                        std::vector<int> cur_eos_ids = buildEosIds(config);
                        eos_nums.push_back((int)cur_eos_ids.size());
                        eos_ids.insert(eos_ids.end(), cur_eos_ids.begin(), cur_eos_ids.end());
                    }
                    FastllmResetLogitsOfEOS(batch, logits, res_lens, eos_nums, eos_ids);
                }
            }
#endif
        } else {
            for (int b = 0; b < batch; b++) {
                auto &config = generationConfigs[b];
                if (config.output_token_least > cacheLen - config.input_token_length) {
                    ToDataType(*logits, DataType::FLOAT32);
                    float *logit = ((float*)logits->cpuData) + logits->Count(0) / batch * b;
                    logit[this->eos_token_id] = 0;
                    for (auto id: this->eos_token_ids)
                        logit[id] = 0;
                    for (auto id: config.stop_token_ids)
                        logit[id] = 0;
                }
            }
        }
        return;
    }

    void basellm::AutoWarmup() {
        if (GetFastllmEnv().skipWarmup) {
            return;
        }
        struct AutoWarmupFinishGuard {
            basellm *model;
            AutoWarmupFinishGuard(basellm *model) : model(model) {}
            ~AutoWarmupFinishGuard() {
                model->OnAutoWarmupFinished();
#ifdef USE_CUDA
                // warmup 结束后切回异步集合通信：此时内存池已热，稳态前向基本不再触发真实 cudaMalloc，
                // 异步发射安全且能恢复通信/计算重叠的吞吐。warmup 及之前(权重加载)保持同步以防死锁。
                FastllmCudaSetNcclForceSync(false);
#endif
            }
        } autoWarmupFinishGuard(this);
        struct AutoWarmupRunningGuard {
            basellm *model;
            AutoWarmupRunningGuard(basellm *model) : model(model) {
                model->autoWarmupRunning.store(true);
            }
            ~AutoWarmupRunningGuard() {
                model->autoWarmupRunning.store(false);
            }
        } autoWarmupRunningGuard(this);
        if (!this->use_new_engine) {
            WarmUp();
            return;
        }

        printf("Warmup...\n");
        Prepare();

        int pageLen = fastllm::GetPageLen();
        int len = this->GetChunkedPrefillSize();
        bool autoCalcPages = (fastllm::GetMaxTokens() <= 0);
        if (!autoCalcPages && fastllm::GetMaxTokens() > 0 && pageLen > 0) {
            int maxWarmupLen = std::max(1, fastllm::GetMaxTokens() - pageLen);
            if (len > maxWarmupLen) {
                if (this->verbose) {
                    printf("[Fastllm] AutoWarmup prefill length clamped from %d to %d by explicit KV cache token limit %d.\n",
                           len, maxWarmupLen, fastllm::GetMaxTokens());
                }
                len = maxWarmupLen;
            }
        }
#ifdef USE_CUDA
        const bool userSetMaxBatch = this->maxBatch > 0;
        const int userMaxBatch = this->maxBatch;
#endif
        int minPages = -1;
        PagedCacheManager *autoWarmupPagedCacheManager = nullptr;
        bool useGPUForwardForWarmup = IsPureGpuMode(this);

#ifdef USE_CUDA
        auto printCudaWarmupPoolStats = [&](const char *stage) {
            if (!this->verbose) {
                return;
            }
            printf("[Fastllm] AutoWarmup CUDA pool after %s:\n", stage);
            FastllmCudaMemPoolStats();
        };
#endif

        auto runWarmupForward = [&](int batch,
                                    const Data &warmupInputIds,
                                    const std::vector <Data*> &warmupAttentionMasks,
                                    const std::vector <Data*> &warmupPositionIds,
                                    const std::vector <int> &warmupSeqLens,
                                    std::vector <std::pair <Data*, Data*> > &warmupPastKeyValues,
                                    const std::vector <GenerationConfig> &warmupGenerationConfigs,
                                    const LastTokensManager &warmupLastTokens) -> std::vector<int> {
            if (useGPUForwardForWarmup) {
                return ForwardGPU(batch, warmupInputIds, warmupAttentionMasks, warmupPositionIds,
                                  warmupSeqLens, warmupPastKeyValues, warmupGenerationConfigs,
                                  warmupLastTokens, nullptr);
            }
            return ForwardV2(batch, warmupInputIds, warmupAttentionMasks, warmupPositionIds,
                             warmupSeqLens, warmupPastKeyValues, warmupGenerationConfigs,
                             warmupLastTokens, nullptr);
        };

        auto captureWarmupPagedCacheManager = [&](std::vector <std::pair <Data, Data> > &storage) {
            if (this->kvCacheId >= 0 && this->kvCacheId < (int)storage.size()) {
                autoWarmupPagedCacheManager = storage[this->kvCacheId].first.pagedKVCacheData;
            }
        };

        if (autoCalcPages) {
            minPages = len / pageLen + 2;
            fastllm::SetMaxTokens(minPages * pageLen);
        }

#ifdef USE_CUDA
        if (len > 1) {
            // Load long-lived weights before the large prefill warmup creates
            // sequence-length-sized activation blocks in the CUDA pool.
            // Step3.5 CUDA graph must avoid a len=1 warmup here: that enters the
            // decode path and leaves decode-side CUDA state incompatible with the
            // later batch graph capture.
            const int weightWarmupLen =
                (GetFastllmEnv().cudaGraph && this->model_type == "step3p5") ? 2 : 1;
            if (weightWarmupLen > 1) {
                printf("[Fastllm] Step3.5 CUDA graph: use AutoWarmup weight prefill forward.\n");
            }
            std::vector <float> weightWarmupIds(weightWarmupLen, 1.0f);
            Data weightWarmupInputIds = Data(DataType::FLOAT32, {1, weightWarmupLen}, weightWarmupIds);
            std::vector <float> weightWarmupPosData(weightWarmupLen);
            for (int i = 0; i < weightWarmupLen; i++) weightWarmupPosData[i] = i;
            Data weightWarmupPositionIds = Data(this->dataType, {1, weightWarmupLen}, weightWarmupPosData);
            std::vector <Data*> weightWarmupAttentionMasks = {nullptr};
            std::vector <Data*> weightWarmupPositionIdsVec = {&weightWarmupPositionIds};
            std::vector <int> weightWarmupSeqLens = {weightWarmupLen};
            std::vector <std::pair <Data, Data> > weightWarmupPastKeyValuesStorage;
            std::vector <std::pair <Data*, Data*> > weightWarmupPastKeyValues;
            for (int i = 0; i < block_cnt; i++) {
                weightWarmupPastKeyValuesStorage.push_back(std::make_pair(Data(this->kvCacheDataType), Data(this->kvCacheDataType)));
                weightWarmupPastKeyValuesStorage.back().first.SetKVCache();
                weightWarmupPastKeyValuesStorage.back().second.SetKVCache();
            }
            for (int i = 0; i < block_cnt; i++) {
                weightWarmupPastKeyValues.push_back(std::make_pair(&weightWarmupPastKeyValuesStorage[i].first, &weightWarmupPastKeyValuesStorage[i].second));
            }
            GenerationConfig weightWarmupGenerationConfig;
            std::vector <GenerationConfig> weightWarmupGenerationConfigs = {weightWarmupGenerationConfig};
            LastTokensManager weightWarmupLastTokens;
            runWarmupForward(1, weightWarmupInputIds, weightWarmupAttentionMasks, weightWarmupPositionIdsVec,
                             weightWarmupSeqLens, weightWarmupPastKeyValues, weightWarmupGenerationConfigs,
                             weightWarmupLastTokens);

            for (auto &kv : weightWarmupPastKeyValuesStorage) {
                kv.first.pageIndex.clear();
                kv.first.pagedKVCacheData = nullptr;
                kv.first.isPagedKVCache = false;
                kv.second.pageIndex.clear();
                kv.second.pagedKVCacheData = nullptr;
                kv.second.isPagedKVCache = false;
            }
            weightWarmupPastKeyValuesStorage.clear();
            weightWarmupPastKeyValues.clear();
            ClearAllPagedCacheManagers();
            FastllmCudaClearBigBuffer();
            printCudaWarmupPoolStats("weight warmup cleanup");
        }
#endif

        std::vector <float> ids(len, 1.0f);
        Data inputIds = Data(DataType::FLOAT32, {1, len}, ids);
        std::vector <float> posData(len);
        for (int i = 0; i < len; i++) posData[i] = i;
        Data positionIds = Data(this->dataType, {1, len}, posData);
        std::vector <Data*> attentionMasks = {nullptr};
        std::vector <Data*> positionIdsVec = {&positionIds};
        std::vector <int> seqLens = {len};
        std::vector <std::pair <Data, Data> > pastKeyValuesStorage;
        std::vector <std::pair <Data*, Data*> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValuesStorage.push_back(std::make_pair(Data(this->kvCacheDataType), Data(this->kvCacheDataType)));
            pastKeyValuesStorage.back().first.SetKVCache();
            pastKeyValuesStorage.back().second.SetKVCache();
        }
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(&pastKeyValuesStorage[i].first, &pastKeyValuesStorage[i].second));
        }
        GenerationConfig generationConfig;
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        LastTokensManager lastTokens;
        runWarmupForward(1, inputIds, attentionMasks, positionIdsVec, seqLens, pastKeyValues, generationConfigs, lastTokens);
#ifdef USE_CUDA
        printCudaWarmupPoolStats("initial prefill");
#endif

        this->kvCacheId = 0;
        elementsInKVCachePerToken = 0;
        bool foundTokenGrowingCache = false;
        std::vector <long long> layerElementsPerToken(block_cnt, 0);
        int tokenGrowingLayerCount = 0, linearLayerCount = 0;
        long long linearFixedBytes = 0;
        for (int i = 0; i < block_cnt; i++) {
            auto &pastKey = pastKeyValuesStorage[i].first;
            auto &pastValue = pastKeyValuesStorage[i].second;
            if (pastKey.isLinearAttention || pastValue.isLinearAttention) {
                linearLayerCount++;
                linearFixedBytes += pastKey.GetBytes() + pastValue.GetBytes();
                continue;
            }
            if (pastKey.dims.size() < 3 || pastValue.dims.size() < 3) {
                continue;
            }
            tokenGrowingLayerCount++;
            if (!foundTokenGrowingCache) {
                this->kvCacheId = i;
                foundTokenGrowingCache = true;
            }

            layerElementsPerToken[i] =
                (long long)pastKey.dims[0] * pastKey.dims[2] +
                (long long)pastValue.dims[0] * pastValue.dims[2];
            elementsInKVCachePerToken += layerElementsPerToken[i];
        }

        long long bytesPerToken = 0;
        if (elementsInKVCachePerToken > 0) {
            bytesPerToken = GetDataBytes(this->kvCacheDataType, 1, elementsInKVCachePerToken);
        }
        if (autoCalcPages) {
            printf("[Fastllm] AutoWarmup stats: tokenGrowingLayers=%d, linearLayers=%d, linearFixedCache=%.2f MB, kvBytesPerToken=%.2f KB, firstKVLayer=%d.\n",
                   tokenGrowingLayerCount, linearLayerCount, linearFixedBytes / 1e6,
                   bytesPerToken / 1024.0, this->kvCacheId);
        }
        captureWarmupPagedCacheManager(pastKeyValuesStorage);

        auto releaseWarmupPagedCachePages = [&](std::vector <std::pair <Data, Data> > &storage) {
            for (auto &kv : storage) {
                ReleasePagedCachePages(kv.first);
                kv.first.pagedKVCacheData = nullptr;
                kv.first.isPagedKVCache = false;
                ReleasePagedCachePages(kv.second);
                kv.second.pagedKVCacheData = nullptr;
                kv.second.isPagedKVCache = false;
            }
        };

        auto runBatchSeqWarmup = [&](const std::vector<int> &warmSeqLens, bool samplingWarmup) {
            int batch = (int)warmSeqLens.size();
            if (batch <= 0) {
                return;
            }
            int totalTokens = 0;
            for (int seqLen : warmSeqLens) {
                if (seqLen <= 0) {
                    return;
                }
                totalTokens += seqLen;
            }

            pastKeyValuesStorage.clear();
            pastKeyValues.clear();
            pastKeyValuesStorage.reserve((size_t)batch * block_cnt);
            pastKeyValues.reserve((size_t)batch * block_cnt);
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    pastKeyValuesStorage.push_back(std::make_pair(Data(this->kvCacheDataType), Data(this->kvCacheDataType)));
                    pastKeyValuesStorage.back().first.SetKVCache();
                    pastKeyValuesStorage.back().second.SetKVCache();
                    pastKeyValues.push_back(std::make_pair(&pastKeyValuesStorage.back().first,
                                                           &pastKeyValuesStorage.back().second));
                }
            }

            std::vector<float> shortInputIdsHost(totalTokens, 1.0f);
            Data shortInputIds = Data(DataType::FLOAT32, {1, totalTokens}, shortInputIdsHost);
            std::vector<Data> shortPositionIdsStorage;
            std::vector<Data*> shortPositionIdsVec;
            shortPositionIdsStorage.reserve(batch);
            shortPositionIdsVec.reserve(batch);
            for (int b = 0; b < batch; b++) {
                int seqLen = warmSeqLens[b];
                std::vector<float> shortPos(seqLen);
                for (int i = 0; i < seqLen; i++) {
                    shortPos[i] = (float)i;
                }
                shortPositionIdsStorage.push_back(Data(this->dataType, {1, seqLen}, shortPos));
                shortPositionIdsVec.push_back(&shortPositionIdsStorage.back());
            }
            std::vector <Data*> shortAttentionMasks(batch, nullptr);
            std::vector <GenerationConfig> shortGenerationConfigs(batch);
            if (samplingWarmup) {
                for (auto &config : shortGenerationConfigs) {
                    config.top_k = 5;
                    config.top_p = 0.95f;
                    config.temperature = 0.6f;
                }
            }
            LastTokensManager shortLastTokens(batch, shortGenerationConfigs[0].last_n);

            runWarmupForward(batch, shortInputIds, shortAttentionMasks, shortPositionIdsVec,
                             warmSeqLens, pastKeyValues, shortGenerationConfigs, shortLastTokens);
            captureWarmupPagedCacheManager(pastKeyValuesStorage);
            releaseWarmupPagedCachePages(pastKeyValuesStorage);
        };

        auto runUniformBatchWarmup = [&](int batch, int tokensPerRequest, bool samplingWarmup) {
            runBatchSeqWarmup(std::vector<int>(batch, tokensPerRequest), samplingWarmup);
        };

        auto runSamplingPrefillWarmup = [&](int targetBatch, const char *label) {
            if (len <= 2 || targetBatch <= 1) {
                return;
            }
            int samplingPrefillBatch = std::max(1, std::min(targetBatch, len));
            std::vector<int> samplingPrefillSeqLens(
                samplingPrefillBatch, len / samplingPrefillBatch);
            int extraTokens = len % samplingPrefillBatch;
            for (int i = 0; i < extraTokens; i++) {
                samplingPrefillSeqLens[i]++;
            }
            int maxTokensPerRequest = samplingPrefillSeqLens.empty() ? 0 : samplingPrefillSeqLens[0];
            printf("[Fastllm] AutoWarmup CUDA sampling prefill warmup (%s): batch %d, total tokens %d, max tokens/request %d.\n",
                   label, samplingPrefillBatch, len, maxTokensPerRequest);
            runBatchSeqWarmup(samplingPrefillSeqLens, true);
        };

        if (autoCalcPages) {
#ifdef USE_CUDA
            std::set <int> deviceIds;

            long long bytesPerPage = 0;
            std::map <int, long long> deviceBytesPerPage;
            std::map <int, int> deviceLayerCount;
            for (int i = 0; i < block_cnt; i++) {
                if (layerElementsPerToken[i] <= 0) {
                    continue;
                }
                auto &pastKey = pastKeyValuesStorage[i].first;
                auto &pastValue = pastKeyValuesStorage[i].second;
                bool accountedByLocalShard = false;
                long long layerBytesPerPage = GetDataBytes(this->kvCacheDataType, pageLen, layerElementsPerToken[i]);

                if (pastKey.multiDeviceData && pastValue.multiDeviceData &&
                    !pastKey.multiDeviceDatas.empty() && !pastValue.multiDeviceDatas.empty()) {
                    for (auto &it : pastKey.multiDeviceDatas) {
                        int id = it.first;
                        if (pastValue.multiDeviceDatas.find(id) == pastValue.multiDeviceDatas.end()) {
                            continue;
                        }
                        Data *localKey = it.second;
                        Data *localValue = pastValue.multiDeviceDatas[id];
                        if (localKey == nullptr || localValue == nullptr ||
                            localKey->dataDevice != DataDevice::CUDA || localValue->dataDevice != DataDevice::CUDA ||
                            localKey->dims.size() < 3 || localValue->dims.size() < 3 ||
                            localKey->isLinearAttention || localValue->isLinearAttention) {
                            continue;
                        }

                        long long localElementsPerToken =
                            (long long)localKey->dims[0] * localKey->dims[2] +
                            (long long)localValue->dims[0] * localValue->dims[2];
                        long long localBytesPerPage = GetDataBytes(this->kvCacheDataType, pageLen, localElementsPerToken);
                        deviceIds.insert(id);
                        deviceBytesPerPage[id] += localBytesPerPage;
                        deviceLayerCount[id]++;
                        accountedByLocalShard = true;
                    }
                }

                if (!accountedByLocalShard) {
                    int id = 0;
                    if (pastKey.dataDevice == DataDevice::CUDA && !pastKey.dataDeviceIds.empty()) {
                        id = pastKey.dataDeviceIds[0];
                    } else if (pastValue.dataDevice == DataDevice::CUDA && !pastValue.dataDeviceIds.empty()) {
                        id = pastValue.dataDeviceIds[0];
                    }
                    deviceIds.insert(id);
                    deviceBytesPerPage[id] += layerBytesPerPage;
                    deviceLayerCount[id]++;
                }
                bytesPerPage += layerBytesPerPage;
            }

            bool updatedPages = false;
            int calculatedMaxPages = -1;
            std::string fallbackReason = "";
            int autoLinearAttentionBatchLimit = -1;

            for (auto &kv : pastKeyValuesStorage) {
                kv.first.pageIndex.clear();
                kv.first.pagedKVCacheData = nullptr;
                kv.first.isPagedKVCache = false;
                kv.second.pageIndex.clear();
                kv.second.pagedKVCacheData = nullptr;
                kv.second.isPagedKVCache = false;
            }
            // Tensor-parallel KV cache shards live in multiDeviceDatas. Destroy those
            // Data objects before tearing down the global paged cache managers, or
            // their destructors will release page indices through dangling managers.
            pastKeyValuesStorage.clear();
            pastKeyValues.clear();
            autoWarmupPagedCacheManager = nullptr;
            ClearAllPagedCacheManagers();
            printCudaWarmupPoolStats("paged cache cleanup");

            auto getBaseBatchLimit = [&]() -> int {
                int batchLimit = userSetMaxBatch ? userMaxBatch : 512;
                batchLimit = NormalizeMaxBatchByModelCapability(this, batchLimit);
                return std::max(1, batchLimit);
            };
            auto getCudaWarmupBatchLimit = [&]() -> int {
                int batchLimit = getBaseBatchLimit();
                if (!userSetMaxBatch && autoLinearAttentionBatchLimit > 0) {
                    batchLimit = std::min(batchLimit, autoLinearAttentionBatchLimit);
                }
                return std::max(1, batchLimit);
            };
            auto updateLinearAttentionBatchLimit = [&](long long avail) {
                if (userSetMaxBatch || linearFixedBytes <= 0 || avail <= 0) {
                    return;
                }
                long long rawLimit = (avail / 2) / linearFixedBytes;
                int limit = (int)std::min<long long>(std::max(1LL, rawLimit), INT_MAX);
                if (autoLinearAttentionBatchLimit <= 0) {
                    autoLinearAttentionBatchLimit = limit;
                } else {
                    autoLinearAttentionBatchLimit = std::min(autoLinearAttentionBatchLimit, limit);
                }
            };
            bool skipShortWarmupForward =
                GetFastllmEnv().cudaGraph && this->model_type == "step3p5";
            if (skipShortWarmupForward) {
                printf("[Fastllm] Step3.5 CUDA graph: skip AutoWarmup pre-page buffer warmup.\n");
            } else {
                int prePageWarmupBatch = getCudaWarmupBatchLimit();
                if (minPages > 0) {
                    prePageWarmupBatch = std::min(prePageWarmupBatch, minPages);
                }
                prePageWarmupBatch = std::max(1, std::min(prePageWarmupBatch, 16));
                printf("[Fastllm] AutoWarmup CUDA pre-page buffer warmup up to %d before cache sizing.\n",
                       prePageWarmupBatch);

                std::vector<int> prePageWarmupBatches;
                for (int batch = 1; batch < prePageWarmupBatch; batch *= 2) {
                    prePageWarmupBatches.push_back(batch);
                }
                if (prePageWarmupBatches.empty() || prePageWarmupBatches.back() != prePageWarmupBatch) {
                    prePageWarmupBatches.push_back(prePageWarmupBatch);
                }
                for (int batch : prePageWarmupBatches) {
                    runUniformBatchWarmup(batch, 1, false);
                }
                runUniformBatchWarmup(prePageWarmupBatch, 1, true);

                int smallPrefillLimit = std::min(len, 16);
                for (int warmLen = 2; warmLen <= smallPrefillLimit; warmLen *= 2) {
                    printf("[Fastllm] AutoWarmup CUDA sampling small prefill warmup: tokens %d.\n", warmLen);
                    runBatchSeqWarmup({warmLen}, true);
                }
                int prePagePrefillBatch = prePageWarmupBatch;
                if (pageLen > 0 && minPages > 0) {
                    auto prePagePrefillPages = [&](int batch) {
                        int pages = 0;
                        int base = len / batch;
                        int extra = len % batch;
                        for (int i = 0; i < batch; i++) {
                            int seqLen = base + (i < extra ? 1 : 0);
                            pages += (seqLen + pageLen - 1) / pageLen;
                        }
                        return pages;
                    };
                    while (prePagePrefillBatch > 1 &&
                           prePagePrefillPages(prePagePrefillBatch) > minPages) {
                        prePagePrefillBatch--;
                    }
                }
                runSamplingPrefillWarmup(prePagePrefillBatch, "pre-page");

                pastKeyValuesStorage.clear();
                pastKeyValues.clear();
                autoWarmupPagedCacheManager = nullptr;
                ClearAllPagedCacheManagers();
                printCudaWarmupPoolStats("pre-page buffer warmup");
            }

            auto freeSizes = FastllmCudaGetFreeSizes();
            auto totalSizes = FastllmCudaGetTotalSizes();
            auto getCudaRuntimeHeadroom = [&](int id, long long avail) -> long long {
                if (avail <= 0) {
                    return 0;
                }

                long long headroom = 512LL * 1024LL * 1024LL;
                if (id >= 0 && id < (int)totalSizes.size()) {
                    headroom = std::max(headroom, totalSizes[id] / 100);
                }
                headroom = std::min(headroom, 2LL * 1024LL * 1024LL * 1024LL);
                headroom = std::min(headroom, avail / 4);
                return std::max(0LL, headroom);
            };
            auto fitPagesWithLinearReserve = [&](int id, long long avail, long long kvBytesPerPage) -> int {
                if (avail <= 0 || kvBytesPerPage <= 0) {
                    return 0;
                }
                long long rawPages = avail / kvBytesPerPage;
                if (rawPages <= 0) {
                    return (int)std::min<long long>(rawPages, INT_MAX);
                }

                int batchLimit = getCudaWarmupBatchLimit();
                auto runtimeReserveBytes = [&](long long activeBatch) -> long long {
                    if (activeBatch <= 0) {
                        return 0;
                    }
                    activeBatch = std::min<long long>(activeBatch, INT_MAX);
                    long long reserve = this->GetAutoWarmupCudaRuntimeReserveBytes(id, (int)activeBatch);
                    return std::max(0LL, reserve);
                };
                long long probeBatch = std::min<long long>(batchLimit, rawPages);
                if (linearFixedBytes <= 0 && runtimeReserveBytes(probeBatch) <= 0) {
                    return (int)std::min<long long>(rawPages, INT_MAX);
                }

                long long low = 0, high = rawPages;
                while (low < high) {
                    long long mid = (low + high + 1) / 2;
                    long long activeBatch = std::min<long long>(batchLimit, mid);
                    __int128 need = (__int128)mid * kvBytesPerPage +
                                    (__int128)activeBatch * linearFixedBytes +
                                    (__int128)runtimeReserveBytes(activeBatch);
                    if (need <= avail) {
                        low = mid;
                    } else {
                        high = mid - 1;
                    }
                }
                if (low < rawPages) {
                    long long activeBatch = std::min<long long>(batchLimit, low);
                    long long runtimeReserve = runtimeReserveBytes(activeBatch);
                    printf("[Fastllm] AutoWarmup GPU %d: limit pages %lld -> %lld, reserve %.2f MB/request linear cache and %.2f MB runtime cache up to batch %lld.\n",
                           id, rawPages, low, linearFixedBytes / 1e6,
                           runtimeReserve / 1e6, activeBatch);
                }
                return (int)std::min<long long>(low, INT_MAX);
            };
            if (deviceBytesPerPage.size() > 1) {
                // 多卡模式：对每张实际承载 token-growing cache 的卡分别限流，取最小值
                int maxPages = INT_MAX;
                for (auto &it : deviceBytesPerPage) {
                    int id = it.first;
                    if (id < (int)freeSizes.size() && id < (int)totalSizes.size()) {
                        long long reserved = (long long)(totalSizes[id] * (1.0 - fastllm::GetGpuMemRatio()));
                        long long rawAvail = freeSizes[id] - reserved;
                        long long runtimeHeadroom = getCudaRuntimeHeadroom(id, rawAvail);
                        long long avail = rawAvail - runtimeHeadroom;
                        long long perPageOnDevice = it.second;
                        updateLinearAttentionBatchLimit(avail);
                        printf("[Fastllm] AutoWarmup GPU %d: free=%.2f GB, total=%.2f GB, reserved=%.2f GB, runtimeHeadroom=%.2f MB, availForKV=%.2f GB, localKVPerPage=%.2f MB, tokenGrowingLayers=%d.\n",
                               id, freeSizes[id] / 1e9, totalSizes[id] / 1e9, reserved / 1e9,
                               runtimeHeadroom / 1e6, avail / 1e9, perPageOnDevice / 1e6,
                               deviceLayerCount.count(id) ? deviceLayerCount[id] : 0);
                        if (perPageOnDevice > 0 && avail > 0) {
                            int pages = fitPagesWithLinearReserve(id, avail, perPageOnDevice);
                            maxPages = std::min(maxPages, pages);
                        }
                    }
                }
                if (maxPages > 0 && maxPages < INT_MAX) {
                    fastllm::SetMaxTokens(maxPages * pageLen);
                    updatedPages = true;
                    calculatedMaxPages = maxPages;
                    printf("Auto set cache pages (multi-gpu): %d (tokens: %d).\n",
                           maxPages, maxPages * pageLen);
                    for (int id : deviceIds) {
                        if (id < (int)freeSizes.size() && id < (int)totalSizes.size()) {
                            long long rawAvail = freeSizes[id] - (long long)(totalSizes[id] * (1.0 - fastllm::GetGpuMemRatio()));
                            long long avail = rawAvail - getCudaRuntimeHeadroom(id, rawAvail);
                            int layers = deviceLayerCount.count(id) ? deviceLayerCount[id] : 0;
                            printf("  GPU %d: layers=%d, avail=%.2f GB.\n", id, layers, avail / 1e9);
                        }
                    }
                } else {
                    fallbackReason = "no GPU has positive availForKV / kvPerPage after reserve.";
                }
            } else {
                // 单卡模式：只看真正承载 paged KV cache 的设备
                long long cacheAvail = 0, cacheBytesPerPage = bytesPerPage;
                int cacheDeviceId = -1;
                if (!deviceBytesPerPage.empty()) {
                    cacheBytesPerPage = deviceBytesPerPage.begin()->second;
                    cacheDeviceId = deviceBytesPerPage.begin()->first;
                    if (cacheDeviceId < (int)freeSizes.size() && cacheDeviceId < (int)totalSizes.size()) {
                        long long reserved = (long long)(totalSizes[cacheDeviceId] * (1.0 - fastllm::GetGpuMemRatio()));
                        long long rawAvail = freeSizes[cacheDeviceId] - reserved;
                        long long runtimeHeadroom = getCudaRuntimeHeadroom(cacheDeviceId, rawAvail);
                        cacheAvail = rawAvail - runtimeHeadroom;
                        updateLinearAttentionBatchLimit(cacheAvail);
                        printf("[Fastllm] AutoWarmup GPU %d: free=%.2f GB, total=%.2f GB, reserved=%.2f GB, runtimeHeadroom=%.2f MB, availForKV=%.2f GB, kvPerPage=%.2f MB, tokenGrowingLayers=%d.\n",
                               cacheDeviceId, freeSizes[cacheDeviceId] / 1e9, totalSizes[cacheDeviceId] / 1e9,
                               reserved / 1e9, runtimeHeadroom / 1e6, cacheAvail / 1e9, cacheBytesPerPage / 1e6,
                               deviceLayerCount.count(cacheDeviceId) ? deviceLayerCount[cacheDeviceId] : 0);
                    }
                }
                if (cacheAvail > 0 && cacheBytesPerPage > 0) {
                    int maxPages = fitPagesWithLinearReserve(cacheDeviceId, cacheAvail, cacheBytesPerPage);
                    if (maxPages > 0) {
                        fastllm::SetMaxTokens(maxPages * pageLen);
                        updatedPages = true;
                        calculatedMaxPages = maxPages;
                        printf("Auto set cache pages: %d (tokens: %d, avail: %.2f GB).\n",
                               maxPages, maxPages * pageLen, cacheAvail / 1e9);
                    }
                } else {
                    if (deviceBytesPerPage.empty()) {
                        fallbackReason = "no CUDA token-growing KV cache layer found.";
                    } else if (cacheBytesPerPage <= 0) {
                        fallbackReason = "kvPerPage <= 0.";
                    } else {
                        fallbackReason = "availForKV <= 0 after reserve and runtime headroom.";
                    }
                }
            }

            if (!updatedPages) {
                if (fallbackReason == "") {
                    fallbackReason = "auto page calculation did not produce a positive page count.";
                }
                printf("[Fastllm] AutoWarmup fallback to minimum cache pages: %d (tokens: %d). Reason: %s\n",
                       minPages, minPages * pageLen, fallbackReason.c_str());
            } else if (minPages > 0 && calculatedMaxPages <= minPages) {
                printf("[Fastllm] AutoWarmup note: calculated pages (%d) did not exceed minimum warmup pages (%d).\n",
                       calculatedMaxPages, minPages);
            }

            if (!userSetMaxBatch && autoLinearAttentionBatchLimit > 0) {
                int baseBatchLimit = getBaseBatchLimit();
                int limitedBatch = std::max(1, std::min(baseBatchLimit, autoLinearAttentionBatchLimit));
                if (limitedBatch < baseBatchLimit) {
                    this->maxBatch = limitedBatch;
                    printf("[Fastllm] AutoWarmup auto max_batch limited %d -> %d: linear attention cache %.2f MB/request, capped to <=50%% of availForKV.\n",
                           baseBatchLimit, limitedBatch, linearFixedBytes / 1e6);
                }
            }

            if (skipShortWarmupForward) {
                printf("[Fastllm] Step3.5 CUDA graph: skip AutoWarmup short decode forward.\n");
            } else {
                int warmupMaxBatch = 512;
                if (this->maxBatch > 0) {
                    warmupMaxBatch = this->maxBatch;
                }
                warmupMaxBatch = NormalizeMaxBatchByModelCapability(this, warmupMaxBatch);
                warmupMaxBatch = std::max(1, std::min(warmupMaxBatch, fastllm::GetMaxTokens() / 128));
                printf("[Fastllm] AutoWarmup CUDA batch warmup up to %d.\n", warmupMaxBatch);
                int warmupTotalPages = pageLen > 0 ?
                    std::max(1, (fastllm::GetMaxTokens() + pageLen - 1) / pageLen) : 0;
                int warmupAddPrefillBatch = warmupTotalPages > 0 ?
                    std::max(1, std::min(warmupMaxBatch, warmupTotalPages * 4 / 5)) : warmupMaxBatch;

                std::vector<int> shortWarmupBatches;
                for (int batch = 1; batch < warmupMaxBatch; batch *= 2) {
                    shortWarmupBatches.push_back(batch);
                }
                if (shortWarmupBatches.empty() || shortWarmupBatches.back() != warmupMaxBatch) {
                    shortWarmupBatches.push_back(warmupMaxBatch);
                }

                for (int batch : shortWarmupBatches) {
                    runUniformBatchWarmup(batch, 1, false);
                }
                printf("[Fastllm] AutoWarmup CUDA sampling batch warmup at %d.\n", warmupMaxBatch);
                runUniformBatchWarmup(warmupMaxBatch, 1, true);
                int warmupMaxPrefillBatch = std::max(1, std::min(warmupMaxBatch, len));
                int warmupAddPrefillEffectiveBatch = std::max(1, std::min(warmupAddPrefillBatch, len));
                runSamplingPrefillWarmup(warmupMaxPrefillBatch, "max-batch");
                if (warmupAddPrefillEffectiveBatch != warmupMaxPrefillBatch) {
                    runSamplingPrefillWarmup(warmupAddPrefillBatch, "add-prefill");
                }
                this->WarmupCudaRuntimeBuffers(warmupMaxBatch);
                printCudaWarmupPoolStats("batch and sampling warmup");

                auto calibrateCachePagesToGpuBudget = [&]() {
                    if (!updatedPages || pageLen <= 0 || deviceBytesPerPage.empty()) {
                        return;
                    }
                    int currentPages = std::max(1, (fastllm::GetMaxTokens() + pageLen - 1) / pageLen);
                    auto freeAfterWarmup = FastllmCudaGetFreeSizes();
                    auto totalAfterWarmup = FastllmCudaGetTotalSizes();
                    long long extraPages = LLONG_MAX;
                    std::map<int, long long> deviceExtraPages;
                    std::map<int, long long> deviceTargetFree;
                    bool canGrow = false;

                    for (auto &it : deviceBytesPerPage) {
                        int id = it.first;
                        long long bytesPerPageOnDevice = it.second;
                        if (bytesPerPageOnDevice <= 0 ||
                            id < 0 || id >= (int)freeAfterWarmup.size() ||
                            id >= (int)totalAfterWarmup.size()) {
                            continue;
                        }

                        long long finalSafety =
                            std::max(128LL * 1024LL * 1024LL, totalAfterWarmup[id] / 200);
                        finalSafety = std::min(finalSafety, 512LL * 1024LL * 1024LL);
                        long long targetFree =
                            (long long)(totalAfterWarmup[id] * (1.0 - fastllm::GetGpuMemRatio())) +
                            finalSafety;
                        targetFree += std::max(
                            0LL,
                            this->GetAutoWarmupCudaRuntimeReserveBytes(id, warmupMaxBatch));
                        long long growBytes = freeAfterWarmup[id] - targetFree;
                        long long pages = growBytes > 0 ? growBytes / bytesPerPageOnDevice : 0;
                        deviceExtraPages[id] = pages;
                        deviceTargetFree[id] = targetFree;
                        if (pages > 0) {
                            extraPages = std::min(extraPages, pages);
                            canGrow = true;
                        } else {
                            extraPages = 0;
                        }
                    }

                    if (!canGrow || extraPages <= 0 || extraPages == LLONG_MAX) {
                        return;
                    }

                    long long calibratedPagesLong = (long long)currentPages + extraPages;
                    calibratedPagesLong = std::min<long long>(
                        calibratedPagesLong, INT_MAX / std::max(1, pageLen));
                    int calibratedPages = (int)calibratedPagesLong;
                    if (calibratedPages <= currentPages) {
                        return;
                    }

                    printf("[Fastllm] AutoWarmup calibrate cache pages by post-warmup GPU budget: %d -> %d (tokens: %lld -> %lld).\n",
                           currentPages, calibratedPages,
                           (long long)currentPages * pageLen,
                           (long long)calibratedPages * pageLen);
                    for (auto &it : deviceBytesPerPage) {
                        int id = it.first;
                        if (id < 0 || id >= (int)freeAfterWarmup.size() ||
                            id >= (int)totalAfterWarmup.size()) {
                            continue;
                        }
                        long long pages = deviceExtraPages.count(id) ? deviceExtraPages[id] : 0;
                        long long targetFree = deviceTargetFree.count(id) ? deviceTargetFree[id] : 0;
                        printf("  GPU %d: freeAfterWarmup=%.2f GB, targetFree=%.2f GB, localKVPerPage=%.2f MB, extraPagesLimit=%lld.\n",
                               id, freeAfterWarmup[id] / 1e9, targetFree / 1e9,
                               it.second / 1e6, pages);
                    }

                    autoWarmupPagedCacheManager = nullptr;
                    ClearAllPagedCacheManagers();
                    fastllm::SetMaxTokens(calibratedPages * pageLen);
                    calculatedMaxPages = calibratedPages;
                    runUniformBatchWarmup(1, 1, false);
                    printCudaWarmupPoolStats("calibrated paged cache allocation");
                };
                calibrateCachePagesToGpuBudget();
            }
#endif
        }
        printf("finish.\n");

        auto *pcm = autoWarmupPagedCacheManager != nullptr ?
            autoWarmupPagedCacheManager : this->GetPagedKVCacheManager(this->kvCacheId, true);
        if (pcm != nullptr) {
            int totalPages = pcm->maxPages;
            int cachePageLen = pcm->pageLen;
            this->tokensLimit = totalPages * cachePageLen;
            this->promptLimit = (totalPages * 4 / 5) * cachePageLen;
            {
                int mBatch = 512;
                if (this->maxBatch > 0) {
                    mBatch = this->maxBatch;
                }
                mBatch = NormalizeMaxBatchByModelCapability(this, mBatch);
                mBatch = std::max(1, std::min(mBatch, this->tokensLimit / 128));
                printf("[Fastllm] KV Cache Token limit: %d tokens (totalPages=%d, pageLen=%d).\n", this->tokensLimit, totalPages, cachePageLen);
                printf("[Fastllm] AddPrefill Pages limit: %d pages (80%% of %d).\n", totalPages * 4 / 5, totalPages);
                printf("[Fastllm] Batch limit: %d.\n", mBatch);
            }
        } else if (fastllm::GetMaxTokens() > 0) {
            int totalPages = (fastllm::GetMaxTokens() + pageLen - 1) / pageLen;
            int cachePageLen = pageLen;
            this->tokensLimit = totalPages * cachePageLen;
            this->promptLimit = (totalPages * 4 / 5) * cachePageLen;
            int mBatch = 512;
            if (this->maxBatch > 0) {
                mBatch = this->maxBatch;
            }
            mBatch = NormalizeMaxBatchByModelCapability(this, mBatch);
            mBatch = std::max(1, std::min(mBatch, this->tokensLimit / 128));
            printf("[Fastllm] KV Cache Token limit: %d tokens (totalPages=%d, pageLen=%d).\n", this->tokensLimit, totalPages, cachePageLen);
            printf("[Fastllm] AddPrefill Pages limit: %d pages (80%% of %d).\n", totalPages * 4 / 5, totalPages);
            printf("[Fastllm] Batch limit: %d.\n", mBatch);
        }
    }
}
