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

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
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

    void ResponseContext::Init(int blocks, DataType dataType) {
        pastKeyValues.clear();
        for (int i = 0; i < blocks; i++) {
            pastKeyValues.push_back(std::make_pair(Data(dataType),
                                                   Data(dataType)));
            pastKeyValues.back().first.SetKVCache();
            pastKeyValues.back().second.SetKVCache();
        }
        intParams.clear();
        currentTokens.clear();
        allTokens.clear();
        while (resultTokenQueue.size() > 0){
            resultTokenQueue.pop();
        }
        isEnding = false;
        preTokens = 0;
    }

    void ResponseContext::TryRecord(basellm *model) {
        if (model->saveHistoryChat) {
            model->pastKVCacheManager.Record(this->allTokens, this->allTokens.size(), &this->pastKeyValues);
        }
    }

    void ResponseContext::TryRecordPagedCache() {
        for (int i = 0; i < (int)this->pastKeyValues.size(); i++) {
            auto &kvFirst = this->pastKeyValues[i].first;
            auto &kvSecond = this->pastKeyValues[i].second;
            PagedCacheManager *kManager = GetPagedCacheManager(i * 2);
            PagedCacheManager *vManager = GetPagedCacheManager(i * 2 + 1);
            if (kManager != nullptr && !kvFirst.pageIndex.empty()) {
                kManager->Record(this->allTokens, kvFirst.pageIndex);
            }
            if (vManager != nullptr && !kvSecond.pageIndex.empty()) {
                vManager->Record(this->allTokens, kvSecond.pageIndex);
            }
        }
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
        dictLocker.lock();
        this->isFree = true;
        dictLocker.unlock();
        dictCV.notify_all();
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
                lastKeyValues->push_back(std::make_pair(Data(this->dataType), Data(this->dataType)));
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
            pastKeyValues.push_back(std::make_pair(Data(dataType),
                                                   Data(dataType)));
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
        basellm *model = this;
        int maxTotalLens = 0;
        int totalPages = 0;
        int pagesLimit = 0; // 默认为totalPages的80%，Prefill不会让已用分页超过此限制
        int pageLen = 128;

        int maxBatch = 512;
        if (model->maxBatch > 0) {
            maxBatch = model->maxBatch;
        }

        int prefillChunkSize;
        if (model->chunkedPrefillSize >= 0) {
            prefillChunkSize = model->chunkedPrefillSize;
        } else {
            prefillChunkSize = 8192;
            if (model->model_struct == "deepseek_v2") {
                prefillChunkSize = 2048;
            }
            if (model->model_struct == "qwen3_next") {
                prefillChunkSize = 2048;
            }
        }

        // 辅助lambda：释放一个请求占用的所有KV Cache分页，并以allTokens重新初始化为pending prefill状态
        auto releaseAndReinitRequest = [&](ResponseContext *ctx) {
            for (int i = 0; i < model->block_cnt; i++) {
                auto &kvFirst = ctx->pastKeyValues[i].first;
                auto &kvSecond = ctx->pastKeyValues[i].second;
                if (kvFirst.isPagedKVCache && kvFirst.pagedKVCacheData != nullptr) {
                    for (auto idx : kvFirst.pageIndex) {
                        kvFirst.pagedKVCacheData->ReleasePageIndex(idx);
                    }
                    kvFirst.pageIndex.clear();
                    kvFirst.lastPageLen = 0;
                    kvFirst.dims.clear();
                }
                if (kvSecond.isPagedKVCache && kvSecond.pagedKVCacheData != nullptr) {
                    for (auto idx : kvSecond.pageIndex) {
                        kvSecond.pagedKVCacheData->ReleasePageIndex(idx);
                    }
                    kvSecond.pageIndex.clear();
                    kvSecond.lastPageLen = 0;
                    kvSecond.dims.clear();
                }
            }
            ctx->currentTokens = ctx->allTokens;
            ctx->preTokens = 0;
            ctx->cacheLen = 0;
            ctx->intParams.clear();
        };

        auto lastRecordTime = std::chrono::system_clock::now();
        long long genTokens = 0;
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
            
            // 单次遍历：处理abort、释放isEnding的KV cache、统计alive、构建orders、检测hasPrefill
            std::vector <int> abortHandles;
            int busyPages = 0, currentActivate = 0;
            bool hasPrefill = false;
            std::vector <std::pair <int, int> > orders;
            int limit = (maxTotalLens > 0) ? maxTotalLens : 999999999;

            for (auto &it: model->responseContextDict.dicts) {
                if (it.second->isAbort) {
                    it.second->TryRecordPagedCache();
                    abortHandles.push_back(it.first);
                    continue;
                }
                if (it.second->isEnding) {
                    for (int i = 0; i < model->block_cnt; i++) {
                        auto &kvFirst = it.second->pastKeyValues[i].first;
                        auto &kvSecond = it.second->pastKeyValues[i].second;
                        if (kvFirst.isPagedKVCache && kvFirst.pagedKVCacheData != nullptr && !kvFirst.pageIndex.empty()) {
                            for (auto idx : kvFirst.pageIndex) {
                                kvFirst.pagedKVCacheData->ReleasePageIndex(idx);
                            }
                            kvFirst.pageIndex.clear();
                            kvFirst.lastPageLen = 0;
                        }
                        if (kvSecond.isPagedKVCache && kvSecond.pagedKVCacheData != nullptr && !kvSecond.pageIndex.empty()) {
                            for (auto idx : kvSecond.pageIndex) {
                                kvSecond.pagedKVCacheData->ReleasePageIndex(idx);
                            }
                            kvSecond.pageIndex.clear();
                            kvSecond.lastPageLen = 0;
                        }
                    }
                    continue;
                }
                if (it.second->preTokens > 0) {
                    currentActivate++;
                }
                if (it.second->preTokens == 0) {
                    hasPrefill = true;
                }
                orders.push_back(std::make_pair(-(int)it.second->currentTokens.size(), it.first));
            }
            for (auto &it : abortHandles) {
                model->responseContextDict.RemoveHandle(it);
            }
            sort(orders.begin(), orders.end());

            // 通过PagedCacheManager获取实际使用的物理页数（复用的页只算一次）
            if (totalPages > 0) {
                PagedCacheManager *probeManager = GetPagedCacheManager(model->kvCacheId * 2);
                if (probeManager != nullptr) {
                    std::lock_guard<std::mutex> guard(probeManager->pageIndexLocker);
                    busyPages = probeManager->maxPages - (int)probeManager->unusedPageIndex.size();
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

                for (auto &ii : orders) {
                    auto &it = *model->responseContextDict.dicts.find(ii.second);

                    if (it.second->isEnding) {
                        continue;
                    }
                    if (isPrompt && it.second->preTokens != 0) {
                        continue;
                    }
                    if (!isPrompt && it.second->preTokens == 0) {
                        continue;
                    }

                    if ((maxTotalLens > 0 && it.second->cacheLen + it.second->currentTokens.size() > maxTotalLens) ||
                        it.second->cacheLen + it.second->currentTokens.size() > model->max_positions) {
                        it.second->isEnding = true;
                        it.second->error = ResponseContextErrorPromptTooLong;
                        printf("[Handle %d] Finished. Reason: prompt too long (cacheLen=%d, currentTokens=%d, maxTotalLens=%d, max_positions=%d).\n",
                               it.first, it.second->cacheLen, (int)it.second->currentTokens.size(), maxTotalLens, model->max_positions);
                        continue;
                    }

                    if (isPrompt) {
                        if (it.second->cacheLen == 0) {
                            PagedCacheManager *probeManager = GetPagedCacheManager(model->kvCacheId * 2);
                            if (probeManager != nullptr) {
                                int minCachedPages = INT_MAX;
                                for (int li = 0; li < model->block_cnt; li++) {
                                    PagedCacheManager *kMgr = GetPagedCacheManager(li * 2);
                                    PagedCacheManager *vMgr = GetPagedCacheManager(li * 2 + 1);
                                    if (kMgr != nullptr) {
                                        std::vector<int> kPages;
                                        kMgr->Query(it.second->currentTokens, kPages);
                                        minCachedPages = std::min(minCachedPages, (int)kPages.size());
                                    } else {
                                        minCachedPages = 0;
                                    }
                                    if (vMgr != nullptr) {
                                        std::vector<int> vPages;
                                        vMgr->Query(it.second->currentTokens, vPages);
                                        minCachedPages = std::min(minCachedPages, (int)vPages.size());
                                    } else {
                                        minCachedPages = 0;
                                    }
                                    if (minCachedPages == 0) break;
                                }
                                if (minCachedPages > 0 && minCachedPages < INT_MAX) {
                                    int cachedLen = minCachedPages * probeManager->pageLen;
                                    if (cachedLen >= (int)it.second->currentTokens.size()) {
                                        minCachedPages--;
                                        cachedLen = minCachedPages * probeManager->pageLen;
                                    }
                                }
                                if (minCachedPages > 0 && minCachedPages < INT_MAX) {
                                    int cachedLen = minCachedPages * probeManager->pageLen;
                                    for (int li = 0; li < model->block_cnt; li++) {
                                        auto &kvFirst = it.second->pastKeyValues[li].first;
                                        auto &kvSecond = it.second->pastKeyValues[li].second;
                                        PagedCacheManager *kMgr = GetPagedCacheManager(li * 2);
                                        PagedCacheManager *vMgr = GetPagedCacheManager(li * 2 + 1);
                                        if (kMgr != nullptr) {
                                            std::vector<int> kPages;
                                            kMgr->Query(it.second->currentTokens, kPages);
                                            kvFirst.isPagedKVCache = true;
                                            kvFirst.pagedKVCacheData = kMgr;
                                            kvFirst.pageLen = kMgr->pageLen;
                                            kvFirst.pageIndex.assign(kPages.begin(), kPages.begin() + minCachedPages);
                                            kMgr->Pick(kvFirst.pageIndex);
                                            kvFirst.lastPageLen = kMgr->pageLen;
                                            int numHeads = ((Data*)kMgr)->dims[2];
                                            int headDim = ((Data*)kMgr)->dims[3];
                                            kvFirst.Resize({numHeads, cachedLen, headDim});
                                        }
                                        if (vMgr != nullptr) {
                                            std::vector<int> vPages;
                                            vMgr->Query(it.second->currentTokens, vPages);
                                            kvSecond.isPagedKVCache = true;
                                            kvSecond.pagedKVCacheData = vMgr;
                                            kvSecond.pageLen = vMgr->pageLen;
                                            kvSecond.pageIndex.assign(vPages.begin(), vPages.begin() + minCachedPages);
                                            vMgr->Pick(kvSecond.pageIndex);
                                            kvSecond.lastPageLen = vMgr->pageLen;
                                            int numHeads = ((Data*)vMgr)->dims[2];
                                            int headDim = ((Data*)vMgr)->dims[3];
                                            kvSecond.Resize({numHeads, cachedLen, headDim});
                                        }
                                    }
                                    it.second->currentTokens.erase(it.second->currentTokens.begin(), it.second->currentTokens.begin() + cachedLen);
                                    it.second->cacheLen = cachedLen;
                                    {
                                        std::lock_guard<std::mutex> guard(probeManager->pageIndexLocker);
                                        curBusyPages = probeManager->maxPages - (int)probeManager->unusedPageIndex.size();
                                    }
                                    if (model->verbose) {
                                        printf("[Handle %d] Prefix cache hit: %d pages (%d tokens).\n", it.first, minCachedPages, cachedLen);
                                    }
                                }
                            }
                        }

                        if (currentActivate + (int)seqLens.size() >= maxBatch) {
                            continue;
                        }

                        int thisLen = (int)it.second->currentTokens.size();
                        int thisPages = (thisLen + pageLen - 1) / pageLen;

                        // Prefill后已用分页不能超过pagesLimit（除非单个请求就超过了）
                        if (pagesLimit > 0 && curBusyPages + thisPages > pagesLimit) {
                            if (seqLens.size() > 0 || thisPages <= pagesLimit) {
                                continue;
                            }
                            if (currentActivate > 0) {
                                continue;
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
                        currentActivate++;
                    } else {
                        // Decode阶段：不在这里限制分页，由后续驱逐逻辑统一处理
                    }

                    generationConfigs.push_back(it.second->generationConfig);
                    if (it.second->generationConfig.output_logits) {
                        it.second->resultLogits.push(new std::vector<float>());
                        logits.push_back(it.second->resultLogits.back());
                    } else {
                        logits.push_back(nullptr);
                    }

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

                    if (seqLens.size() >= maxBatch || (totalPages > 0 && curBusyPages >= totalPages)) {
                        break;
                    }
                }
            }

            // Decode阶段：检查空闲分页是否足够，不够时释放资源
            if (seqLens.size() > 0 && seqLens[0] == 1) {
                PagedCacheManager *pagedManager = nullptr;
                int newPagesNeeded = 0;
                for (int i = 0; i < (int)handles.size(); i++) {
                    auto &ctx = *model->responseContextDict.dicts[handles[i]];
                    auto &kvFirst = ctx.pastKeyValues[model->kvCacheId].first;
                    if (kvFirst.isPagedKVCache) {
                        if (pagedManager == nullptr) {
                            pagedManager = kvFirst.pagedKVCacheData;
                        }
                        if (kvFirst.pageLen == kvFirst.lastPageLen) {
                            newPagesNeeded++;
                        }
                    }
                }
                if (pagedManager != nullptr && newPagesNeeded > 0) {
                    int freePages;
                    {
                        std::lock_guard<std::mutex> guard(pagedManager->pageIndexLocker);
                        freePages = (int)pagedManager->unusedPageIndex.size();
                    }
                    // if (freePages < newPagesNeeded) {
                    //    printf("[Decode] Page shortage: newPagesNeeded=%d, freePages=%d, maxPages=%d, batchSize=%d\n",
                    //           newPagesNeeded, freePages, pagedManager->maxPages, (int)handles.size());
                    // }
                    while (freePages < newPagesNeeded) {
                        // 空闲分页不够，从本轮decode批次中选择上下文最长的请求驱逐
                        int maxLen = -1, evictIdx = -1;
                        for (int i = 0; i < (int)handles.size(); i++) {
                            auto &ctx = *model->responseContextDict.dicts[handles[i]];
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
                        seqLens.erase(seqLens.begin() + evictIdx);
                        generationConfigs.erase(generationConfigs.begin() + evictIdx);
                        tokensManager.units.erase(tokensManager.units.begin() + evictIdx);
                        if (logits[evictIdx] != nullptr) {
                            delete logits[evictIdx];
                        }
                        logits.erase(logits.begin() + evictIdx);
                        if (attentionMasks[evictIdx] != nullptr) {
                            delete attentionMasks[evictIdx];
                        }
                        attentionMasks.erase(attentionMasks.begin() + evictIdx);
                        if (positionIds[evictIdx] != nullptr) {
                            delete positionIds[evictIdx];
                        }
                        positionIds.erase(positionIds.begin() + evictIdx);
                        pastKeyValues.erase(pastKeyValues.begin() + evictIdx * model->block_cnt,
                                            pastKeyValues.begin() + (evictIdx + 1) * model->block_cnt);

                        // 重新统计newPagesNeeded和freePages
                        newPagesNeeded = 0;
                        for (int i = 0; i < (int)handles.size(); i++) {
                            auto &ctx = *model->responseContextDict.dicts[handles[i]];
                            auto &kvFirst = ctx.pastKeyValues[model->kvCacheId].first;
                            if (kvFirst.isPagedKVCache && kvFirst.pageLen == kvFirst.lastPageLen) {
                                newPagesNeeded++;
                            }
                        }
                        {
                            std::lock_guard<std::mutex> guard(pagedManager->pageIndexLocker);
                            freePages = (int)pagedManager->unusedPageIndex.size();
                        }
                    }
                    if (freePages >= newPagesNeeded && newPagesNeeded > 0) {
                        // 重新计算ids
                        ids.clear();
                        for (int i = 0; i < (int)handles.size(); i++) {
                            auto &ctx = *model->responseContextDict.dicts[handles[i]];
                            for (int t : ctx.currentTokens) {
                                ids.push_back((float)t);
                            }
                        }
                    }
                }
                if (handles.empty()) {
                    seqLens.clear();
                }
            }

            if (seqLens.size() > 0) {
                dictLocker.unlock();
                forwardLocker.lock();
#ifdef USE_CUDA
                FastllmCudaClearBigBuffer();
#endif
                Data inputIds = Data(DataType::FLOAT32, {1, (int) ids.size()}, ids);
                std::vector<int> ret;

                if (seqLens.size() == 1 && seqLens[0] > prefillChunkSize) {
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
                        ret = model->ForwardV2(1, curInput, curAttentionMasks,
                                                  curPositionIdsVec, curSeqLens, curPastKeyValues, generationConfigs,
                                                  tokensManager, &logits);
                        st += curLen;
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
                    ret = model->ForwardV2(seqLens.size(), inputIds, attentionMasks,
                                              positionIds, seqLens, pastKeyValues, generationConfigs,
                                              tokensManager, &logits);
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

                forwardLocker.unlock();
                dictLocker.lock();

                if (maxTotalLens == 0 && pastKeyValues.size() > (size_t)model->kvCacheId) {
                    auto &kv = *pastKeyValues[model->kvCacheId].first;
                    if (kv.pagedKVCacheData != nullptr) {
                        totalPages = kv.pagedKVCacheData->maxPages;
                        pageLen = kv.pagedKVCacheData->pageLen;
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
                    auto &it = *model->responseContextDict.dicts.find(handles[i]);
                    int curRet = ret[i];
                    if (curRet == model->eos_token_id || model->eos_token_ids.find(curRet) != model->eos_token_ids.end()) {
                        it.second->isEnding = true;
                        it.second->TryRecordPagedCache();
                        printf("[Handle %d] Finished. Reason: eos token (token_id=%d), total tokens: %d.\n", handles[i], curRet, it.second->curTokens);
                    } else {
                        auto itStopTk = it.second->generationConfig.stop_token_ids.find(curRet);
                        if (itStopTk != it.second->generationConfig.stop_token_ids.end()) {
                            it.second->isEnding = true;
                            it.second->TryRecordPagedCache();
                            printf("[Handle %d] Finished. Reason: stop token (token_id=%d), total tokens: %d.\n", handles[i], curRet, it.second->curTokens);
                        }
                    }
                    if (it.second->isEnding == false) {
                        it.second->currentTokens = std::vector<int>{curRet};
                        it.second->resultTokenQueue.push(curRet);
                        it.second->allTokens.push_back(curRet);
                        it.second->tokens.Push(curRet);
                        it.second->curTokens++;
                        if (it.second->curTokens == it.second->generationConfig.output_token_limit) {
                            it.second->isEnding = true;
                            it.second->TryRecordPagedCache();
                            printf("[Handle %d] Finished. Reason: output token limit reached (curTokens=%d, limit=%d).\n",
                                   handles[i], it.second->curTokens, it.second->generationConfig.output_token_limit);
                        } else if (it.second->allTokens.size() >= model->max_positions) {
                            it.second->isEnding = true;
                            it.second->TryRecordPagedCache();
                            printf("[Handle %d] Finished. Reason: max positions reached (allTokens=%d, max_positions=%d).\n",
                                   handles[i], (int)it.second->allTokens.size(), model->max_positions);
                        }
                    }
                }
            } else {
                // 没有任何请求可以调度时，等待新请求
            }

            for (int i = 0; i < attentionMasks.size(); i++) {
                delete attentionMasks[i];
            }
            for (int i = 0; i < positionIds.size(); i++) {
                delete positionIds[i];
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
                    mainLoop = new std::thread([](basellm *model) {
                        model->NewMainLoop();
                    }, this);
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

                    int unitSize = (model->dataType == DataType::FLOAT32 ? 4 : 2);
                    int maxTotalLens = kvCacheLimit / (model->elementsInKVCachePerToken * unitSize);
                    if (model->elementsInKVCachePerToken <= 0) {
                        maxTotalLens = kvCacheLimit / 1024 / 1024;
                    }
                    if (model->tokensLimit > 0) {
                        maxTotalLens = model->tokensLimit;
                    }

                    int maxBatch = std::max(1, std::min(512, maxTotalLens / 128));
                    if (model->maxBatch > 0) {
                        maxBatch = model->maxBatch;
                    }
                    
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
                            model->responseContextDict.RemoveHandle(it);
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
                                if (it.second->generationConfig.output_logits) {
                                    it.second->resultLogits.push(new std::vector<float>());
                                    logits.push_back(it.second->resultLogits.back());
                                } else {
                                    logits.push_back(nullptr);
                                }

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
auto st = std::chrono::system_clock::now();
//ClearProfiler();
                            if (seqLens.size() > 1) {
                                if (!model->canDoBatchForward) {
                                    dictLocker.lock();
                                    for (int i = 0; i < handles.size(); i++) {
                                        Data inputIdNow = Data(DataType::FLOAT32, {1, 1}, {ids[i]});
                                        ret.push_back(model->Forward(inputIdNow,
                                                            attentionMasks[i] == nullptr ? Data() : *attentionMasks[i],
                                                            *positionIds[i],
                                                            model->responseContextDict.dicts[handles[i]]->pastKeyValues, 
                                                            generationConfigs[i], tokensManager, logits[i]));
                                    }
                                    dictLocker.unlock();
                                } else {
                                    ret = model->ForwardBatch(seqLens.size(), inputIds, attentionMasks,
                                                            positionIds, seqLens, pastKeyValues, generationConfigs,
                                                            tokensManager, &logits);
                                }
                            } else {
                                int first, part;
                                if (model->chunkedPrefillSize >= 0) {
                                    first = part = model->chunkedPrefillSize;
                                } else {
                                    first = part = 8192;
                                    if (model->model_struct == "deepseek_v2") {
                                        first = part = 2048;
                                    }
                                    if (model->model_struct == "qwen3_next") {
                                        first = part = 2048;
                                    }
                                }
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
                                    if (model->responseContextDict.dicts.begin()->second->multimodalInput.size() > 0) {
                                        auto context = model->responseContextDict.dicts.begin()->second;
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
//PrintProfiler();
/*int total = 0;
for (int i : seqLens) total += i;
float spend = GetSpan(st, std::chrono::system_clock::now());
printf("len = %d, spend = %f s. tokens / s = %f\n", (int)total, spend, (float)total / spend);
*/
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
                                    it.second->currentTokens = std::vector<int>{curRet};
                                    it.second->resultTokenQueue.push(curRet);
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
        mainLoopLocker.unlock();

        dictLocker.lock();
        int handleId = responseContextDict.CreateHandle();
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        context->Init(this->block_cnt, this->dataType);
        context->currentTokens = inputTokens;
        context->allTokens = inputTokens;
        context->generationConfig = generationConfig;
        context->multimodalInput = multimodalInput;
        context->tokens = LastTokensUnit(generationConfig.last_n);

        auto cache = pastKVCacheManager.Get(inputTokens);
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
                        responseContextDict.RemoveHandle(handleId);
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
                        responseContextDict.RemoveHandle(handleId);
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
            pastKeyValues.push_back(std::make_pair(Data(this->dataType), Data(this->dataType)));
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

    void basellm::SetDataType(DataType dataType) {
        if (dataType == DataType::FLOAT32) {

        } else if (dataType == DataType::FLOAT16) {
            AssertInFastLLM(this->use_new_engine ||
                            this->model_struct == "chatglm" || 
                            this->model_struct == "llama" ||
                            this->model_struct == "graph" ||
                            this->model_struct == "cogvlm" ||
                            this->model_struct == "deepseek_v2" ||
                            this->model_struct == "qwen3_moe" ||
                            this->model_struct == "minimax_m2" ||
                            this->model_struct == "hunyuan" || 
                            this->model_struct == "ernie4_5" || 
                            this->model_struct == "pangu_moe" ||
                            this->model_struct == "glm4_moe" ||
                            this->model_struct == "qwen3_next",  
                            this->model_struct + " doesn't support float16");
        } else if (dataType == DataType::BFLOAT16) {
            AssertInFastLLM(this->use_new_engine ||
                            this->model_struct == "chatglm" || 
                            this->model_struct == "llama" ||
                            this->model_struct == "graph" ||
                            this->model_struct == "cogvlm" ||
                            this->model_struct == "deepseek_v2" ||
                            this->model_struct == "qwen3_moe" ||
                            this->model_struct == "minimax_m2" ||
                            this->model_struct == "hunyuan" || 
                            this->model_struct == "ernie4_5" || 
                            this->model_struct == "pangu_moe" ||
                            this->model_struct == "glm4_moe" ||
                            this->model_struct == "qwen3_next",  
                            this->model_struct + " doesn't support bfloat16");
        } else {
            ErrorInFastLLM("SetDataType Error: datatype should be float32, float16 or bfloat16");
        }
        this->dataType = dataType;
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

    void basellm::ResetLogitsOfEOS(int batch, Data *logits, std::vector <std::pair <Data, Data> > &pastKeyValues, 
            const GenerationConfig &generationConfig) {
        auto &config = generationConfig;
        if (logits->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            bool need_reset = false;
            std::vector<int> res_lens, eos_nums, eos_ids;
            for (int b = 0; b < batch; b++) {
                res_lens.push_back(config.output_token_least - pastKeyValues[0].first.dims[1] + config.input_token_length);
                need_reset |= res_lens.back() > 0;
                eos_nums.push_back(1 + this->eos_token_ids.size() + config.stop_token_ids.size());
                eos_ids.push_back(this->eos_token_id);
                for (auto id: this->eos_token_ids)
                    eos_ids.push_back(id);
                for (auto id: config.stop_token_ids)
                    eos_ids.push_back(id);
            }
            if (need_reset) {
                ToDataType(*logits, DataType::FLOAT32);
                FastllmResetLogitsOfEOS(batch, logits, res_lens, eos_nums, eos_ids);
            }
#endif
        } else {
            for (int b = 0; b < batch; b++) {
                if (config.output_token_least > pastKeyValues[0].first.dims[1] - config.input_token_length) {
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
        if (logits->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            bool need_reset = false;
            std::vector<int> res_lens, eos_nums, eos_ids;
            for (int b = 0; b < batch; b++) {
                auto &config = generationConfigs[b];
                res_lens.push_back(config.output_token_least - pastKeyValues[0].first->dims[1] + config.input_token_length);
                need_reset |= res_lens.back() > 0;
                eos_nums.push_back(1 + this->eos_token_ids.size() + config.stop_token_ids.size());
                eos_ids.push_back(this->eos_token_id);
                for (auto id: this->eos_token_ids)
                    eos_ids.push_back(id);
                for (auto id: config.stop_token_ids)
                    eos_ids.push_back(id);
            }
            if (need_reset) {
                ToDataType(*logits, DataType::FLOAT32);
                FastllmResetLogitsOfEOS(batch, logits, res_lens, eos_nums, eos_ids);
            }
#endif
        } else {
            for (int b = 0; b < batch; b++) {
                auto &config = generationConfigs[b];
                if (config.output_token_least > pastKeyValues[0].first->dims[1] - config.input_token_length) {
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
}
