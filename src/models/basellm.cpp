//
// Created by huangyuyang on 6/25/23.
//

#include "basellm.h"
#include "utils.h"
#include <sstream>
#include <cstring>

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

    int basellm::LaunchResponseTokens(const std::vector<int> &inputTokens,
                                      const fastllm::GenerationConfig &generationConfig,
                                      const std::map <std::string, std::vector <Data*> > &multimodalInput) {
        mainLoopLocker.lock();
        if (mainLoop == nullptr) {
            if (mainLoop == nullptr) {
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
                                int first = 8192, part = 2048;
                                if (model->model_struct == "deepseek_v2") {
                                    // TODO: ds_v2支持更长的切片
                                    first = 1024;
                                    part = 1024;
                                    if (GetEnableAMX()) {
                                        first = 2048;
                                        part = 2048;
                                    }
                                }
                                if (model->model_struct == "qwen3_next") {
                                    // TODO: qwen3_next支持更长的切片
                                    first = 2048;
                                    part = 1024;
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
                                    printf("alive = %d, pending = %d, contextLen = %d, Speed: %f tokens / s.\n", alive, pending, aliveLen, (float)genTokens / spend);
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
                        responseContextDict.RemoveHandle(handleId);
                        dictLocker.unlock();
                        dictCV.notify_one();
                        if (context->error == ResponseContextErrorNone) {
                            return -1;
                        } else if (context->error == ResponseContextErrorPromptTooLong) {
                            return -2;
                        } else {
                            return -1;
                        }
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

    void basellm::SetDataType(DataType dataType) {
        if (dataType == DataType::FLOAT32) {

        } else if (dataType == DataType::FLOAT16) {
            AssertInFastLLM(this->model_struct == "chatglm" || 
                            this->model_struct == "llama" ||
                            this->model_struct == "graph" ||
                            this->model_struct == "cogvlm" ||
                            this->model_struct == "deepseek_v2" ||
                            this->model_struct == "qwen3_moe" ||
                            this->model_struct == "hunyuan" || 
                            this->model_struct == "ernie4_5" || 
                            this->model_struct == "pangu_moe" ||
                            this->model_struct == "glm4_moe" ||
                            this->model_struct == "qwen3_next",  
                            this->model_struct + " doesn't support float16");
        } else {
            ErrorInFastLLM("SetDataType Error: datatype should be float32 or float16");
        }
        this->dataType = dataType;
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
