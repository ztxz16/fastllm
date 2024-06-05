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
        while (resultTokenQueue.size() > 0){
            resultTokenQueue.pop();
        }
        isEnding = false;
        preTokens = 0;
    }

    PastKVCacheMemory::PastKVCacheMemory(const std::string &prompt, int tokens, long long flushTime, std::vector<std::pair<Data, Data> > *kv) {
        this->prompt = prompt;
        this->tokens = tokens;
        this->flushTime = flushTime;
        this->recordTimes = 1;
        auto dataType = (*kv)[0].first.dataType;
        for (int i = 0; i < kv->size(); i++) {
            this->kv.push_back(std::make_pair(Data(dataType), Data(dataType)));
        }
        for (int i = 0; i < kv->size(); i++) {
            (*kv)[i].first.ToDevice(DataDevice::CPU);
            (*kv)[i].second.ToDevice(DataDevice::CPU);

            this->kv[i].first.CopyFrom((*kv)[i].first);
            this->kv[i].second.CopyFrom((*kv)[i].second);
        }
    }

    void PastKVCacheManager::SetMaxRecordNum(int maxRecordNum) {
        std::lock_guard <std::mutex> lock(this->locker);
        this->maxRecordNum = maxRecordNum;
    }

    void PastKVCacheManager::Record(const std::string &prompt, int tokens, std::vector<std::pair<Data, Data> > *kv) {
        std::lock_guard <std::mutex> lock(this->locker);
        if (this->memorys.find(prompt) != this->memorys.end()) {
            this->memorys[prompt]->recordTimes++;
            this->memorys[prompt]->flushTime = ++flushTime;
            return;
        }

        if (this->memorys.size() >= this->maxRecordNum) {
            std::string prompt = "";
            long long minFlushTime = (1LL << 60);
            for (auto &it : this->memorys) {
                if (it.second->flushTime < minFlushTime) {
                    minFlushTime = it.second->flushTime;
                    prompt = it.first;
                }
            }
            delete this->memorys[prompt];
            this->memorys.erase(this->memorys.find(prompt));
        }

        this->memorys[prompt] = new PastKVCacheMemory(prompt, tokens, ++flushTime, kv);
    }

    void PastKVCacheManager::Remove(std::string prompt) {
        std::lock_guard <std::mutex> lock(this->locker);
        if (this->memorys.find(prompt) != this->memorys.end()) {
            if ((--this->memorys[prompt]->recordTimes) <= 0) {
                delete this->memorys[prompt];
                this->memorys.erase(this->memorys.find(prompt));
            }
        }
    }

    PastKVCacheMemory *PastKVCacheManager::Get(const std::string &prompt) {
        locker.lock();
        std::string maxPrompt = "";
        for (auto &it : this->memorys) {
            const std::string &cur = it.first;
            if (cur.size() > maxPrompt.size() && cur.size() <= prompt.size() && prompt.substr(0, cur.size()) == cur) {
                maxPrompt = cur;
            }
        }
        if (maxPrompt == "") {
            return nullptr;
        }
        this->memorys[maxPrompt]->flushTime = ++this->flushTime;
        return this->memorys[maxPrompt];
    }

    void PastKVCacheManager::Unlock() {
        locker.unlock();
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
        std::vector<std::vector<float> > inputTokens;
        inputTokens.resize(1);
        for (int i = 0; i < inputTokenData.Count(0); i++) {
            inputTokens[0].push_back(((float *) inputTokenData.cpuData)[i]);
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
        LastTokensManager tokens(1, generationConfig.last_n);
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

        std::vector<std::vector<float> > inputTokens;
        inputTokens.resize(batch);

        for (int i = 0; i < batch; i++) {
            Data now = this->weight.tokenizer.Encode(prompts[i]);
            for (int j = 0; j < now.Count(0); j++) {
                inputTokens[i].push_back(((float *) now.cpuData)[j]);
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

        LastTokensManager tokensManager (batch, generationConfig.last_n);
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

    int basellm::LaunchResponseTokens(const std::vector<int> &inputTokens,
                                      const fastllm::GenerationConfig &generationConfig) {
/*
        mainLoopLocker.lock();
        if (mainLoop == nullptr) {
            if (mainLoop == nullptr) {
                mainLoop = new std::thread([](basellm *model) {
                    while (true) {
                        model->dictLocker.lock();
                        std::vector <int> handles;
                        std::vector<std::vector<float> > inputTokens;
                        std::vector <std::map <std::string, int> > params;
                        std::vector <GenerationConfig> generationConfigs;

                        int index = 0;
                        int cnt = 0;
                        std::vector <std::pair <int, int> > lenIdVector;
                        for (auto &it : model->responseContextDict.dicts) {
                            if (it.second->isEnding) {
                                continue;
                            }
                            lenIdVector.push_back(std::make_pair(it.second->generationConfig.output_token_limit,
                                                                it.first));
                        }

                        std::sort(lenIdVector.begin(), lenIdVector.end());
                        std::set <int> currentIds;
                        int maxInput = 0;
                        for (int i = 0; i < lenIdVector.size(); i++) {
                            maxInput = std::max(maxInput,
                                                (int)model->responseContextDict.dicts[lenIdVector[i].second]->currentTokens.size());
                            if ((maxInput + lenIdVector[i].first) * (i + 1) > 512 * 256) {
                                break;
                            }
                            currentIds.insert(lenIdVector[i].second);
                        }

                        int maxOutputLimit = 0;
                        for (auto &it: model->responseContextDict.dicts) {
                            if (it.second->isEnding) {
                                continue;
                            }
                            if (currentIds.find(it.first) == currentIds.end()) {
                                continue;
                            }

                            maxOutputLimit = std::max(maxOutputLimit, it.second->generationConfig.output_token_limit);
                            generationConfigs.push_back(it.second->generationConfig);
                            handles.push_back(it.first);
                            if (it.second->preTokens == 0) {
                                it.second->intParams["promptLen"] = it.second->currentTokens.size();
                                it.second->intParams["index"] = 0;
                            } else {
                                it.second->intParams["index"]++;
                            }

                            inputTokens.push_back(std::vector <float> ());
                            for (int i : it.second->currentTokens) {
                                inputTokens.back().push_back(i);
                            }
                            params.push_back(std::map <std::string, int> ());
                            params.back()["promptLen"] = it.second->currentTokens.size();
                            params.back()["index"] = 0;
                            it.second->preTokens += (int)inputTokens.back().size();

                            //if (inputTokens.size() == 64) {
                              //  break;
                            //}
                        }

                        if (inputTokens.size() > 0) {
                            model->dictLocker.unlock();
#ifdef USE_CUDA
                            FastllmCudaClearBigBuffer();
#endif
                            int batch = (int)inputTokens.size();
                            int last_n = 64; // TODO: 使用真实数据

                            std::vector <int> ret;
                            ret.resize(batch);

                            std::vector <std::vector <std::pair <Data, Data> > > *pkvPointer = new std::vector <std::vector <std::pair <Data, Data> > >();
                            std::vector <std::vector <std::pair <Data, Data> > > &pastKeyValuess = *pkvPointer;
                            pastKeyValuess.resize(batch);
                            for (int b = 0; b < batch; b++) {
printf("%d / %d, (%d + %d = %d)\n", b, batch, inputTokens[b].size(), generationConfigs[b].output_token_limit, inputTokens[b].size() + generationConfigs[b].output_token_limit);
                                Data inputIds, attentionMask, positionIds;
                                std::vector<std::pair<Data, Data> > &pastKeyValues = pastKeyValuess[b];
                                for (int i = 0; i < model->block_cnt; i++) {
                                    pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                                           Data(DataType::FLOAT32)));
                                }

                                LastTokensManager tokens(1, generationConfigs[b].last_n);
                                int promptLen = inputTokens[b].size(), index = 0;
                                std::vector <std::vector <float> > curInputTokens = {inputTokens[b]};
                                model->FillLLMInputs(curInputTokens, {{"promptLen", promptLen}, {"index", index}}, inputIds, attentionMask, positionIds);
                                ret[b] = model->Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfigs[b], tokens);
                            }

                            Data inputIds, attentionMask, positionIds;
                            LastTokensManager tokensManager (batch, last_n);
                            std::vector <bool> isEnding = std::vector <bool> (batch, false);
                            std::vector <std::pair <Data, Data> > pastKeyValues;

                            for (int i = 0; i < model->block_cnt; i++) {
                                pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32), Data(DataType::FLOAT32)));
                            }

                            for (int i = 0; i < model->block_cnt; i++) {
                                auto &key = pastKeyValues[i].first;
                                auto &value = pastKeyValues[i].second;
                                std::vector <int> dims = pastKeyValuess[0][i].first.dims;
                                for (int b = 1; b < batch; b++) {
                                    dims[0] += pastKeyValuess[b][i].first.dims[0];
                                    dims[1] = std::max(dims[1], pastKeyValuess[b][i].first.dims[1]);
                                }
                                std::vector <int> expandDims = dims;
                                expandDims[1] += maxOutputLimit;

                                key.ToDevice(DataDevice::CUDA);
                                value.ToDevice(DataDevice::CUDA);
                                key.Expansion(dims);
                                value.Expansion(dims);
                                key.Resize(dims);
                                value.Resize(dims);

                                int bs = dims[0], perbs = bs / batch, len = dims[1], inner = dims[2];
                                for (int b = 0; b < batch; b++) {
                                    Data &oldKey = pastKeyValuess[b][i].first;
                                    Data &oldValue = pastKeyValuess[b][i].second;

                                    CopyKVCache(oldKey, key, 0, b * perbs, perbs, (dims[1] - oldKey.dims[1]));
                                    CopyKVCache(oldValue, value, 0, b * perbs, perbs, (dims[1] - oldValue.dims[1]));
                                }
                            }

                            delete pkvPointer;
                            std::vector <std::vector <int> > results;
                            results.resize(batch);

                            bool first = true;
                            GenerationConfig config;
                            while (true) {
                                if (first) {
                                    first = false;
                                } else {
auto st = std::chrono::system_clock::now();
                                    ret = model->ForwardBatch(batch, inputIds, attentionMask, positionIds,
                                                              pastKeyValues, config, tokensManager);
printf("batch = %d, spend = %f s.\n", batch, GetSpan(st, std::chrono::system_clock::now()));
                                }
                                for (int i = 0; i < batch; i++) {
                                    tokensManager.units[i].Push(ret[i]);
                                }
                                std::vector <float> fret;
                                int endingCount = 0;
                                std::vector <std::string> curStrings;
                                for (int i = 0; i < batch; i++) {
                                    fret.push_back(ret[i]);
                                    inputTokens[i] = std::vector <float> {(float)ret[i]};
                                    if (ret[i] == model->eos_token_id || (results[i].size() >= generationConfigs[i].output_token_limit)) {
                                        isEnding[i] = true;
                                    }
                                    if (isEnding[i]) {
                                        endingCount++;
                                        continue;
                                    }
                                    results[i].push_back(ret[i]);
                                }
printf("%d / %d\n", endingCount, batch);
                                if (endingCount == batch) {
                                    break;
                                }

                                params[0]["index"]++;
                                model->FillLLMInputsBatch(inputTokens, params, inputIds, attentionMask, positionIds);
                            }

                            model->dictLocker.lock();
                            for (int i = 0; i < handles.size(); i++) {
                                auto &it = *model->responseContextDict.dicts.find(handles[i]);
                                for (int token : results[i]) {
                                    it.second->resultTokenQueue.push(token);
                                }
                                it.second->isEnding = true;
                            }
                        }

                        model->dictLocker.unlock();
                        MySleep(0);
                    }
                }, this);
            }
        }
        mainLoopLocker.unlock();
*/
        mainLoopLocker.lock();
        if (mainLoop == nullptr) {
            if (mainLoop == nullptr) {
                mainLoop = new std::thread([](basellm *model) {
                    while (true) {
                        std::vector <Data*> attentionMasks;
                        std::vector <Data*> positionIds;
                        std::vector <std::pair <Data*, Data*> > pastKeyValues;
                        std::vector <float> ids;
                        std::vector <int> seqLens;
                        std::vector <int> handles;
                        std::vector <GenerationConfig> generationConfigs;
                        LastTokensManager tokensManager;
                        std::vector <std::vector <float>* > logits;
                        model->dictLocker.lock();

                        int limit = model->tokensLimit > 0 ? model->tokensLimit : 1e9;
                        int lenSum = 0;
                        for (auto &it: model->responseContextDict.dicts) {
                            if (it.second->pastKeyValues[0].first.expansionDims.size() > 0 && !it.second->isEnding) {
                                lenSum += it.second->pastKeyValues[0].first.expansionDims[1];
                            }
                        }

                        for (int isPrompt = 1; isPrompt >= 0; isPrompt--) {
                            int cnt = 0;
                            if (isPrompt == 0 && seqLens.size() > 0) {
                                continue;
                            }
                            if (lenSum > limit && isPrompt) {
                                continue;
                            }

                            for (auto &it: model->responseContextDict.dicts) {
                                if (it.second->isEnding) {
                                    continue;
                                }
                                if (isPrompt && it.second->preTokens != 0) {
                                    continue;
                                }
                                if (!isPrompt && it.second->preTokens == 0) {
                                    continue;
                                }

                                int outputLimit = it.second->generationConfig.output_token_limit;
                                outputLimit = (outputLimit < 0 ? 128 : outputLimit);
                                if (isPrompt && lenSum + it.second->currentTokens.size() + outputLimit > limit) {
                                    continue;
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
                                    it.second->intParams["promptLen"] = it.second->currentTokens.size();
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
                                    break;
                                }
                            }
                        }
                        if (seqLens.size() > 0) {
                            std::vector <std::pair <Data, Data> > *pastKeyValue1;
                            if (seqLens.size() == 1) {
                                pastKeyValue1 = &model->responseContextDict.dicts[handles[0]]->pastKeyValues;
                            }
                            model->dictLocker.unlock();
#ifdef USE_CUDA
                            FastllmCudaClearBigBuffer();
#endif
                            Data inputIds = Data(DataType::FLOAT32, {1, (int) ids.size()}, ids);
                            std::vector<int> ret;
auto st = std::chrono::system_clock::now();
//ClearProfiler();
                            if (seqLens.size() > 1) {
                                ret = model->ForwardBatch(seqLens.size(), inputIds, attentionMasks,
                                                          positionIds, seqLens, pastKeyValues, generationConfigs,
                                                          tokensManager, &logits);
                            } else {
                                ret = std::vector <int> {model->Forward(inputIds,
                                                                        attentionMasks[0] == nullptr ? Data() : *attentionMasks[0],
                                                                        *positionIds[0],
                                                                        *pastKeyValue1, generationConfigs[0], tokensManager, logits[0])};
                            }
/*
PrintProfiler();
int total = 0;
for (int i : seqLens) total += i;
float spend = GetSpan(st, std::chrono::system_clock::now());
printf("len = %d, spend = %f s. tokens / s = %f\n", (int)total, spend, (float)total / spend);
*/
                            model->dictLocker.lock();
                            for (int i = 0; i < handles.size(); i++) {
                                auto &it = *model->responseContextDict.dicts.find(handles[i]);
                                int curRet = ret[i];
                                if (curRet == model->eos_token_id || model->eos_token_ids.find(curRet) != model->eos_token_ids.end()) {
                                    it.second->isEnding = true;
                                } else {
                                    auto itStopTk = it.second->generationConfig.stop_token_ids.find(curRet);
                                    if (itStopTk != it.second->generationConfig.stop_token_ids.end()) {
                                            it.second->isEnding = true;
                                    }
                                }
                                if (it.second->isEnding == false) {
                                    it.second->currentTokens = std::vector<int>{curRet};
                                    it.second->resultTokenQueue.push(curRet);
                                    it.second->tokens.Push(curRet);
                                    it.second->curTokens++;
                                    if (it.second->curTokens == it.second->generationConfig.output_token_limit) {
                                        it.second->isEnding = true;
                                    }
                                }
                            }
                        }

                        for (int i = 0; i < attentionMasks.size(); i++) {
                            delete attentionMasks[i];
                        }
                        for (int i = 0; i < positionIds.size(); i++) {
                            delete positionIds[i];
                        }

                        model->dictLocker.unlock();
                        MySleep(0);
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
        context->generationConfig = generationConfig;
        context->tokens = LastTokensUnit(generationConfig.last_n);
        dictLocker.unlock();
        return handleId;
    }

    int basellm::FetchResponseTokens(int handleId) {
        dictLocker.lock();
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            dictLocker.unlock();
            return -1;
        } else {
            while (true) {
                if (context->resultTokenQueue.size() > 0) {
                    int ret = context->resultTokenQueue.front();
                    context->resultTokenQueue.pop();
                    dictLocker.unlock();
                    return ret;
                } else {
                    if (context->isEnding) {
                        responseContextDict.RemoveHandle(handleId);
                        dictLocker.unlock();
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
        dictLocker.lock();
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            dictLocker.unlock();
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
                    dictLocker.unlock();
                    return ret;
                } else {
                    if (context->isEnding) {
                        responseContextDict.RemoveHandle(handleId);
                        dictLocker.unlock();
                        return -1;
                    }
                }
                dictLocker.unlock();
                MySleep(0);
                dictLocker.lock();
            }
        }
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

            std::vector <float> vmask = std::vector <float> (seqLen * promptLen, 0);
            std::vector <float> vpids = std::vector <float> (seqLen, 0);
            for (int i = 0; i < seqLen; i++) {
                vpids[i] = promptLen - seqLen + i;
                for (int j = i + 1; j < seqLen; j++) {
                    vmask[i * promptLen + (promptLen - seqLen + j)] = 1;
                }
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, inputTokens[0]));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {seqLen, promptLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vpids));
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
        if (this->model_type == "llama" || 
            this->model_type == "moe" || 
            this->model_type == "internlm" || 
            this->model_type == "qwen2_moe" || 
            this->model_type == "deepseek_v2" ||
            this->model_type == "qwen") {
            this->saveHistoryChat = save;
            return true;
        }
        return false;
    }

    void basellm::SetDataType(DataType dataType) {
        if (dataType == DataType::FLOAT32) {

        } else if (dataType == DataType::FLOAT16) {
            AssertInFastLLM(this->model_type == "chatglm" || this->model_type == "llama", 
                            this->model_type + " doesn't support float16");
        } else {
            ErrorInFastLLM("SetDataType Error: datatype should be float32 or float16");
        }
        this->dataType = dataType;
    }

    JinjaVar ChatMessagesToJinjaVar(const ChatMessages &messages) {
        JinjaVar ret = {{"messages", fastllm::JinjaArray {}}};
        for (auto &message : messages) {
            ret["messages"].arrayValue.push_back({
                {"role", message.first},
                {"content", message.second}
            });
        }
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
}
