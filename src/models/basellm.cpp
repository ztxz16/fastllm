//
// Created by huangyuyang on 6/25/23.
//

#include "basellm.h"
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

    void ResponseContext::Init(int blocks) {
        pastKeyValues.clear();
        for (int i = 0; i < blocks; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        intParams.clear();
        currentTokens.clear();
        while (resultTokenQueue.size() > 0){
            resultTokenQueue.pop();
        }
        isEnding = false;
        preTokens = 0;
    }

    std::string basellm::Response(const std::string &input, RuntimeResult retCb,
                                  const fastllm::GenerationConfig &generationConfig) {
#ifdef USE_CUDA
        FastllmCudaClearBigBuffer();
#endif
        std::string prompt = input;
#ifdef PY_API
        size_t pos = input.find_last_of("time_stamp:");
        std::string prompt = (generationConfig.enable_hash_id && pos != std::string::npos) ? input.substr(0, pos - 10) : input;
        size_t hash_id = std::hash<std::string>{}(input);
#endif
        Data inputIds, attentionMask, positionIds;

        Data inputTokenData = this->weight.tokenizer.Encode(prompt);
        std::vector<std::vector<float> > inputTokens;
        inputTokens.resize(1);
        for (int i = 0; i < inputTokenData.Count(0); i++) {
            inputTokens[0].push_back(((float *) inputTokenData.cpuData)[i]);
        }
        std::vector<std::pair<Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }

        std::string retString = "";
        std::vector<float> results;
        LastTokensManager tokens(1, generationConfig.last_n);
        int promptLen = inputTokens[0].size(), index = 0;
        FillLMMInputs(inputTokens, {{"promptLen", promptLen}, {"index", index}}, inputIds, attentionMask, positionIds);
        while (true) {
            auto st = std::chrono::system_clock::now();
            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
            tokens.units[0].Push(ret);
            if (ret == eos_token_id) {
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
                        ss << retString << "hash_id:" << hash_id;
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
            FillLMMInputs(inputTokens, {{"promptLen", promptLen}, {"index", index}}, inputIds, attentionMask, positionIds);
            if (index == generationConfig.output_token_limit) {
                break;
            }
            // printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));
        }
        if (retCb)
#ifdef PY_API
            {
                if (generationConfig.enable_hash_id) {
                    std::stringstream ss;
                    ss << retString << "hash_id:" << hash_id;
                    retCb(-1, pybind11::bytes(ss.str()));
                } else {
                    retCb(-1, pybind11::bytes(retString));
                }
            }
#else
            retCb(-1, retString.c_str());
#endif
        return retString;
    }

    void basellm::ResponseBatch(const std::vector<std::string> &inputs, std::vector<std::string> &outputs,
                                RuntimeResultBatch retCb, const fastllm::GenerationConfig &generationConfig) {
#ifdef USE_CUDA
        FastllmCudaClearBigBuffer();
#endif
        // 1. first
        Data inputIds, attentionMask, positionIds;

        int batch = inputs.size();
        outputs.clear();
        outputs.resize(batch, "");

        std::vector<std::vector<float> > inputTokens;
        inputTokens.resize(batch);

        for (int i = 0; i < batch; i++) {
            Data now = this->weight.tokenizer.Encode(inputs[i]);
            for (int j = 0; j < now.Count(0); j++) {
                inputTokens[i].push_back(((float *) now.cpuData)[j]);
            }
        }

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }

        std::vector <std::map <std::string, int> > params;
        params.resize(batch);
        for (int i = 0; i < batch; i++) {
            params[i]["promptLen"] = (int)inputTokens[i].size();
        }
        params[0]["index"] = 0;
        int index = 0;

        LastTokensManager tokensManager (batch, generationConfig.last_n);
        std::vector <bool> isEnding = std::vector <bool> (batch, false);
        FillLMMInputsBatch(inputTokens, params, inputIds, attentionMask, positionIds);
        while (true) {
            auto st = std::chrono::system_clock::now();
            std::vector <int> ret = ForwardBatch(batch, inputIds, attentionMask, positionIds, pastKeyValues,
                                                 generationConfig, tokensManager);
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
                if (ret[i] == eos_token_id) {
                    isEnding[i] = true;
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
                retCb(index, curStrings);
            index++;
            params[0]["index"] = index;
            FillLMMInputsBatch(inputTokens, params, inputIds, attentionMask, positionIds);
            // printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));

            if (index == generationConfig.output_token_limit) {
                break;
            }
        }

        if (retCb)
            retCb(-1, outputs);
    }

    std::vector<int> basellm::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                           const fastllm::Data &positionIds,
                                           std::vector<std::pair<Data, Data>> &pastKeyValues,
                                           const fastllm::GenerationConfig &generationConfig,
                                           const fastllm::LastTokensManager &lastTokens) {
        printf("Unsupport forward batch.\n");
        exit(0);
    }

    // 根据输入的tokens生成LLM推理的输入
    void basellm::FillLMMInputs(std::vector <std::vector <float> > &inputTokens,
                               const std::map <std::string, int> &params,
                               Data &inputIds, Data &attentionMask, Data &positionIds) {
    }

    // 根据输入的tokens生成LLM推理的输入
    void basellm::FillLMMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                     const std::vector<std::map<std::string, int>> &params, fastllm::Data &inputIds,
                                     fastllm::Data &attentionMask, fastllm::Data &positionIds) {
    }
}