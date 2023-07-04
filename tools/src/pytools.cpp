//
// Created by huangyuyang on 6/27/23.
//

#include "model.h"

#include <cstring>

extern "C" {
    struct ModelManager {
        std::mutex locker;
        std::map <int, std::unique_ptr<fastllm::basellm> > models;

        fastllm::basellm *GetModel(int handle) {
            locker.lock();
            auto ret = models[handle].get();
            locker.unlock();
            return ret;
        }
    };

    static ModelManager models;

    char *string_to_chars(const std::string &s) {
        char *svalue = new char[s.size() + 1];
        memcpy(svalue, s.data(), s.size());
        svalue[s.size()] = 0;
        return svalue;
    }

    int create_llm_model(char *path) {
        models.locker.lock();
        int id = models.models.size();
        models.models[id] = fastllm::CreateLLMModelFromFile(path);
        models.locker.unlock();
        return id;
    }

    int create_empty_llm_model(char *type) {
        models.locker.lock();
        int id = models.models.size();
        models.models[id] = fastllm::CreateEmptyLLMModel(type);
        models.locker.unlock();
        return id;
    }

    void add_tokenizer_word_llm_model(int modelId, char *key, int tokenId) {
        auto model = models.GetModel(modelId);
        model->weight.AddTokenizerWord(key, tokenId);
        return;
    }

    void add_dict_llm_model(int modelId, char *key, char *value) {
        auto model = models.GetModel(modelId);
        model->weight.AddDict(key, value);
        return;
    }

    void init_params_llm_model(int modelId) {
        auto model = models.GetModel(modelId);
        model->InitParams();
        return;
    }

    void warmup_llm_model(int modelId) {
        auto model = models.GetModel(modelId);
        model->WarmUp();
        return;
    }

    void save_llm_model(int modelId, char *path) {
        auto model = models.GetModel(modelId);
        model->SaveModel(path);
        return;
    }

    void add_weight_llm_model(int modelId, char *key, int dimsLen, void *dimsData,
                              int dataType, int weightType, int oriDataType, void *oriData) {
        auto model = models.GetModel(modelId);
        std::vector <int> dims = std::vector <int> (dimsLen);
        for (int i = 0; i < dims.size(); i++) {
            dims[i] = ((int*)dimsData)[i];
        }
        model->weight.AddWeight(key, dims,
                                (fastllm::DataType)dataType,
                                (fastllm::WeightType)weightType,
                                (fastllm::DataType)oriDataType,
                                (uint8_t*)oriData);
        return;
    }

    char *make_input_llm_model(int modelId, char *history, int round, char *input) {
        auto model = models.GetModel(modelId);
        char *ret = string_to_chars(model->MakeInput(history, round, input));
        return ret;
    }

    char *make_history_llm_model(int modelId, char *history, int round, char *input, char *output) {
        auto model = models.GetModel(modelId);
        return string_to_chars(model->MakeHistory(history, round, input, output));
    }

    char *response_str_llm_model(int modelId, char *content) {
        auto model = models.GetModel(modelId);
        std::string s = model->Response(content, nullptr);
        return string_to_chars(s);
    }

    int launch_response_str_llm_model(int modelId, char *content) {
        auto model = models.GetModel(modelId);
        std::vector <int> tokens;
        auto v = model->weight.tokenizer.Encode(content);
        for (int i = 0; i < v.Count(0); i++) {
            tokens.push_back((int)((float*)v.cpuData)[i]);
        }

        return model->LaunchResponseTokens(tokens);
    }

    char *fetch_response_str_llm_model(int modelId, int handleId) {
        auto model = models.GetModel(modelId);
        int ret = model->FetchResponseTokens(handleId);
        std::string s = (ret == -1 ? "<flmeos>" : model->weight.tokenizer.DecodeTokens(std::vector <int> {ret}));
        return string_to_chars(s);
    }

    int launch_response_llm_model(int modelId, int len, int *values) {
        std::vector <int> input;
        for (int i = 0; i < len; i++) {
            input.push_back(values[i]);
        }
        auto model = models.GetModel(modelId);
        return model->LaunchResponseTokens(input);
    }

    int fetch_response_llm_model(int modelId, int handleId) {
        auto model = models.GetModel(modelId);
        return model->FetchResponseTokens(handleId);
    }
};