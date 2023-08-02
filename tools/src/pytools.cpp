//
// Created by huangyuyang on 6/27/23.
//

#include "model.h"

#include <cstring>

#ifdef WIN32
#define DLL_EXPORT _declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {
    DLL_EXPORT void print_cpu_ins() {
        fastllm::PrintInstructionInfo();
    }

    DLL_EXPORT void set_cpu_threads(int threads) {
        fastllm::SetThreads(threads);
    }

    DLL_EXPORT int get_cpu_threads() {
        return fastllm::GetThreads();
    }

    DLL_EXPORT void set_cpu_low_mem(bool low) {
        fastllm::SetLowMemMode(low);
    }

    DLL_EXPORT bool get_cpu_low_mem(bool low) {
        return fastllm::GetLowMemMode();
    }

    DLL_EXPORT void set_kvcache_in_cpu(bool in) {
        fastllm::SetKVCacheInCPU(in);
    }

    DLL_EXPORT bool get_kvcache_in_cpu() {
        return fastllm::GetKVCacheInCPU();
    }

    DLL_EXPORT void set_device_map(int device_cnt, int *lens, char *devices, int *values) {
        std::map <std::string, int> deviceMap;
        int cur = 0;
        for (int i = 0; i < device_cnt; i++) {
            std::string key = "";
            for (int j = 0; j < lens[i]; j++) {
                key += devices[cur++];
            }
            deviceMap[key] = values[i];
        }
        fastllm::SetDeviceMap(deviceMap);
    }

    DLL_EXPORT struct ModelManager {
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

    DLL_EXPORT char *string_to_chars(const std::string &s) {
        char *svalue = new char[s.size() + 1];
        memcpy(svalue, s.data(), s.size());
        svalue[s.size()] = 0;
        return svalue;
    }

    DLL_EXPORT fastllm::GenerationConfig make_config(int max_length, bool do_sample, float top_p, int top_k,
                                          float temperature, float repeat_penalty) {
        fastllm::GenerationConfig config;
        config.output_token_limit = max_length;
        config.temperature = temperature;
        config.repeat_penalty = repeat_penalty;
        if (do_sample) {
            config.top_p = top_p;
            config.top_k = top_k;
        }
        return config;
    }

    DLL_EXPORT int create_llm_model(char *path) {
        models.locker.lock();
        int id = models.models.size();
        models.models[id] = fastllm::CreateLLMModelFromFile(path);
        models.locker.unlock();
        return id;
    }

    DLL_EXPORT int create_empty_llm_model(char *type) {
        models.locker.lock();
        int id = models.models.size();
        models.models[id] = fastllm::CreateEmptyLLMModel(type);
        models.locker.unlock();
        return id;
    }

    DLL_EXPORT void add_tokenizer_word_llm_model(int modelId, char *key, int tokenId, float score) {
        auto model = models.GetModel(modelId);
        model->weight.AddTokenizerWord(key, tokenId, score);
        return;
    }

    DLL_EXPORT void add_dict_llm_model(int modelId, char *key, char *value) {
        auto model = models.GetModel(modelId);
        model->weight.AddDict(key, value);
        return;
    }

    DLL_EXPORT void init_params_llm_model(int modelId) {
        auto model = models.GetModel(modelId);
        model->InitParams();
        return;
    }

    DLL_EXPORT void warmup_llm_model(int modelId) {
        auto model = models.GetModel(modelId);
        model->WarmUp();
        return;
    }

    DLL_EXPORT void save_llm_model(int modelId, char *path) {
        auto model = models.GetModel(modelId);
        model->SaveModel(path);
        return;
    }

    DLL_EXPORT void add_weight_llm_model(int modelId, char *key, int dimsLen, void *dimsData,
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

    DLL_EXPORT void add_qlinear_weight_llm_model(int modelId, char *key, int dimsLen, void *dimsData,
                                                 int bit, void *scales, void *oriData) {
        auto model = models.GetModel(modelId);
        std::vector <int> dims = std::vector <int> (dimsLen);
        for (int i = 0; i < dims.size(); i++) {
            dims[i] = ((int*)dimsData)[i];
        }
        model->weight.AddQLinearWeight(key, dims, bit, (float*)scales, (uint8_t*)oriData);
        return;
    }

    DLL_EXPORT char *make_input_llm_model(int modelId, char *history, int round, char *input) {
        auto model = models.GetModel(modelId);
        char *ret = string_to_chars(model->MakeInput(history, round, input));
        return ret;
    }

    DLL_EXPORT char *make_history_llm_model(int modelId, char *history, int round, char *input, char *output) {
        auto model = models.GetModel(modelId);
        return string_to_chars(model->MakeHistory(history, round, input, output));
    }

    DLL_EXPORT char *response_str_llm_model(int modelId, char *content,
                                 int max_length, bool do_sample, float top_p, int top_k,
                                 float temperature, float repeat_penalty) {
        auto model = models.GetModel(modelId);
        auto config = make_config(max_length, do_sample, top_p, top_k, temperature, repeat_penalty);
        std::string s = model->Response(content, nullptr, config);
        return string_to_chars(s);
    }

    DLL_EXPORT int launch_response_str_llm_model(int modelId, char *content,
                                      int max_length, bool do_sample, float top_p, int top_k,
                                      float temperature, float repeat_penalty) {
        auto model = models.GetModel(modelId);
        std::vector <int> tokens;
        auto v = model->weight.tokenizer.Encode(content);
        for (int i = 0; i < v.Count(0); i++) {
            tokens.push_back((int)((float*)v.cpuData)[i]);
        }
        auto config = make_config(max_length, do_sample, top_p, top_k, temperature, repeat_penalty);
        return model->LaunchResponseTokens(tokens, config);
    }

    DLL_EXPORT char *fetch_response_str_llm_model(int modelId, int handleId) {
        auto model = models.GetModel(modelId);
        int ret = model->FetchResponseTokens(handleId);
        std::string s = (ret == -1 ? "<flmeos>" : model->weight.tokenizer.DecodeTokens(std::vector <int> {ret}));
        return string_to_chars(s);
    }

    DLL_EXPORT int launch_response_llm_model(int modelId, int len, int *values,
                                  int max_length, bool do_sample, float top_p, int top_k,
                                  float temperature, float repeat_penalty) {
        std::vector <int> input;
        for (int i = 0; i < len; i++) {
            input.push_back(values[i]);
        }
        auto config = make_config(max_length, do_sample, top_p, top_k, temperature, repeat_penalty);
        auto model = models.GetModel(modelId);
        return model->LaunchResponseTokens(input, config);
    }

    DLL_EXPORT int fetch_response_llm_model(int modelId, int handleId) {
        auto model = models.GetModel(modelId);
        return model->FetchResponseTokens(handleId);
    }
};
