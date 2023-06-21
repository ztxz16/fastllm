#include "utils.h"

#include "model.h"

#include "chatglm.h"
#include "moss.h"
#include "baichuan.h"
#include "llama.h"

namespace fastllm {
    void basellm::LoadFromFile(const std::string &fileName) {
        this->weight.LoadFromFile(fileName);
        if (this->weight.dicts.find("bos_token_id") != this->weight.dicts.end()) {
            this->bos_token_id = atoi(this->weight.dicts["bos_token_id"].c_str());
            this->eos_token_id = atoi(this->weight.dicts["eos_token_id"].c_str());
        }
        if (this->weight.dicts.find("num_hidden_layers") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["num_hidden_layers"].c_str());
        }
        if (this->weight.dicts.find("hidden_size") != this->weight.dicts.end()) {
            embed_dim = atoi(this->weight.dicts["hidden_size"].c_str());
        }
        if (this->weight.dicts.find("num_attention_heads") != this->weight.dicts.end()) {
            num_attention_heads = atoi(this->weight.dicts["num_attention_heads"].c_str());
        }
    }

    void basellm::SaveLowBitModel(const std::string &fileName, int bit) {
        this->weight.SaveLowBitModel(fileName, bit);
    }

    std::unique_ptr<fastllm::basellm> CreateLLMModelFromFile(const std::string &fileName) {
        std::string modelType = GetModelTypeFromFile(fileName);
        basellm *model;
        if (modelType == "chatglm") {
            model = (basellm*)(new ChatGLMModel());
        } else if (modelType == "moss") {
            model = (basellm*)(new MOSSModel());
        } else if (modelType == "baichuan") {
            model = (basellm*)(new BaichuanModel());
        } else if (modelType == "llama") {
            model = (basellm*)(new LlamaModel());
        } else {
            ErrorInFastLLM("Unkown model type: " + modelType);
        }
        model->LoadFromFile(fileName);
        model->WarmUp();
        return std::unique_ptr<fastllm::basellm>(model);
    }
}