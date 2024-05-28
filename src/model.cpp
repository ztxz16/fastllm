#include "utils.h"
#include "json11.hpp"

#include "model.h"
#include "fastllm.h"
#include <sstream>
#include <fstream>

#include "chatglm.h"
#include "moss.h"
#include "llama.h"
#include "moe.h"
#include "deepseekv2.h"
#include "qwen.h"
#include "glm.h"
#include "minicpm.h"
#include "internlm2.h"
#include "bert.h"

namespace fastllm {
    std::string ReadAllFile(const std::string &fileName) {
        if (access(fileName.c_str(), R_OK) != 0) {
            ErrorInFastLLM("Read error: can't find \"" + fileName + "\".");
        }

        std::ifstream t(fileName.c_str());
        std::string ret((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        t.close();
        return ret;
    }

    void ConvertDataType(uint8_t *src, DataType srcDtype, uint8_t *dst, DataType dstDtype, uint64_t len) {
        if (srcDtype == dstDtype) {
            int unitSize = 4;
            if (dstDtype == DataType::FLOAT32) {
                unitSize = 4;
            } else if (dstDtype == DataType::FLOAT16 || dstDtype == DataType::BFLOAT16) {
                unitSize = 2;
            } else {
                ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");    
            }
            memcpy(dst, src, len * unitSize);
        } else if (srcDtype == DataType::BFLOAT16 && dstDtype == DataType::FLOAT32) {
            uint16_t *u16dst = (uint16_t*)dst;
            uint16_t *u16src = (uint16_t*)src;
            for (int i = 0; i < len; i++) {
                u16dst[i * 2] = 0;
                u16dst[i * 2 + 1] = u16src[i];
            }
        } else {
            ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
        }
    }

    void basellm::LoadFromFile(const std::string &fileName) {
        this->weight.LoadFromFile(fileName);
        this->InitParams();
    }

    void basellm::InitParams() {
        if (this->weight.dicts.find("bos_token_id") != this->weight.dicts.end()) {
            if(this->weight.dicts["bos_token_id"]!="None"){
                this->bos_token_id = atoi(this->weight.dicts["bos_token_id"].c_str());
            }
            if(this->weight.dicts["eos_token_id"]!="None"){
                this->eos_token_id = atoi(this->weight.dicts["eos_token_id"].c_str());
            }
        }
        if (this->weight.dicts.find("im_start_id") != this->weight.dicts.end()) {
            this->bos_token_id = atoi(this->weight.dicts["im_start_id"].c_str());
            this->eos_token_id = atoi(this->weight.dicts["im_end_id"].c_str());
        }
        if (this->weight.dicts.find("num_hidden_layers") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["num_hidden_layers"].c_str());
        }else if (this->weight.dicts.find("num_layers") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["num_layers"].c_str());
        }
        if (this->weight.dicts.find("hidden_size") != this->weight.dicts.end()) {
            embed_dim = atoi(this->weight.dicts["hidden_size"].c_str());
        }
        if (this->weight.dicts.find("num_attention_heads") != this->weight.dicts.end()) {
            num_attention_heads = atoi(this->weight.dicts["num_attention_heads"].c_str());
        }
        if (this->weight.dicts.find("pre_prompt") != this->weight.dicts.end()) {
            pre_prompt = this->weight.dicts["pre_prompt"];
        }
        if (this->weight.dicts.find("user_role") != this->weight.dicts.end()) {
            user_role = this->weight.dicts["user_role"];
        }
        if (this->weight.dicts.find("bot_role") != this->weight.dicts.end()) {
            bot_role = this->weight.dicts["bot_role"];
        }
        if (this->weight.dicts.find("history_sep") != this->weight.dicts.end()) {
            history_sep = this->weight.dicts["history_sep"];
        }
        if (this->weight.dicts.find("tokenizer_add_dummy_prefix") != this->weight.dicts.end()) {
            std::string value = this->weight.dicts["tokenizer_add_dummy_prefix"];
            transform(value.begin(), value.end(), value.begin(), ::tolower);
            std::istringstream iss(value);
            iss >> std::boolalpha >> this->weight.tokenizer.addDummyPrefix;
        }
        if (this->weight.dicts.find("tokenizer_remove_extra_whitespaces") != this->weight.dicts.end()) {
            std::string value = this->weight.dicts["tokenizer_remove_extra_whitespaces"];
            transform(value.begin(), value.end(), value.begin(), ::tolower);
            std::istringstream iss(value);
            iss >> std::boolalpha >> this->weight.tokenizer.removeExtraWhitespaces;
        }
        if (this->weight.dicts.find("tokenizer_byte_as_char") != this->weight.dicts.end()) {
            std::string value = this->weight.dicts["tokenizer_byte_as_char"];
            transform(value.begin(), value.end(), value.begin(), ::tolower);
            std::istringstream iss(value);
            iss >> std::boolalpha >> this->weight.tokenizer.byteAsChar;
        }

        this->deviceMap = GetDeviceMap();
    }

    void basellm::SaveLowBitModel(const std::string &fileName, int bit) {
        this->weight.SaveLowBitModel(fileName, bit);
    }

    void basellm::SaveModel(const std::string &fileName) {
        this->weight.SaveLowBitModel(fileName, 0);
    }

    fastllm::basellm *CreateModelWithType(const std::string &modelType) {
        basellm *model = nullptr;
        if (modelType == "chatglm") {
            model = (basellm*)(new ChatGLMModel());
        } else if (modelType == "moss") {
            model = (basellm*)(new MOSSModel());
            model->weight.tokenizer.type = Tokenizer::TokenizerType::BPE;
            model->eos_token_id = 106068;
        } else if (modelType == "baichuan") {
            model = (basellm*)(new LlamaModel());
            model->model_type = "baichuan";
            model->pre_prompt = "";
            model->user_role = "<human>:";
            model->bot_role = "\n<bot>:";
            model->history_sep = "\n";
            model->weight.tokenizer.type = Tokenizer::TokenizerType::BPE;
        } else if (modelType == "internlm") {
            model = new LlamaModel();
            model->model_type = "internlm";
        } else if (modelType == "internlm2") {
            model = new Internlm2Model();
            model->model_type = "internlm";
        } else if (modelType == "llama") {
            model = (basellm*)(new LlamaModel());
        } else if (modelType == "moe" || modelType == "qwen2_moe") {
            model = (basellm*)(new MoeModel());
        } else if (modelType == "deepseek_v2") {
            model = (basellm*)(new DeepSeekV2Model());
        } else if (modelType == "qwen2") {
            model = new LlamaModel();
            model->model_type = "qwen";
        } else if (modelType=="minicpm") {
            model = new MiniCpmModel();
        } else if (modelType == "qwen") {
            model = (basellm *) (new QWenModel());
            model->weight.tokenizer.type = Tokenizer::TokenizerType::QWEN;
        } else if (modelType == "glm") {
            model = (basellm*)(new GLMModel());
        } else if (modelType == "bert") {
            model = (basellm*)(new BertModel());
        } else {
            ErrorInFastLLM("Unkown model type: " + modelType);
        }
        return model;
    }

    std::unique_ptr<BertModel> CreateEmbeddingModelFromFile(const std::string &fileName) {
        BertModel *model = new BertModel();
        model->weight.tokenizer.type = Tokenizer::BERT;
        model->LoadFromFile(fileName);
        model->WarmUp();
        return std::unique_ptr<fastllm::BertModel> (model);
    }

    std::unique_ptr<fastllm::basellm> CreateLLMModelFromFile(const std::string &fileName) {
        std::string modelType = GetModelTypeFromFile(fileName);
        basellm *model = CreateModelWithType(modelType);
        model->LoadFromFile(fileName);
        model->WarmUp();
        return std::unique_ptr<fastllm::basellm> (model);
    }

    std::unique_ptr<basellm> CreateEmptyLLMModel(const std::string &modelType) {
        basellm *model = CreateModelWithType(modelType);
        return std::unique_ptr<fastllm::basellm> (model);
    }

    struct SafeTensorItem {
        std::string tensorName;
        std::string fileName;
        std::string dtype;
        std::vector <std::uint64_t> shape;
        std::vector <int> intShape;
        std::vector <std::uint64_t> data_offsets;

        uint64_t len, bytes;
        uint8_t *buffer = nullptr;

        SafeTensorItem() {} 

        SafeTensorItem(const std::string &tensorName, const std::string &fileName, const json11::Json &config, uint64_t baseOffset) {
            this->tensorName = tensorName;
            this->fileName = fileName;

            this->dtype = config["dtype"].string_value();
            for (auto &it : config["data_offsets"].array_items()) {
                this->data_offsets.push_back(baseOffset + it.ll_value());
            }
            for (auto &it : config["shape"].array_items()) {
                this->shape.push_back(it.ll_value());
                this->intShape.push_back(this->shape.back());
            }

            len = 1;
            for (auto &it : shape) {
                len *= it;
            }
            bytes = this->data_offsets[1] - this->data_offsets[0];
        }

        void CreateBuffer(DataType dstType) {
            DataType srcType;
            if (this->dtype == "BF16") {
                srcType = DataType::BFLOAT16;
            } else {
                ErrorInFastLLM("SafeTensorItem.CreateBuffer: unsupport src dtype " + this->dtype + "\n");
            }
            
            int unitSize = 4;
            if (dstType == DataType::FLOAT32) {
                unitSize = 4;
            } else if (dstType == DataType::FLOAT16 || dstType == DataType::BFLOAT16) {
                unitSize = 2;
            } else {
                ErrorInFastLLM("SafeTensorItem.CreateBuffer: unsupport dst dtype " + std::to_string(dstType) + "\n");
            }
            ClearBuffer();
            buffer = new uint8_t[len * unitSize];

            FILE *fi = fopen(this->fileName.c_str(), "r");
            int ret;
#if defined(_WIN32) or defined(_WIN64)
            _fseeki64(fi, this->data_offsets[0], 0);
#else
            fseek(fi, this->data_offsets[0], 0);
#endif
            if (dstType == srcType) {
                ret = fread(buffer, 1, this->bytes, fi);
            } else {
                uint8_t *ori = new uint8_t[this->bytes];
                ret = fread(ori, 1, this->bytes, fi);
                ConvertDataType(ori, srcType, buffer, dstType, len);
                delete[] ori;
            }
            fclose(fi);
        }

        void ClearBuffer() {
            delete[] buffer;
            buffer = nullptr;
        }
    };

    struct SafeTensors {
        std::set <std::string> fileNames;
        std::map <std::string, SafeTensorItem> itmeDict;

        SafeTensors (const std::set <std::string> &fileNames) {
            std::string error;
            this->fileNames = fileNames;
            for (auto &fileName : fileNames) {
                FILE *f = fopen(fileName.c_str(), "rb");
                uint64_t configBytes;
                int ret = fread(&configBytes, 8, 1, f);
                char *configString = new char[configBytes + 5];
                ret = fread(configString, 1, configBytes, f);
                configString[configBytes] = 0;
                auto config = json11::Json::parse(configString, error);
                for (auto it : config.object_items()) {
                    if (it.first != "__metadata__" ) {
                        itmeDict[it.first] = SafeTensorItem(it.first, fileName, it.second, 8 + configBytes);
                    }
                }

                delete[] configString;
            }
        }

        std::vector <std::string> GetSortedItemNames() {
            std::vector <std::pair <std::pair <std::string, uint64_t>, std::string> > v;
            for (auto &it : itmeDict) {
                v.push_back(std::make_pair(std::make_pair(it.second.fileName, it.second.data_offsets[0]), it.first));
            }
            std::sort(v.begin(), v.end());
            std::vector <std::string> ret;
            for (int i = 0; i < v.size(); i++) {
                ret.push_back(v[i].second);
            }
            return ret;
        }
    };

    // 从hf文件夹读取，仅支持safetensor格式的模型
    std::unique_ptr <basellm> CreateLLMModelFromHF(const std::string &modelPath, 
                                                    DataType linearDataType, int groupCnt) {
        std::string path = modelPath;
        if (path.back() != '/' || path.back() != '\\') {
            path += "/";
        }

        // 1. 检查是否有 model.safetensors.index.json,如果有就读取
        std::string stIndexFile = path + "model.safetensors.index.json";
        std::string error;
        auto stIndex = json11::Json::parse(ReadAllFile(stIndexFile), error)["weight_map"];
        std::set <std::string> stFiles;
        for (auto it : stIndex.object_items()) {
            stFiles.insert(path + it.second.string_value());
        }
        SafeTensors safeTensors(stFiles);

        // 2. 创建网络基本信息
        std::string configFile = path + "config.json";
        auto config = json11::Json::parse(ReadAllFile(configFile), error);
        basellm *model = CreateModelWithType(config["model_type"].string_value());
        for (auto &it : config.object_items()) {
            model->weight.AddDict(it.first, it.second.dump().c_str());
        }

        // 3. 读取分词
        std::string tokenizerConfigFile = path + "tokenizer_config.json";
        auto tokenizerConfig = json11::Json::parse(ReadAllFile(tokenizerConfigFile), error);
        std::string tokenizerClass = tokenizerConfig["tokenizer_class"].string_value();
        if (tokenizerClass == "PreTrainedTokenizerFast") {
            // PreTrainedTokenizerFast
            std::string tokenizerFile = path + "tokenizer.json";
            auto tokenizer = json11::Json::parse(ReadAllFile(tokenizerFile), error);
            auto tokenizerModel = tokenizer["model"];
            auto vocab = tokenizerModel["vocab"];
            for (auto &it : vocab.object_items()) {
                model->weight.AddTokenizerWord(it.first, it.second.int_value(), 1.0f);
            }
            std::map<std::string, int> spTokens;
            for (auto &it : tokenizer["added_tokens"].array_items()) {
                spTokens[it["content"].string_value()] = it["id"].int_value();
            }
            model->weight.tokenizer.SetSpecialTokens(spTokens);

            if (!tokenizer["decoder"].is_null() && !tokenizer["decoder"]["type"].is_null() && 
                tokenizer["decoder"]["type"].string_value() == "ByteLevel") {
                model->weight.tokenizer.byteAsChar = true;
            }
        } else {
            ErrorInFastLLM("Unsupport tokenizer_class: " + tokenizerClass);
        }

        // 4. 读取权重
        int cur = 0;
        for (auto &weightName : safeTensors.GetSortedItemNames()) {
            auto &tensor = safeTensors.itmeDict[weightName];
            tensor.CreateBuffer(DataType::FLOAT32);
            model->weight.AddWeight(weightName, tensor.intShape, linearDataType, WeightType::AUTO, DataType::FLOAT32, tensor.buffer, groupCnt);
            tensor.ClearBuffer();

            printf("Load (%d / %d) \r", (++cur), (int)safeTensors.itmeDict.size());
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);

        model->InitParams();
        model->WarmUp();
        return std::unique_ptr<fastllm::basellm> (model);
    }
}
