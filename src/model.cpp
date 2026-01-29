#include "utils.h"
#include "json11.hpp"

#include "model.h"
#include "fastllm.h"
#include <sstream>
#include <fstream>
#include <regex>
#include <iomanip>

#include "chatglm.h"
#include "moss.h"
#include "llama.h"
#include "moe.h"
#include "qwen3.h"
#include "qwen3_moe.h"
#include "qwen3_next.h"
#include "hunyuan.h"
#include "deepseekv2.h"
#include "qwen.h"
#include "glm.h"
#include "minicpm.h"
#include "minicpm3.h"
#include "internlm2.h"
#include "bert.h"
#include "xlmroberta.h"
#include "graphllm.h"
#include "phi3.h"
#include "cogvlm.h"
#include "minimax.h"
#include "ernie4_5.h"
#include "pangu_moe.h"
#include "glm4_moe.h"
#include "gpt_oss.h"

#include "gguf.h"

#ifdef USE_TFACC
#include "fastllm-tfacc.h"
#endif

#ifdef USE_NUMA
#include "fastllm-numa.h"
#endif

namespace fastllm {
    std::string ReadAllFile(const std::string &fileName) {
        std::ifstream t(fileName.c_str(), std::ios::in);
        if (!t.good()) {
            ErrorInFastLLM("Read error: can't find \"" + fileName + "\".");
        }

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
        } else if (srcDtype == DataType::FP8_E4M3 && dstDtype == DataType::FLOAT16) {
            ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
        } else if (srcDtype == DataType::BFLOAT16 && dstDtype == DataType::FLOAT32) {
            uint16_t *u16dst = (uint16_t*)dst;
            uint16_t *u16src = (uint16_t*)src;
            for (size_t i = 0; i < len; i++) {
                u16dst[i * 2] = 0;
                u16dst[i * 2 + 1] = u16src[i];
            }
        } else if (srcDtype == DataType::FLOAT16 && dstDtype == DataType::FLOAT32) {
            float *fdst = (float*)dst;
            uint16_t *u16src = (uint16_t*)src;
            for (size_t i = 0; i < len; i++) {
                fdst[i] = half_to_float(u16src[i]);
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
            if (this->weight.dicts["bos_token_id"]!="None") {
                this->bos_token_id = atoi(this->weight.dicts["bos_token_id"].c_str());
            }
        }
        if (this->weight.dicts.find("eos_token_id") != this->weight.dicts.end()) {
            if (this->weight.dicts["eos_token_id"]!="None") {
                if (this->weight.dicts["eos_token_id"][0] == '[') {
                    // eos_token_id is array format - parse and add to eos_token_ids set
                    std::string error;
                    json11::Json ids = json11::Json::parse(this->weight.dicts["eos_token_id"], error);
                    for (auto &it : ids.array_items()) {
                        this->eos_token_ids.insert(it.int_value());
                    }
                    // Don't set eos_token_id integer - leave it as -1
                } else {
                    // Single value - set eos_token_id
                    this->eos_token_id = atoi(this->weight.dicts["eos_token_id"].c_str());
                }
            }
        } else if (this->weight.dicts.find("im_start_id") != this->weight.dicts.end()) {
            this->bos_token_id = atoi(this->weight.dicts["im_start_id"].c_str());
            this->eos_token_id = atoi(this->weight.dicts["im_end_id"].c_str());
        }

        if (this->weight.dicts.find("num_hidden_layers") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["num_hidden_layers"].c_str());
        } else if (this->weight.dicts.find("num_layers") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["num_layers"].c_str());
        } else if (this->weight.dicts.find("n_layer") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["n_layer"].c_str());
        }
        if (this->weight.dicts.find("hidden_size") != this->weight.dicts.end()) {
            embed_dim = atoi(this->weight.dicts["hidden_size"].c_str());
        }
        if (this->weight.dicts.find("num_attention_heads") != this->weight.dicts.end()) {
            num_attention_heads = atoi(this->weight.dicts["num_attention_heads"].c_str());
        } else if (this->weight.dicts.find("n_head") != this->weight.dicts.end()) {
            num_attention_heads = atoi(this->weight.dicts["n_head"].c_str());
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
        if (this->weight.dicts.find("use_qk_norm") != this->weight.dicts.end()) {
            std::string value = this->weight.dicts["use_qk_norm"];
            transform(value.begin(), value.end(), value.begin(), ::tolower);
            std::istringstream iss(value);
            iss >> std::boolalpha >> this->use_qk_norm;
        }

#ifdef USE_SENTENCEPIECE
        if (this->weight.dicts.find("tokenizer_serialized") != this->weight.dicts.end()) {
            const std::string &hexString = this->weight.dicts["tokenizer_serialized"];
            if (hexString.length() % 2 != 0) {
                std::cerr << "warning: Invalid SentencePiece hex string.\n";
            } else {
                std::string decoded;
                for (unsigned int i = 0; i < hexString.length(); i += 2) {
                    decoded.push_back(std::stoi(hexString.substr(i, 2), nullptr, 16));
                }
                weight.tokenizer.spProcessor = std::make_unique<sentencepiece::SentencePieceProcessor>();
                weight.tokenizer.spProcessor->LoadFromSerializedProto(decoded);
            }
        }
#endif
        this->deviceMap = GetDeviceMap();
        this->moeDeviceMap = GetMoeDeviceMap();
    }

    void basellm::SaveLowBitModel(const std::string &fileName, int bit) {
        this->weight.SaveLowBitModel(fileName, bit);
    }

    void basellm::SaveModel(const std::string &fileName) {
        if (this->weight.tokenizer.chatTemplate.empty()) {
            if (this->weight.dicts.find("pre_prompt") == this->weight.dicts.end())
                this->weight.dicts["pre_prompt"] = pre_prompt;
            if (this->weight.dicts.find("user_role") == this->weight.dicts.end())
                this->weight.dicts["user_role"] = user_role;
            if (this->weight.dicts.find("bot_role") == this->weight.dicts.end())
                this->weight.dicts["bot_role"] = bot_role;
            if (this->weight.dicts.find("history_sep") == this->weight.dicts.end())
                this->weight.dicts["history_sep"] = history_sep;
        }
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
        } else if (modelType == "internlm" || modelType == "internlm3") {
            model = new LlamaModel();
            model->model_type = "internlm";
        } else if (modelType == "internlm2") {
            model = new Internlm2Model();
            model->model_type = "internlm";
        } else if (modelType == "llama") {
            model = (basellm*)(new LlamaModel());
        } else if (modelType == "moe" || modelType == "qwen2_moe") {
            model = (basellm*)(new MoeModel());
        } else if (modelType == "qwen3_moe") {
            model = (basellm*)(new Qwen3MOEModel());
        } else if (modelType == "qwen3_next") {
            model = (basellm*)(new Qwen3NextModel());
        } else if (modelType == "deepseek_v2" || modelType == "deepseek_v3" || modelType == "kimi_k2" || modelType == "deepseek_v32") {
            model = (basellm*)(new DeepSeekV2Model());
            model->model_type = modelType;
        } else if (modelType == "qwen2") {
            model = new LlamaModel();
            model->model_type = "qwen";
        } else if (modelType == "qwen3") {
            model = new Qwen3Model();
            model->model_type = "qwen3";
        } else if (modelType == "phi3") {
            model = new Phi3Model();
            model->model_type = "phi3";
        } else if (modelType=="minicpm") {
            model = new MiniCpmModel();
        } else if (modelType == "qwen") {
            model = (basellm *) (new QWenModel());
            model->weight.tokenizer.type = Tokenizer::TokenizerType::QWEN;
        } else if (modelType == "glm") {
            model = (basellm*)(new GLMModel());
        } else if (modelType == "bert") {
            model = (basellm*)(new BertModel());
        } else if (modelType == "xlm-roberta") {
            model = (basellm*)(new XlmRobertaModel());
        } else if (modelType == "cogvlm" || modelType == "CogVLMForCausalLM") {
            model = (basellm*)(new CogvlmModel());
        } else if (modelType == "minimax_m1" || modelType == "minimax_text_01") {
            model = (basellm*)(new MinimaxModel());
        } else if (modelType == "hunyuan" || modelType == "hunyuan_v1_dense" || modelType == "hunyuan_v1_moe") {
            model = (basellm*)(new HunyuanModel());
        } else if (modelType == "ernie4_5_moe" || modelType == "ernie4_5") {
            model = (basellm*)(new Ernie4_5Model());
        } else if (modelType == "PanguProMoE") {
            model = (basellm*)(new PanguMOEModel());
        } else if (modelType == "glm4_moe") {
            model = (basellm*)(new Glm4MOEModel());
        } else if (modelType == "gpt_oss") {
            model = (basellm*)(new GptOssModel());
        } else if (modelType == "fastllmJson") {
            model = new GraphLLMModel("fastllmJson");
        } else {
            model = new GraphLLMModel(modelType);
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

    bool IsGGUFFile(const std::string &fileName) {
        int ggufAlignment = GGUF_DEFAULT_ALIGNMENT;
        GGUFBuffer ggufBuffer = GGUFBuffer(fileName);
        int magic = ggufBuffer.Read<int> ();
        if (magic == 1179993927) { // GGUF
            return true;
        }
        return false;
    }

    std::unique_ptr<basellm> CreateEmptyLLMModel(const std::string &modelType) {
        basellm *model = CreateModelWithType(modelType);
        return std::unique_ptr<fastllm::basellm> (model);
    }

    template <typename T>
    void TransposeSimple(T *pDst, T *pSrc, int dstStride, int srcStride, int n, int m) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }
    }
    extern void Transpose(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m);

    struct SafeTensorItem {
        std::string tensorName;
        std::string fileName;
        std::string dtype;
        std::vector <std::uint64_t> shape;
        std::vector <int> intShape;
        std::vector <std::uint64_t> data_offsets;

        uint64_t len, bytes;
        uint8_t *buffer = nullptr;
        float *minsBuffer = nullptr, *scalesBuffer = nullptr;
        int blockK, blockM;

        SafeTensorItem() {} 

        ~SafeTensorItem() {
            ClearBuffer();
        }

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

        struct FP8E4M3ToFP32Manager fp8e4m3tofp32;

        void CreateBufferWithScale(DataType dstType, SafeTensorItem &scale) {
            AssertInFastLLM(this->shape.size() == 2 && scale.shape.size() == 2, "CreateBufferWithScale error: shape.size() should be 2.");
            DataType srcType;
            if (this->dtype == "F8_E4M3") {
                srcType = DataType::FP8_E4M3;
            } else {
                ErrorInFastLLM("CreateBufferWithScale error: dtype should be FP8_E4M3");
            }
            int n = this->shape[0], m = this->shape[1];
            int ns = scale.shape[0], ms = scale.shape[1];
            int blockN = n / ns, blockM = m / ms;

            while ((blockN & -blockN) != blockN && blockN < n) {
                blockN++;
            }
            while ((blockM & -blockM) != blockM && blockM < m) {
                blockM++;
            }
            ClearBuffer();

            if (dstType == DataType::FP8_E4M3) {
                this->blockK = blockN;
                this->blockM = blockM;
                buffer = new uint8_t[n * m];
                FILE *fi = fopen(this->fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
                _fseeki64(fi, this->data_offsets[0], 0);
#else
                fseek(fi, this->data_offsets[0], 0);
#endif
                int ret = fread(buffer, 1, this->bytes, fi);
                fclose(fi);

                scalesBuffer = new float[ns * ms];
                memcpy(scalesBuffer, scale.buffer, ns * ms * sizeof(float));
            } else {
                buffer = new uint8_t[n * m * sizeof(float)];
                float *floatBuffer = (float*)buffer;

                FILE *fi = fopen(this->fileName.c_str(), "rb");
                int ret;
    #if defined(_WIN32) || defined(_WIN64)
                _fseeki64(fi, this->data_offsets[0], 0);
    #else
                fseek(fi, this->data_offsets[0], 0);
    #endif
                uint8_t *ori = new uint8_t[this->bytes];
                ret = fread(ori, 1, this->bytes, fi);
                for (int bi = 0; bi < ns; bi++) {
                    for (int bj = 0; bj < ms; bj++) {
                        float curScale = ((float*)scale.buffer)[bi * ms + bj];
                        for (int i = bi * blockN; i < (bi + 1) * blockN && i < n; i++) {
                            for (int j = bj * blockM; j < (bj + 1) * blockM && j < m; j++) {
                                floatBuffer[i * m + j] = curScale * fp8e4m3tofp32.dict[ori[i * m + j]];
                            }
                        }
                    }
                }

                delete[] ori;
                fclose(fi);
            }
        }

        void CreateBufferWithAWQ(DataType dstType, SafeTensorItem &scale, SafeTensorItem &qzero) {
            const int groupCnt = this->shape[0] / scale.shape[0];
            AssertInFastLLM(this->shape.size() == 2 && scale.shape.size() == 2 && qzero.shape.size() == 2,
                            "CreateBufferWithAWQ error: shape.size() should be 2.");
            AssertInFastLLM(groupCnt * scale.shape[0] == this->shape[0] && groupCnt * qzero.shape[0] == this->shape[0] &&
                            8 * this->shape[1] == scale.shape[1] && this->shape[1] == qzero.shape[1],
                            "CreateBufferWithAWQ error: shape error.");
            AssertInFastLLM(this->dtype == "I32" && qzero.dtype == "I32",
                            "CreateBufferWithAWQ error: dtype shoud be I32.");
            int n = this->shape[0], m = this->shape[1];

            ClearBuffer();
            FILE *fweight = fopen(this->fileName.c_str(), "rb");
            FILE *fqzero  = fopen(qzero.fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
            _fseeki64(fweight, this->data_offsets[0], 0);
            _fseeki64(fqzero,  qzero.data_offsets[0], 0);
#else
            fseek(fweight, this->data_offsets[0], 0);
            fseek(fqzero,  qzero.data_offsets[0], 0);
#endif
            uint8_t *ori_weight = new uint8_t[this->bytes];
            uint8_t *ori_qzero  = new uint8_t[qzero.bytes];
            int ret;
            ret = fread(ori_weight, 1, this->bytes, fweight);
            ret = fread(ori_qzero , 1, qzero.bytes, fqzero);
            unsigned int* weight_int32 = (unsigned int*)ori_weight;
            unsigned int* qzero_int32  = (unsigned int*)ori_qzero;
            float* scale_f32 = (float*)scale.buffer;
            static const int awq_shift[8] = {0,16,4,20,8,24,12,28}; // awq order = [0,2,4,8,1,3,5,7]

            if (dstType == DataType::FLOAT32) {
                buffer = new uint8_t[this->bytes * 8];
                float *floatBuffer = (float*)buffer;
                for (int x = 0; x < n; x++) {
                    for (int y = 0; y < m * 8; y++) {
                        int gx = x / groupCnt;
                        int gy = y >> 3;
                        int w = (weight_int32[x * m + gy] >> awq_shift[y & 7]) & 15;
                        int z = (qzero_int32[gx * m + gy] >> awq_shift[y & 7]) & 15;
                        float s = scale_f32[gx * m * 8 + y];
                        floatBuffer[y * n + x] = (w - z) * s;
                    }
                }
            } else if (dstType == DataType::INT4_GROUP) {
                buffer = new uint8_t[this->bytes];
                memset(buffer, 0, this->bytes);
                int group = (n - 1) / groupCnt + 1;
                scalesBuffer = new float[m * 8 * group];
                minsBuffer = new float[m * 8 * group];
                for (int x = 0; x < n; x += groupCnt) {
                    for (int y = 0; y < m * 8; y++) {
                        int gx = x / groupCnt;
                        int gy = y >> 3;
                        int z = (qzero_int32[gx * m + gy] >> awq_shift[y & 7]) & 15;
                        float s = scale_f32[gx * m * 8 + y];
                        scalesBuffer[y * group + x / groupCnt] = s;
                        minsBuffer[y * group + x / groupCnt] = -s * z;
                    }
                }
                for (int x = 0; x < n; x++) {
                    for (int y = 0; y < m * 8; y++) {
                        int gx = x / groupCnt;
                        int gy = y >> 3;
                        int w = (weight_int32[x * m + gy] >> awq_shift[y & 7]) & 15;
                        buffer[y * n / 2 + x / 2] += (w << ((1 - (x & 1)) * 4));
                    }
                }
            } else {
                ErrorInFastLLM("CreateBufferWithAWQ Error: dst type error.");
            }
            delete[] ori_weight;
            delete[] ori_qzero;
            fclose(fweight);
            fclose(fqzero);
        }

        void CreateBuffer(DataType dstType) {
            //printf("read %s from %s [%llu %llu] (%f M)\n", this->tensorName.c_str(), this->fileName.c_str(), this->data_offsets[0], this->data_offsets[0] + this->bytes, (float)this->bytes / 1e6);
            FILE *fi = fopen(this->fileName.c_str(), "rb");
            int ret;
#if defined(_WIN32) || defined(_WIN64)
            _fseeki64(fi, this->data_offsets[0], 0);
#else
            fseek(fi, this->data_offsets[0], 0);
#endif
            DataType srcType;
            if (this->dtype == "fastllm") {
                ClearBuffer();
                buffer = new uint8_t[this->bytes];
                ret = fread(buffer, 1, this->bytes, fi);
                fclose(fi);
                return;
            } else if (this->dtype == "F8_E4M3") {
                srcType = DataType::FP8_E4M3;
            } else if (this->dtype == "BF16") {
                srcType = DataType::BFLOAT16;
            } else if (this->dtype == "F16") {
                srcType = DataType::FLOAT16;
            } else if (this->dtype == "F32") {
                srcType = DataType::FLOAT32;
                if (dstType != DataType::FLOAT32) {
                    ErrorInFastLLM("SafeTensorItem.CreateBuffer: unsupport src dtype " + this->dtype + "\n");
                }
            } else if (this->dtype == "I64") {
                printf("skip I64 tensor %s\n", this->tensorName.c_str());
                return;
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
            buffer = new uint8_t[(size_t)len * unitSize];
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

        void Transpose(DataType type) {
            int n = intShape[0], m = intShape[1];
            if (type == DataType::FLOAT32) {
                float *temp = new float[len];
                memcpy(temp, this->buffer, len * sizeof(float));
                fastllm::Transpose((float*)this->buffer, temp, n, m, n, m);
                delete[] temp;
            } else if (type == DataType::FLOAT16 || type == DataType::BFLOAT16) {
                uint16_t *temp = new uint16_t[len];
                memcpy(temp, this->buffer, len * sizeof(uint16_t));
                TransposeSimple((uint16_t*)this->buffer, temp, n, m, n, m);
                delete[] temp;
            } else {
                ErrorInFastLLM("SafeTensorItem.Transpose: unsupport dtype " + std::to_string(type) + "\n");
            }
        }

        void ClearBuffer() {
            delete[] buffer;
            buffer = nullptr;
            delete[] minsBuffer;
            minsBuffer = nullptr;
            delete[] scalesBuffer;
            scalesBuffer = nullptr;
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
                if (it.second.intShape.size() > 0 && it.second.dtype != "BOOL") {
                    v.push_back(std::make_pair(std::make_pair(it.second.fileName, it.second.data_offsets[0]), it.first));
                }
            }
            std::sort(v.begin(), v.end());
            std::vector <std::string> ret;
            for (int i = 0; i < v.size(); i++) {
                ret.push_back(v[i].second);
            }
            return ret;
        }
    };

    std::string Base64Decode(const std::string &encoded) {
        static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";
        int in_len = encoded.size();
        int i = 0, j = 0, in_ = 0;
        char char_array_4[4], char_array_3[3];
        std::string ret = "";
        
        while (in_len-- && ( encoded[in_] != '=')) {
            char_array_4[i++] = encoded[in_]; in_++;
            if (i == 4) {
                for (i = 0; i < 4; i++)
                    char_array_4[i] = base64_chars.find(char_array_4[i]);
                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
                for (i = 0; (i < 3); i++)
                    ret.push_back(char_array_3[i]);
                i = 0;
            }
        }

        if (i) {
            for (j = i; j < 4; j++)
                char_array_4[j] = 0;

            for (j = 0; j < 4; j++)
                char_array_4[j] = base64_chars.find(char_array_4[j]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
        }

        return ret;
    }

    void SplitString(const std::string &str, const std::set <char> &chars, std::vector <std::string> &ret) {
        ret.clear();
        std::string now = "";
        for (int i = 0; i < str.size(); i++) {
            if (chars.find(str[i]) == chars.end()) {
                now += str[i];
            } else {
                if (now != "") {
                    ret.push_back(now);
                    now = "";
                }
            }
        }
        if (now != "") {
            ret.push_back(now);
        }
    }

    void DealLLMTokenizerFromHFToModel(const std::string &path, basellm *model) {
        std::string error;
        std::string tokenizerConfigFile = path + "tokenizer_config.json";
        if (!fastllm::FileExists(tokenizerConfigFile)) {
            return;
        }
        auto tokenizerConfig = json11::Json::parse(ReadAllFile(tokenizerConfigFile), error);
        model->weight.tokenizer.SetTokenizerConfig(tokenizerConfig);
        std::string tokenizerClass = tokenizerConfig["tokenizer_class"].string_value();
        if (tokenizerClass == "PreTrainedTokenizerFast" || tokenizerClass == "Qwen2Tokenizer") {
        } else if (tokenizerClass == "ChatGLM4Tokenizer") {
            // 历史遗留问题
            model->bot_role = " ";
        }
    }

    void LoadLLMTokenizerFromHFToModel(const std::string &path, basellm *model) {
        std::string error;
        std::string tokenizerConfigFile = path + "tokenizer_config.json";
        auto tokenizerConfig = json11::Json::parse(ReadAllFile(tokenizerConfigFile), error);
        model->weight.tokenizer.SetTokenizerConfig(tokenizerConfig);
        if (!model->weight.tokenizer.chatTemplate.empty() && model->weight.dicts.find("chat_template") == model->weight.dicts.end())
            model->weight.AddDict("chat_template", model->weight.tokenizer.chatTemplate);
        std::string tokenizerClass = tokenizerConfig["tokenizer_class"].string_value();
        if (tokenizerClass == "PreTrainedTokenizerFast" || tokenizerClass == "LlamaTokenizerFast"
            || tokenizerClass == "Qwen2Tokenizer"
            || tokenizerClass == "BloomTokenizer"
            || tokenizerClass == "LlamaTokenizer" || tokenizerClass == "CodeLlamaTokenizer"
            || tokenizerClass == "MiniCPMTokenizer") {
            // PreTrainedTokenizerFast
            std::string tokenizerFile = path + "tokenizer.json";
            if (!fastllm::FileExists(tokenizerFile)) {
#ifdef USE_SENTENCEPIECE
                tokenizerFile = path + "tokenizer.model";
                if (fastllm::FileExists(tokenizerFile)) {
                    std::string&& tokenizerProto = ReadAllFile(tokenizerFile);
                    model->weight.tokenizer.spProcessor = std::make_unique<sentencepiece::SentencePieceProcessor>();
                    model->weight.tokenizer.spProcessor->LoadFromSerializedProto(tokenizerProto);
                    return;
                }
#endif
                ErrorInFastLLM("Model with a supported tokenizer_class: " + tokenizerClass + "，but has no \"tokenizer.json\"!");
            }
            auto tokenizer = json11::Json::parse(ReadAllFile(tokenizerFile), error);
            for (auto &it : tokenizer["model"]["vocab"].object_items()) {
                model->weight.AddTokenizerWord(it.first, it.second.int_value(), 1.0f);
            }
            std::map<std::string, int> spTokens;
            for (auto &it : tokenizer["added_tokens"].array_items()) {
                spTokens[it["content"].string_value()] = it["id"].int_value();
            }
            if (!spTokens.empty())
                model->weight.AddDict("tokenizer_has_special_tokens", "1");

            if (!tokenizer["decoder"].is_null() && !tokenizer["decoder"]["type"].is_null() && 
                tokenizer["decoder"]["type"].string_value() == "ByteLevel") {
                model->weight.tokenizer.byteAsChar = true;
                model->weight.AddDict("tokenizer_byte_as_char", "True");
            }
            model->weight.tokenizer.SetSpecialTokens(spTokens);
#ifdef USE_SENTENCEPIECE
        } else if (tokenizerClass == "PreTrainedTokenizer"
            || tokenizerClass == "InternLM2Tokenizer" || tokenizerClass == "InternLM3Tokenizer"
            || tokenizerClass == "Ernie4_5_Tokenizer") {
            std::string tokenizerFile = path + "tokenizer.model";
            std::string&& tokenizerProto = ReadAllFile(tokenizerFile);
            model->weight.tokenizer.spProcessor = std::make_unique<sentencepiece::SentencePieceProcessor>();
            model->weight.tokenizer.spProcessor->LoadFromSerializedProto(tokenizerProto);
            if (tokenizerClass == "InternLM2Tokenizer")
                model->eos_token_ids.insert(92542);
#endif
        } else if (tokenizerClass == "ChatGLM4Tokenizer") {
            // GLM4御用的分词
            std::vector <std::string> lines, line;
            SplitString(ReadAllFile(path + "tokenizer.model"), {'\r', '\n'}, lines);
            for (int i = 0; i < lines.size(); i++) {
                SplitString(lines[i], {' '}, line);
                model->weight.AddTokenizerWord(Base64Decode(line[0]), atoi(line[1].c_str()), 1.0f);
            }
            std::map<std::string, int> spTokens;
            for (auto &it : tokenizerConfig["added_tokens_decoder"].object_items()) {
                spTokens[it.second["content"].string_value()] = atoi(it.first.c_str());
            }
            model->weight.tokenizer.SetSpecialTokens(spTokens);
            model->weight.AddDict("tokenizer_has_special_tokens", "1");
            model->weight.AddDict("tokenizer_class", tokenizerClass);
            ((ChatGLMModel*)model)->tokenizerClass = tokenizerClass;

            // ChatGLM采用拼接token的方法，需要强行指定分割词的TokenID
            model->pre_prompt = "[gMASK]<sop>";
            model->user_role = ("<FLM_FIX_TOKEN_" + std::to_string(model->weight.tokenizer.GetTokenId("<|user|>"))  + ">\n");
            model->bot_role = ("<FLM_FIX_TOKEN_" + std::to_string(model->weight.tokenizer.GetTokenId("<|assistant|>")) + ">\n");
            model->history_sep = "";
            model->weight.tokenizer.type = Tokenizer::TokenizerType::QWEN;
            model->weight.tokenizer.chatTemplate = "";
        } else if (tokenizerClass == "QWenTokenizer" || tokenizerClass == "HYTokenizer" || tokenizerClass == "TikTokenTokenizer") {
            // tiktoken分词
            std::map<std::string,std::string> nameMap = {{"QWenTokenizer","qwen.tiktoken"},{"HYTokenizer","hy.tiktoken"},{"TikTokenTokenizer","tiktoken.model"}};
            std::vector <std::string> lines, line;
            SplitString(ReadAllFile(path + nameMap[tokenizerClass]), {'\n'}, lines);
            for (int i = 0; i < lines.size(); i++) {
                SplitString(lines[i], {' '}, line);
                model->weight.AddTokenizerWord(Base64Decode(line[0]), atoi(line[1].c_str()), 1.0f);
            }
            model->weight.tokenizer.type = Tokenizer::TokenizerType::QWEN;
            if (tokenizerClass == "QWenTokenizer") {
                // Qwen用的分词
                model->weight.tokenizer.chatTemplate = "";
                model->weight.dicts["im_start_id"] = std::to_string(lines.size() + 1);
                model->weight.dicts["im_end_id"] = std::to_string(lines.size() + 2);
            }
        } else {
            ErrorInFastLLM("Unsupport tokenizer_class: " + tokenizerClass);
        }
    }

    // 从hf文件夹读取分词
    std::unique_ptr<basellm> CreateLLMTokenizerFromHF(const std::string &modelPath) {
        std::string error;
        std::string path = modelPath;
        if (path.back() != '/' || path.back() != '\\') {
            path += "/";
        }
        std::string configFile = path + "config.json";
        auto config = json11::Json::parse(ReadAllFile(configFile), error);
        basellm *model = CreateModelWithType(config["model_type"].string_value());
        LoadLLMTokenizerFromHFToModel(path, model);
        return std::unique_ptr<fastllm::basellm> (model);
    }

    // 将config中的内容递归地加入model->dict中
    void AddDictRecursion(basellm *model, const std::string &pre, const json11::Json &config) {
        for (auto &it : config.object_items()) {
            if (it.second.is_object()) {
                AddDictRecursion(model, pre + it.first + ".", it.second);
            } else {
                model->weight.AddDict(pre + it.first, it.second.is_string() ? it.second.string_value() : it.second.dump());
            }
        }
    }

    static std::map <DataType, int> DefaultGroupCnts = {
        {DataType::INT4_GROUP, 128},
        {DataType::INT2_GROUP, 128}, 
        {DataType::BASE3_GROUP, 128}
    };

    extern std::map <DataType, std::vector <std::string> > dataTypeNames;

    void ParseDataType(std::string weightName, std::vector <std::pair <std::string, std::string> > &dtypeRules, 
                        DataType &dataType, int &groupCnt, int &ggmlType) {
        std::string matchedType = "";
        for (int i = 0; i < dtypeRules.size(); i++) {
            std::regex pattern(dtypeRules[i].first);
            if (std::regex_search(weightName, pattern)) {
                matchedType = dtypeRules[i].second;
            }
        }
        transform(matchedType.begin(), matchedType.end(), matchedType.begin(), ::tolower);

        ggmlType = -1;
        if (matchedType.size() >= 5 && matchedType.substr(0, 5) == "ggml_") {            
            static std::set <ggml_type> types = {
                GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0
            };
            dataType = DATA_GGUF_FORMAT;
            std::string type = matchedType.substr(5);
            for (ggml_type t : types) {
                std::string x = ggml_type_name(t);
                transform(x.begin(), x.end(), x.begin(), ::tolower);
                if (x == type) {
                    ggmlType = t;
                    break;
                }
            }

            if (ggmlType == -1) {
                ErrorInFastLLM("Failed: Unsupport type " + matchedType);
            }
        } else if (matchedType != "") {
            for (auto &it : dataTypeNames) {
                for (auto &dataTypeName : it.second) {
                    if (DefaultGroupCnts.find(it.first) != DefaultGroupCnts.end()) {
                        if (StringStartWith(matchedType, dataTypeName)) {
                            dataType = it.first;
                            if (matchedType != dataTypeName) {
                                groupCnt = std::atoi(matchedType.substr(dataTypeName.size()).c_str());
                            } else {
                                groupCnt = DefaultGroupCnts[it.first];
                            }
                        }
                    } else {
                        if (matchedType == dataTypeName) {
                            dataType = it.first;
                        }
                    }
                }
            }
        }        
    }

    std::vector<std::string> GenerateGGUFFileList(const std::string& filename) {
        std::vector<std::string> fileList;
        
        // 正则表达式匹配文件名格式：基础名-当前序号-of-总数.扩展名
        std::regex pattern(R"(^(.+)-(\d+)-of-(\d+)\.(.+)$)");
        std::smatch matches;
        
        if (!std::regex_match(filename, matches, pattern)) {
            // 如果不匹配分片格式，返回原文件名
            fileList.push_back(filename);
            return fileList;
        }
        
        // 提取各部分
        std::string baseName = matches[1].str();
        int currentNum = std::stoi(matches[2].str());
        int totalNum = std::stoi(matches[3].str());
        std::string extension = matches[4].str();
        
        // 获取序号的位数（用于补零）
        int digits = matches[2].str().length();
        
        // 生成所有文件名
        for (int i = 1; i <= totalNum; ++i) {
            std::ostringstream oss;
            oss << baseName << "-" 
                << std::setfill('0') << std::setw(digits) << i 
                << "-of-" 
                << std::setfill('0') << std::setw(digits) << totalNum 
                << "." << extension;
            fileList.push_back(oss.str());
        }
        
        return fileList;
    }

    std::string ConvertGGUFTypeToFastllmType(const std::string &type) {
        static std::map <std::string, std::string> ggufTypeToFastllmTypeDict = {
            {"qwen2", "qwen2"}, // llama
            {"qwen3moe", "qwen3_moe"}, {"qwen3_moe", "qwen3_moe"}, // qwen3_moe
            {"glm4_moe", "glm4_moe"}, // glm4_moe
            {"deepseek2", "deepseek_v2"}, {"deepseek_v2", "deepseek_v2"},  {"deepseek_v3", "deepseek_v2"} // deepseek_v2
        };
        if (ggufTypeToFastllmTypeDict.find(type) != ggufTypeToFastllmTypeDict.end()) {
            return ggufTypeToFastllmTypeDict[type];
        } else {
            printf("Warning: Can't convert type \"%s\", try use original type.\n", type.c_str());
            return type;
        }
    }

    extern void RegisterNumas(fastllm::Data *data);

    std::unique_ptr<basellm> CreateLLMModelFromGGUFFile(const std::string &fileName, const std::string &originalPath) {
        std::vector <ReadGGUFTask> readGGUFTasks;
        std::map <std::string, ReadGGUFTask*> readGGUFTaskDict;
        std::vector <std::string> ggufFileNames = GenerateGGUFFileList(fileName);
        AssertInFastLLM(ggufFileNames.size() > 0, "0 gguf file found!");

        printf("Load model from files:\n");
        for (auto &s : ggufFileNames) {
            printf("%s\n", s.c_str());
        }
        json11::Json config;
        ReadGGUFMetaData(ggufFileNames[0], config);
        json11::Json params = config["params"];
        std::string arch = params["general.architecture"].string_value();        

        basellm *model = nullptr; 
        std::string path = originalPath;
        std::vector <std::string> tensors;
        if (path != "") {
            // Load from original config
            if (path.back() != '/' || path.back() != '\\') {
                path += "/";
            }
            std::string error;
            std::string configFile = path + "config.json";
            auto config = json11::Json::parse(ReadAllFile(configFile), error);

            // 1. 创建网络基本信息
            std::string modelType;
            if (!config["model_type"].is_null()) {
                modelType = config["model_type"].string_value();
            } else {
                modelType = config["architectures"].array_items()[0].string_value();
            }
            arch = modelType;
            model = CreateModelWithType(modelType);
            AddDictRecursion(model, "", config);
            // 设置eos_token_id
            if (config["eos_token_id"].is_null()) {
                auto tokenizer = json11::Json::parse(ReadAllFile(path + "tokenizer.json"), error);
                if (error == "") {
                    std::string tokenizerConfigFile = path + "tokenizer_config.json";
                    auto tokenizerConfig = json11::Json::parse(ReadAllFile(tokenizerConfigFile), error);
                    std::string eos_token = tokenizerConfig["eos_token"].string_value();
                    printf("eos_token = %s\n", eos_token.c_str());
                    for (auto added_token : tokenizer["added_tokens"].array_items()) {
                        if (added_token["content"] == eos_token) {
                            model->eos_token_ids.insert(added_token["id"].int_value());
                        }
                    }
                }
            } else if (config["eos_token_id"].is_array()) {
                for (auto &it : config["eos_token_id"].array_items()) {
                    model->eos_token_ids.insert(it.int_value()); 
                }
            } else {
                model->eos_token_id = config["eos_token_id"].int_value();
            }
            std::string generatetionConfigFile = path + "generation_config.json";
            if (FileExists(generatetionConfigFile)) {
                auto generation_config = json11::Json::parse(ReadAllFile(generatetionConfigFile), error);
                for (auto &it : generation_config.object_items()) {
                    if ("eos_token_id" == it.first && it.second.type() == json11::Json::ARRAY)
                        continue;
                    model->weight.AddDict(it.first, it.second.is_string() ? it.second.string_value() : it.second.dump());
                }
                // 更新eos_token_id
                if (generation_config["eos_token_id"].is_array()) {
                    for (auto &it : generation_config["eos_token_id"].array_items()) {
                        model->eos_token_ids.insert(it.int_value());
                    }
                }
            }

            // 2. 读取分词
            if (false) {
                LoadLLMTokenizerFromHFToModel(path, model);
            } else {
                DealLLMTokenizerFromHFToModel(path, model);
            }
        } else {
            // Load params from gguf
            printf("general.architecture = %s\n", arch.c_str());
            printf("general.name = %s\n", params["general.name"].string_value().c_str());

            model = CreateModelWithType(ConvertGGUFTypeToFastllmType(arch));
            if (!params[arch + ".block_count"].is_null()) {
                model->block_cnt = params[arch + ".block_count"].int_value();
                printf("Load block_cnt = %d\n", model->block_cnt);
            }

            if (!params[arch + ".attention.head_count"].is_null()) {
                model->num_attention_heads = params[arch + ".attention.head_count"].int_value();
                printf("Load num_attention_heads = %d\n", model->num_attention_heads);
            }

            if (!params[arch + ".attention.head_count_kv"].is_null()) {
                model->num_key_value_heads = params[arch + ".attention.head_count_kv"].int_value();
                model->weight.dicts["num_key_value_heads"] = std::to_string(model->num_key_value_heads);
                printf("Load num_key_value_heads = %d\n", model->num_key_value_heads);
            }

            if (!params[arch + ".embedding_length"].is_null()) {
                model->embed_dim = params[arch + ".embedding_length"].int_value();
                printf("Load embed_dim = %d\n", model->embed_dim);
            }

            if (!params[arch + ".context_length"].is_null()) {
                model->max_positions = params[arch + ".context_length"].int_value();
                printf("Load max_positions = %d\n", model->max_positions);
            }

            if (!params[arch + ".attention.layer_norm_rms_epsilon"].is_null()) {
                model->rms_norm_eps = params[arch + ".attention.layer_norm_rms_epsilon"].number_value();
                printf("Load rms_norm_eps = %f\n", model->rms_norm_eps);
            }

            if (!params["tokenizer.ggml.eos_token_id"].is_null()) {
                model->eos_token_id = params["tokenizer.ggml.eos_token_id"].number_value();
                printf("Load eos_token_id = %d\n", model->eos_token_id);
            }

            if (!params["tokenizer.chat_template"].is_null()) {
                model->weight.tokenizer.chatTemplate = params["tokenizer.chat_template"].string_value();
                printf("Load chatTemplate = %s\n", model->weight.tokenizer.chatTemplate.c_str());
            }

            int idx = 0;
            for (auto &it : params["tokenizer.ggml.tokens"].array_items()) {
                if (idx < 10) {
                    // printf("%s: %d\n", it.string_value().c_str(), idx);
                }
                model->weight.AddTokenizerWord(it.string_value(), idx, 1.0f);
                idx++;
            }

            model->weight.tokenizer.byteAsChar = true;
            model->weight.AddDict("tokenizer_byte_as_char", "True");

            // printf("config = %s\n", config.dump().c_str());
        }

        arch = ConvertGGUFTypeToFastllmType(arch);

        // 3.0 更新模型信息
        model->InitParams();

        int cur = 0;
        long long totalBytes = 0;
        std::set <std::string> allWeightNames; // 所有创建了的weight name
        std::set <std::string> allFinishNames; // 转换好的weight name

        for (auto &s : ggufFileNames) {
            AppendGGUFTasks(arch, s, readGGUFTasks);
        }
        for (int i = 0; i < readGGUFTasks.size(); i++) {
            std::string &weightName = readGGUFTasks[i].name;
/*
if (false) {
    std::string prefix = "model.layers.";
    if (StartWith(weightName, prefix)) {
        int id = 0;
        for (int i = prefix.size(); weightName[i] >= '0' && weightName[i] <= '9'; i++) {
            id = id * 10 + weightName[i] - '0';
        }
        if (id > 3) {
            continue;
        }
    }
}
*/
            tensors.push_back(weightName);
            allWeightNames.insert(weightName);
            model->weight.AddEmptyWeight(weightName, {1}, DataType::FLOAT32);
            readGGUFTasks[i].weight = &model->weight.weight[weightName];
            readGGUFTaskDict[readGGUFTasks[i].name] = &readGGUFTasks[i];
        }

        std::vector <std::thread*> threads;
        int threadNum = std::min(16, std::max(4, (int)GetAlivePool()->threads.size()));
        std::mutex locker;
        int cnt = 0;

        totalBytes = tensors.size();
        std::vector <std::pair <int, int> > parts;
        int start = 0;
        for (int i = 0; i < threadNum; i++) {
            int cur = start;
            long long now = 0;
            while (true) {
                if (now * threadNum >= totalBytes || start >= tensors.size()) {
                    break;
                }

                // now += safeTensors.itmeDict[tensors[start]].bytes;
                now += 1;

                start++;
            }
            parts.push_back(std::make_pair(cur, start));
        }
        parts.back().second = tensors.size();
        while (parts.size() < threadNum) {
            parts.push_back(std::make_pair(-1, -1));
        }

        // Load 
        for (int i = 0; i < threadNum; i++) {
            threads.push_back(
                new std::thread([&](int st, int end) {
                    for (int i = st; i < end; i++) {
                        auto &weightName = tensors[i];
                        if (readGGUFTaskDict.find(weightName) != readGGUFTaskDict.end()) {
                            WeightImportGGUFTensor(readGGUFTaskDict[weightName]->weight, 
                                        &readGGUFTaskDict[weightName]->tensor, readGGUFTaskDict[weightName]->fileName, 
                                        readGGUFTaskDict[weightName]->offset, readGGUFTaskDict[weightName]->replaceType);
                        } 
                        {
                            // try merge                                
                            locker.lock();
                            allFinishNames.insert(weightName);
                            // 检查是否需要合并权重
                            bool needMerge = false;
                            for (auto &rule : model->weightMergeRules) {
                                if (rule.allInputs.find(weightName) == rule.allInputs.end()) {
                                    continue;
                                }
                                needMerge = true;
                                bool canMerge = true;
                                for (auto &name : rule.allInputs) {
                                    if (allWeightNames.find(name) != allWeightNames.end() && 
                                        allFinishNames.find(name) == allFinishNames.end()) {
                                        canMerge = false;
                                    }
                                }
                                if (!canMerge) {
                                    continue;
                                }
                                for (auto &it : rule.rules) {
                                    for (auto input : it.inputs) {
                                        if (model->weight[input].dims.size() == 2) {
                                            if (model->weight[input].groupCnt != -1 && 
                                                model->weight[input].dims[1] % model->weight[input].groupCnt != 0) {
                                                canMerge = false;
                                                break;
                                            }
                                            if (model->weight[input].blockK != -1 && 
                                                model->weight[input].dims[0] % model->weight[input].blockK != 0) {
                                                canMerge = false;
                                                break;
                                            }
                                            if (model->weight[input].blockM != -1 && 
                                                model->weight[input].dims[1] % model->weight[input].blockM != 0) {
                                                canMerge = false;
                                                break;
                                            }
                                            if (model->weight[input].dataType != model->weight[it.inputs[0]].dataType ||
                                                model->weight[input].ggmlType != model->weight[it.inputs[0]].ggmlType ||
                                                model->weight[input].dims[1] != model->weight[it.inputs[0]].dims[1]) {
                                                canMerge = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (!canMerge) {
                                    continue;
                                }

                                locker.unlock();
                                for (auto &it : rule.rules) {
                                    if (allWeightNames.find(it.inputs[0]) == allWeightNames.end()) {
                                        continue;
                                    }
                                    int dim0Len = 0;
                                    for (auto input : it.inputs) {
                                        dim0Len += model->weight[input].dims[0];
                                    }
                                    if (model->weight[it.inputs[0]].dims.size() == 1) {
                                        std::string input0 = it.inputs[0];
                                        std::string mergeName = it.output;
                                        if (model->weight[input0].dataType == DATA_GGUF_FORMAT) {
                                            model->weight[mergeName] = Data(model->weight[input0].dataType);
                                            model->weight[mergeName].ggmlType = ((ggml_tensor*) model->weight[input0].ggmlTensor)->type;
                                            model->weight[mergeName].Resize({dim0Len});
                                        } else {
                                            model->weight[mergeName] = Data(model->weight[input0].dataType, {dim0Len});
                                        }

                                        Data &mergeData = model->weight[mergeName];
                                        mergeData.name = mergeName;
                                        mergeData.Allocate();
                                        uint64_t offset = 0;
                                        for (auto input : it.inputs) {
                                            memcpy(mergeData.cpuData + offset, model->weight[input].cpuData, model->weight[input].GetBytes());
                                            offset += model->weight[input].GetBytes();
                                        }
                                    } else {
                                        std::string input0 = it.inputs[0];
                                        std::string mergeName = it.output;
                                        if (model->weight[input0].dataType == DATA_GGUF_FORMAT) {
                                            model->weight[mergeName] = Data(model->weight[input0].dataType);
                                            model->weight[mergeName].ggmlType = ((ggml_tensor*) model->weight[input0].ggmlTensor)->type;
                                            model->weight[mergeName].Resize({dim0Len, model->weight[input0].dims[1]});
                                        } else {
                                            model->weight[mergeName] = Data(model->weight[input0].dataType, {dim0Len, model->weight[input0].dims[1]});
                                        }
                                        Data &mergeData = model->weight[mergeName];
                                        mergeData.name = mergeName;
                                        mergeData.perChannelAxis = model->weight[input0].perChannelAxis;
                                        mergeData.group = model->weight[input0].group;
                                        mergeData.groupCnt = model->weight[input0].groupCnt;
                                        mergeData.blockK = model->weight[input0].blockK;
                                        mergeData.blockM = model->weight[input0].blockM;

                                        mergeData.Allocate();
                                        uint64_t offset = 0;
                                        for (auto input : it.inputs) {
                                            mergeData.perChannelsConfigs = AppendVector(mergeData.perChannelsConfigs, model->weight[input].perChannelsConfigs);
                                            mergeData.zeros = AppendVector(mergeData.zeros, model->weight[input].zeros);
                                            mergeData.scales = AppendVector(mergeData.scales, model->weight[input].scales);
                                            mergeData.mins = AppendVector(mergeData.mins, model->weight[input].mins);
                                            mergeData.halfScales = AppendVector(mergeData.halfScales, model->weight[input].halfScales);
                                            memcpy(mergeData.cpuData + offset, model->weight[input].cpuData, model->weight[input].GetBytes());
                                            offset += model->weight[input].GetBytes();
                                        }
                                        mergeData.CalcWeightSum();
#if defined(USE_TFACC) || defined(USE_NUMA)
                                        try {
                                            std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                                            if (s != "" && s != "OFF") {
                                            locker.lock();
                                                if (model->specialWeights.find(mergeName) != model->specialWeights.end()) {
                                                    mergeData.weightSum.resize(1);
                                                    RegisterFastllmData(&mergeData, it.type);       
                                                }
                                                locker.unlock();
                                            }
                                        } catch (...) {
                                        }
#endif
#if defined(USE_NUMAS)
                                        try {
                                            std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                                            if (s != "" && s != "OFF") {
                                                if (model->specialWeights.find(mergeName) != model->specialWeights.end()) {
                                                    mergeData.weightSum.resize(1);
                                                    RegisterNumas(&mergeData);       
                                                }
                                            }
                                        } catch (...) {
                                        }
#endif
                                    }

                                    for (auto input : it.inputs) {
                                        model->weight.weight.erase(input);
                                    }
                                }
                                locker.lock();
                            }
                            locker.unlock();
#if defined(USE_TFACC) || defined(USE_NUMA)
                            try {
                                std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                                if (s != "" && s != "OFF") {
                                    if (!needMerge && model->specialWeights.find(weightName) != model->specialWeights.end()) {
                                        locker.lock();
                                            model->weight.weight[weightName].weightSum.resize(1);
                                            RegisterFastllmData(&model->weight.weight[weightName], model->specialWeights[weightName]);
                                        locker.unlock();
                                    }
                                }
                            } catch (...) {
                            }
#endif
#if defined(USE_NUMAS)
                            try {
                                std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                                if (s != "" && s != "OFF") {
                                    if (!needMerge && model->specialWeights.find(weightName) != model->specialWeights.end()) {
                                        model->weight.weight[weightName].weightSum.resize(1);
                                        RegisterNumas(&model->weight.weight[weightName]);       
                                    }
                                }
                            } catch (...) {
                            }
#endif
                        }

                        if (tensors.size() != 0) {
                            locker.lock();
                            printf("Loading %d \r", (++cnt) * 100 / (int)tensors.size());
                            fflush(stdout);
                            locker.unlock();
                        }
                    }
                }, parts[i].first, parts[i].second)
            );
        }
        for (int i = 0; i < threads.size(); i++) {
            threads[i]->join();
            delete threads[i];
        }

        printf("\n");
        fflush(stdout);

        model->WarmUp();
        return std::unique_ptr<fastllm::basellm> (model);
    }

    std::unique_ptr<fastllm::basellm> CreateLLMModelFromFile(const std::string &fileName) {
        std::string modelType = GetModelTypeFromFile(fileName);
        basellm *model = CreateModelWithType(modelType);
        if(modelType == "bert"){
            BertModel *bertModel = (BertModel*)model;
            bertModel->weight.tokenizer.type = Tokenizer::BERT;
            bertModel->LoadFromFile(fileName);
            bertModel->WarmUp();
        }else{
            model->LoadFromFile(fileName);
            model->WarmUp();
        }
        return std::unique_ptr<fastllm::basellm> (model);
    }

    // 从hf文件夹读取，仅支持safetensor格式的模型
    std::unique_ptr <basellm> CreateLLMModelFromHF(const std::string &modelPath, 
                                                    DataType linearDataType, int groupCnt, bool skipTokenizer, const std::string &modelConfig,
                                                    const std::string &loraPath, bool weightOnly, bool useMoeDataType, DataType moeDataType, int moeGroupCnt,
                                                    const std::string &dtypeConfigString) {
        if (moeGroupCnt == -1) {
            moeGroupCnt = groupCnt;
        }
        std::map <std::string, std::pair <std::string, std::string> > loraDicts;
        SafeTensors *loraTensors = nullptr;
        float loraScaling;
        if (loraPath != "") {
            std::string path = loraPath;
            if (path.back() != '/' || path.back() != '\\') {
                path += "/";
            }
            loraTensors = new SafeTensors({path + "adapter_model.safetensors"});
            for (auto &it : loraTensors->GetSortedItemNames()) {
                if (it.size() >= 31 &&
                    it.substr(0, 17) == "base_model.model." &&
                    (it.substr(it.size() - 14) == ".lora_A.weight" || it.substr(it.size() - 14) == ".lora_B.weight")) {
                    std::string originalName = it.substr(17, it.size() - 31) + ".weight";
                    if (it.substr(it.size() - 14) == ".lora_A.weight") {
                        loraDicts[originalName].first = it;
                    } else {
                        loraDicts[originalName].second = it;
                    }
                }
            }
            std::string loraConfigError;
            auto loraConfig = json11::Json::parse(ReadAllFile(path + "adapter_config.json"), loraConfigError);
            loraScaling = loraConfig["lora_alpha"].number_value() / loraConfig["r"].number_value();
        }

        bool isJsonModel = (modelConfig.size() > 0);
        std::string path = modelPath;
        if (path.back() != '/' || path.back() != '\\') {
            path += "/";
        }

        // 1. 检查是否有 model.safetensors.index.json,如果有就读取
        std::set <std::string> stFiles;
        std::string stIndexFile = path + "model.safetensors.index.json";
        std::string error;
        if (!FileExists(stIndexFile)) {
            stFiles.insert(path + "model.safetensors");
        } else {
            auto stIndex = json11::Json::parse(ReadAllFile(stIndexFile), error)["weight_map"];
            for (auto it : stIndex.object_items()) {
                stFiles.insert(path + it.second.string_value());
            }
        }
        SafeTensors safeTensors(stFiles);

        // 2. 创建网络基本信息
        std::string configFile = path + "config.json";
        auto config = weightOnly ? json11::Json() : json11::Json::parse(ReadAllFile(configFile), error);
        bool isAwqModel = false;
        int awqGroupCnt = 128;
        std::string modelType = "";
        if (weightOnly) {
            modelType = "qwen";
        } else if (isJsonModel) {
            modelType = "fastllmJson";
        } else {
            if (!config["model_type"].is_null()) {
                modelType = config["model_type"].string_value();
            } else {
                modelType = config["architectures"].array_items()[0].string_value();
            }

            if (!config["architectures"].is_null()) {
                std::string arch = config["architectures"].array_items()[0].string_value();
                if (arch == "InternLM2ForCausalLM") {
                    modelType = "internlm2";
                }
            }

            if (!config["quantization_config"].is_null() && config["quantization_config"]["quant_method"] == "awq") {
                auto qconfig = config["quantization_config"];
                AssertInFastLLM(qconfig["quant_method"] == "awq" &&
                                qconfig["bits"] == 4 &&
                                qconfig["version"] == "gemm" &&
                                qconfig["zero_point"].bool_value(), 
                                "Config error: only 4bits AWQ with zero point and gemm version is supported.");
                isAwqModel = true;
                awqGroupCnt = qconfig["group_size"].int_value();
                if (linearDataType != DataType::INT4_GROUP || groupCnt != awqGroupCnt) {
                    printf("WARNING: It is recommended to use \"--dtype int4g%d\" for this AWQ models.\n", awqGroupCnt);
                }
            }
        }
        basellm *model = CreateModelWithType(modelType);
        if (isJsonModel) {
            ((GraphLLMModel*)model)->graphLLMModelConfig->Init(modelConfig);
        }
        AddDictRecursion(model, "", config);
        // 设置eos_token_id
        if (config["eos_token_id"].is_null()) {
            auto tokenizer = json11::Json::parse(ReadAllFile(path + "tokenizer.json"), error);
            if (error == "") {
                std::string tokenizerConfigFile = path + "tokenizer_config.json";
                auto tokenizerConfig = json11::Json::parse(ReadAllFile(tokenizerConfigFile), error);
                std::string eos_token = tokenizerConfig["eos_token"].string_value();
                printf("eos_token = %s\n", eos_token.c_str());
                for (auto added_token : tokenizer["added_tokens"].array_items()) {
                    if (added_token["content"] == eos_token) {
                        model->eos_token_ids.insert(added_token["id"].int_value());
                    }
                }
            }
        } else if (config["eos_token_id"].is_array()) {
            for (auto &it : config["eos_token_id"].array_items()) {
                model->eos_token_ids.insert(it.int_value()); 
            }
        } else {
            model->eos_token_id = config["eos_token_id"].int_value();
        }
        std::string generatetionConfigFile = path + "generation_config.json";
        if (FileExists(generatetionConfigFile)) {
            auto generation_config = json11::Json::parse(ReadAllFile(generatetionConfigFile), error);
            for (auto &it : generation_config.object_items()) {
                if ("eos_token_id" == it.first && it.second.type() == json11::Json::ARRAY)
                    continue;
                model->weight.AddDict(it.first, it.second.is_string() ? it.second.string_value() : it.second.dump());
            }
            // 更新eos_token_id
            if (generation_config["eos_token_id"].is_array()) {
                for (auto &it : generation_config["eos_token_id"].array_items()) {
                    model->eos_token_ids.insert(it.int_value());
                }
            }
        }

        // 3. 读取分词
        if (!skipTokenizer) {
            LoadLLMTokenizerFromHFToModel(path, model);
        } else {
            DealLLMTokenizerFromHFToModel(path, model);
        }

        // 4.0 更新模型信息
        model->InitParams();

        // 4.1 读取权重
        auto tensors = safeTensors.GetSortedItemNames();

if (false) {
    auto temp = tensors;
    tensors.clear();
    for (int i = 0; i < temp.size(); i++) {
        std::string tensorName = temp[i];
        std::string prefix = "model.layers.";
        if (StartWith(tensorName, prefix)) {
            int id = 0;
            for (int i = prefix.size(); tensorName[i] >= '0' && tensorName[i] <= '9'; i++) {
                id = id * 10 + tensorName[i] - '0';
            }
            if (id > 9) {
                continue;
            }
        }
        tensors.push_back(tensorName);
    }
}
        
        // tensorMap[name]代表本名为name的tensor，创建后的名字以及类型
        // 有些tensor被共享，可能需要创建多次
        auto tensorMap = model->GetTensorMap(tensors);

        // 如果有需要，为moe设置特定的量化参数
        if (model->moeLinears.size() > 0 && useMoeDataType) {
            for (auto &it : tensorMap) {
                for (auto &weight : it.second) {
                    if (model->moeLinears.find(weight.first) != model->moeLinears.end()) {
                        weight.second = moeDataType;
                    }
                }
            }
        }

        std::vector <std::pair <std::string, std::string> > dtypeRules;
        if (dtypeConfigString.size() > 0) {
            auto dtypeConfig = json11::Json::parse(dtypeConfigString, error);
            if (error != "") {
                printf("Parse dtype config faild.\n");
                printf("config = %s\n", dtypeConfigString.c_str());
                printf("error = %s\n", error.c_str());
            } else {
                for (auto &it : dtypeConfig.array_items()) {
                    dtypeRules.push_back(std::make_pair(it["key"].string_value(), it["dtype"].string_value()));
                }
            }
        }

        int cur = 0;
        long long totalBytes = 0;
        std::set <std::string> allWeightNames; // 所有创建了的weight name
        std::set <std::string> allFinishNames; // 转换好的weight name

        for (auto &tensorName : tensors) {
            auto &tensor = safeTensors.itmeDict[tensorName];
            auto oriDataType = DataType::FLOAT32;
            for (auto &it : tensorMap[tensorName]) {
                std::string weightName = it.first;
                allWeightNames.insert(weightName);
                auto dataType = it.second;
                int ggmlType = -1;
                if ((dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) && dtypeRules.size() > 0) {
                    int groupCnt = -1;
                    ParseDataType(weightName, dtypeRules, dataType, groupCnt, ggmlType);

                    // 如果原始权重不是FP8_E4M3格式，目前不做转换
                    if (tensor.dtype != "F8_E4M3" && dataType == DataType::FP8_E4M3) {
                        dataType = DataType::FLOAT16;
                    }
                }

                if (dataType >= DATA_AUTO_NONE) {
                    // AUTO类型
                    dataType = (dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) ? linearDataType : oriDataType;
                    
                    // 如果原始权重不是FP8_E4M3格式，目前不做转换
                    if (tensor.dtype != "F8_E4M3" && dataType == DataType::FP8_E4M3) {
                        dataType = DataType::FLOAT16;
                    }
                }
                if (it.second == DATA_AUTO_CONV) {
                    std::vector <int> realShape = tensor.intShape;
                    std::swap(realShape[0], realShape[1]);
                    model->weight.AddEmptyWeight(weightName, realShape, dataType);
                } else if (isAwqModel && StringEndWith(tensorName, ".qweight")) {
                    model->weight.AddEmptyWeight(weightName, {tensor.intShape[1] * 8, tensor.intShape[0]}, dataType);
                } else {
                    if (ggmlType != -1) {
                        model->weight.AddEmptyGGMLWeight(weightName, tensor.intShape, dataType, ggmlType);    
                    } else {
                        model->weight.AddEmptyWeight(weightName, tensor.intShape, dataType);
                    }
                }
            }

            totalBytes += tensor.bytes;
            printf("Load %d \r", (++cur) * 100 / (int)safeTensors.itmeDict.size());
            fflush(stdout);
        }

        // 4.2 读取
        std::vector <std::thread*> threads;
        int threadNum = std::min(16, std::max(4, (int)GetAlivePool()->threads.size()));
        int per = tensors.size() / threadNum;
        std::mutex locker;
        int cnt = 0;

        std::vector <std::pair <int, int> > parts;
        int start = 0;
        for (int i = 0; i < threadNum; i++) {
            int cur = start;
            long long now = 0;
            while (true) {
                if (now * threadNum >= totalBytes || start >= tensors.size()) {
                    break;
                }
                now += safeTensors.itmeDict[tensors[start]].bytes;
                start++;
            }
            parts.push_back(std::make_pair(cur, start));
        }
        parts.back().second = tensors.size();
        while (parts.size() < threadNum) {
            parts.push_back(std::make_pair(-1, -1));
        }

        for (int i = 0; i < threadNum; i++) {
            int st = per * i, end = (i == threadNum - 1) ? tensors.size() : per * (i + 1);
            threads.push_back(
                new std::thread([&](int st, int end) {
                    for (int i = st; i < end; i++) {
                        auto &tensorName = tensors[i];
                        if (StringEndWith(tensorName, "_scale_inv") ||
                            (isAwqModel && (StringEndWith(tensorName, ".scales") || StringEndWith(tensorName, ".qzeros")))) {
                            locker.lock();
                            printf("Loading %d \r", (++cnt) * 100 / (int)tensorMap.size());
                            fflush(stdout);
                            locker.unlock();
                            continue;
                        }
                        auto &tensor = safeTensors.itmeDict[tensorName];
                        std::string scaleTensorName = "";
                        std::string qzeroTensorName = "";

                        for (auto &it : tensorMap[tensorName]) {
                            auto oriDataType = DataType::FLOAT32;
                            std::string weightName = it.first;
                            auto dataType = it.second;
                            int ggmlType = -1;

                            int curGroupCnt = model->moeLinears.find(weightName) != model->moeLinears.end() ? moeGroupCnt : groupCnt;

                            if ((dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) && dtypeRules.size() > 0) {
                                ParseDataType(weightName, dtypeRules, dataType, curGroupCnt, ggmlType);
/*
                                printf("weight \"%s\" -> %s", weightName.c_str(), dataTypeNames[dataType][0].c_str());
                                if (DefaultGroupCnts.find(dataType) != DefaultGroupCnts.end()) {
                                    printf("%d", curGroupCnt);
                                }
                                printf("\n");
*/
                            }
                            if (dataType >= DATA_AUTO_NONE) {
                                // AUTO类型
                                dataType = (dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) ? linearDataType : oriDataType;
                            }
                            if (tensor.dtype == "BF16" &&
                                (dataType == DataType::FLOAT16 || dataType == DataType::BFLOAT16 ||
                                    dataType == DataType::INT8 || dataType == DataType::INT4_GROUP || dataType == DataType::INT4_NOZERO)) {
                                oriDataType = DataType::BFLOAT16;
                            }
                            if (tensor.dtype == "F16" && 
                                dataType == DataType::FLOAT16) {
                                oriDataType = DataType::FLOAT16;
                            }
                            if (tensor.dtype == "F8_E4M3" && 
                                (dataType == DataType::FLOAT32 || dataType == DataType::FLOAT16 || dataType == DataType::INT8 
                                || dataType == DataType::INT4_GROUP || dataType == DataType::INT4_NOZERO
                                || dataType == DataType::INT2_GROUP)
                                || dataType == DataType::DATA_GGUF_FORMAT) {
                                oriDataType = DataType::FLOAT32;
                                scaleTensorName = tensorName + "_scale_inv";
                                if (safeTensors.itmeDict.find(scaleTensorName) == safeTensors.itmeDict.end()) {
                                    scaleTensorName = tensorName + "_scale";
                                }
                                if (safeTensors.itmeDict.find(scaleTensorName) == safeTensors.itmeDict.end()) {
                                    scaleTensorName = "";
                                }
                            }
                            if (tensor.dtype == "F8_E4M3" && 
                                (dataType == FP8_E4M3)) {
                                oriDataType = DataType::FP8_E4M3;
                                scaleTensorName = tensorName + "_scale_inv";
                                if (safeTensors.itmeDict.find(scaleTensorName) == safeTensors.itmeDict.end()) {
                                    scaleTensorName = tensorName + "_scale";
                                }
                                if (safeTensors.itmeDict.find(scaleTensorName) == safeTensors.itmeDict.end()) {
                                    scaleTensorName = "";
                                }
                            }

                            if (tensor.dtype == "I32" && isAwqModel && StringEndWith(tensorName, "qweight")) {
                                std::string name = tensorName.substr(0, tensorName.size() - strlen("qweight"));
                                oriDataType = DataType::FLOAT32;
                                scaleTensorName = name + "scales";
                                qzeroTensorName = name + "qzeros";
                                AssertInFastLLM(safeTensors.itmeDict.find(scaleTensorName) != safeTensors.itmeDict.end() &&
                                                safeTensors.itmeDict.find(qzeroTensorName) != safeTensors.itmeDict.end(),
                                                "Tensor error: can't find AWQ scalse / qzeros.");
                                if (dataType == INT4_GROUP && groupCnt == awqGroupCnt) {
                                    oriDataType = INT4_GROUP;
                                }
                            }

                            if (scaleTensorName == "") {
                                tensor.CreateBuffer(oriDataType);
                            } else if(!isAwqModel) {
                                auto &scaleTensor = safeTensors.itmeDict[scaleTensorName];
                                AssertInFastLLM(scaleTensor.dtype == "F32" || scaleTensor.dtype == "BF16"
                                    , "Tensor scale error: scale's dtype should be F32 or BF16.");
                                scaleTensor.CreateBuffer(DataType::FLOAT32);
                                tensor.CreateBufferWithScale(oriDataType, scaleTensor);
                            } else {
                                auto &scaleTensor = safeTensors.itmeDict[scaleTensorName];
                                auto &qzeroTensor = safeTensors.itmeDict[qzeroTensorName];
                                scaleTensor.CreateBuffer(DataType::FLOAT32);
                                tensor.CreateBufferWithAWQ(oriDataType, scaleTensor, qzeroTensor);
                            }

                            if (loraDicts.find(weightName) != loraDicts.end()) {
                                std::string loraA = loraDicts[weightName].first;
                                std::string loraB = loraDicts[weightName].second;

                                int inDim = loraTensors->itmeDict[loraA].intShape[1];
                                int outDim = loraTensors->itmeDict[loraB].intShape[0];
                                int lora = loraTensors->itmeDict[loraA].intShape[0];

                                AssertInFastLLM((loraTensors->itmeDict[loraA].dtype == "F32" || 
                                                loraTensors->itmeDict[loraA].dtype == "F16" ||
                                                loraTensors->itmeDict[loraA].dtype == "BF16") && 
                                                (loraTensors->itmeDict[loraB].dtype == "F32" || 
                                                loraTensors->itmeDict[loraB].dtype == "F16" ||
                                                loraTensors->itmeDict[loraB].dtype == "BF16"), 
                                                "Lora error: lora's dtype should be F32 or F16 or BF16.");
                                loraTensors->itmeDict[loraA].CreateBuffer(DataType::FLOAT32);
                                loraTensors->itmeDict[loraB].CreateBuffer(DataType::FLOAT32);
                                float *weightA = (float*)loraTensors->itmeDict[loraA].buffer;
                                float *weightB = (float*)loraTensors->itmeDict[loraB].buffer;

                                std::vector <float> loraFactor;
                                loraFactor.resize(inDim * outDim, 0.0f);
                                for (int i = 0; i < outDim; i++) {
                                    for (int j = 0; j < lora; j++) {
                                        for (int k = 0; k < inDim; k++) {
                                            loraFactor[i * inDim + k] += weightB[i * lora + j] * weightA[j * inDim + k];
                                        }
                                    }
                                }
                                for (int i = 0; i < loraFactor.size(); i++) {
                                    loraFactor[i] *= loraScaling;
                                }

                                loraTensors->itmeDict[loraA].ClearBuffer();
                                loraTensors->itmeDict[loraB].ClearBuffer();

                                if (oriDataType == DataType::BFLOAT16) {
                                    uint16_t *fp16Weight = (uint16_t*)tensor.buffer;
                                    for (int i = 0; i < loraFactor.size(); i++) {
                                        uint32_t now = fp16Weight[i] << 16;
                                        float newV = ((float*)&now)[0] + loraFactor[i];
                                        fp16Weight[i] = ((uint32_t*)&newV)[0] >> 16;
                                    }
                                } else if (oriDataType == DataType::FLOAT16) {
                                    uint16_t *fp16Weight = (uint16_t*)tensor.buffer;
                                    for (int i = 0; i < loraFactor.size(); i++) {
                                        fp16Weight[i] = float_to_half(half_to_float(fp16Weight[i]) + loraFactor[i]);
                                    }
                                } else if (oriDataType == DataType::FLOAT32) {
                                    float *fp32Weight = (float*)tensor.buffer;
                                    for (int i = 0; i < loraFactor.size(); i++) {
                                        fp32Weight[i] = fp32Weight[i] + loraFactor[i];
                                    }
                                } else {
                                    ErrorInFastLLM("Lora error, dtype should be float32, float16 or bfloat16.");
                                }
                            }

                            if (tensor.dtype == "fastllm") {
                                model->weight[weightName].CreateFromFastllmFormat(tensor.buffer, tensor.bytes);
                            } else {
                                if (it.second == DATA_AUTO_CONV) {
                                    tensor.Transpose(oriDataType);
                                }
                                model->weight[weightName].CreateFromOriData(WeightType::AUTO, oriDataType, 
                                        tensor.buffer, tensor.minsBuffer, tensor.scalesBuffer,
                                        curGroupCnt, tensor.blockK, tensor.blockM);
                            }
                            if (it.second == DATA_AUTO_LINEAR || it.second == DATA_AUTO_CONV)
                                model->weight[weightName].CalcWeightSum();
                            tensor.ClearBuffer();

                            locker.lock();
                            allFinishNames.insert(weightName);
                            // 检查是否需要合并权重
                            bool needMerge = false;
                            for (auto &rule : model->weightMergeRules) {
                                if (rule.allInputs.find(weightName) == rule.allInputs.end()) {
                                    continue;
                                }
                                needMerge = true;
                                bool canMerge = true;
                                for (auto &name : rule.allInputs) {
                                    if (allWeightNames.find(name) != allWeightNames.end() && 
                                        allFinishNames.find(name) == allFinishNames.end()) {
                                        canMerge = false;
                                    }
                                }
                                if (!canMerge) {
                                    continue;
                                }
                                for (auto &it : rule.rules) {
                                    for (auto input : it.inputs) {
                                        if (model->weight[input].dims.size() == 2) {
                                            if (model->weight[input].groupCnt != -1 && 
                                                model->weight[input].dims[1] % model->weight[input].groupCnt != 0) {
                                                canMerge = false;
                                                break;
                                            }
                                            if (model->weight[input].blockK != -1 && 
                                                model->weight[input].dims[0] % model->weight[input].blockK != 0) {
                                                canMerge = false;
                                                break;
                                            }
                                            if (model->weight[input].blockM != -1 && 
                                                model->weight[input].dims[1] % model->weight[input].blockM != 0) {
                                                canMerge = false;
                                                break;
                                            }
                                            if (model->weight[input].dataType != model->weight[it.inputs[0]].dataType ||
                                                model->weight[input].dims[1] != model->weight[it.inputs[0]].dims[1]) {
                                                canMerge = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (!canMerge) {
                                    continue;
                                }

                                locker.unlock();
                                for (auto &it : rule.rules) {
                                    if (allWeightNames.find(it.inputs[0]) == allWeightNames.end()) {
                                        continue;
                                    }
                                    int dim0Len = 0;
                                    for (auto input : it.inputs) {
                                        dim0Len += model->weight[input].dims[0];
                                    }
                                    if (model->weight[it.inputs[0]].dims.size() == 1) {
                                        std::string input0 = it.inputs[0];
                                        std::string mergeName = it.output;
                                        if (model->weight[input0].dataType == DATA_GGUF_FORMAT) {
                                            model->weight[mergeName] = Data(model->weight[input0].dataType);
                                            model->weight[mergeName].ggmlType = ((ggml_tensor*) model->weight[input0].ggmlTensor)->type;
                                            model->weight[mergeName].Resize({dim0Len});
                                        } else {
                                            model->weight[mergeName] = Data(model->weight[input0].dataType, {dim0Len});
                                        }
                                        Data &mergeData = model->weight[mergeName];
                                        mergeData.name = mergeName;
                                        mergeData.Allocate();
                                        uint64_t offset = 0;
                                        for (auto input : it.inputs) {
                                            memcpy(mergeData.cpuData + offset, model->weight[input].cpuData, model->weight[input].GetBytes());
                                            offset += model->weight[input].GetBytes();
                                        }
                                    } else {
                                        std::string input0 = it.inputs[0];
                                        std::string mergeName = it.output;
                                        if (model->weight[input0].dataType == DATA_GGUF_FORMAT) {
                                            model->weight[mergeName] = Data(model->weight[input0].dataType);
                                            model->weight[mergeName].ggmlType = ((ggml_tensor*) model->weight[input0].ggmlTensor)->type;
                                            model->weight[mergeName].Resize({dim0Len, model->weight[input0].dims[1]});
                                        } else {
                                            model->weight[mergeName] = Data(model->weight[input0].dataType, {dim0Len, model->weight[input0].dims[1]});
                                        }
                                        Data &mergeData = model->weight[mergeName];
                                        mergeData.name = mergeName;
                                        mergeData.perChannelAxis = model->weight[input0].perChannelAxis;
                                        mergeData.group = model->weight[input0].group;
                                        mergeData.groupCnt = model->weight[input0].groupCnt;
                                        mergeData.blockK = model->weight[input0].blockK;
                                        mergeData.blockM = model->weight[input0].blockM;

                                        mergeData.Allocate();
                                        uint64_t offset = 0;
                                        for (auto input : it.inputs) {
                                            mergeData.perChannelsConfigs = AppendVector(mergeData.perChannelsConfigs, model->weight[input].perChannelsConfigs);
                                            mergeData.zeros = AppendVector(mergeData.zeros, model->weight[input].zeros);
                                            mergeData.scales = AppendVector(mergeData.scales, model->weight[input].scales);
                                            mergeData.mins = AppendVector(mergeData.mins, model->weight[input].mins);
                                            mergeData.halfScales = AppendVector(mergeData.halfScales, model->weight[input].halfScales);
                                            memcpy(mergeData.cpuData + offset, model->weight[input].cpuData, model->weight[input].GetBytes());
                                            offset += model->weight[input].GetBytes();
                                        }

                                        mergeData.CalcWeightSum();
#if defined(USE_TFACC) || defined(USE_NUMA)
                                        try {
                                            std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                                            if (s != "" && s != "OFF") {
                                                locker.lock();
                                                if (model->specialWeights.find(mergeName) != model->specialWeights.end()) {
                                                    mergeData.weightSum.resize(1);
                                                    RegisterFastllmData(&mergeData, it.type);       
                                                }
                                                locker.unlock();
                                            }
                                        } catch (...) {
                                        }
#endif
#if defined(USE_NUMAS)
                                        try {
                                            std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                                            if (s != "" && s != "OFF") {
                                                if (model->specialWeights.find(mergeName) != model->specialWeights.end()) {
                                                    mergeData.weightSum.resize(1);
                                                    RegisterNumas(&mergeData);       
                                                }
                                            }
                                        } catch (...) {
                                        }
#endif
                                    }

                                    for (auto input : it.inputs) {
                                        model->weight.weight.erase(input);
                                    }
                                }
                                locker.lock();
                            }
                            locker.unlock();
#if defined(USE_TFACC) || defined(USE_NUMA)
                            try {
                                std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                                if (s != "" && s != "OFF") {
                                    if (!needMerge && model->specialWeights.find(weightName) != model->specialWeights.end()) {
                                        locker.lock();
                                            model->weight.weight[weightName].weightSum.resize(1);
                                            RegisterFastllmData(&model->weight.weight[weightName], model->specialWeights[weightName]);
                                        locker.unlock();
                                    }
                                }
                            } catch (...) {
                            }
#endif
#if defined(USE_NUMAS)
                            try {
                                std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                                if (s != "" && s != "OFF") {
                                    if (!needMerge && model->specialWeights.find(weightName) != model->specialWeights.end()) {
                                        model->weight.weight[weightName].weightSum.resize(1);
                                        RegisterNumas(&model->weight.weight[weightName]);       
                                    }
                                }
                            } catch (...) {
                            }
#endif
                        }

                        locker.lock();
                        printf("Loading %d \r", (++cnt) * 100 / (int)tensorMap.size());
                        fflush(stdout);
                        locker.unlock();
                    }
                }, parts[i].first, parts[i].second)
            );
        }
        for (int i = 0; i < threads.size(); i++) {
            threads[i]->join();
            delete threads[i];
        }

        printf("\n");
        fflush(stdout);

        delete loraTensors;

        if (!weightOnly)
            model->WarmUp();
        return std::unique_ptr<fastllm::basellm> (model);
    }

    // 从hf文件夹读取，仅支持safetensor格式的模型，然后导出成safetensor格式
    void ExportLLMModelFromHF(const std::string &modelPath, 
                            DataType linearDataType, int groupCnt, const std::string &exportPath, const std::string &modelConfig,
                            const std::string &loraPath, bool useMoeDataType, DataType moeDataType, int moeGroupCnt,
                            const std::string &dtypeConfigString) {
        if (moeGroupCnt == -1) {
            moeGroupCnt = groupCnt;
        }
        // 检查源目录是否存在
        if (!fs::exists(modelPath) || !fs::is_directory(modelPath)) {
            std::cerr << "源目录不存在或不是一个目录: " << modelPath << std::endl;
            return;
        }

        // 检查目标目录是否存在，如果不存在则创建
        if (!fs::exists(exportPath)) {
            fs::create_directories(exportPath);
        }

        // 遍历源目录中的所有文件
        for (const auto& entry : fs::directory_iterator(modelPath)) {
            if (fs::is_regular_file(entry)) {
                // 获取文件扩展名
                std::string extension = entry.path().extension().string();

                // 如果文件扩展名不是 ".safetensors"，则复制文件
                if (extension != ".safetensors") {
                    fs::path destinationFile = exportPath / entry.path().filename();
                    fs::copy_file(entry.path(), destinationFile, fs::copy_options::overwrite_existing);
                    std::cout << "Copy file: " << entry.path() << " -> " << destinationFile << std::endl;
                }
            }
        }

        std::map <std::string, std::pair <std::string, std::string> > loraDicts;
        SafeTensors *loraTensors = nullptr;
        float loraScaling;
        if (loraPath != "") {
            std::string path = loraPath;
            if (path.back() != '/' || path.back() != '\\') {
                path += "/";
            }
            loraTensors = new SafeTensors({path + "adapter_model.safetensors"});
            for (auto &it : loraTensors->GetSortedItemNames()) {
                if (it.size() >= 31 &&
                    it.substr(0, 17) == "base_model.model." &&
                    (it.substr(it.size() - 14) == ".lora_A.weight" || it.substr(it.size() - 14) == ".lora_B.weight")) {
                    std::string originalName = it.substr(17, it.size() - 31) + ".weight";
                    if (it.substr(it.size() - 14) == ".lora_A.weight") {
                        loraDicts[originalName].first = it;
                    } else {
                        loraDicts[originalName].second = it;
                    }
                }
            }
            std::string loraConfigError;
            auto loraConfig = json11::Json::parse(ReadAllFile(path + "adapter_config.json"), loraConfigError);
            loraScaling = loraConfig["lora_alpha"].number_value() / loraConfig["r"].number_value();
        }

        bool isJsonModel = (modelConfig.size() > 0);
        std::string path = modelPath;
        if (path.back() != '/' || path.back() != '\\') {
            path += "/";
        }
        std::string outputPath = exportPath;
        if (outputPath.back() != '/' && outputPath.back() != '\\') {
            outputPath += "/";
        }

        // 1. 检查是否有 model.safetensors.index.json,如果有就读取
        std::set <std::string> stFiles;
        std::map <std::string, std::string> outputFileDict;
        std::string stIndexFile = path + "model.safetensors.index.json";
        std::string error;
        if (!FileExists(stIndexFile)) {
            stFiles.insert(path + "model.safetensors");
            outputFileDict[path + "model.safetensors"] = outputPath + "model.safetensors";
        } else {
            auto stIndex = json11::Json::parse(ReadAllFile(stIndexFile), error)["weight_map"];
            for (auto it : stIndex.object_items()) {
                stFiles.insert(path + it.second.string_value());
                outputFileDict[path + it.second.string_value()] = outputPath + it.second.string_value();
            }
        }
        SafeTensors safeTensors(stFiles);

        // 2. 创建网络基本信息
        std::string configFile = path + "config.json";
        auto config = json11::Json::parse(ReadAllFile(configFile), error);
        std::string modelType = "";
        if (!config["model_type"].is_null()) {
            modelType = config["model_type"].string_value();
        } else {
            modelType = config["architectures"].array_items()[0].string_value();
        }
        basellm *model = CreateModelWithType(modelType);
        /*if (isJsonModel) {
            ((GraphLLMModel*)model)->graphLLMModelConfig->Init(modelConfig);
        }*/
        AddDictRecursion(model, "", config);
        // 4.0 更新模型信息
        model->InitParams();

        // 4.1 读取权重
        auto tensors = safeTensors.GetSortedItemNames();
        auto tensorMap = model->GetTensorMap(tensors);

        // 如果有需要，为moe设置特定的量化参数
        if (useMoeDataType && model->moeLinears.size() > 0) {
            for (auto &it : tensorMap) {
                for (auto &weight : it.second) {
                    if (model->moeLinears.find(weight.first) != model->moeLinears.end()) {
                        weight.second = moeDataType;
                    }
                }
            }
        }

        std::vector <std::pair <std::string, std::string> > dtypeRules;
        if (dtypeConfigString.size() > 0) {
            auto dtypeConfig = json11::Json::parse(dtypeConfigString, error);
            if (error != "") {
                printf("Parse dtype config faild.\n");
                printf("config = %s\n", dtypeConfigString.c_str());
                printf("error = %s\n", error.c_str());
            } else {
                for (auto &it : dtypeConfig.array_items()) {
                    dtypeRules.push_back(std::make_pair(it["key"].string_value(), it["dtype"].string_value()));
                }
            }
        }

        if (dtypeRules.size() > 0) {
            printf("Dtype rules:\n");
            for (auto &it : dtypeRules) {
                printf("%s: %s\n", it.first.c_str(), it.second.c_str());
            }
        }

        for (auto &file : safeTensors.fileNames) {
            std::map <std::string, Data> weights;
            std::string outputFileName = outputFileDict[file];
            printf("Export weight model: %s\n", outputFileName.c_str());
            std::vector <SafeTensorItem*> items;
            for (auto &it : safeTensors.itmeDict) {
                if (it.second.fileName == file) {
                    items.push_back(&it.second);
                }
            }

            // 1.0 创建 weights
            json11::Json::object config;
            for (auto it : items) {
                auto &tensor = *it;
                auto oriDataType = DataType::FLOAT32;
                auto dataType = tensorMap[tensor.tensorName][0].second;
                auto weightName = tensor.tensorName;
                int ggmlType = -1;

                if ((dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) && dtypeRules.size() > 0) {
                    int groupCnt = -1;
                    ParseDataType(weightName, dtypeRules, dataType, groupCnt, ggmlType);

                    // 如果原始权重不是FP8_E4M3格式，目前不做转换
                    if (tensor.dtype != "F8_E4M3" && dataType == DataType::FP8_E4M3) {
                        dataType = DataType::FLOAT16;
                    }
                }

                if (dataType >= DATA_AUTO_NONE) {
                    // AUTO类型
                    dataType = (dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) ? linearDataType : oriDataType;

                    // 如果原始权重不是FP8_E4M3格式，目前不做转换
                    if (tensor.dtype != "F8_E4M3" && dataType == DataType::FP8_E4M3) {
                        dataType = DataType::FLOAT16;
                    }
                }
                if (dataType== DATA_AUTO_CONV) {
                    std::vector <int> realShape = tensor.intShape;
                    std::swap(realShape[0], realShape[1]);
                    weights[weightName] = Data(dataType, realShape);
                } else {
                    if (dataType == DATA_GGUF_FORMAT) {
                        weights[weightName] = Data(dataType, ggmlType, tensor.intShape);    
                    } else {
                        weights[weightName] = Data(dataType, tensor.intShape);
                    }
                }
            }

            // 2.0 转模型，存储
            std::vector <std::thread*> threads;
            int threadNum = std::min(16, std::max(4, (int)GetAlivePool()->threads.size()));
            int per = items.size() / threadNum;

            for (int i = 0; i < threadNum; i++) {
                int st = per * i, end = (i == threadNum - 1) ? items.size() : per * (i + 1);
                threads.push_back(
                    new std::thread([&](int st, int end) {
                        for (int i = st; i < end; i++) {
                            auto &tensor = *items[i];
                            if (StringEndWith(tensor.tensorName, "_scale_inv")) {
                                continue;
                            }
                            std::string scaleTensorName = "";
                            std::string weightName = tensor.tensorName;

                            auto dataType = tensorMap[tensor.tensorName][0].second;
                            auto oriDataType = DataType::FLOAT32;
                            int ggmlType = -1;
                            int curGroupCnt = model->moeLinears.find(weightName) != model->moeLinears.end() ? moeGroupCnt : groupCnt;
                            if ((dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) && dtypeRules.size() > 0) {
                                ParseDataType(weightName, dtypeRules, dataType, curGroupCnt, ggmlType);
                                if (dataType == DATA_GGUF_FORMAT) {
                                    printf("weight \"%s\" -> %s\n", weightName.c_str(), ggml_type_name((ggml_type)ggmlType));
                                } else {
                                    printf("weight \"%s\" -> %s", weightName.c_str(), dataTypeNames[dataType][0].c_str());
                                    if (DefaultGroupCnts.find(dataType) != DefaultGroupCnts.end()) {
                                        printf("%d", curGroupCnt);
                                    }
                                    printf("\n");
                                }
                            }

                            if (dataType >= DATA_AUTO_NONE) {
                                // AUTO类型
                                dataType = (dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) ? linearDataType : oriDataType;
                            }
                            if (tensor.dtype == "BF16" &&
                                (dataType == DataType::FLOAT16 || dataType == DataType::INT8 || dataType == DataType::INT4_GROUP || dataType == DataType::INT4_NOZERO)) {
                                oriDataType = DataType::BFLOAT16;
                            }
                            if (tensor.dtype == "F16" && 
                                dataType == DataType::FLOAT16) {
                                oriDataType = DataType::FLOAT16;
                            }
                            if (tensor.dtype == "F8_E4M3" && 
                                (dataType == DataType::FLOAT32 || dataType == DataType::FLOAT16 || dataType == DataType::INT8 || dataType == DataType::INT4_GROUP || dataType == DataType::INT4_NOZERO || dataType == DataType::DATA_GGUF_FORMAT)) {
                                oriDataType = DataType::FLOAT32;
                                scaleTensorName = tensor.tensorName + "_scale_inv";
                                if (safeTensors.itmeDict.find(scaleTensorName) == safeTensors.itmeDict.end()) {
                                    scaleTensorName = "";
                                }
                            }
                            if (tensor.dtype == "F8_E4M3" && 
                                (dataType == FP8_E4M3)) {
                                oriDataType = DataType::FP8_E4M3;
                                scaleTensorName = tensor.tensorName + "_scale_inv";
                                if (safeTensors.itmeDict.find(scaleTensorName) == safeTensors.itmeDict.end()) {
                                    scaleTensorName = "";
                                }
                            }

                            if (scaleTensorName == "") {
                                tensor.CreateBuffer(oriDataType);
                            } else {
                                auto &scaleTensor = safeTensors.itmeDict[scaleTensorName];
                                AssertInFastLLM(scaleTensor.dtype == "F32" || scaleTensor.dtype == "BF16"
                                    , "Tensor scale error: scale's dtype should be F32 or BF16.");
                                scaleTensor.CreateBuffer(DataType::FLOAT32);
                                tensor.CreateBufferWithScale(oriDataType, scaleTensor);
                            }

                            if (loraDicts.find(weightName) != loraDicts.end()) {
                                std::string loraA = loraDicts[weightName].first;
                                std::string loraB = loraDicts[weightName].second;

                                int inDim = loraTensors->itmeDict[loraA].intShape[1];
                                int outDim = loraTensors->itmeDict[loraB].intShape[0];
                                int lora = loraTensors->itmeDict[loraA].intShape[0];

                                AssertInFastLLM((loraTensors->itmeDict[loraA].dtype == "F32" || 
                                                    loraTensors->itmeDict[loraA].dtype == "F16" ||
                                                    loraTensors->itmeDict[loraA].dtype == "BF16") && 
                                                    (loraTensors->itmeDict[loraB].dtype == "F32" || 
                                                    loraTensors->itmeDict[loraB].dtype == "F16" ||
                                                    loraTensors->itmeDict[loraB].dtype == "BF16"), 
                                                    "Lora error: lora's dtype should be F32 or F16 or BF16.");
                                loraTensors->itmeDict[loraA].CreateBuffer(DataType::FLOAT32);
                                loraTensors->itmeDict[loraB].CreateBuffer(DataType::FLOAT32);
                                float *weightA = (float*)loraTensors->itmeDict[loraA].buffer;
                                float *weightB = (float*)loraTensors->itmeDict[loraB].buffer;

                                std::vector <float> loraFactor;
                                loraFactor.resize(inDim * outDim, 0.0f);
                                for (int i = 0; i < outDim; i++) {
                                    for (int j = 0; j < lora; j++) {
                                        for (int k = 0; k < inDim; k++) {
                                            loraFactor[i * inDim + k] += weightB[i * lora + j] * weightA[j * inDim + k];
                                        }
                                    }
                                }
                                for (int i = 0; i < loraFactor.size(); i++) {
                                    loraFactor[i] *= loraScaling;
                                }

                                loraTensors->itmeDict[loraA].ClearBuffer();
                                loraTensors->itmeDict[loraB].ClearBuffer();

                                if (oriDataType == DataType::BFLOAT16) {
                                    uint16_t *fp16Weight = (uint16_t*)tensor.buffer;
                                    for (int i = 0; i < loraFactor.size(); i++) {
                                        uint32_t now = fp16Weight[i] << 16;
                                        float newV = ((float*)&now)[0] + loraFactor[i];
                                        fp16Weight[i] = ((uint32_t*)&newV)[0] >> 16;
                                    }
                                } else if (oriDataType == DataType::FLOAT16) {
                                    uint16_t *fp16Weight = (uint16_t*)tensor.buffer;
                                    for (int i = 0; i < loraFactor.size(); i++) {
                                        fp16Weight[i] = float_to_half(half_to_float(fp16Weight[i]) + loraFactor[i]);
                                    }
                                } else if (oriDataType == DataType::FLOAT32) {
                                    float *fp32Weight = (float*)tensor.buffer;
                                    for (int i = 0; i < loraFactor.size(); i++) {
                                        fp32Weight[i] = fp32Weight[i] + loraFactor[i];
                                    }
                                } else {
                                    ErrorInFastLLM("Lora error, dtype should be float32, float16 or bfloat16.");
                                }
                            }

                            if (dataType == DATA_AUTO_CONV) {
                                tensor.Transpose(oriDataType);
                            }
                            weights[weightName].CreateFromOriData(WeightType::AUTO, oriDataType, 
                                tensor.buffer, tensor.minsBuffer, tensor.scalesBuffer,
                                curGroupCnt, tensor.blockK, tensor.blockM);
                            tensor.ClearBuffer();
                        }
                    }, st, end)
                );
            }
            for (int i = 0; i < threads.size(); i++) {
                threads[i]->join();
                delete threads[i];
            }

            std::map <std::string, std::vector <long long> > offsets;
            long long currentOffset = 0;
            for (auto it : items) {
                std::string weightName = it->tensorName;
                long long currentBytes = weights[weightName].GetFastllmFormateBytes();
                offsets[weightName] = {currentOffset, currentOffset + currentBytes};
                currentOffset += currentBytes;
                std::string dtype = "fastllm";
                DataType realType = weights[weightName].dataType;
                if (realType == FLOAT16) {
                    dtype = "F16";
                } else if (realType == FLOAT32) {
                    dtype = "F32";
                } else if (realType == BFLOAT16) {
                    dtype = "BF16";
                }
                config[weightName] = json11::Json::object {
                        {"dtype", dtype},
                        {"shape", json11::Json(weights[weightName].dims)},
                        {"data_offsets", json11::Json(offsets[weightName])}
                };
            }

            std::string configString = json11::Json(config).dump();
            uint64_t totalLen = 8 + configString.size() + currentOffset;
            std::vector <uint8_t> bytes;
            bytes.resize(currentOffset);

            for (auto it : items) {
                std::string weightName = it->tensorName;
                if (StringEndWith(weightName, "_scale_inv")) {
                    continue;
                }
                weights[weightName].ExportFastllmFormat(bytes.data() + offsets[weightName][0]);
            }

            FILE *outputFile = fopen(outputFileName.c_str(), "wb");
            uint64_t configLen = configString.size();
            fwrite(&configLen, sizeof(uint64_t), 1, outputFile);
            fwrite(configString.data(), 1, configString.size(), outputFile);
            fwrite(bytes.data(), 1, bytes.size(), outputFile);
            fclose(outputFile);
        }
        delete loraTensors;        
        return;
    }
}
