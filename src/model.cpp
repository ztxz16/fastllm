#include "utils.h"
#include "json11.hpp"

#include "model.h"
#include "fastllm.h"
#include "executor.h"
#include <sstream>
#include <fstream>
#include <regex>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <algorithm>
#include <cctype>
#include <mutex>

#include "chatglm.h"
#include "moss.h"
#include "llama.h"
#include "moe.h"
#include "qwen2.h"
#include "qwen3.h"
#include "qwen3_moe.h"
#include "qwen3_next.h"
#include "qwen3_5.h"
#include "step3p5.h"
#include "minimax_m2.h"
#include "hunyuan.h"
#include "deepseekv2.h"
#include "deepseekv4.h"
#include "qwen.h"
#include "glm.h"
#include "minicpm.h"
#include "minicpm3.h"
#include "internlm2.h"
#include "bert.h"
#include "xlmroberta.h"
#include "graphllm.h"
#include "gemma4.h"
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

#ifdef USE_CUDA
#include "devices/multicuda/fastllm-multicuda.cuh"
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
        this->layeredMoeDeviceMap = GetLayeredMoeDeviceMap();
        this->moeDeviceLayers = GetMoeDeviceLayers();
    }

    void basellm::AddSpecialWeight(const std::string &weightName, const std::string &weightType, int layerId) {
        this->specialWeights[weightName] = weightType;
        this->specialWeightLayerIds[weightName] = layerId;
    }

    bool basellm::UseLayeredMoeDevice(int layerId) const {
        if (this->moeDeviceLayers < 0 || this->layeredMoeDeviceMap.empty() ||
            this->block_cnt <= 0 || layerId < 0) {
            return false;
        }
        if (this->moeDeviceLayers <= 0) {
            return false;
        }
        int layeredLayers = std::min(this->moeDeviceLayers, this->block_cnt);
        int firstLayer = this->block_cnt - layeredLayers;
        return layerId >= firstLayer && layerId < this->block_cnt;
    }

    std::string basellm::SelectMoeDeviceForLayer(int layerId) const {
        if (this->block_cnt <= 0) {
            if (!this->moeDeviceMap.empty()) {
                return SelectDeviceFromMap(this->moeDeviceMap, 1, 1);
            }
            return SelectDeviceFromMap(this->deviceMap, 1, 1);
        }

        if (this->UseLayeredMoeDevice(layerId)) {
            int layeredLayers = std::min(this->moeDeviceLayers, this->block_cnt);
            int firstLayer = this->block_cnt - layeredLayers;
            return SelectDeviceFromMap(this->layeredMoeDeviceMap, layerId - firstLayer + 1, layeredLayers);
        }

        const auto &frontMap = this->moeDeviceMap.empty() ? this->deviceMap : this->moeDeviceMap;
        int frontLayers = this->block_cnt;
        if (this->moeDeviceLayers >= 0) {
            int layeredLayers = std::min(std::max(this->moeDeviceLayers, 0), this->block_cnt);
            frontLayers = std::max(1, this->block_cnt - layeredLayers);
        }
        return SelectDeviceFromMap(frontMap, std::min(layerId + 1, frontLayers), frontLayers);
    }

    void basellm::ApplyMoeDeviceMapForLayer(int layerId) const {
        std::string selectedDevice = this->SelectMoeDeviceForLayer(layerId);
        if (selectedDevice.empty()) {
            return;
        }
        ((Executor*)GetExecutor())->SetFirstDevice(selectedDevice);
    }

    static bool DeviceNameMatchesType(const std::string &deviceName, const std::string &deviceType) {
        if (deviceName == deviceType) {
            return true;
        }
        return deviceName.size() > deviceType.size() &&
               deviceName.compare(0, deviceType.size(), deviceType) == 0 &&
               deviceName[deviceType.size()] == ':';
    }

#ifdef USE_CUDA
    static std::mutex multiCudaTpLoadSplitLock;

    static std::string TrimAndLower(const std::string &s) {
        int l = 0, r = (int)s.size();
        while (l < r && std::isspace((unsigned char)s[l])) {
            l++;
        }
        while (r > l && std::isspace((unsigned char)s[r - 1])) {
            r--;
        }
        std::string ret = s.substr(l, r - l);
        std::transform(ret.begin(), ret.end(), ret.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        return ret;
    }

    static bool IsDisabledTpSpec(const std::string &spec) {
        return spec.empty() || spec == "false" || spec == "off" ||
               spec == "none" || spec == "disable";
    }

    static bool IsThreadTensorParallelLoadEnabled() {
        const char *envNames[] = {
            "FASTLLM_TP",
            "FASTLLM_QWEN3_MOE_TP",
            "FASTLLM_QWEN3_THREAD_TP",
            "FASTLLM_STEP3P5_TP"
        };
        for (auto envName : envNames) {
            const char *env = std::getenv(envName);
            if (env != nullptr && !IsDisabledTpSpec(TrimAndLower(env))) {
                return true;
            }
        }
        return false;
    }

    static bool HasExplicitRatiosForAllDevices(const std::vector<int> &devices,
                                               const std::map<int, int> &ratios) {
        if (devices.empty() || ratios.empty()) {
            return false;
        }
        for (int device : devices) {
            if (ratios.find(device) == ratios.end()) {
                return false;
            }
        }
        return true;
    }

    static bool SplitSpecialWeightToCudaTpDevices(const basellm *model,
                                                  const std::string &weightName,
                                                  Data &data,
                                                  const std::vector<int> &deviceIds,
                                                  std::map<int, int> &ratios) {
        if ((model->model_type != "qwen3_moe" &&
             model->model_type != "step3p5" &&
             model->model_type != "minimax_m2") ||
            !IsThreadTensorParallelLoadEnabled() || deviceIds.size() <= 1 ||
            data.isDiskWeight || data.dims.size() != 2 ||
            (data.cpuData == nullptr && data.cudaData == nullptr && data.numasData.empty())) {
            return false;
        }
        auto typeIt = model->specialWeights.find(weightName);
        if (typeIt == model->specialWeights.end()) {
            return false;
        }

        std::vector<int> devices = deviceIds;
        Data emptyBias;
        bool explicitDeviceRatios = HasExplicitRatiosForAllDevices(devices, ratios);
        std::lock_guard<std::mutex> guard(multiCudaTpLoadSplitLock);
        if (typeIt->second == "linearSwiglu") {
            data.tpLinearType = TP_LINEAR_ROW;
            data.tpPackType = TP_PACK_GATEUP;
            DivisionScheme scheme = BuildMultiCudaRowSplitScheme(data, devices, ratios);
            return SplitMultiCudaWeight(data, emptyBias, devices, scheme, 0, explicitDeviceRatios);
        }
        if (typeIt->second == "linearColumn") {
            data.tpLinearType = TP_LINEAR_COLUMN;
            DivisionScheme scheme = BuildMultiCudaColumnSplitScheme(data, devices, ratios);
            return SplitMultiCudaWeight(data, emptyBias, devices, scheme, 1, explicitDeviceRatios);
        }
        return false;
    }
#endif

    static std::string GetSpecialWeightSelectedDevice(const basellm *model, const std::string &weightName) {
        if (model->specialWeights.find(weightName) == model->specialWeights.end()) {
            return "";
        }
        if (model->moeDeviceMap.empty() &&
            (model->moeDeviceLayers < 0 || model->layeredMoeDeviceMap.empty())) {
            return "";
        }
        auto layerIt = model->specialWeightLayerIds.find(weightName);
        if (layerIt == model->specialWeightLayerIds.end() || layerIt->second < 0) {
            return "";
        }
        return model->SelectMoeDeviceForLayer(layerIt->second);
    }

    static std::string GetMoeWeightSelectedDevice(const basellm *model, const std::string &weightName) {
        if (model == nullptr) {
            return "";
        }
        std::string selectedDevice = GetSpecialWeightSelectedDevice(model, weightName);
        if (!selectedDevice.empty()) {
            return selectedDevice;
        }
        if (model->moeLinears.find(weightName) == model->moeLinears.end()) {
            return "";
        }
        for (auto &mergeRule : model->weightMergeRules) {
            for (auto &rule : mergeRule.rules) {
                if (std::find(rule.inputs.begin(), rule.inputs.end(), weightName) == rule.inputs.end()) {
                    continue;
                }
                selectedDevice = GetSpecialWeightSelectedDevice(model, rule.output);
                if (!selectedDevice.empty()) {
                    return selectedDevice;
                }
            }
        }
        return "";
    }

    bool basellm::ShouldRegisterSpecialWeightForDeviceType(const std::string &weightName, const std::string &deviceType) const {
        return this->ShouldRegisterSpecialWeightForDeviceTypes(weightName, {deviceType});
    }

    bool basellm::ShouldRegisterSpecialWeightForDeviceTypes(const std::string &weightName, const std::vector<std::string> &deviceTypes) const {
        if (!GetFastllmEnv().activateNuma || this->specialWeights.find(weightName) == this->specialWeights.end()) {
            return false;
        }
        std::string selectedDevice = GetSpecialWeightSelectedDevice(this, weightName);
        if (selectedDevice.empty()) {
            return true;
        }
        for (auto &deviceType : deviceTypes) {
            if (DeviceNameMatchesType(selectedDevice, deviceType)) {
                return true;
            }
        }
        return false;
    }

    bool basellm::MoveSpecialWeightToCudaIfNeeded(const std::string &weightName, Data &data) const {
        if (this->ShouldDelaySpecialWeightCudaMove(weightName)) {
            return false;
        }
        std::string selectedDevice = GetSpecialWeightSelectedDevice(this, weightName);
        if (!DeviceNameMatchesType(selectedDevice, "cuda")) {
            return false;
        }
#ifdef USE_CUDA
        if (data.isDiskWeight || (data.dataDevice == DataDevice::CPU && data.cpuData == nullptr && data.numasData.empty())) {
            return false;
        }
        std::map <int, int> ratios;
        std::vector <int> deviceIds = ParseDeviceIds(selectedDevice, "cuda", ratios);
        if (SplitSpecialWeightToCudaTpDevices(this, weightName, data, deviceIds, ratios)) {
            return true;
        }
        if (deviceIds.size() > 1) {
            deviceIds = {deviceIds[0]};
        }
        data.ToDevice(DataDevice::CUDA, deviceIds);
        return true;
#else
        return false;
#endif
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
        } else if (modelType == "minimax_m2") {
            model = (basellm*)(new MinimaxM2Model());
        } else if (modelType == "qwen3_next") {
            model = (basellm*)(new Qwen3NextModel());
        } else if (modelType == "deepseek_v2" || modelType == "deepseek_v3" || modelType == "kimi_k2" || modelType == "deepseek_v32") {
            model = (basellm*)(new DeepSeekV2Model());
            model->model_type = modelType;
        } else if (modelType == "deepseek_v4") {
            model = (basellm*)(new DeepSeekV4Model());
            model->model_type = modelType;
        } else if (modelType == "qwen2") {
            model = (basellm*)(new Qwen2Model());
            model->model_type = "qwen2";
        } else if (modelType == "qwen3") {
            model = new Qwen3Model();
            model->model_type = "qwen3";
        } else if (modelType == "qwen3_5" || modelType == "qwen3_5_moe" || modelType == "qwen3_5_moe_text") {
            model = new Qwen3_5Model();
            model->model_type = modelType;
        } else if (modelType == "step3p5" || modelType == "step3p7") {
            model = new Step3p5Model();
            model->model_type = "step3p5";
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
        } else if (modelType == "minimax_m2") {
            model = (basellm*)(new MinimaxM2Model());
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
        } else if (modelType == "gemma4" || modelType == "gemma4_text") {
            model = new Gemma4Model();
            model->model_type = "gemma4";
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

    struct SafeTensors;

    static float FP8E8M0ToFloat(uint8_t v) {
        return std::ldexp(1.0f, (int)v - 127);
    }

    static float FP4E2M1ToFloat(uint8_t v) {
        static const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        float ret = table[v & 7];
        return (v & 8) ? -ret : ret;
    }

    static std::string FindSafeTensorScaleTensorName(const SafeTensors &safeTensors,
                                                     const std::string &tensorName);
    static std::string FindSafeTensorScale2TensorName(const SafeTensors &safeTensors,
                                                      const std::string &tensorName);
    static bool IsPackedFP4Tensor(const SafeTensors &safeTensors, const std::string &name);

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

        void CreateBufferWithScale(DataType dstType, SafeTensorItem &scale, SafeTensorItem *scale2 = nullptr) {
            AssertInFastLLM(this->shape.size() >= 2 && scale.shape.size() >= 2,
                            "CreateBufferWithScale error: shape.size() should be >= 2.");
            bool isFp8 = this->dtype == "F8_E4M3";
            bool isPackedFp4 = this->dtype == "I8" || this->dtype == "U8";
            if (!isFp8 && !isPackedFp4) {
                ErrorInFastLLM("CreateBufferWithScale error: dtype should be FP8_E4M3 or packed FP4 I8/U8");
            }
            long long n64 = 1, ns64 = 1;
            for (int i = 0; i + 1 < (int)this->shape.size(); i++) {
                n64 *= this->shape[i];
            }
            for (int i = 0; i + 1 < (int)scale.shape.size(); i++) {
                ns64 *= scale.shape[i];
            }
            AssertInFastLLM(n64 <= INT_MAX && ns64 <= INT_MAX &&
                            this->shape.back() <= INT_MAX && scale.shape.back() <= INT_MAX,
                            "CreateBufferWithScale error: shape is too large.");
            int n = (int)n64, packedM = (int)this->shape.back();
            int m = isPackedFp4 ? packedM * 2 : packedM;
            int ns = (int)ns64, ms = (int)scale.shape.back();
            int blockN = n / ns, blockM = m / ms;

            while ((blockN & -blockN) != blockN && blockN < n) {
                blockN++;
            }
            while ((blockM & -blockM) != blockM && blockM < m) {
                blockM++;
            }
            ClearBuffer();

            if (dstType == DataType::FP8_E4M3 || dstType == DataType::NVFP4 ||
                dstType == DataType::NVFP4_BLOCK_16 || dstType == DataType::NVFP4_BLOCK_16_E8M0) {
                if (dstType == DataType::FP8_E4M3 && !isFp8) {
                    ErrorInFastLLM("CreateBufferWithScale error: packed FP4 cannot be loaded as FP8_E4M3.");
                }
                if (dstType == DataType::NVFP4 && !isPackedFp4) {
                    ErrorInFastLLM("CreateBufferWithScale error: only packed FP4 I8 can be loaded as NVFP4.");
                }
                if (dstType == DataType::NVFP4 && scale.dtype != "F8_E8M0") {
                    ErrorInFastLLM("CreateBufferWithScale error: NVFP4 scale should be F8_E8M0.");
                }
                if ((dstType == DataType::NVFP4_BLOCK_16 || dstType == DataType::NVFP4_BLOCK_16_E8M0) && !isPackedFp4) {
                    ErrorInFastLLM("CreateBufferWithScale error: only packed FP4 I8/U8 can be loaded as NVFP4_BLOCK_16.");
                }
                this->blockK = blockN;
                this->blockM = blockM;
                if (dstType == DataType::NVFP4_BLOCK_16 || dstType == DataType::NVFP4_BLOCK_16_E8M0) {
                    AssertInFastLLM(blockM == 16,
                                    "CreateBufferWithScale error: NVFP4_BLOCK_16 requires blockM = 16.");
                    AssertInFastLLM(scale.bytes == (size_t)ns * ms,
                                    "CreateBufferWithScale error: NVFP4_BLOCK_16 scale bytes mismatch.");
                    if (dstType == DataType::NVFP4_BLOCK_16 && scale.dtype != "F8_E4M3") {
                        ErrorInFastLLM("CreateBufferWithScale error: NVFP4_BLOCK_16 scale should be F8_E4M3.");
                    }
                    if (dstType == DataType::NVFP4_BLOCK_16_E8M0 && scale.dtype != "F8_E8M0") {
                        ErrorInFastLLM("CreateBufferWithScale error: NVFP4_BLOCK_16_E8M0 scale should be F8_E8M0.");
                    }
                    float scale2Value = 1.0f;
                    if (scale2 != nullptr) {
                        scale2->CreateBuffer(DataType::FLOAT32);
                        AssertInFastLLM(scale2->len == 1,
                                        "CreateBufferWithScale error: NVFP4 scale2 should be scalar.");
                        scale2Value = ((float*)scale2->buffer)[0];
                    }

                    size_t blockBytes = dstType == DataType::NVFP4_BLOCK_16 ? 8 + sizeof(float) : 9;
                    size_t scaleCols = (m - 1) / 16 + 1;
                    size_t outputBytes = GetDataBytes(dstType, n, m);
                    std::vector<uint8_t> packed(this->bytes);
                    std::vector<uint8_t> scaleBytes(scale.bytes);
                    FILE *fw = fopen(this->fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
                    _fseeki64(fw, this->data_offsets[0], 0);
#else
                    fseek(fw, this->data_offsets[0], 0);
#endif
                    size_t ret = fread(packed.data(), 1, this->bytes, fw);
                    fclose(fw);
                    AssertInFastLLM(ret == this->bytes,
                                    "CreateBufferWithScale error: read NVFP4_BLOCK_16 weight failed.");
                    FILE *fs = fopen(scale.fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
                    _fseeki64(fs, scale.data_offsets[0], 0);
#else
                    fseek(fs, scale.data_offsets[0], 0);
#endif
                    ret = fread(scaleBytes.data(), 1, scale.bytes, fs);
                    fclose(fs);
                    AssertInFastLLM(ret == scale.bytes,
                                    "CreateBufferWithScale error: read NVFP4_BLOCK_16 scale failed.");

                    buffer = new uint8_t[outputBytes];
                    memset(buffer, 0, outputBytes);
                    for (int i = 0; i < n; i++) {
                        const uint8_t *srcRow = packed.data() + (size_t)i * packedM;
                        uint8_t *dstRow = buffer + (size_t)i * scaleCols * blockBytes;
                        for (size_t bj = 0; bj < scaleCols; bj++) {
                            uint8_t *dstBlock = dstRow + bj * blockBytes;
                            size_t srcOffset = bj * 8;
                            size_t copyBytes = std::min((size_t)8, (size_t)packedM - srcOffset);
                            memcpy(dstBlock, srcRow + srcOffset, copyBytes);
                            uint8_t scaleByte = scaleBytes[(size_t)i * ms + bj];
                            if (dstType == DataType::NVFP4_BLOCK_16_E8M0) {
                                dstBlock[8] = scaleByte;
                            } else {
                                float curScale = fp8e4m3tofp32.dict[scaleByte] * scale2Value;
                                memcpy(dstBlock + 8, &curScale, sizeof(float));
                            }
                        }
                    }
                    return;
                }
                size_t dataBytes = dstType == DataType::NVFP4 ? GetNVFP4WeightBytes(n, m) : (size_t)n * m;
                size_t scaleBytes = dstType == DataType::NVFP4 ? scale.bytes : 0;
                buffer = new uint8_t[dataBytes + scaleBytes];
                FILE *fi = fopen(this->fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
                _fseeki64(fi, this->data_offsets[0], 0);
#else
                fseek(fi, this->data_offsets[0], 0);
#endif
                size_t ret = fread(buffer, 1, this->bytes, fi);
                fclose(fi);
                AssertInFastLLM(ret == this->bytes && this->bytes == dataBytes,
                                "CreateBufferWithScale error: scaled data bytes mismatch.");

                if (dstType == DataType::NVFP4) {
                    AssertInFastLLM(scale.bytes == GetNVFP4ScaleBytes(n, m, blockN, blockM),
                                    "CreateBufferWithScale error: NVFP4 scale bytes mismatch.");
                    FILE *fs = fopen(scale.fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
                    _fseeki64(fs, scale.data_offsets[0], 0);
#else
                    fseek(fs, scale.data_offsets[0], 0);
#endif
                    ret = fread(buffer + dataBytes, 1, scale.bytes, fs);
                    fclose(fs);
                    AssertInFastLLM(ret == scale.bytes,
                                    "CreateBufferWithScale error: read NVFP4 scale failed.");
                } else {
                    scalesBuffer = new float[ns * ms];
                    memcpy(scalesBuffer, scale.buffer, ns * ms * sizeof(float));
                }
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
                                if (isFp8) {
                                    floatBuffer[i * m + j] = curScale * fp8e4m3tofp32.dict[ori[i * packedM + j]];
                                } else {
                                    uint8_t packed = ori[i * packedM + (j >> 1)];
                                    uint8_t fp4 = (j & 1) ? (packed >> 4) : (packed & 0xF);
                                    floatBuffer[i * m + j] = curScale * FP4E2M1ToFloat(fp4);
                                }
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
            } else if (this->dtype == "F8_E8M0") {
                if (dstType != DataType::FLOAT32) {
                    ErrorInFastLLM("SafeTensorItem.CreateBuffer: F8_E8M0 tensor " + this->tensorName + " should be loaded as float32.\n");
                }
                ClearBuffer();
                buffer = new uint8_t[(size_t)len * sizeof(float)];
                std::vector<uint8_t> ori(len);
                ret = fread(ori.data(), sizeof(uint8_t), len, fi);
                float *dst = (float*)buffer;
                for (int i = 0; i < len; i++) {
                    dst[i] = FP8E8M0ToFloat(ori[i]);
                }
                fclose(fi);
                return;
            } else if (this->dtype == "I64") {
                if (dstType != DataType::INT32 && dstType != DataType::INT32PARAM) {
                    ErrorInFastLLM("SafeTensorItem.CreateBuffer: I64 tensor " + this->tensorName + " should be loaded as int32.\n");
                }
                ClearBuffer();
                buffer = new uint8_t[(size_t)len * sizeof(int32_t)];
                std::vector<int64_t> ori(len);
                ret = fread(ori.data(), sizeof(int64_t), len, fi);
                int32_t *dst = (int32_t*)buffer;
                for (int i = 0; i < len; i++) {
                    dst[i] = (int32_t)ori[i];
                }
                fclose(fi);
                return;
            } else {
                ErrorInFastLLM("SafeTensorItem.CreateBuffer: unsupport src dtype " + this->dtype + "\n");
            }
            
            int unitSize = 4;
            if (dstType == DataType::FLOAT32) {
                unitSize = 4;
            } else if (dstType == DataType::FLOAT16 || dstType == DataType::BFLOAT16) {
                unitSize = 2;
            } else if (dstType == DataType::INT32 || dstType == DataType::INT32PARAM) {
                unitSize = 4;
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

    static bool IsPackedFP4StorageDType(const std::string &dtype) {
        return dtype == "I8" || dtype == "U8";
    }

    static bool TryGetPackedFP4DataType(const SafeTensors &safeTensors, const std::string &name,
                                        DataType &dataType) {
        auto it = safeTensors.itmeDict.find(name);
        if (it == safeTensors.itmeDict.end() || !IsPackedFP4StorageDType(it->second.dtype)) {
            return false;
        }
        std::string scaleName = FindSafeTensorScaleTensorName(safeTensors, name);
        auto scaleIt = safeTensors.itmeDict.find(scaleName);
        if (scaleIt == safeTensors.itmeDict.end()) {
            return false;
        }
        if (scaleIt->second.dtype == "F8_E8M0") {
            dataType = DataType::NVFP4;
            return true;
        }
        if (scaleIt->second.dtype == "F8_E4M3") {
            dataType = DataType::NVFP4_BLOCK_16;
            return true;
        }
        return false;
    }

    static bool IsPackedFP4Tensor(const SafeTensors &safeTensors, const std::string &name) {
        DataType dataType;
        return TryGetPackedFP4DataType(safeTensors, name, dataType);
    }

    static void ResolvePackedFP4DataType(const SafeTensors &safeTensors, const std::string &name,
                                         DataType &dataType) {
        DataType packedDataType;
        if (TryGetPackedFP4DataType(safeTensors, name, packedDataType)) {
            dataType = packedDataType;
        }
    }

    static bool IsSafeTensorQuantScaleTensorName(const SafeTensors &safeTensors,
                                                 const std::string &name) {
        auto isQuantTensor = [&](const std::string &candidate) {
            auto it = safeTensors.itmeDict.find(candidate);
            return it != safeTensors.itmeDict.end() &&
                   (it->second.dtype == "F8_E4M3" || IsPackedFP4StorageDType(it->second.dtype));
        };
        if (StringEndWith(name, "_scale_inv")) {
            return isQuantTensor(name.substr(0, name.size() - strlen("_scale_inv")));
        }
        if (StringEndWith(name, "_scale")) {
            return isQuantTensor(name.substr(0, name.size() - strlen("_scale")));
        }
        if (StringEndWith(name, ".scale_inv")) {
            return isQuantTensor(name.substr(0, name.size() - strlen(".scale_inv")) + ".weight");
        }
        if (StringEndWith(name, ".scale")) {
            return isQuantTensor(name.substr(0, name.size() - strlen(".scale")) + ".weight");
        }
        if (StringEndWith(name, ".weight_scale")) {
            return isQuantTensor(name.substr(0, name.size() - strlen(".weight_scale")) + ".weight");
        }
        if (StringEndWith(name, ".weight_scale_2")) {
            return isQuantTensor(name.substr(0, name.size() - strlen(".weight_scale_2")) + ".weight");
        }
        return false;
    }

    static std::string FindSafeTensorScaleTensorName(const SafeTensors &safeTensors,
                                                     const std::string &tensorName) {
        std::vector<std::string> candidates = {
            tensorName + "_scale_inv",
            tensorName + "_scale",
        };
        if (StringEndWith(tensorName, ".weight")) {
            std::string prefix = tensorName.substr(0, tensorName.size() - strlen(".weight"));
            candidates.push_back(prefix + ".scale_inv");
            candidates.push_back(prefix + ".scale");
            candidates.push_back(prefix + ".weight_scale");
        }
        for (auto &candidate : candidates) {
            if (safeTensors.itmeDict.find(candidate) != safeTensors.itmeDict.end()) {
                return candidate;
            }
        }
        return "";
    }

    static std::string FindSafeTensorScale2TensorName(const SafeTensors &safeTensors,
                                                      const std::string &tensorName) {
        std::vector<std::string> candidates = {
            tensorName + "_scale_2",
        };
        if (StringEndWith(tensorName, ".weight")) {
            std::string prefix = tensorName.substr(0, tensorName.size() - strlen(".weight"));
            candidates.push_back(prefix + ".weight_scale_2");
        }
        for (auto &candidate : candidates) {
            if (safeTensors.itmeDict.find(candidate) != safeTensors.itmeDict.end()) {
                return candidate;
            }
        }
        return "";
    }

    static bool IsDiskMoeWeight(basellm *model, const std::string &weightName) {
        return model != nullptr &&
               model->moeLinears.find(weightName) != model->moeLinears.end() &&
               DeviceNameMatchesType(GetMoeWeightSelectedDevice(model, weightName), "disk");
    }

    static bool GetDiskSourceDataType(const std::string &dtype, DataType &dataType) {
        if (dtype == "F32") {
            dataType = DataType::FLOAT32;
            return true;
        }
        if (dtype == "F16") {
            dataType = DataType::FLOAT16;
            return true;
        }
        if (dtype == "BF16") {
            dataType = DataType::BFLOAT16;
            return true;
        }
        if (dtype == "F8_E4M3") {
            dataType = DataType::FP8_E4M3;
            return true;
        }
        return false;
    }

    static bool IsDiskTargetDataType(DataType dataType) {
        return dataType == DataType::FLOAT32 ||
               dataType == DataType::FLOAT16 ||
               dataType == DataType::BFLOAT16 ||
               dataType == DataType::FP8_E4M3 ||
               dataType == DataType::NVFP4;
    }

    static void ResetDiskWeightMeta(Data &weight, DataType dataType) {
        std::vector<int> dims = weight.dims;
        weight.dataType = dataType;
        weight.UpdateUnitSize();
        weight.Resize(dims);
        weight.isDiskWeight = true;
        weight.diskWeightParts.clear();
        weight.weightType = WeightType::LINEAR;
        weight.expansionSize = 0;
        weight.expansionBytes = 0;
        weight.cpuData = nullptr;
        weight.dataDevice = DataDevice::CPU;
        weight.scales.clear();
        weight.mins.clear();
        weight.zeros.clear();
        weight.halfScales.clear();
        weight.perChannelsConfigs.clear();
        weight.blockK = -1;
        weight.blockM = -1;
        weight.perChannelAxis = -1;
        weight.group = -1;
        weight.groupCnt = -1;
        weight.IsRepacked = false;
    }

    static void ReadDiskTensorRange(const std::string &fileName, long long offset,
                                    uint8_t *dst, uint64_t bytes) {
        std::ifstream fin(fileName, std::ios::binary);
        if (!fin.good()) {
            ErrorInFastLLM("Disk MoE can't open weight file: " + fileName + "\n");
        }
        fin.seekg(offset, std::ios::beg);
        fin.read((char*)dst, bytes);
        if ((uint64_t)fin.gcount() != bytes) {
            ErrorInFastLLM("Disk MoE read weight metadata failed: " + fileName + "\n");
        }
    }

    static int ReadDiskMetaInt(const std::vector<uint8_t> &buffer, size_t &offset) {
        AssertInFastLLM(offset + sizeof(int) <= buffer.size(),
                        "Disk MoE fastllm metadata is truncated.\n");
        int value;
        memcpy(&value, buffer.data() + offset, sizeof(int));
        offset += sizeof(int);
        return value;
    }

    static float ReadDiskMetaFloat(const std::vector<uint8_t> &buffer, size_t &offset) {
        AssertInFastLLM(offset + sizeof(float) <= buffer.size(),
                        "Disk MoE fastllm metadata is truncated.\n");
        float value;
        memcpy(&value, buffer.data() + offset, sizeof(float));
        offset += sizeof(float);
        return value;
    }

    static void SetDiskWeightMeta(Data &weight, const SafeTensorItem &tensor, DataType targetDataType,
                                  SafeTensorItem *scaleTensor = nullptr) {
        DataType sourceDataType;
        if (IsPackedFP4StorageDType(tensor.dtype) && targetDataType == DataType::NVFP4) {
            sourceDataType = DataType::NVFP4;
        } else if (!GetDiskSourceDataType(tensor.dtype, sourceDataType)) {
            ErrorInFastLLM("Disk MoE only supports F32/F16/BF16/FP8/NVFP4 safetensors: " + weight.name + "\n");
        }
        if (!IsDiskTargetDataType(targetDataType)) {
            ErrorInFastLLM("Disk MoE unsupported target dtype: " + weight.name + "\n");
        }
        if (scaleTensor != nullptr &&
            !((sourceDataType == DataType::FP8_E4M3 && targetDataType == DataType::FP8_E4M3) ||
              (sourceDataType == DataType::NVFP4 && targetDataType == DataType::NVFP4))) {
            ErrorInFastLLM("Disk MoE only supports scaled weights for FP8/NVFP4 expert tensors: " + weight.name + "\n");
        }
        ResetDiskWeightMeta(weight, sourceDataType);

        DiskWeightPart part;
        part.fileName = tensor.fileName;
        part.fileOffset = (long long)tensor.data_offsets[0];
        part.bytes = tensor.bytes;
        part.sourceDataType = sourceDataType;
        part.dims = weight.dims;
        weight.diskWeightParts.push_back(part);

        if (scaleTensor != nullptr) {
            long long n64 = 1, ns64 = 1;
            for (int i = 0; i + 1 < (int)tensor.shape.size(); i++) {
                n64 *= tensor.shape[i];
            }
            for (int i = 0; i + 1 < (int)scaleTensor->shape.size(); i++) {
                ns64 *= scaleTensor->shape[i];
            }
            AssertInFastLLM(n64 <= INT_MAX && ns64 <= INT_MAX &&
                            tensor.shape.back() <= INT_MAX && scaleTensor->shape.back() <= INT_MAX,
                            "Disk MoE scaled tensor shape is too large: " + weight.name + "\n");
            int n = (int)n64;
            int m = (int)tensor.shape.back();
            if (targetDataType == DataType::NVFP4) {
                m *= 2;
            }
            int ns = (int)ns64, ms = (int)scaleTensor->shape.back();
            int blockK = n / ns, blockM = m / ms;
            while ((blockK & -blockK) != blockK && blockK < n) {
                blockK++;
            }
            while ((blockM & -blockM) != blockM && blockM < m) {
                blockM++;
            }
            weight.blockK = blockK;
            weight.blockM = blockM;
            if (targetDataType == DataType::NVFP4 && scaleTensor->dtype == "F8_E8M0") {
                AssertInFastLLM(scaleTensor->bytes == GetNVFP4ScaleBytes(n, m, blockK, blockM),
                                "Disk MoE NVFP4 scale tensor bytes mismatch: " + weight.name + "\n");
                DiskWeightPart scalePart;
                scalePart.fileName = scaleTensor->fileName;
                scalePart.fileOffset = (long long)scaleTensor->data_offsets[0];
                scalePart.bytes = scaleTensor->bytes;
                scalePart.sourceDataType = DataType::INT8;
                scalePart.dims = {(int)scaleTensor->bytes};
                scalePart.isScalePart = true;
                weight.diskWeightParts.push_back(scalePart);
                weight.scales.clear();
            } else {
                weight.scales.resize(ns * ms);
                memcpy(weight.scales.data(), scaleTensor->buffer, ns * ms * sizeof(float));
            }
        }
    }

    static void SetDiskFastllmWeightMeta(Data &weight, const SafeTensorItem &tensor) {
        std::vector<uint8_t> header(sizeof(int) * 5);
        ReadDiskTensorRange(tensor.fileName, (long long)tensor.data_offsets[0],
                            header.data(), header.size());
        size_t headerOffset = 0;
        int version = ReadDiskMetaInt(header, headerOffset);
        if (version != 1 && version != 2) {
            ErrorInFastLLM("Disk MoE only supports quantized fastllm expert weights: " + weight.name + "\n");
        }
        DataType dataType = (DataType)ReadDiskMetaInt(header, headerOffset);
        if (dataType == DataType::FLOAT32 || dataType == DataType::FLOAT16 ||
            dataType == DataType::BFLOAT16 || dataType == DataType::INT32 ||
            dataType == DataType::INT32PARAM) {
            ErrorInFastLLM("Disk MoE unsupported fastllm expert dtype: " + weight.name + "\n");
        }

        int fastllmGgmlType = -1;
        if (dataType == DataType::DATA_GGUF_FORMAT) {
            size_t offset = sizeof(int) * 2;
            fastllmGgmlType = ReadDiskMetaInt(header, offset);
            weight.ggmlType = fastllmGgmlType;
        }
        ResetDiskWeightMeta(weight, dataType);
        uint64_t payloadOffset = sizeof(int) * 2;
        bool compactFastllmNVFP4 = false;

        if (dataType == DataType::DATA_GGUF_FORMAT) {
            weight.ggmlType = fastllmGgmlType;
            weight.isGGUFData = true;
            weight.Resize(weight.dims);
            payloadOffset += sizeof(int);
            weight.expansionBytes = weight.GetBytes();
        } else if (dataType == DataType::FP8_E4M3 || dataType == DataType::NVFP4) {
            size_t offset = sizeof(int) * 2;
            weight.blockK = ReadDiskMetaInt(header, offset);
            weight.blockM = ReadDiskMetaInt(header, offset);
            int scaleLen = ReadDiskMetaInt(header, offset);
            AssertInFastLLM(scaleLen >= 0, "Disk MoE fastllm scale length is invalid: " + weight.name + "\n");
            if (version == 2 && dataType == DataType::NVFP4) {
                AssertInFastLLM(scaleLen == (int)GetNVFP4ScaleBytes(weight.dims[0], weight.dims[1], weight.blockK, weight.blockM),
                                "Disk MoE fastllm NVFP4 compact scale length is invalid: " + weight.name + "\n");
                weight.scales.clear();
                payloadOffset = sizeof(int) * 5;
                compactFastllmNVFP4 = true;
            } else {
                std::vector<uint8_t> meta(sizeof(int) * 5 + (uint64_t)scaleLen * sizeof(float));
                ReadDiskTensorRange(tensor.fileName, (long long)tensor.data_offsets[0],
                                    meta.data(), meta.size());
                size_t metaOffset = sizeof(int) * 5;
                weight.scales.resize(scaleLen);
                if (scaleLen > 0) {
                    memcpy(weight.scales.data(), meta.data() + metaOffset, (uint64_t)scaleLen * sizeof(float));
                }
                payloadOffset = meta.size();
            }
        } else if (dataType == DataType::INT8 || dataType == DataType::INT4 ||
                   dataType == DataType::INT4_NOZERO) {
            size_t offset = sizeof(int) * 2;
            weight.perChannelAxis = ReadDiskMetaInt(header, offset);
            int k = weight.perChannelAxis == -1 ? 1 : weight.dims[weight.perChannelAxis];
            std::vector<uint8_t> meta(sizeof(int) * 3 + (uint64_t)k * 2 * sizeof(float));
            ReadDiskTensorRange(tensor.fileName, (long long)tensor.data_offsets[0],
                                meta.data(), meta.size());
            size_t metaOffset = sizeof(int) * 3;
            weight.perChannelsConfigs.resize(k);
            weight.mins.resize(k);
            weight.scales.resize(k);
            weight.zeros.resize(k);
            int bit = dataType == DataType::INT4 ? 4 : 8;
            for (int i = 0; i < k; i++) {
                float minValue = ReadDiskMetaFloat(meta, metaOffset);
                float second = ReadDiskMetaFloat(meta, metaOffset);
                if (dataType == DataType::INT4_NOZERO) {
                    weight.perChannelsConfigs[i] = LowBitConfig(minValue, minValue + 15 * second, 4, 1);
                    weight.perChannelsConfigs[i].min = minValue;
                    weight.perChannelsConfigs[i].scale = second;
                } else {
                    weight.perChannelsConfigs[i] = LowBitConfig(minValue, second, bit, 0);
                }
                weight.mins[i] = weight.perChannelsConfigs[i].min;
                weight.scales[i] = weight.perChannelsConfigs[i].scale;
                weight.zeros[i] = weight.perChannelsConfigs[i].zeroPoint;
            }
            payloadOffset = meta.size();
        } else if (dataType == DataType::INT4_GROUP) {
            size_t offset = sizeof(int) * 2;
            weight.perChannelAxis = ReadDiskMetaInt(header, offset);
            weight.group = ReadDiskMetaInt(header, offset);
            weight.groupCnt = ReadDiskMetaInt(header, offset);
            int k = weight.perChannelAxis == -1 ? 1 : weight.dims[weight.perChannelAxis];
            std::vector<uint8_t> meta(sizeof(int) * 5 + (uint64_t)k * weight.group * 2 * sizeof(float));
            ReadDiskTensorRange(tensor.fileName, (long long)tensor.data_offsets[0],
                                meta.data(), meta.size());
            size_t metaOffset = sizeof(int) * 5;
            weight.mins.resize(k * weight.group);
            weight.scales.resize(k * weight.group);
            for (int i = 0; i < k * weight.group; i++) {
                weight.mins[i] = ReadDiskMetaFloat(meta, metaOffset);
                weight.scales[i] = ReadDiskMetaFloat(meta, metaOffset);
            }
            payloadOffset = meta.size();
        } else {
            ErrorInFastLLM("Disk MoE unsupported fastllm expert dtype: " + weight.name + "\n");
        }

        AssertInFastLLM(payloadOffset <= tensor.bytes,
                        "Disk MoE fastllm payload offset is invalid: " + weight.name + "\n");
        if (compactFastllmNVFP4) {
            uint64_t weightBytes = GetNVFP4WeightBytes(weight.dims[0], weight.dims[1]);
            uint64_t scaleBytes = GetNVFP4ScaleBytes(weight.dims[0], weight.dims[1], weight.blockK, weight.blockM);
            AssertInFastLLM(payloadOffset + weightBytes + scaleBytes == tensor.bytes,
                            "Disk MoE fastllm compact NVFP4 payload size mismatch: " + weight.name + "\n");
            DiskWeightPart weightPart;
            weightPart.fileName = tensor.fileName;
            weightPart.fileOffset = (long long)tensor.data_offsets[0] + (long long)payloadOffset;
            weightPart.bytes = weightBytes;
            weightPart.sourceDataType = dataType;
            weightPart.dims = weight.dims;
            weight.diskWeightParts.push_back(weightPart);

            DiskWeightPart scalePart;
            scalePart.fileName = tensor.fileName;
            scalePart.fileOffset = (long long)tensor.data_offsets[0] + (long long)payloadOffset + (long long)weightBytes;
            scalePart.bytes = scaleBytes;
            scalePart.sourceDataType = DataType::INT8;
            scalePart.dims = {(int)scaleBytes};
            scalePart.isScalePart = true;
            weight.diskWeightParts.push_back(scalePart);
            return;
        }
        DiskWeightPart part;
        part.fileName = tensor.fileName;
        part.fileOffset = (long long)tensor.data_offsets[0] + (long long)payloadOffset;
        part.bytes = tensor.bytes - payloadOffset;
        part.sourceDataType = dataType;
        part.dims = weight.dims;
        weight.diskWeightParts.push_back(part);
    }

    static void UpdateGGUFTensorShape(ggml_tensor *tensor, const std::vector<int> &dims) {
        tensor->dims = dims;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            tensor->ne[i] = 1;
        }
        if (dims.size() > 0) {
            tensor->ne[0] = dims.back();
        }
        if (dims.size() > 1) {
            tensor->ne[1] = dims[dims.size() - 2];
        }
        for (int i = 2; i < dims.size() && i < GGML_MAX_DIMS; i++) {
            tensor->ne[i] = dims[dims.size() - 1 - i];
        }
        const size_t typeSize = ggml_type_size(tensor->type);
        const int64_t blockSize = ggml_blck_size(tensor->type);
        tensor->nb[0] = typeSize;
        tensor->nb[1] = tensor->nb[0] * (tensor->ne[0] / blockSize);
        for (int i = 2; i < GGML_MAX_DIMS; i++) {
            tensor->nb[i] = tensor->nb[i - 1] * tensor->ne[i - 1];
        }
    }

    static void SetDiskGGUFWeightMeta(Data &weight, const ggml_tensor &tensor,
                                      const std::string &fileName, uint64_t offset) {
        if (tensor.type == ggml_type::GGML_TYPE_F32) {
            weight.dataType = DataType::FLOAT32;
        } else if (tensor.type == ggml_type::GGML_TYPE_F16) {
            weight.dataType = DataType::FLOAT16;
        } else {
            weight.dataType = DataType::DATA_GGUF_FORMAT;
            weight.isGGUFData = true;
            weight.ggmlType = tensor.type;
            if (weight.ggmlTensor == nullptr) {
                weight.ggmlTensor = (void*)(new ggml_tensor());
            }
            (*(ggml_tensor*)weight.ggmlTensor) = tensor;
        }
        weight.UpdateUnitSize();
        weight.Resize(tensor.dims);
        weight.isDiskWeight = true;
        weight.diskWeightParts.clear();
        weight.weightType = WeightType::LINEAR;
        weight.expansionSize = 0;
        weight.expansionBytes = weight.dataType == DataType::DATA_GGUF_FORMAT ? ggml_nbytes(&tensor) : 0;
        weight.cpuData = nullptr;
        weight.dataDevice = DataDevice::CPU;

        DiskWeightPart part;
        part.fileName = fileName;
        part.fileOffset = (long long)offset;
        part.bytes = ggml_nbytes(&tensor);
        part.sourceDataType = weight.dataType;
        part.dims = tensor.dims;
        weight.diskWeightParts.push_back(part);
    }

    static bool AllInputsAreDiskWeights(const std::unordered_map<std::string, Data> &weights,
                                        const std::vector<std::string> &inputs) {
        for (auto &input : inputs) {
            auto it = weights.find(input);
            if (it == weights.end() || !it->second.isDiskWeight) {
                return false;
            }
        }
        return !inputs.empty();
    }

    static bool IsCompactNVFP4Weight(const Data &data) {
        return data.dataType == DataType::NVFP4 && data.scales.empty() &&
               data.blockK > 0 && data.blockM > 0 && data.dims.size() == 2;
    }

    static void AppendCompactNVFP4Weight(Data &dst, const Data &src,
                                         uint64_t &weightOffset, uint64_t &scaleOffset) {
        AssertInFastLLM(IsCompactNVFP4Weight(dst) && IsCompactNVFP4Weight(src) &&
                        dst.dims[1] == src.dims[1] &&
                        dst.blockK == src.blockK && dst.blockM == src.blockM,
                        "Compact NVFP4 merge metadata mismatch.");
        AssertInFastLLM(src.dims[0] % src.blockK == 0,
                        "Compact NVFP4 merge requires source rows aligned to blockK.");
        uint64_t srcWeightBytes = GetNVFP4WeightBytes(src.dims[0], src.dims[1]);
        uint64_t srcScaleBytes = GetNVFP4ScaleBytes(src.dims[0], src.dims[1], src.blockK, src.blockM);
        uint64_t dstWeightBytes = GetNVFP4WeightBytes(dst.dims[0], dst.dims[1]);
        uint64_t dstScaleBytes = GetNVFP4ScaleBytes(dst.dims[0], dst.dims[1], dst.blockK, dst.blockM);
        AssertInFastLLM(weightOffset + srcWeightBytes <= dstWeightBytes &&
                        scaleOffset + srcScaleBytes <= dstScaleBytes,
                        "Compact NVFP4 merge payload overflow.");
        memcpy(dst.cpuData + weightOffset, src.cpuData, srcWeightBytes);
        memcpy(dst.cpuData + dstWeightBytes + scaleOffset, src.cpuData + srcWeightBytes, srcScaleBytes);
        weightOffset += srcWeightBytes;
        scaleOffset += srcScaleBytes;
    }

    static void MergeDiskWeightMeta(const std::unordered_map<std::string, Data> &weights,
                                    const std::vector<std::string> &inputs,
                                    Data &mergeData) {
        mergeData.isDiskWeight = true;
        mergeData.diskWeightParts.clear();
        mergeData.cpuData = nullptr;
        mergeData.expansionSize = 0;
        mergeData.expansionBytes = 0;
        mergeData.dataDevice = DataDevice::CPU;
        mergeData.weightType = WeightType::LINEAR;
        mergeData.scales.clear();
        mergeData.mins.clear();
        mergeData.zeros.clear();
        mergeData.halfScales.clear();
        mergeData.perChannelsConfigs.clear();
        uint64_t compactNVFP4ScaleOffset = 0;
        for (auto &input : inputs) {
            auto it = weights.find(input);
            if (it == weights.end()) {
                continue;
            }
            if (mergeData.blockK == -1) {
                mergeData.blockK = it->second.blockK;
            }
            if (mergeData.blockM == -1) {
                mergeData.blockM = it->second.blockM;
            }
            bool compactNVFP4 = it->second.dataType == DataType::NVFP4 &&
                                it->second.scales.empty() &&
                                it->second.blockK > 0 && it->second.blockM > 0 &&
                                it->second.dims.size() == 2;
            if (compactNVFP4 && inputs.size() > 1) {
                AssertInFastLLM(it->second.dims[0] % it->second.blockK == 0,
                                "Compact NVFP4 disk merge requires source rows aligned to blockK.");
            }
            for (auto part : it->second.diskWeightParts) {
                if (compactNVFP4 && part.isScalePart) {
                    part.scaleOffset += compactNVFP4ScaleOffset;
                }
                mergeData.diskWeightParts.push_back(part);
            }
            if (compactNVFP4) {
                compactNVFP4ScaleOffset += GetNVFP4ScaleBytes(it->second.dims[0], it->second.dims[1],
                                                              it->second.blockK, it->second.blockM);
            }
            mergeData.scales.insert(mergeData.scales.end(),
                                    it->second.scales.begin(),
                                    it->second.scales.end());
            mergeData.mins.insert(mergeData.mins.end(),
                                  it->second.mins.begin(),
                                  it->second.mins.end());
            mergeData.zeros.insert(mergeData.zeros.end(),
                                   it->second.zeros.begin(),
                                   it->second.zeros.end());
            mergeData.halfScales.insert(mergeData.halfScales.end(),
                                        it->second.halfScales.begin(),
                                        it->second.halfScales.end());
            mergeData.perChannelsConfigs.insert(mergeData.perChannelsConfigs.end(),
                                                it->second.perChannelsConfigs.begin(),
                                                it->second.perChannelsConfigs.end());
        }
        if (mergeData.dataType == DataType::DATA_GGUF_FORMAT && !inputs.empty()) {
            auto it = weights.find(inputs[0]);
            if (it != weights.end() && it->second.ggmlTensor != nullptr) {
                if (mergeData.ggmlTensor == nullptr) {
                    mergeData.ggmlTensor = (void*)(new ggml_tensor());
                }
                (*(ggml_tensor*)mergeData.ggmlTensor) = (*(ggml_tensor*)it->second.ggmlTensor);
                UpdateGGUFTensorShape((ggml_tensor*)mergeData.ggmlTensor, mergeData.dims);
                mergeData.ggmlType = ((ggml_tensor*)mergeData.ggmlTensor)->type;
                mergeData.isGGUFData = true;
                mergeData.expansionBytes = ggml_nbytes((ggml_tensor*)mergeData.ggmlTensor);
            }
        }
    }

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
            || tokenizerClass == "MiniCPMTokenizer"
            || tokenizerClass == "GemmaTokenizer" || tokenizerClass == "GemmaTokenizerFast") {
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

    static bool IsExportLinearAutoDataType(DataType dataType) {
        return dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV;
    }

    static bool IsExportFp8DataType(DataType dataType) {
        return dataType == DataType::FP8_E4M3 ||
               dataType == DataType::FP8_E4M3_BLOCK_128 ||
               dataType == DataType::FP8_E4M3_PERCHANNEL;
    }

    static void ResolveExportDataTypeForTensor(const SafeTensorItem &tensor, bool isPackedFp4,
                                               DataType linearDataType, DataType oriDataType,
                                               DataType &dataType) {
        if (dataType >= DATA_AUTO_NONE) {
            DataType autoType = dataType;
            dataType = IsExportLinearAutoDataType(autoType) ? linearDataType : oriDataType;
            if (isPackedFp4 && !IsExportLinearAutoDataType(autoType)) {
                dataType = DataType::NVFP4;
            }
        }
        if (isPackedFp4 && dataType >= DATA_AUTO_NONE) {
            dataType = DataType::NVFP4;
        }
        if (isPackedFp4 && IsExportFp8DataType(dataType)) {
            dataType = DataType::FLOAT16;
        } else if (tensor.dtype != "F8_E4M3" && dataType == DataType::FP8_E4M3) {
            dataType = DataType::FLOAT16;
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
            {"minimax_m2", "minimax_m2"}, // minimax_m2
            {"deepseek2", "deepseek_v2"}, {"deepseek_v2", "deepseek_v2"},  {"deepseek_v3", "deepseek_v2"} // deepseek_v2
        };
        if (ggufTypeToFastllmTypeDict.find(type) != ggufTypeToFastllmTypeDict.end()) {
            return ggufTypeToFastllmTypeDict[type];
        } else {
            printf("Warning: Can't convert type \"%s\", try use original type.\n", type.c_str());
            return type;
        }
    }

    extern void RegisterNumas(fastllm::Data *data, std::string weightType);

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
// printf("eos_token = %s\n", eos_token.c_str());
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
        model->OnWeightsCreated(allWeightNames);
        std::stable_sort(tensors.begin(), tensors.end(),
                         [&](const std::string &a, const std::string &b) {
                             return model->GetWeightLoadPriority(a, {}) <
                                    model->GetWeightLoadPriority(b, {});
                         });

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
                            auto *task = readGGUFTaskDict[weightName];
                            if (IsDiskMoeWeight(model, weightName) &&
                                task->replaceType == GGUFWeightReplaceRule::GGUFWeightReplaceDirect) {
                                SetDiskGGUFWeightMeta(*task->weight, task->tensor, task->fileName, task->offset);
                            } else {
                                WeightImportGGUFTensor(task->weight, &task->tensor, task->fileName,
                                                       task->offset, task->replaceType);
                            }
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
                                    std::string mergedWeightName = it.output;
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
                                        mergeData.isModelWeight = true;
                                        if (AllInputsAreDiskWeights(model->weight.weight, it.inputs)) {
                                            MergeDiskWeightMeta(model->weight.weight, it.inputs, mergeData);
                                        } else {
                                            mergeData.Allocate();
                                            uint64_t offset = 0;
                                            for (auto input : it.inputs) {
                                                memcpy(mergeData.cpuData + offset, model->weight[input].cpuData, model->weight[input].GetBytes());
                                                offset += model->weight[input].GetBytes();
                                            }
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
                                        mergeData.isModelWeight = true;
                                        mergeData.perChannelAxis = model->weight[input0].perChannelAxis;
                                        mergeData.group = model->weight[input0].group;
                                        mergeData.groupCnt = model->weight[input0].groupCnt;
                                        mergeData.blockK = model->weight[input0].blockK;
                                        mergeData.blockM = model->weight[input0].blockM;

                                        if (AllInputsAreDiskWeights(model->weight.weight, it.inputs)) {
                                            MergeDiskWeightMeta(model->weight.weight, it.inputs, mergeData);
                                        } else {
                                            mergeData.Allocate();
                                            uint64_t offset = 0;
                                            uint64_t scaleOffset = 0;
                                            bool compactNVFP4 = IsCompactNVFP4Weight(mergeData);
                                            for (auto input : it.inputs) {
                                                mergeData.perChannelsConfigs = AppendVector(mergeData.perChannelsConfigs, model->weight[input].perChannelsConfigs);
                                                mergeData.zeros = AppendVector(mergeData.zeros, model->weight[input].zeros);
                                                mergeData.scales = AppendVector(mergeData.scales, model->weight[input].scales);
                                                mergeData.mins = AppendVector(mergeData.mins, model->weight[input].mins);
                                                mergeData.halfScales = AppendVector(mergeData.halfScales, model->weight[input].halfScales);
                                                if (compactNVFP4) {
                                                    AppendCompactNVFP4Weight(mergeData, model->weight[input], offset, scaleOffset);
                                                } else {
                                                    memcpy(mergeData.cpuData + offset, model->weight[input].cpuData, model->weight[input].GetBytes());
                                                    offset += model->weight[input].GetBytes();
                                                }
                                            }
                                            mergeData.CalcWeightSum();
                                        }
#ifdef USE_TFACC
                                        try {
                                            if (model->ShouldRegisterSpecialWeightForDeviceType(mergeName, "tfacc")) {
                                                locker.lock();
                                                mergeData.weightSum.resize(1);
                                                RegisterFastllmData(&mergeData, it.type);
                                                locker.unlock();
                                            }
                                        } catch (...) {
                                        }
#endif
#if defined(USE_NUMAS)
                                        try {
                                            if (model->ShouldRegisterSpecialWeightForDeviceType(mergeName, "numa")) {
                                                mergeData.weightSum.resize(1);
                                                RegisterNumas(&mergeData, it.type);
                                            }
                                        } catch (...) {
                                        }
#endif
                                        model->MoveSpecialWeightToCudaIfNeeded(mergeName, mergeData);
                                    }

                                    locker.lock();
                                    allFinishNames.insert(mergedWeightName);
                                    model->OnWeightLoaded(mergedWeightName, allFinishNames);
                                    locker.unlock();
                                    for (auto input : it.inputs) {
                                        model->weight.weight.erase(input);
                                    }
                                }
                                locker.lock();
                            }
                            locker.unlock();
#ifdef USE_TFACC
                            try {
                                if (!needMerge && model->ShouldRegisterSpecialWeightForDeviceType(weightName, "tfacc")) {
                                    auto weightIt = model->weight.weight.find(weightName);
                                    if (weightIt != model->weight.weight.end()) {
                                        locker.lock();
                                        weightIt->second.weightSum.resize(1);
                                        RegisterFastllmData(&weightIt->second, model->specialWeights[weightName]);
                                        locker.unlock();
                                    }
                                }
                            } catch (...) {
                            }
#endif
#if defined(USE_NUMAS)
                            try {
                                if (!needMerge && model->ShouldRegisterSpecialWeightForDeviceType(weightName, "numa")) {
                                    auto weightIt = model->weight.weight.find(weightName);
                                    if (weightIt != model->weight.weight.end()) {
                                        weightIt->second.weightSum.resize(1);
                                        RegisterNumas(&weightIt->second, model->specialWeights[weightName]);
                                    }
                                }
                            } catch (...) {
                            }
#endif
                            if (!needMerge) {
                                auto weightIt = model->weight.weight.find(weightName);
                                if (weightIt != model->weight.weight.end()) {
                                    model->MoveSpecialWeightToCudaIfNeeded(weightName, weightIt->second);
                                }
                            }
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
        model->OnModelWeightsLoaded();

        printf("\n");
        fflush(stdout);

        return std::unique_ptr<fastllm::basellm> (model);
    }

    std::unique_ptr<fastllm::basellm> CreateLLMModelFromFile(const std::string &fileName) {
        std::string modelType = GetModelTypeFromFile(fileName);
        basellm *model = CreateModelWithType(modelType);
        if(modelType == "bert"){
            BertModel *bertModel = (BertModel*)model;
            bertModel->weight.tokenizer.type = Tokenizer::BERT;
            bertModel->LoadFromFile(fileName);
        }else{
            model->LoadFromFile(fileName);
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
// printf("eos_token = %s\n", eos_token.c_str());
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
            if (IsSafeTensorQuantScaleTensorName(safeTensors, tensorName)) {
                printf("Load %d \r", (++cur) * 100 / (int)safeTensors.itmeDict.size());
                fflush(stdout);
                continue;
            }
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
                    ResolvePackedFP4DataType(safeTensors, tensorName, dataType);
                }

                if (dataType >= DATA_AUTO_NONE) {
                    // AUTO类型
                    dataType = (dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) ? linearDataType : oriDataType;
                    
                    // 如果原始权重不是FP8_E4M3格式，目前不做转换
                    if (tensor.dtype != "F8_E4M3" && dataType == DataType::FP8_E4M3) {
                        dataType = DataType::FLOAT16;
                    }
                    ResolvePackedFP4DataType(safeTensors, tensorName, dataType);
                }
                ResolvePackedFP4DataType(safeTensors, tensorName, dataType);
                if (tensor.dtype == "I64") {
                    dataType = DataType::INT32PARAM;
                }
                if (it.second == DATA_AUTO_CONV) {
                    std::vector <int> realShape = tensor.intShape;
                    std::swap(realShape[0], realShape[1]);
                    model->weight.AddEmptyWeight(weightName, realShape, dataType);
                } else if (IsPackedFP4Tensor(safeTensors, tensorName)) {
                    std::vector<int> realShape = tensor.intShape;
                    realShape[1] *= 2;
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
        model->OnWeightsCreated(allWeightNames);
        std::stable_sort(tensors.begin(), tensors.end(),
                         [&](const std::string &a, const std::string &b) {
                             return model->GetWeightLoadPriority(a, tensorMap[a]) <
                                    model->GetWeightLoadPriority(b, tensorMap[b]);
                         });

        // 4.2 读取
        std::vector <std::thread*> threads;
        int threadNum = std::min(16, std::max(4, (int)GetAlivePool()->threads.size()));
        std::mutex locker;
        int cnt = 0;
        int loadProgressTotal = std::max(1, (int)tensorMap.size());
        auto printLoadingProgress = [&]() {
            locker.lock();
            int progress = std::min(100, (++cnt) * 100 / loadProgressTotal);
            printf("Loading %d \r", progress);
            fflush(stdout);
            locker.unlock();
        };

        std::vector <std::string> serialTensors, parallelTensors;
        serialTensors.reserve(tensors.size());
        parallelTensors.reserve(tensors.size());
        for (auto &tensorName : tensors) {
            if (model->ShouldLoadWeightSeriallyBeforeOthers(tensorName, tensorMap[tensorName])) {
                serialTensors.push_back(tensorName);
            } else {
                parallelTensors.push_back(tensorName);
            }
        }
        tensors.swap(parallelTensors);
        loadProgressTotal = std::max(1, (int)serialTensors.size() + (int)tensors.size());
        totalBytes = 0;
        for (auto &tensorName : tensors) {
            totalBytes += safeTensors.itmeDict[tensorName].bytes;
        }

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

        std::vector <std::string> *activeTensors = &tensors;
        auto buildSafeTensorParts = [&](const std::vector<std::string> &names,
                                        int rangeStart, int rangeEnd, int partNum) {
            std::vector <std::pair <int, int> > ret;
            partNum = std::max(1, partNum);
            long long rangeBytes = 0;
            for (int i = rangeStart; i < rangeEnd; i++) {
                rangeBytes += safeTensors.itmeDict[names[i]].bytes;
            }
            int curStart = rangeStart;
            for (int i = 0; i < partNum; i++) {
                int cur = curStart;
                long long now = 0;
                while (true) {
                    if (now * partNum >= rangeBytes || curStart >= rangeEnd) {
                        break;
                    }
                    now += safeTensors.itmeDict[names[curStart]].bytes;
                    curStart++;
                }
                ret.push_back(std::make_pair(cur, curStart));
            }
            ret.back().second = rangeEnd;
            return ret;
        };
        auto loadSafeTensorRange = [&](int st, int end) {
                    for (int i = st; i < end; i++) {
                        auto &tensorName = (*activeTensors)[i];
                        if (IsSafeTensorQuantScaleTensorName(safeTensors, tensorName) ||
                            (isAwqModel && (StringEndWith(tensorName, ".scales") || StringEndWith(tensorName, ".qzeros")))) {
                            printLoadingProgress();
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
                            if (tensor.dtype != "F8_E4M3" && dataType == DataType::FP8_E4M3) {
                                dataType = DataType::FLOAT16;
                            }
                            ResolvePackedFP4DataType(safeTensors, tensorName, dataType);
                            if (tensor.dtype == "I64") {
                                dataType = DataType::INT32PARAM;
                                oriDataType = DataType::INT32PARAM;
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
                                || dataType == DataType::INT2_GROUP
                                || dataType == DataType::DATA_GGUF_FORMAT)) {
                                oriDataType = DataType::FLOAT32;
                                scaleTensorName = FindSafeTensorScaleTensorName(safeTensors, tensorName);
                            }
                            if (tensor.dtype == "F8_E4M3" && 
                                (dataType == FP8_E4M3)) {
                                oriDataType = DataType::FP8_E4M3;
                                scaleTensorName = FindSafeTensorScaleTensorName(safeTensors, tensorName);
                            }
                            DataType packedFp4DataType;
                            if (TryGetPackedFP4DataType(safeTensors, tensorName, packedFp4DataType)) {
                                oriDataType = packedFp4DataType;
                                scaleTensorName = FindSafeTensorScaleTensorName(safeTensors, tensorName);
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

                            bool diskLazyWeight = IsDiskMoeWeight(model, weightName);
                            if (diskLazyWeight) {
                                if (isAwqModel || loraDicts.find(weightName) != loraDicts.end()) {
                                    ErrorInFastLLM("Disk MoE does not support AWQ/lora expert weight yet: " + weightName + "\n");
                                }
                                if (tensor.dtype == "fastllm") {
                                    SetDiskFastllmWeightMeta(model->weight[weightName], tensor);
                                } else {
                                    SafeTensorItem *scaleTensor = nullptr;
                                    DataType diskDataType = dataType;
                                    if (scaleTensorName != "") {
                                        if (tensor.dtype == "F8_E4M3") {
                                            diskDataType = DataType::FP8_E4M3;
                                        } else if (TryGetPackedFP4DataType(safeTensors, tensorName, packedFp4DataType)) {
                                            diskDataType = packedFp4DataType;
                                        } else {
                                            ErrorInFastLLM("Disk MoE only supports scaled safetensors for FP8/NVFP4 expert weight: " + weightName + "\n");
                                        }
                                        scaleTensor = &safeTensors.itmeDict[scaleTensorName];
                                        AssertInFastLLM(scaleTensor->dtype == "F32" || scaleTensor->dtype == "BF16" ||
                                                        scaleTensor->dtype == "F8_E8M0" || scaleTensor->dtype == "F8_E4M3",
                                                        "Tensor scale error: scale's dtype should be F32, BF16, F8_E8M0 or F8_E4M3.");
                                        if (!((diskDataType == DataType::NVFP4 && scaleTensor->dtype == "F8_E8M0") ||
                                              (diskDataType == DataType::NVFP4_BLOCK_16 && scaleTensor->dtype == "F8_E4M3"))) {
                                            scaleTensor->CreateBuffer(DataType::FLOAT32);
                                        }
                                    }
                                    SetDiskWeightMeta(model->weight[weightName], tensor, diskDataType, scaleTensor);
                                    if (scaleTensor != nullptr) {
                                        scaleTensor->ClearBuffer();
                                    }
                                }
                            } else {
                                if (scaleTensorName == "") {
                                    tensor.CreateBuffer(oriDataType);
                                } else if(!isAwqModel) {
                                    auto &scaleTensor = safeTensors.itmeDict[scaleTensorName];
                                    AssertInFastLLM(scaleTensor.dtype == "F32" || scaleTensor.dtype == "BF16" ||
                                                    scaleTensor.dtype == "F8_E8M0" || scaleTensor.dtype == "F8_E4M3"
                                        , "Tensor scale error: scale's dtype should be F32, BF16, F8_E8M0 or F8_E4M3.");
                                    bool keepScalePacked = (oriDataType == DataType::NVFP4 && scaleTensor.dtype == "F8_E8M0") ||
                                                           (oriDataType == DataType::NVFP4_BLOCK_16 && scaleTensor.dtype == "F8_E4M3");
                                    if (!keepScalePacked) {
                                        scaleTensor.CreateBuffer(DataType::FLOAT32);
                                    }
                                    SafeTensorItem *scale2Tensor = nullptr;
                                    std::string scale2TensorName = FindSafeTensorScale2TensorName(safeTensors, tensorName);
                                    if (oriDataType == DataType::NVFP4_BLOCK_16 && scale2TensorName != "") {
                                        scale2Tensor = &safeTensors.itmeDict[scale2TensorName];
                                    }
                                    tensor.CreateBufferWithScale(oriDataType, scaleTensor, scale2Tensor);
                                    if (scale2Tensor != nullptr) {
                                        scale2Tensor->ClearBuffer();
                                    }
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
                            }
                            tensor.ClearBuffer();

                            locker.lock();
                            allFinishNames.insert(weightName);
                            model->OnWeightLoaded(weightName, allFinishNames);
                            if (model->IsWeightConsumedAfterLoad(weightName)) {
                                locker.unlock();
                                continue;
                            }
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
                                    std::string mergedWeightName = it.output;
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
                                        mergeData.isModelWeight = true;
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
                                        mergeData.isModelWeight = true;
                                        mergeData.perChannelAxis = model->weight[input0].perChannelAxis;
                                        mergeData.group = model->weight[input0].group;
                                        mergeData.groupCnt = model->weight[input0].groupCnt;
                                        mergeData.blockK = model->weight[input0].blockK;
                                        mergeData.blockM = model->weight[input0].blockM;

                                        if (AllInputsAreDiskWeights(model->weight.weight, it.inputs)) {
                                            MergeDiskWeightMeta(model->weight.weight, it.inputs, mergeData);
                                        } else {
                                            mergeData.Allocate();
                                            uint64_t offset = 0;
                                            uint64_t scaleOffset = 0;
                                            bool compactNVFP4 = IsCompactNVFP4Weight(mergeData);
                                            for (auto input : it.inputs) {
                                                mergeData.perChannelsConfigs = AppendVector(mergeData.perChannelsConfigs, model->weight[input].perChannelsConfigs);
                                                mergeData.zeros = AppendVector(mergeData.zeros, model->weight[input].zeros);
                                                mergeData.scales = AppendVector(mergeData.scales, model->weight[input].scales);
                                                mergeData.mins = AppendVector(mergeData.mins, model->weight[input].mins);
                                                mergeData.halfScales = AppendVector(mergeData.halfScales, model->weight[input].halfScales);
                                                if (compactNVFP4) {
                                                    AppendCompactNVFP4Weight(mergeData, model->weight[input], offset, scaleOffset);
                                                } else {
                                                    memcpy(mergeData.cpuData + offset, model->weight[input].cpuData, model->weight[input].GetBytes());
                                                    offset += model->weight[input].GetBytes();
                                                }
                                            }

                                            mergeData.CalcWeightSum();
#ifdef USE_TFACC
                                            try {
                                                if (model->ShouldRegisterSpecialWeightForDeviceType(mergeName, "tfacc")) {
                                                    locker.lock();
                                                    mergeData.weightSum.resize(1);
                                                    RegisterFastllmData(&mergeData, it.type);
                                                    locker.unlock();
                                                }
                                            } catch (...) {
                                            }
#endif
#if defined(USE_NUMAS)
                                            try {
                                                if (model->ShouldRegisterSpecialWeightForDeviceType(mergeName, "numa")) {
                                                    mergeData.weightSum.resize(1);
                                                    RegisterNumas(&mergeData, it.type);
                                                }
                                            } catch (...) {
                                            }
#endif
                                            model->MoveSpecialWeightToCudaIfNeeded(mergeName, mergeData);
                                        }
                                    }

                                    locker.lock();
                                    allFinishNames.insert(mergedWeightName);
                                    model->OnWeightLoaded(mergedWeightName, allFinishNames);
                                    locker.unlock();
                                    for (auto input : it.inputs) {
                                        model->weight.weight.erase(input);
                                    }
                                }
                                locker.lock();
                            }
                            locker.unlock();
#ifdef USE_TFACC
                            try {
                                if (!needMerge && model->ShouldRegisterSpecialWeightForDeviceType(weightName, "tfacc")) {
                                    auto weightIt = model->weight.weight.find(weightName);
                                    if (weightIt != model->weight.weight.end()) {
                                        locker.lock();
                                        weightIt->second.weightSum.resize(1);
                                        RegisterFastllmData(&weightIt->second, model->specialWeights[weightName]);
                                        locker.unlock();
                                    }
                                }
                            } catch (...) {
                            }
#endif
#if defined(USE_NUMAS)
                            try {
                                if (!needMerge && model->ShouldRegisterSpecialWeightForDeviceType(weightName, "numa")) {
                                    auto weightIt = model->weight.weight.find(weightName);
                                    if (weightIt != model->weight.weight.end()) {
                                        weightIt->second.weightSum.resize(1);
                                        RegisterNumas(&weightIt->second, model->specialWeights[weightName]);
                                    }
                                }
                            } catch (...) {
                            }
#endif
                            if (!needMerge) {
                                auto weightIt = model->weight.weight.find(weightName);
                                if (weightIt != model->weight.weight.end()) {
                                    model->MoveSpecialWeightToCudaIfNeeded(weightName, weightIt->second);
                                }
                            }
                        }

                        printLoadingProgress();
                    }
        };

        activeTensors = &serialTensors;
        int serialStart = 0;
        while (serialStart < (int)serialTensors.size()) {
            int priority = model->GetWeightLoadPriority(serialTensors[serialStart],
                                                        tensorMap[serialTensors[serialStart]]);
            int serialEnd = serialStart + 1;
            while (serialEnd < (int)serialTensors.size() &&
                   model->GetWeightLoadPriority(serialTensors[serialEnd],
                                                tensorMap[serialTensors[serialEnd]]) == priority) {
                serialEnd++;
            }
            std::set<std::string> groupWeightNames;
            for (int i = serialStart; i < serialEnd; i++) {
                for (auto &mapped : tensorMap[serialTensors[i]]) {
                    groupWeightNames.insert(mapped.first);
                }
            }
            model->OnWeightLoadGroupStarted(groupWeightNames);
            int groupThreadNum = std::min(threadNum, std::max(1, serialEnd - serialStart));
            if (groupThreadNum <= 1) {
                loadSafeTensorRange(serialStart, serialEnd);
            } else {
                std::vector <std::thread*> groupThreads;
                auto groupParts = buildSafeTensorParts(serialTensors, serialStart, serialEnd, groupThreadNum);
                for (auto &part : groupParts) {
                    if (part.first < part.second) {
                        groupThreads.push_back(new std::thread(loadSafeTensorRange, part.first, part.second));
                    }
                }
                for (int i = 0; i < groupThreads.size(); i++) {
                    groupThreads[i]->join();
                    delete groupThreads[i];
                }
            }
            model->OnWeightLoadGroupFinished();
            serialStart = serialEnd;
        }
        activeTensors = &tensors;

        for (int i = 0; i < threadNum; i++) {
            threads.push_back(new std::thread(loadSafeTensorRange, parts[i].first, parts[i].second));
        }
        for (int i = 0; i < threads.size(); i++) {
            threads[i]->join();
            delete threads[i];
        }
        model->OnWeightLoadGroupFinished();
        model->OnModelWeightsLoaded();

        printf("\n");
        fflush(stdout);

        delete loraTensors;

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
                if (IsSafeTensorQuantScaleTensorName(safeTensors, tensor.tensorName)) {
                    continue;
                }
                auto oriDataType = DataType::FLOAT32;
                auto dataType = tensorMap[tensor.tensorName][0].second;
                auto weightName = tensor.tensorName;
                bool isPackedFp4 = IsPackedFP4Tensor(safeTensors, tensor.tensorName);
                int ggmlType = -1;

                if ((dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) && dtypeRules.size() > 0) {
                    int groupCnt = -1;
                    ParseDataType(weightName, dtypeRules, dataType, groupCnt, ggmlType);
                }
                ResolveExportDataTypeForTensor(tensor, isPackedFp4, linearDataType, oriDataType, dataType);
                if (tensor.dtype == "I64") {
                    dataType = DataType::INT32PARAM;
                }
                if (dataType== DATA_AUTO_CONV) {
                    std::vector <int> realShape = tensor.intShape;
                    std::swap(realShape[0], realShape[1]);
                    weights[weightName] = Data(dataType, realShape);
                } else if (isPackedFp4) {
                    std::vector<int> realShape = tensor.intShape;
                    realShape[1] *= 2;
                    if (dataType == DATA_GGUF_FORMAT) {
                        weights[weightName] = Data(dataType, ggmlType, realShape);
                    } else {
                        weights[weightName] = Data(dataType, realShape);
                    }
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
                            if (IsSafeTensorQuantScaleTensorName(safeTensors, tensor.tensorName)) {
                                continue;
                            }
                            std::string scaleTensorName = "";
                            std::string weightName = tensor.tensorName;

                            auto dataType = tensorMap[tensor.tensorName][0].second;
                            auto oriDataType = DataType::FLOAT32;
                            int ggmlType = -1;
                            int curGroupCnt = model->moeLinears.find(weightName) != model->moeLinears.end() ? moeGroupCnt : groupCnt;
                            bool isPackedFp4 = IsPackedFP4Tensor(safeTensors, tensor.tensorName);
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

                            ResolveExportDataTypeForTensor(tensor, isPackedFp4, linearDataType, oriDataType, dataType);
                            if (tensor.dtype == "I64") {
                                dataType = DataType::INT32PARAM;
                                oriDataType = DataType::INT32PARAM;
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
                                scaleTensorName = FindSafeTensorScaleTensorName(safeTensors, tensor.tensorName);
                            }
                            if (tensor.dtype == "F8_E4M3" && 
                                (dataType == FP8_E4M3)) {
                                oriDataType = DataType::FP8_E4M3;
                                scaleTensorName = FindSafeTensorScaleTensorName(safeTensors, tensor.tensorName);
                            }
                            if (isPackedFp4) {
                                oriDataType = dataType == DataType::NVFP4 ? DataType::NVFP4 : DataType::FLOAT32;
                                scaleTensorName = FindSafeTensorScaleTensorName(safeTensors, tensor.tensorName);
                            }

                            if (scaleTensorName == "") {
                                tensor.CreateBuffer(oriDataType);
                            } else {
                                auto &scaleTensor = safeTensors.itmeDict[scaleTensorName];
                                AssertInFastLLM(scaleTensor.dtype == "F32" || scaleTensor.dtype == "BF16" || scaleTensor.dtype == "F8_E8M0"
                                    , "Tensor scale error: scale's dtype should be F32, BF16 or F8_E8M0.");
                                if (!(oriDataType == DataType::NVFP4 && scaleTensor.dtype == "F8_E8M0")) {
                                    scaleTensor.CreateBuffer(DataType::FLOAT32);
                                }
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
                if (IsSafeTensorQuantScaleTensorName(safeTensors, weightName)) {
                    continue;
                }
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
                if (IsSafeTensorQuantScaleTensorName(safeTensors, weightName)) {
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
