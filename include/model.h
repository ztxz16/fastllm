//
// Created by huangyuyang on 6/20/23.
//

#ifndef FASTLLM_MODEL_H
#define FASTLLM_MODEL_H

#include "basellm.h"
#include "bert.h"
#include "json11.hpp"
#include "xlmroberta.h"

namespace fastllm {
    std::unique_ptr<BertModel> CreateEmbeddingModelFromFile(const std::string &fileName);

    std::unique_ptr<basellm> CreateLLMModelFromGGUF(const std::string &modelPath);

    std::unique_ptr<basellm> CreateLLMModelFromGGUFFile(
        const std::string &fileName, const std::string &originalPath,
        const std::string &multimodalProjectorPath = "");

    std::string ConvertGGUFTypeToFastllmType(const std::string &type);

    void ApplyDeepSeekV4GGUFMetadata(basellm *model, const json11::Json &params,
                                     const std::string &arch);

    // DeepSeek-V4 GGUF backbone 只含 43 层 target 权重；DSpark 的 mtp.*
    // 权重需要单独从官方 safetensors 分片（46–48）叠加。下面的结构描述
    // 一次通过校验的 DSpark 分片计划。
    struct DeepSeekV4DSparkShardPlan {
        std::set <std::string> shardFiles;
        std::vector <std::string> tensorNames; // 全部以 "mtp." 开头，升序
    };

    // 解析 dsparkPath 下的 safetensors index（或单个 model.safetensors），
    // 只接受 mtp.* 张量；任何 backbone 命名、缺失分片或空计划都直接失败。
    DeepSeekV4DSparkShardPlan PlanDeepSeekV4DSparkShards(
        const std::string &dsparkPath);

    // 顺序加载计划中的全部 mtp.* 张量到 model->weight（含 disk-lazy 元数据
    // 与 .weight/.weight_scale 配对）。不触发 merge；生产 GGUF 路径在并行
    // 装载循环内逐张量调用同一导入核心，并由通用 merge 机制收尾。
    void ImportDeepSeekV4DSparkWeights(basellm *model,
                                       const DeepSeekV4DSparkShardPlan &plan);

    std::unique_ptr<basellm> CreateLLMModelFromFile(
        const std::string &fileName,
        const std::string &multimodalProjectorPath = "");

    std::unique_ptr<basellm> CreateEmptyLLMModel(const std::string &modelType);

    std::unique_ptr<basellm> CreateLLMModelFromHF(const std::string &modelPath, 
                                                    DataType linearDataType, 
                                                    int groupCnt = -1,
                                                    bool skipTokenizer = false,
                                                    const std::string &modelConfig = "",
                                                    const std::string &loraPath = "",
                                                    bool weightOnly = false, 
                                                    bool useMoeDataType = false, 
                                                    DataType moeDataType = DataType::FLOAT32, 
                                                    int moeGroupCnt = -1, const std::string &dtypeConfigString = "");
    
    void ExportLLMModelFromHF(const std::string &modelPath, 
                            DataType linearDataType, 
                            int groupCnt, 
                            const std::string &exportPath, 
                            const std::string &modelConfig = "",
                            const std::string &loraPath = "", 
                            bool useMoeDataType = false, 
                            DataType moeDataType = DataType::FLOAT32, 
                            int moeGroupCnt = -1, const std::string &dtypeConfigString = "");
    
    std::unique_ptr<basellm> CreateLLMTokenizerFromHF(const std::string &modelPath);

    struct ModelMetaInfo {
        DataType autoAtype = fastllm::DataType::FLOAT32; // 当atype设置为auto时采用的atype
        bool autoSaveHistoryChat = false; // 默认是否开启前缀缓存（一般moe模型会开启）
        bool supportFP16Atype = false; // 是否支持atype设置为FP16
        bool isMOE = false; // 是否是MOE模型
        bool isMLP = false; // 是否是MLP模型
    };

    ModelMetaInfo *GetModelMetaInfoByType(const std::string &modelType);
    ModelMetaInfo *GetModelMetaInfoByStruct(const std::string &modelStruct);

}

#endif //FASTLLM_MODEL_H
