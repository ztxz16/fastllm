//
// Created by huangyuyang on 6/20/23.
//

#ifndef FASTLLM_MODEL_H
#define FASTLLM_MODEL_H

#include "basellm.h"
#include "bert.h"
#include "xlmroberta.h"

namespace fastllm {
    std::unique_ptr<BertModel> CreateEmbeddingModelFromFile(const std::string &fileName);

    std::unique_ptr<basellm> CreateLLMModelFromGGUF(const std::string &modelPath);

    std::unique_ptr<basellm> CreateLLMModelFromGGUFFile(const std::string &fileName, const std::string &originalPath);

    std::unique_ptr<basellm> CreateLLMModelFromFile(const std::string &fileName);

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
