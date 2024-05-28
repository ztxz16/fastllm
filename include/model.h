//
// Created by huangyuyang on 6/20/23.
//

#ifndef FASTLLM_MODEL_H
#define FASTLLM_MODEL_H

#include "basellm.h"
#include "bert.h"

namespace fastllm {
    std::unique_ptr<BertModel> CreateEmbeddingModelFromFile(const std::string &fileName);

    std::unique_ptr<basellm> CreateLLMModelFromFile(const std::string &fileName);

    std::unique_ptr<basellm> CreateEmptyLLMModel(const std::string &modelType);

    std::unique_ptr<basellm> CreateLLMModelFromHF(const std::string &modelPath, 
                                                    DataType linearDataType, 
                                                    int groupCnt = -1);
}

#endif //FASTLLM_MODEL_H
