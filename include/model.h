//
// Created by huangyuyang on 6/20/23.
//

#ifndef FASTLLM_MODEL_H
#define FASTLLM_MODEL_H

#include "basellm.h"

namespace fastllm {
    std::unique_ptr<basellm> CreateLLMModelFromFile(const std::string &fileName);
}

#endif //FASTLLM_MODEL_H
