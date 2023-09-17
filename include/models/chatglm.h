//
// Created by huangyuyang on 5/11/23.
//

#ifndef FASTLLM_CHATGLM_H
#define FASTLLM_CHATGLM_H

#include "basellm.h"
#include "cmath"

#include <iostream>

namespace fastllm {
    class ChatGLMModel: public basellm {
	public:
        ChatGLMModel (); // 构造函数

        virtual void InitParams(); // 初始化参数信息

        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr);

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *retLogits = nullptr);

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr);

        // 根据输入的tokens生成LLM推理的输入
        virtual void FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                   const std::map <std::string, int> &params,
                                   Data &inputIds, Data &attentionMask, Data &positionIds);

        // 根据输入的tokens生成LLM推理的输入
        virtual void FillLLMInputsBatch(std::vector <std::vector <float> > &inputTokens,
                                        const std::vector <std::map <std::string, int> > &params,
                                        Data &inputIds, Data &attentionMask, Data &positionIds);

        virtual void WarmUp(); // 预热

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

        int GetVersion();

        void UpdateSinCos(float rope);
    private:
        virtual void CausalMask(Data &data, int start) {}; // 因果mask？

        int gmask_token_id;

        float rope = 1.0f;
    };
}

#endif //FASTLLM_CHATGLM_H
