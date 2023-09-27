//
// Created by huangyuyang on 5/11/23.
//

#ifndef FASTLLM_GLM_H
#define FASTLLM_GLM_H

#include "basellm.h"
#include "cmath"

#include <iostream>

namespace fastllm {
    class GLMModel: public basellm {
	public:
        GLMModel (); // 构造函数

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

        // 根据输入的tokens生成LLM推理的输入
        virtual void FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                   const std::map <std::string, int> &params,
                                   Data &inputIds, Data &attentionMask, Data &positionIds);

        virtual void InitParams();
		virtual void WarmUp(); // 预热

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

    private:

        float scale_attn_1;

        static constexpr int eot_token_id = 50000;//<|endoftext|>
        static constexpr int cls_token_id = 50002;//[CLS]
        static constexpr int mask_token_id = 50003;//[MASK]
        static constexpr int smask_token_id = 50008;//[sMASK]
        static constexpr int gmask_token_id = 50009;//[gMASK]
    };
}

#endif //FASTLLM_GLM_H
