//
// Created by huangyuyang on 5/12/23.
//

#ifndef TEST_MOSS_H
#define TEST_MOSS_H

#include "basellm.h"
#include "cmath"

namespace fastllm {
    class MOSSModel: public basellm {
	public:
        MOSSModel(); // 构造函数

        // 推理
		virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager());

		virtual std::string Response(const std::string &input, RuntimeResult retCb,
                                     const GenerationConfig &generationConfig = GenerationConfig()); // 根据给出的内容回复

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

        virtual void FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                         const std::map <std::string, int> &params,
                                         Data &inputIds, Data &attentionMask, Data &positionIds);

        virtual void WarmUp();
    private:
		virtual void RotatePosition2D(Data &data, const Data &positionIds); // 二维位置编码

		virtual void CausalMask(Data &data, int start); // 因果mask？
    };
}

#endif //TEST_MOSS_H
