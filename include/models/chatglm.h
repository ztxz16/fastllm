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

        // 推理
		virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager());

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager());

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens = LastTokensManager());

		virtual std::string Response(const std::string& input,
                                     RuntimeResult retCb,
                                     const GenerationConfig &generationConfig = GenerationConfig()); // 根据给出的内容回复

        virtual void ResponseBatch(const std::vector <std::string> &inputs,
                                   std::vector <std::string> &outputs,
                                   RuntimeResultBatch retCb,
                                   const GenerationConfig &generationConfig = GenerationConfig());

        virtual int LaunchResponseTokens(const std::vector <int> &inputTokens,
                                         const GenerationConfig &generationConfig = GenerationConfig()); // 启动一个response任务，返回分配的handleId

        virtual int FetchResponseTokens(int handelId); // 获取指定handle的输出, -1代表输出结束了

		virtual void WarmUp(); // 预热

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

        int GetVersion();

        void UpdateSinCos(float rope);
    private:
		virtual void CausalMask(Data &data, int start) {}; // 因果mask？

        float rope = 1.0f;
    };
}

#endif //FASTLLM_CHATGLM_H
