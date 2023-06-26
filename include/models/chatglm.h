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
                const Data &penaltyFactor,
                std::vector <std::pair <Data, Data> > &pastKeyValues);

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                const Data &penaltyFactor,
                std::vector <std::pair <Data, Data> > &pastKeyValues);

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues);

		virtual std::string Response(const std::string& input, RuntimeResult retCb); // 根据给出的内容回复

        virtual void ResponseBatch(const std::vector <std::string> &inputs,
                                   std::vector <std::string> &outputs,
                                   RuntimeResultBatch retCb);

        virtual int LaunchResponseTokens(const std::vector <int> &inputTokens); // 启动一个response任务，返回分配的handleId

        virtual std::pair <bool, std::vector <int> > FetchResponseTokens(int handelId); // 获取指定handle的输出, bool代表这个handle是否已经结束（或者不存在）

		virtual void WarmUp(); // 预热

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history
    private:
		virtual void CausalMask(Data &data, int start) {}; // 因果mask？
    };
}

#endif //FASTLLM_CHATGLM_H
