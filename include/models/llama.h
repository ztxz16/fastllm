//
// Created by huangyuyang on 6/1/23.
//

#ifndef FASTLLM_LLAMA_H
#define FASTLLM_LLAMA_H

#include "basellm.h"
#include "cmath"

#include <iostream>

namespace fastllm {

    enum RoPEType { // 位置编码外推类型
        BASE = 0,
        LINEAR_SCALE = 1,
        STATIC_NTK = 2,
        DYMAMIC_NTK = 3
    };

    class LlamaModel: public basellm {
    public:
        LlamaModel (); // 构造函数

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
                std::vector <std::vector <float>*> *logits = nullptr);

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

        std::pair<std::vector<float>, std::vector<float>> UpdateRotaryPosEmb(float base, float factor, int seqLen = 0); // 更新位置编码

    protected:
        RoPEType rope_type = RoPEType::BASE;

        float rope_base = 10000.f;

        float rope_factor = 1.f;

        int num_key_value_heads = num_attention_heads;

        float rms_norm_eps = 1e-6;
    };
}

#endif //FASTLLM_LLAMA_H
