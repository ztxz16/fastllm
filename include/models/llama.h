//
// Created by huangyuyang on 6/1/23.
//

#ifndef FASTLLM_LLAMA_H
#define FASTLLM_LLAMA_H

#include "basellm.h"
#include "cmath"

#include <iostream>

namespace fastllm {
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

        /**
         * 通过新调度器执行request-major的Llama批量前向。
         *
         * 参数布局与新版ForwardBatch完全一致；本适配器不改变Attention后端、
         * KV Cache布局、同步边界或采样语义。Paged模式使用该入口接入Page
         * Trie的分配、复用和回收流程，Continuous模式仍由旧调度器执行。
         *
         * @param batch             请求数量。
         * @param inputIds          合并后的输入Token张量。
         * @param attentionMask     每个请求独立的Attention Mask。
         * @param positionIds       每个请求独立的位置编码索引。
         * @param seqLens           每个请求本轮输入Token数量。
         * @param pastKeyValues     request-major的K/V Cache描述符。
         * @param generationConfigs 每个请求独立的生成配置。
         * @param lastTokens        每个请求的历史Token状态。
         * @param logits            可选的逐请求Logits输出。
         * @return 每个请求本轮生成的Token ID。
         */
        virtual std::vector<int> ForwardV2(
                int batch,
                const Data &inputIds,
                const std::vector<Data*> &attentionMask,
                const std::vector<Data*> &positionIds,
                const std::vector<int> &seqLens,
                std::vector<std::pair<Data*, Data*>> &pastKeyValues,
                const std::vector<GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector<std::vector<float>*> *logits = nullptr) override;
        
        // 是否需要生成AttentionMask
        virtual bool NeedAttentionMask(int qlen, int klen);

        // 根据输入的tokens生成LLM推理的输入
        virtual void FillLLMInputsBatch(std::vector <std::vector <float> > &inputTokens,
                                        const std::vector <std::map <std::string, int> > &params,
                                        Data &inputIds, Data &attentionMask, Data &positionIds);

        virtual void WarmUp(); // 预热

        /** 标准Llama支持GPU Continuous和Paged KV Cache。 */
        virtual ModelAttentionCapabilities GetModelAttentionCapabilities() const override;

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

        std::pair<std::vector<float>, std::vector<float>> UpdateRotaryPosEmb(float base, float factor, int seqLen = 0); // 更新位置编码

    protected:
        /** 将单请求旧接口适配为request-major的新ForwardBatch接口。 */
        std::vector<int> ForwardSingleWithRequestCaches(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<std::vector<float>*> *logits);

        RoPEType rope_type = RoPEType::BASE;

        float rope_base = 10000.f;

        float rope_factor = 1.f;

        int num_key_value_heads = num_attention_heads;

        float rms_norm_eps = 1e-6;

        bool mergeQKV = false;
        bool mergeSwiglu = false;
    };
}

#endif //FASTLLM_LLAMA_H
