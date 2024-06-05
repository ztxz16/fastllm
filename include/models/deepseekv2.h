//
// Created by huangyuyang on 5/9/24.
//

#ifndef FASTLLM_DEEPSEEKV2_H
#define FASTLLM_DEEPSEEKV2_H

#include "basellm.h"
#include "llama.h"

#include "cmath"

#include <iostream>

namespace fastllm {
    class DeepSeekV2Model: public basellm {
    public:
        DeepSeekV2Model (); // 构造函数

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

        // 根据输入的tokens生成LLM推理的输入
        virtual void FillLLMInputsBatch(std::vector <std::vector <float> > &inputTokens,
                                        const std::vector <std::map <std::string, int> > &params,
                                        Data &inputIds, Data &attentionMask, Data &positionIds);

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

        float routed_scaling_factor;
        int num_experts_per_tok;
        int num_experts;
        bool norm_topk_prob;

        int max_position_embeddings;
        int rope_theta;
        int q_lora_rank;
        int qk_rope_head_dim;
        int kv_lora_rank;
        int v_head_dim;
        int qk_nope_head_dim;
        int q_head_dim;

        int rope_scaling_beta_fast;
        int rope_scaling_beta_slow;
        float rope_scaling_mscale;
        float rope_scaling_mscale_all_dim;
        float rope_scaling_original_max_position_embeddings;
        std::string rope_scaling_type;

        bool mergeSwiglu = false;
        std::vector <std::vector <Data*> > weights;
        std::vector <std::vector <Data*> > biass;
    };
}

#endif //FASTLLM_DEEPSEEKV2_H
