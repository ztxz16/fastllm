//
// Step-3.5 text model support.
//

#ifndef FASTLLM_STEP3P5_H
#define FASTLLM_STEP3P5_H

#include "basellm.h"

#include <set>
#include <string>
#include <vector>

namespace fastllm {
    class Step3p5Model: public basellm {
    public:
        Step3p5Model();

        virtual void InitParams();

        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr);

        virtual std::vector <int> ForwardV2(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector <std::vector <float>*> *retLogits);

        virtual bool NeedAttentionMask(int qlen, int klen);

        virtual void WarmUp();

        virtual std::string ApplyChatTemplate(const ChatMessages &messages);

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input);

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output);

    protected:
        float rope_base = 10000.0f;
        float rope_factor = 1.0f;
        float llama3_original_max_position_embeddings = 131072.0f;
        float llama3_low_freq_factor = 1.0f;
        float llama3_high_freq_factor = 32.0f;
        float rms_norm_eps = 1e-6f;
        int base_attention_heads = 64;
        int base_key_value_heads = 8;
        int sliding_attention_heads = 96;
        int sliding_key_value_heads = 8;
        int sliding_window = 512;
        int dense_intermediate_size = 11264;
        int moe_intermediate_size = 1280;
        int shared_expert_intermediate_size = 1280;
        bool norm_topk_prob = true;
        bool use_moe_router_bias = true;
        bool need_fp32_gate = true;
        bool initialized_add1 = false;
        bool moeWeightsPrepared = false;

        std::vector <std::string> layer_types;
        std::set <int> moe_layers;
        std::vector <float> layer_rope_thetas;
        std::vector <int> layer_rotary_dims;
        std::vector <float> swiglu_limits;
        std::vector <float> swiglu_limits_shared;
        std::vector <std::vector <Data*> > moeGateWeights;
        std::vector <std::vector <Data*> > moeUpWeights;
        std::vector <std::vector <Data*> > moeDownWeights;
        std::vector <std::vector <Data*> > weights;
        std::vector <std::vector <Data*> > biass;

        int LayerAttentionHeads(int layer) const;
        int LayerKeyValueHeads(int layer) const;
        bool IsFullAttentionLayer(int layer) const;
        bool IsMoeLayer(int layer) const;
        bool UseLlama3Rope(int layer) const;
        void PrepareMoeWeights();
        void ApplyStepRotary(Data &input, const Data &positionIds, int layer);
    };
}

#endif
