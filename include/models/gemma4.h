//
// Created for Gemma4 support in fastllm
//

#ifndef FASTLLM_GEMMA4_H
#define FASTLLM_GEMMA4_H

#include "basellm.h"
#include "cmath"

#include <iostream>

namespace fastllm {
    class Gemma4Model: public basellm {
    public:
    Gemma4Model ();

        virtual void InitParams();

        virtual void OnWeightLoaded(const std::string &weightName, const std::set<std::string> &finishedWeightNames) override;

        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr);

        virtual std::vector <int> ForwardMultimodal(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const std::map <std::string, std::vector <Data*> > &multimodalInput,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr) override;

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

        virtual bool NeedAttentionMask(int qlen, int klen);

        virtual void FillLLMInputsBatch(std::vector <std::vector <float> > &inputTokens,
                                        const std::vector <std::map <std::string, int> > &params,
                                        Data &inputIds, Data &attentionMask, Data &positionIds);

        virtual void Prepare();

        virtual void WarmUp();

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input);

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output);

        std::pair<std::vector<float>, std::vector<float>> UpdateRotaryPosEmb(float base, float factor, int seqLen, int dim);

    protected:
        float rms_norm_eps = 1e-6;
        int num_key_value_heads = 32;
        int global_num_key_value_heads = 4;
        int sliding_head_dim = 256;
        int global_head_dim = 512;
        int sliding_window = 1024;
        bool attention_k_eq_v = true;
        float final_logit_softcapping = 0.0f;

        float sliding_rope_base = 10000.0f;
        float global_rope_base = 1000000.0f;
        float global_partial_rotary_factor = 0.25f;

        std::vector<int> layer_types; // 0 = sliding, 1 = full

        Data slidingSinData, slidingCosData;
        Data globalSinData, globalCosData;

        std::vector<std::vector<float>> slidingSin, slidingCos;
        std::vector<std::vector<float>> globalSin, globalCos;

        bool enable_moe_block = false;
        int moe_intermediate_size = 0;
        bool moeWeightsPrepared = false;

        int image_token_id = -1;
        int boi_token_id = -1;
        int eoi_token_id = -1;

        int vision_hidden_size = 0;
        int vision_num_layers = 0;
        int vision_num_heads = 0;
        int vision_num_key_value_heads = 0;
        int vision_head_dim = 0;
        int vision_patch_size = 16;
        int vision_pooling_kernel_size = 3;
        int vision_position_embedding_size = 0;
        int vision_max_soft_tokens = 280;
        bool vision_standardize = false;

        Data visionSinData, visionCosData;

        std::vector<std::vector<Data*>> weights;
        std::vector<std::vector<Data*>> biass;
        std::vector<std::vector<Data*>> expertGateupWeights;
        std::vector<std::vector<Data*>> expertDownWeights;

        void SplitFusedMoeWeightsIfNeeded(const std::string &layerPrefix);
        void PrepareMoeWeights();
        bool TryApplyMoeFeedForward(const std::string &layerPrefix, Data &hiddenStates, Data &denseOutput);
        void PrepareVision();
        void ApplyRMSNormNoScale(Data &input, float eps);
        void BuildVisionPatchPositionIds(const Data &imagePositionIds, Data &posX, Data &posY, std::vector<int> &validPatchCounts);
        void BuildVisionAttentionMask(const Data &imagePositionIds, Data &visionAttentionMask);
        void ApplyVisionRotary(Data &input, const Data &posX, const Data &posY);
        void EncodeImages(const Data &pixelValues, const Data &imagePositionIds, Data &imageFeatures, std::vector<int> &softTokenCounts);
        void MergeImageFeaturesIntoText(const Data &inputIds, const Data &imageFeatures, Data &hiddenStates);
        void BuildVisionAwareTextMask(const Data &attentionMask, const Data &mmTokenTypeIds, Data &visionAwareMask);
        int ForwardTextFromHiddenStates(const Data &inputIds,
                                        Data &hiddenStates,
                                        const Data &attentionMask,
                                        const Data &positionIds,
                                        std::vector <std::pair <Data, Data> > &pastKeyValues,
                                        const GenerationConfig &generationConfig = GenerationConfig(),
                                        const LastTokensManager &lastTokens = LastTokensManager(),
                                        std::vector <float> *retLogits = nullptr);

        bool prepared = false;
        bool visionPrepared = false;
    };
}

#endif //FASTLLM_GEMMA4_H
