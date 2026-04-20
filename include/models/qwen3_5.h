//
// Created by huangyuyang on 3/5/26.
//

#ifndef FASTLLM_QWEN3_5_H
#define FASTLLM_QWEN3_5_H
#include "basellm.h"
#include "cmath"

#include <iostream>
#include <vector>

namespace fastllm {
    class Qwen3_5Model: public basellm {
    public:
    Qwen3_5Model (); // 构造函数

        virtual void InitParams(); // 初始化参数信息

        virtual void OnWeightLoaded(const std::string &weightName, const std::set<std::string> &finishedWeightNames) override;
        
        // 推理
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
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr);

        virtual std::vector <int> ForwardMultimodal(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const std::map <std::string, std::vector <Data*> > &multimodalInput,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr) override;
        
        // 是否需要生成AttentionMask
        virtual bool NeedAttentionMask(int qlen, int klen);

        virtual void WarmUp(); // 预热

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

        std::pair<std::vector<float>, std::vector<float>> UpdateRotaryPosEmb(float base, float factor, int seqLen = 0); // 更新位置编码

        static const std::string language_prefix;
        static const std::string visual_prefix;

    protected:
        RoPEType rope_type = RoPEType::BASE;

        float rope_base = 10000.f;

        float rope_factor = 1.f;

        int num_key_value_heads = num_attention_heads;

        float rms_norm_eps = 1e-6;

        bool mergeQKV = false;
        bool mergeSwiglu = false;

        bool initialized_add1 = false;

        int num_k_heads, num_v_heads, head_k_dim, head_v_dim;

        Data inv_scale_data;

        std::vector <std::vector <Data*> > weights;
        std::vector <std::vector <Data*> > biass;
        bool moeWeightsPrepared = false;
        bool gdnMergedWeightsPrepared = false;
        std::vector <int> mrope_sections = {11, 11, 10};
        bool visionPrepared = false;
        int vision_depth = 0;
        int vision_hidden_size = 0;
        int vision_num_heads = 0;
        int vision_head_dim = 0;
        int vision_intermediate_size = 0;
        int vision_patch_size = 16;
        int vision_temporal_patch_size = 2;
        int vision_spatial_merge_size = 2;
        int vision_out_hidden_size = 0;
        int vision_num_position_embeddings = 0;
        int vision_num_grid_per_side = 0;
        int image_token_id = -1;
        int video_token_id = -1;
        int vision_start_token_id = -1;
        int vision_end_token_id = -1;
        std::vector<int> vision_deepstack_visual_indexes;
        std::vector<float> vision_image_mean = {0.5f, 0.5f, 0.5f};
        std::vector<float> vision_image_std = {0.5f, 0.5f, 0.5f};
        Data visionSinData;
        Data visionCosData;

        void SplitFusedMoeWeightsIfNeeded(const std::string &layerPrefix);
        void PrepareMoeWeights();
        void PrepareGdnWeights();
        void PrepareVision();
        Data BuildFlattenedPositionIds(const std::vector <Data*> &positionIds,
                                      const std::vector <int> &seqLens,
                                      bool all1);
        void MergeMultimodalFeaturesIntoText(const Data &mmTokenTypeIds,
                                             const Data *imageEmbeds,
                                             const Data *videoEmbeds,
                                             Data &hiddenStates);
        void ApplyVisionRotary(Data &input, const Data &posX, const Data &posY);
        void EncodeVisualItems(const std::vector <Data*> &rawInputs,
                               const Data *gridThwData,
                               bool isVideo,
                               Data &features,
                               std::vector<std::vector<int>> &gridThwList);
        void BuildMultimodalPositionData(const Data &inputIds,
                                         const std::vector<std::vector<int>> &imageGridThwList,
                                         const std::vector<std::vector<int>> &videoGridThwList,
                                         Data &mmTokenTypeIds,
                                         Data &mropePositionIds,
                                         Data &mropePositionDelta);
        void ApplyMultimodalRotary(Data &input, const Data &positionIds, float ropeScale);
        void AdjustPositionIdsWithDelta(const Data &positionIds,
                                        const Data &mropePositionDelta,
                                        Data &adjustedPositionIds);
        std::vector <int> ForwardFromHiddenStates(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const Data &allPositionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector <std::vector <float>*> *retLogits,
                Data &hiddenStates,
                bool all1);
    };
}

#endif //FASTLLM_QWEN3_5_H
