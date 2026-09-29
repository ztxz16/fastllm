#ifndef FASTLLM_NAIVE_N05_FLASH_H
#define FASTLLM_NAIVE_N05_FLASH_H

#include "basellm.h"

namespace fastllm {
    class NaiveN05FlashModel : public basellm {
    public:
        NaiveN05FlashModel();
        void InitParams() override;
        std::map<std::string, std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &names) override;
        int Forward(const Data &inputIds, const Data &attentionMask,
                    const Data &positionIds,
                    std::vector<std::pair<Data, Data>> &pastKeyValues,
                    const GenerationConfig &generationConfig = GenerationConfig(),
                    const LastTokensManager &lastTokens = LastTokensManager(),
                    std::vector<float> *logits = nullptr) override;
        bool NeedAttentionMask(int, int) override { return false; }
        // Sliding caches contain an absolute suffix; the generic prefix cache
        // cannot restore them. All DSA state lives in the ordinary KV tensors.
        bool UseGenericHistoryCache() const override { return false; }
        int GetKVCacheRetainedTokens(int layer) const override;
        void WarmUp() override;
        std::string MakeInput(const std::string &history, int,
                              const std::string &input) override { return history + input; }
        std::string MakeHistory(const std::string &history, int,
                                const std::string &input,
                                const std::string &output) override { return history + input + output; }

    private:
        struct AttentionConfig {
            int heads, kvHeads, headDim, valueDim;
            float theta;
        };
        AttentionConfig full, sliding;
        std::vector<int> slidingLayers, moeLayers;
        int window = 128, indexHeads = 16, indexDim = 128, indexTopK = 2048;
        float partialRotary = 0.334f, valueScale = 0.707f;
        bool indexFp8 = true;
        std::vector<std::vector<Data *>> moeWeights, moeBiases;
    };
}
#endif
