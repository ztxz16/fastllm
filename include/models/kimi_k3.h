#ifndef FASTLLM_KIMI_K3_H
#define FASTLLM_KIMI_K3_H

#include "basellm.h"

#include <map>
#include <string>
#include <vector>

namespace fastllm {
    class KimiK3Model : public basellm {
    public:
        KimiK3Model();

        void InitParams() override;

        std::map<std::string, std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &tensorNames) override;

        void OnModelWeightsLoaded() override;

        void WarmUp() override;

        int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector<float> *logits = nullptr) override;

        bool NeedAttentionMask(int qlen, int klen) override;

        std::string MakeInput(
                const std::string &history,
                int round,
                const std::string &input) override;

        std::string MakeHistory(
                const std::string &history,
                int round,
                const std::string &input,
                const std::string &output) override;

    private:
        Data RunFirstLayerImpl(
                const std::vector<int> &tokenIds,
                std::vector<std::pair<Data, Data>> *pastKeyValues);

        Data RunLayersImpl(
                const std::vector<int> &tokenIds,
                int layerCount,
                std::vector<std::pair<Data, Data>> *pastKeyValues);

        int attnResBlockSize = 0;
        int kdaHeads = 0;
        int kdaHeadDim = 0;
        int shortConvKernel = 0;
        int expertCount = 0;
        int expertsPerToken = 0;
        int routedExpertHiddenSize = 0;
        int moeIntermediateSize = 0;
        int qLoraRank = 0;
        int kvLoraRank = 0;
        int qkNopeHeadDim = 0;
        int qkRopeHeadDim = 0;
        int vHeadDim = 0;
        float gateLowerBound = -5.0f;
        float situBeta = 4.0f;
        float situLinearBeta = 25.0f;

        std::vector<bool> kdaLayers;

        std::vector<std::vector<Data*>> expertW1s;
        std::vector<std::vector<Data*>> expertW2s;
        std::vector<std::vector<Data*>> expertW3s;

        static const std::string languagePrefix;
    };
}

#endif
