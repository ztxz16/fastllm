#ifndef FASTLLM_GLM5_NEXT_H
#define FASTLLM_GLM5_NEXT_H

#include "basellm.h"

#include <string>
#include <vector>

namespace fastllm {
    class Glm5NextModel : public basellm {
    public:
        Glm5NextModel();
        ~Glm5NextModel() override;

        void InitParams() override;

        std::map<std::string,
                 std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &tensorNames) override;

        void OnModelWeightsLoaded() override;

        void SetDataType(DataType dataType) override;

        // The generic history cache assumes that every layer can be restored
        // from one shared token-prefix length.  GLM-5.3 mixes fixed KDA
        // recurrent state with token-growing DSA caches, so use a fresh
        // prefill until an atomic model-specific snapshot is available.
        bool UseGenericHistoryCache() const override { return false; }

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
        void RunKdaAttention(
                int layerIndex, Data &input, int sequence,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                Data &output);

        void RunSparseAttention(
                int layerIndex, Data &input, int sequence,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                Data &output);

        void RunClampedMlp(
                Data &input, Data &gateUpWeight, Data &downWeight,
                Data &output);

        void RunMoe(
                int layerIndex, Data &input, int sequence, Data &output);

        int Sample(
                Data &hiddenStates,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *logits);

        int kdaHeads = 0;
        int kdaHeadDim = 0;
        int shortConvKernel = 0;
        float gateLowerBound = -5.0f;

        int qLoraRank = 0;
        int kvLoraRank = 0;
        int qkNopeHeadDim = 0;
        int qkRopeHeadDim = 0;
        int qkHeadDim = 0;
        int valueHeadDim = 0;

        int denseIntermediateSize = 0;
        int moeIntermediateSize = 0;
        int firstDenseLayers = 0;
        float swigluLimit = 10.0f;

        int hcMult = 1;
        int hcSinkhornIters = 1;
        float hcEps = 1e-6f;

        int indexTopK = 0;
        int exactSparseAttentionLimit = 0;

        std::vector<bool> kdaLayers;
        std::vector<bool> denseMlpLayers;
        std::vector<std::vector<Data*>> expertWeights;
        std::vector<std::vector<Data*>> expertBiases;

        static const std::string languagePrefix;
    };
}

#endif
