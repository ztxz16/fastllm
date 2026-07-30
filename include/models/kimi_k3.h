#ifndef FASTLLM_KIMI_K3_H
#define FASTLLM_KIMI_K3_H

#include "basellm.h"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fastllm {
    class KimiK3Model : public basellm {
    public:
        KimiK3Model();

        ~KimiK3Model() override;

        void InitParams() override;

        std::map<std::string, std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &tensorNames) override;

        void OnModelWeightsLoaded() override;

        void WarmUp() override;

        void OnResponseContextCreated(ResponseContext *context) override;

        void OnResponseContextRemoved(ResponseContext *context) override;

        // DSpark owns an additional five-layer draft KV cache derived from
        // target hidden states. FastLLM's generic history cache only stores
        // the target caches, so restoring it would desynchronize the models.
        bool UseGenericHistoryCache() const override {
            return !dsparkEnabled;
        }

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
        struct KdaReplayCapture {
            Data qProjected;
            Data kProjected;
            Data vProjected;
            Data k;
            Data v;
            Data rawGate;
            Data rawBeta;
        };

        struct TargetRunCapture {
            bool captureKda = false;
            std::map<int, Data> targetHidden;
            std::vector<KdaReplayCapture> kda;
        };

        struct DsparkPendingStep {
            int expectedInput = -1;
            int outputToken = -1;
        };

        struct DsparkDraftProposal {
            std::vector<int> tokens;
            std::vector<float> confidence;
        };

        struct DsparkContext {
            bool initialized = false;
            int committedTokens = 0;
            int adaptiveDraftLimit = 0;
            std::vector<std::pair<Data, Data>> draftKeyValues;
            std::vector<std::pair<Data, Data>> kdaSnapshots;
            std::vector<KdaReplayCapture> replay;
            std::deque<DsparkPendingStep> pending;
        };

        Data RunFirstLayerImpl(
                const std::vector<int> &tokenIds,
                std::vector<std::pair<Data, Data>> *pastKeyValues,
                TargetRunCapture *capture = nullptr);

        Data RunLayersImpl(
                const std::vector<int> &tokenIds,
                int layerCount,
                std::vector<std::pair<Data, Data>> *pastKeyValues,
                TargetRunCapture *capture = nullptr);

        int SampleTargetHidden(
                Data &hiddenStates,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *logits);

        DsparkContext &GetDsparkContext(
                std::vector<std::pair<Data, Data>> &pastKeyValues);

        void EnsureDsparkRotary(int positions);

        void AppendDsparkTargetHidden(
                const TargetRunCapture &capture,
                int tokens,
                DsparkContext &context);

        DsparkDraftProposal RunDsparkDraft(
                int anchorToken,
                DsparkContext &context);

        int SelectDsparkVerifyDrafts(
                const DsparkDraftProposal &proposal,
                const DsparkContext &context) const;

        void UpdateDsparkAdaptiveLimit(
                int verifyDrafts,
                int acceptedDrafts,
                DsparkContext &context) const;

        void SnapshotKdaCaches(
                const std::vector<std::pair<Data, Data>> &pastKeyValues,
                DsparkContext &context);

        void CommitTargetVerification(
                int oldTokens,
                int commitTokens,
                int verifyTokens,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                DsparkContext &context);

        int ForwardDspark(
                const std::vector<int> &tokenIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *logits);

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
        bool cudaWeightWarmupRunning = false;

        bool dsparkEnabled = false;
        int dsparkBlockSize = 0;
        int dsparkLayers = 0;
        int dsparkHeads = 0;
        int dsparkKvHeads = 0;
        int dsparkHeadDim = 0;
        int dsparkIntermediateSize = 0;
        int dsparkMaskTokenId = -1;
        int dsparkMarkovRank = 0;
        float dsparkRmsNormEps = 1e-5f;
        float dsparkConfidenceThreshold = 0.5f;
        std::vector<int> dsparkTargetLayerIds;
        Data dsparkSinData;
        Data dsparkCosData;
        int dsparkRotaryCapacity = 0;

        std::vector<bool> kdaLayers;

        std::vector<std::vector<Data*>> expertW1s;
        std::vector<std::vector<Data*>> expertW2s;
        std::vector<std::vector<Data*>> expertW3s;

        std::unordered_map<
            const std::vector<std::pair<Data, Data>>*,
            std::unique_ptr<DsparkContext>> dsparkContexts;
        std::mutex dsparkContextMutex;

        static const std::string languagePrefix;
    };
}

#endif
