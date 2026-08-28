#ifndef FASTLLM_GLM5_NEXT_H
#define FASTLLM_GLM5_NEXT_H

#include "basellm.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
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

        void OnResponseContextCreated(ResponseContext *context) override;

        void OnResponseContextRemoved(ResponseContext *context) override;

        bool TryRestoreHistoryCache(
                std::vector<int> &inputTokens, int &cacheLen) override;

        // KDA recurrent state cannot be rewound by slicing a token dimension.
        // Keep it atomically aligned with every DSA K/V layer in the
        // model-specific snapshots below.
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
        struct HistoryCacheMemory {
            std::vector<int> tokens;
            std::vector<std::pair<Data, Data>> pastKeyValues;
            int sequenceLength = 0;
            uint64_t bytes = 0;
            bool recurrentStateOnCpu = false;
            long long flushTime = 0;
        };

        int ForwardImpl(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *logits,
                bool sampleOutput);

        int GetHistoryCacheSequenceLength(
                const std::vector<std::pair<Data, Data>>
                    &pastKeyValues) const;

        void RecordHistoryCache(
                const std::vector<int> &tokens,
                const std::vector<std::pair<Data, Data>>
                    &pastKeyValues,
                int sequenceLength);

        bool CanRestoreHistoryCache(
                const HistoryCacheMemory &memory) const;

        void RestoreHistoryCache(
                const HistoryCacheMemory &memory,
                ResponseContext *context);

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
        // DSA history is retained by page reference rather than copied.  This
        // limit therefore applies only to the fixed-size KDA recurrent state;
        // larger state snapshots are tiered to host memory.
        uint64_t historyCacheGpuStateLimitBytes =
            1024ULL * 1024ULL * 1024ULL;
        // LRU target rather than a hard refusal threshold: keep at least the
        // newest snapshot so one long-context request remains reusable.
        uint64_t historyCacheMaxBytes =
            16ULL * 1024ULL * 1024ULL * 1024ULL;

        std::vector<bool> kdaLayers;
        std::vector<bool> denseMlpLayers;
        std::vector<std::vector<Data*>> expertWeights;
        std::vector<std::vector<Data*>> expertBiases;

        std::map<std::vector<int>, std::shared_ptr<HistoryCacheMemory>>
            historyCache;
        std::shared_ptr<HistoryCacheMemory> pendingHistoryCache;
        std::map<const std::vector<std::pair<Data, Data>> *,
                 ResponseContext *> responseContexts;
        std::mutex historyCacheMutex;
        std::mutex responseContextsMutex;
        uint64_t historyCacheBytes = 0;
        long long historyCacheFlushTime = 0;
        int historyCacheMaxRecords = 5;

        static const std::string languagePrefix;
    };
}

#endif
