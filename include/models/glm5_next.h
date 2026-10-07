#ifndef FASTLLM_GLM5_NEXT_H
#define FASTLLM_GLM5_NEXT_H

#include "basellm.h"

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fastllm {
    class CudaChunkedPrefillPipeline;

    struct Glm5NextIndexerCache {
        Data keys, tailKeys, tailGates;
        Data hadamard; // immutable workspace, not part of prefix snapshots
        Data pageTable; // derived GPU page map; rebuild after page/history changes
        std::vector<int> pageTableIds;
        int tokens = 0;
    };

    class Glm5NextModel : public basellm {
    public:
        Glm5NextModel();
        ~Glm5NextModel() override;

        void InitParams() override;

        std::map<std::string,
                 std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &tensorNames) override;

        void OnModelWeightsLoaded() override;
        int GetWeightLoadPriority(const std::string &tensorName,
                const std::vector<std::pair<std::string, DataType>> &mappedWeights) const override;
        bool ShouldLoadWeightSeriallyBeforeOthers(const std::string &tensorName,
                const std::vector<std::pair<std::string, DataType>> &mappedWeights) const override;
        void OnWeightLoadGroupStarted(const std::set<std::string> &weightNames) override;
        void OnWeightLoadGroupFinished() override;

        void SetDataType(DataType dataType) override;

        void OnResponseContextCreated(ResponseContext *context) override;

        void OnResponseContextRemoved(ResponseContext *context) override;

        bool TryRestoreHistoryCache(
                std::vector<int> &inputTokens, int &cacheLen) override;

        // KDA recurrent state cannot be rewound by slicing a token dimension.
        // Keep it atomically aligned with every DSA K/V layer in the
        // model-specific snapshots below.
        bool UseGenericHistoryCache() const override { return false; }

        bool RetainCudaWorkspace() const override { return true; }

        bool ShouldDelaySpecialWeightCudaMove(const std::string &) const override;

        int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector<float> *logits = nullptr) override;

        std::vector<int> ForwardBatch(
                int batch, const Data &inputIds,
                const std::vector<Data*> &attentionMasks,
                const std::vector<Data*> &positionIds,
                const std::vector<int> &seqLens,
                std::vector<std::pair<Data*, Data*>> &pastKeyValues,
                const std::vector<GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector<std::vector<float>*> *logits = nullptr) override;

        bool NeedAttentionMask(int qlen, int klen) override;

        bool TryForwardChunkedPrefill(
                const Data &inputIds, const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *logits, int &outputToken) override;

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
        friend struct Glm5NextGGUFTestAccess;
        struct ThreadTpState;
        std::unique_ptr<ThreadTpState> threadTpState;
        ThreadTpState *threadTpOwner = nullptr;
        int threadTpRank = -1;
        void InitThreadTp();
        int ThreadTpExpertLayer(const std::string &name) const;
        int StreamingThreadTpLayer(const std::string &name) const;
        void StageThreadTpWeight(const std::string &name);
        void PrepareThreadTp();
        void ThreadTpAllReduce(Data &data);
        void RemoveThreadTpRequest(const std::vector<std::pair<Data, Data>> *key);
        int ForwardThreadTp(const Data &inputIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens, std::vector<float> *logits);

        struct KdaReplayCapture {
            Data qProjected;
            Data kProjected;
            Data vProjected;
            Data k;
            Data v;
            Data rawGate;
            Data rawBeta;
        };

        struct TargetRuntimeCheckpoint {
            std::vector<Data> kdaFirst;
            std::vector<Data> kdaSecond;
            std::vector<int> sparseLengths;
            bool ready = false;
        };

        struct MtpRuntimeState {
            std::vector<std::pair<Data, Data>> pastKeyValues;
            std::vector<int> proposals;
            std::deque<std::pair<int, int>> pendingOutputTokens;
            Data deferredTargetHidden;
            bool hasDeferredTargetHidden = false;
            int deferredPosition = -1;
            int targetTokensConsumed = 0;
            int activeDraftLimit = 0;
            int consecutiveFullAccepts = 0;
            uint64_t verifiedDrafts = 0, acceptedDrafts = 0, verifySteps = 0;
            uint64_t confidenceChecks = 0, confidenceStops = 0;
            bool disabled = false;
            TargetRuntimeCheckpoint targetCheckpoint;
            std::vector<KdaReplayCapture> kdaReplay;
        };

        struct HistoryCacheMemory {
            std::vector<int> tokens;
            std::vector<std::pair<Data, Data>> pastKeyValues;
            std::vector<Glm5NextIndexerCache> indexer;
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
                bool sampleOutput,
                Data *targetHiddenStates = nullptr,
                std::vector<KdaReplayCapture> *kdaReplay = nullptr);

        int ForwardMtp(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *logits,
                MtpRuntimeState &state);

        void ForwardEmbedding(const Data &inputIds, Data &hiddenStates);

        void ForwardLayers(
                Data &hiddenStates, int firstLayer, int endLayer,
                const std::vector<std::vector<std::pair<Data, Data>>*>
                    &requestCaches,
                std::vector<KdaReplayCapture> *kdaReplay = nullptr,
                std::vector<Glm5NextIndexerCache> *indexer = nullptr);

        int ForwardOutput(
                Data &hiddenStates,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *logits, bool sampleOutput,
                Data *targetHiddenStates = nullptr);

        std::pair<Data &, Data &> PrepareMlaWeights(int layerIndex);

        bool MtpSupportsGenerationConfig(
                const GenerationConfig &generationConfig) const;

        bool CanUseExactBatchedMtpVerification(int rows) const;

        int RunMtpDraft(
                MtpRuntimeState &state,
                const Data &targetHiddenStates,
                const std::vector<int> &inputTokens,
                const std::vector<int> &positions,
                Data *nextHiddenStates,
                bool sampleToken, float *topProbability = nullptr);

        void GenerateMtpProposalChain(
                MtpRuntimeState &state,
                const Data &targetHiddenStates,
                const std::vector<int> &inputTokens,
                const std::vector<int> &positions, float minProbability = 0);

        void CaptureTargetRuntimeCheckpoint(
                const std::vector<std::pair<Data, Data>> &pastKeyValues,
                TargetRuntimeCheckpoint &checkpoint);

        void CommitTargetVerificationPrefix(
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const TargetRuntimeCheckpoint &checkpoint,
                const std::vector<KdaReplayCapture> &kdaReplay,
                int committedInputs,
                int verificationInputs);

        std::vector<int> SampleTargetRows(
                Data &hiddenStates,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                const std::vector<int> &proposals);

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
                const std::vector<std::vector<std::pair<Data, Data>>*> &requestCaches,
                Data &output,
                KdaReplayCapture *replayCapture = nullptr);

        void RunSparseAttention(
                int layerIndex, Data &input, int sequence,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                Data &output, Glm5NextIndexerCache *indexer = nullptr);

        void RunCompressedMlaAttention(
                int layerIndex, Data &input, int sequence,
                const std::vector<std::vector<std::pair<Data, Data>>*> &requestCaches,
                Data &output, Glm5NextIndexerCache *indexer = nullptr);

        void RunExpandedSparseAttention(
                int layerIndex, Data &input, int sequence,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                Data &output);

        void RunClampedMlp(
                Data &input, Data &gateUpWeight, Data &downWeight,
                Data &output);

        void RunMoe(
                int layerIndex, Data &input, int sequence, Data &output);

        void RunMoeWithPrefix(
                int deviceLayer, const std::string &mlpPrefix,
                std::vector<Data*> &weights,
                std::vector<Data*> &biases,
                Data &input, int sequence, Data &output);

        int Sample(
                Data &hiddenStates,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<float> *logits);

        int SampleLogits(
                Data &outputLogits,
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
        // FlashInfer's paged MLA kernel currently has a (512, 64)
        // specialization. GLM-5.3 has no positional K/Q component, so a
        // shared all-zero 64-wide component lets it use that specialization
        // without changing attention scores.
        int mlaPaddedPeHeadDim = 64;
        bool useCompressedMla = true;

        int denseIntermediateSize = 0;
        int moeIntermediateSize = 0;
        int firstDenseLayers = 0;
        float swigluLimit = 10.0f;

        int hcMult = 1;
        int hcSinkhornIters = 1;
        float hcEps = 1e-6f;

        int indexTopK = 0;
        enum class DsaBackend { Auto, BFloat16, Dense };
        DsaBackend dsaBackend = DsaBackend::Auto;
        bool UsesDsa() const { return dsaBackend != DsaBackend::Dense; }
        std::map<const std::vector<std::pair<Data, Data>> *,
                 std::vector<Glm5NextIndexerCache>> indexerCaches;
        std::mutex indexerCachesMutex;
#ifdef USE_CUDA
        std::unique_ptr<CudaChunkedPrefillPipeline> prefillPipeline;
#endif
        // DSA history is retained by page reference rather than copied.  This
        // limit covers KDA recurrent state and the pooled Indexer cache;
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
        std::vector<Data*> mtpExpertWeights;
        std::vector<Data*> mtpExpertBiases;

        bool mtpEnabled = false;
        bool mtpWeightsReady = false;
        int mtpDraftsPerStep = 0;
        float mtpMinProbability = 0;
        std::map<const std::vector<std::pair<Data, Data>> *,
                 std::shared_ptr<MtpRuntimeState>> mtpStates;
        std::mutex mtpStatesMutex;

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
