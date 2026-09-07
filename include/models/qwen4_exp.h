//
// Qwen4-Exp / Qwen3.8-Flash-Next text model.
//

#ifndef FASTLLM_QWEN4_EXP_H
#define FASTLLM_QWEN4_EXP_H

#include "qwen3_next.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace fastllm {
    class Qwen4ExpModel : public Qwen3NextModel {
    public:
        Qwen4ExpModel();
        ~Qwen4ExpModel() override;

        void InitParams() override;

        std::map<std::string, std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &tensorNames) override;
        void OnWeightLoaded(const std::string &weightName,
                            const std::set<std::string> &finishedWeightNames) override;
        void OnModelWeightsLoaded() override;
        bool ShouldDelaySpecialWeightNumaRegistration(
                const std::string &weightName) const override;

        int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector<float> *logits = nullptr) override;

        std::vector<int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector<std::vector<float> *> *logits = nullptr) override;

        std::vector<int> ForwardMultimodal(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const std::map<std::string, std::vector<Data *>> &multimodalInput,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector<std::vector<float> *> *logits = nullptr) override;

        void WarmUp() override;

        bool TryRestoreHistoryCache(std::vector<int> &inputTokens,
                                    int &cacheLen) override;
        void OnResponseContextCreated(ResponseContext *context) override;
        void OnResponseContextRemoved(ResponseContext *context) override;
        bool UseGenericHistoryCache() const override { return false; }

    private:
        struct PrefixSnapshot;
        struct DecodeCudaGraphState;
        struct MtpDraftCudaGraphState;
        struct QsaHostMirrorTransfer;
        struct MtpRuntimeState;

        struct RequestState {
            int previousToken1 = -1;
            int previousToken2 = -1;
            std::vector<float> convHistory;
            // PLE's dilated short-convolution state stays on its execution
            // device between forwards.  Two tensors are used as a ping-pong
            // cache because the standard operation reads the old history and
            // writes the next history separately.  convHistory remains the
            // self-contained host representation used by CPU fallback and
            // prefix snapshots.
            std::shared_ptr<Data> pleConvHistoryTensors[2];
            int pleConvHistoryIndex = 0;
            std::map<int, std::vector<float>> indexerRawKeys;
            std::map<int, std::vector<float>> indexerPositions;
            // CPU/non-contiguous-mask fallback cache. CUDA appends completed
            // compression groups to a pinned host mirror and leaves fewer
            // than indexerCompressRatio raw rows in the device tail. The
            // vectors are materialized only at a fallback or snapshot boundary.
            std::map<int, std::vector<float>> indexerBlockKeys;
            std::map<int, std::shared_ptr<QsaHostMirrorTransfer>>
                indexerHostMirrorTransfers;
            std::map<int, std::shared_ptr<Data>> indexerTailKeyTensors;
            std::map<int, std::shared_ptr<Data>> indexerTailPositionTensors;
            std::map<int, std::shared_ptr<Data>> indexerBlockKeyTensors;
            // Geometric cache growth is enabled per full-attention layer only
            // after that request has completed its first single-token pass,
            // so CUDA Graph capture retains the established allocation order.
            std::set<int> geometricCacheGrowthReadyLayers;
            std::vector<int> processedTokens;
            int prefixRequestId = 0;
            int lastPrefixSnapshotLen = 0;
            // Keeps GPU-resident snapshot storage alive while restored cache
            // tensors borrow it. The next Forward detaches mutable tensors
            // with a device-to-device copy and releases this reference.
            std::shared_ptr<PrefixSnapshot> borrowedPrefixSnapshot;
            // The one-layer MTP cache is request-local. Prefix snapshots keep
            // a compact, independently owned copy only when it is aligned at
            // the token boundary immediately preceding the deferred target
            // hidden state.
            std::shared_ptr<MtpRuntimeState> mtpState;
            bool mtpDisabled = false;
        };

        struct RequestRuntimeCheckpoint {
            int previousToken1 = -1;
            int previousToken2 = -1;
            std::vector<float> convHistory;
            std::shared_ptr<Data> pleConvHistoryTensors[2];
            int pleConvHistoryIndex = 0;
            std::map<int, std::vector<float>> indexerRawKeys;
            std::map<int, std::vector<float>> indexerPositions;
            std::map<int, std::vector<float>> indexerBlockKeys;
            std::map<int, int> indexerLengths;
            std::map<int, std::shared_ptr<Data>> indexerTailKeyTensors;
            std::map<int, std::shared_ptr<Data>> indexerTailPositionTensors;
            std::map<int, std::shared_ptr<Data>> indexerBlockKeyTensors;
            std::set<int> geometricCacheGrowthReadyLayers;
            std::vector<int> processedTokens;
        };

        struct TargetRuntimeCheckpoint {
            std::vector<int> keyLengths;
            std::vector<int> valueLengths;
            std::vector<Data> linearFirst;
            std::vector<Data> linearSecond;
            RequestRuntimeCheckpoint request;
        };

        struct TargetVerificationCapture {
            std::vector<Data> linearConvInputs;
            std::vector<Data> linearAlphas;
            std::vector<Data> linearBetas;
            // Non-owning destination for verifier-time recurrent outputs.
            // The live caches remain the replay baseline while the existing
            // checkpoint buffers provide stable CUDA graph output pointers.
            TargetRuntimeCheckpoint *runtimeCheckpoint = nullptr;
            std::vector<Data> linearReplayOutputs;
            std::vector<Data> linearReplayCoreOutputs;
            Data linearReplayPointerWorkspace;
            std::vector<uint8_t> linearReplayPointerCache;
            std::map<int, Data> qsaRawKeys;
            Data pleInput;
            Data greedyTokenIds;
            Data greedyTokenValues;
        };

        struct MtpRuntimeState {
            Data key;
            Data value;
            RequestState attentionState;
            // Keep sampling outputs resident on CUDA across verifier cycles;
            // the host reads only the compact token-id prefix.
            Data sampledTokenIds;
            Data sampledTokenValues;
            std::shared_ptr<MtpDraftCudaGraphState> draftGraphState;
            std::vector<int> proposals;
            std::deque<int> pendingOutputTokens;
            Data deferredTargetHidden;
            int deferredPosition = -1;
            bool hasDeferredTargetHidden = false;
            TargetRuntimeCheckpoint targetCheckpoint;
            bool targetCheckpointPrepared = false;
            TargetVerificationCapture targetCapture;
        };

        struct PrefixLayerSnapshot {
            bool linear = false;
            Data first;
            Data second;
            DataDevice firstDevice = DataDevice::CPU;
            DataDevice secondDevice = DataDevice::CPU;
            std::vector<int> firstDeviceIds;
            std::vector<int> secondDeviceIds;
            bool qsaDeviceCache = false;
            Data qsaTailKeys;
            Data qsaTailPositions;
            Data qsaBlockKeys;
        };

        struct PrefixSnapshot {
            int cachedLen = 0;
            int requestId = 0;
            long long timestamp = 0;
            uint64_t tensorBytes = 0;
            uint64_t stateBytes = 0;
            std::vector<int> tokens;
            std::vector<PrefixLayerSnapshot> layers;
            RequestState state;
        };

        struct PendingPrefixRestore {
            int cachedLen = 0;
            std::vector<int> tokens;
            std::shared_ptr<PrefixSnapshot> snapshot;
        };

        static const std::string languagePrefix;
        static const std::string visualPrefix;

        int hcCount = 4;
        int hcLowRank = 320;
        int linearConvKernel = 4;
        int pleLayer = 1;
        int pleEmbedDim = 2560;
        int pleConvKernel = 4;
        int ngramSize = 3;
        int headsPerNgram = 8;
        int ngramHeads = 16;
        int ngramHeadDim = 160;
        int ngramVocabBase = 20000000;
        int ngramShardCount = 128;
        int eosToken = 248044;
        int pleSeed = 1234;
        int indexerHeads = 4;
        int indexerKvHeads = 1;
        int indexerHeadDim = 128;
        int indexerBudget = 2048;
        int indexerCompressRatio = 4;
        std::vector<int> mropeSections = {11, 11, 10};

        bool visionPrepared = false;
        int visionDepth = 0;
        int visionHiddenSize = 0;
        int visionNumHeads = 0;
        int visionHeadDim = 0;
        int visionIntermediateSize = 0;
        int visionPatchSize = 16;
        int visionTemporalPatchSize = 2;
        int visionSpatialMergeSize = 2;
        int visionOutHiddenSize = 0;
        int visionNumPositionEmbeddings = 0;
        int visionNumGridPerSide = 0;
        int imageTokenId = -1;
        int videoTokenId = -1;
        std::vector<int> visionDeepstackIndexes;
        std::vector<float> visionImageMean = {0.5f, 0.5f, 0.5f};
        std::vector<float> visionImageStd = {0.5f, 0.5f, 0.5f};
        Data visionSinData;
        Data visionCosData;

        bool preparedWeights = false;
        std::atomic<int> mtpWeightsStatus{-1};
        std::mutex prepareMutex;
        mutable std::mutex stateMutex;
        mutable std::map<const Data *, RequestState> requestStates;
        mutable std::map<const Data *, std::unique_ptr<DecodeCudaGraphState>>
            decodeCudaGraphStates;
        mutable std::mutex prefixCacheMutex;
        std::vector<std::shared_ptr<PrefixSnapshot>> prefixSnapshots;
        std::deque<PendingPrefixRestore> pendingPrefixRestores;
        long long prefixSnapshotTimestamp = 0;
        int prefixRequestCounter = 0;

        std::vector<uint64_t> pleMultipliers;
        std::vector<int64_t> pleHeadVocabSizes;
        std::vector<int64_t> pleHeadOffsets;
        std::vector<bool> linearLayers;
        // Shared Q/K L2-normalization scale. Keep one persistent tensor so
        // prefill does not rebuild and upload the same 128-float constant in
        // every linear-attention layer.
        Data linearInvScaleData;
        // Logical concatenation of the lazy shard metadata used by the
        // standard disk EmbeddingDirect operation.
        Data pleNgramDiskWeight;
        // QSA cache compression applies RoPE on the host while regular
        // attention applies it on its execution device. Keep an immutable
        // host view so cache updates never migrate the shared sinData/cosData
        // tensors away from CUDA during decode.
        std::vector<float> qsaSinValues;
        std::vector<float> qsaCosValues;
        std::map<int, std::vector<float>> qsaKeyNormValues;
        std::vector<Data *> mtpMoeWeights;
        std::vector<Data *> mtpMoeBiass;
        // MTP only needs an approximate proposal distribution: target
        // verification still uses the original lm_head and therefore keeps
        // the generated token stream exact.  A packed FP8 copy cuts the three
        // bandwidth-bound draft head scans in half on supported CUDA models.
        Data mtpDraftLmHeadWeight;
        bool mtpDraftLmHeadReady = false;
        bool mtpDraftLmHeadAttempted = false;

        void PrepareWeights();
        void PrepareMtpDraftLmHeadWeight();
        bool IsLinearAttentionLayer(int layer) const;
        void PrepareVision();
        void ApplyVisionRotary(Data &input, const Data &posX,
                               const Data &posY);
        void ApplyTextRotary(Data &input, const Data &positionIds);
        void EncodeVisualItems(
            const std::vector<Data *> &rawInputs,
            const Data *gridThwData,
            bool isVideo,
            Data &features,
            std::vector<std::vector<int>> &gridThwList);
        void BuildMultimodalPositionData(
            const Data &inputIds,
            const std::vector<std::vector<int>> &imageGridThwList,
            const std::vector<std::vector<int>> &videoGridThwList,
            Data &mmTokenTypeIds,
            Data &mropePositionIds,
            Data &mropePositionDelta);
        void MergeMultimodalFeaturesIntoText(
            const Data &mmTokenTypeIds,
            const Data *imageEmbeds,
            const Data *videoEmbeds,
            Data &hiddenStates);
        void AdjustPositionIdsWithDelta(
            const Data &positionIds,
            const Data &mropePositionDelta,
            Data &adjustedPositionIds);

        void GroupedRMSNorm(const Data &input, Data &normWeight, Data &output);
        void HyperMixNormalized(Data &normalized,
                                const std::string &prefix,
                                Data &mixedInput,
                                Data *injectionWeights,
                                const Data *projectionStorage = nullptr);
        void HyperCombine(const Data &hyperInput, const Data &blockOutput,
                          const Data &injectionWeights, Data &output);
        void HyperCombineRMSNorm(const Data &hyperInput,
                                 const Data &blockOutput,
                                 const Data &injectionWeights,
                                 Data &normWeight,
                                 Data &output,
                                 Data &normalized,
                                 Data *normalizedStorage = nullptr);

        void RunPLE(const Data &hyperInput, const Data &inputIds,
                    RequestState &state, Data &output,
                    const std::vector<int> *hostInputTokens = nullptr);
        void MaterializePLEHostHistory(RequestState &state);
        void BuildQSAMask(int layer, const std::string &attentionPrefix,
                          const Data &input,
                          const Data &baseMask, const Data &positionIds,
                          int previousLength, bool deviceCompatibleMask,
                          RequestState &state,
                          Data &qsaMask, Data &qsaIndices,
                          Data *rawKeyCapture = nullptr,
                          bool fixedCausalWidth = false);
        void MaterializeQsaHostHistory(int layer, int length,
                                       RequestState &state);
        void RunFullAttention(int layer, const Data &input,
                              const Data &attentionMask,
                              const Data &positionIds,
                              bool qsaDeviceCompatibleMask,
                              Data &pastKey, Data &pastValue,
                              RequestState &state, Data &output,
                              Data *qsaRawKeyCapture = nullptr);
        void RunFullAttentionWithPrefix(
                              int stateLayer,
                              const std::string &attentionPrefix,
                              const Data &input,
                              const Data &attentionMask,
                              const Data &positionIds,
                              bool qsaDeviceCompatibleMask,
                              Data &pastKey, Data &pastValue,
                              RequestState &state, Data &output,
                              Data *qsaRawKeyCapture = nullptr);
        void RunLinearAttention(int layer, const Data &input,
                                Data &pastConv, Data &pastRecurrent,
                                Data &output,
                                Data *convInputCapture = nullptr,
                                Data *alphaCapture = nullptr,
                                Data *betaCapture = nullptr,
                                Data *recurrentStateOutput = nullptr);
        void RunMoE(int layer, const Data &input, Data &output);
        void RunMoEWithPrefix(int deviceLayer,
                              const std::string &mlpPrefix,
                              std::vector<Data *> &moeWeights,
                              std::vector<Data *> &moeBiass,
                              const Data &input, Data &output);

        bool HasMtpWeights() const;
        bool MtpSupportsGenerationConfig(
            const GenerationConfig &generationConfig) const;
        bool CanCloneMtpPrefixState(
            const MtpRuntimeState &source, int cachedLen) const;
        std::shared_ptr<MtpRuntimeState> CloneMtpPrefixState(
            const MtpRuntimeState &source, int cachedLen) const;
        int RunMtpDraft(MtpRuntimeState &state,
                        const Data &targetHiddenStates,
                        const std::vector<int> &inputTokens,
                        const std::vector<int> &positions,
                        Data *nextMultiHidden,
                        bool sampleToken,
                        const Data *deviceTokenIds = nullptr,
                        Data *sampledTokenIds = nullptr,
                        Data *sampledTokenValues = nullptr,
                        int sampledTokenOffset = 0);
        void CaptureRequestRuntimeCheckpoint(
            RequestState &state,
            const std::map<int, int> &qsaLengths,
            RequestRuntimeCheckpoint &checkpoint);
        void RestoreRequestRuntimeCheckpoint(
            RequestState &state,
            const RequestRuntimeCheckpoint &checkpoint,
            bool restoreQsa = true);
        void CaptureTargetRuntimeCheckpoint(
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &state,
            TargetRuntimeCheckpoint &checkpoint,
            bool synchronize = true,
            bool captureLinearRecurrent = true);
        void CommitTargetRecurrentState(
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const TargetRuntimeCheckpoint &checkpoint);
        void CommitTargetVerificationPrefix(
            const Data &candidateIds,
            const Data &candidatePositionIds,
            const std::vector<int> &candidateTokens,
            int committedInputs,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &state,
            const TargetRuntimeCheckpoint &checkpoint,
            TargetVerificationCapture &capture);
        std::vector<int> ForwardTarget(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<std::vector<float> *> *logits,
            Data *targetMultiHidden,
            std::vector<int> *allVerificationTokens,
            std::vector<unsigned char> *verificationAccepted,
            TargetVerificationCapture *verificationCapture,
            bool recordPrefixSnapshot,
            bool allowDecodeCudaGraph,
            bool decodeEquivalentGdn,
            const std::vector<int> *hostInputTokens = nullptr,
            bool materializeCausalMaskOnGraphFallback = false,
            const Data *precomputedEmbedding = nullptr);

        bool TryRunDecodeCudaGraphBackbone(
            int graphStartLayer,
            bool startBeforeAttention,
            const Data &hiddenStates,
            const Data &attentionOutput,
            const Data &attentionInjection,
            const Data &attentionMask,
            const Data &positionIds,
            bool qsaDeviceCompatibleMask,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &requestState,
            Data &logits,
            Data *targetMultiHidden,
            TargetVerificationCapture *verificationCapture);

        std::shared_ptr<PrefixSnapshot> FindPrefixSnapshotLocked(
            const std::vector<int> &tokens, int maxCachedLen,
            int exactLen = -1) const;
        bool ShouldRecordPrefixSnapshot(
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            const RequestState &state, int &cachedLen) const;
        void MaybeRecordPrefixSnapshot(
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &state);
        bool RestorePrefixSnapshot(
            ResponseContext *context,
            const std::shared_ptr<PrefixSnapshot> &snapshot);

        void DumpTensorIfRequested(const std::string &name, const Data &data) const;
    };
}

#endif // FASTLLM_QWEN4_EXP_H
