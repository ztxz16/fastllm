//
// Qwen4-Exp / Qwen3.8-Flash-Next text model.
//

#ifndef FASTLLM_QWEN4_EXP_H
#define FASTLLM_QWEN4_EXP_H

#include "qwen3_next.h"
#include "utils/execution_mode.h"
#include "devices/disk/diskdevice.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace fastllm {
    class CudaChunkedPrefillPipeline;
    class HybridMoePeerRuntime;
    class Qwen4ExpModel : public Qwen3NextModel {
    public:
        Qwen4ExpModel();
        ~Qwen4ExpModel() override;

        void InitParams() override;
        bool RetainCudaWorkspace() const override;
        ExecutionMode GetExecutionMode() const { return executionMode; }
        void ConfigureKvMem(const KvMemConfig &config) override;
        bool SupportsKvMem() const override { return executionMode == ExecutionMode::Hybrid; }
        void ValidateKvMemModel(const KvMemConfig &config) const override;

        std::map<std::string, std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &tensorNames) override;
        void OnWeightLoaded(const std::string &weightName,
                            const std::set<std::string> &finishedWeightNames) override;
        void OnModelWeightsLoaded() override;
        int GetWeightLoadPriority(const std::string &tensorName,
                const std::vector<std::pair<std::string, DataType>> &mappedWeights) const override;
        bool ShouldLoadWeightSeriallyBeforeOthers(const std::string &tensorName,
                const std::vector<std::pair<std::string, DataType>> &mappedWeights) const override;
        void OnWeightLoadGroupStarted(const std::set<std::string> &weightNames) override;
        void OnWeightLoadGroupFinished() override;
        bool ShouldDelaySpecialWeightNumaRegistration(
                const std::string &weightName) const override;
        bool ShouldDelaySpecialWeightCudaMove(
                const std::string &weightName) const override;
        std::string SelectSpecialWeightDevice(const std::string &weightName,
                int layerId) const override;

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

        bool TryForwardChunkedPrefill(
                const Data &inputIds, const Data &attentionMask, const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                std::vector<float> *logits, int &outputToken) override;

        void WarmUp() override;
        PagedCacheManager *GetPagedKVCacheManager(int layerIndex, bool isKey) const override;
        std::vector<std::pair<int, PagedCacheManager *>> GetPagedKVCacheManagers(
                int layerIndex, bool isKey) const override;

        bool TryRestoreHistoryCache(std::vector<int> &inputTokens,
                                    int &cacheLen) override;
        void OnResponseContextCreated(ResponseContext *context) override;
        void OnResponseContextRemoved(ResponseContext *context) override;
        bool UseGenericHistoryCache() const override { return false; }

    private:
        friend struct Qwen4PrefixCacheTestAccess;
        friend struct Qwen4GGUFTestAccess;
        struct PrefixSnapshot;
        struct DecodeCudaGraphState;
        struct DecodeGraphExecution;
        struct DecodeGraphPolicy { bool hybridDense = false; bool eligible = false; };
        DecodeGraphPolicy GpuDecodeGraphPolicy() const;
        DecodeGraphPolicy HybridDecodeGraphPolicy(const Data &hiddenStates, bool mtpTargetGraph);
        struct PleStagingState;
        struct MtpDraftCudaGraphState;
        struct QsaHostMirrorTransfer;
        struct ServingCache;
        struct MtpRuntimeState;
        struct HybridTpState;
        struct ThreadTpState;
        struct DecodeServingSlot;
        struct RequestState;
        struct PrefillStageContext;
        std::map<const Data *, std::unique_ptr<DecodeServingSlot>> decodeServingSlots;
        std::unique_ptr<DecodeServingSlot> idleDecodeServingSlot;
        std::mutex decodeServingMutex;
        void ReleaseDecodeServingSlot(std::unique_ptr<DecodeServingSlot> &slot);
        void RemoveDecodeServingRequest(const Data *key, bool reusable);
        void ResetDecodeServingState(RequestState &state, bool freezeGraphs = false);
        bool WithDecodeServingCache(std::vector<std::pair<Data, Data>> &cache,
                const std::function<bool(std::vector<std::pair<Data, Data>> &)> &forward);
        std::vector<int> ForwardBatchInternal(
                int batch, const Data &inputIds, const Data &attentionMask,
                const Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                std::vector<std::vector<float> *> *retLogits,
                PrefillStageContext *prefill = nullptr);
        std::unique_ptr<ThreadTpState> threadTpState;
        ThreadTpState *threadTpOwner = nullptr;
        int threadTpRank = -1;
        void InitThreadTp();
        int StreamingThreadTpExpertLayer(const std::string &weightName) const;
        int StreamingThreadTpReplicaLayer(const std::string &weightName) const;
        bool IsStreamingHybridBackboneWeight(const std::string &weightName) const;
        bool loadingHybridBackbone = false;
        void FinishHybridBackboneLoad();
        int SplitThreadTpWeight(const std::string &name, Data &source);
        void StreamHybridThreadTpWeight(const std::string &name, Data &source);
        void PrepareThreadTp();
        void ThreadTpAllReduce(Data &data);
        bool ThreadTpAllTrue(bool value);
        void RemoveThreadTpRequest(const Data *key, bool completed = false);
        std::vector<int> ForwardThreadTp(
                int batch, const Data &inputIds, const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data>> &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector<std::vector<float> *> *logits,
                const Data *precomputedEmbedding = nullptr);

        struct RequestState {
            bool reusableGraphRequest = false;
            bool decodeGraphsFrozen = false;
            // Keep the model's preallocated pools alive while this request uses them.
            std::shared_ptr<ServingCache> servingCache;
            std::shared_ptr<PleStagingState> pleStaging;
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
            // TP dense graphs retain the legacy attention padding width even
            // when physical KV storage is reserved or reused across requests.
            int denseGraphWidth = 0;
            // Token IDs cannot identify image/video embeddings or M-RoPE state.
            // Keep this flag through decode, including direct C++ forwards.
            bool hasMultimodalInput = false;
            std::vector<int> processedTokens;
            int prefixRequestId = 0;
            int lastPrefixSnapshotLen = 0;
            // Keeps snapshot storage alive while restoring fixed linear slots
            // and request-local QSA state. Nonlinear KV retains page references
            // and detaches only a shared partial page before append.
            std::shared_ptr<PrefixSnapshot> borrowedPrefixSnapshot;
            // The one-layer MTP cache is request-local. Prefix snapshots keep
            // a compact, independently owned copy only when it is aligned at
            // the token boundary immediately preceding the deferred target
            // hidden state.
            std::shared_ptr<MtpRuntimeState> mtpState;
            bool mtpDisabled = false;
        };

        struct PrefillStageContext {
            int firstLayer = 0, endLayer = 0;
            bool sampleOutput = false;
            Data *hidden = nullptr;
            std::mutex *hostExpertMutex = nullptr;
            RequestState *state = nullptr;
        };
#ifdef USE_CUDA
        std::unique_ptr<CudaChunkedPrefillPipeline> prefillPipeline;
#ifndef USE_ROCM
        using HybridPeerState = HybridMoePeerRuntime;
        std::unique_ptr<HybridPeerState> hybridPeer;
        bool TryRunPipelineMoe(int layer, const Data &input, const Data &index,
                const Data &score, Data &output, std::vector<Data *> &moeWeights,
                const std::function<void()> &sharedExpert, Data **deferredPeer, bool preparedReadback);
        bool TryPipelineMoeReadback(const Data &input, const Data &index, const Data &score);
#endif
#endif
        std::map<int, Data> prefillLinearInvScales;

        struct ServingCache {
            using Pools = std::pair<std::shared_ptr<PagedCacheManager>,
                                    std::shared_ptr<PagedCacheManager>>;
            std::vector<Pools> layers;
            Pools mtp;
            // KVMem admits one request. Retain its pinned backing and bounded
            // device allocation after warmup; each new request resets the map.
            std::map<int, std::shared_ptr<KvMemCache>> kvMemStorage;
            // One warmup allocation per layer, moved into the first request
            // that needs it. Mutable QSA history is never shared by requests.
            std::map<int, std::shared_ptr<QsaHostMirrorTransfer>> qsaHostMirrors;
        };
        std::shared_ptr<ServingCache> servingCache;
        void ClearWarmupCache(std::vector<std::pair<Data, Data>> &cache);
        void ReserveServingCache(std::vector<std::pair<Data, Data>> &warmupCache);
        void PrepareHybridKvMemCache(int layer, const Data &key, int heads,
                                    Data &pastKey, Data &pastValue);
        size_t RetainHybridKvMemStorage(const std::vector<std::pair<Data, Data>> &cache,
                                      const std::shared_ptr<KvMemCache> &mtp,
                                      ServingCache &reserved) const;
        void FinishHybridKvMemWarmup();
        void AcquireServingCache(std::vector<std::pair<Data, Data>> &cache,
                                 RequestState &state, int appendTokens = 1);
        std::shared_ptr<QsaHostMirrorTransfer> &GetQsaHostMirror(
                RequestState &state, int layer);

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
            struct LinearReplayWorkspace {
                Data pointers;
                std::vector<uint8_t> cache;
            };
            std::map<int, LinearReplayWorkspace> linearReplayWorkspaces;
            std::map<int, Data> qsaRawKeys;
            Data pleInput;
            Data pleNormalized;
            Data greedyTokenIds;
            Data greedyTokenValues;
        };

        struct MtpRuntimeState {
            // Pool ownership must outlive destruction of the KV page references.
            RequestState attentionState;
            Data key;
            Data value;
            // Keep sampling outputs resident on CUDA across verifier cycles;
            // the host reads only the compact token-id prefix.
            Data sampledTokenIds;
            Data sampledTokenValues;
            // Actual temperature-scaled draft distribution, as in Qwen3.5.
            Data proposalLogits;
            Data proposalLogsumexp;
            float proposalTemperature = 1.0f;
            bool sampleProposal = false;
            int proposalCount = 0;
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
            // Destroy page references before releasing their pool owner.
            RequestState state;
            std::vector<PrefixLayerSnapshot> layers;
            // A TP hit pins the same token boundary on every rank. These
            // references survive rank-local eviction until restore completes.
            std::vector<std::shared_ptr<PrefixSnapshot>> ranks;
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

        bool preparedWeights = false;
        bool ggufWeightsRestored = false;
        void RestoreGgufWeights();
        void RestoreGdnOutputWeight(Data &weight, int tpSize);
        bool GdnOutputUsesTiledLayout(const Data &projection) const;
        void RunGdnOutputProjection(Data &input, Data &weight, Data &output);
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
        std::shared_ptr<DiskEmbeddingRowReader> pleDiskReader;
        // Host QSA compression must not move the device normalization weights.
        std::map<int, std::vector<float>> qsaKeyNormValues;
        std::vector<Data *> mtpMoeWeights;
        std::vector<Data *> mtpMoeBiass;
        // MTP only needs an approximate proposal distribution: target
        // verification still uses the original lm_head and therefore keeps
        // the generated token stream exact.  A packed FP8 copy cuts the three
        // bandwidth-bound draft head scans in half on supported CUDA models.
        Data mtpDraftLmHeadWeight;
        Data mtpDraftSubsetHead;
        std::vector<int> mtpDraftSubsetIds;
        int mtpDraftSubsetOffset = 0;
        bool mtpDraftSubsetAttempted = false;
        bool mtpDraftLmHeadReady = false;
        bool mtpDraftLmHeadAttempted = false;

        void PrepareWeights();
        void PrepareMtpDraftLmHeadWeight();
        void PrepareMtpDraftSubsetHead();
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

        int64_t PLEHashRow(const int *shifted, int head) const;
        void RunPLE(const Data &hyperInput, const Data &inputIds,
                    RequestState &state, Data &output,
                    const std::vector<int> *hostInputTokens = nullptr,
                    const DiskEmbeddingRowReader::Ticket *prefetched = nullptr,
                    Data *normalizedCapture = nullptr);
        bool TryCommitPLEPrefix(RequestState &state, const Data &normalized,
                                const std::vector<int> &tokens, int count);
        void RunThreadTpPLE(const Data &hyperInput, const Data &inputIds,
                    RequestState &state, Data &output,
                    const std::vector<int> *hostInputTokens,
                    const DiskEmbeddingRowReader::Ticket *prefetched = nullptr);
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
                              Data *qsaRawKeyCapture = nullptr,
                              int decodeReserveTokens = 0);
        void RunFullAttentionWithPrefix(
                              int stateLayer,
                              const std::string &attentionPrefix,
                              const Data &input,
                              const Data &attentionMask,
                              const Data &positionIds,
                              bool qsaDeviceCompatibleMask,
                              Data &pastKey, Data &pastValue,
                              RequestState &state, Data &output,
                              Data *qsaRawKeyCapture = nullptr,
                              int decodeReserveTokens = 0);
        void RunLinearAttention(int layer, const Data &input,
                                Data &pastConv, Data &pastRecurrent,
                                Data &output,
                                Data *convInputCapture = nullptr,
                                Data *alphaCapture = nullptr,
                                Data *betaCapture = nullptr,
                                Data *recurrentStateOutput = nullptr);
        struct MoEExecutionPlan {
            ExecutionMode backend = ExecutionMode::Cpu;
            int deviceLayer = 0;
            std::string mlpPrefix;
            // Borrow the original tables: caches key on their data() addresses.
            // Device shards and quantization remain in Data; graph capability
            // is checked for the current shape, format and TP configuration.
            std::vector<Data *> *weights = nullptr, *bias = nullptr;
        };
        std::vector<MoEExecutionPlan> targetMoePlans;
        MoEExecutionPlan draftMoePlan;
        void PrepareMoEExecutionPlans();

        struct MoEResult {
            Data sharedOutput;
            // The backend has queued the peer completion wait on the consumer
            // stream. Consume this borrowed result before the next expert call
            // reuses the peer buffer; only the graph boundary performs the join.
            Data *peerOutput = nullptr;
            bool sharedDeferred = false;
            void Reset() { peerOutput = nullptr; sharedDeferred = false; }
            int JoinMask() const { return int(sharedDeferred) + 2 * int(peerOutput != nullptr); }
        };
        struct MoEPrepared {
            Data index, score;
            MoEResult result;
            // Optional stable, graph-owned destination; never owned here.
            Data *outputTarget = nullptr;
            bool readbackPrepared = false;
        };
        void SelectMoEExperts(Data &routerLogits, Data &index, Data &score);
        void RunSharedExpert(const std::string &mlp, Data &input,
                             Data &gateUp, Data &hidden, Data &gate, Data &output);
        void PrepareMoERouting(int layer, const Data &input, MoEPrepared &prepared);
        void RunMoE(int layer, const Data &input, Data &output,
                   bool reduceOutput = true, PrefillStageContext *prefill = nullptr,
                   MoEPrepared *prepared = nullptr,
                   const std::function<void()> &sharedGraph = {});
        void RunMoE(const MoEExecutionPlan &plan, const Data &input, Data &output,
                    bool reduceOutput = true, PrefillStageContext *prefill = nullptr,
                    MoEPrepared *prepared = nullptr,
                    const std::function<void()> &sharedGraph = {});

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
                        int sampledTokenOffset = 0,
                        float *topProbability = nullptr,
                        std::mutex *embeddingMutex = nullptr);
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
        struct ForwardState;
        struct ForwardWorkspace;
        ExecutionMode executionMode = ExecutionMode::Cpu;
        void InitExecutionMode();
        void PrepareForward(ForwardWorkspace &);
        int BeginForwardLayer(ForwardWorkspace &, int &layer);
        void EndForwardLayer(ForwardWorkspace &, int layer);
        std::vector<int> FinishForward(ForwardWorkspace &);
        struct HybridExecutionScope;
        struct DraftGraphPolicy {
            bool device = false, tp = false, hybrid = false, placementCompatible = false;
        };
        DraftGraphPolicy GpuDraftGraphPolicy(int sequence) const;
        DraftGraphPolicy HybridDraftGraphPolicy(int sequence);
        std::vector<std::vector<int>> HybridWorkerCpuSets(const std::vector<int> &devices) const;
        void PrefetchHybridExperts(const Data &inputIds, bool pipelinedPrefill, int layer, int endLayer);
        void PrepareHybridExperts(bool hasMtpWeights);
        std::vector<int> ForwardGpu(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<std::vector<float> *> *retLogits,
            Data *targetMultiHidden,
            std::vector<int> *allVerificationTokens,
            std::vector<unsigned char> *verificationAccepted,
            TargetVerificationCapture *verificationCapture,
            bool recordPrefixSnapshot,
            bool allowDecodeCudaGraph,
            bool decodeEquivalentGdn,
            const std::vector<int> *hostInputTokens,
            bool materializeCausalMaskOnGraphFallback,
            const Data *precomputedEmbedding, PrefillStageContext *prefill);
        std::vector<int> ForwardHybrid(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<std::vector<float> *> *retLogits,
            Data *targetMultiHidden,
            std::vector<int> *allVerificationTokens,
            std::vector<unsigned char> *verificationAccepted,
            TargetVerificationCapture *verificationCapture,
            bool recordPrefixSnapshot,
            bool allowDecodeCudaGraph,
            bool decodeEquivalentGdn,
            const std::vector<int> *hostInputTokens,
            bool materializeCausalMaskOnGraphFallback,
            const Data *precomputedEmbedding, PrefillStageContext *prefill);
        void RunGpuMoE(const MoEExecutionPlan &plan,
            const Data &input, Data &output, bool reduceOutput, PrefillStageContext *prefill,
            MoEPrepared *prepared, const std::function<void()> &sharedGraph);
        void RunHybridMoE(const MoEExecutionPlan &plan,
            const Data &input, Data &output, bool reduceOutput, PrefillStageContext *prefill,
            MoEPrepared *prepared, const std::function<void()> &sharedGraph);

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
            const Data *precomputedEmbedding = nullptr,
            PrefillStageContext *prefill = nullptr);

        bool TryRunThreadTpPrefixCudaGraph(
            Data &hiddenStates,
            std::vector<std::pair<Data, Data>> &pastKeyValues);
        bool TryRunDecodeCudaGraphBackbone(
            ExecutionMode graphMode,
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
            TargetVerificationCapture *verificationCapture,
            int graphEndLayer = -1, Data *stageOutput = nullptr);

        std::shared_ptr<PrefixSnapshot> FindPrefixSnapshotLocked(
            const std::vector<int> &tokens, int maxCachedLen,
            int exactLen = -1) const;
        static std::shared_ptr<PrefixSnapshot> FindCommonPrefixSnapshot(
            const std::vector<Qwen4ExpModel *> &models,
            const std::vector<int> &tokens, int maxCachedLen);
        bool RestoreThreadTpPrefixSnapshot(
            ResponseContext *context,
            const std::shared_ptr<PrefixSnapshot> &snapshot);
        bool ShouldRecordPrefixSnapshot(
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            const RequestState &state, int &cachedLen) const;
        void MaybeRecordPrefixSnapshot(
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &state);
        bool RestorePrefixSnapshot(
            ResponseContext *context,
            const std::shared_ptr<PrefixSnapshot> &snapshot);
        bool RestorePrefixSnapshot(
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const std::shared_ptr<PrefixSnapshot> &snapshot);

        void DumpTensorIfRequested(const std::string &name, const Data &data) const;
    };
}

#endif // FASTLLM_QWEN4_EXP_H
