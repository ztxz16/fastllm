//
// Qwen4-Exp / Qwen3.8-Flash-Next text model.
//

#ifndef FASTLLM_QWEN4_EXP_H
#define FASTLLM_QWEN4_EXP_H

#include "qwen3_next.h"

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
        void OnModelWeightsLoaded() override;

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

        void WarmUp() override;

        bool TryRestoreHistoryCache(std::vector<int> &inputTokens,
                                    int &cacheLen) override;
        void OnResponseContextCreated(ResponseContext *context) override;
        void OnResponseContextRemoved(ResponseContext *context) override;
        bool UseGenericHistoryCache() const override { return false; }

    private:
        struct PrefixSnapshot;
        struct DecodeCudaGraphState;
        struct QsaHostMirrorTransfer;

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

        bool preparedWeights = false;
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

        void PrepareWeights();
        bool IsLinearAttentionLayer(int layer) const;

        void GroupedRMSNorm(const Data &input, Data &normWeight, Data &output);
        void HyperMixNormalized(Data &normalized,
                                const std::string &prefix,
                                Data &mixedInput,
                                Data *injectionWeights);
        void HyperCombine(const Data &hyperInput, const Data &blockOutput,
                          const Data &injectionWeights, Data &output);
        void HyperCombineRMSNorm(const Data &hyperInput,
                                 const Data &blockOutput,
                                 const Data &injectionWeights,
                                 Data &normWeight,
                                 Data &output,
                                 Data &normalized);

        void RunPLE(const Data &hyperInput, const Data &inputIds,
                    RequestState &state, Data &output);
        void MaterializePLEHostHistory(RequestState &state);
        void BuildQSAMask(int layer, const Data &input,
                          const Data &baseMask, const Data &positionIds,
                          int previousLength, bool deviceCompatibleMask,
                          RequestState &state,
                          Data &qsaMask, Data &qsaIndices);
        void MaterializeQsaHostHistory(int layer, int length,
                                       RequestState &state);
        void RunFullAttention(int layer, const Data &input,
                              const Data &attentionMask,
                              const Data &positionIds,
                              bool qsaDeviceCompatibleMask,
                              Data &pastKey, Data &pastValue,
                              RequestState &state, Data &output);
        void RunLinearAttention(int layer, const Data &input,
                                Data &pastConv, Data &pastRecurrent,
                                Data &output);
        void RunMoE(int layer, const Data &input, Data &output);

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
            Data &logits);

        std::shared_ptr<PrefixSnapshot> FindPrefixSnapshotLocked(
            const std::vector<int> &tokens, int maxCachedLen,
            int exactLen = -1) const;
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
