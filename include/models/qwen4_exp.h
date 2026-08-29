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
        struct RequestState {
            int previousToken1 = -1;
            int previousToken2 = -1;
            std::vector<float> convHistory;
            std::map<int, std::vector<float>> indexerRawKeys;
            std::map<int, std::vector<float>> indexerPositions;
            // Host compressed keys are the snapshot-safe source of truth.
            // The tensor view is rebuilt lazily on the request's execution
            // device and is deliberately excluded from prefix snapshots.
            std::map<int, std::vector<float>> indexerBlockKeys;
            std::map<int, std::shared_ptr<Data>> indexerBlockKeyTensors;
            std::vector<int> processedTokens;
            int prefixRequestId = 0;
            int lastPrefixSnapshotLen = 0;
        };

        struct PrefixLayerSnapshot {
            bool linear = false;
            Data first;
            Data second;
            DataDevice firstDevice = DataDevice::CPU;
            DataDevice secondDevice = DataDevice::CPU;
            std::vector<int> firstDeviceIds;
            std::vector<int> secondDeviceIds;
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
        void HyperMix(const Data &hyperInput, const std::string &prefix,
                      Data &mixedInput, Data *injectionWeights);
        void HyperCombine(const Data &hyperInput, const Data &blockOutput,
                          const Data &injectionWeights, Data &output);

        void RunPLE(const Data &hyperInput, const Data &inputIds,
                    RequestState &state, Data &output);
        void BuildQSAMask(int layer, const Data &input,
                          const Data &baseMask, const Data &positionIds,
                          int previousLength, RequestState &state,
                          Data &qsaMask, Data &qsaIndices);
        void RunFullAttention(int layer, const Data &input,
                              const Data &attentionMask,
                              const Data &positionIds,
                              Data &pastKey, Data &pastValue,
                              RequestState &state, Data &output);
        void RunLinearAttention(int layer, const Data &input,
                                Data &pastConv, Data &pastRecurrent,
                                Data &output);
        void RunMoE(int layer, const Data &input, Data &output);

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
