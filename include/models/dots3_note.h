#ifndef FASTLLM_DOTS3_NOTE_H
#define FASTLLM_DOTS3_NOTE_H

#include "basellm.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fastllm {
    class Dots3NoteModel : public basellm {
    public:
        Dots3NoteModel();
        ~Dots3NoteModel() override;

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

        bool NeedAttentionMask(int qlen, int klen) override;
        void WarmUp() override;

        int GetKVCacheRetainedTokens(int layer) const override;

        void OnResponseContextCreated(ResponseContext *context) override;
        void OnResponseContextRemoved(ResponseContext *context) override;
        bool TryRestoreHistoryCache(std::vector<int> &inputTokens,
                                    int &cacheLen) override;
        void TryRecordResponseContext(ResponseContext *context) override;

        // Sliding layers retain an absolute suffix rather than token zero, so
        // they need the model-specific snapshot cache below.
        bool UseGenericHistoryCache() const override { return false; }

        std::string MakeInput(const std::string &history, int round,
                              const std::string &input) override;
        std::string MakeHistory(const std::string &history, int round,
                                const std::string &input,
                                const std::string &output) override;

    private:
        struct AttentionConfig {
            int numHeads;
            int qLoraRank;
            int kvLoraRank;
            int qkNopeHeadDim;
            int qkRopeHeadDim;
            int vHeadDim;
            float ropeTheta;
        };

        struct IndexerLayerCache {
            // Raw E4M3 payload bytes; dynamic per-token scales are stored
            // independently in scales.
            Data keys;
            Data scales;

            IndexerLayerCache() : keys(DataType::INT8),
                                  scales(DataType::FLOAT32) {}
        };

        struct IndexerCacheMemory {
            std::vector<IndexerLayerCache> layers;
        };

        struct HistoryCacheMemory {
            std::vector<int> tokens;
            std::vector<std::pair<Data, Data>> pastKeyValues;
            std::shared_ptr<IndexerCacheMemory> indexerCache;
            int sequenceLength = 0;
            long long flushTime = 0;
        };

        AttentionConfig GetAttentionConfig(int layer) const;
        bool IsFullAttentionLayer(int layer) const;
        void BuildRotaryTable(float theta, int positions,
                              Data &sinTable, Data &cosTable);
        void EnsureRotaryTableCapacity(int positions);
        void GetRotaryTableForDevice(bool fullAttention,
                                     const std::string &device,
                                     Data *&sinTable, Data *&cosTable);
        void InitMoeWeightViews();
        std::shared_ptr<IndexerCacheMemory> GetOrCreateIndexerCache(
            const std::vector<std::pair<Data, Data>> *pastKeyValues);

        void MaybeRecordPromptHistoryCache(
                std::vector<std::pair<Data, Data>> &pastKeyValues);
        void RecordHistoryCache(
                const std::vector<int> &tokens,
                const std::vector<std::pair<Data, Data>> &pastKeyValues,
                int sequenceLength);
        bool CanRestoreHistoryCache(const HistoryCacheMemory &memory,
                                    int restoreLength) const;
        void RestoreHistoryCache(const HistoryCacheMemory &memory,
                                 int restoreLength,
                                 ResponseContext *context);

        int fullNumHeads = 128;
        int fullQLoraRank = 1024;
        int fullKVLoraRank = 512;
        int fullQKNopeHeadDim = 128;
        int fullQKRopeHeadDim = 64;
        int fullVHeadDim = 128;
        float fullRopeTheta = 80000000.0f;

        int swaNumHeads = 64;
        int swaQLoraRank = 1024;
        int swaKVLoraRank = 1024;
        int swaQKNopeHeadDim = 192;
        int swaQKRopeHeadDim = 64;
        int swaVHeadDim = 128;
        float swaRopeTheta = 50000.0f;

        int shortContextLimit = 513;
        int indexHeads = 64;
        int indexHeadDim = 128;
        int indexTopK = 2048;
        int rotaryCapacity = 0;

        Data fullSinData, fullCosData;
        Data swaSinData, swaCosData;
        std::map<std::string, Data> deviceFullSinData, deviceFullCosData;
        std::map<std::string, Data> deviceSwaSinData, deviceSwaCosData;

        std::vector<std::vector<Data *>> moeWeights;
        std::vector<std::vector<Data *>> moeBiass;

        std::map<std::vector<int>, std::shared_ptr<HistoryCacheMemory>>
            historyCache;
        std::shared_ptr<HistoryCacheMemory> pendingHistoryCache;
        std::map<const std::vector<std::pair<Data, Data>> *, ResponseContext *>
            responseContexts;
        std::map<const std::vector<std::pair<Data, Data>> *,
                 std::shared_ptr<IndexerCacheMemory>> indexerCaches;
        std::mutex historyCacheMutex;
        std::mutex responseContextsMutex;
        std::mutex indexerCacheMutex;
        long long historyCacheFlushTime = 0;
        int historyCacheMaxRecords = 5;
    };
}

#endif
