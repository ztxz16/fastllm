#ifndef FASTLLM_NAIVE_N05_FLASH_H
#define FASTLLM_NAIVE_N05_FLASH_H

#include "basellm.h"
#include <memory>
#include <deque>
#include <random>

namespace fastllm {
    class NaiveN05FlashModel : public basellm {
    public:
        NaiveN05FlashModel();
        ~NaiveN05FlashModel() override;
        void InitParams() override;
        std::map<std::string, std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &names) override;
        int Forward(const Data &inputIds, const Data &attentionMask,
                    const Data &positionIds,
                    std::vector<std::pair<Data, Data>> &pastKeyValues,
                    const GenerationConfig &generationConfig = GenerationConfig(),
                    const LastTokensManager &lastTokens = LastTokensManager(),
                    std::vector<float> *logits = nullptr) override;
        bool NeedAttentionMask(int, int) override { return false; }
        // The history archive also retains keys discarded by sliding attention.
        bool UseGenericHistoryCache() const override { return false; }
        bool TryRestoreHistoryCache(std::vector<int> &tokens, int &cacheLen) override;
        void TryRecordResponseContext(ResponseContext *context) override;
        void OnResponseContextCreated(ResponseContext *context) override;
        void OnResponseContextRemoved(ResponseContext *context) override;
        bool SetSaveHistoryChat(bool save) override;
        void AddPromptCache(const std::vector<int> &tokens) override;
        int GetKVCacheRetainedTokens(int layer) const override;
        void WarmUp() override;
        std::string MakeInput(const std::string &history, int,
                              const std::string &input) override { return history + input; }
        std::string MakeHistory(const std::string &history, int,
                                const std::string &input,
                                const std::string &output) override { return history + input + output; }

    protected:
        struct HistoryChunk {
            int length = 0;
            size_t bytes = 0;
            std::vector<std::pair<Data, Data>> layers;
            Data draftHidden;
        };
        std::shared_ptr<HistoryChunk> BeginHistoryChunk(
            const std::vector<std::pair<Data, Data>> &kv, int past, int length);
        static void CopyHistoryTensor(const Data &source, Data &target, int length);
        void FinishHistoryChunk(const std::vector<std::pair<Data, Data>> &kv,
                                const std::shared_ptr<HistoryChunk> &chunk);

        struct TargetCapture {
            bool verifying = false;
            std::map<int, Data> hidden;
            std::shared_ptr<HistoryChunk> history;
        };
        struct DraftContext {
            int committed = 0;
            std::vector<std::pair<Data, Data>> kv;
            Data restoredHidden;
            std::deque<std::pair<int, int>> pending;
            std::mt19937_64 random{std::random_device{}()};
            uint64_t rounds = 0, proposed = 0, accepted = 0;
            double Uniform() { return std::generate_canonical<double, 53>(random); }
        };
        Data RunTarget(const Data &inputIds, const Data &positions,
                       std::vector<std::pair<Data, Data>> &kv, TargetCapture *capture);
        static void AppendCache(Data &cache, Data &input);
        static void TrimCache(Data &cache, int length);
        int SampleTarget(Data &logits, std::vector<std::pair<Data, Data>> &kv,
                         const GenerationConfig &config, const LastTokensManager &lastTokens,
                         std::vector<float> *retLogits);
        void InitDraft();
        void AppendDraftContext(Data &hidden, int start, DraftContext &context);
        void CommitDraftContext(TargetCapture &capture, int tokens, DraftContext &context,
                                std::vector<std::pair<Data, Data>> &kv);
        Data RunDraft(int anchor, DraftContext &context);
        int ForwardDraft(const Data &inputIds, const Data &positions,
                         std::vector<std::pair<Data, Data>> &kv,
                         const GenerationConfig &config, const LastTokensManager &lastTokens,
                         std::vector<float> *retLogits);
        bool draftEnabled = false;
        int draftLayers = 0, draftBlock = 0, draftTokens = 0;
        int draftHeads = 0, draftKvHeads = 0, draftHeadDim = 0, draftWindow = 0;
        float draftEps = 1e-5f, draftTheta = 10000;
        float draftConfidenceThreshold = 0.5f;
        std::vector<int> draftTargetLayers;
        std::map<const std::vector<std::pair<Data, Data>> *, std::shared_ptr<DraftContext>> draftContexts;

    private:
        struct HistorySpan {
            std::shared_ptr<const HistoryChunk> chunk;
            int length;
        };
        struct HistoryMemory {
            std::vector<int> tokens;
            std::vector<HistorySpan> spans;
            int length = 0;
            size_t bytes = 0;
        };
        // Archives use host memory, never a second persistent GPU KV copy.
        // An active request may build one additional archive of this size.
        static constexpr size_t historyByteLimit = 1ULL << 30;
        static constexpr size_t historyRecordLimit = 5;
        size_t historyBytesPerToken = 0;
        std::mutex historyMutex;
        // Oldest first; completed records and their chunks are immutable.
        std::vector<std::shared_ptr<const HistoryMemory>> history;
        // LaunchResponseTokens holds dictLocker across lookup and creation.
        std::shared_ptr<const HistoryMemory> pendingHistory;
        std::map<const std::vector<std::pair<Data, Data>> *,
                 HistoryMemory> activeHistory;

        struct AttentionConfig {
            int heads, kvHeads, headDim, valueDim;
            float theta;
        };
        AttentionConfig full, sliding;
        std::vector<int> slidingLayers, moeLayers;
        int window = 128, indexHeads = 16, indexDim = 128, indexTopK = 2048;
        float partialRotary = 0.334f, valueScale = 0.707f;
        bool indexFp8 = true;
        std::vector<std::vector<Data *>> moeWeights, moeBiases;
    };
}
#endif
