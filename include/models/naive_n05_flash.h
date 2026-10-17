#ifndef FASTLLM_NAIVE_N05_FLASH_H
#define FASTLLM_NAIVE_N05_FLASH_H

#include "basellm.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include <deque>
#include <random>
#include "utils/persistent_worker_group.h"

struct FastllmCudaMimoEp8Mega;

namespace fastllm {
    class CudaWorkspace;
    class NaiveN05FlashModel : public basellm {
    protected:
        struct TargetCapture;
        struct LogitsSelection;
        struct TargetBatch;
    public:
        NaiveN05FlashModel();
        ~NaiveN05FlashModel() override;
        void InitParams() override;
        void SetIndexerDataType(DataType dataType) override;
        std::map<std::string, std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &names) override;
        int Forward(const Data &inputIds, const Data &attentionMask,
                    const Data &positionIds,
                    std::vector<std::pair<Data, Data>> &pastKeyValues,
                    const GenerationConfig &generationConfig = GenerationConfig(),
                    const LastTokensManager &lastTokens = LastTokensManager(),
                    std::vector<float> *logits = nullptr) override;
        std::vector<int> ForwardBatch(int batch, const Data &inputIds,
                    const std::vector<Data *> &attentionMask,
                    const std::vector<Data *> &positionIds, const std::vector<int> &seqLens,
                    std::vector<std::pair<Data *, Data *>> &pastKeyValues,
                    const std::vector<GenerationConfig> &generationConfigs,
                    const LastTokensManager &lastTokens = LastTokensManager(),
                    std::vector<std::vector<float> *> *logits = nullptr) override;
        Data ForwardSingleGPU(int rank, const Data &inputIds, const Data &positions,
                              std::vector<std::pair<Data, Data>> &kv,
                              const GenerationConfig &config, const Data *embedding = nullptr,
                              TargetCapture *capture = nullptr, float *candidateResult = nullptr,
                              const LogitsSelection *selection = nullptr, const TargetBatch *batch = nullptr);
        bool NeedAttentionMask(int, int) override { return false; }
        bool SupportsPrefillBatching() const override;
        // Bound idle TP prefill workspace with the shared pressure-aware pool.
        bool RetainCudaWorkspace() const override { return tpDevices.size() > 1; }
        // Prefix reuse requires both resident full-attention pages and a GPU SWA checkpoint.
        bool UseGenericHistoryCache() const override { return false; }
        bool TryRestoreHistoryCache(std::vector<int> &tokens, int &cacheLen) override;
        bool DeferHistoryCacheRestore() const override { return !tpPagedCaches.empty(); }
        PagedCacheManager *GetPagedKVCacheManager(int layer, bool key) const override;
        std::vector<std::pair<int, PagedCacheManager *>> GetPagedKVCacheManagers(int layer, bool key) const override;
        bool TryRecordPagedPrefixCacheExtra(ResponseContext *context) override;
        int QueryPagedPrefixCacheExtra(ResponseContext *context, int length) const override;
        bool RestorePagedPrefixCacheExtra(ResponseContext *context, int length) const override;
        void TryRecordResponseContext(ResponseContext *context) override;
        void OnResponseContextCreated(ResponseContext *context) override;
        void OnResponseContextRemoved(ResponseContext *context) override;
        bool SetSaveHistoryChat(bool save) override;
        void AddPromptCache(const std::vector<int> &tokens) override;
        int GetKVCacheRetainedTokens(int layer) const override;
        void WarmUp() override;
        int Ep8ExpertDevice(int expert, const std::vector<int> &loadDevices) const {
            AssertInFastLLM(tpEp8Mega && loadDevices == tpDevices && expert >= 0 && expert < 256,
                           "MiMo EP8 expert loading must use the same ordered eight devices as TP.");
            return tpDevices.at(expert / 32);
        }
        std::string MakeInput(const std::string &history, int,
                              const std::string &input) override { return history + input; }
        std::string MakeHistory(const std::string &history, int,
                                const std::string &input,
                                const std::string &output) override { return history + input + output; }

    protected:
        // Batch-major request descriptors, borrowed only for one forward. KV
        // storage remains owned by each response; weights are shared by all rows.
        struct TargetBatch {
            const std::vector<std::pair<Data *, Data *>> &kv;
            const std::vector<GenerationConfig> &configs;
            const std::vector<int> &lengths;
            int layers;
            int Size() const { return lengths.size(); }
            bool Decode() const {
                return std::all_of(lengths.begin(), lengths.end(), [](int n) { return n == 1; });
            }
            Data &Key(int sequence, int layer) const { return *kv[sequence * layers + layer].first; }
            Data &Value(int sequence, int layer) const { return *kv[sequence * layers + layer].second; }
        };
        // Only window payloads live here; full DSA/indexer KV stays in the page pool.
        struct PrefixPool {
            std::mutex mutex;
            std::vector<int> freeSlots;
            std::vector<Data> storage;
            std::vector<size_t> offsets;
            size_t slotBytes = 0;
        };
        struct PrefixSnapshot {
            std::shared_ptr<PrefixPool> pool;
            int slot = -1, length = 0;
            std::vector<int> tokens;
            // Observe the resident full-prefix endpoint without pinning KV.
            // Generation checks prevent a recycled physical page from reviving
            // a stale window checkpoint.
            PagedCacheManager *fullPool = nullptr;
            int fullPage = -1;
            long long fullGeneration = 0;
            bool HasResidentPrefix() const {
                return fullPool && fullPool->IsCachedPage(fullPage, fullGeneration);
            }
            ~PrefixSnapshot() {
                if (pool && slot >= 0) {
                    std::lock_guard<std::mutex> lock(pool->mutex);
                    pool->freeSlots.push_back(slot);
                }
            }
        };
        using PrefixSnapshots = std::vector<std::shared_ptr<PrefixSnapshot>>;
        void RetirePrefixSnapshot(std::shared_ptr<PrefixSnapshot> &snapshot);
        PrefixSnapshots BeginPrefixSnapshots(const Data &key, int past, int length);
        void FinishPrefixSnapshots(const Data &key, const PrefixSnapshots &snapshots, bool prefill);
        void CopyPrefixWindow(const PrefixSnapshot &snapshot, int rank, int layer,
                              const Data &key, const Data &value, int first, bool ring);

        struct LogitsSelection {
            int count = 0;
            bool greedy = false;
            bool cudaSampling = false;
            float invTemperature = 1.0f;
            Data candidates; // CPU [rows, count * 2], (global token ID, score).
            std::vector<int> sampledTokens;
        };
        LogitsSelection SelectLogits(const GenerationConfig &config, bool speculative = false) const;
        std::vector<int> SampleCudaLogits(Data &logits,
                                        const std::vector<GenerationConfig> &configs);
        struct TargetCapture {
            LogitsSelection *selection = nullptr;
            bool verifying = false;
            bool collectHidden = true;
            std::map<int, Data> hidden;
            std::vector<PrefixSnapshots> prefixes;
        };
        struct DraftTPState;
        std::shared_ptr<DraftTPState> draftTP;
        struct DraftWorkspace;
        struct DraftContext {
            int committed = 0;
            std::vector<std::pair<Data, Data>> kv;
            std::shared_ptr<DraftWorkspace> workspace;
            std::deque<std::pair<int, int>> pending;
            std::mt19937_64 random{std::random_device{}()};
            uint64_t rounds = 0, proposed = 0, accepted = 0;
            double Uniform() { return std::generate_canonical<double, 53>(random); }
        };
        struct TargetWorkspace;
        Data RunTarget(const Data &inputIds, const Data &positions,
                       std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config,
                       TargetCapture *capture, int tpRank = -1, const Data *embedding = nullptr,
                       TargetWorkspace *workspace = nullptr, const TargetBatch *batch = nullptr);
        int CacheReserveCapacity(const GenerationConfig &config) const;
        void AppendCache(Data &cache, Data &input, DataType cacheType, int reserveCapacity = 0);
        void ReserveCache(Data &cache, int capacity, int width);
        void PreparePagedCache(Data &cache, PagedCacheManager &manager, int end);
        static void TrimCache(Data &cache, int length);
        int SampleTarget(Data &logits, std::vector<std::pair<Data, Data>> &kv,
                         const GenerationConfig &config, const LastTokensManager &lastTokens,
                         std::vector<float> *retLogits, LogitsSelection *selection = nullptr);
        void InitDraft();
        void AppendDraftContext(Data &hidden, int start, DraftContext &context);
        bool AppendDraftContextFused(Data &hidden, int start, DraftContext &context);
        void CommitDraftContext(TargetCapture &capture, int tokens, DraftContext &context,
                                std::vector<std::pair<Data, Data>> &kv);
        std::shared_ptr<DraftContext> CreateDraftContext();
        bool PrepareDraftTP();
        bool AppendDraftContextTP(Data &hidden, int start, DraftContext &context);
        bool RunDraftTP(int anchor, DraftContext &context, Data &output);
        bool RunDraftGraph(int anchor, DraftContext &context, Data &output);
        bool RunDraftProposalGraph(int anchor, const Data &baseLogits,
                                   DraftContext &context, std::vector<int> &proposed);
        Data RunDraft(int anchor, DraftContext &context);
        void ApplyDraftDevice();
        Data &DraftWeight(const std::string &name);
        Data RunDraftHead(Data &hidden);
        Data RunDraftTarget(const Data &inputIds, const Data &positions,
                            std::vector<std::pair<Data, Data>> &kv,
                            const GenerationConfig &config, TargetCapture &capture);
        void CommitTargetCache(std::vector<std::pair<Data, Data>> &kv, int past, int count);
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
        std::shared_ptr<DraftContext> idleDraftContext;

    private:
        // Only request lifecycle callbacks (serialized by dictLocker) own this
        // single idle allocation set. No token contents are reused.
        std::vector<std::pair<Data, Data>> tpIdleCache;
        bool CanReuseTensorParallelCache(const ResponseContext *context) const;
        void RestoreTensorParallelCache(ResponseContext *context);
        void RecycleTensorParallelCache(ResponseContext *context);
        bool InitTensorParallel();
        void PrepareTensorParallel();
        void InitTensorParallelKVPool();
        void PrepareSwaPrefillWorkspace(int rank, int length);
        struct TPDecodeState;
        std::shared_ptr<TPDecodeState> tpDecodeState, tpVerifyState, tpBatchState;
        struct TPServingState;
        std::shared_ptr<TPServingState> tpServingState;
        std::atomic<uint64_t> tpCacheEpoch{0};
        bool CanReuseTensorParallelServingCache(const TargetBatch *batch) const;
        void PrepareTensorParallelServing();
        bool PrepareTensorParallelServingDecode(const Data &inputIds,
            const LogitsSelection &selection, std::vector<int *> &cacheLengths,
            const TargetBatch *batch, bool collectHidden);
        bool HasVerificationGraph() const;
        bool PrepareTensorParallelDecode(const Data &inputIds,
                                        std::vector<std::pair<Data, Data>> &kv, bool verifying,
                                        const LogitsSelection &selection, std::vector<int *> &cacheLengths,
                                        const TargetBatch *batch = nullptr,
                                        bool collectHidden = false);
        Data ForwardTensorParallelDecode(int rank, const Data &inputIds, const Data &positions,
                                        std::vector<std::pair<Data, Data>> &kv,
                                        const GenerationConfig &config, const Data *embedding, TargetCapture *capture,
                                        float *candidateResult, const TargetBatch *batch = nullptr,
                                        bool keepCudaLogits = false);
        Data ForwardTensorParallel(const Data &inputIds, const Data &positions,
                                   std::vector<std::pair<Data, Data>> &kv,
                                   const GenerationConfig &config, TargetCapture *capture = nullptr,
                                   LogitsSelection *selection = nullptr, const TargetBatch *batch = nullptr);
        std::vector<int> tpDevices;
        std::map<int, std::shared_ptr<CudaWorkspace>> tpKVPool;
        // One chronological prefill workspace per GPU, reused across layers
        // and requests. Request-owned SWA storage only holds the sliding window.
        std::vector<std::pair<Data, Data>> tpSwaPrefillWorkspaces;
        std::vector<std::vector<std::unique_ptr<PagedCacheManager>>> tpPagedCaches;
        bool tpPrepared = false;
        // Fixed at TP preparation so toggling CUDA Graph cannot change live KV
        // layout. Prefill/eager and verify temporarily use chronological rows.
        bool tpSwaRingCache = false;
        bool tpEp8Mega = false;
        std::shared_ptr<FastllmCudaMimoEp8Mega> tpEp8MegaState;
        std::vector<std::shared_ptr<TargetWorkspace>> tpVerifyWorkspaces;
        std::vector<std::shared_ptr<TargetWorkspace>> tpBatchWorkspaces;
        std::vector<Data> tpDraftHeadInputs, tpDraftHeadLogits;
        Data tpDraftHeadOutput;
        PersistentWorkerGroup tpWorkers;
        std::vector<std::vector<std::vector<Data *>>> tpMoeWeights, tpMoeBiases;
        std::vector<std::pair<int, int>> tpVocabRanges;
        struct ActivePrefixes {
            PrefixSnapshots prefill;
            std::shared_ptr<PrefixSnapshot> decode;
        };
        std::shared_ptr<PrefixPool> prefixPool;
        mutable std::mutex historyMutex;
        PrefixSnapshots prefixHistory; // Oldest first, bounded by GPU pool slots.
        std::map<const Data *, ActivePrefixes> activePrefixes;

        struct AttentionConfig {
            int heads, kvHeads, headDim, valueDim;
            float theta;
        };
        AttentionConfig full, sliding;
        std::vector<int> slidingLayers, moeLayers;
        int window = 128, indexHeads = 16, indexDim = 128, indexTopK = 2048;
        float partialRotary = 0.334f, valueScale = 0.707f;
        bool indexFp8 = true;
        bool fp8IndexerCache = false;
        int IndexerCacheWidth(DataType type) const {
            return fp8IndexerCache ? 144 / (type == FP8_E4M3 ? 1 : 2) : indexDim;
        }
        std::vector<std::vector<Data *>> moeWeights, moeBiases;
    };
}
#endif
