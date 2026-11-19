#ifndef FASTLLM_NAIVE_N05_FLASH_H
#define FASTLLM_NAIVE_N05_FLASH_H

#include "basellm.h"
#include "parallel/model.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include <deque>
#include <random>
#include "utils/persistent_worker_group.h"

struct FastllmCudaMimoEp8Mega;
namespace fastkv { class HostCache; }

namespace fastllm {
    class CudaWorkspace;
    class PDStateWriter;
    class PDStateReader;
    class NaiveN05FlashModel : public basellm, public parallel::AttentionParallelModel {
    protected:
        static constexpr int MaxTPDecodeBatch = 512;
        struct TargetCapture;
        struct LogitsSelection;
        struct TargetBatch;
    public:
        NaiveN05FlashModel();
        ~NaiveN05FlashModel() override;
        void InitParams() override;
        void ConfigureAttentionParallel(int attentionDP) override;
        void SetIndexerDataType(DataType dataType) override;
        void SetIndexerComputeFP8(bool enabled);
        void SetKVCacheDataType(DataType dataType) override;
        void SetIndexerTPStorage(bool shardByLayer, bool shardByContext = false);
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
        bool PopPendingResponseToken(ResponseContext *context, int previousToken,
                                     int &token) override;
        bool NeedAttentionMask(int, int) override { return false; }
        bool SupportsPrefillBatching() const override;
        int GetPrefillBatchGroups() const override { return attentionDP; }
        bool SupportsChunkedPrefillScheduling() const override { return SupportsPrefillBatching(); }
        int GetPrefillBatchGroup(const ResponseContext *context) const override;
        // Bound idle TP prefill workspace with the shared pressure-aware pool.
        bool RetainCudaWorkspace() const override { return tpDevices.size() > 1; }
        // Prefix reuse requires both resident full-attention pages and a GPU SWA checkpoint.
        bool UseGenericHistoryCache() const override { return false; }
        bool TryRestoreHistoryCache(std::vector<int> &tokens, int &cacheLen) override;
        bool DeferHistoryCacheRestore() const override { return !tpPagedCaches.empty(); }
        PagedCacheManager *GetPagedKVCacheManager(int layer, bool key) const override;
        PagedCacheManager *GetPagedKVCacheBudgetManager(const ResponseContext *context) const override;
        std::vector<std::pair<int, PagedCacheManager *>> GetPagedKVCacheManagersForRequest(
            int layer, bool key, const ResponseContext *context) const override;
        std::vector<std::pair<int, PagedCacheManager *>> GetPagedKVCacheManagers(int layer, bool key) const override;
        bool TryRecordPagedPrefixCacheExtra(ResponseContext *context) override;
        int QueryPagedPrefixCacheExtra(ResponseContext *context, int length) const override;
        int PreviewPagedPrefixCacheExtra(ResponseContext *context, int length,
                                         PrefillPrefixPreview &preview) const override;
        bool RestorePagedPrefixCacheExtra(ResponseContext *context, int length) const override;
        void TryRecordResponseContext(ResponseContext *context) override;
        bool CanAdmitResponseContext(const ResponseContext *context) const override;
        bool TryAdmitResponseContext(ResponseContext *context) override;
        void OnResponseContextCreated(ResponseContext *context) override;
        void OnResponseContextRemoved(ResponseContext *context) override;
        void OnResponseContextFinished(ResponseContext *context) override;
        bool SetSaveHistoryChat(bool save) override;
        void ConfigureHostPrefixCache(size_t bytes, int64_t kvBytes = -1, int64_t swaBytes = -1);
        bool TryRestoreHostPrefix(ResponseContext &context, int gpuTokens, bool &requery) override;
        std::string PDLayout() const override;
        bool SupportsPDKvMemDecode() const override { return true; }
        std::shared_ptr<PDReceiveState> PreparePDReceive(ResponseContext &context) override;
        std::vector<int> PDDevices() const override { return tpDevices; }
        std::vector<int> PDRequestDevices(const ResponseContext &context) const override;
        int PDGroupCount() const override { return attentionDP; }
        int PDRequestGroup(const ResponseContext &context) const override { return GetPrefillBatchGroup(&context); }
        bool SelectPDRequestGroup(ResponseContext &context, const std::vector<size_t> &available,
                                  size_t required) override;
        bool HasPDResidentPrefixCache() const override { return saveHistoryChat && !tpPagedCaches.empty(); }
        fastkv::Message ExportPDState(ResponseContext &context) override;
        std::function<fastkv::Message()> PreparePDExport(ResponseContext &context) override;
        void ReleasePDExportSources(ResponseContext &context) override;
        fastkv::CacheSnapshot DescribePDCache(ResponseContext &context, const fastkv::Message &state) override;
        void ImportPDState(ResponseContext &context, const fastkv::Message &state) override;
        void AddPromptCache(const std::vector<int> &tokens) override;
        int GetKVCacheRetainedTokens(int layer) const override;
        void WarmUp() override;
        bool SupportsDecodePipeline() const override;
        bool SupportsDecodePipelineHistory() const override { return true; }
        std::unique_ptr<DecodePipeline> CreateDecodePipeline(
            const std::vector<ResponseContext *> &contexts) override;
        void ConfigureKvMem(const KvMemConfig &config) override;
        bool SupportsKvMem() const override { return true; }
        bool SupportsKvMemBatch() const override { return true; }
        bool SupportsKvMemPrefixCache() const override { return true; }
        bool SupportsSharedPagedKVPool() const override { return !tpPagedCaches.empty(); }
        bool UsesPackedMoeSource() const { return tpPackedMoeSources; }
        int Ep8ExpertDevice(int expert, const std::vector<int> &loadDevices) const {
            AssertInFastLLM(tpEp8Mega && loadDevices == tpDevices && expert >= 0 && expert < 256,
                           "MiMo EP8 expert loading must use the same ordered devices as TP.");
            return tpDevices.at(expert / (256 / tpDevices.size()));
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
        static constexpr int MaxPrefillSnapshots = 3;
        static constexpr int PrefixSnapshotsPerRequest = MaxPrefillSnapshots + 1;
        // Interleaved logical IDs keep draft sharding stable. Only the local row
        // is materialized on the attention group's devices.
        struct WindowSlots {
            std::mutex mutex;
            std::vector<std::vector<int>> free;
            int spareRows = 0;
            void Init(int groups, int perGroup, int spare = 0) {
                spareRows = spare;
                free.resize(groups);
                for (int group = 0; group < groups; ++group) {
                    free[group].reserve(perGroup);
                    for (int row = perGroup - 1; row >= 0; --row)
                        free[group].push_back(row * groups + group);
                }
            }
            int Take(int group, bool request = true) {
                std::lock_guard<std::mutex> lock(mutex);
                auto &slots = free.at(group);
                if ((int)slots.size() <= (request ? spareRows : 0)) return -1;
                int slot = slots.back(); slots.pop_back();
                return slot;
            }
            void Return(int slot) {
                std::lock_guard<std::mutex> lock(mutex);
                free.at(slot % free.size()).push_back(slot);
            }
            bool Available(int group) {
                std::lock_guard<std::mutex> lock(mutex);
                return (int)free.at(group).size() > spareRows;
            }
            int Row(int slot) const { return slot / int(free.size()); }
        };
        struct PrefixPool {
            WindowSlots slots;
            std::vector<Data> storage;
            std::vector<size_t> offsets;
            size_t slotBytes = 0;
            size_t draftSlotBytes = 0;
            // Distribute complete normalized-input checkpoints across TP ranks.
            // Keeping a slot intact also supports peers without direct GPU access.
            std::vector<Data> draftStorage;
            static int DraftShardSlots(int slots, int rank, int ranks) {
                return (slots + ranks - 1 - rank) / ranks;
            }
            Data &DraftShard(int slot) { return draftStorage.at(slot % draftStorage.size()); }
            size_t DraftOffset(int slot) const { return size_t(slot / draftStorage.size()) * draftSlotBytes; }
            // Serialized by historyMutex. Metadata previews never own slots.
            uint64_t historyVersion = 0;
            uint64_t orphanWindows = 0, evictedWindows = 0, recycledWindows = 0, noWindowSlots = 0;
            int SlotCount(int requests, int64_t budget) const {
                if (!slotBytes) return 0;
                // Reserve all active checkpoints before spending spare budget
                // on completed history. The caller must check the total fits.
                return (int)std::max<int64_t>((int64_t)requests * PrefixSnapshotsPerRequest,
                    std::min<int64_t>((int64_t)requests * 8,
                        std::max<int64_t>(0, budget) / 32 / (slotBytes + draftSlotBytes)));
            }
        };
        struct PrefixSnapshot {
            std::shared_ptr<PrefixPool> pool;
            int slot = -1, length = 0;
            int ownerDevice = -1;
            bool draftReady = false;
            std::vector<int> tokens;
            uint64_t tokenHash = 0; // Immutable with tokens after publication.
            // Observe the resident full-prefix endpoint without pinning KV.
            // Generation checks prevent a recycled physical page from reviving
            // a stale window checkpoint.
            PagedCacheManager *fullPool = nullptr;
            int fullPage = -1;
            long long fullGeneration = 0;
            struct CheckpointPage {
                PagedCacheManager *pool;
                int page;
                long long generation;
            };
            std::vector<CheckpointPage> checkpointPages;
            bool HasResidentPrefix() const {
                return fullPool && fullPool->IsCachedPage(fullPage, fullGeneration);
            }
            ~PrefixSnapshot() {
                for (const auto &entry : checkpointPages)
                    entry.pool->ReleasePrefixCheckpoint(entry.page, entry.generation);
                if (pool && slot >= 0) {
                    pool->slots.Return(slot);
                }
            }
        };
        using PrefixSnapshots = std::vector<std::shared_ptr<PrefixSnapshot>>;
        void ReleaseRequestWindowStorage(ResponseContext *context, int threadLimit = GetThreads());
        struct HostPrefixTransfer;
        size_t hostPrefixBytes = 0;
        int64_t hostPrefixSwaBytes = -1;
        std::shared_ptr<fastkv::HostCache> hostPrefixCache;
        void PublishHostPrefix(ResponseContext &context, const std::shared_ptr<PrefixSnapshot> &snapshot);
        bool RestorePrefixWindow(ResponseContext *context, const std::shared_ptr<PrefixSnapshot> &snapshot) const;
        int TakeHistorySlot(int group);
        void RetirePrefixSnapshot(std::shared_ptr<PrefixSnapshot> &snapshot);
        PrefixSnapshots BeginPrefixSnapshots(const Data &key, int past, int length);
        void FinishPrefixSnapshots(const Data &key, const PrefixSnapshots &snapshots, bool prefill);
        struct PrefixWindowCopies {
            std::vector<void *> destinations;
            std::vector<const void *> sources;
            std::vector<size_t> bytes;
            void Submit() const;
        };
        void AppendPrefixWindowCopies(PrefixWindowCopies &copies, const PrefixSnapshot &snapshot,
                                      int rank, int layer, const Data &key, const Data &value,
                                      int first, bool ring);
        void CopyPrefixWindow(const PrefixSnapshot &snapshot, int rank, int layer,
                              const Data &key, const Data &value, int first, bool ring);

        struct LogitsSelection {
            int count = 0;
            bool greedy = false;
            bool cudaSampling = false;
            bool cudaLogits = false; // Keep full logits on device for rejection sampling.
            float invTemperature = 1.0f;
            Data candidates; // CPU [rows, count * 2], (global token ID, score).
            std::vector<int> sampledTokens;
        };
        LogitsSelection SelectLogits(const GenerationConfig &config, bool speculative = false) const;
        bool SupportsCudaRejectionSampling(const GenerationConfig &config) const;
        std::vector<int> SampleCudaLogits(Data &logits,
                                        const std::vector<GenerationConfig> &configs,
                                        const LastTokensManager *lastTokens = nullptr);
        struct TargetCapture {
            LogitsSelection *selection = nullptr;
            bool verifying = false;
            bool collectHidden = true;
            // Internal consumers finish before the next target forward. Public
            // callers keep owning outputs unless they explicitly opt in.
            bool borrowOutputs = false;
            bool ringVerify = false; // Candidates are staged after the retained SWA ring.
            std::map<int, Data> hidden;
            std::vector<PrefixSnapshots> prefixes;
            // Optional per-request CUDA destinations, consumed before the next forward.
            std::vector<Data *> logitsDestinations;
        };
        struct DraftTPState;
        std::shared_ptr<DraftTPState> draftTP;
        bool draftSingleDeviceLogged = false;
        struct DraftWorkspace;
        // Bounded LRU; graph-owned inputs are refreshed on every replay.
        std::vector<std::shared_ptr<DraftWorkspace>> draftBatchWorkspaces;
        // Graph variants execute serially and share their large activation buffers.
        std::shared_ptr<DraftWorkspace> draftBatchStorage, draftContextStorage;
        std::shared_ptr<DraftWorkspace> draftProposalStorage;
        std::vector<std::shared_ptr<DraftWorkspace>> draftSamplingRanks;
        // Single-device draft graphs execute serially, including request churn.
        std::map<int, std::shared_ptr<void>> draftLinearGraphWorkspaces;
        struct DraftContext {
            int committed = 0;
            int deviceRank = -1; // Stable draft DP owner; retained with reusable KV storage.
            Data prefixHidden; // Logical prefill tail; backing belongs to the request workspace.
            std::vector<std::pair<Data, Data>> kv;
            std::shared_ptr<DraftWorkspace> workspace;
            Data proposalProbs{FLOAT32}; // Actual GPU proposal distributions, retained until verification.
            std::deque<std::pair<int, int>> pending;
            std::mt19937_64 random{std::random_device{}()};
            uint64_t rounds = 0, proposed = 0, accepted = 0;
            double Uniform() { return std::generate_canonical<double, 53>(random); }
        };
        struct TargetWorkspace;
        std::vector<int> ForwardTargetBatch(int batch, const Data &inputIds,
            const std::vector<Data *> &attentionMask, const std::vector<Data *> &positionIds,
            const std::vector<int> &seqLens, std::vector<std::pair<Data *, Data *>> &pastKeyValues,
            const std::vector<GenerationConfig> &configs, const LastTokensManager &lastTokens,
            std::vector<std::vector<float> *> *retLogits, TargetCapture *capture = nullptr);
        static void RunLocalPrefillRouter(Data &input, Data &weight,
            const std::vector<int> &rows, int worldRows, int ranks, Data &scratch, Data &output);
        Data RunTarget(const Data &inputIds, const Data &positions,
                       std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config,
                       TargetCapture *capture, int tpRank = -1, const Data *embedding = nullptr,
                       TargetWorkspace *workspace = nullptr, const TargetBatch *batch = nullptr);
        int CacheReserveCapacity(const GenerationConfig &config) const;
        void AppendCache(Data &cache, Data &input, DataType cacheType, int reserveCapacity = 0, bool usePool = true, bool expandFP8 = false);
        void ReserveCache(Data &cache, int capacity, int width, bool usePool = true, const Data *requestKey = nullptr);
        bool PreparePagedCache(Data &cache, PagedCacheManager &manager, int end);
        bool PreparePagedCache(Data &cache, PagedCacheManager &manager, int end,
                               std::vector<Data *> *pendingPageTables, const Data *requestKey = nullptr);
        void UploadPagedCacheTables(const std::vector<Data *> &pendingPageTables);
        static void TrimCache(Data &cache, int length);
        int SampleTarget(Data &logits, std::vector<std::pair<Data, Data>> &kv,
                         const GenerationConfig &config, const LastTokensManager &lastTokens,
                         std::vector<float> *retLogits, LogitsSelection *selection = nullptr);
        void InitDraft();
        void AppendDraftContext(Data &hidden, int start, DraftContext &context);
        void RestoreDraftPrefix(const PrefixSnapshot &snapshot, int length, DraftContext &context);
        bool AppendDraftContextFused(Data &hidden, int start, DraftContext &context);
        void CommitDraftContext(TargetCapture &capture, int tokens, DraftContext &context,
                                std::vector<std::pair<Data, Data>> &kv);
        void ProjectDraftContext(Data &combined, Data &projected, Data &hidden);
        std::shared_ptr<DraftContext> CreateDraftContext();
        bool PrepareDraftTP();
        void PrepareDraftContextTP(DraftContext &context);
        std::string DraftPDLayout() const;
        struct PDExportPlan;
        fastkv::Message PackPDState(ResponseContext &context, const std::vector<int> &devices,
                                    PDExportPlan &plan, bool independent);
        void ExportDraftPDState(PDStateWriter &writer, const DraftContext &context, int length);
        std::shared_ptr<DraftContext> ImportDraftPDState(PDStateReader &reader, int length);
        bool AppendDraftContextTP(Data &hidden, int start, DraftContext &context);
        bool RunDraftGraph(int anchor, DraftContext &context, Data &output);
        bool RunDraftSamplingStep(int previous, int step, const Data &baseLogits,
            DraftContext &context, const GenerationConfig &config, int &token);
        bool RunDraftProposalGraph(int anchor, const Data &baseLogits,
                                   DraftContext &context, std::vector<int> &proposed);
        Data RunDraft(int anchor, DraftContext &context);
        bool RunDraftBatch(const std::vector<int> &anchors,
                           const std::vector<DraftContext *> &contexts, Data &normalized);
        void ApplyDraftDevice();
        Data &DraftWeight(const std::string &name);
        Data RunDraftHead(Data &hidden);
        Data RunDraftTarget(const Data &inputIds, const Data &positions,
                            std::vector<std::pair<Data, Data>> &kv,
                            const GenerationConfig &config, TargetCapture &capture,
                            const TargetBatch *batch = nullptr);
        void CommitTargetCache(std::vector<std::pair<Data, Data>> &kv, int past, int count, bool ringVerify = false);
        void CommitTargetCaches(const std::vector<std::vector<std::pair<Data, Data>> *> &caches,
            const std::vector<int> &past, const std::vector<int> &counts, bool ringVerify);
        bool CommitDraftContexts(TargetCapture &capture, const std::vector<int> &lengths,
            const std::vector<int> &counts, const std::vector<DraftContext *> &contexts);
        struct DraftRound;
        void PrepareDraftRound(DraftRound &round, const Data *preparedLogits = nullptr);
        bool PrepareDraftBatch(std::vector<std::unique_ptr<DraftRound>> &rounds, const Data &baseLogits);
        bool PrepareDraftBatchLocal(const std::vector<DraftRound *> &rounds, const Data &baseLogits, int rank);
        bool PrepareDraftBatchGraph(const std::vector<DraftRound *> &rounds, const Data &baseLogits, int rank);
        void PrepareDraftSamplingMemory(int rows, int steps, int vocab, int latentWidth, int rank = -1);
        void PrepareDraftAcceptance(std::vector<std::unique_ptr<DraftRound>> &rounds, Data &logits);
        void PrepareDraftAcceptanceLocal(const std::vector<DraftRound *> &rounds, Data &logits, int rank);
        int FinishDraftRound(DraftRound &round, Data &logits, TargetCapture &capture,
                             std::vector<std::pair<Data, Data>> &kv);
        std::vector<int> ForwardDraftBatch(int batch, const Data &inputIds,
                const std::vector<Data *> &positions, const std::vector<int> &lengths,
                std::vector<std::pair<Data *, Data *>> &kv,
                const std::vector<GenerationConfig> &configs, const LastTokensManager &lastTokens,
                std::vector<std::vector<float> *> *retLogits);
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
        std::vector<std::shared_ptr<DraftContext>> idleDraftContexts;
        void PrepareDraftServingMemory(bool history);

    private:
        friend struct NaiveN05PrefixCacheTestAccess;
        friend struct NaiveN05FP4KVTestAccess;
        friend struct NaiveN05GraphTestAccess;
        // Only request lifecycle callbacks (serialized by dictLocker) own this
        // single idle allocation set. No token contents are reused.
        std::vector<std::pair<Data, Data>> tpIdleCache;
        bool CanReuseTensorParallelCache(const ResponseContext *context) const;
        void RestoreTensorParallelCache(ResponseContext *context);
        void RecycleTensorParallelCache(ResponseContext *context);
        struct SparseKvState;
        struct SparsePDReceive;
        struct GpuPDReceive;
        std::shared_ptr<SparseKvState> sparseKv;
        struct WindowLease {
            std::shared_ptr<WindowSlots> slots;
            std::shared_ptr<SparseKvState> sparse; // Keeps borrowed sparse buffers alive after binding.
            int id = -1;
            ~WindowLease() { if (id >= 0) slots->Return(id); }
        };
        // Unconfigured library users may enable history after warmup. An explicit
        // startup opt-out gives that memory back to the fixed token pool.
        bool reserveWindowHistory = true;
        std::shared_ptr<WindowSlots> activeWindowSlots;
        // Affinity fixes the KV owner before admission. The global scheduler caps
        // total requests when the batch is not divisible by the number of groups.
        int RunningRequestsPerGroup(int batch) const { return 1 + (batch - 1) / attentionDP; }
        int WindowRequestsPerGroup(int batch) const {
            const int mean = RunningRequestsPerGroup(batch);
            return std::min(batch, mean + (mean + 3) / 4);
        }
        int WindowGroup(const Data &key) const;
        bool ReserveWindow(Data &key, bool request = true) const;
        void InitSparseKv();
        void AttachSparseKv(const std::vector<std::pair<Data *, Data *>> &roots);
        Data SeparateIndexer(int layer, int rank, const Data &key) const;
        bool HasSeparateIndexer() const { return kvCacheDataType == FP4_E2M1 || sparseKv || !tpIndexerCaches.empty(); }
        PagedCacheManager &IndexerStorage(int layer, int rank) const {
            return *tpIndexerSidecars.at(tpPagedCaches[rank][layer * 2].get());
        }
        void StageSparseKv(int rank, int layer, const Data &key, const Data &value, Data &outKey, Data &outValue);
        // FP4 is limited to DSA K/V. SWA and the independent Indexer retain FP8.
        DataType AuxiliaryKVType() const { return kvCacheDataType == FP4_E2M1 ? FP8_E4M3 : kvCacheDataType; }
        DataType LayerKVType(int layer) const { return slidingLayers[layer] ? AuxiliaryKVType() : kvCacheDataType; }
        int PackedIndexerWidth() const { return HasSeparateIndexer() ? 0 : IndexerCacheWidth(AuxiliaryKVType()); }
        bool ResidentMoePlacement() const;
        bool InitTensorParallel();
        void PrepareTensorParallel();
        void InitTensorParallelKVPool();
        std::shared_ptr<CudaWorkspace> RequestKVPool(int device, size_t bytes) const;
        void PrepareSwaPrefillWorkspace(int rank, int length);
        struct TPDecodeState;
        std::shared_ptr<TPDecodeState> tpDecodeState, tpVerifyState, tpBatchState, tpBatchVerifyState;
        std::vector<std::shared_ptr<TPDecodeState>> tpBatchVerifyGraphs;
        std::shared_ptr<TPDecodeState> tpBatchVerifyStorage;
        Data tpSamplingLogits{FLOAT32};
        struct TargetDecodePipeline;
        struct TPServingState;
        std::shared_ptr<TPServingState> tpServingState;
        std::atomic<uint64_t> tpCacheEpoch{0};
        // A fresh identity protects both request lifetime and owned KV layout.
        // Unknown/global changes still invalidate every plan through tpCacheEpoch.
        mutable std::mutex tpRequestMutex;
        std::unordered_map<const Data *, uint64_t> tpRequestIds;
        // Sampling identity survives KV rebinding and lasts until request removal.
        std::unordered_map<const Data *, uint64_t> tpSamplingRequestIds;
        uint64_t tpNextRequestId = 0;
        void InvalidateTensorParallelCache(const Data *requestKey = nullptr);
        bool CanReuseTensorParallelServingCache(const TargetBatch *batch, bool checkCapacity = true) const;
        std::vector<unsigned char> ReusableTensorParallelServingRows(const TargetBatch *batch,
                                                                    bool checkCapacity = true) const;
        void ReorderTensorParallelServingCache(const TargetBatch *batch);
        void PrepareTensorParallelServing();
        bool PrepareTensorParallelServingDecode(const Data &inputIds,
            const LogitsSelection &selection, std::vector<int *> &cacheLengths,
            const TargetBatch *batch, bool collectHidden);
        bool HasVerificationGraph() const;
        bool PrepareTensorParallelDecode(const Data &inputIds,
                                        std::vector<std::pair<Data, Data>> &kv, bool verifying,
                                        const LogitsSelection &selection, std::vector<int *> &cacheLengths,
                                        const TargetBatch *batch = nullptr,
                                        bool collectHidden = false, bool reuseOnly = false);
        Data ForwardTensorParallelDecode(int rank, const Data &inputIds, const Data &positions,
                                        std::vector<std::pair<Data, Data>> &kv,
                                        const GenerationConfig &config, const Data *embedding, TargetCapture *capture,
                                        float *candidateResult, const TargetBatch *batch = nullptr,
                                        bool keepCudaLogits = false);
        Data ForwardTensorParallel(const Data &inputIds, const Data &positions,
                                   std::vector<std::pair<Data, Data>> &kv,
                                   const GenerationConfig &config, TargetCapture *capture = nullptr,
                                   LogitsSelection *selection = nullptr, const TargetBatch *batch = nullptr);
        struct AttentionParallelState;
        std::shared_ptr<AttentionParallelState> attentionParallel;
        int attentionDP = 1;
        void InitAttentionParallelCache(const std::vector<std::pair<Data *, Data *>> &roots, int group = -1);
        void PrepareAttentionParallelSampling();
        int AttentionTPSize() const { return tpDevices.size() / attentionDP; }
        Data ForwardAttentionParallel(const Data &inputIds, const Data &positions,
            std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config,
            TargetCapture *capture, LogitsSelection *selection, const TargetBatch *batch);
        std::shared_ptr<TPDecodeState> SelectAttentionServingGraph(const TargetBatch &batch,
            std::vector<int> &counts, std::vector<std::vector<int>> *samplingRows = nullptr) const;
        bool ForwardAttentionParallelServing(const Data &inputIds, const Data &positions,
            std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config,
            LogitsSelection *selection, const TargetBatch *batch, Data &output);
        bool PrepareAttentionParallelServingDecode(const TargetBatch *batch);
        std::vector<int> tpDevices;
        std::map<int, std::shared_ptr<CudaWorkspace>> tpKVPool;
        int tpSwaCacheCapacity = 0;
        // Separate equal-sized request allocations so prefix restore and cold
        // prefill cannot fragment each other's page tables and SWA windows.
        std::map<std::pair<int, size_t>, std::shared_ptr<CudaWorkspace>> tpRequestKVPool;
        // One chronological prefill workspace per GPU, reused across layers
        // and requests. Request-owned SWA storage only holds the sliding window.
        std::vector<std::pair<Data, Data>> tpSwaPrefillWorkspaces;
        std::vector<std::vector<std::unique_ptr<PagedCacheManager>>> tpPagedCaches;
        std::vector<std::vector<std::unique_ptr<PagedCacheManager>>> tpIndexerCaches;
        std::map<PagedCacheManager *, PagedCacheManager *> tpIndexerSidecars;
        // Stable registrations borrow fixed pool storage; request leases own
        // the individual pages until transfer completion or cancellation.
        struct GpuPDPools;
        std::shared_ptr<GpuPDPools> pdReceivePools;
        bool tpPrepared = false;
        bool tpMoePreparationFailed = false;
        bool tpPackedMoeSources = false;
        // Fixed at TP preparation so toggling CUDA Graph cannot change live KV
        // layout. Prefill/eager fallback temporarily uses chronological rows.
        bool tpSwaRingCache = false;
        bool tpEp8Mega = false;
        std::shared_ptr<FastllmCudaMimoEp8Mega> tpEp8MegaState;
        std::vector<std::shared_ptr<TargetWorkspace>> tpVerifyWorkspaces;
        std::vector<std::shared_ptr<TargetWorkspace>> tpBatchWorkspaces;
        std::vector<std::shared_ptr<TargetWorkspace>> tpPrefillWorkspaces;
        std::vector<Data> tpPrefillScores;
        void PrepareTensorParallelMemory();
        std::vector<Data> tpDraftHeadInputs, tpDraftHeadLogits;
        Data tpDraftHeadOutput;
        struct DraftHeadState;
        std::shared_ptr<DraftHeadState> tpDraftHeadState;
        PersistentWorkerGroup tpWorkers;
        std::vector<std::vector<std::vector<Data *>>> tpMoeWeights, tpMoeBiases;
        std::vector<std::pair<int, int>> tpVocabRanges;
        struct ActivePrefixes {
            // One immutable SWA/draft checkpoint for a bounded queued hit.
            // This never owns full KV pages or request-private GPU storage.
            std::shared_ptr<PrefixSnapshot> queued;
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
        bool indexerComputeFP8 = false;
        bool indexerLayerSharding = false;
        bool indexerContextSharding = false;
        bool tpIndexerPeerAccess = false;
        int IndexerOwnerRank(int layer) const {
            return std::count(slidingLayers.begin(), slidingLayers.begin() + layer, 0) % tpDevices.size();
        }
        bool HasLocalIndexer(int layer, int rank) const {
            return !slidingLayers[layer] &&
                (!indexerLayerSharding || rank < 0 || rank == IndexerOwnerRank(layer));
        }
        int IndexerCacheWidth(DataType type) const {
            return fp8IndexerCache ? 144 / (type == FP8_E4M3 ? 1 : 2) : indexDim;
        }
        int LocalIndexerCacheWidth(int layer, int rank) const {
            return !HasSeparateIndexer() && HasLocalIndexer(layer, rank) ? IndexerCacheWidth(AuxiliaryKVType()) : 0;
        }
        std::vector<std::vector<Data *>> moeWeights, moeBiases;
    };
}
#endif
