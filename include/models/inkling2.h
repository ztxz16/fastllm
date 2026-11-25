#pragma once
#include "basellm.h"

namespace fastllm {
struct ReadGGUFTask;
namespace inkling2 {
inline bool ordinarySampling(const GenerationConfig &cfg) {
    return cfg.repeat_penalty == 1.f && cfg.tool_call_allowed_token_ids.empty() &&
           !cfg.tool_call_token_mask && cfg.top_k > 0 && cfg.temperature > 0.f && cfg.top_p > 0.f &&
           cfg.top_p <= 1.f;
}
// Pool tile-aligned short prompts. Preserve the projection algorithm and each
// request's sampling mode across the scheduler and direct packed forwards.
inline int pooledPromptKey(size_t rows, const GenerationConfig &cfg) {
    if (rows < 64 || rows >= 4096 || rows % 64 || cfg.prefill_chunk_remaining > 0 ||
        (!cfg.IsSimpleGreedy() && !ordinarySampling(cfg))) return -1;
    return int(2 * rows) + int(cfg.IsSimpleGreedy());
}
struct TensorRange {
    std::string file, type;
    std::vector<int> shape;
    uint64_t start, end;
    int ggmlType = -1;
};
struct DraftConfig {
    int steps = 0, block = 0, layers = 0, heads = 0, kvHeads = 0, dim = 0, rank = 0, intermediate = 0, window = 0;
    int maxPositions = 0;
    float theta = 0, eps = 0;
    std::vector<int> taps;
    std::vector<bool> local;
};
constexpr int imageContentToken = 200005, imagePatchToken = 200054;
struct VisionConfig {
    int width = 0, layers = 0, heads = 0, patch = 0, merge = 0, grid = 0, projection = 0;
    float eps = 0;
    std::pair<int, int> tokenGrid(int height, int width) const;
};
using ImageInputs = std::map<std::string, std::vector<Data *>>;
struct Config {
    VisionConfig vision;
    DraftConfig draft;
    int hidden, layers, heads, kvHeads, headDim, relative, extent, window, conv, residualBlock;
    int experts, topK, intermediate, start, end, passes, rank, vocab, logFloor;
    float eps, qClip, rClip, logAlpha, gateCap, upCap, routeScale, logitCap;
    bool normValues = true;
    std::vector<bool> local;
    std::vector<int> visits;
};
enum class ExpertDevice { Cuda, Cpu, Numa };
struct Placement {
    int gpu;
    std::vector<int> tensorDevices;
    std::vector<ExpertDevice> experts;
    DataType denseType = BFLOAT16;
    int world = 1, rank = 0, vocabRows = 0;
    int cacheSlot(int slot) const { return world == 1 ? slot : slot * world + rank; }
};
class Runtime {
public:
    virtual ~Runtime() = default;
    virtual bool supportsPooledPrefill() const { return false; }
    virtual bool hasRankCaches() const { return false; }
    virtual int prefix(ResponseContext &, int limit, bool restore) = 0;
    virtual void setPrefixCache(bool enabled) = 0;
    virtual bool popPending(Data &key, int previous, int &token) = 0;
    virtual void warmupDecodeGraph() = 0;
    virtual std::unique_ptr<basellm::DecodePipeline> pipeline(const std::vector<ResponseContext *> &contexts) = 0;
    virtual std::vector<int> forward(const Data &ids, const std::vector<Data *> &positions,
                                     const std::vector<int> &lengths,
                                     const std::vector<std::pair<Data *, Data *>> &cache,
                                     const std::vector<GenerationConfig> &sampling,
                                     const LastTokensManager &history,
                                     std::vector<std::vector<float> *> *logits, const ImageInputs *images = nullptr) = 0;
};
DataType projectionType(int gpu);
std::unique_ptr<Runtime> create(const Config &, const Placement &, basellm &,
                                const std::map<std::string, TensorRange> &);
} // namespace inkling2
class Inkling2Model final : public basellm {
public:
    Inkling2Model();
    ~Inkling2Model() override;
    void InitParams() override;
    void ConfigureGGUF(const json11::Json &metadata);
    void ConfigureDraftGGUF(const std::string &file, std::vector<ReadGGUFTask> &tasks);
    void LoadDraftGGUF(std::vector<ReadGGUFTask> &tasks);
    void PrepareGGUF(std::vector<ReadGGUFTask> &tasks);
    void SetDataType(DataType type) override;
    void SetKVCacheDataType(DataType type) override;
    void OnModelWeightsLoaded() override;
    void WarmUp() override;
    std::map<std::string, std::vector<std::pair<std::string, DataType>>>
    GetTensorMap(const std::vector<std::string> &names) override;
    std::map<std::string, inkling2::TensorRange> expertTensors;
    long long GetAutoWarmupCudaAdditionalCacheBytesPerToken(int deviceId) const override;
    long long GetAutoWarmupCudaRuntimeReserveBytes(int deviceId, int batch) const override;
    int GetKVCacheRetainedTokens(int slot) const override {
        return cfg.local.at(cfg.visits.at(slot)) ? cfg.window : -1;
    }
    bool NeedAttentionMask(int, int) override { return false; }
    int Forward(const Data &ids, const Data &mask, const Data &positions,
                std::vector<std::pair<Data, Data>> &cache, const GenerationConfig &sampling,
                const LastTokensManager &history, std::vector<float> *logits) override;
    std::vector<int> ForwardBatch(int batch, const Data &ids, const Data &mask, const Data &positions,
                                  std::vector<std::pair<Data, Data>> &cache, const GenerationConfig &sampling,
                                  const LastTokensManager &history,
                                  std::vector<std::vector<float> *> *logits) override;
    std::vector<int> ForwardBatch(int batch, const Data &ids, const std::vector<Data *> &masks,
                                  const std::vector<Data *> &positions, const std::vector<int> &lengths,
                                  std::vector<std::pair<Data *, Data *>> &cache,
                                  const std::vector<GenerationConfig> &sampling,
                                  const LastTokensManager &history,
                                  std::vector<std::vector<float> *> *logits) override {
        return ForwardV2(batch, ids, masks, positions, lengths, cache, sampling, history, logits);
    }
    std::vector<int> ForwardV2(int batch, const Data &ids, const std::vector<Data *> &masks,
                               const std::vector<Data *> &positions, const std::vector<int> &lengths,
                               std::vector<std::pair<Data *, Data *>> &cache,
                               const std::vector<GenerationConfig> &configs, const LastTokensManager &last,
                               std::vector<std::vector<float> *> *logits) override;
    // Count with output == nullptr, or expand the processor's one placeholder per image.
    int PrepareImagePrompt(const int *tokens, int n, const int *sizes, int images,
                           int *output, int capacity, int *offsets) const;
    std::vector<int> ForwardMultimodal(const Data &ids, const Data &mask, const Data &positions,
                std::vector<std::pair<Data, Data>> &cache, const inkling2::ImageInputs &images,
                const GenerationConfig &sampling, const LastTokensManager &history,
                std::vector<std::vector<float> *> *logits) override;
    std::string MakeInput(const std::string &history, int round, const std::string &input) override;
    std::string MakeHistory(const std::string &history, int round, const std::string &input,
                            const std::string &output) override;
    bool draftEnabled() const { return cfg.draft.steps > 0; }
    bool PopPendingResponseToken(ResponseContext *ctx, int previous, int &token) override {
        return draftEnabled() && ctx && !ctx->pastKeyValues.empty() &&
               runtime->popPending(ctx->pastKeyValues[0].first, previous, token);
    }
    bool SupportsPrefillBatching() const override {
        return runtime && runtime->supportsPooledPrefill() && dataType == BFLOAT16 && !saveHistoryChat;
    }
    int GetPrefillBatchKey(const ResponseContext &, int chunkSize) const override;
    bool SupportsDecodePipeline() const override;
    bool SupportsDecodePipelineHistory() const override { return true; }
    std::unique_ptr<DecodePipeline> CreateDecodePipeline(const std::vector<ResponseContext *> &contexts) override;
    bool SetSaveHistoryChat(bool save) override;
    int QueryPagedPrefixCacheExtra(ResponseContext *ctx, int limit) const override;
    bool RestorePagedPrefixCacheExtra(ResponseContext *ctx, int length) const override;
    PagedCacheManager *GetPagedKVCacheManager(int layer, bool key) const override {
        // Admission uses rank zero's pool for the requested layer.
        const int world = runtime && runtime->hasRankCaches() ? int(placement.tensorDevices.size()) : 1;
        return basellm::GetPagedKVCacheManager(layer * world, key);
    }
    std::vector<std::pair<int, PagedCacheManager *>> GetPagedKVCacheManagers(int layer, bool key) const override {
        if (runtime && runtime->hasRankCaches()) {
            std::vector<std::pair<int, PagedCacheManager *>> managers;
            const int world = placement.tensorDevices.size();
            for (int i = 0; i < world; ++i)
                managers.emplace_back(placement.tensorDevices[i],
                    basellm::GetPagedKVCacheManager(layer * world + i, key));
            return managers;
        }
        return basellm::GetPagedKVCacheManagers(layer, key);
    }
    std::vector<std::pair<int, PagedCacheManager *>> GetPagedKVCacheManagersForRequest(
        int layer, bool key, const ResponseContext *) const override {
        // Sliding windows rotate in-place and must have private writable pages.
        if (GetKVCacheRetainedTokens(layer) >= 0) return {};
        return GetPagedKVCacheManagers(layer, key);
    }
    bool DeferHistoryCacheRestore() const override { return saveHistoryChat; }
    bool UseGenericHistoryCache() const override { return false; }

private:
    int draftKvHeadsOnDevice(int deviceId) const;
    inkling2::Config cfg{};
    inkling2::Placement placement{};
    std::unique_ptr<inkling2::Runtime> runtime;
};
} // namespace fastllm
