#pragma once
#include "basellm.h"

namespace fastllm {
namespace inkling2 {
struct TensorRange {
    std::string file, type;
    std::vector<int> shape;
    uint64_t start, end;
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
    std::vector<ExpertDevice> experts;
    DataType denseType = BFLOAT16;
};
class Runtime {
public:
    virtual ~Runtime() = default;
    virtual int prefix(ResponseContext &, int limit, bool restore) = 0;
    virtual void setPrefixCache(bool enabled) = 0;
    virtual bool popPending(Data &key, int previous, int &token) = 0;
    virtual void warmupDecodeGraph() = 0;
    virtual std::unique_ptr<basellm::DecodePipeline> pipeline(ResponseContext &context) = 0;
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
    bool SupportsDecodePipeline() const override;
    bool SupportsDecodePipelineHistory() const override { return true; }
    std::unique_ptr<DecodePipeline> CreateDecodePipeline(const std::vector<ResponseContext *> &contexts) override;
    bool SetSaveHistoryChat(bool save) override;
    int QueryPagedPrefixCacheExtra(ResponseContext *ctx, int limit) const override;
    bool RestorePagedPrefixCacheExtra(ResponseContext *ctx, int length) const override;
    std::vector<std::pair<int, PagedCacheManager *>> GetPagedKVCacheManagersForRequest(
        int layer, bool key, const ResponseContext *ctx) const override {
        // Sliding windows rotate in-place and must have private writable pages.
        if (GetKVCacheRetainedTokens(layer) >= 0) return {};
        return basellm::GetPagedKVCacheManagersForRequest(layer, key, ctx);
    }
    bool DeferHistoryCacheRestore() const override { return saveHistoryChat; }
    bool UseGenericHistoryCache() const override { return false; }

private:
    inkling2::Config cfg{};
    inkling2::Placement placement{};
    std::unique_ptr<inkling2::Runtime> runtime;
};
} // namespace fastllm
