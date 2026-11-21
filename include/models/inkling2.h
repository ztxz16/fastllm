#pragma once
#include "basellm.h"

namespace fastllm {
namespace inkling2 {
struct TensorRange {
    std::string file, type;
    std::vector<int> shape;
    uint64_t start, end;
};
struct Config {
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
};
class Runtime {
public:
    virtual ~Runtime() = default;
    virtual void warmupDecodeGraph() = 0;
    virtual std::unique_ptr<basellm::DecodePipeline> pipeline(ResponseContext &context) = 0;
    virtual std::vector<int> forward(const Data &ids, const std::vector<Data *> &positions,
                                     const std::vector<int> &lengths,
                                     const std::vector<std::pair<Data *, Data *>> &cache,
                                     const std::vector<GenerationConfig> &sampling,
                                     const LastTokensManager &history,
                                     std::vector<std::vector<float> *> *logits) = 0;
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
    void SetKVCacheDataType(DataType type) override;
    void OnModelWeightsLoaded() override;
    void WarmUp() override;
    std::map<std::string, std::vector<std::pair<std::string, DataType>>>
    GetTensorMap(const std::vector<std::string> &names) override;
    std::map<std::string, inkling2::TensorRange> expertTensors;
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
    std::string MakeInput(const std::string &history, int round, const std::string &input) override;
    std::string MakeHistory(const std::string &history, int round, const std::string &input,
                            const std::string &output) override;
    bool SupportsDecodePipeline() const override;
    // Inkling has no restorable partial convolution history. The scheduler
    // drains terminal tickets before retiring their private KV.
    bool SupportsDecodePipelineHistory() const override { return true; }
    std::unique_ptr<DecodePipeline> CreateDecodePipeline(const std::vector<ResponseContext *> &contexts) override;
    // A KV prefix alone cannot restore the four convolution histories per visit.
    int QueryPagedPrefixCacheExtra(ResponseContext *, int) const override { return 0; }
    bool UseGenericHistoryCache() const override { return false; }

private:
    inkling2::Config cfg{};
    inkling2::Placement placement{};
    std::unique_ptr<inkling2::Runtime> runtime;
};
} // namespace fastllm
