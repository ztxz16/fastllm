#pragma once
#include "basellm.h"

namespace fastllm {
namespace hy4 {
struct Scratch;
struct GpuGraphs;
struct HybridGraphs;
struct DraftRequest;
} // namespace hy4
struct ReadGGUFTask;
class Hy4FlashModel final : public basellm {
  public:
    Hy4FlashModel();
    ~Hy4FlashModel() override;
    void InitParams() override;
    void OnModelWeightsLoaded() override;
    using Mapping = std::map<std::string, std::vector<std::pair<std::string, DataType>>>;
    Mapping GetTensorMap(const std::vector<std::string> &names) override;
    int Forward(const Data &ids, const Data &mask, const Data &positions,
                std::vector<std::pair<Data, Data>> &past, const GenerationConfig &cfg,
                const LastTokensManager &last, std::vector<float> *logits) override;
    std::vector<int> ForwardV2(int batch, const Data &ids, const std::vector<Data *> &masks,
                               const std::vector<Data *> &positions, const std::vector<int> &lengths,
                               std::vector<std::pair<Data *, Data *>> &past,
                               const std::vector<GenerationConfig> &cfg, const LastTokensManager &last,
                               std::vector<std::vector<float> *> *logits) override;
    using basellm::ForwardBatch;
    std::vector<int> ForwardBatch(int batch, const Data &ids, const std::vector<Data *> &masks,
                                  const std::vector<Data *> &positions, const std::vector<int> &lengths,
                                  std::vector<std::pair<Data *, Data *>> &past,
                                  const std::vector<GenerationConfig> &cfg, const LastTokensManager &last,
                                  std::vector<std::vector<float> *> *logits) override {
        return ForwardV2(batch, ids, masks, positions, lengths, past, cfg, last, logits);
    }
    std::string MakeInput(const std::string &history, int round, const std::string &input) override;
    std::string MakeHistory(const std::string &history, int round, const std::string &input,
                            const std::string &output) override;
    bool NeedAttentionMask(int qlen, int klen) override {
        return false;
    }
    bool UsesHostExperts() const {
        return hybrid;
    }
    bool UseGenericHistoryCache() const override {
        return false;
    }
    void OnResponseContextCreated(ResponseContext *context) override;
    bool PopPendingResponseToken(ResponseContext *context, int previous, int &token) override;
    struct Weights;
    void ImportGGUF(const json11::Json &params, const std::vector<ReadGGUFTask> &tasks);
    void AddSource(const std::string &name, const std::string &file, const std::string &dtype,
                   const std::vector<int> &dims, uint64_t offset, uint64_t bytes);
    // Enable the embedded draft before creating response contexts.
    bool EnableDFly2(int width);
    void SetConfig(const json11::Json &config);

  private:
    std::unique_ptr<Weights> source;
    std::vector<std::vector<Data *>> hostExperts;
    std::vector<std::shared_ptr<hy4::Scratch>> gpuArenas;
    std::shared_ptr<hy4::GpuGraphs> gpuGraphs;
    std::shared_ptr<hy4::HybridGraphs> hybridGraphs;
    std::vector<std::shared_ptr<hy4::Scratch>> hybridArenas;
    std::vector<int> hybridStageEnds;
    size_t HybridStage(int layer) const;
    int draftWidth = 0, draftMask = 0;
    std::vector<int> draftProviders;
    bool hybrid = false;
    bool hostCacheReady = false, diskGraphReady = false;
    std::vector<int> devices;
    using Cache = std::vector<std::pair<Data *, Data *>>;
    using Logits = std::vector<std::vector<float>>;
    int CommitVerifiedTokens(size_t request, Cache &past, hy4::DraftRequest &state,
                             const GenerationConfig &config, const std::vector<int> &tokens);
    std::vector<int> ProposeGpu(const std::vector<int> &anchors, Cache &past,
                                const std::vector<GenerationConfig> &configs,
                                const std::vector<hy4::DraftRequest *> &states);
    std::vector<int> ProposeHybrid(const std::vector<int> &anchors, Cache &past,
                                   const std::vector<GenerationConfig> &configs,
                                   const std::vector<hy4::DraftRequest *> &states);
    Logits ForwardGpu(const std::vector<int> &tokens, const std::vector<int> &lengths, Cache &past,
                      const std::vector<GenerationConfig> &configs, bool verify = false,
                      std::vector<int> *sampled = nullptr, float **deviceLogits = nullptr);
    Logits ForwardHybrid(const std::vector<int> &tokens, const std::vector<int> &lengths, Cache &past,
                         const std::vector<GenerationConfig> &configs, bool verify = false,
                         std::vector<int> *sampled = nullptr, float **deviceLogits = nullptr);
};
} // namespace fastllm
