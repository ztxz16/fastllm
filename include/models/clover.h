#ifndef FASTLLM_CLOVER_H
#define FASTLLM_CLOVER_H
#include "basellm.h"
#include "clover_sampling.h"

namespace fastllm {
// Clover has FP32 recurrent caches and read-only bidirectional previews.
// Prefix indexing is shared with Qwen3.5; tensor layouts remain model-specific.
struct CloverSnapshots {
    int position = 0, tokens = 0;
    bool compact = false; // Require bounded journals even for short/tail blocks.
    std::vector<std::pair<Data, Data>> layers;
};

class CloverModel : public basellm {
public:
    CloverModel();
    ~CloverModel() override;
    void InitParams() override;
    void SetDataType(DataType type) override;
    void SetKVCacheDataType(DataType type) override;
    std::map<std::string, std::vector<std::pair<std::string, DataType>>>
        GetTensorMap(const std::vector<std::string> &names) override;
    int Forward(const Data &inputIds, const Data &attentionMask, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &cache,
        const GenerationConfig &config = GenerationConfig(),
        const LastTokensManager &lastTokens = LastTokensManager(),
        std::vector<float> *logits = nullptr) override;
    // Explicit preview contract: no committed cache length or contents change.
    void ForwardLogits(const std::vector<int> &tokens,
        std::vector<std::pair<Data, Data>> &cache, bool preview,
        Data &logits, bool lastOnly = false, CloverSnapshots *snapshots = nullptr,
        bool needLogits = true);
    // Equal-width request blocks share dense/MoE work; caches and coreset
    // selection remain independent for each request.
    void ForwardLogitsBatch(const std::vector<std::vector<int>> &tokens,
        const std::vector<std::vector<std::pair<Data, Data>> *> &caches,
        bool preview, std::vector<Data> &logits,
        const std::vector<CloverSnapshots *> &snapshots = {});
    void RestorePrefix(std::vector<std::pair<Data, Data>> &cache,
        const CloverSnapshots &snapshots, int acceptedInputs);
    void RestorePrefixBatch(const std::vector<std::vector<std::pair<Data, Data>> *> &caches,
        const std::vector<const CloverSnapshots *> &snapshots, const std::vector<int> &acceptedInputs);
    ModelContextSpec GetContextSpec() const override { return {true, false, ""}; }
    bool UseModelSpecificScheduler() const override { return true; }
    void RunModelSpecificScheduler() override;
    int LaunchResponseTokens(const std::vector<int> &tokens,
        const GenerationConfig &config = GenerationConfig(),
        const std::map<std::string, std::vector<Data *>> &multimodal = {}) override;
    int LaunchClover(const std::vector<int> &tokens, const json11::Json &config);
    void ConfigureClover(const json11::Json &config);
    void OnResponseContextCreated(ResponseContext *context) override;
    void OnResponseContextRemoved(ResponseContext *context) override;
    bool UseGenericHistoryCache() const override { return false; }
    std::string MakeInput(const std::string &history, int round, const std::string &input) override;
    std::string MakeHistory(const std::string &history, int round,
        const std::string &input, const std::string &output) override;
    void WarmUp() override;
private:
    static constexpr int MaxBatch = 256;
    using CachePair = std::pair<Data *, Data *>;
    using CacheView = std::vector<CachePair>;
    struct SnapshotView {
        int position = 0, tokens = 0;
        bool compact = false;
        CacheView layers;
    };
    struct Parallel;
    std::shared_ptr<Parallel> parallel;
    std::vector<int> parallelDevices;
    int tpRank = 0, tpSize = 1;
    bool rankLocal = false;
    bool InitParallel();
    void RunParallel(const std::function<void(CloverModel &, int)> &task);
    CacheView ViewCache(std::vector<std::pair<Data, Data>> &cache, int device = -1);
    void PublishCache(std::vector<std::pair<Data, Data>> &cache);
    void ForwardLogitsLocal(const std::vector<int> &tokens, CacheView &cache, bool preview,
        Data &logits, bool lastOnly, SnapshotView *snapshots, bool needLogits);
    void ForwardLogitsBatchLocal(const std::vector<std::vector<int>> &tokens,
        const std::vector<CacheView *> &caches, bool preview, std::vector<Data> &logits,
        const std::vector<SnapshotView *> &snapshots);
    void RestorePrefixLocal(CacheView &cache, const SnapshotView &snapshots, int acceptedInputs);
    void Reduce(Data &data);
    void GatherLogits(Data &data);
    std::vector<bool> linearLayers;
    int keyHeads = 16, valueHeads = 32, intermediate = 512;
    int blockSize = 32;
    float ropeTheta = 10000000.f, coresetTopP = .9f;
    bool coresetEnabled = true;
    bool resident = false;
    int cudaDevice = 0;
    // Serialized forwards reuse one prefill workspace across layers. It is
    // released with the model after ShutdownRuntime has joined the worker.
    Data gdnPrefillWorkspace;
    Data samplingWorkspace;
    int maskToken = 151669, vocabSize = 248320;
    std::shared_ptr<class HybridMoeBF16> hybridMoe;
    std::vector<bool> hostExperts;
    struct MoeGraphs;
    std::shared_ptr<MoeGraphs> moeGraphs;
    struct Request;
    struct PrefixCache;
    std::shared_ptr<PrefixCache> prefixCache;
    long long nextRequestId = 0;
    std::map<ResponseContext *, std::shared_ptr<Request>> requests;
    std::mutex launchLocker;
    CloverOptions defaults, launchingOptions;
    void EnsureResident();
    CloverOptions ReadOptions(const json11::Json &config) const;
    void ValidateRequest(const std::vector<int> &tokens, const GenerationConfig &config,
        const CloverOptions &options) const;
    std::vector<int> Step(ResponseContext &context, Request &request);
    std::vector<std::vector<int>> StepBatch(const std::vector<ResponseContext *> &contexts,
        const std::vector<std::shared_ptr<Request>> &batch);
    void Finish(ResponseContext *context);
    void LoadPrefix(ResponseContext &context, Request &request, Data &logits);
    void SavePrefix(ResponseContext &context, Request &request, const Data *logits = nullptr);
    void Norm(Data &input, const std::string &name, Data &output,
        Data *gate = nullptr, bool zeroCentered = true);
    void AddNorm(Data &residual, Data &input, const std::string &name, Data &output);
    void Gdn(Data &input, int layer, CachePair &cache,
        bool preview, Data &output, CachePair *snapshots = nullptr, bool compact = false);
    void FullAttention(Data &input, int layer, CachePair &cache,
        int position, bool preview, Data &output);
    void GdnCoreBatch(Data &projected, int layer, CachePair *const *caches, int batch,
        bool preview, Data &normalized, CachePair *const *snapshots, bool compact = false);
    void FullAttentionCore(Data &qgate, Data &k, Data &v, int layer,
        CachePair &cache, int position, bool preview, Data &output);
    void FullAttentionCoreBatch(Data &projected, int layer,
        const std::vector<CacheView *> &caches,
        const std::vector<int> &positions, int width, bool preview, Data &output);
    void Moe(Data &input, int layer, bool preview, Data &output, bool graphAllowed, int batch = 1);
};
}
#endif
