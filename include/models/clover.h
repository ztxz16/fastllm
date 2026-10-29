#ifndef FASTLLM_CLOVER_H
#define FASTLLM_CLOVER_H
#include "basellm.h"
#include "clover_sampling.h"

namespace fastllm {
// Clover has FP32 recurrent caches and read-only bidirectional previews;
// it cannot share Qwen3.5's recurrent cache representation or generation loop.
struct CloverSnapshots {
    int position = 0, tokens = 0;
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
    void RestorePrefix(std::vector<std::pair<Data, Data>> &cache,
        const CloverSnapshots &snapshots, int acceptedInputs);
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
    int maskToken = 151669, vocabSize = 248320;
    struct MoeGraphs;
    std::shared_ptr<MoeGraphs> moeGraphs;
    struct Request;
    std::map<ResponseContext *, std::shared_ptr<Request>> requests;
    std::mutex launchLocker;
    CloverOptions defaults, launchingOptions;
    void EnsureResident();
    CloverOptions ReadOptions(const json11::Json &config) const;
    void ValidateRequest(const std::vector<int> &tokens, const GenerationConfig &config,
        const CloverOptions &options) const;
    std::vector<int> Step(ResponseContext &context, Request &request);
    void Finish(ResponseContext *context);
    void Norm(Data &input, const std::string &name, Data &output,
        Data *gate = nullptr, bool zeroCentered = true);
    void AddNorm(Data &residual, Data &input, const std::string &name, Data &output);
    void Gdn(Data &input, int layer, std::pair<Data, Data> &cache,
        bool preview, Data &output, std::pair<Data, Data> *snapshots = nullptr);
    void FullAttention(Data &input, int layer, std::pair<Data, Data> &cache,
        int position, bool preview, Data &output);
    void Moe(Data &input, int layer, bool preview, Data &output, bool graphAllowed);
};
}
#endif
