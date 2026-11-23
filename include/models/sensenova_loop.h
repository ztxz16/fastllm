#pragma once
#include "qwen3_5.h"
#include "devices/cuda/fastllm-nvfp4-prefill.h"

namespace fastllm {
class SenseNovaLoopModel final : public Qwen3_5Model {
public:
    SenseNovaLoopModel();
    ~SenseNovaLoopModel() override { ShutdownRuntime(); }
    void OnResponseContextCreated(ResponseContext *context) override;
    void OnResponseContextRemoved(ResponseContext *context) override;
    bool PopPendingResponseToken(ResponseContext *context, int previous, int &token) override;
    void InitParams() override;
    void OnModelWeightsLoaded() override;
    bool CanUseGPUForward() const override { return false; }
    long long GetAutoWarmupCudaRuntimeReserveBytes(int device, int batch) const override;
    bool SupportsKvMem() const override { return false; }
    bool UseModelSpecificScheduler() const override { return false; }
    bool TryRecordPagedPrefixCacheExtra(ResponseContext *) override { return false; }
    int QueryPagedPrefixCacheExtra(ResponseContext *, int) const override { return 0; }
    bool RestorePagedPrefixCacheExtra(ResponseContext *, int) const override { return false; }
    PagedCacheManager *GetPagedKVCacheManager(int layer, bool key) const override;
    std::vector<std::pair<int, PagedCacheManager*>> GetPagedKVCacheManagers(int layer, bool key) const override;
    std::vector<int> ForwardV2(int batch, const Data &ids,
        const std::vector<Data*> &masks, const std::vector<Data*> &positions,
        const std::vector<int> &lengths, std::vector<std::pair<Data*, Data*>> &cache,
        const std::vector<GenerationConfig> &configs, const LastTokensManager &last,
        std::vector<std::vector<float>*> *logits) override;
    std::vector<int> ForwardGPU(int batch, const Data &ids,
        const std::vector<Data*> &masks, const std::vector<Data*> &positions,
        const std::vector<int> &lengths, std::vector<std::pair<Data*, Data*>> &cache,
        const std::vector<GenerationConfig> &configs, const LastTokensManager &last,
        std::vector<std::vector<float>*> *logits) override {
        return ForwardV2(batch, ids, masks, positions, lengths, cache, configs, last, logits);
    }
    std::vector<int> ForwardMultimodal(const Data &, const Data &, const Data &,
        std::vector<std::pair<Data, Data>> &, const std::map<std::string, std::vector<Data*>> &,
        const GenerationConfig &, const LastTokensManager &, std::vector<std::vector<float>*> *) override;

private:
    bool groupFp8Prefill = false;
    FastllmNvfp4PrefillPlan nvfp4PrefillPlan;
    std::shared_ptr<FastllmCompactMarlinCache> nvfp4DecodeCache;
    std::vector<int> visits;
    int loopStart = 0, loopEnd = 0, iterations = 0, physicalLayers = 0, convKernel = 0;
    Data routerNorm, headNorm, rotaryBasis;
    struct DecodePages {
        Data qSizes, pageSizes, indices, lastLens, appendIndices, appendOffsets;
        std::vector<int> pages;
        int last = -1;
    };
    using Cache = std::vector<std::pair<Data*, Data*>>;
    using PrefixStates = std::vector<std::vector<std::pair<Data, Data>>>;
    PrefixStates verifyPrefixes;
    struct DraftRequest {
        MtpKvCache kv;
        int promptLength = 0, emitted = 0;
        int proposed = 0, accepted = 0, rounds = 0, acceptedFirst = 0;
        std::vector<int> drafts;
        std::deque<int> pending;
    };
    int draftCount = 0;
    std::mutex requestMutex;
    std::map<const Data*, std::shared_ptr<DraftRequest>> requests;
    std::vector<int> RunTarget(const Data &ids, const std::vector<Data*> &positions, const std::vector<int> &lengths,
        Cache &cache, const std::vector<GenerationConfig> &configs, const LastTokensManager &last,
        std::vector<std::vector<float>*> *logits, Data *capture, PrefixStates *prefixes, Data *verifyLogits = nullptr);
    bool HasMtpBatchPages(const std::vector<std::shared_ptr<DraftRequest>> &requests,
                          const std::vector<int> &lengths, const Cache &cache) const;
    std::vector<int> DraftForwardBatch(const std::vector<std::shared_ptr<DraftRequest>> &batchRequests,
        const Data &ids, const std::vector<Data*> &positions, const std::vector<int> &lengths,
        Cache &cache, const std::vector<GenerationConfig> &configs, const LastTokensManager &last);
    std::vector<int> SampleVerifyRows(Data &logits, const std::vector<MtpKvCache*> &proposals,
        const std::vector<GenerationConfig> &configs, int drafts, std::vector<int> &accepted);
    void FullAttention(const std::string &prefix, Data &input, const Data &rotary,
                       int slot, std::vector<Data*> &keys, std::vector<Data*> &values, DecodePages &pages, bool verify, Data &output);
    void LinearAttention(const std::string &prefix, Data &input,
                         const std::vector<Data*> &convCaches, const std::vector<Data*> &states, Data &output,
                         std::vector<std::pair<Data, Data>> *prefixes = nullptr);
    void RoutedInput(const std::string &prefix, Data &streams, Data &branch, Data &mix, bool exactRows, int promptRows = 0);
};
}
