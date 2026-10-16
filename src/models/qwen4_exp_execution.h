#pragma once
#include "qwen4_exp.h"
namespace fastllm {
    inline constexpr int kQwen4DenseCacheMaxGrowth = 8192;
    inline const std::string kQwen4MtpExpertPrefix = "mtp.layers.0.mlp.experts.";
    extern thread_local bool qwen4MtpDecodeEquivalentTarget;
    extern thread_local bool qwen4ThreadTpExecution;
    int Qwen4MtpDraftsPerStep();
    bool Qwen4PrefixCacheEnabled();
    bool Qwen4StartsWith(const std::string &value, const std::string &prefix);
    bool Qwen4EndsWith(const std::string &value, const std::string &suffix);
    bool Qwen4CudaOnlyDeviceMap(const std::map<std::string, int> &deviceMap);
    bool Qwen4CudaOrNumaOnlyDeviceMap(const std::map<std::string, int> &deviceMap);
    void Qwen4EnsureAppendCapacity(Data &cache, const Data &append, int axis, int quantum,
                                   int maxGrowth, bool geometricGrowth, int reserveTokens = 0);
    int Qwen4AxisCapacity(const Data &data, int axis);
    int Qwen4RoundUpCacheCapacity(int requiredCapacity, int quantum);
    bool Qwen4TpDecodeFitsCapacityClass(const GenerationConfig &config, int indexerBudget);
    Data Qwen4BuildCausalMask(int previousLength, int sequence);
    float Qwen4TopProbability(Data &logits, float maxLogit);
#ifdef USE_CUDA
    bool Qwen4PrepareCudaWorkspaceOnCurrentDevice(Data &workspace, DataType dataType,
                                                  const std::vector<int> &dims, int device);
    bool Qwen4PrepareDecodeGraphTensor(Data &destination, const Data &source, int device);
    bool Qwen4PrepareDecodeGraphWorkspace(Data &workspace, DataType dataType,
                                          const std::vector<int> &dims, int device);
    void Qwen4BorrowCudaStorage(Data &destination, const Data &source);
    bool Qwen4BorrowCudaTensor(const Data &source, Data &destination, bool linear,
                               bool sharePages = true);
    bool Qwen4TryExactCudaMultiLinear(const Data &input, const Data *const *weights,
                                      Data *const *outputs, int count);
#endif
    // Forward orchestration depends on this interface, not on graph/KV internals.
    struct Qwen4ExpModel::ForwardState {
        enum Action { Skip, Experts, Complete };
        std::unique_ptr<ForwardWorkspace> work;
        ForwardState(Qwen4ExpModel &model, ExecutionMode graphMode, int batch, const Data &inputIds,
                     const Data &attentionMask, const Data &positionIds,
                     std::vector<std::pair<Data, Data>> &pastKeyValues,
                     const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                     std::vector<std::vector<float> *> *retLogits, Data *targetMultiHidden,
                     std::vector<int> *allVerificationTokens,
                     std::vector<unsigned char> *verificationAccepted,
                     TargetVerificationCapture *verificationCapture, bool recordPrefixSnapshot,
                     bool allowDecodeCudaGraph, bool decodeEquivalentGdn,
                     const std::vector<int> *hostInputTokens,
                     bool materializeCausalMaskOnGraphFallback, const Data *precomputedEmbedding,
                     PrefillStageContext *prefill);
        ~ForwardState();
        int FirstLayer() const;
        int EndLayer() const;
        int BeginLayer(int &layer);
        void EndLayer(int layer);
        void RunExperts(int layer);
        void BeginHybridStep();
        void Prepare();
        void PrefetchExperts(int layer);
        std::vector<int> Finish();
    };
} // namespace fastllm
