#pragma once
#include "qwen4_exp.h"
namespace fastllm {
int Qwen4MtpDraftsPerStep();
// Forward orchestration depends on this interface, not on graph/KV internals.
struct Qwen4ExpModel::ForwardState {
    enum Action { Skip, Experts, Complete };
    std::unique_ptr<ForwardWorkspace> work;
    ForwardState(Qwen4ExpModel &model,
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<std::vector<float> *> *retLogits,
            Data *targetMultiHidden,
            std::vector<int> *allVerificationTokens,
            std::vector<unsigned char> *verificationAccepted,
            TargetVerificationCapture *verificationCapture,
            bool recordPrefixSnapshot,
            bool allowDecodeCudaGraph,
            bool decodeEquivalentGdn,
            const std::vector<int> *hostInputTokens,
            bool materializeCausalMaskOnGraphFallback,
            const Data *precomputedEmbedding, PrefillStageContext *prefill);
    ~ForwardState();
    int FirstLayer() const;
    int EndLayer() const;
    int BeginLayer(int &layer);
    void EndLayer(int layer);
    void GpuExperts(int layer);
    void HybridExperts(int layer);
    void BeginHybridStep();
    void Prepare();
    void PrefetchExperts(int layer);
    std::vector<int> Finish();
};
}
