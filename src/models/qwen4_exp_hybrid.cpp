// Independent hybrid forward; tensor blocks and CUDA kernels remain shared.
#include "qwen4_exp_execution.h"
namespace fastllm {
    std::vector<int> Qwen4ExpModel::ForwardHybrid(
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
            const Data *precomputedEmbedding, PrefillStageContext *prefill) {
        ForwardState frame(*this, batch, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig,
            lastTokens, retLogits, targetMultiHidden, allVerificationTokens, verificationAccepted,
            verificationCapture, recordPrefixSnapshot, allowDecodeCudaGraph, decodeEquivalentGdn,
            hostInputTokens, materializeCausalMaskOnGraphFallback, precomputedEmbedding, prefill);
        frame.BeginHybridStep();
        frame.Prepare();
        frame.PrefetchExperts(frame.FirstLayer());
        for (int layer = frame.FirstLayer(); layer < frame.EndLayer(); ++layer) {
            const int action = frame.BeginLayer(layer);
            if (action == ForwardState::Complete) break;
            if (action == ForwardState::Skip) continue;
            frame.HybridExperts(layer);
            frame.PrefetchExperts(layer + 1);
            frame.EndLayer(layer);
        }
        return frame.Finish();
    }
}
