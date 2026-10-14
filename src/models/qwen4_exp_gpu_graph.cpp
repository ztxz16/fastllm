// GPU graph composition; graph execution and attention math are shared.
#include "qwen4_exp_graph.h"
#include "qwen4_exp_execution.h"
#include "models/qwen4_paged_cache.h"
#include "utils.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "fastllm-cuda-mtp.cuh"
#endif
namespace fastllm {
    Qwen4ExpModel::DecodeGraphPolicy Qwen4ExpModel::GpuDecodeGraphPolicy() const {
        return {false, Qwen4CudaOnlyDeviceMap(moeDeviceMap) &&
                           Qwen4CudaOnlyDeviceMap(layeredMoeDeviceMap)};
    }

    Qwen4ExpModel::DraftGraphPolicy Qwen4ExpModel::GpuDraftGraphPolicy(int sequence) const {
        DraftGraphPolicy policy;
        policy.device = threadTpRank < 0 && sequence <= 4 && mtpMoeWeights.size() > 2 &&
                        mtpMoeWeights[2] && mtpMoeWeights[2]->dataDevice == DataDevice::CUDA;
        policy.tp = threadTpRank >= 0 && sequence <= 4 && !mtpMoeWeights.empty();
        // Retain the existing TP capture limits, including mixed target maps.
        policy.placementCompatible = policy.device || (Qwen4CudaOnlyDeviceMap(moeDeviceMap) &&
                                                       Qwen4CudaOnlyDeviceMap(layeredMoeDeviceMap));
        return policy;
    }

#ifdef USE_CUDA
    void Qwen4ExpModel::DecodeGraphExecution::RunGpuSegmentBody(int startLayer, int nextFullLayer,
                                                                bool firstAttentionReady) {
        Data carriedAttentionNorm;
        for (int layer = startLayer; layer < nextFullLayer; layer++) {
            ApplyDeviceMap(model.deviceMap, layer + 1, model.block_cnt);
            const std::string layerPrefix =
                model.languagePrefix + "layers." + std::to_string(layer) + ".";

            Data linearAttentionInput;
            Data linearAttentionInjection;
            Data linearAttentionOutput;
            const Data *currentAttentionOutput = nullptr;
            const Data *currentAttentionInjection = nullptr;
            if (layer == startLayer && firstAttentionReady) {
                currentAttentionOutput = &graphState->attentionOutput;
                currentAttentionInjection = &graphState->attentionInjection;
            } else {
                const std::string attentionHyperPrefix = layerPrefix + "attn_hyper_connection.";
                Data attentionNorm;
                Data *normalizedAttention = &carriedAttentionNorm;
                if (layer == startLayer) {
                    model.GroupedRMSNorm(graphState->hiddenStates[0],
                                         model.weight[attentionHyperPrefix + "hc_norm.weight"],
                                         attentionNorm);
                    normalizedAttention = &attentionNorm;
                }
                model.HyperMixNormalized(*normalizedAttention, attentionHyperPrefix,
                                         linearAttentionInput, &linearAttentionInjection);
                normalizedAttention->FreeSpace();
                AssertInFastLLM(
                    model.IsLinearAttentionLayer(layer),
                    "Qwen4 decode graph can only enter before a linear-attention layer.");
                model.RunLinearAttention(
                    layer, linearAttentionInput, pastKeyValues[layer].first,
                    pastKeyValues[layer].second, linearAttentionOutput,
                    verificationCapture == nullptr ? nullptr
                                                   : &verificationCapture->linearConvInputs[layer],
                    verificationCapture == nullptr ? nullptr
                                                   : &verificationCapture->linearAlphas[layer],
                    verificationCapture == nullptr ? nullptr
                                                   : &verificationCapture->linearBetas[layer],
                    verificationCapture == nullptr || graphSequence <= 1 ||
                            verificationCapture->runtimeCheckpoint == nullptr ||
                            layer >=
                                (int)verificationCapture->runtimeCheckpoint->linearSecond.size()
                        ? nullptr
                        : &verificationCapture->runtimeCheckpoint->linearSecond[layer]);
                currentAttentionOutput = &linearAttentionOutput;
                currentAttentionInjection = &linearAttentionInjection;
            }

            const std::string mlpHyperPrefix = layerPrefix + "mlp_hyper_connection.";
            Data mlpNorm;
            model.HyperCombineRMSNorm(graphState->hiddenStates[0], *currentAttentionOutput,
                                      *currentAttentionInjection,
                                      model.weight[mlpHyperPrefix + "hc_norm.weight"],
                                      graphState->hiddenStates[1], mlpNorm);

            Data mlpInput, mlpInjection, mlpOutput;
            model.HyperMixNormalized(mlpNorm, mlpHyperPrefix, mlpInput, &mlpInjection);
            mlpNorm.FreeSpace();
            model.RunMoE(layer, mlpInput, mlpOutput);

            if (layer + 1 == model.block_cnt) {
                const std::string finalHyperPrefix =
                    model.languagePrefix + "hyper_connection_mixer.";
                Data finalHyperNorm;
                model.HyperCombineRMSNorm(graphState->hiddenStates[1], mlpOutput, mlpInjection,
                                          model.weight[finalHyperPrefix + "hc_norm.weight"],
                                          graphState->hiddenStates[0], finalHyperNorm);
                Data finalHidden;
                model.HyperMixNormalized(finalHyperNorm, finalHyperPrefix, finalHidden, nullptr);
                finalHyperNorm.FreeSpace();
                Linear(finalHidden, model.weight["lm_head.weight"], Data(), graphState->logits);
                ToDataType(graphState->logits, DataType::FLOAT32);
            } else {
                const std::string nextAttentionHyperPrefix = model.languagePrefix + "layers." +
                                                             std::to_string(layer + 1) +
                                                             ".attn_hyper_connection.";
                model.HyperCombineRMSNorm(graphState->hiddenStates[1], mlpOutput, mlpInjection,
                                          model.weight[nextAttentionHyperPrefix + "hc_norm.weight"],
                                          graphState->hiddenStates[0], carriedAttentionNorm);
            }
        }

        if (nextFullLayer < model.block_cnt) {
            ApplyDeviceMap(model.deviceMap, nextFullLayer + 1, model.block_cnt);
            const std::string attentionHyperPrefix = model.languagePrefix + "layers." +
                                                     std::to_string(nextFullLayer) +
                                                     ".attn_hyper_connection.";
            model.HyperMixNormalized(carriedAttentionNorm, attentionHyperPrefix,
                                     graphState->attentionInput, &graphState->attentionInjection);
            carriedAttentionNorm.FreeSpace();
        }
    }
    void Qwen4ExpModel::DecodeGraphExecution::RunGpuWholeBody() {
        int startLayer = graphStartLayer;
        while (startLayer < model.block_cnt) {
            int nextFullLayer = startLayer + 1;
            while (nextFullLayer < model.block_cnt && model.IsLinearAttentionLayer(nextFullLayer)) {
                nextFullLayer++;
            }
            RunGpuSegmentBody(startLayer, nextFullLayer,
                              startLayer != graphStartLayer || !startBeforeAttention);
            if (nextFullLayer >= model.block_cnt) {
                break;
            }
            ApplyDeviceMap(model.deviceMap, nextFullLayer + 1, model.block_cnt);
            if (!runFullAttentionGraph(
                    nextFullLayer, graphState->attentionInput, pastKeyValues[nextFullLayer].first,
                    pastKeyValues[nextFullLayer].second, graphState->attentionOutput)) {
                FastllmCudaSetThreadError();
                return;
            }
            startLayer = nextFullLayer;
        }
    }
    void Qwen4ExpModel::DecodeGraphExecution::RunGpuSegment(int startLayer, int nextFullLayer,
                                                            bool wholeGraph,
                                                            int wholeGraphVariant) {
        const int segmentKey = wholeGraph ? -1 - wholeGraphVariant : startLayer;
        runCapturedSegment(segmentKey, [&]() {
            if (wholeGraph) {
                RunGpuWholeBody();
            } else {
                RunGpuSegmentBody(startLayer, nextFullLayer,
                                  startLayer != graphStartLayer || !startBeforeAttention);
            }
        });
    }
    void Qwen4ExpModel::DecodeGraphExecution::RunGpu() {
        if (wholeGraphReady) {
            RunGpuSegment(graphStartLayer, model.block_cnt, true,
                          wholeDenseGraph  ? model.indexerCompressRatio + wholeGraphSparseWidth
                          : mtpTargetGraph ? model.indexerCompressRatio
                                           : wholeGraphRemainder);

        } else {
            int startLayer = graphStartLayer;
            while (startLayer < model.block_cnt) {
                int nextFullLayer = startLayer + 1;
                while (nextFullLayer < model.block_cnt &&
                       model.IsLinearAttentionLayer(nextFullLayer)) {
                    nextFullLayer++;
                }
                RunGpuSegment(startLayer, nextFullLayer, false, 0);
                if (nextFullLayer >= model.block_cnt) {
                    break;
                }
                ApplyDeviceMap(model.deviceMap, nextFullLayer + 1, model.block_cnt);
                model.RunFullAttention(
                    nextFullLayer, graphState->attentionInput, attentionMask, positionIds,
                    qsaDeviceCompatibleMask, pastKeyValues[nextFullLayer].first,
                    pastKeyValues[nextFullLayer].second, requestState, graphState->attentionOutput,
                    verificationCapture == nullptr
                        ? nullptr
                        : &verificationCapture->qsaRawKeys[nextFullLayer]);
                startLayer = nextFullLayer;
            }
        }
    }
#endif
} // namespace fastllm
