// Hybrid graph boundaries, expert overlap and output joins.
#include "qwen4_exp_tp.h"
#include "qwen4_exp_graph.h"
#include "qwen4_exp_execution.h"
#include "models/qwen4_paged_cache.h"
#include "utils.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#ifndef USE_ROCM
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-moe-prefetch.h"
#endif
#include "fastllm-cuda-mtp.cuh"
#endif
namespace fastllm {
    Qwen4ExpModel::DecodeGraphPolicy
    Qwen4ExpModel::HybridDecodeGraphPolicy(const Data &hiddenStates, bool mtpTargetGraph) {
#ifndef USE_CUDA
        return {};
#else
        // Check the supported cache shape before querying availability,
        // which lazily allocates the device cache. Cached verifier rows run
        // entirely on CUDA, so their configured NUMA fallback is graph-safe.
        // TP host experts run outside the dense graph. Cache availability
        // must not change graph eligibility between ranks with different
        // per-device budgets; expert dispatch handles their cache fallback.
        const bool tpHostMoe =
            threadTpRank >= 0 && threadTpOwner != nullptr && threadTpOwner->hybrid.hostMoe;
        const bool moeCudaCache =
            !tpHostMoe && hiddenStates.dims.size() == 3 && hiddenStates.dims[1] > 0 &&
            hiddenStates.dims[1] <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH &&
            (hiddenStates.dims[1] == 1 || mtpTargetGraph) && !this->weights.empty() &&
            !this->weights[0].empty() && MoeCudaCacheAvailable(this->weights[0]);
        const bool hybridMoe =
            !tpHostMoe && !this->weights.empty() && !this->weights[0].empty() &&
            FastllmCudaUseMoeHybrid(this->weights[0].data(), this->weights[0].size());
        const bool hybridDenseGraph = (tpHostMoe || hybridMoe) && hiddenStates.dims.size() == 3 &&
                                      (hiddenStates.dims[1] == 1 || mtpTargetGraph);
        const bool moeDeviceMapGraphCompatible =
            (Qwen4CudaOnlyDeviceMap(this->moeDeviceMap) &&
             Qwen4CudaOnlyDeviceMap(this->layeredMoeDeviceMap)) ||
            (moeCudaCache && Qwen4CudaOrNumaOnlyDeviceMap(this->moeDeviceMap) &&
             Qwen4CudaOrNumaOnlyDeviceMap(this->layeredMoeDeviceMap)) ||
            hybridDenseGraph;
        return {hybridDenseGraph, (!hybridMoe || hybridDenseGraph) && moeDeviceMapGraphCompatible};
#endif
    }

    Qwen4ExpModel::DraftGraphPolicy Qwen4ExpModel::HybridDraftGraphPolicy(int sequence) {
        DraftGraphPolicy policy;
#ifdef USE_CUDA
        const bool hostExperts = threadTpOwner && threadTpOwner->hybrid.hostMoe &&
                                 !threadTpOwner->mtpGpuExperts;
        const bool hybridExperts =
            !mtpMoeWeights.empty() &&
            FastllmCudaUseMoeHybrid(mtpMoeWeights.data(), mtpMoeWeights.size());
        policy.device = threadTpRank < 0 && sequence <= 4 && mtpMoeWeights.size() > 2 &&
                        mtpMoeWeights[2] && mtpMoeWeights[2]->dataDevice == DataDevice::CUDA &&
                        !hybridExperts;
        policy.tp = threadTpRank >= 0 && sequence <= 4 && !mtpMoeWeights.empty() && !hybridExperts;
        policy.hybrid = sequence <= 4 && !mtpMoeWeights.empty() && (hostExperts || hybridExperts);
        policy.placementCompatible =
            policy.device || policy.tp || (policy.hybrid ? Qwen4CudaOrNumaOnlyDeviceMap(moeDeviceMap) &&
                                                  Qwen4CudaOrNumaOnlyDeviceMap(layeredMoeDeviceMap)
                                            : Qwen4CudaOnlyDeviceMap(moeDeviceMap) &&
                                                  Qwen4CudaOnlyDeviceMap(layeredMoeDeviceMap));
#endif
        return policy;
    }

#ifdef USE_CUDA
    void Qwen4ExpModel::DecodeGraphExecution::RunHybridDenseBody(int layer) {
        ApplyDeviceMap(model.deviceMap, layer + 1, model.block_cnt);
        const std::string attentionHyperPrefix =
            model.languagePrefix + "layers." + std::to_string(layer) + ".attn_hyper_connection.";
        Data attentionNorm;
        if (layer == graphStartLayer) {
            model.GroupedRMSNorm(graphState->hiddenStates[0],
                                 model.weight[attentionHyperPrefix + "hc_norm.weight"],
                                 attentionNorm);
        } else {
            // Join the shared branch with the routed result in the next
            // dense submission, before the ordinary TP reduction.
            auto &previous = graphState->hybridBoundaries[layer - 1];
            if (previous.moe.result.peerOutput)
                FastllmCudaAddTo(previous.output, *previous.moe.result.peerOutput, 1.0f);
            if (previous.moe.result.sharedDeferred)
                AddTo(previous.output, previous.moe.result.sharedOutput);
            model.ThreadTpAllReduce(previous.output);
            model.HyperCombineRMSNorm(graphState->hiddenStates[1],
                                      graphState->hybridBoundaries[layer - 1].output,
                                      graphState->hybridBoundaries[layer - 1].injection,
                                      model.weight[attentionHyperPrefix + "hc_norm.weight"],
                                      graphState->hiddenStates[0], attentionNorm);
        }
        Data input, injection, output;
        model.HyperMixNormalized(attentionNorm, attentionHyperPrefix, input, &injection);
        attentionNorm.FreeSpace();
        if (model.IsLinearAttentionLayer(layer)) {
            model.RunLinearAttention(
                layer, input, pastKeyValues[layer].first, pastKeyValues[layer].second, output,
                verificationCapture ? &verificationCapture->linearConvInputs[layer] : nullptr,
                verificationCapture ? &verificationCapture->linearAlphas[layer] : nullptr,
                verificationCapture ? &verificationCapture->linearBetas[layer] : nullptr,
                mtpTargetGraph && verificationCapture->runtimeCheckpoint
                    ? &verificationCapture->runtimeCheckpoint->linearSecond[layer]
                    : nullptr);
            if (FastllmCudaGetThreadError())
                return;
        } else if (hybridEagerAttention) {
            // Padding the dense KV length changes GEMM reduction shapes.
            // Preserve the ordinary path until QSA selects a fixed width;
            // linear-attention segments remain independently capturable.
            model.RunFullAttention(
                layer, input, hybridCausalMask ? *hybridCausalMask : attentionMask,
                graphState->positionIds, qsaDeviceCompatibleMask, pastKeyValues[layer].first,
                pastKeyValues[layer].second, requestState, output,
                verificationCapture ? &verificationCapture->qsaRawKeys[layer] : nullptr);
        } else if (!runFullAttentionGraph(layer, input, pastKeyValues[layer].first,
                                          pastKeyValues[layer].second, output)) {
            FastllmCudaSetThreadError();
            return;
        }
        const std::string mlpHyperPrefix =
            model.languagePrefix + "layers." + std::to_string(layer) + ".mlp_hyper_connection.";
        Data mlpNorm;
        model.HyperCombineRMSNorm(graphState->hiddenStates[0], output, injection,
                                  model.weight[mlpHyperPrefix + "hc_norm.weight"],
                                  graphState->hiddenStates[1], mlpNorm);
        model.HyperMixNormalized(mlpNorm, mlpHyperPrefix, graphState->hybridBoundaries[layer].input,
                                 &graphState->hybridBoundaries[layer].injection);
        auto &boundary = graphState->hybridBoundaries[layer];
        model.PrepareMoERouting(layer, boundary.input, boundary.moe);
#ifndef USE_ROCM
        boundary.moe.readbackPrepared =
            model.TryPipelineMoeReadback(boundary.input, boundary.moe.index, boundary.moe.score);
#endif
    }
    void Qwen4ExpModel::DecodeGraphExecution::RunHybridFinalBody() {
        auto &last = graphState->hybridBoundaries[graphEndLayer - 1];
        if (last.moe.result.peerOutput)
            FastllmCudaAddTo(last.output, *last.moe.result.peerOutput, 1.0f);
        if (last.moe.result.sharedDeferred)
            AddTo(last.output, last.moe.result.sharedOutput);
        model.ThreadTpAllReduce(last.output);
        if (graphEndLayer < model.block_cnt) {
            // PLE and device handoff consume only the residual. The next
            // stage computes its own norm after the transfer/PLE update.
            model.HyperCombine(graphState->hiddenStates[1], last.output, last.injection,
                               graphState->hiddenStates[0]);
            return;
        }
        const std::string prefix = model.languagePrefix + "hyper_connection_mixer.";
        Data normalized, finalHidden;
        model.HyperCombineRMSNorm(
            graphState->hiddenStates[1], graphState->hybridBoundaries[model.block_cnt - 1].output,
            graphState->hybridBoundaries[model.block_cnt - 1].injection,
            model.weight[prefix + "hc_norm.weight"], graphState->hiddenStates[0], normalized);
        model.HyperMixNormalized(normalized, prefix, finalHidden, nullptr);
        Linear(finalHidden, model.weight["lm_head.weight"], Data(), graphState->logits);
        ToDataType(graphState->logits, DataType::FLOAT32);
    }
    void Qwen4ExpModel::DecodeGraphExecution::PrepareHybridMask() {
        if (mtpTargetGraph && hybridEagerAttention && !linearOnlyStage &&
            attentionMask.dims.empty()) {
            // Dense attention remains eager and needs its explicit causal
            // mask; only the sparse graph reads causality from decodeMeta.
            hybridCausalMask =
                std::make_unique<Data>(Qwen4BuildCausalMask(decodePreviousLength, graphSequence));
        }
    }
    void Qwen4ExpModel::DecodeGraphExecution::RunHybrid() {
        // A fully CUDA-capturable cache retains the existing GPU segment strategy.
        if (!hybridDenseGraph) {
            RunGpu();
            return;
        }

        // EP acceptance is collective in TP. Keep the fallback join
        // topology separate if an expert backend becomes unavailable.
        const int joinedKeyOffset = (model.block_cnt + 1) * (model.indexerCompressRatio + 2);
        for (int layer = graphStartLayer; layer < graphEndLayer; ++layer) {
            const bool linearAttention = model.IsLinearAttentionLayer(layer);
            if (hybridEagerAttention && !linearAttention) {
                RunHybridDenseBody(layer);
            } else {
                const int variant = linearAttention  ? 0
                                    : mtpTargetGraph ? model.indexerCompressRatio
                                                     : wholeGraphRemainder;
                const auto *previous = layer > graphStartLayer
                                           ? &graphState->hybridBoundaries[layer - 1].moe
                                           : nullptr;
                const int joinMask = previous ? previous->result.JoinMask() : 0;
                const int key =
                    layer + (model.block_cnt + 1) * variant + joinMask * joinedKeyOffset;
                runCapturedSegment(key, [&]() { RunHybridDenseBody(layer); });
            }
            Data moeOutput;
            auto &boundary = graphState->hybridBoundaries[layer];
            auto sharedGraph = [&]() {
                // This branch must launch after route readback, beside
                // CPU experts. Placing it before that readback makes
                // the host wait for the entire shared projection.
                const int key = (model.block_cnt + 1) * (model.indexerCompressRatio + 1) + layer;
                runCapturedSegment(key, [&]() {
                    Data flattened, gateUp, hidden, gate;
                    flattened.FakeFrom(boundary.input, 0);
                    flattened.Resize({graphSequence, boundary.input.dims.back()});
                    model.RunSharedExpert(
                        model.languagePrefix + "layers." + std::to_string(layer) + ".mlp.",
                        flattened, gateUp, hidden, gate, boundary.moe.result.sharedOutput);
                    ToDataType(boundary.moe.result.sharedOutput, DataType::FLOAT32);
                });
            };
            const void *previous = boundary.output.cudaData;
            boundary.moe.outputTarget = &boundary.output;
            model.RunMoE(layer, boundary.input, moeOutput, false, nullptr, &boundary.moe,
                         sharedGraph);
            // EP writes directly into this stable boundary. Other
            // backends keep their existing owning result and copy.
            AssertInFastLLM(Qwen4PrepareDecodeGraphTensor(boundary.output, moeOutput, device) &&
                                (previous == nullptr || previous == boundary.output.cudaData),
                            "Qwen4 hybrid MoE graph boundary changed storage.");
        }
        const auto &last = graphState->hybridBoundaries[graphEndLayer - 1].moe;
        const int joinMask = last.result.JoinMask();
        runCapturedSegment(graphEndLayer + joinMask * joinedKeyOffset,
                           [&]() { RunHybridFinalBody(); });
    }
#endif
} // namespace fastllm
