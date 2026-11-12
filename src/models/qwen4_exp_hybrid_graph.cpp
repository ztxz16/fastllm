// Hybrid graph boundaries, expert overlap and output joins.
#include "qwen4_exp_tp.h"
#include <cstring>
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
    bool Qwen4ExpModel::UseFullHybridGraph() const {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        // Match the measured Qwen3.8 Flash Next architecture. An environment
        // left by another model must not enable this path for other Qwen variants.
        const char *enabled = std::getenv("FASTLLM_QWEN38_HYBRID_GRAPH");
        return executionMode == ExecutionMode::Hybrid && threadTpRank < 0 &&
            block_cnt == 48 && embed_dim == 2560 && num_experts == 512 &&
            num_experts_per_tok == 10 && hcCount == 4 &&
            indexerCompressRatio == 4 && indexerHeadDim == 128 &&
            GetFastllmEnv().cudaGraph && enabled && std::strcmp(enabled, "1") == 0;
#else
        return false;
#endif
    }

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
        const bool cacheShape = hiddenStates.dims.size() == 3 && hiddenStates.dims[1] > 0 &&
            hiddenStates.dims[1] <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH &&
            (hiddenStates.dims[1] == 1 || mtpTargetGraph);
        bool moeCudaCache = false, hybridMoe = false;
        if (!tpHostMoe) {
            // A resident prefix has no host-cache entry. Inspect the target
            // layers until finding the cache on the current dense-stage GPU;
            // later NUMA layers still need stable hybrid graph boundaries.
            for (auto &table : this->weights) {
                if (table.empty()) continue;
                moeCudaCache |= cacheShape && MoeCudaCacheAvailable(table);
                hybridMoe |= FastllmCudaUseMoeHybrid(table.data(), table.size());
                if (moeCudaCache && hybridMoe) break;
            }
        }
        // Disk experts execute outside each dense graph, through the same
        // stable input/output boundaries as the host expert backend. Whole
        // CUDA-resident layers use those boundaries too, without disk dispatch.
        const bool diskMoe = !tpHostMoe && !mtpTargetGraph &&
            hiddenStates.dims.size() == 3 && hiddenStates.dims[1] == 1 &&
            !DiskMoeLayerIndices(this->weights).empty();
        const bool hybridDenseGraph = (tpHostMoe || hybridMoe || diskMoe) && hiddenStates.dims.size() == 3 &&
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
        Data input, injection, output;
        const bool runPle = layer == model.pleLayer && graphState->pleEmbedding.cudaData;
        if (layer == graphStartLayer) {
            if (!runPle) model.HyperRead(graphState->hiddenStates[0],
                attentionHyperPrefix, input, &injection,nullptr,nullptr,nullptr,model.dataType);
        } else {
            // Join the shared branch with the routed result in the next
            // dense submission, before the ordinary TP reduction.
            auto &previous = graphState->hybridBoundaries[layer - 1];
            if (previous.moe.result.peerOutput)
                FastllmCudaAddTo(previous.output, *previous.moe.result.peerOutput, 1.0f);
            if (previous.moe.result.sharedDeferred)
                AddTo(previous.output, previous.moe.result.sharedOutput);
            model.ThreadTpAllReduce(previous.output);
            if (runPle) {
                model.HyperCombine(graphState->hiddenStates[1],
                    graphState->hybridBoundaries[layer - 1].output,
                    graphState->hybridBoundaries[layer - 1].injection, graphState->hiddenStates[0]);
            } else model.HyperRead(graphState->hiddenStates[1], attentionHyperPrefix,
                input, &injection, &previous.output, &previous.injection,
                &graphState->hiddenStates[0],model.dataType);
        }
        if (runPle) {
            Data pleOutput;
            if (verificationCapture) verificationCapture->pleInput.CopyFrom(graphState->hiddenStates[0]);
            model.RunPLEProjection(graphState->pleEmbedding, graphState->hiddenStates[0],
                requestState, pleOutput, verificationCapture ? &verificationCapture->pleNormalized : nullptr, true);
            AddTo(graphState->hiddenStates[0], pleOutput);
            model.HyperRead(graphState->hiddenStates[0], attentionHyperPrefix, input, &injection,nullptr,nullptr,nullptr,model.dataType);
        }
        if (model.IsLinearAttentionLayer(layer)) {
            model.RunLinearAttention(
                layer, input, pastKeyValues[layer].first, pastKeyValues[layer].second, output,
                verificationCapture ? &verificationCapture->linearConvInputs[layer] : nullptr,
                verificationCapture ? &verificationCapture->linearAlphas[layer] : nullptr,
                verificationCapture ? &verificationCapture->linearBetas[layer] : nullptr,
                mtpTargetGraph && verificationCapture->runtimeCheckpoint
                    ? &verificationCapture->runtimeCheckpoint->linearSecond[layer]
                    : nullptr, nullptr, nullptr,
                verificationCapture ? &verificationCapture->linearUpdates[layer] : nullptr);
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
        model.HyperRead(graphState->hiddenStates[0], mlpHyperPrefix,
            graphState->hybridBoundaries[layer].input, &graphState->hybridBoundaries[layer].injection,
            &output, &injection, &graphState->hiddenStates[1]);
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
        Data finalHidden;
        model.HyperRead(graphState->hiddenStates[1], prefix, finalHidden, nullptr,
            &last.output, &last.injection, &graphState->hiddenStates[0]);
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
    bool Qwen4ExpModel::DecodeGraphExecution::TryRunHybridFull() {
#if defined(USE_NUMAS) && !defined(USE_ROCM)
        if (!model.UseFullHybridGraph() || hybridEagerAttention ||
            (graphSequence != 1 && !mtpTargetGraph))
            return false;
        const int joinedKeyOffset = (model.block_cnt + 1) * (model.indexerCompressRatio + 2);
        std::vector<int> denseKeys, sharedKeys, residentKeys;
        void *owner = nullptr;
        for (int layer = graphStartLayer; layer < graphEndLayer; ++layer) {
            auto &boundary = graphState->hybridBoundaries[layer];
            if (boundary.moe.result.peerOutput || !boundary.output.cudaData) return false;
            auto &weights = model.weights[layer];
            auto *current = FastllmCudaNativeMoeGraphOwner(weights.data(), weights.size(), graphSequence);
            if (current) {
                if (!boundary.moe.result.sharedDeferred || boundary.output.dataType != FLOAT32 ||
                    (owner && current != owner)) return false;
                owner = current;
                residentKeys.push_back(-1);
            } else {
                if (weights.size() <= 2 || !weights[2] || weights[2]->dataDevice != CUDA ||
                    weights[2]->dataDeviceIds.size() != 1) return false;
                residentKeys.push_back(8 * joinedKeyOffset + layer);
            }
            const auto *previous = layer > graphStartLayer ? &graphState->hybridBoundaries[layer-1].moe : nullptr;
            const int join = previous ? previous->result.JoinMask() : 0;
            const int variant = model.IsLinearAttentionLayer(layer) ? 0 :
                mtpTargetGraph ? model.indexerCompressRatio : wholeGraphRemainder;
            denseKeys.push_back(layer + (model.block_cnt+1) * variant + join * joinedKeyOffset);
            sharedKeys.push_back((model.block_cnt+1) * (model.indexerCompressRatio+1) + layer);
        }
        const int finalKey = graphEndLayer +
            graphState->hybridBoundaries[graphEndLayer-1].moe.result.JoinMask() * joinedKeyOffset;
        auto ready = [&](int key) {
            auto it = graphState->segments.find(key);
            return it != graphState->segments.end() && it->second.captured && it->second.graph;
        };
        if (!ready(finalKey)) return false;
        for (size_t i = 0; i < denseKeys.size(); ++i)
            if (!ready(denseKeys[i]) || !ready(sharedKeys[i]) ||
                (residentKeys[i] >= 0 && !ready(residentKeys[i]))) return false;
        const int fullVariant = mtpTargetGraph ? model.indexerCompressRatio : wholeGraphRemainder;
        auto &full = graphState->hybridFullGraphs[fullVariant];
        if (full.disabled) return false;
        if (!full.graph) {
            FastllmCudaSyncCurrentThreadStream();
            std::vector<const void *> children;
            std::vector<void *> handles;
            bool ok = true;
            for (int layer = graphStartLayer; layer < graphEndLayer; ++layer) {
                const int index = layer - graphStartLayer;
                auto &boundary = graphState->hybridBoundaries[layer];
                children.push_back(graphState->segments[denseKeys[index]].graph);
                if (residentKeys[index] >= 0) {
                    children.push_back(graphState->segments[sharedKeys[index]].graph);
                    children.push_back(graphState->segments[residentKeys[index]].graph);
                    continue;
                }
                Data input;
                input.FakeFrom(boundary.input, 0);
                input.Resize({graphSequence, boundary.input.dims.back()});
                const auto shape = boundary.output.dims;
                const void *storage = boundary.output.cudaData;
                void *child = nullptr;
                void *handle = FastllmCudaCreateNativeMoeFullGraph(input, boundary.moe.index,
                    boundary.moe.score, boundary.output, model.weights[layer].data(), model.weights[layer].size(),
                    layer, graphState->segments[sharedKeys[index]].graph, &child);
                AssertInFastLLM(storage == boundary.output.cudaData,
                    "Qwen4 full hybrid graph replaced a captured MoE output");
                boundary.output.Reshape(shape);
                if (handle) handles.push_back(handle);
                if (!handle || !child) { ok = false; break; }
                children.push_back(child);
            }
            children.push_back(graphState->segments[finalKey].graph);
            if (ok) ok = full.graph.Compose(children, std::move(handles));
            else for (void *handle : handles) FastllmCudaReleaseNativeMoeFullGraph(handle);
            if (!ok) {
                full.disabled = true;
                FastllmCudaClearThreadError(); FastllmCudaClearGraphError();
                std::fprintf(stderr, "[Fastllm] Qwen4 hybrid full graph unavailable for layers %d..%d.\n",
                    graphStartLayer, graphEndLayer);
                return false;
            }
            std::fprintf(stderr, "[Fastllm] Qwen4 hybrid full graph ready: cuda:%d layers %d..%d rows=%d remainder=%d.\n",
                device, graphStartLayer, graphEndLayer, graphSequence, wholeGraphRemainder);
        }
        AssertInFastLLM(full.graph.Launch(), "Qwen4 hybrid full graph replay failed");
        return true;
#else
        return false;
#endif
    }

    void Qwen4ExpModel::DecodeGraphExecution::RunHybrid() {
        // A fully CUDA-capturable cache retains the existing GPU segment strategy.
        if (!hybridDenseGraph) {
            RunGpu();
            return;
        }

        if (TryRunHybridFull()) return;

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
            const char *remoteOption = std::getenv("FASTLLM_QWEN38_REMOTE_GRAPH");
            auto &weights = model.weights[layer];
            const bool remoteResident = remoteOption && std::strcmp(remoteOption,"1") == 0 &&
                model.threadTpRank < 0 && weights.size() > 2 && weights[2] &&
                weights[2]->dataDevice == CUDA && weights[2]->dataDeviceIds.size() == 1 &&
                weights[2]->dataDeviceIds.front() != device;
            const bool localResident = model.UseFullHybridGraph() && weights.size() > 2 && weights[2] &&
                weights[2]->dataDevice == CUDA && weights[2]->dataDeviceIds == std::vector<int>{device};
            const bool captureResident = localResident || remoteResident;
            auto runMoe = [&]() {
                Data moeOutput;
                const void *previous = boundary.output.cudaData;
                boundary.moe.outputTarget = &boundary.output;
                model.RunMoE(layer, boundary.input, moeOutput, false, nullptr, &boundary.moe,
                    captureResident ? std::function<void()>([] {}) : std::function<void()>(sharedGraph));
                // EP writes directly into this stable boundary. Resident
                // experts retain their established result conversion/copy.
                AssertInFastLLM(Qwen4PrepareDecodeGraphTensor(boundary.output, moeOutput, device) &&
                                    (previous == nullptr || previous == boundary.output.cudaData),
                                "Qwen4 hybrid MoE graph boundary changed storage.");
            };
            if (captureResident) {
                // Capture the remote routed projection after its shared
                // expert graph. The callback must not nest another capture.
                sharedGraph();
                const int remoteKey = 8 * joinedKeyOffset + layer;
                runCapturedSegment(remoteKey, runMoe);
                if (remoteResident && !boundary.remoteGraphReported && graphState->segments[remoteKey].captured) {
                    std::fprintf(stderr, "[Fastllm] Qwen4 remote expert graph ready: cuda:%d -> cuda:%d layer=%d rows=%d.\n",
                        device, weights[2]->dataDeviceIds.front(), layer, graphSequence);
                    boundary.remoteGraphReported = true;
                }
            } else runMoe();
        }
        const auto &last = graphState->hybridBoundaries[graphEndLayer - 1].moe;
        const int joinMask = last.result.JoinMask();
        runCapturedSegment(graphEndLayer + joinMask * joinedKeyOffset,
                           [&]() { RunHybridFinalBody(); });
    }
#endif
} // namespace fastllm
