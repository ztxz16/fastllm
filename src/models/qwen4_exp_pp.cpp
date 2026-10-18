// Layer-partitioned chunked prefill pipeline.
#include "qwen4_exp_execution.h"
#include "qwen4_exp_hybrid.h"
#include "models/qwen4_paged_cache.h"
#include "executor.h"
#include "utils.h"
#ifdef USE_CUDA
#include "fastllm-multicuda.cuh"
#endif
namespace fastllm {
    bool Qwen4ExpModel::TryForwardChunkedPrefill(const Data &inputIds, const Data &attentionMask,
                                                 const Data &positionIds,
                                                 std::vector<std::pair<Data, Data>> &pastKeyValues,
                                                 const GenerationConfig &generationConfig,
                                                 const LastTokensManager &lastTokens,
                                                 std::vector<float> *logits, int &outputToken) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        const int chunkSize = GetChunkedPrefillSize();
        if (threadTpState || threadTpRank >= 0 ||
            !generationConfig.IsSimpleGreedy() || saveHistoryChat || GetKVCacheInCPU() ||
            GetHistoryCacheInCPU() || GetLowMemMode() || GetFastllmEnv().printProfile ||
            FastllmCudaGraphIsCapturingFast() || inputIds.dataDevice != DataDevice::CPU ||
            inputIds.dataType != FLOAT32 || !inputIds.cpuData || inputIds.dims.size() != 2 ||
            inputIds.dims[0] != 1 || pastKeyValues.size() < size_t(block_cnt) || chunkSize <= 0 ||
            inputIds.dims[1] <= chunkSize || !attentionMask.dims.empty() ||
            positionIds.dataDevice != DataDevice::CPU || positionIds.dataType != FLOAT32 ||
            !positionIds.cpuData || positionIds.dims != inputIds.dims)
            return false;

        std::vector<CudaPrefillStage> stages;
        if (!BuildCudaPrefillStages(deviceMap, block_cnt, stages))
            return false;
        for (const auto &kv : pastKeyValues)
            if (!kv.first.dims.empty() || !kv.second.dims.empty() || !kv.first.pageIndex.empty() ||
                !kv.second.pageIndex.empty())
                return false;
        PrepareWeights();
        const bool mtp = Qwen4MtpDraftsPerStep() > 0;
        // Keep shared CUDA embedding migration and unsupported MTP modes on
        // their serial path. The hybrid auto policy selects CPU embedding.
        if (mtp && (executionMode != ExecutionMode::Hybrid || GetCudaEmbedding() ||
                    !MtpSupportsGenerationConfig(generationConfig)))
            return false;
        RequestState *request;
        {
            std::lock_guard<std::mutex> lock(stateMutex);
            request = &requestStates[&pastKeyValues[0].first];
            if (request->hasMultimodalInput || request->borrowedPrefixSnapshot ||
                !request->processedTokens.empty() || request->mtpState ||
                (mtp && request->mtpDisabled))
                return false;
        }

        const int sequence = inputIds.dims[1];
        const int chunks = 1 + (sequence - 1) / chunkSize;
        Qwen4CudaDeviceGuard restoreDevice;
        for (const auto &stage : stages) {
            FastllmCudaSetDevice(stage.device);
            for (int layer = stage.firstLayer; layer < stage.endLayer; ++layer) {
                const std::string moe = SelectMoeDeviceForLayer(layer);
                const std::string dense = SelectDeviceFromMap(deviceMap, layer + 1, block_cnt);
                if (moe.empty() || moe == dense)
                    continue;
                const bool host = moe == "cpu" || moe == "numa" || moe.rfind("numa:", 0) == 0;
                // Each pipeline GPU has one compute owner. Resident experts
                // on another stage would share its allocator, cuBLAS workspace
                // and copies with a concurrent worker. Keep that layout on the
                // serial chunked path, before acquiring a slot or appending KV.
                if (!host)
                    return false;
            }
        }

        // Pipeline prefill and decode must own the same fixed cache slot.
        // All fallback checks above precede slot acquisition and worker launch.
        return WithDecodeServingCache(pastKeyValues, [&](auto &cache) {
            {
                std::lock_guard<std::mutex> lock(stateMutex);
                request = &requestStates[&cache[0].first];
            }
            // Once workers start, never retry a partially appended request through
            // serial Forward. All fallback checks must precede this boundary.
            if (mtp && !request->mtpState)
                request->mtpState = std::make_shared<MtpRuntimeState>();
            AcquireServingCache(cache, *request, sequence);
            std::vector<RequestState> states(stages.size(), *request);
            std::vector<PrefillStageContext> contexts(stages.size());
            std::mutex hostExpertMutex;
            std::vector<int> devices;
            int pleOwner = 0;
            for (int rank = 0; rank < int(stages.size()); ++rank) {
                const auto &stage = stages[rank];
                devices.push_back(stage.device);
                contexts[rank].firstLayer = stage.firstLayer;
                contexts[rank].endLayer = stage.endLayer;
                contexts[rank].state = &states[rank];
                contexts[rank].hostExpertMutex = &hostExpertMutex;
                if (stage.firstLayer <= pleLayer && pleLayer < stage.endLayer)
                    pleOwner = rank;
                FastllmCudaSetDevice(stage.device);
                auto &scale = prefillLinearInvScales[stage.device];
                if (scale.dims.empty()) {
                    scale.CopyFrom(linearInvScaleData);
                    scale.ToDevice(CUDA, std::vector<int>{stage.device});
                }
            }
            FastllmCudaPeerAccessInit(devices);
            bool numa = false;
            for (int layer = 0; layer < block_cnt; ++layer) {
                const auto device = SelectMoeDeviceForLayer(layer);
                numa |= device == "numa" || device.rfind("numa:", 0) == 0;
            }
            HybridPrefillScope hybrid(devices, numa);
            if (!prefillPipeline)
                prefillPipeline.reset(new CudaChunkedPrefillPipeline());
            const DataType type =
                weight[languagePrefix + "embed_tokens.weight"].dataType == FLOAT16 ? FLOAT16 : FLOAT32;
            const float *tokens = reinterpret_cast<const float *>(inputIds.cpuData);
            const float *positions = reinterpret_cast<const float *>(positionIds.cpuData);
            std::vector<std::vector<float> *> batchLogits{logits};
            auto joinStates = [&]() {
                // Only the PLE stage advances token/short-convolution history.
                // Attention maps belong to disjoint layer ranges. The last
                // stage needs the complete state before MTP's final snapshot.
                RequestState joined = std::move(states[pleOwner]);
                for (int rank = 0; rank < int(states.size()); ++rank) {
                    if (rank == pleOwner) continue;
                    auto &other = states[rank];
                    joined.indexerRawKeys.merge(other.indexerRawKeys);
                    joined.indexerPositions.merge(other.indexerPositions);
                    joined.indexerBlockKeys.merge(other.indexerBlockKeys);
                    joined.indexerHostMirrorTransfers.merge(other.indexerHostMirrorTransfers);
                    joined.indexerTailKeyTensors.merge(other.indexerTailKeyTensors);
                    joined.indexerTailPositionTensors.merge(other.indexerTailPositionTensors);
                    joined.indexerBlockKeyTensors.merge(other.indexerBlockKeyTensors);
                    joined.geometricCacheGrowthReadyLayers.merge(other.geometricCacheGrowthReadyLayers);
                }
                states.back() = std::move(joined);
            };
            if (verbose) {
                std::printf(
                    "[Prompt] Qwen4 pipelined prefill: %zu CUDA stages, %d chunks (%d tokens/chunk).\n",
                    stages.size(), chunks, chunkSize);
                std::fflush(stdout);
            }
            prefillPipeline->Run(devices, chunks, type, {1, chunkSize, hcCount * embed_dim},
                                 [&](int rank, int chunk, Data &hidden) {
                                     if (chunk == 0)
                                         hybrid.BindWorker(rank);
                                     const int start = chunk * chunkSize;
                                     const int end = start + std::min(chunkSize, sequence - start);
                                     Data ids(FLOAT32, {1, end - start},
                                              std::vector<float>(tokens + start, tokens + end));
                                     Data pos(FLOAT32, {1, end - start},
                                              std::vector<float>(positions + start, positions + end));
                                     auto &context = contexts[rank];
                                     context.hidden = &hidden;
                                     context.sampleOutput =
                                         rank + 1 == int(stages.size()) && chunk + 1 == chunks;
                                     // Receiving the final chunk means every earlier
                                     // stage has published its last state and activation.
                                     // Join before the last MTP call so its snapshot and
                                     // proposal generation reuse the ordinary semantics.
                                     if (mtp && context.sampleOutput) joinStates();
                                     const auto result = mtp && rank + 1 == int(stages.size())
                                         ? ForwardBatchInternal(1, ids, Data(), pos, cache,
                                             generationConfig, lastTokens, &batchLogits, &context)
                                         : ForwardTarget(1, ids, Data(), pos, cache, generationConfig,
                                             lastTokens, &batchLogits, nullptr, nullptr, nullptr, nullptr,
                                             false, false, false, nullptr, false, nullptr, &context);
                                     if (context.sampleOutput)
                                         outputToken = result.front();
                                 });

            if (!mtp) joinStates();
            *request = std::move(states.back());
            return true;
        });
#else
        return false;
#endif
    }

} // namespace fastllm
