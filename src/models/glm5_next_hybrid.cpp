// Independent host-expert forward: NUMA overlap, device groups and cache policy.
#include "glm5_next.h"
#include "glm5_next_tp_state.h"
#include "blocks/baseblock.h"
#include "utils.h"
#include "utils/hybrid_moe.h"
#include "utils/hybrid_parallel.h"
#include "devices/disk/diskdevice.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#endif
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif

namespace fastllm {
    void Glm5NextModel::RunHybridMoeWithPrefix(
            int deviceLayer, const std::string &mlp,
            std::vector<Data*> &weights,
            std::vector<Data*> &biases,
            Data &input, int sequence, Data &output, HybridPrefillContext *prefill) {
        const std::string routedDevice = SelectMoeDeviceForLayer(deviceLayer);
        if (ExecutionDeviceIsGpu(routedDevice)) {
            RunGpuMoeWithPrefix(deviceLayer, mlp, weights, biases, input, sequence, output);
            return;
        }
        const std::vector<int> outputDims = input.dims;
        input.Reshape({sequence, embed_dim});
        HybridMoePipelineGuard hostExperts(input, routedDevice,
            prefill ? prefill->hostExperts : nullptr, prefill ? prefill->firstLayer : -1);

        bool sharedReady = false;
        auto runShared = [&] {
            RunSharedMoe(deviceLayer, mlp, input, output, sharedReady);
        };
#ifdef USE_CUDA
        const bool routedOnNumas = routedDevice == "numa" || routedDevice.rfind("numa:", 0) == 0;
        // Host experts run once on rank zero. Peer ranks still contribute
        // their shared-expert partials to the caller's FFN AllReduce.
        if (threadTpRank >= 0) {
            const bool localNumasBatch = sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH && routedOnNumas;
            if (!localNumasBatch || threadTpRank != 0) runShared();
            if (localNumasBatch) {
                // Publish peer shared-expert completion without draining its
                // GPU on the host. The owner borrows a separate per-thread
                // stream; the following AllReduce waits for the owner on CPU.
                auto &tp = *threadTpOwner;
                if (threadTpRank != 0)
                    FastllmCudaEventRecordCurrentThread(tp.moeReady[threadTpRank]);
                tp.Barrier();
                if (threadTpRank == 0) {
                    for (size_t r = 1; r < tp.devices.size(); ++r) {
                        FastllmCudaSetDevice(tp.devices[r]);
                        FastllmCudaCurrentThreadStreamWaitEvent(tp.moeReady[r]);
                    }
                    FastllmCudaSetDevice(tp.devices[0]);
                }
            } else {
                FastllmCudaSyncCurrentThreadStream();
                threadTpOwner->Barrier();
            }
            if (threadTpRank != 0) {
                input.Reshape(outputDims);
                output.Reshape(outputDims);
                return;
            }
        }
#endif
        const std::vector<int> noDevices;
        const auto &moeDevices = prefill ? prefill->devices :
            (threadTpRank >= 0 ? threadTpOwner->devices : noDevices);
        DiskMoeCudaDeviceScope diskDevices(moeDevices);
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        NumasMoeCudaAssistScope numaDevices(moeDevices.empty() ? nullptr : &moeDevices);
        if (threadTpRank == 0 && routedOnNumas)
            FastllmCudaPrepareMoeExpertCache(weights.data(), weights.size(), moeDevices);
#endif
        Data routerInput, routerScores, expertIndex, expertScore;
        RouteMoe(deviceLayer, mlp, input, routerInput, routerScores, expertIndex, expertScore);

        Data routedOutput;
#if defined(USE_CUDA) && defined(USE_NUMAS)
#ifndef USE_ROCM
        if (!prefill && sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH && input.dataType == DataType::BFLOAT16 &&
            moeAtype == DataType::BFLOAT16 && routedOnNumas &&
            fastllm::RunHybridMoeOnDevices(input, expertIndex, expertScore, routedOutput,
                weights.data(), (int)weights.size(), deviceLayer, moeDevices, runShared)) {
            ApplyDeviceMap(deviceMap, deviceLayer + 1, block_cnt);
            AddTo(output, routedOutput);
            input.Reshape(outputDims);
            output.Reshape(outputDims);
            return;
        }
#endif
        const bool prefetchNumasSmallBatch =
            !prefill && sequence >= 1 &&
            sequence <= kNumasMoePrefetchMaxRows &&
            GetCudaSharedExpert() && routedOnNumas &&
            std::getenv(
                "FASTLLM_GLM5_DISABLE_NUMAS_MOE_OVERLAP") == nullptr;
        if (prefetchNumasSmallBatch) {
            PrefetchNumasMoeDecodeInput(
                input, expertIndex, expertScore, deviceLayer);
        }
#endif

        runShared();

        Data w1, w2, w3, tempInput, tempOutput;
        Data moeInputTemp, moeOutputTemp;
        if (routedDevice == "cpu") hostExperts.LockCpu();
        ApplyMoeDeviceMapForLayer(deviceLayer);
        MergeMOEBlock(
            &input, &expertIndex, &expertScore,
            &weights, &biases,
            &w1, &w2, &w3, &tempInput, &tempOutput,
            0.0f, &routedOutput, deviceLayer,
            input.dataType, moeAtype,
            &moeInputTemp, &moeOutputTemp,
            MoeGateSwiglu, false, swigluLimit, true);

        ApplyDeviceMap(deviceMap, deviceLayer + 1, block_cnt);
        AddTo(output, routedOutput);
        input.Reshape(outputDims);
        output.Reshape(outputDims);
    }

    HybridMoeDecodeStep Glm5NextModel::BeginHybridDraftMoe(
            const Data &input, int sequence) {
        HybridMoeDecodeStep step;
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        if (sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH &&
            input.dataDevice == DataDevice::CUDA && !input.dataDeviceIds.empty()) {
            FastllmCudaSetDevice(input.dataDeviceIds[0]);
            step.Begin(mtpExpertWeights.data(), mtpExpertWeights.size(), num_experts_per_tok);
        }
#endif
        return step;
    }

    void Glm5NextModel::ForwardHybridLayers(
            Data &hiddenStates, int firstLayer, int endLayer,
            const std::vector<std::vector<std::pair<Data, Data>>*> &requestCaches,
            std::vector<KdaReplayCapture> *kdaReplay,
            std::vector<Glm5NextIndexerCache> *indexer, HybridPrefillContext *prefill) {
        const int sequence = hiddenStates.dims[1];
        const int batch = (int)requestCaches.size();
        AssertInFastLLM(batch == 1 || (batch == sequence &&
            kdaReplay == nullptr && indexer == nullptr),
            "GLM-5.3 batched layers require one decode token per request.");
        Data hiddenStatesTemp;
        Data *current = &hiddenStates;
        Data *next = &hiddenStatesTemp;
        HybridMoeDecodeGroup decodeCaches;
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        const bool frequencyDecode = !prefill && sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH;
#endif
        for (int layer = firstLayer; layer < endLayer; layer++) {
            ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
            const std::string prefix = languagePrefix + "layers." +
                std::to_string(layer) + ".";

            RunAttentionBlock(layer, prefix, *current, *next, requestCaches, kdaReplay, indexer);
            std::swap(current, next);

            Data normalizedFfn, ffnPost, ffnComb;
            PrepareFfnBlock(prefix, *current, normalizedFfn, ffnPost, ffnComb);
            Data ffnOutput;
            if (denseMlpLayers[layer]) {
                RunClampedMlp(
                    normalizedFfn,
                    weight[prefix + "mlp.gateup_proj.weight"],
                    weight[prefix + "mlp.down_proj.weight"],
                    ffnOutput);
            } else {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
                auto &experts = expertWeights[layer];
                if (threadTpRank <= 0 && frequencyDecode && normalizedFfn.dataDevice == DataDevice::CUDA &&
                    !normalizedFfn.dataDeviceIds.empty() && experts.size() >= 4 && experts[2] &&
                    (experts[2]->dataType == DataType::DATA_GGUF_FORMAT ||
                     experts[2]->dataType == DataType::NVFP4_BLOCK_16_E4M3_PACKED)) {
                    const int device = normalizedFfn.dataDeviceIds[0];
                    if (!decodeCaches.Contains(device)) {
                        FastllmCudaSetDevice(device);
                        decodeCaches.Begin(device, experts.data(), experts.size(), num_experts_per_tok,
                            threadTpRank >= 0 ? &threadTpOwner->devices : nullptr);
                    }
                }
#endif
                RunHybridMoeWithPrefix(layer, prefix + "mlp.", expertWeights[layer],
                    expertBiases[layer], normalizedFfn, sequence, ffnOutput, prefill);
            }
            FinishFfnBlock(ffnOutput, *current, ffnPost, ffnComb, *next);
            std::swap(current, next);
        }
        // Two residual swaps per layer preserve caller-owned state.
    }

    void Glm5NextModel::PrepareHybridMoeCache() {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        {
            HybridMoeCachePlan cache(HybridMoeMath::Glm5, swigluLimit);
            const int cacheLayers = block_cnt + (mtpEnabled && mtpWeightsReady ? 1 : 0);
            for (int layer = 0; layer < cacheLayers; ++layer) {
                const std::string device = SelectMoeDeviceForLayer(std::min(layer, block_cnt - 1));
                const auto &experts = layer == block_cnt ? mtpExpertWeights : expertWeights[layer];
                if ((device == "numa" || device.rfind("numa:", 0) == 0) &&
                    experts.size() >= 4 && experts[2] &&
                    (experts[2]->dataType == DataType::NVFP4_BLOCK_16_E4M3_PACKED ||
                     experts[2]->dataType == DataType::DATA_GGUF_FORMAT)) {
                    // A draft layer may use a different GGUF quantization from
                    // the target. Leave unsupported layers on the ordinary NUMA
                    // path without disabling the target's dynamic GPU offload.
                    if (experts[2]->dataType == DataType::DATA_GGUF_FORMAT &&
                        (!experts[3] || experts[2]->dims.size() != 2 || experts[3]->dims.size() != 2 ||
                         !FastllmCudaMoeGlm5GGUFCacheSupported(experts[2]->ggmlType, experts[3]->ggmlType,
                             experts[2]->dims[1], experts[3]->dims[1]))) continue;
                    cache.Add(experts, device);
                }
            }
            cache.Prepare([this] { WarmupNumaMoeWeights(); }, true);
        }
#endif
    }

    bool Glm5NextModel::TryForwardHybridChunkedPrefill(
            const Data &inputIds, const Data &attentionMask,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *logits, int &outputToken) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        const int chunkSize = GetChunkedPrefillSize();
        if (threadTpState || threadTpRank >= 0 || !generationConfig.IsSimpleGreedy() ||
            mtpEnabled || saveHistoryChat || !useCompressedMla || GetKVCacheInCPU() ||
            GetHistoryCacheInCPU() || GetLowMemMode() || GetFastllmEnv().printProfile ||
            FastllmCudaGraphIsCapturingFast() || inputIds.dataDevice != DataDevice::CPU ||
            inputIds.dataType != DataType::FLOAT32 || !inputIds.cpuData ||
            inputIds.dims.size() != 2 || inputIds.dims[0] != 1 ||
            chunkSize <= 0 || inputIds.dims[1] <= chunkSize || !attentionMask.dims.empty()) return false;
        std::vector<CudaPrefillStage> stages;
        if (!BuildCudaPrefillStages(deviceMap, block_cnt, stages)) return false;
        bool numa = false;
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (denseMlpLayers[layer]) continue;
            const auto device = SelectMoeDeviceForLayer(layer);
            const bool host = device == "cpu" || device == "numa" || device.rfind("numa:", 0) == 0;
            numa |= device == "numa" || device.rfind("numa:", 0) == 0;
            if (!host && device != SelectDeviceFromMap(deviceMap, layer + 1, block_cnt)) return false;
        }
        for (const auto &cache : pastKeyValues)
            if (!cache.first.dims.empty() || !cache.second.dims.empty() ||
                !cache.first.pageIndex.empty() || !cache.second.pageIndex.empty()) return false;
        // Resolve shared weight transformations and request registry entries
        // before workers acquire disjoint layer ranges. Never retry after Run.
        const int previousDevice = FastllmCudaGetDevice();
        struct RestoreDevice { int device; ~RestoreDevice() { FastllmCudaSetDevice(device); } } restore{previousDevice};
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (!kdaLayers[layer]) {
                ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
                PrepareMlaWeights(layer);
            }
        }
        if ((int)pastKeyValues.size() < block_cnt) pastKeyValues.resize(block_cnt);
        std::lock_guard<std::mutex> indexerGuard(indexerCachesMutex);
        auto &indexer = indexerCaches[&pastKeyValues];
        if (indexer.empty()) indexer.resize(block_cnt);
        std::vector<int> devices;
        for (const auto &stage : stages) devices.push_back(stage.device);
        HybridPrefillScope hybrid(devices, numa);
        if (!prefillPipeline) prefillPipeline.reset(new CudaChunkedPrefillPipeline());
        const int sequence = inputIds.dims[1];
        const int chunks = 1 + (sequence - 1) / chunkSize;
        const float *tokens = reinterpret_cast<const float *>(inputIds.cpuData);
        if (verbose) {
            printf("[Prompt] Hybrid pipelined prefill: %zu CUDA stages, %d chunks (%d tokens/chunk).\n",
                   stages.size(), chunks, chunkSize);
            fflush(stdout);
        }
        prefillPipeline->Run(devices, chunks, DataType::BFLOAT16,
            {1, chunkSize, hcMult, embed_dim}, [&](int rank, int chunk, Data &hidden) {
                if (chunk == 0) hybrid.BindWorker(rank);
                const auto &stage = stages[rank];
                HybridPrefillContext context{stage.firstLayer, &hybrid.hostExperts, {stage.device}};
                if (rank == 0) {
                    const int start = chunk * chunkSize;
                    const int length = std::min(chunkSize, sequence - start);
                    Data input(DataType::FLOAT32, {1, length},
                        std::vector<float>(tokens + start, tokens + start + length));
                    std::lock_guard<std::mutex> lock(hybrid.hostExperts);
                    ForwardEmbedding(input, hidden);
                }
                ForwardHybridLayers(hidden, stage.firstLayer, stage.endLayer,
                    {&pastKeyValues}, nullptr, &indexer, &context);
                if (rank + 1 == (int)stages.size() && chunk + 1 == chunks)
                    outputToken = ForwardOutput(hidden, generationConfig, lastTokens, logits, true);
            });
        return true;
#else
        return false;
#endif
    }
} // namespace fastllm
