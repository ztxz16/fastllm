// Independent host-expert forward: NUMA overlap, device groups and cache policy.
#include "glm5_next.h"
#include "glm5_next_tp_state.h"
#include "blocks/baseblock.h"
#include "utils.h"
#include "utils/hybrid_moe.h"
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
            Data &input, int sequence, Data &output) {
        const std::string routedDevice = SelectMoeDeviceForLayer(deviceLayer);
        if (ExecutionDeviceIsGpu(routedDevice)) {
            RunGpuMoeWithPrefix(deviceLayer, mlp, weights, biases, input, sequence, output);
            return;
        }
        const std::vector<int> outputDims = input.dims;
        input.Reshape({sequence, embed_dim});

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
        const auto &moeDevices = threadTpRank >= 0 ? threadTpOwner->devices : noDevices;
        DiskMoeCudaDeviceScope diskDevices(moeDevices);
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        NumasMoeCudaAssistScope numaDevices(threadTpRank >= 0 ? &moeDevices : nullptr);
        if (threadTpRank == 0 && routedOnNumas)
            FastllmCudaPrepareMoeExpertCache(weights.data(), weights.size(), moeDevices);
#endif
        Data routerInput, routerScores, expertIndex, expertScore;
        RouteMoe(deviceLayer, mlp, input, routerInput, routerScores, expertIndex, expertScore);

        Data routedOutput;
#if defined(USE_CUDA) && defined(USE_NUMAS)
#ifndef USE_ROCM
        if (sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH && input.dataType == DataType::BFLOAT16 &&
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
            sequence >= 1 &&
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

    void Glm5NextModel::ForwardHybridLayers(
            Data &hiddenStates, int firstLayer, int endLayer,
            const std::vector<std::vector<std::pair<Data, Data>>*> &requestCaches,
            std::vector<KdaReplayCapture> *kdaReplay,
            std::vector<Glm5NextIndexerCache> *indexer) {
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
        const bool frequencyDecode = sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH;
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
                    expertBiases[layer], normalizedFfn, sequence, ffnOutput);
            }
            FinishFfnBlock(ffnOutput, *current, ffnPost, ffnComb, *next);
            std::swap(current, next);
        }
        // Two residual swaps per layer preserve caller-owned state.
    }

    void Glm5NextModel::PrepareHybridMoeCache() {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        {
            std::vector<FastllmCudaMoeCacheLayer> layers;
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
                    layers.push_back({experts.data(), (int)experts.size(),
                                      false, swigluLimit, true});
                }
            }
            if (!layers.empty()) {
                FastllmCudaPrepareMoeCache(layers.data(), (int)layers.size(),
                    [this] { WarmupNumaMoeWeights(); }, true);
            }
        }
#endif
    }
} // namespace fastllm
