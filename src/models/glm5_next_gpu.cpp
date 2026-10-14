// Independent GPU forward and resident expert execution. Math blocks stay shared.
#include "glm5_next.h"
#include "glm5_next_tp_state.h"
#include "blocks/baseblock.h"
#include "utils.h"

namespace fastllm {
    void Glm5NextModel::RunGpuMoeWithPrefix(
            int deviceLayer, const std::string &mlp,
            std::vector<Data*> &weights,
            std::vector<Data*> &biases,
            Data &input, int sequence, Data &output) {
        const std::vector<int> outputDims = input.dims;
        input.Reshape({sequence, embed_dim});

        bool sharedReady = false;
        auto runShared = [&] {
            RunSharedMoe(deviceLayer, mlp, input, output, sharedReady);
        };
#ifdef USE_CUDA
        const std::string routedDevice = SelectMoeDeviceForLayer(deviceLayer);
        const bool shardedRouted = threadTpRank >= 0 && threadTpOwner->shardedMoeLayers[deviceLayer];
        // Resident TP ranks either own a channel shard, or let rank zero
        // execute the shared full expert while peers compute shared partials.
        if (threadTpRank >= 0 && !shardedRouted) {
            runShared();
            FastllmCudaSyncCurrentThreadStream();
            threadTpOwner->Barrier();
            if (threadTpRank != 0) {
                input.Reshape(outputDims);
                output.Reshape(outputDims);
                return;
            }
        }
#endif
        Data routerInput, routerScores, expertIndex, expertScore;
        RouteMoe(deviceLayer, mlp, input, routerInput, routerScores, expertIndex, expertScore);

        Data routedOutput;
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if ((routedDevice == "cuda" || routedDevice.rfind("cuda:", 0) == 0) &&
            weights.size() >= 4 && weights[2] && weights[2]->dataType == DataType::DATA_GGUF_FORMAT) {
            runShared();
            const int previous = FastllmCudaGetDevice();
            struct RestoreDevice { int device; ~RestoreDevice() { FastllmCudaSetDevice(device); } } restore{previous};
            const auto &owner = weights[2]->dataDeviceIds;
            AssertInFastLLM(weights[2]->dataDevice == DataDevice::CUDA && owner.size() == 1,
                "GLM GGUF resident experts must be loaded on one CUDA device.");
            // Copy only activations when the MoE owner differs from the dense
            // rank; the original TP input remains on its rank's device.
            Data localInput, activation, workspace;
            Data *routedInput = &input;
            if (input.dataDevice != DataDevice::CUDA || input.dataDeviceIds != owner) {
                localInput.CopyFrom(input);
                localInput.ToDevice(DataDevice::CUDA, owner);
                routedInput = &localInput;
            }
            expertIndex.ToDevice(DataDevice::CUDA, owner);
            expertScore.ToDevice(DataDevice::CUDA, owner);
            FastllmCudaSetDevice(owner[0]);
            AssertInFastLLM(moeAtype == DataType::BFLOAT16 &&
                FastllmCudaMergeMOEGlm5GGUFResident(*routedInput, activation, workspace,
                    routedOutput, weights.data(), (int)weights.size(),
                    static_cast<const int32_t *>(expertIndex.cudaData),
                    static_cast<const float *>(expertScore.cudaData), num_experts_per_tok, swigluLimit),
                "Unsupported GLM GGUF resident MoE dtype, quantization or shape.");
            routedOutput.ToDevice(DataDevice::CUDA, output.dataDeviceIds);
            FastllmCudaSetDevice(previous);
            ApplyDeviceMap(deviceMap, deviceLayer + 1, block_cnt);
            AddTo(output, routedOutput);
            input.Reshape(outputDims);
            output.Reshape(outputDims);
            return;
        }
#endif

        runShared();

        Data w1, w2, w3, tempInput, tempOutput;
        Data moeInputTemp, moeOutputTemp;
#ifdef USE_CUDA
        if (shardedRouted) ApplyDeviceMap(deviceMap, deviceLayer + 1, block_cnt);
        else
#endif
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

    void Glm5NextModel::ForwardGpuLayers(
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
                RunGpuMoeWithPrefix(layer, prefix + "mlp.", expertWeights[layer],
                    expertBiases[layer], normalizedFfn, sequence, ffnOutput);
            }
            FinishFfnBlock(ffnOutput, *current, ffnPost, ffnComb, *next);
            std::swap(current, next);
        }
        // Two residual swaps per layer preserve caller-owned state.
    }

} // namespace fastllm
