// Independent gpu forward; tensor blocks and CUDA kernels remain shared.
#include "qwen4_exp_execution.h"
#include "baseblock.h"
#include "utils.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif
namespace fastllm {
    // GPU-only MoE orchestration. No NUMA locks, cache admission, host prefetch
    // or dynamic split can enter this path, including CUDA graph capture.
    void Qwen4ExpModel::RunGpuMoE(
            const MoEExecutionPlan &plan,
            const Data &input, Data &output, bool reduceOutput,
            PrefillStageContext *, MoEPrepared *prepared,
            const std::function<void()> &sharedGraph) {
        const int deviceLayer = plan.deviceLayer;
        const std::string &mlp = plan.mlpPrefix;
        auto &moeWeights = *plan.weights;
        auto &moeBiass = *plan.bias;
        Data flattened;
        flattened.FakeFrom(input, 0);
        flattened.Resize(input.dims);
        flattened.Reshape({input.dims[0] * input.dims[1], input.dims.back()});
        Data routerLogits, localIndex, localScore, localShared;
        Data &index = prepared ? prepared->index : localIndex;
        Data &score = prepared ? prepared->score : localScore;
        Data &sharedOutput = prepared ? prepared->result.sharedOutput : localShared;
        if (prepared) {
            prepared->result.Reset();
        } else {
            Linear(flattened, weight[mlp + "gate.weight"], Data(), routerLogits);
            ToDataType(routerLogits, DataType::FLOAT32);
        }
#ifdef USE_CUDA
        FastllmCudaGraphMarkParallelFork(deviceLayer);
#endif
        Data sharedGateUp, sharedHidden, sharedGate;
        if (prepared) sharedGraph();
        else RunSharedExpert(mlp, flattened, sharedGateUp, sharedHidden, sharedGate, sharedOutput);
#ifdef USE_CUDA
        FastllmCudaGraphMarkParallelFirstDone(deviceLayer);
        FastllmCudaGraphMarkParallelSecondBegin(deviceLayer);
        if (threadTpRank >= 0 && moeWeights.size() >= 4 && moeWeights[2] && moeWeights[3] &&
            moeWeights[2]->dims == std::vector<int>({0, input.dims.back()}) &&
            moeWeights[3]->dims == std::vector<int>({input.dims.back(), 0})) {
            FastllmCudaGraphMarkParallelJoin(deviceLayer);
            output.CopyFrom(sharedOutput);
            output.Reshape(input.dims);
            if (reduceOutput) ThreadTpAllReduce(output);
            return;
        }
#endif
        if (!prepared) SelectMoEExperts(routerLogits, index, score);
        const auto moeDevice = SelectSpecialWeightDevice(mlp + "experts.0.gateup_proj.weight", deviceLayer);
        const auto outputDevice = SelectDeviceFromMap(deviceMap, deviceLayer + 1, block_cnt);
        const bool direct = moeDevice == outputDevice;
        Data transferredInput;
        Data &expertInput = executionMode == ExecutionMode::Hybrid && !direct ? transferredInput : flattened;
        if (&expertInput != &flattened) {
            // A borrowed view has no allocation capacity, so ToDevice cannot
            // move it between PP stages. Own the input before the executor
            // transfers it to a resident expert on another GPU.
            expertInput.CopyFrom(flattened);
        }
        if (!moeDevice.empty()) ApplyDeviceMap({{moeDevice, 1}}, 1, 1);
        Data routed, w1, w2, w3, temporaryInput, temporaryOutput;
        Data &routedOutput = direct ? output : routed;
        MergeMOE(expertInput, index, score, moeWeights, moeBiass, w1, w2, w3,
                 temporaryInput, temporaryOutput, 1.0f, routedOutput, deviceLayer);
#ifdef USE_CUDA
        FastllmCudaGraphMarkParallelJoin(deviceLayer);
#endif
        routedOutput.Reshape(input.dims);
        sharedOutput.Reshape(input.dims);
        if (direct) {
            ApplyDeviceMap(deviceMap, deviceLayer + 1, block_cnt);
        } else {
            Data copy;
            copy.CopyFrom(routedOutput);
            ApplyDeviceMap(deviceMap, deviceLayer + 1, block_cnt);
            output.CopyFrom(copy);
        }
        if (sharedOutput.dataType != output.dataType) ToDataType(sharedOutput, output.dataType);
        AddTo(output, sharedOutput);
        if (reduceOutput) ThreadTpAllReduce(output);
    }
    std::vector<int> Qwen4ExpModel::ForwardGpu(
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
        ForwardState frame(*this, ExecutionMode::Gpu, batch, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig,
            lastTokens, retLogits, targetMultiHidden, allVerificationTokens, verificationAccepted,
            verificationCapture, recordPrefixSnapshot, allowDecodeCudaGraph, decodeEquivalentGdn,
            hostInputTokens, materializeCausalMaskOnGraphFallback, precomputedEmbedding, prefill);
        frame.Prepare();
        for (int layer = frame.FirstLayer(); layer < frame.EndLayer(); ++layer) {
            const int action = frame.BeginLayer(layer);
            if (action == ForwardState::Complete) break;
            if (action == ForwardState::Skip) continue;
            frame.RunExperts(layer);
            frame.EndLayer(layer);
        }
        return frame.Finish();
    }
}
