// GPU experts only. No NUMA dispatch, expert-cache admission or dynamic split.
// Small router outputs may still need host staging for existing CUDA operators.
#include "deepseekv41_execution.h"
#include "baseblock.h"
#include "utils.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
    bool DeepSeekV41Model::RunGpuMoe(DeepSeekV41MoeContext &ctx) {
        auto *ws = &ctx.workspace;
        auto &moeWeights = ctx.moeWeights;
        const auto &tpDevices = ctx.tpDevices;
        const int layer = ctx.layer, seqlen = ctx.seqlen;
        const bool tp = ctx.tp, single = ctx.single, dsparkVerify = ctx.dsparkVerify;
        const bool graphActive = ctx.graphActive, dumpDebug = ctx.dumpDebug;
        const bool hasSharedExpertOut = ctx.hasSharedExpertOut;
        const auto &runSharedSegment = ctx.runSharedSegment;
        const std::string pre = "layers." + std::to_string(layer);
        Data &ffnInput = ws->ffnInput, &ffnOut = ws->ffnOut;
        Data &expertIndex = ws->expertIndex, &expertScore = ws->expertScore;
        Data &sharedExpertOut = ws->sharedExpertOut;
        Data &cpuMoeInput = ws->cpuMoeInput, &cpuMoeIndex = ws->cpuMoeIndex, &cpuMoeScore = ws->cpuMoeScore;
        Data &w1 = ws->w1, &w2 = ws->w2, &w3 = ws->w3;
        Data &tempInput = ws->tempInput, &tempOutput = ws->tempOutput;
        Data &moeInputTemp = ws->moeInputTemp, &moeOutputTemp = ws->moeOutputTemp;
        runSharedSegment();
        ctx.releaseSharedScratch();
        bool combineFfnPostDispatch = false;
#ifdef USE_CUDA
        combineFfnPostDispatch = tp && single && hasSharedExpertOut && !dumpDebug &&
            (seqlen == 1 || dsparkVerify);
#endif
        ApplyMoeDeviceMapForLayer(layer);
        const bool routedExpertParallel = V41DeviceSpecUsesType(SelectMoeDeviceForLayer(layer), "multicuda");
        Data *input = &ffnInput, *index = &expertIndex, *score = &expertScore;
        // TP attention can feed a single-GPU MoE placement. Its replicated root
        // has no payload. Use the same transfer primitive as the shared path.
        if (tp && !routedExpertParallel) {
            V41ReplicaToCpu(cpuMoeInput, ffnInput, tpDevices);
            input = &cpuMoeInput;
        }
        // CUDA MoE bins routes on the host. Never let it migrate graph-owned
        // tensors; TP expert parallelism instead needs an independent replica.
        if ((tp && !routedExpertParallel) || graphActive) {
            if (routedExpertParallel) {
                V41Assign(cpuMoeIndex, expertIndex, tpDevices);
                V41Assign(cpuMoeScore, expertScore, tpDevices);
            } else {
                V41ReplicaToCpu(cpuMoeIndex, expertIndex, tpDevices);
                V41ReplicaToCpu(cpuMoeScore, expertScore, tpDevices);
            }
            index = &cpuMoeIndex;
            score = &cpuMoeScore;
        }
        MergeMOEBlock(input, index, score, &moeWeights, &biass[layer],
                      &w1, &w2, &w3, &tempInput, &tempOutput, 1.0f, &ffnOut, layer,
                      ffnInput.dataType, ffnInput.dataType, &moeInputTemp, &moeOutputTemp,
                      MoeGateSwiglu, routedExpertParallel, swiglu_limit, true, nullptr,
                      quantizedLinearNames.count(pre + ".ffn.experts.0.w1.weight") ? 32 : 128,
                      quantizedLinearNames.count(pre + ".ffn.shared_experts.w1.weight") != 0);
        ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
#ifdef USE_CUDA
        // Preserve generic-operator fallback semantics for unsupported dtypes.
        if (tp && !combineFfnPostDispatch && ffnOut.dataDevice == DataDevice::CPU && ffnOut.cpuData) {
            V41ResetMultiDevice(ffnOut);
            PrepareMultiCudaReplicatedData(ffnOut, tpDevices, true);
        }
#endif
        if (hasSharedExpertOut && !combineFfnPostDispatch) {
            if (!(tp && ffnOut.multiDeviceData && sharedExpertOut.multiDeviceData)) {
                ffnOut.ToDevice(sharedExpertOut.dataDevice);
            }
            AddTo(ffnOut, sharedExpertOut);
        }
#ifdef USE_CUDA
        if (graphActive && !tp && ffnOut.dataDevice == DataDevice::CPU) ffnOut.ToDevice(DataDevice::CUDA);
#endif
        return combineFfnPostDispatch;
    }
} // namespace fastllm

namespace fastllm {
    std::vector<int> DeepSeekV41Model::ForwardGpu(const std::vector<DeepSeekV41Segment> &inputSegments,
                                                       const Data &inputIds,
                                                       const Data *inputEmbeds,
                                                       const std::vector<int> *imageMask,
                                                       const std::vector<GenerationConfig> &generationConfigsIn,
                                                       const LastTokensManager &lastTokens,
                                                       std::vector<std::vector<float>*> *retLogits,
                                                       std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues) {
        ForwardState frame(*this, inputSegments, inputIds, inputEmbeds, imageMask, generationConfigsIn,
            lastTokens, retLogits, samplingPastKeyValues);
        frame.PrepareInputs();
        for (int layer=0; layer<frame.Layers(); ++layer) {
            frame.BeginLayer(layer);
            frame.GpuExperts(layer);
            frame.EndLayer(layer);
        }
        return frame.Finish();
    }
}
