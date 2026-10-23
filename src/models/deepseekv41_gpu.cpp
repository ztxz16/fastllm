// GPU experts only. No NUMA dispatch, expert-cache admission or dynamic split.
// Small router outputs may still need host staging for existing CUDA operators.
#include "deepseekv41_execution.h"
#include "baseblock.h"
#include "executor.h"
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
#ifdef USE_CUDA
        // Both branches only read ffnInput. Shared GEMV scratch/output and the
        // resident routed buffers are distinct persistent allocations. Mark only
        // GPU decode/verification captures; eager and unsupported paths stay serial.
        auto markParallel = [&](int stage) {
            if (!ctx.parallelShared) return;
            AssertInFastLLM(MultiCudaRunDeviceCallbacks(
                tpDevices, [&](int, int) {
                    if (stage == 0) FastllmCudaGraphMarkParallelFork(layer);
                    if (stage == 1) {
                        FastllmCudaGraphMarkParallelFirstDone(layer);
                        FastllmCudaGraphMarkParallelSecondBegin(layer);
                    }
                    if (stage == 2) FastllmCudaGraphMarkParallelJoin(layer);
                }), "DeepSeekV41: MoE parallel marker dispatch failed.");
        };
        markParallel(0);
#endif
        runSharedSegment();
        ctx.releaseSharedScratch();
#ifdef USE_CUDA
        markParallel(1);
#endif
        const bool fp32ExpertSum =
            quantizedLinearNames.count(pre + ".ffn.experts.0.w1.weight") != 0;
        bool combineFfnPostDispatch = false;
#ifdef USE_CUDA
        combineFfnPostDispatch = (!fp32ExpertSum ||
            (ffnInput.dataType == DataType::BFLOAT16 && sharedExpertOut.dataType == DataType::BFLOAT16)) &&
            tp && single && hasSharedExpertOut && !dumpDebug &&
            (seqlen == 1 || dsparkVerify);
#endif
        // Keep the FP32 routed sum for the fused HC post path. The CUDA
        // kernel still rounds the shared+routed result to BF16 before HC.
        Data &routedOutput = fp32ExpertSum && !combineFfnPostDispatch ?
            ws->reductionBuffers["moe_routed_sum"] : ffnOut;
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
        // The V4.1 native EP path and its generic fallback only read typed
        // routing tensors. They can consume the graph-owned replicas directly.
        if ((tp && !routedExpertParallel) ||
            (graphActive && !(fp32ExpertSum && routedExpertParallel))) {
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
        if (ws->sharedExpertSharded) {
            AssertInFastLLM(combineFfnPostDispatch && fp32ExpertSum && routedExpertParallel,
                "Sharded shared output requires the joint FP32 routed reduction.\n");
            // Native BF16 graph inputs need neither dtype conversion nor a CPU
            // fallback. MultiCUDA keeps the replicated output metadata current.
            DataDict moeDatas = {
                {"input", input}, {"index", index}, {"score", score},
                {"weights", (Data *)moeWeights.data()}, {"biass", (Data *)biass[layer].data()},
                {"w1", &w1}, {"w2", &w2}, {"w3", &w3},
                {"curInput", &tempInput}, {"curOutput", &tempOutput}, {"output", &routedOutput}
            };
            if (ws->ffnInputQuantizedReady) moeDatas["prequantizedInput"] = &ws->ffnInputQuantized;
            ((Executor *)GetExecutor())->Run("MergeMOE", moeDatas,
                {{"sharedScale", 1.0f}, {"swigluLimit", swiglu_limit}}, {
                {"weights___batch", (int)moeWeights.size()}, {"biass___batch", (int)biass[layer].size()},
                {"layer", layer}, {"gateType", (int)MoeGateSwiglu}, {"expertParallel", 1},
                {"deepSeekV4Mode", 1}, {"activationQuantBlock", 32},
                {"quantizeSharedExpert", 0}, {"deferRankReduction", 1}
            });
        } else {
            MergeMOEBlock(input, index, score, &moeWeights, &biass[layer],
                          &w1, &w2, &w3, &tempInput, &tempOutput, 1.0f, &routedOutput, layer,
                          ffnInput.dataType, ffnInput.dataType, &moeInputTemp, &moeOutputTemp,
                          MoeGateSwiglu, routedExpertParallel, swiglu_limit, true, nullptr,
                          quantizedLinearNames.count(pre + ".ffn.experts.0.w1.weight") ? 32 : 128,
                          quantizedLinearNames.count(pre + ".ffn.shared_experts.w1.weight") != 0);
        }
        ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
#ifdef USE_CUDA
        markParallel(2);
        if (ws->sharedExpertSharded) {
            const int count = routedOutput.Count(0);
            AssertInFastLLM(routedOutput.dataType == DataType::FLOAT32 &&
                sharedExpertOut.dataType == DataType::BFLOAT16 &&
                count == sharedExpertOut.Count(0) && count % tpDevices.size() == 0,
                "Joint MoE collective received incompatible tensors.\n");
            AssertInFastLLM(MultiCudaRunDeviceCallbacks(tpDevices, [&](int rank, int device) {
                void *routed = routedOutput.multiDeviceDatas.at(device)->cudaData;
                auto *shared = (uint16_t *)sharedExpertOut.multiDeviceDatas.at(device)->cudaData;
                bool ok = FastllmCudaCustomAllReduceAndGather(
                    routed, routed, shared, count, device);
                if (!ok && !FastllmCudaGetThreadError()) {
                    // Registration/topology rejection retains ordinary collectives.
                    FastllmNcclAllReduce(routed, routed, count, DataType::FLOAT32, device);
                    ok = FastllmNcclAllGather(shared + rank * (count / tpDevices.size()),
                        shared, count / tpDevices.size(), DataType::BFLOAT16, device);
                }
                AssertInFastLLM(ok, "DeepSeekV41 joint MoE collective failed.\n");
            }), "DeepSeekV41 joint MoE dispatch failed.\n");
        }
        // Preserve generic-operator fallback semantics for unsupported dtypes.
        if (tp && !combineFfnPostDispatch && routedOutput.dataDevice == DataDevice::CPU && routedOutput.cpuData) {
            V41ResetMultiDevice(routedOutput);
            PrepareMultiCudaReplicatedData(routedOutput, tpDevices, true);
        }
#endif
        if (hasSharedExpertOut && !combineFfnPostDispatch) {
            if (!(tp && routedOutput.multiDeviceData && sharedExpertOut.multiDeviceData)) {
                routedOutput.ToDevice(sharedExpertOut.dataDevice);
            }
            if (routedOutput.dataType == DataType::FLOAT32 &&
                sharedExpertOut.dataType != DataType::FLOAT32) {
                Data &sharedFloat = ws->reductionBuffers["moe_shared_sum"];
                ToDataType(sharedExpertOut, sharedFloat, DataType::FLOAT32);
                AddTo(routedOutput, sharedFloat);
            } else {
                AddTo(routedOutput, sharedExpertOut);
            }
        }
        if (!combineFfnPostDispatch && routedOutput.dataType != ffnInput.dataType) {
            ToDataType(routedOutput, ffnOut, ffnInput.dataType);
        } else if (&routedOutput != &ffnOut) {
            V41Assign(ffnOut, routedOutput, tpDevices);
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
        if (frame.FullGraph()) return frame.Finish();
        for (int layer=0; layer<frame.Layers(); ++layer) {
            frame.BeginLayer(layer);
            frame.GpuExperts(layer);
            frame.EndLayer(layer);
        }
        return frame.Finish();
    }
}
