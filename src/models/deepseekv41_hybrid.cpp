// CPU/NUMA experts: overlap, GPU cache, dynamic routing and host staging.
// Pure GPU deployments never enter this backend.
#include "deepseekv41_execution.h"
#include "baseblock.h"
#include "utils.h"
#include "devices/cpu/deepseekv41-reference-math.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
    void DeepSeekV41Model::BeginHybridMoeStep(
            HybridMoeDecodeStep &step, bool tp, bool single, int seqlen) {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        if (FastllmCudaMoeCacheRequested() && !moeExpertCacheAttempted &&
            !weights[0][2]->isDiskWeight) {
            moeExpertCacheAttempted = true;
            bool supported = !V41ReferenceMathEnabled() && GetCudaSharedExpert();
            std::vector<std::vector<Data *>> routedWeights = weights;
            std::vector<FastllmCudaMoeCacheLayer> cacheLayers;
            for (int layer = 0; layer < block_cnt; ++layer) {
                supported &= V41DeviceSpecUsesType(SelectMoeDeviceForLayer(layer), "numa") &&
                    quantizedLinearNames.count("layers." + std::to_string(layer) + ".ffn.experts.0.w1.weight");
                routedWeights[layer][0] = routedWeights[layer][1] = nullptr;
                cacheLayers.push_back({routedWeights[layer].data(), (int)routedWeights[layer].size(), true, swiglu_limit});
            }
            if (!supported || !FastllmCudaPrepareMoeCache(cacheLayers.data(), block_cnt,
                    [this] { WarmupNumaMoeWeights(); })) {
                fprintf(stderr, "[Fastllm] V4.1 expert cache requires NUMA NVFP4 block32 or Q2_K/Q4_K R4 experts, "
                    "CUDA shared experts and ordinary math; using the configured backend.\n");
            }
        }
#endif
    }

    bool DeepSeekV41Model::RunHybridMoe(DeepSeekV41MoeContext &ctx) {
        auto *ws = &ctx.workspace;
        auto &moeWeights = ctx.moeWeights;
        const auto &tpDevices = ctx.tpDevices;
        const int layer = ctx.layer, seqlen = ctx.seqlen, dim = embed_dim;
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
        // Copy CPU expert inputs before launching the shared expert. A
        // device-to-host copy after that launch would wait for all shared
        // GPU work, serializing the two otherwise independent branches.
        bool overlapShared = false;
#ifdef USE_CUDA
        const std::string overlapMoeDevice = this->SelectMoeDeviceForLayer(layer);
        overlapShared = hasSharedExpertOut &&
            !V41EnvFlag("FASTLLM_DSV41_DISABLE_SHARED_OVERLAP") &&
            (V41DeviceSpecUsesType(overlapMoeDevice, "cpu") ||
             (V41DeviceSpecUsesType(overlapMoeDevice, "numa") &&
              (tp || (single && seqlen == 1 && !FastllmCudaMoeCacheRequested()))));
#endif
        bool tpCacheHandled = false;
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        if (tp && FastllmCudaMoeCacheRequested() && single &&
            (seqlen == 1 || (dsparkVerify && seqlen <= 8)) && !dumpDebug && hasSharedExpertOut &&
            !V41ReferenceMathEnabled() &&
            moeWeights[0] == nullptr && moeWeights[1] == nullptr &&
            V41DeviceSpecUsesType(SelectMoeDeviceForLayer(layer), "numa") &&
            ffnInput.multiDeviceData && ffnInput.IsTensorParallelReplicated()) {
            // Each layer has one cache owner, using both cards' capacity.
            // CPU experts run once; the existing CPU-result handoff then
            // refreshes every TP replica, including GPUs without peer access.
            const int cacheDevice = tpDevices[(size_t)layer * tpDevices.size() / block_cnt];
            auto local = ffnInput.multiDeviceDatas.find(cacheDevice);
            const int originalDevice = FastllmCudaGetDevice();
            FastllmCudaSetDevice(cacheDevice);
            if (local != ffnInput.multiDeviceDatas.end() && local->second && local->second->cudaData &&
                FastllmCudaCanRunMoeHybrid(moeWeights.data(), (int)moeWeights.size())) {
                auto &cache = ws->moeCache[cacheDevice];
                V41ReplicaToCpu(cache.index, expertIndex, tpDevices);
                V41ReplicaToCpu(cache.score, expertScore, tpDevices);
                cache.index.ToDevice(DataDevice::CUDA, {cacheDevice}, true);
                cache.score.ToDevice(DataDevice::CUDA, {cacheDevice}, true);
                Data input;
                input.FakeFrom(*local->second, 0);
                input.Resize({seqlen, dim});
                input.dataDeviceIds = {cacheDevice};
                tpCacheHandled = fastllm::RunHybridMoe(input, cache.index, cache.score, cache.output,
                    moeWeights.data(), (int)moeWeights.size(), layer, runSharedSegment);
                if (tpCacheHandled) {
                    // Keep the CPU buffer and its GPU replicas reusable.
                    // This staging copy also completes the
                    // cache producer before the joint AddTo/HcPost dispatch.
                    if (ffnOut.dataDevice != DataDevice::CPU || ffnOut.dataType != input.dataType ||
                        ffnOut.expansionBytes < cache.output.GetBytes()) {
                        V41ResetMultiDevice(ffnOut);
                        ffnOut.FreeSpace();
                        ffnOut.dataDevice = DataDevice::CPU;
                        ffnOut.dataDeviceIds.clear();
                        ffnOut.dataType = input.dataType;
                        ffnOut.UpdateUnitSize();
                    }
                    ffnOut.Resize({seqlen, dim});
                    ffnOut.Allocate(false);
                    FastllmCudaCopyFromDeviceToHost(ffnOut.cpuData, cache.output.cudaData, cache.output.GetBytes());
                }
            }
            FastllmCudaSetDevice(originalDevice);
        }
#endif
        if (!tpCacheHandled) {
            if (overlapShared) {
                V41ReplicaToCpu(cpuMoeInput, ffnInput, tpDevices);
                V41ReplicaToCpu(cpuMoeIndex, expertIndex, tpDevices);
                V41ReplicaToCpu(cpuMoeScore, expertScore, tpDevices);
            }
            runSharedSegment();
        }
        ctx.releaseSharedScratch();

        bool combineFfnPostDispatch = false;
#ifdef USE_CUDA
        combineFfnPostDispatch = tp && single && hasSharedExpertOut && !dumpDebug &&
            (seqlen == 1 || dsparkVerify);
#endif
        if (!tpCacheHandled) {
            this->ApplyMoeDeviceMapForLayer(layer);
            // 路由专家在 multicuda 上按专家并行（每卡一部分专家 + all-reduce），
            // 在 cpu / numa 上仍然是单份计算，结果随后广播回两张卡。
            const std::string moeDeviceSpec = this->SelectMoeDeviceForLayer(layer);
            const bool routedExpertParallel = V41DeviceSpecUsesType(moeDeviceSpec, "multicuda");
            const bool routedExpertOnCuda = routedExpertParallel ||
                                            V41DeviceSpecUsesType(moeDeviceSpec, "cuda");
            Data *moeInputPtr = overlapShared ? &cpuMoeInput : &ffnInput;
            Data *moeIndexPtr = overlapShared ? &cpuMoeIndex : &expertIndex;
            Data *moeScorePtr = overlapShared ? &cpuMoeScore : &expertScore;
            // MoE 落在 cpu / numa 时，输入必须从某张卡的副本拷出来：直接交给 CPU 算子
            // 会让 Data::ToDevice 从复制布局已经失效的 root 上读，直接段错误。
            const bool tpStageMoe = tp && !routedExpertParallel;
            bool stageMoeInput = !overlapShared && tpStageMoe;
            bool stageMoeRoute = !overlapShared && tpStageMoe;
#ifdef USE_CUDA
            // 图模式下 ffnInput / expertIndex / expertScore 是 post 段的输出，地址被烤
            // 进图里；而 DoCudaMergeMOE 一定会把 index / score 搬到 CPU 上分桶
            // （cpu / numa 上的 MoE 还会连输入一起搬走），Data::ToDevice 顺手就把显存
            // 释放了，下一次回放于是写到别人的缓冲上。所以先拷到独立的暂存缓冲再交出去。
            stageMoeRoute = stageMoeRoute || (!overlapShared && graphActive);
            stageMoeInput = stageMoeInput || (!overlapShared && graphActive && !routedExpertOnCuda);
#endif
            if (stageMoeInput) {
                V41ReplicaToCpu(cpuMoeInput, ffnInput, tpDevices);
                moeInputPtr = &cpuMoeInput;
            }
            if (stageMoeRoute) {
                if (routedExpertParallel) {
                    // 专家并行要的是每卡一份的复制布局，不能拉到 CPU
                    V41Assign(cpuMoeIndex, expertIndex, tpDevices);
                    V41Assign(cpuMoeScore, expertScore, tpDevices);
                } else {
                    V41ReplicaToCpu(cpuMoeIndex, expertIndex, tpDevices);
                    V41ReplicaToCpu(cpuMoeScore, expertScore, tpDevices);
                }
                moeIndexPtr = &cpuMoeIndex;
                moeScorePtr = &cpuMoeScore;
            }
            bool cacheHandled = false;
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
            if (!tp && !graphActive && !V41ReferenceMathEnabled() && hasSharedExpertOut &&
                moeWeights[0] == nullptr && moeWeights[1] == nullptr &&
                V41DeviceSpecUsesType(moeDeviceSpec, "numa") &&
                moeInputPtr->dims.size() == 2 &&
                (moeInputPtr->dims[0] == 1 || (moeInputPtr->dims[0] <= 8 && single && ctx.verifyAllTokens)) &&
                FastllmCudaCanRunMoeHybrid(moeWeights.data(), (int)moeWeights.size())) {
                // The 384-expert router can return CPU tensors. Upload its
                // small result before entering the device-cache adapter.
                moeIndexPtr->ToDevice(DataDevice::CUDA);
                moeScorePtr->ToDevice(DataDevice::CUDA);
                cacheHandled = fastllm::RunHybridMoe(*moeInputPtr, *moeIndexPtr, *moeScorePtr,
                    ffnOut, moeWeights.data(), (int)moeWeights.size(), layer);
            }
#endif
            if (!cacheHandled) MergeMOEBlock(moeInputPtr, moeIndexPtr, moeScorePtr, &moeWeights, &biass[layer],
                          &w1, &w2, &w3, &tempInput, &tempOutput, 1.0f, &ffnOut, layer,
                          ffnInput.dataType, ffnInput.dataType, &moeInputTemp, &moeOutputTemp,
                          MoeGateSwiglu, routedExpertParallel, swiglu_limit, true, nullptr,
                          quantizedLinearNames.count(pre + ".ffn.experts.0.w1.weight") ? 32 : 128,
                          quantizedLinearNames.count(pre + ".ffn.shared_experts.w1.weight") != 0);
            ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
#ifdef USE_CUDA
            if (tp && !combineFfnPostDispatch && ffnOut.dataDevice == DataDevice::CPU && ffnOut.cpuData != nullptr) {
                // CPU / NUMA 上算出的路由专家结果只有一份，广播到两张卡后才能
                // 与复制布局的共享专家输出、hc 残差相加。
                // A previous decode may have retained replicas; upload the new CPU payload.
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
            if (graphActive && !tp && ffnOut.dataDevice == DataDevice::CPU) {
                // 之后的 hc 残差还要读 curHidden（图的输出，必须留在卡上），
                // 否则 DeepSeekV4HcPost 会把整条残差流搬到 CPU 上。
                ffnOut.ToDevice(DataDevice::CUDA);
            }
#endif
        }
        return combineFfnPostDispatch;
    }
} // namespace fastllm

namespace fastllm {
    std::vector<int> DeepSeekV41Model::ForwardHybrid(const std::vector<DeepSeekV41Segment> &inputSegments,
                                                       const Data &inputIds,
                                                       const Data *inputEmbeds,
                                                       const std::vector<int> *imageMask,
                                                       const std::vector<GenerationConfig> &generationConfigsIn,
                                                       const LastTokensManager &lastTokens,
                                                       std::vector<std::vector<float>*> *retLogits,
                                                       std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues) {
        ForwardState frame(*this, inputSegments, inputIds, inputEmbeds, imageMask, generationConfigsIn,
            lastTokens, retLogits, samplingPastKeyValues);
        frame.BeginHybridStep();
        frame.PrepareInputs();
        for (int layer=0; layer<frame.Layers(); ++layer) {
            frame.BeginLayer(layer);
            frame.HybridExperts(layer);
            frame.EndLayer(layer);
        }
        return frame.Finish();
    }
}
