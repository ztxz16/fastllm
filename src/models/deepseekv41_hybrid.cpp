// CPU/NUMA experts: overlap, GPU cache, dynamic routing and host staging.
// Pure GPU deployments never enter this backend.
#include "deepseekv41_execution.h"
#include "deepseekv41_hybrid.h"
#include "baseblock.h"
#include "utils.h"
#include "utils/hybrid_parallel.h"
#include "devices/cpu/deepseekv41-reference-math.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
    DeepSeekV41DecodeWorkspace::DeepSeekV41DecodeWorkspace() = default;
    DeepSeekV41DecodeWorkspace::~DeepSeekV41DecodeWorkspace() = default;

    void DeepSeekV41Model::BeginHybridMoeStep(
            HybridMoeDecodeStep &step, bool tp, bool single, int seqlen) {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        if (FastllmCudaMoeCacheRequested() && !moeExpertCacheAttempted &&
            !weights[0][2]->isDiskWeight) {
            moeExpertCacheAttempted = true;
            bool supported = !V41ReferenceMathEnabled() && GetCudaSharedExpert();
            std::vector<std::vector<Data *>> routedWeights = weights;
            HybridMoeCachePlan cache(HybridMoeMath::DeepSeekV41, swiglu_limit);
            for (int layer = 0; layer < block_cnt; ++layer) {
                supported &= V41DeviceSpecUsesType(SelectMoeDeviceForLayer(layer), "numa") &&
                    quantizedLinearNames.count("layers." + std::to_string(layer) + ".ffn.experts.0.w1.weight");
                routedWeights[layer][0] = routedWeights[layer][1] = nullptr;
                cache.Add(routedWeights[layer], SelectMoeDeviceForLayer(layer));
            }
            if (!supported || !cache.Prepare(
                    [this] { WarmupNumaMoeWeights(); })) {
                fprintf(stderr, "[Fastllm] V4.1 expert cache requires NUMA FP8 block128, NVFP4 block32 or Q2_K/Q4_K R4 experts, "
                    "CUDA shared experts and ordinary math; using the configured backend.\n");
            }
        }
        // A complete decode step spans every layer. Admission runs afterward,
        // outside graph capture, while the attention/shared segments retain
        // their stable graph-owned tensors. Unsupported layouts retain NUMA.
        // Native cache requests (including --auto) enable this path by default. Keep
        // the explicit legacy switch only as an opt-out for comparisons.
        const char *frequency = std::getenv("FASTLLM_DSV41_MOE_FREQUENCY");
        const bool nativeFrequency = weights[0][2]->dataType == NVFP4_BLOCK_32_E8M0 ||
                                     weights[0][2]->dataType == FP8_E4M3_BLOCK_128;
        const bool frequencyEnabled = frequency ?
            V41EnvFlag("FASTLLM_DSV41_MOE_FREQUENCY") : nativeFrequency;
        if (!tp && single && seqlen == 1 && frequencyEnabled &&
            FastllmCudaMoeCacheRequested() &&
            !V41ReferenceMathEnabled() && GetCudaSharedExpert() &&
            std::getenv("FASTLLM_DSV41_DUMP_DIR") == nullptr &&
            (this->deviceMap.empty() || V41DeviceMapUsesCuda(this->deviceMap)) &&
            V41DeviceMapCudaDeviceCount(this->deviceMap) <= 1) {
            step.Begin(
                weights[0].data(), (int)weights[0].size(), num_experts_per_tok);
        }
#endif
    }

    bool DeepSeekV41Model::RunHybridMoe(DeepSeekV41MoeContext &ctx, HybridMoeDecodeStep *step) {
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
        std::unique_ptr<HybridMoePipelineGuard> pipelineGuard;
        if (ctx.pipelineHostExperts) {
            pipelineGuard.reset(new HybridMoePipelineGuard(ffnInput,
                SelectMoeDeviceForLayer(layer), ctx.pipelineHostExperts, ctx.pipelineFirstLayer));
        }
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        const std::vector<int> stageDevice{FastllmCudaGetDevice()};
        std::optional<NumasMoeCudaAssistScope> stageAssist;
        if (ctx.pipelineHostExperts)
            stageAssist.emplace(&stageDevice);
#endif
        // Copy CPU expert inputs before launching the shared expert. A
        // device-to-host copy after that launch would wait for all shared
        // GPU work, serializing the two otherwise independent branches.
        bool overlapShared = false;
#ifdef USE_CUDA
        const std::string overlapMoeDevice = this->SelectMoeDeviceForLayer(layer);
        overlapShared = hasSharedExpertOut &&
            !V41EnvFlag("FASTLLM_DSV41_DISABLE_SHARED_OVERLAP") &&
            (V41DeviceSpecUsesType(overlapMoeDevice, "cpu") ||
             (!tp && single && seqlen <= 8 && V41DeviceSpecUsesType(overlapMoeDevice, "disk")) ||
             (V41DeviceSpecUsesType(overlapMoeDevice, "numa") &&
              (tp || (single && seqlen == 1 && !FastllmCudaMoeCacheRequested()))));
#endif
        bool tpCacheHandled = false;
        bool singleCacheHandled = false;
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        // Verification runs between graph segments too. Use the same stable
        // staging as decode, rather than migrating tensors owned by a graph.
        // Its multi-row scheduler is independent of the single-token step.
        const bool singleVerifyCache = graphActive && !tp && single && dsparkVerify && ctx.verifyAllTokens &&
            seqlen > 1 && seqlen <= 8 && FastllmCudaMoeCacheRequested() &&
            !dumpDebug && !V41ReferenceMathEnabled() &&
            V41DeviceSpecUsesType(SelectMoeDeviceForLayer(layer), "numa");
        if (((step && *step) || singleVerifyCache) && hasSharedExpertOut &&
            moeWeights[0] == nullptr && moeWeights[1] == nullptr &&
            ffnInput.dataDevice == DataDevice::CUDA && ffnInput.cudaData &&
            FastllmCudaCanRunMoeHybrid(moeWeights.data(), (int)moeWeights.size())) {
            const int device = FastllmCudaGetDevice();
            if (!ws->hybrid) ws->hybrid.reset(new DeepSeekV41HybridWorkspace());
            auto &cache = ws->hybrid->moeCache[device];
            // The router may return CPU tensors. Stage only its small
            // result, never migrate the input/output of a captured segment.
            V41ReplicaToCpu(cache.index, expertIndex, tpDevices);
            V41ReplicaToCpu(cache.score, expertScore, tpDevices);
            cache.index.ToDevice(DataDevice::CUDA, {device}, true);
            cache.score.ToDevice(DataDevice::CUDA, {device}, true);
            Data input;
            input.FakeFrom(ffnInput, 0);
            input.Resize({seqlen, dim});
            input.dataDeviceIds = {device};
            singleCacheHandled = fastllm::RunHybridMoe(
                input, cache.index, cache.score, ffnOut,
                moeWeights.data(), (int)moeWeights.size(), layer, runSharedSegment);
        }
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
                if (!ws->hybrid) ws->hybrid.reset(new DeepSeekV41HybridWorkspace());
                auto &cache = ws->hybrid->moeCache[cacheDevice];
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
        if (!tpCacheHandled && !singleCacheHandled) {
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
        if (!tpCacheHandled && !singleCacheHandled) {
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
        if (singleCacheHandled) {
            ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
            AddTo(ffnOut, sharedExpertOut);
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

namespace fastllm {
bool DeepSeekV41Model::TryForwardHybridChunkedPrefill(
        const Data &inputIds, const Data &attentionMask, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &pastKeyValues,
        const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
        std::vector<float> *logits, int &outputToken) {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
    // Experimental opt-in. The ordinary GPU and TP forwards never enter here.
    const int chunkSize = GetChunkedPrefillSize();
    if (executionMode != ExecutionMode::Hybrid || !V41EnvFlag("FASTLLM_DSV41_HYBRID_PP") ||
        !generationConfig.IsSimpleGreedy() || saveHistoryChat || GetKVCacheInCPU() ||
        GetHistoryCacheInCPU() || GetLowMemMode() || GetFastllmEnv().printProfile ||
        FastllmCudaGraphIsCapturingFast() || FastllmCudaMoeCacheRequested() ||
        !GetCudaSharedExpert() || v41DsparkEnabled || V41ReferenceMathEnabled() ||
        V41EnvFlag("FASTLLM_DSV41_ENGRAM_PREFETCH") ||
        std::getenv("FASTLLM_DSV41_DUMP_DIR") ||
        decoderSwaTailLayer <= 0 || decoderSwaTailLayer >= block_cnt ||
        chunkSize <= window_size || pastKeyValues.empty() ||
        inputIds.dataDevice != CPU || inputIds.dataType != FLOAT32 || !inputIds.cpuData ||
        inputIds.dims.size() != 2 || inputIds.dims[0] != 1 || inputIds.dims[1] <= chunkSize ||
        !attentionMask.dims.empty() || positionIds.dataDevice != CPU ||
        positionIds.dataType != FLOAT32 || !positionIds.cpuData || positionIds.dims != inputIds.dims)
        return false;
    const int sequence = inputIds.dims[1];
    if ((sequence - 1) % chunkSize + 1 <= window_size) return false;
    const float *tokens = reinterpret_cast<const float*>(inputIds.cpuData);
    const float *positions = reinterpret_cast<const float*>(positionIds.cpuData);
    for (int i = 0; i < sequence; ++i)
        if (positions[i] != i || int(tokens[i]) == image_token_id) return false;
    for (int layer : engram_layer_ids) if (layer >= decoderSwaTailLayer) return false;
    for (const auto &kv : pastKeyValues)
        if (!kv.first.dims.empty() || !kv.second.dims.empty() ||
            !kv.first.pageIndex.empty() || !kv.second.pageIndex.empty()) return false;
    std::vector<CudaPrefillStage> stages;
    if (!BuildCudaPrefillStages(deviceMap, block_cnt, stages)) return false;
    // A stage must own every shared KV/index/candidate source it consumes.
    // In particular, arbitrary 10/11-layer splits cannot safely overlap.
    for (const auto &stage : stages) {
        if (stage.firstLayer >= decoderSwaTailLayer) return false;
        for (int layer = stage.firstLayer; layer < stage.endLayer; ++layer) {
            const auto moe = SelectMoeDeviceForLayer(layer);
            if (moe != "numa" && moe.rfind("numa:", 0) != 0) return false;
            if (compress_ratios[layer] > 0 &&
                (kvSourceOf[layer] < stage.firstLayer || indexSourceOf[layer] < stage.firstLayer ||
                 (candidate_source_layer_id >= 0 && layer > candidate_source_layer_id &&
                  candidate_source_layer_id < stage.firstLayer))) return false;
        }
    }
    auto existing = GetStateByFirstKey(&pastKeyValues[0].first);
    if (existing && (existing->pendingMultimodal || existing->totalLen ||
                    existing->restoredLen || existing->dspark)) return false;
    const int callerDevice = FastllmCudaGetDevice();
    struct RestoreDevice { int device; ~RestoreDevice() { FastllmCudaSetDevice(device); } } restore{callerDevice};
    PrepareHybridPrefill();
    auto request = GetOrCreateState(pastKeyValues, true);
    std::vector<std::shared_ptr<DeepSeekV41RequestState>> states(stages.size());
    std::vector<HybridPrefillStage> contexts(stages.size());
    std::vector<int> devices;
    std::mutex hostExperts;
    for (int rank = 0; rank < int(stages.size()); ++rank) {
        states[rank] = std::make_shared<DeepSeekV41RequestState>();
        contexts[rank].firstLayer = stages[rank].firstLayer;
        contexts[rank].endLayer = stages[rank].endLayer;
        contexts[rank].request = request.get();
        contexts[rank].hostExperts = &hostExperts;
        devices.push_back(stages[rank].device);
    }
    FastllmCudaPeerAccessInit(devices);
    HybridPrefillScope hybrid(devices, true);
    CudaChunkedPrefillPipeline pipeline;
    const int chunks = 1 + (sequence - 1) / chunkSize;
    if (verbose) {
        printf("[Prompt] DeepSeek hybrid pipelined prefill: %zu stages, %d chunks, %d tokens/chunk.\n",
            stages.size(), chunks, chunkSize);
        for (const auto &stage : stages)
            printf("[DSV41 PP] device=%d layers=[%d,%d)\n", stage.device, stage.firstLayer, stage.endLayer);
        fflush(stdout);
    }
    std::vector<std::pair<Data*, Data*>> sampling;
    for (auto &kv : pastKeyValues) sampling.emplace_back(&kv.first, &kv.second);
    pipeline.Run(devices, chunks, BFLOAT16, {1, chunkSize, hc_mult * (embed_dim + 2)},
        [&](int rank, int chunk, Data &payload) {
            if (chunk == 0) hybrid.BindWorker(rank);
            const int start = chunk * chunkSize, count = std::min(chunkSize, sequence - start);
            Data ids(FLOAT32, {1, count}, std::vector<float>(tokens + start, tokens + start + count));
            std::vector<DeepSeekV41Segment> segments(1);
            segments[0].state = states[rank];
            segments[0].startPos = start;
            segments[0].seqlen = count;
            segments[0].prefillCacheOnly = chunk + 1 < chunks;
            auto &context = contexts[rank];
            context.payload = &payload;
            context.sampleOutput = rank + 1 == int(stages.size()) && chunk + 1 == chunks;
            std::vector<GenerationConfig> configs{generationConfig};
            std::vector<std::vector<float>*> results{logits};
            ForwardState frame(*this, segments, ids, nullptr, nullptr, configs, lastTokens,
                context.sampleOutput ? &results : nullptr, sampling, &context);
            // Only the first stage embeds. CPU embedding uses the same pool as MoE.
            {
                std::lock_guard<std::mutex> lock(hostExperts);
                frame.PrepareInputs();
            }
            for (int layer = context.firstLayer; layer < frame.Layers(); ++layer) {
                frame.BeginLayer(layer);
                frame.HybridExperts(layer);
                frame.EndLayer(layer);
            }
            const auto output = frame.Finish();
            if (context.sampleOutput) outputToken = output.front();
        });
    // Run has joined all CUDA streams. Layer storage already belongs to request;
    // commit the identical token history and logical position exactly once.
    for (const auto &state : states)
        AssertInFastLLM(state->totalLen == sequence && state->engramHistory == states.front()->engramHistory,
            "DeepSeek hybrid PP stage history mismatch.");
    request->totalLen = sequence;
    request->engramHistory = std::move(states.front()->engramHistory);
    return true;
#else
    return false;
#endif
}
} // namespace fastllm
