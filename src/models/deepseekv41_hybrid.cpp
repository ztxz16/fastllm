// CPU/NUMA experts: overlap, GPU cache, dynamic routing and host staging.
// Pure GPU deployments never enter this backend.
#include "deepseekv41_execution.h"
#include "deepseekv41_hybrid.h"
#include "baseblock.h"
#include "utils.h"
#include "utils/hybrid_parallel.h"
#include "devices/cpu/deepseekv41-reference-math.h"
#include "devices/disk/diskdevice.h"
#include <chrono>
#include <exception>
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-cuda-hybrid-prefill.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
    static bool UsesHostMoe(const std::string &device) {
        return V41DeviceSpecUsesType(device, "numa") || device == "cpu";
    }

    DeepSeekV41DecodeWorkspace::DeepSeekV41DecodeWorkspace() = default;
    DeepSeekV41DecodeWorkspace::~DeepSeekV41DecodeWorkspace() = default;

    void DeepSeekV41Model::OnAutoWarmupFinished() {
        if (executionMode != ExecutionMode::Hybrid || std::uncaught_exceptions() ||
            !GetMoeCpuCacheBytes()) return;
        // Wait until temporary warmup tensors have been released before
        // filling RAM. Preloading during weight conversion raises its peak.
        const auto started = std::chrono::steady_clock::now();
        const uint64_t bytes = PreloadDiskMoeCache(weights);
        if (bytes) {
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            printf("[Fastllm] Disk MoE RAM preload: %.2f GiB in %.3f s (budget %.2f GiB).\n",
                bytes / double(1ULL << 30), seconds, GetMoeCpuCacheBytes() / double(1ULL << 30));
            fflush(stdout);
        }
    }

    void DeepSeekV41Model::BeginHybridMoeStep(
            HybridMoeDecodeStep &step, bool tp, bool single, int seqlen) {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        const bool streamGGUF = !tp && weights[0][2]->dataType == DATA_GGUF_FORMAT;
        bool cpuHost = false;
        if (!moeExpertCacheAttempted && !weights[0][2]->isDiskWeight) {
            for (int layer = 0; layer < block_cnt && !cpuHost; ++layer)
                cpuHost = SelectMoeDeviceForLayer(layer) == "cpu";
        }
        if ((FastllmCudaMoeCacheRequested() || streamGGUF || cpuHost) && !moeExpertCacheAttempted &&
            !weights[0][2]->isDiskWeight) {
            moeExpertCacheAttempted = true;
            std::vector<CudaPrefillStage> stages;
            if (!tp && BuildCudaPrefillStages(deviceMap, block_cnt, stages))
                for (const auto &stage : stages) hybridMoeDevices.push_back(stage.device);
            bool supported = !V41ReferenceMathEnabled() && GetCudaSharedExpert();
            std::vector<std::vector<Data *>> routedWeights = weights;
            std::vector<std::vector<Data *>> draftWeights;
            if (v41DsparkEnabled) {
                DsparkBuildMoeWeights();
                draftWeights = v41DsparkMoeWeights;
                for (int stage = 0; stage < int(draftWeights.size()); ++stage) {
                    const int layer = std::max(0, block_cnt - v41DsparkLayers + stage);
                    const std::string pre = "mtp." + std::to_string(stage) + ".ffn";
                    if (!UsesHostMoe(SelectMoeDeviceForLayer(layer)) ||
                        !quantizedLinearNames.count(pre + ".experts.0.w1.weight")) {
                        draftWeights.clear(); break;
                    }
                }
            }
            auto payload = [](const std::vector<std::vector<Data *>> &tables) {
                uint64_t bytes = 0;
                for (const auto &table : tables) for (size_t i = 2; i < table.size(); ++i)
                    if (table[i]) bytes += table[i]->GetBytes();
                return bytes;
            };
            const uint64_t mainBytes = payload(routedWeights), draftBytes = payload(draftWeights);
            const double draftFraction = draftBytes ? double(draftBytes) / (mainBytes + draftBytes) : 0;
            HybridMoeCachePlan cache(HybridMoeMath::DeepSeekV41, swiglu_limit, 1.0 - draftFraction);
            for (int layer = 0; layer < block_cnt; ++layer) {
                const auto device = SelectMoeDeviceForLayer(layer);
                // GPU-resident layers are not members of the host cache.
                if (!UsesHostMoe(device)) continue;
                supported &= quantizedLinearNames.count("layers." + std::to_string(layer) + ".ffn.experts.0.w1.weight");
                routedWeights[layer][0] = routedWeights[layer][1] = nullptr;
                cache.Add(routedWeights[layer], device);
            }
            if (supported && cpuHost) WarmupNumaMoeWeights(true);
            if (!supported || !cache.Prepare(
                    [this] { WarmupNumaMoeWeights(true); }, streamGGUF)) {
                fprintf(stderr, "[Fastllm] V4.1 expert cache requires host FP8 block128, NVFP4 block32 or supported GGUF experts, "
                    "CUDA shared experts and ordinary math; using the configured backend.\n");
            }
            if (supported && !draftWeights.empty()) {
                HybridMoeCachePlan draftCache(HybridMoeMath::DeepSeekV41, swiglu_limit, draftFraction);
                for (size_t stage = 0; stage < draftWeights.size(); ++stage) {
                    auto &table = draftWeights[stage];
                    table[0] = table[1] = nullptr;
                    draftCache.Add(table, SelectMoeDeviceForLayer(std::max(0, block_cnt - v41DsparkLayers + int(stage))));
                }
                if (!draftCache.Prepare([this] { WarmupNumaMoeWeights(true); }))
                    fprintf(stderr, "[Fastllm] DSpark expert cache unavailable; retaining host fallback.\n");
            }
        }
        // A complete decode step spans every layer. Admission runs afterward,
        // outside graph capture, while the attention/shared segments retain
        // their stable graph-owned tensors. Unsupported layouts retain NUMA.
        // Native cache requests (including --auto) enable admission directly.
        const bool nativeFrequency = weights[0][2]->dataType == NVFP4_BLOCK_32_E8M0 ||
                                     weights[0][2]->dataType == FP8_E4M3_BLOCK_128 || streamGGUF;
        if (!tp && single && seqlen == 1 && nativeFrequency &&
            FastllmCudaMoeCacheRequested() &&
            !V41ReferenceMathEnabled() && GetCudaSharedExpert() &&
            std::getenv("FASTLLM_DSV41_DUMP_DIR") == nullptr &&
            (this->deviceMap.empty() || V41DeviceMapUsesCuda(this->deviceMap)) &&
            (streamGGUF || V41DeviceMapCudaDeviceCount(this->deviceMap) <= 1)) {
            // A fixed cache budget may let auto keep initial layers fully
            // resident. Start the shared scope from its first host table;
            // a GPU-only first layer is not registered with the host cache.
            int first = 0;
            if (streamGGUF)
                while (first < block_cnt && !UsesHostMoe(SelectMoeDeviceForLayer(first))) ++first;
            if (first < block_cnt)
                step.Begin(weights[first].data(), (int)weights[first].size(), num_experts_per_tok,
                    streamGGUF ? &hybridMoeDevices : nullptr);
        }
#endif
    }

    bool DeepSeekV41Model::RunHybridMoe(DeepSeekV41MoeContext &ctx, HybridMoeDecodeStep *step) {
        auto *ws = &ctx.workspace;
        auto &moeWeights = ctx.moeWeights;
#ifdef USE_NUMAS
        HostMoePlacementScope placement(moeWeights.size() > 2 ? moeWeights[2] : nullptr);
#endif
        const auto &tpDevices = ctx.tpDevices;
        const int layer = ctx.layer, seqlen = ctx.seqlen, dim = embed_dim;
        const bool tp = ctx.tp, single = ctx.single, dsparkVerify = ctx.dsparkVerify;
        const bool graphActive = ctx.graphActive, dumpDebug = ctx.dumpDebug;
        const bool hasSharedExpertOut = ctx.hasSharedExpertOut;
#if defined(USE_CUDA) && !defined(USE_ROCM)
        FastllmCudaMoeRouteScope routeScope(dsparkVerify ? FastllmMoeRoutePhase::Verify :
            seqlen > 1 ? FastllmMoeRoutePhase::SmallPrefill : FastllmMoeRoutePhase::Decode);
#endif
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
            ((overlapMoeDevice == "cpu" && !moeWeights[2]->hostMoeUnbound) ||
             (!tp && single && seqlen <= 8 && V41DeviceSpecUsesType(overlapMoeDevice, "disk")) ||
             (UsesHostMoe(overlapMoeDevice) &&
              (tp || (single && seqlen == 1 && !FastllmCudaMoeCacheRequested()))));
#endif
        bool tpCacheHandled = false;
        bool singleCacheHandled = false;
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        // Ordinary cached decode and verification borrow the other PP devices
        // between graph segments. Keep DSpark scalar steps on their existing
        // path so scalar cache history does not bias verify's scheduler.
        // Concurrent prefill retains exclusive GPU ownership.
        const bool cooperativeMoe = !tp && single && !ctx.pipelineHostExperts &&
            hybridMoeDevices.size() > 1 && moeWeights[2]->dataType == DATA_GGUF_FORMAT &&
            ((seqlen == 1 && !v41DsparkEnabled && FastllmCudaMoeCacheRequested()) ||
             (dsparkVerify && seqlen > 1 && seqlen <= 8)) && !dumpDebug &&
            !V41ReferenceMathEnabled() && UsesHostMoe(overlapMoeDevice);
        // Zero-budget decode retains its measured streaming scope. Cached
        // decode and verification use bounded layer-local admission.
        HybridMoeDecodeStep streamingStep;
        if (!tp && !graphActive && single && seqlen == 1 && !FastllmCudaMoeCacheRequested() &&
            !dumpDebug && !V41ReferenceMathEnabled() && hasSharedExpertOut &&
            moeWeights[0] == nullptr && moeWeights[1] == nullptr &&
            UsesHostMoe(overlapMoeDevice) &&
            FastllmCudaCanRunMoeHybrid(moeWeights.data(), (int)moeWeights.size())) {
            streamingStep.Begin(moeWeights.data(), (int)moeWeights.size(), num_experts_per_tok);
        }
        // Verification runs between graph segments too. Use the same stable
        // staging as decode, rather than migrating tensors owned by a graph.
        // Its multi-row scheduler is independent of the single-token step.
        const bool singleStageCache = graphActive && !tp && single &&
            (seqlen >= 1 && seqlen < 32) && FastllmCudaMoeCacheRequested() &&
            !dumpDebug && !V41ReferenceMathEnabled() &&
            UsesHostMoe(overlapMoeDevice);
        // Concurrent requests arrive as one [rows, hidden] activation with
        // single == false. The CUDA hybrid backend already has the exact
        // multi-row expert-parallel/cache implementation, but the model
        // entry point used to bypass it and call the legacy MergeMOEBlock.
        // Keep this bounded to the verifier-sized batch so long prefill keeps
        // its established grouped path.
        const bool batchCacheCandidate = !tp && !graphActive && !ctx.pipelineHostExperts && !single &&
            seqlen > 1 && seqlen < 32 && !dumpDebug &&
            !V41ReferenceMathEnabled() && UsesHostMoe(overlapMoeDevice);
        if (((step && *step) || singleStageCache || streamingStep || cooperativeMoe ||
             batchCacheCandidate) && hasSharedExpertOut &&
            moeWeights[0] == nullptr && moeWeights[1] == nullptr &&
            ffnInput.dataDevice == DataDevice::CUDA && ffnInput.cudaData &&
            FastllmCudaCanRunMoeHybrid(moeWeights.data(), (int)moeWeights.size())) {
            const int device = FastllmCudaGetDevice();
            if (!ws->hybrid) ws->hybrid.reset(new DeepSeekV41HybridWorkspace());
            auto &cache = ws->hybrid->moeCache[device];
            // The router may return CPU tensors. Stage only its small
            // result, never migrate the input/output of a captured segment.
            auto stageRoute = [&](Data &buffer, Data &source) -> Data & {
                // Eager PP routing already owns a valid CUDA tensor. Keep it
                // on-device; the hybrid backend batches its pinned D2H copies.
                if ((streamingStep || cooperativeMoe || batchCacheCandidate) &&
                    source.dataDevice == DataDevice::CUDA &&
                    source.cudaData && !source.multiDeviceData &&
                    GetPointerDeviceId(source.cudaData) == device) return source;
                V41ReplicaToCpu(buffer, source, tpDevices);
                buffer.ToDevice(DataDevice::CUDA, {device}, true);
                return buffer;
            };
            Data &routeIndex = stageRoute(cache.index, expertIndex);
            Data &routeScore = stageRoute(cache.score, expertScore);
            Data input;
            input.FakeFrom(ffnInput, 0);
            input.Resize({seqlen, dim});
            input.dataDeviceIds = {device};
            if ((streamingStep || cooperativeMoe || batchCacheCandidate) &&
                !ctx.pipelineHostExperts && hybridMoeDevices.size() > 1)
                singleCacheHandled = fastllm::RunHybridMoeOnDevices(
                    input, routeIndex, routeScore, ffnOut, moeWeights.data(),
                    (int)moeWeights.size(), layer, hybridMoeDevices, runSharedSegment);
            else
                singleCacheHandled = fastllm::RunHybridMoe(
                    input, routeIndex, routeScore, ffnOut,
                    moeWeights.data(), (int)moeWeights.size(), layer, runSharedSegment);
        }
        const bool tpBatchCache = tp && !single && !graphActive && seqlen > 1 && seqlen <= 8;
        if (tp && FastllmCudaMoeCacheRequested() &&
            ((single && (seqlen == 1 || (dsparkVerify && seqlen <= 8))) || tpBatchCache) &&
            !dumpDebug && hasSharedExpertOut &&
            !V41ReferenceMathEnabled() &&
            moeWeights[0] == nullptr && moeWeights[1] == nullptr &&
            UsesHostMoe(SelectMoeDeviceForLayer(layer)) &&
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
                bool diskStaged = false;
#if defined(USE_CUDA) && !defined(USE_ROCM)
                if (!tp && !graphActive && !dumpDebug && !V41ReferenceMathEnabled() && seqlen == 1 &&
                        V41DeviceSpecUsesType(overlapMoeDevice, "disk") &&
                        moeWeights[2]->isDiskWeight && moeWeights[2]->dataType == DATA_GGUF_FORMAT)
                    diskStaged = StageDiskMoeV41Inputs(ffnInput, expertIndex, expertScore,
                        cpuMoeInput, cpuMoeIndex, cpuMoeScore);
#endif
                if (!diskStaged) {
                    V41ReplicaToCpu(cpuMoeInput, ffnInput, tpDevices);
                    V41ReplicaToCpu(cpuMoeIndex, expertIndex, tpDevices);
                    V41ReplicaToCpu(cpuMoeScore, expertScore, tpDevices);
                }
            }
            runSharedSegment();
        }
        ctx.releaseSharedScratch();

        // Add in the routed result's precision, then round once at the MoE
        // boundary. Resident GGUF returns FP32 while shared experts may be 16-bit.
        auto addSharedExpert = [&] {
            Data *shared = &sharedExpertOut;
            if (shared->dataType != ffnOut.dataType) {
                shared = &ws->reductionBuffers["moe_shared_sum"];
                ToDataType(sharedExpertOut, *shared, ffnOut.dataType);
            }
            AddTo(ffnOut, *shared);
        };
        bool combineFfnPostDispatch = false;
#ifdef USE_CUDA
        combineFfnPostDispatch = tp && hasSharedExpertOut && !dumpDebug &&
            ((single && (seqlen == 1 || dsparkVerify)) || tpCacheHandled);
#endif
        // Decide only after the routed producer has joined and its usual
        // upload has completed. CPU/GPU expert scheduling remains unchanged.
        auto deferLocalPost = [&] {
#ifdef USE_CUDA
            return !tp && !dumpDebug && !V41ReferenceMathEnabled() && hasSharedExpertOut &&
                ffnInput.dataType == DataType::BFLOAT16 &&
                sharedExpertOut.dataType == DataType::BFLOAT16 &&
                (ffnOut.dataType == DataType::BFLOAT16 || ffnOut.dataType == DataType::FLOAT32) &&
                !ffnOut.multiDeviceData && !sharedExpertOut.multiDeviceData &&
                ffnOut.dataDevice == DataDevice::CUDA && sharedExpertOut.dataDevice == DataDevice::CUDA;
#else
            return false;
#endif
        };
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
#if defined(USE_CUDA) && !defined(USE_ROCM)
            if (!tp && !graphActive && !dumpDebug && !V41ReferenceMathEnabled() &&
                hasSharedExpertOut && V41DeviceSpecUsesType(moeDeviceSpec, "disk") &&
                quantizedLinearNames.count(pre + ".ffn.experts.0.w1.weight")) {
                DiskMergeMOE disk;
                cacheHandled = disk.RunGpuPrefill(*moeInputPtr, *moeIndexPtr, *moeScorePtr,
                    ffnOut, moeWeights, swiglu_limit);
                if (!cacheHandled && seqlen == 1)
                    cacheHandled = disk.RunGpuDecode(*moeInputPtr, *moeIndexPtr, *moeScorePtr,
                        ffnOut, moeWeights, swiglu_limit);
            }
#endif
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
            if (!tp && !graphActive && !V41ReferenceMathEnabled() && hasSharedExpertOut &&
                moeWeights[0] == nullptr && moeWeights[1] == nullptr &&
                UsesHostMoe(moeDeviceSpec) &&
                moeInputPtr->dims.size() == 2 &&
                moeInputPtr->dims[0] >= 1 && moeInputPtr->dims[0] < 32 && single &&
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
                combineFfnPostDispatch = deferLocalPost();
                if (!combineFfnPostDispatch) addSharedExpert();
            }
            if (!combineFfnPostDispatch && ffnOut.dataType != ffnInput.dataType) {
                ToDataType(ffnOut, ffnInput.dataType);
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
            combineFfnPostDispatch = deferLocalPost();
            if (!combineFfnPostDispatch) {
                addSharedExpert();
                if (ffnOut.dataType != ffnInput.dataType) ToDataType(ffnOut, ffnInput.dataType);
            }
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
#ifdef USE_CUDA
        FastllmCudaHybridPrefillScope prefillMath(inputSegments.size() == 1 &&
            inputSegments[0].seqlen >= 1024 && !V41ReferenceMathEnabled());
#endif
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
            if (!UsesHostMoe(moe)) return false;
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
