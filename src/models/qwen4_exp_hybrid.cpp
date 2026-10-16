// Independent hybrid forward; tensor blocks and CUDA kernels remain shared.
#include "qwen4_exp_execution.h"
#include "qwen4_exp_hybrid.h"
#include "qwen4_exp_tp.h"
#include "models/qwen4_paged_cache.h"
#include "executor.h"
#include "utils.h"
#ifdef USE_CUDA
#ifndef USE_ROCM
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-moe-prefetch.h"
#endif
#endif
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif
#include "models/qwen4_kvmem.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
namespace fastllm {
    bool Qwen4ExpModel::IsStreamingHybridBackboneWeight(const std::string &name) const {
#ifdef USE_CUDA
        const auto format = weight.dicts.find("gguf_architecture");
        if (executionMode != ExecutionMode::Hybrid || threadTpRank >= 0 ||
            format == weight.dicts.end() || format->second != "qwen4exp" ||
            deviceMap.empty() || !Qwen4CudaOnlyDeviceMap(deviceMap))
            return false;
        // TP already streams these replicas; keep their original groups.
        if (threadTpState && StreamingThreadTpReplicaLayer(name) >= 0) return false;
        // Keep layout-dependent backbone tensors together until GGUF layout
        // restoration has consumed them. Host experts and the large disk PLE
        // table load afterwards; MTP retains its existing preparation path.
        return name == "lm_head.weight" ||
            (Qwen4StartsWith(name, languagePrefix) &&
             name.find(".mlp.experts.") == std::string::npos && !ngramWeights.count(name));
#else
        return false;
#endif
    }

    void Qwen4ExpModel::FinishHybridBackboneLoad() {
#ifdef USE_CUDA
        if (!loadingHybridBackbone) return;
        loadingHybridBackbone = false;
        // Readers and merge jobs have joined. Restore GDN rows before any
        // projection leaves the host, using the same one-time preparation as
        // OnModelWeightsLoaded. Norms and lookup tables keep their CPU owners.
        RestoreGgufWeights();
        std::vector<std::string> names;
        for (const auto &item : weight.weight)
            if (IsStreamingHybridBackboneWeight(item.first)) names.push_back(item.first);
        if (threadTpState) std::sort(names.begin(), names.end());
        for (const std::string &name : names) {
            Data &data = weight.weight.at(name);
            if (!data.isModelWeight ||
                data.isFake || data.isDiskWeight || data.dataDevice != DataDevice::CPU ||
                !data.cpuData || data.dims.size() != 2 ||
                weight.GetWeightType(name) != WeightType::LINEAR)
                continue;
            if (threadTpState) {
                Qwen4CudaDeviceGuard deviceGuard;
                StreamHybridThreadTpWeight(name, data);
                continue;
            }
            int layer = block_cnt - 1;
            const std::string prefix = languagePrefix + "layers.";
            if (Qwen4StartsWith(name, prefix)) {
                char *end = nullptr;
                const char *start = name.c_str() + prefix.size();
                const long parsed = std::strtol(start, &end, 10);
                if (end == start || *end != '.' || parsed < 0 || parsed >= block_cnt) continue;
                layer = int(parsed);
            }
            const std::string device = SelectDeviceFromMap(deviceMap, layer + 1, block_cnt);
            if (device != "cuda" && !Qwen4StartsWith(device, "cuda:")) continue;
            std::map<int, int> ratios;
            const auto devices = ParseDeviceIds(device, "cuda", ratios);
            if (devices.size() > 1) continue; // Legacy sharding still needs its host source.
            Qwen4CudaDeviceGuard deviceGuard;
            if (FastllmCudaGetWeightSlabBytes() == 0) data.directMemory = true;
            data.ToDevice(DataDevice::CUDA, devices);
        }
#if defined(__GLIBC__)
        // Return the completed group's source pages before loading host
        // experts, without adding allocator scans to the per-tensor path.
        malloc_trim(0);
#endif
#endif
    }

    void Qwen4ExpModel::RunHybridMoE(const MoEExecutionPlan &plan, const Data &input, Data &output,
                                     bool reduceOutput, PrefillStageContext *prefill,
                                     MoEPrepared *prepared,
                                     const std::function<void()> &sharedGraph) {
        const int deviceLayer = plan.deviceLayer;
        const std::string &mlp = plan.mlpPrefix;
        auto &moeWeights = *plan.weights;
        auto &moeBiass = *plan.bias;
        const int batch = input.dims[0];
        const int sequence = input.dims[1];
        Data flattened;
        flattened.FakeFrom(input, 0);
        flattened.Resize(input.dims);
        flattened.Reshape({batch * sequence, input.dims.back()});

        const bool residentTpDraft = threadTpRank >= 0 && threadTpOwner->mtpGpuExperts &&
            Qwen4StartsWith(mlp, "mtp.");
        const bool hostMoe = threadTpRank >= 0 && threadTpOwner->hybrid.hostMoeLayers[deviceLayer] &&
            !residentTpDraft;
        const bool runRoutedExperts = !hostMoe || threadTpRank == 0;
        const std::string moeDevice = SelectSpecialWeightDevice(
            mlp + "experts.0.gateup_proj.weight", deviceLayer);
        // GPU routing and the shared expert do not use the host worker pool.
        // Defer its guard until routed MoE; CPU and decode paths retain it.
        HybridMoePipelineGuard hostExperts(flattened, moeDevice,
            prefill ? prefill->hostExpertMutex : nullptr, prefill ? prefill->firstLayer : -1);
        Data routerLogits, localIndex, localScore, localShared;
        if (prepared) {
            prepared->result.Reset();
        }
        Data &expertIndex = prepared ? prepared->index : localIndex;
        Data &expertScore = prepared ? prepared->score : localScore;
        Data &sharedOutput = prepared ? prepared->result.sharedOutput : localShared;
        if (runRoutedExperts && !prepared) {
            Linear(flattened, this->weight[mlp + "gate.weight"], Data(), routerLogits);
        }
        // Expert selection is defined in float32 (and the generic
        // SelectExpert contract requires it). Keep only the narrow router
        // tensor in float32 while larger activations retain their dtype.
        if (runRoutedExperts && !prepared)
            ToDataType(routerLogits, DataType::FLOAT32);
        bool expertsSelected = prepared != nullptr;
        auto selectExperts = [&]() {
            if (!runRoutedExperts || expertsSelected)
                return;
            SelectMoEExperts(routerLogits, expertIndex, expertScore);
            expertsSelected = true;
        };
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        const bool numaMoe = moeDevice == "numa" || moeDevice.rfind("numa:", 0) == 0;
        const bool hostPrefill =
            hostMoe && numaMoe && batch * sequence >= kNumasMoeGpuPrefillMinRows;
        const std::vector<int> prefillDevices =
            prefill ? std::vector<int>{FastllmCudaGetDevice()} : std::vector<int>{};
        NumasMoeCudaAssistScope assistScope(
            prefill ? &prefillDevices : (hostPrefill ? &threadTpOwner->devices : nullptr));
        // Stage buffers remain distinct while CPU jobs share the same pool.
        // Unsupported/CPU-only paths simply keep the outer lock until return.
        if (runRoutedExperts && batch * sequence <= kNumasMoePrefetchMaxRows &&
            flattened.dataDevice == DataDevice::CUDA && numaMoe &&
            !FastllmCudaMoeCacheRequested() && !FastllmCudaGraphIsCapturing()) {
            selectExperts();
            PrefetchNumasMoeDecodeInput(flattened, expertIndex, expertScore, deviceLayer);
        }
#endif
#ifdef USE_CUDA
        // Shared and routed experts consume the same normalized input and
        // router result but do not depend on one another. During CUDA graph
        // capture these markers let the generic MoE graph optimizer turn the
        // following serial region into two branches. They are no-ops for
        // eager, CPU and mixed-device execution.
        FastllmCudaGraphMarkParallelFork(deviceLayer);
#endif
        // Keep every shared-branch allocation alive through the routed
        // branch's join marker.  The graph optimizer may execute those two
        // branches concurrently, so allowing the capture memory pool to
        // recycle a shared intermediate into routed-MoE workspace would turn
        // an otherwise valid dependency rewrite into a data race.
        Data sharedGateUp, sharedHidden, sharedGate;
        auto runSharedExpert = [&](Data &sharedResult) {
            if (prepared) {
                sharedGraph();
                if (&sharedResult != &sharedOutput)
                    sharedResult.CopyFrom(sharedOutput);
                return;
            }
            RunSharedExpert(mlp, flattened, sharedGateUp, sharedHidden, sharedGate, sharedResult);
        };
#if defined(USE_CUDA) && !defined(USE_ROCM)
        // Hybrid experts produce FP32 for every verifier row. Borrow the
        // stable graph boundary for small batches as well as single decode;
        // unsupported backends retain the owning-result/copy fallback.
        auto runExpertParallel = [&](const auto &run) {
            Data borrowed;
            Data *target = &output;
            if (prepared && prepared->outputTarget &&
                batch * sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH &&
                (prepared->outputTarget->cudaData == nullptr ||
                 prepared->outputTarget->dataType == DataType::FLOAT32) &&
                Qwen4PrepareCudaWorkspaceOnCurrentDevice(*prepared->outputTarget, DataType::FLOAT32,
                                                         flattened.dims, FastllmCudaGetDevice())) {
                borrowed.FakeFrom(*prepared->outputTarget, 0);
                borrowed.Resize(flattened.dims);
                target = &borrowed;
            }
            const bool accepted = run(*target);
            if (accepted && target == &borrowed) {
                output.FakeFrom(borrowed, 0);
                output.Resize(borrowed.dims);
            }
            return accepted;
        };
        auto finishHybrid = [&](bool expertParallel = false) {
            FastllmCudaGraphMarkParallelJoin(deviceLayer);
            output.Reshape(input.dims);
            sharedOutput.Reshape(input.dims);
            ApplyDeviceMap(this->deviceMap, deviceLayer + 1, this->block_cnt);
            if (sharedOutput.dataType != output.dataType)
                ToDataType(sharedOutput, output.dataType);
            if (prepared && expertParallel && !reduceOutput)
                prepared->result.sharedDeferred = true;
            else
                AddTo(output, sharedOutput);
            if (reduceOutput)
                ThreadTpAllReduce(output);
        };
#if defined(USE_NUMAS)
        if (!prefill && threadTpRank < 0 && numaMoe &&
            batch * sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH &&
            FastllmCudaMoeCacheRequested()) {
            selectExperts();
            if (runExpertParallel([&](Data &target) {
                    return TryRunPipelineMoe(
                        deviceLayer, flattened, expertIndex, expertScore, target, moeWeights,
                        [&] { runSharedExpert(sharedOutput); },
                        prepared ? &prepared->result.peerOutput : nullptr,
                        prepared && prepared->readbackPrepared);
                })) {
                finishHybrid(true);
                return;
            }
        }
#endif
        if (hostMoe && threadTpOwner->hybrid.expertParallel &&
            batch * sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH && MoeCudaCacheRequested()) {
            selectExperts();
            if (runExpertParallel([&](Data &target) {
                    return FastllmCudaMergeMOEExpertParallel(
                        *threadTpOwner->hybrid.expertParallel, threadTpRank, flattened, expertIndex,
                        expertScore, target, moeWeights.data(), moeWeights.size(), deviceLayer,
                        [&] { runSharedExpert(sharedOutput); }, 0,
                        prepared && prepared->readbackPrepared);
                })) {
                finishHybrid(true);
                return;
            }
        }
#endif
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        if (runRoutedExperts && numaMoe &&
            batch * sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH &&
            SelectDeviceFromMap(this->deviceMap, deviceLayer + 1, this->block_cnt).find("cuda") == 0 &&
            FastllmCudaMoeCacheRequested()) {
            selectExperts();
            // Verification has the same independent shared/routed branches as
            // single-token decode. Submit shared work after routing readback,
            // beside NUMA experts, rather than making every layer's readback
            // wait for the shared projection. Join at the next dense graph.
            if (runExpertParallel([&](Data &target) {
                    return fastllm::RunHybridMoe(flattened, expertIndex, expertScore,
                        target, moeWeights.data(), moeWeights.size(), deviceLayer,
                        [&] { runSharedExpert(sharedOutput); });
                })) {
                finishHybrid(true);
                return;
            }
        }
#endif
        runSharedExpert(runRoutedExperts ? sharedOutput : output);

#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        if (hostPrefill) {
            selectExperts();
            // NUMA prefill launches expert workers on both TP devices. Their
            // streams share the temporary pool with the rank streams, whose
            // released intermediates may still be in use by queued kernels.
            // Drain both producers before either device's worker can reuse
            // those buffers. Single-row decode and MTP verification stay on
            // their existing stream-local paths.
            FastllmCudaSyncCurrentThreadStream();
            threadTpOwner->Barrier();
        }
#endif

#ifdef USE_CUDA
        FastllmCudaGraphMarkParallelFirstDone(deviceLayer);
        FastllmCudaGraphMarkParallelSecondBegin(deviceLayer);
        if (!runRoutedExperts ||
            (threadTpRank >= 0 && moeWeights.size() >= 4 && moeWeights[2] != nullptr &&
             moeWeights[3] != nullptr &&
             moeWeights[2]->dims == std::vector<int>({0, input.dims.back()}) &&
             moeWeights[3]->dims == std::vector<int>({input.dims.back(), 0}))) {
            // Host experts contribute once on their owner; other ranks retain
            // their shared-expert slice and join the same final reduction.
            // Empty CUDA routed slices use the same additive identity.
            FastllmCudaGraphMarkParallelJoin(deviceLayer);
            if (runRoutedExperts)
                output.CopyFrom(sharedOutput);
            output.Reshape(input.dims);
#if defined(USE_NUMAS) && !defined(USE_ROCM)
            // The owner's prefill workers also use this GPU. Do not enqueue
            // a waiting NCCL collective while they may allocate workspace.
            if (hostPrefill)
                threadTpOwner->Barrier();
#endif
            if (reduceOutput)
                ThreadTpAllReduce(output);
            return;
        }
#endif
        selectExperts();
        // NUMA acquires this guard around CPU work after preparing its private
        // CUDA branch. The generic CPU backend retains the outer guard.
        if (moeDevice == "cpu") hostExperts.LockCpu();
        const std::string outputDevice =
            SelectDeviceFromMap(this->deviceMap, deviceLayer + 1, this->block_cnt);
        bool hybridMoe = false;
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (outputDevice.find("cuda") == 0) {
            hybridMoe = fastllm::RunHybridMoe(flattened, expertIndex, expertScore, output,
                                              moeWeights.data(), moeWeights.size(), deviceLayer);
        }
#endif
        const bool useMoeCudaCache =
            hybridMoe || TryApplyMoeCudaCache(flattened, expertIndex, expertScore, moeWeights,
                                              outputDevice, MoeGateSwiglu);
        // Host TP ranks own the result before reduction. Preserve the existing
        // output transfer and ownership for serial layer transitions.
        const bool writeRoutedDirectly = hostMoe || useMoeCudaCache || moeDevice == outputDevice;
        if (!useMoeCudaCache && !moeDevice.empty()) {
            ApplyDeviceMap({{moeDevice, 1}}, 1, 1);
        }

        Data routed;
        Data &routedOutput = writeRoutedDirectly ? output : routed;
        Data w1, w2, w3, temporaryInput, temporaryOutput;
        if (!hybridMoe) {
            MergeMOE(flattened, expertIndex, expertScore, moeWeights, moeBiass, w1, w2, w3,
                     temporaryInput, temporaryOutput, 1.0f, routedOutput, deviceLayer);
        }
#ifdef USE_CUDA
        FastllmCudaGraphMarkParallelJoin(deviceLayer);
#endif
        routedOutput.Reshape(input.dims);
        sharedOutput.Reshape(input.dims);

        if (writeRoutedDirectly) {
            ApplyDeviceMap(this->deviceMap, deviceLayer + 1, this->block_cnt);
        } else {
            Data routedCopy;
            routedCopy.CopyFrom(routedOutput);
            ApplyDeviceMap(this->deviceMap, deviceLayer + 1, this->block_cnt);
            output.CopyFrom(routedCopy);
        }
        if (sharedOutput.dataType != output.dataType)
            ToDataType(sharedOutput, output.dataType);
        AddTo(output, sharedOutput);
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        if (hostPrefill)
            threadTpOwner->Barrier();
#endif
        if (reduceOutput)
            ThreadTpAllReduce(output);
    }

    void Qwen4ExpModel::PrefetchHybridExperts(const Data &inputIds, bool pipelinedPrefill,
                                              int layer, int endLayer) {
        // Resolve expert placement only for eligible pipelined prefill. Decode
        // and ordinary prefill never schedule this host-weight prefetch.
        if (!pipelinedPrefill || inputIds.dims[1] < 1024 || layer < 0 || layer >= endLayer)
            return;
        HybridMoePrefetch(weights, layer, endLayer, true, inputIds.dims[1],
                          SelectMoeDeviceForLayer(layer));
    }

    bool
    Qwen4ExpModel::ShouldDelaySpecialWeightNumaRegistration(const std::string &weightName) const {
        if (!MoeCudaCacheRequested()) {
            return false;
        }
        const std::string mainLayerPrefix = languagePrefix + "layers.";
        return (weightName.rfind(mainLayerPrefix, 0) == 0 ||
                weightName.rfind("mtp.layers.0.mlp.experts.", 0) == 0) &&
               weightName.find(".mlp.experts.") != std::string::npos &&
               (Qwen4EndsWith(weightName, ".gateup_proj.weight") ||
                Qwen4EndsWith(weightName, ".down_proj.weight"));
    }

    void Qwen4ExpModel::PrepareHybridExperts(bool hasMtpWeights) {
        if (MoeCudaCacheRequested()) {
            HybridMoeCachePlan cache;
            const bool includeDraft = hasMtpWeights && !weights.empty() && weights[0].size() > 2 &&
                weights[0][2]->isGGUFData && mtpMoeWeights[2]->isGGUFData;
            const int targetLayers = includeDraft ? block_cnt : static_cast<int>(weights.size());
            for (int layer = 0; layer < targetLayers; ++layer)
                cache.Add(weights[layer], SelectMoeDeviceForLayer(layer));
            if (includeDraft) {
                cache.Add(mtpMoeWeights, SelectSpecialWeightDevice(
                    "mtp.layers.0.mlp.experts.0.gateup_proj.weight", block_cnt - 1));
            }
            std::function<void()> registerNumaWeights;
#ifdef USE_NUMAS
            registerNumaWeights = [this] { WarmupNumaMoeWeights(); };
#endif
            const bool cachePrepared = cache.Prepare(registerNumaWeights);
            if (!cachePrepared) {
                std::fprintf(stderr, "[Fastllm] Qwen4 CUDA expert cache unavailable; using the "
                                     "configured MoE device.\n");
            }
        }
        // Direct API callers can skip AutoWarmup. Register all NUMA experts
        // before requests (and before TP borrows their storage), even without
        // a CUDA expert cache. Lazy per-route registration frees source heaps
        // piecemeal while allocating NUMA shards; unreclaimed source pages can
        // otherwise nearly double host RSS. Warmup also trims those pages.
        WarmupNumaMoeWeights();
    }

    void Qwen4ExpModel::ConfigureKvMem(const KvMemConfig &config) {
        if (mainLoop != nullptr || autoWarmupRunning.load() || kvMemLocked ||
            GetPagedKVCacheManager(kvCacheId, true) != nullptr)
            throw std::invalid_argument("KVMem must be configured before warmup or inference");
        if (!config.enabled) {
            kvMemConfig = config;
            return;
        }
#ifndef USE_CUDA
        throw std::invalid_argument("KVMem requires a CUDA build");
#else
        if (maxBatch > 1) throw std::invalid_argument("KVMem currently requires max_batch=1");
        if (config.maxTokens > max_positions)
            throw std::invalid_argument("KVMem logical context exceeds the model's configured context");
        if (GetFastllmEnv().skipWarmup) throw std::invalid_argument("KVMem requires warmup");
        ValidateKvMemModel(config);
        kvMemConfig = config;
        maxBatch = 1;
        saveHistoryChat = false;
        tokensLimit = promptLimit = config.maxTokens;
#endif
    }

    void Qwen4ExpModel::ValidateKvMemModel(const KvMemConfig &config) const {
        if (!config.enabled) return;
        if (!SupportsKvMem())
            throw std::invalid_argument("Qwen4 KVMem requires the Hybrid execution path");
        if (GetKVCacheInCPU() || deviceMap.empty() || dataType != FLOAT16 ||
            kvCacheDataType != FLOAT16 || (head_dim != 128 && head_dim != 256) ||
            indexerCompressRatio != 4 || indexerBudget <= 0)
            throw std::invalid_argument("Qwen4 KVMem requires CUDA FP16 QSA (head_dim 128/256, compression ratio 4)");
        for (const auto &entry : deviceMap) {
            if (entry.first != "cuda" && entry.first.rfind("cuda:", 0) != 0)
                throw std::invalid_argument("Qwen4 KVMem requires CUDA attention devices");
        }
        if (config.maxTokens < 2 || config.maxTokens > std::numeric_limits<int>::max() - 128 ||
            config.residentTokens < 4 || config.residentTokens % 4 ||
            config.prefillTokens < 1 || config.prefillTokens > config.maxTokens)
            throw std::invalid_argument("Qwen4 KVMem requires positive capacities and resident_tokens divisible by 4");
        // Include a possible MTP attention layer and its provisional suffix.
        // Rank-local caches split KV heads; this validation runs on the root.
        uint64_t layers = 0;
        for (int layer = 0; layer < block_cnt; ++layer) layers += !IsLinearAttentionLayer(layer);
        if (!layers) throw std::invalid_argument("Qwen4 KVMem requires full-attention layers");
        ++layers;
        const uint64_t tokens = (uint64_t(config.maxTokens) + Qwen4MtpDraftsPerStep() + 4) / 4 * 4;
        if (tokens * layers * uint64_t(num_key_value_heads) * head_dim * 4 > config.hostBytes)
            throw std::invalid_argument("Qwen4 KVMem host budget cannot back target and MTP KV");
    }

    void Qwen4ExpModel::PrepareHybridKvMemCache(int layer, const Data &key, int heads,
                                               Data &pastKey, Data &pastValue) {
#ifdef USE_CUDA
        Qwen4CudaDeviceGuard deviceGuard(key.dataDeviceIds);
        if (auto *existing = Qwen4GetKvMem(pastKey)) {
            existing->Reset();
            return;
        }
        AssertInFastLLM(key.dataDevice == DataDevice::CUDA && key.dataDeviceIds.size() == 1,
                        "Qwen4 KVMem requires a rank-local CUDA cache.");
        std::shared_ptr<Qwen4KvMem> storage;
        if (servingCache) {
            const auto found = servingCache->kvMemStorage.find(layer);
            if (found != servingCache->kvMemStorage.end())
                storage = std::dynamic_pointer_cast<Qwen4KvMem>(found->second);
        }
        if (storage) storage->Reset();
        else {
            KvMemConfig storageConfig = kvMemConfig;
            storageConfig.maxTokens += Qwen4MtpDraftsPerStep();
            storage = std::make_shared<Qwen4KvMem>(storageConfig, key.dataType,
                heads, head_dim, key.dataDeviceIds.front());
        }
        storage->Attach(pastKey, pastValue, storage);
#endif
    }

    size_t Qwen4ExpModel::RetainHybridKvMemStorage(
            const std::vector<std::pair<Data, Data>> &cache,
            const std::shared_ptr<KvMemCache> &mtp, ServingCache &reserved) const {
        size_t bytes = 0;
#ifdef USE_CUDA
        for (int layer = 0; layer <= block_cnt; ++layer) {
            auto owner = layer == block_cnt ? mtp : cache[layer].first.kvMemCache;
            if (!owner) continue;
            auto storage = std::dynamic_pointer_cast<Qwen4KvMem>(owner);
            AssertInFastLLM(storage != nullptr, "Qwen4 Hybrid KV storage type mismatch.");
            bytes += storage->gpuBytes;
            reserved.kvMemStorage.emplace(layer, std::move(owner));
        }
#endif
        return bytes;
    }

    void Qwen4ExpModel::FinishHybridKvMemWarmup() {
        if (kvMemConfig.enabled) {
            kvMemLocked = true;
            std::printf("[Fastllm] Qwen4 KVMem: logical=%d, resident=%d tokens/layer; exact QSA selection, host backing, CUDA Graph enabled when available.\n",
                kvMemConfig.maxTokens, kvMemConfig.residentTokens);
        }
    }

    std::vector<int> Qwen4ExpModel::ForwardHybrid(
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
        ForwardState frame(*this, ExecutionMode::Hybrid, batch, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig,
            lastTokens, retLogits, targetMultiHidden, allVerificationTokens, verificationAccepted,
            verificationCapture, recordPrefixSnapshot, allowDecodeCudaGraph, decodeEquivalentGdn,
            hostInputTokens, materializeCausalMaskOnGraphFallback, precomputedEmbedding, prefill);
        frame.BeginHybridStep();
        frame.Prepare();
        frame.PrefetchExperts(frame.FirstLayer());
        for (int layer = frame.FirstLayer(); layer < frame.EndLayer(); ++layer) {
            const int action = frame.BeginLayer(layer);
            if (action == ForwardState::Complete) break;
            if (action == ForwardState::Skip) continue;
            frame.RunExperts(layer);
            frame.PrefetchExperts(layer + 1);
            frame.EndLayer(layer);
        }
        return frame.Finish();
    }
} // namespace fastllm
