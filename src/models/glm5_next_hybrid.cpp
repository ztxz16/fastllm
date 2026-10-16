// Independent host-expert forward: NUMA overlap, device groups and cache policy.
#include "glm5_next.h"
#include "glm5_next_tp_state.h"
#include "glm5_next_gguf.h"
#include "glm5_next_hybrid_graph.h"
#include "glm5_next_dsa.h"
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include "blocks/baseblock.h"
#include "utils.h"
#include "utils/hybrid_moe.h"
#include "utils/hybrid_parallel.h"
#include "devices/disk/diskdevice.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#ifndef USE_ROCM
#include "devices/cuda/fastllm-cuda-hybrid-prefill.h"
#include "devices/cuda/fastllm-cuda-hybrid-kda.h"
#include "devices/cuda/fastllm-cuda-hybrid-dsa.h"
#include "devices/cuda/fastllm-cuda-hybrid-output.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#endif
#endif
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif

namespace fastllm {
    bool Glm5NextModel::TryHybridDsaProjections(int layer, Data &input, Data &normalized,
            Data &key, Data &gate, Data &query, Data &head) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode != ExecutionMode::Hybrid || threadTpRank >= 0 || mtpEnabled) return false;
        const std::string prefix = languagePrefix + "layers." + std::to_string(layer) + ".self_attn.indexer.";
        return FastllmCudaHybridDsaProjections(input, normalized,
            weight[prefix + "wk.weight"], weight[prefix + "index_kpool_compress_gate"],
            weight[prefix + "wq_b.weight"], weight[prefix + "weights_proj.weight"],
            weight[prefix + "k_norm.weight"], weight[prefix + "k_norm.bias"], key, gate, query, head);
#else
        return false;
#endif
    }

    bool Glm5NextModel::TryHybridRouterScores(const std::string &mlp, Data &input, Data &scores) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode != ExecutionMode::Hybrid || threadTpRank >= 0 || mtpEnabled ||
            input.dims.empty() || input.Count(0) != 4096 || input.lockInCPU || input.multiDeviceData)
            return false;
        const auto &router = weight[mlp + "gate.weight"];
        if (router.lockInCPU || router.multiDeviceData) return false;
        // Reuse the existing exact BF16-input/FP32-weight router kernel.
        // Selection, correction bias and route normalization stay shared.
        return FastllmCudaNaiveRouterSigmoid(input, router, scores);
#else
        return false;
#endif
    }

    bool Glm5NextModel::TryHybridOutputNorm(Data &input, Data &output) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode == ExecutionMode::Hybrid && threadTpRank < 0 && !mtpEnabled)
            return FastllmCudaHybridHcMeanNorm(input, weight[languagePrefix + "norm.weight"],
                rms_norm_eps, output);
#endif
        return false;
    }
    bool Glm5NextModel::TryHybridGreedy(Data &logits, const GenerationConfig &config, int &token) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode == ExecutionMode::Hybrid && threadTpRank < 0 && !mtpEnabled &&
            config.IsSimpleGreedy() && !config.output_logits) {
            Data top;
            if (FastllmCudaHybridTop1(logits, top)) {
                top.ToDevice(DataDevice::CPU);
                token = int(((float*)top.cpuData)[0] + 1e-3f);
                return true;
            }
        }
#endif
        return false;
    }
#if defined(USE_CUDA) && !defined(USE_ROCM)
    bool Glm5NextModel::HybridDecodeState::MlaSegment::Run(
            Glm5NextIndexerCache &cache, Data &key, Data &gate, Data &query,
            Data &weights, Data &ape, const Data &pagedCache,
            const std::vector<const Data *> &extraBindings,
            const std::function<void(Data &, Data &)> &finish, Data &output) {
        CudaWorkspaceScope workspace(nullptr);
        const int past = cache.tokens;
        if (past < 2048 || cache.captureReplay || !FastllmCudaHybridDsaSupported() ||
            FastllmCudaGraphIsCapturingFast() || key.dataDeviceIds.size() != 1 ||
            key.dims != std::vector<int>({1, 1, 128}) || gate.dims != key.dims ||
            query.dims != std::vector<int>({1, 1, 32, 128}) || weights.Count(0) != 32 ||
            ape.Count(0) != 512 || cache.keys.dims != std::vector<int>({1, past / 4, 128}) ||
            !pagedCache.isPagedKVCache || pagedCache.dims.size() < 2 || pagedCache.dims[1] != past + 1 ||
            pagedCache.pageLen <= 0 || !pagedCache.pagedKVCacheData ||
            pagedCache.pageIndex.size() < size_t(past / pagedCache.pageLen + 1)) return false;
        const int device = key.dataDeviceIds[0];
        if (FastllmCudaGetDevice() != device) return false;
        for (const auto *x : {&key, &gate, &query, &cache.keys})
            if (x->dataDevice != DataDevice::CUDA || x->dataType != DataType::BFLOAT16 ||
                !x->cudaData || x->dataDeviceIds != key.dataDeviceIds || x->multiDeviceData) return false;
        for (const auto *x : {&weights, &ape})
            if (x->dataDevice != DataDevice::CUDA || x->dataType != DataType::FLOAT32 ||
                !x->cudaData || x->dataDeviceIds != key.dataDeviceIds || x->multiDeviceData) return false;
        if (past % 4 && (cache.tailKeys.dims != std::vector<int>({1, past % 4, 128}) ||
            cache.tailGates.dims != cache.tailKeys.dims)) return false;
        if (past % 4) for (const auto *tail : {&cache.tailKeys, &cache.tailGates})
            if (tail->dataDevice != DataDevice::CUDA || tail->dataType != DataType::BFLOAT16 ||
                !tail->cudaData || tail->dataDeviceIds != key.dataDeviceIds) return false;
        if (pagedCache.pagedKVCacheData->dataDevice != DataDevice::CUDA ||
            pagedCache.pagedKVCacheData->dataType != DataType::BFLOAT16 ||
            pagedCache.pagedKVCacheData->dataDeviceIds != key.dataDeviceIds) return false;

        // Capacity buckets bound graph scratch without rebuilding on every
        // KPool append. Existing valid rows remain in the request's cache.
        const int capacity = ((past + 1) / 4 + 255) / 256 * 256;
        if (cache.keys.expansionBytes < size_t(capacity) * 128 * 2)
            cache.keys.Expansion({1, capacity, 128});
        for (Data *tail : {&cache.tailKeys, &cache.tailGates}) {
            tail->dataType = DataType::BFLOAT16;
            tail->UpdateUnitSize();
            tail->ToDevice(DataDevice::CUDA, key.dataDeviceIds, false);
            if (tail->expansionBytes < 4 * 128 * 2) tail->Expansion({1, 4, 128});
        }
        if (cache.pageTableIds != pagedCache.pageIndex ||
            cache.pageTable.dataDeviceIds != key.dataDeviceIds || !cache.pageTable.cudaData) {
            cache.pageTable.dataType = DataType::INT32;
            cache.pageTable.UpdateUnitSize();
            cache.pageTable.ToDevice(DataDevice::CUDA, key.dataDeviceIds, false);
            int pages = 1;
            while (pages < (int)pagedCache.pageIndex.size()) pages *= 2;
            if (cache.pageTable.expansionBytes < size_t(pages) * sizeof(int32_t))
                cache.pageTable.Expansion({pages});
            cache.pageTable.Resize({int(pagedCache.pageIndex.size())});
            FastllmCudaCopyFromHostToDevice(cache.pageTable.cudaData,
                (void*)pagedCache.pageIndex.data(), pagedCache.pageIndex.size() * sizeof(int32_t));
            cache.pageTableIds = pagedCache.pageIndex;
        }

        std::vector<const void *> bindings;
        for (const auto *x : {&key, &gate, &query, &weights, &ape, &cache.keys,
                &cache.tailKeys, &cache.tailGates, &cache.pageTable}) bindings.push_back(x->cudaData);
        bindings.push_back(pagedCache.pagedKVCacheData->cudaData);
        bindings.push_back(cache.hadamard.cudaData);
        std::vector<std::vector<int>> shapes{{capacity, pagedCache.pageLen}, pagedCache.pagedKVCacheData->dims};
        for (const auto *x : extraBindings) { bindings.push_back(x->cudaData); shapes.push_back(x->dims); }
        if (graph && (nextPosition != past || graph->device != device ||
            graph->bindings != bindings || graph->shapes != shapes))
            graph.reset();
        if (!graph) {
            position.dataType = DataType::INT32;
            position.UpdateUnitSize();
            position.Resize({1});
            position.ToDevice(DataDevice::CUDA, key.dataDeviceIds, false);
            position.Allocate(false);
            FastllmCudaCopyFromHostToDevice(position.cudaData, (void*)&past, sizeof(past));
            graph = std::make_unique<Segment>(device, 1);
            graph->bindings = bindings;
            graph->shapes = shapes;
        }
        graph->RunBound([&](std::vector<Data> &out) {
            Data tailKeys, tailGates, keys, pooled, quantized, scores, indices;
            tailKeys.FakeFrom(cache.tailKeys, 0); tailKeys.Resize({1, 4, 128});
            tailGates.FakeFrom(cache.tailGates, 0); tailGates.Resize({1, 4, 128});
            keys.FakeFrom(cache.keys, 0); keys.Resize({1, capacity, 128});
            AssertInFastLLM(FastllmCudaHybridDsaAppendPool(key, gate, ape, position, tailKeys, tailGates, pooled),
                "GLM hybrid DSA tail append/pooling failed.");
            // Preserve the original BF16/Hadamard boundary. Only a complete
            // four-token group's quantized result is committed to the cache.
            glm5_next_detail::RotateAndQuantizeIndexer(pooled, cache.hadamard, quantized, true);
            AssertInFastLLM(FastllmCudaHybridDsaStorePool(quantized, position, keys),
                "GLM hybrid DSA pooled append failed.");
            AssertInFastLLM(FastllmCudaHybridDsaScore(query, weights, keys, position, scores),
                "GLM hybrid DSA scoring failed.");
            AssertInFastLLM(FastllmCudaHybridDsaSelect(scores, position, cache.pageTable,
                pagedCache.pageLen, indices), "GLM hybrid DSA selection failed.");
            finish(indices, out[0]);
            AssertInFastLLM(FastllmCudaHybridDsaAdvance(position), "GLM hybrid DSA position failed.");
        }, query.GetBytes());
        ++cache.tokens;
        nextPosition = cache.tokens;
        cache.keys.Resize({1, cache.tokens / 4, 128});
        for (Data *tail : {&cache.tailKeys, &cache.tailGates})
            tail->Resize(cache.tokens % 4 ? std::vector<int>{1, cache.tokens % 4, 128} : std::vector<int>{});
        // The caller consumes this result before the next replay for this
        // layer; graph ownership survives the caller's temporary Data view.
        output.FakeFrom(graph->outputs[0], 0);
        output.Resize(graph->outputs[0].dims);
        return true;
    }
#endif

    bool Glm5NextModel::TryHybridMlaDecode(int layer, Data &query, Data &key, Data &gate,
            Data &indexQuery, Data &headWeights, Data &keyWeight, Data &valueWeight, Data &output,
            const std::vector<std::vector<std::pair<Data, Data>>*> &requestCaches) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode != ExecutionMode::Hybrid || threadTpRank >= 0 || mtpEnabled ||
            !GetFastllmEnv().cudaGraph || requestCaches.size() != 1 ||
            !FastllmCudaHybridDsaSupported() || FastllmCudaGraphIsCapturingFast()) return false;
        // RunCompressedMlaAttention holds indexerCachesMutex while preparing
        // this request; this lock only protects the graph ownership registry.
        auto &caches = indexerCaches[requestCaches[0]];
        if ((int)caches.size() <= layer) caches.resize(std::max(block_cnt, layer + 1));
        auto &kv = requestCaches[0]->at(layer);
        const std::string prefix = languagePrefix + "layers." + std::to_string(layer) + ".self_attn.";
        auto &ape = weight[prefix + "indexer.index_kpool_compress_ape"];
        auto &projection = weight[prefix + "o_proj.weight"];
        if (!hybridDecodeState) hybridDecodeState = std::make_shared<HybridDecodeState>();
        std::lock_guard<std::mutex> guard(hybridDecodeState->mutex);
        auto &segment = hybridDecodeState->mlaRequests[requestCaches[0]][layer];
        if (!segment) segment = std::make_unique<HybridDecodeState::MlaSegment>();
        return segment->Run(caches[layer], key, gate, indexQuery, headWeights, ape, kv.second,
            {&query, &keyWeight, &valueWeight, &projection}, [&](Data &indices, Data &out) {
                Data absorbed, attended, heads;
                MatMul(query, keyWeight, absorbed);
                ToDataType(absorbed, DataType::BFLOAT16);
                glm5_next_detail::SparseLatentAttention(absorbed, kv.second, indices,
                    1.0f / std::sqrt((float)qkHeadDim), attended, false, &kv.first, true);
                absorbed.FreeSpace();
                MatMulTransB(attended, valueWeight, heads);
                attended.FreeSpace();
                heads.Reshape({num_attention_heads, 1, 1, valueHeadDim});
                PermuteSelf(heads, {1, 2, 0, 3});
                heads.Reshape({1, 1, num_attention_heads * valueHeadDim});
                Linear(heads, projection, Data(), out);
            }, output);
#else
        return false;
#endif
    }

    bool Glm5NextModel::TryHybridKdaInputProjections(int layer, Data &input, Data &f, Data &beta, Data &g) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode != ExecutionMode::Hybrid || threadTpRank >= 0 || mtpEnabled) return false;
        const std::string prefix = languagePrefix + "layers." + std::to_string(layer) + ".self_attn.";
        return FastllmCudaHybridKdaInputProjections(input, weight[prefix + "f_a_proj.weight"],
            weight[prefix + "b_proj.weight"], weight[prefix + "g_a_proj.weight"], f, beta, g);
#else
        return false;
#endif
    }

    bool Glm5NextModel::TryHybridKdaDecode(int layer, Data &q, Data &k, Data &v,
            Data &f, Data &g, Data &beta, std::pair<Data, Data> &cache, Data &output) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode != ExecutionMode::Hybrid || threadTpRank >= 0 || mtpEnabled) return false;
        const std::string prefix = languagePrefix + "layers." + std::to_string(layer) + ".self_attn.";
        if (!FastllmCudaHybridKdaDecode(q, k, v, f, g, beta,
                weight[prefix + "f_b_proj.weight"], weight[prefix + "g_b_proj.weight"],
                weight[prefix + "q_conv1d.weight"], weight[prefix + "k_conv1d.weight"],
                weight[prefix + "v_conv1d.weight"], weight[prefix + "A_log"],
                weight[prefix + "dt_bias"], weight[prefix + "o_norm.weight"],
                cache.first, cache.second, kdaHeads, kdaHeadDim, shortConvKernel,
                gateLowerBound, rms_norm_eps, output)) return false;
        cache.second.SetKVCache();
        cache.second.isLinearAttention = true;
        return true;
#else
        return false;
#endif
    }

    void Glm5NextModel::RemoveHybridDecodeRequest(const std::vector<std::pair<Data, Data>> *key) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (hybridDecodeState) {
            std::lock_guard<std::mutex> lock(hybridDecodeState->mutex);
            hybridDecodeState->requests.erase(key);
            hybridDecodeState->mlaRequests.erase(key);
        }
#endif
    }

    bool Glm5NextModel::TryHybridDecodePrepare(int layer, const std::string &prefix, Data &input,
            Data &attention, Data &normalizedFfn, Data &post, Data &comb,
            const std::vector<std::vector<std::pair<Data, Data>>*> &requestCaches) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode != ExecutionMode::Hybrid || threadTpRank >= 0 || mtpEnabled ||
            !GetFastllmEnv().cudaGraph || requestCaches.size() != 1 || !kdaLayers[layer] ||
            input.dataDevice != DataDevice::CUDA || input.dataType != DataType::BFLOAT16 ||
            input.dims.size() != 4 || input.dims[0] != 1 || input.dims[1] != 1 ||
            input.dataDeviceIds.size() != 1 || FastllmCudaGraphIsCapturingFast()) return false;
        auto &cache = requestCaches[0]->at(layer);
        if (!cache.first.cudaData || !cache.second.cudaData ||
            cache.first.dataDeviceIds != input.dataDeviceIds ||
            cache.second.dataDeviceIds != input.dataDeviceIds) return false;
        const int device = input.dataDeviceIds[0];
        FastllmCudaSetDevice(device);
        if (!hybridDecodeState) hybridDecodeState = std::make_shared<HybridDecodeState>();
        std::lock_guard<std::mutex> lock(hybridDecodeState->mutex);
        auto &segment = hybridDecodeState->requests[requestCaches[0]][layer];
        const std::vector<const void *> bindings{cache.first.cudaData, cache.second.cudaData};
        const std::vector<std::vector<int>> shapes{input.dims, cache.first.dims, cache.second.dims};
        if (segment && (segment->device != device || segment->bindings != bindings || segment->shapes != shapes))
            segment.reset();
        if (!segment) {
            segment = std::make_unique<HybridDecodeState::Segment>(device, 4);
            segment->bindings = bindings;
            segment->shapes = shapes;
        }
        segment->Run(input, [&](Data &fixedInput, std::vector<Data> &out) {
            RunAttentionBlock(layer, prefix, fixedInput, out[0], requestCaches, nullptr, nullptr);
            PrepareFfnBlock(prefix, out[0], out[1], out[2], out[3]);
        });
        // This tensor is reused by the next layer, possibly on another PP
        // device. Preserve its owning allocation instead of exporting a view.
        HybridDecodeState::Segment::CopyActivation(attention, segment->outputs[0]);
        Data *outputs[]{nullptr, &normalizedFfn, &post, &comb};
        for (int i = 1; i < 4; ++i) {
            outputs[i]->FakeFrom(segment->outputs[i], 0);
            outputs[i]->Resize(segment->outputs[i].dims);
        }
        return true;
#else
        return false;
#endif
    }

    bool Glm5NextModel::TryHybridMlaProjections(int layer, Data &input,
            const std::vector<Data *> &outputs,
            const std::function<void(Data &, std::vector<Data> &)> &body) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode != ExecutionMode::Hybrid || threadTpRank >= 0 || mtpEnabled ||
            !GetFastllmEnv().cudaGraph || input.dataDevice != DataDevice::CUDA ||
            input.dataType != DataType::BFLOAT16 || input.dims.size() != 3 ||
            input.dims[0] != 1 || input.dims[1] != 1 || input.dataDeviceIds.size() != 1 ||
            FastllmCudaGraphIsCapturingFast()) return false;
        const int device = input.dataDeviceIds[0];
        FastllmCudaSetDevice(device);
        if (!hybridDecodeState) hybridDecodeState = std::make_shared<HybridDecodeState>();
        std::lock_guard<std::mutex> lock(hybridDecodeState->mutex);
        auto &segment = hybridDecodeState->mlaProjections[layer];
        const std::vector<std::vector<int>> shapes{input.dims};
        if (segment && (segment->device != device || segment->shapes != shapes)) segment.reset();
        if (!segment) {
            // The last tensor owns the immutable Hadamard transform. It must
            // outlive requests, just like this stateless projection graph.
            segment = std::make_unique<HybridDecodeState::Segment>(device, outputs.size() + 1);
            segment->shapes = shapes;
        }
        segment->Run(input, body);
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i]->FakeFrom(segment->outputs[i], 0);
            outputs[i]->Resize(segment->outputs[i].dims);
        }
        return true;
#else
        return false;
#endif
    }

    void Glm5NextModel::RunHybridMoeWithPrefix(
            int deviceLayer, const std::string &mlp,
            std::vector<Data*> &weights,
            std::vector<Data*> &biases,
            Data &input, int sequence, Data &output, HybridPrefillContext *prefill,
            HybridFfnFinish *finish) {
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
            // A hybrid caller may consume both expert outputs together with
            // its residual update, retaining the sum's BF16 boundary in-kernel.
            if (finish && executionMode == ExecutionMode::Hybrid && threadTpRank < 0 && !mtpEnabled)
                finish->completed = FastllmCudaHybridMoeHcPost(output, routedOutput,
                    finish->residual, finish->post, finish->comb, finish->output);
            if (!finish || !finish->completed) AddTo(output, routedOutput);
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
#if defined(USE_CUDA) && !defined(USE_ROCM)
        // Nested KDA captures use the FastLLM wrappers. Latch an external
        // capture once before avoiding repeated eager-path runtime queries.
        FastllmCudaGraphIsCapturing();
        struct ManagedCaptureScope {
            bool previous = FastllmCudaGraphSetManagedCaptureOnly(true);
            ~ManagedCaptureScope() { FastllmCudaGraphSetManagedCaptureOnly(previous); }
        } managedCapture;
        FastllmCudaHybridPrefillScope prefillMath(batch == 1 && sequence >= 1024);
#endif
        AssertInFastLLM(batch == 1 || (batch == sequence &&
            kdaReplay == nullptr && indexer == nullptr),
            "GLM-5.3 batched layers require one decode token per request.");
        Data hiddenStatesTemp;
        Data *current = &hiddenStates;
        Data *next = &hiddenStatesTemp;
        HybridMoeDecodeGroup decodeCaches;
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        const bool frequencyDecode = !prefill && sequence <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH;
        std::vector<CudaPrefillStage> decodeStages;
        if (frequencyDecode && threadTpRank < 0 && !threadTpState && !mtpEnabled &&
            !kdaReplay && !indexer && firstLayer == 0 && endLayer == block_cnt) {
            // This placement check rejects device revisits. Each completed
            // stage can upload its next-token cache while later stages run.
            if (!BuildCudaPrefillStages(deviceMap, block_cnt, decodeStages)) decodeStages.clear();
        }
        size_t decodeStage = 0;
#endif
        for (int layer = firstLayer; layer < endLayer; layer++) {
            ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
            const std::string prefix = languagePrefix + "layers." +
                std::to_string(layer) + ".";

            Data normalizedFfn, ffnPost, ffnComb;
            const bool prepared = !prefill && !kdaReplay && !indexer &&
                TryHybridDecodePrepare(layer, prefix, *current, *next,
                    normalizedFfn, ffnPost, ffnComb, requestCaches);
            if (!prepared) RunAttentionBlock(layer, prefix, *current, *next, requestCaches, kdaReplay, indexer);
            std::swap(current, next);
            if (!prepared) PrepareFfnBlock(prefix, *current, normalizedFfn, ffnPost, ffnComb);
            Data ffnOutput;
            HybridFfnFinish finish{*current, ffnPost, ffnComb, *next};
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
                     experts[2]->dataType == DataType::FP8_E4M3_BLOCK_128 ||
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
                    expertBiases[layer], normalizedFfn, sequence, ffnOutput, prefill, &finish);
            }
            if (!finish.completed) FinishFfnBlock(ffnOutput, *current, ffnPost, ffnComb, *next);
            std::swap(current, next);
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
            if (decodeStage + 1 < decodeStages.size() &&
                layer + 1 == decodeStages[decodeStage].endLayer) {
                decodeCaches.Finish(decodeStages[decodeStage].device);
                ++decodeStage;
            }
#endif
        }
        // Two residual swaps per layer preserve caller-owned state.
    }

    int Glm5NextModel::StreamingHybridLayer(const std::string &name) const {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        const auto arch = weight.dicts.find("gguf_architecture");
        if (threadTpState || executionMode != ExecutionMode::Hybrid || GetLowMemMode() ||
            arch == weight.dicts.end() ||
            (arch->second != "glm5next" && arch->second != "glm5-next")) return -1;
        const std::string prefix = languagePrefix + "layers.";
        if (name.rfind(prefix, 0) != 0) return -1;
        const char *start = name.c_str() + prefix.size();
        char *end = nullptr;
        const long layer = std::strtol(start, &end, 10);
        if (end == start || *end != '.' || layer < 0 || layer >= block_cnt) return -1;
        const std::string suffix(end + 1);
        if (suffix.rfind("self_attn.", 0) != 0 &&
            !(denseMlpLayers[layer] && suffix.rfind("mlp.", 0) == 0) &&
            !(GetCudaSharedExpert() && suffix.rfind("mlp.shared_experts.", 0) == 0)) return -1;
        const std::string device = SelectDeviceFromMap(deviceMap, layer + 1, block_cnt);
        if (device != "cuda" && device.rfind("cuda:", 0) != 0) return -1;
        std::map<int, int> ratios;
        return ParseDeviceIds(device, "cuda", ratios).size() == 1 ? int(layer) : -1;
#else
        return -1;
#endif
    }

    void Glm5NextModel::StageHybridWeights() {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        if (hybridLoadingLayer < 0) return;
        const int layer = hybridLoadingLayer;
        hybridLoadingLayer = -1;
        // Convert paired GGUF K/V tensors before uploading; the loader has
        // joined the group's workers, and restoration consumes source names.
        glm5_next_detail::RestoreGgufWeights(weight, layer + 1, num_attention_heads,
            qkNopeHeadDim, valueHeadDim, kvLoraRank, layer);
        std::map<int, int> ratios;
        const auto devices = ParseDeviceIds(
            SelectDeviceFromMap(deviceMap, layer + 1, block_cnt), "cuda", ratios);
        const int previousDevice = FastllmCudaGetDevice();
        for (auto &item : weight.weight) {
            if (StreamingHybridLayer(item.first) != layer) continue;
            Data &data = item.second;
            if (data.dims.size() == 2 && data.GetBytes() >= (1ULL << 20) &&
                data.isModelWeight && data.dataDevice == DataDevice::CPU && data.cpuData)
                data.ToDevice(DataDevice::CUDA, devices);
        }
        FastllmCudaSetDevice(previousDevice);
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
#endif
    }

    void Glm5NextModel::PrepareHybridMoeCache() {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        {
            HybridMoeCachePlan cache(HybridMoeMath::Glm5, swigluLimit);
            std::vector<int> cacheDevices(block_cnt, -1);
            std::vector<CudaPrefillStage> stages;
            if (!threadTpState && BuildCudaPrefillStages(deviceMap, block_cnt, stages))
                for (const auto &stage : stages)
                    std::fill(cacheDevices.begin() + stage.firstLayer,
                              cacheDevices.begin() + stage.endLayer, stage.device);
            const int cacheLayers = block_cnt + (mtpEnabled && mtpWeightsReady ? 1 : 0);
            for (int layer = 0; layer < cacheLayers; ++layer) {
                const std::string device = SelectMoeDeviceForLayer(std::min(layer, block_cnt - 1));
                const auto &experts = layer == block_cnt ? mtpExpertWeights : expertWeights[layer];
                if ((device == "numa" || device.rfind("numa:", 0) == 0) &&
                    experts.size() >= 4 && experts[2] &&
                    (experts[2]->dataType == DataType::NVFP4_BLOCK_16_E4M3_PACKED ||
                     experts[2]->dataType == DataType::FP8_E4M3 ||
                     experts[2]->dataType == DataType::FP8_E4M3_BLOCK_128 ||
                     experts[2]->dataType == DataType::DATA_GGUF_FORMAT)) {
                    // A draft layer may use a different GGUF quantization from
                    // the target. Leave unsupported layers on the ordinary NUMA
                    // path without disabling the target's dynamic GPU offload.
                    if (experts[2]->dataType == DataType::DATA_GGUF_FORMAT &&
                        (!experts[3] || experts[2]->dims.size() != 2 || experts[3]->dims.size() != 2 ||
                         !FastllmCudaMoeGlm5GGUFCacheSupported(experts[2]->ggmlType, experts[3]->ggmlType,
                             experts[2]->dims[1], experts[3]->dims[1]))) continue;
                    cache.Add(experts, device, layer < block_cnt ? cacheDevices[layer] : -1);
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
