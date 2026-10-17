#include "models/naive_n05_flash.h"
#include "naive_n05_tp.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "executor.h"
#include "utils.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <set>
#ifdef USE_CUDA
#include "devices/cuda/cudaworkspace.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {

#ifdef USE_CUDA
void FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
bool FastllmCudaMergeMOEUsedGraphUnsafeFallback();
#endif

namespace {
#ifdef USE_CUDA
// KV descriptors belong to individual requests. Partition by request, rather
// than device, so workers write disjoint objects and contiguous result ranges.
// Keep small batches on the caller; their scan is cheaper than a worker handoff.
template <class Task>
void ForSequenceRanges(PersistentWorkerGroup &workers, const std::vector<int> &devices,
        int sequences, bool parallel, const Task &task) {
    if (!parallel || sequences < 32) {
        task(0, sequences);
        return;
    }
    const int count = devices.size();
    std::vector<std::exception_ptr> errors(count);
    workers.Run(devices, [&](int rank) {
        task(sequences * rank / count, sequences * (rank + 1) / count);
    }, errors);
    for (auto error : errors) if (error) std::rethrow_exception(error);
}
#endif

// Expanded BF16/FP8 KV caches keep physical strides/capacity when logical length
// changes. Other layouts retain Data::Resize and its layout invalidation rules.
bool CanResizeKVLength(const Data &cache, int length, int width) {
    return length >= 0 && FastllmNaiveKVType(cache.dataType) && !cache.cudaNativeNvfp4Layout &&
        cache.tpLayout == TP_LAYOUT_NONE && cache.unitSize == (cache.dataType == FP8_E4M3 ? 1 : 2) && cache.unitSizeDiv == 1 &&
        cache.dims.size() == 3 && cache.dims[0] == 1 && cache.dims[2] == width &&
        cache.expansionDims.size() == 3 && cache.expansionDims[0] == 1 &&
        cache.expansionDims[1] >= length && cache.expansionDims[2] == width;
}

void ResizeKVLength(Data &cache, int length, int width) {
    if (CanResizeKVLength(cache, length, width)) {
        cache.dims[1] = length;
    } else {
        cache.Resize({1, length, width});
    }
}
}

bool NaiveN05FlashModel::CanReuseTensorParallelCache(const ResponseContext *context) const {
#ifdef USE_CUDA
    // With a single live handle, removal happens after its last forward and
    // creation precedes its first forward. Never touch another request's KV or
    // the worker-owned decode state from these callbacks.
    return !isFree && tpPagedCaches.empty() && tpDevices.size() > 1 &&
        GetFastllmEnv().cudaGraph && !GetKVCacheInCPU() &&
        FastllmNaiveKVType(kvCacheDataType) && context->multimodalInput.empty() &&
        responseContextDict.dicts.size() == 1 &&
        responseContextDict.dicts.begin()->second == context;
#else
    return false;
#endif
}

void NaiveN05FlashModel::RestoreTensorParallelCache(ResponseContext *context) {
    if (tpIdleCache.empty()) return;
    // Release unused storage on every return path, before a new prefill can
    // allocate another set. A successful swap leaves only empty descriptors.
    std::vector<std::pair<Data, Data>> idle;
    idle.swap(tpIdleCache);
    if (!CanReuseTensorParallelCache(context) ||
        (int)context->pastKeyValues.size() != block_cnt || (int)idle.size() != block_cnt) return;
    const int reserve = CacheReserveCapacity(context->generationConfig);
    if (reserve <= 0) return;
    const int64_t capacity = ((int64_t)reserve + 127) / 128 * 128;
    for (int layer = 0; layer < block_cnt; ++layer) {
        for (Data *root : {&context->pastKeyValues[layer].first,
                           &context->pastKeyValues[layer].second})
            if (root->multiDeviceData || root->cudaData ||
                !root->dims.empty() || root->cpuData)
                return;
        if (slidingLayers[layer]) continue;
        for (Data *root : {&idle[layer].first, &idle[layer].second})
            for (int device : tpDevices)
                if (root->multiDeviceDatas.at(device)->expansionDims[1] != capacity)
                    return;
    }
    context->pastKeyValues.swap(idle);
    // The scheduler identifies a pending prompt by EMPTY root capacity.
    // Keeping the previous root capacity would consume an active slot and
    // could prevent this fresh request from ever being admitted.
    for (auto &layer : context->pastKeyValues) {
        for (Data *root : {&layer.first, &layer.second}) {
            root->dims.clear();
            root->strides.clear();
            root->expansionDims.clear();
            for (auto &item : root->multiDeviceDatas) {
                Data &local = *item.second;
                local.Resize({1, 0, local.dims[2]});
            }
        }
    }
}

void NaiveN05FlashModel::RecycleTensorParallelCache(ResponseContext *context) {
    if (!CanReuseTensorParallelCache(context) ||
        (!context->isEnding && !context->isAbort) ||
        context->error != ResponseContextErrorNone ||
        CacheReserveCapacity(context->generationConfig) <= 0 ||
        (int)context->pastKeyValues.size() != block_cnt) return;
    for (const auto &layer : context->pastKeyValues) {
        for (const Data *root : {&layer.first, &layer.second}) {
            if (!root->multiDeviceData || root->isPagedKVCache || root->isFake ||
                root->cudaData || root->cpuData || root->dataDeviceIds != tpDevices ||
                root->multiDeviceDatas.size() != tpDevices.size()) return;
            for (int device : tpDevices) {
                auto it = root->multiDeviceDatas.find(device);
                if (it == root->multiDeviceDatas.end() || !it->second) return;
                const Data &local = *it->second;
                if (local.isFake || local.cudaDataBorrowed || local.isPagedKVCache ||
                    local.dataDevice != DataDevice::CUDA || local.dataType != kvCacheDataType ||
                    !local.cudaData || local.dims.size() != 3 || local.dims[1] <= 0 ||
                    local.expansionDims.size() != 3 || local.expansionDims[1] < local.dims[1]) return;
            }
        }
    }
    // Bound retention to one completed request's existing allocation set;
    // ownership is transferred, never copied or shared with another handle.
    tpIdleCache.clear();
    tpIdleCache.swap(context->pastKeyValues);
}

bool NaiveN05FlashModel::InitTensorParallel() {
    const char *epValue = std::getenv("FASTLLM_EP");
    const std::string ep = epValue ? epValue : "";
    AssertInFastLLM(ep.empty() || ep == "1" || ep == "8",
                   "MiMo --ep supports 1 (MoE TP) or 8 (MegaMoE EP8).");
    const bool explicitEp8 = ep == "8";
    tpEp8Mega = false;
#ifdef USE_CUDA
    // Use the existing --tp setting; ordinary lists of layer devices retain
    // the serial path. All ranks in this implementation have equal widths.
    const char *value = std::getenv("FASTLLM_TP");
    std::string spec = value ? value : "";
    if (spec.empty() || spec == "false" || spec == "off" || spec == "none") {
        AssertInFastLLM(!explicitEp8, "MiMo --ep 8 requires --tp 8.");
        return false;
    }
    tpDevices.clear();
    std::map<int, int> ratios;
    if (spec == "auto" || spec == "true" || spec == "on" ||
        std::all_of(spec.begin(), spec.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        int count = spec == "auto" || spec == "true" || spec == "on"
            ? FastllmCudaGetDeviceCount() : std::max(1, std::stoi(spec));
        for (int i = 0; i < count; ++i) tpDevices.push_back(i);
    } else {
        if (spec.find(':') == std::string::npos) spec = "cuda:" + spec;
        tpDevices = ParseDeviceIds(spec, StartWith(spec, "multicuda") ? "multicuda" : "cuda", ratios);
    }
    AssertInFastLLM(!tpDevices.empty(), "Naive TP: invalid device list.");
    std::set<int> unique;
    for (int device : tpDevices) {
        AssertInFastLLM(device >= 0 && device < FastllmCudaGetDeviceCount() && unique.insert(device).second,
                        "Naive TP: device is unavailable or repeated.");
        AssertInFastLLM(ratios.empty() || ratios[device] == ratios[tpDevices[0]],
                        "Naive TP requires equal device ratios.");
    }
    AssertInFastLLM(!explicitEp8 || tpDevices.size() == 8,
                   "MiMo --ep 8 requires exactly eight TP devices.");
    if (tpDevices.size() <= 1) { tpDevices.clear(); return false; }
    const int ranks = tpDevices.size();
    for (auto cfg : {full, sliding}) {
        AssertInFastLLM(cfg.heads % ranks == 0 &&
            (cfg.kvHeads % ranks == 0 || ranks % cfg.kvHeads == 0),
            "Naive TP requires complete Q heads and divisible or replicated KV groups.");
    }
    // Resolve once, before weight registration/loading.
    tpEp8Mega = explicitEp8 && FastllmCudaMimoEp8MegaSupported(tpDevices);
    AssertInFastLLM(!explicitEp8 || tpEp8Mega,
        "MiMo --ep 8 requires eight H20 SM90/78-SM GPUs and a FastLLM build with built-in SM90 MegaMoE.");
    AssertInFastLLM(!tpEp8Mega || (embed_dim == 4096 && num_experts == 256 &&
                   num_experts_per_tok == 8 && block_cnt <= 48),
                   "MiMo EP8 requires hidden=4096, experts=256, topk=8 and at most 48 layers.");
    return true;
#else
    AssertInFastLLM(!explicitEp8, "MiMo --ep 8 requires a CUDA build with SM90 MegaMoE.");
    return false;
#endif
}

void NaiveN05FlashModel::PrepareTensorParallel() {
#ifdef USE_CUDA
    if (tpPrepared) return;
    const int ranks = tpDevices.size();
    AssertInFastLLM(ranks > 1 && !GetKVCacheInCPU(), "Naive TP requires CUDA KV cache.");
    if (tpEp8Mega) {
        for (int layer = 0; layer < block_cnt; ++layer) if (moeLayers[layer]) {
            for (int e = 0; e < num_experts; ++e) {
                std::string base = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
                for (auto part : {std::make_pair("gateup_proj.weight", std::vector<int>{4096,4096}),
                                  std::make_pair("down_proj.weight", std::vector<int>{4096,2048})}) {
                    auto it = weight.weight.find(base + part.first);
                    AssertInFastLLM(it != weight.weight.end() && it->second.dataType == FP8_E4M3 &&
                        it->second.dims == part.second && it->second.blockM == 128 && it->second.blockK == 128,
                        "MiMo EP8 MegaMoE requires original complete FP8 expert weights.");
                }
            }
        }
    }
    AssertInFastLLM(FastllmInitNccl(tpDevices), "Naive TP: NCCL initialization failed.");
    auto rangeScheme = [&](int total, int copies = 1) {
        AssertInFastLLM(total % ranks == 0, "Naive TP: projection width must divide the rank count.");
        DivisionScheme scheme;
        for (int r = 0; r < ranks; ++r)
            for (int c = 0; c < copies; ++c)
                scheme[tpDevices[r]].push_back({c * total + r * (total / ranks),
                                                c * total + (r + 1) * (total / ranks)});
        return scheme;
    };
    auto split = [&](const std::string &name, DivisionScheme scheme, int axis,
                     const std::string &biasName = "") {
        auto found = weight.weight.find(name);
        AssertInFastLLM(found != weight.weight.end() && found->second.dims.size() == 2,
                        "Naive TP: missing or invalid matrix " + name);
        Data emptyBias;
        weight[name].tpLinearType = axis == 0 ? TP_LINEAR_ROW : TP_LINEAR_COLUMN;
        Data &bias = biasName.empty() ? emptyBias : weight[biasName];
        if (!bias.dims.empty()) ToDataType(bias, FLOAT32);
        AssertInFastLLM(SplitMultiCudaWeight(weight[name], bias, tpDevices, scheme, axis, true),
                        "Naive TP: cannot split " + name);
    };
    auto replicate = [&](const std::string &name) {
        auto it = weight.weight.find(name);
        if (it != weight.weight.end() && !it->second.dims.empty())
            PrepareMultiCudaReplicatedData(it->second, tpDevices, true);
    };
    if (GetCudaEmbedding() && !GetLowMemMode()) replicate("model.embed_tokens.weight");
    replicate("model.norm.weight");
    for (int layer = 0; layer < block_cnt; ++layer) {
        std::string prefix = "model.layers." + std::to_string(layer);
        std::string ap = prefix + ".self_attn.";
        const auto &cfg = slidingLayers[layer] ? sliding : full;
        replicate(prefix + ".input_layernorm.weight");
        replicate(prefix + ".post_attention_layernorm.weight");
        const auto merged = weight.weight.find(ap + "mergeqkv.weight");
        const bool mergedQkv = merged != weight.weight.end() && !merged->second.dims.empty();
        DivisionScheme qkvScheme = rangeScheme(cfg.heads * cfg.headDim);
        if (!mergedQkv)
            split(ap + "q_proj.weight", qkvScheme, 0, ap + "q_proj.bias");
        int offset = cfg.heads * cfg.headDim;
        for (auto item : {std::make_pair("k", cfg.headDim), std::make_pair("v", cfg.valueDim)}) {
            DivisionScheme scheme;
            for (int r = 0; r < ranks; ++r) {
                // When TP exceeds KV heads, adjacent ranks share one KV head.
                int begin = r * cfg.kvHeads / ranks;
                int count = std::max(1, cfg.kvHeads / ranks);
                scheme[tpDevices[r]] = {{begin * item.second, (begin + count) * item.second}};
                qkvScheme[tpDevices[r]].push_back(
                    {offset + begin * item.second, offset + (begin + count) * item.second});
            }
            if (!mergedQkv)
                split(ap + item.first + "_proj.weight", scheme, 0, ap + item.first + "_proj.bias");
            offset += cfg.kvHeads * item.second;
        }
        // Explicit Q/K/V ranges also support unequal K/V dimensions and KV
        // replication when TP exceeds KV heads; the generic QKV pack assumes
        // equal head dimensions.
        if (mergedQkv)
            split(ap + "mergeqkv.weight", qkvScheme, 0, ap + "mergeqkv.bias");
        split(ap + "o_proj.weight", rangeScheme(cfg.heads * cfg.valueDim), 1);
        auto sinkIt = weight.weight.find(ap + "attention_sink_bias");
        if (sinkIt != weight.weight.end() && !sinkIt->second.dims.empty()) {
            ToDataType(sinkIt->second, FLOAT32);
            AssertInFastLLM(SplitMultiCudaWeight1D(sinkIt->second, tpDevices, rangeScheme(cfg.heads)),
                            "Naive TP: cannot split attention sinks.");
        }
        if (!slidingLayers[layer]) {
            for (const char *name : {"wq.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias"})
                replicate(ap + "indexer." + name);
#ifndef USE_ROCM
            const int previousDevice = FastllmCudaGetDevice();
            for (int device : tpDevices) {
                FastllmCudaSetDevice(device);
                FastllmCudaPackIndexerWeights(*weight[ap + "indexer.wq.weight"].multiDeviceDatas.at(device),
                    *weight[ap + "indexer.wk.weight"].multiDeviceDatas.at(device),
                    *weight[ap + "indexer.weights_proj.weight"].multiDeviceDatas.at(device));
            }
            FastllmCudaSetDevice(previousDevice);
#endif
        }
        if (moeLayers[layer]) {
            replicate(prefix + ".mlp.gate.weight");
            replicate(prefix + ".mlp.gate.e_score_correction_bias");
            for (int e = 0; e < num_experts; ++e) {
                std::string base = prefix + ".mlp.experts." + std::to_string(e) + ".";
                int mid = weight[base + "down_proj.weight"].dims[1];
                if (tpEp8Mega) {
                    const int owner = tpDevices[e / (num_experts / ranks)];
                    for (const char *name : {"gateup_proj.weight", "down_proj.weight"})
                        AssertInFastLLM(PlaceMultiCudaWeightOnDevice(weight[base + name], tpDevices, owner),
                                       "MiMo EP8 expert placement failed.");
                } else {
                    split(base + "gateup_proj.weight", rangeScheme(mid, 2), 0);
                    split(base + "down_proj.weight", rangeScheme(mid), 1);
                }
            }
        } else {
            int mid = weight[prefix + ".mlp.down_proj.weight"].dims[1];
            auto scheme = rangeScheme(mid);
            split(prefix + ".mlp.gate_proj.weight", scheme, 0);
            split(prefix + ".mlp.up_proj.weight", scheme, 0);
            split(prefix + ".mlp.down_proj.weight", scheme, 1);
        }
    }
    // Logits are gathered on the host for the existing sampler, without a
    // full-vocabulary all-reduce or GPU broadcast.
    int vocab = weight["lm_head.weight"].dims[0];
    DivisionScheme vocabScheme;
    tpVocabRanges.clear();
    for (int r = 0; r < ranks; ++r) {
        tpVocabRanges.push_back({(int)((int64_t)vocab * r / ranks), (int)((int64_t)vocab * (r + 1) / ranks)});
        vocabScheme[tpDevices[r]] = {tpVocabRanges.back()};
    }
    split("lm_head.weight", vocabScheme, 0);
    tpMoeWeights.resize(ranks); tpMoeBiases.resize(ranks);
    for (int r = 0; r < ranks; ++r) {
        tpMoeWeights[r].resize(block_cnt); tpMoeBiases[r].resize(block_cnt);
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (!moeLayers[layer]) continue;
            auto &weights = tpMoeWeights[r][layer];
            weights = {nullptr, nullptr};
            for (int e = 0; e < num_experts; ++e) {
                std::string base = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
                for (const char *name : {"gateup_proj.weight", "down_proj.weight"}) {
                    auto &locals = weight[base + name].multiDeviceDatas;
                    auto it = locals.find(tpDevices[r]);
                    weights.push_back(it == locals.end() ? nullptr : it->second);
                    AssertInFastLLM(tpEp8Mega || weights.back() != nullptr, "Missing TP expert shard.");
                }
            }
            tpMoeBiases[r][layer].resize(weights.size(), nullptr);
        }
    }
    if (tpEp8Mega) tpEp8MegaState = FastllmCudaCreateMimoEp8Mega(tpDevices, tpMoeWeights);
    tpSwaRingCache = GetFastllmEnv().cudaGraph && FastllmNaiveKVType(kvCacheDataType) &&
        window == 128 && sliding.headDim == 192 && sliding.valueDim == 128 &&
        sliding.heads / ranks < 64;
    tpPrepared = true;
#endif
}

void NaiveN05FlashModel::PrepareTensorParallelServing() {
#ifdef USE_CUDA
    if (tpServingState || !GetFastllmEnv().cudaGraph || !tpPrepared || draftEnabled ||
        tpDevices.size() <= 1 || !tpSwaRingCache ||
        (!tpEp8MegaState && std::any_of(moeLayers.begin(), moeLayers.end(), [](int x) { return x != 0; })) ||
        dataType != BFLOAT16 || !FastllmNaiveKVType(kvCacheDataType) || !GetCudaEmbedding() ||
        GetLowMemMode() || full.headDim > 256 || (full.valueDim != 128 && full.valueDim != 256) ||
        indexHeads != 16 || indexDim != 128 || indexTopK != 2048) return;
    // Preparing is explicit and ends before serving. Unsupported shapes never
    // trigger another capture, even if a previous request used the same batch.
    auto serving = std::make_shared<TPServingState>();
    serving->preparing = true;
    serving->moeType = moeAtype;
    serving->kvType = kvCacheDataType;
    serving->generation = FastllmGetNcclGeneration();
    const int maxRows = std::min(maxBatch > 0 ? maxBatch : 256, 256);
    const int64_t tokenBudget = tokensLimit > 0 ? tokensLimit : (int64_t)maxRows * 16384;
    serving->batchBuckets = DecodeGraphBatchBuckets(maxRows);
    std::vector<int> batches(serving->batchBuckets.rbegin(), serving->batchBuckets.rend());
    int64_t scoreElements = 0;
    // The pool bounds the sum of live lengths, not each request's length by
    // the average. A ragged batch (including padded rows) may contain a long
    // request at any batch size, so every bucket needs the same length range.
    const int limit = (int)std::min<int64_t>(max_positions, tokenBudget);
    for (int rows : batches) {
        // Round the last bucket up; device live lengths bound KV reads.
        std::vector<int> capacities{256, 2047};
        for (int64_t capacity = 4096; capacity / 2 < limit; capacity *= 2) {
            capacities.push_back((int)std::min<int64_t>(capacity, max_positions));
            if (capacity >= max_positions) break;
        }
        for (int capacity : capacities) {
            auto state = std::make_shared<TPDecodeState>();
            state->devices = tpDevices;
            state->rows = rows;
            state->capacity = capacity;
            state->ncclGeneration = serving->generation;
            state->selection.count = 1;
            state->selection.greedy = true;
            serving->graphs.emplace(std::make_pair(rows, capacity), state);
            if (capacity > 2048) scoreElements = std::max(scoreElements, (int64_t)rows * capacity);
        }
    }
    // All graph variants alias one scratch sized for the largest shape;
    // capturing more length buckets does not duplicate KV or weights.
    for (int rank = 0; rank < (int)tpDevices.size(); ++rank) {
        FastllmCudaSetDevice(tpDevices[rank]);
        if (!FastllmCudaNaiveDecodeGraphSupported()) return;
        auto storage = std::make_shared<TPDecodeState::Storage>();
        storage->maxRows = maxRows;
        storage->hostCache.resize((size_t)block_cnt * maxRows * 4);
        auto allocate = [&](Data &data, DataType type, int elements) {
            data.dataType = type;
            data.Resize({elements});
            data.ToDevice(DataDevice::CUDA, {tpDevices[rank]}, false);
            data.Allocate(false);
        };
        const int tableElements = storage->hostCache.size() * sizeof(void *) / sizeof(int);
        if (!tpKVPool.empty()) {
            auto &table = storage->cacheTable;
            table.dataType = INT32;
            table.Resize({tableElements});
            table.dataDevice = DataDevice::CUDA;
            table.dataDeviceIds = {tpDevices[rank]};
            table.directMemory = true;
            table.cudaData = AllocateCudaWorkspace(tpKVPool.at(tpDevices[rank]), table.GetBytes());
            if (!table.cudaData) throw CudaWorkspaceExhausted("Naive graph cache table pool exhausted.");
        } else allocate(storage->cacheTable, INT32, tableElements);
        allocate(storage->indexScores, INT32, (int)scoreElements);
        storage->buffers.cacheTable = (const void *const *)storage->cacheTable.cudaData;
        storage->buffers.tableRows = maxRows;
        if (kvCacheDataType == BFLOAT16 && FastllmCudaNaiveDSAMapsSupported() && full.headDim == 192 && full.valueDim == 128 &&
            std::max(1, full.kvHeads / (int)tpDevices.size()) == 1) {
            storage->hostMaps.resize((size_t)block_cnt * maxRows * FastllmNaiveDSAMapBytes);
            allocate(storage->cacheMaps, INT32, storage->hostMaps.size() / sizeof(int));
            storage->buffers.cacheMaps = (const unsigned char *)storage->cacheMaps.cudaData;
        }
        for (int i = 0; i < maxRows; ++i) storage->buffers.sequences.emplace_back(new TargetWorkspace());
        for (auto &entry : serving->graphs) {
            auto r = std::make_unique<TPDecodeState::Rank>(storage);
            r->features.collectHidden = false;
            entry.second->ranks.push_back(std::move(r));
        }
    }
    // Private warmup backing: only the first few live rows are read. Descriptors
    // advertise graph capacities, while every kernel uses device live lengths.
    // Reuse it across layers; none of these token contents reaches a request.
    const int ranks = tpDevices.size();
    int maxWidth = 0;
    for (const auto &cfg : {full, sliding})
        maxWidth = std::max(maxWidth, std::max(1, cfg.kvHeads / ranks) *
                           std::max(cfg.headDim, cfg.valueDim) + IndexerCacheWidth(kvCacheDataType));
    std::vector<Data> backing(ranks), pageTables(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        FastllmCudaSetDevice(tpDevices[rank]);
        backing[rank].dataType = kvCacheDataType;
        backing[rank].Resize({maxRows * 2, 256, maxWidth});
        backing[rank].ToDevice(DataDevice::CUDA, {tpDevices[rank]}, false);
        backing[rank].Allocate(true);
        if (!tpPagedCaches.empty()) {
            pageTables[rank].dataType = INT32;
            pageTables[rank].Resize({(max_positions + GetPageLen() - 1) / GetPageLen()});
            pageTables[rank].ToDevice(DataDevice::CUDA, {tpDevices[rank]}, false);
            pageTables[rank].Allocate(true);
        }
    }
    std::vector<std::pair<Data, Data>> roots;
    roots.reserve((size_t)maxRows * block_cnt);
    for (int row = 0; row < maxRows; ++row)
    for (int layer = 0; layer < block_cnt; ++layer) {
        roots.emplace_back(Data(kvCacheDataType), Data(kvCacheDataType));
        const auto &cfg = slidingLayers[layer] ? sliding : full;
        for (int part = 0; part < 2; ++part) {
            Data &root = part ? roots.back().second : roots.back().first;
            const int width = cfg.kvHeads * (part ? cfg.valueDim : cfg.headDim) +
                              (!part && !slidingLayers[layer] ? IndexerCacheWidth(kvCacheDataType) : 0);
            root.kvCacheFp8Indexer = !part && !slidingLayers[layer] && fp8IndexerCache;
            root.Resize({1, 2, width});
            root.expansionDims = {1, max_positions, width};
            root.multiDeviceData = true;
            root.dataDevice = DataDevice::CUDA;
            root.dataDeviceIds = tpDevices;
            for (int rank = 0; rank < ranks; ++rank) {
                const int localWidth = std::max(1, cfg.kvHeads / ranks) * (part ? cfg.valueDim : cfg.headDim) +
                                       (!part && !slidingLayers[layer] ? IndexerCacheWidth(kvCacheDataType) : 0);
                auto *local = new Data(kvCacheDataType);
                local->FakeFrom(backing[rank], ((size_t)row * 2 + part) * 256 * maxWidth * backing[rank].unitSize);
                local->Resize({1, 2, localWidth});
                local->expansionDims = {1, max_positions, localWidth};
                if (!slidingLayers[layer] && !tpPagedCaches.empty()) {
                    local->FakeFrom(pageTables[rank], 0);
                    local->dataType = kvCacheDataType;
                    local->UpdateUnitSize();
                    local->isPagedKVCache = true;
                    local->pageLen = tpPagedCaches[rank][layer * 2 + part]->pageLen;
                    local->pagedKVCacheData = tpPagedCaches[rank][layer * 2 + part].get();
                }
                local->kvCacheFp8Indexer = root.kvCacheFp8Indexer;
                root.multiDeviceDatas[tpDevices[rank]] = local;
            }
        }
    }
    tpServingState = serving;
    const bool saveHistory = saveHistoryChat;
    saveHistoryChat = false;
    struct RestoreHistory { bool &value; bool saved; ~RestoreHistory() { value = saved; } } restoreHistory{saveHistoryChat, saveHistory};
    GenerationConfig config;
    config.top_k = 1;
    config.repeat_penalty = 1.0f;
    std::vector<std::pair<Data, Data>> empty;
    // Warm all shapes before capturing any: growing a shared buffer after the
    // first capture would invalidate addresses retained by earlier graphs.
    for (int pass = 0; pass < 2; ++pass) {
        serving->capturing = pass != 0;
        for (int rows : batches)
        for (auto &entry : serving->graphs) {
            if (entry.first.first != rows) continue;
            auto &state = *entry.second;
            serving->prepareCapacity = state.capacity;
            for (auto &pair : roots) for (Data *root : {&pair.first, &pair.second}) {
                root->dims[1] = 2;
                for (auto &local : root->multiDeviceDatas) local.second->dims[1] = 2;
            }
            std::vector<std::pair<Data *, Data *>> pointers;
            for (int i = 0; i < rows * block_cnt; ++i)
                pointers.emplace_back(&roots[i].first, &roots[i].second);
            std::vector<int> lengths(rows, 1);
            std::vector<GenerationConfig> configs(rows, config);
            TargetBatch batch{pointers, configs, lengths, block_cnt};
            Data ids(FLOAT32, {1, rows}, std::vector<float>(rows, 1.0f));
            Data positions(FLOAT32, {1, rows}, std::vector<float>(rows, 2.0f));
            auto selection = SelectLogits(config);
            ForwardTensorParallel(ids, positions, empty, config, nullptr, &selection, &batch);
        }
    }
    serving->preparing = false;
    serving->sealed = true;
    serving->cacheEpoch = UINT64_MAX;
    serving->requests.clear();
    serving->cacheLengths.clear();
    serving->pageLengths.clear();
    serving->expectedLengths.clear();
    serving->capacityLimits.clear();
    size_t captured = 0;
    for (auto &entry : serving->graphs) {
        auto &state = *entry.second;
        state.active = false;
        if (state.captured && !state.disabled) ++captured;
    }
    printf("[Naive CUDA Graph] prepared %zu/%zu shapes, batch <= %d, shared Indexer scratch %.2f MiB/GPU. Serving only replays prepared shapes.\n",
        captured, serving->graphs.size(), maxRows, scoreElements * 4 / 1048576.0);
    fflush(stdout);
#endif
}

bool NaiveN05FlashModel::CanReuseTensorParallelServingCache(const TargetBatch *batch) const {
#ifdef USE_CUDA
    // Only the owned scheduler promises that descriptors cannot be mutated
    // behind this plan. Direct Forward callers always validate every tensor.
    if (!tpServingState || tpServingState->preparing || !batch || !batch->Decode() ||
        !mainLoop || mainLoop->get_id() != std::this_thread::get_id()) return false;
    const auto &s = *tpServingState;
    if (s.cacheEpoch != tpCacheEpoch.load(std::memory_order_relaxed) ||
        s.requests.size() != (size_t)batch->Size()) return false;
    for (int row = 0; row < batch->Size(); ++row) {
        const Data &root = batch->Key(row, 0);
        if (&root != s.requests[row] || root.dims.size() != 3 ||
            root.dims[1] != s.expectedLengths[row] || root.dims[1] >= s.capacityLimits[row]) return false;
    }
    return true;
#else
    return false;
#endif
}

#ifdef USE_CUDA
bool NaiveN05FlashModel::PrepareTensorParallelServingDecode(const Data &inputIds,
        const LogitsSelection &selection, std::vector<int *> &cacheLengths,
        const TargetBatch *batch, bool collectHidden) {
    auto &serving = *tpServingState;
    if (dataType != BFLOAT16 || kvCacheDataType != serving.kvType || moeAtype != serving.moeType ||
        !GetCudaEmbedding() || GetLowMemMode() || serving.generation != FastllmGetNcclGeneration()) {
        // These changes may replace captured weights/communication resources.
        // Do not revive old graphs if the setting is subsequently switched back.
        serving.sealed = false;
        return false;
    }
    if (!batch || !batch->Decode() || !GetFastllmEnv().cudaGraph || isIntermediateChunkedPrefill ||
        inputIds.dataType != FLOAT32 || inputIds.dims != std::vector<int>({1, batch->Size()}) ||
        collectHidden || (selection.count && (!selection.greedy || selection.count != 1))) return false;
    const int rows = batch->Size(), ranks = tpDevices.size();
    int capacity = serving.prepareCapacity;
    if (!serving.preparing) {
        int region = -1, longest = 0;
        for (int row = 0; row < rows; ++row) {
            const Data &key = batch->Key(row, 0);
            if (key.dims.size() != 3) return false;
            const int length = key.dims[1] + 1;
            if (length <= 1 || length > max_positions) return false;
            const int current = length <= 256 ? 0 : length <= 2048 ? 1 : 2;
            if (region >= 0 && region != current) return false;
            region = current;
            longest = std::max(longest, length);
        }
        capacity = region == 0 ? 256 : region == 1 ? 2047 : 4096;
        // 2048 uses the ordinary eager boundary path, preserving the previous
        // reduction/selection behavior exactly.
        if (longest == 2048) return false;
        if (region == 2) {
            while (capacity < longest && capacity < max_positions)
                capacity = (int)std::min<int64_t>((int64_t)capacity * 2, max_positions);
        }
    }
    const auto shape = DecodeGraphBatch::Select(rows, serving.batchBuckets);
    if (!shape) return false;
    auto found = serving.graphs.find({shape.graphRows, capacity});
    if (found == serving.graphs.end() || found->second->disabled ||
        (!serving.preparing && (!serving.sealed || !found->second->captured))) return false;
    auto state = found->second;
    const size_t slots = (size_t)rows * block_cnt * ranks * 2;
    const bool reuse = CanReuseTensorParallelServingCache(batch);
    const uint64_t epoch = tpCacheEpoch.load(std::memory_order_relaxed);
    if (reuse) cacheLengths = serving.cacheLengths;
    else {
        cacheLengths.resize(slots + (size_t)rows * block_cnt * 2);
        std::vector<void *> pointers(slots), pageTables(slots);
        std::vector<int *> pageLengths(slots);
        std::vector<int> capacities(slots), valid(rows, 1), limits(rows, INT_MAX);
        ForSequenceRanges(tpWorkers, tpDevices, rows, true, [&](int begin, int end) {
            for (int row = begin; row < end; ++row) {
                const int next = batch->Key(row, 0).dims[1] + 1;
                auto validate = [&]() {
                    for (int layer = 0; layer < block_cnt; ++layer) {
                        const auto &cfg = slidingLayers[layer] ? sliding : full;
                        const int required = slidingLayers[layer] ? std::min(next, window) : next;
                        for (int part = 0; part < 2; ++part) {
                            Data &root = part ? batch->Value(row, layer) : batch->Key(row, layer);
                            const int width = cfg.kvHeads * (part ? cfg.valueDim : cfg.headDim) +
                                (!part && !slidingLayers[layer] ? IndexerCacheWidth(kvCacheDataType) : 0);
                            if (!root.multiDeviceData || root.dataDevice != DataDevice::CUDA ||
                                !CanResizeKVLength(root, required, width)) return false;
                            auto it = root.multiDeviceDatas.begin();
                            for (int rank = 0; rank < ranks; ++rank) {
                                if (it == root.multiDeviceDatas.end() || it->first != tpDevices[rank])
                                    it = root.multiDeviceDatas.find(tpDevices[rank]);
                                if (it == root.multiDeviceDatas.end() || !it->second) return false;
                                Data &local = *it->second;
                                const int localWidth = std::max(1, cfg.kvHeads / ranks) *
                                    (part ? cfg.valueDim : cfg.headDim) + (!part && !slidingLayers[layer] ? IndexerCacheWidth(kvCacheDataType) : 0);
                                if (!local.cudaData || local.dataDevice != DataDevice::CUDA ||
                                    (uintptr_t)local.cudaData % 16 != 0 || local.strides.size() != 3 ||
                                    local.strides[1] != localWidth || local.strides[2] != 1 ||
                                    !CanResizeKVLength(local, required, localWidth)) return false;
                                const size_t slot = ((size_t)row * block_cnt + layer) * ranks * 2 + rank * 2 + part;
                                pointers[slot] = local.isPagedKVCache ? local.pagedKVCacheData->cudaData : local.cudaData;
                                pageTables[slot] = local.isPagedKVCache ? local.cudaData : nullptr;
                                pageLengths[slot] = local.isPagedKVCache ? &local.lastPageLen : nullptr;
                                capacities[slot] = local.isPagedKVCache
                                    ? local.pagedKVCacheData->maxPages * local.pageLen : local.expansionDims[1];
                                const int available = local.isPagedKVCache && !serving.preparing
                                    ? (int)local.pageIndex.size() * local.pageLen : local.expansionDims[1];
                                if (available < required) return false;
                                if (!slidingLayers[layer] || available < window)
                                    limits[row] = std::min(limits[row], available);
                                cacheLengths[slot] = &local.dims[1];
                                ++it;
                            }
                            cacheLengths[slots + ((size_t)row * block_cnt + layer) * 2 + part] = &root.dims[1];
                        }
                    }
                    return true;
                };
                valid[row] = validate();
            }
        });
        if (std::find(valid.begin(), valid.end(), 0) != valid.end()) { cacheLengths.clear(); return false; }
        // The table belongs to the shared workspace, not an individual graph. It
        // must be rebound even when returning to a previously used graph variant.
        for (int rank = 0; rank < ranks; ++rank) {
            auto &storage = *state->ranks[rank]->storage;
            // Never retain bindings to requests that left the batch. Inactive
            // device lengths prevent kernels from dereferencing these slots.
            for (int layer = 0; layer < block_cnt; ++layer)
            for (int row = rows; row < storage.maxRows; ++row)
            for (int part = 0; part < 4; ++part) {
                auto &pointer = storage.hostCache[((size_t)layer * storage.maxRows + row) * 4 + part];
                if (pointer) { pointer = nullptr; storage.cacheDirty = true; }
            }
            for (int layer = 0; layer < block_cnt; ++layer)
            for (int row = 0; row < rows; ++row)
            for (int part = 0; part < 2; ++part) {
                const size_t src = ((size_t)row * block_cnt + layer) * ranks * 2 + rank * 2 + part;
                const size_t dst = ((size_t)layer * storage.maxRows + row) * 4 + part;
                if (storage.hostCache[dst] != pointers[src]) {
                    storage.hostCache[dst] = pointers[src];
                    storage.cacheDirty = true;
                }
                if (storage.hostCache[dst + 2] != pageTables[src]) {
                    storage.hostCache[dst + 2] = pageTables[src];
                    storage.cacheDirty = true;
                }
            }
            // Paged maps describe the immutable physical pool, independent
            // of request order, page lists or live lengths.
            if (storage.hostMaps.empty() || (!tpPagedCaches.empty() && !serving.preparing)) continue;
            for (int layer = 0; layer < block_cnt; ++layer) {
                if (slidingLayers[layer]) continue;
                for (int row = 0; row < rows; ++row) {
                    const size_t slot = ((size_t)row * block_cnt + layer) * ranks * 2 + rank * 2;
                    const int stride = std::max(1, full.kvHeads / ranks) * full.headDim + IndexerCacheWidth(kvCacheDataType);
                    if (!FastllmCudaNaiveEncodeDSAMaps(storage.hostMaps.data() +
                            ((size_t)layer * storage.maxRows + row) * FastllmNaiveDSAMapBytes,
                            pointers[slot], pointers[slot + 1],
                            (pageTables[slot] ? std::min(capacities[slot], capacities[slot + 1]) :
                             std::min(2048, std::min(capacities[slot], capacities[slot + 1]))), stride)) {
                        cacheLengths.clear(); return false;
                    }
                }
            }
            storage.mapsDirty = true;
        }
        serving.capacityLimits = std::move(limits);
        serving.cacheLengths = cacheLengths;
        serving.pageLengths = std::move(pageLengths);
        serving.cacheEpoch = epoch;
        serving.requests.resize(rows);
        for (int row = 0; row < rows; ++row) serving.requests[row] = &batch->Key(row, 0);
    }
    serving.expectedLengths.resize(rows);
    for (int row = 0; row < rows; ++row) serving.expectedLengths[row] = batch->Key(row, 0).dims[1] + 1;
    if (serving.preparing) for (int rank = 0; rank < ranks; ++rank) {
        auto &storage = *state->ranks[rank]->storage;
        auto &buf = storage.buffers;
        buf.capacity = capacity;
        if (capacity > 2048) {
            // Decode Indexer CP reuses the score arena after packing local
            // candidates; graph variants keep the same allocation/address.
            buf.decode.indexScores.FakeFrom(storage.indexScores, 0);
            buf.decode.indexScores.Resize({rows * capacity});
        }
        for (int row = 0; row < rows; ++row) {
            auto &part = *buf.sequences[row];
            part.capacity = capacity;
            if (capacity > 2048) {
                part.decode.indexScores.FakeFrom(storage.indexScores, (size_t)row * capacity * sizeof(int));
                part.decode.indexScores.Resize({capacity});
            }
        }
    }
    tpDecodeState = state;
    return state->active = true;
}
#endif

#ifdef USE_CUDA
void NaiveN05FlashModel::TPDecodeState::ClearGraphs() {
    for (size_t rank = 0; rank < ranks.size(); ++rank) {
        FastllmCudaSetDevice(devices[rank]);
        FastllmCudaGraphExecDestroy(ranks[rank]->exec);
        FastllmCudaGraphDestroy(ranks[rank]->graph);
        ranks[rank]->exec = ranks[rank]->graph = nullptr;
        ranks[rank]->ok = true;
    }
    FastllmCudaGraphMemoryPoolRelease(reservedPointers);
    reservedPointers.clear();
    warmed = captured = disabled = active = false;
}

NaiveN05FlashModel::TPDecodeState::~TPDecodeState() { ClearGraphs(); }

bool NaiveN05FlashModel::PrepareTensorParallelDecode(const Data &inputIds,
        std::vector<std::pair<Data, Data>> &kv, bool verifying, const LogitsSelection &selection,
        std::vector<int *> &cacheLengths, const TargetBatch *batch, bool collectHidden) {
    // Prefill has its own scratch. Keep decode and registered communication
    // buffers alive across requests, but only replay after validating all KV
    // addresses/capacities and the collective generation below.
    if (tpDecodeState) tpDecodeState->active = false;
    if (tpServingState && !verifying)
        return PrepareTensorParallelServingDecode(inputIds, selection, cacheLengths, batch, collectHidden);
    const int rows = batch ? batch->Size() : verifying ? 8 : 1;
    const int maxBatchRows = tpEp8MegaState ? 256 : 8;
    if (batch && (!batch->Decode() || rows > maxBatchRows)) return false;
    auto key = [&](int sequence, int layer) -> Data & { return batch ? batch->Key(sequence, layer) : kv[layer].first; };
    auto value = [&](int sequence, int layer) -> Data & { return batch ? batch->Value(sequence, layer) : kv[layer].second; };
    if (!GetFastllmEnv().cudaGraph || inputIds.dims != std::vector<int>({1, rows}) ||
        inputIds.dataType != FLOAT32 || isIntermediateChunkedPrefill ||
        !GetCudaEmbedding() || GetLowMemMode() || dataType != BFLOAT16 ||
        !FastllmNaiveKVType(kvCacheDataType) || (moeAtype != BFLOAT16 && moeAtype != FLOAT16) ||
        indexHeads != 16 || indexDim != 128 || indexTopK != 2048 ||
        window < 2 || window > 128 || key(0, 0).dims.size() != 3)
        return false;
    std::vector<int> regions, sequenceCapacities;
    const int sequences = batch ? batch->Size() : 1;
    for (int sequence = 0; sequence < sequences; ++sequence) {
        if (key(sequence, 0).dims.size() != 3) return false;
        const int firstLength = key(sequence, 0).dims[1] + 1;
        const int nextLength = firstLength + (batch ? 0 : rows - 1);
        // Keep each request on its own dense/sparse attention reduction tree.
        if (firstLength <= 1 || nextLength > max_positions ||
            (firstLength <= indexTopK && nextLength >= indexTopK) ||
            (verifying && firstLength <= 256 && nextLength > 256)) return false;
        int region = nextLength <= 256 ? 0 : nextLength < indexTopK ? 1 : 2;
        regions.push_back(region);
        sequenceCapacities.push_back(region == 0 ? 256 : region == 1 ? indexTopK - 1 :
            (int)std::min<int64_t>(max_positions, ((int64_t)nextLength + 4095) / 4096 * 4096));
    }
    const int ranks = tpDevices.size();
    // Retain the existing FlashInfer path on larger shards until a graph-safe
    // dynamic-length equivalent is available.
    if (sliding.heads / ranks >= 64 && sliding.heads / sliding.kvHeads == 8 &&
        window == 128 && sliding.headDim == 192 && sliding.valueDim == 128)
        return false;
    const size_t slots = (size_t)sequences * block_cnt * ranks * 2;
    // A captured graph only needs an equality check against its existing
    // descriptors. Compare in place instead of rebuilding two large arrays and
    // comparing them again on the caller. Still validate every live descriptor:
    // callers may replace/reallocate KV between forwards, including at the same
    // root address. Any mismatch falls through to the full preparation below.
    const uint64_t generation = FastllmGetNcclGeneration();
    if (tpDecodeState && tpDecodeState->captured && !tpDecodeState->disabled &&
        tpDecodeState->rows == rows && tpDecodeState->verifying == verifying &&
        tpDecodeState->collectHidden == collectHidden &&
        tpDecodeState->selection.greedy == selection.greedy &&
        tpDecodeState->selection.count == selection.count &&
        tpDecodeState->selection.invTemperature == selection.invTemperature &&
        tpDecodeState->sequenceRegions == regions && tpDecodeState->ncclGeneration == generation &&
        tpDecodeState->cachePointers.size() == slots && tpDecodeState->cacheCapacities.size() == slots) {
        const auto &state = *tpDecodeState;
        cacheLengths.resize(slots + (size_t)sequences * block_cnt * 2);
        std::vector<int> matches(sequences, 1);
        ForSequenceRanges(tpWorkers, tpDevices, sequences, true, [&](int begin, int end) {
            for (int sequence = begin; sequence < end; ++sequence) {
                const int firstLength = key(sequence, 0).dims[1] + 1;
                const int nextLength = firstLength + (batch ? 0 : rows - 1);
                int capacity = sequenceCapacities[sequence];
                auto check = [&]() {
                    for (int layer = 0; layer < block_cnt; ++layer) {
                        const auto &cfg = slidingLayers[layer] ? sliding : full;
                        const int required = slidingLayers[layer]
                            ? (verifying ? std::min(firstLength - 1, window - 1) + rows
                                         : std::min(nextLength, window)) : nextLength;
                        for (int part = 0; part < 2; ++part) {
                            Data &root = part ? value(sequence, layer) : key(sequence, layer);
                            const int width = part ? cfg.kvHeads * cfg.valueDim :
                                cfg.kvHeads * cfg.headDim + (slidingLayers[layer] ? 0 : IndexerCacheWidth(kvCacheDataType));
                            if (!root.multiDeviceData || root.dataDevice != DataDevice::CUDA ||
                                !CanResizeKVLength(root, required, width)) return false;
                            auto it = root.multiDeviceDatas.begin();
                            for (int rank = 0; rank < ranks; ++rank) {
                                // Device order is normally sorted. Retain arbitrary
                                // TP device order without repeating tree lookups.
                                if (it == root.multiDeviceDatas.end() || it->first != tpDevices[rank])
                                    it = root.multiDeviceDatas.find(tpDevices[rank]);
                                if (it == root.multiDeviceDatas.end() || !it->second) return false;
                                Data &local = *it->second;
                                const size_t slot = ((size_t)sequence * block_cnt + layer) * ranks * 2 + rank * 2 + part;
                                if (local.dims.size() != 3 || local.dataDevice != DataDevice::CUDA ||
                                    local.cudaData != state.cachePointers[slot] || !local.cudaData ||
                                    !CanResizeKVLength(local, required, local.dims[2]) ||
                                    local.expansionDims[1] != state.cacheCapacities[slot]) return false;
                                if (rank == 0 && root.expansionDims[1] != local.expansionDims[1]) return false;
                                cacheLengths[slot] = &local.dims[1];
                                if (!slidingLayers[layer]) capacity = std::min(capacity, local.expansionDims[1]);
                                ++it;
                            }
                            cacheLengths[slots + ((size_t)sequence * block_cnt + layer) * 2 + part] = &root.dims[1];
                        }
                    }
                    return capacity == state.sequenceCapacities[sequence];
                };
                matches[sequence] = check();
            }
        });
        if (std::find(matches.begin(), matches.end(), 0) == matches.end())
            return tpDecodeState->active = true;
    }
    std::vector<void *> pointers(slots);
    std::vector<int> capacities(slots), valid(sequences, 1);
    // Borrow length fields only for this forward. Replay never resizes the
    // descriptors; warmup/capture and nonstandard layouts use the normal commit.
    cacheLengths.resize(slots);
    std::vector<int> directLengths(sequences, 1);
    auto validateSequence = [&](int sequence) {
        size_t slot = (size_t)sequence * block_cnt * ranks * 2;
        const int firstLength = key(sequence, 0).dims[1] + 1;
        const int nextLength = firstLength + (batch ? 0 : rows - 1);
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (int device : tpDevices) {
                for (Data *root : {&key(sequence, layer), &value(sequence, layer)}) {
                    auto found = root->multiDeviceDatas.find(device);
                    if (found == root->multiDeviceDatas.end() || !found->second) return false;
                    Data &local = *found->second;
                    int required = slidingLayers[layer]
                        ? (verifying ? std::min(firstLength - 1, window - 1) + rows
                                     : std::min(nextLength, window)) : nextLength;
                    if (local.dims.size() != 3 || local.expansionDims.size() != 3 ||
                        local.expansionDims[1] < required || !local.cudaData ||
                        local.dataDevice != DataDevice::CUDA || local.dataType != kvCacheDataType)
                        return false;
                    if (!CanResizeKVLength(local, required, local.dims[2]))
                        directLengths[sequence] = 0;
                    cacheLengths[slot] = &local.dims[1];
                    pointers[slot] = local.cudaData;
                    capacities[slot++] = local.expansionDims[1];
                    if (!slidingLayers[layer]) sequenceCapacities[sequence] =
                        std::min(sequenceCapacities[sequence], local.expansionDims[1]);
                }
            }
        }
        return true;
    };
    ForSequenceRanges(tpWorkers, tpDevices, sequences, true, [&](int begin, int end) {
        for (int sequence = begin; sequence < end; ++sequence)
            valid[sequence] = validateSequence(sequence);
    });
    if (std::find(valid.begin(), valid.end(), 0) != valid.end()) return false;
    const int region = regions[0], capacity = sequenceCapacities[0];
    if (tpDecodeState && tpDecodeState->cachePointers == pointers &&
        tpDecodeState->cacheCapacities == capacities && tpDecodeState->region == region &&
        tpDecodeState->rows == rows && tpDecodeState->verifying == verifying &&
        tpDecodeState->collectHidden == collectHidden &&
        tpDecodeState->selection.greedy == selection.greedy &&
        tpDecodeState->selection.count == selection.count &&
        tpDecodeState->selection.invTemperature == selection.invTemperature &&
        tpDecodeState->sequenceRegions == regions &&
        tpDecodeState->sequenceCapacities == sequenceCapacities && tpDecodeState->ncclGeneration == generation) {
        if (!tpDecodeState->captured || tpDecodeState->disabled ||
            std::find(directLengths.begin(), directLengths.end(), 0) != directLengths.end())
            cacheLengths.clear();
        return tpDecodeState->active = !tpDecodeState->disabled;
    }
    cacheLengths.clear();
    auto state = tpDecodeState ? tpDecodeState : std::make_shared<TPDecodeState>();
    state->ClearGraphs();
    if (state->rows != rows) state->ranks.clear();
    state->sequenceRegions = std::move(regions);
    state->sequenceCapacities = std::move(sequenceCapacities);
    state->disabled = true;
    state->devices = tpDevices;
    state->region = region;
    state->rows = rows;
    state->verifying = verifying;
    state->collectHidden = collectHidden;
    state->selection.greedy = selection.greedy;
    state->selection.count = selection.count;
    state->selection.invTemperature = selection.invTemperature;
    state->capacity = capacity;
    state->cachePointers = std::move(pointers);
    state->cacheCapacities = std::move(capacities);
    state->ncclGeneration = generation;
    if (tpEp8MegaState && verifying) return false;
    for (int rank = 0; rank < ranks; ++rank) {
        FastllmCudaSetDevice(tpDevices[rank]);
        if (!FastllmCudaNaiveDecodeGraphSupported()) return false;
        auto &embedding = weight["model.embed_tokens.weight"];
        auto it = embedding.multiDeviceDatas.find(tpDevices[rank]);
        if (it == embedding.multiDeviceDatas.end() || !it->second->cudaData ||
            it->second->dataType != BFLOAT16) return false;
        // EP8 weights/descriptors were validated when its state was created.
        // The remaining checks apply only to resident indexed TP experts.
        for (int layer = 0; !tpEp8MegaState && layer < block_cnt; ++layer) {
            if (!moeLayers[layer]) continue;
            auto &weights = tpMoeWeights[rank][layer];
            if (weights.size() < 4 || !weights[2] || !weights[3]) return false;
            if (weights[2]->dataType == FP8_E4M3) {
                if (rows != 1 || verifying || weights[0] || weights[1]) return false;
                const int inter = weights[3]->dims.size() == 2 ? weights[3]->dims[1] : 0;
                if (inter <= 0) return false;
                for (size_t i = 2; i < weights.size(); ++i) {
                    Data *w = weights[i];
                    const std::vector<int> dims = i % 2 == 0
                        ? std::vector<int>{inter * 2, embed_dim} : std::vector<int>{embed_dim, inter};
                    if (!w || w->dataType != FP8_E4M3 || w->dataDevice != DataDevice::CUDA ||
                        !w->cudaData || w->dims != dims || w->blockM <= 0 || w->blockK <= 0)
                        return false;
                }
            } else {
                if (moeAtype != BFLOAT16 ||
                    !FastllmCudaNVFP4E4M3GroupedMoeSupported(tpDevices[rank])) return false;
                for (Data *w : weights)
                    if (w && w->dataType != NVFP4_BLOCK_16_E4M3_PACKED) return false;
                if (!FastllmCudaPrepareNVFP4E4M3Moe(weights.data(), weights.size())) return false;
            }
        }
        if (rank >= (int)state->ranks.size())
            state->ranks.emplace_back(new TPDecodeState::Rank());
        auto &buffers = state->ranks[rank]->storage->buffers;
        buffers.capacity = capacity;
        if (batch) {
            while ((int)buffers.sequences.size() < sequences)
                buffers.sequences.emplace_back(new TargetWorkspace());
            for (int sequence = 0; sequence < sequences; ++sequence)
                buffers.sequences[sequence]->capacity = state->sequenceCapacities[sequence];
        }
        state->ranks[rank]->features.verifying = verifying;
        state->ranks[rank]->features.collectHidden = collectHidden && rank == 0;
    }
    state->disabled = false;
    state->active = true;
    tpDecodeState = std::move(state);
    return true;
}

Data NaiveN05FlashModel::ForwardTensorParallelDecode(int rank, const Data &inputIds,
        const Data &positions, std::vector<std::pair<Data, Data>> &kv,
        const GenerationConfig &config, const Data *embedding, TargetCapture *capture,
        float *candidateResult, const TargetBatch *batch, bool keepCudaLogits) {
    auto &state = *tpDecodeState;
    auto &r = *state.ranks.at(rank);
    auto &storage = *r.storage;
    auto &buf = storage.buffers;
    const int actualRows = batch ? batch->Size() : state.rows;
    if (state.mode != TPDecodeState::Capture) {
        const size_t rowBytes = actualRows * sizeof(float);
        const size_t strideBytes = (storage.maxRows ? storage.maxRows : state.rows) * sizeof(float);
        const bool hostInputs = inputIds.dataDevice == DataDevice::CPU &&
            positions.dataDevice == DataDevice::CPU && inputIds.dataType == FLOAT32 &&
            positions.dataType == FLOAT32 &&
            inputIds.GetBytes() == rowBytes && positions.GetBytes() == rowBytes;
        if (storage.inputStorage.dims.empty() && buf.inputIds.dims.empty() && hostInputs) {
            storage.inputStorage.dataType = FLOAT32;
            storage.inputStorage.dataDevice = DataDevice::CUDA;
            storage.inputStorage.dataDeviceIds = {tpDevices[rank]};
            storage.inputStorage.Resize({3, (int)(strideBytes / sizeof(float))});
            storage.inputStorage.Allocate();
            buf.inputIds.FakeFrom(storage.inputStorage, 0);
            buf.inputIds.Resize(inputIds.dims);
            buf.positions.FakeFrom(storage.inputStorage, strideBytes);
            buf.positions.Resize(positions.dims);
            buf.liveKeys.FakeFrom(storage.inputStorage, 2 * strideBytes);
            buf.liveKeys.dataType = INT32;
            buf.liveKeys.Resize({state.rows});
        }
        if (hostInputs && !storage.inputStorage.dims.empty()) {
            // Copy integer lengths as bytes, preserving their representation.
            buf.inputIds.Resize({1, state.rows});
            buf.positions.Resize({1, state.rows});
            buf.liveKeys.Resize({state.rows});
            storage.hostInput.resize(3 * strideBytes);
            unsigned char *input = storage.hostInput.data();
            // Zero length is the device-side inactive-row contract. Clear all
            // tails on every replay, including when reusing the same graph.
            std::memset(input, 0, 3 * strideBytes);
            std::memcpy(input, inputIds.cpuData, rowBytes);
            std::memcpy(input + strideBytes, positions.cpuData, rowBytes);
            for (int row = 0; row < actualRows; ++row) {
                int length = batch ? batch->Key(row, 0).dims[1] + 1 : kv[0].first.dims[1] + row + 1;
                std::memcpy(input + 2 * strideBytes + row * sizeof(int), &length, sizeof(int));
            }
            FastllmCudaCopyFromHostToDevice(storage.inputStorage.cudaData, input, 3 * strideBytes);
        } else {
            // Keep these allocations and their CUDA addresses across all replays.
            auto copyInput = [&](Data &dst, const Data &src) {
                if (dst.dims.empty()) {
                    dst.dataType = src.dataType;
                    dst.UpdateUnitSize();
                    dst.dataDevice = DataDevice::CUDA;
                    dst.dataDeviceIds = {tpDevices[rank]};
                    dst.Resize(src.dims);
                    dst.Allocate();
                }
                if (src.dataDevice == DataDevice::CUDA)
                    FastllmCudaCopyFromDeviceToDevice(dst.cudaData, src.cudaData, src.GetBytes());
                else FastllmCudaCopyFromHostToDevice(dst.cudaData, src.cpuData, src.GetBytes());
                if (actualRows < state.rows)
                    FastllmCudaMemset0((char *)dst.cudaData + src.GetBytes(),
                        (state.rows - actualRows) * sizeof(float));
                dst.Resize({1, state.rows});
            };
            copyInput(buf.inputIds, inputIds);
            Data localPositions(positions);
            ToDataType(localPositions, FLOAT32);
            copyInput(buf.positions, localPositions);
            std::vector<int> lengths(state.rows);
            for (int row = 0; row < actualRows; ++row) lengths[row] = batch ? batch->Key(row, 0).dims[1] + 1 : kv[0].first.dims[1] + row + 1;
            if (buf.liveKeys.dims.empty()) {
                buf.liveKeys.dataType = INT32;
                buf.liveKeys.UpdateUnitSize();
                buf.liveKeys.dataDevice = DataDevice::CUDA;
                buf.liveKeys.dataDeviceIds = {tpDevices[rank]};
                buf.liveKeys.Resize({state.rows});
                buf.liveKeys.Allocate();
            }
            FastllmCudaCopyFromHostToDevice(buf.liveKeys.cudaData, lengths.data(), state.rows * sizeof(int));
        }
        if (storage.cacheDirty) {
            FastllmCudaCopyFromHostToDevice(storage.cacheTable.cudaData,
                storage.hostCache.data(), storage.hostCache.size() * sizeof(void *));
            storage.cacheDirty = false;
        }
        if (storage.mapsDirty) {
            FastllmCudaCopyFromHostToDevice(storage.cacheMaps.cudaData,
                storage.hostMaps.data(), storage.hostMaps.size());
            storage.mapsDirty = false;
        }
        if (state.mode == TPDecodeState::Prepare) return Data();
    }
    if (state.mode == TPDecodeState::Replay) {
        AssertInFastLLM(FastllmCudaGraphLaunch(r.exec), "Naive TP decode graph launch failed.");
    } else {
        if (state.mode == TPDecodeState::Warm) r.communicationPointers.clear();
        FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
#ifndef USE_ROCM
        // These variants already share activation/KV metadata buffers and are
        // replayed serially by the same rank worker. Keep one Lt workspace for
        // the whole family instead of exhausting one lease per captured shape.
        FastllmCudaLinearGraphScope linearScope(storage.linearGraphWorkspace);
#endif
        RunTarget(buf.inputIds, buf.positions, kv, config,
                  state.verifying || state.collectHidden ? &r.features : nullptr, rank, embedding, &buf, batch);
        if (FastllmCudaMergeMOEUsedGraphUnsafeFallback()) r.ok = false;
        if (state.selection.count)
            FastllmCudaNaiveLogitsSelect(buf.logits, tpVocabRanges[rank].first,
                state.selection.count, state.selection.greedy, state.selection.invTemperature,
                storage.logitsPartial, storage.logitsCandidates);
        if (state.mode == TPDecodeState::Capture) {
            bool clean = !FastllmCudaGetThreadError();
            r.ok = FastllmCudaGraphEndCapture(&r.graph) && clean && r.ok;
            if (!r.ok) fprintf(stderr, "[Fastllm] Naive TP graph rank %d: %s\n", rank,
                               FastllmCudaGraphLastError());
            return Data();
        }
    }
    if (capture && capture->collectHidden)
        for (auto &feature : r.features.hidden) Copy(feature.second, capture->hidden[feature.first]);
    if (state.selection.count && candidateResult) {
        FastllmCudaCopyFromDeviceToHost(candidateResult, storage.logitsCandidates.cudaData, (size_t)actualRows * state.selection.count * 2 * sizeof(float));
        return Data();
    }
    if (keepCudaLogits) {
        // Borrow the graph's stable output; never release or move its storage.
        Data output;
        output.FakeFrom(buf.logits, 0);
        // Storage is shared across batch graphs; its host shape may describe
        // the last captured batch rather than the graph being replayed.
        output.Resize({1, actualRows, tpVocabRanges[rank].second - tpVocabRanges[rank].first});
        return output;
    }
    // ToDevice(CPU) would release the graph's persistent output allocation.
    Data output(FLOAT32, {1, actualRows, tpVocabRanges[rank].second - tpVocabRanges[rank].first});
    output.Allocate();
    FastllmCudaCopyFromDeviceToHost(output.cpuData, buf.logits.cudaData, output.GetBytes());
    return output;
}
#endif

Data NaiveN05FlashModel::ForwardSingleGPU(int rank, const Data &inputIds, const Data &positions,
        std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config, const Data *embedding,
        TargetCapture *capture, float *candidateResult, const LogitsSelection *selection, const TargetBatch *batch) {
#ifdef USE_CUDA
    int device = tpDevices.at(rank);
    FastllmCudaSetDevice(device);
    static thread_local std::unique_ptr<Executor> executor;
    if (!executor) executor.reset(new Executor());
    executor->SetFirstDevice("cuda:" + std::to_string(device));
    struct RestoreExecutor {
        void *previous;
        ~RestoreExecutor() { SetCurrentThreadExecutor(previous); }
    } restore{GetExecutor()};
    SetCurrentThreadExecutor(executor.get());
    if (tpDecodeState && tpDecodeState->active && !tpDecodeState->disabled)
        return ForwardTensorParallelDecode(rank, inputIds, positions, kv, config, embedding, capture,
            candidateResult, batch, selection && selection->cudaSampling);
    // Generic operators can move their inputs. Give every worker its own IDs.
    Data localIds(inputIds), localPositions(positions);
    TargetWorkspace *workspace = batch ? tpBatchWorkspaces.at(rank).get() : capture && capture->verifying
        ? tpVerifyWorkspaces.at(rank).get() : nullptr;
    Data logits = RunTarget(localIds, localPositions, kv, config, capture, rank, embedding, workspace, batch);
    if (candidateResult && !isIntermediateChunkedPrefill) {
        Data partial, top;
        FastllmCudaNaiveLogitsSelect(logits, tpVocabRanges[rank].first,
            selection->count, selection->greedy, selection->invTemperature, partial, top);
        FastllmCudaCopyFromDeviceToHost(candidateResult, top.cudaData, top.GetBytes());
        ForceDeviceSync();
        return Data();
    }
    if (!isIntermediateChunkedPrefill && !(selection && selection->cudaSampling))
        logits.ToDevice(DataDevice::CPU);
    ForceDeviceSync();
    return logits;
#else
    ErrorInFastLLM("Naive ForwardSingleGPU requires CUDA.");
    return Data();
#endif
}

Data NaiveN05FlashModel::ForwardTensorParallel(const Data &inputIds, const Data &positions,
        std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config, TargetCapture *capture,
        LogitsSelection *selection, const TargetBatch *batch) {
#ifdef USE_CUDA
    PrepareTensorParallel();
    std::vector<std::pair<Data *, Data *>> singleKV;
    std::vector<GenerationConfig> singleConfigs;
    std::vector<int> singleLengths;
    TargetBatch singleBatch{singleKV, singleConfigs, singleLengths, block_cnt};
    if (tpServingState && !batch && !(capture && capture->verifying) && inputIds.dims == std::vector<int>({1, 1})) {
        for (auto &layer : kv) singleKV.emplace_back(&layer.first, &layer.second);
        singleConfigs.push_back(config);
        singleLengths.push_back(1);
        batch = &singleBatch;
    }
    const int vocab = weight["lm_head.weight"].dims[0];
    const bool compact = selection && selection->count > 0 && !isIntermediateChunkedPrefill &&
        std::all_of(tpVocabRanges.begin(), tpVocabRanges.end(), [&](const auto &range) {
            return FastllmNaiveCanSelectLogits(range.second - range.first, range.first,
                                              selection->count, selection->greedy);
        });
    if (compact) selection->count = std::min(selection->count, vocab);
    const int count = compact ? selection->count : 0;
    const bool cudaSampling = selection && selection->cudaSampling && !isIntermediateChunkedPrefill;
    LogitsSelection fullLogits;
    const int rows = batch ? batch->Size() : capture && capture->verifying ? inputIds.dims[1] : 1;
    const int sequences = batch ? batch->Size() : 1;
    auto key = [&](int sequence, int layer) -> Data & { return batch ? batch->Key(sequence, layer) : kv[layer].first; };
    auto value = [&](int sequence, int layer) -> Data & { return batch ? batch->Value(sequence, layer) : kv[layer].second; };
    AssertInFastLLM(batch || (int)kv.size() == block_cnt, "Naive TP: incomplete KV cache.");
    std::vector<std::exception_ptr> errors(tpDevices.size());
    auto runRanks = [&](const std::function<void(int)> &task) {
        std::fill(errors.begin(), errors.end(), nullptr);
        tpWorkers.Run(tpDevices, task, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
    };
    const bool reuseServingCache = CanReuseTensorParallelServingCache(batch);
    if (!reuseServingCache)
    for (int sequence = 0; sequence < sequences; ++sequence)
    for (int layer = 0; layer < block_cnt; ++layer) {
        for (Data *cache : {&key(sequence, layer), &value(sequence, layer)}) {
            cache->kvCacheFp8Indexer = cache == &key(sequence, layer) && !slidingLayers[layer] && fp8IndexerCache;
            for (auto &entry : cache->multiDeviceDatas) entry.second->kvCacheFp8Indexer = cache->kvCacheFp8Indexer;
            if (cache->multiDeviceData && cache->dataDevice == DataDevice::CUDA) continue;
            AssertInFastLLM(cache->dims.empty(), "Naive TP does not restore CPU KV archives.");
            cache->dataDevice = DataDevice::CUDA;
            cache->multiDeviceData = true;
            cache->dataDeviceIds = tpDevices;
            cache->isKVCache = true;
            for (int device : tpDevices) {
                if (cache->multiDeviceDatas.count(device)) continue;
                Data *local = new Data(kvCacheDataType);
                local->kvCacheFp8Indexer = cache->kvCacheFp8Indexer;
                local->dataDevice = DataDevice::CUDA;
                local->dataDeviceIds = {device};
                local->isKVCache = true;
                cache->multiDeviceDatas[device] = local;
            }
        }
    }
    if (!tpPagedCaches.empty() && !(tpServingState && tpServingState->preparing) && !reuseServingCache) {
        // Own only rank-local page references. The generic prefix restore's
        // root is a borrowed view of rank zero; publish the model's packed
        // global shape without releasing that view a second time.
        for (int sequence = 0; sequence < sequences; ++sequence)
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (slidingLayers[layer]) continue;
            for (int part = 0; part < 2; ++part) {
                Data &root = part ? value(sequence, layer) : key(sequence, layer);
                const int past = root.dims.empty() ? 0 : root.dims[1];
                const int width = full.kvHeads * (part ? full.valueDim : full.headDim) + (!part ? IndexerCacheWidth(kvCacheDataType) : 0);
                root.isPagedKVCache = false;
                root.pagedKVCacheData = nullptr;
                root.pageIndex.clear();
                if (root.dataDevice == DataDevice::CUDA) root.Resize({1, past, width});
                root.expansionDims = {1, std::min(max_positions,
                    tpPagedCaches[0][layer * 2 + part]->maxPages * GetPageLen()), width};
            }
        }
        runRanks([&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            for (int sequence = 0; sequence < sequences; ++sequence)
            for (int layer = 0; layer < block_cnt; ++layer) {
                if (slidingLayers[layer]) continue;
                for (int part = 0; part < 2; ++part) {
                    Data &root = part ? value(sequence, layer) : key(sequence, layer);
                    const int next = root.dims[1] + (batch ? batch->lengths[sequence] : inputIds.dims[1]);
                    PreparePagedCache(*root.multiDeviceDatas.at(tpDevices[rank]),
                        *tpPagedCaches[rank][layer * 2 + part], next);
                }
            }
        });
    }
    // CPU embedding uses the shared worker pool; compute it once on the
    // caller instead of concurrently entering that pool from every rank.
    Data embedding;
    if (!GetCudaEmbedding() || GetLowMemMode())
        Embedding(inputIds, weight["model.embed_tokens.weight"], embedding);
    const bool verify = capture && capture->verifying;
    if (verify && tpSwaRingCache) {
        runRanks([&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            const int past = key(0, 0).dims.empty() ? 0 : key(0, 0).dims[1];
            for (int layer = 0; layer < block_cnt; ++layer)
                if (slidingLayers[layer]) FastllmCudaNaiveSwaCacheLayout(
                    *key(0, layer).multiDeviceDatas.at(tpDevices[rank]),
                    *value(0, layer).multiDeviceDatas.at(tpDevices[rank]), past, false);
        });
    }
    struct RestoreState {
        std::shared_ptr<TPDecodeState> &active, &saved;
        bool swapped;
        ~RestoreState() { if (swapped) active.swap(saved); }
    } restore{tpDecodeState, batch ? tpBatchState : tpVerifyState, verify || batch};
    if (verify || batch) tpDecodeState.swap(batch ? tpBatchState : tpVerifyState);
    // Single-row target forwards may also need draft features (for example,
    // constrained decoding); keep this path eligible for graph replay.
    std::vector<int *> localCacheLengths;
    // The scheduler-owned plan already owns these pointers for this replay.
    // Reuse its vector as well, avoiding an O(batch * layers * ranks) copy.
    auto &cacheLengths = reuseServingCache ? tpServingState->cacheLengths : localCacheLengths;
    bool graphDecode = PrepareTensorParallelDecode(inputIds, kv, verify,
        compact ? *selection : fullLogits, cacheLengths, batch, capture && capture->collectHidden);
    // Retain each rank's scratch and communication addresses per execution mode.
    // The existing collective registration still validates the complete tuple.
    if (capture && capture->verifying && tpVerifyWorkspaces.empty()) {
        for (size_t rank = 0; rank < tpDevices.size(); ++rank)
            tpVerifyWorkspaces.emplace_back(new TargetWorkspace());
    }
    if (batch && tpBatchWorkspaces.empty()) {
        for (size_t rank = 0; rank < tpDevices.size(); ++rank)
            tpBatchWorkspaces.emplace_back(new TargetWorkspace());
    }
    std::vector<PrefixSnapshots> prefixes(sequences);
    bool recording = false;
    if (saveHistoryChat && !verify) {
        for (int sequence = 0; sequence < sequences; ++sequence) {
            const Data &first = key(sequence, 0);
            prefixes[sequence] = BeginPrefixSnapshots(first, first.dims.empty() ? 0 : first.dims[1],
                batch ? batch->lengths[sequence] : inputIds.dims[1]);
            recording = recording || !prefixes[sequence].empty();
        }
    }
    std::vector<TargetCapture> captures(capture || recording ? tpDevices.size() : 0);
    for (int rank = 0; rank < (int)captures.size(); ++rank) {
        captures[rank].verifying = verify;
        captures[rank].collectHidden = capture && capture->collectHidden && rank == 0;
        if (recording) captures[rank].prefixes = prefixes;
    }
    std::vector<Data> logits(compact || cudaSampling ? 0 : tpDevices.size());
    std::vector<float> candidates(compact ? tpDevices.size() * rows * count * 2 : 0);
    // Reuse one root-device matrix, as in the other TP models. Sampling stays
    // outside the graph so per-request parameters and RNG change on each replay.
    static thread_local Data samplingStorage(FLOAT32);
    Data &cudaLogits = samplingStorage;
    if (cudaSampling) {
        FastllmCudaSetDevice(tpDevices[0]);
        cudaLogits.ToDevice(DataDevice::CUDA, {tpDevices[0]}, false);
        cudaLogits.Resize({rows, vocab});
        cudaLogits.Allocate(false);
    }

    auto forwardRanks = [&]() {
        runRanks([&](int rank) {
            Data local = ForwardSingleGPU(rank, inputIds, positions, kv, config,
                                          embedding.dims.empty() ? nullptr : &embedding,
                                          captures.empty() ? nullptr : &captures[rank],
                                          compact ? candidates.data() + (size_t)rank * rows * count * 2 : nullptr,
                                          compact || cudaSampling ? selection : nullptr, batch);
            if (cudaSampling && !local.dims.empty()) {
                const auto range = tpVocabRanges[rank];
                const size_t width = (range.second - range.first) * sizeof(float);
                AssertInFastLLM(local.dataDevice == DataDevice::CUDA &&
                    local.Count(0) == (uint64_t)rows * (range.second - range.first),
                    "Naive CUDA sampling: invalid shard output.");
                // Queue the peer write after this rank's graph on the same
                // stream. All shards complete before root-device sampling.
                AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(
                    (float *)cudaLogits.cudaData + range.first, (size_t)vocab * sizeof(float),
                    local.cudaData, width, width, rows), "Naive CUDA sampling: logits copy failed.");
                FastllmCudaSyncCurrentThreadStream();
            } else if (!compact && !cudaSampling) logits[rank].CopyFrom(local);
        });
    };
    if (graphDecode && tpDecodeState->warmed && !tpDecodeState->captured &&
        (!tpServingState || !tpServingState->preparing || tpServingState->capturing)) {
        auto &state = *tpDecodeState;
        state.mode = TPDecodeState::Prepare;
        forwardRanks();
        // Resolve sentinels on all devices before beginning any capture.
        runRanks([&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            state.ranks[rank]->ok = FastllmCudaGraphPrepareCaptureDevice();
        });
        bool ok = std::all_of(state.ranks.begin(), state.ranks.end(),
                             [](const auto &r) { return r->ok; });
        bool pool = ok && FastllmCudaGraphMemoryPoolBegin();
        if (pool) {
            runRanks([&](int rank) {
                FastllmCudaSetDevice(tpDevices[rank]);
                FastllmCudaClearThreadError();
                state.ranks[rank]->ok = FastllmCudaGraphBeginCapture();
            });
            ok = std::all_of(state.ranks.begin(), state.ranks.end(),
                             [](const auto &r) { return r->ok; });
            if (ok) {
                state.mode = TPDecodeState::Capture;
                forwardRanks();
            } else {
                // Every successful begin is ended on its owning worker.
                runRanks([&](int rank) {
                    if (state.ranks[rank]->ok) {
                        FastllmCudaSetDevice(tpDevices[rank]);
                        FastllmCudaGraphEndCapture(&state.ranks[rank]->graph);
                    }
                    state.ranks[rank]->ok = false;
                });
            }
            ok = FastllmCudaGraphMemoryPoolEnd(state.reservedPointers) && ok;
            ok = ok && std::all_of(state.ranks.begin(), state.ranks.end(),
                                   [](const auto &r) { return r->ok && r->graph; });
            if (ok) {
                runRanks([&](int rank) {
                    FastllmCudaSetDevice(tpDevices[rank]);
                    auto &r = *state.ranks[rank];
                    r.ok = FastllmCudaGraphInstantiate(r.graph, &r.exec);
                });
                ok = std::all_of(state.ranks.begin(), state.ranks.end(),
                                 [](const auto &r) { return r->ok && r->exec; });
            }
        } else ok = false;
        if (!ok) {
            // Capture has not executed the token or committed KV metadata.
            // All ranks agree on fallback before any graph is launched.
            state.disabled = true;
            runRanks([&](int rank) {
                FastllmCudaSetDevice(tpDevices[rank]);
                FastllmCudaClearLastError();
                FastllmCudaClearThreadError();
            });
            fprintf(stderr, "[Fastllm] Naive TP decode graph capture failed; using eager for this cache.\n");
            graphDecode = false;
        } else state.captured = true;
    }
    // Startup uses borrowed descriptor-only caches, not normal request-owned
    // pages. A failed shape is simply unavailable; eager fallback starts with
    // real requests after preparation, never with this private warmup backing.
    if (!graphDecode && tpServingState && tpServingState->preparing) return Data();
    if (!graphDecode || verify) tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
    if (!graphDecode && !tpKVPool.empty()) {
        // Graph preparation already validates every rank's existing capacity.
        // Eager paths must finish all KV allocations before any collective, so
        // one exhausted rank cannot leave the others waiting in all-reduce.
        struct Reservation { Data *cache; int capacity, width; };
        std::vector<Reservation> reservations;
        for (int sequence = 0; sequence < sequences; ++sequence) {
            const int length = batch ? batch->lengths[sequence] : inputIds.dims[1];
            const int reserve = CacheReserveCapacity(batch ? batch->configs[sequence] : config);
            for (int layer = 0; layer < block_cnt; ++layer) {
                const auto &cfg = slidingLayers[layer] ? sliding : full;
                for (Data *cache : {&key(sequence, layer), &value(sequence, layer)}) {
                    const int64_t past = cache->dims.empty() ? 0 : cache->dims[1];
                    const int64_t wanted = slidingLayers[layer] && !tpSwaPrefillWorkspaces.empty()
                        ? window : std::max<int64_t>(past + length,
                            slidingLayers[layer] ? std::min<int64_t>(reserve, (int64_t)window - 1 + length) : reserve);
                    if (cache->expansionDims.size() == 3 && cache->expansionDims[1] >= wanted) continue;
                    const bool isValue = cache == &value(sequence, layer);
                    const int width = std::max(1, cfg.kvHeads / (int)tpDevices.size()) *
                        (isValue ? cfg.valueDim : cfg.headDim) +
                        (!isValue && !slidingLayers[layer] ? IndexerCacheWidth(kvCacheDataType) : 0);
                    reservations.push_back({cache,
                        (int)std::min<int64_t>(INT_MAX, (wanted + 127) / 128 * 128), width});
                }
            }
        }
        const int prefillLength = batch
            ? *std::max_element(batch->lengths.begin(), batch->lengths.end()) : inputIds.dims[1];
        const bool growScratch = std::any_of(tpSwaPrefillWorkspaces.begin(), tpSwaPrefillWorkspaces.end(),
            [&](const auto &workspace) {
                for (const Data *data : {&workspace.first, &workspace.second})
                    if (!data->cudaData || data->expansionDims.empty() ||
                        data->expansionDims[1] < (int64_t)window - 1 + prefillLength) return true;
                return false;
            });
        if (!reservations.empty() || growScratch) runRanks([&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            if (growScratch) PrepareSwaPrefillWorkspace(rank, prefillLength);
            for (const auto &item : reservations)
                ReserveCache(*item.cache->multiDeviceDatas.at(tpDevices[rank]), item.capacity, item.width);
        });
    }
    if (graphDecode)
        tpDecodeState->mode = tpDecodeState->captured ? TPDecodeState::Replay : TPDecodeState::Warm;
    forwardRanks();
    bool committedRootMetadata = false;
    if (graphDecode) {
        const size_t localSlots = (size_t)sequences * block_cnt * tpDevices.size() * 2;
        // Commit root and rank lengths in a single traversal.
        committedRootMetadata =
            cacheLengths.size() == localSlots + (size_t)sequences * block_cnt * 2;
        tpDecodeState->warmed = true;
        if (!std::all_of(tpDecodeState->ranks.begin(), tpDecodeState->ranks.end(),
                         [](const auto &r) { return r->ok; })) {
            // Warmup already executed this token. Commit its KV metadata below,
            // then use eager on the next token instead of capturing host routes.
            tpDecodeState->disabled = true;
            fprintf(stderr, "[Fastllm] Naive TP MoE warmup used a graph-unsafe path; using eager for this cache.\n");
        }
        ForSequenceRanges(tpWorkers, tpDevices, sequences, true, [&](int begin, int end) {
            for (int sequence = begin; sequence < end; ++sequence) {
                int oldLength = key(sequence, 0).dims[1];
                int nextLength = oldLength + (batch ? 1 : tpDecodeState->rows);
                const int lastPageLength = (nextLength - 1) % GetPageLen() + 1;
                // After a validated replay, a full SWA ring retains the same
                // host shape. Only full-attention lengths continue to grow.
                const bool stableSwa = reuseServingCache && !verify && oldLength >= window;
                size_t slot = (size_t)sequence * block_cnt * tpDevices.size() * 2;
                // Graph kernels append/trim on the device. Commit host metadata once,
                // only after all ranks finish a successful warmup or replay.
                for (int layer = 0; layer < block_cnt; ++layer) {
                    if (stableSwa && slidingLayers[layer]) {
                        slot += tpDevices.size() * 2;
                        continue;
                    }
                    int length = slidingLayers[layer]
                        ? (verify ? std::min(oldLength, window - 1) + tpDecodeState->rows
                                  : std::min(nextLength, window - 1)) : nextLength;
                    if (!cacheLengths.empty()) {
                        for (size_t i = 0; i < tpDevices.size() * 2; ++i) {
                            if (tpServingState && tpServingState->pageLengths[slot])
                                *tpServingState->pageLengths[slot] = lastPageLength;
                            *cacheLengths[slot++] = length;
                        }
                    } else for (int device : tpDevices)
                        for (Data *root : {&key(sequence, layer), &value(sequence, layer)}) {
                            Data &local = *root->multiDeviceDatas.at(device);
                            ResizeKVLength(local, length, local.dims[2]);
                        }
                    if (committedRootMetadata) {
                        const size_t rootSlot = localSlots + ((size_t)sequence * block_cnt + layer) * 2;
                        *cacheLengths[rootSlot] = *cacheLengths[rootSlot + 1] = length;
                    }
                }
            }
        });
    }
    if (graphDecode && !tpServingState && !tpPagedCaches.empty()) {
        for (int sequence = 0; sequence < sequences; ++sequence)
        for (int layer = 0; layer < block_cnt; ++layer)
        for (Data *root : {&key(sequence, layer), &value(sequence, layer)})
        for (auto &entry : root->multiDeviceDatas) {
            Data &local = *entry.second;
            if (local.isPagedKVCache) local.lastPageLen = (local.dims[1] - 1) % local.pageLen + 1;
        }
    }
    if (graphDecode && recording) {
        // Copy only at page boundaries, outside every captured graph. Payloads
        // stay on device and retain exactly the stored BF16/FP8 bits.
        runRanks([&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            for (int sequence = 0; sequence < sequences; ++sequence)
                for (const auto &snapshot : prefixes[sequence])
                    for (int layer = 0; layer < block_cnt; ++layer) if (slidingLayers[layer]) {
                        Data &k = *key(sequence, layer).multiDeviceDatas.at(tpDevices[rank]);
                        Data &v = *value(sequence, layer).multiDeviceDatas.at(tpDevices[rank]);
                        int rows = std::min(snapshot->length, window - 1);
                        CopyPrefixWindow(*snapshot, rank, layer, k, v,
                            tpSwaRingCache ? snapshot->length - rows : k.dims[1] - rows, tpSwaRingCache);
                    }
            ForceDeviceSync();
        });
    }
    if (recording) for (int sequence = 0; sequence < sequences; ++sequence)
        FinishPrefixSnapshots(key(sequence, 0), prefixes[sequence],
            (batch ? batch->lengths[sequence] : inputIds.dims[1]) > 1);
    if (!committedRootMetadata) ForSequenceRanges(tpWorkers, tpDevices, sequences, graphDecode, [&](int begin, int end) {
        for (int sequence = begin; sequence < end; ++sequence)
        for (int layer = 0; layer < block_cnt; ++layer) {
            const auto &cfg = slidingLayers[layer] ? sliding : full;
            auto syncMeta = [&](Data &root, int width) {
                const Data &local = *root.multiDeviceDatas.at(tpDevices[0]);
                ResizeKVLength(root, local.dims[1], width);
                if (root.expansionDims.size() == 3) {
                    root.expansionDims[0] = 1;
                    root.expansionDims[1] = local.expansionDims[1];
                    root.expansionDims[2] = width;
                } else root.expansionDims = {1, local.expansionDims[1], width};
            };
            syncMeta(key(sequence, layer), cfg.kvHeads * cfg.headDim + (slidingLayers[layer] ? 0 : IndexerCacheWidth(kvCacheDataType)));
            syncMeta(value(sequence, layer), cfg.kvHeads * cfg.valueDim);
        }
    });
    if (capture) capture->hidden = std::move(captures[0].hidden);
    if (isIntermediateChunkedPrefill) return Data();
    if (cudaSampling) {
        selection->sampledTokens = SampleCudaLogits(cudaLogits,
            batch ? batch->configs : std::vector<GenerationConfig>{config});
        return Data();
    }
    if (compact) {
        auto &output = selection->candidates;
        output.dataType = FLOAT32;
        output.Resize({rows, count * 2});
        output.Allocate();
        std::vector<std::pair<float, int>> choices;
        choices.reserve(tpDevices.size() * count);
        for (int row = 0; row < rows; ++row) {
            choices.clear();
            for (size_t rank = 0; rank < tpDevices.size(); ++rank) {
                const float *local = candidates.data() + (rank * rows + row) * count * 2;
                for (int k = 0; k < count; ++k)
                    if (local[k * 2] >= 0) choices.emplace_back(local[k * 2 + 1], (int)local[k * 2]);
            }
            auto better = [&](const auto &a, const auto &b) {
                return selection->greedy ? FastllmNaiveTop1Better(a.first, a.second, b.first, b.second) :
                    (a.first > b.first || (a.first == b.first && a.second < b.second));
            };
            int kept = std::min<int>(count, choices.size());
            std::partial_sort(choices.begin(), choices.begin() + kept, choices.end(), better);
            float *dst = (float *)output.cpuData + (size_t)row * count * 2;
            for (int k = 0; k < count; ++k) {
                dst[k * 2] = k < kept ? choices[k].second : 0;
                dst[k * 2 + 1] = k < kept ? choices[k].first : -INFINITY;
            }
        }
        return Data();
    }
    Data output(FLOAT32, {1, rows, vocab});
    output.Allocate();
    for (int r = 0; r < (int)tpDevices.size(); ++r) {
        auto range = tpVocabRanges[r];
        const int localVocab = range.second - range.first;
        for (int row = 0; row < rows; ++row)
            std::memcpy((float *)output.cpuData + (size_t)row * vocab + range.first,
                        (float *)logits[r].cpuData + (size_t)row * localVocab,
                        localVocab * sizeof(float));
    }
    return output;
#else
    return Data();
#endif
}

bool NaiveN05FlashModel::HasVerificationGraph() const {
#ifdef USE_CUDA
    return GetFastllmEnv().cudaGraph && tpVerifyState &&
        tpVerifyState->warmed && !tpVerifyState->disabled;
#else
    return false;
#endif
}

Data NaiveN05FlashModel::RunDraftHead(Data &hidden) {
    if (tpDevices.empty()) {
        Data output;
        if (hidden.dims[1] > 1) MatMulTransB(hidden, weight["lm_head.weight"], output);
        else Linear(hidden, weight["lm_head.weight"], Data(), output);
        return output;
    }
#ifdef USE_CUDA
    const int rows = hidden.dims[1], vocab = weight["lm_head.weight"].dims[0];
    const int owner = tpDevices.front();
    if (hidden.dataType == BFLOAT16 && hidden.dataDevice == DataDevice::CUDA &&
        hidden.dataDeviceIds == std::vector<int>{owner} && FastllmCudaPeerAccessInit(tpDevices)) {
        // Keep the shared vocabulary shards on their ranks. The owner publishes
        // its hidden rows once, and ranks copy their logits directly into disjoint
        // columns of the owner's result, without a full CPU round trip.
        FastllmCudaSetDevice(owner);
        if (tpDraftHeadInputs.empty()) {
            tpDraftHeadInputs.resize(tpDevices.size());
            tpDraftHeadLogits.resize(tpDevices.size());
        }
        tpDraftHeadOutput.dataType = BFLOAT16;
        tpDraftHeadOutput.UpdateUnitSize();
        tpDraftHeadOutput.Resize({1, rows, vocab});
        tpDraftHeadOutput.ToDevice(DataDevice::CUDA, std::vector<int>{owner}, false);
        tpDraftHeadOutput.Allocate(false);
        struct ReadyEvent {
            void *event = FastllmCudaEventCreate();
            ~ReadyEvent() { FastllmCudaEventDestroy(event); }
        } ready;
        FastllmCudaEventRecordCurrentThread(ready.event);
        std::vector<std::exception_ptr> errors(tpDevices.size());
        tpWorkers.Run(tpDevices, [&](int rank) {
            const int device = tpDevices[rank];
            FastllmCudaSetDevice(device);
            Executor executor;
            executor.SetFirstDevice("cuda:" + std::to_string(device));
            struct Restore {
                void *previous;
                ~Restore() { SetCurrentThreadExecutor(previous); }
            } restore{GetExecutor()};
            SetCurrentThreadExecutor(&executor);
            Data &input = tpDraftHeadInputs[rank], &logits = tpDraftHeadLogits[rank];
            input.dataType = BFLOAT16;
            input.UpdateUnitSize();
            input.Resize(hidden.dims);
            input.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
            input.Allocate(false);
            FastllmCudaCurrentThreadStreamWaitEvent(ready.event);
            AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(device, input.cudaData,
                owner, hidden.cudaData, hidden.GetBytes()), "Naive draft head input copy failed.");
            Data &head = *weight["lm_head.weight"].multiDeviceDatas.at(device);
            if (rows > 1) MatMulTransB(input, head, logits);
            else Linear(input, head, Data(), logits);
            const auto range = tpVocabRanges[rank];
            const size_t bytes = (range.second - range.first) * sizeof(uint16_t);
            AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(
                (uint16_t *)tpDraftHeadOutput.cudaData + range.first, vocab * sizeof(uint16_t),
                logits.cudaData, bytes, bytes, rows), "Naive draft head logits copy failed.");
            // All peer writes finish before the owner consumes the joined head.
            // This also keeps input/logits storage alive for asynchronous copies.
            ForceDeviceSync();
        }, errors);
        FastllmCudaSetDevice(owner);
        for (auto error : errors) if (error) std::rethrow_exception(error);
        Data output;
        Copy(tpDraftHeadOutput, output);
        return output;
    }
    // Copy once before workers consume the owner stream's result. Each worker
    // owns its inputs and the existing vocabulary shard; no extra full head.
    Data host(hidden);
    host.ToDevice(DataDevice::CPU);
    std::vector<Data> logits(tpDevices.size());
    std::vector<std::exception_ptr> errors(tpDevices.size());
    tpWorkers.Run(tpDevices, [&](int rank) {
        const int device = tpDevices[rank];
        FastllmCudaSetDevice(device);
        Executor executor;
        executor.SetFirstDevice("cuda:" + std::to_string(device));
        struct Restore {
            void *previous;
            ~Restore() { SetCurrentThreadExecutor(previous); }
        } restore{GetExecutor()};
        SetCurrentThreadExecutor(&executor);
        Data input(host);
        input.ToDevice(DataDevice::CUDA, std::vector<int>{device});
        Data &head = *weight["lm_head.weight"].multiDeviceDatas.at(device);
        if (rows > 1) MatMulTransB(input, head, logits[rank]);
        else Linear(input, head, Data(), logits[rank]);
        logits[rank].ToDevice(DataDevice::CPU);
    }, errors);
    for (auto error : errors) if (error) std::rethrow_exception(error);
    Data output(BFLOAT16, {1, rows, vocab});
    output.Allocate();
    for (int rank = 0; rank < (int)tpDevices.size(); ++rank) {
        const auto range = tpVocabRanges[rank];
        const int localVocab = range.second - range.first;
        for (int row = 0; row < rows; ++row)
            std::memcpy((uint16_t *)output.cpuData + (size_t)row * vocab + range.first,
                        (uint16_t *)logits[rank].cpuData + (size_t)row * localVocab,
                        localVocab * sizeof(uint16_t));
    }
    ApplyDraftDevice();
    output.ToDevice(DataDevice::CUDA, std::vector<int>{tpDevices.front()});
    return output;
#else
    return Data();
#endif
}

}
