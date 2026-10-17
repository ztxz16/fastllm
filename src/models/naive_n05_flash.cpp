#include "naive_n05_flash.h"
#include "naive_n05_tp.h"
#include "blocks/baseblock.h"
#include "executor.h"
#include "json11.hpp"
#include "utils.h"
#include <cmath>
#include <climits>
#include "devices/cuda/naive-n05-cuda.cuh"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/cudaworkspace.h"
#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
namespace {
    float ConfigNumber(const WeightMap &weight, const std::string &key, float fallback) {
        auto it = weight.dicts.find(key);
        return it == weight.dicts.end() || it->second == "null"
                   ? fallback : std::stof(it->second);
    }
    std::vector<int> ConfigList(const WeightMap &weight, const std::string &key, int size) {
        auto it = weight.dicts.find(key);
        AssertInFastLLM(it != weight.dicts.end(), "Naive-N0.5 requires " + key);
        std::string error;
        auto values = json11::Json::parse(it->second, error);
        AssertInFastLLM(error.empty() && (int)values.array_items().size() == size,
                        "Invalid Naive-N0.5 " + key);
        std::vector<int> result;
        for (auto &value : values.array_items()) result.push_back(value.int_value());
        return result;
    }
#ifdef USE_CUDA
    // Copy stored KV bits between a request's window and chronological scratch.
    // At most one side wraps; copies are ordered with attention on the rank stream.
    void CopySwaWindow(const Data &key, const Data &value, Data &targetKey, Data &targetValue,
                       int sourceFirst, int targetFirst, int rows) {
        if (rows == 0) return;
        void *dsts[4]; const void *srcs[4]; size_t sizes[4];
        int copies = 0;
        for (int part = 0; part < 2; ++part) {
            const Data &source = part ? value : key;
            Data &target = part ? targetValue : targetKey;
            const size_t rowBytes = (size_t)source.expansionDims[2] * source.unitSize;
            AssertInFastLLM(source.dataType == target.dataType &&
                source.expansionDims[2] == target.expansionDims[2], "Invalid SWA workspace layout.");
            for (int offset = 0; offset < rows;) {
                int from = (sourceFirst + offset) % source.expansionDims[1];
                int to = (targetFirst + offset) % target.expansionDims[1];
                int count = std::min({rows - offset, source.expansionDims[1] - from,
                                     target.expansionDims[1] - to});
                AssertInFastLLM(copies < 4, "Invalid SWA window copy.");
                dsts[copies] = (uint8_t *)target.cudaData + to * rowBytes;
                srcs[copies] = (uint8_t *)source.cudaData + from * rowBytes;
                sizes[copies++] = count * rowBytes;
                offset += count;
            }
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
            dsts, srcs, sizes, copies), "SWA workspace copy failed.");
    }
#endif
}

void NaiveN05FlashModel::SetIndexerDataType(DataType type) {
    AssertInFastLLM(type == DATA_AUTO_NONE || type == FP8_E4M3,
                    "Naive --indexer_dtype supports auto or fp8.");
    AssertInFastLLM(!tpPrepared && !tpServingState && tpKVPool.empty() && activePrefixes.empty(),
                    "Set Indexer dtype before warmup or inference.");
#ifndef USE_CUDA
    AssertInFastLLM(type == DATA_AUTO_NONE, "FP8 Indexer requires CUDA.");
#endif
    fp8IndexerCache = type == FP8_E4M3;
    if (fp8IndexerCache) indexFp8 = true;
    else indexFp8 = weight.dicts["indexer_activation_dtype"] != "bf16";

}

int NaiveN05FlashModel::CacheReserveCapacity(const GenerationConfig &config) const {
    // Like ChatGLM, reserve the request's prompt and bounded output in one
    // allocation. Chunked prefill must use the full prompt length supplied by
    // GenerationConfig, not the current chunk. Warmup/direct calls without
    // this metadata keep the existing incremental allocation path.
    if (config.input_token_length <= 0) return 0;
    int64_t capacity = config.input_token_length;
    if (config.output_token_limit > 0) capacity += (int64_t)config.output_token_limit - 1;
    int limit = max_positions;
    if (tokensLimit > 0) limit = std::min(limit, tokensLimit);
    if (GetMaxTokens() > 0) limit = std::min(limit, GetMaxTokens());
    return (int)std::min<int64_t>(capacity, limit);
}

void NaiveN05FlashModel::AppendCache(Data &cache, Data &input, DataType cacheType, int reserveCapacity) {
    int oldLength = cache.dims.empty() ? 0 : cache.dims[1];
    int length = oldLength + input.dims[1];
    cache.dataType = cacheType;
    cache.UpdateUnitSize();
    cache.ToDevice(input.dataDevice, input.dataDeviceIds);
    const int width = input.dims[2] + (cache.kvCacheFp8Indexer
        ? IndexerCacheWidth(cacheType) - IndexerCacheWidth(BFLOAT16) : 0);
    const int64_t wanted = std::max(length, reserveCapacity);
    if (cache.expansionDims.empty() || cache.expansionDims[1] < wanted) {
        const int capacity = (int)std::min<int64_t>(INT_MAX, (wanted + 127) / 128 * 128);
        ReserveCache(cache, capacity, width);
    }
    if (cache.dims.empty()) cache.Resize({1, 0, width});
    if (cache.isPagedKVCache || cache.dataType == FP8_E4M3 || cache.kvCacheFp8Indexer) {
#ifdef USE_CUDA
        FastllmCudaNaivePagedWrite(cache, input, oldLength);
#endif
        cache.Resize({1, length, width});
        if (cache.isPagedKVCache) cache.lastPageLen = (length - 1) % cache.pageLen + 1;
    } else CatDirect(cache, input, 1);
    cache.isKVCache = true;
}

void NaiveN05FlashModel::ReserveCache(Data &cache, int capacity, int width) {
    if (cache.expansionDims.size() == 3 && cache.expansionDims[1] >= capacity) return;
    tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
#ifdef USE_CUDA
    auto pool = cache.dataDevice == DataDevice::CUDA && cache.dataDeviceIds.size() == 1
        ? tpKVPool.find(cache.dataDeviceIds[0]) : tpKVPool.end();
    if (pool != tpKVPool.end()) {
        AssertInFastLLM(FastllmNaiveKVType(cache.dataType) && !cache.isFake && !cache.cudaDataBorrowed &&
            (cache.dims.empty() || (cache.dims.size() == 3 && cache.dims[0] == 1 &&
                                   cache.dims[2] == width && cache.dims[1] <= capacity)),
            "Naive KV pool: invalid contiguous cache.");
        Data storage(cache.dataType, {1, capacity, width});
        storage.dataDevice = DataDevice::CUDA;
        storage.dataDeviceIds = cache.dataDeviceIds;
        storage.directMemory = true;
        storage.cudaData = AllocateCudaWorkspace(pool->second, storage.GetBytes());
        if (!storage.cudaData)
            throw CudaWorkspaceExhausted("Naive KV pool exhausted; reduce concurrent context/output lengths or increase --tokens before startup.");
        if (cache.cudaData && !cache.dims.empty() && cache.dims[1] > 0)
            FastllmCudaCopyFromDeviceToDevice(storage.cudaData, cache.cudaData, cache.GetBytes());
        cache.FreeSpace();
        cache.cudaData = storage.cudaData;
        storage.cudaData = nullptr;
        cache.directMemory = true;
        cache.expansionDims = storage.dims;
        cache.expansionSize = storage.Count(0);
        cache.expansionBytes = storage.GetBytes();
        cache.strides = storage.strides;
        return;
    }
#endif
    cache.Expansion({1, capacity, width});
}

void NaiveN05FlashModel::PreparePagedCache(Data &cache, PagedCacheManager &manager, int end) {
#ifdef USE_CUDA
    const int width = manager.dims[3], capacity = std::min(max_positions,
        std::min(tokensLimit > 0 ? tokensLimit : INT_MAX, manager.maxPages * manager.pageLen));
    AssertInFastLLM(end >= 0 && end <= capacity && !cache.isFake,
                    "Naive paged KV capacity exceeded.");
    if (!cache.isPagedKVCache) {
        AssertInFastLLM(!cache.cudaData && (cache.dims.empty() || cache.dims[1] == 0),
                        "Naive paged attachment requires empty device storage.");
        cache.kvCacheFp8Indexer = manager.kvCacheFp8Indexer;
        cache.dataType = manager.dataType;
        cache.UpdateUnitSize();
        cache.dataDevice = DataDevice::CUDA;
        cache.dataDeviceIds = manager.dataDeviceIds;
        cache.isKVCache = cache.isPagedKVCache = true;
        cache.pagedKVCacheData = &manager;
        cache.pageLen = manager.pageLen;
        cache.Resize({1, 0, width});
    }
    AssertInFastLLM(cache.pagedKVCacheData == &manager, "Naive paged KV pool changed.");
    bool changed = !cache.cudaData;
    if (!cache.cudaData) {
        cache.cudaData = AllocateCudaWorkspace(tpKVPool.at(manager.dataDeviceIds[0]),
            (size_t)((capacity + manager.pageLen - 1) / manager.pageLen) * sizeof(int));
        if (!cache.cudaData) throw CudaWorkspaceExhausted("Naive KV page metadata pool exhausted.");
        cache.directMemory = true;
        cache.cudaDataBorrowed = false;
    }
    cache.expansionDims = {1, capacity, width};
    cache.strides = {(uint64_t)capacity * width, (uint64_t)width, 1};
    const int previous = cache.dims[1];
    const int pages = (end + cache.pageLen - 1) / cache.pageLen;
    bool copyTail = false;
    {
        std::lock_guard<std::mutex> guard(manager.pageIndexLocker);
        if (previous && previous % cache.pageLen && end > previous)
            copyTail = manager.pageRefCount[cache.pageIndex[(previous - 1) / cache.pageLen]] > 1;
        if (manager.FreePageCount() < std::max(0, pages - (int)cache.pageIndex.size()) + copyTail)
            throw CudaWorkspaceExhausted("Naive physical KV pages exhausted.");
    }
    if (copyTail) {
        int last = (previous - 1) / cache.pageLen;
        int old = cache.pageIndex[last], page = manager.GetUnusedPageIndex(true);
        size_t bytes = (size_t)cache.pageLen * width * manager.unitSize;
        FastllmCudaCopyFromDeviceToDevice((uint8_t *)manager.cudaData + page * bytes,
            (uint8_t *)manager.cudaData + old * bytes, bytes);
        cache.pageIndex[last] = page;
        manager.ReleasePageIndex(old);
        changed = true;
    }
    while ((int)cache.pageIndex.size() < pages) {
        cache.pageIndex.push_back(manager.GetUnusedPageIndex(true));
        changed = true;
    }
    if (changed) {
        FastllmCudaCopyFromHostToDevice(cache.cudaData, cache.pageIndex.data(), cache.pageIndex.size() * sizeof(int));
        tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
    }
    cache.lastPageLen = previous ? (previous - 1) % cache.pageLen + 1 : 0;
#endif
}

void NaiveN05FlashModel::TrimCache(Data &cache, int length) {
    if (cache.dims[1] <= length) return;
    Data suffix;
    Split(cache, 1, cache.dims[1] - length, cache.dims[1], suffix);
    cache.CopyFrom(suffix);
    cache.isKVCache = true;
}

NaiveN05FlashModel::NaiveN05FlashModel() {
    model_type = model_struct = "naive_n05_flash";
    canDoBatchForward = false;
    defaultChunkedPrefillSize = 8192;
    dataType = kvCacheDataType = moeAtype = DataType::BFLOAT16;
    pre_prompt = user_role = bot_role = history_sep = "";
    weight.embeddingNames.insert("model.embed_tokens.weight");
    weight.linearNames = {
        "lm_head.weight", "model.layers.*.self_attn.*_proj.weight",
        "model.layers.*.self_attn.indexer.wq.weight",
        "model.layers.*.self_attn.indexer.wk.weight",
        "model.layers.*.mlp.gate.weight",
        "model.layers.*.mlp*gate_proj.weight",
        "model.layers.*.mlp*up_proj.weight",
        "model.layers.*.mlp*down_proj.weight"
    };
}

NaiveN05FlashModel::~NaiveN05FlashModel() {
    ShutdownRuntime();
    activePrefixes.clear();
    prefixHistory.clear();
    prefixPool.reset();
    tpWorkers.Stop();
    tpServingState.reset();
    tpDecodeState.reset();
    tpVerifyState.reset();
    tpBatchState.reset();
    tpEp8MegaState.reset();
    for (auto &rank : tpPagedCaches) for (auto &pool : rank) {
        if (pool && pool->trieRoot) {
            pool->EvictTrieSubtree(pool->trieRoot);
            pool->trieRoot = nullptr;
        }
    }
}

void NaiveN05FlashModel::InitParams() {
    const bool mimo = weight.dicts["model_type"] == "mimo_v2" ||
        weight.dicts["architectures"].find("\"MiMoV2ForCausalLM\"") != std::string::npos ||
        weight.dicts["architectures"].find("\"MiMoV2DSAForCausalLM\"") != std::string::npos;
    if (mimo) {
        AssertInFastLLM(weight.dicts["enable_dsa"] == "true",
                        "MiMo-V2 currently requires enable_dsa=true (DSA checkpoint).");
        AssertInFastLLM(weight.dicts["attention_projection_layout"] == "split" &&
                        weight.dicts["add_full_attention_sink_bias"] != "true" &&
                        ConfigNumber(weight, "n_shared_experts", 0) == 0,
                        "MiMo-V2 DSA requires split QKV, no full-attention sink and no shared experts.");
    }
    basellm::InitParams();
    auto number = [&](const std::string &name, float fallback) {
        return ConfigNumber(weight, name, fallback);
    };
    auto attention = [&](const std::string &prefix) {
        return AttentionConfig{
            (int)number(prefix + "num_attention_heads", 64),
            (int)number(prefix + "num_key_value_heads", prefix.empty() ? 4 : 8),
            (int)number(prefix + "head_dim", 192),
            (int)number(prefix + "v_head_dim", 128),
            number(prefix + "rope_theta", prefix.empty() ? 1e7f : 1e4f)};
    };
    full = attention("");
    sliding = attention("swa_");
    head_dim = full.headDim;
    num_key_value_heads = full.kvHeads;
    rotary_dim = (int)(head_dim * number("partial_rotary_factor", 0.334f));
    slidingLayers = ConfigList(weight, "hybrid_layer_pattern", block_cnt);
    moeLayers = ConfigList(weight, "moe_layer_freq", block_cnt);
    window = number("sliding_window", 128);
    indexHeads = number("index_n_heads", 16);
    indexDim = number("index_head_dim", 128);
    indexTopK = number("index_top_k", 2048);
    partialRotary = number("partial_rotary_factor", 0.334f);
    valueScale = number("attention_value_scale", 1.0f);
    rms_norm_eps = number("layernorm_epsilon", 1e-5f);
    num_experts = number("n_routed_experts", 256);
    num_experts_per_tok = number("num_experts_per_tok", 8);
    routed_scaling_factor = number("routed_scaling_factor", 1.0f);
    norm_topk_prob = weight.dicts["norm_topk_prob"] != "false";
    max_positions = number("max_position_embeddings", 1048576);
    indexFp8 = weight.dicts["indexer_activation_dtype"] != "bf16";
    InitDraft();
    InitTensorParallel();
    canDoBatchForward = !draftEnabled && tpDevices.size() > 1;
    AssertInFastLLM(window > 0 && indexDim == 128 && indexTopK > 0 &&
                    full.heads % full.kvHeads == 0 && sliding.heads % sliding.kvHeads == 0 &&
                    (int)(full.headDim * partialRotary) <= indexDim &&
                    weight.dicts["scoring_func"] == "sigmoid" &&
                    number("n_group", 1) == 1 && number("topk_group", 1) == 1,
                    "Unsupported Naive-N0.5 attention/indexer/router configuration.");
    if (!useCustomMoeAtype) moeAtype = DataType::BFLOAT16;
    for (int layer = 0; layer < block_cnt; layer++) {
        const std::string ap = "model.layers." + std::to_string(layer) + ".self_attn.";
        weightMergeRules.push_back(WeightMergeRule({
            WeightMergeRuleSingle({ap + "q_proj.weight", ap + "k_proj.weight", ap + "v_proj.weight"},
                                  ap + "mergeqkv.weight", "linear"),
            WeightMergeRuleSingle({ap + "q_proj.bias", ap + "k_proj.bias", ap + "v_proj.bias"},
                                  ap + "mergeqkv.bias", "bias")}));
        if (!moeLayers[layer]) continue;
        for (int expert = 0; expert < num_experts; expert++) {
            std::string base = "model.layers." + std::to_string(layer) +
                               ".mlp.experts." + std::to_string(expert) + ".";
            weightMergeRules.push_back(WeightMergeRule({WeightMergeRuleSingle(
                {base + "gate_proj.weight", base + "up_proj.weight"},
                base + "gateup_proj.weight", "linearSwiglu")}));
            AddSpecialWeight(base + "gateup_proj.weight", tpEp8Mega ? "mimoEp8Expert" : "linearSwiglu", layer);
            AddSpecialWeight(base + "down_proj.weight", tpEp8Mega ? "mimoEp8Expert" : "linearColumn", layer);
            for (auto kind : {"gate", "up", "down"})
                moeLinears.insert(base + kind + "_proj.weight");
        }
    }
}

std::map<std::string, std::vector<std::pair<std::string, DataType>>>
NaiveN05FlashModel::GetTensorMap(const std::vector<std::string> &names) {
    auto result = basellm::GetTensorMap(names);
    const bool compactNvfp4 = std::any_of(names.begin(), names.end(), [](const std::string &name) {
        return name.find(".mlp.experts.") != std::string::npos &&
               (StringEndWith(name, ".weight_scale") || StringEndWith(name, ".weight_scale_2"));
    });
    auto usesCuda = [](const std::map<std::string, int> &devices) {
        return std::any_of(devices.begin(), devices.end(), [](const auto &device) {
            return device.first.rfind("cuda", 0) == 0 ||
                   device.first.rfind("multicuda", 0) == 0;
        });
    };
    // Keep original E4M3 scales in the packed CUDA source layout. Grouped
    // Marlin validates and replaces it once during warmup; unsupported
    // layouts retain the native kernels and independent gate/up globals.
    // CPU/NUMA keeps the original compact checkpoint layout.
    const bool cudaExperts = usesCuda(moeDeviceMap.empty() ? deviceMap : moeDeviceMap) ||
                            (moeDeviceLayers >= 0 && usesCuda(layeredMoeDeviceMap));
    const DataType nvfp4Type = cudaExperts ? DataType::NVFP4_BLOCK_16_E4M3_PACKED :
                                          DataType::NVFP4_BLOCK_16_E4M3;
    for (auto &name : names) {
        if (draftEnabled && (name.rfind("layers.", 0) == 0 ||
                name.rfind("markov_head.", 0) == 0 || name.rfind("confidence_head.", 0) == 0 ||
                name == "fc.weight" || name == "hidden_norm.weight" ||
                name == "norm.weight" || name == "mask_embedding")) {
            auto type = (name.find("norm.weight") != std::string::npos ||
                         name.rfind("confidence_head.", 0) == 0) ? DataType::FLOAT32 : DataType::BFLOAT16;
            result[name] = {{"dspark." + name, type}};
        } else if (compactNvfp4 && moeLinears.count(name)) {
            result[name] = {{name, nvfp4Type}};
        } else if (name.find(".mlp.gate.") != std::string::npos) {
            result[name] = {{name, DataType::FLOAT32}};
        } else if (name.find(".mlp.experts.") == std::string::npos &&
                   (weight.GetWeightType(name) == WeightType::LINEAR ||
                    weight.GetWeightType(name) == WeightType::EMBEDDING)) {
            result[name] = {{name, DataType::BFLOAT16}};
        }
    }
    return result;
}

int NaiveN05FlashModel::GetKVCacheRetainedTokens(int layer) const {
    return slidingLayers.at(layer) ? window - 1 : -1;
}

bool NaiveN05FlashModel::SupportsPrefillBatching() const {
    return !draftEnabled && tpDevices.size() > 1 && CanUseGPUForward() &&
           !GetKVCacheInCPU() && !kvMemConfig.enabled;
}

int NaiveN05FlashModel::Forward(
        const Data &inputIds, const Data &attentionMask, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &pastKeyValues,
        const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
        std::vector<float> *retLogits) {
    if (draftEnabled)
        return ForwardDraft(inputIds, positionIds, pastKeyValues, generationConfig, lastTokens, retLogits);
    auto selection = SelectLogits(generationConfig);
    Data logits = tpDevices.size() > 1
        ? ForwardTensorParallel(inputIds, positionIds, pastKeyValues, generationConfig, nullptr, &selection)
        : RunTarget(inputIds, positionIds, pastKeyValues, generationConfig, nullptr);
    if (isIntermediateChunkedPrefill) return 0;
    return SampleTarget(logits, pastKeyValues, generationConfig, lastTokens, retLogits, &selection);
}

std::vector<int> NaiveN05FlashModel::ForwardBatch(int batch, const Data &inputIds,
        const std::vector<Data *> &attentionMask, const std::vector<Data *> &positionIds,
        const std::vector<int> &seqLens, std::vector<std::pair<Data *, Data *>> &pastKeyValues,
        const std::vector<GenerationConfig> &configs, const LastTokensManager &lastTokens,
        std::vector<std::vector<float> *> *retLogits) {
    AssertInFastLLM(!draftEnabled && tpDevices.size() > 1 && batch > 0 &&
        (int)seqLens.size() == batch && (int)configs.size() == batch &&
        (int)positionIds.size() == batch && (int)pastKeyValues.size() == batch * block_cnt,
        "Naive batch forward requires ordinary TP inference and complete request descriptors.");
    TargetBatch requests{pastKeyValues, configs, seqLens, block_cnt};
    std::vector<float> packedPositions;
    packedPositions.reserve(inputIds.Count(0));
    auto selection = SelectLogits(configs[0]);
    for (int b = 0; b < batch; ++b) {
        AssertInFastLLM(seqLens[b] > 0 && positionIds[b] &&
            (b >= (int)attentionMask.size() || !attentionMask[b] || attentionMask[b]->dims.empty()),
            "Naive batch expects unpadded sequences with explicit positions.");
        for (int layer = 0; layer < block_cnt; ++layer)
            AssertInFastLLM(pastKeyValues[b * block_cnt + layer].first &&
                pastKeyValues[b * block_cnt + layer].second, "Naive batch has a null KV descriptor.");
        const auto &key = requests.Key(b, 0);
        const int past = key.dims.empty() ? 0 : key.dims[1];
        AssertInFastLLM((int64_t)past + seqLens[b] <= max_positions,
            "Naive batch request exceeds the context window.");
        const Data &source = *positionIds[b];
        AssertInFastLLM(source.Count(0) == seqLens[b], "Naive batch position count mismatch.");
        if (source.dataDevice == DataDevice::CPU && source.dataType == FLOAT32) {
            const float *values = (const float *)source.cpuData;
            packedPositions.insert(packedPositions.end(), values, values + seqLens[b]);
        } else {
            Data positions(source);
            ToDataType(positions, FLOAT32);
            positions.ToDevice(DataDevice::CPU);
            const float *values = (const float *)positions.cpuData;
            packedPositions.insert(packedPositions.end(), values, values + seqLens[b]);
        }
        auto other = SelectLogits(configs[b]);
        selection.cudaSampling = selection.cudaSampling && other.cudaSampling;
        if (selection.count != other.count || selection.greedy != other.greedy ||
            selection.invTemperature != other.invTemperature) selection.count = 0;
    }
    AssertInFastLLM(inputIds.dims == std::vector<int>({1, (int)packedPositions.size()}) &&
        inputIds.dataType == FLOAT32, "Naive batch input must pack the unpadded sequences in order.");
    Data positions(FLOAT32, inputIds.dims, DataDevice::CPU, packedPositions.data());
    // The batch descriptor references response-owned KV. No KV payload or
    // ownership is copied into the legacy single-sequence argument.
    std::vector<std::pair<Data, Data>> unused;
    Data logits = ForwardTensorParallel(inputIds, positions, unused, configs[0],
                                        nullptr, &selection, &requests);
    if (!selection.sampledTokens.empty()) return selection.sampledTokens;
    std::vector<int> tokens(batch);
    if (!selection.candidates.dims.empty()) {
        for (int b = 0; b < batch; ++b)
            tokens[b] = selection.greedy
                ? (int)((float *)selection.candidates.cpuData)[b * 2]
                : LLMSamplingOnly(selection.candidates, b, configs[b]);
        return tokens;
    }
    const int vocab = logits.dims.back();
    for (int b = 0; b < batch; ++b)
        if (configs[b].output_logits && retLogits && b < (int)retLogits->size() && (*retLogits)[b]) {
            const float *row = (const float *)logits.cpuData + (size_t)b * vocab;
            (*retLogits)[b]->assign(row, row + vocab);
        }
    ResetLogitsOfEOS(batch, &logits, pastKeyValues, configs);
    for (int b = 0; b < batch; ++b) {
        if (configs[b].IsSimpleGreedy()) {
            const float *row = (const float *)logits.cpuData + (size_t)b * vocab;
            // The owning TP result is already on CPU. Preserve CUDA Top1's
            // tie order without asking the executor to move a borrowed view.
            float best = -INFINITY;
            for (int id = 0; id < vocab; ++id)
                if (row[id] > -INFINITY && FastllmNaiveTop1Better(row[id], id, best, tokens[b])) {
                    best = row[id];
                    tokens[b] = id;
                }
        } else {
            LastTokensUnit empty;
            tokens[b] = LLMSampling(logits, b, configs[b],
                b < (int)lastTokens.units.size() ? lastTokens.units[b] : empty);
        }
    }
    return tokens;
}

Data NaiveN05FlashModel::RunTarget(
        const Data &inputIds, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &pastKeyValues, const GenerationConfig &config,
        TargetCapture *capture, int tpRank, const Data *embedding, TargetWorkspace *workspace,
        const TargetBatch *batch) {
#ifndef USE_CUDA
    ErrorInFastLLM("Naive-N0.5 currently requires the CUDA backend for attention.");
    return Data();
#else
    AssertInFastLLM(dataType == DataType::BFLOAT16 && FastllmNaiveKVType(kvCacheDataType),
                    "Naive-N0.5 requires BF16 activations and BF16 or FP8 E4M3 KV cache.");
    AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1 &&
                    (batch || (int)pastKeyValues.size() == block_cnt),
                    "Naive-N0.5 expects one unpadded sequence and a complete KV cache.");
    const bool tensorParallel = tpRank >= 0;
    const bool decodeWorkspace = workspace && workspace->capacity > 0;
    const bool verifying = capture && capture->verifying;
    const bool graphVerify = decodeWorkspace && verifying;
    const int gpu = tensorParallel ? tpDevices.at(tpRank) : -1;
    Data emptyWeight;
    auto localWeight = [&](const std::string &name) -> Data & {
        if (!tensorParallel) return weight[name];
        auto it = weight.weight.find(name);
        if (it == weight.weight.end() || it->second.dims.empty()) return emptyWeight;
        return *it->second.multiDeviceDatas.at(gpu);
    };
    auto &moeWeights = tensorParallel ? tpMoeWeights.at(tpRank) : this->moeWeights;
    auto &moeBiases = tensorParallel ? tpMoeBiases.at(tpRank) : this->moeBiases;
    size_t communication = 0;
    auto reduce = [&](Data &data, Data *residual = nullptr,
                      const Data *normWeight = nullptr, Data *normOutput = nullptr) {
        if (decodeWorkspace) {
            auto &state = *tpDecodeState;
            auto &rank = *state.ranks.at(tpRank);
            if (state.mode == TPDecodeState::Warm)
                rank.communicationPointers.push_back(data.cudaData);
            else if (communication >= rank.communicationPointers.size() ||
                     rank.communicationPointers[communication] != data.cudaData)
                rank.ok = false;
            ++communication;
        }
        if (!tensorParallel) return false;
#ifndef USE_ROCM
        if (residual && decodeWorkspace && !verifying) {
            normOutput->dataType = BFLOAT16;
            normOutput->Resize(residual->dims);
            normOutput->ToDevice(data.dataDevice, data.dataDeviceIds, false);
            normOutput->Allocate(false);
            if (FastllmCudaCustomAllReduceAddRMSNorm(data.cudaData, residual->cudaData,
                    (const float *)normWeight->cudaData, normOutput->cudaData,
                    data.Count(0), embed_dim, rms_norm_eps, data.dataType, gpu)) return true;
        }
#endif
        if (((batch && batch->Decode()) || (capture && capture->verifying)) && inputIds.dims[1] > 1) {
            if (!FastllmCudaCustomAllReduceRows(data.cudaData, data.cudaData,
                    data.Count(0), embed_dim, data.dataType, gpu)) {
                // An unsupported/disabled custom path must retain ordinary
                // single-token NCCL arithmetic, including its message size.
                for (int row = 0; row < inputIds.dims[1]; ++row) {
                    void *ptr = (uint8_t *)data.cudaData + (size_t)row * embed_dim * data.unitSize;
                    FastllmNcclAllReduce(ptr, ptr, embed_dim, data.dataType, gpu);
                }
            }
        } else FastllmNcclAllReduce(data.cudaData, data.cudaData, data.Count(0), data.dataType, gpu);
        return false;
    };
    int length = inputIds.dims[1];
    const bool shardMoe = tensorParallel && tpEp8MegaState && decodeWorkspace &&
        !verifying && length >= 8 && length % 8 == 0;
    // Small prefill chunks use MegaMoE's replicated (up to 64 rows) path.
    // Larger chunks dispatch contiguous token shards, including uneven tails.
    const bool shardPrefillRouter = tensorParallel && tpEp8MegaState &&
        !decodeWorkspace && !verifying && length > 64;
    const bool shardRouter = shardMoe || shardPrefillRouter;
    const int reserveCapacity = CacheReserveCapacity(config);
    const int previousExactThreshold = FastllmCudaGetLinearExactBatchThreshold();
    struct RestoreExactThreshold {
        int value;
        ~RestoreExactThreshold() { FastllmCudaSetLinearExactBatchThreshold(value); }
    } restoreExactThreshold{previousExactThreshold};
    // Keep the decode reduction order for paths honoring this threshold,
    // especially the FP32 router where rounding can change expert selection.
    // BF16 tensor-core projections can use cuBLAS and allow bounded
    // floating-point differences from independent single-row decoding.
    if ((batch && batch->Decode()) || (capture && capture->verifying))
        FastllmCudaSetLinearExactBatchThreshold(std::max(previousExactThreshold, length + 1));
    Data &firstKey = batch ? batch->Key(0, 0) : pastKeyValues[0].first;
    int pastLength = firstKey.dims.empty() ? 0 : firstKey.dims[1];
    AssertInFastLLM(!slidingLayers[0] && (batch || pastLength + length <= max_positions),
                    "Naive-N0.5 requires a DSA first layer and input within the context window.");
    if (moeWeights.empty()) {
        moeWeights.resize(block_cnt);
        moeBiases.resize(block_cnt);
        for (int layer = 0; layer < block_cnt; layer++) {
            if (!moeLayers[layer]) continue;
            // MergeMOE reserves the first pair for a shared expert.
            moeWeights[layer] = {nullptr, nullptr};
            for (int expert = 0; expert < num_experts; expert++) {
                std::string base = "model.layers." + std::to_string(layer) +
                                   ".mlp.experts." + std::to_string(expert) + ".";
                moeWeights[layer].push_back(&localWeight(base + "gateup_proj.weight"));
                moeWeights[layer].push_back(&localWeight(base + "down_proj.weight"));
            }
            moeBiases[layer].resize(moeWeights[layer].size(), nullptr);
        }
    }
    if (!tensorParallel) ApplyDeviceMap(deviceMap, 1, block_cnt);
    TargetWorkspace temporary;
    TargetWorkspace &buf = workspace ? *workspace : temporary;
    Data &hidden = buf.hidden;
    Data &normed = buf.normed;
    Data &q = buf.q;
    Data &k = buf.k;
    Data &v = buf.v;
    Data &qkv = buf.qkv;
    Data &attn = buf.attn;
    Data &projected = buf.projected;
    Data &routerInput = buf.routerInput;
    Data &router = buf.router;
    Data &expertIndex = buf.expertIndex;
    Data &expertScore = buf.expertScore;
    Data &w1 = buf.w1;
    Data &w2 = buf.w2;
    Data &w3 = buf.w3;
    Data &tempInput = buf.tempInput;
    Data &tempOutput = buf.tempOutput;
    // Attention has consumed projected before MoE starts. Reuse that full
    // output and its already-registered peer pointer tuple for decode gather.
    Data &moeOutput = shardMoe ? projected : buf.moeOutput;
    Data &moeInputTemp = buf.moeInputTemp;
    Data &moeOutputTemp = buf.moeOutputTemp;
    Data &positions = buf.positions;
    if (!decodeWorkspace) {
        positions.CopyFrom(positionIds);
        ToDataType(positions, DataType::FLOAT32);
    }
    if (embedding) hidden.CopyFrom(*embedding);
    // A persistent verification workspace keeps BF16 storage. Embedding's
    // FP32 intermediate cannot reuse that allocation at the same element count.
    else if (decodeWorkspace || (workspace && GetCudaEmbedding() && !GetLowMemMode()))
        EmbeddingDirect(inputIds, localWeight("model.embed_tokens.weight"), hidden);
    else Embedding(inputIds, localWeight("model.embed_tokens.weight"), hidden);
    ToDataType(hidden, DataType::BFLOAT16);
    auto norm = [&](Data &input, Data &weight, Data &output) {
        if (!decodeWorkspace) {
            KimiK3RMSNorm(input, weight, rms_norm_eps, output);
            return;
        }
        output.dataType = DataType::BFLOAT16;
        output.Resize(input.dims);
        output.ToDevice(DataDevice::CUDA, {gpu}, false);
        output.Allocate(false);
        if (!FastllmCudaKimiK3RMSNorm(input, weight, output, rms_norm_eps)) {
            if (FastllmCudaGraphIsCapturingFast()) FastllmCudaSetThreadError();
            else AssertInFastLLM(false, "Naive TP RMSNorm launch failed.");
        }
    };
    const bool fusePrefillNorm = !decodeWorkspace && !verifying && length > 1;
    bool prefillNormReady = false, prefillResidualSharded = false;
    for (int layer = 0; layer < block_cnt; layer++) {
        const bool shardPrefillNorm = fusePrefillNorm && shardPrefillRouter &&
            moeLayers[layer] && layer + 1 < block_cnt && moeLayers[layer + 1] &&
            (!capture || !capture->collectHidden);
        if (prefillResidualSharded && !shardPrefillNorm) {
            Data rows(BFLOAT16, {length, embed_dim});
            rows.FakeFrom(hidden, 0);
            FastllmCudaMimoEp8PrefillAllGather(*tpEp8MegaState, tpRank, rows);
            prefillResidualSharded = false;
        }
        if (!tensorParallel) ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
        std::string prefix = "model.layers." + std::to_string(layer);
        std::string ap = prefix + ".self_attn.";
        auto cfg = slidingLayers[layer] ? sliding : full;
        if (tensorParallel) {
            cfg.heads /= tpDevices.size();
            cfg.kvHeads = std::max(1, cfg.kvHeads / (int)tpDevices.size());
        }
        int rotaryDim = (int)(cfg.headDim * partialRotary);
        // This RMSNorm rounds the normalized activation before multiplying
        // the affine weight, matching the checkpoint's LlamaRMSNorm.
        if ((!decodeWorkspace && !prefillNormReady) || layer == 0)
            norm(hidden, localWeight(prefix + ".input_layernorm.weight"), normed);
        const auto merged = weight.weight.find(ap + "mergeqkv.weight");
        const bool mergedQkv = merged != weight.weight.end() && !merged->second.dims.empty();
        const bool batchRope = batch && (batch->Size() > 1 || buf.cacheTable) &&
            batch->Decode() && decodeWorkspace && !graphVerify &&
            (!capture || !capture->verifying);
        if (mergedQkv) {
            if (!batchRope)
                Linear(normed, localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"), qkv);
            // Q/K and V have different head widths. Read packed projection rows
            // directly in the RoPE/cache kernel instead of launching three Splits.
            for (auto item : {std::make_pair(&q, cfg.heads * cfg.headDim),
                              std::make_pair(&k, cfg.kvHeads * cfg.headDim),
                              std::make_pair(&v, cfg.kvHeads * cfg.valueDim)}) {
                item.first->dataType = normed.dataType;
                item.first->Resize({1, length, item.second});
                item.first->ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
                item.first->Allocate(false);
            }
        } else {
            Linear(normed, localWeight(ap + "q_proj.weight"), localWeight(ap + "q_proj.bias"), q);
            Linear(normed, localWeight(ap + "k_proj.weight"), localWeight(ap + "k_proj.bias"), k);
            Linear(normed, localWeight(ap + "v_proj.weight"), localWeight(ap + "v_proj.bias"), v);
        }
        AssertInFastLLM(q.dataDevice == DataDevice::CUDA,
                        "Naive-N0.5 attention requires --device cuda.");
        positions.ToDevice(q.dataDevice, q.dataDeviceIds);
        // Prepare all independent caches/index lists first, then attend to the
        // complete decode batch in one call. Prefill and verification retain
        // their causal multi-query paths.
        const bool batchedDSA = !slidingLayers[layer] && !graphVerify &&
            (!capture || !capture->verifying) && indexTopK <= 2048 && cfg.headDim <= 256 &&
            (cfg.valueDim == 128 || cfg.valueDim == 256) &&
            (batch ? std::all_of(batch->lengths.begin(), batch->lengths.end(),
                [](int count) { return count == 1; }) : length == 1);
        std::vector<FastllmNaiveDSARow> dsaRows;
        if (batchedDSA) dsaRows.reserve(batch ? batch->Size() : 1);
        const bool batchedSWA = slidingLayers[layer] && decodeWorkspace && !graphVerify &&
            batch && batch->Decode() && tpSwaRingCache;
        std::vector<FastllmNaiveSWARow> swaRows;
        if (batchedSWA) swaRows.reserve(batch->Size());
        auto prepareIndexKey = [&](Data &normed, TargetWorkspace &buf, bool alreadyProjected = false) {
            if (slidingLayers[layer]) return;
            Data &indexKey = buf.indexKey;
            const std::string ip = ap + "indexer.";
            if (!alreadyProjected) Linear(normed, localWeight(ip + "wk.weight"), Data(), indexKey);
            if (FastllmCudaNaiveIndexerKeyNorm(indexKey,
                    localWeight(ip + "k_norm.weight"), localWeight(ip + "k_norm.bias"))) return;
            // LayerNorm accumulates in FP32; the generic CUDA operation does
            // not accept BF16 storage, so round only its final result.
            if (decodeWorkspace) {
                ToDataType(indexKey, buf.indexKeyFloat, DataType::FLOAT32);
                LayerNorm(buf.indexKeyFloat, localWeight(ip + "k_norm.weight"),
                          localWeight(ip + "k_norm.bias"), -1, buf.indexKeyFloat);
                ToDataType(buf.indexKeyFloat, indexKey, DataType::BFLOAT16);
            } else {
                ToDataType(indexKey, DataType::FLOAT32);
                LayerNorm(indexKey, localWeight(ip + "k_norm.weight"), localWeight(ip + "k_norm.bias"), -1, indexKey);
                ToDataType(indexKey, DataType::BFLOAT16);
            }
        };
        bool batchRopePrepared = false;
        bool indexProjectionPrepared = false;
        bool batchNeedsIndex = false;
        const void *const *layerCache = buf.cacheTable ? buf.cacheTable + (size_t)layer * buf.tableRows * 4 : nullptr;
        const unsigned char *layerMaps = buf.cacheMaps ? buf.cacheMaps + (size_t)layer * buf.tableRows * FastllmNaiveDSAMapBytes : nullptr;
        if (batch) {
            while ((int)buf.sequences.size() < batch->Size())
                buf.sequences.emplace_back(new TargetWorkspace());
        }
        if (batchRope) {
            std::vector<FastllmNaiveRopeCacheRow> caches;
            caches.reserve(batch->Size());
            // Project/normalize the complete batch, then borrow individual K rows.
            // Cache descriptors retain the addresses, not these temporary views.
            batchNeedsIndex = !slidingLayers[layer] &&
                std::any_of(buf.sequences.begin(), buf.sequences.begin() + batch->Size(),
                    [&](const std::unique_ptr<TargetWorkspace> &part) { return part->capacity > indexTopK; });
#ifndef USE_ROCM
            if (batchNeedsIndex) {
                const std::string ip = ap + "indexer.";
                // O projection is not live yet. Reuse its full-width buffer for
                // the packed GEMM; graph shapes already share this allocation.
                indexProjectionPrepared = FastllmCudaIndexerProjection(normed,
                    localWeight(ip + "wq.weight"), localWeight(ip + "wk.weight"),
                    localWeight(ip + "weights_proj.weight"), buf.indexQ, buf.indexKey,
                    buf.indexWeights, projected);
            }
#endif
            prepareIndexKey(normed, buf, indexProjectionPrepared);
            std::vector<Data> indexKeys(batch->Size());
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                if (!slidingLayers[layer]) {
                    indexKeys[sequence].Resize({1, 1, indexDim});
                    indexKeys[sequence].FakeFrom(buf.indexKey,
                        (size_t)sequence * indexDim * buf.indexKey.unitSize);
                }
                caches.push_back({batch->Key(sequence, layer).multiDeviceDatas.at(gpu),
                    batch->Value(sequence, layer).multiDeviceDatas.at(gpu),
                    slidingLayers[layer] ? nullptr : &indexKeys[sequence],
                    layerCache ? layerCache + sequence * 4 : nullptr});
            }
#ifndef USE_ROCM
            if (mergedQkv)
                batchRopePrepared = FastllmCudaNaiveBatchQkvRopeAppendCache(normed,
                    localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"),
                    q, k, v, positions, caches, buf.liveKeys, cfg.heads, cfg.kvHeads,
                    cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale,
                    slidingLayers[layer] ? window : 0, slidingLayers[layer] && tpSwaRingCache, projected);
#endif
            if (!batchRopePrepared) {
                if (mergedQkv)
                    Linear(normed, localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"), qkv);
                batchRopePrepared = FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions,
                    caches, buf.liveKeys, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                    rotaryDim, cfg.theta, valueScale, slidingLayers[layer] ? window : 0,
                    mergedQkv ? &qkv : nullptr, slidingLayers[layer] && tpSwaRingCache);
            }
        }
        auto prepareIndexQuery = [&](Data &normed, Data &positions, Data &indexQ, Data &indexWeights) {
            const std::string ip = ap + "indexer.";
            Linear(normed, localWeight(ip + "wq.weight"), Data(), indexQ);
            FastllmCudaNaiveRope(indexQ, positions, indexHeads, indexDim, rotaryDim, cfg.theta);
            Linear(normed, localWeight(ip + "weights_proj.weight"), Data(), indexWeights);
            Mul(indexWeights, 1.0f / std::sqrt((float)indexHeads), indexWeights);
        };
        const bool batchIndexPrepared = batchRopePrepared && batchNeedsIndex;
        Data &batchIndexQ = buf.indexQ, &batchIndexWeights = buf.indexWeights;
        if (batchIndexPrepared) {
            if (indexProjectionPrepared)
                FastllmCudaNaiveRope(batchIndexQ, positions, indexHeads, indexDim, rotaryDim, cfg.theta);
            else prepareIndexQuery(normed, positions, batchIndexQ, batchIndexWeights);
        }
        bool batchIndexerPrepared = false;
        if (batchIndexPrepared) {
            std::vector<FastllmNaiveIndexerRow> rows;
            rows.reserve(batch->Size());
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                auto &part = *buf.sequences[sequence];
                if (part.capacity > indexTopK)
                    rows.push_back({batch->Key(sequence, layer).multiDeviceDatas.at(gpu),
                        &part.decode, &part.indices, sequence, part.capacity,
                        layerCache ? layerCache + sequence * 4 : nullptr});
            }
            const int ranks = tensorParallel ? (int)tpDevices.size() : 1;
            const bool contextParallel = !verifying && decodeWorkspace && ranks > 1 &&
                rows.size() >= 4 && buf.capacity >= std::max(65536, ranks * 4096) &&
                FastllmCudaNaiveIndexerContextSupported(tpDevices) &&
                buf.decode.indexScores.Count(0) >= (size_t)ranks * rows.size() * indexTopK * 2;
            Data &exchange = buf.decode.indexScores;
            batchIndexerPrepared = FastllmCudaNaiveBatchDecodeIndexer(batchIndexQ,
                batchIndexWeights, buf.liveKeys, rows, indexFp8,
                contextParallel ? tpRank : 0, contextParallel ? ranks : 1,
                contextParallel ? &exchange : nullptr);
            if (batchIndexerPrepared && contextParallel) {
                const int part = rows.size() * indexTopK * 2;
                // AllGather is a bitwise copy. FLOAT32 selects the existing
                // 32-bit custom transport; it does not interpret score bits.
                if (!FastllmCudaCustomAllGatherRows(exchange.cudaData, ranks * part,
                        indexTopK * 2, FLOAT32, gpu))
                    AssertInFastLLM(FastllmNcclAllGather((int *)exchange.cudaData + tpRank * part,
                        exchange.cudaData, part, INT32, gpu), "Indexer context all-gather failed.");
                FastllmCudaNaiveMergeDecodeIndexer(buf.liveKeys, rows, exchange, ranks);
            }
        }
        auto attend = [&](Data &normed, Data &q, Data &k, Data &v, Data &qkv,
                          Data &positions, Data &cacheKey, Data &cacheValue,
                          TargetWorkspace &buf, int length, int globalPast, int reserveCapacity, Data &attn, int sequence = 0) {
            Data &packed = buf.packed, &indexKey = buf.indexKey, &indices = buf.indices;
            const bool sharedSwa = tensorParallel && slidingLayers[layer] &&
                !decodeWorkspace && !verifying && !tpSwaPrefillWorkspaces.empty();
            Data &pastKey = sharedSwa ? tpSwaPrefillWorkspaces[tpRank].first : cacheKey;
            Data &pastValue = sharedSwa ? tpSwaPrefillWorkspaces[tpRank].second : cacheValue;
            pastKey.kvCacheFp8Indexer = fp8IndexerCache && !slidingLayers[layer];
            int localPast = cacheKey.dims.empty() ? 0 : cacheKey.dims[1];
            const bool ringCache = tensorParallel && slidingLayers[layer] && tpSwaRingCache;
            if (sharedSwa) {
                pastKey.Resize({1, localPast, cfg.kvHeads * cfg.headDim});
                pastValue.Resize({1, localPast, cfg.kvHeads * cfg.valueDim});
                CopySwaWindow(cacheKey, cacheValue, pastKey, pastValue,
                    ringCache ? (globalPast - localPast) % window : 0, 0, localPast);
            }
            // Graph replay reads/writes the ring directly. Eager prefill/fallback
            // uses the existing contiguous kernels and restores the ring once.
            // Verify was linearized outside capture by ForwardTensorParallel.
            if (ringCache && !decodeWorkspace && !verifying && !sharedSwa)
                FastllmCudaNaiveSwaCacheLayout(pastKey, pastValue, globalPast, false);
            const bool ringDecode = ringCache && decodeWorkspace && !graphVerify;
            // Sliding layers retain only window-1 rows between chunks. Keep their
            // reservation bounded even when the full request is very long.
            const int layerCapacity = slidingLayers[layer]
                ? (int)std::min<int64_t>(reserveCapacity, (int64_t)window - 1 + length)
                : reserveCapacity;
            if (!batchRopePrepared) prepareIndexKey(normed, buf);
            Data noIndexKey, noLiveKeys;
            bool fusedCache = batchRopePrepared || ((decodeWorkspace || (!tensorParallel && length <= 8 && localPast > 0)) &&
                FastllmCudaNaiveRopeAppendCache(q, k, v, slidingLayers[layer] ? noIndexKey : indexKey,
                    positions, pastKey, pastValue, decodeWorkspace ? buf.liveKeys : noLiveKeys, cfg.heads, cfg.kvHeads,
                    cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale,
                    slidingLayers[layer] ? window : 0, mergedQkv ? &qkv : nullptr, ringDecode));
            if (fusedCache && !decodeWorkspace) {
                pastKey.Resize({1, localPast + length, pastKey.dims[2]});
                pastValue.Resize({1, localPast + length, pastValue.dims[2]});
                if (pastKey.isPagedKVCache) {
                    pastKey.lastPageLen = (localPast + length - 1) % pastKey.pageLen + 1;
                    pastValue.lastPageLen = pastKey.lastPageLen;
                }
            }
            if (!fusedCache) {
                FastllmCudaNaiveRopeQKScaleV(q, k, v, positions, cfg.heads, cfg.kvHeads,
                    cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale, mergedQkv ? &qkv : nullptr);
                if (!slidingLayers[layer]) {
                    FastllmCudaNaiveRope(indexKey, positions, 1, indexDim, rotaryDim, cfg.theta);
                    Cat(k, indexKey, 2, packed);
                } else {
                    packed.CopyFrom(k);
                }
                Data *cacheInput = &packed;
                if (pastKey.kvCacheFp8Indexer) {
                    FastllmCudaNaivePackIndexer(packed, buf.packedIndexer);
                    cacheInput = &buf.packedIndexer;
                }
                if (decodeWorkspace) {
                    if (graphVerify) FastllmCudaNaiveAppendVerifyCache(pastKey, pastValue, *cacheInput, v,
                        buf.liveKeys, slidingLayers[layer] ? window : 0);
                    else FastllmCudaNaiveAppendDecodeCache(pastKey, pastValue, *cacheInput, v,
                        buf.liveKeys, slidingLayers[layer] ? window : 0, ringDecode);
                } else {
                    AppendCache(pastKey, *cacheInput, kvCacheDataType, layerCapacity);
                    AppendCache(pastValue, v, kvCacheDataType, layerCapacity);
                }
            }
            Data noIndices;
            Data *selected = &noIndices;
            int balancedIndexRanks = 0;
            if (!slidingLayers[layer] && (decodeWorkspace ? buf.capacity : pastKey.dims[1]) > indexTopK) {
                if (!batchIndexerPrepared) {
                    Data rowIndexQ, rowIndexWeights;
                    Data &indexQ = batchIndexPrepared ? rowIndexQ : buf.indexQ;
                    Data &indexWeights = batchIndexPrepared ? rowIndexWeights : buf.indexWeights;
                    if (batchIndexPrepared) {
                        indexQ.Resize({1, 1, indexHeads * indexDim});
                        indexQ.FakeFrom(batchIndexQ, (size_t)sequence * indexHeads * indexDim * batchIndexQ.unitSize);
                        indexWeights.Resize({1, 1, indexHeads});
                        indexWeights.FakeFrom(batchIndexWeights, (size_t)sequence * indexHeads * batchIndexWeights.unitSize);
                    } else prepareIndexQuery(normed, positions, indexQ, indexWeights);
                    if (decodeWorkspace) {
                        if (graphVerify) FastllmCudaNaiveGraphVerifyIndexer(indexQ, indexWeights, pastKey,
                            buf.liveKeys, buf.capacity, indexFp8, buf.decode, indices);
                        else FastllmCudaNaiveDecodeIndexer(indexQ, indexWeights, pastKey,
                            buf.liveKeys, buf.capacity, indexFp8, buf.decode, indices,
                            layerCache ? layerCache + sequence * 4 : nullptr);
                    } else {
                        const int ranks = tensorParallel ? (int)tpDevices.size() : 1;
                        if (capture && capture->verifying && length <= 8)
                            FastllmCudaNaiveVerifyIndexer(indexQ, indexWeights, pastKey, indexHeads,
                                indexDim, localPast, indexTopK, indexFp8, indices);
                        else if (!verifying && tpEp8MegaState && ranks > 1 && length % (ranks * 64) == 0 &&
                                 (int64_t)(length / ranks) * pastKey.dims[1] >= 1024 * 1024) {
                            // Indexer Q/K are replicated. Score disjoint query
                            // tiles and gather only the selected integer indices.
                            // Pair an early and a late range to balance causal
                            // work while preserving 64-row tiles and the MMA
                            // dispatch threshold, so score arithmetic is unchanged.
                            const int rows = length / ranks;
                            const bool balanced = length % (ranks * 128) == 0 &&
                                (ranks == 2 || ranks == 4 || ranks == 8) &&
                                (int64_t)(rows / 2) * pastKey.dims[1] >= 1024 * 1024;
                            const int parts = balanced ? 2 : 1, partRows = rows / parts;
                            indices.dataType = INT32;
                            indices.Resize({length, indexTopK});
                            indices.ToDevice(indexQ.dataDevice, indexQ.dataDeviceIds, false);
                            indices.Allocate(false);
                            for (int part = 0; part < parts; ++part) {
                                const int first = part == 0 ? tpRank * partRows :
                                    length - (tpRank + 1) * partRows;
                                Data localQ, localWeights, localIndices;
                                localQ.Resize({1, partRows, indexHeads * indexDim});
                                localQ.FakeFrom(indexQ, (size_t)first * indexHeads * indexDim * indexQ.unitSize);
                                localWeights.Resize({1, partRows, indexHeads});
                                localWeights.FakeFrom(indexWeights, (size_t)first * indexHeads * indexWeights.unitSize);
                                localIndices.Resize({partRows, indexTopK});
                                localIndices.FakeFrom(indices,
                                    (size_t)(tpRank * rows + part * partRows) * indexTopK * sizeof(int));
                                FastllmCudaNaiveIndexer(localQ, localWeights, pastKey, indexHeads, indexDim,
                                    localPast + first, indexTopK, indexFp8, localIndices);
                            }
                            FastllmCudaMimoEp8PrefillAllGather(*tpEp8MegaState, tpRank, indices);
                            if (balanced) balancedIndexRanks = ranks;
                        }
                        else FastllmCudaNaiveIndexer(indexQ, indexWeights, pastKey, indexHeads, indexDim,
                                                    localPast, indexTopK, indexFp8, indices);
                    }
                }
                selected = &indices;
            }
            Data &sink = localWeight(ap + "attention_sink_bias");
            if (!sink.dims.empty()) {
                ToDataType(sink, DataType::FLOAT32);
                sink.ToDevice(q.dataDevice, q.dataDeviceIds);
            }
            if (batchedDSA) {
                FastllmNaiveDSARow row;
                row.fp8 = pastKey.dataType == FP8_E4M3;
                row.tensorMaps = layerMaps ? layerMaps + (size_t)sequence * FastllmNaiveDSAMapBytes : nullptr;
                row.cache = layerCache ? layerCache + sequence * 4 : nullptr;
                row.key = pastKey.isPagedKVCache ? pastKey.pagedKVCacheData->cudaData : pastKey.cudaData;
                row.value = pastValue.isPagedKVCache ? pastValue.pagedKVCacheData->cudaData : pastValue.cudaData;
                if (pastKey.isPagedKVCache) {
                    row.keyPages = (const int *)pastKey.cudaData;
                    row.valuePages = (const int *)pastValue.cudaData;
                    row.pageLen = pastKey.pageLen;
                    row.poolTokens = pastKey.pagedKVCacheData->maxPages * pastKey.pageLen;
                }
                row.indices = selected->dims.empty() ? nullptr : (const int *)selected->cudaData;
                row.liveKeys = decodeWorkspace ? (const int *)buf.liveKeys.cudaData : nullptr;
                row.length = localPast + length;
                row.capacity = decodeWorkspace ? buf.capacity : pastKey.dims[1];
                row.keyStride = pastKey.dims[2];
                row.selectedCount = row.indices ? selected->dims.back() : 0;
                dsaRows.push_back(row);
            } else if (batchedSWA) {
                swaRows.push_back({pastKey.cudaData, pastValue.cudaData, layerCache ? layerCache + sequence * 4 : nullptr,
                                   pastKey.dataType == FP8_E4M3});
            } else if (decodeWorkspace) {
                if (graphVerify) FastllmCudaNaiveGraphVerifyAttention(q, pastKey, pastValue, *selected, sink,
                    buf.liveKeys, buf.capacity, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                    slidingLayers[layer] ? window : 0, buf.decode, attn);
                else FastllmCudaNaiveDecodeAttention(q, pastKey, pastValue, *selected, sink,
                    buf.liveKeys, buf.capacity, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                    slidingLayers[layer] ? window : 0, buf.decode, attn, ringDecode);
                if (slidingLayers[layer] && !graphVerify && !ringDecode)
                    FastllmCudaNaiveTrimDecodeCache(pastKey, pastValue, buf.liveKeys, window);
            } else {
                if (verifying && length <= 8)
                    FastllmCudaNaiveVerifyAttention(q, pastKey, pastValue, *selected, sink,
                        cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                        localPast, slidingLayers[layer] ? window : 0, attn);
                else if (!balancedIndexRanks || !FastllmCudaNaiveBalancedAttention(
                        q, pastKey, pastValue, *selected, sink, cfg.heads, cfg.kvHeads,
                        localPast, balancedIndexRanks, attn)) {
                    if (balancedIndexRanks) FastllmCudaNaiveRestoreIndexerRows(*selected, balancedIndexRanks);
                    FastllmCudaNaiveAttention(q, pastKey, pastValue, *selected, sink,
                        cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                        localPast, slidingLayers[layer] ? window : 0, attn);
                }
                if (slidingLayers[layer] && capture && !capture->prefixes.empty()) {
                    for (const auto &snapshot : capture->prefixes[sequence]) {
                        int rows = std::min(snapshot->length, window - 1);
                        CopyPrefixWindow(*snapshot, tpRank, layer, pastKey, pastValue,
                            localPast + snapshot->length - globalPast - rows, false);
                    }
                }
                if (sharedSwa) {
                    const int rows = std::min(localPast + length, window - 1);
                    CopySwaWindow(pastKey, pastValue, cacheKey, cacheValue,
                        localPast + length - rows,
                        ringCache ? (globalPast + length - rows) % window : 0, rows);
                    cacheKey.Resize({1, rows, cfg.kvHeads * cfg.headDim});
                    cacheValue.Resize({1, rows, cfg.kvHeads * cfg.valueDim});
                } else if (slidingLayers[layer] && !verifying)
                    FastllmCudaNaiveTrimCache(pastKey, pastValue, window - 1);
                if (ringCache && !verifying && !sharedSwa)
                    FastllmCudaNaiveSwaCacheLayout(pastKey, pastValue, globalPast + length, true);
            }
        };
        if (batch) {
            attn.dataType = BFLOAT16;
            attn.Resize({1, length, cfg.heads * cfg.valueDim});
            attn.ToDevice(q.dataDevice, q.dataDeviceIds, false);
            attn.Allocate(false);
            int offset = 0;
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                const int count = batch->lengths[sequence];
                auto view = [&](Data &dst, Data &src, int width) {
                    dst.Resize({1, count, width});
                    dst.FakeFrom(src, (size_t)offset * width * src.unitSize);
                };
                Data rowNorm, rowQ, rowK, rowV, rowQkv, rowPositions, rowAttention;
                view(rowNorm, normed, embed_dim);
                view(rowQ, q, cfg.heads * cfg.headDim);
                view(rowK, k, cfg.kvHeads * cfg.headDim);
                view(rowV, v, cfg.kvHeads * cfg.valueDim);
                if (mergedQkv && !batchRopePrepared) view(rowQkv, qkv, qkv.dims.back());
                view(rowAttention, attn, cfg.heads * cfg.valueDim);
                rowPositions.Resize({1, count});
                rowPositions.FakeFrom(positions, (size_t)offset * sizeof(float));
                auto &part = *buf.sequences[sequence];
                if (decodeWorkspace) {
                    part.liveKeys.Resize({1});
                    part.liveKeys.FakeFrom(buf.liveKeys, sequence * sizeof(int));
                }
                attend(rowNorm, rowQ, rowK, rowV, rowQkv, rowPositions,
                       *batch->Key(sequence, layer).multiDeviceDatas.at(gpu),
                       *batch->Value(sequence, layer).multiDeviceDatas.at(gpu), part, count,
                       batch->Key(sequence, 0).dims.empty() ? 0 : batch->Key(sequence, 0).dims[1],
                       CacheReserveCapacity(batch->configs[sequence]), rowAttention,
                       sequence);
                offset += count;
            }
        } else {
            auto &pastKey = tensorParallel ? *pastKeyValues[layer].first.multiDeviceDatas.at(gpu) : pastKeyValues[layer].first;
            auto &pastValue = tensorParallel ? *pastKeyValues[layer].second.multiDeviceDatas.at(gpu) : pastKeyValues[layer].second;
            attend(normed, q, k, v, qkv, positions, pastKey, pastValue, buf, length, pastLength, reserveCapacity, attn);
        }
        if (batchedDSA)
            FastllmCudaNaiveBatchDSADecode(q, localWeight(ap + "attention_sink_bias"), dsaRows,
                cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim, buf.decode, attn);
        if (batchedSWA)
            FastllmCudaNaiveBatchSWADecode(q, localWeight(ap + "attention_sink_bias"), swaRows,
                buf.liveKeys, cfg.heads, cfg.kvHeads, attn);
        Linear(attn, localWeight(ap + "o_proj.weight"), Data(), projected);
        Data &postAttentionNorm = localWeight(prefix + ".post_attention_layernorm.weight");
        bool attentionNormReady = false;
        if (shardPrefillNorm && length >= 1024 && length % 8 == 0) {
            // Only this rank's token shard feeds the following EP Router/MoE.
            // Avoid distributing reduced rows that are never consumed here.
            FastllmCudaMimoEp8PrefillReduceScatter(*tpEp8MegaState, tpRank, projected);
        } else {
            attentionNormReady = reduce(projected, decodeWorkspace ? &hidden : nullptr,
                &postAttentionNorm, &normed);
        }
        if (shardPrefillNorm) {
            const int first = (int)((int64_t)length * tpRank / 8);
            const int rows = (int)((int64_t)length * (tpRank + 1) / 8) - first;
            normed.dataType = BFLOAT16;
            normed.Resize(hidden.dims);
            normed.ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false);
            normed.Allocate(false);
            Data localHidden(BFLOAT16, {rows, embed_dim}), localBranch, localNorm;
            localHidden.FakeFrom(hidden, (size_t)first * embed_dim * sizeof(uint16_t));
            localBranch.FakeFrom(projected, (size_t)first * embed_dim * sizeof(uint16_t));
            localNorm.FakeFrom(normed, (size_t)first * embed_dim * sizeof(uint16_t));
            FastllmCudaNaiveAddDecodeRMSNorm(localHidden, localBranch,
                postAttentionNorm, rms_norm_eps, localNorm);
            prefillResidualSharded = true;
        } else if (decodeWorkspace || fusePrefillNorm) {
            if (!attentionNormReady)
                FastllmCudaNaiveAddDecodeRMSNorm(hidden, projected,
                    postAttentionNorm, rms_norm_eps, normed);
        } else {
            AddTo(hidden, projected);
            norm(hidden, postAttentionNorm, normed);
        }
        Data *mlpNormWeight = nullptr, *mlpNormOutput = nullptr;
        if (decodeWorkspace || (fusePrefillNorm && layer + 1 < block_cnt)) {
            const bool lastLayer = layer + 1 == block_cnt;
            const std::string name = lastLayer ? "model.norm.weight" :
                "model.layers." + std::to_string(layer + 1) + ".input_layernorm.weight";
            mlpNormWeight = &localWeight(name);
            mlpNormOutput = lastLayer ? &buf.last : &normed;
        }
        auto addMlpResidual = [&](Data &branch, bool needsReduce) {
            if (!decodeWorkspace) {
                if (needsReduce) reduce(branch);
                if (fusePrefillNorm && mlpNormWeight)
                    FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, *mlpNormWeight,
                        rms_norm_eps, *mlpNormOutput);
                else AddTo(hidden, branch);
                return;
            }
            if (!needsReduce || !reduce(branch, &hidden, mlpNormWeight, mlpNormOutput))
                FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, *mlpNormWeight,
                    rms_norm_eps, *mlpNormOutput);
        };
        if (!moeLayers[layer]) {
            // Dense and routed-expert scratch have independent lifetimes.
            Data &w1 = workspace ? buf.denseGate : buf.w1;
            Data &w2 = workspace ? buf.denseDown : buf.w2;
            Data &w3 = workspace ? buf.denseUp : buf.w3;
            Linear(normed, localWeight(prefix + ".mlp.gate_proj.weight"), Data(), w1);
            Linear(normed, localWeight(prefix + ".mlp.up_proj.weight"), Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, localWeight(prefix + ".mlp.down_proj.weight"), Data(), w2);
            addMlpResidual(w2, true);
        } else {
            // EP ranks route only their contiguous token shard, borrowing
            // the normalized activation until MegaMoE finishes reading it.
            Data routedInput;
            if (shardRouter) {
                const int first = (int)((int64_t)length * tpRank / 8);
                const int end = (int)((int64_t)length * (tpRank + 1) / 8);
                routedInput.Resize({end - first, embed_dim});
                routedInput.FakeFrom(normed, (size_t)first * embed_dim * normed.unitSize);
            }
            Data &routerSource = shardRouter ? routedInput : normed;
            Data &routerWeight = localWeight(prefix + ".mlp.gate.weight");
            bool hasRouterProbabilities =
                FastllmCudaNaiveRouterSigmoid(routerSource, routerWeight, router);
            if (!hasRouterProbabilities) {
                Data localRouterInput;
                if (shardPrefillRouter) {
                    // Share one backing allocation with the prefill arena;
                    // only the logical local rows are converted/projected.
                    routerInput.dataType = FLOAT32;
                    routerInput.UpdateUnitSize();
                    const size_t rowBytes = (size_t)embed_dim * sizeof(float);
                    const size_t arenaBytes = FastllmCudaMimoEp8PrefillWorkspaceBytes(length);
                    const int reserveRows = std::max<size_t>(length,
                        (arenaBytes + rowBytes - 1) / rowBytes);
                    routerInput.Resize({reserveRows, embed_dim});
                    routerInput.ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
                    routerInput.Allocate(false);
                    routerInput.Resize({length, embed_dim});
                    localRouterInput.FakeFrom(routerInput, 0);
                }
                Data &fp32Input = shardPrefillRouter ? localRouterInput : routerInput;
                ToDataType(routerSource, fp32Input, DataType::FLOAT32);
                Linear(fp32Input, routerWeight, Data(), router);
            }
            Data &routerBias = localWeight(prefix + ".mlp.gate.e_score_correction_bias");
            auto &executor = *(Executor *)GetExecutor();
            bool fusedRouter = false;
            // Small decode/verify batches already include sigmoid in their shared
            // router projection. Other multi-row paths fuse it with selection.
            if (!hasRouterProbabilities && length > 1 && router.dataDevice == DataDevice::CUDA) {
                DataDict routerData = {{"logits", &router}, {"index", &expertIndex},
                                       {"score", &expertScore}, {"gateBias", &routerBias}};
                FloatDict routerFloats = {{"routeScale", routed_scaling_factor}};
                IntDict routerInts = {{"topk", num_experts_per_tok}, {"needNorm", norm_topk_prob ? 1 : 0}};
                fusedRouter = executor.CanRunOnFirstDevice(
                    "FusedSigmoidSelectExpert", routerData, routerFloats, routerInts);
                if (fusedRouter)
                    executor.Run("FusedSigmoidSelectExpert", routerData, routerFloats, routerInts);
            }
            if (!fusedRouter) {
                if (!hasRouterProbabilities) Sigmoid(router, router);
                SelectExpert(router, expertIndex, expertScore, num_experts_per_tok,
                             norm_topk_prob, routed_scaling_factor, &routerBias);
            }
            normed.Reshape({length, embed_dim});
            if (!tensorParallel) ApplyMoeDeviceMapForLayer(layer);
            bool moeNormReady = false;
            if (tensorParallel && tpEp8MegaState) {
                moeNormReady = FastllmCudaRunMimoEp8Mega(*tpEp8MegaState, tpRank, layer, normed,
                    expertIndex, expertScore, moeOutput, decodeWorkspace,
                    (shardMoe || shardPrefillNorm) ? &hidden : nullptr,
                    mlpNormWeight, mlpNormOutput, rms_norm_eps,
                    !decodeWorkspace && !hasRouterProbabilities ? &routerInput : nullptr,
                    decodeWorkspace && !verifying ? (const int *)buf.liveKeys.cudaData : nullptr);
            } else if (executor.firstDevice.find("numa") == 0 &&
                (moeWeights[layer][2]->dataType == DataType::FP8_E4M3 ||
                 moeWeights[layer][2]->dataType == DataType::FP8_E4M3_BLOCK_128)) {
                // The released checkpoint uses W8A8 and rounds each expert
                // projection/activation/route product to BF16. Request that
                // arithmetic explicitly; ordinary NUMA MoE uses W8A16.
                executor.Run("MergeMOE", {
                    {"input", &normed}, {"index", &expertIndex}, {"score", &expertScore},
                    {"weights", (Data *)moeWeights[layer].data()},
                    {"biass", (Data *)moeBiases[layer].data()},
                    {"w1", &w1}, {"w2", &w2}, {"w3", &w3},
                    {"curInput", &tempInput}, {"curOutput", &tempOutput}, {"output", &moeOutput}
                }, {{"sharedScale", 0.0f}}, {
                    {"weights___batch", (int)moeWeights[layer].size()},
                    {"biass___batch", (int)moeBiases[layer].size()},
                    {"layer", layer}, {"gateType", (int)MoeGateSwiglu}, {"fp8EagerMode", 1}
                });
            } else if (((batch && batch->Decode()) || (capture && capture->verifying)) && length > 1 &&
                       (moeWeights[layer][2]->dataType == DataType::BFLOAT16 ||
                        moeWeights[layer][2]->dataType == DataType::NVFP4_BLOCK_16_E4M3_PACKED) &&
                       normed.dataDevice == DataDevice::CUDA) {
                // Expert batching can change BF16 accumulation order, including
                // Marlin's split-K schedule. Verify with decode arithmetic;
                // the following TP reduction still covers the entire block.
                moeOutput.dataType = BFLOAT16;
                moeOutput.Resize({length, embed_dim});
                moeOutput.ToDevice(normed.dataDevice, normed.dataDeviceIds);
                moeOutput.Allocate(false);
                const bool residentRows =
                    moeWeights[layer][2]->dataType == DataType::NVFP4_BLOCK_16_E4M3_PACKED &&
                    FastllmCudaNVFP4E4M3GroupedMoeSupported(normed.dataDeviceIds.at(0)) &&
                    FastllmCudaPrepareNVFP4E4M3Moe(moeWeights[layer].data(), moeWeights[layer].size());
                // The verify kernel uses full-K expert tiles. Ordinary decode
                // may split K, so keep its per-request reduction order here.
                const bool mergedRows = !batch && residentRows &&
                    FastllmCudaMergeMOENVFP4E4M3MarlinRows(normed, w1, w2, moeOutput,
                        moeWeights[layer].data(), moeWeights[layer].size(),
                        (const int32_t *)expertIndex.cudaData, (const float *)expertScore.cudaData,
                        length, num_experts_per_tok);
                for (int row = 0; !mergedRows && row < length; ++row) {
                    Data rowInput(BFLOAT16, {1, embed_dim});
                    Data rowIndex, rowScore, rowOutput;
                    rowInput.FakeFrom(normed, (size_t)row * embed_dim * 2);
                    if (residentRows) {
                        rowIndex.Resize({1, expertIndex.dims[1]});
                        rowScore.Resize({1, expertScore.dims[1]});
                        rowOutput.Resize({1, embed_dim});
                        rowIndex.FakeFrom(expertIndex, (size_t)row * expertIndex.GetBytes() / length);
                        rowScore.FakeFrom(expertScore, (size_t)row * expertScore.GetBytes() / length);
                        rowOutput.FakeFrom(moeOutput, (size_t)row * embed_dim * 2);
                    } else {
                        // Generic expert backends may move routing to CPU;
                        // those need owning tensors instead of device views.
                        Split(expertIndex, 0, row, row + 1, rowIndex);
                        Split(expertScore, 0, row, row + 1, rowScore);
                    }
                    MergeMOEBlock(&rowInput, &rowIndex, &rowScore, &moeWeights[layer],
                        &moeBiases[layer], &w1, &w2, &w3, &tempInput, &tempOutput,
                        0.0f, &rowOutput, layer, DataType::BFLOAT16, moeAtype,
                        &moeInputTemp, &moeOutputTemp);
                    if (!residentRows)
                        FastllmCudaCopyFromDeviceToDevice((uint16_t *)moeOutput.cudaData + (size_t)row * embed_dim,
                                                         rowOutput.cudaData, (size_t)embed_dim * 2);
                }
            } else {
                MergeMOEBlock(&normed, &expertIndex, &expertScore, &moeWeights[layer],
                              &moeBiases[layer], &w1, &w2, &w3, &tempInput, &tempOutput,
                              0.0f, &moeOutput, layer, DataType::BFLOAT16, moeAtype,
                              &moeInputTemp, &moeOutputTemp);
            }
            if (!tensorParallel) ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
            moeOutput.Reshape(hidden.dims);
            if (!moeNormReady)
                addMlpResidual(moeOutput, !(tensorParallel && tpEp8MegaState));
        }
        prefillNormReady = fusePrefillNorm && mlpNormWeight;
        if (capture && capture->collectHidden &&
            std::find(draftTargetLayers.begin(), draftTargetLayers.end(), layer) != draftTargetLayers.end())
            Copy(hidden, capture->hidden[layer]);
    }
    if (isIntermediateChunkedPrefill) return Data();
    Data localLogits;
    Data &last = buf.last, &logits = decodeWorkspace ? buf.logits : localLogits;
    if (!decodeWorkspace) {
        if (batch) {
            last.dataType = BFLOAT16;
            last.Resize({1, batch->Size(), embed_dim});
            last.ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false);
            last.Allocate(false);
            int offset = 0;
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                offset += batch->lengths[sequence];
                FastllmCudaCopyFromDeviceToDevice((uint16_t *)last.cudaData + (size_t)sequence * embed_dim,
                    (uint16_t *)hidden.cudaData + (size_t)(offset - 1) * embed_dim, (size_t)embed_dim * sizeof(uint16_t));
            }
        } else if (capture && capture->verifying) Copy(hidden, last);
        else Split(hidden, 1, length - 1, length, last);
        norm(last, localWeight("model.norm.weight"), last);
    }
    if (decodeWorkspace) {
        // Conversions must not free addresses retained by the graph.
        Linear(last, localWeight("lm_head.weight"), Data(), buf.logitsBf16);
        ToDataType(buf.logitsBf16, logits, DataType::FLOAT32);
        return Data();
    }
    Linear(last, localWeight("lm_head.weight"), Data(), logits);
    ToDataType(logits, DataType::FLOAT32);
    return localLogits;
#endif
}

NaiveN05FlashModel::LogitsSelection NaiveN05FlashModel::SelectLogits(
        const GenerationConfig &config, bool speculative) const {
    LogitsSelection selection;
    // These transforms must happen before selection. Keep the established
    // full-logits path until their branch-local GPU equivalents are available.
    if (config.output_logits || config.output_token_least > 0 ||
        std::abs(config.repeat_penalty - 1.0f) > 1e-8f ||
        !config.tool_call_allowed_token_ids.empty()) return selection;
    selection.greedy = config.IsSimpleGreedy();
    // Full-vocabulary top-p matches the shared CUDA sampler's distribution.
    // Finite top-k keeps its existing top-k-then-top-p semantics, as does
    // speculative verification, which needs probabilities rather than a draw.
    if (!speculative && !selection.greedy &&
        config.top_k >= weight.weight.at("lm_head.weight").dims[0] &&
        std::isfinite(config.temperature) && config.temperature > 0 &&
        std::isfinite(1.0f / config.temperature) &&
        std::isfinite(config.top_p) && config.top_p > 0 && config.top_p <= 1) {
        selection.cudaSampling = true;
        return selection;
    }
    if (selection.greedy) selection.count = 1;
    else if (config.top_k > 1 && config.top_k <= kNaiveLogitsMaxTopK &&
             std::isfinite(config.temperature) && config.temperature > 0 &&
             std::isfinite(1.0f / config.temperature)) {
        selection.count = config.top_k;
        // Ordinary LLMSampling ranks scaled scores; DSpark ranks raw scores
        // and applies temperature after subtracting the largest score.
        selection.invTemperature = speculative ? 1.0f : 1.0f / config.temperature;
    }
    return selection;
}

std::vector<int> NaiveN05FlashModel::SampleCudaLogits(
        Data &logits, const std::vector<GenerationConfig> &configs) {
#ifdef USE_CUDA
    AssertInFastLLM(logits.dataDevice == DataDevice::CUDA && logits.dataType == FLOAT32 &&
        logits.Count(0) / logits.dims.back() == configs.size(), "Naive CUDA sampling: invalid logits.");
    FastllmCudaSetDevice(logits.dataDeviceIds.at(0));
    const int rows = configs.size(), vocab = logits.dims.back();
    std::vector<float> temperatures(rows), topPs(rows);
    std::vector<int> topKs(rows, vocab), tokens(rows);
    for (int row = 0; row < rows; ++row) {
        temperatures[row] = configs[row].temperature;
        topPs[row] = configs[row].top_p;
    }
    AssertInFastLLM(FastllmCudaTopKTopPSampling((float *)logits.cudaData,
        temperatures.data(), topKs.data(), topPs.data(), tokens.data(), rows, vocab),
        "Naive CUDA top-p sampling failed.");
    return tokens;
#else
    ErrorInFastLLM("Naive CUDA sampling requires CUDA.");
    return {};
#endif
}

int NaiveN05FlashModel::SampleTarget(
        Data &logits, std::vector<std::pair<Data, Data>> &pastKeyValues,
        const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
        std::vector<float> *retLogits, LogitsSelection *selection) {
    if (selection && !selection->sampledTokens.empty()) return selection->sampledTokens[0];
    if (selection && selection->cudaSampling)
        return SampleCudaLogits(logits, {generationConfig})[0];
    if (selection && !selection->candidates.dims.empty()) {
        if (selection->greedy) return (int)((float *)selection->candidates.cpuData)[0];
        return LLMSamplingOnly(selection->candidates, 0, generationConfig);
    }
    Data top;
    if (generationConfig.output_logits && retLogits) {
        logits.ToDevice(DataDevice::CPU);
        retLogits->assign((float *)logits.cpuData, (float *)logits.cpuData + logits.Count(0));
    }
    ResetLogitsOfEOS(1, &logits, pastKeyValues, generationConfig);
    if (generationConfig.IsSimpleGreedy()) {
        TopK(logits, top, 1);
        top.ToDevice(DataDevice::CPU);
        return (int)(((float *)top.cpuData)[0] + 1e-3f);
    }
    LastTokensUnit empty;
    return LLMSampling(logits, 0, generationConfig,
                       lastTokens.units.empty() ? empty : lastTokens.units[0]);
}

void NaiveN05FlashModel::WarmUp() {
    Data ids(DataType::FLOAT32, {1, 1}, {1.0f});
    Data positions(DataType::FLOAT32, {1, 1}, {0.0f});
    std::vector<std::pair<Data, Data>> cache;
    for (int layer = 0; layer < block_cnt; layer++)
        cache.emplace_back(Data(kvCacheDataType), Data(kvCacheDataType));
    Forward(ids, Data(), positions, cache);
    if (draftEnabled) {
        auto context = draftContexts.at(&cache);
        RunDraft(1, *context);
        draftContexts.erase(&cache);
    }
    elementsInKVCachePerToken = 0;
    for (int layer = 0; layer < block_cnt; layer++) {
        if (slidingLayers[layer]) continue;
        if (tpDevices.size() > 1) {
            // Count physical storage, including replicated KV heads and the
            // replicated Indexer key, for the scheduler's token budget.
            for (int device : tpDevices)
                elementsInKVCachePerToken += cache[layer].first.multiDeviceDatas.at(device)->dims[2] +
                                            cache[layer].second.multiDeviceDatas.at(device)->dims[2];
        } else {
            elementsInKVCachePerToken += cache[layer].first.dims[2] + cache[layer].second.dims[2];
        }
    }
    cache.clear();
    InitTensorParallelKVPool();
    PrepareTensorParallelServing();
}

void NaiveN05FlashModel::PrepareSwaPrefillWorkspace(int rank, int length) {
#ifdef USE_CUDA
    FastllmCudaSetDevice(tpDevices[rank]);
    auto &workspace = tpSwaPrefillWorkspaces[rank];
    const int capacity = (int)(((int64_t)window - 1 + length + 127) / 128 * 128);
    for (int part = 0; part < 2; ++part) {
        Data &data = part ? workspace.second : workspace.first;
        if (data.cudaData && !data.expansionDims.empty() && data.expansionDims[1] >= capacity) continue;
        // Normal serving is preallocated for the configured chunk. Direct
        // Forward callers may supply larger inputs; only scratch grows then.
        data.FreeSpace();
        data.dims.clear();
        data.dataType = kvCacheDataType;
        data.UpdateUnitSize();
        data.dataDevice = DataDevice::CUDA;
        data.dataDeviceIds = {tpDevices[rank]};
        const int width = std::max(1, sliding.kvHeads / (int)tpDevices.size()) *
            (part ? sliding.valueDim : sliding.headDim);
        data.Expansion({1, capacity, width});
    }
#endif
}

void NaiveN05FlashModel::InitTensorParallelKVPool() {
#ifdef USE_CUDA
    // Keep CPU/layer-serial and speculative execution on their existing cache
    // paths. This pool serves ordinary BF16/FP8 TP, independently of GPU generation
    // and whether CUDA Graph is enabled.
    if (!tpKVPool.empty() || !tpPrepared || tpDevices.size() <= 1 ||
        GetKVCacheInCPU() || !FastllmNaiveKVType(kvCacheDataType) || draftEnabled ||
        elementsInKVCachePerToken <= 0) return;
    const double gpuMemRatio = GetGpuMemRatio();
    AssertInFastLLM(std::isfinite(gpuMemRatio) && gpuMemRatio > 0 && gpuMemRatio <= 1,
        "Naive GPU KV pool requires 0 < --gpu_mem_ratio <= 1.");
    const int ranks = tpDevices.size();
    const int batch = maxBatch > 0 ? maxBatch : 512;
    const int chunk = std::max(1, GetChunkedPrefillSize());
    const int64_t swaCapacity = ((int64_t)window + 127) / 128 * 128;
    if (std::count(slidingLayers.begin(), slidingLayers.end(), 1)) {
        tpSwaPrefillWorkspaces.resize(ranks);
        for (int rank = 0; rank < ranks; ++rank) PrepareSwaPrefillWorkspace(rank, chunk);
    }
    const int kvBytes = kvCacheDataType == FP8_E4M3 ? 1 : 2;
    int64_t fullBytesPerToken = 0, fixedBytes = 0;
    for (int layer = 0; layer < block_cnt; ++layer) {
        const auto &cfg = slidingLayers[layer] ? sliding : full;
        const int64_t width = std::max(1, cfg.kvHeads / ranks) *
            (cfg.headDim + cfg.valueDim) + (slidingLayers[layer] ? 0 : IndexerCacheWidth(kvCacheDataType));
        if (slidingLayers[layer]) fixedBytes += width * kvBytes * swaCapacity * batch;
        else fullBytesPerToken += width * kvBytes;
    }
    AssertInFastLLM(fullBytesPerToken > 0, "Naive KV pool requires full-attention layers.");
    // Scratch is allocated before measuring free memory, so the total budget
    // accounts for it without multiplying it by layers or concurrent requests.
    auto freeSizes = FastllmCudaGetFreeSizes();
    auto totalSizes = FastllmCudaGetTotalSizes();
    int64_t budget = INT64_MAX;
    for (int device : tpDevices) {
        // Match paged AutoWarmup's ratio of total device memory. Measured free
        // memory already excludes weights and warmed workspaces; the remainder
        // outside the ratio is reserved for runtime growth, not KV storage.
        const int64_t reserved = (int64_t)(totalSizes.at(device) * (1.0 - gpuMemRatio));
        const int64_t available = freeSizes.at(device) - reserved;
        budget = std::min(budget, available);
        printf("[Naive KV budget] GPU %d: total %.2f MiB, free %.2f MiB, gpu_mem_ratio %.4f, runtime reserve %.2f MiB, available for KV %.2f MiB.\n",
            device, totalSizes.at(device) / 1048576.0, freeSizes.at(device) / 1048576.0,
            gpuMemRatio, reserved / 1048576.0, available / 1048576.0);
    }
    if (kvCacheLimit > 0) budget = std::min(budget, (int64_t)kvCacheLimit / ranks);
    AssertInFastLLM(budget > 0,
        "Naive GPU KV pool has no memory budget; increase --gpu_mem_ratio or free GPU memory.");
    // Reserve bounded SWA checkpoints inside the same total KV budget. Full
    // DSA/indexer pages are shared, never copied into this snapshot storage.
    auto prefixes = std::make_shared<PrefixPool>();
    prefixes->offsets.resize(block_cnt * 2);
    for (int layer = 0; layer < block_cnt; ++layer) if (slidingLayers[layer]) {
        for (int part = 0; part < 2; ++part) {
            prefixes->offsets[layer * 2 + part] = prefixes->slotBytes;
            const int width = std::max(1, sliding.kvHeads / ranks) *
                (part ? sliding.valueDim : sliding.headDim);
            prefixes->slotBytes += (size_t)window * width * kvBytes;
        }
    }
    prefixes->slotBytes = (prefixes->slotBytes + 255) / 256 * 256;
    const int prefixSlots = prefixes->slotBytes ? (int)std::min<int64_t>(
        (int64_t)batch * 8, budget / 32 / prefixes->slotBytes) : 0;
    const int64_t prefixBytes = (int64_t)prefixSlots * prefixes->slotBytes;
    fixedBytes += prefixBytes;
    int64_t tokens = GetMaxTokens() > 0 ? GetMaxTokens() :
        std::min<int64_t>((int64_t)max_positions * batch,
                         (budget - fixedBytes) / fullBytesPerToken - (int64_t)batch * 127);
    if (tokensLimit > 0) tokens = std::min<int64_t>(tokens, tokensLimit);
    tokens = std::min<int64_t>(tokens, INT_MAX - 127);
    // Preserve the existing budget, including a partial page per request.
    const int64_t bytes = fixedBytes + (tokens + (int64_t)batch * 127) * fullBytesPerToken;
    AssertInFastLLM(tokens >= 128 && bytes > 0 && bytes <= budget,
        "Naive startup KV pool exceeds the memory budget; reduce --tokens/--max_batch or prefill chunk size.");
    std::map<int, std::shared_ptr<CudaWorkspace>> pools;
    for (int device : tpDevices)
        pools.emplace(device, std::make_shared<CudaWorkspace>(device, bytes));
    tpKVPool.swap(pools);
    if (prefixSlots > 0) {
        prefixes->storage.resize(ranks);
        for (int rank = 0; rank < ranks; ++rank) {
            Data &storage = prefixes->storage[rank];
            storage.dataType = INT8;
            storage.UpdateUnitSize();
            storage.Resize({prefixSlots, (int)prefixes->slotBytes});
            storage.dataDevice = DataDevice::CUDA;
            storage.dataDeviceIds = {tpDevices[rank]};
            storage.cudaData = AllocateCudaWorkspace(tpKVPool.at(tpDevices[rank]), prefixBytes);
            if (!storage.cudaData) throw CudaWorkspaceExhausted("Naive GPU prefix pool exhausted.");
            storage.directMemory = true;
        }
        for (int slot = 0; slot < prefixSlots; ++slot) prefixes->freeSlots.push_back(slot);
        prefixPool = std::move(prefixes);
        printf("[Naive GPU prefix cache] %d window slots, %.2f MiB/GPU within KV budget; no CPU KV archive.\n",
               prefixSlots, prefixBytes / 1048576.0);
    }
    const int pageLen = GetPageLen();
    const int fullLayers = std::count(slidingLayers.begin(), slidingLayers.end(), 0);
    const int64_t metadataBytes = (int64_t)fullLayers * 2 * batch *
        ((std::min<int64_t>(max_positions, tokens) + pageLen - 1) / pageLen * sizeof(int) + 512) +
        (int64_t)block_cnt * std::min(batch, 256) * 4 * sizeof(void *) + 512;
    const int pages = (int)((bytes - fixedBytes - metadataBytes) / fullBytesPerToken / pageLen);
    AssertInFastLLM(pages > 0, "Naive KV budget cannot fit paged metadata.");
    tokens = std::min<int64_t>(tokens, (int64_t)pages * pageLen);
    tpPagedCaches.resize(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        FastllmCudaSetDevice(tpDevices[rank]);
        auto &managers = tpPagedCaches[rank];
        managers.resize(block_cnt * 2);
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (slidingLayers[layer]) continue;
            for (int part = 0; part < 2; ++part) {
                int width = std::max(1, full.kvHeads / ranks) * (part ? full.valueDim : full.headDim) + (!part ? IndexerCacheWidth(kvCacheDataType) : 0);
                auto manager = std::make_unique<PagedCacheManager>();
                manager->type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
                manager->kvCacheFp8Indexer = !part && fp8IndexerCache;
                manager->dataType = kvCacheDataType;
                manager->UpdateUnitSize();
                manager->dataDevice = DataDevice::CUDA;
                manager->dataDeviceIds = {tpDevices[rank]};
                manager->pageLen = pageLen;
                manager->Resize({pages, pageLen, 1, width});
                manager->cudaData = AllocateCudaWorkspace(tpKVPool.at(tpDevices[rank]), manager->GetBytes());
                if (!manager->cudaData) throw CudaWorkspaceExhausted("Naive physical KV page pool exhausted.");
                manager->directMemory = true;
                manager->SetMaxPages(pages);
                FastllmCudaMemset0(manager->cudaData, (size_t)pageLen * width * kvBytes);
                managers[layer * 2 + part] = std::move(manager);
            }
        }
        ForceDeviceSync();
    }
    tokensLimit = (int)tokens;
    printf("[Naive KV pool] preallocated %.2f MiB/GPU, %lld total tokens, batch %d, SWA capacity %lld rows/request. No serving-time KV cudaMalloc.\n",
        bytes / 1048576.0, (long long)tokens, batch, (long long)swaCapacity);
    fflush(stdout);
#endif
}
}
