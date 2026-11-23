#include "utils/decode_graph.h"
#include "naive_n05_flash.h"
#include "naive_n05_tp.h"
#ifdef USE_CUDA
#include "naive_n05_flexmoe.h"
#include "naive_n05_shared_flex.h"
#endif
#include "naive_n05_parallel.h"
#include "models/mimo_tool_grammar.h"
#include "naive_n05_kvmem.h"
#include "blocks/baseblock.h"
#include "executor.h"
#include "json11.hpp"
#include "utils.h"
#include "utils/inference_trace.h"
#include <cmath>
#include <climits>
#include <optional>
#include <cstdlib>
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda-decode-linear.h"
#include "devices/cuda/fastllm-cuda-decode-dense.h"
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
#include "devices/cuda/naive-n05-dsa-layer.cuh"
#endif
#ifdef USE_CUDA
#include "naive_n05_indexer_tp.h"
#include "devices/cuda/naive-n05-indexer-attention.cuh"
#include "devices/cuda/naive-n05-swa-prefill.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/cudaworkspace.h"
#include "devices/cuda/fastllm-linear-scratch.h"
#include "devices/cuda/fastllm-mxfp8-projection.h"
#include "devices/cuda/fastllm-cuda-linear-output.h"
#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "devices/cuda/fastllm-mxfp4-marlin-tp.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
namespace {
    int TraceContextBefore(const Data *positions) {
        return positions && positions->dataDevice == DataDevice::CPU &&
            positions->dataType == FLOAT32 && positions->cpuData && positions->Count(0) > 0
            ? (int)((const float *)positions->cpuData)[0] : -1;
    }

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
    using IndexerGather = std::function<void(const void *, void *, size_t)>;
    void GatherContextCandidates(Data &exchange, int part, int rank, int ranks, int gpu,
            const IndexerGather &gather, int rowWords = 4096) {
        if (gather) { gather((int *)exchange.cudaData + rank * part, exchange.cudaData, part); return; }
        if (!FastllmCudaCustomAllGatherRows(exchange.cudaData, ranks * part, rowWords, FLOAT32, gpu))
            AssertInFastLLM(FastllmNcclAllGather((int *)exchange.cudaData + rank * part,
                exchange.cudaData, part, INT32, gpu), "Indexer context all-gather failed.");
    }
    void RunContextVerifyIndexer(const Data &query, const Data &weights,
            const std::vector<FastllmNaiveVerifyIndexerItem> &items,
            FastllmNaiveDecodeScratch &scratch, Data &live, Data &exchange,
            int rank, int ranks, int page, int gpu, bool fp8Query, const IndexerGather &gather = {}) {
        int part = 0;
        for (const auto &item : items) part += item.rows * 2048 * 2;
        exchange.dataType = INT32; exchange.Resize({ranks * part});
        exchange.ToDevice(query.dataDevice, query.dataDeviceIds, false); exchange.Allocate(false);
        AssertInFastLLM(FastllmCudaNaiveBatchVerifyIndexer(query, weights, items, fp8Query,
            scratch, live, rank, ranks, &exchange, page), "Unsupported context-sharded verification layout.");
        GatherContextCandidates(exchange, part, rank, ranks, gpu, gather);
        FastllmCudaNaiveMergeVerifyIndexer(items, scratch, live, exchange, ranks, page);
    }
    void RunContextDecodeIndexer(const Data &query, const Data &weights, const Data &live,
            const std::vector<FastllmNaiveIndexerRow> &rows, Data &exchange,
            int rank, int ranks, int page, int gpu, bool fp8Query, const IndexerGather &gather = {}) {
        // A graph replays different live lengths within this capacity. Size the
        // dedicated exchange for the largest possible shard on any rank.
        int directWidth = 0;
        if (!rows.empty() && rows.size() < 16) {
            int capacity = 0;
            for (const auto &row : rows) capacity = std::max(capacity, row.capacity);
            const int pages = capacity / page + (capacity % page != 0);
            directWidth = std::min(2048, pages * (page / ranks));
        }
        const int rowWords = (directWidth ? directWidth : 2048) * 2;
        const int part = rows.size() * rowWords;
        exchange.dataType = INT32;
        exchange.Resize({ranks * part});
        exchange.ToDevice(query.dataDevice, query.dataDeviceIds, false);
        exchange.Allocate(false);
        AssertInFastLLM(FastllmCudaNaiveBatchDecodeIndexer(query, weights, live, rows, fp8Query,
            rank, ranks, &exchange, page, directWidth), "Unsupported context-sharded decode Indexer layout.");
        GatherContextCandidates(exchange, part, rank, ranks, gpu, gather, rowWords);
        FastllmCudaNaiveMergeDecodeIndexer(live, rows, exchange, ranks, page, directWidth);
    }
    // Copy stored KV bits between a request's window and chronological scratch.
    // At most one side wraps; copies are ordered with attention on the rank stream.
    void CopySwaWindow(const Data &key, const Data &value, Data &targetKey, Data &targetValue,
                       int sourceFirst, int targetFirst, int rows, int sourceWindow = 0, int targetWindow = 0) {
        if (rows == 0) return;
        void *dsts[4]; const void *srcs[4]; size_t sizes[4];
        int copies = 0;
        for (int part = 0; part < 2; ++part) {
            const Data &source = part ? value : key;
            Data &target = part ? targetValue : targetKey;
            const size_t rowBytes = (size_t)source.expansionDims[2] * source.unitSize;
            AssertInFastLLM(source.dataType == target.dataType &&
                source.expansionDims[2] == target.expansionDims[2], "Invalid SWA workspace layout.");
            // Verification reserves staging rows after the ring. Its logical
            // window stays unchanged even when the backing allocation grows.
            const int sourceCapacity = sourceWindow ? sourceWindow : source.expansionDims[1];
            const int targetCapacity = targetWindow ? targetWindow : target.expansionDims[1];
            for (int offset = 0; offset < rows;) {
                int from = (sourceFirst + offset) % sourceCapacity;
                int to = (targetFirst + offset) % targetCapacity;
                int count = std::min({rows - offset, sourceCapacity - from,
                                     targetCapacity - to});
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

void NaiveN05FlashModel::SetKVCacheDataType(DataType type) {
    if (attentionDP > 1 && !FastllmNaiveAttentionKVType(type))
        throw std::invalid_argument("Naive attention DP requires BF16, FP8 or FP4 KV");
    if (type == FP4_E2M1 || kvCacheDataType == FP4_E2M1)
        AssertInFastLLM(!mainLoop && !tpPrepared && !sparseKv && !kvMemLocked && !autoWarmupRunning.load(),
                        "Set Naive FP4 KV dtype before warmup or inference.");
    if (type == FP4_E2M1) {
        // ConfigureKvMem/WarmUp validate TP and storage budgets before allocation.
        kvCacheDataType = type;
        useCustomKVCacheDataType = true;
    } else basellm::SetKVCacheDataType(type);
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

void NaiveN05FlashModel::SetIndexerComputeFP8(bool enabled) {
    SetIndexerCompute(NaiveIndexerComputeFromFP8(enabled));
}

void NaiveN05FlashModel::SetIndexerCompute(NaiveIndexerCompute mode) {
    if (tpPrepared || tpServingState || !tpKVPool.empty() || !activePrefixes.empty())
        throw std::invalid_argument("Set Indexer compute dtype before warmup or inference.");
#ifndef USE_CUDA
    if (mode != NaiveIndexerCompute::Auto) throw std::invalid_argument("FP8 Indexer compute requires CUDA.");
#endif
    indexerCompute = mode;
}

void NaiveN05FlashModel::SetIndexerTPStorage(bool shardByLayer, bool shardByContext) {
    if (attentionDP > 1 && shardByLayer)
        throw std::invalid_argument("Naive attention DP supports replicated or context-sharded Indexer storage");
    AssertInFastLLM(!tpPrepared && !tpServingState && tpKVPool.empty() && activePrefixes.empty(),
                    "Set Indexer TP storage before warmup or inference.");
    AssertInFastLLM(!shardByLayer || (tpDevices.size() > 1 && !GetKVCacheInCPU()),
                    "Indexer layer_shard storage requires CUDA TP with at least two devices.");
    AssertInFastLLM(!shardByContext || (!shardByLayer && tpDevices.size() > 1 &&
        !GetKVCacheInCPU() && GetPageLen() % tpDevices.size() == 0),
        "Indexer context_shard requires CUDA TP and a page length divisible by TP size.");
    indexerLayerSharding = shardByLayer;
    indexerContextSharding = shardByContext;
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

void NaiveN05FlashModel::AppendCache(Data &cache, Data &input, DataType cacheType, int reserveCapacity, bool usePool, bool expandFP8) {
    int oldLength = cache.dims.empty() ? 0 : cache.dims[1];
    int length = oldLength + input.dims[1];
    cache.dataType = cacheType;
    cache.UpdateUnitSize();
    if (!(sparseKv && cache.cudaDataBorrowed && cache.dataDevice == input.dataDevice &&
          cache.dataDeviceIds == input.dataDeviceIds))
        cache.ToDevice(input.dataDevice, input.dataDeviceIds);
    const int width = input.dims[2] + (cache.kvCacheFp8Indexer
        ? IndexerCacheWidth(cacheType) - IndexerCacheWidth(BFLOAT16) : 0);
    const int64_t wanted = std::max(length, reserveCapacity);
    if (cache.expansionDims.empty() || cache.expansionDims[1] < wanted) {
        const int capacity = (int)std::min<int64_t>(INT_MAX, (wanted + 127) / 128 * 128);
        ReserveCache(cache, capacity, width, usePool);
    }
    if (cache.dims.empty()) cache.Resize({1, 0, width});
    if (cache.isPagedKVCache || cache.dataType == FP8_E4M3 || cache.dataType == FP4_E2M1 || cache.kvCacheFp8Indexer ||
        (sparseKv && cache.cudaDataBorrowed)) {
#ifdef USE_CUDA
        if (expandFP8) FastllmCudaNaiveWriteAndExpandFP8(cache, input, oldLength);
        else FastllmCudaNaivePagedWrite(cache, input, oldLength);
#endif
        cache.Resize({1, length, width});
        if (cache.isPagedKVCache) cache.lastPageLen = (length - 1) % cache.pageLen + 1;
    } else CatDirect(cache, input, 1);
    cache.isKVCache = true;
}

void NaiveN05FlashModel::ReserveCache(Data &cache, int capacity, int width, bool usePool, const Data *requestKey) {
    bool invalidated = false;
    ReserveCache(cache, capacity, width, usePool, requestKey, invalidated);
}

void NaiveN05FlashModel::ReserveCache(Data &cache, int capacity, int width, bool usePool, const Data *requestKey,
                                     bool &invalidated) {
    if (cache.expansionDims.size() == 3 && cache.expansionDims[1] >= capacity) return;
    if (!invalidated) {
        InvalidateTensorParallelCache(requestKey);
        invalidated = true;
    }
#ifdef USE_CUDA
    auto pool = usePool && cache.dataDevice == DataDevice::CUDA && cache.dataDeviceIds.size() == 1
        ? tpKVPool.find(cache.dataDeviceIds[0]) : tpKVPool.end();
    if (pool != tpKVPool.end()) {
        AssertInFastLLM(FastllmNaiveAttentionKVType(cache.dataType) && !cache.isFake && !cache.cudaDataBorrowed &&
            (cache.dims.empty() || (cache.dims.size() == 3 && cache.dims[0] == 1 &&
                                   cache.dims[2] == width && cache.dims[1] <= capacity)),
            "Naive KV pool: invalid contiguous cache.");
        Data storage(cache.dataType, {1, capacity, width});
        storage.dataDevice = DataDevice::CUDA;
        storage.dataDeviceIds = cache.dataDeviceIds;
        storage.directMemory = true;
        storage.cudaData = AllocateCudaWorkspace(RequestKVPool(pool->first, storage.GetBytes()), storage.GetBytes());
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

// Page-index vectors are owned by the current rank task until this returns.
// Complete every upload before callers can reuse their host storage or capture
// a graph. General cache preparation keeps its original synchronous path.
void NaiveN05FlashModel::UploadPagedCacheTables(const std::vector<PageTableUpdate> &pendingPageTables) {
#ifdef USE_CUDA
    if (pendingPageTables.empty()) return;
    // A single table is cheaper to copy directly. For a batch, upload packed
    // indices once, then scatter using the generic bounded batch-copy kernel.
    // Scratch is per worker and device; the stream completes before reuse.
    if (pendingPageTables.size() == 1) {
        const auto &update = pendingPageTables.front();
        Data &cache = *update.cache;
        if (cache.pageIndex.size() == update.first) return;
        FastllmCudaCopyFromHostToDevice((int *)cache.cudaData + update.first, cache.pageIndex.data() + update.first,
            (cache.pageIndex.size() - update.first) * sizeof(int));
        return;
    }
    struct Staging {
        Data device{INT32};
        std::vector<int> host;
        std::vector<void *> dst;
        std::vector<const void *> src;
        std::vector<size_t> sizes;
    };
    static thread_local std::map<int, Staging> stagingByDevice;
    const int device = pendingPageTables.front().cache->dataDeviceIds[0];
    auto &staging = stagingByDevice[device];
    size_t count = 0;
    for (const auto &update : pendingPageTables) count += update.cache->pageIndex.size() - update.first;
    if (!count) return;
    staging.host.resize(count);
    staging.device.Resize({(int)count});
    staging.device.ToDevice(DataDevice::CUDA, {device}, false);
    staging.device.Allocate(false);
    staging.dst.clear(); staging.src.clear(); staging.sizes.clear();
    size_t offset = 0;
    for (const auto &update : pendingPageTables) {
        Data &cache = *update.cache;
        const size_t size = cache.pageIndex.size() - update.first;
        if (!size) continue;
        std::copy_n(cache.pageIndex.data() + update.first, size, staging.host.data() + offset);
        staging.dst.push_back((int *)cache.cudaData + update.first);
        staging.src.push_back((int *)staging.device.cudaData + offset);
        staging.sizes.push_back(size * sizeof(int));
        offset += size;
    }
    try {
        FastllmCudaCopyFromHostToDeviceAsync(staging.device.cudaData, staging.host.data(),
            count * sizeof(int), cudaStreamPerThread);
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
            staging.dst.data(), staging.src.data(), staging.sizes.data(), staging.dst.size()),
            "Naive page table upload failed.");
    } catch (...) {
        FastllmCudaSyncCurrentThreadStream();
        throw;
    }
    FastllmCudaSyncCurrentThreadStream();
#endif
}

void NaiveN05FlashModel::PreparePDPrefillTables(
        const std::vector<std::vector<PDPrefillPagePlan *>> &rows, int spare) {
#ifdef USE_CUDA
    std::vector<std::exception_ptr> errors(tpDevices.size());
    tpWorkers.Run(tpDevices, [&](int rank) {
        if (rows[rank].empty()) return;
        const int device = tpDevices[rank];
        FastllmCudaSetDevice(device);
        std::vector<PageTableUpdate> pending;
        for (auto *plan : rows[rank]) {
            auto &ctx = *plan->context;
            bool invalidated = false;
            for (int layer = 0; layer < block_cnt; ++layer) if (!slidingLayers[layer]) {
                for (Data *root : {&ctx.pastKeyValues[layer].first, &ctx.pastKeyValues[layer].second}) {
                    auto &local = *root->multiDeviceDatas.at(device);
                    auto *pool = local.pagedKVCacheData;
                    if (!pool || !local.isPagedKVCache || local.cudaData || ctx.preTokens % pool->pageLen ||
                        local.pageIndex.size() != size_t(ctx.preTokens / pool->pageLen)) continue;
                    long long generation;
                    {
                        std::lock_guard<std::mutex> lock(pool->pageIndexLocker);
                        generation = pool->pageTimestamp[local.pageIndex.back()];
                    }
                    plan->tables.push_back({&local, generation});
                    const int endPages = (plan->end + pool->pageLen - 1) / pool->pageLen;
                    if (!pool->TryExtendFreePages(local.pageIndex, endPages, spare)) {
                        plan->tables.pop_back();
                        continue;
                    }
                    try {
                        PreparePagedCache(local, *pool, plan->end, &pending, &ctx.pastKeyValues.front().first, &invalidated);
                    } catch (const CudaWorkspaceExhausted &) {
                        // Optional metadata storage was busy. Return the fresh suffix and
                        // let ordinary admission allocate this table after current exports.
                        while (int(local.pageIndex.size()) > ctx.preTokens / pool->pageLen) {
                            pool->ReleasePageIndex(local.pageIndex.back());
                            local.pageIndex.pop_back();
                        }
                        plan->tables.pop_back();
                    }
                }
            }
        }
        UploadPagedCacheTables(pending);
    }, errors);
    for (auto &error : errors) if (error) std::rethrow_exception(error);
#endif
}

bool NaiveN05FlashModel::PreparePagedCache(Data &cache, PagedCacheManager &manager, int end) {
    return PreparePagedCache(cache, manager, end, nullptr);
}

bool NaiveN05FlashModel::PreparePagedCache(Data &cache, PagedCacheManager &manager, int end,
                                         std::vector<PageTableUpdate> *pendingPageTables, const Data *requestKey,
                                         bool *requestInvalidated) {
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
    // Admission rollback releases pages and clears shape while retaining the
    // table allocation/pool binding. Reinitialize the empty logical cache
    // before inspecting its length or constructing an Indexer view.
    if (cache.dims.empty()) cache.Resize({1, 0, width});
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
    const bool newTable = !cache.cudaData;
    const bool changed = newTable || copyTail || pages > (int)cache.pageIndex.size();
    if (!cache.cudaData) {
        const size_t bytes = (size_t)((capacity + manager.pageLen - 1) / manager.pageLen) * sizeof(int);
        cache.cudaData = AllocateCudaWorkspace(RequestKVPool(manager.dataDeviceIds[0], bytes), bytes);
        if (!cache.cudaData) throw CudaWorkspaceExhausted("Naive KV page metadata pool exhausted.");
        cache.directMemory = true;
        cache.cudaDataBorrowed = false;
    }
    if (changed) {
        // Register before acquiring pages. A later allocation failure can leave
        // a valid partial extension which the caller flushes before retrying.
        if (pendingPageTables) {
            const size_t first = newTable ? 0 : copyTail ? size_t((previous - 1) / cache.pageLen) : cache.pageIndex.size();
            pendingPageTables->push_back({&cache, first});
        }
        // One scheduler reservation invalidates this request once per rank.
        if (!requestInvalidated || !*requestInvalidated) {
            InvalidateTensorParallelCache(requestKey);
            if (requestInvalidated) *requestInvalidated = true;
        }
    }
    if (copyTail) {
        int last = (previous - 1) / cache.pageLen;
        int old = cache.pageIndex[last], page = manager.GetUnusedPageIndex(true);
        size_t bytes = GetDataBytes(manager.dataType, cache.pageLen, width);
        FastllmCudaCopyFromDeviceToDevice((uint8_t *)manager.cudaData + page * bytes,
            (uint8_t *)manager.cudaData + old * bytes, bytes);
        auto sidecar = tpIndexerSidecars.find(&manager);
        if (sidecar != tpIndexerSidecars.end()) {
            Data &index = *sidecar->second;
            size_t indexBytes = size_t(index.dims[1]) * index.dims[3] * index.unitSize;
            FastllmCudaCopyFromDeviceToDevice((char *)index.cudaData + page * indexBytes,
                (char *)index.cudaData + old * indexBytes, indexBytes);
        }
        cache.pageIndex[last] = page;
        manager.ReleasePageIndex(old);
    }
    manager.ExtendPageIndices(cache.pageIndex, pages);
    if (changed && !pendingPageTables && !cache.pageIndex.empty())
        FastllmCudaCopyFromHostToDevice(cache.cudaData, cache.pageIndex.data(), cache.pageIndex.size() * sizeof(int));
    cache.lastPageLen = previous ? (previous - 1) % cache.pageLen + 1 : 0;
    return changed;
#else
    return false;
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
    hostPrefixCache.reset(); // Drain async readers before retiring the page managers.
    JoinDraftLookahead();
    activePrefixes.clear();
    prefixHistory.clear();
    prefixPool.reset();
    tpWorkers.Stop();
    // Drop the active graph reference before releasing serving-owned logits.
    tpDecodeState.reset();
    tpServingState.reset();
    tpVerifyState.reset();
    tpBatchState.reset();
    tpBatchVerifyState.reset();
    tpBatchVerifyGraphs.clear();
    tpMarlinRanks.clear();
    tpSharedFlex.reset();
    tpEp8MegaState.reset();
    tpFlexMoe.reset();
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
    canDoBatchForward = tpDevices.size() > 1;
    // Single-device and unsupported speculative modes retain request rotation.
    canDoConcurrentForward = draftEnabled;
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
    const std::set<std::string> sourceNames(names.begin(), names.end());
    const bool compactNvfp4 = std::any_of(names.begin(), names.end(), [](const std::string &name) {
        return name.rfind("model.layers.", 0) == 0 && name.find(".mlp.experts.") != std::string::npos &&
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
            const bool fp32 = name.find("norm.weight") != std::string::npos ||
                name.rfind("confidence_head.", 0) == 0 || name.find(".mlp.gate.") != std::string::npos;
            result[name] = {{"dspark." + name, fp32 ? DataType::FLOAT32 : DataType::BFLOAT16}};
        } else if (compactNvfp4 && moeLinears.count(name)) {
            result[name] = {{name, nvfp4Type}};
        } else if (name.find(".mlp.gate.") != std::string::npos) {
            result[name] = {{name, DataType::FLOAT32}};
        } else if (name.find(".mlp.experts.") == std::string::npos &&
                   (weight.GetWeightType(name) == WeightType::LINEAR ||
                    weight.GetWeightType(name) == WeightType::EMBEDDING)) {
            // Preserve serialized FP8 projections and their scales. Unquantized
            // matrices and embeddings retain the established BF16 storage.
            const bool scaledLinear = weight.GetWeightType(name) == WeightType::LINEAR &&
                sourceNames.count(name + "_scale_inv");
            result[name] = {{name, scaledLinear ? DataType::DATA_AUTO_LINEAR : DataType::BFLOAT16}};
        }
    }
    return result;
}

int NaiveN05FlashModel::GetKVCacheRetainedTokens(int layer) const {
    return slidingLayers.at(layer) ? window - 1 : -1;
}

bool NaiveN05FlashModel::SupportsPrefillBatching() const {
    return tpDevices.size() > 1 && CanUseGPUForward() &&
           !GetKVCacheInCPU();
}

int NaiveN05FlashModel::Forward(
        const Data &inputIds, const Data &attentionMask, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &pastKeyValues,
        const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
        std::vector<float> *retLogits) {
    GraphCaptureScope capturePolicy(preparingGraphs);
    std::unique_ptr<inference_trace::Forward> trace;
    if (auto *step = inference_trace::CurrentStep())
        trace.reset(new inference_trace::Forward(*step, {(int)inputIds.Count(0)},
                                                {TraceContextBefore(&positionIds)}));
    if (draftEnabled)
        return ForwardDraft(inputIds, positionIds, pastKeyValues, generationConfig, lastTokens, retLogits);
    auto selection = SelectLogits(generationConfig);
    if (trace) trace->Mark("input_prepare_ms");
    Data logits = tpDevices.size() > 1
        ? ForwardTensorParallel(inputIds, positionIds, pastKeyValues, generationConfig, nullptr, &selection)
        : RunTarget(inputIds, positionIds, pastKeyValues, generationConfig, nullptr);
    if (isIntermediateChunkedPrefill || generationConfig.prefill_chunk_remaining > 0) return 0;
    int token = SampleTarget(logits, pastKeyValues, generationConfig, lastTokens, retLogits, &selection);
    if (trace) trace->Mark("sample_return_ms");
    return token;
}

std::vector<int> NaiveN05FlashModel::ForwardBatch(int batch, const Data &inputIds,
        const std::vector<Data *> &attentionMask, const std::vector<Data *> &positionIds,
        const std::vector<int> &seqLens, std::vector<std::pair<Data *, Data *>> &pastKeyValues,
        const std::vector<GenerationConfig> &configs, const LastTokensManager &lastTokens,
        std::vector<std::vector<float> *> *retLogits) {
    GraphCaptureScope capturePolicy(preparingGraphs);
    std::unique_ptr<inference_trace::Forward> trace;
    if (auto *step = inference_trace::CurrentStep()) {
        std::vector<int> contexts;
        for (auto *position : positionIds) contexts.push_back(TraceContextBefore(position));
        trace.reset(new inference_trace::Forward(*step, seqLens, std::move(contexts)));
    }
    if (draftEnabled)
        return ForwardDraftBatch(batch, inputIds, positionIds, seqLens, pastKeyValues,
                                 configs, lastTokens, retLogits);
    return ForwardTargetBatch(batch, inputIds, attentionMask, positionIds, seqLens,
                              pastKeyValues, configs, lastTokens, retLogits);
}

std::vector<int> NaiveN05FlashModel::ForwardTargetBatch(int batch, const Data &inputIds,
        const std::vector<Data *> &attentionMask, const std::vector<Data *> &positionIds,
        const std::vector<int> &seqLens, std::vector<std::pair<Data *, Data *>> &pastKeyValues,
        const std::vector<GenerationConfig> &configs, const LastTokensManager &lastTokens,
        std::vector<std::vector<float> *> *retLogits, TargetCapture *capture) {
    auto *trace = inference_trace::CurrentForward();
    AssertInFastLLM(tpDevices.size() > 1 && batch > 0 &&
        (int)seqLens.size() == batch && (int)configs.size() == batch &&
        (int)positionIds.size() == batch && (int)pastKeyValues.size() == batch * block_cnt,
        "Naive batch forward requires TP inference and complete request descriptors.");
    TargetBatch requests{pastKeyValues, configs, seqLens, block_cnt};
    std::vector<float> packedPositions;
    packedPositions.reserve(inputIds.Count(0));
    LogitsSelection selection;
    bool hasSamplingRow = false;
    for (int b = 0; b < batch; ++b) {
        AssertInFastLLM(seqLens[b] > 0 && positionIds[b] &&
            (b >= (int)attentionMask.size() || !attentionMask[b] || attentionMask[b]->dims.empty()),
            "Naive batch expects unpadded sequences with explicit positions.");
        for (int layer = 0; layer < block_cnt; ++layer) {
            // Keep every KV check, but construct its diagnostic only on failure.
            if (!pastKeyValues[b * block_cnt + layer].first ||
                !pastKeyValues[b * block_cnt + layer].second)
                AssertInFastLLM(false, "Naive batch has a null KV descriptor.");
        }
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
        // An intermediate chunk writes KV and hidden state but has no next token.
        if (configs[b].prefill_chunk_remaining > 0) continue;
        auto other = SelectLogits(configs[b]);
        if (!hasSamplingRow) {
            selection = std::move(other);
            hasSamplingRow = true;
            continue;
        }
        selection.cudaSampling = selection.cudaSampling && other.cudaSampling;
        selection.cudaLogits = selection.cudaLogits || other.cudaLogits;
        if (selection.count != other.count || selection.greedy != other.greedy ||
            selection.invTemperature != other.invTemperature) selection.count = 0;
    }
    AssertInFastLLM(inputIds.dims == std::vector<int>({1, (int)packedPositions.size()}) &&
        inputIds.dataType == FLOAT32, "Naive batch input must pack the unpadded sequences in order.");
    Data positions(FLOAT32, inputIds.dims, DataDevice::CPU, packedPositions.data());
    if (std::any_of(configs.begin(), configs.end(), [](const GenerationConfig &cfg) { return cfg.output_logits; })) {
        selection.count = 0; selection.cudaLogits = false; selection.cudaSampling = false;
    } else if (selection.cudaLogits) selection.cudaSampling = false;
    // The batch descriptor references response-owned KV. No KV payload or
    // ownership is copied into the legacy single-sequence argument.
    std::vector<std::pair<Data, Data>> unused;
    if (trace) trace->Mark("input_prepare_ms");
    Data logits = ForwardTensorParallel(inputIds, positions, unused, configs[0],
                                        capture, &selection, &requests);
    if (isIntermediateChunkedPrefill) return std::vector<int>(batch, 0);
    if (!selection.sampledTokens.empty()) return selection.sampledTokens;
    std::vector<int> tokens(batch);
    if (!selection.candidates.dims.empty()) {
        for (int b = 0; b < batch; ++b) {
            if (configs[b].prefill_chunk_remaining > 0) continue;
            tokens[b] = selection.greedy
                ? (int)((float *)selection.candidates.cpuData)[b * 2]
                : LLMSamplingOnly(selection.candidates, b, configs[b]);
        }
        return tokens;
    }
    if (logits.dataDevice == CUDA) {
        ResetLogitsOfEOS(batch, &logits, pastKeyValues, configs);
        return SampleCudaLogits(logits, configs, &lastTokens);
    }
    const int vocab = logits.dims.back();
    for (int b = 0; b < batch; ++b)
        if (configs[b].output_logits && retLogits && b < (int)retLogits->size() && (*retLogits)[b]) {
            const float *row = (const float *)logits.cpuData + (size_t)b * vocab;
            (*retLogits)[b]->assign(row, row + vocab);
        }
    ResetLogitsOfEOS(batch, &logits, pastKeyValues, configs);
    for (int b = 0; b < batch; ++b) {
        if (configs[b].prefill_chunk_remaining > 0) { tokens[b] = 0; continue; }
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

void NaiveN05FlashModel::RunLocalPrefillRouter(Data &normed, Data &routerWeight,
        const std::vector<int> &rows, int worldRows, int epRanks, Data &routerInput, Data &router) {
#ifdef USE_CUDA
    const int length = rows.size(), embed_dim = normed.dims.back();
    const int gpu = normed.dataDeviceIds.at(0);
    // Keep the existing world-shard GEMM shapes and row positions:
    // changing FP32 reduction order can change discrete expert routing.
    // Only this owner's rows are populated/read; no activation gather.
    const int experts = routerWeight.dims[0];
    router.dataType = FLOAT32; router.UpdateUnitSize();
    router.Resize({length, experts}); router.ToDevice(CUDA, {gpu}, false); router.Allocate(false);
    Data projectedRouter;
    for (int shard = 0; shard < epRanks; ++shard) {
        const int first = (int)((int64_t)worldRows * shard / epRanks);
        const int end = (int)((int64_t)worldRows * (shard + 1) / epRanks);
        auto begin = std::lower_bound(rows.begin(), rows.end(), first);
        auto finish = std::lower_bound(begin, rows.end(), end);
        if (begin == finish) continue;
        Data input(FLOAT32, {end - first, embed_dim});
        input.FakeFrom(routerInput, 0);
        // Fully owned shards are overwritten below; only holes need clearing.
        if (finish - begin != end - first) FastllmCudaMemset0(input.cudaData, input.GetBytes());
        auto forEachSpan = [&](auto copy) {
            for (auto at = begin; at != finish;) {
                auto next = at + 1;
                while (next != finish && *next == *(next - 1) + 1) ++next;
                copy(size_t(at - rows.begin()), size_t(*at - first), (int)(next - at));
                at = next;
            }
        };
        forEachSpan([&](size_t localRow, size_t shardRow, int count) {
            Data source(BFLOAT16, {count, embed_dim}), destination;
            source.FakeFrom(normed, localRow * embed_dim * sizeof(uint16_t));
            destination.FakeFrom(input, shardRow * embed_dim * sizeof(float));
            ToDataType(source, destination, FLOAT32);
        });
        if (!FastllmCudaNaiveRouterFP32(input, routerWeight, projectedRouter, routerInput))
            Linear(input, routerWeight, Data(), projectedRouter);
        forEachSpan([&](size_t localRow, size_t shardRow, int count) {
            FastllmCudaCopyFromDeviceToDevice((float *)router.cudaData + localRow * experts,
                (float *)projectedRouter.cudaData + shardRow * experts, size_t(count) * experts * sizeof(float));
        });
    }
#else
    throw std::runtime_error("Local prefill routing requires CUDA.");
#endif
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
    AssertInFastLLM(dataType == DataType::BFLOAT16 &&
                    (FastllmNaiveKVType(kvCacheDataType) || (kvCacheDataType == FP4_E2M1 && !tpPagedCaches.empty())),
                    "Naive requires BF16 compute; warm up CUDA TP before using FP4 DSA KV.");
    AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1 &&
                    (batch || (int)pastKeyValues.size() == block_cnt),
                    "Naive-N0.5 expects one unpadded sequence and a complete KV cache.");
    const bool tensorParallel = tpRank >= 0;
    const bool decodeWorkspace = workspace && workspace->capacity > 0;
    const bool verifying = capture && capture->verifying;
    const bool graphVerify = decodeWorkspace && verifying;
    const bool batchDecode = batch && batch->Decode();
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
    auto trackCommunication = [&](Data &data) {
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
    };
    auto reduce = [&](Data &data, Data *residual = nullptr,
                      const Data *normWeight = nullptr, Data *normOutput = nullptr) {
        trackCommunication(data);
        if (!tensorParallel) return false;
#ifndef USE_ROCM
        if (residual && decodeWorkspace &&
            (!verifying || inputIds.dims[1] <= 8 || (inputIds.dims[1] > 16 && inputIds.dims[1] <= 128))) {
            normOutput->dataType = BFLOAT16;
            normOutput->Resize(residual->dims);
            normOutput->ToDevice(data.dataDevice, data.dataDeviceIds, false);
            normOutput->Allocate(false);
            if (FastllmCudaCustomAllReduceAddRMSNorm(data.cudaData, residual->cudaData,
                    (const float *)normWeight->cudaData, normOutput->cudaData,
                    data.Count(0), embed_dim, rms_norm_eps, data.dataType, gpu)) return true;
        }
#endif
        if ((batchDecode || verifying) && inputIds.dims[1] > 1) {
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
    const int epRanks = tensorParallel ? (int)tpDevices.size() : 1;
    // SharedFlex owns replicated TP rows; MegaMoE may shard uneven world rows.
    const bool useSharedFlex = !attentionParallel && tensorParallel && tpSharedFlex && tpSharedFlex->Supports(length);
    const bool shardMoe = !useSharedFlex && tensorParallel && tpEp8MegaState && (decodeWorkspace || verifying) &&
        length >= epRanks;
    // Small prefill chunks use MegaMoE's replicated (up to 64 rows) path.
    // Larger chunks dispatch contiguous token shards, including uneven tails.
    const bool shardPrefillRouter = !useSharedFlex && tensorParallel && tpEp8MegaState &&
        !decodeWorkspace && !verifying && length > 64;
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
    if (batchDecode || verifying)
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
    buf.directSamplingLogits = false;
    Data *prefillScores = tensorParallel && !tpPrefillScores.empty() ? &tpPrefillScores.at(tpRank) : nullptr;
    // Raw fused kernels require the TP path's prebound CUDA norm weights.
    // The generic path keeps the dispatcher that transfers weights on demand.
    const bool fusePrefillNorm = tensorParallel && !decodeWorkspace && !verifying && length > 1;
    bool prefillNormReady = false, prefillResidualSharded = false;
    // Fully data-parallel attention can keep activations local through the
    // consecutive MoE layers, including graph verification. Hybrid attention-TP
    // and unsupported layer layouts retain the existing world-row path.
    // Each owner keeps its real rows, including empty owners. The shared EP
    // arena and dispatch rounds use the maximum local row count on every rank.
    const bool localPrefillRows = attentionParallel && fusePrefillNorm && shardPrefillRouter;
    const bool localMoeRows = attentionParallel && AttentionTPSize() == 1 &&
        tpEp8MegaState && (decodeWorkspace || localPrefillRows) && block_cnt > 1 && !moeLayers[0] &&
        std::all_of(moeLayers.begin() + 1, moeLayers.end(), [](bool moe) { return moe; });
    const bool compactPrefillHead = localMoeRows && !decodeWorkspace && !verifying &&
        (!capture || !capture->collectHidden);
    // A request must keep the same FFN reduction and local router when another
    // KV owner joins a prefill batch.
    const bool stablePrefill = attentionParallel && attentionParallel->stablePrefillReduction &&
        !decodeWorkspace && !verifying && (batch ? !batchDecode : length > 1);
    const bool canonicalPrefillRouter = localMoeRows && !decodeWorkspace && !stablePrefill &&
        std::any_of(attentionParallel->rows.localToGlobal.begin(), attentionParallel->rows.localToGlobal.end(),
            [&](const auto &rows) { return rows.size() != attentionParallel->rows.localToGlobal.front().size(); });
    int maxLocalRows = 0;
    int minLocalRows = 0;
    if (localMoeRows && !attentionParallel->rows.localToGlobal.empty()) {
        minLocalRows = attentionParallel->rows.localToGlobal.front().size();
        for (const auto &rows : attentionParallel->rows.localToGlobal) {
            maxLocalRows = std::max(maxLocalRows, (int)rows.size());
            minLocalRows = std::min(minLocalRows, (int)rows.size());
        }
    }
    const bool localDense = pdPrefillLimits.chunk && compactPrefillHead && workspace &&
        !embedding && inputIds.dataDevice == CPU && inputIds.dataType == FLOAT32 &&
        GetCudaEmbedding() && !GetLowMemMode() && FastllmCudaHasBlackwellIndexerAttention();
    Data &hidden = localDense ? attentionParallel->ranks[tpRank].hidden : buf.hidden;
    Data &q = buf.q;
    Data &qkv = buf.qkv;
    Data &routerInput = buf.routerInput;
    Data &router = buf.router;
    Data &expertIndex = buf.expertIndex;
    Data &expertScore = buf.expertScore;
    Data &w1 = buf.w1;
    Data &w2 = buf.w2;
    Data &w3 = buf.w3;
    Data &tempInput = buf.tempInput;
    Data &tempOutput = buf.tempOutput;
    Data &moeInputTemp = buf.moeInputTemp;
    Data &moeOutputTemp = buf.moeOutputTemp;
    Data &positions = buf.positions;
    if (!decodeWorkspace) {
        positions.CopyFrom(positionIds);
        ToDataType(positions, DataType::FLOAT32);
    }
    if (!decodeWorkspace && verifying && tensorParallel) {
        Data &localPositions = attentionParallel ? attentionParallel->ranks[tpRank].positions : positions;
        if (localPositions.Count(0)) {
            localPositions.ToDevice(DataDevice::CUDA, std::vector<int>{gpu});
            FastllmCudaNaiveVerifyLengths(localPositions, attentionParallel ? attentionParallel->ranks[tpRank].liveKeys : buf.liveKeys);
        }
    }
    if (localDense) {
        const auto &rows = attentionParallel->rows.localToGlobal[tpRank];
        if (rows.empty()) {
            for (Data *data : {&hidden, &attentionParallel->ranks[tpRank].normalized}) {
                data->dataType = BFLOAT16;
                data->Resize({1, 0, embed_dim});
                data->ToDevice(CUDA, {gpu}, false);
            }
        } else {
            const auto *ids = reinterpret_cast<const float *>(inputIds.cpuData);
            std::vector<float> selected;
            selected.reserve(rows.size());
            for (int row : rows) selected.push_back(ids[row]);
            Data localIds(FLOAT32, {1, int(rows.size())}, selected);
            EmbeddingDirect(localIds, localWeight("model.embed_tokens.weight"), hidden);
        }
    } else if (embedding) hidden.CopyFrom(*embedding);
    // A persistent verification workspace keeps BF16 storage. Embedding's
    // FP32 intermediate cannot reuse that allocation at the same element count.
    else if (decodeWorkspace || (workspace && GetCudaEmbedding() && !GetLowMemMode()))
        EmbeddingDirect(inputIds, localWeight("model.embed_tokens.weight"), hidden);
    else Embedding(inputIds, localWeight("model.embed_tokens.weight"), hidden);
    ToDataType(hidden, DataType::BFLOAT16);
    const auto *projectionBackend = attentionParallel && (decodeWorkspace || (!verifying && !batchDecode && length > 1))
        ? FastllmCudaGetMxfp8ProjectionBackend() : nullptr;
    const int maxProjectionWidth = std::max({embed_dim, full.heads * full.valueDim, sliding.heads * sliding.valueDim});
    auto activationFor = [&](const Data &weight, int rows) {
        // The local router arena is idle between a producer and its QKV/O GEMM.
        // Prefill scores can still be live in indexed attention.
        if (!decodeWorkspace) return projectionBackend->Borrow(weight, rows, routerInput, true);
        if (verifying) return buf.verifyProjectionScratch
            ? projectionBackend->Borrow(weight, rows, *buf.verifyProjectionScratch) : FastllmMxfp8Activation{};
        return projectionBackend->Bind(weight, rows, false, maxProjectionWidth, buf.projectionQuant);
    };
    auto projectAttention = [&](Data &input, Data &weight, const Data &bias, Data &output,
                                FastllmMxfp8Activation prepared = {}) {
        if (prepared.values && projectionBackend && projectionBackend->TryProject(prepared, input, weight, bias, output))
            return;
        // The score and router arenas are both dead during prefill QKV/O projection.
        if (!decodeWorkspace && !verifying && length >= 1024) {
            const Data *scratch = &routerInput;
            if (prefillScores && prefillScores->expansionBytes > routerInput.expansionBytes)
                scratch = prefillScores;
            CudaLinearScratchScope loan(scratch);
            Linear(input, weight, bias, output);
        } else {
            Linear(input, weight, bias, output);
        }
    };
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
    bool useSwaLayer = false, useSwaVerifyLayer = false;
#ifndef USE_ROCM
    useSwaLayer = !attentionParallel && tensorParallel && decodeWorkspace && !verifying && length == 1 &&
        tpSwaRingCache && !(capture && capture->collectHidden) &&
        FastllmCudaSwaLayerSupported(gpu);
    // Beyond sixteen rows, the existing full GEMMs outperform the fused
    // kernel's eight-row tiles. Keep that dispatch while the operator itself
    // retains the larger shape contract for explicit callers.
    useSwaVerifyLayer = !attentionParallel && tensorParallel && decodeWorkspace && verifying && batch && length <= 16 &&
        capture->ringVerify && tpSwaRingCache && !tpFlexMoe &&
        (!tpEp8MegaState || useSharedFlex) && FastllmCudaSwaVerifyLayerSupported(gpu, length);
#endif
    bool useDsaVerifyLayer = false;
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
    useDsaVerifyLayer = !attentionParallel && indexerCompute == NaiveIndexerCompute::Auto && tensorParallel && decodeWorkspace && verifying && batch &&
        capture->ringVerify && length <= 16 && buf.capacity > 8192 && buf.capacity <= 65536 &&
        !tpFlexMoe && (!tpEp8MegaState || useSharedFlex) && !sparseKv &&
        !HasSeparateIndexer() && !indexerLayerSharding && indexFp8 && indexHeads == 16 &&
        indexDim == 128 && indexTopK == 2048 &&
        std::all_of(batch->lengths.begin(), batch->lengths.end(), [](int n) { return n >= 4 && n <= 8; });
#endif
    // Keep the previous FFN branch separate until every fused attention CTA has read the
    // old residual. The fused call adds it after all attention readers finish.
    Data *pendingMlp = nullptr, *pendingFeature = nullptr;
    bool fusedHead = false;
#ifndef USE_ROCM
    // Verification exports the materialized hidden state before the head.
    // Backends that update the residual internally keep their existing path;
    // admission happens before changing the final residual/norm contract.
    fusedHead = !attentionParallel && decodeWorkspace && (verifying || !(capture && capture->collectHidden)) &&
        (!tpEp8MegaState || useSharedFlex) &&
        FastllmCudaDecodeNormLinearPreferred(hidden, localWeight("model.norm.weight"),
            localWeight("lm_head.weight"), buf.logits);
#endif
    Data *headBranch = nullptr;
    bool localHiddenReady = localDense;
    const bool canReusePrefillProjections = !decodeWorkspace && !verifying && attentionParallel &&
        AttentionTPSize() == 1 && (!capture || (!capture->collectHidden && capture->prefixes.empty())) &&
        FastllmCudaHasBlackwellIndexerAttention();
    const int dpGroup = attentionParallel ? attentionParallel->runtime.Layout().AttentionOwner(tpRank) : 0;
    const std::vector<GenerationConfig> noConfigs;
    std::optional<TargetBatch> groupBatch;
    if (attentionParallel) {
        const auto &group = attentionParallel->groups[dpGroup];
        groupBatch.emplace(TargetBatch{group.kv, batch ? batch->configs : noConfigs, group.lengths,
            block_cnt, nullptr, &group.sequenceIndices, batch && batch->paddingConfig ? batch->paddingConfig : &config});
    }
    const TargetBatch *attentionBatch = groupBatch ? &*groupBatch : batch;
    const bool attentionBatchDecode = attentionParallel ? attentionBatch && attentionBatch->Decode() : batchDecode;
    struct PrefillProjectionViews { Data key, value, attention; };
    std::optional<PrefillProjectionViews> projectionViews;
    FastllmMxfp8Activation nextQkv{};
    for (int layer = 0; layer < block_cnt; layer++) {
        const auto preparedQkv = nextQkv;
        nextQkv = {};
        FastllmMxfp8Activation preparedO{};
        const bool localMoe = localMoeRows && moeLayers[layer];
        const bool ownerRows = localMoe || (localDense && layer == 0);
        auto *localDP = ownerRows ? &attentionParallel->ranks[tpRank] : nullptr;
        const int length = ownerRows ? attentionParallel->rows.localToGlobal[tpRank].size() : inputIds.dims[1];
        Data &hidden = ownerRows ? localDP->hidden : buf.hidden;
        Data &normed = ownerRows ? localDP->normalized : buf.normed;
        Data &projected = ownerRows ? localDP->projected : buf.projected;
        // World-row decode reuses the registered attention projection buffer;
        // local EP output needs no peer AllGather registration.
        Data &moeOutput = !localMoe && shardMoe ? buf.projected : buf.moeOutput;
        const bool shardRouter = !localMoe && (shardMoe || shardPrefillRouter);
        if (localMoe && !localHiddenReady) {
            for (Data *data : {&hidden, &normed}) {
                data->dataType = BFLOAT16; data->UpdateUnitSize();
                data->Resize({1, length, embed_dim}); data->ToDevice(CUDA, {gpu}, false);
                if (length) data->Allocate(false);
            }
            if (length) {
                attentionParallel->transfer->Select(tpRank, buf.hidden.cudaData, hidden.cudaData, size_t(embed_dim) * 2);
                attentionParallel->transfer->Select(tpRank, buf.normed.cudaData, normed.cudaData, size_t(embed_dim) * 2);
            }
            localHiddenReady = true;
        }
        const bool shardPrefillNorm = !attentionParallel && fusePrefillNorm && shardPrefillRouter &&
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
        Data &postAttentionNorm = localWeight(prefix + ".post_attention_layernorm.weight");
        const bool initialNormPending = layer == 0 && useDsaVerifyLayer && !slidingLayers[layer] && HasLocalIndexer(layer, tpRank);
        const bool denseFused=useSharedFlex && verifying && !moeLayers[layer] &&
            tpSharedFlex->PrepareDense(tpRank,length,hidden,
                localWeight(prefix+".mlp.gate_proj.weight"),localWeight(prefix+".mlp.up_proj.weight"),
                localWeight(prefix+".mlp.down_proj.weight"),localWeight(prefix+".post_attention_layernorm.weight"),
                rms_norm_eps,workspace?buf.denseGate:buf.w1,workspace?buf.denseDown:buf.w2);
        bool attentionLayerReady = false, attentionRouterReady = false;
        FastllmMoePreparedInput preparedInput{};
        const FastllmMoePreparedInput *prepared = nullptr;
        bool attentionInputPrepared = false;
        bool attentionNormReady = ownerRows && !length;
        // Attention consumes its owning DP group's rows. The local MoE path
        // retains that order; other layouts use the canonical world rows.
        if (((!decodeWorkspace && !prefillNormReady) || layer == 0) && !initialNormPending && (!ownerRows || length))
            norm(hidden, localWeight(prefix + ".input_layernorm.weight"), normed);
        const int globalLength = length;
        if (!attentionParallel || !attentionParallel->groups[dpGroup].lengths.empty()) {
        const int length = attentionParallel ? attentionParallel->rows.localToGlobal[dpGroup].size() : globalLength;
        const TargetBatch *batch = attentionBatch;
        Data &positions = attentionParallel ? attentionParallel->ranks[tpRank].positions : buf.positions;
        Data &normed = attentionParallel ? attentionParallel->ranks[tpRank].normalized : buf.normed;
        Data &projected = attentionParallel ? attentionParallel->ranks[tpRank].projected : buf.projected;
        const int pastLength = batch && !batch->Key(0, 0).dims.empty() ? batch->Key(0, 0).dims[1] :
            (pastKeyValues.empty() || pastKeyValues[0].first.dims.empty() ? 0 : pastKeyValues[0].first.dims[1]);
        if (attentionParallel && !ownerRows) {
            normed.dataType = BFLOAT16; normed.UpdateUnitSize();
            normed.Resize({1, length, embed_dim}); normed.ToDevice(CUDA, {gpu}, false); normed.Allocate(false);
            attentionParallel->transfer->Select(tpRank, buf.normed.cudaData, normed.cudaData, size_t(embed_dim) * 2);
        }
        Data &liveKeys = attentionParallel ? attentionParallel->ranks[tpRank].liveKeys : buf.liveKeys;
        const int attentionRank = tensorParallel ? tpRank % AttentionTPSize() : 0;
        IndexerGather indexerGather;
        if (attentionParallel) indexerGather = [&](const void *input, void *output, size_t count) {
            auto &runtime = attentionParallel->runtime;
            runtime.AllGather(runtime.Layout().attention[dpGroup], tpRank, input, output,
                count, parallel::ScalarType::Int32);
        };
        const bool localIndexer = HasLocalIndexer(layer, tpRank);
        const bool sparseLayer = sparseKv && !slidingLayers[layer];
        const bool separateIndex = HasSeparateIndexer() && !slidingLayers[layer];
        const bool contextSharded = indexerContextSharding && separateIndex && AttentionTPSize() > 1;
        auto cfg = slidingLayers[layer] ? sliding : full;
        if (tensorParallel) {
            cfg.heads /= AttentionTPSize();
            cfg.kvHeads = std::max(1, cfg.kvHeads / AttentionTPSize());
        }
        int rotaryDim = (int)(cfg.headDim * partialRotary);
#ifndef USE_ROCM
        if (useSharedFlex && moeLayers[layer] &&
            tpSharedFlex->PreparedInput(tpRank, length, preparedInput, expertIndex, expertScore,
                localWeight(prefix + ".mlp.gate.e_score_correction_bias"), hidden,
                norm_topk_prob, routed_scaling_factor) &&
            FastllmCudaSwaPreparedRouterSupported(gpu, false, length)) {
            // Keep the established prepared-input path if only cooperative
            // selection is unavailable on any rank of this TP group.
            if (preparedInput.routed && !FastllmCudaSwaPreparedRouterSupported(gpu, true))
                preparedInput.routed = false;
            prepared = &preparedInput;
        }
#endif
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
        if (useDsaVerifyLayer && !slidingLayers[layer] && localIndexer && cfg.heads == 8 &&
            cfg.kvHeads == 1 && cfg.headDim == 192 && cfg.valueDim == 128 && rotaryDim == 64) {
            const std::string ip = ap + "indexer.";
            FastllmNaiveDSALayerInput input;
            input.branch = pendingMlp;
            input.hidden = &hidden;
            input.feature = pendingFeature;
            input.inputNorm = &localWeight(prefix + ".input_layernorm.weight");
            input.qkvWeight = &localWeight(ap + "mergeqkv.weight");
            input.qkvBias = &localWeight(ap + "mergeqkv.bias");
            input.indexKeyWeight = &localWeight(ip + "wk.weight");
            input.indexQueryWeight = &localWeight(ip + "wq.weight");
            input.indexWeight = &localWeight(ip + "weights_proj.weight");
            input.indexNorm = &localWeight(ip + "k_norm.weight");
            input.indexBias = &localWeight(ip + "k_norm.bias");
            input.outputWeight = &localWeight(ap + "o_proj.weight");
            input.postNorm = &localWeight(prefix + ".post_attention_layernorm.weight");
            input.sink = &localWeight(ap + "attention_sink_bias");
            input.positions = &positions;
            input.live = &buf.liveKeys;
            input.query = &q;
            input.attention = &buf.attn;
            input.projected = &projected;
            input.normed = &normed;
            input.scratch = &buf.decode;
            input.requests = batch->lengths;
            input.capacity = buf.capacity;
            input.eps = rms_norm_eps;
            input.theta = cfg.theta;
            input.valueScale = valueScale;
            const bool withRouter =
                moeLayers[layer] && (length == 1 || length < FastllmCudaGetLinearExactBatchThreshold());
            if (withRouter) {
                input.routerWeight = &localWeight(prefix + ".mlp.gate.weight");
                input.router = &router;
            }
            const void *const *cache =
                buf.cacheTable ? buf.cacheTable + size_t(layer) * buf.tableRows * 4 : nullptr;
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                Data &key = *batch->Key(sequence, layer).multiDeviceDatas.at(gpu);
                Data &value = *batch->Value(sequence, layer).multiDeviceDatas.at(gpu);
                key.kvCacheFp8Indexer = fp8IndexerCache;
                for (int row = 0; row < batch->lengths[sequence]; ++row)
                    input.caches.push_back({&key, &value, nullptr, cache ? cache + sequence * 4 : nullptr});
            }
            attentionLayerReady = FastllmCudaNaiveDSALayer(input, gpu, withRouter ? prepared : nullptr, denseFused);
            if (attentionLayerReady) {
                trackCommunication(projected);
                attentionRouterReady = withRouter;
                attentionInputPrepared = withRouter && prepared;
                pendingMlp = pendingFeature = nullptr;
            }
        }
#endif
        if (pendingMlp) {
#ifndef USE_ROCM
            const void *const *cache = buf.cacheTable ? buf.cacheTable +
                (size_t)layer * buf.tableRows * (HasSeparateIndexer() ? 8 : 4) : nullptr;
            if (slidingLayers[layer] && useSwaVerifyLayer) {
                std::vector<FastllmNaiveRopeCacheRow> rows;
                rows.reserve(length);
                for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                    Data &key = *batch->Key(sequence, layer).multiDeviceDatas.at(gpu);
                    Data &value = *batch->Value(sequence, layer).multiDeviceDatas.at(gpu);
                    for (int row = 0; row < batch->lengths[sequence]; ++row) {
                        FastllmNaiveRopeCacheRow binding{&key, &value, nullptr};
                        binding.cache = cache ? cache + sequence * 4 : nullptr;
                        binding.verifyOffset = row;
                        rows.push_back(binding);
                    }
                }
                const bool withRouter = length == 1 || length < FastllmCudaGetLinearExactBatchThreshold();
                attentionLayerReady = FastllmCudaSwaVerifyLayerPrepared(*pendingMlp, hidden,
                    localWeight(prefix + ".input_layernorm.weight"),
                    localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"),
                    localWeight(ap + "o_proj.weight"),
                    localWeight(prefix + ".post_attention_layernorm.weight"),
                    localWeight(ap + "attention_sink_bias"), positions, buf.liveKeys, rows,
                    q, buf.attn, projected, normed, pendingFeature,
                    withRouter ? &localWeight(prefix + ".mlp.gate.weight") : nullptr,
                    withRouter ? &router : nullptr,
                    {cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim, rotaryDim, window,
                     rms_norm_eps, cfg.theta, valueScale}, gpu, withRouter ? prepared : nullptr);
                attentionRouterReady = attentionLayerReady && withRouter;
                attentionInputPrepared = attentionRouterReady && prepared;
            } else if (slidingLayers[layer] && useSwaLayer) {
                Data &key = batch ? batch->Key(0, layer) : pastKeyValues[layer].first;
                Data &value = batch ? batch->Value(0, layer) : pastKeyValues[layer].second;
                attentionLayerReady = FastllmCudaSwaLayer(*pendingMlp, hidden,
                    localWeight(prefix + ".input_layernorm.weight"),
                    localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"),
                    localWeight(ap + "o_proj.weight"),
                    localWeight(prefix + ".post_attention_layernorm.weight"),
                    localWeight(prefix + ".mlp.gate.weight"),
                    localWeight(ap + "attention_sink_bias"), positions, buf.liveKeys,
                    *key.multiDeviceDatas.at(gpu), *value.multiDeviceDatas.at(gpu), cache,
                    q, buf.attn, projected, normed, router,
                    {cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim, rotaryDim, window,
                     rms_norm_eps, cfg.theta, valueScale}, gpu);
                attentionRouterReady = attentionLayerReady;
            }
            if (attentionLayerReady) trackCommunication(projected);
#endif
            if (!attentionLayerReady) {
                FastllmCudaNaiveAddDecodeRMSNorm(hidden, *pendingMlp,
                    localWeight(prefix + ".input_layernorm.weight"), rms_norm_eps, normed);
                if (pendingFeature) Copy(hidden, *pendingFeature);
            }
            pendingMlp = pendingFeature = nullptr;
        }
        if (!attentionLayerReady) {
            if (initialNormPending) norm(hidden, localWeight(prefix + ".input_layernorm.weight"), normed);
        const auto merged = weight.weight.find(ap + "mergeqkv.weight");
        const bool mergedQkv = merged != weight.weight.end() && !merged->second.dims.empty();
        const bool batchRope = attentionBatchDecode && (batch->Size() > 1 || buf.cacheTable) &&
            decodeWorkspace && !verifying;
        const bool reusePrefillQkv = canReusePrefillProjections && mergedQkv && batch && !attentionBatchDecode;
        if (reusePrefillQkv && layer == 0) {
            // Full and sliding layers share the projection buffer, but can have
            // different output widths. Reserve the largest actual merged shard
            // once; growing it at the next layer strands the first allocation.
            int width = 0;
            for (int next = 0; next < block_cnt; ++next) {
                const auto &w = localWeight("model.layers." + std::to_string(next) + ".self_attn.mergeqkv.weight");
                if (w.dims.size() == 2) width = std::max(width, w.dims[0]);
            }
            if (width && qkv.expansionSize < size_t(length) * width) {
                qkv.dataType = BFLOAT16;
                qkv.Resize({1, length, width});
                qkv.ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
                qkv.Allocate(false);
            }
        }
        if (reusePrefillQkv && !projectionViews) projectionViews.emplace();
        const int keyWidth = cfg.kvHeads * cfg.headDim, valueWidth = cfg.kvHeads * cfg.valueDim;
        const bool reuseProjected = reusePrefillQkv && keyWidth + valueWidth <= embed_dim;
        Data &k = reuseProjected ? projectionViews->key : buf.k;
        Data &v = reuseProjected ? projectionViews->value : buf.v;
        if (reuseProjected) {
            // K and V expire after the cache append. The local O projection
            // overwrites this buffer only after all attention rows are ready.
            projected.dataType = BFLOAT16;
            projected.Resize({1, length, embed_dim});
            projected.ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
            projected.Allocate(false);
            k.Resize({1, length, keyWidth});
            k.FakeFrom(projected, 0);
            v.Resize({1, length, valueWidth});
            v.FakeFrom(projected, size_t(length) * keyWidth * sizeof(uint16_t));
        }
            bool swaRopePrepared = false, swaAttentionPrepared = false, trySingleSwaQkv = false;
            bool tryVerifyDsaFront = false, tryVerifySwaQkv = false;
#ifndef USE_ROCM
            static const bool swaQkvEnabled = [] {
                const char *value = std::getenv("FASTLLM_NAIVE_SWA_QKV_FUSION");
                return !value || std::string(value) == "1";
            }();
            trySingleSwaQkv = !attentionParallel && swaQkvEnabled && !batch && tensorParallel && decodeWorkspace &&
                !verifying && length == 1 && slidingLayers[layer] && tpSwaRingCache &&
                FastllmCudaRuntimeArch() == 103;
            const bool verifyFrontend = !attentionParallel && indexerCompute == NaiveIndexerCompute::Auto && verifying && batch && capture->ringVerify && tensorParallel &&
                decodeWorkspace && mergedQkv && FastllmCudaRuntimeArch() == 103;
            tryVerifyDsaFront = verifyFrontend && length <= 7 && !slidingLayers[layer] && localIndexer &&
                !sparseLayer && buf.capacity <= 2048;
            tryVerifySwaQkv = verifyFrontend && length <= 16 && swaQkvEnabled && slidingLayers[layer];
#endif
        if (mergedQkv) {
                if (!batchRope && !trySingleSwaQkv && !tryVerifyDsaFront && !tryVerifySwaQkv)
                projectAttention(normed, localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"), qkv, preparedQkv);
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
#ifndef USE_ROCM
                if (trySingleSwaQkv) {
                    positions.ToDevice(q.dataDevice, q.dataDeviceIds);
                    std::vector<FastllmNaiveRopeCacheRow> caches{{
                        pastKeyValues[layer].first.multiDeviceDatas.at(gpu),
                        pastKeyValues[layer].second.multiDeviceDatas.at(gpu), nullptr,
                        buf.cacheTable ? buf.cacheTable + (size_t)layer * buf.tableRows *
                            (HasSeparateIndexer() ? 8 : 4) : nullptr}};
                    swaRopePrepared = FastllmCudaNaiveBatchQkvRopeAppendCache(normed,
                        localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"),
                        q, k, v, positions, caches, buf.liveKeys, cfg.heads, cfg.kvHeads,
                        cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale,
                        window, true, projected);
                }
#endif
                if (trySingleSwaQkv && !swaRopePrepared)
                    Linear(normed, localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"), qkv);
        } else {
            projectAttention(normed, localWeight(ap + "q_proj.weight"), localWeight(ap + "q_proj.bias"), q);
            projectAttention(normed, localWeight(ap + "k_proj.weight"), localWeight(ap + "k_proj.bias"), k);
            projectAttention(normed, localWeight(ap + "v_proj.weight"), localWeight(ap + "v_proj.bias"), v);
        }
        const bool reuseAttention = reusePrefillQkv && qkv.dataType == BFLOAT16 && qkv.dims.size() == 3 &&
            qkv.dims.back() >= cfg.heads * cfg.valueDim;
        Data &attn = reuseAttention ? projectionViews->attention : buf.attn;
        if (reuseAttention) {
            // Each request consumes its packed QKV before producing attention.
            // Requests advance in row order; the narrower output prefix cannot
            // reach the following request's unread projection. Deferred DSA
            // consumes all QKV slices before publishing any output.
            attn.Resize({1, length, cfg.heads * cfg.valueDim});
            attn.FakeFrom(qkv, 0);
        }
        AssertInFastLLM(q.dataDevice == DataDevice::CUDA,
                        "Naive-N0.5 attention requires --device cuda.");
        positions.ToDevice(q.dataDevice, q.dataDeviceIds);
        // Prepare all independent caches/index lists first, then attend to the
        // complete decode batch in one call. Prefill and verification retain
        // their causal multi-query paths.
        const bool batchVerifyDSA = verifying &&
            (batch ? std::all_of(batch->lengths.begin(), batch->lengths.end(),
                [](int count) { return count > 0 && count <= 8; }) : length > 0 && length <= 8);
        const bool batchedDSA = !slidingLayers[layer] && (!graphVerify || batch || separateIndex) &&
            indexTopK <= 2048 && cfg.headDim <= 256 &&
            (cfg.valueDim == 128 || cfg.valueDim == 256) &&
            (batchVerifyDSA || (!verifying &&
                (batch ? attentionBatchDecode : length == 1)));
        std::vector<FastllmNaiveDSARow> dsaRows;
        if (batchedDSA) dsaRows.reserve(length);
        const bool verifySWA = verifying && (!graphVerify || (batch && capture->ringVerify)) && window == 128 &&
            cfg.headDim == 192 && cfg.valueDim == 128 &&
            (attentionParallel || cfg.heads < 64 || cfg.heads / cfg.kvHeads != 8);
        const bool ringBatchDecode = slidingLayers[layer] && tpSwaRingCache && attentionBatchDecode;
        const bool eagerSwa = ringBatchDecode && !verifying && !decodeWorkspace && !tpSwaPrefillWorkspaces.empty();
        const bool batchedSWA = (slidingLayers[layer] && verifySWA) ||
            (ringBatchDecode && decodeWorkspace) || eagerSwa;
        std::vector<FastllmNaiveSWARow> swaRows;
        if (batchedSWA) swaRows.reserve(length);
        auto prepareIndexKey = [&](Data &normed, TargetWorkspace &buf, bool alreadyProjected = false) {
            if (!localIndexer) return;
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
            bool batchRopePrepared = false, dsaFrontPrepared = false;
        bool indexProjectionPrepared = false;
        bool batchNeedsIndex = false;
        // DSA needs a second binding for the independent Indexer. Keep SWA's
        // four-pointer row so existing projection/RoPE fusion stays eligible.
        const int cacheStride = separateIndex ? 8 : 4;
        const void *const *layerCache = buf.cacheTable ? buf.cacheTable + (size_t)layer * buf.tableRows * (HasSeparateIndexer() ? 8 : 4) : nullptr;
        const unsigned char *layerMaps = buf.cacheMaps ? buf.cacheMaps + (size_t)layer * buf.tableRows * FastllmNaiveDSAMapBytes : nullptr;
        if (batch) {
            while ((int)buf.sequences.size() < batch->Size())
                buf.sequences.emplace_back(new TargetWorkspace());
        }
        std::vector<std::unique_ptr<Data>> separateIndexCaches;
        if (separateIndex && localIndexer) {
            for (int sequence = 0; sequence < (batch ? batch->Size() : 1); ++sequence)
                separateIndexCaches.emplace_back(new Data(SeparateIndexer(layer, tpRank,
                    *(batch ? batch->Key(sequence, layer) : pastKeyValues[layer].first).multiDeviceDatas.at(gpu))));
        }
        auto indexCache = [&](int sequence) -> Data * {
            return separateIndex ? separateIndexCaches[sequence].get() :
                batch->Key(sequence, layer).multiDeviceDatas.at(gpu);
        };
        auto indexTable = [&](int sequence) {
            return layerCache ? layerCache + sequence * cacheStride + (separateIndex ? 4 : 0) : nullptr;
        };
        if (batchRope) {
            std::vector<FastllmNaiveRopeCacheRow> caches;
            caches.reserve(batch->Size());
            // Project/normalize the complete batch, then borrow individual K rows.
            // Cache descriptors retain the addresses, not these temporary views.
            batchNeedsIndex = !slidingLayers[layer] &&
                std::any_of(buf.sequences.begin(), buf.sequences.begin() + batch->Size(),
                    [&](const std::unique_ptr<TargetWorkspace> &part) { return part->capacity > indexTopK; });
                bool tryDsaFront = false;
#ifndef USE_ROCM
                tryDsaFront = !attentionParallel && indexerCompute == NaiveIndexerCompute::Auto && !slidingLayers[layer] && !batchNeedsIndex && localIndexer &&
                    mergedQkv && !sparseLayer && length <= 8 && FastllmCudaRuntimeArch() == 103;
                if (tryDsaFront) {
                    // Borrow the existing projection buffers. Graph warmup owns
                    // their capacity; inference never allocates packed weights.
                    for (auto item : {std::make_pair(&qkv, (cfg.heads + cfg.kvHeads) * cfg.headDim + cfg.kvHeads * cfg.valueDim),
                                      std::make_pair(&buf.indexKey, indexDim)}) {
                        item.first->dataType = normed.dataType;
                        item.first->Resize({1, length, item.second});
                        item.first->ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
                        item.first->Allocate(false);
                    }
                }
            if (batchNeedsIndex && localIndexer) {
                const std::string ip = ap + "indexer.";
                // O projection is not live yet. Reuse its full-width buffer for
                // the packed GEMM; graph shapes already share this allocation.
                indexProjectionPrepared = FastllmCudaIndexerProjection(normed,
                    localWeight(ip + "wq.weight"), localWeight(ip + "wk.weight"),
                    localWeight(ip + "weights_proj.weight"), buf.indexQ, buf.indexKey,
                    buf.indexWeights, projected);
            }
#endif
                if (!tryDsaFront) prepareIndexKey(normed, buf, indexProjectionPrepared);
            std::vector<Data> indexKeys(batch->Size());
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                if (localIndexer) {
                    indexKeys[sequence].Resize({1, 1, indexDim});
                    indexKeys[sequence].FakeFrom(buf.indexKey,
                        (size_t)sequence * indexDim * buf.indexKey.unitSize);
                }
                caches.push_back({batch->Key(sequence, layer).multiDeviceDatas.at(gpu),
                    batch->Value(sequence, layer).multiDeviceDatas.at(gpu),
                    localIndexer ? &indexKeys[sequence] : nullptr,
                    layerCache ? layerCache + sequence * cacheStride : nullptr,
                    separateIndex && localIndexer ? separateIndexCaches[sequence].get() : nullptr,
                    contextSharded ? attentionRank : 0});
            }
#ifndef USE_ROCM
                if (tryDsaFront) {
                    const std::string ip = ap + "indexer.";
                    std::vector<int> capacities;
                    for (int sequence = 0; sequence < batch->Size(); ++sequence)
                        capacities.push_back(buf.sequences[sequence]->capacity);
                    dsaFrontPrepared = FastllmCudaNaiveDSAFront(normed,
                        localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"),
                        localWeight(ip + "wk.weight"), localWeight(ip + "k_norm.weight"),
                        localWeight(ip + "k_norm.bias"), qkv, buf.indexKey, q, k, v, positions,
                        caches, capacities, buf.liveKeys, localWeight(ap + "attention_sink_bias"),
                        cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta,
                        valueScale, buf.decode, attn);
                    batchRopePrepared = dsaFrontPrepared;
                    if (!dsaFrontPrepared) prepareIndexKey(normed, buf, indexProjectionPrepared);
                }
                if (mergedQkv && !batchRopePrepared)
                batchRopePrepared = FastllmCudaNaiveBatchQkvRopeAppendCache(normed,
                    localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"),
                    q, k, v, positions, caches, liveKeys, cfg.heads, cfg.kvHeads,
                    cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale,
                    slidingLayers[layer] ? window : 0, slidingLayers[layer] && tpSwaRingCache, projected);
#endif
            if (!batchRopePrepared) {
                if (mergedQkv)
                    projectAttention(normed, localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"), qkv, preparedQkv);
                batchRopePrepared = FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions,
                    caches, liveKeys, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                    rotaryDim, cfg.theta, valueScale, slidingLayers[layer] ? window : 0,
                    mergedQkv ? &qkv : nullptr, slidingLayers[layer] && tpSwaRingCache);
            }
        }
        // Pack all verification queries, including ragged requests, into the
        // existing RoPE/cache kernel. Shared cache addresses are safe because
        // every descriptor owns a distinct candidate/chronological write slot.
        if (tensorParallel && verifying && (!decodeWorkspace || (batch && capture->ringVerify))) {
                if (tryVerifyDsaFront) {
                    for (auto item : {std::make_pair(&qkv, (cfg.heads + cfg.kvHeads) * cfg.headDim + cfg.kvHeads * cfg.valueDim),
                                      std::make_pair(&buf.indexKey, indexDim)}) {
                        item.first->dataType = normed.dataType;
                        item.first->Resize({1, length, item.second});
                        item.first->ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
                        item.first->Allocate(false);
                    }
                } else prepareIndexKey(normed, buf);
            std::vector<FastllmNaiveRopeCacheRow> caches;
            std::vector<Data> indexRows(length);
            caches.reserve(length);
            int offset = 0;
            for (int sequence = 0; sequence < (batch ? batch->Size() : 1); ++sequence) {
                const int count = batch ? batch->lengths[sequence] : length;
                Data &key = *(batch ? batch->Key(sequence, layer) : pastKeyValues[layer].first).multiDeviceDatas.at(gpu);
                Data &value = *(batch ? batch->Value(sequence, layer) : pastKeyValues[layer].second).multiDeviceDatas.at(gpu);
                key.kvCacheFp8Indexer = !separateIndex && fp8IndexerCache && localIndexer;
                // Eager contiguous caches can end inside this verification
                // block. Unlike AppendCache, the fused writer never grows them.
                const int needed = slidingLayers[layer] && capture->ringVerify
                    ? window + count : key.dims[1] + count;
                if (!decodeWorkspace) for (Data *cache : {&key, &value})
                    if (!cache->isPagedKVCache)
                        ReserveCache(*cache, (needed + 127) / 128 * 128, cache->dims[2]);
                for (int row = 0; row < count; ++row) {
                    Data *index = nullptr;
                    if (localIndexer) {
                        index = &indexRows[offset + row];
                        index->Resize({1, 1, indexDim});
                        index->FakeFrom(buf.indexKey, (size_t)(offset + row) * indexDim * buf.indexKey.unitSize);
                    }
                    FastllmNaiveRopeCacheRow item{&key, &value, index,
                        layerCache ? layerCache + sequence * cacheStride : nullptr,
                        separateIndex && localIndexer ? separateIndexCaches[sequence].get() : nullptr,
                        contextSharded ? attentionRank : 0};
                    if (slidingLayers[layer]) {
                        if (capture->ringVerify) item.verifyOffset = row;
                        else item.cacheOffset = key.dims[1] + row;
                    }
                    caches.push_back(item);
                }
                offset += count;
            }
                if (tryVerifyDsaFront) {
                    const std::string ip = ap + "indexer.";
                    dsaFrontPrepared = FastllmCudaNaiveDSAFront(normed,
                        localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"),
                        localWeight(ip + "wk.weight"), localWeight(ip + "k_norm.weight"),
                        localWeight(ip + "k_norm.bias"), qkv, buf.indexKey, q, k, v, positions,
                        caches, std::vector<int>(length, buf.capacity), liveKeys,
                        localWeight(ap + "attention_sink_bias"), cfg.heads, cfg.kvHeads,
                        cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale, buf.decode, attn, true);
                    if (!dsaFrontPrepared) {
                        Linear(normed, localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"), qkv);
                        prepareIndexKey(normed, buf);
                    }
                }
                if (tryVerifySwaQkv) {
                    if (length >= 2 && length <= 7) {
                        attn.dataType = BFLOAT16;
                        attn.Resize({1, length, cfg.heads * cfg.valueDim});
                        attn.ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
                        attn.Allocate(false);
                        swaAttentionPrepared = FastllmCudaNaiveSwaQkvAttention(normed,
                            localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"),
                            q, k, v, positions, caches, liveKeys, cfg.heads, cfg.kvHeads,
                            cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale, window,
                            localWeight(ap + "attention_sink_bias"), attn);
                    }
                    swaRopePrepared = swaAttentionPrepared || FastllmCudaNaiveBatchQkvRopeAppendCache(normed,
                        localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"),
                        q, k, v, positions, caches, liveKeys, cfg.heads, cfg.kvHeads,
                        cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale, window, false, projected);
                    if (!swaRopePrepared)
                        Linear(normed, localWeight(ap + "mergeqkv.weight"), localWeight(ap + "mergeqkv.bias"), qkv);
                }
                batchRopePrepared = dsaFrontPrepared || swaRopePrepared ||
                    FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, caches, liveKeys,
                        cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta,
                        valueScale, slidingLayers[layer] ? window : 0, mergedQkv ? &qkv : nullptr, false);
        }
        // A dynamic binding must reach every KV consumer. An unsupported
        // fused append may still execute eagerly, but cannot be captured.
        if (graphVerify && layerCache && !batchRopePrepared)
            tpDecodeState->ranks[tpRank]->ok = false;
        auto prepareIndexQuery = [&](Data &normed, Data &positions, Data &indexQ, Data &indexWeights) {
            const std::string ip = ap + "indexer.";
            Linear(normed, localWeight(ip + "wq.weight"), Data(), indexQ);
            FastllmCudaNaiveRope(indexQ, positions, indexHeads, indexDim, rotaryDim, cfg.theta);
            Linear(normed, localWeight(ip + "weights_proj.weight"), Data(), indexWeights);
            Mul(indexWeights, 1.0f / std::sqrt((float)indexHeads), indexWeights);
        };
        // SWA has no Indexer: neither stage an unused TopK buffer nor enter
        // the two-phase Indexer/broadcast schedule for its verification rows.
        const bool batchVerifyIndex = !slidingLayers[layer] && batchVerifyDSA && batch &&
            indexHeads == 16 && indexDim == 128 && indexTopK == 2048;
        std::vector<FastllmNaiveVerifyIndexerItem> verifyIndexItems;
            bool deferredDsaIndexer = false, fusedDsaPrepared = false;
        const bool batchIndexPrepared = batchRopePrepared && batchNeedsIndex && localIndexer;
        Data &batchIndexQ = buf.indexQ, &batchIndexWeights = buf.indexWeights;
        if (batchIndexPrepared || (batchVerifyIndex && localIndexer)) {
            if (indexProjectionPrepared)
                FastllmCudaNaiveRope(batchIndexQ, positions, indexHeads, indexDim, rotaryDim, cfg.theta);
            else prepareIndexQuery(normed, positions, batchIndexQ, batchIndexWeights);
        }
            auto runVerifyIndexer = [&] {
                if (FastllmCudaNaiveBatchVerifyIndexer(batchIndexQ, batchIndexWeights,
                    verifyIndexItems, indexFp8, buf.decode, liveKeys)) return;
                if (graphVerify && layerCache) tpDecodeState->ranks[tpRank]->ok = false;
                for (const auto &item : verifyIndexItems) {
                    Data iq(BFLOAT16, {1, item.rows, indexHeads * indexDim});
                    Data iw(BFLOAT16, {1, item.rows, indexHeads});
                    iq.FakeFrom(batchIndexQ, (size_t)item.offset * indexHeads * indexDim * sizeof(uint16_t));
                    iw.FakeFrom(batchIndexWeights, (size_t)item.offset * indexHeads * sizeof(uint16_t));
                    if (graphVerify) {
                        int sequence = 0, offset = 0;
                        while (offset < item.offset) offset += batch->lengths[sequence++];
                        auto &part = *buf.sequences[sequence];
                        FastllmCudaNaiveGraphVerifyIndexer(iq, iw, *item.key,
                            part.liveKeys, item.capacity, indexFp8, part.decode, *item.indices);
                    } else FastllmCudaNaiveVerifyIndexer(iq, iw, *item.key, indexHeads, indexDim,
                        item.past, indexTopK, indexFp8, *item.indices);
                }
            };
        const bool broadcastBatchIndex = (indexerLayerSharding || separateIndex) && batchRopePrepared && batchNeedsIndex;
        // Packed output views live for this layer only. Eager verification and
        // a later long prefill may use the same per-request workspace.
        std::vector<Data> packedIndices;
        if (broadcastBatchIndex || batchVerifyIndex) {
            packedIndices.resize(batch->Size());
            // One contiguous TopK exchange per layer, independent of the number
            // of requests. Graph shapes share the largest warmup allocation.
            buf.indexBroadcast.dataType = INT32;
            buf.indexBroadcast.Resize({batchVerifyIndex ? length : batch->Size(), indexTopK});
            buf.indexBroadcast.ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
            buf.indexBroadcast.Allocate(false);
            int offset = 0;
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                const int rows = batchVerifyIndex ? batch->lengths[sequence] : 1;
                Data &indices = packedIndices[sequence];
                indices.Resize({rows, indexTopK});
                indices.FakeFrom(buf.indexBroadcast, (size_t)offset * indexTopK * sizeof(int));
                offset += rows;
            }
        }
        bool batchIndexerPrepared = false;
        if (batchIndexPrepared) {
            std::vector<FastllmNaiveIndexerRow> rows;
            rows.reserve(batch->Size());
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                auto &part = *buf.sequences[sequence];
                if (part.capacity > indexTopK)
                    rows.push_back({indexCache(sequence),
                        &part.decode, packedIndices.empty() ? &part.indices : &packedIndices[sequence],
                        sequence, part.capacity, indexTable(sequence)});
            }
            const int ranks = tensorParallel ? AttentionTPSize() : 1;
            const bool contextParallel = !attentionParallel && !indexerLayerSharding && !contextSharded && !verifying && decodeWorkspace && ranks > 1 &&
                rows.size() >= 4 && buf.capacity >= std::max(65536, ranks * 4096) &&
                FastllmCudaNaiveIndexerContextSupported(tpDevices) &&
                buf.decode.indexScores.Count(0) >= (size_t)ranks * rows.size() * indexTopK * 2;
            Data &exchange = buf.decode.indexScores;
            if (contextSharded) {
                RunContextDecodeIndexer(batchIndexQ, batchIndexWeights, liveKeys, rows,
                    buf.indexContextExchange, attentionRank, ranks, GetPageLen(), gpu, indexFp8, indexerGather);
                batchIndexerPrepared = true;
            } else batchIndexerPrepared = FastllmCudaNaiveBatchDecodeIndexer(batchIndexQ,
                batchIndexWeights, liveKeys, rows, indexFp8,
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
                FastllmCudaNaiveMergeDecodeIndexer(liveKeys, rows, exchange, ranks);
            }
        }
        if (broadcastBatchIndex) {
            if (localIndexer && !batchIndexerPrepared) {
                for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                    auto &part = *buf.sequences[sequence];
                    if (part.capacity <= indexTopK) continue;
                    part.liveKeys.Resize({1});
                    part.liveKeys.FakeFrom(liveKeys, sequence * sizeof(int));
                    Data indexQ, indexWeights;
                    indexQ.Resize({1, 1, indexHeads * indexDim});
                    indexQ.FakeFrom(batchIndexQ, (size_t)sequence * indexHeads * indexDim * batchIndexQ.unitSize);
                    indexWeights.Resize({1, 1, indexHeads});
                    indexWeights.FakeFrom(batchIndexWeights, (size_t)sequence * indexHeads * batchIndexWeights.unitSize);
                    FastllmCudaNaiveDecodeIndexer(indexQ, indexWeights,
                        *indexCache(sequence), part.liveKeys,
                        part.capacity, indexFp8, part.decode, packedIndices[sequence], indexTable(sequence));
                }
            }
            if (indexerLayerSharding)
                FastllmNcclBroadcast(buf.indexBroadcast.cudaData, buf.indexBroadcast.Count(0), INT32,
                                    tpDevices[IndexerOwnerRank(layer)], gpu);
            batchIndexerPrepared = true;
        }
        // Prepare each request's independent KV and Indexer Q first, then score
        // all requests together. Attention remains in its original request order.
        const int prefillRanks = tensorParallel ? (int)tpDevices.size() : 1;
        bool batchPrefillIndex = !attentionParallel && batch && batch->Size() > 1 && batch->Size() <= 8 &&
            !decodeWorkspace && !verifying && separateIndex && tpEp8MegaState &&
            ((indexerLayerSharding && tpIndexerPeerAccess) || contextSharded) && indexFp8 && indexHeads == 16 &&
            indexDim == 128 && indexTopK == 2048 &&
            (prefillRanks == 2 || prefillRanks == 4 || prefillRanks == 8) &&
            FastllmCudaRuntimeArch() == 90;
        size_t prefillScoreBytes = 0;
        if (batchPrefillIndex) for (int sequence = 0; sequence < batch->Size(); ++sequence) {
            const auto &cache = *batch->Key(sequence, layer).multiDeviceDatas.at(gpu);
            const int count = batch->lengths[sequence];
            const int keys = (cache.dims.empty() ? 0 : cache.dims[1]) + count;
            if (count <= 0 || count % (prefillRanks * 128) || keys < 8192 ||
                (int64_t)(count / (2 * prefillRanks)) * keys < 1024 * 1024) {
                batchPrefillIndex = false;
                break;
            }
            prefillScoreBytes += (size_t)(count / prefillRanks) * keys * sizeof(float);
            if (prefillScoreBytes > (size_t(512) << 20)) { batchPrefillIndex = false; break; }
        }
        // DP owns independent histories on this GPU. Prepare all local full-attention
        // requests before bounded Indexer/Attention batches; SWA retains its shared
        // rolling workspace and existing specialized kernels.
        const bool dpPrefill = attentionParallel && AttentionTPSize() == 1 && batch &&
            !attentionBatchDecode && !decodeWorkspace && !verifying && !sparseLayer &&
            !slidingLayers[layer];
        const bool rankIndexer = dpPrefill && localIndexer && !indexerLayerSharding && indexTopK > 0;
        const bool idsInQkv = rankIndexer && reuseAttention &&
            int64_t(qkv.dims.back()) - int64_t(cfg.heads) * cfg.valueDim >= 2LL * indexTopK;
        if (rankIndexer) {
            // All three deferred outputs live across requests. Pack by local
            // token offset: the rank token budget bounds their total size even
            // when the longest request moves between batch slots.
            auto prepareRows = [&](Data &storage, DataType type, int width) {
                storage.dataType = type;
                storage.Resize({1, length, width});
                storage.ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
                storage.Allocate(false);
            };
            prepareRows(batchIndexQ, BFLOAT16, indexHeads * indexDim);
            prepareRows(batchIndexWeights, BFLOAT16, indexHeads);
            Data *owner = &buf.indexBroadcast;
            size_t offset = 0;
            if (idsInQkv) {
                owner = &qkv;
                offset = size_t(length) * cfg.heads * cfg.valueDim * sizeof(uint16_t);
            } else if (!reuseProjected && embed_dim >= 2LL * indexTopK) {
                // O projection starts after attention has consumed every ID.
                prepareRows(projected, BFLOAT16, embed_dim);
                owner = &projected;
            } else {
                prepareRows(*owner, INT32, indexTopK);
            }
            packedIndices.resize(batch->Size());
            for (int sequence = 0; sequence < batch->Size(); ++sequence) {
                auto &ids = packedIndices[sequence];
                ids.FakeFrom(*owner, offset);
                ids.dataType = INT32;
                ids.Resize({batch->lengths[sequence], indexTopK});
                offset += size_t(batch->lengths[sequence]) * indexTopK * sizeof(int);
            }
        }
        int packedPrefillRows = 0;
        if (projectionBackend && !decodeWorkspace && (slidingLayers[layer] || dpPrefill))
            preparedO = activationFor(localWeight(ap + "o_proj.weight"), length);
        std::vector<FastllmNaivePrefillAttentionItem> prefillAttention;
        std::vector<FastllmNaiveStagedSwa> stagedSwa;
        bool pairedPrefill = false;
        int attendPhase = 0;
        std::vector<FastllmNaivePrefillIndexerItem> prefillItems;
        std::vector<std::unique_ptr<Data>> prefillViews;
        // Retain non-owning metadata until both deferred launches have consumed it.
        auto retainPrefillView = [&](const Data &data) {
            auto view = std::make_unique<Data>(data.dataType);
            if (!data.dims.empty()) {
                view->Resize(data.dims);
                view->FakeFrom(data, 0);
                view->isPagedKVCache = data.isPagedKVCache;
                view->pageLen = data.pageLen;
                view->pagedKVCacheData = data.pagedKVCacheData;
                view->kvCacheFp8Indexer = data.kvCacheFp8Indexer;
            }
            auto *result = view.get();
            prefillViews.push_back(std::move(view));
            return result;
        };
        // Sequential prefill reuses one chronological history per rank. A deferred
        // batch retains its bounded per-request histories until the scoring launch.
        Data &contextGather = buf.indexContextGather;
        auto attend = [&](Data &normed, Data &q, Data &k, Data &v, Data &qkv,
                          Data &positions, Data &cacheKey, Data &cacheValue,
                          TargetWorkspace &buf, int length, int globalPast, int reserveCapacity, Data &attn, int sequence = 0, int firstRow = 0) {
            Data &packed = buf.packed, &indexKey = buf.indexKey;
            Data &indices = packedIndices.empty() ? buf.indices : packedIndices[sequence];
            const bool sharedSwa = tensorParallel && slidingLayers[layer] &&
                !decodeWorkspace && !verifying && !tpSwaPrefillWorkspaces.empty();
            Data &pastKey = sharedSwa ? tpSwaPrefillWorkspaces[tpRank].first : cacheKey;
            Data &pastValue = sharedSwa ? tpSwaPrefillWorkspaces[tpRank].second : cacheValue;
            pastKey.kvCacheFp8Indexer = !separateIndex && fp8IndexerCache && localIndexer;
            int localPast = cacheKey.dims.empty() ? 0 : cacheKey.dims[1];
            if (attendPhase == 2 && !decodeWorkspace) localPast -= length;
            const bool ringCache = tensorParallel && slidingLayers[layer] && tpSwaRingCache;
            const bool ringDecode = ringCache && decodeWorkspace && !graphVerify;
            const bool ringVerify = ringCache && capture && capture->ringVerify;
            bool coldSwa = false;
            Data noIndices;
            Data *selected = attendPhase == 2 &&
                (decodeWorkspace ? buf.capacity : localPast + length) > indexTopK ? &indices : &noIndices;
            int balancedIndexRanks = attendPhase == 2 && batchPrefillIndex ? prefillRanks : 0;
            if (attendPhase != 2) {
                if (sparseLayer && !batchedDSA)
                    sparseKv->ranks[tpRank]->layers[layer]->InvalidateTail(cacheKey.cudaData, localPast, cudaStreamPerThread);
                if (sharedSwa) {
                    const int history = eagerSwa ? 0 : localPast;
                    pastKey.Resize({1, history, cfg.kvHeads * cfg.headDim});
                    pastValue.Resize({1, history, cfg.kvHeads * cfg.valueDim});
                    CopySwaWindow(cacheKey, cacheValue, pastKey, pastValue,
                        ringCache ? (globalPast - history) % window : 0, 0, history, ringCache ? window : 0);
                }
                // Graph replay reads/writes the ring directly. Eager prefill/fallback
                // uses the existing contiguous kernels and restores the ring once.
                // Verification stages candidates after the ring or was linearized outside capture.
                if (ringCache && !decodeWorkspace && !verifying && !sharedSwa)
                    FastllmCudaNaiveSwaCacheLayout(pastKey, pastValue, globalPast, false);
                // Sliding layers retain only window-1 rows between chunks. Keep their
                // reservation bounded even when the full request is very long.
                const int layerCapacity = slidingLayers[layer]
                    ? (int)std::min<int64_t>(reserveCapacity, (int64_t)window - 1 + length)
                    : reserveCapacity;
                Data &scoringKey = separateIndex && localIndexer ? *separateIndexCaches[sequence] : pastKey;
                auto indexLength = [&](int tokens) {
                    if (!contextSharded) return tokens;
                    const int page = GetPageLen(), local = page / AttentionTPSize();
                    return tokens / page * local + std::clamp(tokens % page - attentionRank * local, 0, local);
                };
                if (separateIndex && localIndexer && batchRopePrepared)
                    scoringKey.Resize({1, indexLength(localPast + length), scoringKey.dims[2]});
                // Scoring masks reads by liveKeys; its graph capacity can span
                // future pages of the fixed table. Appends validate page bindings.
                if (separateIndex && localIndexer && graphVerify)
                    scoringKey.expansionDims[1] = std::max(scoringKey.expansionDims[1], indexLength(buf.capacity));
                if (!batchRopePrepared) prepareIndexKey(normed, buf);
                if (separateIndex && localIndexer && !batchRopePrepared) {
                    FastllmCudaNaiveRope(indexKey, positions, 1, indexDim, rotaryDim, cfg.theta);
                    Data *input = &indexKey;
                    if (fp8IndexerCache) {
                        FastllmCudaNaivePackIndexer(indexKey, buf.packedIndexer);
                        input = &buf.packedIndexer;
                    }
                    if (contextSharded)
                        FastllmCudaNaiveContextIndexerWrite(scoringKey, *input, localPast,
                            decodeWorkspace ? &buf.liveKeys : nullptr, attentionRank, AttentionTPSize());
                    else if (decodeWorkspace)
                        FastllmCudaNaivePagedWrite(scoringKey, *input, 0, &buf.liveKeys);
                    else AppendCache(scoringKey, *input, AuxiliaryKVType(), 0);
                }
                Data noIndexKey, noLiveKeys;
                    bool fusedCache = swaRopePrepared || batchRopePrepared || ((decodeWorkspace || ringVerify || (!tensorParallel && length <= 8 && localPast > 0)) &&
                    FastllmCudaNaiveRopeAppendCache(q, k, v, localIndexer && !separateIndex ? indexKey : noIndexKey,
                        positions, pastKey, pastValue, decodeWorkspace || ringVerify ? buf.liveKeys : noLiveKeys, cfg.heads, cfg.kvHeads,
                        cfg.headDim, cfg.valueDim, rotaryDim, cfg.theta, valueScale,
                        slidingLayers[layer] ? window : 0, mergedQkv ? &qkv : nullptr, ringDecode, ringVerify));
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
                    Data *cacheInput = &k;
                    if (localIndexer && !separateIndex) {
                        FastllmCudaNaiveRope(indexKey, positions, 1, indexDim, rotaryDim, cfg.theta);
                        Cat(k, indexKey, 2, packed);
                        cacheInput = &packed;
                    }
                    if (pastKey.kvCacheFp8Indexer) {
                        FastllmCudaNaivePackIndexer(*cacheInput, buf.packedIndexer);
                        cacheInput = &buf.packedIndexer;
                    }
                    if (decodeWorkspace || ringVerify) {
                        if (graphVerify || ringVerify) FastllmCudaNaiveAppendVerifyCache(pastKey, pastValue, *cacheInput, v,
                            buf.liveKeys, slidingLayers[layer] ? window : 0, ringVerify);
                        else FastllmCudaNaiveAppendDecodeCache(pastKey, pastValue, *cacheInput, v,
                            buf.liveKeys, slidingLayers[layer] ? window : 0, ringDecode);
                        if (ringVerify && !decodeWorkspace) {
                            pastKey.Resize({1, localPast + length, pastKey.dims[2]});
                            pastValue.Resize({1, localPast + length, pastValue.dims[2]});
                        }
                    } else {
                        const auto type = LayerKVType(layer);
                        // On a cold SWA chunk every visible KV row belongs to this projection.
                        // Quantize once at the cache write and feed the same values to attention.
                        coldSwa = slidingLayers[layer] && !verifying && localPast == 0 && length >= 1024 &&
                            window == 128 && cfg.headDim == 192 && cfg.valueDim == 128 &&
                            cfg.heads == cfg.kvHeads * 8 && type == FP8_E4M3 && cacheInput == &k &&
                            !pastKey.kvCacheFp8Indexer && !FastllmCudaGraphIsCapturingFast() &&
                            (FastllmCudaRuntimeArch() == 100 || FastllmCudaRuntimeArch() == 103);
                        AppendCache(pastKey, *cacheInput, type, layerCapacity, true, coldSwa);
                        AppendCache(pastValue, v, type, layerCapacity, true, coldSwa);
                    }
                }
                if (!slidingLayers[layer] && (decodeWorkspace ? buf.capacity : pastKey.dims[1]) > indexTopK) {
                    const int ranks = tensorParallel ? AttentionTPSize() : 1;
                    const bool parallelIndex = !attentionParallel && !decodeWorkspace && !verifying && tpEp8MegaState && ranks > 1 &&
                        (!indexerLayerSharding || tpIndexerPeerAccess) && length % (ranks * 64) == 0 &&
                        (int64_t)(length / (2 * ranks)) * pastKey.dims[1] >= 1024 * 1024;
                    // Prefix restoration often leaves a non-64-aligned query tail.
                    // Partition those queries too, retaining the original score dispatch.
                    const bool raggedParallelIndex = !attentionParallel && !decodeWorkspace && !verifying && tensorParallel &&
                        ranks > 1 && !indexerLayerSharding && indexFp8 && indexHeads == 16 && indexDim == 128 &&
                        length % (ranks * 64) != 0 && length >= ranks * 128 &&
                        (int64_t)RaggedIndexerRows(length, ranks) * indexTopK <= INT_MAX &&
                        (int64_t)(length / ranks) * pastKey.dims[1] >= 1024 * 1024;
                    if ((localIndexer || parallelIndex) && !batchIndexerPrepared) {
                        Data rowIndexQ, rowIndexWeights;
                        const bool packedQueries = rankIndexer || batchIndexPrepared || batchVerifyIndex;
                        Data &indexQ = packedQueries ? rowIndexQ : buf.indexQ;
                        Data &indexWeights = packedQueries ? rowIndexWeights : buf.indexWeights;
                        int queryOffset = rankIndexer ? firstRow : sequence;
                        if (batchVerifyIndex) {
                            queryOffset = 0;
                            for (int i = 0; i < sequence; ++i) queryOffset += batch->lengths[i];
                        }
                        if (packedQueries) {
                            indexQ.Resize({1, length, indexHeads * indexDim});
                            indexQ.FakeFrom(batchIndexQ, (size_t)queryOffset * indexHeads * indexDim * batchIndexQ.unitSize);
                            indexWeights.Resize({1, length, indexHeads});
                            indexWeights.FakeFrom(batchIndexWeights, (size_t)queryOffset * indexHeads * batchIndexWeights.unitSize);
                        }
                        if (rankIndexer || (!packedQueries && localIndexer))
                            prepareIndexQuery(normed, positions, indexQ, indexWeights);
                        if (batchVerifyIndex) {
                            verifyIndexItems.push_back({&scoringKey, &indices, queryOffset, length, localPast,
                                graphVerify ? buf.capacity : 0, indexTable(sequence)});
                            return;
                        }
                        if (contextSharded && verifying) {
                            RunContextVerifyIndexer(indexQ, indexWeights,
                                {{&scoringKey, &indices, 0, length, localPast,
                                  graphVerify ? buf.capacity : 0, indexTable(sequence)}},
                                buf.decode, buf.liveKeys, buf.indexContextExchange,
                                attentionRank, ranks, GetPageLen(), gpu, indexFp8, indexerGather);
                        } else {
                        const Data *scoreKey = &scoringKey;
                        if (contextSharded && !decodeWorkspace) {
                            // Prefill consumes a chronological view. Only this request's
                            // active history is gathered; the persistent pool stays sharded.
                            const int keys = localPast + length;
                            FastllmCudaNaiveContextIndexerGather(scoringKey, keys, attentionRank, ranks, contextGather);
                            const int words = contextGather.GetBytes() / (ranks * sizeof(int));
                            if (indexerGather) indexerGather((int *)contextGather.cudaData + attentionRank * words,
                                contextGather.cudaData, words);
                            else AssertInFastLLM(FastllmNcclAllGather((int *)contextGather.cudaData + tpRank * words,
                                contextGather.cudaData, words, INT32, gpu), "Indexer prefill gather failed.");
                            // A deferred prefill batch retains at most eight independent
                            // histories; the sequential path reuses one rank-wide buffer.
                            FastllmCudaNaiveContextIndexerUnpack(contextGather, keys,
                                GetPageLen(), ranks, buf.indexContextFull);
                            scoreKey = &buf.indexContextFull;
                        }
                        Data peerIndex;
                        if (parallelIndex && indexerLayerSharding) {
                            // Keep the only Indexer KV copy on its owning GPU. Peers
                            // read its pages and score disjoint query tiles; only Q,
                            // head weights and final indices cross collectives.
                            if (!localIndexer) {
                                for (auto item : {std::make_pair(&indexQ, indexHeads * indexDim),
                                                  std::make_pair(&indexWeights, indexHeads)}) {
                                    auto &data = *item.first;
                                    data.dataType = BFLOAT16;
                                    data.Resize({1, length, item.second});
                                    data.ToDevice(normed.dataDevice, normed.dataDeviceIds, false);
                                    data.Allocate(false);
                                }
                            }
                            const int owner = tpDevices[IndexerOwnerRank(layer)];
                            FastllmNcclBroadcast(indexQ.cudaData, indexQ.Count(0), BFLOAT16, owner, gpu);
                            FastllmNcclBroadcast(indexWeights.cudaData, indexWeights.Count(0), BFLOAT16, owner, gpu);
                            // Broadcast also orders the owner's preceding KV append
                            // before peer reads. The cache stays live through gather.
                            const Data &root = batch ? batch->Key(sequence, layer) : pastKeyValues[layer].first;
                            scoreKey = root.multiDeviceDatas.at(owner);
                            if (separateIndex) {
                                peerIndex = SeparateIndexer(layer, IndexerOwnerRank(layer), *scoreKey);
                                // The owner appended on its stream before broadcast;
                                // use this query's length rather than stale root metadata.
                                peerIndex.dims[1] = localPast + length;
                                scoreKey = &peerIndex;
                            }
                        }
                        if (decodeWorkspace) {
                            if (contextSharded) {
                                std::vector<FastllmNaiveIndexerRow> rows{{&scoringKey, &buf.decode, &indices,
                                    0, buf.capacity, indexTable(sequence)}};
                                RunContextDecodeIndexer(indexQ, indexWeights, buf.liveKeys, rows,
                                    buf.indexContextExchange, attentionRank, ranks, GetPageLen(), gpu, indexFp8, indexerGather);
                            } else if (graphVerify) FastllmCudaNaiveGraphVerifyIndexer(indexQ, indexWeights, scoringKey,
                                buf.liveKeys, buf.capacity, indexFp8, buf.decode, indices);
                            else FastllmCudaNaiveDecodeIndexer(indexQ, indexWeights, scoringKey,
                                buf.liveKeys, buf.capacity, indexFp8, buf.decode, indices,
                                indexTable(sequence));
                        } else {
                            if (capture && capture->verifying && length <= 8)
                                FastllmCudaNaiveVerifyIndexer(indexQ, indexWeights, *scoreKey, indexHeads,
                                    indexDim, localPast, indexTopK, indexFp8, indices);
                            else if (parallelIndex) {
                                // Score disjoint query tiles and gather only indices.
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
                                if (attendPhase == 1) {
                                    // SeparateIndexer returns backend-approved non-owning views.
                                    // Keep their metadata alive through the batched launch/gathers.
                                    auto keyView = std::make_unique<Data>(scoreKey->dataType, scoreKey->dims);
                                    keyView->FakeFrom(*scoreKey, 0);
                                    keyView->isPagedKVCache = scoreKey->isPagedKVCache;
                                    keyView->pageLen = scoreKey->pageLen;
                                    keyView->pagedKVCacheData = scoreKey->pagedKVCacheData;
                                    auto outputView = std::make_unique<Data>(INT32, std::vector<int>{rows, indexTopK});
                                    outputView->FakeFrom(indices, (size_t)tpRank * rows * indexTopK * sizeof(int));
                                    prefillItems.push_back({&indexQ, &indexWeights, keyView.get(), outputView.get(), localPast});
                                    prefillViews.push_back(std::move(keyView));
                                    prefillViews.push_back(std::move(outputView));
                                    return;
                                }
                                Data rankIndices;
                                rankIndices.Resize({rows, indexTopK});
                                rankIndices.FakeFrom(indices, (size_t)tpRank * rows * indexTopK * sizeof(int));
                                const bool mergedParts = balanced && indexTopK == 2048 &&
                                    FastllmCudaNaiveBalancedIndexer(indexQ, indexWeights, *scoreKey,
                                        localPast, tpRank, ranks, indexFp8, indexerLayerSharding || contextSharded, rankIndices, prefillScores, indexerCompute);
                                for (int part = 0; !mergedParts && part < parts; ++part) {
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
                                    FastllmCudaNaiveIndexer(localQ, localWeights, *scoreKey, indexHeads, indexDim,
                                        localPast + first, indexTopK, indexFp8, localIndices, indexerLayerSharding || contextSharded, prefillScores, indexerCompute);
                                }
                                FastllmCudaMimoEp8PrefillAllGather(*tpEp8MegaState, tpRank, indices);
                                if (balanced) balancedIndexRanks = ranks;
                            }
                            else if (raggedParallelIndex) {
                                Data temporary;
                                Data &gathered = length % ranks ? temporary : indices;
                                ScoreRaggedIndexer(indexQ, indexWeights, *scoreKey, localPast,
                                    indexTopK, indexFp8, tpRank, ranks, gathered, prefillScores, indexerCompute);
                                const int rows = RaggedIndexerRows(length, ranks);
                                AssertInFastLLM(FastllmNcclAllGather(
                                    (int *)gathered.cudaData + (size_t)tpRank * rows * indexTopK,
                                    gathered.cudaData, rows * indexTopK, INT32, gpu),
                                    "Ragged Indexer AllGather failed.");
                                if (length % ranks) CompactRaggedIndexer(gathered, length, ranks, indices);
                            }
                            else if (dpPrefill && attendPhase == 1) {
                                prefillItems.push_back({retainPrefillView(indexQ), retainPrefillView(indexWeights),
                                    retainPrefillView(*scoreKey), &indices, localPast});
                                return;
                            }
                            else FastllmCudaNaiveIndexer(indexQ, indexWeights, *scoreKey, indexHeads, indexDim,
                                                        localPast, indexTopK, indexFp8, indices, false, prefillScores, indexerCompute);
                        }
                    }
                    }
                    if (indexerLayerSharding && !broadcastBatchIndex && !batchVerifyIndex && !parallelIndex) {
                        if (!localIndexer) {
                            indices.dataType = INT32;
                            indices.Resize({length, indexTopK});
                            indices.ToDevice(q.dataDevice, q.dataDeviceIds, false);
                            indices.Allocate(false);
                        }
                        // Integer broadcast preserves the owning rank's exact TopK.
                        // Every rank calls this in the same layer/request order,
                        // including inactive padded rows in a captured graph.
                        FastllmNcclBroadcast(indices.cudaData, indices.Count(0), INT32,
                                            tpDevices[IndexerOwnerRank(layer)], gpu);
                        // The broadcast meets all ranks at host submission, and
                        // attention consumes its indices on the same stream.
                        // No per-request GPU completion barrier is needed here.
                    }
                    selected = &indices;
                }
            }
            if ((batchVerifyIndex || dpPrefill) && attendPhase == 1) return;
            Data stagedKey, stagedValue;
            if (sparseLayer && !batchedDSA) StageSparseKv(tpRank, layer, pastKey, pastValue, stagedKey, stagedValue);
            Data &attentionKey = coldSwa ? k : (sparseLayer && !batchedDSA ? stagedKey : pastKey);
            Data &attentionValue = coldSwa ? v : (sparseLayer && !batchedDSA ? stagedValue : pastValue);
            Data &sink = localWeight(ap + "attention_sink_bias");
            if (!sink.dims.empty()) {
                ToDataType(sink, DataType::FLOAT32);
                sink.ToDevice(q.dataDevice, q.dataDeviceIds);
            }
            if (dpPrefill) {
                prefillAttention.push_back({retainPrefillView(q), &attentionKey, &attentionValue,
                    retainPrefillView(*selected), retainPrefillView(attn), localPast,
                    FastllmMxfp8PrefillOutput::ForRows(preparedO, firstRow, length, packedPrefillRows)});
                return;
            }
            if (batchedDSA) {
                FastllmNaiveDSARow row;
                row.fp8 = pastKey.dataType == FP8_E4M3;
                row.fp4 = pastKey.dataType == FP4_E2M1;
                row.tensorMaps = layerMaps ? layerMaps + (size_t)sequence * FastllmNaiveDSAMapBytes : nullptr;
                row.cache = layerCache ? layerCache + sequence * cacheStride : nullptr;
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
                // Verification queries share KV storage, but retain independent
                // causal lengths and sparse selections, including a dense/sparse
                // boundary inside one block. Reuse the normal batched DSA kernels.
                for (int query = 0; query < length; ++query) {
                    auto item = row;
                    if (batchVerifyDSA) {
                        item.length = localPast + query + 1;
                        item.capacity = graphVerify ? buf.capacity : item.length;
                        item.liveKeys = graphVerify ? (const int *)buf.liveKeys.cudaData + query : nullptr;
                        item.indices = row.indices && (graphVerify ? buf.capacity : item.length) > row.selectedCount
                            ? row.indices + (size_t)query * row.selectedCount : nullptr;
                        item.selectedCount = item.indices ? row.selectedCount : 0;
                    }
                    dsaRows.push_back(item);
                }
            } else if (batchedSWA) {
                for (int query = 0; query < length; ++query) {
                    FastllmNaiveSWARow row{pastKey.cudaData, pastValue.cudaData,
                        layerCache ? layerCache + sequence * cacheStride : nullptr, pastKey.dataType == FP8_E4M3};
                    if (eagerSwa) {
                        // The shared workspace holds only the new KV. Bind attention to
                        // the request's ring before another row reuses that workspace.
                        CopySwaWindow(pastKey, pastValue, cacheKey, cacheValue,
                            0, globalPast % window, 1, 0, window);
                        row.key = cacheKey.cudaData;
                        row.value = cacheValue.cudaData;
                        row.length = globalPast + 1;
                        const int retained = std::min(row.length, window - 1);
                        cacheKey.Resize({1, retained, cfg.kvHeads * cfg.headDim});
                        cacheValue.Resize({1, retained, cfg.kvHeads * cfg.valueDim});
                    } else if (verifySWA) {
                        row.ring = ringVerify;
                        row.length = (ringVerify ? globalPast : localPast) + query + 1;
                        row.verifyOffset = ringVerify ? query : -1;
                    }
                    swaRows.push_back(row);
                }
            } else if (decodeWorkspace) {
                if (graphVerify) FastllmCudaNaiveGraphVerifyAttention(q, attentionKey, attentionValue, *selected, sink,
                    buf.liveKeys, buf.capacity, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                    slidingLayers[layer] ? window : 0, buf.decode, attn, ringVerify);
                else FastllmCudaNaiveDecodeAttention(q, attentionKey, attentionValue, *selected, sink,
                    buf.liveKeys, buf.capacity, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                    slidingLayers[layer] ? window : 0, buf.decode, attn, ringDecode);
                if (slidingLayers[layer] && !graphVerify && !ringDecode)
                    FastllmCudaNaiveTrimDecodeCache(pastKey, pastValue, buf.liveKeys, window);
            } else {
                if (verifying && length <= 8)
                    FastllmCudaNaiveVerifyAttention(q, attentionKey, attentionValue, *selected, sink,
                        cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                        localPast, slidingLayers[layer] ? window : 0, attn);
                else if (!balancedIndexRanks || !FastllmCudaNaiveBalancedAttention(
                        q, attentionKey, attentionValue, *selected, sink, cfg.heads, cfg.kvHeads,
                        localPast, balancedIndexRanks, attn)) {
                    if (balancedIndexRanks) FastllmCudaNaiveRestoreIndexerRows(*selected, balancedIndexRanks);
                    if (slidingLayers[layer] || verifying ||
                        !FastllmCudaTryNaiveIndexerAttention({&q, &attentionKey, &attentionValue,
                            selected, &attn, localPast}, sink, cfg.heads, cfg.kvHeads,
                            cfg.headDim, cfg.valueDim, prefillScores ? *prefillScores : normed)) {
                        // SWA has no Indexer projections. Its input norm is dead
                        // until the post-attention RMSNorm overwrites this slice.
                        FastllmCudaNaivePrefillAttention(q, attentionKey, attentionValue, *selected, sink,
                            cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                            localPast, slidingLayers[layer] ? window : 0, attn, true,
                            slidingLayers[layer] ? &normed : nullptr,
                            batch && batch->Size() > 1 && !verifying && q.isFake && attn.isFake && normed.isFake
                                ? &stagedSwa : nullptr,
                            FastllmMxfp8PrefillOutput::ForRows(preparedO, firstRow, length, packedPrefillRows));
                    }
                }
                if (slidingLayers[layer] && capture && !capture->prefixes.empty()) {
                    for (const auto &snapshot : capture->prefixes[attentionParallel ?
                        attentionParallel->groups[dpGroup].sequenceIndices[sequence] : sequence]) {
                        int rows = std::min(snapshot->length, window - 1);
                        CopyPrefixWindow(*snapshot, tpRank, layer, pastKey, pastValue,
                            localPast + snapshot->length - globalPast - rows, false);
                    }
                }
                if (sharedSwa) {
                    const int rows = std::min(localPast + length, window - 1);
                    CopySwaWindow(pastKey, pastValue, cacheKey, cacheValue,
                        localPast + length - rows,
                        ringCache ? (globalPast + length - rows) % window : 0, rows, 0, ringCache ? window : 0);
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
            for (int phase = 0; phase < (batchPrefillIndex || batchVerifyIndex || dpPrefill ? 2 : 1); ++phase) {
                attendPhase = batchPrefillIndex || batchVerifyIndex || dpPrefill ? phase + 1 : 0;
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
                    // Sequential prefill attention needs one scratch/history
                    // buffer. The DP Indexer retains only the packed rank views;
                    // its per-request KV lives in the request's own cache.
                    const bool sharedPrefill = !attentionBatchDecode && !decodeWorkspace &&
                        !verifying && !batchPrefillIndex && !dpPrefill;
                    auto &part = sharedPrefill || rankIndexer ? buf : *buf.sequences[sequence];
                    if (decodeWorkspace || (capture && capture->ringVerify)) {
                        part.liveKeys.Resize({count});
                        part.liveKeys.FakeFrom(liveKeys, offset * sizeof(int));
                    }
                    attend(rowNorm, rowQ, rowK, rowV, rowQkv, rowPositions,
                           *batch->Key(sequence, layer).multiDeviceDatas.at(gpu),
                           *batch->Value(sequence, layer).multiDeviceDatas.at(gpu), part, count,
                           batch->Key(sequence, 0).dims.empty() ? 0 : batch->Key(sequence, 0).dims[1],
                           CacheReserveCapacity(batch->Config(sequence)), rowAttention,
                           sequence, offset);
                    offset += count;
                }
                if (!stagedSwa.empty()) {
                    FastllmCudaNaiveFlushSwa(stagedSwa, localWeight(ap + "attention_sink_bias"), cfg.heads, cfg.kvHeads);
                    stagedSwa.clear();
                }
                if (batchVerifyIndex && phase == 0) {
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
                        // Build all request descriptors first. A rejected fused call below
                        // executes the original indexer before the original attention.
                        deferredDsaIndexer = !attentionParallel && localIndexer && !verifyIndexItems.empty() &&
                            !contextSharded && !separateIndex && !indexerLayerSharding &&
                            !sparseLayer && !dsaFrontPrepared && indexFp8 &&
                            std::all_of(verifyIndexItems.begin(), verifyIndexItems.end(), [](const auto& item) {
                                return item.rows >= 4 && item.rows <= 8 &&
                                    (item.capacity > 0 ? item.capacity : item.past + item.rows) > 8192;
                            }) &&
                            cfg.heads == 8 && cfg.kvHeads == 1 && cfg.headDim == 192 && cfg.valueDim == 128 &&
                            FastllmCudaRuntimeArch() == 103;
#endif
                        if (deferredDsaIndexer) {
                            // No GPU writes here; admission occurs after phase 1.
                        } else if (contextSharded && !verifyIndexItems.empty()) {
                        RunContextVerifyIndexer(batchIndexQ, batchIndexWeights, verifyIndexItems,
                            buf.decode, liveKeys, buf.indexContextExchange,
                            attentionRank, AttentionTPSize(), GetPageLen(), gpu, indexFp8, indexerGather);
                        } else if (localIndexer && !verifyIndexItems.empty()) {
                            runVerifyIndexer();
                    }
                    if (indexerLayerSharding)
                        FastllmNcclBroadcast(buf.indexBroadcast.cudaData, buf.indexBroadcast.Count(0), INT32,
                            tpDevices[IndexerOwnerRank(layer)], gpu);
                }
                if (dpPrefill && phase == 0) {
                    // Rank-local indices stay private through attention, including when
                    // draft hidden capture places them outside the merged QKV buffer.
                    pairedPrefill = rankIndexer && FastllmCudaHasBlackwellIndexerAttention() &&
                        prefillItems.size() == packedIndices.size();
                    for (size_t i = 0; pairedPrefill && i < prefillItems.size(); ++i)
                        pairedPrefill = prefillItems[i].indices == &packedIndices[i];
                    if (!pairedPrefill)
                        FastllmCudaNaiveBatchIndexer(prefillItems, indexHeads, indexDim, indexTopK,
                            indexFp8, prefillScores, indexerCompute);
                }
                if (batchPrefillIndex && phase == 0) {
                    AssertInFastLLM((int)prefillItems.size() == batch->Size(), "Incomplete prefill Indexer batch.");
                    if (!FastllmCudaNaiveBatchBalancedIndexer(prefillItems, tpRank, prefillRanks, indexFp8, true, prefillScores, indexerCompute)) {
                        for (const auto &item : prefillItems) {
                            if (FastllmCudaNaiveBalancedIndexer(*item.query, *item.weights, *item.keys,
                                    item.past, tpRank, prefillRanks, indexFp8, true, *item.indices, prefillScores, indexerCompute)) continue;
                            const int part = item.query->dims[1] / (2 * prefillRanks);
                            for (int range = 0; range < 2; ++range) {
                                const int first = range ? item.query->dims[1] - (tpRank + 1) * part : tpRank * part;
                                Data localQ(BFLOAT16, {1, part, indexHeads * indexDim});
                                Data localWeights(BFLOAT16, {1, part, indexHeads}), localIndices(INT32, {part, indexTopK});
                                localQ.FakeFrom(*item.query, (size_t)first * indexHeads * indexDim * item.query->unitSize);
                                localWeights.FakeFrom(*item.weights, (size_t)first * indexHeads * item.weights->unitSize);
                                localIndices.FakeFrom(*item.indices, (size_t)range * part * indexTopK * sizeof(int));
                                FastllmCudaNaiveIndexer(localQ, localWeights, *item.keys, indexHeads, indexDim,
                                    item.past + first, indexTopK, indexFp8, localIndices, true, prefillScores, indexerCompute);
                            }
                        }
                    }
                    for (int sequence = 0; sequence < batch->Size(); ++sequence)
                        FastllmCudaMimoEp8PrefillAllGather(*tpEp8MegaState, tpRank, buf.sequences[sequence]->indices);
                }
            }
        } else {
            auto &pastKey = tensorParallel ? *pastKeyValues[layer].first.multiDeviceDatas.at(gpu) : pastKeyValues[layer].first;
            auto &pastValue = tensorParallel ? *pastKeyValues[layer].second.multiDeviceDatas.at(gpu) : pastKeyValues[layer].second;
            attend(normed, q, k, v, qkv, positions, pastKey, pastValue, buf, length, pastLength, reserveCapacity, attn);
        }
        // A paired Indexer hands its scores directly to attention on this stream.
        // The other paths have already published their ordered TopK rows.
        if (pairedPrefill) {
            for (size_t i = 0; i < prefillItems.size(); ++i)
                FastllmCudaNaivePrefillIndexedAttention(prefillItems[i], prefillAttention.at(i),
                    localWeight(ap + "attention_sink_bias"), indexHeads, indexDim, indexTopK,
                    indexFp8, indexerCompute, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim,
                    prefillScores, prefillScores ? *prefillScores : normed);
        } else if (dpPrefill)
            FastllmCudaNaiveBatchIndexerAttention(prefillAttention, localWeight(ap + "attention_sink_bias"),
                cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim, prefillScores ? *prefillScores : normed);
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
            if (deferredDsaIndexer) {
                fusedDsaPrepared = FastllmCudaNaiveFusedDSAVerify(q, batchIndexQ, batchIndexWeights,
                    localWeight(ap + "attention_sink_bias"), verifyIndexItems, dsaRows, indexFp8,
                    cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim, buf.decode, attn);
                if (!fusedDsaPrepared) {
                    runVerifyIndexer();
                    // A per-request fallback can replace an index allocation after
                    // the attention descriptors were built. Refresh those bindings.
                    for (const auto& item : verifyIndexItems)
                        for (int row = 0; row < item.rows; ++row)
                            if (dsaRows[item.offset + row].indices)
                                dsaRows[item.offset + row].indices = (const int*)item.indices->cudaData + (size_t)row * indexTopK;
                }
            }
#endif
        if (batchedDSA && sparseLayer) {
            auto &cache = *sparseKv->ranks[tpRank]->layers[layer];
            // Each group contains at most the configured number of queries per
            // owner. Small hot caches retain a batched fallback across requests.
            std::vector<std::vector<int>> groups(1);
            int offset = 0;
            for (int sequence = 0; sequence < (batch ? batch->Size() : 1); ++sequence) {
                const int count = batchVerifyDSA ? (batch ? batch->lengths[sequence] : length) : 1;
                for (int row = 0; row < count; ++row) {
                    const int group = row / sparseKv->queries;
                    if ((int)groups.size() <= group) groups.resize(group + 1);
                    groups[group].push_back(offset + row);
                }
                offset += count;
            }
            if (groups.size() > 1) {
                attn.dataType = BFLOAT16; attn.UpdateUnitSize();
                attn.Resize({1, length, cfg.heads * cfg.valueDim});
                attn.ToDevice(q.dataDevice, q.dataDeviceIds, false); attn.Allocate(false);
            }
            for (const auto &group : groups) {
                std::vector<KvMemSparseQuery> queries;
                std::vector<FastllmNaiveDSARow> rows;
                for (int index : group) {
                    const auto &d = dsaRows[index];
                    queries.push_back({d.keyPages, d.cache ? d.cache + 2 : nullptr,
                        d.liveKeys, d.indices, d.length, d.indices ? d.selectedCount : indexTopK});
                    rows.push_back(d);
                }
                cache.ResolveQueries(queries.data(), queries.size(), cudaStreamPerThread, true);
                for (size_t row = 0; row < rows.size(); ++row) {
                    auto &d = rows[row];
                    d.cache = cache.CachePointers() + row * 2;
                    d.indices = cache.Indices() + row * indexTopK;
                    d.liveKeys = cache.Lengths() + row;
                    d.capacity = kvMemConfig.residentTokens; d.selectedCount = indexTopK;
                    d.tensorMaps = nullptr; d.pageLen = 0; d.keyPages = d.valuePages = nullptr;
                }
                if (groups.size() == 1) {
                    FastllmCudaNaiveBatchDSADecode(q, localWeight(ap + "attention_sink_bias"), rows,
                        cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim, buf.decode, attn);
                } else {
                    auto &query = buf.sparseQuery;
                    query.dataType = q.dataType; query.UpdateUnitSize();
                    query.Resize({1, (int)group.size(), cfg.heads * cfg.headDim});
                    query.ToDevice(q.dataDevice, q.dataDeviceIds, false); query.Allocate(false);
                    std::vector<void *> dst;
                    std::vector<const void *> src;
                    std::vector<size_t> bytes;
                    const size_t queryBytes = size_t(cfg.heads) * cfg.headDim * q.unitSize;
                    for (size_t row = 0; row < group.size(); ++row) {
                        dst.push_back((char *)query.cudaData + row * queryBytes);
                        src.push_back((char *)q.cudaData + group[row] * queryBytes); bytes.push_back(queryBytes);
                    }
                    AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                        dst.data(), src.data(), bytes.data(), dst.size()), "Sparse query gather failed.");
                    FastllmCudaNaiveBatchDSADecode(query, localWeight(ap + "attention_sink_bias"), rows,
                        cfg.heads, cfg.kvHeads, cfg.headDim, cfg.valueDim, buf.decode, buf.sparseAttention);
                    const size_t outputBytes = size_t(cfg.heads) * cfg.valueDim * attn.unitSize;
                    for (size_t row = 0; row < group.size(); ++row) {
                        dst[row] = (char *)attn.cudaData + group[row] * outputBytes;
                        src[row] = (char *)buf.sparseAttention.cudaData + row * outputBytes; bytes[row] = outputBytes;
                    }
                    AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                        dst.data(), src.data(), bytes.data(), dst.size()), "Sparse attention scatter failed.");
                }
            }
        } else if (batchedDSA && !dsaFrontPrepared && !fusedDsaPrepared) {
            Data &sink = localWeight(ap + "attention_sink_bias");
            if (projectionBackend) preparedO = activationFor(localWeight(ap + "o_proj.weight"), dsaRows.size());
            if (!FastllmCudaNaiveOnlineDSADecode(q, sink, dsaRows, cfg.heads, cfg.kvHeads,
                    cfg.headDim, cfg.valueDim, buf.decode, attn, &preparedO)) {
                FastllmCudaNaiveBatchDSADecode(q, sink, dsaRows, cfg.heads, cfg.kvHeads,
                    cfg.headDim, cfg.valueDim, buf.decode, attn);
            }
        }
        if (batchedSWA && !swaAttentionPrepared) {
            Data eagerLengths;
            if (projectionBackend) preparedO = activationFor(localWeight(ap + "o_proj.weight"), swaRows.size());
            FastllmCudaNaiveBatchSWADecode(q, localWeight(ap + "attention_sink_bias"), swaRows,
                eagerSwa || (verifySWA && !graphVerify) ? eagerLengths : liveKeys, cfg.heads, cfg.kvHeads, attn, &preparedO);
        }
#ifndef USE_ROCM
            // DSA and SWA share the same output-projection contract. The collective
            // admits the complete rank group before launching; other shapes and
            // devices retain the ordinary projection/reduce/residual path below.
            if (!attentionParallel && tensorParallel && decodeWorkspace && length >= 1 && length <= 8) {
                const bool mmaProjection = verifying && slidingLayers[layer] && length > 1;
                if (moeLayers[layer]) {
                    attentionRouterReady = FastllmCudaSwaOutputAddRMSNormPrepared(attn,
                        localWeight(ap + "o_proj.weight"), projected, hidden, postAttentionNorm,
                        normed, rms_norm_eps, gpu, &localWeight(prefix + ".mlp.gate.weight"), &router, verifying && !mmaProjection, mmaProjection, prepared);
                    attentionInputPrepared = attentionRouterReady && prepared;
                    attentionNormReady = attentionRouterReady;
                }
                if (!attentionNormReady)
                    attentionNormReady = FastllmCudaSwaOutputAddRMSNormWithProjection(attn,
                        localWeight(ap + "o_proj.weight"), projected, hidden,
                        postAttentionNorm, normed, rms_norm_eps, gpu, nullptr, nullptr, verifying && !mmaProjection, mmaProjection);
                if (attentionNormReady) trackCommunication(projected);
            }
#endif
            if (!decodeWorkspace && packedPrefillRows != length) preparedO = {};
            if (!attentionNormReady) projectAttention(attn, localWeight(ap + "o_proj.weight"), Data(), projected, preparedO);
        if (attentionParallel && AttentionTPSize() > 1) {
            auto &runtime = attentionParallel->runtime;
            // Match the existing TP reduction: sum BF16 partials in FP32,
            // then round once. NCCL BF16 may round at intermediate tree nodes.
            Data &sum = attentionParallel->ranks[tpRank].reduction;
            ToDataType(projected, sum, FLOAT32);
            runtime.AllReduce(runtime.Layout().attention[dpGroup], tpRank, sum.cudaData,
                sum.Count(0), parallel::ScalarType::Float32);
            ToDataType(sum, projected, BFLOAT16);
        }
        } // Generic attention/projection.
        } // Local attention rows.
        if (attentionParallel && !ownerRows) {
            projected.dataType = BFLOAT16; projected.UpdateUnitSize();
            projected.Resize({1, length, embed_dim}); projected.ToDevice(CUDA, {gpu}, false); projected.Allocate(false);
            attentionParallel->transfer->Collect(tpRank, attentionParallel->ranks[tpRank].projected.cudaData,
                projected.cudaData, size_t(embed_dim) * sizeof(uint16_t));
        }

        if (!attentionLayerReady) {
            if (attentionNormReady) {
                // A fused output path already updated the residual and norm.
            } else if (shardPrefillNorm && length >= 1024 && length % epRanks == 0) {
            // Only this rank's token shard feeds the following EP Router/MoE.
            // Avoid distributing reduced rows that are never consumed here.
            FastllmCudaMimoEp8PrefillReduceScatter(*tpEp8MegaState, tpRank, projected);
        } else if (!attentionParallel && !attentionNormReady) {
            attentionNormReady = reduce(projected, decodeWorkspace ? &hidden : nullptr,
                &postAttentionNorm, &normed);
        }
        if (shardPrefillNorm) {
            const int first = (int)((int64_t)length * tpRank / epRanks);
            const int rows = (int)((int64_t)length * (tpRank + 1) / epRanks) - first;
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
            if (!attentionNormReady && useSharedFlex && verifying && moeLayers[layer] && num_experts_per_tok == 8 &&
                tpSharedFlex->PrepareResidualRouter(tpRank, layer, length, hidden, projected, postAttentionNorm,
                    rms_norm_eps, normed, localWeight(prefix + ".mlp.gate.weight"), router,
                    localWeight(prefix + ".mlp.gate.e_score_correction_bias"), expertIndex, expertScore,
                    norm_topk_prob, routed_scaling_factor)) {
                attentionNormReady = attentionRouterReady = attentionInputPrepared = true;
                preparedInput.routed = true;
                prepared = &preparedInput;
            }
            if (!attentionNormReady)
                FastllmCudaNaiveAddDecodeRMSNorm(hidden, projected,
                    postAttentionNorm, rms_norm_eps, normed);
        } else {
            AddTo(hidden, projected);
            norm(hidden, postAttentionNorm, normed);
        }
        }
        Data *mlpNormWeight = nullptr, *mlpNormOutput = nullptr;
        if ((decodeWorkspace && !(fusedHead && layer + 1 == block_cnt)) ||
            (fusePrefillNorm && layer + 1 < block_cnt)) {
            const bool lastLayer = layer + 1 == block_cnt;
            const std::string name = lastLayer ? "model.norm.weight" :
                "model.layers." + std::to_string(layer + 1) + ".input_layernorm.weight";
            mlpNormWeight = &localWeight(name);
            mlpNormOutput = lastLayer && !localMoe ? &buf.last : &normed;
        }
        auto addMlpResidual = [&](Data &branch, bool needsReduce) {
            if (fusedHead && layer + 1 == block_cnt) {
                if (needsReduce) reduce(branch);
                // Verification may export the final hidden state to the draft.
                // Materialize its BF16 residual before feature collection; the
                // fused head can still normalize/project that stable value.
                if (capture && capture->collectHidden) AddTo(hidden, branch);
                else headBranch = &branch;
                return;
            }
            if (layer + 1 < block_cnt && moeLayers[layer + 1] &&
                (((useSwaLayer || useSwaVerifyLayer) && slidingLayers[layer + 1]) ||
                 (useDsaVerifyLayer && !slidingLayers[layer + 1]))) {
                if (needsReduce) reduce(branch);
                pendingMlp = &branch;
                return;
            }
            if (!decodeWorkspace) {
                if (needsReduce) reduce(branch);
                if (!fusePrefillNorm || !mlpNormWeight) {
                    AddTo(hidden, branch);
                    return;
                }
            } else if (needsReduce && reduce(branch, &hidden, mlpNormWeight, mlpNormOutput)) {
                return;
            }
            if (projectionBackend && localMoe && layer + 1 < block_cnt && moeLayers[layer + 1] &&
                (decodeWorkspace || length >= FastllmNaivePackedNormMinPrefillRows)) {
                nextQkv = activationFor(localWeight("model.layers." + std::to_string(layer + 1) +
                                                    ".self_attn.mergeqkv.weight"), length);
            }
            FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, *mlpNormWeight,
                rms_norm_eps, *mlpNormOutput, &nextQkv);
        };
        auto reservePrefillArena = [&]() {
            const int dispatchRows = localMoe || localDense ? maxLocalRows * epRanks : length;
            const size_t rowBytes = size_t(embed_dim) * sizeof(float);
            size_t bytes = FastllmCudaMimoEp8PrefillWorkspaceBytes(dispatchRows, epRanks);
            if (localDense) {
                // Router math sees only the owner's rows. The symmetric EP plan still
                // covers the most populated owner on every rank.
                const size_t routerBytes = 3ull * num_experts *
                    (embed_dim * sizeof(uint16_t) + size_t(maxLocalRows) * sizeof(float));
                bytes = std::max({bytes, size_t(maxLocalRows) * rowBytes,
                    std::min(routerBytes, size_t(dispatchRows) * rowBytes),
                    size_t(inputIds.dims[1]) * embed_dim * sizeof(uint16_t)});
            } else bytes = std::max(bytes, size_t(dispatchRows) * rowBytes);
            routerInput.dataType = FLOAT32;
            routerInput.Resize({int((bytes + rowBytes - 1) / rowBytes), embed_dim});
            routerInput.ToDevice(CUDA, {gpu}, false);
            routerInput.Allocate(false);
            routerInput.Resize({localDense ? maxLocalRows : dispatchRows, embed_dim});
        };
        if (localDense && layer == 0) {
            // Gather post-attention activations once. Gate/up preserve the original
            // world-row GEMMs and the down projection keeps the same TP all-reduce.
            // Q, QKV and the router/EP workspace are all dead during this FFN.
            reservePrefillArena();
            Data world;
            world.FakeFrom(routerInput, 0);
            world.dataType = BFLOAT16;
            world.Resize({1, inputIds.dims[1], embed_dim});
            attentionParallel->transfer->Collect(tpRank, normed.cudaData, world.cudaData,
                size_t(embed_dim) * sizeof(uint16_t));
            Linear(world, localWeight(prefix + ".mlp.gate_proj.weight"), Data(), q);
            Linear(world, localWeight(prefix + ".mlp.up_proj.weight"), Data(), qkv);
            Silu(q, q);
            MulTo(q, qkv);
            Linear(q, localWeight(prefix + ".mlp.down_proj.weight"), Data(), world);
            reduce(world);
            if (length) {
                attentionParallel->transfer->Select(tpRank, world.cudaData, projected.cudaData,
                    size_t(embed_dim) * sizeof(uint16_t));
                addMlpResidual(projected, false);
            }
        } else if (!moeLayers[layer]) {
            // Dense and routed-expert scratch have independent lifetimes.
            Data &w1 = workspace ? buf.denseGate : buf.w1;
            Data &w2 = workspace ? buf.denseDown : buf.w2;
            Data &w3 = workspace ? buf.denseUp : buf.w3;
            if(denseFused) {
                tpSharedFlex->Dense(tpRank,length,hidden,
                    localWeight(prefix+".mlp.gate_proj.weight"),localWeight(prefix+".mlp.up_proj.weight"),
                    localWeight(prefix+".mlp.down_proj.weight"),localWeight(prefix+".post_attention_layernorm.weight"),
                    rms_norm_eps,w1,w2);
                addMlpResidual(w2,false);
            } else {
#ifndef USE_ROCM
            // A specialized collective miss still retains the local fusion.
            // The existing reduce/residual code owns all TP/DP semantics.
            const bool portableDense = decodeWorkspace &&
                FastllmCudaDecodeDense(normed, localWeight(prefix + ".mlp.gate_proj.weight"),
                    localWeight(prefix + ".mlp.up_proj.weight"),
                    localWeight(prefix + ".mlp.down_proj.weight"), w1, w2);
#else
            const bool portableDense = false;
#endif
            if (!portableDense) {
                auto project = [&](Data &input, Data &weight, Data &output) {
                    if (stablePrefill && FastllmCudaBFloat16PrefillLinear(input, weight, output)) return;
                    Linear(input, weight, Data(), output);
                };
                project(normed, localWeight(prefix + ".mlp.gate_proj.weight"), w1);
                project(normed, localWeight(prefix + ".mlp.up_proj.weight"), w3);
                Silu(w1, w1);
                MulTo(w1, w3);
                project(w1, localWeight(prefix + ".mlp.down_proj.weight"), w2);
            }
            // NCCL BF16 may round partial sums at different ranks as the message
            // grows. The row reduction accumulates ranks in FP32 and chunks large messages.
            const bool reduced = stablePrefill && FastllmCudaCustomAllReduceRows(
                w2.cudaData, w2.cudaData, w2.Count(0), embed_dim, w2.dataType, gpu);
            addMlpResidual(w2, !reduced);
            }
        } else if (tensorParallel && tpFlexMoe) {
            Data &routerWeight = localWeight(prefix + ".mlp.gate.weight");
            if (!attentionRouterReady && !FastllmCudaNaiveRouterSigmoid(normed, routerWeight, router)) {
                // Preserve the existing exact-batch router policy. In particular,
                // an eight-row verify can require FP32 GEMM instead of the small
                // fused router, independently of FlexMoE's supported row count.
                ToDataType(normed, routerInput, DataType::FLOAT32);
                Linear(routerInput, routerWeight, Data(), router);
                Sigmoid(router, router);
            }
            Data &routerBias = localWeight(prefix + ".mlp.gate.e_score_correction_bias");
            tpFlexMoe->Forward(tpRank, layer, length, normed, router, routerBias, hidden);
            if (mlpNormWeight) norm(hidden, *mlpNormWeight, *mlpNormOutput);
        } else {
            // EP ranks route only their contiguous token shard, borrowing
            // the normalized activation until MegaMoE finishes reading it.
            auto &executor = *(Executor *)GetExecutor();
            Data *dispatchScratch = nullptr;
            bool hasRouterProbabilities = false;
            bool sharedRouting = false;
            Data &routerBias = localWeight(prefix + ".mlp.gate.e_score_correction_bias");
            // Every local EP source needs the same arena capacity, even an
            // empty DP group. FP32 router input is dead before MoE dispatch.
            if (localMoe && !decodeWorkspace) reservePrefillArena();
            if (!localMoe || length) {
                Data routedInput;
                if (shardRouter) {
                    const int first = (int)((int64_t)length * tpRank / epRanks);
                    const int end = (int)((int64_t)length * (tpRank + 1) / epRanks);
                    routedInput.Resize({end - first, embed_dim});
                    routedInput.FakeFrom(normed, (size_t)first * embed_dim * normed.unitSize);
                }
                Data &routerSource = shardRouter ? routedInput : normed;
                Data &routerWeight = localWeight(prefix + ".mlp.gate.weight");
                const bool selected = !useSharedFlex && verifying && prefillScores &&
                    FastllmCudaNaiveVerifyRouter(routerSource, routerWeight, routerBias, *prefillScores,
                        expertIndex, expertScore, num_experts_per_tok, norm_topk_prob, routed_scaling_factor);
                if (!selected) {
                    const bool localRouter = localMoe && !decodeWorkspace && !verifying &&
                        FastllmCudaNaivePrefillRouter(routerSource, routerWeight, routerInput, router);
                    if (!localRouter && canonicalPrefillRouter)
                        RunLocalPrefillRouter(normed, routerWeight, attentionParallel->rows.localToGlobal[tpRank],
                            inputIds.dims[1], epRanks, routerInput, router);
                    hasRouterProbabilities = !localRouter && !canonicalPrefillRouter && (attentionRouterReady ||
                        FastllmCudaNaiveRouterSigmoid(routerSource, routerWeight, router));
                    if (!localRouter && !canonicalPrefillRouter && !hasRouterProbabilities) {
                        Data localRouterInput;
                        if (shardPrefillRouter) {
                            // Share one backing allocation with the prefill arena;
                            // only the logical local rows are converted/projected.
                            if (!localMoe) reservePrefillArena();
                            localRouterInput.FakeFrom(routerInput, 0);
                        }
                        Data &fp32Input = shardPrefillRouter ? localRouterInput : routerInput;
                        ToDataType(routerSource, fp32Input, DataType::FLOAT32);
                        const bool prefillRouter = !verifying && !decodeWorkspace && length > 1 &&
                            !batchDecode;
                        if (!prefillRouter || !FastllmCudaNaiveRouterFP32(fp32Input, routerWeight, router, routerInput))
                            Linear(fp32Input, routerWeight, Data(), router);
                    }
                    sharedRouting = useSharedFlex && num_experts_per_tok == 8 && tpSharedFlex->CanRoute(length, router, routerBias);
                    bool fusedRouter = false;
                    // Small decode/verify batches already include sigmoid in their shared
                    // router projection. Other multi-row paths fuse it with selection.
                    if (!sharedRouting && !hasRouterProbabilities && (length > 1 || canonicalPrefillRouter) && router.dataDevice == DataDevice::CUDA) {
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
                        if (!sharedRouting) SelectExpert(router, expertIndex, expertScore, num_experts_per_tok,
                                     norm_topk_prob, routed_scaling_factor, &routerBias);
                    }
                    if (!decodeWorkspace && !hasRouterProbabilities) dispatchScratch = &routerInput;
                }
            } else {
                expertIndex.dataType = INT32; expertIndex.UpdateUnitSize(); expertIndex.Resize({0, num_experts_per_tok});
                expertScore.dataType = FLOAT32; expertScore.UpdateUnitSize(); expertScore.Resize({0, num_experts_per_tok});
            }
            normed.Reshape({length, embed_dim});
            if (!tensorParallel) ApplyMoeDeviceMapForLayer(layer);
            bool moeNormReady = false, moeTpReduced = false;
            if (useSharedFlex) {
                if (sharedRouting) tpSharedFlex->ForwardRouted(tpRank, layer, length, normed, router, routerBias,
                    expertIndex, expertScore, moeOutput, norm_topk_prob, routed_scaling_factor, attentionInputPrepared ? prepared : nullptr);
                else tpSharedFlex->Forward(tpRank,layer,length,normed,expertIndex,expertScore,moeOutput);
#ifndef USE_ROCM
            } else if (tensorParallel && !tpMarlinRanks.empty()) {
                AssertInFastLLM(RunMxfp4MarlinTP(*tpMarlinRanks.at(tpRank), layer, normed,
                    expertIndex, expertScore, w1, moeOutput, verifying && !localMoe ? &moeTpReduced : nullptr),
                    "Prepared MXFP4 Marlin TP execution failed.");
                if (moeTpReduced) trackCommunication(moeOutput);
#endif
            } else if (localMoe) {
                FastllmCudaRunMimoEp8MegaLocal(*tpEp8MegaState, tpRank, layer, normed,
                    expertIndex, expertScore, moeOutput, maxLocalRows,
                    decodeWorkspace ? (const int *)localDP->liveKeys.cudaData : nullptr,
                    decodeWorkspace ? nullptr : &routerInput,
                    !decodeWorkspace && !verifying ? minLocalRows : 0);
            } else if (tensorParallel && tpEp8MegaState) {
                moeNormReady = FastllmCudaRunMimoEp8Mega(*tpEp8MegaState, tpRank, layer, normed,
                    expertIndex, expertScore, moeOutput, decodeWorkspace || shardMoe,
                    (shardMoe || shardPrefillNorm) ? &hidden : nullptr,
                    mlpNormWeight, mlpNormOutput, rms_norm_eps,
                    dispatchScratch,
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
            } else if ((batchDecode || verifying) && length > 1 &&
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
            if (!moeNormReady && (!localMoe || length))
                addMlpResidual(moeOutput, !moeTpReduced && !useSharedFlex && !(tensorParallel && tpEp8MegaState));
        }
        prefillNormReady = fusePrefillNorm && mlpNormWeight;
        const auto featureLayer = capture && capture->collectHidden
            ? std::find(draftTargetLayers.begin(), draftTargetLayers.end(), layer) : draftTargetLayers.end();
        const bool captureHidden = featureLayer != draftTargetLayers.end();
        // Between local MoE layers, only the draft consumer needs world order.
        // Unpack directly to its persistent buffer on the rank that owns it.
        const bool collectIntoFeature = localMoe && layer + 1 < block_cnt && tpRank == 0 &&
            captureHidden && !capture->packedOutput && !capture->peerFeatures && workspace && capture->borrowOutputs &&
            !decodeWorkspace && !capture->verifying;
        if (localMoe) {
            // The vocabulary head and draft feature consumers require world
            // order. No intermediate attention or MoE layer needs that gather.
            if (layer + 1 == block_cnt && decodeWorkspace) {
                buf.last.dataType = BFLOAT16; buf.last.UpdateUnitSize();
                buf.last.Resize({1, inputIds.dims[1], embed_dim});
                buf.last.ToDevice(CUDA, {gpu}, false); buf.last.Allocate(false);
                attentionParallel->transfer->Collect(tpRank, normed.cudaData, buf.last.cudaData, size_t(embed_dim) * 2);
            }
            // Feature capture needs every row. Ordinary prefill passes only
            // the last local row of each sequence to the vocabulary head.
            if ((captureHidden && !capture->packedOutput && !capture->peerFeatures) ||
                (layer + 1 == block_cnt && !decodeWorkspace && !compactPrefillHead)) {
                Data &world = collectIntoFeature ? buf.features[layer] : buf.hidden;
                if (collectIntoFeature) {
                    world.dataType = buf.hidden.dataType;
                    world.UpdateUnitSize();
                    world.Resize(buf.hidden.dims);
                    world.ToDevice(CUDA, {gpu}, false);
                    world.Allocate(false);
                }
                attentionParallel->transfer->Collect(tpRank, hidden.cudaData, world.cudaData, size_t(embed_dim) * 2);
            }
        }
        const bool draftRankFeature = draftFusedContextFeatures && capture && capture->verifying &&
            tpRank > 0 && tpRank < (int)draftTargetLayers.size() && layer == draftTargetLayers[tpRank];
        if (captureHidden && capture->peerFeatures) {
            auto *dest = static_cast<uint16_t *>(capture->peerFeatures->at(layer).cudaData);
            const auto *source = static_cast<const uint16_t *>(hidden.cudaData);
            for (const auto &slice : capture->featureSlices)
                AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(tpDevices.front(),
                    dest + size_t(slice.world) * embed_dim, gpu, source + size_t(slice.local) * embed_dim,
                    size_t(slice.count) * embed_dim * sizeof(*source)), "Cannot copy local draft prefill rows");
        } else if (captureHidden && capture->packedOutput) {
            const size_t column = featureLayer - draftTargetLayers.begin();
            auto dst = (uint16_t *)capture->packedOutput->cudaData + column * embed_dim;
            const size_t rowBytes = (size_t)embed_dim * sizeof(uint16_t);
            AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(dst,
                draftTargetLayers.size() * rowBytes, hidden.cudaData, rowBytes, rowBytes, hidden.dims[1]),
                "Cannot capture local draft features");
        } else if (captureHidden && (tpRank <= 0 || draftRankFeature)) {
            Data &featureHidden = localMoe ? buf.hidden : hidden;
            if (pendingMlp) {
                // The following attention layer consumes the deferred residual and exports
                // this layer's materialized BF16 hidden before adding attention.
                // Warmup reserves the same feature storage used by graph replay.
                auto &feature = capture->hidden[layer];
                feature.dataType = hidden.dataType;
                feature.UpdateUnitSize();
                feature.Resize(hidden.dims);
                feature.ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false);
                feature.Allocate(false);
                pendingFeature = &feature;
            } else if (workspace && capture->borrowOutputs && !capture->verifying) {
                // Internal prefill consumers finish before the next forward.
                // Retain the feature storage just like the other TP scratch.
                auto &backing = buf.features[layer];
                if (!collectIntoFeature) Copy(featureHidden, backing);
                auto &view = capture->hidden[layer];
                view.FreeSpace(); view.FakeFrom(backing, 0); view.Resize(backing.dims);
            } else Copy(featureHidden, capture->hidden[layer]);
        }
    }
    if (isIntermediateChunkedPrefill) return Data();
    Data localLogits;
    Data &last = buf.last, &logits = decodeWorkspace ? buf.logits : localLogits;
    if (!decodeWorkspace) {
        if (verifying) Copy(hidden, last);
        else if (compactPrefillHead || batch) {
            const int sequences = batch ? batch->Size() : 1;
            last.dataType = BFLOAT16;
            last.Resize({1, sequences, embed_dim});
            last.ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false);
            last.Allocate(false);
            if (compactPrefillHead) {
                const auto &group = attentionParallel->groups[tpRank];
                const auto &local = attentionParallel->ranks[tpRank].hidden;
                FastllmCudaMemset0(last.cudaData, last.GetBytes());
                int end = 0;
                for (size_t i = 0; i < group.lengths.size(); ++i) {
                    end += group.lengths[i];
                    auto *dest = static_cast<uint16_t *>(last.cudaData) + size_t(group.sequenceIndices[i]) * embed_dim;
                    auto *source = static_cast<const uint16_t *>(local.cudaData) + size_t(end - 1) * embed_dim;
                    AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(dest, source,
                        size_t(embed_dim) * sizeof(uint16_t)), "Failed to select the last local prefill row");
                }
                // Every owner is a group leader. Summing bytes with zeros from
                // the other ranks preserves the selected BF16 rows bit for bit.
                auto &runtime = attentionParallel->runtime;
                runtime.AllReduce(runtime.Layout().leaders, tpRank, last.cudaData, last.GetBytes(),
                    parallel::ScalarType::Byte);
            } else {
                int offset = 0;
                for (int sequence = 0; sequence < sequences; ++sequence) {
                    offset += batch->lengths[sequence];
                    FastllmCudaCopyFromDeviceToDevice((uint16_t *)last.cudaData + (size_t)sequence * embed_dim,
                        (uint16_t *)hidden.cudaData + (size_t)(offset - 1) * embed_dim, (size_t)embed_dim * sizeof(uint16_t));
                }
            }
        } else Split(hidden, 1, length - 1, length, last);
        norm(last, localWeight("model.norm.weight"), last);
    }
    if (decodeWorkspace) {
#ifndef USE_ROCM
        if (fusedHead) {
            if (FastllmCudaDecodeNormLinear(hidden, headBranch, localWeight("model.norm.weight"),
                    localWeight("lm_head.weight"), logits, rms_norm_eps,
                    buf.samplingLogits, buf.samplingOffset)) {
                buf.directSamplingLogits = buf.samplingLogits != nullptr;
                return Data();
            }
            // A rejected branch view submits no work. Restore the ordinary
            // residual and norm before the established head/cast fallback.
            if (headBranch)
                FastllmCudaNaiveAddDecodeRMSNorm(hidden, *headBranch,
                    localWeight("model.norm.weight"), rms_norm_eps, last);
            else norm(hidden, localWeight("model.norm.weight"), last);
        }
#endif
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

NaiveN05FlashModel::LogitsOptions NaiveN05FlashModel::GetLogitsOptions(
        const GenerationConfig &config, bool speculative) const {
    LogitsOptions selection;
    // Stateful transforms need the request's history/cache lengths, but the
    // logits remain on GPU. Only explicit logits output retains host transfer.
    if (config.prefill_chunk_remaining > 0 || config.output_logits) return selection;
    if (config.output_token_least > 0 || std::abs(config.repeat_penalty - 1.0f) > 1e-8f ||
        !config.tool_call_allowed_token_ids.empty()) {
        selection.cudaLogits = !speculative;
        return selection;
    }
    selection.greedy = config.IsSimpleGreedy();
    // Grammar masks require full logits before sampling. The continuous DP
    // sampler applies compiled masks inside its startup-captured graph.
    if (config.tool_call_token_mask && config.top_k <= 1) selection.greedy = true;
    // The shared FlashInfer path preserves top-k-then-top-p, including finite
    // k and stable boundary ties. Verification still needs probabilities.
    if (!speculative && (selection.greedy ||
        (std::isfinite(config.temperature) && config.temperature > 0 &&
         std::isfinite(1.0f / config.temperature))) &&
        std::isfinite(config.top_p) && config.top_p > 0 && config.top_p <= 1) {
        selection.cudaSampling = true;
        return selection;
    }
    if (config.tool_call_grammar || config.tool_call_token_mask) return LogitsOptions();
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

NaiveN05FlashModel::LogitsSelection NaiveN05FlashModel::SelectLogits(
        const GenerationConfig &config, bool speculative) const {
    return {GetLogitsOptions(config, speculative), {}, {}};
}

bool NaiveN05FlashModel::SupportsCudaRejectionSampling(const GenerationConfig &config) const {
    return config.top_k > 1 && GetLogitsOptions(config, false).cudaSampling;
}

#ifdef USE_CUDA
static void SampleNaiveCudaRows(Data &logits, const GenerationConfig *configs, int rows,
                                const LastTokensManager *lastTokens, int firstRow, int *tokens) {
    const int vocab = logits.dims.back();
    std::vector<float> temperatures(rows), topPs(rows);
    std::vector<int> topKs(rows, vocab);
    for (int row = 0; row < rows; ++row) {
        temperatures[row] = configs[row].top_k <= 1 ? 1.0f : configs[row].temperature;
        topPs[row] = configs[row].top_p;
        topKs[row] = std::max(1, configs[row].top_k);
    }
    // Combine explicit allow-lists with grammar masks before GPU selection.
    // An empty standalone allow-list intersection keeps the established
    // unconstrained fallback; an impossible grammar remains an error.
    std::vector<GenerationConfig> masked(configs, configs + rows);
    for (int row = 0; row < rows; ++row) if (!configs[row].tool_call_allowed_token_ids.empty()) {
        auto mask = std::make_shared<std::vector<uint32_t>>((vocab + 31) / 32, 0);
        bool any = false;
        for (int id : configs[row].tool_call_allowed_token_ids)
            if (id >= 0 && id < vocab && configs[row].AllowsToolToken(id)) {
                (*mask)[id / 32] |= uint32_t(1) << (id % 32); any = true;
            }
        AssertInFastLLM(any || !configs[row].tool_call_token_mask, "Empty tool grammar sampling support.");
        if (any) masked[row].tool_call_token_mask = std::move(mask);
    }
    ApplyToolTokenMasks(logits, masked);
    std::vector<std::vector<std::pair<int, float>>> penalties(rows);
    size_t maxPenalty = 0;
    if (lastTokens) for (int row = 0; row < rows && firstRow + row < (int)lastTokens->units.size(); ++row) {
        if (std::abs(configs[row].repeat_penalty - 1.0f) <= 1e-8f) continue;
        std::map<int, int> counts;
        for (int id : lastTokens->units[firstRow + row].tokenSet) if (id >= 0 && id < vocab) ++counts[id];
        for (auto entry : counts) penalties[row].push_back({entry.first,
            std::pow(configs[row].repeat_penalty, configs[row].last_n <= 0 ? 1 : entry.second)});
        maxPenalty = std::max(maxPenalty, penalties[row].size());
    }
    if (maxPenalty) {
        // Reuse GPU buffers; only small request parameters cross from the CPU.
        struct Workspace { Data probs, params, output, floatOutput, topK; };
        static thread_local std::map<int, Workspace> workspaces;
        const int device = logits.dataDeviceIds.front(); FastllmCudaSetDevice(device);
        auto &w = workspaces[device];
        const size_t fields = size_t(rows) * (3 + 2 * maxPenalty);
        AssertInFastLLM(fields <= INT_MAX, "Sampling parameter buffer too large.");
        auto prepare = [&](Data &data, DataType type, std::vector<int> dims) {
            data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
            data.ToDevice(CUDA, {device}, false); data.Allocate();
        };
        prepare(w.probs, FLOAT32, {rows, vocab}); prepare(w.params, INT32, {int(fields)});
        prepare(w.output, INT32, {rows}); prepare(w.floatOutput, FLOAT32, {rows});
        prepare(w.topK, INT8, {int(FastllmCudaTopKFirstSamplingWorkspaceBytes(rows))});
        std::vector<uint32_t> params(fields, 0);
        std::memcpy(params.data(), temperatures.data(), rows * sizeof(float));
        std::memcpy(params.data() + rows, topKs.data(), rows * sizeof(int));
        std::memcpy(params.data() + 2 * rows, topPs.data(), rows * sizeof(float));
        int *ids = (int *)(params.data() + 3 * rows);
        float *factors = (float *)(ids + rows * maxPenalty);
        std::fill(ids, ids + rows * maxPenalty, -1);
        std::fill(factors, factors + rows * maxPenalty, 1.f);
        for (int row = 0; row < rows; ++row) for (size_t j = 0; j < penalties[row].size(); ++j) {
            ids[row * maxPenalty + j] = penalties[row][j].first;
            factors[row * maxPenalty + j] = penalties[row][j].second;
        }
        FastllmCudaCopyFromHostToDevice(w.params.cudaData, params.data(), fields * sizeof(uint32_t));
        auto *gpu = (uint32_t *)w.params.cudaData;
        AssertInFastLLM(FastllmCudaTopKFirstSamplingToDeviceWithRng((float *)logits.cudaData,
            (float *)w.probs.cudaData, (float *)gpu, (int *)gpu + rows, (float *)gpu + 2 * rows,
            (int *)gpu + 3 * rows, (float *)gpu + 3 * rows + rows * maxPenalty, int(maxPenalty),
            (int *)w.output.cudaData, (float *)w.floatOutput.cudaData, rows, vocab,
            nullptr, nullptr, w.topK.cudaData, w.topK.GetBytes()), "GPU penalized sampling failed.");
        FastllmCudaCopyFromDeviceToHost(tokens, w.output.cudaData, rows * sizeof(int));
        return;
    }
    AssertInFastLLM(FastllmCudaTopKFirstSampling((float *)logits.cudaData,
        temperatures.data(), topKs.data(), topPs.data(), tokens, rows, vocab),
        "Naive CUDA top-p sampling failed.");
}
#endif

std::vector<int> NaiveN05FlashModel::SampleCudaLogits(
        Data &logits, const std::vector<GenerationConfig> &configs, const LastTokensManager *lastTokens) {
#ifdef USE_CUDA
    AssertInFastLLM(logits.dataDevice == DataDevice::CUDA && logits.dataType == FLOAT32 &&
        logits.Count(0) / logits.dims.back() == configs.size(), "Naive CUDA sampling: invalid logits.");
    FastllmCudaSetDevice(logits.dataDeviceIds.at(0));
    const int rows = configs.size(), vocab = logits.dims.back();
    std::vector<int> tokens(rows, 0);
    for (int first = 0; first < rows;) {
        if (configs[first].prefill_chunk_remaining > 0) {
            ++first;
            continue;
        }
        int end = first + 1;
        while (end < rows && configs[end].prefill_chunk_remaining <= 0) ++end;
        // Sample a view of consecutive final rows without gathering full-vocabulary logits.
        Data view;
        view.FakeFrom(logits, size_t(first) * vocab * sizeof(float));
        view.Resize({end - first, vocab});
        SampleNaiveCudaRows(view, configs.data() + first, end - first, lastTokens, first, tokens.data() + first);
        first = end;
    }
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
    if (logits.dataDevice == CUDA)
        return SampleCudaLogits(logits, {generationConfig}, &lastTokens)[0];
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
    preparingGraphs = true;
    draftCapturing = false;
    struct SealDraftGraphs {
        NaiveN05FlashModel &model;
        ~SealDraftGraphs() { model.preparingGraphs = model.draftCapturing = false; model.draftPrepareCapacity = 0; }
    } sealDraftGraphs{*this};
    const DataType requestedKV = kvCacheDataType;
    if (requestedKV == FP4_E2M1) {
        AssertInFastLLM(tpDevices.size() > 1 && !GetKVCacheInCPU() &&
                        (!draftEnabled || (attentionDP > 1 && !kvMemConfig.enabled)),
                        "Naive FP4 DSA KV requires CUDA TP; FP4 with draft requires attention DP and GPU-resident KV.");
        AssertInFastLLM(full.headDim % 16 == 0 && full.valueDim % 16 == 0 &&
                        full.headDim <= 256 && full.valueDim <= 256,
                        "Naive FP4 DSA KV requires block16-aligned head widths <= 256.");
        // The first forward initializes weights before the final KV pool exists.
        // Its private, discarded caches use the established FP8 bootstrap path.
        kvCacheDataType = FP8_E4M3;
    }
    struct RestoreKVType {
        DataType &current; DataType requested;
        ~RestoreKVType() { current = requested; }
    } restoreKV{kvCacheDataType, requestedKV};
    Data ids(DataType::FLOAT32, {1, 1}, {1.0f});
    Data positions(DataType::FLOAT32, {1, 1}, {0.0f});
    std::vector<std::pair<Data, Data>> cache;
    for (int layer = 0; layer < block_cnt; layer++)
        cache.emplace_back(Data(kvCacheDataType), Data(kvCacheDataType));
    {
        struct RestorePhase {
            bool &flag; bool saved;
            ~RestorePhase() { flag = saved; }
        } restore{isIntermediateChunkedPrefill, isIntermediateChunkedPrefill};
        if (pdPrefillOnly && draftEnabled) isIntermediateChunkedPrefill = true;
        Forward(ids, Data(), positions, cache);
    }
    if (draftEnabled) {
        if (!pdPrefillOnly) RunDraft(1, *draftContexts.at(&cache));
        draftContexts.erase(&cache);
    }
    elementsInKVCachePerToken = 0;
    for (int layer = 0; layer < block_cnt; layer++) {
        if (slidingLayers[layer]) continue;
        if (tpDevices.size() > 1) {
            // Count physical storage, including replicated KV heads and the
            // configured Indexer placement, for the scheduler's token budget.
            for (int device : cache[layer].first.dataDeviceIds)
                elementsInKVCachePerToken += cache[layer].first.multiDeviceDatas.at(device)->dims[2] +
                                            cache[layer].second.multiDeviceDatas.at(device)->dims[2];
        } else {
            elementsInKVCachePerToken += cache[layer].first.dims[2] + cache[layer].second.dims[2];
        }
    }
    cache.clear();
#ifdef USE_CUDA
    PrepareTensorParallelMemory();
    // Bootstrap workspaces may use FP8. Permanent pages, Indexer sidecars
    // and serving graphs must all see the requested physical storage type.
    kvCacheDataType = requestedKV;
    if (attentionDP > 1) {
        PrepareAttentionParallelSampling();
        if (kvMemConfig.enabled) InitSparseKv();
        else InitTensorParallelKVPool();
        if (!kvMemConfig.enabled) PrepareTensorParallelServing();
        return;
    }
    // Ragged prefill warmup may leave large idle allocator blocks. Return only
    // those blocks before KVMem sizing; graph-pinned and busy workspaces stay.
    if (draftEnabled && prefillBatchPolicy.targetTokens > 0)
        FastllmCudaReleaseIdleCachedBuffers();
    if (kvMemConfig.enabled) {
        InitSparseKv();
        if (draftEnabled) {
            // Native warmup reserved activations before sizing storage. Prepare
            // the actual sparse bindings/workspaces before accepting requests.
            for (auto &graph : tpBatchVerifyGraphs) graph->ClearGraphs();
            tpBatchVerifyGraphs.clear(); tpBatchVerifyState.reset();
            tpVerifyState.reset(); tpDecodeState.reset();
            PrepareTensorParallelMemory();
            if (prefillBatchPolicy.targetTokens > 0)
                FastllmCudaReleaseIdleCachedBuffers();
        }
    } else InitTensorParallelKVPool();
    PrepareTensorParallelServing();
    if (draftEnabled) {
        draftCapturing = true;
        PrepareTensorParallelVerification();
        PrepareDraftServingMemory(saveHistoryChat);
    }
#endif
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
        data.dataType = AuxiliaryKVType();
        data.UpdateUnitSize();
        data.dataDevice = DataDevice::CUDA;
        data.dataDeviceIds = {tpDevices[rank]};
        const int width = std::max(1, sliding.kvHeads / AttentionTPSize()) *
            (part ? sliding.valueDim : sliding.headDim);
        data.Expansion({1, capacity, width});
    }
#endif
}

std::shared_ptr<CudaWorkspace> NaiveN05FlashModel::RequestKVPool(int device, size_t bytes) const {
    auto it = tpRequestKVPool.find({device, (bytes + 255) / 256 * 256});
    return it != tpRequestKVPool.end() ? it->second : tpKVPool.at(device);
}

void NaiveN05FlashModel::InitTensorParallelKVPool() {
#ifdef USE_CUDA
    // Keep CPU/layer-serial execution on its existing cache path. This pool
    // serves BF16/FP8/FP4 TP, independently of GPU generation and CUDA Graph.
    // Draft checkpoints distribute normalized inputs across the TP GPUs.
    // Restore rebuilds owner-local or sharded draft KV through AppendDraftContext.
    if (!tpKVPool.empty() || !tpPrepared || tpDevices.size() <= 1 ||
        GetKVCacheInCPU() || !FastllmNaiveAttentionKVType(kvCacheDataType) ||
        elementsInKVCachePerToken <= 0) return;
    const double gpuMemRatio = GetGpuMemRatio();
    AssertInFastLLM(std::isfinite(gpuMemRatio) && gpuMemRatio > 0 && gpuMemRatio <= 1,
        "Naive GPU KV pool requires 0 < --gpu_mem_ratio <= 1.");
    const int ranks = tpDevices.size();
    const int batch = maxBatch > 0 ? maxBatch : 512;
    const int chunk = std::max(1, GetChunkedPrefillSize());
    const int localBatch = WindowRequestsPerGroup(batch);
    const int64_t swaCapacity = ((int64_t)window + (draftEnabled ? draftBlock : 0) + 127) / 128 * 128;
    if (std::count(slidingLayers.begin(), slidingLayers.end(), 1)) {
        tpSwaPrefillWorkspaces.resize(ranks);
        for (int rank = 0; rank < ranks; ++rank) PrepareSwaPrefillWorkspace(rank, chunk);
    }
    const int kvBytes = AuxiliaryKVType() == FP8_E4M3 ? 1 : 2;
    const bool separateIndex = kvCacheDataType == FP4_E2M1 || indexerContextSharding;
    int64_t fullBytesPerToken = 0, fixedBytes = 0;
    std::vector<int64_t> rankBytesPerToken(ranks, 0);
    for (int layer = 0; layer < block_cnt; ++layer) {
        const auto &cfg = slidingLayers[layer] ? sliding : full;
        if (slidingLayers[layer]) {
            for (int dim : {cfg.headDim, cfg.valueDim}) {
                const int64_t size = std::max(1, cfg.kvHeads / AttentionTPSize()) * dim * kvBytes * swaCapacity;
                fixedBytes += (size + 255) / 256 * 256 * localBatch;
            }
        }
        else for (int rank = 0; rank < ranks; ++rank) {
            const int heads = std::max(1, cfg.kvHeads / AttentionTPSize());
            rankBytesPerToken[rank] += GetDataBytes(kvCacheDataType, 1,
                heads * cfg.headDim + (separateIndex ? 0 : LocalIndexerCacheWidth(layer, rank)));
            rankBytesPerToken[rank] += GetDataBytes(kvCacheDataType, 1, heads * cfg.valueDim);
            if (separateIndex && HasLocalIndexer(layer, rank))
                rankBytesPerToken[rank] += (int64_t)IndexerCacheWidth(AuxiliaryKVType()) * kvBytes /
                    (indexerContextSharding ? AttentionTPSize() : 1);
        }
    }
    fullBytesPerToken = *std::max_element(rankBytesPerToken.begin(), rankBytesPerToken.end());
    AssertInFastLLM(fullBytesPerToken > 0, "Naive KV pool requires full-attention layers.");
    // Scratch is allocated before measuring free memory, so the total budget
    // accounts for it without multiplying it by layers or concurrent requests.
    if (indexerContextSharding) {
        const int capacity = (max_positions + GetPageLen() - 1) / GetPageLen() * GetPageLen();
        for (int rank = 0; rank < ranks; ++rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            auto &buf = attentionParallel ? attentionParallel->ranks[rank].workspace : *tpPrefillWorkspaces.at(rank);
            auto reserve = [&](Data &data) {
                data.dataType = AuxiliaryKVType();
                data.UpdateUnitSize();
                data.Resize({1, capacity, IndexerCacheWidth(AuxiliaryKVType())});
                data.ToDevice(DataDevice::CUDA, {tpDevices[rank]}, false);
                data.Allocate(false);
            };
            reserve(buf.indexContextGather);
            reserve(buf.indexContextFull);
            // Deferred prefill scoring retains at most eight histories. Charge
            // these bounded buffers before sizing the persistent GPU page pool.
            const int prefillRows = std::min(batch, 8);
            while ((int)buf.sequences.size() < prefillRows)
                buf.sequences.emplace_back(new TargetWorkspace());
            if (prefillRows > 1) for (int row = 0; row < prefillRows; ++row)
                reserve(buf.sequences[row]->indexContextFull);
        }
    }
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
            const int width = std::max(1, sliding.kvHeads / AttentionTPSize()) *
                (part ? sliding.valueDim : sliding.headDim);
            prefixes->slotBytes += (size_t)window * width * kvBytes;
        }
    }
    prefixes->slotBytes = (prefixes->slotBytes + 255) / 256 * 256;
    if (draftEnabled) prefixes->draftSlotBytes = (size_t)(draftWindow - 1) * embed_dim * sizeof(uint16_t);
    const int regularSlots = reserveWindowHistory ? prefixes->SlotCount(localBatch, budget) : 0;
    int prefixSlots = regularSlots;
    int totalSlots = prefixSlots * attentionDP;
    int64_t prefixBytes = (int64_t)prefixSlots * prefixes->slotBytes +
        (int64_t)PrefixPool::DraftShardSlots(totalSlots, 0, ranks) * prefixes->draftSlotBytes;
    fixedBytes += prefixBytes;
    int64_t tokens = GetMaxTokens() > 0 ? GetMaxTokens() :
        std::min<int64_t>((int64_t)max_positions * batch,
                         (budget - fixedBytes) / fullBytesPerToken - (int64_t)batch * 127);
    if (tokensLimit > 0) tokens = std::min<int64_t>(tokens, tokensLimit);
    tokens = std::min<int64_t>(tokens, INT_MAX - 127);
    // Preserve the existing budget, including a partial page per request.
    int64_t bytes = fixedBytes + (tokens + (int64_t)batch * 127) * fullBytesPerToken;
    if (pdPrefillOnly && prefixSlots > 0) {
        const int extraDraftRows = PrefixPool::DraftShardSlots(totalSlots + attentionDP, 0, ranks) -
                                   PrefixPool::DraftShardSlots(totalSlots, 0, ranks);
        const int64_t extra = prefixes->slotBytes + extraDraftRows * prefixes->draftSlotBytes;
        // Preserve the entire computed FULL capacity and every ordinary window.
        // When the startup budget is tight, use only reclaimable spare slots.
        if (extra <= budget - bytes) {
            prefixes->branchStart = totalSlots;
            prefixes->branchSlots.Init(attentionDP, 1);
            ++prefixSlots;
            totalSlots += attentionDP;
            prefixBytes += extra;
            fixedBytes += extra;
            bytes += extra;
        }
    }
    AssertInFastLLM(tokens >= 128 && bytes > 0 && bytes <= budget,
        "Naive startup KV pool exceeds the memory budget; reduce --tokens/--max_batch or prefill chunk size.");
    const int pageLen = GetPageLen();
    const int fullLayers = std::count(slidingLayers.begin(), slidingLayers.end(), 0);
    const auto graphBuckets = LocalServingGraphBuckets(batch);
    AssertInFastLLM(!graphBuckets.empty(), "Naive graph batch exceeds tensor dimensions.");
    const int graphRows = graphBuckets.back();
    const int64_t metadataBytes = (int64_t)fullLayers * 2 * batch *
        ((std::min<int64_t>(max_positions, tokens) + pageLen - 1) / pageLen * sizeof(int) + 512) +
        (int64_t)block_cnt * graphRows * (separateIndex ? 8 : 4) * sizeof(void *) + 512;
    const int pages = (int)((bytes - fixedBytes - metadataBytes) / fullBytesPerToken / pageLen);
    AssertInFastLLM(pages > 0, "Naive KV budget cannot fit paged metadata.");
    tokens = std::min<int64_t>(tokens, (int64_t)pages * pageLen);
    // Charge size classes to the existing budget, not additional device memory.
    // Each arena only serves one aligned block size, so arbitrary request
    // completion/restore order cannot leave unusable holes between live blocks.
    std::map<size_t, int64_t> requestBlocks;
    auto reserveBlocks = [&](size_t size, int64_t count) {
        requestBlocks[(size + 255) / 256 * 256] += count;
    };
    reserveBlocks((std::min<int64_t>(max_positions, tokens) + pageLen - 1) / pageLen * sizeof(int),
                  (int64_t)fullLayers * 2 * batch);
    for (int layer = 0; layer < block_cnt; ++layer) if (slidingLayers[layer]) {
        for (int dim : {sliding.headDim, sliding.valueDim})
            reserveBlocks(swaCapacity * std::max(1, sliding.kvHeads / AttentionTPSize()) * dim * kvBytes, localBatch);
    }
    int64_t requestBytes = 0;
    for (const auto &entry : requestBlocks) requestBytes += entry.first * entry.second;
    std::map<int, std::shared_ptr<CudaWorkspace>> pools;
    std::map<std::pair<int, size_t>, std::shared_ptr<CudaWorkspace>> requestPools;
    for (int rank = 0; rank < ranks; ++rank) {
        const int device = tpDevices[rank];
        // Keep page counts identical for prefix lookup/restore, but charge
        // each rank only for its own physical Indexer pages. Uneven layer
        // counts use the largest per-rank width when deriving total capacity.
        const int64_t rankBytes = bytes - (int64_t)pages * pageLen *
            (fullBytesPerToken - rankBytesPerToken[rank]);
        pools.emplace(device, std::make_shared<CudaWorkspace>(device, rankBytes - requestBytes));
        if (indexerLayerSharding) {
            int layers = 0;
            for (int layer = 0; layer < block_cnt; ++layer) layers += HasLocalIndexer(layer, rank);
            printf("[Naive Indexer storage] GPU %d owns %d/%d DSA layers; KV %lld bytes/token; pool %.2f MiB.\n",
                   device, layers, fullLayers, (long long)rankBytesPerToken[rank], rankBytes / 1048576.0);
        }
        for (const auto &entry : requestBlocks)
            requestPools.emplace(std::make_pair(device, entry.first),
                std::make_shared<CudaWorkspace>(device, entry.first * entry.second));
    }
    tpKVPool.swap(pools);
    tpSwaCacheCapacity = swaCapacity;
    if (attentionDP > 1) {
        activeWindowSlots = std::make_shared<WindowSlots>();
        activeWindowSlots->Init(attentionDP, localBatch, localBatch - RunningRequestsPerGroup(batch));
    }
    tpRequestKVPool.swap(requestPools);
    if (prefixSlots > 0) {
        prefixes->storage.resize(ranks);
        for (int rank = 0; rank < ranks; ++rank) {
            Data &storage = prefixes->storage[rank];
            storage.dataType = INT8;
            storage.UpdateUnitSize();
            storage.Resize({prefixSlots, (int)prefixes->slotBytes});
            storage.dataDevice = DataDevice::CUDA;
            storage.dataDeviceIds = {tpDevices[rank]};
            storage.cudaData = AllocateCudaWorkspace(tpKVPool.at(tpDevices[rank]), (size_t)prefixSlots * prefixes->slotBytes);
            if (!storage.cudaData) throw CudaWorkspaceExhausted("Naive GPU prefix pool exhausted.");
            storage.directMemory = true;
        }
        if (prefixes->draftSlotBytes) {
            prefixes->draftStorage.resize(ranks);
            for (int rank = 0; rank < ranks; ++rank) {
                const int slots = PrefixPool::DraftShardSlots(totalSlots, rank, ranks);
                if (!slots) continue;
                FastllmCudaSetDevice(tpDevices[rank]);
                Data &storage = prefixes->draftStorage[rank];
                storage.dataType = BFLOAT16; storage.UpdateUnitSize();
                storage.Resize({slots, draftWindow - 1, embed_dim});
                storage.dataDevice = DataDevice::CUDA;
                storage.dataDeviceIds = {tpDevices[rank]};
                storage.cudaData = AllocateCudaWorkspace(tpKVPool.at(tpDevices[rank]), storage.GetBytes());
                if (!storage.cudaData) throw CudaWorkspaceExhausted("Naive draft prefix pool exhausted.");
                storage.directMemory = true;
            }
        }
        prefixes->slots.Init(attentionDP, regularSlots);
        prefixPool = std::move(prefixes);
        printf("[Naive GPU prefix cache] %d window slots, %.2f MiB/GPU within KV budget; no CPU KV archive.\n",
               prefixSlots, prefixBytes / 1048576.0);
    }
    tpPagedCaches.resize(ranks);
    if (separateIndex) tpIndexerCaches.resize(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        FastllmCudaSetDevice(tpDevices[rank]);
        auto &managers = tpPagedCaches[rank];
        managers.resize(block_cnt * 2);
        if (separateIndex) tpIndexerCaches[rank].resize(block_cnt);
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (slidingLayers[layer]) continue;
            for (int part = 0; part < 2; ++part) {
                int width = std::max(1, full.kvHeads / AttentionTPSize()) * (part ? full.valueDim : full.headDim) + (!part ? LocalIndexerCacheWidth(layer, rank) : 0);
                auto manager = std::make_unique<PagedCacheManager>();
                manager->type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
                manager->kvCacheFp8Indexer = !separateIndex && !part && fp8IndexerCache && HasLocalIndexer(layer, rank);
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
                FastllmCudaMemset0(manager->cudaData, GetDataBytes(manager->dataType, pageLen, width));
                managers[layer * 2 + part] = std::move(manager);
            }
            if (separateIndex && HasLocalIndexer(layer, rank)) {
                auto index = std::make_unique<PagedCacheManager>();
                index->pageLen = indexerContextSharding ? pageLen / AttentionTPSize() : pageLen;
                index->maxPages = pages;
                index->kvCacheFp8Indexer = fp8IndexerCache;
                index->dataType = AuxiliaryKVType();
                index->UpdateUnitSize();
                index->dataDevice = DataDevice::CUDA;
                index->dataDeviceIds = {tpDevices[rank]};
                index->Resize({pages, index->pageLen, 1, IndexerCacheWidth(AuxiliaryKVType())});
                index->cudaData = AllocateCudaWorkspace(tpKVPool.at(tpDevices[rank]), index->GetBytes());
                if (!index->cudaData) throw CudaWorkspaceExhausted("Naive Indexer pool exhausted.");
                index->directMemory = true;
                FastllmCudaMemset0(index->cudaData, (size_t)index->pageLen * index->dims[3] * kvBytes);
                // Share K's page IDs/refcounts, including prefix forks and rollback.
                tpIndexerSidecars.emplace(managers[layer * 2].get(), index.get());
                tpIndexerCaches[rank][layer] = std::move(index);
            }
        }
        ForceDeviceSync();
    }
    // Keep the public token limit at one group's capacity. Request admission
    // uses each owner's physical pool; total capacity reporting must also
    // account for SWA, Indexer, graph and draft storage before scaling by DP.
    tokensLimit = (int)tokens;
    printf("[Naive KV pool] preallocated %.2f MiB/GPU, %lld total tokens, batch %d, SWA %d requests/group, capacity %lld rows/request. No serving-time KV cudaMalloc.\n",
        bytes / 1048576.0, (long long)tokens, batch, localBatch, (long long)swaCapacity);
    fflush(stdout);
#endif
}
}
