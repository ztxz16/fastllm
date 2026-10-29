#include "naive_n05_kvmem.h"
#include "naive_n05_tp.h"
#include "utils/decode_graph.h"
#include <climits>
#include <stdexcept>
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#endif

namespace fastllm {
void NaiveN05FlashModel::ConfigureKvMem(const KvMemConfig &config) {
    if (mainLoop || tpPrepared || sparseKv || kvMemLocked || autoWarmupRunning.load())
        throw std::invalid_argument("Configure KVMem before warmup or inference");
    if (!config.enabled) { kvMemConfig = config; return; }
#ifndef USE_CUDA
    throw std::invalid_argument("MiMo KVMem requires CUDA");
#else
    if (tpDevices.size() < 2 || GetKVCacheInCPU() || draftEnabled || dataType != BFLOAT16 ||
        !FastllmNaiveKVType(kvCacheDataType) || maxBatch < 1 || maxBatch > MaxTPDecodeBatch)
        throw std::invalid_argument("MiMo KVMem requires CUDA TP, BF16 compute, BF16/FP8 KV, explicit max_batch in [1,512], and no draft");
    if (config.maxTokens < 2 || config.maxTokens > max_positions ||
        config.residentTokens < indexTopK || config.residentTokens > config.maxTokens ||
        config.prefillTokens < 1 || !config.hostBytes || GetFastllmEnv().skipWarmup)
        throw std::invalid_argument("MiMo KVMem requires valid logical/resident/prefill budgets and startup warmup");
    const int owners = DecodeGraphBatchBuckets(maxBatch).back();
    const int page = GetPageLen();
    const uint64_t perRequest = (uint64_t(config.maxTokens) + page - 1) / page * page;
    // --tokens is the physical pool; max_tokens remains a per-request bound.
    const uint64_t tokens = GetMaxTokens() > 0
        ? uint64_t(GetMaxTokens() / page) * page : uint64_t(owners) * perRequest;
    const uint64_t bytes = tokens * std::count(slidingLayers.begin(), slidingLayers.end(), 0) *
        tpDevices.size() * std::max(1, full.kvHeads / (int)tpDevices.size()) *
        (full.headDim + full.valueDim) * (kvCacheDataType == FP8_E4M3 ? 1 : 2);
    if (tokens < perRequest || tokens < uint64_t(owners) * page || tokens > INT_MAX || bytes > config.hostBytes)
        throw std::invalid_argument("MiMo KVMem token pool must fit one max_tokens request, one page per graph slot, and the host budget");
    kvMemConfig = config;
    max_positions = std::min(max_positions, config.maxTokens);
    tokensLimit = promptLimit = (int)tokens;
#endif
}

#ifdef USE_CUDA
NaiveN05FlashModel::SparseKvState::~SparseKvState() {
    for (size_t rank = 0; rank < ranks.size(); ++rank) {
        FastllmCudaSetDevice(devices[rank]);
        ranks[rank].reset();
    }
}
#endif

void NaiveN05FlashModel::OnResponseContextFinished(ResponseContext *context) {
    if (!sparseKv || !context->isEnding || context->pastKeyValues.empty() ||
        !context->pastKeyValues.front().first.kvStorageOwner) return;
    // Retain the lease until all borrowed descriptors are detached. Forward
    // completion already synchronized the rank streams before publication.
    auto lease = std::move(context->pastKeyValues.front().first.kvStorageOwner);
    for (auto &pair : context->pastKeyValues) for (Data *root : {&pair.first, &pair.second}) {
        // The root only mirrors rank zero after generic prefix restoration.
        root->isPagedKVCache = false; root->pageIndex.clear(); root->pagedKVCacheData = nullptr;
        for (auto &entry : root->multiDeviceDatas) delete entry.second;
        root->multiDeviceDatas.clear(); root->multiDeviceData = false;
        root->dims.clear(); root->expansionDims.clear(); root->strides.clear();
        root->expansionSize = root->expansionBytes = 0;
    }
    tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
}

void NaiveN05FlashModel::InitSparseKv() {
#ifdef USE_CUDA
    if (sparseKv) return;
    const int owners = DecodeGraphBatchBuckets(maxBatch).back(), ranks = tpDevices.size();
    const int page = GetPageLen(), windowCapacity = (window + 127) / 128 * 128;
    const int indexPage = indexerContextSharding ? page / ranks : page;
    const int tokens = (kvMemConfig.maxTokens + page - 1) / page * page;
    const int pages = tokensLimit / page, unit = kvCacheDataType == FP8_E4M3 ? 1 : 2;
    auto state = std::make_shared<SparseKvState>();
    state->requests = owners; state->tokens = tokens; state->devices = tpDevices;
    const auto freeBytes = FastllmCudaGetFreeSizes(), totalBytes = FastllmCudaGetTotalSizes();
    auto prefixes = std::make_shared<PrefixPool>();
    prefixes->offsets.resize(block_cnt * 2);
    for (int layer = 0; layer < block_cnt; ++layer) if (slidingLayers[layer])
        for (int part = 0; part < 2; ++part) {
            prefixes->offsets[layer * 2 + part] = prefixes->slotBytes;
            prefixes->slotBytes += size_t(window) * std::max(1, sliding.kvHeads / ranks) *
                (part ? sliding.valueDim : sliding.headDim) * unit;
        }
    int64_t prefixBudget = INT64_MAX;
    for (int device : tpDevices) {
        const int64_t reserve = totalBytes.at(device) * (1.0 - GetGpuMemRatio());
        prefixBudget = std::min<int64_t>(prefixBudget, (int64_t)freeBytes.at(device) - reserve);
    }
    const int prefixSlots = prefixes->SlotCount(owners, prefixBudget);
    prefixes->storage.resize(ranks);
    tpPagedCaches.resize(ranks);
    size_t hostBytes = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        size_t deviceBytes = size_t(prefixSlots) * prefixes->slotBytes;
        for (int layer = 0; layer < block_cnt; ++layer) {
            const auto &cfg = slidingLayers[layer] ? sliding : full;
            const size_t width = std::max(1, cfg.kvHeads / ranks) * (cfg.headDim + cfg.valueDim) * unit;
            if (slidingLayers[layer]) deviceBytes += size_t(owners) * windowCapacity * width;
            else {
                // Sparse storage: hot KV, bidirectional maps, eviction workspace,
                // selected indices and metadata. Charge graph padding as owners.
                deviceBytes += size_t(owners) * (kvMemConfig.residentTokens * (width + 8) +
                    tokens * 4ULL + ((tokens / page + 3) / 4 * 4) * 8ULL + indexTopK * 8ULL + 128) +
                    size_t(owners) * (kvMemConfig.residentTokens * 4ULL + indexTopK * 4ULL + 32);
                if (HasLocalIndexer(layer, rank))
                    deviceBytes += size_t(pages) * indexPage * IndexerCacheWidth(kvCacheDataType) * unit;
            }
        }
        deviceBytes += size_t(tokens) * std::max(1, full.kvHeads / ranks) *
            (full.headDim + full.valueDim) * unit;
        const size_t reserve = totalBytes.at(tpDevices[rank]) * (1.0 - GetGpuMemRatio());
        if (freeBytes.at(tpDevices[rank]) <= reserve || deviceBytes > freeBytes.at(tpDevices[rank]) - reserve)
            throw std::invalid_argument("MiMo KVMem GPU budget insufficient for hot KV, Indexer, metadata and four SWA prefix checkpoints per request; reduce tokens/max_batch/max_tokens/resident_tokens or increase gpu_mem_ratio");
        FastllmCudaSetDevice(tpDevices[rank]);
        auto r = std::make_unique<SparseKvState::Rank>();
        r->layers.resize(block_cnt); r->indexers.resize(block_cnt); r->windows.resize(block_cnt);
        auto &managers = tpPagedCaches[rank];
        managers.resize(block_cnt * 2);
        auto allocate = [&](Data &data, int count, int width) {
            data.dataType = kvCacheDataType; data.UpdateUnitSize();
            data.Resize({owners, count, width});
            data.ToDevice(DataDevice::CUDA, {tpDevices[rank]}, false);
            data.Allocate(false);
        };
        size_t measured = 0;
        for (int layer = 0; layer < block_cnt; ++layer) {
            const auto &cfg = slidingLayers[layer] ? sliding : full;
            const int heads = std::max(1, cfg.kvHeads / ranks);
            if (slidingLayers[layer]) {
                allocate(r->windows[layer].first, windowCapacity, heads * cfg.headDim);
                allocate(r->windows[layer].second, windowCapacity, heads * cfg.valueDim);
                measured += r->windows[layer].first.GetBytes() + r->windows[layer].second.GetBytes();
                continue;
            }
            KvMemSparseConfig sc;
            sc.maxRequests = sc.maxBatch = owners; sc.maxTokens = tokens; sc.pageLen = page;
            sc.totalTokens = tokensLimit;
            sc.residentTokens = kvMemConfig.residentTokens; sc.maxSelected = indexTopK;
            sc.keyRowBytes = heads * cfg.headDim * unit; sc.valueRowBytes = heads * cfg.valueDim * unit;
            sc.hostBudgetBytes = size_t(tokensLimit) * (sc.keyRowBytes + sc.valueRowBytes);
            r->layers[layer] = std::make_unique<KvMemSparseCache>(sc);
            hostBytes += r->layers[layer]->HostBytes();
            measured += r->layers[layer]->DeviceBytes();
            for (int part = 0; part < 2; ++part) {
                auto manager = std::make_unique<PagedCacheManager>();
                manager->type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
                manager->dataType = kvCacheDataType; manager->UpdateUnitSize();
                manager->dataDevice = DataDevice::CUDA; manager->dataDeviceIds = {tpDevices[rank]};
                manager->pageLen = page;
                manager->Resize({pages, page, 1, heads * (part ? cfg.valueDim : cfg.headDim)});
                manager->cudaData = part ? r->layers[layer]->DeviceHostValue(0) : r->layers[layer]->DeviceHostKey(0);
                manager->cudaDataBorrowed = true;
                manager->SetMaxPages(pages);
                managers[layer * 2 + part] = std::move(manager);
            }
            if (HasLocalIndexer(layer, rank)) {
                auto index = std::make_unique<PagedCacheManager>();
                index->pageLen = indexPage; index->maxPages = pages;
                index->kvCacheFp8Indexer = fp8IndexerCache;
                index->dataType = kvCacheDataType; index->UpdateUnitSize();
                index->Resize({pages, indexPage, 1, IndexerCacheWidth(kvCacheDataType)});
                index->ToDevice(DataDevice::CUDA, {tpDevices[rank]}, false); index->Allocate(false);
                measured += index->GetBytes();
                state->indexSidecars.emplace(managers[layer * 2].get(), index.get());
                r->indexers[layer] = std::move(index);
            }
        }
        for (int part = 0; part < 2; ++part) {
            Data &data = part ? r->stageValue : r->stageKey;
            data.dataType = kvCacheDataType; data.UpdateUnitSize();
            data.Resize({1, tokens, std::max(1, full.kvHeads / ranks) * (part ? full.valueDim : full.headDim)});
            data.ToDevice(DataDevice::CUDA, {tpDevices[rank]}, false); data.Allocate(false);
            measured += data.GetBytes();
        }
        if (prefixSlots) {
            auto &data = prefixes->storage[rank];
            data.dataType = INT8; data.UpdateUnitSize();
            data.Resize({prefixSlots, (int)prefixes->slotBytes});
            data.ToDevice(DataDevice::CUDA, {tpDevices[rank]}, false); data.Allocate(false);
            measured += data.GetBytes();
        }
        printf("[MiMo KVMem] GPU %d: allocated hot KV/Indexer/metadata/staging %.2f MiB\n",
            tpDevices[rank], measured / 1048576.0);
        state->ranks.push_back(std::move(r));
    }
    for (int slot = owners - 1; slot >= 0; --slot) state->free.push_back(slot);
    sparseKv = std::move(state);
    if (prefixSlots) {
        for (int slot = 0; slot < prefixSlots; ++slot) prefixes->freeSlots.push_back(slot);
        prefixPool = std::move(prefixes);
        printf("[MiMo KVMem prefix cache] %d window slots, %d reserved per request, %.2f MiB/GPU within KV budget.\n",
            prefixSlots, PrefixSnapshotsPerRequest, prefixPool->slotBytes * prefixSlots / 1048576.0);
    }
    tpSwaPrefillWorkspaces.resize(ranks);
    for (int rank = 0; rank < ranks; ++rank) PrepareSwaPrefillWorkspace(rank, GetChunkedPrefillSize());
    kvMemLocked = true;
    printf("[MiMo KVMem] max_batch=%d owners=%d max_tokens=%d total_tokens=%d resident=%d host=%.2f GiB (budget %.2f GiB). Indexer stays on GPU.\n",
        maxBatch, owners, kvMemConfig.maxTokens, tokensLimit, kvMemConfig.residentTokens,
        hostBytes / 1073741824.0, kvMemConfig.hostBytes / 1073741824.0);
    fflush(stdout);
#endif
}

void NaiveN05FlashModel::AttachSparseKv(const std::vector<std::pair<Data *, Data *>> &roots) {
#ifdef USE_CUDA
    if (!sparseKv) return;
    auto lease = std::static_pointer_cast<SparseKvState::Lease>(roots.front().first->kvStorageOwner);
    if (lease) return;
    {
        lease = std::make_shared<SparseKvState::Lease>();
        lease->storage = sparseKv;
        std::lock_guard<std::mutex> lock(sparseKv->mutex);
        if (sparseKv->free.empty()) { lease->storage.reset(); throw std::runtime_error("MiMo KVMem request slots exhausted"); }
        lease->slot = sparseKv->free.back(); sparseKv->free.pop_back();
        roots.front().first->kvStorageOwner = lease;
    }
    const int slot = lease->slot, ranks = tpDevices.size();
    std::shared_ptr<PrefixSnapshot> snapshot;
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        auto active = activePrefixes.find(roots.front().first);
        if (active != activePrefixes.end() && !active->second.prefill.empty())
            snapshot = active->second.prefill.back();
    }
    const int windowCapacity = (window + 127) / 128 * 128;
    // Descriptor maps are scheduler-owned, before starting rank workers.
    for (int layer = 0; layer < block_cnt; ++layer)
    for (Data *root : {roots[layer].first, roots[layer].second}) {
        root->multiDeviceData = true; root->dataDevice = DataDevice::CUDA;
        root->dataDeviceIds = tpDevices; root->isKVCache = true;
        for (int device : tpDevices)
            if (!root->multiDeviceDatas.count(device)) root->multiDeviceDatas[device] = new Data(kvCacheDataType);
    }
    std::vector<std::exception_ptr> errors(ranks);
    tpWorkers.Run(tpDevices, [&](int rank) {
        FastllmCudaSetDevice(tpDevices[rank]);
        auto &r = *sparseKv->ranks[rank];
        for (int layer = 0; layer < block_cnt; ++layer) {
            const auto &cfg = slidingLayers[layer] ? sliding : full;
            const int capacity = slidingLayers[layer] ? windowCapacity : kvMemConfig.maxTokens;
            if (r.layers[layer]) r.layers[layer]->Reset(slot, cudaStreamPerThread);
            for (int part = 0; part < 2; ++part) {
                Data &root = *(part ? roots[layer].second : roots[layer].first);
                Data &local = *root.multiDeviceDatas.at(tpDevices[rank]);
                const int width = std::max(1, cfg.kvHeads / ranks) * (part ? cfg.valueDim : cfg.headDim);
                const int past = slidingLayers[layer] && snapshot
                    ? std::min(snapshot->length, window - 1) : (root.dims.empty() ? 0 : root.dims[1]);
                {
                    // Restored full pages already carry references. Preserve them.
                    if (!local.isPagedKVCache) local.FreeSpace();
                    local.isFake = false;
                    local.dataType = kvCacheDataType; local.UpdateUnitSize();
                    local.dataDevice = DataDevice::CUDA; local.dataDeviceIds = {tpDevices[rank]};
                    local.expansionDims = {1, capacity, width};
                    local.strides = {size_t(capacity) * width, (size_t)width, 1};
                    local.Resize({1, past, width});
                    local.cudaDataBorrowed = true; local.kvCacheFp8Indexer = false;
                    if (slidingLayers[layer]) {
                        Data &storage = part ? r.windows[layer].second : r.windows[layer].first;
                        local.cudaData = (char *)storage.cudaData + size_t(slot) * windowCapacity * width * local.unitSize;
                        const size_t rowBytes = size_t(width) * local.unitSize;
                        if (snapshot) {
                            auto *source = (char *)snapshot->pool->storage.at(rank).cudaData +
                                size_t(snapshot->slot) * snapshot->pool->slotBytes + snapshot->pool->offsets[layer * 2 + part];
                            const int first = tpSwaRingCache ? (snapshot->length - past) % window : 0;
                            const int count = tpSwaRingCache ? std::min(past, window - first) : past;
                            FastllmCudaCopyFromDeviceToDevice((char *)local.cudaData + first * rowBytes, source, count * rowBytes);
                            if (count < past) FastllmCudaCopyFromDeviceToDevice(local.cudaData,
                                source + count * rowBytes, (past - count) * rowBytes);
                        } else if (past) cudaMemsetAsync(local.cudaData, 0, size_t(past) * rowBytes, cudaStreamPerThread);
                    } else {
                        local.isKVCache = local.isPagedKVCache = true;
                        local.pagedKVCacheData = tpPagedCaches[rank][layer * 2 + part].get();
                        local.pageLen = GetPageLen();
                        local.cudaData = part ? r.layers[layer]->ValuePages(slot) : r.layers[layer]->KeyPages(slot);
                        // Startup graphs use a tiny private prefix, never request data.
                        if (tpServingState && tpServingState->preparing) {
                            local.Resize({1, 0, width});
                            PreparePagedCache(local, *local.pagedKVCacheData, past + 1);
                            for (int page : local.pageIndex) {
                                size_t bytes = size_t(local.pageLen) * width * local.unitSize;
                                cudaMemsetAsync((char *)local.pagedKVCacheData->cudaData + page * bytes, 0, bytes, cudaStreamPerThread);
                                if (!part && r.indexers[layer]) {
                                    auto &index = *r.indexers[layer];
                                    bytes = size_t(index.dims[1]) * index.dims[3] * index.unitSize;
                                    cudaMemsetAsync((char *)index.cudaData + page * bytes, 0, bytes, cudaStreamPerThread);
                                }
                            }
                            local.Resize({1, past, width});
                        }
                    }
                }
                if (!slidingLayers[layer] && !local.pageIndex.empty())
                    FastllmCudaCopyFromHostToDevice(local.cudaData, local.pageIndex.data(), local.pageIndex.size() * sizeof(int));
            }
        }
        FastllmCudaSyncCurrentThreadStream();
    }, errors);
    for (auto error : errors) if (error) std::rethrow_exception(error);
    for (int layer = 0; layer < block_cnt; ++layer) {
        const auto &cfg = slidingLayers[layer] ? sliding : full;
        for (int part = 0; part < 2; ++part) {
            Data &root = *(part ? roots[layer].second : roots[layer].first);
            const int past = slidingLayers[layer] && snapshot
                ? std::min(snapshot->length, window - 1) : (root.dims.empty() ? 0 : root.dims[1]);
            const int width = cfg.kvHeads * (part ? cfg.valueDim : cfg.headDim);
            root.kvCacheFp8Indexer = false;
            root.expansionDims = {1, slidingLayers[layer] ? windowCapacity : kvMemConfig.maxTokens, width};
            root.strides = {size_t(root.expansionDims[1]) * width, (size_t)width, 1};
            root.Resize({1, past, width});
        }
    }
    tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
#endif
}

Data NaiveN05FlashModel::SparseIndexer(int layer, int rank, const Data &key) const {
    Data result;
#ifdef USE_CUDA
    if (!HasLocalIndexer(layer, rank)) return result;
    auto &storage = *sparseKv->ranks[rank]->indexers[layer];
    const int width = storage.dims[3];
    result.FakeFrom(key, 0);
    const int page = storage.dims[1];
    const int length = indexerContextSharding
        ? (key.dims[1] / key.pageLen) * page + std::clamp(key.dims[1] % key.pageLen - rank * page, 0, page)
        : key.dims[1];
    result.Resize({1, length, width});
    result.isPagedKVCache = true;
    result.pageLen = page;
    result.pagedKVCacheData = &storage;
    // This view borrows K's device table, without copying its CPU page vector
    // on every decode step. Bound writes by the pages actually attached.
    const int capacity = key.pageIndex.size() * result.pageLen;
    result.expansionDims = {1, capacity, width};
    result.strides = {size_t(capacity) * width, (size_t)width, 1};
    result.kvCacheFp8Indexer = fp8IndexerCache;
#endif
    return result;
}

void NaiveN05FlashModel::StageSparseKv(int rank, int layer, const Data &key, const Data &value, Data &outKey, Data &outValue) {
#ifdef USE_CUDA
    auto &r = *sparseKv->ranks[rank];
    outKey.Resize(key.dims); outKey.FakeFrom(r.stageKey, 0);
    outValue.Resize(value.dims); outValue.FakeFrom(r.stageValue, 0);
    r.layers[layer]->Gather((const int *)key.cudaData, (const int *)value.cudaData, key.dims[1],
        outKey.cudaData, outValue.cudaData, cudaStreamPerThread);
#endif
}
}
