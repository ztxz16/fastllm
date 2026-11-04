#include "models/naive_n05_flash.h"
#include "pd.h"
#include <algorithm>
#include <stdexcept>
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace fastllm {
namespace {
void CheckPD(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(std::string("FastKV Naive: ") + message);
}
} // namespace

std::string NaiveN05FlashModel::PDLayout() const {
    CheckPD(!sparseKv && !indexerContextSharding, "sparse/offloaded KV needs a separate state adapter");
    CheckPD(!draftTP, "draft TP state transfer is not implemented");
    return "naive-n05-state-v1:" + std::to_string(block_cnt) + ":" +
           std::to_string(std::max(size_t(1), tpDevices.size())) + ":" + std::to_string(dataType) + ":" +
           std::to_string(kvCacheDataType) + ":" + std::to_string(fp8IndexerCache) + ":" +
           std::to_string(indexerLayerSharding) + ":" + std::to_string(draftEnabled);
}

fastkv::Message NaiveN05FlashModel::ExportPDState(ResponseContext &context) {
#ifndef USE_CUDA
    throw std::runtime_error("FastKV Naive requires CUDA");
#else
    CheckPD(context.preTokens > 0, "empty prefill state");
    const auto devices = tpDevices.empty() ? std::vector<int>{FastllmCudaGetDevice()} : tpDevices;
    const bool tp = tpDevices.size() > 1;
    PDStateWriter writer;
    writer.metadata.String(PDLayout());
    writer.metadata.U32(context.preTokens);
    for (const auto &pair : context.pastKeyValues)
        for (const Data *root : {&pair.first, &pair.second})
            writer.metadata.Ints(root->dims);
    std::vector<PDStateWriter> ranks(devices.size());
    auto pack = [&](int rank) {
        FastllmCudaSetDevice(devices[rank]);
        auto &output = ranks[rank];
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (int part = 0; part < 2; ++part) {
                const Data &root =
                    part ? context.pastKeyValues[layer].second : context.pastKeyValues[layer].first;
                const Data &cache = tp ? *root.multiDeviceDatas.at(devices[rank]) : root;
                CheckPD(cache.dims.size() == 3 && cache.dims[0] == 1 && cache.dataDevice == CUDA,
                        "invalid local cache");
                const int rows =
                    slidingLayers[layer] ? std::min(context.preTokens, window - 1) : context.preTokens;
                const int width = cache.dims[2];
                CheckPD(cache.dims[1] == rows, "cache length mismatch");
                Data compact(cache.dataType, {1, rows, width});
                compact.dataDevice = CUDA;
                compact.dataDeviceIds = {devices[rank]};
                compact.Allocate();
                compact.kvCacheFp8Indexer = cache.kvCacheFp8Indexer;
                const size_t rowBytes = size_t(width) * cache.unitSize;
                if (cache.isPagedKVCache) {
                    CheckPD(cache.pagedKVCacheData && cache.pageLen > 0, "missing page owner");
                    const auto &pool = *cache.pagedKVCacheData;
                    for (int at = 0; at < rows; at += cache.pageLen) {
                        const int page = cache.pageIndex.at(at / cache.pageLen);
                        CheckPD(page >= 0 && page < pool.maxPages, "invalid physical page");
                        const size_t bytes = size_t(std::min(cache.pageLen, rows - at)) * rowBytes;
                        FastllmCudaCopyFromDeviceToDevice(
                            (char *)compact.cudaData + size_t(at) * rowBytes,
                            (char *)pool.cudaData + size_t(page) * cache.pageLen * rowBytes, bytes);
                    }
                } else {
                    const bool ring = tp && tpSwaRingCache && slidingLayers[layer];
                    const int first = ring ? (context.preTokens - rows) % window : 0;
                    const int count = ring ? std::min(rows, window - first) : rows;
                    FastllmCudaCopyFromDeviceToDevice(compact.cudaData,
                                                      (char *)cache.cudaData + size_t(first) * rowBytes,
                                                      size_t(count) * rowBytes);
                    if (count < rows)
                        FastllmCudaCopyFromDeviceToDevice((char *)compact.cudaData + size_t(count) * rowBytes,
                                                          cache.cudaData, size_t(rows - count) * rowBytes);
                }
                FastllmCudaSyncCurrentThreadStream();
                output.Tensor(compact);
            }
        }
    };
    if (tp) {
        std::vector<std::exception_ptr> errors(devices.size());
        tpWorkers.Run(devices, pack, errors);
        for (auto error : errors)
            if (error)
                std::rethrow_exception(error);
    } else
        pack(0);
    for (auto &rank : ranks) {
        writer.metadata.Blob(rank.metadata.bytes.data(), rank.metadata.bytes.size());
        writer.metadata.U32(rank.buffers.size());
        writer.buffers.insert(writer.buffers.end(), rank.buffers.begin(), rank.buffers.end());
    }
    writer.metadata.U32(draftEnabled);
    if (draftEnabled) {
        auto it = draftContexts.find(&context.pastKeyValues);
        CheckPD(it != draftContexts.end() && it->second->pending.empty(), "missing committed draft state");
        const auto &draft = *it->second;
        writer.metadata.U32(draft.committed);
        writer.metadata.U32(draft.kv.size());
        for (const auto &pair : draft.kv) {
            writer.Tensor(pair.first);
            writer.Tensor(pair.second);
        }
    }
    return writer.Finish();
#endif
}

void NaiveN05FlashModel::ImportPDState(ResponseContext &context, const fastkv::Message &message) {
#ifndef USE_CUDA
    throw std::runtime_error("FastKV Naive requires CUDA");
#else
    fastkv::Reader metadata(message.metadata);
    CheckPD(metadata.String() == PDLayout(), "incompatible state layout or TP size");
    const int length = metadata.U32();
    CheckPD(length > 0 && length == int(context.allTokens.size()) - 1, "prompt length mismatch");
    const auto devices = tpDevices.empty() ? std::vector<int>{FastllmCudaGetDevice()} : tpDevices;
    const bool tp = tpDevices.size() > 1;
    for (int layer = 0; layer < block_cnt; ++layer) {
        for (int part = 0; part < 2; ++part) {
            auto shape = metadata.Ints(3);
            const int rows = slidingLayers[layer] ? std::min(length, window - 1) : length;
            CheckPD(shape.size() == 3 && shape[0] == 1 && shape[1] == rows && shape[2] > 0,
                    "invalid root shape");
            Data &root = part ? context.pastKeyValues[layer].second : context.pastKeyValues[layer].first;
            root.dataDevice = CUDA;
            root.dataDeviceIds = devices;
            root.multiDeviceData = tp;
            root.Resize(shape);
            root.expansionDims = shape;
            if (tp)
                for (int device : devices) {
                    Data *&local = root.multiDeviceDatas[device];
                    if (local) {
                        CheckPD(local->pageIndex.empty(), "import requires an empty request cache");
                        delete local;
                    }
                    local = new Data(kvCacheDataType);
                    local->dataDevice = CUDA;
                    local->dataDeviceIds = {device};
                    local->isKVCache = true;
                }
        }
    }
    size_t offset = 0;
    std::vector<fastkv::Message> rankStates(devices.size());
    for (auto &rank : rankStates) {
        rank.metadata = metadata.Blob();
        const size_t count = metadata.U32();
        CheckPD(count <= message.buffers.size() - offset, "invalid rank payload count");
        rank.buffers.assign(message.buffers.begin() + offset, message.buffers.begin() + offset + count);
        offset += count;
    }
    auto unpack = [&](int rank) {
        FastllmCudaSetDevice(devices[rank]);
        PDStateReader reader(rankStates[rank]);
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (int part = 0; part < 2; ++part) {
                Data value;
                reader.Tensor(value);
                const int rows = slidingLayers[layer] ? std::min(length, window - 1) : length;
                const auto &cfg = slidingLayers[layer] ? sliding : full;
                const int expected =
                    std::max(1, cfg.kvHeads / int(devices.size())) * (part ? cfg.valueDim : cfg.headDim) +
                    (!part && !slidingLayers[layer]
                         ? (tp ? LocalIndexerCacheWidth(layer, rank) : PackedIndexerWidth())
                         : 0);
                CheckPD(value.dataType == kvCacheDataType &&
                            value.dims == std::vector<int>({1, rows, expected}),
                        "local KV shape mismatch");
                Data &root = part ? context.pastKeyValues[layer].second : context.pastKeyValues[layer].first;
                Data &cache = tp ? *root.multiDeviceDatas.at(devices[rank]) : root;
                cache.dataType = value.dataType;
                cache.UpdateUnitSize();
                cache.kvCacheFp8Indexer = value.kvCacheFp8Indexer;
                const int width = expected;
                const size_t rowBytes = size_t(width) * cache.unitSize;
                // H2D is submitted by the persistent owner worker. Network
                // workers never mutate this request's live page table.
                value.ToDevice(CUDA, std::vector<int>{devices[rank]});
                if (tp && !slidingLayers[layer] && !tpPagedCaches.empty()) {
                    PreparePagedCache(cache, *tpPagedCaches[rank][layer * 2 + part], rows);
                    for (int at = 0; at < rows; at += cache.pageLen) {
                        const int page = cache.pageIndex.at(at / cache.pageLen);
                        FastllmCudaCopyFromDeviceToDevice((char *)cache.pagedKVCacheData->cudaData +
                                                              size_t(page) * cache.pageLen * rowBytes,
                                                          (char *)value.cudaData + size_t(at) * rowBytes,
                                                          size_t(std::min(cache.pageLen, rows - at)) *
                                                              rowBytes);
                    }
                    cache.lastPageLen = (rows - 1) % cache.pageLen + 1;
                } else {
                    const bool ring = tp && tpSwaRingCache && slidingLayers[layer];
                    const int capacity = (std::max(rows, ring ? window : rows) + 127) / 128 * 128;
                    if (!tp) {
                        cache.dims.clear();
                        cache.expansionDims.clear();
                    }
                    ReserveCache(cache, capacity, width);
                    const int first = ring ? (length - rows) % window : 0;
                    const int count = ring ? std::min(rows, window - first) : rows;
                    FastllmCudaCopyFromDeviceToDevice((char *)cache.cudaData + size_t(first) * rowBytes,
                                                      value.cudaData, size_t(count) * rowBytes);
                    if (count < rows)
                        FastllmCudaCopyFromDeviceToDevice(cache.cudaData,
                                                          (char *)value.cudaData + size_t(count) * rowBytes,
                                                          size_t(rows - count) * rowBytes);
                }
                cache.Resize({1, rows, width});
                cache.isKVCache = true;
            }
        }
        FastllmCudaSyncCurrentThreadStream();
        reader.End();
    };
    if (tp) {
        std::vector<std::exception_ptr> errors(devices.size());
        tpWorkers.Run(devices, unpack, errors);
        for (auto error : errors)
            if (error)
                std::rethrow_exception(error);
    } else
        unpack(0);
    const bool hasDraft = metadata.U32();
    CheckPD(hasDraft == draftEnabled, "draft mode mismatch");
    if (hasDraft) {
        auto draft = std::make_shared<DraftContext>();
        draft->committed = metadata.U32();
        const size_t layers = metadata.U32();
        CheckPD(layers == size_t(draftLayers) && draft->committed == length, "invalid draft prefix");
        draft->kv.resize(layers);
        // Remaining metadata and payload belong to the draft adapter.
        fastkv::Message rest;
        const size_t consumed = message.metadata.size() - metadata.Remaining();
        rest.metadata.assign(message.metadata.begin() + consumed, message.metadata.end());
        rest.buffers.assign(message.buffers.begin() + offset, message.buffers.end());
        PDStateReader reader(rest);
        for (auto &pair : draft->kv) {
            reader.Tensor(pair.first);
            reader.Tensor(pair.second);
        }
        reader.End();
        draftContexts[&context.pastKeyValues] = std::move(draft);
    } else {
        metadata.End();
        CheckPD(offset == message.buffers.size(), "unclaimed state payload");
    }
    tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
#endif
}
} // namespace fastllm
