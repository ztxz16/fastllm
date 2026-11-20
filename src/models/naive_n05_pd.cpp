#include "models/naive_n05_flash.h"
#include "pd.h"
#include "naive_n05_kvmem.h"
#include "fastkv/host_cache.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#ifdef USE_CUDA
#include "naive_n05_tp.h"
#include "naive_n05_parallel.h"
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace fastllm {
namespace {
void CheckPD(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(std::string("FastKV Naive: ") + message);
}
#ifdef USE_CUDA
struct RestorePDDevice {
    int previous = FastllmCudaGetDevice();
    ~RestorePDDevice() { FastllmCudaSetDevice(previous); }
};

void SyncImportedDPCapacity(Data &root) {
    // The wire shape only describes valid tokens. The metadata-only DP root
    // can grow as far as every owning shard allows; physical pages are still
    // reserved separately before each decode.
    int rows = std::numeric_limits<int>::max();
    for (const auto &entry : root.multiDeviceDatas)
        rows = std::min(rows, entry.second->expansionDims.at(1));
    root.expansionDims = root.dims;
    root.expansionDims[1] = rows;
}

template <class Task>
void RunPDWorkers(parallel::CudaRuntime *runtime, PersistentWorkerGroup &workers,
        const std::vector<int> &devices, const Task &task) {
    if (runtime) {
        // A request owns only one group. Reuse the full runtime rather than
        // recreating its persistent GPU workers for that subset.
        runtime->Run([&](int, int device) {
            auto it = std::find(devices.begin(), devices.end(), device);
            if (it != devices.end()) task(it - devices.begin());
        });
    } else if (devices.size() > 1) {
        std::vector<std::exception_ptr> errors(devices.size());
        workers.Run(devices, task, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
    } else {
        task(0);
    }
}

// Submit all independent ranges on the owner worker's stream. Adjacent
// physical pages coalesce; the existing generic batch-copy kernel handles
// the remaining fragmented spans without one CUDA API call per page.
struct StateCopies {
    std::vector<void *> dst;
    std::vector<const void *> src;
    std::vector<size_t> bytes;
    std::vector<void *> dst2;
    std::vector<const void *> src2;
    std::vector<size_t> dp, sp, widths, heights;
    void Rows(void *to, size_t toPitch, const void *from, size_t fromPitch, size_t width, size_t rows) {
        if (!width || !rows) return;
        if (toPitch == width && fromPitch == width) { Add(to, from, width * rows); return; }
        dst2.push_back(to); src2.push_back(from); dp.push_back(toPitch); sp.push_back(fromPitch);
        widths.push_back(width); heights.push_back(rows);
    }
    void Add(void *to, const void *from, size_t n) {
        if (!n || to == from) return;
        if (!bytes.empty() && (char *)dst.back() + bytes.back() == to &&
            (const char *)src.back() + bytes.back() == from) bytes.back() += n;
        else { dst.push_back(to); src.push_back(from); bytes.push_back(n); }
    }
    void Submit() {
        const bool copied = FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
            dst.data(), src.data(), bytes.data(), bytes.size());
        const bool strided = FastllmCudaBatchCopy2DAsyncCurrentThread(dst2.data(), src2.data(), dp.data(), sp.data(),
            widths.data(), heights.data(), widths.size());
        // Even an unsuccessful partial submission must finish before borrowed
        // transport buffers can be released.
        FastllmCudaSyncCurrentThreadStream();
        CheckPD(copied && strided && !FastllmCudaGetThreadError(), "state copy failed");
    }
};
// Indexer sidecars share K's physical page IDs, but have their own row width.
void CopyPagedRows(Data &compact, const Data &key, const PagedCacheManager &pool, bool restore, StateCopies &copies, int column = 0) {
    CheckPD(key.isPagedKVCache && key.pageLen > 0 && pool.pageLen == key.pageLen &&
                pool.dataType == compact.dataType && column >= 0 && pool.dims.back() >= column + compact.dims[2],
            "incompatible page storage");
    const int rows = compact.dims[1];
    const size_t rowBytes = GetDataBytes(compact.dataType, 1, compact.dims[2]);
    const size_t pitch = GetDataBytes(pool.dataType, 1, pool.dims.back());
    const size_t offset = GetDataBytes(pool.dataType, 1, column);
    for (int at = 0; at < rows; at += key.pageLen) {
        const int page = key.pageIndex.at(at / key.pageLen);
        CheckPD(page >= 0 && page < pool.maxPages, "invalid physical page");
        void *dense = (char *)compact.cudaData + size_t(at) * rowBytes;
        void *paged = (char *)pool.cudaData + size_t(page) * key.pageLen * pitch + offset;
        copies.Rows(restore ? paged : dense, restore ? pitch : rowBytes,
                    restore ? dense : paged, restore ? rowBytes : pitch, rowBytes,
                    std::min(key.pageLen, rows - at));
    }
}
#endif
} // namespace

#ifdef USE_CUDA
struct NaiveN05FlashModel::SparsePDReceive : PDReceiveState {
    struct Reservation {
        std::shared_ptr<SparseKvState> storage;
        ResponseContext state;
    };
    NaiveN05FlashModel &model;
    std::shared_ptr<Reservation> reservation = std::make_shared<Reservation>();
    std::vector<std::vector<fastkv::Buffer>> targets;
    std::vector<std::vector<fastkv::Buffer>> windows;
    fastkv::Bytes expected;
    int length;
    bool received = false, committed = false;

    SparsePDReceive(NaiveN05FlashModel &model, int length) : model(model), length(length) {
        reservation->storage = model.sparseKv;
        auto &state = reservation->state;
        state.Init(model.block_cnt, model.dataType, model.kvCacheDataType);
        const int ranks = model.tpDevices.size(), rows = std::min(length, model.window - 1);
        fastkv::Writer metadata;
        metadata.String(model.PDLayout()); metadata.U32(length);
        for (int layer = 0; layer < model.block_cnt; ++layer) {
            const bool swa = model.slidingLayers[layer];
            const auto &cfg = swa ? model.sliding : model.full;
            for (int part = 0; part < 2; ++part) {
                const int width = cfg.kvHeads * (part ? cfg.valueDim : cfg.headDim);
                auto &root = part ? state.pastKeyValues[layer].second : state.pastKeyValues[layer].first;
                root.dataType = model.LayerKVType(layer); root.UpdateUnitSize();
                root.Resize({1, swa ? rows : length, width});
                root.multiDeviceData = true;
                // The source's root metadata describes its packed GPU layout.
                metadata.Ints({1, swa ? rows : length, width + (!swa && !part &&
                    model.kvCacheDataType != FP4_E2M1 ? model.IndexerCacheWidth(model.AuxiliaryKVType()) : 0)});
                for (int device : model.tpDevices) root.multiDeviceDatas[device] = new Data(root.dataType);
            }
        }
        windows.resize(ranks);
        for (int rank = 0; rank < ranks; ++rank) {
            const int device = model.tpDevices[rank];
            FastllmCudaSetDevice(device);
            auto &storage = *model.sparseKv->ranks[rank];
            storage.pdKeys.resize(model.block_cnt); storage.pdValues.resize(model.block_cnt);
            storage.pdIndexers.resize(model.block_cnt);
            fastkv::Writer descriptors;
            const size_t begin = targets.size();
            size_t windowBytes = 0;
            for (int layer = 0; layer < model.block_cnt; ++layer) if (model.slidingLayers[layer])
                windowBytes += size_t(rows) * std::max(1, model.sliding.kvHeads / ranks) *
                    (model.sliding.headDim + model.sliding.valueDim) *
                    (model.AuxiliaryKVType() == FP8_E4M3 ? 1 : 2);
            fastkv::Buffer arena;
            if (windowBytes) arena = model.PDExportPool() ? model.PDExportPool()->Acquire(windowBytes, device)
                                                         : fastkv::Buffer::Allocate(windowBytes, device);
            size_t windowOffset = 0;
            windows[rank].resize(model.block_cnt * 2);
            auto add = [&](DataType type, int count, int width, bool fp8, std::vector<fastkv::Buffer> spans) {
                descriptors.U32(type); descriptors.Ints({1, count, width}); descriptors.U32(fp8);
                descriptors.U64(GetDataBytes(type, count, width)); targets.push_back(std::move(spans));
            };
            auto pages = [&](const Data &key, const fastkv::Buffer &base, size_t rowBytes) {
                std::vector<fastkv::Buffer> spans;
                // Coalesce adjacent physical pages; retain fragmented spans.
                int first = 0;
                while (first < length) {
                    const int page = key.pageIndex.at(first / key.pageLen);
                    int count = std::min(key.pageLen, length - first);
                    while (first + count < length && count % key.pageLen == 0 &&
                           key.pageIndex.at((first + count) / key.pageLen) == page + count / key.pageLen)
                        count += std::min(key.pageLen, length - first - count);
                    spans.push_back(base.Slice(size_t(page) * key.pageLen * rowBytes, size_t(count) * rowBytes));
                    first += count;
                }
                return spans;
            };
            for (int layer = 0; layer < model.block_cnt; ++layer) {
                const bool swa = model.slidingLayers[layer];
                const auto &cfg = swa ? model.sliding : model.full;
                for (int part = 0; part < 2; ++part) {
                    auto &root = part ? state.pastKeyValues[layer].second : state.pastKeyValues[layer].first;
                    auto &local = *root.multiDeviceDatas.at(device);
                    const int width = std::max(1, cfg.kvHeads / ranks) * (part ? cfg.valueDim : cfg.headDim);
                    const auto type = model.LayerKVType(layer);
                    local.Resize({1, swa ? rows : length, width});
                    const size_t rowBytes = GetDataBytes(type, 1, width);
                    if (swa) {
                        auto view = arena.Slice(windowOffset, rows * rowBytes); windowOffset += view.Size();
                        windows[rank][layer * 2 + part] = view;
                        add(type, rows, width, false, {view});
                        continue;
                    }
                    auto &manager = *model.tpPagedCaches[rank][layer * 2 + part];
                    local.isKVCache = local.isPagedKVCache = true;
                    local.pagedKVCacheData = &manager; local.pageLen = manager.pageLen;
                    const int pageCount = (length + manager.pageLen - 1) / manager.pageLen;
                    { std::lock_guard<std::mutex> lock(manager.pageIndexLocker);
                      if (manager.FreePageCount() < pageCount) throw fastkv::CapacityBusy("KVMem receive pages exhausted"); }
                    local.pageIndex.reserve(pageCount);
                    for (int at = 0; at < length; at += manager.pageLen) {
                        const int page = manager.GetUnusedPageIndex(true);
                        CheckPD(page >= 0, "KVMem receive pages exhausted");
                        local.pageIndex.push_back(page);
                    }
                    // These pages are private and uninitialized: choose logical
                    // order by physical address to maximize direct RDMA spans.
                    std::sort(local.pageIndex.begin(), local.pageIndex.end());
                    local.lastPageLen = (length - 1) % local.pageLen + 1;
                    auto &base = part ? storage.pdValues[layer] : storage.pdKeys[layer];
                    if (!base.Size()) base = fastkv::Buffer::Borrow(
                        part ? storage.layers[layer]->HostValue(0) : storage.layers[layer]->HostKey(0),
                        size_t(manager.maxPages) * manager.pageLen * rowBytes,
                        fastkv::MemoryKind::Host, -1, storage.layers[layer]);
                    add(type, length, width, false, pages(local, base, rowBytes));
                }
                if (model.HasLocalIndexer(layer, rank)) {
                    auto &index = *storage.indexers[layer];
                    auto &base = storage.pdIndexers[layer];
                    if (!base.Size()) base = fastkv::Buffer::Borrow(index.cudaData, index.GetBytes(),
                        fastkv::MemoryKind::Cuda, device, storage.indexers[layer]);
                    const auto &key = *state.pastKeyValues[layer].first.multiDeviceDatas.at(device);
                    add(index.dataType, length, index.dims.back(), model.fp8IndexerCache,
                        pages(key, base, GetDataBytes(index.dataType, 1, index.dims.back())));
                }
            }
            metadata.Blob(descriptors.bytes.data(), descriptors.bytes.size());
            metadata.U32(targets.size() - begin);
        }
        metadata.U32(0); expected = std::move(metadata.bytes);
    }
    std::vector<std::vector<fastkv::Buffer>> Targets(const fastkv::Bytes &metadata,
                                                   const std::vector<size_t> &sizes) override {
        CheckPD(!received && sizes.size() == targets.size(), "invalid KVMem receive payload count");
        fastkv::Message wire; wire.metadata = metadata; wire.buffers.resize(sizes.size());
        CheckPD(PDStatePayload(wire).metadata == expected, "incompatible KVMem state metadata");
        (void)PDReceivePlacement(metadata, sizes.size(), model.tpDevices);
        size_t index = 0;
        auto result = targets;
        for (auto &spans : result) {
            size_t bytes = 0;
            for (auto &span : spans) { bytes += span.Size(); span = span.WithLease(reservation); }
            CheckPD(bytes == sizes[index++], "invalid KVMem state bytes");
        }
        received = true;
        return result;
    }
    void Commit(ResponseContext &context) override {
        CheckPD(received && !committed, "KVMem transfer not ready");
        for (const auto &pair : context.pastKeyValues)
            CheckPD(pair.first.dims.empty() && pair.second.dims.empty(), "KVMem import needs an empty context");
        CheckPD(context.pastKeyValues.size() == reservation->state.pastKeyValues.size(), "invalid KVMem context");
        // Request identities and prefix bookkeeping use root descriptor
        // addresses. Move rank ownership without replacing that root vector.
        for (int layer = 0; layer < model.block_cnt; ++layer) for (int part = 0; part < 2; ++part) {
            auto &source = part ? reservation->state.pastKeyValues[layer].second : reservation->state.pastKeyValues[layer].first;
            auto &target = part ? context.pastKeyValues[layer].second : context.pastKeyValues[layer].first;
            target.dataType = source.dataType; target.UpdateUnitSize(); target.Resize(source.dims);
            target.dataDevice = CUDA; target.dataDeviceIds = model.tpDevices;
            target.multiDeviceData = true; target.multiDeviceDatas.swap(source.multiDeviceDatas);
            source.multiDeviceData = false;
        }
        std::vector<std::pair<Data *, Data *>> roots;
        for (auto &pair : context.pastKeyValues) roots.emplace_back(&pair.first, &pair.second);
        model.AttachSparseKv(roots); // Reset hot mappings and publish fixed device page tables.
        std::vector<std::exception_ptr> errors(model.tpDevices.size());
        model.tpWorkers.Run(model.tpDevices, [&](int rank) {
            FastllmCudaSetDevice(model.tpDevices[rank]);
            StateCopies copies;
            for (int layer = 0; layer < model.block_cnt; ++layer) if (model.slidingLayers[layer])
                for (int part = 0; part < 2; ++part) {
                    auto &root = part ? context.pastKeyValues[layer].second : context.pastKeyValues[layer].first;
                    auto &local = *root.multiDeviceDatas.at(model.tpDevices[rank]);
                    const int rows = std::min(length, model.window - 1);
                    const size_t rowBytes = GetDataBytes(local.dataType, 1, local.dims.back());
                    const int first = model.tpSwaRingCache ? (length - rows) % model.window : 0;
                    const int count = std::min(rows, model.window - first);
                    const auto *source = static_cast<const char *>(windows[rank][layer * 2 + part].Data());
                    copies.Add(static_cast<char *>(local.cudaData) + first * rowBytes, source, count * rowBytes);
                    if (count < rows) copies.Add(local.cudaData, source + count * rowBytes, (rows - count) * rowBytes);
                }
            copies.Submit();
        }, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
        model.tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
        committed = true;
    }
};

struct NaiveN05FlashModel::GpuPDPools {
    std::map<const Data *,fastkv::Buffer> buffers;
};

struct NaiveN05FlashModel::GpuPDReceive : PDReceiveState {
    NaiveN05FlashModel &model;
    std::shared_ptr<ResponseContext> reservation = std::make_shared<ResponseContext>();
    std::vector<int> devices;
    std::vector<std::vector<fastkv::Buffer>> targets, windows;
    fastkv::Bytes expected;
    std::shared_ptr<DraftContext> draft;
    fastkv::Buffer draftArena;
    std::vector<std::pair<fastkv::Buffer, fastkv::Buffer>> draftCopies;
    int length;
    bool received = false, committed = false;

    GpuPDReceive(NaiveN05FlashModel &m, ResponseContext &context)
        : model(m), devices(m.PDRequestDevices(context)), length(context.allTokens.size() - 1) {
        RestorePDDevice restoreDevice;
        CheckPD(length > 0 && length <= model.max_positions, "invalid GPU receive length");
        if (!model.pdReceivePools) model.pdReceivePools = std::make_shared<GpuPDPools>();
        const int rows = std::min(length, model.window - 1), ranks = devices.size();
        reservation->Init(model.block_cnt, model.dataType, model.kvCacheDataType);
        // Direct DMA can outlive a cancelled request. Keep the window grant
        // with the receive buffers until their last transport lease retires.
        if (!model.TryAdmitResponseContext(&context)) throw fastkv::CapacityBusy("PD window slots exhausted");
        reservation->pastKeyValues.front().first.kvStorageOwner = context.pastKeyValues.front().first.kvStorageOwner;
        fastkv::Writer metadata;
        metadata.String(model.PDLayout()); metadata.U32(length);
        for (int layer = 0; layer < model.block_cnt; ++layer) for (int part = 0; part < 2; ++part) {
            const bool swa = model.slidingLayers[layer]; const auto &cfg = swa ? model.sliding : model.full;
            auto &root = part ? reservation->pastKeyValues[layer].second : reservation->pastKeyValues[layer].first;
            root.dataType = model.LayerKVType(layer); root.UpdateUnitSize();
            root.Resize({1, swa ? rows : length, cfg.kvHeads * (part ? cfg.valueDim : cfg.headDim) +
                (!swa && !part ? model.PackedIndexerWidth() : 0)});
            root.dataDevice = CUDA; root.dataDeviceIds = devices; root.multiDeviceData = true;
            metadata.Ints(root.dims);
            for (int device : devices) {
                auto *local = new Data(root.dataType);
                local->dataDevice = CUDA; local->dataDeviceIds = {device}; local->isKVCache = true;
                root.multiDeviceDatas[device] = local;
            }
        }
        // Check page availability before mutating any pool. Admission on the
        // model lane serializes these checks with forward and other receives.
        std::vector<std::pair<int, size_t>> windowPlan;
        for (int device : devices) {
            const int rank = std::find(model.tpDevices.begin(), model.tpDevices.end(), device) - model.tpDevices.begin();
            size_t bytes = 0;
            for (int layer = 0; layer < model.block_cnt; ++layer) for (int part = 0; part < 2; ++part) {
                if (model.slidingLayers[layer]) {
                    const int width = std::max(1, model.sliding.kvHeads / ranks) *
                        (part ? model.sliding.valueDim : model.sliding.headDim);
                    bytes += GetDataBytes(model.LayerKVType(layer), rows, width);
                } else {
                    auto &pool = *model.tpPagedCaches.at(rank).at(layer * 2 + part);
                    std::lock_guard<std::mutex> lock(pool.pageIndexLocker);
                    if (pool.FreePageCount() < (length + pool.pageLen - 1) / pool.pageLen)
                        throw fastkv::CapacityBusy("GPU receive pages exhausted");
                }
            }
            if (model.draftEnabled) {
                const size_t kv = GetDataBytes(BFLOAT16, std::min(length, model.draftWindow - 1),
                                               model.draftKvHeads * model.draftHeadDim);
                const size_t tail = GetDataBytes(BFLOAT16, std::min(length, model.draftWindow + GetPageLen() - 1),
                                                 model.embed_dim);
                bytes += 2 * model.draftLayers * kv + tail;
            }
            windowPlan.emplace_back(device, bytes);
        }
        std::vector<fastkv::Buffer> arenas;
        if (model.PDExportPool()) arenas = model.PDExportPool()->AcquireBatch(windowPlan);
        else for (const auto &item : windowPlan) arenas.push_back(fastkv::Buffer::Allocate(item.second,item.first));
        windows.resize(ranks);
        for (int rank = 0; rank < ranks; ++rank) {
            const int device = devices[rank]; FastllmCudaSetDevice(device);
            const int worldRank = std::find(model.tpDevices.begin(), model.tpDevices.end(), device) - model.tpDevices.begin();
            fastkv::Writer descriptors; const size_t begin = targets.size(); size_t windowOffset = 0;
            windows[rank].resize(model.block_cnt * 2);
            std::vector<PageTableUpdate> tables;
            auto pages = [&](const Data &key, const Data &pool) {
                auto &base = model.pdReceivePools->buffers[&pool];
                if (!base.Size()) base = fastkv::Buffer::Borrow(pool.cudaData, pool.GetBytes(),
                    fastkv::MemoryKind::Cuda, device, model.tpKVPool.at(device));
                const size_t rowBytes = GetDataBytes(pool.dataType, 1, pool.dims.back());
                std::vector<fastkv::Buffer> spans;
                for (int first = 0; first < length;) {
                    const int page = key.pageIndex.at(first / key.pageLen);
                    int count = std::min(key.pageLen, length - first);
                    while (first + count < length && count % key.pageLen == 0 &&
                           key.pageIndex.at((first + count) / key.pageLen) == page + count / key.pageLen)
                        count += std::min(key.pageLen, length - first - count);
                    spans.push_back(base.Slice(size_t(page) * key.pageLen * rowBytes, size_t(count) * rowBytes));
                    first += count;
                }
                return spans;
            };
            auto add = [&](DataType type, int count, int width, bool fp8, std::vector<fastkv::Buffer> spans) {
                descriptors.U32(type); descriptors.Ints({1,count,width}); descriptors.U32(fp8);
                descriptors.U64(GetDataBytes(type,count,width)); targets.push_back(std::move(spans));
            };
            for (int layer = 0; layer < model.block_cnt; ++layer) {
                const bool swa = model.slidingLayers[layer]; const auto &cfg = swa ? model.sliding : model.full;
                for (int part = 0; part < 2; ++part) {
                    auto &root = part ? reservation->pastKeyValues[layer].second : reservation->pastKeyValues[layer].first;
                    auto &local = *root.multiDeviceDatas.at(device);
                    const int width = std::max(1, cfg.kvHeads / ranks) * (part ? cfg.valueDim : cfg.headDim) +
                        (!swa && !part ? model.LocalIndexerCacheWidth(layer, worldRank) : 0);
                    local.kvCacheFp8Indexer = !swa && !part && !model.HasSeparateIndexer() && model.fp8IndexerCache;
                    if (swa) {
                        const int slots = model.tpSwaRingCache ?
                            model.window + (model.draftEnabled ? model.draftBlock : 0) : rows;
                        const int capacity = (slots + 127) / 128 * 128;
                        model.ReserveCache(local,capacity,width,true,&context.pastKeyValues.front().first);
                        const size_t bytes = GetDataBytes(local.dataType,rows,width);
                        auto view = arenas[rank].Slice(windowOffset,bytes); windowOffset += bytes;
                        windows[rank][layer * 2 + part] = view;
                        add(local.dataType,rows,width,false,{view});
                    } else {
                        auto &pool = *model.tpPagedCaches[worldRank][layer * 2 + part];
                        if (model.PreparePagedCache(local,pool,length,&tables,&context.pastKeyValues.front().first))
                            tables.back().first = 0;
                        // Private unwritten pages may be reordered before their
                        // first upload, reducing scatter spans without copying.
                        std::sort(local.pageIndex.begin(),local.pageIndex.end());
                        local.lastPageLen = (length - 1) % local.pageLen + 1;
                        add(local.dataType,length,width,local.kvCacheFp8Indexer,pages(local,pool));
                    }
                    local.Resize({1,swa ? rows : length,width});
                }
                if (model.HasSeparateIndexer() && model.HasLocalIndexer(layer,worldRank)) {
                    const auto &key = *reservation->pastKeyValues[layer].first.multiDeviceDatas.at(device);
                    const auto &index = model.IndexerStorage(layer,worldRank);
                    add(index.dataType,length,index.dims.back(),model.fp8IndexerCache,pages(key,index));
                }
            }
            if (model.draftEnabled)
                draftArena = arenas[rank].Slice(windowOffset, arenas[rank].Size() - windowOffset);
            model.UploadPagedCacheTables(tables);
            metadata.Blob(descriptors.bytes.data(),descriptors.bytes.size()); metadata.U32(targets.size()-begin);
        }
        metadata.U32(model.draftEnabled);
        if (model.draftEnabled) {
            draft = model.ReservePDDraft(length, devices.front());
            metadata.U32(length); metadata.U32(model.draftLayers); metadata.U32(1);
        }
        expected = std::move(metadata.bytes);
    }
    std::vector<std::vector<fastkv::Buffer>> Targets(const fastkv::Bytes &metadata,
                                                   const std::vector<size_t> &sizes) override {
        CheckPD(!received && (draft || sizes.size() == targets.size()), "invalid GPU receive payload count");
        fastkv::Message wire; wire.metadata = metadata; wire.buffers.resize(sizes.size());
        const auto payload = PDStatePayload(wire);
        CheckPD(draft ? payload.metadata.size() >= expected.size() &&
                std::equal(expected.begin(), expected.end(), payload.metadata.begin()) :
                payload.metadata == expected, "incompatible GPU state metadata");
        (void)PDReceivePlacement(metadata,sizes.size(),devices);
        auto result = targets;
        if (draft) {
            fastkv::Reader reader(payload.metadata.data() + expected.size(), payload.metadata.size() - expected.size());
            // Register the existing SWA arena once. Registering each transient
            // draft allocation on every RDMA rail costs more than the GPU copy.
            size_t offset = 0;
            auto bind = [&](Data &tensor, int boundedAxis = -1) {
                auto destination = PDBindTensor(reader, tensor, draft, boundedAxis);
                auto staging = draftArena.Slice(offset, destination.Size());
                offset += destination.Size();
                draftCopies.emplace_back(std::move(destination), staging);
                result.push_back({std::move(staging)});
            };
            for (auto &pair : draft->kv) { bind(pair.first); bind(pair.second); }
            const auto hasTail = reader.U32();
            CheckPD(hasTail <= 1, "invalid draft tail flag");
            if (hasTail) bind(draft->prefixHidden, 1);
            else {
                draft->prefixHidden.dims.clear();
                draft->prefixHidden.strides.clear();
            }
            reader.End();
            CheckPD(result.size() == sizes.size(), "invalid draft payload count");
        }
        for (size_t i = 0; i < result.size(); ++i) {
            size_t bytes = 0;
            for (auto &span : result[i]) { bytes += span.Size(); span = span.WithLease(reservation); }
            CheckPD(bytes == sizes[i], "invalid GPU state bytes");
        }
        received = true; return result;
    }
    void CommitShards(ResponseContext &context, const fastkv::Message &wire) override {
        PDRowMessage state(wire);
        fastkv::Reader header(state.metadata);
        const int committed = header.U32(), tailRows = header.U32();
        header.End();
        CheckPD(!received && draft && devices.size() == 1 && state.layout == model.PDLayout() &&
                committed == length && tailRows >= 0 && tailRows <= draft->prefixHidden.dims[1],
                "incompatible sharded checkpoint");
        std::map<std::string, PDRowDestination> parts;
        size_t slot = 0;
        for (int layer = 0; layer < model.block_cnt; ++layer) {
            for (int v = 0; v < 2; ++v) {
                const auto &root = v ? reservation->pastKeyValues[layer].second : reservation->pastKeyValues[layer].first;
                const auto &local = *root.multiDeviceDatas.at(devices.front());
                parts.emplace((v ? "v" : "k") + std::to_string(layer), PDRowDestination{
                    size_t(local.dims[1]), GetDataBytes(local.dataType, 1, local.dims[2]), targets.at(slot++)});
            }
            if (!model.slidingLayers[layer] && model.HasSeparateIndexer())
                parts.emplace("i" + std::to_string(layer), PDRowDestination{size_t(length),
                    GetDataBytes(model.AuxiliaryKVType(), 1, model.IndexerCacheWidth(model.AuxiliaryKVType())),
                    targets.at(slot++)});
        }
        auto bind = [&](std::string name, Data &tensor) {
            const size_t rows = tensor.dims[1], pitch = GetDataBytes(tensor.dataType, 1, tensor.dims[2]);
            parts.emplace(std::move(name), PDRowDestination{rows, pitch, {fastkv::Buffer::Borrow(
                tensor.cudaData, rows * pitch, fastkv::MemoryKind::Cuda, tensor.dataDeviceIds.front(), draft)}});
        };
        for (int layer = 0; layer < model.draftLayers; ++layer) {
            bind("dk" + std::to_string(layer), draft->kv[layer].first);
            bind("dv" + std::to_string(layer), draft->kv[layer].second);
        }
        if (tailRows) {
            draft->prefixHidden.Resize({1, tailRows, model.embed_dim});
            bind("hidden", draft->prefixHidden);
        } else {
            draft->prefixHidden.dims.clear();
            draft->prefixHidden.strides.clear();
        }
        state.Scatter(parts, devices.front());
        received = true;
        Commit(context);
    }
    void Commit(ResponseContext &context) override {
        CheckPD(received && !committed && context.pastKeyValues.size() == reservation->pastKeyValues.size(),
                "GPU transfer not ready");
        // Validate every root before adopting anything; never overwrite live KV.
        for (const auto &pair : context.pastKeyValues) for (const Data *root : {&pair.first,&pair.second}) {
            CheckPD(root->dims.empty(), "GPU receive requires empty context");
            for (const auto &local : root->multiDeviceDatas)
                CheckPD(local.second->dims.empty() && local.second->pageIndex.empty(), "GPU receive overwrites state");
        }
        RunPDWorkers(model.attentionParallel ? &model.attentionParallel->runtime : nullptr,model.tpWorkers,devices,[&](int rank) {
            StateCopies copies; const int rows = std::min(length,model.window - 1);
            for (int layer = 0; layer < model.block_cnt; ++layer) if (model.slidingLayers[layer])
                for (int part = 0; part < 2; ++part) {
                    auto &root = part ? reservation->pastKeyValues[layer].second : reservation->pastKeyValues[layer].first;
                    auto &local = *root.multiDeviceDatas.at(devices[rank]);
                    const size_t rowBytes = GetDataBytes(local.dataType,1,local.dims.back());
                    const int first = model.tpSwaRingCache ? (length - rows) % model.window : 0;
                    const int count = std::min(rows,model.window-first);
                    const char *source = static_cast<const char *>(windows[rank][layer * 2 + part].Data());
                    copies.Add(static_cast<char *>(local.cudaData)+first*rowBytes,source,count*rowBytes);
                    if (count < rows) copies.Add(local.cudaData,source+count*rowBytes,(rows-count)*rowBytes);
                }
            bool ok = true;
            for (const auto &copy : draftCopies) {
                const auto &to = copy.first, &from = copy.second;
                if (to.Device() == from.Device()) copies.Add(to.MutableData(), from.Data(), from.Size());
                else if (ok) ok = FastllmCudaMemcpyPeerAsyncCurrentThread(
                    to.Device(), to.MutableData(), from.Device(), from.Data(), from.Size());
            }
            copies.Submit();
            CheckPD(ok, "draft prefix transfer failed");
        });
        model.InvalidateTensorParallelCache(model.attentionParallel ? &context.pastKeyValues.front().first : nullptr);
        for (int layer = 0; layer < model.block_cnt; ++layer) for (int part = 0; part < 2; ++part) {
            auto &source = part ? reservation->pastKeyValues[layer].second : reservation->pastKeyValues[layer].first;
            auto &target = part ? context.pastKeyValues[layer].second : context.pastKeyValues[layer].first;
            for (auto &entry : target.multiDeviceDatas) delete entry.second;
            target.multiDeviceDatas.clear(); target.multiDeviceDatas.swap(source.multiDeviceDatas);
            source.multiDeviceData = false;
            target.dataType = source.dataType; target.UpdateUnitSize(); target.Resize(source.dims);
            SyncImportedDPCapacity(target);
            target.dataDevice = CUDA; target.dataDeviceIds = devices;
            target.multiDeviceData = true;
        }
        if (draft) {
            std::lock_guard<std::mutex> guard(model.historyMutex);
            model.draftContexts.at(&context.pastKeyValues) = std::move(draft);
        }
        committed = true;
    }
};
#endif

std::shared_ptr<PDReceiveState> NaiveN05FlashModel::PreparePDReceive(ResponseContext &context) {
#ifdef USE_CUDA
    if (sparseKv) {
        CheckPD(!draftEnabled && !indexerContextSharding && tpDevices.size() > 1,
                "KVMem PD requires TP, replicated/layer_shard Indexer and no draft");
        CheckPD(context.allTokens.size() > 1 && context.allTokens.size() <= size_t(max_positions), "invalid KVMem prompt");
        return std::make_shared<SparsePDReceive>(*this, context.allTokens.size() - 1);
    }
    if (attentionDP > 1 && !tpPagedCaches.empty())
        return std::make_shared<GpuPDReceive>(*this,context);
#endif
    return {};
}

std::vector<int> NaiveN05FlashModel::PDRequestDevices(const ResponseContext &context) const {
    if (attentionDP > 1) {
        CheckPD(!context.pastKeyValues.empty(), "missing request KV group");
        const auto &devices = context.pastKeyValues.front().first.dataDeviceIds;
        CheckPD(devices.size() == size_t(AttentionTPSize()), "invalid request KV group");
        return devices;
    }
#ifdef USE_CUDA
    if (tpDevices.empty()) return {FastllmCudaGetDevice()};
#endif
    return tpDevices;
}

bool NaiveN05FlashModel::SelectPDRequestGroup(ResponseContext &context,
        const std::vector<size_t> &available, size_t required) {
    CheckPD(available.size() == size_t(attentionDP), "invalid PD group budgets");
    const int previous = PDRequestGroup(context);
    if (available.at(previous) >= required && TryAdmitResponseContext(&context)) return true;
    // A full session lane queues until its budget or SWA slot is released.
    if (context.pdPrefillGroup >= 0) return false;
    if (!context.pastKeyValues.empty() && context.pastKeyValues.front().first.kvStorageOwner) return false;
    int group = -1;
    for (int g = 0; g < attentionDP; ++g)
        if (available[g] >= required && (!activeWindowSlots || activeWindowSlots->Available(g)) &&
            (group < 0 || available[g] > available[group])) group = g;
    if (group < 0) return false;
    // Only an unadmitted, empty context can change owner. Drop the queued
    // prefix hint; no page or cached SWA state has been borrowed yet.
    for (const auto &pair : context.pastKeyValues)
        for (const Data *root : {&pair.first, &pair.second}) {
            CheckPD(root->dims.empty() && !root->cudaData && !root->cpuData,
                    "cannot migrate active PD KV");
            for (const auto &entry : root->multiDeviceDatas)
                CheckPD(entry.second->dims.empty() && entry.second->pageIndex.empty(),
                        "cannot migrate allocated PD KV");
        }
    std::vector<std::pair<Data *, Data *>> roots;
    for (auto &pair : context.pastKeyValues) {
        for (Data *root : {&pair.first, &pair.second}) {
            for (auto &entry : root->multiDeviceDatas) delete entry.second;
            root->multiDeviceDatas.clear(); root->multiDeviceData = false; root->dataDeviceIds.clear();
        }
        roots.emplace_back(&pair.first, &pair.second);
    }
    InitAttentionParallelCache(roots, group);
    std::lock_guard<std::mutex> lock(historyMutex);
    auto it = activePrefixes.find(&context.pastKeyValues.front().first);
    if (it != activePrefixes.end()) it->second.queued.reset();
    return TryAdmitResponseContext(&context);
}

std::string NaiveN05FlashModel::PDGroupLayout(int attentionRanks) const {
    std::string layout = draftEnabled ? "naive-n05-group-draft-v4" : "naive-n05-group-state-v2";
    for (int value : {block_cnt, attentionRanks, int(dataType), int(kvCacheDataType),
                      int(fp8IndexerCache), window, full.kvHeads, full.headDim, full.valueDim,
                      sliding.kvHeads, sliding.headDim, sliding.valueDim, IndexerCacheWidth(AuxiliaryKVType()),
                      int(HasSeparateIndexer())})
        layout += ":" + std::to_string(value);
    return draftEnabled ? layout + DraftPDLayout(true) : layout;
}

std::string NaiveN05FlashModel::PDAlternateLayout() const {
    // Offer a full attention replica to a group-local receiver. The existing
    // TP export and local checkpoints remain available in their native layout.
    if (attentionDP != 1 || tpDevices.size() < 2 || !draftEnabled || sparseKv ||
        indexerContextSharding || indexerLayerSharding) return {};
    return PDGroupLayout(1);
}

fastkv::Message NaiveN05FlashModel::PDAlternateState(const fastkv::Message &wire) const {
    PDRowMessage out; out.layout = PDAlternateLayout();
    CheckPD(!out.layout.empty(), "unsupported TP to replica export");
    const auto state = PDStatePayload(wire);
    fastkv::Reader header(state.metadata);
    CheckPD(header.String() == PDLayout(), "invalid native TP state");
    const int length = header.U32();
    int tailRows = 0;
    for (int i = 0; i < block_cnt * 2; ++i) header.Ints(3);
    size_t index = 0;
    // Consume native tensor descriptors; retained shards alias the registered
    // export arenas. Replicated GQA heads and Indexer rows cross the NIC once.
    auto take = [&](fastkv::Reader &reader, const std::string &name, DataType type, int rows,
                    int fullWidth, int rank, int ranks, size_t baseOffset = 0) {
        const auto dtype = reader.U32(); const auto shape = reader.Ints(3);
        reader.U32(); const size_t bytes = reader.U64();
        CheckPD(dtype == uint32_t(type) && shape.size() == 3 && shape[0] == 1 && shape[1] == rows &&
                shape[2] > 0 && fullWidth % shape[2] == 0 &&
                ranks % (fullWidth / shape[2]) == 0 && bytes == GetDataBytes(type, rows, shape[2]),
                "invalid logical shard geometry");
        const auto &buf = state.buffers.at(index++);
        CheckPD(buf.Size() == bytes, "invalid native shard buffer");
        const int replicas = ranks / (fullWidth / shape[2]);
        const size_t width = GetDataBytes(type, 1, shape[2]);
        if (rank % replicas == 0)
            out.shards.push_back({name, baseOffset + size_t(rank / replicas) * width, width, buf});
    };
    const int ranks = tpDevices.size();
    for (int rank = 0; rank < ranks; ++rank) {
        const auto descriptors = header.Blob();
        const auto count = header.U32(); const size_t first = index;
        fastkv::Reader tensors(descriptors);
        for (int layer = 0; layer < block_cnt; ++layer) {
            const auto &cfg = slidingLayers[layer] ? sliding : full;
            const int rows = slidingLayers[layer] ? std::min(length, window - 1) : length;
            const auto k = "k" + std::to_string(layer);
            take(tensors, k, LayerKVType(layer), rows, cfg.kvHeads * cfg.headDim, rank, ranks);
            take(tensors, "v" + std::to_string(layer), LayerKVType(layer), rows, cfg.kvHeads * cfg.valueDim, rank, ranks);
            if (!slidingLayers[layer]) {
                const size_t kBytes = GetDataBytes(LayerKVType(layer), 1, cfg.kvHeads * cfg.headDim);
                take(tensors, HasSeparateIndexer() ? "i" + std::to_string(layer) : k, AuxiliaryKVType(),
                     rows, IndexerCacheWidth(AuxiliaryKVType()), rank, ranks, HasSeparateIndexer() ? 0 : kBytes);
            }
        }
        tensors.End();
        CheckPD(index - first == count, "invalid native TP tensor count");
    }
    CheckPD(header.U32() == 1 && header.U32() == uint32_t(length) &&
            header.U32() == uint32_t(draftLayers), "invalid native draft checkpoint");
    const int draftRanks = header.U32();
    CheckPD(draftRanks > 0 && draftRanks <= ranks, "invalid draft rank count");
    for (int rank = 0; rank < draftRanks; ++rank)
        for (int layer = 0; layer < draftLayers; ++layer) for (const char *part : {"dk", "dv"})
            take(header, part + std::to_string(layer), BFLOAT16, std::min(length, draftWindow - 1),
                 draftKvHeads * draftHeadDim, rank, draftRanks);
    if (header.U32()) {
        const auto bytes = state.buffers.at(index).Size();
        const size_t rowBytes = GetDataBytes(BFLOAT16, 1, embed_dim);
        CheckPD(bytes % rowBytes == 0 && bytes / rowBytes <= size_t(std::min(length, draftWindow + GetPageLen() - 1)),
                "invalid draft history length");
        tailRows = bytes / rowBytes;
        take(header, "hidden", BFLOAT16, tailRows, embed_dim, 0, 1);
    }
    header.End();
    CheckPD(index == state.buffers.size(), "unclaimed native TP buffer");
    fastkv::Writer attributes;
    attributes.U32(length); attributes.U32(tailRows); out.metadata = std::move(attributes.bytes);
    return out.Encode();
}

std::string NaiveN05FlashModel::PDLayout() const {
    CheckPD(!indexerContextSharding, "Indexer context sharding needs a separate state adapter");
    CheckPD(attentionDP == 1 || !sparseKv, "Attention DP PD requires GPU KV");
    if (attentionDP > 1) {
        // Group-local wire ranks. World/DP/EP sizes are execution details;
        // head sharding and the state representation remain strict.
        // V2 keeps packed K/Indexer rows together so receive targets can point
        // at final pages without one RDMA span per token. Legacy TP/KVMem keep
        // their established component wire layout.
        return PDGroupLayout(AttentionTPSize());
    }
    CheckPD(!(sparseKv && draftEnabled), "KVMem PD draft state transfer is not implemented");
    return std::string(tpDevices.size() > 1 ? "naive-n05-state-v3:" : "naive-n05-state-v1:") +
           std::to_string(block_cnt) + ":" +
           std::to_string(std::max(size_t(1), tpDevices.size())) + ":" + std::to_string(dataType) + ":" +
           std::to_string(kvCacheDataType) + ":" + std::to_string(fp8IndexerCache) + ":" +
           std::to_string(indexerLayerSharding) + ":" + std::to_string(draftEnabled) + DraftPDLayout();
}

struct NaiveN05FlashModel::PDExportPlan {
    PDStateWriter writer;
    std::vector<PDStateWriter> ranks;
    std::shared_ptr<DraftContext> draft;
    bool planned = false;
    PDExportPlan(fastkv::BufferPool *pool, size_t count) : writer(pool) {
        ranks.reserve(count);
        for (size_t i = 0; i < count; ++i) ranks.push_back(writer.Fork());
    }
};

std::function<fastkv::Message()> NaiveN05FlashModel::PreparePDExport(ResponseContext &context) {
#ifdef USE_CUDA
    if (attentionDP > 1 && AttentionTPSize() == 1 && !sparseKv && !HasSeparateIndexer()) {
        auto devices = PDRequestDevices(context);
        // Attention DP joins its workers to the caller before returning. Fence
        // the owner's caller stream once before handing its pages and ring
        // to another thread. Other devices can continue without a dispatch.
        RestorePDDevice restoreDevice;
        FastllmCudaSetDevice(devices.front());
        FastllmCudaSyncCurrentThreadStream();
        CheckPD(!FastllmCudaGetThreadError(), "producer stream failed");
        auto plan = std::make_shared<PDExportPlan>(PDExportPool(), devices.size());
        if (draftEnabled) {
            {
                std::lock_guard<std::mutex> lock(historyMutex);
                plan->draft = draftContexts.at(&context.pastKeyValues);
            }
            CheckPD(bool(plan->draft), "missing draft export state");
            // Draft projection joins its owner stream on the projection device.
            if (tpDevices.front() != devices.front() && !plan->draft->prefixHidden.dims.empty()) {
                FastllmCudaSetDevice(tpDevices.front());
                FastllmCudaSyncCurrentThreadStream();
                CheckPD(!FastllmCudaGetThreadError(), "draft producer stream failed");
            }
        }
        return [this, &context, devices, plan = std::move(plan)]() mutable {
            auto reply = PackPDState(context, devices, *plan, true);
            plan.reset(); // Transport owns the result. Only retain plans across capacity retries.
            return reply;
        };
    }
#endif
    return {};
}

void NaiveN05FlashModel::ReleasePDExportSources(ResponseContext &context) {
    // Release pages and physical SWA allocations before returning the admission
    // slot. The export lane runs inline to leave model workers free for forward.
    ReleaseRequestWindowStorage(&context, 1);
}

fastkv::Message NaiveN05FlashModel::ExportPDState(ResponseContext &context) {
    const auto devices = PDRequestDevices(context);
    PDExportPlan plan(PDExportPool(), devices.size());
    if (draftEnabled) {
        std::lock_guard<std::mutex> lock(historyMutex);
        plan.draft = draftContexts.at(&context.pastKeyValues);
    }
    return PackPDState(context, devices, plan, false);
}

fastkv::Message NaiveN05FlashModel::PackPDState(ResponseContext &context, const std::vector<int> &devices,
                                               PDExportPlan &plan, bool independent) {
#ifndef USE_CUDA
    throw std::runtime_error("FastKV Naive requires CUDA");
#else
    CheckPD(!sparseKv, "PD prefill requires GPU KV storage");
    CheckPD(context.preTokens > 0, "empty prefill state");
    const bool tp = tpDevices.size() > 1;
    auto &writer = plan.writer;
    auto &ranks = plan.ranks;
    auto pack = [&](int rank, bool planning) {
        if (!planning) FastllmCudaSetDevice(devices[rank]);
        const int worldRank = std::find(tpDevices.begin(), tpDevices.end(), devices[rank]) - tpDevices.begin();
        auto &output = ranks[rank];
        StateCopies copies;
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (int part = 0; part < 2; ++part) {
                const Data &root =
                    part ? context.pastKeyValues[layer].second : context.pastKeyValues[layer].first;
                const Data &cache = tp ? *root.multiDeviceDatas.at(devices[rank]) : root;
                CheckPD(cache.dims.size() == 3 && cache.dims[0] == 1 && cache.dataDevice == CUDA,
                        "invalid local cache");
                const int rows =
                    slidingLayers[layer] ? std::min(context.preTokens, window - 1) : context.preTokens;
                const auto &cfg = slidingLayers[layer] ? sliding : full;
                const int width = tp ? std::max(1, cfg.kvHeads / int(devices.size())) *
                    (part ? cfg.valueDim : cfg.headDim) +
                    (attentionDP > 1 && !part && !slidingLayers[layer] ? LocalIndexerCacheWidth(layer, worldRank) : 0)
                    : cache.dims[2];
                CheckPD(cache.dims[1] == rows, "cache length mismatch");
                Data compact(cache.dataType, {1, rows, width});
                compact.dataDevice = CUDA;
                compact.dataDeviceIds = {devices[rank]};
                compact.kvCacheFp8Indexer = (!tp || attentionDP > 1) && cache.kvCacheFp8Indexer;
                if (planning) { output.PlanTensor(compact); continue; }
                output.AllocateTensor(compact);
                const size_t rowBytes = GetDataBytes(cache.dataType, 1, width);
                if (cache.isPagedKVCache) {
                    CheckPD(cache.pagedKVCacheData && cache.pageLen > 0, "missing page owner");
                    CopyPagedRows(compact, cache, *cache.pagedKVCacheData, false, copies);
                } else {
                    const bool ring = tp && tpSwaRingCache && slidingLayers[layer];
                    const int first = ring ? (context.preTokens - rows) % window : 0;
                    const int count = ring ? std::min(rows, window - first) : rows;
                    copies.Add(compact.cudaData,
                                                      (char *)cache.cudaData + size_t(first) * rowBytes,
                                                      size_t(count) * rowBytes);
                    if (count < rows)
                        copies.Add((char *)compact.cudaData + size_t(count) * rowBytes,
                                                          cache.cudaData, size_t(rows - count) * rowBytes);
                }
            }
            if (tp && HasLocalIndexer(layer, worldRank) && (attentionDP == 1 || HasSeparateIndexer())) {
                const auto &root = context.pastKeyValues[layer].first;
                const auto &key = *root.multiDeviceDatas.at(devices[rank]);
                const auto &pool = HasSeparateIndexer() ? IndexerStorage(layer, worldRank) : *key.pagedKVCacheData;
                Data compact(pool.dataType, {1, context.preTokens, IndexerCacheWidth(AuxiliaryKVType())});
                compact.dataDevice = CUDA;
                compact.dataDeviceIds = {devices[rank]};
                compact.kvCacheFp8Indexer = fp8IndexerCache;
                if (planning) output.PlanTensor(compact);
                else {
                    output.AllocateTensor(compact);
                    CopyPagedRows(compact, key, pool, false, copies, HasSeparateIndexer() ? 0 :
                        std::max(1, full.kvHeads / int(devices.size())) * full.headDim);
                }
            }
        }
        if (!planning) copies.Submit();
    };
    auto packDraft = [&](bool planning) {
        if (!draftEnabled) return;
        CheckPD(bool(plan.draft), "missing draft export state");
        const auto &draft = *plan.draft;
        if (attentionDP == 1) {
            ExportDraftPDState(writer, draft, context.preTokens, planning);
            return;
        }
        CheckPD(draft.pending.empty() && draft.committed == context.preTokens &&
                draft.kv.size() == size_t(draftLayers), "invalid committed draft state");
        const int device = devices.front();
        StateCopies copies;
        auto component = [&](const Data &source) {
            CheckPD(source.dataDevice == CUDA && source.dataDeviceIds.size() == 1 &&
                    source.cudaData && source.dims.size() == 3 && source.dims[0] == 1 &&
                    source.dataType == BFLOAT16 && source.strides[2] == 1 &&
                    source.strides[1] == uint64_t(source.dims[2]), "invalid dense draft tensor");
            Data compact(source.dataType, source.dims);
            compact.dataDevice = CUDA; compact.dataDeviceIds = {device};
            if (planning) { writer.PlanTensor(compact); return; }
            writer.AllocateTensor(compact);
            if (source.dataDeviceIds.front() == device)
                copies.Add(compact.cudaData, source.cudaData, compact.GetBytes());
            else {
                // Prefix features can live on the projection GPU. Copy them once
                // directly to the registered arena instead of moving live tensors.
                CheckPD(FastllmCudaMemcpyPeerAsyncCurrentThread(writer.buffers.back().Device(), compact.cudaData,
                        source.dataDeviceIds.front(), source.cudaData, compact.GetBytes()), "draft peer copy failed");
            }
        };
        if (!planning) {
            FastllmCudaSetDevice(device);
            writer.metadata.U32(draft.committed); writer.metadata.U32(draftLayers); writer.metadata.U32(1);
        }
        for (const auto &pair : draft.kv) { component(pair.first); component(pair.second); }
        if (!planning) writer.metadata.U32(!draft.prefixHidden.dims.empty());
        if (!draft.prefixHidden.dims.empty()) component(draft.prefixHidden);
        if (!planning) copies.Submit();
    };
    if (!plan.planned) {
        writer.metadata.String(PDLayout());
        writer.metadata.U32(context.preTokens);
        for (const auto &pair : context.pastKeyValues)
            for (const Data *root : {&pair.first, &pair.second})
                writer.metadata.Ints(root->dims);
        for (int rank = 0; rank < int(devices.size()); ++rank) pack(rank, true);
        packDraft(true);
        plan.planned = true;
    }
    writer.Reserve(independent);
    if (independent) pack(0, false); // Dedicated thread/stream on this request's only owner.
    else RunPDWorkers(attentionParallel ? &attentionParallel->runtime : nullptr, tpWorkers, devices,
                      [&](int rank) { pack(rank, false); });
    for (auto &rank : ranks) {
        writer.metadata.Blob(rank.metadata.bytes.data(), rank.metadata.bytes.size());
        writer.metadata.U32(rank.buffers.size());
        writer.buffers.insert(writer.buffers.end(), rank.buffers.begin(), rank.buffers.end());
    }
    writer.metadata.U32(draftEnabled);
    packDraft(false);
    return writer.Finish(devices);
#endif
}

fastkv::CacheSnapshot NaiveN05FlashModel::DescribePDCache(ResponseContext &context, const fastkv::Message &wire) {
    auto snapshot = basellm::DescribePDCache(context, wire);
#ifdef USE_CUDA
    const auto devices = PDRequestDevices(context);
    size_t index = 0;
    auto describe = [&](const Data &key, const Data &storage, int rows, bool range, int width) {
        auto &payload = snapshot.payloads.at(index++);
        payload.storage = range ? fastkv::CacheStorage::KV : fastkv::CacheStorage::Window;
        if (!range || !key.isPagedKVCache || !key.pagedKVCacheData) return;
        auto &owner = *key.pagedKVCacheData;
        const size_t rowBytes = GetDataBytes(storage.dataType, 1, width);
        CheckPD(size_t(rows) * rowBytes == wire.buffers[index - 1].Size(), "cache region shape mismatch");
        payload.prefixRange = true; payload.regions.clear();
        std::lock_guard<std::mutex> lock(owner.pageIndexLocker);
        for (int at = 0; at < rows; at += key.pageLen) {
            const int page = key.pageIndex.at(at / key.pageLen);
            const size_t bytes = size_t(std::min(key.pageLen, rows - at)) * rowBytes;
            // K and Indexer can share one physical allocation. Their component
            // IDs must differ even when different tail lengths have equal bytes.
            // Generation distinguishes recycled pages; length distinguishes tails.
            const std::string identity = std::to_string(reinterpret_cast<uintptr_t>(&storage)) + ":" +
                std::to_string(index - 1) + ":" +
                std::to_string(page) + ":" + std::to_string(owner.pageTimestamp.at(page)) + ":" + std::to_string(bytes);
            payload.regions.push_back({size_t(at) * rowBytes, bytes, identity});
        }
    };
    for (size_t rank = 0; rank < devices.size(); ++rank) {
        const int worldRank = std::find(tpDevices.begin(), tpDevices.end(), devices[rank]) - tpDevices.begin();
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (int part = 0; part < 2; ++part) {
                const auto &root = part ? context.pastKeyValues[layer].second : context.pastKeyValues[layer].first;
                const auto &cache = tpDevices.size() > 1 ? *root.multiDeviceDatas.at(devices[rank]) : root;
                describe(cache, cache.pagedKVCacheData ? *cache.pagedKVCacheData : cache,
                         context.preTokens, !slidingLayers[layer],
                         tpDevices.size() > 1 ? std::max(1, full.kvHeads / int(devices.size())) *
                             (part ? full.valueDim : full.headDim) +
                             (attentionDP > 1 && !part ? LocalIndexerCacheWidth(layer, worldRank) : 0) : cache.dims.back());
            }
            if (tpDevices.size() > 1 && HasLocalIndexer(layer, worldRank) && (attentionDP == 1 || HasSeparateIndexer())) {
                const auto &key = *context.pastKeyValues[layer].first.multiDeviceDatas.at(devices[rank]);
                describe(key, HasSeparateIndexer() ? IndexerStorage(layer, worldRank) : *key.pagedKVCacheData,
                         context.preTokens, true, IndexerCacheWidth(AuxiliaryKVType()));
            }
        }
    }
    // Draft checkpoints are boundary-local, like the SWA state.
    for (; index < snapshot.payloads.size(); ++index)
        snapshot.payloads[index].storage = fastkv::CacheStorage::Window;
#endif
    return snapshot;
}

void NaiveN05FlashModel::ImportPDState(ResponseContext &context, const fastkv::Message &wire) {
#ifndef USE_CUDA
    throw std::runtime_error("FastKV Naive requires CUDA");
#else
    CheckPD(!sparseKv, "KVMem import requires a prepared receive target");
    const auto message = PDStatePayload(wire);
    fastkv::Reader metadata(message.metadata);
    CheckPD(metadata.String() == PDLayout(), "incompatible state layout or TP size");
    const int length = metadata.U32();
    CheckPD(length > 0 && length == int(context.allTokens.size()) - 1, "prompt length mismatch");
    const auto devices = PDRequestDevices(context);
    InvalidateTensorParallelCache(attentionParallel ? &context.pastKeyValues.front().first : nullptr);
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
            root.dataType = LayerKVType(layer);
            root.UpdateUnitSize();
            root.Resize(shape);
            root.expansionDims = shape;
            if (tp)
                for (int device : devices) {
                    Data *&local = root.multiDeviceDatas[device];
                    if (local) {
                        CheckPD(local->pageIndex.empty(), "import requires an empty request cache");
                        delete local;
                    }
                    local = new Data(LayerKVType(layer));
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
        const int worldRank = std::find(tpDevices.begin(), tpDevices.end(), devices[rank]) - tpDevices.begin();
        PDStateReader reader(rankStates[rank]);
        StateCopies copies;
        std::vector<std::unique_ptr<Data>> imported;
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (int part = 0; part < 2; ++part) {
                imported.emplace_back(new Data());
                Data &value = *imported.back();
                reader.Tensor(value, true);
                const int rows = slidingLayers[layer] ? std::min(length, window - 1) : length;
                const auto &cfg = slidingLayers[layer] ? sliding : full;
                const int expected =
                    std::max(1, cfg.kvHeads / int(devices.size())) * (part ? cfg.valueDim : cfg.headDim) +
                    ((!tp || attentionDP > 1) && !part && !slidingLayers[layer]
                         ? PackedIndexerWidth()
                         : 0);
                CheckPD(value.dataType == LayerKVType(layer) &&
                            value.dims == std::vector<int>({1, rows, expected}),
                        "local KV shape mismatch");
                Data &root = part ? context.pastKeyValues[layer].second : context.pastKeyValues[layer].first;
                Data &cache = tp ? *root.multiDeviceDatas.at(devices[rank]) : root;
                cache.dataType = value.dataType;
                cache.UpdateUnitSize();
                cache.kvCacheFp8Indexer = value.kvCacheFp8Indexer;
                const int width = expected + (tp && attentionDP == 1 && !part && !slidingLayers[layer] ? LocalIndexerCacheWidth(layer, worldRank) : 0);
                if (tp) cache.kvCacheFp8Indexer = !part && !slidingLayers[layer] && !HasSeparateIndexer() && fp8IndexerCache && HasLocalIndexer(layer, worldRank);
                const size_t rowBytes = GetDataBytes(cache.dataType, 1, width);
                // Placement and scatter belong to the persistent owner worker.
                // Network workers never mutate this request's live page table.
                value.ToDevice(CUDA, std::vector<int>{devices[rank]});
                if (tp && !slidingLayers[layer] && !tpPagedCaches.empty()) {
                    PreparePagedCache(cache, *tpPagedCaches[worldRank][layer * 2 + part], rows);
                    CopyPagedRows(value, cache, *cache.pagedKVCacheData, true, copies);
                    cache.lastPageLen = (rows - 1) % cache.pageLen + 1;
                } else {
                    const bool ring = tp && tpSwaRingCache && slidingLayers[layer];
                    // Use the same fixed request slot as native prefill. DSpark
                    // stages verification candidates immediately after the ring.
                    const int capacity = (std::max(rows, ring ? window + (draftEnabled ? draftBlock : 0) : rows) + 127) / 128 * 128;
                    if (!tp) {
                        cache.dims.clear();
                        cache.expansionDims.clear();
                    }
                    ReserveCache(cache, capacity, width, true, &context.pastKeyValues.front().first);
                    const int first = ring ? (length - rows) % window : 0;
                    const int count = ring ? std::min(rows, window - first) : rows;
                    copies.Add((char *)cache.cudaData + size_t(first) * rowBytes,
                                                      value.cudaData, size_t(count) * rowBytes);
                    if (count < rows)
                        copies.Add(cache.cudaData,
                                                          (char *)value.cudaData + size_t(count) * rowBytes,
                                                          size_t(rows - count) * rowBytes);
                }
                cache.Resize({1, rows, width});
                cache.isKVCache = true;
            }
            if (tp && HasLocalIndexer(layer, worldRank) && (attentionDP == 1 || HasSeparateIndexer())) {
                imported.emplace_back(new Data());
                Data &value = *imported.back();
                reader.Tensor(value, true);
                const auto &root = context.pastKeyValues[layer].first;
                const auto &key = *root.multiDeviceDatas.at(devices[rank]);
                const auto &pool = HasSeparateIndexer() ? IndexerStorage(layer, worldRank) : *key.pagedKVCacheData;
                CheckPD(value.dataType == pool.dataType &&
                            value.dims == std::vector<int>({1, length, IndexerCacheWidth(AuxiliaryKVType())}) &&
                            value.kvCacheFp8Indexer == fp8IndexerCache,
                        "local Indexer shape mismatch");
                value.ToDevice(CUDA, std::vector<int>{devices[rank]});
                CopyPagedRows(value, key, pool, true, copies, HasSeparateIndexer() ? 0 :
                    std::max(1, full.kvHeads / int(devices.size())) * full.headDim);
            }
        }
        reader.End();
        copies.Submit();
    };
    RunPDWorkers(attentionParallel ? &attentionParallel->runtime : nullptr, tpWorkers, devices, unpack);
    if (attentionDP > 1) {
        for (auto &layer : context.pastKeyValues) {
            SyncImportedDPCapacity(layer.first);
            SyncImportedDPCapacity(layer.second);
        }
    }
    const bool hasDraft = metadata.U32();
    CheckPD(hasDraft == draftEnabled, "draft mode mismatch");
    if (hasDraft) {
        // Remaining metadata and payload belong to the draft adapter.
        fastkv::Message rest;
        const size_t consumed = message.metadata.size() - metadata.Remaining();
        rest.metadata.assign(message.metadata.begin() + consumed, message.metadata.end());
        rest.buffers.assign(message.buffers.begin() + offset, message.buffers.end());
        PDStateReader reader(rest);
        auto draft = ImportDraftPDState(reader, length, attentionDP > 1 ? devices.front() : -1);
        std::lock_guard<std::mutex> lock(historyMutex);
        draftContexts[&context.pastKeyValues] = std::move(draft);
    } else {
        metadata.End();
        CheckPD(offset == message.buffers.size(), "unclaimed state payload");
    }
#endif
}
} // namespace fastllm
