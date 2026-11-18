// Unified Naive prefix storage. The host directory is private to one loaded model;
// all layers, auxiliary pages and window state belong to a single token boundary.
namespace fastllm {
void NaiveN05FlashModel::ConfigureHostPrefixCache(size_t bytes, int64_t kvBytes, int64_t swaBytes) {
    if (mainLoop || pdRuntime) throw std::invalid_argument("configure standalone L2 before serving");
    fastkv::HostCache::Options options;
    options.SetBudgets(bytes, kvBytes, swaBytes);
    if (!options.bytes) {
        hostPrefixCache.reset();
        hostPrefixBytes = 0;
        hostPrefixSwaBytes = -1;
        return;
    }
    if (!saveHistoryChat || !prefixPool || tpPagedCaches.empty() || sparseKv)
        throw std::invalid_argument("FastKV L2 requires resident paged GPU KV and GPU prefix slots");
    hostPrefixCache = MakeHostPrefixCache(options.bytes, options.swaBytes);
    hostPrefixBytes = options.bytes;
    hostPrefixSwaBytes = options.swaBytes;
}

#ifdef USE_CUDA
struct NaiveN05FlashModel::HostPrefixTransfer {
    struct Pages {
        PagedCacheManager &pool;
        int layer, part;
        std::vector<int> ids;
        Pages(PagedCacheManager &p, int l, int k) : pool(p), layer(l), part(k) {}
        ~Pages() {
            if (!ids.empty()) pool.ReleasePageIndices(ids);
        }
    };
    std::vector<std::shared_ptr<Pages>> pinned;
    fastkv::CacheSnapshot state;

    // Read only complete pages; a writer can append or fork without touching
    // bytes already in flight. Sidecars use their owner's page IDs and versions.
    bool Prepare(NaiveN05FlashModel &m, ResponseContext &ctx, const std::shared_ptr<PrefixSnapshot> &snap,
                 bool restoring) {
        const int length = snap->length, page = GetPageLen();
        if (length <= 0 || length % page) return false;
        const int count = length / page;
        const size_t promptPages = (ctx.currentTokens.size() + page - 1) / page;
        const auto devices = m.PDRequestDevices(ctx);
        auto span = [&](void *addr, size_t bytes, int device, const std::shared_ptr<void> &lease,
                        fastkv::CacheStorage storage, std::string id = {}) {
            if (!bytes) return;
            state.state.buffers.push_back(
                fastkv::Buffer::Borrow(addr, bytes, fastkv::MemoryKind::Cuda, device, lease));
            if (!restoring) state.payloads.push_back({false, {{0, bytes, std::move(id)}}, storage});
        };
        for (int device : devices) {
            const int rank = std::find(m.tpDevices.begin(), m.tpDevices.end(), device) - m.tpDevices.begin();
            for (int layer = 0; layer < m.block_cnt; ++layer)
                for (int part = 0; part < 2; ++part) {
                    if (m.slidingLayers[layer]) {
                        const auto &pool = *snap->pool;
                        size_t offset =
                            pool.slots.Row(snap->slot) * pool.slotBytes + pool.offsets[layer * 2 + part];
                        const int width = std::max(1, m.sliding.kvHeads / m.AttentionTPSize()) *
                                          (part ? m.sliding.valueDim : m.sliding.headDim);
                        span((char *)pool.storage[rank].cudaData + offset,
                             GetDataBytes(m.AuxiliaryKVType(), std::min(length, m.window - 1), width), device,
                             snap, fastkv::CacheStorage::Window);
                        continue;
                    }
                    auto &manager = *m.tpPagedCaches[rank][layer * 2 + part];
                    auto pages = std::make_shared<Pages>(manager, layer, part);
                    if (restoring) {
                        {
                            std::lock_guard<std::mutex> lock(manager.pageIndexLocker);
                            if (size_t(manager.FreePageCount()) < promptPages) return false;
                        }
                        pages->ids.reserve(count);
                        for (int n = 0; n < count; ++n) pages->ids.push_back(manager.GetUnusedPageIndex(true));
                    } else {
                        const Data &root = part ? ctx.pastKeyValues[layer].second : ctx.pastKeyValues[layer].first;
                        auto local = root.multiDeviceDatas.find(device);
                        if (local == root.multiDeviceDatas.end() || local->second->pagedKVCacheData != &manager ||
                            local->second->pageIndex.size() < size_t(count))
                            return false;
                        pages->ids.assign(local->second->pageIndex.begin(),
                                          local->second->pageIndex.begin() + count);
                        manager.Pick(pages->ids);
                    }
                    pinned.push_back(pages);
                    auto addPages = [&](Data &storage) {
                        const size_t bytes = GetDataBytes(storage.dataType, storage.dims[1], storage.dims[3]);
                        std::lock_guard<std::mutex> lock(manager.pageIndexLocker);
                        for (int id : pages->ids) {
                            std::string key;
                            if (!restoring)
                                key = std::to_string(uintptr_t(&storage)) + "/" + std::to_string(id) + "/" +
                                      std::to_string(manager.pageTimestamp.at(id));
                            span((char *)storage.cudaData + size_t(id) * bytes, bytes, device, pages,
                                 fastkv::CacheStorage::KV, std::move(key));
                        }
                    };
                    addPages(manager);
                    auto sidecar = m.tpIndexerSidecars.find(&manager);
                    if (sidecar != m.tpIndexerSidecars.end()) addPages(*sidecar->second);
                }
        }
        if (m.draftEnabled) {
            Data &s = snap->pool->DraftShard(snap->slot);
            const size_t bytes = size_t(std::min(length, m.draftWindow - 1)) * m.embed_dim * sizeof(uint16_t);
            span((char *)s.cudaData + snap->pool->DraftOffset(snap->slot), bytes, s.dataDeviceIds.front(), snap,
                 fastkv::CacheStorage::Window);
        }
        return true;
    }

    void Adopt(NaiveN05FlashModel &m, ResponseContext &ctx, int length) {
        const auto devices = m.PDRequestDevices(ctx);
        for (auto &pages : pinned) {
            auto &pool = pages->pool;
            const int device = pool.dataDeviceIds.front();
            Data &root =
                pages->part ? ctx.pastKeyValues[pages->layer].second : ctx.pastKeyValues[pages->layer].first;
            auto &local = root.multiDeviceDatas[device];
            if (!local) {
                local = new Data(pool.dataType);
                local->SetKVCache();
                local->cacheUid = root.cacheUid;
            }
            AssertInFastLLM(local->pageIndex.empty(), "FastKV L2 requires an empty request");
            // As on an L1 hit, the next forward materializes the page table.
            // A pending request must not reserve an active table or root capacity here.
            local->kvCacheFp8Indexer = pool.kvCacheFp8Indexer;
            local->dataType = pool.dataType;
            local->UpdateUnitSize();
            local->dataDevice = CUDA;
            local->dataDeviceIds = {device};
            local->isKVCache = local->isPagedKVCache = true;
            local->pagedKVCacheData = &pool;
            local->pageLen = local->lastPageLen = pool.pageLen;
            local->pageIndex.swap(pages->ids);
            local->Resize({pool.dims[2], length, pool.dims[3]});
            if (device == devices.front()) {
                root.multiDeviceData = true;
                root.dataDevice = CUDA;
                root.dataDeviceIds = devices;
                root.isKVCache = root.isPagedKVCache = true;
                root.dataType = pool.dataType;
                root.UpdateUnitSize();
                root.pageLen = root.lastPageLen = pool.pageLen;
                root.pageIndex = local->pageIndex;
                root.pagedKVCacheData = &pool;
                root.dims = local->dims;
            }
        }
    }
};
#endif

void NaiveN05FlashModel::PublishHostPrefix(ResponseContext &context,
                                           const std::shared_ptr<PrefixSnapshot> &snapshot) {
#ifdef USE_CUDA
    if (hostPrefixCache->ProbeTokens("local", snapshot->tokens) == snapshot->tokens.size()) return;
    HostPrefixTransfer transfer;
    if (transfer.Prepare(*this, context, snapshot, false))
        hostPrefixCache->PublishAsync("local", snapshot->tokens, std::move(transfer.state));
#endif
}

bool NaiveN05FlashModel::TryRestoreHostPrefix(ResponseContext &ctx, int gpuTokens, bool &requery) {
#ifdef USE_CUDA
    if (!hostPrefixCache || !saveHistoryChat || ctx.currentTokens.size() < 2 || ctx.prefillOffset ||
        !ctx.multimodalInput.empty() || ctx.isAbort)
        return false;
    // Leave a suffix to generate logits. FastKV compares the complete token
    // sequence, including when an exact full-length entry also exists.
    const size_t longest = (ctx.currentTokens.size() - 1) / GetPageLen() * GetPageLen();
    if (size_t(std::max(0, gpuTokens)) >= longest) return false;
    std::vector<int> prefix(ctx.currentTokens.begin(), ctx.currentTokens.end() - 1);
    auto hit = hostPrefixCache->Find("local", prefix);
    if (!hit || hit.Tokens() <= size_t(std::max(0, gpuTokens))) return false;
    auto &key = ctx.pastKeyValues.front().first;
    if (!CanAdmitResponseContext(&ctx)) return false;
    auto snap = std::make_shared<PrefixSnapshot>();
    snap->pool = prefixPool;
    snap->length = int(hit.Tokens());
    const int group = WindowGroup(key);
    snap->ownerDevice = PDRequestDevices(ctx).front();
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        snap->slot = TakeHistorySlot(group);
    }
    if (snap->slot < 0) return false;
    requery = true;
    HostPrefixTransfer transfer;
    if (!transfer.Prepare(*this, ctx, snap, true)) return false;
    std::future<fastkv::Message> done;
    // The scheduler owns publication. On failed DMA the private pages/window
    // are released and no partially restored state is attached to the request.
    try {
        if (!hostPrefixCache->RestoreIntoAsync(std::move(hit), std::move(transfer.state.state.buffers), done))
            return false;
        done.get();
    } catch (const std::exception &e) {
        fprintf(stderr, "[FastKV L2] restore skipped: %s\n", e.what());
        return false;
    }
    snap->tokens.assign(prefix.begin(), prefix.begin() + snap->length);
    snap->tokenHash = PagedCacheManager::HashTokenPage(snap->tokens.data(), snap->length);
    snap->draftReady = draftEnabled;
    if (!RestorePrefixWindow(&ctx, snap)) return false;
    transfer.Adopt(*this, ctx, snap->length);
    ctx.cacheLen = snap->length;
    ctx.currentTokens.erase(ctx.currentTokens.begin(), ctx.currentTokens.begin() + ctx.cacheLen);
    if (verbose) {
        std::cout << "[FastKV L2] restored_tokens=" << ctx.cacheLen << " owner=" << group;
        if (hostPrefixSwaBytes >= 0) {
            const auto usage = hostPrefixCache->Stats();
            std::cout << " kv_bytes=" << usage.kvBytes << " swa_bytes=" << usage.swaBytes;
        }
        std::cout << std::endl;
    }
    return true;
#else
    return false;
#endif
}
} // namespace fastllm
