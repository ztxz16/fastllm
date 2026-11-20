#include "models/naive_n05_flash.h"
#include <algorithm>
#include <climits>
#include "fastkv/host_cache.h"
#include "utils/inference_trace.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#endif

namespace fastllm {

static std::shared_ptr<fastkv::HostCache> MakeHostPrefixCache(size_t bytes, int64_t swaBytes) {
    fastkv::HostCache::Options options;
    options.bytes = bytes;
    options.swaBytes = swaBytes;
    options.inflightBytes = std::min(bytes, options.inflightBytes);
    return std::make_shared<fastkv::HostCache>(options);
}

bool NaiveN05FlashModel::SetSaveHistoryChat(bool save) {
    std::lock_guard<std::mutex> contextGuard(dictLocker);
    std::lock_guard<std::mutex> guard(historyMutex);
    if (tpPagedCaches.empty()) reserveWindowHistory = save;
    else if (save && !reserveWindowHistory) return false;
    if (save && hostPrefixBytes && !hostPrefixCache) {
        hostPrefixCache = MakeHostPrefixCache(hostPrefixBytes, hostPrefixSwaBytes);
    }
    saveHistoryChat = save;
    if (!save) {
        hostPrefixCache.reset();
        prefixHistory.clear();
        if (prefixPool) ++prefixPool->historyVersion;
        activePrefixes.clear();
    }
    return true;
}

// History reuse uses resident page references and GPU SWA checkpoints.
// Unsupported layouts and evicted prefixes safely prefill again.
bool NaiveN05FlashModel::TryRestoreHistoryCache(std::vector<int> &, int &cacheLen) {
    cacheLen = 0;
    return false;
}

int NaiveN05FlashModel::WindowGroup(const Data &key) const {
    if (attentionDP == 1) return 0;
    AssertInFastLLM(!key.dataDeviceIds.empty(), "Missing attention DP window owner.");
    auto owner = std::find(tpDevices.begin(), tpDevices.end(), key.dataDeviceIds.front());
    AssertInFastLLM(owner != tpDevices.end(), "Invalid attention DP window owner.");
    return (owner - tpDevices.begin()) / AttentionTPSize();
}

bool NaiveN05FlashModel::CanAdmitResponseContext(const ResponseContext *context) const {
    if (!activeWindowSlots) return true;
    const Data &key = context->pastKeyValues.front().first;
    return key.kvStorageOwner || activeWindowSlots->Available(WindowGroup(key));
}

bool NaiveN05FlashModel::ReserveWindow(Data &key, bool request) const {
    if (!activeWindowSlots || key.kvStorageOwner) return true;
    auto lease = std::make_shared<WindowLease>();
    lease->slots = activeWindowSlots;
    lease->id = lease->slots->Take(WindowGroup(key), request);
    if (lease->id < 0) return false;
    key.kvStorageOwner = std::move(lease);
    return true;
}

bool NaiveN05FlashModel::TryAdmitResponseContext(ResponseContext *context) {
    return !activeWindowSlots || ReserveWindow(context->pastKeyValues.front().first);
}

void NaiveN05FlashModel::OnResponseContextCreated(ResponseContext *context) {
    RestoreTensorParallelCache(context);
    std::shared_ptr<PrefixSnapshot> queuedPrefix;
    int group = context->pdPrefillGroup, best = 0;
    const int preferred = tpNextRequestId % attentionDP;
    if (saveHistoryChat && context->multimodalInput.empty()) {
        std::lock_guard<std::mutex> guard(historyMutex);
        for (const auto &entry : prefixHistory) {
            if ((draftEnabled && !entry->draftReady) || entry->length < best ||
                entry->tokens.size() >= context->currentTokens.size() ||
                !entry->HasResidentPrefix() ||
                !std::equal(entry->tokens.begin(), entry->tokens.end(), context->currentTokens.begin())) continue;
            int candidate = 0;
            if (attentionDP > 1) {
                auto owner = std::find(tpDevices.begin(), tpDevices.end(), entry->ownerDevice);
                if (owner == tpDevices.end()) continue;
                candidate = (owner - tpDevices.begin()) / AttentionTPSize();
                if (context->pdPrefillGroup >= 0 && candidate != context->pdPrefillGroup) continue;
            }
            if (entry->length > best || candidate == preferred) {
                group = candidate; best = entry->length; queuedPrefix = entry;
            }
        }
    }
    if (attentionDP > 1) {
        std::vector<std::pair<Data *, Data *>> roots;
        for (auto &layer : context->pastKeyValues) roots.emplace_back(&layer.first, &layer.second);
        // Keep continuations on their longest resident prefix; ties prefer RR.
        if (!roots.front().first->multiDeviceData)
            InitAttentionParallelCache(roots, AllocatePrefillGroup(group));
        else if (queuedPrefix && queuedPrefix->ownerDevice != roots.front().first->dataDeviceIds.front())
            queuedPrefix.reset();
    }
    if (!context->pastKeyValues.empty()) {
        std::lock_guard<std::mutex> guard(tpRequestMutex);
        const Data *key = &context->pastKeyValues.front().first;
        tpRequestIds[key] = ++tpNextRequestId;
        tpSamplingRequestIds[key] = tpNextRequestId;
    }
    std::lock_guard<std::mutex> guard(historyMutex);
    // Register ownership without touching CUDA. Batched prefill initializes
    // draft state on the inference thread, just like single-request prefill.
    if (draftEnabled) draftContexts.try_emplace(&context->pastKeyValues);
    if (saveHistoryChat && context->multimodalInput.empty()) {
        auto &active = activePrefixes[&context->pastKeyValues.front().first];
        // Unbounded API queues must not pin every slot. Each protected waiter
        // uses one of the existing per-request checkpoint slots, not full KV.
        if (activePrefixes.size() <= (size_t)(maxBatch > 0 ? maxBatch : 512))
            active.queued = std::move(queuedPrefix);
    }
}

void NaiveN05FlashModel::OnResponseContextRemoved(ResponseContext *context) {
    const Data *key = context->pastKeyValues.empty() ? nullptr : &context->pastKeyValues.front().first;
    {
        std::lock_guard<std::mutex> guard(tpRequestMutex);
        tpRequestIds.erase(key);
        tpSamplingRequestIds.erase(key);
    }
    // A tentative prefix can be returned to the queue before its first forward.
    // Release its local SWA allocation and admission grant together.
    if (key && key->kvStorageOwner && context->cacheLen && !context->isEnding && !context->isAbort)
        ReleaseRequestWindowStorage(context);
    RecycleTensorParallelCache(context);
    std::lock_guard<std::mutex> guard(historyMutex);
    activePrefixes.erase(key);
    auto draft = draftContexts.find(&context->pastKeyValues);
    if (draft != draftContexts.end() && !draft->second) {
        draftContexts.erase(draft);
        return;
    }
    if (draft != draftContexts.end()) {
        const auto &s = *draft->second;
        if (verbose && s.rounds)
            std::cout << "[Naive DSpark] rounds=" << s.rounds << " proposed=" << s.proposed
                      << " accepted=" << s.accepted << " acceptance=" << (double)s.accepted / s.proposed
                      << " tokens_per_round=" << 1.0 + (double)s.accepted / s.rounds << std::endl;
        if (s.workspace || !s.kv.empty()) {
            if (idleDraftContext && idleDraftContexts.size() + 1 < (size_t)(maxBatch > 0 ? maxBatch : 512))
                idleDraftContexts.push_back(std::move(idleDraftContext));
            idleDraftContext = std::move(draft->second);
        }
        draftContexts.erase(draft);
    }
}

// Caller holds historyMutex. A retired checkpoint may still be held by a
// forked request; its physical slot is released only after the final user.
void NaiveN05FlashModel::RetirePrefixSnapshot(std::shared_ptr<PrefixSnapshot> &snapshot) {
    if (!snapshot) return;
    auto it = std::find(prefixHistory.begin(), prefixHistory.end(), snapshot);
    // A fork may still need this published checkpoint. Dropping its history
    // entry cannot free the slot while another request holds it.
    if (it != prefixHistory.end() && snapshot.use_count() == 2) {
        prefixHistory.erase(it);
        ++prefixPool->historyVersion;
    }
    snapshot.reset();
}

// Caller holds historyMutex. A checkpoint with another reader cannot free its slot.
int NaiveN05FlashModel::TakeHistorySlot(int group) {
    int slot = prefixPool->slots.Take(group);
    for (auto it = prefixHistory.begin(); slot < 0 && it != prefixHistory.end();) {
        if (it->use_count() != 1 || (*it)->slot % attentionDP != group) {
            ++it;
            continue;
        }
        it = prefixHistory.erase(it);
        ++prefixPool->evictedWindows;
        ++prefixPool->historyVersion;
        slot = prefixPool->slots.Take(group);
    }
    return slot;
}

NaiveN05FlashModel::PrefixSnapshots NaiveN05FlashModel::BeginPrefixSnapshots(
        const Data &key, int past, int length) {
    PrefixSnapshots result;
    if (length <= 0 || !prefixPool) return result;
    const int page = GetPageLen(), end = past + length;
    // Decode only copies a window at a page boundary. Prefill also retains the
    // preceding boundary so exact prompt repeats can leave a token for logits.
    if (length == 1 && end % page) return result;
    std::lock_guard<std::mutex> guard(historyMutex);
    auto active = activePrefixes.find(&key);
    if (!saveHistoryChat || active == activePrefixes.end()) return result;
    // Admission either restored this checkpoint into active.prefill or fell
    // back to recomputation. Neither path needs the waiting guard anymore.
    active->second.queued.reset();
    const int group = WindowGroup(key);
    std::vector<int> boundaries;
    const int last = end / page * page;
    if (length > 1 && last == end && last - page >= past && last > page)
        boundaries.push_back(last - page);
    if (last > 0 && last >= past) boundaries.push_back(last);
    for (int boundary : boundaries) {
        bool exists = false;
        for (const auto &entry : active->second.prefill) exists |= entry->length == boundary;
        if (active->second.decode) exists |= active->second.decode->length == boundary;
        if (exists) continue;
        PrefixSnapshots *replaceList = nullptr;
        size_t replaceIndex = 0;
        auto *replace = length == 1 ? &active->second.decode : nullptr;
        if (length > 1 && active->second.prefill.size() + result.size() >= MaxPrefillSnapshots) {
            // Keep the newest boundary and spread the remaining checkpoints.
            // Retaining only the last few pages makes a small tail eviction
            // invalidate every SWA checkpoint of a long resident prefix.
            std::vector<int> lengths;
            for (const auto &entry : active->second.prefill) lengths.push_back(entry->length);
            for (const auto &entry : result) lengths.push_back(entry->length);
            int bestGap = INT_MAX;
            size_t victim = 0;
            for (size_t i = 0; i < lengths.size(); ++i) {
                // Exact page-aligned repeats must leave a token for logits.
                // Keep the preceding boundary as well as the new endpoint.
                if (boundary == end && lengths[i] == end - page) continue;
                std::vector<int> retained{0, boundary};
                for (size_t j = 0; j < lengths.size(); ++j)
                    if (j != i) retained.push_back(lengths[j]);
                std::sort(retained.begin(), retained.end());
                int gap = 0;
                for (size_t j = 1; j < retained.size(); ++j)
                    gap = std::max(gap, retained[j] - retained[j - 1]);
                if (gap < bestGap) { bestGap = gap; victim = i; }
            }
            replaceList = victim < active->second.prefill.size() ? &active->second.prefill : &result;
            replaceIndex = victim < active->second.prefill.size() ? victim : victim - active->second.prefill.size();
            replace = &replaceList->at(replaceIndex);
        }
        auto retire = [&]() {
            if (!replace) return;
            RetirePrefixSnapshot(*replace);
            if (replaceList) replaceList->erase(replaceList->begin() + replaceIndex);
            replace = nullptr;
        };
        int slot = prefixPool->slots.Take(group);
        if (slot < 0 && replace && *replace) {
            const bool published = std::find(prefixHistory.begin(), prefixHistory.end(), *replace) != prefixHistory.end();
            if (replace->use_count() == 1 + (int)published) {
                retire();
                slot = prefixPool->slots.Take(group);
            }
        }
        if (slot < 0) {
            ++prefixPool->historyVersion;
            // Reclaim orphan windows before evicting a usable, recently used
            // checkpoint. Full KV and window pools share no physical copies.
            prefixHistory.erase(std::remove_if(prefixHistory.begin(), prefixHistory.end(),
                [&](const auto &entry) {
                    if (entry->HasResidentPrefix()) return false;
                    ++prefixPool->orphanWindows;
                    return true;
                }), prefixHistory.end());
            slot = prefixPool->slots.Take(group);
        }
        // A continuation pins one checkpoint, but its previous request may
        // have left several more in history. Reuse those slots before LRU
        // eviction can remove every checkpoint of a conversation waiting on
        // a tool. Match actual token prefixes; unrelated branches and every
        // checkpoint held by another request remain protected.
        auto overlapsActivePrefix = [&](const auto &entry) {
            auto overlaps = [&](const auto &held) {
                if (!held || held->tokens.empty() || entry->tokens.empty()) return false;
                const size_t count = std::min(entry->tokens.size(), held->tokens.size());
                return std::equal(entry->tokens.begin(), entry->tokens.begin() + count,
                                  held->tokens.begin());
            };
            return std::any_of(active->second.prefill.begin(), active->second.prefill.end(), overlaps) ||
                   overlaps(active->second.decode);
        };
        for (auto it = prefixHistory.begin(); slot < 0 && it != prefixHistory.end();) {
            if ((*it)->slot % attentionDP != group || it->use_count() != 1 || !overlapsActivePrefix(*it)) { ++it; continue; }
            ++prefixPool->recycledWindows;
            it = prefixHistory.erase(it);
            slot = prefixPool->slots.Take(group);
        }
        if (slot < 0) slot = TakeHistorySlot(group);
        if (slot < 0) { ++prefixPool->noWindowSlots; break; }
        // Keep a shared old checkpoint if no replacement could be allocated.
        retire();
        auto snapshot = std::make_shared<PrefixSnapshot>();
        snapshot->pool = prefixPool;
        snapshot->slot = slot;
        snapshot->length = boundary;
        snapshot->ownerDevice = key.dataDeviceIds.empty() ? -1 : key.dataDeviceIds.front();
        result.push_back(std::move(snapshot));
    }
    return result;
}

void NaiveN05FlashModel::FinishPrefixSnapshots(const Data &key,
        const PrefixSnapshots &snapshots, bool prefill) {
    if (snapshots.empty()) return;
    std::lock_guard<std::mutex> guard(historyMutex);
    auto active = activePrefixes.find(&key);
    if (!saveHistoryChat || active == activePrefixes.end()) return;
    for (const auto &snapshot : snapshots) {
        if (prefill) {
            active->second.prefill.push_back(snapshot);
        } else active->second.decode = snapshot;
    }
}

void NaiveN05FlashModel::PrefixWindowCopies::Submit() const {
#ifdef USE_CUDA
    AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
        destinations.data(), sources.data(), bytes.data(), (int)bytes.size()), "GPU prefix copy failed.");
#endif
}

void NaiveN05FlashModel::AppendPrefixWindowCopies(PrefixWindowCopies &copies, const PrefixSnapshot &snapshot,
        int rank, int layer, const Data &key, const Data &value, int first, bool ring) {
#ifdef USE_CUDA
    const int rows = std::min(snapshot.length, window - 1);
    const size_t base = (size_t)snapshot.pool->slots.Row(snapshot.slot) * snapshot.pool->slotBytes;
    for (int part = 0; part < 2; ++part) {
        const Data &cache = part ? value : key;
        AssertInFastLLM(cache.dataDevice == DataDevice::CUDA && !cache.isPagedKVCache &&
            cache.dataType == AuxiliaryKVType() && first >= 0 &&
            (ring || first + rows <= cache.dims[1]), "Invalid GPU prefix window.");
        const size_t rowBytes = (size_t)cache.dims[2] * cache.unitSize;
        auto *target = (uint8_t *)snapshot.pool->storage.at(rank).cudaData + base +
            snapshot.pool->offsets.at(layer * 2 + part);
        int start = ring ? first % window : first;
        int count = ring ? std::min(rows, window - start) : rows;
        copies.destinations.push_back(target);
        copies.sources.push_back((uint8_t *)cache.cudaData + start * rowBytes);
        copies.bytes.push_back(count * rowBytes);
        if (count < rows) {
            copies.destinations.push_back(target + count * rowBytes);
            copies.sources.push_back(cache.cudaData);
            copies.bytes.push_back((rows - count) * rowBytes);
        }
    }
#endif
}

void NaiveN05FlashModel::CopyPrefixWindow(const PrefixSnapshot &snapshot,
        int rank, int layer, const Data &key, const Data &value, int first, bool ring) {
    PrefixWindowCopies copies;
    AppendPrefixWindowCopies(copies, snapshot, rank, layer, key, value, first, ring);
    copies.Submit();
}

void NaiveN05FlashModel::TryRecordResponseContext(ResponseContext *context) {
    auto *detail = inference_trace::CurrentDetail();
    inference_trace::DetailSpan recordTrace("history.publish_lock", detail);
    std::lock_guard<std::mutex> guard(historyMutex);
    recordTrace.Next("history.publish_setup");
    if (!saveHistoryChat || !prefixPool || !context || context->error != ResponseContextErrorNone ||
        !context->multimodalInput.empty()) return;
    auto active = activePrefixes.find(&context->pastKeyValues.front().first);
    if (active == activePrefixes.end()) return;
    ++prefixPool->historyVersion;
    PrefixSnapshots snapshots = active->second.prefill;
    if (active->second.decode) snapshots.push_back(active->second.decode);
    if (detail) {
        detail->Count("history.snapshots", snapshots.size());
        detail->Count("history.entries_before_publish", prefixHistory.size());
    }
    // Several checkpoints share the same immutable token prefix. Hash it
    // once in increasing length order, retaining the exact uint64 polynomial
    // and token vectors used by collision checks and published snapshots.
    recordTrace.Next("history.snapshot_hashes");
    std::map<int, uint64_t> tokenHashes;
    for (const auto &snapshot : snapshots)
        if (snapshot->tokens.empty() && (int)context->allTokens.size() >= snapshot->length)
            tokenHashes.emplace(snapshot->length, 0);
    size_t hashed = 0;
    uint64_t tokenHash = 0;
    for (auto &entry : tokenHashes) {
        for (; hashed < (size_t)entry.first; ++hashed)
            tokenHash = tokenHash * PagedCacheManager::TokenHashMultiplier +
                        (uint64_t)context->allTokens[hashed];
        entry.second = tokenHash;
    }
    for (const auto &snapshot : snapshots) {
        if (draftEnabled && !snapshot->draftReady) continue;
        recordTrace.Next("history.snapshot_copy_hash");
        if ((int)context->allTokens.size() < snapshot->length) continue;
        if (snapshot->tokens.empty()) {
            if (detail) detail->Count("history.copied_tokens", snapshot->length);
            snapshot->tokens.assign(context->allTokens.begin(), context->allTokens.begin() + snapshot->length);
            snapshot->tokenHash = tokenHashes.at(snapshot->length);
        }
        recordTrace.Next("history.snapshot_pool");
        if (!snapshot->fullPool) {
            for (int layer = 0; layer < block_cnt; ++layer) {
                for (int part = 0; part < 2; ++part) {
                    const Data &root = part ? context->pastKeyValues[layer].second : context->pastKeyValues[layer].first;
                    for (auto [device, manager] : GetPagedKVCacheManagersForRequest(layer, !part, context)) {
                        if (!manager) continue;
                        auto local = root.multiDeviceDatas.find(device);
                        const Data &cache = local == root.multiDeviceDatas.end() ? root : *local->second;
                        const int index = snapshot->length / manager->pageLen - 1;
                        if (index < 0 || index >= (int)cache.pageIndex.size()) continue;
                        const int page = cache.pageIndex[index];
                        long long generation;
                        {
                            std::lock_guard<std::mutex> lock(manager->pageIndexLocker);
                            generation = manager->pageTimestamp[page];
                        }
                        // Register every layer/rank so allocation and preview
                        // use the same policy across all TP cache pools.
                        snapshot->checkpointPages.push_back({manager, page, generation});
                        manager->RegisterPrefixCheckpoint(page, generation);
                        if (!snapshot->fullPool) {
                            snapshot->fullPool = manager;
                            snapshot->fullPage = page;
                            snapshot->fullGeneration = generation;
                        }
                    }
                }
            }
        }
        // A checkpoint is immutable after publication. Repeated calls refresh
        // its LRU position. Keep equivalent checkpoints held by live/queued
        // requests discoverable even if the replacement is later evicted.
        recordTrace.Next("history.dedup_publish");
        prefixHistory.erase(std::remove_if(prefixHistory.begin(), prefixHistory.end(),
            [&](const auto &entry) {
            return entry == snapshot || (entry.use_count() == 1 &&
                                         entry->ownerDevice == snapshot->ownerDevice &&
                                         entry->tokenHash == snapshot->tokenHash &&
                                         entry->tokens == snapshot->tokens);
        }), prefixHistory.end());
        prefixHistory.push_back(snapshot);
        if (hostPrefixCache && (snapshot == active->second.decode ||
            (!active->second.prefill.empty() && snapshot == active->second.prefill.back()) ||
            (snapshot->length < context->inputTokens && snapshot->length + GetPageLen() == context->inputTokens))) {
            try { PublishHostPrefix(*context, snapshot); }
            catch (const std::exception &e) { fprintf(stderr, "[FastKV L2] optional backup skipped: %s\n", e.what()); }
        }
    }
    if (verbose) {
        PagedCacheManager::CacheStatistics stats;
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (auto *pool = GetPagedKVCacheManager(layer, true)) { stats = pool->CacheStats(); break; }
        }
        std::cout << "[Naive prefix cache] full_capacity_pages=" << stats.capacityPages
                  << " full_free_pages=" << stats.freePages << " full_unused_pages=" << stats.unusedCachedPages
                  << " full_evicted_pages=" << stats.pressureEvictedPages
                  << " window_entries=" << prefixHistory.size()
                  << " window_evictions=" << prefixPool->evictedWindows
                  << " orphan_windows=" << prefixPool->orphanWindows
                  << " recycled_windows=" << prefixPool->recycledWindows
                  << " no_window_slots=" << prefixPool->noWindowSlots << std::endl;
    }
}

PagedCacheManager *NaiveN05FlashModel::GetPagedKVCacheManager(int layer, bool key) const {
    if (tpPagedCaches.empty() || layer < 0 || layer >= block_cnt || slidingLayers[layer]) return nullptr;
    return tpPagedCaches.front()[layer * 2 + !key].get();
}

int NaiveN05FlashModel::GetPrefillBatchGroup(const ResponseContext *context) const {
    if (attentionDP <= 1 || !context || context->pastKeyValues.empty()) return 0;
    const auto &devices = context->pastKeyValues.front().first.dataDeviceIds;
    if (devices.empty()) return 0;
    const auto owner = std::find(tpDevices.begin(), tpDevices.end(), devices.front());
    return owner == tpDevices.end() ? 0 : (owner - tpDevices.begin()) / AttentionTPSize();
}

PagedCacheManager *NaiveN05FlashModel::GetPagedKVCacheBudgetManager(const ResponseContext *context) const {
    if (attentionDP == 1 || !context || context->pastKeyValues.empty())
        return GetPagedKVCacheManager(kvCacheId, true);
    const auto &devices = context->pastKeyValues.front().first.dataDeviceIds;
    if (devices.empty()) return nullptr;
    auto it = std::find(tpDevices.begin(), tpDevices.end(), devices.front());
    if (it == tpDevices.end() || tpPagedCaches.empty()) return nullptr;
    return tpPagedCaches[it - tpDevices.begin()][kvCacheId * 2].get();
}

std::vector<std::pair<int, PagedCacheManager *>> NaiveN05FlashModel::GetPagedKVCacheManagers(int layer, bool key) const {
    std::vector<std::pair<int, PagedCacheManager *>> result;
    if (layer < 0 || layer >= block_cnt || slidingLayers[layer]) return result;
    for (int rank = 0; rank < (int)tpPagedCaches.size(); ++rank)
        result.emplace_back(tpDevices[rank], tpPagedCaches[rank][layer * 2 + !key].get());
    return result;
}

std::vector<std::pair<int, PagedCacheManager *>> NaiveN05FlashModel::GetPagedKVCacheManagersForRequest(
        int layer, bool key, const ResponseContext *context) const {
    auto refs = GetPagedKVCacheManagers(layer, key);
    if (attentionDP == 1 || !context || context->pastKeyValues.empty()) return refs;
    const auto &devices = context->pastKeyValues.front().first.dataDeviceIds;
    refs.erase(std::remove_if(refs.begin(), refs.end(), [&](const auto &ref) {
        return std::find(devices.begin(), devices.end(), ref.first) == devices.end();
    }), refs.end());
    return refs;
}

bool NaiveN05FlashModel::TryRecordPagedPrefixCacheExtra(ResponseContext *context) {
    return saveHistoryChat && prefixPool && context != nullptr;
}

int NaiveN05FlashModel::QueryPagedPrefixCacheExtra(ResponseContext *context, int length) const {
    if (!saveHistoryChat || !context || !prefixPool) return 0;
    PrefillPrefixPreview temporary;
    auto &preview = context->prefillPreview ? *context->prefillPreview : temporary;
    preview.Reset(context->currentTokens);
    return PreviewPagedPrefixCacheExtra(context, length, preview);
}

int NaiveN05FlashModel::PreviewPagedPrefixCacheExtra(ResponseContext *context, int length,
                                                   PrefillPrefixPreview &preview) const {
    if (!saveHistoryChat || !context || !prefixPool) return 0;
    struct PrefixPreview {
        struct Match { std::weak_ptr<PrefixSnapshot> snapshot; bool equal = false; };
        std::unordered_map<const PrefixSnapshot *, Match> matches;
        std::vector<std::weak_ptr<PrefixSnapshot>> candidates;
        uint64_t version = 0;
        bool initialized = false;
        int ownerDevice = -1;
    };
    if (!preview.modelState) preview.modelState = std::make_shared<PrefixPreview>();
    auto &memo = *std::static_pointer_cast<PrefixPreview>(preview.modelState);
    auto *detail = inference_trace::CurrentDetail();
    inference_trace::DetailSpan previewTrace("history.preview_lock", detail);
    std::lock_guard<std::mutex> guard(historyMutex);
    previewTrace.Next("history.memo_rebuild");
    // Preview metadata stays weak. Request creation may retain one bounded
    // SWA checkpoint; full-KV residency/generation is still checked each pass.
    // PD routing previews run before request KV descriptors are created.
    int ownerDevice = -1;
    if (!context->pastKeyValues.empty() && !context->pastKeyValues.front().first.dataDeviceIds.empty())
        ownerDevice = context->pastKeyValues.front().first.dataDeviceIds.front();
    if (!memo.initialized || memo.version != prefixPool->historyVersion || memo.ownerDevice != ownerDevice) {
        if (detail) {
            detail->Count("history.memo_misses");
            detail->Count("history.entries_scanned", prefixHistory.size());
        }
        if (memo.matches.size() > prefixHistory.size() * 2 + 16) {
            for (auto it = memo.matches.begin(); it != memo.matches.end();)
                if (it->second.snapshot.expired()) it = memo.matches.erase(it); else ++it;
        }
        memo.candidates.clear();
        size_t fingerprintRejects = 0;
        for (const auto &entry : prefixHistory) {
            if ((attentionDP > 1 && entry->ownerDevice != ownerDevice) ||
                (draftEnabled && !entry->draftReady) || entry->tokens.size() > preview.tokens.size()) continue;
            // Most history belongs to unrelated conversations. Reject it with
            // the immutable fingerprint before allocating a memo entry or
            // touching a weak_ptr control block. Equality still checks tokens.
            if (entry->tokenHash != preview.TokenHash(entry->tokens.size())) {
                ++fingerprintRejects;
                continue;
            }
            auto &match = memo.matches[entry.get()];
            if (match.snapshot.lock() != entry) {
                if (detail) detail->Count("history.new_match_checks");
                match.snapshot = entry;
                match.equal = std::equal(entry->tokens.begin(), entry->tokens.end(), preview.tokens.begin());
            }
            if (match.equal) memo.candidates.push_back(entry);
        }
        if (detail) detail->Count("history.fingerprint_rejects", fingerprintRejects);
        memo.version = prefixPool->historyVersion;
        memo.ownerDevice = ownerDevice;
        memo.initialized = true;
    }
    int best = 0;
    previewTrace.Next("history.residency_check");
    if (detail) detail->Count("history.candidates_checked", memo.candidates.size());
    for (const auto &candidate : memo.candidates) {
        auto entry = candidate.lock();
        if (entry && entry->length > best && entry->length <= length && entry->HasResidentPrefix())
            best = entry->length;
    }
    return best;
}

bool NaiveN05FlashModel::RestorePagedPrefixCacheExtra(ResponseContext *context, int length) const {
#ifdef USE_CUDA
    auto *self = const_cast<NaiveN05FlashModel *>(this);
    std::shared_ptr<PrefixSnapshot> snapshot;
    if (length < 0 || (size_t)length > context->currentTokens.size()) return false;
    const uint64_t tokenHash = PagedCacheManager::HashTokenPage(context->currentTokens.data(), length);
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        for (auto it = self->prefixHistory.begin(); it != self->prefixHistory.end(); ++it) {
            if ((attentionDP == 1 || (*it)->ownerDevice == context->pastKeyValues.front().first.dataDeviceIds.front()) &&
                (!draftEnabled || (*it)->draftReady) && (*it)->length == length && (*it)->tokenHash == tokenHash &&
                (*it)->tokens.size() <= context->currentTokens.size() &&
                (*it)->HasResidentPrefix() &&
                std::equal((*it)->tokens.begin(), (*it)->tokens.end(), context->currentTokens.begin())) {
                snapshot = *it;
                std::rotate(it, it + 1, self->prefixHistory.end());
                break;
            }
        }
    }
    return snapshot && RestorePrefixWindow(context, snapshot);
#else
    return false;
#endif
}

bool NaiveN05FlashModel::RestorePrefixWindow(ResponseContext *context,
        const std::shared_ptr<PrefixSnapshot> &snapshot) const {
#ifdef USE_CUDA
    auto *self = const_cast<NaiveN05FlashModel *>(this);
    const int length = snapshot->length;
    // A queued hit pins only the immutable checkpoint and page references.
    // Its private hot/SWA slot is attached when the scheduler admits it.
    if (!ReserveWindow(context->pastKeyValues.front().first)) return false;
    self->InvalidateTensorParallelCache(&context->pastKeyValues.front().first);
    if (!sparseKv) {
        const int rows = std::min(length, window - 1), ranks = tpDevices.size();
        const int capacity = tpSwaCacheCapacity ? tpSwaCacheCapacity :
            (window + (draftEnabled ? draftBlock : 0) + 127) / 128 * 128;
        const auto devices = attentionDP > 1 ? context->pastKeyValues.front().first.dataDeviceIds : tpDevices;
        // Set up shared descriptors on the scheduler thread before rank workers
        // touch their disjoint local caches. Do not mutate the root maps in parallel.
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (!slidingLayers[layer]) continue;
            for (int part = 0; part < 2; ++part) {
                Data &root = part ? context->pastKeyValues[layer].second : context->pastKeyValues[layer].first;
                const int dim = part ? sliding.valueDim : sliding.headDim;
                root.multiDeviceData = true;
                root.dataDevice = DataDevice::CUDA;
                root.dataDeviceIds = devices;
                root.dataType = AuxiliaryKVType(); root.UpdateUnitSize();
                root.Resize({1, rows, sliding.kvHeads * dim});
                root.expansionDims = {1, capacity, sliding.kvHeads * dim};
                for (int device : devices) {
                    Data *&cache = root.multiDeviceDatas[device];
                    if (!cache) cache = new Data(AuxiliaryKVType());
                }
            }
        }
        std::vector<std::exception_ptr> errors(ranks);
        self->tpWorkers.Run(tpDevices, [&](int rank) {
            if (std::find(devices.begin(), devices.end(), tpDevices[rank]) == devices.end()) return;
            FastllmCudaSetDevice(tpDevices[rank]);
            std::vector<void *> dsts;
            std::vector<const void *> srcs;
            std::vector<size_t> sizes;
            for (int layer = 0; layer < block_cnt; ++layer) {
                if (!slidingLayers[layer]) continue;
                for (int part = 0; part < 2; ++part) {
                    Data &root = part ? context->pastKeyValues[layer].second : context->pastKeyValues[layer].first;
                    const int dim = part ? sliding.valueDim : sliding.headDim;
                    const int width = std::max(1, sliding.kvHeads / AttentionTPSize()) * dim;
                    Data *cache = root.multiDeviceDatas.at(tpDevices[rank]);
                    cache->isKVCache = true;
                    cache->dataDevice = DataDevice::CUDA;
                    cache->dataDeviceIds = {tpDevices[rank]};
                    self->ReserveCache(*cache, capacity, width, true, &context->pastKeyValues.front().first);
                    cache->Resize({1, rows, width});
                    const size_t rowBytes = (size_t)width * cache->unitSize;
                    auto *source = (uint8_t *)snapshot->pool->storage.at(rank).cudaData +
                        (size_t)snapshot->pool->slots.Row(snapshot->slot) * snapshot->pool->slotBytes + snapshot->pool->offsets.at(layer * 2 + part);
                    const int first = tpSwaRingCache ? (length - rows) % window : 0;
                    const int count = tpSwaRingCache ? std::min(rows, window - first) : rows;
                    dsts.push_back((uint8_t *)cache->cudaData + first * rowBytes);
                    srcs.push_back(source); sizes.push_back(count * rowBytes);
                    if (count < rows) {
                        dsts.push_back(cache->cudaData); srcs.push_back(source + count * rowBytes);
                        sizes.push_back((rows - count) * rowBytes);
                    }
                }
            }
            AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                dsts.data(), srcs.data(), sizes.data(), (int)sizes.size()), "GPU prefix restore failed.");
            // Copies run concurrently on the persistent rank streams. Complete
            // them before releasing the snapshot or allowing request cancellation
            // to recycle destination storage; no device-wide barrier is needed.
            FastllmCudaSyncCurrentThreadStream();
        }, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
    }
    if (draftEnabled) {
        // Reconstruct the draft's bounded KV from the exact normalized inputs
        // at this checkpoint, including absolute RoPE positions. Proposal/RNG
        // state belongs to the new request and must never be inherited.
        std::shared_ptr<DraftContext> draft;
        {
            std::lock_guard<std::mutex> guard(historyMutex);
            draft = self->CreateDraftContext();
            self->draftContexts[&context->pastKeyValues] = draft;
        }
        self->RestoreDraftPrefix(*snapshot, length, *draft);
    }
    context->preTokens = length;
    context->intParams["add_special_tokens"] = 0;
    context->intParams["promptLen"] = context->inputTokens;
    context->intParams["index"] = -1;
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        auto &active = self->activePrefixes[&context->pastKeyValues.front().first];
        active.prefill = {snapshot};
        active.queued.reset();
    }
    return true;
#else
    return false;
#endif
}

void NaiveN05FlashModel::AddPromptCache(const std::vector<int> &tokens) {
    if (tokens.empty() || !saveHistoryChat) return;
    GenerationConfig config;
    config.output_token_limit = 1;
    config.add_special_tokens = false;
    int handle = LaunchResponseTokens(tokens, config);
    while (FetchResponseTokens(handle) >= 0) {}
}
}

#include "naive_n05_host_prefix.inc"
