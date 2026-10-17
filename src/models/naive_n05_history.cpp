#include "models/naive_n05_flash.h"
#include <algorithm>
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#endif

namespace fastllm {

bool NaiveN05FlashModel::SetSaveHistoryChat(bool save) {
    std::lock_guard<std::mutex> contextGuard(dictLocker);
    std::lock_guard<std::mutex> guard(historyMutex);
    saveHistoryChat = save;
    if (!save) {
        prefixHistory.clear();
        activePrefixes.clear();
    }
    return true;
}

// This model never archives KV or hidden states in host memory. Unsupported
// layouts, speculative decoding and evicted prefixes safely prefill again.
bool NaiveN05FlashModel::TryRestoreHistoryCache(std::vector<int> &, int &cacheLen) {
    cacheLen = 0;
    return false;
}

void NaiveN05FlashModel::OnResponseContextCreated(ResponseContext *context) {
    tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
    RestoreTensorParallelCache(context);
    std::lock_guard<std::mutex> guard(historyMutex);
    if (saveHistoryChat && !draftEnabled && context->multimodalInput.empty())
        activePrefixes.try_emplace(&context->pastKeyValues.front().first);
}

void NaiveN05FlashModel::OnResponseContextRemoved(ResponseContext *context) {
    tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
    const Data *key = context->pastKeyValues.empty() ? nullptr : &context->pastKeyValues.front().first;
    RecycleTensorParallelCache(context);
    std::lock_guard<std::mutex> guard(historyMutex);
    activePrefixes.erase(key);
    auto draft = draftContexts.find(&context->pastKeyValues);
    if (draft != draftContexts.end()) {
        const auto &s = *draft->second;
        if (verbose && s.rounds)
            std::cout << "[Naive DSpark] rounds=" << s.rounds << " proposed=" << s.proposed
                      << " accepted=" << s.accepted << " acceptance=" << (double)s.accepted / s.proposed
                      << " tokens_per_round=" << 1.0 + (double)s.accepted / s.rounds << std::endl;
        if (s.workspace || !s.kv.empty()) idleDraftContext = std::move(draft->second);
        draftContexts.erase(draft);
    }
}

// Caller holds historyMutex. A retired checkpoint may still be held by a
// forked request; its physical slot is released only after the final user.
void NaiveN05FlashModel::RetirePrefixSnapshot(std::shared_ptr<PrefixSnapshot> &snapshot) {
    if (!snapshot) return;
    prefixHistory.erase(std::remove(prefixHistory.begin(), prefixHistory.end(), snapshot), prefixHistory.end());
    snapshot.reset();
}

NaiveN05FlashModel::PrefixSnapshots NaiveN05FlashModel::BeginPrefixSnapshots(
        const Data &key, int past, int length) {
    PrefixSnapshots result;
    if (length <= 0 || !prefixPool) return result;
    std::lock_guard<std::mutex> guard(historyMutex);
    auto active = activePrefixes.find(&key);
    if (!saveHistoryChat || active == activePrefixes.end()) return result;
    const int page = GetPageLen(), end = past + length;
    // Decode only copies a window at a page boundary. Prefill also retains the
    // preceding boundary so exact prompt repeats can leave a token for logits.
    if (length == 1 && end % page) return result;
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
        // Retire replaceable active checkpoints before asking the completed LRU
        // for space. Never overwrite a snapshot shared with another request.
        if (length > 1 && active->second.prefill.size() + result.size() >= 3) {
            RetirePrefixSnapshot(active->second.prefill.front());
            active->second.prefill.erase(active->second.prefill.begin());
        }
        if (length == 1) RetirePrefixSnapshot(active->second.decode);
        auto takeSlot = [&]() {
            std::lock_guard<std::mutex> lock(prefixPool->mutex);
            if (prefixPool->freeSlots.empty()) return -1;
            int slot = prefixPool->freeSlots.back();
            prefixPool->freeSlots.pop_back();
            return slot;
        };
        int slot = takeSlot();
        if (slot < 0) {
            // Reclaim orphan windows before evicting a usable, recently used
            // checkpoint. Full KV and window pools share no physical copies.
            prefixHistory.erase(std::remove_if(prefixHistory.begin(), prefixHistory.end(),
                [](const auto &entry) { return !entry->HasResidentPrefix(); }), prefixHistory.end());
            slot = takeSlot();
        }
        while (slot < 0 && !prefixHistory.empty()) {
            prefixHistory.erase(prefixHistory.begin());
            slot = takeSlot();
        }
        if (slot < 0) break; // All remaining checkpoints are held by live users.
        auto snapshot = std::make_shared<PrefixSnapshot>();
        snapshot->pool = prefixPool;
        snapshot->slot = slot;
        snapshot->length = boundary;
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

void NaiveN05FlashModel::CopyPrefixWindow(const PrefixSnapshot &snapshot,
        int rank, int layer, const Data &key, const Data &value, int first, bool ring) {
#ifdef USE_CUDA
    const int rows = std::min(snapshot.length, window - 1);
    const size_t base = (size_t)snapshot.slot * snapshot.pool->slotBytes;
    std::vector<void *> dsts;
    std::vector<const void *> srcs;
    std::vector<size_t> sizes;
    for (int part = 0; part < 2; ++part) {
        const Data &cache = part ? value : key;
        AssertInFastLLM(cache.dataDevice == DataDevice::CUDA && !cache.isPagedKVCache &&
            cache.dataType == kvCacheDataType && first >= 0 &&
            (ring || first + rows <= cache.dims[1]), "Invalid GPU prefix window.");
        const size_t rowBytes = (size_t)cache.dims[2] * cache.unitSize;
        auto *target = (uint8_t *)snapshot.pool->storage.at(rank).cudaData + base +
            snapshot.pool->offsets.at(layer * 2 + part);
        int start = ring ? first % window : first;
        int count = ring ? std::min(rows, window - start) : rows;
        dsts.push_back(target); srcs.push_back((uint8_t *)cache.cudaData + start * rowBytes);
        sizes.push_back(count * rowBytes);
        if (count < rows) {
            dsts.push_back(target + count * rowBytes); srcs.push_back(cache.cudaData);
            sizes.push_back((rows - count) * rowBytes);
        }
    }
    AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
        dsts.data(), srcs.data(), sizes.data(), (int)sizes.size()), "GPU prefix copy failed.");
#endif
}

void NaiveN05FlashModel::TryRecordResponseContext(ResponseContext *context) {
    std::lock_guard<std::mutex> guard(historyMutex);
    if (!saveHistoryChat || !prefixPool || !context || context->error != ResponseContextErrorNone ||
        !context->multimodalInput.empty()) return;
    auto active = activePrefixes.find(&context->pastKeyValues.front().first);
    if (active == activePrefixes.end()) return;
    PrefixSnapshots snapshots = active->second.prefill;
    if (active->second.decode) snapshots.push_back(active->second.decode);
    for (const auto &snapshot : snapshots) {
        if ((int)context->allTokens.size() < snapshot->length) continue;
        if (snapshot->tokens.empty())
            snapshot->tokens.assign(context->allTokens.begin(), context->allTokens.begin() + snapshot->length);
        if (!snapshot->fullPool) {
            for (int layer = 0; layer < block_cnt; ++layer) {
                auto *manager = GetPagedKVCacheManager(layer, true);
                if (!manager) continue;
                const Data &root = context->pastKeyValues[layer].first;
                auto local = root.multiDeviceDatas.find(tpDevices.front());
                const Data &cache = local == root.multiDeviceDatas.end() ? root : *local->second;
                const int index = snapshot->length / manager->pageLen - 1;
                if (index >= 0 && index < (int)cache.pageIndex.size()) {
                    std::lock_guard<std::mutex> lock(manager->pageIndexLocker);
                    snapshot->fullPage = cache.pageIndex[index];
                    snapshot->fullGeneration = manager->pageTimestamp[snapshot->fullPage];
                    snapshot->fullPool = manager;
                }
                break;
            }
        }
        // A checkpoint is immutable after publication. Repeated calls refresh
        // its LRU position; an equivalent new checkpoint replaces the old one.
        prefixHistory.erase(std::remove_if(prefixHistory.begin(), prefixHistory.end(), [&](const auto &entry) {
            return entry == snapshot || entry->tokens == snapshot->tokens;
        }), prefixHistory.end());
        prefixHistory.push_back(snapshot);
    }
}

PagedCacheManager *NaiveN05FlashModel::GetPagedKVCacheManager(int layer, bool key) const {
    if (tpPagedCaches.empty() || layer < 0 || layer >= block_cnt || slidingLayers[layer]) return nullptr;
    return tpPagedCaches.front()[layer * 2 + !key].get();
}

std::vector<std::pair<int, PagedCacheManager *>> NaiveN05FlashModel::GetPagedKVCacheManagers(int layer, bool key) const {
    std::vector<std::pair<int, PagedCacheManager *>> result;
    if (layer < 0 || layer >= block_cnt || slidingLayers[layer]) return result;
    for (int rank = 0; rank < (int)tpPagedCaches.size(); ++rank)
        result.emplace_back(tpDevices[rank], tpPagedCaches[rank][layer * 2 + !key].get());
    return result;
}

bool NaiveN05FlashModel::TryRecordPagedPrefixCacheExtra(ResponseContext *context) {
    return saveHistoryChat && prefixPool && context != nullptr;
}

int NaiveN05FlashModel::QueryPagedPrefixCacheExtra(ResponseContext *context, int length) const {
    if (!saveHistoryChat || !context || !prefixPool) return 0;
    std::lock_guard<std::mutex> guard(historyMutex);
    int best = 0;
    for (const auto &entry : prefixHistory) {
        if (entry->length > best && entry->length <= length &&
            entry->HasResidentPrefix() &&
            entry->tokens.size() <= context->currentTokens.size() &&
            std::equal(entry->tokens.begin(), entry->tokens.end(), context->currentTokens.begin()))
            best = entry->length;
    }
    return best;
}

bool NaiveN05FlashModel::RestorePagedPrefixCacheExtra(ResponseContext *context, int length) const {
#ifdef USE_CUDA
    auto *self = const_cast<NaiveN05FlashModel *>(this);
    std::shared_ptr<PrefixSnapshot> snapshot;
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        for (auto it = self->prefixHistory.begin(); it != self->prefixHistory.end(); ++it) {
            if ((*it)->length == length && (*it)->tokens.size() <= context->currentTokens.size() &&
                (*it)->HasResidentPrefix() &&
                std::equal((*it)->tokens.begin(), (*it)->tokens.end(), context->currentTokens.begin())) {
                snapshot = *it;
                std::rotate(it, it + 1, self->prefixHistory.end());
                break;
            }
        }
    }
    if (!snapshot) return false;
    const int rows = std::min(length, window - 1), ranks = tpDevices.size();
    const int capacity = (window + 127) / 128 * 128;
    const int previous = FastllmCudaGetDevice();
    for (int rank = 0; rank < ranks; ++rank) {
        FastllmCudaSetDevice(tpDevices[rank]);
        std::vector<void *> dsts;
        std::vector<const void *> srcs;
        std::vector<size_t> sizes;
        for (int layer = 0; layer < block_cnt; ++layer) {
            if (!slidingLayers[layer]) continue;
            for (int part = 0; part < 2; ++part) {
                Data &root = part ? context->pastKeyValues[layer].second : context->pastKeyValues[layer].first;
                const int dim = part ? sliding.valueDim : sliding.headDim;
                const int width = std::max(1, sliding.kvHeads / ranks) * dim;
                root.multiDeviceData = true;
                root.dataDevice = DataDevice::CUDA;
                root.dataDeviceIds = tpDevices;
                root.Resize({1, rows, sliding.kvHeads * dim});
                root.expansionDims = {1, capacity, sliding.kvHeads * dim};
                Data *&cache = root.multiDeviceDatas[tpDevices[rank]];
                if (!cache) cache = new Data(kvCacheDataType);
                cache->isKVCache = true;
                cache->dataDevice = DataDevice::CUDA;
                cache->dataDeviceIds = {tpDevices[rank]};
                self->ReserveCache(*cache, capacity, width);
                cache->Resize({1, rows, width});
                const size_t rowBytes = (size_t)width * cache->unitSize;
                auto *source = (uint8_t *)snapshot->pool->storage.at(rank).cudaData +
                    (size_t)snapshot->slot * snapshot->pool->slotBytes + snapshot->pool->offsets.at(layer * 2 + part);
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
        ForceDeviceSync(); // The next forward runs on a different rank thread.
    }
    FastllmCudaSetDevice(previous);
    context->preTokens = length;
    context->intParams["add_special_tokens"] = 0;
    context->intParams["promptLen"] = context->inputTokens;
    context->intParams["index"] = -1;
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        self->activePrefixes[&context->pastKeyValues.front().first].prefill = {snapshot};
    }
    self->tpCacheEpoch.fetch_add(1, std::memory_order_relaxed);
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
