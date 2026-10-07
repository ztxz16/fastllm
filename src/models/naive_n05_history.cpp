#include "models/naive_n05_flash.h"
#include <algorithm>
#include <cstring>
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace fastllm {

bool NaiveN05FlashModel::SetSaveHistoryChat(bool save) {
    std::lock_guard<std::mutex> contextGuard(dictLocker);
    std::lock_guard<std::mutex> guard(historyMutex);
    saveHistoryChat = save;
    if (!save) {
        history.clear();
        pendingHistory.reset();
        activeHistory.clear();
    }
    return true;
}

std::shared_ptr<NaiveN05FlashModel::HistoryChunk> NaiveN05FlashModel::BeginHistoryChunk(
        const std::vector<std::pair<Data, Data>> &kv, int past, int length) {
    return BeginHistoryChunk(kv.front().first, past, length);
}

std::shared_ptr<NaiveN05FlashModel::HistoryChunk> NaiveN05FlashModel::BeginHistoryChunk(
        const Data &key, int past, int length) {
    std::lock_guard<std::mutex> guard(historyMutex);
    auto it = activeHistory.find(&key);
    if (!saveHistoryChat || it == activeHistory.end() ||
        it->second.length != past || historyBytesPerToken == 0) return nullptr;
    auto &state = it->second;
    int rows = std::min<size_t>(length, (historyRecordByteLimit - state.bytes) / historyBytesPerToken);
    if (rows <= 0) return nullptr;
    auto chunk = std::make_shared<HistoryChunk>();
    chunk->length = rows;
    chunk->bytes = rows * historyBytesPerToken;
    chunk->layers.resize(block_cnt);
    if (!tpDevices.empty()) {
        // Ranks write disjoint KV-head columns into one canonical host archive.
        // Replicated KV heads and index keys are stored only once.
        for (int layer = 0; layer < block_cnt; ++layer) {
            const auto &cfg = slidingLayers[layer] ? sliding : full;
            for (int part = 0; part < 2; ++part) {
                Data &dst = part ? chunk->layers[layer].second : chunk->layers[layer].first;
                int width = part ? cfg.kvHeads * cfg.valueDim :
                    cfg.kvHeads * cfg.headDim + (slidingLayers[layer] ? 0 : indexDim);
                dst.dataType = BFLOAT16;
                dst.Resize({1, rows, width});
                dst.Allocate(false);
                dst.lockInCPU = true;
            }
        }
    }
    return chunk;
}

void NaiveN05FlashModel::CopyHistoryTensor(const Data &source, Data &target, int length) {
    AssertInFastLLM(source.dataType == BFLOAT16 && source.dims.size() == 3 &&
                   source.dims[0] == 1 && length > 0 && length <= source.dims[1],
                   "Invalid Naive history tensor.");
    target.dataType = source.dataType;
    target.UpdateUnitSize();
    target.Resize({1, length, source.dims[2]});
    target.Allocate();
    target.lockInCPU = true;
    size_t bytes = (size_t)length * source.dims[2] * sizeof(uint16_t);
    if (source.dataDevice == DataDevice::CPU) {
        std::memcpy(target.cpuData, source.cpuData, bytes);
    } else {
#ifdef USE_CUDA
        int previous = FastllmCudaGetDevice();
        FastllmCudaSetDevice(GetPointerDeviceId(source.cudaData));
        FastllmCudaCopyFromDeviceToHost(target.cpuData, source.cudaData, bytes);
        FastllmCudaSetDevice(previous);
#else
        ErrorInFastLLM("CUDA history tensor in a CPU build.");
#endif
    }
}

void NaiveN05FlashModel::CopyTensorParallelHistory(const Data &source, Data &target,
        int length, int layer, int rank, bool value) {
    const auto &cfg = slidingLayers[layer] ? sliding : full;
    const int ranks = tpDevices.size();
    if (ranks > cfg.kvHeads && rank % (ranks / cfg.kvHeads)) return;
    const int headDim = value ? cfg.valueDim : cfg.headDim;
    const int width = std::max(1, cfg.kvHeads / ranks) * headDim;
    const int begin = rank * cfg.kvHeads / ranks * headDim;
    const int index = !value && !slidingLayers[layer] ? indexDim : 0;
    AssertInFastLLM(target.dims == std::vector<int>({1, length, cfg.kvHeads * headDim + index}),
                    "Naive TP: invalid host history layout.");
    Data host;
    if (source.dataDevice != DataDevice::CPU) CopyHistoryTensor(source, host, length);
    const auto *data = source.dataDevice == DataDevice::CPU ? source.cpuData : host.cpuData;
    for (int row = 0; row < length; ++row) {
        const uint16_t *src = (const uint16_t *)data + (size_t)row * (width + index);
        uint16_t *dst = (uint16_t *)target.cpuData + (size_t)row * target.dims[2];
        std::memcpy(dst + begin, src, width * sizeof(uint16_t));
        if (index && rank == 0)
            std::memcpy(dst + cfg.kvHeads * headDim, src + width, index * sizeof(uint16_t));
    }
}

void NaiveN05FlashModel::RestoreTensorParallelHistoryRank(Data &cache, int layer,
        bool value, int capacity, int rank) {
#ifdef USE_CUDA
    AssertInFastLLM(cache.dims.size() == 3, "Naive TP: invalid restored cache dimensions.");
    const auto &cfg = slidingLayers[layer] ? sliding : full;
    const int ranks = tpDevices.size(), length = cache.dims[1];
    const int headDim = value ? cfg.valueDim : cfg.headDim;
    const int width = std::max(1, cfg.kvHeads / ranks) * headDim;
    const int index = !value && !slidingLayers[layer] ? indexDim : 0;
    AssertInFastLLM(cache.dataDevice == DataDevice::CPU && cache.dataType == BFLOAT16 &&
                    cache.dims == std::vector<int>({1, length, cfg.kvHeads * headDim + index}) &&
                    cache.cpuData && !cache.isFake && !cache.isPagedKVCache,
                    "Naive TP: invalid restored host history cache.");
    const int device = tpDevices[rank], begin = rank * cfg.kvHeads / ranks * headDim;
    FastllmCudaSetDevice(device);
    Data host(BFLOAT16, {1, length, width + index});
    host.Allocate(false);
    for (int row = 0; row < length; ++row) {
        const uint16_t *src = (const uint16_t *)cache.cpuData + (size_t)row * cache.dims[2];
        uint16_t *dst = (uint16_t *)host.cpuData + (size_t)row * (width + index);
        std::memcpy(dst, src + begin, width * sizeof(uint16_t));
        if (index) std::memcpy(dst + width, src + cfg.kvHeads * headDim, index * sizeof(uint16_t));
    }
    auto *local = cache.multiDeviceDatas.at(device);
    const int required = (std::max(length, capacity) + 127) / 128 * 128;
    // Expansion reallocates even when its requested capacity is unchanged.
    // Preserve adopted allocations and graph addresses when they already fit.
    if (local->expansionDims.size() != 3 || local->expansionDims[1] < required) {
        if (!local->dims.empty()) local->Resize({1, 0, width + index});
        local->Expansion({1, required, width + index});
    }
    local->Resize(host.dims);
    FastllmCudaCopyFromHostToDevice(local->cudaData, host.cpuData, host.GetBytes());
#endif
}

void NaiveN05FlashModel::FinishHistoryChunk(
        const std::vector<std::pair<Data, Data>> &kv,
        const std::shared_ptr<HistoryChunk> &chunk) {
    FinishHistoryChunk(kv.front().first, chunk);
}

void NaiveN05FlashModel::FinishHistoryChunk(const Data &key,
        const std::shared_ptr<HistoryChunk> &chunk) {
    if (!chunk) return;
    std::lock_guard<std::mutex> guard(historyMutex);
    auto it = activeHistory.find(&key);
    if (it == activeHistory.end()) return;
    auto &state = it->second;
    state.spans.push_back({chunk, chunk->length});
    state.length += chunk->length;
    state.bytes += chunk->bytes;
}

bool NaiveN05FlashModel::TryRestoreHistoryCache(std::vector<int> &tokens, int &cacheLen) {
    std::lock_guard<std::mutex> guard(historyMutex);
    pendingHistory.reset();
    cacheLen = 0;
    if (!saveHistoryChat || tokens.size() <= 1) return false;
    auto best = history.end();
    for (auto it = history.begin(); it != history.end(); ++it) {
        const auto &entry = *it;
        // Keep one input token for the logits projection, including an exact
        // repeat or a request shorter than the recorded conversation.
        int limit = std::min(entry->tokens.size(), tokens.size() - 1);
        int common = 0;
        while (common < limit && entry->tokens[common] == tokens[common]) ++common;
        if (common > cacheLen) {
            cacheLen = common;
            best = it;
        }
    }
    if (best == history.end()) return false;
    pendingHistory = *best;
    std::rotate(best, best + 1, history.end());
    tokens.erase(tokens.begin(), tokens.begin() + cacheLen);
    return true;
}

void NaiveN05FlashModel::OnResponseContextCreated(ResponseContext *context) {
    std::shared_ptr<const HistoryMemory> pending;
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        pending.swap(pendingHistory);
        if (!saveHistoryChat || !context->multimodalInput.empty()) {
            RestoreTensorParallelCache(context);
            return;
        }
    }
    HistoryMemory state;
    if (pending) {
        // The new request owns these CPU tensors. Restoring immutable host
        // chunks does not use the shared CUDA workspace or forwardLocker.
        int remaining = context->cacheLen;
        for (const auto &span : pending->spans) {
            if (remaining == 0) break;
            int length = std::min(remaining, span.length);
            state.spans.push_back({span.chunk, length});
            state.length += length;
            state.bytes += span.chunk->bytes;
            remaining -= length;
        }
        AssertInFastLLM(remaining == 0, "Incomplete Naive history archive.");
        if (draftEnabled) {
            std::shared_ptr<DraftContext> draft;
            {
                std::lock_guard<std::mutex> guard(historyMutex);
                draft = CreateDraftContext();
            }
            draft->committed = state.length;
            int first = std::max(0, state.length - draftWindow + 1);
            draft->restoredHidden = Data(BFLOAT16, {1, state.length - first, embed_dim});
            draft->restoredHidden.Allocate();
            int offset = 0;
            for (const auto &span : state.spans) {
                int begin = std::max(first, offset), end = offset + span.length;
                if (begin < end) {
                    const Data &source = span.chunk->draftHidden;
                    AssertInFastLLM(source.dims.size() == 3 && source.dims[1] >= span.length,
                                    "Missing Naive draft history features.");
                    std::memcpy(draft->restoredHidden.cpuData + (size_t)(begin - first) * embed_dim * 2,
                                source.cpuData + (size_t)(begin - offset) * embed_dim * 2,
                                (size_t)(end - begin) * embed_dim * 2);
                }
                offset = end;
            }
            std::lock_guard<std::mutex> guard(historyMutex);
            draftContexts[&context->pastKeyValues] = std::move(draft);
        }
        for (int layer = 0; layer < block_cnt; ++layer) {
            int first = slidingLayers[layer] ? std::max(0, state.length - window + 1) : 0;
            int length = state.length - first;
            for (int part = 0; part < 2; ++part) {
                Data &target = part ? context->pastKeyValues[layer].second : context->pastKeyValues[layer].first;
                const auto &initial = state.spans.front().chunk->layers[layer];
                int width = (part ? initial.second : initial.first).dims[2];
                target.dataType = BFLOAT16;
                target.UpdateUnitSize();
                target.Expansion({1, (length + 127) / 128 * 128, width});
                target.Resize({1, length, width});
                int offset = 0;
                for (const auto &span : state.spans) {
                    int begin = std::max(first, offset), end = offset + span.length;
                    if (begin < end) {
                        const auto &pair = span.chunk->layers[layer];
                        const Data &source = part ? pair.second : pair.first;
                        std::memcpy(target.cpuData + (size_t)(begin - first) * width * sizeof(uint16_t),
                                    source.cpuData + (size_t)(begin - offset) * width * sizeof(uint16_t),
                                    (size_t)(end - begin) * width * sizeof(uint16_t));
                    }
                    offset = end;
                }
            }
        }
        // Restored contiguous KV already counts as an active sequence in the
        // legacy scheduler. Seed positions so it evaluates the uncached suffix.
        context->preTokens = context->cacheLen;
        context->intParams["add_special_tokens"] = 0;
        context->intParams["promptLen"] = context->inputTokens;
        context->intParams["index"] = -1;
    }
    RestoreTensorParallelCache(context);
    std::lock_guard<std::mutex> guard(historyMutex);
    activeHistory[&context->pastKeyValues.front().first] = std::move(state);
}

void NaiveN05FlashModel::OnResponseContextRemoved(ResponseContext *context) {
    const Data *key = context->pastKeyValues.empty() ? nullptr : &context->pastKeyValues.front().first;
    RecycleTensorParallelCache(context);
    std::lock_guard<std::mutex> guard(historyMutex);
    activeHistory.erase(key);
    auto draft = draftContexts.find(&context->pastKeyValues);
    if (draft != draftContexts.end()) {
        const auto &s = *draft->second;
        if (verbose && s.rounds)
            std::cout << "[Naive DSpark] rounds=" << s.rounds << " proposed=" << s.proposed
                      << " accepted=" << s.accepted << " acceptance=" << (double)s.accepted / s.proposed
                      << " tokens_per_round=" << 1.0 + (double)s.accepted / s.rounds << std::endl;
        if (s.workspace || !s.kv.empty())
            idleDraftContext = std::move(draft->second);
        draftContexts.erase(draft);
    }
}

void NaiveN05FlashModel::TryRecordResponseContext(ResponseContext *context) {
    std::lock_guard<std::mutex> guard(historyMutex);
    if (!saveHistoryChat || !context || !context->multimodalInput.empty()) return;
    auto active = activeHistory.find(&context->pastKeyValues.front().first);
    if (active == activeHistory.end() || active->second.length <= 0) return;
    // A speculative block may have committed KV ahead of the scheduler when
    // a request stops or is cancelled. Publish only the emitted prefix.
    int length = std::min<int>(active->second.length, context->allTokens.size());
    if (length <= 0) return;
    // allTokens may include the last sampled token, whose KV has not run yet.
    auto tokenEnd = context->allTokens.begin() + length;
    for (auto it = history.begin(); it != history.end(); ++it) {
        const auto &entry = *it;
        if (entry->tokens.size() >= length &&
            std::equal(context->allTokens.begin(), tokenEnd, entry->tokens.begin())) {
            std::rotate(it, it + 1, history.end());
            return;
        }
    }
    auto memory = std::make_shared<HistoryMemory>(active->second);
    memory->length = length;
    memory->bytes = 0;
    int remaining = length;
    size_t count = 0;
    for (auto &span : memory->spans) {
        if (!remaining) break;
        span.length = std::min(span.length, remaining);
        remaining -= span.length;
        memory->bytes += span.chunk->bytes;
        ++count;
    }
    memory->spans.resize(count);
    memory->tokens.assign(context->allTokens.begin(), tokenEnd);
    // A descendant covers every prefix of its ancestor, sharing the same
    // immutable chunks. Retiring the ancestor avoids duplicate LRU accounting.
    history.erase(std::remove_if(history.begin(), history.end(), [&](const auto &entry) {
        return entry->tokens.size() < memory->tokens.size() &&
            std::equal(entry->tokens.begin(), entry->tokens.end(), memory->tokens.begin());
    }), history.end());
    history.push_back(std::move(memory));
    // BeginHistoryChunk bounds each record; limiting the count also bounds
    // the completed archives to historyRecordLimit * historyRecordByteLimit.
    while (history.size() > historyRecordLimit) {
        history.erase(history.begin());
    }
}

void NaiveN05FlashModel::AddPromptCache(const std::vector<int> &tokens) {
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        if (tokens.empty() || !saveHistoryChat) return;
    }
    GenerationConfig config;
    config.output_token_limit = 1;
    config.add_special_tokens = false;
    int handle = LaunchResponseTokens(tokens, config);
    while (FetchResponseTokens(handle) >= 0) {}
}

}
