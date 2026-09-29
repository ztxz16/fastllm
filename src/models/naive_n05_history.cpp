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
    std::lock_guard<std::mutex> guard(historyMutex);
    auto it = activeHistory.find(&kv);
    if (!saveHistoryChat || it == activeHistory.end() ||
        it->second.length != past || historyBytesPerToken == 0) return nullptr;
    auto &state = it->second;
    int rows = std::min<size_t>(length, (historyByteLimit - state.bytes) / historyBytesPerToken);
    if (rows <= 0) return nullptr;
    auto chunk = std::make_shared<HistoryChunk>();
    chunk->length = rows;
    chunk->bytes = rows * historyBytesPerToken;
    chunk->layers.resize(block_cnt);
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

void NaiveN05FlashModel::FinishHistoryChunk(
        const std::vector<std::pair<Data, Data>> &kv,
        const std::shared_ptr<HistoryChunk> &chunk) {
    if (!chunk) return;
    std::lock_guard<std::mutex> guard(historyMutex);
    auto it = activeHistory.find(&kv);
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
        if (!saveHistoryChat || !context->multimodalInput.empty()) return;
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
    std::lock_guard<std::mutex> guard(historyMutex);
    activeHistory[&context->pastKeyValues] = std::move(state);
}

void NaiveN05FlashModel::OnResponseContextRemoved(ResponseContext *context) {
    std::lock_guard<std::mutex> guard(historyMutex);
    activeHistory.erase(&context->pastKeyValues);
}

void NaiveN05FlashModel::TryRecordResponseContext(ResponseContext *context) {
    std::lock_guard<std::mutex> guard(historyMutex);
    if (!saveHistoryChat || !context || !context->multimodalInput.empty()) return;
    auto active = activeHistory.find(&context->pastKeyValues);
    if (active == activeHistory.end() || active->second.length <= 0 ||
        active->second.length > (int)context->allTokens.size()) return;
    // allTokens may include the last sampled token, whose KV has not run yet.
    auto tokenEnd = context->allTokens.begin() + active->second.length;
    for (auto it = history.begin(); it != history.end(); ++it) {
        const auto &entry = *it;
        if (entry->tokens.size() >= active->second.length &&
            std::equal(context->allTokens.begin(), tokenEnd, entry->tokens.begin())) {
            std::rotate(it, it + 1, history.end());
            return;
        }
    }
    auto memory = std::make_shared<HistoryMemory>(active->second);
    memory->tokens.assign(context->allTokens.begin(), tokenEnd);
    // A descendant covers every prefix of its ancestor, sharing the same
    // immutable chunks. Retiring the ancestor avoids duplicate LRU accounting.
    history.erase(std::remove_if(history.begin(), history.end(), [&](const auto &entry) {
        return entry->tokens.size() < memory->tokens.size() &&
            std::equal(entry->tokens.begin(), entry->tokens.end(), memory->tokens.begin());
    }), history.end());
    history.push_back(std::move(memory));
    size_t bytes = 0;
    for (const auto &entry : history) bytes += entry->bytes;
    while (history.size() > historyRecordLimit || bytes > historyByteLimit) {
        bytes -= history.front()->bytes;
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
