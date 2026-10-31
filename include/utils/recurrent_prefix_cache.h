#ifndef FASTLLM_RECURRENT_PREFIX_CACHE_H
#define FASTLLM_RECURRENT_PREFIX_CACHE_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace fastllm {
// KV layout and recurrent tensors belong to the model. These operations only
// index complete checkpoints; a recurrent state cannot be sliced to an LCP.
struct RecurrentPrefixKey {
    int cachedLen = 0;
    long long requestId = 0, timestamp = 0;
    std::vector<int> tokens;
};

inline bool RecurrentPrefixCacheEnabled() {
    const char *value = std::getenv("FASTLLM_PREFIX_CACHE");
    if (!value || !*value) return true;
    std::string flag(value);
    std::transform(flag.begin(), flag.end(), flag.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return flag == "1" || flag == "true" || flag == "yes" || flag == "on";
}

inline int RecurrentPrefixLimit(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (!value || !*value) return fallback;
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end) return fallback;
    return int(std::max(1L, std::min(long(std::numeric_limits<int>::max()), parsed)));
}

inline int RecurrentPrefixInterval(int pageLen) {
    int pages = RecurrentPrefixLimit("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", 16);
    return int(std::min(1LL * pages * std::max(1, pageLen),
                        1LL * std::numeric_limits<int>::max()));
}
inline int RecurrentPrefixMaxPerRequest() {
    return RecurrentPrefixLimit("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST", 4);
}
inline int RecurrentPrefixMaxRecords(int concurrency = 0) {
    int capacity = int(std::min(1LL * std::numeric_limits<int>::max(),
        std::max(8LL, 1LL * std::max(0, concurrency) * RecurrentPrefixMaxPerRequest())));
    return RecurrentPrefixLimit("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS", capacity);
}

template<class Pointer, class Compatible>
typename Pointer::element_type *FindRecurrentPrefix(const std::vector<Pointer> &items,
        const std::vector<int> &tokens, int maxLength, int exactLength, Compatible compatible) {
    typename Pointer::element_type *best = nullptr;
    for (const auto &item : items) {
        auto *s = item.get();
        if (!s || s->cachedLen <= 0 || s->cachedLen > maxLength ||
            s->tokens.size() != size_t(s->cachedLen) || s->tokens.size() > tokens.size() ||
            (exactLength >= 0 && s->cachedLen != exactLength) ||
            !std::equal(s->tokens.begin(), s->tokens.end(), tokens.begin()) || !compatible(*s)) continue;
        if (!best || s->cachedLen > best->cachedLen ||
            (s->cachedLen == best->cachedLen && s->timestamp > best->timestamp)) best = s;
    }
    return best;
}

// Callers serialize lookup/copy/eviction, and handle model-specific duplicate
// replacement (e.g. retain an MTP-aware checkpoint over a target-only one).
template<class Snapshot>
void TrimRecurrentPrefixes(std::vector<std::unique_ptr<Snapshot>> &items,
        long long requestId, int maxPerRequest, int maxRecords) {
    auto trim = [&](bool perRequest, int limit) {
        auto eligible = [&](const std::unique_ptr<Snapshot> &s) {
            return s && (!perRequest || s->requestId == requestId);
        };
        while (std::count_if(items.begin(), items.end(), eligible) > std::max(1, limit)) {
            auto oldest = items.end();
            for (auto it = items.begin(); it != items.end(); ++it)
                if (eligible(*it) && (oldest == items.end() || (*it)->timestamp < (*oldest)->timestamp)) oldest = it;
            items.erase(oldest);
        }
    };
    trim(true, maxPerRequest);
    trim(false, maxRecords);
}

// Shared checkpoint ownership and branch-aware eviction, following Naive's
// prefix pool. Payloads remain model-specific and immutable after publication.
// Callers serialize access and retain one Entry while a request uses it.
template<class Snapshot>
class RecurrentPrefixStore {
public:
    using Entry = std::shared_ptr<Snapshot>;

    template<class Compatible>
    Entry Find(const std::vector<int> &tokens, int maxLength, int exactLength, Compatible compatible) {
        auto *best = FindRecurrentPrefix(items, tokens, maxLength, exactLength, compatible);
        if (!best) return {};
        best->timestamp = ++timestamp;
        for (const auto &item : items) if (item.get() == best) return item;
        return {};
    }

    // Replacements must contain at least the same model-specific state as an
    // equivalent existing checkpoint. Old readers keep their immutable payload.
    // A failed admission changes neither history nor the caller's held entry.
    bool Publish(Entry snapshot, Entry &held, int maxPerRequest, int maxRecords) {
        if (!snapshot || snapshot->cachedLen <= 0 ||
            snapshot->tokens.size() != size_t(snapshot->cachedLen)) return false;
        for (auto &item : items) if (item->tokens == snapshot->tokens) {
            snapshot->timestamp = ++timestamp;
            item = snapshot;
            held = std::move(snapshot);
            return true;
        }
        const size_t capacity = std::max(1, maxRecords);
        const size_t requestLimit = std::max(1, maxPerRequest);
        std::vector<bool> owned(items.size()), related(items.size()), removed(items.size(), false);
        size_t requestCount = 0, count = items.size();
        for (size_t i = 0; i < items.size(); ++i) {
            const auto &tokens = items[i]->tokens;
            size_t common = std::min(tokens.size(), snapshot->tokens.size());
            owned[i] = items[i]->requestId == snapshot->requestId;
            related[i] = owned[i] ||
                (common && std::equal(tokens.begin(), tokens.begin() + common, snapshot->tokens.begin()));
            requestCount += owned[i];
        }
        auto oldest = [&](const std::vector<bool> *eligible) {
            size_t best = items.size();
            for (size_t i = 0; i < items.size(); ++i) {
                if (removed[i] || (eligible && !(*eligible)[i]) ||
                    items[i].use_count() != 1 + int(items[i] == held)) continue;
                if (best == items.size() || items[i]->timestamp < items[best]->timestamp) best = i;
            }
            return best;
        };
        auto remove = [&](size_t i) {
            removed[i] = true; --count;
            requestCount -= owned[i];
        };
        while (requestCount >= requestLimit) {
            size_t i = oldest(&owned);
            if (i == items.size()) break; // Other active forks own these entries.
            remove(i);
        }
        while (count >= capacity) {
            // Retain older continuation boundaries while there is spare room.
            // Under pressure, reclaim this branch before unrelated tool waits.
            size_t i = oldest(&related);
            if (i == items.size()) i = oldest(nullptr);
            if (i == items.size()) return false;
            remove(i);
        }
        size_t kept = 0;
        for (size_t i = 0; i < items.size(); ++i) if (!removed[i]) {
            if (kept != i) items[kept] = std::move(items[i]);
            ++kept;
        }
        items.resize(kept);
        snapshot->timestamp = ++timestamp;
        items.push_back(snapshot);
        held = std::move(snapshot);
        return true;
    }

    size_t Size() const { return items.size(); }

private:
    std::vector<Entry> items;
    long long timestamp = 0;
};
}
#endif
