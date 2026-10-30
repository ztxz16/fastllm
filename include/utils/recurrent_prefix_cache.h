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
inline int RecurrentPrefixMaxRecords() {
    return RecurrentPrefixLimit("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS", 8);
}

template<class Snapshot, class Compatible>
Snapshot *FindRecurrentPrefix(const std::vector<std::unique_ptr<Snapshot>> &items,
        const std::vector<int> &tokens, int maxLength, int exactLength, Compatible compatible) {
    Snapshot *best = nullptr;
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
}
#endif
