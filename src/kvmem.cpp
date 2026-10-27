#include "kvmem.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace fastllm {
    void KvMemConfig::Validate(int pageLen) const {
        if (!enabled) return;
        if (retrievalInterval <= 0) {
            throw std::invalid_argument("KVMem: retrieval interval must be positive");
        }
        if (pageLen <= 0 || maxTokens < 2 || residentTokens <= 0 ||
            sinkTokens < 0 || recentTokens < pageLen || retrievalTokens < 0 ||
            prefillTokens <= 0 || hostBytes == 0 || maxTokens > std::numeric_limits<int>::max() - pageLen) {
            throw std::invalid_argument("KVMem: invalid cache budgets");
        }
        for (int value : {residentTokens, sinkTokens, recentTokens, retrievalTokens, prefillTokens}) {
            if (value % pageLen != 0) {
                throw std::invalid_argument("KVMem: token budgets must be multiples of the KV page size");
            }
        }
        if ((int64_t)sinkTokens + recentTokens + retrievalTokens + prefillTokens + pageLen > residentTokens) {
            throw std::invalid_argument("KVMem: resident budget must cover sink + recent + retrieval + prefill + one boundary page");
        }
    }

    KvMemPages::KvMemPages(const KvMemConfig &config, int pageLen, size_t pageBytes)
        : config(config), pageLen(pageLen), capacity(0), pageBytes(pageBytes) {
        config.Validate(pageLen);
        if (!config.enabled || !pageBytes || pageLen <= 0) throw std::invalid_argument("KVMem: invalid page store");
        capacity = config.residentTokens / pageLen;
        const size_t count = ((int64_t)config.maxTokens + pageLen - 1) / pageLen;
        if (count > config.hostBytes / pageBytes) {
            throw std::invalid_argument("KVMem: host budget cannot back the configured logical context");
        }
        slots.assign(count, -1);
        lastUsed.assign(count, 0);
        host.resize(count);
    }

    int KvMemPages::Slot(int logicalPage) const {
        if (logicalPage < 0 || logicalPage >= (int)slots.size() || slots[logicalPage] < 0) {
            throw std::out_of_range("KVMem: logical page is not resident");
        }
        return slots[logicalPage];
    }

    void KvMemPages::BeginTransaction(int tokens) {
        if (failed) throw std::runtime_error("KVMem: cache unusable after a failed transfer");
        if (InTransaction() || tokens <= 0 || tokens > config.prefillTokens ||
            tokens > config.maxTokens - stats.tokens) {
            throw std::invalid_argument("KVMem: invalid speculative transaction");
        }
        transactionBase = stats.tokens;
        transactionTokens = tokens;
    }

    void KvMemPages::FinishTransaction(int acceptedTokens) {
        if (failed) throw std::runtime_error("KVMem: cache unusable after a failed transfer");
        if (!InTransaction() || acceptedTokens < 0 || acceptedTokens > stats.tokens - transactionBase) {
            throw std::invalid_argument("KVMem: invalid accepted prefix");
        }
        const int end = transactionBase + acceptedTokens;
        const int count = (end + pageLen - 1) / pageLen;
        // Provisional pages are never cold: one bounded append keeps its
        // entire tail resident. No backup of rejected KV can survive here.
        for (int p = count; p < (int)slots.size(); ++p) {
            if (slots[p] >= 0) --stats.residentPages;
            slots[p] = -1;
            lastUsed[p] = 0;
            stats.hostBytes -= host[p].size();
            host[p].clear();
        }
        visible.erase(std::lower_bound(visible.begin(), visible.end(), count), visible.end());
        stats.tokens = end;
        transactionBase = -1;
        transactionTokens = 0;
    }

    const std::vector<int>& KvMemPages::Append(int tokens, const std::vector<float> &scores,
                                             const ReadPages &read, const WritePages &write) {
        if (failed) throw std::runtime_error("KVMem: cache unusable after a failed transfer");
        if (InTransaction() && (stats.tokens != transactionBase || tokens != transactionTokens)) {
            throw std::invalid_argument("KVMem: transaction requires exactly one append of the reserved length");
        }
        if (tokens <= 0 || tokens > config.prefillTokens ||
            tokens > config.maxTokens - stats.tokens) {
            throw std::invalid_argument("KVMem: append exceeds prefill or logical context budget");
        }
        int oldTokens = stats.tokens, end = oldTokens + tokens;
        int pages = (end + pageLen - 1) / pageLen;
        std::set<int> keep;
        if (pages <= capacity) {
            for (int p = 0; p < pages; ++p) keep.insert(p);
        } else {
            for (int p = 0; p < std::min(pages, config.sinkTokens / pageLen); ++p) keep.insert(p);
            // Use the start of the query, not its end: all rows keep their
            // preceding recent history, even when a prefill straddles pages.
            int first = std::max(0, oldTokens - config.recentTokens) / pageLen;
            for (int p = first; p < pages; ++p) keep.insert(p);
            std::vector<int> candidates;
            for (int p = 0; p < first; ++p) if (!keep.count(p)) candidates.push_back(p);
            auto score = [&](int p) {
                return p < (int)scores.size() && std::isfinite(scores[p]) ? scores[p] : 0.0f;
            };
            std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                return score(a) > score(b);
            });
            int take = std::min((int)candidates.size(), config.retrievalTokens / pageLen);
            for (int i = 0; i < take; ++i) keep.insert(candidates[i]);
        }
        if ((int)keep.size() > capacity) throw std::logic_error("KVMem: mandatory pages exceed resident budget");

        // Selection and residency are independent: retain unselected pages in
        // spare slots, and evict only enough to admit the current attention view.
        // Recency changes transfer costs, never the selected pages or their order.
        int missing = 0;
        for (int p : keep) if (slots[p] < 0) ++missing;
        std::vector<int> victims;
        int evict = std::max(0, stats.residentPages + missing - capacity);
        if (evict) {
            for (int p = 0; p < (int)slots.size(); ++p)
                if (slots[p] >= 0 && !keep.count(p)) victims.push_back(p);
            if (evict > (int)victims.size()) throw std::logic_error("KVMem: no evictable slot");
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return lastUsed[a] != lastUsed[b] ? lastUsed[a] < lastUsed[b] : a < b;
            });
            victims.resize(evict);
        }

        // Allocate all required backing pages before changing device residency.
        // A partial page is always in the query's recent/writable tail.
        std::vector<std::pair<int, std::vector<uint8_t>>> backups;
        for (int p : victims) {
            if ((int64_t)(p + 1) * pageLen > oldTokens) throw std::logic_error("KVMem: attempted partial-page eviction");
            if (host[p].empty()) backups.emplace_back(p, std::vector<uint8_t>(pageBytes));
        }
        if (backups.size() > (config.hostBytes - stats.hostBytes) / pageBytes) {
            throw std::runtime_error("KVMem: host budget exhausted");
        }
        try {
            std::vector<ReadPage> reads;
            reads.reserve(backups.size());
            for (auto &item : backups) reads.push_back({slots[item.first], item.second.data()});
            if (!reads.empty()) read(reads);
            for (auto &item : backups) {
                host[item.first] = std::move(item.second);
                stats.hostBytes += pageBytes;
            }
            std::vector<bool> used(capacity, false);
            for (int p : victims) {
                slots[p] = -1;
                ++stats.evictedPages;
            }
            for (int p = 0; p < (int)slots.size(); ++p) {
                if (slots[p] >= 0) used[slots[p]] = true;
            }
            int free = 0;
            std::vector<WritePage> writes;
            writes.reserve(keep.size());
            for (int p : keep) {
                if (slots[p] >= 0) continue;
                while (free < capacity && used[free]) ++free;
                if (free == capacity) throw std::logic_error("KVMem: no free slot");
                if (!host[p].empty()) writes.push_back({free, host[p].data()});
                else if ((int64_t)p * pageLen < oldTokens) throw std::logic_error("KVMem: missing host backing");
                slots[p] = free;
                used[free] = true;
            }
            if (!writes.empty()) write(writes);
            stats.restoredPages += writes.size();
            ++useSequence;
            for (int p : keep) lastUsed[p] = useSequence;
            visible.assign(keep.begin(), keep.end());
            stats.tokens = end;
            stats.residentPages += missing - evict;
        } catch (...) {
            failed = true;
            throw;
        }
        return visible;
    }
}
