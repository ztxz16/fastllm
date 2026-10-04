#ifndef FASTLLM_KVMEM_H
#define FASTLLM_KVMEM_H

#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>

namespace fastllm {
    // Model-local and opt-in. All token budgets except maxTokens are page aligned.
    struct KvMemConfig {
        bool enabled = false;
        int maxTokens = 32768;
        int residentTokens = 8192;
        int sinkTokens = 128;
        int recentTokens = 2048;
        int retrievalTokens = 4096;
        int prefillTokens = 512;
        // Speculative retrieval refresh interval in committed tokens; 1 refreshes
        // every round. Independent of physical page size and token budgets.
        int retrievalInterval = 64;
        uint64_t hostBytes = 8ULL << 30;

        void Validate(int pageLen) const;
    };

    struct KvMemStats {
        uint64_t hostBytes = 0, evictedPages = 0, restoredPages = 0;
        // Includes cached pages outside the current attention view.
        int tokens = 0, residentPages = 0;
    };

    // Backend-independent append-only page store. Logical IDs retain absolute
    // positions; slots are recycled only when needed. Unselected resident pages
    // are cached by recency; they do not enter the attention view. The view is
    // chronological and contains the current query and its recent causal history.
    // A full immutable page is backed up only once. Batch callbacks must finish
    // all transfers before returning (also on failure). Backups complete before
    // any slot is reused. This object belongs to one request/layer.
    class KvMemPages {
    public:
        struct ReadPage { int slot; uint8_t *data; };
        struct WritePage { int slot; const uint8_t *data; };
        using ReadPages = std::function<void(const std::vector<ReadPage>&)>;
        using WritePages = std::function<void(const std::vector<WritePage>&)>;

        KvMemPages(const KvMemConfig &config, int pageLen, size_t pageBytes);
        const std::vector<int>& Append(int tokens, const std::vector<float> &scores,
                                      const ReadPages &read, const WritePages &write);
        // One provisional append per transaction. Finish keeps an accepted
        // prefix (0 rolls back); the writer/recent tail stays resident.
        void BeginTransaction(int tokens);
        void FinishTransaction(int acceptedTokens);
        bool InTransaction() const { return transactionBase >= 0; }
        const std::vector<int>& Visible() const { return visible; }
        int Slot(int logicalPage) const;
        int Tokens() const { return stats.tokens; }
        const KvMemStats& Stats() const { return stats; }

    private:
        KvMemConfig config;
        int pageLen, capacity;
        size_t pageBytes;
        bool failed = false;
        int transactionBase = -1, transactionTokens = 0;
        std::vector<int> slots, visible;
        std::vector<uint64_t> lastUsed;
        uint64_t useSequence = 0;
        std::vector<std::vector<uint8_t>> host;
        KvMemStats stats;
    };

    class Data;
    // Detach a borrowed sparse view without releasing its slots into a global
    // page pool. Returns false for an ordinary cache; dimensions stay caller-owned.
    bool ReleaseKvMemCache(Data &cache);

    class KvMemCache {
    public:
        virtual ~KvMemCache() = default;
        virtual const KvMemStats& Stats() const = 0;
        virtual void BeginTransaction(int tokens) = 0;
        // Commit only accepted normalized keys to the retrieval index and
        // republish both attention views. May finish with 0 before Append.
        virtual void FinishTransaction(int acceptedTokens, Data &keyCache, Data &valueCache) = 0;
        // rawQ/rawK: normalized pre-RoPE [1,T,H,D]; q/k/v: post-RoPE
        // [H,T,D]. KV and query types must match (FP16 or BF16).
        virtual void Append(const Data &rawQ, const Data &rawK,
                            const Data &k, const Data &v,
                            Data &keyCache, Data &valueCache) = 0;
    };

#ifdef USE_CUDA
    // Publishes a non-owning compact paged-attention view into key/valueCache.
    // RoPE must use absolute positions. Supports ordinary causal MHA/GQA;
    // sliding-window attention and position-dependent biases need another view.
    void KvMemAppend(const Data &rawQ, const Data &rawK, const Data &k, const Data &v,
                     Data &keyCache, Data &valueCache);
#endif
}
#endif
