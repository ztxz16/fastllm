#ifndef FASTLLM_DISK_EXPERT_CACHE_H
#define FASTLLM_DISK_EXPERT_CACHE_H

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace fastllm {
    class Data;

    // Free a disk-loaded temporary expert weight: CUDA extras (if any), the
    // owned GGUF descriptor copy, then the Data itself. Defined in
    // diskdevice.cpp where the CUDA helpers live.
    void ReleaseDiskExpertWeight(Data *weight);

    // Stable identity for a disk expert weight: logical name plus data type,
    // GGML type, shape, and the first non-scale payload part's file/offset.
    // The trailing file/offset term is what keeps same-named experts from
    // different splits (or rematerialized copies) from colliding.
    std::string MakeDiskWeightCacheKey(const Data *weight);

    struct DiskExpertWeightDeleter {
        void operator()(Data *weight) const { ReleaseDiskExpertWeight(weight); }
    };

    struct DiskExpertCacheStats {
        bool enabled = false;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t entries = 0;
        uint64_t usedBytes = 0;
        uint64_t capacityBytes = 0;
    };

    // Bounded, thread-safe LRU of materialized disk expert weights. Only
    // GGUF-format weights are cached (they never take the in-place CUDA prepare
    // path, so a cached copy stays valid across prefill and decode). The cache
    // owns inserted weights and frees them through ReleaseDiskExpertWeight on
    // eviction or Clear. Capacity defaults to 8 GiB and is overridden by
    // FASTLLM_DISK_EXPERT_CACHE_BYTES (0 disables caching entirely).
    class DiskExpertWeightCache {
    public:
        static DiskExpertWeightCache &Get();

        bool Enabled();

        // Returns the cache-owned weight on a hit (promoted to MRU), else null.
        Data *Lookup(const std::string &key);

        // Takes ownership of `weight`. On success the cache owns it and returns
        // the canonical pointer with cachedOut=true. If the key already exists
        // the redundant incoming weight is freed and the existing pointer is
        // returned (cachedOut=true). If a single weight exceeds the whole
        // budget it is not cached: it is returned unchanged with cachedOut=false
        // so the caller keeps ownership.
        Data *Insert(const std::string &key, Data *weight, uint64_t bytes,
                     bool &cachedOut);

        void Clear();

        DiskExpertCacheStats GetStats();

        void ConfigureForTesting(uint64_t capacityBytes);

    private:
        DiskExpertWeightCache() = default;

        uint64_t CapacityBytesLocked();
        std::unique_ptr<Data, DiskExpertWeightDeleter> EvictOneLocked();

        struct Entry {
            Data *weight = nullptr;
            uint64_t bytes = 0;
            std::list<std::string>::iterator orderIt;
        };

        std::mutex locker;
        std::list<std::string> order; // front = most recently used
        std::unordered_map<std::string, Entry> table;
        uint64_t usedBytes = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t capacityBytes = 0;
        bool capacityResolved = false;
    };
}

#endif
