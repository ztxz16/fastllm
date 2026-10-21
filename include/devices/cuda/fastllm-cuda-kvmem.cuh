#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace fastllm {

// Exact storage for externally selected sparse KV. This does not select tokens
// or change attention arithmetic. One instance owns one layer on one device.
struct KvMemSparseConfig {
    int maxRequests = 1;
    int maxBatch = 1;           // Includes graph padding; independent of owners.
    int maxTokens = 32768;
    int residentTokens = 4096;
    int maxSelected = 2048;    // Must fit in residentTokens, even on a total miss.
    size_t keyRowBytes = 0;
    size_t valueRowBytes = 0;  // Opaque rows also preserve FP8 scales/padding.
    size_t hostBudgetBytes = size_t(8) << 30;
    // Optional shared backing. The caller owns page allocation/references;
    // hot buffers remain private to requests. Zero keeps contiguous backing.
    int pageLen = 0;
    // Optional page-aligned global pool, independent of per-request maxTokens.
    // Zero retains maxRequests * maxTokens backing. Requires paged mode.
    int totalTokens = 0;
};

struct KvMemSparseStats {
    unsigned long long hits = 0, misses = 0, evictions = 0, copiedBytes = 0;
};

class KvMemSparseCache {
public:
    explicit KvMemSparseCache(const KvMemSparseConfig &config);
    ~KvMemSparseCache();
    KvMemSparseCache(const KvMemSparseCache &) = delete;
    KvMemSparseCache &operator=(const KvMemSparseCache &) = delete;

    // Complete pinned DDR backing. Contiguous mode is request-major;
    // paged mode only accepts request=0, the global [page, offset, rowBytes]. CPU
    // initialization is allowed only while no device work accesses this store.
    // Reset before reusing an owner or changing previously initialized bytes.
    void *HostKey(int request);
    void *HostValue(int request);
    // CUDA-visible aliases of the same pinned backing, for fused GPU append
    // kernels. Direct writes must be append-only; overwrite via Write instead.
    void *DeviceHostKey(int request);
    void *DeviceHostValue(int request);
    // Fixed per-request device tables, or nullptr in contiguous mode. The
    // caller manages valid page IDs, references and copy-on-write. Shared
    // pages are immutable; replacing bytes/page bindings requires Reset first.
    int *KeyPages(int request);
    int *ValuePages(int request);
    // Copies one paged history to a caller-owned prefill workspace, exactly.
    void Gather(const int *keyPages, const int *valuePages, int tokens,
                void *key, void *value, cudaStream_t stream);
    size_t HostBytes() const;
    size_t DeviceBytes() const;

    // All pointer arguments are device pointers; all operations are ordered on
    // stream. There are no allocations, host callbacks or host reads in these
    // methods. Constructor/destructor and CPU initialization are outside graphs.
    // Owners cannot be accessed concurrently from different streams/graphs.
    void Reset(int request, cudaStream_t stream);
    void ClearErrors(cudaStream_t stream);

    // Append/overwrite packed GPU rows at absolute positions. Pairs
    // (request, position) must be unique within this call. A negative request is
    // a padding row. Writes through to DDR and updates an existing hot copy.
    void Write(const int *requests, const int *positions, const void *keys,
               const void *values, int rows, cudaStream_t stream);

    // One row per active owner (duplicate nonnegative request IDs are invalid).
    // selected has fixed [batch, maxSelected] stride; null means arange.
    // selectedRows optionally supplies one selected pointer per row.
    // counts may be null (all maxSelected entries). Negative/past-length
    // indices are masked to -1.
    // Inactive rows use request=-1. All selected valid rows become GPU resident;
    // hits are protected before recycling any unselected slot. Empty slots are
    // preferred; other victims are selected in rotating slot order.
    void Resolve(const int *requests, const int *lengths, const int *selected,
                 const int *counts, int batch, cudaStream_t stream,
                 const int *const *selectedRows = nullptr);
    // Bind graph requests from full-history K bases (contiguous mode) or
    // KeyPages(request) addresses (paged mode). Entries are device
    // pointers separated by pointerStride elements. Nonpositive lengths mask
    // inactive rows. A null selected pointer selects positions [0,maxSelected).
    // This keeps request/order changes in an existing device address table.
    void ResolveFromPointers(const void *const *keys, int pointerStride,
                 const int *lengths, const int *selected, const int *counts,
                 int batch, cudaStream_t stream,
                 const int *const *selectedRows = nullptr);

    // Stable outputs for attention: physical token IDs in the original TopK
    // order, [batch,2] K/V bases, and physical bounds (0 for padding/error).
    const int *Indices() const;
    const int *Lengths() const;
    const void *const *CachePointers() const;
    const KvMemSparseStats *DeviceStats() const; // [maxRequests], unique-row stats
    const unsigned *DeviceErrors() const;       // Sticky: 1=request, 2=length/count,
                                                // 4=write position. Read explicitly
                                                // outside the graph to diagnose.

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace fastllm
