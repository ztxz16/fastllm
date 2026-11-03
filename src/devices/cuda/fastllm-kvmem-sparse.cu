#include "devices/cuda/fastllm-cuda-kvmem.cuh"
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastllm {
namespace {
void Check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("KVMem sparse: ") + cudaGetErrorString(error));
}
size_t Product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::invalid_argument("KVMem sparse: allocation size overflow");
    return a * b;
}
struct Storage {
    KvMemSparseConfig cfg;
    unsigned char *hostK, *hostV, *key, *value;
    int *tokenSlots, *slotTokens, *cursor;
    int *protectedSlots, *misses, *available, *missCounts;
    int *indices, *lengths;
    int *boundRequests, *queryLengths, *queryCounts;
    const int **querySelections;
    int *groupHeads, *groupCounts, *nextRows;
    int *keyPages = nullptr, *valuePages = nullptr;
    int pagesPerRequest = 0;
    const void **pointers;
    KvMemSparseStats *stats;
    unsigned *errors;
};

// Block-local compaction, reused at barriers by a single request CTA. Work is
// proportional to selected + resident rows, not the historical context length.
using Scan = cub::BlockScan<int, 256>;
__host__ __device__ int FindOwner(const Storage &s, const void *key) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(key);
    const uintptr_t base = reinterpret_cast<uintptr_t>(s.cfg.pageLen ? (void *)s.keyPages : (void *)s.hostK);
    const size_t span = s.cfg.pageLen ? size_t(s.pagesPerRequest) * sizeof(int)
        : size_t(s.cfg.maxTokens) * s.cfg.keyRowBytes;
    if (address < base || (address - base) % span || (address - base) / span >= size_t(s.cfg.maxRequests))
        return -1;
    return (address - base) / span;
}
__global__ void BindRequests(Storage s, const void *const *keys, int stride,
                             const int *lengths, int batch) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    s.boundRequests[b] = -1;
    if (lengths[b] <= 0) {
        if (lengths[b] < 0) atomicOr(s.errors, 2u);
        return;
    }
    s.boundRequests[b] = FindOwner(s, keys[size_t(b) * stride]);
    if (s.boundRequests[b] < 0) atomicOr(s.errors, 1u);
}
__device__ void Enqueue(bool take, int item, int *queue, int &total, Scan::TempStorage &tmp) {
    int offset, count;
    Scan(tmp).ExclusiveSum(int(take), offset, count);
    if (take) queue[total + offset] = item;
    __syncthreads();
    if (!threadIdx.x) total += count;
    __syncthreads();
}

constexpr int kQueryDescriptors = 64;
struct QueryDescriptors { KvMemSparseQuery rows[kQueryDescriptors]; };
static_assert(sizeof(QueryDescriptors) + sizeof(Storage) + 64 < 4096, "sparse query argument limit");
__global__ void UploadQueries(Storage s, QueryDescriptors queries, int first, int count) {
    const int i = threadIdx.x;
    if (i >= count) return;
    const auto &q = queries.rows[i];
    const int row = first + i, length = q.length ? *q.length : q.tokens;
    s.queryLengths[row] = length; s.queryCounts[row] = q.count;
    s.querySelections[row] = q.selected;
    s.boundRequests[row] = length > 0 ? FindOwner(s, q.keyBinding ? *q.keyBinding : q.key) : -1;
    if (length > 0 && s.boundRequests[row] < 0) atomicOr(s.errors, 1u);
    if (length < 0) atomicOr(s.errors, 2u);
}
__global__ void InitQueryGroups(Storage s, int batch) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < s.cfg.maxRequests) {
        s.groupHeads[i] = -1; s.groupCounts[i] = 0; s.missCounts[i] = 0;
    }
    if (i < batch) {
        s.lengths[i] = 0; s.pointers[2 * i] = s.key; s.pointers[2 * i + 1] = s.value;
    }
    for (size_t n = i; n < size_t(batch) * s.cfg.maxSelected; n += size_t(gridDim.x) * blockDim.x)
        s.indices[n] = -1;
}
__global__ void LinkQueryGroups(Storage s, const int *requests, int batch) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || requests[row] < 0) return;
    const int owner = requests[row];
    if (owner >= s.cfg.maxRequests) { atomicOr(s.errors, 1u); return; }
    s.nextRows[row] = atomicExch(s.groupHeads + owner, row);
    atomicAdd(s.groupCounts + owner, 1);
}
__device__ void InvalidateResidentTail(Storage s, int request, int first) {
    int *tokens = s.slotTokens + size_t(request) * s.cfg.residentTokens;
    int *map = s.tokenSlots + size_t(request) * s.cfg.maxTokens;
    for (int slot = threadIdx.x; slot < s.cfg.residentTokens; slot += blockDim.x) {
        const int token = tokens[slot];
        if (token >= first) {
            tokens[slot] = -1; map[token] = -1;
            atomicAdd(&s.stats[request].evictions, 1ULL);
        }
    }
    __syncthreads();
}
__global__ void InvalidateTailKernel(Storage s, int request, int first) {
    InvalidateResidentTail(s, request, first);
}

template<bool Grouped>
__global__ void PlanResidentCount(Storage s, const int *requests, const int *lengths,
        const int *selected, const int *counts, const int *const *selectedRows, bool appendedRows) {
    const int b = blockIdx.x, t = threadIdx.x;
    const int request = Grouped ? b : requests[b];
    const int capacity = s.cfg.residentTokens, width = s.cfg.maxSelected;
    const int first = Grouped ? s.groupHeads[b] : b;
    if (!Grouped) {
        for (int i = t; i < width; i += blockDim.x) s.indices[size_t(b) * width + i] = -1;
        if (!t) {
            s.lengths[b] = 0; s.pointers[2 * b] = s.key; s.pointers[2 * b + 1] = s.value;
            s.missCounts[b] = 0;
        }
    }
    if (first < 0 || request < 0) return;
    if (request >= s.cfg.maxRequests || (Grouped && s.groupCounts[b] > s.cfg.maxQueries)) {
        if (!t) atomicOr(s.errors, 1u); return;
    }
    int appendStart = s.cfg.maxTokens;
    for (int row = first; row >= 0; row = Grouped ? s.nextRows[row] : -1) {
        const int length = lengths[row], count = counts ? counts[row] : width;
        if (length < 0 || length > s.cfg.maxTokens || count < 0 || count > width) {
            if (!t) atomicOr(s.errors, 2u); return;
        }
        if (length) appendStart = min(appendStart, length - 1);
    }
    // Direct mapped-host append overwrites rejected candidates. Drop their hot
    // copies before protecting hits, so every subsequent read observes new KV.
    if (appendedRows) InvalidateResidentTail(s, request, appendStart);
    int *map = s.tokenSlots + size_t(request) * s.cfg.maxTokens;
    int *tokens = s.slotTokens + size_t(request) * capacity;
    int *protect = s.protectedSlots + size_t(b) * capacity;
    int *misses = s.misses + size_t(b) * width * (Grouped ? s.cfg.maxQueries : 1);
    int *available = s.available + size_t(b) * capacity;
    const int start = s.cursor[request];
    __shared__ Scan::TempStorage tmp;
    __shared__ int nmiss, navailable, nhit, nevicted;
    if (!t) nmiss = navailable = nhit = nevicted = 0;
    for (int i = t; i < capacity; i += blockDim.x) protect[i] = 0;
    __syncthreads();
    for (int row = first; row >= 0; row = Grouped ? s.nextRows[row] : -1) {
        const int *selection = selectedRows ? selectedRows[row] : selected ? selected + size_t(row) * width : nullptr;
        const int count = counts ? counts[row] : width;
        for (int begin = 0; begin < count; begin += blockDim.x) {
            const int i = begin + t;
            const int token = i < count ? (selection ? selection[i] : i) : -1;
            int slot = -3;
            if (token >= 0 && token < lengths[row]) {
                slot = atomicCAS(map + token, -1, -2);
                if (slot >= 0 && !atomicExch(protect + slot, 1)) atomicAdd(&nhit, 1);
            }
            Enqueue(slot == -1, token, misses, nmiss, tmp);
        }
    }
    // maxQueries * maxSelected <= capacity bounds the union even if every
    // query is disjoint. Protected hits can never be evicted by a sibling query.
    const bool full = nmiss && s.stats[request].misses - s.stats[request].evictions ==
        static_cast<unsigned long long>(capacity);
    for (int pass = full ? 1 : 0; pass < 2 && navailable < nmiss; ++pass) {
        for (int begin = 0; begin < capacity && navailable < nmiss; begin += blockDim.x) {
            const int i = begin + t, sum = start + i;
            const int slot = sum >= capacity ? sum - capacity : sum;
            const bool take = i < capacity && !protect[slot] &&
                (pass == 0 ? tokens[slot] < 0 : tokens[slot] >= 0);
            Enqueue(take, slot, available, navailable, tmp);
        }
    }
    for (int i = t; i < nmiss; i += blockDim.x) {
        const int slot = available[i], token = misses[i], old = tokens[slot];
        if (old >= 0) { map[old] = -1; atomicAdd(&nevicted, 1); }
        tokens[slot] = token; map[token] = slot;
    }
    __syncthreads();
    for (int row = first; row >= 0; row = Grouped ? s.nextRows[row] : -1) {
        const int *selection = selectedRows ? selectedRows[row] : selected ? selected + size_t(row) * width : nullptr;
        const int count = counts ? counts[row] : width;
        for (int i = t; i < count; i += blockDim.x) {
            const int token = selection ? selection[i] : i;
            if (token >= 0 && token < lengths[row]) s.indices[size_t(row) * width + i] = map[token];
        }
        if (!t) {
            s.lengths[row] = lengths[row] ? capacity : 0;
            s.pointers[2 * row] = s.key + size_t(request) * capacity * s.cfg.keyRowBytes;
            s.pointers[2 * row + 1] = s.value + size_t(request) * capacity * s.cfg.valueRowBytes;
        }
    }
    if (!t) {
        s.missCounts[b] = nmiss;
        if (nmiss) { const int next = available[nmiss - 1] + 1; s.cursor[request] = next == capacity ? 0 : next; }
        auto &stats = s.stats[request];
        stats.hits += nhit; stats.misses += nmiss; stats.evictions += nevicted;
        stats.copiedBytes += size_t(nmiss) * (s.cfg.keyRowBytes + s.cfg.valueRowBytes);
    }
}

// One warp copies one opaque token row. No host staging, token rounding,
// page overfetch, or architecture-specific instructions are required.
__device__ void CopyRow(unsigned char *dst, const unsigned char *src, size_t bytes, int lane) {
    if (bytes % 16 == 0) {
        for (size_t i = lane; i < bytes / 16; i += 32)
            reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
    } else {
        for (size_t i = lane; i < bytes; i += 32) dst[i] = src[i];
    }
}
__device__ size_t HistoryRow(Storage s, const int *pages, int request, int token) {
    return pages ? size_t(pages[size_t(request) * s.pagesPerRequest + token / s.cfg.pageLen]) *
        s.cfg.pageLen + token % s.cfg.pageLen : size_t(request) * s.cfg.maxTokens + token;
}
template<int Threads, bool Grouped>
__global__ void Refill(Storage s, const int *requests) {
    const int b = blockIdx.y, request = Grouped ? b : requests[b], lane = threadIdx.x % 32;
    const int count = s.missCounts[b];
    constexpr int warps = Threads / 32;
    for (size_t i = blockIdx.x * warps + threadIdx.x / 32; i < size_t(count); i += gridDim.x * warps) {
        const int token = s.misses[size_t(b) * s.cfg.maxSelected * (Grouped ? s.cfg.maxQueries : 1) + i];
        const int slot = s.available[size_t(b) * s.cfg.residentTokens + i];
        const size_t srcK = HistoryRow(s, s.keyPages, request, token);
        const size_t srcV = HistoryRow(s, s.valuePages, request, token);
        const size_t dst = size_t(request) * s.cfg.residentTokens + slot;
        CopyRow(s.key + dst * s.cfg.keyRowBytes, s.hostK + srcK * s.cfg.keyRowBytes, s.cfg.keyRowBytes, lane);
        CopyRow(s.value + dst * s.cfg.valueRowBytes, s.hostV + srcV * s.cfg.valueRowBytes, s.cfg.valueRowBytes, lane);
    }
}
__global__ void WriteRows(Storage s, const int *requests, const int *positions,
                          const unsigned char *keys, const unsigned char *values, int rows) {
    const int lane = threadIdx.x % 32;
    for (size_t row = blockIdx.x * 8 + threadIdx.x / 32; row < size_t(rows); row += gridDim.x * 8) {
        const int request = requests[row], token = positions[row];
        if (request < 0) continue;
        if (request >= s.cfg.maxRequests || token < 0 || token >= s.cfg.maxTokens) {
            if (!lane) atomicOr(s.errors, request >= s.cfg.maxRequests ? 1u : 4u);
            continue;
        }
        const size_t dstK = HistoryRow(s, s.keyPages, request, token);
        const size_t dstV = HistoryRow(s, s.valuePages, request, token);
        CopyRow(s.hostK + dstK * s.cfg.keyRowBytes, keys + size_t(row) * s.cfg.keyRowBytes, s.cfg.keyRowBytes, lane);
        CopyRow(s.hostV + dstV * s.cfg.valueRowBytes, values + size_t(row) * s.cfg.valueRowBytes, s.cfg.valueRowBytes, lane);
        const int slot = s.tokenSlots[size_t(request) * s.cfg.maxTokens + token];
        if (slot >= 0) {
            const size_t hot = size_t(request) * s.cfg.residentTokens + slot;
            CopyRow(s.key + hot * s.cfg.keyRowBytes, keys + size_t(row) * s.cfg.keyRowBytes, s.cfg.keyRowBytes, lane);
            CopyRow(s.value + hot * s.cfg.valueRowBytes, values + size_t(row) * s.cfg.valueRowBytes, s.cfg.valueRowBytes, lane);
        }
    }
}
__global__ void GatherHistory(Storage s, const int *kp, const int *vp, int tokens,
        unsigned char *key, unsigned char *value) {
    const int lane = threadIdx.x % 32;
    for (int token = blockIdx.x * 8 + threadIdx.x / 32; token < tokens; token += gridDim.x * 8) {
        size_t k = size_t(kp[token / s.cfg.pageLen]) * s.cfg.pageLen + token % s.cfg.pageLen;
        size_t v = size_t(vp[token / s.cfg.pageLen]) * s.cfg.pageLen + token % s.cfg.pageLen;
        CopyRow(key + size_t(token) * s.cfg.keyRowBytes, s.hostK + k * s.cfg.keyRowBytes, s.cfg.keyRowBytes, lane);
        CopyRow(value + size_t(token) * s.cfg.valueRowBytes, s.hostV + v * s.cfg.valueRowBytes, s.cfg.valueRowBytes, lane);
    }
}
} // namespace

struct KvMemSparseCache::Impl {
    Storage s{};
    int device = 0, multiprocessors = 1;
    bool wideRefillSupported = false;
    void *hostKey = nullptr, *hostValue = nullptr;
    size_t hostBytes = 0, deviceBytes = 0;
    std::vector<void *> allocations;

    void OnDevice() const {
        int current; Check(cudaGetDevice(&current));
        if (current != device) throw std::invalid_argument("KVMem sparse: wrong CUDA device");
    }
    void Request(int request) const {
        if (request < 0 || request >= s.cfg.maxRequests)
            throw std::invalid_argument("KVMem sparse: request outside allocated capacity");
    }
    size_t HostOffset(int request) const {
        Request(request);
        if (s.cfg.pageLen && request != 0)
            throw std::invalid_argument("KVMem sparse: paged backing has one global base");
        return size_t(request) * s.cfg.maxTokens;
    }
    template<class T> void Allocate(T *&pointer, size_t count) {
        const size_t bytes = Product(count, sizeof(T));
        void *p = nullptr; Check(cudaMalloc(&p, bytes));
        try { allocations.push_back(p); } catch (...) { cudaFree(p); throw; }
        pointer = static_cast<T *>(p); deviceBytes += bytes;
    }
    ~Impl() {
        int previous = device;
        cudaGetDevice(&previous); cudaSetDevice(device);
        // Destruction is outside inference/graph replay. Graph executables must
        // be destroyed by the caller before the backing storage is released.
        cudaDeviceSynchronize();
        for (void *p : allocations) cudaFree(p);
        if (hostKey) cudaFreeHost(hostKey);
        if (hostValue) cudaFreeHost(hostValue);
        cudaSetDevice(previous);
    }
};

KvMemSparseCache::KvMemSparseCache(const KvMemSparseConfig &c) : impl(new Impl) {
    if (c.maxRequests <= 0 || c.maxBatch <= 0 || c.maxBatch > 65535 || c.maxTokens <= 0 ||
        c.residentTokens <= 0 || c.residentTokens > c.maxTokens || c.maxSelected <= 0 ||
        c.maxQueries < 1 || int64_t(c.maxSelected) * c.maxQueries > c.residentTokens || !c.keyRowBytes || !c.valueRowBytes ||
        c.keyRowBytes > std::numeric_limits<size_t>::max() - c.valueRowBytes ||
        c.pageLen < 0 || (c.pageLen && (c.maxTokens % c.pageLen || c.maxTokens / c.pageLen > INT_MAX - 3)) ||
        c.totalTokens < 0 || (c.totalTokens && (!c.pageLen || c.totalTokens % c.pageLen ||
                                               c.totalTokens < c.maxTokens)))
        throw std::invalid_argument("KVMem sparse: invalid capacity or row size");
    auto &p = *impl; auto &s = p.s; s.cfg = c;
    Check(cudaGetDevice(&p.device));
    cudaDeviceProp prop{}; Check(cudaGetDeviceProperties(&prop, p.device));
    if (!prop.canMapHostMemory) throw std::runtime_error("KVMem sparse: mapped host memory unavailable");
    p.multiprocessors = prop.multiProcessorCount;
    p.wideRefillSupported = prop.major == 9 && prop.maxThreadsPerBlock >= 1024;
    const size_t logicalRows = Product(c.maxRequests, c.maxTokens);
    const size_t hostRows = c.totalTokens ? c.totalTokens : logicalRows;
    const size_t hotRows = Product(c.maxRequests, c.residentTokens);
    p.hostBytes = Product(hostRows, c.keyRowBytes + c.valueRowBytes);
    if (p.hostBytes > c.hostBudgetBytes) throw std::invalid_argument("KVMem sparse: host budget exceeded");
    Check(cudaHostAlloc(&p.hostKey, Product(hostRows, c.keyRowBytes), cudaHostAllocMapped | cudaHostAllocPortable));
    Check(cudaHostAlloc(&p.hostValue, Product(hostRows, c.valueRowBytes), cudaHostAllocMapped | cudaHostAllocPortable));
    Check(cudaHostGetDevicePointer(&s.hostK, p.hostKey, 0));
    Check(cudaHostGetDevicePointer(&s.hostV, p.hostValue, 0));
    p.Allocate(s.key, Product(hotRows, c.keyRowBytes));
    p.Allocate(s.value, Product(hotRows, c.valueRowBytes));
    p.Allocate(s.tokenSlots, logicalRows); p.Allocate(s.slotTokens, hotRows);
    if (c.pageLen) {
        // Model kernels require 16-byte-aligned table views even when the
        // logical token limit occupies an odd number of pages.
        s.pagesPerRequest = (c.maxTokens / c.pageLen + 3) / 4 * 4;
        p.Allocate(s.keyPages, Product(c.maxRequests, s.pagesPerRequest));
        p.Allocate(s.valuePages, Product(c.maxRequests, s.pagesPerRequest));
        Check(cudaMemset(s.keyPages, 0, size_t(c.maxRequests) * s.pagesPerRequest * sizeof(int)));
        Check(cudaMemset(s.valuePages, 0, size_t(c.maxRequests) * s.pagesPerRequest * sizeof(int)));
    }
    p.Allocate(s.cursor, c.maxRequests);
    const size_t groups = c.maxQueries > 1 ? c.maxRequests : std::max(c.maxBatch, c.maxRequests);
    p.Allocate(s.protectedSlots, Product(groups, c.residentTokens));
    p.Allocate(s.available, Product(groups, c.residentTokens));
    p.Allocate(s.misses, Product(std::max(size_t(c.maxBatch), Product(c.maxRequests, c.maxQueries)), c.maxSelected));
    p.Allocate(s.indices, Product(c.maxBatch, c.maxSelected));
    p.Allocate(s.boundRequests, c.maxBatch);
    p.Allocate(s.queryLengths, c.maxBatch); p.Allocate(s.queryCounts, c.maxBatch);
    p.Allocate(s.querySelections, c.maxBatch);
    p.Allocate(s.groupHeads, c.maxRequests); p.Allocate(s.groupCounts, c.maxRequests);
    p.Allocate(s.nextRows, c.maxBatch);
    p.Allocate(s.missCounts, groups); p.Allocate(s.lengths, c.maxBatch);
    p.Allocate(s.pointers, Product(c.maxBatch, 2));
    p.Allocate(s.stats, c.maxRequests); p.Allocate(s.errors, 1);
    Check(cudaMemset(s.tokenSlots, 0xff, logicalRows * sizeof(int)));
    Check(cudaMemset(s.slotTokens, 0xff, hotRows * sizeof(int)));
    Check(cudaMemset(s.cursor, 0, c.maxRequests * sizeof(int)));
    Check(cudaMemset(s.stats, 0, c.maxRequests * sizeof(KvMemSparseStats)));
    Check(cudaMemset(s.errors, 0, sizeof(unsigned)));
    Check(cudaStreamSynchronize(nullptr)); // Publish initialization to arbitrary caller streams.
}
KvMemSparseCache::~KvMemSparseCache() = default;
void *KvMemSparseCache::HostKey(int request) {
    return static_cast<unsigned char *>(impl->hostKey) +
        impl->HostOffset(request) * impl->s.cfg.keyRowBytes;
}
void *KvMemSparseCache::HostValue(int request) {
    return static_cast<unsigned char *>(impl->hostValue) +
        impl->HostOffset(request) * impl->s.cfg.valueRowBytes;
}
void *KvMemSparseCache::DeviceHostKey(int request) {
    return impl->s.hostK + impl->HostOffset(request) * impl->s.cfg.keyRowBytes;
}
void *KvMemSparseCache::DeviceHostValue(int request) {
    return impl->s.hostV + impl->HostOffset(request) * impl->s.cfg.valueRowBytes;
}
size_t KvMemSparseCache::HostBytes() const { return impl->hostBytes; }
size_t KvMemSparseCache::DeviceBytes() const { return impl->deviceBytes; }
int *KvMemSparseCache::KeyPages(int request) {
    impl->Request(request); if (!impl->s.keyPages) return nullptr;
    return impl->s.keyPages + size_t(request) * impl->s.pagesPerRequest;
}
int *KvMemSparseCache::ValuePages(int request) {
    impl->Request(request); if (!impl->s.valuePages) return nullptr;
    return impl->s.valuePages + size_t(request) * impl->s.pagesPerRequest;
}
void KvMemSparseCache::Gather(const int *kp, const int *vp, int tokens,
        void *key, void *value, cudaStream_t stream) {
    impl->OnDevice();
    if (!impl->s.cfg.pageLen || tokens < 0 || tokens > impl->s.cfg.maxTokens || !kp || !vp || !key || !value)
        throw std::invalid_argument("KVMem sparse: invalid paged gather");
    if (!tokens) return;
    GatherHistory<<<std::min((tokens + 7) / 8, impl->multiprocessors * 4), 256, 0, stream>>>(
        impl->s, kp, vp, tokens, (unsigned char *)key, (unsigned char *)value);
    Check(cudaGetLastError());
}
void KvMemSparseCache::Reset(int request, cudaStream_t stream) {
    impl->OnDevice(); impl->Request(request); const auto &s = impl->s;
    Check(cudaMemsetAsync(s.tokenSlots + size_t(request) * s.cfg.maxTokens, 0xff,
                         size_t(s.cfg.maxTokens) * sizeof(int), stream));
    Check(cudaMemsetAsync(s.slotTokens + size_t(request) * s.cfg.residentTokens, 0xff,
                         size_t(s.cfg.residentTokens) * sizeof(int), stream));
    Check(cudaMemsetAsync(s.cursor + request, 0, sizeof(int), stream));
    Check(cudaMemsetAsync(s.stats + request, 0, sizeof(KvMemSparseStats), stream));
}
void KvMemSparseCache::ClearErrors(cudaStream_t stream) {
    impl->OnDevice(); Check(cudaMemsetAsync(impl->s.errors, 0, sizeof(unsigned), stream));
}
void KvMemSparseCache::Write(const int *requests, const int *positions, const void *keys,
                           const void *values, int rows, cudaStream_t stream) {
    impl->OnDevice();
    if (rows < 0 || (rows && (!requests || !positions || !keys || !values)))
        throw std::invalid_argument("KVMem sparse: invalid write inputs");
    if (!rows) return;
    WriteRows<<<std::min(1024, (rows - 1) / 8 + 1), 256, 0, stream>>>(
        impl->s, requests, positions, static_cast<const unsigned char *>(keys),
        static_cast<const unsigned char *>(values), rows);
    Check(cudaGetLastError());
}
namespace {
template<bool Grouped>
void LaunchResolve(Storage s, const int *requests, const int *lengths, const int *selected,
        const int *counts, int batch, cudaStream_t stream, const int *const *selectedRows,
        int multiprocessors, bool wideRefill, bool appendedRows) {
    const int groups = Grouped ? s.cfg.maxRequests : batch;
    if (Grouped) {
        InitQueryGroups<<<(std::max(batch, groups) + 255) / 256, 256, 0, stream>>>(s, batch);
        LinkQueryGroups<<<(batch + 255) / 256, 256, 0, stream>>>(s, requests, batch);
    }
    PlanResidentCount<Grouped><<<groups, 256, 0, stream>>>(s, requests, lengths, selected, counts, selectedRows, appendedRows);
    Check(cudaGetLastError());
    const int selectedCapacity = s.cfg.maxSelected * (Grouped ? s.cfg.maxQueries : 1);
    if (wideRefill && groups >= 64 && selectedCapacity >= 256) {
        const int blocks = std::min({128, (selectedCapacity - 1) / 32 + 1,
                                    (multiprocessors * 2 + groups - 1) / groups});
        Refill<1024, Grouped><<<dim3(blocks, groups), 1024, 0, stream>>>(s, requests);
    } else {
        const int blocks = std::min({128, (selectedCapacity - 1) / 8 + 1,
                                    (multiprocessors * 4 + groups - 1) / groups});
        Refill<256, Grouped><<<dim3(blocks, groups), 256, 0, stream>>>(s, requests);
    }
    Check(cudaGetLastError());
}
}
void KvMemSparseCache::Resolve(const int *requests, const int *lengths, const int *selected,
                             const int *counts, int batch, cudaStream_t stream, const int *const *selectedRows) {
    impl->OnDevice();
    if (batch < 0 || batch > impl->s.cfg.maxBatch || (batch && (!requests || !lengths)))
        throw std::invalid_argument("KVMem sparse: invalid resolve inputs");
    if (!batch) return;
    if (impl->s.cfg.maxQueries > 1)
        LaunchResolve<true>(impl->s, requests, lengths, selected, counts, batch, stream, selectedRows,
                            impl->multiprocessors, impl->wideRefillSupported, false);
    else LaunchResolve<false>(impl->s, requests, lengths, selected, counts, batch, stream, selectedRows,
                              impl->multiprocessors, impl->wideRefillSupported, false);
}
void KvMemSparseCache::ResolveQueries(const KvMemSparseQuery *queries, int batch, cudaStream_t stream,
                                     bool appendedRows) {
    impl->OnDevice();
    if (batch < 0 || batch > impl->s.cfg.maxBatch || (batch && !queries))
        throw std::invalid_argument("KVMem sparse: invalid query descriptors");
    if (!batch) return;
    for (int first = 0; first < batch; first += kQueryDescriptors) {
        QueryDescriptors descriptors{};
        const int count = std::min(kQueryDescriptors, batch - first);
        std::copy(queries + first, queries + first + count, descriptors.rows);
        UploadQueries<<<1, kQueryDescriptors, 0, stream>>>(impl->s, descriptors, first, count);
    }
    auto &s = impl->s;
    if (s.cfg.maxQueries > 1)
        LaunchResolve<true>(s, s.boundRequests, s.queryLengths, nullptr, s.queryCounts, batch, stream,
                            s.querySelections, impl->multiprocessors, impl->wideRefillSupported, appendedRows);
    else LaunchResolve<false>(s, s.boundRequests, s.queryLengths, nullptr, s.queryCounts, batch, stream,
                              s.querySelections, impl->multiprocessors, impl->wideRefillSupported, appendedRows);
}
void KvMemSparseCache::InvalidateTail(const void *key, int first, cudaStream_t stream) {
    impl->OnDevice();
    const auto &s = impl->s;
    const int owner = FindOwner(s, key);
    if (first < 0 || first > s.cfg.maxTokens || owner < 0)
        throw std::invalid_argument("KVMem sparse: invalid suffix invalidation");
    InvalidateTailKernel<<<1, 256, 0, stream>>>(s, owner, first);
    Check(cudaGetLastError());
}
void KvMemSparseCache::ResolveFromPointers(const void *const *keys, int pointerStride,
        const int *lengths, const int *selected, const int *counts, int batch, cudaStream_t stream, const int *const *selectedRows) {
    impl->OnDevice();
    if (batch < 0 || batch > impl->s.cfg.maxBatch || pointerStride <= 0 ||
        (batch && (!keys || !lengths)))
        throw std::invalid_argument("KVMem sparse: invalid request address table");
    if (!batch) return;
    BindRequests<<<(batch + 255) / 256, 256, 0, stream>>>(impl->s, keys, pointerStride, lengths, batch);
    Check(cudaGetLastError());
    Resolve(impl->s.boundRequests, lengths, selected, counts, batch, stream, selectedRows);
}
const int *KvMemSparseCache::Indices() const { return impl->s.indices; }
const int *KvMemSparseCache::Lengths() const { return impl->s.lengths; }
const void *const *KvMemSparseCache::CachePointers() const { return impl->s.pointers; }
const KvMemSparseStats *KvMemSparseCache::DeviceStats() const { return impl->s.stats; }
const unsigned *KvMemSparseCache::DeviceErrors() const { return impl->s.errors; }
} // namespace fastllm
