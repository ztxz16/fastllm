// Query-conditioned selection for graph backends with mapped, immutable KV pages.
// The caller supplies a binding per request (or a device-side indirection when
// replaying a graph), normalized queries, and a stable hot pool of packed K/V pairs.
#pragma once
#include "kvmem.h"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cmath>
#include <algorithm>

namespace fastllm::kv_selection {
struct MaxScore { __device__ float operator()(float a, float b) const { return fmaxf(a, b); } };
struct State {
    KvMemPagePolicy policy;
    int limit, heads, kvHeads, dim, interval;
    int indexed = 0, sealed = 0, kept = 0, lastPast = -1, lastInterval = 0;
    float *means, *logits, *scores;
    int *selected, *tags, *slotOfPage, *viewSlots, *protectedSlots;
    const int2 *source;
    unsigned char *hot;
    int pairBytes;

    __device__ int omitted() const { return (sealed - kept) * policy.pageLen; }
    __device__ int absolute(int token) const {
        const int page = token / policy.pageLen;
        return page < kept ? selected[page] * policy.pageLen + token % policy.pageLen : token + omitted();
    }
    __device__ const unsigned char *page(int p) const {
        const int2 id = source[p];
        return reinterpret_cast<const unsigned char *>(uint32_t(id.x) | (uintptr_t(uint32_t(id.y)) << 32));
    }
};
static_assert(sizeof(State) == 152, "Update the Inkling KV memory planner when changing the selector descriptor");
inline size_t Bytes(int pages, int capacity, int width, int heads) {
    return sizeof(State) + sizeof(float) * size_t(pages) * (width + heads + 2) + 4 * sizeof(int) * capacity;
}
static __global__ void initializeState(State *device, State state) { *device = state; }
inline cudaError_t Initialize(State *device, const KvMemConfig &cfg, int limit, int resident, int pageLen,
                              int heads, int kvHeads, int dim, const int2 *source, unsigned char *hot,
                              int pairBytes, cudaStream_t stream) {
    const int width = kvHeads * dim;
    State s{};
    const int sink = std::min(cfg.sinkTokens / pageLen, resident);
    const int budget = resident - sink;
    const int recent = cfg.recentTokens / pageLen, retrieve = cfg.retrievalTokens / pageLen;
    // Small existing hot-cache configurations keep the requested proportions.
    const int recentPages = int64_t(recent) + retrieve <= budget ? recent :
        int((int64_t(budget) * recent + recent + retrieve - 1) / (int64_t(recent) + retrieve));
    s.policy = {resident, sink, recentPages * pageLen, std::min(retrieve, budget - recentPages), pageLen};
    s.limit = limit; s.heads = heads; s.kvHeads = kvHeads; s.dim = dim; s.interval = cfg.retrievalInterval;
    s.means = reinterpret_cast<float *>(device + 1);
    s.logits = s.means + size_t(limit) * width;
    s.scores = s.logits + size_t(limit) * heads;
    s.selected = reinterpret_cast<int *>(s.scores + limit);
    s.tags = s.selected + resident;
    s.slotOfPage = s.tags + resident;
    s.viewSlots = s.slotOfPage + limit;
    s.protectedSlots = s.viewSlots + resident;
    s.source = source; s.hot = hot; s.pairBytes = pairBytes;
    initializeState<<<1,1,0,stream>>>(device, s);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    return cudaMemsetAsync(s.tags, -1, sizeof(int) * (size_t(resident) + limit), stream);
}
// Never index the writable tail. A speculative overwrite can change that tail,
// while all pages indexed here have been committed by the scheduler.
template<class KV> __global__ void index(State *state, int first, int end) {
    const int p = first + blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = state->kvHeads * state->dim;
    if (c < width) {
        const auto *k = reinterpret_cast<const KV *>(state->page(p));
        float sum = 0;
        for (int t = 0; t < state->policy.pageLen; ++t) sum += float(k[size_t(t) * width + c]);
        state->means[size_t(p) * width + c] = sum / state->policy.pageLen;
    }
    if (!blockIdx.x && !blockIdx.y && !threadIdx.x) state->indexed = end;
}

struct Query {
    State *state;
    int past;

    struct Refresh { int sealed, interval; bool needed, score; };
    __device__ Refresh refresh(int chain, bool prefill) const {
        const State &s = *state;
        const int sealed = min(past / s.policy.pageLen, s.indexed);
        const int interval = chain > 1 && !prefill ? s.interval : s.policy.pageLen;
        const bool needed = prefill || s.lastPast < 0 || past <= s.lastPast || s.lastInterval != interval ||
                            past / interval != s.lastPast / interval || sealed != s.sealed;
        return {sealed, interval, needed, needed && sealed > s.policy.capacity && s.policy.retrieve > 0};
    }
};

// A warp scores a full KV page for one query head. Rank by the mean across
// query heads of their individually normalized page probabilities (as in Qwen).
template<class Bind> __global__ void score(Bind input, const float *queries, int chain, bool prefill) {
    const auto b = input(blockIdx.z);
    if (!b.state) return;
    const State &s = *b.state;
    const auto refresh = b.refresh(chain, prefill);
    if (!refresh.score) return;
    const int h = blockIdx.y, kh = h / (s.heads / s.kvHeads);
    const float *q = queries + (size_t(blockIdx.z) * chain * s.heads + h) * s.dim;
    const int lane = threadIdx.x % 32;
    for (int p = (blockIdx.x * blockDim.x + threadIdx.x) / 32; p < refresh.sealed; p += gridDim.x * blockDim.x / 32) {
        const float *mean = s.means + (size_t(p) * s.kvHeads + kh) * s.dim;
        float dot = 0;
        for (int c = lane; c < s.dim; c += 32) dot += q[c] * mean[c];
        for (int d = 16; d; d /= 2) dot += __shfl_down_sync(0xffffffff, dot, d);
        if (!lane) s.logits[size_t(h) * s.limit + p] = dot * rsqrtf(float(s.dim));
    }
}
template<class Bind> __global__ void normalize(Bind input, int chain, bool prefill) {
    const auto b = input(blockIdx.y);
    if (!b.state) return;
    const State &s = *b.state;
    const auto refresh = b.refresh(chain, prefill);
    if (!refresh.score) return;
    const int n = refresh.sealed;
    using Reduction = cub::BlockReduce<float, 128>;
    __shared__ typename Reduction::TempStorage temp;
    __shared__ float peak, total;
    float *row = s.logits + size_t(blockIdx.x) * s.limit;
    float hi = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) hi = fmaxf(hi, row[p]);
    hi = Reduction(temp).Reduce(hi, MaxScore{});
    if (!threadIdx.x) peak = hi;
    __syncthreads();
    float sum = 0;
    for (int p = threadIdx.x; p < n; p += blockDim.x) { row[p] = expf(row[p] - peak); sum += row[p]; }
    sum = Reduction(temp).Sum(sum);
    if (!threadIdx.x) total = sum;
    __syncthreads();
    for (int p = threadIdx.x; p < n; p += blockDim.x) row[p] /= total;
}

__device__ inline unsigned long long rankKey(float score, int page) {
    // Probabilities are nonnegative. A tie selects the earlier absolute page.
    return (uint64_t(__float_as_uint(isfinite(score) ? score : 0.f)) << 32) | uint32_t(~page);
}
template<class Bind> __global__ void select(Bind input, int chain, bool prefill) {
    const auto b = input(blockIdx.x);
    if (!b.state) return;
    State &s = *b.state;
    const auto refresh = b.refresh(chain, prefill);
    if (!refresh.needed) return;
    const int n = refresh.sealed;
    const auto r = s.policy.RangeFor(b.past, n);
    const int kept = r.sink + r.take + n - r.recent;
    using Reduce = cub::BlockReduce<int, 128>;
    using Scan = cub::BlockScan<int, 128>;
    __shared__ union { typename Reduce::TempStorage sum; typename Scan::TempStorage scan; } temp;
    __shared__ unsigned long long threshold;
    if (!threadIdx.x) threshold = 0;
    for (int p = r.sink + threadIdx.x; r.take && p < r.recent; p += blockDim.x) {
        float value = 0;
        for (int h = 0; h < s.heads; ++h) value += s.logits[size_t(h) * s.limit + p];
        s.scores[p] = value;
    }
    __syncthreads();
    for (int bit = 63; r.take && bit >= 0; --bit) {
        const uint64_t cut = threshold | (uint64_t(1) << bit);
        int votes = 0;
        for (int p = r.sink + threadIdx.x; p < r.recent; p += blockDim.x) votes += rankKey(s.scores[p], p) >= cut;
        votes = Reduce(temp.sum).Sum(votes);
        if (!threadIdx.x && votes >= r.take) threshold = cut;
        __syncthreads();
    }
    for (int base = 0, written = 0; base < n; base += blockDim.x) {
        const int p = base + threadIdx.x;
        const int take = p < n && (p < r.sink || p >= r.recent ||
            (r.take && rankKey(s.scores[p], p) >= threshold));
        int offset, count;
        Scan(temp.scan).ExclusiveSum(take, offset, count);
        if (take) s.selected[written + offset] = p;
        written += count;
        __syncthreads();
    }
    // Selected IDs remain chronological for position math. Physical slots are
    // independent, so a new early page never shifts or reloads later hits.
    for (int slot = threadIdx.x; slot < s.policy.capacity; slot += blockDim.x) s.protectedSlots[slot] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < kept; i += blockDim.x) {
        const int slot = s.slotOfPage[s.selected[i]];
        s.viewSlots[i] = slot;
        if (slot >= 0) s.protectedSlots[slot] = 1;
    }
    __syncthreads();
    if (!threadIdx.x) {
        int free = 0;
        for (int i = 0; i < kept; ++i) if (s.viewSlots[i] < 0) {
            while (s.protectedSlots[free]) ++free;
            const int old = s.tags[free];
            if (old >= 0) s.slotOfPage[old] = -1;
            s.viewSlots[i] = free;
            s.slotOfPage[s.selected[i]] = free++;
        }
        s.lastPast = b.past;
        s.lastInterval = refresh.interval;
        s.sealed = n;
        s.kept = kept;
    }
}
template<class Bind> __global__ void stage(Bind input) {
    const auto b = input(blockIdx.y);
    if (!b.state) return;
    State &s = *b.state;
    for (int i = blockIdx.x; i < s.kept; i += gridDim.x) {
        const int p = s.selected[i], slot = s.viewSlots[i];
        __shared__ int miss;
        if (!threadIdx.x) miss = s.tags[slot] != p;
        __syncthreads();
        if (miss) {
            const auto *src = reinterpret_cast<const uint4 *>(s.page(p));
            auto *dst = reinterpret_cast<uint4 *>(s.hot + size_t(slot) * s.pairBytes);
            for (int i = threadIdx.x; i < s.pairBytes / int(sizeof(uint4)); i += blockDim.x) dst[i] = src[i];
            if (!threadIdx.x) s.tags[slot] = p;
        }
        __syncthreads();
    }
}
template<class Bind>
void Resolve(Bind bindings, const float *queries, int requests, int chain, int heads, bool prefill, cudaStream_t stream) {
    score<<<dim3(64, heads, requests),128,0,stream>>>(bindings, queries, chain, prefill);
    normalize<<<dim3(heads, requests),128,0,stream>>>(bindings, chain, prefill);
    select<<<requests,128,0,stream>>>(bindings, chain, prefill);
    stage<<<dim3(128,requests),256,0,stream>>>(bindings);
}
}
