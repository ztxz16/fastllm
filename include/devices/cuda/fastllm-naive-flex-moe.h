#pragma once
#include <cstddef>
#include <cstdint>
#include <cmath>

// Borrowed per-rank activation workspace. No weights or allocations are owned
// by this descriptor. Its producer and consumer use the same ordered stream.
struct FastllmMoePreparedInput {
    uint8_t *values = nullptr, *scales = nullptr;
    int *counters = nullptr;
    uint64_t *epoch = nullptr;
    int rows = 0;
    bool routed = false;
    // Consumer preference decided during backend setup, before graph capture.
    bool preferFusedSelection = false;
    int *indices = nullptr, *jobs = nullptr, *count = nullptr;
    int *unique = nullptr, *experts = nullptr, *members = nullptr, *rowCounts = nullptr;
    float *scores = nullptr;
    const float *bias = nullptr;
    int needNorm = 1;
    float routeScale = 1.f;
    // Optional block-128 image in the consumer's existing gate scratch.
    // Eight or sixteen rows with 4112-byte stride, followed by 32 x capacity
    // FP32 scales. Wide producers write only this image, not the legacy xq/xs.
    uint8_t *block128 = nullptr;
    static constexpr size_t Block128Bytes = 33920;
    size_t Block128Size() const { return Block128Bytes * (rows > 8 ? 2 : 1); }
    struct Range { const void *data; size_t bytes; };
    void Ranges(Range (&ranges)[4]) const {
        ranges[0] = {values, size_t(values && rows > 0 ? rows : 0) * 4096};
        ranges[1] = {scales, size_t(scales && rows > 0 ? rows : 0) * 128};
        ranges[2] = {counters, 256 * sizeof(int)};
        ranges[3] = {epoch, sizeof(uint64_t)};
    }
    void RoutingRanges(Range (&ranges)[9]) const {
        ranges[0] = {indices, size_t(rows > 0 ? rows : 0) * 8 * sizeof(int)};
        ranges[1] = {scores, size_t(rows > 0 ? rows : 0) * 8 * sizeof(float)};
        ranges[2] = {jobs, 64 * sizeof(int)};
        ranges[3] = {count, sizeof(int)};
        ranges[4] = {unique, sizeof(int)};
        ranges[5] = {experts, 64 * sizeof(int)};
        ranges[6] = {members, 512 * sizeof(int)};
        ranges[7] = {rowCounts, 64 * sizeof(int)};
        ranges[8] = {bias, bias ? 256 * sizeof(float) : 0};
    }
    static bool Disjoint(Range a, Range b) {
        if (!a.bytes || !b.bytes) return true;
        const uintptr_t x = uintptr_t(a.data), y = uintptr_t(b.data);
        return x <= y ? y - x >= a.bytes : x - y >= b.bytes;
    }
    bool Disjoint(const void *data, size_t bytes) const {
        Range ranges[4]; Ranges(ranges);
        for (auto range : ranges) if (!Disjoint(range, {data, bytes})) return false;
        if (block128 && !Disjoint({block128, Block128Size()}, {data, bytes})) return false;
        if (routed) {
            Range routing[9]; RoutingRanges(routing);
            for (auto range : routing) if (!Disjoint(range, {data, bytes})) return false;
        }
        return true;
    }
    bool Valid(int count) const {
        if (count < 2 || count > 16 || rows != count || !counters || !epoch ||
            (uintptr_t(values) & 15) || (uintptr_t(scales) & 3) ||
            (uintptr_t(counters) & 3) || (uintptr_t(epoch) & 7)) return false;
        // A wide descriptor cannot advertise the separate eight-row buffers.
        if (rows > 8 ? (values || scales || !block128 || routed) : (!values || !scales)) return false;
        Range ranges[4]; Ranges(ranges);
        for (int i = 0; i < 4; ++i) {
            if (uintptr_t(ranges[i].data) > UINTPTR_MAX - ranges[i].bytes) return false;
            for (int j = 0; j < i; ++j) if (!Disjoint(ranges[i], ranges[j])) return false;
        }
        if (block128) {
            if (rows < 4 || (uintptr_t(block128) & 15) ||
                uintptr_t(block128) > UINTPTR_MAX - Block128Size()) return false;
            for (auto base : ranges) if (!Disjoint(base, {block128, Block128Size()})) return false;
        }
        if (routed) {
            if (!std::isfinite(routeScale)) return false;
            Range routing[9]; RoutingRanges(routing);
            for (int i = 0; i < 9; ++i) {
                const auto range = routing[i];
                if (!range.bytes) continue;
                if (!range.data || (uintptr_t(range.data) & 3) ||
                    uintptr_t(range.data) > UINTPTR_MAX - range.bytes) return false;
                if (block128 && !Disjoint(range, {block128, Block128Size()})) return false;
                for (auto base : ranges) if (!Disjoint(base, range)) return false;
                for (int j = 0; j < i; ++j) if (!Disjoint(routing[j], range)) return false;
            }
        }
        return true;
    }
};

// Built-in SM103 specializations. Admission probes never allocate weights or
// launch kernels. The caller must select one backend for the entire TP group.
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
extern "C" {
int fastflex_supported(const int *devices, int ranks);
int fastshared_tp_supported(const int *devices, int ranks);
int fastshared_ep_supported(const int *devices, int ranks);
int fastflex_create(const int *, int layers, void **);
int fastflex_pack(void *, int rank, int layer, const uint64_t *, const float *, const float *);
int fastflex_run(void *, int rank, int layer, int rows, const void *, const void *, const void *, void *);
void fastflex_destroy(void *);
#define FASTLLM_SHARED_FLEX_ABI(kind) \
int fastshared_##kind##_create(const int *, int layers, void **); \
int fastshared_##kind##_bind(void *, int rank, int layer, const void *, const void *, const void *, const void *); \
int fastshared_##kind##_run(void *, int rank, int layer, int rows, const void *, const void *, const void *, void *); \
void fastshared_##kind##_destroy(void *); \
size_t fastshared_##kind##_workspace_bytes();
FASTLLM_SHARED_FLEX_ABI(tp)
FASTLLM_SHARED_FLEX_ABI(ep)
#undef FASTLLM_SHARED_FLEX_ABI
// Dense BF16 verification: H4096, I2048/rank, TP8, 1..64 contiguous rows.
// All ranks call admit before launching; one failed local eligibility check
// returns false for the whole group. run_dense borrows every allocation and
// uses the state's exclusive ordered per-rank streams, chunking rows by eight.
int fastshared_tp_dense_supported(void *state);
int fastshared_tp_dense_admit(void *state,int rank,int rows,int eligible);
int fastshared_tp_run_dense(void *state,int rank,int rows,const void *input,
    const void *gate,const void *up,const void *down,const float *gamma,float eps,void *silu,void *output);

// Optional TP routing fusion: 2..8 BF16 rows, dense FP32 [rows,256]
// sigmoid probabilities (16-byte aligned), optional FP32 [256] correction bias.
// Select before capture/launch; unsupported inputs retain standalone selection.
int fastshared_tp_router_supported(void *state);
int fastshared_tp_run_router(void *state, int rank, int layer, int rows, const void *x,
    const float *routerProbabilities, const float *bias, int *ids, float *scores,
    void *output, int needNorm, float routeScale);
// The group-wide optional capability is finalized during create, before capture.
int fastshared_tp_prepared_input(void *state, int rank, int rows, FastllmMoePreparedInput *input);
int fastshared_tp_run_prepared(void *state, int rank, int layer, int rows,
    const float *routerProbabilities, const float *bias, int *ids, float *scores,
    void *output, int needNorm, float routeScale);
// Uses membership emitted by the ordered router producer. No new preparation
// or selection is launched; the original gate/down arithmetic is retained.
int fastshared_tp_run_preselected(void *state, int rank, int layer, int rows,
    const int *ids, const float *scores, void *output);
int fastshared_tp_pack(void *, int rank, int layer, const uint64_t *, const float *, const float *);
}
#endif
