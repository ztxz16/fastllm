#pragma once
#include <cstddef>
#include <cstdint>

// Native SM90 MXFP4 EP4/EP8. Preparation owns repacked weights; inference
// borrows the caller's persistent symmetric arena and never allocates.
inline constexpr int kFastllmMxfp4LocalMaxRows = 8;
inline constexpr int kFastllmMxfp4LocalRoutedMaxRows = 32;

extern "C" {
const char *fastllm_mxfp4_ep_error();
// One-rank adapters for SM90 TP8. Preparation preflights all scales before
// transferring ownership, so an unsupported checkpoint can still select Marlin.
size_t fastllm_mxfp4_tp8_workspace_bytes(int device);
int fastllm_mxfp4_tp8_validate(const void *const *gate, const void *const *down, int *shifts);
void *fastllm_mxfp4_tp8_create(const void *const *gate, const void *const *down, const int *shifts,
    void *workspace, size_t bytes);
int fastllm_mxfp4_tp8_run_routed(void *layer, const void *input, void *output,
    const float *probabilities, const float *bias, int rows);
int fastllm_mxfp4_tp8_restore(void *layer, void *staging);
// Both capacities share the packed weights but have separate persistent arenas.
int fastllm_mxfp4_ep_capacity();
int fastllm_mxfp4_ep_prefill_capacity();
size_t fastllm_mxfp4_ep_workspace_bytes(int ranks);
// The 128-expert draft uses a smaller arena. prefillCapacity may be null.
size_t fastllm_mxfp4_ep_draft_workspace(int ranks, int *prefillCapacity);
int fastllm_mxfp4_ep_init(void *, size_t, const int *, int rank, int ranks);
void *fastllm_mxfp4_ep_create(const void *const *gate, const void *const *down,
    void *workspace, size_t bytes, const void *const *peers, int rank, int ranks);
void *fastllm_mxfp4_ep_create_shape(const void *const *gate, const void *const *down,
    void *workspace, size_t bytes, const void *const *peers, int rank, int ranks, int experts, int topk);
// Assign logical top-K IDs to the packed slots of an SM90 EP8 target layer.
int fastllm_mxfp4_ep_set_map(void *layer, const uint8_t *logicalToPhysical);
// commonRows must match on every peer. Short/idle sources are padded with
// route -1; only localRows output rows are copied to the caller.
int fastllm_mxfp4_ep_run(void *, const void *, void *, const int32_t *,
    const float *, int localRows, int commonRows, const int *liveKeys);
void fastllm_mxfp4_ep_destroy(void *);
}
