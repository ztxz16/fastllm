#pragma once
#include <cstddef>
#include <cstdint>

// Native SM90 MXFP4 EP4/EP8. Preparation owns repacked weights; inference
// borrows the caller's persistent symmetric arena and never allocates.
extern "C" {
const char *fastllm_mxfp4_ep_error();
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
// commonRows must match on every peer. Short/idle sources are padded with
// route -1; only localRows output rows are copied to the caller.
int fastllm_mxfp4_ep_run(void *, const void *, void *, const int32_t *,
    const float *, int localRows, int commonRows, const int *liveKeys);
void fastllm_mxfp4_ep_destroy(void *);
}
