#pragma once
#include <cstddef>
#include <cstdint>

// Compiled into FastLLM by the SM90 object target. No external adapter is loaded.
extern "C" {
const char *fastllm_ep8_mega_moe_error();
int fastllm_ep8_mega_moe_max_rows();
size_t fastllm_ep8_mega_moe_workspace_bytes();
int fastllm_ep8_mega_moe_init_workspace(void *, size_t, const int *, int);
int fastllm_ep8_mega_moe_supported();
void *fastllm_ep8_mega_moe_create(const void *const *, const void *const *,
    const void *const *, const void *const *, void *, size_t, const void *const *, int, int);
// Every rank must participate. Zero rows represent an idle source rank;
// pointers must remain non-null, and no output rows are written for that rank.
int fastllm_ep8_mega_moe_run(void *, const void *, void *, const int32_t *, const float *, int);
// Zero device lengths mask padded rows inside the existing pre-dispatch kernel.
int fastllm_ep8_mega_moe_run_masked(void *, const void *, void *, const int32_t *, const float *, int, const int *);
// Eager prefill may borrow dead activations instead of allocating a larger
// arena. Prepare enqueues local initialization; the caller must record its
// completion event and make all peer streams wait before Run. Keep every
// arena alive until all ranks finish.
// inputRows is the global row count, identical on every rank; it selects a
// bounded capacity without depending on rank-local buffer sizes.
// Prepare returns the row capacity, 0 for unsuitable storage, -1 on error.
size_t fastllm_ep8_mega_moe_prefill_workspace_bytes(int inputRows);
int fastllm_ep8_mega_moe_prepare_prefill(void *, void *, size_t, int inputRows);
int fastllm_ep8_mega_moe_run_prefill(void *, const void *const *, const void *, void *,
    const int32_t *, const float *, int);
void fastllm_ep8_mega_moe_destroy(void *);
}
