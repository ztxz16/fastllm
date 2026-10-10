#pragma once
#include <cstddef>
#include <cstdint>

// Compiled into FastLLM by the SM90 object target. No external adapter is loaded.
extern "C" {
const char *fastllm_ep8_mega_moe_error();
size_t fastllm_ep8_mega_moe_workspace_bytes();
int fastllm_ep8_mega_moe_init_workspace(void *, size_t, const int *, int);
int fastllm_ep8_mega_moe_supported();
void *fastllm_ep8_mega_moe_create(const void *const *, const void *const *,
    const void *const *, const void *const *, void *, size_t, const void *const *, int, int);
int fastllm_ep8_mega_moe_run(void *, const void *, void *, const int32_t *, const float *, int);
void fastllm_ep8_mega_moe_destroy(void *);
}
