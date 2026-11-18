#pragma once
#include "devices/cuda/fastllm-mimo-ep8-mega-kernel.h"

#ifdef FASTLLM_MIMO_MEGA_SM90
extern "C" decltype(fastllm_ep8_mega_moe_error) fastllm_sm90_mega_moe_error;
extern "C" decltype(fastllm_ep8_mega_moe_max_rows) fastllm_sm90_mega_moe_max_rows;
extern "C" decltype(fastllm_ep8_mega_moe_workspace_bytes) fastllm_sm90_mega_moe_workspace_bytes;
extern "C" decltype(fastllm_ep8_mega_moe_init_workspace) fastllm_sm90_mega_moe_init_workspace;
extern "C" decltype(fastllm_ep8_mega_moe_supported) fastllm_sm90_mega_moe_supported;
extern "C" decltype(fastllm_ep8_mega_moe_create) fastllm_sm90_mega_moe_create;
extern "C" decltype(fastllm_ep8_mega_moe_run) fastllm_sm90_mega_moe_run;
extern "C" decltype(fastllm_ep8_mega_moe_run_masked) fastllm_sm90_mega_moe_run_masked;
extern "C" decltype(fastllm_ep8_mega_moe_decode_capacity) fastllm_sm90_mega_moe_decode_capacity;
extern "C" decltype(fastllm_ep8_mega_moe_run_batched) fastllm_sm90_mega_moe_run_batched;
extern "C" decltype(fastllm_ep8_mega_moe_prefill_workspace_bytes) fastllm_sm90_mega_moe_prefill_workspace_bytes;
extern "C" decltype(fastllm_ep8_mega_moe_prepare_prefill) fastllm_sm90_mega_moe_prepare_prefill;
extern "C" decltype(fastllm_ep8_mega_moe_run_prefill) fastllm_sm90_mega_moe_run_prefill;
extern "C" decltype(fastllm_ep8_mega_moe_destroy) fastllm_sm90_mega_moe_destroy;
#endif

#ifdef FASTLLM_MIMO_MEGA_SM100
extern "C" decltype(fastllm_ep8_mega_moe_create_mxfp4) fastllm_sm100_mega_moe_create_mxfp4;
extern "C" decltype(fastllm_ep8_mega_moe_error) fastllm_sm100_mega_moe_error;
extern "C" decltype(fastllm_ep8_mega_moe_max_rows) fastllm_sm100_mega_moe_max_rows;
extern "C" decltype(fastllm_ep8_mega_moe_workspace_bytes) fastllm_sm100_mega_moe_workspace_bytes;
extern "C" decltype(fastllm_ep8_mega_moe_init_workspace) fastllm_sm100_mega_moe_init_workspace;
extern "C" decltype(fastllm_ep8_mega_moe_supported) fastllm_sm100_mega_moe_supported;
extern "C" decltype(fastllm_ep8_mega_moe_create) fastllm_sm100_mega_moe_create;
extern "C" decltype(fastllm_ep8_mega_moe_run) fastllm_sm100_mega_moe_run;
extern "C" decltype(fastllm_ep8_mega_moe_run_masked) fastllm_sm100_mega_moe_run_masked;
extern "C" decltype(fastllm_ep8_mega_moe_decode_capacity) fastllm_sm100_mega_moe_decode_capacity;
extern "C" decltype(fastllm_ep8_mega_moe_run_batched) fastllm_sm100_mega_moe_run_batched;
extern "C" decltype(fastllm_ep8_mega_moe_prefill_workspace_bytes) fastllm_sm100_mega_moe_prefill_workspace_bytes;
extern "C" decltype(fastllm_ep8_mega_moe_prepare_prefill) fastllm_sm100_mega_moe_prepare_prefill;
extern "C" decltype(fastllm_ep8_mega_moe_run_prefill) fastllm_sm100_mega_moe_run_prefill;
extern "C" decltype(fastllm_ep8_mega_moe_destroy) fastllm_sm100_mega_moe_destroy;
#endif
