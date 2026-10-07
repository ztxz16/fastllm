#pragma once
#include "fastllm-gguf-mmq-common.cuh"

namespace fastllm_gguf_mmq {
// Template definitions live only in the format-specific instantiation units.
template <ggml_type type, typename OutputType, int StoreMode = 0>
void launch_extended_mmvq_type(
        const void *weight, const block_q8_1 *input, OutputType *output,
        int rows, int input_columns, int output_rows, cudaStream_t stream);

template <ggml_type type>
void launch_extended_gate_up_type(
        const void *gate_weight, const void *up_weight,
        const block_q8_1 *input, half *output, int rows,
        int input_columns, int output_rows, cudaStream_t stream);

// IQ1 initialization must target the device table owned by its kernel unit.
void ensure_extended_iq1s_grid(cudaStream_t stream);
} // namespace fastllm_gguf_mmq
