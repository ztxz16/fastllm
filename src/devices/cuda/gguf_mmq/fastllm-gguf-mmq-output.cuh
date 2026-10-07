#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace fastllm_gguf_mmq {
struct mmq_args;

// The reduction depends on the quantization block size, not its weight format.
// Definitions and explicit instantiations live in one CUDA translation unit.
template <int qk, int mmq_x, bool need_check, typename OutputType>
void launch_mmq_fixup_to_output(
        const mmq_args &args, const float *last_tile, OutputType *output,
        dim3 output_tiles, int block_num_mmq, cudaStream_t stream);

template <typename OutputType>
void launch_mmq_output_conversion(
        const float *input, OutputType *output, int64_t count, cudaStream_t stream);
} // namespace fastllm_gguf_mmq
