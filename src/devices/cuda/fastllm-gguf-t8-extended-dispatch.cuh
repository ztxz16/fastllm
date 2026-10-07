#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define GGML_COMMON_DECL_CUDA
#include "gguf.h"

namespace fastllm_gguf_small_mmvq {
// Instantiate each quantization family once, independently of the public dispatch.
template <ggml_type Type>
bool DispatchOutput(int outputKind, int storeMode, const void *w, const block_q8_1 *x, void *y,
                    int k, int n, int inputStride, int outputStride, cudaStream_t stream);
} // namespace fastllm_gguf_small_mmvq
