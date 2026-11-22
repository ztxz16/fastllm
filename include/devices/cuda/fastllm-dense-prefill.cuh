#pragma once
#include <cuda_fp16.h>
#include "attention/default_prefill_params.cuh"

namespace fastllm {
cudaError_t DensePrefillHalf128(flashinfer::SinglePrefillParams<half, half, half> &params,
                                bool causal, cudaStream_t stream,
                                unsigned tileQ = 0, unsigned mmaKvLimit = 0);
}
