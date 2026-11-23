#pragma once
#include "fastllm.h"
#include "devices/cuda/low_latency/norm_linear.h"

namespace fastllm {
// Thin Data adapter for the explicitly BF16-rounded norm/projection contract.
// All Data buffers, including FP32 output, must be allocated before Prepare.
// Caller retains the original unfused path when preparation is unsupported.
// No model, TP/EP, request, sampling or graph ownership lives in this adapter.
cudaError_t FastllmCudaPrepareNormLinear(const Data &input, const Data *branch,
    const Data &norm, const Data &weight, const Data &output,
    cuda::low_latency::NormLinearPlan &plan, cudaStream_t stream);
cudaError_t FastllmCudaRunNormLinear(const cuda::low_latency::NormLinearPlan &plan,
    const Data &input, const Data *branch, const Data &norm, const Data &weight,
    Data &output, float epsilon, cudaStream_t stream);
} // namespace fastllm
