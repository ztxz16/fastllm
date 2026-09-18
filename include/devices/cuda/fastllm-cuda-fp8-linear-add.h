#pragma once
#include "fastllm.h"
namespace fastllm {
// Decode-only, dense row-scaled FP8 weights: N=5120, K=6144/17408,
// batch=1, FP16/BF16 activations and residual. CanRun never writes tensors.
// A false result leaves the existing LinearAdd fallback responsible for execution.
bool FastllmCudaFP8LinearAddCanRun(const Data &input, const Data &weight, const Data &bias,
                                 const Data &output);
// Call only after CanRun succeeds. Output is the already allocated residual;
// graph capture requires the usual eager warmup to prepare device scales.
void FastllmCudaFP8LinearAdd(Data &input, Data &weight, const Data &bias, Data &output);
} // namespace fastllm
