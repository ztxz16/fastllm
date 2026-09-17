#pragma once
#include "fastllm.h"
namespace fastllm {
// Only already-prepared Marlin weights are admitted; this check never repacks or writes tensors.
bool FastllmCudaNvfp4FusedCanRun(const Data &input, const Data &weight, const Data &bias, const Data &output,
                                 bool swiglu);
// Requires a successful CanRun check. Launch failures are errors, never post-write fallback.
void FastllmCudaNvfp4Fused(Data &input, Data &weight, Data &output, bool swiglu);
// Complete blocks: fusion when admitted, otherwise the existing Linear + post-op path.
// Both branches finish the operation. The return value only reports whether fusion ran.
bool CudaNvfp4LinearSwigluBlock(Data &input, Data &weight, const Data &bias, Data &middle, Data &output);
bool CudaNvfp4LinearAddBlock(Data &input, Data &weight, const Data &bias, Data &middle, Data &output);
} // namespace fastllm
