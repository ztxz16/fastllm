#pragma once
#include "fastllm.h"

namespace fastllm {
// Original row-major BF16 weights; logical rows are tiled by device resources. A rejected layout/architecture
// submits no work and leaves all tensors unchanged. Outputs must not alias inputs.
bool FastllmCudaDecodeNormLinearCanRun(const Data &input, const Data &norm,
                                      const Data &weight, const Data &output);
// Automatic performance selection is distinct from the capability query.
// Explicit fusion callers use Prepare/Run from fastllm-cuda-norm-linear.h.
bool FastllmCudaDecodeNormLinearPreferred(const Data &input, const Data &norm,
                                        const Data &weight, const Data &output);
// Fuses optional residual addition, RMSNorm and projection. The input/branch
// are read-only; BF16 rounding is retained before norm, affine and logits cast.
// An optional FP32 mirror retains its own row stride and capacity. Its owner
// must enable peer access before capture and keep the allocation alive through
// all replays. Each caller writes only its disjoint vocabulary slice.
bool FastllmCudaDecodeNormLinear(const Data &input, const Data *branch,
                                const Data &norm, const Data &weight,
                                Data &output, float eps,
                                const Data *mirror = nullptr, int mirrorOffset = 0);
}
