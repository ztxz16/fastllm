#pragma once
#include "fastllm.h"
// Compact only finite FP32 weights that exactly represent BF16 checkpoint values.
bool FastllmCudaTryCompactBFloat16Weight(const fastllm::Data &source, fastllm::Data &compact);
// BF16 operands, FP32 accumulation and FP32 logits (no output rounding to BF16).
bool FastllmCudaBFloat16LinearFloat32(const fastllm::Data &input, const fastllm::Data &weight,
                                     fastllm::Data &output);
// Pack one or two immutable FP32 matrices only when every value is exactly BF16.
// Preparation is eager-only; the owned packed allocation must outlive captured graphs.
bool FastllmCudaTryPackBFloat16Weights(const fastllm::Data &first,
                                      const fastllm::Data *second, fastllm::Data &packed);
// One BF16 GEMM with FP32 output, split into two owned, non-aliasing feature tensors.
// FP32 accumulation order can differ from separate GEMVs; weights are not rounded.
bool FastllmCudaBFloat16LinearPairFloat32(const fastllm::Data &input,
    const fastllm::Data &packed, int firstWidth, fastllm::Data &first,
    fastllm::Data &second, fastllm::Data &scratch);
