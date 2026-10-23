#pragma once
#include "fastllm.h"
// Compact only finite FP32 weights that exactly represent BF16 checkpoint values.
bool FastllmCudaTryCompactBFloat16Weight(const fastllm::Data &source, fastllm::Data &compact);
// BF16 operands, FP32 accumulation and FP32 logits (no output rounding to BF16).
bool FastllmCudaBFloat16LinearFloat32(const fastllm::Data &input, const fastllm::Data &weight,
                                     fastllm::Data &output);
