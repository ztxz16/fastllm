#pragma once
#include "fastllm.h"
// Compact only finite FP32 weights that exactly represent BF16 checkpoint values.
bool FastllmCudaTryCompactBFloat16Weight(const fastllm::Data &source, fastllm::Data &compact);
// BF16 operands, FP32 accumulation and FP32 logits (no output rounding to BF16).
bool FastllmCudaBFloat16LinearFloat32(const fastllm::Data &input, const fastllm::Data &weight,
                                     fastllm::Data &output);
// Combine adjacent 32-row BF16 projections using the same Lt algorithm and
// FP32 split-K reduction. Unsupported shapes/architectures retain the caller's
// ordinary Linear loop. Output must already own enough CUDA storage.
bool FastllmCudaBFloat16LinearFixedRows(const fastllm::Data &input,
    const fastllm::Data &weight, fastllm::Data &output);
// Selected once on the current CUDA device. A null backend or a declined
// shape leaves the caller on its ordinary Linear path.
struct FastllmBf16MmaProjections {
    using Project = bool (*)(const void *input, const void *weight, void *output, int rows, int width, int columns);
    Project attention;
    Project draftLogits;
    bool (*cappedLogits)(const void *input, const void *weight, float *output, int rows, int width, int vocab, float cap);
};
const FastllmBf16MmaProjections *FastllmCudaGetBf16MmaProjections();
void FastllmCudaBf16ExactRows(const void *input, const void *weight, void *output,
                             int rows, int width, int columns);
// Preserve the singleton GEMV tree when the projection feeds recurrent state or routing.
bool FastllmCudaBf16ExactBatchProjection(const void *input, const void *weight, void *output,
                                        int rows, int width, int columns);
// Dense output head with the same cuBLASLt reduction for decode and verification.
bool FastllmCudaBf16DecodeProjection(const void *input, const void *weight, void *output,
                                    int rows, int width, int columns);
// Packed BF16 prefill with a batch-independent reduction order.
bool FastllmCudaBFloat16PrefillLinear(const fastllm::Data &input,
    const fastllm::Data &weight, fastllm::Data &output);
// Pack one or two immutable FP32 matrices only when every value is exactly BF16.
// Preparation is eager-only; the owned packed allocation must outlive captured graphs.
bool FastllmCudaTryPackBFloat16Weights(const fastllm::Data &first,
                                      const fastllm::Data *second, fastllm::Data &packed);
// One BF16 GEMM with FP32 output, split into two owned, non-aliasing feature tensors.
// FP32 accumulation order can differ from separate GEMVs; weights are not rounded.
bool FastllmCudaBFloat16LinearPairFloat32(const fastllm::Data &input,
    const fastllm::Data &packed, int firstWidth, fastllm::Data &first,
    fastllm::Data &second, fastllm::Data &scratch);
