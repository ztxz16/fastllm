#pragma once

// Included inside fastllm_gguf_small_mmvq, after the shared GGUF decoders.
// Keep FastLLM's integer scaling/rounding for the selected formats.
template <ggml_type Type>
static constexpr bool ExtendedT8Type =
    Type == GGML_TYPE_IQ2_S || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_XXS ||
    Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_Q2_K;

// outputKind: 0=FP32, 1=FP16, 2=BF16; fused stores require FP16.
// Implemented in separate CUDA translation units to keep the new
// specializations out of the large legacy GGUF translation units.
bool DispatchExtendedT8(ggml_type type, int outputKind, int storeMode, const void *weights,
                        const block_q8_1 *input, void *output, int columns, int rows, int inputStride,
                        int outputStride, cudaStream_t stream);
