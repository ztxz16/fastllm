#pragma once
// Included inside fastllm_gguf_small_mmvq after the common GGUF decoders.
bool DispatchTiledT8(ggml_type type, int outputKind, int storeMode, const void *weights,
                     const block_q8_1 *input, void *output, int columns, int rows, int inputStride,
                     int outputStride, cudaStream_t stream);
