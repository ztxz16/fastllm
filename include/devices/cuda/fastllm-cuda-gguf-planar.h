#pragma once
#include <cstddef>

// Internal Q8 workspace: signed bytes, half2(scale, original sum), then
// short2(sum of the first/last 16 quantized bytes). K is a multiple of 256,
// so every plane and input row is naturally 16-byte aligned.
size_t FastllmGgufPlanarBytes(int tokens, int columns);
bool FastllmGgufPlanarSupported(int type, int tokens, int columns, int outputRows);
// Input kind: 0 = float, 1 = half, 2 = bfloat16. Optional head permutation
// is applied while reading the input, before quantization.
bool FastllmGgufQuantizePlanar(const void *input, int inputKind, void *workspace, int tokens, int columns,
                               void *stream, int keyHeads = 0, int groups = 0, int headDim = 0);
// FP16 output. Mode 0: plain; 1: residual; 2: up + SiLU(existing gate);
// 3: same-format gate/up, preserving each intermediate FP16 rounding.
bool FastllmGgufProjectPlanar(int type, int mode, const void *weight, const void *upWeight,
                              const void *workspace, void *output, int tokens, int columns, int outputRows,
                              int outputStride, void *stream);

// Fused gate/up with independently selected GGUF formats. All supported
// pairs preserve the same FP16 intermediate rounding as separate projections.
bool FastllmGgufGateUpPlanar(int gateType, int upType, const void *gate, const void *up,
                             const void *workspace, void *output, int tokens, int columns, int outputRows,
                             int outputStride, void *stream);
