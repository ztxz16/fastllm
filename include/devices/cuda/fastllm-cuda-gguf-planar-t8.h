#pragma once
#include <cstddef>

// Internal verifier layout: eight rows of 5120 signed bytes, followed by
// eight rows of 160 half2 (scale, original sum). All planes are 16-byte aligned.
constexpr size_t FASTLLM_GGUF_PLANAR_T8_BYTES = 8 * 5120 + 8 * 160 * 4;
bool FastllmGgufPlanarT8Supported(int type);
void FastllmGgufQuantizePlanarT8(const void *input, void *workspace, void *stream);
// mode 0: projection; 2: SiLU(existing rounded gate) * rounded up;
// mode 3: two same-format weights, with the same intermediate FP16 rounding.
bool FastllmGgufProjectPlanarT8(int type, int mode, const void *weight, const void *upWeight,
                                const void *workspace, void *output, int outputRows, int outputStride,
                                void *stream);
