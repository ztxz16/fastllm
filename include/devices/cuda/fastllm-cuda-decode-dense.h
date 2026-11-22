#pragma once
#include "fastllm.h"

namespace fastllm {
// Portable, local BF16 SwiGLU: gate+up+SiLU+multiply, then down. Input is
// already normalized. Output is a TP partial, without residual or reduction.
// No architecture-specific MoE/collective state and no packed weight copies.
// Capability rejection occurs before output resizing or any kernel launch.
bool FastllmCudaDecodeDense(const Data &input, const Data &gate, const Data &up,
                           const Data &down, Data &intermediate, Data &output);
}
