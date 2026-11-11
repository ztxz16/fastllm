#pragma once
#include "models/hy_nomas/runtime.h"

namespace fastllm::minimax_h3_image {
// Native checkpoint packs [gate, value]. Round SiLU before multiplying the value.
nomas::Tensor gatedSilu(const nomas::Tensor &);
} // namespace fastllm::minimax_h3_image
