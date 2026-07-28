#pragma once

#include <cstddef>

#include "cutlass/numeric_types.h"

namespace fastllm::cuda::w4a8 {

bool EncodeInt4ForCutlass(const cutlass::int4b_t *input,
                          cutlass::int4b_t *output,
                          size_t elementCount);

} // namespace fastllm::cuda::w4a8
