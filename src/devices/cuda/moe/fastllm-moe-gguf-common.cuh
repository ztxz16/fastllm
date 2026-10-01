#pragma once

#include "fastllm.h"
#include <vector>

namespace fastllm_gguf_moe {

// Resident and streamed experts use the same caller-owned CUDA scratch policy.
inline void AllocateTensor(fastllm::Data &data, fastllm::DataType type,
                           const std::vector<int> &shape, int device) {
    if (data.dataType != type || (!data.cudaData && data.expansionSize)) data.FreeSpace();
    data.dataType = type;
    data.UpdateUnitSize();
    data.ToDevice(fastllm::CUDA, {device}, false);
    data.Resize(shape);
    data.Allocate(false);
}

inline bool IsActivationType(fastllm::DataType type) {
    return type == fastllm::FLOAT32 || type == fastllm::FLOAT16 || type == fastllm::BFLOAT16;
}

} // namespace fastllm_gguf_moe
