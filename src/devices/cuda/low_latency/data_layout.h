#pragma once
#include "fastllm.h"

namespace fastllm::cuda::low_latency {
// FastLLM Data adapter only. Raw operator plans never depend on Data/model state.
inline bool DenseData(const Data &data, int device) {
    if (data.dataDevice != DataDevice::CUDA || !data.cudaData || data.multiDeviceData || data.IsRepacked ||
        data.dataDeviceIds != std::vector<int>{device} || uintptr_t(data.cudaData) % 16 ||
        data.dims.empty() || data.strides.size() != data.dims.size()) return false;
    uint64_t stride = 1;
    for (int i = int(data.dims.size())-1; i >= 0; --i) {
        if (data.dims[i] <= 0 || data.strides[i] != stride) return false;
        if (stride > UINT64_MAX / data.dims[i]) return false;
        stride *= data.dims[i];
    }
    return true;
}
inline bool DataOverlap(const Data &a, const Data &b) {
    if (&a == &b) return true;
    if (!a.cudaData || !b.cudaData) return false;
    const uintptr_t x = uintptr_t(a.cudaData), y = uintptr_t(b.cudaData);
    return x <= y ? y-x < a.GetBytes() : x-y < b.GetBytes();
}
}
