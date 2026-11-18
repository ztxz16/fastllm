#pragma once
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"

namespace fastllm {
// Called only after a synchronous successful pack, before workers/graphs can
// see the model. The Data object remains metadata; the backend is now the sole
// owner of its payload. No CPU mirror is retained as a hidden fallback cache.
inline size_t ReleasePackedMoeSource(Data &source) {
    AssertInFastLLM(!source.isFake && !source.cudaDataBorrowed &&
        source.extraCudaData.empty() && source.extraCudaHalfData.empty(),
        "Packed MoE source must have exclusive ownership and no live auxiliary cache.");
    const size_t bytes = source.cudaData ? source.GetBytes() : 0;
    if (source.cudaData) {
        AssertInFastLLM(source.dataDeviceIds.size() == 1 && (source.directMemory ||
            (source.cudaRepackSource && FastllmCudaIsWeightSlabPointer(source.cudaData))),
            "Packed MoE source must use direct storage or a releasable source slab.");
        FastllmCudaSetDevice(source.dataDeviceIds[0]);
    }
    source.FreeSpace();
    source.mapFile.reset();
    std::vector<float>().swap(source.scales);
    std::vector<float>().swap(source.mins);
    std::vector<int>().swap(source.zeros);
    std::vector<int>().swap(source.weightSum);
    std::vector<uint16_t>().swap(source.halfScales);
    return bytes;
}
}
