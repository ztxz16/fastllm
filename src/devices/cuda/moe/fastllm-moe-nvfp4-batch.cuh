#pragma once

// Batch the existing compact E4M3 arithmetic without changing its virtual
// 64-lane accumulation. Gate CTAs cover output rows; down CTAs cover routes.
// Only physical CUDA grid dimensions are chunked, not the logical batch.
template<typename T, bool Gate, int Warps = Gate ? 4 : 16>
__global__ void NVFP4BatchProjection(const T *input, T *activation, float *partial,
        FastllmCudaMoeGGUFCacheView view, int topk, int scaleCols,
        size_t scalesOffset, int firstRoute) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int route = firstRoute + (Gate ? blockIdx.y : blockIdx.y * Warps + warp);
    const int row = Gate ? blockIdx.x * Warps + warp : blockIdx.x;
    if (route >= view.routeCount || (view.activeRouteCount && route >= *view.activeRouteCount) ||
            row >= (Gate ? view.inter : view.hidden)) return;
    const int original = view.routeMap ? view.routeMap[route] : route;
    const int slot = view.routeSlots[route], columns = Gate ? view.hidden : view.inter;
    if (Gate && slot < 0) return;
    float low = 0, high = 0, upLow = 0, upHigh = 0;
    if (slot >= 0) {
        const uint8_t *record = view.records +
            (view.slotOffsets ? view.slotOffsets[slot] : size_t(slot) * view.recordStride);
        const float *scales = reinterpret_cast<const float *>(record + scalesOffset);
        const uint8_t *weight = record + (Gate ? 0 : view.downOffset);
        const T *x = input + size_t(Gate ? original / topk : route) * columns;
        for (int word = lane; word < columns / 8; word += 32) {
            AccumulateCompactE4M3WordPair(x, word, weight, row,
                Gate ? 2 * view.inter : view.hidden, columns, scaleCols,
                scales[Gate ? 0 : 2], low, high);
            if constexpr (Gate)
                AccumulateCompactE4M3WordPair(x, word, weight, row + view.inter,
                    2 * view.inter, columns, scaleCols, scales[1], upLow, upHigh);
        }
    }
    const float value = ActivationTraits<T>::ToFloat(
        ActivationTraits<T>::FromFloat(ReduceVirtual64(low, high)));
    if constexpr (Gate) {
        const float up = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(ReduceVirtual64(upLow, upHigh)));
        if (lane == 0) activation[size_t(route) * view.inter + row] =
            ActivationTraits<T>::FromFloat(value / (1.0f + expf(-value)) * up);
    } else if (lane == 0) partial[size_t(original) * view.hidden + row] = value;
}

template<typename T>
bool LaunchNVFP4Batch(const fastllm::Data &input, fastllm::Data &activation,
        const OffloadLayout &layout, const FastllmCudaMoeGGUFCacheView &view,
        int topk, float *partial) {
    for (int first = 0; first < view.routeCount; first += 65535) {
        const int count = std::min(65535, view.routeCount - first);
        NVFP4BatchProjection<T,true><<<dim3((layout.inter+3)/4,count),128,0,cudaStreamPerThread>>>(
            static_cast<const T *>(input.cudaData),static_cast<T *>(activation.cudaData),partial,
            view,topk,layout.gateScaleCols,layout.scalesOffset,first);
    }
    for (int first = 0; first < view.routeCount; first += 65535 * 16) {
        const int count = std::min(65535 * 16, view.routeCount - first);
        NVFP4BatchProjection<T,false><<<dim3(layout.hidden,(count+15)/16),512,0,cudaStreamPerThread>>>(
            static_cast<const T *>(activation.cudaData),nullptr,partial,
            view,topk,layout.downScaleCols,layout.scalesOffset,first);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool ComputeNVFP4Batch(const fastllm::Data &input, fastllm::Data &activation,
        const OffloadLayout &layout, const FastllmCudaMoeGGUFCacheView &view,
        int topk, float *partial) {
    if (!SupportsNVFP4WideDecode(layout) || !partial || !view.records ||
            !view.routeSlots || view.routeCount < 1) return false;
    switch (input.dataType) {
    case fastllm::FLOAT32: return LaunchNVFP4Batch<float>(input,activation,layout,view,topk,partial);
    case fastllm::FLOAT16: return LaunchNVFP4Batch<half>(input,activation,layout,view,topk,partial);
    case fastllm::BFLOAT16: return LaunchNVFP4Batch<__nv_bfloat16>(input,activation,layout,view,topk,partial);
    default: return false;
    }
}
