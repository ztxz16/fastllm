#pragma once

// Gate/up retains the ordinary compact arithmetic. Down uses the same
// independent-expert kernel as scalar decode, including its exact reduction.
// Route remapping and device active counts are shared by verification/draft.
template<typename T, int Warps = 4>
__global__ void NVFP4BatchGate(const T *input, T *activation,
        FastllmCudaMoeGGUFCacheView view, int topk, int scaleCols,
        size_t scalesOffset, int firstRoute) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int route = firstRoute + blockIdx.y;
    const int row = blockIdx.x * Warps + warp;
    if (route >= view.routeCount || (view.activeRouteCount && route >= *view.activeRouteCount) ||
            row >= view.inter) return;
    const int original = view.routeMap ? view.routeMap[route] : route;
    const int slot = view.routeSlots[route];
    if (slot < 0) return;
    const uint8_t *record = view.records +
        (view.slotOffsets ? view.slotOffsets[slot] : size_t(slot) * view.recordStride);
    const float *scales = reinterpret_cast<const float *>(record + scalesOffset);
    const T *x = input + size_t(original / topk) * view.hidden;
    float low = 0, high = 0, upLow = 0, upHigh = 0;
    for (int word = lane; word < view.hidden / 8; word += 32) {
        AccumulateCompactE4M3WordPair(x, word, record, row,
            2 * view.inter, view.hidden, scaleCols, scales[0], low, high);
        AccumulateCompactE4M3WordPair(x, word, record, row + view.inter,
            2 * view.inter, view.hidden, scaleCols, scales[1], upLow, upHigh);
    }
    const float value = ActivationTraits<T>::ToFloat(
        ActivationTraits<T>::FromFloat(ReduceVirtual64(low, high)));
    const float up = ActivationTraits<T>::ToFloat(
        ActivationTraits<T>::FromFloat(ReduceVirtual64(upLow, upHigh)));
    if (lane == 0) activation[size_t(route) * view.inter + row] =
        ActivationTraits<T>::FromFloat(value / (1.0f + expf(-value)) * up);
}

template<typename T>
bool LaunchNVFP4Batch(const fastllm::Data &input, fastllm::Data &activation,
        const OffloadLayout &layout, const FastllmCudaMoeGGUFCacheView &view,
        int topk, float *partial) {
    for (int first = 0; first < view.routeCount; ) {
        const int count = std::min(65535, view.routeCount - first);
        NVFP4BatchGate<T><<<dim3((layout.inter+3)/4,count),128,0,cudaStreamPerThread>>>(
            static_cast<const T *>(input.cudaData),static_cast<T *>(activation.cudaData),
            view,topk,layout.gateScaleCols,layout.scalesOffset,first);
        first += count;
    }
    for (int first = 0; first < view.routeCount; ) {
        const int count = std::min(65535, view.routeCount - first);
        constexpr int outputRows = 16;
        FastllmNVFP4DownExpertsKernel<T, outputRows>
            <<<dim3((layout.hidden + outputRows - 1) / outputRows, count),
               outputRows * 16, 0, cudaStreamPerThread>>>(
                static_cast<const T *>(activation.cudaData), view.routeSlots,
                view.records, partial, layout.inter, layout.hidden,
                layout.downScaleCols, view.recordStride, view.downOffset,
                layout.scalesOffset, view.routeMap, view.slotOffsets,
                view.activeRouteCount, first);
        first += count;
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
