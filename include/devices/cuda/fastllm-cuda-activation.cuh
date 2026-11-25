#pragma once

#include <cuda_bf16.h>

template<bool FloatLog = false>
__device__ __forceinline__ float FastllmSoftplus(float x) {
    if (x > 20.f) return x;
    if constexpr (FloatLog) return log1pf(expf(x));
    else return log1p(expf(x));
}

__device__ __forceinline__ float FastllmSiTUAndMul(float gate, float up, float gateCap, float upCap) {
    float sigmoid = 1.0f / (1.0f + expf(-gate));
    float activated = gateCap * tanhf(gate / gateCap) * sigmoid;
    float bounded = upCap > 0.0f ? upCap * tanhf(up / upCap) : up;
    return activated * bounded;
}

namespace fastllm::cuda {
template <typename T> struct RoutedSiTU {
    const T *pair;
    const float *scores;
    float gateCap, upCap;
    __device__ float get(size_t row, int col, int d) const { return get(row, row, col, d); }
    __device__ float get(size_t row, size_t route, int col, int d, float outer = 1.f) const {
        const T *gate = pair + row * (2 * d) + col;
        float g = float(__float2bfloat16_rn(float(gate[0]) * outer));
        float u = float(__float2bfloat16_rn(float(gate[d]) * outer));
        return float(__float2bfloat16_rn(FastllmSiTUAndMul(g, u, gateCap, upCap) * scores[route]));
    }
};

// Gate products are in expert order. Route scores and expert IDs remain in top-k order.
struct GroupedRoutedSiTU {
    RoutedSiTU<float> activation;
    const int *routeGroups, *ids;
    const float *scales;

    __device__ float get(int group, int route, int col, int width) const {
        return activation.get(group, route, col, width, scales ? scales[ids[route]] : 1.f);
    }
    __device__ float get(size_t route, int col, int width) const {
        const int group = routeGroups[route];
        return group < 0 ? 0.f : get(group, route, col, width);
    }
};
} // namespace fastllm::cuda
