#pragma once

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
