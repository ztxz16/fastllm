#pragma once

template<bool FloatLog = false>
__device__ __forceinline__ float FastllmSoftplus(float x) {
    if (x > 20.f) return x;
    if constexpr (FloatLog) return log1pf(expf(x));
    else return log1p(expf(x));
}
