#pragma once

// The former CPU table rounds the inverse frequency to float before multiplying
// by position. Computing position / powf(...) loses phase accuracy at long positions.
__device__ __forceinline__ float FastllmPreciseRopeAngle(
        float position, int dim, int rotaryDim, float theta) {
    const float exponent = (float)(2 * dim) / rotaryDim;
    const float inverse = (float)(1.0 / pow((double)theta, (double)exponent));
    return __fmul_rn(position, inverse);
}

// One arithmetic core for separate and fused RoPE. RoundProducts preserves
// eager 16-bit cos/sin and multiply boundaries; false keeps FP32 products.
// The caller controls frequency generation and tensor layout independently.
template<class T, bool RoundProducts = false>
__device__ __forceinline__ void FastllmRopePair(T a, T b, float c, float s,
                                             T &outA, T &outB) {
    float x = float(a), y = float(b);
    if constexpr (RoundProducts) {
        c = float(T(c)); s = float(T(s));
        float xc = float(T(x * c)), ys = float(T(y * s));
        float yc = float(T(y * c)), xs = float(T(x * s));
        outA = T(xc - ys); outB = T(yc + xs);
    } else {
        outA = T(x * c - y * s); outB = T(x * s + y * c);
    }
}
