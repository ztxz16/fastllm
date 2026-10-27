#pragma once

// The former CPU table rounds the inverse frequency to float before multiplying
// by position. Computing position / powf(...) loses phase accuracy at long positions.
__device__ __forceinline__ float FastllmPreciseRopeAngle(
        float position, int dim, int rotaryDim, float theta) {
    const float exponent = (float)(2 * dim) / rotaryDim;
    const float inverse = (float)(1.0 / pow((double)theta, (double)exponent));
    return __fmul_rn(position, inverse);
}
