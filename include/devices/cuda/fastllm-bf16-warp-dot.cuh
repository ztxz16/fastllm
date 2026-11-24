#pragma once

// One warp reduces eight products per lane with the same compensated tree for
// direct BF16 Linear and the Inkling draft score epilogue.
template<int RowsPerWarp, class BeforeInput, class Store>
__device__ __forceinline__ void FastllmBf16WarpDot256(const __nv_bfloat16 *input,
        const __nv_bfloat16 *weight, int outputs, BeforeInput beforeInput, Store store) {
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * RowsPerWarp;
    union alignas(16) Pack { uint4 in; __nv_bfloat16 out[8]; };
    Pack a, b[RowsPerWarp];
    #pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r)
        if (first + r < outputs)
            b[r].in = *reinterpret_cast<const uint4 *>(weight + (size_t)(first + r) * 256 + lane * 8);
    beforeInput();
    if (first >= outputs) return;
    a.in = *reinterpret_cast<const uint4 *>(input + lane * 8);
    #pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) if (first + r < outputs) {
        float value = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            value += __bfloat162float(a.out[i]) * __bfloat162float(b[r].out[i]);
        float correction = 0.0f;
        #pragma unroll
        for (int step = 16; step; step >>= 1) {
            float peer = __shfl_down_sync(0xffffffffu, value, step);
            if (lane < step) {
                float other = peer - correction;
                float sum = value + other;
                correction = (sum - value) - other;
                value = sum;
            }
        }
        if (lane == 0) store(first + r, value);
    }
}
