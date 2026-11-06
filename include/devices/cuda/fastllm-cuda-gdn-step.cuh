#pragma once

namespace fastllm { namespace cuda_gdn {
// One warp owns one value row, with four K elements per lane. Normalize Q/K
// and compute gates before this core. StateT selects the committed precision;
// the output dot always uses the unrounded FP32 update (including FP16 state).
template<class StateT, bool FuseDecayUpdate = false>
__device__ __forceinline__ float Step128(StateT (&state)[4], const float (&q)[4],
        const float (&k)[4], float value, float beta, float decay, float qScale = 1.f) {
    float decayed[4], sumK = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        decayed[j] = float(state[j]) * decay;
        sumK += decayed[j] * k[j];
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        sumK += __shfl_down_sync(0xffffffff, sumK, offset);
    float delta = (value - __shfl_sync(0xffffffff, sumK, 0)) * beta;
    float sumQ = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        // Some FP16 backends fuse decay into the update, while FP32-state
        // recurrence materializes decay before computing the residual.
        float updated;
        if constexpr (FuseDecayUpdate)
            updated = __fmaf_rn(float(state[j]), decay, __fmul_rn(k[j], delta));
        else updated = decayed[j] + k[j] * delta;
        state[j] = StateT(updated);
        sumQ += updated * (q[j] * qScale);
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        sumQ += __shfl_down_sync(0xffffffff, sumQ, offset);
    return __shfl_sync(0xffffffff, sumQ, 0);
}
}} // namespace fastllm::cuda_gdn
