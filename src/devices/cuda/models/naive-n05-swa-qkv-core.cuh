#pragma once
// Small verification blocks share the existing BF16 QKV/RoPE and attention
// bodies. The candidate KV writes complete before any attention reader starts.
// No new persistent scratch or packed weights are needed.
namespace swa_qkv_core {
constexpr int Threads = 1024, Blocks = 128;
__global__ __launch_bounds__(Threads) void QkvAttention(const BF16 *input, const BF16 *weight, BF16 *q,
                                                        BF16 *output,
#if CUDART_VERSION >= 12010 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
                                                        const __grid_constant__
#endif
                                                            SwaQkvBatch qkv,
#if CUDART_VERSION >= 12010 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
                                                        const __grid_constant__
#endif
                                                            BatchSwaKeys attention,
                                                        const float *position, const int *live, int rows,
                                                        float theta, float valueScale, const float *sink) {
    const int task = blockIdx.x;
    if (task < 116)
        swa_qkv_rows::ProjectRope<false, Threads>(input, weight, q, qkv, position, live, rows, theta,
                                                  valueScale);
    // All CTAs are resident; this is a local-device join, not a TP collective.
    __syncthreads();
    cooperative_groups::this_grid().sync();
    if (task < rows * 16)
        SwaAttentionBf16Body<Threads, 64, true, BatchSwaKeys, true, true>(
            q, nullptr, nullptr, sink, output, live, 8, 1, attention, task / 16, task / 2 % 8, task % 2);
}
} // namespace swa_qkv_core
