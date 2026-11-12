#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace fastllm::minimax_h3_image::detail {
// Internal SM120 backend: row-major X[M,K], row-major W[N,K], row-major Y[M,N].
// The epilogue applies the dynamic per-row X and per-output-channel W scales.
void fp8ScaledSm120(const void *x, const void *w, const float *xs, const float *ws, void *y, int m, int n,
                    int k, int device, int multiprocessors, void *workspace, size_t workspaceBytes,
                    cudaStream_t stream);
} // namespace fastllm::minimax_h3_image::detail
