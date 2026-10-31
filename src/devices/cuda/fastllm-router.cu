#include "devices/cuda/fastllm-cuda-router.h"
#include "devices/cuda/fastllm-cuda-ordered-reduce.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
// Reuse the common TopK selector. Only the first warp selects; all 256 lanes
// keep the original expert-id score reduction, including masked/zero scores.
template<class T>
__global__ void StableSoftmaxSelectExpert(const T *logits, int *indices, float *scores,
        int experts, int topk, bool normalizeTopK, int stride,
        const unsigned char *allowed, int maskRows) {
    __shared__ float values[256], maximum, reduction[8];
    __shared__ int ranks[256];
    const int e = threadIdx.x, row = blockIdx.x;
    float x = e < experts ? float(logits[(size_t)row * stride + e]) : -CUDART_INF_F;
    if (allowed && e < experts && !allowed[(row / maskRows) * experts + e]) x = -CUDART_INF_F;
    values[e] = x; ranks[e] = -1;
    __syncthreads();
    if (e < 32) {
        float keys[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) keys[i] = values[e * 8 + i];
        for (int rank = 0; rank < topk; ++rank) {
            auto winner = fastllm::cuda::WarpArgMax(keys, e * 8, experts);
            if (e == 0) {
                ranks[winner.index] = rank;
                if (rank == 0) maximum = winner.value;
                indices[(size_t)row * topk + rank] = winner.index;
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                if (e * 8 + i == winner.index) keys[i] = CUDART_NAN_F;
        }
    }
    __syncthreads();
    const int rank = ranks[e];
    float p = e < experts ? expf(x - maximum) : 0.f;
    float denominator = fastllm::cuda::WarpFirstBlockSum256(
        (!normalizeTopK || rank >= 0) ? p : 0.f, reduction);
    if (rank >= 0) scores[(size_t)row * topk + rank] = p / fmaxf(denominator, 1.e-12f);
}
}

bool FastllmCudaFusedSoftmaxSelectExpert(const void *logits, int *indices, float *scores,
        int rows, int experts, int topk, bool normalizeTopK, int rowStride,
        const unsigned char *allowed, int maskRows, void *stream, bool fp16) {
    if (!logits || !indices || !scores || rows < 1 || experts < 1 || experts > 256 ||
        topk < 1 || topk > experts || topk > 16 || maskRows < 1) return false;
    if (!rowStride) rowStride = experts;
    if (rowStride < experts) return false;
    auto s = stream ? (cudaStream_t)stream : cudaStreamPerThread;
    if (fp16) StableSoftmaxSelectExpert<<<rows, 256, 0, s>>>((const __half *)logits,
        indices, scores, experts, topk, normalizeTopK, rowStride, allowed, maskRows);
    else StableSoftmaxSelectExpert<<<rows, 256, 0, s>>>((const __nv_bfloat16 *)logits,
        indices, scores, experts, topk, normalizeTopK, rowStride, allowed, maskRows);
    return cudaGetLastError() == cudaSuccess;
}
