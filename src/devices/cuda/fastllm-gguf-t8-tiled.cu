#include "fastllm-gguf-kernel-common.cuh"
#include "fastllm-gguf-small-mmvq.cuh"

namespace fastllm_gguf_small_mmvq {

template <ggml_type Type, int K, typename Output, int StoreMode, int Warps, int Tile>
__global__ __launch_bounds__(Warps * 32, Warps <= 8 ? 4 : 2) void TiledT8GemvKernel(
    const void *__restrict__ weights, const block_q8_1 *__restrict__ input, Output *__restrict__ output,
    int outputRows, int inputStride, int outputStride) {
    static_assert(Type == GGML_TYPE_IQ4_XS || Type == GGML_TYPE_IQ3_XXS);
    __shared__ uint32_t table[CodebookWords<Type>];
    __shared__ __align__(16) block_q8_1 cache[8][Tile / 32];
    if constexpr (Type != GGML_TYPE_IQ4_XS)
        LoadCodebook<Type, Warps>(table);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row0 = blockIdx.x * (2 * Warps) + warp;
    float sums0[8] = {}, sums1[8] = {};
    for (int first = 0; first < K; first += Tile) {
        const int count = min(Tile, K - first);
        LoadInputTile<8, Warps, Tile>(cache, input, inputStride, first, count);
        __syncthreads();
        for (int b = lane / 8; b < count / 256; b += 4) {
            int w0[8], w1[8], sc0, sc1;
            float d0, d1;
            DecodeBatchWeights<Type, true>(weights, min(row0, outputRows - 1) * (K / 256) + first / 256 + b,
                                           2 * (lane % 8), table, w0, d0, sc0);
            DecodeBatchWeights<Type, true>(weights,
                                           min(row0 + Warps, outputRows - 1) * (K / 256) + first / 256 + b,
                                           2 * (lane % 8), table, w1, d1, sc1);
            // For nonnegative scale s, trunc((s*d + trunc(d/2))/2)
            // equals trunc((2*s+1)*d/4), including negative d. The bounded
            // 32-value int8 dot cannot overflow either integer expression.
            if constexpr (Type == GGML_TYPE_IQ3_XXS) {
                sc0 = 2 * sc0 + 1;
                sc1 = 2 * sc1 + 1;
            }
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const auto &x = cache[t][b * 8 + lane % 8];
                int dot0 = 0, dot1 = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int a = get_int_b4(x.qs, j);
                    dot0 = ggml_cuda_dp4a(w0[j], a, dot0);
                    dot1 = ggml_cuda_dp4a(w1[j], a, dot1);
                }
                dot0 *= sc0;
                dot1 *= sc1;
                if constexpr (Type == GGML_TYPE_IQ3_XXS) {
                    dot0 /= 4;
                    dot1 /= 4;
                }
                const float dx = __low2float(x.ds);
                sums0[t] += (d0 * dx) * dot0;
                sums1[t] += (d1 * dx) * dot1;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float v0 = warp_reduce_sum(sums0[t]), v1 = warp_reduce_sum(sums1[t]);
        if (lane == 0) {
            if (row0 < outputRows)
                FastllmGgufStore<StoreMode>(output + t * outputStride + row0, v0);
            if (row0 + Warps < outputRows)
                FastllmGgufStore<StoreMode>(output + t * outputStride + row0 + Warps, v1);
        }
    }
}
template <ggml_type Type, int K, typename Output, int StoreMode, int Warps, int Tile>
static void LaunchTiledT8(const void *weights, const block_q8_1 *input, Output *output, int n,
                          int inputStride, int outputStride, cudaStream_t stream) {
    TiledT8GemvKernel<Type, K, Output, StoreMode, Warps, Tile>
        <<<(n + 2 * Warps - 1) / (2 * Warps), Warps * 32, 0, stream>>>(weights, input, output, n,
                                                                       inputStride / 32, outputStride);
}

template <ggml_type Type, int K, typename Output, int StoreMode>
static bool LaunchSelected(const void *w, const block_q8_1 *x, void *y, int n, int inputStride,
                           int outputStride, cudaStream_t stream) {
    // The model trace favors the previous register route for small IQ4_XS
    // matrices. Keep the wider tile only where its model timing improves.
    if constexpr (Type == GGML_TYPE_IQ4_XS && K == 6144) {
        return false;
    } else {
        if constexpr (Type == GGML_TYPE_IQ4_XS && K == 5120) {
            if (n < 12288)
                return false;
        }
        if constexpr (Type == GGML_TYPE_IQ3_XXS && K == 5120 && StoreMode == 0) {
            if (n <= 10240) {
                LaunchTiledT8<Type, K, Output, StoreMode, 16, 4096>(w, x, static_cast<Output *>(y), n,
                                                                    inputStride, outputStride, stream);
                return true;
            }
        }
        // In particular, fused gate stores favor this 8-warp schedule.
        LaunchTiledT8<Type, K, Output, StoreMode, 8, 2048>(w, x, static_cast<Output *>(y), n, inputStride,
                                                           outputStride, stream);
        return true;
    }
}

template <ggml_type Type, typename Output, int StoreMode>
static bool DispatchColumns(const void *w, const block_q8_1 *x, void *y, int k, int n, int inputStride,
                            int outputStride, cudaStream_t stream) {
#define FASTLLM_TILED_K(K)                                                                                   \
    case K:                                                                                                  \
        return LaunchSelected<Type, K, Output, StoreMode>(w, x, y, n, inputStride, outputStride, stream)
    switch (k) {
        FASTLLM_TILED_K(5120);
        FASTLLM_TILED_K(6144);
        FASTLLM_TILED_K(17408);
    }
#undef FASTLLM_TILED_K
    return false;
}

template <ggml_type Type>
static bool DispatchOutput(int outputKind, int storeMode, const void *w, const block_q8_1 *x, void *y, int k,
                           int n, int inputStride, int outputStride, cudaStream_t stream) {
    if (outputKind == 1) {
#define FASTLLM_TILED_STORE(S)                                                                               \
    case S:                                                                                                  \
        return DispatchColumns<Type, half, S>(w, x, y, k, n, inputStride, outputStride, stream)
        switch (storeMode) {
            FASTLLM_TILED_STORE(0);
            FASTLLM_TILED_STORE(1);
            FASTLLM_TILED_STORE(2);
        }
#undef FASTLLM_TILED_STORE
    } else if (storeMode == 0) {
        if (outputKind == 0)
            return DispatchColumns<Type, float, 0>(w, x, y, k, n, inputStride, outputStride, stream);
        if (outputKind == 2)
            return DispatchColumns<Type, __nv_bfloat16, 0>(w, x, y, k, n, inputStride, outputStride, stream);
    }
    return false;
}

bool DispatchTiledT8(ggml_type type, int outputKind, int storeMode, const void *weights,
                     const block_q8_1 *input, void *output, int columns, int rows, int inputStride,
                     int outputStride, cudaStream_t stream) {
    if (rows < 4096)
        return false;
#define FASTLLM_TILED_TYPE(T)                                                                                \
    case T:                                                                                                  \
        return DispatchOutput<T>(outputKind, storeMode, weights, input, output, columns, rows, inputStride,  \
                                 outputStride, stream)
    switch (type) {
        FASTLLM_TILED_TYPE(GGML_TYPE_IQ4_XS);
        FASTLLM_TILED_TYPE(GGML_TYPE_IQ3_XXS);
    default:
        return false;
    }
#undef FASTLLM_TILED_TYPE
}
} // namespace fastllm_gguf_small_mmvq

// The MMQ translation unit includes small-mmvq.cuh inside its own namespace.
// Forward that entry point to the same instantiations instead of duplicating
// all tiled kernels in the legacy MMQ compilation unit.
namespace fastllm_gguf_mmq::fastllm_gguf_small_mmvq {
bool DispatchTiledT8(ggml_type type, int outputKind, int storeMode, const void *weights,
                     const block_q8_1 *input, void *output, int columns, int rows, int inputStride,
                     int outputStride, cudaStream_t stream) {
    return ::fastllm_gguf_small_mmvq::DispatchTiledT8(type, outputKind, storeMode, weights, input, output,
                                                      columns, rows, inputStride, outputStride, stream);
}
} // namespace fastllm_gguf_mmq::fastllm_gguf_small_mmvq
