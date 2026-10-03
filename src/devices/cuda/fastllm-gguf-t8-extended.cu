#include "fastllm-gguf-kernel-common.cuh"
#include "fastllm-gguf-small-mmvq.cuh"

namespace fastllm_gguf_small_mmvq {

template <ggml_type Type, int K, typename Output, int StoreMode = 0>
__global__ __launch_bounds__(128) void ExtendedT8GemvKernel(const void *__restrict__ weights,
                                                            const block_q8_1 *__restrict__ input,
                                                            Output *__restrict__ output, int outputRows,
                                                            int inputStride, int outputStride) {
    static_assert(ExtendedT8Type<Type> && K % 1024 == 0);
    constexpr int tokens = 8, warps = 4;
    __shared__ __align__(8) uint32_t table[CodebookWords<Type>];
    if constexpr (Type != GGML_TYPE_Q2_K) {
        LoadCodebook<Type, warps>(table);
        __syncthreads();
    }
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    for (int row0 = (blockIdx.x * warps + warp) * 2; row0 < outputRows; row0 += gridDim.x * warps * 2) {
        float sums[2][tokens] = {};
#pragma unroll
        for (int it = 0; it < K / 1024; ++it) {
            const int slice = it * WARP_SIZE + lane;
            int activation[tokens][8], sumLo[tokens], sumHi[tokens];
            float dx[tokens];
#pragma unroll
            for (int t = 0; t < tokens; ++t) {
                const auto &x = input[t * inputStride + slice];
                dx[t] = __low2float(x.ds);
                int lo = 0, hi = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    activation[t][j] = get_int_b4(x.qs, j);
                    if constexpr (Type == GGML_TYPE_Q2_K) {
                        if (j < 4)
                            lo = ggml_cuda_dp4a(0x01010101, activation[t][j], lo);
                        else
                            hi = ggml_cuda_dp4a(0x01010101, activation[t][j], hi);
                    }
                }
                sumLo[t] = lo;
                sumHi[t] = hi;
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = min(row0 + r, outputRows - 1);
                const int block = row * (K / QK_K) + slice / 8, group = lane % 8;
                int values[8], scale = 0, sc0 = 0, sc1 = 0;
                float d = 0, dmin = 0;
                if constexpr (Type == GGML_TYPE_Q2_K) {
                    const auto *w = static_cast<const block_q2_K *>(weights) + block;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        values[j] =
                            (get_int_b4(w->qs, (group / 4) * 8 + j) >> (2 * (group % 4))) & 0x03030303;
                    sc0 = w->scales[2 * group];
                    sc1 = w->scales[2 * group + 1];
                    const float2 dm = __half22float2(w->dm);
                    d = dm.x;
                    dmin = dm.y;
                } else {
                    DecodeBatchWeights<Type, true>(weights, block, 2 * group, table, values, d, scale);
                }
#pragma unroll
                for (int t = 0; t < tokens; ++t) {
                    if constexpr (Type == GGML_TYPE_Q2_K || Type == GGML_TYPE_IQ2_S ||
                                  Type == GGML_TYPE_IQ2_XS) {
                        int lo = 0, hi = 0;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            lo = ggml_cuda_dp4a(values[j], activation[t][j], lo);
                            hi = ggml_cuda_dp4a(values[j + 4], activation[t][j + 4], hi);
                        }
                        if constexpr (Type == GGML_TYPE_Q2_K)
                            sums[r][t] += d * (dx[t] * (lo * (sc0 & 15) + hi * (sc1 & 15))) -
                                          dmin * (dx[t] * (sumLo[t] * (sc0 >> 4) + sumHi[t] * (sc1 >> 4)));
                        else {
                            const int dot = (lo * (scale & 15) + hi * (scale >> 4) + (lo + hi) / 2) / 4;
                            sums[r][t] += (d * dx[t]) * dot;
                        }
                    } else {
                        int dot = 0;
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            dot = ggml_cuda_dp4a(values[j], activation[t][j], dot);
                        if constexpr (Type == GGML_TYPE_IQ2_XXS)
                            dot = (scale * dot + dot / 2) / 4;
                        else
                            dot *= scale;
                        sums[r][t] += (d * dx[t]) * dot;
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int t = 0; t < tokens; ++t) {
                const float sum = warp_reduce_sum(sums[r][t]);
                if (lane == 0 && row0 + r < outputRows)
                    FastllmGgufStore<StoreMode>(output + t * outputStride + row0 + r, sum);
            }
        }
    }
}

template <ggml_type Type, int K, typename Output, int StoreMode>
static void LaunchExtendedT8(const void *weights, const block_q8_1 *input, Output *output, int rows,
                             int inputStride, int outputStride, cudaStream_t stream) {
    static thread_local int cachedDevice = -1, cachedLimit = 0;
    int device = -1;
    const int groups = (rows + 4 * 2 - 1) / (4 * 2);
    int blocks = groups;
    if (cudaGetDevice(&device) == cudaSuccess) {
        if (device != cachedDevice) {
            int resident = 0, sms = 0;
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &resident, ExtendedT8GemvKernel<Type, K, Output, StoreMode>, 128, 0) == cudaSuccess &&
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
                resident > 0 && sms > 0) {
                cachedLimit = resident * sms;
                cachedDevice = device;
            }
        }
        if (device == cachedDevice)
            blocks = std::min(groups, cachedLimit);
    }
    ExtendedT8GemvKernel<Type, K, Output, StoreMode>
        <<<blocks, 128, 0, stream>>>(weights, input, output, rows, inputStride / QK8_1, outputStride);
}

template <ggml_type Type, int K, typename Output, int StoreMode>
static bool LaunchSelected(const void *w, const block_q8_1 *x, void *y, int n, int inputStride,
                           int outputStride, cudaStream_t stream) {
    // Other IQ3_S widths retain the shared-memory kernel selected by profiling.
    if constexpr (Type == GGML_TYPE_IQ3_S && K != 5120) {
        return false;
    } else {
        LaunchExtendedT8<Type, K, Output, StoreMode>(w, x, static_cast<Output *>(y), n, inputStride,
                                                     outputStride, stream);
        return true;
    }
}

template <ggml_type Type, typename Output, int StoreMode>
static bool DispatchColumns(const void *w, const block_q8_1 *x, void *y, int k, int n, int inputStride,
                            int outputStride, cudaStream_t stream) {
#define FASTLLM_EXTENDED_K(K)                                                                                \
    case K:                                                                                                  \
        return LaunchSelected<Type, K, Output, StoreMode>(w, x, y, n, inputStride, outputStride, stream)
    switch (k) {
        FASTLLM_EXTENDED_K(5120);
        FASTLLM_EXTENDED_K(6144);
        FASTLLM_EXTENDED_K(17408);
    }
#undef FASTLLM_EXTENDED_K
    return false;
}

template <ggml_type Type>
static bool DispatchOutput(int outputKind, int storeMode, const void *w, const block_q8_1 *x, void *y, int k,
                           int n, int inputStride, int outputStride, cudaStream_t stream) {
    if (outputKind == 1) {
#define FASTLLM_EXTENDED_STORE(S)                                                                            \
    case S:                                                                                                  \
        return DispatchColumns<Type, half, S>(w, x, y, k, n, inputStride, outputStride, stream)
        switch (storeMode) {
            FASTLLM_EXTENDED_STORE(0);
            FASTLLM_EXTENDED_STORE(1);
            FASTLLM_EXTENDED_STORE(2);
        }
#undef FASTLLM_EXTENDED_STORE
    } else if (storeMode == 0) {
        if (outputKind == 0)
            return DispatchColumns<Type, float, 0>(w, x, y, k, n, inputStride, outputStride, stream);
        if (outputKind == 2)
            return DispatchColumns<Type, __nv_bfloat16, 0>(w, x, y, k, n, inputStride, outputStride, stream);
    }
    return false;
}

bool DispatchExtendedT8(ggml_type type, int outputKind, int storeMode, const void *weights,
                        const block_q8_1 *input, void *output, int columns, int rows, int inputStride,
                        int outputStride, cudaStream_t stream) {
    if (rows < 4096)
        return false;
#define FASTLLM_EXTENDED_TYPE(T)                                                                             \
    case T:                                                                                                  \
        return DispatchOutput<T>(outputKind, storeMode, weights, input, output, columns, rows, inputStride,  \
                                 outputStride, stream)
    switch (type) {
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_IQ2_S);
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_IQ2_XS);
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_IQ2_XXS);
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_IQ3_S);
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_Q2_K);
    default:
        return false;
    }
#undef FASTLLM_EXTENDED_TYPE
}
} // namespace fastllm_gguf_small_mmvq

// The MMQ translation unit includes small-mmvq.cuh inside its own namespace.
// Forward that entry point to the same instantiations instead of duplicating
// all extended kernels in the legacy MMQ compilation unit.
namespace fastllm_gguf_mmq::fastllm_gguf_small_mmvq {
bool DispatchExtendedT8(ggml_type type, int outputKind, int storeMode, const void *weights,
                        const block_q8_1 *input, void *output, int columns, int rows, int inputStride,
                        int outputStride, cudaStream_t stream) {
    return ::fastllm_gguf_small_mmvq::DispatchExtendedT8(type, outputKind, storeMode, weights, input, output,
                                                         columns, rows, inputStride, outputStride, stream);
}
} // namespace fastllm_gguf_mmq::fastllm_gguf_small_mmvq
