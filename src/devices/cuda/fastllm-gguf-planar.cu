#include "fastllm-cuda-gguf-planar.h"
#include "fastllm-gguf-planar.cuh"

namespace fastllm_gguf_planar {
template <typename Input, bool Permuted>
__global__ void Quantize(const Input *__restrict__ input, int8_t *__restrict__ qs, half2 *__restrict__ ds,
                         short2 *__restrict__ sums, int k, int keyHeads, int groups, int headDim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= k)
        return;
    int source = i;
    if constexpr (Permuted) {
        const int head = i / headDim;
        source = ((head % keyHeads) * groups + head / keyHeads) * headDim + i % headDim;
    }
    const float x = static_cast<float>(input[size_t(t) * k + source]);
    float a = fabsf(x);
#pragma unroll
    for (int d = 16; d; d >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, d));
    const float scale = a / 127, originalSum = warp_reduce_sum(x);
    const int8_t q = a == 0 ? 0 : static_cast<int8_t>(roundf(x / scale));
    qs[size_t(t) * k + i] = q;
    int sum = q;
#pragma unroll
    for (int d = 8; d; d >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, d);
    const int hi = __shfl_sync(0xffffffff, sum, 16);
    if ((i & 31) == 0) {
        const size_t block = size_t(t) * (k / 32) + i / 32;
        ds[block] = __floats2half2_rn(scale, originalSum);
        sums[block] = make_short2(sum, hi);
    }
}

template <typename Input>
void QuantizeInput(const void *input, void *workspace, int tokens, int k, cudaStream_t stream, int keyHeads,
                   int groups, int headDim) {
    auto *qs = static_cast<int8_t *>(workspace);
    auto *ds = reinterpret_cast<half2 *>(qs + size_t(tokens) * k);
    auto *sums = reinterpret_cast<short2 *>(ds + size_t(tokens) * (k / 32));
    const dim3 grid((k + 255) / 256, tokens);
    if (keyHeads) {
        Quantize<Input, true><<<grid, 256, 0, stream>>>(static_cast<const Input *>(input), qs, ds, sums, k,
                                                        keyHeads, groups, headDim);
    } else {
        Quantize<Input, false>
            <<<grid, 256, 0, stream>>>(static_cast<const Input *>(input), qs, ds, sums, k, 0, 0, 0);
    }
}

template <ggml_type Type, int T, int Mode, int KS, bool SplitGateUp = false, bool StreamingWeights = false>
void LaunchK(const void *w, const void *up, const int8_t *qs, const half2 *ds, const short2 *sums, half *y,
             int k, int n, int stride, cudaStream_t stream) {
    static thread_local int cachedDevice = -1, limit = 0;
    int device = -1;
    constexpr int blockThreads = 32 * BlockWarps<Type, T, Mode>;
    constexpr int outputsPerBlock = BlockWarps<Type, T, Mode> * (Mode == 3 ? 1 : OutputRows<Type, T, Mode>);
    int blocks = (n + outputsPerBlock - 1) / outputsPerBlock;
    if (cudaGetDevice(&device) == cudaSuccess) {
        if (device != cachedDevice) {
            int active = 0, sms = 0;
            cudaError_t occupancy;
            if constexpr (SplitGateUp)
                occupancy = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, ProjectGateUpSingle<Type>,
                                                                          blockThreads, 0);
            else
                occupancy = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, Project<Type, T, KS, Mode, StreamingWeights>,
                                                                          blockThreads, 0);
            if (occupancy == cudaSuccess &&
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
                active > 0 && sms > 0) {
                if constexpr (Mode != 3 &&
                              ((T == 1 && (Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_IQ4_XS)) ||
                               ((T == 3 || T == 4) && Type == GGML_TYPE_IQ2_XS) ||
                               ((T == 1 || T == 3 || T == 6) && Type == GGML_TYPE_Q4_K))) {
                    // Small, memory-bound tiles benefit from reusing each
                    // block across more output rows instead of reloading its
                    // codebook in many short-lived blocks. Scale the cap with
                    // the device's warp capacity, not a fixed SM count.
                    int threads = 0;
                    if (cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device) ==
                            cudaSuccess &&
                        threads > 0)
                        active = std::min(active, std::max(1, threads / (4 * blockThreads)));
                }
                cachedDevice = device;
                limit = active * sms;
            }
        }
        if (device == cachedDevice)
            blocks = std::min(blocks, limit);
    }
    if constexpr (SplitGateUp) {
        static_assert(T == 1 && Mode == 3 && blockThreads == 128);
        ProjectGateUpSingle<Type><<<blocks, blockThreads, 0, stream>>>(w, up, qs, ds, y, k, n);
    } else {
        Project<Type, T, KS, Mode, StreamingWeights><<<blocks, blockThreads, 0, stream>>>(w, up, qs, ds, sums, y, n, stride, k);
    }
}

// Keep cache-resident matrices on the ordinary load path. For larger
// gate/up pairs, streaming loads give reused activations cache priority.
// Query the current device rather than assuming one architecture's L2 size.
static bool StreamIQ4Weights(int k, int n) {
    static thread_local int cachedDevice = -1, l2Bytes = 0;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
        return false;
    if (device != cachedDevice) {
        int bytes = 0;
        if (cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, device) != cudaSuccess || bytes <= 0)
            return false;
        cachedDevice = device;
        l2Bytes = bytes;
    }
    const size_t weightBytes = size_t(n) * (k / 256) * sizeof(block_iq4_xs) * 2;
    return weightBytes > size_t(l2Bytes);
}

template <ggml_type Type, int T, int Mode>
void Launch(const void *w, const void *up, const int8_t *qs, const half2 *ds, const short2 *sums, half *y,
            int k, int n, int stride, cudaStream_t stream) {
    if constexpr (Mode == 3 && T == 1 && (Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_IQ3_XXS)) {
        // Expansion projections are bandwidth-bound at one input row. Long K
        // with fewer output rows keeps the full-warp loop to limit instruction
        // overhead. Both paths use device-derived occupancy limits.
        if (n >= k) {
            LaunchK<Type, T, Mode, 0, true>(w, up, qs, ds, sums, y, k, n, stride, stream);
            return;
        }
    }
    if constexpr (Type == GGML_TYPE_IQ4_XS && Mode == 3) {
        if (StreamIQ4Weights(k, n)) {
            LaunchK<Type, T, Mode, 0, false, true>(w, up, qs, ds, sums, y, k, n, stride, stream);
            return;
        }
    }
    // These common model widths remove stride arithmetic without unrolling
    // the dot loop. The runtime-K kernel handles all other block-aligned widths.
    if constexpr ((Type == GGML_TYPE_Q4_K && T != 1 && T != 3 && T != 6) || Type == GGML_TYPE_Q2_K ||
                  (Type == GGML_TYPE_IQ2_S && T <= 3 && (T != 1 || Mode > 1)) ||
                  (Type == GGML_TYPE_IQ2_XS && T == 1)) {
#define WIDTH(K)                                                                                             \
    case K:                                                                                                  \
        LaunchK<Type, T, Mode, K>(w, up, qs, ds, sums, y, k, n, stride, stream);                             \
        return
        switch (k) {
            WIDTH(4096);
            WIDTH(5120);
            WIDTH(6144);
            WIDTH(17408);
        }
#undef WIDTH
    }
    LaunchK<Type, T, Mode, 0>(w, up, qs, ds, sums, y, k, n, stride, stream);
}

template <ggml_type Type, int Mode>
void DispatchRows(const void *w, const void *up, const void *workspace, half *y, int tokens, int k, int n,
                  int stride, cudaStream_t stream) {
    const auto *qs = static_cast<const int8_t *>(workspace);
    const auto *ds = reinterpret_cast<const half2 *>(qs + size_t(tokens) * k);
    const auto *sums = reinterpret_cast<const short2 *>(ds + size_t(tokens) * (k / 32));
    // Larger verification batches reuse one quantization and process at most
    // eight rows per launch; no padding and no extra activation conversion.
    for (int first = 0; first < tokens; first += 8) {
        const int tile = std::min(tokens - first, 8);
#define ROWS(T)                                                                                              \
    case T:                                                                                                  \
        Launch<Type, T, Mode>(w, up, qs + size_t(first) * k, ds + size_t(first) * (k / 32),                  \
                              sums + size_t(first) * (k / 32), y + size_t(first) * stride, k, n, stride,     \
                              stream);                                                                       \
        break
        switch (tile) {
            ROWS(1);
            ROWS(2);
            ROWS(3);
            ROWS(4);
            ROWS(5);
            ROWS(6);
            ROWS(7);
            ROWS(8);
        }
#undef ROWS
    }
}

template <ggml_type Type>
void DispatchMode(int mode, const void *w, const void *up, const void *workspace, half *y, int tokens, int k,
                  int n, int stride, cudaStream_t stream) {
#define MODE(M)                                                                                              \
    case M:                                                                                                  \
        DispatchRows<Type, M>(w, up, workspace, y, tokens, k, n, stride, stream);                            \
        break
    switch (mode) {
        MODE(0);
        MODE(1);
        MODE(2);
        MODE(3);
    }
#undef MODE
}
} // namespace fastllm_gguf_planar

size_t FastllmGgufPlanarBytes(int tokens, int columns) {
    if (tokens < 1 || tokens > 16 || columns < 256 || columns % 256)
        return 0;
    return size_t(tokens) * columns + size_t(tokens) * (columns / 32) * (sizeof(half2) + sizeof(short2));
}

bool FastllmGgufPlanarSupported(int type, int tokens, int columns, int outputRows) {
    if (!FastllmGgufPlanarBytes(tokens, columns) || outputRows < 1)
        return false;
    switch (static_cast<ggml_type>(type)) {
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ1_M:
        return true;
    default:
        return false;
    }
}

bool FastllmGgufQuantizePlanar(const void *input, int inputKind, void *workspace, int tokens, int columns,
                               void *stream, int keyHeads, int groups, int headDim) {
    if (!input || !workspace || !FastllmGgufPlanarBytes(tokens, columns) ||
        (reinterpret_cast<uintptr_t>(workspace) & 15) || keyHeads < 0 ||
        (keyHeads &&
         (groups < 1 || headDim < 32 || headDim % 32 || int64_t(keyHeads) * groups * headDim != columns)))
        return false;
#define INPUT(KIND, TYPE)                                                                                    \
    case KIND:                                                                                               \
        fastllm_gguf_planar::QuantizeInput<TYPE>(input, workspace, tokens, columns,                          \
                                                 static_cast<cudaStream_t>(stream), keyHeads, groups,        \
                                                 headDim);                                                   \
        return true
    switch (inputKind) {
        INPUT(0, float);
        INPUT(1, half);
        INPUT(2, __nv_bfloat16);
    }
#undef INPUT
    return false;
}

bool FastllmGgufProjectPlanar(int type, int mode, const void *weight, const void *upWeight,
                              const void *workspace, void *output, int tokens, int columns, int outputRows,
                              int outputStride, void *stream) {
    if (!FastllmGgufPlanarSupported(type, tokens, columns, outputRows) || !weight || !workspace || !output ||
        (reinterpret_cast<uintptr_t>(workspace) & 15) || outputStride < outputRows || mode < 0 || mode > 3 ||
        (mode == 3 && !upWeight))
        return false;
#define TYPE(T)                                                                                              \
    case T:                                                                                                  \
        fastllm_gguf_planar::DispatchMode<T>(mode, weight, upWeight, workspace, static_cast<half *>(output), \
                                             tokens, columns, outputRows, outputStride,                      \
                                             static_cast<cudaStream_t>(stream));                             \
        return true
    switch (static_cast<ggml_type>(type)) {
        TYPE(GGML_TYPE_IQ3_S);
        TYPE(GGML_TYPE_IQ3_XXS);
        TYPE(GGML_TYPE_IQ4_XS);
        TYPE(GGML_TYPE_Q4_K);
        TYPE(GGML_TYPE_Q2_K);
        TYPE(GGML_TYPE_IQ2_S);
        TYPE(GGML_TYPE_IQ2_XS);
        TYPE(GGML_TYPE_IQ2_XXS);
        TYPE(GGML_TYPE_IQ1_M);
    default:
        return false;
    }
#undef TYPE
}
