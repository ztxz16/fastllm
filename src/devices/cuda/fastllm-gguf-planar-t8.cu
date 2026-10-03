#include "fastllm-cuda-gguf-planar-t8.h"
#include "fastllm-gguf-kernel-common.cuh"
#include "fastllm-gguf-small-mmvq.cuh"

namespace fastllm_gguf_planar_t8 {
using namespace fastllm_gguf_small_mmvq;
constexpr int K = 5120, T = 8, Warps = 4;

// Same quantization and reduction as quantize_q8_1. Only the destination
// addressing changes: no AoS temporary and no extra conversion launch.
__global__ void Quantize(const half *__restrict__ input, int8_t *__restrict__ qs, half2 *__restrict__ ds) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= K)
        return;
    const float x = __half2float(input[t * K + i]);
    float a = fabsf(x);
#pragma unroll
    for (int d = 16; d; d >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, d));
    const float sum = warp_reduce_sum(x), scale = a / 127;
    qs[t * K + i] = a == 0.0f ? 0 : static_cast<int8_t>(roundf(x / scale));
    if ((i & 31) == 0)
        ds[t * (K / 32) + i / 32] = __floats2half2_rn(scale, sum);
}

template <ggml_type Type, int Mode>
__global__ __launch_bounds__(128, 4) void Project(const void *__restrict__ weights,
                                                  const void *__restrict__ upWeights,
                                                  const int8_t *__restrict__ qs, const half2 *__restrict__ ds,
                                                  half *__restrict__ output, int n, int outputStride) {
    __shared__ __align__(16) uint32_t table[CodebookWords<Type>];
    if constexpr (Type != GGML_TYPE_Q4_K && Type != GGML_TYPE_Q2_K && Type != GGML_TYPE_IQ4_XS) {
        LoadCodebook<Type, Warps>(table);
        __syncthreads();
    }
    constexpr int step = Mode == 3 ? 1 : 2;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int row = (blockIdx.x * Warps + warp) * step; row < n; row += gridDim.x * Warps * step) {
        float sum[2][T] = {};
#pragma unroll
        for (int it = 0; it < K / 1024; ++it) {
            const int slice = it * 32 + lane;
            int a[T][8], loSum[T], hiSum[T];
            float dx[T];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int4 *p = reinterpret_cast<const int4 *>(qs + t * K + slice * 32);
                const int4 lo = p[0], hi = p[1];
                a[t][0] = lo.x;
                a[t][1] = lo.y;
                a[t][2] = lo.z;
                a[t][3] = lo.w;
                a[t][4] = hi.x;
                a[t][5] = hi.y;
                a[t][6] = hi.z;
                a[t][7] = hi.w;
                dx[t] = __low2float(ds[t * (K / 32) + slice]);
                if constexpr (Type == GGML_TYPE_Q4_K || Type == GGML_TYPE_Q2_K) {
                    int ls = 0, hs = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        ls = __dp4a(0x01010101, a[t][j], ls);
                        hs = __dp4a(0x01010101, a[t][j + 4], hs);
                    }
                    loSum[t] = ls;
                    hiSum[t] = hs;
                }
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const void *w = Mode == 3 && r == 1 ? upWeights : weights;
                const int outputRow = Mode == 3 ? row : min(row + r, n - 1);
                const int block = outputRow * (K / 256) + slice / 8, group = lane % 8;
                int v[8], scale = 0, minimum = 0, sc0 = 0, sc1 = 0;
                float d = 0, dmin = 0;
                if constexpr (Type == GGML_TYPE_Q4_K) {
                    const auto *b = static_cast<const block_q4_K *>(w) + block;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        v[j] = (get_int_b4(b->qs, (group / 2) * 8 + j) >> (4 * (group % 2))) & 0x0f0f0f0f;
                    scale = group < 4 ? (b->scales[group] & 63)
                                      : ((b->scales[group + 4] & 15) | ((b->scales[group - 4] >> 6) << 4));
                    minimum = group < 4 ? (b->scales[group + 4] & 63)
                                        : ((b->scales[group + 4] >> 4) | ((b->scales[group] >> 6) << 4));
                    const float2 dm = __half22float2(b->dm);
                    d = dm.x;
                    dmin = dm.y;
                } else if constexpr (Type == GGML_TYPE_Q2_K) {
                    const auto *b = static_cast<const block_q2_K *>(w) + block;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        v[j] = (get_int_b4(b->qs, (group / 4) * 8 + j) >> (2 * (group % 4))) & 0x03030303;
                    sc0 = b->scales[2 * group];
                    sc1 = b->scales[2 * group + 1];
                    const float2 dm = __half22float2(b->dm);
                    d = dm.x;
                    dmin = dm.y;
                } else
                    DecodeBatchWeights<Type, true>(w, block, 2 * group, table, v, d, scale);
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    if constexpr (Type == GGML_TYPE_IQ2_S || Type == GGML_TYPE_IQ2_XS ||
                                  Type == GGML_TYPE_IQ1_M || Type == GGML_TYPE_Q2_K) {
                        int lo = 0, hi = 0;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            lo = __dp4a(v[j], a[t][j], lo);
                            hi = __dp4a(v[j + 4], a[t][j + 4], hi);
                        }
                        if constexpr (Type == GGML_TYPE_Q2_K)
                            sum[r][t] += d * (dx[t] * (lo * (sc0 & 15) + hi * (sc1 & 15))) -
                                         dmin * (dx[t] * (loSum[t] * (sc0 >> 4) + hiSum[t] * (sc1 >> 4)));
                        else if constexpr (Type == GGML_TYPE_IQ1_M) {
                            const int dot = lo * (scale & 15) + hi * (scale >> 4);
                            sum[r][t] += (d * dx[t]) * (dot * .125f);
                        } else {
                            const int dot = (lo * (scale & 15) + hi * (scale >> 4) + (lo + hi) / 2) / 4;
                            sum[r][t] += (d * dx[t]) * dot;
                        }
                    } else {
                        int dot = 0;
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            dot = __dp4a(v[j], a[t][j], dot);
                        if constexpr (Type == GGML_TYPE_Q4_K)
                            sum[r][t] += d * (dx[t] * (dot * scale)) -
                                         dmin * (dx[t] * ((loSum[t] + hiSum[t]) * minimum));
                        else {
                            if constexpr (Type == GGML_TYPE_IQ3_XXS)
                                dot = (scale * dot + dot / 2) / 2;
                            else if constexpr (Type == GGML_TYPE_IQ2_XXS)
                                dot = (scale * dot + dot / 2) / 4;
                            else
                                dot *= scale;
                            sum[r][t] += (d * dx[t]) * dot;
                        }
                    }
                }
            }
        }
        // Each lane receives one final result. The half-precision SiLU/divide
        // runs once with 8 or 16 active lanes, rather than serially per token
        // on lane zero. Every reduction and intermediate rounding is unchanged.
        float value = 0, gateValue = 0;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float x = warp_reduce_sum(sum[0][t]), y = warp_reduce_sum(sum[1][t]);
            if constexpr (Mode == 3) {
                if (lane == t) {
                    gateValue = x;
                    value = y;
                }
            } else {
                if (lane == t)
                    value = x;
                if (lane == t + T)
                    value = y;
            }
        }
        if constexpr (Mode == 3) {
            if (lane < T) {
                const half gate = __float2half_rn(gateValue), up = __float2half_rn(value);
                const half act = __hdiv(gate, __hadd(__float2half(1.0f), hexp(-gate)));
                output[lane * outputStride + row] = __hmul(act, up);
            }
        } else if (lane < 2 * T && row + lane / T < n)
            FastllmGgufStore<Mode>(output + (lane % T) * outputStride + row + lane / T, value);
    }
}

template <ggml_type Type, int Mode>
static void Launch(const void *w, const void *up, const void *workspace, half *y, int n, int stride,
                   cudaStream_t stream) {
    static thread_local int cachedDevice = -1, cachedLimit = 0;
    int dev = -1, blocks = (n + (Mode == 3 ? 4 : 8) - 1) / (Mode == 3 ? 4 : 8);
    if (cudaGetDevice(&dev) == cudaSuccess) {
        if (dev != cachedDevice) {
            int resident = 0, sms = 0;
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, Project<Type, Mode>, 128, 0) ==
                    cudaSuccess &&
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
                resident > 0 && sms > 0) {
                cachedDevice = dev;
                cachedLimit = resident * sms;
            }
        }
        if (dev == cachedDevice)
            blocks = std::min(blocks, cachedLimit);
    }
    const auto *qs = static_cast<const int8_t *>(workspace);
    const auto *ds = reinterpret_cast<const half2 *>(qs + T * K);
    Project<Type, Mode><<<blocks, 128, 0, stream>>>(w, up, qs, ds, y, n, stride);
}
template <ggml_type Type>
static bool Dispatch(int mode, const void *w, const void *up, const void *q, void *y, int n, int stride,
                     cudaStream_t stream) {
    if (mode == 0)
        Launch<Type, 0>(w, up, q, static_cast<half *>(y), n, stride, stream);
    else if (mode == 2)
        Launch<Type, 2>(w, up, q, static_cast<half *>(y), n, stride, stream);
    else if (mode == 3 && up)
        Launch<Type, 3>(w, up, q, static_cast<half *>(y), n, stride, stream);
    else
        return false;
    return true;
}
} // namespace fastllm_gguf_planar_t8

bool FastllmGgufPlanarT8Supported(int type) {
    switch (type) {
    case GGML_TYPE_IQ1_M:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q2_K:
        return true;
    default:
        return false;
    }
}
void FastllmGgufQuantizePlanarT8(const void *input, void *workspace, void *stream) {
    auto *qs = static_cast<int8_t *>(workspace);
    auto *ds = reinterpret_cast<half2 *>(qs + 8 * 5120);
    fastllm_gguf_planar_t8::Quantize<<<dim3(5120 / 256, 8), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const half *>(input), qs, ds);
}
bool FastllmGgufProjectPlanarT8(int type, int mode, const void *w, const void *up, const void *q, void *y,
                                int n, int stride, void *stream) {
    if (!w || !q || !y || n < 1 || stride < n)
        return false;
#define PLANAR_CASE(TY)                                                                                      \
    case TY:                                                                                                 \
        return fastllm_gguf_planar_t8::Dispatch<TY>(mode, w, up, q, y, n, stride,                            \
                                                    static_cast<cudaStream_t>(stream))
    switch (type) {
        PLANAR_CASE(GGML_TYPE_IQ1_M);
        PLANAR_CASE(GGML_TYPE_IQ2_XXS);
        PLANAR_CASE(GGML_TYPE_IQ2_XS);
        PLANAR_CASE(GGML_TYPE_IQ2_S);
        PLANAR_CASE(GGML_TYPE_IQ3_XXS);
        PLANAR_CASE(GGML_TYPE_IQ3_S);
        PLANAR_CASE(GGML_TYPE_IQ4_XS);
        PLANAR_CASE(GGML_TYPE_Q4_K);
        PLANAR_CASE(GGML_TYPE_Q2_K);
    default:
        return false;
    }
#undef PLANAR_CASE
}
