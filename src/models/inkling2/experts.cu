// Expert packing and activation quantization follow the original Inkling 2 Mini checkpoint.
// Reference for the NVFP4 reconstruction policy and the position of the routing weights:
// https://github.com/vllm-project/vllm/tree/fc1443152c0eb655f5c2762f56b540bebe93b196/vllm/models/inkling2/nvidia
#include "runtime.cuh"
#ifdef FASTLLM_INKLING_GROUPED_SM75
#include "mxfp8-sm75.cuh"
#endif
#include "../../devices/cuda/moe/fastllm-nvfp4-routed-sm120.cuh"
#ifdef FASTLLM_INKLING_GROUPED_SM120
#include "../../devices/cuda/moe/fastllm-grouped-sm120.cuh"
#endif
#include "devices/cpu/computeutils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-activation.h"
#include "devices/cuda/fastllm-hybrid-moe-admission.cuh"
#include "devices/cuda/fastllm-cuda-activation.cuh"
#include "devices/cuda/fastllm-cuda-select-expert.cuh"
#include "devices/cpu/alivethreadpool.h"
#ifdef USE_NUMAS
#include "devices/numas/numas.h"
#include "devices/numas/numasdevice.h"
#endif
#include <cuda_bf16.h>
#if CUDART_VERSION >= 12000
#include <cuda/atomic>
#endif
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif
#include <algorithm>
#include <chrono>
#include <numeric>
#include "utils/hybrid_moe_routes.h"
#include "moe_cache_config.h"
#include "utils/utils.h"
#include <cmath>
#include <fstream>
#include <limits>
#include <type_traits>
namespace fastllm {
namespace inkling2 {
namespace {
namespace native = fastllm::cuda::routed_fp4;
__host__ __device__ float fp4Value(int x) {
    const float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return x & 8 ? -values[x & 7] : values[x & 7];
}
__host__ __device__ float fp8Value(uint8_t x) {
    int bits = x & 127, e = bits >> 3, m = bits & 7;
    float val = e ? ldexpf(1 + m * .125f, e - 7) : ldexpf(float(m), -9);
    return x & 128 ? -val : val;
}
__device__ float roundFp4(float x) {
    float a = fabsf(x);
    int i = (a > .25f) + (a >= .75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return copysignf(fp4Value(i), x);
}
__device__ float roundFp8(float x) {
    float a = fminf(fabsf(x), 448.0f);
    if (a == 0)
        return 0;
#if CUDART_VERSION >= 11080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return copysignf(float(__nv_fp8_e4m3(a)), x);
#else
    float step = ldexpf(1.0f, max(-9, ilogbf(a) - 3));
    return copysignf(rintf(a / step) * step, x);
#endif
}
__device__ float reconstructMxfp8(float value, float blockMax) {
    if (!(blockMax > 0)) return 0.f;
    const float scale = exp2f(ceilf(log2f(blockMax / 448.f)));
    return roundFp8(value / scale) * scale;
}

struct Fp4Choice {
    float value, scale, reconstructed;
};
__device__ Fp4Choice chooseFp4(float v, float m, float outer) {
    constexpr int width = 16;
    if (m > 0 && outer > 0) {
        float b6 = roundFp8(fminf(448.0f, (m / outer) / 6));
        float s6 = b6 * outer;
        float b4 = roundFp8(fminf(448.0f, (m / outer) / 4));
        float s4 = b4 * outer;
        float q6 = s6 ? roundFp4(v / s6) : 0, q4 = s4 ? roundFp4(v / s4) : 0;
        // Score the rounded reconstructions identically for packed and unpacked outputs.
        float v6 = __fmul_rn(q6, s6), v4 = __fmul_rn(q4, s4);
        float e4 = fabsf(v - v4), e6 = fabsf(v - v6);
        for (int delta = width / 2; delta; delta /= 2) {
            e4 += __shfl_xor_sync(__activemask(), e4, delta, width);
            e6 += __shfl_xor_sync(__activemask(), e6, delta, width);
        }
        return e4 < e6 ? Fp4Choice{q4, b4, v4} : Fp4Choice{q6, b6, v6};
    }
    return {};
}
struct InputRow {
    const float *x;
    __device__ float get(size_t row, int col, int d) const { return x[row * d + col]; }
};

template <typename T, bool Interleaved = false> struct RoutedSiTU {
    const T *pair;
    const float *scores;
    float gateCap, upCap;
    __device__ float get(size_t row, int col, int d) const { return get(row, row, col, d); }
    __device__ float get(size_t row, size_t route, int col, int d) const {
        const T *gate = pair + row * (2 * d) + (Interleaved ? 2 * col : col);
        float g = float(__float2bfloat16_rn(float(gate[0])));
        float u = float(__float2bfloat16_rn(float(gate[Interleaved ? 1 : d])));
        return float(__float2bfloat16_rn(FastllmSiTUAndMul(g, u, gateCap, upCap) * scores[route]));
    }
};
// Marlin consumes reconstructed BF16 rows. Eight lanes quantize one block
// of 32 channels, sharing its scale across four adjacent values per lane.
__global__ void reconstructMxfp8SiTU(RoutedSiTU<__nv_bfloat16> src, __nv_bfloat16 *out, int width) {
    struct alignas(8) Four { __nv_bfloat16 v[4]; };
    cuda_launch::DependencyStart();
    const size_t vectors = width / 4, row = blockIdx.x;
    const auto *gate = reinterpret_cast<const Four *>(src.pair) + row * 2 * vectors;
    auto *dest = reinterpret_cast<Four *>(out) + row * vectors;
    const float score = src.scores[row];
    for (int i = threadIdx.x; i < vectors; i += blockDim.x) {
        Four result{};
        if (score != 0.f) {
            const Four g = gate[i], u = gate[vectors + i];
            float v[4], peak = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = float(__float2bfloat16_rn(FastllmSiTUAndMul(float(g.v[j]), float(u.v[j]),
                                                                  src.gateCap, src.upCap) * score));
                peak = fmaxf(peak, fabsf(v[j]));
            }
            const unsigned mask = __activemask();
#pragma unroll
            for (int step = 4; step; step /= 2) peak = fmaxf(peak, __shfl_xor_sync(mask, peak, step, 8));
#pragma unroll
            for (int j = 0; j < 4; ++j) result.v[j] = __float2bfloat16_rn(reconstructMxfp8(v[j], peak));
        }
        dest[i] = result;
    }
}
struct LocalSiTU : RoutedSiTU<__nv_bfloat16> {
    const int *ids;
};

#ifdef FASTLLM_INKLING_GROUPED_SM120
namespace mx = fastllm::cuda::routed_mxfp8;
template<bool Local, class Source>
__global__ void packMxfp8Routes(Source source, uint8_t *codes, uint8_t *scales,
                                const int *ids, const int *offsets, const int *ranks, int k) {
    constexpr bool input = std::is_same_v<Source, InputRow>;
    constexpr int copies = input ? 8 : 1;
    __shared__ int sorted[copies], scaleRow[copies];
    bool valid = false;
    if (threadIdx.x < copies) {
        const int r = input ? blockIdx.x * copies + threadIdx.x : blockIdx.x;
        const int e = ids[r];
        valid = !Local || e >= 0;
        if (valid) {
            const int start = offsets[e];
            sorted[threadIdx.x] = start + ranks[r];
            scaleRow[threadIdx.x] = cuda::grouped_sm120::scaleRow(start, e, ranks[r]);
        } else sorted[threadIdx.x] = -1;
    }
    if constexpr (Local) {
        if (!__syncthreads_or(valid)) return;
    } else __syncthreads();
    // Each input token is quantized once and scattered to its eight experts.
    // The down projection consumes the already sorted gate/up output.
    for (int col = 4 * threadIdx.x; col < k; col += 4 * blockDim.x) {
        float4 v;
        if constexpr (input) {
            v = *reinterpret_cast<const float4 *>(source.x + size_t(blockIdx.x) * k + col);
        } else {
            v = make_float4(source.get(sorted[0], blockIdx.x, col, k), source.get(sorted[0], blockIdx.x, col + 1, k),
                            source.get(sorted[0], blockIdx.x, col + 2, k), source.get(sorted[0], blockIdx.x, col + 3, k));
        }
        float peak = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        const unsigned mask = __activemask();
        for (int delta = 4; delta; delta /= 2) peak = fmaxf(peak, __shfl_xor_sync(mask, peak, delta, 8));
        // Inkling rounds peak/448 up to a power of two for each block of 32.
        const int exponent = peak > 0 ? max(-127, min(127, int(ceilf(log2f(peak / 448.f))))) : 0;
        const float inv = ldexpf(1.f, -exponent);
        const auto bytes = __nv_fp8x4_e4m3(make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv)).__x;
        #pragma unroll
        for (int j = 0; j < copies; ++j) {
            if constexpr (Local) if (sorted[j] < 0) continue;
            reinterpret_cast<uint32_t *>(codes)[(size_t(sorted[j]) * k + col) / 4] = bytes;
            if (!(col % 32)) scales[cuda::grouped_sm120::scaleOffset(scaleRow[j], col / 32, k / 32)] = exponent + 127;
        }
    }
}

#endif

#ifdef FASTLLM_INKLING_GROUPED_SM75
template<class Source>
__global__ void packSm75Routes(Source src, half *out, const int *ids, const int *offsets,
                               const int *ranks, int width) {
    constexpr bool input = std::is_same_v<Source, InputRow>;
    constexpr int copies = input ? 8 : 1;
    const int row = blockIdx.x;
    __shared__ int sorted[copies];
    if (threadIdx.x < copies) {
        int route = row * copies + threadIdx.x;
        sorted[threadIdx.x] = offsets[ids[route]] + ranks[route];
    }
    __syncthreads();
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float v;
        if constexpr (input) v = src.get(row, col, width);
        else v = src.get(sorted[0], row, col, width);
        float peak = fabsf(v);
        for (int delta = 16; delta; delta >>= 1)
            peak = fmaxf(peak, __shfl_xor_sync(__activemask(), peak, delta));
        const half value = __float2half_rn(reconstructMxfp8(v, peak));
        #pragma unroll
        for (int j = 0; j < copies; ++j) out[size_t(sorted[j]) * width + col] = value;
    }
}
#endif

#if defined(FASTLLM_INKLING_GROUPED_SM75) || defined(FASTLLM_INKLING_GROUPED_SM120)
template<bool Local = false, class T>
__global__ void sumMxfp8Routes(const T *sorted, float *output, const int *ids,
                               const int *offsets, const int *ranks, int n, int top) {
    const int token = blockIdx.x;
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        float sum = 0;
        for (int j = 0; j < top; ++j) {
            const int r = token * top + j;
            if constexpr (Local) if (ids[r] < 0) continue;
            sum += float(sorted[size_t(offsets[ids[r]] + ranks[r]) * n + col]);
        }
        output[size_t(token) * n + col] = sum;
    }
}
#endif

// BF16 has only 65536 encodings. Cache the two exact FP32 factors once
// per model so every routed row only needs a multiply and its routing score.
__global__ void makeSiTUFactors(float *factors, float gateCap, float upCap) {
    const int code = blockIdx.x * blockDim.x + threadIdx.x;
    const float x = __uint_as_float(unsigned(code) << 16);
    factors[code] = (gateCap * tanhf(x / gateCap)) * (1.f / (1.f + expf(-x)));
    factors[65536 + code] = upCap > 0.f ? upCap * tanhf(x / upCap) : x;
}
struct PackedSiTU {
    const __nv_bfloat16 *pair;
    const float *scores, *factors;
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    __device__ void loadBlock(int row, int route, int col, int width, float (&v)[16]) const {
        const auto *p = pair + size_t(row) * 2 * width + col;
        const float score = scores[route];
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            unsigned short gate[8], up[8];
            uint4 g = reinterpret_cast<const uint4 *>(p)[part], u = reinterpret_cast<const uint4 *>(p + width)[part];
            memcpy(gate, &g, sizeof(g));
            memcpy(up, &u, sizeof(u));
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const float2 x = make_float2(factors[gate[i]], factors[gate[i + 1]]);
                const float2 y = make_float2(factors[65536 + up[i]], factors[65536 + up[i + 1]]);
                const float2 z = __fmul2_rn(__fmul2_rn(x, y), make_float2(score, score));
                v[part * 8 + i] = float(__float2bfloat16_rn(z.x));
                v[part * 8 + i + 1] = float(__float2bfloat16_rn(z.y));
            }
        }
    }
#endif
};

__device__ void storeFp4Word(native::Block16 *out, int row, int col, int width,
                              unsigned word, uint8_t scale, float outer) {
    auto &block = out[(size_t(row) * width + col) / 16];
    block.code[(col % 16) / 8] = word;
    if (!(col % 16)) {
        block.scale = scale;
        block.outer = outer;
    }
}

#if defined(FASTLLM_INKLING_GROUPED_SM120)
struct GroupedFp4Output {
    native::GroupedInputRow row;
    float *outer;
};
__device__ void storeFp4Word(GroupedFp4Output out, int route, int col, int, unsigned word, uint8_t scale, float outer) {
    reinterpret_cast<unsigned *>(out.row.codes)[col / 8] = word;
    if (!(col % 16)) out.row.scale(col / 16) = scale;
    if (!col) out.outer[route] = outer;
}
#endif

template <bool fp4, typename Source, typename Out>
__device__ void reconstructRow(Source src, Out out, int d, bool cacheValues, unsigned tiles) {
    const int row = blockIdx.x;
    constexpr int tile = fp4 ? 512 : 256;
    constexpr bool plain = std::is_same_v<Source, InputRow>;
    extern __shared__ float values[];
    __shared__ float warpMax[tile / 32];
    float outer = 0;
    if constexpr (fp4) {
        float amax = 0;
        if constexpr (plain) {
            // InputRow comes from an aligned CUDA row with a multiple of 16 elements.
            const auto *x = reinterpret_cast<const float4 *>(src.x + size_t(row) * d);
            for (int i = threadIdx.x; i < d / 4; i += tile) {
                float4 v = x[i];
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        } else {
            for (int col = threadIdx.x; col < d; col += tile) {
                float v = src.get(row, col, d);
                if (cacheValues) values[col] = v;
                amax = fmaxf(amax, fabsf(v));
            }
        }
        for (int delta = 16; delta; delta /= 2)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        if ((threadIdx.x & 31) == 0) warpMax[threadIdx.x / 32] = amax;
        __syncthreads();
        amax = warpMax[threadIdx.x % (tile / 32)];
        for (int delta = tile / 64; delta; delta /= 2)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        outer = amax / 1536.0f;
    }
    for (size_t col = blockIdx.y * tile + threadIdx.x; col < d; col += size_t(tiles) * tile) {
        float v = fp4 && !plain && cacheValues ? values[col] : src.get(row, col, d);
        constexpr int width = fp4 ? 16 : 32;
        float peak = fabsf(v);
        for (int delta = width / 2; delta; delta >>= 1)
            peak = fmaxf(peak, __shfl_xor_sync(__activemask(), peak, delta, width));
        if constexpr (std::is_same_v<Out, native::Block16 *> || !std::is_pointer_v<Out>) {
#if CUDART_VERSION >= 11080
            const Fp4Choice q = chooseFp4(v, peak, outer);
            // E2M1 magnitudes from one through six occupy consecutive half-exponent steps.
            const unsigned mag = __float_as_uint(fabsf(q.value)) >> 22;
            const unsigned nibble = (fabsf(q.value) < 1.f ? unsigned(q.value != 0.f) : mag - 252u) |
                                    ((__float_as_uint(q.value) >> 28) & 8u);
            unsigned word = nibble << (4 * (col % 8));
            for (int delta = 1; delta < 8; delta *= 2)
                word |= __shfl_xor_sync(__activemask(), word, delta, 8);
            if (!(col % 8)) storeFp4Word(out, row, col, d, word, __nv_fp8_e4m3(q.scale).__x, outer);
#endif
        } else {
            const float q = fp4 ? chooseFp4(v, peak, outer).reconstructed : reconstructMxfp8(v, peak);
            if constexpr (std::is_same_v<Out, half *>)
                out[size_t(row) * d + col] = __float2half_rz(q);
            else
                out[size_t(row) * d + col] = q;
        }
    }
}

template <bool fp4, bool Pdl = false, typename Source, typename Out>
__global__ void reconstructExpertRows(Source src, Out *out, int d, bool cacheValues) {
    if constexpr (Pdl) cuda_launch::DependencyStart();
    if constexpr (std::is_same_v<Source, LocalSiTU>)
        if (src.ids[blockIdx.x] < 0) return;
    reconstructRow<fp4>(src, out, d, cacheValues, gridDim.y);
}

#if defined(FASTLLM_INKLING_GROUPED_SM120)
// Keep the same two FP32 multiplications before the model's BF16 rounding.
__global__ void sortedGateScales(const native::Block16 *x, const float *ws, const int *ids, const int *offsets,
                                 const int *ranks, float2 *sorted, int routes, int top, int blocks) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < routes) sorted[offsets[ids[r]] + ranks[r]] = make_float2(ws[ids[r]], x[size_t(r / top) * blocks].outer);
}
__global__ void gatherGateBf16(const float *dots, const float2 *scales, __nv_bfloat16 *out,
                               int vectors, int pitch) {
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vectors) return;
    float4 a = reinterpret_cast<const float4 *>(dots)[size_t(row) * vectors + i];
    auto sf = scales[row];
    auto *dst = reinterpret_cast<__nv_bfloat162 *>(out + size_t(row) * pitch) + 2 * i;
    dst[0] = __floats2bfloat162_rn(a.x * sf.x * sf.y, a.y * sf.x * sf.y);
    dst[1] = __floats2bfloat162_rn(a.z * sf.x * sf.y, a.w * sf.x * sf.y);
}

struct GroupedSiTU {
    const float *dots;
    float weightScale, inputScale, score, gateCap, upCap;
    __device__ float get(size_t, int col, int width) const {
        const float gate = native::scaledGroupedValue(dots[col], weightScale, inputScale);
        const float up = native::scaledGroupedValue(dots[width + col], weightScale, inputScale);
        return float(__float2bfloat16_rn(FastllmSiTUAndMul(gate, up, gateCap, upCap) * score));
    }
};

__global__ void quantizeGroupedSiTU(native::GroupedBuffers buffers, const native::Block16 *input,
                                    const float *weightScales, const float *scores, const int *ids,
                                    const int *offsets, const int *ranks, float *midScales,
                                    int hidden, int width, int topK, float gateCap, float upCap, bool cache) {
#if __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    const int route = blockIdx.x, expert = ids[route];
    if (expert < 0) return;
    const int rank = ranks[route], start = offsets[expert];
    const float *dots = buffers.accumulators + size_t(start + rank) * (2 * width);
    const float inputScale = input[size_t(route / topK) * (hidden / 16)].outer;
    GroupedSiTU act{dots, weightScales[expert], inputScale, scores[route], gateCap, upCap};
    GroupedFp4Output dst{native::GroupedInputRow(buffers, start, expert, rank, width / 16), midScales};
    reconstructRow<true>(act, dst, width, cache, gridDim.y);
}
// Each lane owns all 16 values sharing an FP4 scale. Keep them in registers
// through the row maximum and write CUTLASS's expert-sorted input directly.
template<bool Narrow> __global__ void quantizeGroupedSiTUBlocks(native::GroupedBuffers buffers, const native::Block16 *input,
        const float *weightScales, const float *scores, const int *ids, const int *offsets, const int *ranks,
        float *midScales, int hidden, int width, int topK, const float *factors, const __nv_bfloat16 *paired) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    cudaTriggerProgrammaticLaunchCompletion();
    const int route = blockIdx.x, expert = ids[route];
    if (expert < 0) return;
    const int start = offsets[expert], rank = ranks[route];
    const auto base = size_t(start + rank) * (2 * width);
    const float ws = Narrow ? 0.f : weightScales[expert], xs = Narrow ? 0.f : input[size_t(route / topK) * (hidden / 16)].outer;
    const float score = scores[route];
    const int group = threadIdx.x;
    float values[16], peak = 0.f;
    if (group < width / 16) {
#pragma unroll
        for (int k = 0; k < 16; k += 4) {
            const int col = group * 16 + k;
            unsigned short g[4], u[4];
            if constexpr (Narrow) {
                const auto gate = *reinterpret_cast<const uint2 *>(paired + base / 2 + col);
                const auto up = *reinterpret_cast<const uint2 *>(paired + size_t(gridDim.x) * width + base / 2 + col);
                memcpy(g, &gate, sizeof(g));
                memcpy(u, &up, sizeof(u));
            } else {
                const auto *dots = buffers.accumulators + base;
                const float4 gate = *reinterpret_cast<const float4 *>(dots + col);
                const float4 up = *reinterpret_cast<const float4 *>(dots + width + col);
                float gv[4], uv[4];
                memcpy(gv, &gate, sizeof(gv));
                memcpy(uv, &up, sizeof(uv));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    g[j] = __bfloat16_as_ushort(__float2bfloat16_rn(gv[j] * ws * xs));
                    u[j] = __bfloat16_as_ushort(__float2bfloat16_rn(uv[j] * ws * xs));
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                values[k + j] = float(__float2bfloat16_rn(factors[g[j]] * factors[65536 + u[j]] * score));
                peak = fmaxf(peak, fabsf(values[k + j]));
            }
        }
    }
    __shared__ float maxima[native::QuantThreads / 32];
    float rowPeak = peak;
    for (int d = 16; d; d /= 2) rowPeak = fmaxf(rowPeak, __shfl_xor_sync(0xffffffff, rowPeak, d));
    if (!(group % 32)) maxima[group / 32] = rowPeak;
    __syncthreads();
    for (int i = 0; i < blockDim.x / 32; ++i) rowPeak = fmaxf(rowPeak, maxima[i]);
    const float outer = rowPeak / 1536.f;
    if (!group) midScales[route] = outer;
    if (group < width / 16) {
        const auto block = native::encodeBlock(values, peak, outer);
        native::GroupedInputRow dst(buffers, start, expert, rank, width / 16);
        dst.codes[group] = block.words;
        dst.scale(group) = block.scale;
    }
#endif
}

#endif

template<bool Pdl = false, bool Local = false, class Out = float>
__global__ void sumNativeRoutes(const __nv_bfloat16 *routes, Out *out, int rows, int width, int top,
                                int inputPitch, int outputPitch, const int *ids = nullptr) {
    if constexpr (Pdl) cuda_launch::DependencyStart();
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * width) return;
    const auto *src = routes + (i / width) * top * inputPitch + i % width;
    float sum = 0;
    for (int j = 0; j < top; ++j) {
        if constexpr (Local) if (ids[(i / width) * top + j] < 0) continue;
        sum += float(src[size_t(j) * inputPitch]);
    }
    out[(i / width) * outputPitch + i % width] = sum;
}

struct RouteSelection {
    const float *logits, *bias;
    int *ids;
    float *scores;
    int experts, top;
    float scale;
    int firstExpert = 0, localExperts = 0, absentId = 0;

    // Normalize the global top-k before retaining the contributions owned by
    // this rank. The selection CTA already owns the entire row.
    __device__ int restrictId(int i) const {
        const int id = ids[i] - firstExpert;
        if (unsigned(id) < unsigned(localExperts)) return ids[i] = id;
        scores[i] = 0.f;
        return ids[i] = absentId;
    }

    __device__ void restrictToRank(int row) const {
        if (!localExperts) return;
        __syncthreads();
        if (threadIdx.x < top) restrictId(row * top + threadIdx.x);
    }
};

template <int FixedExperts>
__device__ void selectRow(RouteSelection route) {
    if constexpr (FixedExperts) {
        if (threadIdx.x < 32)
            cuda::SelectExpertFixedRow<false, FixedExperts, 8, true>(
                route.logits, route.bias, route.ids, route.scores, route.bias != nullptr, true, route.scale);
    } else {
        cuda::SelectExpertRow<64, 50, true>(route.logits, route.bias, route.ids, route.scores,
                                            route.experts, route.top, route.bias != nullptr, true, route.scale);
    }
}

// Each token has one independent routing CTA alongside its quantization tiles.
template <bool fp4, int FixedExperts, typename Out>
__global__ void selectAndQuantize(InputRow input, Out *out, int d, RouteSelection route) {
    const unsigned tiles = gridDim.y - 1;
    if (blockIdx.y != tiles)
        reconstructRow<fp4>(input, out, d, false, tiles);
    else {
        selectRow<FixedExperts>(route);
        route.restrictToRank(blockIdx.x);
    }
}

#ifdef FASTLLM_INKLING_GROUPED_SM120
// Each lane owns a 16-value block. Keep its values in registers across
// the row-maximum reduction and use native packed E2M1 conversion.
__global__ void selectAndPackFp4(InputRow input, native::Block16 *out, int width, RouteSelection route) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    if (blockIdx.y) {
        selectRow<128>(route);
        route.restrictToRank(blockIdx.x);
        return;
    }
    const int group = threadIdx.x, groups = width / 16;
    float values[16], peak = 0.f;
    if (group < groups) {
        const float *src = input.x + size_t(blockIdx.x) * width + group * 16;
#pragma unroll
        for (int j = 0; j < 16; j += 4) {
            const float4 v = *reinterpret_cast<const float4 *>(src + j);
            memcpy(values + j, &v, sizeof(v));
            peak = fmaxf(peak, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }
    __shared__ float maxima[8];
    float rowPeak = peak;
#pragma unroll
    for (int d = 16; d; d /= 2) rowPeak = fmaxf(rowPeak, __shfl_xor_sync(0xffffffff, rowPeak, d));
    if (!(group % 32)) maxima[group / 32] = rowPeak;
    __syncthreads();
    if (group >= groups) return;
    for (int j = 0; j < blockDim.x / 32; ++j) rowPeak = fmaxf(rowPeak, maxima[j]);
    const float outer = rowPeak / 1536.f;
    const auto block = native::encodeBlock(values, peak, outer);
    auto &dst = out[size_t(blockIdx.x) * groups + group];
    dst.code[0] = block.words.x;
    dst.code[1] = block.words.y;
    dst.scale = block.scale;
    dst.outer = outer;
#endif
}
#endif

template <int FixedExperts>
__global__ void selectRanked(RouteSelection route, int *ranks, int *counts) {
    selectRow<FixedExperts>(route);
    __syncthreads();
    if (threadIdx.x < route.top) {
        const int r = blockIdx.x * route.top + threadIdx.x;
        const int expert = route.localExperts ? route.restrictId(r) : route.ids[r];
        if (expert >= 0) ranks[r] = atomicAdd(counts + expert, 1);
    }
}
// The last projection CTA sees all expert probabilities through the release sequence.
// Quantization CTAs have independent outputs and never take part in the counter.
struct DecodeRouter {
    static constexpr int experts = 128;
    alignas(16) float probabilities[experts];
    unsigned completed;
};
#if CUDART_VERSION >= 12000
// Put each lane's next eight BF16 coefficients in one coalesced 16-byte load.
// Keep the original FP32 weights for multi-row cuBLAS and non-BF16 checkpoints.
__global__ void packDecodeRouter(const float *weight, unsigned short *packed, unsigned *lossy) {
    size_t col = size_t(blockIdx.x) * 256 + threadIdx.x;
    unsigned bits = __float_as_uint(weight[col]);
    packed[size_t(blockIdx.x) * 256 + (threadIdx.x % 32) * 8 + threadIdx.x / 32] = bits >> 16;
    if (__any_sync(0xffffffff, (bits & 65535) != 0) && threadIdx.x % 32 == 0) atomicExch(lossy, 1u);
}
// Each warp reuses a token's input across experts and each coefficient across
// tokens. Keep the singleton's per-lane FMA sequence and final shuffle tree.
template<int Tokens, int Experts>
__global__ void batchedRouterDots(const float *input, const uint4 *packed, float *logits, int rows, int width) {
    constexpr int count = DecodeRouter::experts;
    static_assert(count % (4 * Experts) == 0 && Tokens % 2 == 0);
    const int lane = threadIdx.x % 32, first = blockIdx.y * Tokens;
    const int expert = (blockIdx.x * 4 + threadIdx.x / 32) * Experts;
    float sum[Experts][Tokens]{};
    for (int base = 0; base < width; base += 256) {
        uint4 words[Experts];
#pragma unroll
        for (int e = 0; e < Experts; ++e)
            words[e] = packed[size_t(expert + e) * (width / 8) + base / 8 + lane];
#pragma unroll
        for (int term = 0; term < 8; ++term) {
            float x[Tokens];
#pragma unroll
            for (int r = 0; r < Tokens; ++r)
                x[r] = input[size_t(min(first + r, rows - 1)) * width + base + term * 32 + lane];
#pragma unroll
            for (int e = 0; e < Experts; ++e) {
                const unsigned pairs[] = {words[e].x, words[e].y, words[e].z, words[e].w};
                const unsigned pair = pairs[term / 2];
                const float w = __uint_as_float(term % 2 ? (pair & 0xffff0000u) : (pair << 16));
#pragma unroll
                for (int r = 0; r < Tokens; r += 2) {
#if CUDART_VERSION >= 12080 && __CUDA_ARCH__ >= 1000
                    const float2 v = __ffma2_rn({x[r], x[r + 1]}, {w, w}, {sum[e][r], sum[e][r + 1]});
                    sum[e][r] = v.x;
                    sum[e][r + 1] = v.y;
#else
                    sum[e][r] = fmaf(x[r], w, sum[e][r]);
                    sum[e][r + 1] = fmaf(x[r + 1], w, sum[e][r + 1]);
#endif
                }
            }
        }
    }
#pragma unroll
    for (int e = 0; e < Experts; ++e) {
#pragma unroll
        for (int r = 0; r < Tokens; ++r) {
            float dot = sum[e][r];
#pragma unroll
            for (int step = 16; step; step /= 2) dot += __shfl_down_sync(0xffffffff, dot, step);
            if (!lane && first + r < rows) logits[size_t(first + r) * count + expert + e] = dot;
        }
    }
}
template <bool fp4, class Out, int ExpertTile, bool Pdl = false>
__global__ void projectSelectAndQuantize(InputRow input, Out *out, int d, RouteSelection route,
                                         const uint4 *packed, DecodeRouter *scratch, unsigned tiles) {
    if constexpr (Pdl) {
        // Router coefficients are independent of the incoming residual.
        if (blockIdx.y >= tiles && threadIdx.x / 32 < ExpertTile) {
            int expert = (blockIdx.y - tiles) * ExpertTile + threadIdx.x / 32;
            auto w = packed + size_t(expert) * (d / 8) + threadIdx.x % 32;
            for (int col = 0; col < d; col += 256)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(w + col / 8));
        }
        cuda_launch::DependencyStart();
    }
    if (blockIdx.y < tiles) {
        reconstructRow<fp4>(input, out, d, false, tiles);
        return;
    }
    input.x += size_t(blockIdx.x) * d;
    scratch += blockIdx.x;
    route.ids += size_t(blockIdx.x) * route.top;
    route.scores += size_t(blockIdx.x) * route.top;
    const int group = threadIdx.x / 32;
    if (group < ExpertTile) {
        const int expert = (blockIdx.y - tiles) * ExpertTile + group;
        const int lane = threadIdx.x % 32;
        const uint4 *w = packed + size_t(expert) * (d / 8);
        // Each lane's FMA and warp reduction order match the single-row FP32
        // projection. Sub-ULP score changes can alter later BF16 activations.
        float sum = 0;
        for (int col = 0; col < d; col += 256) {
            uint4 word = w[col / 8 + lane];
            const unsigned pairs[4] = {word.x, word.y, word.z, word.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float lo = __uint_as_float(pairs[j] << 16);
                float hi = __uint_as_float(pairs[j] & 0xffff0000u);
                sum = fmaf(input.x[col + lane + 64 * j], lo, sum);
                sum = fmaf(input.x[col + lane + 64 * j + 32], hi, sum);
            }
        }
        for (int delta = 16; delta; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (lane == 0) scratch->probabilities[expert] = 1.0 / (1.0 + expf(-sum));
    }
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) {
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device> completed(scratch->completed);
        last = completed.fetch_add(1, ::cuda::memory_order_acq_rel) + 1 == DecodeRouter::experts / ExpertTile;
        if (last) completed.store(0, ::cuda::memory_order_relaxed);
    }
    __syncthreads();
    if (last) {
        if (threadIdx.x < 32)
            cuda::SelectExpertFixedRow<false, DecodeRouter::experts, 8>(
                scratch->probabilities, route.bias, route.ids, route.scores, route.bias != nullptr, true, route.scale, 0);
        route.restrictToRank(0);
    }
}
#endif

template <bool fp4> int quantTiles(int rows, int d, int smCount) {
    if (d % (fp4 ? 16 : 32)) throw std::runtime_error("Inkling expert activation block alignment");
    // Replicate a row maximum only while there are idle SMs.
    if constexpr (fp4) return std::min(1 + (d - 1) / 512, 1 + (smCount - 1) / rows);
    return std::min(1 + (d - 1) / 256, 65534);
}

template <typename Source, typename Out> void quantizeFp4Into(Source src, Out *out, int rows, int d, int smCount, bool pdl = false) {
    const size_t bytes = size_t(d) * sizeof(float);
    const bool cache = !std::is_same_v<Source, InputRow> && bytes + 16 * sizeof(float) <= 48 * 1024;
    if (pdl) {
        if (!cuda_launch::Launch<reconstructExpertRows<true, true, Source, Out>>(
                dim3(rows, quantTiles<true>(rows, d, smCount)), 512, cache ? bytes : 0, cudaStreamPerThread, src, out, d, cache))
            throw std::runtime_error("Inkling FP4 activation launch failed");
        return;
    }
    reconstructExpertRows<true>
        <<<dim3(rows, quantTiles<true>(rows, d, smCount)), 512, cache ? bytes : 0, cudaStreamPerThread>>>(
            src, out, d, cache);
}

template <typename Source, typename Out>
void quantizeInto(Source src, Out *out, int rows, int d, bool fp4, int smCount, bool decode = false) {
    if (fp4) {
        quantizeFp4Into(src, out, rows, d, smCount, decode);
    } else if (decode) {
        if (!cuda_launch::Launch<reconstructExpertRows<false, true, Source, Out>>(
                dim3(rows, quantTiles<false>(rows, d, smCount)), 256, 0, cudaStreamPerThread, src, out, d, false))
            throw std::runtime_error("Inkling MXFP8 activation launch failed");
    } else {
        reconstructExpertRows<false>
            <<<dim3(rows, quantTiles<false>(rows, d, smCount)), 256, 0, cudaStreamPerThread>>>(src, out, d, false);
    }
}

template <bool fp4, class Out>
void prepareRoutedInput(InputRow src, Out *out, int rows, int d, int smCount, RouteSelection route,
                        const uint4 *packed, DecodeRouter *scratch) {
#if CUDART_VERSION >= 12000
    if (scratch) {
        unsigned tiles = quantTiles<fp4>(rows, d, smCount);
        constexpr int threads = fp4 ? 512 : 256;
        const int64_t work = int64_t(rows) * DecodeRouter::experts;
        auto launch = [&](auto tile) {
            constexpr int experts = decltype(tile)::value;
            if (!cuda_launch::Launch<projectSelectAndQuantize<fp4, Out, experts, true>>(
                    dim3(rows, tiles + DecodeRouter::experts / experts), threads, 0, cudaStreamPerThread,
                    src, out, d, route, packed, scratch, tiles))
                throw std::runtime_error("Inkling router launch failed");
        };
        // Each warp owns one expert dot. Use more warps per CTA when the
        // batch has enough independent dots to still fill a wave of SMs.
        if constexpr (fp4) {
            if (work >= 16LL * smCount) return launch(std::integral_constant<int, 16>{});
        }
        if (work >= 8LL * smCount) return launch(std::integral_constant<int, 8>{});
        if (work >= 4LL * smCount) return launch(std::integral_constant<int, 4>{});
        return launch(std::integral_constant<int, 2>{});
    }
#endif
#ifdef FASTLLM_INKLING_GROUPED_SM120
    if constexpr (fp4 && std::is_same_v<Out, native::Block16>) {
        if (route.localExperts && route.absentId < 0 && rows >= smCount && route.top == 8 && route.experts == 128 &&
            d % 16 == 0 && d <= 4096) {
            const int threads = (d / 16 + 31) / 32 * 32;
            selectAndPackFp4<<<dim3(rows, 2), threads, 0, cudaStreamPerThread>>>(src, out, d, route);
            return;
        }
    }
#endif
    dim3 grid(rows, 1 + quantTiles<fp4>(rows, d, smCount));
    const auto kernel = route.top == 8 && route.experts == 128 ? selectAndQuantize<fp4,128,Out> :
                        route.top == 8 && route.experts == 256 ? selectAndQuantize<fp4,256,Out> :
                        selectAndQuantize<fp4,0,Out>;
    kernel<<<grid, fp4 ? 512 : 256, 0, cudaStreamPerThread>>>(src, out, d, route);
}

Matrix routeLogits(const Matrix &x, const Data &weight, int promptRows = 0) {
    Matrix logits(x.rows, weight.dims[0]);
    const int rows = promptRows ? promptRows : x.rows;
    for (int first = 0; first < x.rows; first += rows) {
        CudaView input(FLOAT32, {rows, x.cols}, x.p + size_t(first) * x.cols);
        CudaView output(FLOAT32, {rows, logits.cols}, logits.p + size_t(first) * logits.cols);
        FastllmCudaBatchMatMulTransB(input, weight, output, 0, 0, 0, x.cols, x.cols,
                                     1, rows, x.cols, logits.cols, 1.f);
    }
    return logits;
}

struct Packed {
    uint8_t *data;
    float *scales;
};
template <class T>
std::vector<T> readRange(const TensorRange &r, const std::vector<int> &shape, const std::string &type) {
    size_t size = 1;
    for (int d : shape) {
        if (d <= 0 || size > std::numeric_limits<size_t>::max() / size_t(d))
            throw std::runtime_error("Invalid Inkling weight dimension");
        size *= d;
    }
    if (r.shape != shape || r.type != type || r.end < r.start || (r.end - r.start) / sizeof(T) != size ||
        (r.end - r.start) % sizeof(T))
        throw std::runtime_error("Inkling expert shape or dtype mismatch in " + r.file + " (" + r.type + ")");
    std::ifstream f(r.file, std::ios::binary);
    std::vector<T> v(size);
    if (!f.seekg(r.start) || !f.read(reinterpret_cast<char *>(v.data()), r.end - r.start))
        throw std::runtime_error("Truncated Inkling expert tensor in " + r.file);
    return v;
}
#include "tensor-parallel.cuh"
struct ExpertLoadContext {
    std::vector<std::shared_ptr<TensorRank>> ranks;
    Buffer<float> situFactors;
    int nodes = 1, smCount = 0;
    bool marlin = false, nativeFp4 = false;
    int firstExpert = 0, localExperts = 0;
    void enableNuma() {
#ifdef USE_NUMAS
        nodes = GetNumasMoeShardCount();
        if (nodes < 1)
            throw std::runtime_error("Inkling NUMA topology is empty");
#else
        throw std::runtime_error("Inkling NUMA experts require a USE_NUMAS build");
#endif
    }
};

struct HostShard {
    std::shared_ptr<void> storage;
    uint8_t *values = nullptr;
};
struct Projection {
    std::vector<WeightShard> tensorShards;
    int rows, cols, group;
    bool fp4;
#ifdef FASTLLM_INKLING_GROUPED_SM75
    bool exactHalfWeights = false;
#endif
    ExpertDevice device;
    std::vector<std::unique_ptr<Data>> resident;
    Buffer<uint4> nativeCodes;
    Buffer<unsigned> nativeScales;
    Buffer<float> nativeOuter;
#ifdef FASTLLM_INKLING_GROUPED_SM120
    native::GroupedWeightMaps groupedMaps{};
    bool haveGroupedMaps = false;
#endif
    native::Weights nativeView() const { return {nativeCodes.p, nativeScales.p, nativeOuter.p}; }
    template<class Launch>
    void projectShards(Launch launch) const {
        if (tensorShards.empty()) {
            launch(nativeView(), 0, rows, cudaStreamPerThread, nullptr, 0);
            return;
        }
        for (size_t i = 1; i < tensorShards.size(); ++i) {
            const auto &shard = tensorShards[i];
            auto &rank = *shard.rank;
            check(cudaEventRecord(rank.ready, cudaStreamPerThread));
            TensorRank::DeviceScope scope(rank.device);
            check(cudaStreamWaitEvent(rank.stream, rank.ready, 0));
            launch(shard.view(), shard.first, shard.rows, rank.stream, &rank, i);
            check(cudaGetLastError());
            check(cudaEventRecord(rank.done, rank.stream));
        }
        const auto &local = tensorShards.front();
        launch(local.view(), local.first, local.rows, cudaStreamPerThread, nullptr, 0);
        for (size_t i = 1; i < tensorShards.size(); ++i)
            check(cudaStreamWaitEvent(cudaStreamPerThread, tensorShards[i].rank->done, 0));
    }
    std::vector<HostShard> shards;
    std::vector<Packed> host;
    Projection(const std::map<std::string, TensorRange> &files, const std::string &name, int e, int r, int c,
               ExpertDevice where, ExpertLoadContext &topology, bool gateUp)
        : rows(r), cols(c), device(where) {
        const auto &weight = files.at(name);
        fp4 = weight.type == "U8";
        group = fp4 ? 16 : 32;
        if (c % group)
            throw std::runtime_error("Inkling expert columns are not group aligned");
        auto raw = readRange<uint8_t>(weight, {e, r, fp4 ? c / 2 : c}, fp4 ? "U8" : "F8_E4M3");
        auto scale = readRange<uint8_t>(files.at(name + "_scale"), {e, r, c / group}, fp4 ? "F8_E4M3" : "U8");
#ifdef FASTLLM_INKLING_GROUPED_SM75
        if (where == ExpertDevice::Cuda && !fp4 && FastllmCudaRuntimeArch() == 75) {
            // Every finite E4M3 value times 2^exp fits FP16 exactly for -15 <= exp <= 7.
            exactHalfWeights = std::all_of(scale.begin(), scale.end(), [](uint8_t x) { return x >= 112 && x <= 134; });
        }
#endif
        std::vector<float> outer(e, 1);
        if (fp4) {
            outer = readRange<float>(files.at(name + "_amax"), {e}, "F32");
            float den = readRange<float>(files.at(name + "_outer_scale_denominator"), {1}, "F32")[0];
            if (!(den > 0 && std::isfinite(den)))
                throw std::runtime_error("Invalid Inkling outer scale");
            for (auto &a : outer) {
                if (!(a >= 0 && std::isfinite(a)))
                    throw std::runtime_error("Invalid Inkling expert amax");
                a /= den;
            }
        }
        size_t bytes = raw.size() / e, groups = scale.size() / e;
        if (where == ExpertDevice::Cuda && fp4 && topology.nativeFp4) {
            if (std::any_of(scale.begin(), scale.end(), [](uint8_t v) { return (v & 127) == 127; }))
                throw std::runtime_error("Invalid Inkling NVFP4 expert block scale");
            nativeCodes = Buffer<uint4>(raw.size() / 16);
            nativeScales = Buffer<unsigned>(scale.size() / 4);
            nativeOuter = Buffer<float>(e);
#ifdef FASTLLM_INKLING_GROUPED_SM120
            haveGroupedMaps = topology.ranks.empty() && native::makeGroupedWeightMaps(groupedMaps, nativeView(), e, r, c);
#endif
            check(cudaMemcpy(nativeOuter.p, outer.data(), e * sizeof(float), cudaMemcpyHostToDevice));
            Buffer<unsigned> staging((bytes + groups) / sizeof(unsigned));
            for (int id = 0; id < e; ++id) {
                check(cudaMemcpy(staging.p, raw.data() + id * bytes, bytes, cudaMemcpyHostToDevice));
                check(
                    cudaMemcpy(staging.p + bytes / 4, scale.data() + id * groups, groups, cudaMemcpyHostToDevice));
                native::pack<<<(bytes / 16 + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                    staging.p, staging.p + bytes / 4, nativeCodes.p + id * (bytes / 16),
                    nativeScales.p + id * (groups / 4), r, c, gateUp);
            }
            check(cudaGetLastError());
            if (!topology.ranks.empty()) {
                check(cudaStreamSynchronize(cudaStreamPerThread));
                const int tiles = r / 16, nranks = topology.ranks.size();
                if (tiles < nranks) throw std::runtime_error("Too many Inkling TP ranks for expert width");
#ifdef FASTLLM_INKLING_GROUPED_SM120
                haveGroupedMaps = true;
#endif
                for (int i = 0; i < nranks; ++i) {
                    const int first = (tiles * i / nranks) * 16;
                    const int nr = (tiles * (i + 1) / nranks) * 16 - first;
                    const auto &rank = topology.ranks[i];
                    WeightShard shard{rank, first, nr};
                    const size_t codeRow = c / 32, scaleRow = c / 64;
                    shard.codes = copyWeightShard(nativeCodes.p, r * codeRow, first * codeRow, nr * codeRow, e, *rank);
                    shard.scales = copyWeightShard(nativeScales.p, r * scaleRow, first * scaleRow, nr * scaleRow, e, *rank);
                    shard.outer = copyWeightShard(nativeOuter.p, 1, 0, 1, e, *rank);
#ifdef FASTLLM_INKLING_GROUPED_SM120
                    TensorRank::DeviceScope scope(rank->device);
                    haveGroupedMaps &= native::makeGroupedWeightMaps(shard.groupedMaps, shard.view(), e, nr, c);
#endif
                    tensorShards.push_back(std::move(shard));
                }
                nativeCodes = tensorShards.front().codes;
                nativeScales = tensorShards.front().scales;
                nativeOuter = tensorShards.front().outer;
            }
            return;
        }
        const int nodes = where == ExpertDevice::Numa ? topology.nodes : 1;
        if (where != ExpertDevice::Cuda) {
            shards.resize(nodes);
            host.resize(e);
            for (int n = 0; n < nodes; ++n) {
                size_t per = (e + nodes - 1 - n) / nodes;
                if (!per)
                    continue;
                const size_t total = per * (bytes + groups * sizeof(float));
                auto &s = shards[n];
#ifdef USE_NUMAS
                if (where == ExpertDevice::Numa) {
                    s.storage = {allocate_aligned_numa(total, n),
                                 [total](void *v) { free_aligned_numa(v, total); }};
                } else
#endif
                    s.storage = {::operator new(total), [](void *v) { ::operator delete(v); }};
                if (!s.storage)
                    throw std::bad_alloc();
                s.values = static_cast<uint8_t *>(s.storage.get());
            }
        }
        std::vector<float> decoded(groups);
        for (int id = 0; id < e; ++id) {
            for (size_t i = 0; i < groups; ++i) {
                auto code = scale[size_t(id) * groups + i];
                if ((fp4 && (code & 127) == 127) || (!fp4 && code == 255))
                    throw std::runtime_error("Invalid Inkling block scale");
                decoded[i] = fp4 ? fp8Value(code) * outer[id] : std::ldexp(1.0f, int(code) - 127);
            }
            if (where == ExpertDevice::Cuda) {
                const auto type = !fp4 ? FP8_E4M3 : topology.marlin ? NVFP4_BLOCK_16_E4M3 : NVFP4_BLOCK_16;
                auto weight = std::make_unique<Data>(type);
                weight->blockK = 1;
                weight->blockM = group;
                weight->directMemory = true;
                weight->Resize({r, c});
                weight->dataDevice = CUDA;
                weight->dataDeviceIds = {FastllmCudaGetDevice()};
                weight->Allocate(false);
                std::vector<uint8_t> packed(weight->GetBytes());
                const int rowBytes = bytes / r, rowGroups = c / group;
                if (!fp4) weight->scales.resize(groups);
                if (type == NVFP4_BLOCK_16_E4M3)
                    weight->scales.assign(gateUp ? 2 : 1, outer[id] > 0 ? outer[id] : 1.f);
                for (int row = 0; row < r; ++row) {
                    int src = gateUp ? 2 * (row % (r / 2)) + row / (r / 2) : row;
                    const uint8_t *values = raw.data() + id * bytes + size_t(src) * rowBytes;
                    if (type == NVFP4_BLOCK_16) {
                        for (int b = 0; b < rowGroups; ++b) {
                            uint8_t *block = packed.data() + (size_t(row) * rowGroups + b) * 12;
                            std::memcpy(block, values + b * 8, 8);
                            std::memcpy(block + 8, &decoded[size_t(src) * rowGroups + b], 4);
                        }
                    } else {
                        std::memcpy(packed.data() + size_t(row) * rowBytes, values, rowBytes);
                        if (fp4 && outer[id] != 0)
                            std::memcpy(packed.data() + bytes + size_t(row) * rowGroups,
                                        scale.data() + id * groups + size_t(src) * rowGroups, rowGroups);
                        if (!fp4)
                            std::copy_n(decoded.data() + size_t(src) * rowGroups, rowGroups,
                                        weight->scales.data() + size_t(row) * rowGroups);
                    }
                }
                check(cudaMemcpy(weight->cudaData, packed.data(), packed.size(), cudaMemcpyHostToDevice));
                resident.push_back(std::move(weight));
            } else {
                auto &s = shards[id % nodes];
                // NUMA and CUDA consume the same immutable per-expert record.
                auto *w = s.values + (id / nodes) * (bytes + groups * sizeof(float));
                auto *a = fp4 ? nullptr : reinterpret_cast<float *>(w + bytes);
                const int rowBytes = bytes / r, rowGroups = c / group;
                for (int row = 0; row < r; ++row) {
                    const int src = gateUp ? 2 * (row % (r / 2)) + row / (r / 2) : row;
                    auto *codes = raw.data() + size_t(id) * bytes + size_t(src) * rowBytes;
                    auto *scales = decoded.data() + size_t(src) * rowGroups;
                    if (fp4) {
                        auto *dst = w + size_t(row) * rowGroups * 12;
                        for (int block = 0; block < rowGroups; ++block) {
                            std::memcpy(dst + 12 * block, codes + 8 * block, 8);
                            std::memcpy(dst + 12 * block + 8, scales + block, sizeof(float));
                        }
                    } else {
                        std::memcpy(w + size_t(row) * rowBytes, codes, rowBytes);
                        std::copy_n(scales, rowGroups, a + size_t(row) * rowGroups);
                    }
                }
                host[id] = {w, a};
            }
        }
    }
};
class RoutedExperts : public ExpertBackend, protected CudaMoeActivation {
protected:
    const Data &routerWeight;
    Projection gate, down;
    const int smCount;
    float gateCap, upCap;

    void Run(const Data &pair, const Data &scores, Data &mid, int rows, int width) const override {
        auto launch = [&](const auto *pairData, auto *out) {
            using T = std::remove_pointer_t<decltype(out)>;
            RoutedSiTU<T> source{pairData, static_cast<const float *>(scores.cudaData), gateCap, upCap};
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                if (!gate.fp4 && rows >= smCount && width % 32 == 0) {
                    if (!cuda_launch::Launch<reconstructMxfp8SiTU>(rows, 128, 0, cudaStreamPerThread, source, out, width))
                        throw std::runtime_error("Inkling MXFP8 SiTU launch failed");
                    return;
                }
            }
            quantizeInto(source, out, rows, width, gate.fp4, smCount, !gate.fp4 || scores.dims[0] == 1);
        };
        if (pair.dataType == FLOAT16 && mid.dataType == FLOAT16)
            launch(static_cast<const half *>(pair.cudaData), static_cast<half *>(mid.cudaData));
        else if (pair.dataType == BFLOAT16 && mid.dataType == BFLOAT16)
            launch(static_cast<const __nv_bfloat16 *>(pair.cudaData), static_cast<__nv_bfloat16 *>(mid.cudaData));
        else if (pair.dataType == FLOAT32 && mid.dataType == FLOAT32)
            launch(static_cast<const float *>(pair.cudaData), static_cast<float *>(mid.cudaData));
        else
            throw std::runtime_error("Inkling expert activation dtype mismatch");
    }

public:
    RoutedExperts(const std::map<std::string, TensorRange> &files, const std::string &name, const Config &c,
                  ExpertDevice where, ExpertLoadContext &topology, const Data &routerWeight)
        : routerWeight(routerWeight), gate(files, name + "w13_weight", topology.localExperts ? topology.localExperts : c.experts, 2 * c.intermediate, c.hidden, where, topology, true),
          down(files, name + "w2_weight", topology.localExperts ? topology.localExperts : c.experts, c.hidden, c.intermediate, where, topology, false),
          smCount(topology.smCount), gateCap(c.gateCap), upCap(c.upCap) {
        if (gate.fp4 != down.fp4)
            throw std::runtime_error("Inkling expert projection types disagree");
    }
};
class GpuExperts final : public RoutedExperts {
    int firstExpert, localExperts;
    Buffer<float> situFactors;
    Buffer<DecodeRouter> decodeRouter;
    Buffer<uint4> packedRouter;
    std::vector<Data *> weights;
    std::unique_ptr<CudaGroupedMxfp8Moe> groupedFp8;
#ifdef FASTLLM_INKLING_GROUPED_SM120
    bool nativeMxfp8 = false;
#endif

#ifdef FASTLLM_INKLING_GROUPED_SM75
    CudaIndexedMoeSlots sm75Weights{};
    int sm75Blocks = 0, sm75LargeBlocks = 0;

    void sm75Prefill(const Config &c, const Matrix &x, const float *logits, const float *bias, float *dst) const {
        const int rows = x.rows, routes = rows * c.topK;
        const auto stream = cudaStreamPerThread;
        Buffer<sm75::Plan> plan(1);
        Buffer<int> indices(routes), ranks(routes);
        Matrix scores(rows, c.topK);
        Buffer<half> input(size_t(routes) * std::max(gate.cols, down.cols));
        Buffer<float> sorted(size_t(routes) * std::max(gate.rows, down.rows));
        RouteSelection route{logits, bias, indices.p, scores.p, c.experts, c.topK, c.routeScale};
        check(cudaMemsetAsync(plan.p->counts, 0, sizeof(plan.p->counts), stream));
        selectRanked<sm75::Experts><<<rows, 32, 0, stream>>>(route, ranks.p, plan.p->counts);
        sm75::prefixRoutes<<<1, 1, 0, stream>>>(plan.p);
        auto project = [&](const Projection &p, uint8_t **codes, float **scales, int count, auto source) {
            packSm75Routes<<<count, 256, 0, stream>>>(
                source, input.p, indices.p, plan.p->offsets, ranks.p, p.cols);
            if (rows < 1024) {
                sm75::project<16>(plan.p, input.p, sorted.p, codes, scales, p.rows, p.cols, sm75Blocks, stream);
            } else {
                sm75::project<128>(plan.p, input.p, sorted.p, codes, scales, p.rows, p.cols, sm75LargeBlocks, stream);
            }
        };
        project(gate, sm75Weights.gate, sm75Weights.gateScales, rows, InputRow{x.p});
        project(down, sm75Weights.down, sm75Weights.downScales, routes,
                RoutedSiTU<float>{sorted.p, route.scores, gateCap, upCap});
        sumMxfp8Routes<<<rows, 256, 0, stream>>>(sorted.p, dst, indices.p, plan.p->offsets, ranks.p, down.rows, c.topK);
        check(cudaGetLastError());
    }
#endif

    template <int TileRows, int ColumnTiles>
    void projectGrouped(const Config &c, int routes, RouteSelection route, const native::Block16 *input,
                        const int *ranks, const int *counts, InputRow source,
                        __nv_bfloat16 *pair, __nv_bfloat16 *projected, RoutedSiTU<__nv_bfloat16> act, float *reduced) const {
        constexpr int warps = 8 / ColumnTiles, ctaColumns = 16 * warps * ColumnTiles;
        constexpr int threads = 32 * warps * ((TileRows + native::WarpTileRows - 1) / native::WarpTileRows);
        const int experts = localExperts ? localExperts : c.experts;
        const int tiles = (routes + (TileRows - 1) * std::min(experts, routes)) / TileRows;
        Buffer<native::RouteGroup<TileRows>> tasks(tiles);
        Buffer<native::PlanSize> used(1);
        Buffer<float> scales(routes);
        Buffer<native::InputTile> packed(size_t(tiles) * (TileRows / 8) * (std::max(gate.cols, down.cols) / 64));
        auto project = [&](const Projection &w, auto up, auto compact, const native::Block16 *x,
                           const float *rowScales, __nv_bfloat16 *dst, int divisor) {
            w.projectShards([&](native::Weights view, int first, int n, cudaStream_t stream, TensorRank *peer, size_t) {
                const auto *rankPacked = packed.p;
                const auto *rankTasks = tasks.p;
                const auto *rankUsed = used.p;
                const auto *rankScales = rowScales;
                const auto *rankInput = x;
                auto *rankOutput = dst + first;
                const int rows = routes / c.topK;
                const bool combine = !up && reduced;
                if (peer) {
                    const size_t packs = size_t(tiles) * (TileRows / 8) * (w.cols / 64);
                    const int nscales = up ? routes / c.topK : routes;
                    const size_t ninput = x ? size_t(nscales) * (w.cols / 16) : 0;
                    peer->reserve(packs * sizeof(*rankPacked) + tiles * sizeof(*rankTasks) +
                                  nscales * sizeof(float) + ninput * sizeof(*x) +
                                  size_t(routes) * n * sizeof(*dst) + (combine ? size_t(rows) * n * sizeof(float) : 0) + 7 * 256);
                    rankPacked = peer->receive(packed.p, packs);
                    rankTasks = peer->receive(tasks.p, tiles);
                    rankUsed = peer->receive(used.p, 1);
                    rankScales = peer->receive(rowScales, nscales);
                    rankInput = peer->receive(x, ninput);
                    rankOutput = peer->allocate<__nv_bfloat16>(size_t(routes) * n);
                }
                native::groupedProject<warps, ColumnTiles, decltype(up)::value, TileRows, decltype(compact)::value>
                    <<<dim3((n + ctaColumns - 1) / ctaColumns, tiles), threads, 0, stream>>>(
                        view, rankInput, rankScales, rankPacked, rankTasks, rankUsed,
                        rankOutput, n, w.cols, divisor, peer ? n : w.rows);
                if (combine) {
                    auto *out = peer ? peer->allocate<float>(size_t(rows) * n) : reduced + first;
                    const int pitch = peer ? n : w.rows;
                    sumNativeRoutes<><<<(size_t(rows) * n + 255) / 256, 256, 0, stream>>>(
                        rankOutput, out, rows, n, c.topK, pitch, pitch);
                    if (peer) peer->gather(out, reduced + first, rows, n, w.rows);
                } else if (peer) peer->gather(rankOutput, dst + first, routes, n, w.rows);
            });
        };
        if (ranks) {
            Buffer<int2> starts(experts);
            Buffer<float> inputScales(routes / c.topK);
            native::expertOffsets<TileRows><<<1, 32, 0, cudaStreamPerThread>>>(counts, starts.p, experts, used.p);
            const auto scatter = localExperts ? native::scatterInput<TileRows, true> : native::scatterInput<TileRows>;
            scatter<<<routes / c.topK, native::QuantThreads, 0, cudaStreamPerThread>>>(
                source.x, gate.cols, route.ids, ranks, counts, starts.p, c.topK, tasks.p, packed.p, inputScales.p);
            project(gate, std::true_type{}, std::true_type{}, nullptr, inputScales.p, pair, c.topK);
        } else {
            check(cudaMemsetAsync(used.p, 0, sizeof(native::PlanSize), cudaStreamPerThread));
            native::groupRoutes<<<experts, 256, 0, cudaStreamPerThread>>>(route.ids, routes, tasks.p, used.p);
            native::packInput<<<dim3((gate.cols / 64 + 7) / 8, tiles, TileRows / 8), 256, 0, cudaStreamPerThread>>>(
                tasks.p, used.p, input, packed.p, gate.cols, c.topK);
            project(gate, std::true_type{}, std::false_type{}, input, nullptr, pair, c.topK);
        }
        if (situFactors.p && routes >= c.experts) {
            const int threads = ((down.cols / 16 + 31) / 32) * 32;
            native::activateBlocks<TileRows><<<tiles * TileRows, threads, 0, cudaStreamPerThread>>>(
                PackedSiTU{pair, route.scores, situFactors.p}, tasks.p, used.p, down.cols, packed.p, scales.p);
        } else {
            native::launchPackedActivation(act, tasks.p, used.p, tiles, down.cols, packed.p, scales.p);
        }
        project(down, std::false_type{}, std::false_type{}, nullptr, scales.p, projected, 1);
    }

#if defined(FASTLLM_INKLING_GROUPED_SM120)
    void mxfp8Prefill(const Config &c, const Matrix &x, const float *logits, const float *bias, float *dst) const {
        const int rows = x.rows, routes = rows * c.topK;
        const int experts = localExperts ? localExperts : c.experts;
        Buffer<int> offsets(experts + 1), ranks(routes), indices(routes);
        Matrix scores(rows, c.topK);
        RouteSelection route{logits, bias, indices.p, scores.p, c.experts, c.topK, c.routeScale, firstExpert, localExperts, -1};
        const auto stream = cudaStreamPerThread;
        const int k = std::max(gate.cols, down.cols);
        // Gate/up has twice as many weights as down and both projections reuse the buffer.
        const size_t weightBytes = size_t(experts) * gate.rows * gate.cols;
        Buffer<uint8_t> codes(size_t(routes) * k), scales(cuda::grouped_sm120::scaleRows(routes, experts) * (k / 32));
        Buffer<uint8_t> weight(weightBytes + weightBytes / 32);
        Buffer<__nv_bfloat16> sorted(size_t(routes) * std::max(gate.rows, down.rows));
        const auto need = mx::scratchSize(experts, smCount);
        Buffer<uint8_t> metadata(need.metadata), work(need.gemm);
        mx::Buffers buffers{codes.p, scales.p, weight.p, sorted.p, metadata.p, work.p};
        // This dispatch is restricted to the 128-expert, top-8 router.
        check(cudaMemsetAsync(offsets.p, 0, experts * sizeof(int), stream));
        selectRanked<128><<<rows, 32, 0, stream>>>(route, ranks.p, offsets.p);
        check(cuda::grouped_sm120::prefixRouteCounts(offsets.p, experts, stream));
        auto project = [&](const Projection &p, bool isDown, auto source) {
            const auto pack = localExperts ? packMxfp8Routes<true, decltype(source)> : packMxfp8Routes<false, decltype(source)>;
            pack<<<isDown ? routes : rows, 128, 0, stream>>>(
                source, codes.p, scales.p, route.ids, offsets.p, ranks.p, p.cols);
            check(mx::project(groupedFp8->MarlinWeights(isDown), buffers, offsets.p, p.rows, p.cols,
                              experts, smCount, stream));
        };
        project(gate, false, InputRow{x.p});
        project(down, true, RoutedSiTU<__nv_bfloat16>{sorted.p, route.scores, gateCap, upCap});
        const auto sum = localExperts ? sumMxfp8Routes<true, __nv_bfloat16> : sumMxfp8Routes<false, __nv_bfloat16>;
        sum<<<rows, 256, 0, stream>>>(sorted.p, dst, route.ids, offsets.p, ranks.p, down.rows, c.topK);
        check(cudaGetLastError());
    }

    void groupedVerify(const Config &c, int rows, RouteSelection route, const native::Block16 *input, float *dst, __nv_bfloat16 *rankSum) const {
        const int groups = localExperts ? localExperts : c.experts;
        const int *ids = route.ids;
        const int routes = rows * c.topK, width = std::max(gate.cols, down.cols);
        Buffer<int> offsets(groups + 1), rank(routes);
        check(native::planGroupedRows(ids, offsets.p, rank.p, groups, routes, cudaStreamPerThread));
        Buffer<uint2> codes(size_t(routes) * (width / 16));
        Buffer<uint8_t> scales(cuda::grouped_sm120::scaleRows(routes, groups) * (width / 16));
        const bool narrow = gate.tensorShards.size() == 2 && gate.tensorShards.front().rows == c.intermediate &&
                            gate.tensorShards.back().rows == c.intermediate && situFactors.p && routes >= smCount && routes <= 65535;
        Buffer<__nv_bfloat16> paired(narrow ? size_t(routes) * gate.rows : 0);
        Buffer<float2> gateScales(narrow ? routes : 0);
        if (narrow) sortedGateScales<<<(routes + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            input, gate.nativeOuter.p, ids, offsets.p, rank.p, gateScales.p, routes, c.topK, gate.cols / 16);
        Buffer<float> sorted(narrow ? 0 : size_t(routes) * std::max(gate.rows, down.rows));
        const auto space = native::groupedScratch(groups, smCount);
        Buffer<uint8_t> metadata(space.metadata), workspace(space.gemm);
        native::GroupedBuffers buffers{codes.p, scales.p, sorted.p, metadata.p, workspace.p};
        check(native::packGroupedInput(input, ids, offsets.p, rank.p, buffers, routes, gate.cols, c.topK, cudaStreamPerThread));
        Buffer<float> midScales(routes);
        auto project = [&](const Projection &w, bool reduce) {
            if (w.tensorShards.empty()) {
                check(native::projectGroupedRows(w.nativeView(), w.groupedMaps, offsets.p, buffers, w.rows, w.cols,
                                                 groups, smCount, cudaStreamPerThread));
                return;
            }
            Buffer<float> localOutput(size_t(routes) * w.tensorShards.front().rows);
            w.projectShards([&](native::Weights weight, int first, int n, cudaStream_t stream, TensorRank *peer, size_t i) {
                auto b = buffers;
                int *starts = offsets.p;
                const int *routeIds = ids, *routeRanks = rank.p;
                const float *rowScales = midScales.p;
                const float2 *outer = gateScales.p;
                float *reduced = nullptr;
                Buffer<float> localReduced;
                b.accumulators = localOutput.p;
                if (peer) {
                    const auto codeCount = size_t(routes) * (w.cols / 16);
                    const auto scaleCount = cuda::grouped_sm120::scaleRows(routes, groups) * (w.cols / 16);
                    const auto work = native::groupedScratch(groups, peer->smCount);
                    peer->reserve(codeCount * sizeof(uint2) + scaleCount + size_t(routes) * n * sizeof(float) +
                                  (groups + 1) * sizeof(int) + work.metadata + work.gemm +
                                  (reduce ? routes * (2 * sizeof(int) + sizeof(float)) + size_t(rows) * n * sizeof(float) : 0) +
                                  (!reduce && narrow ? size_t(routes) * (sizeof(float2) + n * sizeof(__nv_bfloat16)) : 0) + 10 * 256);
                    b.codes = peer->receive(codes.p, codeCount);
                    b.scales = peer->receive(scales.p, scaleCount);
                    starts = peer->receive(offsets.p, groups + 1);
                    b.accumulators = peer->allocate<float>(size_t(routes) * n);
                    b.metadata = peer->allocate<uint8_t>(work.metadata);
                    b.gemm = peer->allocate<uint8_t>(work.gemm);
                    if (reduce) {
                        routeIds = peer->receive(ids, routes);
                        routeRanks = peer->receive(rank.p, routes);
                        rowScales = peer->receive(midScales.p, routes);
                        reduced = peer->allocate<float>(size_t(rows) * n);
                    } else if (narrow) outer = peer->receive(gateScales.p, routes);
                } else if (reduce) {
                    localReduced = Buffer<float>(size_t(rows) * n);
                    reduced = localReduced.p;
                }
                check(native::projectGroupedRows(weight, w.tensorShards[i].groupedMaps, starts, b, n, w.cols,
                                                 groups, peer ? peer->smCount : smCount, stream));
                if (!reduce && narrow) {
                    // Keep gate and up as contiguous planes. One DMA submits
                    // the entire rounded peer projection over PCIe.
                    const size_t values = size_t(routes) * n;
                    auto *plane = paired.p + size_t(routes) * first;
                    auto *rounded = peer ? peer->allocate<__nv_bfloat16>(values) : plane;
                    gatherGateBf16<<<dim3((n / 4 + 127) / 128, routes), 128, 0, stream>>>(
                        b.accumulators, outer, rounded, n / 4, n);
                    if (peer) check(cudaMemcpyPeerAsync(plane, peer->primary, rounded, peer->device,
                                                        values * sizeof(*rounded), stream));
                    return;
                }
                const size_t width = size_t(n) * sizeof(float), pitch = size_t(w.rows) * sizeof(float);
                if (reduce) {
                    // Each rank owns complete K reductions for its hidden channels.
                    // Combine top-k routes locally before transferring the final rows.
                    check(native::sumGroupedRows(b.accumulators, reduced, routeIds, starts, routeRanks, weight.outer,
                                                 rowScales, rows, n, c.topK, stream));
                }
                int count = reduce ? rows : routes;
                auto *src = reduce ? reduced : b.accumulators;
                auto *out = (reduce ? dst : sorted.p) + first;
                if (peer) peer->gather(src, out, count, n, w.rows);
                else copyRankRows<<<std::min(1024, (count * (n / 4) + 127) / 128), 128, 0, stream>>>(
                    reinterpret_cast<const char *>(src), reinterpret_cast<char *>(out), width, pitch, count);
            });
        };
        project(gate, false);
        const size_t bytes = size_t(c.intermediate) * sizeof(float);
        const bool cache = bytes + 16 * sizeof(float) <= 48 * 1024;
        if (situFactors.p && routes >= smCount) {
            const int threads = ((c.intermediate / 16 + 31) / 32) * 32;
            auto quantize = narrow ? quantizeGroupedSiTUBlocks<true> : quantizeGroupedSiTUBlocks<false>;
            quantize<<<routes, threads, 0, cudaStreamPerThread>>>(buffers, input, gate.nativeOuter.p, route.scores,
                ids, offsets.p, rank.p, midScales.p, c.hidden, c.intermediate, c.topK, situFactors.p, paired.p);
        } else {
            quantizeGroupedSiTU<<<dim3(routes, quantTiles<true>(routes, c.intermediate, smCount)), 512,
                                  cache ? bytes : 0, cudaStreamPerThread>>>(
                buffers, input, gate.nativeOuter.p, route.scores, ids, offsets.p, rank.p, midScales.p,
                c.hidden, c.intermediate, c.topK, gateCap, upCap, cache);
        }
        project(down, true);
        if (!down.tensorShards.empty()) return;
        auto combine = [&](auto *out) {
            check(native::sumGroupedRows(sorted.p, out, ids, offsets.p, rank.p, down.nativeOuter.p, midScales.p,
                                         rows, c.hidden, c.topK, cudaStreamPerThread));
        };
        if (rankSum) combine(rankSum);
        else combine(dst);
    }
#endif

    // The routing score is applied inside SiTU before the second FP4 quantization.
    void runNative(const Config &c, InputRow source, int rows, RouteSelection route, DecodeRouter *scratch,
                   float *dstRows, ExpertPhase phase, __nv_bfloat16 *rankSum, Rows<__nv_bfloat16> *decodeRoutes) const {
        const bool decodeArithmetic = phase == ExpertPhase::Verify;
        const int routes = rows * c.topK;
        // Dense routing amortizes the expert prefix pass. Each scatter lane
        // keeps at most two FP4 blocks in registers until its row scale is known.
        const bool ranked = !decodeArithmetic && (!localExperts || phase == ExpertPhase::Prefill) &&
                            routes / c.experts >= 32 && c.hidden <= native::QuantThreads * 2 * 16;
        Buffer<native::Block16> input(ranked ? 0 : size_t(rows) * c.hidden / 16);
        const int experts = localExperts ? localExperts : c.experts;
        Buffer<int> ranks(ranked ? routes : 0), counts(ranked ? experts : 0);
        if (ranked) {
            check(cudaMemsetAsync(counts.p, 0, size_t(experts) * sizeof(int), cudaStreamPerThread));
            const auto select = c.topK == 8 && c.experts == 128 ? selectRanked<128> :
                                c.topK == 8 && c.experts == 256 ? selectRanked<256> : selectRanked<0>;
            const int threads = c.topK == 8 && (c.experts == 128 || c.experts == 256) ? 32 : 64;
            select<<<rows, threads, 0, cudaStreamPerThread>>>(route, ranks.p, counts.p);
        } else {
            prepareRoutedInput<true>(source, input.p, rows, c.hidden, smCount, route, packedRouter.p, scratch);
        }
#if defined(FASTLLM_INKLING_GROUPED_SM120)
        if (!decodeRoutes && (!localExperts || decodeArithmetic) && phase != ExpertPhase::Prefill && !ranked && gate.haveGroupedMaps && down.haveGroupedMaps &&
            native::useGroupedVerify(rows, c.topK, c.experts, c.hidden, c.intermediate)) {
            groupedVerify(c, rows, route, input.p, dstRows, rankSum);
            return;
        }
#endif
        Rows<__nv_bfloat16> pair(routes, 2 * c.intermediate), projected(routes, c.hidden);
        RoutedSiTU<__nv_bfloat16> act{pair.p, route.scores, gateCap, upCap};
        float *reduced = !localExperts && !gate.tensorShards.empty() && rows > 1 && !decodeArithmetic ? dstRows : nullptr;
        if (rows == 1 || decodeArithmetic) {
            Buffer<native::Block16> mid(size_t(routes) * c.intermediate / 16);
            const int tileRows = native::decodeRouteTileRows(rows, c.topK);
            auto project = [&](const Projection &w, const native::Block16 *x, int divisor, __nv_bfloat16 *y) {
                using Edge = native::DecodeDependency;
                auto launch = [&](auto grouped, auto edge) {
                    const int capacity = tileRows * c.topK;
                    w.projectShards([&](native::Weights view, int first, int n, cudaStream_t stream, TensorRank *peer, size_t) {
                        const dim3 grid(n / 16, tileRows ? capacity : routes, tileRows ? (rows + tileRows - 1) / tileRows : 1);
                        if (!w.tensorShards.empty()) {
                            const native::Block16 *rankInput = x;
                            const int *ids = route.ids;
                            auto *output = y + first;
                            if (peer) {
                                const size_t blocks = size_t(routes / divisor) * (w.cols / 16);
                                peer->reserve(blocks * sizeof(*x) + size_t(routes) * (sizeof(int) + n * sizeof(*y)) + 3 * 256);
                                rankInput = peer->receive(x, blocks);
                                ids = peer->receive(route.ids, routes);
                                output = peer->allocate<__nv_bfloat16>(size_t(routes) * n);
                            }
                            native::decodeProject<4, decltype(grouped)::value><<<grid, 128, 0, stream>>>(
                                view, rankInput, ids, output, n, w.cols, divisor, routes, capacity, peer ? n : w.rows);
                            if (peer) peer->gather(output, y + first, routes, n, w.rows);
                        } else if (localExperts) {
                            if (!cuda_launch::Launch<native::decodeProject<4, decltype(grouped)::value, decltype(edge)::value, true>>(
                                    grid, 128, 0, stream, view, x, route.ids, y, n, w.cols, divisor, routes, capacity, w.rows))
                                throw std::runtime_error("Inkling local expert GEMV launch failed");
                        } else if (!cuda_launch::Launch<native::decodeProject<4, decltype(grouped)::value, decltype(edge)::value>>(
                                    grid, 128, 0, stream, view, x, route.ids, y, n, w.cols, divisor, routes, capacity, w.rows))
                            throw std::runtime_error("Inkling FP4 projection launch failed");
                    });
                };
                constexpr auto gateEdge = std::integral_constant<Edge, Edge::Gate>{};
                tileRows ? launch(std::true_type{}, gateEdge) :
                    &w == &gate ? launch(std::false_type{}, gateEdge) :
                    launch(std::false_type{}, std::integral_constant<Edge, Edge::Down>{});
            };
            project(gate, input.p, c.topK, pair.p);
            if (localExperts)
                quantizeFp4Into(LocalSiTU{act, route.ids}, mid.p, routes, c.intermediate, smCount, true);
            else quantizeFp4Into(act, mid.p, routes, c.intermediate, smCount, true);
            project(down, mid.p, 1, projected.p);
        } else if (routes / c.experts >= 64) {
            // At least one 64-row tile per expert on average amortizes the
            // larger tile and lets each warp reuse activations across columns.
            projectGrouped<64, 2>(c, routes, route, input.p, ranks.p, counts.p, source, pair.p, projected.p, act, reduced);
        } else {
            projectGrouped<32, 1>(c, routes, route, input.p, ranks.p, counts.p, source, pair.p, projected.p, act, reduced);
        }
        if (reduced) return;
        if (rows == 1) {
            if (decodeRoutes) {
                *decodeRoutes = std::move(projected);
                return;
            }
            auto combine = [&](auto *out) {
                using Out = std::remove_pointer_t<decltype(out)>;
                return cuda_launch::Launch<sumNativeRoutes<true, false, Out>>((c.hidden + 255) / 256, 256,
                    0, cudaStreamPerThread, projected.p, out, rows, c.hidden, c.topK, c.hidden, c.hidden,
                    static_cast<const int *>(nullptr));
            };
            if (!(rankSum ? combine(rankSum) : combine(dstRows)))
                throw std::runtime_error("Inkling FP4 route sum launch failed");
            return;
        }
        // Nonlocal routes have no down-projection output in a grouped plan.
        auto combine = [&](auto *out) {
            using Out = std::remove_pointer_t<decltype(out)>;
            const auto sum = localExperts && phase != ExpertPhase::Verify
                                 ? sumNativeRoutes<false, true, Out> : sumNativeRoutes<false, false, Out>;
            sum<<<(size_t(rows) * c.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                projected.p, out, rows, c.hidden, c.topK, c.hidden, c.hidden, route.ids);
        };
        if (rankSum) combine(rankSum);
        else combine(dstRows);
    }

public:
    GpuExperts(const std::map<std::string, TensorRange> &files, const std::string &name,
               const Config &c, ExpertLoadContext &topology, const Data &routerWeight)
        : RoutedExperts(files, name, c, ExpertDevice::Cuda, topology, routerWeight), firstExpert(topology.firstExpert), localExperts(topology.localExperts),
          weights(gate.nativeCodes.p ? 0 : 2 + 2 * (localExperts ? localExperts : c.experts)) {
#if CUDART_VERSION >= 12000
        if (FastllmCudaRuntimeArch() == 120 && c.experts == DecodeRouter::experts && c.topK == 8 && c.hidden % 256 == 0) {
            decodeRouter = Buffer<DecodeRouter>(1);
            check(cudaMemset(decodeRouter.p, 0, sizeof(DecodeRouter)));
            const size_t n = size_t(c.experts) * c.hidden;
            packedRouter = Buffer<uint4>(n / 8);
            packDecodeRouter<<<n / 256, 256, 0, cudaStreamPerThread>>>(static_cast<const float *>(routerWeight.cudaData),
                    reinterpret_cast<unsigned short *>(packedRouter.p), &decodeRouter.p->completed);
            unsigned lossy;
            check(cudaMemcpy(&lossy, &decodeRouter.p->completed, sizeof(lossy), cudaMemcpyDeviceToHost));
            if (lossy) { packedRouter = {}; decodeRouter = {}; }
        }
#endif
        if (gate.nativeCodes.p) {
            if (down.cols <= native::QuantThreads * 16) {
                if (!topology.situFactors.p) {
                    topology.situFactors = Buffer<float>(2 * 65536);
                    makeSiTUFactors<<<256, 256, 0, cudaStreamPerThread>>>(topology.situFactors.p, gateCap, upCap);
                    check(cudaGetLastError());
                }
                situFactors = topology.situFactors;
            }
            return;
        }
        for (int e = 0; e < (localExperts ? localExperts : c.experts); ++e) {
            weights[2 + 2 * e] = gate.resident[e].get();
            weights[3 + 2 * e] = down.resident[e].get();
        }
#ifdef FASTLLM_INKLING_GROUPED_SM75
        if (!localExperts && gate.exactHalfWeights && down.exactHalfWeights && c.experts == sm75::Experts && c.topK == 8) {
            sm75Blocks = sm75::Grouped<16>::Op::sufficient();
            sm75LargeBlocks = sm75::Grouped<128>::Op::sufficient();
            if (sm75Blocks && sm75LargeBlocks && GetCudaIndexedMoeSlots(weights.data(), weights.size(), sm75Weights)) return;
            sm75Blocks = 0;
        }
#endif
        if (!gate.fp4) groupedFp8 = PrepareCudaGroupedMxfp8Moe(weights.data(), weights.size());
        if (groupedFp8) {
#ifdef FASTLLM_INKLING_GROUPED_SM120
            nativeMxfp8 = FastllmCudaRuntimeArch() == 120 && c.experts == 128 && c.topK == 8 &&
                          c.intermediate % 128 == 0 && groupedFp8->MarlinWeights(false).codes &&
                          groupedFp8->MarlinWeights(true).codes;
#endif
            // Both packed projections are ready. There are no other users of these private weights.
            for (size_t i = 2; i < weights.size(); ++i) {
                weights[i]->FreeSpace();
                std::vector<float>().swap(weights[i]->scales);
            }
        } else if (!PrepareCudaMergeMOE(weights.data(), weights.size()))
            throw std::runtime_error("Failed to prepare Inkling MergeMOE weights: " + name);
    }
    ~GpuExperts() override { if (!gate.nativeCodes.p && !groupedFp8) ReleaseCudaMergeMOE(weights.data(), weights.size()); }

    Matrix run(const Config &c, const Matrix &x, const float *bias, ExpertPhase phase,
               Rows<__nv_bfloat16> *rankSum, Rows<__nv_bfloat16> *decodeRoutes, int promptRows) const override {
        const bool decodeArithmetic = phase == ExpertPhase::Verify;
        if (c.topK <= 0 || c.topK > 50 || c.topK > c.experts) throw std::runtime_error("Inkling expert routing geometry");
        const bool batchDots = decodeArithmetic && decodeRouter.p && x.rows >= smCount;
        Buffer<DecodeRouter> batchRouter(decodeArithmetic && !batchDots && x.rows > 1 && decodeRouter.p ? x.rows : 0);
        if (batchRouter.p)
            check(cudaMemsetAsync(batchRouter.p, 0, sizeof(DecodeRouter) * x.rows, cudaStreamPerThread));
        DecodeRouter *scratch = x.rows == 1 ? decodeRouter.p : batchRouter.p;
        Matrix router;
#if CUDART_VERSION >= 12000
        if (batchDots) {
            router = Matrix(x.rows, c.experts);
            auto project = [&](auto tokenTile, auto expertTile) {
                constexpr int tokens = decltype(tokenTile)::value, experts = decltype(expertTile)::value;
                for (int first = 0; first < x.rows;) {
                    const int rows = std::min(x.rows - first, tokens * 65535);
                    dim3 grid(c.experts / (4 * experts), (rows + tokens - 1) / tokens);
                    batchedRouterDots<tokens, experts><<<grid, 128, 0, cudaStreamPerThread>>>(
                        x.p + size_t(first) * x.cols, packedRouter.p, router.p + size_t(first) * c.experts, rows, x.cols);
                    first += rows;
                }
            };
            // Widen the tile only if the smaller grid can still fill the device.
            if ((int64_t(x.rows) + 7) / 8 * (c.experts / 32) >= smCount)
                project(std::integral_constant<int, 8>{}, std::integral_constant<int, 8>{});
            else
                project(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
        } else
#endif
        if (!scratch) router = routeLogits(x, routerWeight, promptRows);
        const bool marlin = !gate.nativeCodes.p && weights[2]->dataType == NVFP4_BLOCK_16_E4M3;
        const int chunk = (gate.nativeCodes.p ? 65536 : 65535) / c.topK;
        constexpr int groupedMxPrefillRows = 4096;
        // Marlin only needs rows routed to experts on this rank. Long prompts use
        // the block-scaled grouped kernel and keep its FP32 destination below.
        const int expertRows = promptRows && !gate.nativeCodes.p ? promptRows : x.rows;
        const bool shortPrefill = phase == ExpertPhase::Prefill && expertRows < groupedMxPrefillRows;
        const bool localMx = localExperts && groupedFp8 && (decodeArithmetic || shortPrefill) &&
                             expertRows >= smCount && expertRows <= chunk;
        const bool nativeDecode = gate.nativeCodes.p && x.rows == 1 && phase == ExpertPhase::Decode &&
                                  down.tensorShards.empty();
        if (!nativeDecode || localExperts) decodeRoutes = nullptr;
        bool directSum = localMx || nativeDecode ||
                         (phase != ExpertPhase::Verify && gate.nativeCodes.p && localExperts && down.tensorShards.empty());
#if defined(FASTLLM_INKLING_GROUPED_SM120)
        if (gate.nativeCodes.p && decodeArithmetic && gate.haveGroupedMaps && down.haveGroupedMaps &&
            native::useGroupedVerify(x.rows, c.topK, c.experts, c.hidden, c.intermediate)) {
            directSum = localExperts && down.tensorShards.empty() && x.rows <= chunk;
        }
#endif
        if (!directSum) rankSum = nullptr;
        if (rankSum) *rankSum = Rows<__nv_bfloat16>(x.rows, c.hidden);
        Matrix out(rankSum || decodeRoutes ? 0 : x.rows, c.hidden);
        int start = 0;
#if defined(FASTLLM_INKLING_GROUPED_SM75) || defined(FASTLLM_INKLING_GROUPED_SM120)
        if (phase == ExpertPhase::Prefill) {
            auto groupedPrefill = [&](int minRows, auto project) {
                while (std::min(expertRows, x.rows - start) >= minRows) {
                    const int rows = std::min({16384, expertRows, x.rows - start});
                    try {
                        (this->*project)(c, x.slice(start, rows), router.p + size_t(start) * c.experts,
                                         bias, out.p + size_t(start) * c.hidden);
                    } catch (const CudaWorkspaceExhausted &) {
                        cudaGetLastError();
                        break; // Retry the remaining rows through the smaller fallback below.
                    }
                    start += rows;
                }
            };
#ifdef FASTLLM_INKLING_GROUPED_SM75
            if (sm75Blocks) groupedPrefill(16, &GpuExperts::sm75Prefill);
#endif
#ifdef FASTLLM_INKLING_GROUPED_SM120
            // Amortize unpacking a projection over at least 4096 input tokens.
            if (nativeMxfp8) groupedPrefill(groupedMxPrefillRows, &GpuExperts::mxfp8Prefill);
#endif
        }
#endif
        const int step = std::min(chunk, expertRows);
        for (; start < x.rows; start += step) {
            const int rows = std::min(step, x.rows - start), routes = rows * c.topK;
            Buffer<int> indices(routes);
            Matrix scores(rows, c.topK);
            InputRow source{x.p + size_t(start) * c.hidden};
            RouteSelection route{scratch ? nullptr : router.p + size_t(start) * c.experts, bias, indices.p, scores.p,
                                 c.experts, c.topK, c.routeScale, firstExpert, localExperts,
                                 gate.nativeCodes.p || localMx ? -1 : 0};
            if (gate.nativeCodes.p) {
                runNative(c, source, rows, route, scratch ? scratch + start : nullptr,
                          out.p ? out.p + size_t(start) * c.hidden : nullptr, phase,
                          rankSum ? rankSum->p + size_t(start) * c.hidden : nullptr, decodeRoutes);
                continue;
            }
            const auto type = groupedFp8 ? BFLOAT16 : marlin ? FLOAT16 : FLOAT32;
            const int size = type == FLOAT32 ? sizeof(float) : sizeof(half);
            const int gateWidth = std::max(c.hidden, c.intermediate * 2);
            Buffer<uint8_t> gateBuffer(size_t(routes) * gateWidth * size);
            Buffer<uint8_t> middleBuffer(std::max(size_t(routes) * c.intermediate,
                                                   size_t(rows) * c.hidden) * size);
            CudaView gateOut(type, {routes, gateWidth}, gateBuffer.p);
            CudaView middle(type, {routes, c.intermediate}, middleBuffer.p);
            Buffer<uint8_t> inputBuffer(size_t(rows) * c.hidden * size);
            if (marlin) prepareRoutedInput<true>(source, reinterpret_cast<half *>(inputBuffer.p), rows, c.hidden, smCount, route, packedRouter.p, scratch);
            else if (groupedFp8) prepareRoutedInput<false>(source, reinterpret_cast<__nv_bfloat16 *>(inputBuffer.p), rows, c.hidden, smCount, route, packedRouter.p, scratch);
            else if (gate.fp4) prepareRoutedInput<true>(source, reinterpret_cast<float *>(inputBuffer.p), rows, c.hidden, smCount, route, packedRouter.p, scratch);
            else prepareRoutedInput<false>(source, reinterpret_cast<float *>(inputBuffer.p), rows, c.hidden, smCount, route, packedRouter.p, scratch);
            CudaView input(type, {rows, c.hidden}, inputBuffer.p);
            CudaView ids(INT32, {rows, c.topK}, indices.p);
            CudaView score(FLOAT32, ids.dims, scores.p);
            input.dataDeviceIds = weights[2]->dataDeviceIds;
            Buffer<int> workspace(groupedFp8 ? groupedFp8->WorkspaceInts(rows, c.topK) : 0);
            if (localMx) {
                groupedFp8->ProjectRoutes(input, ids, score, gateOut, middle, workspace.p, *this);
                // Marlin scatters only local route slots. Leave absent slots unread.
                auto combine = [&](auto *dest) {
                    sumNativeRoutes<false, true><<<(size_t(rows) * c.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                        reinterpret_cast<const __nv_bfloat16 *>(gateBuffer.p), dest, rows, c.hidden,
                        c.topK, c.hidden, c.hidden, indices.p);
                    check(cudaGetLastError());
                };
                if (rankSum) combine(rankSum->p + size_t(start) * c.hidden);
                else combine(out.p + size_t(start) * c.hidden);
                continue;
            }
            CudaView output(FLOAT32, input.dims, out.p + size_t(start) * c.hidden);
            output.dataDeviceIds = input.dataDeviceIds;
            DoCudaMergeMOE(input, output, ids, score, gateOut, middle,
                           const_cast<Data **>(weights.data()), weights.size(), *this, groupedFp8.get(), workspace.p);
        }
        return out;
    }
};
#include "hybrid-experts.cuh"
} // namespace
ExpertLayers loadExperts(const Config &c, const Placement &p, const std::map<std::string, TensorRange> &files,
                         const WeightMap &weights) {
    ExpertLoadContext topology;
    ExpertLayers experts;
    if (p.world > 1) {
        topology.localExperts = c.experts / p.world;
        topology.firstExpert = p.rank * topology.localExperts;
    }
    check(cudaDeviceGetAttribute(&topology.smCount, cudaDevAttrMultiProcessorCount, p.gpu));
    const int cudaLayers = std::count(p.experts.begin(), p.experts.end(), ExpertDevice::Cuda);
    if (cudaLayers) {
        if (FastllmCudaRuntimeArch() == 120 && c.hidden % 64 == 0 && c.intermediate % 64 == 0) {
            int image = 0;
            check(cudaMemcpyFromSymbol(&image, native::nativeImage, sizeof(image)));
            topology.nativeFp4 = image != 0;
        }
        topology.marlin = !topology.nativeFp4 && c.hidden % 128 == 0 && c.intermediate % 128 == 0 && c.topK <= 16 &&
                          FastllmCudaNVFP4E4M3GroupedMoeSupported(p.gpu);
    }
    if (!p.tensorDevices.empty()) {
        if (!topology.nativeFp4 || cudaLayers != c.layers)
            throw std::runtime_error("Inkling expert tensor parallelism requires native NVFP4 CUDA placement");
        for (int gpu : p.tensorDevices) {
            auto rank = std::make_shared<TensorRank>(gpu, p.gpu);
            rank->initialize();
            topology.ranks.push_back(std::move(rank));
        }
        fprintf(stderr, "[Inkling] NVFP4 expert gate/up and down projections sharded across %zu CUDA devices\n",
                topology.ranks.size());
    }
    int hostSlots = 0;
    std::unique_ptr<HybridCacheCycle> cache;
    std::shared_ptr<HybridGraphStorage> graphStorage;
    if (cudaLayers < c.layers) {
        if (std::find(p.experts.begin(), p.experts.end(), ExpertDevice::Numa) != p.experts.end())
            topology.enableNuma();
        cache = planHybridCache(c, p, files, hostSlots);
        if (hostSlots < c.experts) graphStorage = std::make_shared<HybridGraphStorage>();
    }
    for (int i = 0; i < c.layers; ++i) {
        auto mlp = "model.llm.layers." + std::to_string(i) + ".mlp.";
        const auto &router = weights.weight.at(mlp + "gate.weight");
        if (router.dataType != FLOAT32 || router.dims != std::vector<int>{c.experts, c.hidden})
            throw std::runtime_error("Inkling router shape or dtype");
        auto prefix = mlp + "experts.";
        if (p.experts[i] == ExpertDevice::Cuda)
            experts.layers.push_back(std::make_unique<GpuExperts>(files, prefix, c, topology, router));
        else
            experts.layers.push_back(std::make_unique<HostExperts>(files, prefix, c, p.experts[i], topology, router,
                                                                   hostSlots, cache.get(), graphStorage));
    }
    experts.cache = std::move(cache);
    return experts;
}
} // namespace inkling2
} // namespace fastllm
