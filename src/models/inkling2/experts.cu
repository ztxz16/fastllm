// Expert packing and activation quantization follow the original Inkling 2 Mini checkpoint.
// Reference for the NVFP4 reconstruction policy and the position of the routing weights:
// https://github.com/vllm-project/vllm/tree/fc1443152c0eb655f5c2762f56b540bebe93b196/vllm/models/inkling2/nvidia
#include "runtime.cuh"
#include "../../devices/cuda/moe/fastllm-nvfp4-routed-sm120.cuh"
#ifdef FASTLLM_INKLING_GROUPED_NVFP4_SM120
#include "../../devices/cuda/moe/fastllm-grouped-nvfp4-sm120.cuh"
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

#if defined(FASTLLM_INKLING_GROUPED_NVFP4_SM120)
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
            float q = 0;
            if constexpr (fp4) {
                q = chooseFp4(v, peak, outer).reconstructed;
            } else if (peak > 0) {
                float scale = exp2f(ceilf(log2f(peak / 448.f)));
                q = roundFp8(v / scale) * scale;
            }
            if constexpr (std::is_same_v<Out, half *>)
                out[size_t(row) * d + col] = __float2half_rz(q);
            else
                out[size_t(row) * d + col] = q;
        }
    }
}

template <bool fp4, typename Source, typename Out>
__global__ void reconstructExpertRows(Source src, Out *out, int d, bool cacheValues) {
    reconstructRow<fp4>(src, out, d, cacheValues, gridDim.y);
}

#if defined(FASTLLM_INKLING_GROUPED_NVFP4_SM120)
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
    const int route = blockIdx.x, expert = ids[route], rank = ranks[route], start = offsets[expert];
    const float *dots = buffers.accumulators + size_t(start + rank) * (2 * width);
    const float inputScale = input[size_t(route / topK) * (hidden / 16)].outer;
    GroupedSiTU act{dots, weightScales[expert], inputScale, scores[route], gateCap, upCap};
    GroupedFp4Output dst{native::GroupedInputRow(buffers, start, expert, rank, width / 16), midScales};
    reconstructRow<true>(act, dst, width, cache, gridDim.y);
}
#endif

__global__ void sumNativeRoutes(const __nv_bfloat16 *routes, float *out, int rows, int width, int top) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * width) return;
    const auto *src = routes + (i / width) * top * width + i % width;
    float sum = 0;
    for (int j = 0; j < top; ++j)
        sum += float(src[size_t(j) * width]);
    out[i] = sum;
}

struct RouteSelection {
    const float *logits, *bias;
    int *ids;
    float *scores;
    int experts, top;
    float scale;
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
    else
        selectRow<FixedExperts>(route);
}

template <int FixedExperts>
__global__ void selectRanked(RouteSelection route, int *ranks, int *counts) {
    selectRow<FixedExperts>(route);
    __syncthreads();
    if (threadIdx.x < route.top) {
        const int r = blockIdx.x * route.top + threadIdx.x;
        ranks[r] = atomicAdd(counts + route.ids[r], 1);
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
template <bool fp4, class Out, int ExpertTile>
__global__ void projectSelectAndQuantize(InputRow input, Out *out, int d, RouteSelection route,
                                         const uint4 *packed, DecodeRouter *scratch, unsigned tiles) {
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
    if (last && threadIdx.x < 32)
        cuda::SelectExpertFixedRow<false, DecodeRouter::experts, 8>(
            scratch->probabilities, route.bias, route.ids, route.scores, route.bias != nullptr, true, route.scale, 0);
}
#endif

template <bool fp4> int quantTiles(int rows, int d, int smCount) {
    if (d % (fp4 ? 16 : 32)) throw std::runtime_error("Inkling expert activation block alignment");
    // Replicate a row maximum only while there are idle SMs.
    if constexpr (fp4) return std::min(1 + (d - 1) / 512, 1 + (smCount - 1) / rows);
    return std::min(1 + (d - 1) / 256, 65534);
}

template <typename Source, typename Out> void quantizeFp4Into(Source src, Out *out, int rows, int d, int smCount) {
    const size_t bytes = size_t(d) * sizeof(float);
    const bool cache = !std::is_same_v<Source, InputRow> && bytes + 16 * sizeof(float) <= 48 * 1024;
    reconstructExpertRows<true>
        <<<dim3(rows, quantTiles<true>(rows, d, smCount)), 512, cache ? bytes : 0, cudaStreamPerThread>>>(
            src, out, d, cache);
}

template <typename Source, typename Out>
void quantizeInto(Source src, Out *out, int rows, int d, bool fp4, int smCount) {
    if (fp4) {
        quantizeFp4Into(src, out, rows, d, smCount);
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
            projectSelectAndQuantize<fp4, Out, experts>
                <<<dim3(rows, tiles + DecodeRouter::experts / experts), threads, 0, cudaStreamPerThread>>>(
                    src, out, d, route, packed, scratch, tiles);
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
    dim3 grid(rows, 1 + quantTiles<fp4>(rows, d, smCount));
    const auto kernel = route.top == 8 && route.experts == 128 ? selectAndQuantize<fp4,128,Out> :
                        route.top == 8 && route.experts == 256 ? selectAndQuantize<fp4,256,Out> :
                        selectAndQuantize<fp4,0,Out>;
    kernel<<<grid, fp4 ? 512 : 256, 0, cudaStreamPerThread>>>(src, out, d, route);
}

Matrix routeLogits(const Matrix &x, const Data &weight) {
    Matrix logits(x.rows, weight.dims[0]);
    Data input(FLOAT32, {x.rows, x.cols}, CUDA, x.p);
    Data output(FLOAT32, {logits.rows, logits.cols}, CUDA, logits.p);
    FastllmCudaBatchMatMulTransB(input, weight, output, 0, 0, 0, x.cols, x.cols,
                                 1, x.rows, x.cols, logits.cols, 1.f);
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
struct ExpertLoadContext {
    Buffer<float> situFactors;
    int nodes = 1, smCount = 0;
    bool marlin = false, nativeFp4 = false;
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
    int rows, cols, group;
    bool fp4;
    ExpertDevice device;
    std::vector<std::unique_ptr<Data>> resident;
    Buffer<uint4> nativeCodes;
    Buffer<unsigned> nativeScales;
    Buffer<float> nativeOuter;
#ifdef FASTLLM_INKLING_GROUPED_NVFP4_SM120
    native::GroupedWeightMaps groupedMaps{};
    bool haveGroupedMaps = false;
#endif
    native::Weights nativeView() const { return {nativeCodes.p, nativeScales.p, nativeOuter.p}; }
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
#ifdef FASTLLM_INKLING_GROUPED_NVFP4_SM120
            haveGroupedMaps = native::makeGroupedWeightMaps(groupedMaps, nativeView(), e, r, c);
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
            quantizeInto(source, out, rows, width, gate.fp4, smCount);
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
        : routerWeight(routerWeight), gate(files, name + "w13_weight", c.experts, 2 * c.intermediate, c.hidden, where, topology, true),
          down(files, name + "w2_weight", c.experts, c.hidden, c.intermediate, where, topology, false),
          smCount(topology.smCount), gateCap(c.gateCap), upCap(c.upCap) {
        if (gate.fp4 != down.fp4)
            throw std::runtime_error("Inkling expert projection types disagree");
    }
};
class GpuExperts final : public RoutedExperts {
    Buffer<float> situFactors;
    Buffer<DecodeRouter> decodeRouter;
    Buffer<uint4> packedRouter;
    std::vector<Data *> weights;
    std::unique_ptr<CudaGroupedMxfp8Moe> groupedFp8;

    template <int TileRows, int ColumnTiles>
    void projectGrouped(const Config &c, int routes, RouteSelection route, const native::Block16 *input,
                        const int *ranks, const int *counts, InputRow source,
                        __nv_bfloat16 *pair, __nv_bfloat16 *projected, RoutedSiTU<__nv_bfloat16> act) const {
        constexpr int warps = 8 / ColumnTiles, ctaColumns = 16 * warps * ColumnTiles;
        constexpr int threads = 32 * warps * ((TileRows + native::WarpTileRows - 1) / native::WarpTileRows);
        const int tiles = (routes + (TileRows - 1) * std::min(c.experts, routes)) / TileRows;
        Buffer<native::RouteGroup<TileRows>> tasks(tiles);
        Buffer<native::PlanSize> used(1);
        Buffer<float> scales(routes);
        Buffer<native::InputTile> packed(size_t(tiles) * (TileRows / 8) * (std::max(gate.cols, down.cols) / 64));
        const dim3 upGrid((gate.rows + ctaColumns - 1) / ctaColumns, tiles);
        if (ranks) {
            Buffer<int2> starts(c.experts);
            Buffer<float> inputScales(routes / c.topK);
            native::expertOffsets<TileRows><<<1, 32, 0, cudaStreamPerThread>>>(counts, starts.p, c.experts, used.p);
            native::scatterInput<TileRows><<<routes / c.topK, native::QuantThreads, 0, cudaStreamPerThread>>>(
                source.x, gate.cols, route.ids, ranks, counts, starts.p, c.topK, tasks.p, packed.p, inputScales.p);
            native::groupedProject<warps, ColumnTiles, true, TileRows, true><<<upGrid, threads, 0, cudaStreamPerThread>>>(
                gate.nativeView(), nullptr, inputScales.p, packed.p, tasks.p, used.p, pair, gate.rows, gate.cols, c.topK);
        } else {
            check(cudaMemsetAsync(used.p, 0, sizeof(native::PlanSize), cudaStreamPerThread));
            native::groupRoutes<<<c.experts, 256, 0, cudaStreamPerThread>>>(route.ids, routes, tasks.p, used.p);
            native::packInput<<<dim3((gate.cols / 64 + 7) / 8, tiles, TileRows / 8), 256, 0, cudaStreamPerThread>>>(
                tasks.p, used.p, input, packed.p, gate.cols, c.topK);
            native::groupedProject<warps, ColumnTiles, true><<<upGrid, threads, 0, cudaStreamPerThread>>>(
                gate.nativeView(), input, nullptr, packed.p, tasks.p, used.p, pair, gate.rows, gate.cols, c.topK);
        }
        if (situFactors.p && routes >= c.experts) {
            const int threads = ((down.cols / 16 + 31) / 32) * 32;
            native::activateBlocks<TileRows><<<tiles * TileRows, threads, 0, cudaStreamPerThread>>>(
                PackedSiTU{pair, route.scores, situFactors.p}, tasks.p, used.p, down.cols, packed.p, scales.p);
        } else {
            native::launchPackedActivation(act, tasks.p, used.p, tiles, down.cols, packed.p, scales.p);
        }
        native::groupedProject<warps, ColumnTiles, false>
            <<<dim3((down.rows + ctaColumns - 1) / ctaColumns, tiles), threads, 0, cudaStreamPerThread>>>(
                down.nativeView(), nullptr, scales.p, packed.p, tasks.p, used.p, projected, down.rows, down.cols, 1);
    }

#if defined(FASTLLM_INKLING_GROUPED_NVFP4_SM120)
    void groupedVerify(const Config &c, int rows, RouteSelection route, const native::Block16 *input, float *dst) const {
        const int *ids = route.ids;
        const int routes = rows * c.topK, width = std::max(gate.cols, down.cols);
        Buffer<int> offsets(c.experts + 1), rank(routes);
        check(native::planGroupedRows(ids, offsets.p, rank.p, c.experts, routes, cudaStreamPerThread));
        Buffer<uint2> codes(size_t(routes) * (width / 16));
        Buffer<uint8_t> scales(native::groupedScaleRows(routes, c.experts) * (width / 16));
        Buffer<float> sorted(size_t(routes) * std::max(gate.rows, down.rows));
        const auto space = native::groupedScratch(c.experts, smCount);
        Buffer<uint8_t> metadata(space.metadata), workspace(space.gemm);
        native::GroupedBuffers buffers{codes.p, scales.p, sorted.p, metadata.p, workspace.p};
        check(native::packGroupedInput(input, ids, offsets.p, rank.p, buffers, routes, gate.cols, c.topK, cudaStreamPerThread));
        auto project = [&](const Projection &w) {
            check(native::projectGroupedRows(w.nativeView(), w.groupedMaps, offsets.p, buffers, w.rows, w.cols,
                                             c.experts, smCount, cudaStreamPerThread));
        };
        project(gate);
        Buffer<float> midScales(routes);
        const size_t bytes = size_t(c.intermediate) * sizeof(float);
        const bool cache = bytes + 16 * sizeof(float) <= 48 * 1024;
        quantizeGroupedSiTU<<<dim3(routes, quantTiles<true>(routes, c.intermediate, smCount)), 512,
                              cache ? bytes : 0, cudaStreamPerThread>>>(
            buffers, input, gate.nativeOuter.p, route.scores, ids, offsets.p, rank.p, midScales.p,
            c.hidden, c.intermediate, c.topK, gateCap, upCap, cache);
        project(down);
        check(native::sumGroupedRows(sorted.p, dst, ids, offsets.p, rank.p, down.nativeOuter.p, midScales.p,
                                     rows, c.hidden, c.topK, cudaStreamPerThread));
    }
#endif

    // The routing score is applied inside SiTU before the second FP4 quantization.
    void runNative(const Config &c, InputRow source, int rows, RouteSelection route, DecodeRouter *scratch,
                   float *dstRows, bool decodeArithmetic) const {
        const int routes = rows * c.topK;
        // Dense routing amortizes the expert prefix pass. Each scatter lane
        // keeps at most two FP4 blocks in registers until its row scale is known.
        const bool ranked = !decodeArithmetic && routes / c.experts >= 32 && c.hidden <= native::QuantThreads * 2 * 16;
        Buffer<native::Block16> input(ranked ? 0 : size_t(rows) * c.hidden / 16);
        Buffer<int> ranks(ranked ? routes : 0), counts(ranked ? c.experts : 0);
        if (ranked) {
            check(cudaMemsetAsync(counts.p, 0, size_t(c.experts) * sizeof(int), cudaStreamPerThread));
            const auto select = c.topK == 8 && c.experts == 128 ? selectRanked<128> :
                                c.topK == 8 && c.experts == 256 ? selectRanked<256> : selectRanked<0>;
            const int threads = c.topK == 8 && (c.experts == 128 || c.experts == 256) ? 32 : 64;
            select<<<rows, threads, 0, cudaStreamPerThread>>>(route, ranks.p, counts.p);
        } else {
            prepareRoutedInput<true>(source, input.p, rows, c.hidden, smCount, route, packedRouter.p, scratch);
        }
#if defined(FASTLLM_INKLING_GROUPED_NVFP4_SM120)
        if (decodeArithmetic && gate.haveGroupedMaps && down.haveGroupedMaps &&
            native::useGroupedVerify(rows, c.topK, c.experts, c.hidden, c.intermediate)) {
            groupedVerify(c, rows, route, input.p, dstRows);
            return;
        }
#endif
        Rows<__nv_bfloat16> pair(routes, 2 * c.intermediate), projected(routes, c.hidden);
        RoutedSiTU<__nv_bfloat16> act{pair.p, route.scores, gateCap, upCap};
        if (rows == 1 || decodeArithmetic) {
            Buffer<native::Block16> mid(size_t(routes) * c.intermediate / 16);
            const int tileRows = native::decodeRouteTileRows(rows, c.topK);
            auto project = [&](const Projection &w, const native::Block16 *x, int divisor, __nv_bfloat16 *y) {
                if (tileRows) {
                    const int capacity = tileRows * c.topK, tiles = (rows + tileRows - 1) / tileRows;
                    native::decodeProject<4, true><<<dim3(w.rows / 16, capacity, tiles), 128, 0, cudaStreamPerThread>>>(
                        w.nativeView(), x, route.ids, y, w.rows, w.cols, divisor, routes, capacity);
                    return;
                }
                native::decodeProject<4><<<dim3(w.rows / 16, routes), 128, 0, cudaStreamPerThread>>>(
                    w.nativeView(), x, route.ids, y, w.rows, w.cols, divisor);
            };
            project(gate, input.p, c.topK, pair.p);
            quantizeFp4Into(act, mid.p, routes, c.intermediate, smCount);
            project(down, mid.p, 1, projected.p);
        } else if (routes / c.experts >= 64) {
            // At least one 64-row tile per expert on average amortizes the
            // larger tile and lets each warp reuse activations across columns.
            projectGrouped<64, 2>(c, routes, route, input.p, ranks.p, counts.p, source, pair.p, projected.p, act);
        } else {
            projectGrouped<32, 1>(c, routes, route, input.p, ranks.p, counts.p, source, pair.p, projected.p, act);
        }
        sumNativeRoutes<<<(size_t(rows) * c.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            projected.p, dstRows, rows, c.hidden, c.topK);
    }

public:
    GpuExperts(const std::map<std::string, TensorRange> &files, const std::string &name,
               const Config &c, ExpertLoadContext &topology, const Data &routerWeight)
        : RoutedExperts(files, name, c, ExpertDevice::Cuda, topology, routerWeight), weights(gate.nativeCodes.p ? 0 : 2 + 2 * c.experts) {
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
        for (int e = 0; e < c.experts; ++e) {
            weights[2 + 2 * e] = gate.resident[e].get();
            weights[3 + 2 * e] = down.resident[e].get();
        }
        if (!gate.fp4) groupedFp8 = PrepareCudaGroupedMxfp8Moe(weights.data(), weights.size());
        if (groupedFp8) {
            // Both packed projections are ready. There are no other users of these private weights.
            for (size_t i = 2; i < weights.size(); ++i) {
                weights[i]->FreeSpace();
                std::vector<float>().swap(weights[i]->scales);
            }
        } else if (!PrepareCudaMergeMOE(weights.data(), weights.size()))
            throw std::runtime_error("Failed to prepare Inkling MergeMOE weights: " + name);
    }
    ~GpuExperts() override { if (!gate.nativeCodes.p && !groupedFp8) ReleaseCudaMergeMOE(weights.data(), weights.size()); }

    Matrix run(const Config &c, const Matrix &x, const float *bias, ExpertPhase phase) const override {
        const bool decodeArithmetic = phase == ExpertPhase::Verify;
        if (c.topK <= 0 || c.topK > 50 || c.topK > c.experts) throw std::runtime_error("Inkling expert routing geometry");
        Buffer<DecodeRouter> batchRouter(decodeArithmetic && x.rows > 1 && decodeRouter.p
                                        ? x.rows : 0);
        if (batchRouter.p)
            check(cudaMemsetAsync(batchRouter.p, 0, sizeof(DecodeRouter) * x.rows,
                                  cudaStreamPerThread));
        DecodeRouter *scratch = x.rows == 1 ? decodeRouter.p : batchRouter.p;
        Matrix router = scratch ? Matrix{} : routeLogits(x, routerWeight);
        Matrix out(x.rows, c.hidden);
        const bool marlin = !gate.nativeCodes.p && weights[2]->dataType == NVFP4_BLOCK_16_E4M3;
        // Indexed GEMV launches one grid.y entry per route. The native
        // prefill grids index tiles, so a whole 64K-route scratch chunk fits.
        const int chunk = (gate.nativeCodes.p ? 65536 : 65535) / c.topK;
        for (int start = 0; start < x.rows; start += chunk) {
            const int rows = std::min(chunk, x.rows - start), routes = rows * c.topK;
            Buffer<int> indices(routes);
            Matrix scores(rows, c.topK);
            InputRow source{x.p + size_t(start) * c.hidden};
            RouteSelection route{scratch ? nullptr : router.p + size_t(start) * c.experts, bias, indices.p, scores.p,
                                 c.experts, c.topK, c.routeScale};
            if (gate.nativeCodes.p) {
                runNative(c, source, rows, route, scratch ? scratch + start : nullptr, out.p + size_t(start) * c.hidden, decodeArithmetic);
                continue;
            }
            const auto type = groupedFp8 ? BFLOAT16 : marlin ? FLOAT16 : FLOAT32;
            const int size = type == FLOAT32 ? sizeof(float) : sizeof(half);
            const int gateWidth = std::max(c.hidden, c.intermediate * 2);
            Buffer<uint8_t> gateBuffer(size_t(routes) * gateWidth * size);
            Buffer<uint8_t> middleBuffer(std::max(size_t(routes) * c.intermediate,
                                                   size_t(rows) * c.hidden) * size);
            Data gateOut(type, {routes, gateWidth}, CUDA, gateBuffer.p);
            Data middle(type, {routes, c.intermediate}, CUDA, middleBuffer.p);
            Buffer<uint8_t> inputBuffer(size_t(rows) * c.hidden * size);
            if (marlin) prepareRoutedInput<true>(source, reinterpret_cast<half *>(inputBuffer.p), rows, c.hidden, smCount, route, packedRouter.p, scratch);
            else if (groupedFp8) prepareRoutedInput<false>(source, reinterpret_cast<__nv_bfloat16 *>(inputBuffer.p), rows, c.hidden, smCount, route, packedRouter.p, scratch);
            else if (gate.fp4) prepareRoutedInput<true>(source, reinterpret_cast<float *>(inputBuffer.p), rows, c.hidden, smCount, route, packedRouter.p, scratch);
            else prepareRoutedInput<false>(source, reinterpret_cast<float *>(inputBuffer.p), rows, c.hidden, smCount, route, packedRouter.p, scratch);
            Data input(type, {rows, c.hidden}, CUDA, inputBuffer.p);
            Data output(FLOAT32, input.dims, CUDA, out.p + size_t(start) * c.hidden);
            Data ids(INT32, {rows, c.topK}, CUDA, indices.p);
            Data score(FLOAT32, ids.dims, CUDA, scores.p);
            input.dataDeviceIds = output.dataDeviceIds = weights[2]->dataDeviceIds;
            Buffer<int> workspace(groupedFp8 ? groupedFp8->WorkspaceInts(rows, c.topK) : 0);
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
