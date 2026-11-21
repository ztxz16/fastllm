// Expert packing and activation quantization follow the original Inkling 2 Mini checkpoint.
// Reference for the NVFP4 reconstruction policy and the position of the routing weights:
// https://github.com/vllm-project/vllm/tree/fc1443152c0eb655f5c2762f56b540bebe93b196/vllm/models/inkling2/nvidia
#include "runtime.cuh"
#include "../../devices/cuda/moe/fastllm-nvfp4-routed-sm120.cuh"
#include "devices/cpu/computeutils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-activation.h"
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
    __device__ float get(size_t row, int col, int d) const {
        const T *gate = pair + row * (2 * d) + (Interleaved ? 2 * col : col);
        float g = float(__float2bfloat16_rn(float(gate[0])));
        float u = float(__float2bfloat16_rn(float(gate[Interleaved ? 1 : d])));
        return float(__float2bfloat16_rn(FastllmSiTUAndMul(g, u, gateCap, upCap) * scores[row]));
    }
};

template <bool fp4, typename Source, typename Out>
__device__ void reconstructRow(Source src, Out *out, int d, bool cacheValues, unsigned tiles) {
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
        if constexpr (std::is_same_v<Out, native::Block16>) {
#if CUDART_VERSION >= 11080
            const Fp4Choice q = chooseFp4(v, peak, outer);
            // E2M1 magnitudes from one through six occupy consecutive half-exponent steps.
            const unsigned mag = __float_as_uint(fabsf(q.value)) >> 22;
            const unsigned nibble = (fabsf(q.value) < 1.f ? unsigned(q.value != 0.f) : mag - 252u) |
                                    ((__float_as_uint(q.value) >> 28) & 8u);
            unsigned word = nibble << (4 * (col % 8));
            for (int delta = 1; delta < 8; delta *= 2)
                word |= __shfl_xor_sync(__activemask(), word, delta, 8);
            auto &block = out[(size_t(row) * d + col) / 16];
            if (!(col % 8)) block.code[(col % 16) / 8] = word;
            if (!(col % 16)) {
                block.scale = __nv_fp8_e4m3(q.scale).__x;
                block.outer = outer;
            }
#endif
        } else {
            float q = 0;
            if constexpr (fp4) {
                q = chooseFp4(v, peak, outer).reconstructed;
            } else if (peak > 0) {
                float scale = exp2f(ceilf(log2f(peak / 448.f)));
                q = roundFp8(v / scale) * scale;
            }
            if constexpr (std::is_same_v<Out, half>)
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

// Each token has one independent routing CTA alongside its quantization tiles.
// All consumers run afterwards on the same stream; no inter-block waits.
template <bool fp4, int FixedExperts, typename Out>
__global__ void selectAndQuantize(InputRow input, Out *out, int d, RouteSelection route) {
    const unsigned tiles = gridDim.y - 1;
    if (blockIdx.y != tiles) {
        reconstructRow<fp4>(input, out, d, false, tiles);
    } else if constexpr (FixedExperts) {
        if (threadIdx.x < 32)
            cuda::SelectExpertFixedRow<false, FixedExperts, 8, true>(
                route.logits, route.bias, route.ids, route.scores, route.bias != nullptr, true, route.scale);
    } else {
        cuda::SelectExpertRow<64, 50, true>(route.logits, route.bias, route.ids, route.scores,
            route.experts, route.top, route.bias != nullptr, true, route.scale);
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
            scratch->probabilities, route.bias, route.ids, route.scores, route.bias != nullptr, true, route.scale);
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
        unsigned tiles = quantTiles<fp4>(1, d, smCount);
        constexpr int expertTile = 2;
        projectSelectAndQuantize<fp4, Out, expertTile><<<dim3(1, tiles + DecodeRouter::experts / expertTile),
            fp4 ? 512 : 256, 0, cudaStreamPerThread>>>(src, out, d, route,
                    packed, scratch, tiles);
        return;
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
struct HostTopology {
    int nodes = 1, smCount = 0;
    bool marlin = false, nativeFp4 = false;
    void enable() {
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
    float *scales = nullptr;
};
struct Projection {
    int rows, cols, group;
    bool fp4;
    ExpertDevice device;
    std::vector<std::unique_ptr<Data>> resident;
    Buffer<uint4> nativeCodes;
    Buffer<unsigned> nativeScales;
    Buffer<float> nativeOuter;
    native::Weights nativeView() const { return {nativeCodes.p, nativeScales.p, nativeOuter.p}; }
    std::vector<HostShard> shards;
    std::vector<Packed> host;
    Projection(const std::map<std::string, TensorRange> &files, const std::string &name, int e, int r, int c,
               ExpertDevice where, HostTopology &topology, bool gateUp)
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
                size_t offset = (per * bytes + 63) & ~size_t(63),
                       total = offset + per * groups * sizeof(float);
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
                s.scales = reinterpret_cast<float *>(s.values + offset);
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
                auto w = s.values + (id / nodes) * bytes;
                auto a = s.scales + (id / nodes) * groups;
                std::memcpy(w, raw.data() + size_t(id) * bytes, bytes);
                std::memcpy(a, decoded.data(), groups * sizeof(float));
                host[id] = {w, a};
            }
        }
    }
};
struct HostRows : MultiThreadBaseOp {
    const Projection &w;
    const std::vector<int> &ids;
    float *input;
    uint16_t *bf16Input;
    float *output;
    int top, node, nodes, worker, workers;
    bool gate;

    HostRows(const Projection &p, const std::vector<int> &routes, float *x, uint16_t *bx, float *y,
             int k, int n, int nn, int thread, int count, bool up)
        : w(p), ids(routes), input(x), bf16Input(bx), output(y), top(k), node(n), nodes(nn),
          worker(thread), workers(count), gate(up) {}

    void Run() override {
        int first = w.rows * worker / workers, end = w.rows * (worker + 1) / workers;
        for (size_t lane = 0; lane < ids.size(); ++lane) {
            int e = ids[lane];
            if (e % nodes != node) continue;
            const auto &p = w.host[e];
            size_t offset = (gate ? lane / top : lane) * w.cols;
            auto *y = output + lane * w.rows;
            if (w.fp4)
                MultiThreadLinearFloat32NVFP4Op(input + offset, p.data, nullptr, y,
                    1, w.cols, w.rows, first, end, p.scales, nullptr, 1, w.group).Run();
            else
                MultiThreadLinearBFloat16FP8E4M3Op(bf16Input + offset, p.data, nullptr, y,
                    1, w.cols, w.rows, first, end, p.scales, 1, w.group).Run();
        }
    }
};

class RoutedExperts : public ExpertBackend {
protected:
    const Data &routerWeight;
    Projection gate, down;
    const int smCount;
public:
    RoutedExperts(const std::map<std::string, TensorRange> &files, const std::string &name, const Config &c,
                  ExpertDevice where, HostTopology &topology, const Data &routerWeight)
        : routerWeight(routerWeight), gate(files, name + "w13_weight", c.experts, 2 * c.intermediate, c.hidden, where, topology, true),
          down(files, name + "w2_weight", c.experts, c.hidden, c.intermediate, where, topology, false),
          smCount(topology.smCount) {
        if (gate.fp4 != down.fp4)
            throw std::runtime_error("Inkling expert projection types disagree");
    }

};
class GpuExperts final : public RoutedExperts, private CudaMoeActivation {
    Buffer<DecodeRouter> decodeRouter;
    Buffer<uint4> packedRouter;
    std::vector<Data *> weights;
    std::unique_ptr<CudaGroupedMxfp8Moe> groupedFp8;
    float gateCap, upCap;

    void Run(const Data &pair, const Data &scores, Data &mid, int rows, int width) const override {
        auto launch = [&](const auto *pairData, auto *out) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(pairData)>>;
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

    // The routing score is applied inside SiTU before the second FP4 quantization.
    void runNative(const Config &c, InputRow source, int rows, RouteSelection route, DecodeRouter *scratch,
                   float *dstRows) const {
        const int routes = rows * c.topK;
        Buffer<native::Block16> input(size_t(rows) * c.hidden / 16), mid(size_t(routes) * c.intermediate / 16);
        Rows<__nv_bfloat16> pair(routes, 2 * c.intermediate), projected(routes, c.hidden);
        prepareRoutedInput<true>(source, input.p, rows, c.hidden, smCount, route, packedRouter.p, scratch);
        const bool grouped = rows > 1;
        const int tiles = grouped
                              ? (routes + (native::RouteTile - 1) * std::min(c.experts, routes)) / native::RouteTile
                              : routes;
        Buffer<native::Tile> tasks(grouped ? tiles : 0);
        Buffer<int> used(grouped ? 1 : 0);
        if (grouped) {
            check(cudaMemsetAsync(used.p, 0, sizeof(int), cudaStreamPerThread));
            native::groupRoutes<<<c.experts, 256, 0, cudaStreamPerThread>>>(route.ids, routes, tasks.p, used.p);
        }
        Buffer<native::InputTile> packed(
            grouped ? size_t(tiles) * (native::RouteTile / 8) * (std::max(c.hidden, c.intermediate) / 64) : 0);
        auto project = [&](const Projection &w, const native::Block16 *act, int divisor, __nv_bfloat16 *dst) {
            if (grouped) {
                native::packInput<<<dim3((w.cols / 64 + 7) / 8, tiles, native::RouteTile / 8), 256, 0,
                                    cudaStreamPerThread>>>(tasks.p, used.p, act, packed.p, w.cols, divisor);
                native::groupedProject<8><<<dim3((w.rows / 16 + 7) / 8, tiles), 256, 0, cudaStreamPerThread>>>(
                    w.nativeView(), act, packed.p, tasks.p, used.p, dst, w.rows, w.cols, divisor);
            } else {
                native::decodeProject<4><<<dim3(w.rows / 16, tiles), 128, 0, cudaStreamPerThread>>>(
                    w.nativeView(), act, route.ids, dst, w.rows, w.cols, divisor);
            }
        };
        project(gate, input.p, c.topK, pair.p);
        RoutedSiTU<__nv_bfloat16> act{pair.p, route.scores, gateCap, upCap};
        quantizeFp4Into(act, mid.p, routes, c.intermediate, smCount);
        project(down, mid.p, 1, projected.p);
        sumNativeRoutes<<<(size_t(rows) * c.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            projected.p, dstRows, rows, c.hidden, c.topK);
    }

public:
    GpuExperts(const std::map<std::string, TensorRange> &files, const std::string &name,
               const Config &c, ExpertDevice where, HostTopology &topology, const Data &routerWeight)
        : RoutedExperts(files, name, c, where, topology, routerWeight), weights(gate.nativeCodes.p ? 0 : 2 + 2 * c.experts),
          gateCap(c.gateCap), upCap(c.upCap) {
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
        if (gate.nativeCodes.p) return;
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

    Matrix run(const Config &c, const Matrix &x, const float *bias) const override {
        if (c.topK <= 0 || c.topK > 50 || c.topK > c.experts) throw std::runtime_error("Inkling expert routing geometry");
        DecodeRouter *scratch = x.rows == 1 ? decodeRouter.p : nullptr;
        Matrix router = scratch ? Matrix{} : routeLogits(x, routerWeight);
        Matrix out(x.rows, c.hidden);
        const bool marlin = !gate.nativeCodes.p && weights[2]->dataType == NVFP4_BLOCK_16_E4M3;
        // Indexed GEMV has a 16-bit CUDA grid.y. Chunk independent tokens at
        // the device limit without changing their routing or reduction order.
        const int chunk = 65535 / c.topK;
        for (int start = 0; start < x.rows; start += chunk) {
            const int rows = std::min(chunk, x.rows - start), routes = rows * c.topK;
            Buffer<int> indices(routes);
            Matrix scores(rows, c.topK);
            InputRow source{x.p + size_t(start) * c.hidden};
            RouteSelection route{scratch ? nullptr : router.p + size_t(start) * c.experts, bias, indices.p, scores.p,
                                 c.experts, c.topK, c.routeScale};
            if (gate.nativeCodes.p) {
                runNative(c, source, rows, route, scratch, out.p + size_t(start) * c.hidden);
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
class HostExperts final : public RoutedExperts {
    void select(const Config &c, const Matrix &router, const float *bias, const Buffer<int> &ids,
                const Matrix &scores) const {
        Data logits(FLOAT32, {router.rows, c.experts}, CUDA, router.p);
        Data biasView(FLOAT32, {c.experts}, CUDA, const_cast<float *>(bias));
        Data indices(INT32, {router.rows, c.topK}, CUDA, ids.p);
        Data scales(FLOAT32, indices.dims, CUDA, scores.p);
        if (!FastllmCudaFusedSigmoidSelectExpert(logits, &biasView, indices, scales, c.topK, true, c.routeScale)) {
            Matrix probabilities(router.rows, c.experts);
            Data probs(FLOAT32, logits.dims, CUDA, probabilities.p);
            if (!FastllmCudaSigmoid(logits, probs) ||
                !FastllmCudaSelectExpert(probs, &biasView, indices, scales, c.topK, true, c.routeScale))
                throw std::runtime_error("Inkling expert routing geometry");
        }
    }

public:
    using RoutedExperts::RoutedExperts;

    Matrix run(const Config &c, const Matrix &x, const float *bias) const override {
        const int rows = x.rows, top = c.topK;
        Buffer<int> indices(size_t(rows) * top);
        Matrix scores(rows, top);
        select(c, routeLogits(x, routerWeight), bias, indices, scores);
        std::vector<int> hostIds(size_t(rows) * top);
        check(cudaMemcpy(hostIds.data(), indices.p, hostIds.size() * sizeof(int), cudaMemcpyDeviceToHost));
        auto project = [&](const Projection &w, auto source, bool up) {
            Matrix input(up ? rows : rows * top, w.cols);
            quantizeInto(source, input.p, input.rows, input.cols, w.fp4, smCount);
            Matrix out(up ? rows * top : rows, w.rows);
            std::vector<float> hostIn(size_t(input.rows) * input.cols), hostOut(size_t(rows) * top * w.rows);
            check(cudaMemcpy(hostIn.data(), input.p, hostIn.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<uint16_t> bf16Input;
            if (!w.fp4) {
                bf16Input.resize(hostIn.size());
                std::transform(hostIn.begin(), hostIn.end(), bf16Input.begin(), Float32ToBFloat16RNEBits);
            }
            auto *pool = GetAlivePool();
            const auto interval = pool->curActivateThreadInterval;
            std::vector<std::vector<int>> groups;
#ifdef USE_NUMAS
            if (w.device == ExpertDevice::Numa)
                groups = GetNumasExpertWorkerIds();
#endif
            if (groups.empty() ||
                std::any_of(groups.begin(), groups.end(), [](const auto &group) { return group.empty(); })) {
                // With fewer workers than NUMA nodes, the active workers must cover every expert.
                groups.assign(1, {});
                for (int id = interval.first; id < interval.second; ++id)
                    groups[0].push_back(id);
            }
            if (groups.front().empty())
                throw std::runtime_error("Inkling host experts need an active CPU worker");
            std::vector<HostRows> tasks;
            tasks.reserve(interval.second - interval.first);
            for (size_t node = 0; node < groups.size(); ++node) {
                const auto &group = groups[node];
                for (size_t pos = 0; pos < group.size(); ++pos) {
                    tasks.emplace_back(w, hostIds, hostIn.data(), bf16Input.data(), hostOut.data(), top,
                                       node, groups.size(), pos, group.size(), up);
                    pool->PushOp(group[pos], &tasks.back());
                }
            }
            for (const auto &group : groups)
                for (int id : group)
                    pool->Wait(id);
            if (!up) {
                for (int row = 0; row < rows; ++row) {
                    float *dst = hostOut.data() + size_t(row) * w.rows;
                    const float *first = hostOut.data() + size_t(row) * top * w.rows;
                    for (int col = 0; col < w.rows; ++col) {
                        float sum = first[col];
                        for (int lane = 1; lane < top; ++lane) sum += first[size_t(lane) * w.rows + col];
                        dst[col] = sum;
                    }
                }
            }
            check(cudaMemcpy(out.p, hostOut.data(), size_t(out.rows) * out.cols * sizeof(float), cudaMemcpyHostToDevice));
            return out;
        };
        auto pair = project(gate, InputRow{x.p}, true);
        RoutedSiTU<float, true> activation{pair.p, scores.p, c.gateCap, c.upCap};
        return project(down, activation, false);
    }
};
} // namespace
std::vector<std::unique_ptr<ExpertBackend>> loadExperts(const Config &c, const Placement &p,
                                                        const std::map<std::string, TensorRange> &files, const WeightMap &weights) {
    HostTopology topology;
    check(cudaDeviceGetAttribute(&topology.smCount, cudaDevAttrMultiProcessorCount, p.gpu));
    if (FastllmCudaRuntimeArch() == 120 && c.hidden % 64 == 0 && c.intermediate % 64 == 0) {
        int image = 0;
        check(cudaMemcpyFromSymbol(&image, native::nativeImage, sizeof(image)));
        topology.nativeFp4 = image != 0;
    }
    if (std::find(p.experts.begin(), p.experts.end(), ExpertDevice::Cuda) != p.experts.end())
        topology.marlin = !topology.nativeFp4 && c.hidden % 128 == 0 && c.intermediate % 128 == 0 && c.topK <= 16 &&
                          FastllmCudaNVFP4E4M3GroupedMoeSupported(p.gpu);
    if (std::find(p.experts.begin(), p.experts.end(), ExpertDevice::Numa) != p.experts.end())
        topology.enable();
    std::vector<std::unique_ptr<ExpertBackend>> layers;
    for (int i = 0; i < c.layers; ++i) {
        auto mlp = "model.llm.layers." + std::to_string(i) + ".mlp.";
        const auto &router = weights.weight.at(mlp + "gate.weight");
        if (router.dataType != FLOAT32 || router.dims != std::vector<int>{c.experts, c.hidden})
            throw std::runtime_error("Inkling router shape or dtype");
        auto prefix = mlp + "experts.";
        if (p.experts[i] == ExpertDevice::Cuda)
            layers.push_back(std::make_unique<GpuExperts>(files, prefix, c, p.experts[i], topology, router));
        else
            layers.push_back(std::make_unique<HostExperts>(files, prefix, c, p.experts[i], topology, router));
    }
    return layers;
}
} // namespace inkling2
} // namespace fastllm
