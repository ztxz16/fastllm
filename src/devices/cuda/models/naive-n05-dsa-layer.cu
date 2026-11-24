#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-dsa-layer.cuh"
#include "devices/cuda/naive-n05-kv.cuh"
#include "naive-n05-dsa-fused-attention.cuh"
#include "naive-n05-mxfp8-projection.cuh"
#include "naivert/attn_dsa/body.cuh"
#include "naivert/attn_dsa/projection.cuh"
#include "../../multicuda/fastllm-swa-output.cuh"
#include <unordered_map>
#include <cmath>

namespace fastllm_dsa_layer {
using B = __nv_bfloat16;
using Args = FastllmNaiveDSALayerArgs;
using Layout = FastllmNaiveDSALayerLayout;
constexpr int Threads = 384;
__device__ __forceinline__ float Round(float x) {
    return float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float WarpSum(float x) {
    for (int step = 16; step; step >>= 1)
        x += __shfl_down_sync(0xffffffff, x, step);
    return __shfl_sync(0xffffffff, x, 0);
}
// Vectorized global weight loads keep shared storage below the attention
// workspace size, leaving more L1 capacity than staging a complete weight tile.
template <int Rows> struct ProjectionStorage {
    B input[Rows * 4096];
    float partial[12 * 128 * (Rows / 8)];
};
template <int Rows> struct TailStorage {
    B input[Rows * 1024], local[Rows][32];
    float partial[12 * 128 * (Rows / 8)];
    uint32_t received[8][8][16], step[8];
};
template <bool Fp8Key, int Rows = 8, bool PairedAttention = false> union Storage {
    ProjectionStorage<Rows> projection;
    typename fastllm_dsa_fused::Body<Fp8Key, false>::SharedStorage core;
    TailStorage<Rows> tail;
    fastllm_dsa_fused::native_attention::StorageFor<(PairedAttention ? 256 : 128)> attention;
};

// A request-local Indexer needs no attention partials until selection has
// finished. Two 16K scratch views fit in the original 64K single-request arena.
struct RequestScratch {
    using Base = fastllm_dsa_fused::AttnDsaScratchLayout;
    static constexpr int kQueryRows = Base::kQueryRows, kIndexHeads = Base::kIndexHeads;
    static constexpr int kIndexDim = Base::kIndexDim, kIndexTopK = Base::kIndexTopK;
    static constexpr int kMaxSeqLen = 16384, kAttentionControlFloatOffset = 0;
    static constexpr size_t kIqPackedOffset = Base::kIqPackedOffset;
    static constexpr size_t kIndexWeightsScaledOffset = Base::kIndexWeightsScaledOffset;
    static constexpr size_t kIndexLogitsOffset = Base::kIndexLogitsOffset;
    static constexpr size_t kIndexSelectsOffset = fastllm_dsa_fused::Align(
        kIndexLogitsOffset + kQueryRows * kMaxSeqLen * sizeof(float));
    static constexpr size_t kRadixControlOffset = fastllm_dsa_fused::Align(
        kIndexSelectsOffset + kQueryRows * kIndexTopK * sizeof(int));
    static constexpr size_t kAttentionTempOffset = fastllm_dsa_fused::Align(
        kRadixControlOffset + (kQueryRows + 1) * sizeof(int));
    static constexpr size_t kTopKGridOffset = fastllm_dsa_fused::Align(
        kAttentionTempOffset + 2 * kQueryRows * sizeof(unsigned));
    static constexpr size_t kTopKGridBytes = Base::kTopKGridBytes;
    static constexpr size_t kBarrierOffset = fastllm_dsa_fused::Align(kTopKGridOffset + kTopKGridBytes);
    static constexpr size_t kBytes = fastllm_dsa_fused::Align(kBarrierOffset + 2 * sizeof(unsigned));
};
static_assert(RequestScratch::kBytes * 2 <= fastllm_dsa_fused::AttnDsaScratchLayout::kBytes);

struct PackedGroup {
    uint2 values[8][4];
    float scales[8];
};
// The layer's exported indices are produced only after projection. Reuse their
// workspace for eight-row activation groups until all projection tiles finish.
__device__ __forceinline__ PackedGroup *ActivationGroups(const Args &a) {
    static_assert((4096 / 32) * sizeof(PackedGroup) <= Layout::Bytes - Layout::Indices);
    return reinterpret_cast<PackedGroup *>(a.workspace + Layout::Indices);
}
__device__ __forceinline__ void PreparedStep(float (&acc)[4], fastllm_mxfp8_projection::WeightFragment w,
                                             const PackedGroup &group, int rows) {
    const int lane = threadIdx.x % 32, g = lane / 4, quad = lane % 4;
    // Inactive N lanes can belong to an earlier request on graph replay.
    const uint2 q = g < rows ? group.values[g][quad] : make_uint2(0, 0);
    const float x0 = quad * 2 < rows ? group.scales[quad * 2] : 0.f;
    const float x1 = quad * 2 + 1 < rows ? group.scales[quad * 2 + 1] : 0.f;
    fastllm_mxfp8_projection::Accumulate(acc, w, q, x0, x1);
}
// The Indexer query image is dead after selection. It fits all 32 activation
// groups for O without reserving another global workspace.
__device__ __forceinline__ PackedGroup *OutputGroups(const Args &a) {
    using Scratch = fastllm_dsa_fused::AttnDsaScratchLayout;
    static_assert(32 * sizeof(PackedGroup) <= Scratch::kIndexWeightsScaledOffset - Scratch::kIqPackedOffset);
    return reinterpret_cast<PackedGroup *>(a.coreWorkspace + Scratch::kIqPackedOffset);
}

struct PackOutputGroup {
    PackedGroup *groups;
    __device__ __forceinline__ void operator()(int row, int col, B value) const {
        float maximum = fabsf(float(value));
#pragma unroll
        for (int delta = 16; delta; delta /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
        const auto exponent = fastllm_mxfp8::ScaleExponent(maximum);
        const float inv = __uint_as_float(exponent == 254 ? 0x00400000u : (254u - exponent) << 23);
        auto &g = groups[col / 32];
        reinterpret_cast<unsigned char *>(g.values[row])[col % 32] = __nv_fp8_e4m3(float(value) * inv).__x;
        if (!(col % 32))
            g.scales[row] = __uint_as_float(exponent << 23);
    }
};
template <bool Pack> __device__ void NormalizeOwner(const Args &a, int row, B *scratch) {
    const int t = threadIdx.x;
    const int lane = t % 32;
    __shared__ float sums[8];
    uint4 residual[2];
    float square = 0;
    if (t < 256) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int k = (i * 256 + t) * 8;
            auto v = *reinterpret_cast<const uint4 *>(a.hidden + row * 4096 + k);
            auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&v);
            // The BF16 projection rounds residual + 0 even without a branch.
            if (a.branch || !Pack) {
                const uint4 branch = a.branch ? *reinterpret_cast<const uint4 *>(a.branch + row * 4096 + k)
                                              : make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    pairs[j] = __hadd2(pairs[j], reinterpret_cast<const __nv_bfloat162 *>(&branch)[j]);
            }
            residual[i] = v;
            if (a.feature) *reinterpret_cast<uint4 *>(a.feature + row * 4096 + k) = v;
            *reinterpret_cast<uint4 *>(scratch + k) = v;
        }
    }
    __syncthreads();
    if (t < 256) {
        // Keep the original 256-lane RMS reduction order.
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float value = float(scratch[t + j * 256]);
            square += value * value;
        }
        square = WarpSum(square);
        if (!lane) sums[t / 32] = square;
    }
    __syncthreads();
    const float total = WarpSum(lane < 8 ? sums[lane] : 0.f);
    const float inverse = rsqrtf(total / 4096 + a.eps);
    if (t >= 256) return;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int k = (i * 256 + t) * 8;
        const auto *v = reinterpret_cast<const B *>(&residual[i]);
        uint4 norm;
        float4 gamma[2] = {*reinterpret_cast<const float4 *>(a.inputNorm + k),
                           *reinterpret_cast<const float4 *>(a.inputNorm + k + 4)};
#pragma unroll
        for (int j = 0; j < 8; ++j)
            reinterpret_cast<B *>(&norm)[j] = __float2bfloat16_rn(
                Round(float(v[j]) * inverse) * reinterpret_cast<const float *>(gamma)[j]);
        *reinterpret_cast<uint4 *>(a.normed + row * 4096 + k) = norm;
        if constexpr (Pack) {
            const auto packed = fastllm_mxfp8_projection::Quantize(norm);
            auto &group = ActivationGroups(a)[k / 32];
            group.values[row][t % 4] = packed.values;
            if (!(t % 4)) group.scales[row] = packed.scale;
        }
    }
}
template <int Rows, int Width> __device__ void LoadProjectionInput(const B *input, int rows, B *output) {
    constexpr int vectorsPerRow = Width / 8;
    for (int i = threadIdx.x; i < Rows * vectorsPerRow; i += Threads) {
        int row = i / vectorsPerRow, col = (i % vectorsPerRow) * 8;
        B *target = output + (row / 8) * 8 * Width + fastllm_dsa_projection::input_offset(row % 8, col);
        if (row < rows) {
            unsigned dst = unsigned(__cvta_generic_to_shared(target));
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;" : : "r"(dst), "l"(input + row * Width + col)
                         : "memory");
        } else {
            *reinterpret_cast<uint4 *>(target) = make_uint4(0, 0, 0, 0);
        }
    }
    asm volatile("cp.async.commit_group;cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
}

// A CTA owns two 16-output tiles with four K partitions per tile. The
// Indexer uses its original BF16 weights; only the QKV/O tiles use MXFP8.
template<int Width, bool Prepared = false, class Emit>
__device__ void ProjectMx(const void *weight, const float *scales, int column,
                          const B *input, float *partial, Emit emit,
                          const PackedGroup *groups = nullptr, int rows = 8) {
    constexpr int Splits = 4;
    const int t = threadIdx.x, warp = t/32, g = t%32/4, q = t%4;
    if (warp < 2*Splits) {
        const int first = column + warp/Splits*16;
        const int begin = warp%Splits*(Width/Splits);
        float c[4]{};
#pragma unroll 2
        for (int base = begin; base < begin+Width/Splits; base += 32) {
            const int k = base + q*8;
            if constexpr (Prepared) {
                auto w = fastllm_mxfp8_projection::LoadWeights(weight, scales, Width, first + g, first + g + 8, k);
                PreparedStep(c, w, groups[base / 32], rows);
            } else {
                const uint4 x = *reinterpret_cast<const uint4 *>(input + fastllm_dsa_projection::input_offset(g,k));
                fastllm_mxfp8_projection::Step(c,weight,scales,Width,first+g,first+g+8,k,x);
            }
        }
        float *dst = partial+warp*128+g*8+q*2;
        dst[0]=c[0]; dst[1]=c[1]; dst[64]=c[2]; dst[65]=c[3];
    }
    __syncthreads();
    if (t < 256) {
        const int tile = t/128, out = t%128/8, row = t%8;
        float sum = 0.f;
#pragma unroll
        for (int split=0;split<Splits;++split)
            sum += partial[(tile*Splits+split)*128+out*8+row];
        emit(tile*16+out,row,sum);
    }
    __syncthreads();
}
template <int Rows, bool Mx> __device__ void Project(const Args &a, int first, ProjectionStorage<Rows> &s) {
    using namespace fastllm_dsa_projection;
    if (!Mx || blockIdx.x >= 58)
        LoadProjectionInput<Rows, 4096>(a.normed + size_t(first) * 4096, a.rows - first, s.input);
    auto pointer = [&](int col) {
        if (col < 1856)
            return a.qkvWeight + size_t(col) * 4096;
        if (col < 1984)
            return a.indexKeyWeight + size_t(col - 1856) * 4096;
        if (col < 4032)
            return a.indexQueryWeight + size_t(col - 1984) * 4096;
        return a.indexWeight + size_t(col - 4032) * 4096;
    };
    auto emit = [&](int col, int row, float value) {
        if (first + row >= a.rows)
            return;
        if (col < 1856)
            ((B *)(a.workspace + Layout::Qkv))[(first + row) * 1856 + col] = __float2bfloat16_rn(value);
        else if (col < 1984)
            ((B *)(a.workspace + Layout::IndexKey))[(first + row) * 128 + col - 1856] =
                __float2bfloat16_rn(value);
        else if (col < 4032)
            ((B *)(a.workspace + Layout::IndexQuery))[(first + row) * 2048 + col - 1984] =
                __float2bfloat16_rn(value);
        else
            ((B *)(a.workspace + Layout::IndexWeight))[(first + row) * 16 + col - 4032] =
                __float2bfloat16_rn(Round(value) * .25f);
    };
    // One wave covers QKV and all Indexer projections: two output tiles, four K splits each.
    if constexpr (Mx) {
        static_assert(Rows == 8);
        if (blockIdx.x < 58) {
            const int column = blockIdx.x*32;
            auto output = [&](int col, int row, float value) { emit(column+col,row,value); };
            ProjectMx<4096, true>(a.qkvWeight, a.qkvScales, column, s.input, s.partial, output,
                                  ActivationGroups(a), a.rows);
        } else {
            project_rows_mma<Rows,4096,4,2,12>(1856,137,pointer,s.input,s.partial,emit,int(blockIdx.x)-58);
        }
    } else project_rows_mma<Rows, 4096, 4, 2, 12>(0, 253, pointer, s.input, s.partial, emit);
}

__device__ void NormalizeIndex(const Args &a, int row) {
    B *x = (B *)(a.workspace + Layout::IndexKey) + row * 128;
    __shared__ float sums[64], squares[64], mean, inverseStd;
    int t = threadIdx.x;
    if (t < 64) {
        float sum = 0, square = 0;
        for (int j = t; j < 128; j += 64) {
            float v = float(x[j]);
            sum += v;
            square += v * v;
        }
        sums[t] = sum;
        squares[t] = square;
    }
    __syncthreads();
    for (int step = 32; step; step >>= 1) {
        if (t < step) {
            sums[t] += sums[t + step];
            squares[t] += squares[t + step];
        }
        __syncthreads();
    }
    if (!t) {
        mean = sums[0] / 128;
        float variance = squares[0] + mean * mean * 128 - 2 * mean * 128 * mean;
        inverseStd = rsqrtf(variance / 128 + 1e-10f);
    }
    __syncthreads();
    if (t < 128)
        x[t] = __float2bfloat16_rn((float(x[t]) - mean) * inverseStd * a.indexNorm[t] + a.indexBias[t]);
    __syncthreads();
}
__device__ __forceinline__ void RotateWarp(const B *source, B *output, int dim, float cs, float sn) {
    int lane = threadIdx.x % 32;
    float x = float(source[lane]), y = float(source[lane + 32]);
    output[lane] = __float2bfloat16_rn(Round(x * cs) - Round(y * sn));
    output[lane + 32] = __float2bfloat16_rn(Round(y * cs) + Round(x * sn));
    for (int col = 64 + lane; col < dim; col += 32)
        output[col] = source[col];
}
enum class PublishPart { Qkv, IndexKey, IndexQuery };
template <bool Fp8Key> __device__ void Publish(const Args &a, int row, PublishPart part) {
    const int t = threadIdx.x;
    auto r = FastllmNaiveResolveDSARow(a.bindings[row]);
    int length = r.liveKeys ? *r.liveKeys : r.length;
    if (length <= 0 || length > r.capacity) {
        if (part == PublishPart::Qkv) {
            for (int i = t; i < 1536; i += Threads)
                a.query[row * 1536 + i] = __float2bfloat16_rn(0.f);
        }
        return;
    }
    const B *raw = (const B *)(a.workspace + Layout::Qkv) + row * 1856;
    B *iq = (B *)(a.workspace + Layout::IndexQuery) + row * 2048;
    const int lane = t % 32, warp = t / 32;
    if (part == PublishPart::IndexKey)
        NormalizeIndex(a, row);
    __shared__ float cosine[32], sine[32];
    __shared__ B index[128];
    if (t < 32) {
        float angle = a.positions[row] * powf(a.theta, -2.f * t / 64.f);
        float cs, sn;
        sincosf(angle, &sn, &cs);
        cosine[t] = Round(cs);
        sine[t] = Round(sn);
    }
    __syncthreads();
    const float cs = cosine[lane], sn = sine[lane];
    if (part == PublishPart::IndexQuery) {
        for (int head = warp; head < 16; head += Threads / 32)
            RotateWarp(iq + head * 128, iq + head * 128, 128, cs, sn);
        // Packing can read a head rotated by another warp in this CTA.
        __syncthreads();
        return;
    }
    B *k = (B *)r.key + size_t(FastllmNaivePhysicalRow(r, length - 1)) * r.keyStride;
    if (part == PublishPart::Qkv) {
        if (warp < 8)
            RotateWarp(raw + warp * 192, a.query + row * 1536 + warp * 192, 192, cs, sn);
        else if (warp == 8)
            RotateWarp(raw + 1536, k, 192, cs, sn);
        else if (warp == 9) {
            B *v = (B *)r.value + size_t(FastllmNaivePhysicalRow(r, length - 1, true)) * 128;
            for (int col = lane; col < 128; col += 32)
                v[col] = __float2bfloat16_rn(float(raw[1728 + col]) * Round(a.valueScale));
        }
        return;
    }
    if (warp == 0)
        RotateWarp((B *)(a.workspace + Layout::IndexKey) + row * 128, index, 128, cs, sn);
    __syncthreads();
    if constexpr (Fp8Key) {
        // Reduce the 128 Indexer values, then publish the per-row FP8 scale.
        __shared__ float maxima[12];
        float x = t < 128 ? float(index[t]) : 0.f, maximum = fabsf(x);
        for (int delta = 16; delta; delta >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
        if (!(t & 31))
            maxima[t / 32] = maximum;
        __syncthreads();
        maximum = 0;
        for (int j = 0; j < 12; ++j)
            maximum = fmaxf(maximum, maxima[j]);
        float scale = fmaxf(maximum, 1e-4f) / 448.f;
        unsigned char *dst = (unsigned char *)(k + 192);
        if (t < 128)
            ((__nv_fp8_e4m3 *)dst)[t] = __nv_fp8_e4m3(fmaxf(-448.f, fminf(448.f, x / scale)));
        if (!t) {
            *(float *)(dst + 128) = scale;
            *(unsigned *)(dst + 132) = 0;
            *(uint64_t *)(dst + 136) = 0;
        }
    } else if (t < 128)
        k[192 + t] = index[t];
    __syncthreads();
}

template <int Rows, bool Mx, bool SplitOutput = false>
__device__ void Tail(const Args &a, int first, TailStorage<Rows> &s) {
    using namespace fastllm_dsa_projection;
    using namespace fastllm_swa_output;
    const int t = threadIdx.x, col = blockIdx.x * 32;
    if (blockIdx.x >= 128)
        return;
    if constexpr (!SplitOutput) {
        if (Rows == 8 && (reinterpret_cast<uintptr_t>(a.attention) & 15) == 0) {
            LoadProjectionInput<Rows, 1024>(a.attention + size_t(first) * 1024, a.rows - first, s.input);
        } else {
            for (int i = t; i < Rows * 1024; i += Threads)
                s.input[(i / (8 * 1024)) * 8 * 1024 + fastllm_dsa_projection::input_offset((i / 1024) % 8, i % 1024)] =
                    first + i / 1024 < a.rows ? a.attention[first * 1024 + i] : __float2bfloat16_rn(0.f);
            __syncthreads();
        }
    }
    auto pointer = [&](int column) { return a.outputWeight + size_t(col + column) * 1024; };
    auto emit = [&](int column, int row, float value) { s.local[row][column] = __float2bfloat16_rn(value); };
    // Two tiles share each attention load. Four K splits retain FP32 accumulation.
    if constexpr (SplitOutput) ProjectMx<1024, true>(a.outputWeight, a.outputScales, col, s.input, s.partial, emit, OutputGroups(a), 8);
    else if constexpr (Mx) ProjectMx<1024>(a.outputWeight,a.outputScales,col,s.input,s.partial,emit);
    else project_rows_mma<Rows, 1024, 4, 2, 12>(0, 2, pointer, s.input, s.partial, emit, 0);
    // Reuse the established eight-row mailbox generations. Projection weights
    // are shared across sixteen rows; collective storage does not grow.
    for (int base = 0; base < Rows && first + base < a.rows; base += 8) {
      const int rows = min(8, a.rows - first - base);
        if constexpr (SplitOutput) {
            if (a.rank >= 0) {
                if (t < rows) {
                    auto *epoch = reinterpret_cast<uint32_t *>(a.epochs + t * MaxBlocks + blockIdx.x);
                    const unsigned step = *epoch;
                    s.step[t] = step;
                    *epoch = uint64_t(step) + 1 == Period ? 0 : step + 1;
                }
                __syncthreads();
                for (int i = t; i < rows * 8 * 16; i += Threads) {
                    const int row = i / 128, peer = i / 16 % 8, pair = i % 16;
                    if (peer == a.rank) continue;
                    const auto step = s.step[row];
                    const size_t off = size_t(step % Frames) * FrameBytes + row * 8 * SlotBytes +
                                       a.rank * SlotBytes + (col / 2 + pair) * 8;
                    const uint32_t bits = reinterpret_cast<const uint32_t *>(s.local[base+row])[pair];
                    Store(a.peers[peer] + off, (uint64_t(bits) << 32) | (step + 1));
                }
                __syncthreads();
                if (t < rows * 16) {
                    const int row = t / 16, pair = t % 16;
                    const unsigned step = s.step[row], tag = step + 1;
                    const size_t off = size_t(step % Frames) * FrameBytes + row * 8 * SlotBytes + (col/2 + pair) * 8;
                    const char *local = a.peers[a.rank] + off;
                    uint64_t words[8];
#pragma unroll
                    for (int peer = 0; peer < 8; ++peer)
                        words[peer] = peer == a.rank ? (uint64_t(reinterpret_cast<const uint32_t *>(
                                         s.local[base+row])[pair]) << 32 | tag) : Load(local + peer * SlotBytes);
                    bool pending;
                    do {
                        pending = false;
#pragma unroll
                        for (int peer = 0; peer < 8; ++peer) {
                            if (uint32_t(words[peer]) != tag) {
                                words[peer] = Load(local + peer * SlotBytes);
                                pending = true;
                            }
                        }
                    } while (pending);
                    float lo=0.f, hi=0.f;
#pragma unroll
                    for (int peer=0; peer<8; ++peer) {
                        const uint32_t bits = uint32_t(words[peer] >> 32);
                        const float x = float(__ushort_as_bfloat16(bits & 65535)), y = float(__ushort_as_bfloat16(bits >> 16));
                        lo = peer ? lo+x : x; hi = peer ? hi+y : y;
                    }
                    const int idx = (first + base + row) * 4096 + col + 2*pair;
                    const B x = __float2bfloat16_rn(lo), y = __float2bfloat16_rn(hi);
                    a.projected[idx] = x; a.projected[idx+1] = y;
                    a.hidden[idx]=__float2bfloat16_rn(Round(float(a.hidden[idx])+(a.branch?float(a.branch[idx]):0.f))+float(x));
                    a.hidden[idx+1]=__float2bfloat16_rn(Round(float(a.hidden[idx+1])+(a.branch?float(a.branch[idx+1]):0.f))+float(y));
                }
                __syncthreads();
                continue;
            }
        }

        if (a.rank >= 0) {
            if (t < rows) {
                auto *epoch = reinterpret_cast<uint32_t *>(a.epochs + t * MaxBlocks + blockIdx.x);
                s.step[t] = *epoch;
                *epoch = uint64_t(s.step[t]) + 1 == Period ? 0 : s.step[t] + 1;
            }
            __syncthreads();
            for (int i = t; i < rows * 8 * 16; i += Threads) {
                int row = i / 128, peer = i / 16 % 8, pair = i % 16, word = col / 2 + pair;
                size_t frame = (s.step[row] % Frames) * FrameBytes;
                uint32_t bits = reinterpret_cast<const uint32_t *>(s.local[base + row])[pair];
                Store(a.peers[peer] + frame + row * 8 * SlotBytes + a.rank * SlotBytes + word * 8,
                      (uint64_t(bits) << 32) | (s.step[row] + 1));
            }
            __syncthreads();
            for (int i = t; i < rows * 8 * 16; i += Threads) {
                int row = i / 128, peer = i / 16 % 8, pair = i % 16, word = col / 2 + pair;
                size_t frame = (s.step[row] % Frames) * FrameBytes;
                uint64_t bits;
                do {
                    bits = Load(a.peers[a.rank] + frame + row * 8 * SlotBytes + peer * SlotBytes + word * 8);
                } while (uint32_t(bits) != s.step[row] + 1);
                s.received[row][peer][pair] = bits >> 32;
            }
            __syncthreads();
        }
        for (int i = t; i < rows * 32; i += Threads) {
            int row = i / 32, d = i % 32;
            float sum = float(s.local[base + row][d]);
            if (a.rank >= 0) {
                sum = 0;
                for (int peer = 0; peer < 8; ++peer) {
                    uint32_t bits = s.received[row][peer][d / 2];
                    float x = float(__ushort_as_bfloat16((bits >> ((d % 2) * 16)) & 65535));
                    if (!peer)
                        sum = x;
                    else
                        sum += x;
                }
            }
            int index = (first + base + row) * 4096 + col + d;
            B value = __float2bfloat16_rn(sum);
            a.projected[index] = value;
            a.hidden[index] = __float2bfloat16_rn(
                Round(float(a.hidden[index]) + (a.branch ? float(a.branch[index]) : 0.f)) + float(value));
        }
        __syncthreads();
    }
}

struct CoreView {
    const B *query, *indexQuery, *weights;
    const float *sink;
    const FastllmNaiveDSARow *rows;
    FastllmNaiveKV indexKey;
    const void *const *indexCache;
    int indexStride, activeRows;
    B *output;
    unsigned char *workspace;
};

__device__ CoreView CoreArguments(const Args &a, int offset, int request) {
    CoreView core;
    core.query = a.query + offset * 1536;
    core.indexQuery = (const B *)(a.workspace + Layout::IndexQuery) + offset * 2048;
    core.weights = (const B *)(a.workspace + Layout::IndexWeight) + offset * 16;
    core.sink = a.sink;
    core.activeRows = a.requests[request];
    core.indexStride = a.bindings[offset].keyStride;
    core.indexCache = a.bindings[offset].cache;
    core.indexKey =
        FastllmNaiveKV(a.bindings[offset].key, a.bindings[offset].keyPages, a.bindings[offset].pageLen);
    core.indexKey.scaledIndexer = a.fp8Key;
    core.output = a.attention + offset * 1024;
    core.workspace = a.coreWorkspace;
    core.rows = a.bindings + offset;
    return core;
}
template <bool Fp8Key> __device__ void PackPublishedQuery(const Args &a, int row) {
    using Body = fastllm_dsa_fused::Body<Fp8Key, false>;
    using S = typename Body::Scratch;
    const int t = threadIdx.x, head = t / 16, lane = t & 15, d = lane * 8;
    if (head >= 16)
        return;
    const B *src = (const B *)(a.workspace + Layout::IndexQuery) + row * 2048 + head * 128 + d;
    float values[8], maximum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        values[j] = float(src[j]);
        maximum = fmaxf(maximum, fabsf(values[j]));
    }
#pragma unroll
    for (int step = 8; step; step >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
    float scale = fmaxf(maximum, 1e-4f) / 448.f;
    unsigned char *iq = a.coreWorkspace + S::kIqPackedOffset;
    float *scales = (float *)(a.coreWorkspace + S::kIndexWeightsScaledOffset);
    for (int target = row; target < 8; target += (row == a.rows - 1 ? 1 : 8)) {
        int vector = target * 16 + head;
        int dst = Body::ImageOffset(vector, d, 128);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            auto q = __nv_fp8_e4m3(fmaxf(-448.f, fminf(448.f, values[j] / scale)));
            if constexpr (Fp8Key)
                iq[dst + j] = q.__x;
            else
                ((__nv_bfloat16 *)(iq + dst))[j] = __float2bfloat16(float(q));
        }
        if (!lane)
            scales[vector] = scale;
    }
}

__device__ __forceinline__ void PrefetchWeights(const void *data, int bytes) {
    const auto *p = static_cast<const unsigned char *>(data);
    for (int i = threadIdx.x * 128; i < bytes; i += Threads * 128)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + i));
}
template <bool Fp8Key, int Rows = 8, bool PairedAttention = false, bool Mx = false> __global__ __launch_bounds__(Threads, 1) void Layer(const __grid_constant__ Args a) {
    extern __shared__ __align__(1024) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage<Fp8Key, Rows, PairedAttention> *>(bytes);
    auto grid = cooperative_groups::this_grid();
    if constexpr (Mx) {
        const int c = blockIdx.x;
        if (c < 58) {
            PrefetchWeights(reinterpret_cast<const unsigned char *>(a.qkvWeight) + c * 32 * 4096, 32 * 4096);
            PrefetchWeights(a.qkvScales + c * 4096, 4096 * sizeof(float));
        }
    }
    // Single-request control is published by the existing frontend barrier.
    if (a.requestCount == 1)
        fastllm_dsa_fused::Body<Fp8Key, false>::initialize(CoreArguments(a, 0, 0));
    // Low-index CTAs initialize the Indexer; the last CTAs own RMS rows.
    const int ownerRow = int(gridDim.x - 1 - blockIdx.x);
    if (ownerRow < a.rows)
        NormalizeOwner<Mx>(a, ownerRow, s.projection.input);
    grid.sync();
    for (int first = 0; first < a.rows; first += Rows)
        Project<Rows,Mx>(a, first, s.projection);
    grid.sync();
    // Three independent publishers per row join at the existing grid barrier.
    const int publishRow = blockIdx.x / 3;
    if (publishRow < a.rows) {
        const auto part = static_cast<PublishPart>(blockIdx.x % 3);
        Publish<Fp8Key>(a, publishRow, part);
        if (part == PublishPart::IndexQuery && a.requestCount == 1)
            PackPublishedQuery<Fp8Key>(a, publishRow);
    }
    grid.sync();
    if (a.phaseStop == 1)
        return;
    // Read descriptors directly from immutable launch arguments, avoiding
    // shared-memory copies on the attention path.
    if constexpr (Rows > 8) {
        // Short two-request selections use disjoint views of the old scratch;
        // other shapes reuse it serially. Exported indices live in the layer
        // workspace and survive until every request reaches attention.
        bool parallel = a.requestCount == 2 && int(gridDim.x) % 2 == 0;
        for (int row = 0; row < a.rows && parallel; ++row) {
            const auto &binding = a.bindings[row];
            const int visible = max(0, min(binding.capacity, binding.liveKeys ? *binding.liveKeys : binding.length));
            parallel = visible <= RequestScratch::kMaxSeqLen;
        }
        if (parallel) {
            using RequestBody = fastllm_dsa_fused::Body<Fp8Key, false, RequestScratch, 2>;
            static_assert(sizeof(typename RequestBody::SharedStorage) == sizeof(s.core));
            if (blockIdx.x < 2 && !threadIdx.x) {
                auto *control = reinterpret_cast<unsigned *>(a.coreWorkspace +
                    blockIdx.x * RequestScratch::kBytes + RequestScratch::kBarrierOffset);
                control[0] = control[1] = 0;
            }
            grid.sync();
            const int request = blockIdx.x / (gridDim.x / 2);
            CoreView core = CoreArguments(a, request ? a.requests[0] : 0, request);
            core.workspace = a.coreWorkspace + request * RequestScratch::kBytes;
            auto &shared = *reinterpret_cast<typename RequestBody::SharedStorage *>(&s.core);
            RequestBody::template run<false>(core, shared, true, false);
            grid.sync();
        } else {
            int offset = 0;
            for (int request = 0; request < a.requestCount; ++request) {
                CoreView core = CoreArguments(a, offset, request);
                fastllm_dsa_fused::Body<Fp8Key, false>::template run<false>(
                    core, s.core, request + 1 == a.requestCount, false);
                grid.sync();
                offset += a.requests[request];
            }
        }
        using Scratch = fastllm_dsa_fused::AttnDsaScratchLayout;
        namespace Attention = fastllm_dsa_fused::native_attention;
        constexpr int WideAttentionSplits = PairedAttention ? 8 : 16;
        constexpr int WideCombineColumns = PairedAttention ? 128 : 64;
        constexpr int WideCombineTasks = 8 * (128 / WideCombineColumns);
        static_assert(Rows * Attention::Scratch * sizeof(float) <=
                      Scratch::kIndexSelectsOffset - Scratch::kIndexLogitsOffset);
        // Logits are dead after all Top-K selections. Reuse their existing
        // arena for all request/head partials without growing global memory.
        auto *temp = reinterpret_cast<float *>(a.coreWorkspace + Scratch::kIndexLogitsOffset);
        for (int task = blockIdx.x; task < a.rows * WideAttentionSplits; task += gridDim.x) {
            const int row = task / WideAttentionSplits;
            Attention::Attend<true, Attention::Phase::Produce, WideAttentionSplits>(a.query, a.bindings[row], a.sink, a.attention,
                temp, nullptr, 8, 1, row, 0, task % WideAttentionSplits, s.attention);
        }
        grid.sync();
        // Nonwaiting producers permit a task grid larger than residency.
        // The cooperative join publishes all partials before distributed fold.
        for (int task = blockIdx.x; task < a.rows * WideCombineTasks; task += gridDim.x) {
            const int row = task / WideCombineTasks;
            Attention::Attend<true, Attention::Phase::Combine, WideAttentionSplits, WideCombineColumns>(a.query, a.bindings[row], a.sink, a.attention,
                temp, nullptr, 8, 1, row, 0, task % WideCombineTasks, s.attention);
        }
        grid.sync();
    } else {
        int offset = 0;
        for (int request = 0; request < a.requestCount; ++request) {
            CoreView core = CoreArguments(a, offset, request);
            fastllm_dsa_fused::Body<Fp8Key, false>::run(core, s.core, request + 1 == a.requestCount,
                                                        a.requestCount == 1);
            grid.sync();
            offset += a.requests[request];
        }
    }
    // Tail tiles own disjoint columns/mailbox slots; no grid reader follows them.
    for (int first = 0; first < a.rows; first += Rows) {
        Tail<Rows,Mx>(a, first, s.tail);
    }
}

#include "naive-n05-dsa-split-cluster.cuh"
using SplitBody = fastllm_dsa_fused::Body<false, false>;
using SplitLayout = SplitBody::Scratch;
constexpr int FallbackOffset = SplitLayout::kRadixControlOffset + 9 * sizeof(int);
static_assert(FallbackOffset + 8 * sizeof(unsigned) <= SplitLayout::kAttentionTempOffset);
// Each warp owns a complete head; rotating and packing use the same BF16 rounding as Publish.
__device__ void RotateAndPack(const Args &a, int row) {
    using S = SplitBody::Scratch;
    auto r = FastllmNaiveResolveDSARow(a.bindings[row]);
    const int length = r.liveKeys ? *r.liveKeys : r.length;
    const bool rotate = length > 0 && length <= r.capacity;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const float angle = a.positions[row] * powf(a.theta, -2.f * lane / 64.f);
    float cs, sn;
    sincosf(angle, &sn, &cs);
    cs = Round(cs);
    sn = Round(sn);
    auto *base = (B *)(a.workspace + Layout::IndexQuery) + row * 2048;
    auto *packed = (B *)(a.coreWorkspace + S::kIqPackedOffset);
    auto *scales = (float *)(a.coreWorkspace + S::kIndexWeightsScaledOffset);
    for (int h = warp; h < 16; h += Threads / 32) {
        B *p = base + h * 128;
        float x = float(p[lane]), y = float(p[32 + lane]);
        float vals[4] = {x, y, float(p[64 + lane]), float(p[96 + lane])};
        if (rotate) {
            vals[0] = Round(Round(x * cs) - Round(y * sn));
            vals[1] = Round(Round(y * cs) + Round(x * sn));
            p[lane] = __float2bfloat16(vals[0]);
            p[32 + lane] = __float2bfloat16(vals[1]);
        }
        float m = fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])), fmaxf(fabsf(vals[2]), fabsf(vals[3])));
#pragma unroll
        for (int delta = 16; delta; delta >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, delta));
        float scale = fmaxf(m, 1.e-4f) / 448.f;
        int v = row * 16 + h;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_fp8_e4m3 q(fmaxf(-448.f, fminf(448.f, vals[i] / scale)));
            packed[SplitBody::ImageOffset(v, 32 * i + lane, 128) / 2] = __float2bfloat16(float(q));
        }
        if (!lane)
            scales[v] = scale;
    }
}
__device__ void WriteIndexKeyWarp(const Args &a, int row) {
    if (threadIdx.x >= 32)
        return;
    const int lane = threadIdx.x;
    auto r = FastllmNaiveResolveDSARow(a.bindings[row]);
    const int n = r.liveKeys ? *r.liveKeys : r.length;
    if (n <= 0 || n > r.capacity)
        return;
    B *p = (B *)(a.workspace + Layout::IndexKey) + row * 128;
    float v[4], sq[2], sum[2];
#pragma unroll
    for (int i = 0; i < 4; i++)
        v[i] = float(p[i * 32 + lane]);
#pragma unroll
    for (int i = 0; i < 2; i++) {
        sum[i] = v[i] + v[i + 2];
        sq[i] = fmaf(v[i + 2], v[i + 2], v[i] * v[i]);
    }
    float t = sum[0] + sum[1], u = sq[0] + sq[1];
    for (int delta = 16; delta; delta >>= 1) {
        t += __shfl_down_sync(0xffffffff, t, delta);
        u += __shfl_down_sync(0xffffffff, u, delta);
    }
    float mean = __shfl_sync(0xffffffff, t, 0) / 128.f;
    float variance = __shfl_sync(0xffffffff, u, 0) + mean * mean * 128 - 2 * mean * 128 * mean;
    float inv = rsqrtf(variance / 128 + 1.e-10f);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = Round((v[i] - mean) * inv * a.indexNorm[i * 32 + lane] + a.indexBias[i * 32 + lane]);
        p[i * 32 + lane] = __float2bfloat16(v[i]);
    }
    float cs, sn;
    sincosf(a.positions[row] * powf(a.theta, -2.f * lane / 64.f), &sn, &cs);
    cs = Round(cs);
    sn = Round(sn);
    B *out = (B *)r.key + size_t(FastllmNaivePhysicalRow(r, n - 1)) * r.keyStride + 192;
    out[lane] = __float2bfloat16(Round(v[0] * cs) - Round(v[1] * sn));
    out[32 + lane] = __float2bfloat16(Round(v[1] * cs) + Round(v[0] * sn));
    out[64 + lane] = __float2bfloat16(v[2]);
    out[96 + lane] = __float2bfloat16(v[3]);
}

// Five device-scope counters fit in existing aligned control padding. The last
// publisher occupies CTA 142; dispatch checks grid residency before capture.
constexpr int FrontFlags = FallbackOffset + 8 * sizeof(unsigned);
static_assert(FrontFlags + 32 * sizeof(unsigned) <= SplitLayout::kAttentionTempOffset);
__device__ __forceinline__ void FrontSignal(unsigned *p) {
    __syncthreads();
    if (!threadIdx.x)
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}
__device__ __forceinline__ void FrontWait(const unsigned *p, unsigned n) {
    if (!threadIdx.x) {
        for (;;) {
            unsigned v;
            asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory");
            if (v >= n)
                break;
            __nanosleep(32);
        }
    }
    __syncthreads();
}
__device__ void ProjectPair(const Args &a, float *partial) {
    const int t = threadIdx.x, warp = t / 32, lane = t % 32, g = lane / 4, q = lane % 4, c = blockIdx.x;
    float acc[4]{};
    if (warp < 4 && c < 116) {
        const int col = c * 16, first = warp * 1024;
#pragma unroll 8
        for (int kk = first; kk < first + 1024; kk += 32) {
            int k = kk + q * 8;
            fastllm_mxfp8_projection::WeightFragment w;
            w = fastllm_mxfp8_projection::LoadWeights(a.qkvWeight, a.qkvScales, 4096, col + g, col + g + 8, k);
            PreparedStep(acc, w, ActivationGroups(a)[kk / 32], 8);
        }
    } else if (warp >= 4 && c < 137) {

        if (warp < 8) {
            const B *w = c < 8     ? a.indexKeyWeight + c * 16 * 4096
                         : c < 136 ? a.indexQueryWeight + (c - 8) * 16 * 4096
                                   : a.indexWeight;
            const int first = (warp - 4) * 1024;
#pragma unroll 8
            for (int kk = first; kk < first + 1024; kk += 32) {
                const int k = kk + q * 8;
                uint4 a0 = __ldg((const uint4 *)(w + g * 4096 + k)),
                      a1 = __ldg((const uint4 *)(w + (g + 8) * 4096 + k));
                uint4 x = __ldg((const uint4 *)(a.normed + g * 4096 + k));
                fastllm_dsa_projection::mma_bf16_16816(acc, a0.x, a1.x, a0.y, a1.y, x.x, x.y);
                fastllm_dsa_projection::mma_bf16_16816(acc, a0.z, a1.z, a0.w, a1.w, x.z, x.w);
            }
        }
    }
    if (warp < 8) {
        float *p = partial + warp * 128 + g * 8 + q * 2;
        p[0] = acc[0];
        p[1] = acc[1];
        p[64] = acc[2];
        p[65] = acc[3];
    }
    __syncthreads();
    if (t < 256) {
        int tile = t / 128, out = t % 128 / 8, row = t % 8;
        float v = 0;
#pragma unroll
        for (int k = 0; k < 4; k++)
            v += partial[(tile * 4 + k) * 128 + out * 8 + row];
        int col = c * 16 + out;
        if (!tile && c < 116)
            ((B *)(a.workspace + Layout::Qkv))[row * 1856 + col] = __float2bfloat16(v);
        if (tile && c < 137) {
            if (c < 8)
                ((B *)(a.workspace + Layout::IndexKey))[row * 128 + col] = __float2bfloat16(v);
            else if (c < 136)
                ((B *)(a.workspace + Layout::IndexQuery))[row * 2048 + col - 128] = __float2bfloat16(v);
            else
                ((B *)(a.workspace + Layout::IndexWeight))[row * 16 + out] = __float2bfloat16(Round(v) * .25f);
        }
    }
    __syncthreads();
}

// One query block covers at most four key tiles per CTA on the admitted grid.
// The first three use the sweep ring. Only shared memory is needed for the fourth.
constexpr int ScoreTileRows = SplitBody::SweepStorage::kTileM;
constexpr int MaxScoreTiles = 4;
constexpr int SplitScoreBytes = sizeof(Storage<false>) + SplitBody::SweepStorage::kTileBytes;
static_assert(SplitLayout::kMaxSeqLen <= (143 - 2) * ScoreTileRows * MaxScoreTiles);

__global__ __launch_bounds__(Threads, 1) void SplitScores(const __grid_constant__ Args a) {
    extern __shared__ __align__(1024) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage<false> *>(bytes);
    auto *fourth = bytes + sizeof(Storage<false>);
    using Sweep = SplitBody::SweepStorage;
    // The RMS row and projection partials use the beginning of B storage before
    // the packed query arrives. Historical key loads can fill all A slots.
    static_assert(offsetof(Sweep, a_ring) >= 4096 * sizeof(B));
    static_assert(offsetof(Sweep, a_ring) >= 8 * 128 * sizeof(float));
    int earliest = 0x7fffffff, latest = 0;
#pragma unroll
    for (int row = 0; row < 8; ++row) {
        const auto &binding = a.bindings[row];
        const int length = max(0, min(binding.capacity, binding.liveKeys ? *binding.liveKeys : binding.length));
        earliest = min(earliest, length);
        latest = max(latest, length);
    }
    const int total = (latest + ScoreTileRows - 1) / ScoreTileRows;
    const int tailBlocks = total > int(gridDim.x) ? 2 : 0;
    const int stride = int(gridDim.x) - tailBlocks;
    const int block = int(blockIdx.x);
    const bool tail = block >= stride;
    // Give the two newest tiles their own CTAs. Waiting for a current append
    // then leaves the preceding waves of historical tiles free to issue MMA.
    const int first = tail ? total - tailBlocks + block - stride : block;
    const int tiles = tail ? 1 : max(0, (total - tailBlocks - first + stride - 1) / stride);
    const int history = max(0, (earliest - 1) / ScoreTileRows);
    const int prepared = min(tiles, max(0, (history - first + stride - 1) / stride));
    if (prepared) {
        const auto view = CoreArguments(a, 0, 0);
        auto key = view.indexKey;
        key.Bind(view.indexCache);
        for (int j = 0; j < prepared; ++j) {
            const int token = (first + j * stride) * ScoreTileRows;
            auto *dest = SplitBody::score_tile_image(s.core.sweep, fourth, j);
            const B *base = static_cast<const B *>(key.IndexRow(token, view.indexStride).data);
            const bool singlePage = !key.pages || key.pageLen % ScoreTileRows == 0;
            for (int i = threadIdx.x; i < ScoreTileRows * 16; i += Threads) {
                const int row = i / 16, dim = i % 16 * 8;
                const B *src = singlePage ? base + row * view.indexStride
                    : static_cast<const B *>(key.IndexRow(token + row, view.indexStride).data);
                SplitBody::cp_async_16(dest + SplitBody::ImageOffset(row, dim, ScoreTileRows), src + dim, 16);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    auto grid = cooperative_groups::this_grid();
    const int c = blockIdx.x;

    if (c < 116) {
        PrefetchWeights(reinterpret_cast<const unsigned char *>(a.qkvWeight) + c * 16 * 4096, 16 * 4096);
        PrefetchWeights(a.qkvScales + c * 2048, 2048 * sizeof(float));
    }
    // Each cluster selects one query, including dense/empty rows. It needs only
    // the actual lengths and the sparse-attention completion counters.
    if (c == 0 && threadIdx.x < 8) {
        int row = threadIdx.x;
        const auto &binding = a.bindings[row];
        const int visible = max(0, min(binding.capacity, binding.liveKeys ? *binding.liveKeys : binding.length));
        auto *lengths = reinterpret_cast<int *>(a.coreWorkspace + SplitLayout::kRadixControlOffset);
        lengths[row] = visible;
        int longest = __reduce_max_sync(0xffu, visible);
        if (row == 0) lengths[8] = longest;
        auto *temp = reinterpret_cast<float *>(a.coreWorkspace + SplitLayout::kAttentionTempOffset);
        auto *completion = reinterpret_cast<unsigned *>(temp + SplitLayout::kAttentionControlFloatOffset);
        completion[row] = completion[8 + row] = 0;
    }
    // Low-index CTAs initialize the Indexer; the last CTAs own RMS rows.
    auto *ready = (unsigned *)(a.coreWorkspace + FrontFlags);
    if (blockIdx.x == 0 && threadIdx.x < 32)
        ready[threadIdx.x] = 0;
    const int ownerRow = int(gridDim.x - 1 - blockIdx.x);
    if (ownerRow < a.rows)
        NormalizeOwner<true>(a, ownerRow, s.projection.input);
    grid.sync();
    // Four warps compute the main projection tile and four compute its paired
    // Indexer projection. Both preserve the original four-way K reduction.
    if (blockIdx.x < 137)
        ProjectPair(a, reinterpret_cast<float *>(bytes));

    if (c < 116)
        FrontSignal(ready);
    if (c < 137)
        FrontSignal(c < 8 ? ready + 1 : ready + 2);
    if (c < 8) {
        FrontWait(ready, 116);
        Publish<false>(a, c, PublishPart::Qkv);
    }
    if (c >= 127 && c < 135) {
        FrontWait(ready + 2, 129);
        RotateAndPack(a, c - 127);
        FrontSignal(ready + 3);
    }
    if (c >= 135 && c < 143) {
        FrontWait(ready + 1, 8);
        WriteIndexKeyWarp(a, c - 135);
        FrontSignal(ready + 4);
    }
    // A CTA reading only historical keys can start as soon as its query is ready.
    FrontWait(ready + 3, 8);
    if (prepared < tiles) {
        FrontWait(ready + 4, 8);
    }
    if (prepared) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    CoreView core = CoreArguments(a, 0, 0);
    unsigned char *ws = a.coreWorkspace;
    const int *lengths = (const int *)(ws + SplitLayout::kRadixControlOffset);

    if (tiles)
        SplitBody::score_tiles(core, lengths, s.core.sweep, prepared, fourth, first, stride, tiles);
}
template <bool DirectSelection>
__global__ __launch_bounds__(Threads, 1) void SplitAttentionOutput(const __grid_constant__ Args a) {
    extern __shared__ __align__(1024) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage<false> *>(bytes);
    auto core = CoreArguments(a, 0, 0);
    unsigned char *ws = a.coreWorkspace;
    int *selections = DirectSelection ? const_cast<int *>(a.bindings[0].indices)
                                      : reinterpret_cast<int *>(ws + SplitLayout::kIndexSelectsOffset);
    float *temp = (float *)(ws + SplitLayout::kAttentionTempOffset);
    unsigned *flags = (unsigned *)(temp + SplitLayout::kAttentionControlFloatOffset);
    const int t = threadIdx.x;
    if constexpr (!DirectSelection) {
        for (int i = blockIdx.x * Threads + t; i < 8 * 2048; i += gridDim.x * Threads) {
            const auto &r = core.rows[i / 2048];
            if (r.indices)
                const_cast<int *>(r.indices)[i % 2048] = (r.liveKeys ? *r.liveKeys : r.length) > 0 ? selections[i] : -1;
        }
    }
    if (blockIdx.x < 8 * 16) {
        const int row = blockIdx.x / 16;
        auto descriptor = core.rows[row];
        if (descriptor.indices)
            descriptor.indices = selections + row * 2048;
        fastllm_dsa_fused::native_attention::Attend<true, fastllm_dsa_fused::native_attention::Phase::Fused, 16, 64,
                                                    128, true, DirectSelection>(
            core.query, descriptor, core.sink, core.output, temp, flags, 8, 1, row, 0, blockIdx.x % 16, s.core.native,
            PackOutputGroup{OutputGroups(a)});
    }
    cooperative_groups::this_grid().sync();
    Tail<8, true, true>(a, 0, s.tail);
}
// Cache launch eligibility before capture. The pointer table and lengths remain
// device inputs, so changing requests never changes the captured graph.
static bool SplitAvailable(int device, bool directSelection = false) {
    struct Support {
        bool split, direct;
    };
    static thread_local std::unordered_map<int, Support> cache;
    auto found = cache.find(device);
    if (found == cache.end()) {
        if (FastllmCudaGraphIsCapturingFast())
            return false;
        constexpr int shared = sizeof(Storage<false>);
        auto resident = [=](const void *kernel) {
            const int bytes = kernel == (const void *)SplitScores ? SplitScoreBytes : shared;
            int blocks = 0;
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess &&
                   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, Threads, bytes) == cudaSuccess &&
                   blocks >= 1;
        };
        const bool admitted = resident((const void *)SplitScores) && resident((const void *)SplitAttentionOutput<false>) &&
                              cluster_sel::Supported(device);
        const bool direct = admitted && resident((const void *)SplitAttentionOutput<true>);
        if (!direct)
            cudaGetLastError();
        found = cache.emplace(device, Support{admitted, direct}).first;
    }
    return directSelection ? found->second.direct : found->second.split;
}

// Admission uses the output layout bound at capture. Per-replay lengths and page tables stay on device.
static bool DirectSelections(const Args &a) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(a.bindings[0].indices);
    const uintptr_t scratch = reinterpret_cast<uintptr_t>(a.coreWorkspace);
    constexpr size_t bytes = 8 * 2048 * sizeof(int);
    if (!base || (base < scratch + SplitLayout::kBytes && scratch < base + bytes))
        return false;
    for (int row = 0; row < 8; ++row) {
        if (a.bindings[row].selectedCount != 2048 ||
            reinterpret_cast<uintptr_t>(a.bindings[row].indices) != base + row * 2048 * sizeof(int))
            return false;
    }
    return true;
}

static cudaError_t RunSplit(Args &args, int grid) {
    void *params[] = {&args};
    constexpr size_t shared = sizeof(Storage<false>);
    auto stream = cudaStreamPerThread;
    const void *scores = (const void *)SplitScores;
    cudaError_t error = cudaLaunchCooperativeKernel(scores, grid, Threads, params, SplitScoreBytes, stream);
    if (error != cudaSuccess)
        return error;
    const bool direct = DirectSelections(args) && SplitAvailable(FastllmCudaGetDevice(), true);
    int *selected = direct ? const_cast<int *>(args.bindings[0].indices)
                           : reinterpret_cast<int *>(args.coreWorkspace + SplitLayout::kIndexSelectsOffset);
    error = cluster_sel::Launch(
                               (const float *)(args.coreWorkspace + SplitLayout::kIndexLogitsOffset), selected,
                               (unsigned *)(args.coreWorkspace + FallbackOffset),
                               (const int *)(args.coreWorkspace + SplitLayout::kRadixControlOffset), stream);
    if (error != cudaSuccess)
        return error;
    const void *out = direct ? (const void *)SplitAttentionOutput<true> : (const void *)SplitAttentionOutput<false>;
    return cudaLaunchCooperativeKernel(out, grid, Threads, params, shared, stream);
}

template <bool Fp8Key, int Rows = 8, bool PairedAttention = false, bool Mx = false> int GridSize() {
    if (FastllmCudaRuntimeArch() != 103)
        return 0;
    static thread_local std::unordered_map<int, int> grids;
    int device = FastllmCudaGetDevice();
    if (grids.count(device))
        return grids[device];
    if (FastllmCudaGraphIsCapturingFast())
        return 0;
    cudaDeviceProp p{};
    int occupancy = 0;
    const int shared = sizeof(Storage<Fp8Key, Rows, PairedAttention>);
    if (cudaGetDeviceProperties(&p, device) != cudaSuccess || !p.cooperativeLaunch ||
        p.multiProcessorCount < 128 ||
        cudaFuncSetAttribute(Layer<Fp8Key, Rows, PairedAttention, Mx>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) !=
            cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, Layer<Fp8Key, Rows, PairedAttention, Mx>, Threads, shared) !=
            cudaSuccess ||
        occupancy < 1) {
        cudaGetLastError();
        return grids[device] = 0;
    }
    if constexpr (Mx && !Fp8Key && Rows == 8) {
        if (p.multiProcessorCount >= 143)
            SplitAvailable(device);
    }
    return grids[device] = p.multiProcessorCount;
}
} // namespace fastllm_dsa_layer

bool FastllmCudaNaiveDSALayerSupported() {
    fastllm_dsa_layer::GridSize<false, 16, true>();
    fastllm_dsa_layer::GridSize<true, 16, true>();
    fastllm_dsa_layer::GridSize<false, 16>();
    fastllm_dsa_layer::GridSize<true, 16>();
    return fastllm_dsa_layer::GridSize<false>() && fastllm_dsa_layer::GridSize<true>();
}
cudaError_t FastllmCudaNaiveDSALayerLaunch(FastllmNaiveDSALayerArgs args) {
    using namespace fastllm_dsa_layer;
    if (args.qkvScales) {
        int grid = args.fp8Key ? GridSize<true,8,false,true>() : GridSize<false,8,false,true>();
        if (!grid || args.rows > 8) return cudaErrorNotSupported;
        if (grid >= 143 && !args.fp8Key && args.phaseStop == 0 && args.rows == 8 && args.requestCount == 1 &&
            args.bindings[0].capacity >= 16384 && SplitAvailable(FastllmCudaGetDevice()))
            return RunSplit(args, grid);
        const void *kernel = args.fp8Key ? (const void *)Layer<true,8,false,true> : (const void *)Layer<false,8,false,true>;
        size_t bytes = args.fp8Key ? sizeof(Storage<true>) : sizeof(Storage<false>);
        void *params[] = {&args};
        return cudaLaunchCooperativeKernel(kernel,grid,Threads,params,bytes,cudaStreamPerThread);
    }
    int grid = args.rows > 8 ? (args.fp8Key ? GridSize<true, 16>() : GridSize<false, 16>()) : 0;
    const bool wide = grid != 0;
    if (!wide) grid = args.fp8Key ? GridSize<true>() : GridSize<false>();
    if (!grid) return cudaErrorNotSupported;
    const void *kernel = wide ? (args.fp8Key ? (const void *)Layer<true, 16> : (const void *)Layer<false, 16>)
                              : (args.fp8Key ? (const void *)Layer<true> : (const void *)Layer<false>);
    size_t bytes = wide ? (args.fp8Key ? sizeof(Storage<true, 16>) : sizeof(Storage<false, 16>))
                        : (args.fp8Key ? sizeof(Storage<true>) : sizeof(Storage<false>));
    // Two full requests benefit from one 128-task production/fold wave.
    // Ragged tiles retain the measured sixteen-split schedule; optional shared
    // resource misses also use that path without changing global workspace.
    if (wide && args.rows == 16 && args.requestCount == 2) {
        const int pairedGrid = args.fp8Key ? GridSize<true,16,true>() : GridSize<false,16,true>();
        if (pairedGrid) {
            grid=pairedGrid;
            kernel=args.fp8Key ? (const void*)Layer<true,16,true> : (const void*)Layer<false,16,true>;
            bytes=args.fp8Key ? sizeof(Storage<true,16,true>) : sizeof(Storage<false,16,true>);
        }
    }
    void *parameters[] = {&args};
    return cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(Threads), parameters, bytes,
                                       cudaStreamPerThread);
}

#include "naive-n05-dsa-layer-prepare.cuh"
