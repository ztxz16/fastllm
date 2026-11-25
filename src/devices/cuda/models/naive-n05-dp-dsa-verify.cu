#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/naive-n05-kv.cuh"
#include "devices/cuda/naive-n05-dp-dsa-verify.cuh"
#include "devices/cuda/fastllm-mxfp8-activation.cuh"
#include "naive-n05-dsa-fused-attention.cuh"
#include "naivert/attn_dsa/body.cuh"
#include "naive-n05-dsa-verify-primitives.cuh"
#include "naive-n05-exact-qk.cuh"
#include "utils/utils.h"
#include <unordered_map>
#include <cmath>

namespace fastllm_dsa_layer::dp_verify {
using Core = fastllm_dsa_fused::Body<false, false>;
using Arena = Core::Scratch;
using ScoreSort = cub::BlockRadixSort<unsigned, Threads, 6, int, 4>;
namespace native = fastllm_dsa_fused::native_attention;
constexpr int Heads = 64, KvHeads = 4, KeyWidth = KvHeads * 192, ValueWidth = KvHeads * 128;
constexpr int QWidth = Heads * 192, PackedWidth = QWidth + KeyWidth + ValueWidth;

constexpr int IndexColumns = 128 + 2048 + 16;
constexpr int ProjectionTiles = IndexColumns / 16, ProjectionBlocks = (ProjectionTiles + 1) / 2;
constexpr size_t PartialBytes = size_t(8 * Heads / native::Heads) * native::Scratch * sizeof(float);
constexpr size_t ScoreFloats = 8 * Heads * 2048;
constexpr size_t ExactBytes = (ScoreFloats + 8 * Heads * 32 * 128) * sizeof(float);
constexpr size_t Ik = std::max({size_t(Arena::kBytes), PartialBytes, ExactBytes});
constexpr size_t Iq = Ik + 8 * 128 * sizeof(B);
constexpr size_t Iw = Iq + 8 * 2048 * sizeof(B), Query = Iw + 8 * 16 * sizeof(B);
constexpr size_t Bytes = Query + 8 * QWidth * sizeof(B);
constexpr size_t SelectionBytes = 8 * 2048 * sizeof(int);
// Same reserve as SplitDSAAttention. Both paths can reuse a previously
// captured allocation, including when the context bucket changes.
constexpr int FallbackElements = FastllmNaiveDSABatchLimit * (Heads / native::Heads) * native::Scratch;
static_assert(SelectionBytes + Bytes <= FallbackElements * sizeof(float));
struct Input {
    const B *normed, *qkv, *wk, *wq, *weight;
    const float *norm, *bias, *positions;
    float theta, valueScale;
    int *selected;
    FastllmMxfp8Activation packed;
    fastllm_dsa_fused::Args core;
};
union Shared103 {
    ProjectionStorage<8> projection;
    ScoreSort::TempStorage radix;
    Core::SharedStorage core;
    native::StorageFor<256, 16> attention;
};
// The owner has full-width heads. Fit the cooperative attention tile in the
// workstation GPU's 99 KiB opt-in limit and score its E4M3 sidecar with MMA.
template<bool Fp8> union Shared120 {
    ProjectionStorage<8> projection;
    ScoreSort::TempStorage radix;
    Core::TopKBlockRadixSort::TempStorage select;
    struct {
        B keys[128][200], query[8][200];
        int physical[128];
    } qk;
    struct {
        B values[64][136], prob[16][72];
        int physical[64];
    } pv;
    float normalize[258];
    std::conditional_t<Fp8, unsigned, native::StorageFor<128, 8>> split;
    float2 dots[16][16][4];
};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
constexpr bool Workstation = true;
#else
constexpr bool Workstation = false;
#endif

template<bool Fp8KV> __device__ void ScoreOwner(const Input &a, Shared120<Fp8KV> &s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    const auto &core = a.core;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32, group = lane / 4, pair = lane % 4;
    const auto *query = core.workspace + Arena::kIqPackedOffset;
    const auto *scales = reinterpret_cast<const float *>(core.workspace + Arena::kIndexWeightsScaledOffset);
    auto *logits = reinterpret_cast<float *>(core.workspace + Arena::kIndexLogitsOffset);
    const auto *lengths = reinterpret_cast<const int *>(core.workspace + Arena::kRadixControlOffset);
    const auto r = FastllmNaiveResolveDSARow(core.rows[0]);
    const int count = lengths[8];
    auto key = [&](int token) {
        const size_t physical = FastllmNaivePhysicalRow(r, min(token, count - 1));
        if (r.keyStride == KeyWidth)
            return static_cast<const unsigned char *>(r.cache[4]) + physical * 128 * (Fp8KV ? 1 : 2);
        return static_cast<const unsigned char *>(r.key) + (physical * r.keyStride + KeyWidth) * (Fp8KV ? 1 : 2);
    };
    for (int base = blockIdx.x * 16; base < count; base += gridDim.x * 16) {
        const auto *lo = reinterpret_cast<const unsigned *>(key(base + group));
        const auto *hi = reinterpret_cast<const unsigned *>(key(base + group + 8));
        for (int h = warp; h < 16; h += Threads / 32) {
            const auto *q = reinterpret_cast<const unsigned *>(query +
                            (group * 16 + h) * 128 * (Fp8KV ? 1 : 2));
            float v[4]{};
#pragma unroll
            for (int d = 0; d < 128; d += 16) {
                if constexpr (Fp8KV) {
                    const int i = d / 4 + pair;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
                                 "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                                 : "+f"(v[0]), "+f"(v[1]), "+f"(v[2]), "+f"(v[3])
                                 : "r"(lo[i]), "r"(hi[i]), "r"(q[i]));
                } else {
                    const int i = d / 2 + pair;
                    const unsigned ka[4]{lo[i], hi[i], lo[i + 4], hi[i + 4]};
                    const unsigned qb[2]{q[i], q[i + 4]};
                    native::Mma(v, ka, qb);
                }
            }
            s.dots[h][group][pair] = make_float2(v[0], v[1]);
            s.dots[h][group + 8][pair] = make_float2(v[2], v[3]);
        }
        __syncthreads();
        const int row = t % 8, token = base + t / 8;
        if (t < 128 && token < count) {
            float sum = 0.f;
            if (row < core.activeRows && token < *core.rows[row].liveKeys) {
#pragma unroll
                for (int h = 0; h < 16; ++h) {
                    const float dot = reinterpret_cast<const float *>(s.dots[h][t / 8])[row];
                    sum = fmaf(fmaxf(dot * scales[row * 16 + h], 0.f), float(core.weights[row * 16 + h]), sum);
                }
            }
            logits[size_t(row) * Arena::kMaxSeqLen + token] = sum;
        }
        __syncthreads();
    }
    cooperative_groups::this_grid().sync();
    Core::select_per_query_topk_grid(logits, reinterpret_cast<int *>(core.workspace + Arena::kIndexSelectsOffset),
        reinterpret_cast<unsigned *>(core.workspace + Arena::kTopKGridOffset), lengths,
        reinterpret_cast<unsigned *>(&s), &s.select);
#endif
}
// Preserve the SM120 sparse decoder's global BF16 softmax and 32 ordered
// 64-token PV partials. These stages reuse the Indexer arena after selection.
__device__ __forceinline__ void ExactFp8Attention(const Input &a, Shared120<true> &s, const B *query) {
    const auto &c = a.core;
    auto *scores = reinterpret_cast<float *>(c.workspace), *parts = scores + ScoreFloats;
    const int t = threadIdx.x, w = t / 32, l = t % 32;
    auto grid = cooperative_groups::this_grid();
    for (int task = blockIdx.x; task < c.activeRows * (Heads / 8) * 16; task += gridDim.x) {
        const int row = task / (16 * (Heads / 8)), group = task / 16 % (Heads / 8), first = task % 16 * 128;
        const auto r = FastllmNaiveResolveDSARow(c.rows[row]);
        if (*r.liveKeys <= 0)
            continue;
        if (t < 128) {
            const int token = a.selected[row * 2048 + first + t];
            s.qk.physical[t] = token >= 0 ? FastllmNaivePhysicalRow(r, token) : -1;
        }
        for (int i = t; i < 8 * 24; i += blockDim.x)
            *reinterpret_cast<uint4 *>(&s.qk.query[i / 24][i % 24 * 8]) =
                reinterpret_cast<const uint4 *>(query + (row * Heads + group * 8) * 192)[i];
        __syncthreads();
        for (int i = t; i < 128 * 24; i += blockDim.x) {
            const int physical = s.qk.physical[i / 24], d = i % 24 * 8;
            const auto key =
                FastllmNaiveKVRow(r.key, true) + size_t(max(physical, 0)) * r.keyStride + group / 2 * 192 + d;
            *reinterpret_cast<uint4 *>(&s.qk.keys[i / 24][d]) = physical >= 0 ? key.Load8() : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (w < 8) {
            naive_exact_qk::Scores<native::Load4, native::Load2, native::Mma>(
                s.qk.query, s.qk.keys, w, l, [&](int k) { return s.qk.physical[k] >= 0; },
                [&](int h, int k, float value) { scores[(row * Heads + group * 8 + h) * 2048 + first + k] = value; });
        }
        __syncthreads();
    }
    grid.sync();
    for (int h = blockIdx.x; h < c.activeRows * Heads; h += gridDim.x) {
        if (*c.rows[h / Heads].liveKeys <= 0)
            continue;
        float *logits = scores + size_t(h) * 2048;
        naive_dsa_decode::Normalize(logits, logits, c.sink, h % Heads, 2048, s.normalize, s.normalize[256],
                                    s.normalize[257], t);
        __syncthreads();
    }
    grid.sync();
    for (int task = blockIdx.x; task < c.activeRows * (Heads / 16) * 32; task += gridDim.x) {
        const int row = task / ((Heads / 16) * 32), group = task / 32 % (Heads / 16), part = task % 32;
        const auto r = FastllmNaiveResolveDSARow(c.rows[row]);
        if (*r.liveKeys <= 0)
            continue;
        if (t < 64) {
            const int token = a.selected[row * 2048 + part * 64 + t];
            s.pv.physical[t] = token >= 0 ? FastllmNaivePhysicalRow(r, token, true) : -1;
        }
        for (int i = t; i < 16 * 64; i += blockDim.x)
            s.pv.prob[i / 64][i % 64] = B(scores[(row * Heads + group * 16 + i / 64) * 2048 + part * 64 + i % 64]);
        __syncthreads();
        for (int i = t; i < 64 * 16; i += blockDim.x) {
            const int physical = s.pv.physical[i / 16], d = i % 16 * 8;
            const auto v = FastllmNaiveKVRow(r.value, true) + size_t(max(physical, 0)) * ValueWidth + group * 128 + d;
            *reinterpret_cast<uint4 *>(&s.pv.values[i / 16][d]) = physical >= 0 ? v.Load8() : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (t < 256) {
            const int h = t / 16, col = t % 16 * 8;
            float sums[8]{};
            for (int k = 0; k < 64; ++k) {
                float p = float(s.pv.prob[h][k]);
                uint4 v = *reinterpret_cast<const uint4 *>(&s.pv.values[k][col]);
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    sums[i] = fmaf(p, float(reinterpret_cast<B *>(&v)[i]), sums[i]);
            }
            float *out = parts + (((row * Heads + group * 16 + h) * 32 + part) * 128 + col);
            *reinterpret_cast<float4 *>(out) = make_float4(sums[0], sums[1], sums[2], sums[3]);
            *reinterpret_cast<float4 *>(out + 4) = make_float4(sums[4], sums[5], sums[6], sums[7]);
        }
        __syncthreads();
    }
    grid.sync();
    for (int tile = blockIdx.x; tile < c.activeRows * Heads; tile += gridDim.x)
        if (t < 128) {
            float sum = 0;
            if (*c.rows[tile / Heads].liveKeys > 0) {
#pragma unroll
                for (int k = 0; k < 32; ++k)
                    sum += parts[(tile * 32 + k) * 128 + t];
            }
            const B rounded[1]{B(sum)};
            c.output[tile * 128 + t] = rounded[0];
            fastllm_mxfp8::StoreGroup(a.packed, tile / Heads, tile % Heads * 128 + t, rounded);
        }
}
template<bool Fp8KV> __global__ __launch_bounds__(Threads, 1) void DpDSAVerify(const __grid_constant__ Input a) {
    extern __shared__ __align__(1024) unsigned char memory[];
    using Shared = std::conditional_t<Workstation, Shared120<Fp8KV>, Shared103>;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ != 1200
    constexpr int GroupHeads = 16, Groups = Heads / GroupHeads;
#endif
    auto &s = *reinterpret_cast<Shared *>(memory);
    const auto &core = a.core;
    auto *ik = reinterpret_cast<B *>(core.workspace + Ik);
    auto *iq = reinterpret_cast<B *>(core.workspace + Iq);
    auto *iw = reinterpret_cast<B *>(core.workspace + Iw);
    auto *query = reinterpret_cast<B *>(core.workspace + Query);
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    auto grid = cooperative_groups::this_grid();
    // Padded DP owners still replay the same node. Do not fetch model or KV
    // data when every candidate is inactive.
    bool active = false;
    for (int i = 0; i < core.activeRows; ++i) active |= *core.rows[i].liveKeys > 0;
    cudaTriggerProgrammaticLaunchCompletion();
    if (!active) {
        cudaGridDependencySynchronize();
        for (int i = blockIdx.x * Threads + t; i < core.activeRows * 2048; i += gridDim.x * Threads)
            a.selected[i] = -1;
        for (int i = blockIdx.x * Threads + t; i < core.activeRows * Heads * 128 / 8;
             i += gridDim.x * Threads) {
            reinterpret_cast<uint4 *>(core.output)[i] = make_uint4(0, 0, 0, 0);
            fastllm_mxfp8::StoreZero8(a.packed, i * 8 / (Heads * 128), i * 8 % (Heads * 128));
        }
        return;
    }
    Core::initialize(core);
    // Populate all scorer lanes on short/partially live replays. Inactive
    // heads use a zero query and never enter the shortlist.
    for (int head = blockIdx.x * (Threads / 32) + warp; head < 8 * 16; head += gridDim.x * (Threads / 32)) {
        if (head / 16 >= core.activeRows || *core.rows[head / 16].liveKeys <= 0) {
            auto *dst = reinterpret_cast<B *>(core.workspace + Arena::kIqPackedOffset);
            for (int k = lane; k < 128; k += 32) {
                if constexpr (Workstation && Fp8KV) reinterpret_cast<unsigned char *>(dst)[head * 128 + k] = 0;
                else if constexpr (Workstation) dst[head * 128 + k] = B(0.f);
                else dst[Core::ImageOffset(head, k, 128) / 2] = B(0.f);
            }
            if (!lane) reinterpret_cast<float *>(core.workspace + Arena::kIndexWeightsScaledOffset)[head] = 0;
        }
    }
    if (blockIdx.x < ProjectionBlocks) {
        LoadProjectionInput<8, 4096>(a.normed, core.activeRows, s.projection.input);
        auto source = [&](int col) {
            if (col < 128) return a.wk + size_t(col) * 4096;
            if (col < 2176) return a.wq + size_t(col - 128) * 4096;
            return a.weight + size_t(col - 2176) * 4096;
        };
        auto emit = [&](int col, int row, float x) {
            if (row >= core.activeRows) return;
            if (col < 128)
                ik[row * 128 + col] = __float2bfloat16_rn(x);
            else if (col < 2176)
                iq[row * 2048 + col - 128] = __float2bfloat16_rn(x);
            else
                iw[row * 16 + col - 2176] = __float2bfloat16_rn(Round(x) * .25f);
        };
        fastllm_dsa_projection::project_rows_mma<8, 4096, 4, 2, 12>(0, ProjectionTiles, source, s.projection.input,
                                                                    s.projection.partial, emit);
    }
    // The Indexer uses normalized BF16 input. Main Q/K/V and the borrowed
    // O activation arena belong to the preceding full-width projection.
    cudaGridDependencySynchronize();
    // Projection of the Indexer and publication of main Q/K/V use disjoint data.
    for (int row = int(blockIdx.x) - ProjectionBlocks; row >= 0 && row < core.activeRows * 8; row += int(gridDim.x) - ProjectionBlocks) {
        const int rId = row / 8, group = row % 8;
        auto r = FastllmNaiveResolveDSARow(core.rows[rId]);
        const int visible = *r.liveKeys;
        if (visible > 0 && visible <= r.capacity) {
            float cs, sn;
            sincosf(a.positions[rId] * powf(a.theta, -2.f * lane / 64.f), &sn, &cs);
            cs = Round(cs);
            sn = Round(sn);
            const B *raw = a.qkv + rId * PackedWidth;
            if (warp < 8) {
                const int h = group * 8 + warp;
                RotateWarp(raw + h * 192, query + rId * QWidth + h * 192, 192, cs, sn);
            }
            if (!group) {
                auto key = FastllmNaiveKVWriteRow{const_cast<void *>(r.key), Fp8KV} + size_t(FastllmNaivePhysicalRow(r, visible - 1)) * r.keyStride;
                auto value = FastllmNaiveKVWriteRow{const_cast<void *>(r.value), Fp8KV} + size_t(FastllmNaivePhysicalRow(r, visible - 1, true)) * ValueWidth;
                if (warp < KvHeads) RotateWarp(raw + QWidth + warp * 192, key + warp * 192, 192, cs, sn);
                for (int col = t; col < ValueWidth; col += Threads)
                    value[col] =
                        __float2bfloat16_rn(float(raw[QWidth + KeyWidth + col]) * Round(a.valueScale));
            }
        }
    }
    grid.sync();
    if (blockIdx.x < core.activeRows) {
        const int row = blockIdx.x;
        auto r = FastllmNaiveResolveDSARow(core.rows[row]);
        const int n = *r.liveKeys;
        if (n > 0 && n <= r.capacity) {
            NormalizeIndex(ik + row * 128, a.norm, a.bias);
            float cs, sn;
            sincosf(a.positions[row] * powf(a.theta, -2.f * lane / 64.f), &sn, &cs);
            cs = Round(cs);
            sn = Round(sn);
            if (!warp) {
                const int physical = FastllmNaivePhysicalRow(r, n - 1);
                auto index = Workstation && r.keyStride == KeyWidth ?
                    FastllmNaiveKVWriteRow{const_cast<void *>(r.cache[4]), Fp8KV} + size_t(physical) * 128 :
                    FastllmNaiveKVWriteRow{const_cast<void *>(r.key), Fp8KV} + size_t(physical) * r.keyStride + KeyWidth;
                RotateWarp(ik + row * 128, index, 128, cs, sn);
            }
            // Rotate and pack directly into the tensor-core scorer image.
            auto *packed = reinterpret_cast<B *>(core.workspace + Arena::kIqPackedOffset);
            auto *scales = reinterpret_cast<float *>(core.workspace + Arena::kIndexWeightsScaledOffset);
            for (int h = warp; h < 16; h += Threads / 32) {
                const B *p = iq + row * 2048 + h * 128;
                float x = float(p[lane]), y = float(p[32 + lane]);
                const float vals[] = {Round(Round(x * cs) - Round(y * sn)), Round(Round(y * cs) + Round(x * sn)),
                                      float(p[64 + lane]), float(p[96 + lane])};
                float mx = 0;
                for (float v : vals) mx = fmaxf(mx, fabsf(v));
                for (int step = 16; step; step >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, step));
                const float scale = fmaxf(mx, 1e-4f) / 448.f;
                const int head = row * 16 + h;
                if (!lane) scales[head] = scale;
                for (int i = 0; i < 4; ++i) {
                    const auto value = __nv_fp8_e4m3(fmaxf(-448.f,fminf(448.f,vals[i] / scale)));
                    if constexpr (Workstation && Fp8KV)
                        reinterpret_cast<unsigned char *>(packed)[head * 128 + i * 32 + lane] = value.__x;
                    else if constexpr (Workstation) packed[head * 128 + i * 32 + lane] = B(float(value));
                    else packed[Core::ImageOffset(head, i * 32 + lane, 128) / 2] = __float2bfloat16(float(value));
                }
            }
        }
    }
    grid.sync();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    ScoreOwner<Fp8KV>(a, s);
#else
    Core::run<false>(core, s.core, true, true);
#endif
    // Core publishes its candidates before returning. Sort directly from that scratch.
    // The sharded selector emits exact members in token order. Stable radix
    // sorting preserves the baseline score order and lower-token tie break.
    if (blockIdx.x < core.activeRows) {
        const int row = blockIdx.x, count = min(2048, max(0, *core.rows[row].liveKeys));
        const auto *candidates = reinterpret_cast<const int *>(core.workspace + Arena::kIndexSelectsOffset) + row * 2048;
        auto *ids = a.selected + row * 2048;
        auto *logits =
            (const float *)(core.workspace + Arena::kIndexLogitsOffset) + size_t(row) * Arena::kMaxSeqLen;
        unsigned keys[6];
        int tokens[6];
        for (int i = 0; i < 6; ++i) {
            int slot = t * 6 + i, token = slot < count ? (Workstation && *core.rows[row].liveKeys <= 2048 ? slot : candidates[slot]) : -1;
            tokens[i] = token;
            keys[i] = token >= 0 ? Core::ordered_float_key(logits[token]) : 0;
        }
        ScoreSort(s.radix).SortDescending(keys, tokens);
        for (int i = 0; i < 6; ++i)
            if (t * 6 + i < 2048) ids[t * 6 + i] = tokens[i];
    }
    grid.sync();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    if constexpr (Fp8KV) {
        ExactFp8Attention(a, s, query);
    } else {
        // BF16 fallback uses split-local exp / denominator before its final
        // fold. Preserve that order, including the native MMA orientation.
        constexpr int groups = Heads / native::Heads;
        auto *completion = reinterpret_cast<unsigned *>(core.workspace + Ik);
        for (int i = blockIdx.x * Threads + t; i < core.activeRows * groups; i += gridDim.x * Threads)
            completion[i] = 0;
        grid.sync();
        for (int tile = blockIdx.x; tile < core.activeRows * groups * native::Splits; tile += gridDim.x) {
            const int work = tile / native::Splits, row = work / groups;
            auto descriptor = core.rows[row];
            descriptor.indices = a.selected + row * 2048;
            native::Attend<false>(query, descriptor, core.sink, core.output,
                reinterpret_cast<float *>(core.workspace), completion, Heads, KvHeads, row,
                work % groups, tile % native::Splits, s.split);
            __syncthreads();
        }
        grid.sync();
        for (int i = blockIdx.x * Threads + t; i < core.activeRows * Heads * 128 / 8; i += gridDim.x * Threads)
            fastllm_mxfp8::Store8(a.packed, i * 8 / (Heads * 128), i * 8 % (Heads * 128),
                                  reinterpret_cast<const uint4 *>(core.output)[i]);
    }
#else
    constexpr int splits = 8;
    for (int tile = blockIdx.x; tile < core.activeRows * Groups * splits; tile += gridDim.x) {
        const int work = tile / splits, r = work / Groups;
        auto descriptor = core.rows[r];
        descriptor.indices = a.selected + r * 2048;
        native::Attend<true, native::Phase::Produce, splits, 128, 256, true, false,
                       native::NoOutputEpilogue, GroupHeads>(
            query, descriptor, core.sink, core.output, (float *)core.workspace, nullptr, Heads, KvHeads, r,
            work % Groups, tile % splits, s.attention);
        __syncthreads();
    }
    grid.sync();
    // Each warp folds 64 adjacent V columns. Broadcast the eight softmax corrections
    // once per split and preserve the original FP32 accumulation and BF16 rounding.
    constexpr int segments = 2;
    static_assert(splits <= native::Splits && splits <= 32);
    const int warpIndex = blockIdx.x * (Threads / 32) + warp;
    for (int tile = warpIndex; tile < core.activeRows * Heads * segments; tile += gridDim.x * (Threads / 32)) {
        const int attentionHead = tile / segments, row = attentionHead / Heads;
        const int layerHead = attentionHead % Heads;
        const int head = attentionHead % GroupHeads, col = tile % segments * (128 / segments) + lane;
        const int block = attentionHead / GroupHeads;
        const auto *part = reinterpret_cast<const float *>(core.workspace) +
            size_t(block) * native::Splits * GroupHeads * (128 + 2);
        const auto *stats = part + native::Splits * GroupHeads * 128;
        const bool activeRow = *core.rows[row].liveKeys > 0;
        float maximum = -INFINITY, sum = 0.f;
        if (lane < splits && activeRow) {
            const float2 state = reinterpret_cast<const float2 *>(stats)[lane * GroupHeads + head];
            maximum = state.x; sum = state.y;
        }
        const float ownMax = maximum;
        for (int d = 16; d; d /= 2)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, d));
        maximum = __shfl_sync(0xffffffff, maximum, 0);
        const float sink = core.sink ? core.sink[layerHead] : -INFINITY;
        maximum = fmaxf(maximum, sink);
        const float correction = maximum == -INFINITY ? 0.f : __expf(ownMax - maximum);
        float denominator = 0.f, values[4 / segments] = {};
        if (activeRow) {
#pragma unroll
            for (int k = 0; k < splits; ++k) {
                const float c = __shfl_sync(0xffffffff, correction, k);
                denominator = fmaf(__shfl_sync(0xffffffff, sum, k), c, denominator);
#pragma unroll
                for (int n = 0; n < 4 / segments; ++n)
                    values[n] = fmaf(part[(k * GroupHeads + head) * 128 + col + 32 * n], c, values[n]);
            }
            if (core.sink) denominator += __expf(sink - maximum);
        }
#pragma unroll
        for (int n = 0; n < 4 / segments; ++n) {
            const B value = __float2bfloat16(denominator > 0 ? values[n] / denominator : 0.f);
            const int column = layerHead * 128 + col + n * 32;
            core.output[row * Heads * 128 + column] = value;
            const B rounded[1]{value};
            fastllm_mxfp8::StoreGroup(a.packed, row, column, rounded);
        }
    }
#endif
}

template<bool Fp8KV> int ResidentGrid(int device) {
    const int arch = FastllmCudaRuntimeArch();
    if ((arch != 103 || Fp8KV) && arch != 120) return 0;
    const size_t bytes = arch == 120 ? sizeof(Shared120<Fp8KV>) : sizeof(Shared103);
    static thread_local std::unordered_map<int, int> sizes;
    auto it = sizes.find(device);
    if (it != sizes.end()) return it->second;
    if (FastllmCudaGraphIsCapturingFast()) return 0;
    cudaDeviceProp prop{};
    int blocks = 0;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || !prop.cooperativeLaunch ||
        prop.multiProcessorCount < (arch == 120 ? ProjectionBlocks + 1 : 128) || prop.sharedMemPerBlockOptin < bytes ||
        cudaFuncSetAttribute(DpDSAVerify<Fp8KV>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, DpDSAVerify<Fp8KV>, Threads, bytes) !=
            cudaSuccess ||
        !blocks) {
        cudaGetLastError();
        return sizes[device] = 0;
    }
    return sizes[device] = prop.multiProcessorCount;
}
} // namespace fastllm_dsa_layer::dp_verify

bool FastllmCudaTryDpDSAVerify(const fastllm::Data &normalized, const fastllm::Data &qkv,
                               const fastllm::Data &wk, const fastllm::Data &wq, const fastllm::Data &weights,
                               const fastllm::Data &keyNorm, const fastllm::Data &keyBias,
                               const fastllm::Data &sink, const fastllm::Data &positions,
                               const fastllm::Data &live, const fastllm::Data &key,
                               const fastllm::Data &value, const void *const *binding, int capacity,
                               float theta, float valueScale, FastllmMxfp8Activation packed,
                               FastllmNaiveDecodeScratch &scratch, fastllm::Data &attention, bool overlapQkv) {
    using namespace fastllm;
    using namespace fastllm_dsa_layer::dp_verify;
    using fastllm_dsa_layer::B;
    const int rows = packed.rows, device = FastllmCudaGetDevice();
    const bool workstation = FastllmCudaRuntimeArch() == 120;
    const bool fp8 = key.dataType == FP8_E4M3;
    const int stride = key.dims.empty() ? 0 : key.dims.back();
    const bool sidecar = workstation && stride == KeyWidth && key.isPagedKVCache;
    if ((fp8 && !workstation) || (!sidecar && stride != KeyWidth + 128)) return false;
    if (rows < 2 || rows > 8 || !packed.Matches(rows, Heads * 128) || !binding || capacity <= 8192 ||
        capacity > Arena::kMaxSeqLen || !std::isfinite(theta) || theta <= 0 || !std::isfinite(valueScale))
        return false;
    auto resident = [&](const Data &d, DataType type) {
        return d.dataDevice == CUDA && d.dataType == type && d.cudaData &&
               d.dataDeviceIds == std::vector<int>{device} && uintptr_t(d.cudaData) % 16 == 0;
    };
    auto dense = [&](const Data &d, DataType type, size_t count, int width) {
        if (!resident(d, type) || d.Count(0) != count || d.dims.empty() || d.dims.back() != width ||
            d.dims.size() != d.strides.size())
            return false;
        size_t stride = 1;
        for (int i = int(d.dims.size()) - 1; i >= 0; --i) {
            if (d.strides[i] != stride) return false;
            stride *= d.dims[i];
        }
        return true;
    };
    auto cache = [&](const Data &d, int width) {
        if (!resident(d, fp8 ? FP8_E4M3 : BFLOAT16) || d.dims.size() != 3 || d.dims[0] != 1 || d.dims[2] != width)
            return false;
        if (d.isPagedKVCache)
            return d.expansionDims.size() == 3 && d.expansionDims[1] >= capacity && d.pagedKVCacheData &&
                   d.pagedKVCacheData->cudaData && d.pageLen > 0 && d.pageLen % 8 == 0 &&
                   d.pagedKVCacheData->maxPages > 0;
        return d.expansionBytes >= size_t(capacity) * width * (fp8 ? 1 : sizeof(B));
    };
    if (!dense(normalized, BFLOAT16, size_t(rows) * 4096, 4096) ||
        !dense(qkv, BFLOAT16, size_t(rows) * PackedWidth, PackedWidth) ||
        !dense(wk, BFLOAT16, 128 * 4096, 4096) || !dense(wq, BFLOAT16, 2048 * 4096, 4096) ||
        !dense(weights, BFLOAT16, 16 * 4096, 4096) || !dense(keyNorm, FLOAT32, 128, 128) ||
        !dense(keyBias, FLOAT32, 128, 128) || (!sink.dims.empty() && !dense(sink, FLOAT32, Heads, Heads)) ||
        !dense(positions, FLOAT32, rows, rows) || !dense(live, INT32, rows, rows) ||
        !cache(key, stride) || !cache(value, ValueWidth) || key.kvCacheFp8Indexer ||
        key.isPagedKVCache != value.isPagedKVCache || (key.isPagedKVCache && key.pageLen != value.pageLen))
        return false;
    const int grid = fp8 ? ResidentGrid<true>(device) : ResidentGrid<false>(device);
    auto &arena = scratch.attentionPartial;
    if (!grid || (FastllmCudaGraphIsCapturingFast() &&
                  (!arena.cudaData || arena.expansionBytes < FallbackElements * sizeof(float) ||
                   !attention.cudaData || attention.expansionBytes < size_t(rows) * Heads * 128 * sizeof(B))))
        return false;
    arena.dataType = FLOAT32;
    arena.UpdateUnitSize();
    arena.Resize({FastllmNaiveDSABatchLimit, Heads / native::Heads, native::Scratch});
    arena.ToDevice(CUDA, normalized.dataDeviceIds, false);
    arena.Allocate(false);
    attention.dataType = BFLOAT16;
    attention.UpdateUnitSize();
    attention.Resize({1, rows, Heads * 128});
    attention.ToDevice(CUDA, normalized.dataDeviceIds, false);
    attention.Allocate(false);
    Input a{};
    a.selected = static_cast<int *>(arena.cudaData);
    a.normed = (const B *)normalized.cudaData;
    a.qkv = (const B *)qkv.cudaData;
    a.wk = (const B *)wk.cudaData;
    a.wq = (const B *)wq.cudaData;
    a.weight = (const B *)weights.cudaData;
    a.norm = (const float *)keyNorm.cudaData;
    a.bias = (const float *)keyBias.cudaData;
    a.positions = (const float *)positions.cudaData;
    a.theta = theta;
    a.valueScale = valueScale;
    a.packed = packed;
    auto &c = a.core;
    c.workspace = static_cast<unsigned char *>(arena.cudaData) + SelectionBytes;
    c.indexQuery = (B *)(c.workspace + Iq);
    c.weights = (B *)(c.workspace + Iw);
    c.activeRows = rows;
    c.output = (B *)attention.cudaData;
    c.sink = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    c.indexStride = stride;
    c.indexKey = FastllmNaiveKV(key);
    c.indexCache = binding;
    for (int i = 0; i < rows; ++i) {
        auto &d = c.rows[i];
        d.cache = binding;
        d.capacity = capacity;
        d.keyStride = c.indexStride;
        d.pageLen = key.isPagedKVCache ? key.pageLen : 0;
        d.liveKeys = (const int *)live.cudaData + i;
        d.selectedCount = 2048;
        d.indices = nullptr; // Sort the published candidates directly into the attention payload.
    }
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeCooperative;
    attrs[0].val.cooperative = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = fastllm_dsa_layer::Threads;
    cfg.dynamicSmemBytes = workstation ? (fp8 ? sizeof(Shared120<true>) : sizeof(Shared120<false>)) : sizeof(Shared103);
    cfg.stream = cudaStreamPerThread;
    cfg.attrs = attrs;
    cfg.numAttrs = overlapQkv ? 2 : 1;
    auto error = cudaLaunchKernelEx(&cfg, fp8 ? DpDSAVerify<true> : DpDSAVerify<false>, a);
    AssertInFastLLM(error == cudaSuccess,
                    std::string("DP DSA verification launch failed: ") + cudaGetErrorString(error));
    return true;
}
