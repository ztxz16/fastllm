#pragma once

// Included beside the TP8 collective primitives. The grid size is bounded by the
// actual residency on each GPU. No producer spins on an unscheduled block.
namespace hopper_verify {
using BF = __nv_bfloat16;
constexpr int Threads = 512, Heads = 8, QDim = 192, VDim = 128;
constexpr int ShardWidth = 1024;
constexpr int QkvTiles = 1856 / 16;
constexpr int RouterPairBytes = 8192 * sizeof(float);
constexpr int OutStageBytes = 16 * (1024 + 32 * sizeof(float));

constexpr int RingValueTile = 32, RingTilesPerRow = VDim / RingValueTile;
// Wide graphs retain immutable row metadata. KV pool and page pointers are
// resolved through the scheduler's live cache table on every replay.
struct RowBinding {
    const void *const *cache;
    const int *live;
    int capacity, pageLen, offset, fp8;
    static RowBinding From(const FastllmNaiveDSARow &r) {
        return {r.cache, r.liveKeys, r.capacity, r.pageLen, r.verifyOffset, r.fp8};
    }
};
static_assert(sizeof(RowBinding) == 2 * sizeof(void *) + 4 * sizeof(int));
struct BindingTable {
    std::vector<RowBinding> rows;
    fastllm::Data device{fastllm::INT8};
};
struct Args {
    const RowBinding *external;
    RowBinding local[FastllmNaiveDSABatchLimit];
    const BF *input;
    const void *qkvWeight, *outWeight;
    const float *qkvScales, *outScales, *pos, *sink, *gamma;
    BF *query, *attn, *projection, *hidden, *normed;
    const float *routerWeight;
    float *probabilities;
    uint8_t *packed;
    float *scales, *scores, *partials;
    CustomArPointers peers;
    CustomArRankSignals signals;
    TaggedPeerView transport;
    bool stageWeights;
    const BF *iqWeight, *ikWeight, *iwWeight;
    const float *indexNorm, *indexBias;
    BF *indexRaw;
    uint8_t *iqBytes;
    float *iqScale;
    unsigned *indexScores;
    unsigned long long *lists;
    int *selected;
    int rows, rank, stride, signalSlots, indexCapacity;
    float theta, valueScale, eps;

    __device__ __forceinline__ FastllmNaiveDSARow Row(int i) const {
        const RowBinding b = external ? external[i] : local[i];
        FastllmNaiveDSARow r{};
        r.cache = b.cache;
        r.liveKeys = b.live;
        r.capacity = b.capacity;
        r.keyStride = indexCapacity ? QDim + 128 : QDim;
        r.pageLen = b.pageLen;
        r.verifyOffset = b.offset;
        r.fp8 = b.fp8;
        r.indices = selected ? selected + i * stride : nullptr;
        r.selectedCount = selected ? stride : 0;
        return r;
    }
};

// A staged QKV warp reads the same K32 group from eight verification rows.
// Keep the counts compact for short batches and put those rows next to each other.
template <bool GroupMajor> __device__ void Pack(const BF *input, int width, const Args &a) {
    const int lane = threadIdx.x % 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.rows * width / 8; i += gridDim.x * blockDim.x) {
        uint4 raw = reinterpret_cast<const uint4 *>(input)[i];
        auto *v = reinterpret_cast<BF *>(&raw);
        float maximum = 1.e-4f;
#pragma unroll
        for (int d = 0; d < 8; ++d)
            maximum = fmaxf(maximum, fabsf(float(v[d])));
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1, 4));
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2, 4));
        const float scale = __uint_as_float(fastllm_mxfp8::ScaleExponent(maximum) << 23);
        const int srcGroup = i / 4;
        const int dstGroup = GroupMajor ? (srcGroup % (width / 32)) * a.rows + i / (width / 8) : srcGroup;
        if (!lane)
            a.scales[dstGroup] = scale;
        uint2 bytes;
        auto *out = reinterpret_cast<__nv_fp8x2_storage_t *>(&bytes);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            if constexpr (GroupMajor) {
                const float inverse = __frcp_rn(scale);
                out[d] = __nv_cvt_float2_to_fp8x2(make_float2(float(v[d * 2]) * inverse, float(v[d * 2 + 1]) * inverse),
                                                  __NV_SATFINITE, __NV_E4M3);
            } else {
                out[d] = __nv_cvt_float2_to_fp8x2(make_float2(float(v[d * 2]) / scale, float(v[d * 2 + 1]) / scale),
                                                  __NV_SATFINITE, __NV_E4M3);
            }
        }
        reinterpret_cast<uint2 *>(a.packed)[GroupMajor ? dstGroup * 4 + lane : i] = bytes;
    }
}

// One warp owns one K32 output block. Publish its rounded BF16 value and
// native FP8 representation together so the O projection can read it directly.
template <bool Ring = false, bool GroupMajor = false> __device__ __forceinline__ void StoreOutput(const Args &a, int i, float x) {
    BF rounded = __float2bfloat16_rn(x);
    a.attn[i] = rounded;
    float maximum = fmaxf(1.e-4f, fabsf(float(rounded)));
    for (int d = 16; d; d /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, d));
    const float scale = __uint_as_float(fastllm_mxfp8::ScaleExponent(maximum) << 23);
    const int group = (GroupMajor || (Ring && a.transport.epochs && a.rows <= 8)) ? (i % 1024 / 32) * a.rows + i / 1024 : i / 32;
    if (threadIdx.x % 32 == 0)
        a.scales[group] = scale;
    if constexpr (Ring)
        a.packed[group * 32 + i % 32] = __nv_fp8_e4m3(float(rounded) * __frcp_rn(scale)).__x;
    else
        a.packed[i] = __nv_fp8_e4m3(float(rounded) / scale).__x;
}
template <int Width>
__device__ __forceinline__ fastllm_mxfp8_projection::WeightFragment
LoadStagedWeights(const uint8_t *tile, int row, int k, int group) {
    const int col = k ^ (row * 32);
    const float *scales = reinterpret_cast<const float *>(tile + 16 * Width);
    return {*reinterpret_cast<const uint2 *>(tile + row * Width + col),
            *reinterpret_cast<const uint2 *>(tile + (row + 8) * Width + col),
            scales[row * (Width / 32) + group], scales[(row + 8) * (Width / 32) + group]};
}

template <bool Front, bool Ring, bool Staged = false, int Groups = 1, bool Raw = false>
__device__ void Project(const Args &a, float (&part)[16][128], const uint8_t *stage = nullptr) {
    constexpr int Width = Front ? 4096 : 1024, Tiles = Front ? QkvTiles : 256;
    const int t = threadIdx.x, warp = t / 32, g = t % 32 / 4, lane = t % 4;
    if constexpr (Groups > 1 && Staged) {
        if (blockIdx.x >= Tiles) return;
    }
    constexpr bool Wide = Groups > 1;
    const int stepTiles = Wide && Staged ? Tiles : int(gridDim.x);
    for (int task = blockIdx.x; task < (Wide ? Tiles : ((a.rows + 7) / 8) * Tiles); task += stepTiles) {
        const int tile = task % Tiles, baseRow = task / Tiles * 8;
        const int lo = Front ? fastllm_swa_mma::QkvColumn(tile, g) : tile * 16 + g;
        const int hi = Front ? fastllm_swa_mma::QkvColumn(tile, g + 8) : lo + 8;
        float acc[Groups][4]{};
#pragma unroll
        for (int step = 0; step < Width / 512; ++step) {
            const int group = warp + step * 16;
            const int k = group * 32 + lane * 8;
            fastllm_mxfp8_projection::WeightFragment w;
            if constexpr (Staged) {
                w = LoadStagedWeights<Width>(stage, g, k, group);
            } else if constexpr (!Front && Groups == 4) {
                const int slot = task / gridDim.x;
                w = stage && slot < 2 ? LoadStagedWeights<Width>(stage + slot * OutStageBytes, g, k, group)
                                      : fastllm_mxfp8_projection::LoadWeights(a.outWeight, a.outScales, Width, lo, hi, k);
            } else {
                w = fastllm_mxfp8_projection::LoadWeights(Front ? a.qkvWeight : a.outWeight,
                                                          Front ? a.qkvScales : a.outScales, Width, lo, hi, k);
            }
            // Keep each eight-row group's accumulators live while the weight fragment is in registers.
#pragma unroll
            for (int half = 0; half < Groups; ++half) {
                const int row = baseRow + half * 8 + g;
                uint2 x{};
                float scale = 0;
                if (row < a.rows) {
                    if constexpr (Staged || (!Front && Groups == 4)) {
                        const int offset = group * a.rows + row;
                        x = *reinterpret_cast<const uint2 *>(a.packed + offset * 32 + lane * 8);
                        scale = a.scales[offset];
                    } else {
                        x = *reinterpret_cast<const uint2 *>(a.packed + row * Width + k);
                        scale = a.scales[row * (Width / 32) + group];
                    }
                }
                fastllm_mxfp8_projection::Accumulate(acc[half], w, x, __shfl_sync(0xffffffff, scale, lane * 8),
                                                     __shfl_sync(0xffffffff, scale, lane * 8 + 4));
            }
        }
#pragma unroll
        for (int half = 0; half < Groups; ++half) {
            const int first = baseRow + half * 8;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                part[warp][(g + i / 2 * 8) * 8 + lane * 2 + i % 2] = acc[half][i];
            __syncthreads();
            if (t < 128) {
                float sum = 0;
#pragma unroll
                for (int w = 0; w < 16; ++w)
                    sum += part[w][t];
                part[0][t] = CustomArNormRound(sum);
            }
            __syncthreads();
            if (t < 128 && first + t % 8 < a.rows) {
                const int r = first + t % 8, oc = t / 8;
                const int col = Front ? fastllm_swa_mma::QkvColumn(tile, oc) : tile * 16 + oc;
                float val = part[0][t];
                if constexpr (Raw) {
                    a.projection[r * 1856 + col] = __float2bfloat16_rn(val);
                } else if constexpr (!Front) {
                    a.projection[r * 4096 + col] = __float2bfloat16_rn(val);
                    if constexpr (Groups >= 2) {
                        if (!(oc & 1)) {
                            const auto step = *reinterpret_cast<const uint64_t *>(a.scores);
                            const size_t frame = (step % kTaggedPeerBuffers) * a.transport.frameBytes;
                            const auto pair = __floats2bfloat162_rn(val, part[0][t + 8]);
                            const uint64_t payload = (uint64_t(*reinterpret_cast<const unsigned *>(&pair)) << 32) |
                                (uint32_t(step) + 1);
                            if constexpr (Groups == 4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
                                const int word = col / 2;
                                const size_t offset = frame + a.rank * a.transport.slotBytes + size_t(r * 256 + word % 256) * 8;
                                asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::
                                    "l"(a.transport.peers.data[word / 256] + offset), "l"(payload) : "memory");
#endif
                            } else {
                                a.transport.Publish(frame + a.rank * a.transport.slotBytes + size_t(r * 2048 + col / 2) * 8, payload);
                            }
                        }
                    }
                } else {
                    auto desc = FastllmNaiveResolveDSARow(a.Row(r));
                    const int live = *desc.liveKeys;
                    if (tile < 36) {
                        const int d = tile % 4 * 8 + oc % 8;
                        const float other = part[0][t ^ 64];
                        float c, s;
                        if constexpr (Ring) {
                            auto angle = reinterpret_cast<const float2 *>(a.scores + 2)[r * 32 + d];
                            c = angle.x;
                            s = angle.y;
                        } else {
                            const float angle = a.pos[r] * powf(a.theta, -2.f * d / 64.f);
                            c = CustomArNormRound(cosf(angle));
                            s = CustomArNormRound(sinf(angle));
                        }
                        val = CustomArNormRound(val * c) + (oc < 8 ? -1.f : 1.f) * CustomArNormRound(other * s);
                    }
                    if (col < Heads * QDim)
                        a.query[r * Heads * QDim + col] = __float2bfloat16_rn(live > 0 ? val : 0.f);
                    else if (live > 0 && live <= desc.capacity) {
                        const bool value = col >= 1728;
                        const int dst =
                            Ring ? 128 + desc.verifyOffset : FastllmNaivePhysicalRow(desc, live - 1, value);
                        FastllmNaiveKVWriteRow p{const_cast<void *>(value ? desc.value : desc.key), desc.fp8};
                        if (value)
                            p[dst * VDim + col - 1728] = __float2bfloat16_rn(val * a.valueScale);
                        else
                            p[dst * desc.keyStride + col - 1536] = __float2bfloat16_rn(val);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <bool Ring> __device__ int Token(const FastllmNaiveDSARow &r, int slot, int live) {
    if constexpr (Ring) {
        const int logical = max(0, live - 128) + slot, prefix = live - r.verifyOffset - 1;
        return logical >= prefix ? 128 + logical - prefix : logical % 128;
    }
    const int token = r.indices ? r.indices[slot] : slot;
    return token >= 0 && token < live ? token : -1;
}

__device__ int SelectedCount(const FastllmNaiveDSARow &r, int live) {
    return live > 0 ? (r.indices ? r.selectedCount : live) : 0;
}

// A warp computes sixteen selected keys against all eight GQA heads. Gather
// and decode each key once. The score and probability interfaces round to BF16.
__device__ void Scores(const Args &a) {
    const int lane = threadIdx.x % 32, g = lane / 4, quad = lane % 4;
    const int tiles = (a.stride + 15) / 16;
    for (int work = blockIdx.x * 16 + threadIdx.x / 32; work < a.rows * tiles; work += gridDim.x * 16) {
        const int row = work / tiles, first = work % tiles * 16;
        auto r = FastllmNaiveResolveDSARow(a.Row(row));
        const int live = max(0, min(r.capacity, *r.liveKeys)), count = SelectedCount(r, live);
        if (first >= count)
            continue;
        int low = first + g < count ? Token<false>(r, first + g, live) : -1;
        int high = first + 8 + g < count ? Token<false>(r, first + 8 + g, live) : -1;
        FastllmNaiveKV key(r.key, r.keyPages, r.pageLen, r.fp8, false);
        auto kl = key.Row(max(0, low), r.keyStride), kh = key.Row(max(0, high), r.keyStride);
        float acc[4]{};
#pragma unroll
        for (int k = 0; k < QDim; k += 32) {
            int d = k + quad * 8;
            uint4 x = *reinterpret_cast<const uint4 *>(a.query + (row * Heads + g) * QDim + d);
            uint4 l = low < 0 ? make_uint4(0, 0, 0, 0) : (kl + d).Load8();
            uint4 h = high < 0 ? make_uint4(0, 0, 0, 0) : (kh + d).Load8();
            fastllm_swa_mma::Step(acc, l, h, x);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int slot = first + g + j / 2 * 8;
            if (slot < count)
                a.scores[(row * Heads + quad * 2 + j % 2) * a.stride + slot] =
                    (j < 2 ? low : high) < 0 ? -INFINITY
                                             : CustomArNormRound(CustomArNormRound(acc[j]) * rsqrtf(192.f));
        }
    }
}

template <bool Wide> __device__ void ValueTiles(const Args &a, BF *values, float *probabilities) {
    constexpr int Tile = 64;
    const int parts = (a.stride + Tile - 1) / Tile, t = threadIdx.x;
    for (int task = blockIdx.x; task < a.rows * parts; task += gridDim.x) {
        int row = task / parts, p = task % parts, first = p * Tile;
        auto r = FastllmNaiveResolveDSARow(a.Row(row));
        int live = max(0, min(r.capacity, *r.liveKeys)), count = SelectedCount(r, live);
        FastllmNaiveKV v(r.value, r.valuePages, r.pageLen, r.fp8, false);
        for (int i = t; i < Tile * 16; i += Threads) {
            int slot = first + i / 16, col = i % 16 * 8;
            int k = slot < count ? Token<false>(r, slot, live) : -1;
            reinterpret_cast<uint4 *>(values)[i] =
                k < 0 ? make_uint4(0, 0, 0, 0) : (v.Row(k, 128) + col).Load8();
        }
        const int h = t / 64, slot = first + t % 64;
        probabilities[t] = slot < count ? a.scores[(row * 8 + h) * a.stride + slot] : 0.f;
        __syncthreads();
        if constexpr (Wide) {
            const int head = t / 64, pair = t % 64;
            float2 sum{};
#pragma unroll 8
            for (int i = 0; i < Tile; ++i) {
                const float2 value = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(values)[i * 64 + pair]);
                const float probability = probabilities[head * Tile + i];
                sum.x = fmaf(probability, value.x, sum.x);
                sum.y = fmaf(probability, value.y, sum.y);
            }
            reinterpret_cast<float2 *>(a.partials)[(task * Heads + head) * (VDim / 2) + pair] = sum;
        } else {
            const int head = t / 128, col = t % 128;
            float lo = 0.f, hi = 0.f;
#pragma unroll 8
            for (int i = 0; i < Tile; ++i) {
                float val = float(values[i * 128 + col]);
                lo = fmaf(probabilities[head * Tile + i], val, lo);
                hi = fmaf(probabilities[(head + 4) * Tile + i], val, hi);
            }
            a.partials[(task * Heads + head) * VDim + col] = lo;
            a.partials[(task * Heads + head + 4) * VDim + col] = hi;
        }
        __syncthreads();
    }
}

#include "fastllm-naive-sm90-swa.cuh"

} // namespace hopper_verify
#include "fastllm-naive-sm90-indexer.cuh"
namespace hopper_verify {
// Four 128-thread groups share each CTA. Each group implements the same
// ordered FP32 tree as BatchRouterSigmoid4096Vector. An eight-row tile
// amortizes the FP32 expert weight loads over more verified tokens.
template <int Rows, bool TileInput = false> __device__ void RouteTiles(const Args &a, float *scratch, const BF *tileInput = nullptr) {
    const int group = threadIdx.x / 128, t = threadIdx.x % 128;
    for (int block = blockIdx.x; block < ((a.rows + Rows - 1) / Rows) * 64; block += gridDim.x) {
        const int task = block * 4 + group, expert = task % 256, first = task / 256 * Rows;
        float2 partial[Rows] = {};
        #pragma unroll 4
        for (int k = 0; k < 16; ++k) {
            const int c = t * 2 + k * 256;
            const float2 w = *reinterpret_cast<const float2 *>(a.routerWeight + expert * 4096 + c);
            #pragma unroll
            for (int row = 0; row < Rows; ++row) {
                if (first + row >= a.rows) continue;
                float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(
                    (TileInput ? tileInput + row * 4096 : a.normed + (first + row) * 4096) + c));
                partial[row].x = fmaf(x.x, w.x, partial[row].x);
                partial[row].y = fmaf(x.y, w.y, partial[row].y);
            }
        }
        float *tmp = scratch + group * Rows * 8;
        #pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float sum = partial[row].x + partial[row].y;
            #pragma unroll
            for (int d = 1; d < 16; d *= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
            if (t % 16 == 0) tmp[row * 8 + t / 16] = sum;
        }
        __syncthreads();
        const int row = t / (128 / Rows), lane = t % (128 / Rows);
        float sum = lane < 8 ? tmp[row * 8 + lane] : 0.f;
        #pragma unroll
        for (int d = 1; d < 8; d *= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (!lane && first + row < a.rows) a.probabilities[(first + row) * 256 + expert] = 1.0 / (1.0 + expf(-(sum + 0.f)));
        __syncthreads();
    }
}

// The QKV cache is dead once attention finishes. Prefetch two experts per CTA
// into this storage while the wide projection and TP reduction are running.
__device__ __forceinline__ void StageRouterPair(const float *weights, int tile, float *shared) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const float *source = weights + size_t(tile) * 8192;
    for (int i = threadIdx.x; i < 2048; i += Threads) {
        unsigned dst = __cvta_generic_to_shared(shared + i * 4);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(source + i * 4) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// Four 128-thread groups reuse the staged expert pair. Up to 16 rows, a
// group owns one expert and eight rows. At 17..32 rows it accumulates both
// experts while each BF16 input is in registers. Keep the router's FP32
// accumulation and reduction order for both shapes.
template <int ExpertsPerGroup>
__device__ __forceinline__ void RouteStagedPair(const BF *input, const float *weights,
                                                float *output, int count, int tile) {
    constexpr int Rows = 8, lanes = Threads / 4, warps = lanes / 32;
    static_assert(lanes == 128 && (ExpertsPerGroup == 1 || ExpertsPerGroup == 2));
    const int group = threadIdx.x / lanes, t = threadIdx.x % lanes;
    const int first = (ExpertsPerGroup == 2 ? group : group / 2) * Rows;
    const int firstExpert = ExpertsPerGroup == 2 ? 0 : group % 2;
    float2 accum[ExpertsPerGroup][Rows]{};
#pragma unroll 4
    for (int k = 0; k < 16; ++k) {
        const int col = k * 256 + t * 2;
        float2 w[ExpertsPerGroup];
#pragma unroll
        for (int e = 0; e < ExpertsPerGroup; ++e)
            w[e] = *reinterpret_cast<const float2 *>(weights + (firstExpert + e) * 4096 + col);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const auto *x = input + size_t(min(first + row, count - 1)) * 4096 + col;
            const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(x));
#pragma unroll
            for (int e = 0; e < ExpertsPerGroup; ++e) {
                accum[e][row].x = fmaf(v.x, w[e].x, accum[e][row].x);
                accum[e][row].y = fmaf(v.y, w[e].y, accum[e][row].y);
            }
        }
    }
    __shared__ float tree[4][ExpertsPerGroup][Rows][warps];
#pragma unroll
    for (int e = 0; e < ExpertsPerGroup; ++e) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float sum = accum[e][row].x + accum[e][row].y;
#pragma unroll
            for (int delta = 1; delta < 32; delta *= 2) sum += __shfl_down_sync(0xffffffffu, sum, delta);
            if (t % 32 == 0) tree[group][e][row][t / 32] = sum;
        }
    }
    __syncthreads();
    if (t < ExpertsPerGroup * Rows * warps) {
        const int e = t / (Rows * warps), row = t / warps % Rows, lane = t % warps;
        float sum = tree[group][e][row][lane];
#pragma unroll
        for (int delta = 1; delta < warps; delta *= 2) sum += __shfl_down_sync(0xffffffffu, sum, delta, warps);
        if (lane == 0 && first + row < count)
            output[(first + row) * 256 + tile * 2 + firstExpert + e] = 1.0 / (1.0 + expf(-(sum + 0.f)));
    }
}

__device__ void Route(const Args &a, float *scratch) {
    cooperative_groups::this_grid().sync();
    if (a.rows >= 16) RouteTiles<8>(a, scratch);
    else RouteTiles<4>(a, scratch);
}

template<bool Ring, bool WithRouter, int Groups>
__device__ __forceinline__ void FinishLayer(const Args &a, float (&part)[16][128], float *scratch,
                                            uint8_t *staging, bool frontStaged) {
    constexpr bool Wide = Groups > 1, shardedSwa = Ring && Groups == 4;
    auto grid = cooperative_groups::this_grid();
    if (a.transport.epochs) {
        if constexpr (shardedSwa) {
            if (frontStaged) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
                __syncthreads();
            }
        }
        if constexpr (Wide && WithRouter) {
            if (frontStaged && blockIdx.x < 128)
                StageRouterPair(a.routerWeight, blockIdx.x, reinterpret_cast<float *>(staging));
        }
        if constexpr (shardedSwa) {
            Project<false, Ring, false, Groups>(a, part, frontStaged ? staging + RouterPairBytes : nullptr);
            FinishShardedProjection(a, scratch);
        } else {
            if constexpr (Wide) {
                Project<false, Ring, false, Groups>(a, part);
                FinishPublishedProjection(a);
            } else ProjectAndReduceOutput<Ring>(a, part);
            grid.sync();
            for (int row = blockIdx.x; row < a.rows; row += gridDim.x) {
                int t = threadIdx.x;
                auto v = reinterpret_cast<const CustomArNormPacketType *>(a.normed)[row * 512 + t];
                CustomArNormPacket(v, a.hidden + row * 4096, a.gamma, a.normed + row * 4096, a.eps, scratch, t);
                if (row + gridDim.x < a.rows) __syncthreads();
            }
        }
        if constexpr (Wide && WithRouter) {
            if (frontStaged) {
                grid.sync();
                for (int tile = blockIdx.x; tile < 128; tile += gridDim.x) {
                    if (tile != blockIdx.x) StageRouterPair(a.routerWeight, tile, reinterpret_cast<float *>(staging));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
                    __syncthreads();
                    RouteStagedPair<shardedSwa ? 2 : 1>(a.normed, reinterpret_cast<const float *>(staging),
                                                       a.probabilities, a.rows, tile);
                    if (tile + gridDim.x < 128) __syncthreads();
                }
            } else Route(a, &part[0][0]);
        } else if constexpr (WithRouter) Route(a, &part[0][0]);
        return;
    }
    Project<false, Ring>(a, part);
    // Only the row CTAs need a transport slot. A local grid join publishes
    // projection stores from all CTAs before the per-row inter-GPU handshake.
    __threadfence_system();
    grid.sync();
    const int slots = a.signalSlots;
    if (blockIdx.x < min(a.rows, slots)) {
        const int t = threadIdx.x;
        CustomArBarrierStart<8>(a.signals, a.signals.signals[a.rank], a.rank);
        for (int row = blockIdx.x; row < a.rows; row += slots) {
            auto reduced = CustomArReducePacket<8>(a.peers.Load(), row * 512 + t);
            CustomArNormPacket(reduced, a.hidden + row * 4096, a.gamma, a.normed + row * 4096, a.eps, scratch, t);
            if (row + slots < a.rows) __syncthreads();
        }
        CustomArBarrierFinal<8>(a.signals, a.signals.signals[a.rank], a.rank);
    }
    if constexpr (WithRouter) Route(a, &part[0][0]);
}

template <bool Ring, bool WithRouter, int Groups>
__device__ __forceinline__ void LayerBody(const Args &a) {
    constexpr bool Wide = Groups > 1, shardedSwa = Ring && Groups == 4;
    static_assert(Groups == 1 || Groups == 2 || shardedSwa);
    auto grid = cooperative_groups::this_grid();
    __shared__ float part[16][128];
    __shared__ float scratch[256], maximum, denominator;
    if constexpr (Ring)
        if (a.transport.epochs)
            BeginRingReduction(a);
    if constexpr (Ring)
        RingRotary(a);
    __shared__ __align__(16) BF values[Ring ? 128 * 32 : 64 * 128];
    extern __shared__ __align__(16) uint8_t staging[];
    const bool overlap = Ring && !shardedSwa && a.stageWeights && gridDim.x >= QkvTiles + a.rows * RingTilesPerRow;
    const bool frontStaged = Ring && a.stageWeights;
    if (frontStaged && blockIdx.x < QkvTiles)
        StageQkv(a, staging);
    if (frontStaged)
        Pack<true>(a.input, 4096, a);
    else
        Pack<false>(a.input, 4096, a);
    grid.sync();
    if (overlap && blockIdx.x >= QkvTiles)
        StageRingHistory(a, blockIdx.x - QkvTiles, reinterpret_cast<BF *>(staging), values);
    else if (frontStaged) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
        __syncthreads();
        Project<true, Ring, true, Groups>(a, part, staging);
    } else
        Project<true, Ring, false, Groups>(a, part);
    if constexpr (Ring) grid.sync();
    else {
        ProjectIndex<Wide>(a, part);
        grid.sync();
        PrepareIndex(a, &part[0][0]);
        grid.sync();
        ScoreIndex(a);
        grid.sync();
        ExchangeIndex(a);
        grid.sync();
        // The attention score arena is dead until the sorted indices are ready.
        auto *pairs = reinterpret_cast<unsigned long long *>(a.scores);
        MergeIndexPairs<1>(a,a.lists,pairs);
        grid.sync();
        MergeIndexPairs<2>(a,pairs,a.lists);
        grid.sync();
        MergeIndexPairs<4>(a,a.lists,nullptr);
        grid.sync();
    }

    if constexpr (shardedSwa) {
        if (frontStaged) StageWideOutput(a, staging + RouterPairBytes);
    }
    __shared__ float prob[Ring ? Heads * 128 : 512];
    if constexpr (Ring) {
        if (overlap) {
            if (blockIdx.x < QkvTiles)
                PrefetchRingOutput(a);
            else
                RingAttention<true>(a, values, prob, reinterpret_cast<BF *>(staging));
        } else
            RingAttention<false, shardedSwa>(a, values, prob);
        grid.sync();
    } else {
        Scores(a);
        grid.sync();
        for (int task = blockIdx.x; task < a.rows * Heads; task += gridDim.x) {
            const auto r = a.Row(task / Heads);
            const int count = SelectedCount(r, max(0, min(r.capacity, *r.liveKeys)));
            float *s = a.scores + task * a.stride;
            naive_dsa_decode::Normalize(s, s, a.sink, task % Heads, count, scratch, maximum, denominator,
                                        threadIdx.x);
            __syncthreads();
        }
        grid.sync();
        ValueTiles<Wide>(a, values, prob);
        grid.sync();
        const int parts = (a.stride + 63) / 64;
        for (int i = blockIdx.x * Threads + threadIdx.x; i < a.rows * 1024; i += gridDim.x * Threads) {
            float sum = 0;
            for (int p = 0; p < parts; ++p)
                sum += a.partials[(i / 1024 * parts + p) * 1024 + i % 1024];
            StoreOutput(a, i, sum);
        }
        // After ValueTiles joins the grid, no CTA reads DSA scores.
        if (a.transport.epochs)
            BeginRingReduction(a);
        grid.sync();
    }
    FinishLayer<Ring, WithRouter, Groups>(a, part, scratch, staging, frontStaged);
}

#include "fastllm-naive-sm90-swa-stages.cuh"

template<bool KeyOnly> __global__ __launch_bounds__(Threads, 1) void ReplicatedProjection(
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    const __grid_constant__
#else
    const
#endif
    Args a) {
    __shared__ float part[16][128];
    auto grid = cooperative_groups::this_grid();
    if (a.transport.epochs) BeginRingReduction(a);
    Pack<false>(a.input, 4096, a);
    grid.sync();
    Project<true, false>(a, part);
    ProjectIndex<false, KeyOnly>(a, part);
    grid.sync();
    PrepareIndex<true, KeyOnly>(a, &part[0][0]);
}

template<bool WithRouter> __global__ __launch_bounds__(Threads, 1)
void ReplicatedOutput(
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    const __grid_constant__
#else
    const
#endif
    Args a) {
    __shared__ float part[16][128], scratch[256];
    FinishLayer<false, WithRouter, 1>(a, part, scratch, nullptr, false);
}
template <bool Ring, bool WithRouter>
__global__ __launch_bounds__(Threads, Ring ? 2 : 1) void Layer(
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    const __grid_constant__
#else
    const
#endif
        Args a) {
    LayerBody<Ring, WithRouter, 1>(a);
}
template <bool Ring, bool WithRouter, int Groups = 2>
__global__ __launch_bounds__(Threads, Ring ? 2 : 1) void WideLayer(
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    const __grid_constant__
#else
    const
#endif
        Args a) {
    LayerBody<Ring, WithRouter, Groups>(a);
}
} // namespace hopper_verify
