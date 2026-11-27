#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/naive-n05-kv.cuh"
#include "devices/cuda/naive-n05-dp-dsa-verify.cuh"
#include "devices/cuda/fastllm-mxfp8-activation.cuh"
#include "naive-n05-dsa-fused-attention.cuh"
#include "naivert/attn_dsa/body.cuh"
#include "naive-n05-dsa-verify-primitives.cuh"
#include "utils/utils.h"
#include "devices/cuda/naive-n05-verify-groups.cuh"
#include <unordered_map>
#include <cmath>

namespace fastllm_dsa_layer::dp_verify_blackwell {
using Core = fastllm_dsa_fused::Body<false, false>;
using Arena = Core::Scratch;
using ScoreSort = cub::BlockRadixSort<unsigned, Threads, 6, int, 4>;
namespace native = fastllm_dsa_fused::native_attention;
constexpr int Heads = 64, KvHeads = 4, KeyWidth = KvHeads * 192, ValueWidth = KvHeads * 128;
constexpr int QWidth = Heads * 192, PackedWidth = QWidth + KeyWidth + ValueWidth;
constexpr int GroupHeads = Heads / KvHeads, Groups = KvHeads;
constexpr int IndexColumns = 128 + 2048 + 16;
constexpr int ProjectionTiles = IndexColumns / 16, ProjectionBlocks = (ProjectionTiles + 1) / 2;
constexpr size_t PartialBytes = size_t(8 * Heads / native::Heads) * native::Scratch * sizeof(float);
constexpr size_t Ik = Arena::kBytes > PartialBytes ? Arena::kBytes : PartialBytes;
constexpr size_t Iq = Ik + 8 * 128 * sizeof(B);
constexpr size_t Iw = Iq + 8 * 2048 * sizeof(B), Query = Iw + 8 * 16 * sizeof(B);
constexpr size_t Bytes = Query + 8 * QWidth * sizeof(B);
constexpr size_t SelectionBytes = 8 * 2048 * sizeof(int);
// Same reserve as SplitDSAAttention. Both paths can reuse a previously
// captured allocation, including when the context bucket changes.
constexpr int FallbackElements = FastllmNaiveDSABatchLimit * (Heads / native::Heads) * native::Scratch;
static_assert(SelectionBytes + Bytes + PartialBytes <= FallbackElements * sizeof(float));
struct Input {
    const B *normed, *qkv, *wk, *wq, *weight;
    const float *norm, *bias, *positions;
    float theta, valueScale;
    int *selected;
    FastllmMxfp8Activation packed;
    fastllm_dsa_fused::Args core;
};

template<int N> __device__ __forceinline__ void ProjectIndexer(const Input &a, const B *input, int rows,
        int tile, B *ik, B *iq, B *iw, ProjectionStorage<N> &storage) {
    LoadProjectionInput<N, 4096>(input, rows, storage.input);
    auto source = [&](int col) {
        if (col < 128) return a.wk + size_t(col) * 4096;
        if (col < 2176) return a.wq + size_t(col - 128) * 4096;
        return a.weight + size_t(col - 2176) * 4096;
    };
    auto emit = [&](int col, int row, float x) {
        if (row >= rows) return;
        if (col < 128)
            ik[row * 128 + col] = __float2bfloat16_rn(x);
        else if (col < 2176)
            iq[row * 2048 + col - 128] = __float2bfloat16_rn(x);
        else
            iw[row * 16 + col - 2176] = __float2bfloat16_rn(Round(x) * .25f);
    };
    fastllm_dsa_projection::project_rows_mma<N, 4096, 4, 2, 12>(0, ProjectionTiles, source, storage.input,
                                                                storage.partial, emit, tile);
}
template<int Bits = 16> __device__ __forceinline__ void AppendMain(const Input &a, const FastllmNaiveDSARow &r, int rId,
        int group, B *query) {
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
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
        if constexpr (Bits == 16) {
            if (group) return;
            B *key = (B *)r.key + size_t(FastllmNaivePhysicalRow(r, visible - 1)) * r.keyStride;
            B *value = (B *)r.value + size_t(FastllmNaivePhysicalRow(r, visible - 1, true)) * ValueWidth;
            if (warp < KvHeads) RotateWarp(raw + QWidth + warp * 192, key + warp * 192, 192, cs, sn);
            for (int col = t; col < ValueWidth; col += Threads)
                value[col] =
                    __float2bfloat16_rn(float(raw[QWidth + KeyWidth + col]) * Round(a.valueScale));
        } else if constexpr (Bits == 8) {
            if (group) return;
            auto *key = (__nv_fp8_e4m3 *)r.key + size_t(FastllmNaivePhysicalRow(r, visible - 1)) * r.keyStride;
            auto *value = (__nv_fp8_e4m3 *)r.value + size_t(FastllmNaivePhysicalRow(r, visible - 1, true)) * ValueWidth;
            if (warp < KvHeads) RotateWarp(raw + QWidth + warp * 192, key + warp * 192, 192, cs, sn);
            for (int col = t; col < ValueWidth; col += Threads)
                value[col] = __nv_fp8_e4m3(Round(float(raw[QWidth + KeyWidth + col]) * Round(a.valueScale)));
        } else if (!group) {
            __shared__ B k[KeyWidth], v[ValueWidth];
            if (warp < KvHeads) RotateWarp(raw + QWidth + warp * 192, k + warp * 192, 192, cs, sn);
            for (int d = t; d < ValueWidth; d += Threads)
                v[d] = __float2bfloat16_rn(float(raw[QWidth + KeyWidth + d]) * Round(a.valueScale));
            __syncthreads();
            const int pk = FastllmNaivePhysicalRow(r, visible - 1), pv = FastllmNaivePhysicalRow(r, visible - 1, true);
            auto *key = (uint8_t *)r.key + size_t(pk) * (KeyWidth / 16 * 9);
            auto *value = (uint8_t *)r.value + size_t(pv) * (ValueWidth / 16 * 9);
            if (t < KeyWidth / 16) FastllmWriteFP4KVBlock(key, k + t * 16, t * 16, KeyWidth);
            if (t < ValueWidth / 16) FastllmWriteFP4KVBlock(value, v + t * 16, t * 16, ValueWidth);
            // All consumers must finish before this CTA stages the next row.
            __syncthreads();
        }
    }
}
template<int Bits = 16> __device__ __forceinline__ void AppendIndex(const Input &a, const FastllmNaiveDSARow &r, int row,
        int imageRow, B *ik, const B *iq, B *packed, float *scales) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int n = *r.liveKeys;
    if (n > 0 && n <= r.capacity) {
        NormalizeIndex(ik + row * 128, a.norm, a.bias);
        float cs, sn;
        sincosf(a.positions[row] * powf(a.theta, -2.f * lane / 64.f), &sn, &cs);
        cs = Round(cs);
        sn = Round(sn);
        if constexpr (Bits == 16) {
            B *key = (B *)r.key + size_t(FastllmNaivePhysicalRow(r, n - 1)) * r.keyStride;
            if (!warp) RotateWarp(ik + row * 128, key + KeyWidth, 128, cs, sn);
        } else if (!warp) {
            FastllmNaiveKV index(nullptr, nullptr, r.pageLen, true);
            index.Bind(r.cache + (Bits == 4 ? 4 : 0));
            const int stride = Bits == 4 ? 128 : KeyWidth + 128;
            auto *dst = (__nv_fp8_e4m3 *)index.base + size_t(index.PhysicalRow(n - 1)) * stride + stride - 128;
            RotateWarp(ik + row * 128, dst, 128, cs, sn);
        }
        // Rotate and pack directly into the tensor-core scorer image.
        for (int h = warp; h < 16; h += Threads / 32) {
            const B *p = iq + row * 2048 + h * 128;
            float x = float(p[lane]), y = float(p[32 + lane]);
            const float vals[] = {Round(Round(x * cs) - Round(y * sn)), Round(Round(y * cs) + Round(x * sn)),
                                  float(p[64 + lane]), float(p[96 + lane])};
            float mx = 0;
            for (float v : vals) mx = fmaxf(mx, fabsf(v));
            for (int step = 16; step; step >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, step));
            const float scale = fmaxf(mx, 1e-4f) / 448.f;
            const int head = imageRow * 16 + h;
            if (!lane) scales[head] = scale;
            for (int i = 0; i < 4; ++i) {
                auto q = __nv_fp8_e4m3(fmaxf(-448.f, fminf(448.f, vals[i] / scale)));
                if constexpr (Bits == 16)
                    packed[Core::ImageOffset(head, i * 32 + lane, 128) / 2] = __float2bfloat16(float(q));
                else
                    reinterpret_cast<uint8_t *>(packed)[fastllm_dsa_fused::Body<true, false>::ImageOffset(
                        head, i * 32 + lane, 128)] = q.__x;
            }
        }
    }
}
constexpr int splits = 8, segments = 2;
template<bool Interleaved = false> __device__ __forceinline__ void FinishAttention(int row, int attentionHead, int localHead, int segment,
        bool activeRow, const float *scratch, const float *sinkBias, B *output, FastllmMxfp8Activation packed) {
    const int lane = threadIdx.x % 32;
    const int layerHead = attentionHead % Heads;
    const int head = attentionHead % GroupHeads, col = segment * (128 / segments) + lane;
    const int block = localHead / GroupHeads;
    const auto *part = scratch +
        size_t(block) * (Interleaved ? splits : native::Splits) * GroupHeads * (128 + 2);
    const auto *stats = part + native::Splits * GroupHeads * 128;
    float maximum = -INFINITY, sum = 0.f;
    if (lane < splits && activeRow) {
        const auto *splitStats = Interleaved ? part + (lane * (128+2) + 128) * GroupHeads : stats;
        const float2 state = reinterpret_cast<const float2 *>(splitStats)[(Interleaved ? 0 : lane) * GroupHeads + head];
        maximum = state.x; sum = state.y;
    }
    const float ownMax = maximum;
    for (int d = 16; d; d /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, d));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float sink = sinkBias ? sinkBias[layerHead] : -INFINITY;
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
                values[n] = fmaf(part[k * GroupHeads * (Interleaved ? 130 : 128) + head * 128 + col + 32 * n], c, values[n]);
        }
        if (sinkBias) denominator += __expf(sink - maximum);
    }
#pragma unroll
    for (int n = 0; n < 4 / segments; ++n) {
        const B value = __float2bfloat16(denominator > 0 ? values[n] / denominator : 0.f);
        const int column = layerHead * 128 + col + n * 32;
        output[row * Heads * 128 + column] = value;
        const B rounded[1]{value};
        fastllm_mxfp8::StoreGroup(packed, row, column, rounded);
    }
}

template<int Bits> using SingleScorer = fastllm_dsa_fused::Body<Bits == 8, false>;
template<int Bits> union Shared {
    ProjectionStorage<8> projection;
    ScoreSort::TempStorage radix;
    typename SingleScorer<Bits>::SharedStorage core;
    native::StorageFor<256, GroupHeads> attention;
};
template<int Bits> __global__ __launch_bounds__(Threads, 1) void DpDSAVerify(const __grid_constant__ Input a) {
    extern __shared__ __align__(1024) unsigned char memory[];
    auto &s = *reinterpret_cast<Shared<Bits> *>(memory);
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
    SingleScorer<Bits>::initialize(core);
    // Populate all scorer lanes on short/partially live replays. Inactive
    // heads use a zero query and never enter the shortlist.
    for (int head = blockIdx.x * (Threads / 32) + warp; head < 8 * 16; head += gridDim.x * (Threads / 32)) {
        if (head / 16 >= core.activeRows || *core.rows[head / 16].liveKeys <= 0) {
            auto *dst = reinterpret_cast<B *>(core.workspace + Arena::kIqPackedOffset);
            for (int k = lane; k < 128; k += 32) {
                const int offset = SingleScorer<Bits>::ImageOffset(head, k, 128);
                if constexpr (Bits == 8) reinterpret_cast<uint8_t *>(dst)[offset] = 0;
                else dst[offset / 2] = B(0.f);
            }
            if (!lane) reinterpret_cast<float *>(core.workspace + Arena::kIndexWeightsScaledOffset)[head] = 0;
        }
    }
    if (blockIdx.x < ProjectionBlocks) {
        ProjectIndexer(a, a.normed, core.activeRows, blockIdx.x, ik, iq, iw, s.projection);
    }
    // The Indexer uses normalized BF16 input. Main Q/K/V and the borrowed
    // O activation arena belong to the preceding full-width projection.
    cudaGridDependencySynchronize();
    // Projection of the Indexer and publication of main Q/K/V use disjoint data.
    for (int row = int(blockIdx.x) - ProjectionBlocks; row >= 0 && row < core.activeRows * 8; row += int(gridDim.x) - ProjectionBlocks) {
        const int rId = row / 8, group = row % 8;
        auto r = FastllmNaiveResolveDSARow(core.rows[rId]);
        AppendMain<Bits>(a, r, rId, group, query);
    }
    grid.sync();
    if (blockIdx.x < core.activeRows) {
        const int row = blockIdx.x;
        auto r = FastllmNaiveResolveDSARow(core.rows[row]);
        AppendIndex<Bits>(a, r, row, row, ik, iq, reinterpret_cast<B *>(core.workspace + Arena::kIqPackedOffset),
                    reinterpret_cast<float *>(core.workspace + Arena::kIndexWeightsScaledOffset));
    }
    grid.sync();
    SingleScorer<Bits>::template run<false>(core, s.core, true, true);
    // FP8 attention walks the selected tokens in page order. Exporting their
    // stable score order can proceed concurrently in the first row-owning CTAs.
    auto *partial = reinterpret_cast<float *>(core.workspace + (Bits == 8 ? Bytes : 0));
    const auto *members = reinterpret_cast<const int *>(core.workspace + Arena::kIndexSelectsOffset);
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
            int slot = t * 6 + i, token = slot < count ? candidates[slot] : -1;
            tokens[i] = token;
            keys[i] = token >= 0 ? Core::ordered_float_key(logits[token]) : 0;
        }
        if constexpr (Bits == 8) {
            unsigned low = ~0u, high = 0;
            #pragma unroll
            for (int i = 0; i < 6; ++i) if (tokens[i] >= 0) {
                low = min(low, keys[i]); high = max(high, keys[i]);
            }
            #pragma unroll
            for (int d = 16; d; d >>= 1) {
                low = min(low, __shfl_xor_sync(0xffffffff, low, d));
                high = max(high, __shfl_xor_sync(0xffffffff, high, d));
            }
            auto *ranges = reinterpret_cast<unsigned *>(&s);
            if (!lane) { ranges[warp] = low; ranges[Threads/32 + warp] = high; }
            __syncthreads();
            low = ~0u; high = 0;
            #pragma unroll
            for (int w = 0; w < Threads/32; ++w) {
                low = min(low, ranges[w]); high = max(high, ranges[Threads/32+w]);
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 6; ++i) keys[i] = tokens[i] >= 0 ? keys[i] - low : 0;
            ScoreSort(s.radix).SortDescending(keys, tokens, 0, count ? max(1, 32 - __clz(high - low)) : 1);
        } else
            ScoreSort(s.radix).SortDescending(keys, tokens);
        for (int i = 0; i < 6; ++i)
            if (t * 6 + i < 2048) ids[t * 6 + i] = tokens[i];
    }
    if constexpr (Bits != 8) grid.sync();
    // Reserve the last, shorter work wave for CTAs that also export ordered IDs.
    const int firstTile = Bits == 8 ? int(blockIdx.x) - core.activeRows + (int(blockIdx.x) < core.activeRows ? int(gridDim.x) : 0)
                                    : int(blockIdx.x);
    for (int tile = firstTile; tile < core.activeRows * Groups * splits; tile += gridDim.x) {
        const int r = Bits == 8 ? tile % core.activeRows : tile / (splits * Groups);
        const int group = (Bits == 8 ? tile / core.activeRows : tile / splits) % Groups;
        const int split = Bits == 8 ? tile / (core.activeRows * Groups) : tile % splits;
        auto descriptor = core.rows[r];
        descriptor.indices = (Bits == 8 ? members : a.selected) + r * 2048;
        native::Attend<true, native::Phase::Produce, splits, 128, 256, true, false, native::NoOutputEpilogue, GroupHeads, false, Bits>(
            query, descriptor, core.sink, core.output, partial, nullptr, Heads, KvHeads, r,
            group, split, s.attention);
        __syncthreads();
    }
    grid.sync();
    // Each warp folds 64 adjacent V columns. Broadcast the eight softmax corrections
    // once per split and preserve the original FP32 accumulation and BF16 rounding.
    static_assert(splits <= native::Splits && splits <= 32);
    const int warpIndex = blockIdx.x * (Threads / 32) + warp;
    for (int tile = warpIndex; tile < core.activeRows * Heads * segments; tile += gridDim.x * (Threads / 32)) {
        const int attentionHead = tile / segments, row = attentionHead / Heads;
        FinishAttention(row, attentionHead, attentionHead, tile % segments, *core.rows[row].liveKeys > 0,
                        partial, core.sink, core.output, a.packed);
    }
}

#include "naive-n05-dp-dsa-batch.cuh"

template<auto Kernel, size_t SharedBytes> int ResidentGrid(int device) {
    if (FastllmCudaRuntimeArch() != 103) return 0; // The TU is compiled for 103a.
    static thread_local std::unordered_map<int, int> sizes;
    auto it = sizes.find(device);
    if (it != sizes.end()) return it->second;
    if (FastllmCudaGraphIsCapturingFast()) return 0;
    cudaDeviceProp prop{};
    int blocks = 0;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || !prop.cooperativeLaunch ||
        prop.multiProcessorCount < 128 || prop.sharedMemPerBlockOptin < SharedBytes ||
        cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes) !=
            cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, Kernel, Threads, SharedBytes) !=
            cudaSuccess ||
        !blocks) {
        cudaGetLastError();
        return sizes[device] = 0;
    }
    return sizes[device] = prop.multiProcessorCount;
}

bool CacheFits(const fastllm::Data &d, int width, int capacity, int device, fastllm::DataType type) {
    using namespace fastllm;
    if (d.dataDevice != CUDA || d.dataType != type || !d.cudaData || uintptr_t(d.cudaData) % 16 ||
        d.dataDeviceIds.size() != 1 || d.dataDeviceIds.front() != device || d.dims.size() != 3 ||
        d.dims[0] != 1 || d.dims[2] != width) return false;
    if (type == FP4_E2M1 && (d.cudaNativeNvfp4Layout ||
        (d.pagedKVCacheData && d.pagedKVCacheData->cudaNativeNvfp4Layout))) return false;
    if (d.isPagedKVCache)
        return d.expansionDims.size() == 3 && d.expansionDims[1] >= capacity && d.pagedKVCacheData &&
               d.pagedKVCacheData->cudaData && d.pageLen > 0 && d.pageLen % 8 == 0 &&
               d.pagedKVCacheData->maxPages > 0;
    return type != FP4_E2M1 && d.expansionBytes >= size_t(capacity) * width * (type == BFLOAT16 ? 2 : 1);
}

bool CacheLayout(const fastllm::Data &key, const fastllm::Data &value, const fastllm::Data *indexPool,
                 int capacity, int device) {
    using namespace fastllm;
    const auto type = key.dataType;
    const bool split = type == FP4_E2M1;
    if (type != BFLOAT16 && type != FP8_E4M3 && !split) return false;
    if (split != bool(indexPool) || !CacheFits(key, split ? KeyWidth : KeyWidth + 128, capacity, device, type) ||
        !CacheFits(value, ValueWidth, capacity, device, type) || key.kvCacheFp8Indexer ||
        key.isPagedKVCache != value.isPagedKVCache || (key.isPagedKVCache && key.pageLen != value.pageLen))
        return false;
    return !split || (key.isPagedKVCache && indexPool->dataType == FP8_E4M3 && indexPool->cudaData &&
           indexPool->dataDevice == CUDA && uintptr_t(indexPool->cudaData) % 16 == 0 &&
           indexPool->dataDeviceIds == key.dataDeviceIds && !indexPool->kvCacheFp8Indexer &&
           indexPool->dims.size() == 4 && indexPool->dims[0] >= key.pagedKVCacheData->maxPages &&
           indexPool->dims[1] == key.pageLen &&
           indexPool->dims[2] == 1 && indexPool->dims[3] == 128);
}
int BatchGrid(int rows, const FastllmNaiveVerifyGroups &groups, int capacity, const fastllm::Data *scores, const fastllm::Data *query,
              int device, fastllm::DataType type) {
    using namespace fastllm;
    // Large batches have dedicated paged decoders that pipeline KV heads.
    if ((rows >= 32 && type == FP8_E4M3) ||
        (rows >= 64 && type == FP4_E2M1 && FastllmCudaHasGroupedFP4DSADecode())) return 0;
    if (!groups.count || rows > MaxBatchRows || capacity <= 8192 ||
        capacity > BatchPitch || !scores || !query || !scores->cudaData || uintptr_t(scores->cudaData) % 16 ||
        scores->dataType != INT32 || scores->dataDevice != CUDA || scores->dataDeviceIds.size() != 1 ||
        scores->dataDeviceIds.front() != device || scores->GetBytes() < size_t(rows) * BatchPitch * sizeof(float))
        return 0;
    if (FastllmCudaGraphIsCapturingFast() &&
        (!query->cudaData || query->expansionBytes < size_t(rows) * QWidth * sizeof(B))) return 0;
    int grid = type == FP4_E2M1 ? ResidentGrid<BatchedDpDSAVerify<4>, sizeof(BatchShared)>(device) :
               type == FP8_E4M3 ? ResidentGrid<BatchedDpDSAVerify<8>, sizeof(BatchShared)>(device) :
                                  ResidentGrid<BatchedDpDSAVerify<16>, sizeof(BatchShared)>(device);
    grid -= grid % ScoreGroups;
    const size_t images = size_t(groups.count) * 8 * (2048 * sizeof(B) + 16 * sizeof(float));
    if (images > FallbackElements * sizeof(float)) return 0;
    const size_t work = type == FP8_E4M3 ? size_t(rows) * Groups : size_t(grid);
    return size_t(rows) * 2048 + work * splits * GroupHeads * (128 + 2) <= FallbackElements ? grid : 0;
}
} // namespace fastllm_dsa_layer::dp_verify_blackwell

bool FastllmCudaCanBatchDpDSAVerify(const std::vector<FastllmNaiveRopeCacheRow> &caches, int width,
        int capacity, const fastllm::Data &scores, const fastllm::Data *query, const fastllm::Data *indexPool, const std::vector<int> *requestLengths) {
    using namespace fastllm_dsa_layer::dp_verify_blackwell;
    const int device = FastllmCudaGetDevice();
    if (caches.empty() || caches.size() > MaxBatchRows) return false;
    const auto groups = FastllmNaiveVerifyGroups::Plan(caches.size(), width, requestLengths);
    const auto &first = caches.front();
    if (!first.key || !first.cache || !BatchGrid(caches.size(), groups, capacity, &scores, query, device,
                                                 first.key->dataType)) return false;
    for (int owner = 0; owner < groups.count; ++owner) {
        const int i = groups.First(owner);
        const auto &r = caches[i];
        if (r.cache != first.cache + owner * (indexPool ? 8 : 4) || !r.key || !r.value ||
            r.key->dataType != first.key->dataType || !CacheLayout(*r.key, *r.value, indexPool, capacity, device) ||
            r.key->isPagedKVCache != first.key->isPagedKVCache || (r.key->isPagedKVCache && r.key->pageLen != first.key->pageLen))
            return false;
        for (int j = 1; j < groups.Size(owner); ++j)
            if (caches[i + j].cache != r.cache || caches[i + j].key != r.key || caches[i + j].value != r.value)
                return false;
    }
    return true;
}

bool FastllmCudaTryDpDSAVerify(const fastllm::Data &normalized, const fastllm::Data &qkv,
                               const fastllm::Data &wk, const fastllm::Data &wq, const fastllm::Data &weights,
                               const fastllm::Data &keyNorm, const fastllm::Data &keyBias,
                               const fastllm::Data &sink, const fastllm::Data &positions,
                               const fastllm::Data &live, const fastllm::Data &key,
                               const fastllm::Data &value, const void *const *binding, int capacity,
                               float theta, float valueScale, FastllmMxfp8Activation packed,
                               FastllmNaiveDecodeScratch &scratch, fastllm::Data &attention, bool overlapQkv, int requestRows,
                               const fastllm::Data *scoreArena, fastllm::Data *queries, const fastllm::Data *indexPool, const std::vector<int> *requestLengths) {
    using namespace fastllm;
    using namespace fastllm_dsa_layer::dp_verify_blackwell;
    using fastllm_dsa_layer::B;
    const int rows = packed.rows, device = FastllmCudaGetDevice();
    if (!requestRows) requestRows = rows;
    const auto groups = FastllmNaiveVerifyGroups::Plan(rows, requestRows, requestLengths);
    const bool batch = groups.count > 1 || key.dataType == FP4_E2M1;
    if (!groups.count ||
        packed.swizzledRowOffset >= 0 || !packed.Matches(rows, Heads * 128) || !binding || capacity <= 8192 ||
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
    if (!dense(normalized, BFLOAT16, size_t(rows) * 4096, 4096) ||
        !dense(qkv, BFLOAT16, size_t(rows) * PackedWidth, PackedWidth) ||
        !dense(wk, BFLOAT16, 128 * 4096, 4096) || !dense(wq, BFLOAT16, 2048 * 4096, 4096) ||
        !dense(weights, BFLOAT16, 16 * 4096, 4096) || !dense(keyNorm, FLOAT32, 128, 128) ||
        !dense(keyBias, FLOAT32, 128, 128) || (!sink.dims.empty() && !dense(sink, FLOAT32, Heads, Heads)) ||
        !dense(positions, FLOAT32, rows, rows) || !dense(live, INT32, rows, rows) ||
        !CacheLayout(key, value, indexPool, capacity, device))
        return false;
    const int grid = batch ? BatchGrid(rows, groups, capacity, scoreArena, queries, device, key.dataType)
                           : key.dataType == FP8_E4M3 ? ResidentGrid<DpDSAVerify<8>, sizeof(Shared<8>)>(device)
                                                     : ResidentGrid<DpDSAVerify<16>, sizeof(Shared<16>)>(device);
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
    if (batch) {
        queries->dataType = BFLOAT16;
        queries->UpdateUnitSize();
        queries->Resize({1, rows, QWidth});
        queries->ToDevice(CUDA, normalized.dataDeviceIds, false);
        queries->Allocate(false);
    }
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
    c.activeRows = requestRows;
    c.output = (B *)attention.cudaData;
    c.sink = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    c.indexStride = indexPool ? 128 : KeyWidth + 128;
    c.indexKey = FastllmNaiveKV(key);
    if (indexPool) c.indexKey = FastllmNaiveKV(nullptr, nullptr, key.pageLen, true);
    c.indexCache = binding + (indexPool ? 4 : 0);
    for (int i = 0; i < requestRows; ++i) {
        auto &d = c.rows[i];
        d.cache = binding;
        d.capacity = capacity;
        d.keyStride = indexPool ? KeyWidth : c.indexStride;
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
    cfg.dynamicSmemBytes = batch ? sizeof(BatchShared) : key.dataType == FP8_E4M3 ? sizeof(Shared<8>) : sizeof(Shared<16>);
    cfg.stream = cudaStreamPerThread;
    cfg.attrs = attrs;
    cfg.numAttrs = !batch && overlapQkv ? 2 : 1;
    cudaError_t error;
    if (batch) {
        auto kernel = key.dataType == FP4_E2M1 ? BatchedDpDSAVerify<4> :
                      key.dataType == FP8_E4M3 ? BatchedDpDSAVerify<8> : BatchedDpDSAVerify<16>;
        error = cudaLaunchKernelEx(&cfg, kernel, a, groups, (float *)scoreArena->cudaData, (B *)queries->cudaData);
    } else error = cudaLaunchKernelEx(&cfg, key.dataType == FP8_E4M3 ? DpDSAVerify<8> : DpDSAVerify<16>, a);
    AssertInFastLLM(error == cudaSuccess,
                    std::string("DP DSA verification launch failed: ") + cudaGetErrorString(error));
    return true;
}
