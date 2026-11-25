#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/naive-n05-kv.cuh"
#include "devices/cuda/naive-n05-dp-dsa-verify.cuh"
#include "devices/cuda/fastllm-mxfp8-activation.cuh"
#include "naive-n05-dsa-fused-attention.cuh"
#include "naivert/attn_dsa/body.cuh"
#include "naive-n05-dsa-verify-primitives.cuh"
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
static_assert(SelectionBytes + Bytes <= FallbackElements * sizeof(float));
struct Input {
    const B *normed, *qkv, *wk, *wq, *weight;
    const float *norm, *bias, *positions;
    float theta, valueScale;
    int *selected;
    FastllmMxfp8Activation packed;
    fastllm_dsa_fused::Args core;
};
union Shared {
    ProjectionStorage<8> projection;
    ScoreSort::TempStorage radix;
    Core::SharedStorage core;
    native::StorageFor<256, GroupHeads> attention;
};
__global__ __launch_bounds__(Threads, 1) void DpDSAVerify(const __grid_constant__ Input a) {
    extern __shared__ __align__(1024) unsigned char memory[];
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
            for (int k = lane; k < 128; k += 32) dst[Core::ImageOffset(head, k, 128) / 2] = B(0.f);
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
                B *key = (B *)r.key + size_t(FastllmNaivePhysicalRow(r, visible - 1)) * r.keyStride;
                B *value = (B *)r.value + size_t(FastllmNaivePhysicalRow(r, visible - 1, true)) * ValueWidth;
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
            B *key = (B *)r.key + size_t(FastllmNaivePhysicalRow(r, n - 1)) * r.keyStride;
            if (!warp) RotateWarp(ik + row * 128, key + KeyWidth, 128, cs, sn);
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
                for (int i = 0; i < 4; ++i)
                    packed[Core::ImageOffset(head, i * 32 + lane, 128) / 2] =
                        __float2bfloat16(float(__nv_fp8_e4m3(fmaxf(-448.f,fminf(448.f,vals[i] / scale)))));
            }
        }
    }
    grid.sync();
    Core::run<false>(core, s.core, true, true);
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
            int slot = t * 6 + i, token = slot < count ? candidates[slot] : -1;
            tokens[i] = token;
            keys[i] = token >= 0 ? Core::ordered_float_key(logits[token]) : 0;
        }
        ScoreSort(s.radix).SortDescending(keys, tokens);
        for (int i = 0; i < 6; ++i)
            if (t * 6 + i < 2048) ids[t * 6 + i] = tokens[i];
    }
    grid.sync();
    constexpr int splits = 8;
    for (int tile = blockIdx.x; tile < core.activeRows * Groups * splits; tile += gridDim.x) {
        const int work = tile / splits, r = work / Groups;
        auto descriptor = core.rows[r];
        descriptor.indices = a.selected + r * 2048;
        native::Attend<true, native::Phase::Produce, splits, 128, 256, true>(
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
}

int ResidentGrid(int device) {
    if (FastllmCudaRuntimeArch() != 103) return 0; // The TU is compiled for 103a.
    static thread_local std::unordered_map<int, int> sizes;
    auto it = sizes.find(device);
    if (it != sizes.end()) return it->second;
    if (FastllmCudaGraphIsCapturingFast()) return 0;
    cudaDeviceProp prop{};
    int blocks = 0;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || !prop.cooperativeLaunch ||
        prop.multiProcessorCount < 128 || prop.sharedMemPerBlockOptin < sizeof(Shared) ||
        cudaFuncSetAttribute(DpDSAVerify, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared)) !=
            cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, DpDSAVerify, Threads, sizeof(Shared)) !=
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
        if (!resident(d, BFLOAT16) || d.dims.size() != 3 || d.dims[0] != 1 || d.dims[2] != width)
            return false;
        if (d.isPagedKVCache)
            return d.expansionDims.size() == 3 && d.expansionDims[1] >= capacity && d.pagedKVCacheData &&
                   d.pagedKVCacheData->cudaData && d.pageLen > 0 && d.pageLen % 8 == 0 &&
                   d.pagedKVCacheData->maxPages > 0;
        return d.expansionBytes >= size_t(capacity) * width * sizeof(B);
    };
    if (!dense(normalized, BFLOAT16, size_t(rows) * 4096, 4096) ||
        !dense(qkv, BFLOAT16, size_t(rows) * PackedWidth, PackedWidth) ||
        !dense(wk, BFLOAT16, 128 * 4096, 4096) || !dense(wq, BFLOAT16, 2048 * 4096, 4096) ||
        !dense(weights, BFLOAT16, 16 * 4096, 4096) || !dense(keyNorm, FLOAT32, 128, 128) ||
        !dense(keyBias, FLOAT32, 128, 128) || (!sink.dims.empty() && !dense(sink, FLOAT32, Heads, Heads)) ||
        !dense(positions, FLOAT32, rows, rows) || !dense(live, INT32, rows, rows) ||
        !cache(key, KeyWidth + 128) || !cache(value, ValueWidth) || key.kvCacheFp8Indexer ||
        key.isPagedKVCache != value.isPagedKVCache || (key.isPagedKVCache && key.pageLen != value.pageLen))
        return false;
    const int grid = ResidentGrid(device);
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
    c.indexStride = KeyWidth + 128;
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
    cfg.dynamicSmemBytes = sizeof(Shared);
    cfg.stream = cudaStreamPerThread;
    cfg.attrs = attrs;
    cfg.numAttrs = overlapQkv ? 2 : 1;
    auto error = cudaLaunchKernelEx(&cfg, DpDSAVerify, a);
    AssertInFastLLM(error == cudaSuccess,
                    std::string("DP DSA verification launch failed: ") + cudaGetErrorString(error));
    return true;
}
