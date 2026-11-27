#pragma once
#if defined(FASTLLM_ENABLE_NAIVE_INDEXER_SM90) && defined(FASTLLM_NAIVE_DSA_MMA) && !defined(USE_ROCM) &&         \
    CUDART_VERSION >= 12080 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#include "naive-n05-dsa-gather-sm90.cuh"
namespace {
struct DsaVerifyBatch { FastllmNaiveDSARow rows[128]; };
// Prepare the full-width QKV and the packed cuBLAS index projection together.
// Each live row owns one cache slot, with independent K and V page mappings.
template<bool Fp8KV> __global__ void DsaVerifyPrepare32(const __grid_constant__ DsaVerifyBatch batch, const BF16 *qkv,
                                   const BF16 *projection, bool planar, const float *positions, float theta, BF16 valueScale,
                                   const float *gamma, const float *bias, BF16 *query, BF16 *indexQuery,
                                   float *scales, BF16 *weights) {
#if __CUDA_ARCH__ == 900
    const int row = blockIdx.x, tile = blockIdx.y, t = threadIdx.x;
    const auto &desc = batch.rows[row];
    const int length = *desc.liveKeys;
    query += size_t(row) * 12288;
    if (length <= 0) {
        if (tile < 8) {
            auto *dst = reinterpret_cast<uint4 *>(query) + tile * 8 * 24;
            for (int i = t; i < 8 * 24; i += 64) dst[i] = make_uint4(0, 0, 0, 0);
        }
        return;
    }
    if (length > desc.capacity) return;
    qkv += size_t(row) * 13568;
    const BF16 *qRow = projection + size_t(row) * (planar ? 2048 : 2192);
    const BF16 *kRow = planar ? projection + size_t(gridDim.x) * 2048 + row * 128 : qRow + 2048;
    const BF16 *wRow = planar ? projection + size_t(gridDim.x) * 2176 + row * 16 : qRow + 2176;
    const float pos = positions[row];
    if (tile < 9) {
        __shared__ BF16 cosine[32], sine[32];
        LoadPackedRopeAngles(positions, theta, cosine, sine);
        if (tile < 8) {
            for (int i = t; i < 8 * 24; i += 64) {
                const int col = tile * 8 * 24 + i;
                uint4 v = reinterpret_cast<const uint4 *>(qkv)[col];
                RotatePackedQK(v, col, cosine, sine);
                reinterpret_cast<uint4 *>(query)[col] = v;
            }
        } else {
            const auto r = FastllmNaiveResolveDSARow(desc);
            const int kpos = FastllmNaivePhysicalRow(r, length - 1);
            const int vpos = FastllmNaivePhysicalRow(r, length - 1, true);
            using Word = std::conditional_t<Fp8KV, uint2, uint4>;
            auto *kdst = (Word *)r.key + size_t(kpos) * (r.keyStride / 8);
            auto *vdst = (Word *)r.value + size_t(vpos) * (512 / 8);
            for (int i = t; i < 96; i += 64) {
                uint4 v = reinterpret_cast<const uint4 *>(qkv + 12288)[i];
                RotatePackedQK(v, i, cosine, sine);
                if constexpr (Fp8KV) kdst[i] = PackCache8(v);
                else kdst[i] = v;
            }
            uint4 v = reinterpret_cast<const uint4 *>(qkv + 13056)[t];
            ScalePackedV(v, valueScale);
            if constexpr (Fp8KV) vdst[t] = PackCache8(v);
            else vdst[t] = v;
        }
        return;
    }
    __shared__ BF16 rotated[128];
    if (tile == 9) {
        __shared__ BF16 normalized[128];
        NormalizeIndexerKey(kRow, normalized, gamma, bias);
        __syncthreads();
        const auto r = FastllmNaiveResolveDSARow(desc);
        const int physical = FastllmNaivePhysicalRow(r, length - 1);
        auto dst = FastllmNaiveKVWriteRow{const_cast<void *>(r.key), Fp8KV} + size_t(physical) * r.keyStride + 768;
        RopeHeadTo(normalized, dst, pos, 64, theta);
        dst[t + 64] = normalized[t + 64];
        if (t < 16) weights[row * 16 + t] = __float2bfloat16_rn(float(wRow[t]) * .25f);
        return;
    }
    const int head = tile - 10, hrow = row * 16 + head;
    const BF16 *src = qRow + head * 128;
    RopeHeadTo(src, rotated, pos, 64, theta);
    rotated[t + 64] = src[t + 64];
    __syncthreads();
    float mx = fmaxf(fabsf(float(rotated[t])), fabsf(float(rotated[t + 64])));
    for (int delta = 16; delta; delta /= 2)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, delta));
    __shared__ float warpMax[2];
    if (t % 32 == 0) warpMax[t / 32] = mx;
    __syncthreads();
    const float scale = fmaxf(1e-4f, fmaxf(warpMax[0], warpMax[1])) / 448.f;
    if (!t) scales[hrow] = scale;
    for (int i = t; i < 128; i += 64)
        indexQuery[hrow * 128 + i] =
            BF16(float(__nv_fp8_e4m3(fmaxf(-448.f, fminf(448.f, float(rotated[i]) / scale)))));
#endif
}

// A score CTA shares K across its queries. Map the last tile of each request
// back to that request's final query, even if the next request has a different KV.
template<int Queries> struct DsaScoreTiles {
    struct {
        naive_indexer_batch::Row logical[128];
        struct { uint8_t first, last; } tiles[128];
        __device__ const naive_indexer_batch::Row &operator[](int i) const {
            const auto &tile = tiles[i / Queries];
            return logical[min(int(tile.first) + i % Queries, int(tile.last))];
        }
    } rows;
};

template<bool Fp8> void LaunchDsaVerifyScores(const naive_indexer_batch::Batch &index, int rows,
        const std::vector<int> &requests, int capacity, const BF16 *weights) {
    namespace ib = naive_indexer_batch;
    constexpr int queries = Fp8 ? 4 : 2;
    const int keys = (capacity + naive_dsa_mma::kKeys - 1) / naive_dsa_mma::kKeys;
    auto launch = [&](const auto &batch, int groups) {
        using Batch = std::decay_t<decltype(batch)>;
        const int blocks = DecodeIndexerScoreBlocks(groups, keys, Fp8 ? 64 : 32);
        if constexpr (Fp8) ib::HopperScoresSharedHead<queries, Batch><<<dim3(blocks, groups), 128>>>(weights, batch);
        else ib::HopperScores<false, false, queries, Batch><<<dim3(blocks, groups), 128>>>(weights, batch);
    };
    if (std::all_of(requests.begin(), requests.end(), [](int n) { return n % queries == 0; })) {
        launch(index, rows / queries);
        return;
    }
    DsaScoreTiles<queries> batch{};
    static_assert(sizeof(batch) + 16 <= ib::ArgumentBytes);
    std::copy_n(index.rows, rows, batch.rows.logical);
    int groups = 0, first = 0;
    for (int n : requests) {
        for (int offset = 0; offset < n; offset += queries)
            batch.rows.tiles[groups++] = {uint8_t(first + offset), uint8_t(first + n - 1)};
        first += n;
    }
    launch(batch, groups);
}

struct DsaOutputWriter32 {
    FastllmMxfp8Activation packed;
    __device__ __forceinline__ void zero(BF16 *out, int row, int col) const {
        const uint4 zeros = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4 *>(out + size_t(row) * 8192 + col) = zeros;
        if (packed.rows > 16) naive_prefill_mma::StoreSwaBlock32(packed, row, col, zeros);
        else fastllm_mxfp8::StoreZero8(packed, row, col);
    }
    __device__ __forceinline__ void operator()(BF16 *out, int row, int col, uint4 value) const {
        *reinterpret_cast<uint4 *>(out + size_t(row) * 8192 + col) = value;
        if (packed.rows > 16) naive_prefill_mma::StoreSwaBlock32(packed, row, col, value);
        else fastllm_mxfp8::Store8(packed, row, col, value);
    }
};
} // namespace
#endif

bool FastllmCudaSm90VerifyDsaBatch(const FastllmNaiveVerifyAttentionInput &x, FastllmMxfp8Activation packed) {
#if defined(FASTLLM_ENABLE_NAIVE_INDEXER_SM90) && defined(FASTLLM_NAIVE_DSA_MMA) && !defined(USE_ROCM) &&         \
    CUDART_VERSION >= 12080 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
    using namespace fastllm;
    namespace ib = naive_indexer_batch;
    namespace sort = naive_indexer_sortpack;
    const int rows = static_cast<int>(x.caches.size());
    static_assert(ib::BatchLimit >= 128);
    const bool fp8 = rows && x.caches[0].key && x.caches[0].key->dataType == FP8_E4M3;
    if (FastllmCudaRuntimeArch() != 90 || !FastllmCudaFlashInferDataTypeSupported(BFLOAT16) || x.sliding ||
        !x.fp8Query || x.fp8IndexerCache || !x.qkv || !x.normed || !x.live || !x.positions || !x.sink ||
        !x.attention || !x.indexQueryWeight || !x.indexKeyWeight || !x.indexWeight || !x.indexNorm ||
        !x.indexBias || x.config.heads != 64 || x.config.kvHeads != 4 || x.config.headDim != 192 ||
        x.config.valueDim != 128 || x.config.rotaryDim != 64 || x.indexHeads != 16 || x.indexDim != 128 ||
        x.topK != 2048 || rows < 1 || rows > 128 || x.requests.empty() || x.requests.size() > size_t(rows) ||
        !std::all_of(x.requests.begin(), x.requests.end(), [](int n) { return n >= 1 && n <= 8; }) ||
        std::accumulate(x.requests.begin(), x.requests.end(), 0) != rows ||
        !packed.Matches(rows, 8192) || x.capacity <= 2048 || x.capacity > 65536 ||
        !x.indices || !x.query || !x.scratch)
        return false;
    auto dense = [&](const Data &d, DataType type, size_t size) {
        if (!d.cudaData || d.dataType != type || d.dataDevice != CUDA || d.dataDeviceIds != x.qkv->dataDeviceIds ||
            d.dims.size() != d.strides.size() || d.Count(0) != size)
            return false;
        size_t stride = 1;
        for (int i = int(d.dims.size()) - 1; i >= 0; --i) {
            if (d.strides[i] != stride) return false;
            stride *= d.dims[i];
        }
        return (uintptr_t(d.cudaData) & 15) == 0;
    };
    if (!dense(*x.qkv, BFLOAT16, rows * 13568) || !dense(*x.live, INT32, rows) ||
        !dense(*x.positions, FLOAT32, rows) || !dense(*x.indexNorm, FLOAT32, 128) ||
        !dense(*x.indexBias, FLOAT32, 128) || (!x.sink->dims.empty() && !dense(*x.sink, FLOAT32, 64)) ||
        !std::isfinite(x.theta) || x.theta <= 0 || !std::isfinite(x.valueScale))
        return false;
    auto cache = [&](const Data *p, int width) {
        if (!p || !p->cudaData || p->dataDevice != CUDA || p->dataDeviceIds != x.qkv->dataDeviceIds ||
            p->dataType != (fp8 ? FP8_E4M3 : BFLOAT16) || p->dims.size() != 3 || p->dims[0] != 1 || p->dims[2] != width)
            return false;
        if (p->isPagedKVCache)
            return p->pagedKVCacheData && p->pagedKVCacheData->cudaData && p->pageLen > 0 &&
                   p->expansionDims.size() == 3 && p->expansionDims[1] >= x.capacity;
        return p->expansionBytes >= size_t(x.capacity) * width * (fp8 ? 1 : sizeof(BF16));
    };
    DsaVerifyBatch batch{};
    ib::Batch index{};
    int owner = 0, first = 0;
    for (int i = 0; i < rows; ++i) {
        if (i == first + x.requests[owner]) { first = i; ++owner; }
        const auto &b = x.caches[i];
        if (!b.cache || b.indexCache || b.indexerContextRank || b.cacheOffset >= 0 || b.verifyOffset >= 0 ||
            !cache(b.key, 896) || !cache(b.value, 512) || b.key->kvCacheFp8Indexer ||
            b.key->isPagedKVCache != b.value->isPagedKVCache ||
            (b.key->isPagedKVCache && b.key->pageLen != b.value->pageLen) ||
            b.cache != x.caches[first].cache)
            return false;
        auto &r = batch.rows[i];
        r.cache = b.cache;
        r.pageLen = b.key->isPagedKVCache ? b.key->pageLen : 0;
        r.liveKeys = (const int *)x.live->cudaData + i;
        r.capacity = x.capacity;
        r.keyStride = 896;
        r.fp8 = fp8;
        r.selectedCount = 2048;
        index.rows[i].key = FastllmNaiveKV(*b.key);
    }
    auto attentionKernel = fp8
        ? naive_dsa_gather_sm90::GatherAttention32<16, 128, DsaVerifyBatch, DsaOutputWriter32, true>
        : naive_dsa_gather_sm90::GatherAttention32<16, 128, DsaVerifyBatch, DsaOutputWriter32>;
    constexpr size_t attBytes = sizeof(naive_dsa_gather_sm90::Storage<16, 128>);
    auto select = x.capacity <= 4096 ? sort::SelectSort<2048> : sort::SelectSort<4096>;
    const size_t sortBytes = x.capacity <= 4096 ? sort::SortBytes<256, 5, true, false, 2048>()
                                                : sort::SortBytes<256, 5, true, false, 4096>();
    if (!BulkKernelSupported(attentionKernel, attBytes, 90) ||
        (x.capacity <= 16384 && !BulkKernelSupported(select, sortBytes, 90)))
        return false;
    auto &s = *x.scratch;
    bool planar = false;
    if (!FastllmCudaSm90ProjectVerifyIndexer(*x.normed, *x.indexQueryWeight, *x.indexKeyWeight, *x.indexWeight,
                                         s.dsaLayer, planar))
        return false;
    FastllmCudaNaivePrepareVerifyIndexerScratch(rows, x.capacity, true, s);
    Output(s.windowValue, BFLOAT16, {rows, 16});
    Output(*x.query, BFLOAT16, {1, rows, 12288});
    Output(*x.attention, BFLOAT16, {1, rows, 8192});
    Output(*x.indices, INT32, {rows, 2048});
    for (int i = 0; i < rows; ++i) {
        auto &r = index.rows[i];
        int *selected = (int *)x.indices->cudaData + i * 2048;
        batch.rows[i].indices = selected;
        r.indices = selected;
        r.live = batch.rows[i].liveKeys;
        r.cache = batch.rows[i].cache;
        r.query = (BF16 *)s.indexQuery.cudaData + i * 2048;
        r.scale = (float *)s.indexScale.cudaData + i * 16;
        r.scores = (unsigned *)s.indexScores.cudaData + size_t(i) * x.capacity;
        r.workspace = (naive_topk::Workspace *)s.topk.cudaData + i;
        r.sequence = i;
        r.stride = 896;
    }
    auto prepare = fp8 ? DsaVerifyPrepare32<true> : DsaVerifyPrepare32<false>;
    prepare<<<dim3(rows, 26), 64>>>(
        batch, (const BF16 *)x.qkv->cudaData, (const BF16 *)s.dsaLayer.cudaData, planar,
        (const float *)x.positions->cudaData, x.theta, BF16(x.valueScale), (const float *)x.indexNorm->cudaData,
        (const float *)x.indexBias->cudaData, (BF16 *)x.query->cudaData, (BF16 *)s.indexQuery.cudaData,
        (float *)s.indexScale.cudaData, (BF16 *)s.windowValue.cudaData);
    if (fp8) LaunchDsaVerifyScores<true>(index, rows, x.requests, x.capacity, (const BF16 *)s.windowValue.cudaData);
    else LaunchDsaVerifyScores<false>(index, rows, x.requests, x.capacity, (const BF16 *)s.windowValue.cudaData);
    if (x.capacity <= 16384)
        select<<<rows, 256, sortBytes>>>(index);
    else
        LaunchBatchDecodeIndexerTopK(index, rows, x.capacity);
    attentionKernel<<<dim3(rows, 4), 512, attBytes>>>(
        batch, (const BF16 *)x.query->cudaData, x.sink->dims.empty() ? nullptr : (const float *)x.sink->cudaData,
        (BF16 *)x.attention->cudaData, DsaOutputWriter32{packed});
    CheckLaunch();
    return true;
#else
    return false;
#endif
}
