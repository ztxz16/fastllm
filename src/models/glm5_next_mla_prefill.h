#ifndef FASTLLM_GLM5_NEXT_MLA_PREFILL_H
#define FASTLLM_GLM5_NEXT_MLA_PREFILL_H

#ifdef USE_CUDA
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace fastllm {
namespace glm5_next_detail {

// These descriptors borrow temporary projection buffers, not allocator pages.
struct PrefillCacheView : Data {
    PrefillCacheView(PagedCacheManager &pool, int tokens)
        : Data(DataType::BFLOAT16) {
        dataDevice = pool.dataDevice;
        dataDeviceIds = pool.dataDeviceIds;
        Resize({pool.dims[2], tokens, pool.dims[3]});
        isPagedKVCache = true;
        pagedKVCacheData = &pool;
        pageLen = pool.dims[1];
        lastPageLen = (tokens - 1) % pageLen + 1;
        pageIndex.resize(pool.dims[0]);
        std::iota(pageIndex.begin(), pageIndex.end(), 0);
    }
    ~PrefillCacheView() { isPagedKVCache = false; }
};

// Materialize only a bounded group of heads at a time. Each group attends the
// complete prefix, so the existing paged prefill kernel owns causal masking
// and split-KV reduction. No expanded K/V survives this call.
// Reassociation changes BF16 rounding versus absorbed MLA. The byte budget
// covers active K/V projections, not latent/output buffers or allocator caches.
inline bool TryMhaPrefill(Data &query, const Data &latentCache,
                         Data &keyWeight, Data &valueWeight, Data &output,
                         float scale, size_t maxExpandedBytes = 256ull << 20) {
    constexpr int dim = 256, rank = 512;
    const auto *pool = latentCache.pagedKVCacheData;
    if (query.dims.size() != 3 || query.dims[0] <= 0 ||
        query.dims[1] <= 1 || query.dims[2] != dim ||
        query.dataType != DataType::BFLOAT16 ||
        query.dataDevice != DataDevice::CUDA || query.cudaData == nullptr ||
        !latentCache.isPagedKVCache || pool == nullptr ||
        pool->dataType != DataType::BFLOAT16 ||
        pool->dataDevice != query.dataDevice ||
        pool->dataDeviceIds != query.dataDeviceIds || pool->cudaData == nullptr ||
        pool->dims.size() != 4 || pool->dims[2] != 1 || pool->dims[3] != rank ||
        latentCache.pageLen <= 0 || pool->dims[1] != latentCache.pageLen ||
        latentCache.pageIndex.empty() || latentCache.lastPageLen <= 0 ||
        latentCache.lastPageLen > latentCache.pageLen ||
        keyWeight.dataType != DataType::BFLOAT16 ||
        valueWeight.dataType != DataType::BFLOAT16 ||
        keyWeight.dataDevice != query.dataDevice ||
        valueWeight.dataDevice != query.dataDevice ||
        keyWeight.dataDeviceIds != query.dataDeviceIds ||
        valueWeight.dataDeviceIds != query.dataDeviceIds ||
        keyWeight.cudaData == nullptr || valueWeight.cudaData == nullptr ||
        FastllmCudaGraphIsCapturingFast() || !FastllmCudaFlashInferSupported()) {
        return false;
    }
    const int heads = query.dims[0], rows = query.dims[1];
    if (keyWeight.dims != std::vector<int>({heads, dim, rank}) ||
        valueWeight.dims != keyWeight.dims ||
        query.strides != std::vector<uint64_t>({uint64_t(rows) * dim, dim, 1}) ||
        keyWeight.strides != std::vector<uint64_t>({dim * rank, rank, 1}) ||
        valueWeight.strides != keyWeight.strides ||
        pool->strides != std::vector<uint64_t>({uint64_t(pool->dims[1]) * rank, rank, rank, 1})) {
        return false;
    }
    const int pageLen = latentCache.pageLen;
    const uint64_t padded = uint64_t(latentCache.pageIndex.size()) * pageLen;
    if (padded > uint64_t(std::numeric_limits<int>::max())) return false;
    const int paddedTokens = int(padded);
    const int tokens = paddedTokens - pageLen + latentCache.lastPageLen;
    if (tokens < rows) return false;
    for (int page : latentCache.pageIndex) {
        if (page < 0 || page >= pool->dims[0]) return false;
    }
    const size_t bytesPerHead = padded * dim * sizeof(uint16_t);
    const size_t capacity = maxExpandedBytes / (2 * bytesPerHead);
    if (capacity == 0) return false;
    int groupHeads = 1;
    while (groupHeads <= heads / 2 && size_t(groupHeads * 2) <= capacity)
        groupHeads *= 2;
    const size_t latentBytes = padded * rank * sizeof(uint16_t);
    const size_t projectionBytes = bytesPerHead * groupHeads;
    void *scratch = nullptr;
    const auto allocated = FastllmCudaTryMalloc(
        &scratch, latentBytes + 2 * projectionBytes);
    if (allocated == FASTLLM_CUDA_TRY_MALLOC_ERROR)
        throw std::runtime_error("GLM MLA prefill workspace allocation failed");
    if (scratch == nullptr) return false;
    struct ReleaseScratch {
        void *ptr;
        ~ReleaseScratch() {
            FastllmCudaSyncCurrentThreadStream();
            FastllmCudaFree(ptr);
        }
    } release{scratch};
    Data storage(DataType::BFLOAT16);
    storage.dataDevice = query.dataDevice;
    storage.dataDeviceIds = query.dataDeviceIds;
    storage.cudaData = scratch;
    storage.isFake = true;
    Data latent;
    latent.FakeFrom(storage, 0);
    latent.Resize({paddedTokens, rank});

    // Coalesce consecutive physical pages without assuming logical pages are
    // contiguous. Do not read the unused tail of the final cache page.
    for (size_t first = 0; first < latentCache.pageIndex.size();) {
        size_t end = first + 1;
        while (end < latentCache.pageIndex.size() &&
               latentCache.pageIndex[end] == latentCache.pageIndex[end - 1] + 1)
            ++end;
        const size_t beginToken = first * pageLen;
        const size_t count = std::min(end * pageLen, size_t(tokens)) - beginToken;
        auto *dst = static_cast<uint8_t *>(scratch) + beginToken * rank * 2;
        auto *src = static_cast<uint8_t *>(pool->cudaData) +
                    size_t(latentCache.pageIndex[first]) * pageLen * rank * 2;
        if (!FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(dst, src, count * rank * 2))
            throw std::runtime_error("GLM MLA prefill cache gather failed");
        first = end;
    }
    if (tokens < paddedTokens)
        FastllmCudaMemset0(static_cast<uint8_t *>(scratch) + size_t(tokens) * rank * 2,
                          size_t(paddedTokens - tokens) * rank * 2);

    output.dataType = query.dataType;
    output.UpdateUnitSize();
    output.dataDevice = query.dataDevice;
    output.dataDeviceIds = query.dataDeviceIds;
    output.Resize({heads, rows, dim});
    output.Allocate();
    for (int firstHead = 0; firstHead < heads; firstHead += groupHeads) {
        const int count = std::min(groupHeads, heads - firstHead);
        Data wk, wv, k, v, q, groupOutput;
        wk.FakeFrom(keyWeight, size_t(firstHead) * dim * rank * 2);
        wv.FakeFrom(valueWeight, size_t(firstHead) * dim * rank * 2);
        wk.Resize({count * dim, rank});
        wv.Resize(wk.dims);
        k.FakeFrom(storage, latentBytes);
        v.FakeFrom(storage, latentBytes + projectionBytes);
        MatMulTransB(latent, wk, k);
        MatMulTransB(latent, wv, v);
        PagedCacheManager kp, vp;
        kp.FakeFrom(k, 0);
        vp.FakeFrom(v, 0);
        kp.Resize({int(latentCache.pageIndex.size()), pageLen, count, dim});
        vp.Resize(kp.dims);
        PrefillCacheView kc(kp, tokens), vc(vp, tokens);
        q.FakeFrom(query, size_t(firstHead) * rows * dim * 2);
        q.Resize({count, rows, dim});
        // The shared planner also serves other layers; always plan this shape.
        AttentionPaged(q, kc, vc, groupOutput, 1, scale, 1, false);
        auto *dst = static_cast<uint8_t *>(output.cudaData) +
                    size_t(firstHead) * rows * dim * 2;
        if (!FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                dst, groupOutput.cudaData, size_t(count) * rows * dim * 2))
            throw std::runtime_error("GLM MLA prefill output copy failed");
        FastllmCudaSyncCurrentThreadStream();
    }
    return true;
}
} // namespace glm5_next_detail
} // namespace fastllm
#endif // USE_CUDA
#endif // FASTLLM_GLM5_NEXT_MLA_PREFILL_H
