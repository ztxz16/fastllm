#pragma once

#include "fastllm.h"
#include "utils.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace fastllm {
#ifdef USE_CUDA
    class Qwen4CudaDeviceGuard {
    public:
        Qwen4CudaDeviceGuard()
            : previousDevice(FastllmCudaGetDevice()), changed(true) {}

        explicit Qwen4CudaDeviceGuard(const std::vector<int> &deviceIds)
            : previousDevice(FastllmCudaGetDevice()), changed(false) {
            if (!deviceIds.empty() && deviceIds[0] != previousDevice) {
                FastllmCudaSetDevice(deviceIds[0]);
                changed = true;
            }
        }

        ~Qwen4CudaDeviceGuard() {
            if (changed) FastllmCudaSetDevice(previousDevice);
        }

    private:
        int previousDevice;
        bool changed;
    };
#endif
    inline bool Qwen4IsPagedKV(const Data &cache) {
        return cache.isPagedKVCache && !cache.isLinearAttention &&
               cache.pagedKVCacheData != nullptr;
    }

    inline void Qwen4ReleasePagedReference(Data &cache) {
        if (cache.isPagedKVCache && cache.pagedKVCacheData != nullptr) {
            cache.pagedKVCacheData->ReleasePageIndices(cache.pageIndex);
        }
        cache.pageIndex.clear();
        cache.pagedKVCacheData = nullptr;
        cache.isPagedKVCache = false;
        cache.lastPageLen = 0;
    }

    inline void Qwen4UpdatePageTable(Data &cache) {
#ifdef USE_CUDA
        if (!Qwen4IsPagedKV(cache) || cache.dataDevice != DataDevice::CUDA) return;
        Qwen4CudaDeviceGuard deviceGuard(cache.dataDeviceIds);
        // Paged cache payload lives in pagedKVCacheData. Its otherwise unused
        // cudaData owns the request's stable device page table, so normal Data
        // destruction releases metadata without touching shared page storage.
        if (!cache.cudaData) {
            cache.cudaData = FastllmCudaMalloc(cache.pagedKVCacheData->maxPages * sizeof(int));
            cache.cudaDataBorrowed = false;
        }
        if (!cache.pageIndex.empty()) {
            FastllmCudaCopyFromHostToDevice(cache.cudaData, cache.pageIndex.data(),
                                           cache.pageIndex.size() * sizeof(int));
        }
#endif
    }

    inline void Qwen4AttachPagedCache(PagedCacheManager &pool, Data &cache) {
        if (Qwen4IsPagedKV(cache) && cache.pagedKVCacheData == &pool) return;
        AssertInFastLLM(cache.dims.empty() || cache.dims[1] == 0,
                        "Qwen4 paged cache attachment requires an empty cache.\n");
        Qwen4ReleasePagedReference(cache);
        cache.FreeSpace();
        cache.dataType = pool.dataType;
        cache.UpdateUnitSize();
        cache.isFake = false;
        cache.directMemory = false;
        cache.isKVCache = cache.isPagedKVCache = true;
        cache.isLinearAttention = false;
        cache.pagedKVCacheData = &pool;
        cache.pageLen = pool.pageLen;
        cache.dataDevice = pool.dataDevice;
        cache.dataDeviceIds = pool.dataDeviceIds;
        // Logical dimensions remain compatible with QSA, MTP and the scheduler.
        cache.expansionDims.clear();
        cache.Resize({pool.dims[2], pool.maxPages * pool.pageLen, pool.dims[3]});
        cache.expansionDims = cache.dims;
        cache.expansionSize = pool.Count(0);
        cache.expansionBytes = pool.GetBytes();
        cache.Resize({pool.dims[2], 0, pool.dims[3]});
        Qwen4UpdatePageTable(cache);
    }

    inline void Qwen4SharePagedCache(const Data &source, Data &destination) {
        AssertInFastLLM(source.isPagedKVCache && source.pagedKVCacheData,
                        "Qwen4 paged cache share requires a pool.\n");
        if (&source == &destination) return;
        Qwen4ReleasePagedReference(destination);
        destination.FreeSpace();
        destination.isFake = false;
        destination.directMemory = false;
        destination.dataType = source.dataType;
        destination.UpdateUnitSize();
        destination.dataDevice = source.dataDevice;
        destination.dataDeviceIds = source.dataDeviceIds;
        destination.isKVCache = destination.isPagedKVCache = true;
        destination.isLinearAttention = source.isLinearAttention;
        destination.isLinearAttentionTransposed = source.isLinearAttentionTransposed;
        destination.pagedKVCacheData = source.pagedKVCacheData;
        destination.pageLen = source.pageLen;
        destination.pageIndex = source.pageIndex;
        if (!source.isLinearAttention) {
            destination.pageIndex.resize((source.dims[1] + source.pageLen - 1) / source.pageLen);
        }
        destination.lastPageLen = source.lastPageLen;
        destination.pagedKVCacheData->Pick(destination.pageIndex);
        destination.expansionDims = source.expansionDims;
        destination.expansionSize = source.expansionSize;
        destination.expansionBytes = source.expansionBytes;
        destination.Resize(source.dims);
        destination.strides = source.strides;
#ifdef USE_CUDA
        if (source.isLinearAttention) {
            destination.cudaData = source.cudaData;
            destination.cudaDataBorrowed = true;
        } else {
            Qwen4UpdatePageTable(destination);
        }
#endif
    }

    inline void Qwen4ReservePagedAppend(Data &cache, int end) {
        auto &pool = *cache.pagedKVCacheData;
#ifdef USE_CUDA
        Qwen4CudaDeviceGuard deviceGuard(cache.dataDeviceIds);
#endif
        const int previous = cache.dims[1];
        AssertInFastLLM(end >= previous && end <= pool.maxPages * cache.pageLen,
                        "Qwen4 paged cache capacity exceeded.\n");
        bool changed = cache.cudaData == nullptr;
        // Snapshots share committed pages. Only the partially filled last page
        // can be overwritten by append; detach that one page, not the history.
        if (end > previous && previous % cache.pageLen != 0) {
            const int last = (previous - 1) / cache.pageLen;
            int refs;
            {
                std::lock_guard<std::mutex> lock(pool.pageIndexLocker);
                refs = pool.pageRefCount[cache.pageIndex[last]];
            }
            if (refs > 1) {
                const int old = cache.pageIndex[last];
                const int page = pool.GetUnusedPageIndex(true);
#ifdef USE_CUDA
                const size_t bytes = pool.GetBytes() / pool.maxPages;
                FastllmCudaCopyFromDeviceToDevice(
                    (uint8_t*)pool.cudaData + page * bytes,
                    (uint8_t*)pool.cudaData + old * bytes, bytes);
#endif
                cache.pageIndex[last] = page;
                pool.ReleasePageIndex(old);
                changed = true;
            }
        }
        const int pages = (end + cache.pageLen - 1) / cache.pageLen;
        while ((int)cache.pageIndex.size() < pages) {
            cache.pageIndex.push_back(pool.GetUnusedPageIndex(true));
            changed = true;
        }
        if (changed) Qwen4UpdatePageTable(cache);
    }

    inline void Qwen4ResizeKVCache(Data &cache, int length) {
        if (Qwen4IsPagedKV(cache)) {
            const int pages = (length + cache.pageLen - 1) / cache.pageLen;
            AssertInFastLLM(pages <= (int)cache.pageIndex.size(),
                            "Qwen4 KV resize exceeds reserved pages.\n");
            if (pages < (int)cache.pageIndex.size()) {
                std::vector<int> released(cache.pageIndex.begin() + pages, cache.pageIndex.end());
                cache.pagedKVCacheData->ReleasePageIndices(released);
                cache.pageIndex.resize(pages);
            }
            cache.lastPageLen = length == 0 ? 0 : (length - 1) % cache.pageLen + 1;
        }
        auto dims = cache.dims;
        dims[1] = length;
        cache.Resize(dims);
    }

    inline void Qwen4AttachLinearSlot(PagedCacheManager &pool, Data &cache,
                                      const std::vector<int> &dims, bool transposed) {
#ifdef USE_CUDA
        if (cache.isPagedKVCache && cache.pagedKVCacheData == &pool) return;
        Qwen4CudaDeviceGuard deviceGuard(pool.dataDeviceIds);
        const int slot = pool.GetUnusedPageIndex(true);
        const size_t bytes = pool.GetBytes() / pool.maxPages;
        void *pointer = (uint8_t*)pool.cudaData + slot * bytes;
        if (cache.dataDevice == DataDevice::CUDA && cache.cudaData && cache.dims == dims) {
            FastllmCudaCopyFromDeviceToDevice(pointer, cache.cudaData, bytes);
        } else {
            FastllmCudaMemset0(pointer, bytes);
        }
        Qwen4ReleasePagedReference(cache);
        cache.FreeSpace();
        cache.isFake = false;
        cache.dataType = pool.dataType;
        cache.UpdateUnitSize();
        cache.dataDevice = DataDevice::CUDA;
        cache.dataDeviceIds = pool.dataDeviceIds;
        cache.isKVCache = cache.isPagedKVCache = cache.isLinearAttention = true;
        cache.isLinearAttentionTransposed = transposed;
        cache.pagedKVCacheData = &pool;
        cache.pageIndex = {slot};
        cache.pageLen = cache.lastPageLen = 1;
        cache.expansionDims.clear();
        cache.Resize(dims);
        cache.expansionDims = dims;
        cache.expansionSize = cache.Count(0);
        cache.expansionBytes = bytes;
        cache.cudaData = pointer;
        cache.cudaDataBorrowed = true;
#endif
    }

    inline void Qwen4CopyLinearState(const Data &source, Data &destination) {
#ifdef USE_CUDA
        if (destination.isPagedKVCache && destination.isLinearAttention) {
            Qwen4CudaDeviceGuard deviceGuard(destination.dataDeviceIds);
            AssertInFastLLM(source.dataDevice == DataDevice::CUDA &&
                            source.dataType == destination.dataType && source.dims == destination.dims,
                            "Qwen4 fixed linear slot shape mismatch.\n");
            FastllmCudaCopyFromDeviceToDevice(destination.cudaData, source.cudaData, source.GetBytes());
            destination.isLinearAttentionTransposed = source.isLinearAttentionTransposed;
            return;
        }
#endif
        destination.CopyFrom(source);
    }
}
