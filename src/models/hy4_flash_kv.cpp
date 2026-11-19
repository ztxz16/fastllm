#include "hy4_flash_ops.h"

#ifdef FASTLLM_ENABLE_HY4_CUDA
namespace fastllm::hy4 {
namespace {
// Each slot holds one K/V pair. Indexed layers keep the token and pooled
// index keys in the trailing columns of K, as in the contiguous layout.
struct PageSlab : PagedCacheManager {
    PageSlab(int device, int keyWidth, int valueWidth) {
        constexpr int slots = 8;
        type = PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
        pageLen = KVPageTokens;
        dataType = BFLOAT16;
        directMemory = true;
        dataDevice = CUDA;
        dataDeviceIds = {device};
        Resize({slots, pageLen, 1, keyWidth + valueWidth});
        Allocate(false);
        SetMaxPages(slots);
    }
    ~PageSlab() { EvictTrieSubtree(trieRoot); }
};
struct PageRef {
    std::shared_ptr<PageSlab> slab;
    int slot;
};
} // namespace

class KVPagePool {
  public:
    KVPagePool(int device, int keyWidth, int valueWidth)
        : device(device), keyWidth(keyWidth), valueWidth(valueWidth) {}
    PageRef Acquire() {
        std::lock_guard<std::mutex> lock(mutex);
        if (available.empty()) {
            auto slab = std::make_shared<PageSlab>(device, keyWidth, valueWidth);
            available.push_back(std::move(slab));
        }
        auto slab = available.back();
        int slot = slab->GetUnusedPageIndex(true);
        if (!slab->FreePageCount())
            available.pop_back();
        return {std::move(slab), slot};
    }
    void Release(const PageRef &ref) {
        std::lock_guard<std::mutex> lock(mutex);
        auto &slab = ref.slab;
        bool full = !slab->FreePageCount();
        slab->ReleasePageIndex(ref.slot);
        if (full)
            available.push_back(slab);
        // Keep at most one completely idle slab of this layout.
        if (slab->FreePageCount() == slab->maxPages) {
            auto spare = idle.lock();
            if (spare && spare != slab && spare->FreePageCount() == spare->maxPages)
                available.erase(std::find(available.begin(), available.end(), slab));
            else
                idle = slab;
        }
    }

  private:
    int device, keyWidth, valueWidth;
    std::mutex mutex;
    std::vector<std::shared_ptr<PageSlab>> available;
    std::weak_ptr<PageSlab> idle;
};

namespace {
struct RequestPages {
    RequestPages(const std::shared_ptr<KVPagePool> &pool, int device, int retained)
        : pool(pool), device(device), capacity((retained + KVPageTokens - 1) / KVPageTokens),
          table(static_cast<KVPage *>(Allocate(device, size_t(capacity) * sizeof(KVPage)))) {}
    ~RequestPages() {
        Free(device, table);
        for (auto &ref : pages)
            pool->Release(ref);
    }
    void Ensure(int count, int keyWidth, int valueWidth) {
        if (count <= published)
            return;
        AssertInFastLLM(count <= capacity, "HY4 KV exceeds its page table.");
        if (size_t(count) > pages.capacity())
            pages.reserve(std::max(size_t(count), 2 * pages.capacity()));
        while (int(pages.size()) < count)
            pages.push_back(pool->Acquire());
        std::vector<KVPage> suffix;
        suffix.reserve(count - published);
        for (int i = published; i < count; ++i) {
            const auto &ref = pages[i];
            auto base = static_cast<uint16_t *>(ref.slab->cudaData) +
                        size_t(ref.slot) * KVPageTokens * (keyWidth + valueWidth);
            suffix.push_back({base, base + KVPageTokens * keyWidth});
        }
        CopyToGPU(table + published, suffix.data(), suffix.size() * sizeof(KVPage));
        published = count;
    }
    std::shared_ptr<KVPagePool> pool;
    int device, capacity, published = 0;
    KVPage *table;
    std::vector<PageRef> pages;
};
} // namespace

// Keep tentative pages for reuse while hiding rows rejected by verification.
void TruncateKV(Data &key, Data &value, int end) {
    for (auto *data : {&key, &value}) {
        auto dims = data->dims;
        dims[1] = std::min(end, dims[1]);
        data->Resize(dims);
    }
}

const KVPage *ReservePages(KVPools &pools, int device, Data &key, Data &value, int end, int retained,
                           int keyWidth, int valueWidth) {
    // The first key can already own the DFly request. The value owns the paired pages.
    auto pages = static_cast<RequestPages *>(value.kvStorageOwner.get());
    if (!pages) {
        auto &pool = pools.layouts[{keyWidth, valueWidth}];
        if (!pool)
            pool = std::make_shared<KVPagePool>(device, keyWidth, valueWidth);
        value.kvStorageOwner = std::make_shared<RequestPages>(pool, device, retained);
        pages = static_cast<RequestPages *>(value.kvStorageOwner.get());
        for (auto *data : {&key, &value}) {
            data->dataType = BFLOAT16;
            data->dataDevice = CUDA;
            data->dataDeviceIds = {device};
        }
    }
    int length = std::min(end, retained);
    pages->Ensure((length + KVPageTokens - 1) / KVPageTokens, keyWidth, valueWidth);
    // The scheduler accounts for allocated tokens separately from the live
    // prefix. A rejected draft can shorten the prefix while its pages stay owned.
    const int reserved = pages->published * KVPageTokens;
    auto resizeView = [&](Data &data, int width) {
        if (data.expansionDims.empty() || data.expansionDims[1] != reserved) {
            data.expansionDims.clear();
            data.Resize({1, reserved, width});
            data.expansionDims = data.dims;
        }
        data.Resize({1, length, width});
    };
    resizeView(key, keyWidth);
    resizeView(value, valueWidth);
    return pages->table;
}
} // namespace fastllm::hy4
#endif
