// Each loop visit owns independent K/V page references. The request keeps only
// the device address table and the convolution state outside the common pool.
inline int pageShift(int len) {
    if (len & (len - 1)) return -1;
    int shift = 0;
    for (; len > 1; len >>= 1) ++shift;
    return shift;
}

struct PageView {
    void *key{}, *value{};
    const int2 *ids{};
    int first{}, pageLen{}, shift{};

    __device__ size_t offset(int token, bool valueSide, int stride) const {
        int page = shift >= 0 ? token >> shift : token / pageLen;
        int2 id = ids[page - first];
        return (size_t(valueSide ? id.y : id.x) * pageLen + (token - page * pageLen)) * stride;
    }
};

struct VisitCache {
    std::array<Matrix, 4> convState;
    Data *key{}, *value{};
    Buffer<int2> table;
    std::vector<int2> ids;
    int capacity = 0, first = 0, shift = -1;

    PageView view() const {
        return {key->pagedKVCacheData->cudaData, value->pagedKVCacheData->cudaData,
                table.p, first, key->pageLen, shift};
    }

    void attach(Data &dst, int poolId, const Data &desc, int maxPages) {
        if (dst.isPagedKVCache && dst.pagedKVCacheData)
            dst.pagedKVCacheData->ReleasePageIndices(dst.pageIndex);
        dst.pageIndex.clear();
        auto *mgr = AllocatePagedCacheManager(poolId, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
                                              desc, -1, maxPages);
        if (mgr->dataType != desc.dataType || mgr->dims[2] != desc.dims[0] || mgr->dims[3] != desc.dims[2])
            throw std::runtime_error("Incompatible Inkling KV pool");
        bind(dst, mgr, {});
    }

    void bind(Data &dst, PagedCacheManager *mgr, std::vector<int> pages) {
        dst.dataType = mgr->dataType;
        dst.UpdateUnitSize();
        dst.isKVCache = dst.isPagedKVCache = true;
        dst.pagedKVCacheData = mgr;
        dst.pageLen = mgr->pageLen;
        shift = pageShift(dst.pageLen);
        dst.pageIndex = std::move(pages);
        if (!dst.pageIndex.empty()) mgr->Pick(dst.pageIndex);
        dst.lastPageLen = dst.pageIndex.empty() ? 0 : dst.pageLen;
        dst.dataDevice = mgr->dataDevice;
        dst.dataDeviceIds = mgr->dataDeviceIds;
        dst.expansionDims.clear();
        dst.Resize({mgr->dims[2], int(dst.pageIndex.size()) * dst.pageLen, mgr->dims[3]});
    }

    void reserveTable(int n) {
        if (n > capacity) {
            table = Buffer<int2>(n);
            capacity = n;
        }
    }

    void syncTable() {
        ids.resize(key->pageIndex.size());
        reserveTable(int(ids.size()));
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = make_int2(key->pageIndex[i], value->pageIndex[i]);
        if (!ids.empty()) check(cudaMemcpyAsync(table.p, ids.data(), ids.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, cudaStreamPerThread));
    }

    void prepare(int end, int window) {
        const int len = key->pageLen;
        const int begin = window ? std::max(0, end - window) : 0;
        const int next = begin / len, count = (end - 1) / len - next + 1;
        const int old = key->pageIndex.size();
        if (next != first || old != count) {
            // Recycle expired pages of this private request after the last read.
            // Global visits append page IDs; neither operation moves historical KV.
            if (count > capacity) {
                reserveTable(std::max(count, 2 * capacity));
            }

            for (Data *side : {key, value}) side->pageIndex.reserve(count);
            try {
                for (Data *side : {key, value}) {
                    auto &pages = side->pageIndex;
                    while (int(pages.size()) < count)
                        pages.push_back(side->pagedKVCacheData->GetUnusedPageIndex(true));
                }
            } catch (...) {
                for (Data *side : {key, value}) {
                    auto &pages = side->pageIndex;
                    while (int(pages.size()) > old) {
                        side->pagedKVCacheData->ReleasePageIndex(pages.back());
                        pages.pop_back();
                    }
                }
                throw;
            }
            const int drop = std::min(old, next - first);
            for (Data *side : {key, value}) {
                auto &pages = side->pageIndex;
                std::rotate(pages.begin(), pages.begin() + drop, pages.begin() + old);
                while (int(pages.size()) > count) {
                    side->pagedKVCacheData->ReleasePageIndex(pages.back());
                    pages.pop_back();
                }
            }
            syncTable();
            first = next;
        }
        for (Data *side : {key, value}) {
            side->lastPageLen = (end - 1) % len + 1;
            side->Resize({side->dims[0], end - begin, side->dims[2]});
        }
    }
};

struct DecodeVisit {
    PageView pages;
    float *history[4];
    int past;
    int verifyRemaining = 0;
};

template<class T>
__global__ void appendKv(PageView pages, const T *key, const T *value, int past, int skip, int width, int rows) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * width) return;
    const int row = skip + i / width, token = past + row, channel = i % width;
    static_cast<T *>(pages.key)[pages.offset(token, false, width) + channel] = key[size_t(row) * width + channel];
    static_cast<T *>(pages.value)[pages.offset(token, true, width) + channel] = value[size_t(row) * width + channel];
}
