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
    kv_selection::State *selection{};

    template<class T> __device__ T *pageBase(int2 id, bool v, size_t pageStride) const {
        if (key) return static_cast<T *>(v ? value : key) + size_t(v ? id.y : id.x) * pageStride;
        const uintptr_t address = uint32_t(id.x) | (uintptr_t(uint32_t(id.y)) << 32);
        return reinterpret_cast<T *>(address) + (v ? pageStride : 0);
    }
    template<class T> __device__ T *row(int token, bool v, int stride) const {
        const int page = shift >= 0 ? token >> shift : token / pageLen;
        return pageBase<T>(ids[page - first], v, size_t(pageLen) * stride) + size_t(token - page * pageLen) * stride;
    }
};

#include "mapped-kv.cuh"

struct VisitCache {
    std::unique_ptr<MappedKvPages> mapped;
    const int *committed = nullptr;
    std::array<Matrix, 4> convState;
    Data *key{}, *value{};
    Buffer<int2> table;
    std::vector<int2> ids;
    size_t capacity = 0, tableOffset = 0;
    int first = 0, shift = -1, syncedPages = 0;

    int2 *pageTable() const { return table.p ? table.p + tableOffset : nullptr; }

    PageView view(std::vector<VisitCache *> *pending = nullptr) {
        if (mapped) return {nullptr, nullptr, mapped->addresses(), 0, key->pageLen, shift, mapped->selector()};
        if (syncedPages < int(key->pageIndex.size())) {
            if (pending) pending->push_back(this);
            else syncTable();
        }
        return {key->pagedKVCacheData->cudaData, value->pagedKVCacheData->cudaData,
                pageTable(), first, key->pageLen, shift};
    }

    void attachMapped(const Data &desc, MappedPageCounts pages, const int &length,
                      const KvMemConfig *selection = nullptr, int qHeads = 0) {
        const int len = GetPageLen();
        const size_t sideBytes = GetDataBytes(desc.dataType, size_t(len) * desc.dims[0], desc.dims[2]);
        mapped = std::make_unique<MappedKvPages>(pages, 2 * sideBytes);
        if (selection) mapped->selectPages(*selection, desc.dataType, len, qHeads, desc.dims[0], desc.dims[2]);
        committed = &length;
        shift = pageShift(len);
        for (auto *side : {key, value}) {
            side->dataType = desc.dataType;
            side->UpdateUnitSize();
            side->isKVCache = true;
            side->pageLen = len;
            side->Resize({desc.dims[0], 0, desc.dims[2]});
        }
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
        syncedPages = 0;
        if (!dst.pageIndex.empty()) mgr->Pick(dst.pageIndex);
        dst.lastPageLen = dst.pageIndex.empty() ? 0 : dst.pageLen;
        dst.dataDevice = mgr->dataDevice;
        dst.dataDeviceIds = mgr->dataDeviceIds;
        dst.expansionDims.clear();
        dst.Resize({mgr->dims[2], int(dst.pageIndex.size()) * dst.pageLen, mgr->dims[3]});
    }

    // A bounded window advances within the allocation. Only when it runs out
    // of room do the surviving page IDs need republishing at the beginning.
    void reserveTable(int64_t entries, bool sliding = false) {
        if (entries <= 0) return;
        const size_t n = size_t(entries) * (sliding ? 2 : 1);
        if (n > capacity) {
            table = Buffer<int2>(n);
            capacity = n;
            tableOffset = 0;
            syncedPages = 0;
        }
    }

    int2 pageId(int i) const {
        return make_int2(key->pageIndex[i], value->pageIndex[i]);
    }

    void syncTable() {
        const int count = key->pageIndex.size();
        reserveTable(count);
        ids.resize(count - syncedPages);
        for (int i = syncedPages; i < count; ++i) ids[i - syncedPages] = pageId(i);
        if (!ids.empty()) check(cudaMemcpyAsync(pageTable() + syncedPages, ids.data(), ids.size() * sizeof(int2),
                                                cudaMemcpyHostToDevice, cudaStreamPerThread));
        syncedPages = count;
    }

    void reservePages(int end, int window) {
        const int len = key->pageLen;
        if (mapped) {
            mapped->prepare((end - 1) / len + 1, *committed / len);
            return;
        }
        const int begin = window ? std::max(0, end - window) : 0;
        const int next = begin / len, count = (end - 1) / len - next + 1;
        const int old = key->pageIndex.size();
        if (next == first && old == count) return;
        if (next < first) throw std::logic_error("Inkling KV reservation precedes the retained window");
        // Recycle expired pages of this private request after the last read.
        // Global visits append page IDs; neither operation moves historical KV.
        if (size_t(count) > capacity) {
            reserveTable(std::max(size_t(count), 2 * capacity));
        }

        try {
            for (Data *side : {key, value}) {
                auto &pages = side->pageIndex;
                try {
                    side->pagedKVCacheData->ExtendPageIndices(pages, count);
                } catch (const std::length_error &) {
                    throw CudaWorkspaceExhausted("Inkling KV pool exhausted at token " + std::to_string(end) +
                        " (window " + std::to_string(window) + ", pool " +
                        std::to_string(side->pagedKVCacheData->maxPages) + " pages)");
                }
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
        if (tableOffset + drop + count <= capacity) {
            tableOffset += drop;
            syncedPages = std::clamp(syncedPages - drop, 0, count);
        } else {
            tableOffset = 0;
            syncedPages = 0;
        }
        first = next;
    }

    void prepare(int end, int window) {
        // The verifier already reserved its writes. At commit only publish the
        // accepted length; leave promotion to the next forward, before its reads.
        if (!mapped || end > *committed) reservePages(end, window);
        const int length = window ? std::min(end, window) : end;
        const int last = (end - 1) % key->pageLen + 1;
        for (Data *side : {key, value}) {
            side->lastPageLen = last;
            if (side->dims[1] == length) continue;
            // bind() creates a contiguous [heads, tokens, width] KV view.
            if (side->expansionDims.empty() && !side->multiDeviceData) {
                side->dims[1] = length;
                side->strides[0] = uint64_t(length) * side->strides[1];
            } else {
                side->Resize({side->dims[0], length, side->dims[2]});
            }
        }
    }
};

// The scheduler can prepare the next verifier while its draft is running.
// Pinned sources let its metadata copies queue without draining that stream.
template<class T> struct RoundHostAllocator {
    using value_type = T;
    RoundHostAllocator() = default;
    template<class U> RoundHostAllocator(const RoundHostAllocator<U> &) {}
    T *allocate(size_t n) {
        T *p = nullptr;
        check(cudaMallocHost(&p, n * sizeof(T)));
        return p;
    }
    void deallocate(T *p, size_t) { cudaFreeHost(p); }
    template<class U> bool operator==(const RoundHostAllocator<U> &) const { return true; }
};
template<class T> using RoundHostVector = std::vector<T, RoundHostAllocator<T>>;

struct PageTableWrite {
    int2 *dest;
    int2 pages;
};

__global__ void applyPageTableWrites(const PageTableWrite *writes, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) *writes[i].dest = writes[i].pages;
}

// A DSpark round collects one view per cache before either graph runs. The
// table addresses stay request-owned; only changed entries cross the bus.
struct PageTableBatch {
    RoundHostVector<PageTableWrite> host;
    Buffer<PageTableWrite> device;
    size_t capacity = 0;

    void upload(const std::vector<VisitCache *> &pending) {
        if (pending.empty()) return;
        if (pending.size() == 1) {
            pending.front()->syncTable();
            return;
        }
        host.clear();
        for (auto *v : pending) {
            const int count = v->key->pageIndex.size();
            for (int i = v->syncedPages; i < count; ++i)
                host.push_back({v->pageTable() + i, v->pageId(i)});
        }
        if (host.size() > capacity) {
            const size_t n = std::max(host.size(), capacity * 2);
            device = Buffer<PageTableWrite>(n);
            capacity = n;
        }
        check(cudaMemcpyAsync(device.p, host.data(), host.size() * sizeof(PageTableWrite),
                              cudaMemcpyHostToDevice, cudaStreamPerThread));
        applyPageTableWrites<<<(host.size() + 255) / 256, 256, 0, cudaStreamPerThread>>>(device.p, host.size());
        check(cudaGetLastError());
        for (auto *v : pending) v->syncedPages = v->key->pageIndex.size();
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
    pages.row<T>(token, false, width)[channel] = key[size_t(row) * width + channel];
    pages.row<T>(token, true, width)[channel] = value[size_t(row) * width + channel];
}
