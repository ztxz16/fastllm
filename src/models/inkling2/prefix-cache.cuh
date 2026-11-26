// Only bounded, mutable state is copied. Full attention and global draft KV
// stay in the shared page trie. Each TP rank owns its snapshot storage.
struct PrefixCopies {
    struct Range { void *live; size_t offset, bytes; };
    std::vector<Range> ranges;
    size_t bytes = 0;

    void add(void *live, size_t offset, size_t length) {
        if (!length) return;
        ranges.push_back({live, offset, length});
        bytes = std::max(bytes, offset + length);
    }

    void submit(const fastkv::Buffer &buffer, bool restore) const {
        auto *base = static_cast<uint8_t *>(buffer.MutableData());
        if (bytes > buffer.Size()) throw std::logic_error("Inkling prefix exceeds its snapshot");
        if (buffer.Device() < 0) {
            for (const auto &r : ranges) {
                void *host = base + r.offset;
                auto status = cudaMemcpyAsync(restore ? r.live : host, restore ? host : r.live, r.bytes,
                    restore ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, cudaStreamPerThread);
                if (status != cudaSuccess) {
                    cudaStreamSynchronize(cudaStreamPerThread);
                    check(status);
                }
            }
        } else {
            std::vector<void *> destinations;
            std::vector<const void *> sources;
            std::vector<size_t> lengths;
            for (const auto &r : ranges) {
                destinations.push_back(restore ? r.live : base + r.offset);
                sources.push_back(restore ? base + r.offset : r.live);
                lengths.push_back(r.bytes);
            }
            if (!FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                    destinations.data(), sources.data(), lengths.data(), int(lengths.size()))) {
                check(cudaStreamSynchronize(cudaStreamPerThread));
                throw std::runtime_error("Inkling batched prefix copy failed");
            }
        }
        // Lookup and eviction can run on another worker. Publish / release the
        // lease only after the writing / reading stream has finished with it.
        check(cudaStreamSynchronize(cudaStreamPerThread));
    }
};

struct PrefixWindow {
    struct Side { size_t offset = 0, bytes = 0; };
    std::array<Side, 2> sides;

    void transfer(VisitCache &visit, PrefixCopies &copies, bool restore) {
        int index = 0;
        for (auto *side : {visit.key, visit.value}) {
            auto &range = sides[index++];
            auto *pool = side->pagedKVCacheData;
            const size_t pageBytes = pool->GetBytes() / pool->maxPages;
            const size_t length = side->pageIndex.size() * pageBytes;
            if (!restore) range = {copies.bytes, length};
            if (range.bytes != length) throw std::logic_error("Inkling prefix window geometry changed");
            for (size_t i = 0; i < side->pageIndex.size(); ++i) {
                auto *gpu = static_cast<uint8_t *>(pool->cudaData) + size_t(side->pageIndex[i]) * pageBytes;
                copies.add(gpu, range.offset + i * pageBytes, pageBytes);
            }
        }
    }
};

struct PrefixState {
    int rows = 0, cols = 0;
    size_t offset = 0;

    void save(const Matrix &src, PrefixCopies &copies) {
        rows = src.rows;
        cols = src.cols;
        offset = copies.bytes;
        copies.add(src.p, offset, size_t(rows) * cols * sizeof(float));
    }

    Matrix load(PrefixCopies &copies) const {
        Matrix dst(rows, cols);
        copies.add(dst.p, offset, size_t(rows) * cols * sizeof(float));
        return dst;
    }
};

// Rebinding an ancestor in the shared page trie can leave an older recurrent
// snapshot indexed by the same tokens. Check the physical generations before
// pairing the private state with a trie path. These references do not pin KV.
struct PrefixPageGenerations {
    PagedCacheManager *pool;
    int slot, last;
    std::vector<long long> generations;
    PrefixPageGenerations(const Data &side, int slot, size_t count)
        : pool(side.pagedKVCacheData), slot(slot),
          last(count && count <= side.pageIndex.size() ? side.pageIndex[count - 1] : -1) {
        if (last < 0) return;
        std::lock_guard<std::mutex> lock(pool->pageIndexLocker);
        generations.reserve(count);
        for (size_t i = 0; i < count; ++i)
            generations.push_back(pool->pageTimestamp.at(side.pageIndex[i]));
    }

    bool current() const {
        if (last < 0 || GetPagedCacheManager(slot) != pool) return false;
        std::lock_guard<std::mutex> lock(pool->pageIndexLocker);
        auto *node = last < pool->maxPages ? pool->pageToTrieNode[last] : nullptr;
        for (auto it = generations.rbegin(); it != generations.rend(); ++it) {
            if (!node || node->pageId < 0 || node->timestamp != *it ||
                pool->pageTimestamp[node->pageId] != *it) return false;
            node = node->parent;
        }
        return node == pool->trieRoot;
    }
};

struct InklingPrefix : RecurrentPrefixKey {
    // A tail between regular prefill cuts was evaluated with a shorter GEMM.
    // Only a prompt ending in the same partial page reproduces that partition.
    int tail = 0, chunk = 0, interval = 0;
    std::vector<PrefixPageGenerations> targetPages, draftPages;
    bool pagesCurrent(bool draft) const {
        auto current = [](const PrefixPageGenerations &p) { return p.current(); };
        return std::all_of(targetPages.begin(), targetPages.end(), current) &&
               (!draft || std::all_of(draftPages.begin(), draftPages.end(), current));
    }
    fastkv::Buffer storage;
    std::vector<PrefixWindow> windows, draftWindows;
    std::vector<std::array<PrefixState, 4>> conv;
    PrefixState draftContext;
    bool hasDraft() const { return draftContext.rows && draftContext.cols; }
};
