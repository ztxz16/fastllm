// Only bounded, mutable state is copied. Full attention and global draft KV
// use the same evictable page trie as other FastLLM models.
struct PrefixWindow {
    std::array<std::vector<uint8_t>, 2> bytes;

    void transfer(VisitCache &visit, bool restore) {
        int index = 0;
        for (auto *side : {visit.key, visit.value}) {
            auto &buffer = bytes[index++];
            auto *pool = side->pagedKVCacheData;
            const size_t pageBytes = pool->GetBytes() / pool->maxPages;
            if (!restore) buffer.resize(side->pageIndex.size() * pageBytes);
            if (buffer.size() != side->pageIndex.size() * pageBytes)
                throw std::logic_error("Inkling prefix window geometry changed");
            for (size_t i = 0; i < side->pageIndex.size(); ++i) {
                auto *gpu = static_cast<uint8_t *>(pool->cudaData) + size_t(side->pageIndex[i]) * pageBytes;
                auto *host = buffer.data() + i * pageBytes;
                if (restore) check(cudaMemcpyAsync(gpu, host, pageBytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
                else check(cudaMemcpyAsync(host, gpu, pageBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
            }
        }
    }
};

struct PrefixState {
    int rows = 0, cols = 0;
    std::vector<float> values;

    void save(const Matrix &src) {
        rows = src.rows;
        cols = src.cols;
        values.resize(size_t(rows) * cols);
        if (!values.empty()) check(cudaMemcpyAsync(values.data(), src.p, values.size() * sizeof(float),
                                                   cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    Matrix load() const {
        Matrix dst(rows, cols);
        if (dst.p) check(cudaMemcpyAsync(dst.p, values.data(), values.size() * sizeof(float),
                                         cudaMemcpyHostToDevice, cudaStreamPerThread));
        return dst;
    }
};

struct InklingPrefix : RecurrentPrefixKey {
    std::vector<PrefixWindow> windows, draftWindows;
    std::vector<std::array<PrefixState, 4>> conv;
    PrefixState draftContext;
};
