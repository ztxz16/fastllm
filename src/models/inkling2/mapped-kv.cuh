// Publish contiguous K/V page pairs without changing a captured graph's table address.
__global__ void mapKvPages(int2 *table, int count, uint8_t *base, size_t pairBytes, int offset, int slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int slot = offset + i;
        const uintptr_t ptr = reinterpret_cast<uintptr_t>(base + size_t(slot < slots ? slot : slot % slots) * pairBytes);
        table[i] = make_int2(uint32_t(ptr), uint32_t(ptr >> 32));
    }
}

// Host pages own the retained KV. Only sealed (immutable) pages enter the device ring.
// A speculative write therefore always targets the host, even after a rejection.
class MappedKvPages {
    const int limit, resident, hostPages;
    const size_t pairBytes;
    Buffer<int2> table;
    Buffer<uint8_t> hot;
    std::unique_ptr<uint8_t, decltype(&cudaFreeHost)> host{nullptr, &cudaFreeHost};
    uint8_t *mapped = nullptr;
    int sealed = 0;
    Buffer<uint8_t> retrieval;
    DataType type{};
    int width = 0;

    template<class KV> void indexPages(int first, int end) {
        while (first < end) {
            const int next = first + std::min(end - first, 65535);
            kv_selection::index<KV><<<dim3((width + 127) / 128, next - first),128,0,cudaStreamPerThread>>>(
                selector(), first, next);
            check(cudaGetLastError());
            first = next;
        }
    }

    void map(int first, int count, uint8_t *base, int offset, int slots) {
        mapKvPages<<<1 + (count - 1) / 128, 128, 0, cudaStreamPerThread>>>(
            table.p + first, count, base, pairBytes, offset, slots);
        check(cudaGetLastError());
    }
public:
    MappedKvPages(MappedPageCounts pages, size_t bytes)
        : limit(pages.logical), resident(pages.resident), hostPages(pages.host),
          pairBytes(bytes), table(limit), hot(resident * bytes) {
        uint8_t *ptr = nullptr;
        check(cudaHostAlloc(&ptr, size_t(hostPages) * pairBytes, cudaHostAllocMapped | cudaHostAllocPortable));
        host.reset(ptr);
        check(cudaHostGetDevicePointer(&mapped, ptr, 0));
        map(0, limit, mapped, 0, hostPages);
    }
    // Prefill attends to its entire current chunk before appending the KV.
    // Discard the expired prefix of a long write so two CTAs never write the same host slot.
    int appendRows(int rows, int pageLen) const {
        return hostPages == limit ? rows : int(std::min<int64_t>(rows, int64_t(hostPages - 1) * pageLen + 1));
    }
    const int2 *addresses() const { return table.p; }

    kv_selection::State *selector() const { return reinterpret_cast<kv_selection::State *>(retrieval.p); }
    void selectPages(const KvMemConfig &cfg, DataType dtype, int pageLen, int heads, int kvHeads, int dim) {
        type = dtype;
        width = kvHeads * dim;
        retrieval = Buffer<uint8_t>(kv_selection::Bytes(limit, resident, width, heads));
        check(kv_selection::Initialize(selector(), cfg, limit, resident, pageLen, heads, kvHeads, dim,
                                       table.p, hot.p, int(pairBytes), cudaStreamPerThread));
    }

    void prepare(int pages, int committedPages) {
        if (pages > limit || committedPages > limit || committedPages < sealed)
            throw std::length_error("Inkling KVMem request exceeds its logical page range");
        if (committedPages == sealed) return;
        if (selector()) {
            if (type == FLOAT32) indexPages<float>(sealed, committedPages);
            else if (type == BFLOAT16) indexPages<__nv_bfloat16>(sealed, committedPages);
            else if (type == FLOAT16) indexPages<half>(sealed, committedPages);
            else indexPages<__nv_fp8_e4m3>(sealed, committedPages);
            sealed = committedPages;
            return;
        }
        const int begin = std::max(0, committedPages - resident);
        const int firstEvicted = std::max(0, sealed - resident);
        const int endEvicted = std::min(sealed, begin);
        if (endEvicted > firstEvicted)
            map(firstEvicted, endEvicted - firstEvicted, mapped, firstEvicted % hostPages, hostPages);
        for (int p = std::max(begin, sealed); p < committedPages;) {
            const int slot = p % resident;
            const int source = p % hostPages;
            const int n = std::min({committedPages - p, resident - slot, hostPages - source});
            auto *dst = hot.p + slot * pairBytes;
            check(cudaMemcpyAsync(dst, host.get() + size_t(source) * pairBytes, n * pairBytes,
                                  cudaMemcpyHostToDevice, cudaStreamPerThread));
            map(p, n, dst, 0, n);
            p += n;
        }
        sealed = committedPages;
    }
};
