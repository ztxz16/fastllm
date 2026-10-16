// Included once by qwen4-kernels.cu. The cache policy is independent of QSA
// ranking: retain requested hits, then use unrequested slots for missing pages.
// Overflow reads the host backing directly, never drops an attention token.
#include "models/qwen4_paged_cache.h"
namespace {
    void Qwen4KvMemCheck(cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(
            std::string("Qwen4 KVMem: ") + cudaGetErrorString(status));
    }

    __global__ void Qwen4KvMemMark(const int *indices, int count, int *needed,
                                  int length, const int *decodeMeta, int appended) {
        if (decodeMeta) length = decodeMeta[0] + appended;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
             i += blockDim.x * gridDim.x) {
            const int token = indices[i];
            if (token >= 0 && token < length) atomicExch(needed + token / 4, 1);
        }
    }

    __device__ void Qwen4KvMemEnqueue(bool take, int item, int *queue, int *count) {
        const unsigned mask = __ballot_sync(0xffffffffu, take);
        if (!mask) return;
        const int lane = threadIdx.x % 32, leader = __ffs(mask) - 1;
        int base = 0;
        if (lane == leader) base = atomicAdd(count, __popc(mask));
        base = __shfl_sync(0xffffffffu, base, leader);
        if (take) queue[base + __popc(mask & ((1u << lane) - 1))] = item;
    }

    __global__ void Qwen4KvMemFind(const int *needed, const int *pageSlots,
                                  const int *slotPages, int *misses, int *available,
                                  int *counts, int pages, int slots) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        Qwen4KvMemEnqueue(i < pages && needed[i] && pageSlots[i] < 0, i, misses, counts);
        Qwen4KvMemEnqueue(i < slots && (slotPages[i] < 0 || !needed[slotPages[i]]),
                          i, available, counts + 1);
    }

    __global__ void Qwen4KvMemRefill(const uint4 *hostKey, const uint4 *hostValue,
                                    uint4 *deviceKey, uint4 *deviceValue,
                                    int *pageSlots, int *slotPages,
                                    const int *misses, const int *available,
                                    const int *counts, unsigned long long *metrics, int vectors) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            // All slots start mapped to the resident prefix and remain mapped,
            // so every refill both replaces an old page and restores a new one.
            const int loaded = min(counts[0], counts[1]);
            metrics[0] += loaded;
            metrics[1] += loaded;
        }
        for (int i = blockIdx.x; i < min(counts[0], counts[1]); i += gridDim.x) {
            const int page = misses[i], slot = available[i];
            for (int j = threadIdx.x; j < vectors; j += blockDim.x) {
                deviceKey[size_t(slot) * vectors + j] = hostKey[size_t(page) * vectors + j];
                deviceValue[size_t(slot) * vectors + j] = hostValue[size_t(page) * vectors + j];
            }
            if (threadIdx.x == 0) {
                const int old = slotPages[slot];
                if (old >= 0) pageSlots[old] = -1;
                slotPages[slot] = page;
                pageSlots[page] = slot;
            }
        }
    }
}

namespace fastllm {
    Qwen4KvMem::Qwen4KvMem(const KvMemConfig &config, DataType type, int heads,
                          int dim, int device)
        : device(device), heads(heads), dim(dim), type(type), config(config) {
        if (heads <= 0 || dim <= 0 || dim % 8 || config.maxTokens <= 0 ||
            config.maxTokens > std::numeric_limits<int>::max() - pageTokens ||
            config.residentTokens <= 0 || config.residentTokens > std::numeric_limits<int>::max() - pageTokens ||
            (type != FLOAT16 && type != BFLOAT16 && type != FLOAT32))
            throw std::invalid_argument("Qwen4 KVMem: invalid storage shape or type");
        logicalPages = (config.maxTokens + pageTokens - 1) / pageTokens;
        residentPages = std::min(logicalPages,
            (config.residentTokens + pageTokens - 1) / pageTokens);
        capacity = logicalPages * pageTokens;
        resident = residentPages * pageTokens;
        const size_t unit = type == FLOAT32 ? 4 : 2;
        const size_t rowHeadBytes = size_t(dim) * pageTokens * unit;
        if (size_t(heads) > config.hostBytes / 2 / logicalPages / rowHeadBytes)
            throw std::invalid_argument("Qwen4 KVMem: host budget exceeded");
        pageBytes = rowHeadBytes * heads;
        stats.hostBytes = 2 * size_t(logicalPages) * pageBytes;
        int previous;
        Qwen4KvMemCheck(cudaGetDevice(&previous));
        Qwen4KvMemCheck(cudaSetDevice(device));
        try {
            const size_t bytes = size_t(logicalPages) * pageBytes;
            for (auto p : {&hostKey, &hostValue})
                Qwen4KvMemCheck(cudaHostAlloc(p, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
            Qwen4KvMemCheck(cudaHostGetDevicePointer(&mappedKey, hostKey, 0));
            Qwen4KvMemCheck(cudaHostGetDevicePointer(&mappedValue, hostValue, 0));
            for (auto p : {&deviceKey, &deviceValue})
                Qwen4KvMemCheck(cudaMalloc(p, size_t(residentPages) * pageBytes));
            const size_t words = (3 * size_t(logicalPages) + 2 * size_t(residentPages) + 3) & ~size_t(1);
            const size_t metadataBytes = words * sizeof(int) + 2 * sizeof(unsigned long long);
            Qwen4KvMemCheck(cudaMalloc(&metadata, metadataBytes));
            pageSlots = metadata;
            slotPages = pageSlots + logicalPages;
            needed = slotPages + residentPages;
            misses = needed + logicalPages;
            available = misses + logicalPages;
            counts = available + residentPages;
            metrics = reinterpret_cast<unsigned long long *>(metadata + words);
            InitializeSlots();
            gpuBytes = 2 * size_t(residentPages) * pageBytes + metadataBytes;
        } catch (...) {
            Release();
            cudaSetDevice(previous);
            throw;
        }
        Qwen4KvMemCheck(cudaSetDevice(previous));
    }

    void Qwen4KvMem::Release() noexcept {
        int previous = 0;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        // Requests can be destroyed by another host thread. All graph launches
        // referencing the backing store must finish before unpinning its pages.
        cudaDeviceSynchronize();
        if (metadata) cudaFree(metadata);
        if (deviceKey) cudaFree(deviceKey);
        if (deviceValue) cudaFree(deviceValue);
        if (hostKey) cudaFreeHost(hostKey);
        if (hostValue) cudaFreeHost(hostValue);
        cudaSetDevice(previous);
    }
    Qwen4KvMem::~Qwen4KvMem() { Release(); }

    void Qwen4KvMem::SetLength(int tokens) {
        if (tokens < 0 || tokens > capacity)
            throw std::invalid_argument("Qwen4 KVMem logical length exceeds capacity");
        stats.tokens = tokens;
        stats.residentPages = std::min(residentPages, (tokens + pageTokens - 1) / pageTokens);
    }

    const KvMemStats &Qwen4KvMem::Stats() const {
        Qwen4CudaDeviceGuard guard({device});
        Qwen4KvMemCheck(cudaDeviceSynchronize());
        unsigned long long values[2];
        Qwen4KvMemCheck(cudaMemcpy(values, metrics, sizeof(values), cudaMemcpyDeviceToHost));
        stats.restoredPages = values[0];
        stats.evictedPages = values[1];
        return stats;
    }

    void Qwen4KvMem::Reset() {
        Qwen4CudaDeviceGuard guard({device});
        // One admitted request owns this slot. Abort/reuse may arrive from a
        // different worker stream; finish old launches before recycling it.
        Qwen4KvMemCheck(cudaDeviceSynchronize());
        InitializeSlots();
    }

    void Qwen4KvMem::InitializeSlots() {
        Qwen4KvMemCheck(cudaMemset(pageSlots, 0xff, logicalPages * sizeof(int)));
        Qwen4KvMemCheck(cudaMemset(metrics, 0, 2 * sizeof(unsigned long long)));
        // The initial prefix is written directly into both tiers. Until
        // the first eviction, short prefill needs no selection/refill pass.
        std::vector<int> initial(residentPages);
        for (int page = 0; page < residentPages; ++page) initial[page] = page;
        Qwen4KvMemCheck(cudaMemcpy(pageSlots, initial.data(), initial.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        Qwen4KvMemCheck(cudaMemcpy(slotPages, initial.data(), initial.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        prefixResident = true;
        SetLength(0);
    }

    void Qwen4KvMem::Attach(Data &key, Data &value, const std::shared_ptr<Qwen4KvMem> &owner) {
        auto empty = [&](const Data &tensor) {
            return tensor.dims.empty() || tensor.dims == std::vector<int>({heads, 0, dim});
        };
        if (owner.get() != this || !empty(key) || !empty(value))
            throw std::invalid_argument("Qwen4 KVMem must attach to empty request tensors");
        auto sharedConfig = std::make_shared<KvMemConfig>(config);
        for (Data *tensor : {&key, &value}) {
            tensor->FreeSpace();
            tensor->dataType = type;
            tensor->UpdateUnitSize();
            tensor->dataDevice = CUDA;
            tensor->dataDeviceIds = {device};
            tensor->isKVCache = true;
            tensor->isPagedKVCache = false;
            tensor->isFake = false;
            tensor->expansionDims.clear();
            tensor->Resize({heads, capacity, dim});
            tensor->expansionDims = tensor->dims;
            tensor->expansionSize = tensor->Count(0);
            tensor->expansionBytes = size_t(residentPages) * pageBytes;
            tensor->Resize({heads, 0, dim});
            tensor->cudaData = tensor == &key ? deviceKey : deviceValue;
            tensor->cudaDataBorrowed = true;
            tensor->kvMemConfig = sharedConfig;
            tensor->kvMemCache = owner;
        }
        SetLength(0);
    }

    void Qwen4KvMem::Prepare(const Data &indices, int rowStart, int rows, int length,
                             const int32_t *decodeMeta, int appended) {
        if (!decodeMeta && HasResidentPrefix(length)) return;
        // A captured call has a dynamic length. Conservatively retire this
        // host-side shortcut before capture; replay always resolves residency.
        prefixResident = false;
        Qwen4KvMemCheck(cudaMemsetAsync(needed, 0, logicalPages * sizeof(int), cudaStreamPerThread));
        Qwen4KvMemCheck(cudaMemsetAsync(counts, 0, 2 * sizeof(int), cudaStreamPerThread));
        const int count = rows * indices.dims[1];
        Qwen4KvMemMark<<<std::min(1024, (count + 255) / 256), 256, 0, cudaStreamPerThread>>>(
            (const int *)indices.cudaData + size_t(rowStart) * indices.dims[1], count,
            needed, length, decodeMeta, appended);
        Qwen4KvMemFind<<<(logicalPages + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            needed, pageSlots, slotPages, misses, available, counts, logicalPages, residentPages);
        Qwen4KvMemRefill<<<std::min(residentPages, 1024), 128, 0, cudaStreamPerThread>>>(
            (const uint4 *)mappedKey, (const uint4 *)mappedValue,
            (uint4 *)deviceKey, (uint4 *)deviceValue, pageSlots, slotPages,
            misses, available, counts, metrics, pageBytes / sizeof(uint4));
        Qwen4KvMemCheck(cudaGetLastError());
    }

    // Qwen4's MTP already rolls back absolute logical lengths. Writes update
    // host and resident copies, so rejected rows are simply overwritten before
    // any later attention can see them; no cache-map rollback is necessary.
    void Qwen4KvMem::BeginTransaction(int) {
        throw std::logic_error("Qwen4 KVMem uses the model's absolute-length transactions");
    }
    void Qwen4KvMem::FinishTransaction(int, Data &, Data &) {
        throw std::logic_error("Qwen4 KVMem uses the model's absolute-length transactions");
    }
    void Qwen4KvMem::Append(const Data &, const Data &, const Data &, const Data &, Data &, Data &) {
        throw std::logic_error("Qwen4 KVMem requires Qwen4 CUDA append kernels");
    }
}
