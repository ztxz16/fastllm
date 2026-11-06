#pragma once

// Included after the cache/source layouts. Each device cache serializes its
// forwards and reuses one bounded source in graph order. Host admissions use
// independent upload rings, including while a full graph is running.
// The CPU producer copies bytes only: CUDA calls during a graph's acquire
// wait could synchronize with that same graph and deadlock.
struct PageableExpertRequest {
    alignas(64) unsigned published = 0;
    alignas(64) unsigned ready = 0;
    int table = 0, count = 0;
    int32_t experts[kMaxTopK]{}, identity[kMaxTopK]{};
};

__global__ void PublishPageableExperts(PageableExpertRequest *box, int table,
        const int32_t *experts, const int32_t *count) {
    const int n = *count;
    if (threadIdx.x == 0) { box->ready = 0; box->table = table; box->count = n; }
    if (threadIdx.x < n && threadIdx.x < kMaxTopK) box->experts[threadIdx.x] = experts[threadIdx.x];
    __threadfence_system(); __syncthreads();
    if (threadIdx.x == 0 && n) atomicExch_system(&box->published, 1u);
}

__global__ void WaitPageableExperts(const PageableExpertRequest *box, const int32_t *count) {
    if (!*count) return;
    unsigned ready;
    do {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ready) : "l"(&box->ready) : "memory");
        if (!ready) __nanosleep(64);
#else
        ready = *reinterpret_cast<const volatile unsigned *>(&box->ready);
        __threadfence_system();
#endif
    } while (!ready);
}

struct PageableExpertSource {
    OffloadGroup &group;
    fastllm::cuda::MappedHostBuffer storage;
    PageableExpertRequest *host = nullptr, *mapped = nullptr;
    uint8_t *records = nullptr, *mappedRecords = nullptr, *auxiliary = nullptr, *mappedAuxiliary = nullptr;
    void **pointers = nullptr, **mappedPointers = nullptr;
    size_t stride = 0;
    int capacity = 0;
    std::shared_ptr<fastllm::cuda::HostCopyPool> copies;
    std::thread worker;
    std::atomic<bool> stopping{false}, invalid{false};

    explicit PageableExpertSource(OffloadGroup &g) : group(g) {}
    ~PageableExpertSource() {
        stopping.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
    }
    bool Init(int device) {
        capacity = std::min(kMaxTopK, group.layout.experts);
        int shards = group.sharedLayout.shards;
        if (!group.ggufSources.empty()) {
            for (const auto &s : group.ggufSources) { stride = std::max(stride, s.stride); shards = std::max(shards, s.shards); }
        } else stride = size_t(group.sharedLayout.sourceBytes[0]) + group.sharedLayout.sourceBytes[1];
        if (!stride || !shards) return false;
        stride = (stride + 63) & ~size_t(63);
        const size_t pointerOffset = (sizeof(PageableExpertRequest) + 63) & ~size_t(63);
        const size_t dataOffset = (pointerOffset + capacity * 2 * shards * sizeof(void *) + 63) & ~size_t(63);
        const size_t auxiliaryOffset = dataOffset + capacity * stride;
        const size_t bytes = auxiliaryOffset + capacity * group.sharedLayout.auxiliaryBytes;
        const auto status = storage.Allocate(bytes);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "[Fastllm] Pageable expert graph staging needs %.2f MiB of mapped host memory on cuda:%d: %s.\n",
                double(bytes) / (1 << 20), device, cudaGetErrorString(status));
            cudaGetLastError(); return false;
        }
        auto *h = static_cast<uint8_t *>(storage.Host()), *d = static_cast<uint8_t *>(storage.Device());
        memset(h, 0, bytes);
        host = new (h) PageableExpertRequest{}; mapped = reinterpret_cast<PageableExpertRequest *>(d);
        pointers = reinterpret_cast<void **>(h + pointerOffset); mappedPointers = reinterpret_cast<void **>(d + pointerOffset);
        records = h + dataOffset; mappedRecords = d + dataOffset;
        auxiliary = h + auxiliaryOffset; mappedAuxiliary = d + auxiliaryOffset;
        for (int i = 0; i < kMaxTopK; ++i) host->identity[i] = i;
        std::vector<int> cpus;
#ifdef USE_NUMAS
        fastllm::HostMoePlacementScope placement(group.tableKeys.empty() ? nullptr : group.tableKeys.front());
        const auto sets = fastllm::GetNumasCudaWorkerCpuSets({device});
        if (sets.size() == 1) cpus = sets.front();
#endif
        copies = fastllm::cuda::GetHostCopyPool(device, cpus);
        worker = std::thread([this, cpus] {
#ifdef USE_NUMAS
            fastllm::BindNumasWorkerCpuSet(cpus);
#endif
            unsigned idle = 0;
            auto lastActive = std::chrono::steady_clock::now();
            while (!stopping.load(std::memory_order_acquire)) {
                if (!fastllm::cuda::LoadHostSignal(&host->published)) {
                    if (++idle == 512) {
                        idle = 0;
                        const auto quiet = std::chrono::steady_clock::now() - lastActive;
                        if (quiet > std::chrono::milliseconds(100)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        else if (quiet > std::chrono::milliseconds(5)) std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
#if defined(__x86_64__)
                    else __builtin_ia32_pause();
#else
                    else std::this_thread::yield();
#endif
                    continue;
                }
                idle = 0;
                const int table = host->table, count = host->count;
                bool valid = table >= 0 && table < int(group.tableKeys.size()) && count > 0 && count <= capacity;
                for (int i = 0; valid && i < count; ++i)
                    valid = host->experts[i] >= 0 && host->experts[i] < group.layout.experts;
                if (valid) for (int i = 0; i < count; ++i) Stage(table, host->experts[i], i);
                else { host->count = 0; invalid.store(true, std::memory_order_release); }
                lastActive = std::chrono::steady_clock::now();
                fastllm::cuda::StoreHostSignal(&host->published, 0u);
                fastllm::cuda::StoreHostSignal(&host->ready, 1u);
            }
        });
        std::fprintf(stderr, "[Fastllm] Pageable expert source on cuda:%d: %.2f MiB bounded mapped staging for cache, dynamic routing and CUDA graphs.\n",
            device, double(bytes) / (1 << 20));
        return true;
    }
    void Stage(int table, int expert, int slot) {
        const bool gguf = !group.ggufSources.empty();
        const int shards = gguf ? group.ggufSources[table].shards : group.sharedLayout.shards;
        const size_t key = size_t(table) * group.layout.experts + expert;
        size_t offset = 0;
        for (int part = 0; part < 2; ++part) {
            const size_t bytes = gguf ? size_t(group.ggufSources[table].weights[part].rows) *
                group.ggufSources[table].weights[part].rowBytes / shards : group.sharedLayout.weights[part].shardBytes;
            if (gguf && part) offset = group.ggufSources[table].downOffset;
            for (int node = 0; node < shards; ++node) {
                copies->Copy(records + size_t(slot) * stride + offset,
                    group.numaPointers[(key * 2 + part) * shards + node], bytes);
                pointers[(slot * 2 + part) * shards + node] = mappedRecords + size_t(slot) * stride + offset;
                offset += bytes;
            }
        }
        const size_t aux = group.sharedLayout.auxiliaryBytes;
        if (aux) memcpy(auxiliary + size_t(slot) * aux, group.hostRecords + key * aux, aux);
    }
    bool Copy(int table, uint8_t *destination, const int32_t *experts,
            const int32_t *slots, const int32_t *count, const uint64_t *slotOffsets,
            int maxRecords, fastllm::cuda::RecordCopyLaunch launch, cudaStream_t stream) {
        fastllm::AssertInFastLLM(!invalid.load(std::memory_order_acquire), "Invalid pageable expert graph request");
        PublishPageableExperts<<<1, 32, 0, stream>>>(mapped, table, experts, count);
        WaitPageableExperts<<<1, 1, 0, stream>>>(mapped, count);
        if (!group.ggufSources.empty()) {
            const auto &s = group.ggufSources[table];
            return (NativeStandardGGUFRecords(group)
                ? fastllm_gguf_restore::CopyNativeRecords(s, mappedPointers, destination, stream,
                    mapped->identity, slots, &mapped->count, slotOffsets)
                : fastllm_gguf_restore::CopyRecords(s, mappedPointers, destination, stream,
                    mapped->identity, slots, &mapped->count, slotOffsets)) == cudaSuccess;
        }
        return fastllm::cuda::CopySharedExpertRecords(group.sharedLayout, mappedPointers,
            group.sharedLayout.auxiliaryBytes ? mappedAuxiliary : nullptr, destination,
            mapped->identity, slots, &mapped->count, maxRecords, launch, stream);
    }
};
