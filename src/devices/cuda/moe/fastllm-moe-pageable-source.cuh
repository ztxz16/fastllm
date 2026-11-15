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

// Canonical raw matrices need no repacking. If their existing host pages can
// be registered, the graph reads them directly instead of waking a producer
// to duplicate the same bytes into mapped staging. No CUDA calls run from the
// graph's CPU producer, and registration failure retains the bounded path.
template<class Unit>
__global__ void CopyRawMappedExperts(const void *const *sources, uint8_t *destination,
        const int32_t *experts, const int32_t *slots, const int32_t *count,
        size_t gateUnits, size_t downUnits, size_t downOffset, size_t stride,
        const uint64_t *slotOffsets) {
    const size_t units = gateUnits + downUnits;
    const size_t total = size_t(*count) * units;
    for (size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total; flat += size_t(gridDim.x) * blockDim.x) {
        const size_t record = flat / units, offset = flat % units;
        const int part = offset >= gateUnits;
        const auto *src = static_cast<const Unit *>(sources[2 * experts[record] + part]);
        auto *dst = reinterpret_cast<Unit *>(destination +
            (slotOffsets ? slotOffsets[slots[record]] : size_t(slots[record]) * stride) + (part ? downOffset : 0));
        const size_t at = part ? offset - gateUnits : offset;
        dst[at] = src[at];
    }
}

#include "fastllm-moe-raw-dma.cuh"

struct PageableExpertSource {
    OffloadGroup &group;
    fastllm::cuda::MappedHostBuffer storage;
    PageableExpertRequest *host = nullptr, *mapped = nullptr;
    uint8_t *records = nullptr, *mappedRecords = nullptr, *auxiliary = nullptr, *mappedAuxiliary = nullptr;
    void **pointers = nullptr, **mappedPointers = nullptr;
    size_t stride = 0;
    int capacity = 0;
    std::shared_ptr<fastllm::cuda::HostCopyPool> copies;
    std::unique_ptr<RawExpertDmaSource> rawDma;
    std::vector<void *> directRegistrations;
    void **directPointers = nullptr;
    void ResetDirect() {
        cudaFree(directPointers); directPointers = nullptr;
        for (void *p : directRegistrations) cudaHostUnregister(p);
        directRegistrations.clear();
    }
    bool InitDirect(int device) {
#if defined(__linux__)
        if (group.rawHostWeights.empty()) return false;
        // Keep at least half of currently available host memory pageable.
        // Reading MemAvailable also accounts for reclaimable file cache.
        size_t available = 0;
        FILE *memory = std::fopen("/proc/meminfo", "r");
        if (!memory) return false;
        char line[256];
        while (std::fgets(line, sizeof(line), memory)) {
            unsigned long long kib = 0;
            if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) available = size_t(kib) * 1024;
        }
        std::fclose(memory);
        const long pageValue = sysconf(_SC_PAGESIZE);
        if (!available || pageValue <= 0) return false;
        const size_t page = size_t(pageValue);
        struct Range { uintptr_t begin, end, mapped = 0; };
        std::vector<Range> ranges;
        for (const auto *w : group.rawHostWeights) {
            const auto base = reinterpret_cast<uintptr_t>(w->cpuData);
            const size_t bytes = w->GetBytes();
            if (!base || !bytes || (base | bytes) & 15u || base + bytes < base ||
                base + bytes > std::numeric_limits<uintptr_t>::max() - page + 1) return false;
            ranges.push_back({base / page * page, (base + bytes + page - 1) / page * page});
        }
        std::sort(ranges.begin(), ranges.end(), [](const Range &a, const Range &b) { return a.begin < b.begin; });
        std::vector<Range> merged;
        for (const auto &v : ranges) {
            if (!merged.empty() && (v.begin < merged.back().end || (v.begin == merged.back().end && v.end - merged.back().begin <= (size_t(64) << 20)))) merged.back().end = std::max(merged.back().end, v.end);
            else merged.push_back(v);
        }
        size_t bytes = 0;
        for (const auto &v : merged) bytes += v.end - v.begin;
        if (bytes > available / 2) return false;
        // A registration must contain each complete matrix so its CUDA mapping
        // remains contiguous. Adjacent allocations share one page registration.
        for (auto &v : merged) {
            void *base = reinterpret_cast<void *>(v.begin), *mapped = nullptr;
            auto state = cudaHostRegister(base, v.end - v.begin, cudaHostRegisterMapped | cudaHostRegisterPortable);
            if (state == cudaSuccess) {
                directRegistrations.push_back(base);
                state = cudaHostGetDevicePointer(&mapped, base, 0);
            }
            if (state != cudaSuccess || !mapped) { cudaGetLastError(); ResetDirect(); return false; }
            v.mapped = reinterpret_cast<uintptr_t>(mapped);
        }
        std::vector<void *> pointers;
        for (const auto *w : group.rawHostWeights) {
            const auto address = reinterpret_cast<uintptr_t>(w->cpuData);
            auto it = std::upper_bound(merged.begin(), merged.end(), address,
                [](uintptr_t value, const Range &v) { return value < v.begin; });
            --it;
            pointers.push_back(reinterpret_cast<void *>(it->mapped + address - it->begin));
        }
        if (cudaMalloc(&directPointers, pointers.size() * sizeof(void *)) != cudaSuccess ||
            cudaMemcpy(directPointers, pointers.data(), pointers.size() * sizeof(void *), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError(); ResetDirect(); return false;
        }
        std::fprintf(stderr, "[Fastllm] Raw expert direct mapping on cuda:%d: %.3f GiB, %zu registrations; original host weights retained.\n",
            device, double(bytes) / (size_t(1) << 30), directRegistrations.size());
        return true;
#else
        return false;
#endif
    }
    std::thread worker;
    std::atomic<bool> stopping{false}, invalid{false};

    explicit PageableExpertSource(OffloadGroup &g) : group(g) {}
    ~PageableExpertSource() {
        stopping.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
        rawDma.reset();
        ResetDirect();
    }
    bool Init(int device) {
        if (InitDirect(device)) {
            auto candidate = std::make_unique<RawExpertDmaSource>(group);
            if (candidate->Init(device)) {
                rawDma = std::move(candidate);
                std::fprintf(stderr, "[Fastllm] Raw expert full-graph DMA enabled on cuda:%d; mapped-kernel fallback retained.\n", device);
            }
            return true;
        }
        capacity = std::min(kMaxTopK, group.layout.experts);
        int shards = group.sharedLayout.shards;
        if (!group.rawHostWeights.empty()) {
            shards=1; for (const auto &l:group.layerLayouts) stride=std::max(stride,l.recordStride);
        } else if (!group.ggufSources.empty()) {
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
                if (!__atomic_load_n(&host->published, __ATOMIC_ACQUIRE)) {
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
                __atomic_store_n(&host->published, 0u, __ATOMIC_RELEASE);
                __atomic_store_n(&host->ready, 1u, __ATOMIC_RELEASE);
            }
        });
        std::fprintf(stderr, "[Fastllm] Pageable expert source on cuda:%d: %.2f MiB bounded mapped staging for cache, dynamic routing and CUDA graphs.\n",
            device, double(bytes) / (1 << 20));
        return true;
    }
    void Stage(int table, int expert, int slot) {
        if (!group.rawHostWeights.empty()) {
            const auto &l=group.LayerLayout(table);const size_t key=(size_t(table)*l.experts+expert)*2;
            auto *dst=records+size_t(slot)*stride;
            copies->Copy(dst,group.rawHostWeights[key]->cpuData,l.gateBytes);
            copies->Copy(dst+l.downOffset,group.rawHostWeights[key+1]->cpuData,l.downBytes);
            return;
        }
        const bool gguf = !group.ggufSources.empty();
        const int shards = gguf ? group.ggufSources[table].shards : group.sharedLayout.shards;
        const size_t key = size_t(table) * group.layout.experts + expert;
        size_t offset = 0;
        for (int part = 0; part < 2; ++part) {
            const size_t bytes = gguf ? group.ggufSources[table].weights[part].Bytes() / shards : group.sharedLayout.weights[part].shardBytes;
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
    void Check() const {
        fastllm::AssertInFastLLM(!invalid.load(std::memory_order_acquire), "Invalid pageable expert graph request");
        if (rawDma) rawDma->Check();
    }
    bool Copy(int table, uint8_t *destination, const int32_t *experts,
            const int32_t *slots, const int32_t *count, const uint64_t *slotOffsets,
            int maxRecords, fastllm::cuda::RecordCopyLaunch launch, cudaStream_t stream, bool fullGraph = false) {
        Check();
        if (fullGraph && rawDma) return rawDma->Copy(table, destination, experts, slots, count, slotOffsets, stream);
        if (directPointers) {
            const auto &l = group.LayerLayout(table);
            CopyRawMappedExperts<uint4><<<launch.blocks, launch.threads, 0, stream>>>(
                const_cast<const void *const *>(directPointers) + size_t(table) * l.experts * 2,
                destination, experts, slots, count, l.gateBytes / 16, l.downBytes / 16,
                l.downOffset, l.recordStride, slotOffsets);
            return cudaGetLastError() == cudaSuccess;
        }
        PublishPageableExperts<<<1, 32, 0, stream>>>(mapped, table, experts, count);
        WaitPageableExperts<<<1, 1, 0, stream>>>(mapped, count);
        if (!group.rawHostWeights.empty()) {
            const auto &l=group.LayerLayout(table);
            return fastllm::cuda::CopyRecords({mappedRecords,destination,stride,l.recordStride,l.recordStride,slotOffsets},
                mapped->identity,slots,&mapped->count,maxRecords,launch,stream);
        }
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
