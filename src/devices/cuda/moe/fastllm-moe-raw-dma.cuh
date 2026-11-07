// Canonical pinned weights submitted to the copy engine by an
// independent host thread. This is not a CUDA host callback. All CUDA resources
// and API initialization are completed before a graph can publish a request.
struct RawExpertDmaRequest {
    alignas(64) unsigned published = 0;
    alignas(64) unsigned ready = 0;
    int table = 0, count = 0;
    uint8_t *destination = nullptr;
    int32_t experts[kMaxTopK]{};
    uint64_t offsets[kMaxTopK]{};
};

__global__ void PublishRawExpertDma(RawExpertDmaRequest *box, int table, uint8_t *destination,
        const int32_t *experts, const int32_t *slots, const int32_t *count,
        size_t stride, const uint64_t *offsets) {
    const int n = *count;
    if (threadIdx.x == 0) {
        box->ready = 0; box->table = table; box->count = n; box->destination = destination;
    }
    if (threadIdx.x < n && threadIdx.x < kMaxTopK) {
        const int slot = slots[threadIdx.x];
        box->experts[threadIdx.x] = experts[threadIdx.x];
        box->offsets[threadIdx.x] = offsets ? offsets[slot] : size_t(slot) * stride;
    }
    __threadfence_system(); __syncthreads();
    if (threadIdx.x == 0 && n) atomicExch_system(&box->published, 1u);
}

__global__ void WaitRawExpertDma(const RawExpertDmaRequest *box, const int32_t *count) {
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

struct RawExpertDmaSource {
    OffloadGroup &group;
    fastllm::cuda::MappedHostBuffer storage;
    RawExpertDmaRequest *host = nullptr, *mapped = nullptr;
    int ownerDevice = -1;
    cudaStream_t stream = nullptr;
    cudaEvent_t completed = nullptr;
    std::thread worker;
    std::atomic<bool> stopping{false}, initialized{false};
    std::atomic<int> error{int(cudaSuccess)};
    explicit RawExpertDmaSource(OffloadGroup &g) : group(g) {}
    ~RawExpertDmaSource() {
        stopping.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
        int previous = -1;
        cudaGetDevice(&previous);
        if (ownerDevice >= 0 && previous != ownerDevice) cudaSetDevice(ownerDevice);
        if (completed) cudaEventDestroy(completed);
        if (stream) cudaStreamDestroy(stream);
        storage.Reset();
        if (previous >= 0 && previous != ownerDevice) cudaSetDevice(previous);
    }
    bool Init(int device) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) { cudaGetLastError(); return false; }
        // The graph can wait on a concurrent upload only on devices with a
        // separate copy engine. Other devices retain mapped-kernel transfers.
        if (!properties.deviceOverlap || properties.asyncEngineCount == 0) return false;
        ownerDevice = device;
        if (storage.Allocate(sizeof(RawExpertDmaRequest)) != cudaSuccess) { cudaGetLastError(); return false; }
        host = new (storage.Host()) RawExpertDmaRequest;
        mapped = static_cast<RawExpertDmaRequest *>(storage.Device());
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&completed, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError(); return false;
        }
        std::vector<int> cpus;
#ifdef USE_NUMAS
        fastllm::HostMoePlacementScope placement(group.tableKeys.empty() ? nullptr : group.tableKeys.front());
        const auto sets = fastllm::GetNumasCudaWorkerCpuSets({device});
        if (sets.size() == 1) cpus = sets.front();
#endif
        worker = std::thread([this, device, cpus] {
#ifdef USE_NUMAS
            fastllm::BindNumasWorkerCpuSet(cpus);
#endif
            cudaError_t status = cudaSetDevice(device);
            void *scratch = nullptr;
            if (status == cudaSuccess) status = cudaMalloc(&scratch, 16);
            if (status == cudaSuccess) status = cudaMemcpyAsync(scratch,
                group.rawHostWeights.front()->cpuData, 16, cudaMemcpyHostToDevice, stream);
            if (status == cudaSuccess) status = cudaEventRecord(completed, stream);
            if (status == cudaSuccess) status = cudaEventSynchronize(completed);
            if (scratch) cudaFree(scratch);
            if (status != cudaSuccess) error.store(int(status), std::memory_order_release);
            initialized.store(true, std::memory_order_release);
            if (status != cudaSuccess) return;
            unsigned idle = 0;
            auto last = std::chrono::steady_clock::now();
            while (!stopping.load(std::memory_order_acquire)) {
                if (!__atomic_load_n(&host->published, __ATOMIC_ACQUIRE)) {
                    if (++idle == 512) {
                        idle = 0;
                        const auto quiet = std::chrono::steady_clock::now() - last;
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
                bool valid = table >= 0 && table < int(group.tableKeys.size()) && count > 0 && count <= kMaxTopK;
                for (int i = 0; valid && i < count; ++i)
                    valid = host->experts[i] >= 0 && host->experts[i] < group.layout.experts;
                status = valid ? cudaSuccess : cudaErrorInvalidValue;
                if (valid) {
                    const auto &layout = group.LayerLayout(table);
                    for (int i = 0; status == cudaSuccess && i < count; ++i) {
                        const size_t key = (size_t(table) * layout.experts + host->experts[i]) * 2;
                        auto *dst = host->destination + host->offsets[i];
                        status = cudaMemcpyAsync(dst, group.rawHostWeights[key]->cpuData,
                            layout.gateBytes, cudaMemcpyHostToDevice, stream);
                        if (status == cudaSuccess) status = cudaMemcpyAsync(dst + layout.downOffset,
                            group.rawHostWeights[key + 1]->cpuData, layout.downBytes, cudaMemcpyHostToDevice, stream);
                    }
                    // Acknowledge only after both DMA transfers are complete.
                    // The next publication follows the current routed reduction.
                    if (status == cudaSuccess) status = cudaEventRecord(completed, stream);
                    if (status == cudaSuccess) status = cudaEventSynchronize(completed);
                }
                if (status != cudaSuccess) {
                    // Earlier copies may already be queued when a later call
                    // fails; drain them before releasing the waiting graph.
                    cudaStreamSynchronize(stream);
                    error.store(int(status), std::memory_order_release);
                }
                last = std::chrono::steady_clock::now();
                __atomic_store_n(&host->published, 0u, __ATOMIC_RELEASE);
                __atomic_store_n(&host->ready, 1u, __ATOMIC_RELEASE);
            }
        });
        while (!initialized.load(std::memory_order_acquire)) std::this_thread::yield();
        return error.load(std::memory_order_acquire) == int(cudaSuccess);
    }
    void Check() const {
        const auto status = static_cast<cudaError_t>(error.load(std::memory_order_acquire));
        if (status != cudaSuccess) fastllm::AssertInFastLLM(false,
            std::string("Raw expert DMA failed: ") + cudaGetErrorString(status));
    }
    bool Copy(int table, uint8_t *destination, const int32_t *experts, const int32_t *slots,
            const int32_t *count, const uint64_t *offsets, cudaStream_t compute) {
        Check();
        PublishRawExpertDma<<<1, kMaxTopK, 0, compute>>>(mapped, table, destination,
            experts, slots, count, group.LayerLayout(table).recordStride, offsets);
        WaitRawExpertDma<<<1, 1, 0, compute>>>(mapped, count);
        return cudaGetLastError() == cudaSuccess;
    }
};
