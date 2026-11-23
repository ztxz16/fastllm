// One CTA waits on the bounded host staging area. Resident experts and the
// NUMA branch keep running while the copy worker fills the selected records.
__global__ void waitForExpertSource(const unsigned *ready, const int *control) {
    if (!*control) return;
#if CUDART_VERSION >= 12000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_system> signal(*const_cast<unsigned *>(ready));
    while (!signal.load(::cuda::memory_order_acquire)) __nanosleep(64);
#else
    while (!*reinterpret_cast<const volatile unsigned *>(ready)) __threadfence_system();
    __threadfence_system();
#endif
}

__global__ void refillGraphSlots(const uint4 *packed, uint4 *slots, const int *control, size_t words) {
    const int slot = blockIdx.y;
    if (!(unsigned(control[0]) & (1u << slot))) return;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x)
        slots[slot * words + i] = packed[slot * words + i];
}
// Host and device routes use the same activation and quantization, including
// BF16 rounding. The host branch can finish without waiting for resident GEMVs.
template<bool fp4>
__global__ void reconstructHybridRows(RoutedSiTU<float> src, float *out, const int *slots,
                                      bool host, int width, bool cache) {
    if ((slots[blockIdx.x] < 0) == host)
        reconstructRow<fp4>(src, out, width, cache, gridDim.y);
}

struct HybridGraphStorage {
    // Target replays are ordered and both expert branches join before the next
    // visit. Older shapes retain their pinned arena when a larger shape is warmed.
    struct Arena {
        uint8_t *host = nullptr, *mapped = nullptr;
        size_t capacity;
        std::vector<int> expertIds, cpuRoutes, uploadIds;
        std::vector<CpuExpertGroup> groups;
        std::vector<float> input, output;
        std::atomic<unsigned> &ready() const { return *reinterpret_cast<std::atomic<unsigned> *>(host); }
        explicit Arena(size_t bytes) : capacity(bytes) {
            check(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
            static_assert(std::atomic<unsigned>::is_always_lock_free && sizeof(std::atomic<unsigned>) == sizeof(unsigned),
                          "mapped completion flag must be a native word");
            new (host) std::atomic<unsigned>{1};
            auto error = cudaHostGetDevicePointer(&mapped, host, 0);
            if (error != cudaSuccess) { cudaFreeHost(host); check(error); }
        }
        ~Arena() { cudaFreeHost(host); }
    };
    std::shared_ptr<Arena> largest;
    std::shared_ptr<cuda::HostCopyPool> copies;

    struct Visit {
        std::shared_ptr<Arena> arena;
        std::shared_ptr<cuda::HostCopyPool> copies;
        HybridExpertRecords &cache;
        HybridCacheCycle *cycle;
        const Projection &gate, &down;
        int layer, rows, top, routes;
        int *control;
        float *hostInput, *hostOutput;
        uint8_t *uploads;
        void projectHost(const Projection &weight, int divisor) const {
            const int count = arena->cpuRoutes.size();
            if (!count) return;
            auto &input = arena->input;
            auto &output = arena->output;
            input.resize(size_t(count) * weight.cols);
            output.resize(size_t(count) * weight.rows);
            for (int i = 0; i < count; ++i)
                std::copy_n(hostInput + size_t(arena->cpuRoutes[i] / divisor) * weight.cols,
                            weight.cols, input.data() + size_t(i) * weight.cols);
            hostProjection(weight, arena->groups, input, output);
            for (int i = 0; i < count; ++i)
                std::copy_n(output.data() + size_t(i) * weight.rows, weight.rows,
                            hostOutput + size_t(arena->cpuRoutes[i]) * weight.rows);
        }
        void plan(const int *expertIds, cuda::HybridMoeGraph::Visit *sample) const {
            int *slots = control + 1;
            arena->groups.clear();
            arena->cpuRoutes.clear();
            // Original ownership for the reduction, then one masked slot table
            // per GPU branch. A masked projection never writes its route.
            std::fill_n(slots, 3 * routes, -1);
            arena->uploadIds.clear();
            control[0] = 0;
            arena->expertIds.assign(expertIds, expertIds + routes);
            auto *policy = cycle ? &cycle->policy : nullptr;
            const int experts = gate.host.size(), base = layer * experts;
            HybridMoeRoutes plan(arena->expertIds, experts, rows, base, false, policy, cache.timing,
                                 cache.count - cache.resident, sample && sample->probeUpload());
            if (sample) sample->routes(plan);
            for (int e : plan.hits)
                for (int r : plan.grouped[e]) slots[routes + r] = slots[r] = policy->Slot(base + e) - layer * cache.resident;
            for (size_t i = 0; i < plan.misses.size(); ++i) {
                const int expert = plan.misses[i];
                const auto &group = plan.grouped[expert];
                if (int(i) < plan.staged) {
                    if (cache.stagedKeys[i] != expert) {
                        arena->uploadIds.resize(i + 1, -1);
                        arena->uploadIds[i] = expert;
                        ++sample->uploads;
                        control[0] |= 1u << i;
                        cache.stagedKeys[i] = expert;
                    }
                    for (int r : group) slots[2 * routes + r] = slots[r] = cache.resident + i;
                } else {
                    arena->groups.push_back({expert, int(arena->cpuRoutes.size()), int(group.size())});
                    arena->cpuRoutes.insert(arena->cpuRoutes.end(), group.begin(), group.end());
                }
            }
            if (control[0]) {
                arena->ready().store(0, std::memory_order_relaxed);
                copies->Submit([](const void *argument) noexcept {
                    const auto &v = *static_cast<const Visit *>(argument);
                    for (size_t i = 0; i < v.arena->uploadIds.size(); ++i) {
                        const int expert = v.arena->uploadIds[i];
                        if (expert >= 0) v.cache.copyRecord(expert, v.uploads + i * v.cache.stride, v.gate, v.down);
                    }
                    v.arena->ready().store(1, std::memory_order_release);
                }, this);
            }
            ++cache.timing.calls;
        }
    };

    void run(DecodeGraphScratch &scratch, HybridExpertRecords &cache, HybridCacheCycle *cycle, int layer,
             const Config &c, const Projection &gate, const Projection &down, const Matrix &input,
             const Buffer<int> &ids, const Matrix &scores, float *result, int smCount) {
        const int rows = input.rows, top = c.topK, routes = rows * top, h = input.cols, inter = down.cols;
        const int staging = std::min(routes, cache.count - cache.resident);
        constexpr size_t header = 64;
        const size_t metadata = header + ((sizeof(int) * (4 * size_t(routes) + 1) + 15) & ~size_t(15));
        const size_t inBytes = sizeof(float) * std::max(size_t(rows) * h, size_t(routes) * inter);
        const size_t outBytes = sizeof(float) * routes * std::max(2 * inter, h);
        const size_t uploadOffset = (metadata + inBytes + outBytes + 15) & ~size_t(15);
        const size_t bytes = uploadOffset + staging * cache.stride;
        if (!largest || bytes > largest->capacity) {
            if (scratch.recording()) throw std::logic_error("Inkling host expert workspace was not warmed");
            largest = std::make_shared<Arena>(bytes);
        }
        if (staging && !copies) copies = cuda::GetHostCopyPool(FastllmCudaGetDevice());
        int *control = reinterpret_cast<int *>(largest->host + header);
        auto *hostIn = reinterpret_cast<float *>(largest->host + metadata);
        auto *hostOut = reinterpret_cast<float *>(largest->host + metadata + inBytes);
        Visit v{largest, copies, cache, cycle, gate, down, layer, rows, top, routes,
                control, hostIn, hostOut, largest->host + uploadOffset};
        Buffer<int> devicePlan(3 * routes + 1);
        Matrix pair(routes, 2 * inter), mid(routes, inter), parts(routes, h);
        auto *slots = devicePlan.p + 1;
        if (cache.count && !scratch.hybrid) scratch.hybrid = std::make_unique<cuda::HybridMoeGraph>();
        if (!scratch.recording()) {
            check(cudaMemsetAsync(result, 0, size_t(rows) * h * sizeof(float), cudaStreamPerThread));
            return;
        }
        auto *sample = cache.count ? &scratch.hybrid->add(cache.timing) : nullptr;
        int *hostIds = control + 3 * routes + 1;
        check(cudaMemcpyAsync(hostIds, ids.p, routes * sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread));
        check(cudaMemcpyAsync(hostIn, input.p, size_t(rows) * h * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
        scratch.hostNode([v, hostIds, sample] {
            try {
                v.plan(hostIds, sample);
            } catch (...) {
                v.control[0] = 0;
                std::fill_n(v.control + 1, 3 * v.routes, -1);
                std::fill(v.cache.stagedKeys.begin(), v.cache.stagedKeys.end(), -1);
                v.arena->cpuRoutes.clear();
                v.arena->uploadIds.clear();
                throw;
            }
        });
        check(cudaMemcpyAsync(devicePlan.p, control, (3 * routes + 1) * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
        auto *mappedIn = reinterpret_cast<float *>(largest->mapped + metadata);
        auto *mappedOut = reinterpret_cast<float *>(largest->mapped + metadata + inBytes);
        auto activate = [&](float *source, float *out, const int *mask, bool host, cudaStream_t stream) {
            RoutedSiTU<float> src{source, scores.p, c.gateCap, c.upCap};
            if (gate.fp4) {
                const size_t bytes = size_t(inter) * sizeof(float);
                const bool shared = bytes + 16 * sizeof(float) <= 48 * 1024;
                reconstructHybridRows<true><<<dim3(routes, quantTiles<true>(routes, inter, smCount)), 512,
                    shared ? bytes : 0, stream>>>(src, out, mask, host, inter, shared);
            } else {
                reconstructHybridRows<false><<<dim3(routes, quantTiles<false>(routes, inter, smCount)), 256,
                    0, stream>>>(src, out, mask, host, inter, false);
            }
        };
        auto host = [&](cudaStream_t stream) {
            scratch.hostNode([v, sample] {
                if (sample) sample->beginHost();
                v.projectHost(v.gate, v.top);
            }, stream);
            activate(mappedOut, mappedIn, slots, true, stream);
            scratch.hostNode([v, sample] {
                v.projectHost(v.down, 1);
                if (sample) sample->finishHost(v.rows, v.arena->cpuRoutes.size());
            }, stream);
        };
        if (sample) {
            scratch.hybrid->parallel(cudaStreamPerThread, *sample, host, [&] { cache.publication.wait(cudaStreamPerThread, true); }, [&](cudaStream_t stream) {
                if (staging) {
                    waitForExpertSource<<<1, 1, 0, stream>>>(reinterpret_cast<const unsigned *>(largest->mapped), devicePlan.p);
                    const size_t words = cache.stride / sizeof(uint4);
                    refillGraphSlots<<<dim3(std::min<size_t>((words + 255) / 256, 256), staging), 256, 0, stream>>>(
                        reinterpret_cast<const uint4 *>(largest->mapped + uploadOffset),
                        reinterpret_cast<uint4 *>(cache.data.p + cache.resident * cache.stride), devicePlan.p, words);
                }
            }, [&](cudaStream_t stream, bool staged) {
                const int *mask = slots + (staged ? 2 : 1) * routes;
                CudaMoeGraphSlotGate(cache.view, input.p, mask, pair.p, rows, top, h, inter, stream);
                activate(pair.p, mid.p, mask, false, stream);
                CudaMoeGraphSlotDown(cache.view, mid.p, mask, parts.p, routes, h, inter, stream);
            });
        } else {
            host(cudaStreamPerThread);
        }
        reduceExpertRows<<<dim3((h + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
            parts.p, nullptr, result, top, h, slots, mappedOut);
        check(cudaGetLastError());
    }
};
