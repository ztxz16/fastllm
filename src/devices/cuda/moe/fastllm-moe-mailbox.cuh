#pragma once
#include <deque>

namespace {
// A single routed layer is in flight on the GPU. Submission can run ahead:
// the next publication is ordered after the preceding layer's reduction.
// Flags occupy separate cache lines because GPU and CPU are both writers.
struct MailboxRoutes {
    alignas(64) unsigned published = 0;
    alignas(64) unsigned computed = 0;
    alignas(64) int ids[16], owners[16], resident[2][16], slots[2][16];
    float scores[16];
};

__global__ void PublishMoePartials(float *host, const float *device, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        host[i] = device[i];
    __threadfence_system();
}

__global__ void PublishMoeMailbox(MailboxRoutes *box, uint16_t *hostInput, const uint16_t *input, const int32_t *ids,
                                  const float *scores, const int32_t *keys0, const int32_t *keys1, int base,
                                  int experts, int hidden, int topk, int origin, unsigned sequence) {
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        hostInput[c] = input[c];
    if (threadIdx.x < topk) {
        const int r = threadIdx.x, e = ids[r];
        box->ids[r] = e;
        box->scores[r] = scores[r];
        const int a = e >= 0 && e < experts ? keys0[base + e] : -1;
        const int b = e >= 0 && e < experts ? keys1[base + e] : -1;
        const int owner = origin == 0 ? (a >= 0 ? 0 : b >= 0 ? 1 : -1) : (b >= 0 ? 1 : a >= 0 ? 0 : -1);
        box->owners[r] = owner;
        box->resident[0][r] = owner == 0 ? a : -1;
        box->resident[1][r] = owner == 1 ? b : -1;
    }
    // Every writing thread fences its own mapped stores before publication.
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) *reinterpret_cast<volatile unsigned *>(&box->published) = sequence;
}

// Read mapped control data once, then let every projection CTA use VRAM.
// Directly dereferencing the mailbox inside GEMV repeats PCIe reads for each
// output tile, overwhelming the benefit of asynchronous route publication.
__global__ void UploadMoeMailbox(const MailboxRoutes *host, MailboxRoutes *local, const uint16_t *hostInput,
                                 uint16_t *input, int hidden, int topk, int rank, int origin, unsigned long long *ages,
                                 unsigned long long *step, unsigned long long *hits, unsigned long long *misses) {
    if (input)
        for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < hidden; c += blockDim.x * gridDim.x)
            input[c] = hostInput[c];
    if (blockIdx.x) return;
    if (threadIdx.x < topk) {
        const int r = threadIdx.x;
        if (input) {
            local->ids[r] = host->ids[r];
            local->scores[r] = host->scores[r];
            local->resident[0][r] = host->resident[0][r];
            local->resident[1][r] = host->resident[1][r];
        } else {
            local->slots[rank][r] = host->slots[rank][r];
            local->owners[r] = host->owners[r];
        }
    }
    __syncthreads();
    if (!input || threadIdx.x) return;
    const auto age = ++*step;
    for (int r = 0; r < topk; ++r) {
        const int slot = local->resident[rank][r];
        if (slot >= 0) {
            ages[slot] = age;
            ++*hits;
        } else if (rank == origin && local->resident[0][r] < 0 && local->resident[1][r] < 0)
            ++*misses;
    }
}

struct MailboxMoe {
    struct Work {
        DeviceCache *cache = nullptr;
        DecodeOverlapWorkspace copy;
        FastllmCudaMoeExpertParallel::Rank graphs;
        fastllm::Data activation, scratch, partial, input, metadata;
        cudaEvent_t done = nullptr, published = nullptr;
        float *hostResult = nullptr, *mappedResult = nullptr;
        struct Timing {
            cudaEvent_t start = nullptr, resident = nullptr, stagedStart = nullptr, end = nullptr;
            int hits = 0, routes = 0;
            cudaEvent_t layerStart = nullptr, layerEnd = nullptr;
            int layerMisses = 0, layerStaged = 0;
            bool wholeLayer = false, measure = true;
            bool used = false;
            fastllm::MoeDecodeOverlapScheduler cost;
        };
        std::vector<Timing> timing;
        ~Work() {
            if (!cache) return;
            cudaSetDevice(cache->device);
            if (done) cudaEventSynchronize(done);
            for (auto &t : timing)
                for (auto e : {t.start, t.resident, t.stagedStart, t.end, t.layerStart, t.layerEnd})
                    if (e) cudaEventDestroy(e);
            if (done) cudaEventDestroy(done);
            if (published) cudaEventDestroy(published);
            cudaFreeHost(hostResult);
        }
    };
    struct Request {
        int table, layer, origin;
        // Async work outlives the caller's reusable per-layer pointer table.
        // Own its entries; the Data objects themselves remain model-owned.
        std::vector<fastllm::Data *> weights;
        unsigned sequence;
        bool measure;
    };
    OffloadGroup &group;
    std::array<Work, 2> work;
    MailboxRoutes *host = nullptr, *mapped = nullptr;
    uint16_t *hostInput = nullptr, *mappedInput = nullptr;
    float *hostOutput = nullptr, *mappedOutput = nullptr;
    cudaEvent_t previous = nullptr;
    int hidden = 0, topk = 0;
    unsigned sequence = 0;
    unsigned uploadsSubmitted = 0;
    std::thread thread;
    std::mutex mutex;
    std::condition_variable wake, complete;
    std::deque<Request> requests;
    std::exception_ptr error;
    bool busy = false, stopping = false;
    std::vector<fastllm::MoeDecodeOverlapScheduler> cpu;
    std::vector<float> input;
    std::vector<fastllm::MoeDecodeCompletionPolicy> completion;
    std::vector<uint64_t> submissions;

    explicit MailboxMoe(OffloadGroup &g) : group(g) {}
    void Finish() {
        std::unique_lock<std::mutex> lock(mutex);
        complete.wait(lock, [&] { return requests.empty() && !busy; });
        auto failure = std::exchange(error, nullptr);
        if (failure) std::rethrow_exception(failure);
    }
    void Observe() {
        Finish();
        int previousDevice;
        cudaGetDevice(&previousDevice);
        double stepUs = 0;
        for (auto &w : work) {
            cudaSetDevice(w.cache->device);
            checkCudaErrors("Mailbox completion", cudaEventSynchronize(w.done));
            for (size_t layer = 0; layer < w.timing.size(); ++layer) {
                auto &t = w.timing[layer];
                if (!t.used) continue;
                float ms = 0;
                if (t.measure && t.hits) {
                    checkCudaErrors("Mailbox resident timing", cudaEventElapsedTime(&ms, t.start, t.resident));
                    t.cost.residentExpert.Observe(ms * 1000 / t.hits);
                }
                if (t.measure && t.routes) {
                    checkCudaErrors("Mailbox staged timing", cudaEventElapsedTime(&ms, t.stagedStart, t.end));
                    t.cost.stagedExpert.Observe(ms * 1000 / t.routes);
                }
                if (t.wholeLayer) {
                    checkCudaErrors("Mailbox layer timing", cudaEventElapsedTime(&ms, t.layerStart, t.layerEnd));
                    completion[layer].Observe(t.layerMisses, t.layerStaged, ms * 1000);
                    stepUs += ms * 1000;
                    auto &cache = *w.cache;
                    if (!cache.frequencyRuntime) {
                        cache.frequencyRuntime = std::make_unique<fastllm::MoeFrequencyPolicy::RuntimeBudget>();
                        cache.frequencyRuntime->savedUsPerKey.resize(group.totalRecords);
                        cache.frequencyRuntime->horizon = std::max(32.f, fastllm::GetMoeCacheConfig().halfLife);
                    }
                    const double saved = std::max(0.0, cpu[layer].cpuExpert.us - t.cost.residentExpert.us);
                    std::fill_n(cache.frequencyRuntime->savedUsPerKey.begin() + layer * group.layout.experts,
                                group.layout.experts, saved);
                }
                t.used = false;
                t.wholeLayer = false;
            }
        }
        for (auto &w : work)
            w.cache->frequencyStepUs += stepUs;
        cudaSetDevice(previousDevice);
    }
    ~MailboxMoe() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            complete.wait(lock, [&] { return requests.empty() && !busy; });
            stopping = true;
            wake.notify_one();
        }
        if (thread.joinable()) thread.join();
        if (previous) cudaEventSynchronize(previous);
        cudaFreeHost(host);
        cudaFreeHost(hostInput);
        cudaFreeHost(hostOutput);
    }
    void Allocate(fastllm::Data &d, fastllm::DataType type, std::vector<int> dims, int device) {
        d.dataType = type;
        d.Resize(dims);
        d.ToDevice(fastllm::CUDA, {device}, false);
        d.Allocate(false);
    }
    bool Experts(int d, int table, bool staged) {
        auto &w = work[d];
        const auto &l = group.LayerLayout(table);
        auto *local = static_cast<MailboxRoutes *>(w.metadata.cudaData);
        FastllmCudaMoeGGUFCacheView view{staged ? w.copy.records : w.cache->records,
                                         staged ? local->slots[d] : local->resident[d],
                                         l.recordStride,
                                         l.downOffset,
                                         l.gateGgmlType,
                                         l.downGgmlType,
                                         l.hidden,
                                         l.inter,
                                         w.scratch.cudaData,
                                         size_t(w.scratch.GetBytes()),
                                         staged ? nullptr : w.cache->slotOffsets};
        if (staged) {
            view.q8InputPrepared = true;
            view.preserveMissing = true;
            const auto &source = group.ggufSources[table];
            // Match the upload layout, including IQ2 R4 records consumed
            // directly by the projection kernels instead of being restored.
            if (FastllmCudaV41GGUFNumaSupported(l.gateGgmlType, l.downGgmlType,
                    source.weights[0].type, source.weights[1].type)) {
                view.numaGateType = source.weights[0].type;
                view.numaDownType = source.weights[1].type;
            }
        }
        FastllmCudaMoeExpertParallel::Rank::ResidentGraphKey key{};
        key[0] = table;
        key[1] = staged;
        return w.graphs.RunResidentGraph(key, [&] {
            return FastllmCudaMoeV41CanonicalGGUFCacheCompute(w.input, w.activation, view, local->scores, topk,
                                                              l.swigluLimit, static_cast<float *>(w.partial.cudaData));
        });
    }
    bool Prepare(const std::vector<DeviceCache *> &caches, int k) {
        hidden = group.layout.hidden;
        topk = k;
        size_t stride = 0, scratch = 0;
        int inter = 0;
        for (const auto &l : group.layerLayouts) {
            if (l.hidden != hidden || !FastllmCudaV41GGUFSupported(l.gateGgmlType, l.downGgmlType)) return false;
            stride = std::max(stride, l.recordStride);
            inter = std::max(inter, l.inter);
            scratch = std::max(scratch, FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(hidden, l.inter));
        }
        if (!scratch || !stride) return false;
        auto pin = [](auto **h, auto **d, size_t bytes) {
            return cudaHostAlloc(h, bytes, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                   cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
        };
        if (!pin(&host, &mapped, sizeof(MailboxRoutes)) || !pin(&hostInput, &mappedInput, hidden * sizeof(uint16_t)) ||
            !pin(&hostOutput, &mappedOutput, topk * hidden * sizeof(float)))
            return false;
        new (host) MailboxRoutes();
        cpu.resize(group.tableKeys.size());
        completion.resize(group.tableKeys.size());
        input.resize(hidden);
        submissions.resize(group.tableKeys.size());
        for (int d = 0; d < 2; ++d) {
            cudaDeviceProp properties;
            if (cudaGetDeviceProperties(&properties, caches[d]->device) != cudaSuccess ||
                properties.major * 10 + properties.minor < 75 || !properties.unifiedAddressing ||
                !properties.canMapHostMemory)
                return false;
            auto &w = work[d];
            w.cache = caches[d];
            cudaSetDevice(w.cache->device);
            w.graphs.cudaDevice = w.cache->device;
            if (!w.copy.Init(stride, group.tableKeys.size(), topk, true)) return false;
            Allocate(w.input, fastllm::BFLOAT16, {1, hidden}, w.cache->device);
            Allocate(w.metadata, fastllm::INT8, {int(sizeof(MailboxRoutes))}, w.cache->device);
            Allocate(w.activation, fastllm::BFLOAT16, {topk, inter}, w.cache->device);
            Allocate(w.scratch, fastllm::INT8, {int(scratch)}, w.cache->device);
            Allocate(w.partial, fastllm::FLOAT32, {topk, hidden}, w.cache->device);
            if (!pin(&w.hostResult, &w.mappedResult, size_t(topk) * hidden * sizeof(float))) return false;
            if (cudaEventCreateWithFlags(&w.done, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&w.published, cudaEventDisableTiming) != cudaSuccess)
                return false;
            w.timing.resize(group.tableKeys.size());
            for (auto &t : w.timing)
                for (auto *e : {&t.start, &t.resident, &t.stagedStart, &t.end, &t.layerStart, &t.layerEnd})
                    if (cudaEventCreate(e) != cudaSuccess) return false;
        }
        // CUDA may synchronize a context while lazily loading a kernel.
        // Do that before any GPU waits for this coordinator: a first-use
        // restore/projection behind a mailbox waiter can otherwise deadlock.
        MailboxRoutes empty{};
        for (auto &slots : empty.resident)
            std::fill_n(slots, 16, -1);
        for (auto &slots : empty.slots)
            std::fill_n(slots, 16, -1);
        for (auto &w : work) {
            const int d = &w - &work[0];
            cudaSetDevice(w.cache->device);
            auto *local = static_cast<MailboxRoutes *>(w.metadata.cudaData);
            checkCudaErrors("Mailbox warm metadata", cudaMemcpy(local, &empty, sizeof(empty), cudaMemcpyHostToDevice));
            checkCudaErrors("Mailbox warm input", cudaMemset(w.input.cudaData, 0, hidden * sizeof(uint16_t)));
            for (size_t table = 0; table < group.tableKeys.size(); ++table) {
                w.copy.ConfigureLayout(group, table, 1);
                w.copy.CopyExpert(group, table, 0, 0, 1);
                for (int staged = 0; staged < 2; ++staged) {
                    // Warm and instantiate stable leaves while no mailbox is active.
                    for (int replay = 0; replay < 2; ++replay)
                        if (!Experts(d, table, staged)) return false;
                }
            }
            PublishMoePartials<<<(topk * hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                w.mappedResult, static_cast<const float *>(w.partial.cudaData), topk * hidden);
            checkCudaErrors("Mailbox warm projections", cudaStreamSynchronize(cudaStreamPerThread));
            checkCudaErrors("Mailbox warm transfers", cudaStreamSynchronize(w.copy.copyStream));
        }
        try {
            thread = std::thread([this] {
                std::unique_lock<std::mutex> lock(mutex);
                while (true) {
                    wake.wait(lock, [&] { return stopping || !requests.empty(); });
                    if (stopping) return;
                    auto request = std::move(requests.front());
                    requests.pop_front();
                    busy = true;
                    lock.unlock();
                    try {
                        Run(request);
                    } catch (...) {
                        std::lock_guard<std::mutex> guard(mutex);
                        error = std::current_exception();
                        // Release already queued GPU waiters before propagating.
                        for (auto &slots : host->slots)
                            std::fill_n(slots, topk, -1);
                        std::fill_n(hostOutput, topk * hidden, 0.f);
                        __atomic_store_n(&host->computed, request.sequence, __ATOMIC_RELEASE);
                        uploadsSubmitted = request.sequence;
                        complete.notify_all();
                    }
                    lock.lock();
                    busy = false;
                    complete.notify_all();
                }
            });
        } catch (const std::system_error &) {
            return false;
        }
        return true;
    }
    void Run(Request &q) {
        const auto &layout = group.LayerLayout(q.table);
        const auto start = std::chrono::steady_clock::now();
        unsigned polls = 0;
        while (__atomic_load_n(&host->published, __ATOMIC_ACQUIRE) != q.sequence) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
            if (++polls % 65536 == 0) {
                fastllm::AssertInFastLLM(std::chrono::steady_clock::now() - start < std::chrono::seconds(60),
                                         "MoE route mailbox timed out");
                std::this_thread::yield();
            }
        }
        std::vector<int> misses, reuse;
        int hits[2] = {};
        for (int r = 0; r < topk; ++r) {
            fastllm::AssertInFastLLM(host->ids[r] >= 0 && host->ids[r] < layout.experts, "Invalid MoE mailbox route");
            for (int d = 0; d < 2; ++d) {
                host->slots[d][r] = -1;
                hits[d] += host->resident[d][r] >= 0;
            }
            if (host->owners[r] >= 0) continue;
            const int e = host->ids[r];
            auto at = std::find(misses.begin(), misses.end(), e);
            if (at == misses.end()) {
                misses.push_back(e);
                reuse.push_back(1);
            } else
                ++reuse[at - misses.begin()];
        }
        using Scheduler = fastllm::MoeDecodeOverlapScheduler;
        std::vector<Scheduler::SharedRankPlan> plans;
        for (int d = 0; d < 2; ++d)
            plans.push_back({&work[d].timing[q.table].cost, hits[d], topk, 0});
        auto &cost = cpu[q.table];
        const auto call = cost.calls++;
        auto owners = Scheduler::AssignSharedMisses(plans, cost.cpuExpert, reuse, call, &cost);
        int missRoutes = 0, offloaded = 0;
        for (size_t i = 0; i < reuse.size(); ++i) {
            missRoutes += reuse[i];
            if (owners[i] >= 0) offloaded += reuse[i];
        }
        if (!completion[q.table].Allow(missRoutes, offloaded, call)) {
            std::fill(owners.begin(), owners.end(), -1);
            offloaded = 0;
        }
        auto &whole = work[q.origin].timing[q.table];
        whole.layerMisses = missRoutes;
        whole.layerStaged = offloaded;
        whole.wholeLayer = true;
        std::vector<int> staged[2];
        for (size_t i = 0; i < misses.size(); ++i)
            if (owners[i] >= 0) {
                const int d = owners[i], slot = staged[d].size();
                staged[d].push_back(misses[i]);
                for (int r = 0; r < topk; ++r)
                    if (host->ids[r] == misses[i]) {
                        host->owners[r] = d;
                        host->slots[d][r] = slot;
                    }
            }
        const int cpuCount = std::count(host->owners, host->owners + topk, -1);
        for (int d = 0; d < 2; ++d) {
            auto &w = work[d];
            auto &t = w.timing[q.table];
            t.measure = q.measure;
            t.hits = hits[d];
            t.routes = std::count(host->owners, host->owners + topk, d) - hits[d];
            t.used = true;
            auto &stats = w.cache->cooperativeStats;
            const int c = d == q.origin ? cpuCount : 0;
            stats[0] += d == q.origin;
            stats[1] += hits[d] + t.routes + c;
            stats[2] += hits[d];
            stats[3] += t.routes + c;
            stats[4] += hits[d] + t.routes;
            stats[5] += c;
            stats[6] += hits[d];
        }
        auto &cache = *work[q.origin].cache;
        cache.frequency->Observe(q.table * layout.experts, host->ids, topk);
        for (int c = 0; c < hidden; ++c)
            input[c] = fastllm::BFloat16BitsToFloat32(hostInput[c]);
        auto upload = [&] {
            for (int d = 0; d < 2; ++d)
                if (!staged[d].empty()) {
                    auto &w = work[d];
                    cudaSetDevice(w.cache->device);
                    w.copy.ConfigureLayout(group, q.table, 1);
                    // Publication follows the previous layer's gather: staging
                    // buffers are no longer read by either GPU.
                    for (size_t i = 0; i < staged[d].size(); ++i)
                        w.copy.CopyExpert(group, q.table, staged[d][i], i, staged[d].size());
                }
            {
                std::lock_guard<std::mutex> lock(mutex);
                uploadsSubmitted = q.sequence;
                complete.notify_all();
            }
            for (int d = 0; d < 2; ++d)
                if (!staged[d].empty()) {
                    auto &w = work[d];
                    cudaSetDevice(w.cache->device);
                    checkCudaErrors("Mailbox staged payload", cudaEventSynchronize(w.copy.copyDone));
                    float ms = 0;
                    checkCudaErrors("Mailbox DMA timing", cudaEventElapsedTime(&ms, w.copy.copyStart, w.copy.copyDone));
                    w.timing[q.table].cost.copiedExpert.Observe(ms * 1000 / staged[d].size());
                }
        };
        if (cpuCount) {
            double elapsed = 0;
            NumasMoeDecodeExpertsWithOverlap(input.data(), hostOutput, q.weights.data(), host->ids, host->owners, topk,
                                             q.layer, upload, host->scores, layout.swigluLimit, 32, &elapsed);
            cost.ObserveDecodeCpu(cpuCount, elapsed);
        } else
            upload();
        __atomic_store_n(&host->computed, q.sequence, __ATOMIC_RELEASE);
    }
};

bool TryMailboxMoe(OffloadGroup &group, DeviceCache &origin, int table, const fastllm::Data &input,
                   const fastllm::Data &index, const fastllm::Data &score, fastllm::Data &output,
                   fastllm::Data **weights, int layer, const std::function<void()> &shared,
                   const std::vector<int> *devices) {
    using namespace fastllm;
    const auto &l = group.LayerLayout(table);
    if (!l.deepSeekV41 || l.weightType != DATA_GGUF_FORMAT || group.ggufSources.empty() || !group.cpuDecodeReady ||
        !group.expertDevices.empty() || !devices || devices->size() != 2 || (*devices)[0] == (*devices)[1] ||
        input.dims.size() != 2 || input.dims[0] != 1 || input.dims[1] != l.hidden || input.dataType != BFLOAT16 ||
        index.dims.size() != 2 || index.dims[0] != 1 || index.dims != score.dims || index.dims[1] < 1 ||
        index.dims[1] > 16 || index.dataType != INT32 || score.dataType != FLOAT32 || index.dataDevice != CUDA ||
        score.dataDevice != CUDA || !index.cudaData || !score.cudaData || !PackedCacheRows(index) ||
        !PackedCacheRows(score) || !origin.frequencyActive)
        return false;
    std::vector<DeviceCache *> caches;
    for (int d : *devices) {
        auto it = group.deviceCaches.find(d);
        if (it == group.deviceCaches.end() || !it->second || !it->second->ready || !it->second->mappedKeyToSlot ||
            !it->second->frequencyActive)
            return false;
        caches.push_back(it->second.get());
    }
    const int root = caches[0] == &origin ? 0 : caches[1] == &origin ? 1 : -1;
    if (root < 0) return false;
    struct Restore {
        int device;
        ~Restore() { cudaSetDevice(device); }
    } restore{origin.device};
    if (!group.mailboxHybrid) {
        auto state = std::make_shared<MailboxMoe>(group);
        if (!state->Prepare(caches, index.dims[1])) {
            (void)cudaGetLastError();
            return false;
        }
        group.mailboxHybrid = state;
    }
    auto &s = *group.mailboxHybrid;
    if (s.topk != index.dims[1] || s.work[0].cache != caches[0] || s.work[1].cache != caches[1]) return false;
    cudaSetDevice(origin.device);
    s.Allocate(output, BFLOAT16, {1, l.hidden}, origin.device);
    const unsigned seq = ++s.sequence;
    const uint64_t call = s.submissions[table]++;
    // Sample individual projections after calibration, including all sparse
    // alternate-path probes. Layer completion and dependencies stay measured
    // on every call so scheduling responds promptly to contention.
    const bool measure = call < 32 || call % 8 == 0 || call % 127 == 126 || call % 131 == 130;
    if (s.previous) checkCudaErrors("Mailbox previous layer", cudaStreamWaitEvent(cudaStreamPerThread, s.previous, 0));
    for (auto *cache : caches)
        if (cache->admissionDone)
            checkCudaErrors("Mailbox residency", cudaStreamWaitEvent(cudaStreamPerThread, cache->admissionDone, 0));
    checkCudaErrors("Mailbox layer start", cudaEventRecord(s.work[root].timing[table].layerStart, cudaStreamPerThread));
    PublishMoeMailbox<<<1, 256, 0, cudaStreamPerThread>>>(
        s.mapped, s.mappedInput, static_cast<const uint16_t *>(input.cudaData),
        static_cast<const int32_t *>(index.cudaData), static_cast<const float *>(score.cudaData),
        caches[0]->mappedKeyToSlot, caches[1]->mappedKeyToSlot, table * l.experts, l.experts, l.hidden, s.topk, root,
        seq);
    checkCudaErrors("Mailbox published", cudaEventRecord(s.work[root].published, cudaStreamPerThread));
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        const int weightCount = (l.experts + 1) * 2;
        s.requests.push_back({table, layer, root,
                              std::vector<fastllm::Data *>(weights, weights + weightCount),
                              seq, measure});
        s.wake.notify_one();
    }
    if (shared) shared();
    for (int offset = 0; offset < 2; ++offset) {
        const int d = (root + offset) % 2;
        auto &w = s.work[d];
        auto &t = w.timing[table];
        cudaSetDevice(w.cache->device);
        checkCudaErrors("Mailbox route dependency",
                        cudaStreamWaitEvent(cudaStreamPerThread, s.work[root].published, 0));
        auto *local = static_cast<MailboxRoutes *>(w.metadata.cudaData);
        UploadMoeMailbox<<<(l.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            s.mapped, local, d == root ? static_cast<const uint16_t *>(input.cudaData) : s.mappedInput,
            static_cast<uint16_t *>(w.input.cudaData), l.hidden, s.topk, d, root, w.cache->lastUsed, w.cache->step,
            w.cache->hitCount, w.cache->totalMissCount);
        if (measure) checkCudaErrors("Mailbox resident start", cudaEventRecord(t.start, cudaStreamPerThread));
        AssertInFastLLM(s.Experts(d, table, false), "Mailbox resident compute failed");
        if (measure) checkCudaErrors("Mailbox resident end", cudaEventRecord(t.resident, cudaStreamPerThread));
    }
    // Submit weight DMA before any dependent handoff or subsequent layer's
    // transfers. Resident GPU work already runs in parallel with route
    // planning and CPU experts; no upload/CPU completion is awaited here.
    {
        std::unique_lock<std::mutex> lock(s.mutex);
        s.complete.wait(lock, [&] { return s.uploadsSubmitted == seq; });
    }
    for (int offset = 0; offset < 2; ++offset) {
        const int d = (root + offset) % 2;
        auto &w = s.work[d];
        auto &t = w.timing[table];
        cudaSetDevice(w.cache->device);
        auto *local = static_cast<MailboxRoutes *>(w.metadata.cudaData);
        if (t.routes || d == root)
            UploadMoeMailbox<<<1, 32, 0, cudaStreamPerThread>>>(s.mapped, local, nullptr, nullptr, 0, s.topk, d, root,
                                                                nullptr, nullptr, nullptr, nullptr);
        if (t.routes) {
            checkCudaErrors("Mailbox staged payload ready",
                            cudaStreamWaitEvent(cudaStreamPerThread, w.copy.copyDone, 0));
            if (measure) checkCudaErrors("Mailbox staged start", cudaEventRecord(t.stagedStart, cudaStreamPerThread));
            AssertInFastLLM(s.Experts(d, table, true), "Mailbox staged compute failed");
            if (measure) checkCudaErrors("Mailbox staged end", cudaEventRecord(t.end, cudaStreamPerThread));
        }
        if (d != root && t.hits + t.routes)
            PublishMoePartials<<<(s.topk * l.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                w.mappedResult, static_cast<const float *>(w.partial.cudaData), s.topk * l.hidden);
        checkCudaErrors("Mailbox peer done", cudaEventRecord(w.done, cudaStreamPerThread));
    }
    cudaSetDevice(origin.device);
    auto &r = s.work[root];
    auto &peer = s.work[1 - root];
    auto *local = static_cast<MailboxRoutes *>(r.metadata.cudaData);
    const auto &peerTiming = peer.timing[table];
    checkCudaErrors("Mailbox peer ready", cudaStreamWaitEvent(cudaStreamPerThread, peer.done, 0));
    // Small results use mapped host memory. Queuing a copy-engine transfer
    // behind a GPU mailbox wait can block the later expert DMA that releases
    // that very wait, even when the transfers use nonblocking streams.
    if (peerTiming.hits + peerTiming.routes) {
        CopyOwnedMoeRoutes<<<(s.topk * l.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            static_cast<float *>(r.partial.cudaData), peer.mappedResult, local->owners, 1 - root, l.hidden, s.topk);
    }
    if (std::count(s.host->owners, s.host->owners + s.topk, -1))
        CooperativeMoeWaitCpu<<<1, 32, 0, cudaStreamPerThread>>>(&s.mapped->computed, seq);
    cuda::dsv41_cache::Reduce<<<(l.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        s.mappedOutput, static_cast<const float *>(r.partial.cudaData), local->owners, local->ids,
        static_cast<__nv_bfloat16 *>(output.cudaData), l.hidden, s.topk);
    checkCudaErrors("Mailbox reduction", cudaGetLastError());
    checkCudaErrors("Mailbox layer end", cudaEventRecord(r.timing[table].layerEnd, cudaStreamPerThread));
    checkCudaErrors("Mailbox layer done", cudaEventRecord(r.done, cudaStreamPerThread));
    s.previous = r.done;
    return true;
}
void WaitMailboxMoe(OffloadGroup &group) {
    if (group.mailboxHybrid) group.mailboxHybrid->Observe();
}
} // namespace
