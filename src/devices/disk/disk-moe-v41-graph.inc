// A graph leases resident records for its PP stage or TP cache owners. CPU
// misses may read/compute beside GPU hits, but submit CUDA only after retirement.
#include "devices/cuda/fastllm-cuda-disk-v41-graph.h"
#ifdef USE_CUDA
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif
namespace fastllm {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
// Reuse disk's compact CPU kernel, including BF16 and block-32 FP8 boundaries.
static void DiskGraphNativeCpu(const std::vector<float> &input, std::vector<float> &output,
                               const std::vector<Data *> &table, const std::vector<int32_t> &indices,
                               const std::vector<float> &scores, int hidden, bool singleRow, float limit) {
    const int rows = indices.size();
    Data x(BFLOAT16, {rows, hidden}), y(BFLOAT16, {rows, hidden}), ids(INT32PARAM, {rows, 1}),
        scale(FLOAT32, {rows, 1}), a, b, c;
    x.Allocate(false);
    ids.Allocate(false);
    scale.Allocate(false);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < hidden; ++col)
            reinterpret_cast<uint16_t *>(x.cpuData)[size_t(row) * hidden + col] =
                Float32ToBFloat16RNEBits(input[size_t(singleRow ? 0 : row) * hidden + col]);
    memcpy(ids.cpuData, indices.data(), rows * sizeof(int32_t));
    memcpy(scale.cpuData, scores.data(), rows * sizeof(float));
    std::vector<Data *> biases(table.size(), nullptr);
    DataDict data = {{"input", &x},
                     {"output", &y},
                     {"index", &ids},
                     {"score", &scale},
                     {"weights", (Data *)table.data()},
                     {"biass", (Data *)biases.data()},
                     {"w1", &a},
                     {"w2", &b},
                     {"w3", &c}};
    IntDict params = {{"weights___batch", int(table.size())},
                      {"biass___batch", int(table.size())},
                      {"deepSeekV4Mode", 1},
                      {"activationQuantBlock", 32},
                      {"diskMoeDecode", 1}};
    struct Cpu : CpuMergeMOE {
        using CpuMergeMOE::Run;
    };
    Cpu op;
    op.Run("MergeMOE", data, {{"swigluLimit", limit}, {"sharedScale", 0.f}}, params);
    for (size_t i = 0; i < output.size(); ++i)
        output[i] = BFloat16BitsToFloat32(reinterpret_cast<uint16_t *>(y.cpuData)[i]);
}
static bool DiskGraphNativePair(const Data &gate, const Data &down) {
    return gate.dataType == NVFP4 && down.dataType == NVFP4 && gate.scales.empty() && down.scales.empty() &&
           gate.blockK == 1 && down.blockK == 1 && gate.blockM == 32 && down.blockM == 32 &&
           gate.dims.size() == 2 && down.dims.size() == 2 && gate.dims[0] == 2 * down.dims[1] &&
           gate.dims[1] == down.dims[0] && gate.dims[1] > 0 && gate.dims[1] <= 24576 && down.dims[1] > 0 &&
           down.dims[1] <= 24576 && gate.dims[1] % 32 == 0 && down.dims[1] % 32 == 0;
}
struct DiskMoeV41Graph::Impl {
    struct Layer {
        std::vector<Data *> weights;
        std::unique_ptr<FastllmCudaDiskV41Graph> gpu;
        int rows, hidden, topk, device;
        float limit;
        const Data *candidate = nullptr;
    };
    std::vector<Layer> layers;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable wake, changed;
    enum class Preparation { Pending, Ready, Failed };
    Preparation preparation = Preparation::Pending;
    bool stopping = false, pending = false, finished = false, retired = false;
    std::atomic<bool> cancel{false};
    std::exception_ptr error;
    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            wake.notify_all();
        }
        if (worker.joinable())
            worker.join();
    }
    void Compute(Layer &l, DiskExpertCache &cache) {
        auto request = l.gpu->Request();
        auto &steps = cache.entries.at(l.weights[2]).canonicalDecodeSteps;
        steps = l.rows == 1 ? std::min(32u, steps + 1) : 0;
        struct Work {
            int expert;
            std::vector<int> routes;
            std::unique_ptr<DiskCachedExpert> loaded;
            std::future<std::unique_ptr<DiskCachedExpert>> read;
        };
        std::map<int, std::vector<int>> grouped;
        for (int r = 0; r < l.rows * l.topk; ++r) {
            AssertInFastLLM(request.ids[r] >= 0 && request.ids[r] < int(l.weights.size() / 2 - 1),
                            "Disk graph invalid route");
            grouped[request.ids[r]].push_back(r);
        }
        std::vector<Work> todo;
        for (auto &g : grouped) {
            auto &entry = cache.entries.at(l.weights[2 + 2 * g.first]);
            cache.Touch(entry, g.second.size(), l.rows);
            if (request.resident[g.second.front()] >= 0) {
                cache.stats.cudaHits += g.second.size();
                continue;
            }
            if (entry.cpu)
                cache.stats.cpuHits += g.second.size();
            else {
                cache.stats.misses += g.second.size();
                for (int i : {2 + 2 * g.first, 3 + 2 * g.first})
                    for (const auto &part : l.weights[i]->diskWeightParts)
                        cache.stats.diskBytes += part.bytes;
            }
            todo.push_back({g.first, std::move(g.second), {}, {}});
        }
        // Futures are joined even if a later allocation or computation throws.
        struct Drain {
            std::vector<Work> &items;
            ~Drain() {
                for (auto &x : items)
                    if (x.read.valid())
                        x.read.wait();
            }
        } drain{todo};
        while (!todo.empty()) {
            uint64_t inflight = 0;
            int reads = 0;
            for (auto &w : todo)
                if (w.read.valid()) {
                    inflight +=
                        l.weights[2 + 2 * w.expert]->GetBytes() + l.weights[3 + 2 * w.expert]->GetBytes();
                    ++reads;
                }
            for (auto &w : todo) {
                auto &entry = cache.entries.at(l.weights[2 + 2 * w.expert]);
                const uint64_t bytes =
                    l.weights[2 + 2 * w.expert]->GetBytes() + l.weights[3 + 2 * w.expert]->GetBytes();
                if (entry.cpu || w.read.valid() || w.loaded)
                    continue;
                if (reads >= 8 || (reads && inflight + bytes > (128ULL << 20)))
                    break;
                w.read =
                    cache.prefetch.Read(l.weights[2 + 2 * w.expert], l.weights[3 + 2 * w.expert], false, 2);
                inflight += bytes;
                ++reads;
            }
            std::vector<Data *> table{nullptr, nullptr};
            std::vector<int32_t> ids;
            std::vector<float> scores, input;
            std::vector<int> destinations, selected;
            for (size_t i = 0; i < todo.size(); ++i) {
                auto &w = todo[i];
                auto &entry = cache.entries.at(l.weights[2 + 2 * w.expert]);
                auto *host = entry.cpu.get();
                if (!host) {
                    if (!w.loaded && w.read.valid() &&
                        w.read.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                        w.loaded = w.read.get();
                    host = w.loaded.get();
                }
                if (!host)
                    continue;
                const int local = table.size() / 2 - 1;
                table.push_back(host->gate.get());
                table.push_back(host->down.get());
                selected.push_back(i);
                for (int r : w.routes) {
                    ids.push_back(local);
                    scores.push_back(request.scores[r]);
                    destinations.push_back(r);
                    const float *x = request.input + size_t(r / l.topk) * l.hidden;
                    if (l.rows > 1 || input.empty())
                        input.insert(input.end(), x, x + l.hidden);
                }
            }
            if (selected.empty()) {
                std::this_thread::sleep_for(std::chrono::microseconds(20));
                continue;
            }
            std::vector<float> output(size_t(ids.size()) * l.hidden);
            const bool native = l.weights[2]->dataType == NVFP4;
            if (native)
                DiskGraphNativeCpu(input, output, table, ids, scores, l.hidden, l.rows == 1, l.limit);
            const bool ok =
                native ||
                (l.rows == 1
                     ? HostMoeCanonicalV41Decode(input.data(), output.data(), table.data(), table.size(),
                                                 ids.data(), scores.data(), ids.size(), l.limit)
                     : HostMoeCanonicalV41Batch(input.data(), output.data(), ids.size(), table.data(),
                                                table.size(), ids.data(), scores.data(), l.limit));
            AssertInFastLLM(ok, "Disk graph canonical V4.1 CPU computation rejected");
            for (size_t i = 0; i < destinations.size(); ++i)
                memcpy(request.output + size_t(destinations[i]) * l.hidden, output.data() + i * l.hidden,
                       l.hidden * 4);
            for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
                auto &w = todo[*it];
                auto &entry = cache.entries.at(l.weights[2 + 2 * w.expert]);
                if (w.loaded && cache.MakeRoom(&entry, -1, w.loaded->Bytes(), GetMoeCpuCacheBytes()))
                    cache.AdmitCpu(entry, std::move(w.loaded));
                if (entry.cpu && entry.visits >= 2 &&
                    (!l.candidate ||
                     entry.Heat(cache.Epoch()) > cache.entries.at(l.candidate).Heat(cache.Epoch())))
                    l.candidate = l.weights[2 + 2 * w.expert];
                if (w.loaded)
                    cache.retiredCpuBytes += w.loaded->Bytes() - w.loaded->RecycleCpuStorage();
                todo.erase(todo.begin() + *it);
            }
        }
    }
    void Run() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || pending; });
                if (stopping)
                    return;
            }
            auto &cache = GetDiskExpertCache();
            // Keep record addresses leased through error handling too: an
            // already-launched graph may still be reading resident weights.
            std::unique_lock<std::mutex> lease;
            try {
                lease = cache.Lock();
                for (auto &l : layers) {
                    const int device = l.device;
                    FastllmCudaSetDevice(device);
                    l.candidate = nullptr;
                    auto request = l.gpu->Request();
                    for (size_t e = 0; e < l.weights.size() / 2 - 1; ++e) {
                        auto *gpu = cache.entries[l.weights[2 + 2 * e]].Cuda(device);
                        if (gpu && l.weights[2]->dataType == NVFP4) {
                            // Prefill may have repacked a resident record. Restore
                            // it before leasing addresses, never while GPU waits.
                            FastllmCudaSetDiskMxfp4MarlinLayout(*gpu->gate, false);
                            FastllmCudaSetDiskMxfp4MarlinLayout(*gpu->down, false);
                        }
                        request.records[e] = gpu ? static_cast<const uint8_t *>(gpu->cudaRecord) : nullptr;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    preparation = Preparation::Ready;
                    changed.notify_all();
                }
                for (auto &l : layers) {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                    while (!l.gpu->Published() && !cancel.load(std::memory_order_acquire)) {
                        if (std::chrono::steady_clock::now() > deadline)
                            throw std::runtime_error("Disk graph publication timed out");
                        std::this_thread::yield();
                    }
                    try {
                        if (!error && !cancel)
                            Compute(l, cache);
                    } catch (...) {
                        if (!error)
                            error = std::current_exception();
                    }
                    // A callback may fail after submitting CUDA work. Release
                    // every remaining wait node so the caller can drain it.
                    if (error || cancel)
                        std::fill_n(l.gpu->Request().output, size_t(l.rows) * l.topk * l.hidden, 0.f);
                    l.gpu->Complete();
                }
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait(lock, [&] { return retired; });
                }
                // The caller has retired every graph reader. Cache promotion
                // may now issue CUDA and recycle records without stale pointers.
                if (!error && !cancel) {
                    for (auto &l : layers)
                        if (l.candidate) {
                            const int device = l.device;
                            FastllmCudaSetDevice(device);
                            auto &entry = cache.entries.at(l.candidate);
                            if (!entry.cpu || entry.Cuda(device))
                                continue;
                            const auto bytes = entry.cpu->gate->GetBytes() + entry.cpu->down->GetBytes();
                            if (!cache.CanonicalReplacementPays(entry, device, bytes) ||
                                !cache.MakeRoom(&entry, device, bytes, GetMoeCudaCacheBytes(), nullptr,
                                                false))
                                continue;
                            std::unique_ptr<DiskCachedExpert> recycled;
                            if (!cache.MakeRoom(&entry, device, bytes, GetMoeCudaCacheBytes(), &recycled))
                                continue;
                            const bool native = l.weights[2]->dataType == NVFP4;
                            auto candidate =
                                native ? DiskCudaExpert(*entry.cpu, MoeGateSwiglu) : cache.TakeCpu(entry);
                            AssertInFastLLM(bool(candidate), "Disk graph cache conversion rejected");
                            bool uploaded = false;
                            cache.retiredCpuBytes +=
                                UploadDiskExpert(*candidate, device, uploaded, std::move(recycled), &entry);
                            FastllmCudaSyncCurrentThreadStream();
                            if (uploaded || !native)
                                cache.FinishPromotion(entry, candidate, device, uploaded);
                        }
                }
                cache.ReclaimCpu();
            } catch (...) {
                if (!error)
                    error = std::current_exception();
                for (auto &l : layers) {
                    std::fill_n(l.gpu->Request().output, size_t(l.rows) * l.topk * l.hidden, 0.f);
                    l.gpu->Complete();
                }
                std::unique_lock<std::mutex> lock(mutex);
                if (preparation == Preparation::Pending)
                    preparation = Preparation::Failed;
                changed.notify_all();
                wake.wait(lock, [&] { return retired; });
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                finished = true;
                pending = false;
                changed.notify_all();
            }
        }
    }
};
DiskMoeV41Graph::DiskMoeV41Graph() : impl(new Impl) {}
DiskMoeV41Graph::~DiskMoeV41Graph() = default;
bool DiskMoeV41Graph::Supports(const std::vector<Data *> &w) {
    if (w.size() < 4 || w.size() % 2 || !w[2] || !w[3])
        return false;
    for (size_t i = 2; i < w.size(); i += 2)
        if (!w[i] || !w[i + 1] || !w[i]->isDiskWeight || !w[i + 1]->isDiskWeight ||
            (!DiskV41GGUFPair(*w[i], *w[i + 1]) && !DiskGraphNativePair(*w[i], *w[i + 1])) ||
            w[i]->dataType != w[2]->dataType || w[i + 1]->dataType != w[3]->dataType ||
            w[i]->ggmlType != w[2]->ggmlType || w[i + 1]->ggmlType != w[3]->ggmlType ||
            w[i]->dims != w[2]->dims || w[i + 1]->dims != w[3]->dims)
            return false;
    return true;
}
void *DiskMoeV41Graph::Add(const Data &input, const Data &index, const Data &score, Data &output,
                           const std::vector<Data *> &weights, float limit) {
    if (!Supports(weights) || weights[0] || weights[1] || impl->worker.joinable())
        return nullptr;
    auto gpu = FastllmCudaDiskV41Graph::Create(
        input, index, score, output, weights[2]->dataType, weights[2]->ggmlType, weights[3]->ggmlType,
        weights[3]->dims[1], weights.size() / 2 - 1, weights[2]->GetBytes(),
        weights[2]->GetBytes() + weights[3]->GetBytes(), limit);
    if (!gpu)
        return nullptr;
    void *graph = gpu->Graph();
    impl->layers.push_back({weights, std::move(gpu), input.dims[0], input.dims[1], index.dims[1], FastllmCudaGetDevice(), limit});
    return graph;
}
bool DiskMoeV41Graph::Launch(const std::function<bool()> &launch) {
    auto &s = *impl;
    if (s.layers.empty())
        return launch();
    if (!s.worker.joinable())
        s.worker = std::thread([&s] { s.Run(); });
    bool prepared = false;
    {
        std::unique_lock<std::mutex> lock(s.mutex);
        AssertInFastLLM(!s.pending, "Disk graph already in flight");
        for (auto &l : s.layers)
            l.gpu->Reset();
        s.error = nullptr;
        s.cancel = false;
        s.preparation = Impl::Preparation::Pending;
        s.finished = s.retired = false;
        s.pending = true;
        s.wake.notify_one();
        s.changed.wait(lock, [&] { return s.preparation != Impl::Preparation::Pending; });
        prepared = s.preparation == Impl::Preparation::Ready;
    }
    bool ok = false;
    std::exception_ptr failure;
    try {
        // A failed preparation has not leased a complete record table. Retire
        // the worker and propagate its exception without submitting CUDA.
        if (prepared)
            ok = launch();
    } catch (...) {
        failure = std::current_exception();
    }
    if (prepared) {
        s.cancel = !ok || bool(failure) || FastllmCudaGetThreadError();
        try {
            // Drain even a failed callback: it may already have launched the
            // graph. The worker keeps cache records leased until this returns.
            std::set<int> devices;
            for (const auto &layer : s.layers) devices.insert(layer.device);
            if (devices.size() == 1 && *devices.begin() == FastllmCudaGetDevice())
                FastllmCudaSyncCurrentThreadStream();
            else for (int device : devices) FastllmCudaSyncDevice(device);
        } catch (...) {
            if (!failure)
                failure = std::current_exception();
        }
    }
    {
        std::unique_lock<std::mutex> lock(s.mutex);
        s.cancel = !ok || bool(failure) || FastllmCudaGetThreadError();
        s.retired = true;
        s.wake.notify_all();
        s.changed.wait(lock, [&] { return s.finished; });
        if (!failure)
            failure = s.error;
    }
    if (failure)
        std::rethrow_exception(failure);
    return ok && !FastllmCudaGetThreadError();
}
#else
struct DiskMoeV41Graph::Impl {};
DiskMoeV41Graph::DiskMoeV41Graph() : impl(new Impl) {}
DiskMoeV41Graph::~DiskMoeV41Graph() = default;
bool DiskMoeV41Graph::Supports(const std::vector<Data *> &) {
    return false;
}
void *DiskMoeV41Graph::Add(const Data &, const Data &, const Data &, Data &, const std::vector<Data *> &,
                           float) {
    return nullptr;
}
bool DiskMoeV41Graph::Launch(const std::function<bool()> &launch) {
    return launch();
}
#endif
} // namespace fastllm
