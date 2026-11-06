#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <deque>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace fastllm { namespace cuda {

// One helper per device, shared by graph staging and all upload streams.
// Large copies use two CPUs; small copies stay inline. The helper sleeps
// between jobs and inherits the source owner's allowed NUMA/CPU placement.
class HostCopyPool {
    std::mutex submit_, mutex_;
    std::condition_variable work_, complete_;
    std::thread helper_;
    void *destination_ = nullptr;
    const void *source_ = nullptr;
    size_t bytes_ = 0;
    bool pending_ = false, stopping_ = false;
public:
    explicit HostCopyPool(const std::vector<int> &cpus) {
        unsigned available = std::thread::hardware_concurrency();
#if defined(__linux__)
        cpu_set_t mask;
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) available = CPU_COUNT(&mask);
#endif
        if (!cpus.empty()) available = cpus.size();
        if (available < 2) return;
        try {
            helper_ = std::thread([this, cpus] {
#if defined(__linux__)
                if (!cpus.empty()) {
                    cpu_set_t mask; CPU_ZERO(&mask);
                    for (int cpu : cpus) if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &mask);
                    pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
                }
#endif
                std::unique_lock<std::mutex> lock(mutex_);
                for (;;) {
                    work_.wait(lock, [&] { return pending_ || stopping_; });
                    if (stopping_) return;
                    void *dst = destination_; const void *src = source_; const size_t bytes = bytes_;
                    lock.unlock(); std::memcpy(dst, src, bytes); lock.lock();
                    pending_ = false; complete_.notify_one();
                }
            });
        } catch (const std::system_error &) { /* Resource limits retain inline memcpy. */ }
    }
    ~HostCopyPool() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        work_.notify_one();
        if (helper_.joinable()) helper_.join();
    }
    void Copy(void *destination, const void *source, size_t bytes) {
        if (!helper_.joinable() || bytes < (1ULL << 20)) { std::memcpy(destination, source, bytes); return; }
        // Concurrent admission and graph staging may share this helper. No
        // CUDA call or GPU wait is allowed while this CPU-only lock is held.
        std::lock_guard<std::mutex> submission(submit_);
        const size_t first = (bytes / 2) & ~size_t(63);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            destination_ = static_cast<unsigned char *>(destination) + first;
            source_ = static_cast<const unsigned char *>(source) + first;
            bytes_ = bytes - first; pending_ = true;
        }
        work_.notify_one();
        std::memcpy(destination, source, first);
        std::unique_lock<std::mutex> lock(mutex_);
        complete_.wait(lock, [&] { return !pending_; });
    }
};

inline std::shared_ptr<HostCopyPool> GetHostCopyPool(int device, const std::vector<int> &cpus = {}) {
    static std::mutex mutex;
    static std::map<int, std::weak_ptr<HostCopyPool>> pools;
    std::lock_guard<std::mutex> lock(mutex);
    auto pool = pools[device].lock();
    if (!pool) { pool = std::make_shared<HostCopyPool>(cpus); pools[device] = pool; }
    return pool;
}

// Prefill can create transient streams. Retain at most eight upload rings per
// submitting thread instead of retaining pinned storage for every past handle.
// Ring destruction waits for its last DMA before releasing any staging slot.
template<class Ring>
class HostUploadStreams {
public:
    struct Entry {
        std::unique_ptr<Ring> ring;
        uint64_t used = 0;
        bool attempted = false;
    };
private:
    std::map<std::pair<int, cudaStream_t>, Entry> streams_;
    uint64_t clock_ = 0;
public:
    Entry &Get(int device, cudaStream_t stream) {
        const auto key = std::make_pair(device, stream);
        auto found = streams_.find(key);
        if (found == streams_.end()) {
            if (streams_.size() == 8) {
                auto oldest = std::min_element(streams_.begin(), streams_.end(),
                    [](const auto &a, const auto &b) { return a.second.used < b.second.used; });
                streams_.erase(oldest);
            }
            found = streams_.emplace(key, Entry{}).first;
        }
        found->second.used = ++clock_;
        return found->second;
    }
};

// Pageable weights remain in their NUMA owner. Only a bounded window is
// pinned, and a slot cannot be overwritten until its DMA has consumed it.
// This adapter is for host-submitted transfers; graph replay uses mapped
// mailboxes and never calls a potentially synchronous pageable CUDA copy.
class HostUploadRing {
    static constexpr size_t chunkBytes = 4ULL << 20;
    struct Slot {
        void *host = nullptr;
        cudaEvent_t done = nullptr;
        bool pending = false, attempted = false;
    };
    Slot slots_[4];
    unsigned next_ = 0;
    std::shared_ptr<HostCopyPool> copies_;
public:
    explicit HostUploadRing(int device) : copies_(GetHostCopyPool(device)) {}
    ~HostUploadRing() {
        for (auto &s : slots_) {
            if (s.pending) cudaEventSynchronize(s.done);
            if (s.done) cudaEventDestroy(s.done);
            if (s.host) cudaFreeHost(s.host);
        }
    }
    cudaError_t Copy(void *destination, const void *source, size_t bytes, cudaStream_t stream) {
        auto *dst = static_cast<unsigned char *>(destination);
        auto *src = static_cast<const unsigned char *>(source);
        while (bytes) {
            auto &s = slots_[next_++ % 4];
            if (!s.attempted) {
                s.attempted = true;
                if (cudaHostAlloc(&s.host, chunkBytes, cudaHostAllocPortable) != cudaSuccess) {
                    cudaGetLastError(); s.host = nullptr;
                } else if (cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) != cudaSuccess) {
                    cudaGetLastError(); cudaFreeHost(s.host); s.host = nullptr;
                }
            }
            const size_t n = std::min(bytes, chunkBytes);
            cudaError_t status;
            if (s.host) {
                if (s.pending && (status = cudaEventSynchronize(s.done)) != cudaSuccess) return status;
                s.pending = false;
                copies_->Copy(s.host, src, n);
                status = cudaMemcpyAsync(dst, s.host, n, cudaMemcpyHostToDevice, stream);
                if (status != cudaSuccess) return status;
                status = cudaEventRecord(s.done, stream);
                if (status != cudaSuccess) { cudaStreamSynchronize(stream); return status; }
                s.pending = true;
            } else {
                // No pin budget remains. Keep the same cache/compute path;
                // wait only this upload stream before reusing the caller's data.
                status = cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream);
                if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
                if (status != cudaSuccess) return status;
            }
            src += n; dst += n; bytes -= n;
        }
        return cudaSuccess;
    }
};

inline cudaError_t UploadHostWeight(void *destination, const void *source, size_t bytes,
                                   bool pinned, cudaStream_t stream) {
    if (pinned) return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamCaptureStatus capture;
    auto status = cudaStreamIsCapturing(stream, &capture);
    if (status != cudaSuccess) return status;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    int device = 0;
    if ((status = cudaGetDevice(&device)) != cudaSuccess) return status;
    // Per-thread ownership also distinguishes cudaStreamPerThread. Rings on
    // different streams/devices never share an event or staging allocation.
    static thread_local HostUploadStreams<HostUploadRing> rings;
    auto &ring = rings.Get(device, stream).ring;
    if (!ring) ring.reset(new HostUploadRing(device));
    return ring->Copy(destination, source, bytes, stream);
}

// Immutable model weights may be borrowed until the upload stream completes.
// A CPU-only producer fills a bounded mapped ring while the submitter proceeds
// to the next graph. Device stream memory operations protect each reuse; the
// producer must never call CUDA or wait for one stream ahead of another.
struct HostUploadSignal {
    alignas(64) unsigned ready = 0;
    alignas(64) unsigned consumed = 0;
};

class HostUploadProducer {
public:
    struct Job {
        HostUploadSignal *signal;
        unsigned generation;
        void *destination;
        const void *source;
        size_t bytes;
    };
private:
    std::mutex mutex_;
    std::condition_variable work_, space_;
    std::deque<Job> jobs_;
    std::shared_ptr<HostCopyPool> copies_;
    bool stopping_ = false;
    std::thread worker_;
public:
    explicit HostUploadProducer(int device) : copies_(GetHostCopyPool(device)), worker_([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            work_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
            if (stopping_) return;
            auto job = std::find_if(jobs_.begin(), jobs_.end(), [](const Job &j) {
                return __atomic_load_n(&j.signal->consumed, __ATOMIC_ACQUIRE) == j.generation - 1;
            });
            if (job == jobs_.end()) {
                lock.unlock(); std::this_thread::yield(); lock.lock(); continue;
            }
            const Job copy = *job;
            jobs_.erase(job); space_.notify_one();
            lock.unlock();
            copies_->Copy(copy.destination, copy.source, copy.bytes);
            __atomic_store_n(&copy.signal->ready, copy.generation, __ATOMIC_RELEASE);
            lock.lock();
        }
    }) {}
    ~HostUploadProducer() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        work_.notify_one(); worker_.join();
    }
    void Submit(Job job) {
        std::unique_lock<std::mutex> lock(mutex_);
        space_.wait(lock, [&] { return jobs_.size() < 1024; });
        jobs_.push_back(job); work_.notify_one();
    }
};

inline std::shared_ptr<HostUploadProducer> GetHostUploadProducer(int device) {
    static std::mutex mutex;
    static std::map<int, std::weak_ptr<HostUploadProducer>> producers;
    std::lock_guard<std::mutex> lock(mutex);
    auto producer = producers[device].lock();
    if (!producer) { producer = std::make_shared<HostUploadProducer>(device); producers[device] = producer; }
    return producer;
}

class ImmutableHostUploadRing {
    static constexpr size_t chunkBytes = 4ULL << 20;
    struct Slot {
        HostUploadSignal *host = nullptr, *mapped = nullptr;
        cudaEvent_t done = nullptr;
        unsigned generation = 0;
    };
    Slot slots_[4];
    unsigned next_ = 0;
    decltype(&cuStreamWaitValue32) wait_ = nullptr;
    decltype(&cuStreamWriteValue32) write_ = nullptr;
    std::shared_ptr<HostUploadProducer> producer_;
public:
    bool Init(int device) {
        if (cudaGetDriverEntryPoint("cuStreamWaitValue32", reinterpret_cast<void **>(&wait_), cudaEnableDefault) != cudaSuccess ||
            cudaGetDriverEntryPoint("cuStreamWriteValue32", reinterpret_cast<void **>(&write_), cudaEnableDefault) != cudaSuccess ||
            !wait_ || !write_) { cudaGetLastError(); return false; }
        for (auto &s : slots_) {
            if (cudaHostAlloc(&s.host, sizeof(HostUploadSignal) + chunkBytes,
                    cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(&s.mapped, s.host, 0) != cudaSuccess ||
                cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError(); return false;
            }
            new (s.host) HostUploadSignal{};
        }
        cudaStream_t probe;
        if (cudaStreamCreateWithFlags(&probe, cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError(); return false;
        }
        const auto address = reinterpret_cast<CUdeviceptr>(&slots_[0].mapped->consumed);
        const bool supported = write_(reinterpret_cast<CUstream>(probe), address, 0, 0) == CUDA_SUCCESS &&
            wait_(reinterpret_cast<CUstream>(probe), address, 0, CU_STREAM_WAIT_VALUE_EQ) == CUDA_SUCCESS;
        const auto completed = cudaStreamSynchronize(probe);
        cudaStreamDestroy(probe);
        if (!supported || completed != cudaSuccess) { cudaGetLastError(); return false; }
        try { producer_ = GetHostUploadProducer(device); }
        catch (const std::system_error &) { return false; }
        return true;
    }
    ~ImmutableHostUploadRing() {
        for (auto &s : slots_) {
            // Event state remains valid when the generation counter wraps.
            if (s.done) { cudaEventSynchronize(s.done); cudaEventDestroy(s.done); }
            if (s.host) cudaFreeHost(s.host);
        }
    }
    cudaError_t Copy(void *destination, const void *source, size_t bytes, cudaStream_t stream) {
        auto *dst = static_cast<unsigned char *>(destination);
        auto *src = static_cast<const unsigned char *>(source);
        while (bytes) {
            auto &s = slots_[next_++ % 4];
            const size_t n = std::min(bytes, chunkBytes);
            const unsigned generation = s.generation + 1;
            if (wait_(reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(&s.mapped->ready),
                    generation, CU_STREAM_WAIT_VALUE_EQ) != CUDA_SUCCESS) return cudaErrorUnknown;
            s.generation = generation;
            producer_->Submit({s.host, generation, s.host + 1, src, n});
            auto status = cudaMemcpyAsync(dst, s.host + 1, n, cudaMemcpyHostToDevice, stream);
            // A rejected copy still owns a producer job and a queued wait.
            // Retire both before reusing/freeing this slot, even on failure.
            const auto consumed = write_(reinterpret_cast<CUstream>(stream),
                reinterpret_cast<CUdeviceptr>(&s.mapped->consumed), generation, 0);
            const auto recorded = cudaEventRecord(s.done, stream);
            if (consumed != CUDA_SUCCESS || recorded != cudaSuccess) {
                const auto completed = cudaStreamSynchronize(stream);
                if (completed == cudaSuccess)
                    __atomic_store_n(&s.host->consumed, generation, __ATOMIC_RELEASE);
                if (status == cudaSuccess) status = completed != cudaSuccess ? completed :
                    recorded != cudaSuccess ? recorded : cudaErrorUnknown;
            }
            if (status != cudaSuccess) return status;
            dst += n; src += n; bytes -= n;
        }
        return cudaSuccess;
    }
};

inline cudaError_t UploadImmutableHostWeight(void *destination, const void *source, size_t bytes,
                                            bool pinned, cudaStream_t stream) {
    if (pinned) return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamCaptureStatus capture;
    auto status = cudaStreamIsCapturing(stream, &capture);
    if (status != cudaSuccess) return status;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    int device;
    if ((status = cudaGetDevice(&device)) != cudaSuccess) return status;
    static thread_local HostUploadStreams<ImmutableHostUploadRing> rings;
    auto &entry = rings.Get(device, stream);
    if (!entry.attempted) {
        entry.attempted = true;
        entry.ring = std::make_unique<ImmutableHostUploadRing>();
        if (!entry.ring->Init(device)) entry.ring.reset();
    }
    return entry.ring ? entry.ring->Copy(destination, source, bytes, stream) :
        UploadHostWeight(destination, source, bytes, false, stream);
}

}} // namespace fastllm::cuda
