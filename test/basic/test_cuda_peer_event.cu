#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-event.h"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#define CK(call) do { const auto error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s (line %d)\n", #call, cudaGetErrorString(error), __LINE__); \
    std::abort(); } } while (0)

namespace {
constexpr int words = 65536;
__global__ void Fill(int *data, int stamp) {
    // Keep the producer pending when the host publishes its event. Consumers
    // must observe this write through the event, without a producer-side sync.
    if (!threadIdx.x) {
        const auto start = clock64();
        while (clock64() - start < 1000000) {}
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words) data[i] = stamp ^ i;
}
__global__ void Verify(const int *data, int stamp, unsigned *errors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words && data[i] != (stamp ^ i)) atomicAdd(errors, 1u);
}
}

int main(int argc, char **argv) {
    const int ranks = argc > 1 ? std::atoi(argv[1]) : 2;
    const int first = argc > 2 ? std::atoi(argv[2]) : 0;
    const int sources = argc > 3 ? std::atoi(argv[3]) : 128;
    // Diagnostic A/B switch; CTest always exercises the production relay.
    const bool direct = argc > 4 && std::strcmp(argv[4], "direct") == 0;
    if (ranks < 2 || first < 0 || sources < 1 || sources > 1024) return 2;
    int available = 0;
    if (cudaGetDeviceCount(&available) != cudaSuccess || available < first + ranks) return 77;
    for (int d = first; d < first + ranks; ++d) {
        CK(cudaSetDevice(d)); CK(cudaFree(nullptr));
        for (int peer = first; peer < first + ranks; ++peer) if (peer != d) {
            int can = 0; CK(cudaDeviceCanAccessPeer(&can, d, peer));
            if (!can) return 77;
            CK(cudaDeviceEnablePeerAccess(peer, 0));
        }
    }
    CK(cudaSetDevice(first));
    int *source = nullptr;
    CK(cudaMalloc(&source, words * sizeof(int)));
    void *ready = FastllmCudaEventCreate(), *sourceReady = FastllmCudaEventCreate();
    void *relay = FastllmCudaStreamCreate();
    std::atomic<int> phase{0}, done{0}, submitted{0};
    std::atomic<bool> stop{false};
    std::atomic<unsigned> mismatches{0};
    std::vector<std::thread> consumers, producers;
    std::mutex releaseMutex;
    std::condition_variable releaseCV;
    bool release = false;

    for (int rank = 1; rank < ranks; ++rank) consumers.emplace_back([&, rank] {
        const int device = first + rank;
        CK(cudaSetDevice(device));
        int *copy = nullptr;
        unsigned *errors = nullptr;
        CK(cudaMalloc(&copy, words * sizeof(int)));
        CK(cudaMalloc(&errors, sizeof(unsigned)));
        CK(cudaMemset(errors, 0, sizeof(unsigned)));
        int seen = 0;
        while (!stop.load(std::memory_order_acquire)) {
            const int stamp = phase.load(std::memory_order_acquire);
            if (stamp == seen) { std::this_thread::yield(); continue; }
            seen = stamp;
            FastllmCudaCurrentThreadStreamWaitEvent(ready);
            CK(cudaMemcpyPeerAsync(copy, device, source, first, words * sizeof(int), cudaStreamPerThread));
            Verify<<<words / 256, 256, 0, cudaStreamPerThread>>>(copy, stamp, errors);
            CK(cudaGetLastError());
            CK(cudaStreamSynchronize(cudaStreamPerThread));
            done.fetch_add(1, std::memory_order_release);
        }
        unsigned count = 0;
        CK(cudaMemcpy(&count, errors, sizeof(count), cudaMemcpyDeviceToHost));
        mismatches.fetch_add(count);
        CK(cudaFree(errors)); CK(cudaFree(copy));
    });
    auto publish = [&](int stamp) {
        CK(cudaSetDevice(first));
        Fill<<<words / 256, 256, 0, cudaStreamPerThread>>>(source, stamp);
        CK(cudaGetLastError());
        if (direct) FastllmCudaEventRecordCurrentThread(ready);
        else FastllmCudaEventRecordForPeer(ready, sourceReady, relay);
        done.store(0);
        phase.store(stamp, std::memory_order_release);
        while (done.load(std::memory_order_acquire) != ranks - 1) std::this_thread::yield();
    };
    for (int stamp = 1; stamp <= 5; ++stamp) publish(stamp);
    for (int i = 0; i < sources; ++i) {
        producers.emplace_back([&, i] {
            publish(i + 6);
            submitted.store(i + 1, std::memory_order_release);
            // Retain every PTDS until all producer switches have been tested.
            std::unique_lock<std::mutex> lock(releaseMutex);
            releaseCV.wait(lock, [&] { return release; });
        });
        while (submitted.load(std::memory_order_acquire) != i + 1) std::this_thread::yield();
    }
    { std::lock_guard<std::mutex> lock(releaseMutex); release = true; }
    releaseCV.notify_all();
    for (auto &thread : producers) thread.join();
    stop.store(true, std::memory_order_release);
    for (auto &thread : consumers) thread.join();
    CK(cudaSetDevice(first));
    FastllmCudaStreamDestroy(relay);
    FastllmCudaEventDestroy(sourceReady); FastllmCudaEventDestroy(ready);
    CK(cudaFree(source));
    std::printf("PEER_EVENT %s ranks=%d first=%d sources=%d direct=%d mismatches=%u\n",
        mismatches.load() ? "FAIL" : "PASS", ranks, first, sources, direct, mismatches.load());
    return mismatches.load() ? 1 : 0;
}
