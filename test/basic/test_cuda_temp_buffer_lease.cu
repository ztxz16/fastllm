#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <thread>

static void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void Fill(unsigned *p, size_t n, unsigned tag) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = tag;
}
__global__ void Spin() {
    const unsigned long long start = clock64();
    while (clock64() - start < 2000000ULL) {}
}
__global__ void Verify(const unsigned *p, size_t n, unsigned tag, unsigned long long *bad) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (p[i] != tag) atomicAdd(bad, 1ULL);
}

int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: requires a GPU");
            return 0;
        }
        Check(cudaSetDevice(0));
        int failures = 0;
        const size_t bytes = 1 << 20;

        // A borrow while the shared buffer is held gets private memory.
        bool ownA = true, ownB = false;
        void *a = FastllmBorrowCudaTempBuffer(bytes, nullptr, &ownA);
        void *b = FastllmBorrowCudaTempBuffer(bytes, nullptr, &ownB);
        if (a == nullptr || b == nullptr || a == b || ownA || !ownB) {
            std::printf("FAIL nested borrow: a=%p b=%p ownA=%d ownB=%d\n", a, b, ownA, ownB); ++failures;
        }
        FastllmReleaseCudaTempBuffer(b, ownB);
        FastllmReleaseCudaTempBuffer(a, ownA);
        bool ownC = true;
        void *c = FastllmBorrowCudaTempBuffer(bytes, nullptr, &ownC);
        if (c != a || ownC) {
            std::printf("FAIL reuse after release: a=%p c=%p ownC=%d\n", a, c, ownC); ++failures;
        }
        FastllmReleaseCudaTempBuffer(c, ownC);

        // Paged/MLA attention use the FlashInfer float workspace directly, so a borrow must not hand it out.
        void *flashInferWorkspace = FastllmCudaGetFlashInferFloatWorkspace(nullptr);
        bool ownD = true;
        void *d = FastllmBorrowCudaTempBuffer(bytes, nullptr, &ownD);
        if (d == nullptr || d == flashInferWorkspace) {
            std::printf("FAIL borrow returned the FlashInfer workspace: d=%p workspace=%p\n", d, flashInferWorkspace); ++failures;
        }
        FastllmReleaseCudaTempBuffer(d, ownD);

        // Two host threads (two per-thread streams) borrow concurrently, like the DFlash
        // scheduler and a TP worker. Each tags the buffer and checks the tag survived.
        unsigned long long corrupted = 0;
        std::mutex corruptedMutex;
        auto worker = [&](unsigned tag) {
            Check(cudaSetDevice(0));
            const size_t n = bytes / sizeof(unsigned);
            unsigned long long *bad = nullptr, host = 0;
            Check(cudaMalloc(&bad, sizeof(*bad)));
            for (int it = 0; it < 100; ++it) {
                Check(cudaMemsetAsync(bad, 0, sizeof(*bad), cudaStreamPerThread));
                bool own = false;
                auto *p = static_cast<unsigned *>(FastllmBorrowCudaTempBuffer(bytes, nullptr, &own));
                Fill<<<64, 256, 0, cudaStreamPerThread>>>(p, n, tag);
                Spin<<<1, 1, 0, cudaStreamPerThread>>>();
                Verify<<<64, 256, 0, cudaStreamPerThread>>>(p, n, tag, bad);
                FastllmReleaseCudaTempBuffer(p, own);
                Check(cudaMemcpyAsync(&host, bad, sizeof(host), cudaMemcpyDeviceToHost, cudaStreamPerThread));
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                std::lock_guard<std::mutex> guard(corruptedMutex);
                corrupted += host;
            }
            Check(cudaFree(bad));
        };
        std::thread t0(worker, 0xA5A5A5A5u), t1(worker, 0x5A5A5A5Au);
        t0.join();
        t1.join();
        if (corrupted != 0) {
            std::printf("FAIL cross-stream sharing: %llu words overwritten by the other stream\n", corrupted); ++failures;
        }

        if (failures) return 1;
        std::puts("PASS: temp buffer lease and cross-stream release");
        return 0;
    } catch (const std::exception &e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
