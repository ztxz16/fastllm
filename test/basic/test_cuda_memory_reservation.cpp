#include "devices/cuda/cudaworkspace.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace fastllm;

// CUDA translation units receive these declarations from fastllm-cuda.cuh.
extern "C" cudaError_t FastllmCudaCheckedMalloc(void **, size_t, const char *, int);
extern "C" cudaError_t FastllmCudaCheckedFree(void *);

static void Check(bool value, const char *what) {
    if (!value) throw std::runtime_error(what);
}

int main() {
    if (FastllmCudaGetDeviceCount() < 1) return 77;
    try {
        constexpr size_t MiB = 1024 * 1024;
        FastllmCudaSetDevice(0);
        FastllmCudaClearBigBufferAll();
        auto reserve = std::make_unique<CudaMemoryReservation>(0, 64 * MiB);
        void *raw = nullptr;
        Check(FastllmCudaCheckedMalloc(&raw, 0, __FILE__, __LINE__) == cudaSuccess && !raw,
              "zero-size allocation changed CUDA semantics");
        Check(FastllmCudaCheckedMalloc(nullptr, MiB, __FILE__, __LINE__) == cudaErrorInvalidValue,
              "invalid allocation pointer was not rejected");
        Check(FastllmCudaCheckedMalloc(&raw, MiB, __FILE__, __LINE__) == cudaSuccess &&
              IsCudaReservationPointer(raw), "checked allocation escaped reserve");
        Check(FastllmCudaCheckedFree(raw) == cudaSuccess && !IsCudaReservationPointer(raw),
              "checked free did not release a reserved suballocation");
        void *a = FastllmCudaDirectMalloc(16 * MiB);
        void *b = FastllmCudaDirectMalloc(16 * MiB);
        void *c = FastllmCudaDirectMalloc(32 * MiB);
        Check(a && b && c && IsCudaReservationPointer(a) && IsCudaReservationPointer(c), "direct allocation escaped reserve");
        void *overflow = nullptr;
        Check(FastllmCudaTryDirectMalloc(&overflow, MiB) == FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE &&
              !overflow && !FastllmCudaGetThreadError(), "capacity failure escaped/poisoned reserve");
        FastllmCudaDirectFree(b); FastllmCudaDirectFree(a);
        a = FastllmCudaDirectMalloc(32 * MiB);
        Check(a && IsCudaReservationPointer(a), "adjacent ranges did not coalesce");
        FastllmCudaDirectFree(a); FastllmCudaDirectFree(c);

        // On pressure, an asynchronously retired block must be reclaimable
        // after its event completes, rather than causing a false capacity OOM.
        a = FastllmCudaMalloc(48 * MiB);
        Check(cudaLaunchHostFunc(cudaStreamPerThread, [](void *) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }, nullptr) == cudaSuccess, "delayed stream callback failed");
        Check(FastllmCudaFreeAfterCurrentThreadStream(a), "pending free failed");
        b = nullptr;
        Check(FastllmCudaTryMalloc(&b, 64 * MiB) == FASTLLM_CUDA_TRY_MALLOC_SUCCESS && b,
              "pending retired storage caused a false capacity failure");
        FastllmCudaFree(b); FastllmCudaClearBigBufferAll();

        // Cached chunks can be evicted and split again without releasing the
        // backing allocation. Deferred stream writes must finish before reuse.
        a = FastllmCudaMalloc(9 * MiB);
        Check(IsCudaReservationPointer(a), "cached allocation escaped reserve");
        Check(cudaMemsetAsync(a, 0x5a, 9 * MiB, cudaStreamPerThread) == cudaSuccess, "write failed");
        Check(FastllmCudaFreeAfterCurrentThreadStream(a), "deferred release failed");
        cudaDeviceSynchronize();
        b = FastllmCudaMalloc(9 * MiB);
        std::vector<unsigned char> data(4096);
        FastllmCudaCopyFromDeviceToHost(data.data(), b, data.size());
        Check(a == b && data == std::vector<unsigned char>(data.size(), 0x5a), "cached storage corruption");
        FastllmCudaFree(b);
        FastllmCudaClearBigBufferAll();
        c = FastllmCudaDirectMalloc(64 * MiB);
        Check(c && IsCudaReservationPointer(c), "cache eviction did not reclaim arena");

        // Disabling admission must not invalidate already-owned memory.
        reserve.reset();
        Check(IsCudaReservationPointer(c), "reservation destroyed outstanding allocation");
        FastllmCudaMemset0(c, 4096);
        FastllmCudaDirectFree(c);
        Check(!IsCudaReservationPointer(c), "reservation ownership leaked");

        reserve = std::make_unique<CudaMemoryReservation>(0, 64 * MiB);
        std::exception_ptr error;
        std::thread worker([&] {
            try {
                FastllmCudaSetDevice(0);
                auto *p = FastllmCudaDirectMalloc(MiB);
                Check(p && IsCudaReservationPointer(p), "worker bypassed device reservation");
                FastllmCudaDirectFree(p);
            } catch (...) { error = std::current_exception(); }
        });
        worker.join(); if (error) std::rethrow_exception(error);
        if (FastllmCudaGetDeviceCount() > 1) {
            FastllmCudaSetDevice(1);
            auto *p = FastllmCudaDirectMalloc(MiB);
            Check(p && !IsCudaReservationPointer(p), "reservation crossed devices");
            FastllmCudaDirectFree(p);
        }
        FastllmCudaSetDevice(0);
        std::cout << "PASS: CUDA memory reservation" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl; return 1;
    }
}
