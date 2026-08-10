#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

bool RunDeferredBigBufferClearRegression() {
    FastllmCudaSetDevice(0);
    constexpr size_t kDeferredBytes = 304ULL * 1024ULL * 1024ULL;
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess ||
        freeBytes < kDeferredBytes + 128ULL * 1024ULL * 1024ULL) {
        cudaGetLastError();
        std::cout << "deferred big-buffer clear regression: SKIP "
                     "(insufficient free memory)\n";
        return true;
    }

    void *pointer = FastllmCudaMalloc(kDeferredBytes);
    if (pointer == nullptr ||
        cudaMemsetAsync(pointer, 0, 1, cudaStreamPerThread) != cudaSuccess) {
        if (pointer != nullptr) {
            FastllmCudaForceFree(pointer);
        }
        std::cerr << "failed to prepare deferred big-buffer clear test\n";
        return false;
    }
    if (!FastllmCudaFreeAfterCurrentThreadStream(pointer) ||
        cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) {
        FastllmCudaForceFree(pointer);
        std::cerr << "failed to defer the big-buffer pool release\n";
        return false;
    }

    // A completed deferred block larger than the pool's 300 MiB retention
    // budget must be queried and physically released by ClearBigBuffer. Before
    // the regression fix, reusePending was never polled here and this pointer
    // remained a valid CUDA allocation indefinitely.
    FastllmCudaClearBigBuffer();
    cudaPointerAttributes attributes;
    cudaError_t attributeState = cudaPointerGetAttributes(&attributes, pointer);
    bool allocationLive = false;
    if (attributeState == cudaSuccess) {
#if CUDART_VERSION < 10000
        allocationLive =
            attributes.memoryType == cudaMemoryTypeDevice;
#else
        allocationLive = attributes.type == cudaMemoryTypeDevice ||
            attributes.type == cudaMemoryTypeManaged;
#endif
    }
    if (allocationLive) {
        FastllmCudaForceFree(pointer);
        std::cerr << "completed deferred big buffer survived explicit clear\n";
        return false;
    }
    if (attributeState != cudaSuccess) {
        cudaGetLastError();
    }
    return true;
}

}  // namespace

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        std::cerr << "no CUDA device available for graph pool-miss regression\n";
        return 2;
    }
    const int participants = std::min(deviceCount, 4);
    FastllmCudaSetDevice(0);
    if (!FastllmCudaGraphMemoryPoolBegin()) {
        std::cerr << "failed to begin graph memory-pool capture\n";
        return 3;
    }

    std::vector<int> capturePassed(participants, 0);
    std::vector<std::string> captureErrors(participants);
    std::vector<std::unique_ptr<fastllm::Data> > capturedData(participants);
    std::vector<std::unique_ptr<fastllm::Data> > capturedDirectData(participants);
    std::vector<std::thread> workers;
    workers.reserve(participants);
    for (int device = 0; device < participants; device++) {
        workers.emplace_back([&, device]() {
            FastllmCudaSetDevice(device);
            if (!FastllmCudaGraphBeginCapture()) {
                captureErrors[device] = "failed to begin CUDA stream capture";
                return;
            }

            // A fresh process has no reusable FastLLM pool entries on any
            // device. Every rank therefore takes the same deterministic
            // capture-time pool-miss path.
            capturedData[device] = std::make_unique<fastllm::Data>(
                fastllm::DataType::FLOAT32, std::vector<int>{1024});
            fastllm::Data &data = *capturedData[device];
            data.dataDevice = fastllm::DataDevice::CUDA;
            data.dataDeviceIds = {device};
            data.Allocate(false);
            if (data.cudaData == nullptr || !data.cudaDataBorrowed ||
                data.expansionSize != 0 || !FastllmCudaGetThreadError()) {
                captureErrors[device] =
                    "capture pool miss did not produce a retryable placeholder";
                return;
            }
            capturedDirectData[device] = std::make_unique<fastllm::Data>(
                fastllm::DataType::FLOAT32, std::vector<int>{1024});
            fastllm::Data &directData = *capturedDirectData[device];
            directData.dataDevice = fastllm::DataDevice::CUDA;
            directData.dataDeviceIds = {device};
            directData.directMemory = true;
            directData.Allocate(false);
            if (directData.cudaData == nullptr ||
                !directData.cudaDataBorrowed ||
                directData.expansionSize != 0) {
                captureErrors[device] =
                    "direct capture allocation did not use the placeholder";
                return;
            }
            if (cudaMemsetAsync(data.cudaData, 0, 1, cudaStreamPerThread) !=
                    cudaSuccess) {
                captureErrors[device] =
                    "placeholder could not be recorded in the failed graph";
                return;
            }

            void *graph = nullptr;
            if (!FastllmCudaGraphEndCapture(&graph) || graph == nullptr) {
                captureErrors[device] =
                    "failed capture could not be closed cleanly";
                return;
            }
            FastllmCudaGraphDestroy(graph);
            capturePassed[device] = 1;
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }
    FastllmCudaGraphMemoryPoolAbort();

    for (int device = 0; device < participants; device++) {
        if (!capturePassed[device]) {
            std::cerr << "GPU " << device << ": " << captureErrors[device]
                      << "\n";
            return 4;
        }
    }

    FastllmCudaClearGraphError();
    for (int device = 0; device < participants; device++) {
        FastllmCudaSetDevice(device);
        FastllmCudaClearThreadError();
        fastllm::Data &data = *capturedData[device];
        fastllm::Data &directData = *capturedDirectData[device];
        data.Allocate(false);
        if (data.cudaData == nullptr || data.cudaDataBorrowed ||
            data.expansionSize < data.Count(0) ||
            FastllmCudaGetThreadError()) {
            std::cerr << "GPU " << device
                      << ": eager retry did not allocate real storage\n";
            return 5;
        }
        directData.Allocate(false);
        if (directData.cudaData == nullptr || directData.cudaDataBorrowed ||
            directData.expansionSize < directData.Count(0) ||
            FastllmCudaGetThreadError()) {
            std::cerr << "GPU " << device
                      << ": direct eager retry did not allocate real storage\n";
            return 5;
        }
        FastllmCudaMemset0(data.cudaData, data.GetBytes());
        FastllmCudaMemset0(directData.cudaData, directData.GetBytes());
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << "GPU " << device
                      << ": eager retry storage was not usable\n";
            return 6;
        }
    }

    if (!RunDeferredBigBufferClearRegression()) {
        return 7;
    }

    // Reserve construction must expose capacity failures without discarding
    // blocks already prepared for frozen serving or poisoning graph state.
    constexpr size_t reserveBytes = 3ULL * 1024ULL * 1024ULL;
    constexpr int reserveBlockCount = 3;
    FastllmCudaSetDevice(0);
    FastllmCudaClearThreadError();
    if (FastllmCudaTryMallocBigBuffers(
            reserveBytes, reserveBlockCount) != reserveBlockCount ||
        FastllmCudaGetThreadError() || FastllmCudaGetGraphError()) {
        std::cerr << "failed to seed non-destructive CUDA serve reserve\n";
        return 8;
    }
    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess ||
        totalBytes == 0) {
        std::cerr << "failed to query CUDA capacity for reserve regression\n";
        return 9;
    }
    const size_t impossibleReserveBytes =
        totalBytes < std::numeric_limits<size_t>::max()
            ? totalBytes + 1
            : totalBytes;
    if (FastllmCudaTryMallocBigBuffers(impossibleReserveBytes, 1) != 0 ||
        FastllmCudaGetThreadError() || FastllmCudaGetGraphError()) {
        std::cerr << "failed reserve allocation was not reported cleanly\n";
        return 10;
    }
    // The API server requests an allocation freeze after warmup. The request
    // must be ignored by default and enforced only with
    // FASTLLM_CUDA_MEM_CHECK enabled.
    FastllmCudaSetDevice(0);
    constexpr size_t warmedBytes = 2ULL * 1024ULL * 1024ULL;
    constexpr size_t missBytes = 64ULL * 1024ULL * 1024ULL;
    void *warmed = FastllmCudaMalloc(warmedBytes);
    if (warmed == nullptr) {
        std::cerr << "failed to seed CUDA pool for allocation-freeze regression\n";
        return 11;
    }
    FastllmCudaFree(warmed);
    DisableCudaMalloc();

    if (!fastllm::GetFastllmEnv().cudaMemCheck) {
        FastllmCudaClearThreadError();
        void *miss = FastllmCudaMalloc(missBytes);
        if (miss == nullptr || FastllmCudaGetThreadError()) {
            std::cerr << "allocator freeze was enabled without "
                         "FASTLLM_CUDA_MEM_CHECK\n";
            return 10;
        }
        FastllmCudaFree(miss);

        void *direct = FastllmCudaDirectMalloc(1);
        if (direct == nullptr || FastllmCudaGetThreadError()) {
            std::cerr << "direct allocator freeze was enabled without "
                         "FASTLLM_CUDA_MEM_CHECK\n";
            return 10;
        }
        FastllmCudaDirectFree(direct);

        std::cout << "CUDA graph pool-miss regression: PASS; allocation "
                     "freeze disabled by FASTLLM_CUDA_MEM_CHECK ("
                  << participants << " GPU" << (participants == 1 ? "" : "s")
                  << ")\n";
        return 0;
    }

    // Acquire every reserve block simultaneously after the freeze.  This can
    // only succeed from the existing pool, so an OOM retry that discarded the
    // blocks above cannot be hidden by allocating replacements here.
    std::vector<void*> reservedBlocks;
    for (int i = 0; i < reserveBlockCount; i++) {
        void *reserved = FastllmCudaMalloc(reserveBytes);
        if (reserved == nullptr) {
            std::cerr << "failed reserve allocation discarded existing blocks\n";
            return 12;
        }
        reservedBlocks.push_back(reserved);
    }
    for (void *reserved : reservedBlocks) {
        FastllmCudaFree(reserved);
    }

    void *reused = FastllmCudaMalloc(warmedBytes);
    if (reused == nullptr) {
        std::cerr << "frozen allocator did not reuse warmed CUDA storage\n";
        return 13;
    }
    FastllmCudaFree(reused);

    FastllmCudaClearThreadError();
    if (FastllmCudaMalloc(missBytes) != nullptr ||
        !FastllmCudaGetThreadError()) {
        std::cerr << "frozen allocator did not reject a CUDA pool miss\n";
        return 14;
    }
    FastllmCudaClearThreadError();

    reused = FastllmCudaMalloc(warmedBytes);
    if (reused == nullptr) {
        std::cerr << "rejected pool miss discarded warmed CUDA storage\n";
        return 15;
    }
    FastllmCudaFree(reused);

    FastllmCudaClearThreadError();
    if (FastllmCudaDirectMalloc(1) != nullptr ||
        !FastllmCudaGetThreadError()) {
        std::cerr << "frozen allocator allowed a direct CUDA allocation\n";
        return 16;
    }
    FastllmCudaClearThreadError();

    std::cout << "CUDA graph pool-miss, deferred-clear, and allocation-freeze "
                 "regression: PASS; allocation freeze enabled by "
                 "FASTLLM_CUDA_MEM_CHECK ("
              << participants << " GPU" << (participants == 1 ? "" : "s")
              << ")\n";
    return 0;
}
