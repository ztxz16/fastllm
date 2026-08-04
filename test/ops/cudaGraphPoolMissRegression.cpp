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
        return 7;
    }
    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess ||
        totalBytes == 0) {
        std::cerr << "failed to query CUDA capacity for reserve regression\n";
        return 8;
    }
    const size_t impossibleReserveBytes =
        totalBytes < std::numeric_limits<size_t>::max()
            ? totalBytes + 1
            : totalBytes;
    if (FastllmCudaTryMallocBigBuffers(impossibleReserveBytes, 1) != 0 ||
        FastllmCudaGetThreadError() || FastllmCudaGetGraphError()) {
        std::cerr << "failed reserve allocation was not reported cleanly\n";
        return 9;
    }
    // The API server freezes real CUDA allocations after warmup. Verify that
    // the freeze is effective without FASTLLM_CUDA_MEM_CHECK, that warmed pool
    // storage remains reusable, and that a rejected miss does not discard the
    // reserve while attempting an impossible retry.
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

    // Acquire every reserve block simultaneously after the freeze.  This can
    // only succeed from the existing pool, so an OOM retry that discarded the
    // blocks above cannot be hidden by allocating replacements here.
    std::vector<void*> reservedBlocks;
    for (int i = 0; i < reserveBlockCount; i++) {
        void *reserved = FastllmCudaMalloc(reserveBytes);
        if (reserved == nullptr) {
            std::cerr << "failed reserve allocation discarded existing blocks\n";
            return 10;
        }
        reservedBlocks.push_back(reserved);
    }
    for (void *reserved : reservedBlocks) {
        FastllmCudaFree(reserved);
    }

    void *reused = FastllmCudaMalloc(warmedBytes);
    if (reused == nullptr) {
        std::cerr << "frozen allocator did not reuse warmed CUDA storage\n";
        return 12;
    }
    FastllmCudaFree(reused);

    FastllmCudaClearThreadError();
    if (FastllmCudaMalloc(missBytes) != nullptr ||
        !FastllmCudaGetThreadError()) {
        std::cerr << "frozen allocator did not reject a CUDA pool miss\n";
        return 13;
    }
    FastllmCudaClearThreadError();

    reused = FastllmCudaMalloc(warmedBytes);
    if (reused == nullptr) {
        std::cerr << "rejected pool miss discarded warmed CUDA storage\n";
        return 14;
    }
    FastllmCudaFree(reused);

    FastllmCudaClearThreadError();
    if (FastllmCudaDirectMalloc(1) != nullptr ||
        !FastllmCudaGetThreadError()) {
        std::cerr << "frozen allocator allowed a direct CUDA allocation\n";
        return 15;
    }
    FastllmCudaClearThreadError();

    std::cout << "CUDA graph pool-miss and allocation-freeze regression: PASS ("
              << participants << " GPU" << (participants == 1 ? "" : "s")
              << ")\n";
    return 0;
}
