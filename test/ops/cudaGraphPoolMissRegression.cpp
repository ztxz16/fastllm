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

bool RunCaptureMixedSizePoolRegression() {
    FastllmCudaSetDevice(0);
    constexpr size_t largeBytes = 64 * 1024;
    constexpr size_t smallBytes = 4 * 1024;
    // Seed the larger block first. Capture must leave it for the larger request.
    void *large = FastllmCudaMalloc(largeBytes);
    void *small = FastllmCudaMalloc(smallBytes);
    if (large == nullptr || small == nullptr) return false;
    FastllmCudaFree(large);
    FastllmCudaFree(small);
    if (!FastllmCudaGraphPrepareCaptureDevice() ||
        !FastllmCudaGraphMemoryPoolBegin() || !FastllmCudaGraphBeginCapture()) {
        return false;
    }
    void *first = FastllmCudaMalloc(smallBytes);
    void *second = FastllmCudaMalloc(largeBytes);
    bool passed = first != nullptr && second != nullptr &&
                  !FastllmCudaGetThreadError();
    if (passed) {
        passed = cudaMemsetAsync(first, 0x35, smallBytes, cudaStreamPerThread) == cudaSuccess &&
                 cudaMemsetAsync(second, 0x79, largeBytes, cudaStreamPerThread) == cudaSuccess;
    }
    FastllmCudaFree(first);
    FastllmCudaFree(second);
    void *graph = nullptr;
    passed = FastllmCudaGraphEndCapture(&graph) && graph != nullptr && passed;
    std::vector<void*> pins;
    void *exec = nullptr;
    passed = passed && FastllmCudaGraphMemoryPoolEnd(pins);
    if (!passed) FastllmCudaGraphMemoryPoolAbort();
    if (passed) {
        unsigned char values[2] = {};
        passed = FastllmCudaGraphInstantiate(graph, &exec) &&
                 FastllmCudaGraphLaunch(exec) &&
                 cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess &&
                 cudaMemcpy(values, first, 1, cudaMemcpyDeviceToHost) == cudaSuccess &&
                 cudaMemcpy(values + 1, second, 1, cudaMemcpyDeviceToHost) == cudaSuccess &&
                 values[0] == 0x35 && values[1] == 0x79;
    }
    if (exec != nullptr) FastllmCudaGraphExecDestroy(exec);
    if (graph != nullptr) FastllmCudaGraphDestroy(graph);
    FastllmCudaGraphMemoryPoolRelease(pins);
    FastllmCudaClearThreadError();
    FastllmCudaClearGraphError();
    // Leave the pool cold for the deliberate allocation-failure test below.
    FastllmCudaForceFree(large);
    FastllmCudaForceFree(small);
    if (!passed) std::cerr << "capture exhausted a pool with sufficient mixed-size capacity\n";
    else std::cout << "mixed-size capture pool and replay: PASS\n";
    return passed;
}

bool RunExternalCaptureQueryRegression() {
    bool passed = false;
    std::string error;
    std::thread worker([&]() {
        FastllmCudaSetDevice(0);
        void *pointer = nullptr;
        cudaGraph_t graph = nullptr;
        bool captureStarted = false;
        if (FastllmCudaGraphIsCapturing()) {
            error = "exact capture query reported an idle stream as active";
            return;
        }
        if (cudaMalloc(&pointer, 1) != cudaSuccess) {
            error = "failed to allocate external-capture probe storage";
            return;
        }
        if (cudaStreamBeginCapture(
                cudaStreamPerThread,
                cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            error = "failed to start external CUDA stream capture";
        } else {
            captureStarted = true;
        }
        if (error.empty() && !FastllmCudaGraphIsCapturing()) {
            error = "exact query missed an externally started capture";
        }
        if (error.empty() && !FastllmCudaGraphIsCapturingFast()) {
            error = "external capture was not latched for hot-path queries";
        }
        if (error.empty() &&
            cudaMemsetAsync(pointer, 0, 1, cudaStreamPerThread) !=
                cudaSuccess) {
            error = "failed to record work in external CUDA capture";
        }
        if (captureStarted) {
            cudaError_t endState =
                cudaStreamEndCapture(cudaStreamPerThread, &graph);
            if (error.empty() && endState != cudaSuccess) {
                error = "failed to finish external CUDA stream capture";
            }
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
        if (error.empty() && FastllmCudaGraphIsCapturing()) {
            error = "exact capture query remained active after capture end";
        }
        if (error.empty() && FastllmCudaGraphIsCapturingFast()) {
            error = "hot-path capture query remained active after capture end";
        }
        passed = error.empty();
    });
    worker.join();
    if (!passed) {
        std::cerr << error << "\n";
    }
    return passed;
}

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

bool RunExpansionWorkspaceReuseRegression() {
    FastllmCudaSetDevice(0);
    FastllmCudaClearBigBufferAll();
    constexpr size_t bytes = 304ULL * 1024ULL * 1024ULL;
    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess ||
        bytes > totalBytes / 4 || freeBytes < bytes + 128ULL * 1024ULL * 1024ULL) {
        cudaGetLastError();
        std::cout << "expansion workspace reuse: SKIP (insufficient free memory)\n";
        return true;
    }
    auto isLive = [](void *pointer) {
        cudaPointerAttributes attributes;
        if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
#if CUDART_VERSION < 10000
        return attributes.memoryType == cudaMemoryTypeDevice;
#else
        return attributes.type == cudaMemoryTypeDevice;
#endif
    };

    // A small KV-like tensor grows while a much larger operator workspace is
    // idle. Growth must preserve both the tensor contents and workspace reuse.
    const std::vector<float> expected = {1.0f, -2.0f, 3.0f, -4.0f};
    fastllm::Data cache(fastllm::DataType::FLOAT32, {1, 4}, expected);
    cache.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0});
    void *workspace = FastllmCudaMalloc(bytes);
    FastllmCudaFree(workspace);
    cache.Expansion({2, 4});
    std::vector<float> actual(4);
    bool passed = cudaMemcpy(actual.data(), cache.cudaData, 4 * sizeof(float),
                            cudaMemcpyDeviceToHost) == cudaSuccess && actual == expected;
    passed = passed && isLive(workspace);
    if (!passed) {
        std::cerr << "tensor growth discarded a reusable workspace or changed contents\n";
        return false;
    }
    void *reused = FastllmCudaMalloc(bytes);
    passed = reused == workspace;
    FastllmCudaFree(reused);

    // Retention is a cache, not a reservation: pressure must reclaim idle
    // workspace while preserving the live tensor and graph-owned allocations.
    void *probe = nullptr;
    passed = passed && totalBytes < std::numeric_limits<size_t>::max() &&
        FastllmCudaTryDirectMalloc(&probe, totalBytes + 1) ==
            FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE && probe == nullptr &&
        !isLive(workspace) && isLive(cache.cudaData);
    passed = passed && cudaMemcpy(actual.data(), cache.cudaData, 4 * sizeof(float),
                                 cudaMemcpyDeviceToHost) == cudaSuccess && actual == expected;
    if (!passed) std::cerr << "retained workspace was not reusable or reclaimable\n";
    else std::cout << "expansion retains reusable and reclaimable workspace: PASS\n";
    return passed;
}

bool RunPinnedWorkspaceOomRegression() {
    FastllmCudaSetDevice(0);
    struct RestoreCaptureMode {
        bool previous = FastllmCudaGraphSetManagedCaptureOnly(true);
        ~RestoreCaptureMode() { FastllmCudaGraphSetManagedCaptureOnly(previous); }
    } restore;
    constexpr size_t bytes = 4ULL * 1024ULL * 1024ULL;
    void *buffer = FastllmCudaMalloc(bytes);
    FastllmCudaFree(buffer);
    if (!FastllmCudaGraphMemoryPoolBegin() || !FastllmCudaGraphBeginCapture()) return false;
    buffer = FastllmCudaMalloc(bytes);
    if (cudaMemsetAsync(buffer, 0x5a, bytes, cudaStreamPerThread) != cudaSuccess) return false;
    FastllmCudaFree(buffer);
    void *graph = nullptr, *exec = nullptr;
    std::vector<void *> pins;
    if (!FastllmCudaGraphEndCapture(&graph) || !FastllmCudaGraphMemoryPoolEnd(pins) ||
        std::find(pins.begin(), pins.end(), buffer) == pins.end() ||
        !FastllmCudaGraphInstantiate(graph, &exec)) return false;

    // The tensor released its workspace, but the graph still owns the address.
    // Force the allocator's OOM retry without consuming the device's capacity.
    size_t freeBytes = 0, totalBytes = 0;
    void *probe = nullptr;
    bool passed = cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess &&
        totalBytes < std::numeric_limits<size_t>::max() &&
        FastllmCudaTryDirectMalloc(&probe, totalBytes + 1) ==
            FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE && probe == nullptr;
    cudaPointerAttributes attributes;
    passed = passed && cudaPointerGetAttributes(&attributes, buffer) == cudaSuccess;
#if CUDART_VERSION < 10000
    passed = passed && attributes.memoryType == cudaMemoryTypeDevice;
#else
    passed = passed && attributes.type == cudaMemoryTypeDevice;
#endif
    unsigned char value = 0;
    passed = passed && FastllmCudaGraphLaunch(exec) &&
        cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess &&
        cudaMemcpy(&value, buffer, 1, cudaMemcpyDeviceToHost) == cudaSuccess && value == 0x5a;
    FastllmCudaGraphExecDestroy(exec);
    FastllmCudaGraphDestroy(graph);
    FastllmCudaGraphMemoryPoolRelease(pins);
    if (!passed) std::cerr << "OOM retry released a graph-owned workspace\n";
    else std::cout << "graph-owned workspace survives OOM retry: PASS\n";
    return passed;
}

}  // namespace

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        std::cerr << "no CUDA device available for graph pool-miss regression\n";
        return 77;
    }
    if (!RunExternalCaptureQueryRegression()) {
        return 17;
    }
    if (!RunCaptureMixedSizePoolRegression()) {
        return 18;
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
    std::vector<void *> copySources(participants, nullptr);
    std::vector<void *> scoreWorkspaces(participants, nullptr);
    std::vector<std::thread> workers;
    workers.reserve(participants);
    for (int device = 0; device < participants; device++) {
        workers.emplace_back([&, device]() {
            FastllmCudaSetDevice(device);
            if (cudaMalloc(&copySources[device], 4096) != cudaSuccess) {
                captureErrors[device] = "failed to prepare the copy source";
                return;
            }
            scoreWorkspaces[device] = FastllmCudaMalloc(2);
            if (scoreWorkspaces[device] == nullptr) {
                captureErrors[device] = "failed to prepare the score workspace";
                return;
            }
            FastllmCudaFree(scoreWorkspaces[device]);
            if (!FastllmCudaGraphBeginCapture()) {
                captureErrors[device] = "failed to begin CUDA stream capture";
                return;
            }

            // The only pooled block is two bytes: every Data allocation below
            // misses, and attention can allocate only one of its two scores.
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

            // This copy exceeds the placeholder. Keep the failed capture valid
            // until all ranks can abort, including a partial score allocation.
            FastllmCudaMemcpy2DDeviceToDevice(data.cudaData, 4096,
                copySources[device], 4096, 4096, 1);
            fastllm::Data halfInput(fastllm::DataType::FLOAT16, {1, 1, 256});
            halfInput.dataDevice = fastllm::DataDevice::CUDA;
            halfInput.dataDeviceIds = {device};
            halfInput.cudaData = data.cudaData;
            halfInput.cudaDataBorrowed = true;
            fastllm::Data emptyMask;
            if (FastllmCudaHalfAttention(halfInput, halfInput, halfInput,
                    emptyMask, halfInput, 1, 0.0625f, 1) ||
                cudaPeekAtLastError() != cudaSuccess ||
                FastllmCudaGraphCaptureInvalidated()) {
                captureErrors[device] = "pool miss escaped to a CUDA copy or cuBLAS call";
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
    for (int device = 0; device < participants; ++device) {
        FastllmCudaSetDevice(device);
        if (copySources[device] != nullptr) cudaFree(copySources[device]);
        if (!capturePassed[device]) {
            std::cerr << "GPU " << device << ": " << captureErrors[device]
                      << "\n";
            return 4;
        }
        void *reused = FastllmCudaMalloc(2);
        FastllmCudaFree(reused);
        if (reused != scoreWorkspaces[device]) {
            std::cerr << "GPU " << device << ": failed attention capture leaked its score workspace\n";
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
    if (!RunExpansionWorkspaceReuseRegression()) {
        return 23;
    }
    if (!RunPinnedWorkspaceOomRegression()) {
        return 22;
    }

    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess ||
        totalBytes == 0) {
        std::cerr << "failed to query CUDA capacity for allocator regression\n";
        return 8;
    }
    const size_t impossibleReserveBytes =
        totalBytes < std::numeric_limits<size_t>::max()
            ? totalBytes + 1
            : totalBytes;
    // Exercise the real cudaMalloc OOM path before seeding the serve reserve.
    // Both optional allocators must clear CUDA's last-error slot and preserve
    // FastLLM's thread/capture error state when only capacity is insufficient.
    FastllmCudaClearThreadError();
    FastllmCudaClearGraphError();
    cudaGetLastError();
    void *capacityProbe = nullptr;
    if (FastllmCudaTryMalloc(&capacityProbe, impossibleReserveBytes) !=
            FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE ||
        capacityProbe != nullptr || FastllmCudaGetThreadError() ||
        FastllmCudaGetGraphError() ||
        cudaPeekAtLastError() != cudaSuccess) {
        std::cerr << "optional pooled OOM did not preserve CUDA error state\n";
        return 20;
    }
    if (FastllmCudaTryDirectMalloc(
            &capacityProbe, impossibleReserveBytes) !=
            FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE ||
        capacityProbe != nullptr || FastllmCudaGetThreadError() ||
        FastllmCudaGetGraphError() ||
        cudaPeekAtLastError() != cudaSuccess) {
        std::cerr << "optional direct OOM did not preserve CUDA error state\n";
        return 21;
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

    // Optional acceleration workspaces may fall back when the frozen pool has
    // insufficient capacity. Both pooled and direct try-allocation APIs must
    // report that condition without poisoning thread or graph error state.
    FastllmCudaClearThreadError();
    FastllmCudaClearGraphError();
    cudaGetLastError();
    void *optional = nullptr;
    if (FastllmCudaTryMalloc(&optional, missBytes) !=
            FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE ||
        optional != nullptr || FastllmCudaGetThreadError() ||
        FastllmCudaGetGraphError() ||
        cudaPeekAtLastError() != cudaSuccess) {
        std::cerr << "optional pooled allocation did not report a clean "
                     "capacity failure\n";
        return 12;
    }
    if (FastllmCudaTryDirectMalloc(&optional, 1) !=
            FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE ||
        optional != nullptr || FastllmCudaGetThreadError() ||
        FastllmCudaGetGraphError() ||
        cudaPeekAtLastError() != cudaSuccess) {
        std::cerr << "optional direct allocation did not report a clean "
                     "capacity failure\n";
        return 13;
    }
    if (FastllmCudaTryMalloc(&optional, warmedBytes) !=
            FASTLLM_CUDA_TRY_MALLOC_SUCCESS ||
        optional == nullptr || FastllmCudaGetThreadError() ||
        FastllmCudaGetGraphError()) {
        std::cerr << "optional pooled allocation did not reuse warmed storage\n";
        return 14;
    }
    FastllmCudaFree(optional);

    // Acquire every reserve block simultaneously after the freeze.  This can
    // only succeed from the existing pool, so an OOM retry that discarded the
    // blocks above cannot be hidden by allocating replacements here.
    std::vector<void*> reservedBlocks;
    for (int i = 0; i < reserveBlockCount; i++) {
        void *reserved = FastllmCudaMalloc(reserveBytes);
        if (reserved == nullptr) {
            std::cerr << "failed reserve allocation discarded existing blocks\n";
            return 15;
        }
        reservedBlocks.push_back(reserved);
    }
    for (void *reserved : reservedBlocks) {
        FastllmCudaFree(reserved);
    }

    void *reused = FastllmCudaMalloc(warmedBytes);
    if (reused == nullptr) {
        std::cerr << "frozen allocator did not reuse warmed CUDA storage\n";
        return 16;
    }
    FastllmCudaFree(reused);

    FastllmCudaClearThreadError();
    if (FastllmCudaMalloc(missBytes) != nullptr ||
        !FastllmCudaGetThreadError()) {
        std::cerr << "frozen allocator did not reject a CUDA pool miss\n";
        return 17;
    }
    FastllmCudaClearThreadError();

    reused = FastllmCudaMalloc(warmedBytes);
    if (reused == nullptr) {
        std::cerr << "rejected pool miss discarded warmed CUDA storage\n";
        return 18;
    }
    FastllmCudaFree(reused);

    FastllmCudaClearThreadError();
    if (FastllmCudaDirectMalloc(1) != nullptr ||
        !FastllmCudaGetThreadError()) {
        std::cerr << "frozen allocator allowed a direct CUDA allocation\n";
        return 19;
    }
    FastllmCudaClearThreadError();

    std::cout << "CUDA graph pool-miss, deferred-clear, and allocation-freeze "
                 "regression: PASS; allocation freeze enabled by "
                 "FASTLLM_CUDA_MEM_CHECK ("
              << participants << " GPU" << (participants == 1 ? "" : "s")
              << ")\n";
    return 0;
}
