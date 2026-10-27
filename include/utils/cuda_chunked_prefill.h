#ifndef FASTLLM_CUDA_CHUNKED_PREFILL_H
#define FASTLLM_CUDA_CHUNKED_PREFILL_H

#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"
#include "utils/persistent_worker_group.h"

#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <stdexcept>

namespace fastllm {
    struct CudaPrefillStage {
        int device;
        int firstLayer;
        int endLayer;
    };

    // Only contiguous, single-GPU stages are supported. A GPU cannot own two
    // stages: each device's mutable weights, allocator and handles have one
    // compute worker. Use the same weighted placement as ordinary Forward.
    inline bool BuildCudaPrefillStages(
            const std::map<std::string, int> &deviceMap, int layers,
            std::vector<CudaPrefillStage> &stages) {
        stages.clear();
        if (layers <= 0 || deviceMap.empty()) return false;
        for (const auto &entry : deviceMap) {
            if (entry.second <= 0) return false;
        }
        const int deviceCount = FastllmCudaGetDeviceCount();
        std::set<int> seen;
        for (int layer = 0; layer < layers; ++layer) {
            const std::string name = SelectDeviceFromMap(deviceMap, layer + 1, layers);
            int device = 0;
            if (name != "cuda") {
                if (name.compare(0, 5, "cuda:") != 0 || name.size() == 5) return false;
                for (size_t i = 5; i < name.size(); ++i) {
                    if (name[i] < '0' || name[i] > '9' || device > 100000) return false;
                    device = device * 10 + name[i] - '0';
                }
            }
            if (device >= deviceCount) return false;
            if (stages.empty() || stages.back().device != device) {
                if (!seen.insert(device).second) return false;
                stages.push_back({device, layer, layer + 1});
            } else {
                stages.back().endLayer = layer + 1;
            }
        }
        // Embedding and the output head must stay on their adjacent stages.
        return stages.size() > 1 &&
            SelectDeviceFromMap(deviceMap, 0, layers) ==
                SelectDeviceFromMap(deviceMap, 1, layers);
    }

    // Synchronous at the request boundary, asynchronous between stages. The
    // callback embeds a chunk on rank 0, updates only this rank's layers, and
    // samples only the last chunk on the final rank. It must leave a contiguous
    // activation on this rank's GPU, no larger than maxShape.
    class CudaChunkedPrefillPipeline {
    public:
        using ForwardChunk = std::function<void(int, int, Data &)>;

        void Run(const std::vector<int> &devices, int chunks,
                 DataType type, const std::vector<int> &maxShape,
                 const ForwardChunk &forward) {
            const int deviceCount = FastllmCudaGetDeviceCount();
            if (devices.size() < 2 || chunks < 2 || maxShape.empty() ||
                std::any_of(maxShape.begin(), maxShape.end(), [](int dim) { return dim <= 0; }) ||
                std::any_of(devices.begin(), devices.end(),
                    [&](int device) { return device < 0 || device >= deviceCount; }) ||
                std::set<int>(devices.begin(), devices.end()).size() != devices.size()) {
                throw std::invalid_argument("Invalid CUDA prefill pipeline layout.");
            }
            DeviceScope callerDevice;
            CheckCudaError();
            FastllmCudaSetDevice(devices.front());
            // The legacy capability cache has process-wide lazy initialization.
            // Resolve it on the caller before any worker can access it.
            getCudaInfos();
            std::vector<std::array<std::unique_ptr<Slot>, 2>> slots(devices.size());
            for (int rank = 0; rank < (int)devices.size(); ++rank) {
                FastllmCudaSetDevice(devices[rank]);
                // Join earlier serial work, including lazy weight preparation,
                // before transferring ownership to the persistent worker.
                FastllmCudaSyncCurrentThreadStream();
                for (auto &slot : slots[rank]) {
                    slot.reset(new Slot(devices[rank],
                        rank + 1 < (int)devices.size() ? devices[rank + 1] : -1));
                    slot->hidden.dataType = type;
                    slot->hidden.UpdateUnitSize();
                    slot->hidden.dataDevice = DataDevice::CUDA;
                    slot->hidden.dataDeviceIds = {devices[rank]};
                    slot->hidden.Resize(maxShape);
                    slot->hidden.Allocate(false);
                    slot->capacity = slot->hidden.GetBytes();
                    CheckCudaError();
                }
            }

            std::mutex mutex;
            std::condition_variable cv;
            bool cancelled = false;
            std::vector<std::exception_ptr> errors(devices.size());
            workers.Run(devices, [&](int rank) {
                const int device = devices[rank];
                try {
                    FastllmCudaClearThreadError();
                    ExecutorScope executor(device);
                    const bool last = rank + 1 == (int)devices.size();
                    for (int chunk = 0; chunk < chunks; ++chunk) {
                        Slot &slot = *slots[rank][chunk % 2];
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            cv.wait(lock, [&]() {
                                return cancelled || last || chunk < 2 ||
                                       slot.consumed >= chunk - 2;
                            });
                            if (cancelled) break;
                        }
                        if (!last && chunk >= 2) {
                            FastllmCudaCurrentThreadStreamWaitEvent(slot.consumedEvent);
                        }
                        if (rank > 0) {
                            Slot &source = *slots[rank - 1][chunk % 2];
                            {
                                std::unique_lock<std::mutex> lock(mutex);
                                cv.wait(lock, [&]() {
                                    return cancelled || source.published == chunk;
                                });
                                if (cancelled) break;
                            }
                            FastllmCudaCurrentThreadStreamWaitEvent(source.readyEvent);
                            slot.hidden.Resize(source.hidden.dims);
                            slot.hidden.Allocate(false);
                            CheckCudaError();
                            CopyActivation(source, slot);
                            // Release the source as soon as the copy completes;
                            // this rank computes from its own destination buffer.
                            FastllmCudaEventRecordCurrentThread(source.consumedEvent);
                            CheckCudaError();
                            {
                                std::lock_guard<std::mutex> lock(mutex);
                                source.consumed = chunk;
                            }
                            cv.notify_all();
                        }
                        forward(rank, chunk, slot.hidden);
                        CheckCudaError();
                        AssertInFastLLM(!slot.hidden.isFake && !slot.hidden.cudaDataBorrowed &&
                            slot.hidden.dataType == type &&
                            slot.hidden.dataDevice == DataDevice::CUDA &&
                            slot.hidden.dataDeviceIds == std::vector<int>({device}) &&
                            slot.hidden.GetBytes() <= slot.capacity,
                            "CUDA prefill callback changed activation placement or capacity.");
                        if (!last) {
                            FastllmCudaEventRecordCurrentThread(slot.readyEvent);
                            CheckCudaError();
                            {
                                std::lock_guard<std::mutex> lock(mutex);
                                slot.published = chunk;
                            }
                            cv.notify_all();
                        }
                    }
                    // Buffer/state ownership returns to the caller only after
                    // every worker stream is drained, including the final chunk.
                    FastllmCudaSyncCurrentThreadStream();
                    CheckCudaError();
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        cancelled = true;
                    }
                    cv.notify_all();
                    // All waits reference events already recorded by a peer;
                    // cancellation never leaves a wait on an unrecorded event.
                    try {
                        FastllmCudaSetDevice(device);
                        FastllmCudaSyncCurrentThreadStream();
                    } catch (...) {}
                    throw;
                }
            }, errors);
            for (const auto &error : errors) {
                if (error) std::rethrow_exception(error);
            }
        }

    private:
        static void CheckCudaError() {
            if (FastllmCudaGetThreadError()) {
                throw std::runtime_error("CUDA error in chunked prefill pipeline.");
            }
        }

        struct DeviceScope {
            int previous = FastllmCudaGetDevice();
            ~DeviceScope() { FastllmCudaSetDevice(previous); }
        };

        struct ExecutorScope {
            void *previous = GetExecutor();
            explicit ExecutorScope(int device) {
                FastllmCudaSetDevice(device);
                static thread_local std::unique_ptr<Executor> executor(new Executor());
                executor->SetFirstDevice("cuda:" + std::to_string(device));
                executor->ClearProfiler();
                SetCurrentThreadExecutor(executor.get());
            }
            ~ExecutorScope() { SetCurrentThreadExecutor(previous); }
        };

        struct Slot {
            Data hidden;
            int device, consumer;
            int published = -1, consumed = -1; // guarded by Run's mutex
            void *readyEvent = nullptr, *consumedEvent = nullptr;
            void *pinned = nullptr;
            size_t capacity = 0;
            bool usePeerCopy = true; // only the consumer accesses transfer state

            Slot(int device, int consumer) : device(device), consumer(consumer) {
                if (consumer >= 0) {
                    readyEvent = FastllmCudaEventCreate();
                    CheckCudaError();
                    DeviceScope restore;
                    FastllmCudaSetDevice(consumer);
                    try {
                        consumedEvent = FastllmCudaEventCreate();
                        CheckCudaError();
                    } catch (...) {
                        FastllmCudaEventDestroy(readyEvent);
                        throw;
                    }
                }
            }
            ~Slot() {
                DeviceScope restore;
                FastllmCudaSetDevice(device);
                if (readyEvent) FastllmCudaEventDestroy(readyEvent);
                hidden.FreeSpace();
                if (consumedEvent) {
                    FastllmCudaSetDevice(consumer);
                    FastllmCudaEventDestroy(consumedEvent);
                }
                if (pinned) FastllmCudaHostFree(pinned);
            }
        };

        static void CopyActivation(Slot &source, Slot &destination) {
            const size_t bytes = source.hidden.GetBytes();
            AssertInFastLLM(bytes <= destination.capacity,
                           "CUDA prefill activation exceeds the allocated slot.");
            if (source.usePeerCopy && FastllmCudaMemcpyPeerAsyncCurrentThread(destination.device, destination.hidden.cudaData,
                    source.device, source.hidden.cudaData, bytes)) return;
            source.usePeerCopy = false;

            // Reuse the existing pinned-host transfer APIs on systems without
            // peer copies. The source event also orders this thread's source-
            // device stream; the producer's compute stream is never modified.
            if (!source.pinned) source.pinned = FastllmCudaHostMalloc(source.capacity);
            CheckCudaError();
            if (!source.pinned) throw std::bad_alloc();
            {
                DeviceScope restore;
                FastllmCudaSetDevice(source.device);
                FastllmCudaCurrentThreadStreamWaitEvent(source.readyEvent);
                if (!FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(
                        source.pinned, source.hidden.cudaData, bytes)) {
                    throw std::runtime_error("CUDA prefill device-to-host copy failed.");
                }
                FastllmCudaSyncCurrentThreadStream();
                CheckCudaError();
            }
            if (!FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
                    destination.hidden.cudaData, source.pinned, bytes)) {
                throw std::runtime_error("CUDA prefill host-to-device copy failed.");
            }
        }

        PersistentWorkerGroup workers;
    };
}
#endif
#endif
