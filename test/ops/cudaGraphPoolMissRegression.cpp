#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
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

    std::cout << "CUDA graph pool-miss fallback regression: PASS ("
              << participants << " GPU" << (participants == 1 ? "" : "s")
              << ")\n";
    return 0;
}
