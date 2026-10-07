#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <thread>

namespace fastllm_test {

// Hold down weights until the test observes gate completion. The timeout
// releases the stream on a dependency failure, so the test reports an error
// instead of hanging. Destruction also releases any pending callback.
struct DelayedDownUpload {
    cudaStream_t stream = nullptr;
    cudaEvent_t gateReady = nullptr, downReady = nullptr, gateDone = nullptr, downStart = nullptr;
    uint8_t *host = nullptr;
    std::atomic<bool> release{false}, timedOut{false};

    explicit DelayedDownUpload(size_t bytes = 0) {
        Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (auto *e : {&gateReady, &downReady, &gateDone, &downStart}) Check(cudaEventCreate(e));
        if (bytes) Check(cudaMallocHost(&host, bytes));
    }
    DelayedDownUpload(const DelayedDownUpload &) = delete;
    DelayedDownUpload &operator=(const DelayedDownUpload &) = delete;

    void Hold() {
        release = false; timedOut = false;
        Check(cudaLaunchHostFunc(stream, [](void *opaque) {
            auto &self = *static_cast<DelayedDownUpload *>(opaque);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (!self.release.load()) {
                if (std::chrono::steady_clock::now() > deadline) { self.timedOut = true; break; }
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        }, this));
    }
    ~DelayedDownUpload() {
        release = true;
        if (stream) cudaStreamSynchronize(stream);
        for (auto e : {gateReady, downReady, gateDone, downStart}) if (e) cudaEventDestroy(e);
        if (stream) cudaStreamDestroy(stream);
        cudaFreeHost(host);
    }

private:
    static void Check(cudaError_t error) {
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
};

} // namespace fastllm_test
