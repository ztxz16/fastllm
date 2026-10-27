#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

bool SupportsCutlassFp8(int device) {
    cudaDeviceProp properties;
    Check(cudaGetDeviceProperties(&properties, device));
    const int arch = properties.major * 10 + properties.minor;
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120)
    if (arch == 120) return true;
#endif
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121)
    if (arch == 121) return true;
#endif
    (void)arch;
    return false;
}

struct Case {
    static constexpr int rows = 128, dim = 128;
    const float weightValue;
    fastllm::Data input{fastllm::FLOAT16, {rows, dim}};
    fastllm::Data warmWeight{fastllm::FP8_E4M3, {dim, dim}};
    fastllm::Data coldWeight{fastllm::FP8_E4M3, {dim, dim}};
    fastllm::Data output{fastllm::FLOAT16, {rows, dim}};
    fastllm::Data bias;

    explicit Case(int device) : weightValue(float(device + 1)) {
        FastllmCudaSetDevice(device);
        input.Allocate(1.0f);
        input.ToDevice(fastllm::CUDA, std::vector<int>{device});
        output.ToDevice(fastllm::CUDA, {device}, false);
        output.Allocate();
        for (auto *weight : {&warmWeight, &coldWeight}) {
            weight->blockM = weight->blockK = 128;
            weight->scales = {weight == &warmWeight ? 1.0f : 2.0f};
            weight->Allocate();
            // E4M3 encodes 1 as 0x38 and 2 as 0x40. Different devices and
            // warm/cold scales also catch accidental reuse of another entry.
            std::memset(weight->cpuData, device == 0 ? 0x38 : 0x40, weight->GetBytes());
            weight->ToDevice(fastllm::CUDA, std::vector<int>{device});
        }
    }

    bool Run(fastllm::Data &weight) {
        return FastllmCudaCutlassLinearFP8E4M3Block128(
            input, weight, bias, output, rows, dim, dim);
    }

    void Verify(const fastllm::Data &weight) {
        std::vector<half> values(rows * dim);
        Check(cudaMemcpy(values.data(), output.cudaData, values.size() * sizeof(half),
                         cudaMemcpyDeviceToHost));
        const float expected = dim * weightValue * weight.scales[0];
        for (half value : values) {
            float actual = __half2float(value);
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.125f) {
                throw std::runtime_error("FP8 cache regression produced incorrect output");
            }
        }
    }
};

struct Gate {
    std::atomic<bool> entered{false}, release{false}, timedOut{false};
};

void CUDART_CB WaitForPeer(void *argument) {
    auto &gate = *static_cast<Gate*>(argument);
    gate.entered.store(true);
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    // Model a queued collective waiting for another GPU's host submission.
    // The deadline releases the stream even on the broken implementation,
    // so the regression fails cleanly instead of leaving a GPU hung.
    while (!gate.release.load()) {
        if (Clock::now() >= deadline) {
            gate.timedOut.store(true);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) return 77;
    try {
        for (int device = 0; device < 2; ++device) {
            if (!SupportsCutlassFp8(device)) {
                std::printf("SKIP: device %d has no compiled CUTLASS FP8 backend\n", device);
                return 77;
            }
        }
        std::unique_ptr<Case> cases[2];
        for (int device = 0; device < 2; ++device) {
            cases[device] = std::make_unique<Case>(device);
            // Initialize scratch and kernel state, leaving coldWeight uncached.
            if (!cases[device]->Run(cases[device]->warmWeight)) {
                throw std::runtime_error("CUTLASS FP8 warmup failed on a supported device");
            }
            cases[device]->Verify(cases[device]->warmWeight);
        }
        Gate gate;
        std::exception_ptr errors[2];
        auto start = Clock::now();
        std::thread first([&] {
            try {
                FastllmCudaSetDevice(0);
                Check(cudaLaunchHostFunc(cudaStreamPerThread, WaitForPeer, &gate));
                if (!cases[0]->Run(cases[0]->coldWeight)) {
                    throw std::runtime_error("GPU 0 cold CUTLASS cache rejected");
                }
                cases[0]->Verify(cases[0]->coldWeight);
            } catch (...) {
                errors[0] = std::current_exception();
                gate.release.store(true);
                gate.entered.store(true);
            }
            // Drain the callback before Gate can be destroyed, including
            // when a CUDA call after cudaLaunchHostFunc has failed.
            cudaError_t status = cudaStreamSynchronize(cudaStreamPerThread);
            if (status != cudaSuccess && !errors[0]) {
                errors[0] = std::make_exception_ptr(
                    std::runtime_error(cudaGetErrorString(status)));
            }
        });
        std::thread second([&] {
            try {
                FastllmCudaSetDevice(1);
                while (!gate.entered.load()) std::this_thread::yield();
                // Let GPU 0 enter the scale upload/synchronization before
                // its peer requests a different device's cold cache entry.
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (!cases[1]->Run(cases[1]->coldWeight)) {
                    throw std::runtime_error("GPU 1 cold CUTLASS cache rejected");
                }
                cases[1]->Verify(cases[1]->coldWeight);
            } catch (...) {
                errors[1] = std::current_exception();
            }
            gate.release.store(true);
        });
        first.join();
        second.join();
        for (auto &error : errors) if (error) std::rethrow_exception(error);
        if (gate.timedOut.load()) {
            throw std::runtime_error("cross-device FP8 cache lock blocked the peer for 10 seconds");
        }
        std::printf("FP8 cold cache peer progress and numeric output PASS (%.3f s)\n",
                    std::chrono::duration<double>(Clock::now() - start).count());
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
