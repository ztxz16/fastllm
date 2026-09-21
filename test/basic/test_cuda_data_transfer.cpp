#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>

#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void ExpectBytes(const uint8_t *data, size_t size, uint8_t value) {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] != value) throw std::runtime_error("unexpected transferred byte");
    }
}

// Prefill grows a reusable output; decode then shrinks its logical shape.
// Different host/device tails make copying unused capacity observable without
// imposing a machine-dependent timing threshold on the regression test.
void CheckScratchTransfer(fastllm::DataType type, int devices) {
    fastllm::Data data(type, {256, 2560});
    data.Allocate(false);
    const size_t capacity = data.expansionBytes;
    std::memset(data.cpuData, 0x11, capacity);
    data.ToDevice(fastllm::CUDA, {0}, true);
    void *gpu = data.cudaData;
    auto *cpu = data.cpuData;
    data.ToDevice(fastllm::CPU, false);
    data.Resize({1, 2560});
    const size_t bytes = data.GetBytes();
    std::memset(data.cpuData, 0x22, capacity);
    data.ToDevice(fastllm::CUDA, {0}, true);
    if (data.expansionBytes != capacity || data.cpuData != cpu || data.cudaData != gpu) {
        throw std::runtime_error("shrinking scratch tensor reallocated storage");
    }
    std::vector<uint8_t> actual(capacity);
    Check(cudaMemcpy(actual.data(), data.cudaData, capacity, cudaMemcpyDeviceToHost));
    ExpectBytes(actual.data(), bytes, 0x22);
    ExpectBytes(actual.data() + bytes, capacity - bytes, 0x11);

    Check(cudaMemset(data.cudaData, 0x33, capacity));
    Check(cudaDeviceSynchronize());
    data.ToDevice(fastllm::CPU, true);
    ExpectBytes(data.cpuData, bytes, 0x33);
    ExpectBytes(data.cpuData + bytes, capacity - bytes, 0x22);

    // Growing again must retain the allocation and permit a full transfer.
    data.Resize({256, 2560});
    data.ToDevice(fastllm::CUDA, {0}, true);
    Check(cudaMemcpy(actual.data(), data.cudaData, capacity, cudaMemcpyDeviceToHost));
    ExpectBytes(actual.data(), bytes, 0x33);
    ExpectBytes(actual.data() + bytes, capacity - bytes, 0x22);
    if (devices > 1) {
        data.Resize({1, 2560});
        data.ToDevice(fastllm::CUDA, {1}, true);
        Check(cudaMemcpy(actual.data(), data.cudaData, bytes, cudaMemcpyDeviceToHost));
        ExpectBytes(actual.data(), bytes, 0x33);
    }
}

void CheckHostHelpers(fastllm::DataType type) {
    fastllm::Data data(type, {256, 2560});
    data.Allocate(false);
    const size_t capacity = data.expansionBytes;
    std::memset(data.cpuData, 0x55, capacity);
    void *gpu = FastllmCudaPrepareInput(data);
    if (!gpu) throw std::runtime_error("temporary input allocation failed");

    // An in-place operation can return a smaller view of its temporary input.
    data.Resize({1, 2560});
    const size_t bytes = data.GetBytes();
    std::memset(data.cpuData, 0x22, capacity);
    FastllmCudaFinishOutput(data, gpu);
    ExpectBytes(data.cpuData, bytes, 0x55);
    ExpectBytes(data.cpuData + bytes, capacity - bytes, 0x22);

    gpu = FastllmCudaPrepareInput(data);
    if (!gpu) throw std::runtime_error("small temporary input allocation failed");
    std::vector<uint8_t> actual(bytes);
    Check(cudaMemcpy(actual.data(), gpu, bytes, cudaMemcpyDeviceToHost));
    ExpectBytes(actual.data(), bytes, 0x55);
    FastllmCudaFinishInput(data, gpu);
    gpu = FastllmCudaPrepareOutput(data);
    if (!gpu) throw std::runtime_error("small temporary output allocation failed");
    Check(cudaMemset(gpu, 0x77, bytes));
    FastllmCudaFinishOutput(data, gpu);
    ExpectBytes(data.cpuData, bytes, 0x77);
    ExpectBytes(data.cpuData + bytes, capacity - bytes, 0x22);

    // CUDA-resident tensors must still bypass host staging entirely.
    data.ToDevice(fastllm::CUDA, {0}, true);
    gpu = FastllmCudaPrepareInput(data);
    if (gpu != data.cudaData || FastllmCudaPrepareOutput(data) != data.cudaData) {
        throw std::runtime_error("CUDA-resident tensor was staged again");
    }
    FastllmCudaFinishInput(data, gpu);
    FastllmCudaFinishOutput(data, gpu);
    Check(cudaMemcpy(actual.data(), gpu, bytes, cudaMemcpyDeviceToHost));
    ExpectBytes(actual.data(), bytes, 0x77);
}

void CheckHostRMSNorm(fastllm::DataType type) {
    constexpr int columns = 128;
    fastllm::Data input(type, {256, columns}), output(type, {256, columns});
    fastllm::Data weight(fastllm::FLOAT32, {columns});
    input.Allocate(1.0f);
    output.Allocate(false);
    weight.Allocate(1.0f);
    weight.ToDevice(fastllm::CUDA, {0}, true);
    for (int rows : {256, 1, 8, 1}) {
        input.Resize({rows, columns});
        output.Resize({rows, columns});
        if (!FastllmCudaRMSNorm(input, weight, output, 0.0f)) {
            throw std::runtime_error("host-staged RMSNorm failed");
        }
        for (int i = 0; i < rows * columns; ++i) {
            if (type == fastllm::FLOAT32) {
                if (std::fabs(((float*)output.cpuData)[i] - 1.f) > 1e-5f) {
                    throw std::runtime_error("FP32 RMSNorm result changed");
                }
            } else {
                uint16_t one = type == fastllm::FLOAT16 ? 0x3c00 : 0x3f80;
                if (((uint16_t*)output.cpuData)[i] != one) {
                    throw std::runtime_error("16-bit RMSNorm result changed");
                }
            }
        }
    }
}

void CheckScratchZeroFill(fastllm::DataType type) {
    fastllm::Data data(type, {256, 2560});
    data.ToDevice(fastllm::CUDA, {0}, false);
    data.Allocate(false);
    const size_t capacity = data.expansionBytes;
    Check(cudaMemset(data.cudaData, 0x55, capacity));
    data.Resize({1, 2560});
    data.Allocate(0.0f);
    std::vector<uint8_t> actual(capacity);
    Check(cudaMemcpy(actual.data(), data.cudaData, capacity, cudaMemcpyDeviceToHost));
    ExpectBytes(actual.data(), data.GetBytes(), 0);
    ExpectBytes(actual.data() + data.GetBytes(), capacity - data.GetBytes(), 0x55);
}

// Explicit expansion and persistent weights/KV caches keep their full-storage
// transfer semantics, including bytes outside the current logical shape.
void CheckPersistentTransfer(int kind, int devices) {
    Check(cudaSetDevice(0));
    fastllm::Data data(fastllm::BFLOAT16, {1, 2560});
    if (kind == 0) {
        data.Expansion({256, 2560});
    } else {
        data.Resize({256, 2560});
        data.Allocate(false);
        data.Resize({1, 2560});
        data.isKVCache = kind == 1;
        data.isModelWeight = kind == 2;
    }
    const size_t capacity = data.expansionBytes;
    std::memset(data.cpuData, 0x55, capacity);
    data.ToDevice(fastllm::CUDA, {0}, true);
    if (devices > 1) data.ToDevice(fastllm::CUDA, {1}, true);
    std::vector<uint8_t> actual(capacity);
    Check(cudaMemcpy(actual.data(), data.cudaData, capacity, cudaMemcpyDeviceToHost));
    ExpectBytes(actual.data(), capacity, 0x55);
    Check(cudaMemset(data.cudaData, 0x66, capacity));
    Check(cudaDeviceSynchronize());
    data.ToDevice(fastllm::CPU, true);
    ExpectBytes(data.cpuData, capacity, 0x66);

    void *gpu = FastllmCudaPrepareInput(data);
    if (!gpu) throw std::runtime_error("persistent temporary input allocation failed");
    Check(cudaMemcpy(actual.data(), gpu, capacity, cudaMemcpyDeviceToHost));
    ExpectBytes(actual.data(), capacity, 0x66);
    FastllmCudaFinishInput(data, gpu);
    gpu = FastllmCudaPrepareOutput(data);
    if (!gpu) throw std::runtime_error("persistent temporary output allocation failed");
    Check(cudaMemset(gpu, 0x77, capacity));
    FastllmCudaFinishOutput(data, gpu);
    ExpectBytes(data.cpuData, capacity, 0x77);
    data.ToDevice(fastllm::CUDA, {0}, true);
    data.Allocate(0.0f);
    Check(cudaMemcpy(actual.data(), data.cudaData, capacity, cudaMemcpyDeviceToHost));
    ExpectBytes(actual.data(), capacity, 0);
}

void CheckLargePeerTransfer(int devices) {
    if (devices < 2) return;
    // Exercise multiple staging chunks, an odd tail, buffer reuse, and both
    // directions. A second host thread consumes the completed destination.
    const size_t capacity = (65ULL << 20) + 17;
    void *buffers[2] = {nullptr, nullptr};
    for (int device = 0; device < 2; ++device) {
        Check(cudaSetDevice(device));
        Check(cudaMalloc(&buffers[device], capacity));
    }
    for (int pass = 0; pass < 4; ++pass) {
        const int src = pass % 2, dst = 1 - src;
        const size_t bytes = pass < 2 ? capacity : 65539;
        Check(cudaSetDevice(src));
        Check(cudaMemsetAsync(buffers[src], 0x31 + pass, bytes, cudaStreamPerThread));
        FastllmCudaMemcpyBetweenDevices(dst, buffers[dst], src, buffers[src], bytes);
        int restored = -1; Check(cudaGetDevice(&restored));
        if (restored != src) throw std::runtime_error("peer copy changed current device");
        std::exception_ptr error;
        std::thread reader([&]() {
            try {
                Check(cudaSetDevice(dst));
                std::vector<uint8_t> actual(bytes);
                Check(cudaMemcpy(actual.data(), buffers[dst], bytes, cudaMemcpyDeviceToHost));
                ExpectBytes(actual.data(), bytes, 0x31 + pass);
            } catch (...) { error = std::current_exception(); }
        });
        reader.join();
        if (error) std::rethrow_exception(error);
    }
    for (int device = 0; device < 2; ++device) {
        Check(cudaSetDevice(device)); Check(cudaFree(buffers[device]));
    }
    std::puts("PASS large peer transfer, tail, reuse and cross-thread handoff");
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        for (auto type : {fastllm::FLOAT16, fastllm::BFLOAT16, fastllm::FLOAT32,
                          fastllm::FP8_E4M3, fastllm::INT8, fastllm::INT4,
                          fastllm::INT32}) {
            Check(cudaSetDevice(0));
            CheckScratchTransfer(type, devices);
            Check(cudaSetDevice(0));
            CheckHostHelpers(type);
            std::printf("PASS transfer dtype=%s\n", fastllm::GetDataTypeName(type).c_str());
        }
        for (auto type : {fastllm::FLOAT16, fastllm::BFLOAT16, fastllm::FLOAT32}) {
            CheckHostRMSNorm(type);
            CheckScratchZeroFill(type);
            std::printf("PASS host RMSNorm and scratch zero fill dtype=%s\n",
                        fastllm::GetDataTypeName(type).c_str());
        }
        for (int kind = 0; kind < 3; ++kind) CheckPersistentTransfer(kind, devices);
        CheckLargePeerTransfer(devices);
        std::puts("CUDA scratch, host staging, RMSNorm, zero fill and persistent transfer tests passed");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "CUDA data transfer regression: %s\n", error.what());
        return 1;
    }
}
