#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace fastllm;
static void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> __device__ T Value(size_t row, int epoch) {
    return T((row * 1664525u + epoch * 1013904223u) ^ (row >> 9));
}
template<class T> __global__ void Fill(T *data, int offset, int count, int epoch) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        data[offset + i] = Value<T>(offset + i, epoch);
}
template<class T> __global__ void Verify(const T *data, int count, int epoch, unsigned *errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        if (data[i] != Value<T>(i, epoch)) atomicAdd(errors, 1u);
}
__global__ void Delay() {
    auto start = clock64();
    while (clock64() - start < 100000) {}
}
template<class T> static void Test(const std::vector<int> &devices, int type) {
    const int ranks = devices.size();
    for (int pattern = 0; pattern < 4; ++pattern) {
        std::vector<int> counts(ranks), offsets(ranks);
        for (int r = 0; r < ranks; ++r) {
            // Uneven, strongly skewed, multiple empty sources, and equal.
            counts[r] = pattern == 0 ? 31 + r * 137 : pattern == 1 ? (r == 0 ? 65537 : 1) :
                pattern == 2 ? (r == ranks - 1 ? 4099 : 0) : 1024;
            if (r) offsets[r] = offsets[r - 1] + counts[r - 1];
        }
        const int total = offsets.back() + counts.back();
        std::vector<T *> data(ranks);
        std::vector<unsigned *> errors(ranks);
        if (!MultiCudaRunDeviceCallbacks(devices, [&](int r, int device) {
            FastllmCudaSetDevice(device);
            Check(cudaMalloc(&data[r], (size_t(total) + 2) * sizeof(T)));
            Check(cudaMalloc(&errors[r], sizeof(unsigned)));
            Check(cudaMemsetAsync(data[r], 0xa5, (size_t(total) + 2) * sizeof(T), cudaStreamPerThread));
            Check(cudaMemsetAsync(errors[r], 0, sizeof(unsigned), cudaStreamPerThread));
        })) throw std::runtime_error("collective workers unavailable");
        if (!MultiCudaRunDeviceCallbacks(devices, [&](int r, int device) {
            for (int epoch = 0; epoch < 19; ++epoch) {
                if (r == epoch % ranks) Delay<<<1, 1, 0, cudaStreamPerThread>>>();
                Fill<<<32, 128, 0, cudaStreamPerThread>>>(data[r] + 1, offsets[r], counts[r], epoch);
                if (!FastllmNcclAllGatherVInplace(data[r] + 1, counts, type, device))
                    throw std::runtime_error("AllGatherV submission failed");
                Verify<<<32, 128, 0, cudaStreamPerThread>>>(data[r] + 1, total, epoch, errors[r]);
            }
            Check(cudaGetLastError());
        }) || !MultiCudaSynchronizeDeviceCallbacks(devices))
            throw std::runtime_error("collective execution failed");
        for (int r = 0; r < ranks; ++r) {
            FastllmCudaSetDevice(devices[r]);
            unsigned count = 0;
            T guards[2];
            Check(cudaMemcpy(&count, errors[r], sizeof(count), cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(guards, data[r], sizeof(T), cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(guards + 1, data[r] + total + 1, sizeof(T), cudaMemcpyDeviceToHost));
            Check(cudaFree(data[r])); Check(cudaFree(errors[r]));
            const auto *bytes = reinterpret_cast<const uint8_t *>(guards);
            if (count || std::any_of(bytes, bytes + sizeof(guards), [](uint8_t v) { return v != 0xa5; }))
                throw std::runtime_error("AllGatherV data, ordering, or guard mismatch");
        }
    }
}
int main(int argc, char **argv) try {
    const int ranks = argc > 1 ? std::stoi(argv[1]) : 2;
    if (ranks < 2) throw std::invalid_argument("requires at least two ranks");
    int available = 0;
    if (cudaGetDeviceCount(&available) != cudaSuccess || available < ranks) return 77;
    std::vector<int> devices(ranks);
    std::iota(devices.begin(), devices.end(), 0);
    SetThreads(ranks); SetCudaGraph(false);
    if (!FastllmInitNccl(devices)) throw std::runtime_error("NCCL init failed");
    FastllmCudaSetNcclForceSync(false);
    Test<uint16_t>(devices, FLOAT16);
    Test<uint16_t>(devices, BFLOAT16);
    Test<uint32_t>(devices, INT32);
    Test<uint32_t>(devices, FLOAT32);
    Test<uint8_t>(devices, INT8);
    std::cout << "ALLGATHERV PASS" << std::endl;
    return 0;
} catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
