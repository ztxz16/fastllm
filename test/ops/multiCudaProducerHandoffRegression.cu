// Build with nvcc --default-stream=per-thread; link libfastllm_tools and cudart.
// Two GPUs required (exit 77 otherwise). No model, NCCL, or peer access needed.
// A synchronous handoff must wait for caller-stream producers on BOTH devices.
// The async cases exercise the existing event handoff as a control.
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void DelayedProducer(int *input, int value, unsigned long long cycles) {
    const auto start = clock64();
    while (clock64() - start < cycles) asm volatile("");
    if (threadIdx.x == 0) *input = value;
}
__global__ void Consumer(const int *input, int *output) {
    if (threadIdx.x == 0) *output = *input;
}

int main() {
    int available = 0;
    if (cudaGetDeviceCount(&available) != cudaSuccess || available < 2) return 77;
    try {
        const std::vector<int> devices{0, 1};
        std::vector<int *> input(2), output(2);
        std::vector<int> clockRate(2);
        for (int device : devices) {
            Check(cudaSetDevice(device));
            Check(cudaMalloc(&input[device], sizeof(int)));
            Check(cudaMalloc(&output[device], sizeof(int)));
            Check(cudaMemset(input[device], 0, sizeof(int)));
            Check(cudaDeviceSynchronize());
            Check(cudaDeviceGetAttribute(&clockRate[device], cudaDevAttrClockRate, device));
        }
        // Initialize workers and kernel modules before queueing a delayed producer.
        const bool originalAsync = MultiCudaSetPersistentAsyncDispatch(false);
        if (!MultiCudaRunDeviceCallbacks(devices, [&](int, int device) {
                Consumer<<<1, 32, 0, cudaStreamPerThread>>>(input[device], output[device]);
                Check(cudaGetLastError());
            })) throw std::runtime_error("worker initialization failed");
        for (int device : devices) {
            Check(cudaSetDevice(device));
            DelayedProducer<<<1, 32, 0, cudaStreamPerThread>>>(input[device], 0, 0);
            Check(cudaDeviceSynchronize());
        }
        int errors = 0, checks = 0;
        for (bool async : {true, false}) for (int current : devices) {
            MultiCudaSetPersistentAsyncDispatch(async);
            const int delayed = 1 - current;
            const int expected = 100 + checks;
            for (int device : devices) {
                Check(cudaSetDevice(device));
                Check(cudaMemset(input[device], 0, sizeof(int)));
                Check(cudaMemset(output[device], 0, sizeof(int)));
                Check(cudaDeviceSynchronize());
                // Every device has a caller-stream producer. The slow producer
                // deliberately lives on a different GPU from the current device.
                DelayedProducer<<<1, 32, 0, cudaStreamPerThread>>>(input[device], expected,
                    device == delayed ? (unsigned long long)clockRate[device] * 50 : 0);
                Check(cudaGetLastError());
            }
            Check(cudaSetDevice(current));
            if (!MultiCudaRunDeviceCallbacks(devices, [&](int, int device) {
                    Consumer<<<1, 32, 0, cudaStreamPerThread>>>(input[device], output[device]);
                    Check(cudaGetLastError());
                })) throw std::runtime_error("consumer dispatch failed");
            int restoredDevice = -1;
            Check(cudaGetDevice(&restoredDevice));
            if (restoredDevice != current) {
                throw std::runtime_error("dispatch changed the caller's current device");
            }
            for (int device : devices) {
                Check(cudaSetDevice(device));
                int actual = -1;
                Check(cudaMemcpy(&actual, output[device], sizeof(int), cudaMemcpyDeviceToHost));
                if (actual != expected) {
                    std::cerr << "FAIL: async=" << async << " current=" << current
                              << " rank=" << device << " expected=" << expected
                              << " actual=" << actual << '\n';
                    ++errors;
                }
            }
            ++checks;
        }
        MultiCudaSetPersistentAsyncDispatch(originalAsync);
        for (int device : devices) {
            Check(cudaSetDevice(device)); Check(cudaDeviceSynchronize());
            Check(cudaFree(input[device])); Check(cudaFree(output[device]));
        }
        if (errors) return 1;
        std::cout << "PASS: " << checks << " MultiCUDA producer handoffs, both current devices, async and sync\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
