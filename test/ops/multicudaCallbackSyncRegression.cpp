#include "model.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
static void check(bool ok, const char *s) {
    if (!ok)
        throw std::runtime_error(s);
}
static void cu(cudaError_t e) {
    check(e == cudaSuccess, cudaGetErrorString(e));
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 8)
        return 77;
    try {
        SetThreads(16);
        int cases = 0;
        const size_t bytes = 32 * 1024 * 1024;
        for (int count : {2, 4, 8})
            for (int reversed : {0, 1}) {
                std::vector<int> devices;
                for (int i = 0; i < count; ++i)
                    devices.push_back(reversed ? count - 1 - i : i);
                std::vector<void *> data(count);
                std::vector<cudaEvent_t> ends(count);
                for (int i = 0; i < count; ++i) {
                    cu(cudaSetDevice(devices[i]));
                    cu(cudaMalloc(&data[i], bytes));
                    cu(cudaEventCreateWithFlags(&ends[i], cudaEventDisableTiming));
                }
                bool previous = MultiCudaSetPersistentAsyncDispatch(true);
                for (int round = 0; round < 30; ++round) {
                    check(
                        MultiCudaRunDeviceCallbacks(devices,
                                                    [&](int rank, int) {
                                                        for (int j = 0; j < 16; ++j)
                                                            cu(cudaMemsetAsync(data[rank], round + rank,
                                                                               bytes, cudaStreamPerThread));
                                                        cu(cudaEventRecord(ends[rank], cudaStreamPerThread));
                                                    }),
                        "submit");
                    check(MultiCudaSynchronizeDeviceCallbacks(devices), "join");
                    for (int i = 0; i < count; ++i) {
                        cu(cudaSetDevice(devices[i]));
                        check(cudaEventQuery(ends[i]) == cudaSuccess, "join returned before completion");
                        unsigned char got = 0;
                        cu(cudaMemcpy(&got, (char *)data[i] + bytes - 1, 1, cudaMemcpyDeviceToHost));
                        check(got == (unsigned char)(round + i), "data incomplete");
                    }
                    ++cases;
                }
                MultiCudaSetPersistentAsyncDispatch(previous);
                for (int i = 0; i < count; ++i) {
                    cu(cudaSetDevice(devices[i]));
                    cu(cudaEventDestroy(ends[i]));
                    cu(cudaFree(data[i]));
                }
            }
        check(!MultiCudaSynchronizeDeviceCallbacks({}), "empty device set");
        check(!MultiCudaSynchronizeDeviceCallbacks({9999}), "unknown worker");
        std::cout << "PASS " << cases
                  << " completion joins, TP2/4/8, reversed devices, delayed writes; no verification sync "
                     "before event query"
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
