#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

namespace {
void CheckCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << cudaGetErrorString(result) << '\n';
        // A failed rank cannot leave its peer waiting inside a collective.
        std::exit(2);
    }
}

template <typename T> T FromFloat(float value) { return (T)value; }
template <> half FromFloat<half>(float value) { return __float2half_rn(value); }
template <> __nv_bfloat16 FromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}
template <typename T> float ToFloat(T value) { return (float)value; }
template <> float ToFloat<half>(half value) { return __half2float(value); }
template <> float ToFloat<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T> bool Run(const std::vector<int> &devices, int dataType, int count) {
    std::atomic<int> errors{0};
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 2; ++rank) {
        workers.emplace_back([&, rank]() {
            const int device = devices[rank];
            CheckCuda(cudaSetDevice(device));
            std::vector<T> input(count), output(count);
            void *send = nullptr, *recv = nullptr;
            const size_t bytes = count * sizeof(T);
            CheckCuda(cudaMalloc(&send, bytes));
            CheckCuda(cudaMalloc(&recv, bytes));
            // Alternate generations, roots, and in-place/out-of-place calls.
            for (int iteration = 0; iteration < 24; ++iteration) {
                auto value = [&](int r, int i) {
                    return FromFloat<T>((float)((i * 13 + r * 31 + iteration * 17) % 101 - 50));
                };
                for (int i = 0; i < count; ++i)
                    input[i] = value(rank, i);
                CheckCuda(cudaMemcpyAsync(send, input.data(), bytes, cudaMemcpyHostToDevice,
                                          cudaStreamPerThread));
                const bool inPlace = (iteration / 4) % 2 == 0;
                void *destination = inPlace ? send : recv;
                const int operation = iteration % 4;
                const int rootRank = (iteration / 8) % 2;
                if (operation == 0) {
                    FastllmNcclAllReduce(send, destination, count, dataType, device);
                } else if (operation == 1) {
                    FastllmNcclAllReduceNoCustom(send, destination, count, dataType, device);
                } else if (operation == 2) {
                    FastllmNcclBroadcastFrom(send, destination, count, dataType, devices[rootRank],
                                             device);
                } else {
                    FastllmNcclReduce(send, destination, count, dataType, devices[rootRank],
                                      device);
                }
                if (operation == 3 && rank != rootRank)
                    continue;
                CheckCuda(cudaMemcpyAsync(output.data(), destination, bytes, cudaMemcpyDeviceToHost,
                                          cudaStreamPerThread));
                CheckCuda(cudaStreamSynchronize(cudaStreamPerThread));
                for (int i = 0; i < count; ++i) {
                    const T expected =
                        operation == 2 ? value(rootRank, i)
                                       : FromFloat<T>(ToFloat(value(0, i)) + ToFloat(value(1, i)));
                    if (ToFloat(output[i]) != ToFloat(expected)) {
                        if (errors.fetch_add(1) == 0) {
                            std::cerr << "Mismatch: type=" << dataType << " count=" << count
                                      << " rank=" << rank << " iteration=" << iteration
                                      << " index=" << i << " expected=" << ToFloat(expected)
                                      << " actual=" << ToFloat(output[i]) << '\n';
                        }
                    }
                }
            }
            CheckCuda(cudaFree(send));
            CheckCuda(cudaFree(recv));
        });
    }
    for (auto &worker : workers)
        worker.join();
    return errors.load() == 0;
}
} // namespace

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 2) {
        std::cout << "SKIP: host collective regression requires two CUDA GPUs\n";
        return 77;
    }
    // Force host staging even on development machines with CUDA peer access.
#ifdef _WIN32
    _putenv_s("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "0");
#else
    setenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "0", 1);
#endif
    bool ok = true;
    for (int group = 0; group < 2; ++group) {
        // New threaded TP runners initialize their communicator independently
        // of the legacy MultiCuda operator. Neither an empty nor a stale
        // single-device operator list may replace this active group.
        FastllmMultiCudaSetDevice(group == 0 ? std::vector<int>{} : std::vector<int>{0});
        const std::vector<int> devices =
            group == 0 ? std::vector<int>{0, 1} : std::vector<int>{1, 0};
        if (!FastllmInitNccl(devices))
            return 2;
        for (int count : {1, 513, 5120, 112640}) {
            ok &= Run<half>(devices, fastllm::FLOAT16, count);
            ok &= Run<__nv_bfloat16>(devices, fastllm::BFLOAT16, count);
            ok &= Run<float>(devices, fastllm::FLOAT32, count);
            ok &= Run<int8_t>(devices, fastllm::INT8, count);
            ok &= Run<int32_t>(devices, fastllm::INT32, count);
        }
    }
    std::cout << "host collective initialized-group regression: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
