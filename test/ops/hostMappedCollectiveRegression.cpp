#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

namespace {
void Check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << cudaGetErrorString(e) << '\n';
        std::exit(2);
    }
}
void Require(bool ok, const char *message) {
    if (!ok) {
        std::cerr << message << '\n';
        std::exit(3);
    }
}
template <typename T> T FromFloat(float x) { return (T)x; }
template <> half FromFloat(float x) { return __float2half_rn(x); }
template <> __nv_bfloat16 FromFloat(float x) { return __float2bfloat16_rn(x); }
template <typename T> float ToFloat(T x) { return (float)x; }
template <> float ToFloat(half x) { return __half2float(x); }
template <> float ToFloat(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
void Run(const std::vector<int> &devices, int dtype, int count, int root, bool inPlace, bool varySizes = false, int offset = 0) {
    T *hostIn[2], *hostOut[2], *send[2], *recv[2];
    size_t bytes = (size_t)count * sizeof(T);
    auto operationCount = [&](int op) {
        if (!varySizes) return count;
        // Alternate 4, 1, 3, and 2 active partitions inside one graph. A
        // partition's sequence must resume correctly after it was skipped.
        const int counts[] = {count, 1, (int)(81920 / sizeof(T)) + 1,
                              (int)(32768 / sizeof(T)) + 1};
        return counts[op];
    };
    // No allocation may synchronize a device while its peer graph is waiting.
    for (int r = 0; r < 2; ++r) {
        Check(cudaSetDevice(devices[r]));
        Check(cudaMallocHost((void **)&hostIn[r], bytes));
        Check(cudaMallocHost((void **)&hostOut[r], bytes * 4));
        Check(cudaMalloc((void **)&send[r], bytes + offset * sizeof(T)));
        Check(cudaMalloc((void **)&recv[r], bytes + offset * sizeof(T)));
        send[r] += offset;
        recv[r] += offset;
        Check(cudaDeviceSynchronize());
    }
    std::atomic<int> errors{0};
    auto work = [&](int rank) {
        Check(cudaSetDevice(devices[rank]));
        void *graph = nullptr, *exec = nullptr;
        // Recapture on the same transport without resetting its GPU counters.
        for (int capture = 0; capture < 2; ++capture) {
            Require(FastllmCudaGraphBeginCapture(), "begin capture failed");
            for (int op = 0; op < 4; ++op) {
                const int elements = operationCount(op);
                Check(cudaMemcpyAsync(send[rank], hostIn[rank], bytes, cudaMemcpyHostToDevice,
                                      cudaStreamPerThread));
                T *destination = inPlace ? send[rank] : recv[rank];
                if (op == 0)
                    FastllmNcclAllReduce(send[rank], destination, elements, dtype, devices[rank]);
                else if (op == 1)
                    FastllmNcclAllReduceNoCustom(send[rank], destination, elements, dtype,
                                                 devices[rank]);
                else if (op == 2)
                    FastllmNcclBroadcastFrom(send[rank], destination, elements, dtype, devices[root],
                                             devices[rank]);
                else
                    FastllmNcclReduce(send[rank], destination, elements, dtype, devices[root],
                                      devices[rank]);
                if (op != 3 || rank == root)
                    Check(cudaMemcpyAsync(hostOut[rank] + op * count, destination, bytes,
                                          cudaMemcpyDeviceToHost, cudaStreamPerThread));
            }
            Require(!FastllmCudaGetThreadError(), "collective capture reported an error");
            Require(FastllmCudaGraphEndCapture(&graph), "end capture failed");
            Require(FastllmCudaGraphInstantiate(graph, &exec), "instantiate failed");
            for (int iteration = 0; iteration < 32; ++iteration) {
                auto value = [&](int r, int i) {
                    float v = (float)((i * 13 + r * 31 + iteration * 17 + capture * 7) % 101 - 50);
                    if (iteration & 1)
                        v = v * 0.3125f + ((i + iteration) % 3) * 0.001f;
                    return FromFloat<T>(v);
                };
                for (int i = 0; i < count; ++i)
                    hostIn[rank][i] = value(rank, i);
                if (iteration == 3 && rank == root)
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                Require(FastllmCudaGraphLaunch(exec), "launch failed");
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                for (int op = 0; op < 4; ++op) {
                    if (op == 3 && rank != root)
                        continue;
                    for (int i = 0; i < operationCount(op); ++i) {
                        T expected =
                            op == 2 ? value(root, i)
                                    : FromFloat<T>(ToFloat(value(0, i)) + ToFloat(value(1, i)));
                        if (ToFloat(hostOut[rank][op * count + i]) != ToFloat(expected)) {
                            if (errors.fetch_add(1) == 0)
                                std::cerr << "Mismatch dtype=" << dtype << " count=" << count
                                          << " rank=" << rank << " root=" << root
                                          << " inPlace=" << inPlace << " offset=" << offset
                                          << " varying=" << varySizes << " capture=" << capture
                                          << " iteration=" << iteration << " op=" << op
                                          << " index=" << i
                                          << " actual=" << ToFloat(hostOut[rank][op * count + i])
                                          << " expected=" << ToFloat(expected) << '\n';
                            break;
                        }
                    }
                }
            }
            FastllmCudaGraphExecDestroy(exec);
            exec = nullptr;
            FastllmCudaGraphDestroy(graph);
            graph = nullptr;
        }
    };
    std::thread peer(work, 1);
    work(0);
    peer.join();
    for (int r = 0; r < 2; ++r) {
        Check(cudaSetDevice(devices[r]));
        Check(cudaFree(send[r] - offset));
        Check(cudaFree(recv[r] - offset));
        Check(cudaFreeHost(hostIn[r]));
        Check(cudaFreeHost(hostOut[r]));
    }
    Require(errors == 0, "mapped collective result mismatch");
}

void Unsupported(const std::vector<int> &devices) {
    constexpr int count = 128 * 1024 + 1;
    void *buffers[2];
    for (int r = 0; r < 2; ++r) {
        Check(cudaSetDevice(devices[r]));
        Check(cudaMalloc(&buffers[r], count));
    }
    auto work = [&](int rank) {
        Check(cudaSetDevice(devices[rank]));
        FastllmCudaClearThreadError();
        Require(FastllmCudaGraphBeginCapture(), "oversize capture begin failed");
        FastllmNcclAllReduce(buffers[rank], buffers[rank], count, fastllm::DataType::INT8,
                             devices[rank]);
        Require(FastllmCudaGetThreadError(), "oversize graph collective was not rejected");
        void *graph = nullptr;
        FastllmCudaGraphEndCapture(&graph);
        FastllmCudaGraphDestroy(graph);
        FastllmCudaClearThreadError();
        FastllmCudaClearGraphError();
    };
    std::thread peer(work, 1);
    work(0);
    peer.join();
    for (int r = 0; r < 2; ++r) {
        Check(cudaSetDevice(devices[r]));
        Check(cudaFree(buffers[r]));
    }
}
template <typename T> void Types(const std::vector<int> &devices, int dtype) {
    for (int count : {1, 513, 5120, (int)(65536 / sizeof(T)),
                      (int)(81920 / sizeof(T)), (int)(131072 / sizeof(T))}) {
        for (int root = 0; root < 2; ++root)
            for (bool inPlace : {false, true})
                Run<T>(devices, dtype, count, root, inPlace);
    }
    for (int offset : {0, 1})
        for (int root = 0; root < 2; ++root)
            for (bool inPlace : {false, true})
                Run<T>(devices, dtype, 131072 / sizeof(T) - 1, root, inPlace, true, offset);
    std::cout << "PASS mapped graph type " << dtype << " on " << devices[0] << "," << devices[1]
              << std::endl;
}
} // namespace
int main() {
#if !defined(_WIN32) || defined(FASTLLM_USE_NCCL)
    std::cout << "SKIP: requires Windows CUDA without NCCL\n";
    return 77;
#else
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
        return 77;
    _putenv_s("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "0");
    for (auto devices : {std::vector<int>{0, 1}, std::vector<int>{1, 0}, std::vector<int>{0, 1}}) {
        Require(FastllmInitNccl(devices), "communicator initialization failed");
        Unsupported(devices);
        Types<half>(devices, fastllm::DataType::FLOAT16);
        Types<__nv_bfloat16>(devices, fastllm::DataType::BFLOAT16);
        Types<float>(devices, fastllm::DataType::FLOAT32);
        Types<int8_t>(devices, fastllm::DataType::INT8);
        Types<int32_t>(devices, fastllm::DataType::INT32);
    }
    std::cout << "PASS: mapped-host collective graph regression; 122,880 collective generations per "
                 "rank\n";
    return 0;
#endif
}
