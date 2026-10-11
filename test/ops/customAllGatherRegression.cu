#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static void Check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::abort();
    }
}

__host__ __device__ static unsigned short Value(int rank, int offset) {
    return 0x3f00 + ((rank * 41 + offset * 17) & 255);
}

__global__ void CheckGather(const unsigned short *data, int part, int ranks, int *errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < part * ranks; i += blockDim.x * gridDim.x)
        if (data[i] != Value(i / part, i % part)) atomicAdd(errors, 1);
}

int main(int argc, char **argv) {
    const int ranks = argc > 1 ? std::atoi(argv[1]) : 8;
    if (ranks != 2 && ranks != 4 && ranks != 6 && ranks != 8) return 2;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < ranks) {
        std::printf("SKIP: requires %d CUDA GPUs\n", ranks);
        return 77;
    }
    for (int rank = 0; rank < ranks; ++rank) {
        for (int peer = 0; peer < ranks; ++peer) {
            if (peer == rank) continue;
            int canAccess = 0;
            Check(cudaDeviceCanAccessPeer(&canAccess, rank, peer));
            if (!canAccess) {
                std::printf("SKIP: GPU %d cannot access GPU %d\n", rank, peer);
                return 77;
            }
        }
    }
    setenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "1", 1);
    std::vector<int> devices;
    for (int i = 0; i < ranks; ++i) devices.push_back(i);
    if (!FastllmInitNccl(devices) || !FastllmCudaCustomAllReduceInit(devices)) return 2;
    std::vector<std::thread> workers;
    std::vector<int> errors(ranks);
    for (int rank = 0; rank < ranks; ++rank) workers.emplace_back([&, rank] {
        Check(cudaSetDevice(rank));
        unsigned short *data, *host;
        int *deviceErrors;
        constexpr int maxPart = 16 * 4096;
        Check(cudaMalloc(&data, (size_t)ranks * maxPart * sizeof(*data)));
        Check(cudaMallocHost(&host, maxPart * sizeof(*host)));
        Check(cudaMalloc(&deviceErrors, sizeof(int)));
        Check(cudaMemsetAsync(deviceErrors, 0, sizeof(int), cudaStreamPerThread));
        for (int i = 0; i < maxPart; ++i) host[i] = Value(rank, i);
        for (int rows : {1, 2, 4, 8, 16}) {
            const int part = rows * 4096, count = ranks * part;
            auto run = [&] {
                // Poison all foreign slices on every replay. A missing start
                // or completion barrier must not silently reuse prior output.
                Check(cudaMemsetAsync(data, 0xa5, (size_t)count * sizeof(*data), cudaStreamPerThread));
                Check(cudaMemcpyAsync(data + rank * part, host, part * sizeof(*host),
                                      cudaMemcpyHostToDevice, cudaStreamPerThread));
                if (!FastllmCudaCustomAllGatherRows(data, count, 4096, fastllm::BFLOAT16, rank)) std::abort();
                CheckGather<<<32, 256, 0, cudaStreamPerThread>>>(data, part, ranks, deviceErrors);
                Check(cudaGetLastError());
                // Interleave the established reduction to exercise the shared
                // signal counters with differing collective grid sizes.
                if (!FastllmCudaCustomAllReduceRows(data, data, count, 4096, fastllm::BFLOAT16, rank)) std::abort();
            };
            run();
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            run();
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&exec, graph, 0));
            for (int repeat = 0; repeat < 64; ++repeat) {
                if ((repeat + rank) % 7 == 0)
                    std::this_thread::sleep_for(std::chrono::microseconds(137 * (rank + 1)));
                Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            }
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaGraphExecDestroy(exec));
            Check(cudaGraphDestroy(graph));
        }
        Check(cudaMemcpy(&errors[rank], deviceErrors, sizeof(int), cudaMemcpyDeviceToHost));
        Check(cudaFree(deviceErrors));
        Check(cudaFreeHost(host));
        Check(cudaFree(data));
    });
    for (auto &worker : workers) worker.join();
    FastllmCudaCustomAllReduceReset();
    for (int error : errors) if (error) return 1;
    std::printf("Custom AllGather: %d GPUs, five shard sizes, eager +64 skewed graph replays PASS\n", ranks);
    return 0;
}
