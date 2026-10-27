#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__device__ static float RoutedValue(int rank, int index, int step) {
    unsigned bits = unsigned(index) * 1664525u + unsigned(step) * 1013904223u +
                    unsigned(rank) * 2246822519u;
    bits ^= bits >> 13;
    bits = (bits & 0x807fffffu) | ((80 + (bits >> 16) % 80) << 23);
    switch ((index + step) % 97) {
        case 0: bits = 0; break;
        case 1: bits = 0x80000000u; break;
        case 2: bits = 1; break;
        case 3: bits = 0x7f800000u; break;
        case 4: bits = 0xff800000u; break;
        case 5: bits = 0x7fc00001u; break;
    }
    return __uint_as_float(bits);
}

__global__ void PrepareJoint(float *routed, unsigned short *shared, int count,
                             int rank, int ranks, const int *step) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        routed[i] = RoutedValue(rank, i, *step);
        const int part = count / ranks;
        shared[i] = i / part == rank ? Value(rank, i % part + *step) : 0xa5a5;
    }
}

__global__ void CheckJoint(const float *routed, const unsigned short *shared,
                           int count, int ranks, const int *step, int *errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        float expected = RoutedValue(0, i, *step);
        for (int rank = 1; rank < ranks; ++rank)
            expected = __fadd_rn(expected, RoutedValue(rank, i, *step));
        if (__float_as_uint(expected) != __float_as_uint(routed[i]) &&
            !(isnan(expected) && isnan(routed[i]))) atomicAdd(errors, 1);
        const int part = count / ranks;
        if (shared[i] != Value(i / part, i % part + *step)) atomicAdd(errors, 1);
    }
}

__global__ void AdvanceJoint(int *step) { ++*step; }

static void RunJoint(int rank, int ranks, int &errors) {
    Check(cudaSetDevice(rank));
    const int capacity = ranks * 4096;
    float *input, *output;
    unsigned short *shared;
    int *state;
    Check(cudaMalloc(&input, capacity * sizeof(float)));
    Check(cudaMalloc(&output, capacity * sizeof(float)));
    Check(cudaMalloc(&shared, capacity * sizeof(unsigned short)));
    Check(cudaMalloc(&state, 2 * sizeof(int)));
    Check(cudaMemsetAsync(state, 0, 2 * sizeof(int), cudaStreamPerThread));
    for (int invalid = 0; invalid < 5; ++invalid) {
        void *in = input, *out = output, *gather = shared;
        int count = capacity;
        if (invalid == 0 && rank == ranks - 1) in = nullptr;
        if (invalid == 1 && rank == ranks - 1) gather = shared + 1;
        if (invalid == 2 && rank == 0) out = input + 4;
        if (invalid == 3) count += 4;
        if (invalid == 4) count = 4 * 512 * 36 + 64;
        if (FastllmCudaCustomAllReduceAndGather(in, out, gather, count, rank)) std::abort();
    }
    // TP8 crosses the 128-thread / 36-block dispatch boundary.
    for (int part : {256, 640, 2304, 2312, 4096}) {
        const int count = ranks * part;
        for (bool inplace : {false, true}) {
            float *destination = inplace ? input : output;
            auto run = [&] {
                PrepareJoint<<<16, 256, 0, cudaStreamPerThread>>>(input, shared, count, rank, ranks, state);
                if (!FastllmCudaCustomAllReduceAndGather(input, destination, shared, count, rank)) std::abort();
                CheckJoint<<<16, 256, 0, cudaStreamPerThread>>>(destination, shared, count, ranks, state, state + 1);
                AdvanceJoint<<<1, 1, 0, cudaStreamPerThread>>>(state);
                Check(cudaGetLastError());
            };
            run();
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            for (int step = 0; step < 16; ++step) run();
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&exec, graph, 0));
            for (int repeat = 0; repeat < 32; ++repeat) {
                if ((repeat + rank) % 7 == 0)
                    std::this_thread::sleep_for(std::chrono::microseconds(137 * (rank + 1)));
                Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            }
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaGraphExecDestroy(exec));
            Check(cudaGraphDestroy(graph));
        }
    }
    Check(cudaMemcpy(&errors, state + 1, sizeof(int), cudaMemcpyDeviceToHost));
    Check(cudaFree(state));
    Check(cudaFree(shared));
    Check(cudaFree(output));
    Check(cudaFree(input));
}

int main(int argc, char **argv) {
    const int ranks = argc > 1 ? std::atoi(argv[1]) : 8;
    const bool joint = argc > 2 && std::strcmp(argv[2], "joint") == 0;
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
        if (joint) {
            RunJoint(rank, ranks, errors[rank]);
            return;
        }
        Check(cudaSetDevice(rank));
        unsigned short *data, *host;
        int *deviceErrors;
        constexpr int maxPart = 16 * 4096;
        Check(cudaMalloc(&data, (size_t)ranks * maxPart * sizeof(*data)));
        Check(cudaMallocHost(&host, maxPart * sizeof(*host)));
        Check(cudaMalloc(&deviceErrors, sizeof(int)));
        Check(cudaMemsetAsync(deviceErrors, 0, sizeof(int), cudaStreamPerThread));
        for (int i = 0; i < maxPart; ++i) host[i] = Value(rank, i);
        // BF16 shard widths bracket the 256 uint4 / 2048 element flatten
        // threshold, including one-vector messages and larger batches.
        for (int width : {8, 64, 2040, 2048, 2056, 4096})
            for (int rows : {1, 2, 4, 8, 16}) {
                const int part = rows * width, count = ranks * part;
                auto run = [&] {
                    // Poison all foreign slices on every replay. A missing start
                    // or completion barrier must not silently reuse prior output.
                    Check(cudaMemsetAsync(data, 0xa5, (size_t)count * sizeof(*data), cudaStreamPerThread));
                    Check(cudaMemcpyAsync(data + rank * part, host, part * sizeof(*host),
                                          cudaMemcpyHostToDevice, cudaStreamPerThread));
                    if (!FastllmCudaCustomAllGatherRows(data, count, width, fastllm::BFLOAT16, rank)) std::abort();
                    CheckGather<<<32, 256, 0, cudaStreamPerThread>>>(data, part, ranks, deviceErrors);
                    Check(cudaGetLastError());
                    // Interleave the established reduction to exercise the shared
                    // signal counters with differing collective grid sizes.
                    if (!FastllmCudaCustomAllReduceRows(data, data, count, width, fastllm::BFLOAT16, rank)) std::abort();
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
    if (joint)
        std::printf("Joint reduce/gather: %d GPUs, five sizes, in/out-of-place, changing values, skewed graphs PASS\n", ranks);
    else
        std::printf("Custom AllGather: %d GPUs, six shard widths x five batches, eager +64 skewed graph replays PASS\n", ranks);
    return 0;
}
