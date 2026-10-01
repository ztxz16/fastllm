#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static void Check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> __device__ T Cast(float value);
template <> __device__ float Cast<float>(float value) { return value; }
template <> __device__ half Cast<half>(float value) { return __float2half_rn(value); }
template <> __device__ __nv_bfloat16 Cast<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}
template <typename T> __device__ float Float(T value);
template <> __device__ float Float<float>(float value) { return value; }
template <> __device__ float Float<half>(half value) { return __half2float(value); }
template <> __device__ float Float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
__device__ float Input(int rank, int index, int iteration) {
    return (rank ? -0.3125f : 0.15625f) + (index % 17) * 0.0078125f +
           (iteration % 13) * 0.03125f;
}
template <typename T> __global__ void Fill(T *data, int count, int rank, int iteration) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) data[i] = Cast<T>(Input(rank, i, iteration));
}
template <typename T> __global__ void Verify(const T *data, int count, int iteration,
                                          unsigned int *errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        const T a = Cast<T>(Input(0, i, iteration));
        const T b = Cast<T>(Input(1, i, iteration));
        const T expected = Cast<T>(Float(a) + Float(b));
        if (Float(data[i]) != Float(expected)) atomicAdd(errors, 1);
    }
}
static void Delay(void *) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); }

template <typename T> __global__ void VerifyAdd(const T *data, int count,
        int rank, int iteration, unsigned int *errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        const T a = Cast<T>(Input(0, i, iteration));
        const T b = Cast<T>(Input(1, i, iteration));
        const T residual = Cast<T>(Input(rank, i, iteration + 17));
        const T expected = Cast<T>(Float(Cast<T>(Float(residual) + Float(a))) + Float(b));
        if (Float(data[i]) != Float(expected)) atomicAdd(errors, 1);
    }
}

template <typename T> static bool Run(int type) {
    // Includes the model's decode/MTP/prefill shapes, partial last blocks,
    // and the largest tensor admitted by the register-resident protocol.
    const int counts[] = {16 / int(sizeof(T)), 2560, 10240,
                          std::min(81920, 36 * 512 * 16 / int(sizeof(T))),
                          36 * 512 * 16 / int(sizeof(T))};
    constexpr int maxBytes = 36 * 512 * 16;
    constexpr int iterations = 300;
    std::atomic<bool> passed(true);
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 2; ++rank) workers.emplace_back([&, rank]() {
        Check(cudaSetDevice(rank));
        T *inputs[2], *output;
        unsigned int *errors;
        for (auto &p : inputs) Check(cudaMalloc((void**)&p, maxBytes));
        Check(cudaMalloc((void**)&output, maxBytes + 16));
        Check(cudaMalloc((void**)&errors, sizeof(*errors)));
        Check(cudaMemsetAsync(errors, 0, sizeof(*errors), cudaStreamPerThread));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const int count = counts[iteration % 5];
            T *input = inputs[iteration & 1];
            T *dest = iteration & 2 ? output : input;
            if (iteration % 29 == rank)
                Check(cudaLaunchHostFunc(cudaStreamPerThread, Delay, nullptr));
            Fill<T><<<64, 256, 0, cudaStreamPerThread>>>(input, count, rank, iteration);
            if (!FastllmTryTP2P2PAllReduce(input, dest, count, type, rank)) {
                passed.store(false); return;
            }
            Verify<T><<<64, 256, 0, cudaStreamPerThread>>>(dest, count, iteration, errors);
        }
        // One rank's destination is unaligned. Both must reject together,
        // then resume the sequence with a valid reduction.
        bool rejected = !FastllmTryTP2P2PAllReduce(inputs[0], rank ? output + 1 : output,
                                                 2560, type, rank);
        Fill<T><<<64, 256, 0, cudaStreamPerThread>>>(inputs[0], 2560, rank, iterations);
        bool resumed = FastllmTryTP2P2PAllReduce(inputs[0], inputs[0], 2560, type, rank);
        Verify<T><<<64, 256, 0, cudaStreamPerThread>>>(inputs[0], 2560, iterations, errors);
        if (!fastllm::GetFastllmEnv().cudaGraph) {
            // The residual variant shares host pointer exchange and signals
            // with ordinary sum. Alternate shapes and inputs without draining
            // streams; check its two dtype-rounded additions independently.
            for (int iteration = 0; iteration < 64; ++iteration) {
                const int count = 4096 + (iteration % 7) * 256;
                T *input = inputs[iteration & 1];
                if (iteration % 19 == rank)
                    Check(cudaLaunchHostFunc(cudaStreamPerThread, Delay, nullptr));
                Fill<T><<<64, 256, 0, cudaStreamPerThread>>>(input, count, rank, iteration);
                Fill<T><<<64, 256, 0, cudaStreamPerThread>>>(output, count, rank, iteration + 17);
                if (!FastllmTryTP2P2PAllReduceAdd(input, output, count, type, rank)) {
                    passed.store(false); return;
                }
                VerifyAdd<T><<<64, 256, 0, cudaStreamPerThread>>>(output, count, rank, iteration, errors);
            }
        }
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        unsigned int total = 0;
        Check(cudaMemcpy(&total, errors, sizeof(total), cudaMemcpyDeviceToHost));
        bool sizeFallback = !FastllmTryTP2P2PAllReduce(inputs[0], output,
            maxBytes / sizeof(T) + 16, type, rank);
        bool tailFallback = !FastllmTryTP2P2PAllReduce(inputs[0], output, 3, type, rank);
        if (total || !rejected || !resumed || !sizeFallback || !tailFallback) passed.store(false);
        if (fastllm::GetFastllmEnv().cudaGraph) {
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            if (FastllmTryTP2P2PAllReduce(inputs[0], output, 2560, type, rank)) passed.store(false);
            cudaGraph_t graph = nullptr;
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphDestroy(graph));
        }
        std::printf("rank=%d dtype=%d errors=%u rejected=%d resumed=%d\n",
                    rank, type, total, rejected, resumed);
        for (auto p : inputs) Check(cudaFree(p));
        Check(cudaFree(output)); Check(cudaFree(errors));
    });
    for (auto &worker : workers) worker.join();
    return passed.load();
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) return 77;
    for (int rank = 0; rank < 2; ++rank) {
        int peer = 0;
        Check(cudaDeviceCanAccessPeer(&peer, rank, 1 - rank));
        if (!peer) return 77;
    }
    setenv("FASTLLM_CUDA_GRAPH", "0", 1);
    setenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "0", 1);
    if (!FastllmInitNccl({0, 1})) return 1;
    FastllmCudaSetNcclForceSync(false);
    bool passed = Run<half>(fastllm::DataType::FLOAT16);
    passed = Run<__nv_bfloat16>(fastllm::DataType::BFLOAT16) && passed;
    passed = Run<float>(fastllm::DataType::FLOAT32) && passed;
    fastllm::SetCudaGraph(true);
    passed = Run<half>(fastllm::DataType::FLOAT16) && passed;
    passed = Run<float>(fastllm::DataType::FLOAT32) && passed;
    std::puts(passed ? "PASS: TP2 in-place peer sum" : "FAIL: TP2 in-place peer sum");
    return passed ? 0 : 1;
}
