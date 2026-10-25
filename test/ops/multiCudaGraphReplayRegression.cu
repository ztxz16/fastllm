#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(expr)                                                                                     \
    do {                                                                                                     \
        cudaError_t e = (expr);                                                                              \
        if (e != cudaSuccess) {                                                                              \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                       \
            std::exit(2);                                                                                    \
        }                                                                                                    \
    } while (0)
#define REQUIRE(expr)                                                                                        \
    do {                                                                                                     \
        if (!(expr)) {                                                                                       \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);                                       \
            std::exit(3);                                                                                    \
        }                                                                                                    \
    } while (0)

__global__ void producer(int *input, int value, unsigned long long delay) {
    auto start = clock64();
    while (clock64() - start < delay) {
    }
    *input = value;
}
__global__ void transform(const int *input, int *output) {
    *output = (*input) * 7 + 31;
}
__global__ void consumer(const int *output, int *results, int iteration) {
    results[iteration] = *output;
}

struct Rank {
    int device;
    int *input = nullptr, *output = nullptr, *results = nullptr;
    void *graph = nullptr, *exec = nullptr;
};

int main(int, char **) {
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    REQUIRE(count > 0);
    const int iterations = 256;
    std::vector<int> widths{1};
    for (int n : {2, 3, 4, 8})
        if (n <= count) widths.push_back(n);
    int cases = 0;
    for (int width : widths)
        for (int reversed = 0; reversed < 2; ++reversed) {
            std::vector<Rank> ranks(width);
            std::vector<int> devices;
            std::vector<void *> execs;
            for (int i = 0; i < width; ++i) {
                auto &r = ranks[i];
                r.device = reversed ? count - 1 - i : i;
                devices.push_back(r.device);
                CHECK_CUDA(cudaSetDevice(r.device));
                CHECK_CUDA(cudaMalloc(&r.input, sizeof(int)));
                CHECK_CUDA(cudaMalloc(&r.output, sizeof(int)));
                CHECK_CUDA(cudaMalloc(&r.results, iterations * sizeof(int)));
                CHECK_CUDA(cudaMemset(r.input, 0, sizeof(int)));
                CHECK_CUDA(cudaMemset(r.output, 0, sizeof(int)));
                REQUIRE(FastllmCudaGraphPrepareCaptureDevice());
                REQUIRE(FastllmCudaGraphBeginCapture());
                transform<<<1, 1, 0, cudaStreamPerThread>>>(r.input, r.output);
                REQUIRE(FastllmCudaGraphEndCapture(&r.graph));
                REQUIRE(FastllmCudaGraphInstantiate(r.graph, &r.exec));
                execs.push_back(r.exec);
            }
            REQUIRE(!fastllm::MultiCudaLaunchGraphs({}, {}));
            REQUIRE(!fastllm::MultiCudaLaunchGraphs(devices, {}));
            auto bad = execs;
            bad[0] = nullptr;
            REQUIRE(!fastllm::MultiCudaLaunchGraphs(devices, bad));
            if (width > 1) {
                auto duplicate = devices;
                duplicate[1] = duplicate[0];
                REQUIRE(!fastllm::MultiCudaLaunchGraphs(duplicate, execs));
            }
            for (int i = 0; i < iterations; ++i) {
                for (auto &r : ranks) {
                    CHECK_CUDA(cudaSetDevice(r.device));
                    // Delayed producers expose missing caller -> worker ordering.
                    producer<<<1, 1, 0, cudaStreamPerThread>>>(r.input, 1000 * i + r.device,
                                                               (i % 7 == 0) ? 150000 : 0);
                }
                const int original = (i + 3) % count;
                CHECK_CUDA(cudaSetDevice(original));
                REQUIRE(fastllm::MultiCudaLaunchGraphs(devices, execs));
                int actual = -1;
                CHECK_CUDA(cudaGetDevice(&actual));
                REQUIRE(actual == original);
                for (auto &r : ranks) {
                    CHECK_CUDA(cudaSetDevice(r.device));
                    // No host synchronization: consumers and the next iteration
                    // must both observe worker -> caller completion handoffs.
                    consumer<<<1, 1, 0, cudaStreamPerThread>>>(r.output, r.results, i);
                }
            }
            for (auto &r : ranks) {
                CHECK_CUDA(cudaSetDevice(r.device));
                CHECK_CUDA(cudaDeviceSynchronize());
                std::vector<int> got(iterations);
                CHECK_CUDA(
                    cudaMemcpy(got.data(), r.results, iterations * sizeof(int), cudaMemcpyDeviceToHost));
                for (int i = 0; i < iterations; ++i)
                    REQUIRE(got[i] == (1000 * i + r.device) * 7 + 31);
                FastllmCudaGraphExecDestroy(r.exec);
                FastllmCudaGraphDestroy(r.graph);
                CHECK_CUDA(cudaFree(r.results));
                CHECK_CUDA(cudaFree(r.output));
                CHECK_CUDA(cudaFree(r.input));
            }
            ++cases;
        }
    printf("PASS %d topology cases, %d changing-input asynchronous replays; invalid inputs rejected\n", cases,
           cases * iterations);
}
