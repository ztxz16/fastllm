#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

static void Require(bool ok, const char *message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        // A peer could be in a CUDA call; let the test runner bound the whole
        // process instead of joining a rank which can no longer progress.
        std::exit(1);
    }
}

static void Check(cudaError_t state) {
    Require(state == cudaSuccess, cudaGetErrorString(state));
}

static void RunGroup(const std::vector<int> &devices) {
    const int ranks = devices.size();
    Require(FastllmInitNccl(devices), "NCCL initialization");
    const auto generation = FastllmGetNcclGeneration();
    Require(FastllmInitNccl(devices) && FastllmGetNcclGeneration() == generation,
            "ready group should preserve its generation");
    FastllmCudaSetNcclForceSync(true);
    std::atomic<int> warmupReady{0};
    std::vector<std::thread> workers;
    for (int rank = 0; rank < ranks; ++rank) {
        workers.emplace_back([&, rank] {
            const int device = devices[rank];
            Check(cudaSetDevice(device));
            FastllmCudaClearThreadError();
            constexpr int n = 32, count = n * n;
            float *a = nullptr, *b = nullptr, *input = nullptr, *output = nullptr;
            for (auto ptr : {&a, &b, &input, &output}) {
                Check(cudaMalloc((void **)ptr, count * sizeof(float)));
            }
            std::vector<float> host(count, float(rank + 1));
            Check(cudaMemcpy(a, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
            std::fill(host.begin(), host.end(), 1.0f);
            Check(cudaMemcpy(b, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
            cublasHandle_t handle;
            Require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate");
            Require(cublasSetStream(handle, cudaStreamPerThread) == CUBLAS_STATUS_SUCCESS,
                    "cublasSetStream");
            const float alpha = 1.0f, beta = 0.0f;
            auto gemm = [&] {
                Require(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        n, n, n, &alpha, a, n, b, n, &beta, input, n) == CUBLAS_STATUS_SUCCESS,
                        "cublasSgemm");
            };
            auto reduce = [&](bool inPlace, bool noCustom) {
                auto fn = noCustom ? FastllmNcclAllReduceNoCustom : FastllmNcclAllReduce;
                fn(input, inPlace ? input : output, count, fastllm::DataType::FLOAT32, device);
                Require(!FastllmCudaGetThreadError(), "AllReduce thread error");
            };
            auto verify = [&](bool inPlace) {
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                Check(cudaMemcpy(host.data(), inPlace ? input : output,
                                 count * sizeof(float), cudaMemcpyDeviceToHost));
                const float expected = float(n * ranks * (ranks + 1) / 2);
                for (float value : host) Require(value == expected, "AllReduce sum mismatch");
            };

            // Initialize NCCL transports before capture. Exercise both entry
            // points, in-place/out-of-place, and the existing warmup sync.
            gemm(); reduce(false, false); verify(false);
            ++warmupReady;
            while (warmupReady != ranks) std::this_thread::yield();
            FastllmCudaSetNcclForceSync(false);
            for (int i = 0; i < 128; ++i) {
                if (rank == i % ranks) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                gemm();
                reduce(i % 2 != 0, i % 3 == 0);
                // Most iterations have no device/stream synchronization.
                if (i % 32 == 31) verify(i % 2 != 0);
            }
            verify(true);

            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed));
            for (int i = 0; i < 4; ++i) { gemm(); reduce(false, i % 2 != 0); }
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&exec, graph, 0));
            for (int i = 0; i < 8; ++i) Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            verify(false);
            Check(cudaGraphExecDestroy(exec));
            Check(cudaGraphDestroy(graph));
            // Capture/replay must leave eager rendezvous phases unchanged.
            gemm(); reduce(false, true); verify(false);
            Require(cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS, "cublasDestroy");
            for (auto ptr : {a, b, input, output}) Check(cudaFree(ptr));
        });
    }
    for (auto &worker : workers) worker.join();
    std::cout << "PASS: " << ranks << " ranks, eager GEMM/AllReduce and CUDA Graph\n";
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 3) {
        std::cout << "SKIP: requires at least three CUDA GPUs\n";
        return 77;
    }
    RunGroup({0, 1, 2});
    RunGroup({0, 1});
    if (count >= 4) RunGroup({0, 1, 2, 3});
    // Rebuild a same-size odd group with different membership/rank order.
    RunGroup(count >= 4 ? std::vector<int>{3, 1, 2} : std::vector<int>{2, 0, 1});
}
