#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_bf16.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static void Require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::_Exit(1); }
}
static void Check(cudaError_t state) { Require(state == cudaSuccess, cudaGetErrorString(state)); }

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 4) {
        std::puts("SKIP: requires four CUDA GPUs");
        return 77;
    }
    Require(FastllmInitNccl({0, 1, 2, 3}), "NCCL initialization");
    FastllmCudaSetNcclForceSync(false);
    constexpr int ranks = 4, k = 4096, n = 16, maxRows = 5958, words = 65536;
    std::atomic<int> ready{0};
    std::vector<std::thread> workers;
    for (int rank = 0; rank < ranks; ++rank) workers.emplace_back([&, rank] {
        Check(cudaSetDevice(rank));
        __nv_bfloat16 *x, *w, *y;
        int *send, *receive;
        Check(cudaMalloc(&x, size_t(maxRows) * k * sizeof(*x)));
        Check(cudaMalloc(&w, size_t(n) * k * sizeof(*w)));
        Check(cudaMalloc(&y, size_t(maxRows) * n * sizeof(*y)));
        Check(cudaMalloc(&send, words * sizeof(*send)));
        Check(cudaMalloc(&receive, ranks * words * sizeof(*receive)));
        std::vector<__nv_bfloat16> hostX(size_t(maxRows) * k), hostW(size_t(n) * k);
        for (size_t i = 0; i < hostX.size(); ++i)
            hostX[i] = __float2bfloat16(float(int((i + rank) % 17) - 8) / 32);
        for (size_t i = 0; i < hostW.size(); ++i)
            hostW[i] = __float2bfloat16(float(int((i + rank) % 13) - 6) / 64);
        Check(cudaMemcpy(x, hostX.data(), hostX.size() * sizeof(*x), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(w, hostW.data(), hostW.size() * sizeof(*w), cudaMemcpyHostToDevice));
        Check(cudaMemset(send, rank + 1, words * sizeof(*send)));
        cublasHandle_t handle;
        Require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS initialization");
        Require(cublasSetStream(handle, cudaStreamPerThread) == CUBLAS_STATUS_SUCCESS, "cuBLAS stream");
        ++ready;
        while (ready != ranks) std::this_thread::yield();

        // Reproduce Indexer projection sizes with a rotating slow rank. A fast
        // rank must not enter the next cuBLAS initialization while a peer has
        // not submitted AllGather. Do not insert GPU syncs between iterations:
        // those would mask the CUDA-driver/NCCL circular wait this test covers.
        const float alpha = 1, beta = 0;
        for (int step = 0; step < 8; ++step) {
            const int rows = step % 2 ? maxRows : 3655;
            if (rank == step % ranks) std::this_thread::sleep_for(std::chrono::milliseconds(150));
            Require(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k,
                &alpha, w, CUDA_R_16BF, k, x, CUDA_R_16BF, k, &beta, y, CUDA_R_16BF, n,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS, "BF16 GEMM");
            Require(FastllmNcclAllGather(send, receive, words, fastllm::INT32, rank), "AllGather");
        }
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<int> gathered(ranks * words);
        Check(cudaMemcpy(gathered.data(), receive, gathered.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int r = 0; r < ranks; ++r) for (int i = 0; i < words; ++i)
            Require(gathered[r * words + i] == (r + 1) * 0x01010101, "AllGather output");

        // Dyadic inputs keep all FP32 partial sums exact. Check every output
        // element against a CPU reference, reusing the 17-row input period.
        std::vector<__nv_bfloat16> output(size_t(maxRows) * n), expected(17 * n);
        Check(cudaMemcpy(output.data(), y, output.size() * sizeof(*y), cudaMemcpyDeviceToHost));
        for (int row = 0; row < 17; ++row) for (int col = 0; col < n; ++col) {
            float sum = 0;
            for (int j = 0; j < k; ++j)
                sum += __bfloat162float(hostX[row * k + j]) * __bfloat162float(hostW[col * k + j]);
            expected[row * n + col] = __float2bfloat16(sum);
        }
        for (int row = 0; row < maxRows; ++row) for (int col = 0; col < n; ++col)
            Require(__bfloat162float(output[row * n + col]) == __bfloat162float(expected[(row % 17) * n + col]),
                    "BF16 GEMM differs from exact reference");
        Require(cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS cleanup");
        for (void *pointer : {static_cast<void *>(x), static_cast<void *>(w), static_cast<void *>(y),
                              static_cast<void *>(send), static_cast<void *>(receive)}) Check(cudaFree(pointer));
    });
    for (auto &worker : workers) worker.join();
    std::puts("PASS: TP4 BF16 prefill + AllGather submission skew");
}
