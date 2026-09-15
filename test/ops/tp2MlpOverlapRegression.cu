#include "devices/multicuda/tp2mlppipeline.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <chrono>
#include <memory>

#define CUDA_OK(call)                                                                                        \
    do {                                                                                                     \
        auto status = (call);                                                                                \
        if (status != cudaSuccess) {                                                                         \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(status));              \
            std::abort();                                                                                    \
        }                                                                                                    \
    } while (0)
#define BLAS_OK(call)                                                                                        \
    do {                                                                                                     \
        auto status = (call);                                                                                \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                               \
            std::fprintf(stderr, "%s:%d cuBLAS %d\n", __FILE__, __LINE__, int(status));                      \
            std::abort();                                                                                    \
        }                                                                                                    \
    } while (0)
#define NCCL_OK(call)                                                                                        \
    do {                                                                                                     \
        auto status = (call);                                                                                \
        if (status != ncclSuccess) {                                                                         \
            std::fprintf(stderr, "%s:%d NCCL %s\n", __FILE__, __LINE__, ncclGetErrorString(status));         \
            std::abort();                                                                                    \
        }                                                                                                    \
    } while (0)

static __global__ void ReferenceAdd(half *partial, const half *residual, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) partial[i] = __hadd(partial[i], residual[i]);
}

static __global__ void Fill(half *output, size_t count, unsigned seed, float scale) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)blockDim.x * gridDim.x) {
        unsigned x = static_cast<unsigned>(i) * 2654435761u + seed;
        x ^= x >> 16;
        x *= 2246822519u;
        x ^= x >> 13;
        output[i] = __float2half((static_cast<int>(x % 61) - 30) * scale);
    }
}

int main(int argc, char **argv) {
    constexpr int N = 2048, H = 5120, K = 8704, Iterations = 3;
    constexpr size_t Count = (size_t)N * H;
    int devices = 0;
    CUDA_OK(cudaGetDeviceCount(&devices));
    if (devices < 2) return 77;
    using Backend = fastllm::TP2MlpReduction::Backend;
    const char *mode = argc > 1 ? argv[1] : "wht6";
    Backend backend;
    if (!std::strcmp(mode, "nccl"))
        backend = Backend::Nccl;
    else if (!std::strcmp(mode, "wht6"))
        backend = Backend::WHT6;
    else
        return 2;
    bool quantized = backend == Backend::WHT6;
    std::unique_ptr<fastllm::TP2WHT6AllReduce> transport;
    if (backend != Backend::Nccl) {
        transport.reset(new fastllm::TP2WHT6AllReduce());
        CUDA_OK(transport->Init(0, 1));
    }
    ncclComm_t comms[2] = {};
    if (backend == Backend::Nccl) NCCL_OK(ncclCommInitAll(comms, 2, nullptr));
    fastllm::NcclSubmitRendezvous submit(2);
    std::vector<half> results[2][Iterations];
    auto worker = [&](int rank) {
        CUDA_OK(cudaSetDevice(rank));
        fastllm::TP2MlpReduction reduction(rank, backend, comms[rank], &submit, transport.get());
        cublasHandle_t handle;
        BLAS_OK(cublasCreate(&handle));
        BLAS_OK(cublasSetStream(handle, cudaStreamPerThread));
        fastllm::TP2MlpPipeline pipeline;
        CUDA_OK(pipeline.Init(N, H, K));
        half *input, *residual, *reference, *gateup, *swiglu, *partial;
        CUDA_OK(cudaMalloc(&input, Count * sizeof(half)));
        CUDA_OK(cudaMalloc(&residual, Count * sizeof(half)));
        CUDA_OK(cudaMalloc(&reference, Count * sizeof(half)));
        CUDA_OK(cudaMalloc(&gateup, (size_t)N * K * 2 * sizeof(half)));
        CUDA_OK(cudaMalloc(&swiglu, (size_t)N * K * sizeof(half)));
        CUDA_OK(cudaMalloc(&partial, Count * sizeof(half)));
        half *download = nullptr;
        CUDA_OK(cudaMallocHost(&download, Count * sizeof(half)));
        const half beta = __float2half(0.f);
        std::vector<half> hostReference(Count);
        if (!quantized) {
            // Independent CPU reference for rank-0 residual rounding. Include
            // cancellation and an odd tail exactly beyond a 256-thread block.
            constexpr int TailCount = 513;
            std::vector<half> x(TailCount), r(TailCount + 1), expected(TailCount + 1);
            for (int i = 0; i < TailCount; ++i) {
                float local0 = i % 2 ? 0.001f : 1024.f;
                float local1 = i % 2 ? 0.002f : -1024.f;
                x[i] = __float2half(rank == 0 ? local0 : local1);
                r[i] = __float2half(0.125f);
                half first = __float2half(__half2float(__float2half(local0)) + __half2float(r[i]));
                expected[i] = __float2half(__half2float(first) + __half2float(__float2half(local1)));
            }
            r[TailCount] = expected[TailCount] = __float2half(7.f);
            CUDA_OK(cudaMemcpyAsync(partial, x.data(), TailCount * sizeof(half), cudaMemcpyHostToDevice,
                                    cudaStreamPerThread));
            CUDA_OK(cudaMemcpyAsync(residual, r.data(), r.size() * sizeof(half), cudaMemcpyHostToDevice,
                                    cudaStreamPerThread));
            CUDA_OK(reduction.Run(partial, residual, TailCount, cudaStreamPerThread));
            CUDA_OK(cudaMemcpyAsync(download, residual, r.size() * sizeof(half), cudaMemcpyDeviceToHost,
                                    cudaStreamPerThread));
            CUDA_OK(cudaStreamSynchronize(cudaStreamPerThread));
            if (std::memcmp(download, expected.data(), expected.size() * sizeof(half))) std::abort();
            std::printf("%s rank=%d PASS residual order, cancellation, 513-value tail and output guard\n",
                        mode, rank);
        }
        for (int iteration = 0; iteration < Iterations; ++iteration) {
            // Marlin dequantization produces the final FP16 weights (alpha=1).
            const half alpha = __float2half(1.f);
            const float weightScale = 0.000768f;
            // Different rank-local weights; replicated activation and residual.
            Fill<<<1024, 256>>>(pipeline.GateUpWeight(), (size_t)2 * H * K, 123 + rank * 37 + iteration,
                                iteration == 0 ? 0.f : weightScale);
            Fill<<<1024, 256>>>(pipeline.DownWeight(), (size_t)H * K, 456 + rank * 41 + iteration,
                                weightScale);
            Fill<<<1024, 256>>>(input, Count, 789 + iteration, 0.02f);
            Fill<<<1024, 256>>>(residual, Count, 999 + iteration, 0.01f);
            CUDA_OK(cudaMemcpyAsync(reference, residual, Count * sizeof(half), cudaMemcpyDeviceToDevice,
                                    cudaStreamPerThread));
            // Unsplit eager reference: two full GEMMs and one full reduction.
            BLAS_OK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2 * K, N, H, &alpha,
                                 pipeline.GateUpWeight(), CUDA_R_16F, H, input, CUDA_R_16F, H, &beta, gateup,
                                 CUDA_R_16F, 2 * K, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
            fastllm::TP2MlpSwiGluKernel<<<(N * K + 255) / 256, 256>>>(gateup, swiglu, N * K, K);
            BLAS_OK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, H, N, K, &alpha, pipeline.DownWeight(),
                                 CUDA_R_16F, K, swiglu, CUDA_R_16F, K, &beta, partial, CUDA_R_16F, H,
                                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
            if (!quantized && rank == 0)
                ReferenceAdd<<<(Count + 255) / 256, 256>>>(partial, reference, Count);
            if (backend == Backend::Nccl) {
                if (!submit.Wait(rank, fastllm::NcclSubmitRendezvous::Before, Count, ncclHalf)) std::abort();
                NCCL_OK(ncclAllReduce(partial, reference, Count, ncclHalf, ncclSum, comms[rank],
                                      cudaStreamPerThread));
                if (!submit.Wait(rank, fastllm::NcclSubmitRendezvous::After, Count, ncclHalf)) std::abort();
            } else {
                CUDA_OK(transport->Run(rank, partial, reference, Count, cudaStreamPerThread, reference));
            }
            CUDA_OK(cudaStreamSynchronize(cudaStreamPerThread));
            if (rank == (iteration & 1)) std::this_thread::sleep_for(std::chrono::microseconds(200));
            CUDA_OK(pipeline.Run(input, residual, handle, reduction, cudaStreamPerThread));
            // Run promises a dependency on compute before its caller consumes output.
            CUDA_OK(cudaMemcpyAsync(download, residual, Count * sizeof(half), cudaMemcpyDeviceToHost,
                                    cudaStreamPerThread));
            CUDA_OK(cudaStreamSynchronize(cudaStreamPerThread));
            results[rank][iteration].resize(Count);
            std::memcpy(results[rank][iteration].data(), download, Count * sizeof(half));
            CUDA_OK(
                cudaMemcpy(hostReference.data(), reference, Count * sizeof(half), cudaMemcpyDeviceToHost));
            double error = 0, energy = 0, maximum = 0;
            size_t changed = 0;
            for (size_t i = 0; i < Count; ++i) {
                float a = __half2float(results[rank][iteration][i]);
                float b = __half2float(hostReference[i]);
                if (!std::isfinite(a) || !std::isfinite(b)) std::abort();
                double d = a - b;
                error += d * d;
                energy += double(b) * b;
                maximum = std::max(maximum, std::abs(d));
                changed += a != b;
            }
            double relativeRmse = std::sqrt(error / std::max(energy, 1e-30));
            std::printf("%s rank=%d iteration=%d relative_RMSE=%.8g max_abs=%.8g changed=%zu\n", mode, rank,
                        iteration, relativeRmse, maximum, changed);
            // Chunked cuBLAS can change accumulation order. This checks the
            // complete pipeline against its unfused numerical contract.
            if (relativeRmse > 0.001 || maximum > 0.005) std::abort();
        }
        CUDA_OK(cudaFree(input));
        CUDA_OK(cudaFree(residual));
        CUDA_OK(cudaFree(reference));
        CUDA_OK(cudaFree(gateup));
        CUDA_OK(cudaFree(swiglu));
        CUDA_OK(cudaFree(partial));
        CUDA_OK(cudaFreeHost(download));
        BLAS_OK(cublasDestroy(handle));
    };
    std::thread a(worker, 0), b(worker, 1);
    a.join();
    b.join();
    for (int iteration = 0; iteration < Iterations; ++iteration) {
        if (std::memcmp(results[0][iteration].data(), results[1][iteration].data(), Count * sizeof(half))) {
            std::fprintf(stderr, "Rank outputs differ, iteration %d\n", iteration);
            return 1;
        }
    }
    if (backend == Backend::Nccl) {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_OK(cudaSetDevice(rank));
            NCCL_OK(ncclCommDestroy(comms[rank]));
        }
    }
    std::printf("PASS %s: fixed 2048x5120x8704, 3 iterations, zero/nonzero data, rank skew, buffer reuse, "
                "rank-bitwise equality.\n",
                mode);
}
