#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include "fastllm.h"
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <exception>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
struct Memory {
    void *p = nullptr;
    explicit Memory(size_t size) { Check(cudaMalloc(&p, size) == cudaSuccess, "allocate"); }
    ~Memory() { cudaFree(p); }
};
static void Case(int N, int K, const std::vector<int> &sizes, bool withBias = false) {
    int maxRows = *std::max_element(sizes.begin(), sizes.end());
    unsigned seed = 935;
    auto random = [&]() { seed = 1664525u * seed + 1013904223u; return (int(seed >> 16) - 32768) / 32768.f; };
    std::vector<__nv_bfloat16> x(size_t(maxRows) * K), expanded(size_t(N) * K), b(N);
    for (auto &v : x) v = __float2bfloat16(random());
    Data weight(FP8_E4M3_PERCHANNEL, {N, K}); weight.Allocate();
    std::vector<float> biasValues(N);
    for (int c = 0; c < N; ++c) {
        float scale = .0001f * (1 + c % 11);
        biasValues[c] = random() / 16;
        b[c] = __float2bfloat16(biasValues[c]);
        memcpy(weight.cpuData + size_t(c) * (K + 4) + K, &scale, 4);
        for (int i = 0; i < K; ++i) {
            __nv_fp8_e4m3 value(random() * 448);
            weight.cpuData[size_t(c) * (K + 4) + i] = value.__x;
            expanded[size_t(c) * K + i] = __float2bfloat16(float(value) * scale);
        }
    }
    weight.ToDevice(DataDevice::CUDA, {0}, true);
    Data bias(FLOAT32);
    if (withBias) { Data values(FLOAT32, {N}, biasValues); bias.CopyFrom(values); }
    if (withBias) bias.ToDevice(DataDevice::CUDA, {0}, true);
    Data input(BFLOAT16, {maxRows, K}), output(BFLOAT16, {maxRows, N});
    input.ToDevice(DataDevice::CUDA, {0}, false); input.Allocate(false);
    output.ToDevice(DataDevice::CUDA, {0}, false); output.Allocate(false);
    Memory dw(expanded.size() * 2), ref(size_t(maxRows) * N * 2);
    Check(cudaMemcpyAsync(input.cudaData, x.data(), x.size() * 2, cudaMemcpyHostToDevice, cudaStreamPerThread) == cudaSuccess, "input");
    Check(cudaMemcpyAsync(dw.p, expanded.data(), expanded.size() * 2, cudaMemcpyHostToDevice, cudaStreamPerThread) == cudaSuccess, "weight");
    cudaStreamSynchronize(cudaStreamPerThread);
    cublasHandle_t handle; Check(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "handle");
    cublasSetStream(handle, cudaStreamPerThread);
    FastllmCudaSetNcclForceSync(true);
    std::exception_ptr preparationError;
    std::thread prepare([&] {
        try { FastllmCudaSetDevice(0); Check(FastllmCudaPreparePerChannelW8A16(weight), "weight preparation"); }
        catch (...) { preparationError = std::current_exception(); }
    });
    prepare.join(); if (preparationError) std::rethrow_exception(preparationError);
    FastllmCudaSetNcclForceSync(false);
    Check(FastllmCudaPerChannelW8A16(input.cudaData, weight, output.cudaData, 1, K, N), "worker claimed prepared layout");
    if (withBias) {
        // Layout ownership reserves auxiliary slots before any FP32 bias exists.
        // The generic path must initialize slot zero instead of appending past it.
        Data zero(FLOAT32, {1, K}), biased(FLOAT32, {1, N});
        zero.ToDevice(DataDevice::CUDA, {0}, false); zero.Allocate(false);
        biased.ToDevice(DataDevice::CUDA, {0}, false); biased.Allocate(false);
        Check(cudaMemsetAsync(zero.cudaData, 0, size_t(K) * sizeof(float), cudaStreamPerThread) == cudaSuccess,
            "zero FP32 input");
        Check(FastllmCudaMatMulFloatFP8E4M3PerChannel(zero, weight, bias, biased, 1, K, N),
            "FP32 bias after layout preparation");
        std::vector<float> actual(N);
        Check(cudaMemcpyAsync(actual.data(), biased.cudaData, size_t(N) * sizeof(float),
            cudaMemcpyDeviceToHost, cudaStreamPerThread) == cudaSuccess, "FP32 bias result");
        Check(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess, "FP32 bias sync");
        Check(actual == biasValues, "reserved auxiliary slot lost FP32 bias");
    }
    size_t total = 0;
    cudaStreamSynchronize(cudaStreamPerThread);
    for (int M : sizes) {
        auto run = [&]() {
            if (withBias) return FastllmCudaBFloat16MatMulFP8E4M3PerChannel(input, weight, bias, output, M, K, N);
            return FastllmCudaPerChannelW8A16(input.cudaData, weight, output.cudaData, M, K, N);
        };
        Check(run(), "eager dispatch");
        void *graph = nullptr, *exec = nullptr;
        Check(FastllmCudaGraphBeginCapture(), "capture begin");
        Check(run(), "capture dispatch");
        Check(FastllmCudaGraphEndCapture(&graph), "capture end");
        Check(FastllmCudaGraphInstantiate(graph, &exec), "instantiate");
        for (int replay = 0; replay < 2; ++replay) {
            for (auto &v : x) v = __float2bfloat16(random());
            cudaMemcpyAsync(input.cudaData, x.data(), x.size() * 2, cudaMemcpyHostToDevice, cudaStreamPerThread);
            Check(FastllmCudaGraphLaunch(exec), "replay");
            float alpha = 1, beta = 0;
            Check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                &alpha, dw.p, CUDA_R_16BF, K, input.cudaData, CUDA_R_16BF, K,
                &beta, ref.p, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS, "reference");
            std::vector<__nv_bfloat16> actual(size_t(M) * N), expected(actual.size());
            cudaMemcpyAsync(actual.data(), output.cudaData, actual.size() * 2, cudaMemcpyDeviceToHost, cudaStreamPerThread);
            cudaMemcpyAsync(expected.data(), ref.p, expected.size() * 2, cudaMemcpyDeviceToHost, cudaStreamPerThread);
            Check(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess, "sync");
            double err = 0, norm = 0, maxabs = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                float want = float(expected[i]);
                if (withBias) want = float(__float2bfloat16(want + float(b[i % N])));
                float got = float(actual[i]); Check(std::isfinite(got), "nonfinite");
                err += double(got - want) * (got - want); norm += double(want) * want;
                maxabs = std::max(maxabs, double(fabs(got - want)));
            }
            double rrmse = sqrt(err / std::max(norm, 1.e-30));
            Check(rrmse < .001, "BF16 weight reference mismatch");
            printf("N=%d K=%d M=%d bias=%d replay=%d rrmse=%.9g max_abs=%.9g\n", N, K, M, withBias, replay, rrmse, maxabs);
        }
        size_t beforeReplay, afterReplay;
        cudaMemGetInfo(&beforeReplay, &total);
        for (int repeat = 0; repeat < 10; ++repeat) Check(FastllmCudaGraphLaunch(exec), "repeat graph");
        cudaStreamSynchronize(cudaStreamPerThread); cudaMemGetInfo(&afterReplay, &total);
        Check(afterReplay == beforeReplay, "graph replay allocated memory");
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
    }
    cudaStreamSynchronize(cudaStreamPerThread);
    size_t beforeRelease; cudaMemGetInfo(&beforeRelease, &total);
    FastllmCudaReleaseBFloat16FP8Auxiliary(weight);
    size_t afterRelease; cudaMemGetInfo(&afterRelease, &total);
    const size_t minimumReleased = size_t(N) * K > (2u << 20) ? size_t(N) * K - (2u << 20) : 0;
    Check(afterRelease >= beforeRelease + minimumReleased, "layout not released");
    Check(!FastllmCudaPerChannelW8A16(input.cudaData, weight, output.cudaData, 1, K, N), "late allocation forbidden");
    FastllmCudaSetNcclForceSync(true);
    Check(FastllmCudaPerChannelW8A16(input.cudaData, weight, output.cudaData, 1, K, N), "rebuild after invalidation");
    FastllmCudaSetNcclForceSync(false);
    cudaStreamSynchronize(cudaStreamPerThread);
    cudaMemGetInfo(&beforeRelease, &total);
    weight.FreeSpace();
    cudaMemGetInfo(&afterRelease, &total);
    Check(afterRelease >= beforeRelease + minimumReleased, "Data invalidation did not free layout");
    cublasDestroy(handle);
}
static void ConcurrentWorkers() {
    constexpr int N = 256, K = 2048, M = 32;
    Data weight(FP8_E4M3_PERCHANNEL, {N, K}); weight.Allocate();
    for (int c = 0; c < N; ++c) {
        float scale = .125f;
        memcpy(weight.cpuData + size_t(c) * (K + 4) + K, &scale, 4);
        memset(weight.cpuData + size_t(c) * (K + 4), __nv_fp8_e4m3(1.f).__x, K);
    }
    weight.ToDevice(DataDevice::CUDA, {0}, true);
    std::atomic<int> ready{0}; std::atomic<bool> go{false};
    std::exception_ptr errors[2];
    FastllmCudaSetNcclForceSync(true);
    std::thread workers[2];
    for (int worker = 0; worker < 2; ++worker) workers[worker] = std::thread([&, worker] {
        bool announced = false;
        try {
            FastllmCudaSetDevice(0);
            Memory x(size_t(M) * K * 2), y(size_t(M) * N * 2);
            std::vector<__nv_bfloat16> host(size_t(M) * K, __float2bfloat16(worker + 1.f));
            cudaMemcpyAsync(x.p, host.data(), host.size() * 2, cudaMemcpyHostToDevice, cudaStreamPerThread);
            Check(FastllmCudaPerChannelW8A16(x.p, weight, y.p, M, K, N), "worker warmup");
            cudaStreamSynchronize(cudaStreamPerThread);
            ++ready; announced = true; while (!go) std::this_thread::yield();
            void *graph = nullptr, *exec = nullptr;
            Check(FastllmCudaGraphBeginCapture(), "worker capture");
            Check(FastllmCudaPerChannelW8A16(x.p, weight, y.p, M, K, N), "worker dispatch");
            Check(FastllmCudaGraphEndCapture(&graph), "worker capture end");
            Check(FastllmCudaGraphInstantiate(graph, &exec), "worker instantiate");
            for (int i = 0; i < 30; ++i) Check(FastllmCudaGraphLaunch(exec), "worker replay");
            std::vector<__nv_bfloat16> result(size_t(M) * N);
            cudaMemcpyAsync(result.data(), y.p, result.size() * 2, cudaMemcpyDeviceToHost, cudaStreamPerThread);
            Check(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess, "worker sync");
            for (auto value : result) Check(float(value) == (worker + 1.f) * K * .125f, "worker scratch isolation");
            FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
        } catch (...) { errors[worker] = std::current_exception(); if (!announced) ++ready; }
    });
    while (ready < 2) std::this_thread::yield();
    FastllmCudaSetNcclForceSync(false); go = true;
    for (auto &worker : workers) worker.join();
    for (auto &error : errors) if (error) std::rethrow_exception(error);
    puts("PASS concurrent workers: shared immutable weight, isolated split-K buffers");
}
int main() {
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
        if (prop.major != 9 || prop.minor != 0) return 77;
        FastllmCudaSetDevice(0);
        Case(14848, 4096, {1, 2, 7, 16, 17, 32, 33, 64, 127, 128, 129});
        Case(4096, 8192, {1, 3, 5, 8, 16, 31, 32, 63, 64, 128, 257});
        Case(72, 512, {1, 7, 33, 129}, true);
        Case(2048, 4096, {1, 16, 32});
        Case(4096, 2048, {1, 16, 32});
        ConcurrentWorkers();
        puts("PASS W8A16 per-channel: BF16 reference, tail rows, split K, graph replay, bias, cache release");
        return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
