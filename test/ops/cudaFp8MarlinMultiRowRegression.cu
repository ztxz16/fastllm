#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
static void Expect(bool b, const char *s) {
    if (!b) throw std::runtime_error(s);
}
static __global__ void AddBias(half *output, const half *bias, int width) {
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        size_t p = size_t(blockIdx.x) * width + i;
        output[p] = __hadd(output[p], bias[i]);
    }
}
static void Run(int gpu) {
    using namespace fastllm;
    FastllmCudaSetDevice(gpu);
    Executor executor;
    executor.SetFirstDevice("cuda:" + std::to_string(gpu));
    struct Shape {
        int out, in;
        bool bias;
    };
    for (auto shape : {Shape{192, 384, true}, Shape{16384, 4096, true}, Shape{17408, 5120, false},
                       Shape{5120, 8704, false}}) {
        int n = shape.out, k = shape.in;
        std::mt19937 rng(42 + n + k);
        std::vector<uint8_t> bytes(size_t(n) * k);
        for (auto &v : bytes) v = (rng() % 2) * 128 + rng() % 127;
        Data weight(FP8_E4M3, {n, k}), bias(FLOAT32);
        weight.blockM = weight.blockK = 128;
        weight.scales.resize(size_t((n + 127) / 128) * (k / 128));
        for (auto &s : weight.scales) s = std::ldexp(.5f + (rng() % 512) / 1024.f, int(rng() % 8) - 12);
        weight.Allocate();
        std::memcpy(weight.cpuData, bytes.data(), bytes.size());
        weight.ToDevice(CUDA, std::vector<int>{gpu});
        if (shape.bias) {
            bias.Resize({n});
            bias.Allocate();
            for (int i = 0; i < n; ++i) ((float *)bias.cpuData)[i] = (i % 31 - 15) / 32.f;
            bias.ToDevice(CUDA, std::vector<int>{gpu});
        }
        constexpr int MaxRows = 128;
        std::vector<float> values(size_t(MaxRows) * k);
        for (auto &v : values) v = (int(rng() % 2049) - 1024) / 1024.f;
        Data input(FLOAT16, {MaxRows, k}, values);
        Data actual(FLOAT16, {MaxRows + 1, n}, std::vector<float>(size_t(MaxRows + 1) * n, 0));
        Data expected(FLOAT16, {MaxRows + 1, n}, std::vector<float>(size_t(MaxRows + 1) * n, 0));
        for (auto *p : {&input, &actual, &expected}) p->ToDevice(CUDA, std::vector<int>{gpu});
        FastllmCudaSetNcclForceSync(true);
        Expect(FastllmCudaHalfMatMulFloatFP8E4M3(input, weight, bias, actual, 4, k, n), "warmup");
        Check(cudaDeviceSynchronize());
        FastllmCudaSetNcclForceSync(false);
        Expect(FastllmCudaHasFp8MarlinLayout(weight), "layout not packed");
        std::vector<uint8_t> packedBefore(bytes.size());
        Check(cudaMemcpy(packedBefore.data(), weight.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
        for (int rows : {1, 2, 3, 4, 8, 128}) {
            bool gemv = rows == 1 || (rows <= 3 && n >= 16384 && k >= 4096 && k <= n / 2);
            if (gemv) {
                void *a = input.cudaData, *c = expected.cudaData;
                for (int row = 0; row < rows; ++row) {
                    input.cudaData = (half *)a + size_t(row) * k;
                    expected.cudaData = (half *)c + size_t(row) * n;
                    Expect(FastllmCudaHalfMatMulFloatFP8E4M3(input, weight, bias, expected, 1, k, n),
                           "one-row reference");
                }
                input.cudaData = a;
                expected.cudaData = c;
            } else {
                Expect(FastllmCudaMarlinHalfFP8Gemm(input.cudaData, (uint32_t *)weight.cudaData,
                                                    (half *)weight.extraCudaHalfData[1], expected.cudaData,
                                                    rows, n, k, 128, (int *)weight.extraCudaData[3]),
                       "Marlin reference");
                if (shape.bias)
                    AddBias<<<rows, 256, 0, cudaStreamPerThread>>>((half *)expected.cudaData,
                                                                   (half *)weight.extraCudaHalfData[0], n);
            }
            std::vector<half> ref(size_t(rows) * n), result(ref.size());
            Check(cudaMemcpy(ref.data(), expected.cudaData, ref.size() * 2, cudaMemcpyDeviceToHost));
            for (int capture : {0, 1}) {
                Check(
                    cudaMemsetAsync(actual.cudaData, 0xa5, size_t(MaxRows + 1) * n * 2, cudaStreamPerThread));
                cudaGraph_t graph{};
                cudaGraphExec_t executable{};
                if (capture) {
                    Check(cudaStreamSynchronize(cudaStreamPerThread));
                    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal));
                }
                Expect(FastllmCudaHalfMatMulFloatFP8E4M3(input, weight, bias, actual, rows, k, n),
                       "dispatch");
                if (capture) {
                    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    Check(cudaGraphInstantiate(&executable, graph, 0));
                    Check(cudaGraphLaunch(executable, cudaStreamPerThread));
                }
                Check(cudaMemcpy(result.data(), actual.cudaData, result.size() * 2, cudaMemcpyDeviceToHost));
                Expect(!std::memcmp(result.data(), ref.data(), ref.size() * 2),
                       "output differs from selected backend reference");
                for (auto v : result) Expect(std::isfinite(__half2float(v)), "nonfinite result");
                uint16_t guard[64];
                Check(cudaMemcpy(guard, (half *)actual.cudaData + size_t(rows) * n, sizeof(guard),
                                 cudaMemcpyDeviceToHost));
                for (auto v : guard) Expect(v == 0xa5a5, "output overrun");
                if (capture) {
                    Check(cudaGraphExecDestroy(executable));
                    Check(cudaGraphDestroy(graph));
                }
            }
            printf("PASS gpu=%d N=%d K=%d rows=%d bias=%d backend=%s eager+graph bitwise-reference+guard\n",
                   gpu, n, k, rows, shape.bias, gemv ? "layout_gemv" : "marlin");
            fflush(stdout);
        }
        Check(cudaMemcpy(bytes.data(), weight.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
        Expect(bytes == packedBefore, "packed weight changed");
    }
}
int main() {
#ifdef CUDA_NO_TENSOR_CORE
    return 77;
#endif
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess || properties.major != 8 ||
        (properties.minor != 0 && properties.minor != 6))
        return 77;
    try {
        Run(0);
        puts("PASS multirow GEMV dispatch, bias, partial scale block, eager/graph, retained weights, "
             "intermediate Marlin");
    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR %s\n", e.what());
        return 1;
    } catch (const char *e) {
        fprintf(stderr, "ERROR %s\n", e);
        return 1;
    }
}
