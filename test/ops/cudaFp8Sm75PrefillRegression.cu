#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
__global__ void AddReferenceBias(half *output, const half *bias, int width) {
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        size_t index = size_t(blockIdx.x) * width + col;
        output[index] = __hadd(output[index], bias[col]);
    }
}
void Expect(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void CheckBlas(cublasStatus_t status) {
    Expect(status == CUBLAS_STATUS_SUCCESS, "reference cuBLAS failed");
}

uint16_t HalfBits(half value) {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::vector<half> Read(const void *ptr, size_t count) {
    std::vector<half> result(count);
    Check(cudaMemcpy(result.data(), ptr, count * sizeof(half), cudaMemcpyDeviceToHost));
    return result;
}

void Run(int gpu) {
    using namespace fastllm;
    FastllmCudaSetDevice(gpu);
    Executor executor;
    executor.SetFirstDevice("cuda:" + std::to_string(gpu));
    size_t capacity = 0;
    void *arena = FastllmCudaGetFlashInferFloatWorkspace(&capacity);
    Expect(arena != nullptr, "missing float workspace");
    // Explicit process configurations below exercise the full-weight, sliced,
    // and too-small workspace routes, independently of the production policy.
    Expect(capacity == (4ULL << 20) || capacity == (32ULL << 20) ||
           capacity == (64ULL << 20) || capacity == (256ULL << 20),
           "run with FT_FLOAT_WORKSPACE_SIZE=4M,32M,64M,256M");
    const char *flag = std::getenv("FASTLLM_CUDA_FP8_PREFILL_CUBLAS");
    bool disabled = flag && std::strcmp(flag, "0") == 0;
    cublasHandle_t handle;
    void *blasWorkspace;
    Check(cudaMalloc(&blasWorkspace, 8ULL << 20));
    CheckBlas(cublasCreate(&handle));
    CheckBlas(cublasSetStream(handle, cudaStreamPerThread));
    CheckBlas(cublasSetMathMode(handle, cublasMath_t(
        CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    CheckBlas(cublasSetWorkspace(handle, blasWorkspace, 8ULL << 20));

    struct Shape { int n, k; };
    for (Shape shape : {Shape{1024, 1024}, Shape{4160, 4224}, Shape{512, 512}}) {
        for (bool withBias : {false, true}) {
            int n = shape.n, k = shape.k;
            std::mt19937 rng(n + k);
            std::vector<uint8_t> original(size_t(n) * k);
            for (auto &v : original) v = (rng() % 2) * 128 + rng() % 127;
            Data weight(FP8_E4M3, {n, k});
            weight.blockM = weight.blockK = 128;
            weight.scales.resize(((n + 127) / 128) * (k / 128));
            for (auto &s : weight.scales) {
                s = std::ldexp(.5f + (rng() % 512) / 1024.f, int(rng() % 8) - 12);
            }
            weight.Allocate();
            std::memcpy(weight.cpuData, original.data(), original.size());
            weight.ToDevice(CUDA, std::vector<int>{gpu});
            Data bias(FLOAT32);
            if (withBias) {
                std::vector<float> values(n);
                for (auto &v : values) v = (int(rng() % 65) - 32) / 256.f;
                bias.Resize({n});
                bias.Allocate();
                std::memcpy(bias.cpuData, values.data(), values.size() * sizeof(float));
                bias.ToDevice(CUDA, std::vector<int>{gpu});
            }
            std::vector<float> values(size_t(2048) * k);
            for (auto &v : values) v = (int(rng() % 2049) - 1024) / 1024.f;
            Data input(FLOAT16, {2048, k}, values);
            Data actual(FLOAT16, {2048, n}, std::vector<float>(size_t(2048) * n, 0));
            Data expected(FLOAT16, {2048, n}, std::vector<float>(size_t(2048) * n, 0));
            for (Data *data : {&input, &actual, &expected}) data->ToDevice(CUDA, std::vector<int>{gpu});
            FastllmCudaSetNcclForceSync(true);
            Expect(FastllmCudaHalfMatMulFloatFP8E4M3(input, weight, bias, actual, 4, k, n), "warmup failed");
            Check(cudaDeviceSynchronize());
            FastllmCudaSetNcclForceSync(false);
            Expect(FastllmCudaHasFp8MarlinLayout(weight), "weight was not packed");

            // Independent CPU dequant reference from the original FP8 bytes,
            // including the rounded block scale used by the Marlin layout.
            std::vector<half> fp16(original.size());
            for (int row = 0; row < n; ++row) for (int col = 0; col < k; ++col) {
                uint8_t q = original[size_t(row) * k + col];
                uint16_t bits = ((q & 128) << 8) | ((q & 127) << 7);
                half decoded;
                std::memcpy(&decoded, &bits, sizeof(bits));
                half scale = __float2half_rn(weight.scales[size_t(row / 128) * (k / 128) + col / 128] * 256.f);
                fp16[size_t(row) * k + col] = __float2half_rn(__half2float(decoded) * __half2float(scale));
            }
            half *referenceWeight;
            Check(cudaMalloc(&referenceWeight, fp16.size() * sizeof(half)));
            Check(cudaMemcpy(referenceWeight, fp16.data(), fp16.size() * sizeof(half), cudaMemcpyHostToDevice));
            auto reference = [&](int rows, bool useCublas) {
                if (useCublas) {
                    int chunk = std::min(n, int(capacity / (size_t(k) * 2)) / 64 * 64);
                    half alpha = __float2half_rn(1), beta = __float2half_rn(0);
                    for (int offset = 0; offset < n; offset += chunk) {
                        int count = std::min(chunk, n - offset);
                        CheckBlas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            count, rows, k, &alpha, referenceWeight + size_t(offset) * k, CUDA_R_16F, k,
                            input.cudaData, CUDA_R_16F, k, &beta, (half *)expected.cudaData + offset, CUDA_R_16F, n,
                            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    }
                } else {
                    Expect(FastllmCudaMarlinHalfFP8Gemm((half *)input.cudaData,
                        (uint32_t *)weight.cudaData, (half *)weight.extraCudaHalfData[1],
                        (half *)expected.cudaData, rows, n, k, 128, (int *)weight.extraCudaData[3]), "reference Marlin failed");
                }
                if (withBias) AddReferenceBias<<<rows, 256, 0, cudaStreamPerThread>>>(
                    (half *)expected.cudaData, (half *)weight.extraCudaHalfData[0], n);
            };
            for (int rows : {1, 4, 960, 1023, 1024, 1025, 1535, 1536, 2048}) {
                bool useCublas = !disabled && n != 512 && rows >= 1024;
                if (n == 4160 && capacity == (4ULL << 20)) useCublas = false;
                if (n == 4160 && capacity == (32ULL << 20) && rows < 1536) useCublas = false;
                Check(cudaMemsetAsync(arena, 0xa5, capacity, cudaStreamPerThread));
                Expect(FastllmCudaHalfMatMulFloatFP8E4M3(input, weight, bias, actual, rows, k, n), "dispatch failed");
                auto output = Read(actual.cudaData, size_t(rows) * n);
                auto first = Read(arena, 32);
                bool overwritten = false;
                for (half value : first) {
                    overwritten |= HalfBits(value) != 0xa5a5;
                }
                Expect(overwritten == useCublas, "wrong route: workspace was unexpectedly touched or left untouched");
                auto guard = Read((char *)arena + capacity - 256, 128);
                for (half value : guard) {
                    Expect(HalfBits(value) == 0xa5a5, "scratch capacity exceeded");
                }
                if (rows > 1) {
                    reference(rows, useCublas);
                    auto ref = Read(expected.cudaData, output.size());
                    Expect(std::memcmp(output.data(), ref.data(), output.size() * sizeof(half)) == 0, "output differs from selected backend reference");
                } else {
                    for (half value : output) Expect(std::isfinite(__half2float(value)), "GEMV nonfinite output");
                }
                std::cout << "PASS N=" << n << " K=" << k << " M=" << rows
                          << " bias=" << withBias << " backend=" << (useCublas ? "cublas" : rows == 1 ? "gemv" : "marlin") << std::endl;
            }
            // Capture must keep Marlin and must not overwrite shared attention scratch.
            reference(1024, false);
            auto ref = Read(expected.cudaData, size_t(1024) * n);
            Check(cudaMemsetAsync(arena, 0xa5, capacity, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal));
            Expect(FastllmCudaHalfMatMulFloatFP8E4M3(input, weight, bias, actual, 1024, k, n), "capture dispatch failed");
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&executable, graph, 0));
            Check(cudaGraphLaunch(executable, cudaStreamPerThread));
            auto output = Read(actual.cudaData, ref.size());
            Expect(!std::memcmp(output.data(), ref.data(), ref.size() * 2), "captured output differs from Marlin");
            for (half value : Read(arena, 32)) {
                Expect(HalfBits(value) == 0xa5a5, "capture used dequant scratch");
            }
            Check(cudaGraphExecDestroy(executable));
            Check(cudaGraphDestroy(graph));
            Check(cudaFree(referenceWeight));
            size_t currentCapacity = 0;
            Expect(FastllmCudaGetFlashInferFloatWorkspace(&currentCapacity) == arena && currentCapacity == capacity, "workspace moved or grew");
        }
    }
    CheckBlas(cublasDestroy(handle));
    Check(cudaFree(blasWorkspace));
}
} // namespace

int main(int argc, char **argv) {
#ifdef CUDA_NO_TENSOR_CORE
    return 77;
#endif
    int gpu = argc > 1 ? std::atoi(argv[1]) : 0;
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, gpu) != cudaSuccess ||
        properties.major != 7 || properties.minor != 5) {
        return 77;
    }
    try {
        Run(gpu);
        std::cout << "SM75 FP8 prefill regression PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const char *error) {
        std::cerr << error << '\n';
        return 1;
    }
}
