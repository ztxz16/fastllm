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
#include <vector>

namespace {
using namespace fastllm;
void Expect(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void Blas(cublasStatus_t status) { Expect(status == CUBLAS_STATUS_SUCCESS, "cuBLAS failed"); }
uint16_t Bits(half value) { uint16_t bits; std::memcpy(&bits, &value, 2); return bits; }
half Half(uint16_t bits) { half value; std::memcpy(&value, &bits, 2); return value; }
float Fp8(uint8_t value) {
    int exponent = (value >> 3) & 15, mantissa = value & 7;
    return exponent ? std::ldexp(1.f + mantissa / 8.f, exponent - 7) : std::ldexp(mantissa / 8.f, -6);
}
std::vector<half> Read(const void *ptr, size_t count) {
    std::vector<half> values(count);
    Check(cudaMemcpy(values.data(), ptr, count * 2, cudaMemcpyDeviceToHost));
    return values;
}
__global__ void CropBias(const half *input, half *output, const half *bias, int n, int packedN) {
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        half value = input[size_t(blockIdx.x) * packedN + col];
        output[size_t(blockIdx.x) * n + col] = bias ? __hadd(value, bias[col]) : value;
    }
}

void Run(int gpu) {
    FastllmCudaSetDevice(gpu);
    Executor executor;
    executor.SetFirstDevice("cuda:" + std::to_string(gpu));
    size_t capacity = 0;
    void *arena = FastllmCudaGetFlashInferFloatWorkspace(&capacity);
    Expect(arena != nullptr, "missing float arena");
    const char *flag = std::getenv("FASTLLM_CUDA_NVFP4_PREFILL_CUBLAS");
    bool disabled = flag && std::strcmp(flag, "0") == 0;
    const char *fp16Flag = std::getenv("FASTLLM_CUDA_NVFP4_PREFILL_FP16_ACCUM");
    bool fp16Accum = fp16Flag && std::strcmp(fp16Flag, "1") == 0;
    int sms;
    Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gpu));
    cublasHandle_t handle;
    void *blasWorkspace;
    float *zero;
    Check(cudaMalloc(&blasWorkspace, 8ULL << 20));
    Check(cudaMalloc(&zero, sizeof(float)));
    Check(cudaMemset(zero, 0, sizeof(float)));
    Blas(cublasCreate(&handle));
    Blas(cublasSetStream(handle, cudaStreamPerThread));
    Blas(cublasSetMathMode(handle, cublasMath_t(
        CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    Blas(cublasSetWorkspace(handle, blasWorkspace, 8ULL << 20));

    // Alternate FP8 (host half alpha/beta) and NVFP4 (device float alpha/beta)
    // in the same process. They must share the arena without leaking handle mode.
    Data fp8Weight(FP8_E4M3, {1024, 1024});
    fp8Weight.blockM = fp8Weight.blockK = 128;
    fp8Weight.scales.resize(64, 1.f / 256);
    fp8Weight.Allocate();
    std::memset(fp8Weight.cpuData, 0x38, 1024 * 1024);
    Data fp8Input(FLOAT16, {2048, 1024}, std::vector<float>(2048 * 1024, .25f));
    Data fp8Output(FLOAT16, {2048, 1024}, std::vector<float>(2048 * 1024, 0));
    Data empty;
    for (Data *data : {&fp8Weight, &fp8Input, &fp8Output}) data->ToDevice(CUDA, std::vector<int>{gpu});
    FastllmCudaSetNcclForceSync(true);
    Expect(FastllmCudaHalfMatMulFloatFP8E4M3(fp8Input, fp8Weight, empty, fp8Output, 4, 1024, 1024), "FP8 warmup failed");
    Check(cudaDeviceSynchronize());
    FastllmCudaSetNcclForceSync(false);
    auto checkFp8 = [&]() {
        Expect(FastllmCudaHalfMatMulFloatFP8E4M3(fp8Input, fp8Weight, empty, fp8Output, 2048, 1024, 1024), "mixed FP8 dispatch failed");
        for (half value : Read(fp8Output.cudaData, 2048 * 1024)) {
            Expect(Bits(value) == Bits(__float2half_rn(1)), "mixed FP8 scalar mode/output corrupted");
        }
    };

    struct Shape { int n, k; };
    const char *filter = std::getenv("TEST_N");
    const int testN = filter ? std::atoi(filter) : 0;
    int cases = 0;
    for (Shape shape : {Shape{8240, 5120}, Shape{7168, 5120}, Shape{5120, 3072},
                       Shape{17408, 5120}, Shape{5120, 8704}, Shape{4160, 1088}, Shape{4161, 1088}, Shape{512, 1024}}) {
        if (filter && shape.n != testN) continue;
        for (bool withBias : {false, true}) {
            int n = shape.n, k = shape.k, groups = k / 16;
            int alignment = n >= 4096 ? 256 : 64;
            int packedN = (n + alignment - 1) / alignment * alignment;
            const float commonScale = .00371f;
            std::mt19937 rng(n + k + 20260910);
            std::vector<uint8_t> original(size_t(n) * groups * 12);
            std::vector<half> cpuWeight(size_t(n) * k);
            for (int row = 0; row < n; ++row) for (int group = 0; group < groups; ++group) {
                uint8_t *p = original.data() + (size_t(row) * groups + group) * 12;
                for (int i = 0; i < 8; ++i) p[i] = rng() % 256;
                // All finite nonnegative FP8 scales, including zero/subnormal.
                float effective = commonScale * Fp8((row + group) % 127);
                std::memcpy(p + 8, &effective, 4);
                half normalized = __float2half_rn(effective / commonScale);
                half shifted = __float2half_rn(__half2float(normalized) * 128.f);
                uint8_t encoded = __half2float(shifted) < 2.f ? 0 : uint8_t(Bits(shifted) >> 7);
                half scale = Half(uint16_t(encoded) << 7);
                for (int j = 0; j < 16; ++j) {
                    int q = (p[j / 2] >> ((j & 1) * 4)) & 15;
                    half decoded = Half(uint16_t(((q & 8) << 12) | ((q & 7) << 9)));
                    cpuWeight[size_t(row) * k + group * 16 + j] =
                        __float2half_rn(__half2float(decoded) * __half2float(scale));
                }
            }
            Data weight(NVFP4_BLOCK_16, {n, k});
            weight.blockM = 16; weight.blockK = 1;
            weight.weightType = WeightType::LINEAR; weight.scales = {commonScale};
            weight.Allocate();
            std::memcpy(weight.cpuData, original.data(), original.size());
            weight.ToDevice(CUDA, std::vector<int>{gpu});
            Data bias(FLOAT32);
            if (withBias) {
                bias.Resize({n}); bias.Allocate();
                auto *values = reinterpret_cast<float *>(bias.cpuData);
                for (int i = 0; i < n; ++i) values[i] = (int(rng() % 65) - 32) / 256.f;
                bias.ToDevice(CUDA, std::vector<int>{gpu});
            }
            constexpr int maxRows = 4096;
            std::vector<float> values(size_t(maxRows) * k);
            for (int row = 0; row < maxRows; ++row) for (int col = 0; col < k; ++col) {
                float value = (int(rng() % 2049) - 1024) / 1024.f;
                values[size_t(row) * k + col] = row % 4 == 0 ? 0 : row % 4 == 1 ? value / 4096 : row % 4 == 2 ? value : value * 16;
            }
            Data input(FLOAT16, {maxRows, k}, values);
            Data actual(FLOAT16, {maxRows, n}, std::vector<float>(size_t(maxRows) * n, 0));
            Data expected(FLOAT16, {maxRows, n}, std::vector<float>(size_t(maxRows) * n, 0));
            Data padded(FLOAT16, {maxRows, packedN}, std::vector<float>(size_t(maxRows) * packedN, 0));
            for (Data *data : {&input, &actual, &expected, &padded}) data->ToDevice(CUDA, std::vector<int>{gpu});
            FastllmCudaSetNcclForceSync(true);
            Expect(FastllmCudaHalfMatMulFloatNVFP4Block16(input, weight, bias, actual, 4, k, n), "NVFP4 warmup failed");
            Check(cudaDeviceSynchronize());
            FastllmCudaSetNcclForceSync(false);
            Expect(FastllmCudaHasNVFP4MarlinLayout(weight), "NVFP4 not repacked");
            auto *packedWeight = static_cast<uint32_t *>(weight.cudaData);
            auto *scales = static_cast<uint8_t *>(weight.cudaData) + size_t(packedN) * k / 2;
            auto *locks = reinterpret_cast<int *>(scales + size_t(packedN) * groups);
            float *globalScale = reinterpret_cast<float *>(locks + sms * 4);
            half *cudaBias = withBias ? static_cast<half *>(weight.extraCudaHalfData[0]) : nullptr;
            half *referenceWeight;
            Check(cudaMalloc(&referenceWeight, cpuWeight.size() * 2));
            Check(cudaMemcpy(referenceWeight, cpuWeight.data(), cpuWeight.size() * 2, cudaMemcpyHostToDevice));
            auto native = [&](int rows) {
                Expect(FastllmCudaMarlinHalfNVFP4Gemm(static_cast<half *>(input.cudaData), packedWeight,
                    scales, globalScale, static_cast<half *>(padded.cudaData), rows, packedN, k, locks, nullptr), "native Marlin failed");
                CropBias<<<rows, 256, 0, cudaStreamPerThread>>>(static_cast<half *>(padded.cudaData),
                    static_cast<half *>(expected.cudaData), cudaBias, n, packedN);
            };
            auto reference = [&](int rows) {
                if (fp16Accum) {
                    Blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
                    half one = __float2half_rn(1), none = __float2half_rn(0);
                    Blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k,
                        &one, referenceWeight, CUDA_R_16F, k, input.cudaData, CUDA_R_16F, k,
                        &none, expected.cudaData, CUDA_R_16F, n, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    // CPU epilogue independently checks FP32 scaling, FP16 rounding,
                    // and bias, including pairs crossing an odd-N row boundary.
                    auto values = Read(expected.cudaData, size_t(rows) * n);
                    std::vector<half> biasValues = withBias ? Read(cudaBias, n) : std::vector<half>();
                    float tensorScale;
                    Check(cudaMemcpy(&tensorScale, globalScale, sizeof(float), cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < values.size(); ++i) {
                        values[i] = __float2half_rn(__half2float(values[i]) * tensorScale);
                        if (withBias) values[i] = __float2half_rn(__half2float(values[i]) + __half2float(biasValues[i % n]));
                    }
                    Check(cudaMemcpy(expected.cudaData, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
                } else {
                    Blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
                    Blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k,
                        globalScale, referenceWeight, CUDA_R_16F, k, input.cudaData, CUDA_R_16F, k,
                        zero, expected.cudaData, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    if (withBias) CropBias<<<rows, 256, 0, cudaStreamPerThread>>>(static_cast<half *>(expected.cudaData),
                        static_cast<half *>(expected.cudaData), cudaBias, n, n);
                }
            };
            for (int rows : {4, 1024, 2047, 2048, 2049, 4096}) {
                bool useCublas = !disabled && n >= 1024 && rows >= 2048 && cpuWeight.size() * 2 <= capacity;
                checkFp8();
                Check(cudaMemsetAsync(arena, 0xa5, capacity, cudaStreamPerThread));
                Expect(FastllmCudaHalfMatMulFloatNVFP4Block16(input, weight, bias, actual, rows, k, n), "dispatch failed");
                auto output = Read(actual.cudaData, size_t(rows) * n);
                bool overwritten = false;
                for (half value : Read(arena, 32)) overwritten |= Bits(value) != 0xa5a5;
                Expect(overwritten == useCublas, "wrong workspace route");
                for (half value : Read(static_cast<char *>(arena) + capacity - 256, 128)) {
                    Expect(Bits(value) == 0xa5a5, "scratch capacity exceeded");
                }
                if (useCublas) {
                    auto decoded = Read(arena, cpuWeight.size());
                    Expect(!std::memcmp(decoded.data(), cpuWeight.data(), decoded.size() * 2), "dequant differs from CPU reference");
                    reference(rows);
                } else native(rows);
                auto ref = Read(expected.cudaData, output.size());
                Expect(!std::memcmp(output.data(), ref.data(), output.size() * 2), "selected backend reference mismatch");
                double err = 0, norm = 0, maxAbs = 0;
                if (useCublas) {
                    native(rows); ref = Read(expected.cudaData, output.size());
                    for (size_t i = 0; i < output.size(); ++i) {
                        double x = __half2float(output[i]), y = __half2float(ref[i]);
                        Expect(std::isfinite(x) && std::isfinite(y), "nonfinite output");
                        err += (x - y) * (x - y); norm += y * y; maxAbs = std::max(maxAbs, std::abs(x - y));
                    }
                    Expect(std::sqrt(err / std::max(norm, 1e-30)) < (fp16Accum ? 0.01 : 0.0002), "accumulation error exceeds tolerance");
                }
                std::cout << "PASS N=" << n << " K=" << k << " M=" << rows << " bias=" << withBias
                          << " backend=" << (useCublas ? "cublas" : "marlin")
                          << " accumulation=" << (fp16Accum && useCublas ? "fp16" : "fp32")
                          << " relative_l2=" << std::sqrt(err / std::max(norm, 1e-30)) << " max_abs=" << maxAbs << std::endl;
                ++cases;
            }
            // Warm original Marlin first so its fallback reduction scratch exists.
            native(2048);
            auto ref = Read(expected.cudaData, size_t(2048) * n);
            Check(cudaMemsetAsync(arena, 0xa5, capacity, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            // Padded graph output uses the existing FastLLM pool. Reserve a
            // fitting block before capture, then use the production pool API.
            void *temporary = FastllmCudaMalloc(size_t(2048) * packedN * sizeof(half));
            Expect(temporary != nullptr, "graph padding warmup failed");
            FastllmCudaFree(temporary);
            Check(cudaDeviceSynchronize());
            void *graph = nullptr; cudaGraphExec_t executable;
            std::vector<void *> reserved;
            Expect(FastllmCudaGraphMemoryPoolBegin(), "graph pool begin failed");
            Expect(FastllmCudaGraphBeginCapture(), "graph begin failed");
            Expect(FastllmCudaHalfMatMulFloatNVFP4Block16(input, weight, bias, actual, 2048, k, n), "capture dispatch failed");
            Expect(FastllmCudaGraphEndCapture(&graph), "graph end failed");
            Expect(FastllmCudaGraphMemoryPoolEnd(reserved), "graph pool end failed");
            Check(cudaGraphInstantiate(&executable, static_cast<cudaGraph_t>(graph), 0));
            Check(cudaGraphLaunch(executable, cudaStreamPerThread));
            auto output = Read(actual.cudaData, ref.size());
            Expect(!std::memcmp(output.data(), ref.data(), ref.size() * 2), "captured output differs from Marlin");
            for (half value : Read(arena, 32)) Expect(Bits(value) == 0xa5a5, "capture used prefill scratch");
            Check(cudaGraphExecDestroy(executable)); FastllmCudaGraphDestroy(graph);
            FastllmCudaGraphMemoryPoolRelease(reserved);
            Check(cudaFree(referenceWeight));
            size_t now = 0;
            Expect(FastllmCudaGetFlashInferFloatWorkspace(&now) == arena && now == capacity, "workspace moved or grew");
        }
    }
    checkFp8();
    Blas(cublasDestroy(handle)); Check(cudaFree(blasWorkspace)); Check(cudaFree(zero));
    Expect(cases > 0, "TEST_N did not select a regression shape");
}
} // namespace

int main(int argc, char **argv) {
#ifdef CUDA_NO_TENSOR_CORE
    return 77;
#endif
    int gpu = argc > 1 ? std::atoi(argv[1]) : 0;
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, gpu) != cudaSuccess || properties.major != 7 || properties.minor != 5) return 77;
    try {
        Run(gpu);
        std::cout << "SM75 NVFP4 prefill regression PASS\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; }
      catch (const char *error) { std::cerr << error << '\n'; }
    return 1;
}
