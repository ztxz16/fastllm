// Standalone CUDA regression for the disk cache's reversible SM75 layout.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Sync() {
    Require(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution");
}
static void Upload(Data &data) {
    data.ToDevice(DataDevice::CUDA);
    data.Allocate(false);
}
static void CheckCase(int columns, int inner, int rows, bool exact = true) {
    Data weight(DataType::NVFP4_BLOCK_32_E8M0, {columns, inner});
    weight.blockK = 1; weight.blockM = 32;
    weight.Allocate(false);
    for (size_t i = 0; i < weight.GetBytes(); ++i)
        weight.cpuData[i] = i % 17 == 16 ? 120 + (i / 17) % 3 : uint8_t((i * 37) ^ (i >> 5));
    std::vector<uint8_t> original(weight.cpuData, weight.cpuData + weight.GetBytes());
    weight.ToDevice(DataDevice::CUDA);
    void *allocation = weight.cudaData;
    Require(FastllmCudaSetDiskMxfp4MarlinLayout(weight, true), "prepare");
    FastllmCudaSetDiskMxfp4MarlinLayout(weight, false);
    std::vector<uint8_t> restored(original.size());
    FastllmCudaCopyFromDeviceToHost(restored.data(), weight.cudaData, restored.size());
    Require(restored == original && allocation == weight.cudaData, "lossless equal-size round trip");

    Data input(DataType::BFLOAT16, {rows, inner}), reference(DataType::BFLOAT16, {rows, columns});
    Data output(DataType::BFLOAT16, {rows, columns}), bias;
    input.Allocate(false);
    auto *a = reinterpret_cast<__nv_bfloat16 *>(input.cpuData);
    bool convertible = true;
    for (int i = 0; i < rows * inner; ++i) {
        float value = exact ? (i * 13 % 251 - 125) / 256.f : std::sin(i * .71f) * .4f;
        a[i] = __float2bfloat16_rn(value);
        float rounded = __bfloat162float(a[i]);
        convertible &= __half2float(__float2half_rn(rounded)) == rounded;
    }
    input.ToDevice(DataDevice::CUDA);
    Upload(reference); Upload(output);
    Require(FastllmCudaBFloat16MatMulNVFP4Block16E8M0(input, weight, bias, reference, rows, inner, columns), "reference");
    Sync();
    Require(!FastllmCudaDiskMxfp4Enabled(weight), "ordinary weights must not opt in");
    Require(!weight.IsRepacked, "ordinary dispatch changed the weight layout");
    {
        FastllmCudaDiskMxfp4Scope scope(&weight, nullptr);
        {
            FastllmCudaDiskMxfp4Scope nested(nullptr, nullptr);
            Require(!FastllmCudaDiskMxfp4Enabled(weight), "nested scope isolation");
        }
        Require(FastllmCudaDiskMxfp4Enabled(weight), "cache scope");
        Require(FastllmCudaBFloat16MatMulNVFP4Block16E8M0(
            input, weight, bias, output, rows, inner, columns), "cache dispatch");
        Require(weight.IsRepacked == (rows >= 32 && convertible), "dispatch layout");
    }
    Require(!FastllmCudaDiskMxfp4Enabled(weight), "scope release");
    Sync();
    std::vector<__nv_bfloat16> expected(size_t(rows) * columns), actual(expected.size());
    FastllmCudaCopyFromDeviceToHost(expected.data(), reference.cudaData, expected.size() * 2);
    FastllmCudaCopyFromDeviceToHost(actual.data(), output.cudaData, actual.size() * 2);
    size_t differences = 0;
    float maxError = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float x = __bfloat162float(actual[i]), y = __bfloat162float(expected[i]);
        Require(std::isfinite(x), "finite result");
        differences += x != y;
        maxError = std::max(maxError, std::abs(x - y));
        // Different FP32 summation orders may straddle one BF16 rounding bin.
        if (!exact) Require(std::abs(x - y) <= std::abs(y) * .008f + 1e-6f, "FP32 numerical tolerance");
    }
    if (differences)
        fprintf(stderr, "M=%d N=%d K=%d differences=%zu max_error=%g\n", rows, columns, inner, differences, maxError);
    // These dyadic operands fit an exact FP32 dot even at the largest K here.
    if (exact) Require(differences == 0, "projection differs from exact FP32 dot");
    FastllmCudaSetDiskMxfp4MarlinLayout(weight, false);
    FastllmCudaCopyFromDeviceToHost(restored.data(), weight.cudaData, restored.size());
    Require(restored == original && allocation == weight.cudaData, "layout after execution");
    printf("PASS M=%d N=%d K=%d differences=%zu max_error=%g\n", rows, columns, inner, differences, maxError);
}

static void CheckFallbacks() {
    Data weight(DataType::NVFP4_BLOCK_32_E8M0, {128, 128});
    weight.blockK = 1; weight.blockM = 32; weight.Allocate(false);
    std::vector<uint8_t> bytes(weight.GetBytes(), 0x37), copy(bytes.size());
    for (size_t i = 16; i < bytes.size(); i += 17) bytes[i] = 122;
    weight.ToDevice(DataDevice::CUDA, {}, false); weight.Allocate(false);
    for (int scale : {0, 112, 141, 255}) {
        bytes[16] = scale;
        FastllmCudaCopyFromHostToDevice(weight.cudaData, bytes.data(), bytes.size());
        Require(!FastllmCudaSetDiskMxfp4MarlinLayout(weight, true) && !weight.IsRepacked, "unsafe scale");
        FastllmCudaCopyFromDeviceToHost(copy.data(), weight.cudaData, copy.size());
        Require(copy == bytes, "fallback changed weight");
    }
    bytes[16] = 122;
    Data input(DataType::BFLOAT16, {32, 128}), output(DataType::BFLOAT16, {32, 128});
    Upload(input); Upload(output);
    std::vector<__nv_bfloat16> a(32 * 128, __float2bfloat16_rn(1));
    for (float bad : {1e10f, 1e-10f, INFINITY, NAN}) {
        FastllmCudaCopyFromHostToDevice(weight.cudaData, bytes.data(), bytes.size());
        Require(FastllmCudaSetDiskMxfp4MarlinLayout(weight, true), "fallback prepare");
        a[7] = __float2bfloat16_rn(bad);
        FastllmCudaCopyFromHostToDevice(input.cudaData, a.data(), a.size() * 2);
        Require(!FastllmCudaTryDiskMxfp4Marlin(input.cudaData, weight, output.cudaData, 32), "unsafe input");
        Require(!weight.IsRepacked, "fallback must restore native");
        FastllmCudaCopyFromDeviceToHost(copy.data(), weight.cudaData, copy.size());
        Require(copy == bytes, "input fallback corrupted weight");
    }
    for (int kind = 0; kind < 3; ++kind) {
        int scale = kind == 0 ? 140 : kind == 1 ? 113 : 122;
        float value = kind == 2 ? std::ldexp(1.f, -24) : 1.f;
        std::fill(bytes.begin(), bytes.end(), kind == 1 ? 0x11 : 0x77);
        for (size_t i = 16; i < bytes.size(); i += 17) bytes[i] = scale;
        std::fill(a.begin(), a.end(), __float2bfloat16_rn(value));
        FastllmCudaCopyFromHostToDevice(weight.cudaData, bytes.data(), bytes.size());
        FastllmCudaCopyFromHostToDevice(input.cudaData, a.data(), a.size() * 2);
        Require(FastllmCudaTryDiskMxfp4Marlin(input.cudaData, weight, output.cudaData, 32), "numeric range");
        Sync();
        std::vector<__nv_bfloat16> result(a.size());
        FastllmCudaCopyFromDeviceToHost(result.data(), output.cudaData, result.size() * 2);
        float expected = 128 * (kind == 1 ? .5f : 6.f) * std::ldexp(value, scale - 127);
        for (auto x : result) Require(__bfloat162float(x) == expected, "subnormal operands / large BF16 result");
        FastllmCudaSetDiskMxfp4MarlinLayout(weight, false);
    }
    printf("PASS unsafe scale/input fallback\n");
}

int main() {
#ifdef CUDA_NO_TENSOR_CORE
    return 77;
#else
    try {
        int devices = 0;
        cudaGetDeviceCount(&devices);
        bool tested = false;
        for (int device = 0; device < devices; ++device) {
            cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, device);
            if (prop.major != 7 || prop.minor != 5) continue;
            tested = true;
            FastllmCudaSetDevice(device);
            for (int rows = 1; rows <= 32; ++rows) {
                CheckCase(128, 128, rows);
                // A full tile followed by every small Marlin tail.
                CheckCase(128, 128, 64 + rows);
            }
            for (int rows : {33, 47, 48, 49, 63, 64, 65, 127, 128, 255, 256, 257})
                for (auto shape : {std::pair<int, int>{64, 128}, {128, 64}, {256, 256}})
                    CheckCase(shape.first, shape.second, rows);
            for (int rows : {32, 128, 256}) {
                CheckCase(4096, 7168, rows);
                CheckCase(7168, 2048, rows);
                CheckCase(4608, 5120, rows);
                CheckCase(5120, 2304, rows);
            }
            CheckCase(256, 256, 47, false);
            CheckCase(256, 256, 256, false);
            CheckFallbacks();
        }
        if (!tested) return 77;
        puts("PASS disk MXFP4 Marlin");
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
#endif
}
