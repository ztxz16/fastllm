#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static std::vector<unsigned char> Read(const Data &data) {
    std::vector<unsigned char> bytes(data.GetBytes());
    Check(cudaMemcpy(bytes.data(), data.cudaData, bytes.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
          "CUDA read failed");
    return bytes;
}
static void Fill(Data &data, DataType type, const std::vector<int> &dims, int seed) {
    size_t count = 1;
    for (int d : dims) count *= d;
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i)
        values[i] = std::sin(float(i + seed) * .01731f) * std::cos(float(i * 7 + seed) * .037f);
    data.CopyFrom(Data(FLOAT32, dims, values));
    ToDataType(data, type);
    data.ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()});
}
static void Reference(Data &input, Data &weight, Data &output, bool transpose, float alpha) {
    for (int row = 0; row < input.dims[1]; ++row) {
        Data x, y;
        Split(input, 1, row, row + 1, x);
        if (transpose) MatMulTransB(x, weight, y, alpha);
        else MatMul(x, weight, y, alpha);
        if (row == 0) {
            output.CopyFrom(y);
            if (input.dims[1] > 1) output.Expansion({input.dims[0], input.dims[1], y.dims[2]});
        } else CatDirect(output, y, 1);
    }
    output.expansionDims.clear();
}
static void Run(DataType type, int heads, int rows, int inner, int width, bool transpose,
                bool benchmark = false) {
    Data input, weight, expected, actual;
    Fill(input, type, {heads, rows, inner}, 17);
    Fill(weight, type, transpose ? std::vector<int>{heads, width, inner} :
         std::vector<int>{heads, inner, width}, 53);
    const auto inputBefore = Read(input), weightBefore = Read(weight);
    const float alpha = rows % 2 ? 1.f : .75f;
    Reference(input, weight, expected, transpose, alpha);
    Check(FastllmCudaBatchMatMulSingleRows(input, weight, actual, transpose, alpha),
          "single-row projection rejected dense input");
    Check(actual.dims == expected.dims && Read(actual) == Read(expected),
          "single-row projection differs from Split/MatMul/Cat reference");
    Check(Read(input) == inputBefore && Read(weight) == weightBefore, "input or weight mutated");
    // Reuse an allocated output at the next invocation too.
    Check(FastllmCudaBatchMatMulSingleRows(input, weight, actual, transpose, alpha) &&
          Read(actual) == Read(expected), "reused output differs");
    std::printf("type=%d heads=%d rows=%d inner=%d width=%d transpose=%d: bitwise equal\n",
                int(type), heads, rows, inner, width, transpose);
    if (benchmark) {
        for (bool direct : {false, true}) {
            auto start = std::chrono::steady_clock::now();
            constexpr int repeats = 100;
            for (int i = 0; i < repeats; ++i) {
                Data result;
                if (direct) Check(FastllmCudaBatchMatMulSingleRows(input, weight, result, transpose), "projection failed");
                else Reference(input, weight, result, transpose, 1.f);
            }
            Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA benchmark failed");
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / repeats;
            std::printf("BENCH rows=%d transpose=%d direct=%d host_wall_us=%.3f\n", rows, transpose, direct, us);
        }
    }
}
int main(int argc, char **argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try {
        const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
        for (int device = 0; device < count; ++device) {
            FastllmCudaSetDevice(device);
            static_cast<Executor *>(GetExecutor())->SetFirstDevice("cuda:" + std::to_string(device));
            for (DataType type : {FLOAT32, FLOAT16, BFLOAT16})
                for (int rows : {1,2,3,4,7,9,17})
                    for (bool transpose : {false, true}) Run(type, 3, rows, 32, 48, transpose);
            for (int heads : {32,64})
                for (int rows : {2,3,4,7,9,17}) {
                    Run(BFLOAT16, heads, rows, 256, 512, false, benchmark && rows == 3);
                    Run(BFLOAT16, heads, rows, 512, 256, true, benchmark && rows == 3);
                }
            Run(BFLOAT16, 1, 7, 17, 33, false);
            Run(FLOAT32, 5, 9, 33, 17, true);
            Data input(FLOAT32, {1,2,3}), weight(FLOAT32, {1,3,4}), output;
            Check(!FastllmCudaBatchMatMulSingleRows(input, weight, output, false) && output.dims.empty(),
                  "unsupported input must fall back without changing output");
        }
        std::puts("PASS CUDA single-row batched MatMul on all devices");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
