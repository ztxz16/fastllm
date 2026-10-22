#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
static void Check(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
static uint16_t Bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}
static float Float(uint16_t x) {
    uint32_t u = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static float Fp8(float x) {
    static const std::vector<float> grid = [] {
        std::vector<float> g(127);
        for (int i = 0; i <= 126; ++i)
            g[i] = i < 8 ? std::ldexp(float(i), -9) : std::ldexp(1.0f + float(i & 7) / 8, (i >> 3) - 7);
        return g;
    }();
    const float a = std::abs(x);
    auto hi = std::lower_bound(grid.begin(), grid.end(), a);
    int code = hi - grid.begin();
    if (hi == grid.end())
        code = 126;
    else if (code > 0) {
        const float lower = a - grid[code - 1], upper = grid[code] - a;
        if (lower < upper || (lower == upper && (code & 1)))
            --code;
    }
    return std::copysign(grid[code], x);
}
static std::vector<uint16_t> Read(const Data &d) {
    std::vector<uint16_t> a(d.Count(0));
    Check(cudaMemcpy(a.data(), d.cudaData, a.size() * 2, cudaMemcpyDeviceToHost));
    return a;
}
static void Replay(const std::function<void()> &fn) {
    cudaGraph_t g;
    cudaGraphExec_t e;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    fn();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &g));
    Check(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    for (int i = 0; i < 4; ++i)
        Check(cudaGraphLaunch(e, cudaStreamPerThread));
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    Check(cudaGraphExecDestroy(e));
    Check(cudaGraphDestroy(g));
}
static double CheckCase(int rows, int mid, float amplitude, float limit, bool weighted) {
    std::vector<float> values(size_t(rows) * mid * 2), scales(rows);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = Float(Bf16(amplitude * std::sin(float((i * 37 + 13) % 1031))));
        if (i % 131 == 0)
            values[i] = Float(uint16_t(i % 2 ? 0x8001 : 0x0001));
        if (i % 137 == 0)
            values[i] = -0.0f;
    }
    for (int i = 0; i < rows; ++i)
        scales[i] = (i % 5 - 2) * 0.37f;
    float *weights = nullptr;
    Check(cudaMalloc(&weights, rows * 4));
    Check(cudaMemcpy(weights, scales.data(), rows * 4, cudaMemcpyHostToDevice));
    Data input(DataType::FLOAT32, {rows, 2 * mid}, values), output;
    ToDataType(input, DataType::BFLOAT16);
    input.ToDevice(DataDevice::CUDA, {0});
    auto run = [&] {
        Require(
            FastllmCudaDeepSeekV41SharedSwigluQuantized(input, limit, output, weighted ? weights : nullptr),
            "SwiGLU rejected");
    };
    run();
    auto got = Read(output);
    std::vector<float> expected(size_t(rows) * mid);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < mid; ++col) {
            long double gate = values[size_t(row) * 2 * mid + col],
                        up = values[size_t(row) * 2 * mid + mid + col];
            if (limit > 0) {
                gate = std::min(gate, (long double)limit);
                up = std::max(-(long double)limit, std::min(up, (long double)limit));
            }
            // The production contract evaluates expf before division; its
            // overflow at large negative gates intentionally yields zero.
            long double value = -gate > std::log((long double)std::numeric_limits<float>::max())
                                    ? 0.0L
                                    : gate / (1 + std::exp(-gate)) * up;
            if (weighted)
                value *= scales[row];
            expected[size_t(row) * mid + col] = Float(Bf16(float(value)));
        }
    double err = 0, norm = 0;
    for (size_t base = 0; base < expected.size(); base += 32) {
        float maximum = 1e-4f;
        for (int j = 0; j < 32; ++j)
            maximum = std::max(maximum, std::abs(expected[base + j]));
        float scale = std::exp2(std::ceil(std::log2(maximum / 448.0f)));
        for (int j = 0; j < 32; ++j) {
            float ref = Float(Bf16(Fp8(expected[base + j] / scale) * scale)), actual = Float(got[base + j]);
            Require(std::isfinite(actual), "nonfinite SwiGLU result");
            err += double(actual - ref) * (actual - ref);
            norm += double(ref) * ref;
        }
    }
    double relative = std::sqrt(err / std::max(norm, 1e-30));
    Require(relative < 0.002, "SwiGLU differs from independent long-double oracle");
    if (mid == 2304 && amplitude == 2) {
        Replay(run);
        Require(Read(output) == got, "SwiGLU graph mismatch");
        Require(FastllmCudaDeepSeekV41BFloat16SwigluQuantized(input.cudaData, output.cudaData, rows, mid,
                                                              limit, weighted ? weights : nullptr),
                "raw SwiGLU rejected");
        Require(Read(output) == got, "raw SwiGLU changed route weighting");
        void *unalignedIn = nullptr, *unalignedOut = nullptr;
        Check(cudaMalloc(&unalignedIn, input.GetBytes() + 2));
        Check(cudaMalloc(&unalignedOut, output.GetBytes() + 2));
        Check(
            cudaMemcpy((char *)unalignedIn + 2, input.cudaData, input.GetBytes(), cudaMemcpyDeviceToDevice));
        Require(FastllmCudaDeepSeekV41BFloat16SwigluQuantized((char *)unalignedIn + 2,
                                                              (char *)unalignedOut + 2, rows, mid, limit,
                                                              weighted ? weights : nullptr),
                "unaligned SwiGLU fallback rejected");
        std::vector<uint16_t> fallback(got.size());
        Check(
            cudaMemcpy(fallback.data(), (char *)unalignedOut + 2, output.GetBytes(), cudaMemcpyDeviceToHost));
        Require(fallback == got, "unaligned SwiGLU fallback differs");
        Check(cudaFree(unalignedIn));
        Check(cudaFree(unalignedOut));
    }
    if (!weighted) {
        Require(FastllmCudaDeepSeekV41SharedSwiglu(input, limit, output), "unquantized SwiGLU rejected");
        const auto plain = Read(output);
        for (size_t i = 0; i < plain.size(); ++i)
            Require(std::abs(Float(plain[i]) - expected[i]) <=
                        std::max(0.008f * std::abs(expected[i]), std::numeric_limits<float>::min()),
                    "unquantized SwiGLU exceeds one BF16 ULP or FP32 underflow");
    }
    Check(cudaFree(weights));
    return relative;
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    try {
        SetCudaEmbedding(false);
        SetThreads(1);
        FastllmCudaSetDevice(0);
        int cases = 0;
        double worst = 0;
        for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 128})
            for (int mid : {32, 64, 96, 256, 2304, 4096})
                for (float amplitude : {0.0f, 0.01f, 2.0f, 128.0f})
                    for (float limit : {0.0f, 7.0f})
                        for (bool weighted : {false, true}) {
                            worst = std::max(worst, CheckCase(rows, mid, amplitude, limit, weighted));
                            ++cases;
                        }
        printf("PASS %d SwiGLU cases; worst oracle NRMSE %.9g; graph checks passed\n", cases, worst);
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
