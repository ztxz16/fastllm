// BF16 fused projections: FP64 oracle, exact packing, changing-input graph replay.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-linear-output.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *s) {
    if (!ok)
        throw std::runtime_error(s);
}
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static float Float(uint16_t x) {
    uint32_t b = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
static uint16_t Bf16(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    b += 0x7fff + ((b >> 16) & 1);
    return b >> 16;
}
static void Alloc(Data &x, DataType type, std::vector<int> dims, int device) {
    x.dataType = type;
    x.UpdateUnitSize();
    x.dataDevice = CUDA;
    x.dataDeviceIds = {device};
    x.Resize(dims);
    x.Allocate(false);
}
static void Upload(Data &x, const void *p, size_t bytes) {
    Cuda(cudaMemcpyAsync(x.cudaData, p, bytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
}
static std::vector<float> Read(Data &x) {
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<float> r(x.Count(0));
    Cuda(cudaMemcpy(r.data(), x.cudaData, r.size() * 4, cudaMemcpyDeviceToHost));
    return r;
}
static bool quickMode = false;
static double worstRelative = 0, worstAbsolute = 0, worstOldRelative = 0, worstOldAbsolute = 0;
static void Run(int device, int rows, int k, int n1, int n2) {
    Data input, xf, w1, w2, packed, y1, y2, scratch, old1, old2;
    Alloc(input, BFLOAT16, {1, rows, k}, device);
    Alloc(xf, FLOAT32, {1, rows, k}, device);
    Alloc(w1, FLOAT32, {n1, k}, device);
    if (n2)
        Alloc(w2, FLOAT32, {n2, k}, device);
    Alloc(old1, FLOAT32, {1, rows, n1}, device);
    if (n2)
        Alloc(old2, FLOAT32, {1, rows, n2}, device);
    std::mt19937 rng(153 + rows + k + n1);
    std::normal_distribution<float> normal;
    std::vector<uint16_t> x(rows * k);
    std::vector<float> a(n1 * k), b(n2 * k), xfloat(x.size());
    for (auto &v : x)
        v = Bf16(normal(rng));
    for (auto &v : a)
        v = Float(Bf16(normal(rng) * .125f));
    for (auto &v : b)
        v = Float(Bf16(normal(rng) * .125f));
    Upload(input, x.data(), x.size() * 2);
    Upload(w1, a.data(), a.size() * 4);
    if (n2)
        Upload(w2, b.data(), b.size() * 4);
    Check(FastllmCudaTryPackBFloat16Weights(w1, n2 ? &w2 : nullptr, packed),
          "packing rejected exact BF16 values");
    std::vector<uint16_t> packing(a.size() + b.size());
    Cuda(cudaMemcpy(packing.data(), packed.cudaData, packing.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < packing.size(); i++)
        Check(Float(packing[i]) == (i < a.size() ? a[i] : b[i - a.size()]), "packing layout changed");
    auto call = [&] {
        Check(n2 ? FastllmCudaBFloat16LinearPairFloat32(input, packed, n1, y1, y2, scratch)
                 : FastllmCudaBFloat16LinearFloat32(input, packed, y1),
              "projection rejected");
    };
    call();
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    call();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; replay++) {
        std::rotate(x.begin(), x.begin() + 1, x.end());
        Upload(input, x.data(), x.size() * 2);
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        auto actual = Read(y1);
        if (n2) {
            auto z = Read(y2);
            actual.insert(actual.end(), z.begin(), z.end());
        }
        for (size_t i = 0; i < x.size(); i++)
            xfloat[i] = Float(x[i]);
        Upload(xf, xfloat.data(), xfloat.size() * 4);
        Check(FastllmCudaMatMulFloat32(xf, w1, Data(), old1, rows, k, n1), "old wkv");
        auto old = Read(old1);
        if (n2) {
            Check(FastllmCudaMatMulFloat32(xf, w2, Data(), old2, rows, k, n2), "old wgate");
            auto z = Read(old2);
            old.insert(old.end(), z.begin(), z.end());
        }
        double sumDiff = 0, sumRef = 0, maxRef = 0, maxError = 0, oldDiff = 0, oldError = 0;
        for (int branch = 0; branch < (n2 ? 2 : 1); branch++) {
            auto &w = branch ? b : a;
            int n = branch ? n2 : n1, offset = branch ? rows * n1 : 0;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < n; c++) {
                    double ref = 0;
                    for (int j = 0; j < k; j++)
                        ref += double(Float(x[r * k + j])) * w[c * k + j];
                    double err = actual[offset + r * n + c] - ref;
                    Check(std::isfinite(err), "nonfinite output");
                    double oldErr = old[offset + r * n + c] - ref;
                    oldDiff += oldErr * oldErr;
                    oldError = std::max(oldError, std::abs(oldErr));
                    sumDiff += err * err;
                    sumRef += ref * ref;
                    maxRef = std::max(maxRef, std::abs(ref));
                    maxError = std::max(maxError, std::abs(err));
                }
        }
        double relative = std::sqrt(sumDiff / std::max(sumRef, 1e-30));
        worstOldRelative = std::max(worstOldRelative, std::sqrt(oldDiff / std::max(sumRef, 1e-30)));
        worstOldAbsolute = std::max(worstOldAbsolute, oldError);
        worstRelative = std::max(worstRelative, relative);
        worstAbsolute = std::max(worstAbsolute, maxError);
        // Thousands of FP32 additions can differ by a few ppm from FP64.
        if (!(relative < 1e-5 && maxError < 1e-5 * std::max(maxRef, 1.0))) {
            std::cerr << "FAIL device=" << device << " M=" << rows << " K=" << k << " N=" << n1 << "+" << n2
                      << " replay=" << replay << " max_error=" << maxError << " max_ref=" << maxRef
                      << " relative=" << relative << std::endl;
            for (int c = 0; c < std::min(n1, 7); c++) {
                double r = 0;
                for (int j = 0; j < k; j++)
                    r += double(Float(x[j])) * a[c * k + j];
                std::cerr << c << " actual=" << actual[c] << " ref=" << r << std::endl;
            }
            Check(false, "FP64 oracle exceeded FP32 accumulation tolerance");
        }
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
    // Output views can overlap an input at a nonzero offset: reject them before allocation.
    Data view;
    view.FakeFrom(w1, sizeof(float));
    Check(!FastllmCudaTryPackBFloat16Weights(w1, n2 ? &w2 : nullptr, view),
          "non-owning packed output accepted");
    Check(!FastllmCudaTryPackBFloat16Weights(w1, nullptr, w1), "aliased packed output accepted");
    if (n2) {
        view.FakeFrom(y1, sizeof(float));
        Check(!FastllmCudaBFloat16LinearPairFloat32(input, packed, n1, view, y2, scratch),
              "non-owning split output accepted");
        Check(!FastllmCudaBFloat16LinearPairFloat32(input, packed, n1, y1, y2, view),
              "non-owning scratch accepted");
        const auto ids = input.dataDeviceIds;
        input.dataDeviceIds = packed.dataDeviceIds = {device + 1};
        Check(!FastllmCudaBFloat16LinearPairFloat32(input, packed, n1, y1, y2, scratch),
              "wrong current device accepted");
        input.dataDeviceIds = packed.dataDeviceIds = ids;
    }
    // Failed preparation is eager-only, and does not invalidate a caller's capture.
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Data rejected;
    Check(!FastllmCudaTryPackBFloat16Weights(w1, n2 ? &w2 : nullptr, rejected), "packing accepted capture");
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphDestroy(graph));
    a[0] = .1f;
    Upload(w1, a.data(), a.size() * 4);
    Check(!FastllmCudaTryPackBFloat16Weights(w1, n2 ? &w2 : nullptr, rejected),
          "inexact FP32 weight accepted");
    a[0] = INFINITY;
    Upload(w1, a.data(), a.size() * 4);
    Check(!FastllmCudaTryPackBFloat16Weights(w1, nullptr, rejected), "nonfinite weight accepted");
    if (n2) {
        Check(!FastllmCudaBFloat16LinearPairFloat32(input, packed, 0, y1, y2, scratch),
              "invalid split accepted");
        Check(!FastllmCudaBFloat16LinearPairFloat32(input, packed, n1, y1, y1, scratch),
              "aliased outputs accepted");
    }
}
int main(int argc, char **) {
    quickMode = argc > 1;
    try {
        int devices = 0;
        Cuda(cudaGetDeviceCount(&devices));
        if (!devices)
            return 77;
        int cases = 0;
        for (int device = 0; device < (quickMode ? 1 : devices); device++) {
            Cuda(cudaSetDevice(device));
            for (int rows :
                 (quickMode ? std::vector<int>{2, 6, 8} : std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 16, 32})) {
                Run(device, rows, 33, 7, 13);
                ++cases;
                if (device == 0 || rows == 6) {
                    Run(device, rows, 5120, 512, 512);
                    Run(device, rows, 5120, 512, 0);
                    cases += 2;
                }
            }
        }
        std::cout << "PASS " << cases
                  << " cases x3 changing-input graph replay; FP64 oracle max_abs=" << worstAbsolute
                  << " relative_l2=" << worstRelative << " old_max_abs=" << worstOldAbsolute
                  << " old_relative_l2=" << worstOldRelative << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
