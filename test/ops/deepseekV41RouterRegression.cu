// Bitwise original CUDA oracle plus FP64, graph replay and real checkpoint fixtures.
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include "devices/cuda/fastllm-v41-decode.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
__global__ void ReferenceRouterScoresKernel(const __nv_bfloat16 *input, const float *weight, float *scores,
                                            int dim, int experts, float invTemp) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int expert = blockIdx.x, token = blockIdx.y;
    float sum = 0.0f;
#pragma unroll 4
    for (int i = tid; i < dim; i += 256)
        sum += __bfloat162float(input[(size_t)token * dim + i]) * weight[(size_t)expert * dim + i];
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        const float other = __shfl_down_sync(0xffffffff, sum, offset);
        if ((lane & (2 * offset - 1)) == 0)
            sum += other;
    }
    __shared__ float partial[8];
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? partial[lane] : 0.0f;
#pragma unroll
        for (int offset = 1; offset < 8; offset *= 2) {
            const float other = __shfl_down_sync(0xffffffff, sum, offset);
            if ((lane & (2 * offset - 1)) == 0)
                sum += other;
        }
        if (lane == 0) {
            const float value = __fmul_rn(__fadd_rn(sum, 0.0f), invTemp);
            const float softplus =
                value > 20.0f ? value : (value < -20.0f ? expf(value) : log1pf(expf(value)));
            scores[(size_t)token * experts + expert] = sqrtf(softplus);
        }
    }
}

static bool RunRouter(const Data &input, Data &weight, float temp, Data &out, bool reference = false) {
    bool ok = FastllmCudaV41RouterScores(input, weight, temp, out);
    if (ok && reference)
        ReferenceRouterScoresKernel<<<dim3(weight.dims[0], input.Count(0) / input.dims.back()), 256, 0,
                                      cudaStreamPerThread>>>(
            (const __nv_bfloat16 *)input.cudaData, (const float *)weight.cudaData, (float *)out.cudaData,
            weight.dims[1], weight.dims[0], fabsf(temp - 1.f) > 1e-6f ? 1.f / temp : 1.f);
    return ok;
}
static void Check(bool v, const char *s) {
    if (!v)
        throw std::runtime_error(s);
}
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static float Bf(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static uint16_t Round(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
static void Alloc(Data &d, DataType t, std::vector<int> dims) {
    d.dataType = t;
    d.UpdateUnitSize();
    d.Resize(dims);
    d.dataDevice = DataDevice::CUDA;
    d.dataDeviceIds = {FastllmCudaGetDevice()};
    d.Allocate();
}
static std::vector<float> Read(Data &d) {
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<float> v(d.Count(0));
    Cuda(cudaMemcpy(v.data(), d.cudaData, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
}
static std::vector<int> Top(const std::vector<float> &v, int m, int n) {
    std::vector<int> out;
    for (int row = 0; row < m; ++row) {
        std::vector<int> ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        std::stable_sort(ids.begin(), ids.end(),
                         [&](int a, int b) { return v[row * n + a] > v[row * n + b]; });
        out.insert(out.end(), ids.begin(), ids.begin() + std::min(8, n));
    }
    return out;
}
static double Validate(const std::vector<float> &y, const std::vector<uint16_t> &x,
                       const std::vector<float> &w, int m, int n, int k, float temp) {
    std::vector<float> expected(m * n);
    double err = 0, norm = 0, mx = 0;
    float inv = fabsf(temp - 1) > 1e-6f ? 1.f / temp : 1.f;
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            double v = 0;
            for (int j = 0; j < k; ++j)
                v += double(Bf(x[row * k + j])) * w[col * k + j];
            v *= inv;
            double ref = sqrt(v > 20 ? v : v < -20 ? exp(v) : log1p(exp(v))), delta = y[row * n + col] - ref;
            expected[row * n + col] = float(ref);
            if (!std::isfinite(y[row * n + col])) {
                std::cerr << "nonfinite row=" << row << " col=" << col << " M=" << m << " N=" << n
                          << " K=" << k << " value=" << y[row * n + col] << std::endl;
            }
            Check(std::isfinite(y[row * n + col]), "nonfinite output");
            err += delta * delta;
            norm += ref * ref;
            mx = std::max(mx, std::abs(delta) / std::max(ref, 1e-3));
        }
    Check(Top(y, m, n) == Top(expected, m, n), "FP64 top-8 mismatch");
    double rms = sqrt(err / std::max(norm, 1e-30));
    Check(rms < 2e-6 && mx < 2e-5, "FP64 score oracle failed");
    return rms;
}
int main(int argc, char **argv) {
    try {
        SetThreads(8);
        FastllmCudaSetDevice(0);
        std::mt19937 gen(20261024);
        std::normal_distribution<float> normal(0, .1f);
        int cases = 0;
        double worst = 0;
        for (int m = 1; m <= 8; ++m)
            for (auto shape : std::vector<std::pair<int, int>>{{17, 16},
                                                               {128, 80},
                                                               {383, 512},
                                                               {383, 5120},
                                                               {128, 5120},
                                                               {384, 5120},
                                                               {1024, 4096},
                                                               {127, 79}}) {
                int n = shape.first, k = shape.second;
                std::vector<uint16_t> x(m * k);
                std::vector<float> w(n * k);
                for (auto &v : x)
                    v = Round(normal(gen));
                for (auto &v : w)
                    v = n == 383 ? normal(gen) : Bf(Round(normal(gen)));
                Data input, weight, output;
                Alloc(input, BFLOAT16, {m, k});
                Alloc(weight, FLOAT32, {n, k});
                Cuda(cudaMemcpy(input.cudaData, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
                Cuda(cudaMemcpy(weight.cudaData, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
                for (float temp : {1.f, .37f, 2.5f}) {
                    Check(RunRouter(input, weight, temp, output, true), "baseline rejected");
                    auto old = Read(output);
                    Check(RunRouter(input, weight, temp, output), "candidate rejected");
                    auto y = Read(output);
                    Check(!memcmp(old.data(), y.data(), y.size() * 4), "bitwise score mismatch");
                    worst = std::max(worst, Validate(y, x, w, m, n, k, temp));
                    Check(Top(old, m, n) == Top(y, m, n), "top-8 order mismatch");
                }
                // Fresh input on every replay; NaN poison ensures every output is overwritten.
                Check(FastllmCudaGraphPrepareCaptureDevice(), "capture prepare");
                void *graph = nullptr, *exec = nullptr;
                Check(FastllmCudaGraphBeginCapture(), "capture begin");
                Check(RunRouter(input, weight, 1, output), "capture operator");
                Check(FastllmCudaGraphEndCapture(&graph), "capture end");
                Check(FastllmCudaGraphInstantiate(graph, &exec), "instantiate");
                for (int rep = 0; rep < 3; ++rep) {
                    for (auto &v : x)
                        v = Round(normal(gen));
                    Cuda(cudaMemcpy(input.cudaData, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
                    Cuda(cudaMemset(output.cudaData, 0xff, output.GetBytes()));
                    Check(FastllmCudaGraphLaunch(exec), "replay");
                    auto y = Read(output);
                    Check(RunRouter(input, weight, 1, output, true), "graph reference");
                    auto ref = Read(output);
                    Check(!memcmp(y.data(), ref.data(), y.size() * 4), "graph bitwise mismatch");
                    worst = std::max(worst, Validate(y, x, w, m, n, k, 1));
                }
                FastllmCudaGraphExecDestroy(exec);
                FastllmCudaGraphDestroy(graph);
                ++cases;
                std::cout << "PASS M=" << m << " N=" << n << " K=" << k << std::endl;
            }
        if (argc > 1) {
            std::ifstream index(std::string(argv[1]) + "/index.txt");
            Check(bool(index), "weight index");
            std::string name;
            int n, k, weights = 0;
            while (index >> name >> n >> k) {
                std::vector<uint16_t> wb(n * k);
                std::vector<float> w(n * k), bias(n), biasVL(n);
                std::ifstream wf(std::string(argv[1]) + "/" + name + ".weight", std::ios::binary);
                wf.read((char *)wb.data(), wb.size() * 2);
                Check(bool(wf), "weight read");
                std::ifstream bf(std::string(argv[1]) + "/" + name + ".bias", std::ios::binary);
                bf.read((char *)bias.data(), bias.size() * 4);
                Check(bool(bf), "bias read");
                std::ifstream vf(std::string(argv[1]) + "/" + name + ".bias_vl", std::ios::binary);
                vf.read((char *)biasVL.data(), biasVL.size() * 4);
                Check(bool(vf), "VL bias read");
                for (size_t i = 0; i < w.size(); ++i)
                    w[i] = Bf(wb[i]);
                Data weight;
                Alloc(weight, FLOAT32, {n, k});
                Cuda(cudaMemcpy(weight.cudaData, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
                for (int m : {2, 6, 8}) {
                    std::vector<uint16_t> x(m * k);
                    for (auto &v : x)
                        v = Round(normal(gen) * 10.f);
                    Data input, out;
                    Alloc(input, BFLOAT16, {m, k});
                    Cuda(cudaMemcpy(input.cudaData, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
                    Check(RunRouter(input, weight, 1, out, true), "checkpoint baseline");
                    auto old = Read(out);
                    Check(RunRouter(input, weight, 1, out), "checkpoint candidate");
                    auto y = Read(out);
                    Check(!memcmp(old.data(), y.data(), y.size() * 4), "checkpoint bitwise mismatch");
                    worst = std::max(worst, Validate(y, x, w, m, n, k, 1));
                    for (auto &b : {bias, biasVL}) {
                        auto a = old, c = y;
                        for (int i = 0; i < m * n; ++i) {
                            a[i] += b[i % n];
                            c[i] += b[i % n];
                        }
                        Check(Top(a, m, n) == Top(c, m, n), "checkpoint biased top-8 mismatch");
                    }
                }
                ++weights;
            }
            std::cout << "PASS " << weights << " checkpoint router weights, M=2/6/8, both routing biases"
                      << std::endl;
        }
        std::cout << "PASS " << cases
                  << " shapes, FP64 + top8 + changing-input graph + bitwise original CUDA reference; max_rms="
                  << worst << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
