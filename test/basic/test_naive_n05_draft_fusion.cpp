#include "fastllm.h"
#include "executor.h"
#include "utils/utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
static void Require(bool x, const char *msg) {
    if (!x)
        throw std::runtime_error(msg);
}
static Data BF(std::vector<int> dims, int seed) {
    Data x(BFLOAT16, dims);
    x.Allocate();
    unsigned state = seed;
    for (int i = 0; i < x.Count(0); ++i) {
        state = 1664525 * state + 1013904223;
        ((uint16_t *)x.cpuData)[i] = Float32ToBFloat16RNEBits(((int)(state >> 16) - 32768) / 4096.f);
    }
    x.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    return x;
}
static std::vector<uint16_t> Bits(const Data &x, size_t count = 0) {
    if (!count)
        count = x.Count(0);
    std::vector<uint16_t> out(count);
    Require(cudaMemcpy(out.data(), x.cudaData, count * 2, cudaMemcpyDeviceToHost) == cudaSuccess, "copy");
    return out;
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        ApplyDeviceMap({{"cuda:0", 1}}, 1, 1);
        SetCudaEmbedding(true);
        int checks = 0;
        for (int rows : {1, 2, 7, 8, 17})
            for (int dim : {6, 32, 64, 128, 192, 256})
                for (int heads : {4, 32})
                    for (int past : {0, 15, 63})
                        for (bool dynamic : {false, true}) {
                            const int kh = 4, window = 64, cap = 128, qw = heads * dim, kw = kh * dim;
                            Data raw = BF({1, rows, qw + 2 * kw}, 123 + dim), qNorm(FLOAT32, {dim}),
                                 kNorm(FLOAT32, {dim});
                            qNorm.Allocate();
                            kNorm.Allocate();
                            for (int i = 0; i < dim; ++i) {
                                ((float *)qNorm.cpuData)[i] = .9f + std::sin(i) * .2f;
                                ((float *)kNorm.cpuData)[i] = 1.1f + std::cos(i) * .3f;
                            }
                            qNorm.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                            kNorm.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                            Data key = BF({1, cap, kw}, 77), value = BF({1, cap, kw}, 91);
                            key.expansionDims = {1, cap, kw};
                            value.expansionDims = {1, cap, kw};
                            key.Resize({1, past, kw});
                            value.Resize({1, past, kw});
                            Data rk = BF({1, cap, kw}, 77), rv = BF({1, cap, kw}, 91);
                            rk.expansionDims = {1, cap, kw};
                            rv.expansionDims = {1, cap, kw};
                            rk.Resize({1, past, kw});
                            rv.Resize({1, past, kw});
                            std::vector<float> positions(rows);
                            for (int i = 0; i < rows; ++i)
                                positions[i] = past + 20000 + i;
                            Data pos(FLOAT32, {1, rows}, positions);
                            pos.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                            Data live(INT32, {1});
                            live.Allocate();
                            *(int *)live.cpuData = past + 1;
                            live.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                            Data q, k, v, actual;
                            Split(raw, 2, 0, qw, q);
                            Split(raw, 2, qw, qw + kw, k);
                            Split(raw, 2, qw + kw, qw + 2 * kw, v);
                            q.Reshape({1, rows * heads, dim});
                            k.Reshape({1, rows * kh, dim});
                            KimiK3RMSNorm(q, qNorm, 1e-5, q);
                            KimiK3RMSNorm(k, kNorm, 1e-5, k);
                            q.Reshape({1, rows, qw});
                            k.Reshape({1, rows, kw});
                            FastllmCudaNaiveRope(q, pos, heads, dim, dim, 10000);
                            FastllmCudaNaiveRope(k, pos, kh, dim, dim, 10000);
                            FastllmCudaNaiveAppendVerifyCache(rk, rv, k, v, live, window);
                            auto invoke = [&]() {
                                Require(FastllmCudaNaiveDraftQKV(raw, qNorm, kNorm, pos,
                                                                 dynamic ? live : Data(), key, value, actual,
                                                                 heads, kh, dim, window, 1e-5, 10000),
                                        "valid fused QKV rejected");
                            };
                            invoke();
                            Require(Bits(q) == Bits(actual), "Q norm/rope mismatch");
                            if (Bits(key, (size_t)cap * kw) != Bits(rk, (size_t)cap * kw)) {
                                auto a = Bits(key, (size_t)cap * kw), b = Bits(rk, (size_t)cap * kw);
                                for (size_t t = 0; t < a.size(); ++t)
                                    if (a[t] != b[t]) {
                                        std::cerr << "case rows=" << rows << " dim=" << dim
                                                  << " heads=" << heads << " past=" << past
                                                  << " dynamic=" << dynamic << " idx=" << t
                                                  << " actual=" << a[t] << " ref=" << b[t] << std::endl;
                                        break;
                                    }
                                throw std::runtime_error("K norm/rope/cache mismatch");
                            }
                            Require(Bits(value, (size_t)cap * kw) == Bits(rv, (size_t)cap * kw),
                                    "V cache mismatch");
                            Require(key.dims[1] == past && value.dims[1] == past, "cache metadata changed");
                            if (rows == 7 && dim == 128 && heads == 32 && dynamic) {
                                void *g = nullptr, *exec = nullptr;
                                Require(FastllmCudaGraphBeginCapture(), "begin capture");
                                invoke();
                                Require(FastllmCudaGraphEndCapture(&g), "end capture");
                                Require(FastllmCudaGraphInstantiate(g, &exec), "instantiate");
                                Require(FastllmCudaGraphLaunch(exec), "replay");
                                Require(Bits(q) == Bits(actual), "graph replay mismatch");
                                FastllmCudaGraphExecDestroy(exec);
                                FastllmCudaGraphDestroy(g);
                            }
                            auto before = Bits(key, (size_t)cap * kw);
                            auto out = Bits(actual);
                            int old = raw.dims[2];
                            raw.dims[2]--;
                            Require(!FastllmCudaNaiveDraftQKV(raw, qNorm, kNorm, pos, live, key, value,
                                                              actual, heads, kh, dim, window, 1e-5, 10000),
                                    "invalid layout accepted");
                            raw.dims[2] = old;
                            Require(Bits(key, (size_t)cap * kw) == before && Bits(actual) == out,
                                    "fallback wrote output");
                            checks += 6;
                        }
        for (int rows : {1, 2, 7, 8, 33})
            for (int width : {1, 7, 128, 4096}) {
                Data packed = BF({1, rows, 2 * width}, rows + width), gate, up, out;
                Split(packed, 2, 0, width, gate);
                Split(packed, 2, width, 2 * width, up);
                Silu(gate, gate);
                MulTo(gate, up);
                FastllmCudaNaiveDraftSwiGLU(packed, out);
                Require(Bits(gate) == Bits(out), "SwiGLU rounding differs");
                ++checks;
            }
        std::cout << "DRAFT FUSION PASS checks=" << checks << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
