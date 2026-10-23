#include "fastllm.h"
#include "../../src/devices/cuda/fastllm-hybrid-prefill.cuh"
#include "devices/cuda/fastllm-cuda-hybrid-prefill.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;

static void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static uint16_t Bf(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}
static void Fill(Data &x, int device, unsigned seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> uniform(-.25f, .25f);
    x.ToDevice(CUDA, {device}, false); x.Allocate(false);
    std::vector<float> f(x.Count(0));
    for (size_t i = 0; i < f.size(); ++i) f[i] = std::ldexp(uniform(random), int(i % 9) - 4);
    if (x.dataType == FLOAT32) Check(cudaMemcpy(x.cudaData, f.data(), f.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess, "upload float");
    else {
        std::vector<uint16_t> b(f.size());
        for (size_t i = 0; i < b.size(); ++i) b[i] = Bf(f[i]);
        Check(cudaMemcpy(x.cudaData, b.data(), b.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess, "upload bf16");
    }
}
template<class T> static std::vector<T> Read(const Data &x) {
    std::vector<T> v(x.GetBytes() / sizeof(T));
    Check(cudaMemcpy(v.data(), x.cudaData, x.GetBytes(), cudaMemcpyDeviceToHost) == cudaSuccess, "download");
    return v;
}
static void Indexer(int device, bool quick) {
    struct Shape { int rows, keys, ratio, start; };
    for (auto shape : {Shape{128, 131, 2, 0}, Shape{129, 513, 1, 512},
                       Shape{2048, 1024, 2, 0}, Shape{2000, 4048, 1, 2048},
                       Shape{128, 4048, 1, 3920}}) {
        if (quick && shape.rows >= 1024) continue;
        Data q(BFLOAT16, {1, shape.rows, 32, 128}), k(BFLOAT16, {1, shape.keys, 128});
        Data w(FLOAT32, {1, shape.rows, 32});
        Fill(q, device, 17); Fill(k, device, 31); Fill(w, device, 53);
        for (int storage : {0, 1, 2, 3}) {
            Data packed, old, next, oldTop, nextTop;
            if (storage) Check(FastllmCudaDeepSeekV41QuantizeKV(k, packed, storage, storage == 3 ? 16 : 32), "key quantization");
            const Data &keys = storage ? packed : k;
            setenv("FASTLLM_DSV41_LEGACY_INDEXER", "1", 1);
            Check(FastllmCudaDeepSeekV41IndexerScore(q, w, keys, shape.ratio, shape.start, old), "legacy score");
            unsetenv("FASTLLM_DSV41_LEGACY_INDEXER");
            Check(FastllmCudaDeepSeekV41IndexerScore(q, w, keys, shape.ratio, shape.start, next), "new score");
            const auto a = Read<float>(old), b = Read<float>(next);
            size_t visibleCount = 0;
            for (int row = 0; row < shape.rows; ++row) {
                const int visible = std::min(shape.keys, (shape.start + row + 1) / shape.ratio);
                for (int c = 0; c < shape.keys; ++c) {
                    const size_t i = size_t(row) * shape.keys + c;
                    if (c < visible) {
                        if (std::memcmp(&a[i], &b[i], sizeof(float))) {
                            std::fprintf(stderr, "score mismatch row=%d key=%d old=%.9g new=%.9g\n", row, c, a[i], b[i]);
                            throw std::runtime_error("visible score is not bitwise equal");
                        }
                        ++visibleCount;
                    } else Check(std::isinf(b[i]) && b[i] < 0, "future score must be -inf");
                }
            }
            Check(FastllmCudaDeepSeekV41IndexerTopK(old, nullptr, 512, shape.ratio, shape.start, 8, oldTop), "old top-k");
            Check(FastllmCudaDeepSeekV41IndexerTopK(next, nullptr, 512, shape.ratio, shape.start, 8, nextTop), "new top-k");
            Check(Read<int>(oldTop) == Read<int>(nextTop), "top-k changed");
            std::printf("INDEXER PASS gpu=%d rows=%d keys=%d ratio=%d start=%d storage=%d visible=%zu\n",
                device, shape.rows, shape.keys, shape.ratio, shape.start, storage, visibleCount);
            std::fflush(stdout);
        }
    }
    unsetenv("FASTLLM_DSV41_LEGACY_INDEXER");
}

int main(int argc, char **argv) {
    try {
        bool tested = false, quick = false;
        for (int i = 1; i < argc; ++i) {
            Check(!std::strcmp(argv[i], "--quick"), "Usage: deepseek_v41_prefill_test [--quick]");
            quick = true;
        }
        for (int device = 0; device < FastllmCudaGetDeviceCount(); ++device) {
            FastllmCudaSetDevice(device);
            if (FastllmCudaRuntimeArch() != 75) continue;
            tested = true;
            Indexer(device, quick);
        }
        if (!tested) return 77;
        std::puts("DeepSeek V4.1 prefill PASS"); return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
