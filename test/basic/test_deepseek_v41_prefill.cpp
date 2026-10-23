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

static float Float(uint16_t b) {
    uint32_t bits = uint32_t(b) << 16; float f; std::memcpy(&f, &bits, 4); return f;
}
static float E4M3(uint8_t code) {
    const int e = (code >> 3) & 15, m = code & 7;
    float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(float(m), -9);
    return code & 128 ? -v : v;
}
static std::vector<float> Decode(const Data &x, int storage = 0) {
    if (!storage) {
        auto b = Read<uint16_t>(x); std::vector<float> out(b.size());
        for (size_t i = 0; i < b.size(); ++i) out[i] = Float(b[i]);
        return out;
    }
    auto b = Read<uint8_t>(x); std::vector<float> out(size_t(x.dims[1]) * 512);
    const float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (int row = 0; row < x.dims[1]; ++row) for (int d = 0; d < 512; ++d) {
        const auto *p = b.data() + size_t(row) * x.dims[2];
        const uint8_t scale = p[(storage == 1 ? 512 : 256) + d / (storage == 3 ? 16 : 32)];
        const float s = storage == 3 ? E4M3(scale) : std::ldexp(1.f, int(scale) - 127);
        const int code = (p[d / 2] >> (d % 2 * 4)) & 15;
        out[size_t(row) * 512 + d] = storage == 1 ? E4M3(p[d]) * s
            : (code & 8 ? -levels[code & 7] : levels[code & 7]) * s;
    }
    return out;
}
static void Attention(int device, bool quick) {
    struct Shape { int rows, heads, start, top; };
    for (auto shape : {Shape{128, 32, 0, 0}, Shape{129, 64, 31, 33},
                       Shape{128, 64, 2048, 512}, Shape{2048, 64, 0, 512}}) {
        if (quick && shape.rows >= 1024) continue;
        constexpr int window = 128, cap = 1024;
        Data q(BFLOAT16, {1, shape.rows, shape.heads, 512});
        Data chunk(BFLOAT16, {1, shape.rows, 512}), ring(BFLOAT16, {1, window, 512}), cmp(BFLOAT16, {1, cap, 512});
        Data sink(FLOAT32, {shape.heads}), idx(INT32, {1, shape.rows, shape.top});
        Fill(q, device, 3); Fill(chunk, device, 5); Fill(ring, device, 7); Fill(cmp, device, 11); Fill(sink, device, 13);
        std::vector<int> indices(size_t(shape.rows) * shape.top);
        for (int t = 0; t < shape.rows; ++t) for (int c = 0; c < shape.top; ++c)
            indices[size_t(t) * shape.top + c] = c % 23 == 0 ? -1 : c % 29 == 0 ? cap : (c * 17 + t * 31) % cap;
        if (shape.top) {
            idx.ToDevice(CUDA, {device}, false); idx.Allocate(false);
            Check(cudaMemcpy(idx.cudaData, indices.data(), indices.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess, "upload indices");
        }
        auto query = Decode(q), chunks = Decode(chunk); auto sinks = Read<float>(sink);
        for (int storage : {0, 1, 2, 3}) {
            Data packedRing, packedCmp, expected, actual;
            if (storage) {
                Check(FastllmCudaDeepSeekV41QuantizeKV(ring, packedRing, storage, storage == 3 ? 16 : 32), "ring quantization");
                Check(FastllmCudaDeepSeekV41QuantizeKV(cmp, packedCmp, storage, storage == 3 ? 16 : 32), "compressed quantization");
            }
            auto &r = storage ? packedRing : ring; auto &c = storage ? packedCmp : cmp;
            const float scale = 1.f / std::sqrt(512.f);
            auto run = [&](Data &out) {
                Check(FastllmCudaDeepSeekV41SparseAttention(q, chunk, shape.start ? &r : nullptr,
                    shape.top ? &c : nullptr, shape.top ? &idx : nullptr, sink, window, shape.start, scale, out), "attention launch");
            };
            setenv("FASTLLM_DSV41_LEGACY_ATTN", "1", 1); run(expected);
            unsetenv("FASTLLM_DSV41_LEGACY_ATTN"); run(actual);
            const auto old = Decode(expected), next = Decode(actual), rv = Decode(r, storage), cv = Decode(c, storage);
            double square = 0, referenceSquare = 0, maxError = 0; size_t differences = 0;
            for (size_t i = 0; i < next.size(); ++i) {
                Check(std::isfinite(next[i]), "nonfinite attention result");
                const double error = double(next[i]) - old[i];
                square += error * error; referenceSquare += double(old[i]) * old[i];
                differences += next[i] != old[i]; maxError = std::max(maxError, std::abs(error));
            }
            const double relative = std::sqrt(square / std::max(1e-30, referenceSquare));
            Check(relative < .001, "attention relative RMS difference");
            // Independent FP64 reference checks the first/last query and head,
            // including causal windows, invalid/duplicate IDs and sink mass.
            for (int token : {0, shape.rows - 1}) for (int head : {0, shape.heads - 1}) {
                std::vector<const float *> selected;
                const int pos = shape.start + token;
                for (int p = std::max(0, pos - window + 1); p <= pos; ++p)
                    selected.push_back(p >= shape.start ? chunks.data() + size_t(p - shape.start) * 512 : rv.data() + size_t(p % window) * 512);
                for (int j = 0; j < shape.top; ++j) {
                    int id = indices[size_t(token) * shape.top + j];
                    if (id >= 0 && id < cap) selected.push_back(cv.data() + size_t(id) * 512);
                }
                std::vector<double> score(selected.size()); double maximum = sinks[head];
                for (size_t j = 0; j < selected.size(); ++j) {
                    double dot = 0;
                    for (int dim = 0; dim < 512; ++dim)
                        dot += double(query[(size_t(token) * shape.heads + head) * 512 + dim]) * selected[j][dim];
                    score[j] = dot * scale; maximum = std::max(maximum, score[j]);
                }
                double den = std::exp(double(sinks[head]) - maximum);
                for (auto &v : score) { v = std::exp(v - maximum); den += v; }
                for (int dim = 0; dim < 512; dim += 17) {
                    double numerator = 0;
                    for (size_t j = 0; j < score.size(); ++j) numerator += score[j] * selected[j][dim];
                    const double reference = numerator / den;
                    const double error = std::abs(next[(size_t(token) * shape.heads + head) * 512 + dim] - reference);
                    Check(error <= 1e-5 + std::abs(reference) * .0041, "attention FP64 rounding bound");
                }
            }
            std::printf("ATTENTION PASS gpu=%d rows=%d heads=%d start=%d top=%d storage=%d relative_rmse=%.8g max_abs=%.8g different=%zu/%zu\n",
                device, shape.rows, shape.heads, shape.start, shape.top, storage, relative, maxError, differences, next.size());
            std::fflush(stdout);
        }
    }
    unsetenv("FASTLLM_DSV41_LEGACY_ATTN");
}

static void AttentionFallback(int device) {
    // Reject a late query chunk after the first 128 rows were already written.
    // The scalar fallback must replace the entire output.
    constexpr int rows = 129, heads = 32;
    Data q(BFLOAT16, {1, rows, heads, 512}), kv(BFLOAT16, {1, rows, 512}), sink(FLOAT32, {heads});
    Data expected, actual;
    Fill(q, device, 71); Fill(kv, device, 73); Fill(sink, device, 79);
    uint16_t tiny = Bf(1e-30f);
    Check(cudaMemcpy(static_cast<uint16_t *>(q.cudaData) + size_t(128) * heads * 512, &tiny, 2,
        cudaMemcpyHostToDevice) == cudaSuccess, "attention range upload");
    auto run = [&](Data &out) {
        Check(FastllmCudaDeepSeekV41SparseAttention(q, kv, nullptr, nullptr, nullptr,
            sink, 128, 0, 1.f / std::sqrt(512.f), out), "attention fallback launch");
    };
    run(actual);
    // Disable just the generic helper via graph capture; this preserves the
    // optimized scalar fallback rather than selecting the legacy kernel.
    cudaGraph_t graph; cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess, "attention capture begin");
    run(actual);
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "attention capture end");
    Check(cudaGraphInstantiateWithFlags(&exec, graph, 0) == cudaSuccess, "attention graph instantiate");
    Check(cudaGraphLaunch(exec, cudaStreamPerThread) == cudaSuccess, "attention graph replay");
    const auto captured = Read<uint16_t>(actual);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    run(expected);
    Check(Read<uint16_t>(expected) == captured, "late attention fallback or graph output changed");
    // In-place output must not corrupt Q before a late chunk rejects packing.
    run(q);
    Check(Read<uint16_t>(q) == captured, "in-place attention changed Q before late fallback");
    std::printf("ATTENTION FALLBACK PASS gpu=%d late-query range, graph and in-place output\n", device);
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
            Attention(device, quick);
            AttentionFallback(device);
        }
        if (!tested) return 77;
        std::puts("DeepSeek V4.1 prefill PASS"); return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
