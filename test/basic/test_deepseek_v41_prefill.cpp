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
    struct Shape { int rows, heads, start, top; int window = 128; };
    for (auto shape : {Shape{1, 64, 2048, 512}, Shape{2, 32, 126, 33},
                       Shape{4, 33, 126, 33}, Shape{6, 64, 1, 512},
                       Shape{6, 64, 2048, 512}, Shape{8, 32, 2048, 0},
                       Shape{128, 32, 0, 0}, Shape{129, 64, 31, 33},
                       Shape{128, 64, 2048, 512}, Shape{2048, 64, 0, 512},
                       Shape{128, 32, 3, 513, 4}, Shape{128, 64, 3, 513, 4}}) {
        if (quick && shape.rows >= 1024) continue;
        const int window = shape.window;
        constexpr int cap = 1024;
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

static void CompareBf16(const Data &reference, const Data &actual, const char *name) {
    const auto a = Decode(reference), b = Decode(actual);
    Check(a.size() == b.size(), "GEMM output shape");
    double error = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        Check(std::isfinite(b[i]), "nonfinite GEMM output");
        error += std::pow(double(a[i]) - b[i], 2); norm += double(a[i]) * a[i];
    }
    double relative = std::sqrt(error / std::max(norm, 1e-30));
    std::printf("%s relative_rmse=%.8g\n", name, relative);
    Check(relative < .001, "GEMM relative RMS difference");
}

static void Bf16Projection(int device, bool quick) {
    // Grouped wo_a needs a strided activation view and output. Check both
    // padding regions and FP64 sums independently of the cuBLAS reference.
    constexpr int rows = 1024, columns = 136, inputStride = 152, width = 128, outputStride = 144;
    Data x(BFLOAT16, {rows, inputStride}), w(BFLOAT16, {width, columns}), y(BFLOAT16, {rows, outputStride});
    Fill(x, device, 91); Fill(w, device, 93); Fill(y, device, 95);
    auto xv = Decode(x), wv = Decode(w);
    auto sentinel = Read<uint16_t>(y);
    auto handle = getFastllmCublasHandle();
    auto run = [&]() {
        return FastllmCudaHybridBf16Gemm(static_cast<const __nv_bfloat16 *>(x.cudaData),
            static_cast<const __nv_bfloat16 *>(w.cudaData), static_cast<__nv_bfloat16 *>(y.cudaData),
            rows, columns, width, outputStride, handle, inputStride);
    };
    Check(!run(), "BF16 GEMM default scope fallback");
    {
        FastllmCudaHybridPrefillScope scope(true);
        Check(run(), "strided BF16 GEMM");
        const auto got = Decode(y); const auto bits = Read<uint16_t>(y);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < width; ++col) {
                double sum = 0, magnitude = 0;
                for (int k = 0; k < columns; ++k) {
                    const double product = double(xv[size_t(row) * inputStride + k]) * wv[size_t(col) * columns + k];
                    sum += product; magnitude += std::abs(product);
                }
                // Include the FP32 dot-product bound: near cancellation, a
                // relative BF16-only tolerance does not bound accumulation.
                const double gamma = columns * std::numeric_limits<float>::epsilon();
                const double bound = gamma / (1 - gamma) * magnitude + std::abs(sum) * .0041 + 1e-7;
                if (std::abs(got[size_t(row) * outputStride + col] - sum) > bound) {
                    std::fprintf(stderr, "GEMM FP64 mismatch row=%d col=%d actual=%.12g reference=%.12g\n", row, col, got[size_t(row) * outputStride + col], sum);
                    throw std::runtime_error("strided BF16 GEMM FP64 rounding bound");
                }
            }
            for (int col = width; col < outputStride; ++col)
                Check(bits[size_t(row) * outputStride + col] == sentinel[size_t(row) * outputStride + col], "GEMM padding overwrite");
        }
        uint16_t tiny = Bf(1e-30f);
        Check(cudaMemcpy(x.cudaData, &tiny, 2, cudaMemcpyHostToDevice) == cudaSuccess, "range test upload");
        const auto before = Read<uint16_t>(y);
        Check(!run(), "strided BF16 GEMM unsafe range fallback");
        Check(Read<uint16_t>(y) == before, "rejected first chunk changed output");
        std::vector<uint16_t> extreme(inputStride, Bf(std::ldexp(1.f, 40)));
        extreme[0] = Bf(std::ldexp(1.f, -130));
        Check(cudaMemcpy(x.cudaData, extreme.data(), extreme.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess, "FP32 underflow test upload");
        Check(!run(), "BF16 downscale must not silently underflow in FP32");
        Check(Read<uint16_t>(y) == before, "underflow fallback changed output");
    }

    const int tokens = quick ? 1024 : 2048;
    const int groupDim = quick ? 128 : 4096, rank = quick ? 128 : 1024, groups = 8;
    Data input(BFLOAT16, {1, tokens, groups, groupDim}), weight(BFLOAT16, {groups * rank, groupDim});
    Data old, next;
    Fill(input, device, 97); Fill(weight, device, 101);
    auto woa = [&](Data &output, bool enabled) {
        FastllmCudaHybridPrefillScope scope(enabled);
        Check(FastllmCudaDeepSeekV4WoA(input, weight, groups, rank, output, false), "wo_a projection");
    };
    woa(old, false); woa(next, true); CompareBf16(old, next, "WO_A");
    // Fail admission after several groups have already written their output.
    // The caller must overwrite those partial outputs with the full fallback.
    uint16_t tiny = Bf(1e-30f);
    Check(cudaMemcpy(static_cast<uint16_t *>(input.cudaData) + 6 * groupDim, &tiny, 2, cudaMemcpyHostToDevice) == cudaSuccess, "late group upload");
    woa(old, false); woa(next, true);
    Check(Read<uint16_t>(old) == Read<uint16_t>(next), "late group fallback did not overwrite all groups");
    std::printf("WO_A PASS gpu=%d rows=%d group_dim=%d rank=%d\n", device, tokens, groupDim, rank);

    // Exercise the public FP8 weight path, including BF16 expansion and bias.
    constexpr int fp8Rows = 1024, fp8Columns = 1280, fp8Width = 4096;
    Data fp8(FP8_E4M3, {fp8Width, fp8Columns}), activation(BFLOAT16, {fp8Rows, fp8Columns});
    Data bias(FLOAT32, {fp8Width}), expected(BFLOAT16, {fp8Rows, fp8Width}), actual(BFLOAT16, {fp8Rows, fp8Width});
    fp8.blockK = 32; fp8.blockM = 32;
    fp8.scales.resize(size_t(fp8Width / 32) * (fp8Columns / 32));
    for (size_t i = 0; i < fp8.scales.size(); ++i) fp8.scales[i] = std::ldexp(1.f, int(i % 5) - 8);
    fp8.Allocate();
    std::mt19937 random(103);
    for (size_t i = 0; i < fp8.Count(0); ++i) fp8.cpuData[i] = uint8_t(random() % 96) | uint8_t((random() % 2) << 7);
    fp8.ToDevice(CUDA, {device}, true);
    Fill(activation, device, 107); Fill(bias, device, 109); Fill(expected, device, 113); Fill(actual, device, 127);
    auto linear = [&](Data &output, bool enabled) {
        FastllmCudaHybridPrefillScope scope(enabled);
        Check(FastllmCudaBFloat16MatMulFP8E4M3(activation, fp8, bias, output, fp8Rows, fp8Columns, fp8Width), "FP8 prefill linear");
    };
    linear(expected, false); linear(actual, true); CompareBf16(expected, actual, "FP8_LINEAR");
    std::printf("FP8_LINEAR PASS gpu=%d\n", device);
    std::fflush(stdout);
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
            Bf16Projection(device, quick);
        }
        if (!tested) return 77;
        std::puts("DeepSeek V4.1 prefill PASS"); return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
