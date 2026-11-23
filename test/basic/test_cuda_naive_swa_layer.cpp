#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace fastllm;
namespace {
void Require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void Cuda(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
template<class F> void Parallel(F fn) {
    std::array<std::exception_ptr, 8> errors{};
    std::vector<std::thread> workers;
    for (int r = 0; r < 8; ++r) workers.emplace_back([&, r] {
        try { FastllmCudaSetDevice(r); fn(r); } catch (...) { errors[r] = std::current_exception(); }
    });
    for (auto &worker : workers) worker.join();
    for (auto &error : errors) if (error) std::rethrow_exception(error);
}
void Env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
void Init(Data &x, const std::vector<int> &shape, int rank) {
    x.Resize(shape); x.ToDevice(DataDevice::CUDA, {rank}, false); x.Allocate(false);
}
void Fill(Data &x, unsigned seed, float scale = 1.f, float bias = 0.f) {
    std::vector<uint8_t> bytes(x.GetBytes());
    for (size_t i = 0; i < x.Count(0); ++i) {
        seed = seed * 1664525u + 1013904223u;
        float value = (int(seed >> 16) - 32768) / 32768.f * scale + bias;
        if (x.dataType == BFLOAT16) {
            uint16_t bits = Float32ToBFloat16RNEBits(value);
            std::memcpy(bytes.data() + i * 2, &bits, 2);
        } else std::memcpy(bytes.data() + i * 4, &value, 4);
    }
    Cuda(cudaMemcpy(x.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}
std::vector<uint8_t> Read(const Data &x) {
    std::vector<uint8_t> bytes(x.GetBytes());
    Cuda(cudaMemcpy(bytes.data(), x.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
    return bytes;
}
void CopyDevice(Data &dst, const Data &src) {
    Cuda(cudaMemcpy(dst.cudaData, src.cudaData, src.GetBytes(), cudaMemcpyDeviceToDevice));
}
void Compare(const Data &actual, const Data &reference, const char *name) {
    auto a = Read(actual), b = Read(reference);
    double error = 0, magnitude = 0;
    for (size_t i = 0; i < actual.Count(0); ++i) {
        auto value = [&](const std::vector<uint8_t> &bytes) {
            float result;
            if (actual.dataType == BFLOAT16) {
                uint16_t low; std::memcpy(&low, bytes.data() + i * 2, 2);
                uint32_t bits = uint32_t(low) << 16; std::memcpy(&result, &bits, 4);
            } else std::memcpy(&result, bytes.data() + i * 4, 4);
            Require(std::isfinite(result), "non-finite output"); return double(result);
        };
        double x = value(a), y = value(b);
        error += (x - y) * (x - y); magnitude += y * y;
    }
    // BF16 MMA changes accumulation order, but not the documented BF16
    // materialization boundaries. Use relative L2, including zero outputs.
    if (std::sqrt(error) > .003 * std::sqrt(magnitude) + 1e-7)
        throw std::runtime_error(std::string(name) + " relative L2=" +
                                 std::to_string(std::sqrt(error / (magnitude + 1e-30))));
}
struct State {
    Data hidden{BFLOAT16}, normed{BFLOAT16}, q{BFLOAT16}, attn{BFLOAT16}, projected{BFLOAT16};
    Data key{BFLOAT16}, value{BFLOAT16}, router{FLOAT32};
    explicit State(int rank) {
        for (auto *x : {&hidden, &normed, &projected}) Init(*x, {1, 1, 4096}, rank);
        Init(q, {1, 1, 1536}, rank); Init(attn, {1, 1, 1024}, rank);
        Init(router, {1, 1, 256}, rank);
        Init(key, {1, 128, 192}, rank); Init(value, {1, 128, 128}, rank);
        key.expansionDims = key.dims; value.expansionDims = value.dims;
    }
    std::vector<Data *> Outputs() { return {&hidden, &normed, &q, &attn, &projected, &key, &value, &router}; }
};
struct Tensors {
    Data branch{BFLOAT16}, qkv{BFLOAT16}, o{BFLOAT16}, inputNorm{FLOAT32}, postNorm{FLOAT32};
    Data routerWeight{FLOAT32}, sink{FLOAT32}, position{FLOAT32}, live{INT32}, bias;
    Data kTemp{BFLOAT16}, vTemp{BFLOAT16}, scratch{BFLOAT16};
    State actual, reference, alternate;
    const void **table = nullptr;
    cudaGraphExec_t graph = nullptr;
    bool hasSink = true, indirect = false;
    FastllmSwaLayerConfig config{8, 1, 192, 128, 64, 128, 1e-6f, 10000.f, 1.f};
    explicit Tensors(int rank) : actual(rank), reference(rank), alternate(rank) {
        Init(branch, {1, 1, 4096}, rank); Init(qkv, {1856, 4096}, rank);
        Init(o, {4096, 1024}, rank); Init(routerWeight, {256, 4096}, rank);
        Init(inputNorm, {4096}, rank); Init(postNorm, {4096}, rank); Init(sink, {8}, rank);
        Init(position, {1}, rank); Init(live, {1}, rank);
        Init(kTemp, {1, 1, 192}, rank); Init(vTemp, {1, 1, 128}, rank);
        Fill(qkv, 31 + rank, .02f); Fill(o, 93 + rank, .02f);
        Fill(inputNorm, 731, .2f, 1.f); Fill(postNorm, 83, .2f, 1.f);
        Fill(routerWeight, 582, .02f); Fill(sink, 17);
        Cuda(cudaMalloc(reinterpret_cast<void **>(&table), 2 * sizeof(void *)));
    }
    void Bind(bool useAlternate) {
        auto &s = useAlternate ? alternate : actual;
        const void *ptrs[] = {s.key.cudaData, s.value.cudaData};
        Cuda(cudaMemcpy(table, ptrs, sizeof(ptrs), cudaMemcpyHostToDevice));
    }
    void Reset(int seed, int length, int rank) {
        Fill(branch, seed * 17 + 7); Fill(actual.hidden, seed + 801);
        Fill(actual.key, seed * 13 + rank, .4f); Fill(actual.value, seed * 19 + rank);
        for (auto *x : {&actual.q, &actual.attn, &actual.projected, &actual.normed, &actual.router})
            Cuda(cudaMemset(x->cudaData, 0, x->GetBytes()));
        auto a = actual.Outputs(), b = reference.Outputs();
        for (size_t i = 0; i < a.size(); ++i) CopyDevice(*b[i], *a[i]);
        CopyDevice(alternate.key, actual.key); CopyDevice(alternate.value, actual.value);
        float pos = float(length > 0 ? length - 1 : 0);
        Cuda(cudaMemcpy(position.cudaData, &pos, 4, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, &length, 4, cudaMemcpyHostToDevice));
    }
    void Legacy(State &s, int rank) {
        FastllmCudaNaiveAddDecodeRMSNorm(s.hidden, branch, inputNorm, config.eps, s.normed);
        std::vector<FastllmNaiveRopeCacheRow> rows{{&s.key, &s.value, nullptr}};
        Require(FastllmCudaNaiveBatchQkvRopeAppendCache(s.normed, qkv, bias, s.q, kTemp, vTemp,
            position, rows, live, 8, 1, 192, 128, 64, config.theta, config.valueScale, 128, true, scratch),
            "reference QKV/RoPE failed");
        std::vector<FastllmNaiveSWARow> attention{{s.key.cudaData, s.value.cudaData}};
        FastllmCudaNaiveBatchSWADecode(s.q, hasSink ? sink : bias, attention, live, 8, 1, s.attn);
        Require(FastllmCudaBFloat16MatMulBFloat16(s.attn, o, bias, s.projected, 1, 1024, 4096),
                "reference projection failed");
        Require(FastllmCudaCustomAllReduce(s.projected.cudaData, s.projected.cudaData, 4096, BFLOAT16, rank),
                "reference collective failed");
        FastllmCudaNaiveAddDecodeRMSNorm(s.hidden, s.projected, postNorm, config.eps, s.normed);
        Require(FastllmCudaNaiveRouterSigmoid(s.normed, routerWeight, s.router), "reference router failed");
    }
    bool Fused(int rank, int kind = 0) {
        Data b, h, qw, k, v, q, a, p, n, router;
        auto view = [](Data &dst, const Data &src) {
            dst.Resize(src.dims); dst.FakeFrom(src, 0); dst.strides = src.strides;
            dst.expansionDims = src.expansionDims;
        };
        view(b, branch); view(h, actual.hidden); view(qw, qkv);
        view(k, actual.key); view(v, actual.value); view(q, actual.q);
        view(a, actual.attn); view(p, actual.projected); view(n, actual.normed); view(router, actual.router);
        auto c = config;
        if (rank == 3) {
            if (kind == 1) b.strides.back() = 2;
            if (kind == 2) qw.dataType = FLOAT32;
            if (kind == 3) n.cudaData = h.cudaData;
            if (kind == 4) b.Resize({1, 2, 4096});
            if (kind == 5) h.cudaData = (char *)h.cudaData + 2;
            if (kind == 6) c.eps = -1;
            if (kind == 7) n.Resize({16});
            if (kind == 8) k.expansionDims[1] = 127;
            if (kind == 9) v.isPagedKVCache = true;
            if (kind == 10) c.heads = 16;
            if (kind == 11) qw.cudaData = nullptr;
            if (kind == 12) p.cudaData = b.cudaData;
        }
        return FastllmCudaSwaLayer(b, h, inputNorm, qw, bias, o, postNorm, routerWeight,
            hasSink ? sink : bias, position, live, k, v, indirect ? table : nullptr,
            kind == 13 && rank == 3 ? h : q, a, p, n, router, c, rank);
    }
    void Check(bool rebound = false) {
        const char *names[] = {"hidden", "normed", "q", "attention", "projection", "key", "value", "router"};
        auto a = actual.Outputs(), b = reference.Outputs();
        if (rebound) { a[5] = &alternate.key; a[6] = &alternate.value; }
        for (size_t i = 0; i < a.size(); ++i) Compare(*a[i], *b[i], names[i]);
    }
};
}
int main(int argc, char **argv) {
    try {
        bool disabled = argc > 1 && std::strcmp(argv[1], "--disabled") == 0;
        Env("FASTLLM_NAIVE_SWA_FUSION", disabled ? "0" : nullptr);
        Env("FASTLLM_NAIVE_SWA_CORE", nullptr); Env("FASTLLM_NAIVE_SWA_QKV_FUSION", nullptr);
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8) return 77;
        for (int r = 0; r < 8; ++r) {
            cudaDeviceProp p{}; Cuda(cudaGetDeviceProperties(&p, r));
            if (p.major != 10 || p.minor != 3) return 77;
        }
        const std::vector<int> devices{0, 1, 2, 3, 4, 5, 6, 7};
        if (!FastllmInitNccl(devices) || !FastllmCudaCustomAllReduceInit(devices)) return 77;
        for (int r = 0; r < 8; ++r) {
            FastllmCudaSetDevice(r);
            if (!disabled && !FastllmCudaSwaLayerSupported(r)) return 77;
        }
        std::array<std::unique_ptr<Tensors>, 8> data;
        std::array<int, 8> accepted{};
        Parallel([&](int r) { data[r] = std::make_unique<Tensors>(r); });
        int cases = 0;
        for (int sink : {0, 1}) for (int length : {0, 1, 63, 127, 128, 129, 8192}) {
            int seed = ++cases;
            Parallel([&](int r) {
                auto &d = *data[r]; d.hasSink = sink; d.config.valueScale = sink ? 1.f : .75f;
                d.Reset(seed, length, r); d.Legacy(d.reference, r); Cuda(cudaDeviceSynchronize());
            });
            Parallel([&](int r) {
                auto &d = *data[r]; accepted[r] = d.Fused(r);
                if (!accepted[r]) d.Legacy(d.actual, r);
                Cuda(cudaDeviceSynchronize()); d.Check();
            });
            for (int ok : accepted) Require(bool(ok) == !disabled, "default/disabled dispatch mismatch");
        }
        if (!disabled) {
            // A cached registration must still reject the entire group when
            // only rank 3 becomes invalid, before any output or KV write.
            for (int kind = 1; kind <= 13; ++kind) {
                Parallel([&](int r) {
                    auto &d = *data[r]; d.Reset(30 + kind, 129, r);
                    std::vector<std::vector<uint8_t>> before;
                    for (auto *x : d.actual.Outputs()) before.push_back(Read(*x));
                    accepted[r] = d.Fused(r, kind); Cuda(cudaDeviceSynchronize());
                    auto out = d.actual.Outputs();
                    for (size_t i = 0; i < out.size(); ++i)
                        Require(before[i] == Read(*out[i]), "rejection modified output/KV");
                });
                for (int ok : accepted) Require(!ok, "one bad rank did not reject the whole group");
            }
            Parallel([&](int r) {
                auto &d = *data[r]; d.indirect = true; d.Bind(false);
                Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                accepted[r] = d.Fused(r);
                cudaGraph_t graph = nullptr; Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                Cuda(cudaGraphInstantiate(&d.graph, graph, 0)); Cuda(cudaGraphDestroy(graph));
            });
            for (int ok : accepted) Require(ok, "warm Graph capture rejected");
            // Capture once; change lengths and request-owned KV bindings only.
            for (int length : {1, 128, 129, 8192, 63, 0}) {
                int seed = ++cases;
                Parallel([&](int r) {
                    auto &d = *data[r]; d.Reset(seed, length, r); d.Bind(true);
                    d.Legacy(d.reference, r); Cuda(cudaDeviceSynchronize());
                });
                Parallel([&](int r) {
                    auto &d = *data[r]; auto k = Read(d.actual.key), v = Read(d.actual.value);
                    Cuda(cudaGraphLaunch(d.graph, cudaStreamPerThread)); Cuda(cudaDeviceSynchronize()); d.Check(true);
                    Require(k == Read(d.actual.key) && v == Read(d.actual.value), "Graph ignored KV rebinding");
                });
            }
        }
        Parallel([&](int r) {
            if (data[r]->graph) Cuda(cudaGraphExecDestroy(data[r]->graph));
            Cuda(cudaFree(data[r]->table));
        });
        FastllmCudaCustomAllReduceReset();
        Parallel([&](int r) { data[r].reset(); });
        std::printf("SWA layer PASS: enabled=%d, %d reference cases, collective rejection and fixed Graph replay\n",
                    !disabled, cases);
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "SWA layer FAIL: %s\n", e.what()); return 1; }
}
