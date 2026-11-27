#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "utils/utils.h"
#include "naive_n05_mxfp8_test.h"
#include <cuda_runtime_api.h>
#include <cuda_fp8.h>
#include <array>
#include <algorithm>
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
void Require(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
void Cuda(cudaError_t e) {
    Require(e == cudaSuccess, cudaGetErrorString(e));
}
bool testMxProjection = false;
bool testFp8KV = false;
template <class F> void Parallel(F fn) {
    std::array<std::exception_ptr, 8> errors{};
    std::vector<std::thread> workers;
    for (int r = 0; r < 8; ++r)
        workers.emplace_back([&, r] {
            try {
                FastllmCudaSetDevice(r);
                std::shared_ptr<void> lease;
                std::unique_ptr<FastllmCudaLinearGraphScope> scope;
                if (testMxProjection) scope = std::make_unique<FastllmCudaLinearGraphScope>(lease);
                fn(r);
            } catch (...) {
                errors[r] = std::current_exception();
            }
        });
    for (auto &worker : workers)
        worker.join();
    for (auto &error : errors)
        if (error)
            std::rethrow_exception(error);
}
void Env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
void Init(Data &x, const std::vector<int> &shape, int rank) {
    x.Resize(shape);
    x.ToDevice(DataDevice::CUDA, {rank}, false);
    x.Allocate(false);
}
void Fill(Data &x, unsigned seed, float scale = 1.f, float bias = 0.f) {
    std::vector<uint8_t> bytes(x.GetBytes());
    for (size_t i = 0; i < x.Count(0); ++i) {
        seed = seed * 1664525u + 1013904223u;
        float value = (int(seed >> 16) - 32768) / 32768.f * scale + bias;
        if (x.dataType == BFLOAT16) {
            uint16_t bits = Float32ToBFloat16RNEBits(value);
            std::memcpy(bytes.data() + i * 2, &bits, 2);
        } else if (x.dataType == FP8_E4M3) bytes[i] = __nv_fp8_e4m3(value).__x;
        else std::memcpy(bytes.data() + i * 4, &value, 4);
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
double CompareRow(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, DataType type, size_t first,
                  size_t width, const char *name) {
    double error = 0, magnitude = 0;
    for (size_t i = first; i < first + width; ++i) {
        auto value = [&](const std::vector<uint8_t> &bytes) {
            float result;
            if (type == BFLOAT16) {
                uint16_t low;
                std::memcpy(&low, bytes.data() + i * 2, 2);
                uint32_t bits = uint32_t(low) << 16;
                std::memcpy(&result, &bits, 4);
            } else
                std::memcpy(&result, bytes.data() + i * 4, 4);
            Require(std::isfinite(result), "non-finite output");
            return double(result);
        };
        double x = value(a), y = value(b);
        error += (x - y) * (x - y);
        magnitude += y * y;
    }
    // Check each active row independently; untouched cache storage must not
    // dilute the relative L2. The absolute term handles exact-zero outputs.
    const double l2 = std::sqrt(error / (magnitude + 1e-30));
    if (std::sqrt(error) > .003 * std::sqrt(magnitude) + 1e-7)
        throw std::runtime_error(std::string(name) + " relative L2=" + std::to_string(l2));
    return l2;
}
struct State {
    Data hidden{BFLOAT16}, normed{BFLOAT16}, q{BFLOAT16}, attn{BFLOAT16}, projected{BFLOAT16};
    Data key{BFLOAT16}, value{BFLOAT16}, router{FLOAT32}, feature{BFLOAT16};
    explicit State(int rank, int rows, int requests) {
        for (auto *x : {&hidden, &normed, &projected, &feature})
            Init(*x, {1, rows, 4096}, rank);
        Init(q, {1, rows, 1536}, rank);
        Init(attn, {1, rows, 1024}, rank);
        Init(router, {1, rows, 256}, rank);
        if (testFp8KV) {
            key.dataType = value.dataType = FP8_E4M3;
            key.UpdateUnitSize(); value.UpdateUnitSize();
        }
        Init(key, {requests, 136, 192}, rank);
        Init(value, {requests, 136, 128}, rank);
    }
    std::vector<Data *> Outputs() {
        return {&hidden, &normed, &q, &attn, &projected, &key, &value, &router, &feature};
    }
};
struct Tensors {
    int rows, requests;
    std::vector<int> owner, offset;
    Data branch{BFLOAT16}, qkv{BFLOAT16}, o{BFLOAT16}, inputNorm{FLOAT32}, postNorm{FLOAT32};
    Data routerWeight{FLOAT32}, sink{FLOAT32}, position{FLOAT32}, live{INT32}, bias;
    Data kTemp{BFLOAT16}, vTemp{BFLOAT16}, scratch{BFLOAT16};
    State actual, reference, alternate;
    const void **table = nullptr;
    cudaGraphExec_t graph = nullptr;
    bool hasSink = true, indirect = false, withRouter = true;
    std::array<double, 8> maxL2{};
    FastllmSwaLayerConfig config{8, 1, 192, 128, 64, 128, 1e-5f, 10000.f, 1.f};
    Tensors(int rank, int count, bool ragged, bool mx = false)
        : rows(count), requests(ragged ? 8 : (count + 7) / 8), actual(rank, count, requests),
          reference(rank, count, requests), alternate(rank, count, requests) {
        for (int seq = 0; int(owner.size()) < rows; ++seq)
            for (int col = 0, n = ragged ? seq + 1 : std::min(8, rows - int(owner.size())); col < n; ++col) {
                owner.push_back(seq);
                offset.push_back(col);
            }
        if (ragged)
            for (int i = 0; i < rows / 2; ++i) {
                std::swap(owner[i], owner[rows - 1 - i]);
                std::swap(offset[i], offset[rows - 1 - i]);
            }
        Init(branch, {1, rows, 4096}, rank);
        if (mx) { qkv.dataType = o.dataType = FP8_E4M3; qkv.UpdateUnitSize(); o.UpdateUnitSize(); }
        Init(qkv, {1856, 4096}, rank);
        Init(o, {4096, 1024}, rank);
        Init(routerWeight, {256, 4096}, rank);
        Init(inputNorm, {4096}, rank);
        Init(postNorm, {4096}, rank);
        Init(sink, {8}, rank);
        Init(position, {rows}, rank);
        Init(live, {rows}, rank);
        if (mx) { FillVerifyMxWeight(qkv,31+rank); FillVerifyMxWeight(o,93+rank); }
        else { Fill(qkv,31+rank,.02f); Fill(o,93+rank,.02f); }
        Fill(inputNorm, 731, .2f, 1.f);
        Fill(postNorm, 83, .2f, 1.f);
        Fill(routerWeight, 582, .02f);
        Fill(sink, 17);
        Cuda(cudaMalloc(reinterpret_cast<void **>(&table), requests * 2 * sizeof(void *)));
    }
    static void View(Data &dst, const Data &src, size_t element, std::vector<int> shape) {
        dst.Resize(shape);
        dst.FakeFrom(src, element * src.unitSize);
    }
    void Bind(bool useAlternate) {
        auto &s = useAlternate ? alternate : actual;
        std::vector<const void *> ptrs;
        for (int i = 0; i < requests; ++i) {
            ptrs.push_back((char *)s.key.cudaData + i * 136 * 192 * s.key.unitSize);
            ptrs.push_back((char *)s.value.cudaData + i * 136 * 128 * s.value.unitSize);
        }
        Cuda(cudaMemcpy(table, ptrs.data(), ptrs.size() * sizeof(void *), cudaMemcpyHostToDevice));
    }
    void Reset(int seed, int past, int rank) {
        Fill(branch, seed * 17 + 7);
        Fill(actual.hidden, seed + 801);
        Fill(actual.key, seed * 13 + rank, .4f);
        Fill(actual.value, seed * 19 + rank);
        for (auto *x :
             {&actual.q, &actual.attn, &actual.projected, &actual.normed, &actual.router, &actual.feature})
            Cuda(cudaMemset(x->cudaData, 0, x->GetBytes()));
        auto a = actual.Outputs(), b = reference.Outputs();
        for (size_t i = 0; i < a.size(); ++i)
            CopyDevice(*b[i], *a[i]);
        CopyDevice(alternate.key, actual.key);
        CopyDevice(alternate.value, actual.value);
        std::vector<float> pos(rows);
        std::vector<int> lengths(rows);
        for (int i = 0; i < rows; ++i) {
            pos[i] = float(past + owner[i] * 23 + offset[i]);
            lengths[i] = past < 0 ? 0 : int(pos[i]) + 1;
        }
        Cuda(cudaMemcpy(position.cudaData, pos.data(), rows * 4, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), rows * 4, cudaMemcpyHostToDevice));
    }
    std::vector<FastllmNaiveRopeCacheRow> Bindings(State &s, std::vector<Data> &keys,
                                                   std::vector<Data> &values, bool tables) {
        for (int i = 0; i < requests; ++i) {
            View(keys[i], s.key, i * 136 * 192, {1, 136, 192});
            keys[i].expansionDims = keys[i].dims;
            View(values[i], s.value, i * 136 * 128, {1, 136, 128});
            values[i].expansionDims = values[i].dims;
        }
        std::vector<FastllmNaiveRopeCacheRow> result;
        for (int i = 0; i < rows; ++i) {
            FastllmNaiveRopeCacheRow r{&keys[owner[i]], &values[owner[i]], nullptr};
            r.verifyOffset = offset[i];
            r.cache = tables ? table + owner[i] * 2 : nullptr;
            result.push_back(r);
        }
        return result;
    }
    void Legacy(State &s, int rank) {
        FastllmCudaSetLinearExactBatchThreshold(rows + 1);
        FastllmCudaNaiveAddDecodeRMSNorm(s.hidden, branch, inputNorm, config.eps, s.normed);
        CopyDevice(s.feature, s.hidden);
        std::vector<Data> keys(requests), values(requests);
        auto bindings = Bindings(s, keys, values, false);
        if (qkv.dataType == FP8_E4M3) {
            Init(scratch, {1, rows, 1856}, rank);
            Require(VerifyMxLinearReference(s.normed, qkv, bias, scratch, rows, 4096, 1856), "MXFP8 QKV");
        }
        for (int first = 0; first < rows; first += 8) {
            const int count = std::min(8, rows - first);
            Data n, q, p, l;
            View(n, s.normed, first * 4096, {1, count, 4096});
            View(q, s.q, first * 1536, {1, count, 1536});
            View(p, position, first, {count});
            View(l, live, first, {count});
            Init(kTemp, {1, count, 192}, rank);
            Init(vTemp, {1, count, 128}, rank);
            std::vector<FastllmNaiveRopeCacheRow> chunk(bindings.begin() + first,
                                                        bindings.begin() + first + count);
            if (qkv.dataType == FP8_E4M3) {
                Data raw;
                View(raw, scratch, size_t(first) * 1856, {1, count, 1856});
                Require(FastllmCudaNaiveBatchRopeAppendCache(q,kTemp,vTemp,p,chunk,l,8,1,192,128,64,
                    config.theta,config.valueScale,128,&raw,false),"MXFP8 reference RoPE");
            } else
            Require(FastllmCudaNaiveBatchQkvRopeAppendCache(n, qkv, bias, q, kTemp, vTemp, p, chunk, l, 8, 1,
                                                            192, 128, 64, config.theta, config.valueScale,
                                                            128, false, scratch),
                    "reference QKV/RoPE failed");
        }
        std::vector<FastllmNaiveSWARow> attention;
        for (auto &b : bindings)
            attention.push_back(
                {b.key->cudaData, b.value->cudaData, nullptr, b.key->dataType == FP8_E4M3, true, 0, b.verifyOffset});
        FastllmCudaNaiveBatchSWADecode(s.q, hasSink ? sink : bias, attention, live, 8, 1, s.attn);
        Require((o.dataType == FP8_E4M3 ? VerifyMxLinearReference : FastllmCudaBFloat16MatMulBFloat16)(s.attn, o, bias, s.projected, rows, 1024, 4096),
                "reference projection failed");
        Require(FastllmCudaCustomAllReduceRows(s.projected.cudaData, s.projected.cudaData, rows * 4096, 4096,
                                               BFLOAT16, rank),
                "reference collective failed");
        FastllmCudaNaiveAddDecodeRMSNorm(s.hidden, s.projected, postNorm, config.eps, s.normed);
        if (withRouter)
            Require(FastllmCudaNaiveRouterSigmoid(s.normed, routerWeight, s.router),
                    "reference router failed");
    }
    bool Fused(int rank, int invalid = 0) {
        FastllmCudaSetLinearExactBatchThreshold(rows + 1);
        std::vector<Data> keys(requests), values(requests);
        auto bindings = Bindings(actual, keys, values, indirect);
        Data b, h, qw, q, a, p, n, router, feature;
        View(b, branch, 0, branch.dims);
        View(h, actual.hidden, 0, actual.hidden.dims);
        View(qw, qkv, 0, qkv.dims);
        View(q, actual.q, 0, actual.q.dims);
        View(a, actual.attn, 0, actual.attn.dims);
        View(p, actual.projected, 0, actual.projected.dims);
        View(n, actual.normed, 0, actual.normed.dims);
        View(router, actual.router, 0, actual.router.dims);
        View(feature, actual.feature, 0, actual.feature.dims);
        auto c = config;
        if (rank == 3) {
            if (invalid == 1)
                b.strides.back() = 2;
            if (invalid == 2)
                qw.dataType = FLOAT32;
            if (invalid == 3)
                n.cudaData = h.cudaData;
            if (invalid == 4)
                b.Resize({1, rows + 1, 4096});
            if (invalid == 5)
                h.cudaData = (char *)h.cudaData + 2;
            if (invalid == 6)
                c.eps = -1;
            if (invalid == 7)
                n.Resize({16});
            if (invalid == 8)
                keys[0].expansionDims[1] = 135;
            if (invalid == 9)
                values[0].isPagedKVCache = true;
            if (invalid == 10)
                bindings[0].verifyOffset = 8;
            if (invalid == 11)
                qw.cudaData = nullptr;
            if (invalid == 12)
                p.cudaData = b.cudaData;
            if (invalid == 14)
                feature.cudaData = h.cudaData;
            if (invalid == 15)
                bindings[1].verifyOffset = bindings[0].verifyOffset;
        }
        const bool useOwnedWeight = qkv.dataType == FP8_E4M3 && !(rank == 3 && (invalid == 2 || invalid == 11));
        return FastllmCudaSwaVerifyLayer(b, h, inputNorm, useOwnedWeight ? qkv : qw, bias, o, postNorm, hasSink ? sink : bias,
                                         position, live, bindings, invalid == 13 && rank == 3 ? h : q, a, p,
                                         n, &feature, withRouter ? &routerWeight : nullptr,
                                         withRouter ? &router : nullptr, c, rank);
    }
    void Check(bool rebound = false) {
        const char *names[] = {"hidden", "normed", "q", "attention", "projection", "key", "value", "router"};
        auto a = actual.Outputs(), b = reference.Outputs();
        if (rebound) {
            a[5] = &alternate.key;
            a[6] = &alternate.value;
        }
        const auto liveBytes = Read(live);
        for (size_t kind : {5, 6, 2, 3, 4, 0, 1, 7}) {
            auto av = Read(*a[kind]), bv = Read(*b[kind]);
            const bool cache = kind == 5 || kind == 6;
            // Native MXFP8 QKV and BF16 attention retain both rounding boundaries.
            // FP8 standalone attention can use a different reduction order. Q, K and V still match byte for byte.
            if (testMxProjection && (kind == 2 || (kind == 3 && !testFp8KV) || cache) && av != bv)
                throw std::runtime_error(std::string(names[kind]) + " differs from native MXFP8 reference");

            if (cache && testFp8KV) { Require(av == bv, "FP8 KV byte mismatch"); continue; }
            const size_t width = cache ? (kind == 5 ? 192 : 128) : a[kind]->Count(0) / rows;
            for (int row = 0; row < rows; ++row) {
                int length;
                std::memcpy(&length, liveBytes.data() + row * 4, 4);
                if (cache && length <= 0)
                    continue;
                const size_t first = cache ? (owner[row] * 136 + 128 + offset[row]) * width : row * width;
                maxL2[kind] =
                    std::max(maxL2[kind], CompareRow(av, bv, a[kind]->dataType, first, width, names[kind]));
                // Mask only the candidate cells whose numerical error was
                // checked above, then compare every other cache byte exactly.
                if (cache)
                    std::copy_n(bv.begin() + first * 2, width * 2, av.begin() + first * 2);
            }
            if (cache)
                Require(av == bv, "untouched/inactive KV cache changed");
        }
        Require(Read(actual.feature) == Read(reference.feature), "draft feature differs");
    }
};
} // namespace
int main(int argc, char **argv) {
    try {
        testFp8KV = argc > 2 && std::strcmp(argv[2],"--fp8-kv")==0;
        const bool mx = argc > 1 && std::strcmp(argv[1],"--mxfp8")==0;
        testMxProjection = mx;
        const bool fusionDisabled = argc > 1 && std::strcmp(argv[1], "--disabled") == 0;
        const bool qkvDisabled = argc > 1 && std::strcmp(argv[1], "--qkv-disabled") == 0;
        const bool coreDisabled = argc > 1 && std::strcmp(argv[1], "--core-disabled") == 0;
        const bool disabled = fusionDisabled || qkvDisabled || coreDisabled;
        Env("FASTLLM_NAIVE_SWA_FUSION", fusionDisabled ? "0" : nullptr);
        Env("FASTLLM_NAIVE_SWA_CORE", coreDisabled ? "0" : nullptr);
        Env("FASTLLM_NAIVE_SWA_QKV_FUSION", qkvDisabled ? "0" : nullptr);
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8)
            return 77;
        for (int r = 0; r < 8; ++r) {
            cudaDeviceProp p{};
            Cuda(cudaGetDeviceProperties(&p, r));
            if (p.major != 10 || p.minor != 3)
                return 77;
        }
        const std::vector<int> devices{0, 1, 2, 3, 4, 5, 6, 7};
        if (!FastllmInitNccl(devices) || !FastllmCudaCustomAllReduceInit(devices))
            return 77;
        int cases = 0;
        std::array<double, 8> maximum{};
        for (int rows : testFp8KV ? std::vector<int>{2, 5, 8, 9, 16, 31, 36, 64} : mx ? std::vector<int>{2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64} : std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 36, 48, 63, 64, 8}) {
            std::array<std::unique_ptr<Tensors>, 8> data;
            std::array<int, 8> accepted{};
            Parallel([&](int r) {
                Require(disabled || FastllmCudaSwaVerifyLayerSupported(r, rows),
                        "verification backend unavailable");
                data[r] = std::make_unique<Tensors>(r, rows, rows == 36 || (mx && (rows == 5 || rows == 7)), mx);
            });
            if (disabled) {
                Parallel([&](int r) {
                    auto &d = *data[r];
                    d.Reset(53, 129, r);
                    std::vector<std::vector<uint8_t>> before;
                    for (auto *x : d.actual.Outputs())
                        before.push_back(Read(*x));
                    Require(!d.Fused(r), "disabled verification backend accepted");
                    Cuda(cudaDeviceSynchronize());
                    auto outputs = d.actual.Outputs();
                    for (size_t i = 0; i < outputs.size(); ++i)
                        Require(before[i] == Read(*outputs[i]), "disabled backend modified output/KV");
                });
            }
            // The QKV switch also disables the reference's specialization.
            // Test rejection above without assuming that fallback is available.
            if (!qkvDisabled)
                for (int past : {-1, 0, 127, 8192}) {
                    int seed = ++cases;
                    Parallel([&](int r) {
                        auto &d = *data[r];
                        d.hasSink = past != 0;
                        d.withRouter = past != 127;
                        d.config.valueScale = past == 0 ? .75f : 1.f;
                        d.Reset(seed, past, r);
                        d.Legacy(d.reference, r);
                        Cuda(cudaDeviceSynchronize());
                    });
                    Parallel([&](int r) {
                        auto &d = *data[r];
                        accepted[r] = d.Fused(r);
                        if (!accepted[r])
                            d.Legacy(d.actual, r);
                        Cuda(cudaDeviceSynchronize());
                        d.Check();
                    });
                    for (int ok : accepted)
                        Require(bool(ok) == !disabled, "default/disabled dispatch mismatch");
                }
            if (!disabled) {
                if (rows == 8)
                    for (int invalid = 1; invalid <= 15; ++invalid) {
                        Parallel([&](int r) {
                            auto &d = *data[r];
                            d.Reset(90 + invalid, 129, r);
                            std::vector<std::vector<uint8_t>> before;
                            for (auto *x : d.actual.Outputs())
                                before.push_back(Read(*x));
                            accepted[r] = d.Fused(r, invalid);
                            Cuda(cudaDeviceSynchronize());
                            auto out = d.actual.Outputs();
                            for (size_t i = 0; i < out.size(); ++i)
                                Require(before[i] == Read(*out[i]), "rejection modified output/KV");
                        });
                        for (int ok : accepted)
                            Require(!ok, "one bad rank did not reject the entire group");
                    }
                Parallel([&](int r) {
                    auto &d = *data[r];
                    d.indirect = true;
                    d.Bind(false);
                    if (mx) Require(FastllmCudaVerifyMxfp8Scales(d.qkv,1856,4096,r),"cold MXFP8 scales");
                    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                    accepted[r] = d.Fused(r);
                    cudaGraph_t graph = nullptr;
                    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    Cuda(cudaGraphInstantiate(&d.graph, graph, 0));
                    Cuda(cudaGraphDestroy(graph));
                });
                for (int ok : accepted)
                    Require(ok, "warm graph capture rejected");
                // Replay across the committed-key staging boundary on the same graph.
                for (int past : {1, 128, 129, 8192, 63, 95, 96, -1}) {
                    int seed = ++cases;
                    Parallel([&](int r) {
                        auto &d = *data[r];
                        d.Reset(seed, past, r);
                        d.Bind(true);
                        d.Legacy(d.reference, r);
                        Cuda(cudaDeviceSynchronize());
                    });
                    Parallel([&](int r) {
                        auto &d = *data[r];
                        auto k = Read(d.actual.key), v = Read(d.actual.value);
                        Cuda(cudaGraphLaunch(d.graph, cudaStreamPerThread));
                        Cuda(cudaDeviceSynchronize());
                        d.Check(true);
                        Require(k == Read(d.actual.key) && v == Read(d.actual.value),
                                "graph ignored KV rebinding");
                    });
                }
            }
            for (auto &d : data)
                for (size_t kind = 0; kind < maximum.size(); ++kind)
                    maximum[kind] = std::max(maximum[kind], d->maxL2[kind]);
            Parallel([&](int r) {
                if (data[r]->graph)
                    Cuda(cudaGraphExecDestroy(data[r]->graph));
                Cuda(cudaFree(data[r]->table));
                data[r].reset();
            });
            std::printf("SWA verify layer rows=%d PASS\n", rows);
        }
        std::printf("maximum per-row relative L2: hidden=%g normed=%g q=%g attention=%g projection=%g key=%g "
                    "value=%g router=%g\n",
                    maximum[0], maximum[1], maximum[2], maximum[3], maximum[4], maximum[5], maximum[6],
                    maximum[7]);
        FastllmCudaCustomAllReduceReset();
        std::printf(
            "SWA verification layer PASS: enabled=%d, %d TP8 cases, fallback and fixed graph replay\n",
            !disabled, cases);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "SWA verification layer FAIL: %s\n", e.what());
        return 1;
    }
}
