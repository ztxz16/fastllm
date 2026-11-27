#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "utils/utils.h"
#include "naive_n05_mxfp8_test.h"
#include <cuda_runtime_api.h>
#include <tuple>
#include <numeric>
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
void Cuda(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
template <class F> void Parallel(F fn) {
    std::vector<std::thread> workers;
    for (int r = 0; r < 8; ++r)
        workers.emplace_back([&, r] {
            try {
                FastllmCudaSetDevice(r);
                fn(r);
            } catch (const std::exception &e) {
                // Peers may already be inside the next collective.
                std::fprintf(stderr, "Hopper verification rank %d: %s\n", r, e.what());
                std::_Exit(EXIT_FAILURE);
            }
        });
    for (auto &worker : workers)
        worker.join();
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
        } else
            std::memcpy(bytes.data() + i * 4, &value, 4);
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
void CheckRow(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, size_t first, size_t width,
              double tolerance, const char *name) {
    double error = 0, magnitude = 0;
    auto bf16 = [](const uint8_t *p) {
        uint16_t u;
        std::memcpy(&u, p, 2);
        uint32_t bits = uint32_t(u) << 16;
        float x;
        std::memcpy(&x, &bits, 4);
        return double(x);
    };
    for (size_t i = first; i < first + width; ++i) {
        const double x = bf16(a.data() + i * 2), y = bf16(b.data() + i * 2);
        Require(std::isfinite(x) && std::isfinite(y), "non-finite BF16 output");
        error += (x - y) * (x - y);
        magnitude += y * y;
    }
    if (std::sqrt(error) > tolerance * std::sqrt(magnitude) + 1.e-7)
        throw std::runtime_error(std::string(name) +
                                 " relative L2=" + std::to_string(std::sqrt(error / (magnitude + 1.e-30))));
}
struct Sample {
    Data q{BFLOAT16}, attn{BFLOAT16}, projected{BFLOAT16}, hidden{BFLOAT16}, normed{BFLOAT16};
    Data key, value;
    explicit Sample(DataType type) : key(type), value(type) {}
};
void Test(int gpu, int rows, bool ring, DataType cacheType = FP8_E4M3, int requestWidth = 8) {
    const bool sharded = !ring;
    Data input(BFLOAT16), qw(FP8_E4M3), ow(FP8_E4M3), bias, gamma(FLOAT32), sink(FLOAT32), pos(FLOAT32),
        live(INT32), tmp(FLOAT32), qkv(BFLOAT16), k(BFLOAT16), v(BFLOAT16), sel(INT32);
    const int requests = (rows + requestWidth - 1) / requestWidth, cap = ring ? 128 + requestWidth : 4224, stride = ring ? 128 : 2048;
    const bool fp8Cache = cacheType == FP8_E4M3;
    const int cacheBytes = fp8Cache ? 1 : 2;
    Sample real(cacheType), ref(cacheType);
    Init(input, {1, rows, 4096}, gpu);
    Init(qw, {1856, 4096}, gpu);
    Init(ow, {4096, 1024}, gpu);
    Init(gamma, {4096}, gpu);
    Init(sink, {8}, gpu);
    Init(pos, {1, rows}, gpu);
    Init(live, {rows}, gpu);
    Init(qkv, {1, rows, 1856}, gpu);
    Init(k, {1, rows, 192}, gpu);
    Init(v, {1, rows, 128}, gpu);
    Init(sel, {rows, stride}, gpu);
    Fill(input, 13);
    Fill(gamma, 31, .2, 1);
    Fill(sink, 53);
    FillVerifyMxWeight(qw, 71 + gpu);
    FillVerifyMxWeight(ow, 73 + gpu);
    auto alloc = [&](Sample &s) {
        for (auto *x : {&s.projected, &s.hidden, &s.normed})
            Init(*x, {1, rows, 4096}, gpu);
        Init(s.q, {1, rows, 1536}, gpu);
        Init(s.attn, {1, rows, 1024}, gpu);
        Init(s.key, {requests, cap, 192}, gpu);
        Init(s.value, {requests, cap, 128}, gpu);
    };
    alloc(real);
    alloc(ref);
    Data alternateK(cacheType), alternateV(cacheType);
    Init(alternateK, real.key.dims, gpu);
    Init(alternateV, real.value.dims, gpu);
    std::vector<int> ids(rows * stride), length(rows);
    std::vector<float> positions(rows);
    std::vector<const void *> cpuTable(requests * 8);
    const void **table = nullptr;
    Cuda(cudaMalloc(&table, cpuTable.size() * sizeof(void *)));
    std::vector<FastllmNaiveDSARow> bind(rows);
    for (int r = 0; r < rows; ++r) {
        auto &b = bind[r];
        b.key = real.key.cudaData;
        b.value = real.value.cudaData;
        b.keyStride = 192;
        b.capacity = ring ? 8192 : cap;
        b.cache = table + r / requestWidth * 8;
        b.fp8 = fp8Cache;
        b.verifyOffset = ring ? r % requestWidth : -1;
        b.liveKeys = (const int *)live.cudaData + r;
        b.indices = ring ? nullptr : (const int *)sel.cudaData + size_t(r) * stride;
        b.selectedCount = ring ? 0 : stride;
    }

    Data iwq(BFLOAT16), iwk(BFLOAT16), iww(BFLOAT16), gammaI(FLOAT32), biasI(FLOAT32), indices(INT32),
        pages(INT32), indexPool(FP8_E4M3);
    FastllmNaiveHopperIndexer idx{&iwq, &iwk, &iww, &gammaI, &biasI, &indices, gpu};
    if (sharded) {
        Init(iwq, {2048, 4096}, gpu);
        Init(iwk, {128, 4096}, gpu);
        Init(iww, {16, 4096}, gpu);
        for (auto *x : {&iwq, &iwk, &iww})
            Cuda(cudaMemset(x->cudaData, 0, x->GetBytes()));
        auto set = [&](Data &v, int offset, float f) {
            auto bit = Float32ToBFloat16RNEBits(f);
            Cuda(cudaMemcpy((char *)v.cudaData + offset * 2, &bit, 2, cudaMemcpyHostToDevice));
        };
        // One nonrotary query channel and one nonzero head give an independent
        // CPU score oracle. Repeated FP8 key values exercise ties across owners.
        set(iwq, 64 * 4096 + 4094, 1.f);
        set(iww, 4095, 4.f);
        Init(gammaI, {128}, gpu);
        Init(biasI, {128}, gpu);
        Fill(gammaI, 13, 0, 1);
        float one = 1;
        Cuda(cudaMemset(biasI.cudaData, 0, 512));
        Cuda(cudaMemcpy((float *)biasI.cudaData + 64, &one, 4, cudaMemcpyHostToDevice));
        Init(pages, {cap / 128}, gpu);
        Init(indexPool, {requests, cap / 8, 144}, gpu);
        for (auto &r : bind)
            r.pageLen = 128;
    }
    Data routeWeight(FLOAT32), route(FLOAT32), referenceRoute(FLOAT32);
    Init(routeWeight, {256, 4096}, gpu);
    Fill(routeWeight, 73, .05f);
    // Only ranks consuming draft context need a feature output.
    const bool snapshot = ring && rows <= 8 && gpu % 2 == 0;
    Data feature(BFLOAT16);
    if (snapshot) Init(feature, real.hidden.dims, gpu);
    std::shared_ptr<void> wideBindings;
    auto run = [&](bool invalid = false) {
        return FastllmCudaNaiveHopperVerifyLayer(input, qw, (invalid && gpu == 3 ? sink : bias), ow, gamma,
                                                 (ring ? sink : bias), pos, real.q, real.attn, real.projected,
                                                 real.hidden, real.normed, tmp, bind, stride, ring, 10000.f,
                                                 .75f, 1.e-5f, gpu, sharded ? &idx : nullptr,
                                                 &routeWeight, &route, &wideBindings, snapshot ? &feature : nullptr);
    };

    cudaGraphExec_t graph = nullptr;
    for (int iter = 0; iter < (ring ? 3 : 2); ++iter) {
        auto &key = iter ? alternateK : real.key;
        auto &value = iter ? alternateV : real.value;
        for (int seq = 0; seq < requests; ++seq) {
            cpuTable[seq * 8] = (char *)key.cudaData + seq * cap * 192 * cacheBytes;
            cpuTable[seq * 8 + 1] = (char *)value.cudaData + seq * cap * 128 * cacheBytes;
        }
        std::vector<int> pageOrder(cap / 128);
        std::iota(pageOrder.begin(), pageOrder.end(), 0);
        if (sharded) {
            if (iter)
                for (int i = 0; i < cap / 128; ++i)
                    pageOrder[i] = (i * 2 + 5) % (cap / 128);
            Cuda(cudaMemcpy(pages.cudaData, pageOrder.data(), pageOrder.size() * 4, cudaMemcpyHostToDevice));
            std::vector<uint8_t> encoded(indexPool.GetBytes(), 0);
            for (int seq = 0; seq < requests; ++seq) {
                cpuTable[seq * 8 + 2] = cpuTable[seq * 8 + 3] = pages.cudaData;
                cpuTable[seq * 8 + 4] = (char *)indexPool.cudaData + seq * cap / 8 * 144;
                for (int k = 0; k < cap / 8; ++k) {
                    int token = k / 16 * 128 + gpu * 16 + k % 16;
                    auto *dst = encoded.data() + (seq * cap / 8 + pageOrder[k / 16] * 16 + k % 16) * 144;
                    float scale = .5f + float((token * 53) % 193) / 128.f;
                    dst[64] = 0x38;
                    std::memcpy(dst + 128, &scale, 4);
                }
            }
            Cuda(cudaMemcpy(indexPool.cudaData, encoded.data(), encoded.size(), cudaMemcpyHostToDevice));
        }
        Cuda(cudaMemcpy(table, cpuTable.data(), cpuTable.size() * sizeof(void *), cudaMemcpyHostToDevice));
        for (int r = 0; r < rows; ++r) {
            const int past = ring ? (iter == 2 ? 0 : 127 + iter) : (iter ? 2045 : 4089);
            const int n = past + r % requestWidth + 1;
            length[r] = (r == rows - 1 && iter) ? 0 : n;
            positions[r] = n - 1;
            if (sharded) {
                std::vector<int> expected(length[r]);
                for (int j = 0; j < length[r]; ++j)
                    expected[j] = j;
                auto score = [&](int k) {
                    return k >= past ? ((448.f * 448.f) * (1.f / 448.f)) * (1.f / 448.f)
                                     : .5f + float((k * 53) % 193) / 128.f;
                };
                std::stable_sort(expected.begin(), expected.end(),
                                 [&](int x, int y) { return score(x) > score(y); });
                for (int j = 0; j < stride; ++j)
                    ids[r * stride + j] = j < length[r] ? expected[j] : -1;
            }
        }
        Cuda(cudaMemcpy(sel.cudaData, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(pos.cudaData, positions.data(), rows * 4, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(live.cudaData, length.data(), rows * 4, cudaMemcpyHostToDevice));
        auto cache = [&](Data &x, int seed) {
            if (!fp8Cache) {
                Fill(x, seed);
                return;
            }
            std::vector<unsigned char> buf(x.GetBytes());
            for (auto &v : buf) {
                seed = int(unsigned(seed) * 1664525u + 1013904223u);
                v = uint8_t((unsigned(seed) >> 24) % 52 + 16) | ((seed >> 14) & 128);
            }
            Cuda(cudaMemcpy(x.cudaData, buf.data(), buf.size(), cudaMemcpyHostToDevice));
        };
        cache(key, 17 + gpu + iter);
        cache(value, 71 + gpu + iter);
        auto logicalCache = [&](const Data &cache) {
            auto physical = Read(cache);
            if (!sharded || !iter)
                return physical;
            auto logical = physical;
            const size_t bytes = 128 * cache.dims.back();
            for (int seq = 0; seq < requests; ++seq)
                for (int pg = 0; pg < cap / 128; ++pg)
                    std::memcpy(logical.data() + (seq * (cap / 128) + pg) * bytes,
                                physical.data() + (seq * (cap / 128) + pageOrder[pg]) * bytes, bytes);
            return logical;
        };
        for (auto [src, dst] : {std::pair{&key, &ref.key}, std::pair{&value, &ref.value}}) {
            auto logical = logicalCache(*src);
            Cuda(cudaMemcpy(dst->cudaData, logical.data(), logical.size(), cudaMemcpyHostToDevice));
        }
        Fill(real.hidden, 503 + iter);
        CopyDevice(ref.hidden, real.hidden);
        Fill(input, 91 + iter);
        if (sharded) {
            std::vector<uint16_t> ones(rows * 2, Float32ToBFloat16RNEBits(1.f));
            Cuda(cudaMemcpy2D((char *)input.cudaData + 4094 * 2, 4096 * 2, ones.data(), 4, 4, rows,
                              cudaMemcpyHostToDevice));
        }
        Require(FastllmCudaBFloat16MatMulFP8E4M3(input, qw, bias, qkv, rows, 4096, 1856), "reference QKV");
        // The standalone append reference accepts offsets 0..7. Shift a
        // shallow cache view for each subsequent group of eight candidates.
        const int groups = (requestWidth + 7) / 8;
        std::vector<Data> ks(requests * groups), vs(requests * groups);
        for (int i = 0; i < requests * groups; ++i) {
            int shift = i % groups * 8;
            size_t base = size_t(i / groups * cap + shift) * cacheBytes;
            ks[i].Resize({1, cap - shift, 192});
            ks[i].FakeFrom(ref.key, base * 192);
            ks[i].expansionDims = ks[i].dims;
            vs[i].Resize({1, cap - shift, 128});
            vs[i].FakeFrom(ref.value, base * 128);
            vs[i].expansionDims = vs[i].dims;
        }
        std::vector<FastllmNaiveRopeCacheRow> append;
        std::vector<FastllmNaiveSWARow> sr;
        std::vector<FastllmNaiveDSARow> dr;
        for (int r = 0; r < rows; ++r) {
            const int cache = r / requestWidth * groups;
            const int offset = r % requestWidth;
            FastllmNaiveRopeCacheRow x{&ks[cache + offset / 8], &vs[cache + offset / 8], nullptr};
            x.verifyOffset = ring ? offset % 8 : -1;
            append.push_back(x);
            sr.push_back({ks[cache].cudaData, vs[cache].cudaData, nullptr, fp8Cache, true, 0, r % requestWidth});
            auto b = bind[r];
            b.key = ks[cache].cudaData;
            b.value = vs[cache].cudaData;
            b.cache = nullptr;
            b.pageLen = 0;
            dr.push_back(b);
        }
        Require(FastllmCudaNaiveBatchRopeAppendCache(ref.q, k, v, pos, append, live, 8, 1, 192, 128, 64,
                                                     10000.f, .75f, ring ? 128 : 0, &qkv, false),
                "reference append");
        FastllmNaiveDecodeScratch scratch;
        auto referenceStages = [&](bool isolate) {
            const Data &query = isolate ? real.q : ref.q;
            if (ring)
                FastllmCudaNaiveBatchSWADecode(query, sink, sr, live, 8, 1, ref.attn);
            else
                FastllmCudaNaiveBatchDSADecode(query, bias, dr, 8, 1, 192, 128, scratch, ref.attn);
            Require(FastllmCudaBFloat16MatMulFP8E4M3(isolate ? real.attn : ref.attn, ow, bias, ref.projected,
                                                     rows, 1024, 4096), "reference O");
            Require(FastllmCudaCustomAllReduceAddRMSNorm(isolate ? real.projected.cudaData : ref.projected.cudaData,
                        ref.hidden.cudaData, (const float *)gamma.cudaData, ref.normed.cudaData, rows * 4096,
                        4096, 1.e-5f, BFLOAT16, gpu), "reference reduce/norm");
        };
        if (rows <= FastllmNaiveDSABatchLimit) referenceStages(false);
        if (!iter) {
            Require(run(), "Hopper verification rejected a supported case");
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            Fill(real.hidden, 503 + iter);
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Require(run(), "Hopper verification rejected a supported case");
            cudaGraph_t g = nullptr;
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &g));
            size_t nodes = 0;
            Cuda(cudaGraphGetNodes(g, nullptr, &nodes));
            Require(nodes == (ring && rows <= 8 && !fp8Cache ? 4 : snapshot ? 2 : 1), "unexpected Hopper verify graph node count");
            Cuda(cudaGraphInstantiate(&graph, g, 0));
            Cuda(cudaGraphDestroy(g));
        }
        if (ring && rows <= 8) {
            // Interleave an ordinary tagged collective of a different size with
            // SWA replay, and check that neither call consumes the other's tags.
            Data src(BFLOAT16), dst(BFLOAT16);
            const int width = 64 * (iter + 1);
            Init(src, {width}, gpu);
            Init(dst, {width}, gpu);
            Fill(src, 0, 0, float(gpu + iter + 1));
            Require(FastllmCudaCustomAllReduce(src.cudaData, dst.cudaData, width, BFLOAT16, gpu),
                    "tagged all-reduce rejected");
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            Fill(src, 0, 0, float(36 + iter * 8));
            Require(Read(src) == Read(dst), "ordinary reduction consumed stale SWA data");
        }
        const auto previous = snapshot ? Read(real.hidden) : std::vector<uint8_t>{};
        Cuda(cudaGraphLaunch(graph, cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        if (snapshot) Require(Read(feature) == previous, "draft feature differs from pre-attention residual");
        {
            int previous = FastllmCudaGetLinearExactBatchThreshold();
            FastllmCudaSetLinearExactBatchThreshold(std::max(rows + 1, 16));
            Require(FastllmCudaNaiveRouterSigmoid(real.normed, routeWeight, referenceRoute), "reference router");
            FastllmCudaSetLinearExactBatchThreshold(previous);
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            Require(Read(route) == Read(referenceRoute), "router probabilities differ");
        }
        if (sharded) {
            std::vector<int> got(ids.size());
            Cuda(cudaMemcpy(got.data(), indices.cudaData, got.size() * 4, cudaMemcpyDeviceToHost));
            Require(got == ids, "distributed TopK differs from stable score/position ordering");
        }
        for (auto [a, b] : {std::pair{&key, &ref.key}, std::pair{&value, &ref.value}}) {
            auto x = logicalCache(*a), y = Read(*b);
            auto fp8 = [](uint8_t bits) {
                const int e = (bits >> 3) & 15, m = bits & 7;
                float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(float(m), -9);
                return bits & 128 ? -v : v;
            };
            int width = a->dims.back();
            for (int seq = 0; seq < requests; ++seq)
                for (int slot = 0; slot < cap; ++slot) {
                    size_t offset = size_t(seq * cap + slot) * width;
                    if (!std::memcmp(x.data() + offset * cacheBytes, y.data() + offset * cacheBytes,
                                     width * cacheBytes))
                        continue;
                    bool written = false;
                    for (int r = 0; r < rows; ++r)
                        if (r / requestWidth == seq && length[r] > 0 && slot == (ring ? 128 + r % requestWidth : length[r] - 1))
                            written = true;
                    Require(written, "cache write outside live candidate rows");
                    if (!fp8Cache) {
                        CheckRow(x, y, offset, width, .003, "BF16 cache");
                        continue;
                    }
                    double error = 0, magnitude = 0;
                    for (int col = 0; col < width; ++col) {
                        double u = fp8(x[offset + col]), v = fp8(y[offset + col]);
                        error += (u - v) * (u - v);
                        magnitude += v * v;
                    }
                    double rel = std::sqrt(error / (magnitude + 1e-30));
                    if (rel > .025)
                        throw std::runtime_error("FP8 cache row relative L2=" + std::to_string(rel));
                }
        }
        // Isolate each stage on large batches. A different QKV GEMM reduction
        // can move a value across an FP8 cache bin; propagating that difference
        // would measure the cache quantizer a second time in the attention check.
        if (rows > FastllmNaiveDSABatchLimit) {
            for (auto [actual, logical] : {std::pair{&key, &ref.key}, std::pair{&value, &ref.value}}) {
                auto bytes = logicalCache(*actual);
                Cuda(cudaMemcpy(logical->cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
            }
            referenceStages(true);
            Require(Read(ref.hidden) == Read(real.hidden), "wide residual differs");
        }
        for (auto [actual, expected, name] : std::initializer_list<std::tuple<Data *, Data *, const char *>>{
                 {&real.q, &ref.q, "query"},
                 {&real.attn, &ref.attn, "attention"},
                 {&real.projected, &ref.projected, "projection"},
                 {&real.hidden, &ref.hidden, "residual"},
                 {&real.normed, &ref.normed, "norm"}}) {
            auto x = Read(*actual), y = Read(*expected);
            const int width = actual->dims.back();
            // The two Tensor Core paths use different FP32 accumulation orders.
            // O also requantizes the rounded attention vector to FP8.
            const double tolerance = actual == &real.q ? .003 : actual == &real.projected ? .02 : .01;
            for (int r = 0; r < rows; ++r)
                CheckRow(x, y, r * width, width, tolerance, name);
        }
        if (ring && rows >= 2 && rows <= 16) {
            Fill(real.hidden, 1031 + gpu + iter);
            CopyDevice(ref.hidden, real.hidden);
            Require(FastllmCudaCustomAllReduceAddRMSNorm(real.projected.cudaData, ref.hidden.cudaData,
                        static_cast<const float *>(gamma.cudaData), ref.normed.cudaData, rows * 4096, 4096,
                        1.e-5f, BFLOAT16, gpu), "standalone reduce/norm");
            const int threshold = FastllmCudaGetLinearExactBatchThreshold();
            FastllmCudaSetLinearExactBatchThreshold(std::max(threshold, rows + 1));
            Require(FastllmCudaNaiveRouterSigmoid(ref.normed, routeWeight, referenceRoute), "standalone router");
            FastllmCudaSetLinearExactBatchThreshold(threshold);
            Require(FastllmCudaNaiveHopperReduceNormRouter(real.projected, real.hidden, gamma, real.normed,
                        routeWeight, route, 1.e-5f, gpu), "split SWA tail");
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            Require(Read(real.hidden) == Read(ref.hidden) && Read(real.normed) == Read(ref.normed) &&
                    Read(route) == Read(referenceRoute), "split SWA tail differs");
            Require(!FastllmCudaNaiveHopperReduceNormRouter(real.projected, real.hidden, gamma,
                        gpu == 3 ? real.hidden : real.normed, routeWeight, route, 1.e-5f, gpu),
                    "one aliased rank should reject the split tail on all ranks");
        }
    }
    const auto saved = Read(real.hidden);
    Require(!run(true), "one invalid rank should reject the group");
    Require(Read(real.hidden) == saved, "rejected call updated the residual");
    Cuda(cudaGraphExecDestroy(graph));
    Cuda(cudaFree(table));
}
// A short physical pool and a long page table exercise the model's full
// logical window. Only historical pages alias; appended tokens own new pages.
void TestReplicated(int gpu) {
    constexpr int rows = 8, page = 128, pool = 35;
    Data input(BFLOAT16), qw(FP8_E4M3), ow(FP8_E4M3), bias, gamma(FLOAT32), pos(FLOAT32), live(INT32),
        scratch(FLOAT32), q(BFLOAT16), attn(BFLOAT16), projected(BFLOAT16), hidden(BFLOAT16), normed(BFLOAT16),
        iq(BFLOAT16), ik(BFLOAT16), iw(BFLOAT16), inorm(FLOAT32), ibias(FLOAT32), selected(INT32), keys(BFLOAT16),
        values(BFLOAT16), pages(INT32), ref(BFLOAT16);
    Init(input, {1, rows, 4096}, gpu);
    Init(hidden, input.dims, gpu);
    Init(qw, {1856, 4096}, gpu);
    Init(ow, {4096, 1024}, gpu);
    FillVerifyMxWeight(qw, 71 + gpu);
    FillVerifyMxWeight(ow, 73 + gpu);
    Init(gamma, {4096}, gpu);
    Fill(gamma, 31, .2f, 1.f);
    Init(pos, {rows}, gpu);
    Init(live, {rows}, gpu);
    Init(iq, {2048, 4096}, gpu);
    Init(ik, {128, 4096}, gpu);
    Init(iw, {16, 4096}, gpu);
    Init(inorm, {128}, gpu);
    Init(ibias, {128}, gpu);
    Init(keys, {1, pool * page, 320}, gpu);
    Init(values, {1, pool * page, 128}, gpu);
    Init(pages, {1048576 / page}, gpu);
    Fill(values, 45 + gpu);
    for (auto *x : {&iq, &ik, &iw, &input, &ibias}) Cuda(cudaMemset(x->cudaData, 0, x->GetBytes()));
    Fill(inorm, 1, 0, 1);
    auto querySign = [](int row, int head) { return (row * 3 + head * 5) % 7 < 3 ? -1 : 1; };
    auto weight = [](int row, int head) { return (row + head * 7) % 3 - 1; };
    auto keyValue = [](int physical, int head) {
        constexpr int values[] = {-4, -2, -1, 0, 1, 2, 4};
        return values[((physical / (head + 1) + head) * 53) % 7];
    };
    const auto oneB = Float32ToBFloat16RNEBits(1.f), fourB = Float32ToBFloat16RNEBits(4.f);
    for (int h = 0; h < 16; ++h) {
        const float key = h % 2 ? -1.f : 1.f;
        Cuda(cudaMemcpy(static_cast<float *>(ibias.cudaData) + 64 + h, &key, 4, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(static_cast<uint16_t *>(iq.cudaData) + (h * 128 + 64 + h) * 4096 + h, &oneB, 2,
                        cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(static_cast<uint16_t *>(iw.cudaData) + h * 4096 + 32 + h, &fourB, 2,
                        cudaMemcpyHostToDevice));
        for (int row = 0; row < rows; ++row) {
            const auto x = Float32ToBFloat16RNEBits(float(querySign(row, h)));
            const auto w = Float32ToBFloat16RNEBits(float(weight(row, h)));
            auto *in = static_cast<uint16_t *>(input.cudaData) + row * 4096;
            Cuda(cudaMemcpy(in + h, &x, 2, cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(in + 32 + h, &w, 2, cudaMemcpyHostToDevice));
        }
    }
    const void *bindings[] = {keys.cudaData, values.cudaData, pages.cudaData, pages.cudaData};
    const void **table = nullptr;
    Cuda(cudaMalloc(&table, sizeof(bindings)));
    Cuda(cudaMemcpy(table, bindings, sizeof(bindings), cudaMemcpyHostToDevice));
    FastllmNaiveHopperIndexer index{&iq, &ik, &iw, &inorm, &ibias, &selected, gpu};
    for (int cap : {1024, 2048, 2049, 8193, 16391, 65537, 1048576}) {
        std::vector<FastllmNaiveDSARow> desc(rows);
        for (int r = 0; r < rows; ++r) {
            auto &d = desc[r];
            d.key = keys.cudaData;
            d.value = values.cudaData;
            d.cache = table;
            d.capacity = cap;
            d.keyStride = 320;
            d.pageLen = page;
            d.keyPages = d.valuePages = static_cast<const int *>(pages.cudaData);
            d.liveKeys = static_cast<const int *>(live.cudaData) + r;
        }
        auto run = [&] {
            return FastllmCudaNaiveHopperVerifyLayer(input, qw, bias, ow, gamma, bias, pos, q, attn, projected,
                                                     hidden, normed, scratch, desc, std::min(2048, cap), false,
                                                     10000.f, 1.f, 1.e-5f, gpu, &index, nullptr, nullptr);
        };
        cudaGraphExec_t exec = nullptr;
        for (int iter = 0; iter < 2; ++iter) {
            const int end = iter ? std::min(cap, 4099) : cap;
            const int prefix = end - rows;
            std::vector<int> map((cap + page - 1) / page);
            for (int p = 0; p < int(map.size()); ++p)
                map[p] = p < prefix / page ? (p * 11 + iter) % 32 : 32 + (p - prefix / page) % 3;
            Cuda(cudaMemcpy(pages.cudaData, map.data(), map.size() * 4, cudaMemcpyHostToDevice));
            // Many equal maxima deliberately cross pages, CTAs and the 2048th position.
            std::vector<uint16_t> image(pool * page * 320);
            for (int physical = 0; physical < pool * page; ++physical)
                for (int h = 0; h < 16; ++h)
                    image[physical * 320 + 192 + 64 + h] = Float32ToBFloat16RNEBits(float(keyValue(physical, h)));
            Cuda(cudaMemcpy(keys.cudaData, image.data(), image.size() * 2, cudaMemcpyHostToDevice));
            std::vector<int> lengths(rows);
            std::vector<float> positions(rows);
            for (int r = 0; r < rows; ++r) {
                lengths[r] = iter && r == rows - 1 ? 0 : prefix + r + 1;
                positions[r] = prefix + r;
            }
            Cuda(cudaMemcpy(live.cudaData, lengths.data(), rows * 4, cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(pos.cudaData, positions.data(), rows * 4, cudaMemcpyHostToDevice));
            Fill(hidden, 61);
            if (!iter) {
                Require(run(), "replicated DSA admission");
                Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                Require(run(), "replicated DSA capture");
                cudaGraph_t graph = nullptr;
                Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                size_t n = 0;
                Cuda(cudaGraphGetNodes(graph, nullptr, &n));
                Require(n == 3, "replicated DSA graph node count");
                Cuda(cudaGraphInstantiate(&exec, graph, 0));
                Cuda(cudaGraphDestroy(graph));
            }
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            // Dense replays do not score keys, but a later sparse graph will read
            // the Indexer keys appended while this request still fit in Top-K.
            for (int row = 0; row < rows; ++row) {
                if (!lengths[row]) continue;
                const int token = lengths[row] - 1;
                const int physical = map[token / page] * page + token % page;
                uint16_t appended[128];
                Cuda(cudaMemcpy(appended, static_cast<uint16_t *>(keys.cudaData) + physical * 320 + 192,
                                sizeof(appended), cudaMemcpyDeviceToHost));
                for (int d = 0; d < 128; ++d) {
                    const float k = d >= 64 && d < 80 ? (d % 2 ? -1.f : 1.f) : 0.f;
                    Require((k == 0.f ? (appended[d] & 0x7fff) == 0 :
                             appended[d] == Float32ToBFloat16RNEBits(k)), "replicated Indexer key append");
                }
            }
            if (cap > 2048) {
                auto raw = Read(selected);
                for (int r = 0; r < rows; ++r) {
                    std::vector<int> score(pool * page + 1), expected(2048, -1);
                    for (int physical = 0; physical <= pool * page; ++physical)
                        for (int h = 0; h < 16; ++h) {
                            int k = physical == pool * page ? (h % 2 ? -1 : 1) : keyValue(physical, h);
                            score[physical] += std::max(0, querySign(r, h) * k) * weight(r, h);
                        }
                    auto bucket = [&](int t) {
                        return 64 + score[t >= prefix ? pool * page : map[t / page] * page + t % page];
                    };
                    // Stable counting sort is the CPU oracle. All products are
                    // small integers, exactly representable throughout the Indexer.
                    int counts[129]{}, cursor[129]{};
                    for (int t = 0; t < lengths[r]; ++t) ++counts[bucket(t)];
                    for (int b = 127; b >= 0; --b) cursor[b] = cursor[b + 1] + counts[b + 1];
                    for (int t = 0; t < lengths[r]; ++t) {
                        int slot = cursor[bucket(t)]++;
                        if (slot < 2048) expected[slot] = t;
                    }
                    Require(!std::memcmp(raw.data() + r * 2048 * 4, expected.data(), 2048 * 4),
                            "replicated DSA exact ordered TopK");
                    desc[r].indices = static_cast<int *>(selected.cudaData) + r * 2048;
                    desc[r].selectedCount = 2048;
                }
            }
            if (cap <= 2048) {
                auto raw = Read(selected);
                for (int row = 0; row < rows; ++row)
                    for (int t = 0; t < cap; ++t) {
                        int id;
                        std::memcpy(&id, raw.data() + (row * cap + t) * sizeof(int), sizeof(int));
                        Require(id == (t < lengths[row] ? t : -1), "replicated dense indices");
                    }
            }
            FastllmNaiveDecodeScratch decode;
            FastllmCudaNaiveBatchDSADecode(q, bias, desc, 8, 1, 192, 128, decode, ref);
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            auto a = Read(attn), b = Read(ref);
            for (int r = 0; r < rows; ++r) CheckRow(a, b, r * 1024, 1024, .006, "replicated attention");
        }
        Cuda(cudaGraphExecDestroy(exec));
        if (!gpu) std::fprintf(stderr, "Replicated DSA capacity %d passed\n", cap);
    }
    Cuda(cudaFree(table));
}

} // namespace
int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8)
            return 77;
        for (int d = 0; d < 8; ++d) {
            cudaDeviceProp p{};
            Cuda(cudaGetDeviceProperties(&p, d));
            if (p.major != 9 || p.minor != 0)
                return 77;
        }
        std::vector<int> devices{0, 1, 2, 3, 4, 5, 6, 7};
        if (!FastllmInitNccl(devices) || !FastllmCudaCustomAllReduceInit(devices))
            return 77;
        for (bool ring : {false, true})
            for (int rows : {1, 2, 3, 8, 11, 16, 32, 33, 79, 129}) {
                std::fprintf(stderr, "Testing %s, %d rows\n", ring ? "SWA" : "DSA", rows);
                Parallel([&](int gpu) { Test(gpu, rows, ring); });
            }
        for (int rows : {1, 3, 4, 5, 7, 8}) Parallel([=](int gpu) { Test(gpu, rows, true, BFLOAT16); });
        Parallel([](int gpu) { Test(gpu, 8, true, BFLOAT16, 4); });
        Parallel([](int gpu) { Test(gpu, 17, true); });
        Parallel([](int gpu) { Test(gpu, 31, true, BFLOAT16); });
        Parallel([](int gpu) { Test(gpu, 35, true, FP8_E4M3, 17); });
        Parallel(TestReplicated);
        FastllmCudaCustomAllReduceReset();
        puts("PASS");
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
