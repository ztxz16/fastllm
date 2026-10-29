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

static void BatchDraft(int count, int rows, int dim, int heads, int window) {
    const int kh = 4, qw = heads * dim, kw = kh * dim, cap = window + rows;
    Data raw = BF({1, rows * count, qw + 2 * kw}, 119);
    Data qn(FLOAT32, {dim}), kn(FLOAT32, {dim});
    qn.Allocate(); kn.Allocate();
    for (int i = 0; i < dim; ++i) {
        ((float *)qn.cpuData)[i] = .9f + std::sin(i) * .2f;
        ((float *)kn.cpuData)[i] = 1.1f + std::cos(i) * .3f;
    }
    qn.ToDevice(DataDevice::CUDA, std::vector<int>{0}); kn.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    std::vector<Data> keys(count), values(count), rk(count), rv(count), pos(count), live(count);
    std::vector<FastllmNaiveDraftRow> descriptors;
    for (int b = 0; b < count; ++b) {
        int past = b % 3 == 0 ? 3 : b % 3 == 1 ? window - 1 : window * 13;
        for (Data *x : {&keys[b], &rk[b]}) x->CopyFrom(BF({1, cap, kw}, 37 + b));
        for (Data *x : {&values[b], &rv[b]}) x->CopyFrom(BF({1, cap, kw}, 53 + b));
        for (Data *x : {&keys[b], &rk[b], &values[b], &rv[b]}) {
            x->expansionDims = {1, cap, kw}; x->Resize({1, std::min(past, window - 1), kw});
        }
        std::vector<float> positions(rows); for (int r = 0; r < rows; ++r) positions[r] = past + r;
        pos[b].CopyFrom(Data(FLOAT32, {1, rows}, positions)); pos[b].ToDevice(DataDevice::CUDA, std::vector<int>{0});
        live[b].dataType = INT32; live[b].UpdateUnitSize(); live[b].Resize({1}); live[b].Allocate();
        *(int *)live[b].cpuData = past + 1; live[b].ToDevice(DataDevice::CUDA, std::vector<int>{0});
        descriptors.push_back({&keys[b], &values[b], &pos[b], &live[b], std::min(past, window - 1) + rows <= 256});
    }
    Data query, scores, output;
    Require(FastllmCudaNaiveBatchDraftQKV(raw, qn, kn, descriptors, query, heads, kh, dim, window, 1e-5, 10000), "batch QKV rejected");
    Require(FastllmCudaNaiveBatchDraftAttention(query, descriptors, heads, kh, dim, window, scores, output), "batch attention rejected");
    auto qbits = Bits(query), out = Bits(output);
    for (int b = 0; b < count; ++b) {
        Data part, rq, rs, ro; part.Resize({1, rows, raw.dims[2]});
        part.FakeFrom(raw, (size_t)b * rows * raw.dims[2] * 2);
        Require(FastllmCudaNaiveDraftQKV(part, qn, kn, pos[b], live[b], rk[b], rv[b], rq,
            heads, kh, dim, window, 1e-5, 10000), "reference QKV rejected");
        FastllmCudaNaiveDraftAttention(rq, rk[b], rv[b], live[b], heads, kh, dim, window,
            descriptors[b].shortAttention, rs, ro);
        auto expectedQ = Bits(rq), expectedOut = Bits(ro);
        Require(std::equal(expectedQ.begin(), expectedQ.end(), qbits.begin() + (size_t)b * rows * qw), "batch query differs");
        Require(Bits(keys[b], (size_t)cap * kw) == Bits(rk[b], (size_t)cap * kw), "batch K/prefix differs");
        Require(Bits(values[b], (size_t)cap * kw) == Bits(rv[b], (size_t)cap * kw), "batch V/prefix differs");
        Require(std::equal(expectedOut.begin(), expectedOut.end(), out.begin() + (size_t)b * rows * qw), "batch attention bits differ");
    }
    auto before = Bits(keys[0], (size_t)cap * kw);
    descriptors.back().liveKeys = nullptr;
    Require(!FastllmCudaNaiveBatchDraftQKV(raw, qn, kn, descriptors, query, heads, kh, dim, window, 1e-5, 10000), "invalid batch accepted");
    Require(Bits(query) == qbits && Bits(keys[0], (size_t)cap * kw) == before, "preflight changed earlier request");
}

static void BatchInput(int count) {
    constexpr int rows = 7, channels = 256, vocab = 256;
    Data embedding = BF({vocab, channels}, 71), mask = BF({channels}, 51);
    std::vector<float> ids(count); for (int i = 0; i < count; ++i) ids[i] = (i * 19 + 7) % vocab;
    Data tokens(FLOAT32, {1, count}, ids), lengths(INT32, {count}), hidden, positions;
    lengths.Allocate();
    for (int i = 0; i < count; ++i) ((int *)lengths.cpuData)[i] = 1 + i * 4093;
    tokens.ToDevice(DataDevice::CUDA, std::vector<int>{0}); lengths.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    FastllmCudaNaiveDraftInput(tokens, embedding, mask, lengths, rows, hidden, positions);
    auto hb = Bits(hidden), pb = Bits(positions, (size_t)count * rows * 2);
    for (int i = 0; i < count; ++i) {
        Data id, live, h, p; id.Resize({1, 1}); id.FakeFrom(tokens, i * sizeof(float));
        live.Resize({1}); live.FakeFrom(lengths, i * sizeof(int));
        FastllmCudaNaiveDraftInput(id, embedding, mask, live, rows, h, p);
        auto hr = Bits(h), pr = Bits(p, rows * 2);
        Require(std::equal(hr.begin(), hr.end(), hb.begin() + (size_t)i * rows * channels), "batch draft input differs");
        Require(std::equal(pr.begin(), pr.end(), pb.begin() + i * rows * 2), "batch positions differ");
    }
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        ApplyDeviceMap({{"cuda:0", 1}}, 1, 1);
        SetCudaEmbedding(true);
        for (int count : {2, 8, 40}) BatchInput(count);
        for (int count : {2, 8, 40}) for (int dim : {128, 192, 256})
            for (int rows : {2, 7}) for (int window : {64, 1024, 1152})
                BatchDraft(count, rows, dim, 16, window);
        for (int count : {1, 2, 3, 4, 8, 16, 40}) for (int rows : {2, 5, 7, 8, 15})
            BatchDraft(count, rows, 128, 32, 1024);
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
