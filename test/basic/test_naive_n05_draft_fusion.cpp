#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include "utils/utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/naive-n05-draft-backend.h"
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
#include "devices/cuda/naive-n05-draft-fused.h"
#endif
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
static void Require(bool x, const char *msg) {
    if (!x)
        throw std::runtime_error(msg);
}
static Data BF(std::vector<int> dims, int seed, float scale = 1.f) {
    Data x(BFLOAT16, dims);
    x.Allocate();
    unsigned state = seed;
    for (int i = 0; i < x.Count(0); ++i) {
        state = 1664525 * state + 1013904223;
        ((uint16_t *)x.cpuData)[i] = Float32ToBFloat16RNEBits(scale * ((int)(state >> 16) - 32768) / 4096.f);
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

static float Float(uint16_t bits) {
    uint32_t word = uint32_t(bits) << 16; float value;
    std::memcpy(&value, &word, sizeof(value)); return value;
}
#ifndef USE_ROCM
static void OwnerDraftHeadReplay(int owners) {
    constexpr int V = 65568, S = 3; // Partial last vocabulary tile.
    auto allocate = [](DataType type, std::vector<int> dims) {
        Data value(type, dims);
        value.ToDevice(CUDA, std::vector<int>{0}, false); value.Allocate(false);
        return value;
    };
    Data base = BF({1,owners*S,V}, 31, .2f), w1 = BF({V,256}, 91, .04f), w2 = BF({V,256}, 113, .04f);
    Data ids = allocate(FLOAT32,{S+1,owners}), refIds = allocate(FLOAT32,{S+1,owners});
    Data q = allocate(FLOAT32,{owners,S,V}), ref = allocate(FLOAT32,{owners,S,V});
    Data mask = allocate(INT32,{owners,V/32}), bias = allocate(BFLOAT16,{1,owners,V}), logits;
    Require(cudaMemset(mask.cudaData, 0xff, mask.GetBytes()) == cudaSuccess, "proposal mask");
    std::vector<float> temperature(owners, .4f), topP(owners, 1.f), anchor(owners, 19.f);
    std::vector<int> topK(owners, 0);
    std::unique_ptr<void, decltype(&FastllmCudaMtpDestroyDraftSampler)> sampler(
        FastllmCudaMtpCreateDraftSamplerBatch(owners, V, temperature.data(), topK.data(), topP.data()), FastllmCudaMtpDestroyDraftSampler);
    Require(bool(sampler), "create owner sampler");
    DraftProjectedHead head{base,w1,w2,ids,q,sampler.get()};
    auto prepare = [&] {
        Require(FastllmCudaMtpPrepareDraftReplay(sampler.get(), (float *)ids.cudaData, owners, anchor.data(), temperature.data()),
                "prepare owner replay");
        Require(cudaMemcpyAsync(refIds.cudaData, ids.cudaData, size_t(owners)*sizeof(float), cudaMemcpyDeviceToDevice,
                cudaStreamPerThread) == cudaSuccess, "reference anchor");
    };
    prepare();
    Require(!TryCudaDraftHead(head,S-1), "mismatched head shape");
    if (!TryCudaDraftHead(head,S)) return; // Device cannot keep this vocabulary grid resident.
    void *graph = nullptr, *exec = nullptr;
    Require(FastllmCudaGraphBeginCapture() && TryCudaDraftHead(head,S) && FastllmCudaGraphEndCapture(&graph) &&
            FastllmCudaGraphInstantiate(graph,&exec), "owner head capture");
    for (float t : {.4f, 1.f, 1.5f}) {
        for (int owner = 0; owner < owners; ++owner) {
            temperature[owner] = t + .03f * owner;
            anchor[owner] += 31 + owner;
        }
        prepare();
        Require(FastllmCudaGraphLaunch(exec), "owner head replay");
        for (int j = 0; j < S; ++j) {
            Data token(FLOAT32, {1, owners}), latent;
            token.FakeFrom(refIds, size_t(j)*owners*sizeof(float));
            if (owners == 1)
                Require(FastllmCudaIndexedBf16Gemv(token,w1,w2,bias), "reference Markov projection");
            else {
                EmbeddingDirect(token, w1, latent);
                MatMulTransB(latent, w2, bias);
            }
            FastllmCudaNaiveBatchDraftSamplingLogits(base,bias,mask,logits,j,S);
            Require(FastllmCudaMtpDraftSamplerRunBatchStrided(sampler.get(),(float *)logits.cudaData,
                    (float *)ref.cudaData + size_t(j)*V,(float *)refIds.cudaData + (j + 1)*owners,4*j,S*V), "reference draw");
        }
        auto read = [](const Data &v) {
            std::vector<float> out(v.Count(0));
            Require(cudaMemcpy(out.data(), v.cudaData, v.GetBytes(), cudaMemcpyDeviceToHost) == cudaSuccess, "read head");
            return out;
        };
        Require(read(ids) == read(refIds), "fused draw differs from seeded reference");
        const auto got = read(q), want = read(ref);
        for (int j = 0; j < owners*S; ++j) {
            double mass = 0., err = 0., norm = 0.;
            for (int i = j*V; i < (j+1)*V; ++i) {
                double delta = double(got[i])-want[i]; mass += got[i]; err += delta*delta; norm += double(want[i])*want[i];
            }
            Require(std::abs(mass-1.) < 2e-5 && std::sqrt(err/norm) < 1e-3, "owner head distribution");
        }
    }
    std::printf("OWNER_HEAD_PASS owners=%d replay_temperatures=3\n", owners);
    FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
}
#endif

static void SamplingRows(int count, int steps) {
    const int vocab = 263, words = (vocab + 31) / 32;
    Data base = BF({1, count * steps, vocab}, 331), bias = BF({1, count, vocab}, 73);
    Data mask(INT32, {count, words}), output;
    mask.Allocate();
    for (int i = 0; i < count * words; ++i) ((uint32_t *)mask.cpuData)[i] = ~(uint32_t(1) << (i % 32));
    mask.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    const auto b = Bits(bias);
    auto input = Bits(base);
    for (int step = 0; step < steps; ++step) {
        auto body = [&] { FastllmCudaNaiveBatchDraftSamplingLogits(base, bias, mask, output, step, steps); };
        body();
        void *graph = nullptr, *exec = nullptr;
        Require(FastllmCudaGraphBeginCapture(), "sampling rows capture"); body();
        Require(FastllmCudaGraphEndCapture(&graph) && FastllmCudaGraphInstantiate(graph, &exec), "sampling rows graph");
        for (int replay = 0; replay < 2; ++replay) {
            for (int row = 0; row < count; ++row)
                input[((size_t)row * steps + step) * vocab] = Float32ToBFloat16RNEBits(float(replay + row));
            Require(cudaMemcpy(base.cudaData, input.data(), input.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess, "sampling rows upload");
            Require(FastllmCudaGraphLaunch(exec), "sampling rows replay");
            std::vector<float> got(count * vocab);
            Require(cudaMemcpy(got.data(), output.cudaData, got.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "sampling rows read");
            for (int row = 0; row < count; ++row) for (int col = 0; col < vocab; ++col) {
                const bool allowed = ((uint32_t *)mask.cpuData)[row * words + col / 32] & (uint32_t(1) << (col % 32));
                const float expected = allowed ? Float(Float32ToBFloat16RNEBits(
                    Float(input[((size_t)row * steps + step) * vocab + col]) + Float(b[row * vocab + col]))) : -INFINITY;
                if (got[row * vocab + col] != expected) {
                    std::cerr << "sampling count=" << count << " steps=" << steps << " step=" << step
                              << " replay=" << replay << " row=" << row << " col=" << col
                              << " got=" << got[row * vocab + col] << " expected=" << expected << '\n';
                    throw std::runtime_error("strided sampling logits differ from CPU oracle");
                }
            }
        }
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
    }
}
static void BatchDraft(int count, int rows, int dim, int heads, int window, int kvHeads = 4) {
    const int kh = kvHeads, qw = heads * dim, kw = kh * dim, cap = window + rows;
    const bool fused = count >= 8 && dim == 128 && heads / kvHeads <= 32 && window >= rows;
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
    Data query, scores, output, qkvDescriptors;
    Require(FastllmCudaNaiveBatchDraftQKV(raw, qn, kn, descriptors, query, heads, kh, dim, window, 1e-5, 10000,
        &qkvDescriptors), "batch QKV rejected");
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

        if (!fused) {
            Require(std::equal(expectedOut.begin(), expectedOut.end(), out.begin() + (size_t)b * rows * qw), "batch attention bits differ");
        } else {
            // Tensor-core online softmax changes the probability rounding point.
            // Bound its error against the existing full FP32 reduction path.
            double error = 0, norm = 0;
            for (size_t j = 0; j < expectedOut.size(); ++j) {
                double ref = Float(expectedOut[j]);
                double delta = Float(out[(size_t)b * rows * qw + j]) - ref;
                error += delta * delta; norm += ref * ref;
            }
            Require(std::sqrt(error / std::max(norm, 1e-20)) < .01, "fused attention relative error");
            // Independently check the window, request/head addressing and
            // numerical error with a double-precision CPU softmax and PV.
            if (b == 0 || b == count - 1) {
                const auto kb = Bits(rk[b], (size_t)cap * kw), vb = Bits(rv[b], (size_t)cap * kw);
                const int past = std::min(*(int *)live[b].cpuData - 1, window - 1), keys = past + rows;
                for (int row : {0, rows - 1}) for (int h : {0, heads - 1}) {
                    const int start = std::max(0, past + row - window + 1), head = h / (heads / kh);
                    std::vector<double> weights(keys);
                    double maximum = -1e30, sum = 0;
                    for (int k = start; k < keys; ++k) {
                        double dot = 0;
                        for (int d = 0; d < dim; ++d)
                            dot += double(Float(expectedQ[(row * heads + h) * dim + d])) *
                                   Float(kb[k * kw + head * dim + d]);
                        weights[k] = Float(Float32ToBFloat16RNEBits(
                            Float(Float32ToBFloat16RNEBits(float(dot))) / std::sqrt(float(dim))));
                        maximum = std::max(maximum, weights[k]);
                    }
                    for (int k = start; k < keys; ++k) { weights[k] = std::exp(weights[k] - maximum); sum += weights[k]; }
                    for (int d : {0, dim / 2, dim - 1}) {
                        double expected = 0;
                        for (int k = start; k < keys; ++k)
                            expected += weights[k] / sum * Float(vb[k * kw + head * dim + d]);
                        const double actual = Float(out[((size_t)b * rows * heads + row * heads + h) * dim + d]);
                        Require(std::abs(actual - expected) <= .04 + .01 * std::abs(expected), "fused attention CPU reference differs");
                    }
                }
            }
        }
    }

    if (fused) {
        void *graph = nullptr, *exec = nullptr;
        Require(FastllmCudaGraphBeginCapture(), "draft attention capture");
        Require(FastllmCudaNaiveBatchDraftAttention(query, descriptors, heads, kh, dim, window,
            scores, output), "captured attention rejected");
        Require(FastllmCudaGraphEndCapture(&graph), "draft attention end capture");
        Require(FastllmCudaGraphInstantiate(graph, &exec), "draft attention instantiate");
        // Change a short request to a long one without recapturing.
        int longer = window * 9;
        Require(cudaMemcpy(live[0].cudaData, &longer, sizeof(longer), cudaMemcpyHostToDevice) == cudaSuccess, "update live length");
        Require(FastllmCudaGraphLaunch(exec), "draft attention replay");
        auto replay = Bits(output);
        Require(FastllmCudaNaiveBatchDraftAttention(query, descriptors, heads, kh, dim, window,
            scores, output), "eager attention rejected");
        Require(Bits(output) == replay, "dynamic draft attention graph differs");
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);

        // Rebind every request's cache while replaying the SAME QKV+attention
        // graph. The captured descriptors still contain the original pointers;
        // only the device binding table may select the new request slots.
        Data bindings(INT32, {count, 4}); bindings.ToDevice(DataDevice::CUDA, {0}, false); bindings.Allocate(false);
        std::vector<uint64_t> addresses(count * 2);
        auto bind = [&](int rotation) {
            for (int b = 0; b < count; ++b) {
                addresses[b * 2] = (uint64_t)keys[(b + rotation) % count].cudaData;
                addresses[b * 2 + 1] = (uint64_t)values[(b + rotation) % count].cudaData;
            }
            Require(cudaMemcpy(bindings.cudaData, addresses.data(), addresses.size() * sizeof(uint64_t),
                cudaMemcpyHostToDevice) == cudaSuccess, "update draft KV table");
        };
        auto dynamic = [&] {
            Require(FastllmCudaNaiveBatchDraftQKV(raw, qn, kn, descriptors, query, heads, kh,
                dim, window, 1e-5, 10000, &qkvDescriptors, &bindings), "dynamic QKV rejected");
            Require(FastllmCudaNaiveBatchDraftAttention(query, descriptors, heads, kh, dim, window,
                scores, output, &bindings), "dynamic attention rejected");
        };
        bind(0); dynamic();
        Require(FastllmCudaGraphBeginCapture(), "dynamic KV graph capture"); dynamic();
        Require(FastllmCudaGraphEndCapture(&graph), "dynamic KV graph end capture");
        Require(FastllmCudaGraphInstantiate(graph, &exec), "dynamic KV graph instantiate");
        for (int rotation : {1, 3, 0}) {
            bind(rotation);
            Require(FastllmCudaGraphLaunch(exec), "dynamic KV graph replay");
            auto actual = Bits(output);
            std::vector<std::vector<uint16_t>> actualK, actualV;
            for (int b = 0; b < count; ++b) {
                actualK.push_back(Bits(keys[b], (size_t)cap * kw));
                actualV.push_back(Bits(values[b], (size_t)cap * kw));
            }
            auto reference = descriptors;
            for (int b = 0; b < count; ++b) {
                reference[b].key = &keys[(b + rotation) % count];
                reference[b].value = &values[(b + rotation) % count];
            }
            Require(FastllmCudaNaiveBatchDraftQKV(raw, qn, kn, reference, query, heads, kh,
                dim, window, 1e-5, 10000, &qkvDescriptors), "rebound reference QKV rejected");
            Require(FastllmCudaNaiveBatchDraftAttention(query, reference, heads, kh, dim, window,
                scores, output), "rebound reference attention rejected");
            Require(Bits(output) == actual, "rebound draft graph output differs");
            for (int b = 0; b < count; ++b) {
                Require(Bits(keys[b], (size_t)cap * kw) == actualK[b], "rebound graph K differs");
                Require(Bits(values[b], (size_t)cap * kw) == actualV[b], "rebound graph V differs");
            }
        }
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
        std::cout << "DRAFT FLASH PASS batch=" << count << " rows=" << rows << '\n';
    }
    if (count == 1 && rows == 7 && FastllmCudaNaiveDraftCacheBindingsSupported(
            count, rows, heads, kh, dim, window)) {
        Data table(INT32, {1, 4});
        table.ToDevice(DataDevice::CUDA, {0}, false); table.Allocate(false);
        auto bind = [&](Data &key, Data &value) {
            uint64_t addresses[] = {(uint64_t)key.cudaData, (uint64_t)value.cudaData};
            Require(cudaMemcpy(table.cudaData, addresses, sizeof(addresses),
                               cudaMemcpyHostToDevice) == cudaSuccess, "singleton KV binding");
        };
        auto run = [&](const std::vector<FastllmNaiveDraftRow> &inputs) {
            Require(FastllmCudaNaiveBatchDraftQKV(raw, qn, kn, inputs, query, heads, kh,
                dim, window, 1e-5, 10000, &qkvDescriptors, &table), "singleton bound QKV");
            Require(FastllmCudaNaiveBatchDraftAttention(query, inputs, heads, kh,
                dim, window, scores, output, &table), "singleton bound attention");
        };
        Require(FastllmCudaNaiveBatchDraftAttention(query, descriptors, heads, kh, dim, window,
            scores, output), "singleton reference attention");
        auto reference = Bits(output);
        bind(keys[0], values[0]); run(descriptors);
        auto bound = Bits(output);
        double error = 0, norm = 0;
        for (size_t i = 0; i < bound.size(); ++i) {
            const double expected = Float(reference[i]);
            const double delta = Float(bound[i]) - expected;
            error += delta * delta;
            norm += expected * expected;
        }
        Require(std::sqrt(error / std::max(norm, 1e-20)) < .01, "singleton bound attention relative error");
        void *graph = nullptr, *exec = nullptr;
        Require(FastllmCudaGraphBeginCapture(), "singleton capture");
        run(descriptors);
        Require(FastllmCudaGraphEndCapture(&graph) && FastllmCudaGraphInstantiate(graph, &exec),
                "singleton instantiate");
        const size_t elements = size_t(cap) * kw;
        Require(cudaMemset(rk[0].cudaData, 0, elements * sizeof(uint16_t)) == cudaSuccess &&
                cudaMemset(rv[0].cudaData, 0, elements * sizeof(uint16_t)) == cudaSuccess, "clear new KV");
        bind(rk[0], rv[0]);
        Require(FastllmCudaGraphLaunch(exec), "replay on new singleton KV");
        auto k = Bits(rk[0], elements), v = Bits(rv[0], elements), attention = Bits(output);
        Require(std::any_of(v.begin(), v.end(), [](uint16_t bits) { return bits != 0; }), "new KV not written");
        auto rebound = descriptors;
        rebound.front().key = &rk[0]; rebound.front().value = &rv[0];
        run(rebound);
        Require(k == Bits(rk[0], elements) && v == Bits(rv[0], elements) &&
                attention == Bits(output), "singleton rebound differs from eager");
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
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
// Exercise the same entry used in both layer loops. A declined specialization
// must leave the entire graph and its caller-owned buffers available for fallback.
static void DraftMlpFallback() {
    const std::vector<int> devices{0, 1};
    Data x = BF({1, 7, 32}, 3), hidden = BF(x.dims, 11), scratch = BF(x.dims, 17);
    Data gu = BF({64, 32}, 23), down = BF({32, 32}, 29), gamma(FLOAT32, {32}), middle, out;
    const auto inputBefore = Bits(x), hiddenBefore = Bits(hidden);
    cudaGraph_t graph;
    Require(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
            "capture declined MLP");
    const auto tryMlp = [&](int tp, int requests, Data *packed) {
        return TryCudaDraftMlp({devices, 0, tp}, requests, x,
            {packed, down}, hidden, gamma, 1e-5f, middle, scratch, out);
    };
    Require(!tryMlp(2, 1, &gu) && !tryMlp(1, 2, &gu) &&
        !tryMlp(1, 1, nullptr) && !tryMlp(1, 1, &gu), "unexpected draft MLP dispatch");
    std::vector<FastllmNaiveDraftRow> noRequests;
    Require(!TryCudaDraftAttention({devices, 0, 1}, {32,4,128,1024,1e-5f,10000},
        noRequests, nullptr, x, {nullptr,down,gamma,gamma,gamma},
        hidden, middle, out, scratch, out), "unsupported attention must decline");
    Require(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "end declined MLP capture");
    size_t nodes = 1;
    Require(cudaGraphGetNodes(graph, nullptr, &nodes) == cudaSuccess && nodes == 0, "declined MLP launched work");
    cudaGraphDestroy(graph);
    Require(out.dims.empty() && !out.cudaData && middle.dims.empty() && !middle.cudaData &&
        Bits(x) == inputBefore && Bits(hidden) == hiddenBefore, "declined MLP changed inputs or workspace");
}
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
static void CheckAttentionOwners(int owners, Data &qkv, Data &wo, Data &qn, Data &kn, Data &gamma) {
    constexpr int rows=7, window=128, width=4096, cap=window+rows, kw=512;
    const std::vector<int> devices{0}, shape{1,rows*owners,width};
    auto allocate=[&](DataType type,const std::vector<int> &dims) {
        Data x(type,dims); x.ToDevice(CUDA,devices,false); x.Allocate(false); return x;
    };
    auto view=[](Data &dst,const Data &src,size_t offset,const std::vector<int> &dims) {
        dst.FakeFrom(src,offset); dst.Resize(dims);
    };
    auto copy=[](void *dst,const void *src,size_t bytes,cudaMemcpyKind kind) {
        Require(cudaMemcpy(dst,src,bytes,kind)==cudaSuccess,"draft owner copy");
    };
    Data x=BF(shape,23), h=BF(shape,27), hidden=BF(shape,3), normalized=BF(shape,5);
    Data positions=allocate(FLOAT32,{1,rows*owners}), live=allocate(INT32,{owners});
    Data bindings=allocate(INT32,{owners,4}), raw,att,branch,scratch;
    std::vector<Data> pos(owners), length(owners), key(owners), value(owners), rk(owners), rv(owners);
    std::vector<FastllmNaiveDraftRow> requests;
    const DraftLayerParallel local{devices,0,1};
    const DraftAttentionGeometry geometry{32,4,128,window,1e-5f,10000.f};
    const DraftAttentionWeights weights{&qkv,wo,qn,kn,gamma};
    for (int i=0;i<owners;++i) {
        view(pos[i],positions,i*rows*sizeof(float),{1,rows});
        view(length[i],live,i*sizeof(int),{1});
        key[i].CopyFrom(BF({1,cap,kw},83+i)); value[i].CopyFrom(BF({1,cap,kw},91+i));
        rk[i].CopyFrom(key[i]); rv[i].CopyFrom(value[i]);
        for (auto *v:{&key[i],&value[i],&rk[i],&rv[i]}) { v->expansionDims={1,cap,kw}; v->Resize({1,0,kw}); }
        requests.push_back({&key[i],&value[i],&pos[i],&length[i]});
    }
    auto run=[&]{return TryCudaDraftAttention(local,geometry,requests,&bindings,normalized,weights,hidden,raw,att,branch,scratch);};
    auto reset=[&]{
        copy(normalized.cudaData,x.cudaData,x.GetBytes(),cudaMemcpyDeviceToDevice);
        copy(hidden.cudaData,h.cudaData,h.GetBytes(),cudaMemcpyDeviceToDevice);
    };
    Require(cudaMemset(live.cudaData,0,live.GetBytes())==cudaSuccess &&
            cudaMemset(bindings.cudaData,0,bindings.GetBytes())==cudaSuccess,"idle owners");
    reset(); Require(run(),"multi-owner admission");
    cudaGraph_t graph; cudaGraphExec_t exec; size_t nodes=0;
    Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess &&
            run() && cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess &&
            cudaGraphGetNodes(graph,nullptr,&nodes)==cudaSuccess && nodes==1 &&
            cudaGraphInstantiateWithFlags(&exec,graph,0)==cudaSuccess,"one multi-owner attention node");
    for (int replay=0;replay<3;++replay) {
        std::vector<const void*> table;
        std::vector<int> lengths(owners);
        std::vector<float> p(owners*rows);
        for (int i=0;i<owners;++i) {
            int bank=(i+replay)%owners;
            lengths[i]=i==replay%owners ? 0 : 19+(i+replay)*35;
            table.push_back(key[bank].cudaData); table.push_back(value[bank].cudaData);
            for (int j=0;j<rows;++j) p[i*rows+j]=float(lengths[i]+j);
        }
        copy(bindings.cudaData,table.data(),bindings.GetBytes(),cudaMemcpyHostToDevice);
        copy(live.cudaData,lengths.data(),live.GetBytes(),cudaMemcpyHostToDevice);
        copy(positions.cudaData,p.data(),positions.GetBytes(),cudaMemcpyHostToDevice);
        reset(); Require(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess,"multi-owner replay");
        for (int i=0;i<owners;++i) {
            const int bank=(i+replay)%owners;
            Data in,src,actual,actualH,b=allocate(INT32,{1,4}),one,h1,r,a,tmp,ws;
            const size_t off=size_t(i)*rows*width*2;
            view(in,x,off,{1,rows,width}); view(src,h,off,{1,rows,width});
            view(actual,normalized,off,{1,rows,width}); view(actualH,hidden,off,{1,rows,width});
            one.CopyFrom(in); h1.CopyFrom(src);
            const void *pair[]{rk[bank].cudaData,rv[bank].cudaData};
            copy(b.cudaData,pair,sizeof(pair),cudaMemcpyHostToDevice);
            Require(TryCudaDraftAttention(local,geometry,{{&rk[bank],&rv[bank],&pos[i],&length[i]}},&b,
                one,weights,h1,r,a,tmp,ws),"single-owner reference");
            Require(Bits(actual)==Bits(one) && Bits(actualH)==Bits(h1) &&
                Bits(key[bank],cap*kw)==Bits(rk[bank],cap*kw) && Bits(value[bank],cap*kw)==Bits(rv[bank],cap*kw),
                "owner output, idle row or rebound KV differs");
        }
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
}

static void ReplicaAttentionReplay() {
    const std::vector<int> devices{0};
    int scratchSize;
    if (!FastllmDraftLocalAttentionPlan(0, 7, 32, 4, 128, 1024, &scratchSize)) return;
    Data qkvWeight = BF({5120,4096}, 113), wo = BF({4096,4096}, 191);
    auto scale = [](int n) {
        Data a(FLOAT32,{n}); a.Allocate();
        for (int i=0;i<n;++i) ((float*)a.cpuData)[i]=0.7f+(i%93)/128.f;
        a.ToDevice(CUDA,std::vector<int>{0}); return a;
    };
    Data qn=scale(128), kn=scale(128), gamma=scale(4096);
    if (FastllmCudaRuntimeArch()==103) {
        Require(FastllmDraftBatchAttentionPlan(0, 4, 6, 32, 4, 128, 1024, &scratchSize) &&
                !FastllmDraftBatchAttentionPlan(0, 5, 5, 32, 4, 128, 1024, &scratchSize), "attention row crossover");
        for (int owners : {2,3}) CheckAttentionOwners(owners,qkvWeight,wo,qn,kn,gamma);
    }
    for (int rows = FastllmCudaRuntimeArch() == 120 ? 7 : 1; rows <= 8; ++rows) {
        const std::vector<int> dims{1,rows,4096};
        const int cap=1024+rows;
        Data x=BF(dims,23), src=BF(dims,29), normed=BF(dims,1), hidden=BF(dims,2);
        Data pos(FLOAT32,{1,rows}), live(INT32,{1}), bindings(INT32,{1,4}), refBindings(INT32,{1,4});
        for(Data *v:{&pos,&live,&bindings,&refBindings}) {v->Allocate(); v->ToDevice(CUDA,devices);}
        Data key[2], val[2], refK[2], refV[2];
        for(int i=0;i<2;++i) {
            for (auto *k:{&key[i],&refK[i]}) k->CopyFrom(BF({1,cap,512},31+i));
            for (auto *v:{&val[i],&refV[i]}) v->CopyFrom(BF({1,cap,512},47+i));
            for (auto *buf:{&key[i],&refK[i],&val[i],&refV[i]}) {
                buf->expansionDims={1,cap,512}; buf->Resize({1,0,512});
            }
        }
        auto bind=[&](Data &b,Data &k,Data &v) {
            const void *p[]={k.cudaData,v.cudaData};
            Require(cudaMemcpy(b.cudaData,p,sizeof(p),cudaMemcpyHostToDevice)==cudaSuccess,"bind attention KV");
        };
        auto reset=[&](int past,int owner) {
            const int length=past+1; std::vector<float> positions(rows);
            for (int j=0;j<rows;++j) positions[j]=float(past+j);
            Require(cudaMemcpy(live.cudaData,&length,sizeof(int),cudaMemcpyHostToDevice)==cudaSuccess &&
                cudaMemcpy(pos.cudaData,positions.data(),rows*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess,
                "update attention length and positions");
            bind(bindings,key[owner],val[owner]); bind(refBindings,refK[owner],refV[owner]);
            for (auto pair:{std::make_pair(&normed,&x),std::make_pair(&hidden,&src)})
                Require(cudaMemcpy(pair.first->cudaData,pair.second->cudaData,rows*4096*sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice)==cudaSuccess,"restore attention inputs");
        };
        std::vector<FastllmNaiveDraftRow> req{{&key[0],&val[0],&pos,&live}};
        Data raw, att, branch, scratch;
        auto run=[&](int tp) {return TryCudaDraftAttention({devices,0,tp},{32,4,128,1024,1e-5f,10000},
            req,&bindings,normed,{&qkvWeight,wo,qn,kn,gamma},hidden,raw,att,branch,scratch);};
        reset(0,0);
        Require(!run(2) && raw.dims.empty() && scratch.dims.empty(),"declined attention changed workspace");
        Require(run(1),"warmup replica attention");
        cudaGraph_t graph; cudaGraphExec_t exec;
        Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess &&
            run(1) && cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess,"capture replica attention");
        size_t nodes=0;
        Require(cudaGraphGetNodes(graph,nullptr,&nodes)==cudaSuccess && nodes==1 &&
            cudaGraphInstantiateWithFlags(&exec,graph,0)==cudaSuccess,"single attention graph node");
        for(int past:{0,16384,14}) {
            const int owner=past==16384, base=std::min(1023,past);
            reset(past,owner);
            Require(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess,"replay replica attention");
            Data rq,query,ws,scores,ra,ro,rn,rh;
            rh.CopyFrom(src); MatMulTransB(x,qkvWeight,rq);
            std::vector<FastllmNaiveDraftRow> rr{{&refK[owner],&refV[owner],&pos,&live}};
            if (rows == 1) {
                Require(FastllmCudaNaiveDraftQKV(rq,qn,kn,pos,live,refK[owner],refV[owner],
                    query,32,4,128,1024,1e-5f,10000),"single-row draft QKV reference");
                // The batched reference adds every draft position to the softmax.
                // Compute the singleton directly so a second key cannot change its denominator.
                const auto q = Bits(query), k = Bits(refK[owner], cap * 512), v = Bits(refV[owner], cap * 512);
                std::vector<float> values(4096), logits(base + 1);
                auto round = [](float n) { return Float(Float32ToBFloat16RNEBits(n)); };
                for (int head = 0; head < 32; ++head) {
                    for (int t = 0; t <= base; ++t) {
                        float dot = 0;
                        for (int c = 0; c < 128; ++c) dot += Float(q[head*128+c])*Float(k[t*512+head/8*128+c]);
                        logits[t] = round(round(dot) / std::sqrt(128.f));
                    }
                    const float maximum = *std::max_element(logits.begin(), logits.end());
                    float sum = 0;
                    for (int t = 0; t <= base; ++t) {
                        const float weight = std::exp(logits[t] - maximum);
                        sum += weight;
                        for (int c = 0; c < 128; ++c) values[head*128+c] += weight*Float(v[t*512+head/8*128+c]);
                    }
                    for (int c = 0; c < 128; ++c) values[head*128+c] /= sum;
                }
                Data host(FLOAT32,{1,1,4096},values);
                ToDataType(host,BFLOAT16); host.ToDevice(CUDA,devices); ra.CopyFrom(host);
            } else {
                Require(FastllmCudaNaiveBatchDraftQKV(rq,qn,kn,rr,query,32,4,128,1024,1e-5f,10000,&ws,&refBindings) &&
                    FastllmCudaNaiveBatchDraftAttention(query,rr,32,4,128,1024,scores,ra,&refBindings),"attention reference");
            }
            MatMulTransB(ra,wo,ro); FastllmCudaNaiveAddDecodeRMSNorm(rh,ro,gamma,1e-5f,rn);
            auto compare=[&](const Data &a,const Data &b,int width,int offset=0) {
                const auto av=Bits(a,(offset+rows)*size_t(width)),bv=Bits(b,(offset+rows)*size_t(width));
                for(int row=offset;row<offset+rows;++row) {
                    double err=0.,sum=0.;
                    for(int c=0;c<width;++c) {
                        const int i=row*width+c; double r=Float(bv[i]),diff=Float(av[i])-r;
                        err+=diff*diff;sum+=r*r;
                    }
                    if (!(std::sqrt(err/std::max(sum,1e-20)) < 0.012)) {
                        std::cerr << "rows=" << rows << " past=" << past << " width=" << width
                                  << " relative L2=" << std::sqrt(err/std::max(sum,1e-20)) << std::endl;
                        throw std::runtime_error("draft attention row relative L2");
                    }
                }
            };
            compare(key[owner],refK[owner],512,base); compare(val[owner],refV[owner],512,base);
            compare(att,ra,4096); compare(hidden,rh,4096); compare(normed,rn,4096);
        }
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
}
static void ReplicaMlpReplay() {
    if (!FastllmDraftFusedLocalMlpBlocks(0)) return;
    const std::vector<int> devices{0};
    const DraftLayerParallel local{devices, 0, 1};
    Data gu = BF({8192, 4096}, 211), down = BF({4096, 4096}, 271);
    Data gamma(FLOAT32, {4096});
    gamma.Allocate();
    for (int i = 0; i < 4096; ++i) ((float *)gamma.cpuData)[i] = 0.5f + float(i % 127) / 128;
    gamma.ToDevice(CUDA, std::vector<int>{0});
    std::vector<int> cases{7, 8};
    if (FastllmCudaRuntimeArch() != 120) cases = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 31, 33, 65, 127, 128};
    for (int rows : cases) {
        const std::vector<int> dims{1, rows, 4096};
        Data input = BF(dims, 1), middle = BF(dims, 2), branch = BF(dims, 3);
        Data hidden = BF(dims, 4), tail = BF(dims, 5);
        Data &output = rows == 7 ? input : tail;
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        Require(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
                "capture replica MLP");
        const DraftMlpWeights weights{&gu, down};
        Require(!TryCudaDraftMlp({devices, 0, 2}, 1, input, weights, hidden, gamma,
                    1e-5f, middle, branch, output) &&
                !TryCudaDraftMlp(local, rows + 1, input, weights, hidden, gamma, 1e-5f, middle, branch, output),
                "replica MLP cannot take a partial TP result or more requests than rows");
        if (rows > 16) {
            Require(!TryCudaDraftMlp(local, (rows + 7) / 8, input, weights, hidden,
                gamma, 1e-5f, middle, branch, output), "wide draft MLP must use the batched GEMM");
            size_t nodes = 1;
            Require(cudaStreamEndCapture(cudaStreamPerThread,&graph) == cudaSuccess &&
                cudaGraphGetNodes(graph,nullptr,&nodes) == cudaSuccess && nodes == 0,"wide MLP launched work");
            cudaGraphDestroy(graph);
            continue;
        }
        Require(TryCudaDraftMlp(local, (rows + 7) / 8, input, weights, hidden,
            gamma, 1e-5f, middle, branch, output), "replica MLP dispatch");
        Require(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "end replica MLP capture");
        size_t nodes = 0;
        Require(cudaGraphGetNodes(graph, nullptr, &nodes) == cudaSuccess && nodes == 1, "replica MLP node count");
        Require(cudaGraphInstantiateWithFlags(&exec, graph, 0) == cudaSuccess, "instantiate replica MLP");

        for (int replay = 0; replay < 2; ++replay) {
            Data x = BF(dims, 17 + replay), residual = BF(dims, 37 + replay), ref, gate, projected, norm;
            const size_t bytes = size_t(rows) * 4096 * 2;
            if (!replay)
                Require(cudaMemsetAsync(residual.cudaData, 0, bytes, cudaStreamPerThread) == cudaSuccess, "zero residual");
            Require(cudaMemcpyAsync(input.cudaData, x.cudaData, bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread) == cudaSuccess &&
                    cudaMemcpyAsync(hidden.cudaData, residual.cudaData, bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread) == cudaSuccess,
                    "refresh replica MLP inputs");
            Require(cudaGraphLaunch(exec, cudaStreamPerThread) == cudaSuccess, "replay replica MLP");
            Linear(x, gu, Data(), ref);
            FastllmCudaNaiveDraftSwiGLU(ref, gate);
            Linear(gate, down, Data(), projected);
            FastllmCudaNaiveAddDecodeRMSNorm(residual, projected, gamma, 1e-5f, norm);
            auto compare = [&](const Data &actual, const Data &expected) {
                const auto a = Bits(actual), b = Bits(expected);
                for (int row = 0; row < rows; ++row) {
                    double error = 0., scale = 0.;
                    for (int i = 0; i < 4096; ++i) {
                        const int j = row * 4096 + i;
                        const double v = Float(b[j]), diff = Float(a[j]) - v;
                        error += diff * diff; scale += v * v;
                    }
                    Require(std::sqrt(error / scale) < 0.005, "replica MLP row relative L2");
                }
            };
            compare(middle, gate);
            if (!replay) compare(hidden, projected);
            compare(hidden, residual); compare(output, norm);
        }
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
}
#endif

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        ApplyDeviceMap({{"cuda:0", 1}}, 1, 1);
        SetCudaEmbedding(true);
        DraftMlpFallback();
#ifndef USE_ROCM
        for (int owners = 1; owners <= 8; ++owners) OwnerDraftHeadReplay(owners);
#endif
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
        ReplicaMlpReplay();
        ReplicaAttentionReplay();
#endif
        for (int count : {1, 33, 129}) for (int steps : {1, 3, 5, 7}) SamplingRows(count, steps);
        for (int count : {2, 8, 40}) BatchInput(count);
        for (int count : {2, 8, 40}) for (int dim : {128, 192, 256})
            for (int rows : {2, 7}) for (int window : {64, 1024, 1152})
                BatchDraft(count, rows, dim, 16, window);
        for (int count : {1, 2, 3, 4, 8, 16, 40}) for (int rows : {2, 5, 7, 8, 15})
            BatchDraft(count, rows, 128, 32, 1024);
        for (int count : {32, 128}) for (int rows : {3, 5, 7})
            BatchDraft(count, rows, 128, 32, 1024);
        for (int count : {8, 33, 128}) for (int rows : {3, 5, 7})
            BatchDraft(count, rows, 128, 4, 1024, 1);
        // Cross both the old 32-request boundary and the descriptor upload
        // chunk boundary; graph replay must own its descriptor values.
        for (int rows : {3, 5, 7}) BatchDraft(257, rows, 128, 4, 1024, 1);
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
