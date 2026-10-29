#include "fastllm.h"
#include "executor.h"
#include "utils/utils.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace fastllm;
static void Require(bool value, const char *what) { if (!value) throw std::runtime_error(what); }
static Data BF(const std::vector<int> &dims, int seed) {
    Data x(BFLOAT16, dims); x.Allocate();
    for (int i = 0; i < x.Count(0); ++i)
        ((uint16_t *)x.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + seed) * .137f));
    x.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    return x;
}
static std::vector<uint16_t> Bits(const Data &x) {
    Data host(x); host.ToDevice(DataDevice::CPU);
    return std::vector<uint16_t>((uint16_t *)host.cpuData, (uint16_t *)host.cpuData + host.Count(0));
}

static void BatchSampling(int count, int vocab) {
    const int words = (vocab + 31) / 32;
    Data base = BF({1, count, vocab}, 51), bias = BF({1, count, vocab}, 103);
    Data mask(INT32, {count, words}); mask.Allocate();
    std::fill_n((uint32_t *)mask.cpuData, (size_t)count * words, uint32_t(0));
    for (int row = 0; row < count; ++row) for (int id = 0; id < vocab; ++id)
        if (id % 3 == row % 3 || id == vocab - 1)
            ((uint32_t *)mask.cpuData)[row * words + id / 32] |= uint32_t(1) << (id % 32);
    mask.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    Data logits;
    FastllmCudaNaiveBatchDraftSamplingLogits(base, bias, mask, logits);
    Data host(logits); host.ToDevice(DataDevice::CPU);
    std::vector<float> temps(count), ps(count); std::vector<int> ks(count, -1), tokens(count);
    for (int row = 0; row < count; ++row) { temps[row] = .7f + .1f * (row % 5); ps[row] = row % 2 ? .95f : .8f; }
    Data probabilities(FLOAT32, {count, vocab}); probabilities.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); probabilities.Allocate(false);
    Require(FastllmCudaMtpSampleDraft((float *)logits.cudaData, (float *)probabilities.cudaData,
        temps.data(), ks.data(), ps.data(), tokens.data(), count, vocab), "batch proposal sampler failed");
    Data ph(probabilities); ph.ToDevice(DataDevice::CPU);
    for (int row = 0; row < count; ++row) {
        Data rb, rw, rm, reference;
        rb.Resize({1, 1, vocab}); rb.FakeFrom(base, (size_t)row * vocab * 2);
        rw.Resize({1, 1, vocab}); rw.FakeFrom(bias, (size_t)row * vocab * 2);
        rm.Resize({words}); rm.FakeFrom(mask, (size_t)row * words * sizeof(int));
        FastllmCudaNaiveDraftSamplingLogits(rb, rw, rm, 0, reference);
        Data rh(reference); rh.ToDevice(DataDevice::CPU);
        Require(std::memcmp(rh.cpuData, host.cpuData + (size_t)row * vocab * sizeof(float), vocab * sizeof(float)) == 0,
            "batched proposal mask or rounding differs");
        Data prob(FLOAT32, {1, vocab}); prob.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); prob.Allocate(false); int token;
        Require(FastllmCudaMtpSampleDraft((float *)reference.cudaData, (float *)prob.cudaData,
            &temps[row], &ks[row], &ps[row], &token, 1, vocab), "single sampler failed");
        prob.ToDevice(DataDevice::CPU);
        float *actual = (float *)ph.cpuData + (size_t)row * vocab;
        double sum = 0;
        for (int id = 0; id < vocab; ++id) {
            Require(std::isfinite(actual[id]) && std::abs(actual[id] - ((float *)prob.cpuData)[id]) <= 1e-7,
                "batch q distribution changed"); sum += actual[id];
        }
        Require(std::abs(sum - 1) < 1e-5 && actual[tokens[row]] > 0, "sample violates retained q distribution");
    }
}
static void LargeBatchTopP(int count, int vocab) {
    std::vector<float> input((size_t)count * vocab), temp(count, 1), ps(count);
    std::vector<int> ks(count, -1), tokens(count);
    for (int row = 0; row < count; ++row) {
        ps[row] = row % 4 == 0 ? 1.f : row % 4 == 1 ? .95f : row % 4 == 2 ? .8f : .5f;
        for (int id = 0; id < vocab; ++id) {
            // Sparse, broad and tied cutoffs exercise the shortcut and fallback
            // in the same batch, including rows above the tiled-softmax limit.
            input[(size_t)row * vocab + id] = row % 3 == 0
                ? (id < 73 ? 7.f - id * .13f : -20.f)
                : row % 3 == 1 ? std::sin(id * .071f)
                : (id < 128 ? 4.f : -20.f);
        }
    }
    Data logits(FLOAT32, {count, vocab}), probs(FLOAT32, {count, vocab});
    logits.Allocate(); std::memcpy(logits.cpuData, input.data(), input.size() * sizeof(float));
    logits.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    probs.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); probs.Allocate(false);
    Require(FastllmCudaMtpSampleDraft((float *)logits.cudaData, (float *)probs.cudaData,
        temp.data(), ks.data(), ps.data(), tokens.data(), count, vocab), "large top-p failed");
    Data host(probs); host.ToDevice(DataDevice::CPU);
    for (int row = 0; row < count; ++row) {
        const float *x = input.data() + (size_t)row * vocab;
        const float *actual = (const float *)host.cpuData + (size_t)row * vocab;
        float maximum = *std::max_element(x, x + vocab);
        std::vector<double> expected(vocab), sorted;
        double total = 0;
        for (int id = 0; id < vocab; ++id) total += expected[id] = std::exp(double(x[id] - maximum));
        for (double &value : expected) value /= total;
        double threshold = 0;
        if (ps[row] < 1) {
            sorted = expected; std::sort(sorted.begin(), sorted.end(), std::greater<double>());
            double mass = 0;
            for (double value : sorted) { mass += value; if (mass >= ps[row]) { threshold = value; break; } }
        }
        total = 0;
        for (double value : expected) if (value >= threshold) total += value;
        double actualMass = 0;
        for (int id = 0; id < vocab; ++id) {
            double want = expected[id] >= threshold ? expected[id] / total : 0;
            Require(std::isfinite(actual[id]) && std::abs(actual[id] - want) < 2e-6,
                "large top-p differs from independent CPU distribution");
            actualMass += actual[id];
        }
        Require(std::abs(actualMass - 1) < 2e-5 && actual[tokens[row]] > 0,
            "large top-p mass or sampled support differs");
    }
    void *sampler = FastllmCudaMtpCreateDraftSamplerBatch(count, vocab, temp.data(), ks.data(), ps.data());
    Require(sampler, "persistent batch sampler creation failed");
    Data ids(FLOAT32, {2, count}); ids.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); ids.Allocate(false);
    auto sample = [&]() {
        for (int step = 0; step < 2; ++step)
            Require(FastllmCudaMtpDraftSamplerRunBatch(sampler, (float *)logits.cudaData,
                (float *)probs.cudaData, (float *)ids.cudaData + step * count, step * 4),
                "persistent batch sample failed");
    };
    Require(FastllmCudaMtpDraftSamplerSetSeed(sampler), "persistent seed upload failed");
    sample();
    cudaGraph_t graph; cudaGraphExec_t exec;
    Require(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess, "sampler capture begin");
    sample();
    Require(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "sampler capture end");
    Require(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess, "sampler graph instantiate");
    std::vector<float> first, latest(count * 2);
    bool changed = false, independent = false;
    for (int repeat = 0; repeat < 20; ++repeat) {
        Require(FastllmCudaMtpDraftSamplerSetSeed(sampler), "replay seed upload failed");
        Require(cudaGraphLaunch(exec, cudaStreamPerThread) == cudaSuccess, "sampler graph replay");
        Require(cudaMemcpy(latest.data(), ids.cudaData, latest.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "sampler IDs download");
        for (int row = 0; row < count; ++row) for (int step = 0; step < 2; ++step) {
            int id = (int)latest[step * count + row];
            Require(id >= 0 && id < vocab && ((float *)host.cpuData)[(size_t)row * vocab + id] > 0,
                "graph sample outside retained distribution");
        }
        if (repeat == 0) first = latest;
        else changed |= first != latest;
        for (int row = 0; row < count; ++row) independent |= latest[row] != latest[count + row];
    }
    Require(changed && independent, "graph captured a constant seed or reused draw offsets");
    Data replay(probs); replay.ToDevice(DataDevice::CPU);
    for (size_t i = 0; i < input.size(); ++i)
        Require(std::abs(((float *)replay.cpuData)[i] - ((float *)host.cpuData)[i]) < 2e-6,
            "graph proposal distribution differs");
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); FastllmCudaMtpDestroyDraftSampler(sampler);
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        ApplyDeviceMap({{"cuda:0", 1}}, 1, 1);
        for (int count : {1, 3, 8, 40, 65}) LargeBatchTopP(count, 65536);
        LargeBatchTopP(8, 152576);
        for (int count : {2, 8, 40}) for (int vocab : {7, 257, 16385}) BatchSampling(count, vocab);
        int checks = 0;
        // The unchanged two-row GEMV is an exact oracle for the new one-row
        // layout, including bias conversion, tail output tiles and cancellation.
        for (int outputs : {1, 7, 8, 9, 255, 513, 16383, 16384, 16385, 19072, 152576}) {
            Data x = BF({1, 256}, 71);
            Data twice, bias = BF({outputs}, 31);
            ToDataType(bias, FLOAT32);
            Cat(x, x, 0, twice);
            for (bool withBias : {false, true}) {
                Data w = BF({outputs, 256}, outputs);
                Data actual, reference;
                Linear(x, w, withBias ? bias : Data(), actual);
                Linear(twice, w, withBias ? bias : Data(), reference);
                auto a = Bits(actual), b = Bits(reference);
                Require(std::equal(a.begin(), a.end(), b.begin()), "small-K GEMV differs from multirow");
                ++checks;
            }
        }
        for (int vocab : {1, 7, 255, 256, 257, 1023, 1025, 152576}) {
            for (int pattern = 0; pattern < 5; ++pattern) {
                Data base(BFLOAT16, {1, 2, vocab}), bias(BFLOAT16, {1, 1, vocab});
                base.Allocate(); bias.Allocate();
                for (int i = 0; i < vocab; ++i) {
                    float a = std::sin(i * .137f), b = std::cos(i * .213f);
                    if (pattern == 1) { a = -1; b = 0; if (i == 1 || i == 2 || i == 256) a = 2; }
                    if (pattern == 2) { a = std::numeric_limits<float>::quiet_NaN(); b = 0; }
                    if (pattern == 3) { a = -std::numeric_limits<float>::infinity(); b = 0; }
                    if (pattern == 4) { a = i % 7 ? 0 : std::numeric_limits<float>::infinity(); b = 0; }
                    ((uint16_t *)base.cpuData)[i] = Float32ToBFloat16RNEBits(a);
                    ((uint16_t *)base.cpuData)[vocab + i] = Float32ToBFloat16RNEBits(-a);
                    ((uint16_t *)bias.cpuData)[i] = Float32ToBFloat16RNEBits(b);
                }
                base.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                bias.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                Data ids(FLOAT32, {1, 3}, {0, 0, 0}), partial;
                ids.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                for (int row = 0; row < 2; ++row) {
                    Data logits, top;
                    Split(base, 1, row, row + 1, logits);
                    AddTo(logits, bias); ToDataType(logits, FLOAT32); TopK(logits, top, 1);
                    top.ToDevice(DataDevice::CPU);
                    FastllmCudaNaiveDraftArgmax(base, bias, row, partial, ids);
                    Data host(ids); host.ToDevice(DataDevice::CPU);
                    Require(((float *)host.cpuData)[row + 1] == ((float *)top.cpuData)[0], "fused argmax differs from Top1");
                    auto maskBits = std::make_shared<std::vector<uint32_t>>((vocab + 31) / 32, ~uint32_t(0));
                    for (int i = 0; i < vocab; ++i)
                        if ((i + row) % 3 == 0) (*maskBits)[i / 32] &= ~(uint32_t(1) << (i % 32));
                    Data mask(INT32, {(int)maskBits->size()});
                    mask.Allocate();
                    std::memcpy(mask.cpuData, maskBits->data(), maskBits->size() * sizeof(uint32_t));
                    mask.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                    GenerationConfig config; config.tool_call_token_mask = maskBits;
                    ApplyToolTokenMask(logits, 0, config);
                    Data fused;
                    FastllmCudaNaiveDraftSamplingLogits(base, bias, mask, row, fused);
                    fused.ToDevice(DataDevice::CPU); logits.ToDevice(DataDevice::CPU);
                    for (int i = 0; i < vocab; ++i) {
                        float a = ((float *)fused.cpuData)[i], b = ((float *)logits.cpuData)[i];
                        Require((std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, sizeof(float)) == 0,
                                "fused sampling logits changed BF16 rounding or mask");
                    }
                    checks += 2;
                }
            }
        }
        for (int rows : {1, 5, 8}) {
            std::vector<Data> sources;
            for (int i = 0; i < 8; ++i) sources.push_back(BF({1, 8, 257}, i * 91));
            std::vector<const Data *> pointers;
            Data reference;
            for (auto &x : sources) {
                pointers.push_back(&x);
                Data slice, joined; Split(x, 1, 0, rows, slice);
                if (reference.dims.empty()) Copy(slice, reference);
                else { Cat(reference, slice, -1, joined); Copy(joined, reference); }
            }
            Data output;
            Require(FastllmCudaNaiveDraftConcat(pointers, rows, output), "concat fast path rejected valid inputs");
            Require(Bits(output) == Bits(reference), "concat changed feature order");
            Require(!FastllmCudaNaiveDraftConcat(pointers, 9, output), "concat accepted an invalid row count");
            ++checks;
        }
        {
            Data padded = BF({1, 8, 258}, 31), output = BF({1, 1, 7}, 17);
            // A view keeps the 258-element row stride while exposing only 257
            // columns. The dense fast path must reject it without touching output.
            padded.dims[2] = 257;
            auto before = Bits(output);
            Require(!FastllmCudaNaiveDraftConcat({&padded}, 5, output), "concat accepted padded rows");
            Require(output.dims == std::vector<int>({1, 1, 7}) && Bits(output) == before,
                    "concat fallback changed output");
            ++checks;
        }
        std::cout << "DRAFT OPS PASS checks=" << checks << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
