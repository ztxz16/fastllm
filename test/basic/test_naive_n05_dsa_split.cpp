#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace fastllm;
namespace {
void Require(bool ok, const char *s) { if (!ok) throw std::runtime_error(s); }
void Check(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
void Init(Data &d, DataType type, const std::vector<int> &shape, const void *host = nullptr) {
    d.dataType = type; d.Resize(shape); d.ToDevice(DataDevice::CUDA, {0}, false); d.Allocate(false);
    if (host) Check(cudaMemcpy(d.cudaData, host, d.GetBytes(), cudaMemcpyHostToDevice));
}
std::vector<uint16_t> Random(int count, unsigned seed, float scale = 1.f) {
    std::vector<uint16_t> data(count);
    for (auto &x : data) {
        seed = seed * 1664525u + 1013904223u;
        x = Float32ToBFloat16RNEBits((int(seed >> 16) - 32768) / 32768.f * scale);
    }
    return data;
}
std::vector<uint16_t> Read(const Data &d) {
    std::vector<uint16_t> result(d.Count(0));
    Check(cudaMemcpy(result.data(), d.cudaData, result.size() * 2, cudaMemcpyDeviceToHost));
    return result;
}
void Case(int rows, int length, int heads, int kvHeads, bool paged, bool withSink, bool unaligned = false) {
    constexpr int pageLen = 128, dim = 192, vd = 128;
    const int pages = (length + pageLen - 1) / pageLen, pool = pages * pageLen;
    const int stride = kvHeads * dim + (paged ? 128 : 0), count = std::min(length, 2048);
    auto query = Random(rows * heads * dim, 29, 3.f);
    auto key = Random(pool * stride, 37, 2.f), value = Random(pool * kvHeads * vd, 71);
    Data q, queryStorage, k, v, live, index, sink, keyPages, valuePages, table, out;
    if (unaligned) {
        Init(queryStorage, BFLOAT16, {int(query.size()) + 1});
        q.Resize({1, rows, heads * dim}); q.FakeFrom(queryStorage, sizeof(uint16_t));
        Check(cudaMemcpy(q.cudaData, query.data(), query.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else Init(q, BFLOAT16, {1, rows, heads * dim}, query.data());
    Init(k, BFLOAT16, {pool, stride}, key.data()); Init(v, BFLOAT16, {pool, kvHeads * vd}, value.data());
    std::vector<int> lengths(rows, length), kp(pages), vp(pages), selected(rows * count);
    std::vector<float> sinks(heads);
    for (int p = 0; p < pages; ++p) { kp[p] = pages - 1 - p; vp[p] = (p + 1) % pages; }
    for (int r = 0; r < rows; ++r) for (int j = 0; j < count; ++j)
        selected[r * count + j] = j < 129 ? -1 : (j * 31 + r * 7) % length;
    for (int h = 0; h < heads; ++h) sinks[h] = h * .125f;
    Init(live, INT32, {rows}, lengths.data());
    if (length > 2048) Init(index, INT32, {rows, count}, selected.data());
    Init(keyPages, INT32, {pages}, kp.data()); Init(valuePages, INT32, {pages}, vp.data());
    if (withSink) Init(sink, FLOAT32, {heads}, sinks.data());
    std::vector<const void *> binding = {k.cudaData, v.cudaData, keyPages.cudaData, valuePages.cudaData};
    // Pointer table storage is byte-addressed; no ownership of the KV payloads.
    Init(table, INT32, {int(binding.size() * sizeof(void *) / sizeof(int))}, binding.data());
    std::vector<FastllmNaiveDSARow> descriptors(rows);
    for (int r = 0; r < rows; ++r) {
        auto &d = descriptors[r]; d.key = k.cudaData; d.value = v.cudaData;
        d.liveKeys = (int *)live.cudaData + r; d.length = d.capacity = length; d.keyStride = stride;
        d.indices = length > 2048 ? (int *)index.cudaData + r * count : nullptr;
        d.selectedCount = length > 2048 ? count : 0;
        d.cache = (const void *const *)table.cudaData;
        if (paged) { d.pageLen = pageLen; d.poolTokens = pool; }
    }
    FastllmNaiveDecodeScratch scratch;
    auto run = [&] { FastllmCudaNaiveBatchDSADecode(q, sink, descriptors, heads, kvHeads, dim, vd, scratch, out); };
    run(); auto actual = Read(out);
    double worst = 0;
    for (int r : {0, rows - 1}) {
        double error = 0, magnitude = 0;
        for (int h = 0; h < heads; ++h) {
            std::vector<double> score(count), prob(count);
            double maximum = withSink ? sinks[h] : -INFINITY;
            for (int j = 0; j < count; ++j) {
                int logical = length > 2048 ? selected[r * count + j] : j;
                if (logical < 0) { score[j] = -INFINITY; continue; }
                int physical = paged ? kp[logical / pageLen] * pageLen + logical % pageLen : logical;
                double sum = 0;
                for (int d = 0; d < dim; ++d)
                    sum += double(BFloat16BitsToFloat32(query[(r * heads + h) * dim + d])) *
                        BFloat16BitsToFloat32(key[physical * stride + h / (heads / kvHeads) * dim + d]);
                score[j] = sum / std::sqrt(double(dim)); maximum = std::max(maximum, score[j]);
            }
            double denominator = withSink ? std::exp(sinks[h] - maximum) : 0;
            for (int j = 0; j < count; ++j) denominator += (prob[j] = std::exp(score[j] - maximum));
            for (int d = 0; d < vd; ++d) {
                double expected = 0;
                for (int j = 0; j < count; ++j) {
                    int logical = length > 2048 ? selected[r * count + j] : j;
                    if (logical < 0) continue;
                    int physical = paged ? vp[logical / pageLen] * pageLen + logical % pageLen : logical;
                    expected += prob[j] / denominator * BFloat16BitsToFloat32(
                        value[(physical * kvHeads + h / (heads / kvHeads)) * vd + d]);
                }
                float got = BFloat16BitsToFloat32(actual[(r * heads + h) * vd + d]);
                Require(std::isfinite(got), "nonfinite DSA output");
                error += (got - expected) * (got - expected); magnitude += expected * expected;
            }
        }
        worst = std::max(worst, std::sqrt(error / std::max(magnitude, 1e-30)));
    }
    const bool splitPath = scratch.attentionCompletion.cudaData != nullptr;
    // Existing BF16 logits/probabilities round at additional boundaries. Test
    // their fallback separately from the tighter FP32-accumulating split path.
    printf("DSA rows=%d keys=%d heads=%d kv=%d paged=%d sink=%d unaligned=%d split=%d fp32_relative_l2=%.8g\n",
        rows, length, heads, kvHeads, paged, withSink, unaligned, splitPath, worst);
    Require(!unaligned || !splitPath, "unaligned query did not use the scalar fallback");
    Require(worst < (splitPath ? .005 : .025), "DSA relative L2 against FP32 reference exceeded limit");
    void *partial = scratch.attentionPartial.cudaData, *completion = scratch.attentionCompletion.cudaData;
    cudaGraph_t graph{}; cudaGraphExec_t executable{};
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); run();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, 0));
    for (int i = 0; i < 8; ++i) Check(cudaGraphLaunch(executable, cudaStreamPerThread));
    Require(Read(out) == actual, "DSA graph replay changed output");
    Check(cudaMemset(live.cudaData, 0, live.GetBytes())); Check(cudaGraphLaunch(executable, cudaStreamPerThread));
    auto zero = Read(out); for (auto x : zero) Require((x & 0x7fff) == 0, "inactive row was not cleared");
    Check(cudaMemcpy(live.cudaData, lengths.data(), live.GetBytes(), cudaMemcpyHostToDevice));
    // Rebind values through the fixed device table without changing the graph.
    for (auto &x : value) x ^= 0x8000;
    Data rebound; Init(rebound, BFLOAT16, {pool, kvHeads * vd}, value.data());
    binding[1] = rebound.cudaData;
    Check(cudaMemcpy(table.cudaData, binding.data(), table.GetBytes(), cudaMemcpyHostToDevice));
    Check(cudaGraphLaunch(executable, cudaStreamPerThread)); auto negative = Read(out);
    for (size_t i = 0; i < actual.size(); ++i)
        Require(std::abs(BFloat16BitsToFloat32(negative[i]) + BFloat16BitsToFloat32(actual[i])) < 1e-6f,
                "DSA graph ignored rebound KV table");
    Require(scratch.attentionPartial.cudaData == partial && scratch.attentionCompletion.cudaData == completion,
            "DSA graph moved workspace");
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
    if (completion) {
        std::vector<unsigned> counters(scratch.attentionCompletion.Count(0));
        Check(cudaMemcpy(counters.data(), completion, counters.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        Require(std::all_of(counters.begin(), counters.end(), [](unsigned value) { return value == 0; }),
                "split completion counter was not reset after graph replay");
    }
}
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        FastllmCudaSetDevice(0);
        for (int n : {1, 3, 8, 33}) for (int length : {17, 121, 512, 513, 2048, 4096})
            Case(n, length, 8, 1, length % 2, length % 3);
        Case(5, 237, 16, 2, true, true);
        Case(2, 121, 4, 1, true, true); // GQA group not divisible by eight: legacy fallback.
        Case(2, 121, 8, 1, false, false, true); // BF16-aligned but not 16-byte-aligned.
        puts("PASS: DSA split accuracy, variable rows, paging, fallback and graph rebinding");
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
