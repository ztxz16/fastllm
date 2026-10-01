#include "fastllm.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static float BF(float x) { return RoundFloat32ToBFloat16RNE(x); }
static std::vector<float> Values(int size, int seed) {
    std::vector<float> result(size);
    for (int i = 0; i < size; i++) result[i] = BF(std::sin((i + seed) * 0.739f) * 0.7f);
    return result;
}
static void Upload(Data &data, const std::vector<int> &shape, const std::vector<float> &values) {
    data.Resize(shape);
    data.Allocate();
    for (size_t i = 0; i < values.size(); i++) {
        if (data.dataType == BFLOAT16) ((uint16_t *)data.cpuData)[i] = Float32ToBFloat16RNEBits(values[i]);
        else ((float *)data.cpuData)[i] = values[i];
    }
    data.ToDevice(DataDevice::CUDA, {0}, true);
}
static std::vector<float> ReadBF(Data &data) {
    data.ToDevice(DataDevice::CPU);
    std::vector<float> values(data.Count(0));
    for (size_t i = 0; i < values.size(); i++)
        values[i] = BFloat16BitsToFloat32(((uint16_t *)data.cpuData)[i]);
    return values;
}
static void TestRope() {
    const int heads = 4, dim = 192, rotary = 64;
    auto x = Values(3 * heads * dim, 11);
    Data input(BFLOAT16), positions(FLOAT32);
    Upload(input, {1, 3, heads * dim}, x);
    Upload(positions, {1, 3}, {0, 129, 2057});
    FastllmCudaNaiveRope(input, positions, heads, dim, rotary, 1e7f);
    auto actual = ReadBF(input);
    for (int row = 0; row < 3 * heads; row++) {
        int pos = row / heads == 0 ? 0 : row / heads == 1 ? 129 : 2057;
        for (int d = 0; d < dim; d++) {
            float expected = x[row * dim + d];
            if (d < rotary) {
                int pair = d % (rotary / 2);
                float angle = pos / std::pow(1e7f, 2.0f * pair / rotary);
                float c = BF(std::cos(angle)), s = BF(std::sin(angle));
                float a = x[row * dim + pair], b = x[row * dim + pair + rotary / 2];
                expected = d < rotary / 2 ? BF(BF(a*c) - BF(b*s)) : BF(BF(b*c) + BF(a*s));
            }
            Require(std::abs(actual[row * dim + d] - expected) <= 0.008f,
                    "partial GPT-NeoX RoPE mismatch");
        }
    }
}
static void TestAttention(int past, int window, bool sparse, bool withSink, int queries = 3, bool causal = true) {
    const int heads = 4, kvHeads = 2, dim = 192, vd = 128;
    int keys = past + queries, keyStride = kvHeads * dim + (window ? 0 : 128);
    auto q = Values(queries * heads * dim, 7);
    auto k = Values(keys * keyStride, 17), v = Values(keys * kvHeads * vd, 31);
    Data qd(BFLOAT16), kd(BFLOAT16), vdta(BFLOAT16), idx(INT32), sink(FLOAT32), out;
    Upload(qd, {1, queries, heads * dim}, q);
    Upload(kd, {1, keys, keyStride}, k);
    Upload(vdta, {1, keys, kvHeads * vd}, v);
    if (withSink) Upload(sink, {heads}, {1, -1, 0, 3});
    int count = sparse ? 2048 : window ? std::min(window + (causal ? 0 : queries - 1), keys) : keys;
    std::vector<int> selected(queries * count);
    if (sparse) {
        idx.Resize({queries, count}); idx.Allocate();
        for (int row = 0; row < queries; row++) {
            for (int i = 0; i < count; i++) selected[row * count + i] = keys - 1 - i;
            selected[row * count + count - 1] = -1;
        }
        std::memcpy(idx.cpuData, selected.data(), selected.size() * sizeof(int));
        idx.ToDevice(DataDevice::CUDA, {0}, true);
    }
    FastllmCudaNaiveAttention(qd, kd, vdta, idx, sink, heads, kvHeads, dim, vd, past, window, out, causal);
    auto actual = ReadBF(out);
    const float sinks[] = {1, -1, 0, 3};
    for (int row = 0; row < queries; row++) for (int h = 0; h < heads; h++) {
        std::vector<double> prob(keys, 0);
        double denominator = withSink ? std::exp(sinks[h]) : 0;
        for (int t = 0; t < keys; t++) {
            bool allowed = (!causal || t <= past + row) && (!window || past + row - t < window);
            if (sparse) allowed &= std::find(selected.begin() + row * count,
                selected.begin() + (row + 1) * count, t) != selected.begin() + (row + 1) * count;
            if (!allowed) continue;
            double dot = 0;
            for (int d = 0; d < dim; d++) dot += (double)q[(row * heads + h) * dim + d] *
                k[t * keyStride + (h / (heads / kvHeads)) * dim + d];
            prob[t] = std::exp(BF(BF(dot) / std::sqrt((float)dim)));
            denominator += prob[t];
        }
        for (int d = 0; d < vd; d++) {
            double sum = 0;
            for (int t = 0; t < keys; t++) sum += BF(prob[t] / denominator) *
                v[(t * kvHeads + h / (heads / kvHeads)) * vd + d];
            Require(std::abs(actual[(row * heads + h) * vd + d] - BF(sum)) < 0.002f,
                    "GQA/sink/sliding/sparse attention mismatch");
        }
    }
}
static float FP8(float x) {
    // Independent scalar E4M3 round-to-nearest-even reference.
    float magnitude = std::abs(x), best = 0, distance = magnitude;
    int bestBits = 0;
    for (int bits = 1; bits <= 126; bits++) {
        int exp = bits >> 3, mantissa = bits & 7;
        float value = exp ? std::ldexp(1.0f + mantissa / 8.0f, exp - 7) : std::ldexp(mantissa / 8.0f, -6);
        float delta = std::abs(value - magnitude);
        if (delta < distance || (delta == distance && bits % 2 == 0 && bestBits % 2)) {
            best = value; distance = delta; bestBits = bits;
        }
    }
    return std::copysign(best, x);
}
static void TestIndexer(bool zeroWeights, bool fp8, int queries = 3, int keys = 2060,
                        int topK = 2048) {
    const int heads = 4, dim = 128, past = keys - queries;
    const int stride = 2 * 192 + dim;
    auto q = Values(queries * heads * dim, 43), k = Values(keys * stride, 79);
    auto w = Values(queries * heads, 97);
    if (zeroWeights) std::fill(w.begin(), w.end(), 0);
    Data qd(BFLOAT16), kd(BFLOAT16), wd(BFLOAT16), indices;
    Upload(qd, {1, queries, heads * dim}, q);
    Upload(kd, {1, keys, stride}, k);
    Upload(wd, {1, queries, heads}, w);
    FastllmCudaNaiveIndexer(qd, wd, kd, heads, dim, past, topK, fp8, indices);
    indices.ToDevice(DataDevice::CPU);
    if (queries > 32) {
        // Compare every index, including tie order and -1 padding, against
        // a single active worker using exactly the same CUDA scores.
        std::vector<int> parallel((int *)indices.cpuData,
                                  (int *)indices.cpuData + queries * topK);
        auto *pool = GetAlivePool();
        auto active = pool->curActivateThreadInterval;
        pool->curActivateThreadInterval = {active.first, active.first + 1};
        FastllmCudaNaiveIndexer(qd, wd, kd, heads, dim, past, topK, fp8, indices);
        pool->curActivateThreadInterval = active;
        indices.ToDevice(DataDevice::CPU);
        Require(std::equal(parallel.begin(), parallel.end(), (int *)indices.cpuData),
                "parallel indexer differs from serial indexer");
    }
    auto roundRow = [&](float *x) {
        if (!fp8) return;
        float scale = 1e-4f;
        for (int d = 0; d < dim; d++) scale = std::max(scale, std::abs(x[d]));
        scale /= 448;
        for (int d = 0; d < dim; d++) x[d] = FP8(x[d] / scale) * scale;
    };
    for (int row = 0; row < queries * heads; row++) roundRow(q.data() + row * dim);
    for (int t = 0; t < keys; t++) roundRow(k.data() + t * stride + stride - dim);
    for (int row = 0; row < queries; row++) {
        std::vector<double> scores(past + row + 1);
        for (int t = 0; t <= past + row; t++) for (int h = 0; h < heads; h++) {
            double dot = 0;
            for (int d = 0; d < dim; d++) dot += (double)q[(row * heads + h) * dim + d] * k[t * stride + stride - dim + d];
            scores[t] += std::max(dot, 0.0) * w[row * heads + h];
        }
        std::vector<int> order(scores.size()); std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
        for (int i = 0; i < topK; i++) {
            int actual = ((int *)indices.cpuData)[row * topK + i];
            if (i > past + row) {
                Require(actual == -1, "indexer did not pad unavailable causal keys");
                continue;
            }
            Require(actual >= 0 && actual <= past + row, "indexer selected a future key");
            Require(zeroWeights ? actual == i : std::abs(scores[actual] - scores[order[i]]) < 2e-5,
                    "FP8 indexer or stable TopK mismatch");
        }
    }
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        SetThreads(4);
        TestRope();
        TestAttention(15, 0, false, false, 7, false);
        TestAttention(1023, 1024, false, false, 7, false);
        TestAttention(0, 128, false, true);
        TestAttention(127, 128, false, true);
        TestAttention(7, 0, false, false);
        TestAttention(2057, 0, true, false);
        TestAttention(2057, 0, true, false, 1);
        TestAttention(127, 128, false, true, 1);
        TestAttention(255, 0, false, false, 1);
        TestAttention(256, 0, false, true, 1);
        TestAttention(0, 128, false, true, 54);
        TestIndexer(true, true);
        TestIndexer(false, true);
        TestIndexer(false, false);
        TestIndexer(true, true, 1);
        TestIndexer(false, true, 1);
        TestIndexer(false, false, 1);
        auto *pool = GetAlivePool();
        auto active = pool->curActivateThreadInterval;
        pool->curActivateThreadInterval = {1, 4};
        TestIndexer(true, true, 65, 129, 256);
        TestIndexer(false, false, 65, 257, 128);
        TestIndexer(false, true, 65, 257, 128);
        pool->curActivateThreadInterval = active;
        std::puts("Naive-N0.5 CUDA regression passed");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
    return 0;
}
