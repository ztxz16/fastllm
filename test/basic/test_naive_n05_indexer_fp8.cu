#include "../../src/devices/cuda/models/naive-n05-indexer.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "fastllm.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_fp8.h>
#include <set>
#include <stdexcept>
using namespace fastllm;
static void Check(bool x, const char *s) {
    if (!x)
        throw std::runtime_error(s);
}
static void Case(int rows, int past, int part, int gap, bool scaled, bool zero, int padding = 3) {
    const int keys = past + rows + gap + padding, stride = scaled ? 144 : 128;
    unsigned seed = 127;
    auto rnd = [&]() {
        seed = 1664525u * seed + 1013904223u;
        return (int(seed >> 16) - 32768) / 32768.f;
    };
    std::vector<float> qv(size_t(rows) * 2048), wv(size_t(rows) * 16), sv(size_t(rows) * 16);
    std::vector<uint8_t> kv(size_t(keys) * stride);
    std::vector<float> keyscale(keys, 1);
    for (auto &v : qv)
        v = zero ? 0 : float(__nv_fp8_e4m3(rnd() * 4));
    for (auto &v : wv)
        v = float(__float2bfloat16(rnd()));
    for (auto &v : sv)
        v = exp2f(int(fabs(rnd()) * 5) - 3);
    for (int r = 0; r < keys; ++r) {
        for (int d = 0; d < 128; ++d)
            kv[size_t(r) * stride + d] = __nv_fp8_e4m3(rnd() * 3).__x;
        if (scaled) {
            keyscale[r] = exp2f(r % 5 - 3);
            memcpy(kv.data() + size_t(r) * stride + 128, &keyscale[r], 4);
        }
    }
    Data q(BFLOAT16, {rows, 16, 128}, qv), w(BFLOAT16, {rows, 16}, wv), scale(FLOAT32, {rows, 16}, sv),
        key(FP8_E4M3, {1, keys, stride}), scores(FLOAT32, {rows, keys});
    key.kvCacheFp8Indexer = scaled;
    for (Data *d : {&q, &w, &scale})
        d->ToDevice(DataDevice::CUDA, {0}, true);
    for (Data *d : {&key, &scores}) {
        d->ToDevice(DataDevice::CUDA, {0}, false);
        d->Allocate(false);
    }
    cudaMemcpy(key.cudaData, kv.data(), kv.size(), cudaMemcpyHostToDevice);
    naive_indexer::FP8Entry e{FastllmNaiveKV(key),
                              (const __nv_bfloat16 *)q.cudaData,
                              (const __nv_bfloat16 *)w.cudaData,
                              (float *)scale.cudaData,
                              (float *)scores.cudaData,
                              rows,
                              keys,
                              past,
                              stride,
                              part,
                              gap};
    Data staging(FP8_E4M3);
    naive_indexer::FP8Entry stagedEntry{};
    auto launch = [&](const naive_indexer::FP8Entry &entry) {
        return naive_indexer::LaunchFP8(&entry, 1, staging, &stagedEntry);
    };
    Check(launch(e) == 1, "native FP8 did not launch");
    FastllmCudaSyncCurrentThreadStream();
    std::vector<float> actual(size_t(rows) * keys);
    Check(cudaMemcpy(actual.data(), scores.cudaData, actual.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess,
          "score copy");
    // The coarse matrix is scratch: only consumed rows may be reused.
    Check(!naive_indexer::StageRefinementKeys(e, 1), "undersized score-prefix scratch must fall back");
    const int consumed = std::min(512, part ? part : rows), expandedRows = past + rows + gap;
    const size_t expandedBytes = size_t(expandedRows) * 128 * 2;
    const bool eligible = scaled && expandedRows <= int64_t(rows) * 192 &&
                          expandedBytes <= (size_t(32) << 20) && expandedBytes <= size_t(consumed) * keys * 4;
    Check(naive_indexer::StageRefinementKeys(e, consumed) == eligible, "refinement staging eligibility");
    if (eligible) {
        std::vector<unsigned char> staged(actual.size() * 4);
        Check(cudaMemcpy(staged.data(), scores.cudaData, staged.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
              "staging copy");
        Check(std::memcmp(staged.data() + expandedBytes, (const unsigned char *)actual.data() + expandedBytes,
                          staged.size() - expandedBytes) == 0,
              "staging overwrote unconsumed scores");
        for (int r = 0; r < expandedRows; ++r)
            for (int d = 0; d < 128; ++d) {
                __nv_fp8_e4m3 v;
                v.__x = kv[size_t(r) * stride + d];
                __nv_bfloat16 expected = __float2bfloat16(float(v));
                Check(std::memcmp(staged.data() + (size_t(r) * 128 + d) * 2, &expected, 2) == 0,
                      "FP8 expansion changed BF16 bits");
            }
    }
    double err = 0, norm = 0, maxabs = 0;
    for (int r : {0, 1, rows / 2, rows - 1})
        for (int k = 0; k < keys; k += std::max(1, keys / 131)) {
            int end = past + r + (part && r >= part ? gap : 0);
            if (k > end) {
                Check(actual[size_t(r) * keys + k] == -INFINITY, "causal mask");
                continue;
            }
            float sum = 0;
            for (int h = 0; h < 16; ++h) {
                double dot = 0;
                for (int d = 0; d < 128; ++d) {
                    __nv_fp8_e4m3 v;
                    v.__x = kv[size_t(k) * stride + d];
                    dot += double(qv[(size_t(r) * 16 + h) * 128 + d]) * float(v);
                }
                sum += fmaxf(float(dot) * sv[r * 16 + h], 0.f) * wv[r * 16 + h];
            }
            float ref = sum * keyscale[k], got = actual[size_t(r) * keys + k];
            Check(std::isfinite(got), "nonfinite score");
            err += double(got - ref) * (got - ref);
            norm += double(ref) * ref;
            maxabs = fmax(maxabs, fabs(got - ref));
        }
    double nrmse = sqrt(err / std::max(norm, 1.e-30));
    printf("SCORE rows=%d past=%d nrmse=%g maxabs=%g\n", rows, past, nrmse,
           maxabs); // Hopper FP8 MMA has reduced product-accumulation precision, with FP32 output.
    Check(nrmse < 2.e-4, "score math mismatch");
    if (!part) {
        q.Reshape({1, rows, 2048});
        w.Reshape({1, rows, 16});
        Data a, b;
        FastllmCudaNaiveIndexer(q, w, key, 16, 128, past, 2048, true, a, false, nullptr, false);
        FastllmCudaNaiveIndexer(q, w, key, 16, 128, past, 2048, true, b, false, nullptr, true);
        FastllmCudaSyncCurrentThreadStream();
        std::vector<int> av(size_t(rows) * 2048), bv(av.size());
        cudaMemcpy(av.data(), a.cudaData, av.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(bv.data(), b.cudaData, bv.size() * 4, cudaMemcpyDeviceToHost);
        int same = 0, overlap = 0;
        for (size_t i = 0; i < av.size(); ++i)
            same += av[i] == bv[i];
        for (int r = 0; r < rows; ++r) {
            std::set<int> expected(av.begin() + size_t(r) * 2048, av.begin() + size_t(r + 1) * 2048);
            for (int k = 0; k < 2048; ++k)
                overlap += expected.count(bv[size_t(r) * 2048 + k]);
        }
        printf("TopK ordered=%g set=%g\n", double(same) / av.size(), double(overlap) / av.size());
        Check(zero ? same == av.size() : double(overlap) / av.size() >= .999, "FP8 TopK overlap/ties differ");
    }
    auto bad = e;
    bad.rows = 1;
    Check(launch(bad) == 0, "small fallback");
    void *g = nullptr;
    Check(FastllmCudaGraphBeginCapture(), "capture");
    auto mask = launch(e);
    Check(FastllmCudaGraphEndCapture(&g), "capture end");
    if (g)
        FastllmCudaGraphDestroy(g);
    Check(mask == 0, "graph fallback");
    printf("rows=%d keys=%d part=%d gap=%d scaled=%d zero=%d nrmse=%.9g maxabs=%.9g\n", rows, keys, part, gap,
           scaled, zero, nrmse, maxabs);
}
static void BalancedCase() {
    const int rows = 1024, past = 65536, keys = past + rows, stride = 144;
    std::vector<float> qv(size_t(rows) * 2048), wv(size_t(rows) * 16);
    unsigned seed = 41;
    auto rnd = [&]() {
        seed = 1664525u * seed + 1013904223u;
        return (int(seed >> 16) - 32768) / 32768.f;
    };
    for (auto &v : qv)
        v = rnd();
    for (auto &v : wv)
        v = rnd();
    std::vector<unsigned char> kv(size_t(keys) * stride);
    for (int r = 0; r < keys; ++r) {
        for (int d = 0; d < 128; ++d)
            kv[size_t(r) * stride + d] = __nv_fp8_e4m3(rnd() * 4).__x;
        float scale = exp2f(r % 5 - 3);
        memcpy(kv.data() + size_t(r) * stride + 128, &scale, 4);
    }
    Data q(BFLOAT16, {1, rows, 2048}, qv), w(BFLOAT16, {1, rows, 16}, wv), key(FP8_E4M3, {1, keys, stride});
    key.kvCacheFp8Indexer = true;
    q.ToDevice(DataDevice::CUDA, {0}, true);
    w.ToDevice(DataDevice::CUDA, {0}, true);
    key.ToDevice(DataDevice::CUDA, {0}, false);
    key.Allocate(false);
    cudaMemcpy(key.cudaData, kv.data(), kv.size(), cudaMemcpyHostToDevice);
    for (int rank = 0; rank < 2; ++rank) {
        Data reference, actual, batch0, batch1;
        Check(FastllmCudaNaiveBalancedIndexer(q, w, key, past, rank, 2, true, true, reference, nullptr, false),
              "balanced reference dispatch");
        Check(FastllmCudaNaiveBalancedIndexer(q, w, key, past, rank, 2, true, true, actual, nullptr, true),
              "balanced native dispatch");
        std::vector<FastllmNaivePrefillIndexerItem> items{{&q, &w, &key, &batch0, past},
                                                          {&q, &w, &key, &batch1, past}};
        Check(FastllmCudaNaiveBatchBalancedIndexer(items, rank, 2, true, true, nullptr, true),
              "batch native dispatch");
        FastllmCudaSyncCurrentThreadStream();
        std::vector<int> a(size_t(rows / 2) * 2048), b(a.size());
        cudaMemcpy(a.data(), reference.cudaData, a.size() * 4, cudaMemcpyDeviceToHost);
        for (Data *out : {&actual, &batch0, &batch1}) {
            cudaMemcpy(b.data(), out->cudaData, b.size() * 4, cudaMemcpyDeviceToHost);
            size_t overlap = 0;
            for (int row = 0; row < rows / 2; ++row) {
                std::set<int> expected(a.begin() + size_t(row) * 2048, a.begin() + size_t(row + 1) * 2048);
                for (int col = 0; col < 2048; ++col)
                    overlap += expected.count(b[size_t(row) * 2048 + col]);
            }
            Check(double(overlap) / a.size() >= .999, "balanced/batched FP8 TopK overlap differs");
        }
    }
    puts("PASS balanced two-part and multi-entry FP8 TopK overlap");
}
int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n)
        return 77;
    FastllmCudaSetDevice(0);
    if (FastllmCudaRuntimeArch() != 90)
        return 77;
    try {
        Case(4097, 0, 0, 0, true, false);
        Case(33, 4096, 0, 0, false, false);
        Case(65, 65536, 0, 0, true, false);
        Case(65, 65536, 0, 0, true, false, 4096);
        Case(33, 65536, 0, 0, true, true);
        Case(960, 65536, 0, 0, true, false);
        Case(32, 8192, 0, 0, true, true);
        Case(65, 65536, 0, 0, true, true);
        Case(64, 4096, 32, 3072, true, false);
        BalancedCase();
        puts("PASS native FP8 Indexer scores TopK causal ties fallback");
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
