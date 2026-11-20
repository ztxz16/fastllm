#include "../../src/devices/cuda/models/naive-n05-dsa-mma.cuh"
#include <algorithm>
#include "../../src/devices/cuda/models/naive-n05-indexer-batch.cuh"
#include "../../src/devices/cuda/models/naive-n05-indexer-sortpack.cuh"
#include "../../src/devices/cuda/models/naive-n05-indexer-sm100.cuh"
#include "../../src/devices/cuda/models/naive-n05-indexer-wide-select.cuh"
#include "../../src/devices/cuda/models/naive-n05-indexer-staged.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace b = naive_indexer_batch;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Cuda(cudaError_t code) { Check(code == cudaSuccess, cudaGetErrorString(code)); }
template <class T> struct Buffer {
    T *p = nullptr;
    size_t n;
    explicit Buffer(size_t n) : n(n) { Cuda(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    void Put(const std::vector<T> &v) {
        Check(v.size() == n, "upload size");
        Cuda(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> Get() {
        std::vector<T> v(n);
        Cuda(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
static float Decode(unsigned bits) {
    unsigned raw = bits & 0x80000000u ? bits ^ 0x80000000u : ~bits;
    float value;
    std::memcpy(&value, &raw, 4);
    return value;
}
static void Case(int count, int capacity, int stride, int page, bool fp8 = true, bool scaled = true, int queryGroup = 1) {
    const int rowStride = stride / (fp8 ? 1 : 2);
    const int pages = capacity / page;
    unsigned seed = 171117;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (int(seed >> 16) - 32768) / 16384.f;
    };
    Buffer<unsigned char> pool0(size_t(count) * capacity * stride), pool1(pool0.n);
    std::vector<unsigned char> pool(pool0.n, 0);
    for (auto *dst : {&pool0, &pool1}) {
        for (size_t i = 0; i < size_t(count) * capacity; ++i) {
            if (scaled) {
                for (int d = 0; d < 128; ++d)
                    pool[i * stride + stride - 144 + d] = __nv_fp8_e4m3(rnd()).__x;
                float scale = std::fabs(rnd()) + .05f;
                std::memcpy(pool.data() + i * stride + stride - 16, &scale, 4);
            } else if (fp8) {
                for (int d = 0; d < 128; ++d)
                    pool[i * stride + stride - 128 + d] = __nv_fp8_e4m3(rnd()).__x;
            } else
                for (int d = 0; d < 128; ++d) {
                    auto value = __float2bfloat16(rnd());
                    std::memcpy(pool.data() + i * stride + stride - 256 + d * 2, &value, 2);
                }
        }
        dst->Put(pool);
    }
    Buffer<int> table0(count * pages), table1(count * pages), live(count);
    std::vector<int> table(table0.n), lengths(count);
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = table.size() - 1 - i;
    table0.Put(table);
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = (i + 17) % table.size();
    table1.Put(table);
    Buffer<__nv_bfloat16> q(count * 2048), w(count * 16);
    Buffer<float> scale(count * 16);
    std::vector<__nv_bfloat16> query(q.n), weights(w.n);
    std::vector<float> scales(scale.n);
    for (auto &v : query)
        v = __float2bfloat16(float(__nv_fp8_e4m3(rnd())));
    for (auto &v : scales)
        v = std::fabs(rnd()) + .05f;
    q.Put(query);
    scale.Put(scales);
    Buffer<unsigned> ref(size_t(count) * capacity), out(ref.n);
    Buffer<naive_topk::Workspace> workspace0(count), workspace1(count);
    Buffer<int> indices0(count * 2048), indices1(count * 2048);
    Buffer<const void *> binding(count * 4);
    std::vector<const void *> bindings(binding.n, nullptr);
    b::Batch original{}, compact{};
    naive_indexer_sm100::Row native[naive_indexer_sm100::BatchLimit];
    for (int i = 0; i < count; ++i) {
        auto key = FastllmNaiveKV(pool0.p, table0.p + (i / queryGroup) * pages, page, fp8);
        key.scaledIndexer = scaled;
        original.rows[i] = {key,
                            live.p + i,
                            q.p + i * 2048,
                            scale.p + i * 16,
                            ref.p + size_t(i) * capacity,
                            workspace0.p + i,
                            indices0.p + i * 2048,
                            i,
                            rowStride,
                            binding.p + i * 4};
        compact.rows[i] = original.rows[i];
        compact.rows[i].workspace = workspace1.p + i;
        compact.rows[i].indices = indices1.p + i * 2048;
        native[i] = {key, live.p + i, q.p + i * 2048,   scale.p + i * 16, out.p + size_t(i) * capacity,
                     i,   rowStride,  binding.p + i * 4};
    }
    auto fused = naive_indexer_sortpack::SelectSort<4096, true>;
    constexpr size_t sortBytes = naive_indexer_sortpack::SortBytes<256, 5, true, false, 4096>();
    Cuda(cudaFuncSetAttribute(fused, cudaFuncAttributeMaxDynamicSharedMemorySize, sortBytes));
    auto topk = [&]() {
        b::Select<><<<count, 256, 0, cudaStreamPerThread>>>(original);
        b::Sort<><<<count, 256, 0, cudaStreamPerThread>>>(original);
        if (queryGroup > 1) {
            naive_indexer_wide_select::SelectWide<<<count, 1024, 0, cudaStreamPerThread>>>(compact);
            b::Sort<><<<count, 256, 0, cudaStreamPerThread>>>(compact);
        } else {
            fused<<<count, 256, sortBytes, cudaStreamPerThread>>>(compact);
        }
    };
    auto update = [&](int step) {
        const int sizes[] = {0, 1, 63, 64, 65, 127, 128, 129, 2048, 2049, 8191, capacity};
        for (int i = 0; i < count; ++i) {
            lengths[i] = sizes[(i + step) % 12];
            bindings[i * 4] = step % 2 ? pool1.p : pool0.p;
            bindings[i * 4 + 2] = (step % 2 ? table1.p : table0.p) + (i / queryGroup) * pages;
        }
        for (auto &v : weights)
            v = __float2bfloat16(step == 2 ? 0.f : rnd());
        live.Put(lengths);
        binding.Put(bindings);
        w.Put(weights);
    };
    update(0);
    Check(!FastllmCudaNaiveIndexerSm100(native, 1, capacity, w.p, cudaStreamPerThread), "small-batch fallback");
    b::Batch staged = original;
    for (int i = 0; i < count; ++i)
        staged.rows[i].scores = out.p + size_t(i) * capacity;
    auto score = [&] {
        const bool launched = FastllmCudaNaiveIndexerSm100(native, count, capacity, w.p, cudaStreamPerThread);
        Check(launched == (scaled || fp8), "native FP8 / BF16 fallback dispatch");
        if (!launched)
            b::HopperScores<false>
                <<<dim3((148 * 7 + count - 1) / count, count), 128, 0, cudaStreamPerThread>>>(w.p, staged);
    };
    // Warm host descriptor/support caches before graph capture.
    score();
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    b::Scores<<<dim3(8, count), 256, 0, cudaStreamPerThread>>>(w.p, original);
    score();
    topk();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    double worst = 0;
    for (int step = 0; step < 6; ++step) {
        update(step);
        Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        auto a = ref.Get(), z = out.Get();
        b::Batch staged = original;
        for (int i = 0; i < count; ++i)
            staged.rows[i].scores = out.p + size_t(i) * capacity;
        if (scaled || fp8)
            b::HopperScoresSharedHead<><<<dim3(8, count), 128, 0, cudaStreamPerThread>>>(w.p, staged);
        else
            b::HopperScores<false><<<dim3(8, count), 128, 0, cudaStreamPerThread>>>(w.p, staged);
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        auto fallback = out.Get();
        double error = 0, norm = 0;
        for (int row = 0; row < count; ++row)
            for (int i = 0; i < lengths[row]; ++i) {
                size_t j = size_t(row) * capacity + i;
                if (!scaled && !fp8) Check(a[j] == z[j], "BF16 score bits changed");
                double x = Decode(a[j]), y = Decode(z[j]);
                Check(std::fabs(Decode(fallback[j]) - x) <= 1.e-4 * std::max(1., std::fabs(x)),
                      "staged FP8 fallback mismatch");
                Check(std::isfinite(y), "nonfinite score");
                error += (x - y) * (x - y);
                norm += x * x;
            }
        double l2 = std::sqrt(error / std::max(norm, 1.e-30));
        worst = std::max(worst, l2);
        Check(l2 < 3.e-4, "score mismatch after graph replay");
        Check(indices0.Get() == indices1.Get(), "stable TopK order differs");
    }
    // A partially tied threshold must use bounded compaction, not overflow it.
    std::vector<unsigned> ties(ref.n);
    for (size_t i = 0; i < ties.size(); ++i)
        ties[i] = 0x80000000u + unsigned(i % 3);
    ref.Put(ties);
    topk();
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    Check(indices0.Get() == indices1.Get(), "threshold ties differ");
    if (queryGroup == naive_indexer_sm100::VerifyQueries) {
        // Repeat the fixture's owners in a different order in each block of rows.
        // Compare full output spans, including empty, padded and partially live tiles.
        using namespace naive_indexer_sm100;
        VerifyBatch batch{};
        Buffer<unsigned> wide(size_t(VerifyLimit) * capacity);
        Buffer<__nv_bfloat16> input(q.n), packed(size_t(VerifyLimit) * 2048);
        Buffer<float> packedScales(size_t(VerifyLimit) * 16);
        for (auto &value : query) value = __float2bfloat16(rnd());
        input.Put(query);
        auto sourceRow = [&](int row) {
            const int group = (row / queryGroup + row / count) % (count / queryGroup);
            return group * queryGroup + row % queryGroup;
        };
        for (int i = 0; i < VerifyLimit; ++i) {
            const int source = sourceRow(i);
            batch.rows[i] = native[source];
            batch.rows[i].cache = native[source / queryGroup * queryGroup].cache;
            batch.rows[i].query = packed.p + size_t(i) * 2048;
            batch.rows[i].scale = packedScales.p + size_t(i) * 16;
            batch.rows[i].scores = wide.p + size_t(i) * capacity;
        }
        const auto stream = cudaStreamPerThread;
        constexpr size_t scratchBytes = sizeof(naive_topk::Workspace::partials);
        auto wideScore = [&](int n, size_t bytes) {
            return FastllmCudaNaiveVerifyIndexerSm100(batch, n, capacity, input.p, w.p,
                                                       workspace1.p->partials, bytes, stream);
        };
        Check(!wideScore(VerifyLimit - 1, scratchBytes) && !wideScore(VerifyLimit + queryGroup, scratchBytes),
              "invalid shared-query group");
        Check(!wideScore(VerifyLimit, 0), "missing verification scratch must fall back");
        Check(wideScore(VerifyLimit, scratchBytes), "wide scorer warmup");
        Cuda(cudaStreamSynchronize(stream));
        cudaGraph_t verifyGraph;
        cudaGraphExec_t verifyExec;
        Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        Cuda(cudaMemsetAsync(out.p, 0xff, out.n * sizeof(unsigned), stream));
        Cuda(cudaMemsetAsync(wide.p, 0xff, wide.n * sizeof(unsigned), stream));
        b::Quantize<<<dim3(16, count), 128, 0, stream>>>(input.p, original);
        score();
        Check(wideScore(VerifyLimit, scratchBytes), "wide scorer declined capture");
        Cuda(cudaStreamEndCapture(stream, &verifyGraph));
        Cuda(cudaGraphInstantiate(&verifyExec, verifyGraph, nullptr, nullptr, 0));
        for (int step = 0; step < 5; ++step) {
            update(step);
            if (step >= 3) {
                std::fill(lengths.begin(), lengths.end(), step == 3 ? capacity : 0);
                live.Put(lengths);
            }
            Cuda(cudaGraphLaunch(verifyExec, stream));
            Cuda(cudaStreamSynchronize(stream));
            const auto expected = out.Get(), actual = wide.Get();
            for (int i = 0; i < VerifyLimit; ++i) {
                auto row = expected.begin() + size_t(sourceRow(i)) * capacity;
                Check(std::equal(row, row + capacity, actual.begin() + size_t(i) * capacity),
                      "wide verification scores or inactive positions changed after graph replay");
            }
        }
        Cuda(cudaGraphExecDestroy(verifyExec));
        Cuda(cudaGraphDestroy(verifyGraph));
    }
    Cuda(cudaGraphExecDestroy(executable));
    Cuda(cudaGraphDestroy(graph));
    printf("PASS batch=%d capacity=%d stride=%d page=%d scaled=%d L2=%.9g graph/rebinding/TopK\n", count,
           capacity, stride, page, int(scaled), worst);
}
static void PrefillCase(int rows, int keys, bool paged, bool scaled, int keep, int rowBytes) {
    Check(keep <= keys - rows + 1, "TopK must fit every reference row");
    const int page = 64, pages = (keys + page - 1) / page;
    const int stride = scaled ? rowBytes / 2 : rowBytes;
    const int past = keys - rows;
    unsigned seed = 718913;
    auto sample = [&]() {
        seed = 1664525u * seed + 1013904223u;
        return (int(seed >> 16) - 32768) / 16384.f;
    };
    Buffer<unsigned char> pool(size_t(pages) * page * rowBytes);
    std::vector<unsigned char> hostPool(pool.n);
    for (size_t i = 0; i < size_t(pages) * page; ++i) {
        auto *p = hostPool.data() + (i + 1) * rowBytes - (scaled ? FastllmNaiveIndexerBytes : 128);
        for (int d = 0; d < 128; ++d) p[d] = __nv_fp8_e4m3(sample()).__x;
        if (scaled) {
            float scale = .05f + std::fabs(sample());
            std::memcpy(p + 128, &scale, sizeof(scale));
        }
    }
    pool.Put(hostPool);
    Buffer<int> table(pages);
    std::vector<int> map(pages);
    for (int p = 0; p < pages; ++p) map[p] = pages - 1 - p;
    table.Put(map);
    FastllmNaiveKV cache(pool.p, paged ? table.p : nullptr, paged ? page : 0, !scaled);
    cache.scaledIndexer = scaled;
    Buffer<__nv_bfloat16> query(size_t(rows) * 16 * 128), weights(size_t(rows) * 16), rounded(query.n);
    std::vector<__nv_bfloat16> input(query.n), weight(weights.n);
    for (auto &x : input) x = __float2bfloat16(sample());
    for (auto &x : weight) x = __float2bfloat16(sample());
    query.Put(input);
    weights.Put(weight);
    Buffer<float> scales(weights.n), ref(size_t(rows) * keys), actual(ref.n);
    const auto stream = cudaStreamPerThread;
    naive_dsa_mma::QuantizeIndexer<<<rows * 16, 128, 0, stream>>>(query.p, rounded.p, scales.p, 128, 0);
    naive_dsa_mma::IndexerScores<true><<<dim3((keys + 63) / 64, (rows + 31) / 32), 128, 0, stream>>>(
        rounded.p, cache, scales.p, weights.p, ref.p, rows, keys, past, stride, 0, 0);
    Cuda(cudaGetLastError());
    auto run = [&](size_t bytesAvailable) {
        return FastllmCudaNaivePrefillIndexerSm100(query.p, weights.p, cache, stride, rows, keys, past,
            rounded.p, bytesAvailable, scales.p, actual.p, stream);
    };
    Check(!run(1), "prefill scorer accepted undersized scratch");
    Check(run(rounded.n * sizeof(__nv_bfloat16)), "prefill scorer declined supported layout");
    Cuda(cudaStreamSynchronize(stream));
    auto old = ref.Get(), now = actual.Get();
    double error = 0, norm = 0;
    size_t matched = 0;
    std::vector<int> original(keys), selected(keys);
    for (int row = 0; row < rows; ++row) {
        const size_t base = size_t(row) * keys;
        const int live = past + row + 1;
        for (int k = 0; k < keys; ++k) {
            if (k >= live) {
                Check(now[base + k] == -INFINITY, "noncausal score escaped the mask");
                continue;
            }
            const double d = now[base + k] - old[base + k];
            error += d * d;
            norm += double(old[base + k]) * old[base + k];
            original[k] = selected[k] = k;
        }
        auto select = [&](std::vector<int> &ids, const std::vector<float> &values) {
            std::partial_sort(ids.begin(), ids.begin() + keep, ids.begin() + live, [&](int a, int b) {
                return values[base + a] == values[base + b] ? a < b : values[base + a] > values[base + b];
            });
            std::sort(ids.begin(), ids.begin() + keep);
        };
        select(original, old);
        select(selected, now);
        for (int i = 0; i < keep; ++i)
            matched += std::binary_search(original.begin(), original.begin() + keep, selected[i]);
    }
    const double l2 = std::sqrt(error / norm), overlap = double(matched) / (rows * keep);
    printf("Prefill rows=%d keys=%d paged=%d scaled=%d topK=%d score L2=%.9g TopK overlap=%.9f\n",
        rows, keys, paged, scaled, keep, l2, overlap);
    Check(l2 < .003 && overlap >= .999, "prefill scores or TopK differ from reference");
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    cudaDeviceProp prop{};
    Cuda(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 10 || (prop.minor != 0 && prop.minor != 3)) return 77;
    try {
        PrefillCase(129, 65539, true, true, 2048, 400);
        PrefillCase(137, 1109, false, false, 128, 128);
        PrefillCase(259, 2053, true, true, 128, 400);
        PrefillCase(259, 2053, true, true, 128, FastllmNaiveIndexerBytes);
        PrefillCase(1025, 8192, true, true, 2048, 400);
        Case(32, 8704, 128, 128, true, false);
        Case(64, 8704, 896, 128, true, false);
        Case(32, 8704, 144, 128);
        Case(64, 8704, 400, 256);
        Case(128, 20480, 144, 128);
        Case(32, 32768, 144, 128, true, true, 4);
        Case(64, 8704, 1680, 128, false);
        Case(8, 8704, 1792, 128, false, false);
        Case(16, 8704, 1792, 128, false, false);
        Case(64, 8704, 1792, 128, false, false);
        Case(128, 8704, 1792, 128, false, false);
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
