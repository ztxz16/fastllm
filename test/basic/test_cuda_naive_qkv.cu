// QKV/RoPE fusion regression: exact dyadic projection, shared graph
// workspace, mutable page tables, ring wrap, and rejection without side effects.
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/naive-n05-swa-verify.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
namespace {
void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void Init(Data &x, const std::vector<int> &dims) {
    x.Resize(dims);
    x.ToDevice(DataDevice::CUDA, {0}, false);
    x.Allocate(false);
}
void Fill(Data &x, unsigned seed) {
    if (x.dataType == FP8_E4M3) {
        // Vary exponent and mantissa to exercise near-boundary BF16 QK/PV rounding.
        std::mt19937 generator(seed);
        std::normal_distribution<float> normal;
        std::vector<__nv_fp8_e4m3> values(x.Count(0));
        for (auto &v : values) v = __nv_fp8_e4m3(normal(generator));
        Check(cudaMemcpy(x.cudaData, values.data(), values.size(), cudaMemcpyHostToDevice));
        return;
    }
    std::vector<__nv_bfloat16> values(x.Count(0));
    // Every product and FP32 partial sum is exactly representable. This tests
    // the fused epilogue bitwise without hiding errors behind a GEMM tolerance.
    for (auto &v : values) {
        seed = seed * 1664525u + 1013904223u;
        v = __float2bfloat16_rn((int(seed >> 28) - 8) / 16.f);
    }
    Check(cudaMemcpy(x.cudaData, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
}
std::vector<uint8_t> Read(const Data &x) {
    std::vector<uint8_t> result(x.GetBytes());
    Check(cudaMemcpy(result.data(), x.cudaData, result.size(), cudaMemcpyDeviceToHost));
    return result;
}
struct Case {
    int batch, capacity, pageLen, width;
    bool ring;
    int verifyGroup;
    Data x{BFLOAT16}, q{BFLOAT16}, k{BFLOAT16}, v{BFLOAT16}, packed{BFLOAT16}, index{BFLOAT16};
    Data positions{FLOAT32}, lengths{INT32}, pages{INT32}, table{INT32};
    Data key{BFLOAT16}, value{BFLOAT16}, otherKey{BFLOAT16}, otherValue{BFLOAT16};
    PagedCacheManager keyPool, valuePool;
    std::vector<std::unique_ptr<Data>> keys, values, indices;
    std::vector<FastllmNaiveRopeCacheRow> rows;
    cudaGraphExec_t graph = nullptr;
    Case(int batch, bool ring, int pageLen = 64, DataType cacheType = BFLOAT16, int verifyGroup = 0)
        : batch(batch), capacity(ring ? (verifyGroup ? 136 : 128) : pageLen * 4), pageLen(ring ? 0 : pageLen),
          width(ring ? 192 : 320), ring(ring), verifyGroup(verifyGroup) {
        Init(x, {1, batch, 4096}); Init(q, {1, batch, 1536});
        Init(k, {1, batch, 192}); Init(v, {1, batch, 128});
        Init(packed, {1, batch, 1856}); Init(index, {1, batch, 128});
        Init(positions, {1, batch}); Init(lengths, {batch});
        Init(pages, {batch, 8}); Init(table, {batch, 8});
        for (Data *cache : {&key, &value, &otherKey, &otherValue}) {
            cache->dataType = cacheType;
            cache->UpdateUnitSize();
        }
        Init(key, {batch * capacity, width}); Init(value, {batch * capacity, 128});
        Init(otherKey, key.dims); Init(otherValue, value.dims);
        keyPool.Resize(key.dims); keyPool.FakeFrom(key, 0);
        valuePool.Resize(value.dims); valuePool.FakeFrom(value, 0);
        for (int b = 0; b < batch; ++b) {
            keys.emplace_back(new Data(cacheType, {1, capacity, width}, DataDevice::CUDA,
                ring ? (char *)key.cudaData + (size_t)b * capacity * width * key.unitSize : (void *)((int *)pages.cudaData + b * 8)));
            values.emplace_back(new Data(cacheType, {1, capacity, 128}, DataDevice::CUDA,
                ring ? (char *)value.cudaData + (size_t)b * capacity * 128 * value.unitSize : (void *)((int *)pages.cudaData + b * 8 + 4)));
            indices.emplace_back(new Data(BFLOAT16, {1, 1, 128}, DataDevice::CUDA,
                (char *)index.cudaData + b * 128 * 2));
            if (!ring) {
                keys.back()->isPagedKVCache = values.back()->isPagedKVCache = true;
                keys.back()->pageLen = values.back()->pageLen = pageLen;
                keys.back()->pagedKVCacheData = &keyPool;
                values.back()->pagedKVCacheData = &valuePool;
            }
            rows.push_back({keys.back().get(), values.back().get(), ring ? nullptr : indices.back().get(),
                           (const void *const *)table.cudaData + b * 4});
            if (verifyGroup) {
                int first = b / verifyGroup * verifyGroup;
                rows.back().key = keys[first].get(); rows.back().value = values[first].get();
                rows.back().cache = (const void *const *)table.cudaData + first * 4;
                rows.back().verifyOffset = b % verifyGroup;
            }
        }
        Fill(x, batch + 17); Fill(index, batch + 37); Update(0);
    }
    ~Case() { if (graph) cudaGraphExecDestroy(graph); }
    void Update(int iteration) {
        std::vector<float> pos(batch);
        std::vector<int> live(batch), ids(batch * 8);
        std::vector<void *> bindings(batch * 4);
        const int boundaries[] = {0, 1, 16, 17, 63, 64, 127, 128, 129, capacity, capacity + 1};
        Data &dstKey = iteration % 2 ? otherKey : key, &dstValue = iteration % 2 ? otherValue : value;
        for (int b = 0; b < batch; ++b) {
            pos[b] = (iteration > 1 ? 1048500 : 8192) + b;
            live[b] = boundaries[(b + iteration) % 11];
            for (int j = 0; j < 4; ++j) {
                ids[b * 8 + j] = ((b + iteration) % batch) * 4 + (3 - j);
                ids[b * 8 + 4 + j] = ((batch - 1 - b + iteration) % batch) * 4 + j;
            }
            bindings[b * 4] = (char *)dstKey.cudaData + (ring ? (size_t)b * capacity * width * key.unitSize : 0);
            bindings[b * 4 + 1] = (char *)dstValue.cudaData + (ring ? (size_t)b * capacity * 128 * value.unitSize : 0);
            bindings[b * 4 + 2] = (int *)pages.cudaData + b * 8;
            bindings[b * 4 + 3] = (int *)pages.cudaData + b * 8 + 4;
        }
        if (verifyGroup) for (int b = 0; b < batch; ++b) {
            int past = iteration == 0 ? 0 : iteration == 1 ? 127 : iteration == 2 ? 128 : 200003;
            pos[b] = past + b % verifyGroup;
            live[b] = iteration == 3 && b % verifyGroup >= (verifyGroup + 1) / 2 ? 0 : past + b % verifyGroup + 1;
        }
        Check(cudaMemcpy(positions.cudaData, pos.data(), batch * sizeof(float), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(lengths.cudaData, live.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(pages.cudaData, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(table.cudaData, bindings.data(), bindings.size() * sizeof(void *), cudaMemcpyHostToDevice));
    }
    void Reset() {
        for (Data *data : {&q, &key, &value, &otherKey, &otherValue})
            Check(cudaMemset(data->cudaData, 0x3d, data->GetBytes()));
    }
    std::vector<std::vector<uint8_t>> Output() {
        return {Read(q), Read(key), Read(value), Read(otherKey), Read(otherValue)};
    }
    bool Fused(Data &weight, Data &scratch, const Data &bias = Data()) {
        return FastllmCudaNaiveBatchQkvRopeAppendCache(x, weight, bias, q, k, v, positions, rows,
            lengths, 8, 1, 192, 128, 64, ring ? 10000.f : 10000000.f, .707f, ring ? 128 : 0, ring && !verifyGroup, scratch);
    }
    void Reference(Data &weight) {
        Data bias;
        Require(FastllmCudaBFloat16MatMulBFloat16(x, weight, bias, packed, batch, 4096, 1856), "Linear failed");
        Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, rows, lengths,
            8, 1, 192, 128, 64, ring ? 10000.f : 10000000.f, .707f, ring ? 128 : 0, &packed, ring && !verifyGroup), "RoPE failed");
    }
};
void VerifyProjection(Data &weight, int rows, int group) {
    Case c(rows, true, 64, BFLOAT16, group); Data scratch(BFLOAT16);
    if (rows > 16) {
        auto before = c.Output();
        Require(!c.Fused(weight, scratch), "large-row GEMM fallback was replaced");
        Require(before == c.Output(), "fallback changed output");
        return;
    }
    Require(c.Fused(weight, scratch), "verification QKV rejected");
    cudaGraph_t graph;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(c.Fused(weight, scratch), "verification QKV capture rejected");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&c.graph, graph, 0)); Check(cudaGraphDestroy(graph));
    for (int pass = 0; pass < 4; ++pass) {
        c.Update(pass); c.Reset(); Check(cudaGraphLaunch(c.graph, cudaStreamPerThread));
        auto actual = c.Output(); c.Reset(); c.Reference(weight);
        Require(actual == c.Output(), "verification QKV bits, candidate slot or graph rebinding differs");
    }
    c.Update(1);
    auto bench = [&](bool fused) {
        cudaGraph_t graph; cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        for (int repeat = 0; repeat < 100; ++repeat) {
            if (fused) Require(c.Fused(weight, scratch), "benchmark rejected");
            else c.Reference(weight);
        }
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, 0)); Check(cudaGraphDestroy(graph));
        cudaEvent_t a, b; Check(cudaEventCreate(&a)); Check(cudaEventCreate(&b));
        std::vector<float> samples;
        for (int i = 0; i < 8; ++i) {
            Check(cudaEventRecord(a, cudaStreamPerThread)); Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            Check(cudaEventRecord(b, cudaStreamPerThread)); Check(cudaEventSynchronize(b));
            float ms; Check(cudaEventElapsedTime(&ms, a, b)); if (i >= 2) samples.push_back(ms * 10);
        }
        std::sort(samples.begin(), samples.end());
        Check(cudaEventDestroy(a)); Check(cudaEventDestroy(b)); Check(cudaGraphExecDestroy(exec));
        return samples[samples.size()/2];
    };
    printf("VERIFY QKV rows=%d group=%d old_us=%.3f fused_us=%.3f exact_graph=4\n", rows, group, bench(false), bench(true));
    auto before = c.Output(); c.rows.back().verifyOffset = 8;
    Require(!c.Fused(weight, scratch), "invalid verification offset accepted");
    Require(before == c.Output(), "invalid verification offset wrote output");
}
void VerifyProjectionNumerics() {
    Data weight(BFLOAT16), scratch(BFLOAT16);
    Init(weight, {1856, 4096});
    std::mt19937 random(197);
    auto fill = [&](Data &x, float scale) {
        std::uniform_real_distribution<float> uniform(-scale, scale);
        std::vector<__nv_bfloat16> values(x.Count(0));
        for (auto &v : values) v = __float2bfloat16_rn(uniform(random));
        Check(cudaMemcpy(x.cudaData, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
    };
    fill(weight, .1f);
    auto value = [](const std::vector<uint8_t> &bytes, size_t i) {
        unsigned short bits;
        std::memcpy(&bits, bytes.data() + i * 2, 2);
        return __bfloat162float(__ushort_as_bfloat16(bits));
    };
    double maximum = 0;
    for (int count : {2, 3, 7, 8, 9, 13, 16}) {
        Case c(count, true, 64, BFLOAT16, 8);
        fill(c.x, count == 7 ? 16.f : 1.f);
        Require(c.Fused(weight, scratch), "numeric verification QKV rejected");
        cudaGraph_t graph;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Require(c.Fused(weight, scratch), "numeric verification capture rejected");
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&c.graph, graph, 0)); Check(cudaGraphDestroy(graph));
        for (int pass = 0; pass < 4; ++pass) {
            c.Update(pass); c.Reset(); c.Reference(weight); auto reference = c.Output();
            c.Reset(); Check(cudaGraphLaunch(c.graph, cudaStreamPerThread)); auto actual = c.Output();
            for (int row = 0; row < count; ++row) for (int kind = 0; kind < 3; ++kind) {
                const bool active = !(pass == 3 && row % 8 >= 4);
                if (kind && !active) continue;
                const int array = kind ? (pass % 2 ? kind + 2 : kind) : 0;
                const int width = kind == 0 ? 1536 : kind == 1 ? 192 : 128;
                const size_t first = kind ? (size_t(row / 8 * 8) * c.capacity + 128 + row % 8) * width : size_t(row) * width;
                double error = 0, norm = 0;
                for (int i = 0; i < width; ++i) {
                    const float a = value(reference[array], first + i), b = value(actual[array], first + i);
                    Require(std::isfinite(a) && std::isfinite(b), "nonfinite QKV output");
                    const double difference = double(a) - b;
                    error += difference * difference; norm += double(a) * a;
                }
                const double l2 = std::sqrt(error / std::max(norm, 1e-30));
                Require(l2 < .003, "QKV per-row Q/K/V relative L2 exceeded tolerance");
                maximum = std::max(maximum, l2);
            }
        }
    }
    printf("VERIFY QKV random maximum_per_row_L2=%.9g\n", maximum);
}

void VerifyQkvAttention(bool disabled) {
    Data weight(BFLOAT16), scratch(BFLOAT16), sink(FLOAT32);
    Init(weight, {1856, 4096});
    Init(sink, {8});
    std::vector<float> bias(8, -1.f);
    Check(cudaMemcpy(sink.cudaData, bias.data(), 32, cudaMemcpyHostToDevice));
    std::mt19937 rng(7213);
    auto randomFill = [&](Data &x, float scale) {
        std::uniform_real_distribution<float> d(-scale, scale);
        std::vector<__nv_bfloat16> v(x.Count(0));
        for (auto &z : v)
            z = __float2bfloat16_rn(d(rng));
        Check(cudaMemcpy(x.cudaData, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
    };
    auto value = [](const std::vector<uint8_t> &x, size_t i) {
        unsigned short bits;
        std::memcpy(&bits, x.data() + i * 2, 2);
        return float(__ushort_as_bfloat16(bits));
    };
    double maximum[4] = {};
    int checks = 0;
    for (int numeric = 0; numeric < 2; ++numeric) {
        if (numeric)
            randomFill(weight, .1f);
        else
            Fill(weight, 191);
        for (int count : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16})
            for (int group : {3, 8}) {
                Case c(count, true, 64, BFLOAT16, group);
                Data out(BFLOAT16);
                Init(out, {1, count, 1024});
                Check(cudaMemset(out.cudaData, 0x3d, out.GetBytes()));
                if (numeric)
                    randomFill(c.x, count == 7 ? 16.f : 1.f);
                auto fused = [&]() {
                    return FastllmCudaNaiveSwaQkvAttention(c.x, weight, Data(), c.q, c.k, c.v, c.positions,
                                                           c.rows, c.lengths, 8, 1, 192, 128, 64, 10000.f,
                                                           .707f, 128, sink, out);
                };
                std::vector<FastllmNaiveSWARow> rows;
                for (const auto &r : c.rows)
                    rows.push_back(
                        {r.key->cudaData, r.value->cudaData, r.cache, false, true, 0, r.verifyOffset});
                auto reference = [&]() {
                    Require(c.Fused(weight, scratch), "reference QKV rejected");
                    FastllmCudaNaiveBatchSWADecode(c.q, sink, rows, c.lengths, 8, 1, out);
                };
                auto before = c.Output(), beforeOut = std::vector<std::vector<uint8_t>>{Read(out)};
                if (count == 2 && group == 3 && !numeric && !disabled) {
                    cudaGraph_t cold;
                    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                    Require(!fused(), "cold QKV/attention resources prepared during capture");
                    Check(cudaStreamEndCapture(cudaStreamPerThread, &cold));
                    size_t nodes = 0;
                    Check(cudaGraphGetNodes(cold, nullptr, &nodes));
                    Require(nodes == 0, "cold rejection added graph nodes");
                    Check(cudaGraphDestroy(cold));
                }
                bool accepted = fused();
                if (disabled || count < 2 || count > 7) {
                    Require(!accepted, "unsupported QKV/attention accepted");
                    Require(before == c.Output() && beforeOut[0] == Read(out),
                            "QKV/attention rejection wrote output");
                    continue;
                }
                Require(accepted, "supported QKV/attention rejected");
                cudaGraph_t graph;
                Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                Require(fused(), "QKV/attention capture rejected");
                Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                size_t nodes = 0;
                Check(cudaGraphGetNodes(graph, nullptr, &nodes));
                Require(nodes == 1, "QKV/attention is not one node");
                Check(cudaGraphInstantiate(&c.graph, graph, 0));
                Check(cudaGraphDestroy(graph));
                for (int pass = 0; pass < 4; ++pass) {
                    c.Update(pass);
                    c.Reset();
                    reference();
                    auto ref = c.Output();
                    auto refOut = Read(out);
                    c.Reset();
                    Check(cudaGraphLaunch(c.graph, cudaStreamPerThread));
                    auto actual = c.Output();
                    auto actualOut = Read(out);
                    if (!numeric)
                        Require(ref == actual && refOut == actualOut, "dyadic QKV/attention differs");
                    for (int row = 0; row < count; ++row)
                        for (int kind = 0; kind < 4; ++kind) {
                            const bool active = !(pass == 3 && row % group >= (group + 1) / 2);
                            if ((kind == 1 || kind == 2) && !active)
                                continue;
                            const int array = kind == 0 ? 0 : pass % 2 ? kind + 2 : kind;
                            const int width = kind == 0 ? 1536 : kind == 1 ? 192 : kind == 2 ? 128 : 1024;
                            const size_t first =
                                (kind == 0 || kind == 3)
                                    ? size_t(row) * width
                                    : (size_t(row / group * group) * c.capacity + 128 + row % group) * width;
                            const auto &a = kind == 3 ? refOut : ref[array],
                                       &b = kind == 3 ? actualOut : actual[array];
                            double error = 0, norm = 0;
                            for (int j = 0; j < width; ++j) {
                                double x = value(a, first + j), y = value(b, first + j);
                                Require(std::isfinite(x) && std::isfinite(y), "nonfinite QKV/attention");
                                error += (x - y) * (x - y);
                                norm += x * x;
                            }
                            const double l2 = std::sqrt(error / std::max(norm, 1e-30));
                            Require(l2 < .003, "QKV/attention per-row L2 exceeded tolerance");
                            maximum[kind] = std::max(maximum[kind], l2);
                            if (kind == 1 || kind == 2)
                                std::copy_n(ref[array].begin() + first * 2, width * 2,
                                            actual[array].begin() + first * 2);
                        }
                    for (size_t i = 1; i < ref.size(); ++i)
                        Require(ref[i] == actual[i], "untouched cache changed");
                    ++checks;
                }
                before = c.Output();
                auto untouched = Read(out);
                ++out.strides[1];
                Require(!fused(), "noncontiguous attention output accepted");
                --out.strides[1];
                void *save = out.cudaData;
                out.cudaData = c.x.cudaData;
                bool alias = fused();
                out.cudaData = save;
                Require(!alias, "attention/input alias accepted");
                ++sink.strides[0];
                Require(!fused(), "bad sink layout accepted");
                --sink.strides[0];
                c.rows.back().verifyOffset = 8;
                Require(!fused(), "invalid final candidate accepted");
                Require(before == c.Output() && untouched == Read(out), "failed preflight wrote output");
            }
    }
    printf("PASS QKV/attention checks=%d max_Q=%g max_K=%g max_V=%g max_attention=%g disabled=%d\n", checks,
           maximum[0], maximum[1], maximum[2], maximum[3], disabled);
}

void VerifyBatch(int requests, int mode, DataType type) {
    constexpr int heads = 8, dim = 192, vd = 128, width = 1856;
    const int rows = 6, total = requests * rows, capacity = mode == 2 ? 136 : 512;
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), packed(BFLOAT16), index(BFLOAT16);
    Data positions(FLOAT32), live(INT32);
    Init(q, {1, total, heads * dim}); Init(k, {1, total, dim}); Init(v, {1, total, vd});
    Init(packed, {1, total, width}); Init(index, {1, total, 128});
    Init(positions, {1, total}); Init(live, {total});
    Fill(packed, 713); Fill(index, 19);
    std::vector<float> pos(total); std::vector<int> lengths(total);
    std::vector<std::unique_ptr<Data>> keys, values, indices;
    std::vector<FastllmNaiveRopeCacheRow> descriptors;
    auto view = [](Data &x, const Data &src, int offset, int count) {
        auto dims = src.dims; dims[1] = count; x.Resize(dims);
        x.FakeFrom(src, (size_t)offset * src.dims.back() * src.unitSize);
    };
    for (int b = 0; b < requests; ++b) {
        const int past = mode == 2 ? 4090 + b : 111 + b % 9;
        keys.emplace_back(new Data(type)); values.emplace_back(new Data(type));
        Init(*keys.back(), {1, capacity, mode ? dim : dim + 128});
        Init(*values.back(), {1, capacity, vd});
        keys.back()->expansionDims = keys.back()->dims; values.back()->expansionDims = values.back()->dims;
        keys.back()->Resize({1, mode == 2 ? 127 : past, mode ? dim : dim + 128});
        values.back()->Resize({1, mode == 2 ? 127 : past, vd});
        for (int r = 0; r < rows; ++r) {
            int n = b * rows + r; pos[n] = past + r; lengths[n] = past + r + 1;
            indices.emplace_back(new Data()); view(*indices.back(), index, n, 1);
            FastllmNaiveRopeCacheRow item{keys.back().get(), values.back().get(), mode ? nullptr : indices.back().get()};
            if (mode == 1) item.cacheOffset = past + r;
            if (mode == 2) item.verifyOffset = r;
            descriptors.push_back(item);
        }
    }
    Check(cudaMemcpy(positions.cudaData, pos.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(live.cudaData, lengths.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    auto reset = [&]() {
        for (auto *set : {&keys, &values}) for (auto &x : *set)
            Check(cudaMemset(x->cudaData, 0x3d, (size_t)capacity * x->dims.back() * x->unitSize));
        Check(cudaMemset(q.cudaData, 0x3d, q.GetBytes()));
    };
    auto result = [&]() {
        std::vector<std::vector<uint8_t>> out{Read(q)};
        for (auto *set : {&keys, &values}) for (auto &x : *set) {
            out.emplace_back((size_t)capacity * x->dims.back() * x->unitSize);
            Check(cudaMemcpy(out.back().data(), x->cudaData, out.back().size(), cudaMemcpyDeviceToHost));
        }
        return out;
    };
    reset();
    Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, descriptors, live,
        heads, 1, dim, vd, 64, 10000, .707f, mode ? 128 : 0, &packed), "verification batch rejected");
    auto actual = result(); reset();
    for (int b = 0; b < requests; ++b) {
        Data rq, rk, rv, rp, ri, raw, lengthsView;
        view(rq, q, b * rows, rows); view(rk, k, b * rows, rows); view(rv, v, b * rows, rows);
        view(raw, packed, b * rows, rows); if (!mode) view(ri, index, b * rows, rows);
        rp.Resize({1, rows}); rp.FakeFrom(positions, b * rows * sizeof(float));
        if (mode == 2) { lengthsView.Resize({1}); lengthsView.FakeFrom(live, b * rows * sizeof(int)); }
        Require(FastllmCudaNaiveRopeAppendCache(rq, rk, rv, ri, rp, *keys[b], *values[b], lengthsView,
            heads, 1, dim, vd, 64, 10000, .707f, mode ? 128 : 0, &raw, false, mode == 2), "verification reference rejected");
    }
    Require(actual == result(), "batched verification RoPE/cache differs from independent requests");
    reset(); auto before = result();
    descriptors.back().verifyOffset = 8;
    Require(!FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, descriptors, live,
        heads, 1, dim, vd, 64, 10000, .707f, mode ? 128 : 0, &packed), "invalid final descriptor accepted");
    Require(before == result(), "failed preflight changed earlier request");
}

void VerifyPackedSwa(int rows, DataType cacheType, std::vector<int> lengths = {}) {
    if (lengths.empty()) for (int left = rows; left > 0; left -= 8) lengths.push_back(std::min(left, 8));
    std::vector<int> ownerForRow, offsetForRow;
    for (int i = 0; i < int(lengths.size()); ++i) for (int j = 0; j < lengths[i]; ++j) {
        ownerForRow.push_back(i); offsetForRow.push_back(j);
    }
    Require(int(ownerForRow.size()) == rows, "invalid SWA test lengths");
    constexpr int heads = 64, kv = 8, cols = heads * 128;
    Data raw(BFLOAT16), q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), out(BFLOAT16), reference(BFLOAT16);
    Data live(INT32), positions(FLOAT32), table(INT32), arena(INT32), sink(FLOAT32);
    Init(raw, {1, rows, (heads + kv) * 192 + kv * 128});
    std::mt19937 rng(29);
    std::normal_distribution<float> normal;
    std::vector<__nv_bfloat16> input(raw.Count(0));
    for (auto &x : input) x = __float2bfloat16_rn(normal(rng));
    Check(cudaMemcpy(raw.cudaData, input.data(), raw.GetBytes(), cudaMemcpyHostToDevice));
    Init(q, {1, rows, heads * 192}); Init(k, {1, rows, kv * 192}); Init(v, {1, rows, kv * 128});
    const int cacheCount = std::max(2, int(lengths.size()));
    Init(live, {rows}); Init(positions, {1, rows}); Init(table, {int(2 * cacheCount * sizeof(void *) / sizeof(int))}); Init(sink, {heads});
    std::vector<float> biases(heads);
    for (int h = 0; h < heads; ++h) biases[h] = (h % 11 - 5) * .2f;
    // A high sink produces small nonzero activations in the first head.
    if (rows > 16) biases[0] = 24.f;
    Check(cudaMemcpy(sink.cudaData, biases.data(), heads * sizeof(float), cudaMemcpyHostToDevice));
    const int pitch = (rows + 3) / 4 * 4;
    Init(arena, {pitch * cols * 33 / 128});
    auto *values = static_cast<uint8_t *>(arena.cudaData);
    FastllmMxfp8Activation pack{values, values + pitch * cols, rows, cols, pitch};
    std::vector<Data> key(cacheCount, Data(cacheType)), value(cacheCount, Data(cacheType));
    for (int i = 0; i < cacheCount; ++i) {
        Init(key[i], {1, 136, kv * 192}); Init(value[i], {1, 136, kv * 128});
        Fill(key[i], 88 + i); Fill(value[i], 14 + i);
    }
    auto binding = reinterpret_cast<const void *const *>(table.cudaData);
    std::vector<FastllmNaiveRopeCacheRow> caches(rows);
    std::vector<FastllmNaiveSWARow> bindings(rows);
    for (int i = 0; i < rows; ++i) {
        const int owner = ownerForRow[i];
        const auto *table = binding + 2 * owner;
        caches[i] = {&key[owner], &value[owner], nullptr, table};
        bindings[i] = {key[owner].cudaData, value[owner].cudaData, table, cacheType == FP8_E4M3};
        caches[i].verifyOffset = bindings[i].verifyOffset = offsetForRow[i];
    }
    auto launch = [&] {
        return FastllmCudaTryPackedSwaVerify(raw, positions, live, caches.data(), sink,
            heads, kv, 10000, .707f, pack, out, true, lengths.data(), lengths.size());
    };
    Check(cudaMemset(live.cudaData, 0, live.GetBytes())); Check(cudaMemset(table.cudaData, 0, table.GetBytes()));
    Require(launch(), "packed SWA warmup declined");
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(launch(), "packed SWA graph capture declined");
    cudaGraph_t graph; cudaGraphExec_t exec;
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Check(cudaGraphInstantiate(&exec, graph, 0));
    auto resetCaches = [&] {
        for (int i = 0; i < cacheCount; ++i) { Fill(key[i], 83 + i); Fill(value[i], 57 + i); }
    };
    double worst = 0;
    bool hasSmallGroup = false;
    for (int past : {0, 126, 4096, 16384, -1}) {
        const int owner = past == 0 || past == 16384 ? 0 : 1;
        std::vector<float> pos(rows); std::vector<int> lengths(rows);
        for (int i = 0; i < rows; ++i) {
            pos[i] = past < 0 ? -1 : past + offsetForRow[i] + ownerForRow[i] * 83;
            // Replay with an inactive owner and a hole before a live candidate.
            const bool masked = past == 126 && (rows > 8 ? (offsetForRow[i] == 4 || ownerForRow[i] == 1) : i + 1 == rows);
            lengths[i] = past < 0 || masked ? 0 : int(pos[i]) + 1;
            // Exercise a replay whose adjacent live candidates no longer share a prefix.
            if (rows > 16 && past == 4096 && offsetForRow[i] == 3) lengths[i] += 2;
        }
        std::vector<void *> ptrs(2 * cacheCount);
        for (int j = 0; past >= 0 && j < cacheCount; ++j) {
            const int slot = (owner + j) % cacheCount;
            ptrs[2 * j] = key[slot].cudaData;
            ptrs[2 * j + 1] = value[slot].cudaData;
        }
        Check(cudaMemcpy(table.cudaData, ptrs.data(), ptrs.size() * sizeof(void *), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(positions.cudaData, pos.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(live.cudaData, lengths.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
        resetCaches();
        Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, caches, live, heads,
            kv, 192, 128, 64, 10000, .707f, 128, &raw), "packed SWA reference append declined");
        FastllmCudaNaiveBatchSWADecode(q, sink, bindings, live, heads, kv, reference);
        auto ref = Read(reference);
        std::vector<decltype(ref)> savedKeys, savedValues;
        for (int j = 0; j < cacheCount; ++j) {
            savedKeys.push_back(Read(key[j]));
            savedValues.push_back(Read(value[j]));
        }
        resetCaches();
        Check(cudaMemset(arena.cudaData, 0xcd, arena.GetBytes()));
        Check(cudaGraphLaunch(exec, cudaStreamPerThread)); Check(cudaStreamSynchronize(cudaStreamPerThread));
        for (int j = 0; j < cacheCount; ++j)
            Require(Read(key[j]) == savedKeys[j] && Read(value[j]) == savedValues[j], "packed SWA candidate KV differs");
        const auto result = Read(out), encoded = Read(arena);
        if (FastllmCudaRuntimeArch() == 120 || rows >= 32)
            if (result != ref) {
                const size_t byte = std::mismatch(result.begin(), result.end(), ref.begin()).first - result.begin();
                throw std::runtime_error("packed SWA rounding mismatch at total rows " + std::to_string(rows) +
                    ", past " + std::to_string(past) + ", row " + std::to_string(byte / (cols * 2)) +
                    ", cache dtype " + std::to_string(int(cacheType)));
            }
        const auto *actual = reinterpret_cast<const __nv_bfloat16 *>(result.data());
        const auto *expected = reinterpret_cast<const __nv_bfloat16 *>(ref.data());
        for (int row = 0; row < rows; ++row) {
            double error = 0., norm = 0.;
            for (int col = 0; col < cols; col += 32) {
                const int offset = row * cols + col;
                float mx = 0.f;
                for (int j = 0; j < 32; ++j) {
                    float a = float(actual[offset+j]), b = float(expected[offset+j]);
                    mx = std::max(mx, std::abs(a)); error += double(a-b)*(a-b); norm += double(b)*b;
                }
                if (rows > 16) {
                    hasSmallGroup |= mx > 0.f && mx < 1.e-4f;
                    mx = std::max(mx, 1.e-4f);
                }
                int exp = mx > 0 ? std::clamp(int(std::ceil(std::log2(mx / 448.f))) + 127, 1, 254) : 1;
                Require(encoded[pitch*cols + (col/128*pitch+row)*4 + col/32%4] == exp, "packed SWA O scale differs");
                for (int j = 0; j < 32; ++j)
                    Require(encoded[offset+j] == __nv_fp8_e4m3(std::ldexp(float(actual[offset+j]), 127-exp)).__x,
                            "packed SWA O value differs");
            }
            const double l2 = std::sqrt(error / std::max(norm, 1.e-30));
            Require(l2 < .005, "packed SWA output exceeds relative L2 tolerance");
            worst = std::max(worst, l2);
        }
    }
    auto saved = Read(out);
    Require(!FastllmCudaTryPackedSwaVerify(raw, positions, live, caches.front(), sink,
        32, kv, 10000, .707f, pack, out) && Read(out) == saved, "unsupported SWA modified output");
    Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
    if (rows > 16) Require(hasSmallGroup, "SWA Block32 scale floor was not exercised");
    printf("PACKED_SWA_PASS rows=%d max_row_l2=%.8g KV/pack exact, rebound graph\n", rows, worst);
}

}
#ifdef FASTLLM_ENABLE_NAIVE_HOPPER_VERIFY
// A single nonzero FP8 weight per output selects a known input lane and scale.
static void VerifyPackedProjectionRows(int inner, int channels) {
    const auto *regular = FastllmCudaGetMxfp8ProjectionBackend();
    Require(regular != nullptr, "missing SM90 projection backend");
    Data w(FP8_E4M3, {channels, inner});
    w.blockM = 32;
    w.blockK = 1;
    w.Allocate();
    std::memset(w.cpuData, 0, w.GetBytes());
    w.scales.assign(size_t(channels) * inner / 32, 1.f);
    for (int c = 0; c < channels; ++c)
        w.cpuData[size_t(c) * inner + (c * 157 % inner)] = 0x38;
    w.ToDevice(CUDA, {0}, true);
    regular->prepareWeight(w);
    for (int rows : {2, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65, 97, 127, 128}) {
        const auto *backend = inner == 8192 && rows > 16 ? FastllmCudaSm90SwaOutputProjection() : regular;
        const int pitch = (rows + 3) / 4 * 4;
        Data arena(INT32), input(BFLOAT16), output;
        Init(arena, {pitch * (inner + inner / 32) / 4});
        Init(input, {1, rows, inner});
        auto view = backend->Borrow(w, rows, arena);
        Require(view.Matches(rows, inner), "SM90 projection scratch admission");
        std::vector<unsigned char> bytes(pitch * inner, 0), scales(pitch * inner / 32);
        auto exponent = [](int r, int k) { return 125 + ((r + k / 32) % 5); };
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < inner; ++k)
                bytes[r * inner + k] = (k + r) % 3 ? 0x38 : 0xb8;
            for (int k = 0; k < inner; k += 32)
                scales[(k / 128 * pitch + r) * 4 + (k / 32) % 4] = exponent(r, k);
        }
        Check(cudaMemcpy(view.values, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(view.scales, scales.data(), scales.size(), cudaMemcpyHostToDevice));
        auto project = [&] { Require(backend->TryProject(view, input, w, Data(), output),
                                     "SM90 packed projection declined"); };
        project();
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        project();
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, 0));
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        const auto result = Read(output);
        const auto *actual = reinterpret_cast<const __nv_bfloat16 *>(result.data());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < channels; ++c) {
                const int k = c * 157 % inner;
                const float value = std::ldexp((k + r) % 3 ? 1.f : -1.f, exponent(r, k) - 127);
                Require(float(actual[r * channels + c]) == value, "SM90 packed projection value mismatch");
            }
        Check(cudaGraphExecDestroy(exec));
        Check(cudaGraphDestroy(graph));
    }
    std::printf("SM90 packed projection PASS K=%d N=%d\n", inner, channels);
}
#endif

int main(int argc,char**argv) {
    try {
        const bool attention=argc>1 && std::strncmp(argv[1], "--attention", 11)==0;
        const bool disabled=attention && std::strcmp(argv[1], "--attention")!=0;
        if (disabled) {
            const char *name = std::strcmp(argv[1], "--attention-core-disabled") == 0
                ? "FASTLLM_NAIVE_SWA_CORE" : "FASTLLM_NAIVE_SWA_QKV_FUSION";
#ifdef _WIN32
            _putenv_s(name, "0");
#else
            setenv(name, "0", 1);
#endif
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        Check(cudaSetDevice(0));
        if (argc > 1 && std::strcmp(argv[1], "--packed-swa") == 0) {
            const int arch = FastllmCudaRuntimeArch();
            if (CUDART_VERSION < 12080 || (arch != 90 && arch != 100 && arch != 103 && arch != 120)) return 77;
#ifdef FASTLLM_ENABLE_NAIVE_HOPPER_VERIFY
            if (arch == 90) {
                VerifyPackedProjectionRows(4096, 14848);
                VerifyPackedProjectionRows(8192, 4096);
            }
#endif
            if (arch == 90) for (int rows : {9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128}) {
                VerifyPackedSwa(rows, FP8_E4M3);
                VerifyPackedSwa(rows, BFLOAT16);
            }
            if (arch == 90) VerifyPackedSwa(21, FP8_E4M3, {1, 7, 3, 2, 8});
            for (int rows = 2; rows <= 8; ++rows) {
                VerifyPackedSwa(rows, BFLOAT16);
                if (arch == 120 || arch == 90) VerifyPackedSwa(rows, FP8_E4M3);
            }
            return 0;
        }
        if(attention){if(FastllmCudaRuntimeArch()!=103)return 77;VerifyQkvAttention(disabled);return 0;}
        for (auto type : {BFLOAT16, FP8_E4M3})
            for (int mode : {0, 1, 2}) for (int batch : {2, 8, 40}) VerifyBatch(batch, mode, type);
        const int arch = FastllmCudaRuntimeArch();
        if (arch != 90 && arch != 103) { puts("SKIP: requires SM90 or SM103 fusion build"); return 77; }
        Data weight(BFLOAT16), scratch(BFLOAT16);
        Init(weight, {1856, 4096}); Fill(weight, 191);
        if (arch == 103) for (int rows : {1,2,3,4,5,6,7,8,9,13,16,17,24,32,64}) {
            VerifyProjection(weight, rows, std::min(rows, 8));
            if (rows > 3) VerifyProjection(weight, rows, 3);
        }
        if (arch == 103) VerifyProjectionNumerics();
        std::vector<std::unique_ptr<Case>> cases;
        const int batches[] = {256, 129, 128, 65, 64, 33, 32, 17, 16, 8, 7, 6, 5, 4, 3, 2, 1};
        void *address = nullptr;
        for (DataType type : {BFLOAT16, FP8_E4M3})
        for (int batch : batches) for (bool ring : {false, true}) {
            cases.emplace_back(new Case(batch, ring, 64, type));
            auto &c = *cases.back();
            if (arch == 103 && (batch != 1 || !ring || type != BFLOAT16)) {
                auto before = c.Output();
                Require(!c.Fused(weight, scratch), "unsupported SM103 shape accepted");
                Require(before == c.Output(), "unsupported shape changed output");
                cases.pop_back();
                continue;
            }
            Require(c.Fused(weight, scratch), "supported fusion rejected");
            if (!address) address = scratch.cudaData;
            Require(scratch.cudaData == address, "smaller batch relocated graph scratch");
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Require(c.Fused(weight, scratch), "capture rejected");
            cudaGraph_t graph;
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&c.graph, graph, 0));
            Check(cudaGraphDestroy(graph));
        }
        int checks = 0;
        for (auto &fixture : cases) {
            auto &c = *fixture;
            for (int replay = 0; replay < 4; ++replay) {
                c.Update(replay); c.Reset();
                Check(cudaGraphLaunch(c.graph, cudaStreamPerThread));
                auto actual = c.Output();
                c.Reset(); c.Reference(weight);
                Require(actual == c.Output(), "projection, rotary, page binding or untouched cache differs");
                Data bias(BFLOAT16, {1856});
                Require(!c.Fused(weight, scratch, bias), "bias incorrectly accepted");
                ++c.x.strides[1];
                Require(!c.Fused(weight, scratch), "bad stride incorrectly accepted");
                --c.x.strides[1];
                Require(actual == c.Output(), "rejected fusion modified outputs");
                if (arch == 103) {
                    // An aligned output view can still alias a read-only input.
                    void *original = c.q.cudaData;
                    c.q.cudaData = c.x.cudaData;
                    bool accepted = c.Fused(weight, scratch);
                    c.q.cudaData = original;
                    Require(!accepted, "aliased Q/input accepted");
                    Require(actual == c.Output(), "alias rejection changed outputs");
                }
                ++checks;
            }
        }
        printf("PASS: %d exact graph replays, shared workspace and fallback checks\n", checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
