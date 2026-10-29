#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>
using namespace fastllm;
// Exercise the private implementation in this standalone test translation unit.
// Link the shared fastllm library: no production test exports or runtime switches.
#include "../../src/models/deepseekv41_dspark.cpp"

static bool parallelLaunch = true;
static bool RunAttention(const std::vector<int> *devices, Data **inputs, int heads, int groups, int oRank,
                           std::shared_ptr<void> *weightCache, std::shared_ptr<void> *cache,
                           std::shared_ptr<void> *graphCache, Data *output, bool graph) {
    const DsRopeParams rope{64, 10000.0f, 0, 1.0f, 32, 1};
    auto body = [&]() {
        AssertInFastLLM(DsParallelAttention(*devices, *inputs[0], *inputs[1], *inputs[2], *inputs[3],
                                            *inputs[4], *inputs[5], *inputs[6], *inputs[7], *inputs[8], heads,
                                            groups, oRank, rope, 100, 1.0f / std::sqrt(512.0f), *weightCache,
                                            *cache, *output),
                        "TP regression path rejected.");
    };
    if (!graph) {
        body();
        return true;
    }
    if (!*graphCache) {
        auto state = std::make_shared<DeepSeekV41DsparkWorkspace>();
        state->devices = *devices;
        *graphCache = state;
    }
    auto &state = *std::static_pointer_cast<DeepSeekV41DsparkWorkspace>(*graphCache);
    state.parallelAttention = parallelLaunch;
    return DsRunDraftGraph(state, body);
}
static const Data *InspectWorkspace(std::shared_ptr<void> *cache, int rank, int field) {
    auto &w = *std::static_pointer_cast<DsAttentionTpWorkspace>(*cache);
    auto &r = *w.ranks.at(rank);
    if (field == 0) return &r.query;
    if (field == 1) return &r.attention;
    if (field == 2) return &r.woA;
    if (field == 3) return w.weights->wq.multiDeviceDatas.at(w.devices.at(rank));
    if (field == 4) return w.weights->woA.multiDeviceDatas.at(w.devices.at(rank));
    if (field == 5) return w.weights->woB.multiDeviceDatas.at(w.devices.at(rank));
    return nullptr;
}
#define REQUIRE(x)                                                                                           \
    do {                                                                                                     \
        if (!(x)) {                                                                                          \
            fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x);                                      \
            std::exit(3);                                                                                    \
        }                                                                                                    \
    } while (0)
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e));
        std::exit(4);
    }
}
static uint16_t B(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}
static float F(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float Round(double x) {
    return F(B(float(x)));
}
static double FP8(uint8_t b) {
    int e = (b & 127) >> 3, m = b & 7;
    double v = e ? std::ldexp(8.0 + m, e - 10) : std::ldexp(double(m), -9);
    return b & 128 ? -v : v;
}
static void Alloc(Data &d, DataType type, const std::vector<int> &dims) {
    d.dataType = type;
    d.UpdateUnitSize();
    d.Resize(dims);
    d.ToDevice(DataDevice::CUDA, {0}, false);
    d.Allocate();
}
static std::vector<uint8_t> Bytes(const Data &d) {
    std::vector<uint8_t> x(d.GetBytes());
    Cuda(cudaMemcpy(x.data(), d.cudaData, x.size(), cudaMemcpyDeviceToHost));
    return x;
}
static std::vector<float> Values(const Data &d) {
    auto b = Bytes(d);
    std::vector<float> v(d.Count(0));
    if (d.dataType == BFLOAT16) {
        for (size_t i = 0; i < v.size(); ++i) {
            uint16_t q;
            memcpy(&q, b.data() + i * 2, 2);
            v[i] = F(q);
        }
    } else {
        REQUIRE(d.dataType == FLOAT32);
        memcpy(v.data(), b.data(), b.size());
    }
    return v;
}
static void Fill(Data &d, int seed, float amplitude) {
    std::vector<uint16_t> v(d.Count(0));
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = B(amplitude * std::sin(float((i * 29 + seed * 17) % 997)));
    Cuda(cudaMemcpy(d.cudaData, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
}
static void Weight(Data &w, int n, int k, int seed) {
    Alloc(w, FP8_E4M3, {n, k});
    w.blockM = 32;
    w.blockK = 32;
    std::vector<uint8_t> v(size_t(n) * k);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = uint8_t(((i * 37 + seed * 11) % 105) | (((i + seed) % 3 == 0) ? 128 : 0));
    w.scales.resize(size_t(n / 32) * (k / 32));
    for (size_t i = 0; i < w.scales.size(); ++i)
        w.scales[i] = std::ldexp(1.0f, int((i * 7 + seed) % 5) - 12);
    Cuda(cudaMemcpy(w.cudaData, v.data(), v.size(), cudaMemcpyHostToDevice));
}
static std::vector<float> Gather(std::shared_ptr<void> &cache, const std::vector<int> &devices, int field,
                                 int rows, int width) {
    std::vector<float> out(size_t(rows) * width);
    const int local = width / devices.size();
    for (size_t r = 0; r < devices.size(); ++r) {
        Cuda(cudaSetDevice(devices[r]));
        auto d = InspectWorkspace(&cache, r, field);
        auto v = Values(*d);
        REQUIRE(v.size() == size_t(rows) * local);
        for (int m = 0; m < rows; ++m)
            std::copy_n(v.data() + m * local, local, out.data() + m * width + r * local);
    }
    Cuda(cudaSetDevice(devices[0]));
    return out;
}
static double Compare(const std::vector<float> &a, const std::vector<float> &b) {
    REQUIRE(a.size() == b.size());
    double err = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::isfinite(a[i]));
        err += double(a[i] - b[i]) * (a[i] - b[i]);
        den += double(b[i]) * b[i];
    }
    double e = std::sqrt(err / std::max(den, 1e-30));
    REQUIRE(e < 0.006);
    return e;
}
static void CheckShard(std::shared_ptr<void> &cache, const std::vector<int> &devices, const Data &source,
                       int field, int axis) {
    Cuda(cudaSetDevice(0));
    auto whole = Bytes(source);
    int n = source.dims[0], k = source.dims[1], unit = source.dataType == BFLOAT16 ? 2 : 1;
    for (size_t r = 0; r < devices.size(); ++r) {
        Cuda(cudaSetDevice(devices[r]));
        const Data &d = *InspectWorkspace(&cache, r, field);
        auto b = Bytes(d);
        int nn = axis == 0 ? n / devices.size() : n, kk = axis == 1 ? k / devices.size() : k;
        REQUIRE(d.dims == std::vector<int>({nn, kk}));
        for (int i = 0; i < nn; ++i) {
            int si = i + (axis == 0 ? r * nn : 0), sk = axis == 1 ? r * kk : 0;
            if (memcmp(b.data() + size_t(i) * kk * unit, whole.data() + (size_t(si) * k + sk) * unit,
                       size_t(kk) * unit) != 0) {
                fprintf(stderr, "Shard mismatch field=%d rank=%zu row=%d n=%d k=%d\n", field, r, i, n, k);
                for (int z = 0; z < kk * unit; ++z)
                    if (b[size_t(i) * kk * unit + z] != whole[(size_t(si) * k + sk) * unit + z]) {
                        fprintf(stderr, "byte=%d got=%d expected=%d\n", z, b[size_t(i) * kk * unit + z],
                                whole[(size_t(si) * k + sk) * unit + z]);
                        break;
                    }
                std::exit(3);
            }
        }
        if (source.dataType == FP8_E4M3) {
            REQUIRE(d.blockM == 32 && d.blockK == 32);
            for (int i = 0; i < nn / 32; ++i)
                for (int j = 0; j < kk / 32; ++j) {
                    int si = i + (axis == 0 ? r * nn / 32 : 0), sj = j + (axis == 1 ? r * kk / 32 : 0);
                    REQUIRE(d.scales[size_t(i) * (kk / 32) + j] == source.scales[size_t(si) * (k / 32) + sj]);
                }
        }
    }
    Cuda(cudaSetDevice(0));
}
static void CheckWoA(const std::vector<float> &a, const std::vector<float> &weights,
                     const std::vector<float> &got, int rows, int groups, int rank, int inner) {
    for (int m = 0; m < rows; ++m)
        for (int g = 0; g < groups; ++g)
            for (int o = 0; o < rank; ++o) {
                double s = 0, l1 = 0;
                for (int k = 0; k < inner; ++k) {
                    double x = double(a[(size_t(m) * groups + g) * inner + k]) *
                               weights[(size_t(g) * rank + o) * inner + k];
                    s += x;
                    l1 += std::abs(x);
                }
                double e = 4.0 * inner * std::numeric_limits<float>::epsilon() * l1;
                float v = got[(size_t(m) * groups + g) * rank + o];
                REQUIRE(v >= Round(s - e) && v <= Round(s + e));
            }
}
static void CheckWoB(const std::vector<float> &a, const Data &w, const std::vector<float> &got, int rows,
                     int ranks) {
    auto raw = Bytes(w);
    const int n = w.dims[0], k = w.dims[1], local = k / ranks;
    for (int m = 0; m < rows; ++m)
        for (int j = 0; j < n; ++j) {
            double lower = 0, upper = 0, mag = 0;
            for (int r = 0; r < ranks; ++r) {
                double s = 0, l1 = 0;
                for (int z = r * local; z < (r + 1) * local; ++z) {
                    double weight =
                        FP8(raw[size_t(j) * k + z]) * w.scales[size_t(j / 32) * (k / 32) + z / 32];
                    double x = double(a[size_t(m) * k + z]) * weight;
                    s += x;
                    l1 += std::abs(x);
                }
                double e = 4.0 * local * std::numeric_limits<float>::epsilon() * l1;
                lower += Round(s - e);
                upper += Round(s + e);
                mag += std::max(std::abs(Round(s - e)), std::abs(Round(s + e)));
            }
            double e = 2.0 * ranks * std::numeric_limits<float>::epsilon() * mag;
            float v = got[size_t(m) * n + j];
            if (!(v >= Round(lower - e) && v <= Round(upper + e))) {
                fprintf(stderr, "WoB row=%d col=%d got=%g interval=[%g,%g]\n", m, j, v, Round(lower - e),
                        Round(upper + e));
                std::exit(5);
            }
        }
}
int main(int, char **) {
    int available = 0;
    Cuda(cudaGetDeviceCount(&available));
    if (available < 2) return 77;
    double worstQ = 0, worstAttention = 0, worstWoA = 0;
    int cases = 0;
    for (int full : {0, 1})
        for (int count : (full ? std::vector<int>{8} : std::vector<int>{2, 4, 8}))
            for (int rows : (full ? std::vector<int>{5} : std::vector<int>{2, 5, 8}))
                for (int window : (full ? std::vector<int>{512} : std::vector<int>{17, 512})) {
                    if (count > available) continue;
                    Cuda(cudaSetDevice(0));
                    std::vector<int> devices;
                    for (int d = 0; d < count; ++d)
                        devices.push_back(d);
                    const int heads = 64, groups = 8, rank = full ? 1024 : 128, inner = full ? 1280 : 128,
                              out = full ? 5120 : 256;
                    Data input, kv, ring, idx, pos, wq, wa, wb, sink, result;
                    Alloc(input, BFLOAT16, {1, rows, inner});
                    Alloc(kv, BFLOAT16, {1, rows, 512});
                    Alloc(ring, BFLOAT16, {1, window, 512});
                    Alloc(idx, INT32, {1, rows, window + rows - 1});
                    Alloc(pos, INT32, {1});
                    Weight(wq, heads * 512, inner, 3);
                    Alloc(wa, BFLOAT16, {groups * rank, heads / groups * 512});
                    Fill(wa, 4, 0.02f);
                    Weight(wb, out, groups * rank, 7);
                    Alloc(sink, FLOAT32, {heads});
                    Cuda(cudaMemset(sink.cudaData, 0, sink.GetBytes()));
                    std::vector<int> indices;
                    for (int m = 0; m < rows; ++m) {
                        for (int j = 0; j < window; ++j)
                            indices.push_back(j);
                        for (int j = 0; j < rows; ++j)
                            if (j != m) indices.push_back(window + j);
                    }
                    Cuda(cudaMemcpy(idx.cudaData, indices.data(), idx.GetBytes(), cudaMemcpyHostToDevice));
                    Data *inputs[] = {&input, &kv, &ring, &idx, &pos, &wq, &wa, &wb, &sink};
                    std::shared_ptr<void> weightCache, cache, graphCache;
                    const DsRopeParams rope{64, 10000.0f, 0, 1.0f, 32, 1};
                    auto rejects = [&]() {
                        REQUIRE(!DsParallelAttention(devices, input, kv, ring, idx, pos, wq, wa, wb, sink,
                                                     heads, groups, rank, rope, 100, 1.0f / std::sqrt(512.0f),
                                                     weightCache, cache, result));
                        REQUIRE(!weightCache && !cache && !result.cudaData);
                    };
                    input.dataType = FLOAT16;
                    rejects();
                    input.dataType = BFLOAT16;
                    const auto stride = kv.strides.back();
                    kv.strides.back() = 2;
                    rejects();
                    kv.strides.back() = stride;
                    auto raw = wq.cudaData;
                    wq.cudaData = nullptr;
                    rejects();
                    wq.cudaData = raw;
                    const auto shape = ring.dims;
                    ring.Resize({1, 1025, 512});
                    rejects();
                    ring.Resize(shape);
                    for (int iteration = 0; iteration < 4; ++iteration) {
                        if (iteration == 3) {
                            graphCache.reset();
                            cache.reset();
                        }
                        Cuda(cudaSetDevice(0));
                        Fill(input, iteration + 1, 0.1f);
                        Fill(kv, iteration + 2, 0.2f);
                        Fill(ring, iteration + 3, 0.2f);
                        int position = 100 + iteration * 11;
                        Cuda(cudaMemcpy(pos.cudaData, &position, 4, cudaMemcpyHostToDevice));
                        auto kvBefore = Bytes(kv), ringBefore = Bytes(ring);
                        parallelLaunch = iteration % 2 != 0;
                        REQUIRE(RunAttention(&devices, inputs, heads, groups, rank, &weightCache, &cache,
                                               &graphCache, &result, iteration > 0));
                        for (int d : devices) {
                            Cuda(cudaSetDevice(d));
                            Cuda(cudaDeviceSynchronize());
                        }
                        Cuda(cudaSetDevice(0));
                        REQUIRE(Bytes(kv) == kvBefore && Bytes(ring) == ringBefore);
                        if (iteration == 0) {
                            CheckShard(cache, devices, wq, 3, 0);
                            CheckShard(cache, devices, wa, 4, 0);
                            CheckShard(cache, devices, wb, 5, 1);
                        }
                        auto tq = Gather(cache, devices, 0, rows, heads * 512),
                             ta = Gather(cache, devices, 1, rows, heads * 512),
                             tw = Gather(cache, devices, 2, rows, groups * rank);
                        CheckWoA(ta, Values(wa), tw, rows, groups, rank, heads / groups * 512);
                        CheckWoB(tw, wb, Values(result), rows, count);
                        Data q, refKv, table, attention, woa;
                        Alloc(q, BFLOAT16, {1, rows, heads, 512});
                        refKv.CopyFrom(kv);
                        REQUIRE(FastllmCudaTryMarlinBFloat16MatMulFP8E4M3(input.cudaData, wq, q.cudaData,
                                                                          rows, inner, heads * 512));
                        REQUIRE(FastllmCudaDeepSeekV41DraftRotaryTable(q, refKv, ring, 64, 10000, 100, 0, 1,
                                                                       32, 1, (int *)pos.cudaData, table));
                        REQUIRE(FastllmCudaDeepSeekV41SparseAttention(
                            q, refKv, nullptr, &table, &idx, sink, 1, 0, 1 / std::sqrt(512.0f), attention));
                        REQUIRE(FastllmCudaDeepSeekV41RotaryQuantPosition(attention, 64, 10000, 100, 1, true,
                                                                          0, 1, 32, 1, 0, 512, 32,
                                                                          (int *)pos.cudaData));
                        REQUIRE(FastllmCudaDeepSeekV4WoA(attention, wa, groups, rank, woa));
                        Cuda(cudaDeviceSynchronize());
                        worstQ = std::max(worstQ, Compare(tq, Values(q)));
                        worstAttention = std::max(worstAttention, Compare(ta, Values(attention)));
                        worstWoA = std::max(worstWoA, Compare(tw, Values(woa)));
                    }
                    graphCache.reset();
                    cache.reset();
                    ++cases;
                    printf("PASS ranks=%d rows=%d window=%d full=%d\n", count, rows, window, full);
                    fflush(stdout);
                }
    printf("PASS %d cases x4 changing inputs with shared-weight cache reuse; shard bytes/scales exact; FP64 "
           "WoA/WoB bounds; max relative L2 Q=%g attention=%g WoA=%g\n",
           cases, worstQ, worstAttention, worstWoA);
}
