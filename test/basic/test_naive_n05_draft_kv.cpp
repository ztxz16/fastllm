#include "models/naive_n05_flash.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cmath>
#include <climits>
#include <stdexcept>
using namespace fastllm;
static int device = 0;
void Check(bool ok, const char *s) {
    if (!ok)
        throw std::runtime_error(s);
}
Data BF(std::vector<int> dims, int seed) {
    Data x(BFLOAT16, dims);
    x.Allocate();
    for (int i = 0; i < x.Count(0); i++)
        ((uint16_t *)x.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + seed) * .137f));
    x.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    return x;
}
std::vector<uint16_t> Bits(const Data &x) {
    Data host(x);
    host.ToDevice(DataDevice::CPU);
    size_t count = 1;
    for (int dim : x.dims)
        count *= dim;
    return {(uint16_t *)host.cpuData, (uint16_t *)host.cpuData + count};
}
void Append(Data &cache, Data &input, int reserve) {
    cache.dataType = BFLOAT16;
    cache.UpdateUnitSize();
    cache.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    if (cache.expansionDims.empty())
        cache.Expansion({1, reserve, input.dims[2]});
    CatDirect(cache, input, 1);
    cache.isKVCache = true;
}
class ProjectionFixture : public NaiveN05FlashModel {
  public:
    using NaiveN05FlashModel::DraftContext;
    ProjectionFixture(int seed) {
        draftLayers = 3;
        draftKvHeads = 2;
        draftHeadDim = 64;
        draftWindow = 17;
        draftBlock = 7;
        for (int i = 0; i < draftLayers; ++i) {
            auto name = "dspark.layers." + std::to_string(i) + ".self_attn.";
            for (auto suffix : {"k_proj.weight", "v_proj.weight"}) {
                Data x = BF({128, 256}, seed + i + (suffix[0] == 'v' ? 37 : 0));
                weight[name + suffix].CopyFrom(x);
            }
            Data norm(FLOAT32, {64}, std::vector<float>(64, 1.0f));
            norm.ToDevice(DataDevice::CUDA, std::vector<int>{device});
            weight[name + "k_norm.weight"].CopyFrom(norm);
        }
    }
    bool Fused(Data &x, int start, DraftContext &ctx) { return AppendDraftContextFused(x, start, ctx); }
    void Reference(Data &x, int start, DraftContext &ctx) {
        int saved = draftBlock;
        // Force the original path even for a single input row.
        draftBlock = -1;
        AppendDraftContext(x, start, ctx);
        draftBlock = saved;
        Check(!ctx.projectedKV, "reference unexpectedly used the fused path");
    }
    bool Packed() const { return bool(draftKVProjection); }
    Data &FirstNorm() { return weight["dspark.layers.0.self_attn.k_norm.weight"]; }
    std::shared_ptr<DraftContext> Reuse(std::shared_ptr<DraftContext> ctx, bool history = false) {
        SetSaveHistoryChat(history);
        ResponseContext response;
        draftContexts[&response.pastKeyValues] = std::move(ctx);
        OnResponseContextRemoved(&response);
        Check(draftContexts.empty(), "removed request remained registered");
        return CreateDraftContext();
    }
};
static double Relative(const Data &a, const Data &b) {
    auto ah = Bits(a), bh = Bits(b);
    Check(ah.size() == bh.size(), "relative error shape mismatch");
    double error = 0, power = 0;
    for (size_t i = 0; i < ah.size(); ++i) {
        double x = BFloat16BitsToFloat32(ah[i]), y = BFloat16BitsToFloat32(bh[i]);
        error += (x - y) * (x - y);
        power += y * y;
    }
    return std::sqrt(error / std::max(power, 1e-30));
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        ApplyDeviceMap({{"cuda:0", 1}}, 1, 1);
        int checks = 0;
        for (int dim : {64, 128, 256})
            for (int rows : {1, 2, 3, 4, 5, 6, 7, 8})
                for (int start : {0, 80, 1019, 1023, 20000}) {
                    int layers = 5, heads = 4, width = heads * dim, old = std::min(start, 1023);
                    Data raw = BF({1, rows, layers * 2 * width}, 71), norm(FLOAT32, {layers, dim});
                    norm.Allocate();
                    for (int i = 0; i < layers * dim; i++)
                        ((float *)norm.cpuData)[i] = .5f + std::sin(i * .07f);
                    norm.ToDevice(DataDevice::CUDA, std::vector<int>{device});
                    std::vector<std::pair<Data, Data>> a(layers), b(layers);
                    for (int i = 0; i < layers; i++)
                        if (old) {
                            Data k = BF({1, old, width}, i + 81), v = BF({1, old, width}, i + 89);
                            Append(a[i].first, k, 1152);
                            Append(b[i].first, k, 1152);
                            Append(a[i].second, v, 1152);
                            Append(b[i].second, v, 1152);
                        }
                    Check(FastllmCudaNaiveDraftKV(raw, norm, start, a, heads, dim, 1024, 1031, 1e-5f, 10000),
                          "valid fusion rejected");
                    std::vector<float> positions(rows);
                    for (int j = 0; j < rows; j++)
                        positions[j] = start + j;
                    Data pos(FLOAT32, {1, rows}, positions);
                    pos.ToDevice(DataDevice::CUDA, std::vector<int>{device});
                    for (int i = 0; i < layers; i++) {
                        Data k, v, w;
                        Split(raw, 2, i * 2 * width, (i * 2 + 1) * width, k);
                        Split(raw, 2, (i * 2 + 1) * width, (i * 2 + 2) * width, v);
                        Split(norm, 0, i, i + 1, w);
                        w.Reshape({dim});
                        k.Reshape({1, rows * heads, dim});
                        KimiK3RMSNorm(k, w, 1e-5f, k);
                        k.Reshape({1, rows, width});
                        FastllmCudaNaiveRope(k, pos, heads, dim, dim, 10000);
                        Append(b[i].first, k, 1152);
                        Append(b[i].second, v, 1152);
                        FastllmCudaNaiveTrimCache(b[i].first, b[i].second, 1023);
                        Check(a[i].first.dims == b[i].first.dims, "shape mismatch");
                        Check(Bits(a[i].first) == Bits(b[i].first), "K bits differ");
                        Check(Bits(a[i].second) == Bits(b[i].second), "V bits differ");
                        checks += 3;
                    }
                }
        // Scalar tails, multiple launch groups, and a nonzero device.
        for (device = 0; device < std::min(devices, 2); ++device) {
            ApplyDeviceMap({{"cuda:" + std::to_string(device), 1}}, 1, 1);
            for (int layers : {1, 17})
                for (int dim : {2, 6, 96}) {
                    int rows = 3, heads = 1, width = dim;
                    Data raw = BF({1, rows, layers * 2 * width}, 33);
                    Data norm(FLOAT32, {layers, dim}, std::vector<float>(layers * dim, 1));
                    norm.ToDevice(DataDevice::CUDA, std::vector<int>{device});
                    std::vector<std::pair<Data, Data>> kv;
                    Check(FastllmCudaNaiveDraftKV(raw, norm, 91, kv, heads, dim, 8, 8, 1e-5f, 12345),
                          "general shape rejected");
                    Data pos(FLOAT32, {1, rows}, {91, 92, 93});
                    pos.ToDevice(DataDevice::CUDA, std::vector<int>{device});
                    for (int i = 0; i < layers; ++i) {
                        Data k, v, w;
                        Split(raw, 2, i * 2 * width, (i * 2 + 1) * width, k);
                        Split(raw, 2, (i * 2 + 1) * width, (i * 2 + 2) * width, v);
                        Split(norm, 0, i, i + 1, w);
                        w.Reshape({dim});
                        KimiK3RMSNorm(k, w, 1e-5f, k);
                        FastllmCudaNaiveRope(k, pos, heads, dim, dim, 12345);
                        Check(Bits(k) == Bits(kv[i].first), "general shape K bits differ");
                        Check(Bits(v) == Bits(kv[i].second), "general shape V bits differ");
                        ++checks;
                    }
                    auto before = Bits(kv[0].first);
                    auto dims = kv[0].first.dims;
                    // Reject a bad layer at the end before touching the first layer.
                    kv.back().second.dims[1]++;
                    Check(!FastllmCudaNaiveDraftKV(raw, norm, 94, kv, heads, dim, 8, 8, 1e-5f, 12345),
                          "bad cache accepted");
                    Check(kv[0].first.dims == dims && Bits(kv[0].first) == before, "fallback mutated cache");
                    kv.back().second.dims[1]--;
                    raw.strides[1]++;
                    Check(!FastllmCudaNaiveDraftKV(raw, norm, 94, kv, heads, dim, 8, 8, 1e-5f, 12345),
                          "padded raw accepted");
                    raw.strides[1]--;
                    Check(!FastllmCudaNaiveDraftKV(raw, norm, INT_MAX, kv, heads, dim, 8, 8, 1e-5f, 12345),
                          "overflowing position accepted");
                    Check(!FastllmCudaNaiveDraftKV(raw, norm, 94, kv, INT_MAX, dim, 8, 8, 1e-5f, 12345),
                          "overflowing projection shape accepted");
                    Check(kv[0].first.dims == dims && Bits(kv[0].first) == before, "rejection changed cache");
                    checks += 6;
                }
            for (int seed : {17, 99}) {
                ProjectionFixture model(seed);
                auto actual = std::make_shared<ProjectionFixture::DraftContext>();
                ProjectionFixture::DraftContext reference;
                int start = 0;
                for (int rows : {1, 2, 4, 8, 1, 8, 2}) {
                    Data hidden = BF({1, rows, 256}, start + seed);
                    Check(model.Fused(hidden, start, *actual), "model fusion rejected");
                    model.Reference(hidden, start, reference);
                    Check(!actual->workspace && model.Packed(), "fusion requires graph workspace");
                    for (int layer = 0; layer < 3; ++layer) {
                        Check(actual->kv[layer].first.dims == reference.kv[layer].first.dims,
                              "model cache shape differs");
                        Check(Relative(actual->kv[layer].first, reference.kv[layer].first) < .01, "merged K error");
                        Check(Relative(actual->kv[layer].second, reference.kv[layer].second) < .01, "merged V error");
                        checks += 3;
                    }
                    start += rows;
                    Check(actual->committed == start, "committed count differs");
                    ++checks;
                }
                auto scratch = actual->projectedKV;
                auto reused = model.Reuse(actual);
                Check(reused->committed == 0 && reused->projectedKV == scratch, "request storage not recycled");
                for (auto &p : reused->kv)
                    Check(p.first.dims[1] == 0 && p.second.dims[1] == 0, "request prefix leaked");
                Data hidden = BF({1, 2, 256}, seed);
                ProjectionFixture::DraftContext independent;
                Check(model.Fused(hidden, 0, *reused) && model.Fused(hidden, 0, independent), "reused fusion rejected");
                Check(reused->projectedKV != independent.projectedKV, "requests share mutable scratch");
                Check(Bits(reused->kv[0].first) == Bits(independent.kv[0].first), "reuse changed output");
                Data longHidden = BF({1, 9, 256}, seed);
                Check(!model.Fused(longHidden, 2, *reused) && reused->committed == 2,
                      "long input fallback changed state");
                auto before = Bits(reused->kv[0].first);
                bool lowMemory = GetLowMemMode();
                SetLowMemMode(true);
                bool fused = model.Fused(hidden, 2, *reused);
                SetLowMemMode(lowMemory);
                Check(!fused, "low-memory mode did not fall back");
                hidden.dataType = FLOAT16;
                Check(!model.Fused(hidden, 2, *reused), "non-BF16 input did not fall back");
                hidden.dataType = BFLOAT16;
                hidden.strides[1]++;
                Check(!model.Fused(hidden, 2, *reused), "padded input did not fall back");
                hidden.strides[1]--;
                model.FirstNorm().ToDevice(DataDevice::CPU);
                Check(!model.Fused(hidden, 2, *reused), "CPU weight did not fall back");
                Check(reused->committed == 2 && Bits(reused->kv[0].first) == before,
                      "fallback changed committed context");
                model.FirstNorm().ToDevice(DataDevice::CUDA, std::vector<int>{device});
                auto archived = model.Reuse(reused, true);
                Check(!archived->projectedKV && archived->kv.empty(), "history request reused idle storage");
                auto incomplete = std::make_shared<ProjectionFixture::DraftContext>();
                incomplete->projectedKV = std::make_shared<Data>();
                incomplete->kv.resize(3);
                auto recovered = model.Reuse(incomplete);
                Check(model.Fused(hidden, 0, *recovered), "partially initialized request storage failed reuse");
                Check(Bits(recovered->kv[0].first) == before, "recovered context differs");
                checks += 13;
            }
        }
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA failed");
        printf("PASS %d checks (postprocessing exact; merged GEMM relative L2 < 1%%)\n", checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
