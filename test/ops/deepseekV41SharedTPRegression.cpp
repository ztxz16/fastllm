// Intermediate TP: exact slicing/padding, FP64 dot-product oracles, block-32
// activation equivalence, FP32 rank sum, and poisoned multi-GPU graph replays.
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool v, const char *s) {
    if (!v)
        throw std::runtime_error(s);
}
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> static std::vector<T> Read(const Data &d) {
    std::vector<T> v(d.GetBytes() / sizeof(T));
    Cuda(cudaMemcpy(v.data(), d.cudaData, d.GetBytes(), cudaMemcpyDeviceToHost));
    return v;
}
static float Bf(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static uint16_t RoundBf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    bits += 0x7fff + ((bits >> 16) & 1);
    return uint16_t(bits >> 16);
}
static float Fp8(uint8_t b) {
    int e = (b >> 3) & 15, m = b & 7;
    float f = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(float(m), -9);
    return b & 128 ? -f : f;
}
static void Alloc(Data &d, DataType type, std::vector<int> dims, int device) {
    d.dataType = type;
    d.UpdateUnitSize();
    d.Resize(dims);
    d.dataDevice = DataDevice::CUDA;
    d.dataDeviceIds = {device};
    d.Allocate();
}
struct Rank {
    Data input, gate, down, g, a, out, route, rounded;
    void *graph = nullptr, *exec = nullptr;
};
static void Weight(Data &d, int n, int k, const std::vector<uint8_t> &bytes, const std::vector<float> &scales,
                   int device) {
    Alloc(d, DataType::FP8_E4M3, {n, k}, device);
    d.blockM = d.blockK = 32;
    d.scales = scales;
    Cuda(cudaMemcpy(d.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}
struct Error {
    double error = 0, norm = 0, maximum = 0, backward = 0;
    void add(double actual, double expected, double absoluteSum = 0) {
        Check(std::isfinite(actual), "nonfinite result");
        double d = actual - expected;
        error += d * d;
        norm += expected * expected;
        maximum = std::max(maximum, std::abs(d) / std::max(1., std::abs(expected)));
        if (absoluteSum > 0)
            backward = std::max(backward, std::abs(d) / absoluteSum);
    }
    double rms() const {
        return std::sqrt(error / std::max(norm, 1e-30));
    }
};
int main(int argc, char **argv) {
    try {
        int ngpu = 0;
        Cuda(cudaGetDeviceCount(&ngpu));
        if (ngpu < 8)
            return 77;
        const bool small = argc > 1 && std::string(argv[1]) == "--small";
        const bool narrow = argc > 1 && std::string(argv[1]) == "--narrow";
        const bool localOnly = argc > 2 && std::string(argv[2]) == "--local";
        SetThreads(16);
        FastllmCudaSetNcclForceSync(false);
        int cases = 0;
        for (int ranks : {2, 4, 8}) {
            std::vector<int> devices;
            for (int d = 0; d < ranks; ++d)
                devices.push_back(d);
            FastllmMultiCudaSetDevice(devices);
            Check(FastllmInitNccl(devices), "NCCL init");
            Check(FastllmCudaCustomAllReduceInit(devices), "custom reduce init");
            const int h = small ? 256 : narrow ? 192 : 5120, mid = small ? 768 : narrow ? 1024 : 2304;
            const int local = mid / ranks, padded = (local + 63) / 64 * 64;
            std::mt19937 random(701 + ranks);
            std::vector<uint8_t> gw(size_t(2 * mid) * h), dw(size_t(h) * mid);
            for (auto *v : {&gw, &dw})
                for (auto &x : *v)
                    x = uint8_t(random() % 108) | (random() % 2 ? 128 : 0);
            std::vector<float> gs(size_t(2 * mid / 32) * (h / 32)), ds(size_t(h / 32) * (mid / 32));
            for (auto *v : {&gs, &ds})
                for (auto &x : *v)
                    x = std::ldexp(1.f, int(random() % 7) - 10);
            std::vector<std::unique_ptr<Rank>> state;
            for (int d : devices) {
                FastllmCudaSetDevice(d);
                state.emplace_back(new Rank);
                auto &s = *state.back();
                Weight(s.gate, 2 * mid, h, gw, gs, d);
                Weight(s.down, h, mid, dw, ds, d);
                Data *g = FastllmCudaFP8PairedRowShard(s.gate, d * local, local);
                Data *w = FastllmCudaFP8ColumnShard(s.down, d * local, local);
                Check(g && w, "shard construction");
                auto gb = Read<uint8_t>(*g), db = Read<uint8_t>(*w);
                for (int n = 0; n < 2 * padded; ++n)
                    for (int k = 0; k < h; ++k) {
                        int c = n % padded;
                        uint8_t expected =
                            c < local ? gw[size_t((n / padded) * mid + d * local + c) * h + k] : 0;
                        Check(gb[size_t(n) * h + k] == expected, "paired row bytes/padding");
                    }
                for (int n = 0; n < h; ++n)
                    for (int k = 0; k < padded; ++k)
                        Check(db[size_t(n) * padded + k] ==
                                  (k < local ? dw[size_t(n) * mid + d * local + k] : 0),
                              "column bytes/padding");
                for (int n = 0; n < 2 * padded / 32; ++n)
                    for (int k = 0; k < h / 32; ++k) {
                        int c = n % (padded / 32);
                        float expected =
                            c < local / 32
                                ? gs[size_t((n / (padded / 32)) * (mid / 32) + d * local / 32 + c) *
                                         (h / 32) +
                                     k]
                                : 1.f;
                        Check(g->scales[size_t(n) * (h / 32) + k] == expected, "paired scales");
                    }
                for (int n = 0; n < h / 32; ++n)
                    for (int k = 0; k < padded / 32; ++k)
                        Check(w->scales[size_t(n) * (padded / 32) + k] ==
                                  (k < local / 32 ? ds[size_t(n) * (mid / 32) + d * local / 32 + k] : 1.f),
                              "column scales");
            }
            auto sync = [&] {
                for (int d : devices)
                    FastllmCudaSyncDevice(d);
                FastllmCudaSetDevice(0);
            };
            for (int m : {1, 2, 3, 4, 5, 6, 7, 8}) {
                std::vector<uint16_t> xb(size_t(m) * h);
                for (auto &x : xb)
                    x = uint16_t((random() % 2 ? 0x8000 : 0) | 0x3e00 | (random() % 192));
                std::vector<std::vector<float>> partials(ranks), routes(ranks);
                for (int d : devices) {
                    FastllmCudaSetDevice(d);
                    auto &s = *state[d];
                    Alloc(s.input, DataType::BFLOAT16, {m, h}, d);
                    Alloc(s.g, DataType::BFLOAT16, {m, 2 * padded}, d);
                    Alloc(s.a, DataType::BFLOAT16, {m, padded}, d);
                    Alloc(s.out, DataType::FLOAT32, {m, h}, d);
                    Alloc(s.route, DataType::FLOAT32, {m, h}, d);
                    Alloc(s.rounded, DataType::BFLOAT16, {m, h}, d);
                    Cuda(cudaMemcpy(s.input.cudaData, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
                    routes[d].resize(size_t(m) * h);
                    for (size_t i = 0; i < routes[d].size(); ++i)
                        routes[d][i] = (int(i % 17) - 8) * (d + 1) * .001f;
                    Cuda(cudaMemcpy(s.route.cudaData, routes[d].data(), routes[d].size() * 4,
                                    cudaMemcpyHostToDevice));
                }
                auto localOp = [&](int d) {
                    auto &s = *state[d];
                    Check(FastllmCudaV41SharedExpertTP(s.input, s.gate, s.down, s.g, s.a, s.out, d, ranks,
                                                       10.f),
                          "shared TP operator rejected");
                };
                Check(MultiCudaRunDeviceCallbacks(devices, [&](int, int d) { localOp(d); }),
                      "operator dispatch");
                sync();
                std::vector<uint16_t> fullGate(size_t(m) * 2 * mid), fullActivation(size_t(m) * mid);
                Error gateError, partialError, totalError;
                for (int d : devices) {
                    FastllmCudaSetDevice(d);
                    auto &s = *state[d];
                    auto g = Read<uint16_t>(s.g), a = Read<uint16_t>(s.a);
                    partials[d] = Read<float>(s.out);
                    for (int row = 0; row < m; ++row)
                        for (int c = 0; c < padded; ++c) {
                            if (c >= local) {
                                Check(Bf(g[size_t(row) * 2 * padded + c]) == 0 &&
                                          Bf(g[size_t(row) * 2 * padded + padded + c]) == 0 &&
                                          Bf(a[size_t(row) * padded + c]) == 0,
                                      "nonzero padded activation");
                                continue;
                            }
                            fullGate[size_t(row) * 2 * mid + d * local + c] = g[size_t(row) * 2 * padded + c];
                            fullGate[size_t(row) * 2 * mid + mid + d * local + c] =
                                g[size_t(row) * 2 * padded + padded + c];
                            fullActivation[size_t(row) * mid + d * local + c] = a[size_t(row) * padded + c];
                        }
                    for (int row = 0; row < m; ++row)
                        for (int half = 0; half < 2; ++half)
                            for (int c : {0, 1, 31, 32, local - 1}) {
                                int n = half * mid + d * local + c;
                                double sum = 0;
                                for (int k = 0; k < h; ++k)
                                    sum += double(Bf(xb[size_t(row) * h + k])) * Fp8(gw[size_t(n) * h + k]) *
                                           gs[size_t(n / 32) * (h / 32) + k / 32];
                                gateError.add(Bf(g[size_t(row) * 2 * padded + half * padded + c]), sum);
                            }
                    for (int row = 0; row < m; ++row)
                        for (int t = 0; t < 24; ++t) {
                            int n = t * (h - 1) / 23;
                            double sum = 0, absoluteSum = 0;
                            for (int c = 0; c < local; ++c) {
                                int k = d * local + c;
                                double product = double(Bf(a[size_t(row) * padded + c])) *
                                                 Fp8(dw[size_t(n) * mid + k]) *
                                                 ds[size_t(n / 32) * (mid / 32) + k / 32];
                                sum += product;
                                absoluteSum += std::abs(product);
                            }
                            partialError.add(partials[d][size_t(row) * h + n], sum, absoluteSum);
                        }
                }
                FastllmCudaSetDevice(0);
                Data merged, act;
                Alloc(merged, DataType::BFLOAT16, {m, 2 * mid}, 0);
                Cuda(cudaMemcpy(merged.cudaData, fullGate.data(), fullGate.size() * 2,
                                cudaMemcpyHostToDevice));
                Check(FastllmCudaDeepSeekV41SharedSwigluQuantized(merged, 10.f, act), "reference swiglu");
                Check(Read<uint16_t>(act) == fullActivation,
                      "local activation differs from full block-32 activation");
                std::vector<float> expected(size_t(m) * h, 0);
                for (size_t i = 0; i < expected.size(); ++i) {
                    expected[i] = partials[0][i] + routes[0][i];
                    for (int d = 1; d < ranks; ++d)
                        expected[i] += partials[d][i] + routes[d][i];
                }
                for (int row = 0; row < m; ++row)
                    for (int t = 0; t < 24; ++t) {
                        int n = t * (h - 1) / 23;
                        double sum = 0, absoluteSum = 0;
                        for (int k = 0; k < mid; ++k) {
                            double product = double(Bf(fullActivation[size_t(row) * mid + k])) *
                                             Fp8(dw[size_t(n) * mid + k]) *
                                             ds[size_t(n / 32) * (mid / 32) + k / 32];
                            sum += product;
                            absoluteSum += std::abs(product);
                        }
                        for (int d = 0; d < ranks; ++d) {
                            sum += routes[d][size_t(row) * h + n];
                            absoluteSum += std::abs(routes[d][size_t(row) * h + n]);
                        }
                        totalError.add(expected[size_t(row) * h + n], sum, absoluteSum);
                    }
                std::cout << "oracle TP=" << ranks << " M=" << m << " gate=" << gateError.rms() << ","
                          << gateError.maximum << " partial=" << partialError.rms() << ","
                          << partialError.maximum << " total=" << totalError.rms() << ","
                          << totalError.maximum << std::endl;
                Check(gateError.rms() < .004 && gateError.maximum < .02, "gate FP64 oracle");
                Check(partialError.rms() < 1e-5 && partialError.backward < 2e-6, "down partial FP64 oracle");
                Check(totalError.rms() < 1e-5 && totalError.backward < 2e-6, "merged FP64 oracle");
                if (m == 1) {
                    FastllmCudaSetDevice(0);
                    auto &s = *state[0];
                    const auto before = Read<float>(s.out);
                    auto rejected = [&](int rank, int nr) {
                        return !FastllmCudaV41SharedExpertTP(s.input, s.gate, s.down, s.g, s.a, s.out, rank,
                                                             nr, 10.f);
                    };
                    Check(rejected(-1, ranks) && rejected(ranks, ranks) && rejected(0, 1),
                          "invalid rank accepted");
                    s.input.dataType = DataType::FLOAT16;
                    Check(rejected(0, ranks), "FP16 input accepted");
                    s.input.dataType = DataType::BFLOAT16;
                    s.down.blockM = 128;
                    Check(rejected(0, ranks), "unsupported scale layout accepted");
                    s.down.blockM = 32;
                    Data bad;
                    auto badScales = ds;
                    badScales[0] *= 1.1f;
                    Weight(bad, h, mid, dw, badScales, 0);
                    Check(
                        !FastllmCudaV41SharedExpertTP(s.input, s.gate, bad, s.g, s.a, s.out, 0, ranks, 10.f),
                        "inexact E8M0 scale accepted");
                    Check(Read<float>(s.out) == before, "rejected operation modified output");
                    Check(!FastllmCudaTryMarlinBFloat16MatMulFP8E4M3ToFloat(
                              s.a.cudaData, s.down, (float *)s.out.cudaData, 9, mid, h),
                          "FP32 Marlin accepted unsupported M");
                }
                for (bool bf16Reduce : {false, true}) {
                    auto run = [&](int d) {
                        auto &s = *state[d];
                        localOp(d);
                        if (bf16Reduce) {
                            Check(FastllmCudaDeepSeekV41MoeJoinBFloat16(
                                static_cast<const float *>(s.route.cudaData),
                                static_cast<const float *>(s.out.cudaData), s.rounded.cudaData, m * h),
                                "BF16 branch join");
                            if (!localOnly)
                                Check(FastllmCudaCustomAllReduce(s.rounded.cudaData, s.rounded.cudaData,
                                    m * h, DataType::BFLOAT16, d), "BF16 reduce");
                            return;
                        }
                        Check(FastllmCudaAddTo(s.out, s.route, 1.f), "route addition");
                        if (!localOnly)
                            Check(FastllmCudaCustomAllReduce(s.out.cudaData, s.out.cudaData, m * h,
                                                             DataType::FLOAT32, d),
                                  "FP32 reduce");
                        Check(FastllmFloatToBF16(s.out.cudaData, s.rounded.cudaData, m * h), "BF16 rounding");
                    };
                    Check(MultiCudaRunDeviceCallbacks(devices, [&](int, int d) { run(d); }), "reduce dispatch");
                    sync();
                    std::vector<std::vector<float>> expectedByRank(ranks, expected);
                    std::vector<std::vector<uint16_t>> expectedRounded(ranks,
                                                                       std::vector<uint16_t>(expected.size()));
                    for (int d : devices)
                        for (size_t i = 0; i < expected.size(); ++i) {
                            if (localOnly)
                                expectedByRank[d][i] = partials[d][i] + routes[d][i];
                            expectedRounded[d][i] = RoundBf(expectedByRank[d][i]);
                            if (bf16Reduce) {
                                float sum = 0;
                                for (int rank = localOnly ? d : 0; rank < (localOnly ? d + 1 : ranks); ++rank) {
                                    float joined = Bf(RoundBf(Bf(RoundBf(partials[rank][i])) +
                                                             Bf(RoundBf(routes[rank][i]))));
                                    if (rank == (localOnly ? d : 0)) sum = joined;
                                    else sum += joined;
                                }
                                expectedRounded[d][i] = RoundBf(sum);
                            }
                        }
                    for (int d : devices) {
                        FastllmCudaSetDevice(d);
                        if (!bf16Reduce)
                            Check(Read<float>(state[d]->out) == expectedByRank[d], "rank-ordered sum mismatch");
                        Check(Read<uint16_t>(state[d]->rounded) == expectedRounded[d], "BF16 rounding mismatch");
                    }
                    bool oldAsync = MultiCudaSetPersistentAsyncDispatch(true),
                         oldCapture = MultiCudaSetWorkerOrderedCapture(true);
                    Check(MultiCudaRunDeviceCallbacks(devices,
                                                      [&](int, int) {
                                                          Check(FastllmCudaGraphPrepareCaptureDevice(),
                                                                "prepare capture");
                                                      }),
                          "prepare dispatch");
                    Check(MultiCudaRunDeviceCallbacks(
                              devices,
                              [&](int, int d) {
                                  Check(FastllmCudaGraphBeginCapture(), "begin capture");
                                  run(d);
                                  Check(FastllmCudaGraphEndCapture(&state[d]->graph), "end capture");
                                  Check(FastllmCudaGraphInstantiate(state[d]->graph, &state[d]->exec),
                                        "instantiate");
                              }),
                          "capture dispatch");
                    for (int replay = 0; replay < 4; ++replay) {
                        Check(MultiCudaRunDeviceCallbacks(
                                  devices,
                                  [&](int, int d) {
                                      auto &s = *state[d];
                                      Cuda(cudaMemsetAsync(s.out.cudaData, 0xff, s.out.GetBytes(),
                                                           cudaStreamPerThread));
                                      Cuda(cudaMemsetAsync(s.rounded.cudaData, 0xff, s.rounded.GetBytes(),
                                                           cudaStreamPerThread));
                                      Cuda(cudaMemsetAsync(s.g.cudaData, 0xff, s.g.GetBytes(),
                                                           cudaStreamPerThread));
                                      Cuda(cudaMemsetAsync(s.a.cudaData, 0xff, s.a.GetBytes(),
                                                           cudaStreamPerThread));
                                      Check(FastllmCudaGraphLaunch(s.exec), "replay");
                                  }),
                              "replay dispatch");
                        sync();
                        for (int d : devices) {
                            FastllmCudaSetDevice(d);
                            if (!bf16Reduce)
                                Check(Read<float>(state[d]->out) == expectedByRank[d], "poisoned replay mismatch");
                            Check(Read<uint16_t>(state[d]->rounded) == expectedRounded[d],
                                  "poisoned rounded replay mismatch");
                        }
                    }
                    MultiCudaRunDeviceCallbacks(devices, [&](int, int d) {
                        FastllmCudaGraphExecDestroy(state[d]->exec);
                        FastllmCudaGraphDestroy(state[d]->graph);
                    });
                    MultiCudaSetWorkerOrderedCapture(oldCapture);
                    MultiCudaSetPersistentAsyncDispatch(oldAsync);
                    std::cout << (bf16Reduce ? "BF16 " : "FP32 ") << "TP=" << ranks << " M=" << m << " H=" << h << " gate_rms=" << gateError.rms()
                              << " partial_rms=" << partialError.rms() << " total_rms=" << totalError.rms()
                              << " partial_backward=" << partialError.backward
                              << " total_backward=" << totalError.backward << " PASS" << std::endl;
                    ++cases;
                }
            }
            sync();
            FastllmCudaCustomAllReduceReset();
            state.clear();
        }
        std::cout << "PASS " << cases << " shared TP cases, FP64 + activation + "
                  << (localOnly ? "local sum" : "reduction") << " + graph" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
