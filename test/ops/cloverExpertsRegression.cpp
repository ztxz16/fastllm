#include "devices/cuda/fastllm-clover.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
void Check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct Buffer {
    void *p = nullptr;
    explicit Buffer(size_t bytes) { Check(cudaMalloc(&p, bytes)); }
    ~Buffer() { cudaFree(p); }
    template<class T> void Put(const std::vector<T> &x) {
        Check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    template<class T> std::vector<T> Get(size_t count, size_t offset = 0) const {
        std::vector<T> x(count);
        Check(cudaMemcpy(x.data(), (const T *)p + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
        return x;
    }
};
uint16_t BF16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    bits += 0x7fff + ((bits >> 16) & 1);
    return bits >> 16;
}
float Float(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16; float result;
    std::memcpy(&result, &bits, 4); return result;
}
void Close(float actual, float expected, float absolute, float relative) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > absolute + relative * std::abs(expected)) {
        std::cerr << "actual=" << actual << " expected=" << expected << std::endl;
        throw std::runtime_error("numerical mismatch");
    }
}
void Router(int tokens, int experts, int topk, float topP, bool ties, int stride = 0) {
    if (!stride) stride = experts;
    std::mt19937 rng(tokens + 123 * experts);
    std::normal_distribution<float> normal(0.f, 3.f);
    std::vector<uint16_t> logits(tokens * experts);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = BF16(ties ? float(i % experts % 5 == 0) : normal(rng));
    Buffer x((size_t)tokens * stride * 2), ids(tokens * topk * 4), scores(tokens * topk * 4),
        tempIds(tokens * topk * 4), tempScores(tokens * topk * 4), mask(experts);
    auto launch = [&]() {
        Require(FastllmCudaCloverRoute(x.p, (int *)ids.p, (float *)scores.p, (int *)tempIds.p,
            (float *)tempScores.p, (unsigned char *)mask.p, 1, tokens, experts, topk, topP, nullptr, stride), "router launch");
    };
    auto reference = [&]() {
        std::vector<float> masses(experts, 0.f);
        std::vector<bool> allowed(experts, true);
        auto order = [&](int t, bool filter) {
            std::vector<int> indices(experts); std::iota(indices.begin(), indices.end(), 0);
            auto value = [&](int e) { return filter && !allowed[e] ? -INFINITY : Float(logits[t * experts + e]); };
            std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) { return value(a) > value(b); });
            return indices;
        };
        if (topP > 0.f) {
            for (int t = 0; t < tokens; ++t) {
                auto rank = order(t, false); double total = 0.; float maximum = Float(logits[t * experts + rank[0]]);
                for (int e = 0; e < experts; ++e) total += std::exp(Float(logits[t * experts + e]) - maximum);
                for (int k = 0; k < topk; ++k) masses[rank[k]] += float(std::exp(Float(logits[t * experts + rank[k]]) - maximum) / total);
            }
            std::vector<int> rank(experts); std::iota(rank.begin(), rank.end(), 0);
            std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) { return masses[a] > masses[b]; });
            double total = std::accumulate(masses.begin(), masses.end(), 0.), cumulative = 0.;
            for (int i = 0; i < experts; ++i) {
                allowed[rank[i]] = i == 0 || cumulative < topP * std::max(total, 1.e-9);
                cumulative += masses[rank[i]];
            }
        }
        auto actualIds = ids.Get<int>(tokens * topk); auto actualScores = scores.Get<float>(tokens * topk);
        for (int t = 0; t < tokens; ++t) {
            auto rank = order(t, true); double sum = 0.; float maximum = Float(logits[t * experts + rank[0]]);
            for (int k = 0; k < topk; ++k) if (allowed[rank[k]]) sum += std::exp(Float(logits[t * experts + rank[k]]) - maximum);
            for (int k = 0; k < topk; ++k) {
                Require(actualIds[t * topk + k] == rank[k], "stable router/top-p selection");
                float expected = allowed[rank[k]] ? float(std::exp(Float(logits[t * experts + rank[k]]) - maximum) / sum) : 0.f;
                Close(actualScores[t * topk + k], expected, 2.e-7f, 2.e-6f);
            }
        }
    };
    auto upload = [&]() {
        std::vector<uint16_t> packed((size_t)tokens * stride, BF16(123.f));
        for (int row = 0; row < tokens; ++row)
            std::copy_n(logits.begin() + row * experts, experts, packed.begin() + row * stride);
        x.Put(packed);
    };
    upload(); launch(); reference();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (auto &v : logits) v = BF16(normal(rng));
    upload(); Check(cudaGraphLaunch(executable, cudaStreamPerThread)); reference();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
void ProjectionUnpack(int rows, int channels, int heads, bool preview, int inputOffset = 0) {
    const int values = heads * 128, width = channels + values + heads * 2;
    std::mt19937 rng(132 + rows); std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<uint16_t> input(rows * width);
    std::vector<float> aLog(heads), bias(heads);
    for (auto &v : input) v = BF16(normal(rng));
    for (auto &v : aLog) v = normal(rng);
    for (auto &v : bias) v = normal(rng);
    Buffer x((input.size() + inputOffset) * 2), a(heads * 4), b(heads * 4), qkv(rows * channels * 2),
        z(rows * values * 2), g(rows * heads * 4), beta(rows * heads * 2);
    a.Put(aLog); b.Put(bias);
    auto launch = [&]() {
        Require(FastllmCudaCloverUnpackGdn((uint16_t *)x.p + inputOffset, (float *)a.p, (float *)b.p, qkv.p, z.p,
            (float *)g.p, preview ? nullptr : beta.p, rows, channels, heads), "projection unpack launch");
    };
    auto reference = [&]() {
        auto actualQkv = qkv.Get<uint16_t>(rows * channels), actualZ = z.Get<uint16_t>(rows * values);
        auto actualG = g.Get<float>(rows * heads);
        auto actualBeta = preview ? std::vector<uint16_t>() : beta.Get<uint16_t>(rows * heads);
        for (int row = 0; row < rows; ++row) {
            for (int d = 0; d < channels; ++d) Require(actualQkv[row * channels + d] == input[row * width + d], "QKV unpack");
            for (int d = 0; d < values; ++d) Require(actualZ[row * values + d] == input[row * width + channels + d], "Z unpack");
            for (int h = 0; h < heads; ++h) {
                float av = Float(input[row * width + channels + values + h]) + bias[h];
                float expected = -std::exp(aLog[h]) * (av > 20.f ? av : std::log1p(std::exp(av)));
                Close(actualG[row * heads + h], expected, 2.e-6f, 2.e-6f);
                if (!preview) {
                    float bv = Float(input[row * width + channels + values + heads + h]);
                    Close(Float(actualBeta[row * heads + h]), Float(BF16(1.f / (1.f + std::exp(-bv)))), 0.f, .008f);
                }
            }
        }
    };
    auto upload = [&]() {
        Check(cudaMemcpy((uint16_t *)x.p + inputOffset, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
    };
    upload(); launch(); reference();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (auto &v : input) v = BF16(normal(rng) * 20.f);
    upload(); Check(cudaGraphLaunch(executable, cudaStreamPerThread)); reference();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
void Experts(int rows, int hidden, int inter, int experts, int topk, bool clustered, int inputOffset = 0,
             int outputOffset = 0, bool fusedShared = false, int sharedOffset = 0, int gateStride = 1) {
    std::mt19937 rng(734 + hidden);
    std::normal_distribution<float> normal(0.f, 1.f);
    auto random = [&](size_t n, float scale) {
        std::vector<uint16_t> out(n); for (auto &v : out) v = BF16(normal(rng) * scale); return out;
    };
    auto input = random(rows * hidden, .3f);
    auto gate = random((size_t)experts * 2 * inter * hidden, .03f);
    auto down = random((size_t)experts * hidden * inter, .03f);
    std::vector<int> routes(rows * topk), permutation(experts);
    std::vector<float> scores(rows * topk);
    for (int row = 0; row < rows; ++row) {
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), rng);
        for (int k = 0; k < topk; ++k) { routes[row * topk + k] = clustered ? k : permutation[k]; scores[row * topk + k] = 1.f / topk; }
    }
    const int n = experts * (rows + 1) + experts + (rows * topk + 15) / 16 + 1;
    auto shared = fusedShared ? random(rows * hidden, .3f) : std::vector<uint16_t>(1);
    std::vector<uint16_t> sharedGate(rows);
    const float gateValues[] = {-8.f, -1.f, 0.f, 1.f, 8.f};
    for (int row = 0; row < rows; ++row)
        sharedGate[row] = BF16(gateValues[row % 5]);
    Buffer sharedData((shared.size() + sharedOffset) * 2), sharedGateData((size_t)rows * gateStride * 2);
    auto uploadShared = [&]() {
        auto padded = shared;
        padded.insert(padded.begin(), sharedOffset, BF16(123.f));
        sharedData.Put(padded);
        std::vector<uint16_t> strided((size_t)rows * gateStride, BF16(123.f));
        for (int row = 0; row < rows; ++row) strided[(size_t)row * gateStride] = sharedGate[row];
        sharedGateData.Put(strided);
    };
    uploadShared();
    const auto *sharedPointer = (const uint16_t *)sharedData.p + sharedOffset;
    std::vector<int> guard(n + 128, 123456789);
    const int outputPadding = outputOffset ? outputOffset + 1 : 0;
    Buffer x((input.size() + inputOffset) * 2), gu(gate.size() * 2), dw(down.size() * 2), ids(routes.size() * 4),
        weights(scores.size() * 4), mid((rows * topk * inter + outputPadding) * 2),
        parts((rows * topk * hidden + outputPadding) * 2),
        y((rows * hidden + outputPadding) * 2), workspace(guard.size() * 4);
    auto paddedInput = input; paddedInput.insert(paddedInput.begin(), inputOffset, BF16(123.f));
    x.Put(paddedInput); gu.Put(gate); dw.Put(down); weights.Put(scores); workspace.Put(guard);
    const auto *inputPointer = (const char *)x.p + inputOffset * 2;
    const size_t packedBytes = FastllmCudaCloverExpertsPackedBytes(rows, hidden, inter, experts, topk);
    // The generic WMMA loader requires aligned inputs. Exercise the packed
    // path's scalar read fallback only when that architecture is available.
    if ((inputOffset || outputOffset) && !packedBytes) return;
    if (outputOffset) {
        mid.Put(std::vector<uint16_t>(rows * topk * inter + outputPadding, 0x5a5a));
        parts.Put(std::vector<uint16_t>(rows * topk * hidden + outputPadding, 0x5a5a));
        y.Put(std::vector<uint16_t>(rows * hidden + outputPadding, 0x5a5a));
    }
    Buffer packed(packedBytes + 256);
    std::vector<unsigned char> packedGuard(packedBytes + 256, 0xa5);
    packed.Put(packedGuard);
    auto launch = [&](bool usePacked = true) {
        Require(FastllmCudaCloverExperts(
                    inputPointer, gu.p, dw.p, (int *)ids.p, (float *)weights.p,
                    (uint16_t *)mid.p + outputOffset, (uint16_t *)parts.p + outputOffset,
                    (uint16_t *)y.p + outputOffset, rows, hidden, inter, experts, topk, nullptr,
                    (int *)workspace.p + 64, usePacked && packedBytes ? (char *)packed.p + 128 : nullptr,
                    fusedShared ? sharedPointer : nullptr, fusedShared ? sharedGateData.p : nullptr, gateStride),
                "expert launch");
    };
    auto reference = [&]() {
        std::vector<float> sums(rows * hidden, 0.f);
        std::vector<float> middle(inter);
        auto actualMiddle = mid.Get<uint16_t>(rows * topk * inter, outputOffset);
        // Independent host matrix products. Expert-id ordering and the two BF16
        // boundaries follow the checkpoint's published mathematical contract.
        for (int expert = 0; expert < experts; ++expert) for (int task = 0; task < rows * topk; ++task) {
            if (routes[task] != expert) continue;
            int row = task / topk;
            for (int d = 0; d < inter; ++d) {
                double g = 0., u = 0.;
                size_t w = ((size_t)expert * 2 * inter + d) * hidden;
                for (int k = 0; k < hidden; ++k) {
                    float v = Float(input[row * hidden + k]);
                    g += v * Float(gate[w + k]); u += v * Float(gate[w + (size_t)inter * hidden + k]);
                }
                float gv = Float(BF16(float(g))), uv = Float(BF16(float(u)));
                middle[d] = Float(BF16(gv / (1.f + std::exp(-gv)) * uv));
                Close(Float(actualMiddle[task * inter + d]), middle[d], .004f, .008f);
            }
            for (int d = 0; d < hidden; ++d) {
                double sum = 0.; size_t w = ((size_t)expert * hidden + d) * inter;
                for (int k = 0; k < inter; ++k) sum += middle[k] * Float(down[w + k]);
                sums[row * hidden + d] += Float(BF16(float(sum) * scores[task]));
            }
        }
        auto actual = y.Get<uint16_t>(sums.size(), outputOffset);
        if (fusedShared) {
            auto contribution = [&](size_t i) {
                float probability = Float(BF16(1.f / (1.f + std::exp(-Float(sharedGate[i / hidden])))));
                return Float(BF16(Float(shared[i]) * probability));
            };
            // Compare the fused output bit-for-bit with the ordinary expert
            // output plus separately rounded shared sigmoid/product/add.
            Require(FastllmCudaCloverExperts(
                        inputPointer, gu.p, dw.p, (int *)ids.p, (float *)weights.p,
                        (uint16_t *)mid.p + outputOffset, (uint16_t *)parts.p + outputOffset,
                        (uint16_t *)y.p + outputOffset, rows, hidden, inter, experts, topk, nullptr,
                        (int *)workspace.p + 64, packedBytes ? (char *)packed.p + 128 : nullptr),
                    "unfused reference launch");
            auto routed = y.Get<uint16_t>(sums.size(), outputOffset);
            for (size_t i = 0; i < sums.size(); ++i) {
                Require(actual[i] == BF16(Float(routed[i]) + contribution(i)),
                        "shared epilogue differs bitwise");
                sums[i] = Float(BF16(sums[i])) + contribution(i);
            }
            Require(sharedData.Get<uint16_t>(shared.size(), sharedOffset) == shared, "shared input modified");
            auto strided = sharedGateData.Get<uint16_t>((size_t)rows * gateStride);
            for (int row = 0; row < rows; ++row) for (int d = 0; d < gateStride; ++d)
                Require(strided[(size_t)row * gateStride + d] == (d ? BF16(123.f) : sharedGate[row]),
                        "shared gate or padding modified");
        }
        for (size_t i = 0; i < sums.size(); ++i) Close(Float(actual[i]), Float(BF16(sums[i])), .0005f, .02f);
        auto actualGuard = workspace.Get<int>(guard.size());
        for (int i = 0; i < 64; ++i) Require(actualGuard[i] == guard[i] && actualGuard[n + 64 + i] == guard[i], "expert workspace overrun");
        auto actualPacked = packed.Get<unsigned char>(packedGuard.size());
        for (int i = 0; i < 128; ++i)
            Require(actualPacked[i] == 0xa5 && actualPacked[128 + packedBytes + i] == 0xa5, "TMA workspace overrun");
        if (outputOffset) {
            auto checkView = [&](const Buffer &buffer, size_t count) {
                auto values = buffer.Get<uint16_t>(count + outputPadding);
                for (int i = 0; i < outputOffset; ++i)
                    Require(values[i] == 0x5a5a, "expert output prefix overrun");
                Require(values.back() == 0x5a5a, "expert output suffix overrun");
            };
            checkView(mid, rows * topk * inter);
            checkView(parts, rows * topk * hidden);
            checkView(y, rows * hidden);
        }
    };
    ids.Put(routes);
    if (packedBytes && !inputOffset && !outputOffset) {
        launch(false);
        auto expectedMiddle = mid.Get<uint16_t>(rows * topk * inter);
        auto expectedParts = parts.Get<uint16_t>(rows * topk * hidden);
        auto expectedOutput = y.Get<uint16_t>(rows * hidden);
        launch();
        Require(mid.Get<uint16_t>(rows * topk * inter) == expectedMiddle, "TMA gate/up differs from generic");
        Require(parts.Get<uint16_t>(rows * topk * hidden) == expectedParts, "TMA down differs from generic");
        Require(y.Get<uint16_t>(rows * hidden) == expectedOutput, "TMA expert sum differs from generic");
    } else launch();
    reference();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int task = 0; task < rows * topk; ++task) routes[task] = task % topk;
    if (fusedShared) {
        for (auto &v : shared)
            v = BF16(normal(rng));
        for (int row = 0; row < rows; ++row)
            sharedGate[row] = BF16(gateValues[(row + 2) % 5]);
        uploadShared();
    }
    ids.Put(routes); Check(cudaGraphLaunch(executable, cudaStreamPerThread)); reference();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
}
int main() {
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        Check(cudaSetDevice(0));
        for (int tokens : {1, 4, 16, 32}) for (float p : {0.f, .5f, .9f}) for (bool ties : {false, true}) Router(tokens, 256, 8, p, ties);
        for (float p : {0.f, .9f}) for (bool ties : {false, true})
            Router(16, 256, 8, p, ties, 320);
        Router(16, 17, 16, .9f, false); Router(8, 16, 4, .9f, true);
        for (int rows : {1, 16, 512}) for (bool preview : {false, true}) ProjectionUnpack(rows, 8192, 32, preview);
        ProjectionUnpack(7, 383, 3, false);
        for (int rows : {127, 128, 129}) for (bool preview : {false, true})
            ProjectionUnpack(rows, 8192, 32, preview);
        ProjectionUnpack(129, 8192, 32, false, 1);
        ProjectionUnpack(129, 383, 3, false);
        ProjectionUnpack(129, 1024, 3, true);
        Experts(4, 130, 67, 9, 4, false);
        Experts(17, 514, 130, 9, 4, true);
        Experts(7, 520, 136, 11, 4, false);
        Experts(16, 2048, 512, 16, 8, false);
        // Mostly one-to-three-row experts, followed by a graph replay that
        // routes every token to the same experts and leaves the others empty.
        Experts(16, 2048, 512, 256, 8, false);
        Experts(33, 512, 128, 17, 4, false);
        Experts(33, 512, 512, 19, 8, false);
        Experts(16, 512, 128, 17, 4, false, 1);
        Experts(16, 512, 128, 17, 4, false, 0, 1);
        // The wider-K gate is limited to small blocks and128-aligned widths.
        // Cover its row boundary, a partial K fallback and unaligned views.
        Experts(4, 512, 128, 17, 4, false);
        Experts(32, 512, 128, 17, 4, false);
        Experts(16, 576, 128, 17, 4, false);
        Experts(16, 512, 128, 17, 4, false, 1, 1);
        // Long-prefill bucketing and vectorized gather/sum: threshold edges,
        // a partial final warp block, generic GEMMs and an unaligned input.
        for (int rows : {1023, 1024, 1025}) Experts(rows, 512, 128, 33, 2, false);
        Experts(1025, 520, 136, 33, 2, false);
        Experts(1024, 512, 128, 33, 2, false, 1);
        Experts(1024, 512, 128, 17, 1, true);
        // SM90 WGMMA dispatch boundaries, partial tiles, unaligned views and
        // shapes that must retain the generic implementation. Graph replays
        // also change the routes from sparse to concentrated experts.
        for (int rows : {2047, 2048, 2049}) Experts(rows, 512, 128, 33, 2, false);
        Experts(2048, 512, 128, 33, 2, false, 1);
        Experts(2049, 512, 128, 33, 2, false, 0, 1);
        Experts(2049, 520, 136, 33, 2, false);
        Experts(2048, 512, 128, 65, 2, false);
        // Generic GEMV, small TMA and vectorized prefill reductions, including
        // unaligned shared values and refreshed shared/gate graph inputs.
        Experts(4, 130, 67, 9, 4, false, 0, 0, true, 1);
        Experts(16, 512, 128, 17, 4, false, 0, 1, true, 1);
        Experts(1025, 512, 128, 33, 2, false, 0, 0, true, 1);
        for (int rows : {16, 32, 64, 128, 1025})
            Experts(rows, 512, 128, 17, 4, false, 0, 0, true, 1, 320);
        Require(FastllmCudaCloverExpertsPackedBytes(3, 2048, 512, 256, 8) == 0, "sub-tile blocks must keep generic path");
        Require(FastllmCudaCloverExpertsPackedBytes(64, 513, 512, 256, 8) == 0, "unaligned TMA shape");
        Require(FastllmCudaCloverExpertsPackedBytes(64, 2048, 512, 256, 0) == 0, "invalid TMA topk");
        std::cout << "Clover experts/router/projections: CPU reference, stable ties, graph route refresh and workspace guards PASS" << std::endl;
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
