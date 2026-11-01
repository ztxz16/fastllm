#include "devices/cuda/fastllm-moe.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
bool fp16 = false;
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
uint16_t Round16(float value) {
    if (fp16) return __half_raw(__float2half_rn(value)).x;
    uint32_t bits; std::memcpy(&bits, &value, 4);
    bits += 0x7fff + ((bits >> 16) & 1);
    return bits >> 16;
}
float Float(uint16_t value) {
    if (fp16) return __half2float(__half(__half_raw{value}));
    uint32_t bits = uint32_t(value) << 16; float result;
    std::memcpy(&result, &bits, 4); return result;
}
void Close(float actual, float expected, float absolute, float relative) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > absolute + relative * std::abs(expected)) {
        std::cerr << "actual=" << actual << " expected=" << expected << std::endl;
        throw std::runtime_error("numerical mismatch");
    }
}
void Experts(int rows, int hidden, int inter, int experts, int topk, bool clustered, int inputOffset = 0,
             int outputOffset = 0, bool fusedShared = false, int sharedOffset = 0, int gateStride = 1,
             bool varyingScores = false) {
    std::mt19937 rng(734 + hidden);
    std::normal_distribution<float> normal(0.f, 1.f);
    auto random = [&](size_t n, float scale) {
        std::vector<uint16_t> out(n); for (auto &v : out) v = Round16(normal(rng) * scale); return out;
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
    if (varyingScores) for (int row = 0; row < rows; ++row) for (int k = 0; k < topk; ++k)
        scores[row * topk + k] = topk == 1 ? 1.f : float(k) / (topk * (topk - 1) / 2);
    const int n = FastllmCudaMoe16WorkspaceBytes(rows, hidden, inter, experts, topk, fp16) / 4;
    auto shared = fusedShared ? random(rows * hidden, .3f) : std::vector<uint16_t>(1);
    std::vector<uint16_t> sharedGate(rows);
    const float gateValues[] = {-8.f, -1.f, 0.f, 1.f, 8.f};
    for (int row = 0; row < rows; ++row)
        sharedGate[row] = Round16(gateValues[row % 5]);
    Buffer sharedData((shared.size() + sharedOffset) * 2), sharedGateData((size_t)rows * gateStride * 2);
    auto uploadShared = [&]() {
        auto padded = shared;
        padded.insert(padded.begin(), sharedOffset, Round16(123.f));
        sharedData.Put(padded);
        std::vector<uint16_t> strided((size_t)rows * gateStride, Round16(123.f));
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
    auto paddedInput = input; paddedInput.insert(paddedInput.begin(), inputOffset, Round16(123.f));
    x.Put(paddedInput); gu.Put(gate); dw.Put(down); weights.Put(scores); workspace.Put(guard);
    const auto *inputPointer = (const char *)x.p + inputOffset * 2;
    const size_t packedBytes = FastllmCudaMoe16PackedBytes(rows, hidden, inter, experts, topk, fp16);
    // Exercise both packed precisions and FP16 legacy loaders with unaligned
    // views; legacy BF16-only configurations are skipped.
    if ((inputOffset || outputOffset) && !packedBytes && !fp16) return;
    if (outputOffset) {
        mid.Put(std::vector<uint16_t>(rows * topk * inter + outputPadding, 0x5a5a));
        parts.Put(std::vector<uint16_t>(rows * topk * hidden + outputPadding, 0x5a5a));
        y.Put(std::vector<uint16_t>(rows * hidden + outputPadding, 0x5a5a));
    }
    Buffer packed(packedBytes + 256);
    std::vector<unsigned char> packedGuard(packedBytes + 256, 0xa5);
    packed.Put(packedGuard);
    auto launch = [&](bool usePacked = true) {
        Require(FastllmCudaMoe16(
                    inputPointer, gu.p, dw.p, (int *)ids.p, (float *)weights.p,
                    (uint16_t *)mid.p + outputOffset, (uint16_t *)parts.p + outputOffset,
                    (uint16_t *)y.p + outputOffset, rows, hidden, inter, experts, topk, nullptr,
                    (int *)workspace.p + 64, usePacked && packedBytes ? (char *)packed.p + 128 : nullptr,
                    fusedShared ? sharedPointer : nullptr, fusedShared ? sharedGateData.p : nullptr, gateStride, fp16),
                "expert launch");
    };
    auto reference = [&]() {
        std::vector<float> sums(rows * hidden, 0.f);
        std::vector<float> middle(inter);
        auto actualMiddle = mid.Get<uint16_t>(rows * topk * inter, outputOffset);
        // Independent host matrix products with explicit storage-precision rounding
        // at gate/up, SwiGLU and routed-down boundaries.
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
                float gv = Float(Round16(float(g))), uv = Float(Round16(float(u)));
                middle[d] = Float(Round16(gv / (1.f + std::exp(-gv)) * uv));
                Close(Float(actualMiddle[task * inter + d]), middle[d], .004f, .008f);
            }
            for (int d = 0; d < hidden; ++d) {
                double sum = 0.; size_t w = ((size_t)expert * hidden + d) * inter;
                for (int k = 0; k < inter; ++k) sum += middle[k] * Float(down[w + k]);
                sums[row * hidden + d] += Float(Round16(float(sum) * scores[task]));
            }
        }
        auto actual = y.Get<uint16_t>(sums.size(), outputOffset);
        if (fusedShared) {
            auto contribution = [&](size_t i) {
                float probability = Float(Round16(1.f / (1.f + std::exp(-Float(sharedGate[i / hidden])))));
                return Float(Round16(Float(shared[i]) * probability));
            };
            // Compare the fused output bit-for-bit with the ordinary expert
            // output plus separately rounded shared sigmoid/product/add.
            Require(FastllmCudaMoe16(
                        inputPointer, gu.p, dw.p, (int *)ids.p, (float *)weights.p,
                        (uint16_t *)mid.p + outputOffset, (uint16_t *)parts.p + outputOffset,
                        (uint16_t *)y.p + outputOffset, rows, hidden, inter, experts, topk, nullptr,
                        (int *)workspace.p + 64, packedBytes ? (char *)packed.p + 128 : nullptr, nullptr, nullptr, 1, fp16),
                    "unfused reference launch");
            auto routed = y.Get<uint16_t>(sums.size(), outputOffset);
            for (size_t i = 0; i < sums.size(); ++i) {
                Require(actual[i] == Round16(Float(routed[i]) + contribution(i)),
                        "shared epilogue differs bitwise");
                sums[i] = Float(Round16(sums[i])) + contribution(i);
            }
            Require(sharedData.Get<uint16_t>(shared.size(), sharedOffset) == shared, "shared input modified");
            auto strided = sharedGateData.Get<uint16_t>((size_t)rows * gateStride);
            for (int row = 0; row < rows; ++row) for (int d = 0; d < gateStride; ++d)
                Require(strided[(size_t)row * gateStride + d] == (d ? Round16(123.f) : sharedGate[row]),
                        "shared gate or padding modified");
        }
        for (size_t i = 0; i < sums.size(); ++i) Close(Float(actual[i]), Float(Round16(sums[i])), .0005f, .02f);
        auto actualGuard = workspace.Get<int>(guard.size());
        // Scalar GEMV and cuBLAS leave expert metadata untouched. The latter
        // uses the extra scratch after it. Tests may JIT older PTX, so inspect
        // metadata rather than relying on the physical GPU architecture.
        bool grouped = actualGuard[64 + rows] != guard[64 + rows];
        if (grouped && rows >= 4 && rows * topk <= 512 && hidden >= 512 && inter >= 128 && experts <= 256) {
            std::vector<int> expectedTiles;
            for (int e = 0; e < experts; ++e) {
                std::vector<int> tasks;
                for (int task = 0; task < rows * topk; ++task) if (routes[task] == e) tasks.push_back(task);
                const size_t offset = 64 + (size_t)e * (rows + 1);
                Require(actualGuard[offset + rows] == (int)tasks.size(), "fused expert count differs");
                Require(std::equal(tasks.begin(), tasks.end(), actualGuard.begin() + offset), "fused expert order differs");
                for (int first = 0; first < (int)tasks.size(); first += 16)
                    expectedTiles.push_back(e * (rows + 1) + first);
            }
            const size_t offset = 64 + (size_t)experts * (rows + 1);
            Require(actualGuard[offset] == (int)expectedTiles.size(), "fused expert tile count differs");
            Require(std::equal(expectedTiles.begin(), expectedTiles.end(), actualGuard.begin() + offset + 1), "fused expert tiles differ");
        } else if (!grouped) {
            const size_t metadataWords = size_t(experts) * (rows + 1) + experts +
                (rows * topk + 15) / 16 + 1;
            Require(std::equal(guard.begin() + 64, guard.begin() + 64 + metadataWords,
                               actualGuard.begin() + 64), "unused expert metadata modified");
        }

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
            v = Round16(normal(rng));
        for (int row = 0; row < rows; ++row)
            sharedGate[row] = Round16(gateValues[(row + 2) % 5]);
        uploadShared();
    }
    if (varyingScores) {
        for (int row = 0; row < rows; ++row) for (int k = 0; k < topk; ++k)
            scores[row * topk + k] = float(topk - k) / (topk * (topk + 1) / 2);
        weights.Put(scores);
    }
    ids.Put(routes); Check(cudaGraphLaunch(executable, cudaStreamPerThread)); reference();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
}
int main(int argc, char **argv) {
    fp16 = argc > 1 && std::string(argv[1]) == "fp16";
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        Check(cudaSetDevice(0));
        for (int rows : {1, 2, 3}) Experts(rows, 512, 128, 17, 4, false);
        Experts(16, 512, 128, 17, 4, false, 0, 0, true, 0, 1, true);
        Experts(128, 512, 128, 17, 4, false, 1, 1, false, 0, 1, true);
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
        // Adaptive wide tiles must follow changed graph routes and preserve
        // unaligned output/shared views as well as zero routing weights.
        Experts(1024, 512, 128, 33, 2, false, 1, 1, true, 1, 320, true);
        Experts(1024, 512, 128, 65, 2, false);
        // A diffuse route distribution needs multiple narrow tiles per CTA;
        // replaying concentrated routes then switches to wide down tiles.
        Experts(1024, 512, 128, 256, 8, false, 0, 1, false, 0, 1, true);
        // Verification batches use the same atomic buckets and four-row
        // packing as prefill. Cover threshold edges, sparse/clustered routes,
        // dynamic graph updates and unaligned input/output views.
        for (int rows : {255, 256, 257}) Experts(rows, 512, 128, 33, 2, false);
        Experts(512, 2048, 512, 33, 8, false, 0, 0, true, 1, 320);
        // Direct TMA-gate input gather keeps the original MMA ordering.
        // Exercise its upper boundary and an unaligned-input fallback.
        Experts(513, 512, 128, 33, 2, false);
        Experts(256, 2048, 128, 33, 4, false, 1);
        Experts(256, 1024, 128, 33, 4, true, 1, 1);
        // Vector reduction threshold and partial column stripe; preserve the
        // sorted-expert sum and separately rounded shared contribution.
        for (int rows : {127, 128, 129})
            Experts(rows, 520, 136, 17, 4, false, 0, 0, true, 1, 320);
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
        Require(FastllmCudaMoe16PackedBytes(3, 2048, 512, 256, 8, fp16) == 0, "sub-tile blocks must keep generic path");
        Require(FastllmCudaMoe16PackedBytes(64, 513, 512, 256, 8, fp16) == 0, "unaligned TMA shape");
        Require(FastllmCudaMoe16PackedBytes(64, 2048, 512, 256, 0, fp16) == 0, "invalid TMA topk");
        std::cout << "Native FP16/BF16 MoE: CPU reference, dynamic graph routes and guards PASS" << std::endl;
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
