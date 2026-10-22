#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#include "devices/cpu/computeutils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
using namespace fastllm;
static void Require(bool ok, const char *msg) { if (!ok) throw std::runtime_error(msg); }
static void Cu(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
static float Bf(float x) { return RoundFloat32ToBFloat16RNE(x); }
static float Fp8(int code) {
    int e = code >> 3, m = code & 7;
    return e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
}
static void Quant(std::vector<float> &x) {
    for (size_t s = 0; s < x.size(); s += 32) {
        float maximum = 1e-4f;
        for (size_t j = s; j < s + 32; ++j) maximum = std::max(maximum, std::abs(x[j]));
        float scale = std::exp2(std::ceil(std::log2(maximum / 448.0f)));
        for (size_t j = s; j < s + 32; ++j) {
            float value = std::abs(x[j] / scale), distance = INFINITY;
            int best = 0;
            for (int code = 0; code <= 126; ++code) {
                float d = std::abs(value - Fp8(code));
                if (d < distance || (d == distance && !(code & 1))) { best = code; distance = d; }
            }
            x[j] = std::copysign(Fp8(best) * scale, x[j]);
        }
    }
}
static std::unique_ptr<Data> Weight(int rows, int cols, int scale) {
    auto w = std::make_unique<Data>(NVFP4, std::vector<int>{rows, cols});
    w->blockK = 1; w->blockM = 32; w->Allocate(false);
    std::memset(w->cpuData, 0, GetNVFP4WeightBytes(rows, cols));
    for (int row = 0; row < rows; ++row) {
        int col = row % cols;
        w->cpuData[(size_t(row) * cols + col) / 2] |= 2 << ((col & 1) * 4);
    }
    std::memset(GetNVFP4ScaleData(*w), scale, GetNVFP4ScaleBytes(rows, cols, 1, 32));
    w->directMemory = true;
    w->isModelWeight = true;
    return w;
}
static std::vector<float> Read(Data &root) {
    Data &x = root.IsTensorParallelReplicated() ? *root.multiDeviceDatas.at(0) : root;
    Require(x.dataType == FLOAT32, "hybrid MoE lost FP32 accumulation");
    Cu(cudaSetDevice(0)); Cu(cudaDeviceSynchronize());
    std::vector<float> out(x.Count(0));
    Cu(cudaMemcpy(out.data(), x.cudaData, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}
struct Scratch { Data w1, w2, w3, in, out; };
static void Run(Data &in, Data &ix, Data &sc, std::vector<Data *> &weights, Scratch &s, Data &out) {
    std::vector<Data *> bias(weights.size(), nullptr);
    ApplyDeviceMap({{"multicuda:0,1,2,3,4,5,6,7", 1}}, 0, 1);
    MergeMOE(in, ix, sc, weights, bias, s.w1, s.w2, s.w3, s.in, s.out,
             1.0f, out, 0, MoeGateSwiglu, true, 10.0f, true, nullptr, 32);
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 8) return 77;
    try {
        SetThreads(8);
        SetDeviceMap({{"multicuda:0,1,2,3,4,5,6,7", 1}});
        std::vector<int> ranks(8); std::iota(ranks.begin(), ranks.end(), 0);
        constexpr int H = 256, E = 16;
        int checks = 0;
        for (int firstRows : {1, 2, 5, 8}) for (int topk : {1, 3, 6}) {
            // Different layer identities also exercise retirement of the
            // persistent workspace when expert weights leave scope.
            std::vector<std::unique_ptr<Data>> owned;
            std::vector<Data *> weights{nullptr, nullptr};
            for (int e = 0; e < E; ++e) {
                owned.push_back(Weight(2 * H, H, 124)); weights.push_back(owned.back().get());
                owned.push_back(Weight(H, H, 123 + e % 3)); weights.push_back(owned.back().get());
            }
            Scratch candidateScratch, referenceScratch;
            void *stable = nullptr;
            // Shape changes use the conservative synchronous fallback. Return
            // to the original shape must reuse the original persistent buffer.
            for (int iteration = 0; iteration < 12; ++iteration) {
                int rows = iteration == 9 ? 9 : iteration == 10 ? (firstRows == 2 ? 3 : 2) : firstRows;
                std::vector<float> x(rows * H), scores(rows * topk), oracle(rows * H, 0.0f);
                std::vector<int32_t> ids(rows * topk);
                for (size_t j = 0; j < x.size(); ++j)
                    x[j] = Bf(std::ldexp(float(int((j * 17 + iteration * 11) % 79) - 39) / 7.0f, int(j / 32 % 4) - 2));
                for (int row = 0; row < rows; ++row) for (int k = 0; k < topk; ++k) {
                    int e = iteration % 4 == 0 ? (k * 8) % E : (row * 7 + k * 3 + iteration) % E;
                    ids[row * topk + k] = e;
                    float score = (iteration + k) % 5 == 0 ? 0.0f : (k % 2 ? .34f : 1.75f);
                    scores[row * topk + k] = score;
                    std::vector<float> a(x.begin() + row * H, x.begin() + (row + 1) * H);
                    Quant(a);
                    for (float &value : a) {
                        float z = Bf(value * .125f), gate = std::min(z, 10.0f), up = std::clamp(z, -10.0f, 10.0f);
                        value = Bf(score * ((gate / (1 + std::exp(-gate))) * up));
                    }
                    Quant(a);
                    for (int j = 0; j < H; ++j) oracle[row * H + j] += Bf(a[j] * std::ldexp(1.0f, -4 + e % 3));
                }
                Data input(BFLOAT16, {rows, H}), index(INT32, {rows, topk}), score(FLOAT32, {rows, topk}, scores);
                input.Allocate(); index.Allocate();
                for (size_t j = 0; j < x.size(); ++j) {
                    uint32_t bits; std::memcpy(&bits, &x[j], 4);
                    ((uint16_t *)input.cpuData)[j] = bits >> 16;
                }
                std::memcpy(index.cpuData, ids.data(), ids.size() * 4);
                for (Data *d : {&input, &index, &score}) d->ToDevice(DataDevice::CUDA, {0}, true);
                Data output;
                Run(input, index, score, weights, candidateScratch, output);
                auto actual = Read(output);
                if (rows == firstRows) {
                    Require(output.cudaDataBorrowed, "persistent hybrid path was not used");
                    if (!stable) stable = output.cudaData;
                    Require(output.cudaData == stable, "persistent output address moved");
                }
                Data refInput, refIndex, refScore, reference;
                refInput.CopyFrom(input); refIndex.CopyFrom(index); refScore.CopyFrom(score);
                for (Data *d : {&refInput, &refIndex, &refScore}) PrepareMultiCudaReplicatedData(*d, ranks, true);
                Run(refInput, refIndex, refScore, weights, referenceScratch, reference);
                Require(actual == Read(reference), "packed hybrid path differs from replicated EP reference");
                for (size_t j = 0; j < actual.size(); ++j) {
                    if (std::abs(actual[j] - oracle[j]) > 2e-6f * std::max(1.0f, std::abs(oracle[j]))) {
                        std::cerr << "rows=" << rows << " topk=" << topk << " iter=" << iteration
                                  << " at=" << j << " actual=" << actual[j] << " oracle=" << oracle[j] << '\n';
                        throw std::runtime_error("hybrid MoE independent quantization oracle mismatch");
                    }
                }
                if (output.cudaDataBorrowed) {
                    // The model adds its shared expert in FP32, then converts
                    // this borrowed persistent output to BF16 every draft.
                    ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
                    ToDataType(output, BFLOAT16);
                    Require(!output.cudaDataBorrowed, "converted output must own its storage");
                    Data temporary(BFLOAT16, {rows, H});
                    temporary.ToDevice(DataDevice::CUDA, std::vector<int>{0}); temporary.Allocate();
                }
                ++checks;
            }
        }
        std::cout << "PASS " << checks << " hybrid EP batches: exact replicated reference, CPU oracle, changed routes, empty ranks, shape fallback and stable buffers\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
