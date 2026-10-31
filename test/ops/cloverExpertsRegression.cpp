#include "devices/cuda/fastllm-clover.h"
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
void Router(int tokens, int experts, int topk, float topP, bool ties, int stride = 0) {
    if (!stride) stride = experts;
    std::mt19937 rng(tokens + 123 * experts);
    std::normal_distribution<float> normal(0.f, 3.f);
    std::vector<uint16_t> logits(tokens * experts);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = Round16(ties ? float(i % experts % 5 == 0) : normal(rng));
    Buffer x((size_t)tokens * stride * 2), ids(tokens * topk * 4), scores(tokens * topk * 4),
        tempIds(tokens * topk * 4), tempScores(tokens * topk * 4), mask(experts);
    auto launch = [&]() {
        Require(FastllmCudaCloverRoute(x.p, (int *)ids.p, (float *)scores.p, (int *)tempIds.p,
            (float *)tempScores.p, (unsigned char *)mask.p, 1, tokens, experts, topk, topP, nullptr, stride, fp16), "router launch");
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
        std::vector<uint16_t> packed((size_t)tokens * stride, Round16(123.f));
        for (int row = 0; row < tokens; ++row)
            std::copy_n(logits.begin() + row * experts, experts, packed.begin() + row * stride);
        x.Put(packed);
    };
    upload(); launch(); reference();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (auto &v : logits) v = Round16(normal(rng));
    upload(); Check(cudaGraphLaunch(executable, cudaStreamPerThread)); reference();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
void ProjectionUnpack(int rows, int channels, int heads, bool preview, int inputOffset = 0) {
    const int values = heads * 128, width = channels + values + heads * 2;
    std::mt19937 rng(132 + rows); std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<uint16_t> input(rows * width);
    std::vector<float> aLog(heads), bias(heads);
    for (auto &v : input) v = Round16(normal(rng));
    for (auto &v : aLog) v = normal(rng);
    for (auto &v : bias) v = normal(rng);
    Buffer x((input.size() + inputOffset) * 2), a(heads * 4), b(heads * 4), qkv(rows * channels * 2),
        z(rows * values * 2), g(rows * heads * 4), beta(rows * heads * 2);
    a.Put(aLog); b.Put(bias);
    auto launch = [&]() {
        Require(FastllmCudaCloverUnpackGdn((uint16_t *)x.p + inputOffset, (float *)a.p, (float *)b.p, qkv.p, z.p,
            (float *)g.p, preview ? nullptr : beta.p, rows, channels, heads, nullptr, fp16), "projection unpack launch");
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
                    Close(Float(actualBeta[row * heads + h]), Float(Round16(1.f / (1.f + std::exp(-bv)))), 0.f, .008f);
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
    for (auto &v : input) v = Round16(normal(rng) * 20.f);
    upload(); Check(cudaGraphLaunch(executable, cudaStreamPerThread)); reference();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}

}
int main(int argc, char **argv) {
    fp16 = argc > 1 && std::string(argv[1]) == "fp16";
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        Check(cudaSetDevice(0));
        for (int tokens : {1, 4, 16, 32}) for (float p : {0.f, .5f, .9f}) for (bool ties : {false, true}) Router(tokens, 256, 8, p, ties);
        for (float p : {0.f, .9f}) for (bool ties : {false, true})
            Router(16, 256, 8, p, ties, 320);
        Router(16, 17, 16, .9f, false); Router(8, 16, 4, .9f, true);
        for (bool ties : {false, true}) {
            Router(16, 256, 8, 0.f, ties, 257);
            Router(16, 256, 8, 1.e-5f, ties, 257);
            Router(1, 1, 1, 0.f, ties); Router(4, 2, 2, .5f, ties);
        }
        for (int rows : {1, 16, 512}) for (bool preview : {false, true}) ProjectionUnpack(rows, 8192, 32, preview);
        ProjectionUnpack(7, 383, 3, false);
        for (int rows : {127, 128, 129}) for (bool preview : {false, true})
            ProjectionUnpack(rows, 8192, 32, preview);
        ProjectionUnpack(129, 8192, 32, false, 1);
        ProjectionUnpack(129, 383, 3, false);
        ProjectionUnpack(129, 1024, 3, true);
        std::cout << "Clover router/projections: CPU reference and graph refresh PASS" << std::endl;
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
