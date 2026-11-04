#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-cuda-sampling.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fastllm;
namespace {
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void Check(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }
struct Buffer {
    void *p = nullptr;
    explicit Buffer(size_t bytes) { Check(cudaMalloc(&p, bytes)); }
    ~Buffer() { cudaFree(p); }
    void Put(const void *src, size_t bytes) { Check(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice)); }
};
struct Probe : DeepSeekV41Model {
    void Mode(ExecutionMode mode) { executionMode = mode; }
};

// Stable scalar reference: FP32 logits/exp, FP64 probability sums. A large
// flat vocabulary exposes the old CPU sampler's sequential FP32 sum drift.
int Reference(const float *input, int vocab, const GenerationConfig &c,
              const LastTokensUnit &history, double uniform) {
    std::vector<std::pair<float, int>> ranked;
    float invTemp = 1.f / c.temperature;
    for (int t = 0; t < vocab; ++t) {
        if (!c.tool_call_allowed_token_ids.empty() &&
            !std::binary_search(c.tool_call_allowed_token_ids.begin(), c.tool_call_allowed_token_ids.end(), t)) continue;
        float x = input[t];
        int count = history.tokenSet.count(t);
        if (c.last_n <= 0) count = std::min(count, 1);
        if (std::abs(c.repeat_penalty - 1.f) > 1e-6f)
            while (count--) x = x < 0.f ? x * c.repeat_penalty : x / c.repeat_penalty;
        ranked.emplace_back(-x * invTemp, t);
    }
    std::sort(ranked.begin(), ranked.end());
    ranked.resize(std::min(ranked.size(), size_t(std::max(1, c.top_k))));
    std::vector<double> mass;
    double total = 0., kept = 0.;
    for (const auto &x : ranked) { mass.push_back(expf(ranked[0].first - x.first)); total += mass.back(); }
    size_t n = 0;
    for (; n < mass.size(); ++n) { kept += mass[n]; if (kept / total > c.top_p) { ++n; break; } }
    double prefix = 0.;
    for (size_t i = 0; i < n; ++i) { prefix += mass[i]; if (prefix > uniform * kept) return ranked[i].second; }
    return ranked[n - 1].second;
}

// Fixed draws compare with the scalar probability law, including exact-k ties,
// top-k BEFORE top-p, repeated-token multiplicity and independent row masks.
void Kernel(int vocab, int batch) {
    std::mt19937 rng(71);
    std::normal_distribution<float> normal(0.f, 2.f);
    std::vector<float> logits(size_t(batch) * vocab);
    std::vector<GenerationConfig> configs(batch);
    LastTokensManager history(batch, 8);
    std::vector<int> ids;
    std::vector<size_t> offsets(batch), repeatOffsets(batch);
    std::vector<FastllmCudaSamplingRow> rows(batch);
    for (int b = 0; b < batch; ++b) {
        auto &c = configs[b];
        c.top_k = b % 4 == 0 ? vocab : b % 4 == 1 ? 1 : b % 4 == 2 ? 7 : 2;
        c.top_p = b % 3 == 0 ? 1.f : b % 3 == 1 ? .95f : .6f;
        c.temperature = b % 2 ? .7f : 1.f;
        c.repeat_penalty = b % 3 ? 1.f : 1.2f;
        for (int t = 0; t < vocab; ++t) {
            logits[size_t(b) * vocab + t] = b % 5 ? normal(rng) : float(t % 4);
            if (b % 3 && t % (b % 7 + 2) == 0) c.tool_call_allowed_token_ids.push_back(t);
        }
        offsets[b] = ids.size();
        ids.insert(ids.end(), c.tool_call_allowed_token_ids.begin(), c.tool_call_allowed_token_ids.end());
        repeatOffsets[b] = ids.size();
        if (c.repeat_penalty != 1.f) {
            history.units[b].Push(0); history.units[b].Push(0); history.units[b].Push(2);
            ids.insert(ids.end(), {0, 0, 2});
        }
        rows[b].allowedCount = c.tool_call_allowed_token_ids.size();
        rows[b].repeatCount = ids.size() - repeatOffsets[b];
        rows[b].policy = FastllmCudaSamplingPolicy::LLMSampling;
        rows[b].temperature = c.temperature; rows[b].topP = c.top_p; rows[b].topK = c.top_k;
        rows[b].repeatPenalty = c.repeat_penalty;
    }
    Buffer input(logits.size() * sizeof(float)), metadata(std::max(size_t(1), ids.size()) * sizeof(int));
    Buffer output(batch * sizeof(FastllmCudaSamplingResult));
    size_t bytes = FastllmCudaSampleLogitsWorkspaceBytes(batch, vocab);
    Buffer scratch(bytes);
    input.Put(logits.data(), logits.size() * sizeof(float));
    if (!ids.empty()) metadata.Put(ids.data(), ids.size() * sizeof(int));
    for (int b = 0; b < batch; ++b) {
        rows[b].logits = (float *)input.p + size_t(b) * vocab;
        rows[b].allowedTokens = (int *)metadata.p + offsets[b];
        rows[b].repeatTokens = (int *)metadata.p + repeatOffsets[b];
    }
    auto launch = [&] {
        Require(FastllmCudaSampleLogits(rows.data(), batch, vocab,
            (FastllmCudaSamplingResult *)output.p, scratch.p, bytes), "kernel launch");
    };
    for (int pass = 0; pass < (vocab < 1000 ? 32 : 2); ++pass) {
        std::vector<int> expected(batch);
        for (int b = 0; b < batch; ++b) {
            unsigned seed = 719 + b * 31 + pass * 997;
            std::srand(seed);
            rows[b].uniform = std::min(double(float(std::rand() % 10001) * .0001f), std::nextafter(1., 0.));
            expected[b] = Reference(logits.data() + size_t(b) * vocab, vocab, configs[b], history.units[b], rows[b].uniform);
            if (vocab < 1000) {
                Data cpu(FLOAT32, {1, vocab}, std::vector<float>(logits.begin() + size_t(b) * vocab,
                    logits.begin() + size_t(b + 1) * vocab));
                std::srand(seed);
                Require(LLMSampling(cpu, 0, configs[b], history.units[b]) == expected[b], "small-vocabulary CPU reference");
            }
        }
        launch();
        std::vector<FastllmCudaSamplingResult> actual(batch);
        Check(cudaMemcpy(actual.data(), output.p, actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost));
        for (int b = 0; b < batch; ++b) {
            if (actual[b].error || actual[b].token != expected[b]) {
                std::cerr << "vocab=" << vocab << " batch=" << batch << " row=" << b
                          << " pass=" << pass << " expected=" << expected[b] << " actual=" << actual[b].token << '\n';
                throw std::runtime_error("masked sampling differs from scalar probability law");
            }
        }
    }
    // Replaying changed device masks must not retain another request's mask.
    if (batch > 1) {
        rows[1].allowedCount = 1;
        rows[1].allowedTokens = (int *)metadata.p;
        cudaGraph_t graph; cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch();
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int token : {vocab - 1, 0, vocab / 2}) {
            metadata.Put(&token, sizeof(int));
            Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            FastllmCudaSamplingResult actual;
            Check(cudaMemcpy(&actual, (FastllmCudaSamplingResult *)output.p + 1, sizeof(actual), cudaMemcpyDeviceToHost));
            Require(!actual.error && actual.token == token, "stale mask on graph replay");
        }
        Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
    }
    std::vector<float> unchanged(logits.size());
    Check(cudaMemcpy(unchanged.data(), input.p, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    Require(std::memcmp(unchanged.data(), logits.data(), logits.size() * sizeof(float)) == 0, "modified source logits");
}


void Block(bool benchmark) {
    const int batch = 64, vocab = benchmark ? 129280 : 257;
    const int device = FastllmCudaGetDevice();
    Probe model;
    std::vector<float> values(size_t(batch) * vocab);
    std::mt19937 rng(17); std::normal_distribution<float> normal(0.f, 2.f);
    for (float &x : values) x = normal(rng);
    std::vector<GenerationConfig> configs(batch);
    for (auto &c : configs) { c.top_k = vocab; c.top_p = .95f; }
    configs[0].tool_call_allowed_token_ids = {17, 23};
    LastTokensManager history(batch, 8);
    std::vector<std::pair<Data *, Data *>> past;
    std::vector<int> lengths(batch, 1);
    auto run = [&](ExecutionMode mode, bool cuda, bool outputLogits, bool preserve, bool fallback = false) {
        model.Mode(mode);
        Require(model.UseGpuToolCallConstraints() == (mode == ExecutionMode::Gpu), "pure GPU mode isolation");
        Data logits(FLOAT32, {1, batch, vocab}, values);
        if (cuda) {
            logits.ToDevice(DataDevice::CUDA, std::vector<int>{device});
            Require(GetPointerDeviceId(logits.cudaData) == device, "test logits placed on the wrong GPU");
        }
        auto c = configs;
        std::vector<std::vector<float>> returned(batch);
        std::vector<std::vector<float> *> pointers;
        for (int b = 0; b < batch; ++b) { c[b].output_logits = outputLogits; pointers.push_back(&returned[b]); }
        std::vector<int> sampled;
        std::srand(203);
        auto start = std::chrono::steady_clock::now();
        LLMSamplingBlock(&model, nullptr, nullptr, nullptr, 1e-6f, batch, true, lengths, past,
            c, history, &pointers, sampled, &logits, nullptr, nullptr, nullptr, preserve);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        Require(sampled.size() == size_t(batch), "batch result size");
        Require(sampled[0] == 17 || sampled[0] == 23, "mask bypassed");
        if (mode == ExecutionMode::Gpu && cuda && !outputLogits && !fallback) {
            std::srand(203);
            for (int b = 0; b < batch; ++b) {
                double uniform = std::min(double(float(std::rand() % 10001) * .0001f), std::nextafter(1., 0.));
                Require(sampled[b] == Reference(values.data() + size_t(b) * vocab, vocab, c[b], history.units[b], uniform),
                        "GPU block probability law");
            }
        }
        if (outputLogits) for (int b = 0; b < batch; ++b)
            Require(returned[b] == std::vector<float>(values.begin() + size_t(b) * vocab, values.begin() + size_t(b + 1) * vocab), "returned logits changed");
        if (preserve) {
            logits.ToDevice(DataDevice::CPU);
            Require(std::memcmp(logits.cpuData, values.data(), values.size() * sizeof(float)) == 0, "precomputed logits changed");
        }
        if (benchmark) std::cout << "batch=" << batch << " vocab=" << vocab << " mode=" << int(mode) << " cuda=" << cuda << " ms=" << ms << '\n';
        return sampled;
    };
    auto cpu = run(ExecutionMode::Cpu, false, false, false);
    auto hybrid = run(ExecutionMode::Hybrid, true, false, false);
    Require(cpu == hybrid, "hybrid sampler changed");
    auto gpu = run(ExecutionMode::Gpu, true, false, false);
    if (!benchmark) Require(cpu == gpu, "small-vocabulary GPU block differs from CPU");
    Require(run(ExecutionMode::Gpu, true, false, true) == gpu, "preserved graph logits changed sampling");
    if (!benchmark) {
        Require(run(ExecutionMode::Gpu, true, true, false) == cpu, "output-logits fallback");
        for (auto &c : configs) { c.repeat_penalty = 1.3f; c.last_n = 0; }
        for (auto &h : history.units) { h.Push(17); h.Push(17); h.Push(23); }
        Require(run(ExecutionMode::Gpu, true, false, false) == run(ExecutionMode::Hybrid, true, false, false), "unique repeat penalty");
        for (const std::vector<int> mask : {std::vector<int>{-1, vocab}, std::vector<int>{23, 17}}) {
            configs[1].tool_call_allowed_token_ids = mask;
            Require(run(ExecutionMode::Gpu, true, false, true, true) == run(ExecutionMode::Hybrid, true, false, true),
                    "invalid-mask fallback changed sampling");
        }
        configs[1].tool_call_allowed_token_ids.clear();
        configs[1].top_p = 0.f;
        Require(run(ExecutionMode::Gpu, true, false, false, true) == run(ExecutionMode::Hybrid, true, false, false),
                "unsupported-parameter fallback changed sampling");
    } else for (int i = 0; i < 5; ++i) run(ExecutionMode::Gpu, true, false, false);
}
}
int main(int argc, char **argv) {
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "bench") == 0;
    if (argc > 1 && !benchmark) {
        std::cerr << "usage: toolConstrainedSamplingRegression [bench]\n";
        return 2;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try {
        if (!benchmark) {
            for (int batch : {1, 2, 32, 64, 97}) Kernel(67, batch);
            Kernel(129280, 64);
        }
        Block(benchmark);
        if (!benchmark && count > 1) { Check(cudaSetDevice(1)); Block(false); }
        std::cout << "tool constrained GPU sampling PASS\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
