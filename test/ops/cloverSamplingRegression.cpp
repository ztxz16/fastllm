#include "fastllm.h"
#include "models/clover_sampling.h"
#include "devices/cuda/fastllm-clover.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
using namespace fastllm;
namespace {
void Check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct Buffer {
    void *p = nullptr;
    explicit Buffer(size_t n) { Check(cudaMalloc(&p, n)); }
    ~Buffer() { cudaFree(p); }
    void Put(const void *src, size_t n) { Check(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)); }
};
void Close(float actual, float expected, float atol, float rtol, const char *what) {
    if (actual == expected) return;
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > atol + rtol * std::abs(expected)) {
        std::cerr << what << " expected=" << expected << " actual=" << actual << std::endl;
        throw std::runtime_error(what);
    }
}

// CPU references stay in the test binary; production penalties/acceptance run on CUDA.
void ReferencePenalize(float *logits, int rows, int vocab,
        const std::vector<int> &counts, int responseLength,
        const std::vector<int> &visible, int maskToken, const std::vector<int> &stops,
        const GenerationConfig &config, const CloverOptions &o, bool preview, bool initial) {
    auto bias = CloverRowBias(rows, vocab, counts, responseLength, visible, o, preview, initial);
    const int firstPosition = responseLength + (o.speculative && preview ? 0 : 1);
    for (int row = 0; row < rows; ++row) {
        float *scores = logits + (size_t)row * vocab;
        for (int token = 0; token < vocab; ++token) {
            if (!counts[token]) continue;
            // Preserve the reference's separate FP32 additive rounding steps.
            volatile float shift = float(counts[token]) * o.frequency;
            float value = scores[token] - shift;
            value -= o.presence;
            if (config.repeat_penalty != 1.f)
                value = value < 0.f ? value * config.repeat_penalty : value / config.repeat_penalty;
            scores[token] = value;
        }
        const int position = firstPosition + row;
        if (bias[row * 2]) for (int stop : stops) scores[stop] += bias[row * 2];
        if (bias[row * 2 + 1]) scores[o.thinkToken] += bias[row * 2 + 1];
        if (position <= config.output_token_least)
            for (int stop : stops) scores[stop] = -std::numeric_limits<float>::infinity();
        // SGLang's self-speculative draft/verify does not mask this vocabulary
        // entry. HF low-confidence explicitly excludes it in every round.
        if (!o.speculative && maskToken >= 0 && maskToken < vocab)
            scores[maskToken] = -std::numeric_limits<float>::infinity();
    }
}
int ReferenceAcceptedPrefix(const float *logits, int rows, int vocab,
        const std::vector<int> &candidates, const CloverOptions &o) {
    int correct = 0;
    for (; correct + 1 < rows; ++correct) {
        const float *scores = logits + (size_t)correct * vocab;
        const int best = std::max_element(scores, scores + vocab) - scores;
        const int proposal = candidates[correct + 1];
        if (o.exact) { if (proposal != best) break; else continue; }
        double total = 0.;
        for (int i = 0; i < vocab; ++i) total += std::exp(scores[i] - scores[best]);
        double entropy = 0.;
        for (int i = 0; i < vocab; ++i) {
            float p = float(std::exp(scores[i] - scores[best]) / total);
            entropy -= p * std::log(std::max(p, 1.e-12f));
        }
        float probability = float(std::exp(scores[proposal] - scores[best]) / total);
        float threshold = std::min(o.acceptRatio, o.acceptDelta * std::exp(float(-entropy)));
        if (!(probability > 0. && probability >= threshold)) break;
    }
    return correct;
}

void Run(int rows, int vocab, bool speculative, bool preview, bool initial, int position) {
    std::mt19937 random(73 + rows + position);
    std::normal_distribution<float> normal(0.f, 4.f);
    std::vector<float> input((size_t)rows * vocab), gpu(input.size());
    for (auto &v : input) v = normal(random);
    std::vector<int> counts(vocab), stops{0, vocab - 1}, visible(rows, vocab / 2), candidates(rows);
    for (auto &v : counts) v = random() % 5;
    for (auto &v : candidates) v = random() % vocab;
    counts[vocab - 2] = 0;
    GenerationConfig config;
    config.do_sample = false; config.repeat_penalty = 1.05f; config.output_token_least = 2;
    CloverOptions options;
    options.speculative = speculative; options.presence = .9f; options.frequency = .13f;
    options.thinkToken = vocab - 2; options.thinkStart = 3; options.thinkWindow = 7;
    options.lengthStart = 5; options.lengthWindow = 9;
    Buffer x(input.size() * 4), c(counts.size() * 4), stop(stops.size() * 4),
        bias(rows * 2 * 4), proposal(rows * 4), output(rows * sizeof(CloverCudaTokenStats));
    const size_t workspaceBytes = FastllmCudaCloverTokenStatsWorkspaceBytes(rows, vocab);
    Buffer workspace(workspaceBytes ? workspaceBytes : 32), fallback(rows * sizeof(CloverCudaTokenStats));
    stop.Put(stops.data(), stops.size() * 4); proposal.Put(candidates.data(), rows * 4);
    CloverCudaPenalty p{config.repeat_penalty, options.presence, options.frequency,
        speculative ? -1 : vocab / 2, options.thinkToken,
        position + (speculative && preview ? 0 : 1), config.output_token_least};
    auto launch = [&]() {
        Require(FastllmCudaCloverPenalize((float *)x.p, (int *)c.p, (int *)stop.p,
            stops.size(), (float *)bias.p, rows, vocab, p), "GPU penalties failed");
        Require(FastllmCudaCloverTokenStats((float *)x.p, (int *)proposal.p,
            (CloverCudaTokenStats *)output.p, rows, vocab, true, true, nullptr,
            workspaceBytes ? workspace.p : nullptr), "GPU summary failed");
        Require(FastllmCudaCloverTokenStats((float *)x.p, (int *)proposal.p,
            (CloverCudaTokenStats *)fallback.p, rows, vocab, true, true), "GPU fallback summary failed");
    };
    auto prepare = [&]() {
        auto b = CloverRowBias(rows, vocab, counts, position, visible, options, preview, initial);
        x.Put(input.data(), input.size() * 4); c.Put(counts.data(), counts.size() * 4);
        bias.Put(b.data(), b.size() * 4);
    };
    auto verify = [&]() {
        std::vector<CloverCudaTokenStats> stats(rows);
        Check(cudaMemcpy(stats.data(), output.p, stats.size() * sizeof(stats[0]), cudaMemcpyDeviceToHost));
        std::vector<CloverCudaTokenStats> baseline(rows);
        Check(cudaMemcpy(baseline.data(), fallback.p, baseline.size() * sizeof(baseline[0]), cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(gpu.data(), x.p, gpu.size() * 4, cudaMemcpyDeviceToHost));
        auto expected = input;
        ReferencePenalize(expected.data(), rows, vocab, counts, position, visible, vocab / 2,
            stops, config, options, preview, initial);
        for (size_t i = 0; i < gpu.size(); ++i) Close(gpu[i], expected[i], 2.e-6f, 2.e-7f, "penalty");
        std::mt19937_64 rng(1);
        for (int row = 0; row < rows; ++row) {
            float confidence;
            auto *scores = expected.data() + (size_t)row * vocab;
            int best = CloverSample(scores, vocab, config, rng, &confidence);
            Require(stats[row].error == 0 && stats[row].token == best, "argmax differs");
            Require(stats[row].error == baseline[row].error && stats[row].token == baseline[row].token,
                "split argmax differs from fallback");
            Close(stats[row].confidence, baseline[row].confidence, 2.e-7f, 2.e-6f, "split confidence");
            Close(stats[row].probability, baseline[row].probability, 2.e-7f, 2.e-6f, "split probability");
            Close(stats[row].entropy, baseline[row].entropy, 2.e-6f, 2.e-6f, "split entropy");
            Close(stats[row].confidence, confidence, 2.e-7f, 2.e-6f, "confidence");
            double total = 0., entropy = 0.;
            for (int i = 0; i < vocab; ++i) total += std::exp(scores[i] - scores[best]);
            for (int i = 0; i < vocab; ++i) {
                float probability = float(std::exp(scores[i] - scores[best]) / total);
                entropy -= probability * std::log(std::max(probability, 1.e-12f));
            }
            int proposed = row + 1 < rows ? candidates[row + 1] : best;
            float probability = float(std::exp(scores[proposed] - scores[best]) / total);
            Close(stats[row].probability, probability, 2.e-7f, 2.e-6f, "proposal probability");
            Close(stats[row].entropy, float(entropy), 2.e-6f, 2.e-6f, "entropy");
        }
        for (bool exact : {false, true}) {
            options.exact = exact;
            int accepted = 0;
            while (accepted + 1 < rows) {
                auto &s = stats[accepted];
                bool ok = exact ? candidates[accepted + 1] == s.token : s.probability > 0.f &&
                    s.probability >= std::min(options.acceptRatio, options.acceptDelta * std::exp(-s.entropy));
                if (!ok) break;
                ++accepted;
            }
            Require(accepted == ReferenceAcceptedPrefix(expected.data(), rows, vocab, candidates, options),
                "accepted prefix differs");
        }
    };
    prepare(); launch(); verify();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    launch();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    // Refresh response counts and visible-thinking state after capture.
    counts[vocab - 2] = 1; visible[0] = vocab - 2;
    for (auto &v : input) v = normal(random);
    prepare(); Check(cudaGraphLaunch(executable, cudaStreamPerThread)); verify();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
void InvalidAndTies(int vocab, bool confidence, bool entropy) {
    std::vector<float> logits(vocab, 0.f); logits[17] = logits[18] = 4.f;
    Buffer x(logits.size() * 4), out(sizeof(CloverCudaTokenStats));
    const size_t bytes = FastllmCudaCloverTokenStatsWorkspaceBytes(1, vocab);
    Buffer workspace(bytes ? bytes : 32);
    for (int test = 0; test < 4; ++test) {
        if (test == 1) logits[0] = std::numeric_limits<float>::quiet_NaN();
        if (test == 2) logits[0] = std::numeric_limits<float>::infinity();
        if (test == 3) std::fill(logits.begin(), logits.end(), -std::numeric_limits<float>::infinity());
        x.Put(logits.data(), logits.size() * 4);
        Require(FastllmCudaCloverTokenStats((float *)x.p, nullptr, (CloverCudaTokenStats *)out.p,
            1, logits.size(), confidence, entropy, nullptr, bytes ? workspace.p : nullptr), "summary launch failed");
        CloverCudaTokenStats result;
        Check(cudaMemcpy(&result, out.p, sizeof(result), cudaMemcpyDeviceToHost));
        Require(test == 0 ? result.error == 0 && result.token == 17 : result.error != 0,
            "NaN/infinity/empty-candidate/tie handling differs");
    }
}
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        Check(cudaSetDevice(0));
        for (int rows : {1, 16, 32}) for (bool spec : {false, true}) for (bool preview : {false, true}) {
            Run(rows, 513, spec, preview, false, 4);
            Run(rows, 31, spec, preview, true, 0);
        }
        Run(16, 248320, true, false, false, 8192);
        Run(32, 248320, false, true, false, 4096);
        for (int vocab : {513, 248320}) for (bool confidence : {false, true})
            for (bool entropy : {false, true}) InvalidAndTies(vocab, confidence, entropy);
        std::cout << "Clover GPU sampling: penalties, argmax, confidence, acceptance, refreshed graph and errors PASS" << std::endl;
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
