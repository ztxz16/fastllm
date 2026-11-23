#include "hy4_flash_ops.h"
#include "models/speculative_sampling.h"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
using namespace fastllm;
using namespace fastllm::hy4;

static void Require(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
static void CheckPageRollback() {
    // The scheduler reads both shapes before admitting the next decode. A
    // rejected cross-page verify keeps the page allocated for the next replay.
    KVPools pools;
    Data key, value;
    auto table = ReservePages(pools, 0, key, value, 127, 512, 16, 16);
    Require(key.expansionDims.at(1) == 128 && key.dims.at(1) == 127, "first KV page");
    Require(table == ReservePages(pools, 0, key, value, 131, 512, 16, 16), "page table changed");
    TruncateKV(key, value, 127);
    for (const auto *view : {&key, &value})
        Require(view->dims.at(1) == 127 && view->expansionDims.at(1) == 256, "rejected KV suffix");
    Require(table == ReservePages(pools, 0, key, value, 129, 512, 16, 16), "KV page reuse");
}

static void CheckGreedyVerification(Scratch &s) {
    constexpr int width = 4, vocab = 513;
    const std::vector<int> expected = {257, 5, 2, 100};
    std::vector<float> scores(width * vocab, -1.f);
    for (int row = 0; row < width; ++row) {
        scores[row * vocab + expected[row]] = 1.f;
        scores[row * vocab + 511] = 1.f;
    }
    auto target = s.Get("testTarget", scores.size());
    CopyToGPU(target, scores.data(), scores.size() * sizeof(float));
    for (int matched = 0; matched < width; ++matched) {
        auto candidates = expected;
        if (matched < width - 1)
            candidates[matched] = 511;
        auto tokens = VerifyDraft(s, target, width, vocab, GenerationConfig(), {}, candidates.data());
        Require(tokens == std::vector<int>(expected.begin(), expected.begin() + matched + 1),
                "greedy prefix or tie-breaking changed");
    }
}

static void CheckTargetAndRejection(Scratch &s) {
    std::mt19937 rng(3504);
    double maximumError = 0;
    int distributions = 0;
    for (int width : {2, 4, 10, 16}) {
        const int vocab = 1536;
        auto q = PrepareProposal(s, 0, 1, width - 1, vocab);
        auto target = s.Get("testTarget", width * vocab);
        std::vector<float> scores(width * vocab), proposal((width - 1) * vocab, -INFINITY), lse(width - 1);
        std::vector<int> selected(width - 1);
        for (int row = 0; row < width; ++row) {
            for (int i = 0; i < vocab; ++i)
                scores[row * vocab + i] = float(int(rng() % 70) - 30) / 8;
            if (row < width - 1) {
                selected[row] = row;
                proposal[row * vocab + row] = 0;
            }
        }
        CopyToGPU(target, scores.data(), scores.size() * sizeof(float));
        CopyToGPU(q.logits, proposal.data(), proposal.size() * sizeof(float));
        CopyToGPU(q.logsumexp, lse.data(), lse.size() * sizeof(float));
        CopyToGPU(q.tokens, selected.data(), selected.size() * sizeof(int));
        for (int k : {2, 20, 257, 1000, vocab})
            for (float topP : {0.f, .55f, .94f, 1.f}) {
                GenerationConfig cfg;
                cfg.top_k = k;
                cfg.top_p = topP;
                cfg.temperature = .83f;
                cfg.do_sample = true;
                auto probabilities = TargetProbabilities(s, target, width, vocab, cfg);
                std::vector<float> p(width * vocab);
                CopyToHost(p.data(), probabilities, p.size() * sizeof(float));
                for (int row = 0; row < width; ++row) {
                    auto ref =
                        SpeculativeDistribution(scores.data() + row * vocab, vocab, cfg, LastTokensUnit());
                    for (int i = 0; i < vocab; ++i)
                        maximumError = std::max(maximumError, double(std::abs(p[row * vocab + i] - ref[i])));
                    ++distributions;
                }
            }
        for (int reject = 0; reject < width; ++reject) {
            std::fill(scores.begin(), scores.end(), -INFINITY);
            for (int row = 0; row < width; ++row)
                scores[row * vocab + (row == reject ? 100 : row)] = 0;
            CopyToGPU(target, scores.data(), scores.size() * sizeof(float));
            GenerationConfig cfg;
            cfg.do_sample = true;
            cfg.top_k = vocab;
            cfg.top_p = .9f;
            auto out = VerifyDraft(s, target, width, vocab, cfg, q, selected.data());
            Require(int(out.size()) == reject + 1 && out.back() == 100,
                    "residual, bonus or segment boundary");
            for (int i = 0; i < reject; ++i)
                Require(out[i] == i, "accepted prefix changed");
        }
    }
    std::cout << "reference rows " << distributions << " maximum probability error " << maximumError
              << std::endl;
    Require(maximumError < 2e-6, "temperature, top-k, top-p or tie order differs");
}

static void CheckSamplingFrequencies(Scratch &s) {
    // Draw actual GPU proposals and verify against p; the two distributions overlap.
    const int vocab = 512, trials = 6000;
    auto q = PrepareProposal(s, 0, 1, 1, vocab);
    auto target = s.Get("testTarget", 2 * vocab), draft = s.Get("testDraft", vocab);
    std::vector<float> scores(2 * vocab, -INFINITY), qs(vocab, -INFINITY);
    const float pMass[] = {.2f, .45f, .35f}, qMass[] = {.6f, .3f, .1f};
    for (int i = 0; i < 3; ++i) {
        qs[i] = std::log(qMass[i]);
        scores[i] = scores[vocab + i] = std::log(pMass[i]);
    }
    CopyToGPU(target, scores.data(), scores.size() * sizeof(float));
    CopyToGPU(draft, qs.data(), qs.size() * sizeof(float));
    GenerationConfig cfg;
    cfg.top_k = 3;
    cfg.top_p = 1;
    cfg.do_sample = true;
    int observed[3] = {}, generated[3] = {}, bonus[3] = {}, accepted = 0;
    for (int i = 0; i < trials; ++i) {
        int token;
        Require(FastllmCudaMtpSampleDraftLogits(draft, q.logits, q.logsumexp, q.tokens, nullptr,
                                                &cfg.temperature, 1, vocab),
                "proposal failed");
        CopyToHost(&token, q.tokens, sizeof(int));
        Require(token >= 0 && token < 3, "invalid proposal");
        ++generated[token];
        auto out = VerifyDraft(s, target, 2, vocab, cfg, q, &token);
        Require(out[0] >= 0 && out[0] < 3, "invalid sample");
        ++observed[out[0]];
        if (out.size() == 2) {
            Require(out[0] == token && out[1] >= 0 && out[1] < 3, "invalid bonus or accepted proposal");
            ++accepted;
            ++bonus[out[1]];
        }
    }
    for (int i = 0; i < 3; ++i) {
        Require(std::abs(float(observed[i]) / trials - pMass[i]) < .03, "output distribution");
        Require(std::abs(float(generated[i]) / trials - qMass[i]) < .03, "draft distribution");
        Require(std::abs(float(bonus[i]) / accepted - pMass[i]) < .04, "bonus distribution");
    }
    Require(std::abs(float(accepted) / trials - .6f) < .03, "acceptance probability");
    std::cout << "trials " << trials << " output " << observed[0] << " " << observed[1] << " " << observed[2]
              << " accepted " << accepted << " bonus " << bonus[0] << " " << bonus[1] << " " << bonus[2]
              << std::endl;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    Select(0);
    CheckPageRollback();
    Scratch scratch(0);
    CheckGreedyVerification(scratch);
    CheckTargetAndRejection(scratch);
    CheckSamplingFrequencies(scratch);
}
