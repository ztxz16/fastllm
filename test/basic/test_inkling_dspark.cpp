// Exercise production proposal sampling and verification. Interpose just the
// sampler outputs to compare two continuations with different rejected suffixes.
#include "model.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace fastllm;

namespace {
int keep, draftPosition, suffix, expectedSteps = 7;
bool controlled = false;
int jointVerifications = 0;
int verifyLaunches = 0;
int capturedDraws = 0;
int *forcedDraws = nullptr;
cudaGraphExec_t proposalGraph = nullptr;
std::vector<std::vector<float>> verifyScores;
} // namespace

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    static const auto launch = reinterpret_cast<decltype(&cudaGraphLaunch)>(dlsym(RTLD_NEXT, "cudaGraphLaunch"));
    if (capturedDraws) {
        proposalGraph = graph;
        capturedDraws = 0;
    }
    if (graph == proposalGraph && controlled) {
        int draws[7];
        for (int i = 0; i < 7; ++i) draws[i] = 128 + i * 13 + (i >= keep ? suffix : 0);
        auto status = cudaMemcpyAsync(forcedDraws, draws, sizeof(draws), cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) return status;
        draftPosition = expectedSteps;
    } else if (controlled) ++verifyLaunches;
    return launch(graph, stream);
}

extern "C" bool FastllmCudaMtpSampleDraftDeviceTopKFirst(float *scores, float *q, float temperature, int topK, float topP,
                                                int *out, int vocab, Data &workspace) {
    static const auto sample = reinterpret_cast<decltype(&FastllmCudaMtpSampleDraftDeviceTopKFirst)>(
        dlsym(RTLD_NEXT, "FastllmCudaMtpSampleDraftDeviceTopKFirst"));
    if (!sample(scores, q, temperature, topK, topP, out, vocab, workspace))
        return false;
    if (controlled) {
        cudaStreamCaptureStatus capture;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess) return false;
        if (capture == cudaStreamCaptureStatusActive) {
            return cudaMemcpyAsync(out, forcedDraws + capturedDraws++, sizeof(int),
                                   cudaMemcpyDeviceToDevice, cudaStreamPerThread) == cudaSuccess;
        }
        const int index = draftPosition++ % expectedSteps;
        const int token = 128 + index * 13 + (index >= keep ? suffix : 0);
        return cudaMemcpy(out, &token, sizeof(token), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    return true;
}

extern "C" bool FastllmCudaMtpRejectionSamplingTopKFirst(float *p, float *q, const float *temperatures, const int *k,
                                                         const float *topP, const int *ids, int *out, int *accepted,
                                                         int batch, int steps, int vocab) {
    if (controlled) {
        if (steps != expectedSteps || draftPosition != batch * steps)
            return false;
        if (suffix == 0 && keep == steps) {
            verifyScores.assign(batch * (steps + 1), std::vector<float>(vocab));
            for (int row = 0; row < batch * (steps + 1); ++row)
                if (cudaMemcpy(verifyScores[row].data(), p + size_t(row) * vocab, vocab * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        }
        for (int b = 0; b < batch; ++b) {
            std::copy_n(ids + b * steps, keep, out + b * (steps + 1));
            out[b * (steps + 1) + keep] = 512;
            accepted[b] = keep;
        }
        return true;
    }
    jointVerifications += batch > 1;
    static const auto reject = reinterpret_cast<decltype(&FastllmCudaMtpRejectionSamplingTopKFirst)>(
        dlsym(RTLD_NEXT, "FastllmCudaMtpRejectionSamplingTopKFirst"));
    return reject(p, q, temperatures, k, topP, ids, out, accepted, batch, steps, vocab);
}

extern "C" bool FastllmCudaMtpRejectionSamplingTopKFirstDeviceIds(float *p, float *q, const float *t,
        const int *k, const float *topP, const int *ids, int *out, int *accepted, int batch, int steps, int vocab) {
    if (controlled) {
        std::vector<int> host(batch * steps);
        if (cudaMemcpy(host.data(), ids, host.size() * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        return FastllmCudaMtpRejectionSamplingTopKFirst(p, q, t, k, topP, host.data(), out, accepted, batch, steps, vocab);
    }
    static const auto reject = reinterpret_cast<decltype(&FastllmCudaMtpRejectionSamplingTopKFirstDeviceIds)>(
        dlsym(RTLD_NEXT, "FastllmCudaMtpRejectionSamplingTopKFirstDeviceIds"));
    return reject(p, q, t, k, topP, ids, out, accepted, batch, steps, vocab);
}

static int step(basellm &model, ResponseContext &ctx, int pos, const std::vector<int> &ids, const GenerationConfig &cfg,
                std::vector<float> *logits = nullptr) {
    std::vector<float> tokens(ids.begin(), ids.end()), positions(tokens.size());
    std::iota(positions.begin(), positions.end(), float(pos));
    Data input(FLOAT32, {1, int(ids.size())}, tokens), position(FLOAT32, input.dims, positions);
    return model.Forward(input, Data(), position, ctx.pastKeyValues, cfg, LastTokensManager(1, 0), logits);
}

static double relativeL2(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.empty() || a.size() != b.size())
        throw std::runtime_error("Missing target logits");
    double diff = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += std::pow(double(a[i]) - b[i], 2);
        norm += double(b[i]) * b[i];
    }
    return std::sqrt(diff / norm);
}

static std::vector<uint8_t> committedKV(const ResponseContext &ctx, int end, int first, int count) {
    std::vector<uint8_t> result;
    for (const auto &slot : ctx.pastKeyValues) {
        for (const auto *side : {&slot.first, &slot.second}) {
            const size_t bytes = size_t(side->dims.at(0)) * side->dims.at(2) * side->unitSize;
            for (int pos = first; pos < first + count; ++pos) {
                const int back = (end - 1) / side->pageLen - pos / side->pageLen;
                const int page = side->pageIndex.at(side->pageIndex.size() - 1 - back);
                const size_t at = size_t(page) * side->pageLen + pos % side->pageLen;
                const size_t offset = result.size();
                result.resize(offset + bytes);
                auto *pool = static_cast<const uint8_t *>(side->pagedKVCacheData->cudaData);
                if (cudaMemcpy(result.data() + offset, pool + at * bytes, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                    throw std::runtime_error("Failed to read committed KV");
            }
        }
    }
    return result;
}

static void checkBatch(basellm &model, const GenerationConfig &sample) {
    ResponseContext ctx[2];
    std::vector<std::pair<Data *, Data *>> slots;
    std::vector<int> pos{23, 31}, token;
    for (int b = 0; b < 2; ++b) {
        ctx[b].pastKeyValues.resize(model.block_cnt);
        for (auto &kv : ctx[b].pastKeyValues)
            slots.emplace_back(&kv.first, &kv.second);
        auto cfg = sample;
        cfg.input_token_length = pos[b];
        token.push_back(step(model, ctx[b], 0, std::vector<int>(pos[b], 93 + b), cfg));
    }
    for (int i = 0; i < 10; ++i) {
        Data x(FLOAT32, {1, 2}, std::vector<float>{float(token[0]), float(token[1])});
        Data p(FLOAT32, {1, 1}, std::vector<float>{float(pos[0])});
        Data q(FLOAT32, {1, 1}, std::vector<float>{float(pos[1])});
        token = model.ForwardV2(2, x, {}, {&p, &q}, {1, 1}, slots, {sample, sample}, LastTokensManager(2, 0), nullptr);
        for (int b = 0; b < 2; ++b) {
            ++pos[b];
            // Alternate between consuming scheduler bursts and legacy single-token reads.
            if (i % 2 == 0 || b == 1)
                while (model.PopPendingResponseToken(&ctx[b], token[b], token[b]))
                    ++pos[b];
        }
    }
    if (!jointVerifications)
        throw std::runtime_error("The target failed to verify two requests in one batch");
    std::cout << "joint_verifications=" << jointVerifications << std::endl;
}

static void checkClippedRound(basellm &model, GenerationConfig sample, int steps, int batch, bool graph) {
    std::vector<ResponseContext> contexts(batch);
    std::vector<std::pair<Data *, Data *>> slots;
    std::vector<Data> positions;
    std::vector<Data *> positionPtrs;
    std::vector<GenerationConfig> configs(batch, sample);
    std::vector<float> anchors(batch);
    positions.reserve(batch);
    for (int b = 0; b < batch; ++b) {
        const int past = 1022 + b;
        auto &cfg = configs[b];
        cfg.input_token_length = past;
        cfg.output_token_limit = steps + 2;
        auto &ctx = contexts[b];
        ctx.pastKeyValues.resize(model.block_cnt);
        step(model, ctx, 0, std::vector<int>(past, 320 + b), cfg);
        for (auto &kv : ctx.pastKeyValues) slots.emplace_back(&kv.first, &kv.second);
        anchors[b] = 235 + b;
        positions.emplace_back(FLOAT32, std::vector<int>{1, 1}, std::vector<float>{float(past)});
        positionPtrs.push_back(&positions.back());
    }
    keep = expectedSteps = steps;
    suffix = draftPosition = verifyLaunches = 0;
    controlled = true;
    Data input(FLOAT32, {1, batch}, anchors);
    model.ForwardV2(batch, input, {}, positionPtrs, std::vector<int>(batch, 1), slots, configs,
                    LastTokensManager(batch, 0), nullptr);
    controlled = false;
    if (verifyLaunches != int(graph) || int(verifyScores.size()) != batch * (steps + 1))
        throw std::runtime_error("Clipped verification did not use the expected target path");
    for (int b = 0; b < batch; ++b) {
        auto teacher = configs[b];
        teacher.output_logits = true;
        ResponseContext ordinary;
        ordinary.pastKeyValues.resize(model.block_cnt);
        const int past = teacher.input_token_length;
        step(model, ordinary, 0, std::vector<int>(past, 320 + b), teacher);
        for (int r = 0; r <= steps; ++r) {
            std::vector<float> logits;
            step(model, ordinary, past + r, {r ? 128 + (r - 1) * 13 : int(anchors[b])}, teacher, &logits);
            if (logits != verifyScores[b * (steps + 1) + r])
                throw std::runtime_error("Clipped verification logits disagree with ordinary decode");
        }
        const int end = past + steps + 1;
        if (committedKV(ordinary, end, past, steps + 1) != committedKV(contexts[b], end, past, steps + 1))
            throw std::runtime_error("Clipped verification committed incorrect KV");
    }
    std::cout << "clipped_steps=" << steps << " batch=" << batch << " target_graphs=" << verifyLaunches << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 2 && (argc != 3 || std::string(argv[2]) != "--graph"))
        return 77;
    setenv("FASTLLM_DSPARK_TOKENS", "7", 1);
    SetDeviceMap({{"cuda:0", 1}});
    SetMoeDeviceMap({{"cuda:0", 1}});
    SetMaxTokens(8192);
    SetThreads(4);
    const bool graph = argc == 3;
    SetCudaGraph(graph);
    if (cudaMalloc(&forcedDraws, 7 * sizeof(int)) != cudaSuccess) return 1;
    auto model = CreateLLMModelFromHF(argv[1], BFLOAT16);
    model->maxBatch = 3;
    const std::vector<int> prompt(1022, 320);
    GenerationConfig sample;
    sample.do_sample = true;
    sample.top_k = 40;
    sample.top_p = .95f;
    sample.temperature = 1;
    sample.input_token_length = prompt.size();
    sample.output_token_limit = 512;
    auto teacher = sample;
    teacher.output_logits = true;
    for (keep = 0; keep <= 7; ++keep) {
        std::vector<float> results[3];
        std::vector<uint8_t> acceptedKV;
        std::vector<int> committed{235};
        for (int i = 0; i < keep; ++i)
            committed.push_back(128 + i * 13);
        int expectedPosition = int(prompt.size() + committed.size());
        for (int run = 0; run < 4; ++run) {
            ResponseContext ctx;
            ctx.pastKeyValues.resize(model->block_cnt);
            step(*model, ctx, 0, prompt, sample);
            if (run == 3) {
                // Same-width ordinary target. Its committed KV is an independent, bitwise oracle.
                auto full = committed;
                while (full.size() < 8)
                    full.push_back(128 + 13 * (full.size() - 1));
                step(*model, ctx, prompt.size(), full, teacher);
                if (acceptedKV != committedKV(ctx, prompt.size() + full.size(), prompt.size(), committed.size()))
                    throw std::runtime_error("Committed KV differs from the ordinary target");
                continue;
            }
            if (run == 2) {
                step(*model, ctx, prompt.size(), committed, teacher);
            } else {
                draftPosition = 0;
                suffix = run * 4096;
                controlled = true;
                int token = step(*model, ctx, prompt.size(), {committed[0]}, sample);
                std::vector<int> emitted{token};
                while (model->PopPendingResponseToken(&ctx, token, token))
                    emitted.push_back(token);
                controlled = false;
                std::vector<int> expected(committed.begin() + 1, committed.end());
                expected.push_back(512);
                if (emitted != expected)
                    throw std::runtime_error("Invalid emitted prefix or correction");
            }
            if (run == 0)
                acceptedKV = committedKV(ctx, expectedPosition, prompt.size(), committed.size());
            for (const auto &kv : ctx.pastKeyValues) {
                for (auto *side : {&kv.first, &kv.second}) {
                    if (side->lastPageLen != (expectedPosition - 1) % side->pageLen + 1)
                        throw std::runtime_error("Rejected suffix advanced KV");
                }
            }
            step(*model, ctx, expectedPosition, {512}, teacher, &results[run]);
        }
        const double suffixL2 = relativeL2(results[0], results[1]);
        const double teacherL2 = relativeL2(results[0], results[2]);
        std::cout << "accepted=" << keep << " changed_suffix_l2=" << suffixL2 << " teacher_l2=" << teacherL2
                  << std::endl;
        if (results[0] != results[1] || results[0] != results[2])
            throw std::runtime_error("Committed target state disagrees with replay");
    }
    // An independently initialized request has no draft cache and takes normal singleton decode.
    {
        ResponseContext ordinary;
        ordinary.pastKeyValues.resize(model->block_cnt);
        step(*model, ordinary, 0, prompt, teacher);
        for (int row = 0; row < 8; ++row) {
            std::vector<float> logits;
            step(*model, ordinary, int(prompt.size()) + row, {row ? 128 + (row - 1) * 13 : 235}, teacher, &logits);
            double error = relativeL2(verifyScores.at(row), logits);
            std::cout << "verify_row=" << row << " ordinary_l2=" << error << std::endl;
            if (verifyScores[row] != logits) throw std::runtime_error("Verify logits disagree with ordinary decode");
        }
    }
    checkBatch(*model, sample);
    for (int steps : {1, 3, 6})
        for (int batch : {1, 2})
            checkClippedRound(*model, sample, steps, batch, graph);
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess)
        throw std::runtime_error("Cannot query draft graph support");
    if (graph && major >= 8 && !proposalGraph) throw std::runtime_error("Missing full draft graph");
    cudaFree(forcedDraws);
}
