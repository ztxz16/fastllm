// Exercise production proposal sampling and verification. Interpose just the
// sampler outputs to compare two continuations with different rejected suffixes.
#include "model.h"
#include "moe_cache_config.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace fastllm;

void forceInklingDraft(const int *ctrl, int *out, int stride, int step, int capacity);
void forceInklingPrefix(int *ctrl, const int *tokens, int *result, int batch, int drafts, const int *live);

void snapshotInklingDraft(const int *, const void *, int, const void *, const void *, float, bool, const float *, int,
                          float *, int, int, int);

namespace {
int keep, suffix, verifyLaunches;
int drawIndex = 0;
bool controlled = false;
int *control = nullptr;
float *savedScores = nullptr, *savedDraft = nullptr;
const float *targetLogits = nullptr;
std::vector<std::vector<float>> verifyScores;
constexpr int maxTestBatch = 8;
int testVocab;
bool inDraw = false;

template <class F> F original(F current) {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(current), &info) || !info.dli_sname)
        throw std::runtime_error("Missing interposed symbol");
    auto fn = reinterpret_cast<F>(dlsym(RTLD_NEXT, info.dli_sname));
    if (!fn) throw std::runtime_error(info.dli_sname);
    return fn;
}

void configure(int batch, int a = -1, int b = -1, bool capture = false) {
    int params[] = {batch, keep, suffix, int(capture), a, b, -1};
    if (cudaMemcpy(control, params, sizeof(params), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Failed to stage test parameters");
    controlled = batch != 0;
}
void readScores(int batch, int steps) {
    verifyScores.assign(batch * (steps + 1), std::vector<float>(testVocab));
    for (int b = 0; b < batch; ++b)
        for (int r = 0; r <= steps; ++r)
            if (cudaMemcpy(verifyScores[b * (steps + 1) + r].data(), savedScores + size_t(b * 8 + r) * testVocab,
                           testVocab * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("Failed to inspect target logits");
}
} // namespace

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    static const auto launch = original(&cudaGraphLaunch);
    if (controlled) ++verifyLaunches;
    return launch(graph, stream);
}

extern "C" bool FastllmCudaMtpGraphDrawMarkov(void *ws, const void *base, int baseStride, const void *adjust,
                                              const void *bias, float cap, bool bf16, const float *t, int *k, const float *p,
                                              uint64_t *rng, int step, float *q, int qStride, int *out, int stride) {
    static const auto draw = original(&FastllmCudaMtpGraphDrawMarkov);
    inDraw = true;
    bool ok = draw(ws, base, baseStride, adjust, bias, cap, bf16, t, k, p, rng, step, q, qStride, out, stride);
    inDraw = false;
    int index = drawIndex++ % 7;
    snapshotInklingDraft(control, base, baseStride, adjust, bias, cap, bf16, q, qStride, savedDraft, index,
                         testVocab, maxTestBatch);
    forceInklingDraft(control, out, stride, index, maxTestBatch);
    return ok;
}

bool FastllmCudaMtpGraphDistribution(void *ws, float *logits, const float *t, int *k, const float *p, float **out) {
    static const auto normalize = original(&FastllmCudaMtpGraphDistribution);
    if (!inDraw) targetLogits = logits;
    return normalize(ws, logits, t, k, p, out);
}
extern "C" void FastllmCudaMtpGraphReject(const float *p, const float *q, const int *ids, int *result, void *partial,
                                          int batch, int drafts, int vocab, const int *live, uint64_t seed,
                                          unsigned long long *counter) {
    static const auto reject = original(&FastllmCudaMtpGraphReject);
    reject(p, q, ids, result, partial, batch, drafts, vocab, live, seed, counter);
    if (!targetLogits || batch > maxTestBatch || vocab != testVocab) throw std::runtime_error("Bad oracle geometry");
    if (cudaMemcpyAsync(savedScores, targetLogits, size_t(batch) * (drafts + 1) * vocab * sizeof(float),
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread) != cudaSuccess)
        throw std::runtime_error("Failed to snapshot logits");
    forceInklingPrefix(control, ids, result, batch, drafts, live);
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
    cudaMemset(control + 7, 0, sizeof(int));
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
    int jointVerifications = 0;
    cudaMemcpy(&jointVerifications, control + 7, sizeof(int), cudaMemcpyDeviceToHost);
    if (!jointVerifications)
        throw std::runtime_error("The target failed to verify two requests in one batch");
    std::cout << "joint_verifications=" << jointVerifications << std::endl;
}

static void checkClippedRound(basellm &model, GenerationConfig sample, int steps, int batch, bool graph,
                              bool mixed = false) {
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
        cfg.output_token_limit = (mixed && b == 0 ? 0 : steps) + 2;
        auto &ctx = contexts[b];
        ctx.pastKeyValues.resize(model.block_cnt);
        step(model, ctx, 0, std::vector<int>(past, 320 + b), cfg);
        for (auto &kv : ctx.pastKeyValues) slots.emplace_back(&kv.first, &kv.second);
        anchors[b] = 235 + b;
        positions.emplace_back(FLOAT32, std::vector<int>{1, 1}, std::vector<float>{float(past)});
        positionPtrs.push_back(&positions.back());
    }
    keep = steps;
    suffix = verifyLaunches = 0;
    configure(batch);
    Data input(FLOAT32, {1, batch}, anchors);
    model.ForwardV2(batch, input, {}, positionPtrs, std::vector<int>(batch, 1), slots, configs,
                    LastTokensManager(batch, 0), nullptr);
    configure(0);
    readScores(batch, steps);
    if (verifyLaunches != 2 * int(graph) || int(verifyScores.size()) != batch * (steps + 1))
        throw std::runtime_error("Clipped verification did not use the expected target path");
    for (int b = batch - 1; b >= 0; --b) {
        auto teacher = configs[b];
        teacher.output_logits = true;
        const int past = teacher.input_token_length, live = configs[b].output_token_limit - 2;
        const int end = past + live + 1;
        auto expected = committedKV(contexts.back(), end, past, live + 1);
        contexts.pop_back(); // Release one request's pages for the independent decode.
        ResponseContext ordinary;
        ordinary.pastKeyValues.resize(model.block_cnt);
        step(model, ordinary, 0, std::vector<int>(past, 320 + b), teacher);
        for (int r = 0; r <= live; ++r) {
            std::vector<float> logits;
            step(model, ordinary, past + r, {r ? 128 + (r - 1) * 13 : int(anchors[b])}, teacher, &logits);
            if (logits != verifyScores[b * (steps + 1) + r])
                throw std::runtime_error("Clipped verification logits disagree with ordinary decode");
        }
        if (committedKV(ordinary, end, past, live + 1) != expected)
            throw std::runtime_error("Clipped verification committed incorrect KV");
    }
    std::cout << "clipped_steps=" << steps << " batch=" << batch << " mixed=" << mixed
              << " round_graphs=" << verifyLaunches << std::endl;
}

// Replay the same proposal chain after restoring both target and draft state.
static void checkDraftPrefix(basellm &model, GenerationConfig sample) {
    const int page = GetPageLen();
    const int length = std::max(1024, 2 * page) + page + 3;
    std::vector<int> prompt(length);
    for (int i = 0; i < length; ++i) prompt[i] = 320 + i % 31;
    sample.input_token_length = length;
    model.SetSaveHistoryChat(true);
    std::vector<float> draftReference;
    std::vector<std::vector<float>> targetReference;
    for (int run = 0; run < 2; ++run) {
        ResponseContext ctx;
        ctx.pastKeyValues.resize(model.block_cnt);
        ctx.currentTokens = prompt;
        ctx.generationConfig = sample;
        const int cached = model.RestorePDResidentPrefix(&ctx);
        if (cached != (run ? (length - 1) / page * page : 0))
            throw std::runtime_error("Unexpected Inkling DSpark prefix hit");
        step(model, ctx, cached, ctx.currentTokens, sample);
        keep = 3;
        suffix = 0;
        configure(1, 3, -1, true);
        step(model, ctx, length, {235}, sample);
        configure(0);
        readScores(1, 7);
        std::vector<float> draft(size_t(14) * testVocab);
        if (cudaMemcpy(draft.data(), savedDraft, draft.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Failed to inspect restored draft");
        if (run && (draft != draftReference || verifyScores != targetReference))
            throw std::runtime_error("Restored Inkling draft or verification scores changed");
        draftReference = std::move(draft);
        targetReference = verifyScores;
    }
    model.SetSaveHistoryChat(false);
    std::cout << "shared prefix: draft distributions and verification logits matched" << std::endl;
}

static void checkDraftBatch(basellm &model, GenerationConfig sample) {
    const int vocab = testVocab;
    std::vector<float> reference[2];
    double maxError = 0, maxProbabilityError = 0;
    for (const auto indices : {std::vector<int>{0, 1}, std::vector<int>{0}, std::vector<int>{1}}) {
        const int count = indices.size();
        std::vector<ResponseContext> ctx(count);
        std::vector<int> past(count), token(count);
        std::vector<GenerationConfig> configs(count, sample);
        std::vector<std::pair<Data *, Data *>> slots;
        for (int b = 0; b < count; ++b) {
            int id = indices[b];
            auto &cfg = configs[b];
            cfg.temperature = id ? 1.3f : .75f;
            cfg.top_k = id ? 97 : 31;
            cfg.top_p = id ? .9f : .97f;
            past[b] = id ? 1279 : 1022;
            cfg.input_token_length = past[b];
            ctx[b].pastKeyValues.resize(model.block_cnt);
            step(model, ctx[b], 0, std::vector<int>(past[b], 650 + 125 * id), cfg);
            token[b] = 235 + id;
            for (auto &v : ctx[b].pastKeyValues) slots.emplace_back(&v.first, &v.second);
        }
        for (int round = 0; round < 2; ++round) {
            std::vector<float> anchors(token.begin(), token.end());
            std::vector<Data> position;
            position.reserve(count);
            std::vector<Data *> ptrs;
            for (int i : past) {
                position.emplace_back(FLOAT32, std::vector<int>{1, 1}, std::vector<float>{float(i)});
                ptrs.push_back(&position.back());
            }
            Data input(FLOAT32, {1, count}, anchors);
            keep = 7;
            suffix = 0;
            configure(count, indices[0] ? 5 : 2, count == 2 ? 5 : -1, true);
            token = model.ForwardV2(count, input, {}, ptrs, std::vector<int>(count, 1), slots, configs,
                                    LastTokensManager(count, 0), nullptr);
            configure(0);
            for (int b = 0; b < count; ++b) {
                ++past[b];
                while (model.PopPendingResponseToken(&ctx[b], token[b], token[b])) ++past[b];
                std::vector<float> data(size_t(14) * vocab);
                if (cudaMemcpy(data.data(), savedDraft + size_t(b) * data.size(), data.size() * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess)
                    throw std::runtime_error("Failed to inspect draft");
                for (int i = 0; i < 7; ++i) {
                    const float *scores = data.data() + size_t(2 * i) * vocab, *q = scores + vocab;
                    std::vector<int> ids(vocab);
                    std::iota(ids.begin(), ids.end(), 0);
                    const auto &cfg = configs[b];
                    int k = cfg.top_k;
                    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int x, int y) {
                        return scores[x] > scores[y] || (scores[x] == scores[y] && x < y);
                    });
                    std::vector<double> weights(k);
                    double sum = 0;
                    for (int j = 0; j < k; ++j)
                        sum += weights[j] = std::exp(double(scores[ids[j]] - scores[ids[0]]) / cfg.temperature);
                    double kept = 0, minWeight = 0;
                    for (int j = 0; j < k; ++j) {
                        if (kept >= sum * cfg.top_p && weights[j] < minWeight) break;
                        kept += weights[j];
                        minWeight = weights[j];
                    }
                    double mass = std::accumulate(q, q + vocab, 0.);
                    if (!std::isfinite(mass) || std::abs(mass - 1) > 1e-5)
                        throw std::runtime_error("Draft q is not normalized");
                    for (int j = 0; j < k; ++j)
                        maxProbabilityError =
                            std::max(maxProbabilityError,
                                     std::abs(q[ids[j]] - (weights[j] >= minWeight ? weights[j] / kept : 0.)));
                    auto &saved = reference[indices[b]];
                    if (count == 2)
                        saved.insert(saved.end(), scores, scores + vocab);
                    else {
                        auto first = saved.begin() + size_t(round * 7 + i) * vocab;
                        maxError = std::max(maxError, relativeL2({scores, scores + vocab}, {first, first + vocab}));
                    }
                }
            }
        }
    }
    const int layers = std::stoi(model.weight.dicts.at("mtp_config.num_hidden_layers"));
    for (int i = 0; i < 2 * layers; ++i) {
        auto *pool = GetPagedCacheManager(2 * model.block_cnt + i);
        if (!pool || pool->FreePageCount() != pool->maxPages)
            throw std::runtime_error("Draft pages were not returned to their pool");
    }
    std::cout << "batched_draft_max_l2=" << maxError << " max_q_error=" << maxProbabilityError
              << " released_draft_pools=" << 2 * layers << std::endl;
    if (maxError > .02 || maxProbabilityError > 2e-5)
        throw std::runtime_error("Batched draft disagrees with reference");
}

// Zero resident slots forces every gate/down projection through the host.
// Teacher-forced IDs keep the full vocabulary logits comparable across modes.
static int checkHybridDecode(const char *path) {
    setenv("FASTLLM_DSPARK_TOKENS", "0", 1);
    SetDeviceMap({{"cuda:0", 1}});
    SetMoeDeviceMap({{"cpu", 1}});
    SetThreads(8);
    SetMaxTokens(256);
    SetMoeCudaCacheDeviceBytes(0, 0);
    std::vector<std::vector<float>> reference;
    for (bool graph : {false, true}) {
        SetCudaGraph(graph);
        auto model = CreateLLMModelFromHF(path, BFLOAT16);
        model->maxBatch = 1;
        model->WarmUp();
        GenerationConfig cfg;
        cfg.output_logits = true;
        cfg.do_sample = true;
        cfg.top_k = 40;
        cfg.top_p = .95f;
        cfg.temperature = 1.f;
        cfg.input_token_length = 12;
        cfg.output_token_limit = 12;
        ResponseContext ctx;
        ctx.pastKeyValues.resize(model->block_cnt);
        step(*model, ctx, 0, std::vector<int>(12, 320), cfg);
        controlled = true;
        verifyLaunches = 0;
        for (int i = 0; i < 6; ++i) {
            std::vector<float> logits;
            step(*model, ctx, 12 + i, {235 + 13 * i}, cfg, &logits);
            if (!graph) reference.emplace_back(std::move(logits));
            else if (logits != reference.at(i))
                throw std::runtime_error("Inkling host-expert replay changed teacher-forced logits");
        }
        controlled = false;
        if (verifyLaunches != (graph ? 6 : 0))
            throw std::runtime_error("Expected one complete hybrid graph per incremental token");
    }
    std::cout << "PASS hybrid decode: six full replays matched eager logits" << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && std::string(argv[2]) == "--hybrid-decode") return checkHybridDecode(argv[1]);
    if (argc != 2 && (argc != 3 || std::string(argv[2]) != "--graph"))
        return 77;
    setenv("FASTLLM_DSPARK_TOKENS", "7", 1);
    SetDeviceMap({{"cuda:0", 1}});
    SetMoeDeviceMap({{"cuda:0", 1}});
    SetMaxTokens(16384);
    SetThreads(4);
    const bool graph = argc == 3;
    SetCudaGraph(graph);
    if (cudaMalloc(&control, 8 * sizeof(int)) != cudaSuccess) return 1;
    configure(0);
    auto model = CreateLLMModelFromHF(argv[1], BFLOAT16);
    model->maxBatch = maxTestBatch;
    testVocab = std::stoi(model->weight.dicts.at("text_config.unpadded_vocab_size"));
    if (cudaMalloc(&savedScores, size_t(maxTestBatch) * 8 * testVocab * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&savedDraft, size_t(maxTestBatch) * 14 * testVocab * sizeof(float)) != cudaSuccess) return 1;
    if (graph) model->WarmUp();
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
                suffix = run * 4096;
                configure(1);
                int token = step(*model, ctx, prompt.size(), {committed[0]}, sample);
                std::vector<int> emitted{token};
                while (model->PopPendingResponseToken(&ctx, token, token))
                    emitted.push_back(token);
                configure(0);
                if (!suffix && keep == 7) readScores(1, 7);
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
    checkClippedRound(*model, sample, 7, 2, graph, true);
    checkClippedRound(*model, sample, 7, maxTestBatch, graph);
    checkDraftBatch(*model, sample);
    checkDraftPrefix(*model, sample);
    cudaFree(savedDraft);
    cudaFree(savedScores);
    cudaFree(control);
}
