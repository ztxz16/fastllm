#include "models/mimo_tool_grammar.h"
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-token-constraint.h"
#include "naive-n05-cuda.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <limits>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void Cuda(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void Allocate(Data &data, DataType type, std::vector<int> dims) {
    data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
    data.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); data.Allocate(false);
}
static std::shared_ptr<MimoToolVocabulary> Vocabulary(int size) {
    auto tokens = std::make_shared<std::vector<std::string>>(size, "z");
    for (int i = 0; i < 256; ++i) (*tokens)[i] = std::string(1, char(i));
    const std::vector<std::string> extra = {"<tool_call>", "<function=A>", "<parameter=x>",
        "</parameter></function></tool_call>", "<eos>", "<think>", "</think>", "<function=B>", ""};
    std::copy(extra.begin(), extra.end(), tokens->begin() + 256);
    return std::make_shared<MimoToolVocabulary>(tokens);
}
static void Correctness(int steps) {
    const int vocab = 513, count = 17, words = (vocab + 31) / 32;
    auto vocabulary = Vocabulary(vocab);
    MimoToolGrammar a({{"A", {"x"}}}, false, true), b({{"B", {"x"}}}, true, false);
    a.Bind(vocabulary, {260}); b.Bind(vocabulary, {260});
    auto batch = FastllmCudaCreateTokenConstraintBatch(count + 7, steps);
    Data masks, tokens, logits;
    Allocate(masks, INT32, {steps + 1, count, words});
    Allocate(tokens, FLOAT32, {steps, count});
    Allocate(logits, FLOAT32, {count * (steps + 1), vocab});
    auto body = [&] {
        for (int step = 0; step <= steps; ++step) {
            FastllmCudaTokenConstraintMask(batch.get(), step, (uint32_t *)masks.cudaData + (size_t)step * count * words);
            if (step < steps) FastllmCudaTokenConstraintAdvance(batch.get(), step, (float *)tokens.cudaData + step * count);
        }
        FastllmCudaTokenConstraintApply(batch.get(), (float *)logits.cudaData, steps);
    };
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    for (int replay = 0; replay < 4; ++replay) {
        std::vector<std::shared_ptr<const TokenConstraintProgram>> programs;
        std::vector<int> initial, cpuStates;
        std::vector<MimoToolGrammar *> grammars;
        for (int row = 0; row < count; ++row) {
            auto *g = (row + replay) % 3 == 0 ? nullptr : (row + replay) % 2 ? &a : &b;
            grammars.push_back(g); int state = g && row % 2 ? g->Advance(0, "<think>example") : 0;
            cpuStates.push_back(state);
            auto p = g ? g->Compile(state) : std::make_pair(std::shared_ptr<const TokenConstraintProgram>(), 0);
            programs.push_back(p.first); initial.push_back(p.second);
        }
        std::vector<uint32_t> expected((size_t)(steps + 1) * count * words);
        std::vector<float> ids(steps * count), values((size_t)count * (steps + 1) * vocab, 1.f);
        for (int step = 0; step <= steps; ++step) for (int row = 0; row < count; ++row) {
            auto *g = grammars[row];
            auto bits = g ? *g->Mask(cpuStates[row]) : std::vector<uint32_t>(words, ~uint32_t(0));
            std::copy(bits.begin(), bits.end(), expected.begin() + ((size_t)step * count + row) * words);
            if (step == steps) continue;
            // Exercise reasoning exit, split markers, names, raw values and closure.
            const int path[] = {262, 256, g == &a ? 257 : 263, 258, int('x'), 259, 260, int('z')};
            int token = path[step];
            if (g && g->AdvanceToken(cpuStates[row], token) < 0) {
                token = 0;
                while (token < vocab && g->AdvanceToken(cpuStates[row], token) < 0) ++token;
            }
            Check(token < vocab, "no test continuation"); ids[step * count + row] = token;
            if (g) cpuStates[row] = g->AdvanceToken(cpuStates[row], token);
        }
        FastllmCudaPrepareTokenConstraintBatch(batch.get(), programs, initial, vocab);
        FastllmCudaCopyFromHostToDevice(tokens.cudaData, ids.data(), ids.size() * sizeof(float));
        FastllmCudaCopyFromHostToDevice(logits.cudaData, values.data(), values.size() * sizeof(float));
        if (!exec) {
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); body();
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        }
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread)); Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<uint32_t> actual(expected.size());
        FastllmCudaCopyFromDeviceToHost(actual.data(), masks.cudaData, actual.size() * sizeof(uint32_t));
        Check(actual == expected, "captured masks/state advance differ from CPU across replay");
        FastllmCudaCopyFromDeviceToHost(values.data(), logits.cudaData, values.size() * sizeof(float));
        for (int row = 0; row < count; ++row) for (int step = 0; step <= steps; ++step)
            for (int id = 0; id < vocab; ++id) {
                bool valid = expected[((size_t)step * count + row) * words + id / 32] & (uint32_t(1) << (id % 32));
                float v = values[((size_t)row * (steps + 1) + step) * vocab + id];
                Check(valid ? v == 1.f : v == -std::numeric_limits<float>::infinity(), "verifier branch mask mismatch");
            }
    }
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
}
static void InvalidBranch() {
    const int vocab = 513;
    MimoToolGrammar grammar({{"A", {"x"}}, {"B", {"x"}}}); grammar.Bind(Vocabulary(vocab), {260});
    auto p = grammar.Compile(0); auto batch = FastllmCudaCreateTokenConstraintBatch(2, 2);
    FastllmCudaPrepareTokenConstraintBatch(batch.get(), {p.first, p.first}, {p.second, p.second}, vocab);
    Data tokens(FLOAT32, {2, 2}, std::vector<float>{264.f, -1.f, 100.f, 100.f});
    tokens.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    FastllmCudaTokenConstraintAdvance(batch.get(), 0, (float *)tokens.cudaData);
    FastllmCudaTokenConstraintAdvance(batch.get(), 1, (float *)tokens.cudaData + 2, true);
    tokens.ToDevice(DataDevice::CPU);
    const float *ids = (float *)tokens.cpuData;
    Check(ids[0] == 0 && ids[1] == 0 && ids[2] == -1 && ids[3] == -1,
          "invalid branches must remain rejected and use safe embedding IDs");
}
static void RejectionPositions() {
    const int vocab = 513, steps = 5, batch = steps + 1;
    MimoToolGrammar grammar({{"A", {"x"}}, {"B", {"x"}}});
    grammar.Bind(Vocabulary(vocab), {260});
    auto compiled = grammar.Compile(0);
    auto constraints = FastllmCudaCreateTokenConstraintBatch(batch, steps);
    FastllmCudaPrepareTokenConstraintBatch(constraints.get(),
        std::vector<std::shared_ptr<const TokenConstraintProgram>>(batch, compiled.first), std::vector<int>(batch, compiled.second), vocab);
    const int path[] = {256, 257, 258, int('z'), 259};
    std::vector<int> proposals(batch * steps), expected(batch), sampled(batch * (steps + 1)), accepted(batch);
    std::vector<float> deviceIds(batch * steps), q((size_t)batch * steps * vocab, 0.f),
        p((size_t)batch * (steps + 1) * vocab, -std::numeric_limits<float>::infinity());
    for (int row = 0; row < batch; ++row) {
        int state = 0;
        for (int step = 0; step <= steps; ++step) {
            int token = step < steps ? path[step] : 260;
            if (step == row) {
                token = 0;
                while (token < vocab && (grammar.AdvanceToken(state, token) < 0 || (step < steps && token == path[step]))) ++token;
                Check(token < vocab, "missing correction alternative"); expected[row] = token;
            }
            p[((size_t)row * (steps + 1) + step) * vocab + token] = 0;
            p[((size_t)row * (steps + 1) + step) * vocab + 264] = 100; // Padded token must never survive masking.
            if (step < steps) {
                q[((size_t)row * steps + step) * vocab + path[step]] = 1;
                proposals[row * steps + step] = path[step]; deviceIds[step * batch + row] = path[step];
                state = grammar.AdvanceToken(state, path[step]); Check(state >= 0, "invalid proposal path");
            }
        }
    }
    Data ids(FLOAT32, {steps, batch}, deviceIds), probs(FLOAT32, {batch, steps, vocab}, q),
        logits(FLOAT32, {batch, steps + 1, vocab}, p);
    for (Data *data : {&ids, &probs, &logits}) data->ToDevice(DataDevice::CUDA, std::vector<int>{0});
    cudaGraph_t graph; cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    for (int step = 0; step < steps; ++step)
        FastllmCudaTokenConstraintAdvance(constraints.get(), step, (float *)ids.cudaData + step * batch, step == steps - 1);
    FastllmCudaTokenConstraintApply(constraints.get(), (float *)logits.cudaData, steps);
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
    std::vector<float> temperatures(batch * (steps + 1), 1.f), topPs(temperatures.size(), 1.f);
    std::vector<int> topKs(temperatures.size(), vocab);
    Check(FastllmCudaMtpRejectionSampling((float *)logits.cudaData, (float *)probs.cudaData,
        temperatures.data(), topKs.data(), topPs.data(), proposals.data(), sampled.data(), accepted.data(), batch, steps, vocab), "rejection positions");
    for (int row = 0; row < batch; ++row) {
        Check(accepted[row] == row, "wrong accepted prefix length");
        Check(sampled[row * (steps + 1) + row] == expected[row], "wrong correction/bonus token");
        int state = 0;
        for (int step = 0; step < accepted[row]; ++step) state = grammar.AdvanceToken(state, path[step]);
        Check(grammar.AdvanceToken(state, expected[row]) >= 0, "correction committed from wrong grammar state");
    }
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
}
static void Benchmark(int count, bool header = false) {
    const int vocab = 152576, steps = 5, words = (vocab + 31) / 32, repetitions = 12;
    auto vocabulary = Vocabulary(vocab);
    MimoToolGrammar grammar({{"A", {"x"}}, {"B", {"x"}}}); grammar.Bind(vocabulary, {260});
    const int initialState = header ? grammar.Advance(0, "<tool_call><function=") : 0;
    const auto compiled = grammar.Compile(initialState);
    auto constraints = FastllmCudaCreateTokenConstraintBatch(count, steps);
    FastllmCudaPrepareTokenConstraintBatch(constraints.get(),
        std::vector<std::shared_ptr<const TokenConstraintProgram>>(count, compiled.first), std::vector<int>(count, compiled.second), vocab);
    Data base, bias, mask, logits, probabilities, ids, target;
    Allocate(base, BFLOAT16, {1, count, vocab}); Allocate(bias, BFLOAT16, {1, count, vocab});
    Allocate(mask, INT32, {count, words}); Allocate(logits, FLOAT32, {count, vocab});
    Allocate(probabilities, FLOAT32, {count, vocab}); Allocate(ids, FLOAT32, {steps, count});
    Allocate(target, FLOAT32, {count * (steps + 1), vocab});
    Cuda(cudaMemsetAsync(base.cudaData, 0, base.GetBytes(), cudaStreamPerThread));
    Cuda(cudaMemsetAsync(bias.cudaData, 0, bias.GetBytes(), cudaStreamPerThread));
    Cuda(cudaMemsetAsync(target.cudaData, 0, target.GetBytes(), cudaStreamPerThread));
    std::vector<float> temps(count, 1.f), ps(count, 1.f); std::vector<int> ks(count, 0);
    void *sampler = FastllmCudaMtpCreateDraftSamplerBatch(count, vocab, temps.data(), ks.data(), ps.data());
    Check(sampler, "persistent sampler allocation");
    auto graphBody = [&](bool constrained) {
        for (int step = 0; step < steps; ++step) {
            if (constrained) FastllmCudaTokenConstraintMask(constraints.get(), step, (uint32_t *)mask.cudaData);
            FastllmCudaNaiveBatchDraftSamplingLogits(base, bias, mask, logits);
            Check(FastllmCudaMtpDraftSamplerRunBatch(sampler, (float *)logits.cudaData, (float *)probabilities.cudaData,
                (float *)ids.cudaData + (size_t)step * count, step * 4), "graph sampler");
            if (constrained) FastllmCudaTokenConstraintAdvance(constraints.get(), step, (float *)ids.cudaData + step * count, step == steps - 1);
        }
    };
    std::vector<float> sampled((size_t)count * steps);
    auto measure = [&](auto body) {
        body(); Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repetitions; ++i) body();
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / repetitions;
    };
    std::vector<int> states(count), oldIds(count);
    std::vector<uint32_t> bits((size_t)count * words);
    double eager = measure([&] {
        std::fill(states.begin(), states.end(), initialState);
        for (int step = 0; step < steps; ++step) {
            for (int row = 0; row < count; ++row) {
                auto m = grammar.Mask(states[row]); std::copy(m->begin(), m->end(), bits.begin() + (size_t)row * words);
            }
            FastllmCudaCopyFromHostToDevice(mask.cudaData, bits.data(), bits.size() * sizeof(uint32_t));
            FastllmCudaNaiveBatchDraftSamplingLogits(base, bias, mask, logits);
            Check(FastllmCudaMtpSampleDraft((float *)logits.cudaData, (float *)probabilities.cudaData,
                temps.data(), ks.data(), ps.data(), oldIds.data(), count, vocab), "eager sampler");
            for (int row = 0; row < count; ++row) states[row] = grammar.AdvanceToken(states[row], oldIds[row]);
        }
    });
    double graphMs[2];
    for (int constrained = 0; constrained <= 1; ++constrained) {
        std::fill(bits.begin(), bits.end(), ~uint32_t(0));
        FastllmCudaCopyFromHostToDevice(mask.cudaData, bits.data(), bits.size() * sizeof(uint32_t));
        Check(FastllmCudaMtpDraftSamplerSetSeed(sampler), "seed"); graphBody(constrained);
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        cudaGraph_t graph; cudaGraphExec_t exec;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); graphBody(constrained);
        Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        graphMs[constrained] = measure([&] {
            // Include production's per-round host setup and descriptor/state
            // uploads. Only one-time compilation and table uploads are warmed.
            if (constrained) {
                std::vector<std::shared_ptr<const TokenConstraintProgram>> programs;
                std::vector<int> initial;
                for (int row = 0; row < count; ++row) {
                    auto p = grammar.Compile(initialState);
                    programs.push_back(p.first); initial.push_back(p.second);
                }
                FastllmCudaPrepareTokenConstraintBatch(constraints.get(), programs, initial, vocab);
            }
            Check(FastllmCudaMtpDraftSamplerSetParameters(sampler, count, temps.data(), ks.data(), ps.data()), "parameters");
            Check(FastllmCudaMtpDraftSamplerSetSeed(sampler), "reseed");
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            FastllmCudaCopyFromDeviceToHost(sampled.data(), ids.cudaData, sampled.size() * sizeof(float));
        });
        Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
    }
    // Compare only mask dispatch, including its host overhead, without sampling.
    std::vector<std::shared_ptr<const std::vector<uint32_t>>> verifyMasks(count * (steps + 1));
    for (int row = 0; row < count; ++row) {
        int state = initialState;
        for (int step = 0; step <= steps; ++step) {
            verifyMasks[row * (steps + 1) + step] = grammar.Mask(state);
            if (step < steps) state = grammar.AdvanceToken(state, (int)sampled[step * count + row]);
            Check(state >= 0, "benchmark sampled invalid token");
        }
    }
    double oldVerify = measure([&] {
        for (int row = 0; row < count * (steps + 1); ++row)
            FastllmCudaApplyTokenMask((float *)target.cudaData + (size_t)row * vocab, verifyMasks[row]->data(), vocab);
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    });
    double newVerify = measure([&] {
        FastllmCudaTokenConstraintApply(constraints.get(), (float *)target.cudaData, steps);
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    });
    FastllmCudaMtpDestroyDraftSampler(sampler);
    std::cout << "BENCH mode=" << (header ? "header" : "text") << " batch=" << count << " eager_draft_ms=" << eager << " graph_plain_ms=" << graphMs[0]
        << " graph_constraint_ms=" << graphMs[1] << " old_verify_mask_ms=" << oldVerify
        << " new_verify_mask_ms=" << newVerify << '\n';
}
int main(int argc, char **argv) {
    int count; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        FastllmCudaSetDevice(0);
        for (int steps = 2; steps <= 8; ++steps) Correctness(steps);
        InvalidBranch(); RejectionPositions();
        if (argc > 1 && std::string(argv[1]) == "--benchmark") { for (int batch : {1, 16, 64, 128}) Benchmark(batch); Benchmark(128, true); }
        std::cout << "ALL_PASS token constraint CUDA graph\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
