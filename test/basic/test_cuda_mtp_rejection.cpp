#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-dspark.h"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include "baseblock.h"
#include "utils/utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void Check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    float *p = nullptr;
    size_t n;
    explicit Buffer(size_t n) : n(n) { Check(cudaMalloc(&p, n * sizeof(float))); }
    explicit Buffer(const std::vector<float> &v) : Buffer(v.size()) {
        Check(cudaMemcpy(p, v.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~Buffer() { cudaFree(p); }
    std::vector<float> Read() const {
        std::vector<float> v(n);
        Check(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
        return v;
    }
};
std::vector<double> Oracle(const std::vector<float> &logits, float t, int k, float topP, bool topKFirst = false) {
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    std::vector<double> p(logits.size());
    const double maxLogit = logits[order.front()];
    const int count = topKFirst && k > 0 ? std::min(k, (int)order.size()) : (int)order.size();
    for (int i = 0; i < count; ++i) {
        const int token = order[i];
        p[token] = std::exp((logits[token] - maxLogit) / t);
    }
    double sum = std::accumulate(p.begin(), p.end(), 0.0);
    for (auto &v : p) v /= sum;
    auto original = p;
    // FastLLM's joint sampler uses strict-greater counts/mass, including ties.
    double mass = 0;
    for (int first = 0, end; first < (int)order.size(); first = end) {
        for (end = first + 1; end < (int)order.size() &&
                original[order[end]] == original[order[first]]; ++end) {}
        for (int i = first; i < end; ++i)
            if ((k > 0 && first >= k) || mass >= topP) p[order[i]] = 0;
        for (int i = first; i < end; ++i) mass += original[order[i]];
    }
    sum = std::accumulate(p.begin(), p.end(), 0.0);
    for (auto &v : p) v /= sum;
    return p;
}
void FilterCase() {
    const int groups = 5, repeats = 4096, rows = groups * repeats, vocab = 32;
    std::vector<float> l(rows * vocab, -80), t(rows), pp(rows);
    std::vector<int> k(rows), sampled(rows), ordinary(rows);
    const float ts[] = {1, .7f, 1.5f, 1, 1};
    const float ps[] = {.55f, .72f, 1, .85f, 1};
    const int ks[] = {2, 5, 32, 3, 1};
    std::vector<std::vector<double>> expected(groups);
    for (int g = 0; g < groups; ++g) {
        std::vector<float> base(vocab, -80);
        for (int i = 0; i < 8; ++i) base[i] = g == 4 ? 0.0f : -i * .35f;
        expected[g] = Oracle(base, ts[g], ks[g], ps[g]);
        for (int b = g * repeats; b < (g + 1) * repeats; ++b) {
            std::copy(base.begin(), base.end(), l.begin() + b * vocab);
            t[b] = ts[g]; pp[b] = ps[g]; k[b] = ks[g];
        }
    }
    Buffer logits(l), q(l.size());
    Require(FastllmCudaMtpSampleDraft(logits.p, q.p, t.data(), k.data(), pp.data(),
                                     sampled.data(), rows, vocab), "proposal sampling failed");
    auto saved = q.Read();
    Require(FastllmCudaTopKTopPSampling(logits.p, t.data(), k.data(), pp.data(),
                                      ordinary.data(), rows, vocab), "ordinary sampling failed");
    for (int g = 0; g < groups; ++g) {
        std::vector<int> counts(vocab), normalCounts(vocab);
        for (int b = g * repeats; b < (g + 1) * repeats; ++b) {
            for (int i = 0; i < vocab; ++i)
                Require(std::fabs(saved[b * vocab + i] - expected[g][i]) < 2e-5,
                        "saved q disagrees with the joint top-k/top-p CPU oracle");
            Require(sampled[b] >= 0 && sampled[b] < vocab && expected[g][sampled[b]] > 0,
                    "proposal outside its saved support");
            ++counts[sampled[b]]; ++normalCounts[ordinary[b]];
        }
        for (int i = 0; i < vocab; ++i) {
            Require(std::fabs(double(counts[i])/repeats - expected[g][i]) < .035,
                    "proposal frequency disagrees with q");
            Require(std::fabs(double(normalCounts[i])/repeats - expected[g][i]) < .035,
                    "probability preparation changed ordinary sampling semantics");
        }
    }
    Require(saved == q.Read(), "scratch reuse overwrote saved q");
    std::puts("joint top-k/top-p, temperatures, ties, saved q, ordinary sampler agreement: PASS");
}
void DeviceFilterCase() {
    const int vocab = 32;
    std::vector<float> values(vocab, -80);
    for (int i = 0; i < 8; ++i) values[i] = -i * .35f;
    Buffer logits(values), q(vocab);
    int *token;
    Check(cudaMalloc(&token, sizeof(int)));
    for (float t : {.7f, 1.f, 1.5f}) for (int k : {0, 1, 5, 32}) for (float p : {.55f, 1.f}) {
        fastllm::Data workspace;
        const auto expected = Oracle(values, t, k, p);
        Require(FastllmCudaMtpSampleDraftDevice(logits.p, q.p, t, k, p, token, vocab, workspace), "device sampler warmup");
        Check(cudaDeviceSynchronize());
        cudaGraph_t graph; cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Require(FastllmCudaMtpSampleDraftDevice(logits.p, q.p, t, k, p, token, vocab, workspace), "device sampler capture");
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        std::vector<int> counts(vocab);
        constexpr int repeats = 2048;
        for (int i = 0; i < repeats; ++i) {
            Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            int selected = -1;
            Check(cudaMemcpy(&selected, token, sizeof(int), cudaMemcpyDeviceToHost));
            Require(selected >= 0 && selected < vocab && expected[selected] > 0, "device sample outside q");
            ++counts[selected];
        }
        const auto actual = q.Read();
        for (int i = 0; i < vocab; ++i) {
            Require(std::fabs(actual[i] - expected[i]) < 2e-5, "device q differs from CPU joint-filter oracle");
            Require(std::fabs(double(counts[i]) / repeats - expected[i]) < .05, "graph RNG/distribution mismatch");
        }
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
    // Equal target and draft logits must give identical support even when the k-th logit is tied.
    // With three equal candidates, k=2 and p=0.6 keeps both surviving candidates, in token-ID order.
    for (const std::vector<float> scores : {std::vector<float>{2, 1, .7f, .4f, 0}, {1, 1, 1, 0, -1}}) {
        Buffer input(scores), filtered(scores.size());
        fastllm::Data workspace;
        auto expected = Oracle(scores, 1, 2, .6f, true);
        Require(FastllmCudaMtpSampleDraftDeviceTopKFirst(input.p, filtered.p, 1, 2, .6f,
                token, scores.size(), workspace), "top-k-first draft sampling failed");
        auto actual = filtered.Read();
        for (size_t i = 0; i < actual.size(); ++i)
            Require(std::fabs(actual[i] - expected[i]) < 2e-5, "draft and target filter support differ");
        std::vector<float> twoRows(scores);
        twoRows.insert(twoRows.end(), scores.begin(), scores.end());
        Buffer target(twoRows);
        int sampled, out[2], accepted, topKs[2] = {2, 2};
        float temperatures[2] = {1, 1}, topPs[2] = {.6f, .6f};
        Check(cudaMemcpy(&sampled, token, sizeof(sampled), cudaMemcpyDeviceToHost));
        Require(FastllmCudaMtpRejectionSamplingTopKFirst(target.p, filtered.p, temperatures, topKs,
                topPs, &sampled, out, &accepted, 1, 1, scores.size()) && accepted == 1 && sampled == out[0],
                "equal filtered logits rejected the draft");
    }
    // Wide/misaligned vocabulary, ties at k, stale q rows, and fewer than k finite logits.
    for (int k : {1, 7, 40, 64, 65}) {
        const int vocab = 200059;
        std::vector<float> values(vocab);
        for (int i = 0; i < vocab; ++i)
            values[i] = k == 7 && i % 50000 ? -INFINITY : -float(i % 10003) * .25f;
        Buffer input(values), filtered(std::vector<float>(vocab, 1.f));
        fastllm::Data workspace;
        for (float p : {.81f, 1.f}) {
            auto expected = Oracle(values, .75f, k, p, true);
            Require(FastllmCudaMtpSampleDraftDeviceTopKFirst(input.p, filtered.p, .75f, k, p,
                    token, vocab, workspace), "compact proposal failed");
            auto actual = filtered.Read();
            int sampled;
            Check(cudaMemcpy(&sampled, token, sizeof(int), cudaMemcpyDeviceToHost));
            Require(sampled >= 0 && sampled < vocab && actual[sampled] > 0, "sample outside retained q");
            double error = 0;
            for (int i = 0; i < vocab; ++i) {
                Require((actual[i] > 0) == (expected[i] > 0), "compact proposal support differs");
                error += std::abs(actual[i] - expected[i]);
            }
            Require(error < 1.e-5, "compact proposal probabilities differ");
        }
    }
    {
        Buffer input(std::vector<float>(129, -INFINITY)), filtered(129);
        fastllm::Data workspace;
        Require(FastllmCudaMtpSampleDraftDeviceTopKFirst(input.p, filtered.p, 1.f, 40, .9f,
                token, 129, workspace), "masked proposal failed");
        int sampled;
        Check(cudaMemcpy(&sampled, token, sizeof(int), cudaMemcpyDeviceToHost));
        auto q = filtered.Read();
        Require(sampled == 0 && std::all_of(q.begin(), q.end(), [](float p) { return p == 0.f; }),
                "all-masked compact proposals must retain a safe token index and zero mass");
    }
    cudaFree(token);
    std::puts("DEVICE_FILTER_GRAPH_PASS 24 distributions, 49152 replays, saved q and RNG checked");
}
#ifndef USE_ROCM
void IndexedBf16ProjectionCase() {
    using namespace fastllm;
    constexpr int vocab = 131, width = 256;
    std::vector<float> embed(vocab * width), projection(embed.size());
    for (int i = 0; i < vocab * width; ++i) {
        embed[i] = (i * 13 % 19 - 9) / 32.f;
        projection[i] = (i * 7 % 17 - 8) / 64.f;
    }
    Data table(FLOAT32, {vocab, width}, embed), weight(FLOAT32, {vocab, width}, projection);
    for (auto *matrix : {&table, &weight}) {
        matrix->ToDevice(CUDA, {0}, true);
        ToDataType(*matrix, BFLOAT16);
    }
    Data id(FLOAT32, {1, 1}), output(BFLOAT16, {1, 1, vocab});
    for (auto *buffer : {&id, &output}) {
        buffer->ToDevice(CUDA, {0}, false);
        buffer->Allocate(false);
    }
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(FastllmCudaIndexedBf16Gemv(id, table, weight, output), "indexed BF16 projection capture");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int token : {0, 73, vocab - 1}) {
        float input = token;
        Check(cudaMemcpy(id.cudaData, &input, sizeof(input), cudaMemcpyHostToDevice));
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        std::vector<uint16_t> result(vocab);
        Check(cudaMemcpy(result.data(), output.cudaData, output.GetBytes(), cudaMemcpyDeviceToHost));
        for (int row = 0; row < vocab; ++row) {
            float sum = 0;
            for (int col = 0; col < width; ++col)
                sum += embed[token * width + col] * projection[row * width + col];
            Require(result[row] == Float32ToBFloat16RNEBits(sum), "indexed BF16 projection differs from CPU");
        }
    }
    Check(cudaGraphExecDestroy(exec));
    Check(cudaGraphDestroy(graph));
}
#endif

// Several distinct input scores round to exactly the same softcap. The stable top-k
// boundary follows token ID across that entire plateau, including after graph replay.
void MarkovDrawPlateauCase() {
    using namespace fastllm;
    constexpr int rows = 3, vocab = 65539, pitch = vocab + 3;
    const int peaks[rows] = {85, 300, 85};
    std::vector<float> base(rows * pitch, -300.f);
    for (int row = 0; row < rows; ++row)
        for (int i = 0; i < peaks[row]; ++i) base[row * pitch + 10 * i + row] = float(300 + 2 * i);
    for (DataType type : {BFLOAT16, FLOAT16}) {
        auto upload = [&](const std::vector<float> &values) {
            Data x(FLOAT32, {(int)values.size()}, values);
            x.ToDevice(CUDA, {0}, true);
            ToDataType(x, type);
            return x;
        };
        auto b = upload(base), correction = upload(std::vector<float>(rows * vocab, -2.f));
        auto bias = upload(std::vector<float>(vocab, 2.f));
        Buffer temperature(std::vector<float>{.7f, 1.f, 1.3f}), topP(std::vector<float>(rows, .95f));
        Buffer k(rows), seed(2), tokens(rows * 2), q(std::vector<float>(rows * pitch, -1.f));
        Check(cudaMemsetAsync(seed.p, 0, sizeof(uint64_t), cudaStreamPerThread));
        auto *workspace = FastllmCudaMtpCreateTargetDistribution(rows, vocab);
        Require(workspace != nullptr, "Markov draw workspace");
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Require(FastllmCudaMtpGraphDrawMarkov(workspace, b.cudaData, pitch, correction.cudaData, bias.cudaData, 30.f,
                    type == BFLOAT16, temperature.p, reinterpret_cast<int *>(k.p), topP.p,
                    reinterpret_cast<uint64_t *>(seed.p), 0, q.p, pitch, reinterpret_cast<int *>(tokens.p), 2),
                "Markov draw capture");
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (std::vector<int> ks : {std::vector<int>{40, 40, vocab}, std::vector<int>{vocab, 1, 40}}) {
            Check(cudaMemcpy(k.p, ks.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
            Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            auto probs = q.Read();
            int ids[rows * 2];
            Check(cudaMemcpy(ids, tokens.p, sizeof(ids), cudaMemcpyDeviceToHost));
            for (int row = 0; row < rows; ++row) {
                const int keep = std::min(ks[row], peaks[row]);
                for (int i = 0; i < vocab; ++i) {
                    const bool kept = i >= row && (i - row) % 10 == 0 && (i - row) / 10 < keep;
                    Require(std::abs(probs[row * pitch + i] - (kept ? 1.f / keep : 0.f)) < 1.e-6f,
                            "Markov draw changed the softcap plateau");
                }
                Require(probs[row * pitch + vocab] == -1.f, "Markov draw wrote into q padding");
                Require(ids[2 * row] >= 0 && ids[2 * row] < vocab && probs[row * pitch + ids[2 * row]] > 0.f,
                        "Markov draw sampled outside retained q");
            }
        }
        Check(cudaGraphExecDestroy(exec));
        Check(cudaGraphDestroy(graph));
        FastllmCudaMtpDestroyTargetDistribution(workspace);
    }
}
void MarkovGraphCase() {
    using namespace fastllm;
    const int vocab = 64, rank = 32;
    std::vector<float> embed(vocab * rank), head(embed.size());
    // Powers of two keep the independent CPU dot exact for every dtype.
    for (int i = 0; i < vocab * rank; ++i) {
        embed[i] = (int(i * 13 % 17) - 8) / 32.f;
        head[i] = (int(i * 7 % 13) - 6) / 64.f;
    }
    int checks = 0;
    for (DataType dtype : {FLOAT32, BFLOAT16, FLOAT16}) for (int block : {1, 2, 3, 5, 7, 8, 9, 16, 31}) {
        auto values = [&](const std::vector<float> &src) {
            Data x(FLOAT32, {vocab, rank}, src);
            x.ToDevice(CUDA, {0}, true);
            if (dtype != FLOAT32) ToDataType(x, dtype);
            return x;
        };
        Data e = values(embed), h = values(head);
        std::vector<float> input(block * vocab);
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i * 11 % 31) - 15) / 8;
        Data logits(FLOAT32, {1, block, vocab}, input); logits.ToDevice(CUDA, {0}, true);
        Data tokens(INT32, {block+1}), probs, outEmbeds, workspace;
        tokens.ToDevice(CUDA, {0}, false); tokens.Allocate(false);
        const float temperature = .7f, topP = .8f; const int topK = 7;
        int anchor = 3;
        Check(cudaMemcpy(tokens.cudaData, &anchor, sizeof(anchor), cudaMemcpyHostToDevice));
        auto run = [&] { return FastllmCudaDeepSeekV41MarkovFilteredSampleDevice(logits, e, h,
            block, temperature, topK, topP, tokens, probs, outEmbeds, workspace); };
        Require(run(), "filtered Markov warmup"); Check(cudaDeviceSynchronize());
        cudaGraph_t graph; cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Require(run(), "filtered Markov capture");
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 8; ++replay) {
            anchor = (replay * 7) % vocab;
            Check(cudaMemcpy(tokens.cudaData, &anchor, sizeof(anchor), cudaMemcpyHostToDevice));
            Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            std::vector<int> selected(block+1);
            std::vector<float> q(block*vocab), emb(block*rank);
            Check(cudaMemcpy(selected.data(), tokens.cudaData, selected.size()*4, cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(q.data(), probs.cudaData, q.size()*4, cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(emb.data(), outEmbeds.cudaData, emb.size()*4, cudaMemcpyDeviceToHost));
            Require(selected[0] == anchor, "graph anchor stale");
            for (int step=0;step<block;++step) {
                Require(selected[step] >= 0 && selected[step] < vocab, "invalid graph Markov token");
                std::vector<float> row(vocab);
                for (int v=0;v<vocab;++v) {
                    float dot=0;
                    for (int j=0;j<rank;++j) dot += embed[selected[step]*rank+j]*head[v*rank+j];
                    // Dot sums in this fixture are exactly representable in BF16/FP16.
                    row[v]=input[step*vocab+v]+dot;
                }
                const auto expected=Oracle(row,temperature,topK,topP);
                for (int v=0;v<vocab;++v)
                    Require(std::fabs(q[step*vocab+v]-expected[v])<2e-5, "Markov saved q/conditional CPU oracle mismatch");
                Require(selected[step+1] >= 0 && selected[step+1] < vocab && expected[selected[step+1]] > 0,
                    "Markov sample outside conditional q");
                for (int j=0;j<rank;++j)
                    Require(emb[step*rank+j] == embed[selected[step]*rank+j], "Markov confidence embedding mismatch");
            }
            ++checks;
        }
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
    std::printf("MARKOV_FILTER_GRAPH_PASS %d replays; 3 dtypes; block 1/2/3/5/7/8/9/16/31\n", checks);
}
void ChainCase(const char *name, std::vector<float> p, std::vector<float> qdist,
               int drafts, int vocab, float targetTemperature = 1.f,
               int targetK = 0, float targetP = 1.f, bool topKFirst = false) {
    const int batch = 16384, qRows = batch * drafts, pRows = batch * (drafts + 1);
    std::vector<float> qLogits(qRows * vocab, -INFINITY), pLogits(pRows * vocab, -INFINITY);
    const float bonus[] = {.1f, .2f, .7f};
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < drafts; ++d) {
            for (int i = 0; i < 3; ++i) {
                qLogits[(b * drafts + d) * vocab + i] = std::log(qdist[i]);
                pLogits[(b * (drafts + 1) + d) * vocab + i] = std::log(p[i]);
            }
        }
        for (int i = 0; i < 3; ++i)
            pLogits[(b * (drafts + 1) + drafts) * vocab + i] = std::log(bonus[i]);
    }
    Buffer ql(qLogits), pl(pLogits), q(qLogits.size());
    std::vector<float> qt(qRows, 1), qp(qRows, 1), pt(pRows, targetTemperature), pp(pRows, targetP);
    std::vector<int> qk(qRows, 0), pk(pRows, targetK), ids(qRows), out(pRows), accepted(batch);
    Require(FastllmCudaMtpSampleDraft(ql.p, q.p, qt.data(), qk.data(), qp.data(), ids.data(),
                                     qRows, vocab), "chain proposal sampling failed");
    auto snapshot = q.Read();
    auto verify = topKFirst ? FastllmCudaMtpRejectionSamplingTopKFirst : FastllmCudaMtpRejectionSampling;
    Require(verify(pl.p, q.p, pt.data(), pk.data(), pp.data(),
        ids.data(), out.data(), accepted.data(), batch, drafts, vocab), "chain verification failed");
    Require(snapshot == q.Read(), "verification mutated q");
    int counts[3] = {}, bonuses[3] = {}, residuals[3] = {}, allAccepted = 0, rejected = 0;
    std::vector<int> prefix(drafts);
    for (int b = 0; b < batch; ++b) {
        int n = accepted[b], off = b * (drafts + 1);
        Require(n >= 0 && n <= drafts, "invalid commit length");
        Require(out[off] >= 0 && out[off] < 3, "invalid first output");
        ++counts[out[off]];
        for (int d = 0; d < n; ++d) {
            ++prefix[d];
            Require(out[off + d] == ids[b * drafts + d], "accepted prefix token mismatch");
        }
        Require(out[off + n] >= 0 && out[off + n] < 3, "invalid correction/bonus");
        if (n == drafts) { ++allAccepted; ++bonuses[out[off + n]]; }
        else {
            ++rejected; ++residuals[out[off + n]];
            for (int d = n + 1; d <= drafts; ++d)
                Require(out[off + d] == -1, "rejected suffix was emitted");
        }
    }
    // The proposal is full softmax even when target p has a smaller support.
    // Check the final law after rejection/correction, not just sampler bounds.
    std::vector<float> logP, logBonus;
    for (int i = 0; i < 3; ++i) { logP.push_back(std::log(p[i])); logBonus.push_back(std::log(bonus[i])); }
    auto filteredP = Oracle(logP, targetTemperature, targetK, targetP, topKFirst);
    auto filteredBonus = Oracle(logBonus, targetTemperature, targetK, targetP, topKFirst);
    p.assign(filteredP.begin(), filteredP.end());
    double alpha = 0, residualMass = 0;
    for (int i = 0; i < 3; ++i) {
        alpha += std::min(p[i], qdist[i]);
        residualMass += std::max(p[i] - qdist[i], 0.f);
        Require(std::fabs(double(counts[i])/batch - p[i]) < .02, "target distribution changed");
    }
    for (int d = 0; d < drafts; ++d)
        Require(std::fabs(double(prefix[d])/batch - std::pow(alpha, d+1)) < .025,
                "acceptance disagrees with sum min(p,q)");
    if (p == qdist) Require(allAccepted == batch, "identical p/q must accept all drafts");
    if (alpha == 0) Require(allAccepted == 0 && prefix[0] == 0, "disjoint support was accepted");
    for (int i = 0; i < 3; ++i) {
        if (rejected > 1000)
            Require(std::fabs(double(residuals[i])/rejected -
                std::max(p[i]-qdist[i], 0.f)/residualMass) < .03, "wrong residual distribution");
        if (allAccepted > 1000)
            Require(std::fabs(double(bonuses[i])/allAccepted - filteredBonus[i]) < .03, "wrong bonus distribution");
    }
    std::printf("%s drafts=%d vocab=%d p_out=[%.4f %.4f %.4f] accept_first=%.4f all=%.4f: PASS\n",
        name, drafts, vocab, double(counts[0])/batch, double(counts[1])/batch,
        double(counts[2])/batch, double(prefix[0])/batch, double(allAccepted)/batch);
}
void FullVocabularyCase() {
    const int vocab = 152576, rows = 4;
    const float temperatures[] = {1, .7f, 1.5f, 1};
    const float topPs[] = {.95f, .5f, 1, .01f};
    const int topK = vocab;
    std::vector<float> logits(rows * vocab);
    for (int row = 0; row < rows; ++row)
        for (int id = 0; id < vocab; ++id)
            logits[row * vocab + id] = -float((id * 73 + row * 19) % 1000) / 16;
    Buffer input(logits), proposals(logits.size());
    std::vector<float> saved;
    for (int row = 0; row < rows; ++row) {
        int token = -1;
        Require(FastllmCudaMtpSampleDraft(input.p + (size_t)row * vocab,
            proposals.p + (size_t)row * vocab, temperatures + row, &topK,
            topPs + row, &token, 1, vocab), "full-vocabulary proposal failed");
        std::vector<float> source(logits.begin() + row * vocab, logits.begin() + (row + 1) * vocab);
        auto expected = Oracle(source, temperatures[row], vocab, topPs[row]);
        auto actual = proposals.Read();
        Require(token >= 0 && token < vocab && expected[token] > 0,
                "full-vocabulary proposal outside nucleus support");
        double error = 0, sum = 0;
        for (int id = 0; id < vocab; ++id) {
            error += std::fabs(actual[row * vocab + id] - expected[id]);
            sum += actual[row * vocab + id];
        }
        Require(error < 1e-4 && std::fabs(sum - 1) < 1e-5,
                "full-vocabulary q disagrees with the joint sampler oracle");
        // Later proposal steps must not overwrite earlier persistent q rows.
        Require(std::equal(saved.begin(), saved.end(), actual.begin()),
                "later draft step overwrote a saved proposal row");
        saved.assign(actual.begin(), actual.begin() + (row + 1) * vocab);
    }
    std::puts("152576-token vocabulary, sequential q rows, temperatures/top-p/ties: PASS");
}
void PointMassCase() {
    // DSpark can compare a nearly certain draft against a singleton target.
    // Inspect the probabilities directly so a rare rejection draw is unnecessary.
    const float temperature = .6f, topP = .95f;
    for (const int vocab : {32768, 152576}) {
        std::vector<float> values(vocab, -80.f);
        values[0] = 34.f;
        Buffer logits(values), q(vocab), token(1);
        void *draft = FastllmCudaMtpCreateDraftSampler(vocab, temperature, vocab, 1.f);
        Require(draft && FastllmCudaMtpDraftSamplerSetSeed(draft) &&
                FastllmCudaMtpDraftSamplerRun(draft, logits.p, q.p, token.p),
                "point-mass draft sampling failed");
        Require(q.Read()[0] == 1.f, "rounded draft peak must have unit probability");
        FastllmCudaMtpDestroyDraftSampler(draft);

        values[0] = 3.f; values[1] = .78125f; values[2] = -1.21875f;
        Check(cudaMemcpy(logits.p, values.data(), logits.n * sizeof(float), cudaMemcpyHostToDevice));
        void *target = FastllmCudaMtpCreateTargetDistribution(1, vocab);
        float *probabilities = nullptr;
        bool changed = false;
        Require(target && FastllmCudaMtpPrepareTargetDistribution(target, &temperature, &vocab, &topP, &changed) &&
                FastllmCudaMtpNormalizeTargetDistribution(target, logits.p, &probabilities),
                "point-mass target normalization failed");
        Check(cudaMemcpy(values.data(), probabilities, logits.n * sizeof(float), cudaMemcpyDeviceToHost));
        Require(values[0] == 1.f && std::all_of(values.begin() + 1, values.end(),
                [](float p) { return p == 0.f; }), "singleton nucleus must have exactly unit mass");
        FastllmCudaMtpDestroyTargetDistribution(target);
    }
}
void RoundedResidualCase() {
    // Same two logits, different FP32 normalization paths. Both p entries
    // rounded below q even though the proposal and target have full support.
    Buffer target(std::vector<float>{.5004068613052368f, .4995931088924408f});
    Buffer proposal(std::vector<float>{.5004069209098816f, .4995931386947632f});
    Buffer ids(1), output(1);
    int id = 0, correction = -1;
    Check(cudaMemcpy(ids.p, &id, sizeof(id), cudaMemcpyHostToDevice));
    Require(FastllmCudaMtpRecoverRoundedResidual(target.p, proposal.p, (int *)ids.p, (int *)output.p,
                correction, 0, 0, 1, 2, 42, nullptr) && correction == id,
            "rounding-induced rejection should emit the supported draft token");
    // The original draw is near one. After rescaling it still rejects token
    // zero, and the only positive residual belongs to token one.
    const float pq[] = {.499999f, .5f};
    const float qq[] = {.500001f, .5f};
    Check(cudaMemcpy(target.p, pq, sizeof(pq), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(proposal.p, qq, sizeof(qq), cudaMemcpyHostToDevice));
    constexpr unsigned long long seed = 0x70fb98d7a45ca3a6ULL;
    Require(FastllmCudaMtpRecoverRoundedResidual(target.p, proposal.p, (int *)ids.p, (int *)output.p,
                correction, 0, 0, 1, 2, seed, nullptr) && correction == 1,
            "rescaled residual must sample the positive difference");
    Check(cudaMemset(target.p, 0, 2 * sizeof(float)));
    Require(!FastllmCudaMtpRecoverRoundedResidual(target.p, proposal.p, (int *)ids.p, (int *)output.p,
                correction, 0, 0, 1, 2, 42, nullptr) && correction == -1,
            "an empty target must clear the previous correction");
}
void RoundedResidualConsumerCase() {
    constexpr int batch = 3, drafts = 7, vocab = 2, stride = drafts + 1;
    const int prefixes[batch] = {0, 3, drafts};
    std::vector<float> p(batch * stride * vocab, 0.f), q(batch * drafts * vocab, 0.f);
    std::vector<int> ids(batch * drafts, 1);
    for (int b = 0; b < batch; ++b) {
        for (int step = 0; step <= drafts; ++step) p[(b * stride + step) * vocab + 1] = 1.f;
        for (int step = 0; step < drafts; ++step) q[(b * drafts + step) * vocab + 1] = 1.f;
        if (prefixes[b] < drafts) {
            const int i = b * drafts + prefixes[b];
            ids[i] = 0;
            q[i * vocab] = 1e-8f;  // In FP32 the other entry still rounds to one.
        }
    }
    Buffer target(p), proposal(q), observed(batch * stride);
    auto capture = [](const int *counts, void *dst) {
        const int *rows = FastllmCudaMtpSmallResultRows(counts, batch, drafts);
        Require(rows != nullptr, "missing small rejection token rows");
        Check(cudaMemcpyAsync(dst, rows, batch * stride * sizeof(int), cudaMemcpyDeviceToDevice,
                              cudaStreamPerThread));
    };
    std::vector<int> tokens(batch * stride), counts(batch), captured(batch * stride);
    Require(FastllmCudaMtpRejectionFromHostProbabilities(target.p, proposal.p, ids.data(),
                tokens.data(), counts.data(), batch, drafts, vocab, capture, observed.p),
            "rounded probability sampling failed");
    Check(cudaMemcpy(captured.data(), observed.p, captured.size() * sizeof(int), cudaMemcpyDeviceToHost));
    Require(captured == tokens, "early consumer observed unfinished token rows");
    for (int b = 0; b < batch; ++b) {
        Require(counts[b] == prefixes[b] && tokens[b * stride + counts[b]] == 1,
                "rounded recovery changed the accepted prefix or correction");
    }
}
void GraphTopPCase() {
    for (int vocab : {65535, 65536, 152576}) {
        for (int parameter = 0; parameter < 6; ++parameter) {
            const float temperature = parameter == 4 ? .2f : parameter == 5 ? 1.5f : 1.f;
            const float topP = parameter == 0 ? 1.f : parameter == 1 ? .5f :
                               parameter == 2 ? .01f : .95f;
            const int topK = parameter == 4 ? 50 : parameter == 5 ? 1 : vocab;
            Buffer logits(vocab), q(vocab), reference(vocab), token(1);
            void *sampler = FastllmCudaMtpCreateDraftSampler(vocab, temperature, topK, topP);
            Require(sampler != nullptr, "cannot create top-p graph sampler");
            std::vector<float> input(vocab, 0);
            Check(cudaMemcpy(logits.p, input.data(), vocab * sizeof(float), cudaMemcpyHostToDevice));
            Require(FastllmCudaMtpDraftSamplerSetSeed(sampler), "top-p graph seed failed");
            Require(FastllmCudaMtpDraftSamplerRun(sampler, logits.p, q.p, token.p), "top-p warmup failed");
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Require(FastllmCudaMtpDraftSamplerRun(sampler, logits.p, q.p, token.p), "top-p capture failed");
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int phase = 0; phase < 8; ++phase) {
                for (int id = 0; id < vocab; ++id) {
                    // Alternate sparse shortcuts and broad/overflow fallbacks in the same graph.
                    float x = -30.f;
                    if (phase == 0) x = id == 111 ? 0 : -30;
                    if (phase == 1) x = 0;
                    if (phase == 2) x = id < 32 ? -float(id % 7) / 2 : -30;
                    if (phase == 3) x = id < 1800 ? -float(id % 3) / 8 : -30;
                    if (phase == 4) x = id % 101 == 0 ? -float(id % 19) : -INFINITY;
                    if (phase == 5) x = id == vocab - 1 ? 0 : -INFINITY;
                    if (phase == 6) x = id < 2 ? std::log(.5f) : -INFINITY;
                    if (phase == 7) x = -float((id * 73) % 1000) / 16;
                    input[id] = x;
                }
                Check(cudaMemcpy(logits.p, input.data(), vocab * sizeof(float), cudaMemcpyHostToDevice));
                int unused;
                Require(FastllmCudaMtpSampleDraft(logits.p, reference.p, &temperature, &topK,
                    &topP, &unused, 1, vocab), "top-p reference failed");
                Require(FastllmCudaMtpDraftSamplerSetSeed(sampler), "top-p replay seed failed");
                Check(cudaGraphLaunch(executable, cudaStreamPerThread));
                auto actual = q.Read(), expected = reference.Read();
                double error = 0, sum = 0;
                for (int id = 0; id < vocab; ++id) {
                    Require(std::isfinite(actual[id]) && actual[id] >= 0,
                            "top-p graph produced an invalid probability");
                    Require((actual[id] > 0) == (expected[id] > 0),
                            "top-p graph changed the filtered support");
                    error += std::fabs(actual[id] - expected[id]);
                    sum += actual[id];
                }
                Require(error < 1e-4 && std::fabs(sum - 1) < 1e-5,
                        "top-p graph disagrees with full-vocabulary reference");
                int id = (int)token.Read()[0];
                Require(id >= 0 && id < vocab && expected[id] > 0, "top-p graph sampled outside support");
            }
            Check(cudaGraphExecDestroy(executable));
            Check(cudaGraphDestroy(graph));
            FastllmCudaMtpDestroyDraftSampler(sampler);
        }
    }
    std::puts("graph top-p: sparse/dense transitions, ties, masks, joint top-k, fallback boundaries: PASS");
}

void ResizableGraphSamplerCase() {
    const int capacity = 64, vocab = 65536;
    std::vector<float> temperatures(capacity, 1.f), topPs(capacity, 1.f);
    std::vector<int> topKs(capacity, 0);
    void *sampler = FastllmCudaMtpCreateDraftSamplerBatch(capacity, vocab,
        temperatures.data(), topKs.data(), topPs.data());
    Require(sampler, "cannot create bounded sampler");
    Require(!FastllmCudaMtpDraftSamplerSetParameters(sampler, capacity + 1,
        temperatures.data(), topKs.data(), topPs.data()), "sampler accepted out-of-capacity rows");
    Buffer logits((size_t)capacity * vocab), q((size_t)capacity * vocab), expected((size_t)capacity * vocab);
    Buffer token(capacity), referenceToken(capacity);
    std::map<int, std::pair<cudaGraph_t, cudaGraphExec_t>> graphs;
    int phase = 0;
    for (int rows : {1, 8, 33, 64, 3, 32, 64, 1}) {
        std::vector<float> values((size_t)rows * vocab);
        for (int row = 0; row < rows; ++row) {
            temperatures[row] = .7f + ((row + phase) % 9) * .07f;
            for (int id = 0; id < vocab; ++id)
                values[(size_t)row * vocab + id] = std::sin((id + row + phase) * .031f) * 3;
        }
        Check(cudaMemcpy(logits.p, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
        Require(FastllmCudaMtpDraftSamplerSetParameters(sampler, rows, temperatures.data(), topKs.data(), topPs.data()),
            "bounded sampler parameter update failed");
        Require(FastllmCudaMtpDraftSamplerSetSeed(sampler), "bounded sampler seed failed");
        if (!graphs.count(rows)) {
            Require(FastllmCudaMtpDraftSamplerRunBatch(sampler, logits.p, q.p, token.p, 0), "bounded sampler warmup failed");
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            auto &g = graphs[rows];
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Require(FastllmCudaMtpDraftSamplerRunBatch(sampler, logits.p, q.p, token.p, 0), "bounded sampler capture failed");
            Check(cudaStreamEndCapture(cudaStreamPerThread, &g.first));
            Check(cudaGraphInstantiate(&g.second, g.first, nullptr, nullptr, 0));
        }
        Check(cudaGraphLaunch(graphs.at(rows).second, cudaStreamPerThread));
        auto actual = q.Read(); auto ids = token.Read();
        void *reference = FastllmCudaMtpCreateDraftSamplerBatch(rows, vocab,
            temperatures.data(), topKs.data(), topPs.data());
        Require(reference && FastllmCudaMtpDraftSamplerSetSeed(reference) &&
            FastllmCudaMtpDraftSamplerRunBatch(reference, logits.p, expected.p, referenceToken.p, 0),
            "reference sampler failed");
        auto oracle = expected.Read();
        for (size_t i = 0; i < values.size(); ++i)
            Require(actual[i] == oracle[i], "resized sampler changed proposal probabilities");
        for (int row = 0; row < rows; ++row)
            Require(ids[row] >= 0 && ids[row] < vocab && ids[row] == (int)ids[row], "resized sampler returned invalid token");
        FastllmCudaMtpDestroyDraftSampler(reference); ++phase;
    }
    for (auto &entry : graphs) {
        Check(cudaGraphExecDestroy(entry.second.second)); Check(cudaGraphDestroy(entry.second.first));
    }
    FastllmCudaMtpDestroyDraftSampler(sampler);
    std::puts("bounded sampler: batch growth/shrink, changing temperatures and graph replay: PASS");
}

void GraphSamplerCase() {
    const int vocab = 152576, repeats = 2048;
    const int selected[] = {17, 111, vocab - 1};
    const float ts = 1, pp = .95f;
    const int kk = vocab;
    Buffer logits(vocab), q(vocab), reference(vocab), token(1);
    void *state = FastllmCudaMtpCreateDraftSampler(vocab, ts, kk, pp);
    Require(state != nullptr, "cannot create graph sampler");
    std::vector<float> input(vocab, -INFINITY);
    for (int i = 0; i < 3; ++i) input[selected[i]] = std::log((3-i)/6.f);
    Check(cudaMemcpy(logits.p, input.data(), vocab * sizeof(float), cudaMemcpyHostToDevice));
    Require(FastllmCudaMtpDraftSamplerSetSeed(state), "graph seed update failed");
    Require(FastllmCudaMtpDraftSamplerRun(state, logits.p, q.p, token.p), "graph warmup failed");
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(FastllmCudaMtpDraftSamplerRun(state, logits.p, q.p, token.p), "graph capture failed");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int phase = 0; phase < 2; ++phase) {
        const float probabilities[2][3] = {{.6f,.3f,.1f}, {.1f,.2f,.7f}};
        for (int i = 0; i < 3; ++i) input[selected[i]] = std::log(probabilities[phase][i]);
        Check(cudaMemcpy(logits.p, input.data(), vocab * sizeof(float), cudaMemcpyHostToDevice));
        int unused;
        Require(FastllmCudaMtpSampleDraft(logits.p, reference.p, &ts, &kk, &pp, &unused, 1, vocab),
                "graph reference sampling failed");
        auto expected = reference.Read();
        int counts[3] = {}, unchanged = 0, last = -1;
        for (int n = 0; n < repeats; ++n) {
            Require(FastllmCudaMtpDraftSamplerSetSeed(state), "replay seed update failed");
            Check(cudaGraphLaunch(executable, cudaStreamPerThread));
            float id;
            Check(cudaMemcpy(&id, token.p, sizeof(id), cudaMemcpyDeviceToHost));
            auto found = std::find(selected, selected + 3, (int)id);
            Require(found != selected + 3, "graph sampled outside its probability support");
            ++counts[found-selected];
            unchanged += (int)id == last;
            last = (int)id;
        }
        auto actual = q.Read();
        double error = 0;
        for (int i = 0; i < vocab; ++i) {
            Require((actual[i] > 0) == (expected[i] > 0), "graph q support changed");
            error += std::fabs(actual[i] - expected[i]);
        }
        Require(error < 1e-6, "graph q differs from the eager sampler");
        Require(unchanged < repeats * .85, "graph replay reused a fixed random draw");
        for (int i = 0; i < 3; ++i)
            Require(std::fabs(double(counts[i])/repeats - probabilities[phase][i]) < .045,
                    "graph draws disagree with the saved q distribution");
    }
    Check(cudaGraphExecDestroy(executable));
    Check(cudaGraphDestroy(graph));
    FastllmCudaMtpDestroyDraftSampler(state);
    std::puts("sampling Graph: changing logits, fresh RNG, saved q, independent scratch: PASS");
}

void VerifyProposalLayoutCase() {
    constexpr int count = 205, width = 4, stride = count + 7;
    std::vector<float> stepMajor(width * stride);
    std::iota(stepMajor.begin(), stepMajor.end(), 1.f);
    Buffer source(stepMajor), result(count * width);
    std::vector<const float *> rows(count);
    std::vector<int> strides(count, stride), lengths(count);
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> expected(count * width, 0.f);
        for (int row = 0; row < count; ++row) {
            const int slot = pass ? count - 1 - row : row;
            lengths[row] = 1 + (row + pass) % width;
            rows[row] = source.p + slot;
            for (int step = 0; step < lengths[row]; ++step)
                expected[row * width + step] = stepMajor[step * stride + slot];
        }
        Require(FastllmCudaMtpGatherVerifyIds(result.p, width, rows.data(), strides.data(),
                    lengths.data(), count, width * stride + 1), "verification input gather failed");
        Require(result.Read() == expected, "verification tokens changed across slot/length updates");
    }
    lengths.back() = width + 1;
    Require(!FastllmCudaMtpGatherVerifyIds(result.p, width, rows.data(), strides.data(),
                lengths.data(), count, width * stride + 1), "out-of-range proposal length accepted");
}

void LargeVocabularyCase() {
    for (int vocab : {65535, 65536, 65537, 152576, 153001}) {
        const int rows = vocab == 65537 ? 33 : 8;
        std::vector<float> logits((size_t)rows * vocab), ts(rows), ps(rows);
        std::vector<int> ks(rows), ids(rows);
        for (int row = 0; row < rows; ++row) {
            ts[row] = row % 4 == 0 ? .2f : row % 4 == 1 ? .7f : row % 4 == 2 ? 1 : 1.5f;
            ps[row] = row % 4 == 0 ? 1 : row % 4 == 1 ? .95f : row % 4 == 2 ? .5f : .01f;
            ks[row] = row % 3 == 0 ? 0 : row % 3 == 1 ? 50 : vocab;
            for (int id = 0; id < vocab; ++id) {
                float value = -float((id * 73 + row * 19) % 1000) / 16;
                if (row % 4 == 0) value = 0; // Equal-probability ties.
                if (row % 4 == 2 && id % 101 != 0) value = -INFINITY;
                if (row % 4 == 3) value = id == vocab - 1 ? 0 : -INFINITY;
                logits[(size_t)row * vocab + id] = value;
            }
        }
        Buffer input(logits), proposals(logits.size());
        Require(FastllmCudaMtpSampleDraft(input.p, proposals.p, ts.data(), ks.data(), ps.data(),
                                         ids.data(), rows, vocab), "large-vocabulary sampling failed");
        auto actual = proposals.Read();
        for (int row = 0; row < rows; ++row) {
            auto begin = logits.begin() + (size_t)row * vocab;
            auto expected = Oracle(std::vector<float>(begin, begin + vocab), ts[row], ks[row], ps[row]);
            double error = 0, sum = 0;
            for (int id = 0; id < vocab; ++id) {
                float p = actual[(size_t)row * vocab + id];
                Require(std::isfinite(p) && p >= 0, "invalid large-vocabulary probability");
                Require((p > 0) == (expected[id] > 0), "large-vocabulary filter support changed");
                error += std::fabs(p - expected[id]);
                sum += p;
            }
            Require(error < 1e-4 && std::fabs(sum - 1) < 1e-5,
                    "large-vocabulary probabilities disagree with CPU oracle");
            Require(ids[row] >= 0 && ids[row] < vocab && expected[ids[row]] > 0,
                    "large-vocabulary token outside filtered support");
        }
    }
    std::puts("large-vocabulary dispatch boundaries, masks, ties, joint filtering: PASS");
}

void AcceptedConsumerCase(int vocab, std::initializer_list<int> draftCounts) {
    // Point masses give an exact acceptance oracle, including ragged prefixes.
    for (int batch : {1, 2, 4, 8, 9, 16}) for (int drafts : draftCounts) {
        const int rows = batch * (drafts + 1);
        std::vector<float> logits((size_t)rows * vocab, -INFINITY);
        std::vector<float> probs((size_t)batch * drafts * vocab, 0);
        std::vector<float> temperatures(rows, .4f), topPs(rows, .95f);
        std::vector<int> topKs(rows, vocab), ids(batch * drafts), expected(batch);
        for (int b = 0; b < batch; ++b) {
            expected[b] = (b + drafts / 2) % (drafts + 1);
            for (int d = 0; d <= drafts; ++d) {
                int token = 3 + b * 23 + d;
                logits[((size_t)b * (drafts + 1) + d) * vocab + token] = 0;
                if (d < drafts) {
                    int proposal = d < expected[b] ? token : token + 500;
                    ids[b * drafts + d] = proposal;
                    probs[((size_t)b * drafts + d) * vocab + proposal] = 1;
                }
            }
        }
        Buffer input(logits), proposal(probs), consumed(batch);
        struct Consumer { int *output; int count; int calls = 0; };
        Consumer consumer{reinterpret_cast<int *>(consumed.p), batch};
        auto enqueue = [](const int *accepted, void *opaque) {
            auto &c = *static_cast<Consumer *>(opaque);
            ++c.calls;
            Check(cudaMemcpyAsync(c.output, accepted, c.count * sizeof(int),
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        };
        std::vector<int> tokens(rows, -2), accepted(batch, -2), deviceCounts(batch, -2);
        Require(FastllmCudaMtpRejectionSamplingWithConsumer(input.p, proposal.p,
            temperatures.data(), topKs.data(), topPs.data(), ids.data(),
            tokens.data(), accepted.data(), batch, drafts, vocab, enqueue, &consumer),
            "accepted-prefix consumer sampling failed");
        Check(cudaDeviceSynchronize());
        Require(accepted == expected, "accepted prefix differs from exact oracle");
        for (int b = 0; b < batch; ++b) for (int d = 0; d <= accepted[b]; ++d)
            Require(tokens[b * (drafts + 1) + d] == 3 + b * 23 + d,
                    "consumer path emitted an incorrect token");
        Require(consumer.calls == (batch <= 8 ? 1 : 0),
                "consumer capability fallback changed");
        if (consumer.calls) {
            Check(cudaMemcpy(deviceCounts.data(), consumed.p, batch * sizeof(int), cudaMemcpyDeviceToHost));
            Require(deviceCounts == expected, "device prefix counts differ from host results");
        }
    }
    std::printf("ACCEPTED_CONSUMER_PASS vocab=%d, exact ragged prefixes and fallback\n", vocab);
}
}
int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n) return 77;
    try {
        Check(cudaSetDevice(0));
        AcceptedConsumerCase(16384, {1, 3, 7, 8});
        AcceptedConsumerCase(152576, {3}); // Sparse top-p with a singleton nucleus.
        FilterCase();
        DeviceFilterCase();
#ifndef USE_ROCM
        IndexedBf16ProjectionCase();
#endif
        MarkovGraphCase();
        MarkovDrawPlateauCase();
        FullVocabularyCase();
        VerifyProposalLayoutCase();
        LargeVocabularyCase();
        GraphSamplerCase();
        ResizableGraphSamplerCase();
        GraphTopPCase();
        PointMassCase();
        RoundedResidualCase();
        RoundedResidualConsumerCase();
        for (int drafts : {1, 3, 5, 7})
            ChainCase("unfiltered-q filtered-p", {.55f, .30f, .15f}, {.10f, .40f, .50f},
                  drafts, 32, .7f, 2, .8f);
        ChainCase("top-k before nucleus", {.50f,.30f,.20f}, {.25f,.25f,.50f},
                  3, 32, 1.f, 2, .59f, true);
        ChainCase("stable finite-k tie", {.4f,.3f,.3f}, {.3f,.3f,.4f},
                  1, 32, .8f, 2, 1.f, true);
        ChainCase("identical", {.6f,.3f,.1f}, {.6f,.3f,.1f}, 3, 32);
        ChainCase("different", {.6f,.3f,.1f}, {.3f,.5f,.2f}, 3, 17);
        ChainCase("seven drafts", {.6f,.3f,.1f}, {.3f,.5f,.2f}, 7, 32);
        ChainCase("two-wide", {.6f,.3f,.1f}, {.3f,.5f,.2f}, 3, 18);
        ChainCase("loop exit", {.6f,.3f,.1f}, {.95f,.04f,.01f}, 1, 32);
        ChainCase("disjoint", {0,1,0}, {1,0,0}, 3, 32);
        std::puts("MTP rejection sampling: PASS");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "MTP rejection sampling: %s\n", e.what());
        return 1;
    }
}
