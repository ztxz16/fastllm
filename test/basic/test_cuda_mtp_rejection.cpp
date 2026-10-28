#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-dspark.h"
#include "baseblock.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
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
std::vector<double> Oracle(const std::vector<float> &logits, float t, int k, float topP) {
    std::vector<double> p(logits.size());
    double maxLogit = *std::max_element(logits.begin(), logits.end());
    for (int i = 0; i < (int)p.size(); ++i) p[i] = std::exp((logits[i] - maxLogit) / t);
    double sum = std::accumulate(p.begin(), p.end(), 0.0);
    for (auto &v : p) v /= sum;
    auto original = p;
    // FastLLM's joint sampler uses strict-greater counts/mass, including ties.
    std::vector<int> order(p.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return original[a] > original[b]; });
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
    cudaFree(token);
    std::puts("DEVICE_FILTER_GRAPH_PASS 24 distributions, 49152 replays, saved q and RNG checked");
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
               int drafts, int vocab) {
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
    std::vector<float> qt(qRows, 1), qp(qRows, 1), pt(pRows, 1), pp(pRows, 1);
    std::vector<int> qk(qRows, vocab), pk(pRows, vocab), ids(qRows), out(pRows), accepted(batch);
    Require(FastllmCudaMtpSampleDraft(ql.p, q.p, qt.data(), qk.data(), qp.data(), ids.data(),
                                     qRows, vocab), "chain proposal sampling failed");
    auto snapshot = q.Read();
    Require(FastllmCudaMtpRejectionSampling(pl.p, q.p, pt.data(), pk.data(), pp.data(),
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
            Require(std::fabs(double(bonuses[i])/allAccepted - bonus[i]) < .03, "wrong bonus distribution");
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
}
int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n) return 77;
    try {
        Check(cudaSetDevice(0));
        FilterCase();
        DeviceFilterCase();
        MarkovGraphCase();
        FullVocabularyCase();
        LargeVocabularyCase();
        GraphSamplerCase();
        GraphTopPCase();
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
