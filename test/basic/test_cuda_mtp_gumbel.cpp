#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void Check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Buffer {
    T *p = nullptr; size_t n;
    explicit Buffer(size_t n) : n(n) { Check(cudaMalloc(&p, n * sizeof(T))); }
    explicit Buffer(const std::vector<T> &v) : Buffer(v.size()) {
        Check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Buffer() { cudaFree(p); }
    std::vector<T> Read() const {
        std::vector<T> v(n); Check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
};
void ProposalCase() {
    const int batch = 16384, vocab = 257;
    std::vector<float> base(vocab, -INFINITY), x((size_t)batch * vocab), t(batch);
    for (int i = 0; i < 9; ++i) base[i] = -0.37f * i;
    for (int b = 0; b < batch; ++b) {
        std::copy(base.begin(), base.end(), x.begin() + (size_t)b * vocab);
        t[b] = b % 2 ? 0.7f : 1.5f;
    }
    Buffer<float> logits(x), saved(x.size()), lse(batch), floats(batch);
    Buffer<int> ids(batch);
    Require(FastllmCudaMtpSampleDraftLogits(logits.p, saved.p, lse.p, ids.p,
        floats.p, t.data(), batch, vocab), "Gumbel sampling failed");
    auto cache = saved.Read(), normalizers = lse.Read(), floatIds = floats.Read();
    auto tokens = ids.Read();
    int counts[2][9] = {};
    for (int b = 0; b < batch; ++b) {
        Require(tokens[b] >= 0 && tokens[b] < 9 && floatIds[b] == tokens[b], "invalid draft token");
        ++counts[b % 2][tokens[b]];
        double sum = 0;
        for (int i = 0; i < 9; ++i) {
            float expected = base[i] / t[b];
            Require(std::fabs(cache[(size_t)b * vocab + i] - expected) < 1e-6, "cache temperature mismatch");
            sum += std::exp(double(expected));
        }
        Require(std::fabs(normalizers[b] - std::log(sum)) < 2e-6, "proposal normalizer mismatch");
    }
    for (int g = 0; g < 2; ++g) {
        double sum = 0;
        for (int i = 0; i < 9; ++i) sum += std::exp(double(base[i] / t[g]));
        for (int i = 0; i < 9; ++i)
            Require(std::fabs(double(counts[g][i]) / (batch / 2) - std::exp(double(base[i] / t[g])) / sum) < .025,
                "Gumbel frequency does not match q");
    }
    std::puts("proposal temperatures, logits cache, normalizers, frequencies: PASS");
}
std::vector<double> TargetOracle(const std::vector<float> &values, float temperature, int topK, float topP) {
    std::vector<double> probs(values.begin(), values.end());
    for (double &x : probs) x = std::pow(x, 1.0 / temperature);
    double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    for (double &x : probs) x /= sum;
    const auto original = probs;
    for (size_t i = 0; i < probs.size(); ++i) {
        int greater = 0; double mass = 0;
        for (double x : original) if (x > original[i]) { ++greater; mass += x; }
        if ((topK > 0 && greater >= topK) || mass >= topP) probs[i] = 0;
    }
    sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    for (double &x : probs) x /= sum;
    return probs;
}
void ChainCase(const char *label, const std::vector<float> &p, const std::vector<float> &q,
        float temperature = 1, int topK = 32, float topP = 1) {
    const int batch = 16384, drafts = 3, vocab = 32;
    const std::vector<float> bonus{.1f, .2f, .7f};
    const auto expected = TargetOracle(p, temperature, topK, topP);
    const auto expectedBonus = TargetOracle(bonus, temperature, topK, topP);
    std::vector<float> ql((size_t)batch * drafts * vocab, -INFINITY);
    std::vector<float> pl((size_t)batch * (drafts + 1) * vocab, -INFINITY);
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < drafts; ++d)
            for (int i = 0; i < 3; ++i) {
                ql[((size_t)b * drafts + d) * vocab + i] = std::log(q[i]);
                pl[((size_t)b * (drafts + 1) + d) * vocab + i] = std::log(p[i]);
            }
        for (int i = 0; i < 3; ++i) pl[((size_t)b * (drafts + 1) + drafts) * vocab + i] = std::log(bonus[i]);
    }
    Buffer<float> qlogits(ql), target(pl), saved(ql.size()), lse(batch * drafts);
    Buffer<int> ids(batch * drafts);
    std::vector<float> qt(batch * drafts, 1), pt(batch * (drafts + 1), temperature), pp(pt.size(), topP);
    std::vector<int> pk(pt.size(), topK), output(pt.size()), accepted(batch);
    Require(FastllmCudaMtpSampleDraftLogits(qlogits.p, saved.p, lse.p, ids.p, nullptr,
        qt.data(), batch * drafts, vocab), "chain draft failed");
    auto cache = saved.Read();
    Require(FastllmCudaMtpRejectionSamplingLogits(target.p, saved.p, lse.p, ids.p,
        pt.data(), pk.data(), pp.data(), output.data(), accepted.data(), batch, drafts, vocab), "chain verify failed");
    Require(cache == saved.Read(), "verifier mutated q");
    auto proposals = ids.Read();
    int counts[3] = {}, bonuses[3] = {}, prefix[3] = {}, allAccepted = 0;
    for (int b = 0; b < batch; ++b) {
        int n = accepted[b], off = b * (drafts + 1);
        Require(n >= 0 && n <= drafts && output[off] >= 0 && output[off] < 3, "invalid output");
        Require(expected[output[off]] > 0, "target constraint was violated");
        ++counts[output[off]];
        for (int i = 0; i < n; ++i) {
            ++prefix[i]; Require(output[off + i] == proposals[b * drafts + i], "accepted prefix mismatch");
        }
        for (int i = n + 1; i <= drafts; ++i) Require(output[off + i] == -1, "rejected suffix escaped");
        if (n == drafts) { ++allAccepted; ++bonuses[output[off + n]]; }
    }
    double overlap = 0;
    for (int i = 0; i < 3; ++i) {
        overlap += std::min(expected[i], double(q[i]));
        Require(std::fabs(double(counts[i]) / batch - expected[i]) < .025, "target distribution changed");
        if (allAccepted > 1000)
            Require(std::fabs(double(bonuses[i]) / allAccepted - expectedBonus[i]) <
                .005 + 6 * std::sqrt(expectedBonus[i] * (1 - expectedBonus[i]) / allAccepted), "bonus distribution changed");
    }
    for (int i = 0; i < drafts; ++i)
        Require(std::fabs(double(prefix[i]) / batch - std::pow(overlap, i + 1)) < .025, "acceptance ratio mismatch");
    if (p == q && temperature == 1 && topK == 32 && topP == 1) Require(allAccepted == batch, "p=q should accept all drafts");
    std::printf("%s target=[%.4f %.4f %.4f] acceptance=[%.4f %.4f %.4f]: PASS\n", label,
        double(counts[0])/batch, double(counts[1])/batch, double(counts[2])/batch,
        double(prefix[0])/batch, double(prefix[1])/batch, double(prefix[2])/batch);
}
void LargeVocabAndGraphCase() {
    const int vocab = 248320;
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; ++i) logits[i] = -float(i % 997) / 111;
    logits[0] = -INFINITY;
    Buffer<float> input(logits), saved(vocab), lse(1);
    Buffer<int> output(1);
    float t = .73f;
    auto run = [&]() { Require(FastllmCudaMtpSampleDraftLogits(input.p, saved.p, lse.p,
        output.p, nullptr, &t, 1, vocab), "large-vocab sampling failed"); };
    run(); Check(cudaDeviceSynchronize());
    double total = 0;
    for (float x : logits) total += std::exp(double(x / t));
    Require(std::fabs(lse.Read()[0] - std::log(total)) < 3e-6, "large-vocab normalizer failed");
    cudaGraph_t graph; cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    run(); Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    int changed = 0, previous = -1;
    for (int i = 0; i < 64; ++i) {
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        int id = output.Read()[0];
        Require(id > 0 && id < vocab, "graph sampled a masked or invalid token");
        changed += id != previous; previous = id;
    }
    Require(changed > 50, "CUDA Graph replay reused draft randomness");
    cudaEvent_t start, end; Check(cudaEventCreate(&start)); Check(cudaEventCreate(&end));
    Check(cudaEventRecord(start, cudaStreamPerThread));
    for (int i = 0; i < 1000; ++i) Check(cudaGraphLaunch(exec, cudaStreamPerThread));
    Check(cudaEventRecord(end, cudaStreamPerThread)); Check(cudaEventSynchronize(end));
    float ms; Check(cudaEventElapsedTime(&ms, start, end));
    std::printf("vocab=%d graph sample/cache/lse %.3f us, fresh replay RNG: PASS\n", vocab, ms);
    cudaEventDestroy(start); cudaEventDestroy(end); cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
}
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        Check(cudaSetDevice(0)); ProposalCase();
        ChainCase("p=q", {.6f,.3f,.1f}, {.6f,.3f,.1f});
        ChainCase("loop exit", {.6f,.3f,.1f}, {.95f,.04f,.01f});
        ChainCase("different support", {.1f,.2f,.7f}, {.6f,.3f,.1f});
        ChainCase("disjoint support", {0,0,1}, {.8f,.2f,0});
        ChainCase("target top-k/top-p", {.6f,.3f,.1f}, {.2f,.3f,.5f}, 1, 2, .7f);
        ChainCase("target temperature", {.6f,.3f,.1f}, {.2f,.3f,.5f}, 1.5f, 2, .95f);
        ChainCase("target top-k ties", {.4f,.4f,.2f}, {.6f,.3f,.1f}, .7f, 1, 1);
        LargeVocabAndGraphCase(); std::puts("MTP Gumbel/rejection: PASS"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
