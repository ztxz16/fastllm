#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda.cuh"
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
    for (int i = 0; i < (int)p.size(); ++i) {
        double mass = 0; int count = 0;
        for (double v : original) if (v > original[i]) { mass += v; ++count; }
        if ((k > 0 && count >= k) || mass >= topP) p[i] = 0;
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
}
int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n) return 77;
    try {
        Check(cudaSetDevice(0));
        FilterCase();
        ChainCase("identical", {.6f,.3f,.1f}, {.6f,.3f,.1f}, 3, 32);
        ChainCase("different", {.6f,.3f,.1f}, {.3f,.5f,.2f}, 3, 17);
        ChainCase("loop exit", {.6f,.3f,.1f}, {.95f,.04f,.01f}, 1, 32);
        ChainCase("disjoint", {0,1,0}, {1,0,0}, 3, 32);
        std::puts("MTP rejection sampling: PASS");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "MTP rejection sampling: %s\n", e.what());
        return 1;
    }
}
