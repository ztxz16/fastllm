// Device Markov proposals: conditional q, fresh graph RNG, and target joint law.
#include "models/deepseekv41.h"
#include "baseblock.h"
#include "devices/cuda/fastllm-cuda-dspark.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
static void Cuda(cudaError_t e) { Check(e == cudaSuccess, cudaGetErrorString(e)); }
struct Probe : DeepSeekV41Model {
    using DeepSeekV41Model::DsparkSampleVerify;
};
struct Fixture {
    int vocab, rank, block;
    float temperature;
    std::vector<float> e, h, x;
    Data embed, head, logits, tokens, probs, samplingScratch, out;
    Fixture(int v, int r, int b, DataType type, float t, bool smallSupport = false)
        : vocab(v), rank(r), block(b), temperature(t), e(v * r), h(v * r), x(b * v),
          tokens(DataType::INT32, {b + 1}) {
        for (int i = 0; i < v * r; ++i) {
            e[i] = ((i * 13 + i / r * 3) % 31 - 15) / 32.0f;
            h[i] = ((i * 7 + i / r * 5) % 29 - 14) / 32.0f;
        }
        for (int i = 0; i < b * v; ++i)
            x[i] = smallSupport && i % v >= 4 ? -INFINITY : ((i * 11 + i / v * 7) % 23 - 11) / 8.0f;
        embed.CopyFrom(Data(type, {v, r}, e));
        head.CopyFrom(Data(type, {v, r}, h));
        logits.CopyFrom(Data(FLOAT32, {1, b, v}, x));
        for (Data *d : {&embed, &head, &logits, &tokens})
            d->ToDevice(DataDevice::CUDA, std::vector<int>{0});
        tokens.Allocate();
        Anchor(0);
    }
    void Anchor(int a) { Cuda(cudaMemcpy(tokens.cudaData, &a, sizeof(a), cudaMemcpyHostToDevice)); }
    bool Run() {
        return FastllmCudaDeepSeekV41MarkovSampleDevice(logits, embed, head, block, temperature, tokens,
                                                        probs, samplingScratch, out);
    }
    std::vector<int> Tokens() {
        std::vector<int> t(block + 1);
        Cuda(cudaMemcpy(t.data(), tokens.cudaData, t.size() * sizeof(int), cudaMemcpyDeviceToHost));
        return t;
    }
    std::vector<double> Q(int step, int previous) {
        std::vector<double> q(vocab);
        double maximum = -INFINITY, total = 0;
        for (int v = 0; v < vocab; ++v) {
            double score = x[step * vocab + v];
            for (int d = 0; d < rank; ++d)
                score += (double)e[previous * rank + d] * h[v * rank + d];
            q[v] = (float)score;
            maximum = std::max(maximum, q[v]);
        }
        for (double &v : q) {
            v = std::exp((v - maximum) / std::max(temperature, 1.0e-6f));
            total += v;
        }
        for (double &v : q)
            v /= total;
        return q;
    }
    void Numeric() {
        auto t = Tokens();
        std::vector<float> p(block * vocab), embeddings(block * rank);
        Cuda(cudaMemcpy(p.data(), probs.cudaData, p.size() * sizeof(float), cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(embeddings.data(), out.cudaData, embeddings.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
        for (int s = 0; s < block; ++s) {
            Check(t[s + 1] >= 0 && t[s + 1] < vocab, "sampled token out of range");
            auto q = Q(s, t[s]);
            double sum = 0, l1 = 0;
            for (int v = 0; v < vocab; ++v) {
                sum += p[s * vocab + v];
                l1 += std::abs(p[s * vocab + v] - q[v]);
            }
            if (std::abs(sum - 1) >= 1e-5 || l1 >= 1e-5)
                std::cerr << "dtype=" << embed.dataType << " vocab=" << vocab << " block=" << block
                          << " step=" << s << " anchor=" << t[s] << " sum=" << sum << " L1=" << l1
                          << " q0=" << p[s * vocab] << " ref0=" << q[0] << '\n';
            Check(std::abs(sum - 1) < 1e-5 && l1 < 1e-5, "stored q differs from conditional softmax");
            for (int d = 0; d < rank; ++d)
                Check(embeddings[s * rank + d] == e[t[s] * rank + d], "wrong conditional embedding");
        }
    }
};
struct SampleGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    explicit SampleGraph(Fixture &fixture) {
        Check(fixture.Run(), "sampler warmup failed");
        Cuda(cudaDeviceSynchronize());
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Check(fixture.Run(), "sampler capture failed");
        Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    ~SampleGraph() {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }
    void Run() { Cuda(cudaGraphLaunch(exec, cudaStreamPerThread)); }
};
static double JointZ(const std::vector<int> &counts, const std::vector<double> &expected, int samples) {
    double z = 0;
    for (int i = 0; i < 64; ++i)
        z = std::max(z, std::abs(counts[i] - samples * expected[i]) /
                            std::sqrt(samples * expected[i] * (1 - expected[i])));
    return z;
}
static std::array<double, 4> Target(int previous) {
    std::array<double, 4> p{};
    for (int i = 0; i < 4; ++i)
        p[i] = (1 + (i + previous) % 4) / 10.0;
    return p;
}
static void Distribution(bool equal) {
    constexpr int trials = 6000;
    Fixture f(128, 7, 3, BFLOAT16, .7f, true);
    SampleGraph graph(f);
    Probe model;
    DeepSeekV41DsparkState proposal;
    DeepSeekV41SpecScratch scratch;
    GenerationConfig config;
    config.do_sample = true;
    config.top_k = 128;
    config.top_p = 1;
    config.temperature = 1;
    std::vector<float> target(4 * 128, -INFINITY);
    Data gpu(FLOAT32, {1, 4, 128}, target);
    gpu.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    std::mt19937 rng(12731);
    std::vector<int> raw(64), verified(64), accepted(4);
    std::vector<double> qjoint(64), pjoint(64);
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int c = 0; c < 4; ++c) {
                int i = (a * 4 + b) * 4 + c;
                qjoint[i] = f.Q(0, 0)[a] * f.Q(1, a)[b] * f.Q(2, b)[c];
                pjoint[i] = equal ? qjoint[i] : Target(0)[a] * Target(a)[b] * Target(b)[c];
            }
    for (int n = 0; n < trials; ++n) {
        graph.Run();
        auto ids = f.Tokens();
        if (n < 3)
            f.Numeric();
        for (int i = 1; i <= 3; ++i)
            Check(ids[i] >= 0 && ids[i] < 4, "masked or invalid token sampled");
        ++raw[(ids[1] * 4 + ids[2]) * 4 + ids[3]];
        for (int row = 0; row < 4; ++row) {
            auto q = f.Q(std::min(row, 2), ids[row]);
            auto p = Target(ids[row]);
            for (int v = 0; v < 4; ++v)
                target[row * 128 + v] = std::log(equal ? q[v] : p[v]);
        }
        Cuda(cudaMemcpy(gpu.cudaData, target.data(), target.size() * sizeof(float), cudaMemcpyHostToDevice));
        proposal.sampledProposal = true;
        proposal.proposalTokens.assign(ids.begin() + 1, ids.end());
        proposal.proposalProbs.CopyFrom(f.probs);
        scratch.draftTokens = proposal.proposalTokens;
        model.DsparkSampleVerify(gpu, proposal, scratch, config);
        ++accepted.at(scratch.acceptedDrafts);
        int previous = 0, joint = 0;
        for (int i = 0; i < 3; ++i) {
            auto p = Target(previous);
            auto q = f.Q(i, previous);
            int token;
            if (i <= scratch.acceptedDrafts)
                token = scratch.tokens[i];
            else if (equal)
                token = std::discrete_distribution<int>(q.begin(), q.end())(rng);
            else
                token = std::discrete_distribution<int>(p.begin(), p.end())(rng);
            previous = token;
            joint = joint * 4 + token;
        }
        ++verified.at(joint);
    }
    double qz = JointZ(raw, qjoint, trials), pz = JointZ(verified, pjoint, trials);
    Check(qz < 7 && pz < 7, "graph sampling/rejection joint distribution mismatch");
    if (equal)
        Check(accepted[3] >= trials - 2, "p=q did not accept the chain");
    else
        Check(accepted[0] > 0 && accepted[1] > 0 && accepted[3] > 0, "missing rejection branch coverage");
    std::cout << "graph joint p=q:" << equal << " raw_z=" << qz << " verified_z=" << pz << " PASS\n";
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 77;
    try {
        Cuda(cudaSetDevice(0));
        for (auto type : {FLOAT32, FLOAT16, BFLOAT16})
            for (int block = 1; block <= 8; ++block) {
                Fixture f(17, 7, block, type, block % 2 ? .7f : 1.3f);
                Check(f.Run(), "generic sample failed");
                f.Numeric();
                SampleGraph graph(f);
                for (int anchor : {3, 16}) {
                    f.Anchor(anchor);
                    graph.Run();
                    f.Numeric();
                }
            }
        for (auto type : {FLOAT32, FLOAT16, BFLOAT16}) {
            Fixture f(19, 512, 5, type, 1.3f);
            Check(f.Run(), "max-rank sample failed");
            f.Numeric();
        }
        {
            Fixture f(17, 7, 5, BFLOAT16, 1.0e-8f);
            std::fill(f.x.begin(), f.x.end(), 1.0e8f);
            Cuda(cudaMemcpy(f.logits.cudaData, f.x.data(), f.x.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
            int changes = 0, previous = -1;
            for (int n = 0; n < 256; ++n) {
                Check(f.Run(), "large-offset sampling failed");
                f.Numeric();
                int token = f.Tokens()[1];
                changes += token != previous;
                previous = token;
            }
            Check(changes > 128, "large logit offset rounded away random noise");
            std::cout << "large common offset / temperature clamp / fresh RNG PASS\n";
        }
        // Exercise grouped BF16, its partial final lane group, and production vocabulary.
        for (int vocab : {32769, 129280}) {
            Fixture f(vocab, 256, 5, BFLOAT16, 1.0f);
            Check(f.Run(), "grouped sample failed");
            f.Numeric();
            {
                SampleGraph graph(f);
                for (int anchor : {3, vocab - 1, 7}) {
                    f.Anchor(anchor);
                    graph.Run();
                    f.Numeric();
                }
            }
            f.temperature = .7f;
            {
                SampleGraph graph(f);
                graph.Run();
                f.Numeric();
            }
            f.temperature = 0;
            Check(!f.Run(), "invalid temperature accepted");
            f.temperature = 1;
            f.logits.strides[1]++;
            Check(!f.Run(), "strided logits accepted");
            --f.logits.strides[1];
            f.tokens.dataType = FLOAT32;
            Check(!f.Run(), "invalid token dtype accepted");
            f.tokens.dataType = INT32;
            f.head.dataDeviceIds = {1};
            Check(!f.Run(), "cross-device weight accepted");
            f.head.dataDeviceIds = {0};
            ++f.block;
            Check(!f.Run(), "short token buffer accepted");
            std::cout << "vocab=" << vocab << " grouped q/graph inputs/guards PASS\n";
        }
        Distribution(false);
        Distribution(true);
        std::cout << "DSpark device sampling regression PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
