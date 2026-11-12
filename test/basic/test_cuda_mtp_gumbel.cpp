#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <thread>
#include <memory>

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
        float temperature = 1, int topK = 32, float topP = 1, bool indirect = false, int planned = 0, bool probability = false) {
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
    if (probability) {
        const auto normalizers = lse.Read();
        for (size_t i=0;i<cache.size();++i) cache[i]=std::exp(cache[i]-normalizers[i/vocab]);
        Check(cudaMemcpy(saved.p,cache.data(),cache.size()*sizeof(float),cudaMemcpyHostToDevice));
    }
    if (planned) {
        std::vector<FastllmMtpProposalView> views(batch);
        for (int b = 0; b < batch; ++b)
            views[b] = {saved.p + size_t(b) * drafts * vocab, probability ? nullptr : lse.p + b * drafts, ids.p + b * drafts};
        std::unique_ptr<void, decltype(&FastllmCudaMtpDestroyVerifier)> plan(
            FastllmCudaMtpCreateVerifier(), FastllmCudaMtpDestroyVerifier);
        Require((probability ? FastllmCudaMtpVerifierEnqueueProbabilities : FastllmCudaMtpVerifierEnqueue)(plan.get(), target.p, views.data(), pt.data(), pk.data(), pp.data(),
            batch, drafts, vocab, planned == 2), "planned chain enqueue");
        Require(FastllmCudaMtpVerifierResolve(plan.get(), output.data(), accepted.data()), "planned chain resolve");
    } else if (indirect) {
        std::vector<FastllmMtpProposalView> views(batch);
        for (int b = 0; b < batch; ++b)
            views[b] = {saved.p + (size_t)b * drafts * vocab, lse.p + b * drafts, ids.p + b * drafts};
        Require(FastllmCudaMtpRejectionSamplingLogitsBatch(target.p, views.data(),
            pt.data(), pk.data(), pp.data(), output.data(), accepted.data(), batch, drafts, vocab), "indirect chain verify failed");
    } else {
        Require(FastllmCudaMtpRejectionSamplingLogits(target.p, saved.p, lse.p, ids.p,
            pt.data(), pk.data(), pp.data(), output.data(), accepted.data(), batch, drafts, vocab), "chain verify failed");
    }
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

void PlanLifecycle(bool probability = false) {
    const int vocab = 257, batch = 4, drafts = 17, rows = batch * (drafts + 1);
    std::vector<float> target(size_t(rows) * vocab, -INFINITY), saved(size_t(batch)*drafts*vocab, -INFINITY);
    for (int row = 0; row < rows; ++row) { target[size_t(row)*vocab] = std::log(.7f); target[size_t(row)*vocab+1] = std::log(.3f); }
    for (int row = 0; row < batch * drafts; ++row) saved[size_t(row)*vocab+2] = 0;
    if (probability) for (float &x:saved) x=std::exp(x);
    Buffer<float> p(target), q(saved), lse(std::vector<float>(batch*drafts,0));
    Buffer<int> ids(std::vector<int>(batch*drafts,2));
    std::vector<FastllmMtpProposalView> views(batch);
    for (int b=0;b<batch;++b) views[b]={q.p+size_t(b)*drafts*vocab,probability ? nullptr : lse.p+b*drafts,ids.p+b*drafts};
    std::vector<float> ts(rows,1), ps(rows,1);
    std::vector<int> ks(rows,vocab), output(rows), accepted(batch);
    void *plan=FastllmCudaMtpCreateVerifier();
    int randomOnes=0;
    for (int iteration=0;iteration<64;++iteration) {
        // Immutable pinned inputs and device scratch must survive worker handoff.
        auto run=[&] {
            Check(cudaSetDevice(0));
            std::fill(ks.begin(),ks.end(),iteration%4==0?1:vocab);
            std::fill(ps.begin(),ps.end(),iteration%4==1?.5f:1.f);
            ts[0]=iteration%2?.8f:1.f;
            if (iteration%3==0) std::reverse(views.begin(),views.end());
            Require((probability ? FastllmCudaMtpVerifierEnqueueProbabilities : FastllmCudaMtpVerifierEnqueue)(plan,p.p,views.data(),ts.data(),ks.data(),ps.data(),
                batch,drafts,vocab,iteration%8!=0),"plan enqueue/reconfigure");
            Require(!(probability ? FastllmCudaMtpVerifierEnqueueProbabilities : FastllmCudaMtpVerifierEnqueue)(plan,p.p,views.data(),ts.data(),ks.data(),ps.data(),
                batch,drafts,vocab,true),"overlapping plan accepted");
            Require(FastllmCudaMtpVerifierResolve(plan,output.data(),accepted.data()),"plan resolve/handoff");
            for (int b=0;b<batch;++b) {
                Require(accepted[b]==0,"disjoint plan should reject");
                int token=output[b*(drafts+1)];
                Require(token>=0&&token<=1,"plan target support");
                if (iteration%4<2) Require(token==0,"changed top-k/top-p not respected");
                else randomOnes+=token==1;
            }
        };
        std::exception_ptr error;
        std::thread worker([&] { try {run();}catch(...) {error=std::current_exception();} });
        worker.join(); if(error)std::rethrow_exception(error);
    }
    Require(randomOnes>15 && randomOnes<70,"plan replay RNG distribution");
    // Keep the graph's shape and filters fixed while replacing its input
    // address. Each alternating buffer has disjoint target support.
    std::vector<float> alternate(size_t(rows)*vocab,-INFINITY);
    for(int row=0;row<rows;++row) alternate[size_t(row)*vocab+3]=0;
    Buffer<float> p2(alternate);
    std::fill(ts.begin(),ts.end(),1.f);std::fill(ps.begin(),ps.end(),1.f);std::fill(ks.begin(),ks.end(),vocab);
    for(int i=0;i<12;++i) {
        Require((probability ? FastllmCudaMtpVerifierEnqueueProbabilities : FastllmCudaMtpVerifierEnqueue)(plan,i%2?p.p:p2.p,views.data(),ts.data(),ks.data(),ps.data(),
            batch,drafts,vocab,true),"plan input retarget");
        Require(FastllmCudaMtpVerifierResolve(plan,output.data(),accepted.data()),"plan retarget resolve");
        for(int b=0;b<batch;++b) Require(accepted[b]==0 && (i%2 ? output[b*(drafts+1)]<=1 : output[b*(drafts+1)]==3),
            "retargeted graph read old input");
    }

    ts[0]=0;
    Require(!(probability ? FastllmCudaMtpVerifierEnqueueProbabilities : FastllmCudaMtpVerifierEnqueue)(plan,p.p,views.data(),ts.data(),ks.data(),ps.data(),batch,drafts,vocab,true),
        "invalid parameters accepted");
    FastllmCudaMtpDestroyVerifier(plan);
    std::puts("owned verifier lifecycle: parameter changes, graph/eager, 17 drafts, descriptor reorder, fresh workers, RNG: PASS");
}

void ProbabilityPrefixes() {
    const int vocab=257, guard=7;
    std::unique_ptr<void,decltype(&FastllmCudaMtpDestroyVerifier)> plan(
        FastllmCudaMtpCreateVerifier(),FastllmCudaMtpDestroyVerifier);
    for(int batch:{1,2,3,5,9}) for(int drafts:{1,5,9,17}) {
        const int rows=batch*(drafts+1), pitch=drafts+guard;
        std::vector<float> q(size_t(batch)*pitch*vocab,123.f);
        std::vector<int> ids(batch*pitch,-7);
        for(int b=0;b<batch;++b) for(int d=0;d<drafts;++d) {
            std::fill_n(q.data()+size_t(b*pitch+d)*vocab,vocab,0.f);
            q[size_t(b*pitch+d)*vocab+b%3]=1;ids[b*pitch+d]=b%3;
        }
        Buffer<float> probabilities(q), target(size_t(rows)*vocab);
        Buffer<int> tokens(ids);
        std::vector<FastllmMtpProposalView> views(batch);
        std::vector<float> logits(size_t(rows)*vocab), temperatures(rows,1), ps(rows,1);
        std::vector<int> ks(rows,vocab), output(rows), accepted(batch);
        for(int step=0;step<=drafts;++step) {
            std::fill(logits.begin(),logits.end(),-INFINITY);
            for(int b=0;b<batch;++b) {
                const int source=(b+step)%batch, prefix=(b+step)%(drafts+1), token=source%3;
                views[b]={probabilities.p+size_t(source)*pitch*vocab,nullptr,tokens.p+source*pitch};
                for(int d=0;d<=drafts;++d)
                    logits[size_t(b*(drafts+1)+d)*vocab+(d<prefix?token:token+3)]=0;
            }
            Check(cudaMemcpy(target.p,logits.data(),logits.size()*sizeof(float),cudaMemcpyHostToDevice));
            Require(FastllmCudaMtpVerifierEnqueueProbabilities(plan.get(),target.p,views.data(),
                temperatures.data(),ks.data(),ps.data(),batch,drafts,vocab,step%2),"probability prefix enqueue");
            Require(FastllmCudaMtpVerifierResolve(plan.get(),output.data(),accepted.data()),"probability prefix resolve");
            for(int b=0;b<batch;++b) {
                const int prefix=(b+step)%(drafts+1), token=((b+step)%batch)%3;
                Require(accepted[b]==prefix,"wrong probability prefix length");
                for(int d=0;d<=drafts;++d) Require(output[b*(drafts+1)+d]==
                    (d<prefix?token:d==prefix?token+3:-1),"probability suffix/support mismatch");
            }
        }
        Require(q==probabilities.Read()&&ids==tokens.Read(),"probability verifier mutated borrowed input/guards");
    }
    std::puts("probability verifier: all prefixes, irregular batches, 17 drafts, descriptor remap, bounded shape reuse: PASS");
}

void IndirectBatchCase(int batch, int drafts, bool owned = false) {
    const int vocab = 257, guard = 11, pitch = drafts + guard;
    Buffer<float> cache(std::vector<float>((size_t)batch * pitch * vocab, 123)),
                  lse(std::vector<float>(batch * pitch, 123)),
                  floatTokens(std::vector<float>(batch * pitch, 123));
    Buffer<int> tokens(std::vector<int>(batch * pitch, -7));
    std::vector<float> ts(batch), logits((size_t)batch * vocab, -INFINITY);
    std::vector<int> hostTokens(batch);
    auto id = [&](int b, int d) { return (b * 17 + d * 13 + 1) % vocab; };
    // Reverse the request-to-cache mapping and pad slots: contiguous indexing
    // or cross-lane aliasing must fail, even when every row has one-hot logits.
    for (int d = 0; d < drafts; ++d) {
        std::fill(logits.begin(), logits.end(), -INFINITY);
        std::vector<FastllmMtpDraftOutput> outputs(batch);
        for (int b = 0; b < batch; ++b) {
            ts[b] = d % 2 ? 1.0f : (b % 2 ? .7f : 1.3f);
            logits[(size_t)b * vocab + id(b,d)] = .5f;
            size_t slot = (batch - 1 - b) * pitch + d;
            outputs[b] = {cache.p + slot * vocab, lse.p + slot, tokens.p + slot,
                          b % 2 ? nullptr : floatTokens.p + slot};
        }
        Buffer<float> input(logits);
        Require(FastllmCudaMtpSampleDraftLogitsBatch(input.p, outputs.data(), ts.data(),
            d % 2 ? nullptr : hostTokens.data(), batch, vocab), "indirect draft failed");
        auto observed = tokens.Read();
        auto floats = floatTokens.Read(), normalizers = lse.Read(), saved = cache.Read();
        for (int b = 0; b < batch; ++b) {
            size_t slot = (batch - 1 - b) * pitch + d;
            Require(observed[slot] == id(b,d), "indirect token mapping failed");
            if (d % 2 == 0) Require(hostTokens[b] == id(b,d), "batched host tokens mismatch");
            if (b % 2 == 0) Require(floats[slot] == id(b,d), "indirect float token mapping failed");
            Require(std::fabs(normalizers[slot] - .5f/ts[b]) < 1e-6, "indirect normalizer mismatch");
            Require(std::fabs(saved[slot*vocab+id(b,d)] - .5f/ts[b]) < 1e-6, "indirect logits mismatch");
        }
    }
    auto observed = tokens.Read();
    for (int b = 0; b < batch; ++b)
        for (int d = drafts; d < pitch; ++d) Require(observed[b*pitch+d] == -7, "proposal guard overwritten");
    std::vector<float> target((size_t)batch*(drafts+1)*vocab,-INFINITY), pt(batch*(drafts+1),1), pp(pt.size(),.95f);
    std::vector<int> pk(pt.size(),1), output(pt.size()), accepted(batch), expected(batch);
    std::vector<FastllmMtpProposalView> views(batch);
    for (int b = 0; b < batch; ++b) {
        int off = (batch - 1 - b) * pitch;
        views[b] = {cache.p + (size_t)off*vocab, lse.p+off, tokens.p+off};
        expected[b] = b % (drafts+1);
        for (int d = 0; d <= drafts; ++d) {
            int token = d < expected[b] ? id(b,d) : (id(b,d)+1)%vocab;
            target[((size_t)b*(drafts+1)+d)*vocab+token] = 0;
        }
    }
    Buffer<float> p(target);
    if (owned) {
        std::unique_ptr<void, decltype(&FastllmCudaMtpDestroyVerifier)> plan(
            FastllmCudaMtpCreateVerifier(), FastllmCudaMtpDestroyVerifier);
        for (int replay=0;replay<3;++replay) {
            Require(FastllmCudaMtpVerifierEnqueue(plan.get(),p.p,views.data(),pt.data(),pk.data(),pp.data(),
                batch,drafts,vocab,true), "planned mixed-prefix enqueue");
            Require(FastllmCudaMtpVerifierResolve(plan.get(),output.data(),accepted.data()), "planned mixed-prefix resolve");
        }
    } else
    Require(FastllmCudaMtpRejectionSamplingLogitsBatch(p.p,views.data(),pt.data(),pk.data(),pp.data(),
        output.data(),accepted.data(),batch,drafts,vocab), "indirect verification failed");
    for (int b = 0; b < batch; ++b) {
        Require(accepted[b] == expected[b], "mixed accept lengths mismatch");
        for (int d = 0; d <= drafts; ++d) {
            int expectedToken = d < expected[b] ? id(b,d) : d == expected[b] ? (id(b,d)+1)%vocab : -1;
            Require(output[b*(drafts+1)+d] == expectedToken, "indirect verified token mismatch");
        }
    }
    std::printf("indirect B=%d drafts=%d, reordered/padded caches, mixed acceptance: PASS\n",batch,drafts);
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
        ChainCase("batch indirect p=q", {.6f,.3f,.1f}, {.6f,.3f,.1f}, 1, 32, 1, true);
        ChainCase("batch indirect residual", {.1f,.2f,.7f}, {.6f,.3f,.1f}, 1, 32, 1, true);
        ChainCase("batch indirect constraints", {.6f,.3f,.1f}, {.2f,.3f,.5f}, 1.5f, 2, .95f, true);
        for (int mode : {1,2}) {
            ChainCase("owned verifier p=q", {.6f,.3f,.1f}, {.6f,.3f,.1f}, 1, 32, 1, false, mode);
            ChainCase("owned verifier residual", {.1f,.2f,.7f}, {.6f,.3f,.1f}, 1, 32, 1, false, mode);
            ChainCase("owned verifier constraints", {.6f,.3f,.1f}, {.2f,.3f,.5f}, 1.5f, 2, .95f, false, mode);
        }
        ChainCase("probability plan equal", {.2f,.3f,.5f}, {.2f,.3f,.5f},1,32,1,true,2,true);
        ChainCase("probability plan disjoint", {0,1,0}, {1,0,0},1,32,1,true,2,true);
        ChainCase("probability plan filtered", {.15f,.35f,.5f}, {.4f,.3f,.3f},.8f,2,.85f,true,2,true);
        PlanLifecycle(true);
        ProbabilityPrefixes();
        PlanLifecycle();
        for (int batch : {1, 2, 4, 32}) for (int drafts : {1, 3, 17})
            IndirectBatchCase(batch, drafts, true);
        for (int device = 0; device < devices; ++device) {
            Check(cudaSetDevice(device));
            for (int batch : {1,2,4,8,10,16,24,32})
                for (int drafts : {1,3,7}) IndirectBatchCase(batch,drafts);
        }
        Check(cudaSetDevice(0));
        LargeVocabAndGraphCase(); std::puts("MTP Gumbel/rejection: PASS"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
