// Integrated sharded projection, conditional q and graph RNG.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-dspark.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void Sync(const std::vector<int> &devices) {
    for (int d : devices) FastllmCudaSyncDevice(d);
    FastllmCudaSetDevice(devices.front());
}
template<class T> static std::vector<T> Read(const Data &d) {
    std::vector<T> values(d.Count(0));
    FastllmCudaCopyFromDeviceToHost(values.data(), d.cudaData, values.size() * sizeof(T));
    return values;
}
struct Fixture {
    int vocab, block; float temperature;
    std::vector<int> devices, support;
    std::vector<float> e, h, base;
    Data embed, head, input, linear, tokens, probs, out, logits, scratch;
    std::shared_ptr<void> projection, chain;
    Fixture(int v, int b, float t, int ranks, bool sparse, bool largeOffset = false)
        : vocab(v), block(b), temperature(t), e(v * 256), h(v * 256), base(v) {
        devices.resize(ranks); std::iota(devices.begin(), devices.end(), 0);
        FastllmMultiCudaSetDevice(devices); FastllmCudaSetDevice(0);
        support = {1, v / 3, 2 * v / 3, v - 1};
        for (int i = 0; i < v * 256; ++i) {
            e[i] = ((i * 13 + i / 256 * 3) % 31 - 15) / 128.0f;
            h[i] = ((i * 7 + i / 256 * 5) % 29 - 14) / 128.0f;
        }
        std::vector<float> x(b * 8), w(v * 8);
        for (int s = 0; s < b; ++s) x[s * 8] = 1;
        for (int i = 0; i < v; ++i) {
            base[i] = largeOffset ? 0x1p26f : sparse ? -128.0f : (i % 23 - 11) / 8.0f;
            if (sparse && std::find(support.begin(), support.end(), i) != support.end()) base[i] = (i % 7) / 8.0f;
            w[i * 8] = base[i];
        }
        auto init = [](Data &d, DataType type, std::vector<int> dims, const std::vector<float> &values) {
            d.CopyFrom(Data(type, dims, values)); d.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        };
        init(embed, BFLOAT16, {v, 256}, e); init(head, BFLOAT16, {v, 256}, h);
        init(input, BFLOAT16, {1, b, 8}, x); init(linear, FLOAT32, {v, 8}, w);
        tokens.dataType = INT32; tokens.UpdateUnitSize(); tokens.Resize({b + 1});
        tokens.ToDevice(DataDevice::CUDA, std::vector<int>{0}); tokens.Allocate();
        probs.dataType = FLOAT32; probs.UpdateUnitSize(); probs.Resize({b, v});
        probs.ToDevice(DataDevice::CUDA, std::vector<int>{0}); probs.Allocate();
        Anchor(0);
    }
    void Anchor(int id) { FastllmCudaSetDevice(0); FastllmCudaCopyFromHostToDevice(tokens.cudaData, &id, 4); }
    void Run(bool sharded = true) {
        MultiCudaDsparkMarkovRequest request{embed, head, tokens, out, chain};
        request.proposalProbs = &probs; request.temperature = temperature;
        Check(MultiCudaLinearOutputGatherPersistent(input, linear, logits, projection,
            sharded ? &request : nullptr), "projection failed");
        if (sharded) Check(request.completed, "sampling shard fallback");
        else Check(FastllmCudaDeepSeekV41MarkovSampleDevice(logits, embed, head, block,
            temperature, tokens, probs, scratch, out), "unsharded sample failed");
    }
    std::vector<double> Q(int previous, bool sparse = false) const {
        std::vector<double> q(vocab);
        double maximum = -INFINITY;
        auto score = [&](int v) {
            double value = base[v];
            for (int d = 0; d < 256; ++d) value += double(e[previous * 256 + d]) * h[v * 256 + d];
            q[v] = float(value); maximum = std::max(maximum, q[v]);
        };
        if (sparse) { std::fill(q.begin(), q.end(), -INFINITY); for (int v : support) score(v); }
        else for (int v = 0; v < vocab; ++v) score(v);
        double sum = 0;
        for (double &x : q) sum += x = std::exp((x - maximum) / std::max(temperature, 1e-6f));
        for (double &x : q) x /= sum;
        return q;
    }
    void Numeric(bool sparse = false) {
        auto ids = Read<int>(tokens); auto actual = Read<float>(probs); auto embeddings = Read<float>(out);
        for (int s = 0; s < block; ++s) {
            Check(ids[s + 1] >= 0 && ids[s + 1] < vocab, "invalid sampled token");
            auto q = Q(ids[s], sparse); double sum = 0, l1 = 0;
            for (int v = 0; v < vocab; ++v) {
                Check(std::isfinite(actual[s * vocab + v]) && actual[s * vocab + v] >= 0, "invalid q");
                sum += actual[s * vocab + v]; l1 += std::abs(actual[s * vocab + v] - q[v]);
            }
            if (std::abs(sum - 1) > 2e-5 || l1 > 2e-5)
                std::fprintf(stderr, "q step=%d sum=%.9g L1=%.9g\n", s, sum, l1);
            Check(std::abs(sum - 1) < 2e-5 && l1 < 2e-5, "conditional q differs from CPU oracle");
            for (int d = 0; d < 256; ++d)
                Check(embeddings[s * 256 + d] == e[ids[s] * 256 + d], "conditional embedding mismatch");
        }
    }
};
struct Graph {
    Fixture &f;
    std::vector<void*> graphs, execs, starts, ends, reserved;
    explicit Graph(Fixture &fixture, bool sharded = true) : f(fixture) {
        f.Run(sharded); Sync(f.devices);
        int n = f.devices.size(); graphs.resize(n); execs.resize(n); starts.resize(n); ends.resize(n);
        for (int d : f.devices) {
            FastllmCudaSetDevice(d); Check(FastllmCudaGraphPrepareCaptureDevice(), "prepare capture");
            starts[d] = FastllmCudaEventCreate(); ends[d] = FastllmCudaEventCreate();
        }
        Check(FastllmCudaGraphMemoryPoolBegin(), "capture pool");
        for (int d : f.devices) {
            FastllmCudaSetDevice(d); Check(FastllmCudaGraphBeginCapture(), "begin capture");
            FastllmCudaEventRecordCurrentThread(starts[d]);
        }
        Check(MultiCudaGraphWorkersWaitEvents(f.devices, starts), "join workers");
        FastllmCudaSetDevice(0); f.Run(sharded);
        Check(MultiCudaGraphWorkersRecordEvents(f.devices, ends), "finish workers");
        for (int d : f.devices) {
            FastllmCudaSetDevice(d); FastllmCudaCurrentThreadStreamWaitEvent(ends[d]);
            Check(FastllmCudaGraphEndCapture(&graphs[d]), "end capture");
        }
        Check(FastllmCudaGraphMemoryPoolEnd(reserved), "end pool");
        for (int d : f.devices) {
            FastllmCudaSetDevice(d); Check(FastllmCudaGraphInstantiate(graphs[d], &execs[d]), "instantiate");
        }
        FastllmCudaSetDevice(0);
    }
    void Run() {
        for (int d : f.devices) {
            FastllmCudaSetDevice(d); Check(FastllmCudaGraphLaunch(execs[d]), "launch");
        }
        FastllmCudaSetDevice(0);
    }
    ~Graph() {
        Sync(f.devices);
        for (int d : f.devices) {
            FastllmCudaSetDevice(d); FastllmCudaGraphExecDestroy(execs[d]); FastllmCudaGraphDestroy(graphs[d]);
            FastllmCudaEventDestroy(starts[d]); FastllmCudaEventDestroy(ends[d]);
        }
        FastllmCudaGraphMemoryPoolRelease(reserved); FastllmCudaSetDevice(0);
    }
};
static void Distribution() {
    Fixture f(65539, 3, .7f, 8, true); Graph graph(f);
    const int samples = 6000;
    std::vector<int> counts(64); std::vector<double> expected(64);
    auto first = f.Q(0, true);
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
        auto second = f.Q(f.support[a], true), third = f.Q(f.support[b], true);
        for (int c = 0; c < 4; ++c)
            expected[(a * 4 + b) * 4 + c] = first[f.support[a]] * second[f.support[b]] * third[f.support[c]];
    }
    for (int n = 0; n < samples; ++n) {
        graph.Run(); auto ids = Read<int>(f.tokens); int joint = 0;
        for (int s = 1; s <= 3; ++s) {
            auto it = std::find(f.support.begin(), f.support.end(), ids[s]);
            Check(it != f.support.end(), "token outside proposal support");
            joint = joint * 4 + int(it - f.support.begin());
        }
        ++counts[joint];
        if (n < 3) f.Numeric(true);
    }
    double z = 0;
    for (int i = 0; i < 64; ++i)
        z = std::max(z, std::abs(counts[i] - samples * expected[i]) /
            std::sqrt(samples * expected[i] * (1 - expected[i])));
    Check(z < 7, "sharded graph changed conditional joint law or reused RNG");
    std::printf("TP8 conditional joint law, fresh graph RNG: max_z=%.6f PASS\n", z);
}
static void Benchmark() {
    Fixture f(129280, 5, 1, 8, false);
    Graph baseline(f, false), sharded(f);
    auto bench = [&](Graph &g) {
        cudaEvent_t start, end;
        Check(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess, "timer");
        for (int i = 0; i < 10; ++i) g.Run();
        Check(cudaEventRecord(start, cudaStreamPerThread) == cudaSuccess, "start timer");
        for (int i = 0; i < 100; ++i) g.Run();
        Check(cudaEventRecord(end, cudaStreamPerThread) == cudaSuccess &&
            cudaEventSynchronize(end) == cudaSuccess, "end timer");
        float ms = 0; Check(cudaEventElapsedTime(&ms, start, end) == cudaSuccess, "elapsed");
        cudaEventDestroy(start); cudaEventDestroy(end); return ms * 10;
    };
    double a = bench(baseline), b = bench(sharded), c = bench(sharded), d = bench(baseline);
    f.Numeric();
    std::printf("head+Markov+q graph ABBA: unsharded_us=%.6f sharded_us=%.6f\n", (a + d) / 2, (b + c) / 2);
}
int main(int argc, char **argv) {
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 8) return 77;
    try {
        Check(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--bench") == 0),
            "usage: deepseekV41DsparkShardSamplingRegression [--bench]");
        SetThreads(16); SetCudaSlabMB(256); SetCudaGraph(true);
        SetDeviceMap({{"multicuda:0,1,2,3,4,5,6,7", 1}});
        MultiCudaSetPersistentAsyncDispatch(true);
        if (argc == 2) {
            Benchmark();
            std::puts("DSpark sharded sampling benchmark PASS"); return 0;
        }
        for (int ranks : {2, 3, 8}) {
            Fixture f(65539, 5, .7f, ranks, false); f.Run(); Sync(f.devices); f.Numeric();
            Graph graph(f);
            for (int anchor : {3, 65538}) { f.Anchor(anchor); graph.Run(); f.Numeric(); }
            std::printf("TP%d odd vocabulary / q oracle / changing graph inputs PASS\n", ranks);
        }
        for (int block : {2, 7, 8}) {
            Fixture f(129280, block, 1.3f, 8, true); f.Run(); f.Numeric(true);
        }
        {
            Fixture f(65536, 2, 1e-8f, 8, false, true); Graph graph(f);
            std::vector<int> seen;
            for (int n = 0; n < 128; ++n) { graph.Run(); seen.push_back(Read<int>(f.tokens)[1]); }
            f.Numeric(); std::sort(seen.begin(), seen.end());
            Check(std::unique(seen.begin(), seen.end()) - seen.begin() > 100, "large offset removed Gumbel noise");
            std::puts("large offset / temperature clamp PASS");
        }
        Distribution();
        std::puts("DSpark sharded sampling regression PASS"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
