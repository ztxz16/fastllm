#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
struct Buffer {
    float *p = nullptr;
    explicit Buffer(size_t n) { Check(cudaMalloc(&p, n * sizeof(float)) == cudaSuccess, "allocate"); }
    explicit Buffer(const std::vector<float> &v) : Buffer(v.size()) {
        Check(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess, "upload");
    }
    ~Buffer() { cudaFree(p); }
};
static void Softmax() {
    for (int vocab : {16383, 16384, 129281}) for (int rows : {1, 6, 63})
    for (bool uniform : {false, true}) {
        const float offset = vocab >= 16384 && !uniform ? 65536.0f : 0.0f;
        std::vector<float> x((size_t)rows * vocab), ts(rows), ps(rows, 1.0f);
        std::vector<int> ks(rows, vocab), tokens(rows);
        for (int r = 0; r < rows; ++r) {
            ts[r] = uniform ? .7f : r % 3 == 0 ? .7f : r % 3 == 1 ? 1.3f : 1.0f;
            for (int v = 0; v < vocab; ++v)
                x[(size_t)r * vocab + v] = (r == 0 && v < 4096) || v % 31 == 7 ?
                    -INFINITY : offset + ((v * 13 + r * 7) % 29 - 14) / 8.0f;
        }
        Buffer logits(x), probs(x.size());
        Check(FastllmCudaMtpSampleDraft(logits.p, probs.p, ts.data(), ks.data(), ps.data(),
            tokens.data(), rows, vocab), "small-batch sampling");
        std::vector<float> q(x.size());
        Check(cudaMemcpy(q.data(), probs.p, q.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "read q");
        double worst = 0;
        for (int r = 0; r < rows; ++r) {
            auto begin = x.begin() + (size_t)r * vocab;
            double maximum = *std::max_element(begin, begin + vocab), sum = 0, actualSum = 0, l1 = 0;
            std::vector<double> expected(vocab);
            for (int v = 0; v < vocab; ++v) sum += expected[v] = std::exp((begin[v] - maximum) / ts[r]);
            for (int v = 0; v < vocab; ++v) {
                float actual = q[(size_t)r * vocab + v];
                Check(std::isfinite(actual) && actual >= 0, "invalid probability");
                l1 += std::abs(actual - expected[v] / sum); actualSum += actual;
            }
            Check(std::abs(actualSum - 1) < 1e-5 && l1 < 2e-5, "temperature softmax disagrees with CPU oracle");
            Check(tokens[r] >= 0 && tokens[r] < vocab && expected[tokens[r]] > 0, "sampled masked token");
            worst = std::max(worst, l1);
        }
        std::printf("softmax rows=%d vocab=%d uniform=%d offset=%.0f L1=%.9g PASS\n", rows, vocab, uniform, offset, worst);
    }
}
static void Benchmark() {
    const int vocab = 129280, drafts = 5, rows = drafts + 1;
    std::vector<float> x(rows * vocab, -INFINITY), q(drafts * vocab), ts(rows, 1), ps(rows, 1);
    std::vector<int> ks(rows, vocab), ids(drafts, 0), output(rows); int accepted = -1;
    for (int r = 0; r < rows; ++r) {
        x[r * vocab] = std::log(.6f); x[r * vocab + 1] = std::log(.3f); x[r * vocab + 2] = std::log(.1f);
        if (r < drafts) { q[r * vocab] = .6f; q[r * vocab + 1] = .3f; q[r * vocab + 2] = .1f; }
    }
    Buffer logits(x), proposal(q);
    auto run = [&] {
        Check(FastllmCudaMtpRejectionSampling(logits.p, proposal.p, ts.data(), ks.data(), ps.data(),
            ids.data(), output.data(), &accepted, 1, drafts, vocab), "benchmark rejection");
        Check(accepted >= 0 && accepted <= drafts, "benchmark accepted prefix");
    };
    for (int i = 0; i < 100; ++i) run();
    cudaProfilerStart();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) run();
    auto end = std::chrono::steady_clock::now();
    cudaProfilerStop();
    std::printf("DSpark5 target preparation + rejection + readback: %.6f us/call\n",
        std::chrono::duration<double, std::micro>(end - start).count() / 1000);
}
int main(int argc, char **argv) {
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        Check(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--bench") == 0),
            "usage: mtpSmallBatchSamplingRegression [--bench]");
        Check(cudaSetDevice(0) == cudaSuccess, "device");
        if (argc == 2) {
            Benchmark();
            std::puts("small-batch MTP sampling benchmark PASS"); return 0;
        }
        Softmax();
        std::puts("small-batch MTP sampling regression PASS"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
