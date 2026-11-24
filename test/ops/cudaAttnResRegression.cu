// Standalone FP64 comparison for normalized/unnormalized AttnRes, with and
// without a fused output norm. Compile with nvcc -std=c++17 -Iinclude.
#include "devices/cuda/fastllm-cuda-attn-res.cuh"
#include <array>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm::cuda_attn_res;
static void check(cudaError_t rc) {
    if (rc != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(rc));
}
template <class T> struct Device {
    T *p;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
};

// The controller may request a rounded FP32 row for the router or a
// physically narrow row for Linear. Compare both with an explicit cast.
template<class T, class In>
static void checkOutputTypes(SourceList<In> sources, int ns, const Query *queries, int nq, int rows, int width,
                             float eps, bool normalized, const std::vector<float> &reference) {
    const size_t n = reference.size();
    Device<T> narrow(n);
    Device<float> rounded(n);
    const bool needsScratch = NeedsStaging<T>(ns, queries, nq, width);
    Device<float> staging(needsScratch ? n : 1);
    Launch<T>(sources, ns, queries, nq, rows, width, eps, normalized, narrow.p, cudaStreamPerThread,
              needsScratch ? staging.p : nullptr);
    Launch<T>(sources, ns, queries, nq, rows, width, eps, normalized, rounded.p, cudaStreamPerThread);
    std::vector<T> compact(n);
    std::vector<float> wide(n);
    check(cudaMemcpy(compact.data(), narrow.p, n * sizeof(T), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(wide.data(), rounded.p, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        float expected = float(T(reference[i]));
        if (float(compact[i]) != expected || wide[i] != expected)
            throw std::runtime_error("AttnRes output rounding mismatch");
    }
}

template <class In = float>
static double compare(int rows, int width, int ns, int nq, bool normalized, bool unaligned) {
    constexpr float eps = 1.e-6f;
    const size_t stride = size_t(rows) * width, skip = unaligned;
    std::vector<float> x(ns * stride + skip), q(size_t(nq) * width + skip), norm(nq * width);
    std::mt19937 gen(901 + width + ns);
    std::uniform_real_distribution<float> sample(-1.f, 1.f);
    for (auto &v : x)
        v = sample(gen);
    for (auto &v : q)
        v = sample(gen);
    for (auto &v : norm)
        v = sample(gen);
    // Include a zero source, and a query with a large score spread.
    if (ns > 1)
        std::fill_n(x.begin() + skip, stride, 0.f);
    for (int j = 0; j < width; ++j)
        q[skip + j] *= 75.f;

    std::vector<In> stored(x.begin(), x.end());
    std::copy(stored.begin(), stored.end(), x.begin()); // Compare to FP64 on the values actually stored.
    Device<In> dx(stored.size());
    Device<float> dq(q.size()), dw(norm.size()), dy(nq * stride);
    check(cudaMemcpy(dx.p, stored.data(), stored.size() * sizeof(In), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dq.p, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw.p, norm.data(), norm.size() * 4, cudaMemcpyHostToDevice));
    SourceList<In> sources{};
    std::vector<const In *> pointers(ns);
    Device<const In *> table(ns);
    for (int s = 0; s < ns; ++s) {
        pointers[s] = dx.p + skip + s * stride;
        if (s < Sources::Inline)
            sources.items[s] = pointers[s];
    }
    if (ns > Sources::Inline) {
        check(cudaMemcpy(table.p, pointers.data(), ns * sizeof(const In *), cudaMemcpyHostToDevice));
        sources.table = table.p;
    }
    std::vector<Query> queries(nq);
    for (int i = 0; i < nq; ++i)
        queries[i] = {dq.p + skip + i * width, i % 2 ? nullptr : dw.p + i * width};

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Launch(sources, ns, queries.data(), nq, rows, width, eps, normalized, dy.p, cudaStreamPerThread);
    check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    check(cudaGraphLaunch(exec, cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
    check(cudaGraphExecDestroy(exec));
    check(cudaGraphDestroy(graph));
    std::vector<float> actual(nq * stride);
    check(cudaMemcpy(actual.data(), dy.p, actual.size() * 4, cudaMemcpyDeviceToHost));
    checkOutputTypes<half>(sources, ns, queries.data(), nq, rows, width, eps, normalized, actual);
    checkOutputTypes<__nv_bfloat16>(sources, ns, queries.data(), nq, rows, width, eps, normalized, actual);
    double worst = 0.;
    for (int i = 0; i < nq; ++i)
        for (int r = 0; r < rows; ++r) {
            std::vector<double> logits(ns), inv(ns), mix(width);
            for (int s = 0; s < ns; ++s) {
                const float *v = x.data() + skip + s * stride + size_t(r) * width;
                double dot = 0., sq = 0.;
                for (int j = 0; j < width; ++j) {
                    dot += double(v[j]) * q[skip + i * width + j];
                    sq += double(v[j]) * v[j];
                }
                inv[s] = 1. / std::sqrt(sq / width + eps);
                logits[s] = dot * inv[s] / std::sqrt(double(width));
            }
            double mx = *std::max_element(logits.begin(), logits.end()), den = 0.;
            for (double &v : logits) {
                v = std::exp(v - mx);
                den += v;
            }
            for (int s = 0; s < ns; ++s) {
                double a = logits[s] / den * (normalized ? inv[s] : 1.);
                for (int j = 0; j < width; ++j)
                    mix[j] += a * x[skip + s * stride + size_t(r) * width + j];
            }
            double sq = 0., norm2 = 0., err = 0.;
            for (double v : mix)
                sq += v * v;
            for (int j = 0; j < width; ++j) {
                double ref = mix[j];
                if (queries[i].norm)
                    ref *= norm[i * width + j] / std::sqrt(sq / width + eps);
                double diff = actual[i * stride + size_t(r) * width + j] - ref;
                norm2 += ref * ref;
                err += diff * diff;
            }
            double l2 = std::sqrt(err / std::max(norm2, 1.e-16));
            if (!std::isfinite(l2) || l2 > 5.e-5) {
                std::fprintf(stderr, "rows=%d width=%d sources=%d query=%d row=%d normalized=%d relative_l2=%g\n",
                             rows, width, ns, i, r, normalized, l2);
                throw std::runtime_error("AttnRes reference mismatch");
            }
            worst = std::max(worst, l2);
        }
    return worst;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        double worst = 0.;
        int cases = 0;
        for (bool norm : {false, true})
            for (auto shape : std::vector<std::array<int, 5>>{{1, 2304, 2, 4, 0},
                                                              {1, 2304, 2, Queries::Inline + 1, 0},
                                                              {1, 2304, 5, 1, 0},
                                                              {4, 2304, 5, 3, 0},
                                                              {65535, 4, 2, 2, 0},
                                                              {65536, 4, 2, 2, 0},
                                                              {9, 2303, 7, 2, 0},
                                                              {3, 256, 19, 6, 1},
                                                              {16, 1, 1, 1, 0},
                                                              {2, 5003, 35, 5, 0},
                                                              {1, 256, 4, 1, 1}}) {
                worst = std::max(worst, compare(shape[0], shape[1], shape[2], shape[3], norm, shape[4]));
                ++cases;
            }
        for (bool norm : {false, true}) {
            // Exercise packed loads, scalar tails, and the external pointer table on BF16 residuals.
            worst = std::max(worst, compare<__nv_bfloat16>(4, 2304, 5, 3, norm, false));
            worst = std::max(worst, compare<__nv_bfloat16>(2, 259, 3, 2, norm, true));
            worst = std::max(worst, compare<__nv_bfloat16>(3, 256, 19, 6, norm, false));
            cases += 3;
        }
        std::printf("%d shapes/modes, CUDA graph replay, worst relative L2 %.9g\n", cases, worst);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
