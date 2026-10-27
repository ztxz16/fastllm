#pragma once

// Reusable fixtures; references use Linear and separate activation kernels.
#include "cuda_gguf_data_test.cuh"
#include "fastllm-cuda-gguf-linear-add.h"
#include "fastllm-cuda-gguf-projections.h"
#include "fastllm-cuda-rmsnorm-small-linear.h"
#include <functional>
#include <string>

static int cases = 0, graphs = 0;
static std::string context;
static void Same(const Data &data, const std::vector<half> &want, const char *label) {
    const auto got = Download(data, want.size());
    for (size_t i = 0; i < want.size(); ++i)
        if (std::memcmp(&want[i], &got[i], sizeof(half))) {
            std::cerr << context << " " << label << " at " << i << " got=" << float(got[i])
                      << " want=" << float(want[i]) << std::endl;
            throw std::runtime_error("bitwise comparison failed");
        }
}
static std::vector<half> Input(int t, int k, int seed = 0) {
    std::vector<half> x(size_t(t) * k);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = __float2half_rn(i % k < 32 || (t > 1 && i / k == size_t(t - 1))
                                   ? 0
                                   : std::sin(float(i + seed) * .117f) * (.5f + float(i % 7) / 5));
    return x;
}
struct Projection {
    Data w, y, storage, ref;
    int t, k, n;
    Projection(ggml_type type, int t, int k, int n)
        : w(DATA_GGUF_FORMAT, int(type), {n, k}), y(FLOAT16), storage(FLOAT16, {t * n + 8}),
          ref(FLOAT16, {1, t, n}), t(t), k(k), n(n) {
        Allocate(w);
        Allocate(storage);
        Allocate(ref);
        w.strides = {1};
        w.forceGGUFFp32Dequant = true;
        Upload(w, Weights(type, n, k));
        y.FakeFrom(storage, 0);
        y.Resize({1, t, n});
        y.dataDeviceIds = {0};
        Upload(storage, std::vector<half>(t * n + 8, __float2half_rn(42)));
    }
    void Reference(const Data &x) {
        Data bias(FLOAT32);
        Check(FastllmCudaHalfMatMulGGUF(x, w, bias, ref, t, k, n), "reference linear rejected");
    }
    void Compare() {
        Same(y, Download(ref, t * n), "projection");
        auto values = Download(storage, t * n + 8);
        for (int i = t * n; i < t * n + 8; ++i)
            Check(float(values[i]) == 42, "tail overwritten");
    }
};
static void Graph(
    const std::function<void()> &run, const std::function<void()> &compare,
    const std::function<void(int)> &prepare = [](int) {}) {
    prepare(0);
    run();
    Cuda(cudaDeviceSynchronize());
    compare();
    cudaGraph_t g;
    cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    run();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &g));
    Cuda(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    for (int i = 1; i <= 2; ++i) {
        prepare(i);
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaDeviceSynchronize());
        compare();
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(g));
    ++graphs;
    ++cases;
}
static void Gate(ggml_type a, ggml_type b, int t, int k, int n) {
    Data x(FLOAT16, {1, t, k});
    Allocate(x);
    Projection p(a, t, k, n), q(b, t, k, n);
    Graph([&] { Check(FastllmCudaGGUFMixedGateUp(x, p.w, q.w, p.y), "gate rejected"); }, [&] { p.Compare(); },
          [&](int seed) {
              Upload(x, Input(t, k, seed * 71));
              p.Reference(x);
              q.Reference(x);
              FastllmCudaSilu(p.ref, p.ref);
              FastllmCudaMulTo(p.ref, q.ref, 1.0f);
          });
}
