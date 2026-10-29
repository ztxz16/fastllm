#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-linear-output.h"
#include "fastllm.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <cstring>
#include <random>
#include <vector>
#include <functional>
#include <algorithm>
using namespace fastllm;
#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        if (!(x)) {                                                                                          \
            fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x);                                      \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
#define CUDA_OK(x) CHECK((x) == cudaSuccess)
std::mt19937 rng(71);
void allocate(Data &x, DataType type, std::vector<int> dims, const void *source = nullptr) {
    x.dataType = type;
    x.UpdateUnitSize();
    x.Resize(dims);
    x.dataDevice = CUDA;
    x.dataDeviceIds = {0};
    x.Allocate(false);
    if (source)
        CUDA_OK(cudaMemcpy(x.cudaData, source, x.GetBytes(), cudaMemcpyHostToDevice));
}
std::vector<__nv_bfloat16> inputValues(size_t count) {
    std::vector<__nv_bfloat16> x(count);
    for (auto &v : x)
        v = __float2bfloat16_rn(float(int(rng() % 2049) - 1024) / 1024.f);
    return x;
}
std::vector<uint8_t> readFile(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    CHECK(f.good());
    size_t n = f.tellg();
    std::vector<uint8_t> b(n);
    f.seekg(0);
    f.read((char *)b.data(), n);
    CHECK(f.good());
    return b;
}
void compare(const Data &a, const Data &b, float tolerance, const char *label) {
    CHECK(a.dims == b.dims);
    std::vector<float> x(a.Count(0)), y(b.Count(0));
    auto download = [](const Data &d, std::vector<float> &v) {
        if (d.dataType == FLOAT32) {
            CUDA_OK(cudaMemcpy(v.data(), d.cudaData, v.size() * 4, cudaMemcpyDeviceToHost));
        } else {
            std::vector<__nv_bfloat16> q(v.size());
            CUDA_OK(cudaMemcpy(q.data(), d.cudaData, q.size() * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < v.size(); i++)
                v[i] = float(q[i]);
        }
    };
    download(a, x);
    download(b, y);
    double err = 0, den = 0;
    float maxAbs = 0;
    size_t mismatch = 0;
    for (size_t i = 0; i < x.size(); i++) {
        CHECK(std::isfinite(x[i]));
        CHECK(std::isfinite(y[i]));
        double d = x[i] - y[i];
        err += d * d;
        den += double(y[i]) * y[i];
        maxAbs = std::max(maxAbs, float(std::abs(d)));
        mismatch += x[i] != y[i];
    }
    double rel = std::sqrt(err / std::max(den, 1e-30));
    printf("%s rel_l2=%.9g max_abs=%.9g exact=%.6f\n", label, rel, maxAbs, 1. - double(mismatch) / x.size());
    CHECK(rel < tolerance);
}
float bench(const std::function<void()> &f, int repeats = 30) {
    for (int i = 0; i < 3; i++)
        f();
    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));
    CUDA_OK(cudaEventRecord(a, cudaStreamPerThread));
    for (int i = 0; i < repeats; i++)
        f();
    CUDA_OK(cudaEventRecord(b, cudaStreamPerThread));
    CUDA_OK(cudaEventSynchronize(b));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / repeats;
}
void head(int rows, int width, int vocab, const std::string &real) {
    auto x = inputValues(size_t(rows) * width);
    auto w = inputValues(size_t(vocab) * width);
    if (!real.empty()) {
        auto raw = readFile(real);
        CHECK(raw.size() == w.size() * 2);
        std::memcpy(w.data(), raw.data(), raw.size());
    }
    std::vector<float> xf(x.size()), wf(w.size());
    for (size_t i = 0; i < x.size(); i++)
        xf[i] = float(x[i]);
    for (size_t i = 0; i < w.size(); i++)
        wf[i] = float(w[i]);
    Data dx, dw, fx, fw, ref, out;
    allocate(dx, BFLOAT16, {rows, width}, x.data());
    allocate(dw, BFLOAT16, {vocab, width}, w.data());
    allocate(fx, FLOAT32, {rows, width}, xf.data());
    allocate(fw, FLOAT32, {vocab, width}, wf.data());
    allocate(ref, FLOAT32, {rows, vocab});
    float alpha = 1, beta = 0;
    auto baseline = [&] {
        CHECK(cublasGemmEx(getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, vocab, rows, width, &alpha,
                           fw.cudaData, CUDA_R_32F, width, fx.cudaData, CUDA_R_32F, width, &beta,
                           ref.cudaData, CUDA_R_32F, vocab, CUBLAS_COMPUTE_32F_PEDANTIC,
                           CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);
    };
    auto candidate = [&] { CHECK(FastllmCudaBFloat16LinearFloat32(dx, dw, out)); };
    baseline();
    candidate();
    CHECK(out.dataType == FLOAT32);
    compare(out, ref, 1e-4, "HEAD");
    if (rows == 1 || rows == 8)
        printf("HEAD shape=%d,%d,%d ref_ms=%.6f compact_ms=%.6f\n", rows, width, vocab, bench(baseline),
               bench(candidate));
    // Existing exact-compaction helper must reject changed FP32 weights.
    if (vocab < 1000) {
        Data compact;
        CHECK(FastllmCudaTryCompactBFloat16Weight(fw, compact));
        float inexact = 1.000001f;
        CUDA_OK(cudaMemcpy(fw.cudaData, &inexact, 4, cudaMemcpyHostToDevice));
        CHECK(!FastllmCudaTryCompactBFloat16Weight(fw, compact));
    }
}
void woa(int tokens, int groups, int rank, int hidden, const std::string &real, bool graph) {
    auto x = inputValues(size_t(tokens) * groups * hidden);
    std::vector<uint8_t> w(size_t(groups) * rank * hidden);
    std::vector<float> sc(size_t(groups * rank / 32) * (hidden / 32));
    for (auto &v : w) {
        __nv_fp8_e4m3 q(float(int(rng() % 33) - 16) / 8.f);
        v = q.__x;
    }
    for (auto &v : sc)
        v = std::ldexp(1.f, int(rng() % 9) - 8);
    if (!real.empty()) {
        w = readFile(real + ".fp8");
        auto raw = readFile(real + ".scales");
        CHECK(w.size() == size_t(groups) * rank * hidden && raw.size() == sc.size() * 4);
        std::memcpy(sc.data(), raw.data(), raw.size());
    }
    std::vector<__nv_bfloat16> restored(w.size());
    for (size_t i = 0; i < w.size(); i++) {
        __nv_fp8_e4m3 q;
        q.__x = w[i];
        restored[i] = __float2bfloat16_rn(float(q) * sc[(i / hidden / 32) * (hidden / 32) + i % hidden / 32]);
    }
    Data dx, dw, bw, ref, out;
    allocate(dx, BFLOAT16, {1, tokens, groups, hidden}, x.data());
    allocate(dw, FP8_E4M3, {groups * rank, hidden}, w.data());
    dw.blockK = dw.blockM = 32;
    dw.scales = sc;
    allocate(bw, BFLOAT16, {groups * rank, hidden}, restored.data());
    auto baseline = [&] { CHECK(FastllmCudaDeepSeekV4WoA(dx, bw, groups, rank, ref, false)); };
    auto candidate = [&] { CHECK(FastllmCudaDeepSeekV41CompactWoA(dx, dw, groups, rank, out)); };
    // Uniform synthetic inputs are not necessarily already block-32 quantized;
    // disable the model prefill optimization here to test the exact BF16 fallback.
    setenv("FASTLLM_CUTLASS_MXFP8", "0", 1);
    baseline();
    candidate();
    compare(out, ref, 0.0032, "WOA");
    CHECK(!FastllmCudaDeepSeekV41CompactWoA(dx, dw, groups, rank, dx));
    CHECK(!FastllmCudaDeepSeekV41CompactWoA(dx, dw, groups, rank, dw));
    printf("WOA shape=%d,%d,%d,%d", tokens, groups, rank, hidden);
    if (tokens <= 16) {
        printf(" ref_ms=%.6f compact_ms=%.6f", bench(baseline), bench(candidate));
    }
    puts("");
    if (tokens >= 128 && !real.empty()) {
        Data quant;
        allocate(quant, BFLOAT16, dx.dims);
        CHECK(FastllmCudaDeepSeekV41QuantizeActivation(dx, quant));
        dx.CopyFrom(quant);
        baseline();
        setenv("FASTLLM_CUTLASS_MXFP8", "1", 1);
        CHECK(FastllmCudaTryMxfp8WoA(dx, dw, groups, rank, out));
        compare(out, ref, 0.0032, "WOA_MXFP8_PREFILL");
        Data oldPrefill;
        CHECK(FastllmCudaTryMxfp8WoA(dx, bw, groups, rank, oldPrefill));
        compare(out, oldPrefill, 0.0032, "WOA_MXFP8_OLD_PATH");
        setenv("FASTLLM_CUTLASS_MXFP8", "0", 1);
    }
    if (graph) {
        cudaGraph_t g;
        cudaGraphExec_t e;
        CUDA_OK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        candidate();
        CUDA_OK(cudaStreamEndCapture(cudaStreamPerThread, &g));
        CUDA_OK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
        for (int pass = 0; pass < 3; pass++) {
            auto next = inputValues(x.size());
            CUDA_OK(cudaMemcpyAsync(dx.cudaData, next.data(), next.size() * 2, cudaMemcpyHostToDevice,
                                    cudaStreamPerThread));
            CUDA_OK(cudaGraphLaunch(e, cudaStreamPerThread));
            baseline();
            compare(out, ref, 0.0032, "WOA_GRAPH");
        }
        cudaGraphExecDestroy(e);
        cudaGraphDestroy(g);
    }
}
int main(int argc, char **argv) {
    CUDA_OK(cudaSetDevice(0));
    if (FastllmCudaRuntimeArch() < 80)
        return 77;
    std::string real = argc > 1 ? argv[1] : "";
    for (int rows : {1, 4, 8, 16})
        head(rows, 512, 257, "");
    for (int t : {1, 2, 7, 8, 16, 17, 32, 128})
        woa(t, 2, 96, 96, "", t == 1 || t == 8);
    if (!real.empty()) {
        head(1, 5120, 129280, real + "/head.bf16");
        for (int t : {1, 8, 17, 128, 4096})
            woa(t, 8, 1024, 4096, real + "/woa", t == 1 || t == 8);
    }
    CUDA_OK(cudaDeviceSynchronize());
    puts("ALL_PASS: compact head FP32 logits, block32 WoA, fallback and graph replay");
    return 0;
}
