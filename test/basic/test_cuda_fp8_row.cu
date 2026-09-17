#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <string>
extern void LaunchFastllmGemmFp16FP8E4M3(half *, uint8_t *, half *, half *, float *, int, int, int, int, int);
extern void LaunchFastllmGemmBF16FP8E4M3(__nv_bfloat16 *, uint8_t *, __nv_bfloat16 *, __nv_bfloat16 *,
                                         float *, int, int, int, int, int);
void CheckAt(cudaError_t e, int line) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::to_string(line) + ": " + cudaGetErrorString(e));
}
#define Check(e) CheckAt(e, __LINE__)
void Req(bool b, const char *m) {
    if (!b)
        throw std::runtime_error(m);
}
template <class T> T *Upload(const std::vector<T> &v) {
    T *p;
    Check(cudaMalloc(&p, v.size() * sizeof(T)));
    Check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> void Launch(T *x, uint8_t *w, T *y, T *b, float *s, int M, int K, int N, int bm, int bk) {
    if constexpr (__is_same(T, half))
        LaunchFastllmGemmFp16FP8E4M3(x, w, y, b, s, M, K, N, bm, bk);
    else
        LaunchFastllmGemmBF16FP8E4M3(x, w, y, b, s, M, K, N, bm, bk);
}
template <class T> void Run(int K, int N, int M, bool bias, bool block, bool expected, int scaleBlock = 0) {
    if (std::getenv("EXPECT_FUSED") && !std::strcmp(std::getenv("EXPECT_FUSED"), "0"))
        expected = false;
    int bm = block ? 128 : (scaleBlock ? scaleBlock : K), bk = block ? 128 : 1, cols = (K + bm - 1) / bm;
    std::vector<T> x(M * K), b(N);
    std::vector<uint8_t> w(size_t(N) * K);
    std::vector<float> s(size_t((N + bk - 1) / bk) * cols);
    for (int i = 0; i < M * K; ++i)
        x[i] = T(.7f * sinf(i * .17f));
    for (int i = 0; i < N; ++i)
        b[i] = T(.03f * cosf(i * .2f));
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = uint8_t(0x28 + i % 32) | ((i % 7 < 3) ? 0x80 : 0);
    for (size_t i = 0; i < s.size(); ++i)
        s[i] = .01317f + .007f * sinf(i * .37f);
    auto dx = Upload(x);
    auto dw = Upload(w);
    auto ds = Upload(s);
    auto db = Upload(b);
    auto dy = Upload(std::vector<T>(M * N));
    std::vector<T> baseline[2];
    for (int mode = 0; mode < 2; ++mode) {
        setenv("FASTLLM_CUDA_FP8_ROW_GEMV", mode ? "1" : "0", 1);
        Launch(dx, dw, dy, bias ? db : nullptr, ds, M, K, N, bm, bk);
        Check(cudaDeviceSynchronize());
        cudaGraph_t g;
        cudaGraphExec_t e;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Launch(dx, dw, dy, bias ? db : nullptr, ds, M, K, N, bm, bk);
        Check(cudaStreamEndCapture(cudaStreamPerThread, &g));
        size_t count = 0;
        Check(cudaGraphGetNodes(g, nullptr, &count));
        std::vector<cudaGraphNode_t> nodes(count);
        Check(cudaGraphGetNodes(g, nodes.data(), &count));
        bool seen = false;
        for (auto node : nodes) {
            cudaGraphNodeType nt;
            Check(cudaGraphNodeGetType(node, &nt));
            if (nt == cudaGraphNodeTypeKernel) {
                // Driver handles remain valid across separate static runtimes.
                CUDA_KERNEL_NODE_PARAMS p{};
                Req(cuGraphKernelNodeGetParams((CUgraphNode)node, &p) == CUDA_SUCCESS,
                    "driver graph kernel params");
                const char *name = nullptr;
                Req(cuFuncGetName(&name, p.func) == CUDA_SUCCESS, "driver kernel name");
                seen |= std::strstr(name, "fp8row") != nullptr;
            }
        }
        Req(seen == (bool(mode) && expected), "graph dispatch mismatch");
        Check(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            for (int i = 0; i < M * K; ++i)
                x[i] = T((replay ? -.3f : .7f) * sinf(i * .17f));
            Check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
            Check(cudaGraphLaunch(e, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<T> y(M * N);
            Check(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost));
            double err = 0, ss = 0;
            // Sample both ends, odd tail rows and interior rows; LM head is >1GB.
            for (int m = 0; m < M; ++m)
                for (int sample = 0; sample < 33; ++sample) {
                    int row = sample == 32 ? N - 1 : sample * (N - 1) / 32;
                    double ref = 0, magnitude = 0;
                    for (int j = 0; j < K; ++j) {
                        __nv_fp8_e4m3 code;
                        code.__x = w[size_t(row) * K + j];
                        double term = double(float(x[m * K + j])) * float(code) * s[size_t(row / bk) * cols + j / bm];
                        ref += term;
                        magnitude += std::abs(term);
                    }
                    ref += bias ? float(b[row]) : 0.;
                    double a = float(y[m * N + row]);
                    Req(std::isfinite(a), "nonfinite output");
                    // Legacy FP16 arithmetic is cancellation-sensitive for long
                    // dots. Check its forward error against sum(abs(terms));
                    // the new path also retains the tighter relative RMS check.
                    Req(std::abs(a - ref) <= .003 * magnitude + .01 * std::abs(ref) + 1e-5,
                        "dot-product forward error");
                    err += (a - ref) * (a - ref);
                    ss += ref * ref;
                }
            double rel = sqrt(err / (ss + 1e-20));
            printf("row bf16=%d K=%d N=%d M=%d block=%d mode=%d replay=%d rel=%g\n", !__is_same(T, half), K,
                   N, M, block, mode, replay, rel);
            if (seen)
                Req(rel < (__is_same(T, half) ? .004 : .03), "FP8 CPU reference mismatch");
            if (mode == 0)
                baseline[replay] = y;
            else if (!expected)
                Req(!std::memcmp(y.data(), baseline[replay].data(), y.size() * sizeof(T)),
                    "fallback differs from disabled path");
        }
        cudaGraphExecDestroy(e);
        cudaGraphDestroy(g);
    }
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(ds);
    cudaFree(db);
    cudaFree(dy);
}
int main() {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || !n)
            return 77;
#define RUN(T)                                                                                               \
    Run<T>(5120, 14336, 1, false, false, true);                                                              \
    Run<T>(5120, 248320, 1, false, false, true);                                                             \
    Run<T>(1024, 4099, 1, true, false, true);                                                                \
    Run<T>(768, 4096, 1, false, false, true);                                                               \
    Run<T>(1024, 4096, 2, true, false, false);                                                               \
    Run<T>(1024, 4096, 1, false, true, false)
        RUN(half);
        RUN(__nv_bfloat16);
        Run<half>(512, 4099, 1, true, false, true);
        Run<__nv_bfloat16>(512, 4099, 1, true, false, true);
        Run<half>(32768, 4099, 1, true, false, true);
        Run<__nv_bfloat16>(32768, 4099, 1, true, false, true);
        Run<half>(1024, 4095, 1, false, false, false);
        Run<__nv_bfloat16>(1024, 4095, 1, false, false, false);
        for (int K : {1536, 1920, 2048, 2304, 3072, 4352, 5802, 5803, 8704}) {
            Run<half>(K, 5120, 1, true, false, true, 17408);
            Run<__nv_bfloat16>(K, 5120, 1, true, false, true, 17408);
        }
        Run<half>(5803, 4099, 2, true, false, false, 17408);
        Run<__nv_bfloat16>(5803, 4099, 2, true, false, false, 17408);
        puts("PASS row FP8");
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
