#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
using namespace fastllm;
static bool expectSpecialized = true;
void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void Require(bool b, const char *m) {
    if (!b)
        throw std::runtime_error(m);
}
template <class T> void Upload(Data &d, const std::vector<T> &v) {
    Check(cudaMemcpy(d.cudaData, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> std::vector<T> Read(Data &d) {
    std::vector<T> v(d.Count(0));
    Check(cudaMemcpy(v.data(), d.cudaData, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
template <class T> void Run(DataType type, int D, int M, bool inplace) {
    Data x(type, {M, D}, std::vector<float>(M * D, 0)), y(type, {M, D}, std::vector<float>(M * D, 0)),
        ref(type, {M, D}, std::vector<float>(M * D, 0));
    std::vector<float> hw(D);
    for (int i = 0; i < D; ++i)
        hw[i] = 1.00017f + .23f * sinf(i * .013f);
    Data w(FLOAT32, {D}, hw);
    for (Data *d : {&x, &y, &ref, &w})
        d->ToDevice(CUDA, std::vector<int>{0});
    Data &out = inplace ? x : y;
    setenv("FASTLLM_CUDA_RMSNORM_DECODE", "1", 1);
    constexpr float eps = 1e-6f;
    FastllmCudaRMSNorm(x, w, out, eps);
    Check(cudaDeviceSynchronize());
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(FastllmCudaRMSNorm(x, w, out, eps), "dispatch failed");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    size_t count = 0;
    Check(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    Check(cudaGraphGetNodes(graph, nodes.data(), &count));
    bool specialized = false;
    for (auto node : nodes) {
        cudaGraphNodeType nt;
        Check(cudaGraphNodeGetType(node, &nt));
        if (nt == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams p;
            Check(cudaGraphKernelNodeGetParams(node, &p));
#if CUDART_VERSION >= 13000
            const char *name = nullptr;
            Check(cudaFuncGetName(&name, p.func));
            if (std::strstr(name, "normdecode"))
                specialized = true;
#else
            if (D == 5120 && M == 1 && p.gridDim.x == 1 && p.blockDim.x == 512)
                specialized = true;
#endif
        }
    }
    Require(specialized == (expectSpecialized && D == 5120 && M == 1), "specialization/fallback selection incorrect");
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int step = 0; step < 5; ++step) {
        std::vector<T> hx(M * D);
        float mag = step == 0 ? 0 : step == 1 ? 1e-5f : step == 2 ? .7f : step == 3 ? 17.f : 50000.f;
        for (int i = 0; i < M * D; ++i)
            hx[i] = T(mag * sinf(i * .0137f + step * .73f));
        Upload(x, hx);
        setenv("FASTLLM_CUDA_RMSNORM_DECODE", "0", 1);
        Require(FastllmCudaRMSNorm(x, w, ref, eps), "fallback failed");
        setenv("FASTLLM_CUDA_RMSNORM_DECODE", "1", 1);
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaDeviceSynchronize());
        auto a = Read<T>(out), b = Read<T>(ref);
        double sq = 0, den = 0, ce = 0, cd = 0;
        for (int row = 0; row < M; ++row) {
            double ss = 0;
            for (int i = 0; i < D; ++i) {
                double v = float(hx[row * D + i]);
                ss += v * v;
            }
            float inv = 1.f / sqrtf(float(ss / D) + eps);
            for (int i = 0; i < D; ++i) {
                int idx = row * D + i;
                float av = float(a[idx]), bv = float(b[idx]),
                      expected = float(T((float(hx[idx]) * inv) * hw[i]));
                Require(std::isfinite(av), "nonfinite output");
                sq += (av - bv) * (av - bv);
                den += bv * bv;
                ce += (av - expected) * (av - expected);
                cd += expected * expected;
            }
        }
        double r = sqrt(sq / (den + 1e-30)), c = sqrt(ce / (cd + 1e-30));
        printf("type=%d D=%d M=%d inplace=%d step=%d fallback_rms=%.9g cpu_rms=%.9g\n", int(type), D, M,
               inplace, step, r, c);
        Require(r < .001 && c < .001, "numerical disagreement");
        if (!inplace) {
            auto after = Read<T>(x);
            Require(memcmp(after.data(), hx.data(), hx.size() * sizeof(T)) == 0, "input changed");
        }
    }
    Require(Read<float>(w) == hw, "norm weights changed");
    Check(cudaGraphExecDestroy(exec));
    Check(cudaGraphDestroy(graph));
}
int main() {
    try {
        const char *expect = std::getenv("EXPECT_NORM_DECODE");
        expectSpecialized = !expect || std::strcmp(expect, "0");
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || !n)
            return 77;
        Executor ex;
        ex.SetFirstDevice("cuda:0");
        Run<half>(FLOAT16, 5120, 1, false);
        Run<half>(FLOAT16, 5120, 1, true);
        Run<half>(FLOAT16, 5120, 2, false);
        Run<half>(FLOAT16, 4096, 1, false);
        Run<__nv_bfloat16>(BFLOAT16, 5120, 1, false);
        Run<__nv_bfloat16>(BFLOAT16, 5120, 1, true);
        Run<__nv_bfloat16>(BFLOAT16, 5120, 2, false);
        Run<__nv_bfloat16>(BFLOAT16, 4096, 1, false);
        puts("PASS");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
