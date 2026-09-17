#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda-rmsnorm-small-linear.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using namespace fastllm;
void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void Req(bool b, const char *m) {
    if (!b)
        throw std::runtime_error(m);
}
template <class T> std::vector<T> Read(Data &d) {
    std::vector<T> v(d.Count(0));
    Check(cudaMemcpy(v.data(), d.cudaData, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
template <class T> void Run(DataType type, int D, int N, int M, bool bias, bool fused) {
    if (std::getenv("EXPECT_FUSED") && !std::strcmp(std::getenv("EXPECT_FUSED"), "0"))
        fused = false;
    std::vector<float> x(M * D), g(D), w(N * D), b(N);
    for (int i = 0; i < M * D; ++i)
        x[i] = .7f * sinf(i * .127f) + .13f * cosf(i * .071f);
    for (int i = 0; i < D; ++i)
        g[i] = 1.00017f + .2f * cosf(i * .021f);
    for (int i = 0; i < N * D; ++i)
        w[i] = .02f * sinf(i * .217f) + .01f * cosf(i * .111f);
    for (int i = 0; i < N; ++i)
        b[i] = .01f * sinf(i);
    Data dx(type, {M, D}, x), dg(FLOAT32, {D}, g), dw(type, {N, D}, w), db(FLOAT32, {N}, b), empty, y, o;
    Data &biasData = bias ? db : empty;
    for (Data *d : {&dx, &dg, &dw})
        d->ToDevice(CUDA, std::vector<int>{0});
    if (bias)
        db.ToDevice(CUDA, std::vector<int>{0});
    const auto storedWeight = Read<T>(dw);
    setenv("FASTLLM_CUDA_RMSNORM_SMALL_LINEAR", "1", 1);
    Req(FastllmCudaRMSNormSmallLinearCanRun(dx, dg, dw, biasData, y, o) == fused, "CanRun mismatch");
    Req(!FastllmCudaRMSNormSmallLinearCanRun(dx, dg, dw, biasData, dx, o), "input alias accepted");
    // Capability misses must leave metadata/storage untouched; these layouts
    // are deliberately invalid for fusion and must never reach its launch.
    dw.IsRepacked = true;
    Req(!FastllmCudaRMSNormSmallLinearCanRun(dx, dg, dw, biasData, y, o), "repacked weight accepted");
    dw.IsRepacked = false;
    auto savedStride = dx.strides.back();
    dx.strides.back() = 2;
    Req(!FastllmCudaRMSNormSmallLinearCanRun(dx, dg, dw, biasData, y, o), "strided input accepted");
    dx.strides.back() = savedStride;
    auto savedType = dw.dataType;
    dw.dataType = FLOAT32;
    Req(!FastllmCudaRMSNormSmallLinearCanRun(dx, dg, dw, biasData, y, o), "mixed weight type accepted");
    dw.dataType = savedType;
    auto savedIds = dx.dataDeviceIds;
    dx.dataDeviceIds = {1};
    Req(!FastllmCudaRMSNormSmallLinearCanRun(dx, dg, dw, biasData, y, o), "foreign device accepted");
    dx.dataDeviceIds = savedIds;
    auto savedPtr = dg.cudaData;
    dg.cudaData = static_cast<char *>(savedPtr) + 4;
    Req(!FastllmCudaRMSNormSmallLinearCanRun(dx, dg, dw, biasData, y, o), "unaligned norm accepted");
    dg.cudaData = savedPtr;
    Req(!FastllmCudaRMSNormSmallLinearCanRun(dx, dg, dw, biasData, y, y), "output alias accepted");
    if (std::getenv("TEST_SMALL_ALIAS")) {
        setenv("FASTLLM_CUDA_RMSNORM_SMALL_LINEAR", "0", 1);
        CudaRMSNormSmallLinearBlock(dx, dg, dw, biasData, dx, o, 1e-6f);
        fprintf(stderr, "INVALID_ALIAS_RETURNED\n");
        std::exit(1);
    }
    Req(CudaRMSNormSmallLinearBlock(dx, dg, dw, biasData, y, o, 1e-6f) == fused, "Block mismatch");
    Check(cudaDeviceSynchronize());
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    CudaRMSNormSmallLinearBlock(dx, dg, dw, biasData, y, o, 1e-6f);
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
        std::vector<T> hx(M * D);
        for (int i = 0; i < M * D; ++i)
            hx[i] = T(x[i] * (replay ? -.37f : 1.f));
        Check(cudaMemcpy(dx.cudaData, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        auto hy = Read<T>(y), ho = Read<T>(o);
        double se = 0, ss = 0, ne = 0, ns = 0;
        for (int m = 0; m < M; ++m) {
            double sq = 0;
            for (int j = 0; j < D; ++j)
                sq += double(float(hx[m * D + j])) * float(hx[m * D + j]);
            float inv = 1.f / sqrtf(sq / D + 1e-6f);
            std::vector<T> ref(D);
            for (int j = 0; j < D; ++j) {
                ref[j] = T((float(hx[m * D + j]) * inv) * g[j]);
                double a = float(hy[m * D + j]), v = float(ref[j]);
                ne += (a - v) * (a - v);
                ns += v * v;
            }
            for (int i = 0; i < N; ++i) {
                double sum = 0;
                for (int j = 0; j < D; ++j)
                    sum += double(float(ref[j])) * float(storedWeight[i * D + j]);
                float v = float(T(sum + (bias ? b[i] : 0))), a = float(ho[m * N + i]);
                Req(std::isfinite(a), "nonfinite output");
                se += (a - v) * (a - v);
                ss += v * v;
            }
        }
        double rel = sqrt(se / (ss + 1e-20)), nr = sqrt(ne / (ns + 1e-20));
        printf("small type=%d K=%d N=%d M=%d fused=%d replay=%d rel=%g norm=%g\n", type, D, N, M, fused,
               replay, rel, nr);
        Req(rel < (type == FLOAT16 ? .004 : .01), "projection error");
        Req(nr < (type == FLOAT16 ? .0008 : .006), "norm error");
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    setenv("FASTLLM_CUDA_RMSNORM_SMALL_LINEAR", "0", 1);
    Req(!CudaRMSNormSmallLinearBlock(dx, dg, dw, biasData, y, o, 1e-6f), "disabled fusion ran");
    Check(cudaDeviceSynchronize());
    setenv("FASTLLM_CUDA_RMSNORM_SMALL_LINEAR", "1", 1);
}
int main() {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || !n)
            return 77;
        Executor ex;
        ex.SetFirstDevice("cuda:0");
#define RUN(TY, T)                                                                                           \
    Run<T>(TY, 5120, 96, 1, false, true);                                                                    \
    Run<T>(TY, 4096, 97, 3, true, true);                                                                     \
    Run<T>(TY, 1024, 1, 8, false, true);                                                                     \
    Run<T>(TY, 8192, 256, 1, true, true);                                                                    \
    Run<T>(TY, 1536, 96, 1, false, false);                                                                   \
    Run<T>(TY, 1024, 257, 1, false, false);                                                                  \
    Run<T>(TY, 1024, 96, 9, false, false)
        RUN(FLOAT16, half);
        RUN(BFLOAT16, __nv_bfloat16);
        for (int width : {2048, 3072, 6144, 7168}) {
            Run<half>(FLOAT16, width, 95, 2, true, true);
            Run<__nv_bfloat16>(BFLOAT16, width, 95, 2, true, true);
        }
        printf("PASS RMSNormSmallLinear\n");
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
