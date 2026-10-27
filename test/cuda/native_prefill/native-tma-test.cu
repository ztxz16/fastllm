#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
extern "C" bool FastllmCudaGetNcclForceSync() { return true; }
extern "C" cudaError_t FastllmCudaCheckedMalloc(void **p, size_t bytes, const char *, int) {
    return cudaMalloc(p, bytes);
}
#include "fastllm-native-lowbit-prefill.cuh"
#include "fastllm-native-nvfp4-tma.cuh"
#define CK(call)                                                                                             \
    do {                                                                                                     \
        auto e = (call);                                                                                     \
        if (e) {                                                                                             \
            printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__);                                    \
            return 2;                                                                                        \
        }                                                                                                    \
    } while (0)
int main() {
    using namespace fastllm_native_prefill;
    setenv("FASTLLM_CUDA_NATIVE_NVFP4_PREFILL", "1", 1);
    int badTotal = 0;
    for (int mode : {1, 2, 0}) {
        int M = 256, N = mode == 1 ? 34816 : 5120, K = mode == 1 ? 5120 : 17408,
            width = mode == 1 ? N / 2 : N;
        half *x, *reference, *out;
        uint32_t *w;
        uint8_t *sf;
        float *global;
        CK(cudaMalloc(&x, size_t(M) * K * 2));
        CK(cudaMalloc(&w, size_t(N) * K / 2));
        CK(cudaMalloc(&sf, size_t(N) * K / 16));
        CK(cudaMalloc(&global, 4));
        CK(cudaMalloc(&reference, size_t(M) * width * 2));
        CK(cudaMalloc(&out, size_t(M) * width * 2));
        std::mt19937 rng(42);
        std::vector<uint32_t> hw(size_t(N) * K / 8);
        for (auto &v : hw)
            v = rng();
        CK(cudaMemcpy(w, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
        std::vector<half> hx(size_t(M) * K);
        for (auto &v : hx)
            v = __float2half(float(int(rng() % 2000) - 1000) / 1000);
        CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        std::vector<uint8_t> hs(size_t(N) * K / 16);
        for (auto &v : hs)
            v = 128 + rng() % 20;
        CK(cudaMemcpy(sf, hs.data(), hs.size(), cudaMemcpyHostToDevice));
        float alpha = 32;
        CK(cudaMemcpy(global, &alpha, 4, cudaMemcpyHostToDevice));
        std::vector<half> residual(size_t(M) * width);
        for (auto &v : residual)
            v = __float2half(float(int(rng() % 2000) - 1000) / 1000);
        CK(cudaMemcpy(reference, residual.data(), residual.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(out, residual.data(), residual.size() * 2, cudaMemcpyHostToDevice));
        setenv("FASTLLM_CUDA_NATIVE_NVFP4_TMA", "0", 1);
        if (!Fp4(x, w, sf, global, nullptr, reference, M, N, N, K, mode))
            return 3;
        setenv("FASTLLM_CUDA_NATIVE_NVFP4_TMA", "1", 1);
        if (mode != 0 && !Nvfp4TmaCanRun(M, N, K, mode)) {
            puts("TMA image unavailable");
            return 4;
        }
        if (!Fp4(x, w, sf, global, nullptr, out, M, N, N, K, mode))
            return 5;
        CK(cudaDeviceSynchronize());
        std::vector<half> a(residual.size()), b(a.size());
        CK(cudaMemcpy(a.data(), reference, a.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), out, b.size() * 2, cudaMemcpyDeviceToHost));
        int bad = 0;
        float maxe = 0;
        double se = 0, sr = 0;
        for (size_t i = 0; i < a.size(); i++) {
            float v = __half2float(a[i]), u = __half2float(b[i]), e = fabs(v - u);
            bad += !std::isfinite(u) || e > .001f + .002f * fabs(v);
            maxe = fmax(maxe, e);
            se += e * e;
            sr += v * v;
        }
        printf("mode=%d bad=%d max_abs=%g rel_rms=%g\n", mode, bad, maxe, sqrt(se / sr));
        badTotal += bad;
        if (Nvfp4TmaCanRun(128, N, K, mode) || Nvfp4TmaCanRun(M, N + 128, K, mode))
            return 6;
        if (Fp4(x, w, sf, global, nullptr, out, 127, N, N, K, mode))
            return 7;
        CK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        bool captured = Fp4(x, w, sf, global, nullptr, out, M, N, N, K, mode);
        cudaGraph_t graph;
        CK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        CK(cudaGraphDestroy(graph));
        if (captured)
            return 8;
        for (void *p : {(void *)x, (void *)reference, (void *)out, (void *)w, (void *)sf, (void *)global})
            CK(cudaFree(p));
    }
    return badTotal ? 1 : 0;
}
