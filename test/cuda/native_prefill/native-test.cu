#include <cstring>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
extern "C" bool FastllmCudaGetNcclForceSync() { return true; }
extern "C" cudaError_t FastllmCudaCheckedMalloc(void **p, size_t bytes, const char *, int) {
    return cudaMalloc(p, bytes);
}
#include "fastllm-native-lowbit-prefill.cuh"
#include <cmath>
#include <random>
#include <vector>
int main() {
    using namespace fastllm_native_prefill;
    setenv("FASTLLM_CUDA_NATIVE_FP8_PREFILL", "1", 1);
    int M = 32, N = 128, K = 256;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rand(-1, 1);
    std::vector<half> x(M * K), out(M * N), bias(N);
    std::vector<uint8_t> w(N * K);
    std::vector<float> s(N), xs(M), xq(M * K);
    for (auto &a : x)
        a = __float2half(rand(rng));
    for (int n = 0; n < N; n++) {
        s[n] = .01f * (1 + n % 7);
        bias[n] = __float2half(rand(rng));
        for (int k = 0; k < K; k++)
            w[n * K + k] = __nv_fp8_e4m3(rand(rng) * 24).__x;
    }
    for (int m = 0; m < M; m++) {
        float mx = 0;
        for (int k = 0; k < K; k++)
            mx = fmax(mx, fabs(__half2float(x[m * K + k])));
        xs[m] = mx / 448;
        for (int k = 0; k < K; k++)
            xq[m * K + k] = float(__nv_fp8_e4m3(__half2float(x[m * K + k]) / xs[m]));
    }
    half *dx, *dy, *db;
    uint8_t *dw;
    float *ds;
    cudaMalloc(&dx, x.size() * 2);
    cudaMalloc(&dy, out.size() * 2);
    cudaMalloc(&db, N * 2);
    cudaMalloc(&dw, w.size());
    cudaMalloc(&ds, N * 4);
    cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, s.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bias.data(), N * 2, cudaMemcpyHostToDevice);
    if (!Fp8(dx, dw, ds, db, dy, M, N, K)) {
        puts("launch false");
        return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), dy, out.size() * 2, cudaMemcpyDeviceToHost);
    float maxerr = 0, origerr = 0;
    double se = 0, sr = 0;
    int bad = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0, orig = 0;
            for (int k = 0; k < K; k++) {
                __nv_fp8_e4m3 v;
                v.__x = w[n * K + k];
                acc += xq[m * K + k] * float(v);
                orig += __half2float(x[m * K + k]) * float(v) * s[n];
            }
            half z = __hadd(__float2half(acc * xs[m] * s[n]), bias[n]);
            float e = fabs(__half2float(z) - __half2float(out[m * N + n]));
            maxerr = fmax(maxerr, e);
            bad += e > .003f + .001f * fabs(__half2float(z));
            float err = __half2float(out[m * N + n]) - orig - __half2float(bias[n]);
            origerr = fmax(origerr, fabs(err));
            se += err * err;
            sr += orig * orig;
        }
    printf("cpu_quantized max_abs=%g bad=%d activation_quantization rel_rms=%g max_abs=%g\n", maxerr, bad,
           sqrt(se / sr), origerr);

    std::vector<half> plain = out, got(out.size());
    if (!Fp8(dx, dw, ds, db, dy, M, N, K, 1))
        return 4;
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dy, M * N, cudaMemcpyDeviceToHost);
    int fusedBad = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N / 2; n++) {
            half a = plain[m * N + n], b = plain[m * N + n + N / 2];
            half expected =
                __hmul(__hdiv(a, __hadd(__float2half(1), __float2half(expf(-__half2float(a))))), b);
            float v = __half2float(got[m * N / 2 + n]), ref = __half2float(expected);
            if (fabs(v - ref) > .001f + .002f * fabs(ref))
                fusedBad++;
        }
    cudaMemcpy(dy, plain.data(), plain.size() * 2, cudaMemcpyHostToDevice);
    if (!Fp8(dx, dw, ds, db, dy, M, N, K, 2))
        return 5;
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dy, got.size() * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++)
        if (__half_as_ushort(got[i]) != __half_as_ushort(__hadd(plain[i], plain[i])))
            fusedBad++;
    printf("fused bad=%d\n", fusedBad);
    bad += fusedBad;
    if (Fp8(dx, dw, ds, nullptr, dy, 1, N, K)) {
        puts("decode fallback failed");
        return 2;
    }
    cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
    bool captured = Fp8(dx, dw, ds, nullptr, dy, M, N, K);
    cudaGraph_t graph;
    cudaStreamEndCapture(cudaStreamPerThread, &graph);
    cudaGraphDestroy(graph);
    if (captured) {
        puts("capture fallback failed");
        return 3;
    }
    return bad ? 1 : 0;
}
