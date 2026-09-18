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
float decode(int v) {
    float lut[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    return lut[v & 7] * (v & 8 ? -1 : 1);
}
int si(int n) {
    int t = ((n & 7) << 3) | ((n & 63) >> 3), l = t & 3;
    return (n & ~63) + (t & ~3) + ((l & 1) << 1) + (l >> 1);
}
int main() {
    using namespace fastllm_native_prefill;
    setenv("FASTLLM_CUDA_NATIVE_NVFP4_PREFILL", "1", 1);
    const int M = 128, N = 128, K = 256, G = K / 16;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rand(-1, 1);
    std::vector<half> x(M * K), out(M * N), bias(N);
    std::vector<uint32_t> w(N * K / 8, 0);
    std::vector<uint8_t> s(N * G), codes(N * K);
    std::vector<float> sf(N * G), xq(M * K);
    for (auto &a : x)
        a = __float2half(rand(rng) * 2);
    for (int n = 0; n < N; n++) {
        bias[n] = __float2half(rand(rng));
        for (int g = 0; g < G; g++) {
            float f = float(__nv_fp8_e4m3(.02f * (1 + rng() % 128)));
            sf[n * G + g] = f;
            s[g * N + si(n)] = __half_as_ushort(__float2half(f * 128)) >> 7;
            for (int k = 0; k < 16; k++) {
                int q = rng() % 16;
                codes[n * K + g * 16 + k] = q;
                int r = n % 64, j = k / 2, word = (r % 8) * 16 + (j % 4) * 4 + r / 16,
                    shift = (r % 16 >= 8 ? 8 : 0) + (j >= 4 ? 4 : 0) + (k % 2 ? 16 : 0);
                w[(g * (N / 64) + n / 64) * 128 + word] |= uint32_t(q) << shift;
            }
        }
    }
    for (int m = 0; m < M; m++) {
        float mx = 0;
        for (int k = 0; k < K; k++)
            mx = fmax(mx, fabs(__half2float(x[m * K + k])));
        float glob = mx / (448 * 6);
        for (int g = 0; g < G; g++) {
            float maxg = 0;
            for (int j = 0; j < 16; j++)
                maxg = fmax(maxg, fabs(__half2float(x[m * K + g * 16 + j])));
            float sg = float(__nv_fp8_e4m3(maxg / (6 * glob)));
            for (int j = 0; j < 16; j++) {
                float a = __half2float(x[m * K + g * 16 + j]);
                xq[m * K + g * 16 + j] =
                    decode(__nv_cvt_float_to_fp4(a / (sg * glob), __NV_E2M1, cudaRoundNearest)) * sg * glob;
            }
        }
    }
    half *dx, *dy, *db;
    uint32_t *dw;
    uint8_t *ds;
    float *dg;
    cudaMalloc(&dx, x.size() * 2);
    cudaMalloc(&dy, out.size() * 2);
    cudaMalloc(&db, N * 2);
    cudaMalloc(&dw, w.size() * 4);
    cudaMalloc(&ds, s.size());
    cudaMalloc(&dg, 4);
    float global = 128;
    cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, s.data(), s.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db, bias.data(), N * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, &global, 4, cudaMemcpyHostToDevice);
    if (!Fp4(dx, dw, ds, dg, db, dy, M, N, N, K)) {
        puts("launch false");
        return 1;
    }
    auto err = cudaDeviceSynchronize();
    printf("sync=%s\n", cudaGetErrorString(err));
    if (err)
        return 2;
    cudaMemcpy(out.data(), dy, out.size() * 2, cudaMemcpyDeviceToHost);
    float maxerr = 0;
    double se = 0, sr = 0;
    int bad = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0, orig = 0;
            for (int k = 0; k < K; k++) {
                float wt = decode(codes[n * K + k]) * sf[n * G + k / 16];
                acc += xq[m * K + k] * wt;
                orig += __half2float(x[m * K + k]) * wt;
            }
            half z = __hadd(__float2half(acc), bias[n]);
            float e = fabs(__half2float(z) - __half2float(out[m * N + n]));
            maxerr = fmax(maxerr, e);
            bad += e > .01f + .002f * fabs(__half2float(z));
            float d = __half2float(out[m * N + n]) - orig - __half2float(bias[n]);
            se += d * d;
            sr += orig * orig;
        }
    printf("CPU quantized max_abs=%g bad=%d activation_quantization rel_rms=%g\n", maxerr, bad,
           sqrt(se / sr));
    std::vector<half> plain = out, got(out.size());
    if (!Fp4(dx, dw, ds, dg, db, dy, M, N, N, K, 1))
        return 3;
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dy, M * N, cudaMemcpyDeviceToHost);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N / 2; n++) {
            half a = plain[m * N + n], b = plain[m * N + n + N / 2];
            half expected =
                __hmul(__hdiv(a, __hadd(__float2half(1), __float2half(expf(-__half2float(a))))), b);
            float v = __half2float(got[m * N / 2 + n]), ref = __half2float(expected);
            if (!(v == ref || (std::isnan(v) && std::isnan(ref))))
                bad++;
        }
    cudaMemcpy(dy, plain.data(), plain.size() * 2, cudaMemcpyHostToDevice);
    if (!Fp4(dx, dw, ds, dg, db, dy, M, N, N, K, 2))
        return 4;
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dy, got.size() * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++)
        if (__half_as_ushort(got[i]) != __half_as_ushort(__hadd(plain[i], plain[i])))
            bad++;
    printf("fused bad=%d\n", bad);
    return bad ? 1 : 0;
}
