#include "fastllm-gdn-prepare-wy.cuh"
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
__global__ void Mask(half *a, half *g, half *d) {
    int ch = blockIdx.x, t = threadIdx.x;
    if (!t)
        for (int i = 1; i < 64; i++)
            g[ch * 64 + i] = __hadd(g[ch * 64 + i], g[ch * 64 + i - 1]);
    __syncthreads();
    for (int i = t; i < 4096; i += 256) {
        int r = i / 64, c = i % 64;
        half v = __float2half(r >= c ? expf(__half2float(g[ch * 64 + r]) - __half2float(g[ch * 64 + c])) : 0);
        d[ch * 4096 + i] = v;
        a[ch * 4096 + i] = r > c ? __hmul(a[ch * 4096 + i], __hneg(v)) : __float2half(0);
    }
}
int main() {
    constexpr int C = 3;
    size_t size = size_t(C) * 8192;
    half *k, *kb, *v, *g, *gg, *a, *d, *dd, *vo, *ko, *vv, *kk;
    for (half **p : {&k, &kb, &v, &vo, &ko, &vv, &kk})
        cudaMalloc(p, size * 2);
    for (half **p : {&a, &d, &dd})
        cudaMalloc(p, C * 4096 * 2);
    for (half **p : {&g, &gg})
        cudaMalloc(p, C * 64 * 2);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-1, 1);
    std::vector<half> h(size);
    for (auto &x : h)
        x = __float2half(.15f * dist(rng));
    cudaMemcpy(k, h.data(), size * 2, cudaMemcpyHostToDevice);
    for (auto &x : h)
        x = __hmul(x, __float2half(.7f));
    cudaMemcpy(kb, h.data(), size * 2, cudaMemcpyHostToDevice);
    for (auto &x : h)
        x = __float2half(.5f * dist(rng));
    cudaMemcpy(v, h.data(), size * 2, cudaMemcpyHostToDevice);
    for (int i = 0; i < C * 64; i++)
        h[i] = __float2half(-.01f * abs(dist(rng)));
    cudaMemcpy(g, h.data(), C * 64 * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(gg, g, C * 64 * 2, cudaMemcpyDeviceToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float one = 1, zero = 0;
    auto st = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 128, &one, k, CUDA_R_16F,
                                         128, 8192, kb, CUDA_R_16F, 128, 8192, &zero, a, CUDA_R_16F, 64, 4096,
                                         C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st) {
        printf("cublas %d\n", st);
        return 1;
    }
    Mask<<<C, 256>>>(a, g, d);
    fastllm_gdn_wy::Prepare<4><<<C, 256, 34944>>>(a, v, kb, g, vo, ko);
    fastllm_gdn_wy::Prepare<8, true><<<C, 512, 43136>>>(nullptr, v, kb, gg, vv, kk, k, gg, dd);
    auto error = cudaDeviceSynchronize();
    if (error) {
        puts(cudaGetErrorString(error));
        return 2;
    }
    int bad = 0;
    float maxe = 0;
    double se = 0, sr = 0;
    std::vector<half> x(size), y(size);
    for (int pass = 0; pass < 2; pass++) {
        cudaMemcpy(x.data(), pass ? ko : vo, size * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(y.data(), pass ? kk : vv, size * 2, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < size; i++) {
            float a = __half2float(x[i]), b = __half2float(y[i]), e = fabs(a - b);
            bad += e > .0002f + .003f * fabs(a);
            maxe = fmax(maxe, e);
            se += e * e;
            sr += a * a;
        }
    }
    printf("bad=%d max_abs=%g rel_rms=%g\n", bad, maxe, sqrt(se / sr));
    cudaMemcpy(x.data(), g, C * 64 * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), gg, C * 64 * 2, cudaMemcpyDeviceToHost);
    if (memcmp(x.data(), y.data(), C * 64 * 2)) {
        puts("prefix mismatch");
        return 3;
    }
    cudaMemcpy(x.data(), d, C * 4096 * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), dd, C * 4096 * 2, cudaMemcpyDeviceToHost);
    if (memcmp(x.data(), y.data(), C * 4096 * 2)) {
        puts("decay mismatch");
        return 4;
    }

    cudaMemcpy(gg, h.data(), C * 64 * 2, cudaMemcpyHostToDevice);
    fastllm_gdn_wy::Prepare<8, true><<<C, 512, 43136>>>(nullptr, v, kb, gg, vv, v, k, gg, dd);
    error = cudaDeviceSynchronize();
    if (error) {
        puts(cudaGetErrorString(error));
        return 5;
    }
    cudaMemcpy(x.data(), kk, size * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), v, size * 2, cudaMemcpyDeviceToHost);
    int alias = memcmp(x.data(), y.data(), size * 2);
    printf("alias_equal=%d\n", alias == 0);
    return bad || alias ? 1 : 0;
}
