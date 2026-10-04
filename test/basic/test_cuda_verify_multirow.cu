#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace fastllm;
void LaunchFastllmGemmBf16Bf16(__nv_bfloat16 *, __nv_bfloat16 *, __nv_bfloat16 *, __nv_bfloat16 *, int, int,
                               int);
static int checks = 0;
static void Check(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
static void Cuda(cudaError_t e) {
    Check(e == cudaSuccess, cudaGetErrorString(e));
}
static void Allocate(Data &d) {
    d.ToDevice(DataDevice::CUDA, {0}, false);
    d.Allocate(false);
}
template <class T> static std::vector<T> Read(Data &d) {
    std::vector<T> v(d.GetBytes() / sizeof(T));
    Cuda(cudaMemcpy(v.data(), d.cudaData, d.GetBytes(), cudaMemcpyDeviceToHost));
    return v;
}
static void Fill(Data &d, int seed, int pattern) {
    std::vector<__nv_bfloat16> v(d.Count(0));
    unsigned state = seed;
    for (size_t i = 0; i < v.size(); ++i) {
        state = 1664525u * state + 1013904223u;
        float x = ((int)(state >> 16) - 32768) / 1024.f;
        if (pattern)
            x = std::ldexp(x, int(i % 19) - 9);
        v[i] = __float2bfloat16(x);
    }
    Cuda(cudaMemcpy(d.cudaData, v.data(), d.GetBytes(), cudaMemcpyHostToDevice));
}
static void Linear(int rows, int m, int k, bool bias, int pattern) {
    Data a(BFLOAT16, {rows, m}), b(BFLOAT16, {k, m}), c(BFLOAT16, {rows, k}), r(BFLOAT16, {rows, k}),
        z(BFLOAT16, {k});
    for (Data *d : {&a, &b, &c, &r, &z})
        Allocate(*d);
    Fill(a, 17, pattern);
    Fill(b, 123, pattern);
    Fill(z, 27, 0);
    auto *ap = (__nv_bfloat16 *)a.cudaData, *bp = (__nv_bfloat16 *)b.cudaData,
         *cp = (__nv_bfloat16 *)c.cudaData;
    auto *rp = (__nv_bfloat16 *)r.cudaData, *zp = bias ? (__nv_bfloat16 *)z.cudaData : nullptr;
    LaunchFastllmGemmBf16Bf16(ap, bp, cp, zp, rows, m, k);
    for (int i = 0; i < rows; ++i)
        LaunchFastllmGemmBf16Bf16(ap + i * m, bp, rp + i * k, zp, 1, m, k);
    Check(Read<uint16_t>(c) == Read<uint16_t>(r), "BF16 multirow differs bitwise from single rows");
    ++checks;
}
static void Router(int rows, int pattern, bool bias, bool norm, float scale) {
    // At least two rows force the original generic SelectExpert reference.
    int n = std::max(rows, 2);
    Data x(FLOAT32, {n, 256}), p(FLOAT32, {n, 256}), b(FLOAT32, {256});
    Data i(INT32, {n, 8}), s(FLOAT32, {n, 8}), ri(INT32, {n, 8}), rs(FLOAT32, {n, 8});
    for (Data *d : {&x, &p, &b, &i, &s, &ri, &rs})
        Allocate(*d);
    std::vector<float> v(n * 256), bv(256);
    unsigned rng = 1327;
    for (size_t j = 0; j < v.size(); ++j) {
        rng = 1664525u * rng + 1013904223u;
        v[j] = ((int)(rng >> 16) - 32768) / 8192.f;
        if (pattern == 1)
            v[j] = 0; // full ties, including the selection boundary
        if (pattern == 2)
            v[j] = int(j % 7) - 3; // partial ties
        if (pattern == 3)
            v[j] = (j % 3 == 0 ? 1000.f : -1000.f); // saturated sigmoid
        if (pattern == 4 && j % 5 == 0)
            v[j] = std::numeric_limits<float>::quiet_NaN();
        if (pattern == 5)
            v[j] = j % 2 ? INFINITY : -INFINITY;
        if (pattern == 6)
            v[j] = -1000; // zero normalization denominator
    }
    for (int j = 0; j < 256; ++j) {
        bv[j] = pattern == 1 ? 0.f : (int(j % 13) - 6) * .015625f;
        if (pattern == 4 && j % 11 == 0)
            bv[j] = INFINITY;
    }
    Cuda(cudaMemcpy(x.cudaData, v.data(), x.GetBytes(), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(b.cudaData, bv.data(), b.GetBytes(), cudaMemcpyHostToDevice));
    Check(FastllmCudaSigmoid(x, p), "sigmoid reference failed");
    Check(FastllmCudaSelectExpert(p, bias ? &b : nullptr, ri, rs, 8, norm, scale), "router reference failed");
    // The fused kernel uses the requested row count, including the one-row case.
    x.Resize({rows, 256});
    i.Resize({rows, 8});
    s.Resize({rows, 8});
    Check(FastllmCudaFusedSigmoidSelectExpert(x, bias ? &b : nullptr, i, s, 8, norm, scale),
          "fused router failed");
    auto a = Read<int>(i), ref = Read<int>(ri);
    auto sc = Read<unsigned>(s), sr = Read<unsigned>(rs);
    Check(std::equal(a.begin(), a.end(), ref.begin()), "fused router indices differ");
    Check(std::equal(sc.begin(), sc.end(), sr.begin()), "fused router scores differ bitwise");
    p.Resize({rows, 256});
    Check(FastllmCudaSelectExpert(p, bias ? &b : nullptr, i, s, 8, norm, scale), "unfused router failed");
    a = Read<int>(i);
    sc = Read<unsigned>(s);
    Check(std::equal(a.begin(), a.end(), ref.begin()), "unfused router indices differ");
    Check(std::equal(sc.begin(), sc.end(), sr.begin()), "unfused router scores differ bitwise");
    ++checks;
}
int main() {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || !n)
            return 77;
        FastllmCudaSetDevice(0);
        for (int rows = 1; rows <= 8; ++rows) {
            for (int m : {127, 128, 192, 1536, 2048, 4096, 5120})
                for (bool bias : {false, true})
                    for (int pattern : {0, 1})
                        Linear(rows, m, 257, bias, pattern);
            for (int pattern = 0; pattern < 7; ++pattern)
                for (bool bias : {false, true})
                    for (bool norm : {false, true})
                        for (float scale : {1.f, 2.5f})
                            Router(rows, pattern, bias, norm, scale);
        }
        Cuda(cudaDeviceSynchronize());
        printf("VERIFY MULTIROW PASS checks=%d\n", checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL after %d checks: %s\n", checks, e.what());
        return 1;
    }
}
