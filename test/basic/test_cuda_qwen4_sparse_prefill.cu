#include "devices/cuda/fastllm-cuda.cuh"
#include "fastllm.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace fastllm {
void DoCudaAttention(Data &, Data &, Data &, Data &, Data &, int, float, int);
}
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *s) {
    if (!ok)
        throw std::runtime_error(s);
}
static void C(cudaError_t e) { Check(e == cudaSuccess, cudaGetErrorString(e)); }
static void GPU(Data &d) {
    d.dataDevice = CUDA;
    d.dataDeviceIds = {0};
    d.Allocate(false);
}
static void Fill(Data &d, int seed, float scale = 1) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> r(-scale, scale);
    std::vector<half> x(d.expansionBytes / 2);
    for (auto &v : x)
        v = __float2half_rn(r(rng));
    C(cudaMemcpy(d.cudaData, x.data(), d.expansionBytes, cudaMemcpyHostToDevice));
}
template <class F> static float Time(F f, int n) {
    for (int i = 0; i < 2; ++i)
        f();
    C(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    C(cudaEventCreate(&a));
    C(cudaEventCreate(&b));
    C(cudaEventRecord(a, cudaStreamPerThread));
    for (int i = 0; i < n; ++i)
        f();
    C(cudaEventRecord(b, cudaStreamPerThread));
    C(cudaEventSynchronize(b));
    float ms;
    C(cudaEventElapsedTime(&ms, a, b));
    C(cudaEventDestroy(a));
    C(cudaEventDestroy(b));
    return ms / n;
}
static void Sparse(int rows, int kh, int group, int dim, int width, int length, int rounds) {
    int qh = kh * group;
    Data q(FLOAT16, {qh, rows, dim}), k(FLOAT16), v(FLOAT16), ids(INT32, {rows, width});
    GPU(q);
    GPU(ids);
    k.Expansion({kh, length + 17, dim});
    k.Resize({kh, length, dim});
    k.ToDevice(CUDA, {0}, true);
    v.Expansion({kh, length + 31, dim});
    v.Resize({kh, length, dim});
    v.ToDevice(CUDA, {0}, true);
    Fill(q, 71);
    Fill(k, 73);
    Fill(v, 79);
    std::vector<int> ix(rows * width);
    std::mt19937 rng(83);
    for (int row = 0; row < rows; ++row)
        for (int t = 0; t < width; ++t) {
            ix[row * width + t] = rng() % length;
            if (t % 17 == 0)
                ix[row * width + t] = -1;
            if (t % 31 == 0)
                ix[row * width + t] = length;
        }
    C(cudaMemcpy(ids.cudaData, ix.data(), ix.size() * 4, cudaMemcpyHostToDevice));
    Data a(FLOAT16, q.dims), b(FLOAT16, q.dims);
    GPU(a);
    GPU(b);
    float scale = 1 / std::sqrt(float(dim));
    auto old = [&] {
        for (int start = 0; start < rows; start += 128) {
            int n = std::min(128, rows - start);
            Data pq(FLOAT16, {n * qh, 1, dim}), ck(FLOAT16, {n * kh, width, dim}),
                cv(FLOAT16, ck.dims), mask(FLOAT16, {n, 1, width}), o(FLOAT16, pq.dims);
            for (auto *p : {&pq, &ck, &cv, &mask, &o})
                GPU(*p);
            Check(FastllmCudaQwen4PrepareSparseBatch(q, k, v, ids, pq, ck, cv, mask, start, n),
                  "gather");
            DoCudaAttention(pq, ck, cv, mask, o, group, scale, 1);
            Check(FastllmCudaQwen4UnpackSparseBatch(o, a, start, n), "unpack");
        }
    };
    auto now = [&] {
        Check(FastllmCudaQwen4SparsePrefill(q, k, v, ids, b, group, scale), "new sparse rejected");
    };
    old();
    now();
    std::vector<half> av(q.Count(0)), bv(av.size());
    C(cudaMemcpy(av.data(), a.cudaData, av.size() * 2, cudaMemcpyDeviceToHost));
    C(cudaMemcpy(bv.data(), b.cudaData, bv.size() * 2, cudaMemcpyDeviceToHost));
    double err2 = 0, ref2 = 0;
    float max = 0;
    int bad = 0;
    for (size_t i = 0; i < av.size(); ++i) {
        float x = __half2float(av[i]), y = __half2float(bv[i]), e = std::abs(x - y);
        max = std::max(max, e);
        err2 += double(e) * e;
        ref2 += double(x) * x;
        if (!std::isfinite(y) || e > 0.003f + 0.003f * std::abs(x))
            ++bad;
    }
    // Independent high-precision dot products with the framework's FP16
    // score and probability boundaries. Sample output channels, not intermediates.
    std::vector<half> hq(q.Count(0)), hk(k.expansionBytes / 2), hv(v.expansionBytes / 2);
    C(cudaMemcpy(hq.data(), q.cudaData, hq.size() * 2, cudaMemcpyDeviceToHost));
    C(cudaMemcpy(hk.data(), k.cudaData, hk.size() * 2, cudaMemcpyDeviceToHost));
    C(cudaMemcpy(hv.data(), v.cudaData, hv.size() * 2, cudaMemcpyDeviceToHost));
    double cpuOld2 = 0, cpuNew2 = 0;
    float cpuMax = 0;
    const float hs = __half2float(__float2half_rn(scale));
    for (int sample = 0; sample < 32; ++sample) {
        int row = (sample * 97 + 3) % rows, head = (sample * 11 + 1) % qh,
            col = (sample * 17 + 5) % dim;
        std::vector<float> scores(width), p(width);
        float maximum = -INFINITY;
        for (int t = 0; t < width; ++t) {
            int token = ix[row * width + t];
            double dot = 0;
            if (token >= 0 && token < length)
                for (int c = 0; c < dim; ++c)
                    dot += double(__half2float(hq[(head * rows + row) * dim + c])) *
                           __half2float(hk[(head / group) * k.strides[0] + token * dim + c]);
            scores[t] = token >= 0 && token < length
                            ? __half2float(__float2half_rn(float(dot * hs)))
                            : -10000.f;
            maximum = std::max(maximum, scores[t]);
        }
        double denominator = 0;
        for (int t = 0; t < width; ++t) {
            p[t] = expf(scores[t] - maximum);
            denominator += p[t];
        }
        double expected = 0;
        for (int t = 0; t < width; ++t) {
            int token = ix[row * width + t];
            if (token >= 0 && token < length)
                expected += double(__half2float(__float2half_rn(float(p[t] / denominator)))) *
                            __half2float(hv[(head / group) * v.strides[0] + token * dim + col]);
        }
        float reference = __half2float(__float2half_rn(float(expected)));
        size_t i = (head * rows + row) * dim + col;
        float oldError = std::abs(__half2float(av[i]) - reference),
              newError = std::abs(__half2float(bv[i]) - reference);
        cpuOld2 += double(oldError) * oldError;
        cpuNew2 += double(newError) * newError;
        cpuMax = std::max(cpuMax, newError);
        Check(newError < 0.0005f + 0.001f * std::abs(reference),
              "independent sparse CPU oracle mismatch");
    }
    printf("CPU_ORACLE rows=%d heads=%d dim=%d width=%d new_max=%g old_rmse=%g new_rmse=%g PASS\n",
           rows, qh, dim, width, cpuMax, std::sqrt(cpuOld2 / 32), std::sqrt(cpuNew2 / 32));
    // Fragmented K and V page tables differ, including a partial final page.
    const int pageLen = 17, pages = (length + pageLen - 1) / pageLen;
    PagedCacheManager kp, vp;
    Data pk(FLOAT16, k.dims), pv(FLOAT16, v.dims), pagedOut(FLOAT16, q.dims);
    std::vector<int> kpages(pages), vpages(pages);
    for (int i = 0; i < pages; ++i) {
        kpages[i] = pages - 1 - i;
        vpages[i] = (i + 1) % pages;
    }
    for (int which = 0; which < 2; ++which) {
        auto &pool = which ? vp : kp;
        auto &view = which ? pv : pk;
        const auto &table = which ? vpages : kpages;
        const auto &raw = which ? hv : hk;
        const auto &dense = which ? v : k;
        pool.dataType = FLOAT16;
        pool.UpdateUnitSize();
        pool.Resize({pages, pageLen, kh, dim});
        GPU(pool);
        std::vector<half> payload(pool.Count(0));
        for (int token = 0; token < length; ++token)
            for (int head = 0; head < kh; ++head)
                std::copy_n(raw.data() + head * dense.strides[0] + token * dim, dim,
                            payload.data() +
                                ((table[token / pageLen] * pageLen + token % pageLen) * kh + head) *
                                    dim);
        C(cudaMemcpy(pool.cudaData, payload.data(), payload.size() * 2, cudaMemcpyHostToDevice));
        GPU(view);
        C(cudaMemcpy(view.cudaData, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
        view.isPagedKVCache = true;
        view.pagedKVCacheData = &pool;
        view.pageLen = pageLen;
    }
    GPU(pagedOut);
    Check(FastllmCudaQwen4SparsePrefill(q, pk, pv, ids, pagedOut, group, scale),
          "paged sparse rejected");
    std::vector<half> pagedValues(bv.size());
    C(cudaMemcpy(pagedValues.data(), pagedOut.cudaData, pagedValues.size() * 2,
                 cudaMemcpyDeviceToHost));
    pk.isPagedKVCache = false;
    pv.isPagedKVCache = false;
    Check(std::memcmp(pagedValues.data(), bv.data(), bv.size() * 2) == 0,
          "paged sparse differs from dense");
    printf("PAGED rows=%d kh=%d group=%d dim=%d PASS\n", rows, kh, group, dim);
    // Reject unsupported alignment before allocating scratch or output.
    Data untouched(FLOAT16, q.dims);
    untouched.dataDevice = CUDA;
    untouched.dataDeviceIds = {0};
    void *savedKey = k.cudaData;
    k.cudaData = static_cast<char *>(savedKey) + sizeof(half);
    const bool unaligned = FastllmCudaQwen4SparsePrefill(q, k, v, ids, untouched, group, scale);
    k.cudaData = savedKey;
    Check(!unaligned && !untouched.cudaData, "unaligned fallback allocated output");
    // Unsupported decode size/type must leave the old dispatcher available.
    auto savedDims = q.dims;
    q.dims[1] = 1;
    Check(!FastllmCudaQwen4SparsePrefill(q, k, v, ids, pagedOut, group, scale),
          "decode fallback missing");
    q.dims = savedDims;
    q.dataType = FLOAT32;
    Check(!FastllmCudaQwen4SparsePrefill(q, k, v, ids, pagedOut, group, scale),
          "type fallback missing");
    q.dataType = FLOAT16;
    Check(FastllmCudaGraphPrepareCaptureDevice(), "graph prepare");
    Check(FastllmCudaGraphBeginCapture(), "graph begin");
    bool dispatched = FastllmCudaQwen4SparsePrefill(q, k, v, ids, pagedOut, group, scale);
    void *graph = nullptr;
    bool ended = FastllmCudaGraphEndCapture(&graph);
    Check(!dispatched && ended, "graph fallback missing");
    FastllmCudaGraphDestroy(graph);
    double nrms = std::sqrt(err2 / std::max(ref2, 1e-30));
    printf("SPARSE rows=%d kh=%d group=%d dim=%d width=%d max=%g nrms=%g bad=%d", rows, kh, group,
           dim, width, max, nrms, bad);
    fflush(stdout);
    Check(!bad && nrms < 0.003, "sparse numerical mismatch");
    float before = Time(old, rounds), after = Time(now, rounds);
    printf(" before_ms=%.5f after_ms=%.5f speedup=%.3f PASS\n", before, after, before / after);
    fflush(stdout);
}
__global__ void LegacyMix(const float *x, const half *l, float *y, unsigned long long count,
                          int channels) {
    int g = threadIdx.x % 4;
    unsigned long long i = (unsigned long long)blockIdx.x * (blockDim.x / 4) + threadIdx.x / 4;
    float xx = 0, gate = 0;
    if (i < count) {
        auto at = (i / channels) * 4 * channels + g * channels + i % channels;
        xx = x[at];
        gate = 1.0 / (1.0 + expf(-__half2float(l[at])));
    }
    unsigned mask = __activemask();
    float x0 = __shfl_sync(mask, xx, 0, 4), x1 = __shfl_sync(mask, xx, 1, 4),
          x2 = __shfl_sync(mask, xx, 2, 4), x3 = __shfl_sync(mask, xx, 3, 4);
    float g0 = __shfl_sync(mask, gate, 0, 4), g1 = __shfl_sync(mask, gate, 1, 4),
          g2 = __shfl_sync(mask, gate, 2, 4), g3 = __shfl_sync(mask, gate, 3, 4);
    if (g == 0 && i < count) {
        float s = __fmul_rn(x0, g0);
        s = __fmaf_rn(x1, g1, s);
        s = __fmaf_rn(x2, g2, s);
        s = __fmaf_rn(x3, g3, s);
        y[i] = s / 4;
    }
}
static void Mix(int rows, int channels) {
    Data x(FLOAT32, {rows, 4 * channels}), l(FLOAT16, x.dims), a(FLOAT32, {rows, channels}),
        b(FLOAT32, a.dims);
    for (auto *d : {&x, &l, &a, &b})
        GPU(*d);
    std::vector<float> host(x.Count(0));
    for (size_t i = 0; i < host.size(); ++i)
        host[i] = sin(i * 1.13) * 4;
    C(cudaMemcpy(x.cudaData, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    Fill(l, 89, 12);
    if (rows == 65536 && channels == 1) {
        std::vector<unsigned short> bits(l.Count(0));
        for (size_t i = 0; i < bits.size(); ++i)
            bits[i] = i % 65536;
        C(cudaMemcpy(l.cudaData, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice));
    }
    auto old = [&] {
        LegacyMix<<<(a.Count(0) + 63) / 64, 256, 0, cudaStreamPerThread>>>(
            (float *)x.cudaData, (half *)l.cudaData, (float *)a.cudaData, a.Count(0), channels);
    };
    auto now = [&] {
        Check(FastllmCudaQwen4HyperMixPromotedFloatLogits(x, l, b, 4), "mix rejected");
    };
    old();
    now();
    std::vector<float> av(a.Count(0)), bv(av.size());
    C(cudaMemcpy(av.data(), a.cudaData, av.size() * 4, cudaMemcpyDeviceToHost));
    C(cudaMemcpy(bv.data(), b.cudaData, bv.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    float max = 0;
    for (size_t i = 0; i < av.size(); ++i) {
        bad += std::memcmp(&av[i], &bv[i], 4) != 0 && !(std::isnan(av[i]) && std::isnan(bv[i]));
        max = std::max(max, std::abs(av[i] - bv[i]));
    }
    printf("MIX rows=%d channels=%d bad=%d max=%g", rows, channels, bad, max);
    fflush(stdout);
    Check(!bad, "mix bitwise mismatch");
    float before = Time(old, 50), after = Time(now, 50);
    printf(" before_ms=%.5f after_ms=%.5f speedup=%.3f PASS\n", before, after, before / after);
}
int main() {
#if defined(USE_ROCM) || defined(CUDA_NO_TENSOR_CORE)
    puts("FASTLLM_TEST_SKIP_NO_DEVICE: indexed sparse prefill needs CUDA tensor cores");
    return 0;
#else
    try {
        int devices = 0;
        cudaDeviceProp properties{};
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0 ||
            cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
            properties.major * 10 + properties.minor < 75) {
            puts("FASTLLM_TEST_SKIP_NO_DEVICE: sparse prefill requires SM75+");
            return 0;
        }
        FastllmCudaSetDevice(0);
        SetThreads(2);
        for (int n : {1, 8, 128, 2048})
            for (int c : {513, 2560})
                Mix(n, c);
        Mix(65536, 1);
        Sparse(32, 1, 1, 16, 1, 37, 3);
        Sparse(33, 2, 7, 48, 129, 171, 3);
        Sparse(65, 2, 24, 128, 257, 1024, 3);
        Sparse(128, 1, 16, 256, 2064, 16384, 5);
        Sparse(2048, 1, 16, 256, 2064, 16384, 3);
        Sparse(2048, 1, 32, 256, 2064, 16384, 3);
        Sparse(33, 1, 17, 80, 3071, 4096, 3);
        puts("PASS Qwen4 indexed sparse prefill and HyperMix");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
#endif
}
