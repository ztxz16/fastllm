#include "../../src/devices/cuda/linear/fastllm-native-lowbit-prefill.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "fastllm.h"
#include "utils/utils.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp8.h>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool x, const char *s) {
    if (!x)
        throw std::runtime_error(s);
}
struct Memory {
    void *p;
    Memory(size_t n) { Check(cudaMalloc(&p, n) == cudaSuccess, "alloc"); }
    ~Memory() { cudaFree(p); }
};
static void Case(int M, int N, int K, bool bias) {
    unsigned seed = 73;
    auto random = [&]() {
        seed = 1664525u * seed + 1013904223u;
        return (int(seed >> 16) - 32768) / 32768.f;
    };
    std::vector<__nv_bfloat16> x(size_t(M) * K);
    std::vector<uint8_t> w(size_t(N) * (K + 4));
    std::vector<float> xs(M), ws(N), b(N);
    for (auto &v : x)
        v = __float2bfloat16(random());
    for (int r = 0; r < M; ++r) {
        float amax = 0;
        for (int k = 0; k < K; ++k) {
            if (!r)
                x[k] = __float2bfloat16(0);
            amax = fmaxf(amax, fabsf(float(x[size_t(r) * K + k])));
        }
        xs[r] = fmaxf(amax / 448.f, 1.e-12f);
    }
    for (int r = 0; r < N; ++r) {
        ws[r] = .0001f * (1 + r % 11);
        b[r] = random() / 32;
        for (int k = 0; k < K; ++k)
            w[size_t(r) * (K + 4) + k] = __nv_fp8_e4m3(random() * 448).__x;
        memcpy(w.data() + size_t(r) * (K + 4) + K, &ws[r], 4);
    }
    Memory dx(x.size() * 2), dw(w.size()), db(b.size() * 4 + 4), dy(size_t(M) * N * 2);
    float *biasData = reinterpret_cast<float *>(static_cast<char *>(db.p) + 4);
    cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(biasData, b.data(), b.size() * 4, cudaMemcpyHostToDevice);
    auto run = [&](int rows) {
        return fastllm_native_prefill::Fp8BFloat16((__nv_bfloat16 *)dx.p, (uint8_t *)dw.p,
                                                   bias ? biasData : nullptr, (__nv_bfloat16 *)dy.p, rows, N,
                                                   K);
    };
    Check(run(M), "fast path not selected");
    FastllmCudaSyncCurrentThreadStream();
    std::vector<__nv_bfloat16> y(size_t(M) * N);
    Check(cudaMemcpy(y.data(), dy.p, y.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess, "copy");
    // Compare the same tensor with scales interleaved in the checkpoint or
    // stored separately. Small cases cover both the bias and bias-free paths.
    if (M < 512) {
        Memory plain(size_t(N) * K), scales(size_t(N) * sizeof(float));
        Check(cudaMemcpy2D(plain.p, K, w.data(), K + 4, K, N, cudaMemcpyHostToDevice) == cudaSuccess,
              "copy contiguous E4M3 weights");
        Check(cudaMemcpy(scales.p, ws.data(), size_t(N) * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess,
              "copy row scales");
        Check(cudaMemset(dy.p, 0xff, y.size() * sizeof(y[0])) == cudaSuccess, "clear separate-scale output");
        Check(fastllm_native_prefill::Fp8BFloat16((const __nv_bfloat16 *)dx.p, (const uint8_t *)plain.p,
                  bias ? biasData : nullptr, (__nv_bfloat16 *)dy.p, M, N, K, (const float *)scales.p),
              "separate-scale prefill");
        FastllmCudaSyncCurrentThreadStream();
        std::vector<__nv_bfloat16> actual(y.size());
        Check(cudaMemcpy(actual.data(), dy.p, y.size() * sizeof(y[0]), cudaMemcpyDeviceToHost) == cudaSuccess &&
              std::memcmp(actual.data(), y.data(), y.size() * sizeof(y[0])) == 0,
              "packed and separate row scales must agree");
    }
    double err = 0, norm = 0, maxabs = 0;
    int samples = 0;
    for (int r : {0, 1, M / 2, M - 1})
        for (int c = 0; c < N; c += std::max(1, N / 61)) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                __nv_fp8_e4m3 a(float(x[size_t(r) * K + k]) / xs[r]), v;
                v.__x = w[size_t(c) * (K + 4) + k];
                sum += double(float(a)) * float(v);
            }
            float ref = (float(sum) * xs[r]) * ws[c] + (bias ? b[c] : 0.f);
            float expected = float(__float2bfloat16(ref)), got = float(y[size_t(r) * N + c]);
            Check(std::isfinite(got), "nonfinite");
            err += (got - expected) * (got - expected);
            norm += expected * expected;
            maxabs = fmax(maxabs, fabs(got - expected));
            ++samples;
        }
    double nrmse = sqrt(err / std::max(norm, 1.e-30));
    Check(nrmse <= .003, "quantized reference mismatch");
    setenv("FASTLLM_CUDA_NATIVE_FP8_PREFILL", "0", 1);
    Check(!run(M), "disabled compute must fall back");
    setenv("FASTLLM_CUDA_NATIVE_FP8_PREFILL", "1", 1);
    Check(!run(1), "decode must retain weight-only fallback");
    void *graph = nullptr;
    Check(FastllmCudaGraphBeginCapture(), "capture");
    bool captured = run(M);
    Check(FastllmCudaGraphEndCapture(&graph), "end capture");
    if (graph)
        FastllmCudaGraphDestroy(graph);
    Check(!captured, "capture must fall back");
    printf("M=%d N=%d K=%d bias=%d samples=%d nrmse=%.8g max_abs=%.8g\n", M, N, K, bias, samples, nrmse,
           maxabs);
}
static void Fallback(int rows) {
    const int N = 32, K = 256;
    std::vector<float> x(rows * K);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (int(i % 67) - 33) / 32.f;
    Data input(BFLOAT16, {rows, K}, x), weight(FP8_E4M3_PERCHANNEL, {N, K}), out(BFLOAT16, {rows, N}),
        bias(FLOAT32);
    weight.Allocate();
    std::vector<float> expectedWeight(N * K);
    for (int r = 0; r < N; ++r) {
        float scale = std::ldexp(1.f, -20 + r % 12);
        memcpy(weight.cpuData + size_t(r) * (K + 4) + K, &scale, 4);
        for (int k = 0; k < K; ++k) {
            uint8_t v = (k + r) % 127 | ((k & 1) ? 128 : 0);
            weight.cpuData[size_t(r) * (K + 4) + k] = v;
            __nv_fp8_e4m3 fp;
            fp.__x = v;
            expectedWeight[r * K + k] = float(__float2bfloat16(float(fp) * scale));
        }
    }
    input.ToDevice(DataDevice::CUDA, {0}, true);
    weight.ToDevice(DataDevice::CUDA, {0}, true);
    out.ToDevice(DataDevice::CUDA, {0}, false);
    out.Allocate(false);
    Check(FastllmCudaBFloat16MatMulFP8E4M3PerChannel(input, weight, bias, out, rows, K, N), "native fallback");
    FastllmCudaSyncCurrentThreadStream();
    std::vector<__nv_bfloat16> actual(size_t(rows) * N);
    cudaMemcpy(actual.data(), out.cudaData, actual.size() * 2, cudaMemcpyDeviceToHost);
    double err = 0, norm = 0;
    for (int r : {0, rows / 2, rows - 1})
        for (int c = 0; c < N; ++c) {
            double v = 0;
            for (int k = 0; k < K; ++k)
                v += double(float(__float2bfloat16(x[r * K + k]))) * expectedWeight[c * K + k];
            float ref = float(__float2bfloat16(float(v))), got = float(actual[r * N + c]);
            err += double(got - ref) * (got - ref);
            norm += double(ref) * ref;
        }
    Check(std::sqrt(err / std::max(norm, 1.e-30)) < .003, "packed FP8 BF16 underflow/reference");
    printf("PASS fallback rows=%d nrmse=%.9g\n", rows, std::sqrt(err / std::max(norm, 1.e-30)));
}
static void RouterFP32(bool missingArena = false) {
    if (FastllmCudaRuntimeArch() != 90)
        return;
    const int N = 256, K = 4096;
    for (int M : {128, 129, 512, 1024, 4096, 8192}) {
        unsigned seed = 93;
        auto random = [&]() {
            seed = 1664525u * seed + 1013904223u;
            return (int(seed >> 8) - 8388608) / 8388608.f;
        };
        std::vector<float> x(size_t(M) * K), w(size_t(N) * K), reference(size_t(M) * N),
            actual(reference.size());
        for (auto &v : x)
            v = float(__float2bfloat16(random()));
        for (auto &v : w)
            v = random();
        Memory dx(x.size() * 4), dw(w.size() * 4), dy(reference.size() * 4), ref(reference.size() * 4);
        cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw.p, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        auto run = [&](int rows) {
            return fastllm_native_prefill::Fp32((float *)dx.p, (float *)dw.p, (float *)dy.p, rows, N, K);
        };
        if (missingArena) {
            Check(!run(M), "FP32 prefill allocated a missing arena");
            return;
        }
        const float alpha = 1, beta = 0;
        Check(cublasGemmEx(getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dw.p,
                           CUDA_R_32F, K, dx.p, CUDA_R_32F, K, &beta, ref.p, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS,
              "FP32 reference");
        Check(run(M), "FP32 Router path not selected");
        Check(cudaMemcpy(reference.data(), ref.p, reference.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess,
              "FP32 reference read");
        Check(cudaMemcpy(actual.data(), dy.p, actual.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess,
              "FP32 Router read");
        Check(actual == reference, "FP32 Router changed reference reduction results");
        Check(!run(1), "FP32 Router decode fallback");
        void *graph = nullptr;
        Check(FastllmCudaGraphBeginCapture(), "FP32 capture");
        bool captured = run(M);
        Check(FastllmCudaGraphEndCapture(&graph), "FP32 end capture");
        if (graph)
            FastllmCudaGraphDestroy(graph);
        Check(!captured, "FP32 Router capture must fall back");
        printf("PASS FP32 Router rows=%d exact=%zu\n", M, actual.size());
    }
}

static void Quantization() {
    for (auto type : {FLOAT32, BFLOAT16}) {
        const int rows = 128, cols = 513;
        std::vector<float> x(rows * cols);
        std::vector<uint16_t> bf(x.size());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                __nv_fp8_e4m3 low, high;
                low.__x = (c / 3) % 126;
                high.__x = low.__x + 1;
                float v = (float(low) + float(high)) / 2;
                v = std::nextafter(v, c % 3 == 0 ? 0.f : INFINITY);
                if (r & 1)
                    v = -v;
                x[r * cols + c] = c == cols - 1 ? 448.f : v;
                if (!r)
                    x[c] = 0;
                bf[r * cols + c] = Float32ToBFloat16RNEBits(x[r * cols + c]);
                if (type == BFLOAT16)
                    x[r * cols + c] = BFloat16BitsToFloat32(bf[r * cols + c]);
            }
        Data d(FP8_E4M3_PERCHANNEL, {rows, cols});
        d.CreateFromOriData(WeightType::LINEAR, type,
                            type == FLOAT32 ? (uint8_t *)x.data() : (uint8_t *)bf.data(), nullptr, nullptr);
        for (int r = 0; r < rows; ++r) {
            float mx = 0;
            for (int c = 0; c < cols; ++c)
                mx = fmaxf(mx, fabsf(x[r * cols + c]));
            float scale = fmaxf(mx / 448.f, 1.e-12f), stored;
            memcpy(&stored, d.cpuData + size_t(r) * (cols + 4) + cols, 4);
            Check(stored == scale, "weight scale");
            for (int c = 0; c < cols; ++c)
                Check(d.cpuData[size_t(r) * (cols + 4) + c] == __nv_fp8_e4m3(x[r * cols + c] / scale).__x,
                      "weight RNE differs from CUDA");
        }
    }
    puts("PASS CPU weight quantization matches CUDA E4M3 RNE");
}
int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n)
        return 77;
    FastllmCudaSetDevice(0);
    if (!fastllm_native_prefill::Supported(8))
        return 77;
    try {
        Quantization();
        RouterFP32(true);
        setenv("FASTLLM_CUDA_NATIVE_FP8_PREFILL", "1", 1);
        FastllmCudaSetNcclForceSync(true);
        Fallback(1);
        Fallback(17);
        Fallback(4097);
        Case(128, 512, 256, false);
        Case(257, 768, 512, true);
        Case(4096, 13568, 4096, false);
        Case(4096, 14848, 4096, false);
        Case(4096, 4096, 8192, false);
        Case(1024, 4112, 256, true);
        FastllmCudaSetNcclForceSync(false);
        RouterFP32();
        puts("PASS FP8 BF16 prefill reference and fallback");
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
