#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda-fp8-linear-add.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace fastllm;
// The alternate mode verifies fallback when this TU is linked to an unavailable kernel image.
static bool expectFused = true;
void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void Require(bool b, const char *s) {
    if (!b)
        throw std::runtime_error(s);
}
void Allocate(Data &d) {
    d.dataDevice = DataDevice::CUDA;
    d.dataDeviceIds = {0};
    d.Allocate();
}
template <class T> void Upload(Data &d, const std::vector<T> &v) {
    Check(cudaMemcpy(d.cudaData, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> std::vector<T> Download(Data &d) {
    std::vector<T> v(d.Count(0));
    Check(cudaMemcpy(v.data(), d.cudaData, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
template <class T> void Run(DataType type, int K, bool hasBias, int N = 5120, int scaleBlock = 0) {
    Data w(DataType::FP8_E4M3, {N, K}), x(type, {1, 1, K}), o(type, {1, 1, N}), middle(type),
        bias(DataType::FLOAT32);
    if (hasBias)
        bias.Resize({N});
    w.blockK = 1;
    w.blockM = scaleBlock ? scaleBlock : K;
    w.scales.resize(N);
    std::vector<uint8_t> codes(size_t(N) * K);
    std::vector<float> table(256), hb(N);
    std::vector<T> hx(K), res(N);
    for (int i = 0; i < 256; ++i) {
        __nv_fp8_e4m3 f;
        f.__x = i;
        table[i] = float(f);
    }
    for (int r = 0; r < N; ++r) {
        w.scales[r] = .0001f * (1 + r % 9);
        hb[r] = .01f * std::sin(r * .1f);
        for (int c = 0; c < K; ++c) {
            int v = (r * 31 + c * 13 + c / 17) % 254;
            codes[size_t(r) * K + c] = v < 127 ? v : v + 1;
        }
    }
    for (int i = 0; i < K; ++i)
        hx[i] = T(.3f * std::sin(i * .013f));
    for (int i = 0; i < N; ++i)
        res[i] = T(.2f * std::cos(i * .017f));
    Allocate(w);
    Allocate(x);
    Allocate(o);
    if (hasBias)
        Allocate(bias);
    Upload(w, codes);
    Upload(x, hx);
    Upload(o, res);
    if (hasBias)
        Upload(bias, hb);
    setenv("FASTLLM_CUDA_FP8_LINEAR_ADD", "1", 1);
    auto can = [&]() { return FastllmCudaFP8LinearAddCanRun(x, w, bias, o); };
    Require(can() == expectFused, "kernel availability admission wrong");
    Check(cudaGetLastError());
    // A cold scale cache must not allocate or mutate output during graph capture.
    cudaGraph_t coldGraph;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(!can(), "cold scales admitted during capture");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &coldGraph));
    Check(cudaGraphDestroy(coldGraph));
    auto before = Download<T>(o);
    Require(!std::memcmp(before.data(), res.data(), N * sizeof(T)), "CanRun changed residual");
    auto dtype = o.dataType;
    o.dataType = DataType::FLOAT32;
    Require(!can(), "wrong output dtype admitted");
    o.dataType = dtype;
    w.blockK = 2;
    Require(!can(), "non-row scales admitted");
    w.blockK = 1;
    auto scale = w.scales.back();
    w.scales.pop_back();
    Require(!can(), "short scales admitted");
    w.scales.push_back(scale);
    void *aligned = x.cudaData;
    x.cudaData = static_cast<char *>(aligned) + 2;
    Require(!can(), "misaligned input admitted");
    x.cudaData = aligned;
    w.blockM = 128;
    Require(!can(), "block scales admitted");
    w.blockM = scaleBlock ? scaleBlock : K;
    w.IsRepacked = true;
    Require(!can(), "repacked admitted");
    w.IsRepacked = false;
    x.strides.back() = 2;
    Require(!can(), "strided admitted");
    x.strides.back() = 1;
    void *ptr = o.cudaData;
    o.cudaData = x.cudaData;
    Require(!can(), "aliased output admitted");
    o.cudaData = ptr;
    x.dataDeviceIds = {1};
    Require(!can(), "wrong device admitted");
    x.dataDeviceIds = {0};
    Executor op;
    auto launch = [&]() {
        op.RunOnDevice("cuda", "LinearAdd",
                       {{"input", &x}, {"weight", &w}, {"bias", &bias}, {"middle", &middle}, {"output", &o}},
                       {}, {});
    };
    launch();
    Check(cudaDeviceSynchronize());
    Upload(o, res);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    launch();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    float worst = 0, baseRms = 0;
    for (int step = 0; step < 3; ++step) {
        for (int i = 0; i < K; ++i)
            hx[i] = T(.3f * std::sin(i * .013f + step * .37f));
        Upload(x, hx);
        Upload(o, res);
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        auto actual = Download<T>(o);
        for (T f : actual)
            Require(std::isfinite(float(f)), "nonfinite");
        for (int r : {0, 1, 7, 15, 16, 31, 127, 128, 511, 1023, 2047, 2048, 4095, N - 2, N - 1}) {
            if (r >= N) continue;
            double sum = 0;
            for (int c = 0; c < K; ++c)
                sum += double(table[codes[size_t(r) * K + c]]) * float(hx[c]);
            float expected =
                float(T(float(res[r]) + float(T(float(sum * w.scales[r]) + (hasBias ? hb[r] : 0.f)))));
            float error = std::abs(float(actual[r]) - expected);
            worst = std::max(worst, error);
            Require(!expectFused ||
                        error <= (type == DataType::FLOAT16 ? .002f : .016f) * (1 + std::abs(expected)),
                    "reference mismatch");
        }
        setenv("FASTLLM_CUDA_FP8_LINEAR_ADD", "0", 1);
        Require(!can(), "disable ignored");
        Upload(o, res);
        launch();
        Check(cudaDeviceSynchronize());
        auto baseline = Download<T>(o);
        if (!expectFused)
            Require(!std::memcmp(actual.data(), baseline.data(), N * sizeof(T)),
                    "unavailable-image fallback differs");
        double sq = 0, ref = 0;
        for (int r = 0; r < N; ++r) {
            double d = float(actual[r]) - float(baseline[r]);
            sq += d * d;
            ref += double(float(baseline[r])) * float(baseline[r]);
        }
        baseRms = std::max(baseRms, float(std::sqrt(sq / ref)));
        Require(baseRms < (type == DataType::FLOAT16 ? .01f : .04f), "baseline difference too large");
        setenv("FASTLLM_CUDA_FP8_LINEAR_ADD", "1", 1);
    }
    Require(Download<uint8_t>(w) == codes, "weight mutated");
    auto unchanged = Download<T>(x);
    Require(!std::memcmp(unchanged.data(), hx.data(), K * sizeof(T)), "input mutated");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    // Unsupported batches must remain in the original complete LinearAdd path.
    Data bx(type, {1, 2, K}), bo(type, {1, 2, N}), bm(type);
    Allocate(bx);
    Allocate(bo);
    std::vector<T> bxx(2 * K, T(.01f)), boo(2 * N, T(.1f));
    Upload(bx, bxx);
    Upload(bo, boo);
    Require(!FastllmCudaFP8LinearAddCanRun(bx, w, bias, bo), "batch two admitted");
    op.RunOnDevice("cuda", "LinearAdd",
                   {{"input", &bx}, {"weight", &w}, {"bias", &bias}, {"middle", &bm}, {"output", &bo}}, {},
                   {});
    Check(cudaDeviceSynchronize());
    for (T v : Download<T>(bo))
        Require(std::isfinite(float(v)), "fallback nonfinite");
    auto fallback = Download<T>(bo);
    setenv("FASTLLM_CUDA_FP8_LINEAR_ADD", "0", 1);
    Upload(bo, boo);
    op.RunOnDevice("cuda", "LinearAdd",
                   {{"input", &bx}, {"weight", &w}, {"bias", &bias}, {"middle", &bm}, {"output", &bo}}, {},
                   {});
    Check(cudaDeviceSynchronize());
    auto disabled = Download<T>(bo);
    Require(!std::memcmp(fallback.data(), disabled.data(), 2 * N * sizeof(T)), "batch fallback differs");
    // K=256 is valid for generic LinearAdd, but below this fusion's minimum reduction width.
    w.Resize({N, 256});
    w.blockM = 256;
    bx.Resize({1, 1, 256});
    bo.Resize({1, 1, N});
    std::vector<T> smallResidual(N, T(.1f));
    setenv("FASTLLM_CUDA_FP8_LINEAR_ADD", "1", 1);
    Require(!FastllmCudaFP8LinearAddCanRun(bx, w, bias, bo), "unsupported K admitted");
    Upload(bo, smallResidual);
    op.RunOnDevice("cuda", "LinearAdd",
                   {{"input", &bx}, {"weight", &w}, {"bias", &bias}, {"middle", &bm}, {"output", &bo}}, {},
                   {});
    Check(cudaDeviceSynchronize());
    fallback = Download<T>(bo);
    setenv("FASTLLM_CUDA_FP8_LINEAR_ADD", "0", 1);
    Upload(bo, smallResidual);
    op.RunOnDevice("cuda", "LinearAdd",
                   {{"input", &bx}, {"weight", &w}, {"bias", &bias}, {"middle", &bm}, {"output", &bo}}, {},
                   {});
    Check(cudaDeviceSynchronize());
    disabled = Download<T>(bo);
    Require(!std::memcmp(fallback.data(), disabled.data(), N * sizeof(T)), "shape fallback differs");
    printf("PASS dtype=%d K=%d bias=%d graph_replays=3 max_abs=%g baseline_relative_rms=%g\n", int(type), K,
           hasBias, worst, baseRms);
    fflush(stdout);
}
int main(int argc, char **argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--expect-fallback"))
        expectFused = false;
    else if (argc != 1)
        return 2;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 77;
    try {
        for (int k : {6144, 17408, 1536, 1920, 2048, 2304, 3072, 4352, 5802, 5803, 8704})
            for (bool bias : {false, true}) {
                Run<half>(DataType::FLOAT16, k, bias);
                Run<__nv_bfloat16>(DataType::BFLOAT16, k, bias);
            }
        Run<half>(DataType::FLOAT16, 5803, false, 4099, 17408);
        Run<__nv_bfloat16>(DataType::BFLOAT16, 5803, false, 4099, 17408);
        puts("ALL PASS");
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
