#include "fastllm-v41-decode.cuh"
// Fused attention mHC: independent FP64 coefficient oracle, exact residual/
// normalized/quantized-input checks, and changing-input graph replay on every GPU.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
extern "C" bool FastllmCudaDeepSeekV41HcPost(const Data &, const Data &, const Data &, const Data &, Data &);
static void Check(bool ok, const char *s) {
    if (!ok)
        throw std::runtime_error(s);
}
static void Cuda(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static float Float(uint16_t x) {
    uint32_t b = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
static uint16_t Bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    b += 0x7fff + ((b >> 16) & 1);
    return b >> 16;
}
static void Alloc(Data &v, DataType type, std::vector<int> dims, int device) {
    v.dataType = type;
    v.UpdateUnitSize();
    v.dataDevice = CUDA;
    v.dataDeviceIds = {device};
    v.Resize(dims);
    v.Allocate(false);
}
template <class T> static void Upload(Data &v, const std::vector<T> &x) {
    Cuda(cudaMemcpyAsync(v.cudaData, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice,
                         cudaStreamPerThread));
}
template <class T> static std::vector<T> Read(const Data &v) {
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<T> x(v.Count(0));
    Cuda(cudaMemcpy(x.data(), v.cudaData, x.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return x;
}
static bool quick = false;
static double worstOracle = 0;
enum class MixGraph { Captured, SingleAttention, Owner, Cluster };
static void Run(int device, int m, int d, int batch = 1, MixGraph mode = MixGraph::Captured) {
    const int n = m / batch, iter = 20;
    const float eps = 1e-6f;
    Data x, r, prevPre, prevPost, prevComb, fn, scale, base, norm, h, pre, post, comb, y, oh, opre, opost,
        ocomb, oy, quantized, referenceQuantized;
    Alloc(x, BFLOAT16, {batch, n, d}, device);
    Alloc(r, BFLOAT16, {batch, n, 4, d}, device);
    Alloc(prevPre, FLOAT32, {batch, n, 4}, device);
    Alloc(prevPost, FLOAT32, {batch, n, 4}, device);
    Alloc(prevComb, FLOAT32, {batch, n, 4, 4}, device);
    Alloc(fn, FLOAT32, {24, 4 * d}, device);
    Alloc(scale, FLOAT32, {3}, device);
    Alloc(base, FLOAT32, {24}, device);
    Alloc(norm, FLOAT32, {d}, device);
    Alloc(referenceQuantized, BFLOAT16, {batch, n, d}, device);
    std::mt19937 rng(51 + d + m);
    std::normal_distribution<float> normal;
    std::vector<uint16_t> xv(m * d), rv(m * 4 * d);
    std::vector<float> pp(m * 4), po(m * 4), co(m * 16), fw(24 * 4 * d), sw{.5f, .7f, .35f}, bw(24), nw(d);
    for (auto &v : xv)
        v = Bf16(normal(rng) * 2);
    for (auto &v : rv)
        v = Bf16(normal(rng) * 2);
    for (auto &v : pp)
        v = .5f + normal(rng) * .1f;
    for (auto &v : po)
        v = 1.f + normal(rng) * .2f;
    for (auto &v : co)
        v = .25f + normal(rng) * .1f;
    for (auto &v : fw)
        v = normal(rng) * .02f;
    for (auto &v : bw)
        v = normal(rng) * .2f;
    for (auto &v : nw)
        v = 1.f + normal(rng) * .1f;
    Upload(x, xv);
    Upload(r, rv);
    Upload(prevPre, pp);
    Upload(prevPost, po);
    Upload(prevComb, co);
    Upload(fn, fw);
    Upload(scale, sw);
    Upload(base, bw);
    Upload(norm, nw);
    const bool tensorBatch = m >= 16;
    void *clusterPlan = nullptr;
    Data clusterScratch;
    if (mode == MixGraph::Cluster) {
        clusterPlan = FastllmCudaV41CreateOwnerStream(m);
        Check(clusterPlan != nullptr, "cluster owner stream");
        Alloc(clusterScratch, FLOAT32, {m, 8, 25}, device);
        Alloc(quantized, BFLOAT16, {batch, n, d}, device);
        Alloc(h, BFLOAT16, {batch, n, 4, d}, device);
        Alloc(y, BFLOAT16, {batch, n, d}, device);
        Alloc(pre, FLOAT32, {batch, n, 4}, device);
        Alloc(post, FLOAT32, {batch, n, 4}, device);
        Alloc(comb, FLOAT32, {batch, n, 4, 4}, device);
        Check(!FastllmCudaV41OwnerClusterPost(clusterPlan, m + 1, static_cast<float *>(clusterScratch.cudaData),
            x, r, prevPre, prevPost, prevComb, fn, scale, base, norm, iter, eps, eps, h, pre, post, comb, y),
            "owner stream accepted a different row count");
    }
    auto call = [&] {
        if (clusterPlan) {
            auto done = FastllmCudaV41OwnerClusterPost(clusterPlan, m, static_cast<float *>(clusterScratch.cudaData),
                x, r, prevPre, prevPost, prevComb, fn, scale, base, norm, iter, eps, eps, h, pre, post, comb, y);
            Check(done != nullptr, "cluster post + norm rejected");
            Cuda(cudaStreamWaitEvent(cudaStreamPerThread, static_cast<cudaEvent_t>(done)));
            Check(FastllmCudaDeepSeekV41QuantizeActivation(y, quantized), "cluster output quantization");
            return;
        }
        if (tensorBatch || mode != MixGraph::Captured) {
            Check(FastllmCudaDeepSeekV41HcPost(x, r, prevPost, prevComb, h), "batched post");
            Check(FastllmCudaDeepSeekV41HcMixPreNorm(h, prevPre, fn, scale, base, norm,
                iter, eps, eps, pre, post, comb, y, &quantized), "batched mHC rejected");
            return;
        }
        Check(FastllmCudaDeepSeekV41HcPostMixPreNormQuantized(x, r, prevPre, prevPost, prevComb, fn, scale, base, norm,
                                                             iter, eps, eps, h, pre, post, comb, y, quantized),
              "fused rejected");
    };
    auto old = [&] {
        Check(FastllmCudaDeepSeekV41HcPost(x, r, prevPost, prevComb, oh), "old post");
        if (tensorBatch) {
            setenv("FASTLLM_DSV41_DISABLE_HC_TENSOR", "1", 1);
            Check(FastllmCudaDeepSeekV41HcMix(oh, fn, scale, base, 4, iter, eps, eps, opre, opost, ocomb),
                  "batched reference coefficients");
            unsetenv("FASTLLM_DSV41_DISABLE_HC_TENSOR");
            Check(FastllmCudaDeepSeekV41HcPreNorm(oh, prevPre, norm, eps, oy), "batched reference normalization");
            return;
        }
        Check(FastllmCudaDeepSeekV41HcMixPreNorm(oh, prevPre, fn, scale, base, norm, iter, eps, eps, opre,
                                                 opost, ocomb, oy),
              "old pre");
    };
    std::vector<size_t> allocations;
    { FastllmCudaAllocationRecorder record(&allocations); call(); }
    old();
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    Check(FastllmCudaGraphPrepareCaptureDevice(), "prepare capture");
    Check(FastllmCudaReserveGraphWorkspace(allocations), "reserve capture scratch");
    void *graph = nullptr, *exec = nullptr;
    Check(FastllmCudaGraphBeginCapture(), "begin capture");
    call();
    Check(FastllmCudaGraphEndCapture(&graph), "end capture");
    Data splitScratch;
    void *referenceExec = nullptr, *privateGemm = nullptr;
    if (mode == MixGraph::Owner) {
        Alloc(splitScratch, FLOAT32, {m, 8, 25}, device);
        const void *child = graph;
        void *fork = FastllmCudaV41ForkOwnerMix(&child, 1, m, static_cast<float *>(splitScratch.cudaData),
                                               nullptr, nullptr, &privateGemm);
        Check(fork != nullptr, "owner mix graph rewrite");
        if (tensorBatch) Check(privateGemm != nullptr, "owner tensor projection was not separated");
        Check(FastllmCudaGraphInstantiate(graph, &referenceExec), "instantiate original graph");
        FastllmCudaGraphDestroy(graph);
        graph = fork;
    }
    if (mode == MixGraph::SingleAttention) {
        Alloc(splitScratch, FLOAT32, {8 * 25}, device);
        void *clone = FastllmCudaV41CloneSingleAttentionMix(graph, static_cast<float *>(splitScratch.cudaData));
        Check(clone != nullptr, "single-token attention graph rewrite");
        FastllmCudaGraphDestroy(graph); graph = clone;
    }
    Check(FastllmCudaGraphInstantiate(graph, &exec), "instantiate graph");
    for (int replay = 0; replay < 3; replay++) {
        std::rotate(xv.begin(), xv.begin() + 1, xv.end());
        std::rotate(rv.begin(), rv.begin() + 3, rv.end());
        Upload(x, xv);
        Upload(r, rv);
        Check(FastllmCudaGraphLaunch(exec), "replay graph");
        if (referenceExec) {
            auto q = Read<uint16_t>(quantized), normalized = Read<uint16_t>(y);
            auto a = Read<float>(pre), b = Read<float>(post), c = Read<float>(comb);
            Check(FastllmCudaGraphLaunch(referenceExec), "replay original graph");
            Check(a == Read<float>(pre) && b == Read<float>(post) && c == Read<float>(comb) &&
                  q == Read<uint16_t>(quantized) && normalized == Read<uint16_t>(y),
                  "owner scheduling changed graph output");
        }
        if (clusterPlan && tensorBatch) {
            Check(FastllmCudaDeepSeekV41HcPost(x, r, prevPost, prevComb, oh), "cluster reference post");
            Check(FastllmCudaDeepSeekV41HcMixPreNorm(oh, prevPre, fn, scale, base, norm,
                iter, eps, eps, opre, opost, ocomb, oy), "cluster reference tensor projection");
            Check(Read<float>(pre) == Read<float>(opre) && Read<float>(post) == Read<float>(opost) &&
                  Read<float>(comb) == Read<float>(ocomb), "cluster coefficients changed bits");
        }
        old();
        Check(FastllmCudaDeepSeekV41QuantizeActivation(oy, referenceQuantized), "standalone quantizer");
        Check(Read<uint16_t>(quantized) == Read<uint16_t>(referenceQuantized), "quantized output changed bits");
        auto hv = Read<uint16_t>(h);
        Check(hv == Read<uint16_t>(oh), "residual changed bits");
        Check(Read<uint16_t>(y) == Read<uint16_t>(oy), "normalized input changed bits");
        auto pr = Read<float>(pre), ps = Read<float>(post), cb = Read<float>(comb);
        auto opr = Read<float>(opre), ops = Read<float>(opost), ocb = Read<float>(ocomb);
        if (!tensorBatch) Check(std::memcmp(pr.data(), opr.data(), pr.size() * sizeof(float)) == 0 &&
                  std::memcmp(ps.data(), ops.data(), ps.size() * sizeof(float)) == 0 &&
                  std::memcmp(cb.data(), ocb.data(), cb.size() * sizeof(float)) == 0,
              "mHC coefficients changed bits");
        for (size_t i = 0; i < hv.size(); i++) {
            int token = i / (4 * d), target = i / d % 4, col = i % d;
            float sum = 0;
            for (int j = 0; j < 4; j++)
                sum = sum + co[(token * 4 + j) * 4 + target] * Float(rv[(token * 4 + j) * d + col]);
            float scaled = po[token * 4 + target] * Float(xv[token * d + col]);
            Check(hv[i] == Bf16(scaled + sum), "independent post oracle");
        }
        for (int t = 0; t < m; t++) {
            double dots[25] = {};
            for (int j = 0; j < 4 * d; j++) {
                double v = Float(hv[t * 4 * d + j]);
                dots[24] += v * v;
                for (int o = 0; o < 24; o++)
                    dots[o] += v * fw[o * 4 * d + j];
            }
            double inv = 1 / std::sqrt(dots[24] / (4 * d) + eps), refPre[4], refPost[4], refComb[16];
            for (int j = 0; j < 4; j++) {
                refPre[j] = 1 / (1 + std::exp(-(dots[j] * inv * sw[0] + bw[j]))) + eps;
                refPost[j] = 2 / (1 + std::exp(-(dots[4 + j] * inv * sw[1] + bw[4 + j])));
            }
            for (int i = 0; i < 16; i++)
                refComb[i] = dots[8 + i] * inv * sw[2] + bw[8 + i];
            for (int row = 0; row < 4; row++) {
                double maximum = *std::max_element(refComb + row * 4, refComb + row * 4 + 4), sum = 0;
                for (int col = 0; col < 4; col++) {
                    refComb[row * 4 + col] = std::exp(refComb[row * 4 + col] - maximum);
                    sum += refComb[row * 4 + col];
                }
                for (int col = 0; col < 4; col++)
                    refComb[row * 4 + col] = refComb[row * 4 + col] / sum + eps;
            }
            for (int col = 0; col < 4; col++) {
                double sum = 0;
                for (int row = 0; row < 4; row++)
                    sum += refComb[row * 4 + col];
                for (int row = 0; row < 4; row++)
                    refComb[row * 4 + col] /= sum + eps;
            }
            for (int it = 1; it < iter; it++) {
                for (int row = 0; row < 4; row++) {
                    double sum = 0;
                    for (int col = 0; col < 4; col++)
                        sum += refComb[row * 4 + col];
                    for (int col = 0; col < 4; col++)
                        refComb[row * 4 + col] /= sum + eps;
                }
                for (int col = 0; col < 4; col++) {
                    double sum = 0;
                    for (int row = 0; row < 4; row++)
                        sum += refComb[row * 4 + col];
                    for (int row = 0; row < 4; row++)
                        refComb[row * 4 + col] /= sum + eps;
                }
            }
            auto check = [&](float a, double ref) {
                Check(std::isfinite(a), "nonfinite coefficient");
                double err = std::abs(a - ref);
                worstOracle = std::max(worstOracle, err);
                Check(err < 1e-5, "FP64 coefficient oracle error");
            };
            for (int j = 0; j < 4; j++) {
                check(pr[t * 4 + j], refPre[j]);
                check(ps[t * 4 + j], refPost[j]);
            }
            for (int j = 0; j < 16; j++)
                check(cb[t * 16 + j], refComb[j]);
        }
    }
    FastllmCudaGraphExecDestroy(exec);
    if (referenceExec) FastllmCudaGraphExecDestroy(referenceExec);
    FastllmCudaV41DestroyOwnerMixGemm(privateGemm);
    FastllmCudaV41DestroyOwnerStream(clusterPlan);
    FastllmCudaGraphDestroy(graph);
    Check(!FastllmCudaDeepSeekV41HcPostMixPreNorm(x, r, prevPre, prevPost, prevComb, fn, scale, base, norm,
                                                  iter, eps, eps, r, pre, post, comb, y),
          "aliased input output accepted");
    Check(!FastllmCudaDeepSeekV41HcPostMixPreNorm(x, r, prevPre, prevPost, prevComb, fn, scale, base, norm,
                                                  iter, eps, eps, h, pre, pre, comb, y),
          "aliased outputs accepted");
    Check(!FastllmCudaDeepSeekV41HcPostMixPreNormQuantized(x, r, prevPre, prevPost, prevComb, fn, scale, base, norm,
          iter, eps, eps, h, pre, post, comb, y, y), "aliased quantized output accepted");
    if (m > 1) {
        auto dims = r.dims;
        r.dims = {1, 1, 4, d};
        Check(!FastllmCudaDeepSeekV41HcPostMixPreNorm(x, r, prevPre, prevPost, prevComb, fn, scale, base, norm,
                                                      iter, eps, eps, h, pre, post, comb, y),
              "inconsistent single-row metadata accepted");
        r.dims = dims;
    }
    auto ids = prevPre.dataDeviceIds;
    prevPre.dataDeviceIds = {device + 1};
    Check(!FastllmCudaDeepSeekV41HcPostMixPreNorm(x, r, prevPre, prevPost, prevComb, fn, scale, base, norm,
                                                  iter, eps, eps, h, pre, post, comb, y),
          "wrong device accepted");
    prevPre.dataDeviceIds = ids;
}
int main(int argc, char **) {
    quick = argc > 1;
    fastllm::SetCudaGraph(true); // Prepare the same cuBLAS scratch as the graph-serving entry point.
    try {
        const char *policy = "FASTLLM_TEST_HC_ROWS";
        setenv(policy, "small", 1);
        Check(FastllmV41HcEnabledAtRows(policy, 1) && FastllmV41HcEnabledAtRows(policy, 8) &&
              !FastllmV41HcEnabledAtRows(policy, 0) && !FastllmV41HcEnabledAtRows(policy, 9), "small owner policy");
        setenv(policy, "tensor", 1);
        Check(!FastllmV41HcEnabledAtRows(policy, 15) && FastllmV41HcEnabledAtRows(policy, 16), "tensor owner policy");
        unsetenv(policy);
        int devices;
        Cuda(cudaGetDeviceCount(&devices));
        if (!devices)
            return 77;
        int cases = 0;
        for (int dev = 0; dev < (quick ? 1 : devices); dev++) {
            Cuda(cudaSetDevice(dev));
            cudaDeviceProp prop{};
            Cuda(cudaGetDeviceProperties(&prop, dev));
            if (prop.major == 10 && prop.minor == 3) {
                Run(dev, 1, 5120);
                Run(dev, 1, 5120, 1, MixGraph::SingleAttention);
                cases += 2;
                if (dev == 0) {
                    Check(!FastllmCudaV41CreateOwnerStream(9), "unsupported owner row count accepted");
                    setenv("FASTLLM_DSV41_DISABLE_HC_TENSOR", "1", 1);
                    Check(!FastllmCudaV41CreateOwnerStream(16), "disabled tensor path accepted");
                    unsetenv("FASTLLM_DSV41_DISABLE_HC_TENSOR");
                    setenv("FASTLLM_DSV41_HC_OVERLAP", "1", 1);
                    setenv("FASTLLM_DSV41_HC_TENSOR_OVERLAP", "1", 1);
                    for (int rows : {1, 3, 16, 17, 128}) {
                        Run(dev, rows, 5120, rows, MixGraph::Owner);
                        Run(dev, rows, 5120, rows, MixGraph::Cluster);
                        cases += 2;
                    }
                    unsetenv("FASTLLM_DSV41_HC_TENSOR_OVERLAP");
                    unsetenv("FASTLLM_DSV41_HC_OVERLAP");
                }
                if (dev == 0) for (int rows : {16, 17, 128}) {
                    Run(dev, rows, 5120, rows);
                    cases++;
                }
            }
            if (dev == 0 && !quick) {
                for (int d : {4096, 5120, 6144, 7168, 8192})
                    for (int m : {2, 3, 4, 5, 6, 7, 8}) {
                        Run(dev, m, d);
                        cases++;
                    }
                Run(dev, 6, 5120, 2);
                cases++;
            } else
                for (int m : {2, 6, 8}) {
                    Run(dev, m, 5120);
                    cases++;
                }
        }
        std::cout << "PASS " << cases
                  << " cases x3 changing-input graph replay; residual/input/quantized exact; FP64 coefficient max_abs="
                  << worstOracle << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
