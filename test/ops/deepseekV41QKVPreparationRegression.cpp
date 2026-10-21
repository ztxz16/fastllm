#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void Fill(Data &out, const std::vector<int> &dims, DataType type, int seed, float amplitude) {
    size_t count = 1;
    for (int d : dims) count *= d;
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i)
        values[i] = amplitude * std::sin(float((i * 37 + seed * 19) % 1031));
    out.CopyFrom(Data(DataType::FLOAT32, dims, values));
    ToDataType(out, type);
    out.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
static std::vector<unsigned char> Bits(const Data &d) {
    std::vector<unsigned char> result(d.GetBytes());
    Check(cudaMemcpy(result.data(), d.cudaData, result.size(), cudaMemcpyDeviceToHost));
    return result;
}
struct BorrowPointer {
    Data &data;
    void *saved;
    BorrowPointer(Data &data, void *pointer) : data(data), saved(data.cudaData) { data.cudaData = pointer; }
    ~BorrowPointer() { data.cudaData = saved; }
};
static void Weight(Data &w, int inner, int rows, int bm, int bk, int seed) {
    w.dataType = DataType::FP8_E4M3;
    w.UpdateUnitSize();
    w.Resize({rows, inner});
    w.blockM = bm;
    w.blockK = bk;
    w.Allocate();
    for (size_t i = 0; i < w.GetBytes(); ++i)
        w.cpuData[i] = ((i + seed) % 2 ? 128 : 0) | ((i * 31 + i / inner * 7 + seed) % 105);
    w.scales.resize(size_t((rows - 1) / bk + 1) * ((inner - 1) / bm + 1));
    for (size_t i = 0; i < w.scales.size(); ++i)
        w.scales[i] = std::ldexp(1.0f, int((i * 13 + seed) % 19) - 16);
    w.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
static void Quant(const Data &in, Data &out) {
    Fill(out, in.dims, DataType::BFLOAT16, 0, 0);
    Require(FastllmCudaDeepSeekV41QuantizeActivation(in, out), "reference quant rejected");
}
static void Replay(const std::function<void()> &fn) {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    fn();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 4; ++i) Check(cudaGraphLaunch(executable, cudaStreamPerThread));
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    Check(cudaGraphExecDestroy(executable));
    Check(cudaGraphDestroy(graph));
}
static int NormCases() {
    int cases = 0;
    for (int qdim : {512, 544, 1280, 2016, 2048, 2080, 3072, 4064})
    for (int kdim : {512, 1024, 2048, 2080})
    for (int batch : {1, 2, 8}) for (float eps : {0.0f, 1e-6f, 1e-3f}) for (float amplitude : {0.0f, 0.8f, 256.0f}) {
        Data q, kv, qw, kw, qref, kref, quantRef, qnorm, quant, knorm;
        Fill(q, {1, batch, qdim}, DataType::BFLOAT16, 1, amplitude);
        Fill(kv, {1, batch, kdim}, DataType::BFLOAT16, 2, amplitude);
        Fill(qw, {qdim}, DataType::FLOAT32, 3, 1.3f);
        Fill(kw, {kdim}, DataType::FLOAT32, 4, 0.7f);
        RMSNorm(q, qw, eps, qref);
        RMSNorm(kv, kw, eps, kref);
        Quant(qref, quantRef);
        auto fused = [&] { Require(FastllmCudaV41QKVNormQuant(q, kv, qw, kw, eps, qnorm, quant, knorm),
                                   "fused norm rejected"); };
        fused();
        Require(Bits(qref) == Bits(qnorm) && Bits(kref) == Bits(knorm) && Bits(quantRef) == Bits(quant),
                "norm/quant changed BF16 rounding or block-32 scale");
        // Exercise both sides of the register-cache boundary, including KV
        // alone selecting the uncached path and in-place KV normalization.
        const bool replayShape = (qdim == 1280 && (kdim == 512 || kdim == 2048 || kdim == 2080)) ||
            ((qdim == 2048 || qdim == 2080) && kdim == 512);
        if (replayShape && (eps == 0.0f || eps == 1e-6f)) {
            Replay(fused);
            Require(Bits(qref) == Bits(qnorm) && Bits(kref) == Bits(knorm) && Bits(quantRef) == Bits(quant),
                    "norm graph replay changed output");
            Require(FastllmCudaV41QKVNormQuant(q, kv, qw, kw, eps, qnorm, quant, kv), "in-place KV rejected");
            Require(Bits(kref) == Bits(kv), "in-place KV changed output");
        }
        ++cases;
    }
    Data q, kv, qw, kw, qn, qq, kn;
    Fill(q, {1, 1280}, DataType::BFLOAT16, 1, 1);
    Fill(kv, {1, 512}, DataType::BFLOAT16, 2, 1);
    Fill(qw, {1280}, DataType::FLOAT32, 3, 1);
    Fill(kw, {512}, DataType::FLOAT32, 4, 1);
    Fill(qn, {1}, DataType::BFLOAT16, 5, 1);
    Fill(qq, {1}, DataType::BFLOAT16, 6, 1);
    Fill(kn, {1}, DataType::BFLOAT16, 7, 1);
    const auto before = Bits(qn), qb = Bits(q);
    Require(!FastllmCudaV41QKVNormQuant(q, kv, qw, kw, 1e-6f, qn, qn, kn) && Bits(qn) == before,
            "aliased outputs must reject without writes");
    Require(!FastllmCudaV41QKVNormQuant(q, kv, qw, kw, 1e-6f, q, qq, kn) && Bits(q) == qb,
            "aliased Q input must reject without writes");
    {
        BorrowPointer alias(qn, (unsigned char *)q.cudaData + 4);
        Require(!FastllmCudaV41QKVNormQuant(q, kv, qw, kw, 1e-6f, qn, qq, kn),
                "partially overlapping norm input/output must reject");
    }
    Require(Bits(q) == qb && Bits(qn) == before, "norm overlap check changed memory");
    for (float eps : {-1e-6f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        Require(!FastllmCudaV41QKVNormQuant(q, kv, qw, kw, eps, qn, qq, kn) && Bits(qn) == before,
                "invalid epsilon must reject without writes");
        ++cases;
    }
    for (int bad : {511, 513, 4096}) {
        Fill(q, {1, bad}, DataType::BFLOAT16, 1, 1);
        Require(!FastllmCudaV41QKVNormQuant(q, kv, qw, kw, 1e-6f, qn, qq, kn) && Bits(qn) == before,
                "unsupported norm shape changed output");
        ++cases;
    }
    return cases + 2;
}
static int ProjectionCases() {
    int cases = 0;
    for (auto shape : {std::vector<int>{5120, 1280, 512}, {1024, 513, 33}, {516, 65, 129}, {256, 255, 257}})
    for (int block : {4, 32, 64}) {
        const int inner = shape[0], n0 = shape[1], n1 = shape[2];
        Data input, w0, w1, ref0, ref1, out0, out1;
        Fill(input, {1, 1, inner}, DataType::BFLOAT16, 1, 0.8f);
        Weight(w0, inner, n0, block, 32, 2);
        Weight(w1, inner, n1, 32, block, 3);
        Fill(ref0, {1, 1, n0}, DataType::BFLOAT16, 0, 0);
        Fill(ref1, {1, 1, n1}, DataType::BFLOAT16, 0, 0);
        Fill(out0, ref0.dims, DataType::BFLOAT16, 0, 0);
        Fill(out1, ref1.dims, DataType::BFLOAT16, 0, 0);
        Require(FastllmCudaBFloat16MatMulFP8E4M3(input, w0, Data(), ref0, 1, inner, n0), "reference projection 0");
        Require(FastllmCudaBFloat16MatMulFP8E4M3(input, w1, Data(), ref1, 1, inner, n1), "reference projection 1");
        auto fused = [&] { Require(FastllmCudaBFloat16FP8LinearPair(input, w0, w1, out0, out1), "pair rejected"); };
        fused();
        Require(Bits(ref0) == Bits(out0) && Bits(ref1) == Bits(out1), "paired projection changed bits");
        Replay(fused);
        Require(Bits(ref0) == Bits(out0) && Bits(ref1) == Bits(out1), "paired projection graph changed bits");
        const auto before = Bits(out0);
        const auto inputBefore = Bits(input);
        {
            BorrowPointer alias(out0, (unsigned char *)input.cudaData + 8);
            Require(!FastllmCudaBFloat16FP8LinearPair(input, w0, w1, out0, out1),
                    "partially overlapping projection input/output must reject");
        }
        Require(Bits(input) == inputBefore && Bits(out0) == before, "projection overlap changed memory");
        for (int batch : {2, 8}) {
            Fill(input, {1, batch, inner}, DataType::BFLOAT16, 1, 0.8f);
            Require(!FastllmCudaBFloat16FP8LinearPair(input, w0, w1, out0, out1) && Bits(out0) == before,
                    "unsupported batch must fall back without writes");
        }
        ++cases;
    }
    return cases;
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
        FastllmCudaSetDevice(0);
        const int cases = NormCases() + ProjectionCases();
        std::printf("PASS V4.1 QKV preparation: %d exact cases plus graph/fallback checks\n", cases);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
