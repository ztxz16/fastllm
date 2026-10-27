#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>

#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "executor.h"
#include "gguf.h"
#include "models/qwen4_exp.h"

using namespace fastllm;
namespace fastllm {
extern void Float32ToFloat16(float*, uint16_t*, int);
struct Qwen4GGUFTestAccess {
    static void Configure(Qwen4ExpModel& m, int nk, int nv, int hd) {
        m.weight.dicts["gguf_architecture"] = "qwen4exp";
        m.num_k_heads = nk;
        m.num_v_heads = nv;
        m.head_v_dim = hd;
    }
    static void Restore(Qwen4ExpModel& m, Data& w, int tp) { m.RestoreGdnOutputWeight(w, tp); }
    static void Project(Qwen4ExpModel& m, Data& x, Data& w, Data& y) {
        m.RunGdnOutputProjection(x, w, y);
    }
};
}  // namespace fastllm
static void Check(bool x, const char* s) {
    if (!x) throw std::runtime_error(s);
}
static void Cuda(cudaError_t x) {
    if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x));
}
static void Device(int d) {
    FastllmCudaSetDevice(d);
    ((Executor*)GetExecutor())->SetFirstDevice("cuda:" + std::to_string(d));
}
static void Weight(Data& w, ggml_type type, int rows, int cols) {
    w.dataType = DATA_GGUF_FORMAT;
    w.ggmlType = type;
    w.isGGUFData = true;
    w.isModelWeight = true;
    w.disableGGUFRepack = true;
    w.Resize({rows, cols});
    w.Allocate();
    std::mt19937 rng(113);
    std::normal_distribution<float> dist(0, .03);
    std::vector<float> v(cols);
    for (int r = 0; r < rows; ++r) {
        for (auto& x : v) x = dist(rng);
        void* dst = w.cpuData + r * ggml_row_size(type, cols);
        if (type == GGML_TYPE_Q4_K)
            quantize_row_q4_K_ref(v.data(), (block_q4_K*)dst, cols);
        else if (type == GGML_TYPE_Q5_K)
            quantize_row_q5_K_ref(v.data(), (block_q5_K*)dst, cols);
        else
            quantize_row_q6_K_ref(v.data(), (block_q6_K*)dst, cols);
    }
}
static std::vector<float> Values(Data& d) {
    Data copy;
    copy.CopyFrom(d);
    copy.ToDevice(DataDevice::CPU);
    std::vector<float> v(copy.Count(0));
    Check(copy.dataType == FLOAT16, "output dtype");
    for (size_t i = 0; i < v.size(); ++i) v[i] = half_to_float(((uint16_t*)copy.cpuData)[i]);
    return v;
}
static double Rel(const std::vector<float>& a, const std::vector<float>& b) {
    Check(a.size() == b.size(), "output shape");
    double err = 0, mag = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        Check(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite output");
        err += double(a[i] - b[i]) * (a[i] - b[i]);
        mag += double(b[i]) * b[i];
    }
    return std::sqrt(err / std::max(mag, 1e-30));
}
static void Test(ggml_type type, int tokens, bool testTp) {
    const int nk = 16, per = 3, hd = 128, cols = nk * per * hd, rows = 256;
    Device(0);
    Qwen4ExpModel model;
    Qwen4GGUFTestAccess::Configure(model, nk, nk * per, hd);
    Data w;
    Weight(w, type, rows, cols);
    std::vector<uint8_t> packed(w.cpuData, w.cpuData + w.GetBytes());
    Qwen4GGUFTestAccess::Restore(model, w, testTp ? 2 : 1);
    Check(w.dataType == DATA_GGUF_FORMAT && !std::memcmp(w.cpuData, packed.data(), packed.size()),
          "packed bytes changed");
    std::vector<float> grouped(tokens * cols), tiled(tokens * cols), decoded(rows * cols),
        expected(tokens * rows);
    std::mt19937 rng(211);
    std::normal_distribution<float> dist(0, .3);
    for (auto& v : grouped) v = half_to_float(float_to_half(dist(rng)));
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < nk; ++k)
            for (int p = 0; p < per; ++p)
                for (int d = 0; d < hd; ++d)
                    tiled[t * cols + (p * nk + k) * hd + d] =
                        grouped[t * cols + (k * per + p) * hd + d];
    ggml_type_to_float(type)(packed.data(), decoded.data(), decoded.size());
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
            double sum = 0;
            for (int c = 0; c < cols; ++c)
                sum += double(decoded[r * cols + c]) * tiled[t * cols + c];
            expected[t * rows + r] = sum;
        }
    const int batch = tokens == 64 ? 2 : 1;
    const int sequence = tokens / batch;
    Data x(FLOAT16, {batch, sequence, cols}, grouped),
        refX(FLOAT16, {batch, sequence, cols}, tiled), y, refY;
    x.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    refX.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    w.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    Qwen4GGUFTestAccess::Project(model, x, w, y);
    Linear(refX, w, Data(), refY);
    auto single = Values(y), reference = Values(refY);
    Check(single == reference, "activation permutation differs from independent host permutation");
    // HF-origin packed weights already use grouped columns and must skip the permutation.
    model.weight.dicts.erase("gguf_architecture");
    Data hfOutput, hfReference;
    Qwen4GGUFTestAccess::Project(model, x, w, hfOutput);
    Linear(x, w, Data(), hfReference);
    Check(Values(hfOutput) == Values(hfReference), "HF packed projection was permuted");
    model.weight.dicts["gguf_architecture"] = "qwen4exp";
    const double nativeError = Rel(single, expected);
    Check(nativeError < .012, "native projection numeric error");
    double tpError = 0;
    if (testTp) {
        // Test the real multicuda packed-block slicer, preserving disjoint tiled intervals.
        Data split;
        Weight(split, type, rows, cols);
        Data bias;
        std::vector<int> devices = {0, 1};
        DivisionScheme scheme;
        for (int r = 0; r < 2; ++r)
            for (int p = 0; p < per; ++p)
                scheme[r].push_back({(p * nk + nk * r / 2) * hd, (p * nk + nk * (r + 1) / 2) * hd});
        Check(SplitMultiCudaWeight(split, bias, devices, scheme, 1, true, false), "TP split failed");
        std::vector<float> total(tokens * rows);
        for (int r = 0; r < 2; ++r) {
            Device(r);
            Qwen4ExpModel rank;
            Qwen4GGUFTestAccess::Configure(rank, nk / 2, nk * per / 2, hd);
            std::vector<float> local(tokens * cols / 2);
            for (int t = 0; t < tokens; ++t)
                std::copy_n(grouped.begin() + t * cols + r * cols / 2, cols / 2,
                            local.begin() + t * cols / 2);
            Data input(FLOAT16, {batch, sequence, cols / 2}, local), output;
            input.ToDevice(DataDevice::CUDA, std::vector<int>{r});
            Data& shard = *split.multiDeviceDatas.at(r);
            Data cpuShard;
            cpuShard.CopyFrom(shard);
            cpuShard.ToDevice(DataDevice::CPU);
            const size_t rowBytes = ggml_row_size(type, cols),
                         localBytes = ggml_row_size(type, cols / 2);
            for (int row = 0; row < rows; ++row) {
                size_t offset = 0;
                for (auto interval : scheme[r]) {
                    size_t start = ggml_row_size(type, interval.first),
                           bytes = ggml_row_size(type, interval.second - interval.first);
                    Check(!std::memcmp(cpuShard.cpuData + row * localBytes + offset,
                                       packed.data() + row * rowBytes + start, bytes),
                          "TP changed packed blocks");
                    offset += bytes;
                }
            }
            Qwen4GGUFTestAccess::Project(rank, input, shard, output);
            auto v = Values(output);
            for (size_t i = 0; i < v.size(); ++i) total[i] += v[i];
        }
        tpError = Rel(total, single);
        Check(tpError < .003, "TP output differs from single GPU");
    }
    // Sixteen-way sharding cuts 256-value blocks into 128-value heads: exact FP16 fallback.
    Device(0);
    Data fallback;
    Weight(fallback, type, rows, cols);
    Qwen4GGUFTestAccess::Restore(model, fallback, 16);
    Check(fallback.dataType == FLOAT16, "unaligned TP did not fall back");
    auto fp = (uint16_t*)fallback.cpuData;
    std::vector<uint16_t> decodedHalf(decoded.size());
    Float32ToFloat16(decoded.data(), decodedHalf.data(), decoded.size());
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < nk; ++k)
            for (int p = 0; p < per; ++p)
                for (int d = 0; d < hd; ++d)
                    Check(fp[r * cols + (k * per + p) * hd + d] ==
                              decodedHalf[r * cols + (p * nk + k) * hd + d],
                          "fallback column order");
    std::printf(
        "PASS type=%s tokens=%d native_rel_rms=%.8g tp_rel_rms=%.8g packed_exact=1 "
        "fallback_exact=1\n",
        ggml_type_name(type), tokens, nativeError, tpError);
}
static double Time(const std::function<void()>& f, int n) {
    for (int i = 0; i < 20; ++i) f();
    Cuda(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    Cuda(cudaEventCreate(&a));
    Cuda(cudaEventCreate(&b));
    Cuda(cudaEventRecord(a, cudaStreamPerThread));
    for (int i = 0; i < n; ++i) f();
    Cuda(cudaEventRecord(b, cudaStreamPerThread));
    Cuda(cudaEventSynchronize(b));
    float ms;
    Cuda(cudaEventElapsedTime(&ms, a, b));
    Cuda(cudaEventDestroy(a));
    Cuda(cudaEventDestroy(b));
    return ms * 1000 / n;
}
static void Bench(ggml_type type) {
    Device(0);
    const int nk = 16, nv = 48, hd = 128, cols = 6144, rows = 2560;
    Qwen4ExpModel model;
    Qwen4GGUFTestAccess::Configure(model, nk, nv, hd);
    Data w;
    Weight(w, type, rows, cols);
    Data dense;
    dense.CopyFrom(w);
    Qwen4GGUFTestAccess::Restore(model, dense, 16);
    std::vector<float> input(cols);
    for (int i = 0; i < cols; ++i) input[i] = std::sin(i * .31f) * .3f;
    Data x(FLOAT16, {1, 1, cols}, input), y, dy;
    x.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    w.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    dense.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    for (int round = 0; round < 3; ++round) {
        double old = Time([&] { Linear(x, dense, Data(), dy); }, 300);
        double native = Time([&] { Qwen4GGUFTestAccess::Project(model, x, w, y); }, 300);
        std::printf("BENCH type=%s round=%d fp16_us=%.4f native_with_permute_us=%.4f ratio=%.4f\n",
                    ggml_type_name(type), round, old, native, old / native);
    }
}
int main(int argc, char** argv) {
    try {
        SetThreads(2);
        const int deviceCount = FastllmCudaGetDeviceCount();
        if (deviceCount < 1) return 77;
        if (deviceCount < 2) std::puts("SKIP: TP checks require two GPUs");
        const bool benchmark = argc > 1 && std::string(argv[1]) == "--benchmark";
        for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
            for (int n : {1, 7, 64}) Test(type, n, deviceCount >= 2);
            if (benchmark) Bench(type);
        }
        std::puts("ALL_PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
