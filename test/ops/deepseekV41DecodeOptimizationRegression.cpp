#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fastllm;
static int deviceCount = 0;
static void Check(bool ok, const std::string &msg) { if (!ok) throw std::runtime_error(msg); }
static void Cuda(cudaError_t status) { Check(status == cudaSuccess, cudaGetErrorString(status)); }
static void SelectDevice(int device) {
    ApplyDeviceMap({{"cuda:" + std::to_string(device), 1}}, 0, 1);
    FastllmCudaSetDevice(device);
}
static Data Random(DataType type, const std::vector<int> &dims, int seed, float scale = .5f) {
    size_t count = 1; for (int d : dims) count *= d;
    std::mt19937 rng(seed); std::normal_distribution<float> normal(0, scale);
    std::vector<float> values(count); for (float &v : values) v = normal(rng);
    Data out(type, dims, values);
    out.ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()}); return out;
}
static std::vector<uint8_t> Bytes(Data &input) {
    std::vector<uint8_t> bytes(input.GetBytes());
    if (!bytes.empty()) {
        Cuda(cudaDeviceSynchronize());
        Cuda(cudaMemcpy(bytes.data(), input.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
    }
    return bytes;
}
static std::vector<float> Floats(Data &input) {
    auto bytes = Bytes(input);
    std::vector<float> values(input.Count(0));
    for (size_t i = 0; i < values.size(); ++i) {
        if (input.dataType == DataType::FLOAT32) memcpy(&values[i], bytes.data() + 4 * i, 4);
        else if (input.dataType == DataType::FLOAT16) {
            __half h; memcpy(&h, bytes.data() + 2 * i, 2); values[i] = __half2float(h);
        } else {
            Check(input.dataType == DataType::BFLOAT16, "unsupported comparison dtype");
            uint16_t bf; memcpy(&bf, bytes.data() + 2 * i, 2);
            uint32_t bits = (uint32_t)bf << 16; memcpy(&values[i], &bits, 4);
        }
    }
    return values;
}
static void Close(const std::vector<float> &expected, Data &actual, float atol, float rtol,
                  const std::string &label) {
    auto got = Floats(actual); Check(got.size() == expected.size(), label + " shape");
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == expected[i]) continue;
        Check(std::isfinite(got[i]) && std::isfinite(expected[i]) &&
              fabsf(got[i] - expected[i]) <= atol + rtol * fabsf(expected[i]),
              label + " index=" + std::to_string(i) + " expected=" + std::to_string(expected[i]) +
              " actual=" + std::to_string(got[i]));
    }
}
// Duplicate a tensor along one axis to exercise the general multi-token path.
static Data RepeatAxis(Data &input, int axis, int copies) {
    auto bytes = Bytes(input);
    auto dims = input.dims;
    size_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= dims[i];
    const size_t block = bytes.size() / outer;
    dims[axis] *= copies;
    Data out(input.dataType, dims);
    if (bytes.empty()) return out;
    out.Allocate();
    for (size_t i = 0; i < outer; ++i)
        for (int copy = 0; copy < copies; ++copy)
            memcpy(out.cpuData + (i * copies + copy) * block, bytes.data() + i * block, block);
    out.ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()});
    return out;
}
static void ReplayGraph(const std::function<void()> &run) {
    run(); Cuda(cudaDeviceSynchronize());
    Check(FastllmCudaGraphMemoryPoolBegin(), "graph pool begin");
    Check(FastllmCudaGraphBeginCapture(), "graph capture begin");
    run();
    void *captured = nullptr; Check(FastllmCudaGraphEndCapture(&captured), "graph capture end");
    cudaGraph_t graph = (cudaGraph_t)captured;
    std::vector<void *> reserved;
    Check(FastllmCudaGraphMemoryPoolEnd(reserved), "graph pool end");
    cudaGraphExec_t exec; Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 16; ++i) Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
    Cuda(cudaDeviceSynchronize());
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
    FastllmCudaGraphMemoryPoolRelease(reserved);
}
static void AttentionChecks() {
    int checks = 0;
    struct Shape { int tokens, start, window, width; };
    const Shape shapes[] = {{1,0,128,0}, {1,1,128,1}, {1,127,128,63}, {1,129,128,64},
        {1,8192,512,65}, {1,8192,512,1024}, {1,8192,128,512}, {3,0,512,65}, {5,511,512,127}};
    for (int dev = 0; dev < deviceCount; ++dev) {
        SelectDevice(dev);
        for (int heads : {32,64}) for (auto shape : shapes) {
            const int tokens = shape.tokens, cap = 2048;
            auto q = Random(DataType::BFLOAT16, {1,tokens,heads,512}, heads + tokens);
            auto chunk = Random(DataType::BFLOAT16, {1,tokens,512}, 731);
            auto ring = Random(DataType::BFLOAT16, {1,shape.window,512}, 799);
            auto cmp = Random(DataType::BFLOAT16, {1,cap,512}, 887);
            auto sink = Random(DataType::FLOAT32, {heads}, 993, 1.f);
            Data idx(DataType::INT32, {1,tokens,shape.width}); idx.Allocate();
            for (int i = 0; i < tokens * shape.width; ++i)
                ((int32_t *)idx.cpuData)[i] = i % 17 == 0 ? -1 : (i % 19 == 0 ? cap : (i * 37 + 3) % cap);
            idx.ToDevice(DataDevice::CUDA, std::vector<int>{dev});
            Data packedRing, packedCmp;
            Check(FastllmCudaDeepSeekV41QuantizeKV(ring, packedRing, 1, 32), "ring pack rejected");
            Check(FastllmCudaDeepSeekV41QuantizeKV(cmp, packedCmp, 3, 16), "cmp pack rejected");
            for (int storage : {0,1,2}) {
                Data output;
                auto run = [&]() { Check(FastllmCudaDeepSeekV41SparseAttention(q, chunk,
                    &((storage == 1) ? packedRing : ring), &(storage ? packedCmp : cmp),
                    shape.width ? &idx : nullptr, sink, shape.window, shape.start, .0441941738f, output),
                    "sparse attention rejected"); };
                auto qBatch = RepeatAxis(q, 0, 2), chunkBatch = RepeatAxis(chunk, 0, 2);
                auto ringBatch = RepeatAxis(storage == 1 ? packedRing : ring, 0, 2);
                auto cmpBatch = RepeatAxis(storage ? packedCmp : cmp, 0, 2);
                auto idxBatch = RepeatAxis(idx, 0, 2);
                Data reference;
                Check(FastllmCudaDeepSeekV41SparseAttention(qBatch, chunkBatch, &ringBatch, &cmpBatch,
                    shape.width ? &idxBatch : nullptr, sink, shape.window, shape.start, .0441941738f,
                    reference), "batched attention reference rejected");
                auto expected = Bytes(reference);
                expected.resize(expected.size() / 2);
                run();
                Check(expected == Bytes(output), "attention decode/batch mismatch device=" + std::to_string(dev) +
                    " heads=" + std::to_string(heads) + " tokens=" + std::to_string(tokens) +
                    " start=" + std::to_string(shape.start) + " storage=" + std::to_string(storage));
                ++checks;
            }
        }
    }
    std::cout << "PASS attention: " << checks << " comparisons; both devices, BF16/FP8/FP4, decode/prefill, ring wrap and invalid indices bitwise equal\n";
}

static void IndexerChecks() {
    int checks = 0;
    struct Shape { int batch, tokens, heads, candidates; };
    const Shape shapes[] = {{1,1,1,1}, {1,1,16,63}, {1,1,17,65}, {2,1,32,129},
        {1,1,32,2048}, {1,1,32,8193}, {1,1,64,257}, {1,3,32,65}, {2,5,17,129}};
    for (int dev = 0; dev < deviceCount; ++dev) {
        SelectDevice(dev);
        for (auto shape : shapes) {
            auto q = Random(DataType::BFLOAT16, {shape.batch,shape.tokens,shape.heads,128}, 551);
            auto weights = Random(DataType::FLOAT32, {shape.batch,shape.tokens,shape.heads}, 993);
            auto k = Random(DataType::BFLOAT16, {shape.batch,shape.candidates,128}, 777);
            for (int storage : {0,1,2,3}) {
                Data packed, out;
                if (storage) Check(FastllmCudaDeepSeekV41QuantizeKV(k, packed, storage, storage == 3 ? 16 : 32), "index pack");
                auto run = [&]() { Check(FastllmCudaDeepSeekV41IndexerScore(q, weights, storage ? packed : k,
                    1, shape.candidates + 3, out), "index score rejected"); };
                auto qBatch = RepeatAxis(q, 1, 2), weightsBatch = RepeatAxis(weights, 1, 2);
                auto reference = [&](int ratio, int start) {
                    Data batch;
                    Check(FastllmCudaDeepSeekV41IndexerScore(qBatch, weightsBatch, storage ? packed : k,
                        ratio, start, batch), "prefill index reference rejected");
                    auto values = Floats(batch);
                    std::vector<float> expected;
                    const size_t block = shape.tokens * shape.candidates;
                    for (int i = 0; i < shape.batch; ++i)
                        expected.insert(expected.end(), values.begin() + i * 2 * block,
                                        values.begin() + (i * 2 + 1) * block);
                    return expected;
                };
                auto expected = reference(1, shape.candidates + 3);
                Data referenceScores(DataType::FLOAT32, {shape.batch,shape.tokens,shape.candidates}, expected);
                referenceScores.ToDevice(DataDevice::CUDA, std::vector<int>{dev});
                Data top;
                Check(FastllmCudaDeepSeekV41IndexerTopK(referenceScores, nullptr, 64, 1,
                    shape.candidates + 3, 64, top), "reference index top-k");
                const auto expectedTop = Bytes(top);
                run();
                Close(expected, out, 2e-5f, 2e-5f, "index decode/prefill device=" + std::to_string(dev));
                Check(FastllmCudaDeepSeekV41IndexerTopK(out, nullptr, 64, 1,
                    shape.candidates + 3, 64, top), "optimized index top-k");
                Check(expectedTop == Bytes(top), "index top-k set/order changed");
                // Replay must continue writing the same output allocation.
                if (shape.tokens == 1 && shape.candidates == 65) {
                    ReplayGraph(run); Close(expected, out, 2e-5f, 2e-5f, "index graph");
                }
                setenv("FASTLLM_DSV41_LEGACY_INDEXER", "1", 1); run();
                Close(expected, out, 1e-3f, 1e-3f, "index scalar fallback");
                unsetenv("FASTLLM_DSV41_LEGACY_INDEXER");
                // Visibility at a non-tile boundary and wholly masked tiles.
                auto partial = [&]() { Check(FastllmCudaDeepSeekV41IndexerScore(q, weights, storage ? packed : k,
                    4, 259, out), "partial index"); };
                expected = reference(4, 259); partial();
                Close(expected, out, 2e-5f, 2e-5f, "index visibility");
                ++checks;
            }
        }
    }
    std::cout << "PASS indexer: " << checks << " decode/prefill, batched, tail, packed KV, graph and scalar fallback cases\n";
}

static void FallbackChecks() {
    for (int dev = 0; dev < deviceCount; ++dev) {
        SelectDevice(dev);
        for (int heads : {32,64}) for (auto type : {DataType::BFLOAT16, DataType::FLOAT32}) {
            auto q = Random(type, {1,1,heads,512}, 171);
            auto chunk = Random(DataType::BFLOAT16, {1,1,512}, 991);
            auto ring = Random(DataType::BFLOAT16, {1,128,512}, 712);
            auto sink = Random(DataType::FLOAT32, {heads}, 812);
            Data out;
            auto run = [&]() { Check(FastllmCudaDeepSeekV41SparseAttention(q, chunk, &ring, nullptr,
                nullptr, sink, 128, 129, .0441941738f, out), "fallback attention rejected"); };
            run(); auto expected = Floats(out);
            setenv("FASTLLM_DSV41_LEGACY_ATTN", "1", 1); run();
            Close(expected, out, .003f, .02f, "attention scalar tolerance");
            auto scalar = Bytes(out); ReplayGraph(run);
            Check(scalar == Bytes(out), "scalar graph output changed");
            unsetenv("FASTLLM_DSV41_LEGACY_ATTN");
        }
    }
    std::cout << "PASS attention forced scalar fallback and FLOAT32 query dispatch\n";
}

static void AttentionGraphLifetimeChecks() {
    // Keep two captured workspaces alive and replay on independent streams.
    // Different inputs make accidental scratch reuse visible in the outputs.
    for (int dev = 0; dev < deviceCount; ++dev) {
        SelectDevice(dev);
        auto q0 = Random(DataType::BFLOAT16, {1,1,32,512}, 347);
        auto q1 = Random(DataType::BFLOAT16, {1,1,32,512}, 897);
        auto chunk = Random(DataType::BFLOAT16, {1,1,512}, 734);
        auto ring = Random(DataType::BFLOAT16, {1,512,512}, 721);
        auto sink = Random(DataType::FLOAT32, {32}, 613);
        Data outputs[2];
        auto run = [&](int i) { Check(FastllmCudaDeepSeekV41SparseAttention(i ? q1 : q0,
            chunk, &ring, nullptr, nullptr, sink, 512, 8192, .0441941738f, outputs[i]),
            "concurrent graph attention rejected"); };
        std::vector<uint8_t> expected[2];
        for (int i = 0; i < 2; ++i) { run(i); expected[i] = Bytes(outputs[i]); }
        cudaGraph_t graphs[2]; cudaGraphExec_t execs[2]; cudaStream_t streams[2];
        std::vector<void *> reserved[2];
        for (int i = 0; i < 2; ++i) {
            run(i); Cuda(cudaDeviceSynchronize());
            Check(expected[i] == Bytes(outputs[i]), "attention graph warmup already differs");
            Check(FastllmCudaGraphMemoryPoolBegin(), "concurrent graph pool begin");
            Check(FastllmCudaGraphBeginCapture(), "concurrent graph capture begin");
            run(i);
            void *captured = nullptr;
            Check(FastllmCudaGraphEndCapture(&captured), "concurrent graph capture end");
            graphs[i] = (cudaGraph_t)captured;
            Check(FastllmCudaGraphMemoryPoolEnd(reserved[i]), "concurrent graph pool end");
            Cuda(cudaGraphInstantiate(&execs[i], graphs[i], nullptr, nullptr, 0));
            Cuda(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
            Cuda(cudaGraphLaunch(execs[i], streams[i])); Cuda(cudaStreamSynchronize(streams[i]));
            Check(expected[i] == Bytes(outputs[i]), "individual attention graph replay differs");
        }
        for (void *a : reserved[0]) for (void *b : reserved[1])
            Check(a != b, "concurrent graph pool reused a live allocation");
        for (int repeat = 0; repeat < 100; ++repeat)
            for (int i = 0; i < 2; ++i) Cuda(cudaGraphLaunch(execs[i], streams[i]));
        Cuda(cudaDeviceSynchronize());
        for (int i = 0; i < 2; ++i) {
            Check(expected[i] == Bytes(outputs[i]), "concurrent graph scratch corrupted attention device=" +
                std::to_string(dev) + " graph=" + std::to_string(i));
            Cuda(cudaGraphExecDestroy(execs[i])); Cuda(cudaGraphDestroy(graphs[i]));
            Cuda(cudaStreamDestroy(streams[i])); FastllmCudaGraphMemoryPoolRelease(reserved[i]);
        }
    }
    std::cout << "PASS two concurrent attention graphs with independent scratch allocations\n";
}

static void RmsCpuReference(Data &input, Data &weight, Data &output) {
    auto x = Floats(input), w = Floats(weight);
    std::vector<float> expected(x.size());
    const int channels = input.dims.back();
    for (size_t base = 0; base < x.size(); base += channels) {
        double squares = 0;
        for (int j = 0; j < channels; ++j) squares += (double)x[base+j] * x[base+j];
        const double scale = 1. / std::sqrt(squares / channels + 1e-6);
        for (int j = 0; j < channels; ++j) expected[base+j] = x[base+j] * scale * w[j];
    }
    const float tol = input.dataType == DataType::BFLOAT16 ? .004f :
        (input.dataType == DataType::FLOAT16 ? .0006f : 2e-6f);
    Close(expected, output, 2e-6f, tol, "RMSNorm CPU reference");
}

static void RmsNormChecks() {
    int checks = 0;
    for (int dev = 0; dev < deviceCount; ++dev) {
        SelectDevice(dev);
        for (auto type : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32})
            for (int channels : {1280,4096,5120,5121}) for (int tokens : {1,2,8,9,64}) {
                auto input = Random(type, {tokens,channels}, tokens + channels);
                auto weight = Random(DataType::FLOAT32, {channels}, 928);
                Data out; out.CopyFrom(input);
                Cuda(cudaDeviceSynchronize());
                auto run = [&]() { Check(FastllmCudaRMSNorm(input, weight, out, 1e-6f), "RMSNorm rejected"); };
                auto batched = RepeatAxis(input, 0, 9);
                Data reference; reference.CopyFrom(batched);
                Check(FastllmCudaRMSNorm(batched, weight, reference, 1e-6f), "batched RMSNorm rejected");
                auto expected = Bytes(reference); expected.resize(input.GetBytes());
                run();
                Check(expected == Bytes(out), "RMSNorm reduction/rounding mismatch");
                RmsCpuReference(input, weight, out);
                Data inplace; inplace.CopyFrom(input);
                Check(FastllmCudaRMSNorm(inplace, weight, inplace, 1e-6f), "in-place RMSNorm rejected");
                Check(expected == Bytes(inplace), "in-place RMSNorm mismatch");
                ++checks;
                if (type == DataType::BFLOAT16 && channels == 5120 && tokens == 1) {
                    ReplayGraph(run); Check(expected == Bytes(out), "RMSNorm graph mismatch");
                }
            }
        for (auto type : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32}) {
            constexpr int channels = 5120, tokens = 2;
            auto inputOwner = Random(type, {tokens * channels + 1}, 194);
            auto outputOwner = Random(type, {tokens * channels + 1}, 197);
            auto weightOwner = Random(DataType::FLOAT32, {channels + 1}, 192);
            Data input(type, {tokens, channels}), out(type, {tokens, channels});
            Data weight(DataType::FLOAT32, {channels});
            input.FakeFrom(inputOwner, input.unitSize); out.FakeFrom(outputOwner, out.unitSize);
            weight.FakeFrom(weightOwner, sizeof(float));
            Check(FastllmCudaRMSNorm(input, weight, out, 1e-6f), "offset RMSNorm rejected");
            RmsCpuReference(input, weight, out);
            auto expected = Bytes(out);
            Check(FastllmCudaRMSNorm(input, weight, input, 1e-6f), "offset in-place RMSNorm rejected");
            Check(expected == Bytes(input), "offset in-place RMSNorm mismatch");
        }
    }
    std::cout << "PASS RMSNorm: " << checks << " dtype/shape/in-place/graph comparisons, bitwise equal\n";
}

static void AttentionCapacityFallbackCheck() {
    // Occupy every fitting idle pool block during capture, when new device
    // allocations are prohibited. The fused path must still capture/replay.
    SelectDevice(0);
    auto q = Random(DataType::BFLOAT16, {1,1,32,512}, 185);
    auto chunk = Random(DataType::BFLOAT16, {1,1,512}, 832);
    auto ring = Random(DataType::BFLOAT16, {1,512,512}, 314);
    auto sink = Random(DataType::FLOAT32, {32}, 485);
    Data out;
    auto run = [&]() { Check(FastllmCudaDeepSeekV41SparseAttention(q, chunk, &ring,
        nullptr, nullptr, sink, 512, 8192, .0441941738f, out), "capacity fallback rejected"); };
    run(); auto expected = Bytes(out);
    Cuda(cudaDeviceSynchronize());
    Check(FastllmCudaGraphMemoryPoolBegin(), "capacity graph pool begin");
    Check(FastllmCudaGraphBeginCapture(), "capacity graph begin");
    std::vector<void *> held;
    for (;;) {
        void *ptr = nullptr;
        auto status = FastllmCudaTryMalloc(&ptr, 32 * 512 * sizeof(float));
        if (status == FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE) break;
        Check(status == FASTLLM_CUDA_TRY_MALLOC_SUCCESS, "unexpected allocation error");
        held.push_back(ptr);
    }
    run();
    void *captured = nullptr; Check(FastllmCudaGraphEndCapture(&captured), "capacity graph end");
    std::vector<void *> reserved;
    Check(FastllmCudaGraphMemoryPoolEnd(reserved), "capacity graph pool end");
    Check(!FastllmCudaGetThreadError(), "capacity miss poisoned CUDA state");
    cudaGraph_t graph = (cudaGraph_t)captured; cudaGraphExec_t exec;
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    Cuda(cudaGraphLaunch(exec, cudaStreamPerThread)); Cuda(cudaDeviceSynchronize());
    std::vector<uint8_t> got(expected.size());
    Cuda(cudaMemcpy(got.data(), out.cudaData, got.size(), cudaMemcpyDeviceToHost));
    Check(got == expected, "capacity fallback output changed");
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
    FastllmCudaGraphMemoryPoolRelease(reserved);
    for (void *ptr : held) FastllmCudaFree(ptr);
    std::cout << "PASS attention optional workspace capacity fallback during graph capture\n";
}

int main(int argc, char **argv) {
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1) return 77;
    try {
        unsetenv("FASTLLM_DSV41_REFERENCE_MATH");
        std::string section = "all";
        if (argc == 2) section = argv[1];
        Check(argc <= 2, "usage: deepseekV41DecodeOptimizationRegression [section]");
        Check(section == "all" || section == "attention" ||
              section == "indexer" || section == "fallback" || section == "rmsnorm" ||
              section == "capacity", "unknown section");
        if (section == "all" || section == "attention") AttentionChecks();
        if (section == "all" || section == "attention") AttentionGraphLifetimeChecks();
        if (section == "all" || section == "indexer") IndexerChecks();
        if (section == "all" || section == "fallback") FallbackChecks();
        if (section == "all" || section == "rmsnorm") RmsNormChecks();
        if (section == "all" || section == "capacity") AttentionCapacityFallbackCheck();
        return 0;
    } catch (const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
