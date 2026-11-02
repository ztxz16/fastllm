#include "models/clover.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-linear-output.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
using namespace fastllm;
namespace {
bool fp16 = false;
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
uint16_t Bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    bits += 0x7fff + ((bits >> 16) & 1);
    return bits >> 16;
}
void ProjectionPrecision() {
    // Large cancelling partial sums detect reduced-precision accumulation.
    // Cover full projections through 64 requests with a 16-token block.
    auto encode = [](float x) { return fp16 ? __half_raw(__float2half_rn(x)).x : Bf16(x); };
    for (int rows : {1, 2, 8, 16, 32, 64, 128, 512, 1024}) {
        constexpr int width = 512, columns = 16;
        const auto type = fp16 ? FLOAT16 : BFLOAT16;
        Data x(type, {rows, width}), w(type, {columns, width}), y(type, {rows, columns});
        x.Allocate(); w.Allocate(); y.Allocate();
        for (int r = 0; r < rows; ++r) for (int d = 0; d < width; ++d)
            ((uint16_t *)x.cpuData)[r * width + d] = encode(d == width - 1 ? 1.f + (r % 4) * .25f : 1.f);
        for (int c = 0; c < columns; ++c) for (int d = 0; d < width; ++d)
            ((uint16_t *)w.cpuData)[c * width + d] = encode((2048.f + c * 16) * (d < width / 2 ? 1.f : -1.f));
        for (auto *v : {&x, &w, &y}) v->ToDevice(DataDevice::CUDA, std::vector<int>{0});
        if (fp16)
            Require(FastllmCudaHalfMatMulFloat16Fp32(x.cudaData, w.cudaData, y.cudaData, rows, width, columns), "FP16 FP32-accumulating projection");
        else
            Linear(x, w, *GetEmptyData(), y);
        y.ToDevice(DataDevice::CPU);
        for (int r = 0; r < rows; ++r) for (int c = 0; c < columns; ++c) {
            uint16_t expected = encode(-(2048.f + c * 16) * (r % 4) * .25f);
            auto actual = ((uint16_t *)y.cpuData)[r * columns + c];
            Require(actual == expected || ((actual & 0x7fff) == 0 && (expected & 0x7fff) == 0),
                    "projection lost FP32 accumulation");
        }
    }
}
void Populate(CloverModel &m, int seed, bool zeroHead = false, int maxPositions = 1024, int topk = 4) {
    m.weight.dicts = {{"hidden_size", "512"}, {"num_hidden_layers", "1"},
        {"num_attention_heads", "1"}, {"num_key_value_heads", "1"},
        {"head_dim", "256"}, {"moe_intermediate_size", "128"},
        {"num_experts", "16"}, {"num_experts_per_tok", std::to_string(topk)},
        {"max_position_embeddings", std::to_string(maxPositions)}, {"vocab_size", "128"},
        {"mask_token_id", "127"}, {"layer_types", "[\"full_attention\"]"}};
    m.InitParams();
    std::mt19937 rng(seed); std::normal_distribution<float> normal(0.f, .03f);
    auto weight = [&](const std::string &name, const std::vector<int> &shape, bool norm = false) {
        auto &w = m.weight[name]; w.dataType = norm ? FLOAT32 : BFLOAT16;
        w.UpdateUnitSize(); w.Resize(shape); w.Allocate(false);
        for (int i = 0; i < w.Count(0); ++i) {
            float value = normal(rng);
            if (zeroHead && name == "lm_head.weight") value = 0.f;
            if (norm) ((float *)w.cpuData)[i] = value;
            else ((uint16_t *)w.cpuData)[i] = Bf16(value);
        }
    };
    weight("model.embed_tokens.weight", {128, 512}); weight("lm_head.weight", {128, 512});
    weight("model.norm.weight", {512}, true);
    const std::string p = "model.layers.0.";
    weight(p + "input_layernorm.weight", {512}, true);
    weight(p + "post_attention_layernorm.weight", {512}, true);
    weight(p + "self_attn.q_proj.weight", {512, 512});
    weight(p + "self_attn.k_proj.weight", {256, 512});
    weight(p + "self_attn.v_proj.weight", {256, 512});
    weight(p + "self_attn.o_proj.weight", {512, 256});
    weight(p + "self_attn.q_norm.weight", {256}, true);
    weight(p + "self_attn.k_norm.weight", {256}, true);
    weight(p + "mlp.gate.weight", {16, 512});
    weight(p + "mlp.experts.gate_up_proj", {16, 256, 512});
    weight(p + "mlp.experts.down_proj", {16, 512, 128});
    weight(p + "mlp.shared_expert.gate_proj.weight", {128, 512});
    weight(p + "mlp.shared_expert.up_proj.weight", {128, 512});
    weight(p + "mlp.shared_expert.down_proj.weight", {512, 128});
    weight(p + "mlp.shared_expert_gate.weight", {1, 512});
    if (fp16) m.SetDataType(FLOAT16);
    m.WarmUp();
}
std::vector<unsigned char> Bytes(const Data &x) {
    Data host; host.CopyFrom(x); host.ToDevice(DataDevice::CPU);
    return {host.cpuData, host.cpuData + host.GetBytes()};
}
void FixedRowProjection() {
    if (fp16 || FastllmCudaRuntimeArch() != 90) return;
    std::mt19937 rng(1607); std::normal_distribution<float> normal;
    for (int inner : {1024, 4096}) for (int rows : {64, 96, 128, 256, 288, 512, 1024}) {
        const int columns = inner == 4096 ? 2048 : 1024;
        Data x(BFLOAT16, {rows, inner}), w(BFLOAT16, {columns, inner}),
            y(BFLOAT16, {rows, columns}), reference(BFLOAT16, {rows, columns});
        for (auto *v : {&x, &w, &y, &reference}) {
            v->Allocate();
            for (uint64_t i = 0; i < v->Count(0); ++i) ((uint16_t *)v->cpuData)[i] = Bf16(normal(rng));
            v->ToDevice(DataDevice::CUDA, std::vector<int>{0});
        }
        auto serial = [&]() {
            for (int row = 0; row < rows; row += 32) {
                Data a(BFLOAT16, {32, inner}), b(BFLOAT16, {32, columns});
                a.FakeFrom(x, size_t(row) * inner * 2); b.FakeFrom(reference, size_t(row) * columns * 2);
                Linear(a, w, *GetEmptyData(), b);
            }
        };
        auto combined = [&]() {
            Require(FastllmCudaBFloat16LinearFixedRows(x, w, y), "fixed-row projection dispatch");
        };
        serial(); combined();
        Require(Bytes(y) == Bytes(reference), "fixed-row projection changed reduction");
        cudaGraph_t graph; cudaGraphExec_t executable;
        Require(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess, "fixed-row capture begin");
        combined();
        Require(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "fixed-row capture end");
        Require(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == cudaSuccess, "fixed-row graph instantiate");
        std::vector<uint16_t> fresh(x.Count(0)); for (auto &v : fresh) v = Bf16(normal(rng));
        Require(cudaMemcpy(x.cudaData, fresh.data(), fresh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess, "fixed-row refreshed input");
        serial();
        Require(cudaGraphLaunch(executable, cudaStreamPerThread) == cudaSuccess, "fixed-row graph launch");
        Require(Bytes(y) == Bytes(reference), "fixed-row graph changed reduction");
        cudaGraphExecDestroy(executable); cudaGraphDestroy(graph);
    }
}
std::vector<std::vector<unsigned char>> Run(int seed, bool graphs, bool busyPool) {
    setenv("FASTLLM_CUDA_GRAPH", graphs ? "1" : "0", 1);
    CloverModel model; Populate(model, seed);
    std::vector<std::pair<Data, Data>> cache;
    Data logits;
    model.ForwardLogits({1, 4, 2, 7}, cache, false, logits);
    const auto cleanK = Bytes(cache[0].first), cleanV = Bytes(cache[0].second);
    if (busyPool) Require(FastllmCudaGraphMemoryPoolBegin(), "test pool begin");
    std::vector<std::vector<unsigned char>> results;
    // Replay with different inputs and both preview/verification semantics;
    // widths switch back as well as grow, exercising release and recapture.
    for (int width : {4, 4, 16, 16, 1, 2, 16, 32, 32, 8, 8, 16, 4}) {
        std::vector<int> tokens(width);
        for (int i = 0; i < width; ++i) tokens[i] = (seed + i * 11 + int(results.size()) * 3) % 127;
        model.ForwardLogits(tokens, cache, true, logits);
        results.push_back(Bytes(logits));
        Require(Bytes(cache[0].first) == cleanK && Bytes(cache[0].second) == cleanV, "preview modified committed KV");
        std::vector<std::pair<Data, Data>> verify(1);
        verify[0].first.CopyFrom(cache[0].first); verify[0].second.CopyFrom(cache[0].second);
        CloverSnapshots snapshots;
        model.ForwardLogits(tokens, verify, false, logits, false, &snapshots);
        results.push_back(Bytes(logits));
        results.push_back(Bytes(verify[0].first)); results.push_back(Bytes(verify[0].second));
    }
    if (busyPool) {
        std::vector<void *> reserved;
        Require(FastllmCudaGraphMemoryPoolEnd(reserved), "MoE fallback aborted another pool owner");
        FastllmCudaGraphMemoryPoolRelease(reserved);
    }
    return results;
}
std::vector<std::vector<unsigned char>> RunBatch(int seed, bool graphs, int batch = 2, bool busyPool = false, int topk = 4) {
    setenv("FASTLLM_CUDA_GRAPH", graphs ? "1" : "0", 1);
    CloverModel model; Populate(model, seed, false, 1024, topk);
    using Cache = std::vector<std::pair<Data, Data>>;
    std::vector<Cache> caches(batch), reference(batch);
    for (int b = 0; b < batch; ++b) {
        Data ignored;
        std::vector<int> prefix(5 + 3 * b);
        for (int i = 0; i < (int)prefix.size(); ++i) prefix[i] = (b * 31 + i * 7) % 127;
        model.ForwardLogits(prefix, caches[b], false, ignored);
    }
    if (busyPool) Require(FastllmCudaGraphMemoryPoolBegin(), "batch test pool begin");
    std::vector<Cache *> cachePtrs;
    for (auto &cache : caches) cachePtrs.push_back(&cache);
    std::vector<std::vector<unsigned char>> results;
    auto compare = [&](Data &actual, Data &expected) {
        auto a = Bytes(actual), e = Bytes(expected);
        Require(a.size() == e.size(), "batch logits size");
        double diff2 = 0, ref2 = 0; float maximum = 0;
        for (size_t i = 0; i < a.size(); i += sizeof(float)) {
            float x, y; std::memcpy(&x, a.data() + i, 4); std::memcpy(&y, e.data() + i, 4);
            Require(std::isfinite(x) && std::isfinite(y), "nonfinite batch logits");
            maximum = std::max(maximum, std::abs(x-y));
            diff2 += (x-y)*(x-y); ref2 += y*y;
        }
        if (maximum > .04f || diff2 > 2.5e-5 * std::max(ref2, 1e-10))
            std::cerr << "batch=" << batch << " rows=" << actual.dims[0] << " max_abs=" << maximum
                      << " relative_l2=" << std::sqrt(diff2 / std::max(ref2, 1e-10)) << std::endl;
        Require(maximum <= .04f && diff2 <= 2.5e-5 * std::max(ref2, 1e-10), "batch/serial logits differ");
    };
    for (int width : {2, 4, 8, 16, 32, 16}) {
        std::vector<std::vector<int>> tokens(batch, std::vector<int>(width));
        std::vector<Data> logits;
        std::vector<CloverSnapshots> snapshots(batch), serialSnapshots(batch);
        std::vector<CloverSnapshots *> snapshotPtrs;
        for (auto &snapshot : snapshots) snapshotPtrs.push_back(&snapshot);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < width; ++i) tokens[b][i] = (seed + b * 47 + i * 11) % 127;
            reference[b].resize(caches[b].size());
            for (size_t i = 0; i < caches[b].size(); ++i) {
                reference[b][i].first.CopyFrom(caches[b][i].first);
                reference[b][i].second.CopyFrom(caches[b][i].second);
            }
        }
        model.ForwardLogitsBatch(tokens, cachePtrs, true, logits);
        for (int b = 0; b < batch; ++b) {
            Data expected; model.ForwardLogits(tokens[b], reference[b], true, expected);
            compare(logits[b], expected); results.push_back(Bytes(logits[b]));
            Require(Bytes(caches[b][0].first) == Bytes(reference[b][0].first) &&
                    Bytes(caches[b][0].second) == Bytes(reference[b][0].second), "batch preview changed KV");
        }
        // Serial width=32 and batch=2,width=16 share 32 rows but different
        // coreset partitions: graph cache keys must distinguish them.
        model.ForwardLogitsBatch(tokens, cachePtrs, false, logits, snapshotPtrs);
        for (int b = 0; b < batch; ++b) {
            Data expected; model.ForwardLogits(tokens[b], reference[b], false, expected, false, &serialSnapshots[b]);
            compare(logits[b], expected); results.push_back(Bytes(logits[b]));
            int accepted = b % 2 ? width : 1;
            model.RestorePrefix(caches[b], snapshots[b], accepted);
            model.RestorePrefix(reference[b], serialSnapshots[b], accepted);
            Require(Bytes(caches[b][0].first) == Bytes(reference[b][0].first) &&
                    Bytes(caches[b][0].second) == Bytes(reference[b][0].second), "batch rollback crossed request state");
        }
    }
    if (busyPool) {
        std::vector<void *> reserved;
        Require(FastllmCudaGraphMemoryPoolEnd(reserved), "batch fallback aborted another pool owner");
        FastllmCudaGraphMemoryPoolRelease(reserved);
    }
    return results;
}

void SamplingMetadataBatch(int batch, bool sample) {
    // A zero head makes every raw logit identical. Output differences can then
    // only come from request metadata/sampling, not GEMM reduction geometry.
    CloverModel model; Populate(model, 19, true);
    model.eos_token_id = 3; model.eos_token_ids = {3};
    auto config = [&](int b) {
        return json11::Json::object{{"mode", "self_speculative"}, {"block_size", 16},
            {"max_new_tokens", 19 + (b % 4) * 7}, {"ignore_eos", true},
            {"temperature", sample ? .8 : 0.}, {"do_sample", sample}, {"seed", 27 + b},
            {"top_k", -1}, {"top_p", 1.}, {"presence_penalty", .3 * (b % 4)},
            {"frequency_penalty", .13 * (b % 3)}, {"repetition_penalty", b % 2 ? 1.05 : 1.},
            {"accept_rule", b % 2 ? "exact" : "threshold"}, {"accept_delta", b % 3 ? .6 : 1.4},
            {"dllm_think_end_token_id", 17}, {"dllm_think_end_start", 1},
            {"dllm_think_end_window", 3}, {"dllm_think_end_bias", 2.},
            {"dllm_length_penalty_start", 2}, {"dllm_length_penalty_window", 4},
            {"dllm_length_penalty", .5}};
    };
    auto launch = [&](int b) { return model.LaunchClover({1, 17, 3, 1 + b % 11}, config(b)); };
    auto collect = [&](int handle) {
        std::vector<int> result;
        for (;;) {
            int tokens[64], stats[3];
            int n = model.FetchResponseTokensBatch(handle, tokens, 64, stats);
            if (n == -1) break;
            Require(n >= 0, "metadata generation failed");
            result.insert(result.end(), tokens, tokens + n);
        }
        return result;
    };
    model.maxBatch = 1;
    std::vector<std::vector<int>> reference(batch), actual(batch);
    for (int b = 0; b < batch; ++b) reference[b] = collect(launch(b));
    model.maxBatch = batch;
    std::vector<int> handles;
    for (int b = 0; b < batch; ++b) handles.push_back(launch(b));
    std::vector<std::thread> readers;
    std::vector<std::exception_ptr> errors(batch);
    for (int b = 0; b < batch; ++b) readers.emplace_back([&, b] {
        try { actual[b] = collect(handles[b]); } catch (...) { errors[b] = std::current_exception(); }
    });
    for (auto &reader : readers) reader.join();
    for (int b = 0; b < batch; ++b) {
        if (errors[b]) std::rethrow_exception(errors[b]);
        Require(actual[b].size() == size_t(19 + (b % 4) * 7), "metadata generation incomplete");
        Require(actual[b] == reference[b], "batched penalties/counts differ from serial");
    }
}

void CacheGrowth() {
    constexpr int limit = 2065, budget = 2304, width = 32;
    CloverModel model; Populate(model, 23, false, limit);
    using Cache = std::vector<std::pair<Data, Data>>;
    std::vector<Cache> caches(3), reference(3);
    std::vector<Cache *> ptrs, refs;
    const int positions[] = {251, 1009, 2033};
    const int capacities[] = {1024, 2048, budget};
    std::vector<std::vector<int>> tokens(3, std::vector<int>(width));
    for (int b = 0; b < 3; ++b) {
        std::vector<int> prefix(positions[b]);
        for (int i = 0; i < positions[b]; ++i) prefix[i] = (i * 7 + b * 13) % 127;
        for (int i = 0; i < width; ++i) tokens[b][i] = (i * 3 + b * 11) % 127;
        Data out; model.ForwardLogits(prefix, caches[b], false, out, true);
        reference[b].resize(1);
        reference[b][0].first.CopyFrom(caches[b][0].first);
        reference[b][0].second.CopyFrom(caches[b][0].second);
        reference[b][0].first.Expansion({budget, 1, 256});
        reference[b][0].second.Expansion({budget, 1, 256});
        ptrs.push_back(&caches[b]); refs.push_back(&reference[b]);
    }
    // Compare a growing cache against a pre-reserved cache with the same
    // logical contents. Preview may grow storage but must not commit tokens.
    std::vector<Data> actual, expected;
    model.ForwardLogitsBatch(tokens, ptrs, true, actual);
    model.ForwardLogitsBatch(tokens, refs, true, expected);
    for (int b = 0; b < 3; ++b) {
        Require(Bytes(actual[b]) == Bytes(expected[b]), "KV growth changed preview logits");
        for (bool value : {false, true}) {
            auto &a = value ? caches[b][0].second : caches[b][0].first;
            auto &r = value ? reference[b][0].second : reference[b][0].first;
            Require(a.dims[0] == positions[b] && a.expansionDims[0] == capacities[b], "KV growth capacity/logical length");
            Require(Bytes(a) == Bytes(r), "KV growth changed committed prefix");
        }
    }
    std::vector<CloverSnapshots> snapshots(3), referenceSnapshots(3);
    std::vector<CloverSnapshots *> sp, rp;
    for (int b = 0; b < 3; ++b) { sp.push_back(&snapshots[b]); rp.push_back(&referenceSnapshots[b]); }
    model.ForwardLogitsBatch(tokens, ptrs, false, actual, sp);
    model.ForwardLogitsBatch(tokens, refs, false, expected, rp);
    for (int b = 0; b < 3; ++b) {
        Require(Bytes(actual[b]) == Bytes(expected[b]), "KV growth changed verify logits");
        int accepted = b == 0 ? 1 : b == 1 ? 16 : 32;
        model.RestorePrefix(caches[b], snapshots[b], accepted);
        model.RestorePrefix(reference[b], referenceSnapshots[b], accepted);
        Require(Bytes(caches[b][0].first) == Bytes(reference[b][0].first) &&
                Bytes(caches[b][0].second) == Bytes(reference[b][0].second), "KV growth rollback changed prefix");
        if (positions[b] + accepted + 2 <= limit) {
            Data a, r;
            model.ForwardLogits({11, 19}, caches[b], false, a);
            model.ForwardLogits({11, 19}, reference[b], false, r);
            Require(Bytes(a) == Bytes(r), "KV growth continuation changed logits");
        }
    }
}

}
int main(int argc, char **argv) {
    fp16 = argc > 1 && std::string(argv[1]) == "fp16";
    const char *env = std::getenv("FASTLLM_CUDA_GRAPH");
    const bool hadEnv = env != nullptr; const std::string old = env ? env : "";
    auto restore = [&]() { if (hadEnv) setenv("FASTLLM_CUDA_GRAPH", old.c_str(), 1); else unsetenv("FASTLLM_CUDA_GRAPH"); };
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        cudaSetDevice(0); SetThreads(2); SetDeviceMap({{"cuda:0", 1}}); SetCudaEmbedding(true);
        for (int batch : {2, 8, 33}) for (bool sample : {false, true}) SamplingMetadataBatch(batch, sample);
        SamplingMetadataBatch(64, false);
        for (bool sample : {false, true}) SamplingMetadataBatch(128, sample);
        SamplingMetadataBatch(256, false);
        CacheGrowth();
        ProjectionPrecision();
        FixedRowProjection();
        for (int seed : {13, 29, 71}) {
            auto batched = RunBatch(seed, false);
            Require(RunBatch(seed, true) == batched, "batch graph replay differs from eager");
            auto eager = Run(seed, false, false);
            Require(Run(seed, true, false) == eager, "graph replay differs from eager");
            Require(Run(seed, true, true) == eager, "capture-unavailable fallback differs from eager");
        }
        for (int batch : {3, 4, 8, 9, 32, 50, 64, 65, 128, 256})
            Require(RunBatch(13, true, batch) == RunBatch(13, false, batch), "larger batch graph replay differs from eager");
        Require(RunBatch(13, true, 64, true) == RunBatch(13, false, 64), "64-batch capture fallback differs from eager");
        Require(RunBatch(13, true, 256, false, 8) == RunBatch(13, false, 256, false, 8),
            "large routed-task tiling differs from eager");
        restore();
        std::cout << "Clover MoE graphs: changing inputs/widths, model destruction, read-only preview KV, verification and capture fallback PASS" << std::endl;
    } catch (const std::exception &e) { restore(); std::cerr << e.what() << std::endl; return 1; }
}
