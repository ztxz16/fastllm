#include "models/clover.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fastllm;
namespace {
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
uint16_t Bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    bits += 0x7fff + ((bits >> 16) & 1);
    return bits >> 16;
}
void Populate(CloverModel &m, int seed) {
    m.weight.dicts = {{"hidden_size", "512"}, {"num_hidden_layers", "1"},
        {"num_attention_heads", "1"}, {"num_key_value_heads", "1"},
        {"head_dim", "256"}, {"moe_intermediate_size", "128"},
        {"num_experts", "16"}, {"num_experts_per_tok", "4"},
        {"max_position_embeddings", "1024"}, {"vocab_size", "128"},
        {"mask_token_id", "127"}, {"layer_types", "[\"full_attention\"]"}};
    m.InitParams();
    std::mt19937 rng(seed); std::normal_distribution<float> normal(0.f, .03f);
    auto weight = [&](const std::string &name, const std::vector<int> &shape, bool norm = false) {
        auto &w = m.weight[name]; w.dataType = norm ? FLOAT32 : BFLOAT16;
        w.UpdateUnitSize(); w.Resize(shape); w.Allocate(false);
        for (int i = 0; i < w.Count(0); ++i) {
            float value = normal(rng);
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
    m.WarmUp();
}
std::vector<unsigned char> Bytes(const Data &x) {
    Data host; host.CopyFrom(x); host.ToDevice(DataDevice::CPU);
    return {host.cpuData, host.cpuData + host.GetBytes()};
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
std::vector<std::vector<unsigned char>> RunBatch(int seed, bool graphs, int batch = 2) {
    setenv("FASTLLM_CUDA_GRAPH", graphs ? "1" : "0", 1);
    CloverModel model; Populate(model, seed);
    using Cache = std::vector<std::pair<Data, Data>>;
    std::vector<Cache> caches(batch), reference(batch);
    for (int b = 0; b < batch; ++b) {
        Data ignored;
        std::vector<int> prefix(5 + 3 * b);
        for (int i = 0; i < (int)prefix.size(); ++i) prefix[i] = (b * 31 + i * 7) % 127;
        model.ForwardLogits(prefix, caches[b], false, ignored);
    }
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
        Require(maximum <= .04f && diff2 <= 2.5e-5 * std::max(ref2, 1e-10), "batch/serial BF16 logits differ");
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
    return results;
}

}
int main() {
    const char *env = std::getenv("FASTLLM_CUDA_GRAPH");
    const bool hadEnv = env != nullptr; const std::string old = env ? env : "";
    auto restore = [&]() { if (hadEnv) setenv("FASTLLM_CUDA_GRAPH", old.c_str(), 1); else unsetenv("FASTLLM_CUDA_GRAPH"); };
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        cudaSetDevice(0); SetThreads(2); SetDeviceMap({{"cuda:0", 1}}); SetCudaEmbedding(true);
        for (int seed : {13, 29, 71}) {
            auto batched = RunBatch(seed, false);
            Require(RunBatch(seed, true) == batched, "batch graph replay differs from eager");
            auto eager = Run(seed, false, false);
            Require(Run(seed, true, false) == eager, "graph replay differs from eager");
            Require(Run(seed, true, true) == eager, "capture-unavailable fallback differs from eager");
        }
        for (int batch : {3, 4, 8})
            Require(RunBatch(13, true, batch) == RunBatch(13, false, batch), "larger batch graph replay differs from eager");
        restore();
        std::cout << "Clover MoE graphs: changing inputs/widths, model destruction, read-only preview KV, verification and capture fallback PASS" << std::endl;
    } catch (const std::exception &e) { restore(); std::cerr << e.what() << std::endl; return 1; }
}
