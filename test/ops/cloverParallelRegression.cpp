#include "models/clover.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
using namespace fastllm;
namespace {
using Cache = std::vector<std::pair<Data, Data>>;
void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
uint16_t Bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4); bits += 0x7fff + ((bits >> 16) & 1); return bits >> 16;
}
void AllReduceChecks() {
    Check(FastllmInitNccl({0, 1}), "NCCL initialization failed");
    FastllmCudaSetNcclForceSync(false);
    std::exception_ptr errors[2];
    auto worker = [&](int rank) { try {
        Check(cudaSetDevice(rank) == cudaSuccess, "set rank");
        auto cuda = [](cudaError_t e) { Check(e == cudaSuccess, cudaGetErrorString(e)); };
        auto encode = [](float v, int type, unsigned char *p) {
            if (type == FLOAT32) std::memcpy(p, &v, 4);
            else { uint16_t bits = type == FLOAT16 ? __half_raw(__float2half_rn(v)).x : Bf16(v); std::memcpy(p, &bits, 2); }
        };
        auto round = [](float v, int type) {
            if (type == FLOAT16) return __half2float(__float2half_rn(v));
            if (type == BFLOAT16) { uint32_t bits = uint32_t(Bf16(v)) << 16; std::memcpy(&v, &bits, 4); }
            return v;
        };
        auto value = [](int i, int rank, int variant) { return .007f * ((i % 32) - 16 + rank * 3 + variant); };
        for (int type : {int(BFLOAT16), int(FLOAT16), int(FLOAT32)})
        for (int bytes : {16, 1024, 294912, 294928, 1048576, 4194304, 8388608})
        for (bool inplace : {false, true}) {
            int unit = type == FLOAT32 ? 4 : 2, count = bytes / unit;
            unsigned char *seed, *source, *target, *samples;
            cuda(cudaMalloc(&seed, bytes * 2)); cuda(cudaMalloc(&source, bytes + 32));
            cuda(cudaMalloc(&target, bytes + 32)); cuda(cudaMalloc(&samples, 16 * 48));
            std::vector<unsigned char> host(bytes * 2);
            for (int variant = 0; variant < 2; ++variant)
                for (int i = 0; i < count; ++i)
                    encode(value(i, rank, variant), type, host.data() + variant * bytes + i * unit);
            cuda(cudaMemcpy(seed, host.data(), host.size(), cudaMemcpyHostToDevice));
            cuda(cudaMemsetAsync(source, 0xa5, bytes + 32, cudaStreamPerThread));
            cuda(cudaMemsetAsync(target, 0xa5, bytes + 32, cudaStreamPerThread));
            auto *destination = inplace ? source + 16 : target + 16;
            // No intervening stream sync: a fast rank must not overwrite the
            // staged input while its peer is still reading the preceding call.
            for (int iteration = 0; iteration < 16; ++iteration) {
                cuda(cudaMemcpyAsync(source + 16, seed + (iteration % 2) * bytes, bytes,
                                     cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                Check(FastllmTryTP2P2PAllReduce(source + 16, destination, count, type, rank), "P2P sum dispatch");
                Check(FastllmTryTP2P2PAllReduce(destination, destination, count, type, rank), "P2P repeated sum");
                int offsets[] = {0, (bytes / 32) * 16, bytes - 16};
                for (int j = 0; j < 3; ++j)
                    cuda(cudaMemcpyAsync(samples + iteration * 48 + j * 16, destination + offsets[j], 16,
                                         cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            }
            cuda(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<unsigned char> actual(16 * 48), expected(actual.size());
            cuda(cudaMemcpy(actual.data(), samples, actual.size(), cudaMemcpyDeviceToHost));
            for (int iteration = 0; iteration < 16; ++iteration) {
                int offsets[] = {0, (bytes / 32) * 16, bytes - 16};
                for (int j = 0; j < 3; ++j) for (int k = 0; k < 16 / unit; ++k) {
                    int i = offsets[j] / unit + k;
                    float sum = round(round(value(i, 0, iteration % 2), type) +
                                      round(value(i, 1, iteration % 2), type), type);
                    float v = round(sum + sum, type);
                    encode(v, type, expected.data() + iteration * 48 + j * 16 + k * unit);
                }
            }
            Check(actual == expected, "P2P sum bits differ from independent reference");
            for (auto *buffer : {source, target}) {
                unsigned char guards[32]; cuda(cudaMemcpy(guards, buffer, 16, cudaMemcpyDeviceToHost));
                cuda(cudaMemcpy(guards + 16, buffer + bytes + 16, 16, cudaMemcpyDeviceToHost));
                for (auto v : guards) Check(v == 0xa5, "P2P output guard overwritten");
            }
            if (bytes == 8388608) {
                Check(!FastllmTryTP2P2PAllReduce(source + 16 + (rank ? 0 : unit), target + 16, count, type, rank), "unaligned peer fallback");
                Check(!FastllmTryTP2P2PAllReduce(source + 16, target + 16, count + 16, type, rank), "oversize fallback");
            }
            cuda(cudaFree(seed)); cuda(cudaFree(source)); cuda(cudaFree(target)); cuda(cudaFree(samples));
        }
    } catch (...) { errors[rank] = std::current_exception(); FastllmNcclAbort("P2P regression failed"); } };
    std::thread a(worker, 0), b(worker, 1); a.join(); b.join();
    for (auto &error : errors) if (error) std::rethrow_exception(error);
    Check(cudaSetDevice(0) == cudaSuccess, "fallback device");
    void *data = nullptr; Check(cudaMalloc(&data, 4096) == cudaSuccess, "fallback allocation");
    FastllmCudaSetNcclForceSync(true);
    Check(!FastllmTryTP2P2PAllReduce(data, data, 2048, BFLOAT16, 0), "synchronous fallback");
    FastllmCudaSetNcclForceSync(false);
    Check(!FastllmTryTP2P2PAllReduce(data, data, 7, BFLOAT16, 0), "unaligned size fallback");
    Check(!FastllmTryTP2P2PAllReduce(data, data, 1024, INT32, 0), "unsupported type fallback");
    Check(FastllmCudaGraphBeginCapture(), "fallback capture begin");
    Check(!FastllmTryTP2P2PAllReduce(data, data, 2048, BFLOAT16, 0), "capture fallback");
    void *graph = nullptr; Check(FastllmCudaGraphEndCapture(&graph), "fallback capture end");
    if (graph) FastllmCudaGraphDestroy(graph);
    Check(cudaFree(data) == cudaSuccess, "fallback free");
    std::cout << "TP2 allreduce BF16/FP16/FP32, in-place, guards, asynchronous repeats and fallbacks PASS" << std::endl;
}
void Populate(CloverModel &m, const std::string &devices, bool fp16) {
    m.weight.dicts = {{"hidden_size", "512"}, {"num_hidden_layers", "2"},
        {"num_attention_heads", "4"}, {"num_key_value_heads", "2"}, {"head_dim", "256"},
        {"linear_num_key_heads", "4"}, {"linear_num_value_heads", "8"},
        {"moe_intermediate_size", "512"}, {"shared_expert_intermediate_size", "512"},
        {"num_experts", "16"}, {"num_experts_per_tok", "4"}, {"max_position_embeddings", "1024"},
        {"vocab_size", "128"}, {"mask_token_id", "127"}, {"dllm_think_end_token_id", "17"},
        {"layer_types", "[\"linear_attention\",\"full_attention\"]"}};
    m.InitParams();
    // InitParams imports the global placement; override it afterwards.
    m.deviceMap = {{devices, 1}};
    std::mt19937 rng(251); std::normal_distribution<float> normal(0.f, .02f);
    auto add = [&](const std::string &name, std::vector<int> shape, bool scalar = false) {
        auto &w = m.weight[name]; w.name = name; w.dataType = scalar ? FLOAT32 : BFLOAT16;
        w.UpdateUnitSize(); w.Resize(shape); w.Allocate(false);
        for (size_t i = 0; i < w.Count(0); ++i) {
            float value = normal(rng);
            if (name.find("linear_attn.norm.weight") != std::string::npos) value += 1;
            // Keep routing well separated when comparing TP partitions.
            // Random near-ties can select different experts after legitimate
            // 16-bit reduction rounding and obscure cache/addressing errors.
            if (name.find(".mlp.gate.weight") != std::string::npos)
                value = i % shape.back() ? 0.f : float(i / shape.back() + 1) / 16;
            if (name == "model.embed_tokens.weight" && i % shape.back() == 0)
                value = (i / shape.back()) % 2 ? .5f : -.5f;
            if (scalar) ((float *)w.cpuData)[i] = value;
            else ((uint16_t *)w.cpuData)[i] = Bf16(value);
        }
    };
    add("model.embed_tokens.weight", {128, 512}); add("lm_head.weight", {128, 512});
    add("model.norm.weight", {512}, true);
    for (int layer = 0; layer < 2; ++layer) {
        std::string p = "model.layers." + std::to_string(layer) + ".";
        add(p + "input_layernorm.weight", {512}, true);
        add(p + "post_attention_layernorm.weight", {512}, true);
        if (!layer) {
            const auto a = p + "linear_attn.";
            add(a + "in_proj_qkv.weight", {2048, 512}); add(a + "in_proj_z.weight", {1024, 512});
            add(a + "in_proj_a.weight", {8, 512}); add(a + "in_proj_b.weight", {8, 512});
            add(a + "out_proj.weight", {512, 1024});
            add(a + "conv1d.weight", {2048, 1, 4}); add(a + "conv_right_weight", {2048, 1, 3});
            for (auto name : {"A_log", "dt_bias", "bd_softmax_logit_scale", "bd_softmax_history_bias", "bd_softmax_history_c"})
                add(a + name, {8}, true);
            add(a + "norm.weight", {128}, true);
        } else {
            const auto a = p + "self_attn.";
            add(a + "q_proj.weight", {2048, 512}); add(a + "k_proj.weight", {512, 512});
            add(a + "v_proj.weight", {512, 512}); add(a + "o_proj.weight", {512, 1024});
            add(a + "q_norm.weight", {256}, true); add(a + "k_norm.weight", {256}, true);
        }
        add(p + "mlp.gate.weight", {16, 512}); add(p + "mlp.experts.gate_up_proj", {16, 1024, 512});
        add(p + "mlp.experts.down_proj", {16, 512, 512});
        add(p + "mlp.shared_expert.gate_proj.weight", {512, 512});
        add(p + "mlp.shared_expert.up_proj.weight", {512, 512});
        add(p + "mlp.shared_expert.down_proj.weight", {512, 512});
        add(p + "mlp.shared_expert_gate.weight", {1, 512});
    }
    if (fp16) m.SetDataType(FLOAT16);
    // Match ftllm serve: AutoWarmup releases warmup-only NCCL synchronization.
    m.AutoWarmup();
}
void CloneCache(const Data &src, Data &dst) {
    if (!src.multiDeviceData) { dst.CopyFrom(src); return; }
    dst.dataType = src.dataType; dst.UpdateUnitSize(); dst.Resize(src.dims); dst.multiDeviceData = true;
    for (auto &entry : src.multiDeviceDatas) {
        auto &child = dst.multiDeviceDatas[entry.first]; if (!child) child = new Data();
        CloneCache(*entry.second, *child);
    }
}
void CloneCache(const Cache &src, Cache &dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) { CloneCache(src[i].first, dst[i].first); CloneCache(src[i].second, dst[i].second); }
}
void Close(const Data &a, const Data &b, double tolerance = .04) {
    Check(a.dims == b.dims, "comparison shape mismatch");
    if (a.multiDeviceData) {
        Check(a.multiDeviceDatas.size() == b.multiDeviceDatas.size(), "cache ranks differ");
        for (auto &entry : a.multiDeviceDatas) Close(*entry.second, *b.multiDeviceDatas.at(entry.first), tolerance);
        return;
    }
    Data ah, bh; ah.CopyFrom(a); bh.CopyFrom(b);
    ah.ToDevice(DataDevice::CPU); bh.ToDevice(DataDevice::CPU);
    ToDataTypeForceCPU(ah, FLOAT32); ToDataTypeForceCPU(bh, FLOAT32);
    auto *x = (float *)ah.cpuData, *y = (float *)bh.cpuData;
    double error = 0, energy = 0, maximum = 0;
    for (size_t i = 0; i < ah.Count(0); ++i) {
        Check(std::isfinite(x[i]) && std::isfinite(y[i]), "nonfinite result");
        double d = x[i] - y[i]; error += d * d; energy += y[i] * y[i]; maximum = std::max(maximum, std::abs(d));
    }
    double relative = std::sqrt(error / std::max(energy, 1e-30));
    if (relative > tolerance) std::cerr << "relative error=" << relative << " max_abs=" << maximum << std::endl;
    Check(relative <= tolerance, "numerical comparison failed");
}
void Close(const Cache &a, const Cache &b, double tolerance) {
    for (size_t i = 0; i < a.size(); ++i) { Close(a[i].first, b[i].first, tolerance); Close(a[i].second, b[i].second, tolerance); }
}
std::vector<int> Tokens(int count, int seed) {
    std::vector<int> values(count); for (int i = 0; i < count; ++i) values[i] = (seed + i * 17) % 127; return values;
}
void ForwardChecks(CloverModel &single, CloverModel &tp, int ranks) {
    Cache a, b; Data x, y;
    single.ForwardLogits(Tokens(301, 3), a, false, x, true);
    tp.ForwardLogits(Tokens(301, 3), b, false, y, true);
    Check(b[0].first.multiDeviceData && b[0].first.multiDeviceDatas.size() == size_t(ranks),
        "TP regression did not create per-rank caches");
    Close(x, y);
    Cache before; CloneCache(b, before);
    for (int width : {1, 2, 8, 16, 32, 16}) {
        single.ForwardLogits(Tokens(width, 19), a, true, x);
        tp.ForwardLogits(Tokens(width, 19), b, true, y);
        Close(x, y); Close(b, before, 0);
    }
    for (int batch : {2, 8, 64}) {
        std::vector<Cache> caches(batch), refs(batch), serial(batch);
        std::vector<Cache *> cv;
        std::vector<CloverSnapshots> snapshots(batch); std::vector<CloverSnapshots *> sv;
        std::vector<std::vector<int>> tokens;
        for (int i = 0; i < batch; ++i) {
            CloneCache(before, caches[i]); CloneCache(before, refs[i]); cv.push_back(&caches[i]);
            snapshots[i].layers.resize(2);
            // Full attention has no state snapshot; an empty stride vector
            // must never be passed through Data::Resize during publication.
            for (Data *empty : {&snapshots[i].layers[1].first, &snapshots[i].layers[1].second}) {
                empty->strides.clear(); empty->strides.shrink_to_fit();
            }
            sv.push_back(&snapshots[i]); tokens.push_back(Tokens(16, i * 7));
        }
        std::vector<Data> outputs;
        tp.ForwardLogitsBatch(tokens, cv, true, outputs);
        for (int i = 0; i < batch; ++i) Close(caches[i], refs[i], 0);
        tp.ForwardLogitsBatch(tokens, cv, false, outputs, sv);
        std::vector<int> acceptedInputs;
        const int lengths[] = {1, 2, 7, 8, 15, 16};
        for (int i = 0; i < batch; ++i) {
            acceptedInputs.push_back(lengths[i % 6]);
            CloneCache(caches[i], serial[i]);
            tp.RestorePrefix(serial[i], snapshots[i], acceptedInputs[i]);
        }
        tp.RestorePrefixBatch(cv, {sv.begin(), sv.end()}, acceptedInputs);
        for (int i = 0; i < batch; ++i) {
            int accepted = acceptedInputs[i];
            Close(caches[i], serial[i], 0);
            tp.ForwardLogits(std::vector<int>(tokens[i].begin(), tokens[i].begin() + accepted), refs[i], false, y);
            Close(caches[i], refs[i], .015);
            // Next-token comparisons need the same verification geometry:
            // prefix-only FP16 GEMMs can round differently at router ties.
            Cache reference; CloneCache(before, reference);
            CloverSnapshots referenceSnapshots;
            tp.ForwardLogits(tokens[i], reference, false, y, false, &referenceSnapshots);
            Close(outputs[i], y, .015);
            tp.RestorePrefix(reference, referenceSnapshots, accepted);
            Close(caches[i], reference, .015);
            tp.ForwardLogits({21}, caches[i], false, x);
            tp.ForwardLogits({21}, reference, false, y); Close(x, y, .015);
        }
    }
}
void GenerationChecks(CloverModel &m) {
    // Repeated prompts exercise host prefix snapshots for every rank. Seeded
    // sampling must replay the same output when restored from a cached prompt.
    m.maxBatch = 2;
    for (bool sample : {false, true}) {
        auto config = json11::Json::object{{"mode", "self_speculative"}, {"block_size", 16},
            {"max_new_tokens", 35}, {"temperature", sample ? .8 : 0.}, {"do_sample", sample},
            {"seed", 31}, {"top_k", -1}, {"top_p", .95}, {"ignore_eos", true}};
        auto collect = [&](bool expectHit) {
            int handle = m.LaunchClover(Tokens(97, sample ? 9 : 5), config);
            std::vector<int> result;
            int cached = 0;
            for (;;) {
                int tokens[64], stats[3]; int n = m.FetchResponseTokensBatch(handle, tokens, 64, stats);
                cached = std::max(cached, stats[0]);
                if (n == -1) break;
                Check(n >= 0, "generation failed"); result.insert(result.end(), tokens, tokens + n);
            }
            Check(result.size() == 35, "incomplete generation");
            Check(!expectHit || cached == 97, "repeated prompt did not hit prefix cache");
            return result;
        };
        auto first = collect(false); auto second = collect(true); Check(first == second, "prefix restore changed seeded generation");
    }
}
}
int main(int argc, char **argv) {
    try {
        int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) return 77;
        const std::string mode = argc > 1 ? argv[1] : "bf16";
        if (mode == "allreduce" || mode == "abort") {
            for (int rank = 0; rank < 2; ++rank) {
                int peer = 0;
                if (cudaDeviceCanAccessPeer(&peer, rank, 1 - rank) != cudaSuccess || !peer) return 77;
            }
        }
        if (mode == "allreduce") { AllReduceChecks(); return 0; }
        if (mode == "abort") {
            Check(FastllmInitNccl({0, 1}), "NCCL initialization failed");
            FastllmCudaSetNcclForceSync(false);
            bool failed = false;
            std::thread rank([&] {
                FastllmCudaSetDevice(0);
                Data data(BFLOAT16, {16, 512});
                data.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); data.Allocate(false);
                FastllmTryTP2P2PAllReduce(data.cudaData, data.cudaData, data.Count(0), BFLOAT16, 0);
                failed = FastllmCudaGetThreadError();
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            FastllmNcclAbort("injected peer failure");
            rank.join();
            Check(failed, "P2P wait ignored peer failure");
            std::cout << "Clover TP P2P cancellation PASS" << std::endl;
            return 0;
        }
        bool fp16 = mode == "fp16";
        int ranks = argc > 2 ? std::stoi(argv[2]) : 2;
        if (devices < ranks) return 77;
        SetThreads(4); SetCudaEmbedding(true); SetDeviceMap({{"cuda:0", 1}});
        setenv("FASTLLM_PREFIX_CACHE", "1", 1);
        CloverModel single, tp; Populate(single, "cuda:0", fp16);
        std::string spec = "cuda:0"; for (int r = 1; r < ranks; ++r) spec += "," + std::to_string(r);
        Populate(tp, spec, fp16);
        ForwardChecks(single, tp, ranks); GenerationChecks(tp);
        std::cout << "Clover TP" << ranks << (fp16 ? " FP16" : " BF16") << " PASS: logits, preview, full/compact rollback, prefix cache and seeded sampling" << std::endl;
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
