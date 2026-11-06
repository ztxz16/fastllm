#include "models/deepseekv41.h"
#include "devices/cpu/cpudevice.h"
#include "gguf.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace fastllm;
using Json = json11::Json;
namespace {
void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

struct Model : DeepSeekV41Model {
    void ConfigureFixture() {
        embed_dim = 32; hc_mult = 2;
        engram_layer_ids = {1, 3}; engram_num_embeddings = {26, 72};
        engram_max_ngram_size = 3; engram_n_heads = 2; engram_head_dim = 32;
        engram_pad_token_id = 2; engram_compressed_vocab_size = 3;
        weight.AddDict("gguf_architecture", "deepseek41");
        weight.AddDict("ple.layers", "[1,3]"); // Preserved by LoadGGUFModelConfig.
        weight.AddDict("model_directory", "/tmp/");
    }
    void CheckMeta() {
        LoadEngramMeta();
        Check(engramMeta.loaded && engramMeta.tokenMap == std::vector<int>({0, 1, 2}), "token map");
        Check(engramMeta.multipliers[0][0] == 9007199254740993LL, "64-bit hash multiplier rounded");
        Check(engramMeta.offsets[0] == std::vector<int64_t>({0, 3, 8, 15}), "first table offsets");
        Check(engramMeta.offsets[1] == std::vector<int64_t>({0, 13, 30, 49}), "second table offsets");
    }
};

Json Metadata() {
    Json::object p;
    Json::object fields = {
        {"block_count", 4}, {"attention.kv_source_layers", Json::array{0, 2}},
        {"attention.indexer.source_layers", Json::array{0, 2}},
        {"attention.indexer.candidate_source_layer", 2},
        {"attention.indexer.candidate_top_k_blocks", 8}, {"attention.indexer.candidate_block_size", 4},
        {"expert_gating_func", 4}, {"ple.layers", Json::array{1, 3}},
        {"ple.ngram_size", 3}, {"ple.heads_per_ngram", 2}, {"embedding_length_per_layer_input", 32},
        {"engram.pad_token_id", 2}, {"engram.token_map", Json::array{0, 1, 2}},
        {"ple.head_vocab_sizes", Json::array{3, 5, 7, 11, 13, 17, 19, 23}},
        {"ple.head_offsets", Json::array{0, 3, 8, 15, 26, 39, 56, 75}},
        {"ple.layer_multipliers", Json::array{9007199254740993LL, 3LL, 5LL, 7LL, 11LL, 13LL}},
    };
    for (const auto &item : fields) p["deepseek41." + item.first] = item.second;
    return p;
}

ggml_tensor Tensor(const std::string &name, ggml_type type, int rows, int dim) {
    ggml_tensor t;
    t.name = name; t.type = type; t.dims = {rows, dim};
    t.ne[0] = dim; t.ne[1] = rows; t.ne[2] = t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type); t.nb[1] = ggml_row_size(type, dim);
    t.nb[2] = t.nb[3] = rows * t.nb[1];
    return t;
}

void GateEquivalence(bool cuda = false) {
    const int dim = 32, hc = 2, tokens = 3;
    std::vector<float> h(tokens * hc * dim), v(tokens * (hc + 1) * dim);
    std::vector<float> q(hc * dim), k(hc * dim), product(hc * dim);
    for (size_t i = 0; i < h.size(); ++i) h[i] = float(int(i % 37) - 18) / 32;
    for (size_t i = 0; i < v.size(); ++i) v[i] = float(int(i % 29) - 14) / 16;
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = float(int(i % 13) - 6) / 8; k[i] = float(int(i % 17) - 8) / 4;
        product[i] = q[i] * k[i];
    }
    const DataType hiddenType = cuda ? BFLOAT16 : FLOAT32;
    Data oldHidden(hiddenType, {1, tokens, hc, dim}, h), newHidden(hiddenType, {1, tokens, hc, dim}, h);
    Data kv(FLOAT32, {1, tokens, (hc + 1) * dim}, v), qw(FLOAT32, {hc, dim}, q), kw(FLOAT32, {hc, dim}, k);
    Data fused(FLOAT32, {hc, dim}, product), ones(FLOAT32, {hc, dim}, std::vector<float>(hc * dim, 1));
    if (cuda) {
#ifdef USE_CUDA
        for (Data *d : {&oldHidden, &newHidden, &kv, &qw, &kw, &fused, &ones}) d->ToDevice(DataDevice::CUDA, {0});
        Check(FastllmCudaDeepSeekV41EngramApply(oldHidden, kv, qw, kw, nullptr, 1e-20f, 1e-6f), "CUDA legacy gate");
        Check(FastllmCudaDeepSeekV41EngramApply(newHidden, kv, fused, ones, nullptr, 1e-20f, 1e-6f), "CUDA fused gate");
        oldHidden.ToDevice(DataDevice::CPU); newHidden.ToDevice(DataDevice::CPU);
#else
        throw std::runtime_error("CUDA test requires a CUDA build");
#endif
    } else {
        CpuDeepSeekV41EngramApplyOp op;
        BaseOperator &executor = *(BaseOperator*)&op;
        executor.Run("DeepSeekV41EngramApply", {{"hidden", &oldHidden}, {"kv", &kv}, {"qWeight", &qw}, {"kWeight", &kw}}, {{"eps", 1e-20f}}, {});
        executor.Run("DeepSeekV41EngramApply", {{"hidden", &newHidden}, {"kv", &kv}, {"qWeight", &fused}, {"kWeight", &ones}}, {{"eps", 1e-20f}}, {});
    }
    Check(!std::memcmp(oldHidden.cpuData, newHidden.cpuData, oldHidden.GetBytes()), "fused gate arithmetic changed");
}
}

int main(int argc, char **argv) {
    try {
        SetThreads(2);
        Model model;
        Json meta = Metadata();
        LoadDeepSeekV41UpstreamGGUFConfig(&model, meta);
        model.ConfigureFixture();
        model.CheckMeta();
        std::string error;
        const auto rows = Json::parse(model.weight.dicts["engram_num_embeddings"], error);
        Check(error.empty() && rows.array_items().size() == 2 && rows[0].int_value() == 26 &&
              rows[1].int_value() == 72, "Engram rows");
        char path[] = "/tmp/fastllm-v41-upstream-XXXXXX";
        int fd = mkstemp(path); Check(fd >= 0, "temporary table"); close(fd);
        std::vector<uint8_t> payload(64 + 98 * 18);
        for (size_t i = 64; i < payload.size(); ++i) payload[i] = uint8_t(i * 31 + 7);
        { std::ofstream f(path, std::ios::binary); f.write((char*)payload.data(), payload.size()); }
        std::vector<ReadGGUFTask> tasks;
        std::string tableName = "engram_combined.weight";
        tasks.emplace_back(tableName, nullptr, Tensor("per_layer_token_embd.weight", GGML_TYPE_IQ4_NL, 98, 32), path, 64);
        for (int layer : {1, 3}) {
            std::string name = "layers." + std::to_string(layer) + ".engram.q_weight";
            tasks.emplace_back(name, nullptr, Tensor("blk." + std::to_string(layer) + ".engram_gate.weight", GGML_TYPE_F32, 2, 32), path, 0);
            model.weight[name].CopyFrom(Data(FLOAT32, {2, 32}, std::vector<float>(64, .5f)));
        }
        PrepareDeepSeekV41UpstreamGGUFTasks(&model, meta, tasks);
        Check(tasks.size() == 4 && tasks[0].offset == 64 && tasks[1].offset == 64 + 26 * 18,
              "combined table view offsets");
        for (int i = 0; i < 2; ++i) {
            auto &task = tasks[i];
            WeightImportGGUFTensor(&model.weight[task.name], &task.tensor, task.fileName, task.offset, task.replaceType);
            Check(!std::memcmp(model.weight[task.name].cpuData, payload.data() + task.offset, ggml_nbytes(&task.tensor)),
                  "combined table view bytes");
            Check(model.ngramWeights.count(task.name), "lookup table placement registration");
        }
        model.OnModelWeightsLoaded();
        model.OnModelWeightsLoaded(); // Repeated notification must be harmless.
        for (int layer : {1, 3}) {
            const std::string name = "layers." + std::to_string(layer) + ".engram.k_weight";
            Check(model.weight.weight.count(name), "missing neutral gate parameter");
            Data &k = model.weight[name];
            Check(k.cpuData && k.Count(0) == 64, "neutral gate storage");
            for (size_t i = 0; i < k.Count(0); ++i) Check(((float*)k.cpuData)[i] == 1, "neutral gate parameter");
        }
        unlink(path);
        GateEquivalence();
        if (argc > 1 && std::string(argv[1]) == "--cuda") GateEquivalence(true);
        std::cout << "PASS: upstream GGUF hash precision, table views, placement, idempotence and gate equivalence\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
