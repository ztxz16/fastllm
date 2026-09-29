#include "model.h"
#include "models/qwen4_exp.h"
#include "devices/disk/diskdevice.h"
#include "gguf.h"
#include "json11.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace fastllm;
using Json = json11::Json;
namespace {
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
using Bytes = std::vector<uint8_t>;
template<class T> void Put(Bytes &out, T value) {
    const auto *p = reinterpret_cast<const uint8_t *>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}
void String(Bytes &out, const std::string &s) {
    Put<uint64_t>(out, s.size()); out.insert(out.end(), s.begin(), s.end());
}
void Value(Bytes &out, const Json &value) {
    if (value.is_string()) { Put<uint32_t>(out, 8); String(out, value.string_value()); }
    else if (value.is_array()) {
        const bool strings = !value.array_items().empty() && value[0].is_string();
        Put<uint32_t>(out, 9); Put<uint32_t>(out, strings ? 8 : 10);
        Put<uint64_t>(out, value.array_items().size());
        for (const auto &n : value.array_items()) {
            if (strings) String(out, n.string_value());
            else Put<uint64_t>(out, n.number_value());
        }
    } else { Put<uint32_t>(out, 10); Put<uint64_t>(out, value.number_value()); }
}
struct Tensor { std::string name; std::vector<int> dims; ggml_type type; Bytes bytes; };
Tensor FloatTensor(const std::string &name, const std::vector<int> &dims, ggml_type type,
                   const std::vector<float> &values) {
    Tensor t{name, dims, type, {}};
    for (float value : values) {
        if (type == GGML_TYPE_F32) Put(t.bytes, value);
        else if (type == GGML_TYPE_F16) Put<uint16_t>(t.bytes, float_to_half(value));
        else { uint32_t bits; std::memcpy(&bits, &value, 4); Put<uint16_t>(t.bytes, bits >> 16); }
    }
    return t;
}
void Write(const std::string &path, const Json::object &metadata, const std::vector<Tensor> &tensors) {
    Bytes header, payload;
    Put<uint32_t>(header, 0x46554747); Put<uint32_t>(header, 3);
    Put<uint64_t>(header, tensors.size()); Put<uint64_t>(header, metadata.size());
    for (const auto &item : metadata) { String(header, item.first); Value(header, item.second); }
    for (const auto &t : tensors) {
        String(header, t.name); Put<uint32_t>(header, t.dims.size());
        for (auto it = t.dims.rbegin(); it != t.dims.rend(); ++it) Put<uint64_t>(header, *it);
        Put<uint32_t>(header, t.type); Put<uint64_t>(header, payload.size());
        payload.insert(payload.end(), t.bytes.begin(), t.bytes.end());
        payload.resize((payload.size() + 31) / 32 * 32);
    }
    header.resize((header.size() + 31) / 32 * 32);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(header.data()), header.size());
    file.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    Check(file.good(), "fixture write failed");
}
struct Fixture {
    std::string directory;
    std::vector<std::string> files;
    Fixture() {
        char name[] = "/tmp/fastllm-qwen4-gguf-XXXXXX";
        Check(mkdtemp(name) != nullptr, "fixture directory failed"); directory = name;
        for (int i = 1; i <= 3; ++i) files.push_back(directory + "/tiny-0000" + std::to_string(i) + "-of-00003.gguf");
        Json::object meta = {{"general.architecture", "qwen4exp"}, {"general.name", "Qwen4 GGUF regression"},
            {"tokenizer.ggml.eos_token_id", 8}, {"tokenizer.ggml.bos_token_id", 6}};
        Json::array tokens;
        for (int i = 0; i < 32; ++i) tokens.emplace_back("token" + std::to_string(i));
        meta["tokenizer.ggml.tokens"] = tokens;
        meta["tokenizer.ggml.model"] = "gpt2";
        const Json::object config = {{"block_count", 2}, {"embedding_length", 256}, {"vocab_size", 32},
            {"attention.head_count", 2}, {"attention.head_count_kv", 1}, {"attention.key_length", 4},
            {"expert_count", 2}, {"expert_used_count", 1}, {"expert_feed_forward_length", 32},
            {"ssm.group_count", 2}, {"ssm.time_step_rank", 6}, {"ssm.state_size", 4},
            {"ssm.inner_size", 24}, {"ssm.conv_kernel", 4},
            {"hyper_connection.count", 2}, {"hyper_connection.low_rank", 4},
            {"attention.indexer.head_count", 2}, {"attention.indexer.key_length", 4},
            {"attention.indexer.top_k", 8}, {"attention.compress_ratios", Json::array{0, 2}},
            {"ple.layers", Json::array{1}}, {"ple.ngram_size", 3}, {"ple.heads_per_ngram", 1},
            {"embedding_length_per_layer_input", 32}, {"ple.conv_kernel", 4}, {"ple.eos_token_id", 6},
            {"ple.layer_multipliers", Json::array{23703573157769.0, 20109073645365.0, 8052911324071.0}},
            {"ple.head_offsets", Json::array{0, 5}}, {"ple.head_vocab_sizes", Json::array{5, 7}}};
        for (const auto &item : config) meta["qwen4exp." + item.first] = item.second;
        Write(files[0], meta, {}); // Real distribution has a metadata-only first shard.
        std::vector<Tensor> tensors;
        auto rows = [&](const std::string &name, int prefix, int headDim, int columns, ggml_type type) {
            const int count = prefix + 6 * headDim;
            std::vector<float> values(count * columns);
            for (int r = 0; r < prefix; ++r) for (int c = 0; c < columns; ++c) values[r * columns + c] = -1;
            // Encode the original grouped heads into llama.cpp's tiled order.
            for (int key = 0; key < 2; ++key) for (int v = 0; v < 3; ++v)
                for (int d = 0; d < headDim; ++d) for (int c = 0; c < columns; ++c)
                    values[(prefix + (v * 2 + key) * headDim + d) * columns + c] = key * 3 + v + 1;
            tensors.push_back(FloatTensor(name, columns == 1 ? std::vector<int>{count} :
                std::vector<int>{count, columns}, type, values));
        };
        rows("blk.0.attn_qkv.weight", 16, 4, 256, GGML_TYPE_F16);
        rows("blk.0.attn_gate.weight", 0, 4, 256, GGML_TYPE_F32);
        rows("blk.0.ssm_alpha.weight", 0, 1, 256, GGML_TYPE_F32);
        rows("blk.0.ssm_beta.weight", 0, 1, 256, GGML_TYPE_F32);
        rows("blk.0.ssm_conv1d.weight", 16, 4, 4, GGML_TYPE_F32);
        rows("blk.0.ssm_dt.bias", 0, 1, 1, GGML_TYPE_F32);
        tensors.push_back(FloatTensor("blk.0.ssm_a", {6}, GGML_TYPE_F32, {-1, -4, -2, -5, -3, -6}));
        std::vector<float> out(256 * 24);
        for (int row = 0; row < 256; ++row) for (int key = 0; key < 2; ++key)
            for (int v = 0; v < 3; ++v) for (int d = 0; d < 4; ++d)
                out[row * 24 + (v * 2 + key) * 4 + d] = key * 3 + v + 1;
        tensors.push_back(FloatTensor("blk.0.ssm_out.weight", {256, 24}, GGML_TYPE_F16, out));
        for (int layer = 0; layer < 2; ++layer) {
            tensors.push_back(FloatTensor("blk." + std::to_string(layer) + ".ffn_gate_inp_shexp.weight",
                {256}, GGML_TYPE_F32, std::vector<float>(256, .5f)));
        }
        for (const auto &kind : {"q", "k"}) {
            const int rows = kind == std::string("q") ? 8 : 4;
            tensors.push_back(FloatTensor("blk.1.indexer." + std::string(kind) + "_proj.weight",
                {rows, 256}, GGML_TYPE_BF16, std::vector<float>(rows * 256, rows)));
            tensors.push_back(FloatTensor("blk.1.indexer." + std::string(kind) + "_norm.weight",
                {4}, GGML_TYPE_F32, std::vector<float>(4, 1.25f)));
        }
        tensors.push_back(FloatTensor("output_hc_norm.weight", {512}, GGML_TYPE_F32, std::vector<float>(512, 1.5f)));
        for (const auto &kind : {"gate", "up", "down"}) {
            const bool down = kind == std::string("down");
            const ggml_type type = down ? GGML_TYPE_IQ4_NL : GGML_TYPE_IQ2_XS;
            const std::vector<int> dims = down ? std::vector<int>{2, 256, 32} : std::vector<int>{2, 32, 256};
            const size_t bytes = 2 * dims[1] * ggml_row_size(type, dims[2]);
            tensors.push_back({"blk.0.ffn_" + std::string(kind) + "_exps.weight", dims, type, Bytes(bytes, 0)});
        }
        tensors.push_back(FloatTensor("blk.1.ple_conv1d.weight", {512, 4}, GGML_TYPE_F32, std::vector<float>(512 * 4, .25f)));
        Write(files[1], {}, tensors);
        Tensor ple{"per_layer_token_embd.weight", {12, 32}, GGML_TYPE_IQ4_NL, {}};
        for (int row = 0; row < 12; ++row) {
            Put<uint16_t>(ple.bytes, 0x3c00); // IQ4_NL block scale = 1.
            for (int col = 0; col < 16; ++col) ple.bytes.push_back(row | ((15 - row) << 4));
        }
        Write(files[2], {}, {ple});
    }
    ~Fixture() { for (const auto &file : files) unlink(file.c_str()); rmdir(directory.c_str()); }
};
float At(Data &data, size_t i) {
    Check(data.cpuData && data.dataDevice == DataDevice::CPU, "expected CPU weight");
    if (data.dataType == FLOAT32) return reinterpret_cast<float *>(data.cpuData)[i];
    Check(data.dataType == FLOAT16, "unexpected dense dtype");
    return half_to_float(reinterpret_cast<uint16_t *>(data.cpuData)[i]);
}

void TestFloatImport(const std::string &directory) {
    // Exercise both shared floating-point import destinations, including
    // the F32 copy path and an actual quantized source at a nonzero offset.
    const std::string path = directory + "/float-import.bin";
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_IQ4_NL}) {
        Bytes payload;
        std::vector<float> expected(64);
        for (int i = 0; i < 64; ++i) expected[i] = (i % 32 < 16 ? 1.0f : 13.0f);
        if (type == GGML_TYPE_IQ4_NL) {
            for (int row = 0; row < 2; ++row) {
                Put<uint16_t>(payload, 0x3c00);
                payload.insert(payload.end(), 16, 0x98);
            }
        } else {
            payload = FloatTensor("import", {2, 32}, type, expected).bytes;
        }
        {
            std::ofstream file(path, std::ios::binary);
            file << "offset!";
            file.write(reinterpret_cast<const char *>(payload.data()), payload.size());
            Check(file.good(), "float import fixture write failed");
        }
        Data layout(DATA_GGUF_FORMAT, int(type), {2, 32});
        auto tensor = *static_cast<ggml_tensor *>(layout.ggmlTensor);
        tensor.name = "float-import";
        tensor.dims = {2, 32};
        for (auto mode : {GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32,
                          GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16}) {
            Data output;
            std::string filename = path;
            WeightImportGGUFTensor(&output, &tensor, filename, 7, mode);
            Check(output.isGGUFData && output.dims == std::vector<int>({2, 32}), "float import metadata");
            Check(output.dataType == (mode == GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                  ? FLOAT32 : FLOAT16), "float import destination type");
            for (int i = 0; i < 64; ++i) Check(At(output, i) == expected[i], "float import conversion");
        }
    }
    unlink(path.c_str());
}
}
namespace fastllm {
struct Qwen4GGUFTestAccess {
    static void CheckMetadata(Qwen4ExpModel &m) {
        Check(m.eosToken == 6 && m.eos_token_id == 8, "PLE reset token replaced generation EOS");
        Check(m.pleLayer == 1 && m.ngramShardCount == 1 && m.ngramHeadDim == 32, "PLE layout metadata");
        Check(m.pleMultipliers == std::vector<uint64_t>({23703573157769ULL, 20109073645365ULL, 8052911324071ULL}), "PLE uint64 multipliers truncated");
        Check(m.pleHeadOffsets == std::vector<int64_t>({0, 5}) && m.pleHeadVocabSizes == std::vector<int64_t>({5, 7}), "PLE head ranges");
        Check(m.IsLinearAttentionLayer(0) && !m.IsLinearAttentionLayer(1) && m.indexerCompressRatio == 2, "attention layout metadata");
    }
    static void Prepare(Qwen4ExpModel &m) { m.PrepareWeights(); }
};
}
int main() {
    try {
        SetThreads(2); SetDeviceMap({{"cpu", 1}}); SetMoeDeviceMap({{"cpu", 1}});
        SetMoeCudaCacheBytes(0); setenv("FASTLLM_QWEN4_ENABLE_MTP", "0", 1);
        Fixture fixture;
        TestFloatImport(fixture.directory);
        for (bool disk : {true, false}) {
            SetNgramDevice(disk ? "disk" : "cpu");
            auto base = CreateLLMModelFromGGUFFile(fixture.files[0], "");
            auto *m = dynamic_cast<Qwen4ExpModel *>(base.get()); Check(m != nullptr, "architecture dispatch");
            Qwen4GGUFTestAccess::CheckMetadata(*m);
            const std::string p = "model.language_model.layers.0.linear_attn.";
            for (const auto &suffix : {"in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight", "conv1d.weight", "dt_bias"}) {
                Data &w = m->weight[p + suffix];
                const bool prefix = suffix == std::string("in_proj_qkv.weight") || suffix == std::string("conv1d.weight");
                const int offset = prefix ? 16 : 0, hd = w.dims[0] == 6 ? 1 : 4;
                const int cols = w.Count(0) / w.dims[0];
                for (int r = 0; r < w.dims[0]; ++r) for (int c = 0; c < cols; ++c)
                    Check(At(w, r * cols + c) == (r < offset ? -1 : (r - offset) / hd + 1), "GDN row permutation");
            }
            for (int i = 0; i < 6; ++i) Check(std::abs(At(m->weight[p + "A_log"], i) - std::log(float(i + 1))) < 1e-6, "GDN negative-exp decay");
            for (int i = 0; i < 256 * 24; ++i) Check(At(m->weight[p + "out_proj.weight"], i) == i % 24 / 4 + 1, "GDN column permutation");
            Data &index = m->weight["model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight"];
            Check(index.dims == std::vector<int>({12, 256}), "QSA merged shape");
            for (int i = 0; i < 12 * 256; ++i) Check(At(index, i) == (i < 8 * 256 ? 8 : 4), "QSA BF16 Q/K concatenation");
            Check(m->weight["model.language_model.layers.0.mlp.shared_expert_gate.weight"].dims == std::vector<int>({1, 256}), "shared gate shape");
            Data &expert = m->weight["model.language_model.layers.0.mlp.experts.1.gateup_proj.weight"];
            Check(expert.dataType == DATA_GGUF_FORMAT && expert.ggmlType == GGML_TYPE_IQ2_XS && expert.dims == std::vector<int>({64, 256}), "packed expert merge changed format");
            Data &ple = m->weight["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"];
            Check(ple.isDiskWeight == disk && (ple.cpuData == nullptr) == disk && ple.ggmlType == GGML_TYPE_IQ4_NL, "PLE residency/quantization");
            if (disk) {
                Data ids(INT32, {5}), output; ids.Allocate();
                const int32_t rows[] = {11, 0, 5, 11, 1}; std::memcpy(ids.cpuData, rows, sizeof(rows));
                DiskEmbeddingOp op(true); DataDict data{{"input", &ids}, {"weight", &ple}, {"output", &output}};
                op.Reshape("EmbeddingDirect", data, {}, {}); op.Run("EmbeddingDirect", data, {}, {});
                Check(output.dataType == FLOAT32 && output.dims == std::vector<int>({5, 32}), "disk GGUF output type/shape");
                const int values[] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
                for (int i = 0; i < 5; ++i) for (int c = 0; c < 32; ++c)
                    Check(At(output, i * 32 + c) == values[c < 16 ? rows[i] : 15 - rows[i]], "IQ4_NL disk row lookup or duplicate reuse");
            }
            m->OnModelWeightsLoaded(); // Repeat notification must not permute twice.
            Check(At(m->weight[p + "out_proj.weight"], 12) == 4, "GGUF restoration is not idempotent");
            Qwen4GGUFTestAccess::Prepare(*m);
            Check(At(m->weight["model.language_model.hyper_connection_mixer.hc_norm.weight"], 0) == 1.5f, "GGUF RMSNorm offset applied twice");
            Check(At(m->weight["model.language_model.layers.1.self_attn.indexer.k_layernorm.weight"], 0) == 1.25f, "GGUF QSA norm offset applied twice");
            if (!disk) {
                Qwen4ExpModel hf;
                hf.weight.dicts = m->weight.dicts;
                hf.weight.dicts.erase("gguf_architecture");
                hf.deviceMap = hf.moeDeviceMap = {{"cpu", 1}};
                hf.InitParams();
                const std::string norm = "model.language_model.hyper_connection_mixer.hc_norm.weight";
                const std::string key = "model.language_model.layers.1.self_attn.indexer.k_layernorm.weight";
                hf.weight.AddEmptyWeight(norm, {512}, FLOAT32);
                hf.weight.AddEmptyWeight(key, {4}, FLOAT32);
                hf.weight[norm].Allocate();
                hf.weight[key].Allocate();
                std::fill_n(reinterpret_cast<float *>(hf.weight[norm].cpuData), 512, .5f);
                std::fill_n(reinterpret_cast<float *>(hf.weight[key].cpuData), 4, .25f);
                Qwen4GGUFTestAccess::Prepare(hf);
                Check(At(hf.weight[norm], 0) == 1.5f && At(hf.weight[key], 0) == 1.25f,
                      "HF RMSNorm lost its +1 offset");
            }
        }
        std::cout << "PASS: Qwen4 three-shard GGUF import, layouts, quantized PLE and norms\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
