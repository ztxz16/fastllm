#include "model.h"
#include "models/qwen4_exp.h"
#include "executor.h"
#include "devices/disk/diskdevice.h"
#include "gguf.h"
#include "json11.hpp"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

#include <algorithm>
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
    if (type == GGML_TYPE_Q8_0) {
        Check(values.size() % 32 == 0, "Q8 fixture block alignment");
        for (size_t b = 0; b < values.size(); b += 32) {
            Put<uint16_t>(t.bytes, float_to_half(0.125f));
            for (int c = 0; c < 32; ++c) Put<int8_t>(t.bytes, int8_t(values[b + c] * 8));
        }
        return t;
    }
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
    Fixture(bool tpExperts = false, int expertCount = 2) {
        const int expertWidth = tpExperts ? 64 : 32;
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
            {"expert_count", expertCount}, {"expert_used_count", 1}, {"expert_feed_forward_length", expertWidth},
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
        Tensor embedding{"token_embd.weight", {32, 256}, GGML_TYPE_IQ4_XS,
                         Bytes(32 * ggml_row_size(GGML_TYPE_IQ4_XS, 256))};
        for (int row = 0; row < 32; ++row) {
            auto &block = reinterpret_cast<block_iq4_xs *>(embedding.bytes.data())[row];
            block.d = float_to_half((row + 1) / 1024.0f);
            block.scales_h = row * 1973;
            for (int i = 0; i < 4; ++i) block.scales_l[i] = row * 13 + i * 71;
            for (int i = 0; i < 128; ++i) block.qs[i] = row * 17 + i * 29;
        }
        tensors.push_back(std::move(embedding));
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
        rows("blk.0.attn_qkv.weight", 16, 4, 256, GGML_TYPE_Q8_0);
        rows("blk.0.attn_gate.weight", 0, 4, 256, GGML_TYPE_Q8_0);
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
            tensors.push_back(FloatTensor("blk." + std::to_string(layer) + ".hc_attn_down.weight",
                {4, 512}, GGML_TYPE_BF16, std::vector<float>(4 * 512, layer + .25f)));
        }
        for (const auto &kind : {"q", "k"}) {
            const int rows = kind == std::string("q") ? 8 : 4;
            tensors.push_back(FloatTensor("blk.1.indexer." + std::string(kind) + "_proj.weight",
                {rows, 256}, GGML_TYPE_BF16, std::vector<float>(rows * 256, rows)));
            tensors.push_back(FloatTensor("blk.1.indexer." + std::string(kind) + "_norm.weight",
                {4}, GGML_TYPE_F32, std::vector<float>(4, 1.25f)));
        }
        tensors.push_back(FloatTensor("output_hc_norm.weight", {512}, GGML_TYPE_F32, std::vector<float>(512, 1.5f)));
        tensors.push_back(FloatTensor("output_hc_up.weight", {512, 4}, GGML_TYPE_BF16,
                                      std::vector<float>(512 * 4, .75f)));
        for (int layer = 0; layer < (tpExperts ? 2 : 1); ++layer)
        for (const auto &kind : {"gate", "up", "down"}) {
            const bool down = kind == std::string("down");
            const ggml_type type = down ? GGML_TYPE_IQ4_NL : GGML_TYPE_IQ2_XS;
            const std::vector<int> dims = down ? std::vector<int>{expertCount, 256, expertWidth} : std::vector<int>{expertCount, expertWidth, 256};
            const size_t bytes = expertCount * dims[1] * ggml_row_size(type, dims[2]);
            Bytes payload(bytes, 0);
            if (tpExperts) for (size_t i = 0; i < bytes; ++i)
                payload[i] = (i / ggml_row_size(type, dims[2]) + 17 * i + layer) % 63;
            tensors.push_back({"blk." + std::to_string(layer) + ".ffn_" + std::string(kind) + "_exps.weight", dims, type, payload});
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
    if (data.dataType == DATA_GGUF_FORMAT) {
        std::vector<float> row(data.dims.back());
        const size_t columns = data.dims.back();
        ggml_type_to_float(static_cast<ggml_type>(data.ggmlType))(
            data.cpuData + i / columns * ggml_row_size(static_cast<ggml_type>(data.ggmlType), columns), row.data(), columns);
        return row[i % columns];
    }
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

void TestEmbeddingImport(const Fixture &fixture) {
    const std::string name = "model.language_model.embed_tokens.weight";
    // Compare against the previous full-table FP32 import, including repeated
    // indices, the last vocabulary row, and FP16 token IDs/output conversion.
    for (bool cudaEmbedding : {true, false, true, false}) {
        SetCudaEmbedding(cudaEmbedding);
        std::vector<ReadGGUFTask> tasks;
        AppendGGUFTasks("qwen4_exp", fixture.files[1], tasks);
        auto task = std::find_if(tasks.begin(), tasks.end(), [&](const ReadGGUFTask &t) {
            return t.name == name;
        });
        Check(task != tasks.end(), "missing embedding import task");
        Check(task->replaceType == (cudaEmbedding
              ? GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
              : GGUFWeightReplaceRule::GGUFWeightReplaceDirect), "embedding placement policy");
        // Match the loader's AddEmptyWeight placeholder before direct import.
        Data weight(FLOAT32, {1}), reference(FLOAT32, {1});
        WeightImportGGUFTensor(&weight, &task->tensor, task->fileName, task->offset, task->replaceType);
        WeightImportGGUFTensor(&reference, &task->tensor, task->fileName, task->offset,
                              GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32);
        auto *storage = weight.cpuData;
        const size_t bytes = weight.GetBytes();
        for (DataType type : {FLOAT32, FLOAT16}) {
            for (const std::vector<float> &ids : {std::vector<float>{31}, {31, 0, 15, 31, 1}}) {
                Data input(type, {1, (int)ids.size()}, ids), actual, expected;
                auto &executor = *static_cast<Executor *>(GetExecutor());
                DataDict packed{{"input", &input}, {"weight", &weight}, {"output", &actual}};
                DataDict dense{{"input", &input}, {"weight", &reference}, {"output", &expected}};
                executor.RunOnDevice("cpu", "Embedding", packed, {}, {});
                executor.RunOnDevice("cpu", "Embedding", dense, {}, {});
                Check(actual.dims == expected.dims && actual.dataType == expected.dataType &&
                      actual.GetBytes() == expected.GetBytes() &&
                      std::memcmp(actual.cpuData, expected.cpuData, actual.GetBytes()) == 0,
                      "packed embedding changed decoded values or output dtype");
            }
        }
        Check(weight.cpuData == storage && weight.GetBytes() == bytes,
              "embedding lookup reallocated the weight");
        if (!cudaEmbedding) Check(weight.dataType == DATA_GGUF_FORMAT &&
            weight.ggmlType == GGML_TYPE_IQ4_XS && bytes == 32 * ggml_row_size(GGML_TYPE_IQ4_XS, 256),
            "CPU embedding expanded the quantized table");
    }
    std::cout << "PASS: packed Qwen4 embedding matches FP32 import and respects CUDA policy\n";
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
    static void PrepareTp(Qwen4ExpModel &m) { m.PrepareThreadTp(); }
    static bool HasMtp(const Qwen4ExpModel &m) { return m.HasMtpWeights(); }
    static void CheckPlePrefetch(Qwen4ExpModel &m) {
        static_cast<Executor *>(GetExecutor())->SetFirstDevice("cpu");
        const std::string prefix = "model.language_model.layers.1.ple.";
        const int channels = m.hcCount * m.embed_dim;
        auto add = [&](const std::string &name, const std::vector<int> &dims, bool norm) {
            m.weight.AddEmptyWeight(prefix + name, dims, FLOAT32);
            Data &w = m.weight[prefix + name]; w.Allocate();
            for (uint64_t i = 0; i < w.Count(0); ++i)
                ((float*)w.cpuData)[i] = norm ? 1.0f : ((int)(i % 31) - 15) * .0005f;
        };
        add("key_proj.weight", {channels, m.pleEmbedDim}, false);
        add("value_proj.weight", {m.embed_dim, m.pleEmbedDim}, false);
        for (const char *name : {"norm_key.weight", "norm_query.weight", "norm_conv.weight"})
            add(name, {channels}, true);
        auto reader = m.pleDiskReader;
        Check(reader != nullptr, "PLE reader was not initialized");
        Qwen4ExpModel::RequestState oldState, newState;
        for (const std::vector<int> tokens : {std::vector<int>{1, 6, 3, 4}, {5}, {6}, {2}, {9, 8, 7}}) {
            std::vector<float> ids(tokens.begin(), tokens.end());
            Data input(FLOAT32, {1, (int)tokens.size()}, ids);
            std::vector<float> values(tokens.size() * channels);
            for (size_t i = 0; i < values.size(); ++i) values[i] = ((int)(i % 19) - 9) * .01f;
            Data hidden(FLOAT32, {1, (int)tokens.size(), channels}, values), before, after;
            DiskEmbeddingRowReader::Ticket ticket;
            if (tokens.size() == 1) {
                const int shifted[] = {tokens[0], newState.previousToken1 < 0 ? m.eosToken : newState.previousToken1,
                    newState.previousToken2 < 0 ? m.eosToken : newState.previousToken2};
                std::vector<int32_t> rows(m.ngramHeads);
                for (int h = 0; h < m.ngramHeads; ++h) rows[h] = m.PLEHashRow(shifted, h);
                ticket = reader->ReadAsync(rows);
            }
            // The baseline uses the existing disk EmbeddingDirect operation.
            m.pleDiskReader.reset();
            m.RunPLE(hidden, input, oldState, before, &tokens);
            m.pleDiskReader = reader;
            const auto checkpoint = newState;
            m.RunPLE(hidden, input, newState, after, &tokens, &ticket);
            auto same = [&] {
                Check(before.dims == after.dims && before.GetBytes() == after.GetBytes() &&
                      !memcmp(before.cpuData, after.cpuData, before.GetBytes()), "PLE output changed");
                Check(oldState.previousToken1 == newState.previousToken1 &&
                      oldState.previousToken2 == newState.previousToken2 &&
                      oldState.processedTokens == newState.processedTokens &&
                      oldState.convHistory == newState.convHistory, "PLE history changed");
            };
            same();
            // A speculative rollback must be independent of the cached rows.
            newState = checkpoint;
            auto stale = reader->ReadAsync({0});
            m.RunPLE(hidden, input, newState, after, &tokens, &stale);
            same();
        }
        std::cout << "PASS: PLE prefetch matches serial output/history, EOS and rollback\n";
    }
};
}
static std::string WriteMtpFixture(Fixture &fixture, int expertWidth = 32) {
    std::vector<Tensor> tensors;
    auto add = [&](const std::string &name, std::vector<int> dims, ggml_type type) {
        size_t n = 1; for (int d : dims) n *= d;
        tensors.push_back(FloatTensor(name, dims, type, std::vector<float>(n, 0)));
    };
    for (const std::string &name : {"pre_fc_norm_embedding", "pre_fc_norm_hidden"})
        add("mtp." + name + ".weight", {name == "pre_fc_norm_hidden" ? 512 : 256}, GGML_TYPE_F32);
    add("mtp.fc_embedding.weight", {256, 256}, GGML_TYPE_BF16);
    add("mtp.fc_hidden.weight", {256, 256}, GGML_TYPE_BF16);
    for (const std::string &prefix : {"mtp.hyper_connection_mixer.", "mtp.layers.0.attn_hyper_connection.", "mtp.layers.0.mlp_hyper_connection."}) {
        add(prefix + "hc_norm.weight", {512}, GGML_TYPE_F32);
        add(prefix + "input_mix_weight_down.weight", {4, 512}, GGML_TYPE_BF16);
        add(prefix + "input_mix_weight_up.weight", {512, 4}, GGML_TYPE_BF16);
        if (prefix != "mtp.hyper_connection_mixer.") add(prefix + "block_inject_weight.weight", {2, 512}, GGML_TYPE_BF16);
    }
    const std::string attn = "mtp.layers.0.self_attn.";
    add(attn + "q_proj.weight", {16, 256}, GGML_TYPE_BF16);
    add(attn + "k_proj.weight", {4, 256}, GGML_TYPE_BF16);
    add(attn + "v_proj.weight", {4, 256}, GGML_TYPE_BF16);
    add(attn + "o_proj.weight", {256, 8}, GGML_TYPE_BF16);
    for (const std::string &name : {"q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"})
        add(attn + name + ".weight", {4}, GGML_TYPE_F32);
    add(attn + "indexer.index_qk_proj.weight", {8, 256}, GGML_TYPE_BF16);
    const std::string mlp = "mtp.layers.0.mlp.";
    add(mlp + "gate.weight", {2, 256}, GGML_TYPE_BF16);
    add(mlp + "shared_expert_gate.weight", {1, 256}, GGML_TYPE_BF16);
    for (const std::string &name : {"gate", "up"}) add(mlp + "shared_expert." + name + "_proj.weight", {32, 256}, GGML_TYPE_BF16);
    add(mlp + "shared_expert.down_proj.weight", {256, 32}, GGML_TYPE_BF16);
    add(mlp + "experts.gate_up_proj", {2, 2 * expertWidth, 256}, GGML_TYPE_Q8_0);
    add(mlp + "experts.down_proj", {2, 256, expertWidth}, GGML_TYPE_Q8_0);
    const std::string path = fixture.directory + "/mtp.gguf";
    Write(path, {{"general.architecture", "qwen4exp-mtp"}}, tensors);
    fixture.files.push_back(path);
    return path;
}
static void TestMtpImport(Fixture &fixture) {
    const std::string path = WriteMtpFixture(fixture);
    const std::string mlp = "mtp.layers.0.mlp.";
    setenv("FASTLLM_QWEN4_ENABLE_MTP", "3", 1);
    auto loaded = CreateLLMModelFromGGUFFile(fixture.files[0], "", path);
    auto *model = dynamic_cast<Qwen4ExpModel *>(loaded.get());
    Check(model && Qwen4GGUFTestAccess::HasMtp(*model), "external MTP GGUF was silently disabled");
    for (int e = 0; e < 2; ++e) {
        const auto &gate = model->weight[mlp + "experts." + std::to_string(e) + ".gateup_proj.weight"];
        const auto &down = model->weight[mlp + "experts." + std::to_string(e) + ".down_proj.weight"];
        Check(gate.dims == std::vector<int>({64, 256}) && down.dims == std::vector<int>({256, 32}), "MTP packed expert split shape");
        Check(gate.dataType == DATA_GGUF_FORMAT && down.dataType == DATA_GGUF_FORMAT && gate.ggmlType == GGML_TYPE_Q8_0, "MTP packed experts were expanded");
    }
    Check(model->weight["mtp.pre_fc_norm_hidden.weight"].dataType == FLOAT32, "MTP norm must remain FP32");
    Check(model->weight["mtp.fc_hidden.weight"].dataType == FLOAT16, "MTP dense must use FP16");
    Qwen4GGUFTestAccess::Prepare(*model);
    Check(Qwen4GGUFTestAccess::HasMtp(*model), "MTP weights lost during prepare");
    Check(At(model->weight["mtp.pre_fc_norm_hidden.weight"], 0) == 1.0f, "raw HF MTP norm offset missing");
    Qwen4GGUFTestAccess::Prepare(*model);
    Check(At(model->weight["mtp.layers.0.self_attn.indexer.k_layernorm.weight"], 0) == 1.0f, "MTP norm offset applied twice");
    setenv("FASTLLM_QWEN4_ENABLE_MTP", "0", 1);
    std::cout << "PASS: external Qwen4 MTP GGUF, packed expert split, dense/norm dtype and preparation\n";
}
static int TestStreamingTpImport() {
#ifdef USE_CUDA
    if (FastllmCudaGetDeviceCount() < 2) return 77;
    unsetenv("FASTLLM_TP");
    Fixture fixture(true);
    const std::string mtpPath = WriteMtpFixture(fixture, 64);
    setenv("FASTLLM_QWEN4_ENABLE_MTP", "3", 1);
    SetNgramDevice("disk");
    auto cpu = CreateLLMModelFromGGUFFile(fixture.files[0], "", mtpPath);
    SetDeviceMap({{"cuda:0", 1}}); SetMoeDeviceMap({{"cuda:0", 1}});
    SetLayeredMoeDeviceMap({{"cpu", 1}});
    setenv("FASTLLM_TP", "cuda:0,1", 1);
    for (bool hostLastLayer : {true, false}) {
        SetMoeDeviceLayers(hostLastLayer ? 1 : 0);
        auto tp = CreateLLMModelFromGGUFFile(fixture.files[0], "", mtpPath);
        for (int layer = 0; layer < 3; ++layer) for (int expert = 0; expert < 2; ++expert)
        for (const std::string kind : {"gateup", "down"}) {
            const std::string name = (layer == 2 ? "mtp.layers.0.mlp.experts." :
                "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.") +
                std::to_string(expert) + "." + kind + "_proj.weight";
            Data &reference = cpu->weight[name], &weight = tp->weight[name];
            Check(weight.dims == reference.dims && weight.ggmlType == reference.ggmlType,
                  "streaming TP changed parent weight metadata");
            if (hostLastLayer && layer >= 1) {
                Check(!weight.multiDeviceData && weight.cpuData != nullptr,
                      "streaming TP uploaded a CPU expert layer");
                Check(std::memcmp(weight.cpuData, reference.cpuData, reference.GetBytes()) == 0,
                      "streaming TP changed a CPU expert payload");
                continue;
            }
            Check(weight.cpuData == nullptr && weight.multiDeviceDatas.size() == 2,
                  "GPU expert source was retained until warmup");
            const size_t rowBytes = ggml_row_size((ggml_type)reference.ggmlType, reference.dims[1]);
            for (int rank = 0; rank < 2; ++rank) {
                Data &shard = *weight.multiDeviceDatas.at(rank);
                Check(shard.dataDevice == DataDevice::CUDA && shard.cudaData && !shard.isFake,
                      "streamed shard must own its CUDA allocation before rank preparation");
                Bytes actual(shard.GetBytes()), expected;
                FastllmCudaSetDevice(rank);
                FastllmCudaCopyFromDeviceToHost(actual.data(), shard.cudaData, actual.size());
                if (kind == "gateup") {
                    Check(shard.dims == std::vector<int>({64, 256}), "streamed gate/up shard shape");
                    for (int half = 0; half < 2; ++half) {
                        const uint8_t *begin = reference.cpuData + (half * 64 + rank * 32) * rowBytes;
                        expected.insert(expected.end(), begin, begin + 32 * rowBytes);
                    }
                } else {
                    Check(shard.dims == std::vector<int>({256, 32}), "streamed down shard shape");
                    for (int row = 0; row < 256; ++row) {
                        const uint8_t *begin = reference.cpuData + row * rowBytes + rank * rowBytes / 2;
                        expected.insert(expected.end(), begin, begin + rowBytes / 2);
                    }
                }
                Check(actual == expected, "streaming TP changed packed expert shard bytes");
            }
        }
        const std::vector<std::string> replicaNames = {"model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
                "model.language_model.layers.1.attn_hyper_connection.input_mix_weight_down.weight",
                "mtp.fc_hidden.weight"};
        std::map<std::pair<std::string, int>, void *> replicaPointers;
        for (const std::string &name : replicaNames) {
            Data &source = tp->weight[name], &reference = cpu->weight[name];
            Check(!source.cpuData && source.multiDeviceData && source.multiDeviceDatas.size() == 2,
                  "replicated projection remained on CPU until warmup");
            for (int device : {0, 1}) {
                Data &replica = *source.multiDeviceDatas.at(device);
                replicaPointers[{name, device}] = replica.cudaData;
                Check(!replica.isFake && replica.cudaData && replica.dims == reference.dims &&
                      replica.GetBytes() == reference.GetBytes(), "streamed replica ownership or shape");
                Bytes actual(replica.GetBytes());
                FastllmCudaSetDevice(device);
                FastllmCudaCopyFromDeviceToHost(actual.data(), replica.cudaData, actual.size());
                Check(std::memcmp(actual.data(), reference.cpuData, actual.size()) == 0,
                      "streamed replica changed projection bytes");
            }
        }
        // The loader calls this hook again after ordinary weights: it must be a no-op.
        tp->OnWeightLoadGroupFinished();
        const std::string embeddingName = "model.language_model.embed_tokens.weight";
        auto *embeddingStorage = tp->weight[embeddingName].cpuData;
        Check(embeddingStorage != nullptr, "missing shared CPU embedding");
        Qwen4GGUFTestAccess::PrepareTp(*dynamic_cast<Qwen4ExpModel *>(tp.get()));
        Qwen4GGUFTestAccess::PrepareTp(*dynamic_cast<Qwen4ExpModel *>(tp.get()));
        for (const std::string &name : replicaNames) for (int device : {0, 1}) {
            const Data &replica = *tp->weight[name].multiDeviceDatas.at(device);
            Check(replica.isFake && replica.cudaData == replicaPointers.at({name, device}),
                  "TP preparation copied or retained ownership of a streamed replica");
        }
        Check(tp->weight[embeddingName].cpuData == embeddingStorage &&
              tp->weight[embeddingName].dataType == DATA_GGUF_FORMAT,
              "TP preparation released or expanded a borrowed host embedding");
        for (const std::string name : {"model.language_model.hyper_connection_mixer.hc_norm.weight",
                                      "model.language_model.layers.1.ple.conv1d.weight"}) {
            Check(tp->weight[name].cpuData == nullptr && tp->weight[name].cudaData == nullptr,
                  "TP parent retained a replicated weight payload");
        }
        Check((tp->weight["model.language_model.layers.1.mlp.experts.0.down_proj.weight"].cpuData != nullptr) == hostLastLayer,
              "TP preparation changed CPU expert placement");
        Check((tp->weight["mtp.layers.0.mlp.experts.0.down_proj.weight"].cpuData != nullptr) == hostLastLayer,
              "TP preparation changed MTP expert placement");
    }
    unsetenv("FASTLLM_TP");
    std::cout << "PASS: Qwen4 streaming TP expert load, released CPU sources and exact packed shards\n";
    return 0;
#else
    return 77;
#endif
}

static void TestConcurrentExpertMerges() {
    // Many independent gate/up merges used to insert/erase map nodes without
    // the loader mutex. Verify untouched down projections as well as outputs.
    SetThreads(16);
    SetNgramDevice("cpu");
    constexpr int experts = 128;
    Fixture fixture(false, experts);
    for (int repeat = 0; repeat < 8; ++repeat) {
        auto model = CreateLLMModelFromGGUFFile(fixture.files[0], "");
        for (int e = 0; e < experts; ++e) {
            const auto prefix = "model.language_model.layers.0.mlp.experts." + std::to_string(e) + ".";
            for (const auto &item : {std::make_pair("gateup_proj.weight", std::vector<int>{64, 256}),
                                     std::make_pair("down_proj.weight", std::vector<int>{256, 32})}) {
                auto it = model->weight.weight.find(prefix + item.first);
                Check(it != model->weight.weight.end() && it->second.dims == item.second,
                      "parallel GGUF merge lost an expert projection");
                Check(it->second.cpuData != nullptr, "parallel GGUF merge lost weight storage");
            }
            Check(!model->weight.weight.count(prefix + "gate_proj.weight") &&
                  !model->weight.weight.count(prefix + "up_proj.weight"),
                  "parallel GGUF merge retained source projections");
        }
    }
    SetThreads(2);
}

int main(int argc, char **argv) {
    try {
        SetThreads(2); SetDeviceMap({{"cpu", 1}}); SetMoeDeviceMap({{"cpu", 1}});
        SetCudaEmbedding(false);
        SetMoeCudaCacheBytes(0); setenv("FASTLLM_QWEN4_ENABLE_MTP", "0", 1);
        if (argc == 2 && std::string(argv[1]) == "--tp-load") return TestStreamingTpImport();
        TestConcurrentExpertMerges();
        Fixture fixture;
        TestFloatImport(fixture.directory);
        TestEmbeddingImport(fixture);
        TestMtpImport(fixture);
        for (bool disk : {true, false}) {
            SetNgramDevice(disk ? "disk" : "cpu");
            auto base = CreateLLMModelFromGGUFFile(fixture.files[0], "");
            auto *m = dynamic_cast<Qwen4ExpModel *>(base.get()); Check(m != nullptr, "architecture dispatch");
            Qwen4GGUFTestAccess::CheckMetadata(*m);
            const std::string p = "model.language_model.layers.0.linear_attn.";
            for (const char *name : {"in_proj_qkv.weight", "in_proj_z.weight"}) {
                Data &w = m->weight[p + name];
                Check(w.dataType == DATA_GGUF_FORMAT && w.ggmlType == GGML_TYPE_Q8_0 && !w.IsRepacked,
                      "quantized GDN projection was expanded or repacked before row restoration");
            }
            for (const auto &suffix : {"in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight", "conv1d.weight", "dt_bias"}) {
                Data &w = m->weight[p + suffix];
                const bool prefix = suffix == std::string("in_proj_qkv.weight") || suffix == std::string("conv1d.weight");
                const int offset = prefix ? 16 : 0, hd = w.dims[0] == 6 ? 1 : 4;
                const int cols = w.dims.size() == 2 ? w.dims[1] : 1;
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
            if (disk) Qwen4GGUFTestAccess::CheckPlePrefetch(*m);
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
