#include "models/deepseekv41.h"
#include "utils.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

using namespace fastllm;
static void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

struct Fixture {
    static constexpr int rows = 1057, dim = 256, block = 32;
    std::string path;
    std::vector<uint8_t> values, scales;
    DiskWeightPart valuePart, scalePart;
    Fixture() : values(rows * dim), scales(rows * dim / block) {
        char name[] = "/tmp/fastllm-v41-engram-XXXXXX";
        int fd = mkstemp(name); Check(fd >= 0, "fixture file"); close(fd); path = name;
        for (size_t i = 0; i < values.size(); ++i) values[i] = uint8_t(i * 137 + i / dim);
        for (size_t i = 0; i < scales.size(); ++i) scales[i] = uint8_t(120 + i % 15);
        std::ofstream file(path, std::ios::binary);
        const std::vector<char> padding(4093, 0);
        file.write(padding.data(), padding.size());
        valuePart.fileName = path; valuePart.fileOffset = file.tellp();
        valuePart.sourceDataType = FP8_E4M3; valuePart.dims = {rows, dim}; valuePart.bytes = values.size();
        file.write(reinterpret_cast<const char *>(values.data()), values.size());
        file.write(padding.data(), padding.size());
        scalePart.fileName = path; scalePart.fileOffset = file.tellp();
        scalePart.sourceDataType = INT8; scalePart.dims = {rows, dim / block};
        scalePart.bytes = scales.size(); scalePart.isScalePart = true;
        file.write(reinterpret_cast<const char *>(scales.data()), scales.size());
        file.write(padding.data(), padding.size());
        Check(bool(file), "fixture write");
    }
    ~Fixture() { std::remove(path.c_str()); }
};

struct Probe : DeepSeekV41Model {
    using DeepSeekV41Model::GatherEngramRows;
    void Prepare(const Fixture &f, bool async) {
        executionMode = async ? ExecutionMode::Hybrid : ExecutionMode::Cpu;
        ngramDevice = "disk"; engram_layer_ids = {0}; engram_head_dim = f.dim;
        weight.dicts["model_directory"] = "/";
        auto &embedding = weight.weight["layers.0.engram.embed.weight"];
        embedding = Data(FP8_E4M3, {f.rows, f.dim});
        embedding.blockK = 1; embedding.blockM = f.block;
        embedding.isDiskWeight = true; embedding.weightType = WeightType::EMBEDDING;
        embedding.diskWeightParts = {f.valuePart, f.scalePart};
        OnModelWeightsLoaded();
    }
};

static void TestIQ4NL() {
    constexpr int rows = 1057, dim = 256, rowBytes = dim / 32 * 18;
    const int codebook[] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    Fixture file;
    std::vector<uint8_t> packed(rows * rowBytes);
    for (int r = 0; r < rows; ++r) for (int b = 0; b < dim / 32; ++b) {
        auto *block = packed.data() + r * rowBytes + b * 18;
        uint16_t d = float_to_half(std::ldexp(1.f, -8 + (r + b) % 5));
        memcpy(block, &d, 2);
        for (int i = 0; i < 16; ++i) block[2 + i] = uint8_t(r * 13 + b * 19 + i * 7);
    }
    DiskWeightPart part;
    part.fileName = file.path; part.fileOffset = 4093; part.bytes = packed.size();
    part.sourceDataType = DATA_GGUF_FORMAT; part.dims = {rows, dim};
    {
        std::ofstream out(file.path, std::ios::binary);
        std::vector<char> padding(part.fileOffset);
        out.write(padding.data(), padding.size());
        out.write(reinterpret_cast<const char *>(packed.data()), packed.size());
        out.write(padding.data(), padding.size());
    }
    for (bool disk : {false, true}) for (bool async : {false, true}) {
        struct IQProbe : Probe {
            void PrepareIQ(const std::vector<uint8_t> &packed, const DiskWeightPart &part, bool disk, bool async) {
                executionMode = async ? ExecutionMode::Hybrid : ExecutionMode::Cpu;
                ngramDevice = disk ? "disk" : "cpu"; engram_layer_ids = {0}; engram_head_dim = dim;
                engram_num_embeddings = {rows};
                weight.dicts["model_directory"] = "/";
                auto &w = weight.weight["layers.0.engram.embed.weight"];
                w = Data(DATA_GGUF_FORMAT, GGML_TYPE_IQ4_NL, {rows, dim});
                w.isDiskWeight = disk; w.diskWeightParts = {part};
                if (!disk) { w.Allocate(false); memcpy(w.cpuData, packed.data(), packed.size()); }
                OnModelWeightsLoaded();
            }
        } probe;
        probe.PrepareIQ(packed, part, disk, async);
        for (int tokens : {1, 9, 257, 1}) {
            std::vector<int64_t> ids(tokens * 24);
            for (size_t i = 0; i < ids.size(); ++i) ids[i] = i % 3 ? (i * 997 + tokens) % rows : rows - 1;
            Data output;
            probe.GatherEngramRows(0, ids, tokens, output);
            Check(output.dims == std::vector<int>({1,tokens,24*dim}) && output.dataType == BFLOAT16, "IQ4_NL shape/type");
            for (size_t i = 0; i < ids.size(); ++i) for (int c = 0; c < dim; ++c) {
                const int b = c / 32, lane = c % 32;
                const auto *block = packed.data() + ids[i] * rowBytes + b * 18;
                const int code = (block[2 + lane % 16] >> (lane < 16 ? 0 : 4)) & 15;
                const float d = std::ldexp(1.f, -8 + (ids[i] + b) % 5);
                Check(reinterpret_cast<uint16_t *>(output.cpuData)[i*dim+c] == Float32ToBFloat16RNEBits(d * codebook[code]),
                      "IQ4_NL gather differs from scalar codebook oracle");
            }
        }
    }
}

int main() {
    try {
        TestIQ4NL();
        const FP8E4M3ToFP32Manager fp8;
        for (int threads : {1, 4}) for (bool direct : {false, true}) {
            SetThreads(threads);
            setenv("FASTLLM_DISK_DIRECT_IO", direct ? "1" : "0", 1);
            Fixture fixture;
            Probe async, serial;
            async.Prepare(fixture, true); serial.Prepare(fixture, false);
            // A single token has 24 hashes, including duplicates and the last
            // table row. Reuse outputs across changing decode/prefill shapes.
            Data actual, reference;
            for (int tokens : {1, 7, 8, 9, 1, 2, 1}) {
                std::vector<int64_t> rows(tokens * 24);
                for (size_t i = 0; i < rows.size(); ++i) rows[i] = (i * 997 + tokens) % fixture.rows;
                rows[0] = fixture.rows - 1; rows[1] = 0; rows[2] = rows[0];
                async.GatherEngramRows(0, rows, tokens, actual);
                serial.GatherEngramRows(0, rows, tokens, reference);
                Check(actual.dims == std::vector<int>({1, tokens, 24 * fixture.dim}) &&
                      actual.dataType == BFLOAT16 && actual.GetBytes() == reference.GetBytes(), "gather shape/type");
                Check(!memcmp(actual.cpuData, reference.cpuData, actual.GetBytes()), "async/serial gather bits");
                for (size_t i = 0; i < rows.size(); ++i) for (int d = 0; d < fixture.dim; ++d) {
                    const float scale = std::ldexp(1.f, int(fixture.scales[rows[i] * (fixture.dim / fixture.block) + d / fixture.block]) - 127);
                    const uint16_t expected = Float32ToBFloat16RNEBits(fp8.dict[fixture.values[rows[i] * fixture.dim + d]] * scale);
                    Check(reinterpret_cast<const uint16_t *>(actual.cpuData)[i * fixture.dim + d] == expected,
                          "gather differs from scalar FP8/scale oracle");
                }
            }
        }
        std::puts("PASS: V4.1 disk Engram decode and prefill, async/serial/scalar bits, duplicates and direct I/O");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
