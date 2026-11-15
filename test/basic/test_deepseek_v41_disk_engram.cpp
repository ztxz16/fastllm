#include "models/deepseekv41.h"
#include "utils.h"
#include "gguf.h"
#include "../../src/models/deepseekv41-engram-prefetch.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
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
        ngramDevice = "disk"; engram_layer_ids.assign(1, 0); engram_head_dim = f.dim;
        weight.dicts["model_directory"] = "/";
        auto &embedding = weight.weight["layers.0.engram.embed.weight"];
        embedding = Data(FP8_E4M3, {f.rows, f.dim});
        embedding.blockK = 1; embedding.blockM = f.block;
        embedding.isDiskWeight = true; embedding.weightType = WeightType::EMBEDDING;
        embedding.diskWeightParts = {f.valuePart, f.scalePart};
        OnModelWeightsLoaded();
    }
};

// Independent arithmetic oracle, including E4M3FN's two NaN codes and the
// E8M0 reserved scale. Finite values also cover signed zero and subnormals.
static float ReferenceMXFP8(uint8_t code, uint8_t scale) {
    if ((code & 127) == 127 || scale == 255) return std::numeric_limits<float>::quiet_NaN();
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    const double value = exponent ? std::ldexp(double(8 + mantissa), exponent - 10)
                                  : std::ldexp(double(mantissa), -9);
    return float(std::ldexp(code & 128 ? -value : value, int(scale) - 127));
}

static void TestMXFP8Decode() {
    Check(int(GGML_TYPE_MXFP8) == 43 && ggml_blck_size(GGML_TYPE_MXFP8) == 32 &&
          ggml_row_size(GGML_TYPE_MXFP8, 256) == 264, "MXFP8 wire format");
    const auto decode = ggml_type_to_float(GGML_TYPE_MXFP8);
    Check(decode != nullptr, "MXFP8 decoder registration");
    std::vector<block_mxfp8> blocks(8);
    std::vector<float> actual(258, 123.f);
    for (int scale = 0; scale < 256; ++scale) {
        for (int code = 0; code < 256; ++code) {
            blocks[code / 32].e = uint8_t(scale);
            blocks[code / 32].qs[code % 32] = uint8_t(code);
        }
        decode(blocks.data(), actual.data() + 1, 256);
        for (int code = 0; code < 256; ++code) {
            const float expected = ReferenceMXFP8(code, scale), value = actual[code + 1];
            Check(std::isnan(expected) ? std::isnan(value) :
                  (value == expected && std::signbit(value) == std::signbit(expected)), "MXFP8 scalar oracle");
        }
        Check(actual.front() == 123.f && actual.back() == 123.f, "MXFP8 output guard");
    }
    decode(blocks.data(), actual.data(), 0);
    Check(actual.front() == 123.f, "MXFP8 empty decode");
    std::puts("PASS MXFP8: all 65536 value/scale pairs, signed zero, NaN and output guards");
}

static void TestGGUFRows(ggml_type type) {
    constexpr int rows = 1057, dim = 256;
    const size_t rowBytes = ggml_row_size(type, dim);
    const int codebook[] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    Fixture file;
    std::vector<uint8_t> packed(rows * rowBytes);
    std::vector<float> reference(rows * dim);
    for (int r = 0; r < rows; ++r) {
        auto *row = packed.data() + r * rowBytes;
        float *expected = reference.data() + r * dim;
        if (type == GGML_TYPE_IQ4_NL) {
            for (int b = 0; b < dim / 32; ++b) {
                auto *block = row + b * 18;
                const float scale = std::ldexp(1.f, -8 + (r + b) % 5);
                uint16_t d = float_to_half(scale);
                memcpy(block, &d, 2);
                for (int i = 0; i < 16; ++i) {
                    block[2 + i] = uint8_t(r * 13 + b * 19 + i * 7);
                    expected[b * 32 + i] = scale * codebook[block[2 + i] & 15];
                    expected[b * 32 + i + 16] = scale * codebook[block[2 + i] >> 4];
                }
            }
        } else if (type == GGML_TYPE_MXFP8) {
            for (int b = 0; b < dim / 32; ++b) {
                auto *block = reinterpret_cast<block_mxfp8 *>(row) + b;
                block->e = uint8_t(r + b);
                for (int i = 0; i < 32; ++i) {
                    block->qs[i] = uint8_t(b * 32 + i + r);
                    expected[b * 32 + i] = ReferenceMXFP8(block->qs[i], block->e);
                }
            }
        } else {
            for (int c = 0; c < dim; ++c) expected[c] = float((r * 17 + c * 31) % 251 - 125) / 37;
            if (type == GGML_TYPE_Q5_K) quantize_row_q5_K_ref(expected, reinterpret_cast<block_q5_K *>(row), dim);
            else if (type == GGML_TYPE_Q6_K) quantize_row_q6_K_ref(expected, reinterpret_cast<block_q6_K *>(row), dim);
            else throw std::runtime_error("missing GGUF test encoder");
            // Existing K-quant decoders are the reference for row addressing,
            // disk transport and BF16 conversion, not under test themselves.
            ggml_type_to_float(type)(row, expected, dim);
        }
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
        struct GGUFProbe : Probe {
            void PrepareGGUF(ggml_type type, const std::vector<uint8_t> &packed, const DiskWeightPart &part, bool disk, bool async) {
                executionMode = async ? ExecutionMode::Hybrid : ExecutionMode::Cpu;
                ngramDevice = disk ? "disk" : "cpu"; engram_layer_ids.assign(1, 0); engram_head_dim = dim;
                engram_num_embeddings.assign(1, rows);
                weight.dicts["model_directory"] = "/";
                auto &w = weight.weight["layers.0.engram.embed.weight"];
                w = Data(DATA_GGUF_FORMAT, type, {rows, dim});
                w.isDiskWeight = disk; w.diskWeightParts = {part};
                if (!disk) { w.Allocate(false); memcpy(w.cpuData, packed.data(), packed.size()); }
                OnModelWeightsLoaded();
            }
        } probe;
        probe.PrepareGGUF(type, packed, part, disk, async);
        for (int tokens : {1, 9, 257, 1}) {
            std::vector<int64_t> ids(tokens * 24);
            for (size_t i = 0; i < ids.size(); ++i) ids[i] = i % 3 ? (i * 997 + tokens) % rows : rows - 1;
            Data output, prefetched;
            std::future<void> work;
            if (async) work = std::async(std::launch::async, [&]() {
                probe.GatherEngramRows(0, ids, tokens, prefetched);
            });
            probe.GatherEngramRows(0, ids, tokens, output);
            if (async) {
                work.get();
                Check(prefetched.dims == output.dims &&
                      !memcmp(prefetched.cpuData, output.cpuData, output.GetBytes()),
                      "background GGUF gather differs from synchronous gather");
            }
            Check(output.dims == std::vector<int>({1,tokens,24*dim}) && output.dataType == BFLOAT16, "GGUF shape/type");
            for (size_t i = 0; i < ids.size(); ++i) for (int c = 0; c < dim; ++c) {
                const float expected = reference[ids[i] * dim + c];
                const uint16_t actual = reinterpret_cast<uint16_t *>(output.cpuData)[i*dim+c];
                Check(std::isnan(expected) ? ((actual & 0x7f80) == 0x7f80 && (actual & 0x7f)) :
                      actual == Float32ToBFloat16RNEBits(expected), "GGUF gather differs from reference");
            }
        }
    }
    std::printf("PASS %s: CPU/disk, synchronous/background gather and repeated decode/prefill shapes\n", ggml_type_name(type));
}

static void TestNativePrefetch() {
    const FP8E4M3ToFP32Manager fp8;
    size_t checked = 0;
    int cases = 0;
    for (bool direct : {false, true}) {
        setenv("FASTLLM_DISK_DIRECT_IO", direct ? "1" : "0", 1);
        Fixture fixture;
        // Every row contains every FP8 byte; 256 rows cover all UE8M0 codes,
        // including zero/subnormal, overflow and reserved NaN scales.
        for (int row = 0; row < fixture.rows; ++row)
            for (int block = 0; block < fixture.dim / fixture.block; ++block)
                fixture.scales[row * (fixture.dim / fixture.block) + block] = row & 255;
        {
            std::fstream file(fixture.path, std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(fixture.scalePart.fileOffset);
            file.write(reinterpret_cast<const char *>(fixture.scales.data()), fixture.scales.size());
            Check(bool(file), "prefetch scale fixture write");
        }
        Probe serial;
        serial.Prepare(fixture, false);
        Data value(FP8_E4M3, {fixture.rows, fixture.dim});
        value.isDiskWeight = true; value.diskWeightParts = {fixture.valuePart};
        Data scale(INT8, {fixture.rows, fixture.dim / fixture.block});
        scale.isDiskWeight = true; scale.diskWeightParts = {fixture.scalePart};
        scale.diskWeightParts[0].isScalePart = false;
        DiskEmbeddingRowReader valueReader(value, 0), scaleReader(scale, 0);
        for (bool disk : {false, true}) for (int tokens : {1, 7, 128, 129, 8192}) {
            std::vector<int64_t> rows(tokens * 24);
            for (size_t i = 0; i < rows.size(); ++i) rows[i] = (i * 997 + tokens) % fixture.rows;
            rows[0] = fixture.rows - 1; rows[1] = 0; rows[2] = rows[0];
            const size_t count = rows.size() * fixture.dim;
            std::vector<uint16_t> output(count + 32, 0x5a5a);
            auto work = std::async(std::launch::async, [&]() {
                v41_engram_detail::GatherNativeRows(fixture.values.data(), fixture.scales.data(),
                    disk ? &valueReader : nullptr, disk ? &scaleReader : nullptr,
                    fixture.rows, fixture.dim, fixture.block, rows, output.data() + 16);
            });
            // Synchronous conversion uses the shared pool concurrently. The
            // background helper must not publish work into that same pool.
            Data reference;
            serial.GatherEngramRows(0, rows, tokens, reference);
            work.get();
            Check(!memcmp(output.data() + 16, reference.cpuData, count * 2),
                  "native prefetch differs from synchronous gather");
            for (int i = 0; i < 16; ++i)
                Check(output[i] == 0x5a5a && output[count + 16 + i] == 0x5a5a, "native prefetch guard");
            for (size_t i = 0; i < rows.size(); ++i) for (int col = 0; col < fixture.dim; ++col) {
                const uint8_t scaleCode = fixture.scales[rows[i] * (fixture.dim / fixture.block) + col / fixture.block];
                const float factor = scaleCode == 255 ? std::numeric_limits<float>::quiet_NaN()
                    : std::ldexp(1.f, int(scaleCode) - 127);
                const uint16_t expected = Float32ToBFloat16RNEBits(
                    fp8.dict[fixture.values[rows[i] * fixture.dim + col]] * factor);
                Check(output[16 + i * fixture.dim + col] == expected, "native prefetch scalar oracle");
            }
            checked += count; ++cases;
            std::printf("PASS native prefetch direct=%d disk=%d tokens=%d values=%zu bitwise=1\n",
                        int(direct), int(disk), tokens, count);
        }
        std::vector<uint16_t> empty(32, 0x5a5a);
        v41_engram_detail::GatherNativeRows(fixture.values.data(), fixture.scales.data(),
            &valueReader, &scaleReader, fixture.rows, fixture.dim, fixture.block, {}, empty.data() + 16);
        for (uint16_t bits : empty) Check(bits == 0x5a5a, "empty native prefetch guard");
    }
    std::printf("ALL PASS native prefetch cases=%d values=%zu bitwise=1\n", cases, checked);
}

// The future window must hash identically to committed history, including
// padding, dead-token boundaries and a final partial chunk. Request creation
// must snapshot the prompt without committing any of it to live history.
static void TestPromptWindows() {
    struct WindowProbe : DeepSeekV41Model {
        using DeepSeekV41Model::ComputeEngramHashes;
        using DeepSeekV41Model::GetOrCreateState;
        void Prepare() {
            engramMeta.loaded = true;
            engramMeta.padCompressedId = 2;
            engramMeta.tokenMap = {10,11,12,13,14,15,16};
            engram_max_ngram_size = 4; engram_n_heads = 2;
            engramMeta.multipliers = {{17, 151, 1637, 104729}};
            engramMeta.primes = {{101,103,107,109,113,127}};
            engramMeta.offsets = {{0,101,204,311,420,533}};
        }
    } model;
    model.Prepare();
    setenv("FASTLLM_DSV41_ENGRAM_PREFETCH", "1", 1);
    ResponseContext context;
    context.allTokens.resize(1031);
    for (size_t i = 0; i < context.allTokens.size(); ++i)
        context.allTokens[i] = i % 79 == 0 ? -1 : i % 7;
    model.OnResponseContextCreated(&context);
    auto state = model.GetOrCreateState(context.pastKeyValues, false);
    Check(state->engramHistory.empty() && state->totalLen == 0, "lookahead committed future history");
    Check(state->engramPrompt.size() == context.allTokens.size(), "prompt snapshot missing");
    std::vector<int64_t> expected;
    model.ComputeEngramHashes(0, state->engramPrompt, 0, state->engramPrompt.size(), expected);
    for (int chunk : {1,3,128,257}) {
        std::vector<int64_t> actual;
        for (int start = 0; start < int(state->engramPrompt.size()); start += chunk) {
            int count = std::min(chunk, int(state->engramPrompt.size()) - start);
            int lo = std::max(0, start - 3);
            std::vector<int> snapshot(state->engramPrompt.begin() + lo,
                                      state->engramPrompt.begin() + start + count);
            std::vector<int64_t> rows;
            model.ComputeEngramHashes(0, snapshot, start - lo, count, rows);
            actual.insert(actual.end(), rows.begin(), rows.end());
        }
        Check(actual == expected, "future-window hashes differ from full committed prompt");
    }
    const auto prompt = state->engramPrompt;
    state->engramHistory = {10}; state->totalLen = 1;
    model.GetOrCreateState(context.pastKeyValues, true);
    Check(state->engramHistory.empty() && state->engramPrompt == prompt, "initial reset lost prompt plan");
    std::weak_ptr<DeepSeekV41RequestState> lifetime = state;
    state.reset(); model.OnResponseContextRemoved(&context);
    Check(lifetime.expired(), "removed request retains lookahead state");
    setenv("FASTLLM_DSV41_ENGRAM_PREFETCH", "0", 1);
    model.OnResponseContextCreated(&context);
    Check(model.GetOrCreateState(context.pastKeyValues, false)->engramPrompt.empty(), "prefetch-off retained prompt");
    model.OnResponseContextRemoved(&context);
    unsetenv("FASTLLM_DSV41_ENGRAM_PREFETCH");
    std::puts("PASS Engram future-window hashes, request reset/removal and prefetch-off");
}

int main(int argc, char **argv) {
    try {
        SetThreads(4);
        TestMXFP8Decode();
        for (bool direct : {false, true}) {
            setenv("FASTLLM_DISK_DIRECT_IO", direct ? "1" : "0", 1);
            for (auto type : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP8}) TestGGUFRows(type);
        }
        if (argc == 2 && std::string(argv[1]) == "--gguf-only") return 0;
        TestPromptWindows();
        TestNativePrefetch();
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
