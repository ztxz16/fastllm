#include "models/minimax_h3_image/quantized.h"
#include "../../src/models/minimax_h3_image/gguf_checkpoint.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unistd.h>
using namespace fastllm::minimax_h3;
using namespace fastllm::minimax_h3_image;
namespace fs = std::filesystem;
void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
float bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 32767 + ((u >> 16) & 1);
    u &= 0xffff0000;
    std::memcpy(&x, &u, 4);
    return x;
}
float half(const uint8_t *p) {
    int h = p[0] | (p[1] << 8), e = (h >> 10) & 31;
    return (h & 32768 ? -1.f : 1.f) *
           (e ? std::ldexp(1.f + (h & 1023) / 1024.f, e - 15) : std::ldexp(float(h & 1023), -24));
}
// Scalar CPU oracle follows group/lane iteration, independent of CUDA's
// per-element indexing. Real payloads are also checked against gguf-py.
std::vector<float> reference(const std::vector<uint8_t> &data, int type, size_t count) {
    auto block = detail::ggufBlock(type);
    std::vector<float> out(count);
    for (size_t off = 0; off < count; off += block.values) {
        const auto *p = data.data() + off / block.values * block.bytes;
        auto *y = out.data() + off;
        if (type == 0)
            std::memcpy(y, p, 4);
        else if (type == 1)
            y[0] = half(p);
        else if (type == 30) {
            uint32_t bits = uint32_t(p[0] | (p[1] << 8)) << 16;
            std::memcpy(y, &bits, 4);
        } else if (type == 2)
            for (int j = 0; j < 16; ++j) {
                y[j] = half(p) * ((p[2 + j] & 15) - 8);
                y[j + 16] = half(p) * ((p[2 + j] >> 4) - 8);
            }
        else if (type == 8)
            for (int j = 0; j < 32; ++j)
                y[j] = half(p) * int8_t(p[2 + j]);
        else if (type == 10 || type == 11) {
            int scales[16]{};
            if (type == 11)
                for (int j = 0; j < 4; ++j)
                    for (int part = 0; part < 4; ++part)
                        scales[part * 4 + j] = ((p[96 + (part % 2) * 4 + j] >> (part / 2 * 4)) & 15) +
                                               (((p[104 + j] >> (part * 2)) & 3) << 4) - 32;
            for (int h = 0; h < 2; ++h)
                for (int shift = 0; shift < 4; ++shift)
                    for (int lane = 0; lane < 32; ++lane) {
                        int index = h * 128 + shift * 32 + lane, group = h * 8 + shift * 2 + lane / 16;
                        int q = (p[(type == 10 ? 16 : 32) + h * 32 + lane] >> (shift * 2)) & 3;
                        if (type == 10)
                            y[index] = (half(p + 80) * (p[group] & 15)) * q - half(p + 82) * (p[group] >> 4);
                        else {
                            if (!(p[lane] & (1 << (h * 4 + shift))))
                                q -= 4;
                            y[index] = (half(p + 108) * scales[group]) * q;
                        }
                    }
        } else if (type == 12 || type == 13) {
            int scales[8], mins[8];
            for (int j = 0; j < 4; ++j) {
                scales[j] = p[4 + j] & 63;
                mins[j] = p[8 + j] & 63;
                scales[j + 4] = (p[12 + j] & 15) + ((p[4 + j] >> 6) << 4);
                mins[j + 4] = (p[12 + j] >> 4) + ((p[8 + j] >> 6) << 4);
            }
            for (int pair = 0; pair < 4; ++pair)
                for (int lane = 0; lane < 32; ++lane)
                    for (int high = 0; high < 2; ++high) {
                        int g = pair * 2 + high,
                            q = (p[(type == 12 ? 16 : 48) + pair * 32 + lane] >> (4 * high)) & 15;
                        if (type == 13 && (p[16 + lane] & (1 << g)))
                            q += 16;
                        y[g * 32 + lane] = (half(p) * scales[g]) * q - half(p + 2) * mins[g];
                    }
        } else if (type == 14) {
            for (int h = 0; h < 2; ++h)
                for (int lane = 0; lane < 32; ++lane)
                    for (int quarter = 0; quarter < 4; ++quarter) {
                        int lo = (p[h * 64 + (quarter % 2) * 32 + lane] >> (4 * (quarter / 2))) & 15;
                        int hi = (p[128 + h * 32 + lane] >> (2 * quarter)) & 3;
                        y[h * 128 + quarter * 32 + lane] =
                            (half(p + 208) * int8_t(p[192 + h * 8 + quarter * 2 + lane / 16])) *
                            ((lo | hi << 4) - 32);
                    }
        }
    }
    for (auto &v : out)
        v = bf16(v);
    return out;
}
void append(std::vector<uint8_t> &v, uint64_t n, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i)
        v.push_back((n >> (8 * i)) & 255);
}
void string(std::vector<uint8_t> &v, const std::string &s) {
    append(v, s.size(), 8);
    v.insert(v.end(), s.begin(), s.end());
}
struct Fixture {
    std::vector<uint8_t> bytes, data;
    size_t metadataEnd, rankOffset, typeOffset;
};
Fixture fixture(int type, int rows = 31, int cols = 256) {
    Fixture f;
    auto block = detail::ggufBlock(type);
    f.data.resize(size_t(rows) * cols / block.values * block.bytes);
    std::mt19937 rng(1234 + type);
    for (auto &v : f.data)
        v = rng() & 255;
    auto setHalf = [](uint8_t *p, unsigned bits) {
        p[0] = bits & 255;
        p[1] = bits >> 8;
    };
    for (size_t i = 0; i < f.data.size(); i += block.bytes) {
        auto *p = f.data.data() + i;
        if (type == 0) {
            float v = (int(rng() % 1001) - 500) / 128.f;
            std::memcpy(p, &v, 4);
        } else if (type == 30)
            setHalf(p, 0x3f80 | (rng() & 0x807f));
        else if (type == 1)
            setHalf(p, 0x3800 | (rng() & 0x83ff));
        else if (type == 10) {
            setHalf(p + 80, 0x2000);
            setHalf(p + 82, 0x1800);
        } else if (type == 11)
            setHalf(p + 108, 0x2400);
        else if (type == 14)
            setHalf(p + 208, 0x2000);
        else {
            setHalf(p, 0x2400);
            if (type == 12 || type == 13)
                setHalf(p + 2, 0x1800);
        }
    }
    auto &v = f.bytes;
    append(v, 0x46554747, 4);
    append(v, 3, 4);
    append(v, 1, 8);
    append(v, 1, 8);
    string(v, "general.architecture");
    append(v, 8, 4);
    string(v, "wan");
    f.metadataEnd = v.size();
    string(v, "blocks.0.mlp.fc2.weight");
    f.rankOffset = v.size();
    append(v, 2, 4);
    append(v, cols, 8);
    append(v, rows, 8);
    f.typeOffset = v.size();
    append(v, type, 4);
    append(v, 0, 8);
    v.resize((v.size() + 31) / 32 * 32);
    v.insert(v.end(), f.data.begin(), f.data.end());
    return f;
}
void write(const fs::path &p, const std::vector<uint8_t> &data) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), data.size());
}
template <class F> void rejects(F fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "accepted malformed GGUF");
}
int main() {
    auto root = fs::temp_directory_path() / ("h3-gguf-" + std::to_string(getpid()));
    fs::create_directories(root);
    auto file = root / "arbitrary-name.gguf";
    try {
        auto f = fixture(12);
        write(file, f.bytes);
        require(inspectWeightFormat(file.string()) == WeightFormat::GGUF, "GGUF magic detection");
        for (auto name : {"transformer/config.json", "vae/config.json", "text_encoder/config.json",
                          "pdd_config.json", "pdd_fused_heads.safetensors"}) {
            auto p = root / "base_model" / name;
            fs::create_directories(p.parent_path());
            std::ofstream(p) << "{}";
        }
        fs::create_directories(root / "base_model/tokenizer");
        auto selected = resolveModelSource(file.string());
        require(selected.format == WeightFormat::GGUF && selected.directory == (root / "base_model").string(),
                "GGUF base resolution");
        require(resolveModelSource(root.string()).quantizedWeights == file.string(),
                "GGUF directory discovery");
        fs::copy_file(file, root / "second.gguf");
        rejects([&] { resolveModelSource(root.string()); });
        require(resolveModelSource(file.string()).format == WeightFormat::GGUF, "explicit GGUF selection");
        fs::remove(root / "second.gguf");
        fs::rename(file, root / "misleading.safetensors");
        require(resolveModelSource((root / "misleading.safetensors").string()).format == WeightFormat::GGUF,
                "GGUF detection relied on suffix");
        fs::rename(root / "misleading.safetensors", file);
        detail::GgufCheckpoint checkpoint(file.string());
        rejects([&] { checkpoint.validateModel(); });
        for (uint32_t item : {4u, 9u, 13u}) {
            std::vector<uint8_t> metadata;
            string(metadata, "test.array");
            append(metadata, 9, 4);
            append(metadata, item, 4);
            append(metadata, 2, 8);
            append(metadata, 123, 4);
            append(metadata, 456, 4);
            std::vector<uint8_t> bytes(f.bytes.begin(), f.bytes.begin() + f.typeOffset + 12);
            bytes[16] = 2;
            bytes.insert(bytes.begin() + f.metadataEnd, metadata.begin(), metadata.end());
            bytes.resize((bytes.size() + 31) / 32 * 32);
            bytes.insert(bytes.end(), f.data.begin(), f.data.end());
            write(file, bytes);
            if (item == 4) {
                detail::GgufCheckpoint withArray(file.string());
                const auto &tensor = withArray.tensor("blocks.0.mlp.fc2.weight");
                std::vector<uint8_t> data(tensor.bytes);
                withArray.read(tensor, 0, data.data(), data.size());
                require(data == f.data, "metadata array corrupted payload position");
            } else {
                rejects([&] { detail::GgufCheckpoint bad(file.string()); });
            }
        }
        for (int invalid = 0; invalid < 7; ++invalid) {
            auto b = f.bytes;
            if (invalid == 0)
                b[4] = 1;
            if (invalid == 1)
                b[f.rankOffset] = 5;
            if (invalid == 2)
                std::fill(b.begin() + f.rankOffset + 4, b.begin() + f.rankOffset + 12, 0);
            if (invalid == 3)
                b[f.typeOffset] = 255;
            if (invalid == 4)
                b.resize(b.size() - 1);
            if (invalid == 5)
                b[8] = 255;
            if (invalid == 6)
                b[f.typeOffset + 4] = 1;
            write(file, b);
            rejects([&] { detail::GgufCheckpoint bad(file.string()); });
        }
        int count = 0;
        cudaDeviceProp prop{};
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count ||
            cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 8) {
            fs::remove_all(root);
            return 77;
        }
        Runtime rt(0, DType::BF16);
        for (int type : {0, 1, 30, 2, 8, 10, 11, 12, 13, 14}) {
            f = fixture(type);
            write(file, f.bytes);
            auto w = QuantizedWeights::load(file.string());
            auto expected = reference(f.data, type, 31 * 256);
            std::vector<float> identity(256 * 256);
            for (int i = 0; i < 256; ++i)
                identity[i * 256 + i] = 1;
            auto x = rt.upload(identity.data(), 256, 256);
            // Keep the decode oracle on the small-row BF16 fallback. MMQ
            // additionally quantizes activations and has a separate L2 check.
            std::vector<float> actual;
            for (int row = 0; row < 256; row += 7) {
                auto part = rt.download(w->linear(x.rowSlice(row, std::min(7, 256-row)), "blocks.0.mlp.fc2"));
                actual.insert(actual.end(), part.begin(), part.end());
            }
            for (int r = 0; r < 256; ++r)
                for (int c = 0; c < 31; ++c)
                    require(actual[r * 31 + c] == expected[c * 256 + r],
                            "GGUF decoded element differs from CPU oracle");
            w->erasePrefix("blocks.0.");
            w->clear();
            auto reloaded = rt.download(w->linear(x.rowSlice(0, 7), "blocks.0.mlp.fc2"));
            require(std::equal(reloaded.begin(), reloaded.end(), actual.begin()), "GGUF reload differs");
        }
        // Dense Q8-activation MMQ: checked output tiles, partial input tiles,
        // bounded activation chunks, the BF16 cutoff and caller-owned output views.
        for (int type : {2, 8, 10, 11, 12, 13, 14}) {
            for (int rows : {8, 17, 63, 137, 511, 512, 513, 1025}) {
                if (rows >= 511 && type != 12) continue;
                const int columns = rows == 511 ? 32769 : 131;
                f = fixture(type, columns);
                write(file, f.bytes);
                auto weights = QuantizedWeights::load(file.string());
                auto values = reference(f.data, type, columns * 256);
                std::mt19937 gen(123);
                std::normal_distribution<float> normal;
                std::vector<float> input(rows*256);
                for (auto &v : input) v = bf16(normal(gen)*.2f);
                auto x = rt.upload(input.data(), rows, 256);
                auto expected = rt.download(rt.linear(x, rt.upload(values.data(), columns, 256)));
                std::vector<float> sentinels(size_t(rows+2)*columns, -123.f);
                auto storage = rt.upload(sentinels.data(), rows+2, columns);
                auto view = storage.rowSlice(1, rows);
                weights->linear(x, "blocks.0.mlp.fc2", &view);
                auto actual = rt.download(view);
                double diff = 0, norm = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    require(std::isfinite(actual[i]), "non-finite MMQ output");
                    diff += std::pow(double(actual[i])-expected[i], 2);
                    norm += double(expected[i])*expected[i];
                }
                const double error = std::sqrt(diff/std::max(norm, 1e-30));
                std::cout << "GGUF GEMM type " << type << " rows " << rows << " relative_l2 " << error << '\n';
                require(error < .02, "GGUF MMQ relative L2 exceeds 2%");
                if (rows > 512)
                    require(actual == expected, "large-row BF16 fallback changed");
                auto all = rt.download(storage);
                for (int c = 0; c < columns; ++c)
                    require(all[c] == -123.f && all[size_t(rows+1)*columns+c] == -123.f,
                            "GGUF MMQ output clobbered adjacent rows");
            }
        }
        // Cross decode-chunk boundary, a non-square GEMM, and output row views.
        f = fixture(12, 2051);
        write(file, f.bytes);
        auto w = QuantizedWeights::load(file.string());
        auto values = reference(f.data, 12, 2051 * 256);
        std::vector<float> input(7 * 256);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = bf16((int(i * 17 % 101) - 50) / 128.f);
        auto x = rt.upload(input.data(), 7, 256), ref = rt.linear(x, rt.upload(values.data(), 2051, 256));
        std::vector<float> sentinels(9 * 2051, -123.f);
        auto storage = rt.upload(sentinels.data(), 9, 2051), target = storage.rowSlice(1, 7);
        w->linear(x, "blocks.0.mlp.fc2", &target);
        require(rt.download(target) == rt.download(ref), "GGUF GEMM/output stride differs");
        auto out = rt.download(storage);
        for (int c = 0; c < 2051; ++c)
            require(out[c] == -123.f && out[8 * 2051 + c] == -123.f, "GGUF output clobbered adjacent rows");
        rejects([&] { w->linear(x, "blocks.0.mlp.fc2", &x); });
        rt.synchronize();
        fs::remove_all(root);
        std::cout << "GGUF reader, 10 CUDA decode types, GEMM, reload and output views passed\n";
    } catch (const std::exception &e) {
        fs::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
