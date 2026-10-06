#include "devices/disk/diskdevice.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unistd.h>

using namespace fastllm;
static void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
struct Fixture {
    std::string path;
    Data weight;
    Fixture(DataType type, DataType sourceType, int columns) {
        char name[] = "/tmp/fastllm-ple-reader-XXXXXX";
        int fd = mkstemp(name); Check(fd >= 0, "mkstemp"); close(fd); path = name;
        weight.dataType = type; weight.ggmlType = GGML_TYPE_IQ4_NL;
        weight.isGGUFData = type == DATA_GGUF_FORMAT;
        weight.UpdateUnitSize(); weight.Resize({35, columns}); weight.isDiskWeight = true;
        const size_t sourceRowBytes = sourceType == DATA_GGUF_FORMAT
            ? ggml_row_size(GGML_TYPE_IQ4_NL, columns) : GetDataBytes(sourceType, 1, columns);
        std::ofstream file(path, std::ios::binary);
        std::vector<uint8_t> padding(4093, 0); file.write((char*)padding.data(), padding.size());
        for (int shard = 0, first = 0; shard < 2; ++shard) {
            DiskWeightPart part;
            part.fileName = path; part.fileOffset = file.tellp(); part.sourceDataType = sourceType;
            part.dims = {shard ? 18 : 17, columns}; part.bytes = sourceRowBytes * part.dims[0];
            weight.diskWeightParts.push_back(part);
            for (int row = 0; row < part.dims[0]; ++row, ++first) {
                std::vector<uint8_t> bytes(sourceRowBytes);
                for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (first * 31 + i * 7) % 128;
                if (sourceType == FLOAT32) {
                    for (int c = 0; c < columns; ++c) ((float*)bytes.data())[c] = (first - c) * .125f;
                } else if (sourceType == DATA_GGUF_FORMAT) {
                    for (size_t i = 0; i < bytes.size(); i += 18) {
                        const uint16_t d = float_to_half(.125f);
                        memcpy(bytes.data() + i, &d, sizeof(d));
                    }
                }
                file.write((char*)bytes.data(), bytes.size());
            }
        }
        file.write((char*)padding.data(), padding.size());
    }
    ~Fixture() { std::remove(path.c_str()); }
};
static void Verify(Fixture &f, const DiskEmbeddingRowReader::Ticket &ticket) {
    Data ids(INT32, {(int)ticket.rows.size()}), expected;
    ids.Allocate(); memcpy(ids.cpuData, ticket.rows.data(), ticket.rows.size() * sizeof(int32_t));
    DiskEmbeddingOp op(true);
    DataDict data{{"input", &ids}, {"weight", &f.weight}, {"output", &expected}};
    op.Reshape("EmbeddingDirect", data, {}, {}); op.Run("EmbeddingDirect", data, {}, {});
    const auto &raw = ticket.values.get();
    if (f.weight.dataType == DATA_GGUF_FORMAT) {
        const int columns = f.weight.dims[1];
        std::vector<float> actual(ticket.rows.size() * columns);
        const auto decode = ggml_type_to_float(GGML_TYPE_IQ4_NL);
        for (size_t r = 0; r < ticket.rows.size(); ++r)
            decode(raw.data() + r * ggml_row_size(GGML_TYPE_IQ4_NL, columns), actual.data() + r * columns, columns);
        Check(actual.size() * sizeof(float) == expected.GetBytes() &&
              !memcmp(actual.data(), expected.cpuData, expected.GetBytes()), "GGUF bitwise mismatch");
    } else {
        Check(raw.size() == expected.GetBytes() && !memcmp(raw.data(), expected.cpuData, raw.size()),
              "native/conversion bitwise mismatch");
    }
}
int main() {
    try {
        for (int direct : {0, 1}) {
            setenv("FASTLLM_DISK_DIRECT_IO", direct ? "1" : "0", 1);
            for (auto types : {std::make_pair(DATA_GGUF_FORMAT, DATA_GGUF_FORMAT),
                               std::make_pair(FP8_E4M3, FP8_E4M3),
                               std::make_pair(BFLOAT16, BFLOAT16),
                               std::make_pair(FLOAT16, FLOAT32)}) {
                for (int columns : {32, 160, 256}) {
                    Fixture f(types.first, types.second, columns);
                    for (size_t budget : {size_t(0), size_t(512), size_t(65536)}) {
                        DiskEmbeddingRowReader reader(f.weight, budget);
                        const std::vector<int32_t> rows{34, 0, 17, 16, 34, 1, 0};
                        Verify(f, reader.ReadAsync(rows));
                        auto stats = reader.GetStats();
                        Check(stats.reads == 5 && stats.cacheBytes <= budget, "dedup/budget failure");
                        Verify(f, reader.ReadAsync(rows));
                        if (budget == 65536) Check(reader.GetStats().reads == 5, "cache hit performed I/O");
                        if (budget == 0) Check(reader.GetStats().reads == 10, "zero cache retained rows");
                        std::vector<int32_t> all(35); std::iota(all.begin(), all.end(), 0);
                        auto a = reader.ReadAsync(all), b = reader.ReadAsync({1, 2, 3, 1, 34});
                        Verify(f, a); Verify(f, b); Verify(f, reader.ReadAsync(rows));
                    }
                    DiskEmbeddingRowReader::Ticket pending;
                    {
                        DiskEmbeddingRowReader reader(f.weight, 0);
                        pending = reader.ReadAsync({34, 0, 17, 17});
                    }
                    Verify(f, pending); // destruction drains outstanding work
                    std::cout << "PASS reader " << direct << ' ' << (int)types.first << ' ' << columns << '\n';
                }
            }
        }
        std::cout << "PASS disk embedding reader cache, shards, dtypes, concurrent tickets and lifetime\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
