#include "model.h"
#include "models/qwen4_exp.h"
#include "executor.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fastllm {
struct Qwen4PrefixCacheTestAccess {
    static void BeginStreaming(Qwen4ExpModel &model, const std::string &name) {
        model.streamingTpLoad = true;
        model.OnWeightLoadGroupStarted({name});
    }
    static bool GroupContains(const Qwen4ExpModel &model, const std::string &name) {
        return model.streamingTpNames.count(name) != 0;
    }
    static Data &DiskPLE(Qwen4ExpModel &model) { return model.pleNgramDiskWeight; }
};
}

static void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void TestMergedFrontier() {
    using namespace fastllm;
    Qwen4ExpModel model;
    const std::string prefix = "model.language_model.layers.0.mlp.experts.0.";
    const std::string gate = prefix + "gate_proj.weight";
    const std::string up = prefix + "up_proj.weight";
    const std::string merged = prefix + "gateup_proj.weight";
    Qwen4PrefixCacheTestAccess::BeginStreaming(model, gate);
    // The loader reports a newly generated name after it merges gate/up.
    // It must belong to the same frontier even though it is not in the file.
    model.weight.AddEmptyWeight(merged, {256, 16}, FLOAT16);
    model.OnWeightLoaded(merged, {gate, up, merged});
    Require(Qwen4PrefixCacheTestAccess::GroupContains(model, merged),
            "merged gateup missing from streaming frontier; CPU storage would survive");
    std::cout << "PASS merged gateup registered for upload/release\n";
}

static void TestPLE(const std::string &path, bool real = false) {
    using namespace fastllm;
    SetDeviceMap({{"cpu", 1}});
    SetMoeDeviceMap({{"cpu", 1}});
    SetThreads(4);
    SetNgramDevice("disk");
    auto base = CreateLLMModelFromHF(path, FLOAT16, -1, true);
    auto *model = dynamic_cast<Qwen4ExpModel *>(base.get());
    Require(model && model->ngramDevice == "disk", "model ignored disk ngram selection");
    const std::string prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
    uint64_t diskBytes = 0;
    const int shards = real ? 128 : 2;
    for (int i = 0; i < shards; ++i) {
        const auto &shard = model->weight.weight.at(prefix + "shard_" + std::to_string(i) + ".weight");
        Require(shard.isDiskWeight && !shard.isFake && shard.cpuData == nullptr &&
                shard.cudaData == nullptr && shard.expansionBytes == 0 &&
                !shard.diskWeightParts.empty(), "PLE shard has resident payload");
        diskBytes += shard.diskWeightParts.front().bytes;
    }
    Data &combined = Qwen4PrefixCacheTestAccess::DiskPLE(*model);
    Require(combined.isDiskWeight && combined.cpuData == nullptr &&
            combined.dims.size() == 2 && combined.dims[0] > 0 && combined.dims[1] > 0,
            "combined disk PLE is invalid");
    if (!real) Require(combined.dims == std::vector<int>({6, 16}), "fixture PLE shape differs");
    Data rows(INT32, {4});
    rows.Allocate(false);
    const int32_t ids[] = {0, combined.dims[0] / 2 - 1,
                          combined.dims[0] / 2, combined.dims[0] - 1};
    std::memcpy(rows.cpuData, ids, sizeof(ids));
    Data result;
    static_cast<Executor *>(GetExecutor())->RunOnDevice(
        "disk", "EmbeddingDirect", {{"input", &rows}, {"weight", &combined}, {"output", &result}}, {}, {});
    Require(result.cpuData && result.Count(0) == 4ULL * combined.dims[1] && result.dataType == combined.dataType,
            "disk row output layout differs");
    std::vector<uint8_t> expected;
    if (real) {
        for (int32_t row : ids) {
            uint64_t offsetRows = 0;
            bool found = false;
            for (const auto &part : combined.diskWeightParts) {
                if (row < offsetRows + part.dims[0]) {
                    const size_t bytesPerRow = part.bytes / part.dims[0];
                    std::ifstream file(part.fileName, std::ios::binary);
                    file.seekg(part.fileOffset + (row - offsetRows) * bytesPerRow);
                    const size_t start = expected.size();
                    expected.resize(start + bytesPerRow);
                    file.read(reinterpret_cast<char *>(expected.data() + start), bytesPerRow);
                    Require(file.gcount() == static_cast<std::streamsize>(bytesPerRow), "original PLE row truncated");
                    found = true;
                    break;
                }
                offsetRows += part.dims[0];
            }
            Require(found, "real PLE row outside original shards");
        }
    } else {
        std::ifstream expectedFile(path + "/expected.bin", std::ios::binary);
        expected.assign(std::istreambuf_iterator<char>(expectedFile), {});
    }
    Require(expected.size() == result.GetBytes() &&
            std::memcmp(expected.data(), result.cpuData, expected.size()) == 0,
            "first/middle/last PLE rows differ from original file bytes");
    Require(combined.cpuData == nullptr, "row lookup retained full PLE table");
    std::cout << "PASS PLE disk metadata and exact row reads: host_payload_bytes=0 disk_bytes="
              << diskBytes << " shards=" << shards << "\n";
}

int main(int argc, char **argv) {
    try {
        unsetenv("FASTLLM_TP");
        setenv("FASTLLM_QWEN4_ENABLE_MTP", "1", 1);
        if (argc == 3 && std::string(argv[1]) == "--ple") TestPLE(argv[2]);
        else if (argc == 3 && std::string(argv[1]) == "--real") TestPLE(argv[2], true);
        else if (argc == 2 && std::string(argv[1]) == "--group") TestMergedFrontier();
        else throw std::runtime_error("use --ple FIXTURE, --real PLE_VIEW or --group");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
