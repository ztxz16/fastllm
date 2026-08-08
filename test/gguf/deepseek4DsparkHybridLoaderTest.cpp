#include "deepseekv4.h"
#include "model.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

namespace {
    using fastllm::basellm;
    using fastllm::DataType;
    using fastllm::WeightMergeRule;
    using fastllm::WeightMergeRuleSingle;

    // ---------- safetensors fixture writer ----------

    struct FixtureTensor {
        std::string name;
        std::string dtype; // BF16 / F32 / F8_E4M3
        std::vector<int> shape;
        std::vector<uint8_t> bytes;
    };

    uint16_t FloatToBf16(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (uint16_t)(bits >> 16);
    }

    float Bf16ToFloat(uint16_t value) {
        uint32_t bits = (uint32_t)value << 16;
        float ret;
        std::memcpy(&ret, &bits, sizeof(ret));
        return ret;
    }

    FixtureTensor MakeBf16Tensor(const std::string &name,
                                 const std::vector<int> &shape,
                                 const std::vector<float> &values) {
        FixtureTensor tensor;
        tensor.name = name;
        tensor.dtype = "BF16";
        tensor.shape = shape;
        for (float value : values) {
            uint16_t encoded = FloatToBf16(value);
            tensor.bytes.push_back((uint8_t)(encoded & 0xFF));
            tensor.bytes.push_back((uint8_t)(encoded >> 8));
        }
        return tensor;
    }

    FixtureTensor MakeF32Tensor(const std::string &name,
                                const std::vector<int> &shape,
                                const std::vector<float> &values) {
        FixtureTensor tensor;
        tensor.name = name;
        tensor.dtype = "F32";
        tensor.shape = shape;
        for (float value : values) {
            uint8_t raw[4];
            std::memcpy(raw, &value, 4);
            tensor.bytes.insert(tensor.bytes.end(), raw, raw + 4);
        }
        return tensor;
    }

    FixtureTensor MakeF8Tensor(const std::string &name,
                               const std::vector<int> &shape,
                               const std::vector<uint8_t> &encoded) {
        FixtureTensor tensor;
        tensor.name = name;
        tensor.dtype = "F8_E4M3";
        tensor.shape = shape;
        tensor.bytes = encoded;
        return tensor;
    }

    struct ShardLayoutEntry {
        const FixtureTensor *tensor;
        uint64_t absoluteOffset;
    };

    // Writes u64 header length + JSON header + raw payload.  Returns each
    // tensor's absolute file offset so tests can verify disk metadata.
    std::map<std::string, uint64_t> WriteSafetensorsShard(
            const std::filesystem::path &path,
            const std::vector<FixtureTensor> &tensors) {
        json11::Json::object header;
        uint64_t cursor = 0;
        for (const auto &tensor : tensors) {
            json11::Json::array shapeArray;
            for (int dim : tensor.shape) {
                shapeArray.push_back(dim);
            }
            json11::Json::array offsets;
            // json11's ll_value() is only implemented for its long-long
            // variant, so emit plain integers (as real safetensors do);
            // dumped doubles round-trip as 0.
            offsets.push_back((long long)cursor);
            offsets.push_back((long long)(cursor + tensor.bytes.size()));
            header[tensor.name] = json11::Json::object {
                {"dtype", tensor.dtype},
                {"shape", shapeArray},
                {"data_offsets", offsets},
            };
            cursor += tensor.bytes.size();
        }
        const std::string headerString = json11::Json(header).dump();
        const uint64_t headerLength = headerString.size();

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        assert(out.is_open());
        out.write(reinterpret_cast<const char*>(&headerLength), 8);
        out.write(headerString.data(), (std::streamsize)headerLength);
        std::map<std::string, uint64_t> absoluteOffsets;
        uint64_t base = 8 + headerLength;
        for (const auto &tensor : tensors) {
            absoluteOffsets[tensor.name] = base;
            out.write(reinterpret_cast<const char*>(tensor.bytes.data()),
                      (std::streamsize)tensor.bytes.size());
            base += tensor.bytes.size();
        }
        assert(out.good());
        return absoluteOffsets;
    }

    void WriteIndexJson(const std::filesystem::path &path,
                        const std::map<std::string, std::string> &weightMap) {
        json11::Json::object weightMapJson;
        for (const auto &entry : weightMap) {
            weightMapJson[entry.first] = entry.second;
        }
        json11::Json::object root;
        root["metadata"] = json11::Json::object {{"total_size", 4096}};
        root["weight_map"] = weightMapJson;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        assert(out.is_open());
        const std::string dumped = json11::Json(root).dump();
        out.write(dumped.data(), (std::streamsize)dumped.size());
        assert(out.good());
    }

    struct ScopedTempDir {
        std::filesystem::path path;

        explicit ScopedTempDir(const std::string &tag) {
            static int counter = 0;
            path = std::filesystem::temp_directory_path() /
                ("fastllmDsparkHybrid_" + tag + "_" +
                 std::to_string(::getpid()) + "_" + std::to_string(counter++));
            std::filesystem::create_directories(path);
        }

        ~ScopedTempDir() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    };

    void ExpectFailure(const std::function<void()> &operation,
                       const std::string &messagePart) {
        std::string message;
        try {
            operation();
        } catch (const std::runtime_error &error) {
            message = error.what();
        }
        assert(!message.empty());
        assert(message.find(messagePart) != std::string::npos);
    }

    std::unique_ptr<basellm> MakeModel() {
        return fastllm::CreateEmptyLLMModel("deepseek_v4");
    }

    // ---------- tests ----------

    void TestShardedPlanFiltersBackbone() {
        ScopedTempDir dir("plan");
        const std::string shard1 = "model-00001-of-00002.safetensors";
        const std::string shard2 = "model-00002-of-00002.safetensors";
        WriteSafetensorsShard(dir.path / shard1, {
            MakeBf16Tensor("mtp.0.main_proj.weight", {2, 3},
                           {1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f}),
            MakeBf16Tensor("mtp.0.main_norm.weight", {4},
                           {1.0f, 1.0f, 1.0f, 1.0f}),
            MakeBf16Tensor("layers.0.attn.wq_a.weight", {2, 2},
                           {0.0f, 0.0f, 0.0f, 0.0f}),
        });
        WriteSafetensorsShard(dir.path / shard2, {
            MakeBf16Tensor("mtp.2.norm.weight", {4},
                           {1.0f, 1.0f, 1.0f, 1.0f}),
            MakeBf16Tensor("mtp.0.ffn.experts.5.w1.weight", {2, 2},
                           {1.0f, 2.0f, 3.0f, 4.0f}),
        });
        WriteIndexJson(dir.path / "model.safetensors.index.json", {
            {"mtp.0.main_proj.weight", shard1},
            {"mtp.0.main_norm.weight", shard1},
            {"layers.0.attn.wq_a.weight", shard1},
            {"mtp.2.norm.weight", shard2},
            {"mtp.0.ffn.experts.5.w1.weight", shard2},
        });

        auto plan = fastllm::PlanDeepSeekV4DSparkShards(dir.path.string());
        assert(plan.tensorNames.size() == 4);
        assert(plan.tensorNames == std::vector<std::string>({
            "mtp.0.ffn.experts.5.w1.weight",
            "mtp.0.main_norm.weight",
            "mtp.0.main_proj.weight",
            "mtp.2.norm.weight",
        }));
        assert(plan.shardFiles.size() == 2);
        assert(plan.shardFiles.count((dir.path / shard1).string()) == 1);
        assert(plan.shardFiles.count((dir.path / shard2).string()) == 1);
        std::cout << "Sharded plan filters backbone entries: passed.\n";
    }

    void TestPlanRejectsBackboneOnlyIndex() {
        ScopedTempDir dir("backboneOnly");
        const std::string shard1 = "model-00001-of-00001.safetensors";
        WriteSafetensorsShard(dir.path / shard1, {
            MakeBf16Tensor("layers.0.attn.wq_a.weight", {2, 2},
                           {0.0f, 0.0f, 0.0f, 0.0f}),
        });
        WriteIndexJson(dir.path / "model.safetensors.index.json", {
            {"layers.0.attn.wq_a.weight", shard1},
        });
        ExpectFailure([&] {
            fastllm::PlanDeepSeekV4DSparkShards(dir.path.string());
        }, "no mtp.* tensors");
        std::cout << "Plan rejects backbone-only index: passed.\n";
    }

    void TestPlanRejectsMissingShard() {
        ScopedTempDir dir("missingShard");
        WriteIndexJson(dir.path / "model.safetensors.index.json", {
            {"mtp.0.main_norm.weight", "model-00046-of-00048.safetensors"},
        });
        ExpectFailure([&] {
            fastllm::PlanDeepSeekV4DSparkShards(dir.path.string());
        }, "missing");
        std::cout << "Plan rejects missing shard: passed.\n";
    }

    void TestSingleFilePlans() {
        ScopedTempDir ok("singleOk");
        WriteSafetensorsShard(ok.path / "model.safetensors", {
            MakeBf16Tensor("mtp.0.main_norm.weight", {4},
                           {1.0f, 1.0f, 1.0f, 1.0f}),
            MakeBf16Tensor("mtp.2.hc_head_fn", {2}, {1.0f, 1.0f}),
        });
        auto plan = fastllm::PlanDeepSeekV4DSparkShards(ok.path.string());
        assert(plan.tensorNames.size() == 2);
        assert(plan.shardFiles.size() == 1);

        ScopedTempDir mixed("singleMixed");
        WriteSafetensorsShard(mixed.path / "model.safetensors", {
            MakeBf16Tensor("mtp.0.main_norm.weight", {4},
                           {1.0f, 1.0f, 1.0f, 1.0f}),
            MakeBf16Tensor("embed.weight", {2, 2},
                           {0.0f, 0.0f, 0.0f, 0.0f}),
        });
        ExpectFailure([&] {
            fastllm::PlanDeepSeekV4DSparkShards(mixed.path.string());
        }, "must only contain mtp.*");

        ScopedTempDir empty("singleNone");
        ExpectFailure([&] {
            fastllm::PlanDeepSeekV4DSparkShards(empty.path.string());
        }, "neither");
        std::cout << "Single-file plans: passed.\n";
    }

    void TestSequentialImportValues() {
        ScopedTempDir dir("values");
        auto offsets = WriteSafetensorsShard(dir.path / "model.safetensors", {
            MakeBf16Tensor("mtp.0.main_norm.weight", {4},
                           {1.5f, -2.0f, 3.25f, 0.5f}),
        });
        auto plan = fastllm::PlanDeepSeekV4DSparkShards(dir.path.string());
        auto model = MakeModel();
        fastllm::ImportDeepSeekV4DSparkWeights(model.get(), plan);

        auto it = model->weight.weight.find("mtp.0.main_norm.weight");
        assert(it != model->weight.weight.end());
        auto &weight = it->second;
        assert(weight.dataType == DataType::BFLOAT16);
        assert(weight.dims == std::vector<int>({4}));
        assert(weight.cpuData != nullptr);
        const float expected[4] = {1.5f, -2.0f, 3.25f, 0.5f};
        const uint16_t *raw = reinterpret_cast<const uint16_t*>(weight.cpuData);
        bool mismatch = false;
        for (int i = 0; i < 4; i++) {
            if (Bf16ToFloat(raw[i]) != expected[i]) {
                mismatch = true;
            }
        }
        if (mismatch) {
            std::ifstream hdrIn(dir.path / "model.safetensors",
                                std::ios::binary);
            uint64_t headerLen = 0;
            hdrIn.read(reinterpret_cast<char*>(&headerLen), 8);
            std::vector<char> headerText(headerLen);
            hdrIn.read(headerText.data(), (std::streamsize)headerLen);
            std::cerr << "header(" << headerLen << "): "
                      << std::string(headerText.begin(), headerText.end())
                      << "\n";
            std::cerr << "imported bytes:";
            const uint8_t *byteView =
                reinterpret_cast<const uint8_t*>(weight.cpuData);
            for (uint64_t b = 0; b < weight.GetBytes(); b++) {
                char hex[8];
                std::snprintf(hex, sizeof(hex), " %02x", byteView[b]);
                std::cerr << hex;
            }
            std::cerr << " (GetBytes=" << weight.GetBytes()
                      << " Count0=" << weight.Count(0)
                      << " unitSize=" << weight.unitSize << ")\n";
            std::ifstream in(dir.path / "model.safetensors",
                             std::ios::binary);
            in.seekg((std::streamoff)offsets["mtp.0.main_norm.weight"]);
            uint8_t disk[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            in.read(reinterpret_cast<char*>(disk), 8);
            std::cerr << "disk bytes at offset "
                      << offsets["mtp.0.main_norm.weight"] << ":";
            for (int b = 0; b < 8; b++) {
                char hex[8];
                std::snprintf(hex, sizeof(hex), " %02x", disk[b]);
                std::cerr << hex;
            }
            std::cerr << "\n";
        }
        for (int i = 0; i < 4; i++) {
            assert(Bf16ToFloat(raw[i]) == expected[i]);
        }
        std::cout << "Sequential import keeps BF16 values byte-exact: passed.\n";
    }

    void TestRecognitionMatrix() {
        auto model = MakeModel();
        const std::vector<std::string> accepted = {
            "layers.0.ffn.experts.3.w1.weight",
            "embed.weight",
            "mtp.0.main_proj.weight",
            "mtp.0.main_norm.weight",
            "mtp.1.attn.wq_a.weight",
            "mtp.1.attn.q_norm.weight",
            "mtp.1.attn.kv_norm.weight",
            "mtp.1.attn.attn_sink",
            "mtp.1.hc_attn_fn",
            "mtp.1.hc_ffn_base",
            "mtp.1.attn_norm.weight",
            "mtp.1.ffn_norm.weight",
            "mtp.1.ffn.gate.weight",
            "mtp.1.ffn.experts.255.w3.weight",
            "mtp.1.ffn.shared_experts.w2.weight",
            "mtp.2.norm.weight",
            "mtp.2.hc_head_fn",
            "mtp.2.hc_head_scale",
            "mtp.2.hc_head_base",
            "mtp.2.markov_head.markov_w1.weight",
            "mtp.2.markov_head.markov_w2.weight",
            "mtp.2.confidence_head.proj.weight",
        };
        for (const auto &name : accepted) {
            assert(model->IsRecognizedWeightName(name));
        }
        const std::vector<std::string> rejected = {
            "mtp.3.attn.wkv.weight",
            "mtp.0.e_proj.weight",
            "mtp.0.h_proj.weight",
            "mtp.0.bogus.weight",
            "mtp.0.attn.wq_a.weight.extra",
            "mtp.0.ffn.experts.256.w1.weight",
            "mtp.0.ffn.experts.5.w4.weight",
            "mtp.0.ffn.experts.x.w1.weight",
            "mtp.x.attn.wkv.weight",
            "mtp.1.main_proj.weight",
            "mtp.1.main_norm.weight",
            "mtp.0.norm.weight",
            "mtp.0.hc_head_fn",
            "mtp.0.markov_head.markov_w1.weight",
            "mtp.1.confidence_head.proj.weight",
            "mtp.",
            "mtp.0",
        };
        for (const auto &name : rejected) {
            assert(!model->IsRecognizedWeightName(name));
        }
        std::cout << "Recognition matrix (accepted/rejected namespaces): passed.\n";
    }

    void TestImportRejectionPaths() {
        auto importSingle = [](const std::string &tensorName) {
            ScopedTempDir dir("reject");
            WriteSafetensorsShard(dir.path / "model.safetensors", {
                MakeBf16Tensor(tensorName, {2}, {1.0f, 1.0f}),
            });
            auto plan = fastllm::PlanDeepSeekV4DSparkShards(dir.path.string());
            auto model = MakeModel();
            fastllm::ImportDeepSeekV4DSparkWeights(model.get(), plan);
        };
        const std::vector<std::string> unrecognized = {
            "mtp.3.attn.wkv.weight",
            "mtp.0.e_proj.weight",
            "mtp.0.bogus.weight",
            "mtp.0.ffn.experts.256.w1.weight",
        };
        for (const auto &name : unrecognized) {
            ExpectFailure([&] { importSingle(name); }, "not recognized");
        }

        ScopedTempDir collision("collision");
        WriteSafetensorsShard(collision.path / "model.safetensors", {
            MakeBf16Tensor("mtp.2.norm.weight", {4},
                           {1.0f, 1.0f, 1.0f, 1.0f}),
        });
        auto plan = fastllm::PlanDeepSeekV4DSparkShards(collision.path.string());
        auto model = MakeModel();
        model->weight.AddEmptyWeight("mtp.2.norm.weight", {4},
                                     DataType::BFLOAT16);
        ExpectFailure([&] {
            fastllm::ImportDeepSeekV4DSparkWeights(model.get(), plan);
        }, "collides");
        std::cout << "Import rejects unrecognized names and collisions: passed.\n";
    }

    void TestFp8ScalePairing() {
        ScopedTempDir dir("fp8");
        // FP8 E4M3 encodings for 1, 2, 0.5, 4, -1, -2, 0.25, 3.
        WriteSafetensorsShard(dir.path / "model.safetensors", {
            MakeF8Tensor("mtp.0.attn.wq_a.weight", {2, 4},
                         {0x38, 0x40, 0x30, 0x48,
                          0xB8, 0xC0, 0x28, 0x44}),
            MakeF32Tensor("mtp.0.attn.wq_a.weight_scale", {1}, {2.0f}),
        });
        auto plan = fastllm::PlanDeepSeekV4DSparkShards(dir.path.string());
        assert(plan.tensorNames.size() == 2);
        auto model = MakeModel();
        fastllm::ImportDeepSeekV4DSparkWeights(model.get(), plan);

        auto it = model->weight.weight.find("mtp.0.attn.wq_a.weight");
        assert(it != model->weight.weight.end());
        auto &weight = it->second;
        assert(weight.dataType == DataType::FP8_E4M3);
        assert(weight.dims == std::vector<int>({2, 4}));
        assert(weight.blockK == 1 && weight.blockM == 4);
        assert(weight.scales.size() == 2);
        assert(weight.scales[0] == 2.0f && weight.scales[1] == 2.0f);
        const uint8_t expected[8] = {0x38, 0x40, 0x30, 0x48,
                                     0xB8, 0xC0, 0x28, 0x44};
        assert(std::memcmp(weight.cpuData, expected, 8) == 0);
        // The scale tensor itself must not become a model weight.
        assert(model->weight.weight.find("mtp.0.attn.wq_a.weight_scale") ==
               model->weight.weight.end());

        ScopedTempDir noScale("fp8NoScale");
        WriteSafetensorsShard(noScale.path / "model.safetensors", {
            MakeF8Tensor("mtp.0.attn.wkv.weight", {2, 2},
                         {0x38, 0x40, 0x30, 0x48}),
        });
        auto missingPlan =
            fastllm::PlanDeepSeekV4DSparkShards(noScale.path.string());
        auto missingModel = MakeModel();
        ExpectFailure([&] {
            fastllm::ImportDeepSeekV4DSparkWeights(missingModel.get(),
                                                   missingPlan);
        }, "missing its scale");
        std::cout << "FP8 scale pairing and missing-scale rejection: passed.\n";
    }

    void TestDiskLazyMetadata() {
        ScopedTempDir dir("disk");
        const std::string w1Name = "mtp.0.ffn.experts.7.w1.weight";
        const std::string w3Name = "mtp.0.ffn.experts.7.w3.weight";
        const std::string downName = "mtp.0.ffn.experts.7.w2.weight";
        const std::string gateupName = "mtp.0.ffn.experts.7.gateup.weight";
        auto offsets = WriteSafetensorsShard(dir.path / "model.safetensors", {
            MakeBf16Tensor(w1Name, {2, 2}, {2.0f, -1.0f, 0.5f, 4.0f}),
        });

        auto model = MakeModel();
        model->moeLinears.insert(w1Name);
        model->moeLinears.insert(w3Name);
        model->moeLinears.insert(downName);
        model->weightMergeRules.push_back(WeightMergeRule({
            WeightMergeRuleSingle({w1Name, w3Name}, gateupName,
                                  std::string("linearSwiglu"))}));
        model->AddSpecialWeight(gateupName, "linearSwiglu", 0);
        model->AddSpecialWeight(downName, "linearColumn", 0);
        model->moeDeviceMap["disk"] = 1;

        auto plan = fastllm::PlanDeepSeekV4DSparkShards(dir.path.string());
        fastllm::ImportDeepSeekV4DSparkWeights(model.get(), plan);

        auto it = model->weight.weight.find(w1Name);
        assert(it != model->weight.weight.end());
        auto &weight = it->second;
        assert(weight.isDiskWeight);
        assert(weight.cpuData == nullptr);
        assert(weight.dataType == DataType::BFLOAT16);
        assert(weight.dims == std::vector<int>({2, 2}));
        assert(weight.diskWeightParts.size() == 1);
        const auto &part = weight.diskWeightParts.front();
        assert(part.fileName == (dir.path / "model.safetensors").string());
        assert(part.fileOffset == (long long)offsets[w1Name]);
        assert(part.bytes == 8);
        assert(part.sourceDataType == DataType::BFLOAT16);
        assert(part.dims == std::vector<int>({2, 2}));
        std::cout << "Disk MoE metadata for DSpark experts: passed.\n";
    }
}

int main() {
    TestShardedPlanFiltersBackbone();
    TestPlanRejectsBackboneOnlyIndex();
    TestPlanRejectsMissingShard();
    TestSingleFilePlans();
    TestSequentialImportValues();
    TestRecognitionMatrix();
    TestImportRejectionPaths();
    TestFp8ScalePairing();
    TestDiskLazyMetadata();
    std::cout << "DeepSeek V4 DSpark hybrid loader tests passed.\n";
    return 0;
}
