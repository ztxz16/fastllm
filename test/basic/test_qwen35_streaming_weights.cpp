#include "models/qwen3_5.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;

namespace {
void Check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class StreamingModel : public Qwen3_5Model {
public:
    StreamingModel(bool useDFlash, bool singleGpu) {
        block_cnt = 1;
        deviceMap = {{singleGpu ? "cuda:0" : "cuda:0,1", 1}};
        dflashEnabled = useDFlash;
        num_attention_heads = num_key_value_heads = 2;
        head_dim = 128;
        num_k_heads = 2;
        num_v_heads = 4;
        head_k_dim = head_v_dim = 128;
    }
};

void TestStreamingLmHead(bool useDFlash, bool singleGpu) {
    StreamingModel model(useDFlash, singleGpu);
    const std::string prefix = Qwen3_5Model::language_prefix + "layers.0.";
    std::set<std::string> names = {"lm_head.weight"};
    for (const std::string &suffix : {
             "self_attn.q_proj.weight", "self_attn.k_proj.weight",
             "self_attn.v_proj.weight", "self_attn.o_proj.weight",
             "mlp.gate_proj.weight", "mlp.up_proj.weight",
             "mlp.down_proj.weight"}) {
        names.insert(prefix + suffix);
    }
    model.OnWeightsCreated(names);

    model.weight.AddEmptyWeight("lm_head.weight", {256, 128}, FLOAT16);
    Data &head = model.weight["lm_head.weight"];
    head.Allocate();
    std::vector<uint8_t> expected(head.GetBytes());
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = (i * 17 + i / 256) % 251;
    }
    std::memcpy(head.cpuData, expected.data(), expected.size());

    // The draft remains on its existing preparation path. It must neither be
    // confused with a target decoder layer nor be consumed by its callback.
    const std::string draft = "dflash.layers.0.mlp.down_proj.weight";
    model.weight.AddEmptyWeight(draft, {2, 2}, BFLOAT16);
    Data &draftWeight = model.weight[draft];
    draftWeight.Allocate();
    std::memset(draftWeight.cpuData, 0x3c, draftWeight.GetBytes());
    uint8_t *draftStorage = draftWeight.cpuData;
    const std::vector<std::pair<std::string, DataType>> draftMapping = {
        {draft, BFLOAT16}};
    Check(!model.ShouldLoadWeightSeriallyBeforeOthers(
              "layers.0.mlp.down_proj.weight", draftMapping),
          "draft weight entered a target streaming group");

    model.OnWeightLoadGroupStarted({"lm_head.weight"});
    model.OnWeightLoadGroupFinished();
    Check(head.cpuData == nullptr &&
              (singleGpu ? !head.multiDeviceData : head.multiDeviceDatas.size() == 2),
          "lm_head CPU storage survived its load group; target was not streamed");
    size_t sourceOffset = 0;
    for (int device = 0; device < (singleGpu ? 1 : 2); device++) {
        Data &local = singleGpu ? head : *head.multiDeviceDatas.at(device);
        Check(local.dataDevice == DataDevice::CUDA &&
                  local.dims == std::vector<int>({singleGpu ? 256 : 128, 128}) &&
                  local.dataType == FLOAT16,
              "streamed lm_head shard metadata changed");
        std::vector<uint8_t> actual(local.GetBytes());
        FastllmCudaSetDevice(device);
        FastllmCudaCopyFromDeviceToHost(
            actual.data(), local.cudaData, actual.size());
        Check(std::memcmp(actual.data(), expected.data() + sourceOffset,
                          actual.size()) == 0,
              "streaming changed lm_head shard contents");
        sourceOffset += actual.size();
    }
    Check(sourceOffset == expected.size(), "lm_head split lost rows");
    Check(draftWeight.cpuData == draftStorage &&
              !draftWeight.multiDeviceData && draftStorage[0] == 0x3c,
          "target streaming consumed or changed draft storage");
}

void TestStreamingGgufLayer(bool linearAttention, bool merged, bool singleGpu) {
    StreamingModel model(true, singleGpu);
    const std::string prefix = Qwen3_5Model::language_prefix + "layers.0.";
    std::set<std::string> names = {"lm_head.weight"};
    std::vector<std::string> loaded;
    auto add = [&](const std::string &suffix, int rows, int columns,
                   ggml_type type) -> Data & {
        const std::string name = prefix + suffix;
        Data &data = model.weight.weight[name];
        if (type == GGML_TYPE_F32) {
            data = Data(FLOAT32, columns > 0 ? std::vector<int>{rows, columns}
                                           : std::vector<int>{rows});
        } else {
            data = Data(DATA_GGUF_FORMAT, (int)type, {rows, columns});
        }
        data.name = name;
        data.isModelWeight = data.isGGUFData = true;
        data.forceGGUFFp32Dequant = true;
        data.Allocate();
        std::memset(data.cpuData, 0, data.GetBytes());
        names.insert(name);
        loaded.push_back(name);
        return data;
    };
    if (merged) {
        add("mlp.gateup_proj.weight", 1024, 256, GGML_TYPE_Q4_0);
    } else {
        add("mlp.gate_proj.weight", 512, 256, GGML_TYPE_Q4_0);
        add("mlp.up_proj.weight", 512, 256, GGML_TYPE_Q8_0);
    }
    add("mlp.down_proj.weight", 256, 512, GGML_TYPE_Q4_0);
    if (linearAttention) {
        // OnWeightsCreated sees source names; the group callback sees the
        // loader's final, possibly merged, quantized tensors.
        for (const auto &part : {"qkv", "z", "b", "a"}) {
            names.insert(prefix + "linear_attn.in_proj_" + part + ".weight");
        }
        if (merged) {
            add("linear_attn.in_proj_qkvz.weight", 1536, 256, GGML_TYPE_Q4_0);
        } else {
            add("linear_attn.in_proj_qkv.weight", 1024, 256, GGML_TYPE_Q4_0);
            add("linear_attn.in_proj_z.weight", 512, 256, GGML_TYPE_Q8_0);
        }
        add("linear_attn.in_proj_ba.weight", 8, 256, GGML_TYPE_F32);
        add("linear_attn.conv1d.weight", 1024, 4, GGML_TYPE_F32);
        Data &a = add("linear_attn.A_log", 4, 0, GGML_TYPE_F32);
        Data &dt = add("linear_attn.dt_bias", 4, 0, GGML_TYPE_F32);
        for (int i = 0; i < 4; i++) {
            ((float *)a.cpuData)[i] = -std::exp(float(i + 1));
            ((float *)dt.cpuData)[i] = float(i);
        }
        add("linear_attn.out_proj.weight", 256, 512, GGML_TYPE_Q4_0);
    } else {
        for (const auto &part : {"q", "k", "v"}) {
            names.insert(prefix + "self_attn." + part + "_proj.weight");
        }
        if (merged) {
            add("self_attn.mergeqkv.weight", 1024, 256, GGML_TYPE_Q4_0);
        } else {
            add("self_attn.q_proj.weight", 512, 256, GGML_TYPE_Q4_0);
            add("self_attn.k_proj.weight", 256, 256, GGML_TYPE_Q8_0);
            add("self_attn.v_proj.weight", 256, 256, GGML_TYPE_Q4_0);
        }
        add("self_attn.o_proj.weight", 256, 256, GGML_TYPE_Q4_0);
    }
    model.OnWeightsCreated(names);
    model.OnWeightLoadGroupStarted({loaded.begin(), loaded.end()});
    model.OnWeightLoadGroupFinished();
    for (const auto &name : loaded) {
        Data &data = model.weight[name];
        if (singleGpu) {
            Check(!data.multiDeviceData &&
                      (data.dims.size() < 2 ? data.cpuData != nullptr :
                       (data.cpuData == nullptr && data.dataDevice == DataDevice::CUDA)),
                  "GGUF single-GPU layer was not streamed correctly");
            Check(data.isGGUFData && data.forceGGUFFp32Dequant,
                  "GGUF single-GPU weight lost dequantization metadata");
        } else {
            Check(data.cpuData == nullptr && data.multiDeviceDatas.size() == 2,
                  "GGUF layer retained CPU weights after its load group");
        }
        if (data.dataType == DATA_GGUF_FORMAT) {
            for (const auto &shard : data.multiDeviceDatas) {
                Check(shard.second->isGGUFData && shard.second->forceGGUFFp32Dequant,
                      "GGUF TP shard lost dequantization metadata");
            }
        }
    }
    // The final pass must not reorder or log-transform already uploaded GDN
    // weights again. Both operations previously ran only after the full load.
    model.OnModelWeightsLoaded();
    model.OnModelWeightsLoaded();
    if (linearAttention) {
        if (singleGpu) {
            const int expected[] = {0, 2, 1, 3};
            for (const auto &suffix : {"A_log", "dt_bias"}) {
                const Data &data = model.weight[prefix + "linear_attn." + suffix];
                const float offset = std::string(suffix) == "A_log" ? 1.0f : 0.0f;
                for (int i = 0; i < 4; i++) {
                    Check(std::fabs(((float *)data.cpuData)[i] - (expected[i] + offset)) < 1e-6f,
                          "single-GPU GGUF GDN layout or decay conversion changed");
                }
            }
            return;
        }
        for (int device : {0, 1}) {
            FastllmCudaSetDevice(device);
            for (const auto &suffix : {"A_log", "dt_bias"}) {
                Data &local = *model.weight[prefix + "linear_attn." + suffix]
                    .multiDeviceDatas.at(device);
                Check(local.dataType == FLOAT32 && local.Count(0) == 2,
                      "GGUF GDN vector shard has an unexpected shape");
                float actual[2];
                FastllmCudaCopyFromDeviceToHost(actual, local.cudaData, sizeof(actual));
                const float offset = std::string(suffix) == "A_log" ? 1.0f : 0.0f;
                Check(std::fabs(actual[0] - (device + offset)) < 1e-6f &&
                          std::fabs(actual[1] - (device + 2 + offset)) < 1e-6f,
                      "GGUF GDN tiled head order or decay conversion changed");
            }
        }
    }
}

void TestSingleGpuGdnMergeAndTiedEmbedding() {
    StreamingModel model(false, true);
    const std::string prefix = Qwen3_5Model::language_prefix + "layers.0.";
    const std::string embedding = Qwen3_5Model::language_prefix + "embed_tokens.weight";
    std::set<std::string> names = {embedding};
    for (const auto &suffix : {
             "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
             "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
             "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias",
             "linear_attn.out_proj.weight", "mlp.gateup_proj.weight", "mlp.down_proj.weight"}) {
        names.insert(prefix + suffix);
    }
    model.OnWeightsCreated(names);
    Check(model.ShouldLoadWeightSeriallyBeforeOthers(prefix + "mlp.down_proj.weight", {}),
          "tied-embedding single-GPU decoder did not enable streaming");
    const std::string qkvz = prefix + "linear_attn.in_proj_qkvz.weight";
    const std::string ba = prefix + "linear_attn.in_proj_ba.weight";
    std::vector<uint8_t> expected;
    for (const auto &name : {qkvz, ba}) {
        model.weight.AddEmptyWeight(name, {name == qkvz ? 1536 : 8, 256}, FLOAT16);
        Data &data = model.weight[name];
        data.Allocate();
        for (size_t i = 0; i < data.GetBytes(); i++) {
            data.cpuData[i] = (i * 17 + expected.size()) % 251;
        }
        expected.insert(expected.end(), data.cpuData, data.cpuData + data.GetBytes());
    }
    const std::string norm = prefix + "input_layernorm.weight";
    for (const auto &name : {embedding, norm, std::string("dflash.layers.0.sentinel"),
                             std::string("mtp.layers.0.sentinel")}) {
        model.weight.AddEmptyWeight(name, name == norm ? std::vector<int>{256} :
                                                       std::vector<int>{4, 256}, FLOAT32);
        model.weight[name].Allocate();
        std::memset(model.weight[name].cpuData, 0, model.weight[name].GetBytes());
    }
    model.OnWeightLoadGroupStarted({qkvz, ba, norm});
    model.OnWeightLoadGroupFinished();
    Check(model.weight.weight.count(qkvz) == 0 && model.weight.weight.count(ba) == 0,
          "single-GPU merged GDN retained its CPU source tensors");
    Data &merged = model.weight[prefix + "linear_attn.in_proj_qkvzba.weight"];
    Check(merged.cpuData == nullptr && merged.dataDevice == DataDevice::CUDA &&
              !merged.multiDeviceData && merged.GetBytes() == expected.size(),
          "single-GPU merged GDN storage is incorrect");
    std::vector<uint8_t> actual(expected.size());
    FastllmCudaSetDevice(0);
    FastllmCudaCopyFromDeviceToHost(actual.data(), merged.cudaData, actual.size());
    Check(actual == expected, "single-GPU GDN merge changed source bytes");
    for (const auto &name : {embedding, norm, std::string("dflash.layers.0.sentinel"),
                             std::string("mtp.layers.0.sentinel")}) {
        Check(model.weight[name].cpuData != nullptr && model.weight[name].cudaData == nullptr,
              "streaming moved an embedding, CPU norm offset, or draft tensor");
    }
    model.OnModelWeightsLoaded();
}
}

int main(int argc, char **argv) {
    const bool singleGpu = argc == 2 && std::string(argv[1]) == "--single";
    if (FastllmCudaGetDeviceCount() < (singleGpu ? 1 : 2)) {
        std::cout << "SKIP: insufficient CUDA devices\n";
        return 77;
    }
    try {
        TestStreamingLmHead(false, singleGpu);
        TestStreamingLmHead(true, singleGpu);
        for (bool linearAttention : {false, true}) {
            for (bool merged : {false, true}) {
                TestStreamingGgufLayer(linearAttention, merged, singleGpu);
            }
        }
        if (singleGpu) {
            TestSingleGpuGdnMergeAndTiedEmbedding();
        }
        std::cout << "PASS: DFlash target streaming and GGUF layer layouts\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
