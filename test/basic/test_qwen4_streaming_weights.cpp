#include "models/qwen4_exp.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void CheckWeight(Qwen4ExpModel &model, const std::string &name,
                        int expectedDevice, DataType type = BFLOAT16) {
    model.weight.AddEmptyWeight(name, {32, 64}, type);
    Data &data = model.weight[name];
    data.isModelWeight = true;
    data.Allocate();
    std::vector<uint8_t> expected(data.GetBytes());
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = (i * 17 + 3) % 251;
    std::memcpy(data.cpuData, expected.data(), expected.size());
    auto *host = data.cpuData;
    const int previousDevice = FastllmCudaGetDevice();
    model.OnWeightLoaded(name, {name});
    model.OnWeightLoaded(name, {name}); // Duplicate notifications are harmless.
    Check(FastllmCudaGetDevice() == previousDevice, "callback changed CUDA device");
    Check(data.dataType == type && data.dims == std::vector<int>({32, 64}),
          "streaming changed weight metadata");
    if (expectedDevice < 0) {
        Check(data.dataDevice == DataDevice::CPU && data.cpuData == host &&
              std::memcmp(host, expected.data(), expected.size()) == 0,
              "weight needed on CPU was moved or changed");
    } else {
        Check(data.dataDevice == DataDevice::CUDA && data.cpuData == nullptr &&
              data.dataDeviceIds == std::vector<int>{expectedDevice},
              "weight was not streamed to its configured CUDA device");
        std::vector<uint8_t> actual(data.GetBytes());
        FastllmCudaSetDevice(expectedDevice);
        FastllmCudaCopyFromDeviceToHost(actual.data(), data.cudaData, actual.size());
        FastllmCudaSetDevice(previousDevice);
        Check(actual == expected, "streaming changed weight contents");
    }
}

static void TestSharedExpertFastPath(bool explicitDevice) {
    Qwen4ExpModel model;
    model.deviceMap = {{explicitDevice ? "cuda:0" : "cuda", 1}};
    model.moeDeviceMap = {{"numa", 1}};
    const std::string prefix = "model.language_model.layers.0.mlp.";
    const std::vector<std::string> names = {
        prefix + "shared_expert.gateup_proj.weight",
        prefix + "shared_expert.down_proj.weight",
        prefix + "shared_expert_gate.weight"};
    const std::vector<std::vector<int>> shapes = {{1280, 2560}, {2560, 640}, {1, 2560}};
    for (size_t i = 0; i < names.size(); ++i) {
        model.weight.AddEmptyWeight(names[i], shapes[i], FLOAT16);
        Data &data = model.weight[names[i]];
        data.Allocate();
        std::memset(data.cpuData, 0, data.GetBytes());
        model.OnWeightLoaded(names[i], {names[i]});
        Check(data.directMemory == (FastllmCudaGetWeightSlabBytes() == 0),
              "streaming did not respect the model weight slab policy");
    }
    Data input(FLOAT32, {1, 2560});
    input.Allocate();
    std::memset(input.cpuData, 0, input.GetBytes());
    input.ToDevice(DataDevice::CUDA, explicitDevice ? std::vector<int>{0} : std::vector<int>{});
    Data gateUp, hidden, gate, output;
    Check(FastllmCudaQwen4SharedExpert(input,
              model.weight[names[0]], model.weight[names[1]], model.weight[names[2]],
              gateUp, hidden, gate, output),
          "streaming disabled the fused shared expert fast path");
    output.ToDevice(DataDevice::CPU);
    const float *values = reinterpret_cast<const float *>(output.cpuData);
    for (int i = 0; i < 2560; ++i) Check(values[i] == 0, "fused shared expert output changed");
}

static void TestGpuExpertLoadPeak() {
    const std::string prefix = "model.language_model.layers.0.";
    // Implicit placement, bare CUDA and split placement all need the same
    // headroom. The first layer must also wait when only a later layer has GPU
    // experts: all expert sources are loaded before warmup starts.
    for (const auto &map : std::vector<std::map<std::string, int>>{
             {}, {{"cuda", 1}}, {{"cuda:0", 1}}, {{"multicuda", 1}},
             {{"cuda:0,1", 1}}, {{"multicuda:0,1", 1}},
             {{"cpu", 1}, {"cuda:0", 1}}}) {
        Qwen4ExpModel model;
        model.block_cnt = 2;
        model.deviceMap = {{"cuda:0", 1}};
        model.moeDeviceMap = map;
        for (const auto &name : {prefix + "self_attn.q_proj.weight",
                                prefix + "mlp.shared_expert.gateup_proj.weight",
                                std::string("lm_head.weight"),
                                std::string("mtp.fc_hidden.weight")}) {
            CheckWeight(model, name, -1);
        }
    }
    for (int layeredLayers : {0, 1, 2}) {
        Qwen4ExpModel model;
        model.block_cnt = 2;
        model.deviceMap = {{"cuda:0", 1}};
        model.moeDeviceMap = {{"numa", 1}};
        model.layeredMoeDeviceMap = {{"cuda:0", 1}};
        model.moeDeviceLayers = layeredLayers;
        CheckWeight(model, prefix + "self_attn.q_proj.weight", layeredLayers ? -1 : 0);
        CheckWeight(model, "lm_head.weight", layeredLayers ? -1 : 0);
    }
    // An unused GPU entry must not disable streaming when all layers are
    // explicitly overridden to NUMA.
    {
        Qwen4ExpModel model;
        model.block_cnt = 2;
        model.deviceMap = model.moeDeviceMap = {{"cuda:0", 1}};
        model.layeredMoeDeviceMap = {{"numa", 1}};
        model.moeDeviceLayers = 2;
        CheckWeight(model, "lm_head.weight", 0);
    }
    for (const auto &device : {"cpu", "numa", "disk"}) {
        Qwen4ExpModel model;
        model.deviceMap = {{"cuda:0", 1}};
        model.moeDeviceMap = {{device, 1}};
        CheckWeight(model, "lm_head.weight", 0);
    }
}

int main() {
    try {
        if (FastllmCudaGetDeviceCount() == 0) {
            std::cout << "SKIP: CUDA device required\n";
            return 77;
        }
        unsetenv("FASTLLM_TP");
        FastllmCudaSetDevice(0);
        FastllmCudaSetWeightSlabBytes(0);
        TestGpuExpertLoadPeak();
        for (size_t slabBytes : {size_t(0), size_t(64) * 1024 * 1024}) {
            FastllmCudaSetWeightSlabBytes(slabBytes);
            TestSharedExpertFastPath(false);
            TestSharedExpertFastPath(true);
        }
        FastllmCudaSetWeightSlabBytes(0);
        const std::string prefix = "model.language_model.layers.0.";
        {
            Qwen4ExpModel model;
            model.block_cnt = 2;
            model.deviceMap = {{"cuda", 1}};
            model.moeDeviceMap = {{"numa", 1}};
            for (DataType type : {FLOAT32, FLOAT16, BFLOAT16, FP8_E4M3}) {
                CheckWeight(model, prefix + "self_attn.q_proj.weight", 0, type);
            }
            for (const std::string &name : {
                     "lm_head.weight", "mtp.fc_hidden.weight",
                     "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                     "model.language_model.layers.0.ple.key_proj.weight",
                     "model.language_model.layers.0.mlp.gate.weight"}) {
                CheckWeight(model, name, 0);
            }
            const std::string gate = prefix + "mlp.shared_expert.gate_proj.weight";
            const std::string up = prefix + "mlp.shared_expert.up_proj.weight";
            const std::string merged = prefix + "mlp.shared_expert.gateup_proj.weight";
            model.weightMergeRules.emplace_back(std::vector<WeightMergeRuleSingle>{
                WeightMergeRuleSingle({gate, up}, merged, "linearSwiglu")});
            CheckWeight(model, gate, -1);
            CheckWeight(model, up, -1);
            CheckWeight(model, merged, 0);
            for (const std::string &name : {
                     "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
                     "model.language_model.layers.0.self_attn.indexer.k_layernorm.weight",
                     "model.language_model.embed_tokens.weight",
                     "model.language_model.layers.0.ple.ngram_embedding.weight",
                     "model.visual.blocks.0.attn.qkv.weight"}) {
                CheckWeight(model, name, -1);
            }
        }
        for (const auto &map : std::vector<std::map<std::string, int>>{
                 {}, {{"cpu", 1}}, {{"numa", 1}}, {{"cuda:0,1", 1}}, {{"multicuda:0,1", 1}}}) {
            Qwen4ExpModel model;
            model.deviceMap = map;
            CheckWeight(model, "lm_head.weight", -1);
        }
        {
            Qwen4ExpModel model;
            model.block_cnt = 2;
            model.deviceMap = {{"cpu", 1}, {"cuda:0", 1}};
            model.moeDeviceMap = {{"numa", 1}};
            CheckWeight(model, prefix + "self_attn.q_proj.weight", -1);
            CheckWeight(model, "model.language_model.layers.1.self_attn.q_proj.weight", 0);
            CheckWeight(model, "lm_head.weight", 0);
            CheckWeight(model, "mtp.fc_hidden.weight", 0);
        }
        if (FastllmCudaGetDeviceCount() > 1) {
            Qwen4ExpModel model;
            model.block_cnt = 2;
            model.deviceMap = {{"cuda:0", 1}, {"cuda:1", 1}};
            model.moeDeviceMap = {{"numa", 1}};
            CheckWeight(model, prefix + "self_attn.q_proj.weight", 0);
            CheckWeight(model, "model.language_model.layers.1.self_attn.q_proj.weight", 1);
            CheckWeight(model, "lm_head.weight", 1);
            setenv("FASTLLM_TP", "0,1", 1);
            Qwen4ExpModel tp;
            tp.deviceMap = tp.moeDeviceMap = {{"cuda", 1}};
            tp.weight.dicts = {{"num_hidden_layers", "2"}, {"num_experts", "2"},
                               {"max_position_embeddings", "16"}, {"ngram_vocab_size_base", "17"}};
            tp.InitParams();
            Check(tp.ShouldDelaySpecialWeightCudaMove("lm_head.weight"), "TP test did not initialize");
            CheckWeight(tp, "lm_head.weight", -1);
            CheckWeight(tp, prefix + "self_attn.q_proj.weight", -1);
            unsetenv("FASTLLM_TP");
        }
        std::cout << "PASS: Qwen4 dense weight streaming, contents and placement guards\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
