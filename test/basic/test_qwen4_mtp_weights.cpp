#include "models/qwen4_exp.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void TestSplit(DataType type, bool downFirst) {
    Qwen4ExpModel model;
    model.num_experts = 3;
    model.deviceMap = {{"cpu", 1}};
    model.moeDeviceMap = {{"cpu", 1}};
    const std::string prefix = "mtp.layers.0.mlp.experts.";
    const std::string gateName = prefix + "gate_up_proj";
    const std::string downName = prefix + "down_proj";
    std::vector<std::vector<uint8_t>> expected;
    for (bool gate : {true, false}) {
        Data &data = model.weight.weight[gate ? gateName : downName];
        data = Data(type, gate ? std::vector<int>{3, 8, 5} :
                                std::vector<int>{3, 5, 4});
        data.Allocate();
        std::vector<uint8_t> bytes(data.GetBytes());
        for (size_t i = 0; i < bytes.size(); i++) {
            bytes[i] = (i * 17 + (gate ? 3 : 91)) % 251;
        }
        std::memcpy(data.cpuData, bytes.data(), bytes.size());
        expected.push_back(std::move(bytes));
    }
    const std::string first = downFirst ? downName : gateName;
    const std::string second = downFirst ? gateName : downName;
    model.OnWeightLoaded(first, {first});
    Check(model.weight.weight.size() == 2, "split before both weights loaded");
    model.OnWeightLoaded("unrelated.weight", {gateName, downName});
    Check(model.weight.weight.size() == 2, "unrelated callback split weights");
    model.OnWeightLoaded(second, {gateName, downName});
    Check(model.weight.weight.size() == 6, "packed source buffers not removed");
    for (int expert = 0; expert < 3; expert++) {
        for (int projection = 0; projection < 2; projection++) {
            const std::string name = prefix + std::to_string(expert) +
                (projection == 0 ? ".gateup_proj.weight" : ".down_proj.weight");
            const Data &data = model.weight.weight.at(name);
            Check(data.dataType == type, "split changed precision");
            Check(data.dims == (projection == 0 ? std::vector<int>{8, 5} :
                                                std::vector<int>{5, 4}),
                  "split changed matrix orientation");
            Check(data.isModelWeight && !data.isFake &&
                  data.weightType == WeightType::LINEAR && data.name == name,
                  "split weight metadata invalid");
            Check(std::memcmp(data.cpuData,
                              expected[projection].data() + expert * data.GetBytes(),
                              data.GetBytes()) == 0,
                  "expert slice contents differ from checkpoint");
        }
    }
    model.OnWeightLoaded(second, {gateName, downName});
    Check(model.weight.weight.size() == 6, "duplicate callback changed weights");
}

int main() {
    try {
        setenv("FASTLLM_QWEN4_ENABLE_MTP", "3", 1);
        for (DataType type : {DataType::FLOAT16, DataType::BFLOAT16, DataType::FLOAT32}) {
            TestSplit(type, false);
            TestSplit(type, true);
        }
        Qwen4ExpModel model;
        const std::string gate = "mtp.layers.0.mlp.experts.gate_up_proj";
        const std::string down = "mtp.layers.0.mlp.experts.down_proj";
        const std::string marker = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale";
        auto mapped = model.GetTensorMap({gate, down, marker});
        Check(mapped.at(gate).at(0) == std::make_pair(gate, DataType::BFLOAT16) &&
              mapped.at(down).at(0) == std::make_pair(down, DataType::BFLOAT16),
              "NVFP4 target changed BF16 MTP source layout");
        setenv("FASTLLM_QWEN4_ENABLE_MTP", "0", 1);
        mapped = model.GetTensorMap({gate, down, marker});
        Check(mapped.count(gate) == 0 && mapped.count(down) == 0,
              "disabled MTP still loads packed experts");
        std::cout << "PASS: 8 Qwen4 MTP packed-weight cases\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
