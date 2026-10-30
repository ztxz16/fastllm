#include "model.h"
#include "models/deepseekv41.h"
#include <iostream>
#include <stdexcept>

using namespace fastllm;

// Exercise the virtual hook without allocating GPU weights. The external draft
// override must preserve the target model's shared/dense placement policy.
static void CheckDeviceSelection() {
    DeepSeekV41Model model;
    model.block_cnt = 4;
    model.deviceMap = {{"cuda:0", 1}, {"cuda:1", 1}};
    model.moeDeviceMap = {{"disk", 1}};
    model.moeDeviceLayers = -1;
    const basellm &base = model;
    auto expect = [&](const std::string &name, int layer, const std::string &device) {
        const std::string actual = base.SelectSpecialWeightDevice(name, layer);
        if (actual != device)
            throw std::runtime_error(name + ": expected " + device + ", got " + actual);
    };
    SetCudaSharedExpert(true);
    for (int layer : {0, 3}) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        const std::string device = layer == 0 ? "cuda:0" : "cuda:1";
        for (const auto &suffix : {"gateup.weight", "w2.weight"}) {
            expect(prefix + "ffn.shared_experts." + suffix, layer, device);
            expect(prefix + "ffn.experts.0." + suffix, layer, "disk");
        }
        expect(prefix + "attn.wq_b.weight", layer, device);
    }
    expect("head.weight", -1, "cuda:1");

    const std::string draft = "mtp.0.ffn.experts.0.gateup.weight";
    expect(draft, 4, "disk");  // Embedded drafts still follow the MoE map.
    model.weight.dicts["dspark_external_gguf"] = "1";
    const std::string host = GetFastllmEnv().activateNuma ? "numa" : "cpu";
    for (const auto &disk : {"disk", "disk:0"}) {
        model.moeDeviceMap = {{disk, 1}};
        expect(draft, 4, host);
        expect("layers.0.ffn.experts.0.gateup.weight", 0, disk);
        expect("layers.0.ffn.shared_experts.gateup.weight", 0, "cuda:0");
        expect("mtp.0.ffn.shared_experts.gateup.weight", 4, "cuda:1");
        SetCudaSharedExpert(false);
        expect("layers.0.ffn.shared_experts.gateup.weight", 0, disk);
        expect("mtp.0.ffn.shared_experts.gateup.weight", 4, host);
        SetCudaSharedExpert(true);
    }
    model.moeDeviceMap = {{"cpu", 1}};
    expect(draft, 4, "cpu");  // External drafts only override disk placement.
    std::cout << "PASS V4.1 target and external draft device selection\n";
}

// Pass the unpacked deepseek_v41_fixture.npz checkpoint. Loading is sufficient:
// neither a CUDA device nor a full forward pass is required by this regression.
static void CheckPlacement(const std::string &path, bool cudaShared,
                           const std::map<std::string, int> &mainDevices,
                           bool layered) {
    SetCudaSharedExpert(cudaShared);
    SetDeviceMap(mainDevices);
    SetMoeDeviceMap(layered ? std::map<std::string, int>{{"cpu", 1}} :
                             std::map<std::string, int>{{"disk", 1}});
    SetLayeredMoeDeviceMap({{"disk", 1}});
    SetMoeDeviceLayers(layered ? 2 : -1);
    auto model = CreateLLMModelFromHF(path, FLOAT16, -1, true);
    int checked = 0;
    for (int layer = 0; layer < model->block_cnt; ++layer) {
        const bool diskLayer = !layered || layer >= model->block_cnt - 2;
        const std::string prefix = "layers." + std::to_string(layer) + ".ffn.";
        for (const auto &suffix : {"gateup.weight", "w2.weight"}) {
            const Data &shared = model->weight.weight.at(prefix + "shared_experts." + suffix);
            const Data &routed = model->weight.weight.at(prefix + "experts.0." + suffix);
            if (shared.isDiskWeight != (diskLayer && !cudaShared))
                throw std::runtime_error("shared expert ignored its configured placement");
            if (routed.isDiskWeight != diskLayer)
                throw std::runtime_error("shared placement changed routed expert residency");
            if (!shared.isDiskWeight && !shared.cpuData && !shared.cudaData)
                throw std::runtime_error("resident shared expert has no weight payload");
            ++checked;
        }
    }
    if (!checked) throw std::runtime_error("fixture contained no decoder layers");
    std::cout << "PASS shared=" << cudaShared << " layered=" << layered
              << " projections=" << checked << '\n';
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <unpacked-v41-fixture>\n";
        return 2;
    }
    try {
        SetThreads(4);
        CheckDeviceSelection();
        CheckPlacement(argv[1], true, {{"cuda:0", 1}}, false);
        CheckPlacement(argv[1], false, {{"cuda:0", 1}}, false);
        CheckPlacement(argv[1], true, {{"cuda:0", 1}, {"cuda:1", 1}}, false);
        CheckPlacement(argv[1], true, {{"cpu", 1}}, true);
        CheckPlacement(argv[1], false, {{"cpu", 1}}, true);
        std::cout << "PASS disk shared placement\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
