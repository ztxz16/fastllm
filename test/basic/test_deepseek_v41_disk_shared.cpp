#include "model.h"
#include <iostream>
#include <stdexcept>

using namespace fastllm;

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
