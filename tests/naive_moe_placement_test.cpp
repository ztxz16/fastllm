#undef NDEBUG
#include "models/naive_n05_flash.h"
#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

using namespace fastllm;

// Admission only: exercise real model initialization and per-layer placement
// without loading an exp18 checkpoint or allocating expert weights.
static void Check(const char *name, const char *tp, const char *ep,
                  const char *moe, int offloadLayers, const char *offload,
                  bool packed, bool epWeights, bool fused = true) {
    setenv("FASTLLM_TP", tp, 1);
    setenv("FASTLLM_EP", ep, 1);
    setenv("FASTLLM_NAIVE_FLEX_SHARED", "1", 1);
    setenv("FASTLLM_NAIVE_FLEX_MOE", "1", 1);
    setenv("FASTLLM_NAIVE_MOE_FUSION", fused ? "1" : "0", 1);
    SetDeviceMap({{"cuda:0", 1}});
    SetMoeDeviceMap({{moe, 1}});
    SetLayeredMoeDeviceMap({{offload, 1}});
    SetMoeDeviceLayers(offloadLayers);
    NaiveN05FlashModel model;
    model.weight.dicts = {
        {"hidden_size", "4096"}, {"num_hidden_layers", "2"},
        {"hybrid_layer_pattern", "[0,1]"}, {"moe_layer_freq", "[1,1]"},
        {"scoring_func", "sigmoid"}
    };
    model.InitParams();
    assert(model.UsesPackedMoeSource() == packed);
    assert(model.canDoBatchForward == (std::string(tp) == "8"));
    for (int layer = 0; layer < 2; ++layer) {
        const std::string expected = offloadLayers > 0 && layer >= 2 - offloadLayers ? offload : moe;
        assert(model.SelectMoeDeviceForLayer(layer) == expected);
        const std::string key = "model.layers." + std::to_string(layer) + ".mlp.experts.0.gateup_proj.weight";
        assert(model.specialWeights.at(key) == (epWeights ? "mimoEp8Expert" : "linearSwiglu"));
    }
    std::printf("PLACEMENT_PASS %s\n", name);
}

int main(int argc, char **argv) {
    // The optional resident positive cases require the eight-SM103 test machine.
    const bool sm103 = argc == 2 && std::string(argv[1]) == "--sm103-tp8";
    if (sm103) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8) return 77;
        for (int device = 0; device < 8; ++device) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, device) != cudaSuccess ||
                prop.major != 10 || prop.minor != 3) return 77;
            for (int peer = 0; peer < 8; ++peer) {
                if (peer == device) continue;
                int access = 0;
                if (cudaDeviceCanAccessPeer(&access, device, peer) != cudaSuccess || !access) return 77;
            }
        }
    }
    SetMoeCudaCacheBytes(0);
    const char *group = "cuda:0,1,2,3,4,5,6,7";
    for (const char *host : {"cpu", "numa", "numa:0,1", "disk"}) {
        Check(host, "off", "1", host, -1, "cpu", false, false);
        Check("layered-host", "off", "1", "cuda:0", 1, host, false, false);
        if (sm103) {
            Check("TP-host-rejects-native", "8", "8", host, -1, "cpu", false, false);
            Check("TP-layered-rejects-native", "8", "8", group, 1, host, false, false);
        }
    }
    Check("serial-cuda", "off", "1", "cuda:0", -1, "cpu", false, false);
    if (sm103) {
        Check("resident-EP1", "8", "1", group, -1, "cpu", true, false);
        Check("resident-EP8", "8", "8", group, -1, "cpu", true, true);
        Check("disabled", "8", "8", group, -1, "cpu", false, false, false);
        Check("unused-offload-map", "8", "8", group, 0, "cpu", true, true);
        SetMoeCudaCacheBytes(64ULL * 1024 * 1024);
        Check("cache-rejects-native", "8", "8", group, -1, "cpu", false, false);
        SetMoeCudaCacheBytes(0);
        Check("cache-disabled", "8", "8", group, -1, "cpu", true, true);
    }
}
