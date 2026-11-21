#include "models/hy4_flash.h"
#include <cstdio>

using namespace fastllm;

int main() {
    SetDeviceMap({{"cuda:0", 1}});
    auto usesHostExperts = [](const char *front, const char *tail, int count) {
        SetMoeDeviceMap({{front, 1}});
        SetLayeredMoeDeviceMap({{tail, 1}});
        SetMoeDeviceLayers(count);
        Hy4FlashModel model;
        model.weight.dicts["num_hidden_layers"] = "42";
        model.InitParams();
        return model.UsesHostExperts();
    };
    if (usesHostExperts("cuda:0", "numa", 0) || !usesHostExperts("cuda:0", "numa", 1) ||
        !usesHostExperts("disk", "cuda:0", 1) || usesHostExperts("numa", "cuda:0", 41) ||
        !usesHostExperts("cpu", "cuda:0", -1))
        return 1;
    SetDeviceMap({{"cuda:0", 1}, {"cuda:1", 1}});
    if (!usesHostExperts("numa", "numa", 0) || usesHostExperts("cuda:0", "cuda:0", 0))
        return 1;
    std::puts("ALL_PASS");
}
