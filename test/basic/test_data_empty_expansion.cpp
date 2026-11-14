#include "fastllm.h"
#include "utils.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif
#include <cstdio>
#include <stdexcept>
using namespace fastllm;

static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void Values(const Data &value, const std::vector<int> &shape) {
    Check(value.dims == shape, "expansion changed logical dimensions");
    Data host; host.CopyFrom(value); host.ToDevice(CPU);
    uint64_t count = 1;
    for (int dim : shape) count *= dim;
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t rest = i, offset = 0;
        for (int axis = int(shape.size()) - 1; axis >= 0; --axis) {
            offset += (rest % shape[axis]) * host.strides[axis]; rest /= shape[axis];
        }
        Check(reinterpret_cast<const float *>(host.cpuData)[offset] == .75f,
              "expansion corrupted logical values");
    }
}
int main() {
    try {
        std::vector<std::string> devices{"cpu"};
#ifdef USE_CUDA
        if (FastllmCudaGetDeviceCount()) { FastllmCudaSetDevice(0); devices.push_back("cuda:0"); }
#endif
        SetThreads(2);
        for (const auto &device : devices) for (int heads : {1, 3}) for (int width : {1, 7, 128}) {
            ApplyDeviceMap({{device, 1}}, 1, 1);
            Data value(FLOAT32, {heads, 1, width}, std::vector<float>(heads * width, .75f));
            if (device != "cpu") value.ToDevice(CUDA, std::vector<int>{0});
            value.Expansion({heads, 2, width});
            value.Resize({heads, 0, width});
            value.Expansion({heads, 5, width});
            Check(value.dims == std::vector<int>({heads, 0, width}), "empty reserve changed size");
            Data row(FLOAT32, {heads, 1, width}, std::vector<float>(heads * width, .75f));
            CatDirect(value, row, 1); Values(value, {heads, 1, width});
            value.Expansion({heads, 9, width}); Values(value, {heads, 1, width});
            CatDirect(value, row, 1); Values(value, {heads, 2, width});
        }
        std::puts("PASS empty reserved tensor expansion"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
