#include "hy4_flash_hybrid_graph.h"
#include "hy4_flash_ops.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>

namespace hy4 = fastllm::hy4;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2)
        return 77;
    for (int device : {0, 1}) {
        int access = 0;
        if (cudaDeviceCanAccessPeer(&access, device, 1 - device) != cudaSuccess)
            return 1;
        if (!access)
            return 77;
    }
    if (!FastllmCudaPeerAccessInit({0, 1}))
        return 1;

    hy4::Scratch first(0), second(1);
    float *input = first.Get("input", 1);
    float *intermediate = first.Get("intermediate", 1);
    float *output = second.Get("output", 1);
    hy4::HybridGraphs graphs;
    const hy4::GraphKey key{0, 1};
    const std::array<float, 3> values{7.f, -3.f, 11.f};
    for (size_t step = 0; step < values.size(); ++step) {
        const float value = values[step];
        hy4::Select(0);
        hy4::CopyToGPU(input, &value, sizeof(value));
        if (step == 0) {
            hy4::HybridGraphCapture capture(&graphs, false);
            capture.Start(0);
            hy4::CopyStageCaptured(intermediate, input, 1);
            capture.CloseStage();
            capture.Start(1);
            hy4::CopyStageCaptured(output, intermediate, 1);
            capture.Finish(key, 0, 1);
        } else if (!hy4::ReplayHybrid(graphs, key, 0, 1)) {
            return 1;
        }
        hy4::Select(1);
        float actual = 0.f;
        hy4::CopyToHost(&actual, output, sizeof(actual));
        if (actual != value) {
            std::fprintf(stderr, "HY4 PP graph expected %g, got %g\n", value, actual);
            return 1;
        }
    }
    std::puts("PASS: HY4 two-stage graph replay");
}
