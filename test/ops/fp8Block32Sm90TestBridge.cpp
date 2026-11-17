// Test-only Data adapter. Storage belongs to the Python test.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include "devices/cuda/cudaworkspace.h"
#include <exception>
#include <cstdio>

extern "C" void Fp8Block32TestArena(bool enabled) {
    static std::shared_ptr<fastllm::CudaWorkspace> previous;
    if (enabled) previous = fastllm::SetCudaWorkspace(std::make_shared<fastllm::CudaWorkspace>(0, 4u << 20));
    else fastllm::SetCudaWorkspace(std::move(previous));
}

extern "C" bool Fp8Block32TestLinear(void *x, void *w, void *scales, void *biasStorage,
                                    void *y, int rows, int inner, int channels, int mode) {
    try {
        if (mode == 1)
            return FastllmCudaBlock32W8A8SM90(x, w, static_cast<float *>(scales), y, rows, inner, channels);
        using namespace fastllm;
        Data input(BFLOAT16, {rows, inner}), weight(FP8_E4M3, {channels, inner});
        Data bias(FLOAT32), output(BFLOAT16, {rows, channels});
        for (auto *v : {&input, &weight, &bias, &output}) { v->isFake = true; v->dataDevice = DataDevice::CUDA; }
        input.cudaData = x; weight.cudaData = w; output.cudaData = y;
        weight.blockM = 32; weight.blockK = 1;
        weight.scales.resize(size_t(channels) * (inner / 32));
        weight.extraCudaData = {scales, biasStorage};
        // Bias deliberately exercises the unchanged W8A16 fallback.
        if (mode == 2) {
            bias.Resize({channels}); bias.cudaData = biasStorage;
            // Test storage contains the FP32 bias followed by its BF16 cache.
            weight.extraCudaHalfData = {static_cast<float *>(biasStorage) + channels};
        }
        return FastllmCudaBFloat16MatMulFP8E4M3(input, weight, bias, output, rows, inner, channels);
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return false; }
}
