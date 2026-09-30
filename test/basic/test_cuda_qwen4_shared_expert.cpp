#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Init(Data &data, const std::vector<int> &shape, int seed, int device) {
    data.Resize(shape);
    data.Allocate();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-.1f, .1f);
    for (uint64_t i = 0; i < data.Count(0); ++i) {
        const float value = distribution(rng);
        if (data.dataType == FLOAT16) {
            ((uint16_t *)data.cpuData)[i] = float_to_half(value);
        } else {
            ((float *)data.cpuData)[i] = value;
        }
    }
    data.ToDevice(DataDevice::CUDA, {device}, true);
}

static void Equal(const Data &actual, const Data &expected, bool exact) {
    Require(actual.dims == expected.dims, "shared expert output shape differs");
    std::vector<float> a(actual.Count(0)), b(expected.Count(0));
    Require(cudaMemcpy(a.data(), actual.cudaData, actual.GetBytes(),
                       cudaMemcpyDeviceToHost) == cudaSuccess, "read actual output");
    Require(cudaMemcpy(b.data(), expected.cudaData, expected.GetBytes(),
                       cudaMemcpyDeviceToHost) == cudaSuccess, "read reference output");
    if (exact) {
        Require(std::memcmp(a.data(), b.data(), actual.GetBytes()) == 0,
                "shared expert changed ordinary decode output bits");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        Require(std::isfinite(a[i]) && std::isfinite(b[i]) &&
                    std::abs(a[i] - b[i]) <= 2e-6f + 2e-6f * std::abs(b[i]),
                "shared expert differs from separate operators");
    }
}

static void Run(int width, int intermediate, int rows, int device) {
    Data input(FLOAT32), upWeight(FLOAT16), downWeight(FLOAT16), gateWeight(FLOAT16);
    Init(input, {1, rows, width}, 1, device);
    Init(upWeight, {2 * intermediate, width}, 3, device);
    Init(downWeight, {width, intermediate}, 7, device);
    Init(gateWeight, {1, width}, 9, device);
    Data up(FLOAT32), hidden(FLOAT32), gate(FLOAT32), output(FLOAT32);
    Data refUp(FLOAT32), refHidden(FLOAT32), refGate(FLOAT32), refOutput(FLOAT32);
    Init(refUp, {1, rows, 2 * intermediate}, 0, device);
    Init(refHidden, {1, rows, intermediate}, 0, device);
    Init(refGate, {1, rows, 1}, 0, device);
    Init(refOutput, {1, rows, width}, 0, device);
    Data bias(FLOAT32);
    Require(FastllmCudaMatMulFloat16(input, upWeight, bias, refUp,
                                   rows, width, 2 * intermediate), "reference up projection");
    Require(FastllmCudaSwiglu(refUp, refHidden), "reference activation");
    Require(FastllmCudaMatMulFloat16(refHidden, downWeight, bias, refOutput,
                                   rows, intermediate, width), "reference down projection");
    Require(FastllmCudaMatMulFloat16(input, gateWeight, bias, refGate,
                                   rows, width, 1), "reference gate projection");
    Require(FastllmCudaSigmoidMulTo(refOutput, refGate), "reference output gate");
    // The MoE caller creates a view before dispatch. Its device must survive
    // even when no ordinary operator runs before the fused implementation.
    Data view;
    view.FakeFrom(input, 0);
    view.Resize(input.dims);
    Require(view.dataDeviceIds == input.dataDeviceIds, "view lost device placement");
    auto fused = [&] {
        Require(FastllmCudaQwen4SharedExpert(view, upWeight, downWeight,
                    gateWeight, up, hidden, gate, output), "valid fusion rejected");
    };
    const bool exact = width == 2560 && (intermediate == 320 || intermediate == 640);
    fused();
    Equal(output, refOutput, exact);
    Equal(up, refUp, false);

    // Reuse the same workspaces through capture and repeated replay.
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Require(cudaStreamBeginCapture(cudaStreamPerThread,
                cudaStreamCaptureModeThreadLocal) == cudaSuccess, "begin capture");
    fused();
    Require(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "end capture");
    Require(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess, "instantiate graph");
    for (int repeat = 0; repeat < 3; ++repeat) {
        Require(cudaGraphLaunch(executable, cudaStreamPerThread) == cudaSuccess, "replay graph");
    }
    Require(cudaDeviceSynchronize() == cudaSuccess, "finish graph");
    Equal(output, refOutput, exact);
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    std::printf("PASS device=%d width=%d intermediate=%d rows=%d exact=%d\n",
                device, width, intermediate, rows, exact);
}

static void CheckFallbacks(int device) {
    Data input(FLOAT32), upWeight(FLOAT16), downWeight(FLOAT16), gateWeight(FLOAT16);
    Data up, hidden, gate, output;
    Init(input, {8, 64}, 1, device);
    Init(upWeight, {128, 64}, 2, device);
    Init(downWeight, {64, 64}, 3, device);
    Init(gateWeight, {1, 64}, 4, device);
    auto rejected = [&] {
        return !FastllmCudaQwen4SharedExpert(input, upWeight, downWeight,
                    gateWeight, up, hidden, gate, output);
    };
    Require(rejected(), "large batch must use ordinary operators");
    input.Resize({1, 64});
    input.strides[1] = 2;
    Require(rejected(), "strided input must use ordinary operators");
    input.strides[1] = 1;
    void *originalInput = input.cudaData;
    input.cudaData = (uint8_t *)originalInput + sizeof(float);
    const bool unalignedRejected = rejected();
    input.cudaData = originalInput;
    Require(unalignedRejected, "unaligned input must use ordinary operators");
    upWeight.dataType = FLOAT32;
    Require(rejected(), "unsupported weight dtype must use ordinary operators");
    upWeight.dataType = FLOAT16;
    output.isFake = true;
    Require(rejected(), "borrowed output must use ordinary operators");
    output.isFake = false;
}

int main() {
    try {
        SetThreads(2);
        const int devices = std::min(2, FastllmCudaGetDeviceCount());
        if (devices == 0) return 77;
        for (int device = 0; device < devices; ++device) {
            FastllmCudaSetDevice(device);
            for (int rows = 1; rows <= 7; ++rows) {
                for (auto shape : std::vector<std::pair<int, int>>{
                        {64, 64}, {320, 320}, {640, 640}, {2560, 320},
                        {2560, 640}, {2560, 1536}, {3072, 1024}}) {
                    Run(shape.first, shape.second, rows, device);
                }
            }
            CheckFallbacks(device);
        }
        std::puts("ALL_PASS");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
