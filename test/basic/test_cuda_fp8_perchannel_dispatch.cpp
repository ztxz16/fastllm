#include "fastllm.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;

namespace {
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

std::vector<uint16_t> Read(const Data &data) {
    FastllmCudaSyncCurrentThreadStream();
    std::vector<uint16_t> result(data.Count(0));
    Require(cudaMemcpy(result.data(), data.cudaData, result.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess, "copy failed");
    return result;
}

void RunCase(int device, DataType type, int rows, bool hasBias) {
    constexpr int cols = 2048, outputs = 3072;
    std::mt19937 rng(42);
    std::vector<float> values(rows * cols), biasValues(outputs);
    for (float &v : values) v = (int(rng() % 8193) - 4096) / 2048.0f;
    for (float &v : biasValues) v = (int(rng() % 257) - 128) / 1024.0f;
    Data input(type, {rows, cols}, values), weight(FP8_E4M3, {outputs, cols});
    weight.blockK = 1;
    weight.blockM = cols;
    weight.scales.assign(outputs, 0.03125f);
    weight.Allocate();
    for (uint64_t i = 0; i < weight.Count(0); ++i)
        weight.cpuData[i] = uint8_t(rng() % 96) | uint8_t((rng() % 2) << 7);
    Data bias(FLOAT32), output(type, {rows, outputs});
    if (hasBias) {
        bias.Resize({outputs});
        bias.Allocate();
        std::memcpy(bias.cpuData, biasValues.data(), biasValues.size() * sizeof(float));
    }
    input.ToDevice(DataDevice::CUDA, {device}, true);
    weight.ToDevice(DataDevice::CUDA, {device}, true);
    if (hasBias) bias.ToDevice(DataDevice::CUDA, {device}, true);
    output.dataDevice = DataDevice::CUDA;
    output.dataDeviceIds = {device};
    output.Allocate();

    auto native = [&]() {
        bool ok = type == FLOAT16
            ? FastllmCudaHalfMatMulFloatFP8E4M3(input, weight, bias, output, rows, cols, outputs)
            : FastllmCudaBFloat16MatMulFP8E4M3(input, weight, bias, output, rows, cols, outputs);
        Require(ok, "native reference failed");
        return Read(output);
    };
    const auto expected = native();
    Require(FastllmCudaCutlassLinearFP8E4M3PerChannel(
                input, weight, bias, output, rows, cols, outputs), "CUTLASS reference failed");
    const auto quantized = Read(output);
    Require(expected != quantized, "test inputs must distinguish activation precision");

    auto dispatch = [&]() {
        DoCudaLinear(input, weight, bias, output);
        return Read(output);
    };
    Require(unsetenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH") == 0, "unsetenv failed");
    auto observed = dispatch();
    if (rows < 8) Require(observed == expected, "default small batch lost native activation precision");
    else Require(observed == expected || observed == quantized, "unexpected autotuned output");

    // Permit tuning, then raise the threshold after its decision is cached.
    Require(setenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH", "1", 1) == 0, "setenv failed");
    observed = dispatch();
    Require(observed == expected || observed == quantized, "override output is invalid");
    const auto threshold = std::to_string(rows + 1);
    Require(setenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH", threshold.c_str(), 1) == 0, "setenv failed");
    Require(dispatch() == expected, "cached choice bypassed minimum batch");

    void *graph = nullptr, *exec = nullptr;
    Require(FastllmCudaGraphBeginCapture(), "capture begin failed");
    DoCudaLinear(input, weight, bias, output);
    bool ended = FastllmCudaGraphEndCapture(&graph);
    Require(ended && graph, "capture end failed");
    bool instantiated = FastllmCudaGraphInstantiate(graph, &exec);
    FastllmCudaGraphDestroy(graph);
    Require(instantiated, "graph instantiate failed");
    bool launched = FastllmCudaGraphLaunch(exec);
    FastllmCudaSyncCurrentThreadStream();
    FastllmCudaGraphExecDestroy(exec);
    Require(launched && Read(output) == expected, "graph fallback changed output");
    Require(unsetenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH") == 0, "unsetenv failed");
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        Require(setenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8", "1", 1) == 0, "setenv failed");
        SetCudaGraph(true);
        int cases = 0;
        for (int device = 0; device < devices; ++device) {
            cudaDeviceProp props;
            Require(cudaGetDeviceProperties(&props, device) == cudaSuccess, "device query failed");
            if (props.major != 8 || props.minor != 9) continue;
            FastllmCudaSetDevice(device);
            for (auto type : {FLOAT16, BFLOAT16}) {
                for (int rows : {1, 2, 4, 7, 8, 16, 32, 128}) {
                    RunCase(device, type, rows, (rows % 2) == 0);
                    ++cases;
                }
            }
        }
        if (!cases) return 77;
        std::printf("PASS: %d FP8 per-channel dispatch cases (precision, override, cache, graph)\n", cases);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
