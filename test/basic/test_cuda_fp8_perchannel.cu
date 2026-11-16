#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

extern __global__ void FastllmCudaFP8E4M3PERCHANNEL2BF16Kernel(uint8_t *, __nv_bfloat16 *, int);

static void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Preserve the existing per-channel grid, including its reserved-code values.
static float Decode(uint8_t code) {
    int magnitude = code & 127;
    float value = magnitude < 8 ? std::ldexp(float(magnitude), -9) :
        std::ldexp(float(8 + (magnitude & 7)), (magnitude >> 3) - 10);
    return (code & 128) ? -value : value;
}

static void Run(int columns, int inputOffset, int outputOffset) {
    constexpr int rows = 257, guard = 8;
    const float scales[] = {0.f, -0.f, 1.f, -1.f, .01317f, 1.e-30f, 1.e20f};
    const size_t count = size_t(rows) * columns;
    std::vector<uint8_t> input(inputOffset + size_t(rows) * (columns + 4));
    std::vector<uint16_t> output(outputOffset + count + guard, 0x5a5a), expected(output);
    uint8_t *deviceInput = nullptr;
    __nv_bfloat16 *deviceOutput = nullptr;
    Check(cudaMalloc(&deviceInput, input.size()));
    Check(cudaMalloc(&deviceOutput, output.size() * 2));
    Check(cudaMemcpy(deviceOutput, output.data(), output.size() * 2, cudaMemcpyHostToDevice));
    Check(cudaDeviceSynchronize());
    cudaStream_t stream;
    Check(cudaStreamCreate(&stream));
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    FastllmCudaFP8E4M3PERCHANNEL2BF16Kernel<<<rows, 128, 0, stream>>>(
        deviceInput + inputOffset, deviceOutput + outputOffset, columns);
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Check(cudaStreamEndCapture(stream, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
        for (int row = 0; row < rows; ++row) {
            uint8_t *values = input.data() + inputOffset + size_t(row) * (columns + 4);
            float scale = scales[(row + replay) % 7];
            std::memcpy(values + columns, &scale, 4);
            for (int column = 0; column < columns; ++column) {
                values[column] = uint8_t(row + column + replay * 13);
                __nv_bfloat16 value = __float2bfloat16_rn(Decode(values[column]) * scale);
                std::memcpy(&expected[outputOffset + size_t(row) * columns + column], &value, 2);
            }
        }
        Check(cudaMemcpyAsync(deviceInput, input.data(), input.size(), cudaMemcpyHostToDevice, stream));
        Check(cudaGraphLaunch(executable, stream));
        Check(cudaStreamSynchronize(stream));
        Check(cudaMemcpy(output.data(), deviceOutput, output.size() * 2, cudaMemcpyDeviceToHost));
        if (output != expected) {
            for (size_t i = 0; i < output.size(); ++i) if (output[i] != expected[i]) {
                std::fprintf(stderr, "columns=%d offsets=%d/%d replay=%d index=%zu got=%04x expected=%04x\n",
                    columns, inputOffset, outputOffset, replay, i, output[i], expected[i]);
                break;
            }
            throw std::runtime_error("per-channel conversion or guard mismatch");
        }
    }
    Check(cudaGraphExecDestroy(executable));
    Check(cudaGraphDestroy(graph));
    Check(cudaStreamDestroy(stream));
    Check(cudaFree(deviceInput));
    Check(cudaFree(deviceOutput));
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        for (int columns : {1, 3, 4, 7, 16, 31, 128, 257, 511, 512, 1024, 4096, 8192})
            for (int inputOffset : {0, 1})
                for (int outputOffset : {0, 1}) Run(columns, inputOffset, outputOffset);
        puts("PASS FP8 per-channel conversion: 52 layouts, exhaustive codes, two graph replays");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
