#include "fastllm.h"
#include "utils.h"
#include "devices/cpu/computeutils.h"
#include "fastllm-cuda.cuh"
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
using namespace fastllm;
static void Check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
int main() {
 try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    constexpr int columns = 5120, outputs = 19;
    std::mt19937 rng(3179);
    Data weight(NVFP4_BLOCK_32_E8M0, {outputs, columns}); weight.blockK = 1; weight.blockM = 32;
    weight.Allocate(false);
    const size_t bytes = size_t(outputs) * (columns / 32) * 17;
    std::vector<uint8_t> packed(bytes);
    for (size_t b = 0; b < bytes; b += 17) {
        for (int j = 0; j < 16; ++j) packed[b+j] = rng() & 255;
        packed[b+16] = 99 + rng() % 28;
    }
    // Deliberate cancellation: the CPU's block/pair order gives
    // (1 - 1) + 2^-27. The old 64-lane reduction lost the small term first.
    std::fill(packed.begin(), packed.begin() + (columns / 32) * 17, uint8_t(0));
    for (int block = 0; block < columns / 32; ++block) packed[block * 17 + 16] = 127;
    packed[0] = 2; packed[17] = 10; packed[34] = 2; packed[50] = 100;
    std::memcpy(weight.cpuData, packed.data(), bytes);
    weight.ToDevice(CUDA);
    for (int rows : {1, 2, 3, 7, 16, 31}) {
        Data input(BFLOAT16, {rows, columns}), output(BFLOAT16, {rows, outputs}), bias;
        input.Allocate(false);
        std::vector<uint16_t> x(rows * columns);
        for (auto &v : x) v = Float32ToBFloat16RNEBits(float(int(rng() % 255) - 127) / 64.f);
        for (int row = 0; row < rows; ++row)
            for (int col : {0, 32, 64}) x[row * columns + col] = Float32ToBFloat16RNEBits(1.f);
        std::memcpy(input.cpuData, x.data(), x.size() * 2);
        std::vector<float> expected(rows * outputs);
        MultiThreadGemmOp oracle((uint8_t*)x.data(), BFLOAT16, packed.data(), NVFP4_BLOCK_32_E8M0,
            (uint8_t*)expected.data(), FLOAT32, rows, columns, outputs, 0, outputs);
        oracle.Run();
        input.ToDevice(CUDA); output.ToDevice(CUDA, {0}, false); output.Allocate(false);
        auto run = [&] {
            if (!FastllmCudaBFloat16MatMulNVFP4Block16E8M0(input, weight, bias, output, rows, columns, outputs))
                throw std::runtime_error("MXFP4 linear rejected");
        };
        auto verify = [&] {
            std::vector<uint16_t> actual(rows * outputs);
            Check(cudaMemcpy(actual.data(), output.cudaData, actual.size() * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < actual.size(); ++i)
                if (actual[i] != Float32ToBFloat16RNEBits(expected[i])) {
                    std::fprintf(stderr, "Mismatch rows=%d column=%zu actual=%04x expected=%04x\n",
                        rows, i, actual[i], Float32ToBFloat16RNEBits(expected[i]));
                    throw std::runtime_error("MXFP4 CPU/CUDA rounding differs");
                }
        };
        run(); verify();
        cudaGraph_t graph; cudaGraphExec_t executable;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); run();
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        Check(cudaGraphLaunch(executable, cudaStreamPerThread)); Check(cudaStreamSynchronize(cudaStreamPerThread));
        verify(); Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
    }
    std::puts("PASS MXFP4 linear CPU/CUDA exact, row/column tails and graph replay"); return 0;
 } catch (const std::exception &e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
