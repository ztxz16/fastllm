#include "fastllm.h"
#include "utils.h"
#include "devices/cpu/computeutils.h"
#include "fastllm-cuda.cuh"
#include "utils/bfloat16_silu.h"
#include "../../src/devices/cuda/fastllm-bfloat16-silu.cuh"
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
__global__ void LookupAll(const float *table, float *result) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] = FastllmCudaBFloat16Silu(__uint_as_float(i << 16), table);
}
// Regression for CPU libm/CUDA expf differences crossing the BF16 midpoint
// after multiplication by the routing score, before block FP8 quantization.
static void CheckActivation(int rows, int block, bool quantize) {
    constexpr int inter = 128;
    std::vector<float> gateUp(rows * 2 * inter), scores(rows), expected(rows * inter);
    for (int row = 0; row < rows; ++row) {
        scores[row] = row % 3 == 0 ? 0.16227596998214722f :
            std::nextafter(0.16227596998214722f, row % 3 == 1 ? 0.f : 1.f);
        for (int col = 0; col < inter; ++col) {
            // The first pair is the saved layer-19/expert-207 failure.
            float g = col == 0 ? -0.3347304f : (int((row * 73 + col * 31) % 513) - 256) / 16.f;
            float u = col == 0 ? 0.33335018f : (int((row * 19 + col * 17) % 513) - 256) / 16.f;
            g = RoundFloat32ToBFloat16RNE(g); u = RoundFloat32ToBFloat16RNE(u);
            gateUp[row * 2 * inter + 2 * col] = g;
            gateUp[row * 2 * inter + 2 * col + 1] = u;
            g = std::min(g, 10.f); u = std::clamp(u, -10.f, 10.f);
            float activated = g / (1.f + std::exp(-g));
            expected[row * inter + col] = RoundFloat32ToBFloat16RNE((activated * u) * scores[row]);
        }
        if (quantize) for (int col = 0; col < inter; col += block)
            QuantizeDequantizeFP8E4M3Block128(expected.data() + row * inter + col, block);
    }
    Data input(FLOAT32, {rows, 2 * inter}, gateUp), scale(FLOAT32, {rows}, scores), output;
    ToDataType(input, BFLOAT16); input.ToDevice(CUDA); scale.ToDevice(CUDA);
    auto run = [&] {
        Require(FastllmCudaDeepSeekV4PrepareMoeDownInput(input, output,
            (float *)scale.cudaData, 10.f, quantize, block), "activation rejected");
    };
    auto verify = [&] {
        std::vector<uint16_t> actual(rows * inter);
        Check(cudaMemcpy(actual.data(), output.cudaData, actual.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != Float32ToBFloat16RNEBits(expected[i])) {
                std::fprintf(stderr, "activation mismatch rows=%d block=%d quantize=%d index=%zu\n",
                             rows, block, quantize, i);
                throw std::runtime_error("CPU/CUDA activation is not bit-exact");
            }
        }
    };
    run(); verify();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    run();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int repeat = 0; repeat < 3; ++repeat) {
        Check(cudaGraphLaunch(executable, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread)); verify();
    }
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        for (int device = 0; device < std::min(2, devices); ++device) {
            FastllmCudaSetDevice(device);
            const float *table = FastllmCudaBFloat16SiluTable();
            Require(table && table == FastllmCudaBFloat16SiluTable(), "unstable SiLU table");
            float *output; Check(cudaMalloc(&output, 65536 * sizeof(float)));
            LookupAll<<<256, 256, 0, cudaStreamPerThread>>>(table, output);
            std::vector<float> actual(65536);
            Check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            const auto &expected = GetBFloat16SiluLookup();
            for (size_t i = 0; i < actual.size(); ++i)
                Require((std::isnan(actual[i]) && std::isnan(expected[i])) ||
                    std::memcmp(&actual[i], &expected[i], sizeof(float)) == 0, "SiLU lookup differs");
            Check(cudaFree(output));
            for (int rows : {1, 2, 6, 9, 65}) for (int block : {32, 128})
                for (bool quantize : {false, true}) CheckActivation(rows, block, quantize);
            std::printf("PASS device=%d BF16 values=65536 activation cases=20 eager+3 graph replays\n", device);
        }
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
