#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static void Require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}
static void Fill(Data &data, const std::vector<int> &shape, DataType type, int seed, float amplitude) {
    size_t count = 1;
    for (int dimension : shape) count *= dimension;
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i)
        values[i] = std::sin(float((i * 37 + seed * 19) % 1031)) * amplitude;
    data.CopyFrom(Data(DataType::FLOAT32, shape, values));
    ToDataType(data, type);
    data.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
static std::vector<unsigned char> Bits(const Data &data) {
    std::vector<unsigned char> bytes(data.GetBytes());
    Check(cudaMemcpy(bytes.data(), data.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
    return bytes;
}
static std::vector<float> Floats(const Data &data) {
    Data copy;
    copy.CopyFrom(data);
    ToDataType(copy, DataType::FLOAT32);
    copy.ToDevice(DataDevice::CPU);
    const float *p = reinterpret_cast<const float *>(copy.cpuData);
    return {p, p + copy.Count(0)};
}
static int CheckMixNorm() {
    int cases = 0;
    for (int dim : {4096, 5120, 6144, 8192}) {
        for (int batch : {1, 2, 4, 8, 15}) {
            for (float amplitude : {0.3f, 2.0f}) {
                Data input, previousPre, fn, scale, base, weight, pre[2], post[2], comb[2], out[2];
                Fill(input, {1, batch, 4, dim}, DataType::BFLOAT16, 1, amplitude);
                Fill(previousPre, {1, batch, 4}, DataType::FLOAT32, 2, 1.0f);
                Fill(fn, {24, 4 * dim}, DataType::FLOAT32, 3, 0.01f);
                Fill(scale, {3}, DataType::FLOAT32, 4, 1.0f);
                Fill(base, {24}, DataType::FLOAT32, 5, 0.1f);
                Fill(weight, {dim}, DataType::FLOAT32, 6, 1.0f);
                Require(FastllmCudaDeepSeekV41HcMix(input, fn, scale, base, 4, 20, 1e-6f, 1e-6f,
                    pre[0], post[0], comb[0]), "reference mix rejected");
                Require(FastllmCudaDeepSeekV41HcPreNorm(input, previousPre, weight, 1e-6f, out[0]),
                        "reference norm rejected");
                Require(FastllmCudaDeepSeekV41HcMixPreNorm(input, previousPre, fn, scale, base, weight,
                    20, 1e-6f, 1e-6f, pre[1], post[1], comb[1], out[1]), "fused mix/norm rejected");
                Require(Bits(pre[0]) == Bits(pre[1]) && Bits(post[0]) == Bits(post[1]) &&
                        Bits(comb[0]) == Bits(comb[1]) && Bits(out[0]) == Bits(out[1]),
                        "mix/norm changed bits or consumed the wrong pre coefficients");
                ++cases;
            }
        }
    }
    Data input, previousPre, fn, scale, base, weight, pre, post, comb, out;
    Fill(previousPre, {4}, DataType::FLOAT32, 2, 1.0f);
    Fill(out, {1}, DataType::BFLOAT16, 3, 1.0f);
    const auto before = Bits(out);
    for (int dim : {4094, 4097, 8194}) {
        Fill(input, {1, 1, 4, dim}, DataType::BFLOAT16, 1, 0.3f);
        Fill(fn, {24, 4 * dim}, DataType::FLOAT32, 3, 0.01f);
        Fill(scale, {3}, DataType::FLOAT32, 4, 1.0f);
        Fill(base, {24}, DataType::FLOAT32, 5, 0.1f);
        Fill(weight, {dim}, DataType::FLOAT32, 6, 1.0f);
        Require(!FastllmCudaDeepSeekV41HcMixPreNorm(input, previousPre, fn, scale, base, weight,
            20, 1e-6f, 1e-6f, pre, post, comb, out) && Bits(out) == before,
            "unsupported mix/norm dimensions must leave output untouched");
        ++cases;
    }
    return cases;
}
static int CheckRouter() {
    int cases = 0;
    for (int dim : {64, 513, 5120}) for (int experts : {64, 384, 1024})
    for (int batch : {1, 2, 4, 8}) for (float temperature : {1.0f, 0.7f, 2.0f}) {
        Data input, weight, cast, reference, actual;
        Fill(input, {batch, dim}, DataType::BFLOAT16, 1, 0.8f);
        Fill(weight, {experts, dim}, DataType::FLOAT32, 2, 0.15f);
        ToDataType(input, cast, DataType::FLOAT32);
        Linear(cast, weight, Data(), reference);
        if (std::fabs(temperature - 1.0f) > 1e-6f) Mul(reference, 1.0f / temperature, reference);
        Require(FastllmCudaDeepSeekV4RouteScoreTransform(reference, 2), "reference score rejected");
        Require(FastllmCudaV41RouterScores(input, weight, temperature, actual), "router rejected");
        Require(Bits(reference) == Bits(actual), "router score changed bits");
        ++cases;
    }
    Data input, weight, sentinel;
    Fill(input, {9, 64}, DataType::BFLOAT16, 1, 1.0f);
    Fill(weight, {64, 64}, DataType::FLOAT32, 2, 1.0f);
    Fill(sentinel, {1}, DataType::FLOAT32, 3, 1.0f);
    const auto before = Bits(sentinel);
    Require(!FastllmCudaV41RouterScores(input, weight, 1.0f, sentinel) && Bits(sentinel) == before,
            "unsupported router batch must leave the output untouched");
    return cases + 1;
}
static int CheckWoA() {
    int cases = 0;
    for (int dim : {17, 512, 4096}) for (int groups : {1, 3}) for (int rank : {6, 1024}) {
        Data input, weight, output, expected;
        Fill(input, {1, 1, groups, dim}, DataType::BFLOAT16, 1, 0.3f);
        Fill(weight, {groups, rank, dim}, DataType::BFLOAT16, 2, 0.01f);
        const auto x = Floats(input), w = Floats(weight);
        std::vector<float> values(groups * rank);
        for (int group = 0; group < groups; ++group) for (int row = 0; row < rank; ++row) {
            float partial[256] = {};
            for (int lane = 0; lane < 256; ++lane) for (int d = lane; d < dim; d += 256)
                partial[lane] = std::fma(x[group * dim + d], w[(group * rank + row) * dim + d], partial[lane]);
            for (int stride = 128; stride; stride >>= 1)
                for (int lane = 0; lane < stride; ++lane) partial[lane] += partial[lane + stride];
            values[group * rank + row] = partial[0];
        }
        expected.CopyFrom(Data(DataType::FLOAT32, {1, 1, groups, rank}, values));
        ToDataType(expected, DataType::BFLOAT16);
        expected.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        Require(FastllmCudaDeepSeekV4WoA(input, weight, groups, rank, output, false), "WoA rejected");
        Require(Bits(output) == Bits(expected), "WoA changed the original FP32 reduction");
        ++cases;
    }
    return cases;
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
        FastllmCudaSetDevice(0);
        const int cases = CheckMixNorm() + CheckRouter() + CheckWoA();
        std::printf("PASS V4.1 decode projections: %d exact cases\n", cases);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
