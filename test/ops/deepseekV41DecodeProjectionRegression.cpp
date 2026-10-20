#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    for (int dim : {4096, 4128, 5120, 6144, 8192}) {
        for (int batch : {1, 2, 4, 8, 15}) {
            for (float amplitude : {0.3f, 2.0f}) {
                Data input, previousPre, fn, scale, base, weight, pre[2], post[2], comb[2], out[2];
                Data quantReference;
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
                Fill(quantReference, {1, batch, dim}, DataType::BFLOAT16, 7, 0.0f);
                Require(FastllmCudaDeepSeekV41QuantizeActivation(out[0], quantReference),
                        "reference activation quantization rejected");
                Data quantFusedOut;
                Require(FastllmCudaDeepSeekV41HcMixPreNorm(input, previousPre, fn, scale, base, weight,
                    20, 1e-6f, 1e-6f, pre[1], post[1], comb[1], out[1], &quantFusedOut),
                    "fused mix/norm quantization rejected");
                Require(Bits(quantReference) == Bits(quantFusedOut),
                        "fused mHC quantization differs from the standalone block-32 quantizer");
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
// A repeated >=16-token batch uses the independent HcBatchDots implementation.
// Compare all outputs bitwise, including dimensions with partial prefetch tiles
// and dtypes/sizes that retain the portable decode loop.
static int CheckHcDecodeDots() {
    int cases = 0;
    for (auto type : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32})
    for (int dim : {63, 4095, 4096, 4098, 5120, 6144, 8192, 8193})
    for (int batch : {1, 2, 8, 15}) {
        Data input, repeated, fn, scale, base, pre[2], post[2], comb[2];
        Fill(input, {1, batch, 4, dim}, type, batch + dim, dim == 63 ? 0.0f : 0.8f);
        Fill(fn, {24, 4 * dim}, DataType::FLOAT32, 3, 0.07f);
        Fill(scale, {3}, DataType::FLOAT32, 4, 1.0f);
        Fill(base, {24}, DataType::FLOAT32, 5, 0.3f);
        const auto bytes = Bits(input);
        const int copies = (16 + batch - 1) / batch;
        repeated.dataType = type;
        repeated.UpdateUnitSize();
        repeated.Resize({1, batch * copies, 4, dim});
        repeated.Allocate();
        for (int i = 0; i < copies; ++i)
            std::memcpy(repeated.cpuData + i * bytes.size(), bytes.data(), bytes.size());
        repeated.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        const float normEps = dim == 63 ? 1e-6f : 0.0f;
        Require(FastllmCudaDeepSeekV41HcMix(repeated, fn, scale, base, 4, 20, 1e-6f, normEps,
                    pre[0], post[0], comb[0]), "batched HC dot reference rejected");
        Require(FastllmCudaDeepSeekV41HcMix(input, fn, scale, base, 4, 20, 1e-6f, normEps,
                    pre[1], post[1], comb[1]), "HC decode dots rejected");
        for (auto outputs : {pre, post, comb}) {
            auto expected = Bits(outputs[0]);
            const auto actual = Bits(outputs[1]);
            expected.resize(actual.size());
            Require(actual == expected, "HC decode/batch dots changed FP32 accumulation order");
        }
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
static int CheckFusedSwigluQuantization() {
    int cases = 0;
    for (int rows : {1, 3, 6, 17, 96})
    for (int width : {32, 96, 2304, 4096})
    for (float amplitude : {0.0f, 0.3f, 12.0f})
    for (float limit : {0.0f, 10.0f}) {
        Data input, reference, fused;
        Fill(input, {rows, 2 * width}, DataType::BFLOAT16, rows + width, amplitude);
        Require(FastllmCudaSwigluClamped(input, limit, reference), "reference SwiGLU rejected");
        Require(FastllmCudaDeepSeekV41QuantizeActivation(reference, reference),
                "reference quantization rejected");
        Fill(fused, {rows, width}, DataType::BFLOAT16, 4, 0.0f);
        Require(FastllmCudaDeepSeekV41BFloat16SwigluQuantized(input.cudaData, fused.cudaData,
                rows, width, limit, nullptr), "fused SwiGLU/quantization rejected");
        Require(Bits(reference) == Bits(fused), "fused SwiGLU changed the BF16 rounding boundary");
        ++cases;
    }
    Data output;
    Fill(output, {1, 32}, DataType::BFLOAT16, 1, 1.0f);
    const auto before = Bits(output);
    Require(!FastllmCudaDeepSeekV41BFloat16SwigluQuantized(output.cudaData, output.cudaData,
            1, 31, 10.0f, nullptr) && Bits(output) == before,
            "unsupported quantization group mutated output");
    return cases + 1;
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
        FastllmCudaSetDevice(0);
        const int cases = CheckMixNorm() + CheckHcDecodeDots() + CheckRouter() + CheckWoA() +
                          CheckFusedSwigluQuantization();
        std::printf("PASS V4.1 decode projections: %d exact cases\n", cases);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
