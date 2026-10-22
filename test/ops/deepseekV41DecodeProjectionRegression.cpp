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
    for (int dim : {63, 4095, 4096, 4098, 5120, 6144, 8190, 8192, 8193})
    for (int batch : {1, 2, 3, 4, 8, 9, 15}) {
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
// Preserve exact historical reductions on fallback paths. Tensor Core cases
// use an independent FP64 dot-product oracle and exact eager/graph parity.
static int CheckWoA() {
    struct Shape { int batch, seq, groups, rank, dim; };
    const Shape shapes[] = {
        {1,1,3,6,17}, {1,2,1,6,255}, {1,3,3,6,257}, {1,4,1,6,512},
        {1,5,3,6,4095}, {1,6,1,1024,4096}, {1,7,3,6,4097},
        {1,8,1,1024,4096}, {1,9,3,6,513}, {1,15,1,6,1025},
        {2,3,3,6,512}, {2,4,1,6,4096},
        {1,2,1,1024,4096}, {1,3,1,1024,4096}, {1,4,1,1024,4096},
        {1,5,1,1024,4096}, {1,7,1,1024,4096}, {1,6,3,16,512}, {2,4,4,128,256}
    };
    int cases = 0;
    auto check = [&](Shape shape, DataType inputType, DataType weightType,
                     float amplitude, int weightSeed) {
        const int tokens = shape.batch * shape.seq, groups = shape.groups,
                  rank = shape.rank, dim = shape.dim;
        Data input, weight, output, expected;
        Fill(input, {shape.batch, shape.seq, groups, dim}, inputType, tokens, amplitude);
        Fill(weight, {groups, rank, dim}, weightType, weightSeed, 0.01f);
        const auto x = Floats(input), w = Floats(weight);
        std::vector<float> values(tokens * groups * rank);
        for (int token = 0; token < tokens; ++token)
        for (int group = 0; group < groups; ++group) for (int row = 0; row < rank; ++row) {
            float partial[256] = {};
            for (int lane = 0; lane < 256; ++lane) for (int d = lane; d < dim; d += 256)
                partial[lane] = std::fma(x[(token * groups + group) * dim + d],
                                        w[(group * rank + row) * dim + d], partial[lane]);
            for (int stride = 128; stride; stride >>= 1)
                for (int lane = 0; lane < stride; ++lane) partial[lane] += partial[lane + stride];
            values[(token * groups + group) * rank + row] = partial[0];
        }
        expected.CopyFrom(Data(DataType::FLOAT32, {shape.batch, shape.seq, groups * rank}, values));
        ToDataType(expected, DataType::BFLOAT16);
        expected.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        const auto expectedBits = Bits(expected);
        Require(FastllmCudaDeepSeekV4WoA(input, weight, groups, rank, output, false), "WoA rejected");
        Require(output.dims == expected.dims, "WoA output shape changed");
        const bool tensorCore = tokens >= 2 && tokens <= 8 && dim % 8 == 0 && rank % 8 == 0 &&
            inputType == DataType::BFLOAT16 && weightType == DataType::BFLOAT16 &&
            FastllmCudaRuntimeArch() >= 80;
        const auto eagerBits = Bits(output);
        if (tensorCore) {
            const auto actual = Floats(output);
            double error = 0.0, norm = 0.0;
            for (int token = 0; token < tokens; ++token)
            for (int group = 0; group < groups; ++group) for (int row = 0; row < rank; ++row) {
                double exact = 0.0, sumAbs = 0.0;
                for (int d = 0; d < dim; ++d) {
                    const double product = double(x[(token * groups + group) * dim + d]) *
                                           w[(group * rank + row) * dim + d];
                    exact += product; sumAbs += std::abs(product);
                }
                const double delta = actual[(token * groups + group) * rank + row] - exact;
                Require(std::isfinite(delta) && std::abs(delta) <=
                        0.004 * std::abs(exact) + 2e-6 * sumAbs + 1e-30,
                        "WoA Tensor Core exceeds BF16 output / FP32 accumulation error bound");
                error += delta * delta; norm += exact * exact;
            }
            Require(std::sqrt(error / std::max(norm, 1e-30)) < 0.002,
                    "WoA Tensor Core FP64 oracle NRMSE exceeds 0.002");
        } else {
            Require(eagerBits == expectedBits, "WoA fallback changed the original FP32 reduction");
        }
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Require(FastllmCudaDeepSeekV4WoA(input, weight, groups, rank, output, false), "WoA capture rejected");
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, 0));
        Check(cudaMemsetAsync(output.cudaData, 0xff, output.GetBytes(), cudaStreamPerThread));
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        Require(Bits(output) == eagerBits, "WoA graph changed bits or left a token tail unwritten");
        Check(cudaGraphExecDestroy(exec));
        Check(cudaGraphDestroy(graph));
        ++cases;
    };
    // Keep every original single-token case, including its input seeds.
    for (int dim : {17, 512, 4096}) for (int groups : {1, 3}) for (int rank : {6, 1024})
        check({1, 1, groups, rank, dim}, DataType::BFLOAT16, DataType::BFLOAT16, 0.3f, 2);
    // Exercise token-tile tails, batch flattening, TP8 dimensions and mixed dtypes.
    for (auto shape : shapes)
    for (auto inputType : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32})
    for (auto weightType : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32})
    for (float amplitude : {0.f, 0.3f, 12.f})
        check(shape, inputType, weightType, amplitude, shape.dim);
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
        std::printf("PASS V4.1 decode projections: %d cases (exact fallback and FP64 Tensor Core oracle)\n", cases);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
