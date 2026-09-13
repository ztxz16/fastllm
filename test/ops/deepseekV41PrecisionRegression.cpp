// Precision boundaries used by quantized V4.1 checkpoints: FP8 activations
// and CPU MergeMOE with FP16 shared / NVFP4 routed expert weights.
#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#include "devices/cpu/computeutils.h"
#include "devices/cpu/deepseekv41-reference-math.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static int checks = 0;
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
    checks++;
}
static float Bf(float x) { return RoundFloat32ToBFloat16RNE(x); }
static std::vector<float> Read(Data &data) {
    Data copy;
    copy.CopyFrom(data);
    ToDataType(copy, DataType::FLOAT32);
    copy.ToDevice(DataDevice::CPU);
    return std::vector<float>((float *)copy.cpuData, (float *)copy.cpuData + copy.Count(0));
}
// Enumerate the E4M3FN grid independently of the production quantizer.
static float Fp8Value(int code) {
    int e = code >> 3, m = code & 7;
    return e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
}
static void Quant(std::vector<float> &values, int block) {
    for (size_t st = 0; st < values.size(); st += block) {
        size_t end = std::min(values.size(), st + block);
        float maximum = 1e-4f;
        for (size_t i = st; i < end; i++)
            maximum = std::max(maximum, std::fabs(values[i]));
        float scale = std::exp2(std::ceil(std::log2(maximum / 448.0f)));
        for (size_t i = st; i < end; i++) {
            float x = std::fabs(values[i] / scale), distance = INFINITY;
            int best = 0;
            for (int code = 0; code <= 126; code++) {
                float d = std::fabs(x - Fp8Value(code));
                if (d < distance || (d == distance && !(code & 1))) {
                    distance = d;
                    best = code;
                }
            }
            values[i] = std::copysign(Fp8Value(best) * scale, values[i]);
        }
    }
}
static void Activation(bool cuda) {
    ApplyDeviceMap({{"cpu", 1}}, 0, 1);
    for (DataType dtype : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16}) {
        for (int rows : {1, 5, 33}) {
            std::vector<float> x(rows * 128);
            for (size_t i = 0; i < x.size(); i++)
                x[i] = std::ldexp(((int)(i * 13 % 83) - 41) / 16.0f, (int)((i / 32) % 7) - 5);
            x[0] = 0.0f;
            x[1] = -0.0f;
            x[2] = 1e-7f;
            x[3] = -1e-7f;
            Data input(DataType::FLOAT32, {rows, 128}, x), output;
            ToDataType(input, dtype);
            auto expected = Read(input);
            Quant(expected, 32);
            Data expectedData(DataType::FLOAT32, {rows, 128}, expected);
            ToDataType(expectedData, dtype);
            expected = Read(expectedData);
            ApplyDeviceMap({{cuda ? "cuda" : "cpu", 1}}, 0, 1);
            ((Executor *)GetExecutor())
                ->Run("DeepSeekV41QuantizeActivation", {{"input", &input}, {"output", &output}}, {}, {});
            Check(output.dataType == dtype && output.dims == input.dims, "activation metadata changed");
            auto actual = Read(output);
            Check(actual.size() == expected.size() &&
                      !std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)),
                  "block-32 FP8 activation mismatch");
            ApplyDeviceMap({{"cpu", 1}}, 0, 1);
        }
    }
}

static void RotaryBoundary(bool cuda) {
    for (DataType dtype : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16}) {
        for (int start : {1, 129}) {
            ApplyDeviceMap({{"cpu", 1}}, 0, 1);
            std::vector<float> values(5 * 512);
            for (size_t i = 0; i < values.size(); i++)
                values[i] = ((int)(i * 37 % 199) - 99) / 64.0f;
            Data input(DataType::FLOAT32, {1, 5, 512}, values), separate;
            ToDataType(input, dtype);
            separate.CopyFrom(input);
            ApplyDeviceMap({{cuda ? "cuda" : "cpu", 1}}, 0, 1);
            auto &executor = *(Executor *)GetExecutor();
            IntDict params{{"ropeDim", 128}, {"startPos", start}, {"posStep", 1},
                           {"quantMode", 0}, {"quantDim", 512},   {"quantBlock", 32}};
            executor.Run("DeepSeekV41RotaryQuant", {{"input", &separate}}, {{"ropeBase", 10000.0f}}, params);
            auto expected = Read(separate);
            Quant(expected, 32);
            params["quantMode"] = 1;
            executor.Run("DeepSeekV41RotaryQuant", {{"input", &input}}, {{"ropeBase", 10000.0f}}, params);
            Check(Read(input) == expected, "RoPE omitted storage rounding before FP8 quantization");
        }
    }
}

static void AttentionBoundary(bool cuda) {
    constexpr int dim = 512, heads = 32, window = 128, compressed = 70;
    // Only two nonzero KV coordinates: QK is exact, isolating the softmax
    // probability cast, 64-slot boundaries, masked slots and sink placement.
    for (auto config : {std::pair<int, int>{0, 5}, {0, 65}, {0, 130}, {5, 1}, {127, 1}, {129, 3}}) {
        const int start = config.first, count = config.second;
        ApplyDeviceMap({{"cpu", 1}}, 0, 1);
        std::vector<float> query(count * heads * dim), chunk(count * dim), ring(window * dim),
            comp(compressed * dim);
        std::vector<float> sinks(heads), expected(count * heads * dim);
        auto fill = [](float *row, int position) {
            row[0] = float((position / 17) % 5 - 2) * 0.75f;
            row[1] = float(position % 7 - 3) * 0.25f;
        };
        for (int p = 0; p < start; p++)
            fill(ring.data() + (p % window) * dim, p);
        for (int i = 0; i < count; i++)
            fill(chunk.data() + i * dim, start + i);
        for (int i = 0; i < compressed; i++)
            fill(comp.data() + i * dim, i + 53);
        for (int h = 0; h < heads; h++)
            sinks[h] = (h % 3 - 1) * 3.25f;
        Data index(DataType::INT32, {1, count, compressed});
        index.Allocate();
        auto ids = (int32_t *)index.cpuData;
        for (int i = 0; i < count; i++) {
            int pos = start + i;
            std::vector<const float *> candidates;
            int slots = start == 0 ? std::min(count, window) : window;
            int first = start == 0 ? std::max(0, pos - window + 1) : pos - window + 1;
            for (int j = 0; j < slots; j++) {
                int p = first + j;
                candidates.push_back(p < 0 || p > pos ? nullptr
                                     : p >= start     ? chunk.data() + (p - start) * dim
                                                      : ring.data() + (p % window) * dim);
            }
            for (int j = 0; j < compressed; j++) {
                ids[i * compressed + j] = j % 11 == 0 ? -1 : j;
                candidates.push_back(j % 11 == 0 ? nullptr : comp.data() + j * dim);
            }
            for (int h = 0; h < heads; h++) {
                query[(i * heads + h) * dim] = 1.0f;
                double maximum = -1e30, denominator = 0, numerator[2] = {0, 0};
                for (size_t tile = 0; tile < candidates.size(); tile += 64) {
                    size_t end = std::min(candidates.size(), tile + 64);
                    double next = maximum;
                    for (size_t j = tile; j < end; j++)
                        if (candidates[j])
                            next = std::max(next, double(candidates[j][0]));
                    double correction = std::exp(maximum - next);
                    denominator *= correction;
                    numerator[0] *= correction;
                    numerator[1] *= correction;
                    for (size_t j = tile; j < end; j++)
                        if (candidates[j]) {
                            double probability = std::exp(candidates[j][0] - next);
                            denominator += probability;
                            float rounded = Bf(float(probability));
                            for (int d = 0; d < 2; d++)
                                numerator[d] += rounded * candidates[j][d];
                        }
                    maximum = next;
                }
                denominator += std::exp(sinks[h] - maximum);
                for (int d = 0; d < 2; d++)
                    expected[(i * heads + h) * dim + d] = Bf(numerator[d] / denominator);
            }
        }
        Data q(DataType::FLOAT32, {1, count, heads, dim}, query), kv(DataType::FLOAT32, {1, count, dim}, chunk);
        Data r(DataType::FLOAT32, {1, window, dim}, ring), c(DataType::FLOAT32, {1, compressed, dim}, comp);
        Data sink(DataType::FLOAT32, {heads}, sinks), output;
        for (Data *data : {&q, &kv, &r, &c})
            ToDataType(*data, DataType::BFLOAT16);
        // Real caches reserve extra rows. Count(0) includes that capacity,
        // while conversion scratch must only contain the visible logical rows.
        c.Expansion({1, compressed + 1024, dim});
        ApplyDeviceMap({{cuda ? "cuda" : "cpu", 1}}, 0, 1);
        DataDict args{{"q", &q},          {"chunkKV", &kv},    {"compressedKV", &c},
                      {"cmpIdx", &index}, {"attnSink", &sink}, {"output", &output}};
        if (start > 0)
            args["ringKV"] = &r;
        ((Executor *)GetExecutor())
            ->Run("DeepSeekV41SparseAttention", args, {{"softmaxScale", 1.0f}},
                  {{"windowSize", window}, {"startPos", start}});
        auto actual = Read(output);
        double error = 0, norm = 0;
        for (size_t j = 0; j < expected.size(); j++) {
            error += std::pow(double(actual[j]) - expected[j], 2);
            norm += double(expected[j]) * expected[j];
        }
        if (!(error <= 1e-8 * norm)) {
            std::cerr << "Attention start=" << start << " count=" << count << " cuda=" << cuda
                      << " NRMSE=" << std::sqrt(error / norm) << '\n';
        }
        Check(error <= 1e-8 * norm, "attention changed 64-slot/BF16/sink semantics");
    }
}
static Data RoutedWeight(int rows, int cols) {
    Data w(DataType::NVFP4, {rows, cols});
    w.blockK = 1;
    w.blockM = 32;
    w.Allocate(false);
    std::memset(w.cpuData, 0, GetNVFP4WeightBytes(rows, cols));
    // Every row selects one exactly representable input with coefficient 1/8.
    for (int row = 0; row < rows; row++) {
        int col = row % cols;
        w.cpuData[((size_t)row * cols + col) / 2] |= 2 << ((col & 1) * 4);
    }
    std::memset(GetNVFP4ScaleData(w), 124, GetNVFP4ScaleBytes(rows, cols, 1, 32));
    return w;
}
static std::vector<float> Expert(std::vector<float> x, bool quantized, float route, float gateCoefficient,
                                 float downCoefficient) {
    if (quantized)
        Quant(x, 32);
    std::vector<float> act(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        float gate = Bf(x[i] * gateCoefficient), up = gate;
        gate = std::min(gate, 10.0f);
        up = std::clamp(up, -10.0f, 10.0f);
        act[i] = Bf(route * ((gate / (1.0f + std::exp(-gate))) * up));
    }
    if (quantized)
        Quant(act, 32);
    for (float &value : act)
        value = Bf(value * downCoefficient);
    return act;
}
static void MixedMoe() {
    ApplyDeviceMap({{"cpu", 1}}, 0, 1);
    constexpr int dim = 128;
    Data routedGate = RoutedWeight(2 * dim, dim), routedDown = RoutedWeight(dim, dim);
    std::vector<float> g(2 * dim * dim, 0.0f), d(dim * dim, 0.0f);
    for (int row = 0; row < 2 * dim; row++)
        g[row * dim + row % dim] = 0.5f;
    for (int row = 0; row < dim; row++)
        d[row * dim + row] = 0.25f;
    Data sharedGate(DataType::FLOAT32, {2 * dim, dim}, g), sharedDown(DataType::FLOAT32, {dim, dim}, d);
    ToDataType(sharedGate, DataType::FLOAT16);
    ToDataType(sharedDown, DataType::FLOAT16);
    std::vector<Data *> weights{&sharedGate, &sharedDown, &routedGate, &routedDown}, biases(4, nullptr);
    for (int rows : {1, 5, 33})
        for (bool quantShared : {false, true})
            for (float score : {0.0f, 0.75f}) {
                std::vector<float> x(rows * dim), scores(rows, score), expected(rows * dim);
                for (size_t i = 0; i < x.size(); i++)
                    x[i] = Bf(std::ldexp(((int)(i * 17 % 79) - 39) / 8.0f, (int)((i / 32) % 4) - 4));
                x[0] = 48.0f;
                x[1] = -64.0f;
                x[32] = 40.0f;
                x[64] = -48.0f;
                for (int row = 0; row < rows; row++) {
                    std::vector<float> v(x.begin() + row * dim, x.begin() + (row + 1) * dim);
                    auto routed = Expert(v, true, score, 0.125f, 0.125f);
                    auto shared = Expert(v, quantShared, 1.0f, 0.5f, 0.25f);
                    for (int col = 0; col < dim; col++)
                        expected[row * dim + col] = Bf(routed[col] + shared[col]);
                }
                Data input(DataType::FLOAT32, {rows, dim}, x), index(DataType::INT32, {rows, 1});
                ToDataType(input, DataType::BFLOAT16);
                index.Allocate();
                std::memset(index.cpuData, 0, rows * sizeof(int));
                Data scoresData(DataType::FLOAT32, {rows, 1}, scores), output, w1, w2, w3, tempInput,
                    tempOutput;
                // Repeat across different shapes/scores to catch stale shared workspace.
                for (int repeat = 0; repeat < 2; repeat++) {
                    MergeMOE(input, index, scoresData, weights, biases, w1, w2, w3, tempInput, tempOutput, 1.0f,
                             output, 0, MoeGateSwiglu, false, 10.0f, true, nullptr, 32, quantShared);
                    auto actual = Read(output);
                    Check(actual == expected, "mixed FP16 shared / NVFP4 routed MergeMOE mismatch");
                }
            }
}
static void HcFloat32Boundary(bool cuda) {
    ApplyDeviceMap({{cuda ? "cuda" : "cpu", 1}}, 0, 1);
    for (DataType dtype : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16}) {
        for (int rows : {1, 5, 33}) {
            constexpr int hc = 4, dim = 128;
            std::vector<float> x(rows * dim), residual(rows * hc * dim), post(rows * hc, 0.5f),
                comb(rows * hc * hc);
            for (size_t i = 0; i < x.size(); i++)
                x[i] = ((int)(i % 31) - 15) * 0.125f;
            for (int t = 0; t < rows; t++)
                for (int h = 0; h < hc; h++) {
                    for (int d = 0; d < dim; d++)
                        residual[(t * hc + h) * dim + d] = h == 0   ? 4096.0f
                                                           : h == 1 ? 1.0f
                                                           : h == 2 ? -4096.0f
                                                                    : 0.0f;
                    for (int target = 0; target < hc; target++)
                        comb[(t * hc + h) * hc + target] = h == 1 ? 1.0f : 4096.0f;
                }
            Data input(DataType::FLOAT32, {1, rows, dim}, x),
                r(DataType::FLOAT32, {1, rows, hc, dim}, residual);
            ToDataType(input, dtype);
            ToDataType(r, dtype);
            Data p(DataType::FLOAT32, {1, rows, hc}, post), c(DataType::FLOAT32, {1, rows, hc, hc}, comb),
                output;
            std::vector<float> expected(rows * hc * dim);
            // FP32: (2^24 + 1) - 2^24 = 0; an FP64 contraction gives 1.
            for (int t = 0; t < rows; t++)
                for (int h = 0; h < hc; h++)
                    for (int d = 0; d < dim; d++)
                        expected[(t * hc + h) * dim + d] = 0.5f * x[t * dim + d];
            ((Executor *)GetExecutor())
                ->Run("DeepSeekV41HcPost",
                      {{"input", &input}, {"residual", &r}, {"post", &p}, {"comb", &c}, {"output", &output}},
                      {}, {});
            Check(Read(output) == expected, "V4.1 hc_post lost an FP32 arithmetic boundary");
            // The kernel must retain all residual copies before writing any target.
            ApplyDeviceMap({{cuda ? "cuda" : "cpu", 1}}, 0, 1);
            ((Executor *)GetExecutor())
                ->Run("DeepSeekV41HcPost",
                      {{"input", &input}, {"residual", &r}, {"post", &p}, {"comb", &c}, {"output", &r}}, {},
                      {});
            auto inplace = Read(r);
            if (inplace != expected) {
                std::cerr << "inplace cuda=" << cuda << " dtype=" << dtype << " rows=" << rows
                          << " first=" << inplace[0] << " expected=" << expected[0] << "\n";
            }
            Check(inplace == expected, "V4.1 hc_post in-place residual mismatch");
            ApplyDeviceMap({{cuda ? "cuda" : "cpu", 1}}, 0, 1);
        }
    }
}
#ifdef USE_CUDA
extern "C" bool FastllmCudaDeepSeekV41LinearBlock32(const Data &, Data &, Data &);
static void BlockLinearBoundary() {
    ApplyDeviceMap({{"cuda", 1}}, 0, 1);
    for (DataType dtype : {DataType::FLOAT16, DataType::BFLOAT16, DataType::FLOAT32}) {
        for (int rows : {1, 5, 33})
            for (int k : {64, 1280}) {
                constexpr int n = 7;
                std::vector<float> x(rows * k), w(n * k), expected(rows * n);
                for (int t = 0; t < rows; t++)
                    for (int d = 0; d < k; d++)
                        x[t * k + d] = std::ldexp(((t * 13 + d * 7) % 31 - 15) / 8.0f, (d / 32) % 9 - 4);
                for (int r = 0; r < n; r++)
                    for (int d = 0; d < k; d++)
                        w[r * k + d] = std::ldexp(((r * 11 + d * 17) % 29 - 14) / 8.0f, (r + d / 32) % 9 - 4);
                // The block product is exact on this E4M3 grid. Round the block
                // result to FP32, then add it to the FP32 running accumulator.
                for (int t = 0; t < rows; t++)
                    for (int r = 0; r < n; r++) {
                        float total = 0.0f;
                        for (int block = 0; block < k; block += 32) {
                            double part = 0.0;
                            for (int d = block; d < block + 32; d++)
                                part += (double)x[t * k + d] * w[r * k + d];
                            total = total + (float)part;
                        }
                        expected[t * n + r] = Bf(total);
                    }
                Data input(DataType::FLOAT32, {1, rows, k}, x), weight(DataType::FLOAT32, {n, k}, w), output;
                ToDataType(input, DataType::BFLOAT16);
                ToDataType(weight, dtype);
                input.ToDevice(DataDevice::CUDA);
                Check(FastllmCudaDeepSeekV41LinearBlock32(input, weight, output),
                      "block-32 linear refused valid tensors");
                Check(output.dims == std::vector<int>({1, rows, n}) && output.dataType == DataType::BFLOAT16,
                      "block-32 linear changed output metadata");
                Check(Read(output) == expected, "block-32 linear accumulator boundary mismatch");
            }
    }
}
#endif
static void ReferenceFp8BlockBoundary() {
    if (!V41ReferenceMathEnabled())
        return;
    // Three exact block dots: 2^24, 1, -2^24. Ordered FP32 block
    // accumulation is zero; accumulating separate SIMD lanes gives one.
    for (int count : {1, 5}) {
        std::vector<uint16_t> input(count * 96, Float32ToBFloat16RNEBits(1.0f));
        std::vector<uint8_t> weight(32 * 96, 0);
        for (int row = 0; row < 32; ++row) {
            weight[row * 96] = 56;
            weight[row * 96 + 33] = 56;
            weight[row * 96 + 64] = 184;
        }
        float scales[3] = {16777216.0f, 1.0f, 16777216.0f};
        std::vector<float> output(count * 32, 42.0f);
        MultiThreadLinearBFloat16FP8E4M3Op(input.data(), weight.data(), nullptr, output.data(), count, 96, 32,
                                           0, 32, scales, 32, 32)
            .Run();
        Check(std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; }),
              "reference FP8 linear changed block accumulation order");
    }
}
int main(int argc, char **argv) {
    try {
        SetThreads(30);
        Activation(false);
        RotaryBoundary(false);
        AttentionBoundary(false);
        HcFloat32Boundary(false);
#ifdef USE_CUDA
        if (argc > 1 && std::string(argv[1]) == "--cuda") {
            Activation(true);
            RotaryBoundary(true);
            AttentionBoundary(true);
            BlockLinearBoundary();
            HcFloat32Boundary(true);
        }
#endif
        ReferenceFp8BlockBoundary();
        MixedMoe();
        std::cout << "V4.1 precision regression PASS: " << checks << " checks\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
