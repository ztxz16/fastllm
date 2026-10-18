// Precision boundaries used by quantized V4.1 checkpoints: FP8 activations
// and CPU MergeMOE with FP16 shared / NVFP4 routed expert weights.
#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#include "devices/cpu/computeutils.h"
#include "devices/cpu/deepseekv41-reference-math.h"
#include "models/deepseekv41.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef __SSE2__
#include <immintrin.h>
#endif

using namespace fastllm;
static int checks = 0;
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
    checks++;
}
static void ExportedActivationMetadata() {
    class Probe : public DeepSeekV41Model {
    public:
        bool Quantized(const std::string &name) const {
            return quantizedLinearNames.count(name) != 0;
        }
    };
    const std::string query = "layers.0.attn.wq_a.weight";
    const std::string expert = "layers.0.ffn.experts.0.w1.weight";
    Probe original;
    original.GetTensorMap({query, expert});
    Check(!original.Quantized(query) && !original.Quantized(expert),
          "unquantized source acquired activation quantization");
    Probe exported;
    exported.weight.AddDict("fastllm_activation_quantized_linears",
                           "[\"" + query + "\",\"" + expert + "\"]");
    exported.GetTensorMap({query, expert, "layers.0.attn.wq_b.weight"});
    Check(exported.Quantized(query) && exported.Quantized(expert),
          "export lost source activation quantization without scale tensors");
    Check(exported.Quantized("layers.0.ffn.experts.0.gateup.weight"),
          "export lost fused expert activation quantization");
    Check(!exported.Quantized("layers.0.attn.wq_b.weight"),
          "export metadata quantized an unrelated linear");
    exported.GetTensorMap({"layers.0.attn.wq_b.weight", "layers.0.attn.wq_b.scale"});
    Check(exported.Quantized("layers.0.attn.wq_b.weight"),
          "export metadata prevented source scale detection");
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
        const std::vector<int> rowsToCheck = dtype == DataType::BFLOAT16 ?
            std::vector<int>{1, 5, 33, 2047, 2048, 2049} : std::vector<int>{1, 5, 33};
        for (int rows : rowsToCheck) {
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
static void MixedMoe(bool numa = false) {
    ApplyDeviceMap({{numa ? "numa" : "cpu", 1}}, 0, 1);
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
#ifdef USE_CUDA
static void NativeFp8ScaleRange() {
    ApplyDeviceMap({{"cuda", 1}}, 0, 1);
    constexpr int dim = 256, channels = 32;
    for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 11, 31, 32, 33}) {
        for (int scaleExponent : {-16, -12, -8, 0, 12}) {
            Data weight(DataType::FP8_E4M3, {channels, dim});
            weight.blockK = 32;
            weight.blockM = 32;
            weight.scales.assign(dim / 32, std::ldexp(1.0f, scaleExponent));
            weight.Allocate();
            std::memset(weight.cpuData, 0x21, channels * dim); // E4M3FN 0.140625
            std::vector<float> x(rows * dim);
            for (size_t i = 0; i < x.size(); ++i) x[i] = Bf((1.0f + (i % 13) / 8.0f) / 1024.0f);
            Data input(DataType::FLOAT32, {rows, dim}, x), output;
            ToDataType(input, DataType::BFLOAT16);
            Linear(input, weight, *GetEmptyData(), output);
            auto actual = Read(output);
            for (int row = 0; row < rows; ++row) {
                double sum = 0;
                for (int c = 0; c < dim; ++c) sum += x[row * dim + c] * 0.140625 * std::ldexp(1.0, scaleExponent);
                for (int c = 0; c < channels; ++c) {
                    if (actual[row * channels + c] != Bf(sum))
                        std::cerr << "FP8 scale range rows=" << rows << " exponent=" << scaleExponent
                                  << " actual=" << actual[row * channels + c] << " expected=" << Bf(sum) << '\n';
                    Check(actual[row * channels + c] == Bf(sum), "FP8 GEMM lost exponent range in intermediate values");
                }
            }
        }
    }
}
static void NativeLargeExpertRouter() {
    ApplyDeviceMap({{"cuda", 1}}, 0, 1);
    constexpr int topk = 6;
    for (int experts : {256, 384, 768}) for (int rows : {1, 17}) {
        std::vector<float> raw(rows * experts), bias(experts), expected(rows * topk);
        std::vector<int32_t> selected(rows * topk);
        for (int e=0; e<experts; ++e) bias[e] = (e % 7) * 0.0001f;
        for (int row=0; row<rows; ++row) {
            std::vector<std::pair<float,int>> ranked;
            for (int e=0; e<experts; ++e) {
                float v = float((e * 37 + row * 13) % experts) / 19.0f - 12.0f;
                raw[row * experts + e] = v;
                float transformed = std::sqrt(v > 20.0f ? v : std::log1p(std::exp(v)));
                ranked.emplace_back(transformed+bias[e],e);
            }
            std::sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.first>b.first;});
            float sum=0;
            for(int j=0;j<topk;++j) {
                int e=ranked[j].second;selected[row*topk+j]=e;
                float v=raw[row*experts+e];
                expected[row*topk+j]=std::sqrt(v>20.0f?v:std::log1p(std::exp(v)));
                sum+=expected[row*topk+j];
            }
            for(int j=0;j<topk;++j) expected[row*topk+j] *= 2.5f / sum;
        }
        Data logits(DataType::FLOAT32,{rows,experts},raw);
        Data gateBias(DataType::FLOAT32,{experts},bias);
        Data ids(DataType::INT32,{rows,topk}), scores(DataType::FLOAT32,{rows,topk});
        for(Data *d:{&logits,&gateBias}) d->ToDevice(DataDevice::CUDA,{0},true);
        for(Data *d:{&ids,&scores}) {d->ToDevice(DataDevice::CUDA,{0},false);d->Allocate(false);}
        Check(FastllmCudaDeepSeekV4RouteScoreTransform(logits,2),"CUDA route transform rejected a large expert count");
        Check(FastllmCudaSelectExpert(logits,&gateBias,ids,scores,topk,true,2.5f),"CUDA top-k rejected routes");
        Data hostIds;
        hostIds.CopyFrom(ids);
        hostIds.ToDevice(DataDevice::CPU);
        auto actualIds=(const int32_t *)hostIds.cpuData;
        auto actualScores=Read(scores);
        for(int i=0;i<rows*topk;++i) {
            Check(actualIds[i]==selected[i],"CUDA router selected a different expert");
            Check(std::abs(actualScores[i]-expected[i])<2e-6f,"CUDA router weight normalization mismatch");
        }
    }
}

static void NativeMxfp4Moe() {
    ApplyDeviceMap({{"cuda", 1}}, 0, 1);
    constexpr int dim = 256;
    Data gate = RoutedWeight(2 * dim, dim), down = RoutedWeight(dim, dim);
    for (Data *w : {&gate, &down}) {
        w->directMemory = true;
        w->ToDevice(DataDevice::CUDA, {0}, true);
    }
    std::vector<Data *> weights{nullptr, nullptr, &gate, &down};
    Data scratch, output;
    for (int rows : {1, 2, 8, 11, 33, 127, 128, 129, 257, 1}) {
        std::vector<float> x(rows * dim), scores(rows), expected(rows * dim);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = Bf(std::ldexp(float(int(i * 17 % 79) - 39) / 7.0f, int(i / 32 % 4) - 2));
        for (int r = 0; r < rows; ++r) {
            x[r * dim] = 128.0f;
            x[r * dim + 1] = -128.0f;
            scores[r] = r % 3 == 0 ? 0.34f : (r % 3 == 1 ? 0.0f : 1.75f);
            std::vector<float> v(x.begin() + r * dim, x.begin() + (r + 1) * dim);
            const auto e = Expert(v, true, scores[r], 0.125f, 0.125f);
            std::copy(e.begin(), e.end(), expected.begin() + r * dim);
        }
        Data input(DataType::FLOAT32, {rows, dim}, x);
        ToDataType(input, DataType::BFLOAT16);
        Data index(DataType::INT32, {rows, 1});
        index.Allocate();
        std::memset(index.cpuData, 0, rows * sizeof(int32_t));
        Data score(DataType::FLOAT32, {rows, 1}, scores);
        for (Data *d : {&input, &index, &score}) d->ToDevice(DataDevice::CUDA, {0}, true);
        Check(Read(score) == scores, "native MoE score upload failed");
        Check(FastllmCudaBFloat16MergeMOEVllmMarlinBatch1ExpertParallel(
            input, scratch, output, weights.data(), weights.size(),
            (const int32_t *)index.cudaData, (const float *)score.cudaData,
            1, 0, 1, true, 10.0f), "V4.1 native MXFP4 MoE rejected input");
        const auto actual = Read(output);
        if (actual != expected) {
            auto si = Read(input), sa = Read(scratch), ss = Read(score);
            std::cerr << "input=" << si[0] << " activation=" << sa[0] << " score=" << ss[0] << '\n';
            int shown = 0;
            for (size_t j = 0; j < actual.size() && shown < 8; ++j) {
                if (actual[j] != expected[j]) {
                    std::cerr << "native MoE rows=" << rows << " at=" << j
                              << " actual=" << actual[j] << " expected=" << expected[j] << '\n';
                    ++shown;
                }
            }
        }
        Check(actual == expected,
              "V4.1 native MXFP4 changed activation quantization, clipping or pre-down router weighting");
    }
}
static void NativeMxfp4Accumulation() {
    ApplyDeviceMap({{"cuda", 1}}, 0, 1);
    constexpr int dim = 256, topk = 2;
    Data g0 = RoutedWeight(2 * dim, dim), d0 = RoutedWeight(dim, dim);
    Data g1 = RoutedWeight(2 * dim, dim), d1 = RoutedWeight(dim, dim);
    for (Data *w : {&g0, &d0, &g1, &d1}) {
        w->directMemory = true;
        w->ToDevice(DataDevice::CUDA, {0}, true);
    }
    std::vector<Data *> weights{nullptr, nullptr, &g0, &d0, &g1, &d1};
    Data fg0 = RoutedWeight(2 * dim, dim), fd0 = RoutedWeight(dim, dim);
    Data fg1 = RoutedWeight(2 * dim, dim), fd1 = RoutedWeight(dim, dim);
    for (Data *w : {&fg0, &fd0, &fg1, &fd1}) w->ToDevice(DataDevice::CUDA, {0}, true);
    std::vector<Data *> fallbackWeights{nullptr, nullptr, &fg0, &fd0, &fg1, &fd1};
    Data scratch, native, fallback, w2, w3;
    for (auto ownership : std::vector<std::pair<int, int>>{{0, 1}, {0, 2}, {1, 2}, {7, 8}})
    for (int rows : {1, 2, 8, 11, 16, 33, 127, 128, 129, 255, 256, 257}) {
        const int ownerRank = ownership.first, ownerCount = ownership.second;
        std::vector<float> x(rows * dim), scores(rows * topk), expected(rows * dim);
        for (size_t i = 0; i < x.size(); ++i) x[i] = Bf(float(int(i * 17 % 79) - 39) / 7.0f);
        for (int row = 0; row < rows; ++row) {
            scores[row * topk] = 0.34f;
            scores[row * topk + 1] = row % 2 ? 1.75f : -0.17f;
            std::vector<float> v(x.begin() + row * dim, x.begin() + (row + 1) * dim);
            auto a = Expert(v, true, scores[row * topk], 0.125f, 0.125f);
            auto b = Expert(v, true, scores[row * topk + 1], 0.125f, 0.125f);
            for (int c = 0; c < dim; ++c)
                expected[row * dim + c] = (ownerCount == 1 || row % 3 != 0 ? a[c] : 0.0f) +
                                         (ownerCount == 1 || row % 3 != 1 ? b[c] : 0.0f);
        }
        Data input(DataType::FLOAT32, {rows, dim}, x);
        ToDataType(input, DataType::BFLOAT16);
        Data index(DataType::INT32, {rows, topk});
        index.Allocate();
        for (int i = 0; i < rows * topk; ++i) {
            const int rank = (i / topk) % 3 == i % topk ? (ownerRank + 1) % ownerCount : ownerRank;
            ((int32_t *)index.cpuData)[i] = (i % topk) * ownerCount + rank;
        }
        Data score(DataType::FLOAT32, {rows, topk}, scores);
        for (Data *d : {&input, &index, &score}) d->ToDevice(DataDevice::CUDA, {0}, true);
        Check(FastllmCudaBFloat16MergeMOEVllmMarlinBatch1ExpertParallel(
            input, scratch, native, weights.data(), weights.size(),
            (const int32_t *)index.cudaData, (const float *)score.cudaData,
            topk, ownerRank, ownerCount, true, 10.0f), "V4.1 multi-expert native MoE rejected input");
        Check(native.dataType == DataType::FLOAT32 && Read(native) == expected,
              "V4.1 native MoE rounded the expert sum before the collective");
        if (ownerCount == 1) {
            DoCudaMergeMOE(input, fallback, index, score, scratch, w2, w3,
                fallbackWeights.data(), nullptr, 0.0f, MoeGateSwiglu, fallbackWeights.size(), 10.0f, true);
            Check(fallback.dataType == DataType::FLOAT32 && Read(fallback) == expected,
                  "V4.1 general CUDA MoE changed the original arithmetic boundaries");
        }
    }
}

#endif
static void NVFP4TokenReuseOrder() {
    if (V41ReferenceMathEnabled()) return;
    for (auto shape : std::vector<std::pair<int,int>>{{32,19},{96,37},{5120,129},{2304,267}})
    for (int n : {2,3,4,5,6,7,8,9}) for (int blockK : {1,32}) for (bool compact : {false,true}) {
        const int m=shape.first,k=shape.second,st=compact?3:0,end=compact?k-1:k;
        std::vector<uint16_t> input(size_t(n)*m);
        std::vector<uint8_t> weight(size_t(k)*m/2),scaleBytes(size_t((k+blockK-1)/blockK)*(m/32));
        std::vector<float> scales(scaleBytes.size()),bias(k),actual(size_t(n)*k,-777),expected(actual);
        uint32_t rng=789123;
        for (auto &value : weight) {rng=rng*1664525u+1013904223u;value=uint8_t(rng>>24);}
        for (size_t i=0;i<input.size();++i)
            input[i]=Float32ToBFloat16RNEBits(std::ldexp(float(int(i*17%113)-56)/37.0f,int(i%9)-4));
        for (size_t i=0;i<scales.size();++i) {
            scaleBytes[i]=uint8_t(120+i%11);
            scales[i]=std::ldexp(1.0f+float(i%7)/16.0f,int(i%9)-5);
        }
        for (int i=0;i<k;++i)bias[i]=float(i%13-6)/32;
        auto run=[&](int rows,uint16_t *in,float *out) {
            MultiThreadLinearBFloat16NVFP4Op(in,weight.data(),compact?nullptr:bias.data(),out,
                rows,m,k,st,end,compact?nullptr:scales.data(),compact?scaleBytes.data():nullptr,blockK,32).Run();
        };
        run(n,input.data(),actual.data());
        for (int row=0;row<n;++row)run(1,input.data()+size_t(row)*m,expected.data()+size_t(row)*k);
        Check(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,
              "NVFP4 token reuse changed raw FP32 output, scale rows, bias or output tails");
        // Exercise both reasons to keep the original magic-scale operation:
        // extreme inputs and scales whose combined power of two would overflow.
        if (m == 96 && n == 3 && blockK == 32) {
            for (int mode = 0; mode < 3; ++mode) {
                if (mode == 0) input[0] = Float32ToBFloat16RNEBits(0x1p124f);
                if (mode == 1) input[1] = 0x7f80; // infinity
                if (mode == 2) {
                    input[0] = input[1] = Float32ToBFloat16RNEBits(1.0f);
                    scaleBytes[0] = 240; scales[0] = 0x1p100f;
                }
                run(n,input.data(),actual.data());
                for (int row=0;row<n;++row)run(1,input.data()+size_t(row)*m,expected.data()+size_t(row)*k);
                Check(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,
                      "NVFP4 combined scale changed overflow or nonfinite behavior");
            }
#ifdef __SSE2__
            input[0] = input[1] = Float32ToBFloat16RNEBits(1.0f);
            std::fill(scaleBytes.begin(),scaleBytes.end(),0);
            std::fill(scales.begin(),scales.end(),0x1p-127f);
            const unsigned int savedCsr = _mm_getcsr();
            for (unsigned int daz : {0u,0x40u}) {
                _mm_setcsr((savedCsr & ~0x40u) | daz);
                run(n,input.data(),actual.data());
                for (int row=0;row<n;++row)run(1,input.data()+size_t(row)*m,expected.data()+size_t(row)*k);
                _mm_setcsr(savedCsr);
                Check(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,
                      "NVFP4 combined scale changed subnormal/DAZ behavior");
            }
#endif
        }
    }
}

static void ReferenceMoeBatchOrder() {
    ApplyDeviceMap({{"cpu",1}},0,1);
    constexpr int dim=96, topk=3;
    Data gate0=RoutedWeight(dim*2,dim),down0=RoutedWeight(dim,dim);
    Data gate1=RoutedWeight(dim*2,dim),down1=RoutedWeight(dim,dim);
    Data sharedGate(DataType::FP8_E4M3,{dim*2,dim}),sharedDown(DataType::FP8_E4M3,{dim,dim});
    for (Data *w : {&sharedGate,&sharedDown}) {
        w->blockK=w->blockM=32;w->Allocate(false);
        w->scales.resize((w->dims[0]/32)*(dim/32),0.0625f);
        for (int i=0;i<w->dims[0]*dim;++i) w->cpuData[i]=uint8_t((i*17)%126)|((i%3)?128:0);
    }
    for (int rows : {2,5,33,257}) for (bool hasShared : {false,true}) for (bool quantShared : {false,true}) {
        std::vector<Data*> weights{hasShared?&sharedGate:nullptr,hasShared?&sharedDown:nullptr,&gate0,&down0,&gate1,&down1},bias(6,nullptr);
        std::vector<float> x(rows*dim),s(rows*topk);
        std::vector<int> ids(rows*topk);
        for (int i=0;i<rows*dim;++i) x[i]=Bf(std::ldexp(float(i*17%79-39)/9,int(i%9)-4));
        for (int i=0;i<rows*topk;++i) { ids[i]=(i%3)==1?0:1;s[i]=float(i%7)/8; }
        Data input(DataType::FLOAT32,{rows,dim},x),index(DataType::INT32,{rows,topk}),score(DataType::FLOAT32,{rows,topk},s),output,w1,w2,w3,t1,t2;
        if (quantShared) ToDataType(input,DataType::BFLOAT16);
        index.Allocate();memcpy(index.cpuData,ids.data(),ids.size()*sizeof(int));
        MergeMOE(input,index,score,weights,bias,w1,w2,w3,t1,t2,1.0f,output,0,MoeGateSwiglu,false,10.0f,true,nullptr,32,quantShared);
        auto actual=Read(output);
        // Singleton calls exercise the established expert scheduling path.
        // Unsorted, repeated expert IDs expose incorrect gather/reduce maps.
        std::vector<float> expected;
        for (int row=0;row<rows;++row) {
            Data in(DataType::FLOAT32,{1,dim},std::vector<float>(x.begin()+row*dim,x.begin()+(row+1)*dim));
            if (quantShared) ToDataType(in,DataType::BFLOAT16);
            Data ix(DataType::INT32,{1,topk}),sc(DataType::FLOAT32,{1,topk},std::vector<float>(s.begin()+row*topk,s.begin()+(row+1)*topk)),out;
            ix.Allocate();memcpy(ix.cpuData,ids.data()+row*topk,topk*sizeof(int));
            MergeMOE(in,ix,sc,weights,bias,w1,w2,w3,t1,t2,1.0f,out,0,MoeGateSwiglu,false,10.0f,true,nullptr,32,quantShared);
            auto values=Read(out);expected.insert(expected.end(),values.begin(),values.end());
        }
        Check(actual==expected,"batched MoE changed input quantization, route ordering or shared expert");
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
#include "devices/cuda/deepseekv41-reference.cuh"
static void ParallelReferenceBoundary() {
    if (!V41ReferenceMathEnabled()) return;
    // BF16 BRGEMM uses high-pair-first FMA and ordered 512/1024-K blocks.
    // Cancellation across blocks makes tree reductions observably different.
    for (int cols : {2,514,1024,1026,4096}) for (int rows : {1,5}) for (int groups : {1,3}) {
        const int out = groups*7;
        std::vector<float> x(rows*cols*groups),w(out*cols),expected(rows*out);
        for (size_t i=0;i<x.size();++i) x[i]=Bf(std::ldexp(float(int(i*17%131)-65)/41,int(i%13)-6));
        for (size_t i=0;i<w.size();++i) w[i]=Bf(std::ldexp(float(int(i*13%137)-68)/79,int(i%9)-4));
        for (int token=0;token<rows;++token) for (int row=0;row<out;++row) {
            int offset=token*cols*groups+(row/7)*cols;
            int block=rows>1 && cols>1024 ? 1024 : 512;
            float total=0;
            for (int b=0;b<cols;b+=block) {
                float partial=0;
                for (int k=b;k<std::min(b+block,cols);k+=2) {
                    partial=std::fma(x[offset+k+1],w[row*cols+k+1],partial);
                    partial=std::fma(x[offset+k],w[row*cols+k],partial);
                }
                total+=partial;
            }
            expected[token*out+row]=Bf(total);
        }
        Data input(DataType::FLOAT32,{rows,cols*groups},x),weight(DataType::FLOAT32,{out,cols},w),output;
        ToDataType(input,DataType::BFLOAT16);input.ToDevice(DataDevice::CUDA);
        ToDataType(weight,DataType::BFLOAT16);
        Check(FastllmCudaV41ReferenceLinear(input,weight,output,groups,true),"parallel BF16 reference linear refused shape");
        Check(Read(output)==expected,"parallel BF16 reference linear changed pair, block or group ordering");
    }
    for (int cols : {7,33,513,20480}) for (int rows : {1,5}) {
        std::vector<float> x(rows*cols),w(cols),expected(rows*cols);
        for (int i=0;i<rows*cols;++i) x[i]=Bf(float((i*17)%101-50)/31);
        for (int i=0;i<cols;++i) w[i]=float((i*11)%31+1)/17;
        for (int row=0;row<rows;++row) {
            float sum=V41ReferenceSum(cols,[&](size_t j){return x[row*cols+j]*x[row*cols+j];});
            float inv=1.0f/std::sqrt(sum/cols+1e-6f);
            for (int j=0;j<cols;++j) expected[row*cols+j]=Bf((x[row*cols+j]*inv)*w[j]);
        }
        Data input(DataType::FLOAT32,{rows,cols},x),weight(DataType::FLOAT32,{cols},w),output;
        ToDataType(input,DataType::BFLOAT16);input.ToDevice(DataDevice::CUDA);
        Check(FastllmCudaV41ReferenceNorm(input,weight,1e-6f,output),"parallel reference norm refused shape");
        Check(Read(output)==expected,"parallel reference norm changed cascade or tail order");
    }
    for (int cols : {7,193,20480}) for (int rows : {1,2,5}) for (int out : {7,24,57}) {
        std::vector<float> x(rows*cols),w(out*cols),expected(rows*out);
        for (int i=0;i<rows*cols;++i) x[i]=Bf(float((i*17)%131-65)/41);
        for (int i=0;i<out*cols;++i) w[i]=float((i*13)%137-68)/79;
        for (int token=0;token<rows;++token) for (int row=0;row<out;++row) {
            float total=0;
            if (row<out/16*16 && token<rows/2*2) {
                for (int b=0;b<cols;b+=192) {
                    float partial=0;
                    for (int j=b;j<std::min(b+192,cols);++j) partial=std::fma(x[token*cols+j],w[row*cols+j],partial);
                    total+=partial;
                }
            } else {
                float partial[4]={};int prefix=rows==1?std::min(4,cols):0;
                for (int j=0;j<cols;++j) {
                    volatile float product=x[token*cols+j]*w[row*cols+j];
                    partial[j<prefix?0:j%4]+=product;
                }
                total=(partial[0]+partial[2])+(partial[1]+partial[3]);
            }
            expected[token*out+row]=total;
        }
        Data input(DataType::FLOAT32,{rows,cols},x),weight(DataType::FLOAT32,{out,cols},w),output;
        input.ToDevice(DataDevice::CUDA);
        Check(FastllmCudaV41ReferenceLinear(input,weight,output,1,false),"parallel reference linear refused shape");
        Check(Read(output)==expected,"parallel reference linear changed FMA blocks, fringe or tail order");
    }
}
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
static void ReferenceQuantizedLinearOrder() {
    if (!V41ReferenceMathEnabled()) return;
    const float grid[16] = {0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
    FP8E4M3ToFP32Manager fp8;
    uint32_t random = 712727;
    auto next = [&]() { random ^= random << 13; random ^= random >> 17; random ^= random << 5; return random; };
    for (int m : {96, 5120}) for (int n : {1, 5}) for (int format = 0; format < 5; ++format) {
        const bool fp8Weight = format == 0, floatInput = format >= 3;
        const bool compactScale = format == 2 || format == 4;
        const int k = 43, st = 3, end = compactScale ? k : 40, blockK = fp8Weight ? 32 : 1, blocks = m / 32;
        std::vector<float> input(n*m), bias(k), output(n*k, 42), expected(n*k, 42);
        std::vector<uint16_t> bf16(n*m);
        std::vector<uint8_t> weight(k*(fp8Weight ? m : m/2));
        std::vector<float> scales(((k+blockK-1)/blockK)*blocks);
        std::vector<uint8_t> scaleBytes(scales.size());
        for (int i = 0; i < n*m; ++i) {
            float v = std::ldexp(float(int(next()%2001)-1000)/997.0f, int(next()%33)-16);
            bf16[i] = Float32ToBFloat16RNEBits(v);
            input[i] = floatInput ? v : BFloat16BitsToFloat32(bf16[i]);
        }
        for (auto &v : weight) { v = next()%256; if (fp8Weight && (v&127)==127) --v; }
        for (auto &v : bias) v = float(int(next()%101)-50)/17.0f;
        for (size_t i = 0; i < scales.size(); ++i) {
            scaleBytes[i] = 112 + next()%31;
            scales[i] = NVFP4E8M0ScaleToFloat(scaleBytes[i]);
            if (!compactScale) scales[i] *= float(101 + next()%997)/101.0f;
        }
        for (int token = 0; token < n; ++token) for (int row = st; row < end; ++row) {
            float total = 0;
            for (int block = 0; block < blocks; ++block) {
                float partial = 0;
                for (int j = 0; j < 32; ++j) {
                    int col = block*32+j;
                    float w = fp8Weight ? fp8.dict[weight[row*m+col]]
                        : grid[(weight[row*(m/2)+col/2] >> ((col&1)*4))&15];
                    volatile float product = input[token*m+col]*w;
                    partial += product;
                }
                volatile float scaled = partial*scales[(row/blockK)*blocks+block];
                total += scaled;
            }
            expected[token*k+row] = total+bias[row];
        }
        float *scale = compactScale ? nullptr : scales.data();
        uint8_t *bytes = compactScale ? scaleBytes.data() : nullptr;
        if (fp8Weight)
            MultiThreadLinearBFloat16FP8E4M3Op(bf16.data(),weight.data(),bias.data(),output.data(),n,m,k,st,end,scale,blockK,32).Run();
        else if (floatInput)
            MultiThreadLinearFloat32NVFP4Op(input.data(),weight.data(),bias.data(),output.data(),n,m,k,st,end,scale,bytes,blockK,32).Run();
        else
            MultiThreadLinearBFloat16NVFP4Op(bf16.data(),weight.data(),bias.data(),output.data(),n,m,k,st,end,scale,bytes,blockK,32).Run();
        Check(std::memcmp(output.data(), expected.data(), output.size()*sizeof(float)) == 0,
              "reference quantized SIMD changed FP32 product/order, scale rounding, bias or output tail");
    }
}
#ifdef USE_CUDA
static void HcDecodeOrder() {
    if (V41ReferenceMathEnabled()) return;
    for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16})
    for (int dim : {31, 256, 5120})
    for (int iterations : {1, 20}) {
        ApplyDeviceMap({{"cpu", 1}}, 0, 1);
        std::vector<float> x(4 * dim), fn(24 * 4 * dim), base(24);
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i * 17 % 101) - 50) / 32.0f;
        for (size_t i = 0; i < fn.size(); ++i) fn[i] = float(int(i * 31 % 127) - 63) / 1024.0f;
        for (size_t i = 0; i < base.size(); ++i) base[i] = float(int(i % 7) - 3) / 8.0f;
        Data input(DataType::FLOAT32, {1, 1, 4, dim}, x), weight(DataType::FLOAT32, {24, 4 * dim}, fn),
             scale(DataType::FLOAT32, {3}, std::vector<float>{0.25f, 0.5f, 0.125f}), bias(DataType::FLOAT32, {24}, base);
        ToDataType(input, dtype);
        ApplyDeviceMap({{"cuda", 1}}, 0, 1);
        input.ToDevice(DataDevice::CUDA);
        Data pre, post, comb, oldPre, oldPost, oldComb;
        const char *previous = std::getenv("FASTLLM_DSV41_LEGACY_HCMIX_DECODE");
        const std::string saved = previous ? previous : "";
        setenv("FASTLLM_DSV41_LEGACY_HCMIX_DECODE", "1", 1);
        Check(FastllmCudaDeepSeekV41HcMix(input, weight, scale, bias, 4, iterations, 1e-6f, 1e-6f,
                                       oldPre, oldPost, oldComb), "HC baseline rejected input");
        setenv("FASTLLM_DSV41_LEGACY_HCMIX_DECODE", "0", 1);
        Check(FastllmCudaDeepSeekV41HcMix(input, weight, scale, bias, 4, iterations, 1e-6f, 1e-6f,
                                       pre, post, comb), "HC decode rejected input");
        if (previous) setenv("FASTLLM_DSV41_LEGACY_HCMIX_DECODE", saved.c_str(), 1);
        else unsetenv("FASTLLM_DSV41_LEGACY_HCMIX_DECODE");
        Check(Read(pre) == Read(oldPre) && Read(post) == Read(oldPost) && Read(comb) == Read(oldComb),
              "HC decode changed accumulation or Sinkhorn order");
    }
}

static void SharedSwigluBoundary() {
    const float values[] = {-40.0f, -10.04f, -10.0f, -0.0313f, 0.0f, 0.00783f, 1.003f, 9.97f, 10.0f, 10.04f, 40.0f};
    for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16})
    for (int rows : {1, 5})
    for (int mid : {33, 2304})
    for (float limit : {0.0f, 10.0f}) {
        ApplyDeviceMap({{"cpu", 1}}, 0, 1);
        std::vector<float> x(rows * mid * 2);
        for (size_t i = 0; i < x.size(); ++i) x[i] = values[(i * 7 + i / mid) % 11];
        Data input(DataType::FLOAT32, {rows, mid * 2}, x), output;
        ToDataType(input, dtype);
        x = Read(input);
        std::vector<float> expected(rows * mid);
        for (int i = 0; i < rows * mid; ++i) {
            int offset = (i / mid) * mid * 2 + i % mid;
            float gate = Bf(x[offset]), up = Bf(x[offset + mid]);
            if (limit > 0.0f) { gate = std::min(gate, limit); up = std::max(-limit, std::min(up, limit)); }
            expected[i] = Bf((gate / (1.0f + std::exp(-gate))) * up);
        }
        ApplyDeviceMap({{"cuda", 1}}, 0, 1);
        input.ToDevice(DataDevice::CUDA);
        Check(FastllmCudaDeepSeekV41SharedSwiglu(input, limit, output), "shared SwiGLU rejected input");
        Check(output.dataType == DataType::BFLOAT16 && output.dims == std::vector<int>({rows, mid}),
              "shared SwiGLU output boundary changed");
        Check(Read(output) == expected, "shared SwiGLU clamp or BF16 rounding mismatch");
    }
}
#endif
int main(int argc, char **argv) {
    try {
        SetThreads(30);
        ExportedActivationMetadata();
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
            ParallelReferenceBoundary();
            SharedSwigluBoundary();
            HcDecodeOrder();
            NativeLargeExpertRouter();
            NativeMxfp4Moe();
            NativeMxfp4Accumulation();
            NativeFp8ScaleRange();
        }
#endif
        ReferenceFp8BlockBoundary();
        ReferenceQuantizedLinearOrder();
        MixedMoe();
#ifdef USE_NUMAS
        if (argc > 1 && std::string(argv[1]) == "--numa") MixedMoe(true);
#endif
        ReferenceMoeBatchOrder();
        NVFP4TokenReuseOrder();
        std::cout << "V4.1 precision regression PASS: " << checks << " checks\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
