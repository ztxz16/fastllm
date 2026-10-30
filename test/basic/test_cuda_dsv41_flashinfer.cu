// Independent FP64 attention oracle; no Python/FlashInfer runtime dependency.
#include "../../src/devices/cuda/models/deepseekv41-flashinfer.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void Check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
void Require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class T> struct Buffer {
    T *data = nullptr;
    explicit Buffer(size_t count) {
        Check(cudaMalloc(&data, count * sizeof(T)));
    }
    ~Buffer() {
        cudaFree(data);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void Upload(const std::vector<T> &values) {
        Check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Graph() {
        if (exec)
            cudaGraphExecDestroy(exec);
        if (graph)
            cudaGraphDestroy(graph);
    }
};
float Fp8(uint8_t value) {
    const int exponent = (value >> 3) & 15, fraction = value & 7;
    const float magnitude =
        exponent ? std::ldexp(1.f + fraction / 8.f, exponent - 7) : std::ldexp(float(fraction), -9);
    return value & 128 ? -magnitude : magnitude;
}
float Bf16(float value) {
    return float(__float2bfloat16_rn(value));
}
constexpr int kDim = 512, kCapacity = 2048;
struct Cache {
    std::vector<uint8_t> ring, compressed;
    std::vector<float> ringValues, compressedValues;
    Buffer<uint8_t> gpuRing{128 * 528}, gpuCompressed{kCapacity * 288};
    explicit Cache(std::mt19937 &rng)
        : ring(128 * 528), compressed(kCapacity * 288), ringValues(128 * kDim),
          compressedValues(kCapacity * kDim) {
        for (int row = 0; row < 128; ++row) {
            for (int d = 0; d < kDim; ++d)
                ring[row * 528 + d] = (rng() % 33 + 0x20) | ((rng() & 1) << 7);
            for (int group = 0; group < 16; ++group)
                ring[row * 528 + 512 + group] = 125 + rng() % 4;
            for (int d = 0; d < kDim; ++d)
                ringValues[row * kDim + d] =
                    Bf16(std::ldexp(Fp8(ring[row * 528 + d]), int(ring[row * 528 + 512 + d / 32]) - 127));
        }
        const float fp4[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
        for (int row = 0; row < kCapacity; ++row) {
            for (int d = 0; d < 256; ++d)
                compressed[row * 288 + d] = rng() & 255;
            for (int group = 0; group < 32; ++group)
                compressed[row * 288 + 256 + group] = 0x18 + rng() % 17;
            for (int d = 0; d < kDim; ++d) {
                const int value = (compressed[row * 288 + d / 2] >> (4 * (d % 2))) & 15;
                compressedValues[row * kDim + d] = Bf16((value & 8 ? -1.f : 1.f) * fp4[value & 7] *
                                                        Fp8(compressed[row * 288 + 256 + d / 16]));
            }
        }
        gpuRing.Upload(ring);
        gpuCompressed.Upload(compressed);
    }
};
FastllmV41FullDecodeConfig Config(int heads, int batch, int width, int ratio) {
    FastllmV41FullDecodeConfig c;
    c.heads = heads;
    c.batch = batch;
    c.topK = width ? width : 512;
    c.ratio = ratio;
    c.ringMode = 1;
    c.ringRowBytes = 528;
    c.cmpMode = 3;
    c.cmpRowBytes = 288;
    c.softmaxScale = 1.f / std::sqrt(float(kDim));
    return c;
}
void CheckGuards() {
    for (int variant = 0; variant < 12; ++variant) {
        auto c = Config(64, 1, 512, 2);
        switch (variant) {
        case 0:
            c.verifying = true;
            break;
        case 1:
            c.batch = 0;
            break;
        case 2:
            c.batch = 129;
            break;
        case 3:
            c.heads = 0;
            break;
        case 4:
            c.heads = 129;
            break;
        case 5:
            c.window = 256;
            break;
        case 6:
            c.ringMode = 0;
            break;
        case 7:
            c.cmpMode = 1;
            break;
        case 8:
            c.topK = 2048;
            break;
        case 9:
            c.softmaxScale = NAN;
            break;
        case 10:
            c.softmaxScale = 0;
            break;
        case 11:
            c.ratio = 3;
            break;
        }
        std::unique_ptr<void, decltype(&FastllmV41FlashInferDestroy)> rejected(FastllmV41FlashInferCreate(c),
                                                                               FastllmV41FlashInferDestroy);
        Require(!rejected, "unsupported FlashInfer configuration accepted");
    }
}
double RunCase(Cache &first, Cache &second, std::mt19937 &rng, int heads, int batch, int width, int ratio,
               int replays) {
    const auto config = Config(heads, batch, width, ratio);
    std::unique_ptr<void, decltype(&FastllmV41FlashInferDestroy)> plan(FastllmV41FlashInferCreate(config),
                                                                       FastllmV41FlashInferDestroy);
    Require(bool(plan), "supported FlashInfer configuration rejected");
    std::vector<__nv_bfloat16> q(batch * heads * kDim), current(batch * kDim), output(q.size());
    std::vector<float> sink(heads);
    std::vector<int32_t> indices(batch * std::max(1, width));
    std::vector<FastllmV41FullDecodeRequest> requests(batch);
    Buffer<__nv_bfloat16> gpuQ(q.size()), gpuCurrent(current.size()), gpuOutput(output.size());
    Buffer<float> gpuSink(sink.size());
    Buffer<int32_t> gpuIndices(indices.size());
    Buffer<FastllmV41FullDecodeRequest> gpuRequests(batch);
    FastllmV41FullDecodeBuffers buffers;
    buffers.q = gpuQ.data;
    buffers.kv = gpuCurrent.data;
    buffers.output = gpuOutput.data;
    buffers.sink = gpuSink.data;
    buffers.topk = gpuIndices.data;
    Graph graph;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Check(FastllmV41FlashInferRun(plan.get(), buffers, gpuRequests.data, cudaStreamPerThread));
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph.graph));
    Check(cudaGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));
    const int positions[] = {0, 1, 2, 63, 127, 128, 129, 4094, 4095, 4096, 4097, 4098, 127, 128, 255, 256};
    std::normal_distribution<float> random(0.f, 1.f);
    double worst = 0;
    for (int replay = 0; replay < replays; ++replay) {
        for (auto &v : q)
            v = __float2bfloat16_rn(random(rng));
        for (auto &v : current)
            v = __float2bfloat16_rn(random(rng));
        for (auto &v : sink)
            v = random(rng) * 3;
        for (auto &v : indices)
            v = int(rng() % (kCapacity + 11)) - 2;
        for (int token = 0; token < batch; ++token) {
            Cache &cache = (replay + token) % 2 ? first : second;
            requests[token].windowKV = cache.gpuRing.data;
            requests[token].compressedKV = cache.gpuCompressed.data;
            requests[token].position = std::min(positions[(replay + token) % 16], 2047 * std::max(ratio, 1));
        }
        gpuQ.Upload(q);
        gpuCurrent.Upload(current);
        gpuSink.Upload(sink);
        gpuIndices.Upload(indices);
        gpuRequests.Upload(requests);
        Check(cudaGraphLaunch(graph.exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        Check(cudaMemcpy(output.data(), gpuOutput.data, output.size() * sizeof(output[0]),
                         cudaMemcpyDeviceToHost));
        for (int token = 0; token < batch; ++token) {
            Cache &cache = (replay + token) % 2 ? first : second;
            const int position = requests[token].position;
            std::vector<const float *> rows;
            for (int p = std::max(0, position - 127); p < position; ++p)
                rows.push_back(cache.ringValues.data() + p % 128 * kDim);
            float exactCurrent[kDim];
            for (int d = 0; d < kDim; ++d)
                exactCurrent[d] = float(current[token * kDim + d]);
            rows.push_back(exactCurrent);
            const int count = ratio ? (position + 1) / ratio : 0;
            for (int k = 0; k < std::min(width, count); ++k) {
                const int id = indices[token * width + k];
                if (id >= 0 && id < count)
                    rows.push_back(cache.compressedValues.data() + id * kDim);
            }
            double numerator = 0, denominator = 0, maxError = 0, maxReference = 0;
            for (int h = 0; h < heads; ++h) {
                const size_t offset = (size_t(token) * heads + h) * kDim;
                std::vector<double> probabilities(rows.size());
                double maximum = sink[h];
                for (size_t row = 0; row < rows.size(); ++row) {
                    double dot = 0;
                    for (int d = 0; d < kDim; ++d)
                        dot += double(float(q[offset + d])) * rows[row][d];
                    probabilities[row] = dot * config.softmaxScale;
                    maximum = std::max(maximum, probabilities[row]);
                }
                double total = std::exp(sink[h] - maximum);
                for (double &p : probabilities) {
                    p = std::exp(p - maximum);
                    total += p;
                }
                double reference[kDim] = {};
                for (size_t row = 0; row < rows.size(); ++row)
                    for (int d = 0; d < kDim; ++d)
                        reference[d] += probabilities[row] / total * rows[row][d];
                for (int d = 0; d < kDim; ++d) {
                    const double actual = float(output[offset + d]);
                    Require(std::isfinite(actual), "non-finite FlashInfer output");
                    const double diff = actual - reference[d];
                    numerator += diff * diff;
                    denominator += reference[d] * reference[d];
                    maxError = std::max(maxError, std::fabs(diff));
                    maxReference = std::max(maxReference, std::fabs(reference[d]));
                }
            }
            const double relativeL2 = std::sqrt(numerator / std::max(denominator, 1e-30));
            Require(relativeL2 < .015 && maxError / std::max(maxReference, 1e-30) < .025,
                    "FlashInfer exceeded operator error tolerance");
            worst = std::max(worst, relativeL2);
        }
    }
    return worst;
}
} // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE");
            return 77;
        }
        cudaDeviceProp prop{};
        Check(cudaGetDeviceProperties(&prop, 0));
        if (prop.major != 12 || prop.minor > 1) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: FlashInfer adapter requires SM120/SM121");
            return 77;
        }
        CheckGuards();
        std::mt19937 rng(6193);
        Cache first(rng), second(rng);
        double worst = 0;
        int cases = 0;
        for (const auto shape : {std::pair<int, int>{1, 1},
                                 {8, 1},
                                 {15, 2},
                                 {16, 8},
                                 {32, 4},
                                 {48, 2},
                                 {64, 1},
                                 {64, 8},
                                 {96, 3},
                                 {128, 1},
                                 {8, 64},
                                 {1, 128}})
            for (int width : {0, 32, 512}) {
                worst = std::max(
                    worst, RunCase(first, second, rng, shape.first, shape.second, width, width ? 2 : 0, 3));
                cases += 3;
            }
        for (const auto setting :
             {std::pair<int, int>{0, 0}, {32, 1}, {128, 2}, {512, 1}, {512, 2}, {1024, 2}}) {
            worst = std::max(worst, RunCase(first, second, rng, 64, 1, setting.first, setting.second, 16));
            cases += 16;
        }
        std::printf("FLASHINFER_PASS %d graph cases, 12 dispatch guards, worst relative L2 %.9g\n", cases,
                    worst);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
