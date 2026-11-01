// Independent FP64 attention oracle; no Python/FlashInfer runtime dependency.
#include "../../src/devices/cuda/models/deepseekv41-flashinfer.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <climits>
#include <string>
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
        if (count)
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
    const char *flag = std::getenv("FASTLLM_DSV41_FLASHINFER");
    const bool wasSet = flag != nullptr;
    const std::string original = flag ? flag : "";
    unsetenv("FASTLLM_DSV41_FLASHINFER");
    Require(FastllmV41FlashInferEnabled(), "eligible FlashInfer paths must default on");
    for (const char *value : {"", "0", "false", "off", "true", "1", "01"}) {
        setenv("FASTLLM_DSV41_FLASHINFER", value, 1);
        Require(FastllmV41FlashInferEnabled() == (std::strcmp(value, "1") == 0),
                "inconsistent FlashInfer environment policy");
    }
    if (wasSet)
        setenv("FASTLLM_DSV41_FLASHINFER", original.c_str(), 1);
    else
        unsetenv("FASTLLM_DSV41_FLASHINFER");
    for (int variant = 0; variant < 14; ++variant) {
        FastllmV41FlashInferChunkConfig c{1, 8, 64, 512, 2048, 4096, .044194f};
        switch (variant) {
        case 0: c.batches = 0; break;
        case 1: c.length = 0; break;
        case 2: c.length = 129; break;
        case 3: c.batches = 17; break;
        case 4: c.heads = 0; break;
        case 5: c.heads = 129; break;
        case 6: c.topK = -1; break;
        case 7: c.topK = 1025; break;
        case 8: c.compressedRows = 0; break;
        case 9: c.compressedRows = INT_MAX; c.batches = 2; break;
        case 10: c.startPosition = -1; break;
        case 11: c.startPosition = INT_MAX; break;
        case 12: c.scale = NAN; break;
        case 13: c.scale = 0; break;
        }
        Require(FastllmV41FlashInferChunkWorkspace(c) == 0,
                "unsupported ordinary FlashInfer configuration accepted");
    }
    for (int variant = 0; variant < 12; ++variant) {
        auto c = Config(64, 1, 512, 2);
        switch (variant) {
        case 0:
            c.ringRowBytes = 1024;
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
               int replays, bool verifying = false, int denseBatches = 0, bool missingCompressed = false) {
    auto config = Config(heads, batch, width, ratio);
    config.verifying = verifying;
    const int length = denseBatches ? batch / denseBatches : batch;
    std::unique_ptr<void, decltype(&FastllmV41FlashInferDestroy)> plan(
        denseBatches ? nullptr : FastllmV41FlashInferCreate(config), FastllmV41FlashInferDestroy);
    Require(denseBatches || bool(plan), "supported FlashInfer configuration rejected");
    std::vector<__nv_bfloat16> q(batch * heads * kDim), current(batch * kDim), output(q.size());
    std::vector<float> sink(heads);
    std::vector<int32_t> indices(batch * std::max(1, width));
    std::vector<FastllmV41FullDecodeRequest> requests(batch);
    Buffer<__nv_bfloat16> gpuQ(q.size()), gpuCurrent(current.size()), gpuOutput(output.size());
    Buffer<float> gpuSink(sink.size());
    Buffer<int32_t> gpuIndices(indices.size());
    Buffer<FastllmV41FullDecodeRequest> gpuRequests(denseBatches ? 0 : batch);
    FastllmV41FullDecodeBuffers buffers;
    buffers.q = gpuQ.data;
    buffers.kv = gpuCurrent.data;
    buffers.output = gpuOutput.data;
    buffers.sink = gpuSink.data;
    buffers.topk = gpuIndices.data;
    FastllmV41FlashInferChunkConfig chunkConfig{std::max(denseBatches, 1), length, heads, width,
                                               kCapacity, 0, config.softmaxScale};
    const size_t workspaceBytes = denseBatches ? FastllmV41FlashInferChunkWorkspace(chunkConfig) : 0;
    Require(!denseBatches || workspaceBytes > 0, "supported ordinary FlashInfer chunk rejected");
    Buffer<uint8_t> workspace(workspaceBytes);
    Buffer<uint8_t> denseRing(denseBatches * first.ring.size());
    Buffer<uint8_t> denseCompressed(denseBatches * first.compressed.size());
    Graph graph;
    if (!denseBatches) {
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Check(FastllmV41FlashInferRun(plan.get(), buffers, gpuRequests.data, cudaStreamPerThread));
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph.graph));
        Check(cudaGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));
    }
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
            Cache &cache = (replay + (denseBatches ? token / length : verifying ? 0 : token)) % 2 ? first : second;
            requests[token].windowKV = cache.gpuRing.data;
            requests[token].compressedKV = missingCompressed && (replay & 1) ? nullptr : cache.gpuCompressed.data;
            requests[token].position = verifying ?
                std::min(positions[replay % 16], 2047 * std::max(ratio, 1) - length) + token % length :
                std::min(positions[(replay + token) % 16], 2047 * std::max(ratio, 1));
            // Exercise future compressed IDs at the front, valid IDs later in
            // the list, and the -1 padding emitted by the native indexer.
            if (verifying && width) {
                const int visible = (requests[token].position + 1) / ratio;
                for (int k = 0; k < width; ++k) {
                    if (k % 7 == 0) indices[token * width + k] = visible;
                    if (k % 7 == 1) indices[token * width + k] = visible - 1;
                    if (k % 7 == 2) indices[token * width + k] = -1;
                }
            }
        }
        if (denseBatches) {
            std::vector<uint8_t> rings, compressed;
            for (int n = 0; n < denseBatches; ++n) {
                const Cache &cache = (replay + n) % 2 ? first : second;
                rings.insert(rings.end(), cache.ring.begin(), cache.ring.end());
                compressed.insert(compressed.end(), cache.compressed.begin(), cache.compressed.end());
            }
            denseRing.Upload(rings);
            denseCompressed.Upload(compressed);
            chunkConfig.startPosition = requests[0].position;
            // Ordinary sparse attention receives a prefiltered index list from
            // the indexer. Reproduce that contract, including unused list slots.
            for (int token = 0; token < batch && width; ++token) {
                const int visible = (requests[token].position + 1) / ratio;
                const int count = (requests[0].position + length) / ratio;
                for (int k = 0; k < width; ++k)
                    if (k >= count || indices[token * width + k] >= visible)
                        indices[token * width + k] = -1;
            }
        }
        gpuQ.Upload(q);
        gpuCurrent.Upload(current);
        gpuSink.Upload(sink);
        gpuIndices.Upload(indices);
        if (!denseBatches)
            gpuRequests.Upload(requests);
        if (denseBatches) {
            Graph directGraph;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Check(FastllmV41FlashInferChunkRun(chunkConfig, buffers, chunkConfig.startPosition ? denseRing.data : nullptr,
                width ? denseCompressed.data : nullptr, workspace.data, cudaStreamPerThread));
            Check(cudaStreamEndCapture(cudaStreamPerThread, &directGraph.graph));
            Check(cudaGraphInstantiate(&directGraph.exec, directGraph.graph, nullptr, nullptr, 0));
            for (int i = 0; i < 3; ++i)
                Check(cudaGraphLaunch(directGraph.exec, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
        } else {
            Check(cudaGraphLaunch(graph.exec, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
        }
        Check(cudaMemcpy(output.data(), gpuOutput.data, output.size() * sizeof(output[0]),
                         cudaMemcpyDeviceToHost));
        for (int token = 0; token < batch; ++token) {
            Cache &cache = (replay + (denseBatches ? token / length : verifying ? 0 : token)) % 2 ? first : second;
            const int position = requests[token].position;
            std::vector<const float *> rows;
            const int chunkStart = verifying ? requests[0].position : position;
            for (int p = std::max(0, position - 127); p < chunkStart; ++p)
                rows.push_back(cache.ringValues.data() + p % 128 * kDim);
            const int prefixRows = verifying ? token % length + 1 : 1;
            const int prefixStart = verifying ? token - token % length : token;
            std::vector<float> exactCurrent(prefixRows * kDim);
            for (size_t d = 0; d < exactCurrent.size(); ++d)
                exactCurrent[d] = float(current[prefixStart * kDim + d]);
            for (size_t d = 0; d < exactCurrent.size(); d += kDim)
                rows.push_back(exactCurrent.data() + d);
            const int count = ratio ? (position + 1) / ratio : 0;
            const int listCount = ratio ? (verifying ? requests[0].position + length : position + 1) / ratio : 0;
            for (int k = 0; k < std::min(width, listCount); ++k) {
                const int id = indices[token * width + k];
                if (id >= 0 && id < count && !(missingCompressed && (replay & 1)))
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
        // Verification width is independent of the configured DSpark length.
        // Cover odd lengths, padding heads and the plan's maximum chunk size.
        int verifyCases = 0;
        double verifyWorst = 0;
        for (int batch : {1, 2, 3, 4, 6, 8, 16, 32, 64, 128}) {
            const int heads = batch <= 8 ? 64 : 8;
            for (int ratio : {0, 1, 2}) {
                verifyWorst = std::max(verifyWorst,
                    RunCase(first, second, rng, heads, batch, ratio ? 512 : 0, ratio, 16, true));
                verifyCases += 16;
            }
        }
        for (int heads : {1, 8, 15, 16, 32, 48, 96, 128}) {
            verifyWorst = std::max(verifyWorst, RunCase(first, second, rng, heads, 3, 32, 2, 3, true));
            verifyCases += 3;
        }
        verifyWorst = std::max(verifyWorst, RunCase(first, second, rng, 64, 8, 1024, 2, 16, true));
        verifyCases += 16;
        std::printf("FLASHINFER_VERIFY_PASS %d graph cases, worst relative L2 %.9g\n",
                    verifyCases, verifyWorst);
        int chunkCases = 0;
        double chunkWorst = 0;
        for (int length : {1, 2, 3, 4, 6, 8, 16, 128})
            for (int ratio : {0, 1, 2}) {
                chunkWorst = std::max(chunkWorst, RunCase(first, second, rng, length <= 8 ? 64 : 8,
                    length, ratio ? 512 : 0, ratio, 16, true, 1));
                chunkCases += 16;
            }
        for (const auto shape : {std::pair<int, int>{15, 6}, {32, 16}, {8, 128}}) {
            chunkWorst = std::max(chunkWorst, RunCase(first, second, rng, shape.first,
                shape.second, 1024, 2, 16, true, 2));
            chunkCases += 16;
        }
        // Ordinary reuse covers multiple sequences, odd index strides and
        // the temporary-storage cutoff, always against the same FP64 oracle.
        for (int width : {127, 128, 129, 513, 1023})
            for (auto shape : {std::pair<int,int>{33,3}, {64,7}, {127,17}, {64,32}}) {
                chunkWorst = std::max(chunkWorst, RunCase(first, second, rng,
                    shape.first, shape.second * 2, width, 2, 8, true, 2));
                chunkCases += 8;
            }
        std::printf("FLASHINFER_CHUNK_PASS %d cases (3 graph replays each), worst relative L2 %.9g\n",
                    chunkCases, chunkWorst);
        std::printf("FLASHINFER_PASS %d graph cases, 26 dispatch guards, environment policy, worst relative L2 %.9g\n", cases,
                    worst);
        // Exercise reuse selection boundaries, partial head groups, and a
        // compressed cache disappearing/reappearing within the same graph.
        double reuseWorst = 0;
        int reuseCases = 0;
        for (int width : {127, 128, 129, 511, 513, 1023})
            for (auto shape : {std::pair<int,int>{33,3}, {64,6}, {127,17}}) {
                reuseWorst = std::max(reuseWorst, RunCase(first, second, rng,
                    shape.first, shape.second, width, 2, 10, true, 0, true));
                reuseCases += 10;
            }
        std::printf("FLASHINFER_REUSE_PASS %d dynamic graph cases, worst relative L2 %.9g\n",reuseCases,reuseWorst);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
