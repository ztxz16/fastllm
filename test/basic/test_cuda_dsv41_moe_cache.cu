#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/numas/numasdevice.h"
#include "devices/numas/numas.h"
#include "../../src/devices/cuda/moe/fastllm-moe-deepseekv41-cache.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <sys/prctl.h>
namespace fastllm {
NumaConfig *GetNumaConfig();
void DoCudaMergeMOEFromCPU(Data &, Data &, Data &, Data &, Data &, Data &, Data &, Data **, Data **, float, bool,
                           const std::unordered_set<int> &, bool, MoeGateType, bool, float, int, bool);
}
using namespace fastllm;
static void Require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
static void Check(cudaError_t state) {
    Require(state == cudaSuccess, cudaGetErrorString(state));
}
static float Bf(float x) {
    return RoundFloat32ToBFloat16RNE(x);
}
__global__ void CheckFP4Decode(float *output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = fastllm::cuda::dsv41_cache::FP4(i % 16, i / 16);
}
static void Quant(std::vector<float> &v, int blockSize = 32) {
    for (size_t start = 0; start < v.size(); start += blockSize) {
        float maximum = 1e-4f;
        for (size_t c = start; c < start + blockSize; ++c)
            maximum = std::max(maximum, std::abs(v[c]));
        const float scale = std::exp2(std::ceil(std::log2(maximum / 448.0f)));
        for (size_t c = start; c < start + blockSize; ++c) {
            const float x = std::abs(v[c] / scale);
            float distance = INFINITY, best = 0;
            for (int code = 0; code < 127; ++code) {
                const int e = code >> 3, m = code & 7;
                const float candidate = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1 + m / 8.0f, e - 7);
                const float d = std::abs(x - candidate);
                if (d < distance || (d == distance && !(code & 1))) {
                    best = candidate;
                    distance = d;
                }
            }
            v[c] = std::copysign(best * scale, v[c]);
        }
    }
}
static void Gpu(Data &d) {
    d.dataDevice = DataDevice::CUDA;
    d.dataDeviceIds = {0};
    d.Allocate(false);
}
static void RunNumasMoe(Data &input, Data &index, Data &scores, Data &output, std::vector<Data *> &weights, int layer) {
    Data w1, w2, w3, curInput, curOutput;
    std::vector<Data *> biases(weights.size(), nullptr);
    ((Executor *)GetExecutor())
        ->RunOnDevice("numa", "MergeMOE",
                      {{"input", &input},
                       {"index", &index},
                       {"score", &scores},
                       {"weights", (Data *)weights.data()},
                       {"biass", (Data *)biases.data()},
                       {"w1", &w1},
                       {"w2", &w2},
                       {"w3", &w3},
                       {"curInput", &curInput},
                       {"curOutput", &curOutput},
                       {"output", &output}},
                      {{"swigluLimit", 10}},
                      {{"weights___batch", (int)weights.size()},
                       {"biass___batch", (int)biases.size()},
                       {"layer", layer},
                       {"deepSeekV4Mode", 1},
                       {"activationQuantBlock", 32}});
}
static void Compare(const std::vector<float> &actual, const std::vector<float> &expected, const char *label) {
    double error = 0, norm = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        Require(std::isfinite(actual[i]), "nonfinite expert output");
        error += std::pow(double(actual[i]) - expected[i], 2);
        norm += double(expected[i]) * expected[i];
    }
    if (error > 1e-5 * std::max(1e-12, norm)) {
        std::fprintf(stderr, "%s relative L2 %.8f\n", label, std::sqrt(error / norm));
        throw std::runtime_error(label);
    }
}
int main(int argc, char **argv) {
    try {
        Require(prctl(PR_SET_DUMPABLE, 0) == 0, "disable test core dumps");
        const bool dual = argc > 1 && std::string(argv[1]) == "--dual";
        int devices = 0;
        Check(cudaGetDeviceCount(&devices));
        Require(!dual || devices >= 2, "--dual requires two CUDA devices");
        unsetenv("FASTLLM_MOE_CUDA_CACHE_BYTES_0");
        unsetenv("FASTLLM_MOE_CUDA_CACHE_BYTES_1");
        unsetenv("FASTLLM_DSV41_MOE_CACHE_MODE");
        // Exercise GPU prefill deliberately; CPU-oracle inputs below have no
        // CUDA mirror and therefore still run entirely on NUMA.
        setenv("FT_GPU_PREFILL", "1", 1);
        setenv("FT_EXPERT_LIMIT", "0", 1);
        SetThreads(30);
        {
            float *device;
            Check(cudaMalloc(&device, 4096 * sizeof(float)));
            CheckFP4Decode<<<16, 256>>>(device);
            std::vector<float> values(4096);
            Check(cudaMemcpy(values.data(), device, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
            Check(cudaFree(device));
            const float magnitude[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
            for (int i = 0; i < 4096; ++i) {
                const int code = i % 16, exponent = i / 16;
                if (exponent == 255) {
                    Require(std::isnan(values[i]), "FP4 NaN scale");
                    continue;
                }
                const float scale = exponent == 0 ? std::ldexp(1.0f, -127) : std::ldexp(1.0f, exponent - 127);
                const float expected = Bf(std::copysign(magnitude[code & 7], code & 8 ? -1.0f : 1.0f) * scale);
                Require(std::memcmp(&values[i], &expected, sizeof(float)) == 0, "FP4 decode / CPU oracle mismatch");
            }
        }
        // Preserve the existing V4 block-128 helper while extending V4.1.
        for (int block : {32, 128})
            for (bool quantize : {false, true}) {
                std::vector<float> gate(3 * 256), scales{0, .37f, 1.2f}, expected;
                for (size_t i = 0; i < gate.size(); ++i)
                    gate[i] = Bf((int(i * 31 % 211) - 105) / 4.0f);
                for (int row = 0; row < 3; ++row) {
                    std::vector<float> act(128);
                    for (int c = 0; c < 128; ++c) {
                        float g = std::min(gate[row * 256 + c * 2], 10.0f);
                        float u = std::clamp(gate[row * 256 + c * 2 + 1], -10.0f, 10.0f);
                        act[c] = Bf(scales[row] * ((g / (1 + std::exp(-g))) * u));
                    }
                    if (quantize)
                        Quant(act, block);
                    expected.insert(expected.end(), act.begin(), act.end());
                }
                Data g(FLOAT32, {3, 256}, gate), scale(FLOAT32, {3}, scales), output;
                ToDataType(g, BFLOAT16);
                g.ToDevice(DataDevice::CUDA);
                scale.ToDevice(DataDevice::CUDA);
                Require(
                    FastllmCudaDeepSeekV4PrepareMoeDownInput(g, output, (float *)scale.cudaData, 10, quantize, block),
                    "down preparation rejected");
                ToDataType(output, FLOAT32);
                output.ToDevice(DataDevice::CPU);
                Compare(std::vector<float>((float *)output.cpuData, (float *)output.cpuData + 384), expected,
                        "down preparation/oracle mismatch");
            }
        constexpr int hidden = 256, inter = 128, experts = 24, tables = 2, topk = 6;
        const int nodes = GetNumaConfig()->numaCnt;
        std::vector<std::unique_ptr<Data>> owned;
        std::vector<std::unique_ptr<Data>> coldOwned;
        std::vector<Data *> coldWeights(2 * (experts + 1), nullptr);
        std::vector<void *> shards;
        std::vector<Data *> weights[tables];
        std::vector<std::vector<float>> dense[tables];
        FastllmCudaMoeCacheLayer layers[tables];
        std::mt19937 rng(12345);
        for (int t = 0; t < tables; ++t) {
            weights[t].resize(2 * (experts + 1), nullptr);
            dense[t].resize(experts * 2);
            for (int e = 0; e < experts; ++e)
                for (int part = 0; part < 2; ++part) {
                    const int rows = part ? hidden : inter * 2, cols = part ? inter : hidden;
                    auto w = std::make_unique<Data>(DataType::NVFP4_BLOCK_32_E8M0);
                    w->Resize({rows, cols});
                    w->blockK = 1;
                    w->blockM = 32;
                    const size_t pitch = GetDataBytes(w->dataType, 1, cols);
                    std::vector<uint8_t> packed(rows * pitch);
                    auto &matrix = dense[t][e * 2 + part];
                    matrix.resize(rows * cols);
                    const float fp4[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
                    for (int row = 0; row < rows; ++row)
                        for (int c = 0; c < cols; c += 32) {
                            uint8_t *block = packed.data() + row * pitch + c / 32 * 17;
                            const int exponent = 121 + rng() % 4;
                            block[16] = exponent;
                            for (int b = 0; b < 16; ++b) {
                                block[b] = rng() % 256;
                                for (int half = 0; half < 2; ++half) {
                                    const int code = (block[b] >> (half * 4)) & 15;
                                    matrix[row * cols + c + b * 2 + half] =
                                        std::ldexp((code & 8 ? -1 : 1) * fp4[code & 7], exponent - 127);
                                }
                            }
                        }
                    if (t == 0) {
                        // Source checkpoint layout: separate gate/up halves and
                        // planar UE8M0 scales, with no registered NUMA shards.
                        auto source = std::make_unique<Data>(NVFP4, std::vector<int>{rows, cols});
                        source->blockK = 1;
                        source->blockM = 32;
                        source->isModelWeight = true;
                        source->Allocate();
                        uint8_t *scales = GetNVFP4ScaleData(*source);
                        Require(scales != nullptr, "missing source NVFP4 scales");
                        for (int row = 0; row < rows; ++row)
                            for (int c = 0; c < cols; c += 32) {
                                const uint8_t *block = packed.data() + row * pitch + c / 32 * 17;
                                memcpy(source->cpuData + (row * cols + c) / 2, block, 16);
                                scales[row * (cols / 32) + c / 32] = block[16];
                            }
                        coldWeights[2 * (e + 1) + part] = source.get();
                        coldOwned.push_back(std::move(source));
                    }
                    for (int node = 0; node < nodes; ++node) {
                        void *ptr = nullptr;
                        Check(cudaHostAlloc(&ptr, rows / nodes * pitch, cudaHostAllocMapped | cudaHostAllocPortable));
                        for (int r = 0; r < rows / nodes; ++r) {
                            const int physical = node * (rows / nodes) + r;
                            const int logical = part ? physical : physical / 2 + (physical & 1) * inter;
                            memcpy((uint8_t *)ptr + r * pitch, packed.data() + logical * pitch, pitch);
                        }
                        shards.push_back(ptr);
                        w->numasData.push_back((uint8_t *)ptr);
                    }
                    w->isPinned = true;
                    w->isModelWeight = true;
                    weights[t][2 * (e + 1) + part] = w.get();
                    owned.push_back(std::move(w));
                }
            layers[t] = {weights[t].data(), (int)weights[t].size(), true, 10.0f};
        }
        const size_t record = ((size_t(3) * hidden * inter / 32 * 17 + 127) / 128) * 128;
        SetMoeCudaCacheBytes(0);
        Require(!FastllmCudaPrepareMoeCache(layers, tables, [] {}), "disabled cache accepted");
        SetMoeCudaCacheBytes(record);
        Require(!FastllmCudaPrepareMoeCache(layers, tables, [] {}), "undersized cache accepted");
        SetMoeCudaCacheBytes(record * 16);
        Require(!FastllmCudaPrepareMoeCache(layers, tables), "unowned NUMA cache accepted");
        // A disabled, undersized, malformed or overflowing device override
        // must leave that device on its configured fallback backend.
        for (const char *value : {"0", "1", "-1", "1g", "bad", "18446744073709551616"}) {
            setenv("FASTLLM_MOE_CUDA_CACHE_BYTES_0", value, 1);
            Require(FastllmCudaPrepareMoeCache(layers, tables, [] {}), "host preparation rejected device override");
            Require(!FastllmCudaUseMoeHybrid(weights[0].data(), weights[0].size()), "invalid device budget accepted");
            if (dual && std::string(value) == "0") {
                Check(cudaSetDevice(1));
                Require(FastllmCudaUseMoeHybrid(weights[0].data(), weights[0].size()), "disabled GPU 0 affected GPU 1");
                Check(cudaSetDevice(0));
            }
            FastllmCudaReleaseMoeCache(weights[0].data(), weights[0].size());
        }
        unsetenv("FASTLLM_MOE_CUDA_CACHE_BYTES_0");
        if (dual) {
            // Different capacities must remain independent across devices.
            const auto first = std::to_string(record * 24), second = std::to_string(record * 16);
            setenv("FASTLLM_MOE_CUDA_CACHE_BYTES_0", first.c_str(), 1);
            setenv("FASTLLM_MOE_CUDA_CACHE_BYTES_1", second.c_str(), 1);
        }
        Require(FastllmCudaPrepareMoeCache(layers, tables, [] {}), "V4.1 cache preparation failed");
        Require(FastllmCudaCanRunMoeHybrid(weights[0].data(), weights[0].size()), "V4.1 hybrid unavailable");
        Require(!FastllmCudaCanRunMoeCache(weights[0].data(), weights[0].size()), "generic math accepted V4.1");
        Data input(BFLOAT16, {1, hidden}), index(INT32, {1, topk}), scores(FLOAT32, {1, topk}), output;
        Gpu(input);
        Gpu(index);
        Gpu(scores);
        auto supported = [&] {
            return FastllmCudaCanRunMoeCacheSmallBatch(input, index, scores, weights[0].data(), weights[0].size(),
                                                       MoeGateSwiglu);
        };
        Require(supported(), "valid BF16 decode rejected");
        input.dataType = FLOAT32;
        Require(!supported(), "wrong activation math accepted");
        input.dataType = BFLOAT16;
        input.dims[0] = 2;
        index.dims[0] = scores.dims[0] = 2;
        Require(!supported(), "V4.1 multi-token cache accepted");
        input.dims[0] = index.dims[0] = scores.dims[0] = 1;
        input.dims[1] = hidden + 32;
        Require(!supported(), "wrong hidden width accepted");
        input.dims[1] = hidden;
        Require(supported(), "fallback validation damaged cache");
        int rejectedCallbacks = 0;
        auto rejectedCallback = [&] { ++rejectedCallbacks; };
        input.dataType = FLOAT16;
        Require(!FastllmCudaMergeMOEHybrid(input, index, scores, output,
                    weights[0].data(), weights[0].size(), 0, rejectedCallback),
                "parallel hybrid accepted unsupported dtype");
        input.dataType = BFLOAT16;
        input.dims[0] = 2;
        Require(!FastllmCudaMergeMOEHybrid(input, index, scores, output,
                    weights[0].data(), weights[0].size(), 0, rejectedCallback),
                "parallel hybrid accepted a multirow callback");
        input.dims[0] = 1;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        const bool parallelCapture = FastllmCudaMergeMOEHybrid(input, index, scores, output,
            weights[0].data(), weights[0].size(), 0, rejectedCallback);
        cudaGraph_t rejectedGraph;
        Check(cudaStreamEndCapture(cudaStreamPerThread, &rejectedGraph));
        Check(cudaGraphDestroy(rejectedGraph));
        Require(!parallelCapture && rejectedCallbacks == 0, "rejected hybrid launched parallel work");
        const bool verifyOnly = argc > 2 && std::string(argv[2]) == "--verify-only";
        for (int pass = 0; pass < (verifyOnly ? 0 : 48); ++pass) {
            const int t = pass % tables;
            std::vector<float> x(hidden), route(topk), expected(hidden, 0), perExpert(topk * hidden);
            std::vector<__nv_bfloat16> bx(hidden), actual(hidden);
            std::vector<int32_t> ids(topk);
            for (int c = 0; c < hidden; ++c) {
                x[c] = Bf(std::ldexp((int(rng() % 79) - 39) / 16.0f, c / 32 % 4 - 2));
                bx[c] = __float2bfloat16(x[c]);
            }
            if (pass % 3 == 0) {
                x[0] = 48;
                x[1] = -64;
                bx[0] = __float2bfloat16(x[0]);
                bx[1] = __float2bfloat16(x[1]);
            }
            for (int r = 0; r < topk; ++r) {
                ids[r] = (pass / 4 * 7 + r * 5) % experts;
                route[r] = r == 0 ? 0 : (1 + rng() % 15) / 32.0f;
            }
            if (pass % 4 == 1)
                ids[1] = ids[2];
            Check(cudaMemcpy(input.cudaData, bx.data(), hidden * 2, cudaMemcpyHostToDevice));
            Check(cudaMemcpy(index.cudaData, ids.data(), topk * sizeof(int32_t), cudaMemcpyHostToDevice));
            Check(cudaMemcpy(scores.cudaData, route.data(), topk * sizeof(float), cudaMemcpyHostToDevice));
            auto q = x;
            Quant(q);
            for (int r = 0; r < topk; ++r) {
                const auto &g = dense[t][ids[r] * 2], &d = dense[t][ids[r] * 2 + 1];
                std::vector<float> activation(inter);
                for (int row = 0; row < inter; ++row) {
                    float gate = 0, up = 0;
                    for (int c = 0; c < hidden; ++c) {
                        gate += q[c] * g[row * hidden + c];
                        up += q[c] * g[(row + inter) * hidden + c];
                    }
                    gate = std::min(Bf(gate), 10.0f);
                    up = std::clamp(Bf(up), -10.0f, 10.0f);
                    activation[row] = Bf(route[r] * ((gate / (1 + std::exp(-gate))) * up));
                }
                Quant(activation);
                for (int row = 0; row < hidden; ++row) {
                    float sum = 0;
                    for (int c = 0; c < inter; ++c)
                        sum += activation[c] * d[row * inter + c];
                    perExpert[r * hidden + row] = Bf(sum);
                }
            }
            std::vector<int> order(topk);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ids[a] < ids[b]; });
            for (int r : order)
                for (int c = 0; c < hidden; ++c)
                    expected[c] += perExpert[r * hidden + c];
            for (auto &v : expected)
                v = Bf(v);
            // Check sparse CPU subsets, repeated IDs and zero scores against
            // the independent dense oracle; GPU-selected slots stay untouched.
            for (int gpu = 0; gpu < topk; ++gpu) {
                std::vector<int32_t> selected(topk, -1);
                for (int r = 0; r < gpu; ++r)
                    selected[(r + pass) % topk] = ids[(r + pass) % topk];
                std::vector<float> cpuOutput(topk * hidden, 0), expectedCpu = perExpert;
                for (int r = 0; r < topk; ++r)
                    if (selected[r] >= 0)
                        std::fill(expectedCpu.begin() + r * hidden, expectedCpu.begin() + (r + 1) * hidden, 0);
                NumasMoeDecodeExperts(x.data(), cpuOutput.data(), weights[t].data(), ids.data(), selected.data(), topk,
                                      t, route.data(), 10);
                Compare(cpuOutput, expectedCpu, "CPU subset/oracle mismatch");
            }
            // Reuse the same output across device transitions, as the model
            // does at a layer-partition boundary. Each device owns its cache.
            for (int device : (dual ? std::vector<int>{0, 1, 0} : std::vector<int>{0})) {
                if (dual) {
                    input.ToDevice(DataDevice::CUDA, {device}, true);
                    index.ToDevice(DataDevice::CUDA, {device}, true);
                    scores.ToDevice(DataDevice::CUDA, {device}, true);
                    FastllmCudaSetDevice(device);
                }
                Data parallelOutput(FLOAT32, {1, 1});
                parallelOutput.ToDevice(DataDevice::CUDA, {device}, false);
                parallelOutput.Allocate(false);
                // Enter pure mode with cold/evicted routes, enqueue repeatedly,
                // then return to CPU/hybrid on the same tensors and cache.
                // Alternate which mode sees misses so the adaptive policy still
                // receives real hybrid refill calibration samples.
                for (int split :
                     (pass % 2 ? std::vector<int>{-1, 0, 1, 3, 6, -1, 0} : std::vector<int>{0, 1, 3, 6, -1, -1, 0})) {
                    setenv("FASTLLM_DSV41_MOE_CACHE_MODE", split < 0 ? "gpu" : "hybrid", 1);
                    const auto value = std::to_string(split);
                    setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", value.c_str(), 1);
                    for (int repeat = 0; repeat < (split < 0 ? 3 : 1); ++repeat) {
                        int callbacks = 0;
                        Data replicaView;
                        replicaView.FakeFrom(input, 0);
                        replicaView.Resize(input.dims);
                        replicaView.dataDeviceIds = {device};
                        auto launchParallel = [&] {
                            ++callbacks;
                            Check(cudaMemsetAsync(parallelOutput.cudaData, 0, sizeof(float), cudaStreamPerThread));
                            // TP shared work may leave another GPU current.
                            if (dual) Check(cudaSetDevice(1 - device));
                        };
                        const bool accepted = repeat == 0
                            ? FastllmCudaMergeMOEHybrid(replicaView, index, scores, output,
                                weights[t].data(), weights[t].size(), t, launchParallel)
                            : FastllmCudaMergeMOEHybrid(input, index, scores, output,
                                weights[t].data(), weights[t].size(), t);
                        Require(accepted && callbacks == (repeat == 0 ? 1 : 0), "cache parallel handoff failed");
                        int current;
                        Check(cudaGetDevice(&current));
                        Require(current == device, "parallel callback changed cache output device");
                    }
                    float parallelValue = 1.0f;
                    Check(cudaMemcpy(&parallelValue, parallelOutput.cudaData, sizeof(float), cudaMemcpyDeviceToHost));
                    Require(parallelValue == 0.0f, "parallel GPU work did not complete");
                    Check(cudaMemcpy(actual.data(), output.cudaData, hidden * 2, cudaMemcpyDeviceToHost));
                    std::vector<float> values(hidden);
                    for (int c = 0; c < hidden; ++c)
                        values[c] = __bfloat162float(actual[c]);
                    Compare(values, expected, "V4.1 cache/oracle mismatch");
                }
            }
            // Full NUMA fallback must obey the same contract as the subset.
            for (int rows : (pass < 4 && pass % 4 != 1 ? std::vector<int>{1, 5, 33} : std::vector<int>{1})) {
                std::vector<float> batchInput, batchScores, batchExpected;
                std::vector<int32_t> batchIds;
                for (int row = 0; row < rows; ++row) {
                    batchInput.insert(batchInput.end(), x.begin(), x.end());
                    batchScores.insert(batchScores.end(), route.begin(), route.end());
                    batchExpected.insert(batchExpected.end(), expected.begin(), expected.end());
                    batchIds.insert(batchIds.end(), ids.begin(), ids.end());
                }
                std::vector<uint16_t> batchBf16(batchInput.size());
                for (size_t c = 0; c < batchInput.size(); ++c)
                    batchBf16[c] = Float32ToBFloat16RNEBits(batchInput[c]);
                Data cpuInput(BFLOAT16, {rows, hidden}, DataDevice::CPU, batchBf16.data());
                Data cpuIndex(INT32, {rows, topk}), cpuScore(FLOAT32, {rows, topk}, batchScores);
                cpuIndex.Allocate(false);
                memcpy(cpuIndex.cpuData, batchIds.data(), rows * topk * 4);
                Data w1, w2, w3, cpuResult;
                std::vector<Data *> biases(weights[t].size(), nullptr);
                if (pass == 0 && rows == 33) {
                    const auto cacheBytes = GetMoeCudaCacheBytes();
                    SetMoeCudaCacheBytes(0);
                    Data coldInput;
                    coldInput.CopyFrom(cpuInput);
                    coldInput.ToDevice(DataDevice::CUDA);
                    for (int repeat = 0; repeat < 2; ++repeat) {
                        Data coldOutput;
                        RunNumasMoe(coldInput, cpuIndex, cpuScore, coldOutput, coldWeights, t);
                        for (int id : ids)
                            for (int part = 0; part < 2; ++part)
                                Require(!coldWeights[2 * (id + 1) + part]->numasData.empty(),
                                        "cold NVFP4 GPU prefill skipped NUMA registration");
                        ToDataType(coldOutput, FLOAT32);
                        coldOutput.ToDevice(DataDevice::CPU);
                        Compare(std::vector<float>((float *)coldOutput.cpuData,
                                                   (float *)coldOutput.cpuData + rows * hidden),
                                batchExpected, "cold/warm NVFP4 GPU prefill/oracle mismatch");
                    }
                    SetMoeCudaCacheBytes(cacheBytes);
                }
                RunNumasMoe(cpuInput, cpuIndex, cpuScore, cpuResult, weights[t], t);
                ToDataType(cpuResult, FLOAT32);
                cpuResult.ToDevice(DataDevice::CPU);
                const auto cpuLabel =
                    "NUMA/oracle mismatch pass=" + std::to_string(pass) + " rows=" + std::to_string(rows);
                Compare(std::vector<float>((float *)cpuResult.cpuData, (float *)cpuResult.cpuData + rows * hidden),
                        batchExpected, cpuLabel.c_str());
                if (pass < 4 && pass % 4 != 1 && rows > 1) {
                    // Routing top-k is unique in model prefill. Exercise the
                    // legacy weight-streaming adapter before returning to cache.
                    Data gpuInput, gpuOutput(BFLOAT16, {rows, hidden});
                    gpuInput.CopyFrom(cpuInput);
                    gpuInput.ToDevice(DataDevice::CUDA);
                    Gpu(gpuOutput);
                    std::unordered_set<int> all;
                    for (int id : ids)
                        all.insert(id + 1);
                    DoCudaMergeMOEFromCPU(gpuInput, gpuOutput, cpuIndex, cpuScore, w1, w2, w3, weights[t].data(),
                                          biases.data(), 1, true, all, true, MoeGateSwiglu, true, 10, 32, false);
                    ToDataType(gpuOutput, FLOAT32);
                    gpuOutput.ToDevice(DataDevice::CPU);
                    Compare(std::vector<float>((float *)gpuOutput.cpuData, (float *)gpuOutput.cpuData + rows * hidden),
                            batchExpected, "V4.1 GPU prefill/oracle mismatch");
                }
            }
        }

        // Multirow verify uses a different scheduler and workspace. Compare
        // varied rows and duplicate routes against the independent dense
        // FP8/BF16 oracle, including partial and entirely resident GPU work.
        setenv("FASTLLM_DSV41_MOE_CACHE_PREFETCH", "0", 1);
        int verifyChecks = 0, admissionChecks = 0;
        Data verifyOutput;
        for (int pass = 0; pass < 21; ++pass) {
            const int t = pass % tables, rows = std::vector<int>{8, 2, 3, 4, 5, 6, 7}[pass % 7];
            const int base = pass * 7 % experts;
            std::vector<float> x(rows * hidden), route(rows * topk), expected(rows * hidden);
            std::vector<int32_t> ids(rows * topk);
            for (auto &v : x)
                v = Bf((int(rng() % 127) - 63) / 32.0f);
            x[0] = 48;
            x[hidden + 1] = -64;
            for (int row = 0; row < rows; ++row) {
                std::vector<float> q(x.begin() + row * hidden, x.begin() + (row + 1) * hidden);
                Quant(q);
                std::vector<float> perExpert(topk * hidden);
                for (int r = 0; r < topk; ++r) {
                    const int pos = row * topk + r;
                    ids[pos] = (base + (row * 3 + (r == 1 ? 2 : r)) % 10) % experts;
                    route[pos] = r == row % topk ? 0 : float(1 + rng() % 15) / 32;
                    const auto &g = dense[t][ids[pos] * 2], &d = dense[t][ids[pos] * 2 + 1];
                    std::vector<float> act(inter);
                    for (int o = 0; o < inter; ++o) {
                        float gate = 0, up = 0;
                        for (int c = 0; c < hidden; ++c) {
                            gate += q[c] * g[o * hidden + c];
                            up += q[c] * g[(o + inter) * hidden + c];
                        }
                        gate = std::min(Bf(gate), 10.0f);
                        up = std::clamp(Bf(up), -10.0f, 10.0f);
                        act[o] = Bf(route[pos] * ((gate / (1 + std::exp(-gate))) * up));
                    }
                    Quant(act);
                    for (int c = 0; c < hidden; ++c) {
                        float sum = 0;
                        for (int o = 0; o < inter; ++o)
                            sum += act[o] * d[c * inter + o];
                        perExpert[r * hidden + c] = Bf(sum);
                    }
                }
                std::vector<int> order(topk);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                                 [&](int a, int b) { return ids[row * topk + a] < ids[row * topk + b]; });
                for (int r : order)
                    for (int c = 0; c < hidden; ++c)
                        expected[row * hidden + c] += perExpert[r * hidden + c];
            }
            for (auto &v : expected)
                v = Bf(v);
            Data bx(FLOAT32, {rows, hidden}, x), bi(INT32, {rows, topk}), bs(FLOAT32, {rows, topk}, route);
            ToDataType(bx, BFLOAT16);
            bx.ToDevice(DataDevice::CPU);
            bi.Allocate(false);
            memcpy(bi.cpuData, ids.data(), ids.size() * 4);
            std::vector<int32_t> none(ids.size(), -1);
            std::vector<uint16_t> cpuResult(rows * hidden);
            NumasMoeVerifyExperts((uint16_t *)bx.cpuData, cpuResult.data(), rows, weights[t].data(), weights[t].size(),
                                  ids.data(), none.data(), route.data(), topk, t, 10, false);
            std::vector<float> actual(rows * hidden);
            for (size_t c = 0; c < actual.size(); ++c) {
                uint32_t bits = uint32_t(cpuResult[c]) << 16;
                memcpy(&actual[c], &bits, 4);
            }
            Compare(actual, expected, "verify CPU grouped/oracle mismatch");
            for (int device : (dual ? std::vector<int>{0, 1, 0} : std::vector<int>{0})) {
                bx.ToDevice(DataDevice::CUDA, {device}, true);
                bi.ToDevice(DataDevice::CUDA, {device}, true);
                bs.ToDevice(DataDevice::CUDA, {device}, true);
                input.ToDevice(DataDevice::CUDA, {device}, true);
                index.ToDevice(DataDevice::CUDA, {device}, true);
                scores.ToDevice(DataDevice::CUDA, {device}, true);
                FastllmCudaSetDevice(device);
                Data parallelOutput(FLOAT32, {1});
                parallelOutput.ToDevice(DataDevice::CUDA, {device}, false);
                parallelOutput.Allocate(false);
                cudaGraph_t parallelGraph;
                cudaGraphExec_t parallelExec;
                Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                Check(cudaMemsetAsync(parallelOutput.cudaData, 0, parallelOutput.GetBytes(), cudaStreamPerThread));
                Check(cudaStreamEndCapture(cudaStreamPerThread, &parallelGraph));
                Check(cudaGraphInstantiate(&parallelExec, parallelGraph, nullptr, nullptr, 0));
                auto runVerify = [&](bool parallel = true) {
                    int callbacks = 0;
                    Data replicaView;
                    replicaView.FakeFrom(bx, 0);
                    replicaView.Resize(bx.dims);
                    replicaView.dataDeviceIds = {device};
                    auto launchParallel = [&] {
                        ++callbacks;
                        Check(cudaGraphLaunch(parallelExec, cudaStreamPerThread));
                        if (dual) Check(cudaSetDevice(1 - device));
                    };
                    const bool accepted = FastllmCudaMergeMOEHybrid(replicaView, bi, bs, verifyOutput,
                        weights[t].data(), weights[t].size(), t,
                        parallel ? std::function<void()>(launchParallel) : std::function<void()>());
                    Require(callbacks == (accepted && parallel ? 1 : 0), "verify parallel handoff count mismatch");
                    int current;
                    Check(cudaGetDevice(&current));
                    Require(current == device, "verify callback changed cache output device");
                    return accepted;
                };
                auto checkResult = [&] {
                    Require(verifyOutput.dims == std::vector<int>({rows, hidden}), "verify output lost rows");
                    std::vector<__nv_bfloat16> result(rows * hidden);
                    Check(cudaMemcpy(result.data(), verifyOutput.cudaData, result.size() * 2, cudaMemcpyDeviceToHost));
                    for (size_t c = 0; c < actual.size(); ++c)
                        actual[c] = __bfloat162float(result[c]);
                    Compare(actual, expected, "verify mixed CPU/GPU/oracle mismatch");
                    ++verifyChecks;
                };
                setenv("FASTLLM_DSV41_MOE_CACHE_MODE", "hybrid", 1);
                setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", "0", 1);
                Require(!runVerify(),
                        "verify all-CPU fallback rejected");
                if (pass == 0) {
                    setenv("FASTLLM_DSV41_MOE_CACHE_PREFETCH", "1", 1);
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        if (runVerify()) {
                            checkResult();
                            ++admissionChecks;
                        }
                    }
                    setenv("FASTLLM_DSV41_MOE_CACHE_PREFETCH", "0", 1);
                }
                setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", "15", 1);
                if (runVerify())
                    checkResult();
                // Seed all ten experts through the ordinary pure-cache path;
                // every selected verify group must now be resident.
                setenv("FASTLLM_DSV41_MOE_CACHE_MODE", "gpu", 1);
                Check(cudaMemcpy(input.cudaData, bx.cudaData, hidden * 2, cudaMemcpyDeviceToDevice));
                Check(cudaMemcpy(scores.cudaData, bs.cudaData, topk * 4, cudaMemcpyDeviceToDevice));
                for (int offset : {0, 5}) {
                    std::vector<int32_t> warm(topk);
                    for (int r = 0; r < topk; ++r)
                        warm[r] = (base + (offset + r) % 10) % experts;
                    Check(cudaMemcpy(index.cudaData, warm.data(), topk * 4, cudaMemcpyHostToDevice));
                    Require(FastllmCudaMergeMOEHybrid(input, index, scores, output, weights[t].data(),
                                                      weights[t].size(), t),
                            "verify warmup rejected");
                }
                setenv("FASTLLM_DSV41_MOE_CACHE_MODE", "hybrid", 1);
                for (const char *split : {"1", "3", "15"}) {
                    setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", split, 1);
                    for (int repeat = 0; repeat < 2; ++repeat) {
                        Check(cudaMemsetAsync(parallelOutput.cudaData, 0xff, parallelOutput.GetBytes(), cudaStreamPerThread));
                        Require(runVerify(repeat == 0),
                                "resident verify rejected");
                        checkResult();
                        float parallelValue;
                        Check(cudaMemcpy(&parallelValue, parallelOutput.cudaData, sizeof(float), cudaMemcpyDeviceToHost));
                        Require(repeat == 0 ? parallelValue == 0 : std::isnan(parallelValue),
                                "verify parallel graph execution mismatch");
                    }
                }
                bx.dataType = FLOAT16;
                Require(!runVerify(),
                        "verify wrong dtype accepted");
                bx.dataType = BFLOAT16;
                bx.dims[0] = bi.dims[0] = bs.dims[0] = 9;
                Require(!runVerify(),
                        "verify oversized batch accepted");
                bx.dims[0] = bi.dims[0] = bs.dims[0] = rows;
                Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                const bool captureAccepted = runVerify();
                cudaGraph_t graph;
                Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                Check(cudaGraphDestroy(graph));
                Require(!captureAccepted, "verify capture accepted");
                Check(cudaGraphExecDestroy(parallelExec));
                Check(cudaGraphDestroy(parallelGraph));
            }
        }
        Require(admissionChecks > 0, "verify admission overlap was not exercised");
        std::printf("VERIFY_PASS %d multirow CPU/GPU oracle checks, %d CPU/admission overlaps\n", verifyChecks,
                    admissionChecks);
        unsetenv("FASTLLM_DSV41_MOE_CACHE_PREFETCH");
        unsetenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS");
        unsetenv("FASTLLM_DSV41_MOE_CACHE_MODE");
        FastllmCudaReleaseMoeCache(weights[0].data(), weights[0].size());
        Require(!FastllmCudaCanRunMoeHybrid(weights[0].data(), weights[0].size()), "released table retained");
        ClearNumasMoeRuntimeCache();
        for (auto &w : owned)
            w->numasData.clear();
        for (void *p : shards)
            Check(cudaFreeHost(p));
        SetMoeCudaCacheBytes(0);
        unsetenv("FASTLLM_MOE_CUDA_CACHE_BYTES_0");
        unsetenv("FASTLLM_MOE_CUDA_CACHE_BYTES_1");
        if (!verifyOnly)
            std::printf("ALL_PASS V4.1: independent FP8/BF16 oracle, 4096 FP4 code/scale cases, CPU subset oracle "
                        "checks, %d mode/split comparisons including queued pure GPU, cold/hit/eviction, duplicate and "
                        "zero routes, NUMA fallback%s\n",
                        dual ? 1008 : 336, dual ? ", repeated GPU 0/1/0 transitions" : "");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
