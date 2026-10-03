#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include "devices/numas/numasdevice.h"
#include "devices/numas/numas.h"
#include "../../src/devices/cuda/moe/fastllm-moe-glm5-cache.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sys/prctl.h>
namespace fastllm {
NumaConfig *GetNumaConfig();
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
static void Quant(std::vector<float> &v, int blockSize = 128) {
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
                       {"activationQuantBlock", 128}});
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
static void CheckUnsupportedCachePaths(std::vector<Data *> &weights, int hidden, int topk, bool dual) {
    FastllmCudaSetDevice(0);
    for (int rows : {2, 1}) {
        Data input(FLOAT32, {rows, hidden}), index(INT32, {rows, topk}), scores(FLOAT32, {rows, topk});
        Data output, gate;
        Gpu(input); Gpu(index); Gpu(scores);
        std::vector<int32_t> ids(rows * topk);
        std::iota(ids.begin(), ids.end(), 0);
        std::vector<float> route(rows * topk, 1.0f / topk);
        Check(cudaMemset(input.cudaData, 0, rows * hidden * sizeof(float)));
        Check(cudaMemcpy(index.cudaData, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(scores.cudaData, route.data(), route.size() * sizeof(float), cudaMemcpyHostToDevice));
        uint64_t before[5] = {}, after[5] = {};
        Require(fastllm_moe_cuda_cache_stats(0, before, false), "cache snapshot failed");
        int callbacks = 0;
        Require(!FastllmCudaMergeMOEHybrid(input, index, scores, output,
            weights.data(), weights.size(), 0, [&] { ++callbacks; }), "GLM FP32 hybrid accepted");
        if (dual) {
            auto ep = FastllmCudaCreateMoeExpertParallel(2);
            Require(ep != nullptr, "expert parallel context unavailable");
            std::exception_ptr errors[2];
            auto runRank = [&](int rank) {
                try {
                    FastllmCudaSetDevice(rank);
                    Data rankInput(FLOAT32, {rows, hidden}), rankOutput;
                    rankInput.dataDevice = DataDevice::CUDA;
                    rankInput.dataDeviceIds = {rank};
                    rankInput.Allocate(false);
                    Check(cudaMemset(rankInput.cudaData, 0, rows * hidden * sizeof(float)));
                    int calls = 0;
                    Require(!FastllmCudaMergeMOEExpertParallel(*ep, rank, rankInput, index, scores, rankOutput,
                        weights.data(), weights.size(), 0, [&] { ++calls; }), "GLM FP32 expert parallel accepted");
                    Require(calls == 0 && rankOutput.cudaData == nullptr && rankOutput.dims.empty(),
                            "rejected GLM EP call modified output or ran callback");
                } catch (...) {
                    errors[rank] = std::current_exception();
                }
            };
            std::thread peer(runRank, 1);
            runRank(0);
            peer.join();
            for (const auto &error : errors) if (error) std::rethrow_exception(error);
        }
        Require(!FastllmCudaMergeMOECache(input, gate, output, weights.data(), weights.size(),
            (const int32_t *)index.cudaData, (const float *)scores.cudaData, topk), "GLM generic cache accepted");
        Require(callbacks == 0 && output.cudaData == nullptr && output.dims.empty() &&
                gate.cudaData == nullptr && gate.dims.empty(), "rejected GLM call modified output or ran callback");
        Require(fastllm_moe_cuda_cache_stats(0, after, false), "cache snapshot failed");
        Require(std::equal(before, before + 5, after), "rejected GLM call changed cache");
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
        setenv("FT_GPU_PREFILL", "0", 1);
        setenv("FT_EXPERT_LIMIT", "0", 1);
        SetThreads(8);
        constexpr int hidden = 512, inter = 256, experts = 24, tables = 2, topk = 6;
        const int nodes = GetNumaConfig()->numaCnt;
        Require(inter % (128 * nodes) == 0, "test requires 1 or 2 NUMA nodes");
        std::vector<std::unique_ptr<Data>> owned;
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
                    auto w = std::make_unique<Data>(NVFP4_BLOCK_16_E4M3_PACKED);
                    w->Resize({rows, cols});
                    w->blockK = 1;
                    w->blockM = 16;
                    const size_t pitch = GetDataBytes(w->dataType, 1, cols);
                    std::vector<uint8_t> packed(rows * pitch);
                    auto &matrix = dense[t][e * 2 + part];
                    matrix.resize(rows * cols);
                    const float fp4[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
                    for (int row = 0; row < rows; ++row) {
                        // Distinct globals for gate, up and down; exact binary
                        // scales make the independent dense oracle reproducible.
                        const float global = part ? .5f : row < inter ? .75f : 1.25f;
                        memcpy(packed.data() + row * pitch, &global, sizeof(global));
                        for (int c = 0; c < cols; c += 16) {
                            uint8_t *block = packed.data() + row * pitch + 4 + c / 16 * 9;
                            block[8] = 24 + rng() % 8;
                            const float scale = std::ldexp(1 + (block[8] & 7) / 8.0f, (block[8] >> 3) - 7);
                            for (int b = 0; b < 8; ++b) {
                                block[b] = rng() % 256;
                                for (int half = 0; half < 2; ++half) {
                                    const int code = (block[b] >> (half * 4)) & 15;
                                    matrix[row * cols + c + b * 2 + half] =
                                        global * scale * (code & 8 ? -1 : 1) * fp4[code & 7];
                                }
                            }
                        }
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
            layers[t] = {weights[t].data(), (int)weights[t].size(), false, 10.0f, true};
        }
        const size_t record = ((weights[0][2]->GetBytes() + weights[0][3]->GetBytes() + 127) / 128) * 128;
        SetMoeCudaCacheBytes(0);
        Require(!FastllmCudaPrepareMoeCache(layers, tables, [] {}), "disabled cache accepted");
        SetMoeCudaCacheBytes(record);
        Require(!FastllmCudaPrepareMoeCache(layers, tables, [] {}), "undersized cache accepted");
        SetMoeCudaCacheBytes(record * 16);
        Require(!FastllmCudaPrepareMoeCache(layers, tables), "unowned NUMA cache accepted");
        Require(FastllmCudaPrepareMoeCache(layers, tables, [] {}), "GLM cache preparation failed");
        Require(FastllmCudaCanRunMoeHybrid(weights[0].data(), weights[0].size()), "GLM hybrid unavailable");
        Require(!FastllmCudaCanRunMoeCache(weights[0].data(), weights[0].size()), "generic math accepted GLM");
        CheckUnsupportedCachePaths(weights[0], hidden, topk, dual);
        Data input(BFLOAT16, {1, hidden}), index(INT32, {1, topk}), scores(FLOAT32, {1, topk}), output;
        Gpu(input); Gpu(index); Gpu(scores);
        auto supported = [&] {
            return FastllmCudaCanRunMoeCacheSmallBatch(input, index, scores, weights[0].data(), weights[0].size(), MoeGateSwiglu);
        };
        Require(supported(), "valid BF16 decode rejected");
        input.dataType = FLOAT32;
        Require(!supported(), "wrong activation math accepted");
        input.dataType = BFLOAT16;
        input.dims[0] = index.dims[0] = scores.dims[0] = 2;
        Require(!supported(), "GLM multi-token cache accepted");
        Require(!FastllmCudaMergeMOEHybrid(input, index, scores, output, weights[0].data(), weights[0].size(), 0),
                "GLM multi-token hybrid accepted");
        input.dims[0] = index.dims[0] = scores.dims[0] = 1;
        for (int pass = 0; pass < 24; ++pass) {
            const int t = pass % tables;
            std::vector<float> x(hidden), route(topk), expected(hidden, 0), perExpert(topk * hidden);
            std::vector<__nv_bfloat16> bx(hidden), actual(hidden);
            std::vector<int32_t> ids(topk);
            for (int c = 0; c < hidden; ++c) {
                // Within each block-128, varying magnitudes distinguish it
                // from the incorrect V4.1 block-32 quantization path.
                x[c] = Bf(std::ldexp((int(rng() % 79) - 39) / 16.0f, c / 32 % 4 - 2));
                bx[c] = __float2bfloat16(x[c]);
            }
            if (pass % 3 == 0) {
                x[0] = 48; x[1] = -64;
                bx[0] = __float2bfloat16(x[0]); bx[1] = __float2bfloat16(x[1]);
            }
            for (int r = 0; r < topk; ++r) {
                ids[r] = (pass / 4 * 7 + r * 5) % experts;
                route[r] = r == 0 ? 0 : (1 + rng() % 15) / 32.0f;
            }
            if (pass % 4 == 1) ids[1] = ids[2];
            for (int r = 0; r < topk; ++r) {
                const auto &g = dense[t][ids[r] * 2], &d = dense[t][ids[r] * 2 + 1];
                std::vector<float> activation(inter);
                for (int row = 0; row < inter; ++row) {
                    float gate = 0, up = 0;
                    for (int c = 0; c < hidden; ++c) {
                        gate += x[c] * g[row * hidden + c];
                        up += x[c] * g[(row + inter) * hidden + c];
                    }
                    gate = std::min(Bf(gate), 10.0f);
                    up = std::clamp(Bf(up), -10.0f, 10.0f);
                    activation[row] = Bf(route[r] * ((gate / (1 + std::exp(-gate))) * up));
                }
                Quant(activation);
                for (int row = 0; row < hidden; ++row) {
                    float sum = 0;
                    for (int c = 0; c < inter; ++c) sum += activation[c] * d[row * inter + c];
                    perExpert[r * hidden + row] = Bf(sum);
                }
            }
            std::vector<int> order(topk);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ids[a] < ids[b]; });
            for (int r : order)
                for (int c = 0; c < hidden; ++c) expected[c] += perExpert[r * hidden + c];
            for (auto &v : expected) v = Bf(v);
            for (int gpu : {0, 1, 3, 6}) {
                std::vector<int32_t> selected(topk, -1);
                for (int r = 0; r < gpu; ++r) selected[(r + pass) % topk] = ids[(r + pass) % topk];
                std::vector<float> cpuOutput(topk * hidden, 0), expectedCpu = perExpert;
                for (int r = 0; r < topk; ++r)
                    if (selected[r] >= 0) std::fill(expectedCpu.begin() + r * hidden, expectedCpu.begin() + (r + 1) * hidden, 0);
                NumasMoeDecodeExperts(x.data(), cpuOutput.data(), weights[t].data(), ids.data(), selected.data(), topk, t, route.data(), 10, 128);
                Compare(cpuOutput, expectedCpu, "CPU subset/oracle mismatch");
            }
            for (int device : (dual ? std::vector<int>{0, 1, 0} : std::vector<int>{0})) {
                input.ToDevice(DataDevice::CUDA, {device}, false);
                index.ToDevice(DataDevice::CUDA, {device}, false);
                scores.ToDevice(DataDevice::CUDA, {device}, false);
                FastllmCudaSetDevice(device);
                Check(cudaMemcpy(input.cudaData, bx.data(), hidden * 2, cudaMemcpyHostToDevice));
                Check(cudaMemcpy(index.cudaData, ids.data(), topk * 4, cudaMemcpyHostToDevice));
                Check(cudaMemcpy(scores.cudaData, route.data(), topk * 4, cudaMemcpyHostToDevice));
                int splitStep = 0;
                for (int split : {0, 1, 3, 6, 6, 0}) {
                    uint64_t before[8] = {}, after[8] = {}, again[8] = {};
                    Require(fastllm_moe_cuda_cache_route_stats(device, before), "route snapshot failed");
                    const auto value = std::to_string(split);
                    setenv("FASTLLM_GLM5_MOE_CACHE_GPU_EXPERTS", value.c_str(), 1);
                    int callbacks = 0;
                    Require(FastllmCudaMergeMOEHybrid(input, index, scores, output,
                        weights[t].data(), weights[t].size(), t, [&] {
                            ++callbacks;
                            if (dual) Check(cudaSetDevice(1 - device));
                        }), "GLM hybrid rejected");
                    Require(callbacks == 1, "parallel handoff failed");
                    int current;
                    Check(cudaGetDevice(&current));
                    Require(current == device, "callback changed output device");
                    Require(fastllm_moe_cuda_cache_route_stats(device, after), "route snapshot failed");
                    Require(fastllm_moe_cuda_cache_route_stats(device, again), "repeated snapshot failed");
                    for (int i = 0; i < 8; ++i) {
                        Require(after[i] == again[i] && after[i] >= before[i], "snapshot changed counters");
                        after[i] -= before[i];
                    }
                    Require(after[0] == 1 && after[1] == topk, "missing full-route calls");
                    Require(after[2] + after[3] == topk, "residency route total mismatch");
                    Require(after[4] == split && after[5] == topk - split, "CPU/GPU route total mismatch");
                    Require(after[6] == std::min<uint64_t>(split, after[2]), "resident GPU route mismatch");
                    // The previous all-GPU call loaded every selected expert.
                    // Cached routes must still count when all execution is CPU.
                    if (splitStep >= 4) Require(after[2] == topk && after[3] == 0, "warm CPU/GPU residency lost");
                    if (splitStep == 5) Require(after[4] == 0 && after[6] == 0 && after[2] == topk,
                                               "CPU-resident routes excluded from hit rate");
                    ++splitStep;
                    Check(cudaMemcpy(actual.data(), output.cudaData, hidden * 2, cudaMemcpyDeviceToHost));
                    std::vector<float> values(hidden);
                    for (int c = 0; c < hidden; ++c) values[c] = __bfloat162float(actual[c]);
                    Compare(values, expected, "GLM cache/oracle mismatch");
                }
            }
            Data cpuInput(BFLOAT16, {1, hidden}, DataDevice::CPU, bx.data());
            Data cpuIndex(INT32, {1, topk}), cpuScore(FLOAT32, {1, topk}, route), cpuResult;
            cpuIndex.Allocate(false);
            memcpy(cpuIndex.cpuData, ids.data(), topk * 4);
            RunNumasMoe(cpuInput, cpuIndex, cpuScore, cpuResult, weights[t], t);
            ToDataType(cpuResult, FLOAT32);
            cpuResult.ToDevice(DataDevice::CPU);
            Compare(std::vector<float>((float*)cpuResult.cpuData, (float*)cpuResult.cpuData + hidden), expected,
                    "NUMA fallback/oracle mismatch");
        }
        CheckUnsupportedCachePaths(weights[0], hidden, topk, dual);
        for (int device = 0; device < (dual ? 2 : 1); ++device) {
            uint64_t stats[5] = {};
            Require(fastllm_moe_cuda_cache_stats(device, stats, false), "cache stats unavailable");
            Require(stats[0] > 0 && stats[1] > 0 && stats[3] == 16, "cache hits/eviction not exercised");
        }
        FastllmCudaReleaseMoeCache(weights[0].data(), weights[0].size());
        ClearNumasMoeRuntimeCache();
        for (auto &w : owned) w->numasData.clear();
        for (void *p : shards) Check(cudaFreeHost(p));
        std::puts("GLM compact NVFP4 cache: CPU/mixed/GPU, oracle, fallback, eviction and device transitions passed.");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "GLM cache test failed: %s\n", e.what());
        return 1;
    }
}
