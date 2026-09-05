#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t state) {
    if (state != cudaSuccess) throw std::runtime_error(cudaGetErrorString(state));
}
static void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static void AllocateGpu(fastllm::Data &data) {
    data.dataDevice = fastllm::DataDevice::CUDA;
    data.dataDeviceIds = {0};
    data.Allocate(false);
}

template<class T>
static void Run(fastllm::DataType dtype, int hidden, int inter) {
    constexpr int experts = 32, topk = 10;
    std::vector<std::unique_ptr<fastllm::Data>> owned;
    std::vector<fastllm::Data *> weights(2 * (experts + 1), nullptr);
    for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
        auto weight = std::make_unique<fastllm::Data>(fastllm::DataType::NVFP4_BLOCK_16_E4M3);
        int rows = part == 0 ? 2 * inter : hidden;
        int cols = part == 0 ? hidden : inter;
        weight->blockK = 1; weight->blockM = 16;
        weight->Resize({rows, cols}); weight->Allocate(false);
        size_t packed = size_t(rows) * ((cols + 1) / 2);
        // Every E2M1 value is 1, every E4M3 block scale is 1. Distinct
        // per-expert down scales expose wrong table/slot selection.
        std::memset(weight->cpuData, 0x22, packed);
        std::memset(weight->cpuData + packed, 0x38, weight->GetBytes() - packed);
        weight->scales = part == 0 ? std::vector<float>{1, 1}
                                  : std::vector<float>{float(e + 1) / 32};
        weights[2 * (e + 1) + part] = weight.get();
        owned.push_back(std::move(weight));
    }
    FastllmCudaMoeCacheLayer layer{weights.data(), int(weights.size())};
    fastllm::SetMoeCudaCacheBytes(0);
    Require(!FastllmCudaPrepareMoeCache(&layer, 1), "disabled cache prepared");
    fastllm::SetMoeCudaCacheBytes(1);
    Require(!FastllmCudaPrepareMoeCache(&layer, 1), "insufficient budget accepted");
    size_t downOffset = (weights[2]->GetBytes() + 15) / 16 * 16;
    size_t scalesOffset = (downOffset + weights[3]->GetBytes() + 15) / 16 * 16;
    size_t stride = (scalesOffset + 3 * sizeof(float) + 127) / 128 * 128;
    fastllm::SetMoeCudaCacheBytes(16 * stride);
    Require(FastllmCudaPrepareMoeCache(&layer, 1), "valid table rejected");

    fastllm::Data input(dtype, {1, hidden}), index(fastllm::DataType::INT32, {1, topk});
    fastllm::Data score(fastllm::DataType::FLOAT32, {1, topk}), gate, output;
    AllocateGpu(input); AllocateGpu(index); AllocateGpu(score);
    std::vector<T> activation(hidden, T(1.0f / 128));
    std::vector<float> scores(topk, 1.0f / topk);
    Check(cudaMemcpy(input.cudaData, activation.data(), hidden * sizeof(T), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(score.cudaData, scores.data(), topk * sizeof(float), cudaMemcpyHostToDevice));
    auto supported = [&] {
        return FastllmCudaCanRunMoeCacheBatch1(input, index, score, weights.data(),
                                              weights.size(), fastllm::MoeGateSwiglu);
    };
    Require(supported(), "valid decode rejected");
    input.dims[1] = hidden + 1;
    Require(!supported(), "mismatched hidden width accepted");
    input.dims[1] = hidden;

    auto launch = [&] {
        Require(FastllmCudaMergeMOECacheBatch1(input, gate, output, weights.data(),
                    weights.size(), static_cast<int32_t *>(index.cudaData),
                    static_cast<float *>(score.cudaData), topk), "decode failed");
    };
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    for (int pass = 0; pass < 5; ++pass) {
        std::vector<int32_t> ids(topk);
        for (int k = 0; k < topk; ++k) ids[k] = (pass * 9 + k) % experts;
        Check(cudaMemcpy(index.cudaData, ids.data(), topk * sizeof(int32_t), cudaMemcpyHostToDevice));
        if (pass == 0) {
            launch(); Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            launch();
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        }
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<T> actual(hidden);
        Check(cudaMemcpy(actual.data(), output.cudaData, hidden * sizeof(T), cudaMemcpyDeviceToHost));
        float g = float(T(hidden / 128.0f));
        float activated = float(T((g / (1 + std::exp(-g))) * g));
        float expected = 0;
        for (int id : ids) expected += float(T(inter * activated * ((id + 1) / 32.0f))) / topk;
        expected = float(T(expected));
        float tolerance = dtype == fastllm::DataType::FLOAT32 ? 1e-5f :
                          dtype == fastllm::DataType::FLOAT16 ? 0.002f : 0.012f;
        for (T v : actual) Require(std::fabs(float(v) - expected) <= tolerance * std::max(1.0f, std::fabs(expected)),
                                  "expert output mismatch");
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    FastllmCudaReleaseMoeCache(weights.data(), weights.size());
    Require(!supported(), "released table still registered");
    FastllmCudaReleaseMoeCache(weights.data(), weights.size());
    fastllm::SetMoeCudaCacheBytes(0);
    std::printf("PASS adapter dtype=%d hidden=%d inter=%d, graph/eviction/release/validation\n", int(dtype), hidden, inter);
}

int main() {
    try {
        for (bool wide : {false, true}) {
            int hidden = wide ? 128 : 19, inter = wide ? 128 : 23;
            Run<float>(fastllm::DataType::FLOAT32, hidden, inter);
            Run<half>(fastllm::DataType::FLOAT16, hidden, inter);
            Run<__nv_bfloat16>(fastllm::DataType::BFLOAT16, hidden, inter);
        }
        std::puts("ALL_PASS"); return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
