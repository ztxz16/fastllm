// Standalone regression (run from the repository root):
// hipcc -std=c++17 -O3 --offload-arch=gfx1100 -Iinclude -Ithird_party/json11
//   test/ops/rocmRouter.hip src/devices/rocm/fastllm-rocm-router.hip -o rocmRouter
// ./rocmRouter [snapshot.bin]
// An optional argument is a raw [tokens,256] FP16 input
// snapshot; the generated cases do not depend on any model download.
#include "devices/rocm/fastllm-rocm-router.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int Experts = 256;
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void Hip(hipError_t error) { Check(error == hipSuccess, hipGetErrorString(error)); }
template <typename T> struct Buffer {
    T *p = nullptr;
    explicit Buffer(size_t size) { Hip(hipMalloc(reinterpret_cast<void **>(&p), size * sizeof(T))); }
    ~Buffer() { (void)hipFree(p); }
    void Put(const std::vector<T> &values) {
        Hip(hipMemcpy(p, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    std::vector<T> Get(size_t size) {
        std::vector<T> result(size);
        Hip(hipMemcpy(result.data(), p, size * sizeof(T), hipMemcpyDeviceToHost));
        return result;
    }
};

// CPU reference uses a full std::sort, independent of the GPU sorting network.
template <typename T, typename B>
void Validate(const std::vector<T> &input, const std::vector<B> &bias,
        bool sigmoid, int topk, bool norm, float scale,
        const std::vector<int32_t> &ids, const std::vector<float> &scores) {
    for (size_t row = 0; row < input.size() / Experts; ++row) {
        std::vector<float> logits(Experts), probabilities(Experts), keys(Experts);
        float maximum = -INFINITY;
        for (int j = 0; j < Experts; ++j) {
            logits[j] = static_cast<float>(input[row * Experts + j]);
            maximum = std::fmax(maximum, logits[j]);
        }
        double sum = 0;
        for (int j = 0; j < Experts; ++j) {
            const float v = logits[j];
            probabilities[j] = sigmoid ? (std::isnan(v) ? 0.0f : 1.0f / (1.0f + std::exp(-v))) :
                (maximum == INFINITY ? (v == INFINITY ? 1.0f : 0.0f) :
                 (std::isfinite(maximum) && !std::isnan(v) ? std::exp(v - maximum) : 0.0f));
            sum += probabilities[j];
        }
        std::vector<int> order(Experts);
        std::iota(order.begin(), order.end(), 0);
        for (int j = 0; j < Experts; ++j) {
            if (!sigmoid) probabilities[j] = sum > 0 ? probabilities[j] / sum : 0;
            keys[j] = !sigmoid && bias.empty() ? logits[j] : probabilities[j];
            if (!bias.empty()) keys[j] += static_cast<float>(bias[j]);
            if (std::isnan(keys[j])) keys[j] = -INFINITY;
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return keys[a] > keys[b] || (keys[a] == keys[b] && a < b);
        });
        double selectedSum = norm ? 0 : 1;
        std::vector<bool> seen(Experts, false);
        for (int rank = 0; rank < topk; ++rank) {
            const int id = ids[row * topk + rank];
            Check(id >= 0 && id < Experts, "expert id out of bounds");
            Check(!seen[id], "duplicate expert id");
            seen[id] = true;
            const float actualKey = keys[id], expectedKey = keys[order[rank]];
            Check(actualKey == expectedKey || std::fabs(actualKey - expectedKey) < 2e-6f,
                  "incorrect Top-K selection");
            if (!sigmoid && bias.empty()) Check(id == order[rank], "unstable logit tie ordering");
            if (norm) selectedSum += probabilities[id];
        }
        if (!(selectedSum > 0)) selectedSum = 1;
        for (int rank = 0; rank < topk; ++rank) {
            const int id = ids[row * topk + rank];
            const float score = scores[row * topk + rank];
            Check(std::isfinite(score), "non-finite router score");
            Check(std::fabs(score - probabilities[id] / selectedSum * scale) < 2e-6,
                  "incorrect routing weight / normalization");
        }
    }
}

template <typename T, typename B>
void Run(const std::vector<T> &input, fastllm::DataType type,
        const std::vector<B> &bias, int biasType, bool sigmoid, bool norm,
        bool graph = false) {
    const int rows = input.size() / Experts, topk = sigmoid ? 10 : 8;
    const float scale = 1.75f;
    Buffer<T> x(input.size());
    Buffer<B> b(Experts);
    Buffer<int32_t> ids(rows * topk);
    Buffer<float> scores(rows * topk);
    x.Put(input);
    if (!bias.empty()) b.Put(bias);
    auto launch = [&]() {
        FastllmRocmFusedSelectExpert256(x.p, type, bias.empty() ? nullptr : b.p,
            biasType, ids.p, scores.p, rows, topk, sigmoid, norm, scale);
        Hip(hipGetLastError());
    };
    launch();
    Hip(hipDeviceSynchronize());
    Validate(input, bias, sigmoid, topk, norm, scale, ids.Get(rows * topk), scores.Get(rows * topk));
    if (graph) {
        hipGraph_t captured;
        hipGraphExec_t executable;
        Hip(hipStreamBeginCapture(hipStreamPerThread, hipStreamCaptureModeThreadLocal));
        launch();
        Hip(hipStreamEndCapture(hipStreamPerThread, &captured));
        Hip(hipGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        auto changed = input;
        for (int repeat = 0; repeat < 3; ++repeat) {
            std::reverse(changed.begin(), changed.end());
            x.Put(changed);
            Hip(hipGraphLaunch(executable, hipStreamPerThread));
            Hip(hipStreamSynchronize(hipStreamPerThread));
            Validate(changed, bias, sigmoid, topk, norm, scale,
                ids.Get(rows * topk), scores.Get(rows * topk));
        }
        Hip(hipGraphExecDestroy(executable));
        Hip(hipGraphDestroy(captured));
    }
}

template <typename T>
void Suite(const std::vector<float> &values, fastllm::DataType type) {
    std::vector<T> input(values.begin(), values.end());
    std::vector<float> floatBias(Experts);
    for (int j = 0; j < Experts; ++j) floatBias[j] = ((j * 37) % 29 - 14) * 0.017f;
    std::vector<half> halfBias(floatBias.begin(), floatBias.end());
    std::vector<hip_bfloat16> bf16Bias(floatBias.begin(), floatBias.end());
    for (bool sigmoid : {false, true}) for (bool norm : {false, true}) {
        Run(input, type, std::vector<float>{}, 0, sigmoid, norm, true);
        Run(input, type, floatBias, 0, sigmoid, norm);
        Run(input, type, halfBias, 1, sigmoid, norm);
        Run(input, type, bf16Bias, 2, sigmoid, norm);
    }
    std::cout << "dtype=" << static_cast<int>(type) << " rows=" << values.size() / Experts
              << " softmax/sigmoid, bias, normalization, graph replay PASS" << std::endl;
}
} // namespace

int main(int argc, char **argv) {
    try {
        std::mt19937 random(123);
        std::normal_distribution<float> normal(0, 3);
        for (int rows : {1, 17, 256, 513}) {
            std::vector<float> values(rows * Experts);
            for (int row = 0; row < rows; ++row) for (int j = 0; j < Experts; ++j) {
                float v = normal(random);
                switch (row % 8) {
                    case 1: v = 0; break;
                    case 2: v = (j % 11) * 0.5f; break;
                    case 3: v = j - 128; break;
                    case 4: v = 128 - j; break;
                    case 5: v = -5 + (j % 17) * 0.00390625f; break;
                    case 6: v = -INFINITY; break;
                    case 7: if (j < 3) v = j == 0 ? NAN : INFINITY; break;
                }
                values[row * Experts + j] = v;
            }
            Suite<half>(values, fastllm::DataType::FLOAT16);
            Suite<hip_bfloat16>(values, fastllm::DataType::BFLOAT16);
            Suite<float>(values, fastllm::DataType::FLOAT32);
        }
        if (argc > 1) {
            std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
            Check(file.good(), "cannot open FP16 snapshot");
            const auto bytes = file.tellg();
            Check(bytes > 0 && bytes % (Experts * sizeof(half)) == 0, "invalid FP16 snapshot shape");
            std::vector<half> snapshot(static_cast<size_t>(bytes) / sizeof(half));
            file.seekg(0);
            file.read(reinterpret_cast<char *>(snapshot.data()), bytes);
            Check(file.good(), "cannot read FP16 snapshot");
            Run(snapshot, fastllm::DataType::FLOAT16, std::vector<float>{}, 0, false, true, true);
            std::cout << "captured model input PASS" << std::endl;
        }
        std::cout << "ROCm router regression PASS" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
