#include "fastllm.h"
#include "executor.h"
#include "devices/numas/numasdevice.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using namespace fastllm;

// Fixed dimensions match a Naive expert. The expert count and routing pattern
// vary independently of the token count; setup and initial packing are untimed.
int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "Usage: numas_fp8_moe_bench ROWS ROUNDS [EXPERTS=8] [SHARED=8]\n");
        return 2;
    }
    const int rows = std::atoi(argv[1]), rounds = std::atoi(argv[2]);
    const int experts = argc > 3 ? std::atoi(argv[3]) : 8;
    const int shared = argc > 4 ? std::atoi(argv[4]) : 8;
    constexpr int hidden = 4096, inter = 2048, topk = 8;
    if (rows < 1 || rounds < 1 || experts < topk || shared < 0 || shared > topk) return 2;
    auto &executor = *(Executor *)GetExecutor();
    executor.SetFirstDevice("numa");
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights(2 * (experts + 1), nullptr), biases(weights);
    for (int expert = 1; expert <= experts; ++expert) for (int part = 0; part < 2; ++part) {
        int r = part ? hidden : inter * 2, c = part ? inter : hidden;
        auto weight = std::make_unique<Data>(FP8_E4M3, std::vector<int>{r, c});
        weight->name = "benchmark.expert." + std::to_string(expert) + "." + std::to_string(part);
        weight->blockK = weight->blockM = 128;
        weight->scales.assign((r / 128) * (c / 128), .001f);
        weight->Allocate();
        unsigned random = 42 + expert * 67 + part;
        for (size_t i = 0; i < weight->GetBytes(); ++i) {
            random = random * 1664525u + 1013904223u;
            weight->cpuData[i] = ((random >> 24) % 111 + 8) | ((random >> 23) & 128);
        }
        weights[expert * 2 + part] = weight.get();
        owned.push_back(std::move(weight));
    }
    Data input(BFLOAT16, {rows, hidden}), ids(INT32, {rows, topk}), scores(FLOAT32, {rows, topk});
    input.Allocate(); ids.Allocate(); scores.Allocate();
    for (int i = 0; i < rows * hidden; ++i)
        ((uint16_t *)input.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin(i * .37f));
    std::mt19937 random(42);
    std::vector<int> candidates(experts - shared);
    std::iota(candidates.begin(), candidates.end(), shared);
    for (int row = 0; row < rows; ++row) {
        std::shuffle(candidates.begin(), candidates.end(), random);
        for (int j = 0; j < topk; ++j) {
            ((int *)ids.cpuData)[row * topk + j] = j < shared ? j : candidates[j - shared];
            ((float *)scores.cpuData)[row * topk + j] = 1.0f / topk;
        }
    }
    Data output(BFLOAT16), w1, w2, w3, curInput, curOutput;
    std::vector<double> elapsed;
    for (int i = 0; i < rounds + 3; ++i) {
        auto start = std::chrono::steady_clock::now();
        executor.Run("MergeMOE", {{"input", &input}, {"index", &ids}, {"score", &scores},
            {"weights", (Data *)weights.data()}, {"biass", (Data *)biases.data()},
            {"w1", &w1}, {"w2", &w2}, {"w3", &w3}, {"curInput", &curInput},
            {"curOutput", &curOutput}, {"output", &output}}, {{"sharedScale", 0}},
            {{"weights___batch", (int)weights.size()}, {"biass___batch", (int)biases.size()}, {"fp8EagerMode", 1}});
        if (i >= 3) elapsed.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    double mean = std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size();
    std::sort(elapsed.begin(), elapsed.end());
    std::printf("FP8_MOE_RESULT {\"rows\":%d,\"rounds\":%d,\"experts\":%d,\"shared\":%d,"
                "\"mean_ms\":%.6f,\"median_ms\":%.6f}\n", rows, rounds, experts, shared, mean, elapsed[elapsed.size() / 2]);
    ClearNumasMoeRuntimeCache();
}
