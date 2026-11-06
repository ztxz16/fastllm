#include "distributed/qwen3_generation.h"
#include "json11.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace fastllm::distributed;
using json11::Json;

/**
 * 用确定的概率边界验证正式 CPU 采样入口，不启动 worker 或 CUDA。
 * @return 所有黄金输出和拒绝路径符合预期时为 0，否则为 1；输出逐例 JSON。
 */
int main() {
    Json::array cases;
    bool passed = true;
    auto check = [&](const std::string &name, const std::vector<float> &logits,
                     Qwen3GenerationOptions options, double draw, int expected) {
        Json::object row{{"case", name}, {"expected", expected}, {"backend", "coordinator_cpu_sampler"}};
        bool correct = false;
        try {
            auto token = SampleQwen3(logits, options, draw);
            row["actual"] = double(token);
            correct = expected >= 0 && token == uint32_t(expected);
        } catch (const std::exception &error) {
            row["error"] = error.what();
            correct = expected < 0;
        }
        row["status"] = correct ? "PASS" : "FAIL";
        row["max_abs_error"] = correct ? 0 : 1;
        row["mean_abs_error"] = correct ? 0 : 1;
        row["failed_elements"] = correct ? 0 : 1;
        if (!correct) row["reason"] = "sampled ID/rejection differs from golden expectation";
        cases.push_back(std::move(row));
        passed = passed && correct;
    };
    Qwen3GenerationOptions options;
    check("greedy_tie", {1, 3, 3, -2}, options, 0.9, 1);
    check("greedy_negative", {-5, -2, -4}, options, 0, 1);
    options.temperature = 1;
    check("uniform_left", {0, 0, 0, 0}, options, 0, 0);
    check("uniform_boundary", {0, 0, 0, 0}, options, 0.25, 1);
    check("uniform_below_boundary", {0, 0, 0, 0}, options, std::nextafter(0.25, 0.0), 0);
    check("uniform_right", {0, 0, 0, 0}, options, std::nextafter(1.0, 0.0), 3);
    options.topP = 0.5;
    check("top_p_boundary", {0, 0, 0, 0}, options, 0.75, 1);
    options.topP = std::nextafter(0.5, 1.0);
    check("top_p_above_boundary", {0, 0, 0, 0}, options, 0.9, 2);
    options.topP = std::nextafter(0.5, 0.0);
    check("top_p_below_boundary", {0, 0, 0, 0}, options, 0.9, 1);
    options.topP = 1;
    options.topK = 2;
    check("top_k", {0, 3, 2, 1}, options, 0.9, 2);
    options.topP = 0.5;
    check("top_k_then_p", {0, 3, 2, 1}, options, 0.99, 1);
    options.topP = 1;
    options.topK = 999;
    check("top_k_larger_than_vocab", {0, 0}, options, 0.75, 1);
    options.temperature = 1e-300;
    check("tiny_temperature", {-3.4e38f, 3.4e38f}, options, 0.99, 1);
    options = Qwen3GenerationOptions{};
    check("empty", {}, options, 0, -1);
    check("nan_logits", {std::numeric_limits<float>::quiet_NaN()}, options, 0, -1);
    check("inf_logits", {std::numeric_limits<float>::infinity()}, options, 0, -1);
    check("draw_one", {0}, options, 1, -1);
    check("draw_negative", {0}, options, -0.1, -1);
    check("draw_nan", {0}, options, std::numeric_limits<double>::quiet_NaN(), -1);
    options.temperature = -1;
    check("negative_temperature", {0}, options, 0, -1);
    options.temperature = std::numeric_limits<double>::infinity();
    check("infinite_temperature", {0}, options, 0, -1);
    options.temperature = 0;
    options.topP = 0;
    check("zero_top_p", {0}, options, 0, -1);
    options.topP = 1.01;
    check("large_top_p", {0}, options, 0, -1);
    options.topP = std::numeric_limits<double>::quiet_NaN();
    check("nan_top_p", {0}, options, 0, -1);
    std::cout << Json(Json::object{{"status", passed ? "PASS" : "FAIL"}, {"cases", cases}}).dump() << std::endl;
    return passed ? 0 : 1;
}
