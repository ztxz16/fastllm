#include "models/minimax_h3_image/model.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3_image;

static void compare(const std::string &name, const std::vector<float> &expected,
                    const std::vector<float> &actual) {
    if (expected.size() != actual.size())
        throw std::runtime_error("shape mismatch: " + name);
    double delta = 0, norm = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            throw std::runtime_error("nonfinite output: " + name);
        delta += std::pow(double(actual[i]) - expected[i], 2);
        norm += double(expected[i]) * expected[i];
    }
    double error = std::sqrt(delta / std::max(norm, 1e-30));
    std::cout << name << " relative_l2=" << error << '\n';
    if (error > 1e-2)
        throw std::runtime_error("VAE batch exceeds relative L2 tolerance: " + name);
}

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("usage: minimax_h3_image_decode MODEL");
        Runtime rt(0), full(0, true);
        VAE reference(rt, full, argv[1]), batched(rt, full, argv[1]);
        std::mt19937 random(42);
        std::normal_distribution<float> normal(0, 0.5f);
        for (auto shape : {std::array<int, 3>{16, 16, 2}, {5, 11, 3}, {16, 16, 8},
                           {16, 16, 128}, {5, 11, 2}}) {
            int h = shape[0], w = shape[1], batch = shape[2], rows = h * w;
            std::vector<float> input(size_t(rows) * batch * 24), expected;
            for (auto &value : input)
                value = normal(random);
            auto latent = full.upload(input.data(), batch * rows, 24);
            std::map<std::string, std::vector<float>> stages;
            for (int i = 0; i < batch; ++i) {
                auto trace = [&](const std::string &name, const Tensor &x) {
                    if (batch > 3)
                        return;
                    auto values = full.download(x);
                    stages[name].insert(stages[name].end(), values.begin(), values.end());
                };
                auto values = full.download(reference.decodeTile(latent.rowSlice(i * rows, rows), h, w, trace));
                expected.insert(expected.end(), values.begin(), values.end());
            }
            std::string prefix = std::to_string(h) + "x" + std::to_string(w) + "/" + std::to_string(batch);
            auto actual = batched.decodeTiles(latent, h, w, batch, [&](const std::string &name, const Tensor &x) {
                if (batch <= 3)
                    compare(prefix + "/" + name, stages.at(name), full.download(x));
            });
            compare(prefix + "/output", expected, full.download(actual));
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
