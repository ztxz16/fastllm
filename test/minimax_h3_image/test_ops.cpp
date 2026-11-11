#include "models/minimax_h3_image/model.h"
#include "models/minimax_h3_image/ops.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3_image;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    try {
        std::mt19937 random(29);
        std::normal_distribution<float> normal(0, 2);
        for (auto type : {DType::BF16, DType::FP16, DType::FP32}) {
            Runtime rt(0, type);
            // Odd widths cover scalar loads, aligned native FFN widths cover vector loads.
            for (auto shape : {std::pair<int, int>{1, 30}, {3, 64}, {9, 28672}, {4096, 28672}}) {
                std::vector<float> values(size_t(shape.first) * shape.second);
                for (auto &value : values)
                    value = normal(random);
                values[0] = -80;
                values[1] = 80;
                auto input = rt.upload(values.data(), shape.first, shape.second);
                auto expected = rt.download(rt.siluMultiply(rt.columns(input, 0, input.cols / 2),
                                                            rt.columns(input, input.cols / 2, input.cols / 2)));
                auto actual = rt.download(gatedSilu(input));
                if (actual.size() != expected.size())
                    throw std::runtime_error("gated SiLU output shape mismatch");
                for (size_t i = 0; i < actual.size(); ++i)
                    if (!std::isfinite(actual[i]) || actual[i] != expected[i])
                        throw std::runtime_error("gated SiLU changed activation rounding");
                std::cout << "dtype=" << int(type) << " rows=" << shape.first << " cols=" << shape.second
                          << " relative_l2=0\n";
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
