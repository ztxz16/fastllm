#include "models/duo/model.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <random>

using namespace fastllm::duo;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 77;
    try {
        Runtime rt(0, DType::BF16, {true, false});
        rt.activate();
        std::mt19937 rng(73);
        std::normal_distribution<float> normal;
        auto values = [&](size_t count) {
            std::vector<float> data(count);
            for (auto &value : data)
                value = normal(rng);
            return data;
        };
        auto tensor = [&](int rows, int cols) {
            auto data = values(size_t(rows) * cols);
            return rt.upload(data.data(), rows, cols);
        };
        const auto qw = tensor(1, 128), kw = tensor(1, 128);
        for (auto [rows, heads] : {std::pair{1, 3}, std::pair{107, 8}, std::pair{1019, 56}}) {
            auto packed = tensor(rows, heads * 3 * 128);
            auto [cs, sn] = rt.ropeTable(values(rows * 3), 16, 10000.f);
            for (bool rope : {false, true}) {
                auto actual = rt.unpackHeadQkvNormRotary128(packed, qw, kw, 1e-5f,
                                                            rope ? &cs : nullptr, rope ? &sn : nullptr);
                for (int part = 0; part < 3; ++part) {
                    auto expected = splitQkv(rt, packed, heads, 128, part);
                    if (part != 2) {
                        expected = rt.rms(expected, part == 0 ? qw : kw, 1e-5f, 128);
                        if (rope)
                            expected = rt.rotary(expected, cs, sn, 128);
                    }
                    if (rt.download(expected) != rt.download(actual[part]))
                        throw std::runtime_error("fused Duo QKV differs from the separate operations");
                }
            }
        }
        std::cout << "Duo QKV layout, norm and rotary matched\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
