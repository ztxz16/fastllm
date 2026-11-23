#include "tensor.h"
#include <cmath>
#include <iostream>

using namespace fastllm::duo::native;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    try {
        DeviceGuard guard(0);
        auto upload = [](std::vector<float> values, Shape shape) {
            return from_blob(values.data(), shape, kFloat).to(0, kBFloat16);
        };
        auto near = [](const Tensor &tensor, const std::vector<float> &expected, float tolerance) {
            auto host = tensor.to(kCPU, kFloat).contiguous();
            if (host.numel() != static_cast<int64_t>(expected.size()))
                throw std::runtime_error("unexpected output size");
            for (size_t i = 0; i < expected.size(); ++i)
                if (!(std::abs(host.data_ptr<float>()[i] - expected[i]) <= tolerance))
                    throw std::runtime_error("unexpected value at index " + std::to_string(i));
        };
        std::vector<int64_t> ids{1, INT64_MAX, (int64_t(1) << 57) + 1, -100000001};
        auto transposed = tensor(ids).to(0).reshape({2, 2}).transpose(0, 1).contiguous().cpu();
        auto *indices = transposed.data_ptr<int64_t>();
        if (indices[0] != ids[0] || indices[1] != ids[2] || indices[2] != ids[1] || indices[3] != ids[3])
            throw std::runtime_error("strided copy changed an integer");
        auto x = upload({1, 2, 3, 4, 5, 6}, {2, 3});
        near(x.transpose(0, 1).reshape({6}), {1, 4, 2, 5, 3, 6}, 0);
        near(x + upload({3, 2, 1}, {3}), {4, 4, 4, 7, 7, 7}, 0);
        near(linear(x, upload({1, 0, 1, 0, 1, -1}, {2, 3}), upload({2, 1}, {2})), {6, 0, 12, 0}, 0);

        // Two new queries attend after one cached token. Each query head
        // shares its KV head, with a bottom-right aligned causal mask.
        auto q = zeros({1, 2, 2, 64}, {kBFloat16, 0}), k = zeros({1, 1, 3, 64}, q.options());
        std::vector<float> values(3 * 64);
        for (int token = 0; token < 3; ++token)
            std::fill_n(values.begin() + token * 64, 64, float(1 + 2 * token));
        auto a = attention(q, k, upload(values, {1, 1, 3, 64}), true);
        std::vector<float> expected(2 * 2 * 64);
        for (int row = 0; row < 4; ++row)
            std::fill_n(expected.begin() + row * 64, 64, row % 2 ? 3.f : 2.f);
        near(a, expected, 0.015625f);

        std::vector<float> pcm(800, 0.f);
        pcm[320] = 1.f;
        auto power = melPower(pcm, 0).cpu();
        if (power.sizes() != Shape{201, 5})
            throw std::runtime_error("invalid spectrogram shape");
        for (int bin = 0; bin < 201; ++bin)
            if (std::abs(power.data_ptr<float>()[bin * 5 + 2] - 1.f) > 2e-5f)
                throw std::runtime_error("incorrect Hann-window impulse power");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
