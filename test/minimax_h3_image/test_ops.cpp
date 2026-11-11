#include "models/minimax_h3_image/model.h"
#include "models/minimax_h3_image/ops.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3_image;

static float rounded(float x, DType type) {
    if (type == DType::FP32)
        return x;
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
    std::memcpy(&x, &bits, sizeof(bits));
    return x;
}

static void checkStitch(DType type, int h, int w, int overlap, int ny, int nx) {
    Runtime rt(0, type), full(0, true);
    const int height = h * ny - overlap * (ny - 1), width = w * nx - overlap * (nx - 1);
    std::mt19937 random(19);
    std::normal_distribution<float> normal(0, 3);
    std::vector<Tensor> tiles;
    std::vector<std::vector<float>> values;
    for (int i = 0; i < ny * nx; ++i) {
        std::vector<float> x(size_t(h) * w * 3);
        for (auto &v : x)
            v = normal(random);
        tiles.push_back(rt.upload(x.data(), h / 16 * (w / 16), 768));
        values.push_back(rt.download(tiles.back()));
    }
    auto pixel = [&](int tile, int y, int x, int c) {
        return values[tile][(size_t(y / 16) * (w / 16) + x / 16) * 768 + c * 256 + y % 16 * 16 + x % 16];
    };
    auto round = [&](float x) { return rounded(x, type); };
    auto blend = [&](float a, float b, int i) {
        float alpha = round(round(float(i)) / overlap);
        return round(round(a * round(1.f - alpha)) + round(b * alpha));
    };
    Tensor rgba(height, width, DType::FP32);
    const float zero = 0;
    auto invalid = full.upload(&zero, 1, 1);
    std::vector<uint8_t> expected(size_t(height) * width * 4), actual(expected.size());
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix) {
            int index = iy * nx + ix;
            ImageTile s{h,
                        w,
                        iy * (h - overlap),
                        ix * (w - overlap),
                        iy ? overlap : 0,
                        ix ? overlap : 0,
                        h - (iy + 1 < ny ? overlap : 0),
                        w - (ix + 1 < nx ? overlap : 0)};
            stitchImageTile(tiles[index], iy ? &tiles[index - nx] : nullptr, ix ? &tiles[index - 1] : nullptr,
                            s, rgba, invalid);
            for (int y = 0; y < s.copyHeight; ++y)
                for (int x = 0; x < s.copyWidth; ++x) {
                    size_t dst = (size_t(s.y + y) * width + s.x + x) * 4;
                    const float mean[] = {.485f, .456f, .406f}, scale[] = {.229f, .224f, .225f};
                    for (int c = 0; c < 3; ++c) {
                        float v = pixel(index, y, x, c);
                        if (iy && y < overlap)
                            v = blend(pixel(index - nx, h - overlap + y, x, c), v, y);
                        if (ix && x < overlap)
                            v = blend(pixel(index - 1, y, w - overlap + x, c), v, x);
                        expected[dst + c] = uint8_t(std::clamp(v * scale[c] + mean[c], 0.f, 1.f) * 255.f);
                    }
                    expected[dst + 3] = 255;
                }
        }
    rt.synchronize();
    if (cudaMemcpy(actual.data(), rgba.data, actual.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
        actual != expected || full.download(invalid)[0] != 0)
        throw std::runtime_error("GPU tile stitching changed pixels");
    ImageTile spec{h, w, 0, 0, 0, 0, h, w};
    // The sentinel must propagate even when RGB clamping would hide a NaN.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto bad = full.upload(&nan, 1, 1);
    auto poisoned = full.cast(tiles[0], DType::FP32);
    cudaMemcpyAsync(poisoned.data, bad.data, sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    stitchImageTile(poisoned, nullptr, nullptr, spec, rgba, invalid);
    if (full.download(invalid)[0] == 0)
        throw std::runtime_error("GPU tile stitching lost nonfinite check");
    spec.overlapX = 1;
    bool rejected = false;
    try {
        stitchImageTile(tiles[0], nullptr, nullptr, spec, rgba, invalid);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("GPU tile stitching accepted a missing neighbour");
    std::cout << "stitch dtype=" << int(type) << " tile=" << h << 'x' << w << " grid=" << ny << 'x' << nx
              << " rgba_bitwise=1\n";
}

static void checkPackedQkv(int rows, int heads, bool rotary) {
    Runtime rt(0);
    std::mt19937 random(39);
    std::normal_distribution<float> normal(0, .5f);
    std::vector<float> values(size_t(rows) * heads * 384), weight(128), positions(size_t(rows) * 3);
    for (auto &v : values)
        v = normal(random);
    for (auto &v : weight)
        v = 1 + normal(random);
    for (auto &v : positions)
        v = normal(random) * 32;
    auto qw = rt.upload(weight.data(), 1, 128);
    for (auto &v : weight)
        v = 1 + normal(random);
    auto kw = rt.upload(weight.data(), 1, 128);
    auto packed = rt.upload(values.data(), rows, heads * 384);
    auto q = rt.columns(packed, 0, heads * 128), k = rt.columns(packed, heads * 128, heads * 128);
    auto v = rt.columns(packed, heads * 256, heads * 128);
    auto rope = rt.ropeTable(positions, 16, 10000);
    if (rotary) {
        auto result = rt.qkNormRotary(q, k, qw, kw, 1e-5f, rope.first, rope.second, 128);
        q = result.first;
        k = result.second;
    } else {
        q = rt.rms(q, qw, 1e-5f, 128);
        k = rt.rms(k, kw, 1e-5f, 128);
    }
    qkvNormRotaryInPlace(packed, qw, kw, rotary ? &rope.first : nullptr, rotary ? &rope.second : nullptr);
    auto compareQk = [&](const Tensor &reference, const Tensor &actual) {
        auto a = rt.download(reference), b = rt.download(actual);
        double error = 0, norm = 0;
        if (a.size() != b.size())
            throw std::runtime_error("packed QK shape mismatch");
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(b[i]))
                throw std::runtime_error("nonfinite packed QK");
            error += std::pow(double(a[i]) - b[i], 2);
            norm += double(a[i]) * a[i];
        }
        double l2 = std::sqrt(error / std::max(norm, 1e-30));
        if (l2 > 1e-2)
            throw std::runtime_error("packed QK exceeds relative L2 tolerance");
        return l2;
    };
    double qError = compareQk(q, rt.columns(packed, 0, heads * 128));
    double kError = compareQk(k, rt.columns(packed, heads * 128, heads * 128));
    if (rt.download(v) != rt.download(rt.columns(packed, heads * 256, heads * 128)))
        throw std::runtime_error("packed QKV changed V");
    if (rotary) {
        for (int groups : {1, 4, 7, 8}) {
            if (heads % groups)
                continue;
            auto original = rt.upload(values.data(), rows, heads * 384);
            auto kv = prepareSequenceQkv(original, qw, kw, rope.first, rope.second, groups);
            compareQk(q, rt.columns(original, 0, heads * 128));
            int width = heads / groups, first = rows / 2, local = rows - first;
            // Fill only the last rank's destination slots, including a padded
            // final row. Communication must never change Q/K/V ordering.
            auto shard = rt.upload(values.data() + size_t(first) * heads * 384, local, heads * 384);
            auto slots = prepareSequenceQkv(shard, qw, kw, rope.first.rowSlice(first, local),
                                            rope.second.rowSlice(first, local), groups, rows + 1, first);
            for (int g = 0; g < groups; ++g)
                compareQk(kv.rowSlice(g * rows + first, local),
                          slots.rowSlice(g * (rows + 1) + first, local));
            Tensor result(local, heads * 128, original.dtype);
            for (int g = 0; g < groups; ++g) {
                auto part = kv.rowSlice(g * rows, rows);
                compareQk(rt.columns(k, g * width * 128, width * 128), rt.columns(part, 0, width * 128));
                if (rt.download(rt.columns(v, g * width * 128, width * 128)) !=
                    rt.download(rt.columns(part, width * 128, width * 128)))
                    throw std::runtime_error("sequence QKV changed V or head order");
                auto output =
                    sequenceAttention(original.rowSlice(first, local), part, heads, g * width, width);
                copyAttentionHeads(output, result, g * width);
            }
            compareQk(rt.selfAttention(q, k, v, heads).rowSlice(first, local), result);
        }
    }
    auto expected = rt.download(rt.selfAttention(q, k, v, heads));
    auto actual = rt.download(rt.packedSelfAttention(packed, heads));
    double error = 0, norm = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]))
            throw std::runtime_error("nonfinite packed attention output");
        error += std::pow(double(actual[i]) - expected[i], 2);
        norm += std::pow(double(expected[i]), 2);
    }
    error = std::sqrt(error / std::max(norm, 1e-30));
    if (error > 1e-2)
        throw std::runtime_error("packed attention exceeds relative L2 tolerance");
    std::cout << "packed_qkv rows=" << rows << " heads=" << heads << " rotary=" << rotary
              << " q_relative_l2=" << qError << " k_relative_l2=" << kError
              << " v_bitwise=1 attention_relative_l2=" << error << '\n';
}

static void checkModulation(int rows) {
    Runtime rt(0);
    std::mt19937 random(49);
    std::normal_distribution<float> normal(0, .5f);
    auto tensor = [&](int r, int c) {
        std::vector<float> data(size_t(r) * c);
        for (auto &v : data)
            v = normal(random);
        return rt.upload(data.data(), r, c);
    };
    auto x = tensor(rows, 5376), y = tensor(rows, 5376), w = tensor(1, 5376), m = tensor(21, 6 * 5376);
    std::vector<int> indices(rows);
    for (int i = 0; i < rows; ++i)
        indices[i] = (i * 7 + i / 3) % 21;
    auto idx = rt.indices(indices);
    auto compare = [&](const Tensor &expected, const Tensor &actual) {
        auto a = rt.download(expected), b = rt.download(actual);
        double norm = 0, error = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(b[i]))
                throw std::runtime_error("nonfinite image modulation result");
            norm += double(a[i]) * a[i];
            error += std::pow(double(a[i]) - b[i], 2);
        }
        double l2 = std::sqrt(error / std::max(norm, 1e-30));
        if (l2 > 1e-2)
            throw std::runtime_error("image modulation exceeds relative L2 tolerance");
        return l2;
    };
    auto norm = rt.rmsModulateRows(x, w, 1e-5f, m, idx, 0, 1);
    double first = compare(norm, rmsModulate(x, w, m, idx, 0, 1));
    auto residual = rt.residualRows(x, y, m, idx, 2);
    auto fused = residualRmsModulate(x, y, w, m, idx, 2, 3, 4);
    if (rt.download(residual) != rt.download(fused.first))
        throw std::runtime_error("fused image modulation changed residual rounding");
    double second = compare(rt.rmsModulateRows(residual, w, 1e-5f, m, idx, 3, 4), fused.second);
    std::cout << "modulate rows=" << rows << " norm_relative_l2=" << first << " fused_relative_l2=" << second
              << " residual_bitwise=1\n";
}

static void checkVaeQkv() {
    Runtime rt(0), full(0, true);
    std::mt19937 rng(127);
    std::normal_distribution<float> random(0, 2);
    for (int batch : {1, 3, 61}) {
        const int rows = batch * 261;
        std::vector<float> values(size_t(rows) * 3072), cs(261 * 32), sn(cs.size());
        for (auto &v : values)
            v = random(rng);
        for (size_t i = 0; i < cs.size(); ++i) {
            float angle = random(rng);
            cs[i] = std::cos(angle);
            sn[i] = std::sin(angle);
        }
        auto a = rt.upload(values.data(), rows, 3072), b = rt.upload(values.data(), rows, 3072);
        auto cosine = full.upload(cs.data(), 261, 32), sine = full.upload(sn.data(), 261, 32);
        rt.videoQkvNormRotaryInPlace(a, cosine, sine);
        vaeQkvNormRotaryInPlace(b, cosine, sine);
        auto expected = rt.download(a), actual = rt.download(b);
        double error = 0, norm = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!std::isfinite(actual[i]) || (i % 3072 >= 2048 && actual[i] != expected[i]))
                throw std::runtime_error("invalid image VAE QKV output");
            error += std::pow(double(actual[i]) - expected[i], 2);
            norm += std::pow(double(expected[i]), 2);
        }
        double l2 = std::sqrt(error / std::max(norm, 1e-30));
        if (l2 > 1e-2)
            throw std::runtime_error("VAE QKV exceeds relative L2 tolerance");
        std::cout << "vae_qkv batch=" << batch << " relative_l2=" << l2 << std::endl;
    }
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    try {
        checkVaeQkv();
        for (int rows : {1, 17, 513, 4106})
            checkModulation(rows);
        for (auto shape :
             {std::pair<int, int>{1, 1}, {13, 7}, {257, 28}, {1025, 56}, {4106, 28}, {4106, 56}, {9244, 56}})
            for (bool rotary : {false, true})
                checkPackedQkv(shape.first, shape.second, rotary);
        for (auto type : {DType::BF16, DType::FP32}) {
            checkStitch(type, 16, 16, 0, 1, 1);
            checkStitch(type, 32, 48, 16, 2, 3);
            checkStitch(type, 256, 256, 64, 3, 2);
        }
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
                auto expected = rt.download(rt.siluMultiply(
                    rt.columns(input, 0, input.cols / 2), rt.columns(input, input.cols / 2, input.cols / 2)));
                auto actual = rt.download(gatedSilu(input));
                if (actual.size() != expected.size())
                    throw std::runtime_error("gated SiLU output shape mismatch");
                double error = 0, norm = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(actual[i]) || (type != DType::BF16 && actual[i] != expected[i]))
                        throw std::runtime_error("invalid gated SiLU output");
                    error += std::pow(double(actual[i]) - expected[i], 2);
                    norm += std::pow(double(expected[i]), 2);
                }
                double l2 = std::sqrt(error / std::max(norm, 1e-30));
                if (l2 > 1e-2)
                    throw std::runtime_error("gated SiLU exceeds relative L2 tolerance");
                std::cout << "dtype=" << int(type) << " rows=" << shape.first << " cols=" << shape.second
                          << " relative_l2=" << l2 << std::endl;
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
