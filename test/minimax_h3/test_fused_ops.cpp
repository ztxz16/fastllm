// Fused-operator and shared linear-output regression checks. No model weights needed.
#include "models/minimax_h3/runtime.h"
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cfenv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3;

static std::mt19937 rng(1701);
static Tensor randomTensor(Runtime &rt, int rows, int cols) {
    std::normal_distribution<float> normal(0.f, 0.4f);
    std::vector<float> data(size_t(rows) * cols);
    for (auto &v : data)
        v = normal(rng);
    return rt.upload(data.data(), rows, cols);
}
static void equal(Runtime &rt, const Tensor &a, const Tensor &b) {
    if (a.rows != b.rows || a.cols != b.cols || a.dtype != b.dtype)
        throw std::runtime_error("shape/dtype mismatch");
    auto x = rt.download(a), y = rt.download(b);
    size_t different = 0;
    double squared = 0, reference = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
            throw std::runtime_error("non-finite output");
        different += x[i] != y[i];
        squared += double(x[i] - y[i]) * (x[i] - y[i]);
        reference += double(x[i]) * x[i];
    }
    std::cout << " different=" << different << " relative_l2=" << std::sqrt(squared / std::max(reference, 1e-30));
    if (different)
        throw std::runtime_error("fused operator changed rounding");
}
template <class F> static void rejects(F fn) {
    try {
        fn();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid linear output was accepted");
}
static void linearOutputs() {
    for (auto dtype : {DType::FP32, DType::FP16, DType::BF16}) {
        Runtime rt(0, dtype);
        for (int rows : {1, 47, 512, 513}) {
            const int width = 128;
            // Dyadic values make the FP32 dot products exact, so this CPU oracle
            // checks every dtype without depending on the GEMM under test.
            std::vector<float> x(size_t(rows) * width), w(width * width), b(width);
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = (int(i % 31) - 15) / 16.f;
            for (size_t i = 0; i < w.size(); ++i)
                w[i] = (int((i * 7) % 31) - 15) / 16.f;
            for (int i = 0; i < width; ++i)
                b[i] = (i % 7 - 3) / 16.f;
            auto input = rt.upload(x.data(), rows, width), weight = rt.upload(w.data(), width, width);
            auto bias = rt.upload(b.data(), 1, width);
            for (bool biased : {false, true}) {
                std::vector<float> expected(size_t(rows) * width);
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < width; ++c) {
                        float sum = 0;
                        for (int k = 0; k < width; ++k)
                            sum += x[size_t(r) * width + k] * w[c * width + k];
                        expected[size_t(r) * width + c] = sum + (biased ? b[c] : 0.f);
                    }
                std::vector<float> initial(size_t(rows + 2) * width, -17.f);
                auto storage = rt.upload(initial.data(), rows + 2, width);
                auto target = storage.rowSlice(1, rows);
                rt.linearInto(input, weight, target, biased ? &bias : nullptr);
                std::cout << "linear_into dtype=" << int(dtype) << " rows=" << rows << " bias=" << biased;
                equal(rt, rt.upload(expected.data(), rows, width), target);
                equal(rt, rt.linear(input, weight, biased ? &bias : nullptr), target);
                auto result = rt.download(storage);
                for (int i = 0; i < width; ++i)
                    if (result[i] != -17.f || result[size_t(rows + 1) * width + i] != -17.f)
                        throw std::runtime_error("linear wrote outside the output slice");
                std::cout << std::endl;
            }
            auto wrongShape = randomTensor(rt, rows, width + 1);
            rejects([&] { rt.linearInto(input, weight, wrongShape); });
            rejects([&] { rt.linearInto(input, weight, input); });
            Tensor wrongType(rows, width, dtype == DType::FP32 ? DType::BF16 : DType::FP32);
            rejects([&] { rt.linearInto(input, weight, wrongType); });
        }
        if (dtype != DType::FP32) {
            auto input = randomTensor(rt, 1, 128), weight = randomTensor(rt, 127, 128);
            auto bias = randomTensor(rt, 1, 127), storage = randomTensor(rt, 3, 127);
            auto unaligned = storage.rowSlice(1, 1);
            rejects([&] { rt.linearInto(input, weight, unaligned, &bias); });
        }
    }
}
static void residuals(Runtime &rt) {
    for (int rows : {1, 17, 513, 12847}) {
        std::vector<int> indices(rows);
        for (int i = 0; i < rows; ++i)
            indices[i] = (i * 7) % 9;
        auto idx = rt.indices(indices);
        for (int width : {127, 512, 5376}) {
            auto x = randomTensor(rt, rows, width), y = randomTensor(rt, rows, width);
            // Exercise an odd table stride as well as the packed BF16 path.
            for (int extra : {0, 1}) {
                if (extra && rows != 17)
                    continue;
                auto mod = randomTensor(rt, 9, 6 * width + extra);
                for (int gate : {2, 5}) {
                    auto gates = rt.gather(rt.columns(mod, gate * width, width), idx);
                    auto reference = rt.add(x, rt.multiply(y, gates));
                    std::cout << "residual fp32=" << rt.isFloat32() << " rows=" << rows
                              << " width=" << width << " gate=" << gate << " extra=" << extra;
                    equal(rt, reference, rt.residualRows(x, y, mod, idx, gate));
                    std::cout << std::endl;
                }
            }
        }
    }
    const int width = 512;
    std::vector<float> edges = {0.f, -0.f, 1.f, -1.f, std::ldexp(1.f, -133),
                               -std::ldexp(1.f, -133), std::ldexp(1.f, -126),
                               -std::ldexp(1.f, -126), std::ldexp(1.f, -60), std::ldexp(1.f, 60)};
    std::vector<float> a(width), b(width), m(6 * width);
    for (int i = 0; i < width; ++i) {
        a[i] = edges[i % edges.size()];
        b[i] = edges[(i / edges.size()) % edges.size()];
        for (int g = 0; g < 6; ++g)
            m[g * width + i] = edges[(i / (edges.size() * edges.size()) + g) % edges.size()];
    }
    auto x = rt.upload(a.data(), 1, width), y = rt.upload(b.data(), 1, width);
    auto mod = rt.upload(m.data(), 1, 6 * width), idx = rt.indices({0});
    for (int gate : {2, 5}) {
        auto reference = rt.add(x, rt.multiply(y, rt.columns(mod, gate * width, width)));
        std::cout << "residual edges fp32=" << rt.isFloat32() << " gate=" << gate;
        equal(rt, reference, rt.residualRows(x, y, mod, idx, gate));
        std::cout << std::endl;
    }
    if (!rt.isFloat32()) {
        auto storage = randomTensor(rt, 1, width + 1);
        auto unaligned = storage;
        ++unaligned.data;
        unaligned.cols = width;
        auto reference = rt.add(unaligned, rt.multiply(y, rt.columns(mod, 2 * width, width)));
        std::cout << "residual unaligned BF16";
        equal(rt, reference, rt.residualRows(unaligned, y, mod, idx, 2));
        std::cout << std::endl;
    }
}
static void hostWeightCache() {
    auto path = std::filesystem::temp_directory_path() /
                ("fastllm-h3-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
    } cleanup{path};
    std::filesystem::create_directory(path);
    for (bool bf16 : {false, true}) {
        std::string name = bf16 ? "a" : "b";
        std::string header = "{\"" + name + "\":{\"dtype\":\"" + (bf16 ? "BF16" : "F32") +
                             "\",\"shape\":[2,4],\"data_offsets\":[0," + (bf16 ? "16" : "32") + "]}}";
        uint64_t length = header.size();
        float floats[] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint16_t halves[] = {0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100};
        std::ofstream file(path / (name + ".safetensors"), std::ios::binary);
        file.write(reinterpret_cast<const char *>(&length), 8);
        file.write(header.data(), header.size());
        file.write(bf16 ? reinterpret_cast<const char *>(halves) : reinterpret_cast<const char *>(floats), bf16 ? 16 : 32);
    }
    std::ofstream(path / "model.safetensors.index.json") <<
        "{\"weight_map\":{\"a\":\"a.safetensors\",\"b\":\"b.safetensors\"}}";
    for (auto dtype : {DType::FP32, DType::BF16, DType::FP16}) {
        Runtime rt(0, dtype);
        for (size_t budget : {size_t(16), size_t(1024)}) {
            Weights weights(rt, path.string());
            weights.enableHostCache(budget);
            for (int round = 0; round < 3; ++round) {
                for (auto name : {"a", "b"}) {
                    auto result = rt.download(weights.get(name));
                    for (int i = 0; i < 8; ++i)
                        if (result[i] != float(i + 1))
                            throw std::runtime_error("host weight cache changed values");
                }
                weights.erasePrefix("a");
                weights.clear();
                if (round == 1)
                    weights.enableHostCache(0);
            }
        }
    }
    Runtime rt(0, false);
    Tensor first, second;
    {
        Weights weights(rt, path.string());
        weights.enableHostCache(1024);
        std::exception_ptr error;
        std::thread worker([&] {
            try { first = weights.get("a"); } catch (...) { error = std::current_exception(); }
        });
        worker.join();
        if (error)
            std::rethrow_exception(error);
        weights.erasePrefix("a");
        second = weights.get("a");
        // Destroy the host cache immediately after enqueueing its last upload.
        weights.clear();
    }
    equal(rt, first, second);
    std::cout << "host cache: dtype/shard/budget/clear/disable/worker/lifetime PASS" << std::endl;
}
template <class T> static float videoRound(float v) { return float(T(v)); }
static std::vector<float> videoUnpatch(const std::vector<float> &patches, int frames, int height, int width) {
    std::vector<float> pixels(patches.size());
    for (int t = 0; t < frames * 4; ++t)
        for (int y = 0; y < height * 16; ++y)
            for (int x = 0; x < width * 16; ++x)
                for (int c = 0; c < 3; ++c) {
                    size_t row = (size_t(t / 4) * height + y / 16) * width + x / 16;
                    int col = ((c * 4 + t % 4) * 16 + y % 16) * 16 + x % 16;
                    pixels[((size_t(t) * height * 16 + y) * width * 16 + x) * 3 + c] = patches[row * 3072 + col];
                }
    return pixels;
}
template <class T>
static void videoBlend(const std::vector<float> &a, std::vector<float> &b, int frames, int height,
                       int width, int extent, int axis) {
    auto index = [=](int t, int y, int x, int c) { return ((size_t(t) * height + y) * width + x) * 3 + c; };
    for (int t = 0; t < frames; ++t)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                int pos = axis == 0 ? t : axis == 1 ? y : x;
                if (pos >= extent)
                    continue;
                float wb = videoRound<T>(float(pos) / extent), wa = videoRound<T>(1.f - wb);
                for (int c = 0; c < 3; ++c) {
                    size_t i = index(t, y, x, c);
                    size_t j = index(t, axis == 1 ? height - extent + y : y,
                                     axis == 2 ? width - extent + x : x, c);
                    // Keep this CPU oracle unfused, like the released eager operations.
                    volatile float left = videoRound<T>(a[j] * wa), right = videoRound<T>(b[i] * wb);
                    b[i] = videoRound<T>(left + right);
                }
            }
}
template <class T> static void videoStitch(DType dtype) {
    Runtime rt(0, dtype), full(0, true);
    for (const auto &ys : std::vector<std::vector<int>>{{0}, {0, 2, 5}})
        for (const auto &xs : std::vector<std::vector<int>>{{0}, {0, 3, 5}, {0, 5}}) {
            int f = 2, th = 4, tw = 5, height = ys.back() + th, width = xs.back() + tw;
            std::vector<Tensor> patches;
            std::vector<std::vector<float>> raw;
            for (size_t i = 0; i < ys.size() * xs.size(); ++i) {
                patches.push_back(randomTensor(rt, f * th * tw, 3072));
                raw.push_back(videoUnpatch(rt.download(patches.back()), f, th, tw));
            }
            std::vector<float> expected(size_t(f) * height * width * 3072);
            for (size_t y = 0; y < ys.size(); ++y)
                for (size_t x = 0; x < xs.size(); ++x) {
                    auto tile = raw[y * xs.size() + x];
                    if (y)
                        videoBlend<T>(raw[(y - 1) * xs.size() + x], tile, f * 4, th * 16, tw * 16,
                                      (ys[y - 1] + th - ys[y]) * 16, 1);
                    if (x)
                        videoBlend<T>(raw[y * xs.size() + x - 1], tile, f * 4, th * 16, tw * 16,
                                      (xs[x - 1] + tw - xs[x]) * 16, 2);
                    int h = (y + 1 < ys.size() ? ys[y + 1] - ys[y] : th) * 16;
                    int w = (x + 1 < xs.size() ? xs[x + 1] - xs[x] : tw) * 16;
                    for (int t = 0; t < f * 4; ++t)
                        for (int r = 0; r < h; ++r)
                            std::copy_n(tile.data() + (size_t(t) * th * 16 + r) * tw * 16 * 3, w * 3,
                                        expected.data() + ((size_t(t) * height * 16 + ys[y] * 16 + r) * width * 16 + xs[x] * 16) * 3);
                }
            std::cout << "video_stitch dtype=" << int(dtype) << " tiles=" << patches.size();
            equal(full, full.upload(expected.data(), f * height * width * 1024, 3),
                  rt.stitchVideo(patches, f, th, tw, height, width, ys, xs));
            std::cout << std::endl;
        }
    const int pixels = 37;
    auto clip = randomTensor(rt, 28 * pixels, 3), tail = randomTensor(rt, 5 * pixels, 3);
    auto current = full.cast(clip, DType::FP32), previous = full.cast(tail, DType::FP32);
    auto a = full.download(previous), b = full.download(current);
    for (int count : {1, 4, 5, 17})
        for (bool overlap : {false, true}) {
            std::vector<float> expected(b.begin() + 3 * pixels * 3, b.begin() + (3 + count) * pixels * 3);
            if (overlap) {
                // Blend before cropping, including truncated output shorter than the five-frame overlap.
                std::vector<float> temp(b.begin() + 3 * pixels * 3, b.begin() + 20 * pixels * 3);
                videoBlend<T>(a, temp, 17, 1, pixels, 5, 0);
                std::copy_n(temp.begin(), expected.size(), expected.begin());
            }
            std::vector<float> sentinels(size_t(count + 2) * pixels * 3, -17.f);
            auto storage = full.upload(sentinels.data(), (count + 2) * pixels, 3);
            auto target = storage.rowSlice(pixels, count * pixels);
            rt.copyVideoFrames(target, current, overlap ? &previous : nullptr, pixels, 3);
            std::cout << "video_frames dtype=" << int(dtype) << " count=" << count << " overlap=" << overlap;
            equal(full, full.upload(expected.data(), count * pixels, 3), target);
            auto got = full.download(storage);
            for (int i = 0; i < pixels * 3; ++i)
                if (got[i] != -17.f || got[got.size() - 1 - i] != -17.f)
                    throw std::runtime_error("video frame copy wrote outside the output slice");
            std::cout << std::endl;
        }
    rejects([&] { rt.stitchVideo({clip}, 2, 4, 5, 4, 5, {0}, {0}); });
    rejects([&] { rt.stitchVideo({}, 2, 4, 5, 4, 5, {}, {0}); });
    rejects([&] { rt.copyVideoFrames(current, current, nullptr, pixels, 1); });
}

static void packedVideoQkv() {
    for (auto dtype : {DType::FP16, DType::BF16, DType::FP32}) {
        Runtime full(0, true), rt(0, dtype);
        for (int rows : {1, 7, 513, 1797}) {
            auto packed = randomTensor(rt, rows, 3 * 128);
            auto cs = randomTensor(full, rows, 48), sn = randomTensor(full, rows, 48);
            auto q = rt.columns(packed, 0, 128), k = rt.columns(packed, 128, 128);
            auto expected = rt.videoQkNormRotary(q, k, cs, sn);
            auto actual = rt.videoQkvNormRotary(packed, cs, sn);
            std::cout << "packed_video_qkv dtype=" << int(dtype) << " rows=" << rows;
            equal(rt, expected.first, actual[0]);
            equal(rt, expected.second, actual[1]);
            equal(rt, rt.columns(packed, 256, 128), actual[2]);
            auto batch = rt.videoQkvNormRotary(rt.concat({packed, packed}), cs, sn);
            for (int i = 0; i < 3; ++i)
                equal(rt, rt.concat({actual[i], actual[i]}), batch[i]);
            std::cout << std::endl;
        }
    }
}
static void videoRGB() {
    Runtime rt(0, true);
    struct RestoreRounding {
        int mode = std::fegetround();
        ~RestoreRounding() { std::fesetround(mode); }
    } restore;
    const float mean[] = {.485f, .456f, .406f}, scale[] = {.229f, .224f, .225f};
    std::vector<float> boundaries;
    for (int v = -1; v < 257; ++v)
        for (int side = -1; side <= 1; ++side)
            for (int c = 0; c < 3; ++c) {
                float x = ((v + .5f) / 255.f - mean[c]) / scale[c];
                boundaries.push_back(side ? std::nextafter(x, side < 0 ? -INFINITY : INFINITY) : x);
            }
    for (int rows : {1, 2, 3, 7, 513, int(boundaries.size() / 3)}) {
        std::vector<float> input(boundaries.begin(), boundaries.begin() + rows * 3);
        auto pixels = rt.upload(input.data(), rows, 3);
        for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            std::fesetround(mode);
            std::vector<uint8_t> expected(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                volatile float product = input[i] * scale[i % 3];
                volatile float shifted = product + mean[i % 3];
                float clamped = std::clamp(float(shifted), 0.f, 1.f);
                volatile float scaled = clamped * 255.f;
                expected[i] = uint8_t(std::nearbyint(scaled));
            }
            if (rt.downloadVideoRGB(pixels) != expected)
                throw std::runtime_error("video RGB changed rounding: " + std::to_string(mode));
        }
        std::cout << "video_rgb rows=" << rows << " all four rounding modes exact" << std::endl;
    }
    rejects([&] { rt.downloadVideoRGB(Tensor(1, 2, DType::FP32)); });
    rejects([&] { rt.downloadVideoRGB(Tensor(1, 3, DType::FP16)); });
}

static void audioConvolution() {
    Runtime rt(0, true);
    for (bool transposed : {false, true})
        for (int outputs : {8, 64, 128}) {
            const int rows = 17003, channels = 3, kernel = 7, pad = 3;
            int stride = transposed ? 2 : 1, dilation = transposed ? 1 : 3;
            int length = transposed ? (rows - 1) * stride - 2 * pad + kernel
                                    : (rows + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
            std::vector<float> input(rows * channels), weights(outputs * channels * kernel), bias(outputs);
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = (int(i % 31) - 15) / 32.f;
            for (size_t i = 0; i < weights.size(); ++i)
                weights[i] = (int(i % 17) - 8) / 16.f;
            for (int i = 0; i < outputs; ++i)
                bias[i] = (i % 7 - 3) / 64.f;
            std::vector<float> expected(size_t(length) * outputs);
            for (int t = 0; t < length; ++t)
                for (int o = 0; o < outputs; ++o) {
                    float sum = 0;
                    for (int c = 0; c < channels; ++c)
                        for (int k = 0; k < kernel; ++k) {
                            int source = transposed ? t + pad - k : t * stride - pad + k * dilation;
                            if (transposed && source % stride)
                                continue;
                            if (transposed)
                                source /= stride;
                            if (source >= 0 && source < rows)
                                sum += input[source * channels + c] * weights[(o * channels + c) * kernel + k];
                        }
                    expected[size_t(t) * outputs + o] = sum + bias[o];
                }
            auto b = rt.upload(bias.data(), 1, outputs);
            auto actual = rt.conv1d(rt.upload(input.data(), rows, channels),
                                    rt.upload(weights.data(), outputs, channels * kernel), &b,
                                    kernel, stride, pad, dilation, transposed);
            std::cout << "audio_conv outputs=" << outputs << " transposed=" << transposed;
            equal(rt, rt.upload(expected.data(), length, outputs), actual);
            std::cout << std::endl;
        }
}

int main() {
    try {
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess && status != cudaErrorNoDevice)
            throw std::runtime_error(cudaGetErrorString(status));
        if (count != 1) {
            std::cerr << "Expose only the permitted GPU" << std::endl;
            return 77;
        }
        audioConvolution();
        videoRGB();
        packedVideoQkv();
        videoStitch<float>(DType::FP32);
        videoStitch<__half>(DType::FP16);
        videoStitch<__nv_bfloat16>(DType::BF16);
        hostWeightCache();
        linearOutputs();
        Runtime rt(0, false), full(0, true);
        residuals(rt);
        residuals(full);
        for (bool fp32 : {false, true}) {
            Runtime &r = fp32 ? full : rt;
            for (int rows : {1, 17, 513, 12847}) {
                if (fp32 && rows == 12847)
                    continue;
                int heads = rows == 12847 ? 56 : 3;
                for (int dim : {64, 128}) {
                    if (rows == 12847 && dim != 128)
                        continue;
                    auto q = randomTensor(r, rows, heads * dim), k = randomTensor(r, rows, heads * dim);
                    auto qw = randomTensor(r, 1, dim), kw = randomTensor(r, 1, dim);
                    for (int rd : {dim / 2, dim == 128 ? 96 : 48, dim}) {
                        auto cs = randomTensor(full, rows, rd), sn = randomTensor(full, rows, rd);
                        auto qRef = r.rotary(r.rms(q, qw, 1e-5f, dim), cs, sn, dim);
                        auto kRef = r.rotary(r.rms(k, kw, 1e-5f, dim), cs, sn, dim);
                        auto fused = r.qkNormRotary(q, k, qw, kw, 1e-5f, cs, sn, dim);
                        std::cout << "qk fp32=" << fp32 << " rows=" << rows << " dim=" << dim << " rope=" << rd;
                        equal(r, qRef, fused.first);
                        equal(r, kRef, fused.second);
                        std::cout << std::endl;
                    }
                }
            }
        }
        for (auto dtype : {DType::FP16, DType::BF16, DType::FP32}) {
            Runtime r(0, dtype);
            for (int h : {1, 8, 16}) {
                const int frames = 7, width = h == 8 ? 4 : h;
                std::vector<float> input(size_t(frames) * h * width * 3072);
                for (size_t i = 0; i < input.size(); ++i)
                    input[i] = (int(i % 8191) - 4095) / 64.f;
                auto patches = r.upload(input.data(), frames * h * width, 3072);
                auto expected = videoUnpatch(r.download(patches), frames, h, width);
                std::cout << "single_tile_stitch dtype=" << int(dtype) << " height=" << h;
                equal(full, full.upload(expected.data(), frames * h * width * 1024, 3),
                      r.stitchVideo({patches}, frames, h, width, h, width, {0}, {0}));
                std::cout << std::endl;
            }
            for (int rows : {1, 47, 513, 1797}) {
                for (int width : {67, 2048}) {
                    auto x = randomTensor(full, rows, width), w = randomTensor(full, 1, width);
                    std::cout << "video_rms dtype=" << int(dtype) << " rows=" << rows << " width=" << width;
                    equal(r, r.cast(full.norm(x, true, &w, nullptr, 1e-5f), dtype), r.videoRms(x, w));
                    std::cout << std::endl;
                }
                auto q = randomTensor(r, rows, 2048), k = randomTensor(r, rows, 2048);
                for (int dim : {48, 64}) {
                    auto cs = randomTensor(full, rows, dim), sn = randomTensor(full, rows, dim);
                    auto qRef = r.rotary(r.cast(full.norm(r.cast(q, DType::FP32), true, nullptr, nullptr, 1e-5f, 64), dtype), cs, sn, 64);
                    auto kRef = r.rotary(r.cast(full.norm(r.cast(k, DType::FP32), true, nullptr, nullptr, 1e-5f, 64), dtype), cs, sn, 64);
                    auto fused = r.videoQkNormRotary(q, k, cs, sn);
                    std::cout << "video_qk dtype=" << int(dtype) << " rows=" << rows << " rope=" << dim;
                    equal(r, qRef, fused.first); equal(r, kRef, fused.second);
                    std::cout << std::endl;
                }
                for (int width : {67, 2048}) {
                    auto x = randomTensor(full, rows, width), y = randomTensor(r, rows, width);
                    auto scale = randomTensor(full, 1, width);
                    auto reference = full.add(x, full.multiply(full.cast(y, DType::FP32), full.repeatRows(scale, rows)));
                    std::cout << "scaled_residual dtype=" << int(dtype) << " rows=" << rows << " width=" << width;
                    equal(full, reference, r.scaledResidual(x, y, scale));
                    std::cout << std::endl;
                }
            }
            for (int rows : {1, 47, 513, 12847}) {
                int width = rows == 12847 ? 14336 : 67;
                auto x = randomTensor(r, rows, width * 2);
                auto reference = r.multiply(r.columns(x, 0, width), r.activation(r.columns(x, width, width)));
                std::cout << "silu dtype=" << int(dtype) << " rows=" << rows;
                equal(r, reference, r.siluMultiply(x));
                std::cout << std::endl;
            }
        }
        for (bool fp32 : {false, true}) {
            Runtime &r = fp32 ? full : rt;
            for (int rows : {1, 17, 513, 12847}) {
                std::vector<int> indices(rows);
                for (int i = 0; i < rows; ++i)
                    indices[i] = (i * 7) % 9;
                auto idx = r.indices(indices);
                for (int width : {127, 5376}) {
                    auto x = randomTensor(r, rows, width), w = randomTensor(r, 1, width);
                    auto mod = randomTensor(r, 9, 6 * width);
                    for (int shift : {0, 3}) {
                        auto reference = r.modulateRows(r.rms(x, w, 1e-5f), mod, idx, shift, shift + 1);
                        std::cout << "rms_mod fp32=" << fp32 << " rows=" << rows << " width=" << width
                                  << " shift=" << shift;
                        equal(r, reference, r.rmsModulateRows(x, w, 1e-5f, mod, idx, shift, shift + 1));
                        std::cout << std::endl;
                    }
                }
            }
        }
        std::cout << "PASS" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
