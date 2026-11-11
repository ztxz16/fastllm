#include "models/minimax_h3_image/model.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3_image;
using fastllm::minimax_h3::VideoTensor;
using fastllm::minimax_h3::VideoVAE;

static void compare(const std::string &name, const std::vector<float> &expected,
                    const std::vector<float> &actual, bool exact = false) {
    if (expected.size() != actual.size())
        throw std::runtime_error("shape mismatch: " + name);
    double delta = 0, norm = 0, maxAbsolute = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            throw std::runtime_error("nonfinite output: " + name);
        double difference = double(actual[i]) - expected[i];
        delta += difference * difference;
        norm += double(expected[i]) * expected[i];
        maxAbsolute = std::max(maxAbsolute, std::abs(difference));
    }
    double error = std::sqrt(delta / std::max(norm, 1e-30));
    std::cout << name << " relative_l2=" << error << " max_abs=" << maxAbsolute << std::endl;
    if (error > 1e-2)
        throw std::runtime_error("encoder exceeds relative L2 tolerance: " + name);
    if (exact && expected != actual)
        throw std::runtime_error("encoder changed FP32 fallback or cached output: " + name);
}

int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: minimax_h3_image_encode MODEL REFERENCE_IMAGE");
        Runtime full(0, true);
        auto image = readImage(argv[2]);
        VideoVAE reference(full, full, std::string(argv[1]) + "/vae");
        VideoVAE batched(full, full, std::string(argv[1]) + "/vae");
        std::mt19937 random(71);
        std::normal_distribution<float> noise(0, .2f);
        for (auto shape : {std::array<int, 3>{32, 48, 2}, {256, 256, 1}, {256, 256, 3},
                           {256, 256, 8}, {48, 80, 3}}) {
            int h = shape[0], w = shape[1], batch = shape[2], rows = h * w;
            std::vector<float> pixels(size_t(rows) * 3 * batch), expected;
            const float mean[] = {.485f, .456f, .406f}, scale[] = {.229f, .224f, .225f};
            for (int b = 0; b < batch; ++b)
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        for (int c = 0; c < 3; ++c) {
                            auto p = ((size_t(y + b * 143) % image.height) * image.width +
                                      (x + b * 197) % image.width) * 4 + c;
                            pixels[((size_t(b) * h + y) * w + x) * 3 + c] =
                                (image.rgba[p] / 255.f - mean[c]) / scale[c] + noise(random);
                        }
            auto input = full.upload(pixels.data(), batch * rows, 3);
            std::map<std::string, std::vector<float>> stages;
            for (int i = 0; i < batch; ++i) {
                auto output = reference.encodeTile(input.rowSlice(i * rows, rows), 1, h, w,
                    [&](const std::string &name, const Tensor &x) {
                        auto values = full.download(x);
                        stages[name].insert(stages[name].end(), values.begin(), values.end());
                    });
                auto values = full.download(output);
                expected.insert(expected.end(), values.begin(), values.end());
            }
            auto prefix = std::to_string(h) + "x" + std::to_string(w) + "/batch" + std::to_string(batch);
            auto actual = batched.encodeImageTiles(input, h, w, batch,
                [&](const std::string &name, const Tensor &x) {
                    compare(prefix + "/" + name, stages.at(name), full.download(x));
                });
            compare(prefix + "/output", expected, full.download(actual));
        }
        VideoTensor pixels(1, 320, 464, 3);
        for (auto &value : pixels.values)
            value = noise(random);
        auto expected = reference.encodeMoments(pixels);
        auto actual = batched.encodeMoments(pixels, {}, true);
        compare("overlapping_tiles", expected.values, actual.values);
        // The image-only compact weights and private TF32 handle must not leak
        // into the causal video path, even when reusing the same VAE instance.
        for (int frames : {1, 17}) {
            VideoTensor video(frames, 32, 48, 3);
            for (auto &value : video.values)
                value = noise(random);
            auto input = full.upload(video.values.data(), frames * 32 * 48, 3);
            auto prefix = "video_after_image/" + std::to_string(frames);
            compare(prefix + "/tile", full.download(reference.encodeTile(input, frames, 32, 48)),
                    full.download(batched.encodeTile(input, frames, 32, 48)), true);
            compare(prefix + "/moments", reference.encodeMoments(video).values,
                    batched.encodeMoments(video).values, true);
        }
        batched.unload();
        compare("reuse_after_unload", actual.values, batched.encodeMoments(pixels, {}, true).values, true);
        batched.setLowMemory(true);
        compare("low_memory_fallback", expected.values, batched.encodeMoments(pixels, {}, true).values, true);
        batched.setLowMemory(false);
        compare("reuse_after_low_memory", actual.values, batched.encodeMoments(pixels, {}, true).values, true);
        try {
            VideoTensor invalid(2, 32, 32, 3);
            batched.encodeMoments(invalid, {}, true);
            throw std::runtime_error("single-frame encoder accepted video");
        } catch (const std::invalid_argument &) {
        }
        std::cout << "encoder comparison passed" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
