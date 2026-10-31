#include "models/minimax_h3/runtime.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
DType parseVideoPrecision(const std::string &name, bool fp32) {
    if (name.empty())
        return fp32 ? DType::FP32 : DType::FP16;
    if (name == "fp16")
        return DType::FP16;
    if (name == "bf16")
        return DType::BF16;
    if (name == "fp32")
        return DType::FP32;
    throw std::invalid_argument("VAE precision must be fp16, bf16 or fp32");
}
const char *precisionName(DType type) {
    switch (type) {
    case DType::FP16:
        return "fp16";
    case DType::BF16:
        return "bf16";
    case DType::FP32:
        return "fp32";
    }
    throw std::invalid_argument("invalid precision");
}
VideoVAE::VideoVAE(Runtime &rt, Runtime &full, const std::string &dir)
    : rt_(rt), full_(full), weights_(rt, componentWeights(dir)), fullWeights_(full, componentWeights(dir)) {
    if (!full.isFloat32())
        throw std::invalid_argument("video VAE residual runtime must be FP32");
}
Tensor VideoVAE::decodeTile(const Tensor &latent, int frames, int height, int width, Trace trace) {
    return decodeTiles(latent, frames, height, width, 1, std::move(trace));
}
Tensor VideoVAE::decodeTiles(const Tensor &latent, int frames, int height, int width, int batch, Trace trace) {
    int tileRows = frames * height * width;
    if (batch < 1 || tileRows < 1 || latent.rows != tileRows * batch || latent.cols != 24)
        throw std::invalid_argument("invalid H3 VAE tile batch");
    int sequence = tileRows + 5;
    // Keep input/output projection shapes unchanged: batching them changes GEMM rounding.
    std::vector<Tensor> embedded;
    for (int i = 0; i < batch; ++i) {
        auto input = rt_.cast(latent.rowSlice(i * tileRows, tileRows), rt_.dtype());
        embedded.push_back(weights_.linear(weights_.linear(input, "post_quant_conv"), "decoder.proj_in"));
    }
    auto registers = weights_.get("decoder.register_tokens");
    registers.rows = 4;
    registers.cols = 2048;
    const std::array<int, 3> shape{frames, height, width};
    if (!rope_.first.data || ropeShape_ != shape) {
        std::vector<float> positions(size_t(sequence) * 3);
        for (int t = 0; t < frames; ++t)
            for (int h = 0; h < height; ++h)
                for (int w = 0; w < width; ++w) {
                    size_t i = (size_t(t) * height + h) * width + w;
                    positions[i * 3] = (2.f * ((t + .5f) / frames) - 1.f) * (2.f * float(M_PI));
                    positions[i * 3 + 1] = (2.f * ((h + .5f) / height) - 1.f) * (2.f * float(M_PI));
                    positions[i * 3 + 2] = (2.f * ((w + .5f) / width) - 1.f) * (2.f * float(M_PI));
                }
        rope_ = rt_.ropeTable(positions, 8, 100.f);
        ropeShape_ = shape;
    }
    const auto &rope = rope_;
    if (!zero_.data) {
        std::vector<float> zeros(2048);
        zero_ = rt_.upload(zeros.data(), 1, 2048);
    }
    std::vector<Tensor> tokens;
    for (int i = 0; i < batch; ++i) {
        tokens.push_back(embedded[i]);
        tokens.push_back(registers);
        tokens.push_back(zero_);
    }
    auto x = rt_.concat(tokens);
    // Released decode uses low-precision projections, FP32 norms/scales and residuals.
    x = rt_.cast(x, DType::FP32);
    for (int layer = 0; layer < 36; ++layer) {
        auto p = "decoder.transformer_blocks." + std::to_string(layer);
        auto y = rt_.videoRms(x, fullWeights_.get(p + ".norm1.weight"));
        Tensor q, k, v;
        if (rt_.isFloat32()) {
            q = weights_.linear(y, p + ".attn.to_q");
            k = weights_.linear(y, p + ".attn.to_k");
            v = weights_.linear(y, p + ".attn.to_v");
            auto qk = rt_.videoQkNormRotary(q, k, rope.first, rope.second);
            q = qk.first;
            k = qk.second;
        } else {
            if (qkv_.empty())
                qkv_.resize(36);
            auto &packed = qkv_[layer];
            if (!packed.first.data) {
                std::vector<Tensor> weights, biases;
                for (auto name : {"q", "k", "v"}) {
                    auto prefix = p + ".attn.to_" + name;
                    weights.push_back(weights_.get(prefix + ".weight"));
                    if (auto bias = weights_.optional(prefix + ".bias"))
                        biases.push_back(*bias);
                }
                if (!biases.empty() && biases.size() != 3)
                    throw std::runtime_error("inconsistent VAE QKV bias");
                packed.first = rt_.concat(weights);
                if (!biases.empty()) {
                    packed.second = rt_.concat(biases);
                    packed.second.cols *= packed.second.rows;
                    packed.second.rows = 1;
                }
                for (auto name : {"q", "k", "v"})
                    weights_.erasePrefix(p + ".attn.to_" + name);
            }
            auto projections = rt_.linear(y, packed.first, packed.second.data ? &packed.second : nullptr);
            auto qkv = rt_.videoQkvNormRotary(projections, rope.first, rope.second);
            q = qkv[0];
            k = qkv[1];
            v = qkv[2];
        }
        std::vector<Tensor> attention;
        for (int i = 0; i < batch; ++i)
            attention.push_back(rt_.selfAttention(q.rowSlice(i * sequence, sequence),
                                                 k.rowSlice(i * sequence, sequence),
                                                 v.rowSlice(i * sequence, sequence), 32));
        y = weights_.linear(batch == 1 ? attention[0] : rt_.concat(attention), p + ".attn.to_out.0");
        x = rt_.scaledResidual(x, y, fullWeights_.get(p + ".scale1"));
        y = rt_.videoRms(x, fullWeights_.get(p + ".norm2.weight"));
        Tensor output(y.rows, 2048, y.dtype);
        for (int first = 0; first < y.rows;) {
            // Bound low-precision scratch while sharing GEMMs across complete tiles.
            // The FP32 diagnostic path retains its original 512-row chunks.
            int count = std::min(y.isFloat32() ? 512 : 16384, y.rows - first);
            auto ff = weights_.linear(y.rowSlice(first, count), p + ".ff.net.0.proj");
            auto target = output.rowSlice(first, ff.rows);
            weights_.linear(rt_.siluMultiply(ff), p + ".ff.net.2", false, &target);
            first += count;
        }
        x = rt_.scaledResidual(x, output, fullWeights_.get(p + ".scale2"));
        if (trace)
            trace("block." + std::to_string(layer), x);
    }
    std::vector<Tensor> pixels;
    for (int i = 0; i < batch; ++i)
        pixels.push_back(x.rowSlice(i * sequence, tileRows));
    x = full_.norm(batch == 1 ? pixels[0] : full_.concat(pixels), false, &fullWeights_.get("decoder.norm_out.weight"),
                   &fullWeights_.get("decoder.norm_out.bias"), 1e-5f);
    // Output is patchified [T*H*W, 3*4*16*16].
    std::vector<Tensor> projected;
    for (int i = 0; i < batch; ++i)
        projected.push_back(weights_.linear(rt_.cast(x.rowSlice(i * tileRows, tileRows), rt_.dtype()),
                                           "decoder.proj_out"));
    return batch == 1 ? projected[0] : rt_.concat(projected);
}
namespace {
struct Tiles {
    std::vector<int> start, length, overlap;
};
Tiles split(int size, int tile, int minimumOverlap, int quantum) {
    if (size <= tile)
        return {{0}, {size}, {}};
    int count = (size + tile - 1) / tile;
    while (tile * count - minimumOverlap * (count - 1) < size)
        ++count;
    Tiles out;
    out.overlap.assign(count - 1, minimumOverlap);
    int slack = tile * count - minimumOverlap * (count - 1) - size;
    for (int i = 0; i < slack / quantum; ++i)
        out.overlap[i % (count - 1)] += quantum;
    out.start = {0};
    out.length.assign(count, tile);
    for (int i = 0; i < count - 1; ++i)
        out.start.push_back(out.start.back() + tile - out.overlap[i]);
    return out;
}
size_t pixel(const VideoTensor &x, int t, int y, int z, int c = 0) {
    return ((size_t(t) * x.height + y) * x.width + z) * x.channels + c;
}
VideoTensor crop(const VideoTensor &x, int t, int count, int y, int h, int z, int w) {
    VideoTensor out(count, h, w, x.channels);
    for (int k = 0; k < count; ++k)
        for (int j = 0; j < h; ++j)
            std::copy_n(x.values.data() + pixel(x, std::min(t + k, x.frames - 1), y + j, z),
                        size_t(w) * x.channels, out.values.data() + pixel(out, k, j, 0));
    return out;
}
} // namespace
Tensor VideoVAE::decodeClip(const VideoTensor &z, std::function<void()> checkpoint) {
    auto ys = split(z.height, 16, 4, 1), xs = split(z.width, 16, 4, 1);
    std::vector<Tensor> patches;
    patches.reserve(ys.start.size() * xs.start.size());
    // Tiles remain independent attention sequences; only row-wise projections share a batch.
    const size_t capacity = rt_.isFloat32() ? 1 : 4;
    const size_t count = ys.start.size() * xs.start.size();
    for (size_t first = 0; first < count; first += capacity) {
        size_t batch = std::min(capacity, count - first);
        std::vector<float> values;
        int rows = z.frames * ys.length[0] * xs.length[0];
        values.reserve(batch * rows * 24);
        for (size_t i = first; i < first + batch; ++i) {
            size_t yi = i / xs.start.size(), xi = i % xs.start.size();
            auto latent = crop(z, 0, z.frames, ys.start[yi], ys.length[yi], xs.start[xi], xs.length[xi]);
            values.insert(values.end(), latent.values.begin(), latent.values.end());
        }
        auto output = decodeTiles(full_.upload(values.data(), int(batch) * rows, 24), z.frames,
                                  ys.length[0], xs.length[0], int(batch),
                                  [&](const std::string &, const Tensor &) {
                                      if (checkpoint)
                                          checkpoint();
                                  });
        for (size_t i = 0; i < batch; ++i)
            patches.push_back(output.rowSlice(int(i) * rows, rows));
    }
    return rt_.stitchVideo(patches, z.frames, ys.length[0], xs.length[0], z.height, z.width,
                           ys.start, xs.start);
}
Tensor VideoVAE::decodeDevice(const VideoTensor &z, std::function<void(int, int)> progress,
                              std::function<void(const Tensor &)> onFrames) {
    if (z.channels != 24 || z.frames < 2 || z.height < 1 || z.width < 1 ||
        z.values.size() != size_t(z.frames) * z.height * z.width * 24)
        throw std::invalid_argument("invalid H3 video latents");
    int pad = (5 - (z.frames + 3) % 5) % 5, chunks = (z.frames + 3 + pad) / 5 - 1;
    if (!chunks)
        throw std::invalid_argument("too few H3 latent frames");
    int padFrames = 0;
    for (int i = 0; i < pad; ++i)
        padFrames += (z.frames + i) % 5 == 0 ? 1 : 4;
    int frames = chunks * 17 + 5 - padFrames;
    int64_t framePixels = int64_t(z.height) * z.width * 256;
    if (framePixels * frames > INT32_MAX)
        throw std::invalid_argument("H3 video output is too large");
    Tensor out(int(framePixels * frames), 3, DType::FP32), overlap;
    for (int i = 0; i < chunks; ++i) {
        auto clip = decodeClip(crop(z, i * 5, 7, 0, z.height, 0, z.width), [&] {
            if (progress)
                progress(i, chunks);
        });
        auto chunk = out.rowSlice(i * 17 * framePixels, std::min(17, frames - i * 17) * framePixels);
        rt_.copyVideoFrames(chunk, clip, overlap.rows ? &overlap : nullptr, framePixels, 3);
        if (onFrames)
            onFrames(chunk);
        overlap = clip.rowSlice(23 * framePixels, 5 * framePixels);
    }
    if (frames > chunks * 17) {
        auto tail = out.rowSlice(chunks * 17 * framePixels, (frames - chunks * 17) * framePixels);
        rt_.copyVideoFrames(tail, overlap, nullptr, framePixels, 0);
        if (onFrames)
            onFrames(tail);
    }
    if (progress)
        progress(chunks, chunks);
    return out;
}
VideoTensor VideoVAE::decode(const VideoTensor &z, std::function<void(int, int)> progress) {
    auto pixels = decodeDevice(z, progress);
    VideoTensor out;
    out.height = z.height * 16;
    out.width = z.width * 16;
    out.frames = pixels.rows / (out.height * out.width);
    out.channels = 3;
    out.values = full_.download(pixels);
    return out;
}
std::vector<uint8_t> VideoVAE::decodeRGB(const VideoTensor &z, std::function<void(int, int)> progress,
                                       std::function<void(const std::vector<uint8_t> &)> onFrames) {
    if (!onFrames)
        return rt_.downloadVideoRGB(decodeDevice(z, progress));
    std::vector<uint8_t> rgb;
    decodeDevice(z, progress, [&](const Tensor &frames) {
        auto chunk = rt_.downloadVideoRGB(frames);
        onFrames(chunk);
        rgb.insert(rgb.end(), chunk.begin(), chunk.end());
    });
    return rgb;
}
} // namespace minimax_h3
} // namespace fastllm
