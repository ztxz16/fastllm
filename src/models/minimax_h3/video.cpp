#include "models/minimax_h3/runtime.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
VideoVAE::VideoVAE(Runtime &rt, const std::string &dir) : rt_(rt), weights_(rt, componentWeights(dir)) {}
Tensor VideoVAE::decodeTile(const Tensor &latent, int frames, int height, int width, Trace trace) {
    if (latent.rows != frames * height * width || latent.cols != 24)
        throw std::invalid_argument("invalid H3 VAE tile");
    auto x = weights_.linear(weights_.linear(latent, "post_quant_conv"), "decoder.proj_in");
    auto registers = weights_.get("decoder.register_tokens");
    registers.rows = 4;
    registers.cols = 2048;
    std::vector<float> zeros(2048), positions(size_t(x.rows + 5) * 3);
    for (int t = 0; t < frames; ++t)
        for (int h = 0; h < height; ++h)
            for (int w = 0; w < width; ++w) {
                size_t i = (size_t(t) * height + h) * width + w;
                positions[i * 3] = (2.f * ((t + .5f) / frames) - 1.f) * (2.f * float(M_PI));
                positions[i * 3 + 1] = (2.f * ((h + .5f) / height) - 1.f) * (2.f * float(M_PI));
                positions[i * 3 + 2] = (2.f * ((w + .5f) / width) - 1.f) * (2.f * float(M_PI));
            }
    auto rope = rt_.ropeTable(positions, 8, 100.f);
    x = rt_.concat({x, registers, rt_.upload(zeros.data(), 1, 2048)});
    for (int layer = 0; layer < 36; ++layer) {
        auto p = "decoder.transformer_blocks." + std::to_string(layer);
        auto y = rt_.norm(x, true, &weights_.get(p + ".norm1.weight"), nullptr, 1e-5f);
        auto q = weights_.linear(y, p + ".attn.to_q"), k = weights_.linear(y, p + ".attn.to_k"),
             v = weights_.linear(y, p + ".attn.to_v");
        q = rt_.rotary(rt_.norm(q, true, nullptr, nullptr, 1e-5f, 64), rope.first, rope.second, 64);
        k = rt_.rotary(rt_.norm(k, true, nullptr, nullptr, 1e-5f, 64), rope.first, rope.second, 64);
        y = weights_.linear(rt_.selfAttention(q, k, v, 32), p + ".attn.to_out.0");
        x = rt_.add(x, rt_.multiply(y, rt_.repeatRows(weights_.get(p + ".scale1"), x.rows)));
        y = rt_.norm(x, true, &weights_.get(p + ".norm2.weight"), nullptr, 1e-5f);
        std::vector<Tensor> chunks;
        for (int first = 0; first < y.rows; first += 512) {
            auto ff = weights_.linear(y.rowSlice(first, std::min(512, y.rows - first)), p + ".ff.net.0.proj");
            ff = rt_.multiply(rt_.columns(ff, 0, 8192), rt_.activation(rt_.columns(ff, 8192, 8192)));
            chunks.push_back(weights_.linear(ff, p + ".ff.net.2"));
        }
        y = chunks.size() == 1 ? chunks[0] : rt_.concat(chunks);
        x = rt_.add(x, rt_.multiply(y, rt_.repeatRows(weights_.get(p + ".scale2"), x.rows)));
        if (trace)
            trace("block." + std::to_string(layer), x);
    }
    x = rt_.norm(x.rowSlice(0, latent.rows), false, &weights_.get("decoder.norm_out.weight"),
                 &weights_.get("decoder.norm_out.bias"), 1e-5f);
    // Output is patchified [T*H*W, 3*4*16*16].
    return weights_.linear(x, "decoder.proj_out");
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
void blend(const VideoTensor &a, VideoTensor &b, int extent, int axis) {
    for (int t = 0; t < b.frames; ++t)
        for (int y = 0; y < b.height; ++y)
            for (int x = 0; x < b.width; ++x) {
                int position = axis == 0 ? t : axis == 1 ? y : x;
                if (position >= extent)
                    continue;
                int at = axis == 0 ? a.frames - extent + t : t, ay = axis == 1 ? a.height - extent + y : y,
                    ax = axis == 2 ? a.width - extent + x : x;
                float wb = float(position) / extent, wa = 1.f - wb;
                for (int c = 0; c < b.channels; ++c)
                    b.values[pixel(b, t, y, x, c)] =
                        a.values[pixel(a, at, ay, ax, c)] * wa + b.values[pixel(b, t, y, x, c)] * wb;
            }
}
} // namespace
VideoTensor VideoVAE::decodeClip(const VideoTensor &z, std::function<void()> checkpoint) {
    auto ys = split(z.height, 16, 4, 1), xs = split(z.width, 16, 4, 1);
    std::vector<std::vector<VideoTensor>> tiles(ys.start.size());
    for (size_t i = 0; i < ys.start.size(); ++i)
        for (size_t j = 0; j < xs.start.size(); ++j) {
            auto latent = crop(z, 0, z.frames, ys.start[i], ys.length[i], xs.start[j], xs.length[j]);
            auto patches = rt_.download(decodeTile(
                rt_.upload(latent.values.data(), latent.frames * latent.height * latent.width, 24),
                latent.frames, latent.height, latent.width, [&](const std::string &, const Tensor &) {
                    if (checkpoint)
                        checkpoint();
                }));
            VideoTensor tile(z.frames * 4, latent.height * 16, latent.width * 16, 3);
            for (int t = 0; t < tile.frames; ++t)
                for (int y = 0; y < tile.height; ++y)
                    for (int x = 0; x < tile.width; ++x)
                        for (int c = 0; c < 3; ++c) {
                            size_t row = (size_t(t / 4) * latent.height + y / 16) * latent.width + x / 16;
                            int column = ((c * 4 + t % 4) * 16 + y % 16) * 16 + x % 16;
                            tile.values[pixel(tile, t, y, x, c)] = patches[row * 3072 + column];
                        }
            tiles[i].push_back(std::move(tile));
        }
    VideoTensor out(z.frames * 4, z.height * 16, z.width * 16, 3);
    for (size_t i = 0; i < tiles.size(); ++i)
        for (size_t j = 0; j < tiles[i].size(); ++j) {
            auto tile = tiles[i][j];
            // The reference blends against the raw adjacent tiles, not previously blended ones.
            if (i)
                blend(tiles[i - 1][j], tile, ys.overlap[i - 1] * 16, 1);
            if (j)
                blend(tiles[i][j - 1], tile, xs.overlap[j - 1] * 16, 2);
            int h = tile.height - (i + 1 < tiles.size() ? ys.overlap[i] * 16 : 0);
            int w = tile.width - (j + 1 < tiles[i].size() ? xs.overlap[j] * 16 : 0);
            for (int t = 0; t < out.frames; ++t)
                for (int y = 0; y < h; ++y)
                    std::copy_n(tile.values.data() + pixel(tile, t, y, 0), w * 3,
                                out.values.data() + pixel(out, t, ys.start[i] * 16 + y, xs.start[j] * 16));
        }
    return out;
}
VideoTensor VideoVAE::decode(const VideoTensor &z, std::function<void(int, int)> progress) {
    if (z.channels != 24 || z.frames < 2 || z.height < 1 || z.width < 1)
        throw std::invalid_argument("invalid H3 video latents");
    int pad = (5 - (z.frames + 3) % 5) % 5, chunks = (z.frames + 3 + pad) / 5 - 1;
    std::vector<VideoTensor> decoded;
    VideoTensor overlap;
    for (int i = 0; i < chunks; ++i) {
        auto clip = decodeClip(crop(z, i * 5, 7, 0, z.height, 0, z.width), [&] {
            if (progress)
                progress(i, chunks);
        });
        auto chunk = crop(clip, 3, 17, 0, clip.height, 0, clip.width);
        if (overlap.frames)
            blend(overlap, chunk, 5, 0);
        decoded.push_back(std::move(chunk));
        overlap = crop(clip, 23, 5, 0, clip.height, 0, clip.width);
    }
    if (!overlap.frames)
        throw std::invalid_argument("too few H3 latent frames");
    decoded.push_back(std::move(overlap));
    int padFrames = 0;
    for (int i = 0; i < pad; ++i)
        padFrames += (z.frames + i) % 5 == 0 ? 1 : 4;
    VideoTensor out(chunks * 17 + 5 - padFrames, z.height * 16, z.width * 16, 3);
    size_t offset = 0;
    for (auto &clip : decoded) {
        size_t count = std::min(clip.values.size(), out.values.size() - offset);
        std::copy_n(clip.values.data(), count, out.values.data() + offset);
        offset += count;
        if (offset == out.values.size())
            break;
    }
    if (progress)
        progress(chunks, chunks);
    return out;
}
} // namespace minimax_h3
} // namespace fastllm
