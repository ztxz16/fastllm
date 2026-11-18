// C++/CUDA port of SolarX V3 stream_decoder at a7c61a6e02bef8539178a17b6e150a775b3a0ebb.
// See third_party/duo_solarx/LICENSE.
#include "solarx.h"
#include "tables.h"
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <future>

namespace fastllm::duo::live {
using at::Tensor;
namespace {
std::vector<int64_t> tileStarts(int size) {
    if (size <= 16)
        return {0};
    int count = (size - 4 + 11) / 12;
    int extra = 16 * count - 4 * (count - 1) - size;
    std::vector<int64_t> starts{0};
    for (int i = 0; i < count - 1; ++i)
        starts.push_back(starts.back() + 12 - extra / (count - 1) - (i < extra % (count - 1)));
    return starts;
}
Tensor rms(const Tensor &x, const Tensor &weight = {}) {
    return at::rms_norm(x.to(at::kFloat), {x.size(-1)}, weight, 1e-5).to(x.scalar_type());
}
Tensor rotate(const Tensor &x, const Tensor &c, const Tensor &s) {
    auto head = x.narrow(-1, 0, 48);
    auto rotated = at::cat({-head.narrow(-1, 24, 24), head.narrow(-1, 0, 24)}, -1);
    return at::cat({head * c + rotated * s, x.narrow(-1, 48, 16)}, -1);
}
Tensor join(const Tensor &prior, const Tensor &tile, int dim, int overlap) {
    std::vector<int64_t> shape(tile.dim(), 1);
    shape[dim] = overlap;
    auto alpha = (at::arange(overlap, tile.options()) / overlap).view(shape);
    auto mixed = prior.narrow(dim, prior.size(dim) - overlap, overlap) * (1 - alpha) +
                 tile.narrow(dim, 0, overlap) * alpha;
    return at::cat({mixed, tile.slice(dim, overlap)}, dim);
}
} // namespace
SolarX::SolarX(const std::string &file, int device, int secondDevice)
    : weights(at::Device(at::kCUDA, device)), weightFile(file), secondDevice(secondDevice) {
    c10::InferenceMode guard;
    weights.load(file, {"decoder.", "post_quant_conv."});
    auto constant = [&](const float *data, at::IntArrayRef shape) {
        return at::from_blob(const_cast<float *>(data), shape, at::kFloat).to(weights.device);
    };
    using namespace constants;
    mean = constant(video_latents_mean, {24, 1, 1, 1});
    deviation = constant(video_latents_std, {24, 1, 1, 1});
    const float means[]{.485f, .456f, .406f}, stds[]{.229f, .224f, .225f};
    rgbMean = constant(means, {3, 1, 1, 1});
    rgbStd = constant(stds, {3, 1, 1, 1});
    first = {constant(first_window_a, {3, 18}), constant(first_window_b, {3, 18}),
             constant(first_window_c, {3, 18})};
    steady = {constant(steady_a, {7, 3, 4}), constant(steady_b, {7, 3, 4})};
}
void SolarX::reset() {
    history.clear();
    index = 0;
    window = -1;
    failed = false;
    valid.fill(false);
}
void SolarX::plan(int h, int w, int assignedTiles) {
    height = h;
    width = w;
    ys = tileStarts(h);
    xs = tileStarts(w);
    th = std::min(h, 16);
    tw = std::min(w, 16);
    auto opts = weights.get("decoder.x_embedder.weight").options();
    std::vector<Tensor> axes;
    for (int n : {7, th, tw})
        axes.push_back((at::arange(.5, n, opts) / n) * 2. - 1.);
    auto ids = at::stack(at::meshgrid(axes, "ij"), -1).reshape({-1, 3});
    ids = at::cat({ids, at::zeros({5, 3}, opts)});
    // The upstream .to(bfloat16) also converts the inv_freq buffer.
    auto freq = at::pow(100., at::arange(8, opts.dtype(at::kFloat)) / 8.).reciprocal().to(at::kBFloat16);
    auto angles = ((ids.unsqueeze(-1) * (2. * M_PI)) * freq).flatten(1).repeat({1, 2}).view({1, -1, 1, 48});
    cosine = angles.cos();
    sine = angles.sin();
    const int tiles = xs.size() * ys.size();
    localTiles = assignedTiles ? assignedTiles : tiles;
    if (secondDevice >= 0 && tiles > 1) {
        localTiles = (tiles + 1) / 2;
        c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, secondDevice));
        if (!second)
            second = std::make_unique<SolarX>(weightFile, secondDevice);
        second->plan(h, w, tiles - localTiles);
    }
    keys.clear();
    values.clear();
    for (int i = 0; i < 36; ++i) {
        keys.push_back(at::empty({localTiles, 5 + 7 * th * tw, 32, 64}, opts));
        values.push_back(at::empty_like(keys.back()));
    }
}
Tensor SolarX::layers(const Tensor &tiles, const std::vector<int> &fresh, int slot) {
    const int batch = tiles.size(0), hw = th * tw, n = fresh.size() * hw;
    auto x = weights.linear(tiles.permute({0, 2, 3, 4, 1}).reshape({batch, n, 24}), "decoder.x_embedder");
    x = at::cat({x, weights.get("decoder.register_tokens").expand({batch, -1, -1}),
                 at::zeros({batch, 1, 2048}, x.options())},
                1);
    // All re-evaluated slots form one contiguous run at refresh or window rollover.
    int begin = fresh.front();
    auto positions = [&](const Tensor &table) {
        return at::cat({table.narrow(1, begin * hw, n), table.narrow(1, 7 * hw, 5)}, 1);
    };
    auto c = positions(cosine), s = positions(sine);
    for (int i = 0; i < 36; ++i) {
        auto p = "decoder.transformer_blocks." + std::to_string(i);
        auto qkv = weights.linear(rms(x, weights.get(p + ".norm1.weight")), p + ".attn.to_qkv")
                       .view({batch, n + 5, 32, 192})
                       .chunk(3, -1);
        auto q = rotate(rms(qkv[0]), c, s), k = rotate(rms(qkv[1]), c, s);
        auto cache = [&](Tensor &dest, const Tensor &source) {
            dest.narrow(1, 0, 5).copy_(source.narrow(1, n, 5));
            dest.narrow(1, 5 + begin * hw, n).copy_(source.narrow(1, 0, n));
            return dest.narrow(1, 0, 5 + (slot + 1) * hw).transpose(1, 2);
        };
        auto ck = cache(keys[i], k), cv = cache(values[i], qkv[2]);
        auto y = at::scaled_dot_product_attention(q.transpose(1, 2), ck, cv).transpose(1, 2).reshape_as(x);
        x = x + weights.linear(y, p + ".attn.to_out") * weights.get(p + ".scale1");
        auto f = weights.linear(rms(x, weights.get(p + ".norm2.weight")), p + ".ff.w1").chunk(2, -1);
        x = x + weights.linear(at::silu(f[0]) * f[1], p + ".ff.w2") * weights.get(p + ".scale2");
    }
    x = at::layer_norm(x.narrow(1, n - hw, hw).contiguous(), {2048}, weights.get("decoder.norm_out.weight"),
                       weights.get("decoder.norm_out.bias"), 1e-5);
    return weights.linear(x, "decoder.proj_out")
        .view({batch, th, tw, 3, 4, 16, 16})
        .permute({0, 3, 4, 1, 5, 2, 6})
        .reshape({batch, 3, 4, th * 16, tw * 16});
}
Tensor SolarX::blend(const Tensor &tiles) {
    std::vector<Tensor> rows;
    for (size_t i = 0; i < ys.size(); ++i) {
        std::vector<Tensor> row;
        for (size_t j = 0; j < xs.size(); ++j) {
            auto tile = tiles[i * xs.size() + j];
            if (i)
                tile = join(tiles[(i - 1) * xs.size() + j], tile, 2, (ys[i - 1] + th - ys[i]) * 16);
            if (j)
                tile = join(tiles[i * xs.size() + j - 1], tile, 3, (xs[j - 1] + tw - xs[j]) * 16);
            if (i + 1 < ys.size())
                tile = tile.narrow(2, 0, (ys[i + 1] - ys[i]) * 16);
            if (j + 1 < xs.size())
                tile = tile.narrow(3, 0, (xs[j + 1] - xs[j]) * 16);
            row.push_back(tile);
        }
        rows.push_back(at::cat(row, 3));
    }
    return at::cat(rows, 2);
}
Tensor SolarX::push(const Tensor &latent) {
    const int start = ((std::max(0, index - 6) + 4) / 5) * 5, slot = index - start;
    if (start != window) {
        history.erase(history.begin(), history.lower_bound(start));
        valid.fill(false);
        window = start;
    }
    auto z = at::conv3d(latent, weights.get("post_quant_conv.weight"), weights.get("post_quant_conv.bias"));
    std::vector<Tensor> tiles;
    for (auto y : ys)
        for (auto x : xs)
            tiles.push_back(z.narrow(3, y, th).narrow(4, x, tw));
    history[index] = at::cat(tiles);
    std::vector<int> fresh;
    tiles.clear();
    for (int i = 0; i <= slot; ++i)
        if (!valid[i] || (index == 1 && i == 0)) {
            fresh.push_back(i);
            tiles.push_back(history.at(start + i));
            valid[i] = true;
        }
    int count = (index % 5 == 0) ? 1 : 4;
    auto input = at::cat(tiles, 2);
    Tensor decoded;
    if (localTiles < input.size(0)) {
        // Tile attention is independent. Keep each tile's KV on its assigned GPU;
        // merge predictions before the original blending and color correction.
        auto shard = input.slice(0, localTiles).to(second->weights.device, input.scalar_type(), true);
        auto stream = c10::cuda::getCurrentCUDAStream(secondDevice);
        auto other = std::async(std::launch::async, [this, shard, stream, &fresh, slot] {
            c10::InferenceMode inference;
            c10::cuda::CUDAStreamGuard guard(stream);
            return second->layers(shard, fresh, slot);
        });
        auto local = layers(input.narrow(0, 0, localTiles), fresh, slot);
        auto remote = other.get();
        decoded = at::cat({local, remote.to(weights.device, remote.scalar_type(), true)});
    } else {
        decoded = layers(input, fresh, slot);
    }
    auto out = blend(decoded).slice(1, 4 - count).to(at::kFloat);
    auto pick = [&](int k) {
        int frame = 17 * (index / 5) + (index % 5 ? 1 + 4 * (index % 5 - 1) : 0);
        return (index <= 5 ? first[k].narrow(1, frame, count) : steady[k][slot].slice(1, 4 - count))
            .view({3, count, 1, 1});
    };
    auto shift = pick(0) + pick(1) * out;
    if (index <= 5)
        shift = shift + pick(2) * out * out;
    ++index;
    return (out - shift).to(at::kBFloat16);
}
Tensor SolarX::decode(const Tensor &packet) {
    c10::InferenceMode guard;
    if (failed)
        throw std::logic_error("reset SolarX after a failed decode");
    if (packet.dim() != 4 || packet.size(0) != 24 || packet.size(1) != 5 || packet.size(2) < 1 ||
        packet.size(3) < 1)
        throw std::invalid_argument("expected normalized [24,5,h,w] video");
    if (index && (packet.size(2) != height || packet.size(3) != width))
        throw std::invalid_argument("reset SolarX before changing resolution");
    try {
        if (packet.size(2) != height || packet.size(3) != width || keys.empty())
            plan(packet.size(2), packet.size(3));
        auto z =
            (packet.to(weights.device, at::kBFloat16).to(at::kFloat) * deviation + mean).to(at::kBFloat16);
        std::vector<Tensor> frames;
        for (int t = 0; t < 5; ++t)
            frames.push_back(push(z.narrow(1, t, 1).unsqueeze(0)));
        auto rgb = at::cat(frames, 1).to(at::kFloat) * rgbStd + rgbMean;
        if (!at::isfinite(rgb).all().item<bool>())
            throw std::runtime_error("nonfinite SolarX output");
        return (rgb.clamp(0, 1) * 255).round().to(at::kByte).permute({1, 2, 3, 0}).contiguous();
    } catch (...) {
        failed = true;
        throw;
    }
}
} // namespace fastllm::duo::live
