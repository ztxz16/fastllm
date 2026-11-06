#include "models/minimax_h3/runtime.h"
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
AudioVAE::AudioVAE(Runtime &rt, const std::string &dir) : rt_(rt), weights_(rt, componentWeights(dir)) {}
Tensor AudioVAE::conv(const Tensor &x, const std::string &p, int kernel, int stride, int dilation,
                      bool transposed, int padding) {
    auto found = fused_.find(p);
    if (found == fused_.end()) {
        auto weight = rt_.weightNorm(weights_.get(p + ".weight_v"), weights_.get(p + ".weight_g"));
        if (transposed)
            weight = rt_.transposeConvWeight(weight, kernel);
        found = fused_.emplace(p, weight).first;
        weights_.erasePrefix(p + ".weight_");
    }
    if (padding < 0)
        padding = transposed ? (kernel - stride) / 2 : (kernel - 1) * dilation / 2;
    return rt_.conv1d(x, found->second, weights_.optional(p + ".bias"), kernel, stride, padding, dilation,
                      transposed);
}
Tensor AudioVAE::activation(const Tensor &x, const std::string &p) {
    auto y = rt_.audioFilter(x, weights_.get(p + ".upsample.filter"), true);
    y = rt_.snake(y, weights_.get(p + ".act.alpha"), weights_.get(p + ".act.beta"));
    return rt_.audioFilter(y, weights_.get(p + ".downsample.lowpass.filter"), false);
}
Tensor AudioVAE::encode(const Tensor &samples, Trace trace) {
    if (!samples.isFloat32() || samples.cols != 1 || samples.rows < 1)
        throw std::invalid_argument("H3 audio encoder requires mono FP32 PCM");
    auto x = samples;
    int pad = (800 - x.rows % 800) % 800;
    if (pad) {
        std::vector<float> zeros(pad);
        x = rt_.concat({x, rt_.upload(zeros.data(), pad, 1)});
    }
    auto snake = [&](const Tensor &input, const std::string &p) {
        auto &alpha = weights_.get(p + ".alpha");
        return rt_.snake(input, alpha, alpha, false);
    };
    auto norm = [&](const Tensor &input, const std::string &p) {
        return rt_.norm(input, false, &weights_.get(p + ".weight"), &weights_.get(p + ".bias"), 1e-5f);
    };
    x = conv(x, "encoder.block.0", 7);
    if (trace)
        trace("encoder.block.0", x);
    const int strides[] = {2, 4, 4, 5, 5}, dilations[] = {1, 3, 9};
    for (int i = 0; i < 5; ++i) {
        auto p = "encoder.block." + std::to_string(i + 1) + ".block";
        for (int j = 0; j < 3; ++j) {
            auto r = p + "." + std::to_string(j) + ".block";
            auto y = conv(snake(x, r + ".0"), r + ".1", 7, 1, dilations[j]);
            y = conv(snake(y, r + ".2"), r + ".3", 1);
            x = rt_.add(x, y);
        }
        x = conv(snake(x, p + ".3"), p + ".4", 2 * strides[i], strides[i], 1, false, (strides[i] + 1) / 2);
        if (trace)
            trace("encoder.block." + std::to_string(i + 1), x);
    }
    x = conv(snake(x, "encoder.block.6"), "encoder.block.7", 3);
    if (trace)
        trace("encoder", x);
    auto qkv = weights_.linear(norm(x, "pre_block.norm1"), "pre_block.attn.qkv");
    auto q = rt_.bias(rt_.columns(qkv, 0, 2048), weights_.get("pre_block.attn.q_bias"));
    auto k = rt_.columns(qkv, 2048, 2048);
    auto v = rt_.bias(rt_.columns(qkv, 4096, 2048), weights_.get("pre_block.attn.v_bias"));
    auto attended = rt_.poolAudioHeads(rt_.attention(q, k, v, 8, nullptr, 128, 8, true), 8, 32);
    x = rt_.add(weights_.linear(norm(x, "pre_block.norm3"), "pre_block.proj"),
                weights_.linear(attended, "pre_block.attn.proj"));
    auto y = norm(norm(x, "pre_block.norm2"), "pre_block.mlp.norm");
    auto gate = rt_.activation(weights_.linear(y, "pre_block.mlp.w0"), true);
    x = rt_.add(
        x, weights_.linear(rt_.multiply(gate, weights_.linear(y, "pre_block.mlp.w1")), "pre_block.mlp.w2"));
    if (trace)
        trace("pre_block", x);
    x = weights_.linear(x, "mean_proj");
    if (trace)
        trace("mean_proj", x);
    return x;
}
Tensor AudioVAE::decode(const Tensor &latent, Trace trace) {
    auto x = conv(weights_.linear(latent, "dec_in_proj"), "decoder.conv_pre", 7);
    if (trace)
        trace("conv_pre", x);
    const int rates[] = {5, 5, 2, 2, 2, 2, 2}, kernels[] = {9, 9, 4, 4, 4, 4, 4}, resKernels[] = {3, 7, 11},
              dilations[] = {1, 3, 5};
    for (int i = 0; i < 7; ++i) {
        x = conv(x, "decoder.ups." + std::to_string(i) + ".0", kernels[i], rates[i], 1, true);
        if (trace)
            trace("ups." + std::to_string(i), x);
        Tensor sum;
        for (int j = 0; j < 3; ++j) {
            auto p = "decoder.resblocks." + std::to_string(i * 3 + j);
            auto y = x;
            for (int d = 0; d < 3; ++d) {
                auto residual = conv(activation(y, p + ".activations." + std::to_string(d * 2)),
                                     p + ".convs1." + std::to_string(d), resKernels[j], 1, dilations[d]);
                residual = conv(activation(residual, p + ".activations." + std::to_string(d * 2 + 1)),
                                p + ".convs2." + std::to_string(d), resKernels[j]);
                y = rt_.add(residual, y);
            }
            sum = j ? rt_.add(sum, y) : y;
        }
        x = rt_.affine(sum, 1.f / 3.f);
        if (trace)
            trace("stage." + std::to_string(i), x);
    }
    return rt_.affine(conv(activation(x, "decoder.activation_post"), "decoder.conv_post", 7), 1, 0, true);
}
} // namespace minimax_h3
} // namespace fastllm
