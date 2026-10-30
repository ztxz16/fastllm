#include "models/minimax_h3/runtime.h"

namespace fastllm {
namespace minimax_h3 {
AudioVAE::AudioVAE(Runtime &rt, const std::string &dir) : rt_(rt), weights_(rt, componentWeights(dir)) {}
Tensor AudioVAE::conv(const Tensor &x, const std::string &p, int kernel, int stride, int dilation,
                      bool transposed) {
    auto found = fused_.find(p);
    if (found == fused_.end()) {
        auto weight = rt_.weightNorm(weights_.get(p + ".weight_v"), weights_.get(p + ".weight_g"));
        if (transposed)
            weight = rt_.transposeConvWeight(weight, kernel);
        found = fused_.emplace(p, weight).first;
        weights_.erasePrefix(p + ".weight_");
    }
    int padding = transposed ? (kernel - stride) / 2 : (kernel - 1) * dilation / 2;
    return rt_.conv1d(x, found->second, weights_.optional(p + ".bias"), kernel, stride, padding, dilation,
                      transposed);
}
Tensor AudioVAE::activation(const Tensor &x, const std::string &p) {
    auto y = rt_.audioFilter(x, weights_.get(p + ".upsample.filter"), true);
    y = rt_.snake(y, weights_.get(p + ".act.alpha"), weights_.get(p + ".act.beta"));
    return rt_.audioFilter(y, weights_.get(p + ".downsample.lowpass.filter"), false);
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
