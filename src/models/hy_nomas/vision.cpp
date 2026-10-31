#include "models/hy_nomas/runtime.h"

namespace fastllm {
namespace nomas {
std::vector<Tensor> qwenVision(Runtime &rt, Weights &weights, int width, int height,
                               const std::vector<uint8_t> &rgba,
                               std::function<void(const std::string &, const Tensor &)> trace,
                               const std::vector<uint8_t> *secondFrame) {

    const std::string root = "model.visual.";
    auto inputs = rt.visionInputs(width, height, rgba, weights.get(root + "pos_embed.weight"), secondFrame);
    auto x = weights.linear(inputs[0], root + "patch_embed.proj", true);
    if (trace)
        trace("vision.patch", x);
    x = rt.add(x, inputs[1]);
    inputs[0] = {};
    inputs[1] = {};
    if (trace)
        trace("vision.embedding", x);
    auto merge = [&](Tensor input, const std::string &p, bool post) {
        if (post) {
            input.rows /= 4;
            input.cols *= 4;
        }
        input = rt.norm(input, false, &weights.get(p + ".norm.weight"), &weights.get(p + ".norm.bias"));
        if (!post) {
            input.rows /= 4;
            input.cols *= 4;
        }
        return weights.linear(rt.geluExact(weights.linear(input, p + ".linear_fc1")), p + ".linear_fc2");
    };
    const auto &ropePositions = inputs[2];
    std::vector<Tensor> result(4);
    for (int layer = 0; layer < 27; ++layer) {
        auto p = root + "blocks." + std::to_string(layer) + ".";
        if (trace)
            trace("vision.input." + std::to_string(layer), x);
        auto nx = rt.norm(x, false, &weights.get(p + "norm1.weight"), &weights.get(p + "norm1.bias"));
        if (trace && layer == 0)
            trace("vision.norm1", nx);
        auto qkv = weights.linear(nx, p + "attn.qkv");
        if (trace && layer == 0)
            trace("vision.qkv", qkv);
        auto q = rt.visionRope(rt.columns(qkv, 0, 1152), ropePositions),
             k = rt.visionRope(rt.columns(qkv, 1152, 1152), ropePositions), v = rt.columns(qkv, 2304, 1152);
        auto attn = rt.attention(q, k, v, 16);
        if (trace && layer == 0)
            trace("vision.attention", attn);
        auto projected = weights.linear(attn, p + "attn.proj");
        if (trace && layer == 0)
            trace("vision.projected", projected);
        x = rt.add(x, projected);
        if (trace && layer == 0)
            trace("vision.residual", x);
        nx = rt.norm(x, false, &weights.get(p + "norm2.weight"), &weights.get(p + "norm2.bias"));
        if (trace && layer == 0)
            trace("vision.norm2", nx);
        auto hidden = weights.linear(nx, p + "mlp.linear_fc1");
        if (trace && layer == 0)
            trace("vision.fc1", hidden);
        hidden = rt.activation(hidden, true);
        if (trace && layer == 0)
            trace("vision.gelu", hidden);
        hidden = weights.linear(hidden, p + "mlp.linear_fc2");
        if (trace && layer == 0)
            trace("vision.fc2", hidden);
        x = rt.add(x, hidden);
        if (trace)
            trace("vision.block." + std::to_string(layer), x);
        if (layer == 8 || layer == 16 || layer == 24) {
            int i = layer / 8 - 1;
            result[i + 1] = merge(x, root + "deepstack_merger_list." + std::to_string(i), true);
        }
    }
    result[0] = merge(x, root + "merger", false);
    if (trace)
        for (int i = 0; i < 4; ++i)
            trace("vision.output." + std::to_string(i), result[i]);
    return result;
}
} // namespace nomas
} // namespace fastllm
