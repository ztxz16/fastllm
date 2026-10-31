#include "models/minimax_h3/runtime.h"
#include <algorithm>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
TextEncoder::TextEncoder(Runtime &rt, const std::string &dir)
    : rt_(rt), weights_(rt, componentWeights(dir)) {
    weights_.enableHostCache(size_t(64) << 30);
}
Tensor TextEncoder::encode(const std::vector<int> &tokens, Trace trace) {
    if (tokens.empty() || tokens.size() > 4096)
        throw std::invalid_argument("H3 prompt requires 1..4096 tokens");
    const std::string root = "model.language_model.";
    auto x = rt_.embedding(weights_.get(root + "embed_tokens.weight"), tokens);
    weights_.erasePrefix(root + "embed_tokens.");
    for (int layer = 0; layer < 50; ++layer) {
        auto p = root + "layers." + std::to_string(layer) + ".";
        auto y = rt_.norm(x, true, &weights_.get(p + "input_layernorm.weight"));
        auto q = weights_.linear(y, p + "self_attn.q_proj"), k = weights_.linear(y, p + "self_attn.k_proj"),
             v = weights_.linear(y, p + "self_attn.v_proj");
        q = rt_.textRope(rt_.norm(q, true, &weights_.get(p + "self_attn.q_norm.weight"), nullptr, 1e-6f, 128),
                         5000000.f);
        k = rt_.textRope(rt_.norm(k, true, &weights_.get(p + "self_attn.k_norm.weight"), nullptr, 1e-6f, 128),
                         5000000.f);
        x = rt_.add(
            x, weights_.linear(rt_.attention(q, k, v, 64, nullptr, 128, 8, true), p + "self_attn.o_proj"));
        y = rt_.norm(x, true, &weights_.get(p + "post_attention_layernorm.weight"));
        std::vector<Tensor> chunks;
        for (int first = 0; first < y.rows; first += 256) {
            auto input = y.rowSlice(first, std::min(256, y.rows - first));
            auto gate = rt_.activation(weights_.linear(input, p + "mlp.gate_proj"));
            chunks.push_back(weights_.linear(rt_.multiply(gate, weights_.linear(input, p + "mlp.up_proj")),
                                             p + "mlp.down_proj"));
        }
        x = rt_.add(x, chunks.size() == 1 ? chunks[0] : rt_.concat(chunks));
        if (trace)
            trace("layer." + std::to_string(layer), x);
        weights_.erasePrefix(p);
    }
    // hidden_states[50] is the residual before Qwen's final norm, not last_hidden_state.
    return x;
}
} // namespace minimax_h3
} // namespace fastllm
