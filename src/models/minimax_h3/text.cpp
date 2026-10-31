#include "models/minimax_h3/runtime.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <future>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
TextEncoder::TextEncoder(Runtime &rt, const std::string &dir) : rt_(rt), weights_(rt, componentWeights(dir)) {
    weights_.enableHostCache(size_t(64) << 30);
    rt_.activate();
    size_t available = 0, total = 0;
    auto status = cudaMemGetInfo(&available, &total);
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    // Two staging layers fit alongside the resident DiT on 72/80 GiB devices.
    // Keep the original memory footprint for smaller devices and FP32 diagnostics.
    if (!rt.isFloat32() && total >= (size_t(68) << 30) && available >= (size_t(8) << 30)) {
        prefetchedWeights_ = std::make_unique<Weights>(rt, componentWeights(dir));
        prefetchedWeights_->shareHostCache(weights_);
    }
}
Tensor TextEncoder::encode(const std::vector<int> &tokens, Trace trace) {
    return encodeVisual(tokens, {}, std::move(trace));
}
Tensor TextEncoder::encodeVisual(const std::vector<int> &tokens, const std::vector<VisionInput> &images,
                                 Trace trace) {
    if (tokens.empty() || tokens.size() > 65536)
        throw std::invalid_argument("H3 presentation requires 1..65536 tokens");
    const std::string root = "model.language_model.";
    auto x = rt_.embedding(weights_.get(root + "embed_tokens.weight"), tokens);
    weights_.erasePrefix(root + "embed_tokens.");
    std::vector<int> positions(tokens.size() * 3), starts;
    std::vector<std::vector<Tensor>> visual;
    int cursor = 0, position = 0;
    for (const auto &visualInput : images) {
        const auto &image = visualInput.first;
        int gh = image.height / 32, gw = image.width / 32;
        while (cursor < int(tokens.size()) && tokens[cursor] != visualInput.token) {
            for (int axis = 0; axis < 3; ++axis)
                positions[axis * tokens.size() + cursor] = position;
            ++position;
            ++cursor;
        }
        starts.push_back(cursor);
        for (int i = 0; i < gh * gw; ++i, ++cursor) {
            if (cursor >= int(tokens.size()) || tokens[cursor] != visualInput.token)
                throw std::invalid_argument("H3 image slot count mismatch");
            positions[cursor] = position;
            positions[tokens.size() + cursor] = position + i / gw;
            positions[tokens.size() * 2 + cursor] = position + i % gw;
        }
        position += std::max(gh, gw);
        visual.push_back(nomas::qwenVision(rt_, weights_, image.width, image.height, image.rgba, trace,
                                           visualInput.second ? &visualInput.second->rgba : nullptr));
    }
    for (; cursor < int(tokens.size()); ++cursor, ++position)
        for (int axis = 0; axis < 3; ++axis)
            positions[axis * tokens.size() + cursor] = position;
    weights_.erasePrefix("model.visual.");
    auto applyVisual = [&](const Tensor &input, int deep) {
        if (visual.empty())
            return input;
        std::vector<Tensor> parts;
        int first = 0;
        for (size_t i = 0; i < visual.size(); ++i) {
            if (starts[i] > first)
                parts.push_back(input.rowSlice(first, starts[i] - first));
            const auto &features = visual[i][deep < 0 ? 0 : deep + 1];
            parts.push_back(deep < 0 ? features
                                     : rt_.add(input.rowSlice(starts[i], features.rows), features));
            first = starts[i] + features.rows;
        }
        if (first < input.rows)
            parts.push_back(input.rowSlice(first, input.rows - first));
        return rt_.concat(parts);
    };
    x = applyVisual(x, -1);
    Tensor ropePositions;
    if (!images.empty())
        ropePositions = rt_.indices(positions);
    // Populate the CPU cache serially on the first request. Short prompts have
    // too little compute to benefit from an additional staging stream.
    bool prefetch = prefetchedWeights_ && hostCacheWarm_ && tokens.size() >= 256;
    auto layerWeights = [&](int layer) -> Weights & {
        return prefetch && layer % 2 ? *prefetchedWeights_ : weights_;
    };
    auto loadLayer = [&](int layer) {
        // Only this worker populates the shared CPU cache. Inference reads the
        // other, fully uploaded GPU cache and uses the main thread's cuBLAS handle.
        auto &weights = layerWeights(layer);
        auto p = root + "layers." + std::to_string(layer) + ".";
        for (auto name : {"input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                          "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
                          "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
            weights.get(p + name + ".weight");
            weights.optional(p + name + ".bias");
        }
        auto status = cudaStreamSynchronize(cudaStreamPerThread);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    };
    // A future also joins the loader during exception unwinding, before callers
    // can unload either cache. No worker or CUDA copy outlives this request.
    std::future<void> pending;
    if (prefetch)
        pending = std::async(std::launch::async, loadLayer, 0);
    for (int layer = 0; layer < 50; ++layer) {
        if (prefetch) {
            pending.get();
            if (layer + 1 < 50)
                pending = std::async(std::launch::async, loadLayer, layer + 1);
        }
        auto &weights = layerWeights(layer);
        auto p = root + "layers." + std::to_string(layer) + ".";
        auto y = rt_.norm(x, true, &weights.get(p + "input_layernorm.weight"));
        auto q = weights.linear(y, p + "self_attn.q_proj"), k = weights.linear(y, p + "self_attn.k_proj"),
             v = weights.linear(y, p + "self_attn.v_proj");
        q = rt_.norm(q, true, &weights.get(p + "self_attn.q_norm.weight"), nullptr, 1e-6f, 128);
        k = rt_.norm(k, true, &weights.get(p + "self_attn.k_norm.weight"), nullptr, 1e-6f, 128);
        q = images.empty() ? rt_.textRope(q, 5000000.f) : rt_.textRope(q, ropePositions, 5000000.f);
        k = images.empty() ? rt_.textRope(k, 5000000.f) : rt_.textRope(k, ropePositions, 5000000.f);
        x = rt_.add(
            x, weights.linear(rt_.attention(q, k, v, 64, nullptr, 128, 8, true), p + "self_attn.o_proj"));
        y = rt_.norm(x, true, &weights.get(p + "post_attention_layernorm.weight"));
        Tensor output(y.rows, y.cols, y.dtype);
        for (int first = 0; first < y.rows;) {
            // Preserve the original small tail GEMM and FP32 diagnostic shapes.
            int count = std::min(y.isFloat32() ? 256 : 4096, y.rows - first);
            if (count > 256)
                count -= count % 256;
            auto input = y.rowSlice(first, count);
            auto gate = weights.linear(input, p + "mlp.gate_proj");
            auto target = output.rowSlice(first, count);
            weights.linear(rt_.siluMultiply(gate, weights.linear(input, p + "mlp.up_proj")), p + "mlp.down_proj",
                           false, &target);
            first += count;
        }
        x = rt_.add(x, output);
        if (layer < 3)
            x = applyVisual(x, layer);
        if (trace)
            trace("layer." + std::to_string(layer), x);
        weights.erasePrefix(p);
    }
    hostCacheWarm_ = true;
    // hidden_states[50] is the residual before Qwen's final norm, not last_hidden_state.
    return x;
}
} // namespace minimax_h3
} // namespace fastllm
