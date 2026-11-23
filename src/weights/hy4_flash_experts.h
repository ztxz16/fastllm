#pragma once

#include "utils.h"
#include <climits>
#include "json11.hpp"

namespace fastllm {

// The released checkpoint stacks experts along the leading dimension. Describe
// each expert as a direct file range so CPU, CUDA and disk can use normal
// two-dimensional linear weights, including their own block-scale planes.
template <class Item>
void SplitHy4FlashExperts(std::map<std::string, Item> &items, const json11::Json &config) {
    const int count = config["n_routed_experts"].int_value();
    const int hidden = config["routed_expert_hidden_size"].int_value();
    const int inter = config["moe_intermediate_size"].int_value();
    const auto &layers = config["mlp_layer_types"].array_items();
    AssertInFastLLM(count > 0 && hidden > 0 && inter > 0 && inter <= INT_MAX / 2 && !layers.empty(),
                    "Invalid HY4-Flash expert configuration.\n");

    auto split = [&](const std::string &prefix, const std::string &source, const std::string &dest, int rows,
                     int cols, bool scale) {
        auto it = items.find(prefix + source);
        AssertInFastLLM(it != items.end(), "Missing HY4-Flash tensor: " + prefix + source);
        const auto &p = it->second;
        const int bytesPerElement = p.dtype == "F8_E4M3"                    ? 1
                                    : p.dtype == "BF16" || p.dtype == "F16" ? 2
                                    : p.dtype == "F32"                      ? 4
                                                                            : 0;
        AssertInFastLLM(bytesPerElement > 0 && (!scale || bytesPerElement >= 2) &&
                            p.intShape == std::vector<int>({count, rows, cols}) &&
                            p.data_offsets.size() == 2 && p.data_offsets[1] >= p.data_offsets[0] &&
                            p.data_offsets[1] - p.data_offsets[0] == p.bytes && p.bytes % count == 0 &&
                            p.bytes / count == uint64_t(rows) * cols * bytesPerElement,
                        "Invalid HY4-Flash expert shape or file range: " + it->first);
        const uint64_t stride = p.bytes / count;
        for (int id = 0; id < count; ++id) {
            const auto name = prefix + std::to_string(id) + "." + dest;
            auto result = items.try_emplace(name);
            AssertInFastLLM(result.second, "Duplicate HY4-Flash expert: " + name);
            auto &view = result.first->second;
            const uint64_t start = p.data_offsets[0] + stride * id;
            view.tensorName = name;
            view.fileName = p.fileName;
            view.dtype = p.dtype;
            view.shape = {uint64_t(rows), uint64_t(cols)};
            view.intShape = {rows, cols};
            view.data_offsets = {start, start + stride};
            view.len = uint64_t(rows) * cols;
            view.bytes = stride;
        }
        items.erase(it);
    };
    for (size_t layer = 0; layer < layers.size(); ++layer) {
        if (layers[layer].string_value() != "sparse")
            continue;
        const std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.experts.";
        for (bool gateUp : {true, false}) {
            const std::string source = gateUp ? "gate_up_proj" : "down_proj";
            const std::string dest = gateUp ? "gateup_proj.weight" : "down_proj.weight";
            auto it = items.find(prefix + source);
            AssertInFastLLM(it != items.end(), "Missing HY4-Flash expert stack: " + prefix + source);
            const bool fp8 = it->second.dtype == "F8_E4M3";
            const int rows = gateUp ? 2 * inter : hidden, cols = gateUp ? hidden : inter;
            split(prefix, source, dest, rows, cols, false);
            if (fp8) {
                const auto &block = config["quantization_config"]["weight_block_size"].array_items();
                AssertInFastLLM(block.size() == 2 && block[0].int_value() == 128 &&
                                    block[1].int_value() == 128,
                                "HY4-Flash requires 128x128 FP8 blocks.");
                split(prefix, source + "_scale_inv", dest + "_scale_inv", 1 + (rows - 1) / 128,
                      1 + (cols - 1) / 128, true);
            }
        }
    }
}
} // namespace fastllm
