#include "hy4_flash_ops.h"
#ifdef FASTLLM_ENABLE_HY4_CUDA
#include "hy4_flash_gpu_graph.h"
#endif

namespace fastllm {
#ifdef FASTLLM_ENABLE_HY4_CUDA
std::vector<int> Hy4FlashModel::ProposeGpu(const std::vector<int> &anchors, Cache &past,
                                           const std::vector<GenerationConfig> &configs,
                                           const std::vector<hy4::DraftRequest *> &states) {
    auto &w = *source;
    auto &arenas = gpuArenas;
    std::vector<int> expanded;
    const int vocab = w.meta.at("lm_head.weight").dims[0];
    std::vector<hy4::DraftProposal> proposals(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        auto &state = *states[i];
        const int base = past[i * block_cnt].first->cpuIntDatas.size();
        auto target =
            static_cast<const float *>(state.hidden.cudaData) + size_t((base - 1) % (draftWidth + 1)) * 3072;
        size_t deviceIndex = arenas.size() - 1;
        hy4::Select(devices[deviceIndex]);
        auto block = arenas[deviceIndex]->Get("draftBlock", draftWidth * 3072);
        hy4::DraftEmbed(w, *arenas[deviceIndex], anchors[i], draftWidth, draftMask, target, block);
        for (size_t k = 0; k < draftProviders.size(); ++k) {
            size_t index = size_t(draftProviders[k]) * devices.size() / block_cnt;
            if (index != deviceIndex) {
                hy4::Select(devices[index]);
                auto next = arenas[index]->Get("draftBlock", draftWidth * 3072);
                hy4::CopyStage(next, devices[index], block, devices[deviceIndex], size_t(draftWidth) * 3072);
                block = next;
                deviceIndex = index;
            }
            auto &s = *arenas[index];
            auto &provider = state.providers[k];
            auto table = hy4::ReservePages(s.kvPools, s.device, provider.keys, provider.values,
                                           base + draftWidth, max_positions, 8192, 8192);
            auto cache = hy4::UploadAttentionCaches(s, {{nullptr, table}});
            hy4::DraftLayer(w, s, k, base, draftWidth, provider, block, &cache->draft);
        }
        if (!configs[i].IsSimpleGreedy()) {
            // Keep q on the target head's device, including with layer-parallel placement.
            if (deviceIndex != arenas.size() - 1) {
                auto next = arenas.back()->Get("draftBlock", draftWidth * 3072);
                hy4::CopyStage(next, devices.back(), block, devices[deviceIndex], size_t(draftWidth) * 3072);
                block = next;
                deviceIndex = arenas.size() - 1;
            }
            hy4::Select(devices.back());
            proposals[i] = hy4::PrepareProposal(*arenas.back(), i, states.size(), draftWidth - 1, vocab);
        }
        auto candidates = hy4::DraftReadout(w, *arenas[deviceIndex], anchors[i], draftWidth, block,
                                            configs[i], proposals[i]);
        expanded.push_back(anchors[i]);
        expanded.insert(expanded.end(), candidates.begin(), candidates.end());
    }
    float *target = nullptr;
    ForwardGpu(expanded, std::vector<int>(states.size(), draftWidth), past, configs, true, nullptr, &target);
    std::vector<int> output;
    for (size_t i = 0; i < states.size(); ++i) {
        auto &state = *states[i];
        const size_t first = i * draftWidth;
        auto tokens = hy4::VerifyDraft(*arenas.back(), target + first * vocab, draftWidth, vocab, configs[i],
                                       proposals[i], expanded.data() + first + 1);
        output.push_back(CommitVerifiedTokens(i, past, state, configs[i], tokens));
    }
    return output;
}
#endif

Hy4FlashModel::Logits Hy4FlashModel::ForwardGpu(const std::vector<int> &tokens,
                                                const std::vector<int> &lengths, Cache &past,
                                                const std::vector<GenerationConfig> &configs, bool verify,
                                                std::vector<int> *sampled, float **deviceLogits) {
#ifdef FASTLLM_ENABLE_HY4_CUDA
    auto &arenas = gpuArenas;
    if (arenas.empty())
        for (int device : devices)
            arenas.emplace_back(new hy4::Scratch(device));
    if (!gpuGraphs)
        gpuGraphs = std::make_shared<hy4::GpuGraphs>();
    auto &w = *source;
    std::vector<hy4::DraftRequest *> states;
    bool propose = sampled && draftWidth && !verify && tokens.size() == lengths.size();
    for (size_t i = 0; i < lengths.size(); ++i) {
        auto *state = draftWidth
                          ? static_cast<hy4::DraftRequest *>(past[i * block_cnt].first->kvStorageOwner.get())
                          : nullptr;
        states.push_back(state);
        const auto &cfg = configs[i];
        const int base = past[i * block_cnt].first->cpuIntDatas.size();
        propose = propose && state && base + draftWidth <= max_positions && state->CanVerify(cfg);
    }
    if (propose) {
        *sampled = ProposeGpu(tokens, past, configs, states);
        return {};
    }
    const int rows = tokens.size();
    if (GetFastllmEnv().printProfile)
        fprintf(stderr, "[HY4 gpu] requests=%zu tokens=%d verify=%d\n", lengths.size(), rows, int(verify));
    std::vector<int> owners(rows), positions(rows), lastRows;
    std::vector<std::array<int, 3>> grams(rows);
    int offset = 0;
    for (size_t request = 0; request < lengths.size(); ++request) {
        const Data &first = *past[request * block_cnt].first;
        const auto &history = first.cpuIntDatas;
        const int base = history.size();
        AssertInFastLLM(first.dims.empty() ? base == 0 : first.dims[1] <= base,
                        "HY4-Flash missing GPU request history.");
        int previous = base ? history.back() : 120002;
        int older = base > 1 ? history[base - 2] : 120002;
        for (int i = 0; i < lengths[request]; ++i, ++offset) {
            owners[offset] = request;
            positions[offset] = base + i;
            grams[offset] = {tokens[offset], previous, older};
            older = previous;
            previous = tokens[offset];
        }
        lastRows.push_back(offset - 1);
    }
    const bool graphEnabled = GetFastllmEnv().cudaGraph;
    if (graphEnabled && !gpuGraphs->peerChecked) {
        gpuGraphs->peerReady = hy4::EnableGraphPeers(devices);
        gpuGraphs->peerChecked = true;
    }
    bool capture = gpuGraphs->peerReady && graphEnabled && (verify || rows == int(lengths.size())) &&
                   gpuGraphs->warmedRows >= size_t(rows) && (rows != 1 || gpuGraphs->warmedSingle);
    for (size_t i = 0; i < lengths.size() && capture; ++i)
        capture = positions[lastRows[i]] + 1 - lengths[i] >= 4;
    std::vector<const hy4::AttentionCache *> layerCaches(block_cnt);
    int firstLayer = 0;
    for (size_t stage = 0; stage < arenas.size(); ++stage) {
        int endLayer = firstLayer;
        while (endLayer < block_cnt && size_t(endLayer) * arenas.size() / block_cnt == stage)
            ++endLayer;
        if (firstLayer < endLayer) {
            auto table = hy4::PrepareAttentionCaches(w, *arenas[stage], firstLayer, endLayer, block_cnt,
                                                     lengths, past, draftProviders, states, draftWidth);
            for (int layer = firstLayer; layer < endLayer; ++layer)
                layerCaches[layer] = table + (layer - firstLayer) * lengths.size();
        }
        firstLayer = endLayer;
    }
    const int vocab = w.meta.at("lm_head.weight").dims[0];
    const size_t outputs = verify ? rows : lengths.size();
    hy4::Select(devices.back());
    auto projected = arenas.back()->Get("batchedLogits", outputs * vocab);
    hy4::Select(devices.front());
    std::vector<uintptr_t> key;
    std::vector<const hy4::StepInput *> stepInputs;
    if (capture) {
        std::vector<hy4::StepInput> params;
        for (int row = 0; row < rows; ++row)
            params.push_back({tokens[row], grams[row][1], grams[row][2], positions[row]});
        for (auto &arena : arenas)
            stepInputs.push_back(hy4::UploadStep(*arena, params));
        key = {uintptr_t(verify), uintptr_t(rows), uintptr_t(lengths.size()), uintptr_t(projected)};
        for (int row = 0; row < rows; ++row) {
            const bool skip = tokens[row] == w.config["engram_pad_id"].int_value() || tokens[row] == 120120 ||
                              tokens[row] == 120124;
            key.push_back(int(positions[row] >= 2048) * 2 + int(skip));
        }
        for (size_t i = 0; i < lengths.size(); ++i) {
            key.push_back(lengths[i]);
            key.push_back(states[i] ? draftWidth : 0);
            if (states[i]) {
                key.push_back(uintptr_t(states[i]->hidden.cudaData));
                for (const auto &mem : states[i]->providers) {
                    key.push_back(uintptr_t(mem.recalled.cudaData));
                }
            }
        }
        for (auto &arena : arenas)
            key.push_back(arena->generation);
    }
    hy4::Select(devices.back());
    bool replay = capture && hy4::ReplayGpu(*gpuGraphs, key, verify, rows);
    if (!replay) {
        hy4::Select(devices.front());
        hy4::GpuGraphCapture graph(capture ? gpuGraphs.get() : nullptr);
        graph.Start(devices.front());
        auto streams = arenas.front()->Get("streams", size_t(rows) * 4 * 3072);
        for (int row = 0; row < rows; ++row) {
            arenas.front()->step = capture ? stepInputs.front() + row : nullptr;
            hy4::Embed(w, *arenas.front(), tokens[row], streams + size_t(row) * 4 * 3072);
        }
        size_t stage = 0;
        for (int layer = 0; layer < block_cnt; ++layer) {
            const size_t owner = size_t(layer) * devices.size() / block_cnt;
            auto &s = *arenas[owner];
            if (stage != owner) {
                graph.Close();
                hy4::Select(s.device);
                auto next = s.Get("streams", size_t(rows) * 4 * 3072);
                graph.Start(s.device);
                if (capture)
                    hy4::CopyStageCaptured(next, streams, size_t(rows) * 4 * 3072);
                else
                    hy4::CopyStage(next, s.device, streams, devices[stage], size_t(rows) * 4 * 3072);
                stage = owner;
                streams = next;
            }
            const auto p = "model.layers." + std::to_string(layer) + ".";
            auto inputs = s.Get("inputs", size_t(rows) * 3072);
            auto branch = s.Get("branch", size_t(rows) * 3072), posts = s.Get("posts", rows * 4);
            auto latents = s.Get("latents", size_t(rows) * 1536),
                 routed = s.Get("routed", size_t(rows) * 1536);
            auto scores = s.Get("scores", rows * 16);
            auto ids = reinterpret_cast<int *>(s.Get("ids", rows * 16));
            std::vector<hy4::AttentionRow> attentionRows;
            attentionRows.reserve(rows);
            auto attnOutput = s.Get("attentionOutput", size_t(rows) * 3072);
            for (int row = 0; row < rows; ++row) {
                s.step = capture ? stepInputs[owner] + row : nullptr;
                float *h = streams + size_t(row) * 4 * 3072;
                if (w.meta.count(p + "engram_layer.lookup_table"))
                    hy4::Engram(w, s, layer, grams[row], h);
                auto provider = std::find(draftProviders.begin(), draftProviders.end(), layer);
                auto *state = states[owners[row]];
                auto index = provider - draftProviders.begin();
                auto *memory = state && provider != draftProviders.end() ? &state->providers[index] : nullptr;
                attentionRows.push_back({positions[row], state ? draftWidth : 0, memory, int(index), s.step,
                                         layerCaches[layer] + owners[row]});
            }
            hy4::HyperPreNorm(w, s, p + "hc_attn_layer.hc_pre.", p + "input_layernorm.weight",
                              streams, inputs, posts, rows);
            hy4::PagedAttentionBatch(w, s, layer, attentionRows, inputs, attnOutput);
            hy4::HyperPostNorm(w, s, p + "self_attn.post_proj_norm.weight", streams, attnOutput, posts, rows);
            hy4::HyperPreNorm(w, s, p + "hc_mlp_layer.hc_pre.", p + "post_attention_layernorm.weight",
                              streams, inputs, posts, rows);
            if (!layer) {
                hy4::DenseMlpBatch(w, s, p + "mlp.", inputs, branch, rows);
                hy4::HyperPostNorm(w, s, p + "mlp.post_fc2_norm.weight", streams, branch, posts, rows, true);
                continue;
            }
            hy4::RouteBatch(w, s, layer, inputs, ids, scores, latents, rows);
            hy4::ResidentExperts(w, s, layer, latents, ids, scores, routed, rows);
            hy4::CombineExpertsBatch(w, s, layer, inputs, routed, branch, rows);
            hy4::HyperPostNorm(w, s, p + "mlp.combine_output_norm.weight", streams, branch, posts, rows);
        }
        size_t output = 0;
        offset = 0;
        for (size_t i = 0; i < lengths.size(); ++i) {
            for (int row = verify ? offset : lastRows[i]; row <= lastRows[i]; ++row, ++output) {
                arenas.back()->step = capture ? stepInputs.back() + row : nullptr;
                hy4::ProjectHead(w, *arenas.back(), streams + size_t(row) * 4 * 3072,
                                 projected + output * vocab, states[i] ? &states[i]->hidden : nullptr,
                                 positions[row], draftWidth);
            }
            offset += lengths[i];
        }
        graph.Finish(key, verify, rows);
        for (auto &arena : arenas)
            arena->step = nullptr;
    }
    Logits result;
    if (deviceLogits) {
        *deviceLogits = projected;
    } else {
        result.resize(outputs, std::vector<float>(vocab));
        for (size_t i = 0; i < outputs; ++i)
            hy4::CopyToHost(result[i].data(), projected + i * vocab, vocab * sizeof(float));
    }
    if (!capture && rows == 1)
        gpuGraphs->warmedSingle = true;
    if (!capture)
        gpuGraphs->warmedRows = std::max(gpuGraphs->warmedRows, size_t(rows));
    offset = 0;
    for (size_t i = 0; i < lengths.size(); ++i) {
        auto &history = past[i * block_cnt].first->cpuIntDatas;
        history.insert(history.end(), tokens.begin() + offset, tokens.begin() + lastRows[i] + 1);
        offset += lengths[i];
    }
    return result;
#else
    throw std::runtime_error("HY4-Flash requires CUDA.");
#endif
}
} // namespace fastllm
