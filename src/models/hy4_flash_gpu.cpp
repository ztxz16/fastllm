#include "hy4_flash_ops.h"

namespace fastllm {
Hy4FlashModel::Logits Hy4FlashModel::ForwardGpu(const std::vector<int> &tokens,
                                                const std::vector<int> &lengths, Cache &past,
                                                const std::vector<GenerationConfig> &configs, bool verify) {
#ifdef FASTLLM_ENABLE_HY4_CUDA
    auto &arenas = gpuArenas;
    if (arenas.empty())
        for (int device : devices)
            arenas.emplace_back(new hy4::Scratch(device));
    if (!gpuGraphs)
        gpuGraphs = std::make_shared<hy4::GpuGraphs>();
    auto &w = *source;
    std::vector<hy4::DraftRequest *> states;
    bool propose = draftWidth && !verify && tokens.size() == lengths.size();
    for (size_t i = 0; i < lengths.size(); ++i) {
        auto *state = static_cast<hy4::DraftRequest *>(past[i * block_cnt].first->kvStorageOwner.get());
        states.push_back(state);
        const auto &cfg = configs[i];
        const int base = past[i * block_cnt].first->cpuIntDatas.size();
        propose = propose && state && !state->hidden.dims.empty() && state->pending.empty() &&
                  base + draftWidth <= max_positions && cfg.IsSimpleGreedy() && !cfg.do_sample &&
                  !cfg.output_logits && !cfg.tool_call_grammar && !cfg.tool_call_name_constraint_enabled &&
                  !cfg.tool_call_parameter_name_constraint_enabled && cfg.prefill_chunk_remaining < 0;
    }
    if (propose) {
        std::vector<int> expanded;
        std::vector<int> bases;
        for (size_t i = 0; i < states.size(); ++i) {
            auto &state = *states[i];
            const int base = past[i * block_cnt].first->cpuIntDatas.size();
            bases.push_back(base);
            auto target = static_cast<const float *>(state.hidden.cudaData) +
                          size_t((base - 1) % (draftWidth + 1)) * 3072;
            size_t deviceIndex = arenas.size() - 1;
            hy4::Select(devices[deviceIndex]);
            auto block = arenas[deviceIndex]->Get("draftBlock", draftWidth * 3072);
            hy4::DraftEmbed(w, *arenas[deviceIndex], tokens[i], draftWidth, draftMask, target, block);
            for (size_t k = 0; k < draftProviders.size(); ++k) {
                size_t index = size_t(draftProviders[k]) * devices.size() / block_cnt;
                if (index != deviceIndex) {
                    hy4::Select(devices[index]);
                    auto next = arenas[index]->Get("draftBlock", draftWidth * 3072);
                    hy4::CopyStage(next, devices[index], block, devices[deviceIndex],
                                   size_t(draftWidth) * 3072);
                    block = next;
                    deviceIndex = index;
                }
                hy4::DraftLayer(w, *arenas[index], k, base, draftWidth, state.providers[k], block);
            }
            auto candidates = hy4::DraftReadout(w, *arenas[deviceIndex], tokens[i], draftWidth, block);
            expanded.push_back(tokens[i]);
            expanded.insert(expanded.end(), candidates.begin(), candidates.end());
        }
        auto verified =
            ForwardGpu(expanded, std::vector<int>(lengths.size(), draftWidth), past, configs, true);
        Logits output;
        for (size_t i = 0; i < states.size(); ++i) {
            auto &state = *states[i];
            const size_t first = i * draftWidth;
            int accepted = 0;
            for (int j = 0; j < draftWidth; ++j) {
                const auto &row = verified[first + j];
                int next = std::max_element(row.begin(), row.end()) - row.begin();
                if (j)
                    state.pending.push_back(next);
                else
                    state.expected = next;
                if (next == eos_token_id || eos_token_ids.count(next) ||
                    configs[i].stop_token_ids.count(next) || j + 1 == draftWidth ||
                    next != expanded[first + j + 1])
                    break;
                ++accepted;
            }
            const int committed = bases[i] + 1 + accepted;
            past[i * block_cnt].first->cpuIntDatas.resize(committed);
            for (int layer = 0; layer < block_cnt; ++layer) {
                auto &kv = past[i * block_cnt + layer];
                for (auto *data : {kv.first, kv.second}) {
                    auto dims = data->dims;
                    dims[1] = std::min(committed, dims[1]);
                    data->Resize(dims);
                }
            }
            if (std::getenv("FASTLLM_DFLY2_TRACE"))
                fprintf(stderr, "[HY4 DFly2] request=%zu position=%d candidates=%d accepted=%d\n", i,
                        bases[i], draftWidth - 1, accepted);
            output.emplace_back(std::move(verified[first]));
        }
        return output;
    }
    const int rows = tokens.size();
    if (std::getenv("FASTLLM_HY4_BATCH_TRACE"))
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
    const char *graphFlag = std::getenv("FASTLLM_HY4_GPU_GRAPH");
    const bool graphEnabled = graphFlag ? std::string(graphFlag) == "1" : GetFastllmEnv().cudaGraph;
    if (graphEnabled && !gpuGraphs->peerChecked) {
        gpuGraphs->peerReady = hy4::EnableGraphPeers(devices);
        gpuGraphs->peerChecked = true;
    }
    bool capture = gpuGraphs->peerReady && graphEnabled &&
                   (verify || rows == int(lengths.size())) && gpuGraphs->warmedRows >= size_t(rows) &&
                   (rows != 1 || gpuGraphs->warmedSingle);
    for (size_t i = 0; i < lengths.size() && capture; ++i) {
        int end = positions[lastRows[i]] + 1;
        const auto firstPos = positions[lastRows[i]] + 1 - lengths[i];
        if (firstPos < 4 || (firstPos <= 2048 && end > 2048)) {
            capture = false;
            break;
        }
        if (states[i] && end > 2048)
            for (const auto &memory : states[i]->providers)
                if (!memory.recalled.cudaData)
                    capture = false;
        for (int layer = 0; layer < block_cnt && capture; ++layer) {
            auto &kc = *past[i * block_cnt + layer].first;
            bool sliding = w.config["layer_types"][layer].string_value() == "sliding_attention";
            int wanted = sliding ? std::min(end, 512 + (states[i] ? draftWidth : 0)) : end;
            capture = !kc.expansionDims.empty() && kc.expansionDims[1] >= wanted;
        }
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
            key.push_back((positions[row] / hy4::KVPageTokens) * 2 + int(skip));
        }
        for (size_t i = 0; i < lengths.size(); ++i) {
            key.push_back(lengths[i]);
            for (int layer = 0; layer < block_cnt; ++layer) {
                auto &kv = past[i * block_cnt + layer];
                key.push_back(uintptr_t(kv.first->cudaData));
                key.push_back(uintptr_t(kv.second->cudaData));
            }
            if (states[i]) {
                key.push_back(uintptr_t(states[i]->hidden.cudaData));
                for (const auto &mem : states[i]->providers) {
                    key.push_back(uintptr_t(mem.keys.cudaData));
                    key.push_back(uintptr_t(mem.values.cudaData));
                    key.push_back(uintptr_t(mem.recalled.cudaData));
                }
            }
        }
        for (auto &arena : arenas)
            key.push_back(arena->generation);
    }
    hy4::Select(devices.back());
    bool replay = capture && hy4::ReplayGpu(*gpuGraphs, key, verify, rows);
    if (replay) {
        for (size_t i = 0; i < lengths.size(); ++i) {
            const int end = positions[lastRows[i]] + 1;
            for (int layer = 0; layer < block_cnt; ++layer) {
                auto &kv = past[i * block_cnt + layer];
                hy4::AdvanceKV(*kv.first, end);
                hy4::AdvanceKV(*kv.second, end);
            }
            if (states[i])
                for (auto &mem : states[i]->providers) {
                    hy4::AdvanceKV(mem.keys, end);
                    hy4::AdvanceKV(mem.values, end);
                }
        }
    } else {
        hy4::Select(devices.front());
        hy4::GraphCapture graph(capture ? gpuGraphs.get() : nullptr);
        graph.key = key;
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
                    hy4::CopyStageCaptured(next, s.device, streams, devices[stage], size_t(rows) * 4 * 3072);
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
                float *x = inputs + size_t(row) * 3072, *post = posts + row * 4;
                auto &cache = past[owners[row] * block_cnt + layer];
                if (w.meta.count(p + "engram_layer.lookup_table"))
                    hy4::Engram(w, s, layer, grams[row], h);
                hy4::HyperPre(w, s, p + "hc_attn_layer.hc_pre.", h, x, post);
                hy4::Norm(w, s, p + "input_layernorm.weight", x, x, 3072);
                auto provider = std::find(draftProviders.begin(), draftProviders.end(), layer);
                auto *state = states[owners[row]];
                auto index = provider - draftProviders.begin();
                auto *memory = state && provider != draftProviders.end() ? &state->providers[index] : nullptr;
                attentionRows.push_back({positions[row], cache.first, cache.second, state ? draftWidth : 0,
                                         memory, int(index), s.step});
            }
            hy4::AttentionBatch(w, s, layer, attentionRows, inputs, attnOutput);
            for (int row = 0; row < rows; ++row) {
                auto h = streams + size_t(row) * 4 * 3072;
                auto x = inputs + size_t(row) * 3072;
                auto post = posts + row * 4;
                hy4::HyperPost(h, attnOutput + size_t(row) * 3072, post, 3072);
                hy4::HyperPre(w, s, p + "hc_mlp_layer.hc_pre.", h, x, post);
                hy4::Norm(w, s, p + "post_attention_layernorm.weight", x, x, 3072);
            }
            if (!layer) {
                hy4::DenseMlpBatch(w, s, p + "mlp.", inputs, branch, rows);
                for (int row = 0; row < rows; ++row)
                    hy4::HyperPost(streams + size_t(row) * 4 * 3072, branch + size_t(row) * 3072,
                                   posts + row * 4, 3072);
                continue;
            }
            hy4::RouteBatch(w, s, layer, inputs, ids, scores, latents, rows);
            hy4::ResidentExperts(w, s, layer, latents, ids, scores, routed, rows);
            hy4::CombineExpertsBatch(w, s, layer, inputs, routed, branch, rows);
            for (int row = 0; row < rows; ++row)
                hy4::HyperPost(streams + size_t(row) * 4 * 3072, branch + size_t(row) * 3072, posts + row * 4,
                               3072);
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
        graph.Finish(verify, rows);
        for (auto &arena : arenas)
            arena->step = nullptr;
    }
    Logits result(outputs, std::vector<float>(vocab));
    for (size_t i = 0; i < outputs; ++i)
        hy4::CopyToHost(result[i].data(), projected + i * vocab, vocab * sizeof(float));
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
