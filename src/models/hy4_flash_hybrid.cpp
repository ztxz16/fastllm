#include "hy4_flash_ops.h"
#include "utils/hybrid_moe.h"
#include "devices/disk/diskdevice.h"
#include <cstring>
#ifdef FASTLLM_ENABLE_HY4_CUDA
#include "hy4_flash_hybrid_graph.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#endif

namespace fastllm {
void Hy4FlashModel::OnModelWeightsLoaded() {
    if (!hybrid)
        return;
    hostExperts.resize(block_cnt);
    bool allDisk = true, allCpu = true, allNuma = true;
    for (int layer = 1; layer < block_cnt; layer++) {
        const auto device = SelectMoeDeviceForLayer(layer);
        allDisk &= device == "disk";
        allCpu &= device == "cpu";
        allNuma &= device == "numa" || device.rfind("numa:", 0) == 0;
        auto &table = hostExperts[layer];
        table.resize(2 + 2 * 512, nullptr);
        for (int id = 0; id < 512; id++) {
            auto p = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(id) + ".";
            table[2 + 2 * id] = &weight.weight.at(p + "gateup_proj.weight");
            table[3 + 2 * id] = &weight.weight.at(p + "down_proj.weight");
        }
    }
    if (allDisk)
        diskGraphReady =
            std::all_of(hostExperts.begin() + 1, hostExperts.end(), DiskMoeV41Graph::RegisterStandardFp8);
    // A complete graph requires one expert backend across the routed layers.
    // Heterogeneous placements dispatch each layer through its assigned device.
    if (MoeCudaCacheRequested() && (allCpu || allNuma)) {
        HybridMoeCachePlan plan(HybridMoeMath::Hy4Flash, 10.f);
        for (int layer = 1; layer < block_cnt; layer++)
            plan.Add(hostExperts[layer], SelectMoeDeviceForLayer(layer), devices.front());
        hostCacheReady = plan.Prepare([this] { WarmupNumaMoeWeights(true); }, true);
        AssertInFastLLM(hostCacheReady, "Failed to register HY4-Flash expert weights with the CUDA cache.");
    }
}
#ifdef FASTLLM_ENABLE_HY4_CUDA
Hy4FlashModel::Logits Hy4FlashModel::ProposeHybrid(const std::vector<int> &anchors, Cache &past,
                                                   const std::vector<GenerationConfig> &configs,
                                                   const std::vector<hy4::DraftRequest *> &states) {
    auto &w = *source;
    auto &s = *hybridArena;
    std::vector<int> expanded;
    std::vector<int> bases;
    for (size_t i = 0; i < states.size(); ++i) {
        auto &state = *states[i];
        const int base = past[i * block_cnt].first->cpuIntDatas.size();
        bases.push_back(base);
        auto target =
            static_cast<const float *>(state.hidden.cudaData) + size_t((base - 1) % (draftWidth + 1)) * 3072;
        auto block = s.Get("draftBlock", draftWidth * 3072);
        hy4::DraftEmbed(w, s, anchors[i], draftWidth, draftMask, target, block);
        std::vector<hy4::AttentionCache> caches;
        const int limit = w.config["max_position_embeddings"].int_value();
        for (auto &provider : state.providers) {
            auto pages = hy4::ReservePages(s.kvPools, s.device, provider.keys, provider.values,
                                           base + draftWidth, limit, 8192, 8192);
            caches.push_back({nullptr, pages});
        }
        auto draftCaches = hy4::UploadAttentionCaches(s, caches);
        for (size_t k = 0; k < draftProviders.size(); ++k)
            hy4::DraftLayer(w, s, k, base, draftWidth, state.providers[k], block, &draftCaches[k].draft);
        auto candidates = hy4::DraftReadout(w, s, anchors[i], draftWidth, block);
        expanded.push_back(anchors[i]);
        expanded.insert(expanded.end(), candidates.begin(), candidates.end());
    }
    auto verified = ForwardHybrid(expanded, std::vector<int>(states.size(), draftWidth), past, configs, true);
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
            if (next == eos_token_id || eos_token_ids.count(next) || configs[i].stop_token_ids.count(next) ||
                j + 1 == draftWidth || next != expanded[first + j + 1])
                break;
            ++accepted;
        }
        const int committed = bases[i] + 1 + accepted;
        past[i * block_cnt].first->cpuIntDatas.resize(committed);
        for (int layer = 0; layer < block_cnt; ++layer) {
            auto &kv = past[i * block_cnt + layer];
            hy4::TruncateKV(*kv.first, *kv.second, committed);
        }
        for (auto &provider : state.providers)
            hy4::TruncateKV(provider.keys, provider.values, committed);
        if (GetFastllmEnv().printProfile)
            fprintf(stderr, "[HY4 DFly2] request=%zu position=%d candidates=%d accepted=%d\n", i, bases[i],
                    draftWidth - 1, accepted);
        output.emplace_back(std::move(verified[first]));
    }
    return output;
}
#endif

Hy4FlashModel::Logits Hy4FlashModel::ForwardHybrid(const std::vector<int> &tokens,
                                                   const std::vector<int> &lengths, Cache &past,
                                                   const std::vector<GenerationConfig> &configs,
                                                   bool verify) {
#ifdef FASTLLM_ENABLE_HY4_CUDA
    const int device = devices.front();
    hy4::Select(device);
    if (!hybridArena)
        hybridArena = std::make_shared<hy4::Scratch>(device);
    if (!hybridGraphs)
        hybridGraphs = std::make_shared<hy4::HybridGraphs>();
    auto &s = *hybridArena;
    std::vector<Data *> biases(2 + 2 * 512, nullptr);
    auto &w = *source;
    std::vector<hy4::DraftRequest *> states;
    bool propose = draftWidth && !verify && tokens.size() == lengths.size();
    for (size_t i = 0; i < lengths.size(); ++i) {
        auto *state = draftWidth
                          ? static_cast<hy4::DraftRequest *>(past[i * block_cnt].first->kvStorageOwner.get())
                          : nullptr;
        states.push_back(state);
        const auto &cfg = configs[i];
        const int base = past[i * block_cnt].first->cpuIntDatas.size();
        propose = propose && state && !state->hidden.dims.empty() && state->pending.empty() &&
                  base + draftWidth <= max_positions && cfg.IsSimpleGreedy() && !cfg.do_sample &&
                  !cfg.output_logits && !cfg.tool_call_grammar && !cfg.tool_call_name_constraint_enabled &&
                  !cfg.tool_call_parameter_name_constraint_enabled && cfg.prefill_chunk_remaining < 0;
    }
    if (propose)
        return ProposeHybrid(tokens, past, configs, states);
    const int rows = tokens.size();
    if (GetFastllmEnv().printProfile)
        fprintf(stderr, "[HY4 hybrid] requests=%zu tokens=%d verify=%d\n", lengths.size(), rows, int(verify));
    std::vector<int> owners(rows), positions(rows), lastRows;
    std::vector<std::array<int, 3>> grams(rows);
    int offset = 0;
    for (size_t request = 0; request < lengths.size(); ++request) {
        const Data &first = *past[request * block_cnt].first;
        const auto &history = first.cpuIntDatas;
        const int base = history.size();
        AssertInFastLLM(first.dims.empty() ? base == 0 : first.dims[1] <= base,
                        "HY4-Flash missing hybrid request history.");
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
    const bool decode = rows == int(lengths.size()) || verify;
    FastllmCudaMoeRouteScope phase(verify   ? FastllmMoeRoutePhase::Verify
                                   : decode ? FastllmMoeRoutePhase::Decode
                                            : FastllmMoeRoutePhase::SmallPrefill);
    HybridMoeDecodeStep step;
    if (hostCacheReady && decode)
        step.Begin(hostExperts[1].data(), hostExperts[1].size(), 16);
    const bool graphEnabled = GetFastllmEnv().cudaGraph;
    bool capture = graphEnabled && decode && (bool(step) || (diskGraphReady && rows <= 32)) &&
                   hybridGraphs->warmedRows >= size_t(rows) && (rows != 1 || hybridGraphs->warmedSingle) &&
                   (diskGraphReady ||
                    FastllmCudaNativeMoeGraphOwner(hostExperts[1].data(), hostExperts[1].size(), rows));
    for (size_t i = 0; i < lengths.size() && capture; ++i)
        capture = positions[lastRows[i]] + 1 - lengths[i] >= 4;
    auto attentionCaches = hy4::PrepareAttentionCaches(w, s, 0, block_cnt, block_cnt, lengths, past,
                                                       draftProviders, states, draftWidth);
    if (capture)
        for (int layer = 0; layer < block_cnt; ++layer)
            if (w.meta.count("model.layers." + std::to_string(layer) + ".engram_layer.lookup_table"))
                for (int row = 0; row < rows; ++row)
                    hy4::PrepareEngramInput(w, s, layer, row, grams[row]);
    const int vocab = w.meta.at("lm_head.weight").dims[0];
    const size_t outputs = verify ? rows : lengths.size();
    auto projected = s.Get("batchedLogits", outputs * vocab);
    // Reserve the widest request before recording the scratch generation in the graph key.
    if (capture && verify)
        s.Get("headX", size_t(*std::max_element(lengths.begin(), lengths.end())) * 3072);

    std::vector<uintptr_t> key;
    const hy4::StepInput *stepInputs = nullptr;
    if (capture) {
        std::vector<hy4::StepInput> params;
        for (int row = 0; row < rows; ++row)
            params.push_back({tokens[row], grams[row][1], grams[row][2], positions[row]});
        stepInputs = hy4::UploadStep(s, params);
        key = {uintptr_t(verify), uintptr_t(rows), uintptr_t(lengths.size()), uintptr_t(projected)};
        for (int row = 0; row < rows; ++row) {
            const bool skip = tokens[row] == w.config["engram_pad_id"].int_value() || tokens[row] == 120120 ||
                              tokens[row] == 120124;
            // Sparse attention selects recalled groups starting at position 2048.
            key.push_back(2 * int(positions[row] >= 2048) + int(skip));
        }
        for (size_t i = 0; i < lengths.size(); ++i) {
            key.push_back(lengths[i]);
            key.push_back(states[i] ? draftWidth : 0);
            if (states[i]) {
                key.push_back(uintptr_t(states[i]->hidden.cudaData));
                for (const auto &mem : states[i]->providers)
                    key.push_back(uintptr_t(mem.recalled.cudaData));
            }
        }
        key.push_back(s.generation);
    }

    bool replay = capture && hy4::ReplayHybrid(*hybridGraphs, key, verify, rows);
    if (!replay) {
        hy4::HybridGraphCapture graph(capture ? hybridGraphs.get() : nullptr, diskGraphReady);
        graph.Start(device);
        auto streams = s.Get("streams", size_t(rows) * 4 * 3072);
        for (int row = 0; row < rows; ++row) {
            s.step = capture ? stepInputs + row : nullptr;
            hy4::Embed(w, s, tokens[row], streams + size_t(row) * 4 * 3072);
        }
        for (int layer = 0; layer < block_cnt; ++layer) {
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
                s.step = capture ? stepInputs + row : nullptr;
                float *h = streams + size_t(row) * 4 * 3072;
                float *x = inputs + size_t(row) * 3072, *post = posts + row * 4;
                if (w.meta.count(p + "engram_layer.lookup_table")) {
                    if (capture)
                        hy4::RunPreparedEngram(w, s, layer, row, grams[row], h);
                    else {
                        auto table = static_cast<const uint16_t *>(
                            w.Host(p + "engram_layer.multi_head_embedding.embedding.weight"));
                        auto indices = hy4::EngramRows(w, layer, grams[row]);
                        std::array<uint16_t, 16 * 96> staging;
                        for (int j = 0; j < 16; ++j)
                            std::memcpy(staging.data() + j * 96, table + indices[j] * 96,
                                        96 * sizeof(uint16_t));
                        hy4::Engram(w, s, layer, grams[row], h, staging.data());
                    }
                }
                hy4::HyperPre(w, s, p + "hc_attn_layer.hc_pre.", h, x, post);
                auto provider = std::find(draftProviders.begin(), draftProviders.end(), layer);
                auto *state = states[owners[row]];
                auto index = provider - draftProviders.begin();
                auto *memory = state && provider != draftProviders.end() ? &state->providers[index] : nullptr;
                attentionRows.push_back({positions[row], state ? draftWidth : 0, memory, int(index), s.step,
                                         attentionCaches + layer * lengths.size() + owners[row]});
            }
            hy4::NormBatch(w, s, p + "input_layernorm.weight", inputs, rows, 3072);
            hy4::PagedAttentionBatch(w, s, layer, attentionRows, inputs, attnOutput);
            for (int row = 0; row < rows; ++row) {
                auto h = streams + size_t(row) * 4 * 3072;
                auto x = inputs + size_t(row) * 3072;
                auto post = posts + row * 4;
                hy4::HyperPost(h, attnOutput + size_t(row) * 3072, post, 3072);
                hy4::HyperPre(w, s, p + "hc_mlp_layer.hc_pre.", h, x, post);
            }
            hy4::NormBatch(w, s, p + "post_attention_layernorm.weight", inputs, rows, 3072);
            if (!layer) {
                hy4::DenseMlpBatch(w, s, p + "mlp.", inputs, branch, rows);
                for (int row = 0; row < rows; ++row)
                    hy4::HyperPost(streams + size_t(row) * 4 * 3072, branch + size_t(row) * 3072,
                                   posts + row * 4, 3072);
                continue;
            }
            hy4::RouteBatch(w, s, layer, inputs, ids, scores, latents, rows);
            bool dispatched = false;
            if ((hostCacheReady && decode) || capture) {
                Data input(FLOAT32, {rows, 1536}, CUDA, latents);
                Data index(INT32, {rows, 16}, CUDA, ids), score(FLOAT32, {rows, 16}, CUDA, scores);
                Data output;
                input.dataDeviceIds = index.dataDeviceIds = score.dataDeviceIds = {device};
                if (capture) {
                    const auto &out = graph.AddExperts(input, index, score, hostExperts[layer], layer);
                    hy4::CopyStageCaptured(routed, static_cast<const float *>(out.cudaData),
                                           size_t(rows) * 1536);
                    dispatched = true;
                } else {
                    dispatched = RunHybridMoe(input, index, score, output, hostExperts[layer].data(),
                                              hostExperts[layer].size(), layer);
                    if (dispatched)
                        hy4::CopyStage(routed, device, static_cast<const float *>(output.cudaData), device,
                                       size_t(rows) * 1536);
                }
            }
            if (!dispatched) {
                std::vector<float> hostLatents(size_t(rows) * 1536), hostScores(rows * 16);
                hy4::CopyToHost(hostLatents.data(), latents, hostLatents.size() * sizeof(float));
                hy4::CopyToHost(hostScores.data(), scores, hostScores.size() * sizeof(float));
                Data input(BFLOAT16, {rows, 1536}, hostLatents), index(INT32PARAM, {rows, 16});
                Data score(FLOAT32, {rows, 16}, hostScores);
                index.Allocate(false);
                hy4::CopyToHost(index.cpuData, ids, size_t(rows) * 16 * sizeof(int));
                Data out, a, b, c, curIn, curOut;
                ApplyMoeDeviceMapForLayer(layer);
                MergeMOE(input, index, score, hostExperts[layer], biases, a, b, c, curIn, curOut, 1.f, out,
                         layer, MoeGateSwiglu, false, 10.f);
                out.ToDevice(CPU);
                ToDataTypeForceCPU(out, FLOAT32);
                hy4::Select(device);
                hy4::CopyToGPU(routed, out.cpuData, size_t(rows) * 1536 * sizeof(float));
            }
            hy4::CombineExpertsBatch(w, s, layer, inputs, routed, branch, rows);
            for (int row = 0; row < rows; ++row)
                hy4::HyperPost(streams + size_t(row) * 4 * 3072, branch + size_t(row) * 3072, posts + row * 4,
                               3072);
        }
        for (size_t i = 0; i < lengths.size(); ++i) {
            const int count = verify ? lengths[i] : 1;
            const int first = lastRows[i] + 1 - count;
            const size_t output = verify ? size_t(first) : i;
            s.step = capture ? stepInputs + first : nullptr;
            hy4::ProjectHead(w, s, streams + size_t(first) * 4 * 3072, projected + output * vocab,
                             states[i] ? &states[i]->hidden : nullptr, positions[first], draftWidth, count);
        }
        graph.Finish(key, verify, rows);
        s.step = nullptr;
    }
    Logits result(outputs, std::vector<float>(vocab));
    for (size_t i = 0; i < outputs; ++i)
        hy4::CopyToHost(result[i].data(), projected + i * vocab, vocab * sizeof(float));
    if (!capture && rows == 1)
        hybridGraphs->warmedSingle = true;
    if (!capture)
        hybridGraphs->warmedRows = std::max(hybridGraphs->warmedRows, size_t(rows));
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
