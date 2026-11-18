#include "hy4_flash_ops.h"
#include "utils/hybrid_moe.h"
#include "devices/disk/diskdevice.h"
#include <cstring>
#ifdef FASTLLM_ENABLE_HY4_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#endif

namespace fastllm {
void Hy4FlashModel::OnModelWeightsLoaded() {
    if (!hybrid)
        return;
    hostExperts.resize(block_cnt);
    for (int layer = 1; layer < block_cnt; layer++) {
        auto &table = hostExperts[layer];
        table.resize(2 + 2 * 512, nullptr);
        for (int id = 0; id < 512; id++) {
            auto p = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(id) + ".";
            table[2 + 2 * id] = &weight.weight.at(p + "gateup_proj.weight");
            table[3 + 2 * id] = &weight.weight.at(p + "down_proj.weight");
        }
    }
    if (SelectMoeDeviceForLayer(1) == "disk")
        diskGraphReady = std::all_of(hostExperts.begin() + 1, hostExperts.end(), DiskMoeV41Graph::Supports);
    if (MoeCudaCacheRequested() && SelectMoeDeviceForLayer(1) != "disk") {
        HybridMoeCachePlan plan(HybridMoeMath::Standard, 10.f);
        for (int layer = 1; layer < block_cnt; layer++)
            plan.Add(hostExperts[layer], SelectMoeDeviceForLayer(layer), devices.front());
        hostCacheReady = plan.Prepare([this] { WarmupNumaMoeWeights(true); }, true);
        AssertInFastLLM(hostCacheReady, "Failed to register HY4-Flash expert weights with the CUDA cache.");
    }
}
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
        hybridGraphs = std::make_shared<hy4::GpuGraphs>();
    auto &s = *hybridArena;
    std::vector<Data *> biases(2 + 2 * 512, nullptr);
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
            auto block = s.Get("draftBlock", draftWidth * 3072);
            hy4::DraftEmbed(w, s, tokens[i], draftWidth, draftMask, target, block);
            for (size_t k = 0; k < draftProviders.size(); ++k)
                hy4::DraftLayer(w, s, k, base, draftWidth, state.providers[k], block);
            auto candidates = hy4::DraftReadout(w, s, tokens[i], draftWidth, block);
            expanded.push_back(tokens[i]);
            expanded.insert(expanded.end(), candidates.begin(), candidates.end());
        }
        auto verified =
            ForwardHybrid(expanded, std::vector<int>(lengths.size(), draftWidth), past, configs, true);
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
    const char *flag = std::getenv("FASTLLM_HY4_HYBRID_GRAPH");
    const bool graphEnabled = flag ? std::string(flag) == "1" : GetFastllmEnv().cudaGraph;
    bool capture = graphEnabled && decode &&
                   (bool(step) || (diskGraphReady && rows <= 32)) &&
                   hybridGraphs->warmedRows >= size_t(rows) && (rows != 1 || hybridGraphs->warmedSingle) &&
                   (diskGraphReady ||
                    FastllmCudaNativeMoeGraphOwner(hostExperts[1].data(), hostExperts[1].size(), rows));
    for (size_t i = 0; i < lengths.size() && capture; ++i) {
        int end = positions[lastRows[i]] + 1;
        const auto firstPos = positions[lastRows[i]] + 1 - lengths[i];
        if (firstPos <= 2048 && end > 2048) {
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
    if (capture)
        for (int layer = 0; layer < block_cnt; ++layer)
            if (w.meta.count("model.layers." + std::to_string(layer) + ".engram_layer.lookup_table"))
                for (int row = 0; row < rows; ++row)
                    hy4::PrepareEngramInput(w, s, layer, row, grams[row]);
    const int vocab = w.meta.at("lm_head.weight").dims[0];
    const size_t outputs = verify ? rows : lengths.size();
    auto projected = s.Get("batchedLogits", outputs * vocab);

    std::vector<uintptr_t> key;
    std::vector<const hy4::StepInput *> stepInputs;
    if (capture) {
        std::vector<hy4::StepInput> params;
        for (int row = 0; row < rows; ++row)
            params.push_back({tokens[row], grams[row][1], grams[row][2], positions[row]});
        stepInputs.push_back(hy4::UploadStep(s, params));
        key = {uintptr_t(verify), uintptr_t(rows), uintptr_t(lengths.size()), uintptr_t(projected)};
        for (int row = 0; row < rows; ++row) {
            const bool skip = tokens[row] == w.config["engram_pad_id"].int_value() || tokens[row] == 120120 ||
                              tokens[row] == 120124;
            key.push_back((positions[row] / 32) * 4 + 2 * ((positions[row] & 3) == 3) + int(skip));
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
        key.push_back(s.generation);
    }

    bool replay = capture && hy4::ReplayHybrid(*hybridGraphs, key, verify, rows);
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

        hy4::GraphCapture graph(capture ? hybridGraphs.get() : nullptr);
        if (capture && diskGraphReady)
            graph.disk = std::make_shared<DiskMoeV41Graph>();
        graph.key = key;
        graph.Start(device);
        auto streams = s.Get("streams", size_t(rows) * 4 * 3072);
        for (int row = 0; row < rows; ++row) {
            s.step = capture ? stepInputs.front() + row : nullptr;
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
                s.step = capture ? stepInputs[0] + row : nullptr;
                float *h = streams + size_t(row) * 4 * 3072;
                float *x = inputs + size_t(row) * 3072, *post = posts + row * 4;
                auto &cache = past[owners[row] * block_cnt + layer];
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
            bool dispatched = false;
            if ((hostCacheReady && decode) || capture) {
                Data input(FLOAT32, {rows, 1536}, CUDA, latents);
                Data index(INT32, {rows, 16}, CUDA, ids), score(FLOAT32, {rows, 16}, CUDA, scores);
                Data output;
                input.dataDeviceIds = index.dataDeviceIds = score.dataDeviceIds = {device};
                if (capture) {
                    graph.Close();
                    void *child = nullptr;
                    graph.outputs.push_back(std::make_shared<Data>());
                    auto &out = *graph.outputs.back();
                    if (graph.disk) {
                        child = graph.disk->Add(input, index, score, out, hostExperts[layer], 10.f);
                        AssertInFastLLM(child, "HY4-Flash failed to prepare a disk FP8 expert graph.");
                        graph.hybridChildren.push_back(child);
                    } else {
                        auto *handle = FastllmCudaCreateNativeMoeFullGraph(
                            input, index, score, out, hostExperts[layer].data(), hostExperts[layer].size(),
                            layer, nullptr, &child);
                        AssertInFastLLM(handle && child,
                                        "HY4-Flash failed to prepare a native FP8 expert graph.");
                        graph.AddNative(child, handle);
                    }
                    graph.Start(device);
                    hy4::CopyStageCaptured(routed, device, static_cast<const float *>(out.cudaData), device,
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
        size_t output = 0;
        offset = 0;
        for (size_t i = 0; i < lengths.size(); ++i) {
            for (int row = verify ? offset : lastRows[i]; row <= lastRows[i]; ++row, ++output) {
                s.step = capture ? stepInputs.back() + row : nullptr;
                hy4::ProjectHead(w, s, streams + size_t(row) * 4 * 3072, projected + output * vocab,
                                 states[i] ? &states[i]->hidden : nullptr, positions[row], draftWidth);
            }
            offset += lengths[i];
        }
        graph.FinishHybrid(verify, rows);
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
