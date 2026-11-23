#include "inkling2.h"
#include "json11.hpp"
#include <algorithm>

namespace fastllm {
namespace {
int cudaId(const std::string &device) {
    if (device == "cuda" || device.empty())
        return 0;
    if (device.compare(0, 5, "cuda:") == 0 && device.size() > 5 &&
        device.find_first_not_of("0123456789", 5) == std::string::npos)
        return std::stoi(device.substr(5));
    throw std::runtime_error("Inkling 2 Mini needs a CUDA backbone. Set --device cuda:N and use --moe_device "
                             "for expert placement.");
}
} // namespace
Inkling2Model::Inkling2Model() {
    model_type = model_struct = "inkling2_loop";
    canDoBatchForward = true;
    // K and V are normalized per head. FP16 gives these bounded values
    // more mantissa precision at the same bandwidth as BF16.
    SetKVCacheDataType(FLOAT16);
}
Inkling2Model::~Inkling2Model() { ShutdownRuntime(); }
long long Inkling2Model::GetAutoWarmupCudaAdditionalCacheBytesPerToken(int deviceId) const {
    const auto &d = cfg.draft;
    if (!d.steps || deviceId != placement.gpu) return 0;
    const auto layers = maxBatch > 0 ? std::count(d.local.begin(), d.local.end(), false) : d.layers;
    return layers * 4LL * d.kvHeads * d.dim; // Separate FP16 K and V pools.
}

long long Inkling2Model::GetAutoWarmupCudaRuntimeReserveBytes(int deviceId, int batch) const {
    const auto &d = cfg.draft;
    if (!d.steps || deviceId != placement.gpu) return 0;
    const int page = GetPageLen();
    const auto bounded = maxBatch > 0 ? std::count(d.local.begin(), d.local.end(), true) : 0;
    // Sliding histories can start inside a page. Global histories can cross
    // a page boundary while a speculative suffix is reserved for verification.
    const long long windowPages = (int64_t(d.window) + d.block + page - 2) / page + 1;
    const long long extraPages = (int64_t(d.block) - 1 + page - 1) / page;
    return (bounded * windowPages + (d.layers - bounded) * extraPages) * page * std::max(1, batch) * 4LL * d.kvHeads *
           d.dim;
}

void Inkling2Model::SetKVCacheDataType(DataType type) {
    if (type != FLOAT16 && type != BFLOAT16 && type != FLOAT32)
        throw std::runtime_error("Inkling KV cache supports float16, bfloat16 and float32");
    basellm::SetKVCacheDataType(type);
}
void Inkling2Model::InitParams() {
    basellm::InitParams();
    // The HF loader already flattened text_config into the weight dictionary.
    auto setting = [&](const std::string &name) {
        const auto it = weight.dicts.find("text_config." + name);
        if (it == weight.dicts.end()) return json11::Json{};
        std::string error;
        auto value = json11::Json::parse(it->second, error);
        if (!error.empty()) throw std::runtime_error("Invalid Inkling configuration: " + name);
        return value;
    };
    if (setting("loop_config.is_varloop").bool_value())
        throw std::runtime_error("Inkling needs a fixed-loop text configuration");
    auto positive = [&](const char *k) {
        int v = setting(k).int_value();
        if (v <= 0)
            throw std::runtime_error(std::string("Invalid Inkling configuration: ") + k);
        return v;
    };
    cfg.hidden = positive("hidden_size");
    cfg.layers = positive("num_hidden_layers");
    cfg.heads = positive("num_attention_heads");
    cfg.kvHeads = positive("num_key_value_heads");
    cfg.headDim = positive("head_dim");
    cfg.relative = positive("d_rel");
    cfg.extent = positive("rel_extent");
    cfg.window = positive("sliding_window_size");
    cfg.conv = positive("sconv_kernel_size");
    cfg.residualBlock = positive("n_layers_per_attn_res_block");
    cfg.experts = positive("n_routed_experts");
    cfg.topK = positive("num_experts_per_tok");
    cfg.intermediate = positive("intermediate_size");
    cfg.start = positive("loop_config.core_start");
    cfg.end = positive("loop_config.core_end");
    cfg.passes = positive("loop_config.num_passes");
    cfg.rank = positive("loop_config.gate_rank");
    cfg.vocab = positive("unpadded_vocab_size");
    cfg.logFloor = positive("log_scaling_n_floor");
    cfg.eps = setting("rms_norm_eps").number_value();
    auto normalize = setting("attn_res_normalize_values");
    cfg.normValues = !normalize.is_bool() || normalize.bool_value();
    cfg.qClip = setting("qk_l2_clip_threshold").number_value();
    cfg.rClip = setting("rel_l2_clip_threshold").number_value();
    cfg.logAlpha = setting("log_scaling_alpha").number_value();
    cfg.gateCap = setting("mlp_gate_softcap").number_value();
    cfg.upCap = setting("mlp_up_softcap").number_value();
    cfg.routeScale = setting("route_scale").number_value();
    cfg.logitCap = setting("final_logit_softcapping").number_value();
    if (cfg.start >= cfg.end || cfg.end > cfg.layers || cfg.start % cfg.residualBlock ||
        cfg.end % cfg.residualBlock || cfg.heads % cfg.kvHeads || cfg.topK > cfg.experts ||
        cfg.gateCap <= 0 || cfg.upCap <= 0 || cfg.eps <= 0)
        throw std::runtime_error("Invalid Inkling loop or attention geometry");
    cfg.local.assign(cfg.layers, false);
    const auto localLayers = setting("local_layer_ids");
    for (const auto &i : localLayers.array_items())
        cfg.local.at(i.int_value()) = true;
    weightMergeRules.clear();
    auto merge = [&](const std::string &prefix, const std::vector<std::string> &parts,
                     const std::string &output) {
        std::vector<std::string> inputs;
        for (const auto &part : parts) inputs.push_back(prefix + part);
        weightMergeRules.emplace_back(std::vector<WeightMergeRuleSingle>{{inputs, prefix + output, "linear"}});
    };
    for (int i = 0; i < cfg.layers; ++i) {
        const std::string layer = "model.llm.layers." + std::to_string(i);
        std::vector<std::string> parts{"wq_du.weight", "wr_du.weight", "gate_attn.weight",
                                       "attn_sink_proj.weight", "wk_dv.weight"};
        if (!cfg.local[i]) parts.push_back("k_sconv.weight_proj");
        parts.push_back("wv_dv.weight");
        if (!cfg.local[i]) parts.push_back("v_sconv.weight_proj");
        merge(layer + ".attn.", parts, "packed.weight");
        if (i < cfg.start || i >= cfg.end) continue;
        const auto controllers = layer + ".controllers.";
        merge(controllers, {"attn.down_weight", "k.down_weight", "v.down_weight"}, "qkv.down_weight");
        merge(controllers, {"attn.gain_weight", "attn.forget_weight", "k.gain_weight",
                            "k.forget_weight", "v.gain_weight", "v.forget_weight"}, "qkv.gates_weight");
        merge(controllers, {"mlp.gain_weight", "mlp.forget_weight"}, "mlp.gates_weight");
    }
    cfg.visits.clear();
    for (int layer = 0; layer < cfg.start; ++layer)
        cfg.visits.push_back(layer);
    for (int pass = 0; pass < cfg.passes; ++pass)
        for (int layer = cfg.start; layer < cfg.end; ++layer)
            cfg.visits.push_back(layer);
    for (int layer = cfg.end; layer < cfg.layers; ++layer)
        cfg.visits.push_back(layer);
    auto &draft = cfg.draft;
    draft = {};
    const char *steps = std::getenv("FASTLLM_DSPARK_TOKENS");
    draft.steps = steps ? std::stoi(steps) : 0;
    if (draft.steps < 0) throw std::runtime_error("Negative Inkling proposal count");
    if (draft.steps > 0) {
        auto read = [&](const std::string &key) {
            const auto it = weight.dicts.find("mtp_config." + key);
            if (it == weight.dicts.end()) throw std::runtime_error("Missing Inkling draft setting: " + key);
            std::string error;
            auto value = json11::Json::parse(it->second, error);
            if (!error.empty()) throw std::runtime_error("Invalid Inkling draft setting: " + key);
            return value;
        };
        draft.block = read("block_size").int_value();
        draft.maxPositions = read("max_position_embeddings").int_value();
        draft.layers = read("num_hidden_layers").int_value();
        draft.heads = read("num_attention_heads").int_value();
        draft.kvHeads = read("num_key_value_heads").int_value();
        draft.dim = read("head_dim").int_value();
        draft.intermediate = read("intermediate_size").int_value();
        draft.rank = read("markov_rank").int_value();
        draft.window = read("sliding_window_size").int_value();
        draft.theta = read("rope_theta").number_value();
        draft.eps = read("rms_norm_eps").number_value();
        if (draft.steps >= draft.block || draft.maxPositions < draft.block || draft.layers <= 0 || draft.heads <= 0 || draft.kvHeads <= 0 ||
            draft.heads % draft.kvHeads || draft.dim != 128 || draft.rank <= 0 || draft.intermediate <= 0 ||
            draft.theta <= 0 || draft.eps <= 0 || draft.window <= 0 || read("hidden_size").int_value()!=cfg.hidden ||
            weight.dicts.at("mtp_config.markov_head_type") != "rnn_hidden" ||
            read("q_bias").bool_value() || read("o_bias").bool_value())
            throw std::runtime_error("Unsupported Inkling DSpark configuration or proposal count");
        draft.local.assign(draft.layers, false);
        auto local = read("local_layer_ids"), taps = read("target_layer_ids");
        for (auto &i : local.array_items()) draft.local.at(i.int_value()) = true;
        for (auto &i : taps.array_items()) {
            int id = i.int_value();
            if (id < 0 || (!draft.taps.empty() && id <= draft.taps.back()))
                throw std::runtime_error("Invalid Inkling draft target visits");
            draft.taps.push_back(id);
        }
        if (draft.taps.empty() || draft.taps.back() != int(cfg.visits.size()))
            throw std::runtime_error("Inkling draft must include the final target normalization");
    }
    placement.gpu = cudaId(SelectDeviceFromMap(deviceMap, 1, 1));
    for (const auto &item : deviceMap)
        if (cudaId(item.first) != placement.gpu)
            throw std::runtime_error("Inkling currently supports one backbone GPU");
    // The scheduler counts visits. Expert residency belongs to a physical weight.
    placement.experts.clear();
    block_cnt = cfg.layers;
    for (int i = 0; i < cfg.layers; ++i) {
        auto dev = SelectMoeDeviceForLayer(i);
        if (dev == "cpu")
            placement.experts.push_back(inkling2::ExpertDevice::Cpu);
        else if (dev == "numa" || dev == "numas")
            placement.experts.push_back(inkling2::ExpertDevice::Numa);
        else {
            if (!dev.empty() && cudaId(dev) != placement.gpu)
                throw std::runtime_error("Inkling expert GPU must match the backbone GPU");
            placement.experts.push_back(inkling2::ExpertDevice::Cuda);
        }
    }
    block_cnt = cfg.visits.size();
    embed_dim = cfg.hidden;
    num_attention_heads = cfg.heads;
    num_key_value_heads = cfg.kvHeads;
    head_dim = cfg.headDim;
    max_positions = positive("model_max_length");
    for (int slot = 0; slot < block_cnt; ++slot) {
        if (!cfg.local[cfg.visits[slot]]) {
            kvCacheId = slot;
            break;
        }
    }
}
std::map<std::string, std::vector<std::pair<std::string, DataType>>>
Inkling2Model::GetTensorMap(const std::vector<std::string> &names) {
    std::map<std::string, std::vector<std::pair<std::string, DataType>>> result;
    DataType denseType = BFLOAT16;
#if defined(USE_CUDA) && !defined(USE_ROCM)
    denseType = inkling2::projectionType(placement.gpu);
#endif
    std::set<std::string> dense{"model.llm.unembed.weight"};
    for (const auto &group : weightMergeRules) {
        dense.insert(group.allInputs.begin(), group.allInputs.end());
        for (const auto &rule : group.rules) dense.insert(rule.output);
    }
    for (int i = 0; i < cfg.layers; ++i) {
        const auto prefix = "model.llm.layers." + std::to_string(i);
        dense.insert(prefix + ".attn.wo_ud.weight");
        dense.insert(prefix + ".controllers.mlp.down_weight");
    }
    for (const auto &name : names) {
        if (draftEnabled() && name.rfind("inkling_draft.", 0) == 0) {
            const std::string key = name.substr(14);
            bool matrix = (key.find("_proj.weight") != std::string::npos ||
                           key.find("_proj_ctx.weight") != std::string::npos ||
                           key == "markov_head.hidden_out.weight" || key == "markov_head.markov_w1.weight");
            result[name].emplace_back("model.llm.draft." + key, matrix ? denseType : FLOAT32);
        }
        if (name.compare(0, 10, "model.llm.") == 0 && name.find(".mlp.experts.") == std::string::npos)
            result[name].emplace_back(name, dense.count(name) ? denseType : FLOAT32);
    }
    return result;
}
void Inkling2Model::OnModelWeightsLoaded() {
#if defined(USE_CUDA) && !defined(USE_ROCM)
    runtime = inkling2::create(cfg, placement, *this, expertTensors);
#else
    throw std::runtime_error("Inkling 2 Mini requires a CUDA build");
#endif
}
bool Inkling2Model::SetSaveHistoryChat(bool save) {
    std::lock_guard<std::mutex> lock(dictLocker);
    if (runtime) runtime->setPrefixCache(save);
    return basellm::SetSaveHistoryChat(save);
}
int Inkling2Model::QueryPagedPrefixCacheExtra(ResponseContext *ctx, int limit) const {
    return saveHistoryChat && runtime && ctx ? runtime->prefix(*ctx, limit, false) : 0;
}
bool Inkling2Model::RestorePagedPrefixCacheExtra(ResponseContext *ctx, int length) const {
    return length > 0 && saveHistoryChat && runtime && ctx && runtime->prefix(*ctx, length, true) == length;
}
void Inkling2Model::WarmUp() {
    if (SupportsDecodePipeline())
        runtime->warmupDecodeGraph();
}
bool Inkling2Model::SupportsDecodePipeline() const {
    return runtime && GetFastllmEnv().cudaGraph &&
           std::all_of(placement.experts.begin(), placement.experts.end(),
                       [](inkling2::ExpertDevice d) { return d == inkling2::ExpertDevice::Cuda; });
}
std::unique_ptr<basellm::DecodePipeline> Inkling2Model::CreateDecodePipeline(
        const std::vector<ResponseContext *> &contexts) {
    return contexts.size() == 1 && runtime ? runtime->pipeline(*contexts.front()) : nullptr;
}
int Inkling2Model::Forward(const Data &ids, const Data &mask, const Data &pos,
                           std::vector<std::pair<Data, Data>> &cache, const GenerationConfig &sampling,
                           const LastTokensManager &history, std::vector<float> *logits) {
    std::vector<std::pair<Data *, Data *>> slots;
    slots.reserve(cache.size());
    for (auto &item : cache)
        slots.emplace_back(&item.first, &item.second);
    std::vector<std::vector<float> *> outputs{logits};
    return ForwardV2(1, ids, {const_cast<Data *>(&mask)}, {const_cast<Data *>(&pos)},
                     {static_cast<int>(ids.Count(0))}, slots, {sampling}, history, &outputs)
        .at(0);
}
// The legacy fixed-batch API owns one cache vector for every row of a padded batch.
std::vector<int> Inkling2Model::ForwardBatch(int batch, const Data &ids, const Data &, const Data &pos,
                                             std::vector<std::pair<Data, Data>> &cache,
                                             const GenerationConfig &sampling,
                                             const LastTokensManager &history,
                                             std::vector<std::vector<float> *> *logits) {
    using Slots = std::vector<std::pair<Data, Data>>;
    if (batch <= 0 || cache.empty() || ids.dims.size() != 2 || ids.dims[0] != batch || pos.dims != ids.dims ||
        ids.dataType != FLOAT32 || pos.dataType != FLOAT32 || ids.dims[1] <= 0)
        throw std::runtime_error("Invalid Inkling padded batch");
    if (batch == 1) {
        auto *output = logits && !logits->empty() ? logits->front() : nullptr;
        return {Forward(ids, Data(), pos, cache, sampling, history, output)};
    }
    if (cache.front().second.isPagedKVCache)
        throw std::runtime_error("Inkling fixed batch cannot reuse a single-request cache");
    auto &owner = cache.front().second.kvStorageOwner;
    bool starting = !owner;
    if (starting)
        owner = std::make_shared<Slots>(size_t(batch) * block_cnt);
    auto &storage = *static_cast<Slots *>(owner.get());
    if (storage.size() != size_t(batch) * block_cnt)
        throw std::runtime_error("Inkling fixed batch size changed");
    const_cast<Data &>(ids).ToDevice(CPU);
    const_cast<Data &>(pos).ToDevice(CPU);
    auto *values = reinterpret_cast<const float *>(ids.cpuData);
    auto *positions = reinterpret_cast<const float *>(pos.cpuData);
    int width = ids.dims[1];
    std::vector<int> lengths;
    std::vector<float> tokens;
    std::vector<Data> positionRows;
    positionRows.reserve(batch);
    std::vector<Data *> positionPointers;
    std::vector<std::pair<Data *, Data *>> slots;
    for (int b = 0; b < batch; ++b) {
        int end = (b + 1) * width, len = starting ? std::min(width, int(positions[end - 1]) + 1) : width;
        if (len <= 0)
            throw std::runtime_error("Invalid Inkling padded sequence length");
        lengths.push_back(len);
        tokens.insert(tokens.end(), values + end - len, values + end);
        positionRows.emplace_back(FLOAT32, std::vector<int>{1, len},
                                  std::vector<float>(positions + end - len, positions + end));
        positionPointers.push_back(&positionRows.back());
    }
    for (auto &item : storage)
        slots.emplace_back(&item.first, &item.second);
    Data packed(FLOAT32, {1, int(tokens.size())}, tokens);
    return ForwardV2(batch, packed, {}, positionPointers, lengths, slots,
                     std::vector<GenerationConfig>(batch, sampling), history, logits);
}
std::string Inkling2Model::MakeInput(const std::string &history, int, const std::string &input) {
    const std::string prefix =
        history.empty() ? "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                        : history;
    return prefix + "<|message_user|><|content_text|>" + input + "<|end_message|>";
}
std::string Inkling2Model::MakeHistory(const std::string &history, int round, const std::string &input,
                                       const std::string &output) {
    auto message = output.compare(0, 17, "<|message_model|>") == 0
                       ? output
                       : "<|message_model|><|content_text|>" + output + "<|end_message|>";
    return MakeInput(history, round, input) + message + "<|content_model_end_sampling|>";
}
std::vector<int> Inkling2Model::ForwardV2(int batch, const Data &ids, const std::vector<Data *> &,
                                          const std::vector<Data *> &pos, const std::vector<int> &lengths,
                                          std::vector<std::pair<Data *, Data *>> &cache,
                                          const std::vector<GenerationConfig> &sampling,
                                          const LastTokensManager &history,
                                          std::vector<std::vector<float> *> *logits) {
    if (!runtime || batch <= 0 || pos.size() != size_t(batch) || sampling.size() != size_t(batch) ||
        lengths.size() != size_t(batch) || cache.size() != size_t(batch) * cfg.visits.size())
        throw std::runtime_error("Inkling forward received an invalid cache layout");
    return runtime->forward(ids, pos, lengths, cache, sampling, history, logits);
}
} // namespace fastllm
