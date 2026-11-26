#include "inkling2.h"
#include "json11.hpp"
#include "utils/utils.h"
#include "utils/recurrent_prefix_cache.h"
#include <algorithm>
#include <climits>

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
    dataType = BFLOAT16;
    canDoBatchForward = true;
    prefillBatchPolicy.targetTokens = 4096;
    prefillBatchPolicy.maxRequests = 4;
    // K and V are normalized per head. FP16 gives these bounded values
    // more mantissa precision at the same bandwidth as BF16.
    SetKVCacheDataType(FLOAT16);
}
Inkling2Model::~Inkling2Model() { ShutdownRuntime(); }
int Inkling2Model::draftKvHeadsOnDevice(int deviceId) const {
    if (!cfg.draft.steps) return 0;
    if (!runtime || !runtime->hasRankCaches()) return deviceId == placement.gpu ? cfg.draft.kvHeads : 0;
    const auto &devices = placement.tensorDevices;
    return std::find(devices.begin(), devices.end(), deviceId) == devices.end() ? 0 : cfg.draft.kvHeads / int(devices.size());
}
long long Inkling2Model::GetAutoWarmupCudaAdditionalCacheBytesPerToken(int deviceId) const {
    const auto &d = cfg.draft;
    const int heads = draftKvHeadsOnDevice(deviceId);
    if (!heads) return 0;
    const auto layers = maxBatch > 0 ? std::count(d.local.begin(), d.local.end(), false) : d.layers;
    return layers * 4LL * heads * d.dim; // Separate FP16 K and V pools.
}

long long Inkling2Model::GetAutoWarmupCudaRuntimeReserveBytes(int deviceId, int batch) const {
    const auto &d = cfg.draft;
    const int heads = draftKvHeadsOnDevice(deviceId);
    if (!heads) return 0;
    const int page = GetPageLen();
    const auto bounded = maxBatch > 0 ? std::count(d.local.begin(), d.local.end(), true) : 0;
    // Sliding histories can start inside a page. Global histories can cross
    // a page boundary while a speculative suffix is reserved for verification.
    const long long windowPages = (int64_t(d.window) + d.block + page - 2) / page + 1;
    const long long extraPages = (int64_t(d.block) - 1 + page - 1) / page;
    return (bounded * windowPages + (d.layers - bounded) * extraPages) * page * std::max(1, batch) * 4LL * heads *
           d.dim;
}

void Inkling2Model::SetDataType(DataType type) {
    if (type != DATA_AUTO_NONE && type != BFLOAT16 && type != FLOAT32)
        throw std::runtime_error("Inkling activations support auto, bfloat16 and float32");
    // Older GPUs use FP16 projections and keep the accumulated residual stream in FP32.
    dataType = type != FLOAT32 && placement.denseType == BFLOAT16 ? BFLOAT16 : FLOAT32;
}

void Inkling2Model::SetKVCacheDataType(DataType type) {
    if (kvMemConfig.enabled && type != kvCacheDataType) throw std::invalid_argument("Set KV dtype before KVMem");
    if (type != FLOAT16 && type != BFLOAT16 && type != FLOAT32 && type != FP8_E4M3)
        throw std::runtime_error("Inkling KV cache supports float16, bfloat16, float32 and fp8_e4m3");
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
    if (weight.dicts.count("vision_config.hidden_size")) {
        auto dim = [&](const char *key) { return std::stoi(weight.dicts.at(std::string("vision_config.") + key)); };
        cfg.vision = {dim("hidden_size"), dim("num_hidden_layers"), dim("num_attention_heads"),
                      dim("native_patch_size"), dim("adapter_merge_size"), dim("posemb_grid_size"),
                      dim("projection_hidden_dim"), std::stof(weight.dicts.at("vision_config.layer_norm_eps"))};
    }
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
#if defined(USE_CUDA) && !defined(USE_ROCM)
    placement.denseType = inkling2::projectionType(placement.gpu);
    if (const char *spec = std::getenv("FASTLLM_TP")) {
        std::map<int, int> ratios;
        auto devices = ParseDeviceIds(spec, "cuda", ratios);
        if (devices.size() > 1) {
            if (devices.front() != placement.gpu || !ratios.empty() ||
                std::set<int>(devices.begin(), devices.end()).size() != devices.size())
                throw std::runtime_error("Inkling TP needs distinct equally weighted CUDA devices starting with the backbone GPU");
            placement.tensorDevices = std::move(devices);
        }
    }
#endif
    SetDataType(dataType);
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
    const DataType denseType = placement.denseType;
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
        if (name.rfind("model.visual.", 0) == 0) {
            // Linear weights follow the language backbone precision. Small affine parameters
            // and the learned position table are read by FP32 accumulation kernels.
            bool matrix = name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0 &&
                          name.find(".ln_") == std::string::npos;
            result[name].emplace_back(name, matrix ? denseType : FLOAT32);
        }
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
    prefillScheduling.chunked = SupportsChunkedPrefillScheduling();
#else
    throw std::runtime_error("Inkling 2 Mini requires a CUDA build");
#endif
}
void Inkling2Model::ConfigureKvMem(const KvMemConfig &config) {
    if (mainLoop || kvMemLocked || autoWarmupRunning.load())
        throw std::invalid_argument("Configure Inkling KVMem before warmup or inference");
    if (!config.enabled) {
        kvMemConfig = config;
        if (beforeKvMem) {
            max_positions = beforeKvMem->context;
            maxBatch = beforeKvMem->batch;
            tokensLimit = beforeKvMem->tokens;
            promptLimit = beforeKvMem->prompt;
            SetSaveHistoryChat(beforeKvMem->history);
            beforeKvMem.reset();
        }
        return;
    }
    const int page = GetPageLen(), batch = std::max(1, maxBatch);
    const int context = beforeKvMem ? beforeKvMem->context : max_positions;
    if (!runtime || GetKVCacheInCPU() || config.maxTokens < 1 || config.maxTokens > context ||
        config.residentTokens <= 0 || config.residentTokens > config.maxTokens || config.residentTokens % page ||
        config.prefillTokens < 1 || !config.hostBytes || config.maxTokens > INT_MAX - cfg.draft.block ||
        (GetMaxTokens() > 0 && GetMaxTokens() < config.maxTokens))
        throw std::invalid_argument("Inkling KVMem requires a CUDA backbone, a page-aligned resident_tokens within max_tokens, and valid context and prefill limits");
    if (config.slidingHostTokens < 0) throw std::invalid_argument("KVMem sliding_host_tokens must be nonnegative");
    if (config.retrievalInterval < 1 || config.sinkTokens < 0 || config.recentTokens < 0 || config.retrievalTokens < 0 ||
        config.sinkTokens % page || config.recentTokens % page || config.retrievalTokens % page)
        throw std::invalid_argument("Inkling KVMem requires page-aligned nonnegative selection budgets and a positive retrieval interval");
    const auto hostBytes = [&](bool draft, bool local, size_t pairBytes) {
        const auto pages = cfg.mappedKvPages(config, config.maxTokens, page, draft, local);
        return uint64_t(pages.host) * page * pairBytes;
    };
    const size_t targetPair = 2 * GetDataBytes(kvCacheDataType, cfg.kvHeads, cfg.headDim <= 64 ? 64 : 128);
    uint64_t perRequest = 0;
    for (int layer : cfg.visits)
        perRequest += hostBytes(false, cfg.local[layer], targetPair);
    if (cfg.draft.steps) {
        for (bool local : cfg.draft.local)
            perRequest += hostBytes(true, local, 4ULL * cfg.draft.kvHeads * cfg.draft.dim);
    }
    if (perRequest > config.hostBytes / batch)
        throw std::invalid_argument("Inkling KVMem host_mib cannot back the target and draft KV for max_batch requests");
    if (!beforeKvMem) beforeKvMem.reset(new ServingLimits{max_positions, maxBatch, tokensLimit, promptLimit, saveHistoryChat});
    SetSaveHistoryChat(false);
    kvMemConfig = config;
    maxBatch = batch;
    max_positions = promptLimit = config.maxTokens;
    tokensLimit = std::min<int64_t>(GetMaxTokens() > 0 ? GetMaxTokens() : INT_MAX, int64_t(batch) * config.maxTokens);
}

void Inkling2Model::OnResponseContextFinished(ResponseContext *ctx) {
    if (!kvMemConfig.enabled || !ctx->isEnding || ctx->pastKeyValues.empty()) return;
    auto &root = ctx->pastKeyValues.front().first;
    root.kvStorageOwner.reset();
    for (auto &it : root.multiDeviceDatas) it.second->kvStorageOwner.reset();
}

bool Inkling2Model::SetSaveHistoryChat(bool save) {
    if (save && kvMemConfig.enabled) return false;
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
    kvMemLocked = true;
    if (runtime && (kvMemConfig.enabled || SupportsDecodePipeline() || (draftEnabled() && (tokensLimit > 0 || GetMaxTokens() > 0))))
        runtime->warmupDecodeGraph();
}
bool Inkling2Model::SupportsChunkedPrefillScheduling() const {
    return runtime && placement.tensorDevices.size() <= 1 && dataType == BFLOAT16 && kvCacheDataType == FLOAT16 &&
           !kvMemConfig.enabled && !expertTensors.empty() && expertTensors.begin()->second.ggmlType < 0 &&
           std::all_of(placement.experts.begin(), placement.experts.end(),
                       [](auto device) { return device == inkling2::ExpertDevice::Cuda; });
}
int Inkling2Model::GetChunkedPrefillSize() {
    const int chunk = basellm::GetChunkedPrefillSize();
    if (maxBatch <= 1 || !prefillScheduling.chunked || !saveHistoryChat || !SupportsChunkedPrefillScheduling()) return chunk;
    const int interval = RecurrentPrefixInterval(GetPageLen());
    // Yield at a boundary the recurrent model must already materialize. Keep
    // the user's original grid when the chunk and checkpoint lengths differ in phase.
    return chunk > interval && chunk % interval == 0 ? interval : chunk;
}
int Inkling2Model::GetBatchedPrefillTokenLimit() {
    return std::max(basellm::GetChunkedPrefillSize(), basellm::GetBatchedPrefillTokenLimit());
}
int Inkling2Model::GetPrefillBatchKey(const ResponseContext &ctx, int chunkSize) const {
    // The chunk scheduler accounts for restored pages and admits image prompts alone.
    if (prefillScheduling.chunked && SupportsChunkedPrefillScheduling()) return saveHistoryChat ? 0 : -1;
    if (saveHistoryChat || ctx.prefillOffset || ctx.cacheLen || ctx.replayingKV || !ctx.multimodalInput.empty() ||
        ctx.currentTokens.size() > size_t(chunkSize)) return -1;
    return inkling2::pooledPromptKey(ctx.currentTokens.size(), ctx.generationConfig);
}
bool Inkling2Model::SupportsDecodePipeline() const {
    return runtime && GetFastllmEnv().cudaGraph;
}
std::unique_ptr<basellm::DecodePipeline> Inkling2Model::CreateDecodePipeline(
        const std::vector<ResponseContext *> &contexts) {
    return !contexts.empty() && runtime ? runtime->pipeline(contexts) : nullptr;
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
std::pair<int, int> inkling2::VisionConfig::tokenGrid(int height, int width) const {
    if (height <= 0 || width <= 0 || patch <= 0 || merge <= 0 || patch > INT_MAX / merge)
        throw std::invalid_argument("Invalid Inkling image dimensions");
    const int side = patch * merge;
    return {1 + (height - 1) / side, 1 + (width - 1) / side};
}

int Inkling2Model::PrepareImagePrompt(const int *tokens, int n, const int *sizes, int images,
                                      int *output, int capacity, int *offsets) const {
    if (!tokens || !sizes || images <= 0 || n <= 0 || capacity < 0 ||
        !weight.weight.count("model.visual.proj_out.weight"))
        throw std::invalid_argument("Inkling image input requires images and a checkpoint with vision weights");
    int length = 0, index = 0;
    for (int i = 0; i < n; ++i) {
        int64_t repeat = 1;
        if (i && tokens[i - 1] == inkling2::imageContentToken && tokens[i] == inkling2::imagePatchToken) {
            if (index >= images) throw std::invalid_argument("Unexpected Inkling image placeholder");
            auto [rows, cols] = cfg.vision.tokenGrid(sizes[2 * size_t(index)], sizes[2 * size_t(index) + 1]);
            repeat = int64_t(rows) * cols;
            if (offsets) offsets[index] = length;
            ++index;
        }
        if (repeat > INT_MAX - length || (output && repeat > capacity - length))
            throw std::length_error("Inkling image prompt exceeds token buffer");
        if (output) std::fill_n(output + length, repeat, tokens[i]);
        length += int(repeat);
    }
    if (index != images) throw std::invalid_argument("Inkling image count does not match the template");
    return length;
}

std::vector<int> Inkling2Model::ForwardMultimodal(const Data &ids, const Data &, const Data &pos,
        std::vector<std::pair<Data, Data>> &cache, const inkling2::ImageInputs &images,
        const GenerationConfig &sampling, const LastTokensManager &history, std::vector<std::vector<float> *> *logits) {
    if (!runtime || cache.size() != size_t(block_cnt))
        throw std::runtime_error("Invalid Inkling image request cache");
    kvMemLocked = true;
    std::vector<std::pair<Data *, Data *>> slots;
    for (auto &kv : cache) slots.emplace_back(&kv.first, &kv.second);
    return runtime->forward(ids, {const_cast<Data *>(&pos)}, {int(ids.Count(0))}, slots,
                            {sampling}, history, logits, &images);
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
    kvMemLocked = true;
    return runtime->forward(ids, pos, lengths, cache, sampling, history, logits);
}
} // namespace fastllm
