#include "clover.h"
#include "devices/cuda/fastllm-clover.h"
#include "utils/recurrent_prefix_cache.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(std::string("Clover: ") + message);
}
bool Has(const json11::Json &j, const char *key) {
    return j.object_items().count(key) != 0 && !j[key].is_null();
}
float Real(const json11::Json &j, const char *key, float value) {
    if (!Has(j, key)) return value;
    Require(j[key].is_number() && std::isfinite(j[key].number_value()), key);
    float result = j[key].number_value();
    Require(std::isfinite(result), key);
    return result;
}
int Integer(const json11::Json &j, const char *key, int value) {
    if (!Has(j, key)) return value;
    double v = j[key].number_value();
    Require(j[key].is_number() && std::isfinite(v) && std::floor(v) == v &&
        v >= std::numeric_limits<int>::min() && v <= std::numeric_limits<int>::max(), key);
    return int(v);
}
bool Boolean(const json11::Json &j, const char *key, bool value) {
    if (!Has(j, key)) return value;
    Require(j[key].is_bool(), key);
    return j[key].bool_value();
}
}

struct CloverModel::PrefixCache {
    struct KVChunk {
        int begin = 0, end = 0;
        std::shared_ptr<const KVChunk> previous;
        std::vector<std::pair<Data, Data>> layers;
    };
    struct Snapshot : RecurrentPrefixKey {
        DataType type = BFLOAT16;
        std::shared_ptr<const KVChunk> kv;
        std::vector<std::pair<Data, Data>> recurrent;
        Data logits;
    };
    RecurrentPrefixStore<Snapshot> store;
};

struct CloverModel::Request {
    CloverOptions options;
    std::vector<int> counts, stops, block, commit;
    std::vector<bool> pending;
    std::mt19937_64 rng;
    size_t prefilled = 0;
    int bonus = -1, limit = 0;
    CloverSnapshots snapshots;
    Data countsGpu, stopsGpu, biasGpu, candidatesGpu, statsGpu, statsWorkspace;
    int countsVersion = -1;
    bool prefixChecked = false;
    int restoredTokens = 0, lastSavedLength = 0;
    long long cacheRequestId = 0;
    std::shared_ptr<PrefixCache::Snapshot> prefixSnapshot;
#ifdef USE_CUDA
    void Upload(CloverModel &m, Data &data, DataType type, const std::vector<int> &dims, const void *source) {
        data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
        data.ToDevice(DataDevice::CUDA, std::vector<int>{m.cudaDevice}, false);
        data.Allocate(false);
        if (source) FastllmCudaCopyFromHostToDevice(data.cudaData, (void *)source, data.GetBytes());
    }
    void Penalize(CloverModel &m, ResponseContext &ctx, Data &logits, const std::vector<int> &visible, bool preview, bool initial = false) {
        const auto &o = options; const auto &c = ctx.generationConfig;
        if (countsVersion != ctx.curTokens) {
            Upload(m, countsGpu, INT32, {m.vocabSize}, counts.data());
            countsVersion = ctx.curTokens;
        }
        if (stopsGpu.dims.empty() && !stops.empty())
            Upload(m, stopsGpu, INT32, {int(stops.size())}, stops.data());
        int rows = logits.Count(0) / m.vocabSize;
        auto bias = CloverRowBias(rows, m.vocabSize, counts, ctx.curTokens, visible, o, preview, initial);
        Upload(m, biasGpu, FLOAT32, {rows, 2}, bias.data());
        CloverCudaPenalty p{c.repeat_penalty, o.presence, o.frequency, o.speculative ? -1 : m.maskToken,
            o.thinkToken, ctx.curTokens + (o.speculative && preview ? 0 : 1), c.output_token_least};
        if (!FastllmCudaCloverPenalize((float *)logits.cudaData, (int *)countsGpu.cudaData,
            (int *)stopsGpu.cudaData, stops.size(), (float *)biasGpu.cudaData, rows, m.vocabSize, p))
            throw std::runtime_error("Clover: GPU penalties failed");
    }
    std::vector<CloverCudaTokenStats> Summarize(CloverModel &m, Data &logits, const std::vector<int> &candidates, bool confidence, bool entropy) {
        int rows = logits.Count(0) / m.vocabSize;
        if (!candidates.empty()) Upload(m, candidatesGpu, INT32, {rows}, candidates.data());
        static_assert(sizeof(CloverCudaTokenStats) == 5 * sizeof(int), "GPU token statistics layout");
        Upload(m, statsGpu, INT32, {rows, 5}, nullptr);
        size_t bytes = FastllmCudaCloverTokenStatsWorkspaceBytes(rows, m.vocabSize);
        if (bytes) Upload(m, statsWorkspace, INT32, {int((bytes + 3) / 4)}, nullptr);
        if (!FastllmCudaCloverTokenStats((float *)logits.cudaData,
            candidates.empty() ? nullptr : (int *)candidatesGpu.cudaData,
            (CloverCudaTokenStats *)statsGpu.cudaData, rows, m.vocabSize, confidence, entropy,
            nullptr, bytes ? statsWorkspace.cudaData : nullptr))
            throw std::runtime_error("Clover: GPU sampling statistics failed");
        std::vector<CloverCudaTokenStats> result(rows);
        FastllmCudaCopyFromDeviceToHost(result.data(), statsGpu.cudaData, result.size() * sizeof(result[0]));
        for (const auto &x : result) if (x.error) throw std::runtime_error("Clover: invalid sampling logits");
        return result;
    }
    int Sample(CloverModel &m, ResponseContext &ctx, Data &logits, int row, int greedyToken) {
        const auto &c = ctx.generationConfig;
        if (!c.do_sample) return greedyToken;
        // The stochastic bonus needs one row only; draft/acceptance remain on GPU.
        std::vector<float> scores(m.vocabSize);
        FastllmCudaCopyFromDeviceToHost(scores.data(), (float *)logits.cudaData + (size_t)row * m.vocabSize,
            scores.size() * sizeof(float));
        return CloverSample(scores.data(), m.vocabSize, c, rng);
    }
    std::vector<int> Accept(CloverModel &m, ResponseContext &ctx, Data &logits,
            const std::vector<int> &candidates) {
        Penalize(m, ctx, logits, candidates, false);
        auto stats = Summarize(m, logits, candidates, !options.exact, !options.exact);
        int accepted = 0;
        while (accepted + 1 < (int)candidates.size()) {
            const auto &s = stats[accepted];
            bool accept = options.exact ? candidates[accepted + 1] == s.token :
                s.probability > 0.f && s.probability >= std::min(options.acceptRatio,
                    options.acceptDelta * std::exp(-s.entropy));
            if (!accept) break;
            ++accepted;
        }
        bonus = Sample(m, ctx, logits, accepted, stats[accepted].token);
        m.RestorePrefix(ctx.pastKeyValues, snapshots, accepted + 1);
        std::vector<int> output(candidates.begin() + 1, candidates.begin() + accepted + 1);
        output.push_back(bonus);
        return output;
    }
#endif
};

void CloverModel::LoadPrefix(ResponseContext &ctx, Request &r, Data &logits) {
    r.prefixChecked = true;
    if (!RecurrentPrefixCacheEnabled()) {
        prefixCache.reset();
        return;
    }
    if (!prefixCache) return;
#ifdef USE_CUDA
    EnsureResident();
    auto snapshot = prefixCache->store.Find(ctx.currentTokens,
        int(ctx.currentTokens.size()), -1, [&](const PrefixCache::Snapshot &s) {
            return s.type == dataType && (!r.options.speculative ||
                s.cachedLen < int(ctx.currentTokens.size()) || !s.logits.dims.empty());
        });
    if (!snapshot) return;
    FastllmCudaSetDevice(cudaDevice);
    // Restore independent writable storage. Cached chunks and states are immutable.
    auto restore = [&](const Data &src, Data &dst) {
        dst.dataType = src.dataType; dst.UpdateUnitSize();
        dst.Resize(src.dims);
        dst.ToDevice(DataDevice::CUDA, std::vector<int>{cudaDevice}, false);
        dst.Allocate(false);
        FastllmCudaCopyFromHostToDevice(dst.cudaData, src.cpuData, src.GetBytes());
    };
    std::vector<std::pair<Data, Data>> restored(block_cnt);
    for (int i = 0; i < block_cnt; ++i) for (bool second : {false, true}) {
        auto &dst = second ? restored[i].second : restored[i].first;
        if (linearLayers[i]) {
            const auto &src = second ? snapshot->recurrent[i].second : snapshot->recurrent[i].first;
            restore(src, dst);
        } else {
            dst.dataType = dataType; dst.UpdateUnitSize();
            dst.ToDevice(DataDevice::CUDA, std::vector<int>{cudaDevice}, false);
            dst.Expansion({(snapshot->cachedLen + 255) / 256 * 256, num_key_value_heads, head_dim});
            dst.Resize({snapshot->cachedLen, num_key_value_heads, head_dim});
            const size_t rowBytes = size_t(num_key_value_heads) * head_dim * 2;
            for (auto chunk = snapshot->kv; chunk; chunk = chunk->previous) {
                const auto &src = second ? chunk->layers[i].second : chunk->layers[i].first;
                FastllmCudaCopyFromHostToDevice((char *)dst.cudaData + size_t(chunk->begin) * rowBytes,
                    src.cpuData, size_t(chunk->end - chunk->begin) * rowBytes);
            }
        }
        dst.isKVCache = true;
        dst.isLinearAttention = linearLayers[i];
    }
    if (snapshot->cachedLen == int(ctx.currentTokens.size()) && r.options.speculative) {
        restore(snapshot->logits, logits);
    }
    ctx.pastKeyValues.swap(restored);
    r.prefilled = r.restoredTokens = r.lastSavedLength = snapshot->cachedLen;
    r.prefixSnapshot = std::move(snapshot);
#endif
}

void CloverModel::SavePrefix(ResponseContext &ctx, Request &r, const Data *logits) {
    if (!RecurrentPrefixCacheEnabled() || ctx.pastKeyValues.size() != size_t(block_cnt)) return;
    const auto &key = ctx.pastKeyValues[kvCacheId].first;
    int length = key.dims.empty() ? 0 : key.dims[0];
    // Stop/length can truncate an emitted block before its already-committed
    // state. Such a state must never be indexed under the shorter visible text.
    if (length <= r.lastSavedLength || length > int(ctx.allTokens.size())) return;
#ifdef USE_CUDA
    if (!prefixCache) prefixCache = std::make_shared<PrefixCache>();
    auto existing = prefixCache->store.Find(ctx.allTokens, length, length,
        [&](const PrefixCache::Snapshot &s) { return s.type == dataType && (!logits || !s.logits.dims.empty()); });
    if (existing) {
        r.prefixSnapshot = std::move(existing);
        r.lastSavedLength = length;
        return;
    }
    FastllmCudaSetDevice(cudaDevice);
    auto snapshot = std::make_shared<PrefixCache::Snapshot>();
    snapshot->cachedLen = length; snapshot->requestId = r.cacheRequestId;
    snapshot->tokens.assign(ctx.allTokens.begin(), ctx.allTokens.begin() + length);
    snapshot->type = dataType;
    snapshot->recurrent.resize(block_cnt);
    auto chunk = std::make_shared<PrefixCache::KVChunk>();
    chunk->previous = r.prefixSnapshot ? r.prefixSnapshot->kv : nullptr;
    chunk->begin = chunk->previous ? chunk->previous->end : 0;
    chunk->end = length;
    chunk->layers.resize(block_cnt);
    // Host checkpoints do not consume the GPU inference budget. KV is copied
    // incrementally, once per committed span, and shared by interval snapshots.
    auto copy = [&](const Data &src, Data &dst, std::vector<int> shape, size_t offset = 0) {
        dst.dataType = src.dataType; dst.UpdateUnitSize();
        dst.Resize(shape); dst.Allocate();
        dst.isLinearAttention = src.isLinearAttention;
        FastllmCudaCopyFromDeviceToHost(dst.cpuData, (char *)src.cudaData + offset, dst.GetBytes());
    };
    for (int i = 0; i < block_cnt; ++i) for (bool second : {false, true}) {
        const auto &src = second ? ctx.pastKeyValues[i].second : ctx.pastKeyValues[i].first;
        if (linearLayers[i]) {
            auto &dst = second ? snapshot->recurrent[i].second : snapshot->recurrent[i].first;
            copy(src, dst, src.dims);
        } else {
            auto &dst = second ? chunk->layers[i].second : chunk->layers[i].first;
            copy(src, dst, {length - chunk->begin, num_key_value_heads, head_dim},
                size_t(chunk->begin) * num_key_value_heads * head_dim * 2);
        }
    }
    if (logits) copy(*logits, snapshot->logits, logits->dims);
    snapshot->kv = chunk;
    prefixCache->store.Publish(std::move(snapshot), r.prefixSnapshot,
        RecurrentPrefixMaxPerRequest(), RecurrentPrefixMaxRecords(maxBatch));
    r.lastSavedLength = length;
#endif
}

CloverModel::~CloverModel() {
    ShutdownRuntime();
    moeGraphs.reset();
}

CloverOptions CloverModel::ReadOptions(const json11::Json &j) const {
    Require(j.is_object(), "generation options must be an object");
    auto o = defaults;
    if (Has(j, "mode")) {
        const auto mode = j["mode"].string_value();
        Require(mode == "low_confidence" || mode == "self_speculative", "unknown generation mode");
        o.speculative = mode == "self_speculative";
    }
    o.width = Integer(j, "block_size", o.width);
    o.threshold = Real(j, "threshold", o.threshold);
    if (Has(j, "accept_rule")) {
        const auto rule = j["accept_rule"].string_value();
        Require(rule == "threshold" || rule == "exact", "unknown acceptance rule");
        o.exact = rule == "exact";
    }
    o.acceptRatio = Real(j, "accept_ratio", o.acceptRatio);
    o.acceptDelta = Real(j, "accept_delta", o.acceptDelta);
    o.presence = Real(j, "presence_penalty", o.presence);
    o.frequency = Real(j, "frequency_penalty", o.frequency);
    o.lengthPenalty = Real(j, "dllm_length_penalty", o.lengthPenalty);
    o.lengthStart = Integer(j, "dllm_length_penalty_start", o.lengthStart);
    o.lengthWindow = Integer(j, "dllm_length_penalty_window", o.lengthWindow);
    o.thinkBias = Real(j, "dllm_think_end_bias", o.thinkBias);
    o.thinkStart = Integer(j, "dllm_think_end_start", o.thinkStart);
    o.thinkWindow = Integer(j, "dllm_think_end_window", o.thinkWindow);
    o.thinkToken = Integer(j, "dllm_think_end_token_id", o.thinkToken);
    o.ignoreEos = Boolean(j, "ignore_eos", o.ignoreEos);
    if (Has(j, "seed")) {
        int seed = Integer(j, "seed", 0);
        Require(seed >= 0, "seed must be nonnegative");
        o.seed = seed; o.hasSeed = true;
    }
    Require(o.width > 0 && o.width <= blockSize && blockSize % o.width == 0, "block_size must divide 32");
    Require(o.threshold >= 0.f && o.threshold <= 1.f, "threshold must be in [0, 1]");
    Require(o.acceptRatio >= 0.f && o.acceptRatio <= 1.f && o.acceptDelta >= 0.f,
        "invalid acceptance threshold");
    Require(std::abs(o.presence) <= 2.f && std::abs(o.frequency) <= 2.f, "penalties must be in [-2, 2]");
    Require(o.lengthPenalty >= 0.f && o.thinkBias >= 0.f && o.lengthStart >= 0 && o.thinkStart >= 0 &&
        o.lengthWindow > 0 && o.thinkWindow > 0, "invalid length/thinking bias");
    Require(!o.thinkBias || (o.thinkToken >= 0 && o.thinkToken < vocabSize), "invalid thinking-end token");
    return o;
}

void CloverModel::ConfigureClover(const json11::Json &j) {
    std::lock_guard<std::mutex> guard(launchLocker);
    Require(mainLoop == nullptr, "configure generation defaults before launching requests");
    defaults = ReadOptions(j);
}

void CloverModel::ValidateRequest(const std::vector<int> &tokens, const GenerationConfig &c,
        const CloverOptions &o) const {
    const int capacity = contextPlan.configured ? contextPlan.effectiveLength : max_positions;
    Require(maxBatch <= CloverCudaGdnCaches::Capacity, "max_batch exceeds the Clover cache capacity (64)");
    Require(maxBatch <= 1 || o.speculative, "batch generation currently requires self_speculative mode");
    Require(maxBatch <= 1 || o.width >= 16, "batch generation requires block_size 16 or 32");
    Require(!tokens.empty(), "empty prompt");
    for (int t : tokens) Require(t >= 0 && t < vocabSize, "invalid prompt token");
    for (int t : c.stop_token_ids) Require(t >= 0 && t < vocabSize, "invalid stop token");
    Require(std::isfinite(c.repeat_penalty) && c.repeat_penalty > 0.f && c.repeat_penalty <= 2.f,
        "repetition_penalty must be in (0, 2]");
    Require(std::isfinite(c.temperature) && (!c.do_sample || c.temperature > 0.f), "invalid sampling temperature");
    Require(std::isfinite(c.top_p) && c.top_p > 0.f && c.top_p <= 1.f && c.top_k >= -1, "invalid top_k/top_p");
    Require(c.output_token_least >= 0 && (c.output_token_limit <= 0 || c.output_token_least <= c.output_token_limit),
        "min_new_tokens exceeds max_new_tokens");
    Require(!c.output_logits, "output_logits is not supported by block generation");
    Require(!c.tool_call_name_constraint_enabled && !c.tool_call_parameter_name_constraint_enabled &&
        !c.tool_call_content_sampling_enabled && c.tool_call_allowed_token_ids.empty(),
        "constrained tool sampling is not supported by block generation");
    long long space = (long long)capacity - tokens.size();
    long long needed = c.output_token_limit > 0 ? c.output_token_limit : (o.speculative ? 1 : o.width);
    if (!o.speculative) needed = (needed + o.width - 1) / o.width * o.width;
    Require(space >= needed, "prompt plus complete generation blocks exceeds context length");
}

int CloverModel::LaunchResponseTokens(const std::vector<int> &tokens, const GenerationConfig &c,
        const std::map<std::string, std::vector<Data *>> &multimodal) {
    std::lock_guard<std::mutex> guard(launchLocker);
    Require(multimodal.empty(), "only text requests are supported");
    launchingOptions = defaults;
    ValidateRequest(tokens, c, launchingOptions);
    return basellm::LaunchResponseTokens(tokens, c);
}

int CloverModel::LaunchClover(const std::vector<int> &tokens, const json11::Json &j) {
    std::lock_guard<std::mutex> guard(launchLocker);
    auto options = ReadOptions(j);
    GenerationConfig c;
    c.output_token_limit = Integer(j, "max_new_tokens", 24576);
    Require(c.output_token_limit > 0, "max_new_tokens must be positive");
    c.output_token_least = Integer(j, "min_new_tokens", 0);
    c.repeat_penalty = Real(j, "repetition_penalty", 1.05f);
    c.temperature = Real(j, "temperature", 1.f);
    c.do_sample = Boolean(j, "do_sample", false);
    if (c.temperature == 0.f) c.do_sample = false;
    c.top_k = Integer(j, "top_k", 0);
    c.top_p = Real(j, "top_p", 1.f);
    c.output_logits = Boolean(j, "output_logits", false);
    if (Has(j, "stop_token_ids")) {
        Require(j["stop_token_ids"].is_array(), "stop_token_ids must be an array");
        for (const auto &t : j["stop_token_ids"].array_items()) {
            Require(t.is_number() && t.number_value() == t.int_value(), "invalid stop token");
            c.stop_token_ids.insert(t.int_value());
        }
    }
    ValidateRequest(tokens, c, options);
    launchingOptions = options;
    return basellm::LaunchResponseTokens(tokens, c);
}

void CloverModel::OnResponseContextCreated(ResponseContext *ctx) {
    auto r = std::make_shared<Request>();
    r->cacheRequestId = ++nextRequestId;
    r->options = launchingOptions;
    r->counts.assign(vocabSize, 0);
    r->rng.seed(r->options.hasSeed ? r->options.seed : std::random_device{}());
    std::set<int> stops = eos_token_ids;
    stops.insert(eos_token_id);
    stops.insert(ctx->generationConfig.stop_token_ids.begin(), ctx->generationConfig.stop_token_ids.end());
    auto pad = weight.dicts.find("pad_token_id");
    if (pad != weight.dicts.end() && pad->second != "null") stops.insert(std::stoi(pad->second));
    for (int t : stops) if (t >= 0 && t < vocabSize) r->stops.push_back(t);
    int available = (contextPlan.configured ? contextPlan.effectiveLength : max_positions) - ctx->inputTokens;
    if (!r->options.speculative) available = available / r->options.width * r->options.width;
    r->limit = ctx->generationConfig.output_token_limit > 0 ? ctx->generationConfig.output_token_limit : available;
    for (int i = 0; i < block_cnt; ++i) {
        ctx->pastKeyValues[i].first.isLinearAttention = linearLayers[i];
        ctx->pastKeyValues[i].second.isLinearAttention = linearLayers[i];
    }
    requests.emplace(ctx, std::move(r));
}

void CloverModel::OnResponseContextRemoved(ResponseContext *ctx) { requests.erase(ctx); }

std::vector<int> CloverModel::Step(ResponseContext &ctx, Request &r) {
    const auto &o = r.options;
    const auto &c = ctx.generationConfig;
#ifdef USE_CUDA
    if (!r.prefixChecked) {
        Data logits;
        LoadPrefix(ctx, r, logits);
        if (r.prefilled == ctx.currentTokens.size() && o.speculative) {
            r.Penalize(*this, ctx, logits, {}, false, true);
            auto stats = r.Summarize(*this, logits, {}, false, false);
            r.bonus = r.Sample(*this, ctx, logits, 0, stats[0].token);
            return {r.bonus};
        }
    }
    if (r.prefilled < ctx.currentTokens.size()) {
        // Bounded chunks allow aborts between forwards and avoid prompt-wide
        // expert intermediates. All chunks commit causal state, never previews.
        int chunk = GetChunkedPrefillSize();
        if (chunk <= 0) chunk = defaultChunkedPrefillSize;
        size_t count = std::min(ctx.currentTokens.size() - r.prefilled, size_t(chunk));
        // Merge complete 512-token chunks while keeping the final short
        // tail separate. This preserves its GEMV/attention dispatch instead
        // of folding a decode-sized tail into a larger BF16 GEMM.
        if (count > 512) count -= count % 512;
        size_t end = r.prefilled + count;
        std::vector<int> tokens(ctx.currentTokens.begin() + r.prefilled, ctx.currentTokens.begin() + end);
        bool last = end == ctx.currentTokens.size();
        Data logits;
        ForwardLogits(tokens, ctx.pastKeyValues, false, logits, true, nullptr,
            last && (o.speculative || RecurrentPrefixCacheEnabled()));
        r.prefilled = end;
        if (last || end - r.lastSavedLength >= size_t(RecurrentPrefixInterval(GetPageLen())))
            SavePrefix(ctx, r, last ? &logits : nullptr);
        if (!last || !o.speculative) return {};
        r.Penalize(*this, ctx, logits, {}, false, true);
        auto stats = r.Summarize(*this, logits, {}, false, false);
        r.bonus = r.Sample(*this, ctx, logits, 0, stats[0].token);
        return {r.bonus};
    }
    if (!o.speculative) {
        if (!r.commit.empty()) {
            Data ignored;
            ForwardLogits(r.commit, ctx.pastKeyValues, false, ignored, false, nullptr, false);
            r.commit.clear();
            return {};
        }
        if (r.block.empty()) {
            r.block.assign(o.width, maskToken);
            r.pending.assign(o.width, true);
        }
        Data logits;
        ForwardLogits(r.block, ctx.pastKeyValues, true, logits);
        r.Penalize(*this, ctx, logits, r.block, true);
        std::vector<int> predicted(o.width);
        std::vector<float> confidence(o.width);
        int best = -1;
        if (c.do_sample) {
            // HF samples every row, including filled ones. Preserve that RNG order.
            logits.ToDevice(DataDevice::CPU);
            for (int i = 0; i < o.width; ++i)
                predicted[i] = CloverSample((float *)logits.cpuData + (size_t)i * vocabSize,
                    vocabSize, c, r.rng, &confidence[i]);
        } else {
            auto stats = r.Summarize(*this, logits, {}, true, false);
            for (int i = 0; i < o.width; ++i) {
                predicted[i] = stats[i].token; confidence[i] = stats[i].confidence;
            }
        }
        for (int i = 0; i < o.width; ++i)
            if (r.pending[i] && (best < 0 || confidence[i] > confidence[best])) best = i;
        bool transferred = false;
        for (int i = 0; i < o.width; ++i) if (r.pending[i] && confidence[i] > o.threshold) {
            r.block[i] = predicted[i]; r.pending[i] = false; transferred = true;
        }
        if (!transferred && best >= 0) { r.block[best] = predicted[best]; r.pending[best] = false; }
        if (std::find(r.pending.begin(), r.pending.end(), true) != r.pending.end()) return {};
        r.commit.swap(r.block);
        return r.commit;
    }
    const int capacity = contextPlan.configured ? contextPlan.effectiveLength : max_positions;
    const int position = ctx.pastKeyValues[kvCacheId].first.dims[0];
    int width = o.width;
    while (width > capacity - position) width /= 2;
    Require(width > 0 && r.bonus >= 0, "no room for the pending token");
    std::vector<int> candidates(width, maskToken);
    candidates[0] = r.bonus;
    Data logits;
    if (width > 1) {
        ForwardLogits(candidates, ctx.pastKeyValues, true, logits);
        r.Penalize(*this, ctx, logits, candidates, true);
        auto stats = r.Summarize(*this, logits, {}, false, false);
        for (int i = 1; i < width; ++i) candidates[i] = stats[i].token;
    }
    ForwardLogits(candidates, ctx.pastKeyValues, false, logits, false, &r.snapshots);
    return r.Accept(*this, ctx, logits, candidates);
#else
    throw std::runtime_error("Clover: block generation requires CUDA");
#endif
}

std::vector<std::vector<int>> CloverModel::StepBatch(const std::vector<ResponseContext *> &contexts,
        const std::vector<std::shared_ptr<Request>> &batch) {
    const int size = contexts.size();
    Require(size >= 2 && size <= CloverCudaGdnCaches::Capacity && batch.size() == contexts.size(), "invalid active request batch");
    const int capacity = contextPlan.configured ? contextPlan.effectiveLength : max_positions;
    std::vector<std::vector<int>> candidates(size), output(size);
    std::vector<std::vector<std::pair<Data, Data>> *> caches;
    std::vector<CloverSnapshots *> snapshots;
    for (int b = 0; b < size; ++b) {
        auto &ctx = *contexts[b]; auto &r = *batch[b];
        int width = r.options.width;
        int position = ctx.pastKeyValues[kvCacheId].first.dims[0];
        while (width > capacity - position) width /= 2;
        Require(width > 0 && r.bonus >= 0, "no room for the pending token");
        candidates[b].assign(width, maskToken); candidates[b][0] = r.bonus;
        caches.push_back(&ctx.pastKeyValues); snapshots.push_back(&r.snapshots);
    }
    for (int b = 1; b < size; ++b)
        if (candidates[b].size() != candidates[0].size()) {
            for (int i = 0; i < size; ++i) output[i] = Step(*contexts[i], *batch[i]);
            return output;
        }
#ifdef USE_CUDA
    const int width = candidates[0].size();
    std::vector<Data> logits;
    if (width > 1) {
        ForwardLogitsBatch(candidates, caches, true, logits);
        for (int b = 0; b < size; ++b) {
            auto &r = *batch[b];
            r.Penalize(*this, *contexts[b], logits[b], candidates[b], true);
            auto stats = r.Summarize(*this, logits[b], {}, false, false);
            for (int i = 1; i < width; ++i) candidates[b][i] = stats[i].token;
        }
    }
    ForwardLogitsBatch(candidates, caches, false, logits, snapshots);
    for (int b = 0; b < size; ++b)
        output[b] = batch[b]->Accept(*this, *contexts[b], logits[b], candidates[b]);
    return output;
#else
    throw std::runtime_error("Clover: batch generation requires CUDA");
#endif
}

void CloverModel::Finish(ResponseContext *ctx) {
    if (ctx->error == ResponseContextErrorNone) {
        std::lock_guard<std::mutex> forward(forwardLocker);
        try { SavePrefix(*ctx, *requests.at(ctx)); }
        catch (const std::exception &e) {
            prefixCache.reset();
            std::cerr << "Clover: could not retain completed prefix: " << e.what() << std::endl;
        }
    }
    ctx->isEnding = true;
    ctx->pastKeyValues.clear();
    requests.erase(ctx);
}

void CloverModel::RunModelSpecificScheduler() {
    std::unique_lock<std::mutex> lock(dictLocker);
    while (!isFree) {
        std::vector<ResponseContext *> contexts;
        std::vector<std::shared_ptr<Request>> batch;
        std::vector<int> handles, aborted;
        for (const auto &entry : responseContextDict.dicts) {
            if (entry.second->isAbort) aborted.push_back(entry.first);
            else if (!entry.second->isEnding && contexts.size() < (size_t)std::max(1, std::min(CloverCudaGdnCaches::Capacity, maxBatch))) {
                handles.push_back(entry.first); contexts.push_back(entry.second);
                batch.push_back(requests.at(entry.second));
            }
        }
        for (int id : aborted) RemoveResponseContext(id);
        if (contexts.empty()) { dictCV.wait(lock); continue; }
        lock.unlock();
        std::vector<std::vector<int>> output(contexts.size());
        std::vector<std::string> errors(contexts.size());
        {
            std::lock_guard<std::mutex> forward(forwardLocker);
            // Prefill stays chunked and independent. Once all requests are
            // ready, their preview/verify MoE and head execute as one batch.
            bool ready = contexts.size() > 1;
            for (size_t b = 0; b < contexts.size(); ++b)
                ready = ready && batch[b]->options.speculative && batch[b]->prefilled == contexts[b]->currentTokens.size();
            if (ready) {
                try { output = StepBatch(contexts, batch); }
                catch (const std::exception &e) { for (auto &error : errors) error = e.what(); }
            } else for (size_t b = 0; b < contexts.size(); ++b) {
                try { output[b] = Step(*contexts[b], *batch[b]); }
                catch (const std::exception &e) { errors[b] = e.what(); }
            }
        }
        lock.lock();
        for (size_t b = 0; b < contexts.size(); ++b) {
            auto *ctx = contexts[b]; auto &request = batch[b];
            ctx->cacheLen = request->restoredTokens;
            // Fetch cannot destroy a live context; Abort only marks it until here.
            if (ctx->isAbort || isFree) { RemoveResponseContext(handles[b]); continue; }
            if (!errors[b].empty()) {
                std::cerr << "Clover request " << handles[b] << " failed: " << errors[b] << std::endl;
                ctx->error = ResponseContextErrorRuntime;
                Finish(ctx);
            } else {
                for (int token : output[b]) {
                    bool eos = !request->options.ignoreEos && (token == eos_token_id || eos_token_ids.count(token));
                    if (eos || ctx->generationConfig.stop_token_ids.count(token)) { Finish(ctx); break; }
                    ctx->resultTokenQueue.push(token);
                    ctx->allTokens.push_back(token);
                    ++ctx->curTokens;
                    ++request->counts[token];
                    if (ctx->curTokens >= request->limit) { Finish(ctx); break; }
                }
            }
        }
        dictCV.notify_all();
    }
}
}
