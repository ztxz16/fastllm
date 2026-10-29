#include "clover.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-clover.h"
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
};

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
    auto run = [&](const std::vector<int> &tokens, bool preview, Data &logits,
                   bool lastOnly = false, CloverSnapshots *snapshots = nullptr, bool head = true) {
        ForwardLogits(tokens, ctx.pastKeyValues, preview, logits, lastOnly, snapshots, head);
    };
    auto upload = [&](Data &data, DataType type, const std::vector<int> &dims, const void *source) {
        data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
        data.ToDevice(DataDevice::CUDA, std::vector<int>{cudaDevice}, false);
        data.Allocate(false);
        if (source) FastllmCudaCopyFromHostToDevice(data.cudaData, (void *)source, data.GetBytes());
    };
    auto penalize = [&](Data &logits, const std::vector<int> &visible, bool preview, bool initial = false) {
        if (r.countsVersion != ctx.curTokens) {
            upload(r.countsGpu, INT32, {vocabSize}, r.counts.data());
            r.countsVersion = ctx.curTokens;
        }
        if (r.stopsGpu.dims.empty() && !r.stops.empty())
            upload(r.stopsGpu, INT32, {int(r.stops.size())}, r.stops.data());
        int rows = logits.Count(0) / vocabSize;
        auto bias = CloverRowBias(rows, vocabSize, r.counts, ctx.curTokens, visible, o, preview, initial);
        upload(r.biasGpu, FLOAT32, {rows, 2}, bias.data());
        CloverCudaPenalty p{c.repeat_penalty, o.presence, o.frequency, o.speculative ? -1 : maskToken,
            o.thinkToken, ctx.curTokens + (o.speculative && preview ? 0 : 1), c.output_token_least};
        if (!FastllmCudaCloverPenalize((float *)logits.cudaData, (int *)r.countsGpu.cudaData,
            (int *)r.stopsGpu.cudaData, r.stops.size(), (float *)r.biasGpu.cudaData, rows, vocabSize, p))
            throw std::runtime_error("Clover: GPU penalties failed");
    };
    auto summarize = [&](Data &logits, const std::vector<int> &candidates, bool confidence, bool entropy) {
        int rows = logits.Count(0) / vocabSize;
        if (!candidates.empty()) upload(r.candidatesGpu, INT32, {rows}, candidates.data());
        static_assert(sizeof(CloverCudaTokenStats) == 5 * sizeof(int), "GPU token statistics layout");
        upload(r.statsGpu, INT32, {rows, 5}, nullptr);
        size_t bytes = FastllmCudaCloverTokenStatsWorkspaceBytes(rows, vocabSize);
        if (bytes) upload(r.statsWorkspace, INT32, {int((bytes + 3) / 4)}, nullptr);
        if (!FastllmCudaCloverTokenStats((float *)logits.cudaData,
            candidates.empty() ? nullptr : (int *)r.candidatesGpu.cudaData,
            (CloverCudaTokenStats *)r.statsGpu.cudaData, rows, vocabSize, confidence, entropy,
            nullptr, bytes ? r.statsWorkspace.cudaData : nullptr))
            throw std::runtime_error("Clover: GPU sampling statistics failed");
        std::vector<CloverCudaTokenStats> result(rows);
        FastllmCudaCopyFromDeviceToHost(result.data(), r.statsGpu.cudaData, result.size() * sizeof(result[0]));
        for (const auto &x : result) if (x.error) throw std::runtime_error("Clover: invalid sampling logits");
        return result;
    };
    auto sample = [&](Data &logits, int row, int greedyToken) {
        if (!c.do_sample) return greedyToken;
        // The stochastic bonus needs one row only; draft/acceptance remain on GPU.
        std::vector<float> scores(vocabSize);
        FastllmCudaCopyFromDeviceToHost(scores.data(), (float *)logits.cudaData + (size_t)row * vocabSize,
            scores.size() * sizeof(float));
        return CloverSample(scores.data(), vocabSize, c, r.rng);
    };
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
        run(tokens, false, logits, true, nullptr, last && o.speculative);
        r.prefilled = end;
        if (!last || !o.speculative) return {};
        penalize(logits, {}, false, true);
        auto stats = summarize(logits, {}, false, false);
        r.bonus = sample(logits, 0, stats[0].token);
        return {r.bonus};
    }
    if (!o.speculative) {
        if (!r.commit.empty()) {
            Data ignored;
            run(r.commit, false, ignored, false, nullptr, false);
            r.commit.clear();
            return {};
        }
        if (r.block.empty()) {
            r.block.assign(o.width, maskToken);
            r.pending.assign(o.width, true);
        }
        Data logits;
        run(r.block, true, logits);
        penalize(logits, r.block, true);
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
            auto stats = summarize(logits, {}, true, false);
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
        run(candidates, true, logits);
        penalize(logits, candidates, true);
        auto stats = summarize(logits, {}, false, false);
        for (int i = 1; i < width; ++i) candidates[i] = stats[i].token;
    }
    run(candidates, false, logits, false, &r.snapshots);
    penalize(logits, candidates, false);
    auto stats = summarize(logits, candidates, !o.exact, !o.exact);
    int accepted = 0;
    while (accepted + 1 < width) {
        const auto &s = stats[accepted];
        bool accept = o.exact ? candidates[accepted + 1] == s.token :
            s.probability > 0.f && s.probability >= std::min(o.acceptRatio, o.acceptDelta * std::exp(-s.entropy));
        if (!accept) break;
        ++accepted;
    }
    r.bonus = sample(logits, accepted, stats[accepted].token);
    RestorePrefix(ctx.pastKeyValues, r.snapshots, accepted + 1);
    std::vector<int> output(candidates.begin() + 1, candidates.begin() + accepted + 1);
    output.push_back(r.bonus);
    return output;
#else
    throw std::runtime_error("Clover: block generation requires CUDA");
#endif
}

void CloverModel::Finish(ResponseContext *ctx) {
    ctx->isEnding = true;
    ctx->pastKeyValues.clear();
    requests.erase(ctx);
}

void CloverModel::RunModelSpecificScheduler() {
    std::unique_lock<std::mutex> lock(dictLocker);
    while (!isFree) {
        ResponseContext *ctx = nullptr;
        int handle = -1;
        std::vector<int> aborted;
        for (const auto &entry : responseContextDict.dicts) {
            if (entry.second->isAbort) aborted.push_back(entry.first);
            else if (!ctx && !entry.second->isEnding) { handle = entry.first; ctx = entry.second; }
        }
        for (int id : aborted) RemoveResponseContext(id);
        if (!ctx) { dictCV.wait(lock); continue; }
        auto request = requests.at(ctx);
        lock.unlock();
        std::vector<int> output;
        std::string error;
        try {
            std::lock_guard<std::mutex> forward(forwardLocker);
            output = Step(*ctx, *request);
        } catch (const std::exception &e) { error = e.what(); }
        lock.lock();
        // Fetch cannot destroy a live context; Abort only marks it until here.
        if (ctx->isAbort || isFree) { RemoveResponseContext(handle); continue; }
        if (!error.empty()) {
            std::cerr << "Clover request " << handle << " failed: " << error << std::endl;
            ctx->error = ResponseContextErrorRuntime;
            Finish(ctx);
        } else {
            for (int token : output) {
                bool eos = !request->options.ignoreEos && (token == eos_token_id || eos_token_ids.count(token));
                if (eos || ctx->generationConfig.stop_token_ids.count(token)) { Finish(ctx); break; }
                ctx->resultTokenQueue.push(token);
                ctx->allTokens.push_back(token);
                ++ctx->curTokens;
                ++request->counts[token];
                if (ctx->curTokens >= request->limit) { Finish(ctx); break; }
            }
        }
        dictCV.notify_all();
    }
}
}
