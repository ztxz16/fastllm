#include "models/naive_n05_flash.h"
#include "models/speculative_sampling.h"
#include "json11.hpp"
#include <algorithm>
#include <cmath>
#ifdef USE_CUDA
#include "devices/cuda/naive-n05-cuda.cuh"
#endif

namespace fastllm {
namespace {
    void DraftLinear(Data &input, Data &weight, Data &output) {
        // The block backbone benefits from tensor-core GEMM even at 2..7 rows.
        // MatMulTransB uses the same BF16 weights and FP32 accumulation, and
        // avoids the generic Linear small-batch GEMV dispatch.
        if (input.Count(0) / input.dims.back() > 1) MatMulTransB(input, weight, output);
        else Linear(input, weight, Data(), output);
    }
}

void NaiveN05FlashModel::InitDraft() {
    draftEnabled = weight.dicts.count("dspark.model_path") != 0;
    if (!draftEnabled) return;
    auto number = [&](const std::string &name) {
        return std::stof(weight.dicts.at("dspark." + name));
    };
    draftLayers = number("num_hidden_layers");
    draftBlock = number("block_size");
    draftHeads = number("num_attention_heads");
    draftKvHeads = number("num_key_value_heads");
    draftHeadDim = number("head_dim");
    draftWindow = number("sliding_window");
    draftEps = number("rms_norm_eps");
    draftTheta = number("rope_parameters.rope_theta");
    std::string error;
    auto layers = json11::Json::parse(weight.dicts.at("dspark.dflash_config.target_layer_ids"), error);
    AssertInFastLLM(error.empty(), "Invalid DSpark target_layer_ids.");
    draftTargetLayers.clear();
    for (auto &layer : layers.array_items()) draftTargetLayers.push_back(layer.int_value());
    AssertInFastLLM(draftLayers > 0 && draftBlock > 1 && draftWindow > 1 && draftKvHeads > 0 &&
                    draftHeads > 0 && draftHeadDim > 0 && draftHeadDim <= 256 &&
                    draftHeads % draftKvHeads == 0 && number("hidden_size") == embed_dim &&
                    number("num_target_layers") == block_cnt &&
                    number("vocab_size") == std::stof(weight.dicts.at("vocab_size")) &&
                    !draftTargetLayers.empty() &&
                    weight.dicts.at("dspark.markov_head_type") == "vanilla" &&
                    weight.dicts.at("dspark.dflash_config.use_mask_embedding") == "true",
                    "Unsupported Naive DSpark configuration.");
    auto layerTypes = json11::Json::parse(weight.dicts.at("dspark.layer_types"), error);
    AssertInFastLLM(error.empty() && (int)layerTypes.array_items().size() == draftLayers &&
                    weight.dicts["dspark.attention_bias"] != "true" &&
                    weight.dicts["dspark.dflash_config.use_target_kv"] != "true" &&
                    weight.dicts["dspark.dflash_config.use_target_kv_inject"] != "true" &&
                    weight.dicts["dspark.dflash_config.use_target_kv_fuse"] != "true",
                    "Unsupported Naive DSpark attention/context mode.");
    for (const auto &type : layerTypes.array_items())
        AssertInFastLLM(type.string_value() == "sliding_attention", "Naive DSpark expects sliding draft layers.");
    for (int layer : draftTargetLayers)
        AssertInFastLLM(layer >= 0 && layer < block_cnt, "DSpark target layer out of range.");
    draftTokens = draftBlock;
    if (const char *value = std::getenv("FASTLLM_DSPARK_TOKENS")) {
        int requested = std::stoi(value);
        AssertInFastLLM(requested > 0 && requested <= draftBlock,
                        "Naive DSpark draft_tokens must be in [1, block_size].");
        draftTokens = requested;
    }
    if (const char *value = std::getenv("FASTLLM_DSPARK_CONFIDENCE_THRESHOLD"))
        draftConfidenceThreshold = std::stof(value);
    AssertInFastLLM(draftConfidenceThreshold >= 0 && draftConfidenceThreshold <= 1,
                    "Invalid DSpark confidence threshold.");
    weight.embeddingNames.insert("dspark.markov_head.markov_w1.weight");
    for (auto name : {"dspark.fc.weight", "dspark.markov_head.markov_w2.weight",
                      "dspark.confidence_head.proj.weight", "dspark.layers.*.self_attn.*_proj.weight",
                      "dspark.layers.*.mlp.*_proj.weight"}) weight.linearNames.insert(name);
}

void NaiveN05FlashModel::AppendDraftContext(Data &hidden, int start, DraftContext &context) {
#ifdef USE_CUDA
    ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
    int length = hidden.dims[1];
    // Only the last window - 1 context positions can be visible to the next block.
    int begin = std::max(0, length - draftWindow + 1);
    Data selected;
    Split(hidden, 1, begin, length, selected);
    length -= begin;
    std::vector<float> positions(length);
    for (int i = 0; i < length; ++i) positions[i] = start + begin + i;
    Data pos(FLOAT32, {1, length}, positions);
    context.kv.resize(draftLayers);
    for (int i = 0; i < draftLayers; ++i) {
        std::string layer = "dspark.layers." + std::to_string(i) + ".self_attn.";
        Data key, value;
        DraftLinear(selected, weight[layer + "k_proj.weight"], key);
        DraftLinear(selected, weight[layer + "v_proj.weight"], value);
        key.Reshape({1, length * draftKvHeads, draftHeadDim});
        KimiK3RMSNorm(key, weight[layer + "k_norm.weight"], draftEps, key);
        key.Reshape({1, length, draftKvHeads * draftHeadDim});
        pos.ToDevice(key.dataDevice, key.dataDeviceIds);
        FastllmCudaNaiveRope(key, pos, draftKvHeads, draftHeadDim, draftHeadDim, draftTheta);
        AppendCache(context.kv[i].first, key);
        AppendCache(context.kv[i].second, value);
        TrimCache(context.kv[i].first, draftWindow - 1);
        TrimCache(context.kv[i].second, draftWindow - 1);
    }
    context.committed = start + begin + length;
#endif
}

void NaiveN05FlashModel::CommitDraftContext(TargetCapture &capture, int tokens,
        DraftContext &context, std::vector<std::pair<Data, Data>> &kv) {
    ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
    Data combined;
    for (int layer : draftTargetLayers) {
        Data selected, joined;
        Split(capture.hidden.at(layer), 1, 0, tokens, selected);
        if (combined.dims.empty()) Copy(selected, combined);
        else { Cat(combined, selected, -1, joined); Copy(joined, combined); }
    }
    Data projected, hidden;
    DraftLinear(combined, weight["dspark.fc.weight"], projected);
    KimiK3RMSNorm(projected, weight["dspark.hidden_norm.weight"], draftEps, hidden);
    if (capture.history) {
        auto &chunk = *capture.history;
        chunk.length = std::min(chunk.length, tokens);
        chunk.bytes = chunk.length * historyBytesPerToken;
        for (auto &pair : chunk.layers) {
            for (Data *tensor : {&pair.first, &pair.second}) {
                if (tensor->dims[1] != chunk.length) {
                    Data compact;
                    CopyHistoryTensor(*tensor, compact, chunk.length);
                    tensor->CopyFrom(compact);
                    tensor->lockInCPU = true;
                }
            }
        }
        CopyHistoryTensor(hidden, chunk.draftHidden, chunk.length);
        FinishHistoryChunk(kv, capture.history);
    }
    AppendDraftContext(hidden, context.committed, context);
}

Data NaiveN05FlashModel::RunDraft(int anchor, DraftContext &context) {
    Data normalized;
#ifdef USE_CUDA
    ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
    Data id(FLOAT32, {1, 1}, {(float)anchor}), hidden;
    Embedding(id, weight["model.embed_tokens.weight"], hidden);
    ToDataType(hidden, BFLOAT16);
    Data &mask = weight["dspark.mask_embedding"];
    AssertInFastLLM(mask.Count(0) == embed_dim, "Invalid DSpark mask embedding.");
    mask.Reshape({1, 1, embed_dim});
    for (int i = 1; i < draftBlock; ++i) {
        Data joined;
        Cat(hidden, mask, 1, joined);
        Copy(joined, hidden);
    }
    std::vector<float> positions(draftBlock);
    for (int i = 0; i < draftBlock; ++i) positions[i] = context.committed + i;
    Data pos(FLOAT32, {1, draftBlock}, positions);
    for (int i = 0; i < draftLayers; ++i) {
        const std::string layer = "dspark.layers." + std::to_string(i) + ".";
        Data normed, q, k, v, attention, output, gate, up;
        KimiK3RMSNorm(hidden, weight[layer + "input_layernorm.weight"], draftEps, normed);
        DraftLinear(normed, weight[layer + "self_attn.q_proj.weight"], q);
        DraftLinear(normed, weight[layer + "self_attn.k_proj.weight"], k);
        DraftLinear(normed, weight[layer + "self_attn.v_proj.weight"], v);
        q.Reshape({1, draftBlock * draftHeads, draftHeadDim});
        k.Reshape({1, draftBlock * draftKvHeads, draftHeadDim});
        KimiK3RMSNorm(q, weight[layer + "self_attn.q_norm.weight"], draftEps, q);
        KimiK3RMSNorm(k, weight[layer + "self_attn.k_norm.weight"], draftEps, k);
        q.Reshape({1, draftBlock, draftHeads * draftHeadDim});
        k.Reshape({1, draftBlock, draftKvHeads * draftHeadDim});
        pos.ToDevice(q.dataDevice, q.dataDeviceIds);
        FastllmCudaNaiveRope(q, pos, draftHeads, draftHeadDim, draftHeadDim, draftTheta);
        FastllmCudaNaiveRope(k, pos, draftKvHeads, draftHeadDim, draftHeadDim, draftTheta);
        auto &cache = context.kv[i];
        int past = cache.first.dims[1];
        AppendCache(cache.first, k);
        AppendCache(cache.second, v);
        FastllmCudaNaiveAttention(q, cache.first, cache.second, Data(), Data(),
            draftHeads, draftKvHeads, draftHeadDim, draftHeadDim, past, draftWindow, attention, false);
        cache.first.Resize({1, past, draftKvHeads * draftHeadDim});
        cache.second.Resize({1, past, draftKvHeads * draftHeadDim});
        DraftLinear(attention, weight[layer + "self_attn.o_proj.weight"], output);
        AddTo(hidden, output);
        KimiK3RMSNorm(hidden, weight[layer + "post_attention_layernorm.weight"], draftEps, normed);
        DraftLinear(normed, weight[layer + "mlp.gate_proj.weight"], gate);
        DraftLinear(normed, weight[layer + "mlp.up_proj.weight"], up);
        Silu(gate, gate);
        MulTo(gate, up);
        DraftLinear(gate, weight[layer + "mlp.down_proj.weight"], output);
        AddTo(hidden, output);
    }
    KimiK3RMSNorm(hidden, weight["dspark.norm.weight"], draftEps, normalized);
#endif
    return normalized;
}

int NaiveN05FlashModel::ForwardDraft(
        const Data &inputIds, const Data &positions, std::vector<std::pair<Data, Data>> &kv,
        const GenerationConfig &config, const LastTokensManager &lastTokens,
        std::vector<float> *retLogits) {
    std::shared_ptr<DraftContext> owner;
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        auto &entry = draftContexts[&kv];
        if (!entry) entry = std::make_shared<DraftContext>();
        owner = entry;
    }
    auto &context = *owner;
    Data ids;
    ids.CopyFrom(inputIds);
    ids.ToDevice(DataDevice::CPU);
    const int anchor = (int)((float *)ids.cpuData)[0];
    if (!context.pending.empty()) {
        AssertInFastLLM(inputIds.dims[1] == 1 && anchor == context.pending.front().first,
                        "Naive speculative output queue is out of sync.");
        int token = context.pending.front().second;
        context.pending.pop_front();
        return token;
    }
    if (!context.restoredHidden.dims.empty()) {
        int length = context.restoredHidden.dims[1];
        AppendDraftContext(context.restoredHidden, context.committed - length, context);
        context.restoredHidden = Data();
    }
    int oldLength = kv[0].first.dims.empty() ? 0 : kv[0].first.dims[1];
    AssertInFastLLM(context.committed == oldLength, "Naive target/draft cache length mismatch.");
    // Tool constraints and content sampling change after each emitted prefix.
    // Keep these requests single-token until branch-local masks are available.
    bool constrained = config.tool_call_name_constraint_enabled ||
                       config.tool_call_parameter_name_constraint_enabled ||
                       config.tool_call_content_sampling_enabled;
    int limit = std::min(draftTokens, max_positions - oldLength - 1);
    if (config.output_token_limit > 0)
        limit = std::min(limit, config.output_token_limit - (oldLength - config.input_token_length) - 1);
    if (context.kv.empty() || inputIds.dims[1] != 1 || config.output_logits || constrained || limit <= 0) {
        TargetCapture capture;
        Data logits = RunTarget(inputIds, positions, kv, &capture);
        CommitDraftContext(capture, inputIds.dims[1], context, kv);
        if (isIntermediateChunkedPrefill) return 0;
        return SampleTarget(logits, kv, config, lastTokens, retLogits);
    }
    Data draftHidden = RunDraft(anchor, context), selected, baseLogits;
    // DSpark predicts after the anchor at slot 0, unlike DFlash's masked-slot-only head.
    Split(draftHidden, 1, 0, limit, selected);
    DraftLinear(selected, weight["lm_head.weight"], baseLogits);
    const int vocab = baseLogits.dims.back();
    const bool greedy = config.IsSimpleGreedy() && config.output_token_least <= 0;
    LastTokensUnit samplingTokens = lastTokens.units.empty() ? LastTokensUnit(config.last_n) : lastTokens.units[0];
    auto distribution = [&](Data &logits, int row, int position, const LastTokensUnit &history) {
        logits.ToDevice(DataDevice::CPU);
        float *values = (float *)logits.cpuData + (size_t)row * vocab;
        if (config.output_token_least > position - config.input_token_length) {
            if (eos_token_id >= 0 && eos_token_id < vocab) values[eos_token_id] = -1e30f;
            for (int id : eos_token_ids) if (id >= 0 && id < vocab) values[id] = -1e30f;
            for (int id : config.stop_token_ids) if (id >= 0 && id < vocab) values[id] = -1e30f;
        }
        return SpeculativeDistribution(values, vocab, config, history);
    };
    std::vector<int> proposed;
    std::vector<std::vector<float>> q;
    int previous = anchor;
    for (int step = 0; step < limit; ++step) {
        Data previousId(FLOAT32, {1, 1}, {(float)previous}), latent, bias, logits;
        Embedding(previousId, weight["dspark.markov_head.markov_w1.weight"], latent);
        ToDataType(latent, BFLOAT16);
        if (step > 0 && draftConfidenceThreshold > 0) {
            Data h, features, confidence;
            Split(draftHidden, 1, step, step + 1, h);
            Cat(h, latent, -1, features);
            ToDataType(features, FLOAT32);
            Linear(features, weight["dspark.confidence_head.proj.weight"],
                   weight["dspark.confidence_head.proj.bias"], confidence);
            confidence.ToDevice(DataDevice::CPU);
            float probability = 1.0f / (1.0f + std::exp(-((float *)confidence.cpuData)[0]));
            if (probability < draftConfidenceThreshold) break;
        }
        DraftLinear(latent, weight["dspark.markov_head.markov_w2.weight"], bias);
        Split(baseLogits, 1, step, step + 1, logits);
        AddTo(logits, bias);
        ToDataType(logits, FLOAT32);
        if (greedy) {
            Data top;
            TopK(logits, top, 1);
            top.ToDevice(DataDevice::CPU);
            previous = (int)((float *)top.cpuData)[0];
        } else {
            q.push_back(distribution(logits, 0, oldLength + step + 1, samplingTokens));
            previous = SampleSpeculativeDistribution(q.back(), context.Uniform());
        }
        proposed.push_back(previous);
        samplingTokens.Push(previous);
    }
    std::vector<float> verifyIds{(float)anchor}, verifyPositions;
    for (int token : proposed) verifyIds.push_back((float)token);
    for (int i = 0; i < (int)verifyIds.size(); ++i) verifyPositions.push_back(oldLength + i);
    Data verifyInput(FLOAT32, {1, (int)verifyIds.size()}, verifyIds);
    Data verifyPos(FLOAT32, {1, (int)verifyIds.size()}, verifyPositions);
    TargetCapture capture;
    capture.verifying = true;
    Data logits = RunTarget(verifyInput, verifyPos, kv, &capture);
    samplingTokens = lastTokens.units.empty() ? LastTokensUnit(config.last_n) : lastTokens.units[0];
    int accepted = 0, next = -1;
    if (greedy) {
        // For point-mass p and q, standard rejection sampling reduces exactly
        // to matching argmax tokens and emitting the target argmax on rejection.
        Data top;
        TopK(logits, top, 1);
        top.ToDevice(DataDevice::CPU);
        const float *values = (float *)top.cpuData;
        while (accepted < (int)proposed.size() && proposed[accepted] == (int)values[accepted * 2])
            ++accepted;
        next = (int)values[accepted * 2];
    } else {
        while (accepted < (int)proposed.size()) {
            auto p = distribution(logits, accepted, oldLength + accepted + 1, samplingTokens);
            if (!AcceptSpeculativeToken(proposed[accepted], p, q[accepted], context.Uniform())) {
                next = SampleSpeculativeResidual(p, q[accepted], context.Uniform());
                break;
            }
            samplingTokens.Push(proposed[accepted++]);
        }
        if (next < 0) {
            auto p = distribution(logits, accepted, oldLength + accepted + 1, samplingTokens);
            next = SampleSpeculativeDistribution(p, context.Uniform());
        }
    }
    int committed = accepted + 1;
    for (int i = 0; i < block_cnt; ++i) {
        int length = (slidingLayers[i] ? std::min(oldLength, window - 1) : oldLength) + committed;
        auto &cache = kv[i];
        cache.first.Resize({1, length, cache.first.dims[2]});
        cache.second.Resize({1, length, cache.second.dims[2]});
        if (slidingLayers[i]) { TrimCache(cache.first, window - 1); TrimCache(cache.second, window - 1); }
    }
    CommitDraftContext(capture, committed, context, kv);
    ++context.rounds;
    context.proposed += proposed.size();
    context.accepted += accepted;
    proposed.resize(accepted);
    proposed.push_back(next);
    for (int i = 1; i < (int)proposed.size(); ++i)
        context.pending.emplace_back(proposed[i - 1], proposed[i]);
    return proposed[0];
}

}
