#include "models/sensenova_loop.h"
#include "baseblock.h"
#if defined(USE_CUDA) && !defined(USE_ROCM)
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#endif

namespace fastllm {
#if defined(USE_CUDA) && !defined(USE_ROCM)
void SenseNovaBf16GateProjection(const Data &, Data &, Data &);
void SenseNovaVerifyRecurrent(const Data &, const Data &, const Data &, Data &, Data &, Data &,
    Data &, Data &, const std::vector<Data*> &, const std::vector<Data*> &, int, int, float, std::vector<std::pair<Data,Data>> &, Data &, bool, bool);
void SenseNovaBatchRecurrent(const Data &, const Data &, Data &, Data &, Data &, Data &,
    const std::vector<Data*> &, const std::vector<Data*> &, int, int, Data &);
void SenseNovaJoinPrefillRows(const std::vector<Data> &, int, Data &);
void SenseNovaCopyState(const Data &, Data &);
void SenseNovaApplyRotary(Data &, const Data &);
void SenseNovaGateUp(const Data &, Data &);
void SenseNovaConvDecode(const Data &, const Data &, Data &, Data &, Data &);
void SenseNovaGdnOutput(const Data &, const Data &, Data &, float, Data &);
void SenseNovaRouterNormalize(const Data &, Data &);
void SenseNovaRouterLogits(const Data &, Data &, Data &, Data &);
void SenseNovaRoute(const Data &, const Data &, int, Data &, Data &);
void SenseNovaMerge(const Data &, const Data &, const Data &, Data &);
#endif

static bool SenseNovaMtpSupportsSampling(const GenerationConfig &config) {
    if (config.NeedsTokenHistory() || config.output_logits || config.output_token_least > 0 ||
        config.tool_call_grammar || config.tool_call_token_mask || !config.tool_call_allowed_token_ids.empty() ||
        config.tool_call_name_constraint_enabled || config.tool_call_parameter_name_constraint_enabled ||
        config.tool_call_content_sampling_enabled) return false;
    return config.IsSimpleGreedy() || (config.top_k > 1 && std::isfinite(config.temperature) &&
        config.temperature > 0 && config.top_p > 0 && config.top_p <= 1);
}

SenseNovaLoopModel::SenseNovaLoopModel() {
    model_type = model_struct = "sensenova_loop";
    SetDataType(BFLOAT16);
}

void SenseNovaLoopModel::InitParams() {
    Qwen3_5Model::InitParams();
    physicalLayers = block_cnt;
    auto number = [&](const std::string &key) { return std::stoi(weight.dicts.at(key)); };
    loopStart = number("loop_settings.start");
    loopEnd = number("loop_settings.end");
    int passes = number("loop_settings.eval_count");
    iterations = number("mhc.iterations");
    convKernel = number("text_config.linear_conv_kernel_dim");
    AssertInFastLLM(loopStart >= 0 && loopStart < loopEnd && loopEnd <= physicalLayers &&
        passes > 0 && number("mhc.start") == loopStart && number("mhc.end") == loopEnd &&
        number("mhc.streams") == 2 && iterations > 0 && weight.dicts.at("mhc.fixed_identity") == "false" &&
        std::stof(weight.dicts.at("mhc.pre_scale")) == 1.0f, "Unsupported SenseNova Loop routing plan.");
    visits.clear();
    for (int i = 0; i < loopStart; ++i) visits.push_back(i);
    for (int pass = 0; pass < passes; ++pass)
        for (int i = loopStart; i < loopEnd; ++i) visits.push_back(i);
    for (int i = loopEnd; i < physicalLayers; ++i) visits.push_back(i);
    block_cnt = visits.size(); // The scheduler allocates one cache pair per visit.
    std::vector<float> basis(rotary_dim, 0.0f);
    std::fill_n(basis.begin(), rotary_dim / 2, 1.0f);
    rotaryBasis.CopyFrom(Data(BFLOAT16, {1, 1, 1, rotary_dim}, basis));
    headNorm.CopyFrom(Data(FLOAT32, {head_v_dim}, std::vector<float>(head_v_dim, 1.0f)));
    routerNorm.CopyFrom(Data(FLOAT32, {2 * embed_dim}, std::vector<float>(2 * embed_dim, 1.0f)));
}

void SenseNovaLoopModel::OnModelWeightsLoaded() {
    Qwen3_5Model::OnModelWeightsLoaded();
    // Qwen3.5 stores zero-centered scales only for the backbone RMSNorms.
    auto addOffset = [&](const std::string &name) {
        Data &scale = weight.weight.at(name);
        scale.ToDevice(CPU);
        ToDataType(scale, FLOAT32);
        scale.ToDevice(CPU);
        float *values = (float *)scale.cpuData;
        for (uint64_t j = 0; j < scale.Count(0); ++j) values[j] += 1.0f;
    };
    addOffset(language_prefix + "norm.weight");
    for (int i = 0; i < physicalLayers; ++i) {
        std::string layer = language_prefix + "layers." + std::to_string(i) + ".";
        addOffset(layer + "input_layernorm.weight");
        addOffset(layer + "post_attention_layernorm.weight");
        if (weight.weight.count(layer + "self_attn.q_norm.weight")) {
            addOffset(layer + "self_attn.q_norm.weight");
            addOffset(layer + "self_attn.k_norm.weight");
        } else {
            // Transformers casts these checkpoint FP32 parameters to model dtype,
            // then promotes them for the gate and decay calculation.
            for (const char *name : {"linear_attn.dt_bias", "linear_attn.A_log"}) {
                Data &parameter = weight.weight.at(layer + name);
                ToDataType(parameter, BFLOAT16);
                ToDataType(parameter, FLOAT32);
            }
        }
    }
    draftCount = Qwen35ConfiguredMtpDrafts();
    AssertInFastLLM(draftCount >= 0 && draftCount <= 2,
                    "SenseNova Loop accepts zero, one or two MTP draft tokens.");
    if (draftCount > 0) {
        AssertInFastLLM(HasMtpWeights(), "SenseNova Loop MTP requires checkpoint MTP weights.");
        AddMtpRmsNormOffset();
    }
#if defined(USE_CUDA) && !defined(USE_ROCM)
    // Per-row FP8 projections can share BF16 weight staging across prompts.
    const std::string layers = language_prefix + "layers.";
    bool sawFp8 = false, allRowScaledFp8 = true;
    nvfp4PrefillPlan.clear();
    for (const auto &entry : weight.weight) {
        const auto &name = entry.first;
        const auto &w = entry.second;
        if (name.compare(0, layers.size(), layers) == 0 && name.find("proj") != std::string::npos &&
            name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0) {
            sawFp8 |= w.dataType == FP8_E4M3;
            allRowScaledFp8 &= w.dataType == FP8_E4M3 && w.dims.size() == 2 &&
                               w.blockM == w.dims[1] && w.blockK == 1;
        }
        FastllmCudaPlanNvfp4Prefill(nvfp4PrefillPlan, w);
    }
    groupFp8Prefill = !draftCount && sawFp8 && allRowScaledFp8;
#ifndef CUDA_NO_TENSOR_CORE
    nvfp4DecodeCache = !draftCount && !nvfp4PrefillPlan.empty() && FastllmCudaNvfp4MarlinEnabled(false)
        ? FastllmCreateCompactMarlinCache() : nullptr;
#endif
#endif
    initialized_add1 = true;
}

long long SenseNovaLoopModel::GetAutoWarmupCudaRuntimeReserveBytes(int device, int batch) const {
    const long long base = Qwen3_5Model::GetAutoWarmupCudaRuntimeReserveBytes(device, batch);
    if (!draftCount || !GetAutoWarmupCudaAdditionalCacheBytesPerToken(device)) return base;
    // The Qwen estimate uses model precision and physical layer count. The
    // Loop verifier retains FP32 states for *visits* in one shared workspace.
    const long long channels = 2LL * num_k_heads * head_k_dim + 1LL * num_v_heads * head_v_dim;
    const long long elements = channels * convKernel + 1LL * num_v_heads * head_k_dim * head_v_dim;
    long long extra = 0;
    for (int layer : visits) {
        const auto prefix = language_prefix + "layers." + std::to_string(layer) + ".";
        if (weight.weight.count(prefix + "linear_attn.conv1d.weight"))
            extra += draftCount * elements * sizeof(float);
    }
    return base + extra * std::max(1, std::min(batch, 8));
}

void SenseNovaLoopModel::OnResponseContextCreated(ResponseContext *context) {
    if (!context) return;
    for (size_t slot = 0; slot < std::min(visits.size(), context->pastKeyValues.size()); ++slot) {
        const auto prefix = language_prefix + "layers." + std::to_string(visits[slot]) + ".";
        bool recurrent = weight.weight.count(prefix + "linear_attn.conv1d.weight") != 0;
        context->pastKeyValues[slot].first.isLinearAttention = recurrent;
        context->pastKeyValues[slot].second.isLinearAttention = recurrent;
    }
    const auto &config = context->generationConfig;
    // Latch the switch at admission; a subsequent change cannot invalidate
    // the caches or pending tokens of an in-flight chain.
    const int depth = std::min(draftCount, Qwen35ConfiguredMtpDrafts());
    if (depth > 0 && !context->pastKeyValues.empty() && !saveHistoryChat &&
        !context->cacheLen && context->multimodalInput.empty() && SenseNovaMtpSupportsSampling(config)) {
        auto request = std::make_shared<DraftRequest>();
        request->kv.proposalDepth = depth;
        request->promptLength = context->currentTokens.size();
        std::lock_guard<std::mutex> guard(requestMutex);
        requests[&context->pastKeyValues[0].first] = request;
    }
}

void SenseNovaLoopModel::OnResponseContextRemoved(ResponseContext *context) {
    if (context && !context->pastKeyValues.empty() && draftCount) {
        std::lock_guard<std::mutex> guard(requestMutex);
        auto it = requests.find(&context->pastKeyValues.front().first);
        if (it != requests.end()) {
            auto &r = *it->second;
            if (Qwen35MtpProfileInterval() > 0)
                fprintf(stderr, "[SenseNova MTP] prompt=%d emitted=%d rounds=%d accepted=%d proposed=%d"
                                " accepted_by_position=%d,%d\n", r.promptLength, r.emitted, r.rounds,
                    r.accepted, r.proposed, r.acceptedFirst, r.accepted - r.acceptedFirst);
            requests.erase(it);
        }
    }
    Qwen3_5Model::OnResponseContextRemoved(context);
}

bool SenseNovaLoopModel::PopPendingResponseToken(ResponseContext *context, int, int &token) {
    if (!draftCount || !context || context->pastKeyValues.empty()) return false;
    std::lock_guard<std::mutex> guard(requestMutex);
    auto it = requests.find(&context->pastKeyValues.front().first);
    if (it == requests.end() || it->second->pending.empty()) return false;
    token = it->second->pending.front();
    it->second->pending.pop_front();
    return true;
}

PagedCacheManager *SenseNovaLoopModel::GetPagedKVCacheManager(int slot, bool key) const {
    return basellm::GetPagedKVCacheManager(slot, key);
}
std::vector<std::pair<int, PagedCacheManager*>> SenseNovaLoopModel::GetPagedKVCacheManagers(int slot, bool key) const {
    return basellm::GetPagedKVCacheManagers(slot, key);
}

std::vector<int> SenseNovaLoopModel::ForwardV2(int batch, const Data &ids, const std::vector<Data*> &masks,
        const std::vector<Data*> &positions, const std::vector<int> &lengths,
        std::vector<std::pair<Data*, Data*>> &cache, const std::vector<GenerationConfig> &configs,
        const LastTokensManager &last, std::vector<std::vector<float>*> *logits) {
#if !defined(USE_CUDA) || defined(USE_ROCM)
    throw std::runtime_error("SenseNova Loop requires a CUDA build.");
#else
    AssertInFastLLM(dataType == BFLOAT16, "SenseNova Loop requires BF16 activations.");
    AssertInFastLLM(cache.size() == (size_t)batch * visits.size(), "Invalid SenseNova Loop cache layout.");
    if (!draftCount && batch > 1 && convKernel == 4 && head_k_dim == 128 && head_v_dim == 128 &&
        std::all_of(lengths.begin(), lengths.end(), [](int n) { return n == 1; }) &&
        std::all_of(cache.begin(), cache.end(), [](const auto &pair) {
            return pair.first && pair.second && !pair.first->dims.empty() && !pair.second->dims.empty();
        })) {
        return RunTarget(ids, positions, lengths, cache, configs, last, logits, nullptr, nullptr);
    }
    // Process already-admitted prompts together. Each request owns its KV
    // pages and GDN state; projections share one unpack of each weight panel.
    if (batch > 1 && batch <= 8 && (nvfp4DecodeCache || (groupFp8Prefill && lengths.front() >= 128)) &&
        lengths.front() >= 32 && ids.dims[1] <= 4096 &&
        std::all_of(lengths.begin(), lengths.end(), [&](int n) { return n == lengths.front(); })) {
        return RunTarget(ids, positions, lengths, cache, configs, last, logits, nullptr, nullptr);
    }
    if (draftCount && batch <= 8) {
        std::vector<std::shared_ptr<DraftRequest>> active;
        {
            std::lock_guard<std::mutex> guard(requestMutex);
            for (int b = 0; b < batch; ++b) {
                auto it = requests.find(cache[b * visits.size()].first);
                if (it == requests.end()) break;
                if ((!it->second->emitted && lengths[b] != it->second->promptLength) ||
                    !SenseNovaMtpSupportsSampling(configs[b])) {
                    requests.erase(it);
                    break;
                }
                active.push_back(it->second);
            }
        }
        if ((int)active.size() == batch && HasMtpBatchPages(active, lengths, cache)) {
            FastllmCudaNvfp4Scope scope(nvfp4PrefillPlan);
            return DraftForwardBatch(active, ids, positions, lengths, cache, configs, last);
        }
    }
    if (batch > 1) {
        std::vector<int> outputs;
        int offset = 0;
        for (int b = 0; b < batch; ++b) {
            Data slice;
            Split(ids, 1, offset, offset + lengths[b], slice);
            offset += lengths[b];
            std::vector<std::pair<Data*, Data*>> row(cache.begin() + b * block_cnt, cache.begin() + (b + 1) * block_cnt);
            LastTokensManager history(1, configs[b].last_n);
            if (b < (int)last.units.size()) history.units[0] = last.units[b];
            std::vector<std::vector<float>*> outputLogits{logits ? logits->at(b) : nullptr};
            outputs.push_back(ForwardV2(1, slice, {b < (int)masks.size() ? masks[b] : nullptr}, {positions[b]}, {lengths[b]}, row,
                                        {configs[b]}, history, &outputLogits)[0]);
        }
        return outputs;
    }
    AssertInFastLLM(!positions.empty() && positions[0], "SenseNova Loop needs text position IDs.");
    return RunTarget(ids, positions, lengths, cache, configs, last, logits, nullptr, nullptr);
#endif
}

#if defined(USE_CUDA) && !defined(USE_ROCM)
void SenseNovaLoopModel::FullAttention(const std::string &p, Data &input, const Data &rotary,
        int slot, std::vector<Data*> &keys, std::vector<Data*> &values, DecodePages &pages, bool verify, Data &out) {
    Data &pastKey = *keys.front(), &pastValue = *values.front();
    int length = input.dims[1], qdim = num_attention_heads * head_dim, kvdim = num_key_value_heads * head_dim;
    Data packed, qgate, k, v;
    if (weight.weight.count(p + "mergeqkv.weight")) {
        Linear(input, weight[p + "mergeqkv.weight"], Data(), packed);
        if (length == 1) {
            qgate.FakeFrom(packed, 0);
            qgate.Resize({1, 1, 2 * qdim});
            k.FakeFrom(packed, (size_t)2 * qdim * packed.unitSize);
            k.Resize({1, 1, kvdim});
            v.FakeFrom(packed, (size_t)(2 * qdim + kvdim) * packed.unitSize);
            v.Resize({1, 1, kvdim});
        } else {
            Split(packed, -1, 0, 2 * qdim, qgate);
            Split(packed, -1, 2 * qdim, 2 * qdim + kvdim, k);
            Split(packed, -1, 2 * qdim + kvdim, 2 * qdim + 2 * kvdim, v);
        }
    } else {
        Linear(input, weight.weight.at(p + "q_proj.weight"), Data(), qgate);
        Linear(input, weight.weight.at(p + "k_proj.weight"), Data(), k);
        Linear(input, weight.weight.at(p + "v_proj.weight"), Data(), v);
    }
    Data attn;
    Data q, gate;
    qgate.Reshape({1, length, num_attention_heads, head_dim * 2});
    Split(qgate, -1, 0, head_dim, q);
    Split(qgate, -1, head_dim, 2 * head_dim, gate);
    gate.Reshape({1, length, qdim});
    k.Reshape({1, length, num_key_value_heads, head_dim});
    v.Reshape(k.dims);
    RMSNorm(q, weight[p + "q_norm.weight"], rms_norm_eps, q);
    RMSNorm(k, weight[p + "k_norm.weight"], rms_norm_eps, k);
    SenseNovaApplyRotary(q, rotary);
    SenseNovaApplyRotary(k, rotary);
    const int group = num_attention_heads / num_key_value_heads;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    auto attendOne = [&](Data &q, Data &k, Data &v, int length, Data &pastKey, Data &pastValue, Data &attn) {
        for (Data *t : {&q, &k, &v}) {
            if (length > 1) PermuteSelf(*t, {0, 2, 1, 3});
            t->Reshape({-1, length, head_dim});
        }
        auto pool = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
        AppendPagedCache(*AllocatePagedCacheManager(2 * slot, pool, k), pastKey, k);
        AppendPagedCache(*AllocatePagedCacheManager(2 * slot + 1, pool, v), pastValue, v);
        if (length == 1 || verify) {
            // Each visit has its own KV pool. Reuse metadata only when its physical
            // page indices and tail length match this request's previous visit.
            if (pages.last != pastKey.lastPageLen || pages.pages != pastKey.pageIndex) {
                GeneratePagedBatchParams(q, {&pastKey}, 1, pages.qSizes, pages.pageSizes,
                                         pages.indices, pages.lastLens, {length});
                pages.pages = pastKey.pageIndex;
                pages.last = pastKey.lastPageLen;
            }
            AttentionPagedBatch(q, pastKey, pastValue, pages.qSizes, pages.pageSizes,
                                pages.indices, pages.lastLens, attn, group, scale, 1, false, false);
        } else {
            AttentionPaged(q, pastKey, pastValue, attn, group, scale, 1, false);
        }
        // AttentionPagedBatch already returns token-major output for the causal query block.
        if (length > 1 && !verify) PermuteSelf(attn, {1, 0, 2});
    };
    if (keys.size() > 1 && verify) {
        // Each verification cohort has a common lookahead, but independent KV pages.
        const int rows = length / keys.size();
        for (Data *x : {&q, &k, &v}) {
            PermuteSelf(*x, {0,2,1,3});
            x->Reshape({-1,length,head_dim});
        }
        for (int b = 0; b < (int)keys.size(); ++b) {
            Data kb, vb;
            Split(k, 1, b * rows, (b + 1) * rows, kb);
            Split(v, 1, b * rows, (b + 1) * rows, vb);
            AppendPagedCache(*keys[b]->pagedKVCacheData, *keys[b], kb);
            AppendPagedCache(*values[b]->pagedKVCacheData, *values[b], vb);
        }
        GeneratePagedBatchParams(q, keys, keys.size(), pages.qSizes, pages.pageSizes,
            pages.indices, pages.lastLens, std::vector<int>(keys.size(), rows));
        AttentionPagedBatch(q, pastKey, pastValue, pages.qSizes, pages.pageSizes, pages.indices,
            pages.lastLens, attn, group, scale, 1, false, false);
    } else if (keys.size() > 1 && length > (int) keys.size()) {
        const int perRequest = length / keys.size();
        std::vector<Data> results(keys.size());
        for (int b = 0; b < (int) keys.size(); ++b) {
            Data qb, kb, vb;
            Split(q, 1, b * perRequest, (b + 1) * perRequest, qb);
            Split(k, 1, b * perRequest, (b + 1) * perRequest, kb);
            Split(v, 1, b * perRequest, (b + 1) * perRequest, vb);
            attendOne(qb, kb, vb, perRequest, *keys[b], *values[b], results[b]);
        }
        SenseNovaJoinPrefillRows(results, perRequest, attn);
    } else if (keys.size() > 1) {
        // Packed token rows are independent requests. FlashInfer consumes Q
        // head-major, the append operator consumes K and V token-major.
        PermuteSelf(q, {0, 2, 1, 3});
        q.Reshape({num_attention_heads, length, head_dim});
        k.Reshape({length, num_key_value_heads, head_dim});
        v.Reshape(k.dims);
        // Each visit owns a separate pool. Reuse append locations only for
        // existing pages; at a boundary the manager must preview its free list.
        bool sameAppend = pages.appendOffsets.cpuIntDatas.size() == keys.size();
        for (int b = 0; sameAppend && b < length; ++b) {
            const Data &key = *keys[b];
            sameAppend = key.lastPageLen < key.pageLen &&
                pages.appendOffsets.cpuIntDatas[b] == key.lastPageLen &&
                pages.appendIndices.cpuIntDatas[b] == key.pageIndex.back();
        }
        if (!sameAppend)
            GenerateAppendPagedCacheBatchParams(*pastKey.pagedKVCacheData, keys, length,
                                                pages.appendIndices, pages.appendOffsets);
        AppendPagedCacheBatch(*pastKey.pagedKVCacheData, keys, k, pages.appendIndices, pages.appendOffsets);
        AppendPagedCacheBatch(*pastValue.pagedKVCacheData, values, v, pages.appendIndices, pages.appendOffsets);
        bool sameTable = pages.pageSizes.cpuIntDatas.size() == keys.size() + 1;
        for (int b = 0; b < length; ++b) {
            const Data &key = *keys[b];
            const int tokens = (key.pageIndex.size() - 1) * key.pageLen + key.lastPageLen;
            for (Data *cache : {keys[b], values[b]})
                cache->Resize({num_key_value_heads, tokens, head_dim});
            if (sameTable) {
                const int begin = pages.pageSizes.cpuIntDatas[b], end = pages.pageSizes.cpuIntDatas[b + 1];
                sameTable = size_t(end - begin) == key.pageIndex.size() && pages.lastLens.cpuIntDatas[b] == key.lastPageLen &&
                    std::equal(key.pageIndex.begin(), key.pageIndex.end(), pages.indices.cpuIntDatas.begin() + begin);
            }
        }
        if (!sameTable)
            GeneratePagedBatchParams(q, keys, length, pages.qSizes, pages.pageSizes, pages.indices, pages.lastLens);
        AttentionPagedBatch(q, pastKey, pastValue, pages.qSizes, pages.pageSizes, pages.indices, pages.lastLens,
                            attn, group, scale, 1, false, false);
    } else {
        attendOne(q, k, v, length, pastKey, pastValue, attn);
    }
    attn.Reshape(gate.dims);
    if (length == 1 || verify || length == (int) keys.size()) {
        SigmoidMulTo(attn, gate);
    } else {
        Sigmoid(gate, gate);
        MulTo(attn, gate);
    }
    Linear(attn, weight[p + "o_proj.weight"], Data(), out);
}

void SenseNovaLoopModel::LinearAttention(const std::string &p, Data &input,
        const std::vector<Data*> &convCaches, const std::vector<Data*> &states, Data &out,
        std::vector<std::pair<Data, Data>> *prefixes) {
    Data &conv = *convCaches.front(), &state = *states.front();
    int length = input.dims[1];
    int channels = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
    Data packedQkvZ, packedBA, qkv, z, a, b;
    auto projectPair = [&](const std::string &combined, const std::string &left, const std::string &right,
                           int split, Data &packed, Data &x, Data &y) {
        auto project = [&](const std::string &name, Data &result) {
            Data &w = weight.weight.at(p + name);
            if (combined == "in_proj_ba.weight" && input.dataDevice == CUDA &&
                input.dataType == BFLOAT16 && w.dataType == BFLOAT16 && input.dims.back() % 8 == 0) {
                SenseNovaBf16GateProjection(input, w, result);
                return;
            }
            Linear(input, w, Data(), result);
        };
        if (weight.weight.count(p + combined)) {
            project(combined, packed);
            // The verifier consumes merged projections with their row pitch.
            if (prefixes) return;
            int rightWidth = packed.dims.back() - split;
            if (length == 1) {
                x.FakeFrom(packed, 0);
                x.Resize({1, 1, split});
                y.FakeFrom(packed, (size_t)split * packed.unitSize);
                y.Resize({1, 1, rightWidth});
            } else {
                Split(packed, -1, 0, split, x);
                Split(packed, -1, split, packed.dims.back(), y);
            }
        } else {
            project(left, x);
            project(right, y);
        }
    };
    projectPair("in_proj_qkvz.weight", "in_proj_qkv.weight", "in_proj_z.weight", channels, packedQkvZ, qkv, z);
    projectPair("in_proj_ba.weight", "in_proj_b.weight", "in_proj_a.weight", num_v_heads, packedBA, b, a);
    auto prepareState = [](Data &conv, Data &state) {
        for (Data *s : {&conv, &state}) {
            s->isKVCache = s->isLinearAttention = true;
            if (s->dims.empty()) {
                s->dataType = FLOAT32;
                s->UpdateUnitSize();
            }
        }
    };
    for (size_t b = 0; b < states.size(); ++b) prepareState(*convCaches[b], *states[b]);
    auto recurrent = [&](Data &qkv, Data &z, Data &a, Data &b, Data &conv, Data &state, Data &gatedOut) {
        const int length = qkv.dims[1];
        Data convolved, core;
        Data &convWeight = weight[p + "conv1d.weight"];
        if (length == 1 && convKernel == 4 && conv.dataDevice == CUDA && conv.cudaData &&
            convWeight.dataDevice == CUDA && convWeight.dataType == FLOAT32) {
            SenseNovaConvDecode(qkv, convWeight, conv, b, convolved);
        } else {
            CausalDepthwiseConv1DPrefill(qkv, convWeight, conv, convKernel, false, convolved, BFLOAT16);
            Silu(convolved, convolved);
            ToDataType(convolved, FLOAT32);
            Sigmoid(b, b);
        }
        if (state.dims.empty()) {
            state.Resize({1, num_v_heads, head_k_dim, head_v_dim});
            state.ToDevice(input.dataDevice, input.dataDeviceIds);
            state.Allocate(0.0f);
        }
        // This operator normalizes by mean-square. Transformers uses sum-square + epsilon.
        GatedDeltaRuleSequence(convolved, a, b, weight[p + "A_log"], weight[p + "dt_bias"],
                               state, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                               1e-6f / head_k_dim, core, nullptr, true);
        Data &normWeight = weight[p + "norm.weight"];
        if (head_v_dim == 128 && core.dataDevice == CUDA && z.dataType == BFLOAT16 &&
            normWeight.dataType == FLOAT32) {
            SenseNovaGdnOutput(core, z, normWeight, rms_norm_eps, gatedOut);
            return;
        }
        core.Reshape({-1, head_v_dim});
        ToDataType(core, BFLOAT16);
        RMSNorm(core, headNorm, rms_norm_eps, core);
        Data gatedScale;
        ToDataType(weight[p + "norm.weight"], gatedScale, BFLOAT16);
        gatedScale.Reshape({1, head_v_dim});
        Data scaleRows;
        Repeat(gatedScale, 0, core.dims[0], scaleRows);
        MulTo(core, scaleRows);
        ToDataType(core, FLOAT32);
        z.Reshape(core.dims);
        ToDataType(z, FLOAT32);
        Silu(z, z);
        MulTo(core, z);
        ToDataType(core, BFLOAT16);
        core.Reshape({1, length, num_v_heads * head_v_dim});
        gatedOut.CopyFrom(core);
    };
    Data gated;
    if (prefixes) {
        const bool mergedQkv = !packedQkvZ.dims.empty(), mergedBA = !packedBA.dims.empty();
        SenseNovaVerifyRecurrent(mergedQkv ? packedQkvZ : qkv, mergedQkv ? packedQkvZ : z,
            mergedBA ? packedBA : a, mergedBA ? packedBA : b, weight[p+"conv1d.weight"],
            weight[p+"A_log"], weight[p+"dt_bias"], weight[p+"norm.weight"], convCaches, states,
            num_k_heads, num_v_heads, rms_norm_eps, *prefixes, gated, mergedQkv, mergedBA);
    } else if (states.size() > 1 && length > (int) states.size()) {
        const int perRequest = length / states.size();
        std::vector<Data> results(states.size());
        for (int i = 0; i < (int) states.size(); ++i) {
            Data qi, zi, ai, bi;
            auto view = [&](Data &target, const Data &source) {
                target.FakeFrom(source, size_t(i) * perRequest * source.dims.back() * source.unitSize);
                target.Resize({1, perRequest, source.dims.back()});
            };
            view(qi, qkv);
            view(zi, z);
            view(ai, a);
            view(bi, b);
            recurrent(qi, zi, ai, bi, *convCaches[i], *states[i], results[i]);
        }
        SenseNovaJoinPrefillRows(results, perRequest, gated);
    } else if (states.size() > 1) {
        Data core;
        SenseNovaBatchRecurrent(qkv, a, b, weight[p+"conv1d.weight"], weight[p+"A_log"], weight[p+"dt_bias"],
                                convCaches, states, num_k_heads, num_v_heads, core);
        SenseNovaGdnOutput(core, z, weight[p+"norm.weight"], rms_norm_eps, gated);
    } else {
        recurrent(qkv, z, a, b, conv, state, gated);
    }
    Linear(gated, weight[p + "out_proj.weight"], Data(), out);
}

void SenseNovaLoopModel::RoutedInput(const std::string &p, Data &streams, Data &branch, Data &mix, bool exactRows, int promptRows) {
    Data logits;
    Data &w = weight[p + ".weight"], &bias = weight[p + ".bias"];
    if ((streams.dims[1] == 1 || exactRows) && streams.dataDevice == CUDA && embed_dim == 4096 &&
        w.dataType == FLOAT32 && w.dims == std::vector<int>({8, 8192}) &&
        bias.dataType == FLOAT32 && bias.Count(0) == 8) {
        SenseNovaRouterLogits(streams, w, bias, logits);
    } else {
        Data norm;
        if (streams.dataDevice == CUDA && streams.dataType == BFLOAT16 && embed_dim == 4096) {
            SenseNovaRouterNormalize(streams, norm);
        } else {
            Data flat;
            ToDataType(streams, flat, FLOAT32);
            flat.Reshape({1, streams.dims[1], 2 * embed_dim});
            RMSNorm(flat, routerNorm, 1e-6f, norm);
        }
        if (promptRows > 0) {
            // The FP32 router GEMM selects a different reduction for a wider M.
            // Keep the single-prompt reduction before normalizing the route weights.
            std::vector<Data> rows(streams.dims[1] / promptRows);
            for (int b = 0; b < (int)rows.size(); ++b) {
                Data slice;
                slice.FakeFrom(norm, size_t(b) * promptRows * norm.dims.back() * norm.unitSize);
                slice.Resize({1, promptRows, norm.dims.back()});
                Linear(slice, w, bias, rows[b]);
            }
            SenseNovaJoinPrefillRows(rows, promptRows, logits);
        } else {
            Linear(norm, w, bias, logits);
        }
    }
    AssertInFastLLM(logits.dataDevice == CUDA && logits.dataType == FLOAT32 && streams.dataDevice == CUDA,
                    "SenseNova Loop mHC requires a single CUDA device.");
    SenseNovaRoute(logits, streams, iterations, branch, mix);
}

std::vector<int> SenseNovaLoopModel::RunTarget(const Data &ids, const std::vector<Data*> &positions,
        const std::vector<int> &lengths, Cache &cache, const std::vector<GenerationConfig> &configs,
        const LastTokensManager &last, std::vector<std::vector<float>*> *logits, Data *capture, PrefixStates *prefixes, Data *verifyLogits) {
    const int length = ids.dims[1], batch = lengths.size();
    const bool allDecode = batch == length;
    const int promptRows = batch > 1 && !allDecode && !prefixes ? lengths.front() : 0;
    FastllmCudaNvfp4Scope nvfp4(nvfp4PrefillPlan, batch > 1, promptRows, allDecode ? nvfp4DecodeCache.get() : nullptr,
                                prefixes ? lengths.front() : 0);
    ApplyDeviceMap(deviceMap, 1, 1);
    Data embedded, hidden, streams, intermediate, mix, branch, normalized, update;
    Embedding(ids, weight[language_prefix + "embed_tokens.weight"], embedded);
    ToDataType(embedded, hidden, BFLOAT16);
    Data pos = BuildFlattenedPositionIds(positions, lengths, allDecode);
    // Rotating (1, 0) with the standard Qwen helper produces the checkpoint's
    // BF16 cosine/sine table once for all full-attention visits.
    Data rotary;
    Repeat(rotaryBasis, 1, length, rotary);
    AssertInFastLLM(rotary.dataDevice == CUDA, "SenseNova Loop requires a single CUDA device.");
    ApplyMultimodalRotary(rotary, pos, 1.0f);
    DecodePages pages;
    std::vector<Data*> keys(batch), values(batch);
    for (size_t slot = 0; slot < visits.size(); ++slot) {
        int physical = visits[slot];
        std::string layer = language_prefix + "layers." + std::to_string(physical) + ".";
        bool routed = physical >= loopStart && physical < loopEnd;
        if (physical == loopStart) {
            hidden.Reshape({1, length, 1, embed_dim});
            Repeat(hidden, 2, 2, streams);
        }
        if (routed) RoutedInput(layer + "mhc_attn", streams, branch, mix, prefixes != nullptr || (batch > 1 && allDecode), promptRows);
        RMSNorm(routed ? branch : hidden, weight[layer + "input_layernorm.weight"], rms_norm_eps, normalized);
        for (int b = 0; b < batch; ++b) {
            keys[b] = cache[b * visits.size() + slot].first;
            values[b] = cache[b * visits.size() + slot].second;
        }
        if (weight.weight.count(layer + "self_attn.o_proj.weight")) {
            FullAttention(layer + "self_attn.", normalized, rotary, slot, keys, values, pages, prefixes != nullptr, update);
        } else {
            LinearAttention(layer + "linear_attn.", normalized, keys, values, update, prefixes ? &(*prefixes)[slot] : nullptr);
        }
        if (routed) {
            SenseNovaMerge(update, mix, streams, intermediate);
            RoutedInput(layer + "mhc_ffn", intermediate, branch, mix, prefixes != nullptr || (batch > 1 && allDecode), promptRows);
        } else {
            AddTo(hidden, update);
        }
        RMSNorm(routed ? branch : hidden, weight[layer + "post_attention_layernorm.weight"], rms_norm_eps, normalized);
        Data activated, gateUp;
        if (weight.weight.count(layer + "mlp.gateup_proj.weight")) {
            Linear(normalized, weight[layer + "mlp.gateup_proj.weight"], Data(), gateUp);
            if (length <= 65535 && gateUp.dataType == BFLOAT16) {
                SenseNovaGateUp(gateUp, activated);
            } else {
                Data up;
                int width = gateUp.dims.back() / 2;
                Split(gateUp, -1, 0, width, activated);
                Split(gateUp, -1, width, 2 * width, up);
                Silu(activated, activated);
                MulTo(activated, up);
            }
        } else {
            Linear(normalized, weight.weight.at(layer + "mlp.gate_proj.weight"), Data(), activated);
            Silu(activated, activated);
            Linear(normalized, weight.weight.at(layer + "mlp.up_proj.weight"), Data(), gateUp);
            MulTo(activated, gateUp);
        }
        Linear(activated, weight[layer + "mlp.down_proj.weight"], Data(), update);
        if (routed) {
            SenseNovaMerge(update, mix, intermediate, streams);
        } else {
            AddTo(hidden, update);
        }
        if (physical == loopEnd - 1) {
            Data left, right;
            Split(streams, 2, 0, 1, left);
            Split(streams, 2, 1, 2, right);
            AddTo(left, right);
            Mul(left, 0.5f, hidden);
            hidden.Reshape({1, length, embed_dim});
        }
    }
    std::vector<int> output;
    if (capture)
        RMSNorm(hidden, weight[language_prefix + "norm.weight"], rms_norm_eps, *capture);
    if (prefixes) {
        Linear(*capture, weight["lm_head.weight"], Data(), *verifyLogits);
        ToDataType(*verifyLogits, FLOAT32);
        return {};
    }
    LLMSamplingBlock(this, &hidden, &weight[language_prefix + "norm.weight"], &weight["lm_head.weight"],
                     rms_norm_eps, batch, allDecode, lengths, cache, configs, last, logits, output);
    return output;
}

std::vector<int> SenseNovaLoopModel::SampleVerifyRows(Data &logits, const std::vector<MtpKvCache *> &proposals,
                                                      const std::vector<GenerationConfig> &configs, int drafts,
                                                      std::vector<int> &accepted) {
    const int batch = proposals.size(), rows = drafts + 1, vocab = logits.dims.back();
    std::vector<int> sampled(batch * rows);
    accepted.assign(batch, -1); // Greedy acceptance is determined by matching the proposed IDs.
    auto greedy = [](Data &scores, int count, int *out) {
        Data best;
        TopK(scores, best, 1);
        best.ToDevice(CPU);
        const auto *pairs = reinterpret_cast<const float *>(best.cpuData);
        for (int i = 0; i < count; ++i) out[i] = int(pairs[2 * i]);
    };
    if (std::all_of(proposals.begin(), proposals.end(), [](auto *p) { return !p->sampleProposal; })) {
        greedy(logits, batch * rows, sampled.data());
        return sampled;
    }
    auto *scores = (float *)logits.cudaData;
    if (batch > 1 && std::all_of(proposals.begin(), proposals.end(),
                                 [](auto *p) { return p->sampleProposal && p->proposalUsesLogits; })) {
        std::vector<FastllmMtpProposalView> views(batch);
        std::vector<float> temperatures, topPs;
        std::vector<int> topKs;
        for (int b = 0; b < batch; ++b) {
            const auto &q = *proposals[b];
            views[b] = {(float *)q.proposalProbs.cudaData, (float *)q.proposalLogsumexp.cudaData,
                        (int *)q.proposalDeviceTokens.cudaData};
            temperatures.insert(temperatures.end(), rows, configs[b].temperature);
            topPs.insert(topPs.end(), rows, configs[b].top_p);
            topKs.insert(topKs.end(), rows, configs[b].top_k);
        }
        AssertInFastLLM(FastllmCudaMtpRejectionSamplingLogitsBatch(scores, views.data(), temperatures.data(),
                                                                   topKs.data(), topPs.data(), sampled.data(),
                                                                   accepted.data(), batch, drafts, vocab),
                        "SenseNova batched rejection sampling failed.");
    } else {
        for (int b = 0; b < batch; ++b) {
            auto *out = sampled.data() + b * rows;
            if (proposals[b]->sampleProposal) {
                accepted[b] =
                    VerifyMtpSamplingRows(scores + size_t(b) * rows * vocab, vocab, logits.dataDeviceIds.front(),
                                          *proposals[b], drafts, configs[b], out);
            } else {
                Data slice;
                slice.FakeFrom(logits, size_t(b) * rows * vocab * sizeof(float));
                slice.Resize({1, rows, vocab});
                greedy(slice, rows, out);
            }
        }
    }
    return sampled;
}

// Near exhaustion, the scheduler's per-request fallback can reuse pages released
// by the preceding request. Account for shared pools before advancing any lane.
bool SenseNovaLoopModel::HasMtpBatchPages(const std::vector<std::shared_ptr<DraftRequest>> &requests,
                                         const std::vector<int> &lengths, const Cache &cache) const {
    if (requests.size() == 1) return true;
    std::map<PagedCacheManager *, int> needed;
    for (size_t b = 0; b < requests.size(); ++b) {
        const int rows = lengths[b] + requests[b]->drafts.size();
        for (size_t slot = 0; slot < visits.size(); ++slot) {
            auto [key, value] = cache[b * visits.size() + slot];
            for (Data *kv : {key, value}) {
                if (!kv->isPagedKVCache || !kv->pagedKVCacheData) continue;
                const int extra = rows - (kv->pageLen - kv->lastPageLen);
                if (extra > 0) needed[kv->pagedKVCacheData] += (extra + kv->pageLen - 1) / kv->pageLen;
            }
        }
    }
    return std::all_of(needed.begin(), needed.end(),
                       [](const auto &item) { return item.first->FreePageCount() >= item.second; });
}

std::vector<int> SenseNovaLoopModel::DraftForwardBatch(const std::vector<std::shared_ptr<DraftRequest>> &active,
                                                       const Data &ids, const std::vector<Data *> &positions,
                                                       const std::vector<int> &lengths, Cache &cache,
                                                       const std::vector<GenerationConfig> &configs,
                                                       const LastTokensManager &last) {
    const int batch = active.size(), slots = visits.size();
    Data cpuIds(ids);
    cpuIds.ToDevice(CPU);
    const float *id = (const float *)cpuIds.cpuData;
    std::vector<std::vector<int>> inputs(batch), predicted(batch);
    std::vector<int> groups[3], output(batch);
    std::vector<Data> pos(batch), hidden(batch);
    Data captures[3];
    auto stopped = [&](int b, int token) {
        return token == eos_token_id || eos_token_ids.count(token) || configs[b].stop_token_ids.count(token);
    };
    int offset = 0;
    for (int b = 0; b < batch; ++b) {
        auto &r = *active[b];
        const int base = r.kv.tokens;
        int budget = max_positions - base;
        if (tokensLimit > 0)
            budget = std::min(budget, tokensLimit - base);
        if (configs[b].output_token_limit > 0)
            budget = std::min(budget, configs[b].output_token_limit - r.emitted);
        for (int slot = 0; slot < slots; ++slot) {
            Data &k = *cache[b * slots + slot].first;
            if (k.isPagedKVCache && k.pagedKVCacheData)
                budget =
                    std::min(budget, k.pageLen - k.lastPageLen + k.pageLen * k.pagedKVCacheData->FreePageCount());
        }
        int look = std::min((int)r.drafts.size(), std::max(0, budget - 1));
        inputs[b].assign(id + offset, id + offset + lengths[b]);
        inputs[b].insert(inputs[b].end(), r.drafts.begin(), r.drafts.begin() + look);
        if (look) {
            groups[look].push_back(b);
            AssertInFastLLM(lengths[b] == 1, "Expected a single SenseNova MTP anchor.");
            if (r.kv.sampleProposal)
                AssertInFastLLM(
                    (int)r.kv.proposalTokens.size() >= look &&
                        std::equal(r.drafts.begin(), r.drafts.begin() + look, r.kv.proposalTokens.begin()),
                    "SenseNova MTP proposal tokens and densities are out of sync.");
            Data first = BuildMtpPositionIdsSlice(*positions[b], 0, 1, 0);
            std::vector<float> steps(look + 1);
            for (int t = 0; t <= look; ++t)
                steps[t] = ((float *)first.cpuData)[0] + t;
            pos[b].CopyFrom(Data(FLOAT32, {1, look + 1}, steps));
        } else {
            pos[b].CopyFrom(*positions[b]);
            Data input(FLOAT32, {1, lengths[b]}, std::vector<float>(id + offset, id + offset + lengths[b]));
            Cache lane(cache.begin() + b * slots, cache.begin() + (b + 1) * slots);
            LastTokensManager history(1, configs[b].last_n);
            if (b < (int)last.units.size())
                history.units[0] = last.units[b];
            predicted[b] = RunTarget(input, {&pos[b]}, {lengths[b]}, lane, {configs[b]}, history, nullptr,
                                     &hidden[b], nullptr);
        }
        offset += lengths[b];
    }

    verifyPrefixes.resize(slots);
    for (int look = 1; look <= 2; ++look) {
        const auto &group = groups[look];
        if (group.empty())
            continue;
        const int rows = look + 1;
        std::vector<float> tokens;
        std::vector<Data *> groupPos;
        std::vector<MtpKvCache *> proposals;
        std::vector<GenerationConfig> groupConfigs;
        Cache groupCache;
        for (int b : group) {
            tokens.insert(tokens.end(), inputs[b].begin(), inputs[b].end());
            groupPos.push_back(&pos[b]);
            proposals.push_back(&active[b]->kv);
            groupConfigs.push_back(configs[b]);
            groupCache.insert(groupCache.end(), cache.begin() + b * slots, cache.begin() + (b + 1) * slots);
        }
        Data packed(FLOAT32, {1, (int)tokens.size()}, tokens), logits;
        RunTarget(packed, groupPos, std::vector<int>(group.size(), rows), groupCache, groupConfigs, last, nullptr,
                  &captures[look], &verifyPrefixes, &logits);
        std::vector<int> acceptedRows;
        auto sampled = SampleVerifyRows(logits, proposals, groupConfigs, look, acceptedRows);
        for (int i = 0; i < (int)group.size(); ++i) {
            int b = group[i], accepted = acceptedRows[i];
            auto &r = *active[b];
            const int *pred = sampled.data() + i * rows;
            if (accepted < 0) {
                accepted = 0;
                while (accepted < look && !stopped(b, pred[accepted]) && pred[accepted] == r.drafts[accepted])
                    ++accepted;
            }
            int committed = accepted + 1;
            for (int t = 0; t < accepted; ++t)
                if (stopped(b, pred[t])) {
                    committed = t + 1;
                    break;
                }
            r.rounds++;
            r.acceptedFirst += accepted > 0;
            r.accepted += std::min(accepted, committed);
            r.proposed += look;
            if (committed < rows)
                for (int slot = 0; slot < slots; ++slot) {
                    auto [k, v] = cache[b * slots + slot];
                    if (k->isLinearAttention) {
                        auto &prefix = verifyPrefixes[slot][i * look + committed - 1];
                        SenseNovaCopyState(prefix.first, *k);
                        SenseNovaCopyState(prefix.second, *v);
                    } else {
                        for (Data *kv : {k, v}) {
                            const int len = kv->dims[1] - (rows - committed),
                                      pages = (len + kv->pageLen - 1) / kv->pageLen;
                            std::vector<int> released(kv->pageIndex.begin() + pages, kv->pageIndex.end());
                            kv->pageIndex.resize(pages);
                            kv->lastPageLen = (len - 1) % kv->pageLen + 1;
                            kv->Resize({kv->dims[0], len, kv->dims[2]});
                            kv->pagedKVCacheData->ReleasePageIndices(released);
                        }
                    }
                }
            predicted[b].assign(pred, pred + committed);
            inputs[b].resize(committed);
            const Data &capture = captures[look];
            hidden[b].FakeFrom(capture, size_t(i) * rows * embed_dim * capture.unitSize);
            hidden[b].Resize({1, committed, embed_dim});
        }
    }

    std::vector<int> firstRows, secondRows;
    std::vector<int> saved(batch);
    for (int b = 0; b < batch; ++b) {
        auto &r = *active[b];
        auto &pred = predicted[b];
        r.pending.assign(pred.begin() + 1, pred.end());
        r.emitted += pred.size();
        r.drafts.clear();
        output[b] = pred.front();
        if (stopped(b, pred.back()) ||
            (configs[b].output_token_limit > 0 && r.emitted + 1 >= configs[b].output_token_limit))
            continue;
        inputs[b].erase(inputs[b].begin());
        inputs[b].push_back(pred.back());
        r.kv.BeginProposal(configs[b]);
        firstRows.push_back(b);
    }
    if (firstRows.empty())
        return output;
    const int device = hidden[firstRows.front()].dataDeviceIds.front();
    std::vector<Data> firstHidden;
    auto draft = [&](const std::vector<int> &lanes, bool second) {
        std::vector<MtpKvCache *> caches;
        std::vector<const Data *> hs;
        std::vector<std::vector<int>> tokens;
        std::vector<int> sampleRows;
        std::vector<Data> slices(lanes.size());
        std::vector<Data *> positionPtrs;
        for (int i = 0; i < (int)lanes.size(); ++i) {
            int b = lanes[i], count = inputs[b].size();
            auto &r = *active[b];
            caches.push_back(&r.kv);
            if (second) {
                hs.push_back(&firstHidden[saved[b]]);
                tokens.push_back({r.drafts.front()});
            } else {
                hs.push_back(&hidden[b]);
                tokens.push_back(inputs[b]);
            }
            slices[i].CopyFrom(BuildMtpPositionIdsSlice(pos[b], second ? count - 1 : 0, count, second ? 1 : 0));
            positionPtrs.push_back(&slices[i]);
            sampleRows.push_back(second ? 0 : count - 1);
        }
        return RunMtpDraftBatch(device, {device}, caches, hs, tokens, positionPtrs, sampleRows,
                                !second && draftCount == 2 ? &firstHidden : nullptr);
    };
    auto first = draft(firstRows, false);
    std::vector<int> committedTokens(batch);
    for (int i = 0; i < (int)firstRows.size(); ++i) {
        int b = firstRows[i];
        auto &r = *active[b];
        r.drafts.push_back(first[i]);
        if (r.kv.proposalDepth == 2 && !stopped(b, first[i]) &&
            (configs[b].output_token_limit <= 0 || configs[b].output_token_limit > r.emitted + 2)) {
            saved[b] = i;
            committedTokens[b] = r.kv.tokens;
            secondRows.push_back(b);
        }
    }
    if (!secondRows.empty()) {
        try {
            auto second = draft(secondRows, true);
            for (int i = 0; i < (int)secondRows.size(); ++i)
                active[secondRows[i]]->drafts.push_back(second[i]);
        } catch (...) {
            for (int b : secondRows)
                active[b]->kv.Truncate(committedTokens[b]);
            throw;
        }
        for (int b : secondRows)
            active[b]->kv.Truncate(committedTokens[b]);
    }
    return output;
}

#endif

std::vector<int> SenseNovaLoopModel::ForwardMultimodal(const Data &, const Data &, const Data &,
        std::vector<std::pair<Data, Data>> &, const std::map<std::string, std::vector<Data*>> &,
        const GenerationConfig &, const LastTokensManager &, std::vector<std::vector<float>*> *) {
    throw std::runtime_error("SenseNova Loop currently supports text inputs.");
}
}
