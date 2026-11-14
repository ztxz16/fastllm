// Request orchestration shares stateless math across concatenated row segments.
// Attention, rejection sampling and recurrent checkpoints keep request ownership.
namespace {
    void Qwen4BatchView(Data &view, const Data &source, int start, int rows) {
        AssertInFastLLM(source.dims.size() == 3 && source.dims[0] == 1 &&
            start >= 0 && rows > 0 && rows <= source.dims[1] && start <= source.dims[1] - rows,
            "Qwen4 batch row view is out of range.");
        view.FakeFrom(source, (size_t)start * source.dims.back() * source.unitSize);
        view.Resize({1, rows, source.dims.back()});
    }

    void Qwen4BatchPack(const std::vector<Data*> &parts, Data &output) {
        AssertInFastLLM(!parts.empty(), "Qwen4 cannot concatenate an empty batch.");
        int rows = 0;
        for (auto *part : parts) {
            AssertInFastLLM(part->dims.size() == 3 && part->dims[0] == 1 &&
                part->dims.back() == parts.front()->dims.back() &&
                part->dataType == parts.front()->dataType,
                "Qwen4 batch tensors have incompatible shapes.");
            AssertInFastLLM(part->dims[1] > 0 && rows <= std::numeric_limits<int>::max() - part->dims[1],
                "Qwen4 batch row count exceeds tensor indexing capacity.");
            rows += part->dims[1];
        }
        output.CopyFrom(*parts.front());
        if (parts.size() == 1) return;
        output.Expansion({1, rows, output.dims.back()});
        for (size_t i = 1; i < parts.size(); ++i) CatDirect(output, *parts[i], 1);
    }

    std::vector<int> Qwen4HostInts(const Data &input) {
        Data host;
        ToDataType(input, host, FLOAT32);
        host.ToDevice(CPU);
        std::vector<int> values(host.Count(0));
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = int(reinterpret_cast<const float*>(host.cpuData)[i] + .01f);
        return values;
    }
}

#include "qwen4_exp_batch_graph.inc"
struct Qwen4ExpModel::RequestBatchState {
#ifdef USE_CUDA
    struct Entry { std::shared_ptr<BatchGraph> graph; uint64_t used = 0; size_t rows = 0; };
    using Key = std::vector<uint64_t>;
    std::map<int, std::map<Key, Entry>> graphs;
    std::map<int, size_t> peakRows;
    uint64_t clock = 0;
    std::shared_ptr<BatchGraph> &Select(int phase, const Key &key, int capacity) {
        auto &variants = graphs[phase];
        size_t rows = 0;
        for (size_t i = 1; i < key.size(); i += 2) rows += key[i];
        peakRows[phase] = std::max(peakRows[phase], rows);
        // Cache reuse must not restore the old quadratic graph-memory budget.
        // Keep at most two largest observed row worksets per stage, with an
        // entry bound tied to admitted requests. This limits retention only;
        // every valid shape can still capture and run.
        size_t retained = 0;
        for (const auto &item : variants) retained += item.second.rows;
        if (!variants.count(key)) while (!variants.empty() &&
                (variants.size() >= size_t(std::max(1, capacity)) ||
                 retained + rows > 2 * peakRows[phase])) {
            auto oldest = std::min_element(variants.begin(), variants.end(),
                [](const auto &a, const auto &b) { return a.second.used < b.second.used; });
            retained -= oldest->second.rows; variants.erase(oldest);
        }
        auto &entry = variants[key]; entry.used = ++clock; entry.rows = rows;
        return entry.graph;
    }
    void Remove(const RequestState *state) {
        for (auto &phase : graphs) for (auto it = phase.second.begin(); it != phase.second.end();) {
            bool owns = false;
            for (size_t i = 0; i < it->first.size(); i += 2)
                owns |= it->first[i] == uint64_t(uintptr_t(state));
            if (owns) it = phase.second.erase(it); else ++it;
        }
    }
#endif
    std::map<int, HybridMoeBatch> experts;
};

void Qwen4ExpModel::InvalidateServingBatchGraphs(const Data *key) {
#ifdef USE_CUDA
    std::lock_guard<std::mutex> guard(stateMutex);
    auto state = requestStates.find(key);
    if (state != requestStates.end() && state->second.batchState)
        state->second.batchState->Remove(&state->second);
#endif
}

struct Qwen4ExpModel::BatchedRequest {
    std::vector<std::pair<Data, Data>> *cache = nullptr;
    const GenerationConfig *config = nullptr;
    RequestState *state = nullptr;
    LastTokensManager lastTokens;
    std::vector<std::vector<float>*> logits;
    Data input, mask, position, hidden;
    std::vector<int> tokens, positions, output, verified;
    std::vector<unsigned char> accepted;
    bool verify = false;
    int resultIndex = 0;
};

#include "qwen4_exp_batch_graph_prepare.inc"

bool Qwen4ExpModel::PopPendingResponseToken(ResponseContext *context, int, int &token) {
    if (!context || context->pastKeyValues.empty() || threadTpState || threadTpRank >= 0)
        return false;
    // Match serving-slot lock order. The scheduler checks stop tokens, output
    // limits and grammar after every published token and advances its position
    // once for each extra result. The next batch therefore sees one new anchor
    // per request, regardless of how many proposals each request accepted.
    std::lock_guard<std::mutex> servingGuard(decodeServingMutex);
    std::lock_guard<std::mutex> stateGuard(stateMutex);
    const Data *key = &context->pastKeyValues.front().first;
    const auto slot = decodeServingSlots.find(key);
    if (slot != decodeServingSlots.end()) key = &slot->second->cache.front().first;
    const auto found = requestStates.find(key);
    if (found == requestStates.end() || !found->second.mtpState ||
        found->second.mtpState->pendingOutputTokens.empty()) return false;
    auto &pending = found->second.mtpState->pendingOutputTokens;
    token = pending.front(); pending.pop_front();
    return true;
}

std::vector<int> Qwen4ExpModel::ForwardBatch(int batch, const Data &inputIds,
        const std::vector<Data*> &attentionMasks, const std::vector<Data*> &positionIds,
        const std::vector<int> &seqLens, std::vector<std::pair<Data*, Data*>> &pastKeyValues,
        const std::vector<GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens, std::vector<std::vector<float>*> *logits) {
    AssertInFastLLM(batch > 0 && batch == int(seqLens.size()) &&
        pastKeyValues.size() == size_t(batch) * block_cnt,
        "Qwen4 batched cache pointer count is invalid.");
    std::vector<std::vector<std::pair<Data, Data>>*> owners;
    {
        std::lock_guard<std::mutex> guard(stateMutex);
        for (int i = 0; i < batch; ++i) {
            auto found = requestStates.find(pastKeyValues[i * block_cnt].first);
            AssertInFastLLM(found != requestStates.end() && found->second.cacheOwner,
                "Qwen4 pointer batches require registered request caches; use ForwardRequests for owned cache vectors.");
            auto *owner = found->second.cacheOwner;
            for (int layer = 0; layer < block_cnt; ++layer)
                AssertInFastLLM(&owner->at(layer).first == pastKeyValues[i * block_cnt + layer].first &&
                    &owner->at(layer).second == pastKeyValues[i * block_cnt + layer].second,
                    "Qwen4 batched cache pointers do not belong to the same request.");
            owners.push_back(owner);
        }
    }
    return ForwardRequests(inputIds, attentionMasks, positionIds, seqLens, owners,
                           generationConfigs, lastTokens, logits);
}

std::vector<int> Qwen4ExpModel::ForwardRequests(
        const Data &inputIds, const std::vector<Data*> &attentionMasks,
        const std::vector<Data*> &positionIds, const std::vector<int> &seqLens,
        const std::vector<std::vector<std::pair<Data, Data>>*> &caches,
        const std::vector<GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens, std::vector<std::vector<float>*> *logits) {
    const size_t count = seqLens.size();
    AssertInFastLLM(count > 0 && caches.size() == count &&
        attentionMasks.size() == count && positionIds.size() == count &&
        generationConfigs.size() == count &&
        inputIds.dims.size() == 2 && inputIds.dims[0] == 1 &&
        inputIds.Count(0) == std::accumulate(seqLens.begin(), seqLens.end(), uint64_t(0)),
        "Qwen4 received an invalid request batch.");
    PrepareWeights();
    std::lock_guard<std::mutex> servingGuard(decodeServingMutex);
    std::vector<std::vector<std::pair<Data, Data>>*> bound;
    struct FailureGuard {
        Qwen4ExpModel &model;
        const std::vector<std::vector<std::pair<Data, Data>>*> &caches;
        bool completed = false;
        ~FailureGuard() {
            if (completed) return;
            for (auto *cache : caches) if (cache && !cache->empty()) {
                auto slot = model.decodeServingSlots.find(&cache->front().first);
                if (slot != model.decodeServingSlots.end() && slot->second) slot->second->failed = true;
            }
        }
    } failure{*this, caches};
    std::vector<std::unique_ptr<BatchedRequest>> storage;
    std::vector<BatchedRequest*> ready;
    std::vector<int> result(count, -1);
    int offset = 0;
    std::set<const void*> owners;
    for (size_t i = 0; i < count; ++i) {
        AssertInFastLLM(seqLens[i] > 0 && caches[i] &&
            caches[i]->size() >= (size_t)block_cnt && positionIds[i] && owners.insert(caches[i]).second,
            "Qwen4 request batch has missing or duplicate request storage.");
        storage.emplace_back(new BatchedRequest);
        auto &r = *storage.back();
        r.cache = &BindDecodeServingCache(*caches[i]); bound.push_back(r.cache); r.config = &generationConfigs[i]; r.resultIndex = i;
        Split(inputIds, 1, offset, offset + seqLens[i], r.input);
        offset += seqLens[i];
        r.position.CopyFrom(*positionIds[i]);
        if (attentionMasks[i]) r.mask.CopyFrom(*attentionMasks[i]);
        if (i < lastTokens.units.size()) r.lastTokens.units.push_back(lastTokens.units[i]);
        if (logits && i < logits->size()) r.logits.push_back((*logits)[i]);
        {
            std::lock_guard<std::mutex> guard(stateMutex);
            r.state = &requestStates[&r.cache->front().first];
            requestStates[&caches[i]->front().first].cacheOwner = caches[i];
        }
        auto *mtp = r.state->mtpState.get();
        if (mtp && !mtp->pendingOutputTokens.empty()) {
            result[i] = mtp->pendingOutputTokens.front();
            mtp->pendingOutputTokens.pop_front();
            continue;
        }
        const bool usesMtp = !r.state->mtpDisabled && MtpSupportsGenerationConfig(*r.config);
        if (!usesMtp && mtp) {
            mtp->proposals.clear();
            mtp->targetCheckpointPrepared = false;
        }
        r.verify = usesMtp && mtp && !mtp->proposals.empty();
        // Prefill, TP and multimodal/KVMem preserve their existing
        // admission/ownership path. Decode requests can have different positions,
        // accepted prefix lengths and sampling configurations in the same batch.
        if (seqLens[i] != 1 || threadTpState || threadTpRank >= 0 ||
            r.state->hasMultimodalInput || kvMemConfig.enabled || (usesMtp && !r.verify)) {
            result[i] = ForwardBatchInternal(1, r.input, r.mask, r.position, *r.cache,
                *r.config, r.lastTokens, r.logits.empty() ? nullptr : &r.logits)[0];
        } else {
            ready.push_back(&r);
        }
    }
    // Preserve row bindings when scheduler handle order changes or slots serve
    // new clients. Sampling/results still use each request's resultIndex.
    std::sort(ready.begin(), ready.end(), [](const auto *a, const auto *b) {
        return std::less<const Data*>()(&a->cache->front().first, &b->cache->front().first);
    });
    if (ready.size() == 1) {
        auto &r = *ready.front();
        result[r.resultIndex] = ForwardBatchInternal(1, r.input, r.mask, r.position, *r.cache,
            *r.config, r.lastTokens, r.logits.empty() ? nullptr : &r.logits)[0];
    } else if (!ready.empty()) {
        ForwardMtpBatch(ready);
        for (auto *r : ready) result[r->resultIndex] = r->output.front();
    }
    for (size_t i = 0; i < count; ++i) PublishDecodeServingCache(*caches[i], *bound[i]);
    failure.completed = true;
    return result;
}

std::vector<std::vector<int>> Qwen4ExpModel::ForwardTargetBatch(
        const std::vector<ForwardState*> &frames) {
    AssertInFastLLM(!frames.empty(), "Qwen4 target batch is empty.");
    Qwen4MtpGdnModeGuard mtpMode(true);
    std::vector<Data*> initial;
    std::vector<int> starts;
    int rows = 0;
    for (auto *frame : frames) {
        frame->Prepare();
        auto &f = *frame->work;
        AssertInFastLLM(f.firstLayer == 0 && f.endLayer == block_cnt && !f.prefill,
            "Qwen4 target batch requires complete request segments.");
        starts.push_back(rows); rows += f.inputIds.dims[1];
        initial.push_back(f.hiddenStates);
    }
    Data hidden[2], carriedInput, carriedInjection, finalHidden;
#ifdef USE_CUDA
    RequestBatchState::Key graphKey;
    for (auto *frame : frames) {
        graphKey.push_back(uint64_t(uintptr_t(frame->work->requestState)));
        graphKey.push_back(frame->work->inputIds.dims[1]);
    }
    auto &ownedGraph = frames.front()->work->requestState->batchState->Select(-1, graphKey, maxBatch);
    BatchGraph *graph = PrepareBatchGraph(ownedGraph, frames, {}, initial) ? ownedGraph.get() : nullptr;
#else
    constexpr bool graph = false;
#endif
    HybridMoeDecodeGroup expertSteps;
    if (!graph) Qwen4BatchPack(initial, hidden[0]);
    Data logits;
#ifdef USE_CUDA
    if (graph && graph->complete) {
        for (auto &table : weights)
            if (expertSteps.Begin(FastllmCudaGetDevice(), table.data(), table.size(), num_experts_per_tok)) break;
        AssertInFastLLM(graph->complete.Launch(), "Qwen cross-request target graph replay failed");
    } else
#endif
    {
#ifdef USE_CUDA
    if (graph) graph->Begin();
#endif
    int current = 0;
    bool hasCarriedRead = false;
    for (int layer = 0; layer < block_cnt; ++layer) {
        ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
        Data localInput, localInjection, localOutput;
#ifdef USE_CUDA
        auto &residual = graph ? (layer ? graph->layers[layer-1].residual : graph->hidden) : hidden[current];
        auto &next = graph ? graph->layers[layer].afterAttention : hidden[1-current];
        auto &mlpInput = graph ? graph->layers[layer].input : localInput;
        auto &mlpInjection = graph ? graph->layers[layer].injection : localInjection;
        auto &mlpOutput = graph ? graph->layers[layer].output : localOutput;
#else
        auto &residual = hidden[current]; auto &next = hidden[1-current];
        auto &mlpInput = localInput; auto &mlpInjection = localInjection; auto &mlpOutput = localOutput;
#endif
        auto dense = [&] {
        if (layer == pleLayer) {
            for (size_t i = 0; i < frames.size(); ++i) {
                auto &f = *frames[i]->work;
                Data part, ple;
                Qwen4BatchView(part, residual, starts[i], f.inputIds.dims[1]);
#ifdef USE_CUDA
                auto *capture = graph && f.verificationCapture
                    ? &graph->requests[i]->attention.verification : f.verificationCapture;
                if (capture) capture->pleInput.CopyFrom(part);
                if (graph) RunPLEProjection(graph->requests[i]->ple, part, *f.requestState, ple,
                    capture ? &capture->pleNormalized : nullptr, true);
                else
#else
                auto *capture = f.verificationCapture;
                if (capture) capture->pleInput.CopyFrom(part);
#endif
                RunPLE(part, f.inputIds, *f.requestState, ple, f.hostInputTokens,
                    &f.plePrefetch, capture ? &capture->pleNormalized : nullptr);
                AddTo(part, ple);
            }
            hasCarriedRead = false;
        }
        const std::string prefix = languagePrefix + "layers." + std::to_string(layer) + ".";
        const std::string attn = prefix + "attn_hyper_connection.";
        const std::string mlp = prefix + "mlp_hyper_connection.";
        Data attentionInput, injection, attentionOutput;
        auto borrow = [](Data &out, Data &in) {out.FakeFrom(in,0);out.Resize(in.dims);};
#ifdef USE_CUDA
        if (graph && layer && layer != pleLayer) {
            borrow(attentionInput, graph->layers[layer-1].readInput);
            borrow(injection, graph->layers[layer-1].readInjection);
        } else
#endif
        if (hasCarriedRead) {
            borrow(attentionInput,carriedInput); borrow(injection,carriedInjection);
        } else HyperRead(residual,attn,attentionInput,&injection,nullptr,nullptr,nullptr,dataType);
        hasCarriedRead = false;
        Data typedInput, projectionValues[4];
        Qwen4AttentionInput(attentionInput, typedInput, dataType);
        const bool linearLayer = IsLinearAttentionLayer(layer);
        std::vector<LinearAttentionConvolution> convolutions(linearLayer ? frames.size() : 0);
        if (linearLayer) {
            LinearAttentionProjections projections{&projectionValues[0], &projectionValues[1],
                                                   &projectionValues[2], &projectionValues[3]};
            ProjectLinearAttention(layer, typedInput, projections, [&] {
                // Keep each request's history and verifier capture independent,
                // while all convolutions overlap the shared Z/gate projections.
                for (size_t i = 0; i < frames.size(); ++i) {
                    auto &f = *frames[i]->work;
                    auto *capture = f.verificationCapture;
#ifdef USE_CUDA
                    if (graph && capture) capture = &graph->requests[i]->attention.verification;
#endif
                    Data view, qkv;
                    Qwen4BatchView(view, projectionValues[0], starts[i], f.inputIds.dims[1]);
                    qkv.CopyFrom(view);
                    ConvolveLinearAttention(layer, qkv, f.pastKeyValues[layer].first,
                        convolutions[i], capture ? &capture->linearConvInputs[layer] : nullptr);
                }
            });
        }
        std::vector<Data> attentionParts(frames.size());
        std::vector<Data*> parts;
        bool batchedGdn = false;
        bool batchedQsa = false;
        Data unprojected;
#ifdef USE_CUDA
        if (!linearLayer && graph) {
            std::vector<FullAttentionGraphRequest> requests;
            for (size_t i = 0; i < frames.size(); ++i) {
                auto &f = *frames[i]->work;
                auto &a = *graph->requests[i];
                auto &kv = f.pastKeyValues[layer];
                requests.push_back({&kv.first,&kv.second,f.requestState,&a.attention,
                    &a.attention.verification,a.rows});
            }
            FullAttentionProjections projections{&projectionValues[0],&projectionValues[1],&projectionValues[2]};
            batchedQsa = RunFullAttentionBatchGraph(layer,prefix+"self_attn.",typedInput,
                requests,projections,unprojected,false);
        }
        // Projections are already packed, while histories remain owned by
        // each request. Batch only independent decode states; verification
        // retains its out-of-place sequence update and rollback captures.
        if (linearLayer && frames.size()>1 && dataType==FLOAT16 &&
            head_k_dim==128 && head_v_dim==128 && typedInput.dataDevice==CUDA) {
            bool eligible=true;
            for (auto *frame:frames) {
                const auto &f=*frame->work;
                eligible &= f.inputIds.dims[1]==1 && !f.verificationCapture &&
                    !f.pastKeyValues[layer].second.dims.empty();
            }
            if (eligible) {
                Qwen4CudaDeviceGuard deviceGuard(typedInput.dataDeviceIds);
                std::vector<Data> gates(frames.size()), alphas(frames.size()), betas(frames.size());
                std::vector<FastllmCudaQwen4GdnDecodeRequest> requests;
                for (size_t i=0;i<frames.size();++i) {
                    Qwen4BatchView(gates[i],projectionValues[1],starts[i],1);
                    Qwen4BatchView(betas[i],projectionValues[2],starts[i],1);
                    Qwen4BatchView(alphas[i],projectionValues[3],starts[i],1);
                    if (!Qwen4PrepareCudaWorkspaceOnCurrentDevice(attentionParts[i],FLOAT16,
                            {1,1,num_v_heads*head_v_dim},FastllmCudaGetDevice())) {
                        eligible=false; break;
                    }
                    requests.push_back({&convolutions[i].values,&alphas[i],&betas[i],&gates[i],
                        &frames[i]->work->pastKeyValues[layer].second,&attentionParts[i]});
                }
                if (eligible) batchedGdn=FastllmCudaQwen4GatedDeltaRuleDecodeOutputBatch(requests,
                    weight[prefix+"linear_attn.A_log"],weight[prefix+"linear_attn.dt_bias"],
                    weight[prefix+"linear_attn.norm.weight"],num_k_heads,num_v_heads,1e-6f,rms_norm_eps);
            }
        }
#endif
        if (!linearLayer && !batchedQsa) {
            FullAttentionProjections projections{&projectionValues[0],&projectionValues[1],&projectionValues[2]};
            ProjectFullAttention(prefix+"self_attn.",typedInput,projections);
        }
        for (size_t i = 0; i < frames.size(); ++i) {
            if (batchedQsa) break;
            if (batchedGdn) { parts.push_back(&attentionParts[i]); continue; }
            auto &f = *frames[i]->work;
            auto *capture = f.verificationCapture;
#ifdef USE_CUDA
            if (graph && capture) capture = &graph->requests[i]->attention.verification;
#endif
            const int sequence = f.inputIds.dims[1];
            Data part;
            Qwen4BatchView(part, attentionInput, starts[i], sequence);
            auto &kv = f.pastKeyValues[layer];
            Data slices[4], unusedOutput;
            for (int j = 0; j < (linearLayer ? 4 : 3); ++j)
                Qwen4BatchView(slices[j], projectionValues[j], starts[i], sequence);
            if (linearLayer) {
                LinearAttentionProjections projections{&slices[0], &slices[1], &slices[2], &slices[3], &convolutions[i]};
                RunLinearAttention(layer, part, kv.first, kv.second, unusedOutput,
                    capture ? &capture->linearConvInputs[layer] : nullptr,
                    capture ? &capture->linearAlphas[layer] : nullptr,
                    capture ? &capture->linearBetas[layer] : nullptr,
                    capture && sequence > 1 && capture->runtimeCheckpoint
                        ? &capture->runtimeCheckpoint->linearSecond[layer] : nullptr,
                    &projections, &attentionParts[i],
                    capture ? &capture->linearUpdates[layer] : nullptr);
            } else {
                Data causal, positions;
                if (!graph) {
                    positions.CopyFrom(f.positionIds);
                    positions.ToDevice(part.dataDevice, part.dataDeviceIds);
                }
                const Data *mask = &f.attentionMask;
                if (!graph && sequence > 1 && mask->dims.empty() &&
                    !(part.dataDevice == CUDA && f.qsaPreviousLength >= indexerBudget)) {
                    causal.CopyFrom(Qwen4BuildCausalMask(f.qsaPreviousLength, sequence));
                    mask = &causal;
                }
                FullAttentionProjections projections{&slices[0], &slices[1], &slices[2]};
#ifdef USE_CUDA
                if (graph) {
                    auto &a = *graph->requests[i];
                    AssertInFastLLM(RunFullAttentionGraph(layer, prefix + "self_attn.", part,
                        kv.first, kv.second, *f.requestState, &a.attention, graph->device, a.width,
                        false, true, &a.attention.verification, unusedOutput, &projections, &attentionParts[i]),
                        "Qwen batch target graph attention failed");
                } else
#endif
                RunFullAttention(layer, part, *mask, positions, f.qsaDeviceCompatibleMask,
                    kv.first, kv.second, *f.requestState, unusedOutput,
                    capture ? &capture->qsaRawKeys[layer] : nullptr, f.decodeReserveTokens,
                    &projections, &attentionParts[i]);
            }
            parts.push_back(&attentionParts[i]);
        }
        if (!batchedQsa) Qwen4BatchPack(parts, unprojected);
        if (linearLayer)
            RunGdnOutputProjection(unprojected, weight[prefix + "linear_attn.out_proj.weight"], attentionOutput);
        else Linear(unprojected, weight[prefix + "self_attn.o_proj.weight"], Data(), attentionOutput);
        HyperRead(residual,mlp,mlpInput,&mlpInjection,&attentionOutput,&injection,&next);
#ifdef USE_CUDA
        if (graph) graph->Route(*this, targetMoePlans[layer], graph->layers[layer]);
#endif
        };
#ifdef USE_CUDA
        if (graph) graph->Dense(layer*3, dense); else
#endif
        dense();
        current = 1 - current;
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode == ExecutionMode::Hybrid && mlpInput.dataDevice == CUDA)
            expertSteps.Begin(FastllmCudaGetDevice(), weights[layer].data(),
                weights[layer].size(), num_experts_per_tok);
#endif
        // One route and expert invocation for every request segment. Cache
        // admission, dynamic routing and NUMA small-batch dispatch see all rows.
#ifdef USE_CUDA
        if (graph) graph->Experts(*this, layer*3+1, targetMoePlans[layer], graph->layers[layer],
            frames.front()->work->requestState->batchState->experts[layer]); else
#endif
        RunMoE(layer, mlpInput, mlpOutput, true, nullptr, nullptr, {},
            &frames.front()->work->requestState->batchState->experts[layer]);
        auto tail = [&] {
#ifdef USE_CUDA
        if (graph) {
            graph->Join(graph->layers[layer]);
            if (layer + 1 == pleLayer) {
                HyperCombine(next, mlpOutput, mlpInjection, graph->layers[layer].residual);
                return;
            }
            const std::string prefix = layer + 1 == block_cnt
                ? languagePrefix + "hyper_connection_mixer."
                : languagePrefix + "layers." + std::to_string(layer + 1) + ".attn_hyper_connection.";
            HyperRead(next,prefix,graph->layers[layer].readInput,
                layer+1==block_cnt?nullptr:&graph->layers[layer].readInjection,
                &mlpOutput,&mlpInjection,&graph->layers[layer].residual,layer+1==block_cnt?FLOAT32:dataType);
            return;
        }
#endif
        if (layer + 1 == block_cnt) {
            HyperRead(hidden[current],languagePrefix+"hyper_connection_mixer.",finalHidden,nullptr,
                &mlpOutput,&mlpInjection,&hidden[1-current]);
        } else if (layer + 1 != pleLayer &&
            SelectDeviceFromMap(deviceMap, layer + 1, block_cnt) ==
            SelectDeviceFromMap(deviceMap, layer + 2, block_cnt)) {
            HyperRead(hidden[current],languagePrefix+"layers."+std::to_string(layer+1)+".attn_hyper_connection.",
                carriedInput,&carriedInjection,&mlpOutput,&mlpInjection,&hidden[1-current],dataType);
            hasCarriedRead = true;
        } else {
            HyperCombine(hidden[current], mlpOutput, mlpInjection, hidden[1 - current]);
        }
        };
#ifdef USE_CUDA
        if (graph) graph->Dense(layer*3+2, tail); else
#endif
        tail();
        current = 1 - current;
    }
    auto head = [&] {
#ifdef USE_CUDA
        Data &headInput = graph ? graph->layers[block_cnt - 1].readInput : finalHidden;
        Data &output = graph ? graph->logits : logits;
#else
        Data &headInput = finalHidden, &output = logits;
#endif
        Data projected;
        Linear(headInput, weight["lm_head.weight"], Data(), projected);
        ToDataType(projected, output, FLOAT32);
    };
#ifdef USE_CUDA
    if (graph) {
        graph->Dense(block_cnt*3, head);
        const bool verifying = std::any_of(frames.begin(), frames.end(),
            [](const ForwardState *frame) { return frame->work->verificationCapture != nullptr; });
        graph->Seal(verifying ? "verify" : "decode", rows);
    } else
#endif
    head();
    }
#ifdef USE_CUDA
    if (graph) {
        // Retire the expert mailbox before sampling, as the single-request
        // graph does. Admissions retain their reader/event dependencies and
        // may overlap host sampling and preparation of the next request batch.
        const bool sampledVerification = std::any_of(frames.begin(), frames.end(), [](const ForwardState *frame) {
            return frame->work->verificationAccepted && !frame->work->generationConfig.IsSimpleGreedy();
        });
        if (sampledVerification) {
            for (auto *frame : frames)
                frame->work->samplingSubmitted = [&] { expertSteps.Finish(graph->device); };
        } else expertSteps.Finish(graph->device);
        FinishBatchGraph(*graph, frames, {});
        Qwen4BorrowCudaStorage(logits, graph->logits);
        if (std::any_of(frames.begin(), frames.end(), [](const ForwardState *frame) {
                return frame->work->targetMultiHidden != nullptr;
            }))
            Qwen4BorrowCudaStorage(hidden[0], graph->layers[block_cnt-1].residual);
    }
#endif
    std::vector<std::vector<int>> results;
    for (size_t i = 0; i < frames.size(); ++i) {
        auto &f = *frames[i]->work;
        const int sequence = f.inputIds.dims[1];
#ifdef USE_CUDA
        if (graph) {
            // Each sampler owns its metadata and optional CPU copy. The graph
            // owns the CUDA allocation until all rows have been consumed;
            // sampling/resetting EOS cannot touch another request's slice.
            Qwen4BorrowCudaStorage(f.logits, logits);
            f.logits.cudaData = static_cast<uint8_t*>(logits.cudaData) +
                size_t(starts[i]) * logits.dims.back() * logits.unitSize;
            f.logits.expansionDims.clear();
            f.logits.Resize({1, sequence, logits.dims.back()});
            f.logits.expansionSize = f.logits.Count(0);
            f.logits.expansionBytes = f.logits.GetBytes();
        } else
#endif
        Split(logits, 1, starts[i], starts[i] + sequence, f.logits);
        if (f.targetMultiHidden)
            Split(hidden[0], 1, starts[i], starts[i] + sequence, *f.targetMultiHidden);
        // Share sampling/EOS handling without running a second projection.
        f.logitsReady = true;
        results.push_back(frames[i]->Finish());
        f.samplingSubmitted = {};
    }
    return results;
}

void Qwen4ExpModel::ForwardMtpBatch(const std::vector<BatchedRequest*> &requests) {
    Qwen38ForwardScope dispatch;
    const bool persistent = std::all_of(requests.begin(), requests.end(),
        [](const auto *r) { return r->state->servingSlot != nullptr; });
    auto &pool = persistent ? servingBatchState : requests.front()->state->batchState;
    if (!pool) pool = std::make_shared<RequestBatchState>();
    auto batchState = pool;
    for (auto *r : requests) r->state->batchState = batchState;
    // All frames are stack-like RAII scopes. Destroy in reverse construction
    // order, including on exceptions, to restore thread-local dispatch state.
    struct Frames {
        std::vector<std::unique_ptr<ForwardState>> owned;
        ~Frames() { while (!owned.empty()) owned.pop_back(); }
    } storage;
    std::vector<ForwardState*> frames;
    std::vector<BatchedRequest*> drafts;
    for (auto *r : requests) {
        r->tokens = Qwen4HostInts(r->input);
        r->positions = Qwen4HostInts(r->position);
        if (r->verify) {
            auto &mtp = *r->state->mtpState;
            r->tokens.insert(r->tokens.end(), mtp.proposals.begin(), mtp.proposals.end());
            const int start = r->positions.front();
            r->positions.resize(r->tokens.size());
            std::iota(r->positions.begin(), r->positions.end(), start);
            r->input.CopyFrom(Data(FLOAT32, {1, int(r->tokens.size())},
                std::vector<float>(r->tokens.begin(), r->tokens.end())));
            r->position.CopyFrom(Data(FLOAT32, {1, int(r->positions.size())},
                std::vector<float>(r->positions.begin(), r->positions.end())));
            if (!mtp.targetCheckpointPrepared)
                CaptureTargetRuntimeCheckpoint(*r->cache, *r->state, mtp.targetCheckpoint);
            mtp.targetCheckpointPrepared = false;
            mtp.targetCapture.runtimeCheckpoint = &mtp.targetCheckpoint;
        }
        storage.owned.emplace_back(new ForwardState(*this, executionMode, 1,
            r->input, r->mask, r->position, *r->cache, *r->config, r->lastTokens, &r->logits,
            r->verify ? &r->hidden : nullptr, r->verify ? &r->verified : nullptr,
            r->verify ? &r->accepted : nullptr,
            r->verify ? &r->state->mtpState->targetCapture : nullptr,
            !r->verify, false, r->verify, &r->tokens, r->verify, nullptr, nullptr));
        frames.push_back(storage.owned.back().get());
    }
    auto outputs = ForwardTargetBatch(frames);
    // Restore the target's exact-projection/GDN scopes before draft execution,
    // just as the single-request ForwardTarget call does.
    frames.clear();
    while (!storage.owned.empty()) storage.owned.pop_back();
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &r = *requests[i];
        if (!r.verify) { r.output = outputs[i]; continue; }
        auto &mtp = *r.state->mtpState;
        int accepted = 0;
        AssertInFastLLM(r.config->IsSimpleGreedy() || r.accepted.size() == mtp.proposals.size(),
            "Qwen4 batched verifier returned an invalid acceptance shape.");
        while (accepted < (int)mtp.proposals.size() &&
            (r.config->IsSimpleGreedy() ? r.verified[accepted] == mtp.proposals[accepted]
                                       : r.accepted[accepted] != 0)) ++accepted;
        const int committed = accepted + 1;
        r.output.assign(mtp.proposals.begin(), mtp.proposals.begin() + accepted);
        r.output.push_back(r.verified[accepted]);
        if (committed < (int)r.tokens.size())
            CommitTargetVerificationPrefix(r.input, r.position, r.tokens, committed,
                *r.cache, *r.state, mtp.targetCheckpoint, mtp.targetCapture, false);
        if (committed < r.hidden.dims[1]) {
            Data prefix;
            Split(r.hidden, 1, 0, committed, prefix);
            r.hidden.CopyFrom(prefix);
        }
        CaptureTargetRuntimeCheckpoint(*r.cache, *r.state, mtp.targetCheckpoint,
            false, false, committed == (int)r.tokens.size());
        mtp.targetCheckpointPrepared = true;
        r.tokens = r.output;
        r.positions.resize(committed);
        mtp.pendingOutputTokens.assign(r.output.begin() + 1, r.output.end());
        drafts.push_back(&r);
    }
    if (!drafts.empty()) GenerateMtpProposalBatch(drafts);
#ifdef USE_CUDA
    // Publish request state, including speculative rollback, without draining
    // the independent expert-admission stream. Captured expert readers wait
    // on admissionDone; async cache uploads own their storage separately.
    FastllmCudaSyncCurrentThreadStream();
#endif
}

std::vector<int> Qwen4ExpModel::RunMtpDraftBatch(
        const std::vector<BatchedRequest*> &requests, const std::vector<Data*> &hidden,
        std::vector<Data> &nextHidden, std::vector<float> &probabilities) {
    ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
    std::vector<int> starts;
    std::vector<float> tokens;
    int rows = 0;
    for (auto *r : requests) {
        starts.push_back(rows); rows += r->tokens.size();
        tokens.insert(tokens.end(), r->tokens.begin(), r->tokens.end());
    }
    Data localHidden, localEmbeddings;
#ifdef USE_CUDA
    for (auto *r : requests) {
        Data position(FLOAT32, {1, (int)r->positions.size()},
            std::vector<float>(r->positions.begin(), r->positions.end()));
        r->position.CopyFrom(position);
    }
    RequestBatchState::Key graphKey;
    for (size_t i = 0; i < requests.size(); ++i) {
        graphKey.push_back(uint64_t(uintptr_t(requests[i]->state)));
        const auto &mtp = *requests[i]->state->mtpState;
        graphKey.push_back(std::max(hidden[i]->dims[1], mtp.proposals.empty() ? Qwen4MtpDraftsPerStep() + 1 : 1));
    }
    auto &ownedGraph = requests.front()->state->batchState->Select(
        requests.front()->state->mtpState->proposals.size(), graphKey, maxBatch);
    BatchGraph *graph = PrepareBatchGraph(ownedGraph, {}, requests, hidden) ? ownedGraph.get() : nullptr;
    auto &targetHidden = graph ? graph->hidden : localHidden;
    auto &embeddings = graph ? graph->embeddings : localEmbeddings;
#else
    constexpr bool graph = false;
    auto &targetHidden = localHidden; auto &embeddings = localEmbeddings;
#endif
#ifdef USE_CUDA
    if (graph) {
        rows = 0; tokens.clear(); starts.clear();
        for (size_t i = 0; i < requests.size(); ++i) {
            starts.push_back(rows); rows += graph->requests[i]->rows;
            auto &source = requests[i]->tokens;
            tokens.insert(tokens.end(), source.begin(), source.end());
            tokens.resize(rows, source.back());
        }
    }
#endif
    Data tokenIds(FLOAT32, {1, rows}, tokens);
    if (!graph) Qwen4BatchPack(hidden, targetHidden);
    RunMtpEmbedding(tokenIds, localEmbeddings);
#ifdef USE_CUDA
    if (graph) AssertInFastLLM(Qwen4PrepareDecodeGraphTensor(embeddings, localEmbeddings, graph->device),
        "Qwen batch draft embedding staging failed");
#endif
    Data localAfter, localInput, localInjection, localOutput, localMultiHidden, localLogits;
#ifdef USE_CUDA
    auto &afterAttention = graph ? graph->layers[block_cnt].afterAttention : localAfter;
    auto &mlpInput = graph ? graph->layers[block_cnt].input : localInput;
    auto &mlpInjection = graph ? graph->layers[block_cnt].injection : localInjection;
    auto &mlpOutput = graph ? graph->layers[block_cnt].output : localOutput;
    auto &multiHidden = graph ? graph->finalHidden : localMultiHidden;
    auto &logits = graph ? graph->logits : localLogits;
#else
    auto &afterAttention = localAfter; auto &mlpInput = localInput;
    auto &mlpInjection = localInjection; auto &mlpOutput = localOutput;
    auto &multiHidden = localMultiHidden; auto &logits = localLogits;
#endif
    PrepareMtpDraftLmHeadWeight();
    HybridMoeDecodeGroup expertStep;
#ifdef USE_CUDA
    if (graph && graph->complete) {
        expertStep.Begin(graph->device, draftMoePlan.weights->data(), draftMoePlan.weights->size(), num_experts_per_tok);
        AssertInFastLLM(graph->complete.Launch(), "Qwen cross-request draft graph replay failed");
    } else
#endif
    {
#ifdef USE_CUDA
    if (graph) graph->Begin();
#endif
    auto prefixBody = [&] {
    Data normalizedEmbeds, projectedEmbeds, normalizedHidden, projectedHidden;
    RMSNorm(embeddings, weight["mtp.pre_fc_norm_embedding.weight"], rms_norm_eps, normalizedEmbeds);
    Linear(normalizedEmbeds, weight["mtp.fc_embedding.weight"], Data(), projectedEmbeds);
    RMSNorm(targetHidden, weight["mtp.pre_fc_norm_hidden.weight"], rms_norm_eps, normalizedHidden);
    normalizedHidden.Reshape({1, rows * hcCount, embed_dim});
    Linear(normalizedHidden, weight["mtp.fc_hidden.weight"], Data(), projectedHidden);
    projectedHidden.Reshape({1, rows, hcCount, embed_dim});
    projectedEmbeds.Reshape({1, rows, 1, embed_dim});
    Qwen4CastLike(projectedEmbeds, projectedHidden);
    RepeatAddTo(projectedHidden, projectedEmbeds, 2, hcCount);
    projectedHidden.Reshape({1, rows, hcCount * embed_dim});
    const std::string prefix = "mtp.layers.0.";
    Data attentionInput, attentionInjection;
    HyperRead(projectedHidden,prefix+"attn_hyper_connection.",attentionInput,&attentionInjection,nullptr,nullptr,nullptr,dataType);
    Data typedAttention, projectionValues[3];
    Qwen4AttentionInput(attentionInput, typedAttention, dataType);
    FullAttentionProjections projections{&projectionValues[0], &projectionValues[1], &projectionValues[2]};
    std::vector<Data> projectedParts(requests.size()), attentionParts(requests.size()), injectionParts(requests.size());
    std::vector<Data*> projectedPointers, attentionPointers, injectionPointers;
    Data packedAttention;
    bool batchedQsa = false;
#ifdef USE_CUDA
    if (graph) {
        std::vector<FullAttentionGraphRequest> batch;
        for (size_t i = 0; i < requests.size(); ++i) {
            auto &mtp = *requests[i]->state->mtpState;
            auto &a = *graph->requests[i];
            batch.push_back({&mtp.key,&mtp.value,&mtp.attentionState,&a.attention,
                &a.attention.verification,a.rows});
        }
        batchedQsa = RunFullAttentionBatchGraph(block_cnt,prefix+"self_attn.",typedAttention,
            batch,projections,packedAttention,false);
    }
#endif
    if (!batchedQsa) ProjectFullAttention(prefix+"self_attn.",typedAttention,projections);
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &r = *requests[i];
        auto &mtp = *r.state->mtpState;
#ifdef USE_CUDA
        const int sequence = graph ? graph->requests[i]->rows : r.tokens.size();
#else
        const int sequence = r.tokens.size();
#endif
        Data part, output;
        Qwen4BatchView(part, attentionInput, starts[i], sequence);
        Data position;
        if (!graph) {
            Data hostPosition(FLOAT32, {1, sequence}, std::vector<float>(r.positions.begin(), r.positions.end()));
            position.CopyFrom(hostPosition); position.ToDevice(part.dataDevice, part.dataDeviceIds);
        }
        const int previous = mtp.key.dims.empty() ? 0 : mtp.key.dims[1];
        Data mask;
        if (!graph && sequence > 1 && !(executionMode == ExecutionMode::Hybrid &&
            part.dataDevice == CUDA && previous + sequence > indexerBudget))
            mask.CopyFrom(Qwen4BuildCausalMask(previous, sequence));
        Data slices[3], unprojected;
        for (int j = 0; j < 3; ++j)
            Qwen4BatchView(slices[j], projectionValues[j], starts[i], sequence);
        FullAttentionProjections prepared{&slices[0], &slices[1], &slices[2]};
#ifdef USE_CUDA
        if (batchedQsa) {
            Qwen4BatchView(unprojected,packedAttention,starts[i],sequence);
        } else if (graph) {
            auto &a = *graph->requests[i];
            AssertInFastLLM(RunFullAttentionGraph(block_cnt, prefix + "self_attn.", part,
                mtp.key, mtp.value, mtp.attentionState, &a.attention, graph->device, a.width,
                false, true, &a.attention.verification, output, &prepared, &unprojected),
                "Qwen batch draft graph attention failed");
        } else
#endif
        RunFullAttentionWithPrefix(block_cnt, prefix + "self_attn.", part, mask, position,
            true, mtp.key, mtp.value, mtp.attentionState, output, nullptr, 0, &prepared, &unprojected);
#ifdef USE_CUDA
        if (graph) {
            const auto *validRows = (const int32_t*)graph->requests[i]->attention.decodeMeta.cudaData + 1;
            Data projectedView, injectionView;
            Qwen4BatchView(projectedView, projectedHidden, starts[i], sequence);
            Qwen4BatchView(injectionView, attentionInjection, starts[i], sequence);
            AssertInFastLLM(FastllmCudaQwen4LastValidRow(projectedView, validRows, projectedParts[i]) &&
                FastllmCudaQwen4LastValidRow(injectionView, validRows, injectionParts[i]) &&
                FastllmCudaQwen4LastValidRow(unprojected, validRows, attentionParts[i]),
                "Qwen batch draft last-row gather failed");
        } else
#endif
        {
        Qwen4BatchView(projectedParts[i], projectedHidden, starts[i] + sequence - 1, 1);
        Qwen4BatchView(injectionParts[i], attentionInjection, starts[i] + sequence - 1, 1);
        Split(unprojected, 1, sequence - 1, sequence, attentionParts[i]);
        }
        projectedPointers.push_back(&projectedParts[i]);
        attentionPointers.push_back(&attentionParts[i]);
        injectionPointers.push_back(&injectionParts[i]);
    }
    Data projected, attention, injection;
    Qwen4BatchPack(projectedPointers, projected);
    Data unprojected;
    Qwen4BatchPack(attentionPointers, unprojected);
    Linear(unprojected, weight[prefix + "self_attn.o_proj.weight"], Data(), attention);
    Qwen4BatchPack(injectionPointers, injection);
    HyperRead(projected,prefix+"mlp_hyper_connection.",mlpInput,&mlpInjection,
        &attention,&injection,&afterAttention);
#ifdef USE_CUDA
    if (graph) graph->Route(*this, draftMoePlan, graph->layers[block_cnt]);
#endif
    };
#ifdef USE_CUDA
    if (graph) graph->Dense(0, prefixBody); else
#endif
    prefixBody();
#if defined(USE_CUDA) && !defined(USE_ROCM)
    if (mlpInput.dataDevice == CUDA)
        expertStep.Begin(FastllmCudaGetDevice(), draftMoePlan.weights->data(), draftMoePlan.weights->size(), num_experts_per_tok);
#endif
#ifdef USE_CUDA
    if (graph) graph->Experts(*this, 1, draftMoePlan, graph->layers[block_cnt],
        requests.front()->state->batchState->experts[block_cnt]); else
#endif
    RunMoE(draftMoePlan, mlpInput, mlpOutput, true, nullptr, nullptr, {},
        &requests.front()->state->batchState->experts[block_cnt]);
    auto tail = [&] {
#ifdef USE_CUDA
    if (graph) graph->Join(graph->layers[block_cnt]);
#endif
    Data sampleHidden, halfHeadInput, headLogits;
    HyperRead(afterAttention,"mtp.hyper_connection_mixer.",sampleHidden,nullptr,
        &mlpOutput,&mlpInjection,&multiHidden);
    Data *headInput = &sampleHidden;
    if (mtpDraftLmHeadReady && sampleHidden.dataType == FLOAT32) {
        ToDataType(sampleHidden, halfHeadInput, FLOAT16); headInput = &halfHeadInput;
    }
    Linear(*headInput, mtpDraftLmHeadReady ? mtpDraftLmHeadWeight : weight["lm_head.weight"], Data(), headLogits);
    ToDataType(headLogits, logits, FLOAT32);
    };
#ifdef USE_CUDA
    if (graph) { graph->Dense(2, tail); graph->Seal("draft", rows); } else
#endif
    tail();
    }
#ifdef USE_CUDA
    if (graph) FinishBatchGraph(*graph, {}, requests);
#endif
    std::vector<int> result(requests.size(), -1);
    nextHidden.resize(requests.size()); probabilities.assign(requests.size(), 1);
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &mtp = *requests[i]->state->mtpState;
        Split(multiHidden, 1, i, i + 1, nextHidden[i]);
        Data row;
        Qwen4BatchView(row, logits, i, 1);
#ifdef USE_CUDA
        if (mtp.sampleProposal) {
            SampleMtpDraftLogits(mtp, row, true, mtp.proposalCount);
            continue;
        }
#endif
        Data top;
        TopK(row, top, 1); top.ToDevice(CPU);
        auto *values = reinterpret_cast<float*>(top.cpuData);
        result[i] = int(values[0] + 1e-3f);
        if (Qwen4MtpMinProbability() > 0) probabilities[i] = Qwen4TopProbability(row, values[1]);
    }
#ifdef USE_CUDA
    // Queue sampling for the whole batch before any host readback.
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &mtp = *requests[i]->state->mtpState;
        if (mtp.sampleProposal)
            FastllmCudaCopyFromDeviceToHost(&result[i],
                (int*)mtp.sampledTokenIds.cudaData + mtp.proposalCount - 1, sizeof(int));
    }
#endif
    return result;
}

void Qwen4ExpModel::GenerateMtpProposalBatch(const std::vector<BatchedRequest*> &requests) {
    const int count = Qwen4MtpDraftsPerStep();
    std::vector<BatchedRequest*> active = requests;
    std::vector<Data*> hidden;
    std::map<BatchedRequest*, RequestRuntimeCheckpoint> checkpoints;
    std::map<BatchedRequest*, std::pair<int, int>> lengths;
    for (auto *r : active) {
        auto &mtp = *r->state->mtpState;
        mtp.proposals.clear(); mtp.proposalCount = 0;
        mtp.sampleProposal = !r->config->IsSimpleGreedy();
        mtp.proposalTemperature = r->config->temperature;
        hidden.push_back(&r->hidden);
    }
    std::vector<Data> previous;
    for (int step = 0; step < count && !active.empty(); ++step) {
        std::vector<Data> next;
        std::vector<float> probabilities;
        auto tokens = RunMtpDraftBatch(active, hidden, next, probabilities);
        std::vector<BatchedRequest*> remaining;
        hidden.clear();
        for (size_t i = 0; i < active.size(); ++i) {
            auto *r = active[i]; auto &mtp = *r->state->mtpState;
            if (step == 0 && count > 1) {
                lengths[r] = {mtp.key.dims[1], mtp.value.dims[1]};
                CaptureRequestRuntimeCheckpoint(mtp.attentionState,
                    {{block_cnt, mtp.key.dims[1]}}, checkpoints[r]);
            }
            if (!mtp.sampleProposal && probabilities[i] < Qwen4MtpMinProbability()) {
                if (step == 0) mtp.targetCheckpointPrepared = false;
                continue;
            }
            mtp.proposals.push_back(tokens[i]);
            r->tokens = {tokens[i]}; r->positions = {r->positions.back() + 1};
            remaining.push_back(r); hidden.push_back(&next[i]);
        }
        // Moving the vector preserves each Data object's address; views in
        // hidden remain valid until the next complete draft call has consumed it.
        previous = std::move(next);
        active = std::move(remaining);
    }
    for (auto &entry : checkpoints) {
        auto &mtp = *entry.first->state->mtpState;
        Qwen4ResizeKVCache(mtp.key, lengths[entry.first].first);
        Qwen4ResizeKVCache(mtp.value, lengths[entry.first].second);
        RestoreRequestRuntimeCheckpoint(mtp.attentionState, entry.second);
    }
}
