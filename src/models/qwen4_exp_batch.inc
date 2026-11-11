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

struct Qwen4ExpModel::RequestBatchState {
    std::map<int, HybridMoeBatch> experts;
};

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
        r.cache = caches[i]; r.config = &generationConfigs[i]; r.resultIndex = i;
        Split(inputIds, 1, offset, offset + seqLens[i], r.input);
        offset += seqLens[i];
        r.position.CopyFrom(*positionIds[i]);
        if (attentionMasks[i]) r.mask.CopyFrom(*attentionMasks[i]);
        if (i < lastTokens.units.size()) r.lastTokens.units.push_back(lastTokens.units[i]);
        if (logits && i < logits->size()) r.logits.push_back((*logits)[i]);
        {
            std::lock_guard<std::mutex> guard(stateMutex);
            r.state = &requestStates[&r.cache->front().first];
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
        // Prefill, TP, multimodal/KVMem and reusable slots preserve their existing
        // admission/ownership path. Decode requests can have different positions,
        // accepted prefix lengths and sampling configurations in the same batch.
        if (seqLens[i] != 1 || threadTpState || threadTpRank >= 0 ||
            r.state->hasMultimodalInput || kvMemConfig.enabled || r.state->reusableGraphRequest || (usesMtp && !r.verify)) {
            result[i] = Forward(r.input, r.mask, r.position, *r.cache,
                *r.config, r.lastTokens, r.logits.empty() ? nullptr : r.logits[0]);
        } else {
            ready.push_back(&r);
        }
    }
    if (ready.size() == 1) {
        auto &r = *ready.front();
        result[r.resultIndex] = Forward(r.input, r.mask, r.position, *r.cache,
            *r.config, r.lastTokens, r.logits.empty() ? nullptr : r.logits[0]);
    } else if (!ready.empty()) {
        ForwardMtpBatch(ready);
        for (auto *r : ready) result[r->resultIndex] = r->output.front();
    }
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
    Data hidden[2], carriedNorm, finalNorm;
    HybridMoeDecodeGroup expertSteps;
    Qwen4BatchPack(initial, hidden[0]);
    int current = 0;
    bool hasCarriedNorm = false;
    for (int layer = 0; layer < block_cnt; ++layer) {
        ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
        auto &residual = hidden[current];
        auto &next = hidden[1 - current];
        if (layer == pleLayer) {
            for (size_t i = 0; i < frames.size(); ++i) {
                auto &f = *frames[i]->work;
                Data part, ple;
                Qwen4BatchView(part, residual, starts[i], f.inputIds.dims[1]);
                if (f.verificationCapture) f.verificationCapture->pleInput.CopyFrom(part);
                RunPLE(part, f.inputIds, *f.requestState, ple, f.hostInputTokens,
                    &f.plePrefetch, f.verificationCapture ? &f.verificationCapture->pleNormalized : nullptr);
                AddTo(part, ple);
            }
            hasCarriedNorm = false;
        }
        const std::string prefix = languagePrefix + "layers." + std::to_string(layer) + ".";
        const std::string attn = prefix + "attn_hyper_connection.";
        const std::string mlp = prefix + "mlp_hyper_connection.";
        Data norm, attentionInput, injection, attentionOutput;
        if (!hasCarriedNorm) GroupedRMSNorm(residual, weight[attn + "hc_norm.weight"], norm);
        HyperMixNormalized(hasCarriedNorm ? carriedNorm : norm, attn, attentionInput, &injection);
        hasCarriedNorm = false;
        Data typedInput, projectionValues[4];
        ToDataType(attentionInput, typedInput, dataType);
        const bool linearLayer = IsLinearAttentionLayer(layer);
        if (linearLayer) {
            LinearAttentionProjections projections{&projectionValues[0], &projectionValues[1],
                                                   &projectionValues[2], &projectionValues[3]};
            ProjectLinearAttention(layer, typedInput, projections);
        } else {
            FullAttentionProjections projections{&projectionValues[0], &projectionValues[1], &projectionValues[2]};
            ProjectFullAttention(prefix + "self_attn.", typedInput, projections);
        }
        std::vector<Data> attentionParts(frames.size());
        std::vector<Data*> parts;
        for (size_t i = 0; i < frames.size(); ++i) {
            auto &f = *frames[i]->work;
            auto *capture = f.verificationCapture;
            const int sequence = f.inputIds.dims[1];
            Data part;
            Qwen4BatchView(part, attentionInput, starts[i], sequence);
            auto &kv = f.pastKeyValues[layer];
            Data slices[4], unusedOutput;
            for (int j = 0; j < (linearLayer ? 4 : 3); ++j)
                Qwen4BatchView(slices[j], projectionValues[j], starts[i], sequence);
            if (linearLayer) {
                LinearAttentionProjections projections{&slices[0], &slices[1], &slices[2], &slices[3]};
                RunLinearAttention(layer, part, kv.first, kv.second, unusedOutput,
                    capture ? &capture->linearConvInputs[layer] : nullptr,
                    capture ? &capture->linearAlphas[layer] : nullptr,
                    capture ? &capture->linearBetas[layer] : nullptr,
                    capture && sequence > 1 && capture->runtimeCheckpoint
                        ? &capture->runtimeCheckpoint->linearSecond[layer] : nullptr,
                    &projections, &attentionParts[i]);
            } else {
                Data causal, positions;
                positions.CopyFrom(f.positionIds);
                positions.ToDevice(part.dataDevice, part.dataDeviceIds);
                const Data *mask = &f.attentionMask;
                if (sequence > 1 && mask->dims.empty() &&
                    !(part.dataDevice == CUDA && f.qsaPreviousLength >= indexerBudget)) {
                    causal.CopyFrom(Qwen4BuildCausalMask(f.qsaPreviousLength, sequence));
                    mask = &causal;
                }
                FullAttentionProjections projections{&slices[0], &slices[1], &slices[2]};
                RunFullAttention(layer, part, *mask, positions, f.qsaDeviceCompatibleMask,
                    kv.first, kv.second, *f.requestState, unusedOutput,
                    capture ? &capture->qsaRawKeys[layer] : nullptr, f.decodeReserveTokens,
                    &projections, &attentionParts[i]);
            }
            parts.push_back(&attentionParts[i]);
        }
        Data unprojected;
        Qwen4BatchPack(parts, unprojected);
        if (linearLayer)
            RunGdnOutputProjection(unprojected, weight[prefix + "linear_attn.out_proj.weight"], attentionOutput);
        else Linear(unprojected, weight[prefix + "self_attn.o_proj.weight"], Data(), attentionOutput);
        Data mlpNorm, mlpInput, mlpInjection, mlpOutput;
        HyperCombineRMSNorm(residual, attentionOutput, injection, weight[mlp + "hc_norm.weight"], next, mlpNorm);
        current = 1 - current;
        HyperMixNormalized(mlpNorm, mlp, mlpInput, &mlpInjection);
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (executionMode == ExecutionMode::Hybrid && mlpInput.dataDevice == CUDA)
            expertSteps.Begin(FastllmCudaGetDevice(), weights[layer].data(),
                weights[layer].size(), num_experts_per_tok);
#endif
        // One route and expert invocation for every request segment. Cache
        // admission, dynamic routing and NUMA small-batch dispatch see all rows.
        RunMoE(layer, mlpInput, mlpOutput, true, nullptr, nullptr, {},
            &frames.front()->work->requestState->batchState->experts[layer]);
        if (layer + 1 == block_cnt) {
            HyperCombineRMSNorm(hidden[current], mlpOutput, mlpInjection,
                weight[languagePrefix + "hyper_connection_mixer.hc_norm.weight"],
                hidden[1 - current], finalNorm);
        } else if (layer + 1 != pleLayer &&
            SelectDeviceFromMap(deviceMap, layer + 1, block_cnt) ==
            SelectDeviceFromMap(deviceMap, layer + 2, block_cnt)) {
            HyperCombineRMSNorm(hidden[current], mlpOutput, mlpInjection,
                weight[languagePrefix + "layers." + std::to_string(layer + 1) +
                       ".attn_hyper_connection.hc_norm.weight"], hidden[1 - current], carriedNorm);
            hasCarriedNorm = true;
        } else {
            HyperCombine(hidden[current], mlpOutput, mlpInjection, hidden[1 - current]);
        }
        current = 1 - current;
    }
    Data finalHidden, logits;
    HyperMixNormalized(finalNorm, languagePrefix + "hyper_connection_mixer.", finalHidden, nullptr);
    Linear(finalHidden, weight["lm_head.weight"], Data(), logits);
    ToDataType(logits, FLOAT32);
    std::vector<std::vector<int>> results;
    for (size_t i = 0; i < frames.size(); ++i) {
        auto &f = *frames[i]->work;
        const int sequence = f.inputIds.dims[1];
        Split(logits, 1, starts[i], starts[i] + sequence, f.logits);
        if (f.targetMultiHidden)
            Split(hidden[current], 1, starts[i], starts[i] + sequence, *f.targetMultiHidden);
        // Share sampling/EOS handling without running a second projection.
        f.logitsReady = true;
        results.push_back(frames[i]->Finish());
    }
    return results;
}

void Qwen4ExpModel::ForwardMtpBatch(const std::vector<BatchedRequest*> &requests) {
    Qwen38ForwardScope dispatch;
    auto batchState = requests.front()->state->batchState;
    if (!batchState) batchState = std::make_shared<RequestBatchState>();
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
                *r.cache, *r.state, mtp.targetCheckpoint, mtp.targetCapture);
        else CommitTargetRecurrentState(*r.cache, mtp.targetCheckpoint);
        if (committed < r.hidden.dims[1]) {
            Data prefix;
            Split(r.hidden, 1, 0, committed, prefix);
            r.hidden.CopyFrom(prefix);
        }
        CaptureTargetRuntimeCheckpoint(*r.cache, *r.state, mtp.targetCheckpoint, false, false);
        mtp.targetCheckpointPrepared = true;
        r.tokens = r.output;
        r.positions.resize(committed);
        mtp.pendingOutputTokens.assign(r.output.begin() + 1, r.output.end());
        drafts.push_back(&r);
    }
    if (!drafts.empty()) GenerateMtpProposalBatch(drafts);
#ifdef USE_CUDA
    ForceDeviceSync();
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
    Data targetHidden, embeddings, tokenIds(FLOAT32, {1, rows}, tokens);
    Qwen4BatchPack(hidden, targetHidden);
    Embedding(tokenIds, weight[languagePrefix + "embed_tokens.weight"], embeddings);
    Data normalizedEmbeds, projectedEmbeds, normalizedHidden, projectedHidden, attentionNorm;
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
    GroupedRMSNorm(projectedHidden, weight[prefix + "attn_hyper_connection.hc_norm.weight"], attentionNorm);
    HyperMixNormalized(attentionNorm, prefix + "attn_hyper_connection.", attentionInput, &attentionInjection);
    Data typedAttention, projectionValues[3];
    ToDataType(attentionInput, typedAttention, dataType);
    FullAttentionProjections projections{&projectionValues[0], &projectionValues[1], &projectionValues[2]};
    ProjectFullAttention(prefix + "self_attn.", typedAttention, projections);
    std::vector<Data> projectedParts(requests.size()), attentionParts(requests.size()), injectionParts(requests.size());
    std::vector<Data*> projectedPointers, attentionPointers, injectionPointers;
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &r = *requests[i];
        auto &mtp = *r.state->mtpState;
        const int sequence = r.tokens.size();
        Data part, output;
        Qwen4BatchView(part, attentionInput, starts[i], sequence);
        Data position(FLOAT32, {1, sequence}, std::vector<float>(r.positions.begin(), r.positions.end()));
        position.ToDevice(part.dataDevice, part.dataDeviceIds);
        const int previous = mtp.key.dims.empty() ? 0 : mtp.key.dims[1];
        Data mask;
        if (sequence > 1 && !(executionMode == ExecutionMode::Hybrid &&
            part.dataDevice == CUDA && previous + sequence > indexerBudget))
            mask.CopyFrom(Qwen4BuildCausalMask(previous, sequence));
        Data slices[3], unprojected;
        for (int j = 0; j < 3; ++j)
            Qwen4BatchView(slices[j], projectionValues[j], starts[i], sequence);
        FullAttentionProjections prepared{&slices[0], &slices[1], &slices[2]};
        RunFullAttentionWithPrefix(block_cnt, prefix + "self_attn.", part, mask, position,
            true, mtp.key, mtp.value, mtp.attentionState, output, nullptr, 0, &prepared, &unprojected);
        Qwen4BatchView(projectedParts[i], projectedHidden, starts[i] + sequence - 1, 1);
        Qwen4BatchView(injectionParts[i], attentionInjection, starts[i] + sequence - 1, 1);
        Split(unprojected, 1, sequence - 1, sequence, attentionParts[i]);
        projectedPointers.push_back(&projectedParts[i]);
        attentionPointers.push_back(&attentionParts[i]);
        injectionPointers.push_back(&injectionParts[i]);
    }
    Data projected, attention, injection, afterAttention, mlpNorm, mlpInput, mlpInjection, mlpOutput;
    Qwen4BatchPack(projectedPointers, projected);
    Data unprojected;
    Qwen4BatchPack(attentionPointers, unprojected);
    Linear(unprojected, weight[prefix + "self_attn.o_proj.weight"], Data(), attention);
    Qwen4BatchPack(injectionPointers, injection);
    HyperCombineRMSNorm(projected, attention, injection, weight[prefix + "mlp_hyper_connection.hc_norm.weight"],
        afterAttention, mlpNorm);
    HyperMixNormalized(mlpNorm, prefix + "mlp_hyper_connection.", mlpInput, &mlpInjection);
    RunMoE(draftMoePlan, mlpInput, mlpOutput, true, nullptr, nullptr, {},
        &requests.front()->state->batchState->experts[block_cnt]);
    Data multiHidden, finalNorm, sampleHidden, halfHeadInput, headLogits, logits;
    HyperCombineRMSNorm(afterAttention, mlpOutput, mlpInjection,
        weight["mtp.hyper_connection_mixer.hc_norm.weight"], multiHidden, finalNorm);
    HyperMixNormalized(finalNorm, "mtp.hyper_connection_mixer.", sampleHidden, nullptr);
    PrepareMtpDraftLmHeadWeight();
    Data *headInput = &sampleHidden;
    if (mtpDraftLmHeadReady && sampleHidden.dataType == FLOAT32) {
        ToDataType(sampleHidden, halfHeadInput, FLOAT16); headInput = &halfHeadInput;
    }
    Linear(*headInput, mtpDraftLmHeadReady ? mtpDraftLmHeadWeight : weight["lm_head.weight"], Data(), headLogits);
    ToDataType(headLogits, logits, FLOAT32);
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
