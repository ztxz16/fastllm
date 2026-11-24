void rejectDraft(RoundPlan &p, float *logits) {
    const int b = p.batch;
    float *probs;
    if (!FastllmCudaMtpGraphDistribution(p.targetSampler, logits, p.temperature.p + b, p.topK.p + b, p.topP.p + b,
                                         &probs))
        throw std::runtime_error("Inkling batched target normalization failed");
    FastllmCudaMtpGraphReject(probs, p.probs.p, p.draftIds.p, p.result.p, p.partials.p, b, p.steps, c.vocab, p.live.p,
                              p.rejectSeed, p.counter.p);
}

// Reject sampling and feature publication are part of target replay. A rejected
// suffix can occupy reserved pages but can never enter the committed prefix.
template <class T> void verifyBody(RoundPlan &p) {
    const int b = p.batch, chain = p.chain, rows = b * chain, slots = c.visits.size();
    const int kv = c.kvHeads * c.headDim, h = c.hidden, depth = c.conv - 1;
    int *accepted = p.result.p + rows;
    verifyHistories<false><<<dim3(slots * b, 4), 256, 0, cudaStreamPerThread>>>(p.source.p, p.visits.p, p.panels.p,
                                                                                nullptr, b, chain, depth, kv, h);
    std::vector<Segment> segments(b);
    for (int i = 0; i < b; ++i) segments[i] = {nullptr, i * chain, chain, nullptr, nullptr, p.visits.p};
    Matrix logits(rows, c.vocab);
    std::vector<Matrix> features;
    target(p.ids.p, rows, segments, logits, &features);
    rejectDraft(p, logits.p);
    verifyHistories<true><<<dim3(slots * b, 4), 256, 0, cudaStreamPerThread>>>(p.source.p, p.visits.p, p.panels.p,
                                                                               accepted, b, chain, depth, kv, h);

    const auto &d = c.draft;
    const int count = features.size();
    if (count != int(d.taps.size())) throw std::runtime_error("Missing Inkling verification features");
    using cuda_attn_res::Sources;
    Buffer<const float *> taps(count);
    for (int i = 0; i < count; i += Sources::Inline) {
        Sources src{};
        int n = std::min(count - i, int(Sources::Inline));
        for (int j = 0; j < n; ++j) src.items[j] = features[i + j].p;
        cuda_attn_res::StoreSources<<<1, Sources::Inline, 0, cudaStreamPerThread>>>(taps.p + i, src, n);
    }
    for (int layer = 0; layer <= d.layers; ++layer) {
        bool last = layer == d.layers;
        std::string pre = last ? "draft.markov_head." : "draft.layers." + std::to_string(layer) + ".";
        Matrix mixed(rows, h);
        draftMix<T><<<dim3(rows, (h + 255) / 256), 256, 0, cudaStreamPerThread>>>(taps.p, draftAlpha.p + layer * count,
                                                                                  mixed.p, count, h, 0);
        auto normalized = norm(mixed, param(pre + "dflare_norm.weight", h), d.eps);
        if (last) {
            storeDraftContext<<<dim3((h + 255) / 256, b), 256, 0, cudaStreamPerThread>>>(normalized.p, p.views.p,
                                                                                         accepted, chain, h);
        } else {
            auto input = castRows<T>(normalized);
            auto k = linear(input, pre + "self_attn.k_proj_ctx.weight", d.kvHeads * d.dim);
            auto v = linear(input, pre + "self_attn.v_proj_ctx.weight", d.kvHeads * d.dim);
            dim3 grid(rows, d.kvHeads);
            draftHead<T, 1><<<grid, d.dim, 0, cudaStreamPerThread>>>(
                k.p, nullptr, param(pre + "self_attn.k_norm_ctx.weight", d.dim), d.kvHeads, d.dim, d.eps, d.theta,
                chain, {}, p.views.p + layer * b, accepted);
            draftHead<T, 2><<<grid, d.dim, 0, cudaStreamPerThread>>>(
                v.p, nullptr, nullptr, d.kvHeads, d.dim, d.eps, d.theta, chain, {}, p.views.p + layer * b, accepted);
        }
    }
}

void uploadRound(RoundPlan &p) {
    auto upload = [](auto *dest, const auto &src) {
        check(
            cudaMemcpyAsync(dest, src.data(), src.size() * sizeof(*dest), cudaMemcpyHostToDevice, cudaStreamPerThread));
    };
    upload(p.source.p, p.descriptors);
    upload(p.views.p, p.draftViews);
    upload(p.inputs.p, p.hostInputs);
    std::generate(p.hostSeeds.begin(), p.hostSeeds.end(), newSamplingSeed);
    upload(p.seeds.p, p.hostSeeds);
}

// Warm against private, initialized pages. No real KV, histories, or request
// RNG state participates in capture. Every address used after capture is indirect.
template <class T> void captureRound(RoundPlan &p) {
    int b = p.batch, len = std::max(c.draft.block, p.chain), h = c.hidden;
    int width = std::max(c.kvHeads * c.headDim, c.draft.kvHeads * c.draft.dim);
    Buffer<float> key(size_t(b) * len * width), value(size_t(b) * len * width);
    Buffer<int2> pageIds(b);
    Buffer<float> histories(size_t(b) * 4 * std::max(1, c.conv - 1) * std::max(h, width)), context(size_t(b) * h);
    auto clear = [](auto &x, size_t n) { check(cudaMemsetAsync(x.p, 0, n * sizeof(*x.p), cudaStreamPerThread)); };
    clear(key, size_t(b) * len * width);
    clear(value, size_t(b) * len * width);
    clear(context, size_t(b) * h);
    clear(histories, size_t(b) * 4 * std::max(1, c.conv - 1) * std::max(h, width));
    std::vector<int2> ids(b);
    for (int i = 0; i < b; ++i) {
        ids[i] = make_int2(i, i);
        PageView pages{key.p, value.p, pageIds.p + i, 0, len, pageShift(len)};
        for (int l = 0; l < c.draft.layers; ++l) p.draftViews[l * b + i] = {pages, context.p + size_t(i) * h, 0, 0};
        for (int l = 0; l < int(c.visits.size()); ++l) {
            auto &v = p.descriptors[l * b + i];
            v.pages = pages;
            v.past = 0;
            v.verifyRemaining = p.chain;
            for (int s = 0; s < 4; ++s)
                v.history[s] = histories.p + size_t(4 * i + s) * std::max(1, c.conv - 1) * std::max(h, width);
        }
    }
    check(cudaMemcpyAsync(pageIds.p, ids.data(), b * sizeof(int2), cudaMemcpyHostToDevice, cudaStreamPerThread));
    std::fill(p.hostInputs.begin(), p.hostInputs.end(), DraftInput{0, 40, p.steps, 1.f, .95f});
    uploadRound(p);
    auto draft = [&] { draftBody<T>(p); };
    auto verify = [&] { verifyBody<T>(p); };
    p.draftPool.Warmup(draft);
    p.draftPool.Capture(p.draftExec, draft);
    p.verifyPool.Warmup(verify);
    p.verifyPool.Capture(p.verifyExec, verify);
    printf("[Inkling] captured full draft and verification graphs (%d requests x %d rows)\n", b, p.chain);
}

bool timelineCompatible(int batch) const {
    return attentionWidth(c.headDim) == c.headDim && c.relative <= threads &&
           int64_t(batch) * (c.draft.steps + 1) <= 65535;
}

template <class T> RoundPlan &roundPlan(int batch) {
    auto &plan = roundPlans[batch];
    if (!plan) {
        auto entry = std::make_unique<RoundPlan>(batch, c);
        entry->draftPool.ShareStorage(roundScratch);
        entry->verifyPool.ShareStorage(roundScratch);
        if (GetFastllmEnv().cudaGraph && timelineCompatible(batch)) {
            try {
                captureRound<T>(*entry);
            } catch (const std::exception &error) {
                // A completed draft graph can still feed an eager verifier.
                cudaGetLastError();
                check(cudaStreamSynchronize(cudaStreamPerThread));
                fprintf(stderr, "[Inkling] %s for %d requests: %s\n",
                        entry->draftExec ? "eager verification with captured draft" : "eager DSpark",
                        batch, error.what());
            }
        }
        plan = std::move(entry);
    }
    return *plan;
}

// Unpadded target head geometries use the ordinary attention implementation.
template <class T> void verifyUnaligned(RoundPlan &p, const std::vector<Segment> &batch) {
    const int b = p.batch;
    std::vector<Verification> commits(b);
    std::vector<Segment> segments;
    Matrix input(b, p.chain);
    int rows = 0;
    for (int i = 0; i < b; ++i) {
        int width = p.hostInputs[i].steps + 1;
        segments.push_back({batch[i].request, rows, width, nullptr, &commits[i]});
        check(cudaMemcpyAsync(input.p + rows, p.ids.p + i * p.chain, width * sizeof(float), cudaMemcpyDeviceToDevice,
                              cudaStreamPerThread));
        rows += width;
    }
    Matrix logits(rows, c.vocab), padded(b * p.chain, c.vocab);
    std::vector<Matrix> features;
    target(input.p, rows, segments, logits, &features);
    size_t pitch = size_t(c.vocab) * sizeof(float);
    check(cudaMemsetAsync(padded.p, 0, size_t(b) * p.chain * pitch, cudaStreamPerThread));
    for (int i = 0; i < b; ++i) {
        const auto &seg = segments[i];
        check(cudaMemcpyAsync(padded.p + size_t(i) * p.chain * c.vocab, logits.p + size_t(seg.offset) * c.vocab,
                              seg.length * pitch, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    rejectDraft(p, padded.p);
    std::vector<int> accepted(b);
    check(cudaMemcpy(accepted.data(), p.result.p + b * p.chain, b * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < b; ++i) {
        if (accepted[i] < 0 || accepted[i] > p.hostInputs[i].steps)
            throw std::runtime_error("Invalid Inkling accepted prefix");
        for (auto &write : commits[i].writes) write(accepted[i] + 1);
        storeFeatures<T>(*batch[i].request, features, segments[i].offset, accepted[i] + 1);
    }
}

PageTableBatch roundPageTables;

template <class T>
std::vector<int> verifyRound(const std::vector<Segment> &batch, const float *anchors,
                             const std::vector<GenerationConfig> &sampling) {
    const auto start = std::chrono::steady_clock::now();
    const int b = batch.size(), slots = c.visits.size();
    auto &p = roundPlan<T>(b);
    std::vector<VisitCache *> pendingTables;
    pendingTables.reserve(size_t(b) * (slots + c.draft.layers));

    for (int i = 0; i < b; ++i) {
        auto &req = *batch[i].request;
        const auto &cfg = sampling[i];
        const int steps = std::clamp(req.limit - req.length - 2, 0, p.steps);
        p.hostInputs[i] = {int(anchors[batch[i].offset]), cfg.top_k, steps, cfg.temperature, cfg.top_p};
        for (int l = 0; l < c.draft.layers; ++l) {
            auto &kv = req.draft->kv[l]->cache;
            bool local = c.draft.local[l];
            kv.reservePages(req.length + c.draft.block, local ? c.draft.window + c.draft.block : 0);
            p.draftViews[l * b + i] = {kv.view(&pendingTables), req.draft->recurrentContext.p, req.length,
                                       local ? std::max(0, req.length - c.draft.window) : 0};
        }
        for (int l = 0; l < slots; ++l) {
            auto &v = req.visits[l];
            v.reservePages(req.length + steps + 1, c.local[c.visits[l]] ? c.window + steps : 0);
            auto &dest = p.descriptors[l * b + i];
            dest.pages = v.view(&pendingTables);
            dest.past = req.length;
            dest.verifyRemaining = steps + 1;
            for (int s = 0; s < 4; ++s) dest.history[s] = v.convState[s].p;
        }
    }
    roundPageTables.upload(pendingTables);
    uploadRound(p);
    if (p.draftExec) check(cudaGraphLaunch(p.draftExec, cudaStreamPerThread));
    else draftBody<T>(p);
    if (p.verifyExec) launchTargetGraph(p.verifyExec, p.verifyPool);
    else if (timelineCompatible(b)) verifyBody<T>(p);
    else verifyUnaligned<T>(p, batch);
    std::vector<int> host(b * (p.chain + 1));
    check(cudaMemcpyAsync(host.data(), p.result.p, host.size() * sizeof(int), cudaMemcpyDeviceToHost,
                          cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<int> result(b);
    for (int i = 0; i < b; ++i) {
        auto &req = *batch[i].request;
        auto &state = *req.draft;
        int accepted = host[b * p.chain + i];
        if (accepted < 0 || accepted > p.hostInputs[i].steps)
            throw std::runtime_error("Invalid Inkling DSpark accepted length");
        auto *out = host.data() + i * p.chain;
        for (int j = 0; j <= accepted; ++j)
            if (out[j] < 0 || out[j] >= c.vocab) throw std::runtime_error("Invalid Inkling DSpark token");
        if (req.prefixId && req.prefixTokens.size() == size_t(req.length)) {
            req.prefixTokens.push_back(p.hostInputs[i].anchor);
            req.prefixTokens.insert(req.prefixTokens.end(), out, out + accepted);
        }
        req.length += accepted + 1;
        for (int l = 0; l < slots; ++l) req.visits[l].prepare(req.length, c.local[c.visits[l]] ? c.window : 0);
        for (int l = 0; l < c.draft.layers; ++l)
            state.kv[l]->cache.prepare(req.length, c.draft.local[l] ? c.draft.window : 0);
        checkpointDecode(req);
        for (int j = 0; j < accepted; ++j) state.pending.emplace_back(out[j], out[j + 1]);
        result[i] = out[0];
        if (model.verbose) {
            ++state.rounds;
            state.proposed += p.hostInputs[i].steps;
            state.accepted += accepted;
            state.roundMs +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
    }
    return result;
}
