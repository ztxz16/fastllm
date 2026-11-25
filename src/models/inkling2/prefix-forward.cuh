// Included inside Decoder. Page ownership stays with the standard KV managers.
void setPrefixCache(bool enabled) override {
    std::lock_guard<std::mutex> lock(prefixMutex);
    prefixEnabled = enabled && RecurrentPrefixCacheEnabled();
    if (!prefixEnabled) prefixes = {};
}

int prefix(ResponseContext &ctx, int limit, bool restore) override {
    if (!prefixEnabled || limit <= 0 || !ctx.multimodalInput.empty()) return 0;
    std::lock_guard<std::mutex> lock(prefixMutex);
    const bool draft = c.draft.steps && speculativeSampling(ctx.generationConfig) &&
        ctx.currentTokens.size() <= size_t(c.draft.maxPositions - c.draft.block);
    if (!prefixes.Size()) return 0;
    using PageChain = std::pair<PagedCacheManager *, std::vector<int>>;
    std::vector<std::array<PageChain, 2>> chains(draft ? c.draft.layers : 0);
    int available = limit;
    for (size_t i = 0; i < chains.size(); ++i) {
        if (c.draft.local[i]) continue;
        for (int side = 0; side < 2; ++side) {
            auto &[pool, pages] = chains[i][side];
            pool = GetPagedCacheManager(2 * placement.cacheSlot(int(c.visits.size() + i)) + side);
            if (!pool) return 0;
            // The scheduler may have evicted the draft before the target.
            // Intersect its current pages before choosing a recurrent checkpoint.
            if (!ctx.prefillPreview) {
                ctx.prefillPreview = std::make_unique<PrefillPrefixPreview>();
                ctx.prefillPreview->Reset(ctx.currentTokens);
            }
            auto &preview = *ctx.prefillPreview;
            pool->QueryPrefix(preview.Hashes(pool->pageLen), preview.matches[pool], pages);
            available = std::min(available, int(pages.size()) * pool->pageLen);
        }
    }
    auto found = prefixes.Find(ctx.currentTokens, available, restore ? limit : -1,
        [&](const InklingPrefix &p) { return !draft || !p.draftContext.values.empty(); });
    if (!found) return 0;
    if (restore) {
        auto req = std::make_shared<Request>(placement.gpu);
        if (draft) req->draft = std::make_unique<DraftState>();
        for (auto &chain : chains) {
            auto kv = std::make_unique<DraftKV>();
            for (int side = 0; side < 2; ++side) {
                auto &[pool, pages] = chain[side];
                if (!pool) continue;
                pages.resize(found->cachedLen / pool->pageLen);
                kv->cache.bind(side ? kv->value : kv->key, pool, std::move(pages));
            }
            req->draft->kv.push_back(std::move(kv));
        }
        req->restored = found;
        req->length = found->cachedLen;
        req->prefixTokens = found->tokens;
        ctx.pastKeyValues.front().first.kvStorageOwner = std::move(req);
    }
    return found->cachedLen;
}

// Stop at exact recurrent-state boundaries. Keep at least one prompt row
// for fresh logits; every request crossing a boundary runs in one batch.
void checkpointPrompt(const float *&values, std::vector<Segment> &batch, int &rows,
                      const GenerationConfig &sampling, bool bf16, std::vector<float> &suffix) {
    const int page = GetPageLen(), interval = RecurrentPrefixInterval(page);
    for (;;) {
        std::vector<float> tokens;
        std::vector<Segment> prefix;
        std::vector<int> consumed(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto &s = batch[i];
            auto &req = *s.request;
            if (!req.prefixId || req.length >= req.promptLength || s.length < 2 ||
                size_t(req.length) != req.prefixTokens.size()) continue;
            const int last = (req.length + s.length - 1) / page * page;
            const int boundary = int(std::min<int64_t>(last, (int64_t(req.length) / interval + 1) * interval));
            if (boundary <= req.length) continue;
            const int n = boundary - req.length;
            consumed[i] = n;
            prefix.push_back({&req, int(tokens.size()), n});
            tokens.insert(tokens.end(), values + s.offset, values + s.offset + n);
        }
        if (prefix.empty()) return;
        std::vector<Matrix> features;
        const bool collect = std::any_of(prefix.begin(), prefix.end(), [](const Segment &s) { return bool(s.request->draft); });
        bool sampled = false;
        runTarget(tokens.data(), tokens.size(), prefix, sampling, sampled, collect ? &features : nullptr, false);
        storeDraftFeatures(prefix, features, bf16);
        for (const auto &s : prefix) {
            auto &req = *s.request;
            req.prefixTokens.insert(req.prefixTokens.end(), tokens.begin() + s.offset, tokens.begin() + s.offset + s.length);
            req.length += s.length;
            savePrefix(req);
        }
        std::vector<float> remaining;
        for (size_t i = 0; i < batch.size(); ++i) {
            auto &s = batch[i];
            const auto *first = values + s.offset;
            s.offset = int(remaining.size());
            remaining.insert(remaining.end(), first + consumed[i], first + s.length);
            s.length -= consumed[i];
        }
        suffix.swap(remaining);
        values = suffix.data();
        rows = int(suffix.size());
    }
}

void checkpointDecode(Request &req) {
    if (!req.prefixId || req.length < req.nextDecodeCheckpoint || req.length % GetPageLen() ||
        req.prefixTokens.size() != size_t(req.length)) return;
    savePrefix(req);
    req.nextDecodeCheckpoint = int64_t(req.length) + RecurrentPrefixInterval(GetPageLen());
}

void savePrefix(Request &req) {
    std::lock_guard<std::mutex> lock(prefixMutex);
    if (!prefixEnabled) return;
    if (!req.draft && prefixes.Find(req.prefixTokens, req.length, req.length,
            [](const InklingPrefix &p) { return !p.draftContext.values.empty(); })) return;
    auto entry = std::make_shared<InklingPrefix>();
    entry->cachedLen = req.length;
    entry->tokens = req.prefixTokens;
    entry->requestId = req.prefixId;
    entry->windows.resize(req.visits.size());
    entry->conv.resize(req.visits.size());
    std::vector<uint64_t> hashes;
    const int page = GetPageLen();
    for (int pos = 0; pos < req.length; pos += page)
        hashes.push_back(PagedCacheManager::HashTokenPage(entry->tokens.data() + pos, page));
    auto record = [&](VisitCache &v, bool local, PrefixWindow &window) {
        if (local) window.transfer(v, false);
        else for (auto *side : {v.key, v.value}) side->pagedKVCacheData->RecordPrefix(hashes, side->pageIndex);
    };
    for (size_t i = 0; i < req.visits.size(); ++i) {
        auto &v = req.visits[i];
        record(v, c.local[c.visits[i]], entry->windows[i]);
        for (int j = 0; j < 4; ++j) entry->conv[i][j].save(v.convState[j]);
    }
    if (req.draft) {
        entry->draftContext.save(req.draft->recurrentContext);
        entry->draftWindows.resize(c.draft.layers);
        for (int i = 0; i < c.draft.layers; ++i)
            record(req.draft->kv[i]->cache, c.draft.local[i], entry->draftWindows[i]);
    }
    check(cudaStreamSynchronize(cudaStreamPerThread));
    std::shared_ptr<InklingPrefix> held;
    prefixes.Publish(std::move(entry), held, RecurrentPrefixMaxPerRequest(), RecurrentPrefixMaxRecords(model.maxBatch));
}
