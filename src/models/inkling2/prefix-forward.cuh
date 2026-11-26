// Included inside Decoder. Page ownership stays with the standard KV managers.
void setPrefixCache(bool enabled) override {
    std::lock_guard<std::mutex> lock(prefixMutex);
    prefixEnabled = enabled && RecurrentPrefixCacheEnabled();
    if (!prefixEnabled) {
        prefixes = {};
        prefixBuffers.reset();
    }
}

int prefixChunkSize() const {
    const int chunk = model.GetChunkedPrefillSize();
    return chunk > 0 ? chunk : std::numeric_limits<int>::max();
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
    const int page = GetPageLen();
    const int prompt = int(ctx.currentTokens.size()), tail = (prompt - 1) / page * page;
    const int chunk = prefixChunkSize(), interval = RecurrentPrefixInterval(page);
    auto found = prefixes.Find(ctx.currentTokens, available, restore ? limit : -1,
        [&](const InklingPrefix &p) {
            if ((draft && !p.hasDraft()) || (p.tail && p.tail != tail) ||
                p.chunk != chunk || p.interval != interval) return false;
            // A restored suffix may span several external scheduler chunks.
            // Its shifted chunk ends must still land on the cold prompt grid.
            if (prompt - p.cachedLen > chunk && p.cachedLen % chunk &&
                (chunk % interval || p.cachedLen % interval)) return false;
            return p.pagesCurrent(draft);
        });
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
    const int chunk = prefixChunkSize();
    for (;;) {
        std::vector<float> tokens;
        std::vector<Segment> prefix;
        std::vector<int> consumed(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto &s = batch[i];
            auto &req = *s.request;
            if (!req.prefixId || req.length >= req.promptLength || s.length < 2 ||
                size_t(req.length) != req.prefixTokens.size()) continue;
            const int end = req.length + s.length;
            auto nextMultiple = [&](int step) { return (int64_t(req.length) / step + 1) * step; };
            // Anchor both grids at token zero so a restored suffix follows the
            // cold prompt's GEMM shapes, including the last partial interval.
            int64_t boundary = std::min(nextMultiple(interval), nextMultiple(chunk));
            const int tail = (req.promptLength - 1) / page * page;
            if (tail > req.length) boundary = std::min<int64_t>(boundary, tail);
            if (boundary >= end) continue;
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
        features.clear();
        for (const auto &s : prefix) {
            auto &req = *s.request;
            req.prefixTokens.insert(req.prefixTokens.end(), tokens.begin() + s.offset, tokens.begin() + s.offset + s.length);
            req.length += s.length;
            if (req.length % page == 0) savePrefix(req);
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

fastkv::Buffer acquirePrefixBuffer(size_t bytes, const InklingPrefix &entry) {
    // Keep the same per-rank ceiling in the auto planner. Allocate lazily so a
    // short conversation only pays for the window pages it actually retains.
    const size_t limit = (size_t(256) << 20) * std::clamp(model.maxBatch, 4, 16) / placement.world;
    if (bytes <= limit) {
        if (!prefixBuffers) {
            auto allocate = [](size_t size, int gpu) {
                size_t free, total;
                check(cudaMemGetInfo(&free, &total));
                const size_t reserve = size_t(total * (1.0 - std::clamp(double(GetGpuMemRatio()), 0.0, 1.0)));
                if (size > free || free - size < reserve)
                    throw fastkv::CapacityBusy("Inkling prefix would consume GPU headroom");
                return fastkv::Buffer::Allocate(size, gpu);
            };
            prefixBuffers = std::make_unique<fastkv::BufferPool>(limit, std::map<int, size_t>{}, allocate);
        }
        for (;;) {
            try {
                return prefixBuffers->Acquire(bytes, placement.gpu);
            } catch (const fastkv::CapacityBusy &) {
                if (!prefixes.EvictIdle(entry, [](const InklingPrefix &p) { return p.storage.Device() >= 0; },
                                        [](const InklingPrefix &p) { return !p.tail; })) break;
            }
        }
    }
    // Oversized checkpoints or pressure from active readers must still allow
    // an exact cache hit on memory-constrained deployments.
    return fastkv::Buffer::Allocate(bytes);
}

void savePrefix(Request &req) {
    std::lock_guard<std::mutex> lock(prefixMutex);
    if (!prefixEnabled) return;
    const int page = GetPageLen();
    const int chunk = prefixChunkSize(), interval = RecurrentPrefixInterval(page);
    int requiredTail = 0;
    if (req.length <= req.promptLength) {
        const int tail = (req.promptLength - 1) / page * page;
        // The physical end of a page-aligned prompt may be off both grids.
        // A longer cold prefill will not reproduce that short final GEMM.
        const auto regular = [&](int position) { return position % interval == 0 || position % chunk == 0; };
        if (req.length != tail && !regular(req.length)) return;
        if (!regular(tail)) {
            if (req.length > tail) return;
            if (req.length == tail) requiredTail = tail;
        }
    }
    if (!req.draft && prefixes.Find(req.prefixTokens, req.length, req.length,
            [&](const InklingPrefix &p) {
                return p.hasDraft() && p.tail == requiredTail && p.chunk == chunk && p.interval == interval &&
                       p.pagesCurrent(true);
            })) return;
    auto entry = std::make_shared<InklingPrefix>();
    entry->cachedLen = req.length;
    entry->tail = requiredTail;
    entry->interval = interval;
    entry->chunk = chunk;
    entry->tokens = req.prefixTokens;
    entry->requestId = req.prefixId;
    entry->windows.resize(req.visits.size());
    entry->conv.resize(req.visits.size());
    std::vector<uint64_t> hashes;
    for (int pos = 0; pos < req.length; pos += page)
        hashes.push_back(PagedCacheManager::HashTokenPage(entry->tokens.data() + pos, page));
    PrefixCopies copies;
    auto record = [&](VisitCache &v, int slot, bool local, PrefixWindow &window, std::vector<PrefixPageGenerations> &pages) {
        if (local) window.transfer(v, copies, false);
        else for (auto *side : {v.key, v.value}) {
            side->pagedKVCacheData->RecordPrefix(hashes, side->pageIndex);
            pages.emplace_back(*side, 2 * placement.cacheSlot(slot) + int(side == v.value), hashes.size());
        }
    };
    for (size_t i = 0; i < req.visits.size(); ++i) {
        auto &v = req.visits[i];
        record(v, int(i), c.local[c.visits[i]], entry->windows[i], entry->targetPages);
        for (int j = 0; j < 4; ++j) entry->conv[i][j].save(v.convState[j], copies);
    }
    if (req.draft) {
        entry->draftContext.save(req.draft->recurrentContext, copies);
        entry->draftWindows.resize(c.draft.layers);
        for (int i = 0; i < c.draft.layers; ++i)
            record(req.draft->kv[i]->cache, int(c.visits.size()) + i, c.draft.local[i], entry->draftWindows[i], entry->draftPages);
    }
    entry->storage = acquirePrefixBuffer(copies.bytes, *entry);
    copies.submit(entry->storage, false);
    std::shared_ptr<InklingPrefix> held;
    prefixes.Publish(std::move(entry), held, RecurrentPrefixMaxPerRequest(), RecurrentPrefixMaxRecords(model.maxBatch));
}
