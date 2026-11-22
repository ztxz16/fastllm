// The graph binds stable device buffers. Page tables and request histories
// are passed indirectly for each replay, so context growth never changes the graph.
struct VerifyGraph {
    cudaGraphExec_t exec = nullptr;
    DecodeGraphScratch pool;
    Buffer<float> ids, panels;
    Matrix logits;
    std::vector<Matrix> taps;
    Buffer<DecodeVisit> source, visits;
    Buffer<int> accepted;
    std::vector<DecodeVisit> descriptors;
    ~VerifyGraph() { if (exec) cudaGraphExecDestroy(exec); }
};
std::map<int, std::unique_ptr<VerifyGraph>> verifyGraphs;

VerifyGraph *replayVerify(const std::vector<Segment> &sequences, const std::vector<float> &input,
                          Matrix &logits, std::vector<Matrix> &features, const float *deviceIds) {
    const int batch = sequences.size(), chain = c.draft.steps + 1, live = sequences.front().length;
    if (!model.SupportsDecodePipeline() || live > chain || int64_t(batch) * chain > 65535 ||
        attentionWidth(c.headDim) != c.headDim || c.relative > threads) return nullptr;
    const int n = batch * chain;
    auto found = verifyGraphs.find(batch);
    if (found != verifyGraphs.end() && !found->second) return nullptr;
    auto &entry = verifyGraphs[batch];
    const int slots = c.visits.size(), kv = c.kvHeads * c.headDim, depth = c.conv - 1;
    if (!entry) {
        try {
            auto p = std::make_unique<VerifyGraph>();
            p->ids = Buffer<float>(n);
            check(cudaMemsetAsync(p->ids.p, 0, size_t(n) * sizeof(float), cudaStreamPerThread));
            p->logits = Matrix(n, c.vocab);
            for (size_t i = 0; i < c.draft.taps.size(); ++i) p->taps.emplace_back(n, c.hidden);
            p->descriptors.resize(size_t(slots) * batch);
            p->source = Buffer<DecodeVisit>(p->descriptors.size());
            p->visits = Buffer<DecodeVisit>(size_t(slots) * n);
            p->accepted = Buffer<int>(batch);
            p->panels = Buffer<float>(p->descriptors.size() * (depth + chain) * 2 * (kv + c.hidden));
            entry = std::move(p);
        } catch (const CudaWorkspaceExhausted &e) {
            fprintf(stderr, "[Inkling] verify graph workspace unavailable for %d x %d: %s\n",batch,chain,e.what());
            return nullptr;
        }
    }
    auto &p = *entry;
    for (int b = 0; b < batch; ++b) {
        auto &req = *sequences[b].request;
        for (int slot = 0; slot < slots; ++slot) {
            auto &v = req.visits[slot];
            v.prepare(req.length + live, c.local[c.visits[slot]] ? c.window + live - 1 : 0);
            auto &item = p.descriptors[slot * batch + b];
            item.pages = v.view();
            item.past = req.length;
            item.verifyRemaining = live;
            for (int side = 0; side < 4; ++side) item.history[side] = v.convState[side].p;
        }
    }
    check(cudaMemcpyAsync(p.source.p, p.descriptors.data(), p.descriptors.size()*sizeof(DecodeVisit),
                          cudaMemcpyHostToDevice, cudaStreamPerThread));
    // A clipped last round reuses the same graph. Padding never appends to the KV cache.
    check(cudaMemcpy2DAsync(p.ids.p, chain * sizeof(float), deviceIds ? deviceIds : input.data(), live * sizeof(float),
                            live * sizeof(float), batch, deviceIds ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, cudaStreamPerThread));
    if (!p.exec) {
        std::vector<Segment> staged(batch);
        for (int b = 0; b < batch; ++b) staged[b] = {nullptr, b * chain, chain, nullptr, nullptr, p.visits.p};
        auto body = [&] {
            verifyHistories<false><<<dim3(slots * batch, 4), 256, 0, cudaStreamPerThread>>>(
                p.source.p, p.visits.p, p.panels.p, nullptr, batch, chain, depth, kv, c.hidden);
            std::vector<Matrix> states;
            target(p.ids.p, n, staged, p.logits, &states);
            for (size_t i = 0; i < states.size(); ++i)
                check(cudaMemcpyAsync(p.taps.at(i).p, states[i].p, size_t(n)*c.hidden*sizeof(float),
                                      cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        };
        try {
            p.pool.Warmup(body);
            p.pool.Capture(p.exec, body);
            printf("[Inkling] captured full target verification graph (%d requests x %d rows)\n",batch,chain);
        } catch (const std::exception &e) {
            entry.reset();
            fprintf(stderr, "[Inkling] eager verification for %d x %d: %s\n",batch,chain,e.what());
            return nullptr;
        }
    }
    check(cudaGraphLaunch(p.exec, cudaStreamPerThread));
    if (live == chain) {
        logits = p.logits;
    } else {
        logits = Matrix(batch * live, c.vocab);
        const size_t rowBytes = size_t(c.vocab) * sizeof(float);
        check(cudaMemcpy2DAsync(logits.p, live * rowBytes, p.logits.p, chain * rowBytes, live * rowBytes,
                                batch, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    features = p.taps;
    return &p;
}

void finishVerify(VerifyGraph &p, const std::vector<Segment> &sequences, const std::vector<int> &accepted) {
    const int batch = sequences.size(), chain = c.draft.steps + 1;
    check(cudaMemcpyAsync(p.accepted.p, accepted.data(), batch*sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
    verifyHistories<true><<<dim3(c.visits.size() * batch, 4), 256, 0, cudaStreamPerThread>>>(
        p.source.p, p.visits.p, p.panels.p, p.accepted.p, batch, chain, c.conv-1, c.kvHeads*c.headDim, c.hidden);
    for (int b = 0; b < batch; ++b) {
        auto &req = *sequences[b].request;
        for (size_t slot = 0; slot < c.visits.size(); ++slot)
            req.visits[slot].prepare(req.length + accepted[b] + 1, c.local[c.visits[slot]] ? c.window : 0);
    }
}
