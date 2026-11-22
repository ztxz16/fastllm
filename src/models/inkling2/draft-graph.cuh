// One reusable proposal graph for the active sampling configuration. Cache
// addresses, sequence position and the anchor are rebound before each replay.
struct DraftGraph {
    cudaGraphExec_t exec = nullptr;
    DecodeGraphScratch pool;
    Buffer<DraftGraphKV> views;
    std::vector<DraftGraphKV> hostViews;
    Buffer<int> tokens;
    Matrix ids, probs, context;
    Data sampler;
    float temperature, topP;
    int topK;
    ~DraftGraph() { if (exec) cudaGraphExecDestroy(exec); }
};
std::unique_ptr<DraftGraph> draftGraph;
bool draftGraphUnavailable = false;

template <class T> DraftGraph *replayDraft(Request &req, int anchor, const GenerationConfig &cfg) {
    const auto &d = c.draft;
    if (draftGraphUnavailable || !model.SupportsDecodePipeline() || d.block > 16 ||
        d.heads / d.kvHeads > 64 / d.block || FastllmCudaRuntimeArch() < 80) return nullptr;
    if (draftGraph && (draftGraph->temperature != cfg.temperature || draftGraph->topK != cfg.top_k ||
                       draftGraph->topP != cfg.top_p)) return nullptr;
    // A clipped round uses the same graph. The recurrent chain is causal and
    // its unused suffix is discarded before verification.
    if (!draftGraph) {
        try {
            auto p = std::make_unique<DraftGraph>();
            p->views = Buffer<DraftGraphKV>(d.layers);
            p->hostViews.resize(d.layers);
            p->tokens = Buffer<int>(d.steps + 1);
            p->ids = Matrix(1, d.steps + 1);
            p->probs = Matrix(d.steps, c.vocab);
            p->context = Matrix(1, c.hidden);
            p->temperature = cfg.temperature;
            p->topK = cfg.top_k;
            p->topP = cfg.top_p;
            draftGraph = std::move(p);
        } catch (const CudaWorkspaceExhausted &e) {
            draftGraphUnavailable = true;
            fprintf(stderr, "[Inkling] eager draft: %s\n", e.what());
            return nullptr;
        }
    }
    auto &p = *draftGraph;
    for (int i = 0; i < d.layers; ++i) {
        const int past = req.length, begin = d.local[i] ? std::max(0, past - d.window) : 0;
        auto &kv = req.draft->kv[i];
        kv.reserve(begin, past, past + d.block, d.kvHeads, d.dim, d.local[i] ? d.window : 0, d.block);
        p.hostViews[i] = {kv.key.p, kv.value.p, past, kv.first, kv.capacity, begin};
    }
    check(cudaMemcpyAsync(p.views.p, p.hostViews.data(), d.layers * sizeof(DraftGraphKV),
                          cudaMemcpyHostToDevice, cudaStreamPerThread));
    check(cudaMemcpyAsync(p.context.p, req.draft->recurrentContext.p, c.hidden * sizeof(float),
                          cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    check(cudaMemcpyAsync(p.tokens.p, &anchor, sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
    if (!p.exec) {
        auto body = [&] {
            propose<T>(req, d.steps, cfg, p.probs, p.tokens.p, p.context, p.sampler, p.views.p);
            castProjection<<<(d.steps + 256) / 256, 256, 0, cudaStreamPerThread>>>(p.tokens.p, p.ids.p, d.steps + 1);
        };
        try {
            p.pool.Warmup(body);
            p.pool.Capture(p.exec, body);
            printf("[Inkling] captured full single-request draft graph (%d proposals)\n", d.steps);
        } catch (const std::exception &e) {
            cudaGetLastError();
            draftGraph.reset();
            draftGraphUnavailable = true;
            fprintf(stderr, "[Inkling] eager draft: %s\n", e.what());
            return nullptr;
        }
    }
    check(cudaGraphLaunch(p.exec, cudaStreamPerThread));
    return &p;
}
