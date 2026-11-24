// Both graphs bind the same request-major chain. Only descriptors, sampling
// parameters and fresh RNG seeds are uploaded between replays.
struct RoundPlan {
    int batch, steps, chain;
    Buffer<DraftView> views;
    Buffer<DraftInput> inputs;
    Buffer<float> temperature, topP, ids;
    Buffer<int> topK, tokens, draftIds, live, result;
    Buffer<uint64_t> seeds;
    Buffer<unsigned long long> counter;
    Buffer<uint8_t> partials;
    Matrix probs;
    void *sampler = nullptr, *targetSampler = nullptr;
    uint64_t rejectSeed = newSamplingSeed();
    cudaGraphExec_t draftExec = nullptr, verifyExec = nullptr;
    DecodeGraphScratch draftPool, verifyPool;
    Buffer<DecodeVisit> source, visits;
    Buffer<float> panels;
    RoundHostVector<DraftInput> hostInputs;
    RoundHostVector<uint64_t> hostSeeds;
    RoundHostVector<DraftView> draftViews;
    RoundHostVector<DecodeVisit> descriptors;
    RoundPlan(int b, const Config &c)
        : batch(b), steps(c.draft.steps), chain(steps + 1), views(b * c.draft.layers), inputs(b),
          temperature(b * chain + b), topP(b * chain + b), ids(b * chain), topK(b * chain + b), tokens(b * chain),
          draftIds(b * steps), live(b), result(b * (chain + 1)), seeds(steps), counter(1),
          partials(FastllmCudaMtpGraphRejectBytes(b, c.vocab)), probs(b * steps, c.vocab), source(b * c.visits.size()),
          visits(b * chain * c.visits.size()),
          panels(size_t(b) * c.visits.size() * (c.conv - 1 + chain) * 2 * (c.kvHeads * c.headDim + c.hidden)),
          hostInputs(b), hostSeeds(steps), draftViews(b * c.draft.layers), descriptors(b * c.visits.size()) {
        check(cudaMemsetAsync(counter.p, 0, sizeof(*counter.p), cudaStreamPerThread));
        sampler = FastllmCudaMtpCreateTargetDistribution(b, c.vocab);
        targetSampler = FastllmCudaMtpCreateTargetDistribution(b * chain, c.vocab);
        if (!sampler || !targetSampler) {
            FastllmCudaMtpDestroyTargetDistribution(sampler);
            FastllmCudaMtpDestroyTargetDistribution(targetSampler);
            throw CudaWorkspaceExhausted("Insufficient sampling workspace for Inkling DSpark");
        }
    }
    ~RoundPlan() {
        if (draftExec) cudaGraphExecDestroy(draftExec);
        if (verifyExec) cudaGraphExecDestroy(verifyExec);
        FastllmCudaMtpDestroyTargetDistribution(sampler);
        FastllmCudaMtpDestroyTargetDistribution(targetSampler);
    }
};
DecodeGraphScratch roundScratch;
std::map<int, std::unique_ptr<RoundPlan>> roundPlans;

template <class T> void draftBody(RoundPlan &p) {
    const auto &d = c.draft;
    const int b = p.batch, h = c.hidden, n = b * d.block, rank = d.rank, steps = p.steps;
    Matrix anchors(b, 1), emb(b, h);
    stageDraftInputs<<<b, 128, 0, cudaStreamPerThread>>>(p.inputs.p, anchors.p, p.tokens.p, p.temperature.p, p.topK.p,
                                                         p.topP.p, p.live.p, b, p.chain, c.vocab);
    Data ids(FLOAT32, {b}, CUDA, anchors.p), view(FLOAT32, {b, h}, CUDA, emb.p);
    FastllmCudaEmbeddingDirect(ids, weight("embed.weight"), view);
    auto first = castRows<T>(norm(emb, param("embed_norm.weight", h), c.eps));
    Rows<T> x(n, h);
    draftBlockInput<<<dim3((h + 255) / 256, n), 256, 0, cudaStreamPerThread>>>(first.p, param("draft.mask_embed", h),
                                                                               x.p, d.block, h);
    for (int i = 0; i < d.layers; ++i) {
        std::string pre = "draft.layers." + std::to_string(i) + ".";
        auto a = draftAttention(draftNorm(x, pre + "input_layernorm.weight"), i, pre + "self_attn.", p.views.p + i * b);
        draftAdd<<<(n * h + 255) / 256, 256, 0, cudaStreamPerThread>>>(x.p, a.p, n * h);
        auto in = draftNorm(x, pre + "post_attention_layernorm.weight");
        auto gate = linear(in, pre + "mlp.gate_proj.weight", d.intermediate);
        auto up = linear(in, pre + "mlp.up_proj.weight", d.intermediate);
        int size = n * d.intermediate;
        draftSwiGLU<<<(size + 255) / 256, 256, 0, cudaStreamPerThread>>>(gate.p, up.p, size);
        auto mlp = linear(gate, pre + "mlp.down_proj.weight", h);
        draftAdd<<<(n * h + 255) / 256, 256, 0, cudaStreamPerThread>>>(x.p, mlp.p, n * h);
    }
    Rows<T> selected(b * steps, h);
    check(cudaMemcpy2DAsync(selected.p, size_t(steps) * h * sizeof(T), x.p + h, size_t(d.block) * h * sizeof(T),
                            size_t(steps) * h * sizeof(T), b, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    auto hidden = draftNorm(selected, "draft.norm.weight");
    auto base = linear(hidden, "unembed.weight", c.vocab, 1, Projection::DraftHead);
    const int pitch = 2 * rank + 2 * h;
    Rows<T> recurrent(b, pitch), latent(b, rank);
    const std::string pre = "draft.markov_head.";
    const auto &w = weights.weight.at("model.llm." + pre + "markov_w1.weight");
    if (w.dims.size() != 2 || w.dims[0] < c.vocab || w.dims[1] != rank)
        throw std::runtime_error("Invalid Markov embedding");
    for (int i = 0; i < steps; ++i) {
        draftMarkovInput<<<dim3((std::max(h, rank) + 255) / 256, b), 256, 0, cudaStreamPerThread>>>(
            recurrent.p, static_cast<const T *>(w.cudaData), p.tokens.p, hidden.p, p.views.p, i, steps, rank, h);
        Matrix raw(b, 3 * rank);
        projectInto(recurrent, pre + "rnn_proj.weight", raw, 0, 1);
        draftRecur<<<dim3((rank + 255) / 256, b), 256, 0, cudaStreamPerThread>>>(
            raw.p, param(pre + "rnn_proj.bias", 3 * rank), recurrent.p, latent.p, rank, pitch);
        constexpr auto type = std::is_same_v<T, half> ? FLOAT16 : BFLOAT16;
        Rows<T> adjust(b, c.vocab);
        Data l(type, {b, rank}, CUDA, latent.p), head(type, {c.vocab, rank}, CUDA, foldedHead.p);
        Data out(type, {b, c.vocab}, CUDA, adjust.p);
        bool projected = false;
        if constexpr (std::is_same_v<T, __nv_bfloat16>)
            projected = bf16Mma && b > 1 && bf16Mma->draftLogits(latent.p, foldedHead.p, adjust.p, b, rank, c.vocab);
        if (!projected) DoCudaLinear(l, head, *GetEmptyData(), out);
        if (!FastllmCudaMtpGraphDrawMarkov(p.sampler, base.p + size_t(i) * c.vocab, steps * c.vocab, adjust.p,
                                           foldedBias.p, c.logitCap, type == BFLOAT16, p.temperature.p, p.topK.p,
                                           p.topP.p, p.seeds.p + i, 0, p.probs.p + size_t(i) * c.vocab,
                                           steps * c.vocab, p.tokens.p + i + 1, p.chain))
            throw std::runtime_error("Inkling batched draft sampling failed");
    }
    castProjection<<<(b * p.chain + 255) / 256, 256, 0, cudaStreamPerThread>>>(p.tokens.p, p.ids.p, b * p.chain);
    check(cudaMemcpy2DAsync(p.draftIds.p, steps * sizeof(int), p.tokens.p + 1, p.chain * sizeof(int),
                            steps * sizeof(int), b, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
}
