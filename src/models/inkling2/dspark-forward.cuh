// Member functions of the Inkling decoder. Projections use the same native
// Linear as the target. The head-major cache is shared with DFlash attention.
Matrix draftAlpha;
Buffer<uint16_t> foldedHead, foldedBias;

template <class T> void foldDraftHead() {
    const int h = c.hidden, rank = c.draft.rank;
    const auto type = std::is_same_v<T, half> ? FLOAT16 : BFLOAT16;
    auto &head = weights.weight.at("model.llm.unembed.weight");
    auto &project = weights.weight.at("model.llm.draft.markov_head.hidden_out.weight");
    if (head.dims.size() != 2 || head.dims[0] < c.vocab || head.dims[1] != h ||
        project.dims != std::vector<int>{h, rank} || head.dataType != type || project.dataType != type)
        throw std::runtime_error("Invalid Inkling draft output projection");
    foldedHead = Buffer<uint16_t>(size_t(c.vocab) * rank);
    foldedBias = Buffer<uint16_t>(c.vocab);
    Data out(type, {c.vocab, rank}, CUDA, foldedHead.p);
    if (!FastllmCudaBatchMatMul(head, project, out, 0, 0, 0, h, rank, 1, c.vocab, h, rank, 1.f))
        throw std::runtime_error("Inkling draft head folding failed");
    Rows<T> bias(1, h);
    castProjection<<<(h + 255) / 256, 256, 0, cudaStreamPerThread>>>(param("draft.markov_head.hidden_out.bias", h),
                                                                     bias.p, h);
    Rows<T> folded;
    folded.p = reinterpret_cast<T *>(foldedBias.p);
    folded.rows = 1;
    folded.cols = c.vocab;
    projectInto(bias, "unembed.weight", folded, 0, 1);
}

void initDraft() {
    if (projectionType(placement.gpu) == BFLOAT16)
        foldDraftHead<__nv_bfloat16>();
    else
        foldDraftHead<half>();
    const auto &d = c.draft;
    int nt = d.taps.size();
    std::vector<float> alpha(size_t(d.layers + 1) * nt);
    check(cudaMemcpy(alpha.data(), param("draft.fuse_weight", alpha.size()), alpha.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    for (int i = 0; i <= d.layers; ++i) {
        auto first = alpha.begin() + i * nt;
        float m = *std::max_element(first, first + nt), sum = 0.f;
        for (int j = 0; j < nt; ++j)
            sum += first[j] = std::exp(first[j] - m);
        for (int j = 0; j < nt; ++j)
            first[j] /= sum;
    }
    draftAlpha = Matrix(d.layers + 1, nt);
    check(cudaMemcpy(draftAlpha.p, alpha.data(), alpha.size() * sizeof(float), cudaMemcpyHostToDevice));
}

template <class T> Rows<T> draftNorm(const Rows<T> &input, const std::string &key) {
    return castRows<T>(norm(castRows<float>(input), param(key, input.cols), c.draft.eps));
}

template <class T> void storeFeatures(Request &req, const std::vector<Matrix> &features, int offset, int rows) {
    auto &state = *req.draft;
    const auto &d = c.draft;
    const int nt = d.taps.size();
    std::vector<const float *> pointers;
    for (const auto &f : features)
        pointers.push_back(f.p);
    if (pointers.size() != size_t(nt))
        throw std::runtime_error("Missing Inkling draft features");
    Buffer<const float *> gpu(nt);
    check(cudaMemcpyAsync(gpu.p, pointers.data(), nt * sizeof(float *), cudaMemcpyHostToDevice, cudaStreamPerThread));
    for (int i = 0; i <= d.layers; ++i) {
        const bool last = i == d.layers;
        const bool local = !last && d.local[i];
        int count = last ? 1 : local ? std::min(rows, d.window) : rows;
        int skip = rows - count;
        Matrix mixed(count, c.hidden);
        draftMix<T><<<dim3(count, (c.hidden + 255) / 256), 256, 0, cudaStreamPerThread>>>(
            gpu.p, draftAlpha.p + i * nt, mixed.p, nt, c.hidden, offset + skip);
        std::string pre = last ? "draft.markov_head." : "draft.layers." + std::to_string(i) + ".";
        auto normalized = norm(mixed, param(pre + "dflare_norm.weight", c.hidden), d.eps);
        if (last) {
            state.recurrentContext = std::move(normalized);
            continue;
        }
        auto input = castRows<T>(normalized);
        auto k = linear(input, pre + "self_attn.k_proj_ctx.weight", d.kvHeads * d.dim);
        auto v = linear(input, pre + "self_attn.v_proj_ctx.weight", d.kvHeads * d.dim);
        int past = req.length + skip, end = past + count, begin = local ? std::max(0, end - d.window) : 0;
        auto &cache = state.kv[i];
        cache.reserve(begin, past, end + d.block, d.kvHeads, d.dim, local ? d.window : 0, d.block);
        dim3 grid(count, d.kvHeads);
        draftHead<<<grid, d.dim, 0, cudaStreamPerThread>>>(k.p, cache.key.p,
                                                           param(pre + "self_attn.k_norm_ctx.weight", d.dim), d.kvHeads,
                                                           d.dim, past, cache.first, cache.capacity, d.eps, d.theta);
        draftHead<<<grid, d.dim, 0, cudaStreamPerThread>>>(v.p, cache.value.p, nullptr, d.kvHeads, d.dim, past,
                                                           cache.first, cache.capacity, d.eps, d.theta);
    }
}

template <class T> Rows<T> draftAttention(const Rows<T> &input, Request &req, int layer, const std::string &pre,
                                                const DraftGraphKV *live) {
    const auto &d = c.draft;
    const int n = input.rows, h = d.dim, past = req.length;
    auto q = linear(input, pre + "q_proj.weight", d.heads * h);
    auto k = linear(input, pre + "k_proj.weight", d.kvHeads * h);
    auto v = linear(input, pre + "v_proj.weight", d.kvHeads * h);
    Buffer<half> query(size_t(n) * d.heads * h), out(size_t(n) * d.heads * h);
    if (live) {
        draftHead<T, 0><<<dim3(n, d.heads), h, 0, cudaStreamPerThread>>>(q.p, query.p, param(pre + "q_norm.weight", h),
            d.heads, h, 0, 0, n, d.eps, d.theta, live);
        draftHead<T, 1><<<dim3(n, d.kvHeads), h, 0, cudaStreamPerThread>>>(k.p, nullptr, param(pre + "k_norm.weight", h),
            d.kvHeads, h, 0, 0, 0, d.eps, d.theta, live);
        draftHead<T, 2><<<dim3(n, d.kvHeads), h, 0, cudaStreamPerThread>>>(v.p, nullptr, nullptr,
            d.kvHeads, h, 0, 0, 0, d.eps, d.theta, live);
        const int splits = d.local[layer] ? std::min(128, (d.window + n + 127) / 128) : 128;
        const int rowHeads = n * d.heads;
        Buffer<half> partial(size_t(rowHeads) * splits * h);
        Buffer<float> lse(size_t(rowHeads) * splits);
        Buffer<int> indptr(n + 1);
        DraftFlashParams params(query.p, nullptr, nullptr, nullptr, partial.p, lse.p, nullptr,
            d.heads, d.kvHeads, n, n, h, n * h, h, 0, h, -1, 0.f, 1.f / std::sqrt(float(h)), 1.f, 10000.f);
        draftGraphAttention<<<dim3((n * (d.heads / d.kvHeads) + 63) / 64, splits, d.kvHeads),
            dim3(32, 4), sizeof(DraftFlashTraits::SharedStorageSingle), cudaStreamPerThread>>>(params, live, indptr.p);
        check(flashinfer::VariableLengthMergeStatesDispatched<128>(partial.p, lse.p, indptr.p, out.p, nullptr,
            n, nullptr, d.heads, false, cudaStreamPerThread));
        Rows<half> packed;
        packed.p = out.p; packed.rows = n; packed.cols = d.heads * h;
        return linear(castRows<T>(packed), pre + "o_proj.weight", c.hidden);
    }
    auto &cache = req.draft->kv[layer];
    int begin = d.local[layer] ? std::max(0, past - d.window) : 0;
    int keys = past + n - begin;
    cache.reserve(begin, past, past + n, d.kvHeads, h, d.local[layer] ? d.window : 0, n);
    draftHead<<<dim3(n, d.heads), h, 0, cudaStreamPerThread>>>(q.p, query.p, param(pre + "q_norm.weight", h), d.heads,
                                                               h, past, past, n, d.eps, d.theta);
    draftHead<<<dim3(n, d.kvHeads), h, 0, cudaStreamPerThread>>>(k.p, cache.key.p, param(pre + "k_norm.weight", h),
                                                                 d.kvHeads, h, past, cache.first, cache.capacity, d.eps,
                                                                 d.theta);
    draftHead<<<dim3(n, d.kvHeads), h, 0, cudaStreamPerThread>>>(v.p, cache.value.p, nullptr, d.kvHeads, h, past,
                                                                 cache.first, cache.capacity, d.eps, d.theta);
    size_t off = size_t(begin - cache.first) * h;
    Data qv(FLOAT16, {d.heads, n, h}, CUDA, query.p), ov(FLOAT16, qv.dims, CUDA, out.p);
    Data kv(FLOAT16, {d.kvHeads, keys, h}, CUDA, cache.key.p + off), vv(FLOAT16, kv.dims, CUDA, cache.value.p + off);
    kv.strides[0] = vv.strides[0] = size_t(cache.capacity) * h;
    for (Data *p : {&qv, &kv, &vv, &ov})
        p->dataDeviceIds = {placement.gpu};
    // Crop local context once: every masked position attends the entire block.
    if (!FastllmCudaDFlashAttention(qv, kv, vv, ov, d.heads / d.kvHeads, 1.f / std::sqrt(float(h)), n, keys + n))
        throw std::runtime_error("Inkling draft block attention failed");
    Rows<T> merged(n, d.heads * h);
    draftUnpack<<<dim3(n, d.heads), h, 0, cudaStreamPerThread>>>(out.p, merged.p, n, d.heads, h);
    return linear(merged, pre + "o_proj.weight", c.hidden);
}

template <class T>
void propose(Request &req, int steps, const GenerationConfig &cfg, Matrix &probs, int *tokens,
             const Matrix &contextInput, Data &sampler, const DraftGraphKV *views) {
    const auto &d = c.draft;
    const int h = c.hidden, rank = d.rank, block = d.block;
    Rows<T> x(block, h);
    Matrix token(1, 1), emb(1, h);
    castProjection<<<1, 1, 0, cudaStreamPerThread>>>(tokens, token.p, 1);
    Data ids(FLOAT32, {1}, CUDA, token.p), view(FLOAT32, {1, h}, CUDA, emb.p);
    FastllmCudaEmbeddingDirect(ids, weight("embed.weight"), view);
    auto first = castRows<T>(norm(emb, param("embed_norm.weight", h), c.eps));
    check(cudaMemcpyAsync(x.p, first.p, h * sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    auto &mask = weight("draft.mask_embed");
    if (mask.Count(0) != h)
        throw std::runtime_error("Invalid Inkling draft mask embedding");
    castProjection<<<(h + 255) / 256, 256, 0, cudaStreamPerThread>>>(static_cast<float *>(mask.cudaData), x.p + h, h);
    for (int i = 2; i < block; ++i)
        check(cudaMemcpyAsync(x.p + i * h, x.p + h, h * sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    for (int i = 0; i < d.layers; ++i) {
        std::string pre = "draft.layers." + std::to_string(i) + ".";
        auto a = draftAttention(draftNorm(x, pre + "input_layernorm.weight"), req, i, pre + "self_attn.", views ? views + i : nullptr);
        draftAdd<<<(block * h + 255) / 256, 256, 0, cudaStreamPerThread>>>(x.p, a.p, block * h);
        auto in = draftNorm(x, pre + "post_attention_layernorm.weight");
        auto gate = linear(in, pre + "mlp.gate_proj.weight", d.intermediate);
        auto up = linear(in, pre + "mlp.up_proj.weight", d.intermediate);
        int size = block * d.intermediate;
        draftSwiGLU<<<(size + 255) / 256, 256, 0, cudaStreamPerThread>>>(gate.p, up.p, size);
        auto mlp = linear(gate, pre + "mlp.down_proj.weight", h);
        draftAdd<<<(block * h + 255) / 256, 256, 0, cudaStreamPerThread>>>(x.p, mlp.p, block * h);
    }
    auto hidden = draftNorm(x.slice(1, steps), "draft.norm.weight");
    auto base = linear(hidden, "unembed.weight", c.vocab);
    auto context = castRows<T>(contextInput);
    Rows<T> recurrent(1, 2 * rank + 2 * h), latent(1, rank);
    check(cudaMemsetAsync(recurrent.p, 0, rank * sizeof(T), cudaStreamPerThread));
    check(cudaMemcpyAsync(recurrent.p + 2 * rank + h, context.p, h * sizeof(T), cudaMemcpyDeviceToDevice,
                          cudaStreamPerThread));
    const std::string pre = "draft.markov_head.";
    const auto &w = weights.weight.at("model.llm." + pre + "markov_w1.weight");
    if (w.dims.size() != 2 || w.dims[0] < c.vocab || w.dims[1] != rank)
        throw std::runtime_error("Invalid Inkling Markov embedding");
    for (int i = 0; i < steps; ++i) {
        draftTokenInput<<<(rank + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            recurrent.p, static_cast<const T *>(w.cudaData), tokens + i, rank);
        check(cudaMemcpyAsync(recurrent.p + 2 * rank, hidden.p + size_t(i) * h, h * sizeof(T), cudaMemcpyDeviceToDevice,
                              cudaStreamPerThread));
        Matrix raw(1, 3 * rank);
        projectInto(recurrent, pre + "rnn_proj.weight", raw, 0, 1);
        draftRecur<<<(rank + 255) / 256, 256, 0, cudaStreamPerThread>>>(raw.p, param(pre + "rnn_proj.bias", 3 * rank),
                                                                        recurrent.p, latent.p, rank);
        constexpr auto type = std::is_same_v<T, half> ? FLOAT16 : BFLOAT16;
        Rows<T> adjust(1, c.vocab);
        Data l(type, {1, rank}, CUDA, latent.p), whead(type, {c.vocab, rank}, CUDA, foldedHead.p);
        Data out(type, {1, c.vocab}, CUDA, adjust.p);
        DoCudaLinear(l, whead, *GetEmptyData(), out);
        Matrix scores(1, c.vocab);
        draftScores<<<(c.vocab + 255) / 256, 256, 0, cudaStreamPerThread>>>(base.p + size_t(i) * c.vocab, adjust.p,
                                                                            reinterpret_cast<const T *>(foldedBias.p),
                                                                            scores.p, c.vocab, c.logitCap);
        if (!FastllmCudaMtpSampleDraftDeviceTopKFirst(scores.p, probs.p + size_t(i) * c.vocab, cfg.temperature, cfg.top_k,
                                             cfg.top_p, tokens + i + 1, c.vocab, sampler))
            throw std::runtime_error("Inkling draft sampling failed");
    }
}
