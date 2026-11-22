// Member functions of the Inkling decoder. Projections use the same native
// Linear as the target. Draft and target use request-owned handles into the common page manager.
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
        auto &cache = state.kv[i]->cache;
        cache.prepare(end, local ? d.window : 0);
        DraftView view{cache.view(), nullptr, past, begin};
        dim3 grid(count, d.kvHeads);
        draftHead<T, 1>
            <<<grid, d.dim, 0, cudaStreamPerThread>>>(k.p, nullptr, param(pre + "self_attn.k_norm_ctx.weight", d.dim),
                                                      d.kvHeads, d.dim, d.eps, d.theta, count, view, nullptr);
        draftHead<T, 2><<<grid, d.dim, 0, cudaStreamPerThread>>>(v.p, nullptr, nullptr, d.kvHeads, d.dim, d.eps,
                                                                 d.theta, count, view, nullptr);
    }
}

int draftAttentionWarps = 0;

template<int Warps> void launchDraftAttention(DraftFlashParams args, int batch, int kvHeads) {
    using Traits = DraftFlashTraits<Warps>;
    const int tiles = (args.block * int(args.group_size) + Traits::CTA_TILE_Q - 1) / Traits::CTA_TILE_Q;
    const int groups = batch * tiles * kvHeads;
    const int splits = std::min(args.maxSplits, std::max(1, (8 * smCount + groups - 1) / groups));
    draftGraphAttention<Traits><<<dim3(batch * tiles, splits, kvHeads), dim3(32, Warps),
        sizeof(typename Traits::SharedStoragePaged), cudaStreamPerThread>>>(args);
}

template <class T>
Rows<T> draftAttention(const Rows<T> &input, int layer, const std::string &pre, const DraftView *views) {
    const auto &d = c.draft;
    const int n = input.rows, dim = d.dim, block = d.block, batch = n / block;
    auto q = linear(input, pre + "q_proj.weight", d.heads * dim);
    auto k = linear(input, pre + "k_proj.weight", d.kvHeads * dim);
    auto v = linear(input, pre + "v_proj.weight", d.kvHeads * dim);
    Buffer<half> query(size_t(n) * d.heads * dim);
    draftHead<T, 0><<<dim3(n, d.heads), dim, 0, cudaStreamPerThread>>>(q.p, query.p, param(pre + "q_norm.weight", dim),
                                                                       d.heads, dim, d.eps, d.theta, block, {}, views);
    draftHead<T, 1><<<dim3(n, d.kvHeads), dim, 0, cudaStreamPerThread>>>(
        k.p, nullptr, param(pre + "k_norm.weight", dim), d.kvHeads, dim, d.eps, d.theta, block, {}, views);
    draftHead<T, 2><<<dim3(n, d.kvHeads), dim, 0, cudaStreamPerThread>>>(v.p, nullptr, nullptr, d.kvHeads, dim, d.eps,
                                                                         d.theta, block, {}, views);
    const int splits = d.local[layer] ? std::min(128, (d.window + block + 127) / 128) : 128;
    Buffer<half> partial(size_t(n) * splits * d.heads * dim);
    Buffer<float> lse(size_t(n) * splits * d.heads);
    DraftFlashParams args{};
    args.views = views;
    args.q = query.p;
    args.o = partial.p;
    args.lse = lse.p;
    args.num_qo_heads = d.heads;
    args.group_size = d.heads / d.kvHeads;
    args.q_stride_n = d.heads * dim;
    args.q_stride_h = dim;
    args.block = block;
    args.window_left = -1;
    args.sm_scale = 1.f / std::sqrt(float(dim));
    args.partition_kv = true;
    args.maxSplits = splits;
    if (draftAttentionWarps == 2) {
        launchDraftAttention<2>(args, batch, d.kvHeads);
    } else if (draftAttentionWarps == 4) {
        launchDraftAttention<4>(args, batch, d.kvHeads);
    } else {
        draftAttentionPartial<<<dim3(n, splits, d.heads), dim, 0, cudaStreamPerThread>>>(args, views, d.kvHeads);
    }
    Rows<T> joined(n, d.heads * dim);
    mergeDraftAttention<<<dim3(n, d.heads), dim, 0, cudaStreamPerThread>>>(args, views, joined.p, splits);
    return linear(joined, pre + "o_proj.weight", c.hidden);
}
