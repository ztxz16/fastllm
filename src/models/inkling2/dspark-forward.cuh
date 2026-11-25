// Member functions of the Inkling decoder. Projections use the same native
// Linear as the target. Draft and target use request-owned handles into the common page manager.
Matrix draftAlpha;
Buffer<uint16_t> foldedHead, foldedBias;
const Data *quantizedFold = nullptr;
const float *quantizedFoldBias = nullptr;

int draftVocabRows() const { return placement.world > 1 ? placement.vocabRows : c.vocab; }

template <class T> void foldDraftHead() {
    const int h = c.hidden, rank = c.draft.rank, vocab = draftVocabRows();
    const auto type = std::is_same_v<T, half> ? FLOAT16 : BFLOAT16;
    auto &head = weights.weight.at("model.llm.unembed.weight");
    auto &project = weights.weight.at("model.llm.draft.markov_head.hidden_out.weight");
    auto saved = weights.weight.find("model.llm.draft.markov_head.folded.weight");
    if (saved != weights.weight.end()) {
        if (!q8Projection(saved->second) || saved->second.dims != std::vector<int>{vocab, rank} ||
            !q8Projection(head)) throw std::runtime_error("Invalid quantized Inkling draft head");
        quantizedFold = &saved->second;
        quantizedFoldBias = param("draft.markov_head.folded.bias", vocab);
        return;
    }
    if (head.dims.size() != 2 || head.dims[0] < vocab || head.dims[1] != h ||
        project.dims != std::vector<int>{h, rank} || head.dataType != type || project.dataType != type)
        throw std::runtime_error("Invalid Inkling draft output projection");
    foldedHead = Buffer<uint16_t>(size_t(vocab) * rank);
    foldedBias = Buffer<uint16_t>(vocab);
    CudaView out(type, {vocab, rank}, foldedHead.p);
    if (!FastllmCudaBatchMatMul(head, project, out, 0, 0, 0, h, rank, 1, vocab, h, rank, 1.f))
        throw std::runtime_error("Inkling draft head folding failed");
    Rows<T> bias(1, h);
    castProjection<<<(h + 255) / 256, 256, 0, cudaStreamPerThread>>>(param("draft.markov_head.hidden_out.bias", h),
                                                                     bias.p, h);
    Rows<T> folded;
    folded.p = reinterpret_cast<T *>(foldedBias.p);
    folded.rows = 1;
    folded.cols = vocab;
    projectInto(bias, "unembed.weight", folded, 0, 1);
}

void initDraft() {
    if (placement.denseType == BFLOAT16)
        foldDraftHead<__nv_bfloat16>();
    else
        foldDraftHead<half>();
    const auto &d = c.draft;
    int nt = d.taps.size();
    if (weights.weight.count("model.llm.draft.fuse_prob")) {
        draftAlpha.rows = d.layers + 1;
        draftAlpha.cols = nt;
        draftAlpha.p = const_cast<float *>(param("draft.fuse_prob", (d.layers + 1) * nt));
        return;
    }
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

template<auto Kernel, class... Args> static void launchDraft(dim3 grid, dim3 block, size_t shared, Args... args) {
    if (!cuda_launch::Launch<Kernel>(grid, block, shared, cudaStreamPerThread, args...))
        throw std::runtime_error("Inkling draft kernel launch failed");
}

template<class T> Rows<T> draftGraphNorm(Rows<T> &x, const std::string &key, const Rows<T> &delta = {}) {
    Rows<T> out(x.rows, x.cols);
    auto run = [&](auto add) {
        launchDraft<draftResidualNorm<T, decltype(add)::value>>(x.rows, threads, 0, x.p, delta.p,
            param(key, x.cols), out.p, x.cols, c.draft.eps);
    };
    if (delta.p) run(std::true_type{}); else run(std::false_type{});
    return out;
}

// cuBLAS reads its batch pointer arrays on the device. Install them once during
// graph warmup; the arrays belong to this round plan for the graph's lifetime.
struct DraftProjectionGroup {
    std::array<void *, 6> pointers{};
    void **device = nullptr;
    bool ready = false;
};

bool draftBf16GateUp(const Rows<__nv_bfloat16> &input, const std::array<const Data *, 2> &w,
                     const std::array<void *, 2> &out, int columns, DraftProjectionGroup &plan) {
    auto *pool = DecodeGraphScratch::active;
    if (!pool || !plan.device || !decodePdlSupported) return false;
    const std::array<void *, 6> pointers{w[0]->cudaData, w[1]->cudaData, input.p, input.p, out[0], out[1]};
    if (pool->recording()) {
        if (!plan.ready || plan.pointers != pointers) return false;
    } else {
        plan.pointers = pointers;
        check(cudaMemcpyAsync(plan.device, plan.pointers.data(), sizeof(pointers), cudaMemcpyHostToDevice,
                              cudaStreamPerThread));
    }
    const float one = 1.f, zero = 0.f;
    auto status = cublasGemmBatchedEx(getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, columns,
            input.rows, input.cols, &one, plan.device, CUDA_R_16BF, input.cols, plan.device + 2,
            CUDA_R_16BF, input.cols, &zero, plan.device + 4, CUDA_R_16BF, columns, 2,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) return false;
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Inkling grouped BF16 gate/up failed");
    plan.ready = true;
    return true;
}

// The graph owns the individual result buffers. Sharing a GEMV grid requires
// neither concatenating the checkpoint weights nor splitting the output afterward.
template<class T, size_t N>
std::array<Rows<T>, N> draftProjections(const Rows<T> &input, const std::string &pre,
        const std::array<const char *, N> &names, const std::array<int, N> &cols, DraftProjectionGroup *group = nullptr,
        std::shared_ptr<FastllmBf16ProjectionFork> *fork = nullptr) {
    std::array<Rows<T>, N> out;
    std::array<const Data *, N> weightsIn;
    std::array<void *, N> outputs;
    bool fullWidths = true;
    for (size_t i = 0; i < N; ++i) {
        const auto &w = weights.weight.at("model.llm." + pre + names[i]);
        fullWidths &= w.dims.size() == 2 && w.dims[0] == cols[i] && w.dims[1] == input.cols;
        weightsIn[i] = &w;
        out[i] = Rows<T>(input.rows, cols[i]);
        outputs[i] = out[i].p;
    }
    if constexpr (std::is_same_v<T, __nv_bfloat16> && N <= 2) {
        if (bf16Mma && fullWidths && input.rows >= smCount) {
            for (size_t i = 0; i < N; ++i) {
                const auto &w = *weightsIn[i];
                if (w.dataType != BFLOAT16 || w.IsRepacked || !w.cudaData ||
                    !FastllmCudaBf16HeuristicGemm(input.p, w.cudaData, outputs[i],
                                                  input.rows, input.cols, cols[i]))
                    projectInto(input, pre + names[i], out[i], 0, 1);
            }
            return out;
        }
    }
    if constexpr (std::is_same_v<T, __nv_bfloat16> && N > 1) {
        const bool native = fullWidths && DecodeGraphScratch::active && decodePdlSupported &&
            std::all_of(weightsIn.begin(), weightsIn.end(), [](const Data *w) {
                return w->dataType == BFLOAT16 && w->cudaData && !w->IsRepacked;
            });
        if constexpr (N == 2) {
            if (native && group && cols[0] == cols[1] && draftBf16GateUp(input, weightsIn, outputs, cols[0], *group))
                return out;
        } else if constexpr (N == 3) {
            if (native && fork) {
                auto project = [&](int j) { projectInto(input, pre + names[j], out[j], 0, 1); };
                bool recording = DecodeGraphScratch::active->recording();
                if (!recording) project(0); // Warm the same Q algorithm as the serial path.
                bool parallel = FastllmCudaBf16ParallelProjection(input.p, weightsIn[0]->cudaData, out[0].p,
                      input.rows, input.cols, cols[0], *fork, [&] { project(1); project(2); });
                if (!parallel) {
                    if (recording) project(0);
                    project(1);
                    project(2);
                }
                return out;
            }
        }
    }
    if (!fullWidths || !FastllmCudaNVFP4SiblingProjections(input.p, std::is_same_v<T, __nv_bfloat16>, input.rows, input.cols,
                                            weightsIn.data(), outputs.data(), N))
        for (size_t i = 0; i < N; ++i) projectInto(input, pre + names[i], out[i], 0, 1);
    return out;
}

template<int Warps> void launchDraftAttention(DraftFlashParams args, int batch, int kvHeads) {
    using Traits = DraftFlashTraits<Warps>;
    const int tiles = (args.block * int(args.group_size) + Traits::CTA_TILE_Q - 1) / Traits::CTA_TILE_Q;
    const int groups = batch * tiles * kvHeads;
    const int splits = std::min(args.maxSplits, std::max(1, (8 * smCount + groups - 1) / groups));
    launchDraft<draftGraphAttention<Traits>>(dim3(batch * tiles, splits, kvHeads), dim3(32, Warps),
        sizeof(typename Traits::SharedStoragePaged), args);
}

template <class T>
Rows<T> draftAttention(const Rows<T> &input, int layer, const std::string &pre, const DraftView *views, std::shared_ptr<FastllmBf16ProjectionFork> *fork, bool privateSuffix) {
    const auto &d = c.draft;
    const int n = input.rows, dim = d.dim, block = d.block, batch = n / block;
    auto qkv = draftProjections<T, 3>(input, pre, {"q_proj.weight", "k_proj.weight", "v_proj.weight"},
                                     {d.heads * dim, d.kvHeads * dim, d.kvHeads * dim}, nullptr, fork);
    Buffer<half> query(size_t(n) * d.heads * dim);
    const size_t suffixElements = privateSuffix ? size_t(n) * d.kvHeads * dim : 0;
    Buffer<half> suffix(2 * suffixElements);
    half *valueSuffix = suffix.p ? suffix.p + suffixElements : nullptr;
    launchDraft<draftPrepareQkv<T>>(dim3(n, d.heads + 2 * d.kvHeads), dim, 0, qkv[0].p, qkv[1].p, qkv[2].p,
        query.p, param(pre + "q_norm.weight", dim), param(pre + "k_norm.weight", dim), d.heads, d.kvHeads,
        dim, d.eps, d.theta, block, views, suffix.p, valueSuffix);
    int splits = d.local[layer] ? std::min(128, (d.window + block + 127) / 128) : 128;
    if (privateSuffix && batch * d.kvHeads >= smCount) {
        // All KV heads already cover the device. Assign each CTA its own
        // contiguous partition instead of looping over short partial tiles.
        splits = std::min(splits, (8 * smCount + batch * d.kvHeads - 1) / (batch * d.kvHeads));
    }
    Buffer<half> partial(size_t(n) * splits * d.heads * dim);
    Buffer<float> lse(size_t(n) * splits * d.heads);
    DraftFlashParams args{};
    args.views = views;
    args.keySuffix = suffix.p;
    args.valueSuffix = valueSuffix;
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
        launchDraft<draftAttentionPartial>(dim3(n, splits, d.heads), dim, 0, args, views, d.kvHeads);
    }
    Rows<T> joined(n, d.heads * dim);
    launchDraft<mergeDraftAttention<T>>(dim3(n, d.heads), dim, 0, args, views, joined.p, splits);
    return draftProjections<T, 1>(joined, pre, {"o_proj.weight"}, {c.hidden})[0];
}
