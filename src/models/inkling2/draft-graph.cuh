// Both graphs bind the same request-major chain. Only descriptors, sampling
// parameters and fresh RNG seeds are uploaded between replays.
// Return verified tokens while the next draft reads the committed prefix.
struct RoundPublication {
    RoundHostVector<int> tokens;
    RoundHostVector<uint64_t> seeds;
    cudaEvent_t verified = nullptr;
    std::shared_ptr<DraftFence> tail;
    void allocate(size_t count, int steps) {
        tokens.resize(count);
        seeds.resize(steps);
        check(cudaEventCreateWithFlags(&verified, cudaEventDisableTiming));
        tail = std::make_shared<DraftFence>();
    }
    ~RoundPublication() {
        if (verified) cudaEventDestroy(verified);
    }
};
struct RoundPlan {
    RoundPublication publication;
    int batch, steps, chain;
    Buffer<DraftView> views;
    Buffer<DraftInput> inputs;
    Buffer<float> temperature, topP, ids;
    Buffer<int> topK, tokens, draftIds, live, result;
    Buffer<uint64_t> seeds;
    Buffer<unsigned long long> counter;
    Buffer<uint8_t> partials;
    Matrix probs;
    Buffer<int> proposalSupport;
    Buffer<void *> groupPointers;
    std::vector<DraftProjectionGroup> groups;
    std::shared_ptr<FastllmBf16ProjectionFork> projectionFork;
    void *sampler = nullptr, *targetSampler = nullptr;
    uint64_t rejectSeed = newSamplingSeed();
    cudaGraphExec_t draftExec = nullptr, verifyExec = nullptr;
    DecodeGraphScratch draftPool, verifyPool;
    Buffer<DecodeVisit> source, visits;
    Buffer<float> panels;
    std::vector<int> nextAnchor, nextPast;
    RoundHostVector<DraftInput> hostInputs;
    RoundHostVector<uint64_t> hostSeeds;
    RoundHostVector<DraftView> draftViews;
    RoundHostVector<DecodeVisit> descriptors;
    RoundPlan(int b, const Config &c, bool nativeProjections, bool earlyPublication)
        : batch(b), steps(c.draft.steps), chain(steps + 1), views(b * c.draft.layers), inputs(b),
          temperature(b * chain + b), topP(b * chain + b), ids(b * chain), topK(b * chain + b), tokens(b * chain),
          draftIds(b * steps), live(b), result(b * (chain + 1)), seeds(steps), counter(1),
          partials(FastllmCudaMtpGraphRejectBytes(b, c.vocab)), probs(b * steps, c.vocab),
          proposalSupport(b > 1 ? size_t(b) * steps * FastllmFoldSelection::maxCandidates : 0),
          groupPointers(nativeProjections ? 6 * c.draft.layers : 0), groups(nativeProjections ? c.draft.layers : 0), source(b * c.visits.size()),
          visits(b * chain * c.visits.size()),
          panels(size_t(b) * c.visits.size() * (c.conv - 1 + chain) * 2 * (c.kvHeads * c.headDim + c.hidden)),
          nextAnchor(earlyPublication ? b : 0), nextPast(earlyPublication ? b : 0), hostInputs(b), hostSeeds(steps),
          draftViews(b * c.draft.layers), descriptors(b * c.visits.size()) {
        check(cudaMemsetAsync(counter.p, 0, sizeof(*counter.p), cudaStreamPerThread));
        for (size_t i = 0; i < groups.size(); ++i) {
            groups[i].device = groupPointers.p + 6 * i;
        }
        if (earlyPublication) publication.allocate(size_t(b) * (chain + 1), steps);
        if (proposalSupport.p) check(cudaMemsetAsync(proposalSupport.p, 0xfe, size_t(b) * steps *
                FastllmFoldSelection::maxCandidates * sizeof(int), cudaStreamPerThread));
        sampler = FastllmCudaMtpCreateTargetDistribution(b, c.vocab);
        targetSampler = FastllmCudaMtpCreateTargetDistribution(b * chain, c.vocab);
        if (!sampler || !targetSampler) {
            FastllmCudaMtpDestroyTargetDistribution(sampler);
            FastllmCudaMtpDestroyTargetDistribution(targetSampler);
            throw CudaWorkspaceExhausted("Insufficient sampling workspace for Inkling DSpark");
        }
        if (proposalSupport.p) FastllmCudaMtpBindProposalSupport(sampler, probs.p, proposalSupport.p, steps);
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
RoundPlan *publishedPlan = nullptr;
bool draftQueued = false;
std::thread::id publicationThread;
std::vector<std::weak_ptr<Request>> publicationOwners;

void finishPublication() {
    if (!publishedPlan) return;
    if (draftQueued) {
        check(cudaEventSynchronize(publishedPlan->publication.tail->event));
        for (auto &weak : publicationOwners) {
            if (auto owner = weak.lock()) owner->draft->reader.reset();
        }
    }
    publishedPlan = nullptr;
    draftQueued = false;
    publicationOwners.clear();
}

// The cold branch starts from the original anchor and immutable base/context
// terms. Its private states, tokens and q can therefore be overwritten before
// publication without advancing any request KV or RNG counter.
template<bool Q8, class T, class Base, class MarkovStep>
void drawFoldRound(RoundPlan &p, const Rows<T> &latent, const Base &base, const FastllmFoldSelection &selection,
                   const MarkovStep &markov) {
    Buffer<int> complete(1);
    Buffer<InklingQ81> encoded(Q8 ? latent.cols / 32 : 0);
    Buffer<float> scores(Q8 ? c.vocab : 0);
    Buffer<__nv_bfloat16> nativeAdjust(Q8 ? 0 : c.vocab);
    auto *adjust = nativeAdjust.p;
    const auto *native = reinterpret_cast<const __nv_bfloat16 *>(foldedHead.p);
    const auto *bias = reinterpret_cast<const __nv_bfloat16 *>(foldedBias.p);
    const auto *fold = Q8 ? static_cast<const InklingQ80 *>(quantizedFold->cudaData) : nullptr;
    auto project = [&](auto score, int i, cudaStream_t stream) {
        constexpr bool hot = decltype(score)::value;
        if (!markov(i, stream)) return false;
        if constexpr (Q8) {
            if (!cuda_launch::Launch<encodeQ8Row<T>>(1, 128, 0, stream, latent.p, encoded.p, latent.cols))
                return false;
            const dim3 grid((c.vocab + 15) / 16);
            if constexpr (hot)
                return cuda_launch::Launch<q8FoldScores>(grid, 128, 0, stream, fold, encoded.p, base.p + size_t(i) * base.cols,
                    quantizedFoldBias, p.temperature.p, c.logitCap, c.vocab, p.probs.p + size_t(i) * c.vocab, scores.p, selection,
                    p.topK.p, complete.p, i == 0);
            else
                return cuda_launch::Launch<q8WindowProjection<float>>(grid, 128, 0, stream, fold, encoded.p, scores.p,
                    1, latent.cols / 32, c.vocab, c.vocab, c.vocab, 0.f);
        } else {
            return cuda_launch::Launch<bf16FoldProjection<hot>>((c.vocab + 7) / 8, 256, 0, stream,
                native, latent.p, c.vocab, adjust, selection);
        }
    };
    for (int i = 0; i < p.steps; ++i) {
        const size_t offset = size_t(i) * c.vocab;
        if (!project(std::true_type{}, i, cudaStreamPerThread))
            throw std::runtime_error("Inkling singleton fold launch failed");
        bool drawn;
        if constexpr (Q8)
            drawn = FastllmCudaMtpDrawFoldScores(p.sampler, scores.p, selection, p.topK.p, p.topP.p, p.seeds.p + i,
                                            p.probs.p + offset, p.tokens.p + i + 1, complete.p, i == 0);
        else
            drawn = FastllmCudaMtpDrawBf16Fold(p.sampler, base.p + size_t(i) * base.cols, adjust, bias, selection, c.logitCap,
                    p.temperature.p, p.topK.p, p.topP.p, p.seeds.p + i, p.probs.p + offset, p.tokens.p + i + 1,
                    complete.p, i == 0);
        if (!drawn) throw std::runtime_error("Inkling singleton proposal draw failed");
    }
    if (!FastllmCudaMtpFoldFallback(complete.p, [&](cudaStream_t stream) {
        for (int i = 0; i < p.steps; ++i) {
            const size_t offset = size_t(i) * c.vocab;
            if (!project(std::false_type{}, i, stream)) return false;
            if constexpr (Q8) {
                if (!FastllmCudaMtpRepairFold(p.sampler, base.p + size_t(i) * base.cols, scores.p, quantizedFoldBias, c.logitCap,
                    p.temperature.p, p.topK.p, p.topP.p, p.seeds.p + i, p.probs.p + offset, p.tokens.p + i + 1,
                    c.vocab, stream)) return false;
            } else if (!FastllmCudaMtpRepairBf16Fold(p.sampler, base.p + size_t(i) * base.cols, adjust, bias, c.logitCap,
                    p.temperature.p, p.topK.p, p.topP.p, p.seeds.p + i, p.probs.p + offset, p.tokens.p + i + 1,
                    c.vocab, stream)) return false;
        }
        return true;
    })) throw std::runtime_error("Inkling exact fold recovery failed");
}

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
    Rows<T> pending;
    for (int i = 0; i < d.layers; ++i) {
        std::string pre = "draft.layers." + std::to_string(i) + ".";
        auto a = draftAttention(draftGraphNorm(x, pre + "input_layernorm.weight", pending), i,
                                pre + "self_attn.", p.views.p + i * b, p.groups.empty() ? nullptr : &p.projectionFork, p.publication.verified != nullptr);
        auto in = draftGraphNorm(x, pre + "post_attention_layernorm.weight", a);
        auto pair = draftProjections<T, 2>(in, pre + "mlp.", {"gate_proj.weight", "up_proj.weight"},
                                           {d.intermediate, d.intermediate}, p.groups.empty() ? nullptr : &p.groups[i]);
        const int size = n * d.intermediate;
        launchDraft<draftSwiGLU<T>>((size + 255) / 256, 256, 0, pair[0].p, pair[1].p, size);
        pending = draftProjections<T, 1>(pair[0], pre + "mlp.", {"down_proj.weight"}, {h})[0];
    }
    if (pending.p) draftAdd<<<(n * h + 255) / 256, 256, 0, cudaStreamPerThread>>>(x.p, pending.p, n * h);
    Rows<T> selected(b * steps, h);
    check(cudaMemcpy2DAsync(selected.p, size_t(steps) * h * sizeof(T), x.p + h, size_t(d.block) * h * sizeof(T),
                            size_t(steps) * h * sizeof(T), b, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    auto hidden = draftGraphNorm(selected, "draft.norm.weight");
    Matrix baseFloat;
    Rows<T> base;
    if (quantizedFold) {
        baseFloat = Matrix(hidden.rows, c.vocab);
        projectQ8(hidden, weights.weight.at("model.llm.unembed.weight"), 0, baseFloat, true);
    } else {
        const auto &head = weights.weight.at("model.llm.unembed.weight");
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // The checkpoint can include padding rows. Keep the GEMM's aligned
            // leading dimension through sampling; padding is never a candidate.
            const int padded = (c.vocab + 127) / 128 * 128;
            if (bf16Mma && hidden.rows >= smCount && head.dataType == BFLOAT16 && !head.IsRepacked &&
                head.dims[0] >= padded && head.cudaData) {
                base = Rows<T>(hidden.rows, padded);
                if (!FastllmCudaBf16HeuristicGemm(hidden.p, head.cudaData, base.p, hidden.rows, h, padded))
                    base = {};
            }
        }
        if (!base.p) base = linear(hidden, "unembed.weight", c.vocab, 1, Projection::DraftHead);
    }
    const int pitch = 2 * rank + 2 * h;
    Rows<T> latent(b, rank);
    const std::string pre = "draft.markov_head.";
    const auto &w = weights.weight.at("model.llm." + pre + "markov_w1.weight");
    if (w.dims.size() != 2 || w.dims[0] < c.vocab || w.dims[1] != rank)
        throw std::runtime_error("Invalid Markov embedding");
    const auto &rnn = weights.weight.at("model.llm." + pre + "rnn_proj.weight");
    constexpr bool bf = std::is_same_v<T, __nv_bfloat16>;
    const bool split = steps > 1 && rnn.dataType == (bf ? BFLOAT16 : FLOAT16) && !rnn.IsRepacked &&
        rnn.cudaData && rnn.dims == std::vector<int>{3 * rank, pitch};
    const auto *rw = static_cast<const T *>(rnn.cudaData);
    const float *bias = param(pre + "rnn_proj.bias", 3 * rank);
    Matrix hiddenTerm, contextTerm;
    Rows<T> recurrent, states;
    if (split) {
        states = Rows<T>(2 * b, rank);
        hiddenTerm = Matrix(b * steps, 3 * rank);
        contextTerm = Matrix(b, 3 * rank);
        Rows<T> context(b, h);
        launchDraft<draftMarkovContext<T>>(dim3((h + 255) / 256, b), 256, 0, p.views.p, context.p, h);
        auto projectTerm = [&](const Rows<T> &input, int offset, Matrix &out) {
            const float one = 1.f, zero = 0.f;
            constexpr auto dtype = bf ? CUDA_R_16BF : CUDA_R_16F;
            if (cublasGemmEx(getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, 3 * rank, input.rows, h,
                    &one, rw + offset, dtype, pitch, input.p, dtype, h, &zero, out.p, CUDA_R_32F,
                    3 * rank, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Inkling Markov context projection failed");
        };
        projectTerm(hidden, 2 * rank, hiddenTerm);
        projectTerm(context, 2 * rank + h, contextTerm);
    } else recurrent = Rows<T>(b, pitch);
    auto markov = [&](int i, cudaStream_t stream) {
        return cuda_launch::Launch<draftMarkovStep<T>>(dim3((rank + 3) / 4, b), 128, 0, stream, rw,
            static_cast<const T *>(w.cudaData), p.tokens.p, hiddenTerm.p, contextTerm.p, bias,
            states.p + size_t(i % 2) * b * rank, states.p + size_t((i+1) % 2) * b * rank,
            latent.p, rank, pitch, steps, i);
    };
    // The exact fold specializes one 256-wide recurrent row. Other
    // geometries, and eager execution, use the ordinary projection and sampler.
    FastllmFoldSelection selection{};
    const bool tiled = CUDART_VERSION >= 12040 && split && b == 1 && rank == 256 && (quantizedFold || bf) &&
        singletonControllerPdl && std::isfinite(c.logitCap) && c.logitCap > 0 &&
        DecodeGraphScratch::active == &p.draftPool && FastllmCudaMtpFoldSelection(p.sampler, smCount, selection);
    if (tiled) {
        if (quantizedFold) drawFoldRound<true>(p, latent, baseFloat, selection, markov);
        else if constexpr (bf) drawFoldRound<false>(p, latent, base, selection, markov);
    } else {
        for (int i = 0; i < steps; ++i) {
            if (split) {
                if (!markov(i, cudaStreamPerThread)) throw std::runtime_error("Inkling Markov step launch failed");
            } else {
                launchDraft<draftMarkovInput<T>>(dim3((std::max(h, rank) + 255) / 256, b), 256, 0,
                    recurrent.p, static_cast<const T *>(w.cudaData), p.tokens.p, hidden.p, p.views.p, i, steps, rank, h);
                Matrix raw(b, 3 * rank);
                projectInto(recurrent, pre + "rnn_proj.weight", raw, 0, 1);
                launchDraft<draftRecur<T>>(dim3((rank + 255) / 256, b), 256, 0,
                    raw.p, bias, recurrent.p, latent.p, rank, pitch);
            }
            if (quantizedFold) {
                Matrix scores(b, c.vocab);
                projectQ8(latent, *quantizedFold, 0, scores, true);
                if (!FastllmCudaMtpGraphDrawMarkovFloat(p.sampler, baseFloat.p + size_t(i) * c.vocab, steps * c.vocab,
                        scores.p, quantizedFoldBias, c.logitCap, p.temperature.p, p.topK.p, p.topP.p,
                        p.seeds.p + i, 0, p.probs.p + size_t(i) * c.vocab, steps * c.vocab, p.tokens.p + i + 1, p.chain))
                    throw std::runtime_error("Inkling quantized draft sampling failed");
                continue;
            }
            constexpr auto type = std::is_same_v<T, half> ? FLOAT16 : BFLOAT16;
            Rows<T> adjust(b, c.vocab);
            Data l(type, {b, rank}, CUDA, latent.p), head(type, {c.vocab, rank}, CUDA, foldedHead.p);
            Data out(type, {b, c.vocab}, CUDA, adjust.p);
            bool projected = false;
            if constexpr (std::is_same_v<T, __nv_bfloat16>)
                projected = bf16Mma && b > 1 && bf16Mma->draftLogits(latent.p, foldedHead.p, adjust.p, b, rank, c.vocab);
            if (!projected) DoCudaLinear(l, head, *GetEmptyData(), out);
            if (!FastllmCudaMtpGraphDrawMarkov(p.sampler, base.p + size_t(i) * base.cols, steps * base.cols, adjust.p,
                                               foldedBias.p, c.logitCap, type == BFLOAT16, p.temperature.p, p.topK.p,
                                               p.topP.p, p.seeds.p + i, 0, p.probs.p + size_t(i) * c.vocab,
                                               steps * c.vocab, p.tokens.p + i + 1, p.chain))
                throw std::runtime_error("Inkling batched draft sampling failed");
        }
    }
    castProjection<<<(b * p.chain + 255) / 256, 256, 0, cudaStreamPerThread>>>(p.tokens.p, p.ids.p, b * p.chain);
    check(cudaMemcpy2DAsync(p.draftIds.p, steps * sizeof(int), p.tokens.p + 1, p.chain * sizeof(int),
                            steps * sizeof(int), b, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
}
