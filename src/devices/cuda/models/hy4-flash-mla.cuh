// Included in the HY4 CUDA translation unit to share projection, RoPE and IndexPool operators.
static void PreparePagedMla(Scratch &s, int rows) {
    if (!s.mlaSupported)
        return;
    // A full tile and a short trailing tile have independent captured schedules.
    for (int n : {std::min(rows, 16), rows > 16 ? rows % 16 : 0}) {
        if (!n)
            continue;
        for (int heads : {32, 64}) {
            auto &plan = s.mlaPlans[{heads, n}];
            if (!plan)
                plan = FastllmCudaPrepareMLAPaged(n, heads, heads == 32 ? 2048 : 512);
        }
    }
}

__global__ static void appendMlaRow(const float *kv, const float *q, __nv_bfloat16 *qNope, __nv_bfloat16 *qPe,
                                    int pos, const StepInput *step, int keep, int keyStride, float theta,
                                    const KVPage *const *pages) {
    if (step)
        pos = step->position;
    const int h = blockIdx.x, d = threadIdx.x;
    if (h == 0) {
        auto *latent = kvRow<true>(pages, pos % keep, 512);
        latent[d] = __float2bfloat16(kv[d]);
        latent[d + 256] = __float2bfloat16(kv[d + 256]);
        if (d < 64) {
            auto *key = kvRow(pages, pos % keep, keyStride);
            key[d] = __float2bfloat16(rope(kv[512 + d], kv[512 + (d ^ 1)], d, pos, theta));
        }
    }
    const float *query = q + h * 256;
    if (d < 192)
        qNope[h * 192 + d] = __float2bfloat16(query[d]);
    if (d < 64)
        qPe[h * 64 + d] = __float2bfloat16(rope(query[192 + d], query[192 + (d ^ 1)], d, pos, theta));
}

__global__ static void mlaTokenRows(const KVPage *const *pages, const int *groups, int pos,
                                    const StepInput *step, int width, int keep, int keyStride,
                                    FastllmCudaMLAToken *tokens, int *length) {
    if (step)
        pos = step->position;
    // 511 completed four-token groups plus the visible prefix of the current group.
    const int count = groups ? 2045 + (pos & 3) : min(pos + 1, width);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
        *length = count;
    if (i >= count)
        return;
    const int slot = groups ? groups[i / 4] * 4 + (i & 3) : (pos + 1 - count + i) % keep;
    tokens[i] = {kvRow<true>(pages, slot, 512), kvRow(pages, slot, keyStride)};
}

// The sink is a zero-valued extra softmax column. Its scale is shared by
// every V channel of a head, so apply it after the head's V projection.
__global__ static void mlaSinkGate(const __nv_bfloat16 *values, const float *lse, Hy4View sink,
                                   const float *gate, float *out, int heads) {
    const int head = blockIdx.x, row = blockIdx.y;
    const int i = (row * heads + head) * 256 + threadIdx.x;
    const float scale = sigmoid(lse[row * heads + head] - read(sink.data, sink.type, head));
    out[i] = bf(bf(__bfloat162float(values[i]) * scale) * bf(sigmoid(gate[i])));
}

static void PagedMlaTile(Weights &w, Scratch &s, const std::string &p, int heads, const AttentionRow *rows,
                         int n, const float *x, const float *qr, const float *q, const float *kv,
                         const float *gate, float *mix) {
    const bool window = heads == 64;
    const int width = heads * 256, candidates = window ? 512 : 2048;
    const int stride = 64 + (window ? 0 : 256);
    const float theta = window ? 10000.f : 10000000.f;
    auto bfScratch = [&](const char *name, size_t count) {
        return reinterpret_cast<__nv_bfloat16 *>(s.Get(name, (count + 1) / 2));
    };
    auto qNope = bfScratch("mlaQNope", n * heads * 192), qPe = bfScratch("mlaQPe", n * heads * 64);
    auto absorbed = bfScratch("mlaAbsorbed", n * heads * 512),
         latent = bfScratch("mlaAttended", n * heads * 512);
    auto values = bfScratch("mlaValues", n * width);
    auto lengths = reinterpret_cast<int *>(s.Get("mlaLengths", n));
    auto slots = reinterpret_cast<FastllmCudaMLAToken *>(
        s.Get("mlaCandidates", size_t(n) * candidates * sizeof(FastllmCudaMLAToken) / sizeof(float)));
    auto lse = s.Get("mlaLse", n * heads);
    for (int i = 0; i < n; ++i) {
        const auto &r = rows[i];
        s.step = r.step;
        const int keep =
            window ? 512 + std::max(15, r.draftWidth) : w.config["max_position_embeddings"].int_value();
        const auto pages = &r.cache->target;
        appendMlaRow<<<heads, 256>>>(kv + i * 576, q + i * width, qNope + i * heads * 192,
                                     qPe + i * heads * 64, r.position, r.step, keep, stride, theta, pages);
        const int *chosen =
            window ? nullptr : IndexPool(w, s, p, x + i * 3072, qr + i * 1024, stride, r.position, pages);
        mlaTokenRows<<<candidates / 128, 128>>>(pages, chosen, r.position, r.step, candidates, keep, stride,
                                                slots + i * candidates, lengths + i);
        SaveDraftAttentionKV(w, s, r.draft, r.draftLayer, r.draftWidth, kv + i * 576, r.position, keep, theta,
                             chosen, r.cache);
    }
    const auto weight = static_cast<const __nv_bfloat16 *>(w.Device(p + "kv_b_proj.weight", s.device));
    const float one = 1, zero = 0;
    auto handle = getFastllmCublasHandle();
    // Each head owns 192 key rows followed by 256 value rows of the [448, 512] matrix.
    auto status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 512, n, 192, &one, weight,
                                             CUDA_R_16BF, 512, 448 * 512, qNope, CUDA_R_16BF, heads * 192,
                                             192, &zero, absorbed, CUDA_R_16BF, heads * 512, 512, heads,
                                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    AssertInFastLLM(status == CUBLAS_STATUS_SUCCESS, "HY4 MLA query absorption failed.");
    AssertInFastLLM(
        FastllmCudaMLAPaged(*s.mlaPlans.at({heads, n}), absorbed, qPe, slots, lengths, 0.0625f, latent, lse),
        "HY4 paged MLA launch failed.");
    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 256, n, 512, &one,
                                        weight + 192 * 512, CUDA_R_16BF, 512, 448 * 512, latent, CUDA_R_16BF,
                                        heads * 512, 512, &zero, values, CUDA_R_16BF, heads * 256, 256, heads,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    AssertInFastLLM(status == CUBLAS_STATUS_SUCCESS, "HY4 MLA value projection failed.");
    mlaSinkGate<<<dim3(heads, n), 256>>>(values, lse, w.View(p + "learnable_sink_param", s.device), gate, mix,
                                         heads);
}

void PagedAttentionBatch(Weights &w, Scratch &s, int layer, const std::vector<AttentionRow> &rows,
                         const float *inputs, float *outputs) {
    const std::string p = "model.layers." + std::to_string(layer) + ".self_attn.";
    const int heads = w.config["layer_types"][layer].string_value() == "sliding_attention" ? 64 : 32;
    const int width = heads * 256;
    const bool mla = UsesPagedMla(w, s, p);
    for (int first = 0; first < int(rows.size()); first += 16) {
        const int n = std::min(16, int(rows.size()) - first);
        const auto *x = inputs + size_t(first) * 3072;
        auto *y = outputs + size_t(first) * 3072;
        auto qr = s.Get("batchQr", n * 1024), q = s.Get("batchQ", n * width), kv = s.Get("batchKv", n * 576);
        auto gate = s.Get("batchGate", n * width), mix = s.Get("batchMix", n * width);
        LinearBatch(w, s, p + "q_a_proj.weight", x, qr, n);
        LinearBatch(w, s, p + "kv_a_proj_with_mqa.weight", x, kv, n);
        NormBatch(w, s, p + "q_a_layernorm.weight", qr, n, 1024);
        NormBatch(w, s, p + "kv_a_layernorm.weight", kv, n, 512, 576);
        LinearBatch(w, s, p + "q_b_proj.weight", qr, q, n);
        LinearBatch(w, s, p + "linear_gate.weight", x, gate, n);
        if (mla)
            PagedMlaTile(w, s, p, heads, rows.data() + first, n, x, qr, q, kv, gate, mix);
        else
            ExpandedAttentionTile(w, s, p, heads, rows.data() + first, n, x, qr, q, kv, gate, mix);
        LinearBatch(w, s, p + "o_proj.weight", mix, y, n);
    }
}
