// Graphs replay serially on the decoder stream and share intermediate storage.
// A padded row owns private history and KV so it can never advance a live request.
struct DecodePadding {
    Buffer<float> kv, history;
    Buffer<int2> page{1};
    std::vector<DecodeVisit> visits;

    DecodePadding(int rows, const Config &c) : visits(rows) {
        const size_t width = size_t(c.kvHeads) * attentionWidth(c.headDim);
        const size_t depth = std::max(1, c.conv - 1);
        const size_t kvHistory = depth * c.kvHeads * c.headDim, hiddenHistory = depth * c.hidden;
        const size_t stride = 2 * (kvHistory + hiddenHistory);
        kv = Buffer<float>(size_t(rows) * 2 * width);
        history = Buffer<float>(size_t(rows) * stride);
        check(cudaMemsetAsync(page.p, 0, sizeof(int2), cudaStreamPerThread));
        check(cudaMemsetAsync(history.p, 0, rows * stride * sizeof(float), cudaStreamPerThread));
        for (int i = 0; i < rows; ++i) {
            float *key = kv.p + i * 2 * width, *h = history.p + i * stride;
            visits[i] = {{key, key + width, page.p, 0, 1, 0},
                         {h, h + kvHistory, h + 2 * kvHistory, h + 2 * kvHistory + hiddenHistory}, 0, 1};
        }
    }
};

struct BatchDecodePlan {
    int rows;
    DecodeGraphScratch scratch;
    cudaGraphExec_t exec = nullptr;
    Matrix logits;
    Buffer<DecodeInput> inputs;
    Buffer<DecodeVisit> visits;
    Buffer<int> sampled;
    DecodePadding padding;
    std::vector<DecodeInput> hostInputs;
    std::vector<DecodeVisit> hostVisits;

    BatchDecodePlan(int rows, const Config &c, Matrix output)
        : rows(rows), logits(std::move(output)), inputs(rows), visits(size_t(rows) * c.visits.size()),
          sampled(rows), padding(rows, c), hostInputs(rows), hostVisits(size_t(rows) * c.visits.size()) {
        for (size_t slot = 0; slot < c.visits.size(); ++slot)
            std::copy(padding.visits.begin(), padding.visits.end(), hostVisits.begin() + slot * rows);
    }
    ~BatchDecodePlan() { if (exec) cudaGraphExecDestroy(exec); }
    void upload() {
        check(cudaMemcpyAsync(inputs.p, hostInputs.data(), hostInputs.size() * sizeof(DecodeInput),
                              cudaMemcpyHostToDevice, cudaStreamPerThread));
        check(cudaMemcpyAsync(visits.p, hostVisits.data(), hostVisits.size() * sizeof(DecodeVisit),
                              cudaMemcpyHostToDevice, cudaStreamPerThread));
    }
};
std::map<int, std::unique_ptr<BatchDecodePlan>> batchDecodePlans;

int decodeCapacity(int count) const {
    int capacity = 1;
    const int limit = count <= model.maxBatch ? model.maxBatch : INT_MAX;
    while (capacity < count) capacity += std::min(capacity, limit - capacity);
    return capacity;
}

BatchDecodePlan &batchDecodePlan(int rows) {
    auto it = batchDecodePlans.lower_bound(rows);
    if (it != batchDecodePlans.end() && it->first == rows) return *it->second;
    // Warmup visits the largest capacity first. Serial replays can write
    // into the same logits allocation without changing any captured address.
    Matrix logits = it == batchDecodePlans.end() ? Matrix(rows, c.vocab) : it->second->logits.slice(0, rows);
    auto entry = std::make_unique<BatchDecodePlan>(rows, c, std::move(logits));
    auto &p = *entry;
    p.scratch.ShareStorage(roundScratch);
    p.upload();
    auto body = [&] {
        Buffer<float> ids(rows), temperature(rows), topP(rows);
        Buffer<int> topK(rows);
        Buffer<uint64_t> seeds(rows), offsets(rows);
        unpackDecodeInputs<<<(rows + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            p.inputs.p, rows, ids.p, topK.p, temperature.p, topP.p, seeds.p, offsets.p);
        std::vector<Segment> segments(rows);
        for (int i = 0; i < rows; ++i) segments[i] = {nullptr, i, 1, nullptr, nullptr, p.visits.p};
        target(ids.p, rows, segments, p.logits);
        Matrix probabilities(rows, c.vocab);
        size_t bytes = FastllmCudaTopKFirstSamplingWorkspaceBytes(rows);
        Buffer<uint8_t> workspace(bytes);
        if (!FastllmCudaTopKFirstSamplingToDeviceWithRng(p.logits.p, probabilities.p, temperature.p, topK.p,
                topP.p, nullptr, nullptr, 0, p.sampled.p, ids.p, rows, c.vocab, seeds.p, offsets.p, workspace.p, bytes))
            throw std::runtime_error("Inkling batch decode sampling failed");
    };
    p.scratch.Warmup(body);
    p.scratch.Capture(p.exec, body);
    printf("[Inkling] captured full batch decode graph (%d rows)\n", rows);
    return *batchDecodePlans.emplace_hint(it, rows, std::move(entry))->second;
}

BatchDecodePlan &replayBatchDecode(const std::vector<Segment> &batch, const float *tokens,
                                   const std::vector<GenerationConfig> &sampling) {
    auto &p = batchDecodePlan(decodeCapacity(batch.size()));
    for (int i = 0; i < p.rows; ++i) {
        const bool live = i < int(batch.size());
        if (live) {
            const auto &s = batch[i];
            p.hostInputs[i] = decodeInput(*s.request, tokens[s.offset], sampling[i],
                                          ordinarySampling(sampling[i]) && !sampling[i].IsSimpleGreedy());
        } else p.hostInputs[i] = {};
        for (size_t slot = 0; slot < c.visits.size(); ++slot)
            p.hostVisits[slot * p.rows + i] = live ? decodeVisit(*batch[i].request, slot) : p.padding.visits[i];
    }
    p.upload();
    launchTargetGraph(p.exec, p.scratch);
    return p;
}
