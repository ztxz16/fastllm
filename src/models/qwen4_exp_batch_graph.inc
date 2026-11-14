#ifdef USE_CUDA
// Graph storage belongs to persistent serving slots or caller-owned requests.
// The stage cache retains bounded shape/slot combinations across Forward calls.
// Slot release invalidates compositions before captured storage is destroyed.
struct Qwen4ExpModel::BatchGraph {
    struct Request {
        DecodeCudaGraphState attention;
        Data ple;
        int previous = 0, rows = 0, validRows = 0, width = 0;
    };
    struct Layer {
        Data input, injection, output, afterAttention, residual, readInput, readInjection;
        MoEPrepared moe;
    };
    int device = -1, firstLayer = 0, endLayer = 0;
    std::vector<uint64_t> signature;
    std::vector<std::unique_ptr<Request>> requests;
    std::map<int, Layer> layers;
    Data hidden, logits, finalHidden, embeddings;
    MtpDraftCudaGraphState recorder;
    HybridMoeGraph complete;
    std::vector<const void*> children;
    std::vector<void*> handles;
    bool ready = true;
    ~BatchGraph() { Reset(); }
    void Reset() {
        complete.Reset();
        for (void *h : handles) FastllmCudaReleaseNativeMoeFullGraph(h);
        handles.clear(); children.clear();
        recorder.Reset(device);
    }
    void Begin() { children.clear(); ready = true; }
    template<class F> void Dense(int key, F &&body) {
        auto &segment = recorder.segments[key];
        recorder.RunPrefix(segment, body, true);
        ready &= segment.captured;
        if (segment.captured) children.push_back(segment.graph);
    }
    void Experts(Qwen4ExpModel &m, int key, const MoEExecutionPlan &plan, Layer &layer,
                 HybridMoeBatch &eager) {
        const bool hybrid = FastllmCudaNativeMoeGraphOwner(plan.weights->data(), plan.weights->size(), layer.input.dims[1]);
        if (!hybrid) {
            Dense(key, [&] {
                Data output;
                m.RunMoE(plan, layer.input, output);
                AssertInFastLLM(Qwen4PrepareDecodeGraphTensor(layer.output, output, device),
                    "Qwen batch resident expert staging failed");
            });
            return;
        }
        auto &shared = recorder.segments[key];
        auto sharedBody = [&] {
            Data x, gate, mid, scale, projected;
            x.FakeFrom(layer.input, 0); x.Resize({layer.input.dims[1], m.embed_dim});
            m.RunSharedExpert(plan.mlpPrefix, x, gate, mid, scale, projected);
            ToDataType(projected, layer.moe.result.sharedOutput, FLOAT32);
        };
        recorder.RunPrefix(shared, sharedBody, true);
        Data x; x.FakeFrom(layer.input, 0); x.Resize({layer.input.dims[1], m.embed_dim});
        AssertInFastLLM(eager.Run(x, layer.moe.index, layer.moe.score, layer.output,
            plan.weights->data(), plan.weights->size(), plan.deviceLayer), "Qwen batch graph expert warmup failed");
        ready &= shared.captured;
        if (shared.captured) {
            void *child = nullptr;
            const auto *storage = layer.output.cudaData;
            void *handle = FastllmCudaCreateNativeMoeFullGraph(x, layer.moe.index, layer.moe.score,
                layer.output, plan.weights->data(), plan.weights->size(), plan.deviceLayer, shared.graph, &child);
            AssertInFastLLM(storage == layer.output.cudaData, "Qwen batch graph expert storage changed");
            if (handle) handles.push_back(handle);
            if (child) children.push_back(child);
            ready &= handle && child;
        }
        layer.output.Reshape({1, layer.input.dims[1], m.embed_dim});
    }
    void Route(Qwen4ExpModel &m, const MoEExecutionPlan &plan, Layer &layer) {
        if (!FastllmCudaCanRunMoeHybrid(plan.weights->data(), plan.weights->size())) return;
        Data x, logits;
        x.FakeFrom(layer.input, 0); x.Resize({layer.input.dims[1], m.embed_dim});
        Linear(x, m.weight[plan.mlpPrefix + "gate.weight"], Data(), logits);
        ToDataType(logits, FLOAT32); m.SelectMoEExperts(logits, layer.moe.index, layer.moe.score);
    }
    void Join(Layer &layer) {
        if (!layer.moe.result.sharedOutput.dims.empty()) {
            layer.moe.result.sharedOutput.Reshape(layer.output.dims);
            AddTo(layer.output, layer.moe.result.sharedOutput);
        }
    }
    void Seal(const char *phase, int rows) {
        if (ready && complete.Compose(children, std::move(handles)))
            std::fprintf(stderr, "[Fastllm] Qwen cross-request %s full graph ready: requests=%zu rows=%d cuda:%d.\n",
                phase, requests.size(), rows, device);
        for (void *h : handles) FastllmCudaReleaseNativeMoeFullGraph(h);
        handles.clear();
    }
};
#endif
