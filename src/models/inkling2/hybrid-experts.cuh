// Host expert weights stay on their configured CPU/NUMA nodes. Only routed
// expert records and activations cross the device boundary.
struct CpuExpertGroup {
    int expert, first, rows;
};

struct HybridHostRows final : MultiThreadBaseOp {
    const Projection &weight;
    const std::vector<CpuExpertGroup> &groups;
    float *output;
    const float *rowScales;
    uint8_t *input;
    size_t inputStride;
    int node, nodes, worker, workers;
    HybridHostRows(const Projection &w, const std::vector<CpuExpertGroup> &g, const float *scales, uint8_t *x, size_t stride, float *y,
                   int n, int count, int t, int nt)
        : weight(w), groups(g), output(y), rowScales(scales), input(x), inputStride(stride), node(n), nodes(count), worker(t), workers(nt) {}
    void Run() override {
        const int begin = weight.rows * worker / workers, end = weight.rows * (worker + 1) / workers;
        std::vector<float> decoded;
        for (const auto &g : groups) {
            if (g.expert % nodes != node) continue;
            const auto &p = weight.host[g.expert];
            auto *x = input + size_t(g.first) * inputStride;
            auto *bf16 = reinterpret_cast<uint16_t *>(x);
            float *result = output + size_t(g.first) * weight.rows;
            if (weight.unpackHostRow) {
                decoded.resize(weight.cols);
                const auto *in = reinterpret_cast<const float *>(x);
                for (int j = begin; j < end; ++j) {
                    weight.unpackHostRow(p.data + size_t(j) * weight.rowBytes, decoded.data(), weight.cols);
                    for (int i = 0; i < g.rows; ++i)
                        result[size_t(i) * weight.rows + j] = std::inner_product(decoded.begin(), decoded.end(),
                                in + size_t(i) * weight.cols, 0.f);
                }
            } else if (weight.ggufSlots) {
                const auto t = ggml_type(weight.ggufType);
                const bool bf = t == GGML_TYPE_BF16;
                const auto a = bf ? BFLOAT16 : DataType(DATA_GGUF_FORMAT + int(weight.hostInputType));
                const auto b = bf ? BFLOAT16 : DataType(DATA_GGUF_FORMAT + int(t));
                FastllmGemm(g.rows, weight.cols, weight.rows, x, inputStride, p.data, weight.rowBytes,
                            result, weight.rows * sizeof(float), begin, end, a, b, FLOAT32);
            } else if (weight.fp4) {
                FastllmGemm(g.rows, weight.cols, weight.rows, bf16, inputStride,
                            p.data, weight.cols / 16 * 12, result, weight.rows * sizeof(float),
                            begin, end, BFLOAT16, NVFP4_BLOCK_16, FLOAT32);
            } else {
                MultiThreadLinearBFloat16FP8E4M3Op(bf16, p.data, nullptr, result, g.rows,
                    weight.cols, weight.rows, begin, end, p.scales, 1, weight.group).Run();
            }
            if (rowScales || (weight.ggufSlots && *p.scales != 1.f)) {
                for (int i = 0; i < g.rows; ++i) {
                    const float scale = rowScales ? rowScales[g.first + i] : *p.scales;
                    float *out = result + size_t(i) * weight.rows;
                    for (int j = begin; j < end; ++j) out[j] *= scale;
                }
            }
        }
    }
};

struct HostProjectionInput {
    std::vector<uint8_t> values;
    size_t stride = 0;
    std::vector<float> scales;
    std::vector<int> firstRoute;

    void prepare(const Projection &w, const float *input, int count, const std::vector<int> *routes = nullptr,
                 int routesPerInput = 1, int inputRows = 0) {
        const auto a = w.hostInputType;
        stride = ggml_row_size(a, w.cols);
        values.resize(size_t(count) * stride);
        scales.resize(w.fp4 && !w.plainInput() ? count : 0);
        if (routesPerInput > 1) firstRoute.assign(inputRows, -1);
        for (int i = 0; i < count; ++i) {
            const int row = routes ? (*routes)[i] / routesPerInput : i;
            uint8_t *packed = values.data() + size_t(i) * stride;
            auto *dst = reinterpret_cast<uint16_t *>(packed);
            if (routesPerInput > 1) {
                int &first = firstRoute[row];
                if (first >= 0) {
                    std::memcpy(packed, values.data() + size_t(first) * stride, stride);
                    if (!scales.empty()) scales[i] = scales[first];
                    continue;
                }
                first = i;
            }
            const float *x = input + size_t(row) * w.cols;
            if (a == GGML_TYPE_F32) {
                std::memcpy(packed, x, stride);
            } else if (a != GGML_TYPE_BF16) {
                iqk_quantize_row_q8_K(x, packed, w.cols, a, ggml_type(w.ggufType));
            } else if (!scales.empty()) {
                // Removing 512 * outer keeps reconstructed FP4 * E4M3 exact in BF16.
                float peak = 0;
                for (int j = 0; j < w.cols; ++j) peak = std::max(peak, std::abs(x[j]));
                scales[i] = peak / 3.f;
                const double inverse = scales[i] ? 1. / scales[i] : 0.;
                for (int j = 0; j < w.cols; ++j) dst[j] = Float32ToBFloat16RNEBits(float(x[j] * inverse));
            } else {
                std::transform(x, x + w.cols, dst, Float32ToBFloat16RNEBits);
            }
        }
    }
};

void hostProjection(const Projection &w, const std::vector<CpuExpertGroup> &groups,
                    HostProjectionInput &input, std::vector<float> &output) {
    auto *pool = GetAlivePool();
    const auto interval = pool->curActivateThreadInterval;
    std::vector<std::vector<int>> workers;
#ifdef USE_NUMAS
    if (w.device == ExpertDevice::Numa) workers = GetNumasExpertWorkerIds();
#endif
    if (workers.empty() || std::any_of(workers.begin(), workers.end(), [](const auto &v) { return v.empty(); })) {
        workers.assign(1, {});
        for (int t = interval.first; t < interval.second; ++t) workers[0].push_back(t);
    }
    if (workers[0].empty()) throw std::runtime_error("Inkling host experts need an active CPU worker");
    std::vector<HybridHostRows> tasks;
    size_t count = 0;
    for (const auto &v : workers) count += v.size();
    tasks.reserve(count);
    for (size_t n = 0; n < workers.size(); ++n) for (size_t t = 0; t < workers[n].size(); ++t) {
        tasks.emplace_back(w, groups, input.scales.empty() ? nullptr : input.scales.data(), input.values.data(), input.stride, output.data(), n, workers.size(), t, workers[n].size());
        pool->PushOp(workers[n][t], &tasks.back());
    }
    for (const auto &v : workers) for (int t : v) pool->Wait(t);
}

// The same two indexed MergeMOE projections serve permanent and replaceable slots.
// Slot pointers are immutable; readers wait for the latest publication for their layer.
struct HybridExpertRecords {
    int count = 0, resident = 0;
    size_t gateBytes = 0, stride = 0;
    Buffer<uint8_t> data;
    Buffer<uint8_t *> weights;
    Buffer<float *> scales;
    CudaIndexedMoeSlots view{};
    HybridMoeGraphSlots staging;
    MoeDecodeOverlapScheduler timing;
    cudaEvent_t start = nullptr, middle = nullptr, done = nullptr;
    cuda::HybridMoeAdmissions::Publication publication;
    static int permanentSlots(int slots) { return slots - std::min(16, std::max(1, slots / 8)); }
    HybridExpertRecords(int slots, int experts, const Projection &gate, const Projection &down) : count(slots) {
        if (!slots) return;
        resident = slots == experts ? slots : permanentSlots(slots);
        gateBytes = gate.recordBytes;
        stride = gateBytes + down.recordBytes;
        data = Buffer<uint8_t>(size_t(slots) * stride);
        // Packed GGUF kernels address the slot buffer directly. Only native
        // projections consume a table of weight pointers.
        if (!gate.ggufSlots || !down.ggufSlots) {
            weights = Buffer<uint8_t *>(2 * slots);
            std::vector<uint8_t *> w(2 * slots);
            for (int i = 0; i < slots; ++i) {
                w[i] = data.p + size_t(i) * stride;
                w[slots + i] = w[i] + gateBytes;
            }
            check(cudaMemcpy(weights.p, w.data(), w.size() * sizeof(w[0]), cudaMemcpyHostToDevice));
            view = {weights.p, weights.p + slots, nullptr, nullptr, gate.group, 1, down.group, 1};
        }
        if (!gate.fp4 && gate.ggufType < 0) {
            scales = Buffer<float *>(2 * slots);
            std::vector<float *> s(2 * slots);
            for (int i = 0; i < slots; ++i) {
                auto *base = data.p + size_t(i) * stride;
                s[i] = reinterpret_cast<float *>(base + size_t(gate.rows) * gate.cols);
                s[slots + i] = reinterpret_cast<float *>(base + gateBytes + size_t(down.rows) * down.cols);
            }
            check(cudaMemcpy(scales.p, s.data(), s.size() * sizeof(s[0]), cudaMemcpyHostToDevice));
            view.gateScales = scales.p;
            view.downScales = scales.p + slots;
        }
        staging.keys.assign(count - resident, -1);
        if (slots == experts) {
            for (int i = 0; i < experts; ++i) upload(i, i, gate, down);
        }
    }
    void project(const Projection &w, bool gateUp, float *input, const int *slots, float *out,
                 int routes, int copies, cudaStream_t stream) const {
        if (w.ggufSlots)
            w.ggufSlots(data.p + (gateUp ? 0 : gateBytes), stride, slots, input, out, routes, copies,
                        w.cols, w.rows, w.rowBytes, stream);
        else if (gateUp) CudaMoeGraphSlotGate(view, input, slots, out, routes / copies, copies, w.cols, w.rows / 2, stream);
        else CudaMoeGraphSlotDown(view, input, slots, out, routes, w.rows, w.cols, stream);
    }
    ~HybridExpertRecords() {
        for (auto event : {start, middle, done}) if (event) cudaEventDestroy(event);
    }
    void upload(int expert, int slot, const Projection &gate, const Projection &down) {
        auto *dst = data.p + size_t(slot) * stride;
        check(cudaMemcpy(dst, gate.host[expert].data, gateBytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dst + gateBytes, down.host[expert].data, stride - gateBytes, cudaMemcpyHostToDevice));
    }
    void admit(int expert, int slot, const Projection &gate, const Projection &down,
               cudaStream_t stream, cuda::ImmutableHostUploadRing *ring = nullptr) {
        const int staged = staging.Find(expert);
        auto *dst = data.p + size_t(slot) * stride;
        if (staged >= 0) {
            auto *src = data.p + size_t(resident + staged) * stride;
            check(cudaMemcpyAsync(dst, src, stride, cudaMemcpyDeviceToDevice, stream));
            return;
        }
        auto copy = [&](uint8_t *destination, const uint8_t *source, size_t bytes) {
            check(ring ? ring->Copy(destination, source, bytes, stream)
                       : cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
        };
        copy(dst, gate.host[expert].data, gateBytes);
        copy(dst + gateBytes, down.host[expert].data, stride - gateBytes);
    }
    void record(cudaEvent_t &event) {
        if (!event) check(cudaEventCreate(&event));
        check(cudaEventRecord(event, cudaStreamPerThread));
    }
};

// One admission budget covers the entire target forward, including repeated visits
// to a physical layer. Each partition owns stable local CUDA slot tables.
struct HybridCacheCycle final : ExpertCacheCycle {
    struct Layer { HybridExpertRecords &records; const Projection &gate, &down; };
    std::vector<Layer> layers;
    const int experts, resident;
    MoeFrequencyPolicy policy;
    cuda::HybridMoeAdmissions admissions;
    template<class T> static std::vector<T> repeat(const std::vector<T> &values, int each) {
        std::vector<T> result;
        result.reserve(values.size() * each);
        for (const auto &v : values) result.insert(result.end(), each, v);
        return result;
    }
    static std::vector<int> partitions(size_t layers, int each) {
        std::vector<int> ids(layers);
        std::iota(ids.begin(), ids.end(), 0);
        return repeat(ids, each);
    }
    HybridCacheCycle(const std::vector<uint64_t> &bytes, int experts, int slots)
        : experts(experts), resident(HybridExpertRecords::permanentSlots(slots)),
          policy(partitions(bytes.size(), experts), partitions(bytes.size(), resident),
                 bytes.size(), GetMoeCacheConfig(), repeat(bytes, experts)) {}
    cuda::HybridMoeAdmissions::Publication &uploadAdmission(MoeFrequencyPolicy::Admission a,
            cudaStream_t stream, cuda::ImmutableHostUploadRing *ring = nullptr) {
        const int index = a.key / experts;
        auto &layer = layers.at(index);
        layer.records.admit(a.key % experts, a.slot - index * resident, layer.gate, layer.down, stream, ring);
        return layer.records.publication;
    }
    std::shared_ptr<CudaWorkspace> borrowPrefillWorkspace(size_t bytes, int keep) override {
        if (!bytes || resident <= keep) return {};
        // The decoder mutex serializes forwards. The staging suffix stays writable
        // throughout prefill. Prefer wide records so large temporaries fit one region.
        std::vector<int> order(layers.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return layers[a].records.stride > layers[b].records.stride;
        });
        std::vector<CudaWorkspace::Region> regions;
        std::vector<std::pair<int, int>> spans;
        size_t capacity = 0;
        for (int layer : order) {
            auto &records = layers[layer].records;
            if (records.stride % 256) continue; // CudaWorkspace region alignment
            const int count = std::min<size_t>(resident - keep, 1 + (bytes - capacity - 1) / records.stride);
            const int first = resident - count;
            const size_t size = size_t(count) * records.stride;
            regions.push_back({records.data.p + size_t(first) * records.stride, size});
            spans.emplace_back(layer * resident + first, (layer + 1) * resident);
            records.publication.wait(cudaStreamPerThread, false);
            capacity += size;
            if (capacity >= bytes) break;
        }
        if (regions.empty()) return {};
        int device;
        check(cudaGetDevice(&device));
        auto arena = std::make_shared<CudaWorkspace>(device, std::move(regions), [this, spans] {
            for (auto span : spans) for (int i = span.first; i < span.second; ++i)
                policy.SetSlotAvailable(i, true);
        });
        for (auto span : spans) for (int i = span.first; i < span.second; ++i)
            policy.SetSlotAvailable(i, false);
        return arena;
    }
    void beginStep() override { policy.BeginStep(); }
    void endStep() override {
        admissions.publish(policy, cudaStreamPerThread, [&](int key) { return key / experts; },
            [&](auto a, cudaStream_t stream, cuda::ImmutableHostUploadRing *ring) -> auto & {
                return uploadAdmission(a, stream, ring);
            });
    }
};

// Reserve the same number of slots per host layer, charging each format's actual payload.
std::unique_ptr<HybridCacheCycle> planHybridCache(const Config &c, const Placement &p,
        const std::map<std::string, TensorRange> &files, int &slots) {
    std::vector<uint64_t> recordBytes;
    for (int i = 0; i < c.layers; ++i) {
        if (p.experts[i] == ExpertDevice::Cuda) continue;
        const auto prefix = "model.llm.layers." + std::to_string(i) + ".mlp.experts.";
        recordBytes.push_back(Projection::hostBytes(files.at(prefix + "w13_weight"), 2 * c.intermediate, c.hidden)
                            + Projection::hostBytes(files.at(prefix + "w2_weight"), c.hidden, c.intermediate));
    }
    const uint64_t bytes = std::accumulate(recordBytes.begin(), recordBytes.end(), uint64_t{0});
    slots = int(std::min<uint64_t>(c.experts, GetMoeCudaCacheDeviceBytes(p.gpu) / bytes));
    printf("[Inkling] hybrid expert cache: %d/%d slots per host layer, %.1f MiB including staging\n",
           slots, c.experts, double(bytes * slots) / (1024 * 1024));
    if (!slots || slots == c.experts) return nullptr;
    return std::make_unique<HybridCacheCycle>(recordBytes, c.experts, slots);
}

__global__ void gatherExpertRows(const float *input, const float *scores, const int *order, float *packed,
                                 float *packedScores, int top, int hidden) {
    const int row = blockIdx.y, route = order[row];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < hidden) packed[size_t(row) * hidden + col] = input[size_t(route / top) * hidden + col];
    if (!col) packedScores[row] = scores[route];
}

__global__ void reduceExpertRows(const float *parts, const int *inverse, float *out, int top, int hidden,
                                 const int *slots = nullptr, const float *host = nullptr) {
    const int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden) return;
    float sum = 0;
    for (int i = 0; i < top; ++i) {
        const int route = row * top + i;
        const float *source = slots && slots[route] < 0 ? host : parts;
        sum += source[size_t(inverse ? inverse[route] : route) * hidden + col];
    }
    out[size_t(row) * hidden + col] = sum;
}

#include "hybrid-graph.cuh"

class HostExperts final : public RoutedExperts {
    std::shared_ptr<HybridGraphStorage> graphStorage;
    mutable HybridExpertRecords cache;
    HybridCacheCycle *cycle;
    int layer = 0;
    void select(const Config &c, const Matrix &router, const float *bias, const Buffer<int> &ids,
                const Matrix &scores) const {
        Data logits(FLOAT32, {router.rows, c.experts}, CUDA, router.p);
        Data biasView(FLOAT32, {c.experts}, CUDA, const_cast<float *>(bias));
        Data indices(INT32, {router.rows, c.topK}, CUDA, ids.p);
        Data scales(FLOAT32, indices.dims, CUDA, scores.p);
        if (!FastllmCudaFusedSigmoidSelectExpert(logits, &biasView, indices, scales, c.topK, true, c.routeScale)) {
            Matrix probabilities(router.rows, c.experts);
            Data probs(FLOAT32, logits.dims, CUDA, probabilities.p);
            if (!FastllmCudaSigmoid(logits, probs) ||
                !FastllmCudaSelectExpert(probs, &biasView, indices, scales, c.topK, true, c.routeScale))
                throw std::runtime_error("Inkling expert routing geometry");
        }
    }

    Matrix prepareInput(const Matrix &x) const {
        if (gate.plainInput()) return x;
        Matrix result(x.rows, x.cols);
        quantizeInto(InputRow{x.p}, result.p, x.rows, x.cols, gate.fp4, smCount);
        return result;
    }
    void activate(float *pair, float *scores, float *mid, int routes) const {
        RoutedSiTU<float> source{pair, scores, gateCap, upCap};
        if (down.plainInput()) {
            const size_t n = size_t(routes) * down.cols;
            ggufRoutedActivation<<<(n + 255) / 256, 256, 0, cudaStreamPerThread>>>(source, mid, routes, down.cols);
        } else quantizeInto(source, mid, routes, down.cols, down.fp4, smCount);
    }

    void projectSlots(int rows, int top, float *input, int *ids, float *scores, float *output,
                      const Matrix &pair, const Matrix &mid) const {
        const int routes = rows * top, inter = down.cols;
        if (gate.plainInput() || down.plainInput()) {
            const auto stream = cudaStreamPerThread;
            Matrix parts(top == 1 ? 0 : routes, down.rows);
            cache.project(gate, true, input, ids, pair.p, routes, top, stream);
            activate(pair.p, scores, mid.p, routes);
            cache.project(down, false, mid.p, ids, top == 1 ? output : parts.p, routes, 1, stream);
            if (top > 1)
                reduceExpertRows<<<dim3((down.rows + 255) / 256, rows), 256, 0, stream>>>(
                    parts.p, nullptr, output, top, down.rows);
            return;
        }
        Data in(FLOAT32, {rows, gate.cols}, CUDA, input), out(FLOAT32, in.dims, CUDA, output);
        Data ix(INT32, {rows, top}, CUDA, ids), score(FLOAT32, ix.dims, CUDA, scores);
        Data g(FLOAT32, {routes, 2 * inter}, CUDA, pair.p), m(FLOAT32, {routes, inter}, CUDA, mid.p);
        if (!RunCudaIndexedMoeSlots(cache.view, in, ix, score, g, m, out, inter, *this))
            throw std::runtime_error("Inkling cached expert projection failed");
    }

    void runChunk(const Config &c, const Matrix &x, float *result, const float *bias, bool prefill) const {
        const int rows = x.rows, top = c.topK, routes = rows * top, h = c.hidden, inter = c.intermediate;
        Buffer<int> ids(routes);
        Matrix scores(rows, top);
        select(c, routeLogits(x, routerWeight), bias, ids, scores);
        if (cache.count == c.experts) {
            Matrix input = prepareInput(x), pair(routes, 2 * inter), mid(routes, inter);
            projectSlots(rows, top, input.p, ids.p, scores.p, result, pair, mid);
            return;
        }
        if (auto *scratch = DecodeGraphScratch::active) {
            Matrix input = prepareInput(x);
            graphStorage->run(*scratch, cache, cycle, layer, c, gate, down, input, ids, scores, result, smCount);
            return;
        }
        cache.publication.wait(cudaStreamPerThread, false);
        std::vector<int> hostIds(routes);
        check(cudaMemcpy(hostIds.data(), ids.p, routes * sizeof(int), cudaMemcpyDeviceToHost));
        auto &timing = cache.timing;
        auto *frequency = cycle ? &cycle->policy : nullptr;
        const int keyBase = layer * c.experts, slotBase = layer * cache.resident;
        HybridMoeRoutes plan(hostIds, c.experts, rows, keyBase, prefill, frequency, timing,
                             cache.count - cache.resident);
        const auto &misses = plan.misses;
        const int staged = plan.staged, residentRows = plan.residentRows;
        std::vector<int> order, gpuSlots, inverse(routes);
        auto append = [&](int e, int slot) {
            for (int r : plan.grouped[e]) {
                inverse[r] = order.size();
                order.push_back(r);
                if (slot >= 0) gpuSlots.push_back(slot);
            }
        };
        for (int e : plan.hits) append(e, frequency->Slot(keyBase + e) - slotBase);
        using Clock = std::chrono::steady_clock;
        const auto uploadStart = Clock::now();
        for (int i = 0; i < staged; ++i) {
            cache.staging.keys[i] = -1;
            cache.upload(misses[i], cache.resident + i, gate, down);
            cache.staging.keys[i] = misses[i];
            append(misses[i], cache.resident + i);
        }
        if (staged) timing.copiedExpert.Observe(std::chrono::duration<double, std::micro>(Clock::now() - uploadStart).count() / staged);
        const int gpu = order.size(), cpu = routes - gpu;
        std::vector<CpuExpertGroup> cpuGroups;
        for (int i = staged; i < int(misses.size()); ++i) {
            int e = misses[i];
            cpuGroups.push_back({e, int(order.size()) - gpu, int(plan.grouped[e].size())});
            append(e, -1);
        }
        Buffer<int> positions(routes), reverse(routes), slots(gpu);
        auto copyIds = [](int *dst, const std::vector<int> &src) {
            if (!src.empty()) check(cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
        };
        copyIds(positions.p, order);
        copyIds(reverse.p, inverse);
        copyIds(slots.p, gpuSlots);
        Matrix input = prepareInput(x), packed(routes, h), parts(routes, h), routeScore(routes, 1);
        gatherExpertRows<<<dim3((h + 255) / 256, routes), 256, 0, cudaStreamPerThread>>>(
            input.p, scores.p, positions.p, packed.p, routeScore.p, top, h);
        std::vector<float> hostIn(size_t(cpu) * h);
        if (cpu) check(cudaMemcpy(hostIn.data(), packed.p + size_t(gpu) * h, hostIn.size() * sizeof(float), cudaMemcpyDeviceToHost));
        Matrix gateGpu(gpu, 2 * inter), midGpu(gpu, inter);
        auto submit = [&](int first, int n) {
            if (!n) return;
            projectSlots(n, 1, packed.p + size_t(first) * h, slots.p + first, routeScore.p + first,
                         parts.p + size_t(first) * h, gateGpu, midGpu);
        };
        if (gpu) {
            cache.record(cache.start);
            submit(0, residentRows);
            cache.record(cache.middle);
            submit(residentRows, gpu - residentRows);
            cache.record(cache.done);
        }
        if (cpu) {
            std::vector<float> hostPair(size_t(cpu) * 2 * inter), hostMid(size_t(cpu) * inter), hostOut(size_t(cpu) * h);
            auto started = Clock::now();
            HostProjectionInput input;
            input.prepare(gate, hostIn.data(), cpu);
            hostProjection(gate, cpuGroups, input, hostPair);
            double cpuUs = std::chrono::duration<double, std::micro>(Clock::now() - started).count();
            Matrix pair(cpu, 2 * inter), mid(cpu, inter);
            check(cudaMemcpy(pair.p, hostPair.data(), hostPair.size() * sizeof(float), cudaMemcpyHostToDevice));
            activate(pair.p, routeScore.p + gpu, mid.p, cpu);
            check(cudaMemcpy(hostMid.data(), mid.p, hostMid.size() * sizeof(float), cudaMemcpyDeviceToHost));
            started = Clock::now();
            input.prepare(down, hostMid.data(), cpu);
            hostProjection(down, cpuGroups, input, hostOut);
            cpuUs += std::chrono::duration<double, std::micro>(Clock::now() - started).count();
            HybridMoeRoutes::ObserveCpu(timing, rows, cpu, cpuUs);
            check(cudaMemcpy(parts.p + size_t(gpu) * h, hostOut.data(), hostOut.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        reduceExpertRows<<<dim3((h + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(parts.p, reverse.p, result, top, h);
        check(cudaGetLastError());
        if (gpu) {
            check(cudaEventSynchronize(cache.done));
            auto observe = [](auto &estimate, cudaEvent_t a, cudaEvent_t b, int n) {
                if (n) { float ms; check(cudaEventElapsedTime(&ms, a, b)); estimate.Observe(1000. * ms / n); }
            };
            observe(timing.residentExpert, cache.start, cache.middle, residentRows);
            observe(timing.stagedExpert, cache.middle, cache.done, gpu - residentRows);
        }
        ++timing.calls;
        if (frequency && prefill) {
            for (int i = 0; i < staged; ++i) {
                auto a = plan.selectPrefill(*frequency, keyBase, misses[i]);
                if (a.slot >= 0) {
                    cycle->uploadAdmission(a, cudaStreamPerThread);
                    frequency->Admit(layer, a);
                }
            }
        }
    }
public:
    HostExperts(const std::map<std::string, TensorRange> &files, const std::string &name, const Config &c,
                ExpertDevice where, ExpertLoadContext &topology, const Data &router, int slots, HybridCacheCycle *cycle,
                std::shared_ptr<HybridGraphStorage> storage)
        : RoutedExperts(files, name, c, where, topology, router), graphStorage(std::move(storage)), cache(slots, c.experts, gate, down), cycle(cycle) {
        if (cycle) {
            cycle->admissions.attach(cache.publication);
            layer = cycle->layers.size();
            cycle->layers.push_back({cache, gate, down});
        }
    }
    Matrix run(const Config &c, const Matrix &x, const float *bias, ExpertPhase phase, Rows<__nv_bfloat16> *, Rows<__nv_bfloat16> *, int) const override {
        Matrix out(x.rows, c.hidden);
        // The gather and indexed GEMVs use a grid.y entry per route. Keep the
        // packed CPU/GPU scratch bounded independently of total prompt length.
        const size_t perRoute = sizeof(float) * size_t(5 * c.hidden + 6 * c.intermediate);
        const int chunk = std::max(1, int(std::min<size_t>(65535, (64ULL << 20) / perRoute) / c.topK));
        for (int i = 0; i < x.rows; i += chunk)
            runChunk(c, x.slice(i, std::min(chunk, x.rows - i)), out.p + size_t(i) * c.hidden, bias, phase == ExpertPhase::Prefill);
        return out;
    }
};
