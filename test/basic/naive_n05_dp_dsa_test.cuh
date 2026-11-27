// Dense dyadic operands give exactly representable FP32 dot products in every
// split-K order, so the unfused reference can check the ordered shortlist exactly.
void DpDsaVerify(int rows, bool paged, int bits = 16) {
    const int cap = paged ? 32768 : 16640;
    constexpr int qcols = 64 * 192, cols = 64 * 128, width = qcols + 4 * 192 + 4 * 128;
    constexpr float theta = 10000000.f, valueScale = .75f;
    Cache primary(cap, paged, false, 11, 4, bits), alternate(cap, paged, false, 32, 4, bits);
    Cache refPrimary(cap, paged, false, 11, 4, bits), refAlternate(cap, paged, false, 32, 4, bits);
    Data normalized, raw, wk, wq, ww, norm, bias, sink, positions, live, out, arena;
    Data ik, iq, iw, q, k, v, ref, indices, empty;
    auto matrix = [&](Data &d, int width) { Init(d, BFLOAT16, {1, rows, width}); };
    matrix(normalized, 4096);
    matrix(raw, width);
    matrix(ik, 128);
    matrix(iq, 2048);
    matrix(iw, 16);
    matrix(q, qcols);
    matrix(k, 4 * 192);
    matrix(v, 4 * 128);
    Fill(raw, 45, .3f);
    auto dyadic = [&](Data &d, unsigned seed) {
        std::vector<B> x(d.Count(0));
        for (auto &value : x) {
            seed = 1664525u * seed + 1013904223u;
            value = __float2bfloat16((int(seed >> 27) - 16) / 32.f);
        }
        Cuda(cudaMemcpy(d.cudaData, x.data(), x.size() * sizeof(B), cudaMemcpyHostToDevice));
    };
    dyadic(normalized, 17);
    Init(wk, BFLOAT16, {128, 4096});
    Init(wq, BFLOAT16, {2048, 4096});
    Init(ww, BFLOAT16, {16, 4096});
    dyadic(wk, 29);
    dyadic(wq, 47);
    dyadic(ww, 59);
    Init(norm, FLOAT32, {128});
    Init(bias, FLOAT32, {128});
    Init(sink, FLOAT32, {64});
    Fill(norm, 7, .15f, 1);
    Fill(bias, 9, .07f);
    Fill(sink, 22, .1f);
    Init(positions, FLOAT32, {1, rows});
    Init(live, INT32, {rows});
    Init(indices, INT32, {rows, 2048});
    const int pitch = (rows + 3) / 4 * 4;
    Init(arena, INT32, {pitch * (cols + cols / 32) / 4});
    FastllmMxfp8Activation target{(uint8_t *)arena.cudaData, (uint8_t *)arena.cudaData + pitch * cols, rows,
                                  cols, pitch};
    FastllmNaiveDecodeScratch scratch, reference;
    Data queries, scores;
    if (bits != 16) Init(scores, INT32, {rows * 32768});
    auto binding = (const void *const *)primary.table.cudaData;
    auto fused = [&](const void *const *table, bool overlap = false) {
        return FastllmCudaTryDpDSAVerify(normalized, raw, wk, wq, ww, norm, bias, sink, positions, live,
                                         *primary.K(), *primary.V(), table, cap, theta, valueScale, target,
                                         scratch, out, overlap, rows, &scores, &queries, bits == 4 ? &primary.indexPool : nullptr);
    };
    auto setLengths = [&](int past) {
        std::vector<float> pos(rows);
        std::vector<int> lengths(rows);
        for (int i = 0; i < rows; ++i) {
            // Mixed live and padded rows reuse the same captured graph and arena.
            lengths[i] = past == -2 ? (i % 3 == 1 ? 0 : 16381 + i) : past < 0 ? 0 : past + i + 1;
            pos[i] = std::max(0, lengths[i] - 1);
        }
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), rows * 4, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(positions.cudaData, pos.data(), rows * 4, cudaMemcpyHostToDevice));
    };
    setLengths(-1);
    Require(!fused(nullptr), "missing replay binding accepted");
    Require(fused(binding), "valid DP verify rejected");
    Cuda(cudaDeviceSynchronize());
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(fused(binding, true), "DP verify capture failed");
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    size_t nodes = 0;
    Cuda(cudaGraphGetNodes(graph, nullptr, &nodes));
    Require(nodes == 1, "DP middle stage launch count");
    std::vector<Data> views(rows);
    std::vector<FastllmNaiveRopeCacheRow> caches;
    std::vector<FastllmNaiveDSARow> descriptors(rows);
    auto refBinding = (const void *const *)refPrimary.table.cudaData;
    for (int i = 0; i < rows; ++i) {
        View(views[i], ik, {1, 1, 128}, size_t(i) * 128 * 2);
        caches.push_back({refPrimary.K(), refPrimary.V(), &views[i], refBinding, bits == 4 ? &refPrimary.indexView : nullptr});
        auto &d = descriptors[i];
        d.key = refPrimary.key.cudaData;
        d.value = refPrimary.value.cudaData;
        d.cache = refBinding;
        d.capacity = cap;
        d.keyStride = refPrimary.stride;
        d.fp8 = bits == 8; d.fp4 = bits == 4;
        d.selectedCount = 2048;
        d.pageLen = paged ? 128 : 0;
        d.keyPages = (const int *)refPrimary.kp.cudaData;
        d.valuePages = (const int *)refPrimary.vp.cudaData;
        d.indices = (const int *)indices.cudaData + size_t(i) * 2048;
        d.liveKeys = (const int *)live.cudaData + i;
    }
    std::vector<FastllmNaiveVerifyIndexerItem> items{{bits == 4 ? &refPrimary.indexView : refPrimary.K(), &indices, 0, rows, 0, cap,
        bits == 4 ? refBinding + 4 : refBinding}};
    std::vector<int> cases{16380, 0, 16, 509, 2045, 12285, -1, -2};
    if (paged) cases.push_back(32760);
    for (int past : cases) {
        // At the long-context limit every score ties. The original indexer
        // breaks ties by token position, including the exact cutoff.
        if (past == 32760) Cuda(cudaMemset(ww.cudaData, 0, ww.GetBytes()));
        const bool second = past != 16380;
        auto &data = second ? alternate : primary, &expected = second ? refAlternate : refPrimary;
        primary.Bind(data);
        refPrimary.Bind(expected);
        setLengths(past);
        Require(FastllmCudaBFloat16MatMulBFloat16(normalized, wk, empty, ik, rows, 4096, 128), "wk");
        Require(FastllmCudaBFloat16MatMulBFloat16(normalized, wq, empty, iq, rows, 4096, 2048), "wq");
        Require(FastllmCudaBFloat16MatMulBFloat16(normalized, ww, empty, iw, rows, 4096, 16), "weights");
        Require(FastllmCudaNaiveIndexerKeyNorm(ik, norm, bias), "key norm");
        FastllmCudaNaiveRope(iq, positions, 16, 128, 64, theta);
        FastllmCudaMul(iw, .25f, iw);
        Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, caches, live, 64, 4, 192, 128, 64,
                                                     theta, valueScale, 0, &raw, false),
                "reference append");
        Require(FastllmCudaNaiveBatchVerifyIndexer(iq, iw, items, true, reference, live),
                "reference Indexer");
        FastllmCudaNaiveBatchDSADecode(q, sink, descriptors, 64, 4, 192, 128, reference, ref);
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaDeviceSynchronize());
        Require(Read(data.key) == Read(expected.key) && Read(data.value) == Read(expected.value),
                "DP KV mismatch");
        if (bits == 4) Require(Read(data.indexPool) == Read(expected.indexPool), "FP4 Indexer sidecar mismatch");
        auto referenceIds = Read(indices);
        std::vector<unsigned char> selected(referenceIds.size());
        Cuda(cudaMemcpy(selected.data(), scratch.attentionPartial.cudaData, selected.size(),
                        cudaMemcpyDeviceToHost));
        Require(selected == referenceIds, "ordered DP shortlist mismatch");
        const auto err = Compare(out, ref, rows, "DP sparse attention");
        auto actual = Read(out), packed = Read(arena);
        const B *bf = (const B *)actual.data();
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < cols; col += 32) {
                const B *x = bf + size_t(row) * cols + col;
                float mx = 0;
                for (int i = 0; i < 32; ++i) mx = std::max(mx, std::abs(float(x[i])));
                const int e = mx > 0 ? std::clamp(int(std::ceil(std::log2(mx / 448.f))) + 127, 1, 254) : 1;
                Require(packed[pitch * cols + (col / 128 * pitch + row) * 4 + col / 32 % 4] == e,
                        "DP O scale mismatch");
                for (int i = 0; i < 32; ++i)
                    Require(packed[row * cols + col + i] ==
                                __nv_fp8_e4m3(std::ldexp(float(x[i]), 127 - e)).__x,
                            "DP O pack mismatch");
            }
        printf("DP DSA kv_bits=%d rows=%d paged=%d past=%d ordered_topk=exact KV=exact attention_l2=%.8g O_pack=exact\n",
               bits, rows, paged, past, err);
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
}

// Replay a single batched node with different KV owners. A separate invocation
// of the tested single-slot path supplies each reference output and packed scale.
void BatchedDpDsa(int requests, bool benchmark, int bits = 16, std::vector<int> widths = {}) {
    constexpr int cap = 16640, columns = 8192, qkvCols = 13568;
    if (widths.empty()) widths.assign(requests, 8);
    Require(int(widths.size()) == requests, "wrong request count");
    std::vector<int> start{0};
    for (int n : widths) start.push_back(start.back() + n);
    const int rows = start.back(), pitch = (rows + 3) / 4 * 4, bindingWidth = bits == 4 ? 8 : 4;
    Data x, qkv, wk, wq, weights, norm, bias, sink, pos, live, bindings, out, scoreArena, queries;
    for (auto [tensor, cols] : {std::make_pair(&x, 4096), {&qkv, qkvCols}}) {
        Init(*tensor, BFLOAT16, {1, rows, cols});
        Fill(*tensor, cols, .3f);
    }
    for (auto [tensor, cols] : {std::make_pair(&wk, 128), {&wq, 2048}, {&weights, 16}}) {
        Init(*tensor, BFLOAT16, {cols, 4096});
        Fill(*tensor, cols, .2f);
    }
    Init(norm, FLOAT32, {128}); Fill(norm, 3, .15f, 1);
    Init(bias, FLOAT32, {128}); Fill(bias, 7, .05f);
    Init(sink, FLOAT32, {64}); Fill(sink, 9, .2f);
    Init(pos, FLOAT32, {1, rows}); Init(live, INT32, {rows});
    Init(bindings, INT32, {requests * bindingWidth * 2});
    Init(scoreArena, INT32, {rows * 32768});
    auto pack = [](Data &data, int n) {
        auto *p = static_cast<uint8_t *>(data.cudaData);
        const int pitch = (n + 3) / 4 * 4;
        return FastllmMxfp8Activation{p, p + size_t(pitch) * columns, n, columns, pitch};
    };
    std::vector<std::unique_ptr<Cache>> caches, reference;
    for (int i = 0; i < requests; ++i) {
        caches.emplace_back(new Cache(cap, true, false, 32 + i, 4, bits));
        if (!benchmark) reference.emplace_back(new Cache(cap, true, false, 32 + i, 4, bits));
    }
    FastllmNaiveDecodeScratch scratch, singleScratch;
    Data singleScores, singleQuery;
    if (bits != 16 && !benchmark) Init(singleScores, INT32, {8 * 32768});
    auto binding = (const void *const *)bindings.cudaData;
    if (!benchmark) {
        std::vector<FastllmNaiveRopeCacheRow> slots;
        for (int i = 0; i < requests; ++i)
            for (int row = 0; row < widths[i]; ++row)
                slots.push_back({caches[i]->K(), caches[i]->V(), nullptr, binding + i * bindingWidth});
        auto admitted = [&](int w, int capacity, const Data &storage) {
            return FastllmCudaCanBatchDpDSAVerify(slots, w, capacity, storage, &queries, bits == 4 ? &caches.front()->indexPool : nullptr, &widths);
        };
        Require(admitted(widths.front(), cap, scoreArena), "batched DSA cache layout rejected");
        Require(!admitted(0, cap, scoreArena) && !admitted(widths.front(), 65536, scoreArena) &&
                !admitted(widths.front(), cap, bindings), "batched DSA fallback admission");
        slots.back().cache = binding;
        Require(!admitted(widths.front(), cap, scoreArena), "inconsistent DSA replay binding accepted");
    }
    auto launch = [&] {
        Require(FastllmCudaTryDpDSAVerify(x, qkv, wk, wq, weights, norm, bias, sink, pos, live,
            *caches.front()->K(), *caches.front()->V(), binding, cap, 1e7f, .75f,
            pack(scoreArena, rows), scratch, out, false, widths.front(), &scoreArena, &queries,
            bits == 4 ? &caches.front()->indexPool : nullptr, &widths), "batched DSA admission");
    };
    auto bind = [&](int pass) {
        std::vector<int> lengths(rows);
        std::vector<float> positions(rows);
        for (int i = 0; i < requests; ++i) {
            auto &c = *caches[(i + pass) % requests];
            Cuda(cudaMemcpy((void *)(binding + i * bindingWidth), c.table.cudaData, bindingWidth * sizeof(void *), cudaMemcpyDeviceToDevice));
            for (int j = 0; j < widths[i]; ++j) {
                int n = 16380 + j + i;
                if (pass && i % 3 == 0) n = 0;
                if (pass && i % 3 == 1) n = j == 1 ? 0 : 2045 + j;
                lengths[start[i] + j] = n;
                positions[start[i] + j] = std::max(n - 1, 0);
            }
        }
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), rows * 4, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(pos.cudaData, positions.data(), rows * 4, cudaMemcpyHostToDevice));
    };
    bind(0); launch(); Cuda(cudaDeviceSynchronize());
    cudaGraph_t graph; cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    launch();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    size_t nodes = 0; Cuda(cudaGraphGetNodes(graph, nullptr, &nodes));
    Require(nodes == 1, "batched DSA must capture one node");
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    if (benchmark) {
        cudaEvent_t begin, end;
        Cuda(cudaEventCreate(&begin)); Cuda(cudaEventCreate(&end));
        Cuda(cudaEventRecord(begin, cudaStreamPerThread));
        for (int i = 0; i < 20; ++i) Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaEventRecord(end, cudaStreamPerThread)); Cuda(cudaEventSynchronize(end));
        float ms; Cuda(cudaEventElapsedTime(&ms, begin, end));
        printf("batched DSA requests=%d kernel=%.1f us\n", requests, ms * 50);
        Cuda(cudaEventDestroy(begin)); Cuda(cudaEventDestroy(end));
    } else for (int pass = 0; pass < 3; ++pass) {
        if (pass == 2) Cuda(cudaMemset(weights.cudaData, 0, weights.GetBytes()));
        bind(pass); Cuda(cudaGraphLaunch(exec, cudaStreamPerThread)); Cuda(cudaDeviceSynchronize());
        auto output = Read(out);
        std::vector<unsigned char> activation(size_t(pitch) * (columns + columns / 32));
        Cuda(cudaMemcpy(activation.data(), scoreArena.cudaData, activation.size(), cudaMemcpyDeviceToHost));
        for (int i = 0; i < requests; ++i) {
            const int width = widths[i], refPitch = (width + 3) / 4 * 4;
            Data xx, qq, pp, ll, expected, arena;
            auto slice = [&](Data &v, Data &parent, int cols) {
                v.dataType = parent.dataType; v.UpdateUnitSize();
                auto shape = parent.dims; shape[parent.dims.size() == 3 ? 1 : shape.size()-1] = width;
                v.Resize(shape); v.FakeFrom(parent, size_t(start[i]) * cols * parent.unitSize);
            };
            slice(xx, x, 4096); slice(qq, qkv, qkvCols); Init(pp, FLOAT32, {1,width}); Init(ll, INT32, {width});
            Cuda(cudaMemcpy(pp.cudaData, (float *)pos.cudaData + start[i], width * sizeof(float), cudaMemcpyDeviceToDevice));
            Cuda(cudaMemcpy(ll.cudaData, (int *)live.cudaData + start[i], width * sizeof(int), cudaMemcpyDeviceToDevice));
            Init(arena, INT32, {refPitch * (columns + columns / 32) / 4});
            auto &ref = *reference[(i + pass) % requests], &actual = *caches[(i + pass) % requests];
            Require(FastllmCudaTryDpDSAVerify(xx, qq, wk, wq, weights, norm, bias, sink, pp, ll,
                *ref.K(), *ref.V(), (const void *const *)ref.table.cudaData, cap, 1e7f, .75f,
                pack(arena, width), singleScratch, expected, false, width, &singleScores, &singleQuery,
                bits == 4 ? &ref.indexPool : nullptr), "single DSA reference");
            Cuda(cudaDeviceSynchronize());
            Require(Read(actual.key) == Read(ref.key) && Read(actual.value) == Read(ref.value), "batched DSA KV");
            if (bits == 4) Require(Read(actual.indexPool) == Read(ref.indexPool), "batched DSA Indexer cache");
            std::vector<int> ids(width * 2048), refIds(ids.size());
            Cuda(cudaMemcpy(ids.data(), (int *)scratch.attentionPartial.cudaData + size_t(start[i]) * 2048,
                            ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
            Cuda(cudaMemcpy(refIds.data(), singleScratch.attentionPartial.cudaData, refIds.size() * sizeof(int),
                            cudaMemcpyDeviceToHost));
            Require(ids == refIds, "batched DSA ordered shortlist");
            auto results = Read(expected), bytes = Read(arena);
            Require(!memcmp(output.data() + size_t(start[i]) * columns * 2, results.data(), results.size()),
                    "batched DSA attention");
            Require(!memcmp(activation.data() + size_t(start[i]) * columns, bytes.data(), width * columns),
                    "batched DSA packed values");
            for (int tile = 0; tile < columns / 128; ++tile)
                Require(!memcmp(activation.data() + pitch * columns + (tile * pitch + start[i]) * 4,
                                bytes.data() + refPitch * columns + tile * refPitch * 4, width * 4), "batched DSA scales");
        }
        printf("batched DSA kv_bits=%d requests=%d pass=%d topk=exact KV=exact attention=exact O_pack=exact nodes=%zu\n", bits, requests, pass, nodes);
    }
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
}
