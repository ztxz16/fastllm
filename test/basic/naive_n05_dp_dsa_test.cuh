// Dense dyadic operands give exactly representable FP32 dot products in every
// split-K order, so the unfused reference can check the ordered shortlist exactly.
void DpDsaVerify(int rows, bool paged) {
    const int cap = paged ? 32768 : 16640;
    constexpr int qcols = 64 * 192, cols = 64 * 128, width = qcols + 4 * 192 + 4 * 128;
    constexpr float theta = 10000000.f, valueScale = .75f;
    Cache primary(cap, paged, false, 11, 4), alternate(cap, paged, false, 32, 4);
    Cache refPrimary(cap, paged, false, 11, 4), refAlternate(cap, paged, false, 32, 4);
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
    auto binding = (const void *const *)primary.table.cudaData;
    auto fused = [&](const void *const *table, bool overlap = false) {
        return FastllmCudaTryDpDSAVerify(normalized, raw, wk, wq, ww, norm, bias, sink, positions, live,
                                         *primary.K(), *primary.V(), table, cap, theta, valueScale, target,
                                         scratch, out, overlap);
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
        caches.push_back({refPrimary.K(), refPrimary.V(), &views[i], refBinding});
        auto &d = descriptors[i];
        d.key = refPrimary.key.cudaData;
        d.value = refPrimary.value.cudaData;
        d.cache = refBinding;
        d.capacity = cap;
        d.keyStride = 896;
        d.selectedCount = 2048;
        d.pageLen = paged ? 128 : 0;
        d.keyPages = (const int *)refPrimary.kp.cudaData;
        d.valuePages = (const int *)refPrimary.vp.cudaData;
        d.indices = (const int *)indices.cudaData + size_t(i) * 2048;
        d.liveKeys = (const int *)live.cudaData + i;
    }
    std::vector<FastllmNaiveVerifyIndexerItem> items{{refPrimary.K(), &indices, 0, rows, 0, cap, refBinding}};
    std::vector<int> cases{16380, 2045, -1, -2};
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
        printf("DP DSA rows=%d paged=%d past=%d ordered_topk=exact KV=exact attention_l2=%.8g O_pack=exact\n",
               rows, paged, past, err);
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
}
