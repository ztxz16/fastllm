// Compare a captured FP8/FP4 plan with the existing eager packed-KV operators.
// The oracle owns separate cache storage and checks the encoded bytes as well
// as attention/top-k, including compressor boundaries and pointer relocation.
static void RunPackedDecodeRegression(bool fp4) {
    int checked = 0;
    for (int heads : {8, 16})
    for (int batch : (largeBatch ? std::vector<int>{17, 32, 64, 128} : std::vector<int>{1, 2, 8, 16}))
    for (int ratio : {0, 1, 2})
    for (bool source : {false, true})
    for (int mask : {0, 1, 2}) {
        if ((!ratio && (source || mask)) || (longContext && (batch != 1 || heads != 8))) continue;
        FastllmV41FullDecodeConfig c;
        c.ringMode = 1; c.cmpMode = fp4 ? 3 : 1; c.indexMode = fp4 ? 2 : 1;
        c.ringRowBytes = 528; c.cmpRowBytes = fp4 ? 288 : 528; c.indexRowBytes = fp4 ? 68 : 132;
        c.batch = batch; c.heads = heads; c.ratio = ratio;
        c.kvSource = source; c.indexSource = ratio != 0;
        c.maxKeys = longContext ? 262144 : largeBatch ? 8192 : 32768;
        c.writeCandidates = mask == 1; c.useCandidates = mask == 2;
        c.blockSize = 8; c.candidateTopK = 16;
        c.softmaxScale = 1.f / std::sqrt(512.f);
        const int maxBlocks = (c.maxKeys + c.blockSize - 1) / c.blockSize;
        Data q, kv, out, iq, iw, sink, top, cand, raw, rawScore, joined, scores, latent, nk, norm;
        Fill(q, {1, batch, heads, 512}, 1); Fill(kv, {1, batch, 512}, 2);
        Fill(out, q.dims, 3); Fill(iq, {1, batch, c.indexHeads, 128}, 4);
        Fill(iw, {1, batch, c.indexHeads}, 5, FLOAT32); Fill(sink, {heads}, 6, FLOAT32);
        Fill(top, {1, batch, c.topK}, 7, INT32); Fill(cand, {1, batch, maxBlocks}, 8, INT8);
        Fill(raw, {1, batch, 512}, 9, FLOAT32); Fill(rawScore, raw.dims, 10, FLOAT32);
        Fill(joined, {batch, std::max(1, ratio), 512}, 11, FLOAT32);
        Fill(scores, joined.dims, 12, FLOAT32); Fill(latent, {1, batch, 512}, 13);
        Fill(nk, {1, batch, 128}, 14); Fill(norm, {512}, 15, FLOAT32);
        Data q0(q), kv0(kv), iq0(iq), nk0(nk);
        struct Cache { Data ring, cmp, keys, tail, tailScore; };
        std::vector<Cache> initial(batch), relocated(batch), expected(batch);
        auto packed = [&](Data &dst, int rows, int width, int seed, int mode) {
            Data values; Fill(values, {1, rows, width}, seed);
            Check(FastllmCudaDeepSeekV41QuantizeKV(values, dst, mode, mode == 3 ? 16 : 32), "packed initial pack");
        };
        for (auto *set : {&initial, &relocated})
            for (int i = 0; i < batch; ++i) {
                auto &cache = (*set)[i];
                packed(cache.ring, c.window, 512, 20 + i, c.ringMode);
                if (ratio) {
                    packed(cache.cmp, c.maxKeys, 512, 40 + i, c.cmpMode);
                    packed(cache.keys, c.maxKeys, 128, 60 + i, c.indexMode);
                }
                Fill(cache.tail, {1, 1, 512}, 80 + i, FLOAT32);
                Fill(cache.tailScore, cache.tail.dims, 100 + i, FLOAT32);
            }
        FastllmV41FullDecodeBuffers b;
        b.q = q.cudaData; b.kv = kv.cudaData; b.output = out.cudaData;
        b.indexQ = iq.cudaData; b.indexWeights = iw.cudaData; b.sink = sink.cudaData;
        b.topk = top.cudaData; b.candidates = cand.cudaData;
        b.rawKV = raw.cudaData; b.rawScore = rawScore.cudaData;
        b.joined = joined.cudaData; b.scores = scores.cudaData;
        b.latents = latent.cudaData; b.newIndexKeys = nk.cudaData; b.compressNorm = norm.cudaData;
        Check(!FastllmCudaV41CreateFullDecode(c, b), "packed plan accepted missing packed scratch");
        Data saved; Fill(saved, {1, batch, c.ringRowBytes}, 0, INT8);
        b.savedKV = saved.cudaData;
        auto unsupported = c; unsupported.ringRowBytes = 512;
        Check(!FastllmCudaV41CreateFullDecode(unsupported, b), "packed plan accepted invalid row stride");
        void *plan = FastllmCudaV41CreateFullDecode(c, b);
        Check(plan, "packed plan create");
        std::vector<FastllmV41FullDecodeRequest> requests(batch);
        auto upload = [&](int position, std::vector<Cache> &caches) {
            for (int i = 0; i < batch; ++i) {
                auto &cache = caches[i];
                requests[i] = {cache.ring.cudaData, cache.cmp.cudaData, cache.keys.cudaData,
                    cache.tail.cudaData, cache.tailScore.cudaData, position + i};
            }
            Check(FastllmCudaV41UploadFullDecode(plan, requests.data(), batch), "packed upload");
        };
        upload(1, initial);
        Cuda(cudaMemsetAsync(cand.cudaData, 1, cand.GetBytes(), cudaStreamPerThread));
        Check(FastllmCudaV41PrepareFullDecode(plan) && FastllmCudaV41FinishFullDecode(plan), "packed warmup");
        Cuda(cudaDeviceSynchronize());
        cudaGraph_t graph; cudaGraphExec_t exec;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Check(FastllmCudaV41PrepareFullDecode(plan) && FastllmCudaV41FinishFullDecode(plan), "packed capture");
        Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Cuda(cudaGraphInstantiate(&exec, graph, 0));
        for (int position : (longContext ? std::vector<int>{131071, 131072, 262127}
             : largeBatch ? std::vector<int>{1, 2, 126, 127, 128, 510, 511, 1023, 4095, 8063}
             : std::vector<int>{1, 2, 126, 127, 128, 510, 511, 1023, 4095, 16385, 32750})) {
            auto &caches = position >= 510 ? relocated : initial;
            Check(initial[0].ring.cudaData != relocated[0].ring.cudaData, "packed relocation alias");
            CopyDevice(q, q0); CopyDevice(kv, kv0); CopyDevice(iq, iq0); CopyDevice(nk, nk0);
            std::vector<uint8_t> hostMask(size_t(batch) * maxBlocks, 1);
            if (mask == 2)
                for (int i = 0; i < batch; ++i)
                    for (int j = 0; j < maxBlocks; ++j)
                        hostMask[size_t(i) * maxBlocks + j] = position % 5 == 0 ? 0 :
                            position % 3 == 0 ? 1 : ((j + i) % 11 < 2 ? 255 : 0);
            Cuda(cudaMemcpyAsync(cand.cudaData, hostMask.data(), hostMask.size(),
                                 cudaMemcpyHostToDevice, cudaStreamPerThread));
            std::vector<Data> refQ(batch), refKV(batch), refIQ(batch), refOut(batch), refTop(batch), refCand(batch);
            for (int i = 0; i < batch; ++i) {
                auto &a = caches[i]; auto &e = expected[i];
                e.ring.CopyFrom(a.ring); e.cmp.CopyFrom(a.cmp); e.keys.CopyFrom(a.keys);
                e.tail.CopyFrom(a.tail); e.tailScore.CopyFrom(a.tailScore);
                const int pos = position + i, count = ratio ? (pos + 1) / ratio : 0;
                auto cloneRow = [&](Data &dst, const Data &src, int width, const std::vector<int> &dims) {
                    Data view; View(view, src, size_t(i) * width * src.unitSize, dims); dst.CopyFrom(view);
                };
                cloneRow(refQ[i], q0, heads * 512, {1, 1, heads, 512});
                cloneRow(refKV[i], kv0, 512, {1, 1, 512});
                cloneRow(refIQ[i], iq0, c.indexHeads * 128, {1, 1, c.indexHeads, 128});
                auto rotary = [&](Data &x, int start, int mode, bool inverse = false) {
                    Check(FastllmCudaDeepSeekV41RotaryQuant(x, c.ropeDim, c.ropeBase, start, 1,
                        inverse, c.originalSeqLen, c.ropeFactor, c.betaFast, c.betaSlow,
                        mode, -1, mode == 3 ? 16 : 32), "packed reference rotary");
                };
                rotary(refQ[i], pos, 0); rotary(refKV[i], pos, 1);
                Data currentRaw, currentScore;
                View(currentRaw, raw, size_t(i) * 512 * 4, {1, 1, 512});
                View(currentScore, rawScore, size_t(i) * 512 * 4, {1, 1, 512});
                if (source) {
                    if ((pos + 1) % ratio == 0) {
                        Data block, scoreBlock, l, k, lp, kp;
                        Fill(block, {1, ratio, 512}, 0, FLOAT32); Fill(scoreBlock, block.dims, 0, FLOAT32);
                        if (ratio == 2) {
                            Cuda(cudaMemcpyAsync(block.cudaData, e.tail.cudaData, 512 * 4,
                                cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                            Cuda(cudaMemcpyAsync(scoreBlock.cudaData, e.tailScore.cudaData, 512 * 4,
                                cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                        }
                        Cuda(cudaMemcpyAsync((float *)block.cudaData + (ratio - 1) * 512,
                            currentRaw.cudaData, 512 * 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                        Cuda(cudaMemcpyAsync((float *)scoreBlock.cudaData + (ratio - 1) * 512,
                            currentScore.cudaData, 512 * 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                        Check(FastllmCudaDeepSeekV41Compress(block, ratio == 2 ? &scoreBlock : nullptr,
                            norm, ratio, c.normEps, l), "packed reference compress");
                        cloneRow(k, nk0, 128, {1, 1, 128});
                        rotary(l, pos / ratio * ratio, 3); rotary(k, pos / ratio * ratio, 2);
                        Check(FastllmCudaDeepSeekV41QuantizeKV(l, lp, c.cmpMode, fp4 ? 16 : 32) &&
                            FastllmCudaDeepSeekV41QuantizeKV(k, kp, c.indexMode, 32), "packed reference append pack");
                        Cuda(cudaMemcpyAsync((uint8_t *)e.cmp.cudaData + size_t(pos / ratio) * c.cmpRowBytes,
                            lp.cudaData, c.cmpRowBytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                        Cuda(cudaMemcpyAsync((uint8_t *)e.keys.cudaData + size_t(pos / ratio) * c.indexRowBytes,
                            kp.cudaData, c.indexRowBytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                    } else {
                        CopyDevice(e.tail, currentRaw); CopyDevice(e.tailScore, currentScore);
                    }
                }
                Data cmp, keys, weights, indexScores;
                if (ratio) {
                    View(cmp, e.cmp, 0, {1, count, c.cmpRowBytes}); View(keys, e.keys, 0, {1, count, c.indexRowBytes});
                    View(weights, iw, size_t(i) * c.indexHeads * 4, {1, 1, c.indexHeads});
                    rotary(refIQ[i], pos, 2);
                    Check(FastllmCudaDeepSeekV41IndexerScore(refIQ[i], weights, keys, ratio, pos,
                        indexScores), "packed reference indexer");
                    if (mask == 1)
                        Check(FastllmCudaDeepSeekV41CandidateBlocks(indexScores, c.blockSize,
                            c.candidateTopK, ratio, pos, refCand[i]), "packed reference candidates");
                    if (mask == 2) {
                        Fill(refCand[i], {1, 1, (count + c.blockSize - 1) / c.blockSize}, 0, INT8);
                        Cuda(cudaMemcpyAsync(refCand[i].cudaData, hostMask.data() + size_t(i) * maxBlocks,
                            refCand[i].GetBytes(), cudaMemcpyHostToDevice, cudaStreamPerThread));
                    }
                    Check(FastllmCudaDeepSeekV41IndexerTopK(indexScores, mask == 2 ? &refCand[i] : nullptr,
                        c.topK, ratio, pos, c.blockSize, refTop[i]), "packed reference topk");
                }
                Check(FastllmCudaDeepSeekV41SparseAttention(refQ[i], refKV[i], &e.ring,
                    ratio ? &cmp : nullptr, ratio ? &refTop[i] : nullptr, sink, c.window, pos,
                    c.softmaxScale, refOut[i]), "packed reference attention");
                rotary(refOut[i], pos, 1, true);
                Data encoded;
                Check(FastllmCudaDeepSeekV41QuantizeKV(refKV[i], encoded, 1, 32), "packed reference ring pack");
                Cuda(cudaMemcpyAsync((uint8_t *)e.ring.cudaData + size_t(pos % c.window) * 528,
                    encoded.cudaData, 528, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            }
            upload(position, caches);
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread)); Cuda(cudaDeviceSynchronize());
            for (int i = 0; i < batch; ++i) {
                Data view;
                View(view, q, size_t(i) * heads * 512 * 2, refQ[i].dims); Equal(view, refQ[i], "packed q");
                View(view, kv, size_t(i) * 512 * 2, refKV[i].dims); Equal(view, refKV[i], "packed kv");
                View(view, out, size_t(i) * heads * 512 * 2, refOut[i].dims); Equal(view, refOut[i], "packed output");
                Equal(caches[i].ring, expected[i].ring, "packed ring bytes");
                Equal(caches[i].tail, expected[i].tail, "packed raw tail");
                Equal(caches[i].tailScore, expected[i].tailScore, "packed raw scores");
                if (ratio) {
                    Equal(caches[i].cmp, expected[i].cmp, "packed compressed bytes");
                    Equal(caches[i].keys, expected[i].keys, "packed index bytes");
                    View(view, iq, size_t(i) * c.indexHeads * 128 * 2, refIQ[i].dims);
                    Equal(view, refIQ[i], "packed index query");
                    View(view, top, size_t(i) * c.topK * 4, refTop[i].dims); Equal(view, refTop[i], "packed topk");
                    if (mask == 1) {
                        View(view, cand, size_t(i) * maxBlocks, refCand[i].dims);
                        Equal(view, refCand[i], "packed candidate mask");
                    }
                }
                ++checked;
            }
        }
        Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
        FastllmCudaV41DestroyFullDecode(plan);
        std::cout << (fp4 ? "FP4" : "FP8") << " heads=" << heads << " batch=" << batch << " ratio=" << ratio
                  << " source=" << source << " mask=" << mask << " checked=" << checked << std::endl;
    }
    std::cout << "PASS " << checked << (fp4 ? " FP4" : " FP8") << " full decode rows; packed bytes, boundaries, relocation" << std::endl;
}
