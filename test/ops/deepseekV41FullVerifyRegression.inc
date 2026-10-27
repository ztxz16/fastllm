// Full verification graphs must preserve the eager chunk's causal attention,
// compressor stream and deferred ring writes, including changing cache pointers.
static void RunVerifyRegression() {
    int checked = 0;
    for (int heads : {8, 17, 64})
    for (int tokens : {2, 3, 4, 5, 6, 7, 8})
        for (int ratio : {0, 1, 2})
            for (int mask : {0, 1, 2}) {
                if ((!ratio && mask) || (longContext && heads != 8) ||
                    (sharedCandidates && (heads != 8 || mask != 2)))
                    continue;
                FastllmV41FullDecodeConfig c;
                c.batch = tokens;
                c.heads = heads;
                c.indexHeads = indexHeads;
                c.verifying = true;
                c.ratio = ratio;
                c.kvSource = c.indexSource = ratio != 0;
                c.maxKeys = longContext ? 1048576 : 32768;
                c.candidateTopK = 16;
                if (sharedCandidates || mask == 1) c.blockSize = 8;
                c.writeCandidates = mask == 1;
                c.useCandidates = mask == 2;
                c.softmaxScale = 1 / std::sqrt(512.f);
                const int maxBlocks = (c.maxKeys + c.blockSize - 1) / c.blockSize;
                Data q, kv, out, iq, iw, sink, top, cand, raw, rawScore, joined, scores, latent,
                    newKeys, norm, saved, savedRaw, savedScore;
                Fill(q, {1, tokens, heads, 512}, 1);
                Fill(kv, {1, tokens, 512}, 2);
                Fill(out, q.dims, 3);
                Fill(iq, {1, tokens, indexHeads, 128}, 4);
                Fill(iw, {1, tokens, indexHeads}, 5, FLOAT32);
                Fill(sink, {heads}, 6, FLOAT32);
                Fill(top, {1, tokens, 512}, 7, INT32);
                Fill(cand, {1, tokens, maxBlocks}, 8, INT8);
                Fill(raw, {1, tokens, 512}, 9, FLOAT32);
                Fill(rawScore, raw.dims, 10, FLOAT32);
                Fill(joined, {tokens, std::max(1, ratio), 512}, 11, FLOAT32);
                Fill(scores, joined.dims, 12, FLOAT32);
                Fill(latent, {1, tokens, 512}, 13);
                Fill(newKeys, {1, tokens, 128}, 14);
                Fill(norm, {512}, 15, FLOAT32);
                Fill(saved, kv.dims, 16);
                Fill(savedRaw, {1, tokens + 1, 512}, 17, FLOAT32);
                Fill(savedScore, savedRaw.dims, 18, FLOAT32);
                Data q0(q), k0(kv), iq0(iq), nk0(newKeys), qr(q), kr(kv), iqr(iq), ring, relocated,
                    ring0, cmp, rc, keys, rk, tail, tailScore, tail0, tailScore0;
                Fill(ring, {1, 128, 512}, 20);
                Fill(relocated, ring.dims, 21);
                Fill(cmp, {1, c.maxKeys, 512}, 22);
                Fill(keys, {1, c.maxKeys, 128}, 23);
                Fill(tail, {1, 1, 512}, 24, FLOAT32);
                Fill(tailScore, tail.dims, 25, FLOAT32);
                tail0.CopyFrom(tail);
                tailScore0.CopyFrom(tailScore);
                rc.CopyFrom(cmp);
                rk.CopyFrom(keys);
                ring0.CopyFrom(ring);
                FastllmV41FullDecodeBuffers b;
                b.q = q.cudaData;
                b.kv = kv.cudaData;
                b.output = out.cudaData;
                b.indexQ = iq.cudaData;
                b.indexWeights = iw.cudaData;
                b.sink = sink.cudaData;
                b.topk = top.cudaData;
                b.candidates = cand.cudaData;
                b.rawKV = raw.cudaData;
                b.rawScore = rawScore.cudaData;
                b.joined = joined.cudaData;
                b.scores = scores.cudaData;
                b.latents = latent.cudaData;
                b.newIndexKeys = newKeys.cudaData;
                b.compressNorm = norm.cudaData;
                b.savedKV = saved.cudaData;
                b.savedRawKV = savedRaw.cudaData;
                b.savedRawScore = savedScore.cudaData;
                CandidateReuseProducer producer;
                producer.Setup(c, b);
                void *plan = FastllmCudaV41CreateFullDecode(c, b);
                Check(plan, "verify plan");
                std::vector<FastllmV41FullDecodeRequest> requests(tokens);
                auto upload = [&](int pos, Data &r) {
                    for (int i = 0; i < tokens; ++i)
                        requests[i] = {r.cudaData,    cmp.cudaData,       keys.cudaData,
                                       tail.cudaData, tailScore.cudaData, pos + i};
                    Check(producer.Upload(requests.data(), tokens), "candidate producer upload");
                    Check(FastllmCudaV41UploadFullDecode(plan, requests.data(), tokens),
                          "verify upload");
                };
                upload(1, ring);
                producer.Launch();
                Check(FastllmCudaV41PrepareFullDecode(plan) && FastllmCudaV41FinishFullDecode(plan),
                      "verify warmup");
                Cuda(cudaDeviceSynchronize());
                cudaGraph_t graph;
                cudaGraphExec_t exec;
                Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                producer.Launch();
                Check(FastllmCudaV41PrepareFullDecode(plan) && FastllmCudaV41FinishFullDecode(plan),
                      "verify capture");
                Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                Cuda(cudaGraphInstantiate(&exec, graph, 0));
                for (int pos : (longContext
                                ? std::vector<int>{131071, 131072, 262143, 262144, 1048560}
                                : std::vector<int>{1, 2, 125, 126, 127, 128, 509, 510, 511, 1021, 1022, 1023, 4095,
                                8192, 16383})) {
                    Data &r = pos >= 509 ? relocated : ring;
                    CopyDevice(q, q0);
                    CopyDevice(kv, k0);
                    CopyDevice(iq, iq0);
                    CopyDevice(newKeys, nk0);
                    CopyDevice(qr, q0);
                    CopyDevice(kr, k0);
                    CopyDevice(iqr, iq0);
                    CopyDevice(cmp, rc);
                    CopyDevice(keys, rk);
                    ring0.CopyFrom(r);
                    std::vector<uint8_t> hostMask(size_t(tokens) * maxBlocks, 1);
                    if (mask == 2) {
                        for (int row = 0; row < tokens; ++row)
                            for (int block = 0; block < maxBlocks; ++block)
                                hostMask[size_t(row) * maxBlocks + block] = pos % 5 == 0 ? 0 :
                                    pos % 3 == 0 ? 1 : (block + row * 3 + pos) % 11 < 2;
                    }
                    // Compaction must interpret every nonzero byte as selected.
                    if (sharedCandidates)
                        for (size_t i = 0; i < hostMask.size(); ++i)
                            if (hostMask[i]) hostMask[i] = i % 3 == 0 ? 255 : i % 3 == 1 ? 7 : 1;
                    Cuda(cudaMemcpyAsync(cand.cudaData, hostMask.data(), hostMask.size(),
                                         cudaMemcpyHostToDevice, cudaStreamPerThread));
                    auto rotary = [&](Data &x, int start, int step, int mode,
                                      bool inverse = false) {
                        Check(FastllmCudaDeepSeekV41RotaryQuant(
                                  x, c.ropeDim, c.ropeBase, start, step, inverse, c.originalSeqLen,
                                  c.ropeFactor, c.betaFast, c.betaSlow, mode, -1,
                                  mode == 3 ? 16 : 32),
                              "verify rotary");
                    };
                    rotary(qr, pos, 1, 0);
                    rotary(kr, pos, 1, 1);
                    const int old = ratio ? pos / ratio : 0, rem = ratio ? pos % ratio : 0;
                    const int count = ratio ? (pos + tokens) / ratio : 0, added = count - old;
                    Data expectedRaw, expectedScore, refLatent, refNewKeys, refCmp, refKeys, refTop,
                        refCand, refScores;
                    Data cmpReference(rc), keyReference(rk);
                    if (ratio) {
                        Fill(expectedRaw, {1, rem + tokens, 512}, 1, FLOAT32);
                        Fill(expectedScore, expectedRaw.dims, 1, FLOAT32);
                        if (rem) {
                            Cuda(cudaMemcpyAsync(expectedRaw.cudaData, tail.cudaData, 512 * 4,
                                                 cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                            Cuda(cudaMemcpyAsync(expectedScore.cudaData, tailScore.cudaData,
                                                 512 * 4, cudaMemcpyDeviceToDevice,
                                                 cudaStreamPerThread));
                        }
                        Cuda(cudaMemcpyAsync((float *)expectedRaw.cudaData + rem * 512,
                                             raw.cudaData, raw.GetBytes(), cudaMemcpyDeviceToDevice,
                                             cudaStreamPerThread));
                        Cuda(cudaMemcpyAsync((float *)expectedScore.cudaData + rem * 512,
                                             rawScore.cudaData, rawScore.GetBytes(),
                                             cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                        Data part, sp;
                        View(part, expectedRaw, 0, {1, added * ratio, 512});
                        View(sp, expectedScore, 0, part.dims);
                        Fill(refLatent, {1, added, 512}, 1);
                        Fill(refNewKeys, {1, added, 128}, 1);
                        Check(FastllmCudaDeepSeekV41Compress(part, ratio == 2 ? &sp : nullptr, norm,
                                                             ratio, c.normEps, refLatent),
                              "verify compressor");
                        for (int j = 0; j < added; ++j) {
                            int token = (j + 1) * ratio - rem - 1;
                            Cuda(cudaMemcpyAsync((__nv_bfloat16 *)refNewKeys.cudaData + j * 128,
                                                 (const __nv_bfloat16 *)nk0.cudaData + token * 128,
                                                 128 * 2, cudaMemcpyDeviceToDevice,
                                                 cudaStreamPerThread));
                        }
                        rotary(refLatent, old * ratio, ratio, 3);
                        rotary(refNewKeys, old * ratio, ratio, 2);
                        Cuda(cudaMemcpyAsync((char *)cmpReference.cudaData + size_t(old) * 512 * 2,
                                             refLatent.cudaData, refLatent.GetBytes(),
                                             cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                        Cuda(cudaMemcpyAsync((char *)keyReference.cudaData + size_t(old) * 128 * 2,
                                             refNewKeys.cudaData, refNewKeys.GetBytes(),
                                             cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                        View(refCmp, cmpReference, 0, {1, count, 512});
                        View(refKeys, keyReference, 0, {1, count, 128});
                        rotary(iqr, pos, 1, 2);
                        Check(FastllmCudaDeepSeekV41IndexerScore(iqr, iw, refKeys, ratio, pos,
                                                                 refScores),
                              "verify scores");
                        if (mask == 1)
                            Check(FastllmCudaDeepSeekV41CandidateBlocks(
                                      refScores, c.blockSize, c.candidateTopK, ratio, pos, refCand),
                                  "verify candidates");
                        if (mask == 2) {
                            Fill(refCand, {1, tokens, (count + c.blockSize - 1) / c.blockSize}, 0,
                                 INT8);
                            const int blocks = (count + c.blockSize - 1) / c.blockSize;
                            for (int row = 0; row < tokens; ++row)
                                Cuda(cudaMemcpyAsync((uint8_t*)refCand.cudaData + size_t(row) * blocks,
                                    hostMask.data() + size_t(row) * maxBlocks, blocks,
                                    cudaMemcpyHostToDevice, cudaStreamPerThread));
                        }
                        Check(FastllmCudaDeepSeekV41IndexerTopK(
                                  refScores, mask == 2 ? &refCand : nullptr, c.topK, ratio, pos,
                                  c.blockSize, refTop),
                              "verify topk");
                    }
                    Data refOut;
                    Check(FastllmCudaDeepSeekV41SparseAttention(
                              qr, kr, &r, ratio ? &refCmp : nullptr, ratio ? &refTop : nullptr,
                              sink, c.window, pos, c.softmaxScale, refOut),
                          "verify attention");
                    rotary(refOut, pos, 1, 1, true);
                    upload(pos, r);
                    Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
                    Cuda(cudaDeviceSynchronize());
                    producer.Validate(c, hostMask, pos);
                    Equal(q, qr, "verify q");
                    Equal(kv, kr, "verify kv");
                    Equal(out, refOut, "verify output");
                    Equal(saved, kr, "deferred ring rows");
                    Equal(r, ring0, "committed ring untouched");
                    Equal(tail, tail0, "committed tail untouched");
                    Equal(tailScore, tailScore0, "tail score untouched");
                    if (ratio) {
                        Equal(cmp, cmpReference, "compressed append");
                        Equal(keys, keyReference, "index append");
                        Data sv;
                        View(sv, savedRaw, 0, expectedRaw.dims);
                        Equal(sv, expectedRaw, "rollback stream");
                        if (ratio == 2) {
                            Data ss;
                            View(ss, savedScore, 0, expectedScore.dims);
                            Equal(ss, expectedScore, "rollback scores");
                        }
                        Equal(iq, iqr, "verify index q");
                        for (int i = 0; i < tokens; ++i) {
                            Data a, e;
                            const int width = std::min(count, c.topK);
                            View(a, top, size_t(i) * 512 * 4, {1, 1, width});
                            View(e, refTop, size_t(i) * width * 4, a.dims);
                            Equal(a, e, "verify topk");
                        }
                    }
                    ++checked;
                }
                Cuda(cudaGraphExecDestroy(exec));
                Cuda(cudaGraphDestroy(graph));
                FastllmCudaV41DestroyFullDecode(plan);
                std::cout << "verify heads=" << heads << " tokens=" << tokens << " ratio=" << ratio << " mask=" << mask
                          << " cumulative=" << checked << std::endl;
            }
    std::cout << "PASS " << checked
              << " full verify graph replays, causal chunks and rollback buffers" << std::endl;
}
