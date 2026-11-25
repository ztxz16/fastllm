// Pure CUDA TP uses the same persistent rank workers, per-thread execution
// contexts and graph-safe collective implementation as NaiveN05FlashModel.
class TensorRuntime final : public Runtime {
    struct Rank {
        WeightMap weights;
        std::unique_ptr<Executor> executor;
        std::unique_ptr<Decoder> decoder;
        bool visionReady = false;
    };
    basellm &owner;
    bool pooledPrefill;
    std::vector<int> devices;
    std::vector<std::unique_ptr<Rank>> ranks;
    PersistentWorkerGroup workers;
    std::mutex mutex;

    template<class F> void run(F f) {
        std::vector<std::exception_ptr> errors(devices.size());
        workers.Run(devices, [&](int i) {
            check(cudaSetDevice(devices[i]));
            auto previous = GetExecutor();
            struct Restore { void *previous; ~Restore() { SetCurrentThreadExecutor(previous); } } restore{previous};
            SetCurrentThreadExecutor(ranks[i]->executor.get());
            try { f(i); } catch (...) { FastllmNcclAbort("Inkling TP rank failed"); throw; }
        }, errors);
        for (const auto &e : errors) if (e) std::rethrow_exception(e);
    }

    // Materialize one tensor on its final device. QKV weights contain one
    // row range per logical projection. W_o is sharded along its input axis.
    static void weightShard(WeightMap &dst, const std::string &name, const Data &src, const Config &c,
                            const Placement &p) {
        auto dims = src.dims;
        if (dims.empty() || src.Count(0) == 0) return;
        const auto ends = [&](std::string_view suffix) {
            return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        const bool packed = ends(".attn.packed.weight");
        const bool columns = ends(".attn.wo_ud.weight") || ends(".attn.k_sconv.static_w") ||
                             ends(".attn.v_sconv.static_w") || ends(".self_attn.o_proj.weight") || ends(".mlp.down_proj.weight");
        const bool vocab = name == "model.llm.embed.weight" || name == "model.llm.unembed.weight" ||
                           name == "model.llm.draft.markov_head.folded.weight" ||
                           name == "model.llm.draft.markov_head.folded.bias";
        const bool heads = ends(".attn.q_l2_clip.scale") || ends(".attn.rel_l2_clip.scale") ||
            ends(".attn.k_sconv.weight") || ends(".attn.v_sconv.weight") ||
            ends(".self_attn.q_proj.weight") || ends(".self_attn.k_proj.weight") || ends(".self_attn.v_proj.weight") ||
            ends(".self_attn.k_proj_ctx.weight") || ends(".self_attn.v_proj_ctx.weight") ||
            ends(".mlp.gate_proj.weight") || ends(".mlp.up_proj.weight");
        if ((columns && dims.at(1) % p.world) || (heads && dims[0] % p.world))
            throw std::runtime_error("Unaligned Inkling TP weight " + name);
        if (columns) dims.at(1) /= p.world;
        if (packed || heads) dims[0] /= p.world;
        if (vocab) dims[0] = p.vocabRows;
        auto &value = dst.weight.try_emplace(name, src.dataType, dims).first->second;
        value.dataDevice = CUDA; value.dataDeviceIds = {p.gpu};
        value.Allocate(false);
        const size_t unit = src.unitSize;
        auto *out = static_cast<char *>(value.cudaData);
        const char *input = reinterpret_cast<const char *>(src.dataDevice == CPU ? src.cpuData : src.cudaData);
        if (!input || src.unitSizeDiv != 1) throw std::runtime_error("Unsupported Inkling TP weight " + name);
        auto copyRows = [&](size_t offset, int count, size_t width) {
            check(cudaMemcpyAsync(out, input + offset * width, size_t(count) * width, cudaMemcpyDefault, cudaStreamPerThread));
            out += size_t(count) * width;
        };
        if (packed) {
            const int q = c.heads * c.headDim, kv = c.kvHeads * c.headDim;
            const int taps = (src.dims[0] - 2 * q - c.heads * (c.relative + 1) - 2 * kv) / 2;
            int first = 0;
            for (int rows : {q, c.heads * c.relative, q, c.heads, kv, taps, kv, taps}) {
                if (rows % p.world) throw std::runtime_error("Inkling TP attention partition is not head aligned");
                if (rows) copyRows(first + p.rank * (rows / p.world), rows / p.world, src.dims[1] * unit);
                first += rows;
            }
        } else if (columns) {
            const size_t width = dims[1] * unit;
            check(cudaMemcpy2DAsync(out, width, input + p.rank * width, src.dims[1] * unit, width,
                                    dims[0], cudaMemcpyDefault, cudaStreamPerThread));
        } else if (heads || vocab) {
            const size_t width = src.GetBytes() / src.dims[0];
            const int begin = p.rank * dims[0];
            const int live = std::min(dims[0], std::max(0, src.dims[0] - begin));
            if (live) copyRows(begin, live, width);
            if (live < dims[0]) check(cudaMemsetAsync(out, 0, (dims[0] - live) * width, cudaStreamPerThread));
        } else check(cudaMemcpyAsync(out, input, value.GetBytes(), cudaMemcpyDefault, cudaStreamPerThread));
    }

    void ensureCache(Data &root) const {
        root.multiDeviceData = true;
        root.dataDevice = CUDA;
        root.dataDeviceIds = devices;
        for (int gpu : devices) {
            auto &local = root.multiDeviceDatas[gpu];
            if (!local) {
                local = new Data(root.dataType); local->isKVCache = true;
                local->dataDevice = CUDA; local->dataDeviceIds = {gpu};
            }
        }
    }

    static void syncCache(Data &root, int first, int world) {
        const auto &local = *root.multiDeviceDatas.at(first);
        if (local.dims.size() != 3) return;
        auto dims = local.dims; dims[0] *= world;
        root.Resize(dims);
        root.pageLen = local.pageLen; root.lastPageLen = local.lastPageLen;
        root.isKVCache = true;
    }

    class Pipeline final : public basellm::DecodePipeline {
        TensorRuntime &runtime;
        std::vector<ResponseContext *> contexts;
        bool pending[2]{};
    public:
        std::vector<std::unique_ptr<basellm::DecodePipeline>> local;
        Pipeline(TensorRuntime &r, const std::vector<ResponseContext *> &c)
            : runtime(r), contexts(c), local(r.devices.size()) {}
        ~Pipeline() override { runtime.run([&](int i) { local[i].reset(); }); }
        bool Submit(int out, int in) override {
            std::vector<int> submitted(local.size());
            runtime.run([&](int i) { submitted[i] = local[i]->Submit(out, in); });
            if (!std::all_of(submitted.begin(), submitted.end(), [&](int x) { return x == submitted[0]; }))
                throw std::runtime_error("Inkling TP decode pipelines disagree on request state");
            // Keep the TP forward RNG sequence for requests admitted after a chained batch.
            if (submitted[0] && contexts.size() > 1) (void)newSamplingSeed();
            pending[out] = submitted[0];
            return submitted[0];
        }
        std::vector<int> Collect(int slot) override {
            std::vector<std::vector<int>> tokens(local.size());
            runtime.run([&](int i) { tokens[i] = local[i]->Collect(slot); });
            for (size_t i = 1; i < tokens.size(); ++i)
                if (tokens[i] != tokens.front()) throw std::runtime_error("Inkling TP ranks sampled different tokens");
            pending[slot] = false;
            if (!pending[1 - slot] && contexts.size() > 1) {
                for (auto *ctx : contexts) for (auto &[key, value] : ctx->pastKeyValues) {
                    syncCache(key, runtime.devices.front(), runtime.devices.size());
                    syncCache(value, runtime.devices.front(), runtime.devices.size());
                }
            }
            return std::move(tokens.front());
        }
        void Drain() override { runtime.run([&](int i) { local[i]->Drain(); }); }
    };

public:
    TensorRuntime(const Config &c, const Placement &p, basellm &model, const std::map<std::string, TensorRange> &files)
        : owner(model), pooledPrefill(p.tensorDevices.size() == 2 && !c.draft.steps), devices(p.tensorDevices) {
        const int world = devices.size();
        if (c.kvHeads % world || c.heads % world || c.experts % world ||
            (c.draft.steps && (c.draft.heads % world || c.draft.kvHeads % world || c.draft.intermediate % world)) ||
            std::any_of(p.experts.begin(), p.experts.end(), [](ExpertDevice e) { return e != ExpertDevice::Cuda; }))
            throw std::runtime_error("Inkling TP requires divisible attention/expert groups on CUDA");
        if (!FastllmInitNccl(devices)) throw std::runtime_error("Inkling TP communicator initialization failed");
#if CUDART_VERSION >= 11020
        // The small-message collective reads peer inputs directly. Eager Inkling
        // buffers come from the CUDA async pool, independently of ordinary P2P mappings.
        for (int device : devices) {
            cudaMemPool_t pool;
            check(cudaDeviceGetMemPool(&pool, device));
            for (int peer : devices) if (peer != device) {
                int supported;
                check(cudaDeviceCanAccessPeer(&supported, peer, device));
                if (supported) {
                    cudaMemAccessDesc access{};
                    access.location = {cudaMemLocationTypeDevice, peer};
                    access.flags = cudaMemAccessFlagsProtReadWrite;
                    check(cudaMemPoolSetAccess(pool, &access, 1));
                }
            }
        }
#endif
        const int vocabRows = (1 + (c.vocab - 1) / (8 * world)) * 8;
        for (int gpu : devices) {
            check(cudaSetDevice(gpu));
            auto r = std::make_unique<Rank>();
            r->executor = std::make_unique<Executor>(true);
            r->executor->SetFirstDevice("cuda:" + std::to_string(gpu));
            ranks.push_back(std::move(r));
        }
        // Preparation is sequential to bound temporary checkpoint buffers. Every
        // collective subsequently runs on the persistent worker of its device.
        for (int i = 0; i < world; ++i) {
            auto placement = p;
            placement.gpu = devices[i]; placement.rank = i; placement.world = world;
            placement.tensorDevices.clear(); placement.vocabRows = vocabRows;
            check(cudaSetDevice(placement.gpu));
            pooledPrefill &= FastllmCudaRuntimeArch() == 120;
            for (const auto &[name, weight] : model.weight.weight)
                if (name.compare(0, 10, "model.llm.") == 0) weightShard(ranks[i]->weights, name, weight, c, placement);
            check(cudaStreamSynchronize(cudaStreamPerThread));
            auto localFiles = files;
            for (auto &[name, entry] : localFiles) {
                if (entry.shape.empty() || entry.shape[0] != c.experts) continue;
                const size_t part = (entry.end - entry.start) / world;
                entry.start += part * i; entry.end = entry.start + part; entry.shape[0] /= world;
            }
            auto local = c; local.heads /= world; local.kvHeads /= world;
            local.draft.heads /= world; local.draft.kvHeads /= world; local.draft.intermediate /= world;
            ranks[i]->decoder = std::make_unique<Decoder>(local, placement, model, localFiles, &ranks[i]->weights);
        }
        check(cudaSetDevice(p.gpu));
        // Keep the exportable checkpoint in host memory once both rank-local
        // CUDA copies are ready. The loader can place merged tensors on GPU 0.
        for (auto &[name, weight] : model.weight.weight)
            if (weight.dataDevice == CUDA) weight.ToDevice(CPU);
    }
    ~TensorRuntime() override {
        run([&](int i) { ranks[i]->decoder.reset(); ranks[i]->weights.weight.clear(); });
    }
    bool hasRankCaches() const override { return true; }
    bool supportsPooledPrefill() const override { return pooledPrefill; }
    void warmupDecodeGraph() override {
        const auto seed = newSamplingSeed();
        run([&](int i) {
            ranks[i]->decoder->setRequestSeed(seed);
            ranks[i]->decoder->warmupDecodeGraph();
        });
    }
    void setPrefixCache(bool enabled) override { run([&](int i) { ranks[i]->decoder->setPrefixCache(enabled); }); }
    bool popPending(Data &key, int previous, int &token) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (key.multiDeviceDatas.empty()) return false;
        bool ready = false;
        for (size_t i = 0; i < devices.size(); ++i) {
            int local = -1;
            bool popped = ranks[i]->decoder->popPending(*key.multiDeviceDatas.at(devices[i]), previous, local);
            if (i && (popped != ready || (popped && local != token)))
                throw std::runtime_error("Inkling TP ranks disagree on the verified prefix");
            if (popped) token = local;
            ready = popped;
        }
        return ready;
    }
    int prefix(ResponseContext &ctx, int limit, bool restore) override {
        if (limit <= 0 || !ctx.multimodalInput.empty()) return 0;
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<int> lengths(devices.size());
        std::vector<std::shared_ptr<void>> owners(devices.size());
        run([&](int i) {
            ResponseContext local;
            local.currentTokens = ctx.currentTokens; local.generationConfig = ctx.generationConfig;
            local.pastKeyValues.resize(1);
            lengths[i] = ranks[i]->decoder->prefix(local, limit, restore);
            if (restore) owners[i] = std::move(local.pastKeyValues.front().first.kvStorageOwner);
        });
        const int common = *std::min_element(lengths.begin(), lengths.end());
        if (restore && common == limit) {
            auto &root = ctx.pastKeyValues.front().first;
            ensureCache(root);
            for (size_t i = 0; i < devices.size(); ++i)
                root.multiDeviceDatas.at(devices[i])->kvStorageOwner = std::move(owners[i]);
        }
        return common;
    }
    std::unique_ptr<basellm::DecodePipeline> pipeline(const std::vector<ResponseContext *> &contexts) override {
        for (auto *ctx : contexts)
            if (ctx->pastKeyValues.empty() || ctx->pastKeyValues.front().first.multiDeviceDatas.size() != ranks.size()) return {};
        auto result = std::make_unique<Pipeline>(*this, contexts);
        run([&](int i) { result->local[i] = ranks[i]->decoder->pipeline(contexts); });
        for (const auto &r : result->local) if (!r) return {};
        return result;
    }
    std::vector<int> forward(const Data &ids, const std::vector<Data *> &positions, const std::vector<int> &lengths,
                             const std::vector<std::pair<Data *, Data *>> &cache, const std::vector<GenerationConfig> &sampling,
                             const LastTokensManager &history, std::vector<std::vector<float> *> *logits,
                             const ImageInputs *images = nullptr) override {
        std::lock_guard<std::mutex> lock(mutex);
        const_cast<Data &>(ids).ToDevice(CPU);
        for (auto *p : positions) {
            if (!p) throw std::runtime_error("Invalid Inkling request");
            p->ToDevice(CPU);
        }
        if (images) for (const auto &entry : *images)
            for (auto *tensor : entry.second) if (tensor) tensor->ToDevice(CPU);
        for (auto [k, v] : cache) { ensureCache(*k); ensureCache(*v); }
        bool pool = owner.SupportsPrefillBatching() && !images && lengths.size() > 1 && lengths.front() > 1;
        if (pool) {
            const int key = pooledPromptKey(lengths.front(), sampling.front());
            for (size_t b = 0; pool && b < lengths.size(); ++b) {
                const auto *pos = positions[b];
                pool = key >= 0 && key == pooledPromptKey(lengths[b], sampling[b]) &&
                       pos->dataType == FLOAT32 && pos->Count(0) == lengths[b] &&
                       *reinterpret_cast<const float *>(pos->cpuData) == 0;
            }
        }
        const auto seed = newSamplingSeed();
        std::vector<uint64_t> seeds;
        if (pool) {
            seeds.resize(lengths.size());
            seeds.front() = seed;
            std::generate(seeds.begin() + 1, seeds.end(), newSamplingSeed);
        }
        std::vector<int> result;
        run([&](int i) {
            auto &rank = *ranks[i];
            if (images && !images->empty() && !rank.visionReady) {
                for (const auto &[name, weight] : owner.weight.weight)
                    if (name.compare(0, 13, "model.visual.") == 0)
                        rank.weights.weight[name].CopyFrom(weight);
                rank.visionReady = true;
            }
            std::vector<std::pair<Data *, Data *>> local;
            local.reserve(cache.size());
            for (auto [k, v] : cache)
                local.emplace_back(k->multiDeviceDatas.at(devices[i]), v->multiDeviceDatas.at(devices[i]));
            rank.decoder->setRequestSeed(seed, seeds);
            auto tokens = rank.decoder->forward(ids, positions, lengths, local, sampling, history, logits, images);
            if (!i) result = std::move(tokens);
        });
        for (auto [k, v] : cache) { syncCache(*k, devices.front(), devices.size()); syncCache(*v, devices.front(), devices.size()); }
        return result;
    }
};
