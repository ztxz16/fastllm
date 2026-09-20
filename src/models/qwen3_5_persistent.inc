// Included inside namespace fastllm after the native snapshot helpers. This
// adapter deliberately shares those helpers and the existing paged KV writers.
#if defined(FASTLLM_DISK_PREFIX_CACHE) && defined(USE_CUDA)
class Qwen35PersistentCache {
    using Json = json11::Json;
    using Clock = std::chrono::steady_clock;
    static constexpr size_t HostLimit = size_t(512) << 20;
    static constexpr size_t BufferBytes = size_t(8) << 20;
    struct ExportState {
        std::unique_ptr<DiskPrefixCache::WriteSession> writer;
        std::unique_ptr<DiskPrefixCache::ReadSession> inherited;
        Json::array kv;
        int issued = 0;
        uint64_t reservation = 0;
        std::atomic<bool> failed{false};
        std::string identity, family, provenance;
        Json source;
    };
    struct Pages {
        PagedCacheManager *pool = nullptr;
        int layer = 0, side = 0, rank = 0, begin = 0;
        size_t pageBytes = 0;
        std::vector<int> ids;
        void *ready = nullptr;
        ~Pages() {
            if (ready) {
                const int previous = FastllmCudaGetDevice();
                FastllmCudaSetDevice(rank); FastllmCudaEventDestroy(ready);
                FastllmCudaSetDevice(previous);
            }
            if (pool) {
                std::vector<int> held;
                for (int id : ids) if (id >= 0) held.push_back(id);
                if (!held.empty()) pool->ReleasePageIndices(held);
            }
        }
    };
    struct Job {
        std::shared_ptr<ExportState> state;
        std::vector<std::shared_ptr<Pages>> pages;
        std::shared_ptr<Qwen35LinearPrefixSnapshot> snapshot;
        std::string prefix, key;
        int length = 0;
        size_t stateBytes = 0;
        std::shared_ptr<std::vector<float>> image;
    };
    Qwen3_5Model *model;
    std::string directory, baseFamily, encoderIdentity, policy;
    uint64_t diskLimit = uint64_t(256) << 30;
    std::shared_ptr<DiskPrefixCache> store;
    std::shared_ptr<DiskPrefixCache> imageStore;
    std::string identity, family;
    std::map<ResponseContext*, std::shared_ptr<ExportState>> exports;
    std::map<std::string,Json> provenanceSources;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::deque<Job> queue;
    std::thread worker;
    bool stopping = false, working = false;
    size_t queuedStateBytes = 0;
    int checkpointJobs = 0;
    std::map<int, size_t> pinnedGpuBytes;
    std::map<int, size_t> gpuBudget;
    double prefillMsPerToken = 0, nativeReadMsPerToken = 0, convertedReadMsPerToken = 0;
    uint64_t storageDevice = 0;
    std::atomic<uint64_t> writes{0}, reads{0}, committed{0}, restored{0}, skipped{0};

    static void Check(bool ok, const char *message) { DiskPrefixCache::Require(ok, message); }
    static std::string Env(const char *name) {
        const char *v = std::getenv(name); return v ? v : "";
    }
    static std::string Dtype(DataType type) {
        return type == DataType::FP8_E4M3 ? "fp8_e4m3" :
               type == DataType::FLOAT16 ? "float16" : "unsupported";
    }
    static std::vector<int> Ints(const Json &value) {
        Check(value.is_array() && value.array_items().size() <= 262144, "ssd_dimensions");
        std::vector<int> out;
        for (const auto &x : value.array_items()) {
            Check(x.is_number() && x.number_value() == x.int_value(), "ssd_integer");
            out.push_back(x.int_value());
        }
        return out;
    }
    static size_t TensorBytes(const Data &data) { return data.GetBytes(); }
    static size_t SnapshotBytes(const Qwen35LinearPrefixSnapshot &snapshot) {
        size_t size = 0;
        for (auto &layer : snapshot.layers) if (layer.linear) {
            for (auto *cache : {&layer.first, &layer.second}) {
                if (cache->multiDevice) for (auto &local : cache->locals) size += TensorBytes(local.second);
                else size += TensorBytes(cache->single);
            }
        }
        for (auto &kv : snapshot.dflashKeyValues) size += kv.first.GetBytes() + kv.second.GetBytes();
        return size;
    }
    Json RuntimeLayout() const {
        Json::array pools;
        for (int layer = 0; layer < model->block_cnt; ++layer) {
            if (Qwen35LayerIsLinearAttention(model, layer)) continue;
            for (int side = 0; side < 2; ++side)
                for (auto ref : model->GetPagedKVCacheManagers(layer, side == 0)) {
                    Check(ref.second && ref.second->dims.size() == 4, "ssd_pool_not_ready");
                    pools.emplace_back(Json::object{{"layer",layer},{"side",side},{"rank",ref.first},
                        {"page",ref.second->pageLen},{"heads",ref.second->dims[2]},
                        {"head_dim",ref.second->dims[3]}});
                }
        }
        Check(!pools.empty(), "ssd_pool_not_ready");
        return Json::object{{"abi",2},{"layers",model->block_cnt},{"pools",pools},
            {"chunk",model->GetChunkedPrefillSize()},{"atype",(int)model->dataType},
            {"draft_layers",model->dflashLayers},{"draft_heads",model->dflashKvHeads},
            {"draft_dim",model->dflashHeadDim},{"draft_window",model->dflashSlidingWindow}};
    }
    bool Ensure() {
        if (worker.joinable()) return true;
        try {
            const auto layout = RuntimeLayout();
            family = DiskPrefixCache::Digest(baseFamily + layout.dump());
            identity = DiskPrefixCache::Digest(family + Dtype(model->kvCacheDataType));
            auto targetStore = std::make_shared<DiskPrefixCache>(directory, identity, diskLimit, family, 3);
            auto featuresStore = std::make_shared<DiskPrefixCache>(directory,encoderIdentity,diskLimit,encoderIdentity,1);
            store = std::move(targetStore); imageStore = std::move(featuresStore);
            for (int layer = 0; layer < model->block_cnt; ++layer)
                for (int side = 0; side < 2; ++side)
                    for (auto ref : model->GetPagedKVCacheManagers(layer,side == 0)) {
                        auto pool = ref.second;
                        gpuBudget[ref.first] += (size_t)pool->maxPages*pool->pageLen*pool->dims[2]*pool->dims[3]*pool->unitSize;
                    }
            for (auto &budget : gpuBudget) budget.second = std::min(size_t(256)<<20,budget.second/10);
            struct stat filesystem{};
            if (stat(directory.c_str(),&filesystem) == 0) storageDevice = filesystem.st_dev;
            std::ifstream profile(store->Directory() / ("cost-"+identity+".json"));
            std::string profileText((std::istreambuf_iterator<char>(profile)),{}), error;
            auto cost = Json::parse(profileText,error);
            if (error.empty() && cost["device"].string_value() == std::to_string(storageDevice)) {
                prefillMsPerToken = cost["prefill"].number_value();
                nativeReadMsPerToken = cost["native_read"].number_value();
                convertedReadMsPerToken = cost["converted_read"].number_value();
            }
            worker = std::thread([this] { Work(); });
            return true;
        } catch (const std::exception &e) {
            std::fprintf(stderr,"[Prefix SSD] unavailable: %s.\n", e.what());
            return false;
        }
    }
    static std::string Prefix(const ResponseContext *ctx, int length) {
        std::string extra;
        int pages = length / fastllm::GetPageLen();
        const auto &keys = ctx->prefixCacheBasePageKeys;
        if (!keys.empty()) {
            Check((int)keys.size() >= pages, "ssd_media_identity");
            for (int i = 0; i < pages; ++i) {
                AppendPrefixKeyInt(extra, (uint32_t)keys[i].size()); extra += keys[i];
            }
        }
        return DiskPrefixCache::PrefixKey(ctx->allTokens, length, extra);
    }
    static void SetProvenance(ResponseContext *ctx, const std::string &provenance) {
        ctx->prefixCacheProvenance = provenance;
        ctx->prefixCachePageKeys = ctx->prefixCacheBasePageKeys;
        if (provenance.empty()) return;
        int pages = (int)ctx->allTokens.size() / fastllm::GetPageLen();
        if (ctx->prefixCachePageKeys.empty()) {
            ctx->prefixCachePageKeys.resize(pages);
            for (int p = 0; p < pages; ++p)
                for (int t = 0; t < fastllm::GetPageLen(); ++t)
                    AppendPrefixKeyInt(ctx->prefixCachePageKeys[p], ctx->allTokens[p * fastllm::GetPageLen() + t]);
        }
        for (auto &key : ctx->prefixCachePageKeys) {
            AppendPrefixKeyInt(key, (uint32_t)provenance.size()); key += provenance;
        }
    }
    Json SaveTensor(DiskPrefixCache::WriteSession &writer, const Data &tensor) {
        Check(tensor.dataDevice == DataDevice::CPU && tensor.cpuData, "ssd_snapshot_cpu");
        size_t bytes = tensor.GetBytes();
        auto chunks = writer.StoreBytes(bytes, [&](size_t offset, char *out, size_t count) {
            memcpy(out, tensor.cpuData + offset, count);
        });
        Json::array strides;
        for (auto value : tensor.strides) strides.emplace_back((double)value);
        return Json::object{{"dims",tensor.dims},{"strides",strides},{"expansion",tensor.expansionDims},
            {"dtype",(int)tensor.dataType},{"bytes",(double)bytes},{"chunks",chunks},
            {"transposed",tensor.isLinearAttentionTransposed}};
    }
    void LoadTensor(DiskPrefixCache::ReadSession &reader, const Json &value, Data &tensor) {
        auto dims = Ints(value["dims"]);
        auto strides = Ints(value["strides"]);
        Check(!dims.empty() && dims.size() <= 6 && strides.size() == dims.size(), "ssd_tensor_rank");
        uint64_t count = 1;
        for (int dim : dims) { Check(dim > 0 && (uint64_t)dim <= HostLimit / count, "ssd_tensor_shape"); count *= dim; }
        int type = value["dtype"].int_value();
        Check(type == (int)DataType::FLOAT32 || type == (int)DataType::FLOAT16 || type == (int)DataType::BFLOAT16,
              "ssd_state_dtype");
        tensor.dataType = (DataType)type; tensor.UpdateUnitSize();
        tensor.dataDevice = DataDevice::CPU; tensor.Resize(dims);
        auto expansion = Ints(value["expansion"]);
        if (!expansion.empty()) {
            Check(expansion.size() == dims.size(),"ssd_state_expansion");
            uint64_t capacity = tensor.unitSize;
            for (size_t i = 0; i < dims.size(); ++i) {
                Check(expansion[i] >= dims[i] && (uint64_t)expansion[i] <= HostLimit/capacity,"ssd_state_expansion");
                capacity *= expansion[i];
            }
            tensor.Expansion(expansion);
        }
        Check(tensor.strides == std::vector<uint64_t>(strides.begin(),strides.end()), "ssd_state_strides");
        size_t bytes = tensor.GetBytes();
        Check(bytes <= HostLimit && value["bytes"].number_value() == (double)bytes, "ssd_state_bytes");
        tensor.Allocate();
        reader.ReadBytes(value["chunks"], bytes, [&](size_t offset, const char *p, size_t n) {
            memcpy(tensor.cpuData + offset, p, n);
        });
        tensor.isKVCache = true; tensor.isLinearAttentionTransposed = value["transposed"].bool_value();
    }
    Json SaveLinear(DiskPrefixCache::WriteSession &writer, const Qwen35LinearPrefixSnapshotCache &cache) {
        Json::array locals, ranges;
        for (auto &it : cache.locals)
            locals.emplace_back(Json::object{{"rank",it.first},{"tensor",SaveTensor(writer,it.second)}});
        for (auto &it : cache.tpRanges) for (auto range : it.second)
            ranges.emplace_back(Json::array{it.first,range.first,range.second});
        return Json::object{{"multi",cache.multiDevice},{"locals",locals},
            {"single",cache.multiDevice ? Json() : SaveTensor(writer,cache.single)},
            {"devices",cache.dataDeviceIds},{"layout",(int)cache.tpLayout},{"axis",cache.tpAxis},
            {"ranges",ranges},{"global_dims",cache.tpGlobalDims},{"dims",cache.dims},
            {"dtype",(int)cache.dataType},{"transposed",cache.transposed}};
    }
    void LoadLinear(DiskPrefixCache::ReadSession &reader, const Json &value, Qwen35LinearPrefixSnapshotCache &cache) {
        cache.multiDevice = value["multi"].bool_value(); cache.dataDeviceIds = Ints(value["devices"]);
        cache.tpLayout = (TensorParallelLayoutType)value["layout"].int_value(); cache.tpAxis = value["axis"].int_value();
        cache.tpGlobalDims = Ints(value["global_dims"]); cache.dims = Ints(value["dims"]);
        cache.dataType = (DataType)value["dtype"].int_value(); cache.transposed = value["transposed"].bool_value();
        for (auto &entry : value["ranges"].array_items()) {
            auto r = Ints(entry); Check(r.size() == 3 && r[1] >= 0 && r[2] >= r[1], "ssd_tp_range");
            cache.tpRanges[r[0]].push_back({r[1],r[2]});
        }
        if (cache.multiDevice) {
            Check(value["locals"].array_items().size() == cache.dataDeviceIds.size(), "ssd_linear_ranks");
            for (auto &entry : value["locals"].array_items()) {
                int rank = entry["rank"].int_value();
                Check(std::find(cache.dataDeviceIds.begin(),cache.dataDeviceIds.end(),rank) != cache.dataDeviceIds.end()
                    && cache.locals.count(rank) == 0, "ssd_linear_rank");
                LoadTensor(reader, entry["tensor"],cache.locals[rank]);
            }
        } else LoadTensor(reader,value["single"],cache.single);
        cache.valid = true;
    }

    void Work();
    void Export(Job &job, void *buffer, void *stream);
    bool Import(ResponseContext *ctx, const DiskPrefixCache::Checkpoint &candidate);
public:
    void ObservePrefill(int tokens, double ms) {
        if (tokens <= 0 || ms <= 0 || !std::isfinite(ms)) return;
        std::lock_guard<std::mutex> lock(mutex);
        const double sample = ms/tokens;
        prefillMsPerToken = prefillMsPerToken > 0 ? 0.8*prefillMsPerToken+0.2*sample : sample;
    }
    void SaveCost() {
        if (!store) return;
        std::string contents;
        { std::lock_guard<std::mutex> lock(mutex);
          contents = Json(Json::object{{"device",std::to_string(storageDevice)},
              {"prefill",prefillMsPerToken},{"native_read",nativeReadMsPerToken},
              {"converted_read",convertedReadMsPerToken}}).dump(); }
        auto path = store->Directory() / ("cost-"+identity+".json");
        std::string pattern = path.string()+".XXXXXX";
        std::vector<char> temporary(pattern.begin(),pattern.end()); temporary.push_back(0);
        int fd = mkstemp(temporary.data());
        if (fd < 0) return;
        size_t offset = 0;
        while (offset < contents.size()) {
            ssize_t count = write(fd,contents.data()+offset,contents.size()-offset);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) break;
            offset += count;
        }
        close(fd);
        // Calibration loss is a safe auto miss, never a correctness failure.
        if (offset == contents.size()) rename(temporary.data(),path.c_str());
        unlink(temporary.data());
    }
    static std::shared_ptr<Qwen35PersistentCache> Create(Qwen3_5Model *model) {
        if (Env("FASTLLM_PREFIX_CACHE_DIR").empty()) return {};
        Check(Env("FASTLLM_PREFIX_CACHE_MANIFEST_VERSION") == "2", "ssd_identity_not_configured");
        return std::shared_ptr<Qwen35PersistentCache>(new Qwen35PersistentCache(model));
    }
    explicit Qwen35PersistentCache(Qwen3_5Model *model) : model(model),
        directory(Env("FASTLLM_PREFIX_CACHE_DIR")),baseFamily(Env("FASTLLM_PREFIX_CACHE_FAMILY_IDENTITY")),
        encoderIdentity(Env("FASTLLM_PREFIX_CACHE_ENCODER_IDENTITY")),policy(Env("FASTLLM_PREFIX_CACHE_RESTORE_POLICY")) {
        Check(DiskPrefixCache::IsDigest(baseFamily), "ssd_invalid_family");
        auto quota = Env("FASTLLM_PREFIX_CACHE_DISK_BYTES");
        if (!quota.empty()) diskLimit = std::stoull(quota);
        if (policy.empty()) policy = "auto";
    }
    ~Qwen35PersistentCache() { Stop(); }
    void Created(ResponseContext *ctx) {
        if (ctx->multimodalInput.empty()) {
            ctx->prefixCachePageKeys.clear();
            std::string parent = "qwen35-text-prefix-v2";
            const int pageLen = fastllm::GetPageLen();
            for (size_t start = 0; start + pageLen <= ctx->allTokens.size(); start += pageLen) {
                std::string content = parent;
                for (int offset = 0; offset < pageLen; ++offset)
                    AppendPrefixKeyInt(content,ctx->allTokens[start+offset]);
                parent = DiskPrefixCache::Digest(content);
                ctx->prefixCachePageKeys.push_back(parent);
            }
        }
        ctx->prefixCacheBasePageKeys = ctx->prefixCachePageKeys;
        ctx->prefixCacheProvenance.clear();
    }
    void Replanned(ResponseContext *ctx) {
        ctx->prefixCacheBasePageKeys = ctx->prefixCachePageKeys;
        SetProvenance(ctx,ctx->prefixCacheProvenance);
    }
    void Removed(ResponseContext *ctx) { std::lock_guard<std::mutex> lock(mutex); exports.erase(ctx); }
    void Stop() {
        { std::lock_guard<std::mutex> lock(mutex); exports.clear(); stopping = true; }
        changed.notify_all();
        if (worker.joinable()) worker.join();
        SaveCost();
    }
    bool Wait(int timeoutMs) {
        std::unique_lock<std::mutex> lock(mutex);
        bool complete = changed.wait_for(lock,std::chrono::milliseconds(std::max(0,timeoutMs)),[&] { return queue.empty() && !working; });
        lock.unlock(); SaveCost(); return complete;
    }
    std::string Stats() const {
        return Json(Json::object{{"written_bytes",(double)writes.load()},{"read_bytes",(double)reads.load()},
            {"committed",(double)committed.load()},{"restored_tokens",(double)restored.load()},
            {"skipped",(double)skipped.load()}}).dump();
    }
    void Capture(ResponseContext *ctx, const std::shared_ptr<Qwen35LinearPrefixSnapshot> &snapshot);
    void Prepare(ResponseContext *ctx);
    size_t LoadImage(const std::string &key, size_t expected, std::vector<float> &output) {
        if (!Ensure() || policy == "never") return 0;
        try {
            auto reader = imageStore->BeginRead();
            auto state = reader.Load(DiskPrefixCache::Digest(encoderIdentity+key));
            Check(state["count"].number_value() == (double)expected && expected <= HostLimit/sizeof(float),"ssd_image_shape");
            std::vector<float> data(expected);
            reader.ReadBytes(state["chunks"],expected*sizeof(float),[&](size_t offset,const char *p,size_t n) {
                memcpy((char*)data.data()+offset,p,n);
            });
            output.insert(output.end(),data.begin(),data.end()); reads += reader.ReadBytesCount();
            printf("[Vision] Image SSD cache hit (%zu bytes).\n",expected*sizeof(float)); fflush(stdout);
            return expected;
        } catch (const std::exception &) { return 0; }
    }
    void SaveImage(const std::string &key,const float *data,size_t count) {
        if (!Ensure() || count > HostLimit/sizeof(float)) return;
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping || queue.size() >= 16 || count*sizeof(float) > HostLimit-queuedStateBytes) { ++skipped; return; }
        Job job; job.key = DiskPrefixCache::Digest(encoderIdentity+key);
        job.image = std::make_shared<std::vector<float>>(data,data+count); job.stateBytes = count*sizeof(float);
        queuedStateBytes += job.stateBytes; queue.push_back(std::move(job)); changed.notify_one();
    }
    void Restored(ResponseContext *ctx) {
        if (ctx->cacheLen <= 0) return;
        const bool converted = ctx->intParams["ssd_loaded_len"] > 0 && ctx->intParams["ssd_source_fp16"] &&
            model->kvCacheDataType == DataType::FP8_E4M3;
        if (ctx->intParams["ssd_loaded_len"] <= 0 || converted) {
            std::shared_ptr<Qwen35LinearPrefixSnapshot> snapshot;
            { std::lock_guard<std::mutex> guard(Qwen35LinearPrefixSnapshotsMutex());
              auto &items = Qwen35LinearPrefixSnapshots()[model];
              for (auto &item : items) {
                  if (item->cachedLen != ctx->cacheLen || item->tokens.size() > ctx->allTokens.size() ||
                      !std::equal(item->tokens.begin(),item->tokens.end(),ctx->allTokens.begin())) continue;
                  if (item->mediaPageKeys.empty() != ctx->prefixCachePageKeys.empty() ||
                      item->mediaPageKeys.size() > ctx->prefixCachePageKeys.size() ||
                      !std::equal(item->mediaPageKeys.begin(),item->mediaPageKeys.end(),ctx->prefixCachePageKeys.begin())) continue;
                  snapshot = item; break;
              }
            }
            if (converted) ctx->intParams["ssd_force_checkpoint"] = ctx->cacheLen;
            if (snapshot && (converted || ctx->intParams["ssd_branch_boundary"] == ctx->cacheLen)) Capture(ctx,snapshot);
            if (!converted) return;
        }
        restored += ctx->cacheLen;
        printf("[Prefix SSD] restored: tokens=%d source_dtype=%s target_dtype=%s.\n",ctx->cacheLen,
            ctx->intParams["ssd_source_fp16"] ? "float16" : "fp8_e4m3",Dtype(model->kvCacheDataType).c_str());
        fflush(stdout); ctx->intParams["ssd_loaded_len"] = 0;
    }
};

void Qwen35PersistentCache::Capture(ResponseContext *ctx,
        const std::shared_ptr<Qwen35LinearPrefixSnapshot> &snapshot) {
    if (!ctx || !ctx->CanUsePagedPrefixCache() || !Qwen35LinearPrefixCacheEnabled()) return;
    const int length = snapshot->cachedLen, chunk = model->GetChunkedPrefillSize();
    if (length > ctx->inputTokens || chunk <= 0 || length % chunk || snapshot->mtpValid) return;
    if (!Ensure()) return;
    std::shared_ptr<ExportState> state;
    { std::lock_guard<std::mutex> lock(mutex);
      auto &slot = exports[ctx];
      if (!slot) {
          slot = std::make_shared<ExportState>(); slot->identity = identity; slot->family = family;
          slot->provenance = ctx->prefixCacheProvenance;
          auto source = provenanceSources.find(slot->provenance);
          if (source != provenanceSources.end()) slot->source = source->second;
          uint64_t kvBytes = 0;
          for (int layer = 0; layer < model->block_cnt; ++layer)
              for (int side = 0; side < 2; ++side)
                  for (auto ref : model->GetPagedKVCacheManagers(layer,side == 0))
                      kvBytes += (uint64_t)ctx->inputTokens*ref.second->dims[2]*ref.second->dims[3]*ref.second->unitSize;
          slot->reservation = kvBytes + (ctx->inputTokens/8192 + ctx->multimodalPrefixSpans.size()+2)*SnapshotBytes(*snapshot) + (16<<20);
      }
      state = slot; }
    if (state->failed) return;
    if (state->issued == 0 && ctx->cacheLen > 0) {
        // A branch may start from an already durable hot prefix. Adopt its
        // immutable references rather than re-exporting it in a different
        // grouping/order and trying to republish the same checkpoint key.
        try {
            const auto parentPrefix = Prefix(ctx,ctx->cacheLen);
            const auto parentKey = DiskPrefixCache::Digest(parentPrefix+state->provenance+
                (snapshot->dflashValid ? ":draft" : ":plain"));
            auto inherited = std::make_unique<DiskPrefixCache::ReadSession>(store->BeginRead());
            const auto parent = inherited->Load(parentKey);
            Check(parent["length"].int_value() == ctx->cacheLen && parent["prefix"].string_value() == parentPrefix &&
                parent["kv_dtype"].string_value() == Dtype(model->kvCacheDataType) &&
                parent["provenance"].string_value() == state->provenance &&
                parent["layout"].dump() == RuntimeLayout().dump(),"ssd_parent_identity");
            state->kv = parent["kv"].array_items();
            uint64_t inheritedBytes = 0;
            for (const auto &bundle : state->kv) inheritedBytes += (uint64_t)bundle["bytes"].number_value();
            state->reservation -= std::min(state->reservation,inheritedBytes);
            state->inherited = std::move(inherited);
            state->issued = ctx->cacheLen;
        } catch (const std::exception &) {
            // No durable parent: the currently owned GPU pages can still be
            // exported, subject to the ordinary queue and memory budgets.
        }
    }
    if (length <= state->issued) return;
    bool checkpoint = length % 8192 == 0 || ctx->inputTokens - length < chunk ||
        length == ctx->intParams["ssd_branch_boundary"] || length == ctx->intParams["ssd_force_checkpoint"];
    for (auto &span : ctx->multimodalPrefixSpans)
        if (length >= span.end && length - span.end < chunk) checkpoint = true;
    Job job; job.state = state; job.length = length;
    job.prefix = Prefix(ctx,length);
    job.key = DiskPrefixCache::Digest(job.prefix + state->provenance + (snapshot->dflashValid ? ":draft" : ":plain"));
    if (checkpoint) { job.snapshot = snapshot; job.stateBytes = SnapshotBytes(*snapshot); }
    std::map<int,size_t> need;
    const int previousDevice = FastllmCudaGetDevice();
    struct RestoreDevice { int previous; ~RestoreDevice() { FastllmCudaSetDevice(previous); } } restore{previousDevice};
    try {
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping || queue.size() >= 16 || (checkpoint &&
            (checkpointJobs >= 2 || job.stateBytes > HostLimit - queuedStateBytes))) { ++skipped; return; }
        for (int layer = 0; layer < model->block_cnt; ++layer) {
            if (Qwen35LayerIsLinearAttention(model,layer)) continue;
            for (int side = 0; side < 2; ++side) {
                auto &global = side ? ctx->pastKeyValues[layer].second : ctx->pastKeyValues[layer].first;
                for (auto ref : model->GetPagedKVCacheManagers(layer,side == 0)) {
                    Data *local = &global;
                    if (global.multiDeviceData) {
                        auto it = global.multiDeviceDatas.find(ref.first);
                        Check(it != global.multiDeviceDatas.end() && it->second,"ssd_missing_export_rank"); local = it->second;
                    }
                    Check(local->isPagedKVCache && local->pagedKVCacheData == ref.second &&
                        length % ref.second->pageLen == 0 && (int)local->pageIndex.size() >= length / ref.second->pageLen,
                        "ssd_export_pages");
                    auto lease = std::make_shared<Pages>(); lease->pool = ref.second;
                    lease->layer = layer; lease->side = side; lease->rank = ref.first; lease->begin = state->issued;
                    lease->pageBytes = (size_t)ref.second->pageLen * ref.second->dims[2] * ref.second->dims[3] * ref.second->unitSize;
                    std::vector<int> ids(local->pageIndex.begin() + state->issued / ref.second->pageLen,
                                         local->pageIndex.begin() + length / ref.second->pageLen);
                    need[ref.first] += ids.size() * lease->pageBytes;
                    size_t budget = gpuBudget[ref.first];
                    if (need[ref.first] + pinnedGpuBytes[ref.first] > budget) { ++skipped; return; }
                    ref.second->Pick(ids); lease->ids = std::move(ids);
                    FastllmCudaSetDevice(ref.first);
                    lease->ready = FastllmCudaEventCreate();
                    job.pages.push_back(std::move(lease));
                }
            }
        }
        // Record on the actual persistent TP producers, not an unrelated
        // scheduler-thread stream on the same device.
        auto recordRank = [&](int device) {
            FastllmCudaSetDevice(device);
            for (auto &lease : job.pages)
                if (lease->rank == device) FastllmCudaEventRecordCurrentThread(lease->ready);
        };
        if (model->threadTpPreparedDevices.size() > 1 && model->threadTpWorkerGroup.HasWorkers()) {
            std::vector<std::exception_ptr> errors(model->threadTpPreparedDevices.size());
            model->threadTpWorkerGroup.RunWithCaller(model->threadTpPreparedDevices,[&](int rank) {
                recordRank(model->threadTpPreparedDevices[rank]);
            },errors);
            for (auto &error : errors) if (error) std::rethrow_exception(error);
        } else for (auto n : need) recordRank(n.first);
        for (auto n : need) pinnedGpuBytes[n.first] += n.second;
        state->issued = length;
        queuedStateBytes += job.stateBytes; checkpointJobs += checkpoint;
        queue.push_back(std::move(job)); changed.notify_one();
    } catch (const std::exception &e) {
        ++skipped; std::fprintf(stderr,"[Prefix SSD] capture skipped: %s.\n",e.what());
    }
    FastllmCudaSetDevice(previousDevice);
}

void Qwen35PersistentCache::Export(Job &job, void *buffer, void *stream) {
    if (job.image) {
        auto writer = imageStore->BeginWrite(job.stateBytes + (1<<20));
        auto chunks = writer.StoreBytes(job.stateBytes,[&](size_t offset,char *out,size_t count) {
            memcpy(out,(char*)job.image->data()+offset,count);
        });
        writer.Commit(job.key,Json::object{{"kind","encoder"},{"length",0},
            {"count",(double)job.image->size()},{"chunks",chunks}});
        writes += writer.WrittenBytes(); return;
    }
    auto &state = *job.state;
    if (state.failed) return;
    if (!state.writer) state.writer = std::make_unique<DiskPrefixCache::WriteSession>(store->BeginWrite(state.reservation));
    state.inherited.reset(); // The writer's shared lease now protects all refs.
    auto &writer = *state.writer;
    const auto before = writer.WrittenBytes();
    std::map<int,std::vector<std::shared_ptr<Pages>>> ranks;
    for (auto &lease : job.pages) ranks[lease->rank].push_back(lease);
    for (auto &rank : ranks) {
        FastllmCudaSetDevice(rank.first);
        void *copyStream = FastllmCudaStreamCreate();
        for (auto &lease : rank.second) FastllmCudaStreamWaitEvent(copyStream,lease->ready);
        auto &first = *rank.second.front();
        const int pagesPerBundle = std::max(1,std::lcm(256,first.pool->pageLen)/first.pool->pageLen);
        try {
            for (size_t p = 0; p < first.ids.size(); p += pagesPerBundle) {
                const size_t count = std::min((size_t)pagesPerBundle,first.ids.size()-p);
                Json::array segments;
                size_t bytes = 0;
                for (auto &lease : rank.second) {
                    Check(lease->ids.size() == first.ids.size() && lease->begin == first.begin,"ssd_bundle_ranges");
                    size_t segmentBytes = count*lease->pageBytes;
                    Check(segmentBytes <= BufferBytes-bytes,"ssd_bundle_exceeds_buffer");
                    for (size_t i=0;i<count;++i)
                        Check(FastllmCudaCopyFromDeviceToPinnedHostAsync((char*)buffer+bytes+i*lease->pageBytes,
                            (char*)lease->pool->cudaData+(size_t)lease->ids[p+i]*lease->pageBytes,
                            lease->pageBytes,copyStream),"ssd_d2h");
                    segments.emplace_back(Json::object{{"layer",lease->layer},{"side",lease->side},{"rank",lease->rank},
                        {"begin",lease->begin+(int)p*lease->pool->pageLen},{"tokens",(int)count*lease->pool->pageLen},
                        {"page",lease->pool->pageLen},{"heads",lease->pool->dims[2]},{"dim",lease->pool->dims[3]},
                        {"dtype",(int)lease->pool->dataType},{"bytes",(double)segmentBytes},{"offset",(double)bytes}});
                    bytes += segmentBytes;
                }
                FastllmCudaStreamSynchronize(copyStream);
                // The immutable CPU bundle now owns the data; GPU pages may be
                // recycled before filesystem I/O completes.
                for (auto &lease : rank.second) {
                    std::vector<int> copied(lease->ids.begin()+p,lease->ids.begin()+p+count);
                    lease->pool->ReleasePageIndices(copied);
                    std::fill(lease->ids.begin()+p,lease->ids.begin()+p+count,-1);
                    std::lock_guard<std::mutex> lock(mutex);
                    pinnedGpuBytes[lease->rank] -= count*lease->pageBytes;
                }
                auto chunks = writer.StoreBytes(bytes,[&](size_t offset,char *out,size_t n){memcpy(out,(char*)buffer+offset,n);});
                state.kv.emplace_back(Json::object{{"bytes",(double)bytes},{"segments",segments},{"chunks",chunks}});
            }
        } catch (...) { FastllmCudaStreamDestroy(copyStream); throw; }
        FastllmCudaStreamDestroy(copyStream);
    }
    if (job.snapshot) {
        Json::array linear, draft;
        for (int layer = 0; layer < (int)job.snapshot->layers.size(); ++layer) {
            auto &entry = job.snapshot->layers[layer];
            if (entry.linear) linear.emplace_back(Json::object{{"layer",layer},
                {"first",SaveLinear(writer,entry.first)},{"second",SaveLinear(writer,entry.second)}});
        }
        for (auto &entry : job.snapshot->dflashKeyValues)
            draft.emplace_back(Json::object{{"key",SaveTensor(writer,entry.first)},{"value",SaveTensor(writer,entry.second)}});
        writer.Commit(job.key,Json::object{{"kind","prefix"},{"length",job.length},
            {"family_identity",family},{"kv_dtype",Dtype(model->kvCacheDataType)},
            {"provenance",state.provenance},{"source",state.source},{"prefix",job.prefix},{"layout",RuntimeLayout()},
            {"kv",state.kv},{"linear",linear},{"draft",draft},{"draft_valid",job.snapshot->dflashValid},
            {"draft_tokens",job.snapshot->dflashTokens}});
        ++committed;
        printf("[Prefix SSD] committed: tokens=%d key=%s.\n",job.length,job.key.c_str()); fflush(stdout);
    }
    writes += writer.WrittenBytes()-before;
}

void Qwen35PersistentCache::Work() {
    void *buffer = FastllmCudaHostMalloc(BufferBytes);
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock,[&] { return stopping || !queue.empty(); });
            if (queue.empty() && stopping) break;
            job = std::move(queue.front()); queue.pop_front(); working = true;
        }
        try { Check(buffer != nullptr,"ssd_pinned_allocation"); Export(job,buffer,nullptr); }
        catch (const std::exception &e) {
            if (job.state) job.state->failed = true;
            ++skipped;
            std::fprintf(stderr,"[Prefix SSD] write skipped: tokens=%d reason=%s.\n",job.length,e.what());
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto &lease : job.pages)
                pinnedGpuBytes[lease->rank] -= std::count_if(lease->ids.begin(),lease->ids.end(),[](int id){return id>=0;})*lease->pageBytes;
            queuedStateBytes -= job.stateBytes; checkpointJobs -= job.snapshot != nullptr;
            // Destroy leases before Wait() can report the queue drained.
            job = Job(); working = false;
        }
        changed.notify_all();
    }
    if (buffer) FastllmCudaHostFree(buffer);
}

void Qwen35PersistentCache::Prepare(ResponseContext *ctx) {
    if (!ctx || ctx->cacheLen || !ctx->CanUsePagedPrefixCache() || !Qwen35LinearPrefixCacheEnabled() ||
        ctx->intParams["ssd_lookup_done"] || policy == "never") return;
    if (!Ensure()) return;
    ctx->intParams["ssd_lookup_done"] = 1;
    try {
        int memoryPages = INT_MAX;
        for (int layer = 0; layer < model->block_cnt; ++layer)
            for (int side = 0; side < 2; ++side)
                for (auto ref : model->GetPagedKVCacheManagers(layer,side == 0)) {
                    std::vector<int> pages;
                    ref.second->Query(ctx->currentTokens,pages,ctx->PrefixCachePageKeys());
                    memoryPages = std::min(memoryPages,(int)pages.size());
                }
        int memoryLength = memoryPages == INT_MAX ? 0 : std::min(memoryPages * fastllm::GetPageLen(),
            ((int)ctx->currentTokens.size()-1) / fastllm::GetPageLen() * fastllm::GetPageLen());
        ctx->intParams["ssd_branch_boundary"] = memoryLength / model->GetChunkedPrefillSize() * model->GetChunkedPrefillSize();
        memoryLength = model->QueryPagedPrefixCacheExtra(ctx,memoryLength);
        auto candidates = store->ListCheckpoints("",family,(int)ctx->allTokens.size()-1);
        std::sort(candidates.begin(),candidates.end(),[&](auto &a,auto &b) {
            if (a.length != b.length) return a.length > b.length;
            if (a.kvDtype != b.kvDtype) return a.kvDtype == Dtype(model->kvCacheDataType);
            return a.provenance.empty() && !b.provenance.empty();
        });
        for (auto &candidate : candidates) {
            if (candidate.length <= memoryLength) break;
            bool same = candidate.kvDtype == Dtype(model->kvCacheDataType);
            bool convert = model->kvCacheDataType == DataType::FP8_E4M3 && candidate.kvDtype == "float16";
            if ((!same && !convert) || candidate.length <= 0 || candidate.length % model->GetChunkedPrefillSize()) continue;
            const std::string prefixKey = Prefix(ctx,candidate.length)+candidate.provenance;
            if (candidate.key != DiskPrefixCache::Digest(prefixKey+":draft") &&
                candidate.key != DiskPrefixCache::Digest(prefixKey+":plain")) continue;
            if (policy == "auto") {
                std::lock_guard<std::mutex> lock(mutex);
                double estimate = convert ? convertedReadMsPerToken : nativeReadMsPerToken;
                if (prefillMsPerToken <= 0 || estimate <= 0 || !std::isfinite(estimate) ||
                    estimate*1.15 >= prefillMsPerToken) { ++skipped; continue; }
            }
            if (Import(ctx,candidate)) break;
        }
    } catch (const std::exception &e) {
        ++skipped; std::fprintf(stderr,"[Prefix SSD] lookup miss: %s.\n",e.what());
    }
}

bool Qwen35PersistentCache::Import(ResponseContext *ctx, const DiskPrefixCache::Checkpoint &candidate) {
    const int oldDevice = FastllmCudaGetDevice();
    const int length = candidate.length;
    const bool convert = candidate.kvDtype == "float16" && model->kvCacheDataType == DataType::FP8_E4M3;
    std::map<std::tuple<int,int,int>,std::shared_ptr<Pages>> pools;
    const auto started = Clock::now();
    try {
        auto reader = store->BeginRead();
        auto state = reader.Load(candidate.key,candidate.identity);
        Check(state["prefix"].string_value() == Prefix(ctx,length) &&
            state["layout"].dump() == RuntimeLayout().dump(),"ssd_checkpoint_identity");
        if (model->RequiresDFlashPrefixSnapshot(ctx))
            Check(state["draft_valid"].bool_value() && state["draft_tokens"].int_value() == length,"ssd_missing_draft");
        Check(!model->RequiresMtpPrefixSnapshot(ctx),"ssd_mtp_not_supported");
        auto snapshot = std::make_shared<Qwen35LinearPrefixSnapshot>();
        snapshot->cachedLen = length; snapshot->tokens.assign(ctx->allTokens.begin(),ctx->allTokens.begin()+length);
        snapshot->layers.resize(model->block_cnt);
        size_t stateBytes = 0;
        for (auto &entry : state["linear"].array_items()) {
            int layer = entry["layer"].int_value();
            Check(layer >= 0 && layer < model->block_cnt && Qwen35LayerIsLinearAttention(model,layer) &&
                !snapshot->layers[layer].linear,"ssd_linear_layer");
            auto &out = snapshot->layers[layer]; out.linear = true;
            LoadLinear(reader,entry["first"],out.first); LoadLinear(reader,entry["second"],out.second);
            for (auto *v : {&out.first,&out.second}) {
                if (v->multiDevice) for (auto &item : v->locals) stateBytes += item.second.GetBytes();
                else stateBytes += v->single.GetBytes();
            }
            Check(stateBytes <= HostLimit,"ssd_state_budget");
        }
        for (int layer = 0; layer < model->block_cnt; ++layer)
            Check(snapshot->layers[layer].linear == Qwen35LayerIsLinearAttention(model,layer),"ssd_incomplete_linear");
        snapshot->dflashValid = state["draft_valid"].bool_value();
        snapshot->dflashTokens = state["draft_tokens"].int_value();
        Check(state["draft"].array_items().size() <= (size_t)model->dflashLayers,"ssd_draft_layers");
        snapshot->dflashKeyValues.resize(state["draft"].array_items().size());
        for (size_t i = 0; i < snapshot->dflashKeyValues.size(); ++i) {
            LoadTensor(reader,state["draft"][i]["key"],snapshot->dflashKeyValues[i].first);
            LoadTensor(reader,state["draft"][i]["value"],snapshot->dflashKeyValues[i].second);
        }
        if (snapshot->dflashValid) Check(Qwen35DFlashSnapshotMatches(*snapshot,length,
            model->dflashLayers,model->dflashKvHeads,model->dflashHeadDim),"ssd_draft_shape");
        for (int layer = 0; layer < model->block_cnt; ++layer) {
            if (Qwen35LayerIsLinearAttention(model,layer)) continue;
            for (int side = 0; side < 2; ++side) {
                auto refs = model->GetPagedKVCacheManagers(layer,side == 0);
                Check(!refs.empty(),"ssd_missing_pool");
                for (auto ref : refs) {
                    auto pool = ref.second;
                    int pages = length / pool->pageLen;
                    int suffixPages = ((int)ctx->allTokens.size() - length + 16 + pool->pageLen-1) / pool->pageLen;
                    { std::lock_guard<std::mutex> lock(pool->pageIndexLocker);
                      Check(pool->FreePageCount() >= pages + suffixPages,"ssd_gpu_capacity"); }
                    auto lease = std::make_shared<Pages>(); lease->pool = pool; lease->rank = ref.first;
                    lease->layer = layer; lease->side = side;
                    lease->pageBytes = (size_t)pool->pageLen*pool->dims[2]*pool->dims[3]*pool->unitSize;
                    for (int p = 0; p < pages; ++p) lease->ids.push_back(pool->GetUnusedPageIndex(true));
                    pools[{layer,side,ref.first}] = std::move(lease);
                }
            }
        }
        std::map<std::tuple<int,int,int>,int> coverage;
        for (auto &bundle : state["kv"].array_items()) {
            size_t bundleBytes = (size_t)bundle["bytes"].number_value();
            Check(bundleBytes > 0 && bundleBytes <= BufferBytes,"ssd_bundle_bytes");
            std::vector<char> bundleData(bundleBytes);
            reader.ReadBytes(bundle["chunks"],bundleBytes,[&](size_t offset,const char *p,size_t n){memcpy(bundleData.data()+offset,p,n);});
            size_t nextOffset = 0;
            for (auto &entry : bundle["segments"].array_items()) {
            auto key = std::make_tuple(entry["layer"].int_value(),entry["side"].int_value(),entry["rank"].int_value());
            auto found = pools.find(key); Check(found != pools.end(),"ssd_kv_rank");
            auto &lease = *found->second; auto pool = lease.pool;
            int begin = entry["begin"].int_value(), tokens = entry["tokens"].int_value();
            Check(begin == coverage[key] && tokens > 0 && tokens <= length-begin && tokens % pool->pageLen == 0 &&
                entry["page"].int_value() == pool->pageLen && entry["heads"].int_value() == pool->dims[2] &&
                entry["dim"].int_value() == pool->dims[3],"ssd_kv_layout");
            const auto sourceType = (DataType)entry["dtype"].int_value();
            Check(sourceType == pool->dataType || (convert && sourceType == DataType::FLOAT16),"ssd_kv_dtype");
            size_t sourceUnit = sourceType == DataType::FP8_E4M3 ? 1 : 2;
            size_t bytes = (size_t)tokens*pool->dims[2]*pool->dims[3]*sourceUnit;
            Check(bytes <= BufferBytes && entry["bytes"].number_value() == (double)bytes,"ssd_kv_bytes");
            size_t offset = (size_t)entry["offset"].number_value();
            Check(offset == nextOffset && bytes <= bundleBytes-offset,"ssd_bundle_segment");
            const char *payload = bundleData.data()+offset;
            nextOffset += bytes;
            FastllmCudaSetDevice(lease.rank);
            if (convert) {
                // Existing KV writer expects [heads,tokens,dim], whereas on
                // disk the exact pool representation is [tokens,heads,dim].
                Data source(DataType::FLOAT16,{pool->dims[2],tokens,pool->dims[3]}); source.Allocate();
                const size_t row = (size_t)pool->dims[3]*2;
                for (int t = 0; t < tokens; ++t) for (int h = 0; h < pool->dims[2]; ++h)
                    memcpy(source.cpuData+((size_t)h*tokens+t)*row,payload+((size_t)t*pool->dims[2]+h)*row,row);
                source.ToDevice(DataDevice::CUDA,{lease.rank},true);
                Check(FastllmCudaPagedCacheCopyMultiPage((uint8_t*)pool->cudaData,
                    lease.ids.data()+begin/pool->pageLen,tokens/pool->pageLen,0,pool->pageLen,
                    pool->dims[2],pool->dims[3],pool->dataType,(uint8_t*)source.cudaData,source.dataType,tokens),"ssd_kv_convert");
                FastllmCudaSyncCurrentThreadStream();
            } else {
                for (int p = 0; p < tokens/pool->pageLen; ++p)
                    FastllmCudaCopyFromHostToDevice((char*)pool->cudaData+(size_t)lease.ids[begin/pool->pageLen+p]*lease.pageBytes,
                        (void*)(payload+(size_t)p*lease.pageBytes),lease.pageBytes);
            }
            coverage[key] += tokens;
            }
            Check(nextOffset == bundleBytes,"ssd_bundle_coverage");
        }
        for (auto &entry : pools) Check(coverage[entry.first] == length,"ssd_incomplete_kv");
        std::string provenance = candidate.provenance;
        if (convert) provenance = DiskPrefixCache::Digest("fp16-to-fp8-v1:"+candidate.identity+candidate.key+provenance);
        if (!provenance.empty()) {
            std::lock_guard<std::mutex> lock(mutex);
            provenanceSources[provenance] = convert ? Json(Json::object{{"checkpoint",candidate.key},
                {"identity",candidate.identity},{"source_dtype",candidate.kvDtype},
                {"target_dtype",Dtype(model->kvCacheDataType)},{"conversion_version",1},
                {"parent_provenance",candidate.provenance}}) : state["source"];
        }
        SetProvenance(ctx,provenance);
        if (!ctx->prefixCachePageKeys.empty()) snapshot->mediaPageKeys.assign(ctx->prefixCachePageKeys.begin(),
            ctx->prefixCachePageKeys.begin()+length/fastllm::GetPageLen());
        // Publish only verified complete data. No active request tensor has
        // changed; both schedulers now use their existing restore protocol.
        { std::lock_guard<std::mutex> lock(Qwen35LinearPrefixSnapshotsMutex());
          snapshot->timestamp = ++Qwen35LinearPrefixSnapshotTimestamp();
          auto &items = Qwen35LinearPrefixSnapshots()[model]; items.push_back(snapshot);
          while ((int)items.size() > Qwen35LinearPrefixSnapshotMaxRecords()) items.erase(items.begin()); }
        for (auto &entry : pools) entry.second->pool->Record(snapshot->tokens,entry.second->ids,ctx->PrefixCachePageKeys());
        ctx->intParams["ssd_loaded_len"] = length;
        ctx->intParams["ssd_source_fp16"] = candidate.kvDtype == "float16";
        reads += reader.ReadBytesCount();
        const double elapsedMs = std::chrono::duration<double,std::milli>(Clock::now()-started).count();
        { std::lock_guard<std::mutex> lock(mutex);
          double &rate = convert ? convertedReadMsPerToken : nativeReadMsPerToken;
          rate = rate > 0 ? 0.8*rate + 0.2*elapsedMs/length : elapsedMs/length; }
        SaveCost();
        printf("[Prefix SSD] loaded: tokens=%d source_dtype=%s target_dtype=%s ms=%.3f.\n",length,
            candidate.kvDtype.c_str(),Dtype(model->kvCacheDataType).c_str(),
            std::chrono::duration<double,std::milli>(Clock::now()-started).count()); fflush(stdout);
        FastllmCudaSetDevice(oldDevice);
        return true;
    } catch (const std::exception &e) {
        SetProvenance(ctx,""); ++skipped;
        std::fprintf(stderr,"[Prefix SSD] restore miss: tokens=%d reason=%s.\n",length,e.what());
        FastllmCudaSetDevice(oldDevice);
        return false;
    }
}

#else
class Qwen35PersistentCache {
public:
    static std::shared_ptr<Qwen35PersistentCache> Create(Qwen3_5Model*) { return {}; }
    void Created(ResponseContext*) {}
    void Replanned(ResponseContext*) {}
    void Removed(ResponseContext*) {}
    void Stop() {}
    void ObservePrefill(int,double) {}
    bool Wait(int) { return true; }
    std::string Stats() const { return "{}"; }
    void Prepare(ResponseContext*) {}
    size_t LoadImage(const std::string&,size_t,std::vector<float>&) { return 0; }
    void SaveImage(const std::string&,const float*,size_t) {}
    void Restored(ResponseContext*) {}
    void Capture(ResponseContext*,const std::shared_ptr<Qwen35LinearPrefixSnapshot>&) {}
};
#endif

void Qwen3_5Model::PreparePersistentPrefixCache(ResponseContext *context) {
    if (persistentPrefixCache) persistentPrefixCache->Prepare(context);
}
void Qwen3_5Model::OnPersistentPrefixRestored(ResponseContext *context) {
    if (persistentPrefixCache) persistentPrefixCache->Restored(context);
}
bool Qwen3_5Model::WaitPersistentPrefixCache(int timeoutMs) {
    return !persistentPrefixCache || persistentPrefixCache->Wait(timeoutMs);
}
std::string Qwen3_5Model::PersistentPrefixCacheStatistics() const {
    return persistentPrefixCache ? persistentPrefixCache->Stats() : "{}";
}
void Qwen3_5Model::ObservePersistentPrefill(int tokens, double milliseconds) {
    if (persistentPrefixCache) persistentPrefixCache->ObservePrefill(tokens,milliseconds);
}
