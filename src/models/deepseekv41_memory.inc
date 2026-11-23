// Fixed contiguous request slots preserve the existing attention kernels. The
// remaining serving allocations use a bounded, coalescing backing arena under
// the ordinary CUDA cache; graph pins and stream ordering remain unchanged.
#ifdef USE_CUDA
namespace {
struct V41RequestStorage : std::enable_shared_from_this<V41RequestStorage> {
    struct Entry { int layer, part, rows, width; DataType type; size_t offset; };
    int device;
    size_t slotBytes = 0;
    Data storage;
    std::vector<Entry> entries;
    std::vector<bool> used;
    std::mutex mutex;
    struct Lease {
        std::shared_ptr<V41RequestStorage> owner;
        int slot;
        Lease(std::shared_ptr<V41RequestStorage> owner, int slot) : owner(std::move(owner)), slot(slot) {}
        ~Lease() {
            FastllmCudaSyncDevice(owner->device);
            std::lock_guard<std::mutex> guard(owner->mutex);
            owner->used[slot] = false;
        }
    };
    explicit V41RequestStorage(int device, int slots) : device(device), used(slots, false) {}
    void Add(int layer, int part, int rows, int width, DataType type) {
        entries.push_back({layer, part, rows, width, type, slotBytes});
        const size_t bytes = (size_t)rows * width * (type == BFLOAT16 ? 2 : type == FLOAT32 ? 4 : 1);
        slotBytes += (bytes + 255) / 256 * 256;
    }
    void Allocate() {
        AssertInFastLLM(slotBytes > 0 && slotBytes <= INT_MAX, "DeepSeekV41 KV slot is too large");
        storage.dataType = INT8; storage.UpdateUnitSize();
        storage.dataDevice = CUDA; storage.dataDeviceIds = {device};
        storage.directMemory = true;
        storage.Resize({(int)used.size(), (int)slotBytes});
        storage.Allocate(false);
    }
    bool HasFreeSlot() {
        std::lock_guard<std::mutex> guard(mutex);
        return std::find(used.begin(), used.end(), false) != used.end();
    }
    std::shared_ptr<void> Reserve() {
        std::lock_guard<std::mutex> guard(mutex);
        int slot = std::find(used.begin(), used.end(), false) - used.begin();
        if (slot == (int)used.size()) return {};
        auto lease = std::make_shared<Lease>(shared_from_this(), slot);
        used[slot] = true;
        return lease;
    }
    void Bind(const std::shared_ptr<DeepSeekV41RequestState> &state, std::shared_ptr<void> reservation) {
        if (state->storageLease) return;
        if (!reservation) reservation = Reserve();
        if (!reservation) throw std::runtime_error("DeepSeekV41 request storage was not admitted");
        auto lease = std::static_pointer_cast<Lease>(reservation);
        if (lease->owner.get() != this) throw std::logic_error("DeepSeekV41 request storage owner mismatch");
        const int slot = lease->slot;
        ++state->storageRevision;
        state->storageLease = std::move(reservation);
        for (const auto &entry : entries) {
            auto &layer = state->layers.at(entry.layer);
            Data *parts[] = {&layer.windowKV, &layer.compressedKV, &layer.indexK,
                             &layer.rawTailKV, &layer.rawTailScore};
            Data &data = *parts[entry.part];
            AssertInFastLLM(!data.cudaData && !data.cpuData, "DeepSeekV41 KV slot must bind before restore");
            data.dataType = entry.type; data.UpdateUnitSize();
            data.dataDevice = CUDA; data.dataDeviceIds = {device};
            data.cudaData = (uint8_t *)storage.cudaData + slot * slotBytes + entry.offset;
            data.isFake = true;
            data.SetKVCache();
            // Expansion strides describe physical capacity even when the
            // logical sequence is empty. CatDirect uses these for row copies.
            data.Resize({1, entry.rows, entry.width});
            data.expansionDims = {1, entry.rows, entry.width};
            data.expansionSize = (size_t)entry.rows * entry.width;
            data.expansionBytes = data.expansionSize * data.unitSize;
            if (entry.part != 0) data.Resize({1, 0, entry.width});
        }
    }
};
}
#endif

bool DeepSeekV41Model::HasFreeRequestStorage() const {
#ifdef USE_CUDA
    return v41RequestStorage && std::static_pointer_cast<V41RequestStorage>(v41RequestStorage)->HasFreeSlot();
#else
    return false;
#endif
}

std::shared_ptr<void> DeepSeekV41Model::ReserveRequestStorage() {
#ifdef USE_CUDA
    if (v41RequestStorage) return std::static_pointer_cast<V41RequestStorage>(v41RequestStorage)->Reserve();
#endif
    return {};
}

void DeepSeekV41Model::PrepareRequestStorage(const std::shared_ptr<DeepSeekV41RequestState> &state,
                                           std::shared_ptr<void> reservation) {
#ifdef USE_CUDA
    if (v41RequestStorage) std::static_pointer_cast<V41RequestStorage>(v41RequestStorage)->Bind(state, std::move(reservation));
#endif
}

bool DeepSeekV41Model::TryAdmitResponseContext(ResponseContext *context) {
#ifdef USE_CUDA
    if (!v41ServingReservations.empty()) {
        // Cancellation can retire the last live request without another
        // finished context or forward to drain its deferred storage.
        ReleaseRetiredStates();
        auto state = GetOrCreateState(context->pastKeyValues, false);
        return v41AttentionParallel->Place(state, (maxBatch + v41AttentionDP - 1) / v41AttentionDP, true) >= 0;
    }
#endif
    return true;
}

void DeepSeekV41Model::PrepareServingMemory() {
#ifdef USE_CUDA
    // Other execution modes retain their existing allocator. Start with the
    // independently owned DP attention caches; TP and speculative rollback can
    // adopt the same storage contract without changing the general allocator.
    if (executionMode != ExecutionMode::Gpu || !v41AttentionParallel ||
        v41DsparkEnabled || v41AttentionParallel->runtime.Layout().attentionTP != 1 ||
        !v41ServingReservations.empty() || maxBatch <= 0 || max_positions <= 0) return;
    auto &parallel = *v41AttentionParallel;
    const auto &layout = parallel.runtime.Layout();
    const int slots = (maxBatch + layout.attentionDP - 1) / layout.attentionDP;
    const bool packed = kvCacheDataType == FP8_E4M3 || kvCacheDataType == FP4_E2M1;
    const bool fp4 = kvCacheDataType == FP4_E2M1;
    const int windowWidth = head_dim_full + head_dim_full / 32;
    const int compressedWidth = fp4 ? head_dim_full / 2 + head_dim_full / 16 : windowWidth;
    const int indexWidth = (fp4 ? index_head_dim / 2 : index_head_dim) + index_head_dim / 32;
    std::vector<std::shared_ptr<V41RequestStorage>> pools;
    for (int device : layout.devices) {
        auto pool = std::make_shared<V41RequestStorage>(device, slots);
        for (int layer = 0; layer < block_cnt; ++layer) {
            pool->Add(layer, 0, window_size, packed ? windowWidth : head_dim_full, packed ? INT8 : BFLOAT16);
            if (!isKvSource[layer]) continue;
            const int ratio = std::max(1, compress_ratios[layer]);
            const int rows = (max_positions + ratio - 1) / ratio;
            pool->Add(layer, 1, rows, packed ? compressedWidth : head_dim_full, packed ? INT8 : BFLOAT16);
            if (isIndexSource[layer])
                pool->Add(layer, 2, rows, packed ? indexWidth : index_head_dim, packed ? INT8 : BFLOAT16);
            if (ratio > 1) {
                pool->Add(layer, 3, ratio, head_dim_full, FLOAT32);
                pool->Add(layer, 4, ratio, head_dim_full, FLOAT32);
            }
        }
        pools.push_back(std::move(pool));
    }
    for (int device : layout.devices) FastllmCudaSyncDevice(device);
    FastllmCudaClearBigBufferAll();
    auto free = FastllmCudaGetFreeSizes(), total = FastllmCudaGetTotalSizes();
    const double ratio = GetGpuMemRatio();
    AssertInFastLLM(ratio > 0 && ratio <= 1, "Invalid DeepSeekV41 GPU memory ratio");
    for (size_t rank = 0; rank < pools.size(); ++rank) {
        const int device = layout.devices[rank];
        const int64_t available = free.at(device) - (int64_t)(total.at(device) * (1 - ratio));
        const int64_t kv = pools[rank]->slotBytes * slots;
        AssertInFastLLM(available > kv + (1LL << 30),
            "DeepSeekV41 fixed KV/workspace budget does not fit; reduce max_batch or max_context_length");
    }
    const int original = FastllmCudaGetDevice();
    for (size_t rank = 0; rank < pools.size(); ++rank) {
        const int device = layout.devices[rank];
        FastllmCudaSetDevice(device);
        pools[rank]->Allocate();
        parallel.models[rank]->v41RequestStorage = pools[rank];
        const int64_t kv = pools[rank]->slotBytes * slots;
        const int64_t workspace = free.at(device) - (int64_t)(total.at(device) * (1 - ratio)) - kv;
        v41ServingReservations.push_back(std::make_shared<CudaMemoryReservation>(device, workspace));
        printf("[DeepSeekV41 memory] GPU %d: %d fixed KV slots x %d tokens, KV %.2f MiB, bounded workspace %.2f MiB.\n",
               device, slots, max_positions, kv / 1048576.0, workspace / 1048576.0);
    }
    FastllmCudaSetDevice(original);
#endif
}
