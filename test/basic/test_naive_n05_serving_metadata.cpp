// Host-only descriptor regression. Fake device addresses are never submitted
// to CUDA; exercise the real validation, rebinding and request lifecycle code.
#include "naive_n05_tp.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <future>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

namespace fastllm {
struct NaiveN05PrefixCacheTestAccess : NaiveN05FlashModel {
    std::vector<std::unique_ptr<ResponseContext>> contexts;
    std::vector<std::shared_ptr<TPDecodeState::Storage>> storage;
    NaiveN05PrefixCacheTestAccess() {
        block_cnt = 48;
        maxBatch = 352;
        max_positions = 200000;
        window = 128;
        tpDevices = {0, 1, 2, 3};
        dataType = kvCacheDataType = moeAtype = BFLOAT16;
        full = {16, 4, 32, 16, 10000.f};
        sliding = {16, 8, 32, 16, 10000.f};
        slidingLayers.resize(block_cnt);
        for (int layer = 0; layer < block_cnt; ++layer)
            slidingLayers[layer] = layer % 2;
        SetSaveHistoryChat(false);
        SetCudaEmbedding(true);
        auto s = std::make_shared<TPServingState>();
        tpServingState = s;
        s->sealed = true;
        s->generation = FastllmGetNcclGeneration();
        s->batchBuckets = DecodeGraphBatchBuckets(maxBatch);
        for (int rank = 0; rank < 4; ++rank) {
            auto v = std::make_shared<TPDecodeState::Storage>();
            v->maxRows = s->batchBuckets.back();
            v->hostCache.resize((size_t)v->maxRows * block_cnt * 4);
            storage.push_back(v);
        }
        for (int rows : s->batchBuckets) {
            auto state = std::make_shared<TPDecodeState>();
            state->captured = true;
            state->capacity = 131072;
            state->rows = rows;
            for (auto &v : storage)
                state->ranks.emplace_back(new TPDecodeState::Rank(v));
            s->graphs[{rows, 131072}] = state;
        }
        for (int row = 0; row < maxBatch; ++row) {
            auto context = std::make_unique<ResponseContext>();
            context->Init(block_cnt, BFLOAT16, BFLOAT16);
            OnResponseContextCreated(context.get());
            for (int layer = 0; layer < block_cnt; ++layer)
                for (int part = 0; part < 2; ++part) {
                    auto &cfg = slidingLayers[layer] ? sliding : full;
                    auto &pair = context->pastKeyValues[layer];
                    Data &root = part ? pair.second : pair.first;
                    const int width = cfg.kvHeads * (part ? cfg.valueDim : cfg.headDim) +
                                      (!part && !slidingLayers[layer] ? PackedIndexerWidth() : 0);
                    const int length = slidingLayers[layer] ? 127 : 81920 + row % 97;
                    const int capacity = slidingLayers[layer] ? 128 : 200000;
                    root.Resize({1, length, width});
                    root.expansionDims = {1, capacity, width};
                    root.dataDevice = DataDevice::CUDA;
                    root.multiDeviceData = true;
                    root.dataDeviceIds = tpDevices;
                    for (int rank = 0; rank < 4; ++rank) {
                        auto *local = new Data(BFLOAT16);
                        const int localWidth =
                            std::max(1, cfg.kvHeads / 4) * (part ? cfg.valueDim : cfg.headDim) +
                            (!part ? LocalIndexerCacheWidth(layer, rank) : 0);
                        local->Resize({1, length, localWidth});
                        local->expansionDims = {1, capacity, localWidth};
                        local->strides = {(size_t)capacity * localWidth, (size_t)localWidth, 1};
                        local->dataDevice = DataDevice::CUDA;
                        local->dataDeviceIds = {rank};
                        local->isFake = true;
                        local->cudaData =
                            (void *)(uintptr_t(0x100000) +
                                     (((size_t)row * block_cnt + layer) * 8 + rank * 2 + part) * 256);
                        root.multiDeviceDatas[rank] = local;
                    }
                }
            contexts.push_back(std::move(context));
        }
    }
    ~NaiveN05PrefixCacheTestAccess() override {
        // These are CPU-only mock graph records, never CUDA graph handles.
        for (auto &entry : tpServingState->graphs)
            entry.second->ranks.clear();
        tpDecodeState.reset();
        tpServingState.reset();
        storage.clear();
    }
    struct Batch {
        std::vector<std::pair<Data *, Data *>> kv;
        std::vector<GenerationConfig> configs;
        std::vector<int> lengths;
        TargetBatch target;
        Data ids;
        Batch(NaiveN05PrefixCacheTestAccess &model, const std::vector<int> &order)
            : configs(order.size()), lengths(order.size(), 1), target{kv, configs, lengths, model.block_cnt},
              ids(FLOAT32, {1, (int)order.size()}) {
            for (int row : order)
                for (auto &pair : model.contexts[row]->pastKeyValues)
                    kv.emplace_back(&pair.first, &pair.second);
        }
    };
    void Prepare(Batch &batch, bool force) {
        if (force)
            ++tpCacheEpoch;
        ReorderTensorParallelServingCache(&batch.target);
        std::vector<int *> lengths;
        LogitsSelection selection;
        Check(PrepareTensorParallelServingDecode(batch.ids, selection, lengths, &batch.target, false),
              "graph preparation failed");
        const int rows = batch.target.Size();
        for (int rank = 0; rank < 4; ++rank)
            for (int layer = 0; layer < block_cnt; ++layer) {
                auto &v = *storage[rank];
                for (int row = 0; row < v.maxRows; ++row)
                    for (int part = 0; part < 2; ++part) {
                        void *wanted = nullptr;
                        if (row < rows) {
                            Data &root = part ? batch.target.Value(row, layer) : batch.target.Key(row, layer);
                            wanted = root.multiDeviceDatas.at(rank)->cudaData;
                        }
                        Check(v.hostCache[((size_t)layer * v.maxRows + row) * 4 + part] == wanted,
                              "stale/reordered graph pointer");
                    }
            }
        // Simulate a successful graph's metadata commit, preserving SWA rings.
        for (int row = 0; row < rows; ++row) {
            const int next = batch.target.Key(row, 0).dims[1] + 1;
            for (int layer = 0; layer < block_cnt; ++layer)
                for (int part = 0; part < 2; ++part) {
                    const int length = slidingLayers[layer] ? 127 : next;
                    size_t slot = ((size_t)row * block_cnt + layer) * 8 + part;
                    for (int rank = 0; rank < 4; ++rank)
                        *lengths[slot + rank * 2] = length;
                    *lengths[(size_t)rows * block_cnt * 8 + ((size_t)row * block_cnt + layer) * 2 + part] =
                        length;
                }
        }
    }
    void Run() {
        std::vector<int> order(maxBatch);
        std::iota(order.begin(), order.end(), 0);
        // Model a live decode batch interrupted by prefill on twelve other
        // requests, including rows inserted ahead of surviving graph bindings.
        std::vector<int> initial(order.begin() + 12, order.end());
        Batch beforePrefill(*this, initial);
        Prepare(beforePrefill, false);
        for (int i = 0; i < 12; ++i)
            InvalidateTensorParallelCache(&contexts[i]->pastKeyValues.front().first);
        Batch first(*this, order);
        ReorderTensorParallelServingCache(&first.target);
        auto retainedRows = ReusableTensorParallelServingRows(&first.target);
        Check(std::count(retainedRows.begin(), retainedRows.end(), 1) == maxBatch - 12,
              "new prefill invalidated surviving rows");
        Prepare(first, false);
        // An owned layout mutation revalidates only that request and binds its
        // actual replacement addresses, even if the row order did not change.
        InvalidateTensorParallelCache(&first.target.Key(7, 0));
        for (int layer = 0; layer < block_cnt; ++layer)
            for (Data *root : {&first.target.Key(7, layer), &first.target.Value(7, layer)})
                for (auto &entry : root->multiDeviceDatas)
                    entry.second->cudaData = (void *)(uintptr_t(entry.second->cudaData) + 16);
        ReorderTensorParallelServingCache(&first.target);
        retainedRows = ReusableTensorParallelServingRows(&first.target);
        Check(std::count(retainedRows.begin(), retainedRows.end(), 1) == maxBatch - 1,
              "owned mutation invalidated unrelated rows");
        // A failed dirty-row validation must not publish stale descriptors.
        auto *local = first.target.Key(7, 0).multiDeviceDatas.at(0);
        void *address = local->cudaData;
        local->cudaData = nullptr;
        std::vector<int *> rejectedLengths;
        LogitsSelection selection;
        Check(!PrepareTensorParallelServingDecode(first.ids, selection, rejectedLengths, &first.target, false),
              "invalid new row accepted");
        Check(rejectedLengths.empty(), "failed validation published cache lengths");
        local->cudaData = address;
        Prepare(first, false);
        InvalidateTensorParallelCache();
        retainedRows = ReusableTensorParallelServingRows(&first.target);
        Check(std::count(retainedRows.begin(), retainedRows.end(), 1) == 0,
              "unknown layout mutation did not invalidate all rows");
        Prepare(first, false);
        Check(CanReuseTensorParallelServingCache(&first.target), "stable batch not reusable");
        // A retired row is physically destroyed; surviving cache lengths must
        // never dereference it, even when its row is in the middle of the batch.
        OnResponseContextRemoved(contexts[173].get());
        contexts[173].reset();
        order.erase(order.begin() + 173);
        Batch smaller(*this, order);
        ReorderTensorParallelServingCache(&smaller.target);
        Check(CanReuseTensorParallelServingCache(&smaller.target), "surviving requests invalidated");
        Prepare(smaller, false);
        std::reverse(order.begin(), order.end());
        Batch reversed(*this, order);
        Prepare(reversed, false);
        Check(CanReuseTensorParallelServingCache(&reversed.target), "reordered plan not reusable");
        auto *ctx = contexts[order[7]].get();
        OnResponseContextRemoved(ctx);
        OnResponseContextCreated(ctx);
        Check(!CanReuseTensorParallelServingCache(&reversed.target), "ABA identity revived an old plan");
        Prepare(reversed, false);
        ++tpCacheEpoch;
        Check(!CanReuseTensorParallelServingCache(&reversed.target), "layout epoch ignored");
        Prepare(reversed, false);
        const int row = 5;
        auto &root = reversed.target.Key(row, 0);
        root.dims[1]++;
        Check(!CanReuseTensorParallelServingCache(&reversed.target), "unexpected length accepted");
        root.dims[1]--;
        const int limit = tpServingState->capacityLimits[row];
        tpServingState->capacityLimits[row] = root.dims[1];
        Check(!CanReuseTensorParallelServingCache(&reversed.target), "capacity boundary ignored");
        Check(CanReuseTensorParallelServingCache(&reversed.target, false),
              "growth preparation must retain descriptors");
        tpServingState->capacityLimits[row] = limit;
        std::thread lifecycle([&] {
            for (int i = 0; i < 1000; ++i) {
                ResponseContext unrelated;
                unrelated.Init(block_cnt, BFLOAT16, BFLOAT16);
                OnResponseContextCreated(&unrelated);
                OnResponseContextRemoved(&unrelated);
            }
        });
        bool retained = true;
        for (int i = 0; i < 2000; ++i)
            retained &= CanReuseTensorParallelServingCache(&reversed.target);
        lifecycle.join();
        Check(retained, "unrelated request lifecycle invalidated survivors");
        // Check both full validation and incremental rebinding for permutations.
        for (bool force : {true, false}) {
            for (int step = 0; step < 8; ++step) {
                std::rotate(order.begin(), order.begin() + 1, order.end());
                Batch batch(*this, order);
                Prepare(batch, force);
            }
        }
        // Exercise in-place compaction repeatedly, including the shifted root
        // length suffix, with the removed Data objects already freed.
        for (int i = 0; i < 40; ++i) {
            size_t slot = (size_t)(i * 37) % order.size();
            int removed = order[slot];
            OnResponseContextRemoved(contexts[removed].get());
            contexts[removed].reset();
            order.erase(order.begin() + slot);
            Batch b(*this, order);
            Prepare(b, false);
            Check(CanReuseTensorParallelServingCache(&b.target), "compacted plan not reusable");
        }
    }
};
} // namespace fastllm
int main() {
    try {
        SetDeviceMap({{"cpu", 1}});
        SetThreads(1);
        NaiveN05PrefixCacheTestAccess model;
        std::promise<void> ready;
        auto readyFuture = ready.get_future();
        std::exception_ptr error;
        model.mainLoop = new std::thread([&] {
            readyFuture.wait();
            try {
                model.Run();
            } catch (...) {
                error = std::current_exception();
            }
        });
        ready.set_value();
        model.mainLoop->join();
        delete model.mainLoop;
        model.mainLoop = nullptr;
        if (error)
            std::rethrow_exception(error);
        std::cout << "SERVING METADATA PASS\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
