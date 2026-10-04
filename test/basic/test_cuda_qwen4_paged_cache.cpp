#include "models/qwen4_paged_cache.h"
#include "models/qwen4_exp.h"
#include "executor.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace fastllm;
namespace {
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
std::unique_ptr<PagedCacheManager> Pool(DataType type, int device, int pages,
                                        int pageLen, int heads, int dim) {
    auto p = std::make_unique<PagedCacheManager>();
    p->type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
    p->dataType = type;
    p->UpdateUnitSize();
    p->dataDevice = DataDevice::CUDA;
    p->dataDeviceIds = {device};
    p->directMemory = true;
    p->pageLen = pageLen;
    p->Resize({pages, pageLen, heads, dim});
    p->Allocate();
    p->SetMaxPages(pages);
    return p;
}
void Tensor(Data &out, DataType type, const std::vector<int> &dims,
            const std::vector<float> &values, int device) {
    Data raw(FLOAT32, dims, values);
    ToDataType(raw, out, type);
    out.ToDevice(DataDevice::CUDA, std::vector<int>{device});
}
void Allocate(Data &out, DataType type, const std::vector<int> &dims, int device) {
    out.dataType = type;
    out.UpdateUnitSize();
    out.Resize(dims);
    out.ToDevice(DataDevice::CUDA, {device}, false);
    out.Allocate();
}
void Indices(Data &out, const std::vector<int> &values, int rows, int device) {
    out.dataType = INT32;
    out.UpdateUnitSize();
    out.Resize({rows, (int)values.size() / rows});
    out.Allocate();
    std::memcpy(out.cpuData, values.data(), values.size() * sizeof(int));
    out.ToDevice(DataDevice::CUDA, std::vector<int>{device});
}
void Equal(Data &a, Data &b, const char *message) {
    Data x, y;
    ToDataType(a, x, FLOAT32);
    ToDataType(b, y, FLOAT32);
    x.ToDevice(DataDevice::CPU); y.ToDevice(DataDevice::CPU);
    Check(x.dims == y.dims && x.GetBytes() == y.GetBytes(), "comparison shape mismatch");
    Check(std::memcmp(x.cpuData, y.cpuData, x.GetBytes()) == 0, message);
}
void Input(Data &k, Data &v, DataType type, int heads, int count, int offset, int device) {
    std::vector<float> x(heads * count * 128), y(x.size());
    for (int h = 0; h < heads; ++h) for (int t = 0; t < count; ++t) for (int d = 0; d < 128; ++d) {
        int i = (h * count + t) * 128 + d;
        x[i] = ((h * 19 + (offset + t) * 7 + d) % 127 - 63) / 64.0f;
        y[i] = ((h * 13 + (offset + t) * 3 + d) % 113 - 56) / 64.0f;
    }
    Tensor(k, type, {heads, count, 128}, x, device);
    Tensor(v, type, {heads, count, 128}, y, device);
}
void Gather(Data &k, Data &v, Data &outK, Data &outV, int device) {
    std::vector<int> ids(k.dims[1]); std::iota(ids.begin(), ids.end(), 0);
    Data indices; Indices(indices, ids, 1, device);
    Allocate(outK, k.dataType, k.dims, device);
    Allocate(outV, v.dataType, v.dims, device);
    Check(FastllmCudaQwen4GatherKV(k, v, indices, outK, outV), "paged gather failed");
}
void Verify(Data &k, Data &v, int device) {
    Data actualK, actualV, expectedK, expectedV;
    Gather(k, v, actualK, actualV, device);
    Input(expectedK, expectedV, k.dataType, k.dims[0], k.dims[1], 0, device);
    Equal(actualK, expectedK, "K token address mismatch");
    Equal(actualV, expectedV, "V token address mismatch");
}
void Append(Data &k, Data &v, int count, int device) {
    const int previous = k.dims[1];
    Qwen4ReservePagedAppend(k, previous + count);
    Qwen4ReservePagedAppend(v, previous + count);
    Data x, y; Input(x, y, k.dataType, k.dims[0], count, previous, device);
    Check(FastllmCudaQwen4KVAppend(x, y, previous, k, v), "paged append failed");
    Qwen4ResizeKVCache(k, previous + count); Qwen4ResizeKVCache(v, previous + count);
}
void TestCache(int device, DataType type, int heads) {
    auto kp = Pool(type, device, 16, 128, heads, 128);
    auto vp = Pool(type, device, 16, 128, heads, 128);
    // K and V deliberately use different, nonconsecutive physical pages.
    std::vector<int> heldK, heldV;
    for (int i = 0; i < 5; ++i) heldK.push_back(kp->GetUnusedPageIndex(true));
    for (int i = 0; i < 3; ++i) heldV.push_back(vp->GetUnusedPageIndex(true));
    kp->ReleasePageIndex(heldK.back()); heldK.pop_back();
    Data k, v, snapshotK, snapshotV;
    Qwen4AttachPagedCache(*kp, k); Qwen4AttachPagedCache(*vp, v);
    void *table = k.cudaData, *payload = kp->cudaData;
    Append(k, v, 127, device);
    Qwen4SharePagedCache(k, snapshotK); Qwen4SharePagedCache(v, snapshotV);
    Append(k, v, 132, device);
    Check(k.pageIndex[0] != snapshotK.pageIndex[0], "shared partial page was overwritten");
    Check(k.pageIndex != v.pageIndex, "fragmented K/V fixture failed");
    Check(k.cudaData == table && kp->cudaData == payload, "cache reallocated during append");
    Verify(k, v, device); Verify(snapshotK, snapshotV, device);
    Qwen4ResizeKVCache(k, 129); Qwen4ResizeKVCache(v, 129);
    Check(k.pageIndex.size() == 2 && k.lastPageLen == 1, "rollback did not return pages");
    Append(k, v, 128, device); Verify(k, v, device);

    Data query, ids, packed, compactK, compactV, mask;
    Tensor(query, type, {heads * 2, 3, 128}, std::vector<float>(heads * 2 * 3 * 128, 0.5f), device);
    Indices(ids, {0, 127, 128, -1, 256, 12, -1, -1, 129, 255, 1, -1}, 3, device);
    Allocate(packed, type, {3 * heads * 2, 1, 128}, device);
    Allocate(compactK, type, {3 * heads, 4, 128}, device);
    Allocate(compactV, type, {3 * heads, 4, 128}, device);
    Allocate(mask, type, {3, 1, 4}, device);
    Check(FastllmCudaQwen4PrepareSparseBatch(query, k, v, ids, packed, compactK, compactV, mask, 0, 3),
          "paged prefill gather failed");
    Data denseK, denseV, expectedPacked, expectedK, expectedV, expectedMask;
    Gather(k, v, denseK, denseV, device);
    Allocate(expectedPacked, type, packed.dims, device);
    Allocate(expectedK, type, compactK.dims, device);
    Allocate(expectedV, type, compactV.dims, device);
    Allocate(expectedMask, type, mask.dims, device);
    Check(FastllmCudaQwen4PrepareSparseBatch(query, denseK, denseV, ids, expectedPacked,
          expectedK, expectedV, expectedMask, 0, 3), "reference gather failed");
    Equal(compactK, expectedK, "prefill K mismatch"); Equal(compactV, expectedV, "prefill V mismatch");
    Equal(mask, expectedMask, "prefill causal/padding mask mismatch");
    kp->ReleasePageIndices(heldK); vp->ReleasePageIndices(heldV);
    std::cout << "CACHE_PASS device=" << device << " type=" << type << " heads=" << heads << '\n';
}
void TestAttention(int device) {
    auto kp = Pool(FLOAT16, device, 8, 128, 2, 128), vp = Pool(FLOAT16, device, 8, 128, 2, 128);
    Data k, v; Qwen4AttachPagedCache(*kp, k); Qwen4AttachPagedCache(*vp, v);
    Append(k, v, 259, device);
    for (int sequence : {1, 7, 129}) {
        Data query, out, expected, denseK, denseV, qs, ps, pages, last, mask;
        std::vector<float> values(4 * sequence * 128);
        for (int i = 0; i < (int)values.size(); ++i) values[i] = (i % 97 - 48) / 128.0f;
        Tensor(query, FLOAT16, {4, sequence, 128}, values, device);
        GeneratePagedBatchParams(query, {&k}, 1, qs, ps, pages, last, {sequence});
        AttentionPagedBatch(query, k, v, qs, ps, pages, last, out, 2, 1.0f / std::sqrt(128.0f), 1);
        Check(out.dims == std::vector<int>({sequence, 4, 128}), "unexpected paged attention layout");
        PermuteSelf(out, {1, 0, 2});
        Gather(k, v, denseK, denseV, device);
        Attention(query, denseK, denseV, mask, expected, 2, 1.0f / std::sqrt(128.0f), 1);
        Data a, b; ToDataType(out, a, FLOAT32); ToDataType(expected, b, FLOAT32);
        a.ToDevice(DataDevice::CPU); b.ToDevice(DataDevice::CPU);
        Check(a.dims == b.dims, "attention shape mismatch");
        float error = 0;
        for (size_t i = 0; i < a.Count(0); ++i) error = std::max(error, std::abs(((float*)a.cpuData)[i] - ((float*)b.cpuData)[i]));
        Check(error < 0.005f, "paged attention differs from dense causal attention");
        std::cout << "ATTENTION_PASS device=" << device << " tokens=" << sequence << " max_error=" << error << '\n';
    }
}
void TestFused(int device, DataType type) {
    constexpr int heads = 2, qHeads = 4, sequence = 7, previous = 127;
    auto kp = Pool(type, device, 8, 128, heads, 128), vp = Pool(type, device, 8, 128, heads, 128);
    Data k, v, denseK(type), denseV(type);
    Qwen4AttachPagedCache(*kp, k); Qwen4AttachPagedCache(*vp, v);
    Append(k, v, previous, device);
    for (Data *tensor : {&denseK, &denseV}) {
        tensor->ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
        tensor->Expansion({heads, 1024, 128});
        tensor->Resize({heads, previous, 128});
    }
    Data initialK, initialV; Input(initialK, initialV, type, heads, previous, 0, device);
    Check(FastllmCudaQwen4KVAppend(initialK, initialV, 0, denseK, denseV), "dense fixture append failed");
    Data qg, rawK, rawV, norm, positions, query, gate, expectedQuery, expectedGate;
    std::vector<float> qvalues(sequence * qHeads * 256), values(sequence * heads * 128);
    for (int i = 0; i < (int)qvalues.size(); ++i) qvalues[i] = (i % 67 - 33) / 64.0f;
    for (int i = 0; i < (int)values.size(); ++i) values[i] = (i % 83 - 41) / 64.0f;
    Tensor(qg, type, {1, sequence, qHeads * 256}, qvalues, device);
    Tensor(rawK, type, {1, sequence, heads * 128}, values, device);
    std::reverse(values.begin(), values.end());
    Tensor(rawV, type, {1, sequence, heads * 128}, values, device);
    Tensor(norm, FLOAT32, {128}, std::vector<float>(128, 1.0f), device);
    std::vector<float> pos(sequence); std::iota(pos.begin(), pos.end(), previous);
    Tensor(positions, FLOAT32, {1, sequence}, pos, device);
    Qwen4ReservePagedAppend(k, previous + sequence); Qwen4ReservePagedAppend(v, previous + sequence);
    Check(FastllmCudaQwen4AttentionPrepare(qg, rawK, rawV, norm, norm, positions,
          query, gate, k, v, 128, 64, 0, 0, 1e-6f, 10000.0f, previous), "paged fused prepare failed");
    Check(FastllmCudaQwen4AttentionPrepare(qg, rawK, rawV, norm, norm, positions,
          expectedQuery, expectedGate, denseK, denseV, 128, 64, 0, 0, 1e-6f, 10000.0f, previous),
          "dense fused prepare failed");
    for (Data *tensor : {&k, &v, &denseK, &denseV}) Qwen4ResizeKVCache(*tensor, previous + sequence);
    Equal(query, expectedQuery, "paged fused Q mismatch"); Equal(gate, expectedGate, "paged fused gate mismatch");
    Data a, b, c, d; Gather(k, v, a, b, device); Gather(denseK, denseV, c, d, device);
    Equal(a, c, "paged fused K mismatch"); Equal(b, d, "paged fused V mismatch");
    std::cout << "FUSED_PASS device=" << device << " type=" << type << '\n';
}
void TestGraph(int device) {
    auto kp = Pool(FLOAT16, device, 8, 128, 2, 128), vp = Pool(FLOAT16, device, 8, 128, 2, 128);
    Data k, v; Qwen4AttachPagedCache(*kp, k); Qwen4AttachPagedCache(*vp, v);
    Append(k, v, 127, device);
    Data inK, inV, meta, indices, outK, outV;
    Input(inK, inV, FLOAT16, 2, 1, 127, device);
    Indices(meta, {127}, 1, device); Indices(indices, {0, 126, 127}, 1, device);
    Allocate(outK, FLOAT16, {2, 3, 128}, device); Allocate(outV, FLOAT16, outK.dims, device);
    Qwen4ReservePagedAppend(k, 128); Qwen4ReservePagedAppend(v, 128);
    Check(FastllmCudaGraphPrepareCaptureDevice(), "graph prepare failed");
    Check(FastllmCudaGraphBeginCapture(), "graph begin failed");
    Check(FastllmCudaQwen4KVAppendGraph(inK, inV, (int32_t*)meta.cudaData, k, v), "graph append failed");
    Check(FastllmCudaQwen4GatherKVGraph(k, v, indices, (int32_t*)meta.cudaData, outK, outV), "graph gather failed");
    void *graph = nullptr, *exec = nullptr;
    Check(FastllmCudaGraphEndCapture(&graph), "graph end failed");
    Check(FastllmCudaGraphInstantiate(graph, &exec), "graph instantiate failed");
    void *table = k.cudaData;
    for (int previous : {127, 128, 129}) {
        Qwen4ReservePagedAppend(k, previous + 1); Qwen4ReservePagedAppend(v, previous + 1);
        Data nextK, nextV; Input(nextK, nextV, FLOAT16, 2, 1, previous, device);
        FastllmCudaCopyFromDeviceToDevice(inK.cudaData, nextK.cudaData, nextK.GetBytes());
        FastllmCudaCopyFromDeviceToDevice(inV.cudaData, nextV.cudaData, nextV.GetBytes());
        FastllmCudaCopyFromHostToDevice(meta.cudaData, &previous, sizeof(previous));
        Check(FastllmCudaGraphLaunch(exec), "graph replay failed");
        FastllmCudaSyncCurrentThreadStream();
        Qwen4ResizeKVCache(k, previous + 1); Qwen4ResizeKVCache(v, previous + 1);
        Verify(k, v, device);
        Check(k.cudaData == table, "graph page table address changed");
    }
    FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
    std::cout << "GRAPH_PASS device=" << device << '\n';
}
void TestSlots(int device) {
    auto pool = Pool(FLOAT32, device, 2, 1, 2, 128);
    pool->type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_MLP_CACHE;
    Data a, b, restored;
    const std::vector<int> dims{1, 2, 128};
    Qwen4AttachLinearSlot(*pool, a, dims, false); Qwen4AttachLinearSlot(*pool, b, dims, false);
    Check(a.cudaData != b.cudaData && pool->FreePageCount() == 0, "linear slots overlap");
    a.Allocate(3.0f); b.Allocate(7.0f);
    Data snapshot; snapshot.CopyFrom(a);
    void *pointer = a.cudaData;
    Qwen4ReleasePagedReference(a); a.FreeSpace();
    Qwen4AttachLinearSlot(*pool, restored, dims, false);
    Check(restored.cudaData == pointer, "linear slot not recycled");
    Data zeros; Tensor(zeros, FLOAT32, dims, std::vector<float>(256, 0), device);
    Equal(restored, zeros, "recycled linear slot not zeroed");
    Qwen4CopyLinearState(snapshot, restored);
    Equal(restored, snapshot, "linear state restore failed");
    Check(restored.cudaData == pointer, "linear restore detached slot");
    std::cout << "SLOTS_PASS device=" << device << '\n';
}
}
namespace fastllm {
struct Qwen4PrefixCacheTestAccess {
    static void TestPools(int device) {
        Qwen4ExpModel model;
        model.block_cnt = 2;
        model.linearLayers = {true, false};
        model.maxBatch = 2;
        model.max_positions = model.tokensLimit = 1024;
        model.kvCacheLimit = 0;
        model.indexerHeadDim = 128;
        model.indexerKvHeads = 1;
        model.indexerCompressRatio = 4;
        std::vector<std::pair<Data, Data>> warmup(2);
        for (Data *t : {&warmup[0].first, &warmup[0].second}) {
            Tensor(*t, FLOAT32, {1, 2, 128}, std::vector<float>(256, 0), device);
            t->isLinearAttention = true;
        }
        Input(warmup[1].first, warmup[1].second, FLOAT16, 1, 1, 0, device);
        model.ReserveServingCache(warmup);
        Check(warmup.empty() && model.servingCache, "startup pool reservation failed");
        Check(model.servingCache->layers[0].first->maxPages == 2, "linear slot budget mismatch");
        ResponseContext request, restored;
        request.Init(2, FLOAT16, FLOAT16); restored.Init(2, FLOAT16, FLOAT16);
        auto &cache = request.pastKeyValues;
        auto &state = model.requestStates[&cache[0].first];
        model.AcquireServingCache(cache, state, 131);
        Append(cache[1].first, cache[1].second, 131, device);
        cache[0].first.Allocate(5.0f);
        state.processedTokens.resize(131); std::iota(state.processedTokens.begin(), state.processedTokens.end(), 0);
        state.indexerRawKeys[1].assign(131 * 128, 0.0f);
        state.indexerPositions[1].resize(131); std::iota(state.indexerPositions[1].begin(), state.indexerPositions[1].end(), 0.0f);
        state.indexerBlockKeys[1].assign((131 / 4) * 128, 0.0f);
        model.MaybeRecordPrefixSnapshot(cache, state);
        Check(model.prefixSnapshots.size() == 1, "paged prefix snapshot missing");
        auto &snapshot = model.prefixSnapshots[0];
        Check(snapshot->layers[1].first.pageIndex == cache[1].first.pageIndex,
              "prefix snapshot copied nonlinear KV instead of sharing pages");
        Check(!snapshot->layers[0].first.isPagedKVCache, "prefix snapshot retained mutable linear slot");
        Check(model.RestorePrefixSnapshot(restored.pastKeyValues, GenerationConfig(), snapshot), "paged prefix restore failed");
        auto &restoreState = model.requestStates[&restored.pastKeyValues[0].first];
        model.AcquireServingCache(restored.pastKeyValues, restoreState, 1);
        Check(restored.pastKeyValues[0].first.cudaData != cache[0].first.cudaData, "restored request shares linear slot");
        Equal(restored.pastKeyValues[0].first, cache[0].first, "restored linear slot contents differ");
        Append(restored.pastKeyValues[1].first, restored.pastKeyValues[1].second, 1, device);
        Verify(restored.pastKeyValues[1].first, restored.pastKeyValues[1].second, device);
        Verify(cache[1].first, cache[1].second, device);
        Check(restored.pastKeyValues[1].first.pageIndex[0] == cache[1].first.pageIndex[0] &&
              restored.pastKeyValues[1].first.pageIndex[1] != cache[1].first.pageIndex[1],
              "prefix COW copied more than the shared partial page");
        // A retained snapshot must keep its pools alive after the model drops
        // its own references, then release page references before the pool dies.
        auto survivor = model.prefixSnapshots.front();
        std::weak_ptr<PagedCacheManager> pool = model.servingCache->layers[1].first;
        for (auto *context : {&request, &restored}) {
            for (auto &kv : context->pastKeyValues) for (Data *tensor : {&kv.first, &kv.second}) {
                Qwen4ReleasePagedReference(*tensor); tensor->FreeSpace();
            }
        }
        model.requestStates.clear();
        model.prefixSnapshots.clear();
        model.servingCache.reset();
        Check(!pool.expired(), "snapshot lost its pool owner");
        survivor.reset();
        Check(pool.expired(), "snapshot leaked its pool owner");
        std::cout << "MODEL_POOLS_PASS device=" << device << '\n';
    }
};
}
int main() {
    try {
        setenv("FASTLLM_PREFIX_CACHE", "1", 1);
        setenv("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", "1", 1);
        setenv("FASTLLM_QWEN4_ENABLE_MTP", "0", 1);
        auto devices = FastllmCudaGetTotalSizes();
        Check(!devices.empty(), "CUDA device required");
        for (int device = 0; device < std::min(2, (int)devices.size()); ++device) {
            FastllmCudaSetDevice(device);
            Executor executor;
            executor.SetFirstDevice("cuda:" + std::to_string(device));
            struct RestoreExecutor { void *previous; ~RestoreExecutor() { SetCurrentThreadExecutor(previous); } } restore{GetExecutor()};
            SetCurrentThreadExecutor(&executor);
            for (DataType type : {FLOAT32, FLOAT16, BFLOAT16}) {
                for (int heads : {1, 2}) TestCache(device, type, heads);
                TestFused(device, type);
            }
            TestSlots(device); TestGraph(device); TestAttention(device);
            Qwen4PrefixCacheTestAccess::TestPools(device);
        }
        std::cout << "ALL_PASS\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
