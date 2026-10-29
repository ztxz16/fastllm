// Exact storage, score, global TopK and graph replay checks for context-sharded Indexer.
#define main original_regression_main
#include "test_naive_n05_fp8_kv.cpp"
#undef main

static void Device(Data &d, DataType type, std::vector<int> shape) {
    d.dataType = type;
    d.UpdateUnitSize();
    d.Resize(shape);
    d.ToDevice(DataDevice::CUDA, {0}, false);
    d.Allocate(false);
}
struct ContextShard {
    PagedCacheManager pool;
    Data table{INT32}, view;
    std::vector<int> ids;
    ContextShard(int capacity, int page, int rank, int ranks, int width, DataType type, bool scaled) {
        int pages = (capacity + page - 1) / page, local = page / ranks;
        pool.dataType = type;
        pool.UpdateUnitSize();
        pool.pageLen = local;
        pool.Resize({pages, local, 1, width});
        pool.ToDevice(DataDevice::CUDA, {0}, false);
        pool.Allocate(false);
        pool.SetMaxPages(pages);
        ids.resize(pages);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), std::mt19937(123 + rank));
        Device(table, INT32, {pages});
        Cuda(cudaMemcpy(table.cudaData, ids.data(), pages * 4, cudaMemcpyHostToDevice));
        view.FakeFrom(table, 0);
        view.dataType = type;
        view.UpdateUnitSize();
        view.Resize({1, 0, width});
        view.isPagedKVCache = true;
        view.pageLen = local;
        view.pagedKVCacheData = &pool;
        view.kvCacheFp8Indexer = scaled;
        view.expansionDims = {1, pages * local, width};
        view.strides = {size_t(pages * local) * width, size_t(width), 1};
    }
};
static void ContextCase(int capacity, int page, int ranks, int batch, int kind, bool uniform, bool graph) {
    DataType type = (kind == 0 || kind == 3) ? BFLOAT16 : FP8_E4M3;
    bool scaled = kind == 2 || kind == 3;
    int width = scaled ? (type == FP8_E4M3 ? 144 : 72) : 128;
    Data raw(BFLOAT16), packed, reference, exchange, full, q(BFLOAT16), weights(BFLOAT16), live(INT32);
    Upload(raw, {1, capacity, 128}, 137);
    const Data *input = &raw;
    if (scaled) {
        FastllmCudaNaivePackIndexer(raw, packed);
        input = &packed;
    }
    Device(reference, type, {1, capacity, width});
    reference.kvCacheFp8Indexer = scaled;
    reference.expansionDims = reference.dims;
    FastllmCudaNaivePagedWrite(reference, *input, 0);
    auto expectedBytes = Read<unsigned char>(reference);
    std::vector<std::unique_ptr<ContextShard>> shards;
    for (int rank = 0; rank < ranks; rank++) {
        shards.emplace_back(new ContextShard(capacity, page, rank, ranks, width, type, scaled));
        auto &shard = *shards.back();
        Cuda(cudaMemset(shard.pool.cudaData, 0xa5, shard.pool.GetBytes()));
        for (int start = 0; start < capacity;) {
            int count = std::min(capacity - start, start ? 4093 : 17);
            Data part;
            part.FakeFrom(*input, size_t(start) * input->dims[2] * input->unitSize);
            part.Resize({1, count, input->dims[2]});
            part.kvCacheFp8Indexer = scaled;
            FastllmCudaNaiveContextIndexerWrite(shard.view, part, start, nullptr, rank, ranks);
            start += count;
        }
        auto stored = Read<unsigned char>(shard.pool);
        int localPage = page / ranks, rowBytes = width * reference.unitSize;
        for (int token = 0; token < capacity; token++)
            if (token % page / localPage == rank) {
                int physical = shard.ids[token / page] * localPage + token % localPage;
                Require(!std::memcmp(stored.data() + size_t(physical) * rowBytes,
                                     expectedBytes.data() + size_t(token) * rowBytes, rowBytes),
                        "sharded append bytes differ");
            }
        FastllmCudaNaiveContextIndexerGather(shard.view, capacity, rank, ranks, exchange);
    }
    FastllmCudaNaiveContextIndexerUnpack(exchange, capacity, page, ranks, full);
    Require(Read<unsigned char>(full) == expectedBytes, "prefill gather changed payload/scales");
    int baseBatch = std::max(4, batch);
    Upload(q, {1, baseBatch, 2048}, 143);
    Upload(weights, {1, baseBatch, 16}, 149);
    if (uniform)
        Cuda(cudaMemset(q.cudaData, 0, q.GetBytes()));
    Device(live, INT32, {baseBatch});
    std::vector<int> lengths(baseBatch, capacity);
    for (int i = 1; i < baseBatch; i++)
        lengths[i] = std::max(0, capacity - i * 37);
    Cuda(cudaMemcpy(live.cudaData, lengths.data(), baseBatch * 4, cudaMemcpyHostToDevice));
    // The legacy batch API accepts an embedded Indexer tail of at least 128
    // BF16 elements. Keep that existing layout as the independent reference.
    Data paddedReference;
    const Data *scoreReference = &reference;
    if (kind == 3) {
        Device(paddedReference, BFLOAT16, {1, capacity, 128});
        paddedReference.kvCacheFp8Indexer = true;
        Cuda(cudaMemset(paddedReference.cudaData, 0, paddedReference.GetBytes()));
        Cuda(cudaMemcpy2D((char *)paddedReference.cudaData + 256 - 144, 256,
                          reference.cudaData, 144, 144, capacity, cudaMemcpyDeviceToDevice));
        scoreReference = &paddedReference;
    }
    std::vector<FastllmNaiveDecodeScratch> refScratch(baseBatch);
    std::vector<Data> refIndices(baseBatch), result(batch);
    std::vector<FastllmNaiveIndexerRow> refRows;
    for (int i = 0; i < baseBatch; i++)
        refRows.push_back({scoreReference, &refScratch[i], &refIndices[i], i, capacity, nullptr});
    std::vector<std::vector<FastllmNaiveDecodeScratch>> scratch(ranks);
    std::vector<std::vector<Data>> localIndices(ranks);
    std::vector<std::vector<FastllmNaiveIndexerRow>> localRows(ranks);
    // Device pointer bindings are updated after graph capture. Poisoning the old
    // pool makes an accidental use of stale captured descriptors observable.
    std::vector<Data> bindings(ranks), movedPools(ranks), movedTables(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        Device(bindings[rank], INT32, {6});
        const void *table[3] = {shards[rank]->pool.cudaData, nullptr, shards[rank]->table.cudaData};
        Cuda(cudaMemcpy(bindings[rank].cudaData, table, sizeof(table), cudaMemcpyHostToDevice));
    }
    Data candidates;
    Device(candidates, INT32, {ranks * batch * 4096});
    for (int rank = 0; rank < ranks; rank++) {
        scratch[rank].resize(batch);
        localIndices[rank].resize(batch);
        for (int i = 0; i < batch; i++)
            localRows[rank].push_back({&shards[rank]->view, &scratch[rank][i], &localIndices[rank][i], i,
                                       capacity, (const void *const *)bindings[rank].cudaData});
    }
    std::vector<FastllmNaiveIndexerRow> merged;
    for (int i = 0; i < batch; i++) {
        Device(result[i], INT32, {1, 2048});
        merged.push_back({&reference, &scratch[0][i], &result[i], i, capacity, nullptr});
    }
    auto run = [&] {
        Require(FastllmCudaNaiveBatchDecodeIndexer(q, weights, live, refRows, true), "reference dispatch");
        for (int rank = 0; rank < ranks; rank++)
            Require(FastllmCudaNaiveBatchDecodeIndexer(q, weights, live, localRows[rank], true, rank, ranks,
                                                       &candidates, page),
                    "context dispatch");
        FastllmCudaNaiveMergeDecodeIndexer(live, merged, candidates, ranks, page);
    };
    Graph captured;
    if (graph)
        captured.Capture(run);
    for (int pass = 0; pass < (graph ? 3 : 1); pass++) {
        if (pass == 1)
            for (int rank = 0; rank < ranks; ++rank) {
                auto &shard = *shards[rank];
                Device(movedPools[rank], type, shard.pool.dims);
                Device(movedTables[rank], INT32, shard.table.dims);
                Cuda(cudaMemcpy(movedPools[rank].cudaData, shard.pool.cudaData, shard.pool.GetBytes(),
                                cudaMemcpyDeviceToDevice));
                Cuda(cudaMemcpy(movedTables[rank].cudaData, shard.table.cudaData, shard.table.GetBytes(),
                                cudaMemcpyDeviceToDevice));
                const void *table[3] = {movedPools[rank].cudaData, nullptr, movedTables[rank].cudaData};
                Cuda(cudaMemcpy(bindings[rank].cudaData, table, sizeof(table), cudaMemcpyHostToDevice));
                Cuda(cudaMemset(shard.pool.cudaData, 0, shard.pool.GetBytes()));
            }
        if (pass)
            for (int i = 0; i < baseBatch; i++)
                lengths[i] = std::max(0, lengths[i] - 29 - i);
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), baseBatch * 4, cudaMemcpyHostToDevice));
        if (graph)
            captured.Run();
        else
            run();
        for (int i = 0; i < batch; i++) {
            Require(Read<int>(result[i]) == Read<int>(refIndices[i]), "context global TopK differs");
            auto scores = Read<unsigned>(refScratch[i].indexScores);
            for (int rank = 0; rank < ranks; rank++) {
                auto local = Read<unsigned>(scratch[rank][i].indexScores);
                int p = page / ranks;
                int count = lengths[i] / page * p + std::clamp(lengths[i] % page - rank * p, 0, p);
                for (int j = 0; j < count; j++) {
                    int global = j / p * page + rank * p + j % p;
                    Require(local[j] == scores[global], "context score bits differ");
                }
            }
        }
        checks++;
    }
    printf("CONTEXT PASS len=%d page=%d TP=%d B=%d kind=%d uniform=%d graph=%d\n", capacity, page, ranks,
           batch, kind, uniform, graph);
    fflush(stdout);
}
static void ContextRope(int ranks, bool scaled) {
    const int page = 128, capacity = 512, width = scaled ? 144 : 128;
    for (int rank = 0; rank < ranks; rank++) {
        FP8Tensor key(capacity, 192, 51 + rank, true), value(capacity, 128, 73 + rank, true);
        ContextShard shard(capacity, page, rank, ranks, width, FP8_E4M3, scaled);
        shard.view.cudaData = key.data().cudaData;
        Data q(BFLOAT16), qr(BFLOAT16), k(BFLOAT16), kr(BFLOAT16), v(BFLOAT16), vr(BFLOAT16), ik(BFLOAT16),
            ir(BFLOAT16), pos(FLOAT32), live(INT32), packed, expected;
        Upload(q, {1, 1, 16 * 192}, 11);
        Upload(qr, q.dims, 11);
        Upload(k, {1, 1, 192}, 17);
        Upload(kr, k.dims, 17);
        Upload(v, {1, 1, 128}, 31);
        Upload(vr, v.dims, 31);
        Upload(ik, {1, 1, 128}, 41);
        Upload(ir, ik.dims, 41);
        Upload(pos, {1, 1}, 77);
        Device(live, INT32, {1});
        Device(expected, FP8_E4M3, {1, 1, width});
        expected.expansionDims = expected.dims;
        expected.kvCacheFp8Indexer = scaled;
        const auto q0 = Read<uint16_t>(q), k0 = Read<uint16_t>(k), v0 = Read<uint16_t>(v),
                   i0 = Read<uint16_t>(ik);
        std::vector<FastllmNaiveRopeCacheRow> rows{
            {&key.data(), &value.data(), &ik, nullptr, &shard.view, rank}};
        int length = 1;
        Cuda(cudaMemcpy(live.cudaData, &length, 4, cudaMemcpyHostToDevice));
        auto run = [&] {
            Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, pos, rows, live, 16, 1, 192, 128, 64, 1e7,
                                                         .707, 0),
                    "context fused rope declined");
        };
        Graph graph;
        graph.Capture(run);
        for (int past :
             {0, page / ranks - 1, page / ranks, page - 1, page, page + page / ranks, 2 * page - 1}) {
            Cuda(cudaMemcpy(q.cudaData, q0.data(), q0.size() * 2, cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(qr.cudaData, q0.data(), q0.size() * 2, cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(kr.cudaData, k0.data(), k0.size() * 2, cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(vr.cudaData, v0.data(), v0.size() * 2, cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(ir.cudaData, i0.data(), i0.size() * 2, cudaMemcpyHostToDevice));
            Cuda(cudaMemset(shard.pool.cudaData, 0xa5, shard.pool.GetBytes()));
            auto expectedShard = Read<uint8_t>(shard.pool), expectedKey = Read<uint8_t>(key.pool),
                 expectedValue = Read<uint8_t>(value.pool);
            length = past + 1;
            float position = 8192 + past;
            Cuda(cudaMemcpy(live.cudaData, &length, 4, cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(pos.cudaData, &position, 4, cudaMemcpyHostToDevice));
            graph.Run();
            FastllmCudaNaiveRopeQKScaleV(qr, kr, vr, pos, 16, 1, 192, 128, 64, 1e7, .707);
            FastllmCudaNaiveRope(ir, pos, 1, 128, 64, 1e7);
            const Data *input = &ir;
            if (scaled) {
                FastllmCudaNaivePackIndexer(ir, packed);
                input = &packed;
            }
            FastllmCudaNaivePagedWrite(expected, *input, 0);
            auto bytes = Read<uint8_t>(expected);
            if (past % page / (page / ranks) == rank) {
                int physical = key.ids[past / page] * (page / ranks) + past % (page / ranks);
                std::copy(bytes.begin(), bytes.end(), expectedShard.begin() + size_t(physical) * width);
            }
            auto hk = Read<uint16_t>(kr), hv = Read<uint16_t>(vr);
            int kp = key.ids[past / page] * page + past % page,
                vp = value.ids[past / page] * page + past % page;
            for (int j = 0; j < 192; j++)
                expectedKey[size_t(kp) * 192 + j] = __nv_fp8_e4m3(Float(hk[j])).__x;
            for (int j = 0; j < 128; j++)
                expectedValue[size_t(vp) * 128 + j] = __nv_fp8_e4m3(Float(hv[j])).__x;
            Require(Read<uint16_t>(q) == Read<uint16_t>(qr), "context rope changed Q");
            Require(Read<uint8_t>(key.pool) == expectedKey, "context rope changed main K");
            Require(Read<uint8_t>(value.pool) == expectedValue, "context rope changed main V");
            Require(Read<uint8_t>(shard.pool) == expectedShard, "context rope index shard ownership/bytes");
            checks++;
        }
    }
    printf("CONTEXT ROPE PASS TP=%d scaled=%d, page boundaries and graph length updates\n", ranks, scaled);
    fflush(stdout);
}

#include "../../src/models/naive_n05_kvmem.h"
namespace fastllm {
// Existing friend access exercises the real production copy-on-write path.
struct NaiveN05PrefixCacheTestAccess : NaiveN05FlashModel {
    static void CheckContextCopyOnWrite(int ranks, DataType type, bool scaled) {
        NaiveN05PrefixCacheTestAccess model;
        model.max_positions = 1024;
        model.tokensLimit = 1024;
        model.sparseKv = std::make_shared<SparseKvState>();
        constexpr int pages = 8, page = 128, oldLength = 97;
        PagedCacheManager main, index;
        Device(main, type, {pages, page, 1, 192});
        main.pageLen = page;
        main.SetMaxPages(pages);
        int localPage = page / ranks, width = scaled ? (type == FP8_E4M3 ? 144 : 72) : 128;
        Device(index, type, {pages, localPage, 1, width});
        index.pageLen = localPage;
        index.SetMaxPages(pages);
        model.sparseKv->indexSidecars[&main] = &index;
        int old = main.GetUnusedPageIndex(true);
        main.pageRefCount[old]++;
        Data cache(INT32, {pages});
        cache.ToDevice(DataDevice::CUDA, {0}, false);
        cache.Allocate(false);
        cache.dataType = type;
        cache.UpdateUnitSize();
        cache.isPagedKVCache = true;
        cache.pagedKVCacheData = &main;
        cache.pageLen = page;
        cache.pageIndex = {old};
        cache.Resize({1, oldLength, 192});
        size_t bytes = index.GetBytes(), pageBytes = bytes / pages;
        std::vector<unsigned char> initial(bytes);
        for (size_t i = 0; i < bytes; ++i)
            initial[i] = (i * 13 + i / pageBytes * 29) % 251;
        Cuda(cudaMemcpy(index.cudaData, initial.data(), bytes, cudaMemcpyHostToDevice));
        Require(model.PreparePagedCache(cache, main, oldLength + 1), "shared tail was not copied");
        int copy = cache.pageIndex[0];
        Require(copy != old && main.pageRefCount[old] == 1, "shared page reference count changed");
        std::vector<unsigned char> actual(bytes);
        Cuda(cudaMemcpy(actual.data(), index.cudaData, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < pages; ++i) {
            const unsigned char *expected = initial.data() + (i == copy ? old : i) * pageBytes;
            Require(std::equal(expected, expected + pageBytes, actual.data() + i * pageBytes),
                    "context Indexer tail copy changed payload or neighboring pages");
        }
        Cuda(cudaMemset((char *)index.cudaData + copy * pageBytes, 0x5a, pageBytes));
        Cuda(cudaMemcpy(actual.data(), index.cudaData, bytes, cudaMemcpyDeviceToHost));
        Require(std::equal(initial.data() + old * pageBytes, initial.data() + (old + 1) * pageBytes,
                           actual.data() + old * pageBytes),
                "copy-on-write changed shared prefix");
        model.sparseKv->indexSidecars.clear();
        checks++;
    }
};
} // namespace fastllm
int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 77;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess || properties.major < 8)
        return 77;
    try {
        for (int ranks : {2, 4, 8})
            for (DataType type : {BFLOAT16, FP8_E4M3})
                for (bool scaled : {false, true})
                    NaiveN05PrefixCacheTestAccess::CheckContextCopyOnWrite(ranks, type, scaled);
        for (int ranks : {2, 4, 8})
            for (bool scaled : {false, true})
                ContextRope(ranks, scaled);
        for (int ranks : {2, 4, 8})
            for (int kind : {0, 1, 2, 3})
                for (int page : {96, 128}) {
                    for (int n : {33, 129, 2049, 8193, 65537})
                        ContextCase(n, page, ranks, 4, kind, false, true);
                    ContextCase(32769, page, ranks, 16, kind, true, true);
                }
        // Cross common batch sizes and the large-batch descriptor grouping.
        for (int batch : {15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 256, 384})
            ContextCase(8193, 128, 4, batch, 1, false, true);
        ContextCase(200000, 128, 4, 1, 1, false, true);
        ContextCase(200000, 128, 4, 32, 1, false, true);
        printf("PASS %d context-shard checks\n", checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
