#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#include "devices/cuda/naive-n05-dsa-layer.cuh"
#include "devices/cuda/naive-n05-dp-dsa-verify.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "utils/utils.h"
#include "naive_n05_mxfp8_test.h"
#include <cuda_fp8.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace fastllm;
using B = __nv_bfloat16;
#include "naive_n05_dsa_cache_fixture.cuh"
namespace {
double Compare(const Data &a, const Data &b, int rows, const char *name) {
    Require(a.Count(0) == b.Count(0), "output shape mismatch");
    auto x = Read(a), y = Read(b);
    size_t width = a.Count(0) / rows;
    double worst = 0;
    for (int row = 0; row < rows; ++row) {
        double error = 0, magnitude = 0;
        for (size_t j = 0; j < width; ++j) {
            size_t i = row * width + j;
            B p, q;
            std::memcpy(&p, x.data() + 2 * i, 2);
            std::memcpy(&q, y.data() + 2 * i, 2);
            double u = float(p), v = float(q);
            Require(std::isfinite(u) && std::isfinite(v), "non-finite output");
            error += (u - v) * (u - v);
            magnitude += v * v;
        }
        double relative = std::sqrt(error / (magnitude + 1e-30));
        worst = std::max(worst, relative);
        if (std::sqrt(error) > .006 * std::sqrt(magnitude) + 1e-6)
            throw std::runtime_error(std::string(name) + " row=" + std::to_string(row) +
                                     " relative L2=" + std::to_string(relative));
    }
    return worst;
}
bool testFp8KV = false;
// Decode the actual FP8 bytes into the BF16 reference's logical token order.
// This keeps both attention paths on identical Q/K/V, including independent page tables.
__global__ void ExpandCache(FastllmNaiveDSARow binding, B *keys, B *values) {
    const auto r = FastllmNaiveResolveDSARow(binding);
    const int token = blockIdx.x, d = threadIdx.x;
    if (d < 320) keys[token * 320 + d] = __float2bfloat16(float(
        static_cast<const __nv_fp8_e4m3 *>(r.key)[size_t(FastllmNaivePhysicalRow(r, token)) * 320 + d]));
    if (d < 128) values[token * 128 + d] = __float2bfloat16(float(
        static_cast<const __nv_fp8_e4m3 *>(r.value)[size_t(FastllmNaivePhysicalRow(r, token, true)) * 128 + d]));
}
// Read only the newly published physical K/Indexer and V rows. The two
// page tables are independent and may change when the graph is replayed.
__global__ void ReadFp8Candidate(FastllmNaiveDSARow binding, uint8_t *bytes) {
    const auto r = FastllmNaiveResolveDSARow(binding);
    const int length = r.liveKeys ? *r.liveKeys : r.length, col = threadIdx.x;
    if (length <= 0 || length > r.capacity) { bytes[col] = 0; return; }
    if (col < 320)
        bytes[col] = static_cast<const uint8_t *>(r.key)[size_t(FastllmNaivePhysicalRow(r, length - 1)) * r.keyStride + col];
    else
        bytes[col] = static_cast<const uint8_t *>(r.value)[size_t(FastllmNaivePhysicalRow(r, length - 1, true)) * 128 + col - 320];
}
struct Case {
    int rows, capacity, device;
    bool fp8, paged;
    std::vector<int> requests;
    Data feature, referenceFeature;
    Data hidden, referenceHidden, initial, branch, gamma, post, qkvWeight, ikWeight, iqWeight, iwWeight,
        oWeight;
    Data ikNorm, ikBias, sink, positions, live, empty, q, attention, projected, normed;
    Data rnorm, raw, ik, iq, iw, rq, rk, rv, rattn, rprojected, indices, exactAttention;
    FastllmNaiveDecodeScratch scratch, referenceScratch, exactScratch;
    std::vector<std::unique_ptr<Cache>> caches, referenceCaches, alternates, referenceAlternates;
    FastllmNaiveDSALayerInput input;
    FastllmNaiveDSALayerArgs args{};
    std::vector<Data> indexViews;
    std::vector<FastllmNaiveRopeCacheRow> referenceBindings;
    std::vector<FastllmNaiveVerifyIndexerItem> items;
    std::vector<Data> indexRequests;
    int finalTileRows = 0;
    std::vector<FastllmNaiveDSARow> referenceRows;
    Case(std::vector<int> requests, int capacity, bool paged, bool fp8, int device = 0, bool mx = false)
        : rows(0), capacity(capacity), device(device), fp8(fp8), paged(paged), requests(requests) {
        for (int n : requests)
            rows += n;
        for (Data *d : {&hidden, &referenceHidden, &initial, &branch, &feature, &referenceFeature})
            Init(*d, BFLOAT16, {1, rows, 4096});
        Fill(initial, 3);
        Fill(branch, 5, .2f);
        for (Data *d : {&gamma, &post}) {
            Init(*d, FLOAT32, {4096});
            Fill(*d, 7, .1f, 1);
        }
        Init(qkvWeight, mx ? FP8_E4M3 : BFLOAT16, {1856, 4096});
        Init(ikWeight, BFLOAT16, {128, 4096});
        Init(iqWeight, BFLOAT16, {2048, 4096});
        Init(iwWeight, BFLOAT16, {16, 4096});
        Init(oWeight, mx ? FP8_E4M3 : BFLOAT16, {4096, 1024});
        if (mx) FillVerifyMxWeight(qkvWeight,13+device); else Fill(qkvWeight,13 + device,.03f);
        Fill(ikWeight, 17, .03f);
        Fill(iqWeight, 19, .03f);
        Fill(iwWeight, 23, .03f);
        if (mx) FillVerifyMxWeight(oWeight,29+device); else Fill(oWeight,29 + device,.03f);
        Init(ikNorm, FLOAT32, {128});
        Init(ikBias, FLOAT32, {128});
        Init(sink, FLOAT32, {8});
        Fill(ikNorm, 31, .1f, 1);
        Fill(ikBias, 37, .1f);
        Fill(sink, 41, .2f);
        Init(positions, FLOAT32, {1, rows});
        Init(live, INT32, {rows});
        Init(indices, INT32, {rows, 2048});
        for (auto [d, w] : std::vector<std::pair<Data *, int>>{{&rnorm, 4096},
                                                               {&raw, 1856},
                                                               {&ik, 128},
                                                               {&iq, 2048},
                                                               {&iw, 16},
                                                               {&rq, 1536},
                                                               {&rk, 192},
                                                               {&rv, 128},
                                                               {&rprojected, 4096}})
            Init(*d, BFLOAT16, {1, rows, w});
        const int bits = testFp8KV ? 8 : 16;
        for (size_t i = 0; i < requests.size(); ++i) {
            caches.emplace_back(new Cache(capacity, paged, fp8, 43 + i * 10, 1, bits));
            referenceCaches.emplace_back(new Cache(capacity, paged, fp8, 43 + i * 10, 1, bits));
            alternates.emplace_back(new Cache(capacity, paged, fp8, 83 + i * 10, 1, bits));
            referenceAlternates.emplace_back(new Cache(capacity, paged, fp8, 83 + i * 10, 1, bits));
        }
        input.branch = &branch;
        input.hidden = &hidden;
        input.feature = &feature;
        input.inputNorm = &gamma;
        input.postNorm = &post;
        input.qkvWeight = &qkvWeight;
        input.qkvBias = &empty;
        input.indexKeyWeight = &ikWeight;
        input.indexQueryWeight = &iqWeight;
        input.indexWeight = &iwWeight;
        input.indexNorm = &ikNorm;
        input.indexBias = &ikBias;
        input.outputWeight = &oWeight;
        input.sink = &sink;
        input.positions = &positions;
        input.live = &live;
        input.query = &q;
        input.attention = &attention;
        input.projected = &projected;
        input.normed = &normed;
        input.scratch = &scratch;
        input.requests = requests;
        input.capacity = capacity;
        input.theta = 10000000;
        input.valueScale = .75f;
        indexViews.resize(rows);
        indexRequests.resize(requests.size());
        int offset = 0;
        for (size_t req = 0; req < requests.size(); ++req) {
            auto &a = *caches[req], &b = *referenceCaches[req];
            indexRequests[req].Resize({requests[req], 2048});
            indexRequests[req].FakeFrom(indices, size_t(offset) * 2048 * 4);
            items.push_back({b.K(), &indexRequests[req], offset, requests[req], 0, capacity,
                             (const void *const *)b.table.cudaData});
            for (int j = 0; j < requests[req]; ++j) {
                input.caches.push_back({a.K(), a.V(), nullptr, (const void *const *)a.table.cudaData});
                View(indexViews[offset + j], ik, {1, 1, 128}, size_t(offset + j) * 128 * 2);
                referenceBindings.push_back(
                    {b.K(), b.V(), &indexViews[offset + j], (const void *const *)b.table.cudaData});
            }
            offset += requests[req];
        }
        Lengths(16384, false);
        Reset();
        if (rows <= 16 && !(mx && !testFp8KV && rows > 8)) {
            Require(FastllmCudaNaiveDSALayerPrepare(input, device, args), "valid DSA layer rejected");
            referenceRows.assign(args.bindings, args.bindings + rows);
        } else {
            std::vector<FastllmNaiveDSALayerArgs> tiles;
            Require(FastllmCudaNaiveDSALayerPrepare(input, device, args, &tiles), "tiled DSA rejected");
            if (testFp8KV)
                Require(tiles.size() == 2 && tiles[0].rows == 16 && tiles[1].rows == rows - 16, "DSA pair partition");
            else {
                Require(tiles.size() == requests.size(), "BF16 DSA request partition");
                for (size_t i = 0; i < tiles.size(); ++i)
                    Require(tiles[i].rows == requests[i], "BF16 DSA tail length");
            }
            finalTileRows = tiles.back().rows;
            for (const auto &tile : tiles)
                referenceRows.insert(referenceRows.end(), tile.bindings, tile.bindings + tile.rows);
        }
        offset = 0;
        for (size_t req = 0; req < requests.size(); ++req) {
            auto &b = *referenceCaches[req];
            for (int j = 0; j < requests[req]; ++j) {
                auto &r = referenceRows[offset + j];
                r.key = b.key.cudaData;
                r.value = b.value.cudaData;
                r.cache = (const void *const *)b.table.cudaData;
                r.keyPages = paged ? (const int *)b.kp.cudaData : nullptr;
                r.valuePages = paged ? (const int *)b.vp.cudaData : nullptr;
                r.indices = (const int *)indices.cudaData + size_t(offset + j) * 2048;
            }
            offset += requests[req];
        }
    }
    void Reset() {
        CopyDevice(hidden, initial);
        CopyDevice(referenceHidden, initial);
    }
    void Lengths(int past, bool empty) {
        std::vector<int> lengths(rows);
        std::vector<float> pos(rows);
        int offset = 0;
        for (size_t req = 0; req < requests.size(); ++req)
            for (int j = 0; j < requests[req]; ++j) {
                int length =
                    empty && req == 0 ? 0 : std::min(capacity, std::max(0, past - int(req) * 257 + j + 1));
                lengths[offset] = length;
                pos[offset] = std::max(0, length - 1);
                ++offset;
            }
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), rows * 4, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(positions.cudaData, pos.data(), rows * 4, cudaMemcpyHostToDevice));
    }
    void Rebind(bool alternate) {
        for (size_t i = 0; i < caches.size(); ++i) {
            caches[i]->Bind(alternate ? *alternates[i] : *caches[i]);
            referenceCaches[i]->Bind(alternate ? *referenceAlternates[i] : *referenceCaches[i]);
        }
    }
    void ReferenceCore(const Data &query, const Data &indexQuery, const Data &weights,
                       const std::vector<FastllmNaiveVerifyIndexerItem> &indexItems,
                       const std::vector<FastllmNaiveDSARow> &descriptors, FastllmNaiveDecodeScratch &workspace, Data &out) {
        if (testFp8KV) {
            auto items = indexItems;
            auto rows = descriptors;
            std::vector<Data> key(items.size()), value(items.size());
            for (size_t i = 0; i < items.size(); ++i) {
                auto &item = items[i];
                Init(key[i], BFLOAT16, {1, capacity, 320});
                Init(value[i], BFLOAT16, {1, capacity, 128});
                ExpandCache<<<capacity, 320, 0, cudaStreamPerThread>>>(rows[item.offset], (B *)key[i].cudaData, (B *)value[i].cudaData);
                Cuda(cudaGetLastError());
                item.key = &key[i]; item.cache = nullptr;
                for (int j = 0; j < item.rows; ++j) {
                    auto &r = rows[item.offset+j];
                    r.key = key[i].cudaData; r.value = value[i].cudaData; r.cache = nullptr;
                    r.keyPages = r.valuePages = nullptr; r.pageLen = 0; r.fp8 = false;
                }
            }
            Require(FastllmCudaNaiveFusedDSAVerify(query, indexQuery, weights, sink, items, rows, true,
                        8, 1, 192, 128, workspace, out), "expanded FP8 reference");
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        } else {
            Require(FastllmCudaNaiveFusedDSAVerify(query, indexQuery, weights, sink, indexItems, descriptors, true,
                                                   8, 1, 192, 128, workspace, out), "core reference");
        }
    }
    void Legacy(bool tp = false) {
        FastllmCudaNaiveAddDecodeRMSNorm(referenceHidden, branch, gamma, input.eps, rnorm);
        CopyDevice(referenceFeature, referenceHidden);
        Require((qkvWeight.dataType == FP8_E4M3 ? VerifyMxLinearReference : FastllmCudaBFloat16MatMulBFloat16)(rnorm, qkvWeight, empty, raw, rows, 4096, 1856),
                "QKV reference");
        Require(FastllmCudaBFloat16MatMulBFloat16(rnorm, ikWeight, empty, ik, rows, 4096, 128),
                "IK reference");
        Require(FastllmCudaBFloat16MatMulBFloat16(rnorm, iqWeight, empty, iq, rows, 4096, 2048),
                "IQ reference");
        Require(FastllmCudaBFloat16MatMulBFloat16(rnorm, iwWeight, empty, iw, rows, 4096, 16),
                "IW reference");
        FastllmCudaNaiveIndexerKeyNorm(ik, ikNorm, ikBias);
        FastllmCudaNaiveRope(iq, positions, 16, 128, 64, input.theta);
        FastllmCudaMul(iw, .25f, iw);
        Require(FastllmCudaNaiveBatchRopeAppendCache(rq, rk, rv, positions, referenceBindings, live, 8, 1,
                                                     192, 128, 64, input.theta, input.valueScale, 0, &raw,
                                                     false),
                "append reference");
        if (rows <= 16) {
            ReferenceCore(rq, iq, iw, items, referenceRows, referenceScratch, rattn);
        } else {
            Init(rattn, BFLOAT16, {1,rows,1024});
            for (auto item : items) {
                const int n = item.rows, first = item.offset;
                Data q1, iq1, iw1, attn1;
                View(q1, rq, {1,n,1536}, size_t(first)*1536*2);
                View(iq1, iq, {1,n,2048}, size_t(first)*2048*2);
                View(iw1, iw, {1,n,16}, size_t(first)*16*2);
                View(attn1, rattn, {1,n,1024}, size_t(first)*1024*2);
                std::vector<FastllmNaiveDSARow> desc(referenceRows.begin()+first, referenceRows.begin()+first+n);
                item.offset = 0;
                ReferenceCore(q1, iq1, iw1, {item}, desc, referenceScratch, attn1);
            }
        }
        Require((oWeight.dataType == FP8_E4M3 ? VerifyMxLinearReference : FastllmCudaBFloat16MatMulBFloat16)(rattn, oWeight, empty, rprojected, rows, 1024, 4096),
                "O reference");
        if (tp)
            Require(FastllmCudaCustomAllReduce(rprojected.cudaData, rprojected.cudaData, rows * 4096,
                                               BFLOAT16, device),
                    "TP reference");
        FastllmCudaNaiveAddDecodeRMSNorm(referenceHidden, rprojected, post, input.eps, rnorm);
    }
    double Check(bool tp = false, bool checkProjectedResidual = false) {
        // Projection rounding can change sparse Top-K membership. Verify the
        // attention body with identical projected operands/cache, then compare
        // the complete residual against the independent unfused implementation.
        if (finalTileRows) {
            // Only the last tile is retained in scratch; queries, residuals and
            // normalized output are published into distinct rows for every tile.
            Data actual, reference;
            View(actual, scratch.dsaLayer, {1, finalTileRows, 1856}, FastllmNaiveDSALayerLayout::Qkv);
            View(reference, raw, {1, finalTileRows, 1856}, size_t(rows - finalTileRows) * 1856 * 2);
            Compare(actual, reference, finalTileRows, "final tile QKV");
            const auto lengths = Read(live);
            for (int row = 0; row < rows; ++row) {
                int visible;
                std::memcpy(&visible, lengths.data() + row * sizeof(int), sizeof(int));
                if (visible <= 0) continue;
                Data actualQuery, referenceQuery;
                View(actualQuery, q, {1, 1, 1536}, size_t(row)*1536*2);
                View(referenceQuery, rq, {1, 1, 1536}, size_t(row)*1536*2);
                Compare(actualQuery, referenceQuery, 1, "tiled query/RoPE");
            }
            Require(Read(feature) == Read(referenceFeature), "tiled feature differs");
            Compare(normed, rnorm, rows, "tiled post norm");
            return Compare(hidden, referenceHidden, rows, "tiled hidden");
        }
        Data rawView, ikView, iqView, iwView;
        View(rawView, scratch.dsaLayer, {1, rows, 1856}, FastllmNaiveDSALayerLayout::Qkv);
        View(ikView, scratch.dsaLayer, {1, rows, 128}, FastllmNaiveDSALayerLayout::IndexKey);
        View(iqView, scratch.dsaLayer, {1, rows, 2048}, FastllmNaiveDSALayerLayout::IndexQuery);
        View(iwView, scratch.dsaLayer, {1, rows, 16}, FastllmNaiveDSALayerLayout::IndexWeight);
        Compare(rawView, raw, rows, "QKV projection");
        Compare(iwView, iw, rows, "index weights");
        auto lengths = Read(live);
        for (int row = 0; row < rows; ++row) {
            int length;
            std::memcpy(&length, lengths.data() + row * 4, 4);
            if (length <= 0)
                continue; // Inactive requests have no query/cache publication contract.
            for (auto item : std::vector<std::pair<const Data *, const Data *>>{
                     {&ikView, &ik}, {&iqView, &iq}, {&q, &rq}}) {
                int width = item.first->Count(0) / rows;
                Data actual, reference;
                View(actual, *item.first, {1, 1, width}, size_t(row) * width * 2);
                View(reference, *item.second, {1, 1, width}, size_t(row) * width * 2);
                Compare(actual, reference, 1, "active projection/RoPE");
            }
        }
        if (testFp8KV) {
            auto expectedBindings = referenceBindings;
            std::vector<Data> indexKeys(rows);
            for (int row = 0; row < rows; ++row) {
                View(indexKeys[row], ikView, {1, 1, 128}, size_t(row) * 128 * 2);
                expectedBindings[row].indexKey = &indexKeys[row];
            }
            Require(FastllmCudaNaiveBatchRopeAppendCache(rq, rk, rv, positions, expectedBindings, live,
                8, 1, 192, 128, 64, input.theta, input.valueScale, 0, &rawView, false), "same-input FP8 append");
            Data actual, expected;
            Init(actual, FP8_E4M3, {rows, 448});
            Init(expected, FP8_E4M3, {rows, 448});
            for (int row = 0; row < rows; ++row) {
                ReadFp8Candidate<<<1, 448, 0, cudaStreamPerThread>>>(args.bindings[row], static_cast<uint8_t *>(actual.cudaData) + row * 448);
                ReadFp8Candidate<<<1, 448, 0, cudaStreamPerThread>>>(referenceRows[row], static_cast<uint8_t *>(expected.cudaData) + row * 448);
            }
            Cuda(cudaGetLastError());
            Require(Read(actual) == Read(expected), "same-input FP8 K/Indexer/V bytes differ");
        }
        auto exactItems = items;
        auto exactRows = referenceRows;
        int offset = 0;
        for (size_t req = 0; req < requests.size(); ++req) {
            exactItems[req].key = input.caches[offset].key;
            exactItems[req].cache = input.caches[offset].cache;
            for (int j = 0; j < requests[req]; ++j) {
                exactRows[offset + j] = args.bindings[offset + j];
                exactRows[offset + j].indices = (const int *)indices.cudaData + size_t(offset + j) * 2048;
            }
            offset += requests[req];
        }
        ReferenceCore(q, iqView, iwView, exactItems, exactRows, exactScratch, exactAttention);
        // The selected set is exact; token ordering and QK/PV accumulation may
        // change BF16 rounding, so validate attention numerically on identical inputs.
        double attentionError = Compare(attention, exactAttention, rows, "identical-input attention");
        Data selectedView;
        selectedView.dataType = INT32;
        selectedView.Resize({rows, 2048});
        selectedView.FakeFrom(scratch.dsaLayer, FastllmNaiveDSALayerLayout::Indices);
        auto actualSelections = Read(selectedView), referenceSelections = Read(indices);
        for (int row = 0; row < rows; ++row) {
            int visible;
            std::memcpy(&visible, lengths.data() + row * 4, 4);
            const int count = std::min(2048, std::max(0, visible));
            std::vector<int> actual(count), reference(count);
            if (count) {
                std::memcpy(actual.data(), actualSelections.data() + row * 2048 * 4, count * 4);
                std::memcpy(reference.data(), referenceSelections.data() + row * 2048 * 4, count * 4);
                Require(std::is_sorted(actual.begin(), actual.end()), "selected tokens not ordered");
                std::sort(reference.begin(), reference.end());
                Require(actual == reference, "selected token set differs");
            }
        }
        printf("ATTENTION rows=%d relative_l2=%.8g exact_topk_set=PASS\n", rows, attentionError);
        Require(Read(feature) == Read(referenceFeature), "exported feature differs");
        if (checkProjectedResidual) {
            // Isolate the sparse pipeline beyond 32K. Small projection rounding differences
            // can move a key across the cutoff. Compare the full O, TP and residual chain
            // on the independent attention reference for the same projected operands.
            Require(tp && qkvWeight.dataType == FP8_E4M3, "MX TP reference expected");
            Require(VerifyMxLinearReference(exactAttention, oWeight, empty, rprojected, rows, 1024, 4096),
                    "identical-input O projection");
            Require(FastllmCudaCustomAllReduce(rprojected.cudaData, rprojected.cudaData, rows * 4096, BFLOAT16, device),
                    "identical-input TP reduction");
            CopyDevice(referenceHidden, initial);
            FastllmCudaNaiveAddDecodeRMSNorm(referenceHidden, branch, gamma, input.eps, rnorm);
            FastllmCudaNaiveAddDecodeRMSNorm(referenceHidden, rprojected, post, input.eps, rnorm);
        }
        double err = Compare(hidden, referenceHidden, rows, "hidden");
        if (tp)
            Compare(normed, rnorm, rows, "post norm");
        return err;
    }
    void Reject() {
        auto h = Read(hidden), k = Read(caches[0]->key), v = Read(caches[0]->value);
        auto bad = input;
        bad.capacity = 65537;
        FastllmNaiveDSALayerArgs ignored{};
        Require(!FastllmCudaNaiveDSALayerPrepare(bad, device, ignored), "capacity admitted");
        bad = input;
        bad.fp8Query = false;
        Require(!FastllmCudaNaiveDSALayerPrepare(bad, device, ignored), "query format admitted");
        bad = input;
        bad.query = &hidden;
        Require(!FastllmCudaNaiveDSALayerPrepare(bad, device, ignored), "output alias admitted");
        bad = input;
        bad.requests = {rows + 1};
        Require(!FastllmCudaNaiveDSALayerPrepare(bad, device, ignored), "request mismatch admitted");
        Require(h == Read(hidden) && k == Read(caches[0]->key) && v == Read(caches[0]->value),
                "fallback mutated data");
    }
};
bool testMxProjection = false;
template <class F> void Parallel(F f) {
    std::array<std::exception_ptr, 8> errors{};
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 8; ++rank)
        workers.emplace_back([&, rank] {
            try {
                FastllmCudaSetDevice(rank);
                std::shared_ptr<void> lease;
                std::unique_ptr<FastllmCudaLinearGraphScope> scope;
                if (testMxProjection) scope = std::make_unique<FastllmCudaLinearGraphScope>(lease);
                f(rank);
            } catch (...) {
                errors[rank] = std::current_exception();
            }
        });
    for (auto &worker : workers)
        worker.join();
    for (auto &error : errors)
        if (error)
            std::rethrow_exception(error);
}
void Tp(bool fp8, bool mx = false) {
    const std::vector<int> devices{0, 1, 2, 3, 4, 5, 6, 7};
    Require(FastllmInitNccl(devices) && FastllmCudaCustomAllReduceInit(devices), "TP init");
    for (auto shape : testFp8KV && mx ? std::vector<std::vector<int>>{{1},{2},{3},{8},{4,4},{8,1},{3,7},{8,7},{7,8},{8,8},{8,8,4},{8,8,8,8}} : mx ? std::vector<std::vector<int>>{{1},{3},{4},{8},{4,4},{8,1},{3,7},{8,8,4}} : std::vector<std::vector<int>>{{8}, {4, 4}, {8, 8}, {1, 8}, {7, 8}, {1, 2, 5, 8}}) {
        std::array<std::unique_ptr<Case>, 8> cases;
        const bool fullMx = mx && shape == std::vector<int>{8};
        Parallel([&](int r) { cases[r] = std::make_unique<Case>(shape, fullMx ? 65536 : 32768, testFp8KV, fp8, r, mx); });
        Parallel([&](int r) {
            auto &c = *cases[r];
            auto h = Read(c.hidden), k = Read(c.caches[0]->key), v = Read(c.caches[0]->value);
            auto bad = c.input;
            if (r == 3)
                bad.capacity = 65537;
            Require(!FastllmCudaNaiveDSALayer(bad, r), "partial rank admission");
            Require(h == Read(c.hidden) && k == Read(c.caches[0]->key) && v == Read(c.caches[0]->value),
                    "TP fallback wrote data");
        });
        Parallel([&](int r) {
            cases[r]->Legacy(true);
            Cuda(cudaDeviceSynchronize());
        });
        Parallel([&](int r) {
            FastllmNaiveVerifyAttentionInput verify;
            static_cast<FastllmNaiveDSALayerInput &>(verify) = cases[r]->input;
            verify.config = {8, 1, 192, 128, 64, 0, verify.eps, verify.theta, verify.valueScale};
            verify.fp8IndexerCache = fp8;
            Require(FastllmCudaVerifyAttention(verify, r, nullptr) == FastllmVerifyOutput::Normalized,
                    "TP verification dispatch rejected the index query/cache combination");
            Cuda(cudaDeviceSynchronize());
        });
        Parallel([&](int r) {
            double e = cases[r]->Check(true);
            printf("TP rank=%d rows=%d L2=%.8g PASS\n", r, cases[r]->rows, e);
        });
        Parallel([&](int r) {
            auto &c = *cases[r];
            c.Reset();
            Require(FastllmCudaNaiveDSALayer(c.input, r), "TP graph warmup");
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Require(FastllmCudaNaiveDSALayer(c.input, r), "TP graph capture rejected");
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            // Replay the same MX verification graph across short, boundary and long contexts.
            const std::vector<int> lengths = fullMx
                ? std::vector<int>{121, 2045, 8192, 16384, 18940, 24576, 32760, 49152, 65528}
                : testFp8KV && shape == std::vector<int>{8,8} ? std::vector<int>{121,2045,8192,16384,24576,32760} : std::vector<int>{8192, 9216, 10240};
            for (size_t replay = 0; replay < lengths.size(); ++replay) {
                c.Lengths(lengths[replay], testFp8KV && shape == std::vector<int>{8,8} && replay == 2);
                c.Rebind(replay % 2);
                c.Reset();
                c.Legacy(true);
                Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
                Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                c.Check(true, fullMx && lengths[replay] > 32768);
            }
            if (fullMx) {
                // Force cutoff ties at 64K so both independently projected paths select
                // the same keys. Check the original unfused end-to-end residual as well.
                Cuda(cudaMemsetAsync(c.iwWeight.cudaData, 0, c.iwWeight.Count(0) * sizeof(B), cudaStreamPerThread));
                c.Reset();
                c.Legacy(true);
                Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
                Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                c.Check(true);
            }
            Cuda(cudaGraphExecDestroy(executable));
            Cuda(cudaGraphDestroy(graph));
        });
        Parallel([&](int r) { cases[r].reset(); });
    }
    FastllmCudaCustomAllReduceReset();
}
void CheckRouter(const Case &c, const Data &weight, const Data &router) {
    auto h = Read(c.normed), w = Read(weight), y = Read(router);
    double worst = 0;
    for (int row = 0; row < c.rows; ++row) {
        double error = 0, magnitude = 0;
        for (int expert = 0; expert < 256; ++expert) {
            double sum = 0;
            for (int k = 0; k < 4096; ++k) {
                B v; float a;
                std::memcpy(&v, h.data() + (row * 4096 + k) * 2, 2);
                std::memcpy(&a, w.data() + (expert * 4096 + k) * 4, 4);
                sum += float(v) * double(a);
            }
            float actual;
            std::memcpy(&actual, y.data() + (row * 256 + expert) * 4, 4);
            const double reference = 1 / (1 + std::exp(-sum));
            Require(std::isfinite(actual), "non-finite router probability");
            error += (actual - reference) * (actual - reference);
            magnitude += reference * reference;
        }
        worst = std::max(worst, std::sqrt(error / magnitude));
    }
    Require(worst < 2e-6, "router FP32 reference L2");
    printf("ROUTER rank=%d rows=%d L2=%.8g PASS\n", c.device, c.rows, worst);
}
void TpRouter(bool fp8) {
    const std::vector<int> devices{0, 1, 2, 3, 4, 5, 6, 7};
    Require(FastllmInitNccl(devices) && FastllmCudaCustomAllReduceInit(devices), "router TP init");
    for (auto shape : std::vector<std::vector<int>>{{4}, {8}, {4, 4}, {8, 8}}) {
        std::array<std::unique_ptr<Case>, 8> cases;
        std::array<Data, 8> weights, arenas, routers, aliases;
        constexpr int pages = 32768 / 128;
        Parallel([&](int rank) {
            cases[rank] = std::make_unique<Case>(shape, 32768, true, fp8, rank);
            auto &c = *cases[rank];
            Init(weights[rank], FLOAT32, {256, 4096});
            Fill(weights[rank], 731, .02f);
            Init(arenas[rank], INT32, {pages + c.rows * 256 + 16});
            Cuda(cudaMemset(arenas[rank].cudaData, 0x5a, arenas[rank].GetBytes()));
            Cuda(cudaMemcpy(arenas[rank].cudaData, c.caches[0]->kp.cudaData,
                            pages * sizeof(int), cudaMemcpyDeviceToDevice));
            // The router follows a real page table in one allocation. Logical
            // KV tensor bytes would falsely overlap this otherwise valid view.
            c.caches[0]->keyView.cudaData = arenas[rank].cudaData;
            auto bind = [&](Data &d, size_t offset) {
                d.Resize({1, c.rows, 256});
                d.FakeFrom(arenas[rank], offset);
                d.dataType = FLOAT32; d.UpdateUnitSize();
            };
            bind(routers[rank], pages * sizeof(int));
            bind(aliases[rank], (pages - 4) * sizeof(int));
            c.input.routerWeight = &weights[rank];
            c.input.router = &routers[rank];
        });
        Parallel([&](int rank) {
            FastllmCudaSetLinearExactBatchThreshold(17);
            auto &c = *cases[rank];
            auto hidden = Read(c.hidden), arena = Read(arenas[rank]);
            auto bad = c.input;
            if (rank == 3) bad.router = &aliases[rank];
            Require(!FastllmCudaNaiveDSALayer(bad, rank), "real page-table alias admitted");
            Require(hidden == Read(c.hidden) && arena == Read(arenas[rank]), "router fallback wrote data");
        });
        Parallel([&](int rank) {
            FastllmCudaSetLinearExactBatchThreshold(17);
            auto &c = *cases[rank];
            c.Legacy(true);
            Require(FastllmCudaNaiveDSALayer(c.input, rank), "paged router launch rejected");
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            c.Check(true); CheckRouter(c, weights[rank], routers[rank]);
            cudaGraph_t graph; cudaGraphExec_t executable;
            c.Reset();
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Require(FastllmCudaNaiveDSALayer(c.input, rank), "paged router capture rejected");
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 2; ++replay) {
                c.Lengths(8192 + replay * 1024, false);
                c.Reset(); c.Legacy(true);
                Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
                Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                c.Check(true); CheckRouter(c, weights[rank], routers[rank]);
            }
            Cuda(cudaGraphExecDestroy(executable)); Cuda(cudaGraphDestroy(graph));
            auto arena = Read(arenas[rank]), pageTable = Read(c.caches[0]->kp);
            Require(std::equal(pageTable.begin(), pageTable.end(), arena.begin()), "router overwrote page table");
            Require(std::all_of(arena.end() - 64, arena.end(), [](unsigned char v) { return v == 0x5a; }),
                    "router output guard");
        });
        Parallel([&](int rank) { cases[rank].reset(); });
    }
    FastllmCudaCustomAllReduceReset();
}
void Concurrent(bool fp8) {
    std::array<std::exception_ptr, 2> errors{};
    std::vector<std::thread> workers;
    for (int i = 0; i < 2; ++i)
        workers.emplace_back([&, i] {
            try {
                FastllmCudaSetDevice(0);
                Case c(i ? std::vector<int>{8, 8} : std::vector<int>{3, 6}, 32768, true, fp8);
                c.Legacy();
                Cuda(FastllmCudaNaiveDSALayerLaunch(c.args));
                c.Check();
                cudaGraph_t graph;
                cudaGraphExec_t executable;
                Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                Cuda(FastllmCudaNaiveDSALayerLaunch(c.args));
                Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                for (int repeat = 0; repeat < 8; ++repeat) {
                    c.Lengths(8192 + i * 1024 + repeat * 13, false);
                    c.Rebind(repeat % 2);
                    c.Reset();
                    c.Legacy();
                    Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
                    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                    c.Check();
                }
                Cuda(cudaGraphExecDestroy(executable));
                Cuda(cudaGraphDestroy(graph));
            } catch (...) {
                errors[i] = std::current_exception();
            }
        });
    for (auto &worker : workers)
        worker.join();
    for (auto &e : errors)
        if (e)
            std::rethrow_exception(e);
    puts("Independent stream graph replay PASS");
}
void Local(bool fp8, bool mixed) {
    cudaDeviceProp prop{};
    Cuda(cudaGetDeviceProperties(&prop, 0));
    std::vector<std::vector<int>> shapes =
        mixed ? std::vector<std::vector<int>>{{3, 5}, {8, 8}, {1, 8}, {7, 8}, {1, 2, 5, 8}} : std::vector<std::vector<int>>{{1}, {4}, {8}};
    for (bool paged : {false, true})
        for (auto shape : shapes) {
            printf("START paged=%d rows0=%d requests=%zu\n", paged, shape[0], shape.size());
            fflush(stdout);
            Case c(shape, 32768, paged, fp8);
            c.Legacy();
            Cuda(cudaDeviceSynchronize());
            puts("REFERENCE READY");
            fflush(stdout);
            Cuda(FastllmCudaNaiveDSALayerLaunch(c.args));
            Cuda(cudaDeviceSynchronize());
            c.Check();
            c.Reject();
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Cuda(FastllmCudaNaiveDSALayerLaunch(c.args));
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            int replay = 0;
            // Exercise both sides of the one-tile-per-CTA histogram reuse guard.
            const int oneTilePast = prop.multiProcessorCount * 128 - shape[0];
            std::vector<int> pasts{121, 2044, 8192, 16384, oneTilePast, oneTilePast + 1, 32760};
            if (c.rows > 8 && shape.size() == 2) {
                const int requestTilePast = (prop.multiProcessorCount / 2) * 128 - shape[0];
                for (int boundary : {requestTilePast, requestTilePast + 1, 16384 - shape[0], 16385 - shape[0]})
                    pasts.push_back(boundary);
            }
            for (int past : pasts) {
                c.Lengths(past, replay == 2);
                c.Rebind(replay % 2);
                c.Reset();
                c.Legacy();
                Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
                Cuda(cudaDeviceSynchronize());
                double error = c.Check();
                printf("LAYER fp8=%d paged=%d rows=%d replay=%d residual_l2=%.8g PASS\n", fp8, paged, c.rows,
                       replay++, error);
            }
            Cuda(cudaGraphExecDestroy(executable));
            Cuda(cudaGraphDestroy(graph));
        }
    if (mixed) {
        // Request 0 has tied zero Indexer logits at the compact-selection
        // boundary, while request 1 uses a different selection branch.
        Case c({8, 8}, 32768, true, fp8);
        Cuda(cudaMemset(c.initial.cudaData, 0, 8 * 4096 * sizeof(B)));
        Cuda(cudaMemset(c.branch.cudaData, 0, 8 * 4096 * sizeof(B)));
        for (int past : {16376, 16377, 8192}) {
            c.Lengths(past, false);
            c.Reset();
            c.Legacy();
            Cuda(FastllmCudaNaiveDSALayerLaunch(c.args));
            Cuda(cudaDeviceSynchronize());
            c.Check();
        }
        puts("Independent request Top-K branches and 16K fallback PASS");
    }
    if (!mixed) {
        Case c({4}, 65536, true, fp8);
        c.Lengths(65532, false);
        c.Reset();
        c.Legacy();
        Cuda(FastllmCudaNaiveDSALayerLaunch(c.args));
        Cuda(cudaDeviceSynchronize());
        c.Check();
        puts("64K page boundary PASS");
    }
}
#include "naive_n05_dp_dsa_test.cuh"
} // namespace
int main(int argc, char **argv) {
    try {
        FastllmCudaSetDevice(0);
        if (!FastllmCudaNaiveDSALayerSupported())
            return 77;
        if (argc > 1 && std::strcmp(argv[1], "dp-kv") == 0) {
            const int bits = argc > 2 ? std::stoi(argv[2]) : 8;
            DpDsaVerify(8, true, bits);
            BatchedDpDsa(bits == 8 ? 2 : 5, false, bits);
            if (bits == 8) BatchedDpDsa(3, false, bits);
            BatchedDpDsa(2, false, bits, {4,4});
            BatchedDpDsa(4, false, bits, {1,3,7,5});
            return 0;
        }
        if (argc > 1 && std::strcmp(argv[1], "dp-batch") == 0) {
            BatchedDpDsa(argc > 2 ? std::stoi(argv[2]) : 5, argc == 3);
            return 0;
        }
        if (argc > 1 && std::strcmp(argv[1], "dp") == 0) {
            DpDsaVerify(8, true);
            DpDsaVerify(8, true, 8);
            DpDsaVerify(8, true, 4);
            DpDsaVerify(3, false);
            BatchedDpDsa(5, false);
            BatchedDpDsa(5, false, 4);
            BatchedDpDsa(3, false, 16, {2,6,8});
            BatchedDpDsa(3, false, 4, {5,1,3});
            return 0;
        }
        if (argc > 2 && (std::strcmp(argv[2], "tp") == 0 || std::strcmp(argv[2], "router") == 0)) {
            int count = 0;
            Cuda(cudaGetDeviceCount(&count));
            if (count < 8)
                return 77;
            for (int r = 0; r < 8; ++r) {
                FastllmCudaSetDevice(r);
                if (!FastllmCudaNaiveDSALayerSupported())
                    return 77;
            }
            FastllmCudaSetDevice(0);
        }
        testMxProjection = argc > 3 && std::strcmp(argv[3],"--mxfp8")==0;
        testFp8KV = argc > 1 && std::strcmp(argv[1], "fp8-kv") == 0;
        bool fp8 = argc > 1 && std::strcmp(argv[1], "fp8") == 0;
        if (argc > 2 && std::strcmp(argv[2], "tp") == 0)
            Tp(fp8, testMxProjection);
        else if (argc > 2 && std::strcmp(argv[2], "router") == 0)
            TpRouter(fp8);
        else if (argc > 2 && std::strcmp(argv[2], "streams") == 0)
            Concurrent(fp8);
        else
            Local(fp8, argc > 2 && std::strcmp(argv[2], "mixed") == 0);
        puts("DSA LAYER PASS");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
