#include "fastllm.h"
#include "../../src/models/naive_n05_indexer_tp.h"
#include "devices/cuda/naive-n05-indexer-attention.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static int checks=0;
static void Require(bool ok,const char *message) { if (!ok) throw std::runtime_error(message); }
static float FromBits(uint16_t x) { return BFloat16BitsToFloat32(x); }
static void Upload(Data &d, const std::vector<int> &shape, int seed) {
    d.Resize(shape); d.Allocate();
    unsigned x = seed;
    for (uint64_t i = 0; i < d.Count(0); ++i) {
        x = 1664525u * x + 1013904223u;
        float v = ((int)(x >> 16) - 32768) / 32768.0f;
        if (d.dataType == BFLOAT16) ((uint16_t *)d.cpuData)[i] = Float32ToBFloat16RNEBits(v);
        else ((float *)d.cpuData)[i] = v;
    }
    d.ToDevice(DataDevice::CUDA, {0}, true);
}
template<class T> static std::vector<T> Read(const Data &d) {
    size_t count=1;for(int dim:d.dims)count*=dim;
    std::vector<T> v(count);
    Require(cudaMemcpy(v.data(), d.cudaData, count*sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess, "read failed");
    return v;
}
static void TestAttentionGroupedScores() {
    struct Shape { int queries, heads, kvHeads, dim, valueDim, keys, selected; };
    for (auto s : {Shape{32,8,1,192,128,513,0}, Shape{33,16,2,192,128,2113,2051},
                   Shape{33,16,2,192,128,2113,2047},
                   Shape{32,16,1,192,128,2305,2048}, Shape{33,32,2,192,128,2305,2047},
                   Shape{257,16,1,192,128,2305,2048}, Shape{65,64,4,192,128,2305,2048},
                   Shape{257,8,1,192,128,2305,2048}, Shape{31,8,1,192,128,513,512}})
    for (bool causal : {false, true}) for (bool withSink : {false, true}) {
        Data query(BFLOAT16),key(BFLOAT16),value(BFLOAT16),sink(FLOAT32),indices(INT32),output;
        int keyStride = s.kvHeads * s.dim + 128, past = s.keys - s.queries;
        Upload(query,{1,s.queries,s.heads*s.dim},53);
        Upload(key,{1,s.keys,keyStride},71);
        Upload(value,{1,s.keys,s.kvHeads*s.valueDim},113);
        if (withSink) Upload(sink,{s.heads},171);
        if (s.selected) {
            indices.Resize({s.queries,s.selected}); indices.Allocate();
            auto *p = (int *)indices.cpuData;
            for (int q=0;q<s.queries;++q) for (int slot=0;slot<s.selected;++slot) {
                int k = (slot*137+q*13)%s.keys;
                if (slot%97==0 || (q==0 && s.queries>1)) k=-1;
                else if (slot%193==1) k=s.keys+3;
                p[q*s.selected+slot]=k;
            }
            indices.ToDevice(DataDevice::CUDA,{0},true);
        }
        FastllmCudaNaiveAttention(query,key,value,indices,sink,s.heads,s.kvHeads,s.dim,
                                  s.valueDim,past,0,output,causal);
        auto actual=Read<uint16_t>(output),qbits=Read<uint16_t>(query),
             kbits=Read<uint16_t>(key),vbits=Read<uint16_t>(value);
        auto copyHead = [](Data &dst, const std::vector<uint16_t> &src,
                           int rows, int stride, int offset, int width) {
            dst.Resize({1,rows,width}); dst.Allocate();
            for (int r=0;r<rows;++r)
                std::memcpy((uint16_t *)dst.cpuData+r*width,src.data()+r*stride+offset,width*sizeof(uint16_t));
            dst.ToDevice(DataDevice::CUDA,{0},true);
        };
        for (int h=0;h<s.heads;++h) {
            Data q(BFLOAT16),k(BFLOAT16),v(BFLOAT16),headSink(FLOAT32),reference;
            int kvHead=h/(s.heads/s.kvHeads);
            copyHead(q,qbits,s.queries,s.heads*s.dim,h*s.dim,s.dim);
            copyHead(k,kbits,s.keys,keyStride,kvHead*s.dim,s.dim);
            copyHead(v,vbits,s.keys,s.kvHeads*s.valueDim,kvHead*s.valueDim,s.valueDim);
            if (withSink) {
                headSink.Resize({1}); headSink.Allocate();
                Require(cudaMemcpy(headSink.cpuData,(float *)sink.cudaData+h,sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess,
                        "copy head sink failed");
                headSink.ToDevice(DataDevice::CUDA,{0},true);
            }
            FastllmCudaNaiveAttention(q,k,v,indices,headSink,1,1,s.dim,s.valueDim,past,0,reference,causal);
            auto expected=Read<uint16_t>(reference);
            // MMA changes the sum order; the Hopper path additionally rounds
            // unnormalized online probabilities. Compare against the existing
            // isolated-head implementation with a bounded BF16 error contract.
            double error=0,norm=0,maxError=0,maxReference=0;
            for (int r=0;r<s.queries;++r) for (int d=0;d<s.valueDim;++d) {
                double got=FromBits(actual[(r*s.heads+h)*s.valueDim+d]);
                double ref=FromBits(expected[r*s.valueDim+d]);
                if (s.queries<32 || (s.selected?s.selected:s.keys)>2048)
                    Require(actual[(r*s.heads+h)*s.valueDim+d]==expected[r*s.valueDim+d],
                            "SIMT fallback differs from isolated head");
                Require(std::isfinite(got),"Grouped Attention produced nonfinite output");
                error+=(got-ref)*(got-ref);norm+=ref*ref;
                maxError=std::max(maxError,std::abs(got-ref));
                maxReference=std::max(maxReference,std::abs(ref));
            }
            Require(std::sqrt(error/std::max(norm,1e-30))<=.01 &&
                    maxError<=.005+.01*maxReference,
                    "Grouped Attention exceeds isolated-head error bound");
        }
        ++checks;
    }
}

static void TestAttentionSwa() {
    for (bool fp8 : {false, true}) {
    struct Shape {
    int queries, past, padding, mode;
    float amplitude;
    int heads = 64, kvHeads = 8;
    };
    for (auto s : {Shape{1, 127, 0, 1, 1}, Shape{257, 7935, 0, 1, 1, 8, 1}, Shape{257, 7935, 0, 0, 1, 8, 1},
                   Shape{1024, 127, 0, 1, 1, 8, 1}, Shape{471, 127, 0, 2, 1}, Shape{512, 127, 0, 3, 1},
                   Shape{33, 127, 0, 1, 4}, Shape{65, 0, 16, 1, 1},
                   Shape{2049, 127, 0, 1, 1}}) {
    const int heads = s.heads, kvHeads = s.kvHeads;
    constexpr int dim = 192, valueDim = 128;
    int keys = s.past + s.queries, stride = kvHeads * dim + s.padding;
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), sink(FLOAT32), indices, out;
    Upload(q, {1, s.queries, heads * dim}, 53);
    Upload(k, {1, keys, stride}, 71);
    Upload(v, {1, keys, kvHeads * valueDim}, 113);
    auto qb = Read<uint16_t>(q), kb = Read<uint16_t>(k), vb = Read<uint16_t>(v);
    for (auto &x : qb)
        x = Float32ToBFloat16RNEBits(s.mode == 3 ? 0 : FromBits(x) * s.amplitude);
    for (auto &x : kb)
        x = Float32ToBFloat16RNEBits(s.mode == 3 ? 0 : FromBits(x) * s.amplitude);
    if (s.mode == 3)
        for (auto &x : vb)
            x = Float32ToBFloat16RNEBits(.375f);
    Data k8(FP8_E4M3), v8(FP8_E4M3);
    if (fp8) {
        auto encode = [&](Data &dst, const std::vector<int> &shape, std::vector<uint16_t> &values) {
            std::vector<uint8_t> bytes(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                __nv_fp8_e4m3 x(FromBits(values[i]));
                bytes[i] = x.__x;
                values[i] = Float32ToBFloat16RNEBits((float)x);
            }
            dst.Resize(shape);
            dst.ToDevice(DataDevice::CUDA, {0}, false);
            dst.Allocate(false);
            Require(cudaMemcpy(dst.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) == cudaSuccess,
                    "FP8 SWA upload");
        };
        encode(k8, k.dims, kb);
        encode(v8, v.dims, vb);
    }
    Require(cudaMemcpy(q.cudaData, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess,
            "SWA query upload");
    Require(cudaMemcpy(k.cudaData, kb.data(), kb.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess, "SWA key upload");
    Require(cudaMemcpy(v.cudaData, vb.data(), vb.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess,
            "SWA value upload");
    std::vector<float> bias(heads);
    if (s.mode) {
        sink.Resize({heads});
        sink.Allocate();
        for (int h = 0; h < heads; ++h)
            bias[h] = s.mode == 2 ? (h % 3 - 1) * 80.f : (h % 5 - 2) * 1.3f;
        std::memcpy(sink.cpuData, bias.data(), heads * sizeof(float));
        sink.ToDevice(DataDevice::CUDA, {0}, true);
    }
    FastllmCudaNaiveAttention(q, fp8 ? k8 : k, fp8 ? v8 : v, indices, sink, heads, kvHeads, dim, valueDim, s.past, 128,
                              out, true);
    auto actual = Read<uint16_t>(out);
    for (auto x : actual)
        Require(std::isfinite(FromBits(x)), "SWA nonfinite output");
    double error = 0, norm = 0;
    std::vector<int> rows = {0, s.queries / 2, s.queries - 1};
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (int row : rows)
        for (int h : {0, heads / 2, heads - 1}) {
            int end = s.past + row + 1, start = std::max(0, end - 128);
            std::vector<double> scores(end - start);
            double maximum = s.mode ? bias[h] : -INFINITY;
            for (int z = start; z < end; ++z) {
            double dot = 0;
            for (int d = 0; d < dim; ++d)
                dot += double(FromBits(qb[((size_t)row * heads + h) * dim + d])) *
                       FromBits(kb[(size_t)z * stride + h / (heads / kvHeads) * dim + d]);
            if (fp8) {
                float rounded = FromBits(Float32ToBFloat16RNEBits((float)dot));
                scores[z - start] = FromBits(Float32ToBFloat16RNEBits(rounded * (1.f / std::sqrt(float(dim)))));
            } else
                scores[z - start] = dot / std::sqrt(double(dim));
            maximum = std::max(maximum, scores[z - start]);
            }
            double denominator = s.mode ? std::exp(bias[h] - maximum) : 0;
            for (auto &x : scores) {
            x = std::exp(x - maximum);
            denominator += x;
            }
            for (int d : {0, 31, 64, 127}) {
            double expected = 0;
            for (int z = start; z < end; ++z) {
                double weight = scores[z - start] / denominator;
                if (fp8)
                    weight = FromBits(Float32ToBFloat16RNEBits((float)weight));
                expected += weight * FromBits(vb[((size_t)z * kvHeads + h / (heads / kvHeads)) * valueDim + d]);
            }
            double got = FromBits(actual[((size_t)row * heads + h) * valueDim + d]);
            Require(std::abs(got - expected) <= .012, "SWA differs from independent reference");
            error += (got - expected) * (got - expected);
            norm += expected * expected;
            }
        }
    Require(std::sqrt(error / std::max(norm, 1e-30)) <= .01, "SWA relative error exceeds 1 percent");
    ++checks;
    }
    }
}

// Query sharding must preserve every selected index, including causal padding
// and stable tie ordering. Borrowed output slices model the in-place AllGather.
static void TestIndexerPrefillShards() {
    for (auto shape : {std::vector<int>{1024,4096,2}, {4096,8192,8}}) {
        const int rows=shape[0], keys=shape[1], ranks=shape[2], local=rows/ranks;
        constexpr int heads=16, dim=128, topK=2048, stride=192+128;
        Data q(BFLOAT16),w(BFLOAT16),k(BFLOAT16),reference,actual(INT32);
        Upload(q,{1,rows,heads*dim},53); Upload(w,{1,rows,heads},71);
        Upload(k,{1,keys,stride},113);
        FastllmCudaNaiveIndexer(q,w,k,heads,dim,keys-rows,topK,true,reference);
        auto expected=Read<int>(reference);
        actual.Resize({rows,topK}); actual.ToDevice(DataDevice::CUDA,{0},false); actual.Allocate(false);
        for (int rank=0;rank<ranks;++rank) {
            int first=rank*local;
            Data qr,wr,ir;
            qr.Resize({1,local,heads*dim}); qr.FakeFrom(q,(size_t)first*heads*dim*q.unitSize);
            wr.Resize({1,local,heads}); wr.FakeFrom(w,(size_t)first*heads*w.unitSize);
            ir.Resize({local,topK}); ir.FakeFrom(actual,(size_t)first*topK*sizeof(int));
            FastllmCudaNaiveIndexer(qr,wr,k,heads,dim,keys-rows+first,topK,true,ir);
        }
        Require(Read<int>(actual)==expected,"sharded prefill Indexer differs from replicated queries");
        ++checks;
    }
}

static void TestIndexerRaggedShards() {
    for (int rows : {513, 1025, 4144}) for (int ranks : {2, 4, 8}) {
        if (rows < ranks * 128) continue;
        constexpr int past = 8192, heads = 16, dim = 128, topK = 2048;
        Data q(BFLOAT16), w(BFLOAT16), key(BFLOAT16), full, gathered, compact;
        Upload(q, {1, rows, heads * dim}, 53);
        Upload(w, {1, rows, heads}, 71);
        Upload(key, {1, past + rows, dim}, 113);
        for (bool tied : {false, true}) {
            if (tied) Require(cudaMemset(w.cudaData, 0, w.GetBytes()) == cudaSuccess, "zero weights failed");
            FastllmCudaNaiveIndexer(q, w, key, heads, dim, past, topK, true, full);
            for (int rank = 0; rank < ranks; ++rank)
                ScoreRaggedIndexer(q, w, key, past, topK, true, rank, ranks, gathered);
            CompactRaggedIndexer(gathered, rows, ranks, compact);
            Require(Read<int>(full) == Read<int>(compact), "Ragged Indexer changed exact TopK ordering");
            ++checks;
        }
    }
}

// Compare scratch-backed FP8 expansion with the direct attention path.
// Insufficient workspace must retain the same result without resizing storage.
static void TestSwaPrefillScratch() {
    for (auto shape : {std::vector<int>{1025, 127, 16}, {4096, 65536, 64},
                       {1025, 64, 16}, {4096, 0, 64}, {8193, 193, 16}, {1027, 0, 16, 0}, {1153, 191, 16, 0}}) {
        const int rows=shape[0], past=shape[1], heads=shape[2], kvHeads=heads/8, keys=past+rows;
        const int padding = shape.size() == 4 ? shape[3] : 16;
        const int keyStride=kvHeads*192+padding, valueStride=kvHeads*128;
        Data q(BFLOAT16), key(FP8_E4M3), value(FP8_E4M3), sink(FLOAT32), indices(INT32), reference, actual;
        Upload(q,{1,rows,heads*192},53); Upload(sink,{heads},71);
        auto uploadFp8 = [&](Data &data, int width, unsigned seed) {
            data.Resize({1,keys,width}); data.Allocate();
            for (uint64_t i=0;i<data.Count(0);++i) {
                seed=seed*1664525u+1013904223u;
                data.cpuData[i]=__nv_fp8_e4m3(((int)(seed>>16)-32768)/16384.f).__x;
            }
            data.ToDevice(DataDevice::CUDA,{0},true);
        };
        uploadFp8(key,keyStride,113); uploadFp8(value,valueStride,139);
        FastllmCudaNaiveAttention(q,key,value,indices,sink,heads,kvHeads,192,128,past,128,reference);
        const auto expected=Read<uint16_t>(reference);
        for (auto type : {FLOAT32, BFLOAT16}) {
            const size_t bytes=(size_t)(rows+std::min(past,127))*(keyStride+valueStride)*2;
            const size_t elements=bytes/(type==FLOAT32?4:2);
            Data storage(type,{(int)elements+32}), scratch(type,{(int)elements});
            storage.ToDevice(DataDevice::CUDA,{0},false); storage.Allocate(false);
            // A per-request slice must not overwrite neighboring requests.
            cudaMemset(storage.cudaData,0xa5,storage.GetBytes());
            scratch.FakeFrom(storage,16*storage.unitSize);
            const auto pointer=scratch.cudaData;
            for (bool full : {true,false}) {
                scratch.Resize({(int)(full?elements:elements-1)});
                FastllmCudaNaiveAttention(q,key,value,indices,sink,heads,kvHeads,192,128,past,128,actual,true,&scratch);
                const auto got = Read<uint16_t>(actual);
                if (full && (rows >= 8192 || (padding == 0 && FastllmCudaHasBlackwellIndexerAttention()))) {
                    double norm = 0, error = 0, maxAbs = 0;
                    for (size_t i = 0; i < got.size(); ++i) {
                        const double value = FromBits(expected[i]), delta = FromBits(got[i]) - value;
                        Require(std::isfinite(delta), "nonfinite staged SWA output");
                        norm += value * value;
                        error += delta * delta;
                        maxAbs = std::max(maxAbs, std::abs(delta));
                    }
                    double l2 = std::sqrt(error / std::max(norm, 1e-30));
                    std::printf("Long staged SWA: relL2 %.6f maxAbs %.6f\n", l2, maxAbs);
                    Require(l2 < .008 && maxAbs < .02, "long staged SWA exceeded BF16 error budget");
                } else {
                    Require(got == expected, "SWA prefill scratch changed BF16 output bits");
                }
                Require(scratch.cudaData==pointer && scratch.Count(0)==(full?elements:elements-1),"SWA scratch was resized");
                ++checks;
            }
            std::vector<unsigned char> copy(storage.GetBytes());
            cudaMemcpy(copy.data(),storage.cudaData,copy.size(),cudaMemcpyDeviceToHost);
            const size_t guard=16*storage.unitSize;
            for (size_t i=0;i<guard;++i)
                Require(copy[i]==0xa5 && copy[guard+bytes+i]==0xa5,"SWA scratch crossed request boundary");
        }
    }
}

static void TestLocalPrefillBatch() {
    for (auto compute : {NaiveIndexerCompute::Auto, NaiveIndexerCompute::FP8, NaiveIndexerCompute::FP8Direct}) {
        struct Item { Data q{BFLOAT16}, w{BFLOAT16}, k{BFLOAT16}, packed, reference, actual; };
        std::vector<std::unique_ptr<Item>> storage;
        std::vector<FastllmNaivePrefillIndexerItem> items;
        // Include a short fallback, ragged score sizes requiring aligned FP8 refinement
        // scratch, >8 items, and bounded long-context tiles.
        for (int i = 0; i < 10; ++i) {
            auto item = std::make_unique<Item>();
            const int rows = i == 0 ? 1 : i == 9 ? 1153 : 65 + i * 17;
            const int keys = i == 9 ? 131075 : 8193 + i * 131;
            Upload(item->q, {1, rows, 2048}, 53 + i);
            Upload(item->w, {1, rows, 16}, 71 + i);
            Upload(item->k, {1, keys, 128}, 113 + i);
            const bool packKey = compute != NaiveIndexerCompute::Auto || i % 2;
            if (packKey) FastllmCudaNaivePackIndexer(item->k, item->packed);
            Data &key = packKey ? item->packed : item->k;
            FastllmCudaNaiveIndexer(item->q, item->w, key, 16, 128, keys-rows, 2048, true,
                item->reference, false, nullptr, compute);
            items.push_back({&item->q, &item->w, &key, &item->actual, keys-rows});
            storage.push_back(std::move(item));
        }
        FastllmCudaNaiveBatchIndexer(items, 16, 128, 2048, true, nullptr, compute);
        for (const auto &item : storage)
            Require(Read<int>(item->reference) == Read<int>(item->actual), "DP Indexer batch changed exact indices");
        ++checks;
    }
    for (bool fp8 : {false, true}) {
        struct Item {
            Data q{BFLOAT16}, k{BFLOAT16}, v{BFLOAT16}, ids{INT32}, reference, actual;
            PagedCacheManager keyPool, valuePool;
            Data keyTable{INT32}, valueTable{INT32}, pagedKey{FP8_E4M3}, pagedValue{FP8_E4M3};
        };
        std::vector<std::unique_ptr<Item>> storage;
        std::vector<FastllmNaivePrefillAttentionItem> items;
        const int heads = fp8 ? 128 : 32, kvHeads = fp8 ? 4 : 2;
        Data sink(FLOAT32); Upload(sink, {heads}, 171);
        for (int i = 0; i < 4; ++i) {
            auto item = std::make_unique<Item>();
            const int rows = i == 0 ? 1 : fp8 ? 2110 + i : 257 + i * 137, keys = 65536 + rows;
            Upload(item->q, {1, rows, heads*192}, 53+i);
            auto fill = [&](Data &data, int width, unsigned seed) {
                if (!fp8) { Upload(data, {1, keys, width}, seed); return; }
                data.dataType = FP8_E4M3; data.UpdateUnitSize();
                data.Resize({1, keys, width}); data.Allocate();
                for (uint64_t j=0;j<data.Count(0);++j) {
                    seed=seed*1664525u+1013904223u;
                    data.cpuData[j]=__nv_fp8_e4m3(((int)(seed>>16)-32768)/32768.f).__x;
                }
                data.ToDevice(DataDevice::CUDA,{0},true);
            };
            fill(item->k, kvHeads*192, 71+i); fill(item->v, kvHeads*128, 113+i);
            auto &ids=item->ids; ids.Resize({rows,2048}); ids.Allocate();
            for (int row=0;row<rows;++row) for(int j=0;j<2048;++j)
                ((int *)ids.cpuData)[size_t(row)*2048+j] = row==0 || j%97==0 ? -1 : (j*137+row*13)%keys;
            ids.ToDevice(DataDevice::CUDA,{0},true);
            auto page = [&](const Data &src, Data &view, Data &table, PagedCacheManager &pool) {
                const int pages = (keys + 127) / 128, width = src.dims[2];
                pool.dataType = FP8_E4M3; pool.UpdateUnitSize(); pool.pageLen = 128;
                pool.Resize({pages,128,1,width}); pool.ToDevice(DataDevice::CUDA,{0},false);
                pool.Allocate(false); pool.SetMaxPages(pages);
                table.Resize({pages}); table.Allocate();
                for (int j=0;j<pages;++j) ((int *)table.cpuData)[j]=pages-1-j;
                table.ToDevice(DataDevice::CUDA,{0},true);
                for (int first=0;first<keys;first+=128)
                    Require(cudaMemcpyAsync((uint8_t *)pool.cudaData+size_t(pages-1-first/128)*128*width,
                        (const uint8_t *)src.cudaData+size_t(first)*width,size_t(std::min(128,keys-first))*width,
                        cudaMemcpyDeviceToDevice,cudaStreamPerThread)==cudaSuccess,"paged Attention setup");
                view.FakeFrom(table,0); view.dataType=FP8_E4M3; view.UpdateUnitSize(); view.Resize(src.dims);
                view.isPagedKVCache=true; view.pagedKVCacheData=&pool; view.pageLen=128;
                view.expansionDims={1,pages*128,width};
            };
            if (fp8 && i%2) {
                page(item->k,item->pagedKey,item->keyTable,item->keyPool);
                page(item->v,item->pagedValue,item->valueTable,item->valuePool);
            }
            Data &key=fp8 && i%2?item->pagedKey:item->k, &value=fp8 && i%2?item->pagedValue:item->v;
            FastllmCudaNaiveAttention(item->q,key,value,ids,sink,heads,kvHeads,192,128,keys-rows,0,item->reference);
            items.push_back({&item->q,&key,&value,&ids,&item->actual,keys-rows});
            storage.push_back(std::move(item));
        }
        FastllmCudaNaiveBatchAttention(items, sink, heads, kvHeads, 192, 128);
        for (const auto &item : storage)
            Require(Read<uint16_t>(item->reference) == Read<uint16_t>(item->actual), "DP Attention batch changed BF16 output bits");
        ++checks;
    }
}

// Exercise the Indexer contract separately from arbitrary index lists. Tail
// rows, empty/future selections, and shuffled KV pages must survive packing.
static void TestBlackwellIndexerAttention() {
    for (int keys : {1181, 2305, 6144, 8192, 16384, 32768, 65536, 65543, 98305, 131072, 262209}) {
        const bool paged = keys != 1181 && keys != 32768;
        const int rows = keys > 2305 ? 149 : (paged ? 2051 : 149);
        const int kvHeads = paged ? 2 : 1, heads = 16 * kvHeads, topK = keys > 2305 ? 2048 : 512, pageLen = 64;
        Data query(BFLOAT16), key(BFLOAT16), value(BFLOAT16), indices(INT32), sink, actual, reference;
        Upload(query, {1, rows, heads * 192}, 31);
        Data kTable(INT32), vTable(INT32);
        PagedCacheManager kPool, vPool;
        auto fillCache = [&](Data &cache, Data &table, PagedCacheManager &pool, int width, int seed) {
            if (!paged) {
                Upload(cache, {1, keys, width}, seed);
                return;
            }
            int pages = (keys + pageLen - 1) / pageLen;
            pool.dataType = FP8_E4M3;
            pool.UpdateUnitSize();
            pool.pageLen = pageLen;
            pool.Resize({pages, pageLen, 1, width});
            pool.Allocate();
            unsigned x = seed;
            for (size_t i = 0; i < pool.Count(0); ++i) {
                x = x * 1664525u + 1013904223u;
                pool.cpuData[i] = __nv_fp8_e4m3((int(x >> 16) - 32768) / 32768.f).__x;
            }
            pool.ToDevice(DataDevice::CUDA, {0}, true);
            pool.SetMaxPages(pages);
            table.Resize({pages});
            table.Allocate();
            for (int i = 0; i < pages; ++i) ((int *)table.cpuData)[i] = (pages - 1 - i + seed) % pages;
            table.ToDevice(DataDevice::CUDA, {0}, true);
            cache.FakeFrom(table, 0);
            cache.dataType = FP8_E4M3;
            cache.UpdateUnitSize();
            cache.Resize({1, keys, width});
            cache.isPagedKVCache = true;
            cache.pagedKVCacheData = &pool;
            cache.pageLen = pageLen;
        };
        fillCache(key, kTable, kPool, kvHeads * 192 + 128, 71);
        fillCache(value, vTable, vPool, kvHeads * 128, 113);
        indices.Resize({rows, topK});
        indices.Allocate();
        for (int r = 0; r < rows; ++r) for (int j = 0; j < topK; ++j) {
            // Mix shared and dispersed selections in one FMHA launch, including
            // a partial last group and a causal span crossing the bitmap capacity.
            int pattern = keys >= 32768 && (r / 16) % 2 ? (r / 16) * 16 : r;
            int k = (j * 97 + pattern * 37) % keys;
            if (keys >= 6144 && (r / 16) % 3 == 0)
                k = keys - rows - topK + j;  // Dense island with a nonzero physical tile offset.
            int id = k;
            if (r % 101 == 0) id = -1;  // Empty selection.
            else if (r % 11 == 0) id = j == 0 ? keys - rows + r : -1;
            else if (r % 13 == 0) id = k > 511 ? k : -1;
            else if (j % 37 == 0) id = -1;
            else if (j % 31 == 0) id = keys;  // Out of range.
            ((int *)indices.cpuData)[size_t(r) * topK + j] = id;
        }
        indices.ToDevice(DataDevice::CUDA, {0}, true);
        FastllmNaivePrefillAttentionItem item{&query, &key, &value, &indices, &actual, keys - rows};
        FastllmCudaNaiveAttention(query, key, value, indices, sink, heads, kvHeads,
            192, 128, item.past, 0, reference);
        // Use a bounded view inside a larger allocation to exercise chunking
        // and catch writes beyond the logical loan.
        constexpr int guard = 256;
        // Exercise full staging, head-striped staging and the cache reader.
        const int bytes = keys == 98305 ? 170 << 20 : keys == 32768 ? 100 << 20 :
            (keys == 65536 || keys == 65543) ? 52 << 20 : keys >= 6144 && keys <= 16384 ? 38 << 20 : 12 << 20;
        Data storage(BFLOAT16, {(bytes + 2 * guard) / 2}), scratch(BFLOAT16, {bytes / 2});
        storage.ToDevice(DataDevice::CUDA, {0}, false);
        storage.Allocate(false);
        Require(cudaMemset(storage.cudaData, 0xa5, bytes + 2 * guard) == cudaSuccess, "scratch guard setup");
        scratch.FakeFrom(storage, guard);
        if (FastllmCudaHasBlackwellIndexerAttention()) {
            Data tiny(BFLOAT16, {32});
            tiny.FakeFrom(storage, guard);
            Require(!FastllmCudaTryNaiveIndexerAttention(item, sink, heads, kvHeads, 192, 128, tiny),
                    "insufficient scratch must fall back");
            Require(!FastllmCudaTryNaiveIndexerAttention(item, sink, heads, kvHeads, 192, 128, query),
                    "overlapping scratch must fall back");
            Require(FastllmCudaTryNaiveIndexerAttention(item, sink, heads, kvHeads, 192, 128, scratch),
                    "supported Blackwell prefill did not use the fused kernel");
        } else {
            FastllmCudaNaiveBatchIndexerAttention({item}, sink, heads, kvHeads, 192, 128, scratch);
        }
        auto guards = Read<uint16_t>(storage);
        for (int i = 0; i < guard / 2; ++i)
            Require(guards[i] == 0xa5a5 && guards[(bytes + guard) / 2 + i] == 0xa5a5, "DSA scratch overrun");
        auto got = Read<uint16_t>(actual), expected = Read<uint16_t>(reference);
        double error = 0, norm = 0, maxAbs = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            double a = FromBits(got[i]), b = FromBits(expected[i]);
            Require(std::isfinite(a), "Indexer attention produced a nonfinite value");
            if (i / (heads * 128) % 101 == 0) Require(a == 0, "nonzero empty TopK row");
            error += (a - b) * (a - b);
            norm += b * b;
            maxAbs = std::max(maxAbs, std::abs(a - b));
        }
        double l2 = std::sqrt(error / std::max(norm, 1e-30));
        std::printf("Indexer prefill %s: relL2 %.6f, maxAbs %.6f\n", paged ? "paged FP8" : "BF16", l2, maxAbs);
        Require(l2 < .01 && maxAbs < .01, "Indexer attention exceeds BF16 error budget");
        ++checks;
    }
}

static void TestPackedPrefillRope() {
    constexpr int rows = 1025, heads = 16, kvHeads = 2;
    constexpr int widths[] = {heads * 192, kvHeads * 192, kvHeads * 128};
    constexpr int stride = widths[0] + widths[1] + widths[2];
    Data packed(BFLOAT16), positions(FLOAT32);
    Upload(packed, {1, rows, stride}, 43); Upload(positions, {rows}, 53);
    Data actual[3], expected[3];
    int offset = 0;
    for (int j = 0; j < 3; ++j) {
        for (Data *d : {actual + j, expected + j}) {
            d->dataType = BFLOAT16; d->UpdateUnitSize(); d->Resize({1, rows, widths[j]});
            d->ToDevice(DataDevice::CUDA, {0}, false); d->Allocate(false);
        }
        Require(cudaMemcpy2DAsync(expected[j].cudaData, widths[j] * 2,
            (const uint16_t *)packed.cudaData + offset, stride * 2, widths[j] * 2, rows,
            cudaMemcpyDeviceToDevice, cudaStreamPerThread) == cudaSuccess, "unpack RoPE reference");
        offset += widths[j];
    }
    FastllmCudaNaiveRopeQKScaleV(expected[0], expected[1], expected[2], positions,
        heads, kvHeads, 192, 128, 64, 300000.f, 1.543f);
    FastllmCudaNaiveRopeQKScaleV(actual[0], actual[1], actual[2], positions,
        heads, kvHeads, 192, 128, 64, 300000.f, 1.543f, &packed);
    for (int j = 0; j < 3; ++j)
        Require(Read<uint16_t>(actual[j]) == Read<uint16_t>(expected[j]), "packed RoPE changed BF16 output bits");
    ++checks;
}

static void TestColdWindowCacheWrite() {
    constexpr int rows = 1025, heads = 16, kvHeads = 2, past = 7;
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), key(FP8_E4M3), value(FP8_E4M3);
    Data selected, sink(FLOAT32), actual, expected;
    Upload(q, {1, rows, heads * 192}, 67);
    Upload(sink, {heads}, 47);
    auto write = [&](Data &cache, Data &input, int width, int seed) {
        Upload(input, {1, rows, width}, seed);
        auto original = Read<uint16_t>(input);
        cache.Resize({1, rows + past, width});
        cache.ToDevice(DataDevice::CUDA, {0}, false); cache.Allocate(false);
        Require(cudaMemset(cache.cudaData, 0, cache.GetBytes()) == cudaSuccess, "FP8 cache initialization");
        FastllmCudaNaiveWriteAndExpandFP8(cache, input, past);
        auto stored = Read<unsigned char>(cache);
        auto expanded = Read<uint16_t>(input);
        for (size_t i = 0; i < original.size(); ++i) {
            __nv_fp8_e4m3 fp8(FromBits(original[i]));
            Require(stored[size_t(past) * width + i] == fp8.__x, "FP8 cache write changed bytes");
            Require(expanded[i] == Float32ToBFloat16RNEBits(float(fp8)), "FP8 expansion changed values");
        }
        ++checks;
    };
    write(key, k, kvHeads * 192, 73);
    write(value, v, kvHeads * 128, 79);
    // Skip the prefix used to exercise the nonzero cache write offset.
    Data kd(FP8_E4M3, k.dims), vd(FP8_E4M3, v.dims);
    kd.FakeFrom(key, past * kvHeads * 192); vd.FakeFrom(value, past * kvHeads * 128);
    FastllmCudaNaiveAttention(q, kd, vd, selected, sink, heads, kvHeads, 192, 128, 0, 128, expected);
    FastllmCudaNaiveAttention(q, k, v, selected, sink, heads, kvHeads, 192, 128, 0, 128, actual);
    auto a = Read<uint16_t>(actual), b = Read<uint16_t>(expected);
    double error = 0, magnitude = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double delta = FromBits(a[i]) - FromBits(b[i]);
        error += delta * delta; magnitude += double(FromBits(b[i])) * FromBits(b[i]);
    }
    double l2 = std::sqrt(error / magnitude);
    std::printf("Cold SWA vs FP8 cache attention: relL2 %.7f\n", l2);
    Require(l2 < .008, "Cold SWA exceeds BF16 error budget");
    ++checks;
}

static void TestPairedPrefillAttention(int keys = 2017, int keep = 512) {
    if (!FastllmCudaHasBlackwellIndexerAttention()) return;
    constexpr int rows = 257, heads = 16;
    const int past = keys - rows;
    Data iq(BFLOAT16), weights(BFLOAT16), ik(BFLOAT16), q(BFLOAT16), k(BFLOAT16), v(BFLOAT16);
    Data ordered(INT32), privateIds(INT32, {rows, keep}), sink, expected, actual;
    Upload(iq, {1, rows, 2048}, 43); Upload(weights, {1, rows, 16}, 53); Upload(ik, {1, keys, 128}, 59);
    Upload(q, {1, rows, heads * 192}, 67); Upload(k, {1, keys, 192}, 71); Upload(v, {1, keys, 128}, 73);
    privateIds.ToDevice(DataDevice::CUDA, {0}, false); privateIds.Allocate(false);
    // Leave room after the score matrix for the long selector's candidates.
    Data loan(FLOAT32, {keys > 16384 ? rows * (keys + 8192 + 3) + 2 : 4 * 1024 * 1024});
    loan.ToDevice(DataDevice::CUDA, {0}, false); loan.Allocate(false);
    for (bool tied : {false, true}) {
        if (tied) Require(cudaMemset(weights.cudaData, 0, weights.GetBytes()) == cudaSuccess, "zero Indexer weights");
        FastllmCudaNaiveIndexer(iq, weights, ik, 16, 128, past, keep, true, ordered);
        FastllmNaivePrefillAttentionItem reference{&q, &k, &v, &ordered, &expected, past};
        Require(FastllmCudaTryNaiveIndexerAttention(reference, sink, heads, 1, 192, 128, loan),
                "ordered Blackwell attention declined reference");
        FastllmNaivePrefillIndexerItem index{&iq, &weights, &ik, &privateIds, past};
        FastllmNaivePrefillAttentionItem attention{&q, &k, &v, &privateIds, &actual, past};
        FastllmCudaNaivePrefillIndexedAttention(index, attention, sink, 16, 128, keep, true,
            false, heads, 1, 192, 128, &loan, loan);
        Require(Read<uint16_t>(actual) == Read<uint16_t>(expected), "paired prefill changed ordered TopK attention");
        ++checks;
    }
}

static void TestBlackwellLocalRouter() {
    if (!FastllmCudaHasBlackwellIndexerAttention()) return;
    constexpr int rows = 257, channels = 4096, experts = 64, top = 8;
    Data x(BFLOAT16), w(FLOAT32), out, scratch(FLOAT32, {rows, channels});
    Upload(x, {rows, channels}, 71);
    Upload(w, {experts, channels}, 113);
    scratch.ToDevice(DataDevice::CUDA, {0}, false); scratch.Allocate(false);
    Require(FastllmCudaNaivePrefillRouter(x, w, scratch, out), "compact router did not dispatch");
    auto a = Read<uint16_t>(x);
    auto b = Read<float>(w), actual = Read<float>(out);
    double error = 0, magnitude = 0;
    for (int i = 0; i < rows; ++i) {
        std::vector<double> ref(experts);
        std::vector<int> expected(experts), routed(experts);
        for (int e = 0; e < experts; ++e) {
            for (int k = 0; k < channels; ++k)
                ref[e] += double(FromBits(a[i * channels + k])) * b[e * channels + k];
            expected[e] = routed[e] = e;
            double diff = actual[i * experts + e] - ref[e];
            error += diff * diff; magnitude += ref[e] * ref[e];
        }
        std::partial_sort(expected.begin(), expected.begin()+top, expected.end(), [&](int u,int v){return ref[u]>ref[v];});
        std::partial_sort(routed.begin(), routed.begin()+top, routed.end(), [&](int u,int v){return actual[i*experts+u]>actual[i*experts+v];});
        std::sort(expected.begin(),expected.begin()+top); std::sort(routed.begin(),routed.begin()+top);
        Require(std::equal(expected.begin(),expected.begin()+top,routed.begin()), "router expert set mismatch");
    }
    double l2 = std::sqrt(error / magnitude);
    std::printf("Local router FP64 reference: relL2 %.9g, %d exact top-%d sets\n", l2, rows, top);
    Require(l2 < 2e-5, "router exceeds FP32 error budget");
    ++checks;
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    try {
        SetThreads(4);
        TestPackedPrefillRope();
        TestColdWindowCacheWrite();
        TestPairedPrefillAttention(); TestPairedPrefillAttention(8193, 2048); TestPairedPrefillAttention(16342, 2048);
        TestPairedPrefillAttention(65537, 2048);
        TestBlackwellLocalRouter();
        TestBlackwellIndexerAttention();
        TestLocalPrefillBatch();
        TestSwaPrefillScratch(); TestAttentionGroupedScores(); TestAttentionSwa(); TestIndexerPrefillShards(); TestIndexerRaggedShards();
        Require(cudaDeviceSynchronize()==cudaSuccess,"prefill CUDA synchronization failed");
        std::printf("Naive prefill regression passed: %d cases\n",checks);
    } catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
