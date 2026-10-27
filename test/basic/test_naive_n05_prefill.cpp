#include "fastllm.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
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
    struct Shape { int queries, past, padding, mode; float amplitude; int heads=64, kvHeads=8; };
    for (auto s : {Shape{1,127,0,1,1}, Shape{257,7935,0,1,1,8,1},
                   Shape{257,7935,0,0,1,8,1}, Shape{1024,127,0,1,1,8,1},
                   Shape{471,127,0,2,1}, Shape{512,127,0,3,1}, Shape{33,127,0,1,4}}) {
        const int heads=s.heads, kvHeads=s.kvHeads;
        constexpr int dim=192, valueDim=128;
        int keys=s.past+s.queries, stride=kvHeads*dim+s.padding;
        Data q(BFLOAT16),k(BFLOAT16),v(BFLOAT16),sink(FLOAT32),indices,out;
        Upload(q,{1,s.queries,heads*dim},53); Upload(k,{1,keys,stride},71);
        Upload(v,{1,keys,kvHeads*valueDim},113);
        auto qb=Read<uint16_t>(q),kb=Read<uint16_t>(k),vb=Read<uint16_t>(v);
        for (auto &x:qb) x=Float32ToBFloat16RNEBits(s.mode==3 ? 0 : FromBits(x)*s.amplitude);
        for (auto &x:kb) x=Float32ToBFloat16RNEBits(s.mode==3 ? 0 : FromBits(x)*s.amplitude);
        if (s.mode==3) for (auto &x:vb) x=Float32ToBFloat16RNEBits(.375f);
        Require(cudaMemcpy(q.cudaData,qb.data(),qb.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"SWA query upload");
        Require(cudaMemcpy(k.cudaData,kb.data(),kb.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"SWA key upload");
        Require(cudaMemcpy(v.cudaData,vb.data(),vb.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"SWA value upload");
        std::vector<float>bias(heads);
        if (s.mode) {
            sink.Resize({heads});sink.Allocate();
            for (int h=0;h<heads;++h) bias[h]=s.mode==2 ? (h%3-1)*80.f : (h%5-2)*1.3f;
            std::memcpy(sink.cpuData,bias.data(),heads*sizeof(float));sink.ToDevice(DataDevice::CUDA,{0},true);
        }
        FastllmCudaNaiveAttention(q,k,v,indices,sink,heads,kvHeads,dim,valueDim,s.past,128,out,true);
        auto actual=Read<uint16_t>(out);
        for (auto x:actual) Require(std::isfinite(FromBits(x)),"SWA nonfinite output");
        double error=0,norm=0;
        std::vector<int> rows = {0, s.queries / 2, s.queries - 1};
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        for (int row:rows) for (int h:{0,heads/2,heads-1}) {
            int end=s.past+row+1,start=std::max(0,end-128);
            std::vector<double>scores(end-start);double maximum=s.mode ? bias[h] : -INFINITY;
            for (int z=start;z<end;++z) {
                double dot=0;
                for (int d=0;d<dim;++d)
                    dot+=double(FromBits(qb[((size_t)row*heads+h)*dim+d]))*FromBits(kb[(size_t)z*stride+h/(heads/kvHeads)*dim+d]);
                scores[z-start]=dot/std::sqrt(double(dim));maximum=std::max(maximum,scores[z-start]);
            }
            double denominator=s.mode ? std::exp(bias[h]-maximum) : 0;
            for (auto &x:scores) { x=std::exp(x-maximum);denominator+=x; }
            for (int d:{0,31,64,127}) {
                double expected=0;
                for (int z=start;z<end;++z)
                    expected+=scores[z-start]*FromBits(vb[((size_t)z*kvHeads+h/(heads/kvHeads))*valueDim+d]);
                expected/=denominator;
                double got=FromBits(actual[((size_t)row*heads+h)*valueDim+d]);
                Require(std::abs(got-expected)<=.012,"SWA differs from independent reference");
                error+=(got-expected)*(got-expected);norm+=expected*expected;
            }
        }
        Require(std::sqrt(error/std::max(norm,1e-30))<=.01,"SWA relative error exceeds 1 percent");
        ++checks;
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

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    try {
        SetThreads(4);
        TestAttentionGroupedScores(); TestAttentionSwa(); TestIndexerPrefillShards();
        Require(cudaDeviceSynchronize()==cudaSuccess,"prefill CUDA synchronization failed");
        std::printf("Naive prefill regression passed: %d cases\n",checks);
    } catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
