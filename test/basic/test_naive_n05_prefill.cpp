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

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    try {
        SetThreads(4);
        TestAttentionGroupedScores();
        Require(cudaDeviceSynchronize()==cudaSuccess,"prefill CUDA synchronization failed");
        std::printf("Naive prefill regression passed: %d cases\n",checks);
    } catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
