#include "fastllm.h"
#include "models/naive_n05_flash.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
using namespace fastllm;
static void Require(bool v, const char *s) { if (!v) throw std::runtime_error(s); }
static int checks = 0;
static bool quick = false;
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
static void TestTopK() {
    for (int n : {1, 127, 2047, 2048, 2049, 7587, 8192, 8193, 32769, 131073})
    for (int mode = 0; mode < 3; ++mode) for (int fraction : {1, 2}) {
        if (quick && n != 127 && n != 7587) continue;
        Data scores(FLOAT32), result;
        scores.Resize({1, n}); scores.Allocate();
        auto *p = (float *)scores.cpuData;
        unsigned state = n;
        for (int i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            p[i] = mode == 0 ? (i & 1 ? -0.0f : 0.0f) : mode == 1
                ? float(int(state % 31) - 15) : float(int(state >> 8) - (1 << 23)) / 128;
        }
        if (mode == 2 && n > 5) {
            p[1] = p[2] = std::numeric_limits<float>::infinity();
            p[3] = -std::numeric_limits<float>::infinity();
        }
        int valid = std::max(1, n / fraction), top = n < 2048 ? n + 7 : 2048;
        std::vector<int> order(valid); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
        scores.ToDevice(DataDevice::CUDA, {0}, true);
        FastllmCudaNaiveTopK(scores, valid - 1, top, result);
        auto actual = Read<int>(result);
        for (int i = 0; i < top; ++i) {
            int expected=i < valid ? order[i] : -1;
            if(actual[i]!=expected) std::fprintf(stderr,"TopK n=%d mode=%d valid=%d i=%d got=%d expected=%d\n",n,mode,valid,i,actual[i],expected);
            Require(actual[i] == expected, "GPU TopK differs from stable CPU order");
        }
        ++checks;
    }
}
static std::vector<int> BatchedReference(const std::vector<float> &scores, int rows,
                                        int keys, int past, int top) {
    std::vector<int> result((size_t)rows * top, -1), order(keys);
    for (int row = 0; row < rows; ++row) {
        int valid = past + row + 1, keep = std::min(valid, top);
        std::iota(order.begin(), order.begin() + valid, 0);
        const float *p = scores.data() + (size_t)row * keys;
        std::partial_sort(order.begin(), order.begin() + keep, order.begin() + valid,
            [p](int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
        std::copy_n(order.data(), keep, result.data() + (size_t)row * top);
    }
    return result;
}
static std::vector<float> BatchedScores(int rows, int keys, int past, int mode, unsigned seed) {
    std::vector<float> values((size_t)rows * keys);
    for (int row = 0; row < rows; ++row) for (int col = 0; col < keys; ++col) {
        seed = seed * 1664525u + 1013904223u;
        float v = mode == 0 ? (col & 1 ? -0.0f : 0.0f) : mode == 1
            ? float(int(seed % 19) - 9) : float(int(seed >> 8) - (1 << 23)) / 8192.0f;
        if (mode == 1 && col < 3) v = col == 2 ? -std::numeric_limits<float>::infinity()
                                                            : std::numeric_limits<float>::infinity();
        // Future positions must never win, even with +infinity or NaN there.
        if (col > past + row) v = col & 1 ? std::numeric_limits<float>::infinity()
                                         : std::numeric_limits<float>::quiet_NaN();
        values[(size_t)row * keys + col] = v;
    }
    return values;
}
static void TestBatchedTopK() {
    Data result;
    struct Shape { int rows, keys, past, top; };
    const Shape shapes[] = {{2,2,0,7}, {3,127,0,129}, {33,2049,17,2048},
        {31,8192,8000,1}, {32,8192,8160,17}, {129,8193,8064,2048},
        {512,32768,32256,2048}, {512,65536,65024,2048},
        // Compact-selection dispatch boundary, partial causal rows, and odd K.
        {2,8191,8189,2048}, {2,8192,8190,2048}, {2,8193,8191,2048},
        {17,32769,0,7}, {3,65537,4095,2049}};
    for (const auto &s : shapes) for (int mode = 0; mode < 3; ++mode) {
        if (quick && s.rows > 33) continue;
        auto values = BatchedScores(s.rows, s.keys, s.past, mode, 713 + mode);
        auto expected = BatchedReference(values, s.rows, s.keys, s.past, s.top);
        Data scores(FLOAT32); scores.Resize({s.rows,s.keys}); scores.Allocate();
        std::memcpy(scores.cpuData, values.data(), values.size()*sizeof(float));
        scores.ToDevice(DataDevice::CUDA,{0},true);
        FastllmCudaNaiveTopK(scores,s.past,s.top,result);
        Require(result.dataDevice==DataDevice::CUDA && result.dims==std::vector<int>({s.rows,s.top}),
                "batched TopK changed output shape/device");
        auto actual=Read<int>(result);
        if (actual!=expected) {
            for(size_t i=0;i<actual.size();++i) if(actual[i]!=expected[i]) {
                std::fprintf(stderr,"BatchedTopK rows=%d keys=%d mode=%d i=%zu got=%d expected=%d\n",
                    s.rows,s.keys,mode,i,actual[i],expected[i]);break;
            }
        }
        Require(actual==expected,"Batched TopK differs from independent CPU comparator");
        ++checks;
    }
    // Change score contents between graph replays to catch stale host selection.
    for (const auto &s : {Shape{3,2049,1000,2048}, Shape{33,8192,8159,2048},
                          Shape{17,32769,32752,2048}}) {
        Data scores(FLOAT32), output; auto values=BatchedScores(s.rows,s.keys,s.past,2,19);
        scores.Resize({s.rows,s.keys});scores.Allocate();
        std::memcpy(scores.cpuData,values.data(),values.size()*sizeof(float));scores.ToDevice(DataDevice::CUDA,{0},true);
        FastllmCudaNaiveTopK(scores,s.past,s.top,output);
        Require(cudaDeviceSynchronize()==cudaSuccess,"TopK warmup failed");
        cudaGraph_t graph;cudaGraphExec_t exec;
        Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"capture begin");
        FastllmCudaNaiveTopK(scores,s.past,s.top,output);
        Require(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess,"capture end");
        Require(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)==cudaSuccess,"graph instantiate");
        for(int seed=41;seed<44;++seed) {
            values=BatchedScores(s.rows,s.keys,s.past,seed%3,seed);
            auto expected=BatchedReference(values,s.rows,s.keys,s.past,s.top);
            Require(cudaMemcpyAsync(scores.cudaData,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice,cudaStreamPerThread)==cudaSuccess,"graph upload");
            Require(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess,"graph launch");
            Require(cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess,"graph sync");
            Require(Read<int>(output)==expected,"graph TopK differs after score update");++checks;
        }
        cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
    }
}

struct CacheOps : NaiveN05FlashModel {
    using NaiveN05FlashModel::AppendCache;
    using NaiveN05FlashModel::CacheReserveCapacity;
};
static void TestCache() {
    for (auto dims : {std::pair<int,int>{1536,1024}, {8,16}, {200,56}, {7,13}}) {
        Data key(BFLOAT16), value(BFLOAT16);
        std::vector<uint16_t> refK, refV;
        for (int step = 0; step < (quick ? 4 : 80); ++step) {
            int n = step == 0 ? 512 : step % 23 == 0 ? 256 : step % 17 == 0 ? 7 : 1;
            Data k(BFLOAT16), v(BFLOAT16);
            Upload(k, {1,n,dims.first}, 13 + step); Upload(v, {1,n,dims.second}, 137 + step);
            auto kr = Read<uint16_t>(k), vr = Read<uint16_t>(v);
            refK.insert(refK.end(), kr.begin(), kr.end()); refV.insert(refV.end(), vr.begin(), vr.end());
            if (refK.size() > 127u * dims.first) refK.erase(refK.begin(), refK.end() - 127 * dims.first);
            if (refV.size() > 127u * dims.second) refV.erase(refV.begin(), refV.end() - 127 * dims.second);
            void *oldK = key.cudaData, *oldV = value.cudaData;
            CacheOps::AppendCache(key,k); CacheOps::AppendCache(value,v);
            if (step && n == 1) Require(key.cudaData == oldK && value.cudaData == oldV, "single-token append reallocated retained capacity");
            auto kd = key.expansionDims, vd = value.expansionDims;
            auto kp = key.cudaData, vp = value.cudaData;
            FastllmCudaNaiveTrimCache(key,value,127);
            Require(key.cudaData == kp && value.cudaData == vp && key.expansionDims == kd && value.expansionDims == vd,
                    "trim changed pointer or capacity");
            Require(key.isKVCache && value.isKVCache && Read<uint16_t>(key) == refK && Read<uint16_t>(value) == refV,
                    "sliding suffix content mismatch");
            ++checks;
        }
    }
}

static void TestCacheReservation() {
    CacheOps model;
    model.max_positions = 65536;
    const int oldMaxTokens = GetMaxTokens();
    struct RestoreLimit { int value; ~RestoreLimit() { SetMaxTokens(value); } } restore{oldMaxTokens};
    SetMaxTokens(65536);
    GenerationConfig config;
    config.input_token_length = 32727;
    config.output_token_limit = 256;
    Require(model.CacheReserveCapacity(config) == 32982, "full request reservation lost chunked prompt length");
    model.tokensLimit = 8192;
    Require(model.CacheReserveCapacity(config) == 8192, "model token budget not respected");
    model.tokensLimit = -1;
    SetMaxTokens(16384);
    Require(model.CacheReserveCapacity(config) == 16384, "configured token budget not respected");
    SetMaxTokens(65536);
    config.input_token_length = config.output_token_limit = std::numeric_limits<int>::max();
    Require(model.CacheReserveCapacity(config) == 65536, "large generation limit overflowed reservation");
    config.input_token_length = 512; config.output_token_limit = -1;
    Require(model.CacheReserveCapacity(config) == 512, "unbounded output should still reserve known prompt");
    config.input_token_length = 0;
    Require(model.CacheReserveCapacity(config) == 0, "warmup without request metadata changed reservation");
    ++checks;

    // Cross the old 32768-token expansion boundary while preserving every
    // logical row. A large physical capacity must not become logical length.
    for (int width : {896, 512, 8}) {
        const int prompt = quick ? 1405 : width == 8 ? 65519 : 32727;
        const int decode = quick ? 12 : width == 8 ? 17 : 68;
        config.input_token_length = prompt; config.output_token_limit = 256;
        const int reserve = model.CacheReserveCapacity(config);
        Data cache(BFLOAT16); std::vector<uint16_t> reference;
        void *address = nullptr; int done = 0, seed = 300;
        while (done < prompt + decode) {
            const int rows = done < prompt ? std::min(512, prompt - done) : 1;
            Data current(BFLOAT16); Upload(current, {1, rows, width}, seed++);
            auto values = Read<uint16_t>(current);reference.insert(reference.end(), values.begin(), values.end());
            CacheOps::AppendCache(cache, current, reserve);
            if (address) Require(cache.cudaData == address, "reserved global cache grew inside request horizon");
            address = cache.cudaData;done += rows;
            Require(cache.dims == std::vector<int>({1,done,width}), "physical reservation changed logical KV length");
            Require(cache.expansionDims[1] >= reserve && cache.isKVCache, "global cache capacity/flag incorrect");
        }
        Require(Read<uint16_t>(cache) == reference, "reserved global cache content differs");
        ++checks;
    }
    // A restored prefix may already own a smaller allocation. Growing it once
    // must preserve the prefix, and a later smaller hint must never truncate it.
    {
        Data cache(BFLOAT16), prefix(BFLOAT16), input(BFLOAT16);
        Upload(prefix,{1,129,16},701);Upload(input,{1,257,16},702);
        auto reference=Read<uint16_t>(prefix), tail=Read<uint16_t>(input);
        reference.insert(reference.end(),tail.begin(),tail.end());
        CacheOps::AppendCache(cache,prefix);
        CacheOps::AppendCache(cache,input,1024);
        auto address=cache.cudaData;
        CacheOps::AppendCache(cache,input,0);
        reference.insert(reference.end(),tail.begin(),tail.end());
        Require(cache.cudaData==address && cache.dims[1]==643 && Read<uint16_t>(cache)==reference,
                "restored-prefix reservation lost cache contents or shrank capacity");
        ++checks;
    }
    // Sliding caches reserve only window-1 + a prefill chunk. Reuse the same
    // pointers through mixed prefill/decode and verify the retained suffix.
    {
        Data key(BFLOAT16),value(BFLOAT16);std::vector<uint16_t> refK,refV;
        void *kp=nullptr,*vp=nullptr;
        for (int step=0;step<(quick?5:70);++step) {
            int rows=step<3?512:step==3?471:1;
            Data k(BFLOAT16),v(BFLOAT16);Upload(k,{1,rows,1536},900+step);Upload(v,{1,rows,1024},1000+step);
            auto kr=Read<uint16_t>(k),vr=Read<uint16_t>(v);refK.insert(refK.end(),kr.begin(),kr.end());refV.insert(refV.end(),vr.begin(),vr.end());
            CacheOps::AppendCache(key,k,127+rows);CacheOps::AppendCache(value,v,127+rows);
            if(kp)Require(key.cudaData==kp && value.cudaData==vp,"sliding reservation reallocated between chunks");
            kp=key.cudaData;vp=value.cudaData;
            Require(key.expansionDims[1]==640 && value.expansionDims[1]==640,"sliding reservation grew to full context");
            FastllmCudaNaiveTrimCache(key,value,127);
            refK.erase(refK.begin(),refK.end()-127*1536);refV.erase(refV.begin(),refV.end()-127*1024);
            Require(Read<uint16_t>(key)==refK && Read<uint16_t>(value)==refV,"reserved sliding suffix differs");
        }
        ++checks;
    }
}

static float FromBits(uint16_t bits) {
    uint32_t raw = uint32_t(bits) << 16;
    float value; std::memcpy(&value, &raw, sizeof(value)); return value;
}
static float Rounded(float value) { return FromBits(Float32ToBFloat16RNEBits(value)); }
static void Zeros(Data &data, const std::vector<int> &shape) {
    data.Resize(shape); data.Allocate();
    std::memset(data.cpuData, 0, data.GetBytes());
    data.ToDevice(DataDevice::CUDA, {0}, true);
}
static void TestRopeWidths() {
    for (int dim : {64, 192, 384}) {
        const int heads = 3, rows = 3, rotary = dim == 192 ? 64 : dim;
        const float theta = 10000.0f;
        Data input(BFLOAT16), positions(FLOAT32, {1,rows}, {0.0f,7.0f,1973.0f});
        Upload(input, {1,rows,heads*dim}, 817 + dim);
        auto before=Read<uint16_t>(input);
        positions.ToDevice(DataDevice::CUDA,{0},true);
        FastllmCudaNaiveRope(input,positions,heads,dim,rotary,theta);
        auto actual=Read<uint16_t>(input);
        const float pos[] = {0,7,1973};
        for (int r=0;r<rows;++r) for(int h=0;h<heads;++h) {
            size_t base=(r*heads+h)*dim;
            for(int d=0;d<rotary/2;++d) {
                float angle=pos[r]*std::pow(theta,-2.0f*d/rotary);
                float c=Rounded(std::cos(angle)), sn=Rounded(std::sin(angle));
                float x=FromBits(before[base+d]), y=FromBits(before[base+d+rotary/2]);
                float lo=Rounded(Rounded(x*c)-Rounded(y*sn));
                float hi=Rounded(Rounded(y*c)+Rounded(x*sn));
                // CPU/GPU transcendental implementations can round differently.
                Require(std::abs(FromBits(actual[base+d])-lo)<=0.012f &&
                        std::abs(FromBits(actual[base+d+rotary/2])-hi)<=0.012f,
                        "RoPE width differs from CPU reference");
            }
            for(int d=rotary;d<dim;++d)
                Require(actual[base+d]==before[base+d],"RoPE changed nonrotary coordinates");
        }
        ++checks;
    }
}
static void TestAttentionWidths() {
    struct Shape { int queries, keys, valueDim; };
    for(auto shape : {Shape{2,300,384},Shape{1,300,384},Shape{2,127,384},Shape{2,300,128}}) {
        const int heads=4,kvHeads=2,dim=96,past=shape.keys-shape.queries;
        Data query(BFLOAT16),key(BFLOAT16),value(BFLOAT16),sink(FLOAT32),indices,output;
        Zeros(query,{1,shape.queries,heads*dim});
        Zeros(key,{1,shape.keys,kvHeads*dim+128});
        Zeros(sink,{heads});
        Upload(value,{1,shape.keys,kvHeads*shape.valueDim},135);
        auto values=Read<uint16_t>(value);
        FastllmCudaNaiveAttention(query,key,value,indices,sink,heads,kvHeads,dim,
                                  shape.valueDim,past,0,output);
        auto actual=Read<uint16_t>(output);
        for(int q=0;q<shape.queries;++q) for(int h=0;h<heads;++h) for(int d=0;d<shape.valueDim;++d) {
            int valid=past+q+1;
            // All QK logits and the sink are zero: uniform probabilities have
            // an independent, exact denominator, including one sink position.
            float probability=Rounded(1.0f/(valid+1));
            float sum=0;
            for(int k=0;k<valid;++k)
                sum=std::fma(probability,FromBits(values[(k*kvHeads+h/(heads/kvHeads))*shape.valueDim+d]),sum);
            Require(actual[(q*heads+h)*shape.valueDim+d]==Float32ToBFloat16RNEBits(sum),
                    "Attention width differs from independent uniform-softmax reference");
        }
        ++checks;
    }
}

static void TestAttentionSelectedValues() {
    struct Shape { int queries, heads, kvHeads, dim, valueDim, keys, selected; };
    for (auto s : {Shape{2,1,1,64,4,257,257}, Shape{5,3,1,129,132,513,511},
                   Shape{7,8,2,192,128,1027,2051}, Shape{3,4,2,256,384,300,301},
                   Shape{2,7,1,384,127,259,259}, Shape{2,4,2,193,8,257,257}})
    for (bool causal : {false, true}) for (bool withSink : {false, true}) {
        int past = s.keys - s.queries;
        Data query(BFLOAT16),key(BFLOAT16),value(BFLOAT16),sink(FLOAT32),indices(INT32),output;
        Zeros(query,{1,s.queries,s.heads*s.dim});
        Zeros(key,{1,s.keys,s.kvHeads*s.dim+128});
        Upload(value,{1,s.keys,s.kvHeads*s.valueDim},157);
        if (withSink) Zeros(sink,{s.heads});
        indices.Resize({s.queries,s.selected}); indices.Allocate();
        std::vector<int> selected(s.queries*s.selected);
        for (int q=0;q<s.queries;++q) for (int slot=0;slot<s.selected;++slot) {
            int k=(slot*137+q*13)%s.keys;
            if (slot%17==0) k=-1;
            if (slot%31==1) k=s.keys+3;
            // The first row also covers the all-masked case.
            if (q==0) k=-1;
            selected[q*s.selected+slot]=k;
        }
        std::memcpy(indices.cpuData,selected.data(),selected.size()*sizeof(int));
        indices.ToDevice(DataDevice::CUDA,{0},true);
        auto values=Read<uint16_t>(value);
        FastllmCudaNaiveAttention(query,key,value,indices,sink,s.heads,s.kvHeads,s.dim,
                                  s.valueDim,past,0,output,causal);
        auto actual=Read<uint16_t>(output);
        for (int q=0;q<s.queries;++q) {
            std::vector<int> validKeys;
            for (int slot=0;slot<s.selected;++slot) {
                int k=selected[q*s.selected+slot];
                if (k>=0 && k<s.keys && (!causal || k<=past+q)) validKeys.push_back(k);
            }
            float probability=validKeys.empty() ? 0 : Rounded(1.0f/(validKeys.size()+int(withSink)));
            for (int h=0;h<s.heads;++h) for (int d=0;d<s.valueDim;++d) {
                float sum=0;
                for (int k : validKeys)
                    sum=std::fma(probability,FromBits(values[(k*s.kvHeads+h/(s.heads/s.kvHeads))*s.valueDim+d]),sum);
                Require(actual[(q*s.heads+h)*s.valueDim+d]==Float32ToBFloat16RNEBits(sum),
                        "Selected Attention differs from independent uniform-softmax reference");
            }
        }
        ++checks;
    }
}

// Compare grouped QK with independent one-head calls, which use the original
// QK path. Nonzero Q/K make this sensitive to dot/reduction/rounding changes.
static void TestAttentionGroupedScores() {
    struct Shape { int queries, heads, kvHeads, dim, valueDim, keys, selected; };
    for (auto s : {Shape{32,4,2,64,8,513,511}, Shape{33,14,2,192,128,1027,2051},
                   Shape{32,8,2,129,132,513,0}, Shape{31,8,2,192,128,513,512}})
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
                if (slot%97==0 || q==0) k=-1;
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
            for (int r=0;r<s.queries;++r) for (int d=0;d<s.valueDim;++d)
                Require(actual[(r*s.heads+h)*s.valueDim+d]==expected[r*s.valueDim+d],
                        "Grouped Attention differs from isolated head");
        }
        ++checks;
    }
}

// Independent FP64 reference for the global GQA path. Reproduce the model's
// two logits roundings and probability rounding, but not either GPU reduction.
static void TestAttentionGlobalMma() {
    struct Shape { int rows, keys, count, heads, kvHeads, padding; bool selected; };
    for (auto s : {Shape{32,512,512,64,4,0,false}, Shape{33,515,513,64,4,128,true},
                   Shape{32,2048,2051,64,4,0,true}, Shape{64,2048,2048,32,1,8,true},
                   Shape{32,515,257,128,4,128,true}, Shape{32,4096,2048,64,4,128,true}})
    for (bool causal : {false,true}) for (int mode=0;mode<5;++mode) {
        constexpr int dim=192,valueDim=128;
        int past=s.keys-s.rows,stride=s.kvHeads*dim+s.padding;
        Data q(BFLOAT16),k(BFLOAT16),v(BFLOAT16),indices(INT32),sink(FLOAT32),out;
        Upload(q,{1,s.rows,s.heads*dim},53); Upload(k,{1,s.keys,stride},71);
        Upload(v,{1,s.keys,s.kvHeads*valueDim},113);
        auto qb=Read<uint16_t>(q),kb=Read<uint16_t>(k),vb=Read<uint16_t>(v);
        if (mode==1 || mode==3) {
            for (auto &x:qb) x=Float32ToBFloat16RNEBits(mode==3?0:4*FromBits(x));
            for (auto &x:kb) x=Float32ToBFloat16RNEBits(mode==3?0:4*FromBits(x));
            if (mode==3) {
                for (auto &x:vb) x=Float32ToBFloat16RNEBits(.375f);
                Require(cudaMemcpy(v.cudaData,vb.data(),vb.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"global V upload");
            }
            Require(cudaMemcpy(q.cudaData,qb.data(),qb.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"global Q upload");
            Require(cudaMemcpy(k.cudaData,kb.data(),kb.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"global K upload");
        }
        std::vector<int> selected;
        if (s.selected) {
            selected.resize(s.rows*s.count);
            indices.Resize({s.rows,s.count});indices.Allocate();
            for (int r=0;r<s.rows;++r) for (int slot=0;slot<s.count;++slot) {
                int key=(slot*137+r*13)%s.keys;
                if (slot%97==0 || r==0 || mode==4) key=-1;
                else if (slot%193==1) key=s.keys+7;
                selected[r*s.count+slot]=key;
            }
            std::memcpy(indices.cpuData,selected.data(),selected.size()*sizeof(int));
            indices.ToDevice(DataDevice::CUDA,{0},true);
        }
        bool withSink=mode!=0;
        std::vector<float> sinks;
        if (withSink) {
            sinks.resize(s.heads);
            sink.Resize({s.heads});sink.Allocate();
            for(int h=0;h<s.heads;++h) sinks[h]=(h%3-1)*(mode==2?80.f:1.3f);
            std::memcpy(sink.cpuData,sinks.data(),sinks.size()*sizeof(float));
            sink.ToDevice(DataDevice::CUDA,{0},true);
        }
        FastllmCudaNaiveAttention(q,k,v,indices,sink,s.heads,s.kvHeads,dim,valueDim,past,0,out,causal);
        auto actual=Read<uint16_t>(out);
        for (auto x:actual) Require(std::isfinite(FromBits(x)),"global nonfinite output");
        double error=0,norm=0;
        for (int r:{0,s.rows/2,s.rows-1}) for (int h:{0,s.heads/2,s.heads-1}) {
            std::vector<double> probs(s.count);std::vector<int> valid(s.count,-1);
            double maximum=withSink?sinks[h]:-INFINITY;
            for (int slot=0;slot<s.count;++slot) {
                int key=s.selected?selected[r*s.count+slot]:slot;
                double score=-INFINITY;
                if (key>=0 && key<s.keys && (!causal || key<=past+r)) {
                    valid[slot]=key; double dot=0;
                    for(int d=0;d<dim;++d) dot+=double(FromBits(qb[((size_t)r*s.heads+h)*dim+d]))*
                        FromBits(kb[(size_t)key*stride+(h/(s.heads/s.kvHeads))*dim+d]);
                    score=Rounded(Rounded(float(dot))/std::sqrt(float(dim)));
                }
                probs[slot]=score;maximum=std::max(maximum,score);
            }
            double denominator=withSink?std::exp(sinks[h]-maximum):0;
            for(auto &p:probs){p=std::isfinite(maximum)?std::exp(p-maximum):0;denominator+=p;}
            for(auto &p:probs)p=denominator>0?Rounded(float(p/denominator)):0;
            for(int d:{0,31,64,127}) {
                double expected=0;
                for(int slot=0;slot<s.count;++slot) if(valid[slot]>=0)
                    expected+=probs[slot]*FromBits(vb[((size_t)valid[slot]*s.kvHeads+h/(s.heads/s.kvHeads))*valueDim+d]);
                expected=Rounded(float(expected));
                double got=FromBits(actual[((size_t)r*s.heads+h)*valueDim+d]);
                Require(std::abs(got-expected)<=.012,"global attention differs from independent reference");
                error+=(got-expected)*(got-expected);norm+=expected*expected;
            }
        }
        Require(std::sqrt(error/std::max(norm,1e-30))<=.01,"global relative error exceeds 1 percent");
        ++checks;
    }
}

// Standard causal SWA, including the learned zero-value sink, evaluated with
// independent double-precision dots/softmax on sampled rows and output columns.
// FlashInfer changes the reduction/rounding order, so compare numerically.
static void TestAttentionSwa() {
    struct Shape { int queries, past, padding, mode; float amplitude; };
    for (auto s : {Shape{31,127,0,1,1}, Shape{32,0,0,0,1}, Shape{56,0,128,1,1},
                   Shape{80,0,0,3,1}, Shape{33,127,0,2,1}, Shape{33,127,0,1,4},
                   Shape{512,0,0,1,1}, Shape{512,127,0,1,1}, Shape{471,127,0,1,1},
                   Shape{1,0,0,0,1}, Shape{1,31,0,1,1}, Shape{1,63,0,1,1},
                   Shape{1,126,0,1,1}, Shape{1,127,0,0,1}, Shape{1,127,0,1,1},
                   Shape{1,127,0,2,1}, Shape{1,127,0,3,1}, Shape{1,127,0,1,4}}) {
        constexpr int heads=64, kvHeads=8, dim=192, valueDim=128;
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
        for (int row:rows) for (int h:{0,17,63}) {
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

// Independent CPU E4M3 nearest-even quantizer; use dequantized FP32 operands
// and FP64 score accumulation to check both the old and MMA dispatch paths.
static float IndexerE4M3(float x) {
    float a = std::min(448.f, std::abs(x)), best = 0.f;
    float distance = a;
    for (int bits = 1; bits <= 126; ++bits) {
        int exponent = bits >> 3, mantissa = bits & 7;
        float value = exponent ? std::ldexp(float(8 + mantissa), exponent - 10)
                               : std::ldexp(float(mantissa), -9);
        float delta = std::abs(value - a);
        if (delta < distance || (delta == distance && !(bits & 1))) {
            best = value; distance = delta;
        }
    }
    return std::copysign(best, x);
}
static std::vector<float> IndexerOperands(const std::vector<uint16_t> &input,
                                         int rows, int stride, int offset, bool fp8) {
    std::vector<float> out((size_t)rows * 128);
    for (int row = 0; row < rows; ++row) {
        float maximum = 0.f;
        for (int d = 0; d < 128; ++d)
            maximum = std::max(maximum, std::abs(FromBits(input[(size_t)row * stride + offset + d])));
        float scale = std::max(maximum, 1e-4f) / 448.f;
        for (int d = 0; d < 128; ++d) {
            float x = FromBits(input[(size_t)row * stride + offset + d]);
            out[(size_t)row * 128 + d] = fp8 ? IndexerE4M3(x / scale) * scale : x;
        }
    }
    return out;
}
static void TestIndexer() {
    struct Shape { int rows, keys, heads, stride; bool fp8; };
    const Shape shapes[] = {
        {1,257,16,896,true}, {31,4099,16,897,true},
        {32,32767,16,896,true}, {32,32768,16,896,true},
        {33,32769,16,897,true}, {64,16384,16,896,true},
        {65,16387,16,896,true}, {512,2051,16,896,true},
        {64,16384,8,896,true}, {64,16384,16,896,false}};
    for (const auto &s : shapes) for (int mode = 0; mode < 5; ++mode) {
        if (quick && s.rows != 33 && s.rows != 1) continue;
        Data q(BFLOAT16), k(BFLOAT16), w(BFLOAT16), out;
        Upload(q, {1,s.rows,s.heads*128}, 123);
        Upload(k, {1,s.keys,s.stride}, 287);
        Upload(w, {1,s.rows,s.heads}, 555);
        auto qb = Read<uint16_t>(q), kb = Read<uint16_t>(k), wb = Read<uint16_t>(w);
        for (auto &v : wb) if (mode != 1) v &= 0x7fff;
        if (mode == 2) std::fill(qb.begin(),qb.end(),0);
        if (mode == 3) std::fill(kb.begin(),kb.end(),Float32ToBFloat16RNEBits(.125f));
        if (mode == 4) {
            for (auto &v : qb) v = Float32ToBFloat16RNEBits(FromBits(v) * 16.f);
            for (auto &v : kb) v = Float32ToBFloat16RNEBits(FromBits(v) / 256.f);
        }
        Require(cudaMemcpy(q.cudaData,qb.data(),qb.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"Indexer Q upload");
        Require(cudaMemcpy(k.cudaData,kb.data(),kb.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"Indexer K upload");
        Require(cudaMemcpy(w.cudaData,wb.data(),wb.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"Indexer W upload");
        // Include causal future keys and output padding in the small case.
        int past = s.rows == 1 ? 63 : s.keys-s.rows;
        int top = s.rows == 1 ? 80 : 2048;
        FastllmCudaNaiveIndexer(q,w,k,s.heads,128,past,top,s.fp8,out);
        auto got = Read<int>(out);
        auto qv = IndexerOperands(qb,s.rows*s.heads,128,0,s.fp8);
        auto kv = IndexerOperands(kb,s.keys,s.stride,s.stride-128,s.fp8);
        std::vector<int> rows={0,s.rows/2,s.rows-1};
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        for (int row : rows) {
            int valid = past+row+1, keep = std::min(top,valid);
            std::vector<double> scores(valid);
            double magnitude = 0;
            for (int key=0; key<valid; ++key) {
                double score=0;
                for (int head=0; head<s.heads; ++head) {
                    double dot=0;
                    for (int d=0; d<128; ++d)
                        dot+=double(qv[((size_t)row*s.heads+head)*128+d])*kv[(size_t)key*128+d];
                    score+=std::max(dot,0.)*FromBits(wb[row*s.heads+head]);
                }
                scores[key] = score;
                magnitude = std::max(magnitude, std::abs(score));
            }
            std::vector<int> order(valid);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(),order.begin()+keep,order.end(),[&](int a,int b){
                return scores[a]>scores[b] || (scores[a]==scores[b] && a<b);
            });
            double tolerance=std::max(1e-6,magnitude*3e-6);
            std::vector<bool> seen(valid,false);
            for (int i=0;i<top;++i) {
                int key=got[(size_t)row*top+i];
                if (i >= keep) {
                    Require(key == -1, "Indexer invalid output padding");
                    continue;
                }
                Require(key>=0&&key<valid,"Indexer selected a future/out-of-range key");
                Require(!seen[key], "Indexer selected a duplicate key");
                seen[key] = true;
                Require(std::abs(scores[key]-scores[order[i]])<=tolerance,
                        "Indexer ranking differs from independent FP64 reference");
                if (mode==2||mode==3) Require(key==i,"Indexer equal-score tie order changed");
            }
        }
        ++checks;
    }
}

int main(int argc,char **argv) {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    if (argc == 2 && std::strcmp(argv[1], "--invalid-indexer") == 0) {
        Data query, weights, key, indices;
        FastllmCudaNaiveIndexer(query,weights,key,1,64,0,1,false,indices);
        std::fprintf(stderr,"Invalid Indexer width was accepted\n"); return 1;
    }
    try {
        Require(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--quick") == 0),
                "usage: naive_n05_decode_test [--quick]");
        quick = argc == 2;
        SetThreads(4);
        TestTopK(); TestBatchedTopK(); TestIndexer(); TestCache(); TestCacheReservation(); TestRopeWidths(); TestAttentionWidths(); TestAttentionSelectedValues(); TestAttentionGroupedScores(); TestAttentionGlobalMma(); TestAttentionSwa();
        Require(cudaDeviceSynchronize()==cudaSuccess,"CUDA final synchronization failed");
        std::printf("Naive decode regression passed: %d cases\n",checks);
    }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
    return 0;
}
