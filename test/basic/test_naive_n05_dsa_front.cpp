#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace fastllm;
namespace {
void Require(bool ok, const char *s) { if (!ok) throw std::runtime_error(s); }
void Cuda(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
void Init(Data &d, DataType type, const std::vector<int> &shape) {
    d.dataType = type; d.Resize(shape); d.ToDevice(DataDevice::CUDA, {FastllmCudaGetDevice()}, false); d.Allocate(false);
}
void Fill(Data &d, unsigned seed, float scale = 1, float offset = 0) {
    std::vector<unsigned char> bytes(d.GetBytes());
    for (size_t i = 0; i < d.Count(0); ++i) {
        seed = seed * 1664525u + 1013904223u;
        float x = (int(seed >> 16) - 32768) / 32768.f * scale + offset;
        if (d.dataType == BFLOAT16) { uint16_t y = Float32ToBFloat16RNEBits(x); std::memcpy(bytes.data() + 2*i, &y, 2); }
        else std::memcpy(bytes.data() + 4*i, &x, 4);
    }
    Cuda(cudaMemcpy(d.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}
std::vector<uint16_t> Read(const Data &d) {
    std::vector<uint16_t> b(d.GetBytes()/2); Cuda(cudaMemcpy(b.data(), d.cudaData, d.GetBytes(), cudaMemcpyDeviceToHost)); return b;
}
double Compare(const Data &x, const Data &y, const char *name) {
    auto a = Read(x), b = Read(y); Require(a.size() == b.size(), "shape mismatch");
    double err = 0, mag = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double va = BFloat16BitsToFloat32(a[i]), vb = BFloat16BitsToFloat32(b[i]);
        Require(std::isfinite(va) && std::isfinite(vb), "nonfinite output");
        err += (va-vb)*(va-vb); mag += vb*vb;
    }
    double l2 = std::sqrt(err / (mag + 1e-30));
    if (std::sqrt(err) > .006 * std::sqrt(mag) + 1e-6)
        throw std::runtime_error(std::string(name) + " L2=" + std::to_string(l2));
    return l2;
}
struct State {
    Data raw, index, q, out, table;
    std::vector<std::unique_ptr<Data>> keys, values;
    std::vector<std::unique_ptr<Data>> pageIds, pageViews;
    std::vector<std::unique_ptr<PagedCacheManager>> pools;
    std::vector<Data> indexViews;
    std::vector<FastllmNaiveRopeCacheRow> rows;
    FastllmNaiveDecodeScratch scratch;
    State(int n, int capacity, bool indirect, bool scaled) : indexViews(n) {
        Init(raw, BFLOAT16, {1,n,1856}); Init(index, BFLOAT16, {1,n,128});
        Init(q, BFLOAT16, {1,n,1536}); Init(out, BFLOAT16, {1,n,1024});
        Init(table, INT32, {n*8});
        std::vector<const void *> pointers;
        for (int r=0;r<n;++r) {
            keys.emplace_back(new Data()); values.emplace_back(new Data());
            Init(*keys.back(), BFLOAT16, {1,capacity,192+(scaled?72:128)});
            Init(*values.back(), BFLOAT16, {1,capacity,128});
            keys.back()->expansionDims=keys.back()->dims; values.back()->expansionDims=values.back()->dims;
            keys.back()->kvCacheFp8Indexer=scaled;
            Fill(*keys.back(), 71+r,.5f); Fill(*values.back(),83+r,.5f);
            indexViews[r].Resize({1,1,128}); indexViews[r].FakeFrom(index,r*128*2);
            pointers.insert(pointers.end(),{keys.back()->cudaData,values.back()->cudaData,nullptr,nullptr});
            rows.push_back({keys.back().get(),values.back().get(),&indexViews[r],
                indirect?(const void *const *)table.cudaData+r*4:nullptr});
        }
        Cuda(cudaMemcpy(table.cudaData,pointers.data(),n*4*sizeof(void*),cudaMemcpyHostToDevice));
    }
    void Paged(int capacity) {
        const int pages = (capacity + 127) / 128;
        std::vector<const void *> binding;
        for (size_t r = 0; r < rows.size(); ++r) {
            for (int part = 0; part < 2; ++part) {
                Data &payload = *(part ? values[r] : keys[r]);
                auto pool = std::make_unique<PagedCacheManager>();
                pool->Resize(payload.dims); pool->FakeFrom(payload, 0);
                pool->pageLen = 128; pool->maxPages = pages;
                auto ids = std::make_unique<Data>(); Init(*ids, INT32, {pages});
                std::vector<int> map(pages);
                for (int i = 0; i < pages; ++i) map[i] = part ? (i + 1) % pages : pages - 1 - i;
                Cuda(cudaMemcpy(ids->cudaData, map.data(), pages * 4, cudaMemcpyHostToDevice));
                auto view = std::make_unique<Data>(); view->Resize(payload.dims); view->FakeFrom(*ids, 0);
                view->dataType = BFLOAT16; view->UpdateUnitSize(); view->expansionDims = payload.dims;
                view->isPagedKVCache = true; view->pageLen = 128; view->pagedKVCacheData = pool.get();
                view->kvCacheFp8Indexer = payload.kvCacheFp8Indexer;
                if (part) rows[r].value = view.get(); else rows[r].key = view.get();
                pools.push_back(std::move(pool)); pageIds.push_back(std::move(ids)); pageViews.push_back(std::move(view));
            }
            binding.insert(binding.end(), {keys[r]->cudaData, values[r]->cudaData,
                rows[r].key->cudaData, rows[r].value->cudaData});
        }
        Cuda(cudaMemcpy(table.cudaData, binding.data(), binding.size() * sizeof(void *), cudaMemcpyHostToDevice));
    }

};
struct Test {
    int n, capacity; bool scaled, ordered = false;
    Data input, weight, iw, nw, nb, live, positions, sink, k, v, empty;
    State actual, reference, alternate;
    std::vector<int> capacities;
    Test(int rows,int cap,bool indirect,bool scaled) : n(rows),capacity(cap),scaled(scaled),
        actual(rows,cap,indirect,scaled),reference(rows,cap,false,scaled),alternate(rows,cap,false,scaled),capacities(rows,cap) {
        Init(input,BFLOAT16,{1,n,4096}); Init(weight,BFLOAT16,{1856,4096}); Init(iw,BFLOAT16,{128,4096});
        Init(nw,FLOAT32,{128}); Init(nb,FLOAT32,{128}); Init(sink,FLOAT32,{8});
        Init(live,INT32,{n}); Init(positions,FLOAT32,{1,n});
        Init(k,BFLOAT16,{1,n,192}); Init(v,BFLOAT16,{1,n,128});
        Fill(input,17); Fill(weight,31,.03f); Fill(iw,41,.02f);
        Fill(nw,53,.1f,1); Fill(nb,61,.1f); Fill(sink,67,.5f);
        Lengths(cap);
    }
    void Lengths(int length) {
        std::vector<int> lengths(n); std::vector<float> pos(n);
        for(int r=0;r<n;++r) { lengths[r]=std::max(0,length-r);pos[r]=std::max(0,lengths[r]-1); }
        Cuda(cudaMemcpy(live.cudaData,lengths.data(),n*4,cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(positions.cudaData,pos.data(),n*4,cudaMemcpyHostToDevice));
    }
    bool Fused() {
        return FastllmCudaNaiveDSAFront(input,weight,empty,iw,nw,nb,actual.raw,actual.index,
            actual.q,k,v,positions,actual.rows,capacities,live,sink,8,1,192,128,64,10000.f,.75f,actual.scratch,actual.out,ordered);
    }
    void Legacy() {
        Require(FastllmCudaBFloat16MatMulBFloat16(input,weight,empty,reference.raw,n,4096,1856),"QKV projection");
        Require(FastllmCudaBFloat16MatMulBFloat16(input,iw,empty,reference.index,n,4096,128),"index projection");
        Require(FastllmCudaNaiveIndexerKeyNorm(reference.index,nw,nb),"index norm");
        Require(FastllmCudaNaiveBatchRopeAppendCache(reference.q,k,v,positions,reference.rows,live,
            8,1,192,128,64,10000.f,.75f,0,&reference.raw,false),"rope append");
        std::vector<FastllmNaiveDSARow> attention(n);
        for(int r=0;r<n;++r) {
            auto &d=attention[r];
            d.key=reference.rows[r].key->isPagedKVCache ? reference.rows[r].key->pagedKVCacheData->cudaData : reference.rows[r].key->cudaData;
            d.value=reference.rows[r].value->isPagedKVCache ? reference.rows[r].value->pagedKVCacheData->cudaData : reference.rows[r].value->cudaData;
            if (reference.rows[r].key->isPagedKVCache) {
                d.keyPages = (const int *)reference.rows[r].key->cudaData;
                d.valuePages = (const int *)reference.rows[r].value->cudaData;
                d.pageLen = 128; d.poolTokens = capacity;
            }
            d.capacity=d.length=capacity; d.keyStride=192+(scaled?72:128);d.liveKeys=(const int*)live.cudaData+r;
        }
        FastllmCudaNaiveBatchDSADecode(reference.q,sink,attention,8,1,192,128,reference.scratch,reference.out);
    }
    double Check(bool rebound=false) {
        Compare(actual.raw,reference.raw,"QKV"); Compare(actual.index,reference.index,"index");
        Compare(actual.q,reference.q,"Q");
        for(int r=0;r<n;++r) {
            // Scaled Indexer bytes are not BF16 numbers. Attention K/V and
            // normalized index values are compared independently above.
            if(!scaled) Compare(*(rebound?alternate:actual).keys[r],*reference.keys[r],"K cache");
            else {
                auto a=Read(*(rebound?alternate:actual).keys[r]), b=Read(*reference.keys[r]);
                double error=0, magnitude=0;
                for(int token=0;token<capacity;++token)for(int d=0;d<192;++d) {
                    double x=BFloat16BitsToFloat32(a[token*264+d]), y=BFloat16BitsToFloat32(b[token*264+d]);
                    error+=(x-y)*(x-y);magnitude+=y*y;
                }
                Require(std::sqrt(error)<=.006*std::sqrt(magnitude)+1e-6,"scaled K cache L2");
                int length=0; Cuda(cudaMemcpy(&length,(const int*)live.cudaData+r,4,cudaMemcpyDeviceToHost));
                if(length>0) {
                    int physical=length-1;
                    if(reference.rows[r].key->isPagedKVCache) {
                        int page=0; Cuda(cudaMemcpy(&page,(const int*)reference.rows[r].key->cudaData+physical/128,4,cudaMemcpyDeviceToHost));
                        physical=page*128+physical%128;
                    }
                    auto fp8=[](unsigned char x) {
                        int exp=(x>>3)&15, mant=x&7;
                        return (x&128?-1.f:1.f)*std::ldexp(exp?1.f+mant/8.f:mant/8.f,exp?exp-7:-6);
                    };
                    const auto *x=(const unsigned char*)a.data()+physical*528+384;
                    const auto *y=(const unsigned char*)b.data()+physical*528+384;
                    float sx,sy;std::memcpy(&sx,x+128,4);std::memcpy(&sy,y+128,4);
                    Require(std::isfinite(sx)&&std::isfinite(sy)&&sx>0&&sy>0,"invalid index scale");
                    error=magnitude=0;
                    for(int d=0;d<128;++d) {double va=fp8(x[d])*sx,vb=fp8(y[d])*sy;error+=(va-vb)*(va-vb);magnitude+=vb*vb;}
                    Require(std::sqrt(error)<=.03*std::sqrt(magnitude)+1e-7,"FP8 Indexer cache L2");
                }
            }
            Compare(*(rebound?alternate:actual).values[r],*reference.values[r],"V cache");
        }
        return Compare(actual.out,reference.out,"attention");
    }
};
template<class F> cudaGraphExec_t Capture(F f) {
    f(); Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)); f();
    cudaGraph_t g=nullptr; Cuda(cudaStreamEndCapture(cudaStreamPerThread,&g));
    cudaGraphExec_t e=nullptr; Cuda(cudaGraphInstantiate(&e,g,0)); Cuda(cudaGraphDestroy(g)); return e;
}
float Time(cudaGraphExec_t e) {
    for(int i=0;i<20;++i) Cuda(cudaGraphLaunch(e,cudaStreamPerThread));
    cudaEvent_t a,b; Cuda(cudaEventCreate(&a)); Cuda(cudaEventCreate(&b));
    Cuda(cudaEventRecord(a,cudaStreamPerThread));
    for(int i=0;i<200;++i) Cuda(cudaGraphLaunch(e,cudaStreamPerThread));
    Cuda(cudaEventRecord(b,cudaStreamPerThread)); Cuda(cudaEventSynchronize(b));float ms=0;
    Cuda(cudaEventElapsedTime(&ms,a,b)); Cuda(cudaEventDestroy(a)); Cuda(cudaEventDestroy(b)); return ms*5;
}
void Case(int n,int cap,bool indirect,bool scaled,bool paged=false) {
    Test t(n,cap,indirect,scaled);
    if(paged) for(State *state : {&t.actual,&t.reference,&t.alternate}) state->Paged(cap);
    t.Legacy(); Require(t.Fused(),"fused admission failed");
    double l2=t.Check();
    auto old=Capture([&]{t.Legacy();}); auto fused=Capture([&]{Require(t.Fused(),"capture rejected");});
    float oldUs=Time(old),newUs=Time(fused);t.Check();
    for(int length:{0,1,17,cap}) {
        t.Lengths(std::min(length,cap)); t.Legacy(); Cuda(cudaGraphLaunch(fused,cudaStreamPerThread));t.Check();
    }
    if(indirect) {
        auto oldKeys=Read(*t.actual.keys[0]); std::vector<const void*> pointers;
        for(int r=0;r<n;++r) {
            Cuda(cudaMemcpy(t.alternate.keys[r]->cudaData,t.reference.keys[r]->cudaData,t.reference.keys[r]->GetBytes(),cudaMemcpyDeviceToDevice));
            Cuda(cudaMemcpy(t.alternate.values[r]->cudaData,t.reference.values[r]->cudaData,t.reference.values[r]->GetBytes(),cudaMemcpyDeviceToDevice));
            pointers.insert(pointers.end(),{t.alternate.keys[r]->cudaData,t.alternate.values[r]->cudaData,
                paged ? t.alternate.rows[r].key->cudaData : nullptr, paged ? t.alternate.rows[r].value->cudaData : nullptr});
        }
        Cuda(cudaMemcpy(t.actual.table.cudaData,pointers.data(),n*4*sizeof(void*),cudaMemcpyHostToDevice));
        t.Lengths(std::min(cap,31));t.Legacy();Cuda(cudaGraphLaunch(fused,cudaStreamPerThread));t.Check(true);
        Require(oldKeys==Read(*t.actual.keys[0]),"table rebinding touched old request");
    }
    Cuda(cudaMemset(t.actual.scratch.dsaFrontReady.cudaData, 0xff, t.actual.scratch.dsaFrontReady.GetBytes()));
    t.Lengths(cap); t.Legacy(); Cuda(cudaGraphLaunch(fused,cudaStreamPerThread));t.Check(indirect);
    std::vector<unsigned> tags(t.actual.scratch.dsaFrontReady.Count(0));
    Cuda(cudaMemcpy(tags.data(),t.actual.scratch.dsaFrontReady.cudaData,tags.size()*4,cudaMemcpyDeviceToHost));
    Require(std::all_of(tags.begin(),tags.end(),[](unsigned x){return x==0;}),"generation wrap left stale rows");
    auto saved=Read(t.actual.q); t.capacities[0]=2049;Require(!t.Fused(),"oversized capacity accepted");
    t.capacities[0]=cap;t.weight.strides[1]=2;Require(!t.Fused(),"bad stride accepted");t.weight.strides[1]=1;
    auto dtype=t.weight.dataType;t.weight.dataType=FLOAT32;Require(!t.Fused(),"wrong dtype accepted");t.weight.dataType=dtype;
    void *out=t.actual.out.cudaData;t.actual.out.cudaData=t.actual.q.cudaData;
    Require(!t.Fused(),"aliased output accepted");t.actual.out.cudaData=out;
    if (paged) {
        auto *pool=t.actual.rows[0].key->pagedKVCacheData;t.actual.rows[0].key->pagedKVCacheData=nullptr;
        Require(!t.Fused(),"missing paged pool accepted");t.actual.rows[0].key->pagedKVCacheData=pool;
    } else {
        void *key=t.actual.rows[0].key->cudaData;t.actual.rows[0].key->cudaData=t.input.cudaData;
        Require(!t.Fused(),"cache aliases input");t.actual.rows[0].key->cudaData=key;
    }
    Require(saved==Read(t.actual.q),"rejection wrote output");
    Cuda(cudaGraphExecDestroy(old));Cuda(cudaGraphExecDestroy(fused));
    printf("rows=%d capacity=%d indirect=%d scaled=%d paged=%d legacy_us=%.3f fused_us=%.3f attention_l2=%.6g\n",n,cap,indirect,scaled,paged,oldUs,newUs,l2);fflush(stdout);
}
void VerifyCase(int n, int group) {
    Test t(n,512,true,false); t.ordered = true;
    for (State *state : {&t.actual,&t.reference,&t.alternate}) {
        std::vector<const void *> pointers;
        for (int row=0; row<n; ++row) {
            const int base=row/group*group;
            state->rows[row].key=state->keys[base].get();
            state->rows[row].value=state->values[base].get();
            state->rows[row].cache=(const void *const *)state->table.cudaData+base*4;
            pointers.insert(pointers.end(),{state->keys[row]->cudaData,state->values[row]->cudaData,nullptr,nullptr});
        }
        Cuda(cudaMemcpy(state->table.cudaData,pointers.data(),pointers.size()*sizeof(void*),cudaMemcpyHostToDevice));
    }
    auto lengths=[&](int past, bool inactive) {
        std::vector<int> live(n);std::vector<float> pos(n);
        for(int row=0;row<n;++row) {
            live[row]=inactive && row>=group ? 0 : past+row%group+1;
            pos[row]=std::max(0,live[row]-1);
        }
        Cuda(cudaMemcpy(t.live.cudaData,live.data(),n*4,cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(t.positions.cudaData,pos.data(),n*4,cudaMemcpyHostToDevice));
    };
    lengths(119,false);
    if (n == 8) { Require(!t.Fused(),"eight-row ordered Front must keep GEMM fallback"); return; }
    t.Legacy();Require(t.Fused(),"verification admission rejected");t.Check();
    auto graph=Capture([&]{Require(t.Fused(),"verification capture rejected");});
    double maximum=0;
    for(int past:{0,1,119,127,128,503})for(bool inactive:{false,true}) {
        lengths(past,inactive);t.Legacy();Cuda(cudaGraphLaunch(graph,cudaStreamPerThread));
        maximum=std::max(maximum,t.Check());
    }
    Cuda(cudaMemset(t.actual.scratch.dsaFrontReady.cudaData,0xff,t.actual.scratch.dsaFrontReady.GetBytes()));
    lengths(127,false);t.Legacy();Cuda(cudaGraphLaunch(graph,cudaStreamPerThread));t.Check();
    auto old=Capture([&]{t.Legacy();});
    printf("verify_rows=%d group=%d legacy_us=%.3f fused_us=%.3f max_l2=%.6g\n",n,group,Time(old),Time(graph),maximum);
    Cuda(cudaGraphExecDestroy(old));Cuda(cudaGraphExecDestroy(graph));
}
}
int main(int argc,char **argv) {
    try {
        int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
        FastllmCudaSetDevice(0);if(FastllmCudaRuntimeArch()!=103)return 77;
        bool quick=argc>1;
        for(int n:{1,2,3,4,5,6,7,8}) { VerifyCase(n,n); if(n>2) VerifyCase(n,3); }
        for(int n:{1,2,3,4,5,6,7,8})for(int cap:(quick?std::vector<int>{512}:std::vector<int>{1,128,512,1024,2048}))
            Case(n,cap,true,cap%2==0);
        if (!quick) {
            { Test oversized(9,128,false,false); auto saved=Read(oversized.actual.q);
              Require(!oversized.Fused(),"nine rows accepted"); Require(saved==Read(oversized.actual.q),"batch rejection wrote output"); }
            for (int n : {1,3,8}) for (bool scaled : {false,true}) Case(n,512,true,scaled,true);
            std::exception_ptr errors[2];
            std::thread workers[2];
            for(int i=0;i<2;++i) workers[i]=std::thread([&,i]{
                try { FastllmCudaSetDevice(0); Case(i+1,128,true,true,true); VerifyCase(i ? 7 : 3,3); }
                catch(...) { errors[i]=std::current_exception(); }
            });
            for(auto &worker:workers) worker.join();
            for(auto &error:errors) if(error)std::rethrow_exception(error);
        }
        printf("PASS: fused DSA reference, fixed Graph replay and fallback\n"); return 0;
    } catch(const std::exception &e) {fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
