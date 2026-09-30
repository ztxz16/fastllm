#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>
#include "dflash_rope_reference.cuh"

using namespace fastllm;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
#define CU(x) CHECK((x)==cudaSuccess)
static size_t elements=0,mismatches=0,cases=0;
static double maxError=0,errorEnergy=0,referenceEnergy=0;
static std::vector<float> Values(size_t n,int seed,float scale=1.0f) {
    std::mt19937 rng(seed);std::uniform_real_distribution<float> dist(-scale,scale);
    std::vector<float> v(n);for(float &x:v)x=dist(rng);return v;
}
static void Fill(Data &x,DataType type,std::vector<int> shape,const std::vector<float> &values) {
    x.CopyFrom(Data(type,shape,values));x.ToDevice(DataDevice::CUDA,{0},true);
}
static void Output(Data &x,std::vector<int> shape) {
    x.dataType=FLOAT16;x.UpdateUnitSize();x.Resize(shape);x.dataDevice=DataDevice::CUDA;x.dataDeviceIds={0};x.Allocate(false);
}
static std::vector<uint16_t> ReadHalf(const Data &x) {
    std::vector<uint16_t> v(x.Count(0));CU(cudaMemcpy(v.data(),x.cudaData,v.size()*2,cudaMemcpyDeviceToHost));return v;
}
static float Half(uint16_t u) {half x;memcpy(&x,&u,2);return __half2float(x);}
static void Compare(const std::vector<uint16_t> &a,const std::vector<uint16_t> &b,bool exact=false) {
    CHECK(a.size()==b.size());
    for(size_t i=0;i<a.size();++i) {
        float x=Half(a[i]),y=Half(b[i]);CHECK(std::isfinite(x)&&std::isfinite(y));
        double d=std::abs(double(x)-y);maxError=std::max(maxError,d);
        errorEnergy+=d*d;referenceEnergy+=double(y)*y;++elements;
        if(a[i]!=b[i])++mismatches;
        CHECK(!exact || a[i]==b[i]);
        CHECK(d<=2.e-5+std::abs(y)*0.0079);
    }
}
struct Rope {
    Data pos,refPos,inv,sin,cos;
    std::vector<float> frequency;
    int tokens,dim;
    Rope(int t,int d,int start,float theta):tokens(t),dim(d) {
        frequency.resize(dim/2);
        for(int i=0;i<dim/2;++i)frequency[i]=1.0f/std::pow(theta,float(2*i)/float(dim));
        Fill(inv,FLOAT32,{dim/2},frequency);Set(start);
    }
    void Set(int start) {
        std::vector<float> p(tokens),rp(tokens),s(size_t(tokens)*dim),c(s.size());
        for(int t=0;t<tokens;++t) {
            p[t]=start+t;rp[t]=t;
            for(int i=0;i<dim/2;++i) {
                float angle=p[t]*frequency[i];s[size_t(t)*dim+i]=std::sin(angle);c[size_t(t)*dim+i]=std::cos(angle);
            }
        }
        if(pos.cudaData && pos.dims==std::vector<int>({1,tokens})) {
            CU(cudaMemcpyAsync(pos.cudaData,p.data(),p.size()*4,cudaMemcpyHostToDevice,cudaStreamPerThread));
            CU(cudaStreamSynchronize(cudaStreamPerThread));
        } else Fill(pos,FLOAT32,{1,tokens},p);
        Fill(refPos,FLOAT32,{1,tokens},rp);Fill(sin,FLOAT32,{tokens,dim},s);Fill(cos,FLOAT32,{tokens,dim},c);
    }
};
static void GraphReplay(const std::function<void()> &launch,const std::function<void()> &change) {
    launch();CU(cudaStreamSynchronize(cudaStreamPerThread));
    cudaGraph_t graph;cudaGraphExec_t exec;
    CU(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch();
    CU(cudaStreamEndCapture(cudaStreamPerThread,&graph));CU(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    change();CU(cudaGraphLaunch(exec,cudaStreamPerThread));CU(cudaStreamSynchronize(cudaStreamPerThread));
    CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));
}
static double Time(const std::function<void()> &launch) {
    cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));
    for(int i=0;i<10;++i)launch();
    CU(cudaEventRecord(a,cudaStreamPerThread));for(int i=0;i<200;++i)launch();CU(cudaEventRecord(b,cudaStreamPerThread));
    CU(cudaEventSynchronize(b));float ms=0;CU(cudaEventElapsedTime(&ms,a,b));CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));return ms*1000/200;
}
static void Qkv(int tokens,int start,float theta,bool graph=false,bool bench=false) {
    const int qh=32,kh=8,dim=128,cols=(qh+2*kh)*dim;
    Data input,qn,kn,q,k,v,rq,rk,rv;Rope rope(tokens,dim,start,theta);
    Fill(input,BFLOAT16,{1,tokens,cols},Values(size_t(tokens)*cols,17+tokens));
    auto norm=Values(dim,11,.25f);for(float &x:norm)x+=1;
    Fill(qn,FLOAT32,{dim},norm);Fill(kn,FLOAT32,{dim},norm);
    for(Data *x:{&q,&rq})Output(*x,{qh,tokens,dim});
    for(Data *x:{&k,&v,&rk,&rv})Output(*x,{kh,tokens,dim});
    auto actual=[&]{CHECK(FastllmCudaDFlashPrepareQKV(input,qn,kn,rope.pos,rope.inv,q,k,v,tokens,qh,kh,dim,1.e-6f));};
    auto reference=[&]{reference::FastllmDFlashPrepareQkvBf16Kernel<<<tokens*(qh+2*kh),64,0,cudaStreamPerThread>>>(
        (const __nv_bfloat16*)input.cudaData,(const float*)qn.cudaData,(const float*)kn.cudaData,
        (const float*)rope.refPos.cudaData,(const float*)rope.sin.cudaData,(const float*)rope.cos.cudaData,
        (half*)rq.cudaData,(half*)rk.cudaData,(half*)rv.cudaData,tokens,cols,qh,kh,dim,dim,1.e-6f);};
    if(graph)GraphReplay(actual,[&]{rope.Set(start+65536);});else actual();
    reference();CU(cudaDeviceSynchronize());Compare(ReadHalf(q),ReadHalf(rq));Compare(ReadHalf(k),ReadHalf(rk));Compare(ReadHalf(v),ReadHalf(rv),true);++cases;
    if(bench)printf("BENCH qkv T=%d old_us=%.3f new_us=%.3f\n",tokens,Time(reference),Time(actual));
}
static void Kv(int tokens,int start,float theta,int layers=5,bool graph=false,bool bench=false) {
    const int kh=8,dim=128,cols=layers*2*kh*dim;
    Data input,norm,output,ref;Rope rope(tokens,dim,start,theta);
    Fill(input,BFLOAT16,{1,tokens,cols},Values(size_t(tokens)*cols,47+tokens));
    auto weights=Values(layers*dim,7,.25f);for(float &x:weights)x+=1;
    Fill(norm,FLOAT32,{layers,dim},weights);Output(output,{layers,2,kh,tokens,dim});Output(ref,{layers,2,kh,tokens,dim});
    auto actual=[&]{CHECK(FastllmCudaDFlashMaterializeKV(input,norm,rope.pos,rope.inv,output,layers,tokens,kh,dim,1.e-6f));};
    auto reference=[&]{reference::FastllmDFlashKvCacheOutput none={};reference::FastllmDFlashMaterializeKvBf16Kernel<<<layers*tokens*kh*2,64,0,cudaStreamPerThread>>>(
        (const __nv_bfloat16*)input.cudaData,(const float*)norm.cudaData,(const float*)rope.refPos.cudaData,
        (const float*)rope.sin.cudaData,(const float*)rope.cos.cudaData,(half*)ref.cudaData,none,tokens,cols,kh,dim,dim,1.e-6f);};
    if(graph)GraphReplay(actual,[&]{rope.Set(start+65536);});else actual();
    reference();CU(cudaDeviceSynchronize());auto expected=ReadHalf(ref);Compare(ReadHalf(output),expected);++cases;
    if(layers==5) {
        constexpr int prefix=7,extra=13;const int capacity=prefix+tokens+extra;
        std::vector<Data> caches(10);std::vector<Data*> pointers;
        for(auto &cache:caches) {
            cache.dataType=FLOAT16;cache.UpdateUnitSize();cache.Expansion({kh,capacity,dim});cache.Resize({kh,prefix,dim});
            std::fill_n((uint16_t*)cache.cpuData,cache.expansionBytes/2,uint16_t(0x5555));cache.ToDevice(DataDevice::CUDA,{0},true);pointers.push_back(&cache);
        }
        auto direct=[&]{CHECK(FastllmCudaDFlashMaterializeKVToCache(input,norm,rope.pos,rope.inv,pointers,layers,tokens,kh,dim,1.e-6f));};
        if(graph)GraphReplay(direct,[]{});else direct();CU(cudaDeviceSynchronize());
        for(int j=0;j<10;++j) {
            std::vector<uint16_t> raw(size_t(kh)*capacity*dim);CU(cudaMemcpy(raw.data(),caches[j].cudaData,raw.size()*2,cudaMemcpyDeviceToHost));
            std::vector<uint16_t> got,referenceValues;got.reserve(kh*tokens*dim);referenceValues.reserve(got.capacity());
            for(int h=0;h<kh;++h)for(int t=0;t<capacity;++t)for(int c=0;c<dim;++c) {
                uint16_t value=raw[(size_t(h)*capacity+t)*dim+c];
                if(t<prefix||t>=prefix+tokens)CHECK(value==0x5555);
                else {got.push_back(value);referenceValues.push_back(expected[((size_t(j)*kh+h)*tokens+t-prefix)*dim+c]);}
            }
            Compare(got,referenceValues,j%2==1);
        }
        ++cases;
    }
    if(bench)printf("BENCH kv T=%d old_us=%.3f new_us=%.3f\n",tokens,Time(reference),Time(actual));
}
static void Fallback(int tokens,int dim,int start,float theta,DataType type) {
    const int heads=3;Data input,ref;Rope rope(tokens,dim,start,theta);
    auto values=Values(size_t(tokens)*heads*dim,61+tokens);Fill(input,type,{1,tokens,heads,dim},values);Fill(ref,type,{1,tokens,heads,dim},values);
    CHECK(FastllmCudaDFlashApplyRope(input,rope.pos,rope.inv));
    CHECK(FastllmCudaLlamaRotatePosition2D(ref,rope.refPos,rope.sin,rope.cos,dim));CU(cudaDeviceSynchronize());
    const size_t n=input.Count(0);std::vector<float> a(n),b(n);
    if(type==FLOAT32) {CU(cudaMemcpy(a.data(),input.cudaData,n*4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(b.data(),ref.cudaData,n*4,cudaMemcpyDeviceToHost));}
    else {
        auto rawA=ReadHalf(input),rawB=ReadHalf(ref);
        for(size_t i=0;i<n;++i) {
            if(type==FLOAT16) {a[i]=Half(rawA[i]);b[i]=Half(rawB[i]);}
            else {uint32_t x=uint32_t(rawA[i])<<16,y=uint32_t(rawB[i])<<16;memcpy(&a[i],&x,4);memcpy(&b[i],&y,4);}
        }
    }
    for(size_t i=0;i<n;++i) {
        CHECK(std::isfinite(a[i])&&std::isfinite(b[i]));double d=std::abs(double(a[i])-b[i]);
        const double tolerance=type==FLOAT32?2.e-6:(type==FLOAT16?.001:.0079);
        CHECK(d<=2.e-6+tolerance*std::max(.02f,std::abs(b[i])));
        if(a[i]!=b[i])++mismatches;maxError=std::max(maxError,d);errorEnergy+=d*d;referenceEnergy+=double(b[i])*b[i];++elements;
    }
    ++cases;
}
int main() {
    try {
        int devices=0;
        if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) {
            puts("SKIP: no CUDA device");return 77;
        }
        FastllmCudaSetDevice(0);Executor executor;SetCurrentThreadExecutor(&executor);executor.SetFirstDevice("cuda:0");
        for(float theta:{10000.f,10000000.f}) {
            for(int position:{0,4093,65528,65535,65536,131065,203940,262128}) {
                for(int tokens=1;tokens<=16;++tokens) {Qkv(tokens,position,theta);Kv(tokens,position,theta);}
                for(int dim:{64,128,256})for(DataType type:{FLOAT32,FLOAT16,BFLOAT16})for(int tokens:{1,3,8,16})Fallback(tokens,dim,position,theta,type);
            }
            for(int tokens:{32,127,1024})for(int position:{4090,65500,203000}) {Qkv(tokens,position,theta);Kv(tokens,position,theta);}
        }
        for(int tokens:{1,2,7,8,16}) {Qkv(tokens,65532,10000000.f,true);Kv(tokens,65532,10000000.f,5,true);}
        Kv(8,65535,10000000.f,1);Kv(8,203940,10000000.f,7);
        for(int tokens:{1,8,1024}) {Qkv(tokens,65535,10000000.f,false,true);Kv(tokens,65535,10000000.f,5,false,true);}
        Data invalid(FLOAT32),positions,inv;
        CHECK(!FastllmCudaDFlashApplyRope(invalid,positions,inv));
        CHECK(!FastllmCudaDFlashPrepareQKV(invalid,invalid,invalid,positions,inv,invalid,invalid,invalid,0,1,1,128,1.e-6));
        double relativeRms=std::sqrt(errorEnergy/std::max(referenceEnergy,1.e-30));CHECK(relativeRms<1.e-4);
        printf("ALL_PASS cases=%zu elements=%zu mismatches=%zu max_abs=%.9g relative_rms=%.9g\n",cases,elements,mismatches,maxError,relativeRms);
        return 0;
    } catch(const std::exception &e) {fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
