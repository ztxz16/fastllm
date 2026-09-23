#include <cuda_runtime.h>
#define FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace fastllm;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
#define CUDA(x) CHECK((x) == cudaSuccess)

static float roundHalf(float x) { return half_to_float(float_to_half(x)); }
static float qvalue(int h, int t, int d) {
    return roundHalf(float((h*37+t*19+d*11)%127-63)/80.f);
}
static float kvalue(int h, int t, int d) {
    return roundHalf(float((h*17+t*23+d*7)%109-54)/80.f);
}
static float vvalue(int h, int t, int d) {
    return roundHalf(float((h*13+t*31+d*17)%113-56)/96.f);
}

template<class F> static void init(Data &x, int heads, int rows, int capacity, int device, F value) {
    x.Expansion({heads,capacity,128});
    x.Resize({heads,rows,128});
    auto *host=(uint16_t*)x.cpuData;
    std::fill_n(host,x.expansionBytes/2,float_to_half(NAN));
    for(int h=0;h<heads;h++) for(int t=0;t<rows;t++) for(int d=0;d<128;d++)
        host[h*x.strides[0]+t*128+d]=float_to_half(value(h,t,d));
    x.ToDevice(DataDevice::CUDA,{device},true);
}

static std::vector<float> read(Data &x) {
    std::vector<uint16_t> raw(x.Count(0));
    FastllmCudaCopyFromDeviceToHost(raw.data(),x.cudaData,raw.size()*2);
    std::vector<float> result(raw.size());
    for(size_t i=0;i<raw.size();i++) result[i]=half_to_float(raw[i]);
    return result;
}

static void runCase(int queries,int cached,int runtime,int window,int group,int kvHeads,int device,bool bench=false) {
    const int heads=group*kvHeads,keys=cached+queries;
    Data q(FLOAT16),k(FLOAT16),v(FLOAT16),o(FLOAT16);
    init(q,heads,queries,queries,device,qvalue);
    init(k,kvHeads,keys,keys+17,device,kvalue);
    init(v,kvHeads,keys,keys+37,device,vvalue);
    o.Resize({heads,queries,128});o.dataDevice=CUDA;o.dataDeviceIds={device};o.Allocate();
    const float scale=1.f/std::sqrt(128.f);
    auto launch=[&] { CHECK(FastllmCudaDFlashAttention(q,k,v,o,group,scale,runtime,window)); };
    setenv("FASTLLM_DFLASH_ATTENTION","0",1);launch(); auto baseline=read(o);
    setenv("FASTLLM_DFLASH_ATTENTION","1",1);launch(); auto actual=read(o);
    // The default must select the same implementation as explicit opt-in,
    // including unsupported shapes that fall back to cuBLAS.
    unsetenv("FASTLLM_DFLASH_ATTENTION");launch(); auto automatic=read(o);
    CHECK(automatic == actual);
    double sqError=0,baseSqError=0,sqRef=0,maxError=0,baseMax=0;
    for(int h=0;h<heads;h++) for(int t=0;t<queries;t++) {
        int lo=std::max(0,cached+t-window+1),hi=std::min(cached+runtime,cached+t+window);
        std::vector<double> scores(hi-lo);double maxScore=-INFINITY,sum=0;
        for(int j=lo;j<hi;j++) {
            double dot=0;for(int d=0;d<128;d++) dot+=double(qvalue(h,t,d))*kvalue(h/group,j,d);
            scores[j-lo]=dot*scale;maxScore=std::max(maxScore,scores[j-lo]);
        }
        for(double &s:scores){s=std::exp(s-maxScore);sum+=s;}
        for(int d=0;d<128;d++) {
            double ref=0;for(int j=lo;j<hi;j++) ref+=scores[j-lo]*vvalue(h/group,j,d);ref/=sum;
            size_t i=((size_t)h*queries+t)*128+d;
            CHECK(std::isfinite(actual[i]) && std::isfinite(baseline[i]));
            double e=actual[i]-ref,b=baseline[i]-ref;
            sqError+=e*e;baseSqError+=b*b;sqRef+=ref*ref;
            maxError=std::max(maxError,std::abs(e));baseMax=std::max(baseMax,std::abs(b));
        }
    }
    double rrms=std::sqrt(sqError/std::max(sqRef,1.e-20));
    printf("CASE q=%d cached=%d runtime=%d window=%d group=%d kvh=%d rrms=%.8g max=%.8g base_rrms=%.8g base_max=%.8g\n",
        queries,cached,runtime,window,group,kvHeads,rrms,maxError,std::sqrt(baseSqError/std::max(sqRef,1.e-20)),baseMax);
    // Periodic, signed V can make the reference almost zero after a long
    // window. Use an absolute + relative RMS bound rather than dividing
    // FP16 rounding noise by an arbitrarily small reference norm.
    const double elements=actual.size();
    CHECK(std::sqrt(sqError/elements)<1.e-5+0.003*std::sqrt(sqRef/elements));
    CHECK(maxError<0.001);
    if(bench) {
        cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));
        for(int state:{0,1,1,0}) {
            setenv("FASTLLM_DFLASH_ATTENTION",state?"1":"0",1);
            for(int i=0;i<10;i++)launch();
            CUDA(cudaEventRecord(a,cudaStreamPerThread));
            for(int i=0;i<100;i++)launch();
            CUDA(cudaEventRecord(b,cudaStreamPerThread));CUDA(cudaEventSynchronize(b));
            float ms;CUDA(cudaEventElapsedTime(&ms,a,b));
            printf("BENCH q=%d cached=%d state=%d us=%.4f\n",queries,cached,state,ms*10);
        }
        CUDA(cudaEventDestroy(a));CUDA(cudaEventDestroy(b));
    }
}

int main(int argc,char **argv) {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
        int device=argc>1?std::atoi(argv[1]):0;CUDA(cudaSetDevice(device));
        cudaDeviceProp prop;CUDA(cudaGetDeviceProperties(&prop,device));
        if(prop.major!=7 || prop.minor!=5) return 77;
        // Capacity padding poisons all unused slots. These cases cover growing
        // caches, window crossing, and rollback represented by partial blocks.
        for(int q:{1,2,4,8,16}) for(int cached:{0,1,127,128,129,2047,2048,2177})
            runCase(q,cached,std::max(1,q-2),2048,4,2,device);
        for(int runtime:{1,3,8}) for(int window:{8,128,2048,4096})
            runCase(8,window+129,runtime,window,4,2,device);
        for(int group:{1,2,8})runCase(8,257,8,128,group,2,device);
        // Unsupported query count goes through the complete baseline fallback.
        runCase(17,257,15,128,4,2,device);
        for(int cached:{128,512,2048,4096})
            runCase(8,cached,8,2048,4,8,device,!std::getenv("DFLASH_ATTN_NO_BENCH"));
        CUDA(cudaDeviceSynchronize());printf("PASS DFlash SM75 attention device=%d\n",device);
    } catch(const std::exception &e) {fprintf(stderr,"FAIL %s\n",e.what());return 1;}
    return 0; // Also valid when the sanitizer loader renames main for dlopen.
}
