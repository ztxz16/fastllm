#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void Require(bool ok, const char *s) { if (!ok) throw std::runtime_error(s); }
static void Allocate(fastllm::Data &d) {
    d.dataDevice = fastllm::DataDevice::CUDA; d.dataDeviceIds = {0}; d.Allocate();
}
static void Ints(fastllm::Data &d, const std::vector<int> &x) {
    d.dataType = fastllm::DataType::INT32; d.UpdateUnitSize();
    d.Resize({int(x.size())}); d.cpuIntDatas = x; Allocate(d);
    Check(cudaMemcpy(d.cudaData, x.data(), x.size()*sizeof(int), cudaMemcpyHostToDevice));
}

template<class T>
static void Run(fastllm::DataType queryType, fastllm::DataType cacheType) {
    using namespace fastllm;
    constexpr int D=256, H=4, QH=24, P=128, NP=128;
    constexpr size_t PE=P*H*D;
    const bool fp4=cacheType==DataType::FP4_E2M1;
    const size_t PB=fp4 ? PE/16*9 : PE;
    PagedCacheManager kp, vp;
    std::vector<float> decoded[2];
    for (int kv=0;kv<2;++kv) {
        auto &pool=kv ? vp : kp;
        pool.dataType=cacheType; pool.UpdateUnitSize(); pool.Resize({NP,P,H,D}); Allocate(pool);
        Require(pool.GetBytes()==NP*PB,"unexpected quantized KV layout");
        std::vector<unsigned char> raw(pool.GetBytes(),0);
        decoded[kv].resize(NP*PE);
        for (int p=0;p<NP;++p) for (size_t i=0;i<PE;++i) {
            if (fp4) {
                const float levels[]={0,.5f,1,1.5f,2,3,4,6};
                int code=int((i*17+i/37+p*3)%8);
                if (!kv && ((i/7+p)&1)) code|=8;
                __nv_fp8_e4m3 sf(.125f*(1+p%4));
                raw[p*PB+i/2]|=code<<((i&1)*4);
                raw[p*PB+PE/2+i/16]=sf.__x;
                decoded[kv][p*PE+i]=levels[code&7]*(code&8 ? -1.f : 1.f)*float(sf);
            } else {
                float value=kv ? .15f*(p%7)+.1f*std::sin(float(i)*.017f)
                               : .3f*std::cos(float(i)*.013f+p);
                __nv_fp8_e4m3 word(value); raw[p*PB+i]=word.__x;
                decoded[kv][p*PE+i]=float(word);
            }
        }
        Check(cudaMemcpy(pool.cudaData,raw.data(),raw.size(),cudaMemcpyHostToDevice));
    }
    Data k(cacheType),v(cacheType);
    std::vector<int> ids(NP);
    for (int i=0;i<NP;++i) ids[i]=NP-i-1;
    for (auto *c:{&k,&v}) {
        c->Resize({H,8,D}); c->isPagedKVCache=true; c->isFake=true;
        c->pageLen=P; c->pageIndex=ids; c->lastPageLen=8;
    }
    k.pagedKVCacheData=&kp; v.pagedKVCacheData=&vp;
    Data q(queryType,{QH,1,D}),out(queryType,{QH,1,D}); Allocate(q); Allocate(out);
    Data qs,ps,pi,ll; Ints(qs,{0,1}); Ints(ps,{0,NP}); Ints(pi,ids); Ints(ll,{8});
    std::vector<T> query(QH*D),actual(QH*D),eager(QH*D);
    for (size_t i=0;i<query.size();++i) query[i]=T(.2f*std::cos(float(i)*.027f));
    Check(cudaMemcpy(q.cudaData,query.data(),q.GetBytes(),cudaMemcpyHostToDevice));
    int initial[]={0,1}; Check(cudaMemcpy(ps.cudaData,initial,sizeof(initial),cudaMemcpyHostToDevice));
    auto launch=[&]() {
        out.Resize({QH,1,D});
        return FastllmCudaHalfPagedAttentionBatch(q,k,v,qs,ps,pi,ll,out,QH/H,
                                                 1.f/std::sqrt(float(D)),1,false,false,true,1);
    };
    Require(launch(),"quantized graph attention warmup failed"); Check(cudaDeviceSynchronize());
    cudaGraph_t graph; cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    Require(launch(),"quantized attention capture failed");
    Check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    Check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    float worst=0;
    // Grow, cross pages, then shrink to another request. Host metadata is never consulted by graph replay.
    for (int tokens:{8,18,127,128,129,255,256,257,2048,8192,10880,10881,16384,33}) {
        int pages[]={0,(tokens+P-1)/P},last=(tokens-1)%P+1;
        for (int i=0;i<NP;++i) ids[i]=(i*17+tokens%NP)%NP;
        for (size_t i=0;i<query.size();++i) query[i]=T(.2f*std::cos(float(i)*.027f+tokens*.01f));
        Check(cudaMemcpy(q.cudaData,query.data(),q.GetBytes(),cudaMemcpyHostToDevice));
        Check(cudaMemcpy(ps.cudaData,pages,sizeof(pages),cudaMemcpyHostToDevice));
        Check(cudaMemcpy(ll.cudaData,&last,sizeof(last),cudaMemcpyHostToDevice));
        Check(cudaMemcpy(pi.cudaData,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice));
        Check(cudaMemset(out.cudaData,0xff,out.GetBytes()));
        Check(cudaGraphLaunch(exec,cudaStreamPerThread)); Check(cudaStreamSynchronize(cudaStreamPerThread));
        Check(cudaMemcpy(actual.data(),out.cudaData,out.GetBytes(),cudaMemcpyDeviceToHost));
        // Compare both schedules with an independent oracle, not just each other.
        ps.cpuIntDatas={0,pages[1]}; ll.cpuIntDatas={last};
        out.Resize({QH,1,D});
        Require(FastllmCudaHalfPagedAttentionBatch(q,k,v,qs,ps,pi,ll,out,QH/H,
                    1.f/std::sqrt(float(D)),1,false,false,false,0), "eager attention failed");
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        Check(cudaMemcpy(eager.data(),out.cudaData,out.GetBytes(),cudaMemcpyDeviceToHost));
        // Independent FP64 softmax attention over the represented quantized words.
        float maxError=0,graphEagerError=0;
        for (int h=0;h<QH;++h) {
            std::vector<double> scores(tokens); double mx=-1e100,denom=0;
            for (int t=0;t<tokens;++t) {
                size_t off=size_t(ids[t/P])*PE+((t%P)*H+h/(QH/H))*D;
                double dot=0; for (int d=0;d<D;++d) dot+=double(float(query[h*D+d]))*decoded[0][off+d];
                scores[t]=dot/std::sqrt(double(D)); mx=std::max(mx,scores[t]);
            }
            for (double &score:scores) { score=std::exp(score-mx); denom+=score; }
            for (int d=0;d<D;++d) {
                double value=0;
                for (int t=0;t<tokens;++t) {
                    size_t off=size_t(ids[t/P])*PE+((t%P)*H+h/(QH/H))*D;
                    value+=scores[t]*decoded[1][off+d];
                }
                float got=float(actual[h*D+d]); Require(std::isfinite(got),"non-finite graph result");
                float other=float(eager[h*D+d]); Require(std::isfinite(other),"non-finite eager result");
                maxError=std::max(maxError,float(std::abs(got-value/denom)));
                maxError=std::max(maxError,float(std::abs(other-value/denom)));
                graphEagerError=std::max(graphEagerError,std::abs(got-other));
            }
        }
        worst=std::max(worst,maxError);
        std::printf("Q=%d KV=%d tokens=%d max_abs_error=%.7f graph_eager_delta=%.7f\n",
                    int(queryType),int(cacheType),tokens,maxError,graphEagerError);
        Require(maxError < (queryType==DataType::FLOAT16 ? .003f : .025f),"graph replay differs from CPU attention oracle");
    }
    Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
    std::printf("PASS Q=%d KV=%d worst=%.7f\n",int(queryType),int(cacheType),worst);
}
int main() {
    try {
        int count=0; if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) return 77;
        Check(cudaSetDevice(0));
        if (!FastllmCudaFlashInferSupported()) return 77;
        std::vector<fastllm::DataType> cacheTypes = {fastllm::DataType::FP8_E4M3};
#if CUDART_VERSION >= 12080
        // FlashInfer's FP4 graph dispatch requires CUDA 12.8 or newer.
        cacheTypes.push_back(fastllm::DataType::FP4_E2M1);
#endif
        for (auto kv : cacheTypes) {
            Run<half>(fastllm::DataType::FLOAT16,kv);
            Run<__nv_bfloat16>(fastllm::DataType::BFLOAT16,kv);
        }
    } catch (const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
    return 0;
}
