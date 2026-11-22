#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/fastllm-cuda-decode-linear.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>
using namespace fastllm;
static void Check(bool ok,const char *s) {if(!ok) throw std::runtime_error(s);}
static void Cuda(cudaError_t e) {Check(e==cudaSuccess,cudaGetErrorString(e));}
static void Init(Data &d,DataType type,std::vector<int> dims) {
    d.dataType=type;d.Resize(dims);d.ToDevice(DataDevice::CUDA,{0},false);d.Allocate(false);
}
static void Fill(Data &d,unsigned seed,float scale=1,float offset=0) {
    std::vector<unsigned char> bytes(d.GetBytes());
    for(size_t i=0;i<d.Count(0);++i) {
        seed=1664525u*seed+1013904223u;float x=(int(seed>>16)-32768)/32768.f*scale+offset;
        if(d.dataType==BFLOAT16) {uint16_t b=Float32ToBFloat16RNEBits(x);std::memcpy(bytes.data()+2*i,&b,2);}
        else std::memcpy(bytes.data()+4*i,&x,4);
    }
    Cuda(cudaMemcpy(d.cudaData,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
}
static std::vector<float> Read(const Data &d) {
    std::vector<unsigned char> raw(d.GetBytes());std::vector<float> out(d.Count(0));
    Cuda(cudaMemcpy(raw.data(),d.cudaData,raw.size(),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<out.size();++i) {
        if(d.dataType==BFLOAT16) {uint16_t b;std::memcpy(&b,raw.data()+i*2,2);out[i]=BFloat16BitsToFloat32(b);}
        else std::memcpy(&out[i],raw.data()+i*4,4);
    }
    return out;
}
static double Compare(const Data &a,const Data &b) {
    auto x=Read(a),y=Read(b);Check(x.size()==y.size(),"output shape mismatch");
    double err=0,mag=0;
    for(size_t i=0;i<x.size();++i) {Check(std::isfinite(x[i])&&std::isfinite(y[i]),"nonfinite projection");err+=std::pow(double(x[i])-y[i],2);mag+=double(y[i])*y[i];}
    double l2=std::sqrt(err/(mag+1e-30));if(l2>=.001) { std::fprintf(stderr,"projection L2=%.9g\n",l2); for(int i=0;i<8;++i) std::fprintf(stderr,"%d: %.9g %.9g\n",i,x[i],y[i]); } Check(l2<.001,"BF16 decode projection relative L2");return l2;
}
static double Time(const std::function<void()> &run) {
    for(int i=0;i<3;++i)run();Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    cudaGraph_t graph;cudaGraphExec_t exec;cudaEvent_t a,b;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<16;++i)run();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));Cuda(cudaGraphInstantiate(&exec,graph,0));
    Cuda(cudaEventCreate(&a));Cuda(cudaEventCreate(&b));
    for(int i=0;i<3;++i)Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
    Cuda(cudaEventRecord(a,cudaStreamPerThread));for(int i=0;i<10;++i)Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
    Cuda(cudaEventRecord(b,cudaStreamPerThread));Cuda(cudaEventSynchronize(b));float ms;
    Cuda(cudaEventElapsedTime(&ms,a,b));Cuda(cudaEventDestroy(a));Cuda(cudaEventDestroy(b));
    Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));return ms*1000/160;
}
struct Case {
    int rows,inner,channels;
    Data x,branch,gamma,w,out,ref,normed,headBf16,hidden,empty;
    Case(int r,int k,int n,unsigned seed=10):rows(r),inner(k),channels(n) {
        Init(x,BFLOAT16,{1,r,k});Init(branch,BFLOAT16,x.dims);Init(hidden,BFLOAT16,x.dims);
        Init(gamma,FLOAT32,{k});Init(w,BFLOAT16,{n,k});
        Init(normed,BFLOAT16,x.dims);Init(headBf16,BFLOAT16,{1,r,n});
        Init(ref,FLOAT32,{1,r,n});
        Fill(x,seed);Fill(branch,seed+1,.1f);Fill(gamma,seed+2,.1f,1);Fill(w,seed+3,.1f);
    }
    bool Fused(bool residual=false) {
        return FastllmCudaDecodeNormLinear(x,residual?&branch:nullptr,gamma,w,out,1e-5f);
    }
    void Legacy(bool residual=false) {
        if(residual) {
            Cuda(cudaMemcpyAsync(hidden.cudaData,x.cudaData,x.GetBytes(),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            FastllmCudaNaiveAddDecodeRMSNorm(hidden,branch,gamma,1e-5f,normed);
        } else Check(FastllmCudaKimiK3RMSNorm(x,gamma,normed,1e-5f),"reference norm");
        Check(FastllmCudaBFloat16MatMulBFloat16(normed,w,empty,headBf16,rows,inner,channels),"reference head");
        Check(FastllmBF16ToFloat(headBf16.cudaData,ref.cudaData,rows*channels),"reference cast");
    }
    void CheckWideReference() {
        auto input=Read(normed), weight=Read(w), actual=Read(ref);
        // Sample the whole vocabulary at a coprime stride. The independent
        // scalar FP32 dot is rounded to BF16 at the projection boundary.
        for(int row=0;row<rows;++row) {
            double error=0,magnitude=0;
            for(int j=0;j<512;++j) {
                const int col=(j*127)%channels;
                float dot=0;
                for(int k=0;k<inner;++k)dot+=input[row*inner+k]*weight[col*inner+k];
                const float expected=BFloat16BitsToFloat32(Float32ToBFloat16RNEBits(dot));
                error+=std::pow(double(actual[row*channels+col])-expected,2);
                magnitude+=double(expected)*expected;
            }
            Check(std::sqrt(error/(magnitude+1e-30))<.001,"wide Tensor Core projection row L2");
        }
    }

    void Mirror(int target) {
        FastllmCudaSetDevice(target);
        Data mirror;mirror.dataType=FLOAT32;mirror.Resize({rows+1,channels+32});
        mirror.ToDevice(DataDevice::CUDA,{target},false);mirror.Allocate(false);Fill(mirror,99,0,-1234);
        FastllmCudaSetDevice(0);
        static bool peerEnabled=false;
        if (target && !peerEnabled) {
            int capable = 0;
            Cuda(cudaDeviceCanAccessPeer(&capable, 0, target));
            if (capable) {
                auto status = cudaDeviceDisablePeerAccess(target);
                if (status == cudaErrorPeerAccessNotEnabled) cudaGetLastError(); else Cuda(status);
            }
            auto saved = Read(out);
            Check(!FastllmCudaDecodeNormLinear(x,&branch,gamma,w,out,1e-5f,&mirror,16),
                  "inaccessible peer mirror was admitted");
            Check(Read(out) == saved, "peer fallback changed the output");
            if (!capable) return;
            Cuda(cudaDeviceEnablePeerAccess(target,0));
            peerEnabled = true;
        }
        Check(!FastllmCudaDecodeNormLinear(x,&branch,gamma,w,out,1e-5f,&mirror,33),"mirror overflow accepted");
        auto run=[&]{Check(FastllmCudaDecodeNormLinear(x,&branch,gamma,w,out,1e-5f,&mirror,16),"mirror admission");};
        run();Time(run);Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        auto local=Read(out);FastllmCudaSetDevice(target);auto actual=Read(mirror);FastllmCudaSetDevice(0);
        for(int r=0;r<rows+1;++r)for(int c=0;c<channels+32;++c) {
            const float expected=r<rows && c>=16 && c<channels+16?local[r*channels+c-16]:-1234;
            Check(actual[r*(channels+32)+c]==expected,"mirror stride/row guard or visibility failure");
        }
    }
    void Reject() {
        auto saved=Read(out);auto dims=out.dims;
        auto stride=w.strides[0];++w.strides[0];Check(!Fused(),"bad stride accepted");w.strides[0]=stride;
        w.IsRepacked=true;Check(!Fused(),"packed weights accepted");w.IsRepacked=false;
        auto type=x.dataType;x.dataType=FLOAT16;Check(!Fused(),"unsupported dtype accepted");x.dataType=type;
        auto branchType=branch.dataType;branch.dataType=FLOAT16;
        Check(!Fused(true),"unsupported residual dtype accepted");branch.dataType=branchType;
        Check(!FastllmCudaDecodeNormLinear(x,nullptr,gamma,w,out,0),"invalid epsilon accepted");
        Check(!FastllmCudaDecodeNormLinear(x,nullptr,gamma,w,out,NAN),"nonfinite epsilon accepted");
        Check(saved==Read(out)&&dims==out.dims,"rejected operation changed output");
        Check(!FastllmCudaDecodeNormLinear(x,nullptr,gamma,w,x,1e-5f),"source alias accepted");
        Data alias;alias.Resize(out.dims);alias.FakeFrom(w,0);
        Check(!FastllmCudaDecodeNormLinear(x,nullptr,gamma,w,alias,1e-5f),"overlapping weight/output accepted");
    }
};
int main(int argc,char **argv) {
    try {
        int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
        FastllmCudaSetDevice(0);cudaDeviceProp p{};Cuda(cudaGetDeviceProperties(&p,0));if(p.major<8)return 77;
        const bool wideRows=p.major==10 && p.minor==3;
        const bool quick=argc>1 && std::string(argv[1])=="quick";
        for(int rows=1;rows<=8;++rows) {
            Case c(rows,4096,19072);
            if(wideRows && rows>=3 && rows<8) {
                Check(!c.Fused(),"wide head must use Tensor Cores");
                c.Legacy();c.CheckWideReference();
                c.Legacy(true);c.CheckWideReference();
                const double us=quick?0:Time([&]{c.Legacy();});
                std::printf("WIDE_PROJECTION rows=%d tensor_core_us=%.4f\n",rows,us);
                continue;
            }
            if(rows>7 || (rows>2 && !wideRows)) {Check(!c.Fused(),"GEMM head must fall back");continue;}
            Check(c.Fused(),"fused admission failed");c.Legacy();
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));double l2=Compare(c.out,c.ref);
            c.Reject();
            {Check(c.Fused(true),"residual admission");c.Legacy(true);Cuda(cudaStreamSynchronize(cudaStreamPerThread));l2=std::max(l2,Compare(c.out,c.ref));}
            c.Mirror(0);
            if (count > 1) c.Mirror(1);
            double fast=0,old=0;
            if(!quick) {fast=Time([&]{Check(c.Fused(),"fused replay");});old=Time([&]{c.Legacy();});}
            std::printf("PROJECTION rows=%d l2=%.8g fused_us=%.4f legacy_us=%.4f\n",rows,l2,fast,old);std::fflush(stdout);
        }
        for(int rows:{1,2,3,4,5,6,7}) for(int k:{256,1024,6144,8192}) for(int n:{16,48,1024}) {
            Case h(rows,k,n);Check(h.Fused(true),"generic head shape admission");h.Legacy(true);
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));Compare(h.out,h.ref);
        }
        for(auto shape:{std::pair<int,int>{384,48},{256,17}}) {
            Case c(1,shape.first,shape.second);Check(!c.Fused(),"unsupported shape accepted");
            Check(!c.out.cudaData,"rejected shape allocated output");
        }
        // Both shapes use the same Rows=7 kernel image. Preparing a narrower
        // opt-in shape must not invalidate the large graph's resource ceiling.
        if(wideRows) {
            Case large(7,7680,48), small(7,5120,48);
            Check(large.Fused(),"large opt-in head admission");
            cudaGraph_t graph; cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            Check(large.Fused(),"large opt-in head capture");
            Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            Cuda(cudaGraphInstantiate(&exec,graph,0)); Cuda(cudaGraphDestroy(graph));
            Check(small.Fused(),"small opt-in head admission");
            for(int replay=0;replay<8;++replay) Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
            large.Legacy(); Cuda(cudaStreamSynchronize(cudaStreamPerThread)); Compare(large.out,large.ref);
            Cuda(cudaGraphExecDestroy(exec));
        }
        std::exception_ptr errors[2];
        auto run=[&](int worker) {try {FastllmCudaSetDevice(0);Case c(wideRows ? (worker ? 7 : 3) : 2,worker ? 6144 : 4096,2048,100+worker);for(int i=0;i<10;++i)Check(c.Fused(),"concurrent projection");c.Legacy();Cuda(cudaStreamSynchronize(cudaStreamPerThread));Compare(c.out,c.ref);}catch(...){errors[worker]=std::current_exception();}};
        std::thread a(run,0),b(run,1);a.join();b.join();for(auto e:errors)if(e)std::rethrow_exception(e);
        std::puts("DECODE LINEAR PASS");return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
