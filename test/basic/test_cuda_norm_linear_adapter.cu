#include "devices/cuda/fastllm-cuda-norm-linear.h"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool value,const char *message) { if(!value)throw std::runtime_error(message); }
static void Cuda(cudaError_t error) { Check(error==cudaSuccess,cudaGetErrorString(error)); }
static void Init(Data &d,DataType type,const std::vector<int> &dims) {
    d.dataType=type;d.Resize(dims);d.ToDevice(DataDevice::CUDA,{0},false);d.Allocate(false);
}
static std::vector<float> Read(const Data &d) {
    std::vector<float> values(d.Count(0));Cuda(cudaMemcpy(values.data(),d.cudaData,d.GetBytes(),cudaMemcpyDeviceToHost));return values;
}
int main() {
    try {
        int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
        Cuda(cudaSetDevice(0));cudaDeviceProp p{};Cuda(cudaGetDeviceProperties(&p,0));if(p.major<7)return 77;
        cudaStream_t stream;Cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for(int rows:{1,2,9}) {
            constexpr int K=256,N=17;Data x,r,g,w,out;
            Init(x,BFLOAT16,{1,rows,K});Init(r,BFLOAT16,x.dims);Init(g,FLOAT32,{K});
            Init(w,BFLOAT16,{N,K});Init(out,FLOAT32,{1,rows,N});
            std::vector<__nv_bfloat16> hx(rows*K),hr(rows*K),hw(N*K);std::vector<float> hg(K);
            for(int i=0;i<rows*K;++i){hx[i]=__nv_bfloat16((i%53-26)*.03125f);hr[i]=__nv_bfloat16((i%13-6)*.0078125f);}
            for(int i=0;i<N*K;++i)hw[i]=__nv_bfloat16((i%43-21)*.015625f);
            for(int i=0;i<K;++i)hg[i]=1+(i%17-8)*.0078125f;
            Cuda(cudaMemcpy(x.cudaData,hx.data(),x.GetBytes(),cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(r.cudaData,hr.data(),r.GetBytes(),cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(w.cudaData,hw.data(),w.GetBytes(),cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(g.cudaData,hg.data(),g.GetBytes(),cudaMemcpyHostToDevice));
            fastllm::cuda::low_latency::NormLinearPlan plan;
            Cuda(FastllmCudaPrepareNormLinear(x,&r,g,w,out,plan,stream));
            cudaGraph_t graph;cudaGraphExec_t exec;Cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            Cuda(FastllmCudaRunNormLinear(plan,x,&r,g,w,out,1e-5f,stream));
            Cuda(cudaStreamEndCapture(stream,&graph));Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            for(int replay=0;replay<4;++replay)Cuda(cudaGraphLaunch(exec,stream));
            Cuda(cudaStreamSynchronize(stream));auto actual=Read(out);
            for(int row=0;row<rows;++row) {
                float h[K];double sum=0;
                for(int k=0;k<K;++k){h[k]=float(__nv_bfloat16(float(hx[row*K+k])+float(hr[row*K+k])));sum+=double(h[k])*h[k];}
                const float inv=float(1/std::sqrt(sum/K+1e-5f));
                for(int k=0;k<K;++k)h[k]=float(__nv_bfloat16(float(__nv_bfloat16(h[k]*inv))*hg[k]));
                double error=0,mag=0;
                for(int n=0;n<N;++n){double dot=0;for(int k=0;k<K;++k)dot+=double(h[k])*float(hw[n*K+k]);
                    const float expected=float(__nv_bfloat16(float(dot)));error+=std::pow(double(actual[row*N+n])-expected,2);mag+=double(expected)*expected;}
                Check(std::sqrt(error/(mag+1e-30))<.001,"adapter row L2");
            }
            x.dataType=FLOAT16;
            Check(FastllmCudaRunNormLinear(plan,x,&r,g,w,out,1e-5f,stream)==cudaErrorInvalidValue,"unsupported dtype ran");
            x.dataType=BFLOAT16;
            auto shape=out.dims;out.dims={rows,N};
            Check(FastllmCudaRunNormLinear(plan,x,&r,g,w,out,1e-5f,stream)==cudaErrorInvalidValue,"changed shape ran");out.dims=shape;
            auto address=out.cudaData;out.cudaData=x.cudaData;
            Check(FastllmCudaRunNormLinear(plan,x,&r,g,w,out,1e-5f,stream)==cudaErrorInvalidValue,"alias ran");out.cudaData=address;
            Cuda(cudaStreamSynchronize(stream));Check(actual==Read(out),"rejected adapter call wrote output");
            Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
        }
        Cuda(cudaStreamDestroy(stream));std::puts("PASS norm linear Data adapter");return 0;
    } catch(const std::exception &e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
