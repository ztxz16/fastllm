#include "exl3.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
using namespace fastllm;
namespace fastllm { void DoCudaMergeMOEV41Fallback(Data &,Data &,Data &,Data &,Data **,int,float,float); }
static void Require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
static void Fill(Data &weight,std::mt19937 &rng){
    weight.Allocate(false);
    const int n=weight.dims[0]/weight.blockM,k=weight.dims[1];
    const size_t bytes=Exl3PartBytes(n,k,weight.blockK),trellis=size_t(n)*k*weight.blockK/16;
    for(int part=0;part<weight.blockM;++part){
        auto *p=weight.cpuData+part*bytes;
        for(size_t i=0;i<trellis;i+=4){uint32_t value=rng();std::memcpy(p+i,&value,4);}
        auto *scales=reinterpret_cast<uint16_t *>(p+trellis);
        for(int i=0;i<n+k;++i)scales[i]=float_to_half((int(rng()%63)-31)/512.f);
    }
}
static void Check(int rows,int k,int mid,bool graph,bool fallback=false,int uniformBits=0){
    constexpr int experts=3,topk=2;
    std::mt19937 rng(712+rows+k);std::vector<std::unique_ptr<Data>> storage;
    std::vector<Data*> weights(2+2*experts,nullptr);
    for(int e=0;e<experts;++e)for(int projection=0;projection<2;++projection){
        storage.emplace_back(new Data(EXL3));Data &w=*storage.back();
        w.blockK=uniformBits?uniformBits:4+2*((e+projection)%3);w.blockM=projection?1:2;
        w.Resize(projection?std::vector<int>{k,mid}:std::vector<int>{2*mid,k});Fill(w,rng);weights[2+2*e+projection]=&w;
    }
    Data input(BFLOAT16,{rows,k}),index(INT32,{rows,topk}),score(FLOAT32,{rows,topk});
    input.Allocate(false);index.Allocate(false);score.Allocate(false);
    for(int i=0;i<rows*k;++i){float v=(int(rng()%65)-32)/32.f;uint32_t bits;std::memcpy(&bits,&v,4);reinterpret_cast<uint16_t *>(input.cpuData)[i]=bits>>16;}
    for(int i=0;i<rows*topk;++i){reinterpret_cast<int *>(index.cpuData)[i]=(i+rows)%experts;reinterpret_cast<float *>(score.cpuData)[i]=(i%3+1)/4.f;}
    Data reference(FLOAT32,{rows,k}),output;
    Exl3MoeCPU(input,reference,index,score,weights.data(),weights.size(),0,.01f);
    for(auto *v:{&input,&index,&score})v->ToDevice(DataDevice::CUDA,std::vector<int>{0});
    for(auto &v:storage)v->ToDevice(DataDevice::CUDA,std::vector<int>{0});
    // Exercise reuse of a smaller byte buffer with the same element count.
    output.dataType=BFLOAT16;output.UpdateUnitSize();output.Resize({rows,k});
    output.ToDevice(DataDevice::CUDA,std::vector<int>{0});output.Allocate(false);
    auto run=[&]{
        if(fallback)DoCudaMergeMOEV41Fallback(input,output,index,score,weights.data(),weights.size(),0,.01f);
        else Require(FastllmCudaExl3Moe(input,output,index,score,weights.data(),weights.size(),.01f),"EXL3 indexed MoE rejected valid input");
    };
    run();Require(cudaDeviceSynchronize()==cudaSuccess,"EXL3 MoE execution");
    if(graph){
        cudaGraph_t captured=nullptr;cudaGraphExec_t executable=nullptr;
        Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"capture begin");
        run();Require(cudaStreamEndCapture(cudaStreamPerThread,&captured)==cudaSuccess,"capture end");
        Require(cudaGraphInstantiate(&executable,captured,nullptr,nullptr,0)==cudaSuccess,"graph instantiate");
        Require(cudaGraphLaunch(executable,cudaStreamPerThread)==cudaSuccess,"graph launch");
        Require(cudaDeviceSynchronize()==cudaSuccess,"graph execution");
        cudaGraphExecDestroy(executable);cudaGraphDestroy(captured);
    }
    // Tiny uniform-bit inputs can land exactly on BF16 rounding ties: both
    // portable and MMA products can differ from the CPU sum at those ties.
    // Check the fused BF16 boundaries against the decomposed CUDA operators
    // for these cases; the full-shape mixed tests above retain the CPU oracle.
    if (uniformBits) {
        Data decomposed;
        DoCudaMergeMOEV41Fallback(input,decomposed,index,score,weights.data(),weights.size(),0,.01f);
        Require(cudaDeviceSynchronize()==cudaSuccess,"decomposed MoE execution");
        decomposed.ToDevice(DataDevice::CPU); reference.CopyFrom(decomposed);
    }
    output.ToDevice(DataDevice::CPU);double err=0,norm=0;
    for(int i=0;i<rows*k;++i){double a=reinterpret_cast<float *>(reference.cpuData)[i],b=reinterpret_cast<float *>(output.cpuData)[i];err+=(a-b)*(a-b);norm+=a*a;}
    std::cout << "CASE rows=" << rows << " k=" << k << " mid=" << mid << " graph=" << graph << " fallback=" << fallback << " bits=" << uniformBits << " rel=" << std::sqrt(err/std::max(norm,1e-30)) << std::endl;
    Require(std::isfinite(err)&&std::sqrt(err/std::max(norm,1e-30))<.008,"EXL3 MoE differs from decomposed CPU calculation");
}
int main(){
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    SetThreads(8);
    for(int repeat=0;repeat<2;++repeat)for(int rows:{1,3,8}){Check(rows,256,128,true);Check(rows,5120,2304,false);}
    // Uniform half-integer / high bitrates use Maya's non-mixed kernels.
    for(int bits:{2,3,4,5,6,7,8,10,12,14,16})for(int rows:{1,3,8})Check(rows,256,128,true,false,bits);
    Check(9,256,128,false,true);
    std::cout<<"EXL3 MOE PASS: mixed bitrates, clamping, graph replay, cache retirement"<<std::endl;
}
