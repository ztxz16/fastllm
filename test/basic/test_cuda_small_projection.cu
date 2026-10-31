#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstring>
using namespace fastllm;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
static std::mt19937 rng(481);
static double maxL2=0;
static int cases=0;
static void allocate(Data &d, DataType type, std::vector<int> dims, const void *data) {
    d.dataType=type;d.UpdateUnitSize();d.Resize(dims);d.dataDevice=CUDA;d.dataDeviceIds={0};d.Allocate(false);
    if(data)CHECK(cudaMemcpy(d.cudaData,data,d.GetBytes(),cudaMemcpyHostToDevice)==cudaSuccess);
}
static void run(int rows,int n,int k,int groups,bool fp8,bool graph) {
    std::vector<__nv_bfloat16> x(rows*groups*k),w(groups*n*k),got(rows*groups*n);
    std::vector<uint8_t> q(w.size());
    std::vector<float> scales(groups*n/32*(k/32));
    for(auto &v:x)v=__float2bfloat16_rn(float(int(rng()%1001)-500)/257.f);
    for(auto &v:scales)v=std::ldexp(1.03125f,int(rng()%7)-8);
    for(int r=0;r<groups*n;r++)for(int c=0;c<k;c++) {
        size_t i=size_t(r)*k+c;
        q[i]=uint8_t((rng()%127)|((rng()&1)<<7));
        __nv_fp8_e4m3 f;f.__x=q[i];
        w[i]=__float2bfloat16_rn(fp8?float(f)*scales[(r/32)*(k/32)+c/32]:float(int(rng()%997)-498)/251.f);
    }
    Data input,weight,out;
    allocate(input,BFLOAT16,{rows,groups,k},x.data());
    allocate(weight,fp8?FP8_E4M3:BFLOAT16,{groups*n,k},fp8?(void*)q.data():(void*)w.data());
    if(fp8){weight.blockM=weight.blockK=32;weight.scales=scales;Data bias;FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight,bias,groups*n);}
    allocate(out,BFLOAT16,{rows,groups,n},nullptr);
    CHECK(TryCudaTritonSmallProjection(input,weight,out,groups));
    CHECK(cudaDeviceSynchronize()==cudaSuccess);
    if(graph){
        cudaGraph_t g;cudaGraphExec_t e;
        CHECK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
        CHECK(TryCudaTritonSmallProjection(input,weight,out,groups));
        CHECK(cudaStreamEndCapture(cudaStreamPerThread,&g)==cudaSuccess);
        CHECK(cudaGraphInstantiate(&e,g,0)==cudaSuccess);
        // Replay reads fresh operands, rather than a capture-time result.
        for(auto &v:x)v=__float2bfloat16_rn(float(v)*0.75f+0.125f);
        CHECK(cudaMemcpy(input.cudaData,x.data(),input.GetBytes(),cudaMemcpyHostToDevice)==cudaSuccess);
        CHECK(cudaGraphLaunch(e,cudaStreamPerThread)==cudaSuccess);
        CHECK(cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess);
        cudaGraphExecDestroy(e);cudaGraphDestroy(g);
    }
    CHECK(cudaMemcpy(got.data(),out.cudaData,out.GetBytes(),cudaMemcpyDeviceToHost)==cudaSuccess);
    double err=0,norm=0;
    for(int t=0;t<rows;t++)for(int g=0;g<groups;g++)for(int j=0;j<n;j++){
        double ref=0;for(int h=0;h<k;h++)ref+=double(float(x[(t*groups+g)*k+h]))*float(w[(g*n+j)*k+h]);
        float actual=float(got[(t*groups+g)*n+j]);CHECK(std::isfinite(actual));
        err+=(actual-ref)*(actual-ref);norm+=ref*ref;
    }
    double l2=std::sqrt(err/std::max(norm,1e-20));maxL2=std::max(maxL2,l2);CHECK(l2<0.004);++cases;
    // Unsupported policies must leave the fallback available.
    setenv("FASTLLM_CUDA_TRITON_SMALL_PROJECTION","0",1);CHECK(!TryCudaTritonSmallProjection(input,weight,out,groups));unsetenv("FASTLLM_CUDA_TRITON_SMALL_PROJECTION");
    CHECK(!TryCudaTritonSmallProjection(input,weight,input,groups));
    CHECK(!TryCudaTritonSmallProjection(input,weight,out,0));
    input.strides[0]++;CHECK(!TryCudaTritonSmallProjection(input,weight,out,groups));input.strides[0]--;
    if(fp8){weight.blockK=128;CHECK(!TryCudaTritonSmallProjection(input,weight,out,groups));weight.blockK=32;}
}
int main(int argc, char **argv){
    const bool wideOnly = argc == 2 && std::strcmp(argv[1], "--wide") == 0;
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    cudaSetDevice(0);FastllmCudaSetDevice(0);
    if(FastllmCudaRuntimeArch()!=120&&FastllmCudaRuntimeArch()!=121)return 77;
    setenv("FASTLLM_CUDA_TRITON","1",1);
    if (!wideOnly) for(bool fp8:{false,true})for(auto shape:std::vector<std::vector<int>>{{32,32},{96,160},{128,256},{512,1280}})
        for(int rows:{2,3,5,6,8,9,16})for(int groups:{1,2,8})run(rows,shape[0],shape[1],groups,fp8,rows==6||rows==16);
    // Wide FP8 tiles include a partial N tile and a masked K tail.
    for (int rows : {3, 6, 16}) run(rows, 16416, 96, 1, true, true);
    run(6, 32768, 1280, 1, true, true);
    std::printf("SMALL_PROJECTION_PASS cases=%d max_l2=%.10g\n",cases,maxL2);
}
