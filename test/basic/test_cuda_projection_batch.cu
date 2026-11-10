// Row-tiled FP8/BF16 projections: independent CPU oracle and changing-input graph replay.
#include "utils/kernel_registry.h"
#include "devices/cuda/fastllm-kernel-benchmark.cuh"
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
using namespace fastllm;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
static std::mt19937 rng(481);
static double maxL2=0;
static int cases=0;
static void allocate(Data &d, DataType type, std::vector<int> dims, const void *data) {
    d.dataType=type;d.UpdateUnitSize();d.Resize(dims);d.dataDevice=CUDA;d.dataDeviceIds={0};d.Allocate(false);
    if(data)CHECK(cudaMemcpy(d.cudaData,data,d.GetBytes(),cudaMemcpyHostToDevice)==cudaSuccess);
}
static void run(int rows,int n,int k,int groups,bool fp8,int blockRows=0) {
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
    allocate(input,BFLOAT16,{rows,1,groups,k},x.data());
    allocate(weight,fp8?FP8_E4M3:BFLOAT16,{groups*n,k},fp8?(void*)q.data():(void*)w.data());
    if(fp8){weight.blockM=weight.blockK=32;weight.scales=scales;Data bias;FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight,bias,groups*n);}
    allocate(out,BFLOAT16,{rows,groups,n},nullptr);
    auto dispatch=[&]() {
        if(blockRows)return TryCudaTritonSmallProjectionCandidate(input,weight,out,groups,true,blockRows);
        if(groups>1)return FastllmCudaDeepSeekV41CompactWoA(input,weight,groups,n,out);
        Data bias;return FastllmCudaBFloat16MatMulFP8E4M3(input,weight,bias,out,rows,k,n);
    };
    CHECK(dispatch());
    CHECK(cudaDeviceSynchronize()==cudaSuccess);
    {
        cudaGraph_t g;cudaGraphExec_t e;
        CHECK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
        CHECK(dispatch());
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

}
int main(){
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    cudaSetDevice(0);FastllmCudaSetDevice(0);
    if(FastllmCudaRuntimeArch()!=120&&FastllmCudaRuntimeArch()!=121)return 77;
    unsetenv("FASTLLM_CUDA_TRITON");
    CHECK(fastllm_kernel_configure("{\"mode\":\"quick\",\"budget_ms\":120000}"));
    for(auto shape:std::vector<std::vector<int>>{{32,32},{96,160},{512,1280}})
        for(int rows:{1,2,3,5,6,8,9,12,16,17,32})for(int groups:{1,2,8})run(rows,shape[0],shape[1],groups,true);
    // Wide FP8 tiles include a partial N tile and a masked K tail.
    for (int rows : {3, 6, 16}) run(rows, 16416, 96, 1, true);
    run(6, 32768, 1280, 1, true);
    for(int tile:{32,64}) {
        for(bool fp8:{true,false}) for(int rows:{2,17,24,33,48,65,97,129})
            for(int groups:{1,2,8}) run(rows,96,160,groups,fp8,tile);
        for(int rows:{6,12,24,48,97})run(rows,512,1280,8,true,tile);
        for(int rows:{24,48,65})run(rows,16416,96,1,true,tile);
    }
    fastllm_kernel_freeze();
    std::printf("PROJECTION_BATCH_PASS cases=%d max_l2=%.10g\n",cases,maxL2);
    return 0;
}
