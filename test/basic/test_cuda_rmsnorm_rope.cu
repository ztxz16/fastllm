#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
using namespace fastllm;
static void Check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void Allocate(Data &d) { d.dataDevice=DataDevice::CUDA; d.dataDeviceIds={0}; d.Allocate(false); }
static double maxRelativeL2 = 0, maxAbsolute = 0;
static size_t totalMismatches = 0;
static void Compare(int batch, int sequence, int heads, int dim, int rotate, bool precise, float theta=1000000.f, float ropeScale=1.f) {
    Data input(FLOAT16,{batch,sequence,heads,dim}), reference(FLOAT16,input.dims), original(FLOAT16,input.dims);
    Data weight(FLOAT32,{dim}), positions(FLOAT32,{batch,sequence+2});
    for (Data *p : {&input,&reference,&original,&weight,&positions}) Allocate(*p);
    std::mt19937 rng(943+dim+sequence); std::vector<half>x(input.Count(0)); std::vector<float>w(dim),p(positions.Count(0));
    for (auto &v:x) v=__float2half((int(rng()%4001)-2000)/997.f);
    for (auto &v:w) v=(int(rng()%2001)-500)/1000.f;
    for (auto &v:p) v=rng()%2097152;
    Check(cudaMemcpy(original.cudaData,x.data(),x.size()*sizeof(half),cudaMemcpyHostToDevice));
    Check(cudaMemcpy(weight.cudaData,w.data(),w.size()*4,cudaMemcpyHostToDevice));
    Check(cudaMemcpy(positions.cudaData,p.data(),p.size()*4,cudaMemcpyHostToDevice));
    auto restore=[&]{Check(cudaMemcpyAsync(input.cudaData,original.cudaData,input.GetBytes(),cudaMemcpyDeviceToDevice,cudaStreamPerThread));};
    restore();
    if (!FastllmCudaRMSNormRope(input,weight,positions,1e-6f,rotate,theta,ropeScale,precise)) throw std::runtime_error("fused declined");
    Check(cudaDeviceSynchronize());
    cudaGraph_t graph; cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    restore(); FastllmCudaRMSNormRope(input,weight,positions,1e-6f,rotate,theta,ropeScale,precise);
    Check(cudaStreamEndCapture(cudaStreamPerThread,&graph)); Check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (int replay=0;replay<3;++replay) {
        for (auto &v:p) v=rng()%2097152;
        Check(cudaMemcpy(positions.cudaData,p.data(),p.size()*4,cudaMemcpyHostToDevice));
        FastllmCudaRMSNorm(original,weight,reference,1e-6f);
        FastllmCudaRopeEncoding(reference,positions,rotate,theta,ropeScale,precise);
        Check(cudaGraphLaunch(exec,cudaStreamPerThread)); Check(cudaDeviceSynchronize());
        std::vector<half>a(x.size()),b(x.size());
        Check(cudaMemcpy(a.data(),input.cudaData,input.GetBytes(),cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(b.data(),reference.cudaData,reference.GetBytes(),cudaMemcpyDeviceToHost));
        double error=0, energy=0, maximum=0;
        for (size_t i=0;i<a.size();++i) {
            const double difference=double(float(a[i]))-float(b[i]);
            error+=difference*difference; energy+=double(float(b[i]))*float(b[i]);
            maximum=std::max(maximum,std::abs(difference));
            totalMismatches+=memcmp(&a[i],&b[i],2)!=0;
        }
        const double l2=std::sqrt(error/std::max(energy,1e-30));
        maxRelativeL2=std::max(maxRelativeL2,l2);maxAbsolute=std::max(maxAbsolute,maximum);
        // Fusion can change the compiler's FP32 FMA contraction at RoPE's
        // final half rounding. Bound that error well below FP16 precision.
        if (!std::isfinite(l2) || l2>0.00005 || maximum>0.00390625) {
            std::fprintf(stderr,"Mismatch dim%d rotate%d b%d s%d h%d precise%d l2%g max%g\n",dim,rotate,batch,sequence,heads,precise,l2,maximum);
            throw std::runtime_error("norm+rope accuracy regression");
        }
    }
    Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
}
extern "C" int main(int argc,char **argv) {try {
    int count=0;
    for (int dim : {64,128,192,256,384,510}) for (int rows : {1,2,4,7,17}) for(bool precise : {false,true}) {
        Compare(rows,1,3,dim,64,precise); Compare(3,rows,2,dim,dim,precise);count+=2;
    }
    for (float theta : {10000.f, 500000.f, 1000000.f}) for (float scale : {1.f, 2.f, 8.f}) {
        Compare(2,3,4,128,64,true,theta,scale); ++count;
    }
    Data bad(FLOAT32,{1,1,1,128}),w(FLOAT32,{128}),p(FLOAT32,{1,1});
    if(FastllmCudaRMSNormRope(bad,w,p,1e-6f,64,10000.f,1.f,true)) throw std::runtime_error("unsupported dtype accepted");
    std::printf("ALL_PASS %d norm+rope cases, dynamic positions and graph replay; max_l2=%.9g max_abs=%.9g differing_elements=%zu\n",count,maxRelativeL2,maxAbsolute,totalMismatches);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
