#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);std::abort();} } while(0)
#define REQUIRE(x) do { if(!(x)){fprintf(stderr,"CHECK line %d: %s\n",__LINE__,#x);std::abort();} } while(0)
template<class T> T convert(float x) { return (T)x; }
template<class T> void checkShape(fastllm::DataType type,int rows,int vocab,int seed) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> dist(-4,4);
    size_t n=(size_t)rows*vocab;
    T *x; int *out,*map; float *fout; void *scratch=nullptr;
    size_t bytes=FastllmCudaGreedySamplingWorkspaceBytes(rows,vocab);
    CK(cudaMalloc(&x,n*sizeof(T))); CK(cudaMalloc(&out,rows*sizeof(int)));
    CK(cudaMalloc(&fout,rows*sizeof(float))); CK(cudaMalloc(&map,vocab*sizeof(int)));
    if(bytes) CK(cudaMalloc(&scratch,bytes));
    std::vector<T> host(n); std::vector<int> mapping(vocab),want(rows),got(rows);
    std::vector<float> fg(rows), wantScore(rows); for(int i=0;i<vocab;i++)mapping[i]=vocab-i;
    CK(cudaMemcpy(map,mapping.data(),vocab*sizeof(int),cudaMemcpyHostToDevice));
    auto launch=[&](bool mapped,bool floating){ return FastllmCudaGreedySamplingTyped(x,type,out,floating?fout:nullptr,mapped?map:nullptr,rows,vocab,scratch,bytes); };
    for(int test=0;test<7;test++) {
        for(size_t i=0;i<n;i++) {
            int col=i%vocab; float v=dist(rng);
            if(test==1)v=-INFINITY;
            if(test==2)v=NAN;
            if(test==3)v=col%129==0?1.f:0.f;
            if(test==4)v=col%997==0?NAN:(col%499==0?INFINITY:-1.f);
            if(test==5)v=col&1?0.f:-0.f;
            if(test==6)v=col==vocab-1?INFINITY:-INFINITY;
            host[i]=convert<T>(v);
        }
        for(int r=0;r<rows;r++){float best=-INFINITY;int id=0;for(int c=0;c<vocab;c++){float v=(float)host[(size_t)r*vocab+c];if(v>best){best=v;id=c;}}want[r]=id;wantScore[r]=best;}
        CK(cudaMemcpy(x,host.data(),n*sizeof(T),cudaMemcpyHostToDevice));
        for(bool mapped:{false,true})for(bool floating:{false,true}){
            REQUIRE(launch(mapped,floating)); CK(cudaMemcpy(got.data(),out,rows*sizeof(int),cudaMemcpyDeviceToHost));
            if(floating)CK(cudaMemcpy(fg.data(),fout,rows*sizeof(float),cudaMemcpyDeviceToHost));
            for(int r=0;r<rows;r++){int w=mapped?mapping[want[r]]:want[r];if(got[r]!=w || (floating && fg[r]!=(float)w)){fprintf(stderr,"dtype=%d rows=%d vocab=%d case=%d row=%d got=%d want=%d\n",(int)type,rows,vocab,test,r,got[r],w);std::abort();}}
        }
        REQUIRE(FastllmCudaGreedySamplingTypedWithScores(x,type,out,fout,rows,vocab,scratch,bytes));
        CK(cudaMemcpy(got.data(),out,rows*sizeof(int),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fg.data(),fout,rows*sizeof(float),cudaMemcpyDeviceToHost));
        for(int r=0;r<rows;r++){REQUIRE(got[r]==want[r]);REQUIRE(fg[r]==wantScore[r]);}
    }
    if(bytes){REQUIRE(!FastllmCudaGreedySamplingTyped(x,type,out,nullptr,nullptr,rows,vocab,nullptr,0));REQUIRE(!FastllmCudaGreedySamplingTyped(x,type,out,nullptr,nullptr,rows,vocab,scratch,bytes-1));}
    REQUIRE(!FastllmCudaGreedySamplingTyped(x,fastllm::DataType::INT8,out,nullptr,nullptr,rows,vocab,scratch,bytes));
    if(bytes){REQUIRE(!FastllmCudaGreedySamplingTypedWithScores(x,type,out,fout,rows,vocab,nullptr,0));REQUIRE(!FastllmCudaGreedySamplingTypedWithScores(x,type,out,fout,rows,vocab,scratch,bytes-1));}
    REQUIRE(!FastllmCudaGreedySamplingTypedWithScores(x,fastllm::DataType::INT8,out,fout,rows,vocab,scratch,bytes));
    // The SM75 TP2 eager regression can skip graph capture entirely.
    if(!std::getenv("SKIP_GREEDY_GRAPH")) {
    // Stable graph, changing inputs and map contents across replays.
    cudaGraph_t graph; cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)); REQUIRE(launch(true,true));
    CK(cudaStreamEndCapture(cudaStreamPerThread,&graph)); CK(cudaGraphInstantiate(&exec,graph,0));
    for(int repeat=0;repeat<3;repeat++) {
        std::fill(host.begin(),host.end(),convert<T>(-1));
        for(int r=0;r<rows;r++){want[r]=(seed+repeat*59+r*127)%vocab;host[(size_t)r*vocab+want[r]]=convert<T>(2);}
        for(int c=0;c<vocab;c++)mapping[c]=(c+repeat*31)%vocab;
        CK(cudaMemcpy(x,host.data(),n*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemcpy(map,mapping.data(),vocab*sizeof(int),cudaMemcpyHostToDevice));
        CK(cudaGraphLaunch(exec,cudaStreamPerThread));CK(cudaMemcpy(got.data(),out,rows*sizeof(int),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fg.data(),fout,rows*sizeof(float),cudaMemcpyDeviceToHost));
        for(int r=0;r<rows;r++){REQUIRE(got[r]==mapping[want[r]]);REQUIRE(fg[r]==(float)got[r]);}
    }
    CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
    }
    CK(cudaFree(x));CK(cudaFree(out));CK(cudaFree(fout));CK(cudaFree(map));if(scratch)CK(cudaFree(scratch));
}
int main(int argc,char **argv){int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || count==0)return 77;
    int device=argc>1?std::atoi(argv[1]):0;CK(cudaSetDevice(device));
    int cases=0;
    for(int rows:{1,2,4,8,9,17})for(int vocab:{1,31,257,4097,16383,16384,32771,65536,124160,131073,248320}){
        checkShape<float>(fastllm::DataType::FLOAT32,rows,vocab,123);checkShape<half>(fastllm::DataType::FLOAT16,rows,vocab,456);checkShape<__nv_bfloat16>(fastllm::DataType::BFLOAT16,rows,vocab,789);cases+=3;
    }
    std::thread a([&]{CK(cudaSetDevice(device));for(int i=0;i<5;i++)checkShape<half>(fastllm::DataType::FLOAT16,4,248320,100+i);});
    std::thread b([&]{CK(cudaSetDevice(device));for(int i=0;i<5;i++)checkShape<float>(fastllm::DataType::FLOAT32,1,131073,200+i);});a.join();b.join();
    printf("PASS device=%d shapes=%d types=FP32/FP16/BF16 special_values=7 mapped_and_float=4 scores=checked graph_replay=%d concurrent_threads=2\n",device,cases,std::getenv("SKIP_GREEDY_GRAPH")?0:3);return 0;
}
