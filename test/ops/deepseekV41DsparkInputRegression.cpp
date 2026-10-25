#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-dspark.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
template<typename T> static std::vector<T> Read(const Data &d) {
    std::vector<T> out(d.Count(0));
    Require(cudaMemcpy(out.data(), d.cudaData, out.size()*sizeof(T), cudaMemcpyDeviceToHost)==cudaSuccess,"read failed");
    return out;
}
static void Check(int dim, int hc, int window, int block) {
    constexpr int vocab=257, noise=vocab-1;
    Data embedding(BFLOAT16,{vocab,dim}), metadata(INT32,{2}), hidden, mix, indices, tokens;
    for (Data *d : {&embedding,&metadata}) { d->ToDevice(DataDevice::CUDA,{0},false); d->Allocate(false); }
    std::vector<uint16_t> weights(vocab*dim);
    auto upload=[&](int start,int anchor) {
        for(size_t i=0;i<weights.size();++i) weights[i]=uint16_t(i*31337+start*17+anchor);
        Require(cudaMemcpy(embedding.cudaData,weights.data(),weights.size()*2,cudaMemcpyHostToDevice)==cudaSuccess,"embedding upload");
        int meta[2]={start,anchor};
        Require(cudaMemcpy(metadata.cudaData,meta,sizeof(meta),cudaMemcpyHostToDevice)==cudaSuccess,"metadata upload");
    };
    auto run=[&]() {
        Require(FastllmCudaDeepSeekV41DraftInput(embedding,metadata,noise,block,hc,window,hidden,mix,indices,tokens),"valid input rejected");
    };
    auto verify=[&](int start,int anchor) {
        const int width=window+block-1;
        Require(hidden.dims==std::vector<int>({1,block,hc,dim}) && hidden.dataType==BFLOAT16,"hidden layout");
        Require(mix.dims==std::vector<int>({1,block,hc}) && mix.dataType==FLOAT32,"mix layout");
        Require(indices.dims==std::vector<int>({1,block,width}) && indices.dataType==INT32,"index layout");
        auto h=Read<uint16_t>(hidden);auto m=Read<float>(mix);auto ix=Read<int>(indices);
        for(int r=0;r<block;++r) {
            for(int c=0;c<hc;++c) {
                Require(m[r*hc+c]==(c==0?1.f:0.f),"premix mismatch");
                for(int d=0;d<dim;++d) Require(h[(r*hc+c)*dim+d]==weights[(r?noise:anchor)*dim+d],"embedding bits mismatch");
            }
            std::vector<int> expected;
            for(int p=start-window;p<start;++p) expected.push_back(p%window);
            for(int j=0;j<block;++j) if(j!=r) expected.push_back(window+j);
            for(int c=0;c<width;++c) Require(ix[r*width+c]==expected[c],"gather index mismatch");
        }
        Require(Read<int>(tokens)[0]==anchor,"anchor mismatch");
    };
    upload(window,0);run();verify(window,0);
    cudaGraph_t graph;cudaGraphExec_t exec;
    Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"capture begin");
    run();
    Require(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess,"capture end");
    Require(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)==cudaSuccess,"instantiate");
    for(int start : {window,window+1,2*window-1,2*window,1000003}) for(int anchor : {0,128,vocab-1}) {
        upload(start,anchor);
        for(bool replay : {false,true}) {
            for(Data *d : {&hidden,&mix,&indices,&tokens})
                Require(cudaMemsetAsync(d->cudaData,0xff,d->GetBytes(),cudaStreamPerThread)==cudaSuccess,"poison");
            if(replay) Require(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess,"replay"); else run();
            verify(start,anchor);
        }
    }
    cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
    auto before=Read<uint16_t>(hidden);void *ptr=hidden.cudaData;
    auto reject=[&]() {
        Require(!FastllmCudaDeepSeekV41DraftInput(embedding,metadata,noise,block,hc,window,hidden,mix,indices,tokens),"invalid input accepted");
        Require(hidden.cudaData==ptr && Read<uint16_t>(hidden)==before,"rejection mutated output");
    };
    for(DataType type : {FLOAT16,FLOAT32}) { embedding.dataType=type;reject(); }
    embedding.dataType=BFLOAT16;embedding.strides[0]++;reject();embedding.strides[0]--;
    embedding.dataDeviceIds={0,1,2,3,4,5,6,7};run();verify(1000003,vocab-1);
    embedding.dataDeviceIds={1};reject();embedding.dataDeviceIds={0};
    metadata.strides[0]=2;reject();metadata.strides[0]=1;
    metadata.dims={1};reject();metadata.dims={2};
    printf("dim=%d hc=%d window=%d block=%d eager+graph exact input/index/premix/anchor PASS\n",dim,hc,window,block);
}
int main() {
    try {
        FastllmCudaSetDevice(0);
        for(int block=1;block<=8;++block) {
            Check(7,1,1,block);Check(33,3,17,block);Check(5120,4,128,block);
        }
        puts("DSpark input 24 cases PASS");
    } catch(const std::exception &e) { fprintf(stderr,"%s\n",e.what());return 1; }
}
