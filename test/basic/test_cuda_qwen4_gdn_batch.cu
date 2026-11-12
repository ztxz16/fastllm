#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
static void Cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void Allocate(Data &d){d.ToDevice(CUDA,{0},false);d.Allocate(false);}
static void Fill(Data &d,int seed){std::mt19937 rng(seed);std::uniform_real_distribution<float> f(-.2f,.2f);
    if(d.dataType==FLOAT32){std::vector<float>x(d.Count(0));for(auto &v:x)v=f(rng);Cuda(cudaMemcpy(d.cudaData,x.data(),d.GetBytes(),cudaMemcpyHostToDevice));}
    else {std::vector<half>x(d.Count(0));for(auto &v:x)v=__float2half(f(rng));Cuda(cudaMemcpy(d.cudaData,x.data(),d.GetBytes(),cudaMemcpyHostToDevice));}}
static std::vector<char> Bytes(Data &d){std::vector<char>x(d.GetBytes());Cuda(cudaMemcpy(x.data(),d.cudaData,x.size(),cudaMemcpyDeviceToHost));return x;}
static void Case(int count,int heads){
    int keys=heads/3,channels=(2*keys+heads)*128;
    Data log(FLOAT32,{heads}),bias(FLOAT32,{heads}),norm(FLOAT32,{128});
    for(Data*d:{&log,&bias,&norm}){Allocate(*d);Fill(*d,7);}
    struct Request{
        Data q,a,b,g,state,out,referenceState,referenceOut;
        Request(int heads,int channels):q(FLOAT32,{1,1,channels}),a(FLOAT16,{1,1,heads}),b(FLOAT16,a.dims),g(FLOAT16,{1,1,heads*128}),state(FLOAT32,{1,heads,128,128}),out(FLOAT16,g.dims),referenceState(FLOAT32,state.dims),referenceOut(FLOAT16,g.dims){for(Data*d:{&q,&a,&b,&g,&state,&out,&referenceState,&referenceOut})Allocate(*d);}
    };
    std::vector<std::unique_ptr<Request>> owned;std::vector<FastllmCudaQwen4GdnDecodeRequest> requests;
    for(int i=0;i<count;++i){owned.emplace_back(new Request(heads,channels));auto&r=*owned.back();Fill(r.state,100+i);Cuda(cudaMemcpy(r.referenceState.cudaData,r.state.cudaData,r.state.GetBytes(),cudaMemcpyDeviceToDevice));requests.push_back({&r.q,&r.a,&r.b,&r.g,&r.state,&r.out});}
    auto batch=[&]{Check(FastllmCudaQwen4GatedDeltaRuleDecodeOutputBatch(requests,log,bias,norm,keys,heads,1e-6f,1e-6f),"batch rejected");};
    // A bad final request must reject before updating any earlier state.
    auto snapshot=Bytes(owned[0]->state);auto invalid=requests;invalid.back().output=invalid.back().state;
    Check(!FastllmCudaQwen4GatedDeltaRuleDecodeOutputBatch(invalid,log,bias,norm,keys,heads,1e-6f,1e-6f),"bad output accepted");
    Check(snapshot==Bytes(owned[0]->state),"partial update on rejection");
    if(count>1){invalid=requests;invalid.back().state=invalid.front().state;Check(!FastllmCudaQwen4GatedDeltaRuleDecodeOutputBatch(invalid,log,bias,norm,keys,heads,1e-6f,1e-6f),"shared state accepted");}
    cudaGraph_t graph;cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));batch();Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    size_t nodes=0;Cuda(cudaGraphGetNodes(graph,nullptr,&nodes));Check(nodes==size_t((count+31)/32),"unexpected launch count");Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int step=0;step<3;++step){
        for(int i=0;i<count;++i){auto&r=*owned[i];Fill(r.q,step*73+i);Fill(r.a,step*79+i);Fill(r.b,step*83+i);Fill(r.g,step*89+i);
            Check(FastllmCudaQwen4GatedDeltaRuleDecodeOutput(r.q,r.a,r.b,log,bias,norm,r.g,r.referenceState,r.referenceOut,keys,heads,1e-6f,1e-6f,false),"reference rejected");}
        Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));Cuda(cudaDeviceSynchronize());
        for(auto &p:owned){Check(Bytes(p->state)==Bytes(p->referenceState),"state not bitwise");Check(Bytes(p->out)==Bytes(p->referenceOut),"output not bitwise");}
    }
    Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
    printf("PASS requests=%d value_heads=%d nodes=%zu state/output bitwise after three changed-input replays\n",count,heads,nodes);fflush(stdout);
}
int main(int,char**){try{FastllmCudaSetDevice(0);for(int n:{1,2,3,4,7,33})Case(n,n<=4?48:6);return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
