#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include "fastllm-cuda-sampling.h"
#include "utils/kernel_registry.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
using Candidate = FastllmCudaTopKCandidate;
constexpr int K = FastllmCudaLogitsTopKCapacity;
static int cases=0, sampled=0;
static void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void Cuda(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    void *p=nullptr;
    explicit Buffer(size_t n) { Cuda(cudaMalloc(&p,n)); }
    ~Buffer() { cudaFree(p); }
};
static void Policy(int tile) {
    std::string config = tile ? "{\"mode\":\"quick\",\"budget_ms\":30000,\"samples\":3,\"repeats\":2,\"force\":{\"sampling.logits_topk\":\"cuda.sampling.topk.tile"+std::to_string(tile)+"\"}}"
        : "{\"mode\":\"off\"}";
    Require(fastllm_kernel_configure(config.c_str())==1,fastllm_kernel_error());
}
static std::vector<Candidate> Reference(const float *input, int vocab, float inverse) {
    std::vector<Candidate> ranked(vocab);
    for(int i=0;i<vocab;++i) ranked[i]={input[i]*inverse,i};
    std::partial_sort(ranked.begin(),ranked.begin()+std::min(vocab,K),ranked.end(),[](Candidate a,Candidate b){
        return a.score>b.score || (a.score==b.score && a.token<b.token);
    });
    ranked.resize(std::min(vocab,K)); return ranked;
}
static void Case(int vocab,int batch) {
    Buffer input(size_t(batch)*vocab*sizeof(float)), output(size_t(batch)*K*sizeof(Candidate));
    const size_t bytes=FastllmCudaLogitsTopKWorkspaceBytes(batch,vocab);
    Buffer workspace(bytes);
    std::vector<FastllmCudaTopKRow> rows(batch);
    std::vector<float> values(size_t(batch)*vocab);
    for(int i=0;i<batch;++i) rows[i]={static_cast<float *>(input.p)+size_t(i)*vocab,1.f/(.5f+(i%4)*.25f)};
    auto run=[&] { Require(FastllmCudaLogitsTopK(rows.data(),batch,vocab,
        static_cast<Candidate *>(output.p),workspace.p,bytes),"top-k rejected"); };
    auto fill=[&](int pass) {
        std::mt19937 rng(731+pass*37+vocab);
        for(size_t i=0;i<values.size();++i) {
            const int mode=(i/vocab+pass)%4;
            values[i]=mode==0 ? 0.f : mode==1 ? float(int(rng()%101)-50)*.125f : float(int(rng()%50001)-25000)/1024.f;
            if(mode==2 && i%13==0) values[i]=-INFINITY;
        }
        Cuda(cudaMemcpy(input.p,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice));
    };
    auto check=[&] {
        std::vector<Candidate> result(size_t(batch)*K);
        Cuda(cudaMemcpy(result.data(),output.p,result.size()*sizeof(Candidate),cudaMemcpyDeviceToHost));
        for(int b=0;b<batch;++b) {
            auto ref=Reference(values.data()+size_t(b)*vocab,vocab,rows[b].inverseTemperature);
            if (!std::isfinite(ref[0].score)) {
                Require(result[size_t(b)*K].token<0,"all-infinite row accepted on replay");
                continue;
            }
            for(int i=0;i<(int)ref.size();++i) {
                const auto got=result[size_t(b)*K+i];
                if(got.token!=ref[i].token || got.score!=ref[i].score) {
                    std::cerr<<"vocab="<<vocab<<" batch="<<batch<<" row="<<b<<" rank="<<i
                        <<" token="<<got.token<<" want="<<ref[i].token<<" score="<<got.score<<" want="<<ref[i].score<<'\n';
                    throw std::runtime_error("selection differs from CPU partial sort");
                }
            }
            for(int i=vocab;i<K;++i) Require(result[size_t(b)*K+i].token==-1,"missing tail mask");
        }
    };
    fill(0);run();check();
    cudaGraph_t graph;cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));run();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int pass=1;pass<=2;++pass) {fill(pass);Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));check();}
    Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
    // Invalid inputs are reported for that row; no host RNG was consumed.
    for(float bad : {INFINITY,std::numeric_limits<float>::quiet_NaN()}) {
        values[0]=bad;Cuda(cudaMemcpy(input.p,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice));run();
        Candidate got;Cuda(cudaMemcpy(&got,output.p,sizeof(got),cudaMemcpyDeviceToHost));Require(got.token<0,"invalid logit accepted");
    }
    std::fill(values.begin(),values.begin()+vocab,-INFINITY);
    Cuda(cudaMemcpy(input.p,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice));run();
    Candidate got;Cuda(cudaMemcpy(&got,output.p,sizeof(got),cudaMemcpyDeviceToHost));Require(got.token<0,"all-infinite row accepted");
    Require(!FastllmCudaLogitsTopK(rows.data(),batch,vocab,static_cast<Candidate *>(output.p),workspace.p,bytes-1),"undersized scratch accepted");
    ++cases;
}
static void Sampling() {
    Policy(0);
    constexpr int vocab=4097;
    std::vector<float> values(2*vocab);
    for(size_t i=0;i<values.size();++i) values[i]=float(int((i*317)%5003)-2501)/128.f;
    LastTokensUnit history(8);history.Push(3);history.Push(3);history.Push(17);
    for(int k : {0,1,7,50,64,65}) for(float temperature : {.5f,.8f,1.7f}) for(float topP : {.1f,.95f,1.f}) {
        GenerationConfig cfg;cfg.top_k=k;cfg.top_p=topP;cfg.temperature=temperature;
        for(int mode=0;mode<3;++mode) {
            cfg.repeat_penalty=mode==1 ? 1.2f : 1.f;
            cfg.tool_call_allowed_token_ids=mode==2 ? std::vector<int>{3,17,512,1025,3073} : std::vector<int>{};
            for(int seed=0;seed<8;++seed) {
                Data host(FLOAT32,{2,vocab},values), device;device.CopyFrom(host);device.ToDevice(CUDA,std::vector<int>{0});
                std::srand(157+seed*131);const int expected=LLMSampling(host,1,cfg,history);const int next=std::rand();
                std::srand(157+seed*131);const int got=LLMSampling(device,1,cfg,history);
                Require(got==expected,"GPU candidates changed sampled token");
                Require(std::rand()==next,"GPU candidates changed RNG advancement");
                Require((device.dataDevice==CUDA)==(mode==0 && k<=64),"unexpected CPU fallback or mutation");
                ++sampled;
            }
        }
    }
}
static double Now() {return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static void Bench(const char *path) {
    constexpr int vocab=248320, loops=200;
    std::vector<float> values(vocab);std::ifstream f(path,std::ios::binary);f.read((char *)values.data(),vocab*sizeof(float));
    Require(f.gcount()==vocab*sizeof(float),"benchmark logits size");
    Data host(FLOAT32,{1,vocab},values),device;device.CopyFrom(host);device.ToDevice(CUDA,std::vector<int>{0});
    GenerationConfig cfg;cfg.top_k=50;cfg.top_p=.95f;cfg.temperature=.8f;LastTokensUnit history;
    volatile int sink=0;
    double start=Now();for(int i=0;i<loops;++i)sink=LLMSampling(host,0,cfg,history);
    printf("{\"kind\":\"cpu_sampling\",\"ms\":%.9f}\n",(Now()-start)/loops);
    const size_t bytes=FastllmCudaLogitsTopKWorkspaceBytes(1,vocab);
    Buffer scratch(bytes),out(K*sizeof(Candidate));
    FastllmCudaTopKRow row{static_cast<const float *>(device.cudaData),1.f/.8f};
    auto run=[&] {Require(FastllmCudaLogitsTopK(&row,1,vocab,static_cast<Candidate *>(out.p),scratch.p,bytes),"micro launch");};
    for(int tile : {1024,2048,4096}) {
        Policy(tile);run();Cuda(cudaDeviceSynchronize());
        cudaGraph_t graph;cudaGraphExec_t exec;cudaEvent_t a,b;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));run();
        Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        Cuda(cudaEventCreate(&a));Cuda(cudaEventCreate(&b));
        Cuda(cudaEventRecord(a,cudaStreamPerThread));for(int i=0;i<loops;++i)Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
        Cuda(cudaEventRecord(b,cudaStreamPerThread));Cuda(cudaEventSynchronize(b));float ms;Cuda(cudaEventElapsedTime(&ms,a,b));
        for(int i=0;i<10;++i)sink=LLMSampling(device,0,cfg,history);
        start=Now();for(int i=0;i<loops;++i)sink=LLMSampling(device,0,cfg,history);
        printf("{\"kind\":\"gpu_candidates\",\"tile\":%d,\"workspace\":%zu,\"gpu_ms\":%.9f,\"sampling_host_ms\":%.9f}\n",tile,bytes,ms/loops,(Now()-start)/loops);
        Cuda(cudaEventDestroy(a));Cuda(cudaEventDestroy(b));Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
    }
    (void)sink;
}
int main(int argc,char **argv) {
    try {
        FastllmCudaSetDevice(0);
        if(argc==3 && std::string(argv[1])=="--bench") {Bench(argv[2]);return 0;}
        for(int tile : {1024,2048,4096}) {
            Policy(tile);
            for(int vocab : {1,17,63,64,65,1023,1024,1025,4097,248320}) Case(vocab,1);
            for(int batch : {3,17,32,33,65}) Case(4097,batch);
            Case(248320,4);
        }
        Sampling();
        printf("PASS top-k %d CPU-oracle graph cases, %d exact CPU/GPU sampled tokens and RNG checks\n",cases,sampled);
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
