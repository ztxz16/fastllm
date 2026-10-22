#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda-dspark.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void Cuda(cudaError_t status) { Check(status == cudaSuccess, cudaGetErrorString(status)); }
static std::vector<char> Read(Data &x) {
    std::vector<char> bytes(x.GetBytes());
    Cuda(cudaDeviceSynchronize());
    Cuda(cudaMemcpy(bytes.data(), x.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
    return bytes;
}
static void WindowChecks() {
    for (int rows : {1,2,5,8,17}) {
        std::vector<float> initial(16 * 32, -1.0f), values(rows * 32);
        Data actual(BFLOAT16,{1,16,32},initial), expected(BFLOAT16,{1,16,32},initial);
        Data source(BFLOAT16,{1,rows,32},values);
        for (Data *d : {&actual,&expected,&source}) d->ToDevice(DataDevice::CUDA,std::vector<int>{0});
        int *position; Cuda(cudaMalloc(&position,sizeof(int))); int zero = 0;
        Cuda(cudaMemcpy(position,&zero,sizeof(int),cudaMemcpyHostToDevice));
        void *graph=nullptr,*exec=nullptr;
        Check(FastllmCudaGraphBeginCapture(),"window capture begin");
        Check(FastllmCudaDeepSeekV41WindowStorePosition(source,actual,0,16,position),"window capture");
        Check(FastllmCudaGraphEndCapture(&graph),"window capture end");
        Check(FastllmCudaGraphInstantiate(graph,&exec),"window instantiate");
        for (int start : {0,1,15,16,17,31,32,129}) {
            for (size_t j=0;j<values.size();++j) values[j]=float(int(j%31)-15+start)*0.125f;
            Data changed(BFLOAT16,{1,rows,32},values); changed.ToDevice(DataDevice::CUDA,std::vector<int>{0});
            source.CopyFrom(changed);
            Cuda(cudaMemcpy(position,&start,sizeof(int),cudaMemcpyHostToDevice));
            Check(FastllmCudaDeepSeekV41WindowStore(source,expected,start,16),"window reference");
            Check(FastllmCudaGraphLaunch(exec),"window replay");
            auto aa=Read(actual),bb=Read(expected);
            if(aa!=bb){for(size_t j=0;j<aa.size();++j)if(aa[j]!=bb[j]){std::cerr<<"window rows="<<rows<<" start="<<start<<" byte="<<j<<" actual="<<(int)(unsigned char)aa[j]<<" expected="<<(int)(unsigned char)bb[j]<<std::endl;break;}}
            Check(aa==bb,"window wrap differs");
        }
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph); Cuda(cudaFree(position));
    }
    std::cout << "PASS 40 dynamic window stores, including overwritten windows\n";
}
static void MarkovChecks() {
    constexpr int vocab=37;
    int checks=0;
    for (auto type : {FLOAT32,FLOAT16,BFLOAT16}) for (int block : {1,5,8}) for (int rank : {16,128,512}) {
        std::vector<float> e(vocab*rank),h(vocab*rank),v(block*vocab);
        for (int i=0;i<vocab*rank;++i) { e[i]=(int((i*11+7)%31)-15)*0.125f; h[i]=(int((i*13+3)%29)-14)*0.125f; }
        Data embed(type,{vocab,rank},e),head(type,{vocab,rank},h),logits(FLOAT32,{1,block,vocab},v);
        Data tokens(INT32,{block+1}),pv,pi,out;
        tokens.Allocate(); std::memset(tokens.cpuData,0,tokens.GetBytes());
        for (Data *d : {&embed,&head,&logits,&tokens}) d->ToDevice(DataDevice::CUDA,std::vector<int>{0});
        auto run=[&](){return FastllmCudaDeepSeekV41MarkovChainDevice(logits,embed,head,block,tokens,pv,pi,out);};
        Check(run(),"Markov warmup"); Cuda(cudaDeviceSynchronize());
        void *graph=nullptr,*exec=nullptr;
        Check(FastllmCudaGraphBeginCapture(),"Markov capture begin"); Check(run(),"Markov capture");
        Check(FastllmCudaGraphEndCapture(&graph),"Markov capture end");
        Check(FastllmCudaGraphInstantiate(graph,&exec),"Markov instantiate");
        for (int iteration=0;iteration<10;++iteration) {
            int anchor=(iteration*7)%vocab,previous=anchor; std::vector<int> oracle;
            for (size_t i=0;i<v.size();++i) v[i]=(int((i*17+iteration*19)%71)-35)*0.0625f;
            Data changed(FLOAT32,{1,block,vocab},v); changed.ToDevice(DataDevice::CUDA,std::vector<int>{0}); logits.CopyFrom(changed);
            for (int step=0;step<block;++step) {
                double best=-1e300; int chosen=-1;
                for (int token=0;token<vocab;++token) {
                    double score=v[step*vocab+token];
                    for (int j=0;j<rank;++j) score+=double(e[previous*rank+j])*h[token*rank+j];
                    if(score>best){best=score;chosen=token;}
                }
                oracle.push_back(chosen); previous=chosen;
            }
            Cuda(cudaMemcpy(tokens.cudaData,&anchor,sizeof(int),cudaMemcpyHostToDevice));
            Check(FastllmCudaGraphLaunch(exec),"Markov replay");
            auto bytes=Read(tokens); std::vector<int> got(block); std::memcpy(got.data(),bytes.data()+4,block*4);
            Check(got==oracle,"Markov differs from FP64 chain oracle");
            std::vector<int> legacy; Data legacyEmbeds;
            Check(FastllmCudaDeepSeekV41MarkovChain(logits,embed,head,anchor,block,legacy,legacyEmbeds),"Markov host reference");
            Check(got==legacy && Read(out)==Read(legacyEmbeds),"Markov wrapper differs"); ++checks;
        }
        FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
    }
    std::cout << "PASS " << checks << " Markov chains against FP64 oracle and changed graph inputs\n";
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        // Dynamic-position kernels deliberately reject these reference modes.
        unsetenv("FASTLLM_DSV41_REFERENCE_MATH"); unsetenv("FASTLLM_DSV41_LEGACY_ROTARY");
        FastllmCudaSetDevice(0);
        int *position; Cuda(cudaMalloc(&position, sizeof(int)));
        int checks = 0;
        std::mt19937 rng(19); std::uniform_real_distribution<float> uniform(-2.f, 2.f);
        for (auto type : {FLOAT32, FLOAT16, BFLOAT16}) for (int rows : {1,2,5,8})
        for (int heads : {1,4}) for (int mode : {0,1,3}) for (bool inverse : {false,true}) {
            std::vector<float> values(rows * heads * 512);
            for (float &v : values) v = uniform(rng);
            std::vector<int> dims = {1,rows,heads,512};
            Data source(type,dims,values), expected, actual;
            source.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            expected.CopyFrom(source); actual.CopyFrom(source);
            auto run = [&]() {
                return FastllmCudaDeepSeekV41RotaryQuantPosition(actual,128,10000.f,0,1,inverse,
                    4096,4.f,32,1,mode,512,mode == 3 ? 16 : 32,position);
            };
            int zero = 0; Cuda(cudaMemcpy(position,&zero,sizeof(int),cudaMemcpyHostToDevice));
            Check(run(),"dynamic rotary warmup"); Cuda(cudaDeviceSynchronize());
            void *graph = nullptr; Check(FastllmCudaGraphBeginCapture(),"capture begin");
            Check(run(),"capture rotary"); Check(FastllmCudaGraphEndCapture(&graph),"capture end");
            void *exec = nullptr; Check(FastllmCudaGraphInstantiate(graph,&exec),"instantiate");
            for (int start : {0,1,127,128,129,511,512,2048,8192,127}) {
                // Change both the position and tensor contents on every replay.
                for (float &v : values) v = uniform(rng);
                Data changed(type,dims,values); changed.ToDevice(DataDevice::CUDA,std::vector<int>{0});
                expected.CopyFrom(changed); actual.CopyFrom(changed);
                Cuda(cudaMemcpy(position,&start,sizeof(int),cudaMemcpyHostToDevice));
                Check(FastllmCudaDeepSeekV41RotaryQuant(expected,128,10000.f,start,1,inverse,
                    4096,4.f,32,1,mode,512,mode == 3 ? 16 : 32),"host rotary reference");
                Check(FastllmCudaGraphLaunch(exec),"replay");
                Check(Read(expected) == Read(actual),"dynamic position differs from host position");
                ++checks;
            }
            FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
        }
        Cuda(cudaFree(position));
        WindowChecks(); MarkovChecks();
        std::cout << "PASS " << checks << " dynamic draft positions: exact eager reference and changed graph inputs\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
