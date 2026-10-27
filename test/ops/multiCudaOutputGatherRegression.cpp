#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <functional>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok,const char *text){if(!ok)throw std::runtime_error(text);}
static void Cuda(cudaError_t e){Check(e==cudaSuccess,cudaGetErrorString(e));}
struct Graph {
    std::vector<int> devices;
    std::vector<void*> graphs,execs,starts,ends,pinned;
    Graph(const std::vector<int> &d,const std::function<void()> &body):devices(d) {
        graphs.resize(d.size());execs.resize(d.size());starts.resize(d.size());ends.resize(d.size());
        for(size_t i=0;i<d.size();++i){FastllmCudaSetDevice(d[i]);Cuda(cudaDeviceSynchronize());
            Check(FastllmCudaGraphPrepareCaptureDevice(),"prepare capture");
            starts[i]=FastllmCudaEventCreate();ends[i]=FastllmCudaEventCreate();}
        Check(FastllmCudaGraphMemoryPoolBegin(),"pool begin");
        for(size_t i=0;i<d.size();++i){FastllmCudaSetDevice(d[i]);Check(FastllmCudaGraphBeginCapture(),"capture begin");
            FastllmCudaEventRecordCurrentThread(starts[i]);}
        Check(MultiCudaGraphWorkersWaitEvents(d,starts),"worker join");
        FastllmCudaSetDevice(d[0]);body();
        Check(MultiCudaGraphWorkersRecordEvents(d,ends),"worker end");
        for(size_t i=0;i<d.size();++i){FastllmCudaSetDevice(d[i]);FastllmCudaCurrentThreadStreamWaitEvent(ends[i]);
            Check(FastllmCudaGraphEndCapture(&graphs[i]),"capture end");}
        Check(FastllmCudaGraphMemoryPoolEnd(pinned),"pool end");
        for(size_t i=0;i<d.size();++i){FastllmCudaSetDevice(d[i]);Check(FastllmCudaGraphInstantiate(graphs[i],&execs[i]),"instantiate");}
        FastllmCudaSetDevice(d[0]);
    }
    void Run() {
        for (size_t i = 0; i < devices.size(); ++i) {
            FastllmCudaSetDevice(devices[i]);
            Check(FastllmCudaGraphLaunch(execs[i]), "launch");
        }
        FastllmCudaSetDevice(devices[0]);
    }
    ~Graph(){for(size_t i=0;i<devices.size();++i){FastllmCudaSetDevice(devices[i]);cudaDeviceSynchronize();
        FastllmCudaGraphExecDestroy(execs[i]);FastllmCudaGraphDestroy(graphs[i]);FastllmCudaEventDestroy(starts[i]);FastllmCudaEventDestroy(ends[i]);}
        FastllmCudaGraphMemoryPoolRelease(pinned);FastllmCudaSetDevice(devices[0]);}
};
int main(){
    int available=0;if(cudaGetDeviceCount(&available)!=cudaSuccess||available<8)return 77;
    try{
        SetThreads(8);SetCudaGraph(true);FastllmCudaSetNcclForceSync(false);
        const std::string spec="multicuda:0,1,2,3,4,5,6,7";
        SetDeviceMap({{spec,1}});ApplyDeviceMap({{spec,1}},0,1);
        std::vector<int> devices(8);std::iota(devices.begin(),devices.end(),0);FastllmMultiCudaSetDevice(devices);
        int checks=0;
        for(auto type:{FLOAT32,BFLOAT16})for(int vocab:{1024,1051})for(int rows:{1,2,5,8}){
            constexpr int hidden=256;
            std::vector<float> w(vocab*hidden),x(rows*hidden),oracle(rows*vocab);
            for(size_t i=0;i<w.size();++i)w[i]=(int((i*13+7)%31)-15)/16.f;
            Data weight(FLOAT32,{vocab,hidden},w),input(type,{1,rows,hidden},x),output;
            weight.isModelWeight=true;input.ToDevice(DataDevice::CUDA,std::vector<int>{0});
            std::shared_ptr<void> workspace;
            auto run = [&]() {
                ApplyDeviceMap({{spec, 1}}, 0, 1);
                Check(MultiCudaLinearOutputGatherPersistent(input, weight, output, workspace),
                      "persistent projection rejected");
            };
            run();Graph graph(devices,run);
            for(int iteration=0;iteration<12;++iteration){
                for(size_t i=0;i<x.size();++i)x[i]=(int((i*17+iteration*19)%29)-14)/16.f;
                for(int row=0;row<rows;++row)for(int v=0;v<vocab;++v){double sum=0;
                    for(int h=0;h<hidden;++h)sum+=double(x[row*hidden+h])*w[v*hidden+h];
                    oracle[row*vocab+v]=sum;}
                Data changed(type,{1,rows,hidden},x);changed.ToDevice(DataDevice::CUDA,std::vector<int>{0});input.CopyFrom(changed);
                if(iteration%3)graph.Run();else run();
                Cuda(cudaDeviceSynchronize());std::vector<float> actual(oracle.size());
                Cuda(cudaMemcpy(actual.data(),output.cudaData,output.GetBytes(),cudaMemcpyDeviceToHost));
                Check(actual==oracle,"gather logits differ from exact FP64 oracle");++checks;
            }
            Data changedShape(FLOAT32,{1,rows+1,hidden});changedShape.ToDevice(DataDevice::CUDA,{0},false);changedShape.Allocate();
            Check(!MultiCudaLinearOutputGatherPersistent(changedShape,weight,output,workspace),"shape fallback accepted");
        }
        {
            // Arbitrary FP32 checkpoints must keep the original FP32 path.
            Data weight(FLOAT32,{1024,256},std::vector<float>(1024*256,0.1f));weight.isModelWeight=true;
            Data input(BFLOAT16,{1,5,256},std::vector<float>(5*256,0.5f)),output;
            input.ToDevice(DataDevice::CUDA,std::vector<int>{0});std::shared_ptr<void> workspace;
            Check(!MultiCudaLinearOutputGatherPersistent(input,weight,output,workspace),"inexact BF16 checkpoint accepted");
            Check(bool(workspace),"unsupported compaction was not remembered");
            Check(!MultiCudaLinearOutputGatherPersistent(input,weight,output,workspace),"repeated inexact BF16 checkpoint accepted");
            Data exact(FLOAT32,{1,5,256},std::vector<float>(5*256,0.5f));exact.ToDevice(DataDevice::CUDA,std::vector<int>{0});
            Check(MultiCudaLinearOutputGatherPersistent(exact,weight,output,workspace),"FP32 fallback rejected");
            std::vector<float> actual(5*1024);Cuda(cudaMemcpy(actual.data(),output.cudaData,actual.size()*4,cudaMemcpyDeviceToHost));
            for(float x:actual)Check(std::abs(x-12.8f)<2e-5f,"FP32 fallback lost checkpoint precision");
            ++checks;
        }
        std::cout<<"PASS "<<checks<<" persistent gathered projections: uneven shards, changed graph inputs and FP64 oracle\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
